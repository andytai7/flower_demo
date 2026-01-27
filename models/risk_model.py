"""
CKD Risk Prediction Model

A simple feedforward network for binary risk classification.
Designed to be small enough for quick experimentation while
complex enough to demonstrate federated learning concepts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class CKDRiskModel(nn.Module):
    """
    Simple MLP for CKD risk prediction.
    
    Architecture chosen to be:
    - Small enough for CPU training during experimentation
    - Large enough to show meaningful FL dynamics
    - Similar complexity to what you might use for tabular EHR data
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: list = [64, 32],
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of high risk."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ewc_penalty: Optional[Dict] = None,
    ewc_lambda: float = 0.0
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        ewc_penalty: Dict with 'fisher' and 'optimal_params' for EWC regularization
        ewc_lambda: EWC regularization strength
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    criterion = nn.BCEWithLogitsLoss()
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Add EWC penalty if provided (for continual learning)
        if ewc_penalty is not None and ewc_lambda > 0:
            ewc_loss = compute_ewc_loss(
                model,
                ewc_penalty['fisher'],
                ewc_penalty['optimal_params'],
                ewc_lambda
            )
            loss = loss + ewc_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Evaluate model on test set.
    
    Returns:
        loss: Average BCE loss
        accuracy: Classification accuracy
        auc: Area under ROC curve
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * len(labels)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    # Compute AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5  # If only one class present
    
    return avg_loss, accuracy, auc


def compute_fisher_information(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 200
) -> Dict[str, torch.Tensor]:
    """
    Compute empirical Fisher information matrix (diagonal approximation).
    
    This is the foundation for your FSC algorithm - you'll need to
    aggregate these across clients in a privacy-preserving way.
    
    Returns:
        Dict mapping parameter names to Fisher information estimates
    """
    model.eval()
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    
    criterion = nn.BCEWithLogitsLoss()
    samples_seen = 0
    
    for features, labels in data_loader:
        if samples_seen >= num_samples:
            break
            
        features, labels = features.to(device), labels.to(device)
        
        model.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data ** 2
        
        samples_seen += len(labels)
    
    # Normalize by number of samples
    for name in fisher:
        fisher[name] /= samples_seen
    
    return fisher


def compute_ewc_loss(
    model: nn.Module,
    fisher: Dict[str, torch.Tensor],
    optimal_params: Dict[str, torch.Tensor],
    ewc_lambda: float
) -> torch.Tensor:
    """
    Compute EWC regularization loss.
    
    L_EWC = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
    """
    loss = 0.0
    
    for name, param in model.named_parameters():
        if name in fisher:
            loss += (fisher[name] * (param - optimal_params[name]) ** 2).sum()
    
    return (ewc_lambda / 2) * loss


def get_model_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract model parameters as a dictionary."""
    return {name: param.data.clone() for name, param in model.named_parameters()}


def set_model_parameters(model: nn.Module, params: Dict[str, torch.Tensor]):
    """Set model parameters from a dictionary."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in params:
                param.copy_(params[name])


if __name__ == "__main__":
    # Test the model
    print("Testing CKD Risk Model...")
    
    model = CKDRiskModel(input_dim=10, hidden_dims=[64, 32])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(32, 10)
    y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    
    # Test probability output
    probs = model.predict_proba(x)
    print(f"Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    # Test Fisher computation
    from torch.utils.data import TensorDataset, DataLoader
    dummy_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)).float())
    dummy_loader = DataLoader(dummy_data, batch_size=32)
    
    fisher = compute_fisher_information(model, dummy_loader, torch.device('cpu'))
    print(f"Fisher information computed for {len(fisher)} parameter groups")
