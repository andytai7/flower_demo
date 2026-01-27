"""
Flower Client for CKD Risk Prediction

This client handles local training and evaluation for federated learning.
It's designed to be extended for your FCL algorithms.
"""

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import numpy as np

import sys
sys.path.append('..')
from models.risk_model import (
    CKDRiskModel, 
    train_epoch, 
    evaluate,
    compute_fisher_information,
    get_model_parameters
)


class CKDFlowerClient(fl.client.NumPyClient):
    """
    Basic Flower client for CKD risk prediction.
    
    This implements the standard FedAvg client interface.
    You'll extend this for FSC, FHR, and PCFL.
    """
    
    def __init__(
        self,
        client_id: int,
        model: CKDRiskModel,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 1,
        learning_rate: float = 0.01
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        
        # For continual learning experiments
        self.fisher_info: Optional[Dict] = None
        self.optimal_params: Optional[Dict] = None
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model parameters as a list of NumPy arrays."""
        return [
            val.cpu().numpy() 
            for val in self.model.state_dict().values()
        ]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(
            self.model.state_dict().keys(),
            parameters
        )
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on local data.
        
        Args:
            parameters: Current global model parameters
            config: Training configuration from server
        
        Returns:
            Updated parameters, number of samples, metrics dict
        """
        # Update local model with global parameters
        self.set_parameters(parameters)
        
        # Get training config
        local_epochs = config.get("local_epochs", self.local_epochs)
        lr = config.get("learning_rate", self.learning_rate)
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Train locally
        total_loss = 0.0
        for epoch in range(local_epochs):
            loss = train_epoch(
                self.model,
                self.train_loader,
                optimizer,
                self.device
            )
            total_loss += loss
        
        avg_loss = total_loss / local_epochs
        
        # Return updated parameters
        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            {"train_loss": avg_loss, "client_id": self.client_id}
        )
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local test data.
        
        Returns:
            Loss, number of samples, metrics dict
        """
        self.set_parameters(parameters)
        
        loss, accuracy, auc = evaluate(
            self.model,
            self.test_loader,
            self.device
        )
        
        return (
            float(loss),
            len(self.test_loader.dataset),
            {
                "accuracy": accuracy,
                "auc": auc,
                "client_id": self.client_id
            }
        )


class FSCFlowerClient(CKDFlowerClient):
    """
    Flower client with Federated Synaptic Consolidation support.
    
    This extends the basic client to compute and send Fisher information
    for importance-weighted parameter protection.
    """
    
    def __init__(
        self,
        client_id: int,
        model: CKDRiskModel,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 1,
        learning_rate: float = 0.01,
        ewc_lambda: float = 0.4
    ):
        super().__init__(
            client_id, model, train_loader, test_loader,
            device, local_epochs, learning_rate
        )
        self.ewc_lambda = ewc_lambda
        self.fisher_info = None
        self.optimal_params = None
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train with EWC regularization if Fisher info is available.
        """
        self.set_parameters(parameters)
        
        local_epochs = config.get("local_epochs", self.local_epochs)
        lr = config.get("learning_rate", self.learning_rate)
        compute_fisher = config.get("compute_fisher", False)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Prepare EWC penalty if we have Fisher info from previous task
        ewc_penalty = None
        if self.fisher_info is not None and self.optimal_params is not None:
            ewc_penalty = {
                'fisher': self.fisher_info,
                'optimal_params': self.optimal_params
            }
        
        # Train locally with optional EWC
        total_loss = 0.0
        for epoch in range(local_epochs):
            loss = train_epoch(
                self.model,
                self.train_loader,
                optimizer,
                self.device,
                ewc_penalty=ewc_penalty,
                ewc_lambda=self.ewc_lambda
            )
            total_loss += loss
        
        avg_loss = total_loss / local_epochs
        
        # Compute Fisher information if requested by server
        metrics = {"train_loss": avg_loss, "client_id": self.client_id}
        
        if compute_fisher:
            fisher = compute_fisher_information(
                self.model, 
                self.train_loader, 
                self.device
            )
            # Store for next round's EWC penalty
            self.fisher_info = fisher
            self.optimal_params = get_model_parameters(self.model)
            
            # Send Fisher diagonal as flattened array (for aggregation)
            fisher_flat = np.concatenate([
                f.cpu().numpy().flatten() for f in fisher.values()
            ])
            metrics["fisher_info"] = fisher_flat.tolist()[:100]  # Truncate for demo
        
        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            metrics
        )


def create_client_fn(
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    model_fn,
    device: torch.device,
    client_class: type = CKDFlowerClient
):
    """
    Factory function to create clients for Flower simulation.
    
    Args:
        train_loaders: List of training DataLoaders
        test_loaders: List of test DataLoaders
        model_fn: Function that returns a new model instance
        device: Device to use
        client_class: Client class to instantiate
    
    Returns:
        Function that creates a client given a client ID
    """
    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        model = model_fn()
        
        return client_class(
            client_id=client_id,
            model=model,
            train_loader=train_loaders[client_id],
            test_loader=test_loaders[client_id],
            device=device
        ).to_client()
    
    return client_fn


if __name__ == "__main__":
    # Test client creation
    import sys
    sys.path.append('..')
    from data.generate_synthetic import create_federated_datasets
    
    print("Testing Flower client...")
    
    # Create data
    train_loaders, test_loaders, _ = create_federated_datasets(
        num_clients=3,
        samples_per_client=200
    )
    
    # Create client
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CKDRiskModel()
    
    client = CKDFlowerClient(
        client_id=0,
        model=model,
        train_loader=train_loaders[0],
        test_loader=test_loaders[0],
        device=device
    )
    
    # Test get_parameters
    params = client.get_parameters({})
    print(f"Number of parameter arrays: {len(params)}")
    print(f"Total parameters: {sum(p.size for p in params):,}")
    
    # Test fit
    updated_params, num_samples, metrics = client.fit(params, {"local_epochs": 1})
    print(f"After fit - samples: {num_samples}, loss: {metrics['train_loss']:.4f}")
    
    # Test evaluate
    loss, num_samples, metrics = client.evaluate(updated_params, {})
    print(f"Evaluation - loss: {loss:.4f}, accuracy: {metrics['accuracy']:.2%}, "
          f"AUC: {metrics['auc']:.3f}")
