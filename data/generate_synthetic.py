"""
Synthetic CKD Risk Data Generator

Creates non-IID federated datasets simulating different practice populations.
This mirrors the data heterogeneity you'll encounter in FLIP-IT.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
import os


class CKDRiskDataset(Dataset):
    """PyTorch Dataset for CKD risk prediction."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def generate_practice_data(
    n_samples: int,
    practice_id: int,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic CKD risk data for a single practice.
    
    Each practice has slightly different population characteristics
    to simulate real-world non-IID conditions.
    
    Features (based on CKD risk factors):
    - age: Patient age
    - systolic_bp: Systolic blood pressure
    - diastolic_bp: Diastolic blood pressure  
    - serum_creatinine: Serum creatinine level
    - egfr: Estimated GFR
    - albumin: Albumin level
    - diabetes: Diabetes indicator (0/1)
    - hypertension: Hypertension indicator (0/1)
    - bmi: Body mass index
    - smoking: Smoking status (0/1)
    """
    np.random.seed(random_state + practice_id)
    
    # Practice-specific population shifts (non-IID)
    # Urban practices might have younger, different comorbidity profiles
    # Rural practices might have older populations
    age_shift = practice_id * 3  # Different age distributions
    bp_shift = practice_id * 2   # Different BP baselines
    
    # Generate features
    age = np.clip(np.random.normal(55 + age_shift, 15, n_samples), 18, 95)
    systolic_bp = np.clip(np.random.normal(130 + bp_shift, 20, n_samples), 90, 200)
    diastolic_bp = np.clip(np.random.normal(80 + bp_shift/2, 12, n_samples), 50, 120)
    serum_creatinine = np.clip(np.random.exponential(1.2, n_samples), 0.5, 10)
    egfr = np.clip(120 - age * 0.5 - serum_creatinine * 15 + np.random.normal(0, 10, n_samples), 5, 150)
    albumin = np.clip(np.random.normal(4.0, 0.5, n_samples), 2, 5.5)
    diabetes = np.random.binomial(1, 0.15 + practice_id * 0.02, n_samples)
    hypertension = np.random.binomial(1, 0.25 + practice_id * 0.03, n_samples)
    bmi = np.clip(np.random.normal(27 + practice_id, 5, n_samples), 15, 50)
    smoking = np.random.binomial(1, 0.20, n_samples)
    
    features = np.column_stack([
        age, systolic_bp, diastolic_bp, serum_creatinine, egfr,
        albumin, diabetes, hypertension, bmi, smoking
    ])
    
    # Generate labels based on clinical risk factors
    # This is a simplified risk model
    risk_score = (
        0.02 * (age - 50) +
        0.01 * (systolic_bp - 120) +
        0.3 * (serum_creatinine - 1.0) +
        -0.02 * (egfr - 90) +
        0.5 * diabetes +
        0.3 * hypertension +
        0.02 * (bmi - 25) +
        0.2 * smoking +
        np.random.normal(0, 0.5, n_samples)  # Noise
    )
    
    # Convert to binary classification (high risk threshold)
    labels = (risk_score > np.percentile(risk_score, 70)).astype(float)
    
    return features, labels


def create_federated_datasets(
    num_clients: int = 5,
    samples_per_client: int = 500,
    test_split: float = 0.2,
    random_state: int = 42
) -> Tuple[List[DataLoader], List[DataLoader], StandardScaler]:
    """
    Create federated train/test datasets for multiple clients.
    
    Returns:
        train_loaders: List of training DataLoaders (one per client)
        test_loaders: List of test DataLoaders (one per client)
        scaler: Fitted StandardScaler for feature normalization
    """
    train_loaders = []
    test_loaders = []
    
    # Fit scaler on combined data from all clients
    all_features = []
    for client_id in range(num_clients):
        features, _ = generate_practice_data(
            samples_per_client, client_id, random_state
        )
        all_features.append(features)
    
    scaler = StandardScaler()
    scaler.fit(np.vstack(all_features))
    
    # Create datasets for each client
    for client_id in range(num_clients):
        features, labels = generate_practice_data(
            samples_per_client, client_id, random_state
        )
        
        # Normalize features
        features_scaled = scaler.transform(features)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels,
            test_size=test_split,
            random_state=random_state,
            stratify=labels
        )
        
        # Create DataLoaders
        train_dataset = CKDRiskDataset(X_train, y_train)
        test_dataset = CKDRiskDataset(X_test, y_test)
        
        train_loaders.append(DataLoader(train_dataset, batch_size=32, shuffle=True))
        test_loaders.append(DataLoader(test_dataset, batch_size=32, shuffle=False))
    
    return train_loaders, test_loaders, scaler


def create_temporal_shift_data(
    num_clients: int = 5,
    num_time_periods: int = 3,
    samples_per_period: int = 300,
    random_state: int = 42
) -> Dict[int, List[Tuple[DataLoader, DataLoader]]]:
    """
    Create datasets with temporal distribution shift.
    
    This simulates the continual learning scenario where data
    distribution changes over time (e.g., COVID impact on CKD risk factors).
    
    Returns:
        Dict mapping client_id -> list of (train_loader, test_loader) per time period
    """
    client_data = {}
    
    for client_id in range(num_clients):
        period_data = []
        
        for period in range(num_time_periods):
            # Shift the random state to create different distributions
            shifted_state = random_state + period * 1000 + client_id
            
            features, labels = generate_practice_data(
                samples_per_period,
                client_id + period * 2,  # Shift population characteristics
                shifted_state
            )
            
            # Apply temporal shift (e.g., aging population, changing comorbidities)
            temporal_shift = period * 0.1
            features[:, 0] += period * 2  # Age increases over time
            features[:, 6] += temporal_shift  # Diabetes prevalence increases
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, labels,
                test_size=0.2,
                random_state=random_state
            )
            
            train_dataset = CKDRiskDataset(X_train, y_train)
            test_dataset = CKDRiskDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            period_data.append((train_loader, test_loader))
        
        client_data[client_id] = period_data
    
    return client_data


if __name__ == "__main__":
    # Test data generation
    print("Generating federated datasets...")
    
    train_loaders, test_loaders, scaler = create_federated_datasets(
        num_clients=5,
        samples_per_client=500
    )
    
    print(f"Created {len(train_loaders)} client datasets")
    
    for i, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
        train_samples = len(train_loader.dataset)
        test_samples = len(test_loader.dataset)
        
        # Get label distribution
        train_labels = train_loader.dataset.labels.numpy()
        pos_ratio = train_labels.mean()
        
        print(f"Client {i}: {train_samples} train, {test_samples} test, "
              f"positive ratio: {pos_ratio:.2%}")
    
    print("\nGenerating temporal shift data...")
    temporal_data = create_temporal_shift_data(num_clients=3, num_time_periods=3)
    
    for client_id, periods in temporal_data.items():
        print(f"Client {client_id}: {len(periods)} time periods")
