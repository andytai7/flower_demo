"""
Federated Learning Simulation

Run federated learning experiments with different strategies.
This is your main entry point for experimentation.
"""

import argparse
import torch
import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.insert(0, '.')

from data.generate_synthetic import create_federated_datasets, create_temporal_shift_data
from models.risk_model import CKDRiskModel
from clients.client import CKDFlowerClient, FSCFlowerClient, create_client_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate evaluation metrics across clients using weighted average.
    
    This is called by Flower after each evaluation round.
    """
    accuracies = [num_samples * m["accuracy"] for num_samples, m in metrics]
    aucs = [num_samples * m["auc"] for num_samples, m in metrics]
    total_samples = sum(num_samples for num_samples, _ in metrics)
    
    return {
        "accuracy": sum(accuracies) / total_samples,
        "auc": sum(aucs) / total_samples,
    }


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate training metrics."""
    losses = [num_samples * m["train_loss"] for num_samples, m in metrics]
    total_samples = sum(num_samples for num_samples, _ in metrics)
    
    return {"train_loss": sum(losses) / total_samples}


def get_strategy(
    strategy_name: str,
    num_clients: int,
    fraction_fit: float = 0.5,
    min_fit_clients: int = 2
) -> fl.server.strategy.Strategy:
    """
    Get a Flower strategy by name.
    
    Available strategies:
    - fedavg: Standard Federated Averaging
    - fedprox: FedAvg with proximal term
    - fedadam: Server-side Adam optimization
    """
    common_args = {
        "fraction_fit": fraction_fit,
        "fraction_evaluate": 1.0,  # Evaluate all clients
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": num_clients,
        "min_available_clients": num_clients,
        "evaluate_metrics_aggregation_fn": weighted_average,
        "fit_metrics_aggregation_fn": fit_metrics_aggregation,
    }
    
    if strategy_name == "fedavg":
        return fl.server.strategy.FedAvg(**common_args)
    
    elif strategy_name == "fedprox":
        return fl.server.strategy.FedProx(
            proximal_mu=0.1,
            **common_args
        )
    
    elif strategy_name == "fedadam":
        return fl.server.strategy.FedAdam(
            eta=0.01,
            eta_l=0.01,
            beta_1=0.9,
            beta_2=0.99,
            tau=1e-3,
            **common_args
        )
    
    elif strategy_name == "fsc":
        # Custom FSC strategy - you'll implement this fully
        # For now, use FedAvg as placeholder
        print("Note: FSC strategy using FedAvg baseline - implement in server/strategies/")
        return fl.server.strategy.FedAvg(**common_args)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def run_simulation(
    num_clients: int = 5,
    num_rounds: int = 10,
    samples_per_client: int = 500,
    strategy_name: str = "fedavg",
    local_epochs: int = 1,
    fraction_fit: float = 0.5,
    device: str = "auto",
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Run a federated learning simulation.
    
    Args:
        num_clients: Number of federated clients (practices)
        num_rounds: Number of federation rounds
        samples_per_client: Training samples per client
        strategy_name: Aggregation strategy
        local_epochs: Local training epochs per round
        fraction_fit: Fraction of clients to sample per round
        device: Device to use (auto, cpu, cuda)
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with training history and final metrics
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Determine device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    if verbose:
        print(f"Running simulation on {device}")
        print(f"Clients: {num_clients}, Rounds: {num_rounds}, Strategy: {strategy_name}")
        print("-" * 50)
    
    # Create federated datasets
    train_loaders, test_loaders, scaler = create_federated_datasets(
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        random_state=seed
    )
    
    if verbose:
        print("Data distribution across clients:")
        for i, loader in enumerate(train_loaders):
            labels = loader.dataset.labels.numpy()
            print(f"  Client {i}: {len(labels)} samples, "
                  f"positive rate: {labels.mean():.1%}")
        print("-" * 50)
    
    # Model factory function
    def model_fn():
        return CKDRiskModel(input_dim=10, hidden_dims=[64, 32])
    
    # Create client factory
    client_fn = create_client_fn(
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        model_fn=model_fn,
        device=device,
        client_class=CKDFlowerClient
    )
    
    # Get strategy
    strategy = get_strategy(
        strategy_name=strategy_name,
        num_clients=num_clients,
        fraction_fit=fraction_fit,
        min_fit_clients=max(2, int(num_clients * fraction_fit))
    )
    
    # Configure resources for simulation
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    if device.type == "cuda":
        client_resources["num_gpus"] = 0.1  # Share GPU across clients
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
    
    # Extract results
    results = {
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "strategy": strategy_name,
        "losses_distributed": history.losses_distributed,
        "metrics_distributed": history.metrics_distributed,
        "losses_centralized": history.losses_centralized,
        "metrics_centralized": history.metrics_centralized,
    }
    
    if verbose:
        print("\n" + "=" * 50)
        print("SIMULATION COMPLETE")
        print("=" * 50)
        
        if history.metrics_distributed:
            final_metrics = history.metrics_distributed[-1][1]
            print(f"Final distributed metrics:")
            print(f"  Accuracy: {final_metrics.get('accuracy', 'N/A'):.2%}")
            print(f"  AUC: {final_metrics.get('auc', 'N/A'):.3f}")
        
        if history.losses_distributed:
            final_loss = history.losses_distributed[-1][1]
            print(f"  Final loss: {final_loss:.4f}")
    
    return results


def run_continual_learning_experiment(
    num_clients: int = 3,
    num_time_periods: int = 3,
    rounds_per_period: int = 5,
    strategy_name: str = "fedavg",
    verbose: bool = True
) -> Dict:
    """
    Run a continual learning experiment with temporal distribution shift.
    
    This simulates the scenario where data distribution changes over time,
    which is the core challenge your FCL algorithms address.
    """
    if verbose:
        print("=" * 50)
        print("CONTINUAL LEARNING EXPERIMENT")
        print(f"Clients: {num_clients}, Periods: {num_time_periods}")
        print("=" * 50)
    
    # Create temporal data
    temporal_data = create_temporal_shift_data(
        num_clients=num_clients,
        num_time_periods=num_time_periods,
        samples_per_period=300
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Track metrics across time periods
    all_results = []
    
    for period in range(num_time_periods):
        if verbose:
            print(f"\n--- Time Period {period + 1}/{num_time_periods} ---")
        
        # Get data for this period
        train_loaders = [temporal_data[c][period][0] for c in range(num_clients)]
        test_loaders = [temporal_data[c][period][1] for c in range(num_clients)]
        
        # Model factory
        def model_fn():
            return CKDRiskModel(input_dim=10, hidden_dims=[64, 32])
        
        # Create clients
        client_fn = create_client_fn(
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            model_fn=model_fn,
            device=device,
            client_class=CKDFlowerClient
        )
        
        # Run FL for this period
        strategy = get_strategy(strategy_name, num_clients)
        
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=rounds_per_period),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )
        
        # Evaluate on ALL previous periods (backward transfer)
        period_results = {
            "period": period,
            "current_metrics": history.metrics_distributed[-1][1] if history.metrics_distributed else {},
        }
        
        all_results.append(period_results)
        
        if verbose and history.metrics_distributed:
            metrics = history.metrics_distributed[-1][1]
            print(f"Period {period + 1} results: "
                  f"Accuracy={metrics.get('accuracy', 0):.2%}, "
                  f"AUC={metrics.get('auc', 0):.3f}")
    
    return {"periods": all_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flower FL simulation")
    
    parser.add_argument("--num_clients", type=int, default=5,
                        help="Number of federated clients")
    parser.add_argument("--num_rounds", type=int, default=10,
                        help="Number of federation rounds")
    parser.add_argument("--samples_per_client", type=int, default=500,
                        help="Training samples per client")
    parser.add_argument("--strategy", type=str, default="fedavg",
                        choices=["fedavg", "fedprox", "fedadam", "fsc"],
                        help="Aggregation strategy")
    parser.add_argument("--local_epochs", type=int, default=1,
                        help="Local training epochs per round")
    parser.add_argument("--fraction_fit", type=float, default=0.5,
                        help="Fraction of clients to sample per round")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--experiment", type=str, default="basic",
                        choices=["basic", "continual"],
                        help="Type of experiment to run")
    
    args = parser.parse_args()
    
    if args.experiment == "basic":
        results = run_simulation(
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            samples_per_client=args.samples_per_client,
            strategy_name=args.strategy,
            local_epochs=args.local_epochs,
            fraction_fit=args.fraction_fit,
            device=args.device,
            seed=args.seed
        )
    else:
        results = run_continual_learning_experiment(
            num_clients=args.num_clients,
            num_time_periods=3,
            rounds_per_period=args.num_rounds,
            strategy_name=args.strategy
        )
    
    print("\nExperiment complete!")
