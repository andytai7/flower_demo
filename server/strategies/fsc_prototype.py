"""
Federated Synaptic Consolidation (FSC) Strategy

This is your first novel algorithm. It extends EWC to federated settings
with privacy-preserving Fisher information aggregation.

Key Innovation: Aggregate importance weights (Fisher information) across
clients without sharing raw gradients or data.

TODO: Implement the following:
1. Privacy-preserving Fisher aggregation (consider DP noise addition)
2. Weighted parameter protection during aggregation
3. Task boundary detection for continual learning scenarios
"""

import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from functools import reduce


class FederatedSynapticConsolidation(Strategy):
    """
    Federated Synaptic Consolidation Strategy.
    
    This strategy extends FedAvg with:
    1. Fisher information aggregation for importance weighting
    2. EWC-style regularization applied during aggregation
    3. Support for continual learning with task boundaries
    
    The FSC loss at each client is:
    L_FSC = L_task(theta) + (lambda/2) * sum_i F_bar_i * (theta_i - theta*_i)^2
    
    where F_bar_i is the federated Fisher information aggregated across clients.
    """
    
    def __init__(
        self,
        fraction_fit: float = 0.5,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        ewc_lambda: float = 0.4,
        fisher_aggregation: str = "weighted_mean",  # or "max", "sum"
        dp_epsilon: Optional[float] = None,  # Differential privacy budget
        dp_delta: float = 1e-5,
    ):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        
        # FSC-specific parameters
        self.ewc_lambda = ewc_lambda
        self.fisher_aggregation = fisher_aggregation
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        
        # State tracking
        self.current_parameters: Optional[Parameters] = None
        self.aggregated_fisher: Optional[np.ndarray] = None
        self.optimal_parameters: Optional[List[np.ndarray]] = None
        self.current_task: int = 0
    
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.current_parameters
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        # Sample clients
        sample_size = max(
            int(client_manager.num_available() * self.fraction_fit),
            self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients
        )
        
        # Configuration for clients
        config = {
            "server_round": server_round,
            "local_epochs": 1,
            "compute_fisher": True,  # Request Fisher computation
            "ewc_lambda": self.ewc_lambda,
        }
        
        # Send current parameters and config to selected clients
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates with Fisher-weighted consolidation.
        
        This is the core of FSC. Steps:
        1. Aggregate model parameters (standard FedAvg)
        2. Aggregate Fisher information across clients
        3. Apply importance weighting to protect critical parameters
        """
        if not results:
            return None, {}
        
        # Extract parameters and Fisher info from results
        weights_results = []
        fisher_results = []
        
        for client, fit_res in results:
            parameters = parameters_to_ndarrays(fit_res.parameters)
            num_samples = fit_res.num_examples
            weights_results.append((parameters, num_samples))
            
            # Extract Fisher information if provided
            if "fisher_info" in fit_res.metrics:
                fisher = np.array(fit_res.metrics["fisher_info"])
                fisher_results.append((fisher, num_samples))
        
        # Step 1: Aggregate parameters (weighted average)
        aggregated_params = self._aggregate_parameters(weights_results)
        
        # Step 2: Aggregate Fisher information
        if fisher_results:
            self.aggregated_fisher = self._aggregate_fisher(fisher_results)
        
        # Step 3: Apply Fisher-weighted protection (optional refinement)
        # In basic FSC, the protection happens at the client during training
        # Advanced: could apply server-side parameter adjustment here
        
        # Store optimal parameters for next round's EWC
        if self.optimal_parameters is None:
            self.optimal_parameters = aggregated_params
        
        # Update current parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_params)
        self.current_parameters = parameters_aggregated
        
        # Metrics
        metrics = {
            "num_clients": len(results),
            "num_failures": len(failures),
        }
        if self.aggregated_fisher is not None:
            metrics["fisher_mean"] = float(np.mean(self.aggregated_fisher))
            metrics["fisher_max"] = float(np.max(self.aggregated_fisher))
        
        return parameters_aggregated, metrics
    
    def _aggregate_parameters(
        self,
        results: List[Tuple[List[np.ndarray], int]]
    ) -> List[np.ndarray]:
        """Weighted average of model parameters."""
        
        # Calculate total samples
        total_samples = sum(num_samples for _, num_samples in results)
        
        # Weighted average
        aggregated = [
            np.zeros_like(params[0]) for params in [results[0][0]]
        ][0:len(results[0][0])]
        
        aggregated = []
        for layer_idx in range(len(results[0][0])):
            layer_sum = np.zeros_like(results[0][0][layer_idx])
            for params, num_samples in results:
                weight = num_samples / total_samples
                layer_sum += params[layer_idx] * weight
            aggregated.append(layer_sum)
        
        return aggregated
    
    def _aggregate_fisher(
        self,
        fisher_results: List[Tuple[np.ndarray, int]]
    ) -> np.ndarray:
        """
        Aggregate Fisher information across clients.
        
        Options:
        - weighted_mean: Weight by number of samples
        - max: Take element-wise maximum (most conservative)
        - sum: Sum all Fisher values (increases regularization)
        
        TODO: Add differential privacy noise here if dp_epsilon is set
        """
        total_samples = sum(n for _, n in fisher_results)
        
        if self.fisher_aggregation == "weighted_mean":
            aggregated = np.zeros_like(fisher_results[0][0])
            for fisher, num_samples in fisher_results:
                aggregated += fisher * (num_samples / total_samples)
        
        elif self.fisher_aggregation == "max":
            all_fisher = np.stack([f for f, _ in fisher_results])
            aggregated = np.max(all_fisher, axis=0)
        
        elif self.fisher_aggregation == "sum":
            aggregated = np.sum([f for f, _ in fisher_results], axis=0)
        
        else:
            raise ValueError(f"Unknown aggregation: {self.fisher_aggregation}")
        
        # Add DP noise if configured
        if self.dp_epsilon is not None:
            aggregated = self._add_dp_noise(aggregated)
        
        return aggregated
    
    def _add_dp_noise(self, fisher: np.ndarray) -> np.ndarray:
        """
        Add differential privacy noise to Fisher information.
        
        Uses Gaussian mechanism with calibrated noise.
        
        TODO: Implement proper sensitivity calculation and noise calibration
        """
        # Placeholder - implement proper DP mechanism
        sensitivity = 1.0  # Need to calculate actual sensitivity
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.dp_delta)) / self.dp_epsilon
        noise = np.random.normal(0, sigma, fisher.shape)
        return np.maximum(fisher + noise, 0)  # Fisher should be non-negative
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure model evaluation."""
        
        if self.fraction_evaluate == 0.0:
            return []
        
        config = {"server_round": server_round}
        evaluate_ins = EvaluateIns(parameters, config)
        
        # Sample clients for evaluation
        sample_size = max(
            int(client_manager.num_available() * self.fraction_evaluate),
            self.min_evaluate_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_evaluate_clients
        )
        
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        
        if not results:
            return None, {}
        
        # Weighted average of loss and metrics
        total_samples = sum(r.num_examples for _, r in results)
        
        weighted_loss = sum(
            r.num_examples * r.loss for _, r in results
        ) / total_samples
        
        # Aggregate custom metrics
        metrics = {}
        metric_keys = results[0][1].metrics.keys() if results[0][1].metrics else []
        
        for key in metric_keys:
            if key == "client_id":
                continue
            weighted_sum = sum(
                r.num_examples * r.metrics.get(key, 0) 
                for _, r in results
            )
            metrics[key] = weighted_sum / total_samples
        
        return weighted_loss, metrics
    
    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation (optional)."""
        return None
    
    def mark_task_boundary(self):
        """
        Mark a task boundary for continual learning.
        
        Call this when transitioning to a new task/time period.
        This stores the current parameters as optimal and increments task counter.
        """
        if self.current_parameters is not None:
            self.optimal_parameters = parameters_to_ndarrays(self.current_parameters)
        self.current_task += 1
        print(f"Task boundary marked. Now on task {self.current_task}")


# Factory function for easy strategy creation
def create_fsc_strategy(
    num_clients: int,
    ewc_lambda: float = 0.4,
    dp_epsilon: Optional[float] = None,
    **kwargs
) -> FederatedSynapticConsolidation:
    """Create an FSC strategy with reasonable defaults."""
    
    return FederatedSynapticConsolidation(
        fraction_fit=0.5,
        fraction_evaluate=1.0,
        min_fit_clients=max(2, num_clients // 2),
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        ewc_lambda=ewc_lambda,
        dp_epsilon=dp_epsilon,
        **kwargs
    )


if __name__ == "__main__":
    # Test strategy creation
    strategy = create_fsc_strategy(num_clients=5, ewc_lambda=0.4)
    print(f"Created FSC strategy with lambda={strategy.ewc_lambda}")
    print(f"Fisher aggregation method: {strategy.fisher_aggregation}")
