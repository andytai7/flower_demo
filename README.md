# Flower FCL Sandbox

A minimal federated learning project to experiment with Flower before implementing FSC, FHR, and PCFL algorithms.

## Project Structure

```
flower-fcl-sandbox/
├── README.md
├── requirements.txt
├── data/
│   └── generate_synthetic.py    # Create non-IID synthetic health data
├── models/
│   └── risk_model.py            # Simple risk prediction model
├── clients/
│   └── client.py                # Flower client implementation
├── server/
│   ├── server.py                # Basic server
│   └── strategies/
│       ├── fedavg_baseline.py   # FedAvg baseline
│       └── fsc_prototype.py     # Your FSC algorithm skeleton
├── simulation.py                # Run federated simulation
└── experiments/
    └── run_experiment.py        # Experiment runner with logging
```

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run simulation with 5 clients
python simulation.py --num_clients 5 --num_rounds 10

# Run with custom strategy
python simulation.py --strategy fsc --num_clients 5
```

## Learning Progression

1. **Day 1**: Run the basic simulation, understand client-server flow
2. **Day 2**: Modify the model, observe convergence behavior
3. **Day 3**: Implement non-IID data splits, see heterogeneity effects
4. **Day 4**: Add EWC regularization to the client (FSC foundation)
5. **Day 5**: Implement Fisher information aggregation in custom strategy

## Connecting to Your Research

This sandbox uses a synthetic "CKD risk" dataset with features similar to what FLIP-IT will use:
- Age, blood pressure, creatinine, GFR estimates
- Binary classification (high risk / low risk)
- Non-IID splits simulate different practice populations

## Next Steps

After mastering this sandbox:
1. Replace synthetic data with MIMIC-IV subset
2. Add differential privacy (Opacus integration)
3. Implement temporal distribution shift simulation
4. Scale to multi-GPU with Flower simulation
