# Flower FCL Sandbox

A minimal federated learning project to experiment with Flower before implementing FSC, FHR, and PCFL algorithms.

## Project Structure

```
flower_demo/
├── README.md
├── pyproject.toml               # Project config and dependencies
├── data/
│   └── generate_synthetic.py    # Create non-IID synthetic health data
├── models/
│   └── risk_model.py            # Simple risk prediction model
├── clients/
│   └── client.py                # Flower client implementation
├── server/
│   └── strategies/
│       └── fsc_prototype.py     # Your FSC algorithm skeleton
└── simulation.py                # Run federated simulation
```

## Quick Start

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter project
git clone https://github.com/andytai7/flower_demo.git
cd flower_demo

# Create venv and install dependencies (uv does both in one step)
uv sync

# Run simulation with 5 clients
uv run python simulation.py --num_clients 5 --num_rounds 10

# Run with custom strategy
uv run python simulation.py --strategy fsc --num_clients 5
```

## Alternative: Direct uv run (no explicit sync needed)

```bash
# uv will automatically create venv and install deps on first run
uv run python simulation.py --num_clients 5 --num_rounds 10
```

## Adding Dependencies

```bash
# Add a new package
uv add requests

# Add dev dependency
uv add --dev pytest

# Add optional dependency group
uv add --optional privacy opacus
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

## Running Tests

```bash
uv run pytest
uv run pytest --cov=.  # with coverage
```

## Code Quality

```bash
uv run ruff check .     # linting
uv run ruff format .    # formatting
uv run mypy .           # type checking
```

## Next Steps

After mastering this sandbox:
1. Replace synthetic data with MIMIC-IV subset
2. Add differential privacy (`uv add --optional privacy opacus`)
3. Implement temporal distribution shift simulation
4. Scale to multi-GPU with Flower simulation
