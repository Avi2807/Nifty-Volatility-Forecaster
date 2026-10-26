# Nifty Volatility Forecaster

Forecast next-day Nifty 50 realized volatility with six models: Naive, HAR-RV,
GARCH(1,1), MLP, LSTM, and a time-series Transformer. The models are built
from scratch with NumPy; news sentiment is an optional ablation.

## Status

The repository currently contains the Phase 0 project scaffold and the
leakage-safe data foundation. Dataset-specific column mappings and the model
ladder are implemented incrementally according to
[the project blueprint](docs/PROJECT_BLUEPRINT.md).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Place downloaded datasets in `data/raw/` and record their sources in
[SOURCES.md](data/raw/SOURCES.md). Raw datasets are ignored by Git.

## Constraints

Only Python, NumPy, pandas, matplotlib, and the standard library are allowed.
No ML libraries or pretrained models are used. The complete data contracts,
chronological splits, model specifications, and phase gates are documented in
[docs/PROJECT_BLUEPRINT.md](docs/PROJECT_BLUEPRINT.md).
