# Nifty Volatility Forecaster

Forecast next-day Nifty 50 realized variance with Naive, HAR-RV, GARCH(1,1),
MLP, LSTM and a time-series Transformer implemented from scratch.
The complete specification is in `docs/PROJECT_BLUEPRINT.md`.

## Hard rules

- Allowed imports are NumPy, pandas, matplotlib, and the Python standard library.
- Do not use ML libraries or pretrained models.
- Neural layers use float64 arrays, explicit seeded `np.random.default_rng`
  instances, and accumulating gradients after `zero_grad()`.
- Split by `target_date`; fit scalers and target statistics on training rows only.
- `log_rv` is log realized variance. Convert to volatility only with
  `src.io.log_rv_to_vol`.
- `src/eval` reads forecast CSVs and must not import `src/models`.
