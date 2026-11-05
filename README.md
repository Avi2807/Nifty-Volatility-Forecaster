# Nifty Volatility Forecaster

Forecasts next-day realized volatility of the Nifty 50. The target for day t is the log
realized variance of day t+1, where RV is the sum of squared 5-minute intraday log returns.

## The no-ML-library rule

Every model, layer, gradient and optimiser is written by hand in NumPy. These are banned:
scikit-learn, PyTorch, TensorFlow, JAX, statsmodels, arch, NLTK, spaCy, transformers and
LangChain. Allowed: Python 3.11, NumPy, pandas (CSV loading and date handling only, confined
to `src/data/loaders.py`), matplotlib (figures only), FastAPI and uvicorn (serving only).
`tests/test_loaders.py` scans the imports and fails if the rule is broken.

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python run_all.py
```

`python -m src.nn.gradcheck` prints the gradient check for every layer.

`requirements.txt` is what the project runs on. `requirements-dev.txt` adds the test tools.

## Placing the data files

Put these in `data/raw/`. Names are matched case-insensitively and subfolders are searched.

| Filename must match | Contents | Required |
|---|---|---|
| `*minute*.csv` | Nifty 50 1-minute OHLC (Kaggle): date, open, high, low, close | yes |
| `*vix*.csv` | India VIX daily from NSE: date, close. Several yearly files are fine | yes |
| `*headlines*.csv` | India News Headlines: publish_date, headline_category, headline_text | yes |
| `*66agree*` | Financial PhraseBank `Sentences_66Agree.txt` | yes |
| `*turnover*.csv` or `NIFTY 50-*.csv` | Nifty 50 daily turnover from NSE: date, a column containing "turnover". Several yearly files are fine | no (feature set to 0, with a warning) |

NSE's `Historical_PR` price files have no turnover column, so they don't count as the turnover file.
They are optional, and supply the published close in the app's ticker. `python run_all.py` copies them to
`data/processed/index_closes.csv`. No feature, model or score reads them, and `daily_bars.csv` keeps its
own close, which is the last intraday bar's and differs slightly from the published figure.

`config/expiry_dates.csv` holds the weekly expiry calendar: Thursdays to 2025-08-28, Tuesdays
from 2025-09-02, moved to the previous session on holidays. `python -m src.data.expiry`
regenerates it from the minute file.

## Synthetic data

If any required file is missing, the whole run uses simulated data, never a mix. Its inputs
and outputs stay in `data/synthetic/` and `artifacts_synthetic/`. Real results only ever come
from `data/raw/` and go to `data/processed/` and `artifacts/`. Synthetic numbers are for
testing the pipeline and must not be reported. `python run_all.py --synthetic` forces this mode.

## Data rules

- A session needs at least 60 of 75 bars. Dropped dates go to `dropped_days.csv`.
- "Previous day" and "next day" mean the previous and next retained session.
- VIX and turnover forward-fill by at most one session, and a longer gap drops the row.
  There are no backward fills.
- `expiry_next` is the only feature that looks past day t. It reads the next session's date
  and the expiry calendar, both published in advance.
- Splits use the feature date t: train to 2019-12-31, validation 2020, test from 2021-01-01.

## API

```
.venv\Scripts\python -m uvicorn src.api.main:app --port 8000
```

The API only reads files that `run_all.py` has already written, and it never loads model code.
It stops at startup with a clear message if an artifact is missing or if the artifacts are
labelled as synthetic. CORS allows the frontend at `localhost:5173`. Interactive docs are at
`http://localhost:8000/docs`.

| Endpoint | Returns |
|---|---|
| `GET /api/health` | Status, data source, number of test days, model names |
| `GET /api/summary` | `metrics.json`, `model_card.json` and the last forecast of the test period |
| `GET /api/forecasts?start=&end=&models=` | Rows from `forecasts.csv`, plus `nifty_close`: NSE's published close on `date`, or null without the NSE files. `models` is comma-separated, and dates filter on `date`, inclusive |
| `GET /api/day/{date}` | That day's details. 404 with a helpful message if it isn't a test date |
| `GET /api/var?model=` | Test-period VaR series, breach dates and Kupiec result. `model` defaults to `transformer` |
| `GET /api/sentiment?start=&end=` | Daily news sentiment, ending on 2023-06-30 |
| `GET /api/live` | The latest next-session forecast from `python -m src.live`, or 404 if none has been made. It's read per request, so a new one appears without a restart. |
| `GET /api/forward` | The forward record and its running scores, plus the simpler models' later-check scores, or 404 if `python -m src.live` hasn't made a record yet. Also read per request. |
| `GET /api/later?start=&end=&models=` | The later check's rows and scores. Rows add the no-news model's `var99_transformer_nosent` and `breach_transformer_nosent`, and `nifty_close`. The with-news model isn't in them |

Bad dates, unknown models and a start after the end return 422 with a message. The later check is
kept apart from the main results: `/api/forecasts` never returns its rows. Its scores also appear
in `/api/summary` under `metrics.later_check`.

## Frontend

`frontend/` is a React app that replays the test period from this API. With the API running,
`cd frontend`, `npm install`, then `npm run dev` and open http://localhost:5173. See
`frontend/README.md`.

## The forward record

`python -m src.live` adds each forecast to `artifacts/forward_record.json` and scores the earlier ones whose
session has now happened. These scores are kept apart from the study's.

- A row is **live** only if it was saved before its session opened. Anything later is **backfilled**: it uses
  only the data up to the evening before, but its outcome could already have been known, so it isn't live
  evidence. That's also how a missed evening gets filled in, as far back as Yahoo's bars allow.
- The first forecast saved for a date is never rewritten. A model added after a row was saved stays null on it.
- Bars stop at 15 May 2026 in the minute file and reach back only about 60 days at Yahoo, so there are no
  forecasts from mid-May until October 2026. GARCH needs unbroken returns, so NSE's daily open and close
  figures fill that gap for it alone.

`python -m src.eval.challengers` trains no-news versions of the MLP and LSTM with their existing recipes and no
tuning, then scores them and GARCH on the later check. They run beside the live model and never replace it.
It also sets the calibrated VaR multiplier from the later check, so it counts as post-test and only later days
can judge it. The rule's own line and warning stay as they were.

## Results

All numbers come from real data. `python run_all.py` reproduces them and writes them to
`artifacts/metrics.json`. The models were frozen, with checkpoint checksums in
`artifacts/frozen_models.json`, before any forecast was made for the test set. The test set
was scored once, and no model changed afterwards.

**Test period: 615 forecasts for 2021-01-04 to 2023-07-03.** MSE on log RV was the main score,
set before scoring. Lower is better. 99% VaR should breach about 6.2 times.

| Model | MSE | QLIKE | VaR breaches | Kupiec LR (reject above 3.84) |
|---|---|---|---|---|
| Transformer, with news | 0.2463 | 0.1338 | 24 | 30.19, rejected |
| Transformer, no news | 0.2453 | 0.1437 | 28 | 41.98, rejected |
| HAR-RV | 0.2705 | 0.1571 | 26 | 35.92, rejected |
| Naive | 0.3609 | 0.2042 | 26 | 35.92, rejected |
| VIX-implied | 1.6748 | 0.5352 | 0 | 12.36, rejected |

**Diebold-Mariano tests, with Newey-West variance and 5 lags.** A negative statistic means the
first model had the lower loss. The p-values are not adjusted for running six tests. A
Bonferroni limit would be 0.0083.

| Comparison | MSE statistic | MSE p | QLIKE statistic | QLIKE p |
|---|---|---|---|---|
| Transformer vs HAR-RV | -2.24 | 0.0250 | -2.89 | 0.0038 |
| Transformer vs naive | -5.89 | <0.0001 | -6.31 | <0.0001 |
| With news vs no news | 0.12 | 0.9010 | -1.91 | 0.0558 |

The comparison below adds more tests to this family, which tightens the limit.

**Next-session forecast.** `python -m src.live` forecasts the next trading session with the frozen
no-news model and writes `artifacts/live_forecast.json`. Run it each evening after NSE publishes that
day's closing file. It uses about 60 days of Yahoo's unofficial 5-minute bars, and India VIX, turnover
and holidays from NSE, with downloads kept in `data/live/`. Nothing is retrained, and the forecast is in
no score, because it has no outcome yet. Run on the training feed, the same steps reproduce the
published later-check forecasts (`tests/test_live.py`). Yahoo's bars differ from that feed in one known
way: the day's last bar is usually NSE's official close. Applying that to later-check days moved the
forecast by a median of 0.003 percentage points and at most 0.08. The two feeds have no dates in common,
so they haven't been compared directly.

**Later check: 705 forecasts for 2023-07-04 to 2026-05-14.** This is a separate check, kept in
`artifacts/forecasts_later.csv`. The with-news model is excluded because the news archive ends on
2023-06-30, and news is never filled in. VaR should breach about 7.0 times.

| Model | MSE | QLIKE | VaR breaches | Kupiec LR |
|---|---|---|---|---|
| Transformer, no news | 0.3559 | 0.2619 | 24 | 25.32, rejected |
| HAR-RV | 0.3953 | 0.2834 | 28 | 35.97, rejected |
| Naive | 0.5656 | 0.3885 | 34 | 54.14, rejected |
| VIX-implied | 1.5451 | 0.4999 | 6 | 0.17, not rejected |

Against HAR-RV, the no-news Transformer's statistics are -3.56 on MSE (p = 0.0004) and -1.51 on
QLIKE (p = 0.1310).

### What the results do and don't show

- **The Transformer beats naive clearly and HAR-RV more modestly.** Against HAR-RV, the MSE
  result does not survive a Bonferroni limit for these six tests, while the QLIKE result does.
  It narrowly misses the stricter limit for the 14 tests in the comparison below. The later check
  also favours it on MSE, but not significantly on QLIKE.
- **Averaging seeds produces much of the lead.** Single-seed test MSEs are 0.278, 0.310 and 0.264
  with news, and 0.288, 0.264 and 0.244 without. Only three of the six beat HAR-RV's 0.2705.
- **News adds no measurable accuracy.** The MSE difference is nil (p = 0.90) and the QLIKE gain
  misses the 5% level (p = 0.056).
- **Every RV-based VaR breaches about four times too often.** This comes from how the brief's
  VaR is built, not from one model. The mean squared open-to-close return is 1.31 times mean RV
  in the test period, partly because RV leaves out the move from the open to the first 5-minute
  close. Using exp(forecast) as the variance also understates its mean. With RV known exactly,
  the same rule breaches 6 times, which is on target, so the rule is sound on the right scale.
- **VIX-implied is biased high by 1.17 log points.** VIX covers overnight moves and carries a
  risk premium, while RV covers only the session. Its VaR is too cautious rather than too low.
- **Attention is close to uniform.** Mean weights run from about 0.04 to 0.05 against a uniform
  0.045, so they don't show which days matter.

### Comparison with GARCH, an MLP and an LSTM

These three were added after the results above, without changing them or any Transformer forecast.
`artifacts/comparison.md` has the full verdict. All were trained on the same train and validation
periods, and their checkpoints were checksummed before any test forecast was made
(`artifacts/frozen_comparison_models.json`). `python run_all.py` runs them after the original
pipeline, and each has its own entry point: `python -m src.models.garch`, `python -m src.models.mlp`,
`python -m src.models.lstm` and `python -m src.eval.compare`. Together they add 46.4s of training on a
laptop CPU.

| Model | MSE | MSE, seed mean ± spread | QLIKE | QLIKE, seed mean ± spread | Parameters | Training | VaR breaches | Kupiec LR |
|---|---|---|---|---|---|---|---|---|
| Naive | 0.3609 | n/a | 0.2042 | n/a | 0 | none | 26 | 35.92, rejected |
| VIX-implied | 1.6748 | n/a | 0.5352 | n/a | 0 | none | 0 | 12.36, rejected |
| HAR-RV | 0.2705 | n/a | 0.1571 | n/a | 4 | under 0.001s | 26 | 35.92, rejected |
| GARCH(1,1) | 0.5443 | n/a | 0.2144 | n/a | 3 | 0.3s | 8 | 0.51, not rejected |
| MLP | 0.2536 | 0.2782 ± 0.0122 | 0.1438 | 0.1577 ± 0.0124 | 7,937 | 0.7s | 25 | 33.01, rejected |
| LSTM | 0.2485 | 0.2581 ± 0.0108 | 0.1434 | 0.1493 ± 0.0146 | 6,561 | 45.4s | 24 | 30.19, rejected |
| Transformer, no news | 0.2453 | 0.2654 ± 0.0220 | 0.1437 | 0.1562 ± 0.0188 | 9,665 | 31.2s | 28 | 41.98, rejected |
| Transformer, with news | 0.2463 | 0.2841 ± 0.0236 | 0.1338 | 0.1560 ± 0.0133 | 9,761 | 37.2s | 24 | 30.19, rejected |

**The with-news Transformer against each model.** Diebold-Mariano tests with 5 Newey-West lags. A
negative statistic means the Transformer had the lower loss. These 14 tests have a Bonferroni limit of
0.0036.

| Against | MSE statistic | MSE p | QLIKE statistic | QLIKE p |
|---|---|---|---|---|
| Naive | -5.89 | <0.0001 | -6.31 | <0.0001 |
| VIX-implied | -18.55 | <0.0001 | -15.91 | <0.0001 |
| HAR-RV | -2.24 | 0.0250 | -2.89 | 0.0038 |
| GARCH(1,1) | -9.38 | <0.0001 | -6.08 | <0.0001 |
| MLP | -0.70 | 0.4861 | -1.44 | 0.1503 |
| LSTM | -0.26 | 0.7959 | -1.84 | 0.0657 |
| Transformer, no news | 0.12 | 0.9010 | -1.91 | 0.0558 |

- **The extra complexity isn't shown to pay off.** The with-news Transformer ranks 2nd on MSE and 1st on
  QLIKE, but its leads over the MLP, the LSTM and the no-news Transformer aren't statistically
  significant. Scored one seed at a time, the LSTM has the lowest mean MSE and QLIKE.
- **The lead over HAR-RV is weaker than it first looked.** With 14 tests in the family, QLIKE's p of 0.0038
  narrowly misses the limit of 0.0036, so neither score passes after adjustment. The lead also comes mainly
  from a few sessions: the 10 best days supply 59% of it. Only the leads over naive, VIX-implied and GARCH
  hold up on both scores.
- **GARCH is the only model to pass the VaR test, but it's a poor RV forecaster.** It models the variance of
  the open-to-close return, which averages 1.31 times RV, so its forecasts sit 0.472 log units above log RV
  on average. That helps its VaR and hurts its MSE, so its pass doesn't make it the better forecaster.
- **Calm and volatile periods.** Sorted by the 22-day volatility known in advance, the Transformer's QLIKE
  beats HAR-RV's most clearly in the volatile third (p = 0.0003), and the MLP has the best QLIKE in the calm
  third. "Volatile" here tops out at 15% annualised, because the 2020 crash was in the validation period, so
  this says little about a crisis.
- **The training settings differ.** The MLP and LSTM use the settings the comparison brief gave, untuned
  (learning rate 0.001, weight decay 0.0001, no warm-up), while the Transformer's were chosen on validation
  data. So this tests them as specified, not at their best. They use the project's 18 features, so the MLP
  has 90 inputs rather than 85.
- **The later check isn't extended,** so nothing here says how the three new models did after June 2023.

### Departures from the brief and choices to disclose

- 18 and 15 input features rather than 17 and 14, because the weekday one-hot has 5 columns.
- **Last-day pooling replaced mean pooling after validation results.** Mean pooling scored 0.73
  and 0.76 on validation MSE against 0.41 for HAR-RV. Last-day pooling scored 0.39 and 0.40. The
  test set had not been scored at that point, but validation had guided the choice, so its
  validation numbers are slightly optimistic. Input clipping and a log Parkinson feature were
  also tried on validation and rejected, and removing international headlines was tried and not
  adopted.
- Pre-2019 Thursdays are flagged as expiries by project choice, and they aren't all real ones.
  The flag almost duplicates the Wednesday column before 2025.
- An early data audit printed full-sample correlations and yearly volatility that included test
  rows. Nothing was tuned from them.
- The sentiment classifier scored 0.806 accuracy and 0.748 macro F1 on a PhraseBank holdout that
  also chose its stopping epoch. Its accuracy on Indian news headlines was not measured.

## Output files

In `forecasts.csv`, `date` is the day t the forecast was made and `target_date` is the session it
is for. `actual_log_rv`, `return_oc` and the breach flag all describe `target_date`. The file has
every split, so train rows are in-sample and only test rows are scored.

`day_details/<date>.json` uses the same `date`. Its `expected_daily_move_pct` is one standard
deviation of the open-to-close return. That figure and `annualised_vol_pct` cover the session only,
so they run below close-to-close volatility and the VIX. Only `last_day_row` in the attention
matrix feeds the forecast directly. 3 of the 615 test days have no headlines in the archive.

`var_series.csv` holds the VaR series for every model on test dates, including GARCH, the MLP and
the LSTM.

`forecasts.csv` has `garch`, `mlp` and `lstm` columns, plus `var99_` and `breach_` for each, and
`metrics.json` has their scores under `test.models` and the comparison under `comparison`.
`comparison.md` is the written verdict, `comparison_models/` holds their weights,
`comparison_training.json` their training records, and `figures/rolling_qlike.png` and
`figures/model_ladder.png` the two plots. `day_details/` and `forecasts_later.csv` don't cover them.
