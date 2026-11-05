# Nifty Volatility Forecaster: frontend

A single-page replay of the model's next-day volatility forecasts. The test data ends in June 2023, so nothing here is live: you pick a test date and see what the frozen model forecast for the next session and what actually happened.

Built with React 18, Vite, TypeScript, TanStack Query, lightweight-charts, d3 and CSS modules.

## Run it

You need Node 20 or newer. Commands run from `frontend/` unless noted.

```
npm install
```

**With the real backend**, in two terminals:

```
# from the project root
.venv\Scripts\python -m uvicorn src.api.main:app --port 8000

# from frontend/
npm run dev
```

Then open http://localhost:5173. The backend only accepts requests from port 5173, so the dev server and `npm run preview` both refuse to start on another port.

**Without a backend**, using fixtures:

```
npm run dev:mock
```

This sets `VITE_USE_MOCKS=true` from `.env.mock`. A banner on every view says the numbers are made up.

## Commands

| Command | What it does |
|---|---|
| `npm run dev` | Dev server against the backend |
| `npm run dev:mock` | Dev server against the fixtures |
| `npm test` | Unit and component tests |
| `npm run typecheck` | TypeScript check only |
| `npm run build` | Type-checks, then builds to `dist/` |
| `npm run preview` | Serves `dist/` on port 5173 |
| `npm run fixtures` | Regenerates the fixtures from their fixed seed |

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | Where the backend is |
| `VITE_USE_MOCKS` | unset | `true` uses the fixtures. Mock code is left out of a normal build |

## The page

| Section | Component folder | What it shows |
|---|---|---|
| Next session | `components/hero/NextSession.tsx` | The live forecast from `/api/live`, with its error range, other views and caveats. It's marked out of date once its session closes. |
| Replay | `components/hero` | The forecast, what happened and India VIX's view for the selected date, with the date scrubber. Runs through the test period and the later check |
| Forecast vs actual | `components/forecast` | Annualised volatility for each model, with toggles and range presets. GARCH, the MLP and the LSTM start off, and are unavailable on the later check |
| Risk desk | `components/risk` | Daily returns against the 99% VaR line, breaches and the Kupiec result |
| Why this forecast | `components/why` | Attention strip and heatmap, headlines, and inputs vs their train-period average |
| Scoreboard | `components/scoreboard` | Test scores for all 8 models, Diebold-Mariano p-values, the model ladder, the Transformer's tests against every other model, the news ablation and the later check |
| Under the hood | `components/hood` | Architecture diagram, parameter count, training time, gradient checks and classifier accuracy |

The selected date lives in the URL as `?date=YYYY-MM-DD`. A date outside the test set moves to the next test day, and the page says so. Left and right arrow keys step one trading day, except while focus is in a form control or a scrollable table.

## Restyling

Every visual decision is a custom property in `src/styles/tokens.css`: colours, type, spacing, radii, chart colours and sizes. Components contain no colours or raw sizes, so a design pass can restyle the app from that one file. Canvas charts and d3 scales read the same tokens when they draw, through `useTokenReader`, and redraw when the theme changes.

The look is "The Volatility Desk": a business daily's markets page by day, on salmon newsprint, and the same desk's terminal by night.

- **Themes.** Day values are on `:root`, and night values are a separate block in the same file. The toggle in the masthead offers Auto, Day and Night. Auto follows the system setting, and a choice is remembered under `nv-theme`. An inline script in `index.html` sets the theme before first paint.
- **Fonts.** Fraunces for headlines, IBM Plex Sans for body text and IBM Plex Mono for every number. They are self-hosted through `@fontsource` packages and imported in `src/main.tsx`, so the page makes no third-party font requests.
- **Chart colours.** Ink for actuals, the accent for the Transformer and greys for references. The greys share a colour, so they're told apart by dash: HAR-RV dashed, Naive and VIX-implied dotted, at different weights. Change a series' dash in `CHART_SERIES` in `src/components/forecast/SeriesToggles.tsx`.
- **Column rules.** Above 75rem, `App.tsx` draws 11 hairlines behind the page, in the gaps of the 12-column grid. Tables sit on paper so that no rule runs through a digit.
- **Contrast.** `src/styles/tokens.test.ts` checks the tokens in both themes: 4.5:1 for text and 3:1 for chart marks. Run it after changing a colour.

## Reading the data

- **Two dates per row.** `date` is the day a forecast was made. `target_date` is the session it's for, and `actual_log_rv`, `return_oc` and the breach flag all describe `target_date`. Charts draw on `target_date`. Selection and `?date=` use `date`.
- **Nifty close.** `nifty_close` is NSE's published close on `date`, so it's the level known when the forecast was made. It's null in synthetic runs and in the fixtures, and the ticker then leaves it out.
- **One conversion.** Forecasts are log realized variance. `src/lib/volatility.ts` holds the only conversions: annualised volatility % is `sqrt(exp(x) * 252) * 100`, and expected daily move % is `sqrt(exp(x)) * 100`.
- **Responses are typed against the backend.** `src/api/types.ts` matches what `src/api/main.py` actually returns.

## Differences from the original spec

The UI follows what the backend actually returns, so these differ from the spec:

- **Architecture diagram:** it shows 22×18 inputs (22×15 without news) and last-day pooling, not 22×17. The weekday one-hot has 5 columns, and last-day pooling replaced mean pooling.
- **`/api/summary`:** the latest forecast is under `latest_test_forecast`, not `latest`.
- **List endpoints:** `/api/forecasts` and `/api/sentiment` wrap their rows in an object with a `rows` field, rather than returning a bare array. Forecast rows also carry `target_date`.
- **`/api/day/{date}`:** details are nested: `attention.matrix`, `attention.mean_over_input_days`, `attention.last_day_row`, `headlines.negative`, `headlines.positive`, `forecast.annualised_vol_pct` and `forecast.expected_daily_move_pct`.
- **`/api/var`:** it uses `breach_dates`, and Kupiec fields are `likelihood_ratio`, `rejected`, `expected_breaches` and `breaches`.
- **`/api/sentiment`:** rows have `log_count` rather than `count`.
- **Diebold-Mariano p-values:** the score table's tests against HAR are only the two Transformers', so other rows say "not tested". The no-news Transformer's come from the comparison, and the Scoreboard says which limit applies to which tests.
- **"The data ends June 2023":** this is true of the test window. The replay carries on through the later check to May 2026, using the no-news model and labelled as such.

These were choices or additions made while building:

- **Naive starts switched off** on the Forecast vs actual chart. It's the actual series a day late, so it hid Actual and Transformer. One click turns it on.
- **Later check:** the Scoreboard has its own table, and the replay and Forecast vs actual chart cover its days, in a range of their own. The page opens on the latest forecast. The Risk desk stays on the test period, and "Why this forecast" has nothing to show for later days, because only test days have day files.
- **`feature_train_stats`:** added to the backend's model card to supply the train-period averages, with no retraining.
- **`/api/sentiment`:** it's in the client, but no section displays it, because the spec didn't ask for one.
- **Comparison models:** GARCH, the MLP and the LSTM come through `/api/summary`, `/api/forecasts` and `/api/var` only. Reference lines stay grey, so only the Transformers wear the accent, and each line has its own width and dash so colour isn't the only cue. They only have test-period forecasts, and day files weren't extended, so `/api/day` still returns the original five models.

## Mock data

`scripts/generate-fixtures.mjs` simulates every fixture from a fixed seed. No fixture is copied from a real result, and the scores are computed from the simulated rows with the backend's formulas, so the page stays self-consistent. Don't report anything from mock mode: its numbers mean nothing.

`src/mocks/mockClient.test.ts` checks the fixtures at run time, because the compiler can't fully check imported JSON. If the backend's responses change, update the generator and `src/api/types.ts` together.

## Tests

`npm test` covers the volatility conversions, date resolution, keyboard handling, range presets, the HTTP client's error handling, error and data-source notices, winner marking on the scoreboard, and the mock client.
