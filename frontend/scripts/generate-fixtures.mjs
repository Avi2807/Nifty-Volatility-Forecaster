/**
 * Writes the mock-mode fixtures in src/mocks/fixtures. Run with: npm run fixtures
 *
 * Every number here is simulated from a fixed seed. Nothing comes from the real data or model,
 * so mock results can't be mistaken for or mixed with real ones. Scores are computed from the
 * simulated rows with the same formulas as the backend, so the page stays self-consistent.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "src", "mocks", "fixtures");
mkdirSync(OUT, { recursive: true });

// A small seeded generator (mulberry32), so the fixtures are the same on every run.
let state = 20240115;
const uniform = () => {
  state = (state + 0x6d2b79f5) | 0;
  let t = Math.imul(state ^ (state >>> 15), 1 | state);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const normal = () => Math.sqrt(-2 * Math.log(uniform() || 1e-12)) * Math.cos(2 * Math.PI * uniform());
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;

function weekdays(from, to) {
  const days = [];
  for (let day = new Date(`${from}T00:00:00Z`); day <= new Date(`${to}T00:00:00Z`); day.setUTCDate(day.getUTCDate() + 1)) {
    if (day.getUTCDay() % 6 !== 0) days.push(day.toISOString().slice(0, 10));
  }
  return days;
}

const sessions = weekdays("2021-01-01", "2023-07-03");
const level = (index) => -10.3 + 0.55 * Math.sin(index / 70) + 0.25 * Math.sin(index / 19);

// Simulated log realized variance for each session.
const logRv = [];
let persistent = 0;
sessions.forEach((_, index) => {
  persistent = 0.8 * persistent + 0.28 * normal();
  logRv.push(level(index) + persistent + 0.32 * normal());
});

const rows = [];
for (let index = 0; index < sessions.length - 1; index += 1) {
  const today = logRv[index];
  const week = mean(logRv.slice(Math.max(0, index - 4), index + 1));
  const month = mean(logRv.slice(Math.max(0, index - 21), index + 1));
  const actual = logRv[index + 1];
  const har = -1.3 + 0.3 * today + 0.35 * week + 0.22 * month;
  const transformer = 0.75 * har + 0.25 * level(index + 1) + 0.1 * normal();
  const nosent = 0.75 * har + 0.25 * level(index + 1) + 0.1 * normal();
  const vix = level(index + 1) + 1.15 + 0.25 * normal();
  const garch = 0.55 * today + 0.45 * level(index + 1) + 0.4 + 0.25 * normal();
  const mlp = 0.75 * har + 0.25 * level(index + 1) + 0.16 * normal();
  const lstm = 0.75 * har + 0.25 * level(index + 1) + 0.13 * normal();
  const returnOc = Math.exp(actual / 2) * 1.14 * (normal() * 0.9 + 0.1 * normal() * 3);
  const var99 = 2.33 * Math.exp(transformer / 2);
  rows.push({
    date: sessions[index],
    actual_log_rv: actual,
    transformer,
    transformer_nosent: nosent,
    har,
    naive: today,
    vix_implied: vix,
    return_oc: returnOc,
    var99_transformer: var99,
    breach_transformer: -returnOc > var99 ? 1 : 0,
    garch,
    mlp,
    lstm,
    split: "test",
    target_date: sessions[index + 1],
    nifty_close: null, // the simulation has no index level, as in a synthetic run
  });
}

// The original study's models have day files and later-check scores. The comparison models don't.
const BASE_MODELS = ["transformer", "transformer_nosent", "har", "naive", "vix_implied"];
const MODELS = [...BASE_MODELS, "garch", "mlp", "lstm"];
const LADDER = ["naive", "vix_implied", "har", "garch", "mlp", "lstm", "transformer_nosent", "transformer"];
const squaredError = (row, model) => (row.actual_log_rv - row[model]) ** 2;
const qlike = (row, model) => {
  const logRatio = row.actual_log_rv - row[model];
  return Math.exp(logRatio) - logRatio - 1;
};

function kupiec(breaches, n, rate = 0.01) {
  const logLikelihood = (p) => (breaches > 0 ? breaches * Math.log(p) : 0) + (n - breaches > 0 ? (n - breaches) * Math.log(1 - p) : 0);
  return 2 * (logLikelihood(breaches / n) - logLikelihood(rate));
}

function dieboldMariano(a, b, lossFunction, lossName, period) {
  const difference = period.map((row) => lossFunction(row, a) - lossFunction(row, b));
  const average = mean(difference);
  const centred = difference.map((value) => value - average);
  const n = centred.length;
  let variance = centred.reduce((sum, value) => sum + value * value, 0) / n;
  for (let lag = 1; lag <= 5; lag += 1) {
    let product = 0;
    for (let t = lag; t < n; t += 1) product += centred[t] * centred[t - lag];
    variance += (2 * (1 - lag / 6) * product) / n;
  }
  const statistic = average / Math.sqrt(variance / n);
  // Two-sided normal p-value, by a standard series approximation of the normal tail.
  const z = Math.abs(statistic);
  const tail = (Math.exp((-z * z) / 2) / Math.sqrt(2 * Math.PI)) * (1 / (z + 1 / (z + 2 / (z + 3 / (z + 4 / (z + 5))))));
  return { model_a: a, model_b: b, loss: lossName, mean_loss_difference: average, statistic, p_value: Math.min(1, 2 * tail) };
}

function periodScores(period, models, pairs) {
  const scoreFor = (model) => {
    const breaches = period.filter((row) => -row.return_oc > 2.33 * Math.exp(row[model] / 2)).length;
    const lr = kupiec(breaches, period.length);
    return {
      mse_log_rv: mean(period.map((row) => squaredError(row, model))),
      qlike: mean(period.map((row) => qlike(row, model))),
      mean_error_log_rv: mean(period.map((row) => row[model] - row.actual_log_rv)),
      var99: {
        breaches,
        expected_breaches: period.length * 0.01,
        breach_rate: breaches / period.length,
        kupiec_lr: lr,
        kupiec_rejected: lr > 3.84,
      },
    };
  };
  const seed = (value, spread) => [value * (1 + spread), value * (1 + spread * 1.6), value * (1 - spread * 0.2)];
  const scores = Object.fromEntries(models.map((model) => [model, scoreFor(model)]));
  return {
    forecast_dates: [period[0].date, period[period.length - 1].date],
    target_dates: [period[0].target_date, period[period.length - 1].target_date],
    n_days: period.length,
    models: scores,
    seed_mse_log_rv: Object.fromEntries(models.filter((model) => model.startsWith("transformer")).map((model) => [model, seed(scores[model].mse_log_rv, 0.08)])),
    diebold_mariano: pairs.flatMap(([a, b]) => [
      dieboldMariano(a, b, squaredError, "squared_error_log_rv", period),
      dieboldMariano(a, b, qlike, "qlike", period),
    ]),
    var_diagnostics: {
      mean_squared_return_over_mean_rv: mean(period.map((row) => row.return_oc ** 2)) / mean(period.map((row) => Math.exp(row.actual_log_rv))),
      breaches_if_rv_were_known: period.filter((row) => -row.return_oc > 2.33 * Math.exp(row.actual_log_rv / 2)).length,
    },
  };
}

const allFeatures = {
  transformer: [
    "log_rv", "log_rv_w", "log_rv_m", "parkinson", "ret", "ret_neg", "gap", "log_vix", "log_turnover_ratio", "expiry_next",
    "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "news_mean_sentiment", "news_negative_share", "news_log_count",
  ],
};
allFeatures.transformer_nosent = allFeatures.transformer.slice(0, 15);

const featureTrainStats = {
  log_rv: { mean: -10.4, std: 0.75 },
  log_rv_w: { mean: -10.4, std: 0.55 },
  log_rv_m: { mean: -10.4, std: 0.45 },
  parkinson: { mean: 0.00006, std: 0.00007 },
  ret: { mean: 0.0003, std: 0.009 },
  ret_neg: { mean: -0.0028, std: 0.005 },
  gap: { mean: 0.0002, std: 0.004 },
  log_vix: { mean: 2.75, std: 0.25 },
  log_turnover_ratio: { mean: 0.0, std: 0.22 },
  expiry_next: { mean: 0.2, std: 0.4 },
  dow_mon: { mean: 0.2, std: 0.4 },
  dow_tue: { mean: 0.2, std: 0.4 },
  dow_wed: { mean: 0.2, std: 0.4 },
  dow_thu: { mean: 0.2, std: 0.4 },
  dow_fri: { mean: 0.2, std: 0.4 },
  news_mean_sentiment: { mean: 0.12, std: 0.1 },
  news_negative_share: { mean: 0.15, std: 0.08 },
  news_log_count: { mean: 3.3, std: 0.7 },
};

const gradcheckCases = ["ReLU", "LSTMCell (2 in, 3 hidden)", "LSTM (2 in, 3 hid, T=4)", "Dense", "GELU", "Dropout (training)", "LayerNorm", "Softmax", "ScaledDotProductAttention", "MultiHeadAttention", "PositionalEmbedding", "MeanPool", "LastDayPool", "MSELoss", "EncoderBlock (T=22, d=32)"];
const gradcheckResults = gradcheckCases.map((name) => ({ case: name, array: "input", shape: [2, 4, 8], relative_error: 1e-10 + uniform() * 1e-9, passed: true }));

const summary = {
  data_source: "mock",
  metrics: {
    data_source: "mock",
    test: periodScores(rows, MODELS, [["transformer", "har"], ["transformer", "naive"], ["transformer", "transformer_nosent"]]),
    validation_mse_log_rv: { transformer: 0.4, transformer_nosent: 0.41, har: 0.42, naive: 0.43 },
    har_coefficients: { intercept: -1.3, log_rv: 0.3, log_rv_w: 0.35, log_rv_m: 0.22 },
    dm_lags: 5,
    kupiec_critical_value: 3.84,
  },
  model_card: {
    data_source: "mock",
    architecture: {
      layers: ["Dense(n_features -> 32)", "Learned positional embedding (22 x 32)", "Pre-norm Transformer encoder block (d=32, 2 heads, FFN 64, dropout 0.1)", "Last-day pooling", "Dense(32 -> 1)"],
      window_days: 22,
      d_model: 32,
      heads: 2,
      dropout: 0.1,
      features: allFeatures,
      departures_from_brief: ["Mock text: this list shows the real departures when the backend is running."],
    },
    feature_train_stats: featureTrainStats,
    parameter_count: { transformer: 9761, transformer_nosent: 9665, garch: 3, mlp: 7937, lstm: 6561 },
    training: {
      seconds_all_models: 70,
      seeds: [1, 2, 3],
      batch_size: 32,
      learning_rate: 0.0005,
      weight_decay: 0.001,
      warmup_epochs: 5,
      clip_norm: 1,
      patience: 15,
      max_epochs: 200,
      best_epochs: { transformer: [20, 30, 15], transformer_nosent: [25, 18, 16] },
    },
    gradcheck: { checks: gradcheckResults.length, passed: gradcheckResults.length, worst_relative_error: Math.max(...gradcheckResults.map((row) => row.relative_error)), tolerance: 1e-6, results: gradcheckResults },
    sentiment_classifier: { holdout_accuracy: 0.8, holdout_macro_f1: 0.75, vocabulary_size: 6000, note: "Mock value, not a measured result." },
  },
};

// The comparison block, computed from the simulated rows with the same formulas as the backend.
const comparisonTests = (base, others) =>
  others.flatMap((other) =>
    [
      dieboldMariano(base, other, squaredError, "squared_error_log_rv", rows),
      dieboldMariano(base, other, qlike, "qlike", rows),
    ].map((test) => {
      const limit = 0.05 / (others.length * 2);
      return { ...test, lower_loss: test.mean_loss_difference < 0 ? base : other, significant_5pct: test.p_value < 0.05, bonferroni_limit: limit, significant_bonferroni: test.p_value < limit };
    }),
  );
const testScores = summary.metrics.test.models;
const rankedBy = (key) => [...LADDER].sort((a, b) => testScores[a][key] - testScores[b][key]);
const seedScores = (model) => {
  const spreadOut = (value) => [value * 1.1, value * 1.14, value * 1.04];
  const spreadOf = (values) => {
    const average = mean(values);
    return Math.sqrt(values.reduce((sum, value) => sum + (value - average) ** 2, 0) / (values.length - 1));
  };
  const mse = spreadOut(testScores[model].mse_log_rv);
  const qlikes = spreadOut(testScores[model].qlike);
  return { mse_log_rv: mse, mse_mean: mean(mse), mse_spread: spreadOf(mse), qlike: qlikes, qlike_mean: mean(qlikes), qlike_spread: spreadOf(qlikes) };
};
const COSTS = {
  naive: [0, null], vix_implied: [0, null], har: [4, 0.001], garch: [3, 0.3], mlp: [7937, 0.7], lstm: [6561, 45], transformer_nosent: [9665, 31], transformer: [9761, 37],
};
summary.metrics.comparison = {
  n_days: rows.length,
  ladder: LADDER,
  models: Object.fromEntries(LADDER.map((model) => [model, {
    n_parameters: COSTS[model][0],
    training_seconds: COSTS[model][1],
    seed_scores: ["mlp", "lstm", "transformer_nosent", "transformer"].includes(model) ? seedScores(model) : null,
    mse_without_mean_error: testScores[model].mse_log_rv - testScores[model].mean_error_log_rv ** 2,
  }])),
  ranking_mse: rankedBy("mse_log_rv"),
  ranking_qlike: rankedBy("qlike"),
  diebold_mariano: [
    ...comparisonTests("transformer", LADDER.filter((model) => model !== "transformer")),
    ...comparisonTests("transformer_nosent", LADDER.filter((model) => !model.startsWith("transformer"))),
  ],
  added_training_seconds: 46,
};

const lastRow = rows[rows.length - 1];
const annualVol = (logValue) => Math.sqrt(Math.exp(logValue) * 252) * 100;
const dailyMove = (logValue) => Math.sqrt(Math.exp(logValue)) * 100;
summary.latest_test_forecast = {
  date: lastRow.date,
  target_date: lastRow.target_date,
  forecast: {
    log_rv: lastRow.transformer,
    annualised_vol_pct: annualVol(lastRow.transformer),
    expected_daily_move_pct: dailyMove(lastRow.transformer),
    var99_pct: lastRow.var99_transformer * 100,
    all_models_log_rv: Object.fromEntries(BASE_MODELS.map((model) => [model, lastRow[model]])),
  },
  actual: {
    log_rv: lastRow.actual_log_rv,
    annualised_vol_pct: annualVol(lastRow.actual_log_rv),
    daily_move_pct: dailyMove(lastRow.actual_log_rv),
    return_oc_pct: lastRow.return_oc * 100,
    var_breached: lastRow.breach_transformer === 1,
  },
};

const later = weekdays("2023-07-03", "2023-12-29");
const laterLogRv = later.map((_, index) => level(index + 640) + 0.5 * normal());
const laterRows = later.slice(0, -1).map((date, index) => {
  const har = -1.3 + 0.87 * laterLogRv[index];
  return {
    date,
    actual_log_rv: laterLogRv[index + 1],
    transformer_nosent: har + 0.1 * normal(),
    har,
    naive: laterLogRv[index],
    vix_implied: level(index + 640) + 1.15,
    return_oc: Math.exp(laterLogRv[index + 1] / 2) * normal(),
    split: "later",
    target_date: later[index + 1],
    nifty_close: null,
  };
});
for (const row of laterRows) {
  row.var99_transformer_nosent = 2.33 * Math.sqrt(Math.exp(row.transformer_nosent));
  row.breach_transformer_nosent = -row.return_oc > row.var99_transformer_nosent ? 1 : 0;
}
summary.metrics.later_check = {
  ...periodScores(laterRows, BASE_MODELS.slice(1), [["transformer_nosent", "har"], ["transformer_nosent", "naive"]]),
  excluded: "The with-sentiment Transformer: the news archive ends before this period.",
};

const sentimentRows = rows.map((row) => ({
  date: row.date,
  mean_sentiment: 0.12 + 0.08 * normal(),
  negative_share: Math.max(0, 0.15 + 0.07 * normal()),
  log_count: Math.max(0.7, 2.8 + 0.4 * normal()),
}));

const headlines = {
  negative: [
    "Mock: Rupee slips against the dollar in early trade", "Mock: Oil prices climb on supply worries", "Mock: Lender reports a wider quarterly loss",
    "Mock: Bank stocks fall after rate comments", "Mock: Exporters warn of weaker orders", "Mock: Auto sales drop for a second month",
    "Mock: Inflation runs above the central bank's target", "Mock: Factory output misses forecasts",
  ],
  positive: [
    "Mock: IT major beats profit estimates", "Mock: Foreign investors return to Indian equities", "Mock: GST collections hit a new high",
    "Mock: Steelmaker raises its output target", "Mock: Retailer posts strong festive sales", "Mock: Services activity expands at a faster pace",
    "Mock: Bank's bad loans fall for a fourth quarter", "Mock: Monsoon forecast lifts farm outlook",
  ],
};

const write = (name, content) => writeFileSync(join(OUT, name), `${JSON.stringify(content)}\n`);
write("summary.json", summary);
write("forecasts.json", rows);
write("later.json", laterRows);
write("sentiment.json", sentimentRows);
write("headlines.json", headlines);
console.log(`Wrote ${rows.length} simulated test rows to ${OUT}`);
console.log("Mock Transformer MSE", summary.metrics.test.models.transformer.mse_log_rv.toFixed(3), "| breaches", summary.metrics.test.models.transformer.var99.breaches);
