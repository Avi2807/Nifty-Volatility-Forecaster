/** Response types for the FastAPI backend, matching what it actually returns. */

export type DataSource = "real" | "synthetic" | "mock";

/** The models with a day file and a later-check score, from the original study. */
export type BaseModelName = "transformer" | "transformer_nosent" | "har" | "naive" | "vix_implied";
export type ComparisonModelName = "garch" | "mlp" | "lstm";
export type ModelName = BaseModelName | ComparisonModelName;
export type LaterModelName = Exclude<BaseModelName, "transformer">;

export interface VarScore {
  breaches: number;
  expected_breaches: number;
  breach_rate: number;
  kupiec_lr: number;
  kupiec_rejected: boolean;
}

export interface ModelScore {
  mse_log_rv: number;
  qlike: number;
  mean_error_log_rv: number;
  var99: VarScore;
}

export interface DieboldMarianoTest {
  model_a: string;
  model_b: string;
  loss: "squared_error_log_rv" | "qlike";
  mean_loss_difference: number;
  /** A negative statistic means model_a had the lower loss. */
  statistic: number;
  p_value: number;
}

export interface VarDiagnostics {
  mean_squared_return_over_mean_rv: number;
  breaches_if_rv_were_known: number;
}

export interface PeriodScores {
  forecast_dates: [string, string];
  target_dates: [string, string];
  n_days: number;
  models: Partial<Record<ModelName, ModelScore>>;
  seed_mse_log_rv: Partial<Record<ModelName, number[]>>;
  diebold_mariano: DieboldMarianoTest[];
  var_diagnostics: VarDiagnostics;
}

/** A Diebold-Mariano test of a Transformer against one other model. model_a is always the Transformer. */
export interface ComparisonTest extends DieboldMarianoTest {
  lower_loss: string;
  significant_5pct: boolean;
  /** 5% divided by the number of tests run for that Transformer. */
  bonferroni_limit: number;
  significant_bonferroni: boolean;
}

export interface SeedScores {
  mse_log_rv: number[];
  mse_mean: number;
  mse_spread: number;
  qlike: number[];
  qlike_mean: number;
  qlike_spread: number;
}

/** Every model compared on the test period. Holds more than this, such as the calm and volatile split. */
export interface ModelComparison {
  n_days: number;
  /** Simplest model first. */
  ladder: ModelName[];
  models: Record<ModelName, { n_parameters: number; training_seconds: number | null; seed_scores: SeedScores | null; mse_without_mean_error: number }>;
  ranking_mse: ModelName[];
  ranking_qlike: ModelName[];
  diebold_mariano: ComparisonTest[];
  added_training_seconds: number;
}

export interface Metrics {
  data_source: DataSource;
  test: PeriodScores;
  comparison?: ModelComparison;
  validation_mse_log_rv: Record<string, number>;
  har_coefficients: Record<string, number>;
  dm_lags: number;
  kupiec_critical_value: number;
  later_check?: PeriodScores & { excluded: string };
}

export interface GradcheckResult {
  case: string;
  array: string;
  shape: number[];
  relative_error: number;
  passed: boolean;
}

export interface ModelCard {
  data_source: DataSource;
  architecture: {
    layers: string[];
    window_days: number;
    d_model: number;
    heads: number;
    dropout: number;
    features: Record<string, string[]>;
    departures_from_brief: string[];
  };
  feature_train_stats: Record<string, { mean: number; std: number }>;
  parameter_count: Record<string, number>;
  training: {
    seconds_all_models: number;
    seeds: number[];
    batch_size: number;
    learning_rate: number;
    weight_decay: number;
    warmup_epochs: number;
    clip_norm: number;
    patience: number;
    max_epochs: number;
    best_epochs: Record<string, number[]>;
  };
  gradcheck: {
    checks: number;
    passed: number;
    worst_relative_error: number;
    tolerance: number;
    results: GradcheckResult[];
  };
  sentiment_classifier: {
    holdout_accuracy: number;
    holdout_macro_f1: number;
    vocabulary_size: number;
    note: string;
  };
}

export interface DayForecast {
  log_rv: number;
  annualised_vol_pct: number;
  /** One standard deviation of the open-to-close return. */
  expected_daily_move_pct: number;
  var99_pct: number;
  all_models_log_rv: Record<BaseModelName, number>;
}

export interface DayActual {
  log_rv: number;
  annualised_vol_pct: number;
  daily_move_pct: number;
  return_oc_pct: number;
  var_breached: boolean;
}

export interface SummaryResponse {
  data_source: DataSource;
  metrics: Metrics;
  model_card: ModelCard;
  latest_test_forecast: { date: string; target_date: string; forecast: DayForecast; actual: DayActual };
}

/** `date` is the day the forecast was made. Everything measured belongs to `target_date`. */
export interface ForecastRow {
  date: string;
  actual_log_rv: number;
  transformer: number;
  transformer_nosent: number;
  har: number;
  naive: number;
  vix_implied: number;
  return_oc: number;
  var99_transformer: number;
  breach_transformer: 0 | 1;
  garch: number;
  mlp: number;
  lstm: number;
  split: "train" | "val" | "test";
  target_date: string;
  /** NSE's published close on `date`. Null when the run has no NSE price files. */
  nifty_close: number | null;
}

/** A later-check row. The with-news Transformer isn't in these, because the news archive had ended. */
export interface LaterRow {
  date: string;
  actual_log_rv: number;
  transformer_nosent: number;
  har: number;
  naive: number;
  vix_implied: number;
  return_oc: number;
  var99_transformer_nosent: number;
  breach_transformer_nosent: 0 | 1;
  split: "later";
  target_date: string;
  nifty_close: number | null;
}

export interface LaterResponse {
  data_source: DataSource;
  models: LaterModelName[];
  count: number;
  summary: NonNullable<Metrics["later_check"]>;
  rows: LaterRow[];
}

/** A forecast for a session that hasn't happened, so it has no outcome and is in no score. */
export interface LiveForecast {
  data_source: DataSource;
  made_at: string;
  model: "transformer_nosent";
  date: string;
  target_date: string;
  forecast: number;
  seed_forecasts: number[];
  references: { har: number; naive: number; vix_implied: number };
  features: Record<string, number> & { log_turnover_ratio: number; expiry_next: number };
  first_session: string;
  sessions_used: number;
  var99: number;
  nifty_close: number;
  india_vix: number;
  turnover_crore: number;
  track_record: {
    n_days: number;
    first_date: string;
    last_date: string;
    var99_breaches: number;
    var99_expected_breaches: number;
    var99_rejected: boolean;
    error_band_share: number;
    error_band_low: number;
    error_band_high: number;
  };
  sources: { bars: string; closing_file: string; holidays: string };
}

export interface ForecastsResponse {
  data_source: DataSource;
  models: ModelName[];
  count: number;
  rows: ForecastRow[];
}

export interface ScoredHeadline {
  headline: string;
  score: number;
}

export interface DayDetails {
  date: string;
  target_date: string;
  data_source: DataSource;
  features: Record<string, number>;
  input_dates: string[];
  attention: {
    /** Row = the day paying attention, column = the day attended to, oldest first. */
    matrix: number[][];
    mean_over_input_days: number[];
    /** The only row that feeds the forecast directly, because the model pools on the last day. */
    last_day_row: number[];
    note: string;
  };
  headlines: { negative: ScoredHeadline[]; positive: ScoredHeadline[] };
  forecast: DayForecast;
  actual: DayActual;
}

export interface VarPoint {
  date: string;
  target_date: string;
  return_oc: number;
  loss: number;
  var99: number;
  breach: boolean;
}

export interface VarResponse {
  data_source: DataSource;
  model: ModelName;
  multiplier: number;
  series: VarPoint[];
  breach_dates: string[];
  kupiec: {
    breaches: number;
    expected_breaches: number;
    breach_rate: number;
    likelihood_ratio: number;
    critical_value: number;
    rejected: boolean;
  };
  diagnostics: VarDiagnostics;
}

export interface SentimentRow {
  date: string;
  mean_sentiment: number;
  negative_share: number;
  log_count: number;
}

export interface SentimentResponse {
  data_source: DataSource;
  count: number;
  rows: SentimentRow[];
}

export interface DateRange {
  start?: string;
  end?: string;
}

export interface ApiClient {
  summary(): Promise<SummaryResponse>;
  forecasts(range?: DateRange, models?: ModelName[]): Promise<ForecastsResponse>;
  later(): Promise<LaterResponse>;
  live(): Promise<LiveForecast>;
  day(date: string): Promise<DayDetails>;
  valueAtRisk(model: ModelName): Promise<VarResponse>;
  sentiment(range?: DateRange): Promise<SentimentResponse>;
}
