import { ApiError } from "../api/client";
import type {
  ApiClient,
  DateRange,
  DayDetails,
  ForecastRow,
  ForecastsResponse,
  LaterModelName,
  LaterRow,
  LiveForecast,
  ModelName,
  SentimentResponse,
  SentimentRow,
  SummaryResponse,
  VarPoint,
  VarResponse,
} from "../api/types";
import { logRvToAnnualVolPct, logRvToDailyMovePct } from "../lib/volatility";
import forecastFixture from "./fixtures/forecasts.json";
import headlineFixture from "./fixtures/headlines.json";
import laterFixture from "./fixtures/later.json";
import sentimentFixture from "./fixtures/sentiment.json";
import summaryFixture from "./fixtures/summary.json";

// JSON imports widen tuples and unions, so the compiler can't check these. mockClient.test.ts checks them at run time.
const summary = summaryFixture as unknown as SummaryResponse;
const rows = forecastFixture as ForecastRow[];
const laterRows = laterFixture as LaterRow[];
const LATER_MODELS: LaterModelName[] = ["transformer_nosent", "har", "naive", "vix_implied"];
const sentimentRows: SentimentRow[] = sentimentFixture;
const LATENCY_MS = 180; // long enough to see the loading states
const WINDOW = 22;

const later = <T,>(value: T): Promise<T> => new Promise((resolve) => setTimeout(() => resolve(value), LATENCY_MS));

function inRange<T extends { date: string }>(items: T[], { start, end }: DateRange): T[] {
  return items.filter((item) => (!start || item.date >= start) && (!end || item.date <= end));
}

/** A repeatable pseudo-random number for a date, so a day always looks the same. */
function seeded(date: string, salt: number): number {
  let hash = 2166136261 ^ salt;
  for (const character of date) hash = Math.imul(hash ^ character.charCodeAt(0), 16777619);
  return ((hash >>> 0) % 100000) / 100000;
}

function attentionMatrix(date: string): number[][] {
  return Array.from({ length: WINDOW }, (_, row) => {
    const raw = Array.from({ length: WINDOW }, (_, column) => Math.exp(0.5 * seeded(date, row * 31 + column) + (row === column ? 0.15 : 0)));
    const total = raw.reduce((sum, value) => sum + value, 0);
    return raw.map((value) => value / total);
  });
}

function buildDay(date: string): DayDetails {
  const position = rows.findIndex((row) => row.date === date);
  const row = rows[position];
  const first = rows[0];
  const last = rows[rows.length - 1];
  if (!row || !first || !last) {
    throw new ApiError(`${date} is not a test date. Day files exist for the ${rows.length} days from ${first?.date} to ${last?.date}.`, 404);
  }
  const matrix = attentionMatrix(date);
  const inputDates = Array.from({ length: WINDOW }, (_, index) => {
    const day = new Date(`${date}T00:00:00Z`);
    day.setUTCDate(day.getUTCDate() - (WINDOW - 1 - index));
    return rows[position - (WINDOW - 1 - index)]?.date ?? day.toISOString().slice(0, 10);
  });
  const scored = (items: string[], sign: number, salt: number) =>
    items
      .map((headline, index) => ({ headline, score: sign * (0.35 + 0.55 * seeded(date, salt + index)) }))
      .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
      .slice(0, 5);
  const noNews = seeded(date, 999) < 0.01;
  const weekday = new Date(`${date}T00:00:00Z`).getUTCDay();

  return {
    date,
    target_date: row.target_date,
    data_source: "mock",
    features: {
      log_rv: row.naive,
      log_rv_w: row.naive * 0.5 - 5.2,
      log_rv_m: -10.3,
      parkinson: Math.exp(row.naive),
      ret: (seeded(date, 1) - 0.5) * 0.03,
      ret_neg: -seeded(date, 2) * 0.012,
      gap: (seeded(date, 3) - 0.5) * 0.012,
      log_vix: Math.log(logRvToAnnualVolPct(row.vix_implied)),
      log_turnover_ratio: (seeded(date, 4) - 0.5) * 0.7,
      expiry_next: weekday === 3 ? 1 : 0,
      dow_mon: weekday === 1 ? 1 : 0,
      dow_tue: weekday === 2 ? 1 : 0,
      dow_wed: weekday === 3 ? 1 : 0,
      dow_thu: weekday === 4 ? 1 : 0,
      dow_fri: weekday === 5 ? 1 : 0,
      news_mean_sentiment: noNews ? 0 : 0.05 + seeded(date, 5) * 0.2,
      news_negative_share: noNews ? 0 : 0.05 + seeded(date, 6) * 0.25,
      news_log_count: noNews ? 0 : 2 + seeded(date, 7) * 1.5,
    },
    input_dates: inputDates,
    attention: {
      matrix,
      mean_over_input_days: matrix[0]!.map((_, column) => matrix.reduce((sum, cells) => sum + cells[column]!, 0) / WINDOW),
      last_day_row: matrix[WINDOW - 1]!,
      note: "Mock attention, not from a model.",
    },
    headlines: noNews
      ? { negative: [], positive: [] }
      : { negative: scored(headlineFixture.negative, -1, 40), positive: scored(headlineFixture.positive, 1, 60) },
    forecast: {
      log_rv: row.transformer,
      annualised_vol_pct: logRvToAnnualVolPct(row.transformer),
      expected_daily_move_pct: logRvToDailyMovePct(row.transformer),
      var99_pct: row.var99_transformer * 100,
      all_models_log_rv: { transformer: row.transformer, transformer_nosent: row.transformer_nosent, har: row.har, naive: row.naive, vix_implied: row.vix_implied },
    },
    actual: {
      log_rv: row.actual_log_rv,
      annualised_vol_pct: logRvToAnnualVolPct(row.actual_log_rv),
      daily_move_pct: logRvToDailyMovePct(row.actual_log_rv),
      return_oc_pct: row.return_oc * 100,
      var_breached: row.breach_transformer === 1,
    },
  };
}

function varResponse(model: ModelName): VarResponse {
  const series: VarPoint[] = rows.map((row) => {
    const var99 = 2.33 * Math.sqrt(Math.exp(row[model]));
    return { date: row.date, target_date: row.target_date, return_oc: row.return_oc, loss: -row.return_oc, var99, breach: -row.return_oc > var99 };
  });
  const score = summary.metrics.test.models[model];
  if (!score) throw new ApiError(`Unknown model '${model}'.`, 422);
  return {
    data_source: "mock",
    model,
    multiplier: 2.33,
    series,
    breach_dates: series.filter((point) => point.breach).map((point) => point.target_date),
    kupiec: {
      breaches: score.var99.breaches,
      expected_breaches: score.var99.expected_breaches,
      breach_rate: score.var99.breach_rate,
      likelihood_ratio: score.var99.kupiec_lr,
      critical_value: summary.metrics.kupiec_critical_value,
      rejected: score.var99.kupiec_rejected,
    },
    diagnostics: summary.metrics.test.var_diagnostics,
  };
}

/** Simulated like everything else here, from the last later-check row. Its session has passed, so it shows as out of date. */
function liveForecast(): LiveForecast {
  const last = laterRows[laterRows.length - 1]!;
  const record = summary.metrics.later_check!;
  const var99 = record.models.transformer_nosent!.var99;
  const target = new Date(`${last.target_date}T00:00:00Z`);
  target.setUTCDate(target.getUTCDate() + (target.getUTCDay() === 5 ? 3 : 1));
  return {
    data_source: "mock",
    made_at: `${last.target_date}T19:00:00+05:30`,
    model: "transformer_nosent",
    date: last.target_date,
    target_date: target.toISOString().slice(0, 10),
    forecast: last.transformer_nosent,
    seed_forecasts: [last.transformer_nosent - 0.05, last.transformer_nosent, last.transformer_nosent + 0.05],
    references: { har: last.har, naive: last.actual_log_rv, vix_implied: last.vix_implied },
    features: { log_turnover_ratio: 0.1, expiry_next: 0 },
    first_session: last.date,
    sessions_used: 60,
    var99: last.var99_transformer_nosent,
    nifty_close: last.nifty_close ?? 20000,
    india_vix: 15,
    turnover_crore: 30000,
    track_record: {
      n_days: record.n_days,
      first_date: record.forecast_dates[0],
      last_date: record.forecast_dates[1],
      var99_breaches: var99.breaches,
      var99_expected_breaches: var99.expected_breaches,
      var99_rejected: var99.kupiec_rejected,
      error_band_share: 0.8,
      error_band_low: -0.65,
      error_band_high: 0.75,
    },
    sources: { bars: "mock", closing_file: "mock", holidays: "mock" },
  };
}

/** Same shapes as the backend, including its refusal of non-test dates. Everything is labelled "mock". */
export const mockClient: ApiClient = {
  summary: () => later(summary),
  forecasts: (range: DateRange = {}): Promise<ForecastsResponse> => {
    const selected = inRange(rows, range);
    return later({ data_source: "mock", models: ["transformer", "transformer_nosent", "har", "naive", "vix_implied", "garch", "mlp", "lstm"], count: selected.length, rows: selected });
  },
  later: () => later({ data_source: "mock", models: LATER_MODELS, count: laterRows.length, summary: summary.metrics.later_check!, rows: laterRows }),
  live: () => later(liveForecast()),
  day: async (date: string) => later(buildDay(date)),
  valueAtRisk: async (model: ModelName) => later(varResponse(model)),
  sentiment: (range: DateRange = {}): Promise<SentimentResponse> => {
    const selected = inRange(sentimentRows, range);
    return later({ data_source: "mock", count: selected.length, rows: selected });
  },
};
