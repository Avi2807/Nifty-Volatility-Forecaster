import type { ForecastRow, LaterRow } from "../api/types";

export type ReplayPeriod = "test" | "later";

/**
 * One day of the replay. `forecast`, `var99` and `breached` belong to `model`, which changes between
 * periods: the news archive ends with the test period, so later days come from the no-news Transformer.
 */
export interface ReplayRow {
  date: string;
  target_date: string;
  period: ReplayPeriod;
  model: "transformer" | "transformer_nosent";
  forecast: number;
  var99: number;
  breached: boolean;
  actual_log_rv: number;
  return_oc: number;
  transformer: number | null;
  transformer_nosent: number;
  har: number;
  naive: number;
  vix_implied: number;
  /** These three were only scored on the test period. */
  garch: number | null;
  mlp: number | null;
  lstm: number | null;
  nifty_close: number | null;
}

export function buildReplay(test: ForecastRow[], later: LaterRow[]): ReplayRow[] {
  return [
    ...test.map(
      (row): ReplayRow => ({
        ...row,
        period: "test",
        model: "transformer",
        forecast: row.transformer,
        var99: row.var99_transformer,
        breached: row.breach_transformer === 1,
      }),
    ),
    ...later.map(
      (row): ReplayRow => ({
        ...row,
        period: "later",
        model: "transformer_nosent",
        forecast: row.transformer_nosent,
        var99: row.var99_transformer_nosent,
        breached: row.breach_transformer_nosent === 1,
        transformer: null,
        garch: null,
        mlp: null,
        lstm: null,
      }),
    ),
  ];
}
