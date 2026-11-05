import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { DieboldMarianoTest, ForwardRecord, ForwardRow, ForwardScores, Metrics, ModelScore, RecordModel } from "../../api/types";
import { challengerFindings, ForwardRecordContent } from "./ForwardRecord";

const names: RecordModel[] = ["transformer_nosent", "har", "naive", "vix_implied", "garch", "mlp_nosent", "lstm_nosent"];
const forecasts = Object.fromEntries(names.map((name) => [name, Math.log(0.006 ** 2)])) as Record<RecordModel, number>;

const liveRow: ForwardRow = {
  date: "2026-11-03",
  target_date: "2026-11-04",
  made_at: "2026-11-03T19:42:05+05:30",
  kind: "live",
  forecasts: { ...forecasts, garch: null, mlp_nosent: null, lstm_nosent: null },
  var99: { rule: 0.0139, calibrated: null, garch: null },
  actual_log_rv: Math.log(0.0066 ** 2),
  return_oc: 0.0095,
};
const pendingRow: ForwardRow = {
  date: "2026-11-05",
  target_date: "2026-11-06",
  made_at: "2026-11-05T19:30:00+05:30",
  kind: "live",
  forecasts,
  var99: { rule: 0.0145, calibrated: 0.021, garch: 0.019 },
  actual_log_rv: null,
  return_oc: null,
};

const scores = (n: number): ForwardScores => ({
  n_days: n,
  models: Object.fromEntries(names.map((name) => [name, { n_days: n, mse_log_rv: n ? 0.1 : null, qlike: n ? 0.05 : null }])) as ForwardScores["models"],
  var99: {
    rule: { n_days: n, breaches: 0, expected_breaches: n * 0.01 },
    calibrated: { n_days: n, breaches: 0, expected_breaches: n * 0.01 },
    garch: { n_days: n, breaches: 0, expected_breaches: n * 0.01 },
  },
});

const score = (mse: number, qlike: number, breaches: number): ModelScore => ({
  mse_log_rv: mse,
  qlike,
  mean_error_log_rv: 0,
  var99: { breaches, expected_breaches: 7.05, breach_rate: breaches / 705, kupiec_lr: 1, kupiec_rejected: false },
});

const test = (model_b: string, loss: DieboldMarianoTest["loss"], difference: number, p: number): DieboldMarianoTest => ({
  model_a: "transformer_nosent",
  model_b,
  loss,
  mean_loss_difference: difference,
  statistic: 1,
  p_value: p,
});

const record: ForwardRecord = {
  data_source: "real",
  updated_at: "2026-11-05T19:30:00+05:30",
  calibrated_multiplier: 3.38,
  rule_multiplier: 2.33,
  rows: [liveRow, pendingRow],
  scores: { live: scores(1), backfilled: scores(15) },
  challengers: {
    data_source: "real",
    later_check: {
      n_days: 705,
      target_dates: ["2023-07-04", "2026-05-14"],
      models: { garch: score(0.79, 0.33, 10), mlp_nosent: score(0.37, 0.27, 28), lstm_nosent: score(0.35, 0.24, 25) },
      diebold_mariano: [test("lstm_nosent", "qlike", 0.0255, 0.0022), test("garch", "qlike", -0.07, 0.0022), test("mlp_nosent", "qlike", 0.01, 0.2)],
      bonferroni_limit: 0.0083,
    },
    calibrated_var: { model: "transformer_nosent", multiplier: 3.38, rule_multiplier: 2.33, n_days: 705, fitted_on: ["2023-07-04", "2026-05-14"] },
  },
};
const later = { models: { transformer_nosent: score(0.36, 0.26, 24), har: score(0.4, 0.28, 28) } } as unknown as NonNullable<Metrics["later_check"]>;

describe("ForwardRecordContent", () => {
  afterEach(cleanup);

  it("shows live and backfilled rows apart, and doesn't call a session with no outcome scored", () => {
    const { container } = render(<ForwardRecordContent record={record} later={later} />);
    const text = container.textContent ?? "";
    expect(text).toContain("1 day of live forecasts and 15 days of backfilled ones");
    expect(text).toContain("Live");
    expect(text).toContain("Pending");
    expect(text).toContain("far too few to rank the models");
  });

  it("says the VaR candidates haven't been judged, and gives their figures for the next session", () => {
    const { container } = render(<ForwardRecordContent record={record} later={later} />);
    const text = container.textContent ?? "";
    expect(text).toContain("Neither candidate has been judged yet");
    expect(text).toContain("3.38 × that same forecast");
    expect(text).toContain("2.10%");
    expect(text).toContain("1.90%");
  });

  it("says a line has no days yet instead of a count of nothing", () => {
    const empty = scores(0);
    const { container } = render(<ForwardRecordContent record={{ ...record, scores: { live: empty, backfilled: scores(15) } }} later={later} />);
    expect(container.textContent).toContain("No days yet");
    expect(container.textContent).not.toContain("over 0 days");
  });

  it("states it plainly when a simpler model beats the Transformer", () => {
    const { container } = render(<ForwardRecordContent record={record} later={later} />);
    expect(container.textContent).toContain("The no-news LSTM beat the no-news Transformer on QLIKE (p 0.0022");
  });

  it("marks the best later-check score, which can be a challenger's", () => {
    const { container } = render(<ForwardRecordContent record={record} later={later} />);
    const rows = [...container.querySelectorAll("tr")].filter((item) => item.textContent?.startsWith("LSTM, no news"));
    expect(rows[rows.length - 1]?.textContent).toContain("Best: 0.2400");
  });

  it("never mentions the Kupiec test", () => {
    const { container } = render(<ForwardRecordContent record={record} later={later} />);
    expect(container.textContent?.toLowerCase()).not.toContain("kupiec");
  });
});

describe("challengerFindings", () => {
  const limit = 0.0083;

  it("only reports significant wins over the Transformer", () => {
    const findings = challengerFindings([test("lstm_nosent", "qlike", 0.02, 0.002), test("garch", "qlike", -0.07, 0.002), test("mlp_nosent", "qlike", 0.01, 0.2)], limit);
    expect(findings).toHaveLength(1);
    expect(findings[0]).toContain("no-news LSTM");
  });

  it("says nothing when no challenger has a significant win", () => {
    expect(challengerFindings([test("mlp_nosent", "qlike", 0.01, 0.2)], limit)).toEqual([]);
  });
});
