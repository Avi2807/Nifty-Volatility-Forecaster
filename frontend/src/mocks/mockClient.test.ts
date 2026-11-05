import { describe, expect, it } from "vitest";
import { ApiError } from "../api/client";
import { logRvToAnnualVolPct } from "../lib/volatility";
import { mockClient } from "./mockClient";

describe("mock client", () => {
  it("labels every response as mock", async () => {
    const summary = await mockClient.summary();
    const forecasts = await mockClient.forecasts();
    const day = await mockClient.day(forecasts.rows[0]!.date);
    const risk = await mockClient.valueAtRisk("har");
    const sentiment = await mockClient.sentiment();
    for (const source of [summary.data_source, summary.metrics.data_source, summary.model_card.data_source, forecasts.data_source, day.data_source, risk.data_source, sentiment.data_source]) {
      expect(source).toBe("mock");
    }
  });

  it("has fixtures with the fields the app reads", async () => {
    const { metrics, model_card: card, latest_test_forecast: latest } = await mockClient.summary();
    const { rows } = await mockClient.forecasts();
    expect(metrics.test.forecast_dates).toEqual([rows[0]!.date, rows[rows.length - 1]!.date]);
    expect(metrics.test.n_days).toBe(rows.length);
    expect(Object.keys(metrics.test.models)).toEqual(["transformer", "transformer_nosent", "har", "naive", "vix_implied", "garch", "mlp", "lstm"]);
    expect(metrics.comparison?.ladder).toHaveLength(8);
    expect(metrics.test.diebold_mariano).toHaveLength(6);
    expect(latest.date).toBe(rows[rows.length - 1]!.date);
    expect(card.architecture.features.transformer).toHaveLength(18);
    expect(card.architecture.features.transformer_nosent).toHaveLength(15);
    for (const name of card.architecture.features.transformer ?? []) expect(card.feature_train_stats[name]).toBeDefined();
    expect(rows.every((row) => row.split === "test" && row.target_date > row.date)).toBe(true);
  });

  it("has later-check rows that follow on from the test period and agree with the later scores", async () => {
    const { rows: test } = await mockClient.forecasts();
    const later = await mockClient.later();
    expect(later.data_source).toBe("mock");
    expect(later.rows[0]!.date > test[test.length - 1]!.date).toBe(true);
    expect(later.rows.every((row) => row.split === "later" && row.target_date > row.date)).toBe(true);
    const breaches = later.rows.reduce((total, row) => total + row.breach_transformer_nosent, 0);
    expect(breaches).toBe(later.summary.models.transformer_nosent!.var99.breaches);
  });

  it("builds a day that agrees with its forecast row", async () => {
    const { rows } = await mockClient.forecasts();
    const row = rows[100]!;
    const day = await mockClient.day(row.date);
    expect(day.target_date).toBe(row.target_date);
    expect(day.forecast.annualised_vol_pct).toBeCloseTo(logRvToAnnualVolPct(row.transformer), 10);
    expect(day.input_dates).toHaveLength(22);
    expect(day.input_dates[21]).toBe(row.date);
    expect(day.attention.matrix).toHaveLength(22);
    for (const weights of day.attention.matrix) {
      expect(weights.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 10);
    }
  });

  it("keeps a VaR series consistent with its Kupiec count", async () => {
    for (const model of ["transformer", "har", "vix_implied", "garch", "mlp", "lstm"] as const) {
      const risk = await mockClient.valueAtRisk(model);
      expect(risk.series.filter((point) => point.breach)).toHaveLength(risk.kupiec.breaches);
      expect(risk.breach_dates).toHaveLength(risk.kupiec.breaches);
    }
  });

  it("refuses a date outside the test set, like the backend", async () => {
    const error = (await mockClient.day("2020-03-23").catch((caught: unknown) => caught)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(404);
  });
});
