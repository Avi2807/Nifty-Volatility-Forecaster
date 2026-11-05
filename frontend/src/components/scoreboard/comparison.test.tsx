import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { ComparisonTest, Metrics, ModelComparison, ModelName, ModelScore, PeriodScores } from "../../api/types";
import { ALL_MODELS, MODEL_LABELS, SERIES_TOKENS } from "../../lib/models";
import { CHART_SERIES } from "../forecast/SeriesToggles";
import { ComparisonTests } from "./ComparisonTests";
import { ModelLadder } from "./ModelLadder";
import { ModelScoreTable } from "./ModelScoreTable";
import { bestSingleSeed, groupLeads, leadReading } from "./scoring";

afterEach(cleanup);

const LADDER: ModelName[] = ["naive", "vix_implied", "har", "garch", "mlp", "lstm", "transformer_nosent", "transformer"];

const test_ = (model_b: ModelName, loss: ComparisonTest["loss"], statistic: number, p_value: number, significant_bonferroni = false): ComparisonTest => ({
  model_a: "transformer",
  model_b,
  loss,
  mean_loss_difference: statistic < 0 ? -0.01 : 0.01,
  statistic,
  p_value,
  lower_loss: statistic < 0 ? "transformer" : model_b,
  significant_5pct: p_value < 0.05,
  bonferroni_limit: 0.0036,
  significant_bonferroni,
});

const seeds = (mean: number) => ({ mse_log_rv: [mean], mse_mean: mean, mse_spread: 0.01, qlike: [mean], qlike_mean: mean, qlike_spread: 0.02 });

const comparison = {
  n_days: 615,
  ladder: LADDER,
  models: {
    naive: { n_parameters: 0, training_seconds: null, seed_scores: null, mse_without_mean_error: 0.3 },
    vix_implied: { n_parameters: 0, training_seconds: null, seed_scores: null, mse_without_mean_error: 0.3 },
    har: { n_parameters: 4, training_seconds: 0.0001, seed_scores: null, mse_without_mean_error: 0.3 },
    garch: { n_parameters: 3, training_seconds: 0.3, seed_scores: null, mse_without_mean_error: 0.3 },
    mlp: { n_parameters: 7937, training_seconds: 0.7, seed_scores: seeds(0.28), mse_without_mean_error: 0.25 },
    lstm: { n_parameters: 6561, training_seconds: 45, seed_scores: seeds(0.2), mse_without_mean_error: 0.25 },
    transformer_nosent: { n_parameters: 9665, training_seconds: 31, seed_scores: seeds(0.27), mse_without_mean_error: 0.25 },
    transformer: { n_parameters: 9761, training_seconds: 37, seed_scores: seeds(0.29), mse_without_mean_error: 0.25 },
  },
  ranking_mse: ["transformer_nosent", "transformer", "lstm", "mlp", "har", "naive", "garch", "vix_implied"],
  ranking_qlike: ["transformer", "lstm", "transformer_nosent", "mlp", "har", "naive", "garch", "vix_implied"],
  diebold_mariano: [
    test_("naive", "squared_error_log_rv", -5.9, 0.00001, true),
    test_("naive", "qlike", -6.3, 0.00001, true),
    test_("vix_implied", "squared_error_log_rv", -18, 0.00001, true),
    test_("vix_implied", "qlike", -15, 0.00001, true),
    test_("har", "squared_error_log_rv", -2.2, 0.025),
    test_("har", "qlike", -2.9, 0.0038),
    test_("garch", "squared_error_log_rv", -9.4, 0.00001, true),
    test_("garch", "qlike", -6, 0.00001, true),
    test_("mlp", "squared_error_log_rv", -0.7, 0.49),
    test_("mlp", "qlike", -1.4, 0.15),
    test_("lstm", "squared_error_log_rv", -0.3, 0.8),
    test_("lstm", "qlike", -1.8, 0.066),
    test_("transformer_nosent", "squared_error_log_rv", 0.1, 0.9),
    test_("transformer_nosent", "qlike", -1.9, 0.056),
    { ...test_("har", "qlike", -1.5, 0.13), model_a: "transformer_nosent" },
  ],
  added_training_seconds: 46.4,
} as ModelComparison;

const score = (mse: number, qlike: number): ModelScore => ({
  mse_log_rv: mse,
  qlike,
  mean_error_log_rv: 0,
  var99: { breaches: 10, expected_breaches: 6, breach_rate: 0.016, kupiec_lr: 4.2, kupiec_rejected: true },
});

const period = {
  models: {
    naive: score(0.36, 0.2), vix_implied: score(1.67, 0.535), har: score(0.27, 0.157), garch: score(0.54, 0.214),
    mlp: score(0.253, 0.144), lstm: score(0.248, 0.143), transformer_nosent: score(0.245, 0.144), transformer: score(0.246, 0.134),
  },
  diebold_mariano: [],
} as unknown as PeriodScores;

describe("the new models are everywhere the app lists models", () => {
  it.each(["garch", "mlp", "lstm"] as const)("%s has a label, a colour, a chart toggle and a menu entry", (name) => {
    expect(MODEL_LABELS[name]).toBeTruthy();
    expect(SERIES_TOKENS[name]).toBe(`--series-${name}`);
    expect(CHART_SERIES.some((series) => series.id === name)).toBe(true);
    expect(ALL_MODELS).toContain(name);
  });

  it("gives every line a different look, so they don't rely on colour alone", () => {
    const looks = CHART_SERIES.map((series) => `${series.lineWidth}-${series.lineStyle}`);
    expect(new Set(looks).size).toBe(looks.length);
  });

  it("keeps the accent colour for the Transformer lines only", () => {
    const accented = CHART_SERIES.filter((series) => series.colorToken.startsWith("--series-transformer")).map((series) => series.id);
    expect(accented).toEqual(["transformer", "transformer_nosent"]);
  });
});

describe("ModelLadder", () => {
  it("runs from the simplest model to the most complex, whatever the scores", () => {
    render(<ModelLadder comparison={comparison} period={period} />);
    const names = screen.getAllByRole("listitem").map((item) => item.querySelector("span")?.textContent);
    expect(names).toEqual(LADDER.map((name) => MODEL_LABELS[name]));
  });

  it("scales each bar from zero against the largest score, and marks the best in text", () => {
    render(<ModelLadder comparison={comparison} period={period} />);
    const rows = screen.getAllByRole("listitem");
    const widthOf = (row: HTMLElement) => parseFloat((row.querySelector("[style]") as HTMLElement).style.width);
    expect(widthOf(rows[1]!)).toBeCloseTo(100);
    expect(widthOf(rows[7]!)).toBeCloseTo((0.134 / 0.535) * 100, 5);
    expect(within(rows[7]!).getByText("Best:")).toBeTruthy();
    expect(screen.getAllByText("Best:")).toHaveLength(1);
  });

  it("states the score as text, so the bar isn't the only way to read it", () => {
    render(<ModelLadder comparison={comparison} period={period} />);
    expect(within(screen.getAllByRole("listitem")[3]!).getByText("0.2140")).toBeTruthy();
  });
});

describe("leadReading", () => {
  it.each([
    [test_("garch", "qlike", -6, 0.00001, true), "Ahead, holds up"],
    [test_("har", "qlike", -2.9, 0.0038), "Ahead, at 5% only"],
    [test_("lstm", "qlike", -1.8, 0.066), "Ahead, not significant"],
    [test_("transformer_nosent", "squared_error_log_rv", 0.1, 0.9), "Behind, not significant"],
    [test_("har", "qlike", 3, 0.001, true), "Behind, holds up"],
  ])("reads the evidence, not just the sign", (test, reading) => {
    expect(leadReading(test)).toBe(reading);
  });
});

describe("groupLeads", () => {
  it("only calls a lead clear if it holds up on both scores", () => {
    expect(groupLeads(comparison)).toEqual({
      clear: ["naive", "vix_implied", "garch"],
      mixed: ["har"],
      inseparable: ["mlp", "lstm", "transformer_nosent"],
    });
  });

  it("ignores the no-news Transformer's own tests", () => {
    const onlyOther = { ...comparison, diebold_mariano: comparison.diebold_mariano.filter((test) => test.model_a !== "transformer") };
    expect(groupLeads(onlyOther)).toEqual({ clear: [], mixed: [], inseparable: [] });
  });
});

describe("ComparisonTests", () => {
  it("has one row per other model, and states the loss to a simpler model", () => {
    render(<ComparisonTests comparison={comparison} dmLags={5} originalTestCount={6} />);
    const nosent = within(screen.getByRole("row", { name: /Transformer, no news/ }));
    expect(nosent.getByText("Behind, not significant")).toBeTruthy();
    expect(screen.getAllByRole("row")).toHaveLength(8);
  });

  it("says the Bonferroni limit is for these tests, and how the leads group", () => {
    render(<ComparisonTests comparison={comparison} dmLags={5} originalTestCount={6} />);
    expect(screen.getByText(/below 0\.0036, the Bonferroni limit for these 14 tests/)).toBeTruthy();
    expect(screen.getByText(/ranks 2nd on MSE and 1st on QLIKE/)).toBeTruthy();
    expect(screen.getByText(/Its lead over Naive, VIX-implied and GARCH\(1,1\) holds up on both scores/)).toBeTruthy();
    expect(screen.getByText(/can't be told apart from MLP, LSTM and the no-news Transformer/)).toBeTruthy();
  });

  it("says when averaging seeds flatters a model", () => {
    expect(bestSingleSeed(comparison, "mse_mean")).toBe("lstm");
    render(<ComparisonTests comparison={comparison} dmLags={5} originalTestCount={6} />);
    expect(screen.getByText(/One seed at a time, LSTM has the lowest mean MSE/)).toBeTruthy();
  });
});

describe("ModelScoreTable with the comparison", () => {
  it("fills in p vs HAR from the comparison, and still says not tested where none ran", () => {
    const metrics = { kupiec_critical_value: 3.84, comparison } as unknown as Metrics;
    render(<ModelScoreTable period={period} metrics={metrics} />);
    expect(within(screen.getByRole("row", { name: /no news/ })).getAllByText("not tested")).toHaveLength(1);
    expect(within(screen.getByRole("row", { name: /no news/ })).getByText("0.1300")).toBeTruthy();
    expect(within(screen.getByRole("row", { name: /^LSTM/ })).getAllByText("not tested")).toHaveLength(2);
  });

  it("lists all eight models", () => {
    render(<ModelScoreTable period={period} metrics={{ kupiec_critical_value: 3.84 } as Metrics} />);
    expect(screen.getAllByRole("row")).toHaveLength(9);
  });
});
