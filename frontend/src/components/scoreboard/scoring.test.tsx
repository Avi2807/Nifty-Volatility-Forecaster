import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { Metrics, ModelScore, PeriodScores } from "../../api/types";
import { AblationTable } from "./AblationTable";
import { ModelScoreTable } from "./ModelScoreTable";
import { winner } from "./scoring";

afterEach(cleanup);

const score = (mse: number, qlike: number): ModelScore => ({
  mse_log_rv: mse,
  qlike,
  mean_error_log_rv: 0,
  var99: { breaches: 10, expected_breaches: 6, breach_rate: 0.016, kupiec_lr: 4.2, kupiec_rejected: true },
});

const period = {
  models: { transformer: score(0.2, 0.15), transformer_nosent: score(0.19, 0.16), har: score(0.3, 0.14) },
  diebold_mariano: [
    { model_a: "transformer", model_b: "har", loss: "squared_error_log_rv", mean_loss_difference: -0.1, statistic: -2, p_value: 0.03 },
    { model_a: "transformer", model_b: "transformer_nosent", loss: "squared_error_log_rv", mean_loss_difference: 0.01, statistic: 0.1, p_value: 0.9 },
  ],
} as unknown as PeriodScores;

const metrics = { kupiec_critical_value: 3.84 } as Metrics;

describe("winner", () => {
  it("picks the lowest loss for each score on its own", () => {
    expect(winner(period, "mse_log_rv")).toBe("transformer_nosent");
    expect(winner(period, "qlike")).toBe("har");
  });
});

describe("ModelScoreTable", () => {
  it("marks one winning cell per column, in text as well as style", () => {
    render(<ModelScoreTable period={period} metrics={metrics} />);
    expect(screen.getAllByText("Best:")).toHaveLength(2);
    expect(within(screen.getByRole("row", { name: /no news/ })).getByText("Best:")).toBeTruthy();
    expect(within(screen.getByRole("row", { name: /^HAR-RV/ })).getByText("Best:")).toBeTruthy();
  });

  it("says when no test exists instead of showing a p-value", () => {
    render(<ModelScoreTable period={period} metrics={metrics} />);
    const noNews = within(screen.getByRole("row", { name: /no news/ }));
    expect(noNews.getAllByText("not tested")).toHaveLength(2);
    expect(within(screen.getByRole("row", { name: /^HAR-RV/ })).getAllByText("reference")).toHaveLength(2);
  });
});

describe("AblationTable", () => {
  it("says a lower score without significance is not significant", () => {
    render(<AblationTable period={period} />);
    const mse = within(screen.getByRole("row", { name: /MSE/ }));
    expect(mse.getByText("Not significant at 5%")).toBeTruthy();
    expect(mse.getByText("Best:")).toBeTruthy();
  });
});
