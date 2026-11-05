import { describe, expect, it } from "vitest";
import type { ForecastRow, LaterRow } from "../api/types";
import { buildReplay } from "./replay";

const test_row = {
  date: "2023-06-30",
  target_date: "2023-07-03",
  actual_log_rv: -10,
  transformer: -9.1,
  transformer_nosent: -9.2,
  har: -9.3,
  naive: -9.4,
  vix_implied: -9.5,
  return_oc: 0.001,
  var99_transformer: 0.02,
  breach_transformer: 0,
  split: "test",
  nifty_close: 19189.05,
} as ForecastRow;

const later_row = {
  date: "2023-07-03",
  target_date: "2023-07-04",
  actual_log_rv: -11,
  transformer_nosent: -8.2,
  har: -8.3,
  naive: -8.4,
  vix_implied: -8.5,
  return_oc: -0.03,
  var99_transformer_nosent: 0.025,
  breach_transformer_nosent: 1,
  split: "later",
  nifty_close: null,
} as LaterRow;

describe("buildReplay", () => {
  const [test, later] = buildReplay([test_row], [later_row]);

  it("reads a test day from the with-news Transformer", () => {
    expect(test).toMatchObject({ period: "test", model: "transformer", forecast: -9.1, var99: 0.02, breached: false, transformer: -9.1 });
  });

  it("reads a later day from the no-news Transformer, and has no with-news forecast", () => {
    expect(later).toMatchObject({ period: "later", model: "transformer_nosent", forecast: -8.2, var99: 0.025, breached: true, transformer: null });
  });

  it("puts the test period first, so a run with no later check is just the test period", () => {
    expect(buildReplay([test_row], [later_row]).map((row) => row.date)).toEqual(["2023-06-30", "2023-07-03"]);
    expect(buildReplay([test_row], []).map((row) => row.period)).toEqual(["test"]);
  });
});
