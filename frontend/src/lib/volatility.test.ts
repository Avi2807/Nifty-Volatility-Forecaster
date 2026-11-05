import { describe, expect, it } from "vitest";
import { logRvToAnnualVolPct, logRvToDailyMovePct } from "./volatility";

describe("log realized variance conversions", () => {
  const logRv = Math.log(0.0001); // a variance of 1e-4 is a daily standard deviation of 1%

  it("converts to annualised volatility with sqrt(exp(x) * 252) * 100", () => {
    expect(logRvToAnnualVolPct(logRv)).toBeCloseTo(Math.sqrt(0.0252) * 100, 10);
    expect(logRvToAnnualVolPct(logRv)).toBeCloseTo(15.8745, 3);
  });

  it("converts to an expected daily move in percent", () => {
    expect(logRvToDailyMovePct(logRv)).toBeCloseTo(1, 10);
  });
});
