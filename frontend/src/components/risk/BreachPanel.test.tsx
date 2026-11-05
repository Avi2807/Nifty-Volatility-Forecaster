import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { VarResponse } from "../../api/types";
import { BreachPanel } from "./BreachPanel";

const response = {
  kupiec: { breaches: 24, expected_breaches: 6.15, breach_rate: 0.039, likelihood_ratio: 25.3, critical_value: 3.84, rejected: true },
  diagnostics: { mean_squared_return_over_mean_rv: 1.31, breaches_if_rv_were_known: 6 },
} as unknown as VarResponse;

describe("BreachPanel", () => {
  afterEach(cleanup);

  it("keeps the breach count, which the VaR warning rests on, and leaves the Kupiec result off the page", () => {
    const { container } = render(<BreachPanel response={response} />);
    const text = container.textContent ?? "";
    expect(text).toContain("24");
    expect(text).toContain("against about 6.2 expected");
    expect(text.toLowerCase()).not.toContain("kupiec");
    expect(text).not.toContain("25.30");
    expect(text).not.toContain("3.84");
  });
});
