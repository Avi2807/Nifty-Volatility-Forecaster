import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { ReplayRow } from "../../lib/replay";
import { Ticker } from "./Ticker";

const row: ReplayRow = {
  date: "2022-06-13",
  target_date: "2022-06-14",
  period: "test",
  model: "transformer",
  forecast: -9.7,
  var99: 0.018,
  breached: false,
  actual_log_rv: -10.2,
  return_oc: 0.0043,
  transformer: -9.7,
  transformer_nosent: -9.7,
  har: -9.7,
  naive: -9.7,
  vix_implied: -8.5,
  garch: -9.2,
  mlp: -9.7,
  lstm: -9.7,
  nifty_close: 15774.4,
};

describe("Ticker", () => {
  afterEach(cleanup);

  it("shows the published close to two decimals", () => {
    const { container } = render(<Ticker row={row} />);
    expect(container.textContent).toContain("Nifty close 15,774.40");
  });

  it("leaves the item out when there is no published close", () => {
    const { container } = render(<Ticker row={{ ...row, nifty_close: null }} />);
    expect(container.textContent).not.toContain("Nifty close");
    expect(container.textContent).toContain("India VIX");
  });
});
