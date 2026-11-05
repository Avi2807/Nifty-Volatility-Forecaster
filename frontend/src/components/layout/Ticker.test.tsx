import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { LiveForecast } from "../../api/types";
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

  it("follows the live forecast when there is one", () => {
    const live = { target_date: "2026-11-06", nifty_close: 23810.85, india_vix: 15.72, forecast: Math.log(0.006 ** 2), var99: 0.0145 } as LiveForecast;
    const { container, getByLabelText } = render(<Ticker row={row} live={live} />);
    expect(getByLabelText("The live forecast's numbers")).toBeTruthy();
    expect(container.textContent).toContain("Nifty close 23,810.85");
    expect(container.textContent).toContain("India VIX 15.72");
    expect(container.textContent).toContain("Forecast move ±0.60%");
    expect(container.textContent).not.toContain("Actual move");
  });
});
