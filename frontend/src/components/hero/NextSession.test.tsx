import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { LiveForecast } from "../../api/types";
import { NextSessionForecast } from "./NextSession";

const forecast: LiveForecast = {
  data_source: "real",
  made_at: "2026-11-03T19:38:53+05:30",
  model: "transformer_nosent",
  date: "2026-11-03",
  target_date: "2026-11-04",
  forecast: Math.log(0.006 ** 2),
  seed_forecasts: [-10.3, -10.2, -10.25],
  references: { har: Math.log(0.0059 ** 2), naive: Math.log(0.0054 ** 2), vix_implied: Math.log(0.0099 ** 2) },
  features: { log_turnover_ratio: Math.log(1.32), expiry_next: 0 },
  first_session: "2026-08-07",
  sessions_used: 60,
  var99: 0.01398,
  challengers: { garch: Math.log(0.008 ** 2), mlp_nosent: Math.log(0.0055 ** 2), lstm_nosent: Math.log(0.0062 ** 2) },
  var99_candidates: { calibrated: 0.021, garch: 0.019 },
  nifty_close: 23810.85,
  india_vix: 15.72,
  turnover_crore: 40323.17,
  track_record: {
    n_days: 705,
    first_date: "2023-07-03",
    last_date: "2026-05-13",
    var99_breaches: 24,
    var99_expected_breaches: 7.05,
    var99_rejected: true,
    error_band_share: 0.8,
    error_band_low: Math.log(0.5 ** 2),
    error_band_high: Math.log(1.5 ** 2),
  },
  sources: { bars: "https://example.test/bars", closing_file: "https://example.test/close", holidays: "https://example.test/holidays" },
};

const beforeTheSession = new Date("2026-11-03T14:30:00Z");
const afterTheSession = new Date("2026-11-04T10:00:00Z");

describe("NextSessionForecast", () => {
  afterEach(cleanup);

  it("shows the forecast in the units people read, with its index points and error range", () => {
    const { container } = render(<NextSessionForecast forecast={forecast} now={beforeTheSession} />);
    const text = container.textContent ?? "";
    expect(text).toContain("forecasts for Wed, 4 Nov 2026");
    expect(text).toContain("±0.60%");
    expect(text).toContain("about 143 points from the 23,810.85 close");
    expect(text).toContain("9.5% annualised");
    expect(text).toContain("a loss of 1.40%");
    expect(text).toContain("±0.30% to ±0.90%");
    expect(text).toContain("about 80% of the time");
    expect(text).toContain("+32%");
  });

  it("says plainly that the VaR ran too low and where the prices came from", () => {
    const { container } = render(<NextSessionForecast forecast={forecast} now={beforeTheSession} />);
    const text = container.textContent ?? "";
    expect(text).toContain("on 24 of 705 days, against about 7 expected");
    expect(text).toContain("Yahoo's unofficial 5-minute feed");
    expect(text).toContain("in none of the study's scores");
  });

  it("sets the simpler models and the untested VaR candidates beside the forecast", () => {
    const { container } = render(<NextSessionForecast forecast={forecast} now={beforeTheSession} />);
    const text = container.textContent ?? "";
    expect(text).toContain("LSTM, no news±0.62%");
    expect(text).toContain("GARCH(1,1)±0.80%");
    expect(text).toContain("haven't been judged yet: a calibrated line at 2.10% and GARCH's at 1.90%");
  });

  it("isn't marked out of date until the session has closed", () => {
    const { queryByRole } = render(<NextSessionForecast forecast={forecast} now={new Date("2026-11-04T09:59:00Z")} />);
    expect(queryByRole("status")).toBeNull();
  });

  it("is marked out of date after the session, and says how to make a new one", () => {
    const { getByRole, container } = render(<NextSessionForecast forecast={forecast} now={afterTheSession} />);
    expect(getByRole("status").textContent).toContain("python -m src.live");
    expect(container.textContent).toContain("Last live forecast");
    expect(container.textContent).toContain("What the no-news model forecast for");
  });
});
