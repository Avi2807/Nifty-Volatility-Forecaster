"""The live forecast: parsing what's fetched, the calendar, refusing bad data, and matching the later check."""

from __future__ import annotations

import json
from datetime import datetime, time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.live as live
from src.api.main import create_app
from src.config import REAL_PATHS
from src.data.bars import daily_from_minute
from src.data.expiry import weekly_expiry_dates
from src.data.features import build_feature_table
from src.data.loaders import DatedSeries, MinuteData, load_expiry_dates, load_minute, load_turnover, load_vix, read_table
from src.eval.challengers import CHALLENGERS, run_challengers
from src.forward import RECORD_FILE
from src.live import (
    IST,
    LiveDataError,
    build_forecast,
    complete_sessions_only,
    parse_bars,
    parse_closing_file,
    parse_holidays,
    upcoming_sessions,
)
from src.models.garch import run_garch
from tests.test_api import paths as served_paths  # noqa: F401  (a fixture: a fake run the API can serve)
from tests.test_eval import run  # noqa: F401  (a fixture: the whole evaluation on fake data)

DAY = np.datetime64("2026-11-03")
HEADER = "Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield\n"


def closing_file(date: str = "03-11-2026", close: float = 23810.85) -> str:
    return (
        HEADER
        + f"Nifty 50,{date},23582.25,23810.85,23482.8,{close},370.05,1.58,408375815,40323.17,19.71,2.92,1.46\n"
        + f"India VIX,{date},15.5,16.2225,14.2525,15.72,0.22,1.44,-,-,-,-,-\n"
    )


def bars_payload(stamps, prices, offset: int = 19800) -> bytes:
    quote = {name: list(prices) for name in ("open", "high", "low", "close")}
    return json.dumps(
        {"chart": {"result": [{"meta": {"gmtoffset": offset}, "timestamp": stamps, "indicators": {"quote": [quote]}}]}}
    ).encode()


def test_bars_are_read_on_the_ist_clock_with_null_prices_as_nan():
    # 1793677500 is 2026-11-03 03:45 UTC, which is 09:15 in India.
    minute = parse_bars(bars_payload([1793677500, 1793677800], [23582.25, None]))
    assert str(minute.timestamps[0]) == "2026-11-03T09:15"
    assert str(minute.timestamps[1]) == "2026-11-03T09:20"
    assert minute.close[0] == 23582.25
    assert np.isnan(minute.close[1])


def test_bars_in_another_time_zone_are_refused():
    with pytest.raises(LiveDataError, match="expected IST"):
        parse_bars(bars_payload([1793677500], [23582.25], offset=0))


def part_day_bars() -> MinuteData:
    stamps = np.array(["2026-11-02T15:25", "2026-11-03T09:15", "2026-11-03T15:25"], dtype="datetime64[m]")
    values = np.array([1.0, 2.0, 3.0])
    return MinuteData(stamps, values, values, values, values)


def test_todays_bars_are_dropped_until_the_session_closes():
    kept = complete_sessions_only(part_day_bars(), datetime(2026, 11, 3, 15, 29, tzinfo=IST))
    assert kept.timestamps.size == 1
    assert str(kept.timestamps[0]) == "2026-11-02T15:25"


def test_todays_bars_are_kept_once_the_session_has_closed():
    kept = complete_sessions_only(part_day_bars(), datetime(2026, 11, 3, 15, 30, tzinfo=IST))
    assert kept.timestamps.size == 3


def test_closing_file_gives_the_nifty_and_vix_figures():
    close = parse_closing_file(closing_file(), DAY)
    assert (close.nifty_close, close.turnover_crore, close.india_vix) == (23810.85, 40323.17, 15.72)


def test_closing_file_for_another_date_is_refused():
    with pytest.raises(LiveDataError, match="is dated 02-11-2026"):
        parse_closing_file(closing_file("02-11-2026"), DAY)


def test_closing_file_with_a_changed_layout_is_refused():
    with pytest.raises(LiveDataError, match="unexpected layout"):
        parse_closing_file("Name,Date\nNifty 50,03-11-2026\n", DAY)


NOVEMBER_HOLIDAYS = parse_holidays(
    json.dumps({"CM": [{"tradingDate": "10-Nov-2026"}, {"tradingDate": "08-Nov-2026"}, {"tradingDate": "24-Nov-2026"}]}).encode()
)


def test_holidays_are_parsed_in_date_order():
    assert [str(day) for day in NOVEMBER_HOLIDAYS] == ["2026-11-08", "2026-11-10", "2026-11-24"]


def test_the_next_session_after_tuesday_is_wednesday():
    assert str(upcoming_sessions(DAY, NOVEMBER_HOLIDAYS)[0]) == "2026-11-04"


def test_the_next_session_skips_a_weekend():
    assert str(upcoming_sessions(np.datetime64("2026-11-06"), NOVEMBER_HOLIDAYS)[0]) == "2026-11-09"


def test_the_next_session_skips_a_holiday():
    assert str(upcoming_sessions(np.datetime64("2026-11-09"), NOVEMBER_HOLIDAYS)[0]) == "2026-11-11"


def test_a_tuesday_holiday_moves_the_weekly_expiry_to_monday():
    sessions = np.union1d(np.array([DAY]), upcoming_sessions(DAY, NOVEMBER_HOLIDAYS))
    expiries = weekly_expiry_dates(sessions)
    assert np.datetime64("2026-11-09") in expiries
    assert np.datetime64("2026-11-10") not in expiries


def fake_market(sessions: int = 60, seed: int = 4) -> tuple[MinuteData, DatedSeries, DatedSeries]:
    """Simulated bars and daily series, used only to exercise the code in these tests."""
    generator = np.random.default_rng(seed)
    days = np.busday_offset("2026-08-07", np.arange(sessions), roll="forward")
    stamps = np.concatenate([np.datetime64(day, "m") + np.timedelta64(555, "m") + np.arange(75) * np.timedelta64(5, "m") for day in days])
    close = 23000.0 * np.exp(np.cumsum(generator.normal(0.0, 0.0006, stamps.size)))
    minute = MinuteData(stamps, close, close * 1.0001, close * 0.9999, close)
    return (
        minute,
        DatedSeries(days, 15.0 + generator.normal(0.0, 0.5, sessions)),
        DatedSeries(days, 30000.0 + generator.normal(0.0, 3000.0, sessions)),
    )


def test_next_session_keeps_the_last_row_without_a_target_and_changes_nothing_else():
    minute, vix, turnover = fake_market()
    daily, _ = daily_from_minute(minute)
    wide = dict(start=daily.dates[0], end=daily.dates[-1])
    expiries = np.array(["2026-11-04"], dtype="datetime64[D]")
    plain = build_feature_table(daily, vix, turnover, expiries, None, **wide)
    with_next = build_feature_table(daily, vix, turnover, expiries, None, **wide, next_session=np.datetime64("2026-11-04"))
    assert with_next.dates[-1] == daily.dates[-1]
    assert np.isnan(with_next.target[-1])
    assert with_next.X[-1, with_next.columns.index("expiry_next")] == 1.0
    np.testing.assert_array_equal(with_next.dates[:-1], plain.dates)
    np.testing.assert_array_equal(with_next.X[:-1], plain.X)
    np.testing.assert_array_equal(with_next.target[:-1], plain.target)


def test_too_few_sessions_are_refused(run):  # noqa: F811
    paths, _ = run
    minute, vix, turnover = fake_market(sessions=30)
    with pytest.raises(LiveDataError, match="22 are needed"):
        build_forecast(paths, minute, vix, turnover, np.array([], dtype="datetime64[D]"), np.datetime64("2026-11-04"))


def test_a_missing_vix_day_is_refused_rather_than_filled_in(run):  # noqa: F811
    paths, _ = run
    minute, vix, turnover = fake_market()
    gap = DatedSeries(vix.dates[:-2], vix.values[:-2])
    with pytest.raises(LiveDataError, match="complete feature rows"):
        build_forecast(paths, minute, gap, turnover, np.array([], dtype="datetime64[D]"), np.datetime64("2026-11-04"))


def test_a_changed_checkpoint_stops_the_forecast(run, tmp_path):  # noqa: F811
    paths, _ = run
    checkpoint = paths.artifacts / "models" / "transformer_nosent_seed1.npz"
    original = checkpoint.read_bytes()
    minute, vix, turnover = fake_market()
    try:
        checkpoint.write_bytes(original + b"changed")
        with pytest.raises(RuntimeError, match="has changed since training was frozen"):
            build_forecast(paths, minute, vix, turnover, np.array([], dtype="datetime64[D]"), np.datetime64("2026-11-04"))
    finally:
        checkpoint.write_bytes(original)


@pytest.fixture
def fetched(monkeypatch, tmp_path, run):  # noqa: F811
    """A whole run with no network: simulated bars, the closing file and the holiday list."""
    paths, _ = run
    if not (paths.artifacts / "challengers.json").exists():
        run_garch(paths)
        run_challengers(paths)
    minute, vix, turnover = fake_market()
    last = daily_from_minute(minute)[0]
    day = last.dates[-1]
    stamps = ((minute.timestamps - np.timedelta64(330, "m")).astype("datetime64[s]").astype(np.int64)).tolist()
    payload = json.dumps({
        "chart": {"result": [{
            "meta": {"gmtoffset": 19800},
            "timestamp": stamps,
            "indicators": {"quote": [{name: getattr(minute, name).tolist() for name in ("open", "high", "low", "close")}]},
        }]}
    }).encode()
    responses = {
        live.BARS_URL: payload,
        live.HOLIDAYS_URL: json.dumps({"CM": []}).encode(),
    }
    dated = day.item().strftime("%d-%m-%Y")
    file_url = live.CLOSING_FILE_URL.format(day=day.item().strftime("%d%m%Y"))
    responses[file_url] = closing_file(dated, float(last.close[-1])).encode()
    monkeypatch.setattr(live, "_fetch", lambda url: responses[url])
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path)
    # Series stop the day before the last session, as NSE's raw files would.
    monkeypatch.setattr(live, "load_vix", lambda _: DatedSeries(vix.dates[:-1], vix.values[:-1]))
    monkeypatch.setattr(live, "load_turnover", lambda _: DatedSeries(turnover.dates[:-1], turnover.values[:-1]))
    output = paths.artifacts / "live_forecast.json"
    yield paths, responses, file_url, dated, day, output
    output.unlink(missing_ok=True)
    (paths.artifacts / RECORD_FILE).unlink(missing_ok=True)


AFTER_THE_CLOSE = datetime(2026, 11, 3, 19, 0, tzinfo=IST)


def test_a_run_writes_a_forecast_for_the_next_session(fetched):
    paths, _, _, _, day, output = fetched
    result = live.main(paths, AFTER_THE_CLOSE)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == result
    assert saved["date"] == str(day)
    assert saved["target_date"] > saved["date"]
    assert saved["forecast"] == pytest.approx(np.mean(saved["seed_forecasts"]))
    assert saved["var99"] == pytest.approx(2.33 * np.sqrt(np.exp(saved["forecast"])))
    assert saved["india_vix"] == 15.72
    assert saved["turnover_crore"] == 40323.17
    assert saved["features"]["log_vix"] == pytest.approx(np.log(15.72))
    assert saved["track_record"]["error_band_low"] < 0 < saved["track_record"]["error_band_high"]


def test_a_run_gives_the_simpler_models_and_var_candidates_their_own_figures(fetched):
    paths, *_ = fetched
    saved = live.main(paths, AFTER_THE_CLOSE)
    assert set(saved["challengers"]) == set(CHALLENGERS)
    assert saved["var99_candidates"]["garch"] == pytest.approx(2.33 * np.sqrt(np.exp(saved["challengers"]["garch"])))
    multiplier = json.loads((paths.artifacts / "challengers.json").read_text(encoding="utf-8"))["calibrated_var"]["multiplier"]
    assert saved["var99_candidates"]["calibrated"] == pytest.approx(multiplier * np.sqrt(np.exp(saved["forecast"])))


def test_a_run_adds_its_forecast_to_the_record_and_backfills_what_it_can(fetched):
    paths, _, _, _, day, _ = fetched
    saved = live.main(paths, datetime.combine(day.item(), time(19, 0), IST))  # that session's own evening
    record = json.loads((paths.artifacts / RECORD_FILE).read_text(encoding="utf-8"))
    newest = record["rows"][-1]
    assert newest["date"] == str(day)
    assert newest["kind"] == "live"
    assert newest["forecasts"]["transformer_nosent"] == pytest.approx(saved["forecast"])
    assert newest["forecasts"]["har"] == pytest.approx(saved["references"]["har"])
    assert newest["var99"]["rule"] == pytest.approx(saved["var99"])
    assert newest["actual_log_rv"] is None
    assert len(record["rows"]) > 1
    assert all(row["kind"] == "backfilled" and row["actual_log_rv"] is not None for row in record["rows"][:-1])
    assert record["scores"]["backfilled"]["n_days"] == len(record["rows"]) - 1


def test_the_forecast_a_run_replaces_stays_on_the_record(fetched):
    paths, _, _, _, day, output = fetched
    earlier = (day - 1).item().isoformat()
    output.write_text(json.dumps({
        "date": earlier, "target_date": str(day), "made_at": f"{earlier}T19:00:00+05:30", "forecast": -1.5, "var99": 0.5,
        "references": {"har": -2.0, "naive": -3.0, "vix_implied": -4.0},
    }), encoding="utf-8")
    live.main(paths, AFTER_THE_CLOSE)
    record = json.loads((paths.artifacts / RECORD_FILE).read_text(encoding="utf-8"))
    kept = next(row for row in record["rows"] if row["date"] == earlier)
    assert kept["kind"] == "live"
    assert kept["forecasts"]["transformer_nosent"] == -1.5
    assert kept["forecasts"]["garch"] is None
    assert kept["actual_log_rv"] is not None
    assert record["scores"]["live"]["n_days"] == 1


def test_a_run_refuses_without_the_challengers(fetched):
    paths, *_ = fetched
    saved = paths.artifacts / "challengers.json"
    original = saved.read_bytes()
    saved.unlink()
    try:
        with pytest.raises(LiveDataError, match="src.eval.challengers"):
            live.main(paths, AFTER_THE_CLOSE)
    finally:
        saved.write_bytes(original)


def test_a_run_touches_no_scored_artifact(fetched):
    paths, *_ = fetched
    before = {name: (paths.artifacts / name).read_bytes() for name in ("forecasts.csv", "forecasts_later.csv", "metrics.json")}
    live.main(paths, AFTER_THE_CLOSE)
    assert before == {name: (paths.artifacts / name).read_bytes() for name in before}


def test_a_price_that_disagrees_with_nse_leaves_the_old_forecast_in_place(fetched):
    paths, responses, file_url, dated, _, output = fetched
    output.write_text('{"previous": true}', encoding="utf-8")
    responses[file_url] = closing_file(dated, 99999.0).encode()
    with pytest.raises(LiveDataError, match="can't be trusted"):
        live.main(paths, AFTER_THE_CLOSE)
    assert json.loads(output.read_text(encoding="utf-8")) == {"previous": True}


def test_the_api_serves_a_new_forecast_without_a_restart(served_paths):  # noqa: F811
    paths = served_paths
    output = paths.artifacts / "live_forecast.json"
    output.unlink(missing_ok=True)
    with TestClient(create_app(paths)) as client:
        missing = client.get("/api/live")
        assert missing.status_code == 404
        assert "python -m src.live" in missing.json()["detail"]
        try:
            output.write_text('{"target_date": "2026-11-04"}', encoding="utf-8")
            assert client.get("/api/live").json() == {"target_date": "2026-11-04"}
            output.write_text('{"target_date": "2026-11-05"}', encoding="utf-8")
            assert client.get("/api/live").json() == {"target_date": "2026-11-05"}
        finally:
            output.unlink()


@pytest.mark.skipif(not (REAL_PATHS.artifacts / "forecasts_later.csv").exists(), reason="needs the real run's files")
def test_the_live_steps_reproduce_the_published_later_check_forecast():
    """The last 60 sessions of the training feed, through build_forecast, must give the row already on file."""
    later = read_table(REAL_PATHS.artifacts / "forecasts_later.csv")
    day, target = np.datetime64(later["date"][-1]), np.datetime64(later["target_date"][-1])
    minute = load_minute(REAL_PATHS.raw)
    daily, _ = daily_from_minute(minute)
    stamps = minute.timestamps.astype("datetime64[D]")
    chosen = np.isin(stamps, daily.dates[daily.dates <= day][-60:])
    recent = MinuteData(*(getattr(minute, name)[chosen] for name in ("timestamps", "open", "high", "low", "close")))
    expiry_dates, _ = load_expiry_dates()
    result = build_forecast(REAL_PATHS, recent, load_vix(REAL_PATHS.raw), load_turnover(REAL_PATHS.raw), expiry_dates, target)
    assert result["forecast"] == pytest.approx(float(later["transformer_nosent"][-1]), abs=1e-9)
    assert result["references"]["har"] == pytest.approx(float(later["har"][-1]), abs=1e-9)
    assert result["references"]["naive"] == pytest.approx(float(later["naive"][-1]), abs=1e-9)
    assert result["references"]["vix_implied"] == pytest.approx(float(later["vix_implied"][-1]), abs=1e-9)
