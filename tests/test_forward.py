"""The forward record and the challengers: honest labels, no look-ahead, and nothing existing changed."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.forward as forward
from src.api.main import create_app
from src.data.bars import DailyBars, daily_from_minute
from src.data.expiry import weekly_expiry_dates
from src.data.features import build_feature_table
from src.data.loaders import DatedSeries, MinuteData, read_table
from src.eval.challengers import CHALLENGERS, calibrated_multiplier, run_challengers
from src.forward import (
    add_outcomes,
    garch_return_history,
    kind_of_forecast,
    merge_rows,
    row_from_saved_forecast,
    score_rows,
    session_rows,
    update_record,
)
from src.live import IST
from src.models.garch import PERCENT, run_garch
from tests.test_eval import run  # noqa: F401  (a fixture: the whole evaluation on fake data)
from tests.test_live import fake_market

NOW = datetime(2026, 11, 3, 19, 0, tzinfo=IST)
FORECASTS = {"transformer_nosent": -10.0, "har": -10.1, "naive": -10.2, "vix_implied": -9.0, "garch": -9.5, "mlp_nosent": -10.0, "lstm_nosent": -10.0}


def row(date: str = "2026-11-02", target: str = "2026-11-03", kind: str = "backfilled", **changes) -> dict:
    base = {
        "date": date, "target_date": target, "made_at": "2026-11-03T19:00:00+05:30", "kind": kind,
        "forecasts": dict(FORECASTS), "var99": {"rule": 0.01, "calibrated": 0.02, "garch": 0.03},
        "actual_log_rv": None, "return_oc": None,
    }
    return {**base, **changes}


def bars(dates: list[str], rv: float = np.exp(-10.0), open_: float = 100.0, close: float = 101.0) -> DailyBars:
    n = len(dates)
    full = lambda value: np.full(n, value)  # noqa: E731
    return DailyBars(np.array(dates, dtype="datetime64[D]"), full(open_), full(close), full(open_), full(close), full(rv), full(75))


SAVED = {
    "date": "2026-11-03", "target_date": "2026-11-04", "made_at": "2026-11-03T19:42:05+05:30", "forecast": -10.25, "var99": 0.0139,
    "references": {"har": -10.28, "naive": -10.45, "vix_implied": -9.23},
}


def test_a_forecast_is_live_only_if_made_before_its_session_opens():
    assert kind_of_forecast("2026-11-04", datetime(2026, 11, 3, 19, 42, tzinfo=IST)) == "live"
    assert kind_of_forecast("2026-11-04", datetime(2026, 11, 4, 9, 14, tzinfo=IST)) == "live"
    assert kind_of_forecast("2026-11-04", datetime(2026, 11, 4, 9, 15, tzinfo=IST)) == "backfilled"
    assert kind_of_forecast("2026-11-04", datetime(2026, 11, 5, 10, 0, tzinfo=IST)) == "backfilled"


def test_the_first_forecast_made_for_a_date_is_the_one_kept():
    first = row(forecasts={**FORECASTS, "transformer_nosent": -11.0})
    assert merge_rows([first], [row()])[0]["forecasts"]["transformer_nosent"] == -11.0


def test_rows_stay_in_date_order():
    merged = merge_rows([row("2026-11-03", "2026-11-04")], [row("2026-11-02", "2026-11-03")])
    assert [item["date"] for item in merged] == ["2026-11-02", "2026-11-03"]


def test_an_earlier_live_forecast_holds_only_what_was_saved_at_the_time():
    saved = row_from_saved_forecast(SAVED)
    assert saved["kind"] == "live"
    assert saved["made_at"] == SAVED["made_at"]
    assert saved["forecasts"]["transformer_nosent"] == -10.25
    assert saved["forecasts"]["har"] == -10.28
    assert all(saved["forecasts"][name] is None for name in CHALLENGERS)
    assert saved["var99"] == {"rule": 0.0139, "calibrated": None, "garch": None}


def test_outcomes_fill_in_only_for_sessions_that_have_happened():
    done, pending = row("2026-11-02", "2026-11-03"), row("2026-11-03", "2026-11-04")
    filled = add_outcomes([done, pending], bars(["2026-11-03"], open_=100.0, close=101.0))
    assert filled[0]["actual_log_rv"] == pytest.approx(-10.0)
    assert filled[0]["return_oc"] == pytest.approx(np.log(1.01))
    assert filled[0]["forecasts"] == done["forecasts"]
    assert filled[1]["actual_log_rv"] is None


def test_an_outcome_already_on_file_is_not_replaced():
    known = row(actual_log_rv=-9.0, return_oc=0.5)
    assert add_outcomes([known], bars(["2026-11-03"]))[0]["actual_log_rv"] == -9.0


def test_scores_leave_out_pending_rows_and_days_a_model_has_no_figure_for():
    rows = [
        row(actual_log_rv=-10.0, return_oc=-0.015),
        row(actual_log_rv=-9.0, return_oc=0.0, forecasts={**FORECASTS, "garch": None}, var99={"rule": 0.01, "calibrated": None, "garch": None}),
        row(),
    ]
    scores = score_rows(rows)
    assert scores["n_days"] == 2
    assert scores["models"]["transformer_nosent"]["n_days"] == 2
    assert scores["models"]["transformer_nosent"]["mse_log_rv"] == pytest.approx((0.0 + 1.0) / 2)
    assert scores["models"]["garch"]["n_days"] == 1
    assert scores["var99"]["rule"] == {"n_days": 2, "breaches": 1, "expected_breaches": pytest.approx(0.02)}
    assert scores["var99"]["calibrated"]["n_days"] == 1


def test_scores_with_no_outcomes_are_null_rather_than_nan():
    scores = score_rows([row()])
    assert scores["models"]["har"] == {"n_days": 0, "mse_log_rv": None, "qlike": None}
    json.dumps(scores, allow_nan=False)


def test_the_calibrated_multiplier_is_passed_on_about_one_day_in_a_hundred():
    generator = np.random.default_rng(1)
    forecast = np.full(5000, -10.0)
    return_oc = generator.normal(0.0, 0.015, 5000)
    multiplier = calibrated_multiplier(forecast, return_oc)
    assert np.mean(-return_oc > multiplier * np.sqrt(np.exp(forecast))) == pytest.approx(0.01, abs=0.001)


@pytest.fixture(scope="module")
def challenged(run):  # noqa: F811
    """GARCH and both no-news networks on the fake run, recording what the existing files held before."""
    paths, _ = run
    names = ("forecasts.csv", "forecasts_later.csv", "metrics.json", "frozen_models.json")
    before = {name: (paths.artifacts / name).read_bytes() for name in names}
    run_garch(paths)
    summary = run_challengers(paths)
    return paths, summary, before


def test_challengers_change_no_existing_result(challenged):
    paths, _, before = challenged
    assert before == {name: (paths.artifacts / name).read_bytes() for name in before}


def test_challengers_are_scored_on_every_later_check_day_and_read_no_news(challenged):
    paths, summary, _ = challenged
    later = read_table(paths.artifacts / "forecasts_later.csv")
    table = read_table(paths.artifacts / "challengers_later.csv")
    np.testing.assert_array_equal(table["date"], later["date"])
    assert summary["later_check"]["n_days"] == later["date"].size
    assert set(summary["later_check"]["models"]) == set(CHALLENGERS)
    assert len(summary["later_check"]["diebold_mariano"]) == 2 * len(CHALLENGERS)
    with np.load(paths.artifacts / "comparison_models" / "mlp_nosent_seed1.npz") as saved:
        assert not any(str(name).startswith("news_") for name in saved["feature_names"])


def _table(daily, vix, turnover, next_session):
    expiries = weekly_expiry_dates(np.append(daily.dates, next_session))
    return build_feature_table(daily, vix, turnover, expiries, None, daily.dates[0], daily.dates[-1], next_session)


def test_a_row_is_the_same_whatever_happened_afterwards(challenged):
    """A backfilled forecast must depend only on the data up to its own date, or it would see its outcome."""
    paths, _, _ = challenged
    minute, vix, turnover = fake_market()
    daily, _ = daily_from_minute(minute)
    full = session_rows(paths, _table(daily, vix, turnover, np.datetime64("2026-11-04")), daily, np.datetime64("2026-11-04"), 3.0, NOW)

    cut_at = daily.dates[-6]
    keep = minute.timestamps.astype("datetime64[D]") <= cut_at
    shorter_minute = MinuteData(*(getattr(minute, name)[keep] for name in ("timestamps", "open", "high", "low", "close")))
    shorter, _ = daily_from_minute(shorter_minute)
    next_day = daily.dates[-5]
    cut = session_rows(paths, _table(shorter, vix, turnover, next_day), shorter, next_day, 3.0, NOW)

    assert cut[-1]["date"] == str(cut_at)
    match = next(item for item in full if item["date"] == str(cut_at))
    assert match["target_date"] == cut[-1]["target_date"]
    for name, value in match["forecasts"].items():
        assert cut[-1]["forecasts"][name] == pytest.approx(value, abs=1e-9), name
    assert cut[-1]["var99"] == pytest.approx(match["var99"])


def test_session_rows_label_past_sessions_as_backfilled_and_only_the_next_as_live(challenged):
    paths, _, _ = challenged
    minute, vix, turnover = fake_market()
    daily, _ = daily_from_minute(minute)
    target = np.datetime64("2026-11-04")
    rows = session_rows(paths, _table(daily, vix, turnover, target), daily, target, 3.0, datetime(2026, 11, 3, 19, 0, tzinfo=IST))
    assert daily.dates[-1] == np.datetime64("2026-10-29")  # the fake market stops before the clock above
    assert rows[-1]["target_date"] == "2026-11-04"
    assert rows[-1]["kind"] == "live"
    assert all(item["kind"] == "backfilled" for item in rows[:-1])
    assert all(item["var99"]["calibrated"] == pytest.approx(item["var99"]["rule"] * 3.0 / 2.33) for item in rows)


def test_garch_history_fills_the_gap_from_nse_in_date_order(challenged, monkeypatch):
    paths, _, _ = challenged
    minute, *_ = fake_market()
    daily, _ = daily_from_minute(minute)
    study_end = read_table(paths.processed / "daily_bars.csv")["date"][-1]
    in_gap = np.array([study_end + 1, study_end + 2, daily.dates[0] + 400], dtype="datetime64[D]")
    monkeypatch.setattr(forward, "load_index_opens", lambda _: DatedSeries(in_gap, np.array([100.0, 100.0, 100.0])))
    monkeypatch.setattr(forward, "load_index_closes", lambda _: DatedSeries(in_gap, np.array([101.0, 99.0, 500.0])))
    dates, returns = garch_return_history(paths, daily)
    assert np.all(dates[1:] > dates[:-1])
    position = np.searchsorted(dates, in_gap[0])
    assert dates[position] == in_gap[0]
    assert returns[position] == pytest.approx(np.log(1.01) * PERCENT)
    assert in_gap[2] not in dates  # a day after the live bars start isn't part of the gap
    assert dates[-1] == daily.dates[-1]


def test_the_record_survives_a_second_run_and_gains_outcomes(challenged):
    paths, _, _ = challenged
    path = paths.artifacts / forward.RECORD_FILE
    path.unlink(missing_ok=True)
    try:
        first_day = bars(["2026-11-02", "2026-11-03"])
        update_record(paths, [row("2026-11-02", "2026-11-03")], first_day, SAVED, 3.0, NOW)
        record = json.loads(path.read_text(encoding="utf-8"))
        assert [item["date"] for item in record["rows"]] == ["2026-11-02", "2026-11-03"]
        assert record["rows"][0]["actual_log_rv"] is not None
        assert record["rows"][1]["kind"] == "live"
        assert record["rows"][1]["actual_log_rv"] is None
        assert record["scores"]["live"]["n_days"] == 0
        assert record["scores"]["backfilled"]["n_days"] == 1

        # The next evening's run must not replace the live forecast, even with a different figure for that date.
        rerun = row("2026-11-03", "2026-11-04", forecasts={**FORECASTS, "transformer_nosent": -1.0})
        update_record(paths, [rerun, row("2026-11-04", "2026-11-05", kind="live")], bars(["2026-11-04"]), SAVED, 3.0, NOW)
        record = json.loads(path.read_text(encoding="utf-8"))
        kept = record["rows"][1]
        assert kept["forecasts"]["transformer_nosent"] == SAVED["forecast"]
        assert kept["actual_log_rv"] is not None
        assert record["scores"]["live"]["n_days"] == 1
        assert [item["date"] for item in record["rows"]] == ["2026-11-02", "2026-11-03", "2026-11-04"]
    finally:
        path.unlink(missing_ok=True)


def test_the_api_serves_the_record_without_a_restart_and_says_how_to_make_one(challenged, tmp_path):
    paths, summary, _ = challenged
    path = paths.artifacts / forward.RECORD_FILE
    path.unlink(missing_ok=True)
    (paths.processed / "sentiment_daily.csv").write_text("date,mean_sentiment,negative_share,log_count\n2021-01-04,0.1,0.2,1.0\n", encoding="utf-8")
    with TestClient(create_app(paths)) as client:
        missing = client.get("/api/forward")
        assert missing.status_code == 404
        assert "python -m src.live" in missing.json()["detail"]
        try:
            path.write_text('{"rows": []}', encoding="utf-8")
            served = client.get("/api/forward").json()
            assert served["rows"] == []
            assert served["challengers"]["calibrated_var"]["multiplier"] == pytest.approx(summary["calibrated_var"]["multiplier"])
        finally:
            path.unlink()
