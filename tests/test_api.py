"""The API against a small fake run, including what it must refuse."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app as real_app
from src.api.main import create_app
from src.config import REAL_PATHS
from src.data.loaders import read_table, write_table
from src.eval.metrics import var99
from tests.test_eval import run  # noqa: F401  (a fixture: the whole evaluation on fake data)


@pytest.fixture(scope="module")
def paths(run):  # noqa: F811
    paths, evaluation = run
    days = evaluation.dates[evaluation.labels != "later"]
    write_table(paths.processed / "sentiment_daily.csv", {
        "date": days,
        "mean_sentiment": np.linspace(-0.5, 0.5, days.size),
        "negative_share": np.linspace(0.0, 1.0, days.size),
        "log_count": np.linspace(1.0, 3.0, days.size),
    })
    # The first forecast day is left out, as a day with no published figure.
    forecast_days = np.concatenate([
        read_table(paths.artifacts / "forecasts.csv")["date"][1:],
        read_table(paths.artifacts / "forecasts_later.csv")["date"],
    ])
    write_table(paths.processed / "index_closes.csv", {
        "date": forecast_days,
        "close": 15000.0 + np.arange(forecast_days.size),
    })
    return paths


@pytest.fixture(scope="module")
def client(paths):
    with TestClient(create_app(paths)) as test_client:
        yield test_client


def days_in_test(paths) -> list[str]:
    table = read_table(paths.artifacts / "forecasts.csv")
    return [str(d) for d in table["date"][table["split"].astype(str) == "test"]]


def test_health_names_the_data_source(client, paths):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["data_source"] == "synthetic"
    assert body["test_days"] == len(days_in_test(paths))


def test_summary_holds_metrics_model_card_and_the_latest_test_forecast(client, paths):
    body = client.get("/api/summary").json()
    assert body["metrics"] == json.loads((paths.artifacts / "metrics.json").read_text(encoding="utf-8"))
    assert body["model_card"] == json.loads((paths.artifacts / "model_card.json").read_text(encoding="utf-8"))
    latest = days_in_test(paths)[-1]
    detail = json.loads((paths.artifacts / "day_details" / f"{latest}.json").read_text(encoding="utf-8"))
    assert body["latest_test_forecast"]["date"] == latest
    assert body["latest_test_forecast"]["target_date"] == detail["target_date"]
    assert body["latest_test_forecast"]["forecast"] == detail["forecast"]


def test_forecasts_returns_every_row_with_csv_values(client, paths):
    table = read_table(paths.artifacts / "forecasts.csv")
    body = client.get("/api/forecasts").json()
    assert body["count"] == len(body["rows"]) == table["date"].size
    row = body["rows"][5]
    assert row["date"] == str(table["date"][5])
    assert row["transformer"] == pytest.approx(table["transformer"][5])
    assert row["split"] == str(table["split"][5])
    assert row["target_date"] == str(table["target_date"][5])


def test_rows_carry_the_published_close_of_the_day_the_forecast_was_made(client, paths):
    closes = read_table(paths.processed / "index_closes.csv")
    rows = client.get("/api/forecasts").json()["rows"]
    published = {row["date"]: row["nifty_close"] for row in rows}
    assert published[str(closes["date"][5])] == pytest.approx(closes["close"][5])
    assert rows[0]["nifty_close"] is None  # no published figure that day


def test_a_run_without_published_closes_serves_null_and_no_other_price(paths):
    closes = paths.processed / "index_closes.csv"
    original = closes.read_bytes()
    try:
        closes.unlink()
        with TestClient(create_app(paths)) as other:
            rows = other.get("/api/forecasts").json()["rows"]
    finally:
        closes.write_bytes(original)
    assert all(row["nifty_close"] is None for row in rows)


def test_forecast_date_filter_includes_both_ends(client, paths):
    dates = [str(d) for d in read_table(paths.artifacts / "forecasts.csv")["date"]]
    body = client.get("/api/forecasts", params={"start": dates[10], "end": dates[14]}).json()
    assert [row["date"] for row in body["rows"]] == dates[10:15]


def test_a_start_date_between_sessions_rolls_to_the_next_row(client, paths):
    dates = read_table(paths.artifacts / "forecasts.csv")["date"]
    saturday = next(d for d in np.arange(dates[0], dates[-1]) if (d.astype(int) + 3) % 7 == 5)
    body = client.get("/api/forecasts", params={"start": str(saturday)}).json()
    assert body["rows"][0]["date"] == str(dates[dates > saturday][0])


def test_forecasts_can_select_models_and_keep_the_context_columns(client):
    body = client.get("/api/forecasts", params={"models": "har, naive"}).json()
    row = body["rows"][0]
    assert body["models"] == ["har", "naive"]
    assert {"har", "naive", "date", "target_date", "actual_log_rv", "return_oc", "split"} <= set(row)
    assert not {"transformer", "transformer_nosent", "vix_implied"} & set(row)


@pytest.mark.parametrize("params", [{"models": "har,lstm"}, {"start": "not-a-date"}, {"end": "2021-13-40"}])
def test_bad_forecast_requests_are_refused(client, params):
    assert client.get("/api/forecasts", params=params).status_code == 422


def test_unknown_model_message_lists_the_choices(client):
    detail = client.get("/api/forecasts", params={"models": "lstm"}).json()["detail"]
    assert "lstm" in detail and "transformer_nosent" in detail


@pytest.mark.parametrize("route", ["/api/forecasts", "/api/sentiment"])
def test_start_after_end_is_refused(client, route):
    response = client.get(route, params={"start": "2022-01-02", "end": "2022-01-01"})
    assert response.status_code == 422
    assert "after" in response.json()["detail"]


def test_later_rows_have_var_that_agrees_with_the_later_score(client, paths):
    table = read_table(paths.artifacts / "forecasts_later.csv")
    body = client.get("/api/later").json()
    row = body["rows"][2]
    assert row["var99_transformer_nosent"] == pytest.approx(var99(table["transformer_nosent"][2]))
    breaches = sum(item["breach_transformer_nosent"] for item in body["rows"])
    assert breaches == body["summary"]["models"]["transformer_nosent"]["var99"]["breaches"]


def test_later_rows_carry_the_published_close(client, paths):
    closes = read_table(paths.processed / "index_closes.csv")
    published = {str(day): close for day, close in zip(closes["date"], closes["close"])}
    row = client.get("/api/later").json()["rows"][2]
    assert row["nifty_close"] == pytest.approx(published[row["date"]])


def test_later_returns_its_own_rows_and_scores(client, paths):
    table = read_table(paths.artifacts / "forecasts_later.csv")
    metrics = json.loads((paths.artifacts / "metrics.json").read_text(encoding="utf-8"))
    body = client.get("/api/later").json()
    assert body["count"] == len(body["rows"]) == table["date"].size
    assert body["summary"] == metrics["later_check"]
    assert body["models"] == ["transformer_nosent", "har", "naive", "vix_implied"]
    row = body["rows"][2]
    assert row["date"] == str(table["date"][2])
    assert row["target_date"] == str(table["target_date"][2])
    assert row["transformer_nosent"] == pytest.approx(table["transformer_nosent"][2])
    assert row["split"] == "later"


def test_later_rows_never_appear_in_the_main_forecasts(client):
    later_dates = {row["date"] for row in client.get("/api/later").json()["rows"]}
    main_dates = {row["date"] for row in client.get("/api/forecasts").json()["rows"]}
    assert later_dates and not later_dates & main_dates


def test_later_filters_by_date_and_model(client, paths):
    days = [str(d) for d in read_table(paths.artifacts / "forecasts_later.csv")["date"]]
    body = client.get("/api/later", params={"start": days[1], "end": days[3], "models": "har"}).json()
    assert [row["date"] for row in body["rows"]] == days[1:4]
    assert "har" in body["rows"][0]
    assert not {"transformer_nosent", "naive", "vix_implied"} & set(body["rows"][0])
    assert {"actual_log_rv", "return_oc", "split"} <= set(body["rows"][0])


def test_later_refuses_the_news_model_and_lists_the_choices(client):
    response = client.get("/api/later", params={"models": "transformer"})
    assert response.status_code == 422
    assert "transformer_nosent" in response.json()["detail"]


def test_later_refuses_start_after_end(client):
    assert client.get("/api/later", params={"start": "2024-01-02", "end": "2024-01-01"}).status_code == 422


def test_later_gives_a_clear_404_when_the_run_has_none(paths):
    later = paths.artifacts / "forecasts_later.csv"
    original = later.read_bytes()
    try:
        later.unlink()
        with TestClient(create_app(paths)) as other:
            response = other.get("/api/later")
    finally:
        later.write_bytes(original)
    assert response.status_code == 404
    assert "no later check" in response.json()["detail"]


def test_day_returns_that_days_file(client, paths):
    day = days_in_test(paths)[3]
    expected = json.loads((paths.artifacts / "day_details" / f"{day}.json").read_text(encoding="utf-8"))
    assert client.get(f"/api/day/{day}").json() == expected


def test_a_non_test_date_gets_a_helpful_404(client, paths):
    table = read_table(paths.artifacts / "forecasts.csv")
    validation_day = str(table["date"][table["split"].astype(str) == "val"][0])
    response = client.get(f"/api/day/{validation_day}")
    assert response.status_code == 404
    detail = response.json()["detail"]
    days = days_in_test(paths)
    assert validation_day in detail and days[0] in detail and days[-1] in detail
    assert str(len(days)) in detail


def test_a_date_after_the_test_period_points_to_the_last_day(client, paths):
    detail = client.get("/api/day/2099-01-01").json()["detail"]
    assert days_in_test(paths)[-1] in detail


@pytest.mark.parametrize("day", ["banana", "2021-02-30", "..%2F..%2Fmetrics", "metrics", ""])
def test_malformed_days_never_reach_a_file(client, day):
    assert client.get(f"/api/day/{day}").status_code == 404


def test_malformed_day_message_shows_the_format(client):
    assert "YYYY-MM-DD" in client.get("/api/day/banana").json()["detail"]


def test_var_defaults_to_the_transformer(client):
    assert client.get("/api/var").json()["model"] == "transformer"


@pytest.mark.parametrize("model", ["transformer", "transformer_nosent", "har", "naive", "vix_implied"])
def test_var_series_matches_the_export_and_kupiec(client, paths, model):
    table = read_table(paths.artifacts / "var_series.csv")
    score = json.loads((paths.artifacts / "metrics.json").read_text(encoding="utf-8"))["test"]["models"][model]["var99"]
    body = client.get("/api/var", params={"model": model}).json()
    assert body["multiplier"] == 2.33
    assert [row["var99"] for row in body["series"]] == pytest.approx(table[f"var99_{model}"].tolist())
    for row in body["series"]:
        assert row["loss"] == -row["return_oc"]
        assert row["breach"] is (row["loss"] > row["var99"])
    assert len(body["breach_dates"]) == score["breaches"] == body["kupiec"]["breaches"]
    assert body["kupiec"]["likelihood_ratio"] == score["kupiec_lr"]
    assert body["kupiec"]["rejected"] is score["kupiec_rejected"]
    assert body["kupiec"]["critical_value"] == 3.84
    assert "mean_squared_return_over_mean_rv" in body["diagnostics"]


def test_var_breach_dates_are_target_dates(client):
    body = client.get("/api/var").json()
    expected = [row["target_date"] for row in body["series"] if row["breach"]]
    assert body["breach_dates"] == expected


def test_var_refuses_an_unknown_model_and_lists_the_choices(client):
    response = client.get("/api/var", params={"model": "lstm"})
    assert response.status_code == 422
    assert "har" in response.json()["detail"]


def test_sentiment_returns_and_filters_the_daily_series(client, paths):
    table = read_table(paths.processed / "sentiment_daily.csv")
    everything = client.get("/api/sentiment").json()
    assert everything["count"] == table["date"].size
    assert everything["rows"][0]["mean_sentiment"] == pytest.approx(table["mean_sentiment"][0])
    assert set(everything["rows"][0]) == {"date", "mean_sentiment", "negative_share", "log_count"}
    days = [str(d) for d in table["date"]]
    some = client.get("/api/sentiment", params={"start": days[2], "end": days[4]}).json()
    assert [row["date"] for row in some["rows"]] == days[2:5]


def test_cors_allows_the_frontend_only(client):
    allowed = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_cors_preflight_for_a_get_succeeds(client):
    response = client.options(
        "/api/forecasts", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"}
    )
    assert response.status_code == 200


@pytest.mark.parametrize("route", ["/api/forecasts", "/api/summary", "/api/var", "/api/sentiment", "/api/health", "/api/later"])
def test_the_api_cannot_be_written_to(client, route):
    assert client.post(route, json={}).status_code == 405


def test_requests_change_no_files(client, paths):
    def fingerprint():
        files = [p for folder in (paths.artifacts, paths.processed) for p in folder.rglob("*") if p.is_file()]
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

    before = fingerprint()
    day = days_in_test(paths)[0]
    for route in ("health", "summary", "forecasts", "later", "var", "sentiment", f"day/{day}", "day/banana"):
        client.get(f"/api/{route}")
    assert fingerprint() == before


def test_nan_becomes_null_not_invalid_json():
    from src.api.main import _rows

    rows = _rows({"date": np.array(["2021-01-01"], dtype="datetime64[D]"), "x": np.array([np.nan]), "n": np.array([1])})
    assert rows == [{"date": "2021-01-01", "x": None, "n": 1}]


def test_serving_never_loads_model_or_plotting_code():
    code = (
        "import sys, src.api.main;"
        "bad = [m for m in sys.modules if m.startswith(('src.models', 'src.nn', 'src.sentiment', 'matplotlib'))];"
        "print(bad)"
    )
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "[]"


def test_the_served_app_uses_real_artifacts():
    assert real_app.state.paths is REAL_PATHS


def test_startup_fails_clearly_when_artifacts_are_missing(paths, tmp_path):
    empty = replace(paths, artifacts=tmp_path / "none")
    with pytest.raises(RuntimeError, match="run_all"):
        with TestClient(create_app(empty)):
            pass


def test_a_real_server_refuses_synthetic_results(paths):
    """The fixture's artifacts are labelled synthetic, so serving them as real must fail."""
    with pytest.raises(RuntimeError, match="synthetic results"):
        with TestClient(create_app(replace(paths, is_synthetic=False))):
            pass


def test_startup_fails_when_a_day_file_is_missing(paths):
    day = paths.artifacts / "day_details" / f"{days_in_test(paths)[0]}.json"
    original = day.read_bytes()
    try:
        day.unlink()
        with pytest.raises(RuntimeError, match="no day file"):
            with TestClient(create_app(paths)):
                pass
    finally:
        day.write_bytes(original)
