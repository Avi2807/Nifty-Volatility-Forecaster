"""Loaders must give clear errors, and real and synthetic runs must never touch each other."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.config import MARKET_FEATURES, NEWS_FEATURES, REAL_PATHS, ROOT, SYNTHETIC_PATHS, RunPaths
from src.data import loaders
from src.data.features import write_index_closes
from src.data.loaders import DataFileError


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_missing_files_are_listed_with_what_to_download(tmp_path):
    missing = loaders.missing_required(tmp_path)
    assert len(missing) == 4
    assert any("Kaggle" in item for item in missing)
    # Turnover is optional, so its absence is not an error.
    assert not any(item.startswith("turnover") for item in missing)


def test_missing_column_error_names_the_column_and_the_columns_found(tmp_path):
    write(tmp_path / "nifty_minute.csv", "date,open,high,close\n2021-01-01 09:15:00,1,1,1\n")
    with pytest.raises(DataFileError, match=r"missing column.*low.*Columns found"):
        loaders.load_minute(tmp_path)


def test_missing_file_error_names_the_source(tmp_path):
    with pytest.raises(DataFileError, match="India VIX"):
        loaders.load_vix(tmp_path)


def test_nse_style_vix_file_is_understood(tmp_path):
    write(
        tmp_path / "India_VIX_2021.csv",
        'Date ,Open ,Close \n05-JAN-2021,10,"1,020.5"\n04-JAN-2021,10,20.5\n04-JAN-2021,10,21.5\n06-JAN-2021,10,-\n',
    )
    series = loaders.load_vix(tmp_path)
    assert series.dates.astype(str).tolist() == ["2021-01-04", "2021-01-05"]  # sorted, "-" removed
    assert series.values.tolist() == [21.5, 1020.5]  # last duplicate kept, thousands comma parsed


def test_byte_order_mark_in_an_nse_header_is_ignored(tmp_path):
    (tmp_path / "vix.csv").write_bytes(b"\xef\xbb\xbfDate ,Close \n01-JAN-2015,15.0\n")
    assert loaders.load_vix(tmp_path).values.tolist() == [15.0]


def test_vix_files_split_by_year_are_combined(tmp_path):
    write(tmp_path / "vix_2020.csv", "date,close\n2020-12-31,20\n")
    write(tmp_path / "vix_2021.csv", "date,close\n2021-01-01,21\n")
    assert loaders.load_vix(tmp_path).values.tolist() == [20.0, 21.0]


def test_turnover_column_is_found_by_substring_and_is_optional(tmp_path):
    assert loaders.load_turnover(tmp_path) is None
    write(tmp_path / "nifty_turnover.csv", "Date,Shares Traded,Turnover (Rs. Cr)\n01 Jan 2021,5,123.5\n")
    assert loaders.load_turnover(tmp_path).values.tolist() == [123.5]


def test_series_with_no_usable_values_is_an_error(tmp_path):
    write(tmp_path / "vix.csv", "date,close\n2021-01-01,-\n")
    with pytest.raises(DataFileError, match="no rows with a positive"):
        loaders.load_vix(tmp_path)


def test_nse_turnover_download_is_found_under_its_own_name(tmp_path):
    (tmp_path / "NIFTY 50-02-01-2023-to-01-01-2024.csv").write_text(
        "\ufeffDate ,Open ,High ,Low ,Close ,Shares Traded ,Turnover (\u20b9 Cr)\n"
        "02-JAN-2023,1,1,1,1,256073979,\"11,793.06\"\n", encoding="utf-8",
    )
    # Neither of these has a turnover column, so neither may be picked up as turnover.
    write(tmp_path / "NIFTY 50_minute.csv", "date,open,high,low,close\n")
    write(tmp_path / "NIFTY 50_Historical_PR_01012015to01012016.csv", '"Index Name","Date","Close"\n')
    series = loaders.load_turnover(tmp_path)
    assert series.dates.astype(str).tolist() == ["2023-01-02"]
    assert series.values.tolist() == [11793.06]


def test_index_closes_come_from_nse_price_files_and_are_optional(tmp_path):
    assert loaders.load_index_closes(tmp_path) is None
    write(tmp_path / "NIFTY 50-02-01-2022-to-01-01-2023.csv", "Date ,Close ,Turnover (Cr)\n13-JUN-2022,1,1\n")
    assert loaders.load_index_closes(tmp_path) is None  # the turnover download isn't this source
    write(
        tmp_path / "NIFTY 50_Historical_PR_02012022to01012023.csv",
        '"Index Name","Date","Open","High","Low","Close"\n'
        '"NIFTY 50","14 Jun 2022","1","1","1","15732.10"\n"NIFTY 50","13 Jun 2022","1","1","1","15774.40"\n',
    )
    series = loaders.load_index_closes(tmp_path)
    assert series.dates.astype(str).tolist() == ["2022-06-13", "2022-06-14"]
    assert series.values.tolist() == [15774.40, 15732.10]


def test_index_closes_file_is_written_only_when_the_source_exists(tmp_path):
    paths = RunPaths(tmp_path / "raw", tmp_path / "processed", tmp_path / "artifacts", True)
    paths.raw.mkdir()
    write_index_closes(paths)
    assert not (paths.processed / "index_closes.csv").exists()
    write(paths.raw / "NIFTY 50_Historical_PR_x.csv", 'Date,Close\n"13 Jun 2022",15774.40\n')
    write_index_closes(paths)
    table = loaders.read_table(paths.processed / "index_closes.csv")
    assert table["close"].tolist() == [15774.40]


def test_ambiguous_dates_are_parsed_as_iso_not_guessed(tmp_path):
    write(tmp_path / "vix.csv", "date,close\n2021-02-03,20\n")
    assert str(loaders.load_vix(tmp_path).dates[0]) == "2021-02-03"


def test_unknown_date_format_is_an_error(tmp_path):
    write(tmp_path / "vix.csv", "date,close\nsoon,20\n")
    with pytest.raises(DataFileError, match="unrecognised date format"):
        loaders.load_vix(tmp_path)


def test_minute_timezone_aware_timestamps_become_ist(tmp_path):
    write(tmp_path / "nifty_minute.csv", "date,open,high,low,close\n2021-01-04 03:45:00+00:00,1,1,1,1\n")
    assert str(loaders.load_minute(tmp_path).timestamps[0]) == "2021-01-04T09:15"


def test_headlines_filter_category_and_date_range(tmp_path):
    write(
        tmp_path / "india_headlines.csv",
        "publish_date,headline_category,headline_text\n"
        "20141231,business.india-business,too early\n"
        "20150101,business.india-business,kept one\n"
        "20150102,sports.cricket,wrong section\n"
        "20230630,Business.markets,kept two\n"
        "20230701,business.markets,too late\n",
    )
    result = loaders.load_business_headlines(tmp_path, np.datetime64("2015-01-01"), np.datetime64("2023-06-30"))
    assert result.texts.tolist() == ["kept one", "kept two"]


def test_phrasebank_splits_on_the_last_at_sign(tmp_path):
    write(tmp_path / "Sentences_66Agree.txt", "Email a@b.com for results .@neutral\nProfit fell .@negative\n")
    sentences, labels = loaders.load_phrasebank(tmp_path)
    assert sentences[0] == "Email a@b.com for results ."
    assert labels == ["neutral", "negative"]


def test_phrasebank_bad_line_reports_its_line_number(tmp_path):
    write(tmp_path / "Sentences_66Agree.txt", "Fine .@neutral\nno label here\n")
    with pytest.raises(DataFileError, match=":2:"):
        loaders.load_phrasebank(tmp_path)


def test_placeholder_expiry_file_is_last_thursdays_and_flagged(tmp_path):
    path = tmp_path / "expiry.csv"
    dates, is_placeholder = loaders.load_expiry_dates(path)
    assert is_placeholder
    assert "TODO" in path.read_text(encoding="utf-8")
    assert dates.size == 120
    assert ((dates.astype(np.int64) + 3) % 7 == 3).all()  # all Thursdays
    assert ((dates + 7).astype("datetime64[M]") != dates.astype("datetime64[M]")).all()  # last of month
    assert str(dates[0]) == "2015-01-29"


def test_edited_expiry_file_is_not_flagged_as_placeholder(tmp_path):
    path = tmp_path / "expiry.csv"
    write(path, "date\n2021-01-28\n")
    dates, is_placeholder = loaders.load_expiry_dates(path)
    assert not is_placeholder
    assert dates.size == 1


def test_real_and_synthetic_runs_share_no_directories():
    real = (REAL_PATHS.raw, REAL_PATHS.processed, REAL_PATHS.artifacts)
    synthetic = (SYNTHETIC_PATHS.raw, SYNTHETIC_PATHS.processed, SYNTHETIC_PATHS.artifacts)
    assert not set(real) & set(synthetic)
    assert SYNTHETIC_PATHS.is_synthetic and not REAL_PATHS.is_synthetic


def test_synthetic_file_in_the_real_folder_is_refused(tmp_path):
    write(tmp_path / "SYNTHETIC_india_vix.csv", "date,close\n2021-01-01,20\n")
    with pytest.raises(DataFileError, match="Synthetic files"):
        loaders.assert_no_synthetic_files(tmp_path)


def test_feature_counts_match_the_model_inputs():
    assert len(MARKET_FEATURES) == 15
    assert len(MARKET_FEATURES) + len(NEWS_FEATURES) == 18


BANNED = {
    "sklearn", "torch", "tensorflow", "jax", "statsmodels", "arch", "nltk", "spacy", "transformers", "langchain", "scipy",
}


def imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_project_rule_no_ml_libraries_and_pandas_only_in_loaders():
    sources = [*ROOT.glob("src/**/*.py"), ROOT / "run_all.py"]
    assert sources
    for path in sources:
        names = imported_modules(path)
        assert not names & BANNED, f"{path} imports {names & BANNED}"
        if "pandas" in names:
            assert path.name == "loaders.py", f"{path} imports pandas outside the loaders"
