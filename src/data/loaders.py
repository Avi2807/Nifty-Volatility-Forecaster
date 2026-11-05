"""File input and output. This is the only module that imports pandas.

The project rule limits pandas to CSV loading and date handling, so every loader hands
back plain NumPy arrays and all computation happens elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EXPIRY_FILE


class DataFileError(Exception):
    """A raw data file is missing or malformed. The message says how to fix it."""


@dataclass(frozen=True)
class SourceSpec:
    pattern: str  # case-insensitive filename pattern, searched recursively
    description: str
    alternate_pattern: str | None = None


SOURCES: dict[str, SourceSpec] = {
    "minute": SourceSpec(
        "*minute*.csv",
        "Nifty 50 1-minute OHLC (Kaggle 'NSE - Nifty 50 Index Minute Data 2015 to 2024'), "
        "columns: date, open, high, low, close",
    ),
    "vix": SourceSpec("*vix*.csv", "India VIX daily history from NSE, columns: date, close"),
    "turnover": SourceSpec(
        "*turnover*.csv",
        "Nifty 50 daily turnover from NSE, columns: date, turnover (Rs crore)",
        "nifty 50-*.csv",  # NSE's own download name: NIFTY 50-<from>-to-<to>.csv
    ),
    "headlines": SourceSpec(
        "*headlines*.csv",
        "India News Headlines (Times of India archive), "
        "columns: publish_date, headline_category, headline_text",
    ),
    "phrasebank": SourceSpec(
        "*66agree*", "Financial PhraseBank 66%-agreement file (Sentences_66Agree.txt)"
    ),
    "index_close": SourceSpec(
        "*historical_pr*.csv", "Nifty 50 daily index history from NSE, columns: date, close"
    ),
}
# Turnover is optional: without it the feature is set to 0 with a warning.
# Index closes are optional and display-only: no feature, model or score reads them.
REQUIRED_SOURCES = ("minute", "vix", "headlines", "phrasebank")


def find_source(raw_dir: Path, key: str) -> list[Path]:
    """All files in raw_dir matching a source's pattern, sorted by name."""
    spec = SOURCES[key]
    if not raw_dir.exists():
        return []
    patterns = [spec.pattern] + ([spec.alternate_pattern] if spec.alternate_pattern else [])
    return sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and any(fnmatch(p.name.lower(), pattern) for pattern in patterns)
    )


def missing_required(raw_dir: Path) -> list[str]:
    """Readable descriptions of the required sources absent from raw_dir."""
    return [
        f"{key}: no file matching '{SOURCES[key].pattern}' in {raw_dir}\n    expected: {SOURCES[key].description}"
        for key in REQUIRED_SOURCES
        if not find_source(raw_dir, key)
    ]


def assert_no_synthetic_files(raw_dir: Path) -> None:
    """Refuse a real run if a synthetic file has been copied into the real data folder."""
    if raw_dir.exists():
        strays = [p.name for p in raw_dir.rglob("*") if "synthetic" in p.name.lower()]
        if strays:
            raise DataFileError(f"Synthetic files found in the real data folder {raw_dir}: {strays}")


def _single_file(raw_dir: Path, key: str) -> Path:
    files = find_source(raw_dir, key)
    if not files:
        raise DataFileError(f"Missing file for '{key}' in {raw_dir}. Expected: {SOURCES[key].description}")
    if len(files) > 1:
        raise DataFileError(f"Expected one file for '{key}' but found {[f.name for f in files]}")
    return files[0]


def _read_csv(path: Path, required: list[str], **kwargs) -> pd.DataFrame:
    """Read a CSV with normalised (stripped, lowercase) headers and check its columns."""
    frame = pd.read_csv(path, skipinitialspace=True, low_memory=False, **kwargs)
    frame.columns = [str(c).lstrip("\ufeff").strip().lower() for c in frame.columns]
    absent = [c for c in required if c not in frame.columns]
    if absent:
        raise DataFileError(f"{path.name}: missing column(s) {absent}. Columns found: {list(frame.columns)}")
    if frame.empty:
        raise DataFileError(f"{path.name}: file has a header but no rows")
    return frame


def _to_float(values: pd.Series) -> np.ndarray:
    """Numeric column to float64. Thousands commas are removed; '-' or blank become NaN.

    read_csv(thousands=",") is not used because it does nothing to a column holding even
    one non-numeric entry, which would silently turn every value above 999 into NaN.
    """
    text = values.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(text, errors="coerce").to_numpy(dtype=np.float64)


def parse_daily_dates(values: pd.Series, source: str) -> np.ndarray:
    """Parse a date column to datetime64[D] using an explicit format.

    Formats are tried whole-column, so a day/month ambiguity cannot be parsed row by row.
    """
    text = values.astype(str).str.strip()
    formats = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%b-%Y", "%d-%b-%y", "%d %b %Y", "%b %d, %Y", "%d-%m-%Y", "%d/%m/%Y", "%Y%m%d")
    for fmt in formats:
        try:
            return pd.to_datetime(text, format=fmt).to_numpy().astype("datetime64[D]")
        except ValueError:
            continue
    raise DataFileError(f"{source}: unrecognised date format, e.g. {text.iloc[0]!r}")


@dataclass(frozen=True)
class MinuteData:
    timestamps: np.ndarray  # datetime64[m], IST wall clock
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


def load_minute(raw_dir: Path) -> MinuteData:
    path = _single_file(raw_dir, "minute")
    frame = _read_csv(path, ["date", "open", "high", "low", "close"], usecols=lambda c: str(c).strip().lower() != "volume")
    try:
        stamps = pd.to_datetime(frame["date"], format="ISO8601")
    except ValueError as exc:
        raise DataFileError(f"{path.name}: cannot parse 'date' as timestamps ({exc})") from exc
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return MinuteData(
        stamps.to_numpy().astype("datetime64[m]"),
        *(_to_float(frame[c]) for c in ("open", "high", "low", "close")),
    )


@dataclass(frozen=True)
class DatedSeries:
    """A daily series: strictly increasing dates and positive, finite values."""

    dates: np.ndarray  # datetime64[D]
    values: np.ndarray


def clean_series(dates: np.ndarray, values: np.ndarray) -> DatedSeries:
    """Sort by date, keep the last of duplicate dates, drop missing or non-positive values."""
    order = np.argsort(dates, kind="stable")
    dates, values = dates[order], values[order]
    last_of_date = np.append(dates[1:] != dates[:-1], True)
    valid = last_of_date & np.isfinite(values) & (values > 0)
    return DatedSeries(dates[valid], values[valid])


def _load_daily_series(raw_dir: Path, key: str, value_column: str | None) -> DatedSeries:
    """Load one or more daily files (NSE downloads are often split by year) into one series."""
    files = find_source(raw_dir, key)
    if not files:
        raise DataFileError(f"Missing file for '{key}' in {raw_dir}. Expected: {SOURCES[key].description}")
    all_dates, all_values = [], []
    for path in files:
        frame = _read_csv(path, ["date"])
        # NSE labels the column e.g. "Turnover (Rs. Cr)", so turnover is matched by substring.
        matches = [c for c in frame.columns if (c == value_column if value_column else "turnover" in c)]
        if not matches:
            wanted = value_column or "a column containing 'turnover'"
            raise DataFileError(f"{path.name}: missing column {wanted}. Columns found: {list(frame.columns)}")
        all_dates.append(parse_daily_dates(frame["date"], path.name))
        all_values.append(_to_float(frame[matches[0]]))
    series = clean_series(np.concatenate(all_dates), np.concatenate(all_values))
    if series.dates.size == 0:
        raise DataFileError(f"{[p.name for p in files]}: no rows with a positive numeric value")
    return series


def load_vix(raw_dir: Path) -> DatedSeries:
    return _load_daily_series(raw_dir, "vix", "close")


def load_turnover(raw_dir: Path) -> DatedSeries | None:
    """The turnover series, or None if the optional file is absent."""
    return _load_daily_series(raw_dir, "turnover", None) if find_source(raw_dir, "turnover") else None


def load_index_closes(raw_dir: Path) -> DatedSeries | None:
    """NSE's published daily closes, or None if the optional files are absent."""
    return _load_daily_series(raw_dir, "index_close", "close") if find_source(raw_dir, "index_close") else None


@dataclass(frozen=True)
class Headlines:
    dates: np.ndarray  # datetime64[D]
    categories: np.ndarray  # object array of str
    texts: np.ndarray  # object array of str


def load_business_headlines(
    raw_dir: Path, start: np.datetime64, end: np.datetime64, chunk_rows: int = 500_000
) -> Headlines:
    """Headlines whose category contains 'business', dated start..end inclusive."""
    path = _single_file(raw_dir, "headlines")
    columns = ["publish_date", "headline_category", "headline_text"]
    _read_csv(path, columns, nrows=5)  # header check before streaming a file of several GB
    dates, categories, texts = [], [], []
    for chunk in pd.read_csv(path, usecols=columns, dtype=str, chunksize=chunk_rows, keep_default_na=False):
        keep = np.fromiter(("business" in c.lower() for c in chunk["headline_category"]), bool, len(chunk))
        if not keep.any():
            continue
        chunk_dates = parse_daily_dates(chunk["publish_date"], path.name)[keep]
        in_window = (chunk_dates >= start) & (chunk_dates <= end)
        dates.append(chunk_dates[in_window])
        categories.append(chunk["headline_category"].to_numpy(dtype=object)[keep][in_window])
        texts.append(chunk["headline_text"].to_numpy(dtype=object)[keep][in_window])
    if not dates or sum(map(len, dates)) == 0:
        raise DataFileError(f"{path.name}: no headline has a category containing 'business' in {start}..{end}")
    return Headlines(np.concatenate(dates), np.concatenate(categories), np.concatenate(texts))


def load_phrasebank(raw_dir: Path) -> tuple[list[str], list[str]]:
    """(sentences, labels) from the 'sentence@label' PhraseBank text file (latin-1 encoded)."""
    path = _single_file(raw_dir, "phrasebank")
    sentences, labels = [], []
    for line_number, line in enumerate(path.read_text(encoding="latin-1").splitlines(), start=1):
        if not line.strip():
            continue
        sentence, separator, label = line.rpartition("@")
        label = label.strip().lower()
        if not separator or label not in {"negative", "neutral", "positive"}:
            raise DataFileError(f"{path.name}:{line_number}: expected 'sentence@label', got {line[:80]!r}")
        sentences.append(sentence.strip())
        labels.append(label)
    return sentences, labels


PLACEHOLDER_MARKER = "TODO"


def last_thursdays(first_year: int, last_year: int) -> np.ndarray:
    """The last Thursday of every month in the given years."""
    months = np.arange(f"{first_year}-01", f"{last_year + 1}-01", dtype="datetime64[M]")
    month_ends = (months + 1).astype("datetime64[D]") - 1
    # 1970-01-01 was a Thursday, so days-since-epoch modulo 7 is days since the last Thursday.
    return month_ends - (month_ends.astype(np.int64) % 7)


def write_placeholder_expiry_file(path: Path = EXPIRY_FILE, first_year: int = 2015, last_year: int = 2024) -> None:
    """Create the default expiry calendar: last Thursday of each month."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {PLACEHOLDER_MARKER}: placeholder calendar (last Thursday of each month).",
        "# Replace with actual expiry dates from NSE circulars, then delete these comment lines.",
        "# Real expiries move when the Thursday is a holiday, and weekly expiries are not listed here.",
        "date",
        *(str(d) for d in last_thursdays(first_year, last_year)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_expiry_dates(path: Path = EXPIRY_FILE) -> tuple[np.ndarray, bool]:
    """(expiry dates, is_placeholder). Creates the placeholder file on first use."""
    if not path.exists():
        write_placeholder_expiry_file(path)
    is_placeholder = PLACEHOLDER_MARKER in path.read_text(encoding="utf-8").split("date", 1)[0]
    frame = _read_csv(path, ["date"], comment="#")
    return np.unique(parse_daily_dates(frame["date"], path.name)), is_placeholder


def write_table(path: Path, columns: dict[str, np.ndarray], float_format: str | None = None) -> None:
    """Write named NumPy columns to a CSV. Missing values are written as empty fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(path, index=False, float_format=float_format)


def append_columns(path: Path, columns: dict[str, np.ndarray]) -> None:
    """Add columns on the right of a CSV, leaving the text of every other cell as it is.

    A round trip through read_table and write_table could change a float's last digit, because pandas
    parses them quickly rather than exactly. A column that's already there is replaced, so a rerun is safe.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    header, *rows = [line.split(",") for line in lines]
    if any(len(row) != len(header) for row in rows):
        raise ValueError(f"{path.name} has a row with a different number of cells from its header")
    for name, values in columns.items():
        if values.size != len(rows):
            raise ValueError(f"Column {name} has {values.size} values, but {path.name} has {len(rows)} rows")
    keep = [position for position, name in enumerate(header) if name not in columns]
    new_cells = [[repr(value) for value in values.tolist()] for values in columns.values()]
    output = [",".join([header[position] for position in keep] + list(columns))]
    for number, row in enumerate(rows):
        output.append(",".join([row[position] for position in keep] + [cells[number] for cells in new_cells]))
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def read_table(path: Path, date_column: str = "date") -> dict[str, np.ndarray]:
    """Read a CSV written by write_table back into NumPy columns, parsing the date column."""
    frame = _read_csv(path, [date_column])
    table = {c: frame[c].to_numpy() for c in frame.columns}
    table[date_column] = parse_daily_dates(frame[date_column], path.name)
    return table
