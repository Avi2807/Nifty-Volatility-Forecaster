"""The weekly expiry calendar: Thursdays, then Tuesdays, moved back on holidays."""

from __future__ import annotations

import numpy as np

from src.data.expiry import weekly_expiry_dates
from src.data.features import weekday_index


def weekdays_between(start: str, end: str) -> np.ndarray:
    days = np.arange(np.datetime64(start), np.datetime64(end) + 1)
    return days[weekday_index(days) < 5]


def as_text(dates: np.ndarray) -> list[str]:
    return dates.astype(str).tolist()


def test_every_expiry_before_the_switch_is_a_thursday():
    dates = weekly_expiry_dates(weekdays_between("2023-01-02", "2023-03-31"))
    assert (weekday_index(dates) == 3).all()
    assert dates.size == 13


def test_the_switch_goes_from_thursday_28_august_to_tuesday_2_september_2025():
    dates = weekly_expiry_dates(weekdays_between("2025-08-18", "2025-09-12"))
    assert as_text(dates) == ["2025-08-21", "2025-08-28", "2025-09-02", "2025-09-09"]


def test_a_holiday_moves_the_expiry_to_the_previous_session():
    sessions = weekdays_between("2023-03-27", "2023-03-31")
    sessions = sessions[sessions != np.datetime64("2023-03-30")]  # Thursday holiday
    assert as_text(weekly_expiry_dates(sessions)) == ["2023-03-29"]


def test_a_holiday_after_the_switch_also_moves_back():
    sessions = weekdays_between("2025-09-08", "2025-09-12")
    sessions = sessions[sessions != np.datetime64("2025-09-09")]  # Tuesday holiday
    assert as_text(weekly_expiry_dates(sessions)) == ["2025-09-08"]
