from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable


@dataclass(slots=True)
class TimeWindow:
    label: str
    start: datetime
    end: datetime


def parse_period(period: str) -> TimeWindow:
    value = period.strip()
    if "-H" in value:
        year_text, half_text = value.split("-H", 1)
        year = int(year_text)
        half = int(half_text)
        if half == 1:
            return TimeWindow(value, datetime(year, 1, 1, tzinfo=timezone.utc), datetime(year, 6, 30, 23, 59, 59, tzinfo=timezone.utc))
        if half == 2:
            return TimeWindow(value, datetime(year, 7, 1, tzinfo=timezone.utc), datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc))
    if "-Q" in value:
        year_text, quarter_text = value.split("-Q", 1)
        year = int(year_text)
        quarter = int(quarter_text)
        start_month = 1 + (quarter - 1) * 3
        end_month = start_month + 2
        start = datetime(year, start_month, 1, tzinfo=timezone.utc)
        if end_month == 12:
            end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        else:
            next_month = datetime(year, end_month + 1, 1, tzinfo=timezone.utc)
            end = next_month.replace(day=1) - timedelta(seconds=1)  # type: ignore[name-defined]
        return TimeWindow(value, start, end)
    if ":" in value:
        start_text, end_text = value.split(":", 1)
        start = _parse_iso_datetime(start_text)
        end = _parse_iso_datetime(end_text)
        return TimeWindow(value, start, end)
    raise ValueError(f"Unsupported period format: {period}")


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return _parse_iso_datetime(value)


def in_window(value: str | None, window: TimeWindow) -> bool:
    date_value = parse_datetime(value)
    if date_value is None:
        return False
    return window.start <= date_value <= window.end


def latest(values: Iterable[str | None]) -> str | None:
    parsed = [parse_datetime(value) for value in values if value]
    parsed = [value for value in parsed if value is not None]
    if not parsed:
        return None
    return max(parsed).isoformat()


def earliest(values: Iterable[str | None]) -> str | None:
    parsed = [parse_datetime(value) for value in values if value]
    parsed = [value for value in parsed if value is not None]
    if not parsed:
        return None
    return min(parsed).isoformat()


def _parse_iso_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
