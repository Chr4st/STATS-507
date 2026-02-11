"""Time utilities: timezone handling, session checks, formatting."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
DETROIT = ZoneInfo("America/Detroit")
CHICAGO = ZoneInfo("America/Chicago")
NEW_YORK = ZoneInfo("America/New_York")

_TZ_MAP: dict[str, ZoneInfo] = {
    "America/Detroit": DETROIT,
    "America/Chicago": CHICAGO,
    "America/New_York": NEW_YORK,
    "UTC": UTC,
}


def get_tz(name: str) -> ZoneInfo:
    """Resolve timezone by name, caching common ones."""
    if name in _TZ_MAP:
        return _TZ_MAP[name]
    tz = ZoneInfo(name)
    _TZ_MAP[name] = tz
    return tz


def now_utc() -> datetime:
    """Current UTC datetime."""
    return datetime.now(UTC)


def now_detroit() -> datetime:
    """Current Detroit (Eastern) datetime."""
    return datetime.now(DETROIT)


def utc_to_tz(dt: datetime, tz_name: str) -> datetime:
    """Convert UTC datetime to a named timezone."""
    return dt.astimezone(get_tz(tz_name))


def parse_time(t: str) -> time:
    """Parse HH:MM string to time object."""
    parts = t.split(":")
    return time(int(parts[0]), int(parts[1]))


def is_in_session(
    now: datetime,
    tz_name: str,
    electronic_open: str,
    electronic_close: str,
    pit_open: str,
    pit_close: str,
    break_start: str | None,
    break_end: str | None,
    holidays: list[str] | None = None,
) -> tuple[bool, str]:
    """Check if current time falls within an instrument's trading session.

    Returns (is_tradable, reason_or_session_type).
    """
    local_now = utc_to_tz(now, tz_name) if now.tzinfo == UTC else now.astimezone(get_tz(tz_name))
    date_str = local_now.strftime("%Y-%m-%d")

    if holidays and date_str in holidays:
        return False, "HOLIDAY"

    # Weekday check (0=Mon, 6=Sun). CME electronic: Sun evening–Fri afternoon.
    wd = local_now.weekday()
    current_t = local_now.time()

    e_open = parse_time(electronic_open)
    e_close = parse_time(electronic_close)
    p_open = parse_time(pit_open)
    p_close = parse_time(pit_close)

    # Saturday: closed
    if wd == 5:
        return False, "WEEKEND"
    # Sunday: only open after electronic_open
    if wd == 6:
        if current_t >= e_open:
            return True, "ELECTRONIC"
        return False, "WEEKEND"

    # Electronic session spans overnight (e.g., 19:00–07:45)
    if e_open > e_close:
        # Overnight: open if after e_open OR before e_close
        in_electronic = current_t >= e_open or current_t < e_close
    else:
        in_electronic = e_open <= current_t < e_close

    # Break check
    if break_start and break_end:
        b_start = parse_time(break_start)
        b_end = parse_time(break_end)
        if b_start <= current_t < b_end:
            return False, "SESSION_BREAK"

    # Pit session
    in_pit = p_open <= current_t < p_close

    if in_pit:
        return True, "PIT"
    if in_electronic:
        return True, "ELECTRONIC"

    # Friday after pit close: closed for weekend
    if wd == 4 and current_t >= p_close:
        return False, "WEEKEND"

    return False, "SESSION_CLOSED"


def next_session_open(
    now: datetime,
    tz_name: str,
    electronic_open: str,
    holidays: list[str] | None = None,
) -> datetime:
    """Compute next session open time."""
    tz = get_tz(tz_name)
    local_now = utc_to_tz(now, tz_name) if now.tzinfo == UTC else now.astimezone(tz)
    e_open = parse_time(electronic_open)

    candidate = local_now.replace(hour=e_open.hour, minute=e_open.minute, second=0, microsecond=0)
    if candidate <= local_now:
        candidate += timedelta(days=1)

    # Skip weekends and holidays
    for _ in range(10):
        wd = candidate.weekday()
        date_str = candidate.strftime("%Y-%m-%d")
        if wd < 6 and (not holidays or date_str not in holidays):
            # Sunday: electronic opens Sunday evening
            if wd == 6:
                continue
            return candidate.astimezone(UTC)
        candidate += timedelta(days=1)

    return candidate.astimezone(UTC)


def format_ts(dt: datetime) -> str:
    """ISO-8601 timestamp string."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
