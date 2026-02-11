"""Market session checker: determines tradability per instrument."""

from __future__ import annotations

from datetime import datetime

from geoag.common.config import ConfigStore, get_config
from geoag.common.logging import get_logger
from geoag.common.timeutils import format_ts, is_in_session, next_session_open, now_utc

logger = get_logger("signals.sessions")


class SessionChecker:
    """Check whether instruments are currently tradable."""

    def __init__(self, config: ConfigStore | None = None) -> None:
        self.config = config or get_config()

    def check(
        self, symbol: str, at: datetime | None = None
    ) -> dict[str, object]:
        """Check session status for an instrument.

        Returns dict with:
          - tradable_now: bool
          - session_type: str (ELECTRONIC, PIT, SESSION_CLOSED, etc.)
          - best_window: str | None (ISO timestamp of next open if closed)
        """
        spec = self.config.instruments.get(symbol)
        if spec is None:
            return {
                "tradable_now": False,
                "session_type": "UNKNOWN_INSTRUMENT",
                "best_window": None,
            }

        check_time = at or now_utc()
        holidays = self.config.get_holidays_for_exchange(spec.exchange)
        sess = spec.session

        tradable, session_type = is_in_session(
            now=check_time,
            tz_name=spec.timezone,
            electronic_open=sess.electronic_open,
            electronic_close=sess.electronic_close,
            pit_open=sess.pit_open,
            pit_close=sess.pit_close,
            break_start=sess.break_start,
            break_end=sess.break_end,
            holidays=holidays,
        )

        best_window: str | None = None
        if not tradable:
            nso = next_session_open(
                now=check_time,
                tz_name=spec.timezone,
                electronic_open=sess.electronic_open,
                holidays=holidays,
            )
            best_window = format_ts(nso)

        return {
            "tradable_now": tradable,
            "session_type": session_type,
            "best_window": best_window,
        }

    def check_all(
        self, at: datetime | None = None
    ) -> dict[str, dict[str, object]]:
        """Check all instruments."""
        return {sym: self.check(sym, at) for sym in self.config.instruments}
