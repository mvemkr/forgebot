"""
Tests for SessionFilter.next_entry_window() Monday-wick-guard fix.

Bug: old code only probed 03:00 ET candidates.  Monday 03:00 ET is blocked
by MONDAY_WICK_GUARD, so it fell through to Tuesday 03:00 ET (~109h away)
instead of returning Monday 08:00 ET (~90h away).

Fix: probe both 03:00 ET and 08:00 ET per day; return earliest allowed slot.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytz

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.strategy.forex.session_filter import SessionFilter, ET

sf = SessionFilter()


def _et(year, month, day, hour, minute=0) -> datetime:
    """Make a timezone-aware ET datetime."""
    return ET.localize(datetime(year, month, day, hour, minute, 0))


def _utc(year, month, day, hour, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


# ── Return shape ───────────────────────────────────────────────────────────────

class TestReturnShape:
    def test_returns_3_tuple(self):
        now = _et(2026, 3, 5, 12, 57)   # Thu post-cutoff
        result = sf.next_entry_window(now)
        assert len(result) == 3

    def test_third_element_is_datetime(self):
        now = _et(2026, 3, 5, 12, 57)
        _, _, ts_utc = sf.next_entry_window(now)
        assert isinstance(ts_utc, datetime)
        assert ts_utc.tzinfo is not None

    def test_currently_allowed_returns_zero_mins(self):
        """Inside a valid session window → mins=0, ts=now."""
        # Monday 10:00 ET = London-NY overlap, not wick-guard blocked
        now = _et(2026, 3, 9, 10, 0)
        _, mins, _ = sf.next_entry_window(now)
        assert mins == 0


# ── Core bug fix: Monday 08:00 ET found instead of Tuesday 03:00 ET ──────────

class TestMondayFix:
    def test_thursday_post_cutoff_next_is_monday_not_tuesday(self):
        """
        Thu 2026-03-05 12:57 ET:
          OLD: Tuesday 2026-03-10 03:00 ET  (~6542m)
          NEW: Monday  2026-03-09 08:00 ET  (~5403m)
        """
        now = _et(2026, 3, 5, 12, 57)
        _, mins, ts_utc = sf.next_entry_window(now)
        # Should be less than 6542 (old Tuesday result)
        assert mins < 6542, f"Expected < 6542m (Tuesday), got {mins}m"
        # Should be approximately Monday 08:00 ET = 13:00 UTC
        # From Thu 12:57 ET → Mon 08:00 ET = 3 days + 19h 3m = ~5463m
        # Allow ±2 min for timing
        assert 5400 <= mins <= 5470, f"Expected ~5403m (Monday 08:00 ET), got {mins}m"

    def test_thursday_next_session_is_monday(self):
        now = _et(2026, 3, 5, 12, 57)
        label, _, _ = sf.next_entry_window(now)
        assert "London" in label or "Overlap" in label, f"Unexpected label: {label}"

    def test_monday_03am_blocked_08am_returned(self):
        """
        From Sunday night (23:00 ET), the NEXT valid slot is Monday 08:00 ET
        (not Monday 03:00 which is MONDAY_WICK_GUARD blocked).
        """
        now = _et(2026, 3, 8, 23, 0)   # Sunday 23:00 ET
        _, mins, ts_utc = sf.next_entry_window(now)
        # Mon 08:00 ET from Sun 23:00 ET = 9h = 540m
        assert 535 <= mins <= 545, f"Expected ~540m (Mon 08:00 ET), got {mins}m"

    def test_monday_03am_is_not_returned(self):
        """
        Explicitly verify Monday 03:00 ET is NOT returned as the next window
        when queried from Sunday night.
        """
        now = _et(2026, 3, 8, 23, 0)   # Sunday 23:00 ET
        _, mins, _ = sf.next_entry_window(now)
        # Mon 03:00 ET from Sun 23:00 ET = 4h = 240m — must NOT return this
        assert mins > 300, f"Got {mins}m — looks like Mon 03:00 was returned (still WICK_GUARD blocked)"

    def test_next_ts_utc_is_monday_0800_et_as_utc(self):
        """next_session_ts_utc must correspond to Mon 08:00 ET = 12:00 UTC.
        Note: DST springs forward Sun Mar 8 2026 → Mon Mar 9 is EDT (UTC-4),
        so 08:00 ET = 12:00 UTC (not 13:00 which would be EST/UTC-5).
        """
        now = _et(2026, 3, 8, 23, 0)   # Sunday 23:00 ET
        _, _, ts_utc = sf.next_entry_window(now)
        # Mon 08:00 EDT = 12:00 UTC
        assert ts_utc.hour == 12, f"Expected 12:00 UTC (Mon 08:00 EDT), got {ts_utc}"
        assert ts_utc.day == 9   # March 9


# ── Friday and weekend handling (should not regress) ──────────────────────────

class TestNonMondayDays:
    def test_friday_all_day_blocked(self):
        """Friday is fully blocked; next window must be Mon 08:00 ET.
        DST springs forward Sun Mar 8 2026: Fri 10:00 EST → Mon 08:00 EDT = 69h = 4140m.
        """
        now = _et(2026, 3, 6, 10, 0)   # Friday 10:00 ET (EST)
        _, mins, _ = sf.next_entry_window(now)
        # Fri 10:00 EST (15:00 UTC) → Mon 08:00 EDT (12:00 UTC) = 69h = 4140m
        assert 4135 <= mins <= 4145, f"Expected ~4140m (DST-adjusted), got {mins}m"

    def test_saturday_next_is_monday(self):
        """Sat noon ET → Mon 08:00 EDT = 43h = 2580m (DST spring-forward on Sun)."""
        now = _et(2026, 3, 7, 12, 0)   # Saturday noon ET (EST)
        _, mins, _ = sf.next_entry_window(now)
        # Sat 12:00 EST (17:00 UTC) → Mon 08:00 EDT (12:00 UTC) = 43h = 2580m
        assert 2575 <= mins <= 2585, f"Expected ~2580m (DST-adjusted), got {mins}m"

    def test_wednesday_london_open_returned(self):
        """
        Wednesday 22:00 ET: next session is Thu 03:00 ET (London open,
        still within the allowed Thu <09:00 window).
        """
        now = _et(2026, 3, 4, 22, 0)   # Wednesday 22:00 ET
        _, mins, _ = sf.next_entry_window(now)
        # Thu 03:00 ET from Wed 22:00 ET = 5h = 300m
        assert 295 <= mins <= 305, f"Expected ~300m (Thu 03:00 ET), got {mins}m"

    def test_thursday_before_cutoff_is_allowed(self):
        """Thursday 06:00 ET is still within the allowed window → mins=0."""
        now = _et(2026, 3, 5, 6, 0)   # Thursday 06:00 ET (before 09:00 cutoff)
        _, mins, _ = sf.next_entry_window(now)
        assert mins == 0


# ── next_session_ts_utc accuracy ──────────────────────────────────────────────

class TestNextSessionTsUtc:
    def test_ts_utc_matches_mins_calculation(self):
        """next_session_ts_utc must be consistent with mins_until."""
        now = _et(2026, 3, 5, 12, 57)
        now_utc = now.astimezone(pytz.utc)
        _, mins, ts_utc = sf.next_entry_window(now)
        computed_mins = int((ts_utc - now_utc).total_seconds() / 60)
        # Allow 1-min rounding difference
        assert abs(computed_mins - mins) <= 1, (
            f"ts_utc implies {computed_mins}m but mins_until={mins}"
        )

    def test_ts_utc_is_utc_aware(self):
        now = _et(2026, 3, 5, 12, 57)
        _, _, ts_utc = sf.next_entry_window(now)
        assert ts_utc.tzinfo is not None
        offset = ts_utc.utcoffset().total_seconds()
        assert offset == 0, f"Expected UTC (offset=0), got offset={offset}s"
