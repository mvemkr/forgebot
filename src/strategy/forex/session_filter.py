"""
Session Filter — The Patience Gate

Blocks trades during low-quality sessions and restricted days/times.

Alex's time rules (from transcript):
  • No trades Sunday  — forex wick creation period, fills happen at worst prices
  • No trades Thursday ≥ 09:00 ET or any Friday — spread widens into week-end,
    positions often stopped at week-close wick.  "Not worth the spread."
  • Prefer Mon–Wed entries; London session (23:00–05:00 ET) optimal.

Implementation note: all hard-block reasons carry a unique reason code so the
backtester and orchestrator can count blocks separately in funnel reporting.
"""
from datetime import datetime, time
import pytz

from . import strategy_config as _cfg

ET  = pytz.timezone("America/New_York")
LON = pytz.timezone("Europe/London")


class SessionFilter:
    # Monday opening wick guard (unchanged from original)
    MONDAY_HARD_BLOCK_END   = time(8,  0)   # Monday 8AM ET

    # Preferred entry windows (ET)
    LONDON_OPEN_ET  = time(3,  0)   # 8AM London ≈ 3AM ET
    LONDON_CLOSE_ET = time(12, 0)
    NY_OPEN_ET      = time(8,  0)
    NY_CLOSE_ET     = time(12, 0)

    def __init__(self):
        pass

    def now_et(self) -> datetime:
        return datetime.now(ET)

    def is_hard_blocked(self, dt: datetime = None) -> tuple[bool, str]:
        """
        Returns (blocked: bool, reason: str).
        Hard blocks prevent ANY entry — non-negotiable.

        Reason codes (used by backtester funnel counter):
          NO_SUNDAY_TRADES    — all of Sunday
          NO_THU_FRI_TRADES   — Thursday ≥ THU_ENTRY_CUTOFF_HOUR_ET, all Friday
          MARKET_CLOSED       — Saturday
          MONDAY_WICK_GUARD   — Monday pre-8AM ET
        """
        if dt is None:
            dt = self.now_et()
        dt_et = dt.astimezone(ET)
        weekday = dt_et.weekday()   # 0=Mon … 6=Sun
        t       = dt_et.time()

        # Saturday — market closed
        if weekday == 5:
            return True, "MARKET_CLOSED: Saturday"

        # Sunday ALL DAY (Alex: wick creation, awful fills)
        if weekday == 6 and _cfg.NO_SUNDAY_TRADES_ENABLED:
            return True, "NO_SUNDAY_TRADES"

        # Monday before 8AM ET — weekly opening wick still printing
        if weekday == 0 and t < self.MONDAY_HARD_BLOCK_END:
            return True, "MONDAY_WICK_GUARD: pre-8AM ET"

        # Thursday > 09:00 ET — spread widens into week-end.
        # Alex rule: "allow entries only ≤ 09:00 NY (London/early NY ok), block after."
        # Use strict-greater-than so the 09:00 AM bar itself (entry at bar close ≈ 09:00)
        # is still allowed.  ≥ would block the 9AM bar which is still London/NY-overlap.
        if weekday == 3 and _cfg.NO_THU_FRI_TRADES_ENABLED:
            cutoff = time(_cfg.THU_ENTRY_CUTOFF_HOUR_ET, 0)
            if t > cutoff:
                return True, "NO_THU_FRI_TRADES: Thursday post-09:00 ET"

        # Friday ALL DAY (Alex: "not worth it")
        if weekday == 4 and _cfg.NO_THU_FRI_TRADES_ENABLED:
            return True, "NO_THU_FRI_TRADES: Friday"

        return False, ""

    def session_quality(self, dt: datetime = None) -> tuple[str, float]:
        """
        Returns (session_name: str, quality_score: 0.0–1.0).
        Used to weight signal confidence, not block trades outright.
        """
        if dt is None:
            dt = self.now_et()
        dt_et = dt.astimezone(ET)
        t = dt_et.time()

        # London + NY overlap (best)
        if time(8, 0) <= t < time(12, 0):
            return "LONDON_NY_OVERLAP", 1.0

        # Pure London (very good)
        if time(3, 0) <= t < time(8, 0):
            return "LONDON", 0.85

        # Late NY (decent)
        if time(12, 0) <= t < time(17, 0):
            return "NY_LATE", 0.6

        # Asian (avoid new entries)
        return "ASIAN", 0.3

    def next_entry_window(self, dt: datetime = None) -> tuple[str, int, datetime]:
        """
        Returns (session_name, minutes_until_open, next_session_ts_utc).
        minutes_until_open = 0 if currently inside a valid entry window.

        Probes two candidate times per day:
          • 03:00 ET  — London open (LONDON_OPEN_ET)
          • 08:00 ET  — London-NY overlap start / MONDAY_WICK_GUARD expiry

        The old implementation only probed 03:00 ET, causing it to skip
        Monday 08:00 ET entirely (blocked at 03:00 by MONDAY_WICK_GUARD) and
        report Tuesday as the next session — off by ~19 hours.

        Both candidates are tested per day; the earliest allowed one wins.
        """
        from datetime import timedelta, datetime as dt_cls
        import pytz as _pytz

        if dt is None:
            dt = datetime.now(ET)
        dt_et = dt.astimezone(ET)

        allowed, _ = self.is_entry_allowed(dt_et)
        if allowed:
            q, _ = self.session_quality(dt_et)
            ts_utc = dt_et.astimezone(_pytz.utc)
            return q, 0, ts_utc

        # Candidate session-open times to probe (ET hour, minute, label)
        # Order matters: earlier in the day first so we return the soonest slot.
        _candidates = [
            (self.LONDON_OPEN_ET.hour,    self.LONDON_OPEN_ET.minute,    "London"),
            (self.NY_OPEN_ET.hour,        self.NY_OPEN_ET.minute,        "London_NY_Overlap"),
        ]

        today = dt_et.date()
        best_candidate = None
        best_mins      = None

        for days_ahead in range(8):
            candidate_date = today + timedelta(days=days_ahead)
            day_best = None
            day_best_mins = None

            for hour, minute, label in _candidates:
                candidate = ET.localize(dt_cls(
                    candidate_date.year, candidate_date.month, candidate_date.day,
                    hour, minute, 0,
                ))
                if candidate <= dt_et:
                    continue   # already past
                allowed, _ = self.is_entry_allowed(candidate)
                if not allowed:
                    continue
                mins = int((candidate - dt_et).total_seconds() / 60)
                # Take the earliest allowed slot on this day
                if day_best_mins is None or mins < day_best_mins:
                    day_best      = candidate
                    day_best_mins = mins
                    day_label     = label

            if day_best is not None:
                # First day that has any allowed slot wins overall
                if best_mins is None or day_best_mins < best_mins:
                    best_candidate = day_best
                    best_mins      = day_best_mins
                    best_label     = day_label
                break   # found the nearest day — no need to look further

        if best_candidate is not None:
            ts_utc = best_candidate.astimezone(_pytz.utc)
            return best_label, best_mins, ts_utc

        # Fallback — should never be reached
        fallback_ts = (dt_et + timedelta(days=7)).astimezone(_pytz.utc)
        return "London", 999, fallback_ts

    def is_entry_allowed(self, dt: datetime = None) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Combines hard block + session quality check.

        Reason string on hard-block is the SPECIFIC code from is_hard_blocked()
        (e.g. "NO_THU_FRI_TRADES: Thursday post-09:00 ET") so callers can propagate
        it directly into failed_filters and the backtester counter can see it.
        """
        blocked, reason = self.is_hard_blocked(dt)
        if blocked:
            return False, reason   # ← specific code, not generic "session"

        session, quality = self.session_quality(dt)
        if quality < 0.5:
            return False, f"LOW_QUALITY_SESSION: {session} (score={quality:.1f})"

        return True, f"Entry allowed — {session} session (quality={quality:.1f})"


if __name__ == "__main__":
    sf = SessionFilter()
    blocked, reason = sf.is_hard_blocked()
    print(f"Hard blocked: {blocked} — {reason}")
    allowed, reason = sf.is_entry_allowed()
    print(f"Entry allowed: {allowed} — {reason}")
    session, quality = sf.session_quality()
    print(f"Session: {session} (quality={quality})")
