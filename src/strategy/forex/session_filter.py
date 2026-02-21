"""
Session Filter — The Patience Gate

Blocks trades during low-quality sessions and Sunday night.
Rule: No trades Sunday 5PM ET → Monday 8AM ET (weekly wick creation period).
Rule: Prefer London session (8AM–12PM London / 3AM–8AM ET) and NY session (8AM–12PM ET).
"""
from datetime import datetime, time
import pytz

ET  = pytz.timezone("America/New_York")
LON = pytz.timezone("Europe/London")


class SessionFilter:
    # Hard block: Sunday 5PM ET → Monday 8AM ET
    SUNDAY_HARD_BLOCK_START = time(17, 0)   # Sunday 5PM ET
    MONDAY_HARD_BLOCK_END   = time(8,  0)   # Monday 8AM ET

    # Preferred entry windows (ET)
    LONDON_OPEN_ET  = time(3,  0)   # 8AM London
    LONDON_CLOSE_ET = time(12, 0)   # 12PM London / NY open
    NY_OPEN_ET      = time(8,  0)
    NY_CLOSE_ET     = time(12, 0)   # Best overlap

    def __init__(self):
        pass

    def now_et(self) -> datetime:
        return datetime.now(ET)

    def is_hard_blocked(self, dt: datetime = None) -> tuple[bool, str]:
        """
        Returns (blocked: bool, reason: str).
        Hard blocks prevent ANY entry — non-negotiable.
        """
        if dt is None:
            dt = self.now_et()
        dt_et = dt.astimezone(ET)
        weekday = dt_et.weekday()  # 0=Mon, 6=Sun
        t = dt_et.time()

        # Sunday after 5PM ET
        if weekday == 6 and t >= self.SUNDAY_HARD_BLOCK_START:
            return True, "Sunday post-5PM ET — weekly wick creation, no entries"

        # Monday before 8AM ET
        if weekday == 0 and t < self.MONDAY_HARD_BLOCK_END:
            return True, "Monday pre-8AM ET — market still creating weekly wick"

        # Saturday entirely
        if weekday == 5:
            return True, "Saturday — market closed"

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

    def is_entry_allowed(self, dt: datetime = None) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Combines hard block + session quality check.
        """
        blocked, reason = self.is_hard_blocked(dt)
        if blocked:
            return False, reason

        session, quality = self.session_quality(dt)
        if quality < 0.5:
            return False, f"Low quality session ({session}, score={quality:.1f}) — wait for London/NY"

        return True, f"Entry allowed — {session} session (quality={quality:.1f})"


if __name__ == "__main__":
    sf = SessionFilter()
    blocked, reason = sf.is_hard_blocked()
    print(f"Hard blocked: {blocked} — {reason}")
    allowed, reason = sf.is_entry_allowed()
    print(f"Entry allowed: {allowed} — {reason}")
    session, quality = sf.session_quality()
    print(f"Session: {session} (quality={quality})")
