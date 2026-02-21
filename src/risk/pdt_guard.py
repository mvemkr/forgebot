"""
Pattern Day Trading (PDT) Guard.
FINRA rule: accounts under $25,000 equity may not execute more than
3 day trades in any rolling 5-business-day window.
A day trade = opening AND closing the same position on the same calendar day.
Violation = account flagged PDT, restricted for 90 days.
"""

import logging
import json
from datetime import datetime, date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

PDT_LOG_PATH = Path(__file__).parent.parent.parent / "data/state/pdt_log.json"
PDT_THRESHOLD = 25_000.0   # Account equity must exceed this to bypass PDT rules
MAX_DAY_TRADES = 3         # Max allowed in rolling 5-business-day window


def _business_days_back(n: int) -> list[date]:
    """Return the last N business days including today."""
    days = []
    d = date.today()
    while len(days) < n:
        if d.weekday() < 5:  # Mon-Fri
            days.append(d)
        d -= timedelta(days=1)
    return days


class PDTGuard:
    """
    Tracks day trades and blocks entries that would trigger a PDT violation.
    A 'day trade' is recorded when a position is opened and closed same day.
    """

    def __init__(self, account_equity: float = 0.0):
        self.account_equity = account_equity
        PDT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._log = self._load_log()

    def _load_log(self) -> list[dict]:
        if PDT_LOG_PATH.exists():
            try:
                return json.loads(PDT_LOG_PATH.read_text())
            except Exception:
                pass
        return []

    def _save_log(self):
        PDT_LOG_PATH.write_text(json.dumps(self._log, indent=2, default=str))

    def _rolling_window_dates(self) -> list[str]:
        return [d.isoformat() for d in _business_days_back(5)]

    def day_trades_in_window(self) -> int:
        window = set(self._rolling_window_dates())
        return sum(1 for t in self._log if t.get('date') in window)

    def remaining_day_trades(self) -> int:
        if self.account_equity >= PDT_THRESHOLD:
            return 999  # PDT rules don't apply
        used = self.day_trades_in_window()
        return max(0, MAX_DAY_TRADES - used)

    def can_day_trade(self) -> bool:
        """Returns True if another day trade is allowed."""
        if self.account_equity >= PDT_THRESHOLD:
            return True
        return self.remaining_day_trades() > 0

    def record_day_trade(self, symbol: str, strategy: str = ""):
        """Call this when a position is opened and closed on the same day."""
        entry = {
            'date':     date.today().isoformat(),
            'symbol':   symbol,
            'strategy': strategy,
            'used':     self.day_trades_in_window() + 1,
        }
        self._log.append(entry)
        self._save_log()
        remaining = self.remaining_day_trades()
        logger.warning(
            f"PDT: Day trade recorded — {symbol}. "
            f"Used: {entry['used']}/{MAX_DAY_TRADES} in rolling window. "
            f"Remaining: {remaining}"
        )
        if remaining == 0:
            logger.critical(
                "PDT: DAY TRADE LIMIT REACHED. No more same-day opens+closes until "
                f"{_business_days_back(5)[-1].isoformat()} rolls off the window."
            )

    def check_entry(self, symbol: str, intended_as_swing: bool = True) -> tuple[bool, str]:
        """
        Check before entering a position.
        intended_as_swing: if True, we plan to hold overnight (not a day trade).
        Returns (allowed, reason).
        """
        if self.account_equity >= PDT_THRESHOLD:
            return True, f"Account equity ${self.account_equity:,.0f} exceeds PDT threshold"

        remaining = self.remaining_day_trades()

        if intended_as_swing:
            # Swing trades are fine — but warn if we're close to the limit
            # because an unexpected same-day exit would consume a day trade
            if remaining == 0:
                return False, (
                    f"PDT BLOCK: 0 day trades remaining. Even swing entries are risky — "
                    f"if you need to exit same day (stop loss hits), it counts as a day trade."
                )
            if remaining == 1:
                logger.warning(
                    f"PDT WARNING: Only 1 day trade remaining. "
                    f"If this position is stopped out same-day, you'll hit the PDT limit."
                )
            return True, f"Swing trade allowed. {remaining} day trades remaining in window."
        else:
            # Explicit day trade
            if remaining <= 0:
                return False, f"PDT BLOCK: {MAX_DAY_TRADES} day trades already used in rolling 5-day window."
            return True, f"Day trade allowed. {remaining - 1} remaining after this one."

    def status(self) -> dict:
        window = self._rolling_window_dates()
        return {
            'account_equity':       self.account_equity,
            'pdt_applies':          self.account_equity < PDT_THRESHOLD,
            'day_trades_used':      self.day_trades_in_window(),
            'day_trades_remaining': self.remaining_day_trades(),
            'rolling_window':       window,
            'recent_trades':        [t for t in self._log if t.get('date') in set(window)],
        }
