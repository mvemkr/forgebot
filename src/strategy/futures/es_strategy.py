"""
ES Futures Strategy — ORB Breakout + Session High/Low Fade

Strategy 1 — Opening Range Breakout (ORB)
    • Capture the 8:00–8:15 AM ET (pre-market) 15-min candle as the opening range
    • Price breaks above orb_high + buffer  → LONG
    • Price breaks below orb_low  - buffer  → SHORT
    • Stop  = ORB_STOP_POINTS from entry
    • Target = stop × ORB_TARGET_MULTIPLIER
    • Valid until 11:00 AM ET only
    • Skip if range > ORB_MAX_RANGE_PTS or < ORB_MIN_RANGE_PTS

Strategy 2 — Session High/Low Fade
    • Track rolling 5-day session high/low from daily candles
    • Price touches session high → SHORT (fade the rally)
    • Price touches session low  → LONG  (fade the selloff)
    • Valid 10:00 AM – 3:00 PM ET only (avoids open/close volatility)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, time
from typing import Optional

import pandas as pd
import pytz

from . import es_config as cfg

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")

# ── Trade Setup ───────────────────────────────────────────────────────────────

@dataclass
class TradeSetup:
    """Fully-specified entry signal from ESFuturesStrategy."""
    strategy_type:  str              # "ORB_BREAKOUT" or "SESSION_FADE"
    direction:      str              # "long" or "short"
    entry_price:    float
    stop_price:     float
    target_price:   float
    risk_points:    float            # |entry - stop|
    reward_points:  float            # |target - entry|
    reason:         str
    timestamp:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def rr_ratio(self) -> float:
        return self.reward_points / self.risk_points if self.risk_points else 0.0


# ── Strategy ──────────────────────────────────────────────────────────────────

class ESFuturesStrategy:
    """
    Stateful ES futures intraday strategy.

    State is reset each session (call reset_session() at end of day).

    Parameters
    ----------
    schwab_client : SchwabClient | None
        Live Schwab client.  Pass None for offline / test use.
    """

    def __init__(self, schwab_client=None):
        self.client = schwab_client

        # ORB state
        self.orb_high:   Optional[float] = None
        self.orb_low:    Optional[float] = None
        self.orb_set:    bool            = False
        self.orb_fired:  bool            = False   # one trade per session per strategy

        # Fade state
        self.session_high: Optional[float] = None
        self.session_low:  Optional[float] = None
        self.fade_fired:   bool            = False

    # ── Session management ────────────────────────────────────────────────────

    def reset_session(self) -> None:
        """Call at end of each RTH session to clear intraday state."""
        self.orb_high   = None
        self.orb_low    = None
        self.orb_set    = False
        self.orb_fired  = False
        self.session_high = None
        self.session_low  = None
        self.fade_fired = False
        logger.info("ESFuturesStrategy: session state reset")

    # ── ORB ───────────────────────────────────────────────────────────────────

    def update_opening_range(self, candles_15m: pd.DataFrame) -> bool:
        """
        Set orb_high / orb_low from the 8:00–8:15 AM ET 15-min candle.

        Parameters
        ----------
        candles_15m : pd.DataFrame
            15-min OHLCV, UTC-indexed.  Must contain at least one bar.

        Returns True if opening range was captured (or was already set).
        """
        if self.orb_set:
            return True

        if candles_15m.empty:
            return False

        # Convert to ET and find the 8:00 AM bar
        df_et = candles_15m.copy()
        df_et.index = candles_15m.index.tz_convert(ET)

        orb_bars = df_et[
            (df_et.index.hour == cfg.PREMARKET_START_ET) &
            (df_et.index.minute == 0)
        ]

        if orb_bars.empty:
            return False

        bar = orb_bars.iloc[-1]
        rng = bar["high"] - bar["low"]

        if rng > cfg.ORB_MAX_RANGE_PTS:
            logger.info(f"ORB: range {rng:.1f}pts too wide (>{cfg.ORB_MAX_RANGE_PTS}) — skip")
            self.orb_set = True   # mark as set so we don't keep checking
            return False

        if rng < cfg.ORB_MIN_RANGE_PTS:
            logger.info(f"ORB: range {rng:.1f}pts too narrow (<{cfg.ORB_MIN_RANGE_PTS}) — skip")
            self.orb_set = True
            return False

        self.orb_high = float(bar["high"])
        self.orb_low  = float(bar["low"])
        self.orb_set  = True
        logger.info(
            f"ORB captured: high={self.orb_high:.2f}  low={self.orb_low:.2f}  "
            f"range={rng:.1f}pts"
        )
        return True

    def check_orb_breakout(
        self,
        current_price: float,
        now_utc: datetime,
    ) -> Optional[TradeSetup]:
        """
        Check if price has broken out of the opening range.

        Returns TradeSetup on breakout, None otherwise.
        Valid window: 8:15 AM – 11:00 AM ET.
        """
        if not self.orb_set or self.orb_high is None or self.orb_fired:
            return None

        now_et = now_utc.astimezone(ET)
        hour, minute = now_et.hour, now_et.minute

        # Valid: 8:15 AM to 11:00 AM ET
        after_orb   = (hour == 8 and minute >= 15) or (hour == 9) or (hour == 10)
        before_close = hour < 11
        if not (after_orb and before_close):
            return None

        breakout_long  = current_price > self.orb_high + cfg.ORB_BREAKOUT_BUFFER_PTS
        breakout_short = current_price < self.orb_low  - cfg.ORB_BREAKOUT_BUFFER_PTS

        if breakout_long:
            entry  = current_price
            stop   = entry - cfg.ORB_STOP_POINTS
            target = entry + cfg.ORB_STOP_POINTS * cfg.ORB_TARGET_MULTIPLIER
            self.orb_fired = True
            return TradeSetup(
                strategy_type = "ORB_BREAKOUT",
                direction     = "long",
                entry_price   = entry,
                stop_price    = stop,
                target_price  = target,
                risk_points   = cfg.ORB_STOP_POINTS,
                reward_points = cfg.ORB_STOP_POINTS * cfg.ORB_TARGET_MULTIPLIER,
                reason        = f"ORB LONG: price {current_price:.2f} > orb_high {self.orb_high:.2f} + buffer",
            )

        if breakout_short:
            entry  = current_price
            stop   = entry + cfg.ORB_STOP_POINTS
            target = entry - cfg.ORB_STOP_POINTS * cfg.ORB_TARGET_MULTIPLIER
            self.orb_fired = True
            return TradeSetup(
                strategy_type = "ORB_BREAKOUT",
                direction     = "short",
                entry_price   = entry,
                stop_price    = stop,
                target_price  = target,
                risk_points   = cfg.ORB_STOP_POINTS,
                reward_points = cfg.ORB_STOP_POINTS * cfg.ORB_TARGET_MULTIPLIER,
                reason        = f"ORB SHORT: price {current_price:.2f} < orb_low {self.orb_low:.2f} - buffer",
            )

        return None

    # ── Session Fade ──────────────────────────────────────────────────────────

    def check_session_fade(
        self,
        current_price: float,
        daily_candles:  pd.DataFrame,
        now_utc:        datetime,
    ) -> Optional[TradeSetup]:
        """
        Check if price has reached a 5-day session extreme (fade setup).

        Parameters
        ----------
        current_price : float
            Current ES/MES price
        daily_candles  : pd.DataFrame
            Daily OHLCV, last FADE_LOOKBACK_DAYS used.
        now_utc        : datetime
            Current time in UTC.

        Returns TradeSetup on fade trigger, None otherwise.
        Valid window: 10:00 AM – 3:00 PM ET.
        """
        if self.fade_fired:
            return None

        now_et = now_utc.astimezone(ET)
        hour   = now_et.hour

        # Valid: 10:00 AM – 3:00 PM ET
        if not (cfg.FADE_VALID_START_ET <= hour < cfg.FADE_VALID_END_ET):
            return None

        if daily_candles.empty:
            return None

        # Build session high/low from lookback window
        recent  = daily_candles.iloc[-(cfg.FADE_LOOKBACK_DAYS):]
        s_high  = float(recent["high"].max())
        s_low   = float(recent["low"].min())

        self.session_high = s_high
        self.session_low  = s_low

        at_high = current_price >= s_high - cfg.FADE_ENTRY_BUFFER_PTS
        at_low  = current_price <= s_low  + cfg.FADE_ENTRY_BUFFER_PTS

        if at_high:
            entry  = current_price
            stop   = entry + cfg.FADE_STOP_POINTS
            target = entry - cfg.FADE_STOP_POINTS * cfg.FADE_TARGET_MULTIPLIER
            self.fade_fired = True
            return TradeSetup(
                strategy_type = "SESSION_FADE",
                direction     = "short",
                entry_price   = entry,
                stop_price    = stop,
                target_price  = target,
                risk_points   = cfg.FADE_STOP_POINTS,
                reward_points = cfg.FADE_STOP_POINTS * cfg.FADE_TARGET_MULTIPLIER,
                reason        = f"FADE SHORT: price {current_price:.2f} at 5d-high {s_high:.2f}",
            )

        if at_low:
            entry  = current_price
            stop   = entry - cfg.FADE_STOP_POINTS
            target = entry + cfg.FADE_STOP_POINTS * cfg.FADE_TARGET_MULTIPLIER
            self.fade_fired = True
            return TradeSetup(
                strategy_type = "SESSION_FADE",
                direction     = "long",
                entry_price   = entry,
                stop_price    = stop,
                target_price  = target,
                risk_points   = cfg.FADE_STOP_POINTS,
                reward_points = cfg.FADE_STOP_POINTS * cfg.FADE_TARGET_MULTIPLIER,
                reason        = f"FADE LONG: price {current_price:.2f} at 5d-low {s_low:.2f}",
            )

        return None

    # ── Main entry point ──────────────────────────────────────────────────────

    def evaluate(self, now_utc: Optional[datetime] = None) -> Optional[TradeSetup]:
        """
        Main scan tick.  Fetches live price + candles, runs both strategies.

        Returns a TradeSetup if an entry signal fires, None otherwise.

        Requires self.client to be set.  In dry-run / test mode, call
        check_orb_breakout() and check_session_fade() directly.
        """
        if self.client is None:
            raise RuntimeError("evaluate() requires a live SchwabClient. "
                               "Use check_orb_breakout() / check_session_fade() directly in tests.")

        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        now_et = now_utc.astimezone(ET)
        hour   = now_et.hour

        # Outside trading window entirely
        if hour < cfg.PREMARKET_START_ET or hour >= cfg.RTH_CLOSE_ET:
            return None

        quote = self.client.get_es_quote()
        price = quote.get("mid") or quote.get("last") or 0.0
        if price == 0.0:
            logger.warning("evaluate(): zero price — skipping tick")
            return None

        # 1. Capture opening range at / after 8:15 AM
        if not self.orb_set and hour == cfg.PREMARKET_START_ET:
            candles_15m = self.client.get_es_candles(timeframe="15m", lookback=10)
            self.update_opening_range(candles_15m)

        # 2. ORB breakout (8:15–11 AM)
        setup = self.check_orb_breakout(price, now_utc)
        if setup:
            return setup

        # 3. Session fade (10 AM–3 PM)
        daily = self.client.get_es_candles(timeframe="1D", lookback=cfg.FADE_LOOKBACK_DAYS + 2)
        setup = self.check_session_fade(price, daily, now_utc)
        return setup
