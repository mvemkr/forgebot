"""
Set & Forget Strategy — Main Coordinator

Implements the complete strategy from the $100→$1M Forex journey research.

Decision flow:
  1. Session filter (hard block if Sunday night / Asian session)
  2. Multi-timeframe trend alignment (weekly + daily + 4H must agree)
  3. Key level proximity (price must be near a high-confluence level)
  4. Pattern detection (H&S, double top, break & retest)
  5. Engulfing candle confirmation (THE entry signal — no exceptions)
  6. Risk calculation (stop loss placement, position sizing)
  7. Output: TradeDecision (ENTER / WAIT / NO_TRADE)

The default action is always WAIT or NO_TRADE.
ENTER requires ALL filters to pass.
This is the patience filter encoded in software.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
import logging

from .session_filter import SessionFilter
from .level_detector import LevelDetector, KeyLevel
from .pattern_detector import PatternDetector, PatternResult, Trend
from .entry_signal import EntrySignalDetector, EntrySignal

logger = logging.getLogger(__name__)


class Decision(Enum):
    ENTER    = "ENTER"      # All conditions met — take the trade
    WAIT     = "WAIT"       # Setup is forming — monitor closely
    NO_TRADE = "NO_TRADE"   # Setup doesn't qualify — do nothing
    BLOCKED  = "BLOCKED"    # Hard block (session, existing trade, etc.)


@dataclass
class TradeDecision:
    decision: Decision
    pair: str
    direction: Optional[str]          # 'long' or 'short'
    reason: str                        # human-readable explanation
    confidence: float                  # 0.0–1.0

    # Entry details (only set when decision == ENTER)
    entry_price: Optional[float] = None
    stop_loss: Optional[float]   = None
    target_1: Optional[float]    = None
    target_2: Optional[float]    = None
    risk_pct: Optional[float]    = None   # % of account to risk
    lot_size: Optional[float]    = None

    # Context
    nearest_level: Optional[KeyLevel]      = None
    pattern: Optional[PatternResult]       = None
    entry_signal: Optional[EntrySignal]    = None
    trend_weekly: Optional[Trend]          = None
    trend_daily: Optional[Trend]           = None
    trend_4h: Optional[Trend]             = None
    failed_filters: List[str]             = field(default_factory=list)

    def __str__(self):
        lines = [
            f"{'='*60}",
            f"PAIR: {self.pair}  |  DECISION: {self.decision.value}  |  CONFIDENCE: {self.confidence:.0%}",
            f"{'='*60}",
            f"REASON: {self.reason}",
        ]
        if self.direction:
            lines.append(f"DIRECTION: {self.direction.upper()}")
        if self.entry_price:
            lines.append(f"ENTRY: {self.entry_price:.5f}")
        if self.stop_loss:
            lines.append(f"STOP LOSS: {self.stop_loss:.5f}")
        if self.target_1:
            lines.append(f"TARGET 1: {self.target_1:.5f}")
        if self.target_2:
            lines.append(f"TARGET 2: {self.target_2:.5f}")
        if self.risk_pct:
            lines.append(f"RISK: {self.risk_pct:.1f}% of account")
        if self.lot_size:
            lines.append(f"LOT SIZE: {self.lot_size:.2f}")
        if self.failed_filters:
            lines.append(f"FAILED FILTERS: {', '.join(self.failed_filters)}")
        if self.trend_weekly:
            lines.append(f"TRENDS → Weekly: {self.trend_weekly.value} | Daily: {self.trend_daily.value} | 4H: {self.trend_4h.value}")
        if self.nearest_level:
            lines.append(f"LEVEL: {self.nearest_level.price:.5f} ({self.nearest_level.level_type}) score={self.nearest_level.score} confluences={self.nearest_level.confluences}")
        if self.pattern:
            lines.append(f"PATTERN: {self.pattern.pattern_type} clarity={self.pattern.clarity:.2f}")
        if self.entry_signal:
            lines.append(f"SIGNAL: {self.entry_signal.signal_type} strength={self.entry_signal.strength:.2f}")
        lines.append(f"{'='*60}")
        return '\n'.join(lines)


class SetAndForgetStrategy:
    """
    The complete Set & Forget Forex strategy.

    Usage:
        strategy = SetAndForgetStrategy(account_balance=1000.0)
        decision = strategy.evaluate(
            pair="EUR/USD",
            df_weekly=df_w,
            df_daily=df_d,
            df_4h=df_4h,
            df_1h=df_1h,
        )
        print(decision)

    Parameters
    ----------
    account_balance : float
        Current account balance in USD
    risk_pct : float
        % of account to risk per trade (default 1.0%)
    max_level_distance_pct : float
        Max distance from key level to consider setup active (default 0.3%)
    min_level_score : int
        Minimum confluence score for a key level (default 2)
    min_pattern_clarity : float
        Minimum pattern clarity score (default 0.4)
    min_signal_strength : float
        Minimum engulfing candle strength (default 0.4)
    require_all_trends_aligned : bool
        If True, weekly+daily+4H must ALL agree. If False, 2/3 minimum.
    """

    def __init__(
        self,
        account_balance: float = 1000.0,
        risk_pct: float = 1.0,
        max_level_distance_pct: float = 0.3,
        min_level_score: int = 2,
        min_pattern_clarity: float = 0.4,
        min_signal_strength: float = 0.40,
        require_all_trends_aligned: bool = False,  # 2/3 is more forgiving, matches research
    ):
        self.account_balance    = account_balance
        self.risk_pct           = risk_pct
        self.max_level_dist     = max_level_distance_pct
        self.min_level_score    = min_level_score
        self.min_pattern_clarity = min_pattern_clarity
        self.min_signal_strength = min_signal_strength
        self.require_all_aligned = require_all_trends_aligned

        self.session_filter  = SessionFilter()
        self.level_detector  = LevelDetector(min_confluence=min_level_score)
        self.pattern_detector = PatternDetector()
        self.signal_detector  = EntrySignalDetector(min_body_ratio=0.45)

        # Track open positions (one at a time rule)
        self.open_positions: Dict[str, dict] = {}

    def update_balance(self, balance: float):
        self.account_balance = balance

    def evaluate(
        self,
        pair: str,
        df_weekly: pd.DataFrame,
        df_daily: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        current_price: Optional[float] = None,
    ) -> TradeDecision:
        """
        Run the complete strategy evaluation for a pair.
        Returns a TradeDecision with full context.
        """
        if current_price is None:
            current_price = df_1h['close'].iloc[-1]

        failed_filters = []

        # ─────────────────────────────────────────────
        # FILTER 0: One trade at a time
        # ─────────────────────────────────────────────
        if self.open_positions:
            open_pair = list(self.open_positions.keys())[0]
            return TradeDecision(
                decision=Decision.BLOCKED,
                pair=pair,
                direction=None,
                reason=f"Position already open: {open_pair}. One trade at a time — non-negotiable.",
                confidence=0.0,
                failed_filters=["one_trade_rule"],
            )

        # ─────────────────────────────────────────────
        # FILTER 1: Session filter (hard block)
        # ─────────────────────────────────────────────
        entry_allowed, session_reason = self.session_filter.is_entry_allowed()
        if not entry_allowed:
            return TradeDecision(
                decision=Decision.BLOCKED,
                pair=pair,
                direction=None,
                reason=f"Session blocked: {session_reason}",
                confidence=0.0,
                failed_filters=["session"],
            )

        session, session_quality = self.session_filter.session_quality()

        # ─────────────────────────────────────────────
        # FILTER 2: Multi-timeframe trend alignment
        # ─────────────────────────────────────────────
        trend_w = self.pattern_detector.detect_trend(df_weekly)
        trend_d = self.pattern_detector.detect_trend(df_daily)
        trend_4 = self.pattern_detector.detect_trend(df_4h)

        bullish_trends = [t for t in [trend_w, trend_d, trend_4]
                         if t in (Trend.BULLISH, Trend.STRONG_BULLISH)]
        bearish_trends = [t for t in [trend_w, trend_d, trend_4]
                         if t in (Trend.BEARISH, Trend.STRONG_BEARISH)]

        required = 3 if self.require_all_aligned else 2

        if len(bullish_trends) >= required:
            trade_direction = 'long'
        elif len(bearish_trends) >= required:
            trade_direction = 'short'
        else:
            # No clear alignment — WAIT but continue analysis
            trade_direction = None
            failed_filters.append("trend_alignment")
            logger.debug(f"{pair}: Trends not aligned. W={trend_w.value} D={trend_d.value} 4H={trend_4.value}")

        # ─────────────────────────────────────────────
        # FILTER 3: Key level proximity
        # ─────────────────────────────────────────────
        levels = self.level_detector.detect(df_daily, current_price)

        # Find nearest level matching trade direction (or any level if direction unknown)
        nearest_level = None
        for level in levels:
            if level.distance_pct > self.max_level_dist:
                continue
            if level.score < self.min_level_score:
                continue
            if trade_direction == 'long' and level.level_type != 'support':
                continue
            if trade_direction == 'short' and level.level_type != 'resistance':
                continue
            nearest_level = level
            break

        if nearest_level is None:
            failed_filters.append("no_key_level")
            # If no level, we're not in a zone — return WAIT
            if trade_direction is None or "trend_alignment" in failed_filters:
                return TradeDecision(
                    decision=Decision.NO_TRADE,
                    pair=pair,
                    direction=trade_direction,
                    reason=f"No key level within {self.max_level_dist}% and trends not aligned.",
                    confidence=0.0,
                    failed_filters=failed_filters,
                    trend_weekly=trend_w,
                    trend_daily=trend_d,
                    trend_4h=trend_4,
                )
            return TradeDecision(
                decision=Decision.WAIT,
                pair=pair,
                direction=trade_direction,
                reason=f"Trend aligned ({trade_direction}) but no key level within {self.max_level_dist}%. Watching for price to approach a level.",
                confidence=0.3,
                failed_filters=failed_filters,
                trend_weekly=trend_w,
                trend_daily=trend_d,
                trend_4h=trend_4,
            )

        # Level found but trend not aligned — still useful info
        if "trend_alignment" in failed_filters:
            return TradeDecision(
                decision=Decision.WAIT,
                pair=pair,
                direction=None,
                reason=f"Key level found ({nearest_level}) but trends not aligned across timeframes.",
                confidence=0.25,
                failed_filters=failed_filters,
                nearest_level=nearest_level,
                trend_weekly=trend_w,
                trend_daily=trend_d,
                trend_4h=trend_4,
            )

        # ─────────────────────────────────────────────
        # FILTER 4: Pattern detection
        # ─────────────────────────────────────────────
        patterns = self.pattern_detector.detect_all(df_daily)

        # Find best pattern matching trade direction
        matching_pattern = None
        for p in patterns:
            if p.direction == ('bearish' if trade_direction == 'short' else 'bullish'):
                if p.clarity >= self.min_pattern_clarity:
                    matching_pattern = p
                    break

        if matching_pattern is None:
            failed_filters.append("no_pattern")
            # Pattern missing: still a WAIT (level + trend aligned is meaningful)
            return TradeDecision(
                decision=Decision.WAIT,
                pair=pair,
                direction=trade_direction,
                reason=f"Level + trend aligned but no qualifying pattern (H&S/double top/break-retest) detected. Continue monitoring.",
                confidence=0.45,
                failed_filters=failed_filters,
                nearest_level=nearest_level,
                trend_weekly=trend_w,
                trend_daily=trend_d,
                trend_4h=trend_4,
            )

        # ─────────────────────────────────────────────
        # FILTER 5: Engulfing candle — THE PATIENCE FILTER
        # THE most critical check. No signal = NO ENTRY. Ever.
        # ─────────────────────────────────────────────
        has_signal, signal = self.signal_detector.has_signal(df_1h, trade_direction)

        if not has_signal or signal is None:
            # Setup is valid but we're waiting for the candle
            # This is the correct state 90% of the time
            return TradeDecision(
                decision=Decision.WAIT,
                pair=pair,
                direction=trade_direction,
                reason=(
                    f"Setup ACTIVE — {matching_pattern.pattern_type} at level {nearest_level.price:.5f} "
                    f"(score={nearest_level.score}). "
                    f"Trend aligned. WAITING FOR ENGULFING CANDLE. Do not enter without it."
                ),
                confidence=0.70,
                failed_filters=["awaiting_entry_signal"],
                nearest_level=nearest_level,
                pattern=matching_pattern,
                trend_weekly=trend_w,
                trend_daily=trend_d,
                trend_4h=trend_4,
            )

        if signal.strength < self.min_signal_strength:
            return TradeDecision(
                decision=Decision.WAIT,
                pair=pair,
                direction=trade_direction,
                reason=f"Entry signal detected but too weak (strength={signal.strength:.2f}, min={self.min_signal_strength}). Wait for a stronger candle.",
                confidence=0.55,
                failed_filters=["weak_signal"],
                nearest_level=nearest_level,
                pattern=matching_pattern,
                entry_signal=signal,
                trend_weekly=trend_w,
                trend_daily=trend_d,
                trend_4h=trend_4,
            )

        # ─────────────────────────────────────────────
        # ALL FILTERS PASSED → CALCULATE ENTRY
        # ─────────────────────────────────────────────
        entry_price, stop_loss, target_1, target_2, lot_size = self._calculate_entry(
            current_price=current_price,
            direction=trade_direction,
            level=nearest_level,
            pattern=matching_pattern,
        )

        # Final confidence score
        confidence = (
            0.30 * min(1.0, nearest_level.score / 4)  # level quality
          + 0.25 * matching_pattern.clarity             # pattern quality
          + 0.25 * signal.strength                      # signal quality
          + 0.20 * session_quality                      # session quality
        )

        risk_dollars = self.account_balance * (self.risk_pct / 100)
        rr_1 = abs(entry_price - target_1) / max(abs(entry_price - stop_loss), 0.000001)

        return TradeDecision(
            decision=Decision.ENTER,
            pair=pair,
            direction=trade_direction,
            reason=(
                f"ALL FILTERS PASSED. "
                f"{matching_pattern.pattern_type.upper()} pattern | "
                f"Level {nearest_level.price:.5f} score={nearest_level.score} | "
                f"{signal.signal_type} (strength={signal.strength:.2f}) | "
                f"Trend: W={trend_w.value} D={trend_d.value} 4H={trend_4.value} | "
                f"Session: {session} | "
                f"R:R (T1) = 1:{rr_1:.1f} | "
                f"Risk: ${risk_dollars:.2f}"
            ),
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            risk_pct=self.risk_pct,
            lot_size=lot_size,
            nearest_level=nearest_level,
            pattern=matching_pattern,
            entry_signal=signal,
            trend_weekly=trend_w,
            trend_daily=trend_d,
            trend_4h=trend_4,
        )

    def register_open_position(self, pair: str, entry_price: float, stop_loss: float, direction: str):
        """Call this when a trade is actually opened."""
        self.open_positions[pair] = {
            'entry': entry_price,
            'stop': stop_loss,
            'direction': direction,
        }
        logger.info(f"Position opened: {pair} {direction} at {entry_price:.5f} SL={stop_loss:.5f}")

    def close_position(self, pair: str, exit_price: float):
        """Call this when a trade is closed."""
        if pair in self.open_positions:
            pos = self.open_positions.pop(pair)
            pnl = (exit_price - pos['entry']) * (1 if pos['direction'] == 'long' else -1)
            logger.info(f"Position closed: {pair} at {exit_price:.5f} P&L direction = {'+' if pnl > 0 else ''}{pnl:.5f}")

    def move_stop_to_breakeven(self, pair: str):
        """
        Move stop to breakeven — ONLY allowed when price has moved 1:1 in our favor.
        Never move stop FURTHER from entry (that was the $50K lesson).
        """
        if pair in self.open_positions:
            pos = self.open_positions[pair]
            pos['stop'] = pos['entry']
            logger.info(f"Stop moved to breakeven for {pair} at {pos['entry']:.5f}")

    # ------------------------------------------------------------------ #
    # Internal: Entry Calculation
    # ------------------------------------------------------------------ #

    def _calculate_entry(
        self,
        current_price: float,
        direction: str,
        level: KeyLevel,
        pattern: PatternResult,
    ) -> tuple:
        """
        Returns (entry_price, stop_loss, target_1, target_2, lot_size).
        """
        # Entry: current price (limit order near current close)
        entry = current_price

        # Stop loss: use pattern's stop if it makes sense, otherwise use level
        if direction == 'short':
            sl_pattern = pattern.stop_loss
            sl_level   = level.price * 1.003  # just above level
            stop_loss  = max(sl_pattern, sl_level)  # pick the safer (higher) one for shorts
        else:
            sl_pattern = pattern.stop_loss
            sl_level   = level.price * 0.997  # just below level
            stop_loss  = min(sl_pattern, sl_level)  # pick the safer (lower) one for longs

        # Targets: use pattern's targets
        target_1 = pattern.target_1
        target_2 = pattern.target_2

        # Position sizing: fixed fractional
        risk_dollars = self.account_balance * (self.risk_pct / 100)
        risk_per_pip = abs(entry - stop_loss)

        if risk_per_pip > 0:
            # Lot size: how many units such that risk_dollars = risk_per_pip * units
            # For forex: 1 standard lot = 100,000 units; 1 pip ≈ $10 for majors
            # Approximate: lot_size = risk_dollars / (risk_pips * 10)
            risk_pips  = risk_per_pip * 10000  # convert to pips (4-decimal pairs)
            lot_size   = risk_dollars / max(risk_pips * 10, 0.01)
            lot_size   = max(0.01, round(lot_size, 2))  # minimum 0.01 (micro lot)
        else:
            lot_size = 0.01

        return entry, stop_loss, target_1, target_2, lot_size


if __name__ == "__main__":
    # Demo: create fake multi-TF data and run evaluation
    import numpy as np

    np.random.seed(42)
    def make_df(n, start=1.3, drift=-0.0002, noise=0.003):
        closes = start + np.cumsum(np.random.randn(n) * noise + drift)
        return pd.DataFrame({
            'open':  closes - noise/2,
            'high':  closes + abs(np.random.randn(n) * noise),
            'low':   closes - abs(np.random.randn(n) * noise),
            'close': closes,
        })

    strategy = SetAndForgetStrategy(account_balance=10000.0, risk_pct=1.0)
    decision = strategy.evaluate(
        pair="EUR/USD",
        df_weekly=make_df(52),
        df_daily=make_df(200),
        df_4h=make_df(300),
        df_1h=make_df(100),
    )
    print(decision)
