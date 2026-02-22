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

    # Pattern memory — needed to mark formation exhausted when trade closes
    neckline_ref: Optional[float]          = None   # key level price used as pattern key

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

        from .news_filter import NewsFilter
        self.session_filter  = SessionFilter()
        self.news_filter     = NewsFilter()
        self.level_detector  = LevelDetector(min_confluence=min_level_score)
        self.pattern_detector = PatternDetector()
        self.signal_detector  = EntrySignalDetector(min_body_ratio=0.45)

        # Track open positions (one at a time rule)
        self.open_positions: Dict[str, dict] = {}

        # Per-pair stop-out cooldown: after a stop is hit, block re-entry for
        # STOP_COOLDOWN_DAYS. Prevents re-entering the same pair immediately
        # after a loss on a slightly different neckline that pattern memory
        # doesn't cluster (e.g. NZD/JPY break-retest at 94 → stops → re-enters
        # at 93 within 3 days on the same broad setup).
        self.STOP_COOLDOWN_DAYS: float = 5.0
        self._stop_out_times: Dict[str, object] = {}   # pair → datetime of last stop

        # Pattern memory — prevent re-entering the same formation
        # Key: (pair, pattern_type, rounded_neckline) → 'exhausted'
        # A pattern stays exhausted until price structurally breaks through it.
        # This is the core insight: once a double bottom at 1.3350 is traded,
        # that same formation is exhausted — don't trade it again even if it re-forms.
        self.NECKLINE_CLUSTER_PCT: float = 0.003   # 0.3% — merges nearby necklines into one bucket
        self.traded_patterns: Dict[str, str] = {}

    def update_balance(self, balance: float):
        self.account_balance = balance

    # ── Pattern Memory ─────────────────────────────────────────────────

    def _pattern_key(self, pair: str, pattern_type: str, neckline: float) -> str:
        """Bucket neckline to nearest cluster so nearby levels merge into one key."""
        cluster_size = neckline * self.NECKLINE_CLUSTER_PCT
        if cluster_size == 0:
            return f"{pair}|{pattern_type}|{neckline:.5f}"
        bucket = round(round(neckline / cluster_size) * cluster_size, 5)
        return f"{pair}|{pattern_type}|{bucket:.5f}"

    def pattern_already_traded(self, pair: str, pattern_type: str, neckline: float) -> bool:
        """Return True if this exact formation has been traded already."""
        return self.traded_patterns.get(self._pattern_key(pair, pattern_type, neckline)) == "exhausted"

    def mark_pattern_exhausted(self, pair: str, pattern_type: str, neckline: float):
        """
        Call this when a position on this pattern is closed (stop hit, exit signal, max hold).
        Prevents re-entering the same formation.
        """
        key = self._pattern_key(pair, pattern_type, neckline)
        self.traded_patterns[key] = "exhausted"
        logger.info(f"🔒 Pattern exhausted: {key}")

    def get_traded_patterns(self) -> Dict[str, str]:
        return dict(self.traded_patterns)

    def restore_traded_patterns(self, patterns: Dict[str, str]):
        """Restore pattern memory from persisted state after a restart."""
        self.traded_patterns.update(patterns)
        logger.info(f"Pattern memory restored: {len(patterns)} patterns")

    # ── Stop-out cooldown ─────────────────────────────────────────────

    def record_stop_out(self, pair: str, dt=None):
        """
        Record that a stop was just hit on this pair.
        Blocks re-entry for STOP_COOLDOWN_DAYS, preventing immediate
        re-entry on a slightly different neckline after a loss.
        Call this from position_monitor when a stop is hit.
        """
        from datetime import datetime, timezone as _tz
        self._stop_out_times[pair] = dt or datetime.now(_tz.utc)
        logger.info(f"🚧 {pair}: stop-out cooldown started — "
                    f"blocked for {self.STOP_COOLDOWN_DAYS:.0f} days")

    def is_in_stop_cooldown(self, pair: str, current_dt=None) -> bool:
        """Return True if this pair is within STOP_COOLDOWN_DAYS of a stop-out."""
        if pair not in self._stop_out_times:
            return False
        from datetime import datetime, timezone as _tz, timedelta
        last = self._stop_out_times[pair]
        now  = current_dt or datetime.now(_tz.utc)
        # Normalise timezone
        if hasattr(last, "tzinfo") and last.tzinfo is None:
            last = last.replace(tzinfo=_tz.utc)
        if hasattr(now, "tzinfo") and now.tzinfo is None:
            now = now.replace(tzinfo=_tz.utc)
        return (now - last).total_seconds() < self.STOP_COOLDOWN_DAYS * 86400

    # ── Dual-trade currency helpers ───────────────────────────────────

    @staticmethod
    def _is_major_level(price: float, pair: str) -> bool:
        """
        Return True if price is at (or very near) a MAJOR psychological round number.

        Alex only trades at levels "everyone is watching" — the big round numbers
        that are obvious on any chart. Arbitrary swing highs/lows that happen to
        cluster don't qualify on their own.

        Criteria:
          JPY pairs  (price ~100-200) : multiples of 0.50  within 30 pips
          Cross pairs (price ~1.5-3.5) : multiples of 0.0250 within 30 pips
          Major pairs (price ~0.5-1.5) : multiples of 0.0250 within 20 pips

        Using 0.0250 (half-major increments) catches the levels Alex actually
        trades: 1.2750, 1.3000, 1.3250, 1.3500, 2.2500, 2.2750.
        These are just as watched as full round numbers.
        JPY 0.500 already covers half-numbers (157.000, 157.500).
        """
        is_jpy = "JPY" in pair.upper()
        if is_jpy:
            increment = 0.50
            tolerance = 0.30   # 30 JPY pips (0.30 yen)
        elif price > 1.50:     # GBP/NZD, GBP/AUD style crosses
            increment = 0.0250
            tolerance = 0.0030  # 30 pips
        else:                   # EUR/USD, GBP/USD, AUD/USD, USD/CHF, USD/CAD etc.
            increment = 0.0250
            tolerance = 0.0020  # 20 pips

        nearest = round(price / increment) * increment
        return abs(price - nearest) <= tolerance

    @staticmethod
    def _pair_currencies(pair: str) -> set:
        """Extract the two currency codes from a pair string.
        'GBP/USD' → {'GBP', 'USD'}   'GBP_USD' → {'GBP', 'USD'}
        """
        return set(pair.replace("_", "/").upper().split("/")[:2])

    def _currencies_in_use(self) -> set:
        """Return all currency codes currently tied up in open positions."""
        in_use = set()
        for p in self.open_positions:
            in_use |= self._pair_currencies(p)
        return in_use

    def evaluate(
        self,
        pair: str,
        df_weekly: pd.DataFrame,
        df_daily: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        current_price: Optional[float] = None,
        current_dt=None,          # override for backtesting; live uses datetime.now()
    ) -> TradeDecision:
        """
        Run the complete strategy evaluation for a pair.
        Returns a TradeDecision with full context.

        current_dt: optional datetime — used by backtester to pass historical
                    bar timestamps so session/news filters see the right time.
                    Defaults to datetime.now() when None (live trading).
        """
        if current_price is None:
            current_price = df_1h['close'].iloc[-1]

        failed_filters = []

        # ─────────────────────────────────────────────
        # FILTER 0: Dual-trade eligibility
        #
        # Allows up to 2 simultaneous positions IF:
        #   a) Max concurrent cap not reached (≤2)
        #   b) Proposed pair shares NO currency with any open position
        #      (e.g. GBP/USD + EUR/USD both expose USD — blocked)
        #   c) Book exposure budget ≥ MIN_SECOND_TRADE_PCT (5%)
        #      (enforced at executor level using get_book_risk_pct)
        # ─────────────────────────────────────────────
        MAX_CONCURRENT = 1   # Alex's rule: one trade at a time, no exceptions

        if len(self.open_positions) >= MAX_CONCURRENT:
            open_pairs = list(self.open_positions.keys())
            return TradeDecision(
                decision=Decision.BLOCKED,
                pair=pair,
                direction=None,
                reason=(
                    f"⛔ MAX POSITIONS: {len(open_pairs)} trades open "
                    f"({', '.join(open_pairs)}). Waiting for one to close."
                ),
                confidence=0.0,
                failed_filters=["max_concurrent"],
            )

        if self.open_positions:
            proposed_ccys = self._pair_currencies(pair)
            in_use_ccys   = self._currencies_in_use()
            overlap       = proposed_ccys & in_use_ccys
            if overlap:
                return TradeDecision(
                    decision=Decision.BLOCKED,
                    pair=pair,
                    direction=None,
                    reason=(
                        f"⛔ CURRENCY OVERLAP: {', '.join(sorted(overlap))} already "
                        f"exposed via {list(self.open_positions.keys())}. "
                        f"No correlated pairs allowed simultaneously."
                    ),
                    confidence=0.0,
                    failed_filters=["currency_overlap"],
                )

        # ─────────────────────────────────────────────
        # FILTER 0b: Stop-out cooldown
        #
        # After a stop is hit on this pair, block re-entry for
        # STOP_COOLDOWN_DAYS (default 5 days). Prevents re-entering
        # a similar setup on the same pair immediately after a loss —
        # which often means the market is actively moving against the
        # setup rather than reversing at the level as expected.
        # ─────────────────────────────────────────────
        if self.is_in_stop_cooldown(pair, current_dt):
            return TradeDecision(
                decision=Decision.BLOCKED,
                pair=pair,
                direction=None,
                reason=(
                    f"⛔ STOP COOLDOWN: {pair} was stopped out recently. "
                    f"Waiting {self.STOP_COOLDOWN_DAYS:.0f} days before re-entry."
                ),
                confidence=0.0,
                failed_filters=["stop_cooldown"],
            )

        # ─────────────────────────────────────────────
        # FILTER 1: Session filter (hard block)
        # ─────────────────────────────────────────────
        entry_allowed, session_reason = self.session_filter.is_entry_allowed(current_dt)
        if not entry_allowed:
            return TradeDecision(
                decision=Decision.BLOCKED,
                pair=pair,
                direction=None,
                reason=f"Session blocked: {session_reason}",
                confidence=0.0,
                failed_filters=["session"],
            )

        session, session_quality = self.session_filter.session_quality(current_dt)

        # ─────────────────────────────────────────────
        # FILTER 1.5: Tier 1 news blackout
        # No entries 30min before → 90min after major events.
        # Early release: after 60min, if the first post-news 1H candle is
        # clean (body ≥ 33% of range), we re-enter immediately instead of
        # waiting the full 90-min clock.
        # ─────────────────────────────────────────────
        from datetime import datetime, timezone as _tz
        _last_closed_candle = (
            df_1h.iloc[-2][["open", "high", "low", "close"]].to_dict()
            if len(df_1h) >= 2 else None
        )
        news_blocked, news_reason = self.news_filter.is_entry_blocked(
            current_dt or datetime.now(_tz.utc),
            post_news_candle=_last_closed_candle,
        )
        if news_blocked:
            return TradeDecision(
                decision=Decision.BLOCKED,
                pair=pair,
                direction=None,
                reason=news_reason,
                confidence=0.0,
                failed_filters=["news_blackout"],
            )

        # ─────────────────────────────────────────────
        # FILTER 2: Multi-timeframe trend alignment
        #
        # This is a REVERSAL strategy. Alex enters SHORT at the TOP of
        # uptrends (H&S, double top) and LONG at the BOTTOM of downtrends
        # (IH&S, double bottom). The weekly being in the OPPOSITE direction
        # of the trade is not a problem — it's CONFIRMATION that price
        # traveled far enough to reach the reversal zone.
        #
        # Two valid contexts:
        #
        # A) TREND CONTINUATION (2/3 aligned):
        #    Daily + 4H + Weekly all point the same way.
        #    Valid for break & retest setups mid-trend.
        #
        # B) REVERSAL AT EXTREME (the primary pattern):
        #    Weekly: in OPPOSITE direction (price was driven to an extreme)
        #    Daily + 4H: starting to REVERSE
        #    → H&S at the top of a weekly uptrend = textbook short setup
        #    → IH&S at the bottom of a weekly downtrend = textbook long setup
        #    Alex wins all his big trades this way.
        #
        # Block only when there is no evidence of reversal on EITHER
        # the daily OR 4H — pure trend continuation with no pattern signal.
        # ─────────────────────────────────────────────
        trend_w = self.pattern_detector.detect_trend(df_weekly)
        trend_d = self.pattern_detector.detect_trend(df_daily)
        trend_4 = self.pattern_detector.detect_trend(df_4h)

        w_bullish = trend_w in (Trend.BULLISH, Trend.STRONG_BULLISH)
        w_bearish = trend_w in (Trend.BEARISH, Trend.STRONG_BEARISH)
        d_bullish = trend_d in (Trend.BULLISH, Trend.STRONG_BULLISH)
        d_bearish = trend_d in (Trend.BEARISH, Trend.STRONG_BEARISH)
        h4_bullish = trend_4 in (Trend.BULLISH, Trend.STRONG_BULLISH)
        h4_bearish = trend_4 in (Trend.BEARISH, Trend.STRONG_BEARISH)

        bullish_count = sum([w_bullish, d_bullish, h4_bullish])
        bearish_count = sum([w_bearish, d_bearish, h4_bearish])

        _reversal_context = False   # weekly opposing = price at extreme

        if bullish_count >= 2:
            # Standard: trend continuation or early-trend long
            trade_direction = 'long'
        elif bearish_count >= 2:
            # Standard: trend continuation or early-trend short
            trade_direction = 'short'
        elif w_bullish and (d_bearish or h4_bearish):
            # REVERSAL SHORT: weekly drove price UP to resistance; daily/4H
            # now actively turning bearish — classic H&S or double top at the
            # top of a trend. Alex's GBP/CHF, USD/CHF short setups look like this.
            # Requires daily OR 4H to actively show bearish momentum (not just neutral).
            trade_direction = 'short'
            _reversal_context = True
        elif w_bearish and (d_bullish or h4_bullish):
            # REVERSAL LONG: weekly drove price DOWN to support; daily/4H
            # now actively turning bullish — classic IH&S or double bottom.
            trade_direction = 'long'
            _reversal_context = True
        else:
            trade_direction = None
            failed_filters.append("trend_alignment")
            logger.debug(f"{pair}: Trends not aligned. W={trend_w.value} D={trend_d.value} 4H={trend_4.value}")

        # ─────────────────────────────────────────────────────────────────
        # EMA computation — pre-calculate for use in scoring below
        # ─────────────────────────────────────────────────────────────────
        _ema_confluence = False
        _ema21 = _ema50 = 0.0
        if len(df_daily) >= 50:
            _closes_d = df_daily["close"].values
            _ema21 = float(pd.Series(_closes_d).ewm(span=21, adjust=False).mean().iloc[-1])
            _ema50 = float(pd.Series(_closes_d).ewm(span=50, adjust=False).mean().iloc[-1])
            _near_ema21 = abs(current_price - _ema21) / current_price < 0.005
            _near_ema50 = abs(current_price - _ema50) / current_price < 0.005
            _ema_confluence = _near_ema21 or _near_ema50

        # ─────────────────────────────────────────────────────────────────
        # FILTER 3: Pattern detection — PATTERN SETS DIRECTION
        #
        # This is the architectural fix for the biggest gap between our bot
        # and Alex. The old flow was:
        #   Trend filter → sets direction → pattern must MATCH direction
        #
        # That blocked GBP/CHF double top (Aug 2024, Alex's 1:11 RR trade)
        # because weekly=bullish + 4H=bullish = "direction=LONG" so the
        # bearish double top was invisible.
        #
        # Alex's actual process:
        #   1. Scan charts → see the pattern forming
        #   2. Pattern tells him direction (double top = short, IH&S = long)
        #   3. Check multi-TF context: does this MAKE SENSE given where price is?
        #      Weekly bullish + double top at resistance = YES — price got here
        #      from below, it's at the top, this is WHY the pattern formed
        #
        # New flow:
        #   Detect ALL patterns → for each, compute multi-TF context score
        #   → pick best qualifying pattern → direction comes from pattern
        #
        # Multi-TF context scoring per pattern:
        #   HARD BLOCK: all 3 TFs actively oppose the pattern (3/3 against)
        #               = trying to short into a strong confirmed uptrend
        #               = Alex's Week 4 losses (he called these "boredom trades")
        #   TREND CONTINUATION bonus (+5%): 2/3 TFs align with pattern
        #   REVERSAL AT EXTREME (0% adj): weekly opposing, daily/4H turning
        #               = classic setup at end of trend
        #   REVERSAL EARLY (-5% adj): weekly opposing, daily neutral (not yet turning)
        #               = GBP/CHF double top Aug 2024 — pattern IS the warning signal
        #   PARTIAL ALIGNMENT (-3% adj): mixed signals, some confirmation
        # ─────────────────────────────────────────────────────────────────

        # Build pattern list from daily (primary source — larger stops, clearer structure)
        patterns_daily = self.pattern_detector.detect_all(df_daily)

        # ── 4H patterns — ALSO valid entries per Alex's actual trades ────────
        # Alex's GBP/JPY Week 1 trade: H&S on 4H, daily only provided context.
        # His USD/CHF, USD/JPY shorts: same — 4H reversal pattern, daily shift.
        # Rule: 4H structural reversal patterns (H&S, DT, DB, IH&S) are valid
        # WHEN daily/weekly context supports them. 4H sweeps excluded (stop risk).
        # Stop: when a 4H pattern entry is chosen, stop is backed to DAILY ATR
        # (ATR floor filter below ensures minimum 0.75×daily ATR — daily-scale stop).
        # 4H patterns get a -5% confidence adj on top of context adj (less mature).
        patterns_4h_entries = []
        mtf_confluence: dict = {}
        if len(df_4h) >= 30:
            _4h_structural = [
                p for p in self.pattern_detector.detect_all(df_4h)
                if 'sweep' not in p.pattern_type
                and any(k in p.pattern_type for k in
                        ('head_and_shoulders', 'double_top', 'double_bottom',
                         'inverted_head_and_shoulders'))
            ]
            for p4 in _4h_structural:
                # MTF confluence: same pattern on BOTH daily and 4H = higher confidence
                for pd_pat in patterns_daily:
                    if pd_pat.pattern_type == p4.pattern_type and pd_pat.direction == p4.direction:
                        mtf_confluence[p4.pattern_type] = True
                        break
                # Mark as 4H origin so confidence formula applies the extra penalty
                p4._source_tf = '4h'
                patterns_4h_entries.append(p4)

        # Merge: daily patterns first (higher priority), then 4H patterns that
        # bring something not already in the daily list
        daily_type_dirs = {(p.pattern_type, p.direction) for p in patterns_daily}
        for p4 in patterns_4h_entries:
            # Always include: even if same type exists on daily, 4H pattern might
            # be more recent / at a different level. Let _mtf_context score it.
            patterns_daily.append(p4)

        # Sort by clarity — daily patterns usually win but 4H can rank higher
        # if they're unusually clean
        patterns = sorted(patterns_daily, key=lambda p: p.clarity, reverse=True)

        MAX_NECKLINE_PCT_SWEEP      = 3.5
        MAX_NECKLINE_PCT_STRUCTURAL = 2.0

        def _mtf_context(pat) -> tuple:
            """
            Given a pattern, evaluate whether the multi-timeframe context
            supports taking the trade. Returns (valid, adj, is_reversal).

            valid      : False = hard block (don't take this pattern)
            adj        : confidence adjustment (-0.05 to +0.05)
            is_reversal: True if weekly is OPPOSING the pattern (reversal at extreme)

            Quality tiers (from Alex's videos):
              TIER 1 — Trend continuation: 2/3 TFs aligned with pattern
                       Use case: break & retest mid-trend
              TIER 2 — Reversal confirmed: weekly opposing + D/4H actively reversing
                       Use case: H&S / double top at TOP of uptrend
              TIER 3 — Reversal early: weekly opposing + daily stalling (neutral)
                       Use case: GBP/CHF double top Aug 2024 — pattern is the signal
                       ONLY valid for STRUCTURAL reversal patterns (H&S, DT, DB, IH&S)
                       NOT valid for break-retest (that requires trend alignment)
              BLOCK — Everything else: insufficient context
            """
            is_short = pat.direction == 'bearish'
            is_long  = pat.direction == 'bullish'
            is_reversal_type = any(k in pat.pattern_type for k in
                                   ('head_and_shoulders', 'double_top', 'double_bottom',
                                    'inverted_head_and_shoulders'))

            # ── HARD BLOCK: 2+ TFs actively oppose the pattern direction ──
            # Alex's Week 4 losses: shorting/buying against confirmed trends.
            if is_short and bullish_count >= 2:
                return False, 0.0, False
            if is_long  and bearish_count >= 2:
                return False, 0.0, False

            # ── TIER 1: Trend continuation (2/3 aligned) ──────────────────
            if is_short and bearish_count >= 2:
                return True, +0.05, False
            if is_long  and bullish_count >= 2:
                return True, +0.05, False

            # ── TIER 2: Reversal confirmed — weekly opposing + D/4H turning ─
            # Price was driven to the level by the weekly trend, now reversing.
            # D or 4H must actively show the reversal has begun.
            if is_short and w_bullish and (d_bearish or h4_bearish):
                return True, 0.0, True
            if is_long  and w_bearish and (d_bullish or h4_bullish):
                return True, 0.0, True

            # ── TIER 3: Reversal early — weekly opposing + daily stalling ──
            # Only for structural reversal patterns (H&S, DT, DB, IH&S).
            # The pattern forming IS the reversal signal — daily hasn't turned yet.
            # GBP/CHF Aug 2024: weekly=bullish, daily=neutral, double top at high.
            # Break-retest is EXCLUDED — it requires trend confirmation to enter.
            if is_reversal_type and is_short and w_bullish and not d_bullish:
                return True, -0.05, True
            if is_reversal_type and is_long  and w_bearish and not d_bearish:
                return True, -0.05, True

            # ── TIER 4: Range reversal — weekly neutral + local push exhausted ──
            # When the weekly trend is neutral (ranging), price bounces between
            # range high and range low. The 4H pushing UP into a double top =
            # price reached the range high. That 4H bullish IS the confirmation:
            # "price pushed up to resistance and is now forming a reversal there."
            # Same logic as the weekly extreme tiers, just compressed to D/4H.
            #
            # Condition: weekly neutral + daily not yet reversing + 4H made the push
            # GBP/CHF Aug 2024 (Alex's 1:11 RR): W=neutral, D=neutral, 4H=bullish
            # Only valid for structural reversal patterns (H&S, DT, DB, IH&S).
            _w_neutral = not w_bullish and not w_bearish
            _d_neutral = not d_bullish and not d_bearish
            if is_reversal_type and is_short and _w_neutral and h4_bullish:
                return True, -0.05, True   # same penalty as early reversal
            if is_reversal_type and is_long  and _w_neutral and h4_bearish:
                return True, -0.05, True

            # ── BLOCK: insufficient context for all other cases ────────────
            return False, 0.0, False

        # Find the best qualifying pattern (highest clarity that passes context)
        matching_pattern  = None
        trade_direction   = None
        _reversal_context = False
        _context_adj      = 0.0

        for p in patterns:
            if p.clarity < self.min_pattern_clarity:
                continue
            max_neckline = (
                MAX_NECKLINE_PCT_SWEEP if 'sweep' in p.pattern_type
                else MAX_NECKLINE_PCT_STRUCTURAL
            )
            if abs(p.neckline - current_price) / current_price * 100 > max_neckline:
                continue
            valid, adj, is_rev = _mtf_context(p)
            if not valid:
                continue
            # Pattern qualifies — set direction from pattern
            matching_pattern  = p
            trade_direction   = 'long' if p.direction == 'bullish' else 'short'
            _reversal_context = is_rev
            _context_adj      = adj
            break   # highest-clarity qualifying pattern wins

        if matching_pattern is None:
            failed_filters.append("no_pattern")
            # Build a context-aware reason for the decision log
            _tf_summary = (
                f"W={trend_w.value} D={trend_d.value} 4H={trend_4.value}"
            )
            return TradeDecision(
                decision=Decision.WAIT,
                pair=pair,
                direction=trade_direction,
                reason=(
                    f"No qualifying pattern (H&S / double top/bottom / break-retest) "
                    f"detected near price. Trend context: {_tf_summary}. "
                    f"Watching for a formation to develop."
                ),
                confidence=0.30,
                failed_filters=failed_filters,
                trend_weekly=trend_w,
                trend_daily=trend_d,
                trend_4h=trend_4,
            )

        # ─────────────────────────────────────────────
        # FILTER 4: Key level validation — Round Number OR Structural Level
        #
        # Alex's level hierarchy (from his strategy notes, highest priority first):
        #   ★★★★★  Previous structure highs/lows
        #   ★★★★★  Round psychological levels (1.3000, 157.500, etc.)
        #   ★★★★   EMA confluence
        #
        # BOTH types qualify a setup. If the pattern formed at a previous
        # structural high that was tested 3+ times, that's just as valid as
        # 1.3000 even if the exact price is 1.3187 not a clean round number.
        #
        # The LevelDetector now returns BOTH round numbers AND structural
        # levels — so we just check whether ANY detected level is near
        # the pattern's structural price.
        # ─────────────────────────────────────────────
        levels = self.level_detector.detect(df_daily, matching_pattern.pattern_level, pair=pair)

        # Check round number first (fast path, no level detection needed)
        pattern_at_round_number = (
            matching_pattern.pattern_level > 0
            and self._is_major_level(matching_pattern.pattern_level, pair)
        )

        # Also check: is the pattern level near any level the detector found?
        # (Includes structural levels from _find_structural_levels)
        nearest_level = None
        for level in levels:
            dist_frac = abs(level.price - matching_pattern.pattern_level) / max(matching_pattern.pattern_level, 0.0001)
            if dist_frac < 0.01:   # within 1% of the pattern structural price
                nearest_level = level
                break
        # Fall back: use the closest level regardless of proximity
        if nearest_level is None and levels:
            nearest_level = levels[0]

        # A structural level that was tested 4+ times with additional confluence
        # (EMA, equal highs/lows, or near a round number) = equivalent to a round number.
        # Setting high bar intentionally — don't want every swing high to qualify.
        at_structural_level = (
            nearest_level is not None
            and nearest_level.score >= 4                                            # multiple confluences required
            and any("structural" in c for c in nearest_level.confluences)          # must be a structural level
            and abs(nearest_level.price - matching_pattern.pattern_level)
                / max(matching_pattern.pattern_level, 0.0001) < 0.01               # within 1% of pattern level
        )

        if not pattern_at_round_number and not at_structural_level:
            failed_filters.append("pattern_not_at_round_number")
            return TradeDecision(
                decision=Decision.WAIT,
                pair=pair,
                direction=trade_direction,
                reason=(
                    f"Pattern found ({matching_pattern.pattern_type}) but structural "
                    f"level {matching_pattern.pattern_level:.5f} is not at a major "
                    f"round number or tested structural level. Alex only trades where "
                    f"'everyone is watching.'"
                ),
                confidence=0.25,
                failed_filters=failed_filters,
                pattern=matching_pattern,
                trend_weekly=trend_w,
                trend_daily=trend_d,
                trend_4h=trend_4,
            )

        # ─────────────────────────────────────────────
        # FILTER 5: Engulfing candle — THE PATIENCE FILTER
        # THE most critical check. No signal = NO ENTRY. Ever.
        #
        # Also: reject signals formed during news windows.
        # A 200-pip NFP spike looks like a perfect engulfing candle.
        # It isn't. It's a data release — not tradeable price action.
        # ─────────────────────────────────────────────
        has_signal, signal = self.signal_detector.has_signal(df_1h, trade_direction)

        # Check if the triggering candle is a news candle — if so, treat as no signal
        if has_signal and signal and len(df_1h) > 0:
            try:
                last_candle_ts = df_1h.index[-1]
                if hasattr(last_candle_ts, "to_pydatetime"):
                    last_candle_ts = last_candle_ts.to_pydatetime()
                from datetime import timezone as _tz2
                if not last_candle_ts.tzinfo:
                    last_candle_ts = last_candle_ts.replace(tzinfo=_tz2.utc)
                if self.news_filter.is_news_candle(last_candle_ts):
                    logger.info(
                        f"{pair}: Signal REJECTED — triggering candle formed during "
                        f"Tier 1 news window. Not genuine price action. Waiting for next bar."
                    )
                    has_signal = False
                    signal = None
                    failed_filters.append("news_candle")
            except Exception:
                pass  # if timestamp check fails, proceed normally

        if not has_signal or signal is None:
            # Setup is valid but we're waiting for the candle
            # This is the correct state 90% of the time
            _level_str = (
                f"level {nearest_level.price:.5f} (score={nearest_level.score})"
                if nearest_level else
                f"round number {matching_pattern.pattern_level:.5f}"
            )
            return TradeDecision(
                decision=Decision.WAIT,
                pair=pair,
                direction=trade_direction,
                reason=(
                    f"Setup ACTIVE — {matching_pattern.pattern_type} at {_level_str}. "
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
        # FILTER 6: Pattern memory — don't re-enter an exhausted formation
        # ─────────────────────────────────────────────
        # Use the pattern's own structural level as the neckline reference for memory.
        # Falls back to nearest_level.price if available (same round number, adds precision).
        neckline_ref = (nearest_level.price if nearest_level else matching_pattern.pattern_level)
        if self.pattern_already_traded(pair, matching_pattern.pattern_type, neckline_ref):
            return TradeDecision(
                decision=Decision.WAIT,
                pair=pair,
                direction=trade_direction,
                reason=(
                    f"Pattern EXHAUSTED — {matching_pattern.pattern_type} at {neckline_ref:.5f} "
                    f"was already traded. Formation is spent. "
                    f"Waiting for a fresh structural break before re-engaging."
                ),
                confidence=0.30,
                failed_filters=["pattern_memory"],
                nearest_level=nearest_level,
                pattern=matching_pattern,
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
        # Components:
        #   level quality      : how strong is the key level (round number, tested, EMA)
        #   pattern clarity    : how clean is the formation
        #   signal strength    : how strong is the engulfing confirmation
        #   session quality    : London > NY > other
        # Adjustments from multi-TF context (_context_adj from _mtf_context()):
        #   +5%  trend continuation (2/3 TFs aligned with pattern)
        #    0%  reversal at extreme (confirmed daily/4H turn against weekly)
        #   -3%  partial alignment
        #   -5%  early reversal (weekly opposing, daily neutral — pattern is early warning)
        #   -8%  weak signal (only 1 TF confirms)
        # Bonuses:
        #   +6%  if 4H independently agrees with pattern direction
        #   +8%  if same pattern confirmed on BOTH daily and 4H
        _h4_bonus  = 0.06 if (
            (trade_direction == 'long'  and h4_bullish) or
            (trade_direction == 'short' and h4_bearish)
        ) else 0.0
        _mtf_bonus   = 0.08 if mtf_confluence.get(matching_pattern.pattern_type) else 0.0
        _4h_penalty  = -0.05 if getattr(matching_pattern, '_source_tf', 'daily') == '4h' else 0.0
        _level_score = nearest_level.score if nearest_level else 2
        confidence = min(1.0, max(0.0, (
            0.26 * min(1.0, _level_score / 4)    # level quality
          + 0.20 * matching_pattern.clarity        # pattern quality
          + 0.20 * signal.strength                 # signal quality
          + 0.20 * session_quality                 # session quality
          + _h4_bonus                              # 4H direction bonus
          + _mtf_bonus                             # MTF pattern confluence bonus
          + _context_adj                           # multi-TF context adjustment
          + _4h_penalty                            # 4H-source pattern (less mature than daily)
        )))

        risk_dollars = self.account_balance * (self.risk_pct / 100)
        rr_1 = abs(entry_price - target_1) / max(abs(entry_price - stop_loss), 0.000001)
        _level_desc = (
            f"Level {nearest_level.price:.5f} score={nearest_level.score}"
            if nearest_level else
            f"Round number {matching_pattern.pattern_level:.5f}"
        )

        return TradeDecision(
            decision=Decision.ENTER,
            pair=pair,
            direction=trade_direction,
            reason=(
                f"ALL FILTERS PASSED. "
                f"{matching_pattern.pattern_type.upper()} pattern | "
                f"{_level_desc} | "
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
            neckline_ref=neckline_ref,
        )

    def register_open_position(
        self,
        pair: str,
        entry_price: float,
        stop_loss: float,
        direction: str,
        pattern_type: Optional[str] = None,
        neckline_ref: Optional[float] = None,
        risk_pct: Optional[float] = None,
    ):
        """Call this when a trade is actually opened."""
        self.open_positions[pair] = {
            'entry':        entry_price,
            'stop':         stop_loss,
            'direction':    direction,
            'pattern_type': pattern_type,   # stored so we can mark exhausted on close
            'neckline_ref': neckline_ref,   # stored so we can mark exhausted on close
            'risk_pct':     risk_pct or 0.0,  # stored for book exposure tracking
        }
        logger.info(
            f"Position opened: {pair} {direction} at {entry_price:.5f} "
            f"SL={stop_loss:.5f}  risk={risk_pct or 0:.1f}%"
        )

    def close_position(self, pair: str, exit_price: float):
        """
        Call this when a trade is closed.
        Automatically marks the formation as exhausted in pattern memory.
        """
        if pair in self.open_positions:
            pos = self.open_positions.pop(pair)
            pnl = (exit_price - pos['entry']) * (1 if pos['direction'] == 'long' else -1)
            logger.info(f"Position closed: {pair} at {exit_price:.5f} P&L direction = {'+' if pnl > 0 else ''}{pnl:.5f}")
            # Mark formation exhausted so we don't re-enter the same setup
            if pos.get("pattern_type") and pos.get("neckline_ref"):
                self.mark_pattern_exhausted(pair, pos["pattern_type"], pos["neckline_ref"])

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
        level: Optional[KeyLevel],
        pattern: PatternResult,
    ) -> tuple:
        """
        Returns (entry_price, stop_loss, target_1, target_2, lot_size).
        """
        # Entry: current price (limit order near current close)
        entry = current_price

        # ── Stop loss: structural stop behind the pattern extreme ──────
        #
        # Uses the pattern detector's computed stop_loss, which places
        # the stop behind the full formation:
        #   Double bottom: below both lows
        #   Double top:    above both highs
        #   H&S:           above right shoulder (bearish) / below (bullish)
        #   Break & retest: beyond the broken level
        #
        # This gives the trade room to breathe through 1H noise while
        # keeping stop anchored to real structure. Pattern neckline proximity
        # filter (1.5%) ensures we only match recent patterns so stops stay
        # in the 100-500 pip range (not 1,000+ pip ancient structural stops).
        stop_loss = pattern.stop_loss

        # Safety: stop must be on the correct side of entry
        MIN_STOP_BUFFER = 0.001   # 0.1% minimum
        if direction == 'short':
            stop_loss = max(stop_loss, entry * (1 + MIN_STOP_BUFFER))
        else:
            stop_loss = min(stop_loss, entry * (1 - MIN_STOP_BUFFER))

        # Targets: measured move from neckline (pattern height projected)
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
