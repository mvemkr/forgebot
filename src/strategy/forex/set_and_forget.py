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
from typing import Optional, Dict, List, Tuple
from enum import Enum
import logging

from .session_filter import SessionFilter
from .level_detector import LevelDetector, KeyLevel
from .pattern_detector import PatternDetector, PatternResult, Trend
from .entry_signal import EntrySignalDetector, EntrySignal
from . import strategy_config as _cfg   # module-ref import so apply_levers() patches propagate

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

    # Macro theme stacking flag — set True when this is a theme-driven stacked trade.
    # Passed to trade_executor so check_entry_eligibility can apply STACK_MAX + waive
    # currency overlap (matching backtester _entry_eligible() logic exactly).
    is_macro_theme_pair: bool              = False

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
        self.STOP_COOLDOWN_DAYS: float = _cfg.STOP_COOLDOWN_DAYS
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
    def _price_extension(df_daily: pd.DataFrame) -> Tuple[float, bool, bool]:
        """
        Compute how far current price is from its 2-year range using percentile rank.

        Returns (z_score, is_extended_high, is_extended_low) where:
          is_extended_high = price in TOP 10% of 2-year range → only SHORTs valid
          is_extended_low  = price in BTM 10% of 2-year range → only LONGs valid

        WHY PERCENTILE RANK, NOT Z-SCORE:
          Z-score assumes a normal distribution around a mean. For a currency pair
          in a sustained multi-year trend (e.g. USD/JPY 110→160 over 3 years),
          the 1-year mean moves with the trend, so Z stays low even at extreme prices.
          USD/JPY at 158 in July 2024 (a 40-year high) had a 1-year Z ≈ +1.0
          because the whole year trended up. Z-score completely missed the extreme.

          Percentile rank over 2 years answers the right question:
          "Is this price near the top/bottom of where it's traded recently?"
          USD/JPY at 158 → 97th percentile of 2-year range → EXTENDED HIGH. Correct.

          Alex's exact reasoning: "It's at a 30-year high, I'm only taking shorts."
          Percentile rank captures this. Z-score doesn't.

        Threshold: 90th / 10th percentile over 504 trading days (~2 years).
        Also returns Z-score for display/bonus calculations (less critical now).
        """
        if len(df_daily) < 60:
            return 0.0, False, False

        lookback = min(504, len(df_daily))   # 2 years of daily data
        closes   = df_daily["close"].values[-lookback:]
        current  = float(df_daily["close"].iloc[-1])

        # Percentile rank: fraction of bars where close was BELOW current price
        pct_rank = float(np.mean(closes <= current))

        # Z-score retained for confidence bonus display (not for block logic)
        mean = float(np.mean(closes))
        std  = float(np.std(closes))
        z    = (current - mean) / std if std > 0 else 0.0

        # Extended = top or bottom N% of 2-year price range.
        # Threshold controlled by OVEREXTENSION_THRESHOLD lever (default 0.90 = top/bottom 10%).
        _ext_thr = _cfg.OVEREXTENSION_THRESHOLD
        is_extended_high = pct_rank >= _ext_thr          # top (1-thr)% → reckless to go long
        is_extended_low  = pct_rank <= (1.0 - _ext_thr)  # btm (1-thr)% → reckless to go short

        return z, is_extended_high, is_extended_low

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
        if abs(price - nearest) <= tolerance:
            return True

        # Secondary: 0.0100-increment psychological levels.
        # Alex trades levels like USD/CHF 0.880/0.884/0.890, GBP/CHF 1.120/1.130
        # that fall between the 0.025 grid. These are just as "watched" on real
        # charts — the 0.025 check was too coarse and blocked valid setups.
        # JPY pairs already covered by 0.50 increment above.
        # Controlled by LEVEL_ALLOW_FINE_INCREMENT lever (default True).
        if not is_jpy and _cfg.LEVEL_ALLOW_FINE_INCREMENT:
            fine_increment = 0.0100
            fine_tolerance = 0.0030   # 30 pips — catches 0.882 near 0.880, 1.132 near 1.130
            nearest_fine = round(price / fine_increment) * fine_increment
            if abs(price - nearest_fine) <= fine_tolerance:
                return True

        return False

    @staticmethod
    def _nearest_round_number(price: float, pair: str) -> float:
        """
        Return the value of the nearest psychological round number for this pair.
        Uses the same increment logic as _is_major_level.
        """
        is_jpy = "JPY" in pair.upper()
        if is_jpy:
            if price > 50:
                inc = 0.50
            else:
                inc = 0.025
        elif price > 1.50:
            inc = 0.025
        else:
            inc = 0.025
        return round(price / inc) * inc

    @staticmethod
    def _peaks_at_level(pattern, pair: str) -> bool:
        """
        Rule: for double_top SHORT, pattern_level (avg of peaks) must be AT or
        ABOVE the nearest round number. If below, price is still approaching
        the round number from below = continuation, not a rejection/reversal.

        For double_bottom LONG, pattern_level (avg of troughs) must be AT or
        BELOW the nearest round number.

        This blocks "false double tops" where price is grinding up toward a
        round number — two bars just below the level look like a double top
        but are actually a bullish flag about to break through.

        Tolerance: 15 pips non-JPY / 25 pips JPY.
        Bypassed when REQUIRE_PEAKS_AT_LEVEL lever is False.
        """
        if not _cfg.REQUIRE_PEAKS_AT_LEVEL:
            return True
        if pattern.pattern_type not in ('double_top', 'double_bottom'):
            return True   # only applies to double patterns

        is_jpy = "JPY" in pair.upper()
        tol = 0.0025 if is_jpy else 0.0015   # 25 pip / 15 pip tolerance

        nearest = SetAndForgetStrategy._nearest_round_number(pattern.pattern_level, pair)

        if pattern.pattern_type == 'double_top':
            # Peaks must have REACHED the round number from below
            # i.e. pattern_level >= nearest_round - tolerance
            return pattern.pattern_level >= nearest - tol

        else:  # double_bottom
            # Troughs must have REACHED the round number from above
            # i.e. pattern_level <= nearest_round + tolerance
            return pattern.pattern_level <= nearest + tol

    @staticmethod
    def _neckline_at_level(pattern, pair: str) -> bool:
        """
        Rule: for H&S, IH&S, and break_retest patterns, the neckline/break
        level must sit within NECKLINE_LEVEL_TOLERANCE_PCT of a round number.

        Double top/bottom peaks are already gated by _peaks_at_level().
        This extends the same discipline to neckline-based entries.

        Why: NZD/CAD Sep 2024 — H&S neckline at ~0.835, not near 0.840 or
        any meaningful level. Breaking an arbitrary structure point is a
        momentum trade, not a set-and-forget reversal at a psychological price.
        Alex ALWAYS trades levels humans are watching (0.84, 1.10, 160, etc.).

        Bypassed when REQUIRE_NECKLINE_AT_LEVEL lever is False.
        """
        if not _cfg.REQUIRE_NECKLINE_AT_LEVEL:
            return True
        # Only applies to H&S and IH&S — their neckline is not the pattern_level
        # so it can float far from any round number.
        # Double top/bottom: _peaks_at_level handles level validation.
        # Break_retest: Filter 4 (_is_major_level / structural level) already
        #               validates the break level — no additional check needed.
        if pattern.pattern_type not in ('head_and_shoulders',
                                        'inverted_head_and_shoulders'):
            return True

        nearest = SetAndForgetStrategy._nearest_round_number(pattern.pattern_level, pair)
        tol = pattern.pattern_level * _cfg.NECKLINE_LEVEL_TOLERANCE_PCT
        return abs(pattern.pattern_level - nearest) <= tol

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
        macro_theme=None,         # CurrencyTheme object if a macro theme is active
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
        # Alex's standard rule: 1 trade at a time.
        # EXCEPTION: when a macro currency theme is detected (e.g. JPY weakness
        # across all pairs in Sep 2024), allow stacking up to STACK_MAX positions
        # on that theme. Each position gets reduced size (normal / n_positions).
        # This is how Alex made $70K in one week — 4 JPY shorts as one theme.
        from .currency_strength import STACK_MAX
        _is_macro_theme_pair = (
            macro_theme is not None
            and pair in [p for p, _ in macro_theme.suggested_trades]
        )
        MAX_CONCURRENT = STACK_MAX if _is_macro_theme_pair else 1

        if len(self.open_positions) >= MAX_CONCURRENT:
            open_pairs = list(self.open_positions.keys())
            return TradeDecision(
                decision=Decision.BLOCKED,
                pair=pair,
                direction=None,
                reason=(
                    f"⛔ MAX POSITIONS: {len(open_pairs)} trades open "
                    f"({', '.join(open_pairs)}). Waiting for one to close."
                    + (f" [Macro theme: {macro_theme}]" if macro_theme else "")
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
        news_blocked, news_reason = (
            self.news_filter.is_entry_blocked(
                current_dt or datetime.now(_tz.utc),
                post_news_candle=_last_closed_candle,
            )
            if _cfg.NEWS_FILTER_ENABLED else (False, "")
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
        # EMA + price extension — pre-calculate for use in scoring below
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

        # Price extension Z-score — how far is current price from 1-year mean?
        # Used in _mtf_context() to hard-block wrong-direction trades at extremes
        # and add a confidence bonus to correct-direction trades (e.g., short
        # at Z>2 = price at extreme high = ideal reversal short setup).
        _ext_z, _ext_high, _ext_low = self._price_extension(df_daily)

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
                         'inverted_head_and_shoulders', 'consolidation_breakout'))
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
                                    'inverted_head_and_shoulders', 'consolidation_breakout'))

            # ── OVEREXTENSION BLOCK / BONUS (Feature #1) ──────────────────
            # When price is in the top/bottom OVEREXTENSION_THRESHOLD % of its
            # 2-year range, only reversals AGAINST the extension are valid.
            #
            # USD/JPY at 157 (30-year high): LONG is reckless. SHORT is the trade.
            # Alex went SHORT and made 13× his money.
            #
            # Controlled by OVEREXTENSION_CHECK lever (default True).
            # Disabled → overextension analysis is advisory only (no blocks).
            if _cfg.OVEREXTENSION_CHECK:
                if _ext_high and is_long:
                    return False, 0.0, False   # price at extreme HIGH — no new longs
                if _ext_low  and is_short:
                    return False, 0.0, False   # price at extreme LOW  — no new shorts
            _ext_bonus = 0.08 if (
                _cfg.OVEREXTENSION_CHECK and (
                    (_ext_high and is_short) or (_ext_low and is_long)
                )
            ) else 0.0

            # ── HARD BLOCK: 2+ TFs actively oppose the pattern direction ──
            # Alex's Week 4 losses: shorting/buying against confirmed trends.
            if is_short and bullish_count >= 2 and not _ext_high:
                return False, 0.0, False
            if is_long  and bearish_count >= 2 and not _ext_low:
                return False, 0.0, False

            # ── TIER 1: Trend continuation (2/3 aligned, weekly not opposing) ──
            # Weekly is the master timeframe. D+4H bullish in a weekly downtrend
            # is a short-term bounce — NOT a continuation trade. Alex's rule:
            # "trade WITH the weekly or at a confirmed extreme." If weekly opposes,
            # the trade must fall to TIER 2/3 which properly gates reversals.
            if is_short and bearish_count >= 2 and not w_bullish:
                return True, +0.05 + _ext_bonus, False
            if is_long  and bullish_count >= 2 and not w_bearish:
                return True, +0.05 + _ext_bonus, False

            # ── TIER 2: Reversal confirmed — weekly opposing + D/4H turning ─
            # Price was driven to the level by the weekly trend, now reversing.
            # D or 4H must actively show the reversal has begun.
            if is_short and w_bullish and (d_bearish or h4_bearish):
                return True, 0.0 + _ext_bonus, True
            if is_long  and w_bearish and (d_bullish or h4_bullish):
                return True, 0.0 + _ext_bonus, True

            # ── TIER 3: Reversal early — weekly opposing + daily stalling ──
            # Only for structural reversal patterns (H&S, DT, DB, IH&S).
            # The pattern forming IS the reversal signal — daily hasn't turned yet.
            # GBP/CHF Aug 2024: weekly=bullish, daily=neutral, double top at high.
            # Break-retest is EXCLUDED — it requires trend confirmation to enter.
            # Controlled by ALLOW_TIER3_REVERSALS lever (default True).
            if _cfg.ALLOW_TIER3_REVERSALS:
                # Weekly stall gate (TIER3_REQUIRE_WEEKLY_STALL lever).
                # Require the most recent weekly candle to show compression or a
                # reversal close before taking a counter-trend Tier 3 position.
                # Prevents "fresh trend" reversals: 8 consecutive red weeks with
                # no pause, or 4-month uptrend with no doji — neither is ready.
                # Alex's rule: "I look for the weekly doji before I sell the top."
                def _weekly_stalling() -> bool:
                    if not _cfg.TIER3_REQUIRE_WEEKLY_STALL:
                        return True   # gate disabled — always allow
                    if len(df_weekly) < 5:
                        return True   # not enough history — assume ok
                    ranges = (df_weekly['high'] - df_weekly['low']).values
                    recent_rng = float(ranges[-1])
                    avg_4wk    = float(np.mean(ranges[-5:-1]))  # prior 4 weeks
                    # Compression: recent week < STALL_RATIO of prior avg
                    compressed = avg_4wk > 0 and recent_rng < avg_4wk * _cfg.TIER3_WEEKLY_STALL_RATIO
                    # OR: recent candle closed against the trend (crack in momentum)
                    recent_close = float(df_weekly['close'].iloc[-1])
                    recent_open  = float(df_weekly['open'].iloc[-1])
                    crack = (w_bullish and recent_close < recent_open) or \
                            (w_bearish and recent_close > recent_open)
                    return compressed or crack

                if is_reversal_type and is_short and w_bullish and not d_bullish:
                    if _weekly_stalling():
                        return True, -0.05 + _ext_bonus, True
                if is_reversal_type and is_long  and w_bearish and not d_bearish:
                    if _weekly_stalling():
                        return True, -0.05 + _ext_bonus, True

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
                return True, -0.05 + _ext_bonus, True
            if is_reversal_type and is_long  and _w_neutral and h4_bearish:
                return True, -0.05 + _ext_bonus, True

            # ── TIER 5: Overextension reversal — Z > 2, pattern agrees ────
            # Even when no standard tier fires (e.g. all trends neutral but
            # price is at a 2-year extreme), a clear reversal pattern at an
            # extreme IS a valid setup. Alex's USD/JPY Week 8 shorts happened
            # precisely when daily/weekly trends were all "bullish" — but Z was
            # ~+3.5. The overextension IS the signal.
            if _ext_high and is_short and is_reversal_type:
                return True, +0.10, True   # high confidence — extreme + pattern
            if _ext_low  and is_long  and is_reversal_type:
                return True, +0.10, True

            # ── BLOCK: insufficient context for all other cases ────────────
            return False, 0.0, False

        # Find the best qualifying pattern (highest clarity that passes context)
        matching_pattern  = None
        trade_direction   = None
        _reversal_context = False
        _context_adj      = 0.0

        # ── Pattern proximity sort (PATTERN_PREFER_PROXIMITY lever) ──────────
        # Re-rank patterns so that those nearest current price are evaluated
        # first. Prevents old, far-away patterns (high clarity but stale) from
        # outranking fresh patterns near current price.
        # Primary key: combined distance of pattern_level + neckline to price.
        # Secondary key: clarity (higher = better) as tiebreaker.
        # Example: June DT at 200.698 (241 pips away) loses to July DT at
        # 206.422 (86 pips away) even if June clarity=1.00 vs July clarity=0.98.
        if _cfg.PATTERN_PREFER_PROXIMITY and current_price > 0:
            patterns = sorted(
                patterns,
                key=lambda p: (
                    (abs(p.pattern_level - current_price)
                     + abs(p.neckline - current_price)) / current_price,
                    -p.clarity,          # tiebreak: higher clarity first
                ),
            )

        for p in patterns:
            if p.clarity < self.min_pattern_clarity:
                continue
            # Lever: ALLOW_BREAK_RETEST — skip break_retest patterns when disabled.
            # Default True (Alex uses them). False = core reversal patterns only.
            if not _cfg.ALLOW_BREAK_RETEST and 'break_retest' in p.pattern_type:
                continue
            # Proximity window: how far from the neckline can current price be?
            # Standard:  2.0% — price is near the neckline (break or retest)
            # Sweep:     3.5% — liquidity sweeps can pull price further
            # Overext:   5.0% — at price extremes, Alex enters at the RIGHT
            #            SHOULDER or double-top HIGH (far above the neckline).
            #            USD/JPY SHORT at 158: neckline ~155, entry at 158 = 1.9%.
            #            Relaxing to 5% catches these pattern-extreme entries.
            _is_overext_reversal = (
                (_ext_high and p.direction == 'bearish') or
                (_ext_low  and p.direction == 'bullish')
            )
            max_neckline = (
                MAX_NECKLINE_PCT_SWEEP      if 'sweep'  in p.pattern_type else
                MAX_NECKLINE_PCT_STRUCTURAL * 2.5 if _is_overext_reversal   else
                MAX_NECKLINE_PCT_STRUCTURAL
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

        # A structural level that was tested N+ times with additional confluence
        # (EMA, equal highs/lows, or near a round number) = equivalent to a round number.
        # Score threshold controlled by STRUCTURAL_LEVEL_MIN_SCORE lever (default 3).
        # Max distance controlled by STRUCTURAL_LEVEL_MAX_DIST_PCT lever (default 1.5%).
        at_structural_level = (
            nearest_level is not None
            and nearest_level.score >= _cfg.STRUCTURAL_LEVEL_MIN_SCORE
            and any("structural" in c for c in nearest_level.confluences)          # must be a structural level
            and abs(nearest_level.price - matching_pattern.pattern_level)
                / max(matching_pattern.pattern_level, 0.0001) < _cfg.STRUCTURAL_LEVEL_MAX_DIST_PCT
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

        # ── Peak/trough direction check (REQUIRE_PEAKS_AT_LEVEL lever) ──
        # For double_top SHORT: peaks must be AT the round number, not still
        # approaching it from below. Peaks below the round number = price is
        # still heading toward the level = continuation, NOT a reversal.
        # AUD/NZD Jul 2024: peaks at 1.099, round number 1.10 above → blocked.
        # For double_bottom LONG: troughs must be AT or below the round number.
        if not self._peaks_at_level(matching_pattern, pair):
            failed_filters.append("peaks_below_level")
            return TradeDecision(
                decision=Decision.WAIT,
                pair=pair,
                direction=trade_direction,
                reason=(
                    f"{matching_pattern.pattern_type} at {matching_pattern.pattern_level:.5f}: "
                    f"peaks/troughs are still APPROACHING the round number from below/above. "
                    f"Price is continuing toward the level, not being rejected from it. "
                    f"Wait for price to actually reach and reject."
                ),
                confidence=0.20,
                failed_filters=failed_filters,
                pattern=matching_pattern,
                trend_weekly=trend_w,
                trend_daily=trend_d,
                trend_4h=trend_4,
            )

        # ── Neckline level anchor (REQUIRE_NECKLINE_AT_LEVEL lever) ─────────
        # For H&S, IH&S and break_retest patterns: the neckline/break level
        # must sit within NECKLINE_LEVEL_TOLERANCE_PCT of a round number.
        # NZD/CAD Sep 2024: neckline at 0.835, nearest round level 0.840 —
        # breaking an arbitrary structure point is momentum, not a reversal.
        if not self._neckline_at_level(matching_pattern, pair):
            failed_filters.append("neckline_not_at_level")
            nearest = self._nearest_round_number(matching_pattern.pattern_level, pair)
            return TradeDecision(
                decision=Decision.WAIT,
                pair=pair,
                direction=trade_direction,
                reason=(
                    f"{matching_pattern.pattern_type} neckline at "
                    f"{matching_pattern.pattern_level:.5f} is not near a round number "
                    f"(nearest: {nearest:.5f}, tolerance: "
                    f"{matching_pattern.pattern_level * _cfg.NECKLINE_LEVEL_TOLERANCE_PCT:.5f}). "
                    f"Set-and-forget trades anchor to psychological levels humans watch."
                ),
                confidence=0.15,
                failed_filters=failed_filters,
                pattern=matching_pattern,
                trend_weekly=trend_w,
                trend_daily=trend_d,
                trend_4h=trend_4,
            )

        # ── Weekly candle agreement (REQUIRE_WEEKLY_CANDLE_AGREEMENT lever) ─
        # The current forming weekly candle must not be running strongly
        # AGAINST the trade direction. A daily bearish signal inside a massive
        # bull week is profit-taking, not a reversal.
        # USD/JPY Sep 2024: daily bearish engulfing on Sep 26 — but the
        # Sep 23-27 weekly closed +6.4 pips BULL. Weekly told the truth.
        if _cfg.REQUIRE_WEEKLY_CANDLE_AGREEMENT and len(df_weekly) >= 1:
            w_open  = float(df_weekly['open'].iloc[-1])
            w_close = float(df_weekly['close'].iloc[-1])
            w_body  = w_close - w_open
            ref_price = matching_pattern.pattern_level
            tol = ref_price * _cfg.WEEKLY_AGREEMENT_MAX_OPPOSING_BODY_PCT
            weekly_blocked = (
                (trade_direction == 'SHORT' and w_body > tol) or
                (trade_direction == 'LONG'  and w_body < -tol)
            )
            if weekly_blocked:
                failed_filters.append("weekly_candle_opposing")
                direction_word = "bullish" if w_body > 0 else "bearish"
                return TradeDecision(
                    decision=Decision.WAIT,
                    pair=pair,
                    direction=trade_direction,
                    reason=(
                        f"Weekly candle is forming strongly {direction_word} "
                        f"(body={w_body:+.5f}, threshold=±{tol:.5f}). "
                        f"A {trade_direction} daily signal inside an opposing weekly "
                        f"is institutional profit-taking, not a reversal. "
                        f"Wait for weekly to close in alignment."
                    ),
                    confidence=0.20,
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
                if _cfg.NEWS_FILTER_ENABLED and self.news_filter.is_news_candle(last_candle_ts, pair=pair):
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

        # Macro theme position sizing: if this is a stacked macro theme trade,
        # divide the normal risk % by the number of positions in the theme.
        # Total exposure stays the same — just spread across correlated pairs.
        _effective_risk_pct = self.risk_pct
        _macro_note = ""
        if macro_theme and _is_macro_theme_pair:
            _effective_risk_pct = self.risk_pct * macro_theme.position_fraction
            _macro_note = (f" | 📊 MACRO THEME: {macro_theme.currency} "
                          f"{macro_theme.direction.upper()} (score={macro_theme.score:.1f}, "
                          f"{macro_theme.trade_count} trades, "
                          f"{_effective_risk_pct:.1f}% risk each)")

        risk_dollars = self.account_balance * (_effective_risk_pct / 100)
        rr_1 = abs(entry_price - target_1) / max(abs(entry_price - stop_loss), 0.000001)

        # ── Minimum R:R quality gate ───────────────────────────────────────
        # If the pattern's measured move (T1 amplitude) is less than MIN_RR× the
        # stop distance, the pattern is geometrically weak — either stop is
        # too wide or the measured move is tiny. Both are quality failures.
        # Alex's trades all had measured moves well beyond their stop distances.
        # 1:0.4 R:R (like AUD/CAD IH&S Jul 2024) = clear junk setup.
        # Controlled by MIN_RR lever in strategy_config (default 1.0).
        if rr_1 < _cfg.MIN_RR:
            return TradeDecision(
                decision=Decision.WAIT,
                pair=pair,
                direction=trade_direction,
                reason=(
                    f"R:R too low ({rr_1:.2f} < {_cfg.MIN_RR:.1f} minimum). "
                    f"Pattern amplitude too small vs stop distance. "
                    f"{matching_pattern.pattern_type.upper()} at {entry_price:.5f} "
                    f"SL={stop_loss:.5f} T1={target_1:.5f}"
                ),
                confidence=confidence * 0.5,
                failed_filters=["rr_minimum"],
            )

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
                f"{_macro_note}"
            ),
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            risk_pct=_effective_risk_pct,
            lot_size=lot_size,
            nearest_level=nearest_level,
            pattern=matching_pattern,
            entry_signal=signal,
            trend_weekly=trend_w,
            trend_daily=trend_d,
            trend_4h=trend_4,
            neckline_ref=neckline_ref,
            is_macro_theme_pair=_is_macro_theme_pair,
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
