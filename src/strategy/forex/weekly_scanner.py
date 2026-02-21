"""
Weekly Forex Scanner — Automated "Sunday Swings"

This replaces the manual Sunday analysis process from the research:
  - Every Sunday before market open, scan all watched pairs
  - Score each pair by setup quality
  - Flag the top 2-3 pairs for the week
  - Set price alerts at key levels

The human (or automated monitor) then watches only flagged pairs.
When a flagged pair reaches a key level → activate entry mode.
When entry mode → wait for engulfing candle → enter.

This is the upstream funnel. Quality here = quality entries all week.
"""
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

from .set_and_forget import SetAndForgetStrategy, TradeDecision, Decision
from .pattern_detector import PatternDetector, Trend
from .level_detector import LevelDetector

logger = logging.getLogger(__name__)


class SetupStatus(Enum):
    PRIME     = "PRIME"       # Setup active, at level, entry signal possible
    WATCHING  = "WATCHING"    # Setup forming, not at level yet
    EARLY     = "EARLY"       # Good structure but still developing
    SKIP      = "SKIP"        # No qualifying setup this week


@dataclass
class PairSetup:
    pair: str
    status: SetupStatus
    direction: Optional[str]          # 'long' or 'short'
    score: float                       # 0–10 overall setup quality
    trend_weekly: Optional[Trend]
    trend_daily: Optional[Trend]
    trend_4h: Optional[Trend]
    key_level: float                   # price level to watch
    key_level_score: int               # confluence count
    key_level_confluences: List[str]
    alert_price: float                 # set alert here (at or near the level)
    notes: str
    decision: Optional[TradeDecision] = None

    def __str__(self):
        stars = "★" * int(self.score) + "☆" * (10 - int(self.score))
        return (
            f"{self.pair:12} [{self.status.value:8}] {stars} "
            f"{'↑' if self.direction=='long' else '↓' if self.direction=='short' else '?'} "
            f"Level: {self.key_level:.5f} (score={self.key_level_score}) "
            f"Alert@{self.alert_price:.5f} | {self.notes}"
        )


class WeeklyScanner:
    """
    Scans a watchlist of pairs and produces a ranked setup list for the week.

    Usage:
        scanner = WeeklyScanner(strategy)
        results = scanner.scan(pair_data_dict)
        scanner.print_weekly_brief(results)

    pair_data_dict format:
        {
            "EUR/USD": {
                "weekly": df_weekly,
                "daily": df_daily,
                "4h": df_4h,
                "1h": df_1h,
            },
            ...
        }
    """

    # Default watchlist — the pairs most used in the research
    DEFAULT_WATCHLIST = [
        "USD/JPY",   # Most consistent in research — strong trends
        "GBP/CHF",   # Week 6 big trade pair
        "USD/CHF",   # Week 13 final million-dollar pair
        "USD/CAD",   # Week 11 comeback pair
        "GBP/JPY",   # Week 1 first trade pair
        "EUR/USD",   # Most liquid, tightest spreads
        "GBP/USD",   # High volatility, big moves
        "NZD/USD",   # Appeared multiple weeks
        "GBP/NZD",   # Week 12 pair
        "EUR/GBP",   # Regular analysis pair
        "AUD/USD",   # Later weeks
        "NZD/JPY",   # Later weeks
    ]

    def __init__(
        self,
        strategy: SetAndForgetStrategy,
        max_pairs_to_trade: int = 3,
    ):
        self.strategy = strategy
        self.max_pairs = max_pairs_to_trade
        self.pattern_detector = PatternDetector()
        self.level_detector   = LevelDetector()

    def scan(self, pair_data: Dict[str, Dict[str, pd.DataFrame]]) -> List[PairSetup]:
        """
        Scan all pairs and return ranked setup list.
        """
        setups = []

        for pair, data in pair_data.items():
            df_w  = data.get('weekly')
            df_d  = data.get('daily')
            df_4h = data.get('4h')
            df_1h = data.get('1h')

            if any(df is None or len(df) < 10 for df in [df_w, df_d, df_4h, df_1h]):
                logger.warning(f"{pair}: Insufficient data, skipping")
                continue

            setup = self._evaluate_pair(pair, df_w, df_d, df_4h, df_1h)
            setups.append(setup)

        # Sort: PRIME first, then by score descending
        priority = {SetupStatus.PRIME: 0, SetupStatus.WATCHING: 1,
                    SetupStatus.EARLY: 2, SetupStatus.SKIP: 3}
        setups.sort(key=lambda s: (priority[s.status], -s.score))
        return setups

    def top_picks(self, setups: List[PairSetup]) -> List[PairSetup]:
        """Return the top N pairs to watch this week."""
        watchable = [s for s in setups if s.status != SetupStatus.SKIP]
        return watchable[:self.max_pairs]

    def print_weekly_brief(self, setups: List[PairSetup]):
        """Print a formatted weekly setup brief."""
        print("\n" + "="*80)
        print("WEEKLY FOREX SETUP BRIEF — SET & FORGET STRATEGY")
        print("="*80)

        prime  = [s for s in setups if s.status == SetupStatus.PRIME]
        watch  = [s for s in setups if s.status == SetupStatus.WATCHING]
        early  = [s for s in setups if s.status == SetupStatus.EARLY]
        skip   = [s for s in setups if s.status == SetupStatus.SKIP]

        if prime:
            print("\n🎯 PRIME SETUPS — AT LEVEL, WATCHING FOR ENTRY SIGNAL:")
            for s in prime:
                print(f"   {s}")
                if s.decision:
                    print(f"   └─ {s.decision.reason}")

        if watch:
            print("\n👁  WATCHING — SETUP FORMING, NOT YET AT LEVEL:")
            for s in watch:
                print(f"   {s}")

        if early:
            print("\n📋 EARLY STAGE — GOOD STRUCTURE, SET ALERTS:")
            for s in early:
                print(f"   {s}")

        if skip:
            print(f"\n⏭  SKIPPED ({len(skip)} pairs): " + ", ".join(s.pair for s in skip))

        top = self.top_picks(setups)
        print("\n" + "="*80)
        print(f"TOP {len(top)} PAIRS FOR THIS WEEK:")
        for i, s in enumerate(top, 1):
            print(f"  {i}. {s.pair:12} → {s.direction or '?':5} at {s.key_level:.5f}  "
                  f"(Alert@{s.alert_price:.5f})")
        print("="*80)
        print("\nRemember:")
        print("  • Price reaching the level is NOT an entry")
        print("  • Wait for the 1H engulfing candle at the level")
        print("  • Set SL, no TP, set and forget")
        print("  • One trade at a time")
        print("  • If nothing qualifies this week → take no trades\n")

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _evaluate_pair(
        self, pair: str,
        df_w: pd.DataFrame, df_d: pd.DataFrame,
        df_4h: pd.DataFrame, df_1h: pd.DataFrame
    ) -> PairSetup:

        current_price = df_1h['close'].iloc[-1]

        # Trends
        trend_w = self.pattern_detector.detect_trend(df_w)
        trend_d = self.pattern_detector.detect_trend(df_d)
        trend_4 = self.pattern_detector.detect_trend(df_4h)

        # Determine direction
        bullish = sum(1 for t in [trend_w, trend_d, trend_4]
                      if t in (Trend.BULLISH, Trend.STRONG_BULLISH))
        bearish = sum(1 for t in [trend_w, trend_d, trend_4]
                      if t in (Trend.BEARISH, Trend.STRONG_BEARISH))

        if bullish >= 2:
            direction = 'long'
        elif bearish >= 2:
            direction = 'short'
        else:
            direction = None

        # Key levels
        levels = self.level_detector.detect(df_d, current_price)
        if not levels:
            return PairSetup(
                pair=pair, status=SetupStatus.SKIP, direction=direction,
                score=0.0, trend_weekly=trend_w, trend_daily=trend_d, trend_4h=trend_4,
                key_level=current_price, key_level_score=0, key_level_confluences=[],
                alert_price=current_price, notes="No key levels detected",
            )

        # Find best level matching direction
        best_level = None
        for l in levels:
            if direction == 'long' and l.level_type != 'support':
                continue
            if direction == 'short' and l.level_type != 'resistance':
                continue
            best_level = l
            break

        if best_level is None:
            best_level = levels[0]  # use closest regardless

        # Patterns
        patterns = self.pattern_detector.detect_all(df_d)
        best_pattern = None
        for p in patterns:
            expected_dir = 'bearish' if direction == 'short' else 'bullish'
            if p.direction == expected_dir and p.clarity >= 0.3:
                best_pattern = p
                break

        # Entry check (is price at the level right now?)
        at_level = best_level.distance_pct <= 0.3

        # Run full strategy evaluation
        try:
            decision = self.strategy.evaluate(
                pair=pair,
                df_weekly=df_w, df_daily=df_d, df_4h=df_4h, df_1h=df_1h,
                current_price=current_price,
            )
        except Exception as e:
            logger.error(f"{pair}: Strategy evaluation failed: {e}")
            decision = None

        # Score the setup (0–10)
        score = 0.0
        if direction is not None:        score += 1.5
        if bullish >= 3 or bearish >= 3: score += 1.0  # strong alignment
        if best_level.score >= 3:        score += 2.0
        elif best_level.score >= 2:      score += 1.0
        if best_pattern:                 score += best_pattern.clarity * 2
        if at_level:                     score += 1.5
        if decision and decision.decision == Decision.ENTER: score += 1.0

        score = min(10.0, score)

        # Status
        if decision and decision.decision == Decision.ENTER:
            status = SetupStatus.PRIME
        elif at_level and direction and best_pattern:
            status = SetupStatus.PRIME
        elif direction and best_level.score >= 2:
            status = SetupStatus.WATCHING
        elif direction:
            status = SetupStatus.EARLY
        else:
            status = SetupStatus.SKIP

        # Alert price: at the key level
        alert_price = best_level.price

        notes_parts = []
        if direction:
            notes_parts.append(f"Trends {bullish}B/{bearish}Be")
        if best_pattern:
            notes_parts.append(best_pattern.pattern_type)
        if at_level:
            notes_parts.append("AT LEVEL")
        notes = " | ".join(notes_parts) if notes_parts else "No qualifying setup"

        return PairSetup(
            pair=pair,
            status=status,
            direction=direction,
            score=score,
            trend_weekly=trend_w,
            trend_daily=trend_d,
            trend_4h=trend_4,
            key_level=best_level.price,
            key_level_score=best_level.score,
            key_level_confluences=best_level.confluences,
            alert_price=alert_price,
            notes=notes,
            decision=decision,
        )
