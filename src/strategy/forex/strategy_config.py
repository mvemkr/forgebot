"""
strategy_config.py — Single Source of Truth for All Execution Filters
======================================================================

THIS IS THE ONLY PLACE THESE CONSTANTS ARE DEFINED.

Both the backtester (oanda_backtest_v2.py) and the live orchestrator
(orchestrator.py) import from here. If you need to change a threshold,
change it HERE. Do not hardcode it in the backtester or the live bot
separately — that's exactly how they get out of sync.

Rule: if a filter exists in the backtester, it must exist identically
in the live bot, and vice versa. Both must import from this file.
See GUIDELINES.md for the full policy.

LEVER SYSTEM
============
Every filter is a named lever. To run a one-off experiment without editing
source code, use the backtester's --lever and --profile flags:

    python3 -m backtesting.oanda_backtest_v2 --window alex \\
        --lever LEVEL_ALLOW_FINE_INCREMENT=False \\
        --lever ALLOW_BREAK_RETEST=False

Or load a named profile:

    python3 -m backtesting.oanda_backtest_v2 --window jan --profile core_reversals

apply_levers(overrides) patches module globals at runtime so that
set_and_forget.py (which imports this module by reference) sees the changes.
"""
import sys as _sys

# ── Signal quality gates ───────────────────────────────────────────────────
# Minimum confidence score to execute a trade.
# Signals below this are logged as "forming" but never executed.
# Tuned via backtest: 0.65 filters ~30% of marginal setups, improves WR.
MIN_CONFIDENCE: float = 0.77

# Minimum pip equity (measured move in pips) required to enter.
# Blocks low-potential setups that consume a slot but can't run far.
# EUR/CAD at 62p blocked EUR/CAD Sep 23 entry, preventing GBP/CHF
# from entering 2 days later with a 226p setup.
# 150p = meaningful move for Alex-style set-and-forget trades.
MIN_PIP_EQUITY: float = 100.0

# Pips from target at which we consider price to have "reached the target area."
# Alex monitors actively and closes when price touches his 4H level — High/Low
# touch within this threshold triggers the backtester exit (and live alert).
TARGET_PROXIMITY_PIPS: float = 15.0

# Minimum R:R ratio based on pattern amplitude vs stop distance.
# Not a take-profit target — geometric quality check only.
# Blocks patterns where the measured move is smaller than the stop.
MIN_RR: float = 2.5   # minimum exec R:R — select_target() rejects any candidate below this

# ── Small-account / Alex money-game rules ─────────────────────────────────
# Alex treats equity < $25K as "money game" — capital preservation is the
# primary objective.  Dollar caps become primary; percent caps secondary.
# Equity ≥ $25K flips back to percent-cap-primary mode.
#
# These constants drive the backtester and live risk manager; all gates
# log their reason codes for funnel reporting.
SMALL_ACCOUNT_THRESHOLD:  float = 25_000.0   # USD equity boundary
MIN_RR_STANDARD:          float = 2.5        # pro-trend baseline (W+D+4H all agree)
# ── PROD CONFIG LOCKED: Arm D (2026-02-26) ──────────────────────────────────
# Dynamic 3.0R non-protrend gate removed — net negative in W1 (kills +5.80R
# winner, doubles DD 13→30%). Both pro-trend and non-protrend use flat 2.5R.
# select_target() already guarantees exec_rr ≥ MIN_RR before entry; this alias
# being equal to MIN_RR makes check_dynamic_min_rr() a no-op gate (always pass).
MIN_RR_COUNTERTREND:      float = MIN_RR     # Arm D: flat 2.5R — no protrend penalty
MIN_RR_SMALL_ACCOUNT:     float = MIN_RR_COUNTERTREND  # legacy alias — kept for compat

# Weekly punch-card behaviour — Alex: "pretend you only have 2 trades per month"
# Implementation: per-calendar-week cap (ISO week).
MAX_TRADES_PER_WEEK_SMALL:    int = 1   # equity < SMALL_ACCOUNT_THRESHOLD
MAX_TRADES_PER_WEEK_STANDARD: int = 2   # equity ≥ SMALL_ACCOUNT_THRESHOLD

# HTF trend alignment — require ALL of weekly/daily/4H to agree with direction.
# When True, block counter-trend entries with reason COUNTERTREND_HTF.
# NOTE: Alex trades counter-trend at extremes; set False to preserve his style.
# Default True here for measurement; production default may differ after backtests.
REQUIRE_HTF_TREND_ALIGNMENT:  bool = True

# Time-based session rules (Alex: "no Thu/Fri, not worth the spread")
# NO_SUNDAY_TRADES: block all of Sunday (forex wick creation period)
# NO_THU_FRI_TRADES: block Thu from 09:00 ET onward and all Friday
NO_SUNDAY_TRADES_ENABLED:    bool = True
NO_THU_FRI_TRADES_ENABLED:   bool = True
THU_ENTRY_CUTOFF_HOUR_ET:     int  = 9   # 9:00 AM ET = start of block on Thu

# ── ATR stop filter ────────────────────────────────────────────────────────
# Stop must be ≤ ATR_STOP_MULTIPLIER × 14-day ATR.
# Rejects entries where the pattern stop is wider than typical volatility
# would allow the trade to survive (ancient structural levels).
# Raised from 5x → 8x to allow Alex's USD/CHF trade (5-7.5x ATR stop).
ATR_STOP_MULTIPLIER: float = 8.0

# Stop must be ≥ ATR_MIN_MULTIPLIER × 14-day ATR.
# Rejects micro-stops from 4H patterns that would be killed by daily noise.
# Set conservatively — Alex's real 4H stops are typically 0.15-0.50× daily ATR.
ATR_MIN_MULTIPLIER: float = 0.15

# Lookback window for ATR calculation (calendar days of daily candles).
ATR_LOOKBACK: int = 14

# ── Concurrency limits ─────────────────────────────────────────────────────
# PARITY RULE: backtest must mirror live. Default = 1 for both.
# Changing BACKTEST without changing LIVE produces meaningless comparisons —
# different exposure, different compounding, different drawdown paths.
#
#   MAX_CONCURRENT_TRADES_LIVE     — orchestrator + risk_manager (real account).
#   MAX_CONCURRENT_TRADES_BACKTEST — backtester default. Must equal LIVE.
#
# Want to test multi-position behaviour? Use the CLI flag --max-trades N
# in oanda_backtest_v2.py. Those runs are tagged EXPERIMENTAL in results
# and notes so they're never confused with parity runs.
#
# Note on Alex: he runs 4+ simultaneous positions in his challenge. That's
# irrelevant here — this bot is designed around 1 position at a time by
# explicit choice (simple rules, clear attribution, matches live constraints
# at $4K–$30K account size). Alex's multi-position behaviour is documented
# for reference only; it is NOT the default operating mode.
#
# Macro theme stacking (JPY/CHF carry) is a separate concept — handled by
# STACK_MAX in currency_strength.py and always bypasses this cap.
MAX_CONCURRENT_TRADES_LIVE:     int = 1   # live — one position at a time
MAX_CONCURRENT_TRADES_BACKTEST: int = 1   # backtest — must equal live (parity rule)
MAX_CONCURRENT_TRADES:          int = MAX_CONCURRENT_TRADES_LIVE  # alias

# ── Alex's confirmed watchlist (extracted from first video screenshot) ─────────
# Only evaluate pairs on this list. Anything else is outside Alex's universe.
# Removed: EUR/GBP, EUR/NZD, AUD/NZD (not on Alex's watchlist)
# Added: GBP/CAD, AUD/JPY, CAD/JPY, EUR/JPY (on his list, were missing from bot)
# JPY crosses: Alex watches ALL 6 — key for theme stacking (Week 7-8 $70K)
ALLOWED_PAIRS: frozenset = frozenset({
    # Alex's exact trading universe — extracted from full transcript.
    # He trades ONLY these 7 pairs across all 13 weeks. No exceptions.
    "GBP/JPY",   # Wk1 (short 205), largest position
    "USD/JPY",   # Wk2 (short 157.5), Wk8 (short 144)
    "USD/CHF",   # Wk3 (short 0.88), Wk13 (long 0.876)
    "GBP/CHF",   # Wk6/7 (short 1.125), Wk12b (short 1.124)
    "USD/CAD",   # Wk10/11 (short 1.37)
    "EUR/USD",   # mentioned as reference pair, Wk4 skip
    "GBP/USD",   # mentioned in channel discussion
})

# ── Winner rule ("don't compete with your winner") ────────────────────────
# When an open position is past breakeven (≥1R in profit), block all new entries.
#
# Rationale: Alex's behaviour — once a trade is running and safe (stop at BE),
# he's not opening new positions. Every second-trade loss in the Jan 2026
# backtest happened while NZD/CAD was already past breakeven and running.
# The progressive confluence gate raised the confidence bar but didn't ask
# the right question: SHOULD we even be adding a position right now?
#
# "Winner" definition: any open position whose stop has been moved to breakeven.
# In the backtester: pos["be_moved"] == True
# In the live bot:   pos["stop"] == pos["entry"]
# Both conditions mean exactly the same thing — price traveled ≥1R in our favor.
#
# Rule: winner running → no new entries. Full stop.
# No exceptions for high-confidence setups — the whole point is simplicity.
BLOCK_ENTRY_WHILE_WINNER_RUNNING: bool = False

# ── Entry trigger mode ─────────────────────────────────────────────────────
# Controls which 1H candlestick pattern is accepted as an entry trigger.
#
#   "engulf_only"   — only bearish/bullish engulfing closes.
#                     Alex's stated rule in every video: "No engulfing candle =
#                     no trade." Confirmed as the #1 filter that separates his
#                     real trades from near-misses. Production default.
#
#   "engulf_or_pin" — also allows tight pin bars meeting the strict spec below.
#                     Use for research runs to measure whether pin-bar retest
#                     rejections add independent edge, not as a prod default.
#                     Backtest tag: "pin_bars_allowed"
#
# To compare: run backtester with ENTRY_TRIGGER_MODE overridden per arm.
# Never change this to "engulf_or_pin" in production without data supporting it.
ENTRY_TRIGGER_MODE: str = "engulf_only"
# Valid modes (sourced from Alex's confirmed "daily driver" confirmations):
#   "engulf_only"                         — baseline: engulfing only (Alex #1 rule)
#   "engulf_or_pin"                       — engulfing OR loose pin bar
#   "engulf_or_star_at_level"             — engulfing OR Morning/Evening Star at tested level
#   "engulf_or_strict_pin_at_level"       — engulfing OR strict hammer/shooting star at level
#   "engulf_or_star_or_strict_pin_at_level" — all three at-level confirmations
# PRODUCTION DEFAULT: engulf_only.  Do NOT change without backtest support.

# Backward-compatible derived flag — all existing call sites work unchanged.
# Do not set ENGULFING_ONLY directly; change ENTRY_TRIGGER_MODE instead.
ENGULFING_ONLY: bool = (ENTRY_TRIGGER_MODE == "engulf_only")

# ── Pin bar entry spec ─────────────────────────────────────────────────────
# Only active when ENTRY_TRIGGER_MODE == "engulf_or_pin".
PIN_BAR_MIN_WICK_BODY_RATIO: float = 2.0   # wick ≥ 2× body
PIN_BAR_CLOSE_OUTER_PCT:     float = 0.30  # close in outer 30% (trade-direction end)

# ── At-level candle confirmation spec ─────────────────────────────────────
# Governs Morning/Evening Star and Strict Hammer/Shooting Star modes.
# ALL of the following are only valid when price has RECENTLY TESTED the level.
#
# Level-touch check:  retest zone is the last LEVEL_TOUCH_H1_BARS H1 bars.
# "5 M15 bars" ≈ 75 min ≈ 1-2 H1 bars; we use 5 H1 bars as conservative window.
LEVEL_TOUCH_H1_BARS:        int   = 5       # how many recent H1 bars to check for level touch
LEVEL_TOUCH_TOLERANCE_PCT:  float = 0.0015  # price within 0.15% of level = "touched"

# Morning/Evening Star thresholds
STAR_BAR1_MIN_BODY_PCT:     float = 0.45    # bar-1 body ≥ 45% of its range (large candle)
STAR_BAR2_MAX_BODY_PCT:     float = 0.30    # bar-2 body ≤ 30% of its range (small star)
STAR_BAR3_MIN_BODY_PCT:     float = 0.40    # bar-3 body ≥ 40% of its range (confirmation)
STAR_BAR3_MIDPOINT_CLOSE:   float = 0.50    # bar-3 close must pass mid of bar-1 by this fraction

# Strict Hammer / Shooting Star thresholds (stricter than regular pin bar)
STRICT_HAMMER_WICK_BODY_RATIO:     float = 3.0   # rejection wick ≥ 3× body (vs 2× for pin)
STRICT_HAMMER_CLOSE_OUTER_PCT:     float = 0.35  # close in outer 35% of range
STRICT_HAMMER_MAX_OPPOSITE_PCT:    float = 0.10  # opposite wick ≤ 10% of range
STRICT_HAMMER_MAX_BODY_RANGE_PCT:  float = 0.25  # body ≤ 25% of range (not a big candle)

# ── Doji / Indecision candle filter ───────────────────────────────────────
# Blocks entries where the TRIGGER CANDLE is an indecision candle.
# Objective definition (from transcript): body ≤ 15% of range AND
# both upper AND lower wicks ≥ 35% of range (classic doji / spinning top).
#
# Exception: Morning/Evening Star — bar2 (the "star") MAY be indecision;
# bar3 (the final confirmation) must still pass.
#
# Set INDECISION_FILTER_ENABLED=False to disable globally (backtest comparison).
INDECISION_FILTER_ENABLED:      bool  = True
INDECISION_MAX_BODY_RATIO:      float = 0.15   # body ≤ 15% of range → tiny body
INDECISION_MIN_WICK_RATIO:      float = 0.35   # both wicks ≥ 35% → long-legged

# Unrealized-R threshold that defines a "winner running."
# A position must be THIS many R's in profit RIGHT NOW (at current price)
# to trigger the winner rule and block new entries.
#
# Why not use be_moved (stop at breakeven)?
# be_moved fires at 1R then stays True even if price drifts back to entry
# and the position sits there for months making $0 unrealized.
# A position coasting at 0R unrealized is NOT a winner — it's a free trade.
# A position actively up 2R right now IS a winner worth protecting.
#
# 2.0R means: risk $800, position must be up $1,600+ unrealized to block new entries.
WINNER_THRESHOLD_R: float = 2.0


def winner_rule_check(
    n_open: int,
    max_unrealized_r: float,
) -> tuple:
    """
    "Don't compete with your winner" gate.
    Returns (blocked: bool, reason: str).

    Called by both the backtester and the live orchestrator — same logic,
    same threshold. Constants defined above; change them once, both inherit.

    n_open           : number of currently open positions BEFORE this entry
    max_unrealized_r : highest current unrealized R across all open positions
                       (computed from live price vs entry/stop, NOT be_moved flag)
    """
    if n_open == 0 or not BLOCK_ENTRY_WHILE_WINNER_RUNNING:
        return False, ""

    if max_unrealized_r >= WINNER_THRESHOLD_R:
        return True, (
            f"winner_rule: open position at {max_unrealized_r:.1f}R unrealized — "
            f"don't compete with your winner (threshold={WINNER_THRESHOLD_R:.1f}R). "
            f"Let it run uncontested."
        )

    return False, ""

# ── Session windows (UTC hours) ────────────────────────────────────────────
# Only auto-execute during London session (3–8 AM ET = 08–13 UTC).
# Outside this window: detect signals and notify, but do not execute.
# Alex always enters London session, sets alarm, goes to sleep.
LONDON_SESSION_START_UTC: int = 8
LONDON_SESSION_END_UTC:   int = 13

# ── Re-entry cooldowns ─────────────────────────────────────────────────────
# Days to wait before re-entering a pair after a stop-out.
# Prevents immediately re-entering the same setup that just failed.
STOP_COOLDOWN_DAYS: float = 5.0

# ── Pattern memory ─────────────────────────────────────────────────────────
# Cluster tolerance for neckline deduplication.
# Patterns within 0.3% of each other on the same pair are treated as one.
NECKLINE_CLUSTER_PCT: float = 0.003

# ── Dry run / paper trading ────────────────────────────────────────────────
# Virtual account balance when OANDA is unfunded (dry_run=True, real balance < $100).
# Used for risk sizing, kill-switch math, and P&L tracking.
# Set this to the capital you intend to fund with.
DRY_RUN_PAPER_BALANCE: float = 8_000.0

# ── Backtest candle cache ──────────────────────────────────────────────────
# Version string baked into every per-pair/TF cache filename.
# Bump this to v2, v3 etc. to invalidate all on-disk caches globally
# (e.g. after changing OANDA price_type, instrument mapping, or TF set).
# Cache dir: ~/.cache/forge_backtester/
CACHE_VERSION: str = "v1"


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY LEVERS — toggleable filters, all with defaults matching Alex's rules
# ══════════════════════════════════════════════════════════════════════════════
# Override at runtime via apply_levers() or the backtester's --lever CLI flag.
# Default state = current production config. Change a lever here to change both
# the backtester AND the live bot simultaneously.

# ── Level quality levers ───────────────────────────────────────────────────
# Whether 0.01-increment price levels count as "major psychological levels."
# True  → levels like USD/CHF 0.880, GBP/CHF 1.130 are valid anchors.
# False → only 0.025-increment levels qualify (stricter, fewer trades).
# Alex trades both 0.025 and 0.010 increments; False is more conservative.
LEVEL_ALLOW_FINE_INCREMENT: bool = True

# Minimum confluence score for structural S/R levels to substitute for a
# round-number level. Score ≥ N means N confluence factors (EMA, equal
# highs/lows, round number proximity, etc.). 3 = 3 confirmations required.
# Lowering to 2 opens the door to noise levels (AUD/NZD 1.09184).
STRUCTURAL_LEVEL_MIN_SCORE: int = 3

# Max distance (as a fraction of price) between the structural level and the
# pattern's structural price for the level to count as "at the pattern."
# 0.015 = within 1.5% of price. Increase to catch sloppier setups.
STRUCTURAL_LEVEL_MAX_DIST_PCT: float = 0.015

# ── Pattern type levers ────────────────────────────────────────────────────
# Allow break_retest patterns (continuation setups) as entry triggers.
# True  → break & retest valid (requires TIER 1 trend alignment by design).
# False → only reversal patterns: H&S, double top/bottom, IH&S.
# Alex trades both; False tests whether reversals alone outperform.
ALLOW_BREAK_RETEST: bool = True

# ── Multi-timeframe context levers ────────────────────────────────────────
# Overextension check: block trades going WITH a price extreme.
# True  → if price is in top/bottom OVEREXTENSION_THRESHOLD % of 2yr range,
#         only take reversals AGAINST the extension, never with it.
#         This is what caught USD/JPY at 157-160 as a SHORT not a LONG.
# False → overextension analysis is advisory only (no blocks, no bonuses).
OVEREXTENSION_CHECK: bool = True

# Percentile rank threshold that defines "price is at an extreme."
# 0.90 → top or bottom 10% of the 2-year price range.
# Lower = more aggressive (0.80 = top 20% is "extended").
OVEREXTENSION_THRESHOLD: float = 0.90

# Allow Tier 3 early reversals: weekly trend opposing + daily stalling (neutral).
# True  → trades like GBP/CHF Aug 2024 are valid (weekly bullish, daily neutral,
#         double top at high — the pattern IS the reversal signal).
# False → require daily to be actively turning (confirmed reversal only).
ALLOW_TIER3_REVERSALS: bool = True

# ── Peak/trough direction lever ────────────────────────────────────────────
# For double_top SHORT: pattern peaks must be AT or ABOVE the round number,
# not approaching it from below. Peaks below the round number = price is
# still heading toward it = continuation, not reversal.
# For double_bottom LONG: pattern troughs must be AT or BELOW round number.
# This blocked AUD/NZD SHORT (peaks at 1.099, below 1.10 round level —
# price was grinding UP toward 1.10, not being rejected FROM it).
REQUIRE_PEAKS_AT_LEVEL: bool = True

# ── Tier 3 weekly stall gate ───────────────────────────────────────────────
# Tier 3 reversals (weekly opposing + daily stalling) require the most recent
# weekly candle to show momentum compression OR a reversal close, before the
# bot takes a counter-trend position.
# Prevents "fresh trend" reversals: EUR/GBP in 8-week downtrend with no
# pause, AUD/NZD in 4-month uptrend with no pause.
# Alex explicitly looks for weekly doji/narrow bar before acting counter-trend.
TIER3_REQUIRE_WEEKLY_STALL: bool = True
# Recent weekly range must be < this fraction of 4-week avg to count as stalling.
# 0.65 = recent week range less than 65% of prior average = compression/doji.
TIER3_WEEKLY_STALL_RATIO: float = 0.65

# ── Zone touch + reject gate ──────────────────────────────────────────────
# Before checking for an entry trigger (engulf / pin), verify that recent price
# action actually wicked into the zone around the pattern level.
#
# Two tests:
#   1. Touch test: wick reaches within ZONE_TOUCH_ATR_MULT × ATR of the level
#      (0.35×ATR for majors; 0.5×ATR for crosses — broader spreads + noisier)
#   2. Reject test: enforced by the trigger candle itself (engulf body / pin wick)
#
# "Touch" is tested on the last ZONE_TOUCH_LOOKBACK_BARS 1H bars.
# This stops engulfing candles from firing well away from the level —
# a valid setup must have price at least approaching the zone first.
ZONE_REQUIRE_TOUCH: bool = True
ZONE_TOUCH_ATR_MULT: float = 0.35    # majors (EUR/USD, GBP/USD, USD/JPY, etc.)
ZONE_TOUCH_ATR_MULT_CROSS: float = 0.50  # crosses (GBP/JPY, GBP/CHF, EUR/CAD, etc.)
ZONE_TOUCH_LOOKBACK_BARS: int = 5    # look back this many 1H bars for a zone touch

# ── Drawdown circuit breaker ───────────────────────────────────────────────
# Graduated risk caps when account draws down from peak equity.
# Does NOT replace the 40% REGROUP killswitch — works alongside it.
#
# Tiers (applied against peak_equity, not starting balance):
#   DD ≥ DD_L1_PCT → cap risk at DD_L1_CAP_PCT (default 10%)
#   DD ≥ DD_L2_PCT → cap risk at DD_L2_CAP_PCT (default 6%)
#   DD ≥ 40%       → REGROUP killswitch (existing, unchanged)
#
# Full risk resumes only when equity recovers to DD_RESUME_PCT of peak.
# Default 0.95 = must recover to 95% of prior peak before full risk returns.
# This prevents "recovered 1% above the L1 threshold → full risk immediately."
DD_CIRCUIT_BREAKER_ENABLED: bool = True
DD_L1_PCT: float = 15.0    # first cap threshold (% drawdown from peak)
DD_L1_CAP: float = 10.0    # max risk % when DD ≥ L1
DD_L2_PCT: float = 25.0    # second cap threshold (harder brake)
DD_L2_CAP: float = 6.0     # max risk % when DD ≥ L2
DD_RESUME_PCT: float = 0.95 # equity must recover to 95% of peak to lift the cap

# ── DD Kill-switch (hard block, not just throttle) ────────────────────────
# When DD ≥ DD_KILLSWITCH_PCT from peak, get_risk_pct_with_dd() returns
# (0.0, "DD_KILLSWITCH") — the caller MUST block the new entry entirely.
# This is deterministic: any call after breaching this threshold is blocked
# until equity recovers above DD_RESUME_PCT × peak.
DD_KILLSWITCH_PCT: float = 40.0   # hard no-new-entries threshold

# ── Absolute dollar risk cap ─────────────────────────────────────────────
# Prevents "same % → much bigger $ at high equity" compounding blowups.
# Applied after all % caps: final_pct = min(final_pct, max_dollar/balance×100)
# Format: list of (equity_threshold, max_dollar_risk) — last entry is the cap
# for any equity above the previous threshold.
#   equity < $25K  → no dollar cap (% caps are sufficient)
#   $25K–$100K     → cap at $2,500 per trade
#   > $100K        → cap at $5,000 per trade
DOLLAR_RISK_CAP_ENABLED: bool = True
DOLLAR_RISK_TIERS: list = [
    (25_000,       None),    # below $25K: no dollar cap, % tiers dominate
    (100_000,      2_500),   # $25K–$100K: max $2,500 at risk per trade
    (float("inf"), 5_000),   # > $100K: max $5,000 at risk per trade
]

# ── Loss-streak brake ─────────────────────────────────────────────────────
# After N consecutive losses, cap risk until a win resets the streak.
# Targets the variance tail: 4+ loss clusters compound badly at high equity.
#   2 consecutive losses → cap at STREAK_L2_CAP (6%)
#   3+ consecutive losses → cap at STREAK_L3_CAP (3%)
# Streak resets on any winning trade.  Applied AFTER DD caps.
LOSS_STREAK_BRAKE_ENABLED: bool = True
STREAK_L2_LOSSES: int   = 2     # streak length for L2 brake
STREAK_L2_CAP:    float = 6.0   # cap at 6% after 2 consecutive losses
STREAK_L3_LOSSES: int   = 3     # streak length for L3 brake
STREAK_L3_CAP:    float = 3.0   # cap at 3% after 3+ consecutive losses

# ── Spread model (backtester only) ───────────────────────────────────────
# Models bid/ask spread cost as a round-turn deduction from each trade's P&L.
# The backtester uses mid-price candles; real OANDA orders fill at ask (longs)
# or bid (shorts).  Round-turn cost = spread_pips on entry + spread_pips on exit
# = 2 × half_spread.  For v1 we apply the full round-turn as a P&L deduction
# at close (conservative and simple).
#
# Spreads in pips (typical OANDA practice account values):
#   Majors (EUR/USD, GBP/USD, USD/JPY, USD/CHF, USD/CAD):  ~1.0–2.0 pips
#   JPY crosses (GBP/JPY, EUR/JPY):                         ~2.5–4.0 pips
#   Other crosses (GBP/CHF, EUR/CHF, etc.):                 ~3.0–5.0 pips
SPREAD_MODEL_ENABLED: bool = True

# Per-pair round-trip spread in pips.  Missing pairs fall back to SPREAD_DEFAULT_PIPS.
SPREAD_PIPS: dict = {
    "EUR/USD": 1.0,
    "GBP/USD": 1.5,
    "USD/JPY": 1.2,
    "USD/CHF": 2.0,
    "USD/CAD": 2.0,
    "GBP/JPY": 3.0,
    "EUR/JPY": 2.5,
    "GBP/CHF": 4.5,
    "EUR/CHF": 3.5,
    "AUD/USD": 1.5,
    "NZD/USD": 2.0,
    "AUD/JPY": 3.0,
    "EUR/AUD": 3.5,
    "EUR/CAD": 3.0,
    "GBP/AUD": 5.0,
    "GBP/CAD": 5.0,
    "NZD/JPY": 3.5,
    "AUD/CAD": 3.5,
    "AUD/NZD": 3.0,
}
SPREAD_DEFAULT_PIPS: float = 3.0   # fallback for unlisted pairs

# ── One-way ratchet trailing stop ─────────────────────────────────────────
# Option B: replaces hard breakeven lock at 1:1.
# When price moves TRAIL_ACTIVATE_R in our favour, the stop trails
# TRAIL_DIST_R behind the running max-favourable price (never moves backward).
#
# Example (default 1R activate, 0.5R trail distance):
#   MFE=0.9R  → stop stays at initial stop (not activated yet)
#   MFE=1.0R  → trail activates; stop = max_fav - 0.5R = entry+0.5R (floor)
#   MFE=2.0R  → stop = entry + 1.5R (trailing 0.5R behind max)
#   MFE=3.5R  → stop = entry + 3.0R
#   MFE pulls back → stop stays locked at whatever it ratcheted to
#
# Set TRAIL_DIST_R = 0.0 to disable trailing (reverts to hard BE lock).
TRAIL_ACTIVATE_R: float = 1.0   # start trailing once MFE reaches this R multiple
TRAIL_DIST_R:     float = 0.5   # trail this many R-multiples behind running max MFE

# ── D1 strong-opposite veto for reversal patterns ─────────────────────────
# For reversal patterns (H&S, DT, DB, IH&S): only apply a hard block when the
# Daily timeframe is STRONGLY trending opposite to the trade direction.
# A regular "bullish" daily is not sufficient to veto a double top SHORT —
# the pattern IS the reversal signal. Over-blocking here causes us to miss
# the best setups (Alex's most profitable trades are counter-weekly-trend).
#
# "Strongly opposite" = daily trend is STRONG_BULLISH for a SHORT, or
# STRONG_BEARISH for a LONG. Regular BULLISH / BEARISH daily → allowed with penalty.
#
# For continuation patterns (break_retest, consolidation_breakout):
# keep the existing weekly alignment requirement — continuations should flow
# with trend, not fight it.
D1_VETO_ONLY_STRONG_OPPOSITE: bool = True

# ── News filter lever ─────────────────────────────────────────────────────
# Whether to block entries when the triggering candle forms during a
# high-impact news event (NFP, CPI, FOMC, BoE, ECB rate decisions).
# Alex's rule: engulfing candle = entry. He never mentioned avoiding news.
# This filter was added as a safeguard but is NOT in Alex's methodology.
# Default False = matches backtest behavior (news filter has no historical
# data and runs as a no-op). Live bot and backtest must match.
NEWS_FILTER_ENABLED: bool = False

# ── Pattern proximity preference ───────────────────────────────────────────
# When multiple patterns qualify (similar clarity), prefer the one whose
# peaks/troughs (pattern_level) are closest to current price.
#
# Why: the pattern detector returns patterns sorted by clarity. High-clarity
# old patterns (e.g. a June double top at 200.698 when price is at 205.56)
# can outrank newer, more relevant patterns (July double top at 206.422).
# The overextension proximity window (5%) makes this worse by allowing
# patterns whose necklines are 3-4% below price.
#
# With PATTERN_PREFER_PROXIMITY=True, patterns are re-ranked by combined
# proximity of pattern_level + neckline to current price before selection.
# Clarity still acts as a tiebreaker within the same proximity bucket.
PATTERN_PREFER_PROXIMITY: bool = True

# ── Neckline level anchor (Week 9 / NZD-CAD rule) ─────────────────────────
# For H&S, IH&S, and break_retest patterns the neckline/break level must sit
# within NECKLINE_LEVEL_TOLERANCE_PCT of a round number.
#
# Double top/bottom peaks are already gated by REQUIRE_PEAKS_AT_LEVEL.
# This extends that discipline to necklines and break levels.
#
# Why: NZD/CAD Sep 2024 H&S neckline was at ~0.835 — not near 0.840 or any
# meaningful round level. A break of an arbitrary structure level is a momentum
# trade, not a set-and-forget reversal anchored to a psychological price.
# Alex ALWAYS trades levels that humans are watching (0.84, 1.10, 160, etc.).
#
# Tolerance: fraction of price (0.0020 = 20 pips on a 1.000 pair).
REQUIRE_NECKLINE_AT_LEVEL: bool = True
NECKLINE_LEVEL_TOLERANCE_PCT: float = 0.0020

# ── Weekly candle agreement (Week 12 / USD-JPY Diddy rule) ─────────────────
# The currently-forming weekly candle must not be running strongly AGAINST
# the intended trade direction at the time of entry.
#
# Why: USD/JPY Oct 2024 — Alex saw a clean daily bearish engulfing on Sep 26,
# but the Sep 23-27 weekly candle closed +6.4 pips BULL. A daily dip inside
# a massive bull week is institutional profit-taking, not a reversal.
# The weekly context told the truth; the daily candle lied.
#
# Rule: if the current forming weekly body > threshold in the OPPOSING direction,
# block the entry. The weekly narrative overrides the daily signal.
#
# Threshold: fraction of price. 0.0035 = 35 pips on a 1.000 pair (350 pips JPY).
# Tune higher to loosen (fewer blocks), lower to tighten (more blocks).
REQUIRE_WEEKLY_CANDLE_AGREEMENT: bool = True
WEEKLY_AGREEMENT_MAX_OPPOSING_BODY_PCT: float = 0.0035

# ── Theme stacking lever ───────────────────────────────────────────────────
# Whether an active macro theme can BLOCK an entry that contradicts it.
# True  → if USD_strong theme is active and pattern wants EUR/USD LONG,
#         that entry is blocked (EUR = USD counter-currency, bad timing).
# False → theme analysis informs sizing only; never blocks a trade.
REQUIRE_THEME_GATE: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# LEVER RUNTIME SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def apply_levers(overrides: dict) -> dict:
    """
    Patch module-level constants at runtime.

    This enables the backtester's --lever CLI flag to change strategy behaviour
    without editing source files. Because set_and_forget.py imports this module
    BY REFERENCE (import strategy_config as _cfg), it sees patched values
    immediately — no reload needed.

    Type coercion is automatic based on the existing type of each constant.
    Booleans accept: True/False/true/false/1/0/yes/no.

    Returns the dict of applied overrides (useful for logging).
    Raises ValueError for unknown or non-overridable keys.

    Example:
        apply_levers({"LEVEL_ALLOW_FINE_INCREMENT": False, "MIN_CONFIDENCE": 0.70})
    """
    m = _sys.modules[__name__]
    applied = {}
    for key, raw_val in overrides.items():
        existing = getattr(m, key, _MISSING := object())
        if existing is _MISSING:
            raise ValueError(f"apply_levers: unknown lever '{key}'")
        if callable(existing):
            raise ValueError(f"apply_levers: '{key}' is a function, not a lever")
        # Type coercion
        if isinstance(existing, bool):
            if isinstance(raw_val, str):
                val = raw_val.strip().lower() not in ("false", "0", "no", "off")
            else:
                val = bool(raw_val)
        elif isinstance(existing, float):
            val = float(raw_val)
        elif isinstance(existing, int):
            val = int(raw_val)
        elif isinstance(existing, str):
            val = str(raw_val)
        else:
            val = raw_val
        setattr(m, key, val)
        applied[key] = val
    return applied


def load_profile(profile_name: str) -> dict:
    """
    Load a named lever profile from profiles/<name>.json and apply it.
    Returns the dict of applied overrides.

    Profiles live in <repo>/profiles/ relative to this file's package root.
    """
    import json, pathlib
    # Walk up from this file to find the profiles/ directory
    here = pathlib.Path(__file__).resolve()
    # src/strategy/forex/strategy_config.py → go up 3 levels → repo root
    repo_root = here.parents[3]
    profile_path = repo_root / "profiles" / f"{profile_name}.json"
    if not profile_path.exists():
        available = [p.stem for p in (repo_root / "profiles").glob("*.json")]
        raise FileNotFoundError(
            f"Profile '{profile_name}' not found at {profile_path}. "
            f"Available: {available}"
        )
    with open(profile_path) as f:
        overrides = json.load(f)
    # Strip comment/metadata keys (anything starting with "_")
    overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}
    return apply_levers(overrides)


# ── Model tag helper ────────────────────────────────────────────────────────
def get_model_tags(trail_arm: str = "", pairs_hash: str = "") -> list:
    """
    Returns a list of short descriptive tags capturing ALL active levers.
    Written into every backtest result record so you can always reproduce
    exactly what config produced a given number.

    Tags are stable short strings — sort + join them to get a run fingerprint.

    Args:
        trail_arm:   Trail arm key ("A", "B", "C") — injected by run_backtest().
        pairs_hash:  MD5[:6] of sorted active pairs list — injected by run_backtest().
    """
    m = _sys.modules[__name__]
    tags = []

    # ── Trail arm (injected by caller) ───────────────────────────────────────
    if trail_arm:
        tags.append(f"trail_{trail_arm}")

    # ── Peak direction + weekly stall ───────────────────────────────────────
    if not m.REQUIRE_PEAKS_AT_LEVEL:
        tags.append("no_peak_dir_check")
    if not m.TIER3_REQUIRE_WEEKLY_STALL:
        tags.append("no_tier3_stall_gate")
    elif m.TIER3_WEEKLY_STALL_RATIO != 0.65:
        tags.append(f"stall_ratio_{m.TIER3_WEEKLY_STALL_RATIO:.2f}")

    # ── Pattern proximity preference ─────────────────────────────────────────
    if not m.PATTERN_PREFER_PROXIMITY:
        tags.append("no_proximity_sort")

    # ── Neckline level anchor ────────────────────────────────────────────────
    if not m.REQUIRE_NECKLINE_AT_LEVEL:
        tags.append("no_neckline_anchor")
    elif m.NECKLINE_LEVEL_TOLERANCE_PCT != 0.0020:
        tags.append(f"neckline_tol_{m.NECKLINE_LEVEL_TOLERANCE_PCT:.4f}")

    # ── Weekly candle agreement ──────────────────────────────────────────────
    if not m.REQUIRE_WEEKLY_CANDLE_AGREEMENT:
        tags.append("no_weekly_agreement")
    elif m.WEEKLY_AGREEMENT_MAX_OPPOSING_BODY_PCT != 0.0035:
        tags.append(f"weekly_agree_{m.WEEKLY_AGREEMENT_MAX_OPPOSING_BODY_PCT:.4f}")

    # ── News filter ─────────────────────────────────────────────────────────
    if m.NEWS_FILTER_ENABLED:
        tags.append("news_filter_on")

    # ── Entry signal ────────────────────────────────────────────────────────
    _mode_tag = {
        "engulf_only":                          "engulfing_only",
        "engulf_or_pin":                        "pin_bars_allowed",
        "engulf_or_star_at_level":              "star_at_level",
        "engulf_or_strict_pin_at_level":        "strict_pin_at_level",
        "engulf_or_star_or_strict_pin_at_level":"star_or_strict_pin",
    }.get(m.ENTRY_TRIGGER_MODE, m.ENTRY_TRIGGER_MODE)
    tags.append(_mode_tag)
    if m.ENTRY_TRIGGER_MODE == "engulf_or_pin":
        tags.append(f"pin_wick_{m.PIN_BAR_MIN_WICK_BODY_RATIO:.1f}x")

    # ── Zone touch gate ──────────────────────────────────────────────────────
    if not m.ZONE_REQUIRE_TOUCH:
        tags.append("no_zone_touch")
    else:
        tags.append(f"zone_{m.ZONE_TOUCH_ATR_MULT:.2f}atr")

    # ── DD circuit breaker ──────────────────────────────────────────────────
    if m.DD_CIRCUIT_BREAKER_ENABLED:
        tags.append(f"dd_brake_{int(m.DD_L1_PCT)}pct_cap{int(m.DD_L1_CAP)}")
    else:
        tags.append("no_dd_brake")

    # ── D1 veto ──────────────────────────────────────────────────────────────
    if m.D1_VETO_ONLY_STRONG_OPPOSITE:
        tags.append("d1_strong_veto")
    else:
        tags.append("weekly_veto")

    # ── Pattern types ───────────────────────────────────────────────────────
    if not m.ALLOW_BREAK_RETEST:
        tags.append("no_break_retest")

    # ── Level quality ───────────────────────────────────────────────────────
    if not m.LEVEL_ALLOW_FINE_INCREMENT:
        tags.append("no_fine_levels")
    if m.STRUCTURAL_LEVEL_MIN_SCORE != 3:
        tags.append(f"struct_score_{m.STRUCTURAL_LEVEL_MIN_SCORE}")

    # ── MTF context ─────────────────────────────────────────────────────────
    if not m.OVEREXTENSION_CHECK:
        tags.append("no_ext_check")
    elif m.OVEREXTENSION_THRESHOLD != 0.90:
        tags.append(f"ext_{int(m.OVEREXTENSION_THRESHOLD * 100)}pct")
    if not m.ALLOW_TIER3_REVERSALS:
        tags.append("no_tier3")

    # ── Theme gate ──────────────────────────────────────────────────────────
    if not m.REQUIRE_THEME_GATE:
        tags.append("no_theme_gate")

    # ── Winner / concurrency ────────────────────────────────────────────────
    tags.append("no_gate" if not m.BLOCK_ENTRY_WHILE_WINNER_RUNNING
                else f"winner_gate_{m.WINNER_THRESHOLD_R:.0f}R")

    # ── Signal / ATR ────────────────────────────────────────────────────────
    tags.append(f"atr_min_{m.ATR_MIN_MULTIPLIER}")
    tags.append(f"atr_max_{int(m.ATR_STOP_MULTIPLIER)}")
    tags.append(f"conf_{int(m.MIN_CONFIDENCE * 100)}")
    tags.append(f"rr_{m.MIN_RR:.1f}")
    tags.append(f"max_concurrent_{m.MAX_CONCURRENT_TRADES_BACKTEST}")

    # ── Alex small-account gate flags ────────────────────────────────────────
    # These must be recorded per-run so compare harnesses are reproducible.
    # Convention: "on" suffix = gate active; absence = gate off.
    if getattr(m, "NO_SUNDAY_TRADES_ENABLED",    False): tags.append("no_sun")
    if getattr(m, "NO_THU_FRI_TRADES_ENABLED",   False): tags.append("no_thu_fri")
    if getattr(m, "REQUIRE_HTF_TREND_ALIGNMENT",  False): tags.append("htf_gate")
    _wk_small = getattr(m, "MAX_TRADES_PER_WEEK_SMALL", 999)
    if _wk_small < 999:
        _wk_std = getattr(m, "MAX_TRADES_PER_WEEK_STANDARD", 999)
        tags.append(f"wk_{_wk_small}s_{_wk_std}n")   # e.g. wk_1s_2n = 1/wk small, 2/wk normal
    _rr_ct  = getattr(m, "MIN_RR_COUNTERTREND", 0.0)
    _rr_std = getattr(m, "MIN_RR_STANDARD",     0.0)
    if _rr_ct and _rr_ct != _rr_std:
        tags.append(f"rr_ct_{_rr_ct:.1f}_std_{_rr_std:.1f}")
    if getattr(m, "INDECISION_FILTER_ENABLED", False): tags.append("doji_gate")

    # ── Pairs hash (injected by caller) ─────────────────────────────────────
    if pairs_hash:
        tags.append(f"pairs_{pairs_hash}")

    # ── Defensive: ensure all tags are plain strings (no dict / object leak) ─
    return [str(t) for t in tags]
