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
MIN_CONFIDENCE: float = 0.65

# Minimum R:R ratio based on pattern amplitude vs stop distance.
# Not a take-profit target — geometric quality check only.
# Blocks patterns where the measured move is smaller than the stop.
MIN_RR: float = 1.0

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
# Maximum simultaneous open positions (non-theme trades).
# Macro theme stacking bypasses this — 4 JPY shorts = 1 theme = allowed.
MAX_CONCURRENT_TRADES: int = 2

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

# ── Entry signal quality ────────────────────────────────────────────────────
# Alex's rule: "No engulfing candle = no trade." Every video. Absolute rule.
# When True: only bearish_engulfing / bullish_engulfing trigger entry.
# Pin bars, rejection wicks, etc. are NOT valid entry signals.
ENGULFING_ONLY: bool = True

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
def get_model_tags() -> list:
    """
    Returns a list of short descriptive tags capturing ALL active levers.
    Written into every backtest result record so you can always reproduce
    exactly what config produced a given number.

    Tags are stable short strings — sort + join them to get a run fingerprint.
    """
    m = _sys.modules[__name__]
    tags = []

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
    tags.append("engulfing_only" if m.ENGULFING_ONLY else "pin_bars_allowed")

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
    tags.append(f"max_concurrent_{m.MAX_CONCURRENT_TRADES}")

    return tags
