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
"""

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

# ── Progressive confluence gate ────────────────────────────────────────────
# The bar to enter a second concurrent trade is higher than the first.
# Rationale: you're already deployed — the second trade has to EARN its way in.
# Alex was very selective about adding positions while already in a trade.
#
# Trade 1: standard MIN_CONFIDENCE (65%) — any qualifying setup
# Trade 2: SECOND_TRADE_MIN_CONFIDENCE (75%) + structural pattern only
#          Break-retests are mid-trend entries that don't carry enough
#          reversal evidence to justify opening alongside an existing position.
#          H&S, double top/bottom, IH&S required — clear pattern structure.
#
# Future: Trade 3+ (stacking) — 85% + confirmed macro theme score ≥ 6.0.
#         Not implemented yet — requires theme carve-out in currency overlap check.
SECOND_TRADE_MIN_CONFIDENCE: float = 0.75
SECOND_TRADE_STRUCTURAL_ONLY: bool = True   # break-retests blocked as 2nd entry


def progressive_confluence_check(
    n_open: int,
    confidence: float,
    pattern_type: str,
) -> tuple:
    """
    Progressive gate for concurrent positions.
    Returns (blocked: bool, reason: str).

    Called by both the backtester and the live orchestrator — same logic,
    same thresholds. Constants defined above; change them once, both inherit.

    n_open       : number of currently open positions BEFORE this entry
    confidence   : signal confidence from strategy.evaluate()
    pattern_type : pattern type string from PatternResult
    """
    if n_open == 0:
        return False, ""   # first trade — standard MIN_CONFIDENCE gate handles it

    if n_open >= 1:
        # ── Second trade: higher confidence required ───────────────────
        if confidence < SECOND_TRADE_MIN_CONFIDENCE:
            return True, (
                f"progressive_confluence: 2nd trade requires "
                f"{SECOND_TRADE_MIN_CONFIDENCE:.0%} confidence "
                f"(got {confidence:.0%}). Bar rises when already in a position."
            )
        # ── Second trade: structural pattern required ──────────────────
        if SECOND_TRADE_STRUCTURAL_ONLY:
            _structural = (
                "head_and_shoulders", "double_top",
                "double_bottom", "inverted_head_and_shoulders",
            )
            if not any(k in pattern_type for k in _structural):
                return True, (
                    f"progressive_confluence: 2nd trade requires structural pattern "
                    f"(H&S / DT / DB / IH&S). Got '{pattern_type}' — "
                    f"break-retest lacks reversal evidence needed alongside open position."
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
