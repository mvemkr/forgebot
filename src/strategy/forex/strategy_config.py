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
