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
