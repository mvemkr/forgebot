"""
risk_grid.py
============
Prod Config Lock (Arm D) — Risk Scaling Grid

Runs W2 (Oct 1 2025 – Feb 28 2026) at 5 flat risk percentages.
Strategy config is fully locked:
  • Entry trigger:  engulf_only
  • MIN_RR:         2.5R flat (no protrend penalty — Arm D)
  • Time rules:     ON (no Sun, no Thu after 09:00 ET, no Fri)
  • HTF block:      ON (block only when W+D+4H all oppose)
  • Weekly cap:     ON (1/wk <$25K, 2/wk ≥$25K)
  • Trail:          Arm C
  • Concurrency:    1
  • Starting bal:   $8,000

No gate changes. No trail changes. No pattern changes.
Calibrating leverage only.
"""

from __future__ import annotations
import sys, os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.strategy.forex.strategy_config as _cfg
from backtesting.oanda_backtest_v2 import run_backtest
from src.strategy.forex.backtest_schema import BacktestResult

# ── Parity assertions (Arm D lock) ────────────────────────────────────────────
assert _cfg.MAX_CONCURRENT_TRADES_LIVE == 1,     "concurrency must be 1"
assert _cfg.MIN_RR == 2.5,                        "MIN_RR must be 2.5"
assert _cfg.ENTRY_TRIGGER_MODE == "engulf_only",  "trigger must be engulf_only"
assert _cfg.MIN_RR_COUNTERTREND == _cfg.MIN_RR,  "Arm D: no protrend penalty (must be 2.5)"
assert _cfg.NO_SUNDAY_TRADES_ENABLED,             "no_sunday must be ON"
assert _cfg.NO_THU_FRI_TRADES_ENABLED,            "no_thu_fri must be ON"
assert _cfg.REQUIRE_HTF_TREND_ALIGNMENT,          "HTF gate must be ON"
assert _cfg.MAX_TRADES_PER_WEEK_SMALL    == 1,   "weekly cap small must be 1"
assert _cfg.MAX_TRADES_PER_WEEK_STANDARD == 2,   "weekly cap standard must be 2"

STARTING_BAL = 8_000.0
WINDOW_START = datetime(2025, 10,  1, tzinfo=timezone.utc)
WINDOW_END   = datetime(2026,  2, 28, tzinfo=timezone.utc)

RISK_LEVELS = [6.0, 8.0, 10.0, 12.0, 15.0]


def _worst_3loss_cluster(trades: list) -> float:
    """
    Worst consecutive 3-loss cluster drawdown in %.
    Walks trades in order, finds the worst run of 3 consecutive losses.
    Returns the combined DD% as |sum(r) × avg_risk| approximated from
    actual pnl vs starting_bal (since R captures all compounding).
    """
    if not trades or len(trades) < 3:
        return 0.0
    # Use actual dollar pnl to compute equity path DD
    running = STARTING_BAL
    peak    = running
    worst   = 0.0
    window: list[float] = []  # rolling 3-trade pnl
    for t in trades:
        pnl = t.get("pnl", 0.0)
        running += pnl
        window.append(pnl)
        if len(window) > 3:
            window.pop(0)
        if len(window) == 3 and all(p < 0 for p in window):
            cluster_dd = sum(window)
            # express as % of balance before the 3-loss run
            bal_before = running - cluster_dd
            pct = abs(cluster_dd) / bal_before * 100 if bal_before > 0 else 0.0
            worst = max(worst, pct)
        peak    = max(peak, running)
    return worst


def _peak_to_trough(trades: list) -> float:
    """Equity curve peak-to-trough in %."""
    if not trades:
        return 0.0
    running = STARTING_BAL
    peak    = running
    worst   = 0.0
    for t in trades:
        running += t.get("pnl", 0.0)
        if running < peak:
            dd = (peak - running) / peak * 100
            worst = max(worst, dd)
        else:
            peak = running
    return worst


def run_grid() -> None:
    print("=" * 90)
    print("  RISK SCALING GRID — Arm D Production Config")
    print(f"  Window: {WINDOW_START.date()} → {WINDOW_END.date()}")
    print(f"  Starting balance: ${STARTING_BAL:,.0f}")
    print(f"  MIN_RR={_cfg.MIN_RR}R flat | Trail C | Weekly cap ON | HTF ON | Time rules ON")
    print("=" * 90)
    print()

    results: list[tuple[float, BacktestResult]] = []
    candle_data: dict = {}  # populated from first run, reused for the rest

    for rpct in RISK_LEVELS:
        print(f"  Running risk={rpct:.0f}% …", end=" ", flush=True)
        r = run_backtest(
            start_dt=WINDOW_START,
            end_dt=WINDOW_END,
            starting_bal=STARTING_BAL,
            notes=f"risk_grid_{rpct:.0f}pct",
            trail_arm_key="C",
            preloaded_candle_data=candle_data if candle_data else None,
            use_cache=True,
            quiet=True,
            flat_risk_pct=rpct,
        )
        if not candle_data and r.candle_data:
            candle_data = r.candle_data   # reuse from first run
        results.append((rpct, r))
        print(f"N={r.n_trades} AvgR={r.avg_r:+.2f}R DD={r.max_dd_pct:.1f}% Ret={r.return_pct:+.1f}%")

    # Print table
    print()
    print(f"{'─'*90}")
    print(f"  {'Risk%':<8} {'Trades':<8} {'AvgR':<9} {'BestR':<9} {'WorstR':<9} "
          f"{'MaxDD':<9} {'Return%':<10} {'Worst3L':<10} {'PkTrough'}")
    print(f"  {'─'*85}")

    for rpct, r in results:
        w3_dd  = _worst_3loss_cluster(r.trades or [])
        pt_dd  = _peak_to_trough(r.trades or [])
        wins   = r.n_target + r.n_ratchet
        stall  = r.n_trades - wins - r.n_sl
        print(f"  {rpct:<8.0f} {r.n_trades:<8} {r.avg_r:+.2f}R    {r.best_r:+.2f}R    "
              f"{r.worst_r:+.2f}R    {r.max_dd_pct:<9.1f} {r.return_pct:+.1f}%     "
              f"{w3_dd:<10.1f} {pt_dd:.1f}%")

    print()
    # Summary conclusions
    under_25 = [(rpct, r) for rpct, r in results if r.max_dd_pct <= 25.0]
    under_35 = [(rpct, r) for rpct, r in results if r.max_dd_pct <= 35.0]

    print(f"  ── Conclusions ──")
    if under_25:
        best_25 = max(under_25, key=lambda x: x[1].return_pct)
        print(f"  MaxDD ≤ 25%:  up to {max(r for r,_ in under_25):.0f}% risk  "
              f"(best return at {best_25[0]:.0f}%: {best_25[1].return_pct:+.1f}%)")
    else:
        print(f"  MaxDD ≤ 25%:  NO risk level achieves this on W2")

    if under_35:
        best_35 = max(under_35, key=lambda x: x[1].return_pct)
        print(f"  MaxDD ≤ 35%:  up to {max(r for r,_ in under_35):.0f}% risk  "
              f"(best return at {best_35[0]:.0f}%: {best_35[1].return_pct:+.1f}%)")
    else:
        print(f"  MaxDD ≤ 35%:  NO risk level achieves this on W2")

    # Convexity: highest AvgR × BestR product
    best_conv = max(results, key=lambda x: x[1].avg_r * x[1].best_r)
    print(f"  Strongest convexity: {best_conv[0]:.0f}% risk  "
          f"(AvgR={best_conv[1].avg_r:+.2f}R × BestR={best_conv[1].best_r:+.2f}R"
          f" = {best_conv[1].avg_r * best_conv[1].best_r:.2f})")

    # Also show W1 reference at 10% for context
    print()
    print(f"  Reference: Arm D W1 at 10% risk → AvgR=+0.97R, MaxDD≈18%, Ret≈+237% (N=20)")
    print(f"  W2 is a harder window (mixed macro, 44% WR) — these DDs reflect that.")


if __name__ == "__main__":
    run_grid()
