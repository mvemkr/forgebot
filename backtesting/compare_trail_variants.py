"""
compare_trail_variants.py
==========================
Step 3 — Trail C drawdown optimisation.

Runs Arm C (baseline) + C1, C2, C3 variants on W1 and W2.
Selects winner that satisfies ALL three criteria:
  ✓ AvgR ≥ 0.30R (W1) / ≥ 0.20R (W2)
  ✓ MaxDD < 25%
  ✓ BestR ≥ 3.0R (at least one 3R+ winner preserved)

All other strategy config unchanged from production baseline.
"""

from __future__ import annotations
import sys, os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.strategy.forex.strategy_config as _cfg
from backtesting.oanda_backtest_v2 import run_backtest, TRAIL_ARMS
from src.strategy.forex.backtest_schema import BacktestResult

# ── Parity assertions ──────────────────────────────────────────────────────
assert _cfg.MAX_CONCURRENT_TRADES_LIVE == 1
assert _cfg.MIN_RR == 2.5
assert _cfg.ENTRY_TRIGGER_MODE == "engulf_only"

STARTING_BAL = 8_000.0

WINDOWS = {
    "W1_Jul15-Oct31_2024": (
        datetime(2024, 7, 15, tzinfo=timezone.utc),
        datetime(2024, 10, 31, tzinfo=timezone.utc),
        0.30,   # min AvgR threshold for W1
    ),
    "W2_Oct1_2025-Feb28_2026": (
        datetime(2025, 10,  1, tzinfo=timezone.utc),
        datetime(2026,  2, 28, tzinfo=timezone.utc),
        0.20,   # min AvgR threshold for W2
    ),
}

# Arms to compare: C baseline + three variants
ARMS_TO_RUN = ["C", "C1", "C2", "C3"]

# Selection criteria
MAX_DD_TARGET   = 25.0   # % — hard ceiling
MIN_BEST_R      = 3.0    # at least one 3R+ winner must survive


def _wl_str(r: BacktestResult) -> str:
    wins = r.n_target + r.n_ratchet
    los  = r.n_sl
    stall = r.n_trades - wins - los
    return f"{wins}W/{los}L/{stall}S"


def run_window(
    wname: str, start: datetime, end: datetime, min_avg_r: float
) -> dict[str, BacktestResult]:
    results: dict[str, BacktestResult] = {}
    for arm_key in ARMS_TO_RUN:
        print(f"  [{wname}] Running arm {arm_key} …", end=" ", flush=True)
        r = run_backtest(
            start_dt=start, end_dt=end, starting_bal=STARTING_BAL,
            notes=f"trail_{arm_key}",
            trail_arm_key=arm_key,
            use_cache=True, quiet=True,
        )
        results[arm_key] = r
        print(f"N={r.n_trades} AvgR={r.avg_r:+.2f}R DD={r.max_dd_pct:.1f}% ✓" if _meets(r, min_avg_r) else
              f"N={r.n_trades} AvgR={r.avg_r:+.2f}R DD={r.max_dd_pct:.1f}% ✗")
    return results


def _meets(r: BacktestResult, min_avg_r: float) -> bool:
    return (r.avg_r   >= min_avg_r
            and r.max_dd_pct < MAX_DD_TARGET
            and r.best_r >= MIN_BEST_R)


def print_table(wname: str, results: dict[str, BacktestResult], min_avg_r: float) -> None:
    COL = 90
    print(f"\n{'─'*COL}")
    print(f"  {wname}")
    print(f"{'─'*COL}")
    hdr = (f"  {'Arm':<8} {'N':<5} {'W/L/S':<12} {'AvgR':<9} {'BestR':<9} "
           f"{'WorstR':<9} {'MaxDD':<8} {'Ret%':<9} {'Crit'}")
    print(hdr)
    print(f"  {'-'*85}")

    for arm_key in ARMS_TO_RUN:
        r   = results[arm_key]
        ok  = _meets(r, min_avg_r)
        tag = "✅" if ok else "  "
        stall_n = r.n_trades - r.n_target - r.n_ratchet - r.n_sl
        wls = f"{r.n_target+r.n_ratchet}W/{r.n_sl}L/{max(0,stall_n)}S"
        arm_label = TRAIL_ARMS[arm_key]["label"].split("—")[0].strip()
        print(f"  {arm_label:<8} {r.n_trades:<5} {wls:<12} "
              f"{r.avg_r:+.2f}R    {r.best_r:+.2f}R    {r.worst_r:+.2f}R    "
              f"{r.max_dd_pct:.1f}%    {r.return_pct:+.1f}%   {tag}")

    # Detail: stall exit counts (C3 only)
    c3 = results.get("C3")
    if c3 and c3.trades:
        stall_count = sum(1 for t in c3.trades if t.get("reason") == "stall_exit")
        avg_r_stall = (sum(t["r"] for t in c3.trades if t.get("reason") == "stall_exit")
                       / max(1, stall_count))
        print(f"\n  C3 stall exits: {stall_count}  avg_r={avg_r_stall:+.2f}R")

    # Criteria reminder
    print(f"\n  Criteria: AvgR≥{min_avg_r}R | MaxDD<{MAX_DD_TARGET}% | BestR≥{MIN_BEST_R}R")

    # Pick winner
    passing = [k for k in ARMS_TO_RUN if _meets(results[k], min_avg_r)]
    if passing:
        # Prefer highest AvgR among passing arms; tie-break by lowest DD
        best_arm = max(passing, key=lambda k: (results[k].avg_r, -results[k].max_dd_pct))
        print(f"  → Winner: {best_arm}  ({TRAIL_ARMS[best_arm]['label']})")
    else:
        # Find closest arm (fewest failing criteria)
        def _fail_count(k: str) -> tuple:
            r = results[k]
            fails = (
                int(r.avg_r < min_avg_r),
                int(r.max_dd_pct >= MAX_DD_TARGET),
                int(r.best_r < MIN_BEST_R),
            )
            return (sum(fails), -r.avg_r, r.max_dd_pct)
        closest = min(ARMS_TO_RUN, key=_fail_count)
        r_c = results[closest]
        fails = []
        if r_c.avg_r < min_avg_r:        fails.append(f"AvgR={r_c.avg_r:+.2f}R<{min_avg_r}")
        if r_c.max_dd_pct >= MAX_DD_TARGET: fails.append(f"DD={r_c.max_dd_pct:.1f}%≥{MAX_DD_TARGET}%")
        if r_c.best_r < MIN_BEST_R:      fails.append(f"BestR={r_c.best_r:.2f}R<{MIN_BEST_R}")
        print(f"  ⚠  No arm meets all criteria. Closest: {closest} (fails: {', '.join(fails)})")


def main() -> None:
    print("=" * 90)
    print("  TRAIL VARIANTS — Step 3 of 6K% path")
    print("  C vs C1 (earlier lock) vs C2 (tighter trail) vs C3 (+ stall exit)")
    print("  Target: AvgR≥0.30R W1 / AvgR≥0.20R W2 | MaxDD<25% | BestR≥3R")
    print("=" * 90)

    all_results: dict[str, dict[str, BacktestResult]] = {}

    for wname, (ws, we, min_avg_r) in WINDOWS.items():
        print(f"\n  Window: {wname}")
        all_results[wname] = run_window(wname, ws, we, min_avg_r)

    for wname, (_, _, min_avg_r) in WINDOWS.items():
        print_table(wname, all_results[wname], min_avg_r)

    # Cross-window: winner must pass BOTH windows
    print(f"\n{'═'*90}")
    print("  CROSS-WINDOW CONSENSUS")
    print(f"{'═'*90}")
    w_thresholds = list(WINDOWS.values())
    passing_both = [
        k for k in ARMS_TO_RUN
        if all(
            _meets(all_results[wn][k], wt)
            for wn, (*_, wt) in WINDOWS.items()
        )
    ]
    if passing_both:
        champion = max(
            passing_both,
            key=lambda k: sum(all_results[wn][k].avg_r for wn in WINDOWS)
        )
        print(f"  Arms passing BOTH windows: {passing_both}")
        print(f"  Champion: {champion}  ({TRAIL_ARMS[champion]['label']})")
        print(f"  → Port this arm to live position_monitor.py as the new default.")
    else:
        print("  No arm passes both windows under current criteria.")
        # Show per-window winners for guidance
        for wname, (*_, min_avg_r) in WINDOWS.items():
            passing = [k for k in ARMS_TO_RUN if _meets(all_results[wname][k], min_avg_r)]
            if passing:
                best = max(passing, key=lambda k: (all_results[wname][k].avg_r,
                                                    -all_results[wname][k].max_dd_pct))
                print(f"  {wname} winner: {best}")
            else:
                print(f"  {wname}: no arm meets all criteria")


if __name__ == "__main__":
    main()
