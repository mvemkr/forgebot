"""
diagnose_rr_blocks.py
=====================
Step 2 — exec_rr_min diagnostic.

For every exec_rr_min blocked setup, prints:
  pair | ts | dir | stop_type | stop_pips | rr_4h | rr_mm | rr_mm_t2 | best | gap_to_qualify

Reads from logs/backtest_v2_decisions.json (populated by run_backtest).
Two windows: W1 Jul15–Oct31 2024, W2 Oct1 2025–Feb28 2026.
"""

from __future__ import annotations
import sys, os, re, json
from datetime import datetime, timezone
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.strategy.forex.strategy_config as _cfg
from backtesting.oanda_backtest_v2 import run_backtest

WINDOWS = {
    "W1_Jul15-Oct31_2024": (
        datetime(2024, 7, 15, tzinfo=timezone.utc),
        datetime(2024, 10, 31, tzinfo=timezone.utc),
    ),
    "W2_Oct1_2025-Feb28_2026": (
        datetime(2025, 10,  1, tzinfo=timezone.utc),
        datetime(2026,  2, 28, tzinfo=timezone.utc),
    ),
}

DECISION_LOG = os.path.join(
    os.path.dirname(__file__), "../logs/backtest_v2_decisions.json"
)

MIN_RR = _cfg.MIN_RR   # 2.5


def _parse_rr_candidates(reason: str) -> dict[str, float]:
    """Extract {candidate_type: rr_value} from decision reason string."""
    out: dict[str, float] = {}
    for m in re.finditer(r"([\w_]+)=([\d.]+)R\s*\((\w+)\)", reason):
        out[m.group(1)] = float(m.group(2))
    return out


def analyse_window(name: str, start: datetime, end: datetime) -> None:
    print(f"\n{'═'*100}")
    print(f"  {name}  [{start.date()} → {end.date()}]")
    print(f"{'═'*100}")

    # Run backtest to populate decision log
    print("  Running backtest …")
    run_backtest(
        start_dt=start, end_dt=end, starting_bal=8_000.0,
        notes="rr_diag", trail_arm_key="C",
        use_cache=True, quiet=True,
    )

    # Load decision log
    if not os.path.exists(DECISION_LOG):
        print("  ERROR: decision log not found at", DECISION_LOG)
        return

    with open(DECISION_LOG) as f:
        log_data = json.load(f)

    decisions = log_data.get("decisions", [])
    ws, we = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    # Filter: WAIT decisions with exec_rr_min, in this window
    rr_blocks = [
        d for d in decisions
        if d.get("decision") == "WAIT"
        and "exec_rr_min" in (d.get("failed_filters") or [])
        and ws <= d.get("ts", "")[:10] <= we
    ]

    # Deduplicate: same pair+ts can appear multiple times from re-evaluation
    seen: set = set()
    unique_blocks: list[dict] = []
    for d in rr_blocks:
        key = (d["ts"][:16], d["pair"])
        if key not in seen:
            seen.add(key)
            unique_blocks.append(d)

    if not unique_blocks:
        print(f"  No exec_rr_min blocks found for this window.")
        return

    print(f"  exec_rr_min blocks: {len(unique_blocks)}")
    print()
    HDR = f"  {'Pair':<10} {'Dir':<6} {'StopType':<28} {'Sp':<5} {'4h(R)':<7} {'mm(R)':<7} {'mm×1.5(R)':<11} {'Best(R)':<9} {'Gap':<7} {'Ts'}"
    print(HDR)
    print(f"  {'-'*100}")

    best_rrs: list[float]  = []
    gap_rrs:  list[float]  = []
    stop_pips_all: list[float] = []
    by_pair:    Counter = Counter()
    by_dir:     Counter = Counter()
    gap_buckets: Counter = Counter()

    for d in sorted(unique_blocks, key=lambda x: x.get("ts", "")):
        pair      = d.get("pair", "?")
        direction = d.get("direction", "?")
        stop_type = d.get("stop_type", "?")
        stop_p    = float(d.get("initial_stop_pips", 0) or 0)
        reason    = d.get("reason", "")
        ts        = d.get("ts", "")[:16]

        cands  = _parse_rr_candidates(reason)
        rr_4h  = cands.get("4h_structure",     0.0)
        rr_mm  = cands.get("measured_move",    0.0)
        rr_mm2 = cands.get("measured_move_t2", 0.0)
        best   = max(rr_4h, rr_mm, rr_mm2, 0.001)
        gap    = MIN_RR - best

        best_rrs.append(best)
        gap_rrs.append(gap)
        if stop_p > 0:
            stop_pips_all.append(stop_p)
        by_pair[pair] += 1
        by_dir[direction] += 1

        if gap < 0.5:
            gap_buckets["<0.5R (close miss)"] += 1
        elif gap < 1.0:
            gap_buckets["0.5–1R"] += 1
        elif gap < 1.5:
            gap_buckets["1–1.5R"] += 1
        else:
            gap_buckets["≥1.5R (far)"] += 1

        def fmt(v: float) -> str:
            return f"{v:.2f}" if v > 0.01 else "  —  "

        print(f"  {pair:<10} {direction:<6} {stop_type:<28} {stop_p:<5.0f} "
              f"{fmt(rr_4h):<7} {fmt(rr_mm):<7} {fmt(rr_mm2):<11} "
              f"{best:<9.2f} {gap:<7.2f} {ts}")

    # Summary
    n = len(unique_blocks)
    p50_best = sorted(best_rrs)[n // 2]
    p50_gap  = sorted(gap_rrs)[n // 2]
    p50_sp   = sorted(stop_pips_all)[len(stop_pips_all)//2] if stop_pips_all else 0

    print(f"\n  ── Summary ({'─'*55})")
    print(f"  Total exec_rr_min blocks: {n}")
    print(f"  best-candidate p50:       {p50_best:.2f}R   (worst has best={min(best_rrs):.2f}R)")
    print(f"  gap-to-2.5R p50:          {p50_gap:.2f}R   (closest gap={min(gap_rrs):.2f}R)")
    print(f"  stop_pips p50:            {p50_sp:.0f}p")
    print(f"  By gap bucket:  {dict(gap_buckets)}")
    print(f"  By pair:        {dict(by_pair.most_common())}")
    print(f"  By direction:   {dict(by_dir)}")

    # What-if: how many would unblock under each scenario
    def _count_pass(fn) -> int:
        return sum(1 for d in unique_blocks if fn(_parse_rr_candidates(d.get("reason", ""))))

    t2_pass      = _count_pass(lambda c: c.get("measured_move_t2",    0) >= MIN_RR)
    mm_pass      = _count_pass(lambda c: c.get("measured_move",       0) >= MIN_RR)
    fs_pass      = _count_pass(lambda c: c.get("4h_structure",        0) >= MIN_RR)
    any_2_0      = _count_pass(lambda c: max(c.values(), default=0) >= 2.0)
    any_2_2      = _count_pass(lambda c: max(c.values(), default=0) >= 2.2)
    never_2_0    = _count_pass(lambda c: max(c.values(), default=0) < 2.0)
    never_2_2    = _count_pass(lambda c: max(c.values(), default=0) < 2.2)

    print(f"\n  ── What-if (unblocked count / {n}) ──")
    print(f"  Swap candidate order → t2 primary:        {t2_pass:>3}  ({100*t2_pass//n}%)  — mm×1.5 ≥ 2.5R")
    print(f"  t1 alone (original mm):                   {mm_pass:>3}  ({100*mm_pass//n}%)")
    print(f"  4h_structure only:                        {fs_pass:>3}  ({100*fs_pass//n}%)")
    print(f"  Lower MIN_RR → 2.2R:                      {any_2_2:>3}  ({100*any_2_2//n}%)")
    print(f"  Lower MIN_RR → 2.0R:                      {any_2_0:>3}  ({100*any_2_0//n}%)")
    print(f"  Hard blocks (best < 2.0R, no fix helps):  {never_2_0:>3}  ({100*never_2_0//n}%)")

    print(f"\n  ── Conclusion ──")
    if t2_pass >= n * 0.4:
        print(f"  ✅ HYPOTHESIS CONFIRMED: measured_move is too conservative.")
        print(f"     {t2_pass}/{n} blocks ({100*t2_pass//n}%) would pass if mm×1.5 used as primary.")
        print(f"     → Recommended fix: swap [4h_structure, measured_move_t2, measured_move]")
        print(f"       or use neckline-to-head height × 1.0 as measured_move baseline.")
    elif never_2_0 >= n * 0.5:
        print(f"  ⚠  STRUCTURE TOO TIGHT: {never_2_0}/{n} blocks have best < 2.0R.")
        print(f"     Pattern setups genuinely lack room — not a calculation issue.")
        print(f"     → Consider lowering MIN_RR or widening stop to improve ratio.")
    else:
        print(f"  Mixed: {t2_pass} from conservative mm, {never_2_0} from genuinely tight structure.")
        print(f"  → Partial fix: swap t2 to primary (gains {t2_pass}), accept {never_2_0} remain blocked.")


if __name__ == "__main__":
    for wname, (ws, we) in WINDOWS.items():
        analyse_window(wname, ws, we)
    print("\nDone.")
