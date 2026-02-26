"""
compare_alex_rules.py
=====================
Before/after comparison of Alex's small-account hard constraints.

Arm A — Baseline (current production: engulf_only + doji filter, no new gates)
Arm B — + Time rules (NO_SUNDAY, NO_THU_FRI)
Arm C — + HTF alignment gate (COUNTERTREND_HTF; requires W+D+4H all agree)
Arm D — + Weekly punch-card (MAX_TRADES_PER_WEEK by equity tier)
Arm E — + Dynamic MIN_RR (3.0R when equity < $25K)
Arm F — ALL gates combined (B+C+D+E)

Windows:
  W1  Jul 15 – Oct 31 2024  (Alex benchmark)
  W2  Jan 1  – Jan 31 2026  (recent live)

Usage:
    cd ~/trading-bot
    PYTHONPATH=/home/forge/trading-bot venv/bin/python backtesting/compare_alex_rules.py
"""
from __future__ import annotations
import sys
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.strategy.forex.strategy_config as _cfg
from backtesting.oanda_backtest_v2 import run_backtest, TRAIL_ARMS

# ── Parity assertions ─────────────────────────────────────────────────────
assert _cfg.MAX_CONCURRENT_TRADES_LIVE == 1, "MAX_CONCURRENT_LIVE must be 1"
assert _cfg.MIN_RR == 2.5,                   "MIN_RR must be 2.5"
assert _cfg.ENTRY_TRIGGER_MODE == "engulf_only", "Must start as engulf_only"

STARTING_BAL  = 8_000.0
ARM_KEY       = "C"


@dataclass
class ArmConfig:
    label:                str
    time_rules:           bool = False   # NO_SUNDAY + NO_THU_FRI
    htf_alignment:        bool = False   # REQUIRE_HTF_TREND_ALIGNMENT
    weekly_limit:         bool = False   # MAX_TRADES_PER_WEEK active
    dynamic_min_rr:       bool = False   # MIN_RR 3.0 when < $25K


ARMS: dict[str, ArmConfig] = {
    "A_baseline": ArmConfig(
        label="A — baseline (engulf_only + doji filter)",
    ),
    "B_time":     ArmConfig(
        label="B — +time rules (no Sun, no Thu PM/Fri)",
        time_rules=True,
    ),
    "C_htf":      ArmConfig(
        label="C — +HTF alignment (W+D+4H must agree)",
        time_rules=True, htf_alignment=True,
    ),
    "D_weekly":   ArmConfig(
        label="D — +weekly punch-card (1/wk<$25K, 2/wk≥$25K)",
        time_rules=True, htf_alignment=True, weekly_limit=True,
    ),
    "E_rr":       ArmConfig(
        label="E — +dynamic MIN_RR (3.0R when <$25K)",
        time_rules=True, htf_alignment=True, weekly_limit=True, dynamic_min_rr=True,
    ),
}

WINDOWS = {
    "W1_Alex_Jul15-Oct31_2024": (
        datetime(2024, 7, 15, tzinfo=timezone.utc),
        datetime(2024, 10, 31, tzinfo=timezone.utc),
    ),
    "W2_Jan_2026": (
        datetime(2026, 1,  1, tzinfo=timezone.utc),
        datetime(2026, 1, 31, tzinfo=timezone.utc),
    ),
}


def _patch(arm: ArmConfig) -> None:
    _cfg.NO_SUNDAY_TRADES_ENABLED    = arm.time_rules
    _cfg.NO_THU_FRI_TRADES_ENABLED   = arm.time_rules
    _cfg.REQUIRE_HTF_TREND_ALIGNMENT = arm.htf_alignment
    # Weekly limit and dynamic MIN_RR: controlled by the strategy_config flags
    # but actually enforced in the backtester. We disable them by setting
    # very high caps when arm doesn't include them.
    if arm.weekly_limit:
        _cfg.MAX_TRADES_PER_WEEK_SMALL    = 1
        _cfg.MAX_TRADES_PER_WEEK_STANDARD = 2
    else:
        _cfg.MAX_TRADES_PER_WEEK_SMALL    = 999
        _cfg.MAX_TRADES_PER_WEEK_STANDARD = 999
    if arm.dynamic_min_rr:
        _cfg.MIN_RR_SMALL_ACCOUNT = 3.0
        _cfg.MIN_RR_STANDARD      = 2.5
    else:
        _cfg.MIN_RR_SMALL_ACCOUNT = 2.5   # same as standard → gate is a no-op
        _cfg.MIN_RR_STANDARD      = 2.5


def _restore() -> None:
    _cfg.NO_SUNDAY_TRADES_ENABLED    = True
    _cfg.NO_THU_FRI_TRADES_ENABLED   = True
    _cfg.REQUIRE_HTF_TREND_ALIGNMENT = True
    _cfg.MAX_TRADES_PER_WEEK_SMALL   = 1
    _cfg.MAX_TRADES_PER_WEEK_STANDARD = 2
    _cfg.MIN_RR_SMALL_ACCOUNT        = 3.0
    _cfg.MIN_RR_STANDARD             = 2.5


def _run(arm: ArmConfig, start: datetime, end: datetime) -> dict:
    _patch(arm)
    try:
        return run_backtest(
            start_dt     = start,
            end_dt       = end,
            starting_bal = STARTING_BAL,
            trail_cfg    = TRAIL_ARMS[ARM_KEY],
            quiet        = True,
        )
    finally:
        _restore()


def _fmt(r: dict, arm: ArmConfig) -> tuple:
    trades  = r.get("trades", [])
    n       = len(trades)
    ret     = r.get("ret_pct", 0.0)
    dd      = r.get("max_dd_pct", 0.0)
    wr      = r.get("win_rate", 0.0)
    avg_r   = sum(t.get("r", 0) for t in trades) / n if n else 0.0
    best_r  = max((t.get("r", 0) for t in trades), default=0.0)
    worst_r = min((t.get("r", 0) for t in trades), default=0.0)
    wins    = sum(1 for t in trades if t.get("r", 0) > 0.1)
    losses  = sum(1 for t in trades if t.get("r", 0) < -0.1)
    scratch = n - wins - losses
    wl = f"{wins}W/{losses}L/{scratch}S"
    blocks = (
        f"time={r.get('time_blocks',0)} "
        f"htf={r.get('countertrend_htf_blocks',0)} "
        f"wkly={r.get('weekly_limit_blocks',0)} "
        f"rr={r.get('min_rr_small_blocks',0)} "
        f"doji={r.get('indecision_doji_blocks',0)}"
    )
    return (arm.label, n, f"{ret:+.1f}%", f"{dd:.1f}%", f"{wr:.0%}",
            f"{avg_r:+.2f}R", f"{best_r:+.2f}R", f"{worst_r:+.2f}R", wl, blocks)


def _exit_summary(trades: list) -> str:
    from collections import Counter
    ctr: dict = {}
    for t in trades:
        r = t.get("reason", "?")
        ctr.setdefault(r, []).append(t.get("r", 0.0))
    parts = []
    for reason, rs in sorted(ctr.items(), key=lambda x: -len(x[1])):
        avg = sum(rs) / len(rs) if rs else 0
        s   = "+" if avg >= 0 else ""
        parts.append(f"{reason}×{len(rs)}({s}{avg:.1f}R)")
    return "  ".join(parts)


def main() -> None:
    HDR = ["Arm", "N", "Ret%", "MaxDD", "WR", "AvgR", "BestR", "WorstR", "W/L/S", "Blocks"]
    WID = [48,     4,   8,      7,       5,    8,       8,       8,        14,       55]

    print("\n" + "═" * 100)
    print("  ALEX SMALL-ACCOUNT RULES — GATE-BY-GATE IMPACT")
    print("═" * 100)
    print(f"  Trail: Arm {ARM_KEY} | MAX_CONCURRENT=1 | MIN_RR=2.5→3.0(dynamic) | START=${STARTING_BAL:,.0f}")
    print(f"  Each arm adds one gate cumulatively (B = A + time; C = B + HTF; etc.)")

    all_results: dict[str, dict[str, dict]] = {}

    for win_name, (start, end) in WINDOWS.items():
        print(f"\n{'─' * 100}")
        print(f"  Window: {win_name}  [{start.date()} → {end.date()}]")
        print(f"{'─' * 100}")
        hdr_str = "  ".join(f"{h:<{w}}" for h, w in zip(HDR, WID))
        print(f"  {hdr_str}")
        print(f"  {'─' * 95}")

        win_res: dict[str, dict] = {}
        for arm_name, arm in ARMS.items():
            r    = _run(arm, start, end)
            win_res[arm_name] = r
            row  = _fmt(r, arm)
            rstr = "  ".join(f"{str(c):<{w}}" for c, w in zip(row, WID))
            print(f"  {rstr}")
        all_results[win_name] = win_res

    # ── Detailed exit breakdown ───────────────────────────────────────────
    print(f"\n\n{'═' * 100}")
    print("  EXIT BREAKDOWN BY ARM")
    print("═" * 100)
    for win_name, win_res in all_results.items():
        print(f"\n  [{win_name}]")
        for arm_name, arm in ARMS.items():
            trades = win_res[arm_name].get("trades", [])
            if not trades:
                print(f"    {arm.label[:46]}  — no trades")
                continue
            print(f"    {arm.label}")
            print(f"      {_exit_summary(trades)}")

    # ── Δ table: gate-by-gate impact ─────────────────────────────────────
    print(f"\n\n{'═' * 100}")
    print("  GATE-BY-GATE DELTA (vs A_baseline)")
    print("═" * 100)
    arm_seq = list(ARMS.keys())
    for win_name, win_res in all_results.items():
        print(f"\n  [{win_name}]")
        base = win_res["A_baseline"]
        base_ret  = base.get("return_pct", 0)
        base_dd   = base.get("max_dd_pct", 0)
        base_n    = len(base.get("trades", []))
        base_wr   = base.get("win_rate", 0)
        base_avgr = sum(t.get("r",0) for t in base.get("trades",[])) / max(base_n,1)
        print(f"    {'Gate':<48}  N-Δ   RetΔ    DDΔ     WRΔ    AvgRΔ  Blocks")
        print(f"    {'─' * 95}")
        for arm_name in arm_seq[1:]:
            arm    = ARMS[arm_name]
            r      = win_res[arm_name]
            n      = len(r.get("trades", []))
            avgr   = sum(t.get("r",0) for t in r.get("trades",[])) / max(n,1)
            nd     = n - base_n
            retd   = r.get("ret_pct",0) - base_ret
            ddd    = r.get("max_dd",0) - base_dd
            wrd    = r.get("win_rate",0) - base_wr
            avgrd  = avgr - base_avgr
            total_blk = (r.get("time_blocks",0) + r.get("countertrend_htf_blocks",0) +
                         r.get("weekly_limit_blocks",0) + r.get("min_rr_small_blocks",0) +
                         r.get("indecision_doji_blocks",0))
            print(f"    {arm.label:<48}  "
                  f"{nd:+3}   {retd:+6.1f}%  {ddd:+6.1f}%  {wrd:+5.0%}  {avgrd:+.2f}R  {total_blk}")

    # ── Entry hour histogram ─────────────────────────────────────────────
    print(f"\n\n{'═' * 100}")
    print("  ENTRY HOUR HISTOGRAM (UTC) — Arm A baseline")
    print("═" * 100)
    from collections import Counter
    for win_name, win_res in all_results.items():
        trades = win_res["A_baseline"].get("trades", [])
        hours  = Counter()
        for t in trades:
            ts = t.get("entry_ts", "")
            if len(ts) >= 13:
                try:
                    h = int(ts[11:13])
                    hours[h] += 1
                except ValueError:
                    pass
        print(f"\n  [{win_name}]  ({len(trades)} trades)")
        for h in sorted(hours):
            bar = "█" * hours[h]
            session = ("London" if 7 <= h < 12 else
                       "NY"     if 12 <= h < 17 else
                       "Asian"  if 0 <= h < 7  else "Late")
            print(f"    {h:02d}:00 UTC  {bar:<12}  {hours[h]}  ({session})")

    print(f"\n{'═' * 100}")
    print("  VERDICT")
    print("═" * 100)
    for win_name, win_res in all_results.items():
        ra = win_res["A_baseline"]
        rf = win_res[list(ARMS.keys())[-1]]
        nA = len(ra.get("trades", []))
        nF = len(rf.get("trades", []))
        print(f"\n  {win_name}:")
        print(f"    Baseline → All-gates:  {nA} trades → {nF} trades  "
              f"return {ra.get('return_pct',0):+.1f}% → {rf.get('return_pct',0):+.1f}%  "
              f"DD {ra.get('max_dd_pct',0):.1f}% → {rf.get('max_dd_pct',0):.1f}%  "
              f"avgR {sum(t.get('r',0) for t in ra.get('trades',[])) / max(nA,1):+.2f} → "
              f"{sum(t.get('r',0) for t in rf.get('trades',[])) / max(nF,1):+.2f}")
        total_blk = (rf.get("time_blocks",0) + rf.get("countertrend_htf_blocks",0) +
                     rf.get("weekly_limit_blocks",0) + rf.get("min_rr_small_blocks",0))
        print(f"    Total entries blocked by new gates: {total_blk}")
    print()


if __name__ == "__main__":
    main()
