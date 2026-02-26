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
from src.strategy.forex.backtest_schema import BacktestResult

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
    # W2: extended to Dec–Feb to get 20+ trades (Jan alone is 8 — too small).
    # Cache covers through Feb 25 2026; keep end at Feb 25 to avoid OANDA refetch.
    "W2_Dec2025-Feb2026": (
        datetime(2025, 12,  1, tzinfo=timezone.utc),
        datetime(2026,  2, 25, tzinfo=timezone.utc),
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


def _run(arm: ArmConfig, start: datetime, end: datetime) -> BacktestResult:
    _patch(arm)
    try:
        return run_backtest(
            start_dt      = start,
            end_dt        = end,
            starting_bal  = STARTING_BAL,
            trail_cfg     = TRAIL_ARMS[ARM_KEY],
            trail_arm_key = ARM_KEY,
            quiet         = True,
        )
    finally:
        _restore()


def _arm_config_str(arm: ArmConfig) -> str:
    """One-line config summary for a table row."""
    parts = [f"trail={ARM_KEY}", f"min_rr=2.5", f"pairs=alex7"]
    parts.append("sun=off" if arm.time_rules else "sun=ON")
    parts.append("thu_fri=off" if arm.time_rules else "thu_fri=ON")
    parts.append("htf=off" if arm.htf_alignment else "htf=ON")
    parts.append(f"wk={'1s/2n' if arm.weekly_limit else 'off'}")
    parts.append(f"dyn_rr={'3.0s' if arm.dynamic_min_rr else 'off'}")
    parts.append("doji=on")   # always on (production default)
    return "  ".join(parts)


def _fmt(r: BacktestResult, arm: ArmConfig) -> tuple:
    n       = r.n_trades
    avg_r   = r.avg_r
    best_r  = r.best_r
    worst_r = r.worst_r
    dd      = r.max_dd_pct
    ret     = r.return_pct
    wr      = r.win_rate
    trades  = r.trades
    wins    = sum(1 for t in trades if t.get("r", 0) > 0.1)
    losses  = sum(1 for t in trades if t.get("r", 0) < -0.1)
    scratch = n - wins - losses
    wl = f"{wins}W/{losses}L/{scratch}S"
    # Gate hit counts — per-gate columns so delta is instantly readable
    return (
        arm.label,
        n,
        wl,
        f"{avg_r:+.2f}R",
        f"{best_r:+.2f}R",
        f"{worst_r:+.2f}R",
        f"{dd:.1f}%",
        f"{ret:+.1f}%",      # return% secondary — small N makes it noisy
        f"{wr:.0%}",
        # gate hit counts as individual cells
        str(r.time_blocks),
        str(r.countertrend_htf_blocks),
        str(r.weekly_limit_blocks),
        str(r.min_rr_small_blocks),
        str(r.indecision_doji_blocks),
    )


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
    # ── Column layout — performance FIRST, gate counts last ──────────────
    # Primary headline: N trades / W/L/S / AvgR / BestR / WorstR / MaxDD
    # Return% is secondary: small N makes it statistically noisy.
    # Gate counts: individual columns (not a compound string) so Δ is instant.
    HDR = ["Arm", "N", "W/L/S", "AvgR", "BestR", "WrstR", "MaxDD", "Ret%", "WR",
           "sun", "htf", "wkly", "rr", "doji"]
    WID = [48,     4,   14,      8,      8,        8,       7,       8,     5,
           4,     4,    5,      4,     5]

    print("\n" + "═" * 110)
    print("  ALEX SMALL-ACCOUNT RULES — GATE-BY-GATE IMPACT")
    print("═" * 110)
    print(f"  Trail Arm {ARM_KEY} | MAX_CONCURRENT=1 | START=${STARTING_BAL:,.0f} | pairs=Alex 7")
    print(f"  Arms are CUMULATIVE: B = A + time gates; C = B + HTF; D = C + weekly; E = D + dyn-RR")
    print(f"  Headline: N / W/L/S / AvgR / BestR / WorstR / MaxDD")
    print(f"  Return% is secondary — N<20 makes it statistically noisy (1 outlier = ±20%)")
    print(f"  Gate cols: blocked entries by gate type (not trades — one entry can be blocked N times)")

    all_results: dict[str, dict[str, BacktestResult]] = {}

    for win_name, (start, end) in WINDOWS.items():
        days = (end - start).days
        print(f"\n{'─' * 110}")
        print(f"  Window: {win_name}  [{start.date()} → {end.date()}]  ({days}d)")
        print(f"{'─' * 110}")

        # Config header — one line per arm showing active gates
        print(f"  {'Config':<48}  {'Gates active'}")
        print(f"  {'─' * 107}")
        for arm_name, arm in ARMS.items():
            print(f"  {arm.label:<48}  {_arm_config_str(arm)}")
        print()

        hdr_str = "  ".join(f"{h:<{w}}" for h, w in zip(HDR, WID))
        print(f"  {hdr_str}")
        print(f"  {'─' * 107}")

        win_res: dict[str, BacktestResult] = {}
        for arm_name, arm in ARMS.items():
            r    = _run(arm, start, end)
            win_res[arm_name] = r
            row  = _fmt(r, arm)
            rstr = "  ".join(f"{str(c):<{w}}" for c, w in zip(row, WID))
            print(f"  {rstr}")
        all_results[win_name] = win_res

    # ── Detailed exit breakdown ───────────────────────────────────────────
    print(f"\n\n{'═' * 110}")
    print("  EXIT BREAKDOWN BY ARM")
    print("═" * 100)
    for win_name, win_res in all_results.items():
        print(f"\n  [{win_name}]")
        for arm_name, arm in ARMS.items():
            r: BacktestResult = win_res[arm_name]
            if not r.trades:
                print(f"    {arm.label[:46]}  — no trades")
                continue
            print(f"    {arm.label}")
            print(f"      {_exit_summary(r.trades)}")

    # ── Δ table: gate-by-gate impact ─────────────────────────────────────
    print(f"\n\n{'═' * 110}")
    print("  GATE-BY-GATE DELTA (vs A_baseline)  — primary: AvgRΔ / N-Δ / DDΔ")
    print("═" * 110)
    arm_seq = list(ARMS.keys())
    for win_name, win_res in all_results.items():
        print(f"\n  [{win_name}]")
        base: BacktestResult = win_res["A_baseline"]
        base_n    = base.n_trades
        base_ret  = base.return_pct
        base_dd   = base.max_dd_pct
        base_wr   = base.win_rate
        base_avgr = base.avg_r
        base_best = base.best_r
        print(f"    {'Gate':<48}  N-Δ  AvgRΔ    BestRΔ   DDΔ     WRΔ    RetΔ    sun htf wkly rr doji")
        print(f"    {'─' * 107}")
        for arm_name in arm_seq[1:]:
            arm = ARMS[arm_name]
            r: BacktestResult = win_res[arm_name]
            nd    = r.n_trades - base_n
            avgrd = r.avg_r      - base_avgr
            bestd = r.best_r     - base_best
            ddd   = r.max_dd_pct - base_dd
            wrd   = r.win_rate   - base_wr
            retd  = r.return_pct - base_ret
            print(f"    {arm.label:<48}  "
                  f"{nd:+3}  {avgrd:+.2f}R   {bestd:+.2f}R   "
                  f"{ddd:+5.1f}%  {wrd:+4.0%}  {retd:+6.1f}%  "
                  f"{r.time_blocks:3} {r.countertrend_htf_blocks:3} "
                  f"{r.weekly_limit_blocks:4} {r.min_rr_small_blocks:2} "
                  f"{r.indecision_doji_blocks:4}")

    # ── Entry hour histogram ─────────────────────────────────────────────
    print(f"\n\n{'═' * 110}")
    print("  ENTRY HOUR HISTOGRAM (UTC) — Arm A baseline")
    print("═" * 100)
    from collections import Counter
    for win_name, win_res in all_results.items():
        r_base: BacktestResult = win_res["A_baseline"]
        trades = r_base.trades
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

    print(f"\n{'═' * 110}")
    print("  VERDICT")
    print("═" * 100)
    for win_name, win_res in all_results.items():
        ra: BacktestResult = win_res["A_baseline"]
        rf: BacktestResult = win_res[list(ARMS.keys())[-1]]
        print(f"\n  {win_name}:")
        print(f"    Baseline → All-gates:  {ra.n_trades} trades → {rf.n_trades} trades  "
              f"return {ra.return_pct:+.1f}% → {rf.return_pct:+.1f}%  "
              f"DD {ra.max_dd_pct:.1f}% → {rf.max_dd_pct:.1f}%  "
              f"avgR {ra.avg_r:+.2f}R → {rf.avg_r:+.2f}R")
        total_blk = (rf.time_blocks + rf.countertrend_htf_blocks +
                     rf.weekly_limit_blocks + rf.min_rr_small_blocks +
                     rf.indecision_doji_blocks)
        print(f"    Total entries blocked by new gates: {total_blk}")
    print()


if __name__ == "__main__":
    main()
