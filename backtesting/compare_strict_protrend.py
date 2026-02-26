"""
compare_strict_protrend.py
==========================
Single arm comparison: Baseline vs STRICT_PROTREND

STRICT_PROTREND rules (all must pass):
  ✓ Weekly == Daily == 4H bias required (all 3 HTFs agree with direction)
  ✓ MIN_RR = 3.0 flat (select_target rejects < 3.0; alex_policy gate enforces it)
  ✓ Engulf-only trigger (ENTRY_TRIGGER_MODE = "engulf_only")
  ✓ No Sunday entries
  ✓ No Thursday entries after 09:00 NY
  ✓ 1 trade per week (both small- and standard-account cap = 1)
  ✓ Doji / indecision filter (INDECISION_FILTER_ENABLED = True)
  ✓ Dynamic pip equity = stop_pips × MIN_RR (not fixed 100p floor)

Windows:
  W1  Jul 15 – Oct 31 2024   (Alex benchmark, 108d)
  W2  Dec  1 – Feb 28 2026   (recent live,    89d)

Report columns:
  N | W/L | Avg-R | Best-R | Worst-R | Max-DD | Return% | Trades/wk
  plus gate hit counts for the STRICT arm

Usage:
    cd ~/trading-bot
    PYTHONPATH=/home/forge/trading-bot venv/bin/python backtesting/compare_strict_protrend.py
"""
from __future__ import annotations

import sys, os
from datetime import datetime, timezone
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.strategy.forex.strategy_config as _cfg
from backtesting.oanda_backtest_v2 import run_backtest, TRAIL_ARMS
from src.strategy.forex.backtest_schema import BacktestResult

# ── Constants ─────────────────────────────────────────────────────────────────
STARTING_BAL = 8_000.0
ARM_KEY      = "C"           # trail arm — must match existing compare scripts
TRAIL_CFG    = TRAIL_ARMS[ARM_KEY]
ADAPTIVE_THR = 0.5

WINDOWS = {
    "W1_Jul15-Oct31_2024": (
        datetime(2024, 7, 15, tzinfo=timezone.utc),
        datetime(2024, 10, 31, tzinfo=timezone.utc),
    ),
    "W2_Dec1-Feb28_2026": (
        datetime(2025, 12, 1, tzinfo=timezone.utc),
        datetime(2026, 2, 28, tzinfo=timezone.utc),
    ),
}

# ── Arm definitions ───────────────────────────────────────────────────────────

@dataclass
class ArmDef:
    label:        str
    note:         str
    # strategy_config patches
    min_rr:       float = 2.5
    min_rr_std:   float = 2.5
    min_rr_ct:    float = 3.0
    wk_cap_small: int   = 999
    wk_cap_std:   int   = 999
    no_sun:       bool  = True
    no_thu_fri:   bool  = False
    htf_align:    bool  = True   # REQUIRE_HTF_TREND_ALIGNMENT (block full counter-trend)
    doji_filter:  bool  = True
    # run_backtest extra params
    strict_protrend_htf: bool = False
    dynamic_pip_equity:  bool = False
    policy:       str  = "baseline"


ARMS: dict[str, ArmDef] = {
    "BASE": ArmDef(
        label     = "Baseline",
        note      = "Current production defaults: engulf_only + doji + no_sun + weekly cap 1/2",
        no_sun    = True,
        no_thu_fri= False,
        wk_cap_small = 1,
        wk_cap_std   = 2,
        policy    = "baseline",
    ),
    "STRICT": ArmDef(
        label     = "STRICT_PROTREND",
        note      = "W==D==4H required, MIN_RR 3.0, no_thu_fri, 1/wk, dyn-pip-eq",
        min_rr    = 3.0,        # strategy: select_target rejects < 3.0R
        min_rr_std= 3.0,        # alex_policy: pro-trend threshold
        min_rr_ct = 3.0,        # alex_policy: non-protrend threshold (same — always 3.0)
        wk_cap_small = 1,       # 1/week regardless of balance
        wk_cap_std   = 1,
        no_sun    = True,
        no_thu_fri= True,
        htf_align = True,       # also keep existing counter-trend block
        doji_filter = True,
        strict_protrend_htf = True,   # all 3 HTFs must AGREE (not just not all oppose)
        dynamic_pip_equity  = True,   # threshold = stop_pips × MIN_RR
        policy    = "alex_strict",
    ),
}

# ── Saved defaults ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "MIN_RR":                      _cfg.MIN_RR,
    "MIN_RR_STANDARD":             _cfg.MIN_RR_STANDARD,
    "MIN_RR_COUNTERTREND":         _cfg.MIN_RR_COUNTERTREND,
    "MIN_RR_SMALL_ACCOUNT":        _cfg.MIN_RR_SMALL_ACCOUNT,
    "MAX_TRADES_PER_WEEK_SMALL":   _cfg.MAX_TRADES_PER_WEEK_SMALL,
    "MAX_TRADES_PER_WEEK_STANDARD":_cfg.MAX_TRADES_PER_WEEK_STANDARD,
    "NO_SUNDAY_TRADES_ENABLED":    _cfg.NO_SUNDAY_TRADES_ENABLED,
    "NO_THU_FRI_TRADES_ENABLED":   _cfg.NO_THU_FRI_TRADES_ENABLED,
    "REQUIRE_HTF_TREND_ALIGNMENT": _cfg.REQUIRE_HTF_TREND_ALIGNMENT,
    "INDECISION_FILTER_ENABLED":   _cfg.INDECISION_FILTER_ENABLED,
}


def _patch(arm: ArmDef) -> None:
    _cfg.MIN_RR                      = arm.min_rr
    _cfg.MIN_RR_STANDARD             = arm.min_rr_std
    _cfg.MIN_RR_COUNTERTREND         = arm.min_rr_ct
    _cfg.MIN_RR_SMALL_ACCOUNT        = arm.min_rr_ct   # keep alias in sync
    _cfg.MAX_TRADES_PER_WEEK_SMALL   = arm.wk_cap_small
    _cfg.MAX_TRADES_PER_WEEK_STANDARD= arm.wk_cap_std
    _cfg.NO_SUNDAY_TRADES_ENABLED    = arm.no_sun
    _cfg.NO_THU_FRI_TRADES_ENABLED   = arm.no_thu_fri
    _cfg.REQUIRE_HTF_TREND_ALIGNMENT = arm.htf_align
    _cfg.INDECISION_FILTER_ENABLED   = arm.doji_filter


def _restore() -> None:
    for k, v in _DEFAULTS.items():
        setattr(_cfg, k, v)


def _run(arm_key: str, arm: ArmDef, start: datetime, end: datetime,
         candle_data: dict | None) -> BacktestResult:
    _patch(arm)
    try:
        return run_backtest(
            start_dt              = start,
            end_dt                = end,
            starting_bal          = STARTING_BAL,
            trail_cfg             = TRAIL_CFG,
            trail_arm_key         = ARM_KEY,
            preloaded_candle_data = candle_data,
            use_cache             = True,
            quiet                 = True,
            strict_protrend_htf   = arm.strict_protrend_htf,
            dynamic_pip_equity    = arm.dynamic_pip_equity,
            policy_tag            = arm.policy,
        )
    finally:
        _restore()


def _trades_per_week(result: BacktestResult, start: datetime, end: datetime) -> float:
    days  = (end - start).days
    weeks = days / 7.0
    return round(result.n_trades / weeks, 2) if weeks > 0 else 0.0


def _fmt_row(label: str, r: BacktestResult, start: datetime, end: datetime,
             days: int, show_gates: bool = False) -> str:
    tpw  = _trades_per_week(r, start, end)
    wl   = f"{r.n_trades-int(r.n_trades*r.win_rate)}W/{int(r.n_trades*(1-r.win_rate))}L"
    # recompute W from n_wins (safer)
    n_w  = round(r.n_trades * r.win_rate)
    n_l  = r.n_trades - n_w
    wl   = f"{n_w}W/{n_l}L"
    row  = (
        f"  {label:<28}  "
        f"N={r.n_trades:<3}  {wl:<8}  "
        f"Avg={r.avg_r:+.2f}R  "
        f"Best={r.best_r:+.2f}R  "
        f"Wrst={r.worst_r:+.2f}R  "
        f"DD={r.max_dd_pct:.1f}%  "
        f"Ret={r.return_pct:+.1f}%  "
        f"{tpw:.2f}/wk"
    )
    if show_gates:
        row += (
            f"\n    Gate hits → "
            f"time={r.time_blocks}  htf={r.countertrend_htf_blocks}  "
            f"strict_htf={r.strict_htf_blocks}  "
            f"wkly={r.weekly_limit_blocks}  "
            f"rr={r.min_rr_small_blocks}  "
            f"dyn_pe={r.dyn_pip_eq_blocks}  "
            f"doji={r.indecision_doji_blocks}"
        )
    return row


def _delta(base: BacktestResult, strict: BacktestResult,
           start: datetime, end: datetime) -> str:
    dn    = strict.n_trades - base.n_trades
    davgr = strict.avg_r    - base.avg_r
    ddd   = strict.max_dd_pct - base.max_dd_pct
    dret  = strict.return_pct - base.return_pct
    dtpw  = _trades_per_week(strict, start, end) - _trades_per_week(base, start, end)
    return (
        f"  Δ strict−base  "
        f"N={dn:+d}  "
        f"AvgR={davgr:+.2f}R  "
        f"DD={ddd:+.1f}%  "
        f"Ret={dret:+.1f}%  "
        f"Trades/wk={dtpw:+.2f}"
    )


def main() -> None:
    print("\n" + "═"*90)
    print("  STRICT_PROTREND vs BASELINE — Gate Experiment")
    print("  Trail Arm C | MAX_CONCURRENT=1 | START=$8,000 | Alex 7 pairs")
    print("  Strict rules: W==D==4H, MIN_RR=3.0, no_sun, no_thu_fri, 1/wk, dyn-pip-eq")
    print("═"*90)

    all_results: dict[str, dict[str, BacktestResult]] = {}

    for win_label, (wstart, wend) in WINDOWS.items():
        days = (wend - wstart).days
        print(f"\n{'─'*90}")
        print(f"  Window: {win_label}  [{wstart.date()} → {wend.date()}]  ({days}d)")
        print(f"{'─'*90}")

        # Pre-load candle data once; share across arms (no re-fetch)
        print("  Pre-loading candle data…")
        _candle_data = None  # first run will load+cache; subsequent arm reuses

        win_results: dict[str, BacktestResult] = {}
        for arm_key, arm in ARMS.items():
            print(f"  Running {arm.label}…")
            r = _run(arm_key, arm, wstart, wend, _candle_data)
            win_results[arm_key] = r
            # After first run, extract preloaded data to share
            # (run_backtest already uses disk cache; this just avoids re-parsing)

        all_results[win_label] = win_results

        print()
        print(f"  {'Arm':<28}  {'N':<4}  {'W/L':<8}  {'Avg-R':<8}  {'Best-R':<8}"
              f"  {'Wrst-R':<8}  {'MaxDD':<7}  {'Ret%':<8}  {'Trd/wk'}")
        print("  " + "─"*86)

        base   = win_results["BASE"]
        strict = win_results["STRICT"]

        print(_fmt_row("Baseline",        base,   wstart, wend, days, show_gates=False))
        print(_fmt_row("STRICT_PROTREND", strict, wstart, wend, days, show_gates=True))
        print()
        print(_delta(base, strict, wstart, wend))

    # ── Summary across both windows ───────────────────────────────────────────
    print("\n" + "═"*90)
    print("  SUMMARY — both windows combined")
    print("═"*90)
    for win_label, win_results in all_results.items():
        wstart, wend = WINDOWS[win_label]
        days = (wend - wstart).days
        base   = win_results["BASE"]
        strict = win_results["STRICT"]
        print(f"\n  {win_label}  ({days}d)")
        print(_fmt_row("  Baseline       ", base,   wstart, wend, days))
        print(_fmt_row("  STRICT_PROTREND", strict, wstart, wend, days, show_gates=True))

    # ── STRICT_PROTREND spec card ─────────────────────────────────────────────
    print("\n" + "═"*90)
    print("  STRICT_PROTREND spec")
    print("═"*90)
    arm = ARMS["STRICT"]
    print(f"  MIN_RR (strategy + gate):      {arm.min_rr:.1f} R")
    print(f"  Entry trigger:                 engulf_only")
    print(f"  No Sunday:                     yes")
    print(f"  No Thu after 09:00 NY:         yes")
    print(f"  No Friday:                     yes")
    print(f"  Weekly cap:                    1 trade/week (all account sizes)")
    print(f"  Doji / indecision filter:      yes")
    print(f"  Strict HTF (W==D==4H agree):   yes — all 3 must align with direction")
    print(f"  Dynamic pip equity floor:      stop_pips × {arm.min_rr:.1f} (replaces 100p fixed)")
    print(f"  Trail arm:                     {ARM_KEY}")
    print()


if __name__ == "__main__":
    main()
