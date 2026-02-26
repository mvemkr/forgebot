"""
compare_strict_protrend.py
==========================
Baseline vs STRICT_PROTREND (revised)

STRICT_PROTREND v2 rules:
  ✓ Weekly == Daily bias required (macro direction confirmed on both HTFs)
  ✓ 4H MAY be mixed / counter — it is often mid-retracement at entry
      → The 1H engulfing at the key level IS the 4H confirmation
  ✓ MIN_RR = 3.0 enforced at final gate only (engine stays at 2.5)
      → select_target() still proposes 2.5R+ setups; final gate logs rejections
      → prevents pipeline starvation while still enforcing the 3R floor
  ✓ Engulf-only trigger (ENTRY_TRIGGER_MODE = "engulf_only")
  ✓ No Sunday entries
  ✓ No Thursday entries after 09:00 NY; no Friday entries
  ✓ 1 trade per week (both small- and standard-account cap = 1)
  ✓ Doji / indecision filter active
  ✓ Dynamic pip equity floor = stop_pips × MIN_RR_EFFECTIVE (3.0)

Report adds:
  % W==D aligned — of all ENTERED trades, what fraction had W+D macro confirmation

Windows:
  W1  Jul 15 – Oct 31 2024   (Alex benchmark, 108d)
  W2  Dec  1 – Feb 28 2026   (recent live,    89d)

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
STARTING_BAL      = 8_000.0
ARM_KEY           = "C"
TRAIL_CFG         = TRAIL_ARMS[ARM_KEY]
MIN_RR_EFFECTIVE  = 3.0    # final-gate threshold; engine stays at 2.5

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
    label:          str
    note:           str
    # strategy_config gate patches  (engine MIN_RR stays at 2.5 always)
    min_rr_gate:    float = 2.5    # final-gate threshold (MIN_RR_STANDARD = MIN_RR_COUNTERTREND)
    wk_cap_small:   int   = 999
    wk_cap_std:     int   = 999
    no_sun:         bool  = True
    no_thu_fri:     bool  = False
    htf_align:      bool  = True   # counter-trend block (all 3 oppose)
    doji_filter:    bool  = True
    # run_backtest extra params
    wd_protrend_htf:     bool = False   # require W+D agree, 4H exempt
    strict_protrend_htf: bool = False   # require W+D+4H all agree (overconstrained)
    dynamic_pip_equity:  bool = False
    policy: str = "baseline"


ARMS: dict[str, ArmDef] = {
    "BASE": ArmDef(
        label          = "Baseline",
        note           = "Current production: engulf_only + doji + no_sun + weekly 1/2",
        no_sun         = True,
        no_thu_fri     = False,
        wk_cap_small   = 1,
        wk_cap_std     = 2,
        min_rr_gate    = 2.5,
        policy         = "baseline",
    ),
    "STRICT": ArmDef(
        label          = "STRICT_PROTREND v2",
        note           = "W==D required; 4H exempt; MIN_RR gate=3.0; no_thu_fri; 1/wk; dyn-pip-eq",
        # Engine MIN_RR stays at 2.5 — only gate is raised
        min_rr_gate    = MIN_RR_EFFECTIVE,   # 3.0 at gate, not in select_target()
        wk_cap_small   = 1,
        wk_cap_std     = 1,   # 1/week for ALL account sizes
        no_sun         = True,
        no_thu_fri     = True,
        htf_align      = True,
        doji_filter    = True,
        wd_protrend_htf     = True,    # W+D must agree; 4H may be counter/mixed
        strict_protrend_htf = False,   # NOT requiring 4H (that was overconstrained)
        dynamic_pip_equity  = True,    # threshold = stop_pips × 3.0
        policy         = "alex_strict",
    ),
}

# ── Config save / patch / restore ────────────────────────────────────────────

_DEFAULTS = {
    # NOTE: MIN_RR (engine) is NOT patched — stays at 2.5.
    # Only the final-gate thresholds are raised for STRICT arm.
    "MIN_RR_STANDARD":              _cfg.MIN_RR_STANDARD,
    "MIN_RR_COUNTERTREND":          _cfg.MIN_RR_COUNTERTREND,
    "MIN_RR_SMALL_ACCOUNT":         _cfg.MIN_RR_SMALL_ACCOUNT,
    "MAX_TRADES_PER_WEEK_SMALL":    _cfg.MAX_TRADES_PER_WEEK_SMALL,
    "MAX_TRADES_PER_WEEK_STANDARD": _cfg.MAX_TRADES_PER_WEEK_STANDARD,
    "NO_SUNDAY_TRADES_ENABLED":     _cfg.NO_SUNDAY_TRADES_ENABLED,
    "NO_THU_FRI_TRADES_ENABLED":    _cfg.NO_THU_FRI_TRADES_ENABLED,
    "REQUIRE_HTF_TREND_ALIGNMENT":  _cfg.REQUIRE_HTF_TREND_ALIGNMENT,
    "INDECISION_FILTER_ENABLED":    _cfg.INDECISION_FILTER_ENABLED,
}


def _patch(arm: ArmDef) -> None:
    # Engine MIN_RR = 2.5 always: select_target() proposes 2.5R+ setups.
    # Raising the gate thresholds here controls the FINAL decision filter.
    _cfg.MIN_RR_STANDARD              = arm.min_rr_gate
    _cfg.MIN_RR_COUNTERTREND          = arm.min_rr_gate
    _cfg.MIN_RR_SMALL_ACCOUNT         = arm.min_rr_gate   # keep alias in sync
    _cfg.MAX_TRADES_PER_WEEK_SMALL    = arm.wk_cap_small
    _cfg.MAX_TRADES_PER_WEEK_STANDARD = arm.wk_cap_std
    _cfg.NO_SUNDAY_TRADES_ENABLED     = arm.no_sun
    _cfg.NO_THU_FRI_TRADES_ENABLED    = arm.no_thu_fri
    _cfg.REQUIRE_HTF_TREND_ALIGNMENT  = arm.htf_align
    _cfg.INDECISION_FILTER_ENABLED    = arm.doji_filter


def _restore() -> None:
    for k, v in _DEFAULTS.items():
        setattr(_cfg, k, v)


def _run(arm_key: str, arm: ArmDef, start: datetime, end: datetime) -> BacktestResult:
    _patch(arm)
    try:
        return run_backtest(
            start_dt             = start,
            end_dt               = end,
            starting_bal         = STARTING_BAL,
            trail_cfg            = TRAIL_CFG,
            trail_arm_key        = ARM_KEY,
            use_cache            = True,
            quiet                = True,
            wd_protrend_htf      = arm.wd_protrend_htf,
            strict_protrend_htf  = arm.strict_protrend_htf,
            dynamic_pip_equity   = arm.dynamic_pip_equity,
            policy_tag           = arm.policy,
        )
    finally:
        _restore()


def _trades_per_week(n: int, start: datetime, end: datetime) -> float:
    weeks = (end - start).days / 7.0
    return round(n / weeks, 2) if weeks > 0 else 0.0


def _fmt_row(label: str, r: BacktestResult, start: datetime, end: datetime,
             show_gates: bool = False) -> str:
    n_w = round(r.n_trades * r.win_rate)
    n_l = r.n_trades - n_w
    tpw = _trades_per_week(r.n_trades, start, end)

    row = (
        f"  {label:<28}  "
        f"N={r.n_trades:<3}  {n_w}W/{n_l}L  "
        f"Avg={r.avg_r:+.2f}R  "
        f"Best={r.best_r:+.2f}R  "
        f"Wrst={r.worst_r:+.2f}R  "
        f"DD={r.max_dd_pct:.1f}%  "
        f"Ret={r.return_pct:+.1f}%  "
        f"{tpw:.2f}/wk  "
        f"W==D={r.wd_alignment_pct:.0f}%"
    )
    if show_gates:
        row += (
            f"\n    Gate hits → "
            f"time={r.time_blocks}  "
            f"htf_ct={r.countertrend_htf_blocks}  "
            f"wd_htf={r.wd_htf_blocks}  "
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
    dtpw  = (_trades_per_week(strict.n_trades, start, end)
             - _trades_per_week(base.n_trades, start, end))
    dwd   = strict.wd_alignment_pct - base.wd_alignment_pct
    return (
        f"  Δ strict−base  "
        f"N={dn:+d}  AvgR={davgr:+.2f}R  "
        f"DD={ddd:+.1f}%  Ret={dret:+.1f}%  "
        f"Trades/wk={dtpw:+.2f}  W==D%={dwd:+.0f}pp"
    )


def main() -> None:
    SEP = "═" * 92
    sep = "─" * 92
    print(f"\n{SEP}")
    print("  STRICT_PROTREND v2 vs BASELINE")
    print("  W+D macro direction required; 4H exempt (retracement entry)")
    print(f"  Engine MIN_RR=2.5 (unchanged) | Gate MIN_RR={MIN_RR_EFFECTIVE:.1f} | Trail={ARM_KEY} | $8K | Alex 7 pairs")
    print(SEP)

    all_results: dict[str, dict[str, BacktestResult]] = {}

    for win_label, (wstart, wend) in WINDOWS.items():
        days = (wend - wstart).days
        print(f"\n{sep}")
        print(f"  Window: {win_label}  [{wstart.date()} → {wend.date()}]  ({days}d)")
        print(sep)

        win_results: dict[str, BacktestResult] = {}
        for arm_key, arm in ARMS.items():
            print(f"  Running {arm.label}…")
            r = _run(arm_key, arm, wstart, wend)
            win_results[arm_key] = r

        all_results[win_label] = win_results

        base   = win_results["BASE"]
        strict = win_results["STRICT"]

        print()
        print(f"  {'Arm':<28}  {'N':<4} W/L     Avg-R    Best-R   Wrst-R"
              f"   MaxDD   Ret%      Trd/wk  W==D%")
        print("  " + "─"*88)
        print(_fmt_row("Baseline",          base,   wstart, wend, show_gates=False))
        print(_fmt_row("STRICT_PROTREND v2", strict, wstart, wend, show_gates=True))
        print()
        print(_delta(base, strict, wstart, wend))

    # ── Combined summary ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SUMMARY — both windows")
    print(SEP)
    for win_label, win_results in all_results.items():
        wstart, wend = WINDOWS[win_label]
        days = (wend - wstart).days
        base   = win_results["BASE"]
        strict = win_results["STRICT"]
        print(f"\n  {win_label}  ({days}d)")
        print(_fmt_row("  Baseline          ", base,   wstart, wend))
        print(_fmt_row("  STRICT_PROTREND v2", strict, wstart, wend, show_gates=True))

    # ── Spec card ─────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  STRICT_PROTREND v2 spec")
    print(SEP)
    arm = ARMS["STRICT"]
    print(f"  Engine MIN_RR:                 2.5R  (select_target proposes 2.5R+ setups)")
    print(f"  Gate MIN_RR (effective):       {arm.min_rr_gate:.1f}R  (final decision filter — rejects 2.5-2.9R setups)")
    print(f"  Entry trigger:                 engulf_only")
    print(f"  HTF requirement:               Weekly == Daily bias (4H exempt — may be counter at entry)")
    print(f"  No Sunday:                     yes")
    print(f"  No Thu after 09:00 NY:         yes  (entry at bar close ≤ 09:00 ET allowed)")
    print(f"  No Friday:                     yes")
    print(f"  Weekly cap:                    1 trade/week (all account sizes)")
    print(f"  Doji / indecision filter:      yes")
    print(f"  Dynamic pip equity floor:      stop_pips × {arm.min_rr_gate:.1f}R")
    print(f"  Trail arm:                     {ARM_KEY}")
    print()


if __name__ == "__main__":
    main()
