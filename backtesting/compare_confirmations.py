"""
compare_confirmations.py
========================
Strict parity comparison of confirmation trigger modes + indecision filter.

Arms:
  A  engulf_only          + no indecision filter  (pure baseline)
  B  engulf_only          + indecision filter ON   (doji gate added)
  C  engulf_or_star_at_level + indecision filter ON (Morning/Evening Star at level)

Windows:
  W1  Jul 15 – Oct 31 2024  (Alex's benchmark window)
  W2  Jan 1  – Jan 31 2026  (recent live window)

Strict parity: MAX_CONCURRENT=1, MIN_RR=2.5, Alex 7-pair whitelist, Arm C trail.
All runs share the same candle_cache (no extra API calls after first warm-up).

Usage:
    cd ~/trading-bot
    PYTHONPATH=/home/forge/trading-bot venv/bin/python backtesting/compare_confirmations.py
"""
from __future__ import annotations
import sys
import os
import io
from datetime import datetime, timezone
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.strategy.forex.strategy_config as _cfg
from backtesting.oanda_backtest_v2 import run_backtest, TRAIL_ARMS
from src.strategy.forex.backtest_schema import BacktestResult

_MAX_CONCURRENT = _cfg.MAX_CONCURRENT_TRADES_LIVE
_MIN_RR         = _cfg.MIN_RR

# ── Parity assertions ─────────────────────────────────────────────────────
assert _MAX_CONCURRENT == 1, f"MAX_CONCURRENT_TRADES_LIVE must be 1, got {_MAX_CONCURRENT}"
assert _MIN_RR == 2.5,       f"MIN_RR must be 2.5, got {_MIN_RR}"
assert _cfg.ENTRY_TRIGGER_MODE == "engulf_only", \
    f"Strategy default must be engulf_only, got {_cfg.ENTRY_TRIGGER_MODE}"

# ── Windows ───────────────────────────────────────────────────────────────
WINDOWS = {
    "W1_Alex_Jul15-Oct31_2024": (
        datetime(2024, 7, 15, tzinfo=timezone.utc),
        datetime(2024, 10, 31, tzinfo=timezone.utc),
    ),
    "W2_Jan_2026": (
        datetime(2026, 1,  1,  tzinfo=timezone.utc),
        datetime(2026, 1, 31,  tzinfo=timezone.utc),
    ),
}

# ── Arm definitions ───────────────────────────────────────────────────────
ARMS = {
    "A_engulf_only_no_filter": {
        "trigger_mode":       "engulf_only",
        "indecision_filter":  False,
        "label":              "engulf_only (no doji filter)",
    },
    "B_engulf_only_doji_gate": {
        "trigger_mode":       "engulf_only",
        "indecision_filter":  True,
        "label":              "engulf_only + doji gate",
    },
    "C_star_at_level_doji_gate": {
        "trigger_mode":       "engulf_or_star_at_level",
        "indecision_filter":  True,
        "label":              "engulf_or_star_at_level + doji gate",
    },
}

STARTING_BAL = 8_000.0
ARM_KEY = "C"  # ratchet trail arm


def _patch(trigger_mode: str, indecision_filter: bool) -> None:
    """Patch strategy_config for one backtest run."""
    _cfg.ENTRY_TRIGGER_MODE     = trigger_mode
    _cfg.ENGULFING_ONLY         = (trigger_mode == "engulf_only")
    _cfg.INDECISION_FILTER_ENABLED = indecision_filter


def _restore() -> None:
    """Restore strategy_config to production defaults."""
    _cfg.ENTRY_TRIGGER_MODE        = "engulf_only"
    _cfg.ENGULFING_ONLY            = True
    _cfg.INDECISION_FILTER_ENABLED = True


def _exit_table(trades: list[dict]) -> str:
    """Compact exit-reason breakdown: 'target_reached×3(+3.2R) stop_hit×9(-1.0R)'"""
    ctr: dict[str, list] = {}
    for t in trades:
        r = t.get("reason", "?")
        ctr.setdefault(r, []).append(t.get("r", 0.0))
    parts = []
    for reason, rs in sorted(ctr.items(), key=lambda x: -len(x[1])):
        avg_r = sum(rs) / len(rs) if rs else 0
        sign  = "+" if avg_r >= 0 else ""
        parts.append(f"{reason}×{len(rs)}({sign}{avg_r:.2f}R)")
    return "  ".join(parts)


def _signal_table(trades: list[dict]) -> str:
    """Signal type breakdown."""
    ctr: Counter = Counter(t.get("signal_type", "?") for t in trades)
    return "  ".join(f"{sig}×{cnt}" for sig, cnt in ctr.most_common())


def _indecision_blocks(result: BacktestResult) -> int:
    """Return the INDECISION_DOJI block count from BacktestResult."""
    return result.indecision_doji_blocks


def _run_arm(arm_cfg: dict, start: datetime, end: datetime) -> BacktestResult:
    """Run one backtest arm, capturing all output."""
    _patch(arm_cfg["trigger_mode"], arm_cfg["indecision_filter"])
    try:
        result = run_backtest(
            start_dt     = start,
            end_dt       = end,
            starting_bal = STARTING_BAL,
            trail_cfg    = TRAIL_ARMS[ARM_KEY],
            quiet        = True,
        )
    finally:
        _restore()
    return result


def _fmt_row(arm_name: str, arm_cfg: dict, result: BacktestResult) -> list:
    """Return a list of cells for the results table."""
    n_trades = result.n_trades
    ret_pct  = result.return_pct
    max_dd   = result.max_dd_pct
    wr       = result.win_rate
    avg_r    = result.avg_r
    best_r   = result.best_r
    worst_r  = result.worst_r
    trades   = result.trades
    n_doji   = _indecision_blocks(result)
    wins     = sum(1 for t in trades if t.get("r", 0) > 0.1)
    losses   = sum(1 for t in trades if t.get("r", 0) < -0.1)
    scratch  = n_trades - wins - losses
    return [
        arm_cfg["label"],
        n_trades,
        f"{ret_pct:+.1f}%",
        f"{max_dd:.1f}%",
        f"{wr:.0%}",
        f"{avg_r:+.2f}R",
        f"{best_r:+.2f}R",
        f"{worst_r:+.2f}R",
        f"{wins}W/{losses}L/{scratch}S",
        f"{n_doji}",
    ]


def main():
    print("\n" + "═" * 90)
    print("  CONFIRMATION MODE COMPARISON — Morning/Evening Star + Indecision Filter")
    print("═" * 90)
    print(f"  Trail: Arm {ARM_KEY}  |  MAX_CONCURRENT={_MAX_CONCURRENT}  |  MIN_RR={_MIN_RR}  |  START_BAL=${STARTING_BAL:,.0f}")
    print(f"  Parity: engulf_only baseline ✓  |  indecision filter on/off comparison ✓")
    print()

    COL_HDR = ["Mode", "N", "Ret%", "MaxDD", "WR", "AvgR", "BestR", "WorstR", "W/L/S", "DojiBlk"]
    COL_W   = [42,      5,   8,      7,       6,    8,      8,       8,        12,       8]

    shared_cache: dict = {}

    all_results: dict[str, dict[str, dict]] = {}   # window → arm → result

    for win_name, (start, end) in WINDOWS.items():
        print(f"\n{'─' * 90}")
        print(f"  Window: {win_name}  [{start.date()} → {end.date()}]")
        print(f"{'─' * 90}")

        # Header
        header = "  ".join(f"{h:<{w}}" for h, w in zip(COL_HDR, COL_W))
        print(f"  {header}")
        print(f"  {'─' * 85}")

        win_results: dict[str, dict] = {}

        for arm_name, arm_cfg in ARMS.items():
            result = _run_arm(arm_cfg, start, end)
            win_results[arm_name] = result

            row = _fmt_row(arm_name, arm_cfg, result)
            row_str = "  ".join(f"{str(cell):<{w}}" for cell, w in zip(row, COL_W))
            print(f"  {row_str}")

        all_results[win_name] = win_results

    # ── Per-arm detailed breakdown ────────────────────────────────────────
    print(f"\n\n{'═' * 90}")
    print("  DETAILED EXIT & SIGNAL BREAKDOWN")
    print("═" * 90)

    for win_name, win_results in all_results.items():
        print(f"\n  [{win_name}]")
        for arm_name, arm_cfg in ARMS.items():
            result: BacktestResult = win_results[arm_name]
            if not result.trades:
                print(f"    {arm_cfg['label']:42}  — no trades")
                continue
            print(f"    {arm_cfg['label']}")
            print(f"      Exits:   {_exit_table(result.trades)}")
            print(f"      Signals: {_signal_table(result.trades)}")
            star_trades = [t for t in result.trades
                           if t.get("signal_type","") in ("morning_star","evening_star")]
            if star_trades:
                print(f"      Stars ({len(star_trades)}):")
                for t in star_trades:
                    r = t.get('r', 0)
                    print(f"        {t.get('entry_ts','')[:10]} {t['pair']:8} {t['direction']:5} "
                          f"{t.get('signal_type','?'):15} R={r:+.2f}")

    # ── Indecision filter impact ──────────────────────────────────────────
    print(f"\n\n{'═' * 90}")
    print("  INDECISION FILTER IMPACT  (Arm A [no filter] → Arm B [filter ON])")
    print("═" * 90)
    for win_name, win_results in all_results.items():
        rA: BacktestResult = win_results.get("A_engulf_only_no_filter")
        rB: BacktestResult = win_results.get("B_engulf_only_doji_gate")
        if not rA or not rB:
            continue
        dA = _indecision_blocks(rA); dB = _indecision_blocks(rB)
        nA = rA.n_trades; nB = rB.n_trades
        print(f"\n  {win_name}")
        print(f"    Doji blocks:  A={dA}  B={dB}  (net removed = {nA-nB} trades)")
        print(f"    Return:       A={rA.return_pct:+.1f}%  B={rB.return_pct:+.1f}%  "
              f"(Δ={rB.return_pct-rA.return_pct:+.1f}%)")
        print(f"    Max DD:       A={rA.max_dd_pct:.1f}%   B={rB.max_dd_pct:.1f}%")
        print(f"    WR:           A={rA.win_rate:.0%}   B={rB.win_rate:.0%}")
        print(f"    Avg R:        A={rA.avg_r:+.2f}   B={rB.avg_r:+.2f}")

    print(f"\n{'═' * 90}")
    print("  VERDICT")
    print("═" * 90)
    for win_name, win_results in all_results.items():
        rA: BacktestResult = win_results.get("A_engulf_only_no_filter")
        rB: BacktestResult = win_results.get("B_engulf_only_doji_gate")
        rC: BacktestResult = win_results.get("C_star_at_level_doji_gate")
        if not rA or not rB or not rC:
            continue
        star_count = sum(1 for t in rC.trades
                         if t.get("signal_type","") in ("morning_star","evening_star"))
        print(f"\n  {win_name}:")
        print(f"    Doji filter:   {rA.return_pct:+.1f}% → {rB.return_pct:+.1f}% "
              f"(Δ={rB.return_pct-rA.return_pct:+.1f}%)")
        print(f"    Stars added:   {rB.return_pct:+.1f}% → {rC.return_pct:+.1f}% "
              f"(Δ={rC.return_pct-rB.return_pct:+.1f}%)  "
              f"| {star_count} star trades out of {rC.n_trades} total")

    _restore()
    print()


if __name__ == "__main__":
    main()
