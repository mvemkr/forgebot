#!/usr/bin/env python3
"""
Weekly trade cap ablation study — offline replay only.

Context
-------
  Every prior ablation (MIN_RR, confidence, session filter) was degraded by
  cascade displacement through the weekly cap.  This study isolates the cap
  itself as the variable to determine if it is correctly calibrated or if it
  is suppressing net return by blocking additional quality setups.

Variants
--------
  A  cap=1  MAX_TRADES_PER_WEEK_SMALL=1  (current production — small account)
  B  cap=2  MAX_TRADES_PER_WEEK_SMALL=2
  C  cap=3  MAX_TRADES_PER_WEEK_SMALL=3

Analysis
--------
  • Standard per-window metrics (trades, WR, SumR, MaxDD, Worst3L, MAE, MFE)
  • Per-slot quality: avg R of trade #1, #2, #3 across all weeks
  • Weeks with 2+ qualifying setups (cap blocked 2nd in Variant A)
  • Weeks with 3+ qualifying setups (cap blocked 3rd in Variant B)
  • Quality verdict: are additional trades same/better/worse than first?

Safety
------
  atexit() resets MAX_TRADES_PER_WEEK_SMALL to production default even on crash.
"""

from __future__ import annotations

import atexit
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

import src.strategy.forex.strategy_config as _sc

# ── atexit guard ──────────────────────────────────────────────────────────────
_ORIG_CAP_SMALL    = getattr(_sc, "MAX_TRADES_PER_WEEK_SMALL",    1)
_ORIG_CAP_STANDARD = getattr(_sc, "MAX_TRADES_PER_WEEK_STANDARD", 2)

def _reset_cap_config():
    _sc.MAX_TRADES_PER_WEEK_SMALL    = _ORIG_CAP_SMALL
    _sc.MAX_TRADES_PER_WEEK_STANDARD = _ORIG_CAP_STANDARD

atexit.register(_reset_cap_config)

# ── backtester import ─────────────────────────────────────────────────────────
from backtesting.oanda_backtest_v2 import run_backtest, BacktestResult  # noqa

# ──────────────────────────────────────────────────────────────────────────────
CAPITAL     = 8_000.0
REPORT_PATH = REPO / "backtesting/results/ablation_weekly_cap.md"
UTC = timezone.utc

WINDOWS: List[Tuple[str, datetime, datetime]] = [
    ("Q1-2025",      datetime(2025, 1, 1,  tzinfo=UTC), datetime(2025, 3, 31, tzinfo=UTC)),
    ("Q2-2025",      datetime(2025, 4, 1,  tzinfo=UTC), datetime(2025, 6, 30, tzinfo=UTC)),
    ("Q3-2025",      datetime(2025, 7, 1,  tzinfo=UTC), datetime(2025, 9, 30, tzinfo=UTC)),
    ("Q4-2025",      datetime(2025, 10, 1, tzinfo=UTC), datetime(2025, 12, 31, tzinfo=UTC)),
    ("Jan-Feb-2026", datetime(2026, 1, 1,  tzinfo=UTC), datetime(2026, 2, 28, tzinfo=UTC)),
    ("W1",           datetime(2026, 2, 17, tzinfo=UTC), datetime(2026, 2, 21, tzinfo=UTC)),
    ("W2",           datetime(2026, 2, 24, tzinfo=UTC), datetime(2026, 2, 28, tzinfo=UTC)),
    ("live-parity",  datetime(2026, 3, 2,  tzinfo=UTC), datetime(2026, 3, 8,  tzinfo=UTC)),
]

# (label, cap_small, short_desc)
VARIANTS: List[Tuple[str, int, str]] = [
    ("A", 1, "Baseline — cap=1 (current production, small account)"),
    ("B", 2, "cap=2 — allow two trades per week"),
    ("C", 3, "cap=3 — allow three trades per week"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _iso_week(ts) -> Tuple[int, int]:
    """Return (iso_year, iso_week) from a trade's entry_ts."""
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return (0, 0)
    if ts is None:
        return (0, 0)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.isocalendar()[:2]


def _assign_slots(trades: List[Dict]) -> List[Dict]:
    """
    Sort trades by entry_ts and assign a slot number (1-based) per ISO week.
    Returns a new list of dicts with 'week_slot' added.
    """
    def _sort_key(t):
        ts = t.get("entry_ts") or ""
        if isinstance(ts, str):
            return ts
        if hasattr(ts, "isoformat"):
            return ts.isoformat()
        return str(ts)

    sorted_trades = sorted(trades, key=_sort_key)
    week_counts: Dict[Tuple, int] = defaultdict(int)
    result = []
    for t in sorted_trades:
        wk = _iso_week(t.get("entry_ts"))
        week_counts[wk] += 1
        t2 = dict(t)
        t2["week_slot"] = week_counts[wk]
        t2["iso_week"]  = wk
        result.append(t2)
    return result


def _count_weeks_with_cap_blocks(gap_log: List[Dict], min_blocks: int = 1) -> int:
    """
    Count distinct ISO weeks that had ≥ min_blocks WEEKLY_TRADE_LIMIT blocks.
    Used to determine: 'how many weeks had a 2nd/3rd setup available but blocked'.
    """
    week_blocks: Dict[Tuple, int] = defaultdict(int)
    for g in gap_log:
        if g.get("gap_type") == "WEEKLY_TRADE_LIMIT":
            ts = g.get("ts") or g.get("timestamp") or ""
            wk = _iso_week(ts)
            week_blocks[wk] += 1
    return sum(1 for cnt in week_blocks.values() if cnt >= min_blocks)


def _slot_quality(all_slotted: List[Dict]) -> Dict[int, Dict]:
    """
    Compute aggregate quality metrics per slot number across all trades.
    Returns {slot: {count, sumr, wins, avg_r, win_rate, mae_avg, mfe_avg}}.
    """
    slots: Dict[int, Dict] = defaultdict(lambda: {"count": 0, "sumr": 0.0,
                                                   "wins": 0, "mae": [], "mfe": []})
    for t in all_slotted:
        s = t.get("week_slot", 1)
        r = t.get("r", 0.0)
        slots[s]["count"] += 1
        slots[s]["sumr"]  += r
        if r > 0:
            slots[s]["wins"] += 1
        mae = t.get("mae_r")
        mfe = t.get("mfe_r")
        if mae is not None:
            slots[s]["mae"].append(mae)
        if mfe is not None:
            slots[s]["mfe"].append(mfe)
    result = {}
    for s, d in slots.items():
        n = d["count"]
        result[s] = {
            "count":    n,
            "sumr":     d["sumr"],
            "avg_r":    d["sumr"] / n if n else 0.0,
            "win_rate": d["wins"] / n if n else 0.0,
            "mae_avg":  sum(d["mae"]) / len(d["mae"]) if d["mae"] else None,
            "mfe_avg":  sum(d["mfe"]) / len(d["mfe"]) if d["mfe"] else None,
        }
    return result


def _additional_trades(
    slotted_a: List[Dict],
    slotted_x: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split slotted_x into:
      - baseline_trades: slot=1 trades (same as A's universe)
      - additional_trades: slot≥2 trades (new relative to A)
    """
    baseline   = [t for t in slotted_x if t.get("week_slot", 1) == 1]
    additional = [t for t in slotted_x if t.get("week_slot", 1) >= 2]
    return baseline, additional


def _r(v) -> str:
    if v is None:
        return "—"
    return f"{v:+.2f}R"


def _pct(v) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}%"


def _wl(t: Dict) -> str:
    return "✅ W" if t.get("r", 0) > 0 else "❌ L"


def _fmt_ts(ts) -> str:
    if isinstance(ts, str):
        return ts[:16].replace("T", " ")
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d %H:%M")
    return str(ts)


# ──────────────────────────────────────────────────────────────────────────────
# Run variants
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation(verbose: bool = False) -> Dict[str, Dict[str, BacktestResult]]:
    """Run all 8 windows × 3 variants. Returns results[window][variant]."""
    results: Dict[str, Dict[str, BacktestResult]] = {}

    for win_name, win_start, win_end in WINDOWS:
        results[win_name] = {}
        preloaded: Optional[Dict] = None

        print(f"\n  ── Window: {win_name} ──────────────────────────────")
        for vlabel, cap, _ in VARIANTS:
            _sc.MAX_TRADES_PER_WEEK_SMALL    = cap
            _sc.MAX_TRADES_PER_WEEK_STANDARD = max(cap, _ORIG_CAP_STANDARD)

            print(f"    Variant {vlabel} (cap={cap})…", flush=True)
            result = run_backtest(
                win_start, win_end,
                starting_bal=CAPITAL,
                notes=f"wkcap_{vlabel}_{win_name}",
                trail_arm_key="A",
                preloaded_candle_data=preloaded,
                use_cache=True,
                quiet=not verbose,
            )
            results[win_name][vlabel] = result
            if preloaded is None:
                preloaded = result.candle_data
            sumr = sum(t.get("r", 0) for t in result.trades)
            print(f"      → {result.n_trades}T  {result.win_rate*100:.0f}%WR  "
                  f"SumR={sumr:+.2f}R  wk_blocks={result.weekly_limit_blocks}")

        # Reset after each window
        _sc.MAX_TRADES_PER_WEEK_SMALL    = _ORIG_CAP_SMALL
        _sc.MAX_TRADES_PER_WEEK_STANDARD = _ORIG_CAP_STANDARD

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(results: Dict[str, Dict[str, BacktestResult]]) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    L = lines.append

    L("# Weekly Trade Cap Ablation Study")
    L(f"\nGenerated: {now}  |  Branch: `feat/weekly-cap-ablation`")
    L("")
    L("## Context")
    L("")
    L("Every prior ablation (MIN_RR, confidence, session filter) saw SumR")
    L("degraded by cascade displacement through the weekly cap.  This study")
    L("isolates the cap as the variable to determine if it is correctly")
    L("calibrated or if it is blocking net-positive additional setups.")
    L("")
    L("## Setup")
    L("")
    L("| | |")
    L("|---|---|")
    L("| Capital | $8,000 (small-account tier, cap=SMALL throughout) |")
    L("| Stop logic | C8 (structural + 3×ATR_1H ceiling + 8-pip floor) |")
    L("| Trigger mode | `engulf_or_strict_pin_at_level` (B-Prime) |")
    L("| MIN_RR_STANDARD | 2.5 |")
    L("| MIN_CONFIDENCE | 0.77 |")
    L("| Session filter | Unchanged (Thu≤09:00 ET, Mon≥08:00 ET) |")
    L("")
    L("### Variants")
    L("")
    L("| Variant | MAX_TRADES_PER_WEEK_SMALL | Description |")
    L("|---------|:------------------------:|-------------|")
    for vlabel, cap, desc in VARIANTS:
        prod = " **(production)**" if vlabel == "A" else ""
        L(f"| **{vlabel}** | {cap}{prod} | {desc} |")
    L("")

    # ── 1. Per-window breakdown ───────────────────────────────────────────
    L("## 1. Per-Window Breakdown")
    L("")
    L("| Window | Var | T | WR | SumR | AvgR | MaxDD | WkBlocks | Worst3L |")
    L("|--------|-----|:--:|:--:|:----:|:----:|:-----:|:--------:|---------|")

    agg: Dict[str, Dict] = {v[0]: {"trades": [], "sumr": 0.0, "dd": []} for v in VARIANTS}

    for win_name, _, _ in WINDOWS:
        for vi, (vlabel, cap, _) in enumerate(VARIANTS):
            r  = results[win_name][vlabel]
            ts = r.trades
            sumr  = sum(t.get("r", 0) for t in ts)
            avgr  = sumr / len(ts) if ts else 0.0
            dd    = r.max_dd_pct or 0.0
            worst = sorted(t.get("r", 0) for t in ts)[:3]
            w3    = ", ".join(f"{x:+.2f}R" for x in worst) if worst else "—"
            wblk  = r.weekly_limit_blocks

            L(f"| {win_name if vi == 0 else ''} | {vlabel} "
              f"| {len(ts)} | {r.win_rate*100:.0f}% "
              f"| {sumr:+.2f}R | {avgr:+.3f}R "
              f"| {dd:.1f}% | {wblk} | {w3} |")

            agg[vlabel]["trades"].extend(ts)
            agg[vlabel]["sumr"]  += sumr
            agg[vlabel]["dd"].append(dd)
        L("| | | | | | | | | |")

    L("")

    # ── 2. Aggregate summary ──────────────────────────────────────────────
    L("## 2. Aggregate Summary")
    L("")
    L("| Variant | Cap | Trades | WR | SumR | vs A | AvgR | Avg MaxDD |")
    L("|---------|:---:|:------:|:--:|:----:|:----:|:----:|:---------:|")
    a_sumr = agg["A"]["sumr"]
    for vlabel, cap, _ in VARIANTS:
        a  = agg[vlabel]
        ts = a["trades"]
        s  = a["sumr"]
        wr = sum(1 for t in ts if t.get("r", 0) > 0) / len(ts) * 100 if ts else 0.0
        ar = s / len(ts) if ts else 0.0
        ad = sum(a["dd"]) / len(a["dd"]) if a["dd"] else 0.0
        vs = "—" if vlabel == "A" else f"{s - a_sumr:+.2f}R"
        L(f"| **{vlabel}** | {cap} | {len(ts)} | {wr:.0f}% | {s:+.2f}R | {vs} | {ar:+.3f}R | {ad:.1f}% |")
    L("")

    # ── 3. Per-slot quality analysis ──────────────────────────────────────
    L("## 3. Per-Slot Quality Analysis")
    L("")
    L("Average R, WR, MAE, MFE broken down by which trade in the week it was")
    L("(slot 1 = first trade entered that ISO week, slot 2 = second, etc.).")
    L("")

    all_slotted_by_variant: Dict[str, List[Dict]] = {}

    for vlabel, cap, _ in VARIANTS:
        all_trades: List[Dict] = []
        for win_name, _, _ in WINDOWS:
            r  = results[win_name][vlabel]
            all_trades.extend(r.trades)
        slotted = _assign_slots(all_trades)
        all_slotted_by_variant[vlabel] = slotted

    # Combined slot table
    L("### Slot quality — all windows combined")
    L("")
    L("| Variant | Slot | Count | WR | AvgR | MAE avg | MFE avg |")
    L("|---------|:----:|:-----:|:--:|:----:|:-------:|:-------:|")
    for vlabel, cap, _ in VARIANTS:
        slotted = all_slotted_by_variant[vlabel]
        sq = _slot_quality(slotted)
        for slot in sorted(sq.keys()):
            d   = sq[slot]
            wr  = d["win_rate"] * 100
            ar  = d["avg_r"]
            mae = f"{d['mae_avg']:+.3f}R" if d["mae_avg"] is not None else "—"
            mfe = f"{d['mfe_avg']:+.3f}R" if d["mfe_avg"] is not None else "—"
            L(f"| {vlabel} | #{slot} | {d['count']} | {wr:.0f}% | {ar:+.3f}R | {mae} | {mfe} |")
        L(f"| | | | | | | |")
    L("")

    # Quality verdict
    L("### Quality degradation assessment")
    L("")
    for vlabel, cap, _ in VARIANTS:
        if vlabel == "A":
            continue
        slotted = all_slotted_by_variant[vlabel]
        sq      = _slot_quality(slotted)
        s1      = sq.get(1, {})
        s2      = sq.get(2, {})
        s3      = sq.get(3, {})
        r1 = s1.get("avg_r", 0.0)
        r2 = s2.get("avg_r")
        r3 = s3.get("avg_r")
        wr1 = s1.get("win_rate", 0.0) * 100
        wr2 = (s2.get("win_rate", 0.0) * 100) if s2 else None
        wr3 = (s3.get("win_rate", 0.0) * 100) if s3 else None

        if r2 is not None:
            diff2 = r2 - r1
            qual2 = "better" if diff2 > 0.05 else ("worse" if diff2 < -0.05 else "similar")
        else:
            qual2 = "no data"
            diff2 = None

        L(f"**Variant {vlabel} (cap={cap})**")
        L(f"- Slot #1: {s1.get('count',0)} trades, AvgR={r1:+.3f}R, WR={wr1:.0f}%")
        if r2 is not None:
            L(f"- Slot #2: {s2.get('count',0)} trades, AvgR={r2:+.3f}R, "
              f"WR={wr2:.0f}% → **{qual2} than slot #1** "
              f"(ΔR={diff2:+.3f}R)")
        else:
            L(f"- Slot #2: no trades")
        if r3 is not None and s3:
            diff3 = r3 - r1
            qual3 = "better" if diff3 > 0.05 else ("worse" if diff3 < -0.05 else "similar")
            L(f"- Slot #3: {s3.get('count',0)} trades, AvgR={r3:+.3f}R, "
              f"WR={wr3:.0f}% → **{qual3} than slot #1** "
              f"(ΔR={diff3:+.3f}R)")
        L("")

    # ── 4. Weeks with blocked additional setups ───────────────────────────
    L("## 4. Weeks With Additional Qualifying Setups Blocked")
    L("")
    L("A WEEKLY_TRADE_LIMIT block in the gap_log means a setup passed ALL other")
    L("gates (session, HTF, confidence, zone, RR, stop-size) but was blocked")
    L("purely by the weekly cap.  These are the highest-quality blocked trades.")
    L("")
    L("| Window | A: wks w/ 2nd blocked | B: wks w/ 3rd blocked |")
    L("|--------|:---------------------:|:---------------------:|")

    total_a_blocks = 0
    total_b_blocks = 0
    for win_name, _, _ in WINDOWS:
        r_a = results[win_name]["A"]
        r_b = results[win_name]["B"]
        n_a = _count_weeks_with_cap_blocks(r_a.gap_log, min_blocks=1)
        n_b = _count_weeks_with_cap_blocks(r_b.gap_log, min_blocks=1)
        total_a_blocks += n_a
        total_b_blocks += n_b
        L(f"| {win_name} | {n_a} | {n_b} |")
    L(f"| **Total** | **{total_a_blocks}** | **{total_b_blocks}** |")
    L("")

    # ── 5. Detailed additional trade listing (B and C vs A) ───────────────
    L("## 5. Additional Trade Detail (new trades in B and C vs A)")
    L("")

    for vlabel, cap, _ in VARIANTS:
        if vlabel == "A":
            continue
        L(f"### Variant {vlabel} (cap={cap}) — additional trades by window")
        L("")

        any_window = False
        for win_name, _, _ in WINDOWS:
            r_a      = results[win_name]["A"]
            r_x      = results[win_name][vlabel]
            slotted_x = _assign_slots(r_x.trades)
            additional = [t for t in slotted_x if t.get("week_slot", 1) >= 2]
            if not additional:
                continue
            any_window = True

            sumr_x = sum(t.get("r", 0) for t in r_x.trades)
            sumr_a = sum(t.get("r", 0) for t in r_a.trades)
            delta  = sumr_x - sumr_a

            L(f"#### {win_name}  (baseline={sumr_a:+.2f}R → variant={sumr_x:+.2f}R, "
              f"Δ={delta:+.2f}R)")
            L("")
            L("| Slot | Pair | Pattern | Dir | Entry (UTC) | R | MAE | MFE | W/L |")
            L("|:----:|------|---------|-----|-------------|:--:|:---:|:---:|:---:|")

            # Show all trades in order (slot 1 = baseline, slot 2+ = additional)
            for t in slotted_x:
                slot    = t.get("week_slot", 1)
                pair    = t.get("pair", "?")
                pat     = t.get("pattern", "?")
                d       = t.get("direction", "?")
                ts      = _fmt_ts(t.get("entry_ts", ""))
                r_val   = t.get("r", 0.0)
                mae     = t.get("mae_r")
                mfe     = t.get("mfe_r")
                new_tag = " ⭐ NEW" if slot >= 2 else ""
                L(f"| #{slot}{new_tag} | {pair} | {pat} | {d} | {ts} "
                  f"| {r_val:+.2f}R | {_r(mae)} | {_r(mfe)} | {_wl(t)} |")
            L("")

        if not any_window:
            L(f"_No additional trades unlocked in Variant {vlabel}._")
            L("")

    # ── 6. ATR floor check ────────────────────────────────────────────────
    L("## 6. ATR Floor Check (C8 8-pip minimum)")
    L("")
    L("| Window | A violations | B violations | C violations |")
    L("|--------|:-----------:|:-----------:|:-----------:|")
    total_v = {"A": 0, "B": 0, "C": 0}
    for win_name, _, _ in WINDOWS:
        row = f"| {win_name}"
        for vlabel, _, _ in VARIANTS:
            r  = results[win_name][vlabel]
            vv = sum(1 for t in r.trades if (t.get("initial_stop_pips") or 999) < 8.0)
            total_v[vlabel] += vv
            row += f" | {vv} {'✅' if vv == 0 else '⚠️'}"
        L(row + " |")
    row_t = "| **Total**"
    for vlabel, _, _ in VARIANTS:
        vv = total_v[vlabel]
        row_t += f" | **{vv} {'✅' if vv == 0 else '⚠️'}**"
    L(row_t + " |")
    L("")

    # ── 7. Verdict ────────────────────────────────────────────────────────
    L("## 7. Verdict")
    L("")
    L("### Summary comparison")
    L("")
    L("| Variant | Cap | Trades | SumR | vs A |")
    L("|---------|:---:|:------:|:----:|:----:|")
    for vlabel, cap, _ in VARIANTS:
        a    = agg[vlabel]
        ts   = a["trades"]
        s    = a["sumr"]
        vs_a = "—" if vlabel == "A" else f"{s - a_sumr:+.2f}R"
        L(f"| {vlabel} | {cap} | {len(ts)} | {s:+.2f}R | {vs_a} |")
    L("")
    L("### Decision")
    L("")

    for vlabel, cap, _ in VARIANTS:
        if vlabel == "A":
            L(f"**Variant A (baseline)**: {a_sumr:+.2f}R — current production.")
            continue
        a    = agg[vlabel]
        s    = a["sumr"]
        delta = s - a_sumr
        slotted = all_slotted_by_variant[vlabel]
        sq      = _slot_quality(slotted)
        s1_r = sq.get(1, {}).get("avg_r", 0.0)
        s2    = sq.get(2)
        s2_r  = s2.get("avg_r", 0.0) if s2 else None

        if delta > 0.5:
            if s2_r is not None and s2_r >= s1_r - 0.05:
                decision = (f"**Variant {vlabel}: PROMOTE** — SumR improved {delta:+.2f}R "
                            f"and slot-2 quality ({s2_r:+.3f}R) is comparable to slot-1 "
                            f"({s1_r:+.3f}R). Raising cap to {cap} is supported.")
            else:
                decision = (f"**Variant {vlabel}: CONSIDER WITH CAUTION** — "
                            f"SumR improved {delta:+.2f}R but slot-2 quality "
                            f"({s2_r:+.3f}R if s2_r else '—') is worse than slot-1 "
                            f"({s1_r:+.3f}R). Cap increase adds noise.")
        elif -0.5 <= delta <= 0.5:
            decision = (f"**Variant {vlabel}: NEUTRAL** — SumR within ±0.5R of baseline "
                        f"({delta:+.2f}R). No clear benefit to raising cap to {cap}.")
        else:
            decision = (f"**Variant {vlabel}: REJECT** — SumR regressed vs baseline "
                        f"({delta:+.2f}R). Additional trades at cap={cap} destroy value.")
        L(decision)
        L("")

    L("### Final recommendation")
    L("")
    b_delta = agg["B"]["sumr"] - a_sumr
    c_delta = agg["C"]["sumr"] - a_sumr
    if b_delta > 0.5:
        L(f"**Raise cap to 2.** Variant B adds {b_delta:+.2f}R with "
          f"additional trades of comparable quality to baseline slot-1 trades.")
        L("Consider making cap dynamic: cap=2 when regime score ≥ 0.65, cap=1 otherwise.")
    elif c_delta > 0.5 and b_delta <= 0.5:
        L(f"**Consider cap=3 only.** Variant C adds {c_delta:+.2f}R but B is neutral. "
          f"Unusual pattern — review additional trade detail carefully.")
    elif b_delta <= 0 and c_delta <= 0:
        L("**Keep cap=1.** Both B and C degrade SumR. The current weekly cap "
          "is the right discipline: additional trades at this equity level and "
          "pattern selectivity level do not add expected value.")
    else:
        L("**Mixed signal.** Review per-slot quality and cascade displacement "
          "data above before deciding.")

    L("")
    L("_Report generated by `scripts/ablation_weekly_cap.py`._")
    L("_Offline replay only — no live changes, no master merge._")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Weekly cap ablation study")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  WEEKLY CAP ABLATION STUDY")
    print("  Offline replay — no live changes")
    print("═"*60)

    results = run_ablation(verbose=args.verbose)

    print("\n  Generating report…")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(results)
    REPORT_PATH.write_text(report)
    print(f"  Report → {REPORT_PATH}")

    # Belt + suspenders reset
    _sc.MAX_TRADES_PER_WEEK_SMALL    = _ORIG_CAP_SMALL
    _sc.MAX_TRADES_PER_WEEK_STANDARD = _ORIG_CAP_STANDARD

    print("\n✅  Done. Config reset to production defaults.")
    print(f"    MAX_TRADES_PER_WEEK_SMALL    = {_sc.MAX_TRADES_PER_WEEK_SMALL}")
    print(f"    MAX_TRADES_PER_WEEK_STANDARD = {_sc.MAX_TRADES_PER_WEEK_STANDARD}")


if __name__ == "__main__":
    main()
