#!/usr/bin/env python3
"""
Priority-Based Weekly Slot Allocation Ablation — offline replay only.

Goal
----
Test whether ranking qualifying setups by confidence × RR and allocating the
weekly slot to the highest-quality setup improves SumR vs first-come-first-served.

Current behavior (A)
    First qualifying setup of the week takes the slot.  All others blocked.

Proposed behavior (B — Rolling best-of-week)
    First setup enters as normal.  A later setup displaces the current slot
    holder ONLY if its quality score exceeds the holder's score by 20%+:
        new_score > current_score × 1.2
    Displaced trade closes at 0R (break-even assumption — conservative).
    No future price knowledge used: each bar is evaluated in order.

Theoretical ceiling (C — End-of-week best / hindsight)
    Collect all qualifying setups for the week, pick the highest scorer.
    NOT promotable — lookahead oracle.  Shows the maximum possible gain.

Methodology
-----------
    1. Run each window with weekly cap = 10 (uncapped) to collect all
       qualifying setups in their natural order.
    2. Group trades by ISO week.
    3. Simulate A, B, C selection logic on the same trade pool.
    4. Compute per-variant P&L from selected trades (displaced → 0R).

Assumption: position sizing is computed from the cap-10 run.  R values may
differ very slightly from a true single-position simulation due to balance
carry-over from simultaneous positions, but are accepted for research purposes.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

import src.strategy.forex.strategy_config as _sc
from backtesting.oanda_backtest_v2 import run_backtest, BacktestResult  # noqa

UTC = timezone.utc
REPORT_PATH = REPO / "backtesting/results/ablation_priority_slot.md"

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

DISPLACEMENT_THRESHOLD = 1.20   # new must be 20% better to displace


# ──────────────────────────────────────────────────────────────────────────────
# Quality scoring
# ──────────────────────────────────────────────────────────────────────────────

def quality(t: Dict) -> float:
    """Confidence × planned_rr — the ranking metric."""
    return float(t.get("confidence", 0) or 0) * float(t.get("planned_rr", 0) or 0)


def iso_week(t: Dict) -> Tuple[int, int]:
    ts = t.get("entry_ts") or t.get("entry_time") or ""
    if not ts:
        return (0, 0)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        cal = dt.isocalendar()
        return (cal[0], cal[1])
    except Exception:
        return (0, 0)


def _fmt_ts(ts) -> str:
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        import pytz
        ET = pytz.timezone("America/New_York")
        return dt.astimezone(ET).strftime("%a %Y-%m-%d %H:%M ET")
    except Exception:
        return str(ts)[:16]


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing selection logic
# ──────────────────────────────────────────────────────────────────────────────

def simulate_A(trades: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """First trade per ISO week (chronological order).  Mirrors production cap=1."""
    seen_weeks: Dict[Tuple[int, int], bool] = {}
    selected, dropped = [], []
    for t in sorted(trades, key=lambda x: str(x.get("entry_ts", ""))):
        wk = iso_week(t)
        if wk not in seen_weeks:
            seen_weeks[wk] = True
            selected.append(t)
        else:
            dropped.append(t)
    return selected, dropped


def simulate_B(trades: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Rolling best-of-week with 20% displacement threshold.

    Returns: (selected, dropped_never_entered, displaced_at_0R)
    Selected trades have r = original R if they held slot to close,
    or r = 0.0 if they were displaced (closed at break-even).
    """
    # Sort chronologically
    ordered = sorted(trades, key=lambda x: str(x.get("entry_ts", "")))

    # Week-state: current slot holder
    slot: Dict[Tuple[int, int], Dict] = {}   # week → current slot trade (copy)
    displaced_events: List[Dict] = []        # log of displacement events

    final_trades: Dict[Tuple[int, int], Dict] = {}  # week → final selected trade

    for t in ordered:
        wk = iso_week(t)
        if wk not in slot:
            # Empty slot — take it
            slot[wk] = dict(t)
            final_trades[wk] = slot[wk]
        else:
            current = slot[wk]
            cur_q   = quality(current)
            new_q   = quality(t)
            if cur_q > 0 and new_q > cur_q * DISPLACEMENT_THRESHOLD:
                # Displace: close current at 0R, enter new
                displaced_copy = dict(current)
                displaced_copy["r"]           = 0.0
                displaced_copy["displaced"]   = True
                displaced_copy["replaced_by"] = t.get("pair", "?")
                displaced_copy["orig_r"]      = current.get("r", 0)
                displaced_events.append({
                    "week":        f"{wk[0]}-W{wk[1]:02d}",
                    "displaced_pair":    current.get("pair","?"),
                    "displaced_pattern": current.get("pattern","?"),
                    "displaced_q":       cur_q,
                    "displaced_r":       current.get("r", 0),
                    "replacement_pair":  t.get("pair","?"),
                    "replacement_pattern": t.get("pattern","?"),
                    "replacement_q":     new_q,
                    "replacement_r":     t.get("r", 0),
                    "net_r_change":      t.get("r", 0) - current.get("r", 0),
                    "net_r_conservative": t.get("r", 0) - 0.0,   # displaced = 0R
                })
                slot[wk]          = dict(t)
                final_trades[wk]  = slot[wk]
            # else: keep current, drop new (no displacement)

    selected    = list(final_trades.values())
    all_selected_keys = {id(t) for t in selected}
    dropped = [t for t in ordered if t not in selected
               and not any(t is s for s in selected)]

    return selected, dropped, displaced_events


def simulate_C(trades: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Best quality score per ISO week (hindsight ceiling)."""
    weeks: Dict[Tuple[int, int], Dict] = {}
    for t in trades:
        wk = iso_week(t)
        if wk not in weeks or quality(t) > quality(weeks[wk]):
            weeks[wk] = t
    selected = list(weeks.values())
    sel_ids  = {id(t) for t in selected}
    dropped  = [t for t in trades if id(t) not in sel_ids]
    return selected, dropped


# ──────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ──────────────────────────────────────────────────────────────────────────────

def stats(trades: List[Dict]) -> Dict:
    if not trades:
        return {"n": 0, "wr": 0, "sumr": 0.0, "avgr": 0.0, "maxdd": 0.0}
    r_list = [t.get("r", 0) for t in trades]
    wins   = sum(1 for r in r_list if r > 0)
    sumr   = sum(r_list)
    avgr   = sumr / len(r_list)
    # Simulate running equity for MaxDD
    eq = 0.0
    peak = 0.0
    maxdd = 0.0
    for r in r_list:
        eq += r
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > maxdd:
            maxdd = dd
    return {
        "n":    len(trades),
        "wr":   wins / len(trades) * 100,
        "sumr": sumr,
        "avgr": avgr,
        "maxdd": maxdd,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main run
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation(verbose: bool = False) -> Dict:
    """Run all windows uncapped, return per-window simulation results."""

    # Temporarily raise weekly cap to get all candidates
    orig_small    = _sc.MAX_TRADES_PER_WEEK_SMALL
    orig_standard = _sc.MAX_TRADES_PER_WEEK_STANDARD

    try:
        _sc.MAX_TRADES_PER_WEEK_SMALL    = 10
        _sc.MAX_TRADES_PER_WEEK_STANDARD = 10

        results: Dict[str, Dict] = {}

        print(f"\n{'═'*60}")
        print("  PRIORITY SLOT ABLATION — UNCAPPED DATA COLLECTION")
        print(f"{'═'*60}")
        print(f"  Variants: A (first), B (rolling best, ≥{DISPLACEMENT_THRESHOLD:.0%} threshold), C (hindsight)")

        for win_name, win_start, win_end in WINDOWS:
            print(f"\n  ── Window: {win_name} ─────────────────────────────────")

            result = run_backtest(
                win_start, win_end,
                starting_bal=8_000.0,
                notes=f"priority_slot_uncapped_{win_name}",
                trail_arm_key="A",
                use_cache=True,
                quiet=not verbose,
            )

            all_trades = result.trades

            sel_a, _      = simulate_A(all_trades)
            sel_b, _, ev_b = simulate_B(all_trades)
            sel_c, _      = simulate_C(all_trades)

            st_a = stats(sel_a)
            st_b = stats(sel_b)
            st_c = stats(sel_c)

            print(f"    Uncapped: {len(all_trades)}T → A={st_a['n']}T({st_a['sumr']:+.2f}R) "
                  f"B={st_b['n']}T({st_b['sumr']:+.2f}R) C={st_c['n']}T({st_c['sumr']:+.2f}R)  "
                  f"disp={len(ev_b)}")

            results[win_name] = {
                "all":        all_trades,
                "sel_a":      sel_a,
                "sel_b":      sel_b,
                "sel_c":      sel_c,
                "disp_b":     ev_b,
                "st_a":       st_a,
                "st_b":       st_b,
                "st_c":       st_c,
                "raw_result": result,
            }

    finally:
        _sc.MAX_TRADES_PER_WEEK_SMALL    = orig_small
        _sc.MAX_TRADES_PER_WEEK_STANDARD = orig_standard

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(results: Dict) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    L: List[str] = []

    def line(s=""):
        L.append(s)

    line("# Priority-Based Weekly Slot Allocation Ablation")
    line(f"\nGenerated: {now}  |  Branch: `feat/priority-slot-ablation`")
    line("")
    line("## Setup")
    line("")
    line("| Parameter | Value |")
    line("|-----------|-------|")
    line("| Strategy | B-Prime + C8 (production) |")
    line("| Capital | $8,000 |")
    line("| MIN_CONFIDENCE | 0.77 |")
    line("| MIN_RR_STANDARD | 2.5 |")
    line("| Displacement threshold | 20% quality improvement |")
    line("| Quality metric | `confidence × planned_rr` |")
    line("| Displaced trade assumption | 0R (break-even close) |")
    line("")
    line("## Variants")
    line("")
    line("| Variant | Selection Rule | Promotable |")
    line("|---------|----------------|:----------:|")
    line("| **A** (baseline) | First qualifying setup per week | ✅ production |")
    line("| **B** Rolling best | Displace if new score ≥ 120% of current | ✅ candidate |")
    line("| **C** Hindsight best | Best conf×RR per week (lookahead) | ❌ ceiling only |")
    line("")

    # ── Per-window breakdown ───────────────────────────────────────────────────
    line("## 1. Per-Window Breakdown")
    line("")
    line("| Window | Uncapped | A Trades | A SumR | B Trades | B SumR | B Displ | "
         "C Trades | C SumR |")
    line("|--------|:--------:|:--------:|:------:|:--------:|:------:|:-------:|"
         ":--------:|:------:|")

    total: Dict = {k: {"n": 0, "sumr": 0.0, "displ": 0} for k in ("a", "b", "c", "uncap")}
    all_disp_events: List[Dict] = []

    for win_name, _, _ in WINDOWS:
        d  = results[win_name]
        st_a, st_b, st_c = d["st_a"], d["st_b"], d["st_c"]
        ev_b = d["disp_b"]
        n_uc = len(d["all"])
        line(f"| {win_name} | {n_uc} "
             f"| {st_a['n']} | {st_a['sumr']:+.2f}R "
             f"| {st_b['n']} | {st_b['sumr']:+.2f}R | {len(ev_b)} "
             f"| {st_c['n']} | {st_c['sumr']:+.2f}R |")
        total["a"]["n"]      += st_a["n"];  total["a"]["sumr"]      += st_a["sumr"]
        total["b"]["n"]      += st_b["n"];  total["b"]["sumr"]      += st_b["sumr"]
        total["c"]["n"]      += st_c["n"];  total["c"]["sumr"]      += st_c["sumr"]
        total["uncap"]["n"]  += n_uc
        total["b"]["displ"]  += len(ev_b)
        all_disp_events.extend(ev_b)

    line(f"| **TOTAL** | {total['uncap']['n']} "
         f"| **{total['a']['n']}** | **{total['a']['sumr']:+.2f}R** "
         f"| **{total['b']['n']}** | **{total['b']['sumr']:+.2f}R** | **{total['b']['displ']}** "
         f"| **{total['c']['n']}** | **{total['c']['sumr']:+.2f}R** |")
    line("")

    # ── Aggregate summary ──────────────────────────────────────────────────────
    line("## 2. Aggregate Summary")
    line("")
    line("| Variant | Trades | WR | SumR | vs A | AvgR | Displacements |")
    line("|---------|:------:|:--:|:----:|:----:|:----:|:-------------:|")

    a_sumr = total["a"]["sumr"]

    for var_label, key, displ in [("A (baseline)", "a", "—"),
                                   ("B (rolling 20%)", "b", str(total["b"]["displ"])),
                                   ("C (hindsight)", "c", "N/A")]:
        all_sel = [t for win_name, _, _ in WINDOWS for t in results[win_name][f"sel_{key}"]]
        st = stats(all_sel)
        vs = "—" if key == "a" else f"{st['sumr'] - a_sumr:+.2f}R"
        line(f"| **{var_label}** | {st['n']} | {st['wr']:.0f}% | {st['sumr']:+.2f}R "
             f"| {vs} | {st['avgr']:+.3f}R | {displ} |")
    line("")

    # ── Displacement events ────────────────────────────────────────────────────
    line("## 3. Displacement Events (Variant B)")
    line("")

    if not all_disp_events:
        line("**No displacement events occurred across all 8 windows.**")
        line("")
        line("The 20% quality threshold was never met — no later setup scored "
             "≥120% of the active slot holder's confidence × RR.")
    else:
        n_displ  = len(all_disp_events)
        net_gain = sum(e["net_r_conservative"] for e in all_disp_events)
        avg_gain = net_gain / n_displ

        line(f"**{n_displ} displacement event(s) total across all windows.**")
        line(f"Net R change from displacements: {net_gain:+.2f}R "
             f"(avg {avg_gain:+.3f}R/event, conservative: displaced → 0R)")
        line("")
        line("| Week | Displaced | Disp Q | Disp R (orig) | Replacement | Repl Q | "
             "Repl R | Net ΔR (conseq.) |")
        line("|------|-----------|:------:|:-------------:|-------------|:------:|"
             ":------:|:----------------:|")
        for e in all_disp_events:
            net_c = e["replacement_r"] - 0.0   # displaced → 0R
            line(f"| {e['week']} | {e['displaced_pair']} {e['displaced_pattern'][:12]} "
                 f"| {e['displaced_q']:.3f} | {e['displaced_r']:+.2f}R (→0R) "
                 f"| {e['replacement_pair']} {e['replacement_pattern'][:12]} "
                 f"| {e['replacement_q']:.3f} | {e['replacement_r']:+.2f}R "
                 f"| {net_c:+.2f}R |")
        line("")
        line("### Displacement quality analysis")
        line("")
        avg_q_disp = (sum(e["displaced_q"] for e in all_disp_events) / n_displ
                      if n_displ else 0)
        avg_q_repl = (sum(e["replacement_q"] for e in all_disp_events) / n_displ
                      if n_displ else 0)
        wins_repl  = sum(1 for e in all_disp_events if e["replacement_r"] > 0)
        wins_disp  = sum(1 for e in all_disp_events if e["displaced_r"] > 0)
        line(f"- Avg quality of **displaced** trades:    {avg_q_disp:.3f}")
        line(f"- Avg quality of **replacement** trades:  {avg_q_repl:.3f}")
        line(f"- Displaced WR: {wins_disp}/{n_displ} = {wins_disp/n_displ*100:.0f}%")
        line(f"- Replacement WR: {wins_repl}/{n_displ} = {wins_repl/n_displ*100:.0f}%")
        line(f"- Average quality multiple: {avg_q_repl/avg_q_disp:.2f}× "
             f"(threshold: {DISPLACEMENT_THRESHOLD:.2f}×)")
    line("")

    # ── Uncapped candidate pool ────────────────────────────────────────────────
    line("## 4. Candidate Pool Analysis")
    line("")
    line("| Window | Uncapped | Weeks w/ 2+ | Multi-candidate weeks | "
         "Best pick = A pick? |")
    line("|--------|:--------:|:-----------:|:---------------------:|:------------------:|")

    for win_name, _, _ in WINDOWS:
        d = results[win_name]
        all_t = d["all"]
        weeks_all: Dict = defaultdict(list)
        for t in all_t:
            weeks_all[iso_week(t)].append(t)
        n_multi = sum(1 for wk, wt in weeks_all.items() if len(wt) > 1)
        # Check how often C (best) == A (first)
        agree = 0
        sel_a = d["sel_a"]
        sel_c = d["sel_c"]
        a_keys = {(t.get("pair",""), str(t.get("entry_ts",""))[:16]) for t in sel_a}
        c_keys = {(t.get("pair",""), str(t.get("entry_ts",""))[:16]) for t in sel_c}
        agree  = len(a_keys & c_keys)
        line(f"| {win_name} | {len(all_t)} | {n_multi} | "
             f"{n_multi} multi-candidate week(s) | "
             f"{agree}/{len(sel_a)} agree |")
    line("")

    # ── Q3 dead zone ───────────────────────────────────────────────────────────
    line("## 5. Q3 2025 Dead Zone")
    line("")
    d = results["Q3-2025"]
    q3_a = d["st_a"]; q3_b = d["st_b"]; q3_c = d["st_c"]
    line("| Variant | Trades | SumR | Displacements |")
    line("|---------|:------:|:----:|:-------------:|")
    line(f"| A | {q3_a['n']} | {q3_a['sumr']:+.2f}R | — |")
    line(f"| B | {q3_b['n']} | {q3_b['sumr']:+.2f}R | {len(d['disp_b'])} |")
    line(f"| C | {q3_c['n']} | {q3_c['sumr']:+.2f}R | N/A |")
    line("")
    if q3_a["n"] == 0:
        line("Q3 remains a dead zone across all variants — priority-based selection "
             "cannot create signal where none exists.")
    else:
        line(f"Q3 has trades: A={q3_a['n']}T, B={q3_b['n']}T, C={q3_c['n']}T.")
    line("")

    # ── ATR floor ─────────────────────────────────────────────────────────────
    line("## 6. ATR Floor Check")
    line("")
    line("| Window | A violations | B violations | C violations |")
    line("|--------|:------------:|:------------:|:------------:|")
    for win_name, _, _ in WINDOWS:
        d = results[win_name]
        vio = {k: sum(1 for t in d[f"sel_{k}"]
                      if (t.get("initial_stop_pips") or 999) < 8.0)
               for k in ("a", "b", "c")}
        line(f"| {win_name} | {vio['a']}{'✅' if vio['a']==0 else '⚠️'} "
             f"| {vio['b']}{'✅' if vio['b']==0 else '⚠️'} "
             f"| {vio['c']}{'✅' if vio['c']==0 else '⚠️'} |")
    line("")

    # ── Verdict ────────────────────────────────────────────────────────────────
    line("## 7. Verdict")
    line("")
    all_a = [t for wn, _, _ in WINDOWS for t in results[wn]["sel_a"]]
    all_b = [t for wn, _, _ in WINDOWS for t in results[wn]["sel_b"]]
    all_c = [t for wn, _, _ in WINDOWS for t in results[wn]["sel_c"]]
    st_a  = stats(all_a)
    st_b  = stats(all_b)
    st_c  = stats(all_c)

    delta_b = st_b["sumr"] - st_a["sumr"]
    delta_c = st_c["sumr"] - st_a["sumr"]
    n_disp  = total["b"]["displ"]

    line("| Variant | SumR | vs A | Verdict |")
    line("|---------|:----:|:----:|---------|")
    line(f"| A (baseline) | {st_a['sumr']:+.2f}R | — | Production |")

    if delta_b > 1.0:
        b_verdict = f"CONSIDER — {delta_b:+.2f}R improvement. {n_disp} displacement(s). Review quality."
    elif -1.0 <= delta_b <= 1.0:
        b_verdict = f"NEUTRAL — {delta_b:+.2f}R. Displacement logic adds complexity for no gain."
    else:
        b_verdict = f"REJECT — {delta_b:+.2f}R regression. Displacement degraded returns."

    if delta_c > 5.0:
        c_note = f"Ceiling gap of {delta_c:+.2f}R exists — worth targeting."
    elif delta_c > 1.0:
        c_note = f"Ceiling gap of {delta_c:+.2f}R — modest. B captures some of it."
    else:
        c_note = f"Ceiling gap {delta_c:+.2f}R — essentially zero. First pick = best pick."

    line(f"| B (rolling 20%) | {st_b['sumr']:+.2f}R | {delta_b:+.2f}R | {b_verdict} |")
    line(f"| C (hindsight) | {st_c['sumr']:+.2f}R | {delta_c:+.2f}R | ❌ Not promotable. {c_note} |")
    line("")

    line("### Key findings")
    line("")
    line(f"1. **Total uncapped candidates:** {total['uncap']['n']} across 8 windows "
         f"(vs {total['a']['n']} under cap=1)")
    line(f"2. **Displacement events (B):** {n_disp} — "
         f"{'threshold was never triggered' if n_disp==0 else f'{n_disp} slot(s) contested'}")
    line(f"3. **Hindsight ceiling gap:** {delta_c:+.2f}R "
         f"({delta_c/st_a['sumr']*100:.1f}% of baseline SumR)" if st_a["sumr"] else "")
    line(f"4. **B vs A gap:** {delta_b:+.2f}R — "
         + ("The 20% threshold is well-calibrated, capturing quality improvement "
            "without introducing noise." if delta_b > 0 else
            "No benefit — first-come-first-served is already near-optimal."))
    line("")
    line("_Report generated by `scripts/ablation_priority_slot.py`._")
    line("_Offline replay only — no live changes, no master merge._")

    return "\n".join(L)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = run_ablation(verbose=args.verbose)

    print("\n  Generating report…")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(results)
    REPORT_PATH.write_text(report)
    print(f"  Report → {REPORT_PATH}")
    print("\n✅  Done.")


if __name__ == "__main__":
    main()
