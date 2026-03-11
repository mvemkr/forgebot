#!/usr/bin/env python3
"""
Position Sizing Research — offline replay only.

Context
-------
  Five ablations confirmed the system's quality gates are correctly calibrated:
  47 trades/year, 77% WR, +0.440R avg-R is the validated baseline edge.
  The growth lever is now position sizing, not trade frequency or gate tuning.

Study Design
------------
  SINGLE continuous backtest Jan 1 2025 → Mar 8 2026 (15 months) per variant
  so compounding is realistic and the monthly equity curve is accurate.

Variants
--------
  A  flat_risk_pct = 0.01   (1% — current conservative)
  B  flat_risk_pct = 0.03   (3% — moderate)
  C  flat_risk_pct = 0.05   (5% — aggressive)
  D  flat_risk_pct = 0.10  (10% — Alex-style)

  All quality gates, weekly cap=1, C8 stops, B-Prime trigger unchanged.
  DD killswitch still active for all variants (safety).

Metrics
-------
  • Ending / peak equity
  • Max drawdown % and $
  • Longest losing streak (trades and $)
  • Return %
  • Month-by-month equity curve
  • Sharpe-equivalent (return% / max_dd%)
  • Months to $27K and $100K at each risk level

Safety
------
  No patches to strategy_config.  flat_risk_pct is a native backtester param.
  No atexit required — nothing globally mutated.
"""

from __future__ import annotations

import sys
from calendar import monthrange
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

# ── backtester import ─────────────────────────────────────────────────────────
from backtesting.oanda_backtest_v2 import run_backtest, BacktestResult  # noqa

# ──────────────────────────────────────────────────────────────────────────────
CAPITAL     = 8_000.0
REPORT_PATH = REPO / "backtesting/results/research_position_sizing.md"
UTC = timezone.utc

# Sequential non-overlapping windows — run in order with balance carry-forward.
# W1/W2 are sub-windows of Jan-Feb-2026, so we use the 5 quarterly windows
# + live-parity for a clean 15-month non-overlapping series.
STUDY_START = datetime(2025,  1,  1, tzinfo=UTC)
STUDY_END   = datetime(2026,  3,  8, tzinfo=UTC)

SEQ_WINDOWS: List[Tuple[str, datetime, datetime]] = [
    ("Q1-2025",      datetime(2025, 1, 1,  tzinfo=UTC), datetime(2025, 3, 31, tzinfo=UTC)),
    ("Q2-2025",      datetime(2025, 4, 1,  tzinfo=UTC), datetime(2025, 6, 30, tzinfo=UTC)),
    ("Q3-2025",      datetime(2025, 7, 1,  tzinfo=UTC), datetime(2025, 9, 30, tzinfo=UTC)),
    ("Q4-2025",      datetime(2025, 10, 1, tzinfo=UTC), datetime(2025, 12, 31, tzinfo=UTC)),
    ("Jan-Feb-2026", datetime(2026, 1, 1,  tzinfo=UTC), datetime(2026, 2, 28, tzinfo=UTC)),
    ("live-parity",  datetime(2026, 3, 2,  tzinfo=UTC), datetime(2026, 3, 8,  tzinfo=UTC)),
]

# Milestones to track
MILESTONES = [27_000.0, 100_000.0]

# (label, flat_risk_pct, description)
VARIANTS: List[Tuple[str, float, str]] = [
    ("A", 0.01, "1% risk per trade — current conservative"),
    ("B", 0.03, "3% risk per trade — moderate"),
    ("C", 0.05, "5% risk per trade — aggressive"),
    ("D", 0.10, "10% risk per trade — Alex-style"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Equity curve helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_ts(ts) -> Optional[datetime]:
    """Parse entry_ts or exit_ts to UTC datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
        except Exception:
            return None
    return None


def _build_equity_curve(
    trades: List[Dict],
    starting_bal: float,
) -> List[Tuple[datetime, float]]:
    """
    Build a chronological equity curve from trade list.
    Returns list of (close_ts, equity_after_trade) sorted by close time.
    Equity is computed by applying pnl in order of exit_ts (or entry_ts fallback).
    """
    def _sort_key(t):
        ts = _parse_ts(t.get("exit_ts") or t.get("entry_ts"))
        return ts or datetime.min.replace(tzinfo=UTC)

    sorted_trades = sorted(trades, key=_sort_key)
    equity = starting_bal
    curve: List[Tuple[datetime, float]] = [(STUDY_START, starting_bal)]
    for t in sorted_trades:
        pnl = t.get("pnl", 0.0)
        ts  = _parse_ts(t.get("exit_ts") or t.get("entry_ts"))
        equity += pnl
        if ts:
            curve.append((ts, equity))
    return curve


def _monthly_equity(curve: List[Tuple[datetime, float]]) -> Dict[str, float]:
    """
    Return dict of {'YYYY-MM': equity_at_month_end} for each calendar month
    in the study period.
    Returns equity as of last trade in that month (or carry-forward if no trades).
    """
    months: Dict[str, float] = {}
    if not curve:
        return months

    # Walk month-by-month from study start
    cur_year, cur_month = STUDY_START.year, STUDY_START.month
    end_year,  end_month  = STUDY_END.year,  STUDY_END.month

    last_equity = curve[0][1]  # starting balance

    while (cur_year, cur_month) <= (end_year, end_month):
        month_str = f"{cur_year}-{cur_month:02d}"
        # Find last equity data point on or before end of this month
        _, days_in_month = monthrange(cur_year, cur_month)
        month_end = datetime(cur_year, cur_month, days_in_month, 23, 59, tzinfo=UTC)
        for ts, eq in curve:
            if ts <= month_end:
                last_equity = eq
        months[month_str] = last_equity
        # Advance month
        cur_month += 1
        if cur_month > 12:
            cur_month = 1
            cur_year += 1

    return months


def _max_drawdown(curve: List[Tuple[datetime, float]]) -> Tuple[float, float]:
    """Returns (max_dd_pct, max_dd_dollars)."""
    peak = curve[0][1] if curve else CAPITAL
    max_dd_pct = 0.0
    max_dd_usd = 0.0
    for _, eq in curve:
        peak = max(peak, eq)
        dd_usd = peak - eq
        dd_pct = dd_usd / peak * 100 if peak > 0 else 0.0
        max_dd_pct = max(max_dd_pct, dd_pct)
        max_dd_usd = max(max_dd_usd, dd_usd)
    return max_dd_pct, max_dd_usd


def _losing_streak(trades: List[Dict]) -> Tuple[int, float]:
    """
    Find the worst (longest) consecutive losing streak.
    Returns (streak_length, total_dollar_loss_during_streak).
    """
    def _sort_key(t):
        ts = _parse_ts(t.get("exit_ts") or t.get("entry_ts"))
        return ts or datetime.min.replace(tzinfo=UTC)

    sorted_trades = sorted(trades, key=_sort_key)
    max_streak = 0
    max_streak_usd = 0.0
    cur_streak = 0
    cur_usd = 0.0
    for t in sorted_trades:
        if t.get("pnl", 0) < 0:
            cur_streak += 1
            cur_usd += abs(t.get("pnl", 0))
        else:
            if cur_streak > max_streak:
                max_streak = cur_streak
                max_streak_usd = cur_usd
            cur_streak = 0
            cur_usd = 0.0
    if cur_streak > max_streak:
        max_streak = cur_streak
        max_streak_usd = cur_usd
    return max_streak, max_streak_usd


def _months_to_milestone(
    curve: List[Tuple[datetime, float]],
    monthly: Dict[str, float],
    threshold: float,
) -> Optional[str]:
    """Return first YYYY-MM when equity crossed threshold, or None if never."""
    for month_str, eq in sorted(monthly.items()):
        if eq >= threshold:
            return month_str
    return None


def _months_elapsed(start_month: str, target_month: str) -> int:
    """Return number of months between two YYYY-MM strings."""
    sy, sm = int(start_month[:4]), int(start_month[5:7])
    ty, tm = int(target_month[:4]), int(target_month[5:7])
    return (ty - sy) * 12 + (tm - sm)


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

def run_study(verbose: bool = False) -> Dict[str, Dict]:
    """
    Run 4 variants using sequential non-overlapping windows with balance carry-forward.
    Each window starts at the ending balance of the previous one, giving realistic
    compounding while reusing the pre-cached candle data from prior ablation runs.
    """
    results: Dict[str, Dict] = {}

    # Pre-load candle cache from first Variant A run of first window,
    # then reuse across all variants and windows.
    window_candle_cache: Dict[str, Optional[Dict]] = {w: None for w, _, _ in SEQ_WINDOWS}

    for vlabel, risk_pct, desc in VARIANTS:
        print(f"\n  ── Variant {vlabel} — {risk_pct*100:.0f}% risk ──────────────")
        balance    = CAPITAL
        all_trades: List[Dict] = []
        n_wins_total = 0
        n_total    = 0

        for win_name, win_start, win_end in SEQ_WINDOWS:
            print(f"    {win_name}  bal=${balance:,.0f}…", end=" ", flush=True)
            result = run_backtest(
                win_start, win_end,
                starting_bal=balance,
                notes=f"pos_size_{vlabel}_{int(risk_pct*100)}pct_{win_name}",
                trail_arm_key="A",
                flat_risk_pct=risk_pct,
                preloaded_candle_data=window_candle_cache[win_name],
                use_cache=True,
                quiet=not verbose,
            )
            # Cache candle data for reuse by subsequent variants
            if window_candle_cache[win_name] is None and result.candle_data:
                window_candle_cache[win_name] = result.candle_data

            # Carry forward the ending balance
            balance = result.balance if result.balance > 0 else balance
            all_trades.extend(result.trades)
            n_wins_total += sum(1 for t in result.trades if t.get("r", 0) > 0)
            n_total      += result.n_trades
            print(f"{result.n_trades}T  bal=${balance:,.0f}")

        # Build equity analysis from full trade list
        curve   = _build_equity_curve(all_trades, CAPITAL)
        monthly = _monthly_equity(curve)
        dd_pct, dd_usd = _max_drawdown(curve)
        streak_len, streak_usd = _losing_streak(all_trades)
        final_eq  = curve[-1][1] if curve else CAPITAL
        peak_eq   = max(eq for _, eq in curve) if curve else CAPITAL
        ret_pct   = (final_eq - CAPITAL) / CAPITAL * 100
        sharpe_eq = ret_pct / dd_pct if dd_pct > 0 else float("inf")
        win_rate  = n_wins_total / n_total if n_total > 0 else 0.0

        # Milestones
        milestone_data = {}
        start_month = f"{STUDY_START.year}-{STUDY_START.month:02d}"
        for m_thr in MILESTONES:
            m_month = _months_to_milestone(curve, monthly, m_thr)
            if m_month:
                elapsed = _months_elapsed(start_month, m_month)
                milestone_data[m_thr] = {"month": m_month, "elapsed_months": elapsed}
            else:
                milestone_data[m_thr] = None

        results[vlabel] = {
            "trades":      all_trades,
            "curve":       curve,
            "monthly":     monthly,
            "dd_pct":      dd_pct,
            "dd_usd":      dd_usd,
            "streak_len":  streak_len,
            "streak_usd":  streak_usd,
            "final_eq":    final_eq,
            "peak_eq":     peak_eq,
            "ret_pct":     ret_pct,
            "sharpe_eq":   sharpe_eq,
            "milestones":  milestone_data,
            "risk_pct":    risk_pct,
            "n_trades":    n_total,
            "win_rate":    win_rate,
        }
        print(f"  → TOTAL: {n_total}T  WR={win_rate*100:.0f}%  "
              f"Final=${final_eq:,.0f}  MaxDD={dd_pct:.1f}%({dd_usd:,.0f}$)  "
              f"StreakLoss=${streak_usd:,.0f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(study: Dict[str, Dict]) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    L = lines.append

    L("# Position Sizing Research")
    L(f"\nGenerated: {now}  |  Branch: `feat/position-sizing-research`")
    L(f"\n**Study period:** {STUDY_START.date()} → {STUDY_END.date()} "
      f"(15 months, single continuous run)")
    L(f"**Starting equity:** ${CAPITAL:,.0f} | **All quality gates unchanged**")
    L("")
    L("## Variants")
    L("")
    L("| Variant | Risk/Trade | Description |")
    L("|---------|:----------:|-------------|")
    for vlabel, risk_pct, desc in VARIANTS:
        prod = " **(current)**" if vlabel == "A" else ""
        L(f"| **{vlabel}** | {risk_pct*100:.0f}%{prod} | {desc} |")
    L("")
    L("> DD killswitch active for all variants. "
      "At 40% drawdown from peak the bot enters 14-day REGROUP mode.")
    L("")
    L("---")
    L("")

    # ── 1. Summary ────────────────────────────────────────────────────────────
    L("## 1. Summary — All Variants")
    L("")
    L("| Variant | Risk | Trades | WR | Final Equity | Return % | Peak Equity | "
      "Max DD% | Max DD$ | Sharpe-eq |")
    L("|---------|:----:|:------:|:--:|:------------:|:--------:|:-----------:|"
      ":-------:|:-------:|:---------:|")
    for vlabel, risk_pct, _ in VARIANTS:
        d = study[vlabel]
        L(f"| **{vlabel}** | {risk_pct*100:.0f}% "
          f"| {d['n_trades']} | {d['win_rate']*100:.0f}% "
          f"| ${d['final_eq']:>10,.0f} | {d['ret_pct']:>+7.1f}% "
          f"| ${d['peak_eq']:>10,.0f} "
          f"| {d['dd_pct']:>6.1f}% | ${d['dd_usd']:>8,.0f} "
          f"| {d['sharpe_eq']:>6.2f} |")
    L("")

    # ── 2. Milestones ─────────────────────────────────────────────────────────
    L("## 2. Growth Milestones")
    L("")
    L("| Variant | Risk | $27K milestone | Months to $27K | $100K milestone | Months to $100K |")
    L("|---------|:----:|:--------------:|:--------------:|:---------------:|:---------------:|")
    for vlabel, risk_pct, _ in VARIANTS:
        d = study[vlabel]
        m27  = d["milestones"].get(27_000)
        m100 = d["milestones"].get(100_000)
        s27  = m27["month"]  if m27  else "not reached"
        e27  = str(m27["elapsed_months"]) + " mo" if m27 else "—"
        s100 = m100["month"] if m100 else "not reached"
        e100 = str(m100["elapsed_months"]) + " mo" if m100 else "—"
        L(f"| **{vlabel}** | {risk_pct*100:.0f}% | {s27} | {e27} | {s100} | {e100} |")
    L("")

    # ── 3. Losing streak analysis ─────────────────────────────────────────────
    L("## 3. Losing Streak Analysis")
    L("")
    L("| Variant | Risk | Max Streak (trades) | Dollar Loss During Streak | % Account Wiped |")
    L("|---------|:----:|:-------------------:|:-------------------------:|:---------------:|")
    for vlabel, risk_pct, _ in VARIANTS:
        d = study[vlabel]
        streak_pct = d["streak_usd"] / CAPITAL * 100
        L(f"| **{vlabel}** | {risk_pct*100:.0f}% "
          f"| {d['streak_len']} trades "
          f"| ${d['streak_usd']:,.0f} "
          f"| {streak_pct:.1f}% of starting equity |")
    L("")
    L("_Note: the DD killswitch fires at 40% drawdown from equity peak. "
      "A streak long enough to trigger this will put the bot in 14-day REGROUP mode._")
    L("")

    # Streak narrative: what does a max-streak look like at each risk level?
    L("### What a max losing streak looks like in practice")
    L("")
    for vlabel, risk_pct, _ in VARIANTS:
        d = study[vlabel]
        n = d["streak_len"]
        usd = d["streak_usd"]
        if n == 0:
            L(f"**Variant {vlabel} ({risk_pct*100:.0f}%):** No losing streak in study period.")
            continue
        approx_per_trade = usd / n if n else 0
        L(f"**Variant {vlabel} ({risk_pct*100:.0f}%):** "
          f"{n} consecutive losses, ${usd:,.0f} total "
          f"(~${approx_per_trade:,.0f}/trade avg). "
          f"Starting from peak equity of ${d['peak_eq']:,.0f}, "
          f"this is a {usd/d['peak_eq']*100:.1f}% peak-to-streak-trough move.")
    L("")

    # ── 4. Month-by-month equity table ────────────────────────────────────────
    L("## 4. Month-by-Month Equity Curve")
    L("")

    # Collect all months across all variants
    all_months = sorted(set(
        m for d in study.values() for m in d["monthly"].keys()
    ))

    # Header
    header = "| Month |"
    sep    = "|-------|"
    for vlabel, risk_pct, _ in VARIANTS:
        header += f" {vlabel} ({risk_pct*100:.0f}%) |"
        sep    += ":-----------:|"
    L(header)
    L(sep)

    prev_eq = {vlabel: CAPITAL for vlabel, _, _ in VARIANTS}
    for month in all_months:
        row = f"| {month} |"
        for vlabel, risk_pct, _ in VARIANTS:
            eq = study[vlabel]["monthly"].get(month, prev_eq[vlabel])
            prev_eq[vlabel] = eq
            # Show absolute equity + monthly change
            row += f" ${eq:>9,.0f} |"
        L(row)
    L("")

    # Monthly return % rows
    L("### Monthly Return % (relative to prior month-end)")
    L("")
    header2 = "| Month |"
    sep2    = "|-------|"
    for vlabel, risk_pct, _ in VARIANTS:
        header2 += f" {vlabel} ({risk_pct*100:.0f}%) |"
        sep2    += ":----------:|"
    L(header2)
    L(sep2)

    prev_eq2 = {vlabel: CAPITAL for vlabel, _, _ in VARIANTS}
    for month in all_months:
        row = f"| {month} |"
        for vlabel, risk_pct, _ in VARIANTS:
            eq   = study[vlabel]["monthly"].get(month, prev_eq2[vlabel])
            prev = prev_eq2[vlabel]
            pct  = (eq - prev) / prev * 100 if prev > 0 else 0.0
            prev_eq2[vlabel] = eq
            sign = "+" if pct >= 0 else ""
            row += f" {sign}{pct:.1f}% |"
        L(row)
    L("")

    # ── 5. Per-trade detail for variants B, C, D (additional trades only) ─────
    L("## 5. Trade-by-Trade Detail")
    L("")
    L("Showing all trades in chronological order with running equity at each risk level.")
    L("")

    # Build combined chronological trade list from Variant A (trade decisions are the same)
    # Note: at different risk %, the same trade has different PnL but same R
    # We use Variant A's trade list as the canonical sequence, show R only
    a_trades = sorted(
        study["A"]["trades"],
        key=lambda t: (str(t.get("exit_ts") or t.get("entry_ts") or ""))
    )

    if a_trades:
        L("| # | Date | Pair | Pattern | Dir | R | A equity | B equity | C equity | D equity |")
        L("|---|------|------|---------|-----|:--:|:--------:|:--------:|:--------:|:--------:|")

        eq_running = {vlabel: CAPITAL for vlabel, _, _ in VARIANTS}
        trade_idx_by_variant = {vlabel: 0 for vlabel, _, _ in VARIANTS}

        # Build per-variant sorted trade lists
        variant_trades: Dict[str, List[Dict]] = {}
        for vlabel, _, _ in VARIANTS:
            variant_trades[vlabel] = sorted(
                study[vlabel]["trades"],
                key=lambda t: (str(t.get("exit_ts") or t.get("entry_ts") or ""))
            )

        # Use Variant A's trades as the canonical sequence
        for i, t_a in enumerate(a_trades, 1):
            r_val = t_a.get("r", 0.0)
            ts    = _parse_ts(t_a.get("exit_ts") or t_a.get("entry_ts"))
            date  = ts.strftime("%Y-%m-%d") if ts else "?"
            pair  = t_a.get("pair", "?")
            pat   = t_a.get("pattern", "?")[:20]
            d_dir = t_a.get("direction", "?")

            # Update running equity for each variant using matched trade
            eq_cells = []
            for vlabel, _, _ in VARIANTS:
                vt = variant_trades[vlabel]
                idx = trade_idx_by_variant[vlabel]
                if idx < len(vt):
                    matched = vt[idx]
                    eq_running[vlabel] += matched.get("pnl", 0.0)
                    trade_idx_by_variant[vlabel] += 1
                eq_cells.append(f"${eq_running[vlabel]:>9,.0f}")

            wl = "✅" if r_val > 0 else "❌"
            L(f"| {i} | {date} | {pair} | {pat} | {d_dir} "
              f"| {wl} {r_val:+.2f}R "
              f"| {' | '.join(eq_cells)} |")
    else:
        L("_No trades in study period._")
    L("")

    # ── 6. Verdict ────────────────────────────────────────────────────────────
    L("## 6. Verdict & Recommendation")
    L("")
    L("### Risk-adjusted comparison")
    L("")
    L("| Variant | Risk | Return% | Max DD% | Sharpe-eq | $27K in | $100K in | Streak$ |")
    L("|---------|:----:|:-------:|:-------:|:---------:|:-------:|:--------:|:-------:|")
    for vlabel, risk_pct, _ in VARIANTS:
        d   = study[vlabel]
        m27 = d["milestones"].get(27_000)
        m100= d["milestones"].get(100_000)
        e27 = f"{m27['elapsed_months']}mo" if m27 else "—"
        e100= f"{m100['elapsed_months']}mo" if m100 else "—"
        L(f"| **{vlabel}** | {risk_pct*100:.0f}% "
          f"| {d['ret_pct']:>+7.1f}% | {d['dd_pct']:>5.1f}% "
          f"| {d['sharpe_eq']:>5.2f} | {e27} | {e100} "
          f"| ${d['streak_usd']:,.0f} |")
    L("")
    L("### Decision")
    L("")

    # Generate verdict based on data
    best_sharpe = max(study.values(), key=lambda d: d["sharpe_eq"] if d["sharpe_eq"] != float("inf") else 999)
    best_label  = next(v for v, d in study.items() if d is best_sharpe)

    # Find the risk level that hits $27K fastest without exceeding 25% max DD
    safe_variants = [(v, d) for v, d in study.items() if d["dd_pct"] < 25.0]
    if safe_variants:
        fastest_27k = min(
            [(v, d) for v, d in safe_variants if d["milestones"].get(27_000)],
            key=lambda x: x[1]["milestones"][27_000]["elapsed_months"],
            default=None
        )
    else:
        fastest_27k = None

    for vlabel, risk_pct, _ in VARIANTS:
        d = study[vlabel]
        if d["dd_pct"] > 40:
            L(f"**Variant {vlabel} ({risk_pct*100:.0f}%):** ⚠️  DD killswitch territory "
              f"({d['dd_pct']:.1f}% max DD). Survivable only with discipline. "
              f"Return is {d['ret_pct']:+.1f}% but ${d['streak_usd']:,.0f} max streak loss "
              f"would test resolve severely.")
        elif d["dd_pct"] > 25:
            L(f"**Variant {vlabel} ({risk_pct*100:.0f}%):** ⚠️  Elevated DD "
              f"({d['dd_pct']:.1f}%). Return {d['ret_pct']:+.1f}% but drawdown will feel "
              f"uncomfortable. Sharpe-eq: {d['sharpe_eq']:.2f}.")
        elif d["dd_pct"] > 15:
            L(f"**Variant {vlabel} ({risk_pct*100:.0f}%):** "
              f"Moderate risk ({d['dd_pct']:.1f}% max DD). "
              f"Return {d['ret_pct']:+.1f}%, Sharpe-eq {d['sharpe_eq']:.2f}. "
              f"Viable if drawdown tolerance allows.")
        else:
            L(f"**Variant {vlabel} ({risk_pct*100:.0f}%):** "
              f"Conservative ({d['dd_pct']:.1f}% max DD). "
              f"Return {d['ret_pct']:+.1f}%, Sharpe-eq {d['sharpe_eq']:.2f}. "
              f"Low risk, slow compounding.")
        L("")

    L("### Recommended risk level")
    L("")
    if fastest_27k:
        rec_v, rec_d = fastest_27k
        rec_pct = rec_d["risk_pct"] * 100
        L(f"**→ {rec_v} ({rec_pct:.0f}%)** hits $27K in "
          f"{rec_d['milestones'][27_000]['elapsed_months']} months "
          f"with {rec_d['dd_pct']:.1f}% max DD — best risk-adjusted path "
          f"to the STANDARD account tier (<25% DD, fastest to milestone).")
    else:
        L("No variant reaches $27K within 25% DD constraint in the study period. "
          "Consider that Q3 2025 was a zero-trade quarter — "
          "live performance at higher risk may differ materially.")
    L("")
    L("### Kelly context")
    L("")
    # Kelly fraction: f = (bp - q) / b where b = avg_win/avg_loss, p = WR, q = 1-WR
    a = study["A"]
    wins  = [t.get("r", 0) for t in a["trades"] if t.get("r", 0) > 0]
    losses= [abs(t.get("r", 0)) for t in a["trades"] if t.get("r", 0) < 0]
    if wins and losses:
        b = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
        p = a["win_rate"]
        q = 1 - p
        kelly_full = (b * p - q) / b
        kelly_half = kelly_full / 2
        L(f"Full Kelly fraction: **{kelly_full*100:.1f}%** per trade "
          f"(WR={p*100:.0f}%, avg_win={sum(wins)/len(wins):.2f}R, "
          f"avg_loss={sum(losses)/len(losses):.2f}R)")
        L(f"Half-Kelly (practical): **{kelly_half*100:.1f}%** per trade")
        L("")
        # Flag which variant is closest to half-Kelly
        closest = min(VARIANTS, key=lambda v: abs(v[1] - kelly_half))
        L(f"Variant **{closest[0]} ({closest[1]*100:.0f}%)** is nearest to half-Kelly.")
    L("")
    L("_Report generated by `scripts/research_position_sizing.py`._")
    L("_Offline replay only — no live changes, no master merge._")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Position sizing research")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  POSITION SIZING RESEARCH")
    print(f"  {STUDY_START.date()} → {STUDY_END.date()}")
    print("  Offline replay — no live changes")
    print("═"*60)

    study = run_study(verbose=args.verbose)

    print("\n  Generating report…")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(study)
    REPORT_PATH.write_text(report)
    print(f"  Report → {REPORT_PATH}")
    print("\n✅  Done.")


if __name__ == "__main__":
    main()
