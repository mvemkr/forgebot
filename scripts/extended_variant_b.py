#!/usr/bin/env python3
"""
Extended Variant B confirmation study — offline replay only.

Compares:
  A  engulf_only / lb=2                        Baseline (current production)
  B  engulf_or_strict_pin_at_level / lb=2      Candidate

Across:
  Full-year     2025-01-01 → 2026-03-01   (primary statistical window)
  Q1-2025       2025-01-01 → 2025-03-31
  Q2-2025       2025-04-01 → 2025-06-30
  Q3-2025       2025-07-01 → 2025-09-30
  Q4-2025       2025-10-01 → 2025-12-31
  Jan-Feb-2026  2026-01-01 → 2026-02-28
  W1            2026-02-01 → 2026-02-14   (prior study reference)
  W2            2026-02-15 → 2026-02-28   (prior study reference)

Metrics per window:
  Primary:  trades, WR, avg R, return%, maxDD, worst-3L
  Unlocks:  newly unlocked B trades vs A, MAE/MFE, pair/pattern/signal breakdown
  Trigger:  strict-pin-fired vs engulf-fired counts and performance split
  Regime:   sub-period summary table (is B improvement persistent?)

Promotion criteria (all four must pass for LIVE_PAPER consideration):
  1. Full-year return: B ≥ A − 0.5%
  2. Full-year maxDD:  B ≤ A + 1.5%
  3. Unlock concentration (≥5 unlocks): no single pair/pattern > 60%
  4. Strict-pin WR:    strict-pin triggered trades WR ≥ 45%

Safety:
  atexit() resets ENTRY_TRIGGER_MODE + ENGULF_CONFIRM_LOOKBACK_BARS to
  production defaults even on crash or KeyboardInterrupt.
"""

from __future__ import annotations

import atexit
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

import src.strategy.forex.strategy_config as _sc

# ── atexit guard (must come before any config mutation) ───────────────────────
_ORIG_TRIGGER_MODE = getattr(_sc, "ENTRY_TRIGGER_MODE",          "engulf_only")
_ORIG_LB           = getattr(_sc, "ENGULF_CONFIRM_LOOKBACK_BARS", 2)

def _reset_config():
    _sc.ENTRY_TRIGGER_MODE           = _ORIG_TRIGGER_MODE
    _sc.ENGULF_CONFIRM_LOOKBACK_BARS = _ORIG_LB

atexit.register(_reset_config)

# ── backtester ────────────────────────────────────────────────────────────────
from backtesting.oanda_backtest_v2 import run_backtest, BacktestResult  # noqa

# ──────────────────────────────────────────────────────────────────────────────
# Window definitions
# ──────────────────────────────────────────────────────────────────────────────
CAPITAL = 8_000.0

# (label, start_dt, end_dt, is_subperiod)
# is_subperiod=True → included in regime breakdown table; False → standalone row
FULL_YEAR_LABEL = "Full-year"

WINDOWS: List[Tuple[str, datetime, datetime, bool]] = [
    # Full statistical window
    ("Full-year",    datetime(2025, 1,  1, tzinfo=timezone.utc),
                     datetime(2026, 3,  1, tzinfo=timezone.utc), False),
    # Quarterly sub-periods
    ("Q1-2025",      datetime(2025, 1,  1, tzinfo=timezone.utc),
                     datetime(2025, 3, 31, tzinfo=timezone.utc), True),
    ("Q2-2025",      datetime(2025, 4,  1, tzinfo=timezone.utc),
                     datetime(2025, 6, 30, tzinfo=timezone.utc), True),
    ("Q3-2025",      datetime(2025, 7,  1, tzinfo=timezone.utc),
                     datetime(2025, 9, 30, tzinfo=timezone.utc), True),
    ("Q4-2025",      datetime(2025, 10, 1, tzinfo=timezone.utc),
                     datetime(2025, 12, 31, tzinfo=timezone.utc), True),
    ("Jan-Feb-2026", datetime(2026, 1,  1, tzinfo=timezone.utc),
                     datetime(2026, 2, 28, tzinfo=timezone.utc), True),
    # Prior study reference windows (short, fast, cross-reference)
    ("W1",           datetime(2026, 2,  1, tzinfo=timezone.utc),
                     datetime(2026, 2, 14, tzinfo=timezone.utc), False),
    ("W2",           datetime(2026, 2, 15, tzinfo=timezone.utc),
                     datetime(2026, 2, 28, tzinfo=timezone.utc), False),
]

REPORT_PATH = REPO / "backtesting/results/extended_variant_b.md"

# (label, trigger_mode, lookback, short_desc)
VARIANTS: List[Tuple[str, str, int, str]] = [
    ("A", "engulf_only",                   2, "Baseline — engulf_only, lb=2 (production)"),
    ("B", "engulf_or_strict_pin_at_level", 2, "Candidate — engulf OR strict-pin, lb=2"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _trade_key(t: Dict) -> str:
    ts = t.get("entry_ts") or t.get("open_ts") or ""
    if hasattr(ts, "strftime"):
        ts = ts.strftime("%Y%m%d%H")
    else:
        ts = str(ts)[:13].replace("-", "").replace("T", "").replace(":", "").replace(" ", "")
    return f"{t.get('pair', '?')}|{ts}"


def _r(t: Dict) -> float:
    for k in ("r", "realised_r", "result_r"):
        v = t.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def _is_win(t: Dict) -> bool:
    return _r(t) > 0.0


def _pct(v: float) -> str:
    return f"{v:+.1f}%"


def _rs(v: float) -> str:
    return f"{v:+.2f}R"


def _wr_str(trades: List[Dict]) -> str:
    if not trades:
        return "—"
    wins = sum(1 for t in trades if _is_win(t))
    return f"{int(wins / len(trades) * 100)}%"


def _wr_pct(trades: List[Dict]) -> Optional[float]:
    if not trades:
        return None
    return sum(1 for t in trades if _is_win(t)) / len(trades) * 100.0


def _avg_r_val(trades: List[Dict]) -> float:
    if not trades:
        return 0.0
    return sum(_r(t) for t in trades) / len(trades)


def _worst3l(trades: List[Dict]) -> float:
    rs = sorted(_r(t) for t in trades)
    return sum(rs[:3])


def _mae(t: Dict) -> Optional[float]:
    v = t.get("mae_r") or t.get("mae") or t.get("max_adverse_excursion")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mfe(t: Dict) -> Optional[float]:
    v = t.get("mfe_r") or t.get("mfe") or t.get("max_favorable_excursion")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _signal_type(t: Dict) -> str:
    """Normalised trigger/signal type key from trade dict."""
    return (t.get("signal_type") or
            t.get("trigger_type") or
            t.get("entry_trigger") or
            "unknown")


def _pattern(t: Dict) -> str:
    return (t.get("pattern") or
            t.get("pattern_type") or
            "unknown")


def _is_strict_pin(t: Dict) -> bool:
    """True if the entry was fired by a strict-rejection candle (not engulfing)."""
    st = _signal_type(t).lower()
    return "shooting_star" in st or "hammer_strict" in st or "strict_pin" in st


def _is_engulf(t: Dict) -> bool:
    st = _signal_type(t).lower()
    return "engulf" in st


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_variant(
    window_label: str,
    v_label: str,
    trigger_mode: str,
    lookback: int,
    start_dt: datetime,
    end_dt: datetime,
    preloaded: Optional[Dict] = None,
) -> Tuple[BacktestResult, Optional[Dict]]:
    _sc.ENTRY_TRIGGER_MODE           = trigger_mode
    _sc.ENGULF_CONFIRM_LOOKBACK_BARS = lookback

    print(f"  [{window_label}] Var {v_label}: {trigger_mode!r} lb={lookback} "
          f"({start_dt.date()}→{end_dt.date()})", end="", flush=True)

    t0 = time.time()
    result = run_backtest(
        start_dt              = start_dt,
        end_dt                = end_dt,
        starting_bal          = CAPITAL,
        notes                 = f"ext_b_{window_label}_{v_label}",
        trail_arm_key         = f"evb_{window_label}_{v_label}",
        preloaded_candle_data = preloaded,
        use_cache             = True,
        quiet                 = True,
    )
    elapsed = time.time() - t0
    _reset_config()

    print(f"  → {result.n_trades}t ret={result.return_pct:+.1f}% "
          f"WR={getattr(result,'win_rate',0.0):.0%} "
          f"maxDD={result.max_dd_pct:.1f}% ({elapsed:.0f}s)")

    return result, getattr(result, "candle_data", None)


# ──────────────────────────────────────────────────────────────────────────────
# WindowComparison
# ──────────────────────────────────────────────────────────────────────────────

class WindowComparison:
    """Stores A+B results and derived analytics for one window."""

    def __init__(
        self,
        label: str,
        start_dt: datetime,
        end_dt: datetime,
        res_a: BacktestResult,
        res_b: BacktestResult,
        is_subperiod: bool = False,
    ):
        self.label        = label
        self.start_dt     = start_dt
        self.end_dt       = end_dt
        self.res_a        = res_a
        self.res_b        = res_b
        self.is_subperiod = is_subperiod

        trades_a = res_a.trades or []
        trades_b = res_b.trades or []

        a_keys = {_trade_key(t): t for t in trades_a}
        b_keys = {_trade_key(t): t for t in trades_b}

        self.unlocked_b: List[Dict] = [b_keys[k] for k in b_keys if k not in a_keys]
        self.locked_by_b: List[Dict] = [a_keys[k] for k in a_keys if k not in b_keys]
        # All B trades (shared + unlocked)
        self.all_b_trades: List[Dict] = trades_b

    # ── scalar accessors ────────────────────────────────────────────────────

    @property
    def n_a(self) -> int: return self.res_a.n_trades
    @property
    def n_b(self) -> int: return self.res_b.n_trades
    @property
    def ret_a(self) -> float: return self.res_a.return_pct
    @property
    def ret_b(self) -> float: return self.res_b.return_pct
    @property
    def dd_a(self) -> float: return self.res_a.max_dd_pct
    @property
    def dd_b(self) -> float: return self.res_b.max_dd_pct
    @property
    def wr_a(self) -> int: return int(round(getattr(self.res_a, "win_rate", 0.0) * 100))
    @property
    def wr_b(self) -> int: return int(round(getattr(self.res_b, "win_rate", 0.0) * 100))
    @property
    def avg_r_a(self) -> float: return getattr(self.res_a, "avg_r", 0.0) or _avg_r_val(self.res_a.trades or [])
    @property
    def avg_r_b(self) -> float: return getattr(self.res_b, "avg_r", 0.0) or _avg_r_val(self.res_b.trades or [])
    @property
    def w3l_a(self) -> float: return _worst3l(self.res_a.trades or [])
    @property
    def w3l_b(self) -> float: return _worst3l(self.res_b.trades or [])

    # ── trigger breakdown ────────────────────────────────────────────────────

    def strict_pin_trades(self) -> List[Dict]:
        return [t for t in self.all_b_trades if _is_strict_pin(t)]

    def engulf_trades(self) -> List[Dict]:
        return [t for t in self.all_b_trades if _is_engulf(t)]

    def trigger_table(self) -> Dict[str, List[Dict]]:
        """Group B trades by signal_type bucket."""
        out: Dict[str, List[Dict]] = {}
        for t in self.all_b_trades:
            st = _signal_type(t)
            out.setdefault(st, []).append(t)
        return dict(sorted(out.items(), key=lambda x: -len(x[1])))

    # ── unlock statistics ────────────────────────────────────────────────────

    def unlock_mae_mfe(self) -> Tuple[Optional[str], Optional[str]]:
        maes = [_mae(t) for t in self.unlocked_b if _mae(t) is not None]
        mfes = [_mfe(t) for t in self.unlocked_b if _mfe(t) is not None]
        mae_s = _rs(sum(maes) / len(maes)) if maes else None
        mfe_s = _rs(sum(mfes) / len(mfes)) if mfes else None
        return mae_s, mfe_s

    def pair_concentration(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for t in self.unlocked_b:
            p = t.get("pair", "?")
            out[p] = out.get(p, 0) + 1
        return dict(sorted(out.items(), key=lambda x: -x[1]))

    def pattern_concentration(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for t in self.unlocked_b:
            p = _pattern(t)
            out[p] = out.get(p, 0) + 1
        return dict(sorted(out.items(), key=lambda x: -x[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Promotion gate
# ──────────────────────────────────────────────────────────────────────────────

def _promotion_check(
    wcs: List[WindowComparison],
) -> Tuple[bool, List[str]]:
    """
    Evaluate 4 promotion criteria against the full-year window.
    Returns (promote: bool, notes: List[str]).
    """
    by_label = {wc.label: wc for wc in wcs}
    notes: List[str] = []
    passed = [True, True, True, True]

    # 1. Full-year return: B ≥ A − 0.5%
    fy = by_label.get(FULL_YEAR_LABEL)
    if fy:
        ok = fy.ret_b >= fy.ret_a - 0.5
        passed[0] = ok
        notes.append(
            f"{'✅' if ok else '❌'} **Criterion 1 — Full-year return**: "
            f"B={_pct(fy.ret_b)} vs A={_pct(fy.ret_a)} "
            f"(threshold: A−0.5%) → {'PASS' if ok else 'FAIL'}"
        )
    else:
        notes.append("⚠️  **Criterion 1** — Full-year data missing")

    # 2. Full-year maxDD: B ≤ A + 1.5%
    if fy:
        ok = fy.dd_b <= fy.dd_a + 1.5
        passed[1] = ok
        notes.append(
            f"{'✅' if ok else '❌'} **Criterion 2 — Full-year maxDD**: "
            f"B={fy.dd_b:.1f}% vs A={fy.dd_a:.1f}% "
            f"(limit A+1.5%) → {'PASS' if ok else 'FAIL'}"
        )
    else:
        notes.append("⚠️  **Criterion 2** — Full-year data missing")

    # 3. Unlock concentration (only evaluated when ≥5 unlocked trades)
    all_unlocked = [t for wc in wcs for t in wc.unlocked_b]
    n_ul = len(all_unlocked)
    if n_ul < 5:
        notes.append(
            f"⚠️  **Criterion 3 — Unlock concentration**: "
            f"Only {n_ul} total unlocked trade(s) across all windows — "
            f"insufficient sample for concentration judgment (need ≥5). "
            f"Criterion marked N/A."
        )
        # Don't block on small-n
    else:
        pair_counts: Dict[str, int] = {}
        pat_counts:  Dict[str, int] = {}
        for t in all_unlocked:
            pair_counts[t.get("pair", "?")] = pair_counts.get(t.get("pair", "?"), 0) + 1
            pt = _pattern(t)
            pat_counts[pt] = pat_counts.get(pt, 0) + 1
        top_pair = max(pair_counts.items(), key=lambda x: x[1])
        top_pat  = max(pat_counts.items(),  key=lambda x: x[1])
        pair_pct = top_pair[1] / n_ul
        pat_pct  = top_pat[1]  / n_ul
        ok = pair_pct <= 0.60 and pat_pct <= 0.60
        passed[2] = ok
        notes.append(
            f"{'✅' if ok else '❌'} **Criterion 3 — Unlock concentration** "
            f"({n_ul} unlocks): "
            f"top pair={top_pair[0]} {pair_pct:.0%} "
            f"({'ok' if pair_pct <= 0.60 else 'CONCENTRATED'}), "
            f"top pattern={top_pat[0]} {pat_pct:.0%} "
            f"({'ok' if pat_pct <= 0.60 else 'CONCENTRATED'}) "
            f"→ {'PASS' if ok else 'FAIL'}"
        )

    # 4. Strict-pin WR ≥ 45% (across all windows combined)
    all_b_strict_pin = [t for wc in wcs for t in wc.strict_pin_trades()]
    if all_b_strict_pin:
        sp_wr = sum(1 for t in all_b_strict_pin if _is_win(t)) / len(all_b_strict_pin)
        ok = sp_wr >= 0.45
        passed[3] = ok
        notes.append(
            f"{'✅' if ok else '❌'} **Criterion 4 — Strict-pin WR**: "
            f"{sp_wr:.0%} ({sum(1 for t in all_b_strict_pin if _is_win(t))}W/"
            f"{len(all_b_strict_pin)-sum(1 for t in all_b_strict_pin if _is_win(t))}L "
            f"over {len(all_b_strict_pin)} trades) "
            f"(threshold: ≥45%) → {'PASS' if ok else 'FAIL'}"
        )
    else:
        notes.append(
            "⚠️  **Criterion 4 — Strict-pin WR**: "
            "No strict-pin triggered trades found across all windows. "
            "Criterion not evaluated."
        )

    promote = all(passed)
    return promote, notes


# ──────────────────────────────────────────────────────────────────────────────
# Report builder
# ──────────────────────────────────────────────────────────────────────────────

def build_report(wcs: List[WindowComparison]) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = []
    a = lines.extend

    # Header
    a([
        "# Extended Variant B Confirmation Study",
        f"Generated: {now_utc}",
        "",
        "## Study Design",
        "",
        "| Variant | Trigger mode | LB | Description |",
        "|---------|-------------|----|-------------|",
        "| A | `engulf_only` | 2 | Baseline — current production |",
        "| B | `engulf_or_strict_pin_at_level` | 2 | Candidate — adds strict-rejection trigger |",
        "",
        "**Strict-rejection candle** (`strict_pin`): wick ≥ 3× body, body ≤ 25% range,",
        "opposite wick ≤ 10% range, close in outer 35%, level touched within last 5 H1 bars.",
        "",
        "Capital: $8,000 | Pairs: 7 (Alex universe) | H1 cadence",
        "",
    ])

    # Window table
    a([
        "## Windows",
        "",
        "| Window | Start | End | Days | Role |",
        "|--------|-------|-----|------|------|",
    ])
    for label, ws, we, is_sub in WINDOWS:
        days = (we - ws).days
        role = "Sub-period" if is_sub else ("Primary" if label == "Full-year" else "Reference")
        a([f"| {label} | {ws.date()} | {we.date()} | {days} | {role} |"])

    a(["", "---", ""])

    # ── Section 1: Full-year primary comparison ───────────────────────────────
    fy_wcs = [wc for wc in wcs if wc.label == FULL_YEAR_LABEL]
    if fy_wcs:
        fy = fy_wcs[0]
        n_ul = len(fy.unlocked_b)
        a([
            "## Full-Year Comparison (2025-01-01 → 2026-03-01)",
            "",
            "| Metric | Variant A (baseline) | Variant B (candidate) | Δ |",
            "|--------|---------------------|----------------------|---|",
            f"| Total trades | {fy.n_a} | {fy.n_b} | {fy.n_b - fy.n_a:+d} |",
            f"| Win rate | {fy.wr_a}% | {fy.wr_b}% | {fy.wr_b - fy.wr_a:+d}pp |",
            f"| Average R | {_rs(fy.avg_r_a)} | {_rs(fy.avg_r_b)} | {_rs(fy.avg_r_b - fy.avg_r_a)} |",
            f"| Return % | {_pct(fy.ret_a)} | {_pct(fy.ret_b)} | {_pct(fy.ret_b - fy.ret_a)} |",
            f"| Max drawdown | {fy.dd_a:.1f}% | {fy.dd_b:.1f}% | {fy.dd_b - fy.dd_a:+.1f}pp |",
            f"| Worst-3-loss R | {_rs(fy.w3l_a)} | {_rs(fy.w3l_b)} | {_rs(fy.w3l_b - fy.w3l_a)} |",
            f"| Unlocked by B | — | +{n_ul} | — |",
            f"| Locked by B   | — | −{len(fy.locked_by_b)} | — |",
            "",
        ])

        # Trigger breakdown for full year
        strict_b = fy.strict_pin_trades()
        engulf_b = fy.engulf_trades()
        other_b  = [t for t in fy.all_b_trades
                    if not _is_strict_pin(t) and not _is_engulf(t)]
        a([
            "### Full-Year Trigger Breakdown (Variant B)",
            "",
            "| Trigger bucket | Trades | Wins | WR | Avg R | Notes |",
            "|----------------|:------:|-----:|---:|------:|-------|",
        ])
        for bucket_label, bucket in [
            ("Engulfing",    engulf_b),
            ("Strict-pin",   strict_b),
            ("Other/unknown", other_b),
        ]:
            if bucket:
                wr  = f"{int(sum(1 for t in bucket if _is_win(t))/len(bucket)*100)}%"
                ar  = _rs(_avg_r_val(bucket))
                w   = sum(1 for t in bucket if _is_win(t))
                note = "(new unlock signal)" if bucket_label == "Strict-pin" else ""
                a([f"| {bucket_label} | {len(bucket)} | {w} | {wr} | {ar} | {note} |"])
        a([""])

        # Full-year signal_type detail
        trig_tbl = fy.trigger_table()
        if trig_tbl:
            a([
                "### Full-Year Signal Type Detail (Variant B)",
                "",
                "| Signal type | Count | Wins | WR | Avg R |",
                "|------------|:-----:|-----:|---:|------:|",
            ])
            for st, ts in trig_tbl.items():
                w  = sum(1 for t in ts if _is_win(t))
                wr = f"{int(w/len(ts)*100)}%"
                ar = _rs(_avg_r_val(ts))
                a([f"| `{st}` | {len(ts)} | {w} | {wr} | {ar} |"])
            a([""])

        # Full-year unlocked trade list
        if fy.unlocked_b:
            a([
                "### Full-Year Unlocked Trades (B ∖ A)",
                "",
                "| # | Pair | Dir | Entry | Pattern | Signal type | R | MAE | MFE |",
                "|---|------|-----|-------|---------|-------------|---|-----|-----|",
            ])
            for i, t in enumerate(fy.unlocked_b, 1):
                mae_s = _rs(_mae(t)) if _mae(t) is not None else "n/a"
                mfe_s = _rs(_mfe(t)) if _mfe(t) is not None else "n/a"
                a([
                    f"| {i} | {t.get('pair','?')} | {t.get('direction','?')} "
                    f"| {str(t.get('entry_ts','?'))[:16]} "
                    f"| {_pattern(t)} | `{_signal_type(t)}` "
                    f"| {_rs(_r(t))} | {mae_s} | {mfe_s} |"
                ])
            mae_ul, mfe_ul = fy.unlock_mae_mfe()
            a(["",
               f"**Unlocked avg MAE**: {mae_ul if mae_ul else 'n/a'} | "
               f"**Unlocked avg MFE**: {mfe_ul if mfe_ul else 'n/a'}",
               ""])
        else:
            a(["*No new trades unlocked by Variant B in the full-year window.*", ""])

    a(["---", ""])

    # ── Section 2: Sub-period regime breakdown ────────────────────────────────
    sub_wcs  = [wc for wc in wcs if wc.is_subperiod]
    ref_wcs  = [wc for wc in wcs
                if not wc.is_subperiod and wc.label != FULL_YEAR_LABEL]

    a([
        "## Sub-Period Regime Breakdown",
        "",
        "*(Is Variant B's improvement persistent across regimes, or concentrated?)*",
        "",
        "| Period | A trades | B trades | Δ | A ret% | B ret% | Δ ret | A WR | B WR | A maxDD | B maxDD | B unlocked |",
        "|--------|:--------:|:--------:|:-:|:------:|:------:|:-----:|:----:|:----:|:-------:|:-------:|:----------:|",
    ])
    for wc in sub_wcs:
        a([
            f"| {wc.label} "
            f"| {wc.n_a} | {wc.n_b} | {wc.n_b - wc.n_a:+d} "
            f"| {_pct(wc.ret_a)} | {_pct(wc.ret_b)} | {_pct(wc.ret_b - wc.ret_a)} "
            f"| {wc.wr_a}% | {wc.wr_b}% "
            f"| {wc.dd_a:.1f}% | {wc.dd_b:.1f}% "
            f"| +{len(wc.unlocked_b)} |"
        ])
    a([""])

    # Sub-period persistence check
    improved = [wc for wc in sub_wcs if wc.ret_b > wc.ret_a]
    neutral  = [wc for wc in sub_wcs if abs(wc.ret_b - wc.ret_a) < 0.05]
    degraded = [wc for wc in sub_wcs if wc.ret_b < wc.ret_a - 0.05]
    a([
        "**Regime persistence**:",
        f"- Improved: {len(improved)}/{len(sub_wcs)} periods "
        f"({', '.join(wc.label for wc in improved) or 'none'})",
        f"- Neutral (Δ<0.05%): {len(neutral)}/{len(sub_wcs)} periods "
        f"({', '.join(wc.label for wc in neutral) or 'none'})",
        f"- Degraded: {len(degraded)}/{len(sub_wcs)} periods "
        f"({', '.join(wc.label for wc in degraded) or 'none'})",
        "",
    ])

    a(["---", ""])

    # ── Section 3: Reference window comparison (W1/W2) ────────────────────────
    if ref_wcs:
        a([
            "## Reference Windows (Prior Study Cross-Check)",
            "",
            "| Window | A trades | B trades | Δ | A ret | B ret | Δ | B unlocked | B unlock WR |",
            "|--------|:--------:|:--------:|:-:|:-----:|:-----:|:-:|:----------:|:-----------:|",
        ])
        for wc in ref_wcs:
            ul_wr = _wr_str(wc.unlocked_b) if wc.unlocked_b else "—"
            a([
                f"| {wc.label} "
                f"| {wc.n_a} | {wc.n_b} | {wc.n_b - wc.n_a:+d} "
                f"| {_pct(wc.ret_a)} | {_pct(wc.ret_b)} | {_pct(wc.ret_b - wc.ret_a)} "
                f"| +{len(wc.unlocked_b)} | {ul_wr} |"
            ])
        a(["",
           "> W1 and W2 are sub-windows of Jan-Feb-2026 and are provided for cross-reference",
           "> with `confirm_variant_b.md`.",
           ""])
        a(["---", ""])

    # ── Section 4: Aggregate unlock summary (all windows) ─────────────────────
    all_unlocked = [t for wc in wcs for t in wc.unlocked_b]
    all_b_strict = [t for wc in wcs for t in wc.strict_pin_trades()]
    all_b_engulf = [t for wc in wcs for t in wc.engulf_trades()]

    a([
        "## Aggregate Unlock Summary (All Windows)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total A trades (sum) | {sum(wc.n_a for wc in wcs)} |",
        f"| Total B trades (sum) | {sum(wc.n_b for wc in wcs)} |",
        f"| Total newly unlocked | {len(all_unlocked)} |",
        f"| Total locked-by-B    | {sum(len(wc.locked_by_b) for wc in wcs)} |",
        f"| Strict-pin triggered (Var B, all wins) | {len(all_b_strict)} |",
        f"| Engulf triggered (Var B)    | {len(all_b_engulf)} |",
    ])
    if all_unlocked:
        wins_ul = sum(1 for t in all_unlocked if _is_win(t))
        avg_r_ul = _avg_r_val(all_unlocked)
        a([
            f"| Unlocked WR | {int(wins_ul/len(all_unlocked)*100)}% |",
            f"| Unlocked avg R | {_rs(avg_r_ul)} |",
        ])
    if all_b_strict:
        sp_wr_n = sum(1 for t in all_b_strict if _is_win(t))
        a([
            f"| Strict-pin WR | {int(sp_wr_n/len(all_b_strict)*100)}% |",
            f"| Strict-pin avg R | {_rs(_avg_r_val(all_b_strict))} |",
        ])
    a([""])

    if all_unlocked:
        # Pair breakdown
        a([
            "### Unlocked Trade Pair Distribution",
            "",
            "| Pair | Count | % | Wins | WR | Avg R |",
            "|------|:-----:|:-:|-----:|---:|------:|",
        ])
        pair_map: Dict[str, List[Dict]] = {}
        for t in all_unlocked:
            p = t.get("pair", "?")
            pair_map.setdefault(p, []).append(t)
        for pair, ts in sorted(pair_map.items(), key=lambda x: -len(x[1])):
            w   = sum(1 for t in ts if _is_win(t))
            pct = f"{int(len(ts)/len(all_unlocked)*100)}%"
            wr  = f"{int(w/len(ts)*100)}%" if ts else "—"
            ar  = _rs(_avg_r_val(ts))
            a([f"| {pair} | {len(ts)} | {pct} | {w} | {wr} | {ar} |"])
        a([""])

        # Pattern breakdown
        a([
            "### Unlocked Trade Pattern Distribution",
            "",
            "| Pattern | Count | % | Wins | WR | Avg R |",
            "|---------|:-----:|:-:|-----:|---:|------:|",
        ])
        pat_map: Dict[str, List[Dict]] = {}
        for t in all_unlocked:
            p = _pattern(t)
            pat_map.setdefault(p, []).append(t)
        for pat, ts in sorted(pat_map.items(), key=lambda x: -len(x[1])):
            w   = sum(1 for t in ts if _is_win(t))
            pct = f"{int(len(ts)/len(all_unlocked)*100)}%"
            wr  = f"{int(w/len(ts)*100)}%" if ts else "—"
            ar  = _rs(_avg_r_val(ts))
            a([f"| {pat} | {len(ts)} | {pct} | {w} | {wr} | {ar} |"])
        a([""])

        # Window concentration
        a([
            "### Unlocked Trade Window Concentration",
            "",
            "| Window | Unlocked | % | Wins | WR | Avg R |",
            "|--------|:--------:|:-:|-----:|---:|------:|",
        ])
        for wc in wcs:
            ul = wc.unlocked_b
            n  = len(ul)
            pct = f"{int(n/len(all_unlocked)*100)}%" if all_unlocked else "—"
            w   = sum(1 for t in ul if _is_win(t))
            wr  = f"{int(w/n*100)}%" if n else "—"
            ar  = _rs(_avg_r_val(ul)) if n else "—"
            a([f"| {wc.label} | {n} | {pct} | {w} | {wr} | {ar} |"])
        a([""])

    # ── Section 5: Strict-pin vs engulf performance split ─────────────────────
    a([
        "## Strict-Pin vs Engulf Performance Split (Variant B, All Windows)",
        "",
        "| Trigger | Trades | Wins | WR | Avg R | Avg MAE | Avg MFE |",
        "|---------|:------:|-----:|---:|------:|--------:|--------:|",
    ])
    for bucket_label, bucket in [("Engulfing", all_b_engulf), ("Strict-pin", all_b_strict)]:
        if bucket:
            w    = sum(1 for t in bucket if _is_win(t))
            wr   = f"{int(w/len(bucket)*100)}%"
            ar   = _rs(_avg_r_val(bucket))
            maes = [_mae(t) for t in bucket if _mae(t) is not None]
            mfes = [_mfe(t) for t in bucket if _mfe(t) is not None]
            mae_s = _rs(sum(maes)/len(maes)) if maes else "n/a"
            mfe_s = _rs(sum(mfes)/len(mfes)) if mfes else "n/a"
            a([f"| {bucket_label} | {len(bucket)} | {w} | {wr} | {ar} | {mae_s} | {mfe_s} |"])
    a([""])

    a(["---", ""])

    # ── Section 6: Return comparison table ────────────────────────────────────
    a([
        "## Return Comparison — All Windows",
        "",
        "| Window | A ret | B ret | Δ | A WR | B WR | A maxDD | B maxDD | Unlocked |",
        "|--------|:-----:|:-----:|:-:|:----:|:----:|:-------:|:-------:|:--------:|",
    ])
    for wc in wcs:
        a([
            f"| {wc.label} "
            f"| {_pct(wc.ret_a)} | {_pct(wc.ret_b)} | {_pct(wc.ret_b - wc.ret_a)} "
            f"| {wc.wr_a}% | {wc.wr_b}% "
            f"| {wc.dd_a:.1f}% | {wc.dd_b:.1f}% "
            f"| +{len(wc.unlocked_b)} |"
        ])
    a([""])

    a(["---", ""])

    # ── Section 7: Promotion gate ─────────────────────────────────────────────
    promote, crit_notes = _promotion_check(wcs)
    promote_word = ("✅ **PROMOTED — Variant B qualifies for LIVE_PAPER testing**"
                    if promote else
                    "❌ **NOT PROMOTED — Variant B does not yet qualify**")

    a([
        "## Promotion Evaluation",
        "",
        "### Criteria",
        "",
    ])
    for note in crit_notes:
        a([note, ""])

    a([
        "### Conclusion",
        "",
        promote_word,
        "",
    ])

    if promote:
        a([
            "**Next step**: Set `ENTRY_TRIGGER_MODE = \"engulf_or_strict_pin_at_level\"` on a",
            "new `live_paper` branch. Run 2-week shadow monitoring before changing production.",
            "",
        ])
    else:
        a([
            "**Next step**: Address failing criteria before re-evaluating.",
            "Do NOT change `ENTRY_TRIGGER_MODE` in production.",
            "",
        ])

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Extended Variant B confirmation study — offline replay only")
    print(f"  {len(WINDOWS)} windows × 2 variants = {len(WINDOWS) * 2} backtest runs")
    print("=" * 72)

    wcs: List[WindowComparison] = []

    for label, ws, we, is_sub in WINDOWS:
        print(f"\n── {label} ({ws.date()} → {we.date()}) ──")
        res_a, candles = run_variant(label, "A", "engulf_only", 2, ws, we)
        res_b, _       = run_variant(label, "B", "engulf_or_strict_pin_at_level", 2,
                                     ws, we, preloaded=candles)
        wcs.append(WindowComparison(label, ws, we, res_a, res_b, is_sub))

    print(f"\n{'=' * 72}")
    report = build_report(wcs)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"  Report → {REPORT_PATH.relative_to(REPO)}")
    print("=" * 72)
    print()
    print(report)


if __name__ == "__main__":
    main()
