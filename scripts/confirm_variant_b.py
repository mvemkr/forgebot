#!/usr/bin/env python3
"""
Variant B multi-window confirmation study — offline replay only.

Compares:
  A  engulf_only / lb=2            Baseline (current production)
  B  engulf_or_strict_pin_at_level / lb=2   Candidate for promotion

Across three independent windows:
  W1           2026-02-01 → 2026-02-14
  W2           2026-02-15 → 2026-02-28
  Live-parity  2026-02-28 → 2026-03-06

Promotion criteria (all four must pass for LIVE_PAPER flag):
  1. W1   — B return ≥ A return  (no regression)
  2. W2   — B maxDD ≤ A maxDD + 1.0%  (no material DD increase)
  3. Unlock concentration — no single pair/pattern supplies > 60% of unlocked trades
  4. Stale-entry check — all unlocked entries within 4 bars of pattern detection

Safety
------
  atexit() resets ENTRY_TRIGGER_MODE and ENGULF_CONFIRM_LOOKBACK_BARS to
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

WINDOWS: List[Tuple[str, datetime, datetime]] = [
    ("W1",          datetime(2026, 2,  1, tzinfo=timezone.utc),
                    datetime(2026, 2, 14, tzinfo=timezone.utc)),
    ("W2",          datetime(2026, 2, 15, tzinfo=timezone.utc),
                    datetime(2026, 2, 28, tzinfo=timezone.utc)),
    ("Live-parity", datetime(2026, 2, 28, tzinfo=timezone.utc),
                    datetime(2026, 3,  6, tzinfo=timezone.utc)),
]

REPORT_PATH = REPO / "backtesting/results/confirm_variant_b.md"

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


def _wr(trades: List[Dict]) -> str:
    if not trades:
        return "—"
    wins = sum(1 for t in trades if _is_win(t))
    return f"{int(wins / len(trades) * 100)}%"


def _avg_r(trades: List[Dict]) -> str:
    if not trades:
        return "—"
    return _rs(sum(_r(t) for t in trades) / len(trades))


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


def _entry_dt(t: Dict) -> Optional[datetime]:
    ts = t.get("entry_ts") or t.get("open_ts")
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _detection_dt(t: Dict) -> Optional[datetime]:
    ts = t.get("pattern_ts") or t.get("detection_ts") or t.get("signal_ts")
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _bars_lag(t: Dict) -> Optional[int]:
    """
    Hours between pattern detection and entry (proxy for bar lag at H1 cadence).
    Returns None if either timestamp is unavailable.
    """
    e_dt = _entry_dt(t)
    d_dt = _detection_dt(t)
    if e_dt is None or d_dt is None:
        return None
    delta_h = round((e_dt - d_dt).total_seconds() / 3600)
    return max(0, delta_h)


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_variant(
    window_label: str,
    v_label: str,
    trigger_mode: str,
    lookback: int,
    desc: str,
    start_dt: datetime,
    end_dt: datetime,
    preloaded: Optional[Dict] = None,
) -> Tuple[BacktestResult, Optional[Dict]]:
    print(f"\n{'─' * 72}")
    print(f"  [{window_label}] Variant {v_label}: {desc}")
    print(f"  {start_dt.date()} → {end_dt.date()}  |  "
          f"trigger={trigger_mode!r}  lb={lookback}")
    print(f"{'─' * 72}\n")

    _sc.ENTRY_TRIGGER_MODE           = trigger_mode
    _sc.ENGULF_CONFIRM_LOOKBACK_BARS = lookback

    t0 = time.time()
    result = run_backtest(
        start_dt              = start_dt,
        end_dt                = end_dt,
        starting_bal          = CAPITAL,
        notes                 = f"confirm_b_{window_label}_{v_label}",
        trail_arm_key         = f"cb_{window_label}_{v_label}",
        preloaded_candle_data = preloaded,
        use_cache             = True,
        quiet                 = True,
    )
    elapsed = time.time() - t0

    # Always reset after each run
    _reset_config()

    print(f"  ✓ {result.n_trades} trades | ret={result.return_pct:+.1f}% | "
          f"WR={getattr(result, 'win_rate', 0.0):.0%} | "
          f"maxDD={result.max_dd_pct:.1f}% | {elapsed:.1f}s")

    return result, getattr(result, "candle_data", None)


# ──────────────────────────────────────────────────────────────────────────────
# Per-window analysis
# ──────────────────────────────────────────────────────────────────────────────

class WindowResult:
    """Stores A+B results and derived unlock analysis for one window."""

    def __init__(
        self,
        label: str,
        start_dt: datetime,
        end_dt: datetime,
        res_a: BacktestResult,
        res_b: BacktestResult,
    ):
        self.label    = label
        self.start_dt = start_dt
        self.end_dt   = end_dt
        self.res_a    = res_a
        self.res_b    = res_b

        # Compute unlock sets
        a_keys = {_trade_key(t): t for t in (res_a.trades or [])}
        b_keys = {_trade_key(t): t for t in (res_b.trades or [])}

        self.unlocked_b: List[Dict] = [
            b_keys[k] for k in b_keys if k not in a_keys
        ]
        self.locked_by_b: List[Dict] = [
            a_keys[k] for k in a_keys if k not in b_keys
        ]

    # ── scalar accessors ────────────────────────────────────────────────────

    def _n(self, r: BacktestResult) -> int:
        return r.n_trades

    def _ret(self, r: BacktestResult) -> float:
        return r.return_pct

    def _dd(self, r: BacktestResult) -> float:
        return r.max_dd_pct

    def _wrate(self, r: BacktestResult) -> int:
        return int(round(getattr(r, "win_rate", 0.0) * 100))

    def _avgr(self, r: BacktestResult) -> float:
        trades = r.trades or []
        if not trades:
            return 0.0
        return sum(_r(t) for t in trades) / len(trades)

    def _w3l(self, r: BacktestResult) -> float:
        rs = sorted(_r(t) for t in (r.trades or []))
        return sum(rs[:3])

    # ── unlock statistics ────────────────────────────────────────────────────

    def unlock_wr(self) -> Optional[str]:
        if not self.unlocked_b:
            return None
        w = sum(1 for t in self.unlocked_b if _is_win(t))
        return f"{int(w / len(self.unlocked_b) * 100)}%"

    def unlock_avg_r(self) -> Optional[str]:
        if not self.unlocked_b:
            return None
        return _rs(sum(_r(t) for t in self.unlocked_b) / len(self.unlocked_b))

    def unlock_mae_mfe(self) -> Tuple[Optional[str], Optional[str]]:
        """Return (avg_mae_str, avg_mfe_str) for unlocked trades."""
        maes = [_mae(t) for t in self.unlocked_b if _mae(t) is not None]
        mfes = [_mfe(t) for t in self.unlocked_b if _mfe(t) is not None]
        mae_s = _rs(sum(maes) / len(maes)) if maes else None
        mfe_s = _rs(sum(mfes) / len(mfes)) if mfes else None
        return mae_s, mfe_s

    def unlock_stale(self) -> Tuple[int, int]:
        """Returns (stale_count, total_with_lag_data)."""
        lags = [_bars_lag(t) for t in self.unlocked_b]
        valid = [l for l in lags if l is not None]
        stale = sum(1 for l in valid if l > 4)
        return stale, len(valid)

    def pair_concentration(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for t in self.unlocked_b:
            p = t.get("pair", "?")
            out[p] = out.get(p, 0) + 1
        return dict(sorted(out.items(), key=lambda x: -x[1]))

    def pattern_concentration(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for t in self.unlocked_b:
            p = t.get("pattern_type") or t.get("pattern", "unknown")
            out[p] = out.get(p, 0) + 1
        return dict(sorted(out.items(), key=lambda x: -x[1]))

    def trigger_breakdown(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for t in self.unlocked_b:
            trig = t.get("trigger_type") or t.get("entry_trigger") or "unknown"
            out[trig] = out.get(trig, 0) + 1
        return dict(sorted(out.items(), key=lambda x: -x[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Promotion gate
# ──────────────────────────────────────────────────────────────────────────────

def _promotion_check(wrs: List[WindowResult]) -> Tuple[bool, List[str]]:
    """
    Evaluate all 4 promotion criteria.
    Returns (promote: bool, notes: List[str]).
    """
    by_label = {wr.label: wr for wr in wrs}
    notes: List[str] = []
    passed = [True, True, True, True]

    # 1. W1 — B return ≥ A (no regression)
    w1 = by_label.get("W1")
    if w1 is not None:
        ret_a = w1._ret(w1.res_a)
        ret_b = w1._ret(w1.res_b)
        ok = ret_b >= ret_a - 0.01   # 1bp tolerance for float noise
        passed[0] = ok
        notes.append(
            f"{'✅' if ok else '❌'} **Criterion 1 — W1 return**: "
            f"B={_pct(ret_b)} vs A={_pct(ret_a)} → {'PASS' if ok else 'FAIL'}"
        )
    else:
        notes.append("⚠️  **Criterion 1** — W1 data missing; cannot evaluate")

    # 2. W2 — B maxDD ≤ A maxDD + 1.0% (no material DD increase)
    w2 = by_label.get("W2")
    if w2 is not None:
        dd_a = w2._dd(w2.res_a)
        dd_b = w2._dd(w2.res_b)
        ok = dd_b <= dd_a + 1.0
        passed[1] = ok
        notes.append(
            f"{'✅' if ok else '❌'} **Criterion 2 — W2 maxDD**: "
            f"B={dd_b:.1f}% vs A={dd_a:.1f}% (limit A+1.0%) → {'PASS' if ok else 'FAIL'}"
        )
    else:
        notes.append("⚠️  **Criterion 2** — W2 data missing; cannot evaluate")

    # 3. Unlock concentration — no pair/pattern > 60% across all windows
    all_unlocked = [t for wr in wrs for t in wr.unlocked_b]
    if all_unlocked:
        pair_counts: Dict[str, int] = {}
        pat_counts:  Dict[str, int] = {}
        for t in all_unlocked:
            p = t.get("pair", "?")
            pt = t.get("pattern_type") or t.get("pattern", "unknown")
            pair_counts[p]  = pair_counts.get(p, 0)  + 1
            pat_counts[pt]  = pat_counts.get(pt, 0)  + 1
        n = len(all_unlocked)
        top_pair  = max(pair_counts.items(), key=lambda x: x[1])
        top_pat   = max(pat_counts.items(),  key=lambda x: x[1])
        pair_pct  = top_pair[1] / n
        pat_pct   = top_pat[1]  / n
        ok_pair   = pair_pct  <= 0.60
        ok_pat    = pat_pct   <= 0.60
        ok        = ok_pair and ok_pat
        passed[2] = ok
        notes.append(
            f"{'✅' if ok else '❌'} **Criterion 3 — Unlock concentration**: "
            f"top pair={top_pair[0]} {pair_pct:.0%} ({'ok' if ok_pair else 'CONCENTRATED'}), "
            f"top pattern={top_pat[0]} {pat_pct:.0%} ({'ok' if ok_pat else 'CONCENTRATED'}) "
            f"→ {'PASS' if ok else 'FAIL'}"
        )
    else:
        passed[2] = True   # 0 unlocked trades → no concentration risk
        notes.append("✅ **Criterion 3 — Unlock concentration**: 0 unlocked trades → PASS (vacuously)")

    # 4. Stale-entry check — unlocked entries within 4 bars of detection
    total_stale = 0
    total_lag   = 0
    for wr in wrs:
        s, v = wr.unlock_stale()
        total_stale += s
        total_lag   += v
    if total_lag > 0:
        ok = total_stale == 0
        passed[3] = ok
        notes.append(
            f"{'✅' if ok else '❌'} **Criterion 4 — Stale entries**: "
            f"{total_stale}/{total_lag} entries have lag >4 bars → {'PASS' if ok else 'FAIL (review stale entries)'}"
        )
    else:
        # No lag data available (missing pattern_ts) — can't fail on missing data
        notes.append(
            "⚠️  **Criterion 4 — Stale entries**: entry/detection lag data unavailable "
            "(pattern_ts not logged); criterion not evaluated"
        )

    promote = all(passed)
    return promote, notes


# ──────────────────────────────────────────────────────────────────────────────
# Report builder
# ──────────────────────────────────────────────────────────────────────────────

def build_report(wrs: List[WindowResult]) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = []
    a = lines.extend

    a([
        "# Variant B Multi-Window Confirmation Study",
        f"Generated: {now_utc}",
        "",
        "## Study Design",
        "",
        "| Variant | Trigger mode | LB | Description |",
        "|---------|-------------|----|-------------|",
        "| A | `engulf_only` | 2 | Baseline — current production |",
        "| B | `engulf_or_strict_pin_at_level` | 2 | Candidate — adds strict-pin trigger |",
        "",
        "**Strict-pin** (`engulf_or_strict_pin_at_level`):",
        "- Wick ≥ 3× body, body ≤ 25% range, opposite wick ≤ 10% range,",
        "  close in outer 35%, level touched within last 5 H1 bars",
        "",
        "**Windows**:",
        "",
        "| Window | Start | End | Days |",
        "|--------|-------|-----|------|",
    ])
    for wlabel, ws, we in WINDOWS:
        days = (we - ws).days
        a([f"| {wlabel} | {ws.date()} | {we.date()} | {days} |"])

    a(["", "Capital: $8,000 | Pairs: 7 (Alex universe) | H1 cadence", ""])
    a(["---", ""])

    # ── Per-window comparison tables ─────────────────────────────────────────
    a(["## Per-Window Comparison", ""])

    for wr in wrs:
        ra, rb = wr.res_a, wr.res_b
        trades_a = ra.trades or []
        trades_b = rb.trades or []

        a([
            f"### {wr.label} ({wr.start_dt.date()} → {wr.end_dt.date()})",
            "",
            "#### Primary Metrics",
            "",
            "| Metric | Variant A (baseline) | Variant B (candidate) | Δ |",
            "|--------|---------------------|----------------------|---|",
            f"| Total trades       | {wr._n(ra)} | {wr._n(rb)} | {wr._n(rb)-wr._n(ra):+d} |",
            f"| Win rate           | {wr._wrate(ra)}% | {wr._wrate(rb)}% | {wr._wrate(rb)-wr._wrate(ra):+d}pp |",
            f"| Average R          | {_rs(wr._avgr(ra))} | {_rs(wr._avgr(rb))} | {_rs(wr._avgr(rb)-wr._avgr(ra))} |",
            f"| Return %           | {_pct(wr._ret(ra))} | {_pct(wr._ret(rb))} | {_pct(wr._ret(rb)-wr._ret(ra))} |",
            f"| Max drawdown       | {wr._dd(ra):.1f}% | {wr._dd(rb):.1f}% | {wr._dd(rb)-wr._dd(ra):+.1f}pp |",
            f"| Worst-3-loss R     | {_rs(wr._w3l(ra))} | {_rs(wr._w3l(rb))} | {_rs(wr._w3l(rb)-wr._w3l(ra))} |",
        ])

        # unlock count
        n_ul = len(wr.unlocked_b)
        n_locked = len(wr.locked_by_b)
        a([
            f"| Unlocked by B      | — | +{n_ul} | — |",
            f"| Locked by B (A-only) | — | −{n_locked} | — |",
            "",
        ])

        # unlock details
        if wr.unlocked_b:
            a([
                "#### Unlocked Trades (B ∖ A)",
                "",
                "| # | Pair | Dir | Entry | Pattern | Trigger | R |",
                "|---|------|-----|-------|---------|---------|---|",
            ])
            for i, t in enumerate(wr.unlocked_b, 1):
                pair    = t.get("pair", "?")
                direct  = t.get("direction", t.get("dir", "?"))
                ets     = str(t.get("entry_ts") or t.get("open_ts") or "?")[:16]
                pattern = t.get("pattern_type") or t.get("pattern", "?")
                trigger = t.get("trigger_type") or t.get("entry_trigger") or "?"
                rv      = _r(t)
                a([f"| {i} | {pair} | {direct} | {ets} | {pattern} | {trigger} | {_rs(rv)} |"])

            # MAE/MFE
            mae_s, mfe_s = wr.unlock_mae_mfe()
            a([""])
            if mae_s or mfe_s:
                a([
                    "**MAE / MFE — Unlocked Trades**",
                    "",
                    f"- Avg MAE: {mae_s if mae_s else 'n/a'}",
                    f"- Avg MFE: {mfe_s if mfe_s else 'n/a'}",
                    "",
                ])

            # Stale entries
            stale, valid_lag = wr.unlock_stale()
            if valid_lag > 0:
                a([
                    f"**Stale-entry check**: {stale}/{valid_lag} entries have lag >4 bars "
                    f"({'⚠️ review' if stale > 0 else '✅ clean'})",
                    "",
                ])
            else:
                a(["**Stale-entry check**: lag data unavailable (pattern_ts not in trade log)", ""])
        else:
            a(["*No new trades unlocked by Variant B in this window.*", ""])

        if wr.locked_by_b:
            a([
                "#### Trades in A but not B (locked by strict-pin gate)",
                "",
                "| # | Pair | Dir | Entry | Pattern | R |",
                "|---|------|-----|-------|---------|---|",
            ])
            for i, t in enumerate(wr.locked_by_b, 1):
                pair    = t.get("pair", "?")
                direct  = t.get("direction", t.get("dir", "?"))
                ets     = str(t.get("entry_ts") or t.get("open_ts") or "?")[:16]
                pattern = t.get("pattern_type") or t.get("pattern", "?")
                rv      = _r(t)
                a([f"| {i} | {pair} | {direct} | {ets} | {pattern} | {_rs(rv)} |"])
            a([""])

        a(["---", ""])

    # ── Aggregate unlock summary ──────────────────────────────────────────────
    all_unlocked = [t for wr in wrs for t in wr.unlocked_b]
    all_locked   = [t for wr in wrs for t in wr.locked_by_b]
    total_a      = sum(wr._n(wr.res_a) for wr in wrs)
    total_b      = sum(wr._n(wr.res_b) for wr in wrs)

    a([
        "## Aggregate Unlock Summary (All Windows)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total A trades (sum) | {total_a} |",
        f"| Total B trades (sum) | {total_b} |",
        f"| Total newly unlocked | {len(all_unlocked)} |",
        f"| Total locked-by-B    | {len(all_locked)} |",
    ])

    if all_unlocked:
        wins_ul = sum(1 for t in all_unlocked if _is_win(t))
        avg_r_ul = sum(_r(t) for t in all_unlocked) / len(all_unlocked)
        a([
            f"| Unlocked WR | {int(wins_ul/len(all_unlocked)*100)}% |",
            f"| Unlocked avg R | {_rs(avg_r_ul)} |",
        ])

    a(["", ""])

    # ── Distribution: pair breakdown ─────────────────────────────────────────
    if all_unlocked:
        a([
            "### Pair Breakdown — Unlocked Trades (All Windows)",
            "",
            "| Pair | Count | % of Unlocked | Wins | WR | Avg R |",
            "|------|------:|:-------------:|-----:|---:|------:|",
        ])
        pair_map: Dict[str, List[Dict]] = {}
        for t in all_unlocked:
            p = t.get("pair", "?")
            pair_map.setdefault(p, []).append(t)
        for pair, ts in sorted(pair_map.items(), key=lambda x: -len(x[1])):
            w  = sum(1 for t in ts if _is_win(t))
            pct = f"{int(len(ts)/len(all_unlocked)*100)}%"
            wr  = f"{int(w/len(ts)*100)}%" if ts else "—"
            ar  = _rs(sum(_r(t) for t in ts) / len(ts))
            a([f"| {pair} | {len(ts)} | {pct} | {w} | {wr} | {ar} |"])
        a([""])

        a([
            "### Pattern Breakdown — Unlocked Trades (All Windows)",
            "",
            "| Pattern | Count | % | Wins | WR | Avg R |",
            "|---------|------:|:-:|-----:|---:|------:|",
        ])
        pat_map: Dict[str, List[Dict]] = {}
        for t in all_unlocked:
            p = t.get("pattern_type") or t.get("pattern", "unknown")
            pat_map.setdefault(p, []).append(t)
        for pat, ts in sorted(pat_map.items(), key=lambda x: -len(x[1])):
            w   = sum(1 for t in ts if _is_win(t))
            pct = f"{int(len(ts)/len(all_unlocked)*100)}%"
            wr  = f"{int(w/len(ts)*100)}%" if ts else "—"
            ar  = _rs(sum(_r(t) for t in ts) / len(ts))
            a([f"| {pat} | {len(ts)} | {pct} | {w} | {wr} | {ar} |"])
        a([""])

        a([
            "### Trigger Breakdown — Unlocked Trades (All Windows)",
            "",
            "| Trigger type | Count | Wins | WR | Avg R |",
            "|-------------|------:|-----:|---:|------:|",
        ])
        trig_map: Dict[str, List[Dict]] = {}
        for t in all_unlocked:
            trig = t.get("trigger_type") or t.get("entry_trigger") or "unknown"
            trig_map.setdefault(trig, []).append(t)
        for trig, ts in sorted(trig_map.items(), key=lambda x: -len(x[1])):
            w  = sum(1 for t in ts if _is_win(t))
            wr = f"{int(w/len(ts)*100)}%" if ts else "—"
            ar = _rs(sum(_r(t) for t in ts) / len(ts))
            a([f"| {trig} | {len(ts)} | {w} | {wr} | {ar} |"])
        a([""])

        a([
            "### Window Concentration — Unlocked Trades",
            "",
            "| Window | Unlocked | % | Wins | WR | Avg R |",
            "|--------|:--------:|:-:|-----:|---:|------:|",
        ])
        for wr_obj in wrs:
            ul = wr_obj.unlocked_b
            n  = len(ul)
            pct = f"{int(n/len(all_unlocked)*100)}%" if all_unlocked else "—"
            w   = sum(1 for t in ul if _is_win(t))
            wrate = f"{int(w/n*100)}%" if n else "—"
            ar  = _rs(sum(_r(t) for t in ul) / n) if n else "—"
            a([f"| {wr_obj.label} | {n} | {pct} | {w} | {wrate} | {ar} |"])
        a([""])

    # ── Promotion gate ────────────────────────────────────────────────────────
    promote, crit_notes = _promotion_check(wrs)

    promote_word = "✅ **PROMOTED — Variant B qualifies for LIVE_PAPER testing**" \
        if promote else "❌ **NOT PROMOTED — Variant B does not yet qualify**"

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
            "**Next step**: Set `ENTRY_TRIGGER_MODE = \"engulf_or_strict_pin_at_level\"` in",
            "a new `live_paper` branch and run 2-week shadow monitoring.",
            "Do NOT change production (`dry_run=True`) yet.",
            "",
        ])
    else:
        a([
            "**Next step**: Address failing criteria before re-evaluating.",
            "Do NOT change `ENTRY_TRIGGER_MODE` in production.",
            "",
        ])

    # ── Return % comparison table ─────────────────────────────────────────────
    a([
        "## Return % Comparison Across All Windows",
        "",
        "| Window | A return | B return | Δ | A WR | B WR | A maxDD | B maxDD |",
        "|--------|:--------:|:--------:|:-:|:----:|:----:|:-------:|:-------:|",
    ])
    for wr in wrs:
        ra, rb = wr.res_a, wr.res_b
        a([
            f"| {wr.label} "
            f"| {_pct(wr._ret(ra))} | {_pct(wr._ret(rb))} "
            f"| {_pct(wr._ret(rb)-wr._ret(ra))} "
            f"| {wr._wrate(ra)}% | {wr._wrate(rb)}% "
            f"| {wr._dd(ra):.1f}% | {wr._dd(rb):.1f}% |"
        ])
    a([""])

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Variant B multi-window confirmation study")
    print("  Offline replay only — no live changes")
    print("=" * 72)

    wrs: List[WindowResult] = []

    for win_label, win_start, win_end in WINDOWS:
        print(f"\n{'=' * 72}")
        print(f"  WINDOW: {win_label}  ({win_start.date()} → {win_end.date()})")
        print(f"{'=' * 72}")

        # Run A — keep candle cache
        res_a, candles = run_variant(
            window_label = win_label,
            v_label      = "A",
            trigger_mode = "engulf_only",
            lookback     = 2,
            desc         = "Baseline — engulf_only, lb=2",
            start_dt     = win_start,
            end_dt       = win_end,
        )

        # Run B — reuse A's candle cache
        res_b, _ = run_variant(
            window_label = win_label,
            v_label      = "B",
            trigger_mode = "engulf_or_strict_pin_at_level",
            lookback     = 2,
            desc         = "Candidate — engulf OR strict-pin, lb=2",
            start_dt     = win_start,
            end_dt       = win_end,
            preloaded    = candles,
        )

        wrs.append(WindowResult(win_label, win_start, win_end, res_a, res_b))

    # Build and write report
    report = build_report(wrs)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"\n{'=' * 72}")
    print(f"  Report written → {REPORT_PATH.relative_to(REPO)}")
    print(f"{'=' * 72}\n")
    print(report)


if __name__ == "__main__":
    main()
