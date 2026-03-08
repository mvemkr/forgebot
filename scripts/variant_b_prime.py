#!/usr/bin/env python3
"""
Variant B-Prime three-way comparison — offline replay only.

Variants
--------
A        engulf_only / lb=2 / whitelist=None       Baseline (production)
B        engulf_or_strict_pin_at_level / lb=2 / whitelist=None
           Strict-pin fires on ANY pattern
B-Prime  engulf_or_strict_pin_at_level / lb=2 / whitelist=[H&S, IH&S]
           Strict-pin fires ONLY on head_and_shoulders / inverted_head_and_shoulders;
           all other patterns (double_top, double_bottom, break_retest_*) stay engulf-only

Hypothesis
----------
The Q1-2025 regression (-17.9%) and all non-H&S unlock losses in the
extended study were caused by strict-pin misfiring at double_bottom,
inverted_H&S, and break_retest_bullish.  The B-Prime whitelist removes
those false-positive paths while keeping the five profitable H&S unlocks.

Windows (full-year fetch is broken — only sub-period windows run)
----------------------------------------------------------------
  Q1-2025       2025-01-01 → 2025-03-31
  Q2-2025       2025-04-01 → 2025-06-30
  Q3-2025       2025-07-01 → 2025-09-30
  Q4-2025       2025-10-01 → 2025-12-31
  Jan-Feb-2026  2026-01-01 → 2026-02-28
  W1            2026-02-01 → 2026-02-14
  W2            2026-02-15 → 2026-02-28

Output
------
  backtesting/results/variant_b_prime.md

Safety
------
  atexit() resets ENTRY_TRIGGER_MODE, ENGULF_CONFIRM_LOOKBACK_BARS, and
  STRICT_PIN_PATTERN_WHITELIST to production defaults on any exit.
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

# ── atexit guard ──────────────────────────────────────────────────────────────
_ORIG_TRIGGER_MODE = getattr(_sc, "ENTRY_TRIGGER_MODE",           "engulf_only")
_ORIG_LB           = getattr(_sc, "ENGULF_CONFIRM_LOOKBACK_BARS",  2)
_ORIG_SP_WL        = getattr(_sc, "STRICT_PIN_PATTERN_WHITELIST",  None)

def _reset_config():
    _sc.ENTRY_TRIGGER_MODE           = _ORIG_TRIGGER_MODE
    _sc.ENGULF_CONFIRM_LOOKBACK_BARS = _ORIG_LB
    _sc.STRICT_PIN_PATTERN_WHITELIST = _ORIG_SP_WL

atexit.register(_reset_config)

# ── backtester ────────────────────────────────────────────────────────────────
from backtesting.oanda_backtest_v2 import run_backtest, BacktestResult  # noqa

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
CAPITAL = 8_000.0

# Strict-pin is valid at H&S family necklines; blocked everywhere else
HNS_PATTERNS = ["head_and_shoulders", "inverted_head_and_shoulders"]

REPORT_PATH = REPO / "backtesting/results/variant_b_prime.md"

# (label, trigger_mode, lookback, sp_whitelist, short_desc)
VARIANTS: List[Tuple[str, str, int, Optional[List[str]], str]] = [
    ("A",       "engulf_only",                   2, None,        "Baseline — engulf only (production)"),
    ("B",       "engulf_or_strict_pin_at_level", 2, None,        "Broad strict-pin — any pattern"),
    ("B-Prime", "engulf_or_strict_pin_at_level", 2, HNS_PATTERNS, "Strict-pin H&S/IH&S only"),
]

# Sub-period windows only (full-year fetch is unreliable for 14-month spans)
WINDOWS: List[Tuple[str, datetime, datetime]] = [
    ("Q1-2025",      datetime(2025, 1,  1, tzinfo=timezone.utc),
                     datetime(2025, 3, 31, tzinfo=timezone.utc)),
    ("Q2-2025",      datetime(2025, 4,  1, tzinfo=timezone.utc),
                     datetime(2025, 6, 30, tzinfo=timezone.utc)),
    ("Q3-2025",      datetime(2025, 7,  1, tzinfo=timezone.utc),
                     datetime(2025, 9, 30, tzinfo=timezone.utc)),
    ("Q4-2025",      datetime(2025, 10, 1, tzinfo=timezone.utc),
                     datetime(2025, 12, 31, tzinfo=timezone.utc)),
    ("Jan-Feb-2026", datetime(2026, 1,  1, tzinfo=timezone.utc),
                     datetime(2026, 2, 28, tzinfo=timezone.utc)),
    ("W1",           datetime(2026, 2,  1, tzinfo=timezone.utc),
                     datetime(2026, 2, 14, tzinfo=timezone.utc)),
    ("W2",           datetime(2026, 2, 15, tzinfo=timezone.utc),
                     datetime(2026, 2, 28, tzinfo=timezone.utc)),
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _trade_key(t: Dict) -> str:
    ts = t.get("entry_ts") or t.get("open_ts") or ""
    if hasattr(ts, "strftime"):
        ts = ts.strftime("%Y%m%d%H")
    else:
        ts = str(ts)[:13].replace("-","").replace("T","").replace(":","").replace(" ","")
    return f"{t.get('pair','?')}|{ts}"


def _r(t: Dict) -> float:
    for k in ("r", "realised_r", "result_r"):
        v = t.get(k)
        if v is not None:
            try: return float(v)
            except (TypeError, ValueError): pass
    return 0.0


def _is_win(t: Dict) -> bool:
    return _r(t) > 0.0


def _pct(v: float) -> str:
    return f"{v:+.1f}%"


def _rs(v: float) -> str:
    return f"{v:+.2f}R"


def _wr(trades: List[Dict]) -> str:
    if not trades: return "—"
    w = sum(1 for t in trades if _is_win(t))
    return f"{int(w/len(trades)*100)}%"


def _avg_r(trades: List[Dict]) -> float:
    if not trades: return 0.0
    return sum(_r(t) for t in trades) / len(trades)


def _worst3(trades: List[Dict]) -> float:
    return sum(sorted(_r(t) for t in trades)[:3])


def _mae(t: Dict) -> Optional[float]:
    for k in ("mae_r","mae","max_adverse_excursion"):
        v = t.get(k)
        if v is not None:
            try: return float(v)
            except: pass
    return None


def _mfe(t: Dict) -> Optional[float]:
    for k in ("mfe_r","mfe","max_favorable_excursion"):
        v = t.get(k)
        if v is not None:
            try: return float(v)
            except: pass
    return None


def _signal_type(t: Dict) -> str:
    return (t.get("signal_type") or t.get("trigger_type") or t.get("entry_trigger") or "unknown")


def _pattern(t: Dict) -> str:
    return (t.get("pattern") or t.get("pattern_type") or "unknown")


def _is_strict_pin(t: Dict) -> bool:
    st = _signal_type(t).lower()
    return "shooting_star" in st or "hammer_strict" in st


def _is_hns_pattern(t: Dict) -> bool:
    p = _pattern(t).lower()
    return "head_and_shoulders" in p


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_variant(
    win_label: str,
    v_label: str,
    trigger_mode: str,
    lookback: int,
    sp_whitelist: Optional[List[str]],
    start_dt: datetime,
    end_dt: datetime,
    preloaded: Optional[Dict] = None,
) -> Tuple[BacktestResult, Optional[Dict]]:
    _sc.ENTRY_TRIGGER_MODE           = trigger_mode
    _sc.ENGULF_CONFIRM_LOOKBACK_BARS = lookback
    _sc.STRICT_PIN_PATTERN_WHITELIST = sp_whitelist

    wl_tag = f"wl={sp_whitelist}" if sp_whitelist else "wl=None"
    print(f"  [{win_label}] Var {v_label}: {trigger_mode!r} {wl_tag}", end="", flush=True)

    t0 = time.time()
    result = run_backtest(
        start_dt              = start_dt,
        end_dt                = end_dt,
        starting_bal          = CAPITAL,
        notes                 = f"bprime_{win_label}_{v_label}",
        trail_arm_key         = f"bp_{win_label}_{v_label}",
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
# WindowTriple: holds A, B, B-Prime for one window
# ──────────────────────────────────────────────────────────────────────────────

class WindowTriple:
    def __init__(
        self,
        label: str,
        start_dt: datetime,
        end_dt: datetime,
        res_a: BacktestResult,
        res_b: BacktestResult,
        res_bp: BacktestResult,
    ):
        self.label    = label
        self.start_dt = start_dt
        self.end_dt   = end_dt
        self.res_a    = res_a
        self.res_b    = res_b
        self.res_bp   = res_bp

        def _keys(r): return {_trade_key(t): t for t in (r.trades or [])}
        a_keys  = _keys(res_a)
        b_keys  = _keys(res_b)
        bp_keys = _keys(res_bp)

        # B unlocks vs A
        self.unlocked_b:      List[Dict] = [b_keys[k]  for k in b_keys  if k not in a_keys]
        # B-Prime unlocks vs A
        self.unlocked_bp:     List[Dict] = [bp_keys[k] for k in bp_keys if k not in a_keys]
        # In B but not in B-Prime (removed by whitelist)
        self.removed_by_wl:   List[Dict] = [b_keys[k]  for k in b_keys  if k not in bp_keys and k not in a_keys]
        # In A but not in B-Prime (locked out by B-Prime — should be empty if B-Prime only adds)
        self.locked_by_bp:    List[Dict] = [a_keys[k]  for k in a_keys  if k not in bp_keys]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ret(self, r: BacktestResult) -> float: return r.return_pct
    def _dd(self,  r: BacktestResult) -> float: return r.max_dd_pct
    def _wr(self,  r: BacktestResult) -> int:
        return int(round(getattr(r, "win_rate", 0.0) * 100))
    def _avgr(self, r: BacktestResult) -> float:
        return getattr(r, "avg_r", 0.0) or _avg_r(r.trades or [])
    def _w3l(self,  r: BacktestResult) -> float:
        return _worst3(r.trades or [])
    def _n(self,    r: BacktestResult) -> int:
        return r.n_trades

    def unlock_mae_mfe(self, trades: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        maes = [_mae(t) for t in trades if _mae(t) is not None]
        mfes = [_mfe(t) for t in trades if _mfe(t) is not None]
        return (
            _rs(sum(maes)/len(maes)) if maes else None,
            _rs(sum(mfes)/len(mfes)) if mfes else None,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Report builder
# ──────────────────────────────────────────────────────────────────────────────

def build_report(triples: List[WindowTriple]) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = []
    a = lines.extend

    # ── Header ───────────────────────────────────────────────────────────────
    a([
        "# Variant B-Prime Three-Way Comparison",
        f"Generated: {now_utc}",
        "",
        "## Variants",
        "",
        "| Variant | Trigger mode | LB | Strict-pin whitelist |",
        "|---------|-------------|----|----------------------|",
        "| A | `engulf_only` | 2 | — (production) |",
        "| B | `engulf_or_strict_pin_at_level` | 2 | None — any pattern |",
        "| B-Prime | `engulf_or_strict_pin_at_level` | 2 | `head_and_shoulders`, `inverted_head_and_shoulders` only |",
        "",
        "> Full-year window (2025-01-01→2026-03-01) omitted — backtester returns",
        "> 0 trades for 14-month spans. Sub-period data is the reliable dataset.",
        "",
        "---",
        "",
    ])

    # ── Section 1: Per-window three-way table ─────────────────────────────────
    a([
        "## Per-Window Three-Way Comparison",
        "",
        "| Window | A ret | A WR | A DD | B ret | B WR | B DD | B' ret | B' WR | B' DD |",
        "|--------|:-----:|:----:|:----:|:-----:|:----:|:----:|:------:|:-----:|:-----:|",
    ])
    for t in triples:
        a([
            f"| {t.label} "
            f"| {_pct(t._ret(t.res_a))} | {t._wr(t.res_a)}% | {t._dd(t.res_a):.1f}% "
            f"| {_pct(t._ret(t.res_b))} | {t._wr(t.res_b)}% | {t._dd(t.res_b):.1f}% "
            f"| {_pct(t._ret(t.res_bp))} | {t._wr(t.res_bp)}% | {t._dd(t.res_bp):.1f}% |"
        ])
    a([""])

    # ── Section 2: Unlock count comparison ───────────────────────────────────
    a([
        "## Unlock Counts per Window",
        "",
        "| Window | B ul vs A | B' ul vs A | Removed by WL | B' vs B Δ ret |",
        "|--------|:---------:|:----------:|:-------------:|:-------------:|",
    ])
    for t in triples:
        delta_ret_bp_vs_b = t._ret(t.res_bp) - t._ret(t.res_b)
        a([
            f"| {t.label} "
            f"| +{len(t.unlocked_b)} "
            f"| +{len(t.unlocked_bp)} "
            f"| -{len(t.removed_by_wl)} "
            f"| {_pct(delta_ret_bp_vs_b)} |"
        ])
    a([""])

    # ── Section 3: Per-window detail ─────────────────────────────────────────
    a(["## Per-Window Detail", ""])

    for t in triples:
        a([
            f"### {t.label}  ({t.start_dt.date()} → {t.end_dt.date()})",
            "",
            "| Metric | A | B | B-Prime | B' vs A | B' vs B |",
            "|--------|:-:|:-:|:-------:|:-------:|:-------:|",
            f"| Trades | {t._n(t.res_a)} | {t._n(t.res_b)} | {t._n(t.res_bp)} "
            f"| {t._n(t.res_bp)-t._n(t.res_a):+d} | {t._n(t.res_bp)-t._n(t.res_b):+d} |",
            f"| Win rate | {t._wr(t.res_a)}% | {t._wr(t.res_b)}% | {t._wr(t.res_bp)}% "
            f"| {t._wr(t.res_bp)-t._wr(t.res_a):+d}pp | {t._wr(t.res_bp)-t._wr(t.res_b):+d}pp |",
            f"| Avg R | {_rs(t._avgr(t.res_a))} | {_rs(t._avgr(t.res_b))} | {_rs(t._avgr(t.res_bp))} | — | — |",
            f"| Return % | {_pct(t._ret(t.res_a))} | {_pct(t._ret(t.res_b))} | {_pct(t._ret(t.res_bp))} "
            f"| {_pct(t._ret(t.res_bp)-t._ret(t.res_a))} | {_pct(t._ret(t.res_bp)-t._ret(t.res_b))} |",
            f"| Max DD | {t._dd(t.res_a):.1f}% | {t._dd(t.res_b):.1f}% | {t._dd(t.res_bp):.1f}% "
            f"| {t._dd(t.res_bp)-t._dd(t.res_a):+.1f}pp | {t._dd(t.res_bp)-t._dd(t.res_b):+.1f}pp |",
            f"| Worst-3L | {_rs(t._w3l(t.res_a))} | {_rs(t._w3l(t.res_b))} | {_rs(t._w3l(t.res_bp))} | — | — |",
            "",
        ])

        # B-Prime unlocked trades
        if t.unlocked_bp:
            a([
                "#### Trades unlocked by B-Prime (vs A)",
                "",
                "| # | Pair | Dir | Entry | Pattern | Signal | R | MAE | MFE |",
                "|---|------|-----|-------|---------|--------|---|-----|-----|",
            ])
            for i, tr in enumerate(t.unlocked_bp, 1):
                mae_s = _rs(_mae(tr)) if _mae(tr) is not None else "n/a"
                mfe_s = _rs(_mfe(tr)) if _mfe(tr) is not None else "n/a"
                hns_mark = "✅ H&S" if _is_hns_pattern(tr) else "⚠️ non-H&S"
                a([
                    f"| {i} | {tr.get('pair','?')} | {tr.get('direction','?')} "
                    f"| {str(tr.get('entry_ts','?'))[:16]} "
                    f"| {_pattern(tr)} {hns_mark} "
                    f"| `{_signal_type(tr)}` | {_rs(_r(tr))} | {mae_s} | {mfe_s} |"
                ])
            mae_u, mfe_u = t.unlock_mae_mfe(t.unlocked_bp)
            a(["", f"Avg MAE: {mae_u or 'n/a'} | Avg MFE: {mfe_u or 'n/a'}", ""])
        else:
            a(["*No new trades unlocked by B-Prime in this window.*", ""])

        # Removed-by-whitelist trades (B had them, B-Prime doesn't)
        if t.removed_by_wl:
            a([
                "#### Trades REMOVED by B-Prime whitelist (were in B, blocked in B')",
                "",
                "| # | Pair | Dir | Entry | Pattern | Signal | R |",
                "|---|------|-----|-------|---------|--------|---|",
            ])
            for i, tr in enumerate(t.removed_by_wl, 1):
                hns_mark = "H&S ✅" if _is_hns_pattern(tr) else "non-H&S 🚫"
                a([
                    f"| {i} | {tr.get('pair','?')} | {tr.get('direction','?')} "
                    f"| {str(tr.get('entry_ts','?'))[:16]} "
                    f"| {_pattern(tr)} ({hns_mark}) "
                    f"| `{_signal_type(tr)}` | {_rs(_r(tr))} |"
                ])
            a([""])
        a(["---", ""])

    # ── Section 4: Aggregate unlock summary ──────────────────────────────────
    all_ul_b  = [t for wt in triples for t in wt.unlocked_b]
    all_ul_bp = [t for wt in triples for t in wt.unlocked_bp]
    all_rmv   = [t for wt in triples for t in wt.removed_by_wl]

    a([
        "## Aggregate Unlock Summary (All Windows)",
        "",
        "| Metric | B vs A | B-Prime vs A | Removed by WL |",
        "|--------|:------:|:------------:|:-------------:|",
        f"| Total trades unlocked | {len(all_ul_b)} | {len(all_ul_bp)} | {len(all_rmv)} |",
    ])

    if all_ul_b:
        w_b = sum(1 for t in all_ul_b if _is_win(t))
        a([f"| Win rate | {int(w_b/len(all_ul_b)*100)}% | ", ])

    if all_ul_bp:
        w_bp = sum(1 for t in all_ul_bp if _is_win(t))
        a([f"| B-Prime unlock WR | — | {int(w_bp/len(all_ul_bp)*100)}% | — |"])
        a([f"| B-Prime unlock avg R | — | {_rs(_avg_r(all_ul_bp))} | — |"])

    if all_rmv:
        w_rm = sum(1 for t in all_rmv if _is_win(t))
        a([f"| Removed trades WR | — | — | {int(w_rm/len(all_rmv)*100)}% |"])
        a([f"| Removed trades avg R | — | — | {_rs(_avg_r(all_rmv))} |"])
    a([""])

    # ── Section 5: Pattern split of unlocked trades ───────────────────────────
    if all_ul_bp:
        hns_ul  = [t for t in all_ul_bp if _is_hns_pattern(t)]
        nhns_ul = [t for t in all_ul_bp if not _is_hns_pattern(t)]
        a([
            "### B-Prime Unlocked — H&S vs non-H&S split",
            "",
            "| Category | Count | WR | Avg R | Avg MAE | Avg MFE |",
            "|----------|:-----:|:--:|:-----:|--------:|--------:|",
        ])
        for cat_label, cat in [("H&S / IH&S", hns_ul), ("Non-H&S", nhns_ul)]:
            if cat:
                w   = sum(1 for t in cat if _is_win(t))
                wr  = f"{int(w/len(cat)*100)}%"
                ar  = _rs(_avg_r(cat))
                maes = [_mae(t) for t in cat if _mae(t) is not None]
                mfes = [_mfe(t) for t in cat if _mfe(t) is not None]
                mae_s = _rs(sum(maes)/len(maes)) if maes else "n/a"
                mfe_s = _rs(sum(mfes)/len(mfes)) if mfes else "n/a"
                a([f"| {cat_label} | {len(cat)} | {wr} | {ar} | {mae_s} | {mfe_s} |"])
        a([""])

    # ── Section 6: Regime summary table ──────────────────────────────────────
    a([
        "## Regime Summary (B-Prime vs A)",
        "",
        "| Period | A ret | B' ret | Δ | Assessment |",
        "|--------|:-----:|:------:|:-:|-----------|",
    ])
    for t in triples:
        d = t._ret(t.res_bp) - t._ret(t.res_a)
        if abs(d) < 0.05:
            verdict = "🟡 neutral"
        elif d > 0:
            verdict = "✅ improved"
        else:
            verdict = "❌ degraded"
        a([
            f"| {t.label} "
            f"| {_pct(t._ret(t.res_a))} | {_pct(t._ret(t.res_bp))} "
            f"| {_pct(d)} | {verdict} |"
        ])
    a([""])

    # ── Section 7: Key questions ──────────────────────────────────────────────
    # Q1 regression
    q1_triples = [t for t in triples if t.label == "Q1-2025"]
    q1_b_ret  = q1_triples[0]._ret(q1_triples[0].res_b)  if q1_triples else None
    q1_bp_ret = q1_triples[0]._ret(q1_triples[0].res_bp) if q1_triples else None
    q1_a_ret  = q1_triples[0]._ret(q1_triples[0].res_a)  if q1_triples else None
    q1_fixed  = (q1_bp_ret is not None and q1_a_ret is not None
                 and q1_bp_ret >= q1_a_ret - 0.1)

    # Jan-Feb gains preserved
    jf_triples = [t for t in triples if t.label == "Jan-Feb-2026"]
    jf_gained  = (jf_triples and
                  jf_triples[0]._ret(jf_triples[0].res_bp) >=
                  jf_triples[0]._ret(jf_triples[0].res_a) - 0.1)

    # W1 gains preserved
    w1_triples = [t for t in triples if t.label == "W1"]
    w1_gained  = (w1_triples and
                  w1_triples[0]._ret(w1_triples[0].res_bp) >=
                  w1_triples[0]._ret(w1_triples[0].res_a) - 0.1)

    a([
        "## Key Questions",
        "",
        "| Question | Answer |",
        "|----------|--------|",
        f"| Does Q1 regression disappear? | {'✅ YES' if q1_fixed else '❌ NO'} "
        f"(B={_pct(q1_b_ret) if q1_b_ret is not None else '?'} → "
        f"B'={_pct(q1_bp_ret) if q1_bp_ret is not None else '?'} "
        f"vs A={_pct(q1_a_ret) if q1_a_ret is not None else '?'}) |",
        f"| Jan-Feb-2026 gains preserved? | {'✅ YES' if jf_gained else '❌ NO'} |",
        f"| W1 gains preserved? | {'✅ YES' if w1_gained else '❌ NO'} |",
        f"| Non-H&S losses removed by whitelist? | {len(all_rmv)} trades removed, "
        f"WR was {int(sum(1 for t in all_rmv if _is_win(t))/len(all_rmv)*100) if all_rmv else 0}% |",
        f"| B-Prime unlocked H&S trades WR | "
        f"{_wr([t for t in all_ul_bp if _is_hns_pattern(t)]) if all_ul_bp else '—'} |",
        "",
    ])

    # ── Section 8: Return comparison — all three variants ─────────────────────
    total_a  = sum(t._n(t.res_a) for t in triples)
    total_b  = sum(t._n(t.res_b) for t in triples)
    total_bp = sum(t._n(t.res_bp) for t in triples)

    a([
        "## Full Return Comparison (All Three Variants, All Windows)",
        "",
        "| Window | A | B | B' | B'−A | B'−B |",
        "|--------|:-:|:-:|:--:|:----:|:----:|",
    ])
    for t in triples:
        a([
            f"| {t.label} "
            f"| {_pct(t._ret(t.res_a))} | {_pct(t._ret(t.res_b))} | {_pct(t._ret(t.res_bp))} "
            f"| {_pct(t._ret(t.res_bp)-t._ret(t.res_a))} "
            f"| {_pct(t._ret(t.res_bp)-t._ret(t.res_b))} |"
        ])
    a([
        f"| **Total trades** | {total_a} | {total_b} | {total_bp} | — | — |",
        "",
    ])

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Variant B-Prime three-way comparison — offline replay only")
    print(f"  {len(WINDOWS)} windows × {len(VARIANTS)} variants = {len(WINDOWS)*len(VARIANTS)} runs")
    print("=" * 72)

    triples: List[WindowTriple] = []

    for win_label, win_start, win_end in WINDOWS:
        print(f"\n── {win_label} ({win_start.date()} → {win_end.date()}) ──")

        # Run A — capture candle cache
        res_a, candles = run_variant(
            win_label, "A", "engulf_only", 2, None,
            win_start, win_end,
        )
        # Run B — reuse candle cache
        res_b, _ = run_variant(
            win_label, "B", "engulf_or_strict_pin_at_level", 2, None,
            win_start, win_end, preloaded=candles,
        )
        # Run B-Prime — reuse candle cache
        res_bp, _ = run_variant(
            win_label, "B-Prime", "engulf_or_strict_pin_at_level", 2, HNS_PATTERNS,
            win_start, win_end, preloaded=candles,
        )

        triples.append(WindowTriple(win_label, win_start, win_end, res_a, res_b, res_bp))

    report = build_report(triples)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)

    print(f"\n{'=' * 72}")
    print(f"  Report → {REPORT_PATH.relative_to(REPO)}")
    print("=" * 72)
    print()
    print(report)


if __name__ == "__main__":
    main()
