#!/usr/bin/env python3
"""
Confidence Threshold Ablation Study — offline replay only.

Variants
--------
A  MIN_CONFIDENCE = 0.77  (baseline / current production)
B  MIN_CONFIDENCE = 0.73  (moderate relaxation)
C  MIN_CONFIDENCE = 0.70  (lower bound)

All else unchanged across variants:
  - MIN_RR_STANDARD = 2.5
  - C8 structural stop (targeting.py _MAX_FRAC_ATR=3.0, _MIN_ABS_PIPS=8.0)
  - B-Prime trigger config (engulf_or_strict_pin_at_level)
  - ENGULF_CONFIRM_LOOKBACK_BARS = 2
  - STRICT_PIN_PATTERN_WHITELIST = [head_and_shoulders, inverted_head_and_shoulders]
  - session filter, zone-touch logic, weekly cap

Windows (same as prior ablations)
----------------------------------
  Q1-2025  Q2-2025  Q3-2025  Q4-2025
  Jan-Feb-2026  W1  W2  live-parity

Safety
------
  atexit() restores MIN_CONFIDENCE to production default even on crash.
  No live changes, no master commits.
"""
from __future__ import annotations

import atexit
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

import src.strategy.forex.strategy_config as _sc

# ── atexit guard ──────────────────────────────────────────────────────────────
_ORIG_CONF = float(getattr(_sc, "MIN_CONFIDENCE", 0.77))

def _reset_conf():
    _sc.MIN_CONFIDENCE = _ORIG_CONF

atexit.register(_reset_conf)

# ── backtester import ─────────────────────────────────────────────────────────
from backtesting.oanda_backtest_v2 import run_backtest, BacktestResult, DECISION_LOG  # noqa

# ─────────────────────────────────────────────────────────────────────────────
_UTC = timezone.utc

WINDOWS: List[Tuple[str, datetime, datetime]] = [
    ("Q1-2025",      datetime(2025,  1,  1, tzinfo=_UTC), datetime(2025,  3, 31, tzinfo=_UTC)),
    ("Q2-2025",      datetime(2025,  4,  1, tzinfo=_UTC), datetime(2025,  6, 30, tzinfo=_UTC)),
    ("Q3-2025",      datetime(2025,  7,  1, tzinfo=_UTC), datetime(2025,  9, 30, tzinfo=_UTC)),
    ("Q4-2025",      datetime(2025, 10,  1, tzinfo=_UTC), datetime(2025, 12, 31, tzinfo=_UTC)),
    ("Jan-Feb-2026", datetime(2026,  1,  1, tzinfo=_UTC), datetime(2026,  2, 28, tzinfo=_UTC)),
    ("W1",           datetime(2026,  2, 17, tzinfo=_UTC), datetime(2026,  2, 21, tzinfo=_UTC)),
    ("W2",           datetime(2026,  2, 24, tzinfo=_UTC), datetime(2026,  2, 28, tzinfo=_UTC)),
    ("live-parity",  datetime(2026,  3,  2, tzinfo=_UTC), datetime(2026,  3,  8, tzinfo=_UTC)),
]

CAPITAL   = 8_000.0
VARIANTS  = [
    ("A", 0.77, "Baseline — current production"),
    ("B", 0.73, "Moderate relaxation"),
    ("C", 0.70, "Lower bound"),
]

REPORT_PATH = REPO / "backtesting/results/ablation_confidence_threshold.md"
DECISION_LOG_PATH = Path(str(DECISION_LOG))


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WindowResult:
    variant:     str
    window:      str
    threshold:   float
    result:      BacktestResult
    candle_data: Optional[dict]
    # ENTER decisions with confidence, loaded from DECISION_LOG after each run
    enter_decisions: List[dict] = field(default_factory=list)

    @property
    def trades(self) -> List[dict]:
        return self.result.trades or []

    @property
    def n(self) -> int:
        return self.result.n_trades

    @property
    def total_r(self) -> float:
        return sum(_r(t) for t in self.trades)

    @property
    def wr(self) -> float:
        return self.result.win_rate

    @property
    def avg_r(self) -> float:
        return self.result.avg_r

    @property
    def max_dd(self) -> float:
        return self.result.max_dd_pct

    @property
    def ret(self) -> float:
        return self.result.return_pct

    @property
    def worst3(self) -> List[float]:
        return sorted(_r(t) for t in self.trades)[:3]

    @property
    def expectancy(self) -> float:
        rs = [_r(t) for t in self.trades]
        return sum(rs) / len(rs) if rs else 0.0

    def trade_key(self, t: dict) -> str:
        return _trade_key(t)

    def trade_map(self) -> dict:
        return {_trade_key(t): t for t in self.trades}

    def decision_conf_map(self) -> dict:
        """Map entry_ts[:13]+pair → confidence from ENTER decisions."""
        out = {}
        for d in self.enter_decisions:
            ts = str(d.get("ts", ""))[:13]
            pair = d.get("pair", "")
            out[f"{pair}|{ts}"] = float(d.get("confidence", 0.0))
        return out

    def stop_floor_violations(self) -> List[dict]:
        """Trades where initial_stop_pips < 8 — C8 floor should prevent these."""
        return [t for t in self.trades
                if 0 < (t.get("initial_stop_pips") or 999) < 8.0]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _r(t: dict) -> float:
    for k in ("r", "realised_r", "result_r"):
        v = t.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def _is_win(t: dict) -> bool:
    return _r(t) > 0.0


def _trade_key(t: dict) -> str:
    ts = t.get("entry_ts") or t.get("open_ts") or ""
    if hasattr(ts, "strftime"):
        ts = ts.strftime("%Y%m%d%H")
    else:
        ts = str(ts)[:13].replace("-", "").replace("T", "").replace(":", "").replace(" ", "")
    return f"{t.get('pair', '?')}|{ts}"


def _load_enter_decisions() -> List[dict]:
    """Read all ENTER decisions from the last run's DECISION_LOG."""
    try:
        with open(DECISION_LOG_PATH) as f:
            data = json.load(f)
        return [d for d in data.get("decisions", []) if d.get("decision") == "ENTER"]
    except Exception:
        return []


def _unlocked(base: WindowResult, other: WindowResult) -> List[dict]:
    """Trades present in other but not in base (newly unlocked by looser threshold)."""
    base_keys = set(base.trade_map().keys())
    return [t for t in other.trades if _trade_key(t) not in base_keys]


def _displaced(base: WindowResult, other: WindowResult) -> List[Tuple[dict, Optional[dict]]]:
    """
    Baseline trades absent from other (displaced by weekly cap when a new
    unlocked trade consumed the slot).
    Returns list of (displaced_base_trade, replacing_trade_or_None).
    """
    other_keys = set(other.trade_map().keys())
    displaced = [t for t in base.trades if _trade_key(t) not in other_keys]
    other_map = other.trade_map()
    # Best-effort: match by week block
    result = []
    for bt in displaced:
        ts = str(bt.get("entry_ts", ""))
        try:
            wk = datetime.fromisoformat(ts.replace("Z", "+00:00")).isocalendar()[:2]
        except Exception:
            wk = None
        # find an unlocked trade in the same ISO week
        replacement = None
        if wk:
            for ut in _unlocked(base, other):
                try:
                    uts = str(ut.get("entry_ts", ""))
                    uwk = datetime.fromisoformat(uts.replace("Z", "+00:00")).isocalendar()[:2]
                    if uwk == wk:
                        replacement = ut
                        break
                except Exception:
                    pass
        result.append((bt, replacement))
    return result


def _conf_for_trade(wr: WindowResult, t: dict) -> Optional[float]:
    """Look up confidence for a trade from the ENTER decision log."""
    ts = str(t.get("entry_ts", ""))[:13]
    pair = t.get("pair", "")
    return wr.decision_conf_map().get(f"{pair}|{ts}")


def _flag_unlocked(unlocked_trades: List[dict], window_sum_r: float,
                   conf_map: dict, threshold: float) -> List[str]:
    """Return list of flag strings for the unlocked set."""
    flags = []
    for t in unlocked_trades:
        tr = _r(t)
        pair = t.get("pair", "?")
        pat  = t.get("pattern", "?")
        key  = _trade_key(t)
        conf = conf_map.get(key)
        # Flag 1: low confidence loss
        if conf is not None and conf < 0.72 and tr < 0.0:
            flags.append(
                f"⚠️  LOW_CONF_LOSS: {pair} {pat} conf={conf:.3f} < 0.72 → {tr:+.2f}R"
            )
        # Flag 2: single trade >20% of window SumR
        if window_sum_r != 0 and abs(tr) / abs(window_sum_r) > 0.20:
            pct = tr / window_sum_r * 100
            flags.append(
                f"⚠️  HIGH_CONCENTRATION: {pair} {pat} = {pct:+.0f}% of window SumR ({tr:+.2f}R / {window_sum_r:+.2f}R)"
            )
    # Flag 3: MAE > MFE pattern in unlocked set
    mae_wins = sum(1 for t in unlocked_trades
                   if abs(t.get("mae_r", 0.0)) > t.get("mfe_r", 0.0) and _is_win(t))
    if len(unlocked_trades) >= 3 and mae_wins / len(unlocked_trades) > 0.50:
        flags.append(
            f"⚠️  MAE>MFE_PATTERN: {mae_wins}/{len(unlocked_trades)} unlocked winners "
            f"have |MAE| > MFE — suggests marginal entry quality"
        )
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_variant(
    var_id: str,
    threshold: float,
    desc: str,
    win_name: str,
    start: datetime,
    end: datetime,
    preloaded: Optional[dict],
) -> WindowResult:
    print(f"  [{win_name}] Variant {var_id} (MIN_CONF={threshold:.2f}) … ", end="", flush=True)
    _sc.MIN_CONFIDENCE = threshold
    t0 = time.time()
    result = run_backtest(
        start_dt=start,
        end_dt=end,
        starting_bal=CAPITAL,
        notes=f"conf_ablation_{var_id}_{win_name}",
        trail_arm_key="A",   # standard trail arm — unchanged
        preloaded_candle_data=preloaded,
        use_cache=True,
        quiet=True,
    )
    elapsed = time.time() - t0
    _sc.MIN_CONFIDENCE = _ORIG_CONF   # reset immediately

    enter_decisions = _load_enter_decisions()
    candles = getattr(result, "candle_data", None) or preloaded

    print(f"T={result.n_trades} WR={result.win_rate:.0%} SumR={sum(_r(t) for t in (result.trades or [])):+.2f}R "
          f"DD={result.max_dd_pct:.1f}% ({elapsed:.1f}s)")

    return WindowResult(
        variant=var_id,
        window=win_name,
        threshold=threshold,
        result=result,
        candle_data=candles,
        enter_decisions=enter_decisions,
    )


def run_all() -> Dict[str, List[WindowResult]]:
    """Run all variants across all windows. Returns {variant_id: [WindowResult, ...]}."""
    results: Dict[str, List[WindowResult]] = {v[0]: [] for v in VARIANTS}

    for win_name, start, end in WINDOWS:
        print(f"\n{'═'*72}")
        print(f"  Window: {win_name}  ({start.date()} → {end.date()})")
        print(f"{'═'*72}")
        preloaded = None
        for var_id, threshold, desc in VARIANTS:
            wr = run_variant(var_id, threshold, desc, win_name, start, end, preloaded)
            results[var_id].append(wr)
            if preloaded is None and wr.candle_data:
                preloaded = wr.candle_data

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def build_report(results: Dict[str, List[WindowResult]], windows=None) -> str:
    if windows is None:
        windows = WINDOWS
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = []
    a = lines.append

    a(f"# Confidence Threshold Ablation Study")
    a(f"")
    a(f"Generated: {now}  |  Branch: `feat/confidence-threshold-ablation`")
    a(f"")
    a(f"## Setup")
    a(f"")
    a(f"| | |")
    a(f"|---|---|")
    a(f"| Capital | $8,000 |")
    a(f"| Stop logic | C8 (structural + 3×ATR_1H ceiling + 8-pip floor) |")
    a(f"| Trigger mode | `engulf_or_strict_pin_at_level` (B-Prime) |")
    a(f"| ENGULF_CONFIRM_LOOKBACK_BARS | 2 |")
    a(f"| STRICT_PIN_PATTERN_WHITELIST | head_and_shoulders, inverted_head_and_shoulders |")
    a(f"| MIN_RR_STANDARD | 2.5 |")
    a(f"| ATR_MIN_MULTIPLIER | 0.0 (C8: 8-pip hard floor active) |")
    a(f"")
    a(f"### Variants")
    a(f"")
    a(f"| Variant | MIN_CONFIDENCE | Description |")
    a(f"|---------|:--------------:|-------------|")
    for var_id, thr, desc in VARIANTS:
        a(f"| **{var_id}** | {thr:.2f} | {desc} |")
    a(f"")
    a(f"### Windows")
    a(f"")
    for win_name, start, end in windows:
        a(f"- **{win_name}**: {start.date()} → {end.date()}")
    a(f"")

    # ── Section 1: Aggregate summary tables ────────────────────────────────
    a(f"---")
    a(f"")
    a(f"## 1. Aggregate Summary")
    a(f"")
    for var_id, thr, desc in VARIANTS:
        wrs = results[var_id]
        all_trades = [t for wr in wrs for t in wr.trades]
        total_r    = sum(_r(t) for t in all_trades)
        n_trades   = sum(wr.n for wr in wrs)
        wins       = sum(1 for t in all_trades if _is_win(t))
        wr_agg     = wins / n_trades if n_trades else 0.0
        avg_r_agg  = total_r / n_trades if n_trades else 0.0
        exp_agg    = avg_r_agg
        avg_dd     = sum(wr.max_dd for wr in wrs) / len(wrs) if wrs else 0.0
        worst3_all = sorted(_r(t) for t in all_trades)[:3]
        a(f"### Variant {var_id} — MIN_CONFIDENCE = {thr:.2f}")
        a(f"")
        a(f"| Metric | Value |")
        a(f"|--------|-------|")
        a(f"| Total trades (all windows) | {n_trades} |")
        a(f"| Win rate | {wr_agg:.0%} |")
        a(f"| Total SumR | {total_r:+.2f}R |")
        a(f"| Avg R/trade | {avg_r_agg:+.3f}R |")
        a(f"| Expectancy | {exp_agg:+.3f}R |")
        a(f"| Avg max DD (per window) | {avg_dd:.1f}% |")
        a(f"| Worst 3 losses | {', '.join(f'{r:+.2f}R' for r in worst3_all)} |")
        a(f"")

    # ── Section 2: Per-window breakdown ───────────────────────────────────
    a(f"---")
    a(f"")
    a(f"## 2. Per-Window Breakdown")
    a(f"")
    a(f"| Window | Var | T | WR | SumR | AvgR | Exp | MaxDD | Worst3L |")
    a(f"|--------|-----|:--:|:--:|:----:|:----:|:---:|:-----:|---------|")
    for i, (win_name, _, _) in enumerate(windows):
        for var_id, thr, _ in VARIANTS:
            wr = results[var_id][i]
            w3 = ", ".join(f"{r:+.2f}R" for r in wr.worst3[:3]) or "—"
            a(f"| {win_name} | {var_id} ({thr:.2f}) | {wr.n} | "
              f"{wr.wr:.0%} | {wr.total_r:+.2f}R | {wr.avg_r:+.3f}R | "
              f"{wr.expectancy:+.3f}R | {wr.max_dd:.1f}% | {w3} |")
        a(f"| | | | | | | | | |")

    # ── Section 3: ATR floor check ────────────────────────────────────────
    a(f"---")
    a(f"")
    a(f"## 3. ATR Floor Check (C8 8-pip floor)")
    a(f"")
    a(f"Counts trades with `initial_stop_pips < 8.0` — any non-zero count indicates")
    a(f"the floor is not fully enforced in that window.")
    a(f"")
    a(f"| Window | Var A violations | Var B violations | Var C violations |")
    a(f"|--------|:----------------:|:----------------:|:----------------:|")
    total_violations = {v[0]: 0 for v in VARIANTS}
    for i, (win_name, _, _) in enumerate(windows):
        row_parts = [win_name]
        for var_id, _, _ in VARIANTS:
            viol = results[var_id][i].stop_floor_violations()
            total_violations[var_id] += len(viol)
            row_parts.append(str(len(viol)) if viol else "0 ✅")
        a(f"| {' | '.join(row_parts)} |")
    totals_row = ["**Total**"] + [
        (f"**{total_violations[v[0]]}**" if total_violations[v[0]] else "**0 ✅**")
        for v in VARIANTS
    ]
    a(f"| {' | '.join(totals_row)} |")
    a(f"")

    # ── Section 4: Unlocked trade analysis (B vs A, C vs A) ──────────────
    a(f"---")
    a(f"")
    a(f"## 4. Unlocked Trade Analysis")
    a(f"")
    for cmp_id, cmp_thr, _ in VARIANTS[1:]:   # B and C vs A
        base_id = "A"
        a(f"### Variant {cmp_id} vs A — Newly unlocked trades (conf ≥ {cmp_thr:.2f}, < {_ORIG_CONF:.2f})")
        a(f"")
        all_unlocked = []
        all_flags    = []
        for i, (win_name, _, _) in enumerate(windows):
            base_wr = results[base_id][i]
            cmp_wr  = results[cmp_id][i]
            unlocked = _unlocked(base_wr, cmp_wr)
            if not unlocked:
                continue
            all_unlocked.extend(unlocked)
            win_sum_r = cmp_wr.total_r
            conf_map  = {_trade_key(t): _conf_for_trade(cmp_wr, t) for t in unlocked}

            a(f"#### {win_name} — {len(unlocked)} unlocked trade(s)")
            a(f"")
            a(f"| Pair | Pattern | Dir | Stop(p) | ConfRange | R | MAE | MFE | W/L |")
            a(f"|------|---------|-----|:-------:|:---------:|:--:|:---:|:---:|:---:|")
            for t in unlocked:
                conf = conf_map.get(_trade_key(t))
                conf_str = f"{conf:.3f}" if conf is not None else f"[{cmp_thr:.2f}–{_ORIG_CONF:.2f})"
                sp = t.get("initial_stop_pips", 0) or 0
                a(f"| {t.get('pair','?')} | {t.get('pattern','?')} | {t.get('direction','?')} | "
                  f"{sp:.0f}p | {conf_str} | {_r(t):+.2f}R | "
                  f"{t.get('mae_r',0.0):.2f}R | {t.get('mfe_r',0.0):.2f}R | "
                  f"{'✅ W' if _is_win(t) else '❌ L'} |")
            a(f"")

            # Flags
            all_conf_map_flat = {}
            for t in unlocked:
                all_conf_map_flat[_trade_key(t)] = conf_map.get(_trade_key(t))
            flags = _flag_unlocked(unlocked, win_sum_r, all_conf_map_flat, cmp_thr)
            if flags:
                for f in flags:
                    a(f"> {f}")
                a(f"")
            all_flags.extend(flags)

        if not all_unlocked:
            a(f"_No trades unlocked by Variant {cmp_id} across all windows._")
            a(f"")
        else:
            wins   = sum(1 for t in all_unlocked if _is_win(t))
            sum_r  = sum(_r(t) for t in all_unlocked)
            avg_r  = sum_r / len(all_unlocked) if all_unlocked else 0.0
            a(f"#### Summary — All windows")
            a(f"")
            a(f"| Metric | Value |")
            a(f"|--------|-------|")
            a(f"| Total unlocked trades | {len(all_unlocked)} |")
            a(f"| Win rate | {wins/len(all_unlocked):.0%} if {len(all_unlocked)} else '—' |")
            a(f"| SumR | {sum_r:+.2f}R |")
            a(f"| Avg R | {avg_r:+.3f}R |")
            a(f"| Total flags raised | {len(all_flags)} |")
            a(f"")
            if all_flags:
                a(f"**Flags:**")
                a(f"")
                for f in all_flags:
                    a(f"- {f}")
                a(f"")

    # ── Section 5: Cascade displacement analysis ─────────────────────────
    a(f"---")
    a(f"")
    a(f"## 5. Cascade Displacement Analysis")
    a(f"")
    a(f"Baseline (A) trades displaced by weekly cap when an unlocked trade consumed the slot.")
    a(f"")
    for cmp_id, cmp_thr, _ in VARIANTS[1:]:
        a(f"### Variant {cmp_id} displacements vs A")
        a(f"")
        has_any = False
        net_delta_total = 0.0
        a(f"| Window | Displaced trade | Displaced R | Replacement | Replacement R | Net ΔR |")
        a(f"|--------|----------------|:-----------:|-------------|:-------------:|:------:|")
        for i, (win_name, _, _) in enumerate(windows):
            base_wr = results["A"][i]
            cmp_wr  = results[cmp_id][i]
            disp    = _displaced(base_wr, cmp_wr)
            for bt, ut in disp:
                has_any = True
                dr  = _r(bt)
                ur  = _r(ut) if ut else 0.0
                net = ur - dr
                net_delta_total += net
                bt_label = f"{bt.get('pair','?')} {bt.get('pattern','?')}"
                ut_label = f"{ut.get('pair','?')} {ut.get('pattern','?')}" if ut else "_(no replacement)_"
                a(f"| {win_name} | {bt_label} | {dr:+.2f}R | {ut_label} | {ur:+.2f}R | {net:+.2f}R |")
        if not has_any:
            a(f"| — | No displacements detected | — | — | — | — |")
        a(f"")
        a(f"**Net displacement delta (all windows): {net_delta_total:+.2f}R**")
        a(f"")

    # ── Section 6: Confidence distribution of unlocked trades ────────────
    a(f"---")
    a(f"")
    a(f"## 6. Confidence Distribution of Unlocked Trades")
    a(f"")
    a(f"Confidence values are loaded from the ENTER decision log (`backtest_v2_decisions.json`)")
    a(f"written by the backtester after each run.")
    a(f"")
    for cmp_id, cmp_thr, _ in VARIANTS[1:]:
        a(f"### Variant {cmp_id} unlocked trades (conf range: [{cmp_thr:.2f}, {_ORIG_CONF:.2f}))")
        a(f"")
        all_unlocked_conf: List[float] = []
        for i, (win_name, _, _) in enumerate(windows):
            base_wr = results["A"][i]
            cmp_wr  = results[cmp_id][i]
            unlocked = _unlocked(base_wr, cmp_wr)
            for t in unlocked:
                c = _conf_for_trade(cmp_wr, t)
                if c is not None:
                    all_unlocked_conf.append(c)

        if not all_unlocked_conf:
            a(f"_No unlocked trades with resolved confidence values._")
            a(f"")
            a(f"> Note: Confidence in [{cmp_thr:.2f}, {_ORIG_CONF:.2f}) by construction")
            a(f"  (variant threshold allows them, baseline threshold blocks them).")
            a(f"")
            continue

        all_unlocked_conf.sort()
        buckets = [
            (cmp_thr,         cmp_thr + 0.01, 0),
            (cmp_thr + 0.01,  cmp_thr + 0.02, 0),
            (cmp_thr + 0.02,  cmp_thr + 0.03, 0),
            (cmp_thr + 0.03,  _ORIG_CONF,     0),
        ]
        bucket_counts = [0] * len(buckets)
        for c in all_unlocked_conf:
            for bi, (lo, hi, _) in enumerate(buckets):
                if lo <= c < hi or (bi == len(buckets) - 1 and c < _ORIG_CONF):
                    bucket_counts[bi] += 1
                    break

        a(f"| Confidence range | Count | % of unlocked |")
        a(f"|:----------------:|:-----:|:-------------:|")
        total_conf = len(all_unlocked_conf) or 1
        for bi, (lo, hi, _) in enumerate(buckets):
            cnt = bucket_counts[bi]
            a(f"| [{lo:.2f}, {hi:.2f}) | {cnt} | {cnt/total_conf:.0%} |")
        a(f"")
        if all_unlocked_conf:
            a(f"- Min: {min(all_unlocked_conf):.4f}   Max: {max(all_unlocked_conf):.4f}"
              f"   Median: {all_unlocked_conf[len(all_unlocked_conf)//2]:.4f}")
            a(f"")

    # ── Section 7: Pattern distribution of unlocked trades ───────────────
    a(f"---")
    a(f"")
    a(f"## 7. Pattern Distribution of Unlocked Trades")
    a(f"")
    for cmp_id, cmp_thr, _ in VARIANTS[1:]:
        a(f"### Variant {cmp_id} unlocked vs A")
        a(f"")
        pat_counts: Dict[str, List[float]] = {}
        for i, (win_name, _, _) in enumerate(windows):
            for t in _unlocked(results["A"][i], results[cmp_id][i]):
                pat = t.get("pattern", "unknown")
                pat_counts.setdefault(pat, []).append(_r(t))

        if not pat_counts:
            a(f"_None._")
            a(f"")
            continue

        a(f"| Pattern | Count | WR | SumR | AvgR |")
        a(f"|---------|:-----:|:--:|:----:|:----:|")
        for pat, rs in sorted(pat_counts.items(), key=lambda x: -sum(x[1])):
            wins = sum(1 for r in rs if r > 0)
            a(f"| {pat} | {len(rs)} | {wins/len(rs):.0%} | {sum(rs):+.2f}R | {sum(rs)/len(rs):+.3f}R |")
        a(f"")

    # ── Section 8: Verdict ────────────────────────────────────────────────
    a(f"---")
    a(f"")
    a(f"## 8. Verdict")
    a(f"")
    # Compute summary stats for verdict
    a_total_r = sum(_r(t) for wr in results["A"] for t in wr.trades)
    b_total_r = sum(_r(t) for wr in results["B"] for t in wr.trades)
    c_total_r = sum(_r(t) for wr in results["C"] for t in wr.trades)
    a_trades  = sum(wr.n for wr in results["A"])
    b_trades  = sum(wr.n for wr in results["B"])
    c_trades  = sum(wr.n for wr in results["C"])
    b_unlocked_all = sum(len(_unlocked(results["A"][i], results["B"][i])) for i in range(len(windows)))
    c_unlocked_all = sum(len(_unlocked(results["A"][i], results["C"][i])) for i in range(len(windows)))
    b_unlocked_r   = sum(_r(t) for i in range(len(windows)) for t in _unlocked(results["A"][i], results["B"][i]))
    c_unlocked_r   = sum(_r(t) for i in range(len(windows)) for t in _unlocked(results["A"][i], results["C"][i]))
    b_wins = sum(1 for i in range(len(windows)) for t in _unlocked(results["A"][i], results["B"][i]) if _is_win(t))
    c_wins = sum(1 for i in range(len(windows)) for t in _unlocked(results["A"][i], results["C"][i]) if _is_win(t))
    b_wr_u = b_wins / b_unlocked_all if b_unlocked_all else 0.0
    c_wr_u = c_wins / c_unlocked_all if c_unlocked_all else 0.0

    a(f"### Summary comparison")
    a(f"")
    a(f"| Variant | Threshold | Trades | SumR | vs A | Unlocked | Unlocked SumR | Unlocked WR |")
    a(f"|---------|:---------:|:------:|:----:|:----:|:--------:|:-------------:|:-----------:|")
    a(f"| A | 0.77 | {a_trades} | {a_total_r:+.2f}R | — | — | — | — |")
    a(f"| B | 0.73 | {b_trades} | {b_total_r:+.2f}R | {b_total_r-a_total_r:+.2f}R "
      f"| {b_unlocked_all} | {b_unlocked_r:+.2f}R | {b_wr_u:.0%} |")
    a(f"| C | 0.70 | {c_trades} | {c_total_r:+.2f}R | {c_total_r-a_total_r:+.2f}R "
      f"| {c_unlocked_all} | {c_unlocked_r:+.2f}R | {c_wr_u:.0%} |")
    a(f"")
    a(f"### Decision")
    a(f"")

    b_delta = b_total_r - a_total_r
    c_delta = c_total_r - a_total_r

    def _verdict(var_id, delta, unlocked_n, unlocked_wr, unlocked_r, all_flags_count):
        lines2 = []
        if delta < 0:
            lines2.append(f"**Variant {var_id}: REJECT** — SumR regressed vs baseline ({delta:+.2f}R).")
        elif unlocked_n == 0:
            lines2.append(f"**Variant {var_id}: NO_CHANGE** — no new trades unlocked; threshold change has no effect.")
        elif unlocked_wr < 0.45:
            lines2.append(f"**Variant {var_id}: GUARDRAILS** — unlocked trades WR={unlocked_wr:.0%} below 45%; "
                          f"improvement fragile. Consider pair/pattern whitelist on sub-0.77 entries.")
        elif all_flags_count > 0:
            lines2.append(f"**Variant {var_id}: GUARDRAILS** — SumR improves ({delta:+.2f}R) but flags raised. "
                          f"See flag section before promoting.")
        else:
            lines2.append(f"**Variant {var_id}: PROMOTE** — SumR +{delta:.2f}R, unlocked WR={unlocked_wr:.0%}, "
                          f"no flags. Safe to lower threshold.")
        return lines2

    # Count flags
    b_flag_count = sum(
        len(_flag_unlocked(
            _unlocked(results["A"][i], results["B"][i]),
            results["B"][i].total_r,
            {_trade_key(t): _conf_for_trade(results["B"][i], t)
             for t in _unlocked(results["A"][i], results["B"][i])},
            0.73,
        ))
        for i in range(len(windows))
    )
    c_flag_count = sum(
        len(_flag_unlocked(
            _unlocked(results["A"][i], results["C"][i]),
            results["C"][i].total_r,
            {_trade_key(t): _conf_for_trade(results["C"][i], t)
             for t in _unlocked(results["A"][i], results["C"][i])},
            0.70,
        ))
        for i in range(len(windows))
    )

    for line in _verdict("B", b_delta, b_unlocked_all, b_wr_u, b_unlocked_r, b_flag_count):
        a(line)
    a(f"")
    for line in _verdict("C", c_delta, c_unlocked_all, c_wr_u, c_unlocked_r, c_flag_count):
        a(line)
    a(f"")
    a(f"_Report generated by `scripts/confidence_threshold_ablation.py`._")
    a(f"_Offline replay only — no live changes, no master merge._")
    a(f"")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  CONFIDENCE THRESHOLD ABLATION STUDY")
    print(f"  Variants: A=0.77 (baseline), B=0.73, C=0.70")
    print(f"  Windows:  {len(WINDOWS)}")
    print(f"  Capital:  ${CAPITAL:,.0f}")
    print("=" * 72)

    t_total = time.time()
    results = run_all()
    elapsed = time.time() - t_total

    print(f"\n\nAll runs complete ({elapsed/60:.1f} min). Building report…")
    report = build_report(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\n✅ Report written to: {REPORT_PATH}")
    print(f"\n{'─'*72}")
    print(report[:3000])
    if len(report) > 3000:
        print(f"\n… (truncated — see {REPORT_PATH})")
    print(f"{'─'*72}")


if __name__ == "__main__":
    main()
