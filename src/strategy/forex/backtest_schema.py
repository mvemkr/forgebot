"""
backtest_schema.py — Canonical BacktestResult schema
=====================================================
Single source of truth for the data contract between run_backtest() and
any consumer (compare harnesses, dashboard API, show_results, tests).

Canonical field names
---------------------
  return_pct      — (balance - start) / start × 100          [was: ret_pct]
  max_dd_pct      — peak-to-trough drawdown %                 [was: max_dd]
  win_rate        — fraction 0..1   (e.g. 0.29 = 29%)
  avg_r           — avg realised R per trade
  best_r          — max realised R
  worst_r         — min realised R   (negative on a loser)
  n_trades        — total closed trades

Backward-compat aliases
-----------------------
  BacktestResult.get("ret_pct")   → return_pct
  BacktestResult.get("max_dd")    → max_dd_pct
  BacktestResult["ret_pct"]       → return_pct
  BacktestResult["max_dd"]        → max_dd_pct

Usage
-----
  from src.strategy.forex.backtest_schema import BacktestResult

  r: BacktestResult = run_backtest(...)
  print(r.return_pct, r.max_dd_pct)        # attribute access (preferred)
  print(r.get("ret_pct"))                   # legacy key still works
  print(r["trades"])                        # dict-style access also OK
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Legacy field-name aliases for backward compatibility
_ALIASES: Dict[str, str] = {
    "ret_pct":   "return_pct",
    "max_dd":    "max_dd_pct",
}


@dataclass
class BacktestResult:
    """Typed result returned by run_backtest()."""

    # ── Core performance ─────────────────────────────────────────────────
    return_pct:      float = 0.0   # (balance − start) / start × 100
    max_dd_pct:      float = 0.0   # peak-to-trough drawdown %
    win_rate:        float = 0.0   # fraction 0..1
    avg_r:           float = 0.0   # mean realised R across all trades
    best_r:          float = 0.0   # max realised R
    worst_r:         float = 0.0   # min realised R (≤ 0 on loss)
    n_trades:        int   = 0
    balance:         float = 0.0   # final balance

    # ── Trade breakdown ──────────────────────────────────────────────────
    avg_r_win:       float = 0.0
    avg_r_loss:      float = 0.0
    n_target:        int   = 0     # target_reached + weekend_proximity exits
    n_ratchet:       int   = 0     # ratchet_stop_hit exits
    n_sl:            int   = 0     # stop_hit exits
    exec_rr_p50:     float = 0.0   # median planned RR of entered trades
    max_dollar_loss: float = 0.0

    # ── Gate hit counts (all 6 Alex rules + adaptive) ───────────────────
    time_blocks:              int = 0   # NO_SUNDAY + NO_THU_FRI
    countertrend_htf_blocks:  int = 0   # COUNTERTREND_HTF
    weekly_limit_blocks:      int = 0   # WEEKLY_TRADE_LIMIT
    min_rr_small_blocks:      int = 0   # MIN_RR_SMALL_ACCOUNT
    indecision_doji_blocks:   int = 0   # INDECISION_DOJI
    dd_killswitch_blocks:     int = 0   # DD kill-switch blocks
    adaptive_time_blocks:     int = 0   # ADAPTIVE_TIME_GATE (regime-gated Thu/Fri)
    strict_htf_blocks:        int = 0   # STRICT_PROTREND_HTF (all 3 must agree)
    wd_htf_blocks:            int = 0   # WD_PROTREND_HTF (W+D required, 4H exempt)
    dyn_pip_eq_blocks:        int = 0   # DYN_PIP_EQUITY (stop_pips × MIN_RR)
    wd_alignment_pct:         float = 0.0  # % entered trades where W==D agreed with direction

    # ── Regime score analytics ───────────────────────────────────────────
    regime_avg:          float = 0.0
    regime_pct_high:     float = 0.0    # % trades with score ≥ 3.0
    regime_pct_extreme:  float = 0.0    # % trades with score ≥ 3.5
    # ── Risk mode: % of TRADES entered in each mode ─────────────────────
    risk_mode_pct_low:     float = 0.0   # % trades entered while RiskMode == LOW
    risk_mode_pct_medium:  float = 0.0   # % trades entered while RiskMode == MEDIUM
    risk_mode_pct_high:    float = 0.0   # % trades entered while RiskMode == HIGH
    risk_mode_pct_extreme: float = 0.0   # % trades entered while RiskMode == EXTREME

    # ── Risk mode: % of CALENDAR TIME (H4 bars) in each mode ────────────
    # Sampled at each H4 bar using compute_risk_mode(); gives market-condition
    # distribution independent of when trades were actually entered.
    time_in_mode_pct_low:     float = 0.0
    time_in_mode_pct_medium:  float = 0.0
    time_in_mode_pct_high:    float = 0.0
    time_in_mode_pct_extreme: float = 0.0

    # ── Stop selection quality ───────────────────────────────────────────
    stop_type_counts:  dict  = field(default_factory=dict)  # {stop_type: count}
    atr_fallback_pct:  float = 0.0    # % of trades using atr_fallback — >40% = regression
    stop_pips_p50:     float = 0.0    # median initial stop distance in pips (all trades)

    # ── Profiling ────────────────────────────────────────────────────────
    api_calls:     int   = 0
    eval_calls:    int   = 0
    eval_ms_avg:   float = 0.0

    # ── Raw lists (large — last in repr) ────────────────────────────────
    trades:          List[Dict[str, Any]] = field(default_factory=list)
    regime_scores:   List[float]           = field(default_factory=list)
    trades_per_week: Dict[Any, int]        = field(default_factory=dict)
    gap_log:         List[Dict[str, Any]]  = field(default_factory=list)
    candle_data:     Dict[str, Any]        = field(default_factory=dict)

    # ── Dict-style access (backward compat) ─────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Support r.get("ret_pct") → returns return_pct value."""
        actual = _ALIASES.get(key, key)
        return getattr(self, actual, default)

    def __getitem__(self, key: str) -> Any:
        actual = _ALIASES.get(key, key)
        try:
            return getattr(self, actual)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        actual = _ALIASES.get(key, key)
        return hasattr(self, actual)

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict with canonical names (for JSON serialisation)."""
        from dataclasses import asdict
        return asdict(self)
