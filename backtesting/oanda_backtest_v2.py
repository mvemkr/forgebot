"""
OANDA 1H Backtester v2 — Real Strategy Code
============================================

Key upgrade over v1: this backtester calls SetAndForgetStrategy.evaluate()
directly instead of reimplementing pattern/level/session logic inline.
When the live strategy improves, the backtest inherits it automatically.

Gap log:
  Any bar where v1 and v2 produce different decisions is written to
  logs/backtest_gap_log.jsonl with the full reason from both sides.
  This is how we track where the two implementations diverge and
  understand which set of rules produces better outcomes.

Known gaps (documented, not bugs):
  - NEWS FILTER: disabled in backtest — ForexFactory historical data
    unavailable. Backtest may show trades the live bot would have blocked
    during Tier 1 events. Logged as gap type "news_filter_skipped".
  - WEEKLY CANDLES: resampled from daily data. Live fetches dedicated W
    bars from OANDA. Minor edge differences possible around week boundaries.
  - EXIT LOGIC: live bot alerts Mike and he decides. Backtest auto-exits
    on stop hit or max hold period (same as v1). Logged as known gap.

Usage:
    python -m backtesting.oanda_backtest_v2
    python -m backtesting.oanda_backtest_v2 --start 2025-10-01 --balance 8000
"""

import sys
import json
import time
import pickle
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta, time as dtime
from typing import Optional, Dict, List
import pytz

_ET_TZ = pytz.timezone("America/New_York")   # for adaptive Thu/Fri gate

# ── Real strategy imports ─────────────────────────────────────────────────────
from src.exchange.oanda_client import OandaClient, INSTRUMENT_MAP
from src.strategy.forex.set_and_forget   import SetAndForgetStrategy, Decision
from src.strategy.forex.news_filter      import NewsFilter
from src.strategy.forex.regime_score     import (
    compute_regime_score, RegimeScore,
    compute_risk_mode, RISK_MODE_PARAMS,
)
from src.strategy.forex.targeting        import select_target, find_next_structure_level
from src.execution.risk_manager_forex    import ForexRiskManager
from src.execution.trade_journal         import TradeJournal
from src.strategy.forex.currency_strength import CurrencyStrengthAnalyzer, CurrencyTheme, STACK_MAX

# ── Shared execution config (single source of truth) ─────────────────────────
# These constants are also used by the live orchestrator. Change them here,
# they apply everywhere. Do NOT hardcode these values anywhere else.
#
# LEVER SYSTEM: import the module by reference (_sc) so that apply_levers()
# patches propagate to every _sc.X access in this file.  The flat imports
# below provide IDE auto-complete and defaults; in the hot loop we read
# through _sc so overrides take effect.
import src.strategy.forex.strategy_config as _sc
from src.strategy.forex.strategy_config import (
    MIN_CONFIDENCE,
    MIN_RR,
    ATR_STOP_MULTIPLIER,
    ATR_MIN_MULTIPLIER,
    ATR_LOOKBACK,
    MAX_CONCURRENT_TRADES_BACKTEST,
    LONDON_SESSION_START_UTC,
    LONDON_SESSION_END_UTC,
    STOP_COOLDOWN_DAYS,
    winner_rule_check,
    BLOCK_ENTRY_WHILE_WINNER_RUNNING,
    WINNER_THRESHOLD_R,
    NECKLINE_CLUSTER_PCT,
    get_model_tags,
    DRY_RUN_PAPER_BALANCE,
    CACHE_VERSION,
)
from src.strategy.forex.backtest_schema import BacktestResult
from src.strategy.forex import alex_policy

# ── Backtest-only config ──────────────────────────────────────────────────────
BACKTEST_START  = datetime(2025, 7, 1, tzinfo=timezone.utc)   # ~7 months of history
STARTING_BAL    = 8_000.0
MAX_HOLD_BARS   = 365 * 24         # effectively no cap — strategy has no TP or hold limit
                                   # (live bot runs until stop hit or Mike manually closes)
GAP_LOG_PATH         = Path.home() / "trading-bot" / "logs" / "backtest_gap_log.jsonl"
WHITELIST_BACKTEST_FILE = Path.home() / "trading-bot" / "logs" / "whitelist_backtest.json"
DECISION_LOG    = Path.home() / "trading-bot" / "logs" / "backtest_v2_decisions.json"
V1_DECISION_LOG = Path.home() / "trading-bot" / "logs" / "backtest_decisions.json"

WATCHLIST = [
    # USD-based majors
    "GBP/USD", "EUR/USD", "USD/JPY", "USD/CHF", "USD/CAD", "NZD/USD", "AUD/USD",
    # GBP crosses (Alex's confirmed list)
    "GBP/JPY", "GBP/CHF", "GBP/NZD", "GBP/CAD",
    # EUR crosses
    "EUR/AUD", "EUR/CAD", "EUR/JPY",
    # JPY crosses (Alex runs all 6 for theme stacking — Week 7-8 $70K)
    "AUD/JPY", "CAD/JPY", "NZD/JPY",
    # Commodity crosses
    "AUD/CAD", "NZD/CAD",
    # Removed (not on Alex's watchlist): EUR/GBP, EUR/NZD, AUD/NZD
]

# ── Historical news filter — CSV-backed ───────────────────────────────────────
class _HistoricalNewsFilter(NewsFilter):
    """
    Backtest news filter backed by data/news/high_impact_events.csv.
    Source: TradingView Economic Calendar API (high-impact only).

    Blocks entries when the candle falls within BLOCK_WINDOW_MIN of a
    high-impact event for any currency in the pair being evaluated.
    Falls back to no-op if CSV not found (logs a warning once).
    """
    BLOCK_WINDOW_MIN = 60   # ±60 min around each news event

    def __init__(self):
        super().__init__()
        self._events: list = []          # list of (datetime_utc, set_of_currencies)
        self._loaded = False
        self._warned = False
        self._load()

    def _load(self):
        import csv as _csv, pathlib
        csv_path = pathlib.Path(__file__).resolve().parents[1] / "data" / "news" / "high_impact_events.csv"
        if not csv_path.exists():
            self._warned = True
            return
        from datetime import timezone as _tz
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                try:
                    dt = datetime.strptime(row["datetime_utc"], "%Y-%m-%d %H:%M").replace(tzinfo=_tz.utc)
                    ccy = row["currency"].upper().strip()
                    self._events.append((dt, ccy))
                except Exception:
                    pass
        self._loaded = True

    def _pair_currencies(self, pair: str):
        return set(pair.replace("_", "/").upper().split("/")[:2])

    def _near_news(self, candle_dt: datetime, currencies: set) -> tuple:
        from datetime import timedelta
        window = timedelta(minutes=self.BLOCK_WINDOW_MIN)
        for ev_dt, ev_ccy in self._events:
            if ev_ccy in currencies and abs(candle_dt - ev_dt) <= window:
                return True, ev_ccy, ev_dt
        return False, None, None

    def is_entry_blocked(self, dt_utc, post_news_candle=None):
        """Not used in backtest path — candle-level check used instead."""
        return False, ""

    def is_news_candle(self, candle_dt_utc, pair: str = "") -> bool:
        if not self._loaded:
            if not self._warned:
                print("  ⚠ Historical news CSV not found — news filter disabled")
                self._warned = True
            return False
        ccys = self._pair_currencies(pair) if pair else set()
        hit, _, _ = self._near_news(candle_dt_utc, ccys if ccys else
                                    {"USD","EUR","GBP","JPY","AUD","NZD","CAD","CHF"})
        return hit

    def refresh_if_needed(self):
        pass


# ── OANDA candle fetch (uses OandaClient — same auth as live bot) ─────────────
_oanda_client = OandaClient()   # loads API key from .env automatically

def _fetch_range(pair: str, granularity: str, from_dt: datetime, to_dt: datetime = None) -> Optional[pd.DataFrame]:
    """
    Fetch candles by date range, paginating in 5000-bar chunks.
    OANDA rule: cannot use 'count' when both 'from' AND 'to' are set.
    So we paginate by stepping 'from' forward each page.
    """
    instrument = INSTRUMENT_MAP.get(pair, pair.replace("/", "_"))
    granularity_delta = {
        "H1": pd.Timedelta(hours=1),
        "H4": pd.Timedelta(hours=4),
        "D":  pd.Timedelta(days=1),
        "W":  pd.Timedelta(weeks=1),
    }.get(granularity, pd.Timedelta(hours=1))

    all_rows = []
    current_from = from_dt
    final_to = to_dt

    while True:
        try:
            # Fetch one page: from=current_from, count=5000 (no 'to' → OANDA accepts this)
            # Then stop when we've passed final_to
            page_to = None
            if final_to:
                # Compute what a 5000-bar window from current_from looks like
                tentative_end = current_from + granularity_delta * 5000
                if tentative_end >= final_to:
                    # Last page — use 'to' without 'count'
                    params = {
                        "granularity": granularity,
                        "from": current_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "to":   final_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "price": "M",
                    }
                else:
                    params = {
                        "granularity": granularity,
                        "from": current_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "count": 5000,
                        "price": "M",
                    }
            else:
                params = {
                    "granularity": granularity,
                    "from": current_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "count": 5000,
                    "price": "M",
                }

            resp = requests.get(
                f"{_oanda_client.base}/v3/instruments/{instrument}/candles",
                headers=_oanda_client.headers,
                params=params,
                timeout=30,
            )
            data = resp.json()
            if resp.status_code != 200:
                break
            candles = data.get("candles", [])
            if not candles:
                break
            rows = []
            for c in candles:
                mid = c.get("mid", {})
                rows.append({
                    "time":   pd.Timestamp(c["time"]).tz_localize(None),
                    "open":   float(mid.get("o", 0)),
                    "high":   float(mid.get("h", 0)),
                    "low":    float(mid.get("l", 0)),
                    "close":  float(mid.get("c", 0)),
                    "volume": int(c.get("volume", 0)),
                })
            all_rows.extend(rows)
            # Done if: fewer than 5000 bars returned, or we used a 'to' param
            if len(candles) < 5000 or "to" in params:
                break
            # Advance to next page
            last_time = pd.Timestamp(candles[-1]["time"]).tz_localize(None)
            current_from = (last_time + granularity_delta).to_pydatetime().replace(tzinfo=timezone.utc)
            time.sleep(0.2)
        except Exception as e:
            print(f"    ⚠ Fetch range error {pair} {granularity}: {e}")
            break

    if not all_rows:
        return None
    df = pd.DataFrame(all_rows).set_index("time")
    df = df[~df.index.duplicated(keep='last')]
    df.sort_index(inplace=True)
    return df


def _fetch(pair: str, granularity: str, count: int) -> Optional[pd.DataFrame]:
    instrument = INSTRUMENT_MAP.get(pair, pair.replace("/", "_"))
    try:
        resp = requests.get(
            f"{_oanda_client.base}/v3/instruments/{instrument}/candles",
            headers=_oanda_client.headers,
            params={"granularity": granularity, "count": count, "price": "M"},
            timeout=30,
        )
        candles = resp.json().get("candles", [])
        rows = []
        for c in candles:
            mid = c.get("mid", {})
            rows.append({
                "time":   pd.Timestamp(c["time"]).tz_localize(None),
                "open":   float(mid.get("o", 0)),
                "high":   float(mid.get("h", 0)),
                "low":    float(mid.get("l", 0)),
                "close":  float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })
        if not rows:
            return None
        df = pd.DataFrame(rows).set_index("time")
        return df
    except Exception as e:
        print(f"    ⚠ Fetch error {pair} {granularity}: {e}")
        return None


def _resample_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily → weekly (Mon open, Fri close)."""
    return df_daily.resample("W-FRI").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()


# ── 4H structure target ───────────────────────────────────────────────────────



# ── Gap logger ────────────────────────────────────────────────────────────────
gap_log: List[dict] = []

def log_gap(ts, pair, v1_result, v2_result, gap_type, detail):
    entry = {
        "ts":       ts.isoformat(),
        "pair":     pair,
        "v1":       v1_result,
        "v2":       v2_result,
        "gap_type": gap_type,
        "detail":   detail,
    }
    gap_log.append(entry)


# ── Load v1 decisions for comparison ─────────────────────────────────────────
def _load_v1_decisions():
    if not V1_DECISION_LOG.exists():
        return {}
    try:
        with open(V1_DECISION_LOG) as f:
            data = json.load(f)
        # key by (pair, bar_ts_iso)
        index = {}
        for d in data.get("decisions", []):
            key = (d.get("pair", ""), d.get("ts", ""))
            index[key] = d
        return index
    except Exception:
        return {}


# ── Trail arm configurations ──────────────────────────────────────────────────
# Used by ab9 multi-arm comparison.  Each arm overrides only the trail params.
#
# Two-stage trail (Arm C):
#   Stage 1 — at TRAIL_ACTIVATE_R MFE: lock stop to entry + TRAIL_LOCK_R (one-time)
#   Stage 2 — at TRAIL_STAGE2_R  MFE: start trailing at (trail_max − TRAIL_STAGE2_DIST_R)
#
# Standard trail (Arms A/B):
#   At TRAIL_ACTIVATE_R MFE: trail continuously at (trail_max − TRAIL_LOCK_R),
#   floor at entry + TRAIL_LOCK_R.  (TRAIL_STAGE2_R = None → stage 2 disabled)
TRAIL_ARMS: Dict[str, dict] = {
    "A": {
        "label":               "Arm A — activate 1.0R, trail 0.5R  (ab8 baseline)",
        "TRAIL_ACTIVATE_R":    1.0,
        "TRAIL_LOCK_R":        0.5,
        "TRAIL_STAGE2_R":      None,
        "TRAIL_STAGE2_DIST_R": None,
    },
    "B": {
        "label":               "Arm B — activate 1.5R, trail 0.5R",
        "TRAIL_ACTIVATE_R":    1.5,
        "TRAIL_LOCK_R":        0.5,
        "TRAIL_STAGE2_R":      None,
        "TRAIL_STAGE2_DIST_R": None,
    },
    "C": {
        "label":               "Arm C — 2-stage: lock +0.5R at 2R, trail 1.0R after 3R",
        "TRAIL_ACTIVATE_R":    2.0,
        "TRAIL_LOCK_R":        0.5,
        "TRAIL_STAGE2_R":      3.0,
        "TRAIL_STAGE2_DIST_R": 1.0,
        "STALL_EXIT_BARS":     None,   # no stall exit
        "STALL_EXIT_MFE_R":    None,
    },
    # ── Tuning variants (Step 3 of 6K% path) ────────────────────────────────
    # C1: earlier lock (1.8R, +0.3R), same stage-2 → smaller initial lock gain,
    #     but activates sooner → less exposure during early adverse moves.
    "C1": {
        "label":               "Arm C1 — lock +0.3R at 1.8R, trail 1.0R after 3R",
        "TRAIL_ACTIVATE_R":    1.8,
        "TRAIL_LOCK_R":        0.3,
        "TRAIL_STAGE2_R":      3.0,
        "TRAIL_STAGE2_DIST_R": 1.0,
        "STALL_EXIT_BARS":     None,
        "STALL_EXIT_MFE_R":    None,
    },
    # C2: same lock point, tighter trailing distance → stops closer, smaller
    #     wins but less retracement to exit → lower MaxDD expected.
    "C2": {
        "label":               "Arm C2 — lock +0.5R at 2R, trail 0.8R after 3R",
        "TRAIL_ACTIVATE_R":    2.0,
        "TRAIL_LOCK_R":        0.5,
        "TRAIL_STAGE2_R":      3.0,
        "TRAIL_STAGE2_DIST_R": 0.8,
        "STALL_EXIT_BARS":     None,
        "STALL_EXIT_MFE_R":    None,
    },
    # C3: same as C + stall exit — if MFE hasn't reached 0.5R after 5 bars,
    #     close at market to free capital for better setups.
    "C3": {
        "label":               "Arm C3 — lock +0.5R at 2R, trail 1.0R + stall-exit 5b/0.5R",
        "TRAIL_ACTIVATE_R":    2.0,
        "TRAIL_LOCK_R":        0.5,
        "TRAIL_STAGE2_R":      3.0,
        "TRAIL_STAGE2_DIST_R": 1.0,
        "STALL_EXIT_BARS":     5,      # bars after entry to check
        "STALL_EXIT_MFE_R":    0.5,    # if MFE < 0.5R at that point → close
    },
    # D: Arm C trail + Chop Shield gates (Part A: 48h auto-pause on streak≥3;
    #    Part B: recovery selectivity — exec_rr≥3.0R, conf+5%, weekly_cap=1).
    #    Signal/stop/target logic is IDENTICAL to Arm C.
    "D": {
        "label":               "Arm D — Arm C trail + Chop Shield (streak≥3 auto-pause + recovery)",
        "TRAIL_ACTIVATE_R":    2.0,
        "TRAIL_LOCK_R":        0.5,
        "TRAIL_STAGE2_R":      3.0,
        "TRAIL_STAGE2_DIST_R": 1.0,
        "STALL_EXIT_BARS":     None,
        "STALL_EXIT_MFE_R":    None,
        "_chop_shield":        True,   # marker: run_backtest auto-enables chop shield
    },
}
_DEFAULT_ARM = "A"   # used when trail_cfg=None

# ─────────────────────────────────────────────────────────────────────────────
# ── Per-pair/TF raw candle cache ─────────────────────────────────────────────
# One file per (pair, TF, price_type, env, CACHE_VERSION).
# Any run whose date window falls within the cached range = zero OANDA requests.
# Different pair lists share the same per-pair files — no re-fetch on list change.
#
# Invalidation:
#   • Bump CACHE_VERSION in strategy_config.py  → wipes all pairs/TFs at once
#   • Delete ~/.cache/forge_backtester/         → manual wipe
#   • Window extends beyond cached range        → only that pair/TF is re-fetched
#
# Default: ON (pass --no-cache to disable; --cache kept as no-op for compat)
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR  = Path.home() / ".cache" / "forge_backtester"
PRICE_TYPE = "M"   # OANDA mid (open/high/low/close)

# Module-level API call counter — reset to 0 at the start of every fetch block.
# After a full cache hit this stays 0; callers can assert on it.
_api_call_count: int = 0

def _cache_env() -> str:
    """'live' or 'prac' — baked into cache filenames so account envs don't cross-contaminate."""
    return "live" if "fxtrade" in _oanda_client.base else "prac"

def _pair_slug(pair: str) -> str:
    return pair.replace("/", "").replace("_", "")

def _pair_cache_path(pair: str, tf: str) -> Path:
    return CACHE_DIR / f"{_pair_slug(pair)}_{tf}_{PRICE_TYPE}_{_cache_env()}_{CACHE_VERSION}.pkl"

def _pair_cache_load(pair: str, tf: str,
                     need_start: datetime, need_end: datetime) -> Optional[pd.DataFrame]:
    """
    Load (pair, TF) from disk cache.
    Returns the DataFrame if the cache exists, version matches, and data covers
    [need_start, need_end].  Returns None on any miss/mismatch.
    """
    path = _pair_cache_path(pair, tf)
    if not path.exists():
        return None
    try:
        obj = pickle.load(open(path, "rb"))
        if obj.get("version") != CACHE_VERSION:
            return None
        df: pd.DataFrame = obj["df"]
        ns = need_start.replace(tzinfo=None) if need_start.tzinfo else need_start
        ne = need_end.replace(tzinfo=None)   if need_end.tzinfo   else need_end
        # Allow tolerance at both ends — OANDA bar timestamps don't land exactly
        # on calendar boundaries (weekends, bank holidays, UTC-offset drift):
        #   start: allow up to 7 days (first bar may be the next Monday after a weekend/holiday)
        #   end:   allow up to 5 days (last bar may be a few days before computed runout_dt)
        ns_ceil    = ns + pd.Timedelta(days=7)
        ne_floor   = ne - pd.Timedelta(days=5)
        if df.index.min() > ns_ceil or df.index.max() < ne_floor:
            return None   # coverage gap — stale or shorter window cached
        return df
    except Exception:
        return None

def _pair_cache_save(pair: str, tf: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _pair_cache_path(pair, tf)
    with open(path, "wb") as f:
        pickle.dump({
            "df":          df,
            "version":     CACHE_VERSION,
            "saved_at":    datetime.now(timezone.utc).isoformat(),
            "pair":        pair,
            "tf":          tf,
            "price_type":  PRICE_TYPE,
            "bars":        len(df),
            "range_start": str(df.index.min()),
            "range_end":   str(df.index.max()),
        }, f)

# ── Counted fetch wrappers ────────────────────────────────────────────────────
# Every call to these increments _api_call_count so callers can assert "0 API
# calls on second run".  Raw _fetch / _fetch_range are unchanged.

def _counted_fetch_range(pair: str, tf: str,
                         from_dt: datetime, to_dt: datetime = None) -> Optional[pd.DataFrame]:
    global _api_call_count
    _api_call_count += 1
    return _fetch_range(pair, tf, from_dt=from_dt, to_dt=to_dt)

def _counted_fetch(pair: str, tf: str, count: int) -> Optional[pd.DataFrame]:
    global _api_call_count
    _api_call_count += 1
    return _fetch(pair, tf, count)

# ── Legacy monolithic cache (kept as no-op stubs for external callers) ────────
_CACHE_PATH = Path("/tmp/backtest_candle_cache.pkl")   # old path — kept for reference

def _save_cache(candle_data: dict, meta: dict) -> None:  # noqa: deprecated
    """Deprecated — replaced by per-pair _pair_cache_save(). No-op."""
    pass

def _load_cache() -> Optional[tuple]:  # noqa: deprecated
    """Deprecated — replaced by per-pair _pair_cache_load(). Always returns None."""
    return None

# ── Main backtest ─────────────────────────────────────────────────────────────
def run_backtest(start_dt: datetime = BACKTEST_START, end_dt: datetime = None,
                 starting_bal: float = STARTING_BAL, notes: str = "",
                 trail_cfg: Optional[dict] = None,
                 trail_arm_key: str = "",
                 preloaded_candle_data: Optional[dict] = None,
                 use_cache: bool = True,
                 quiet: bool = False,
                 adaptive_gates: bool = False,
                 adaptive_threshold: float = 0.5,
                 policy_tag: str = "",
                 strict_protrend_htf: bool = False,
                 dynamic_pip_equity: bool = False,
                 wd_protrend_htf: bool = False,
                 flat_risk_pct: Optional[float] = None,
                 force_risk_mode: Optional[str] = None,
                 streak_demotion_thresh: int = 1,
                 chop_shield: bool = False):
    """Run a single backtest simulation.

    quiet=True suppresses all stdout (useful when called from compare scripts).
    The BacktestResult is always populated regardless of quiet mode.

    trail_arm_key: label stored in model_tags and _result_record (e.g. "A", "B", "C").
                   Auto-detected from trail_cfg if omitted.
    force_risk_mode: pin risk mode for every entry ("LOW"|"MEDIUM"|"HIGH"|"EXTREME").
                     None = AUTO (compute dynamically per entry — default).
    chop_shield: enable Chop Shield gates (auto-detected True when trail_arm_key="D").
    """
    import io, sys as _sys
    # Resolve arm key ↔ trail_cfg (bidirectional)
    if not trail_arm_key and trail_cfg:
        # Auto-detect key from explicit cfg dict
        for _k, _v in TRAIL_ARMS.items():
            if all(trail_cfg.get(f) == _v.get(f)
                   for f in ("TRAIL_ACTIVATE_R", "TRAIL_LOCK_R", "TRAIL_STAGE2_R")):
                trail_arm_key = _k
                break
    if trail_arm_key in TRAIL_ARMS and not trail_cfg:
        # Resolve cfg from key so callers can pass just trail_arm_key="C"
        trail_cfg = TRAIL_ARMS[trail_arm_key]
    # Auto-detect chop shield from Arm D marker
    if (trail_cfg or {}).get("_chop_shield"):
        chop_shield = True
    _orig_stdout = _sys.stdout
    if quiet:
        _sys.stdout = io.StringIO()
    try:
        return _run_backtest_body(
            start_dt=start_dt, end_dt=end_dt, starting_bal=starting_bal,
            notes=notes, trail_cfg=trail_cfg, trail_arm_key=trail_arm_key,
            preloaded_candle_data=preloaded_candle_data, use_cache=use_cache,
            adaptive_gates=adaptive_gates, adaptive_threshold=adaptive_threshold,
            policy_tag=policy_tag,
            strict_protrend_htf=strict_protrend_htf,
            dynamic_pip_equity=dynamic_pip_equity,
            wd_protrend_htf=wd_protrend_htf,
            flat_risk_pct=flat_risk_pct,
            force_risk_mode=force_risk_mode,
            streak_demotion_thresh=streak_demotion_thresh,
            chop_shield=chop_shield,
        )
    finally:
        _sys.stdout = _orig_stdout


def _run_backtest_body(start_dt: datetime = BACKTEST_START, end_dt: datetime = None,
                       starting_bal: float = STARTING_BAL, notes: str = "",
                       trail_cfg: Optional[dict] = None,
                       trail_arm_key: str = "",
                       preloaded_candle_data: Optional[dict] = None,
                       use_cache: bool = True,
                       adaptive_gates: bool = False,
                       adaptive_threshold: float = 0.5,
                       policy_tag: str = "",
                       strict_protrend_htf: bool = False,
                       dynamic_pip_equity: bool = False,
                       wd_protrend_htf: bool = False,
                       flat_risk_pct: Optional[float] = None,
                       force_risk_mode: Optional[str] = None,
                       streak_demotion_thresh: int = 1,
                       chop_shield: bool = False):
    # PROTREND_ONLY config flag → wd_protrend_htf gate.
    # Config wins when True; explicit True caller arg also wins.
    if getattr(_sc, "PROTREND_ONLY", False) and not wd_protrend_htf:
        wd_protrend_htf = True
    end_naive = end_dt.replace(tzinfo=None) if end_dt else None
    # Extend data fetch so open positions can run to natural close after the entry window.
    # Entries stop at end_dt; monitoring continues up to end_dt + RUNOUT_DAYS.
    RUNOUT_DAYS = 180
    runout_dt  = (end_dt + pd.Timedelta(days=RUNOUT_DAYS)).replace(tzinfo=timezone.utc) if end_dt else None

    # Resolve trail config: explicit cfg > key lookup > default arm
    _tcfg  = trail_cfg or (TRAIL_ARMS[trail_arm_key] if trail_arm_key in TRAIL_ARMS else TRAIL_ARMS[_DEFAULT_ARM])
    _tlabel = _tcfg.get("label", "")

    print(f"\n{'='*65}")
    print(f"OANDA 1H BACKTEST v2 — Real Strategy Code")
    print(f"Start: {start_dt.date()}  |  End: {end_dt.date() if end_dt else 'today'}  |  Capital: ${starting_bal:,.2f}")
    if _tlabel:
        print(f"Trail: {_tlabel}")
    print(f"{'='*65}")

    # ── Parity header — stamped on every run ──────────────────────────
    # Lets you verify at a glance that backtest mirrors live before trusting results.
    # If MAX_CONCURRENT live ≠ backtest, results are NOT comparable to production.
    try:
        import hashlib as _hl, inspect as _ins
        _cfg_src = _ins.getsource(_sc)
        _cfg_hash = _hl.md5(_cfg_src.encode()).hexdigest()[:8]
    except Exception:
        _cfg_hash = "unknown"
    try:
        import subprocess as _sp
        _sha = _sp.check_output(["git", "-C", str(Path(__file__).parent),
                                  "rev-parse", "--short", "HEAD"],
                                 stderr=_sp.DEVNULL).decode().strip()
    except Exception:
        _sha = "unknown"

    _parity_ok  = _sc.MAX_CONCURRENT_TRADES_LIVE == _sc.MAX_CONCURRENT_TRADES_BACKTEST
    _parity_sym = "✓" if _parity_ok else "⚠ MISMATCH"
    _exp_flag   = (_sc.MAX_CONCURRENT_TRADES_BACKTEST > 1)

    print(f"┌─ Parity {'─'*54}┐")
    print(f"│  engine SHA      : {_sha:<10}  config md5 : {_cfg_hash:<12}{'':>8}│")
    print(f"│  concurrent LIVE : {_sc.MAX_CONCURRENT_TRADES_LIVE:<4}  "
          f"concurrent BT : {_sc.MAX_CONCURRENT_TRADES_BACKTEST:<4}  "
          f"parity: {_parity_sym:<14}│")
    print(f"│  trigger mode    : {_sc.ENTRY_TRIGGER_MODE:<15}  "
          f"spread model : {'ON' if _sc.SPREAD_MODEL_ENABLED else 'OFF'}{'':>16}│")
    _pt_label = "ON (W+D gate, 4H exempt)" if wd_protrend_htf else "OFF (full counter-trend)"
    _fm_label = f"FORCED:{force_risk_mode}" if force_risk_mode else "AUTO (dynamic per entry)"
    print(f"│  protrend_only   : {_pt_label:<43}│")
    print(f"│  risk_mode       : {_fm_label:<43}│")
    if _exp_flag:
        print(f"│  ⚠  EXPERIMENTAL RUN — BT concurrency > 1, not comparable to live{'':>3}│")
    print(f"└{'─'*63}┘")

    # ── Fetch candle data ─────────────────────────────────────────────
    global _api_call_count
    _api_call_count = 0   # reset for this run

    if preloaded_candle_data is not None:
        # In-memory bypass: caller already holds data (e.g. multi-mode comparison script).
        # Skip all cache I/O — no disk reads, no API calls.
        candle_data = preloaded_candle_data
        print(f"\n  ⚡ Preloaded candle data ({len(candle_data)} pairs) — skipping cache")

    else:
        # Per-pair/TF cache-aware fetch.
        # Each (pair, TF) is loaded from disk if cached + coverage OK; fetched otherwise.
        env       = _cache_env()
        need_end  = runout_dt or datetime.now(tz=timezone.utc)
        use_range = (datetime.now(tz=timezone.utc) - start_dt).days > 200

        # TF → (fetch_start, fetch_end) mapping
        tf_ranges = {
            "H1": ((start_dt - pd.Timedelta(days=180)).replace(tzinfo=timezone.utc), need_end),
            "H4": ((start_dt - pd.Timedelta(days=180)).replace(tzinfo=timezone.utc), need_end),
            "D":  ((start_dt - pd.Timedelta(days=730)).replace(tzinfo=timezone.utc), need_end),
        }

        # ── Cache header ──────────────────────────────────────────────
        W = 72

        def _box(text: str) -> str:
            """Left-justify text inside a │ box of interior width W."""
            return f"│  {text:<{W}}│"

        pairs_str = ", ".join(WATCHLIST[:6]) + (f" +{len(WATCHLIST)-6}" if len(WATCHLIST) > 6 else "")
        end_disp  = (end_dt or datetime.now(tz=timezone.utc)).date()

        print(f"\n┌─ Candle cache {'─' * W}┐")
        print(_box(f"cache_version : {CACHE_VERSION}   price_type : {PRICE_TYPE} (mid)   env : {env}"))
        print(_box(f"cache_dir     : {CACHE_DIR}"))
        print(_box(f"pairs ({len(WATCHLIST):2d})     : {pairs_str}"))
        print(_box(f"TFs           : {', '.join(tf_ranges)}"))
        print(_box(f"window        : {start_dt.date()} → {end_disp}  (+{RUNOUT_DAYS}d runout)"))
        print(_box(f"cache         : {'ON' if use_cache else 'OFF (--no-cache)'}"))
        print(f"├{'─' * (W + 2)}┤")

        candle_data: dict = {}
        n_hits = n_miss = 0
        _miss_pairs: list = []

        for pair in WATCHLIST:
            pair_tfs: dict = {}
            pair_failed = False

            for tf, (fs, fe) in tf_ranges.items():
                hit_df = _pair_cache_load(pair, tf, fs, fe) if use_cache else None
                if hit_df is not None:
                    n_hits += 1
                    hit_info = (f"{pair:<10} {tf:<4} ✓ HIT   "
                                f"({len(hit_df):5d} bars  "
                                f"{hit_df.index.min().date()} → {hit_df.index.max().date()})")
                    print(_box(hit_info))
                    pair_tfs[tf] = hit_df
                else:
                    n_miss += 1
                    _miss_pairs.append(f"{pair}/{tf}")
                    print(_box(f"{pair:<10} {tf:<4} ✗ MISS  → fetching …"), flush=True)
                    # Fetch from OANDA
                    if use_range:
                        df = _counted_fetch_range(pair, tf, from_dt=fs, to_dt=fe)
                    else:
                        count = {"H1": 5000, "H4": 1500, "D": 500}.get(tf, 500)
                        df = _counted_fetch(pair, tf, count)

                    if df is not None and len(df) >= (50 if tf == "H1" else 1):
                        if use_cache:
                            _pair_cache_save(pair, tf, df)
                        pair_tfs[tf] = df
                        fetch_info = (f"  └─ fetched {len(df)} bars  "
                                      f"({df.index.min().date()} → {df.index.max().date()})"
                                      f"  saved={'yes' if use_cache else 'no'}")
                        print(_box(fetch_info))
                        time.sleep(0.3)   # OANDA rate limit
                    else:
                        print(_box("  └─ ✗ FAILED or insufficient data"))
                        if tf == "H1":
                            pair_failed = True
                            break

            if pair_failed or "H1" not in pair_tfs or len(pair_tfs["H1"]) < 50:
                print(_box(f"{pair:<10} ✗ skipped (insufficient H1 data)"))
                continue

            df_w = _resample_weekly(pair_tfs["D"]) if pair_tfs.get("D") is not None else None
            candle_data[pair] = {
                "1h": pair_tfs["H1"],
                "4h": pair_tfs.get("H4"),
                "d":  pair_tfs.get("D"),
                "w":  df_w,
            }

        # ── Cache summary ─────────────────────────────────────────────
        total_tfs = n_hits + n_miss
        print(f"├{'─' * (W + 2)}┤")
        summary = (f"{n_hits}/{total_tfs} TF slots cached  |  "
                   f"{n_miss} MISS  |  API calls: {_api_call_count}  |  "
                   f"{len(candle_data)}/{len(WATCHLIST)} pairs loaded")
        print(_box(summary))
        if _miss_pairs:
            miss_str = ", ".join(_miss_pairs[:8]) + (f" +{len(_miss_pairs)-8}" if len(_miss_pairs) > 8 else "")
            print(_box(f"Fetched: {miss_str}"))
        print(f"└{'─' * (W + 2)}┘")

    if not candle_data:
        print("No data loaded. Exiting.")
        return

    # Build unified 1H timeline from start_dt through runout_dt.
    # Entries are gated at end_naive in the main loop; the extra bars let
    # open positions close via target/stop rather than being force-closed.
    start_naive  = start_dt.replace(tzinfo=None)
    runout_naive = runout_dt.replace(tzinfo=None) if runout_dt else None
    all_1h = sorted(set(
        ts for pdata in candle_data.values()
        for ts in pdata["1h"].index
        if ts >= start_naive and (runout_naive is None or ts <= runout_naive)
    ))
    # ── Backtest whitelist filter ─────────────────────────────────────
    # Reads logs/whitelist_backtest.json when enabled — lets you run Alex-only
    # or any subset without re-fetching data. Skipped pairs keep their cache.
    if WHITELIST_BACKTEST_FILE.exists():
        try:
            _wl = json.load(open(WHITELIST_BACKTEST_FILE))
            if _wl.get("enabled") and _wl.get("pairs"):
                _wl_pairs = set(_wl["pairs"])
                _before   = len(candle_data)
                candle_data = {p: v for p, v in candle_data.items() if p in _wl_pairs}
                _filtered  = _before - len(candle_data)
                print(f"\n  🔒 Backtest whitelist ACTIVE ({len(candle_data)}/{_before} pairs, "
                      f"{_filtered} filtered): {', '.join(sorted(candle_data))}")
                # Recompute all_1h after filter
                all_1h = sorted(set(
                    ts for pdata in candle_data.values()
                    for ts in pdata["1h"].index
                    if ts >= start_naive and (runout_naive is None or ts <= runout_naive)
                ))
        except Exception as _e:
            print(f"  ⚠ Could not load backtest whitelist: {_e}")

    print(f"\nPairs loaded: {len(candle_data)}")
    print(f"Backtesting {len(all_1h)} hourly bars: "
          f"{all_1h[0].date()} → {all_1h[-1].date()}")
    print(f"Starting balance: ${starting_bal:,.2f}\n")

    # ── Strategy instances (one per pair for independent state) ───────
    strategies: Dict[str, SetAndForgetStrategy] = {}
    for pair in candle_data:
        s = SetAndForgetStrategy(account_balance=starting_bal, risk_pct=15.0)
        s.news_filter = _HistoricalNewsFilter()   # CSV-backed historical filter
        strategies[pair] = s

    # ── Risk manager ──────────────────────────────────────────────────
    # backtest=True: disable all disk I/O (never write regroup_state.json /
    # kill_switch.log / trade_journal.jsonl — backtest must not corrupt live state)
    journal = TradeJournal()
    risk    = ForexRiskManager(journal=journal, backtest=True)

    # ── State ─────────────────────────────────────────────────────────
    balance       = starting_bal
    peak_balance  = starting_bal    # tracks equity high-water mark for DD circuit breaker
    open_pos: Dict[str, dict] = {}   # pair → position dict
    trades        = []
    all_decisions = []
    v1_decisions  = _load_v1_decisions()
    consecutive_losses: int = 0     # streak counter: reset on win, incremented on loss
    dd_killswitch_blocks: int = 0   # count of entries blocked by 40% DD killswitch
    _eval_calls: int = 0            # performance counter: evaluate() call count
    _eval_ms:    float = 0.0        # total ms spent in evaluate()
    # Regime mode at entry: instantaneous snapshot (no hysteresis state).
    # Alex's ~1 trade/week cadence is too sparse for 2-bar consecutive
    # accumulation across entries.  compute_risk_mode(instantaneous=True)
    # grants HIGH/EXTREME immediately when ALL conditions are met at the
    # entry bar — no consec/demote state carried between entries.
    # 2-bar hysteresis runs only in the H4 time-sampling loop below.

    # ── Alex small-account gate counters ───────────────────────────────────
    _weekly_trade_counts:   dict = {}  # {(iso_year, iso_week): count} — opened trades
    _weekly_limit_blocks:   int  = 0   # blocked by MAX_TRADES_PER_WEEK
    _min_rr_small_blocks:   int  = 0   # blocked by MIN_RR_ALIGN (non-protrend)
    _adaptive_time_blocks:  int  = 0   # blocked by adaptive Thu/Fri regime gate
    _strict_htf_blocks:     int  = 0   # blocked by strict protrend HTF gate (all 3 must agree)
    _wd_htf_blocks:         int  = 0   # blocked by W+D protrend gate (W==D required, 4H exempt)
    _dyn_pip_eq_blocks:     int  = 0   # blocked by dynamic pip equity (stop_pips × MIN_RR)
    _wd_aligned_entries:    int  = 0   # entered trades where W==D agreed with direction
    _countertrend_htf_blocks: int = 0  # blocked by COUNTERTREND_HTF filter
    _time_block_counts:     dict = {}  # {reason_code: count} — Sunday/Thu/Fri blocks

    # ── Chop Shield state (Arm D) ──────────────────────────────────────────
    # consecutive_losses already tracks the running streak (see line ~845).
    _bt_chop_pause_until:     Optional[datetime] = None  # None = not in 48h pause
    _bt_chop_auto_pauses:     int = 0    # times streak≥3 armed the shield
    _bt_chop_paused_blocks:   int = 0    # entries blocked during active 48h pause
    _bt_chop_recovery_blocks: int = 0    # entries blocked by Part B recovery gates

    def _risk_pct(bal):
        """DD-aware risk. When flat_risk_pct is set, overrides tiers (killswitch still applies)."""
        pct, _dd_flag = _risk_pct_with_flag(bal)
        return pct

    def _risk_pct_with_flag(bal):
        """Returns (pct, dd_flag). flat_risk_pct overrides tier pct; killswitch wins."""
        pct, dd_flag = risk.get_risk_pct_with_dd(
            bal, peak_equity=peak_balance, consecutive_losses=consecutive_losses)
        if flat_risk_pct is not None and dd_flag != "DD_KILLSWITCH":
            pct = flat_risk_pct   # grid test: flat rate, no tier ramp
        return pct, dd_flag

    def _calc_units(pair, bal, rpct, entry, stop):
        pip    = 0.01 if "JPY" in pair else 0.0001
        dist   = abs(entry - stop)
        if dist == 0: return 0
        risk_usd = bal * rpct / 100
        return int(risk_usd / dist)

    def _spread_deduction(pair: str, units: int) -> float:
        """
        Round-trip spread cost in dollars for one trade.

        Formula: spread_pips × pip_mult × units
          LONG:  buy at ask, sell at bid → pay spread on entry + spread on exit
          SHORT: sell at bid, buy at ask → same cost
          v1 simplification: apply full round-turn as a P&L deduction at close.

        Returns 0.0 when SPREAD_MODEL_ENABLED is False.
        """
        if not _sc.SPREAD_MODEL_ENABLED:
            return 0.0
        pair_key  = pair.replace("_", "/")
        spread_p  = _sc.SPREAD_PIPS.get(pair_key, _sc.SPREAD_DEFAULT_PIPS)
        pip_mult  = 0.01 if "JPY" in pair else 0.0001
        return spread_p * pip_mult * units

    def _daily_atr(pair) -> Optional[float]:
        """14-day ATR for the pair (in price terms, not pips)."""
        df_d = candle_data[pair].get("d")
        if df_d is None or len(df_d) < _sc.ATR_LOOKBACK + 1:
            return None
        recent = df_d.tail(_sc.ATR_LOOKBACK + 1)
        tr_list = []
        for i in range(1, len(recent)):
            h = float(recent["high"].iloc[i])
            l = float(recent["low"].iloc[i])
            pc = float(recent["close"].iloc[i - 1])
            tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))
        return float(np.mean(tr_list)) if tr_list else None

    def _stop_ok(pair, entry, stop):
        """
        Return True if stop distance is within acceptable ATR range.
        Max: ≤ ATR_STOP_MULTIPLIER × 14-day ATR (rejects ancient structural stops)
        Min: ≥ ATR_MIN_MULTIPLIER × 14-day ATR (blocks micro-stops from noise entries).
        Both thresholds read live from _sc so --lever overrides propagate.
        """
        atr = _daily_atr(pair)
        if atr is None:
            return True
        dist = abs(entry - stop)
        max_dist = atr * _sc.ATR_STOP_MULTIPLIER
        min_dist = atr * _sc.ATR_MIN_MULTIPLIER
        return min_dist <= dist <= max_dist

    def _pip_equity(decision, pair: str) -> float:
        """
        Measured move in pips for a pattern entry.
        Used to rank competing entry candidates — highest pip equity gets priority.
        Measured move = |neckline − target_1| (conservative target, 1:1 R:R minimum).
        Exception: consolidation_breakout uses target_2 (2× measured move) because
        T1 = 1× range which is small by construction. Alex consistently runs these
        for 2-3× the range — T2 reflects actual pip opportunity more accurately.
        """
        if not decision.pattern or not decision.pattern.target_1:
            return 0.0
        is_cb = 'consolidation_breakout' in (decision.pattern.pattern_type or '')
        target = decision.pattern.target_2 if (is_cb and decision.pattern.target_2) else decision.pattern.target_1
        raw = abs(decision.pattern.neckline - target)
        mult = 100.0 if "JPY" in pair.upper() else 10000.0
        return raw * mult

    def _currencies_in_use():
        """Set of individual currency codes currently in open positions."""
        ccys = set()
        for p in open_pos:
            a, b = p.split("/")
            ccys.add(a); ccys.add(b)
        return ccys

    def _pair_currencies(pair):
        a, b = pair.split("/")
        return {a, b}

    # ── Macro theme detector ──────────────────────────────────────────────────
    theme_analyzer = CurrencyStrengthAnalyzer()
    _theme_cache: Dict[str, Optional[CurrencyTheme]] = {}   # "YYYY-MM-DD" → theme

    def _get_theme(ts: pd.Timestamp) -> Optional[CurrencyTheme]:
        """Compute macro theme from daily data sliced up to ts. Cached per day."""
        date_key = ts.strftime("%Y-%m-%d")
        if date_key in _theme_cache:
            return _theme_cache[date_key]
        snapshot: Dict[str, dict] = {}
        for p, pdata in candle_data.items():
            df_d = pdata.get("d")
            if df_d is not None:
                sliced = df_d[df_d.index < ts]
                if len(sliced) >= 22:    # need at least 20 days of closes for momentum
                    snapshot[p] = {"d": sliced}
        theme = theme_analyzer.get_dominant_theme(snapshot) if snapshot else None
        _theme_cache[date_key] = theme
        return theme

    def _entry_eligible(pair, macro_theme: Optional[CurrencyTheme] = None,
                        direction: str = ""):
        """4-layer eligibility gate matching live risk_manager_forex logic.

        Macro theme exception: when a currency theme is active and this pair
        is one of the suggested stacked trades, we:
          (a) raise the concurrent cap to STACK_MAX (default 4), and
          (b) waive the currency-overlap check — correlated exposure is intentional.
        This is exactly how Alex stacked 4 JPY shorts in Week 7-8 for $70K.

        Layer 4 (open-position theme gate): if ≥2 open positions express the same
        directional bias on a currency, block any new trade that contradicts it.
        E.g. GBP/JPY SHORT + USD/JPY SHORT → JPY LONG theme → block NZD/JPY LONG,
        allow NZD/JPY SHORT. Only runs when direction is provided (post-decision).
        """
        _is_theme = (
            macro_theme is not None
            and pair in [p for p, _ in macro_theme.suggested_trades]
        )
        max_concurrent = STACK_MAX if _is_theme else _sc.MAX_CONCURRENT_TRADES_BACKTEST

        # Layer 1: max concurrent
        if _is_theme:
            if len(open_pos) >= max_concurrent:
                return False, "max_concurrent"
        else:
            non_theme_count = sum(
                1 for p in open_pos
                if open_pos[p].get("macro_theme") is None
                and not open_pos[p].get("be_moved", False)
            )
            if non_theme_count >= _sc.MAX_CONCURRENT_TRADES_BACKTEST:
                return False, "max_concurrent"

        # Layer 2: same-pair block
        if pair in open_pos:
            return False, "same_pair_open"

        # Layer 3 (theme contradiction gate — only when direction is known)
        # Only applies to macro carry/safe-haven currencies (JPY, CHF) where a
        # directional theme is meaningful. USD, EUR, GBP etc. selling in two pairs
        # is coincidental (e.g. USD/JPY + USD/CHF both sell USD but that's not a
        # "USD is crashing" macro theme — it's JPY strong + CHF strong individually).
        MACRO_THEME_CCYS = {"JPY", "CHF"}

        if direction and "/" in pair and len(open_pos) >= 2:
            from collections import Counter
            ccy_bias: Counter = Counter()
            for op, pos in open_pos.items():
                if "/" not in op:
                    continue
                ob, oq = op.split("/")
                bias = 1 if pos.get("direction") == "long" else -1
                ccy_bias[ob] += bias
                ccy_bias[oq] -= bias

            base, quote = pair.split("/")
            new_bias = 1 if direction == "long" else -1
            THEME_THRESHOLD = 2

            for ccy, trade_impact in [(base, new_bias), (quote, -new_bias)]:
                if ccy not in MACRO_THEME_CCYS:
                    continue   # only gate on macro carry currencies
                existing = ccy_bias.get(ccy, 0)
                if existing >= THEME_THRESHOLD and trade_impact < 0:
                    return False, f"theme_contradiction:{ccy}_long_theme"
                if existing <= -THEME_THRESHOLD and trade_impact > 0:
                    return False, f"theme_contradiction:{ccy}_short_theme"

        return True, ""

    # ── Main loop ─────────────────────────────────────────────────────
    prev_pct = -1
    for bar_idx, ts in enumerate(all_1h):
        pct = int(bar_idx / len(all_1h) * 100)
        if pct != prev_pct and pct % 10 == 0:
            open_count = len(open_pos)
            print(f"  {pct}%... (balance ${balance:,.2f}, open={open_count}, "
                  f"trades={len(trades)})")
            prev_pct = pct

        ts_utc = ts.to_pydatetime().replace(tzinfo=timezone.utc)

        # ── Monitor open positions ────────────────────────────────────
        for pair in list(open_pos.keys()):
            if pair not in candle_data: continue
            df_1h_p = candle_data[pair]["1h"]
            if ts not in df_1h_p.index: continue

            bar       = df_1h_p.loc[ts]
            pos       = open_pos[pair]
            entry     = pos["entry_price"]
            stop      = pos["stop_loss"]
            direction = pos["direction"]
            units     = pos["units"]
            bars_held = bar_idx - pos["bar_idx"]
            pip       = 0.01 if "JPY" in pair else 0.0001

            high  = float(bar["high"])
            low   = float(bar["low"])
            close = float(bar["close"])

            # ── Parameterized trailing stop ───────────────────────────────────
            # Trail behaviour is controlled by _tcfg (trail arm config dict).
            #
            # Two-stage trail (Arm C):
            #   Stage 1 at TRAIL_ACTIVATE_R: lock stop to entry + TRAIL_LOCK_R once
            #   Stage 2 at TRAIL_STAGE2_R:   trail continuously at trail_max ± TRAIL_STAGE2_DIST_R
            #
            # Standard trail (Arms A/B):
            #   At TRAIL_ACTIVATE_R: trail at trail_max ± TRAIL_LOCK_R, floor at entry+TRAIL_LOCK_R
            #   (TRAIL_STAGE2_R is None → stage 2 disabled)
            risk_dist        = pos.get("initial_risk") or abs(entry - stop)
            _act_r           = _tcfg.get("TRAIL_ACTIVATE_R", 1.0)
            _lock_r          = _tcfg.get("TRAIL_LOCK_R",     0.5)
            _stage2_r        = _tcfg.get("TRAIL_STAGE2_R")        # None → one-stage
            _stage2_dist_r   = _tcfg.get("TRAIL_STAGE2_DIST_R",  _lock_r)

            _act_dist        = _act_r    * risk_dist
            _lock_dist       = _lock_r   * risk_dist
            _stage2_dist     = _stage2_dist_r * risk_dist if _stage2_r else None

            if direction == "long":
                if high > pos.get("trail_max", entry):
                    pos["trail_max"] = high
                _mfe = pos["trail_max"] - entry

                if _mfe >= _act_dist:
                    if _stage2_r is None:
                        # Standard trail: trail continuously (Arms A/B)
                        _new_stop = pos["trail_max"] - _lock_dist
                        _new_stop = max(_new_stop, entry + _lock_dist)  # floor at +lock_r
                        if _new_stop > stop:
                            stop = _new_stop
                            pos["stop_loss"] = stop
                            pos["ratchet_moved"] = True
                    else:
                        # Two-stage trail (Arm C)
                        # Stage 1: lock stop once at entry + TRAIL_LOCK_R
                        if not pos.get("trail_locked"):
                            _lock_stop = entry + _lock_dist
                            if _lock_stop > stop:
                                stop = _lock_stop
                                pos["stop_loss"] = stop
                                pos["ratchet_moved"] = True
                            pos["trail_locked"] = True
                        # Stage 2: trail from trail_max − TRAIL_STAGE2_DIST_R
                        if _mfe >= _stage2_r * risk_dist:
                            _new_stop = pos["trail_max"] - _stage2_dist
                            _new_stop = max(_new_stop, stop)  # never go backward
                            if _new_stop > stop:
                                stop = _new_stop
                                pos["stop_loss"] = stop
                                pos["ratchet_moved"] = True

            else:  # short
                if low < pos.get("trail_max", entry):
                    pos["trail_max"] = low
                _mfe = entry - pos["trail_max"]

                if _mfe >= _act_dist:
                    if _stage2_r is None:
                        # Standard trail: trail continuously (Arms A/B)
                        _new_stop = pos["trail_max"] + _lock_dist
                        _new_stop = min(_new_stop, entry - _lock_dist)  # ceiling at −lock_r
                        if _new_stop < stop:
                            stop = _new_stop
                            pos["stop_loss"] = stop
                            pos["ratchet_moved"] = True
                    else:
                        # Two-stage trail (Arm C)
                        if not pos.get("trail_locked"):
                            _lock_stop = entry - _lock_dist
                            if _lock_stop < stop:
                                stop = _lock_stop
                                pos["stop_loss"] = stop
                                pos["ratchet_moved"] = True
                            pos["trail_locked"] = True
                        if _mfe >= _stage2_r * risk_dist:
                            _new_stop = pos["trail_max"] + _stage2_dist
                            _new_stop = min(_new_stop, stop)  # never go backward
                            if _new_stop < stop:
                                stop = _new_stop
                                pos["stop_loss"] = stop
                                pos["ratchet_moved"] = True

            # ── Target-price exit (Alex's manual close) ───────────────────────
            # Alex closes when price reaches his "next 4H structure low/high"
            # target. He actively monitors: if price hits the target area and
            # bounces he closes — especially going into the weekend.
            # Two exit triggers:
            #   1. at_target  — High/Low touches within TARGET_PROXIMITY_PIPS
            #   2. late_week  — Thu/Fri PM, profitable, and ≥70% of the way to
            #                   target ("don't want it to go into the weekend")
            target = pos.get("target_price")
            if target and target != 0:
                pip_sz    = 0.01 if "JPY" in pair else 0.0001
                near_pips = getattr(_sc, "TARGET_PROXIMITY_PIPS", 15.0)
                at_target = (
                    (direction == "short" and low  <= target + near_pips * pip_sz) or
                    (direction == "long"  and high >= target - near_pips * pip_sz)
                )
                # Weekend proximity close
                is_late_week = ts_utc.weekday() in (3, 4) and ts_utc.hour >= 15
                if is_late_week and not at_target:
                    dist_total   = abs(entry - target)
                    dist_covered = ((entry - close) if direction == "short"
                                    else (close - entry))
                    at_target = (dist_total > 0
                                 and dist_covered / dist_total >= 0.70
                                 and dist_covered > 0)

                if at_target:
                    close_price  = target if not is_late_week else close
                    delta        = ((entry - close_price) if direction == "short"
                                    else (close_price - entry))
                    spread_cost  = _spread_deduction(pair, units)
                    pnl = delta * units - spread_cost
                    balance += pnl
                    peak_balance = max(peak_balance, balance)  # DD circuit breaker HWM
                    close_reason = ("weekend_proximity" if is_late_week
                                    else "target_reached")
                    risk_r = (pnl / pos.get("entry_risk_dollars", 1)
                              if pos.get("entry_risk_dollars") else delta / risk_dist if risk_dist else 0)
                    consecutive_losses = 0   # target/weekend win resets streak
                    _bt_chop_pause_until = None   # win clears chop shield pause
                    trades.append({
                        "pair":        pair,   "direction": direction,
                        "entry":       entry,  "exit":      close_price,
                        "pnl":         round(pnl, 2),
                        "spread_cost": round(spread_cost, 2),
                        "r":           risk_r,
                        "reason":      close_reason,
                        "entry_ts":    pos["entry_ts"].isoformat(),
                        "exit_ts":     ts_utc.isoformat(),
                        "bars_held":   bars_held,
                        "pattern":     pos.get("pattern", "?"),
                        "notes":       pos.get("notes", ""),
                        "macro_theme": pos.get("macro_theme"),
                        "signal_type":  pos.get("signal_type", "unknown"),
                        "planned_rr":   pos.get("planned_rr", 0.0),
                        "stop_type":    pos.get("stop_type", "unknown"),
                        "initial_stop_pips": pos.get("initial_stop_pips", 0),
                        "mfe_r":        pos.get("mfe_r", 0.0),
                        "mae_r":        pos.get("mae_r", 0.0),
                        "dd_flag":      pos.get("dd_flag", ""),
                        "streak_at_entry":    pos.get("streak_at_entry", 0),
                        "entry_equity":       pos.get("entry_equity", 0),
                        "final_risk_pct":     pos.get("final_risk_pct", 0),
                        "entry_risk_dollars": pos.get("entry_risk_dollars", 0),
                        "target_1":     target,
                        "regime_score_at_entry": pos.get("regime_score", 0.0),
                        "risk_mode_at_entry": pos.get("risk_mode_at_entry", "MEDIUM"),
                    })
                    r_sign   = "+" if risk_r >= 0 else ""
                    pnl_sign = "+" if pnl >= 0 else ""
                    print(f"  {'✅' if pnl >= 0 else '❌'} {ts_utc.strftime('%Y-%m-%d')} "
                          f"| EXIT {pair} {direction.upper()} "
                          f"@ {close_price:.5f}  {r_sign}{risk_r:.1f}R  "
                          f"${pnl_sign}{pnl:,.2f}  [{close_reason}]")
                    strategies[pair].close_position(pair, close_price)
                    del open_pos[pair]
                    continue

            # Track MFE / MAE every bar (in R units)
            _init_r = pos.get("initial_risk") or 1e-9
            _fav  = ((high - entry) if direction == "long" else (entry - low))  / _init_r
            _adv  = ((entry - low)  if direction == "long" else (high - entry)) / _init_r
            pos["mfe_r"] = max(pos.get("mfe_r", 0.0), _fav)
            pos["mae_r"] = min(pos.get("mae_r", 0.0), -abs(_adv))

            # ── Stall exit (Arm C3) ───────────────────────────────────
            # If trail config specifies STALL_EXIT_BARS, check at exactly that
            # bar-count whether MFE has reached the minimum threshold.
            # Purpose: free capital from stagnant trades early.
            _stall_bars = _tcfg.get("STALL_EXIT_BARS")
            if _stall_bars and bars_held == _stall_bars:
                _stall_mfe_r   = _tcfg.get("STALL_EXIT_MFE_R", 0.5)
                _cur_mfe       = pos.get("mfe_r", 0.0)
                if _cur_mfe < _stall_mfe_r and not pos.get("trail_locked"):
                    exit_p      = close
                    delta       = (exit_p - entry) if direction == "long" else (entry - exit_p)
                    spread_cost = _spread_deduction(pair, units)
                    pnl         = delta * units - spread_cost
                    balance    += pnl
                    peak_balance = max(peak_balance, balance)
                    risk_r = (pnl / pos.get("entry_risk_dollars", 1)
                              if pos.get("entry_risk_dollars") else
                              delta / risk_dist if risk_dist else 0)
                    if risk_r < -0.10:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                        _bt_chop_pause_until = None   # win/scratch clears chop shield
                    trades.append({
                        "pair": pair, "direction": direction,
                        "entry": entry, "exit": exit_p,
                        "pnl": pnl, "r": risk_r, "reason": "stall_exit",
                        "spread_cost": round(spread_cost, 2),
                        "entry_ts": pos["entry_ts"].isoformat(),
                        "exit_ts": ts_utc.isoformat(),
                        "bars_held": bars_held,
                        "pattern": pos.get("pattern", "?"),
                        "notes": pos.get("notes", ""),
                        "macro_theme": pos.get("macro_theme"),
                        "signal_type":  pos.get("signal_type", "unknown"),
                        "planned_rr":   pos.get("planned_rr", 0.0),
                        "stop_type":    pos.get("stop_type", "unknown"),
                        "initial_stop_pips": pos.get("initial_stop_pips", 0),
                        "mfe_r": _cur_mfe, "mae_r": pos.get("mae_r", 0.0),
                        "dd_flag": pos.get("dd_flag", ""),
                        "streak_at_entry":    pos.get("streak_at_entry", 0),
                        "entry_equity":       pos.get("entry_equity", 0),
                        "final_risk_pct":     pos.get("final_risk_pct", 0),
                        "entry_risk_dollars": pos.get("entry_risk_dollars", 0),
                        "target_1": pos.get("target_price"),
                        "regime_score_at_entry": pos.get("regime_score", 0.0),
                        "risk_mode_at_entry": pos.get("risk_mode_at_entry", "MEDIUM"),
                    })
                    print(f"  ⏹ {ts_utc.strftime('%Y-%m-%d')} "
                          f"| STALL {pair} {direction.upper()} "
                          f"@ {exit_p:.5f}  {'+' if risk_r>=0 else ''}{risk_r:.1f}R  "
                          f"(MFE={_cur_mfe:.2f}R < {_stall_mfe_r}R at bar {_stall_bars})")
                    strategies[pair].close_position(pair, exit_p)
                    del open_pos[pair]
                    continue

            # Stop hit?
            stopped = (direction == "long"  and low  <= stop) or \
                      (direction == "short" and high >= stop)
            max_hold = bars_held >= MAX_HOLD_BARS

            if stopped or max_hold:
                exit_p      = stop if stopped else close
                delta       = (exit_p - entry) if direction == "long" else (entry - exit_p)
                spread_cost = _spread_deduction(pair, units)
                pnl         = delta * units - spread_cost
                balance    += pnl
                peak_balance = max(peak_balance, balance)  # DD circuit breaker HWM
                if stopped and pos.get("ratchet_moved"):
                    reason = "ratchet_stop_hit"
                elif stopped:
                    reason = "stop_hit"
                else:
                    reason = "max_hold"
                risk_r = (pnl / pos.get("entry_risk_dollars", 1)
                          if pos.get("entry_risk_dollars") else delta / risk_dist if risk_dist else 0)

                # ── Streak tracking ───────────────────────────────────
                if risk_r < -0.10:   # definite loss
                    consecutive_losses += 1
                else:                # win or scratch → reset streak
                    consecutive_losses = 0
                    _bt_chop_pause_until = None   # win clears chop shield pause

                trades.append({
                    "pair": pair, "direction": direction,
                    "entry": entry, "exit": exit_p,
                    "pnl": pnl, "r": risk_r, "reason": reason,
                    "spread_cost": round(spread_cost, 2),
                    "entry_ts": pos["entry_ts"].isoformat(),
                    "exit_ts": ts_utc.isoformat(),
                    "bars_held": bars_held,
                    "pattern": pos.get("pattern", "?"),
                    "notes": pos.get("notes", ""),
                    "macro_theme": pos.get("macro_theme"),
                    "signal_type":  pos.get("signal_type", "unknown"),
                    "planned_rr":   pos.get("planned_rr", 0.0),
                    "stop_type":    pos.get("stop_type", "unknown"),
                    "initial_stop_pips": pos.get("initial_stop_pips", 0),
                    "mfe_r":        pos.get("mfe_r", 0.0),
                    "mae_r":        pos.get("mae_r", 0.0),
                    "dd_flag":      pos.get("dd_flag", ""),
                    "streak_at_entry":    pos.get("streak_at_entry", 0),
                    "entry_equity":       pos.get("entry_equity", 0),
                    "final_risk_pct":     pos.get("final_risk_pct", 0),
                    "entry_risk_dollars": pos.get("entry_risk_dollars", 0),
                    "regime_score_at_entry": pos.get("regime_score", 0.0),
                        "risk_mode_at_entry": pos.get("risk_mode_at_entry", "MEDIUM"),
                })

                r_sign = "+" if risk_r >= 0 else ""
                pnl_sign = "+" if pnl >= 0 else ""
                print(f"  {'✅' if pnl >= 0 else '❌'} {ts_utc.strftime('%Y-%m-%d')} "
                      f"| EXIT {pair} {direction.upper()} "
                      f"@ {exit_p:.5f}  {r_sign}{risk_r:.1f}R  "
                      f"${pnl_sign}{pnl:,.2f}  [{reason}]")

                strategies[pair].close_position(pair, exit_p)
                if stopped:
                    strategies[pair].record_stop_out(pair, ts_utc)
                del open_pos[pair]
                continue

        # ── Evaluate new entries ──────────────────────────────────────
        # After the entry window closes (ts > end_naive): stop taking new trades.
        # Existing positions continue to monitor and close via target or stop.
        # This eliminates "open_at_end" phantom P&L — results are fully realized.
        in_entry_window = (end_naive is None or ts <= end_naive)
        if not in_entry_window:
            if not open_pos:
                break  # all positions closed, no new entries — done
            continue   # keep monitoring open positions, skip entry evaluation

        # Macro theme: computed once per calendar day (daily data doesn't change intraday)
        active_theme = _get_theme(ts)

        # ── Max unrealized R across all open positions (for winner rule) ──
        # Computed once per bar using current 1H close — NOT the be_moved flag.
        # be_moved is set at 1R and stays True forever even if price drifts back
        # to entry. We want "actively up X R right now", not "was once at 1R".
        _max_open_r = 0.0
        for _wp, _wpos in open_pos.items():
            if _wp not in candle_data:
                continue
            _wdf = candle_data[_wp]["1h"]
            if ts not in _wdf.index:
                continue
            _wclose    = float(_wdf.loc[ts]["close"])
            _wentry    = _wpos["entry_price"]
            _wstop     = _wpos["stop_loss"]
            _wrisk     = abs(_wentry - _wstop)
            if _wrisk == 0:
                continue
            if _wpos["direction"] == "long":
                _wr = (_wclose - _wentry) / _wrisk
            else:
                _wr = (_wentry - _wclose) / _wrisk
            if _wr > _max_open_r:
                _max_open_r = _wr

        # ── Phase 1: evaluate all pairs, collect entry candidates ────────
        # Evaluate every eligible pair, log decisions, then sort ENTER
        # candidates by pip equity before filling slots.
        # This ensures highest-potential setups get priority when slots are scarce.
        entry_candidates = []  # (pip_equity, pair, decision, _is_theme_pair)

        for pair, pdata in candle_data.items():
            if pair not in _sc.ALLOWED_PAIRS:
                continue   # not in Alex's trading universe

            if pair in open_pos:
                continue   # already in this pair

            _is_theme_pair = (
                active_theme is not None
                and pair in [p for p, _ in active_theme.suggested_trades]
            )

            # Eligibility gate (mirrors live risk_manager_forex)
            eligible, elig_reason = _entry_eligible(pair, active_theme)
            if not eligible:
                continue

            df_1h_p = pdata["1h"]
            df_4h_p = pdata.get("4h")
            df_d_p  = pdata.get("d")
            df_w_p  = pdata.get("w")

            hist_1h = df_1h_p[df_1h_p.index < ts]
            hist_4h = df_4h_p[df_4h_p.index < ts] if df_4h_p is not None else pd.DataFrame()
            hist_d  = df_d_p[df_d_p.index < ts]   if df_d_p  is not None else pd.DataFrame()
            hist_w  = df_w_p[df_w_p.index < ts]   if df_w_p  is not None else pd.DataFrame()

            if len(hist_1h) < 20 or len(hist_4h) < 10:
                continue

            strat = strategies[pair]
            strat.account_balance = balance
            strat.risk_pct = _risk_pct(balance)

            try:
                _t0 = time.perf_counter()
                decision = strat.evaluate(
                    pair        = pair,
                    df_weekly   = hist_w  if len(hist_w)  >= 4  else pd.DataFrame(),
                    df_daily    = hist_d  if len(hist_d)  >= 10 else pd.DataFrame(),
                    df_4h       = hist_4h,
                    df_1h       = hist_1h,
                    current_dt  = ts_utc,
                    macro_theme = active_theme if _is_theme_pair else None,
                )
                _eval_calls += 1
                _eval_ms    += (time.perf_counter() - _t0) * 1000
            except Exception:
                continue

            # Log all decisions (WAIT / BLOCKED / ENTER)
            all_decisions.append({
                "ts":            ts_utc.isoformat(),
                "pair":          pair,
                "decision":      decision.decision.value,
                "confidence":    decision.confidence,
                "reason":        decision.reason,
                "direction":     decision.direction,
                "entry_price":   decision.entry_price,
                "stop_loss":     decision.stop_loss,
                "stop_type":     getattr(decision, "stop_type", ""),
                "initial_stop_pips": getattr(decision, "initial_stop_pips", 0),
                "exec_rr":       getattr(decision, "exec_rr", 0),
                "failed_filters": decision.failed_filters,
                "balance":       balance,
            })

            # Gap analysis vs v1
            v1_key = (pair, ts_utc.isoformat())
            v1_dec = v1_decisions.get(v1_key)
            if v1_dec:
                v1_action = v1_dec.get("decision", "UNKNOWN")
                v2_action = decision.decision.value
                if v1_action != v2_action:
                    gap_type = (
                        "v1_entered_v2_blocked"  if v1_action == "ENTER" else
                        "v2_entered_v1_missed"   if v2_action == "ENTER" else
                        "confidence_divergence"
                    )
                    log_gap(ts_utc, pair, v1_action, v2_action, gap_type,
                            f"v2 reason: {decision.reason[:120]}")

            if decision.decision != Decision.ENTER:
                continue

            # ── Gates that apply only to ENTER decisions ──────────────

            # Theme direction gate
            if _sc.REQUIRE_THEME_GATE and _is_theme_pair and active_theme:
                theme_dir_map = dict(active_theme.suggested_trades)
                theme_dir     = theme_dir_map.get(pair)
                if theme_dir and theme_dir != decision.direction:
                    log_gap(ts_utc, pair, "ENTER", "BLOCKED",
                            "theme_direction_conflict",
                            f"Theme={active_theme.currency}_{active_theme.direction} "
                            f"wants {theme_dir}, pattern wants {decision.direction}. "
                            f"Blocked to avoid contradicting macro view.")
                    continue

            # Confidence gate
            if decision.confidence < _sc.MIN_CONFIDENCE:
                log_gap(ts_utc, pair, "ENTER", "BLOCKED", "low_confidence",
                        f"Confidence {decision.confidence:.0%} < {_sc.MIN_CONFIDENCE:.0%} threshold  "
                        f"entry={decision.entry_price:.5f}  reason={decision.reason[:60]}")
                continue

            # Winner rule
            win_blocked, win_reason = winner_rule_check(
                n_open=len(open_pos), max_unrealized_r=_max_open_r,
            )
            if win_blocked:
                log_gap(ts_utc, pair, "ENTER", "BLOCKED", "winner_rule", win_reason)
                continue

            # Stop-distance guard
            if (decision.entry_price and decision.stop_loss
                    and not _stop_ok(pair, decision.entry_price, decision.stop_loss)):
                atr  = _daily_atr(pair)
                dist = abs(decision.entry_price - decision.stop_loss)
                pip  = 0.01 if "JPY" in pair else 0.0001
                too_wide = atr is not None and dist > atr * _sc.ATR_STOP_MULTIPLIER
                gap_type = "stop_too_wide" if too_wide else "stop_too_tight"
                msg = (
                    f"Stop too wide: {dist/pip:.0f}p > max {atr*_sc.ATR_STOP_MULTIPLIER/pip:.0f}p "
                    f"({_sc.ATR_STOP_MULTIPLIER:.0f}×ATR={atr/pip:.0f}p)" if too_wide else
                    f"Stop too tight: {dist/pip:.0f}p < min {atr*_sc.ATR_MIN_MULTIPLIER/pip:.0f}p "
                    f"({_sc.ATR_MIN_MULTIPLIER:.2f}×ATR={atr/pip:.0f}p) — micro-stop, noise will hit it"
                )
                log_gap(ts_utc, pair, "ENTER", "BLOCKED", gap_type,
                        f"{msg}  entry={decision.entry_price:.5f}  sl={decision.stop_loss:.5f}")
                print(f"  ⚠ {ts_utc.strftime('%Y-%m-%d %H:%M')} | SKIP {pair} — {msg}  sl={decision.stop_loss:.5f}")
                continue

            if not decision.entry_price or not decision.stop_loss:
                continue

            # Valid candidate — queue for pip-equity-ranked entry
            entry_candidates.append((_pip_equity(decision, pair), pair, decision, _is_theme_pair))

        # ── Phase 2: sort by pip equity, enter highest-potential first ─
        # When slots are scarce, this ensures the best setup wins, not
        # whichever pair happened to come first in the dict iteration.
        entry_candidates.sort(key=lambda x: x[0], reverse=True)

        for pe, pair, decision, _is_theme_pair in entry_candidates:
            if pair in open_pos:
                continue  # entered by an earlier candidate this bar

            # Minimum pip equity gate — blocks low-potential setups from
            # consuming slots that a higher-equity trade might need later.
            # Macro theme trades are exempt (their size is already fractional).
            # Consolidation_breakout is also exempt: the 46p range looks small
            # but Alex captures 3-5× by trailing. Its own quality filters
            # (round number + 10-bar consol + live break + 4H engulfing) are
            # sufficient quality gating without a pip equity floor.
            _is_cb_pattern = 'consolidation_breakout' in (
                decision.pattern.pattern_type if decision.pattern else '')
            # Dynamic pip equity gate (always-on, replaces fixed 100p floor).
            # Threshold = stop_pips × MIN_RR_EFFECTIVE:
            #   pro-trend (W+D+4H agree)   → stop_pips × MIN_RR_STANDARD   (2.5R)
            #   non-protrend / mixed / ?   → stop_pips × MIN_RR_COUNTERTREND (3.0R)
            # Ensures reward potential in pips ≥ required R multiple of actual risk.
            # Falls back to 10p absolute minimum if stop_pips unavailable.
            if not _is_theme_pair and not _is_cb_pattern:
                _stop_pips_pe = getattr(decision, "initial_stop_pips", 0.0) or 0.0
                if _stop_pips_pe > 0:
                    _htf_pe = alex_policy.htf_aligned(
                        decision.direction or "",
                        decision.trend_weekly, decision.trend_daily, decision.trend_4h,
                    )
                    _rr_pe    = (_sc.MIN_RR_STANDARD
                                 if _htf_pe is True else _sc.MIN_RR_COUNTERTREND)
                    _pe_min   = _stop_pips_pe * _rr_pe
                else:
                    _pe_min   = _sc.MIN_PIP_EQUITY   # fallback: no stop data (rare)
                if pe < _pe_min:
                    _dyn_pip_eq_blocks += 1
                    log_gap(ts_utc, pair, "ENTER", "BLOCKED", "DYN_PIP_EQUITY",
                            f"Pip equity {pe:.0f}p < stop({_stop_pips_pe:.0f}p)"
                            f" × {_rr_pe:.1f}R = {_pe_min:.0f}p min")
                    continue

            # Re-check eligibility — earlier entries this bar may have changed
            # slot counts; also runs theme contradiction gate (needs direction)
            eligible, elig_reason2 = _entry_eligible(pair, active_theme,
                                                      direction=decision.direction)
            if not eligible:
                if "theme_contradiction" in elig_reason2:
                    log_gap(ts_utc, pair, decision.direction.upper(), "BLOCKED",
                            "theme_contradiction",
                            f"Open-position theme contradicts {decision.direction} {pair}: "
                            f"{elig_reason2}")
                continue

            # ── Risk mode: compute dynamically at each entry ─────────────
            # Mirrors live orchestrator: ALL-4 conditions required for HIGH,
            # plus 2-bar hysteresis.  force_risk_mode pins mode (comparison runs).
            _dd_pct_entry = (
                (peak_balance - balance) / peak_balance * 100
                if peak_balance > 0 else 0.0
            )
            try:
                _rms_entry = compute_risk_mode(
                    trend_weekly            = (decision.trend_weekly.value
                                               if hasattr(decision.trend_weekly, "value")
                                               else str(decision.trend_weekly or "")),
                    trend_daily             = (decision.trend_daily.value
                                               if hasattr(decision.trend_daily, "value")
                                               else str(decision.trend_daily or "")),
                    df_h4                   = hist_4h,
                    recent_trades           = trades,      # closed trades so far
                    loss_streak             = consecutive_losses,
                    dd_pct                  = _dd_pct_entry,
                    # Instantaneous evaluation: Alex's ~1 trade/week cadence is too
                    # sparse for 2-bar consecutive accumulation across entries.
                    # HIGH/EXTREME granted immediately when ALL conditions are met at
                    # this exact bar — no state carried between entries.
                    # 2-bar hysteresis runs only in the H4 time-sampling loop.
                    instantaneous           = True,
                    streak_demotion_thresh  = streak_demotion_thresh,
                )
                # Debug: log every HIGH/EXTREME promotion with full inputs.
                if _rms_entry.promotion_note:
                    print(f"  🔺 {ts_utc.strftime('%Y-%m-%d')} {pair} → {_rms_entry.mode.value}"
                          f" | {_rms_entry.promotion_note}")
                # force_risk_mode pins mode for every entry (used in comparison runs).
                risk.set_regime_mode(force_risk_mode or _rms_entry.mode.value)
            except Exception:
                _rms_entry = None
                risk.set_regime_mode(force_risk_mode or None)

            _base_rpct, _dd_flag = _risk_pct_with_flag(balance)

            # ── DD kill-switch: hard block at ≥ 40% DD ────────────────
            if _dd_flag == "DD_KILLSWITCH":
                dd_killswitch_blocks += 1
                _dd_pct = (peak_balance - balance) / peak_balance * 100 if peak_balance else 0
                log_gap(ts_utc, pair, decision.direction.upper(), "BLOCKED",
                        "dd_killswitch",
                        f"DD {_dd_pct:.1f}% ≥ {_sc.DD_KILLSWITCH_PCT:.0f}% kill-switch — "
                        f"no new entries until equity recovers above "
                        f"{_sc.DD_RESUME_PCT*100:.0f}% of peak (${peak_balance:,.0f})")
                continue

            rpct = _base_rpct
            if _is_theme_pair and active_theme:
                rpct = rpct * active_theme.position_fraction
            units = _calc_units(pair, balance, rpct,
                                decision.entry_price, decision.stop_loss)
            if units <= 0:
                continue

            # ── Alex small-account gates (via shared alex_policy module) ──
            # Parity note: live orchestrator calls the same functions from
            # src/strategy/forex/alex_policy.py — single source of truth.

            # ── Chop Shield (Arm D) ──────────────────────────────────────────
            # Part A: 48h auto-pause when streak first hits THRESH.
            # Part B: recovery-selectivity after pause expires.
            # Identical policy to live orchestrator; backtester uses wall-clock
            # timestamps anchored to the candle bar time (ts_utc) for determinism.
            if chop_shield:
                _cs_thresh = getattr(_sc, "CHOP_SHIELD_STREAK_THRESH", 3)
                _cs_hours  = getattr(_sc, "CHOP_SHIELD_PAUSE_HOURS", 48.0)
                # Arm the shield on first hit of streak threshold
                if consecutive_losses >= _cs_thresh and _bt_chop_pause_until is None:
                    from datetime import timedelta as _td
                    _bt_chop_pause_until = ts_utc + _td(hours=_cs_hours)
                    _bt_chop_auto_pauses += 1
                    log_gap(ts_utc, pair, "ENTER", "BLOCKED", "AUTO_PAUSE_STREAK3",
                            f"Chop Shield: streak={consecutive_losses} ≥ {_cs_thresh}"
                            f" — paused until {_bt_chop_pause_until.strftime('%Y-%m-%d %H:%M')} UTC")
                # Part A: block during active pause window
                if _bt_chop_pause_until is not None and ts_utc < _bt_chop_pause_until:
                    _bt_chop_paused_blocks += 1
                    log_gap(ts_utc, pair, "ENTER", "BLOCKED", "CHOP_PAUSED",
                            f"Auto-pause active until {_bt_chop_pause_until.strftime('%Y-%m-%d %H:%M')} UTC"
                            f" (streak={consecutive_losses})")
                    continue
                # Part B: recovery gates after pause expires and streak still ≥ thresh
                if (_bt_chop_pause_until is not None
                        and ts_utc >= _bt_chop_pause_until
                        and consecutive_losses >= _cs_thresh):
                    _rec_rr    = getattr(_sc, "RECOVERY_MIN_RR", 3.0)
                    _rec_boost = getattr(_sc, "RECOVERY_CONF_BOOST", 0.05)
                    _rec_wcap  = getattr(_sc, "RECOVERY_WEEKLY_CAP", 1)
                    _rec_conf  = getattr(_sc, "MIN_CONFIDENCE", 0.60) + _rec_boost
                    _rec_iso   = ts_utc.isocalendar()[:2]
                    _rec_wk    = _weekly_trade_counts.get(_rec_iso, 0)
                    if decision.exec_rr < _rec_rr:
                        _bt_chop_recovery_blocks += 1
                        log_gap(ts_utc, pair, "ENTER", "BLOCKED", "RECOVERY_MIN_RR",
                                f"Recovery: exec_rr={decision.exec_rr:.2f}R < {_rec_rr:.1f}R"
                                f" (streak={consecutive_losses})")
                        continue
                    if decision.confidence < _rec_conf:
                        _bt_chop_recovery_blocks += 1
                        log_gap(ts_utc, pair, "ENTER", "BLOCKED", "RECOVERY_CONF",
                                f"Recovery: conf={decision.confidence:.0%} < {_rec_conf:.0%}"
                                f" (streak={consecutive_losses})")
                        continue
                    if _rec_wk >= _rec_wcap:
                        _bt_chop_recovery_blocks += 1
                        log_gap(ts_utc, pair, "ENTER", "BLOCKED", "RECOVERY_WEEKLY_CAP",
                                f"Recovery: weekly trades={_rec_wk} ≥ cap={_rec_wcap}"
                                f" (streak={consecutive_losses})")
                        continue

            # Gate 1: Alignment-based MIN_RR
            # Pro-trend (W+D+4H all agree) → 2.5R; non-protrend/mixed → 3.0R
            _htf_aligned_flag = alex_policy.htf_aligned(
                decision.direction or "",
                decision.trend_weekly,
                decision.trend_daily,
                decision.trend_4h,
            )
            _rr_blocked, _rr_reason = alex_policy.check_dynamic_min_rr(
                decision.exec_rr,
                htf_aligned_flag=_htf_aligned_flag,
                balance=balance,
            )
            if _rr_blocked:
                _min_rr_small_blocks += 1
                log_gap(ts_utc, pair, "ENTER", "BLOCKED", "MIN_RR_ALIGN",
                        _rr_reason)
                continue

            # Gate 2: Weekly punch-card limit (ISO week)
            _iso_key = ts_utc.isocalendar()[:2]   # (year, week)
            _wk_blocked, _wk_reason = alex_policy.check_weekly_trade_limit(
                _weekly_trade_counts.get(_iso_key, 0), balance
            )
            if _wk_blocked:
                _weekly_limit_blocks += 1
                log_gap(ts_utc, pair, "ENTER", "BLOCKED", "WEEKLY_TRADE_LIMIT",
                        _wk_reason + f" (ISO {_iso_key[0]}-W{_iso_key[1]:02d})")
                continue

            # ── Gate 3 (adaptive): regime-gated Thu/Fri block + weekly cap ──
            # When adaptive_gates=True the session filter does NOT block Thu PM / Fri
            # at strategy eval time (NO_THU_FRI_TRADES_ENABLED=False).  Instead we
            # compute regime_score here and apply the time/weekly gate conditionally:
            #   • Thu PM or Fri AND regime_score < adaptive_threshold  →  block
            #   • Weekly cap = 2 if regime_score ≥ 0.65, else 1 (regardless of balance)
            # This skips poor-regime entries at the end of the week, but allows
            # high-conviction setups through even on Thursday/Friday.
            _rs_early = None   # will hold early RegimeScore when adaptive_gates=True
            if adaptive_gates:
                _h4_sl = {
                    _p: candle_data[_p]["4h"][candle_data[_p]["4h"].index < ts]
                    for _p in candle_data
                    if candle_data[_p].get("4h") is not None
                    and len(candle_data[_p]["4h"][candle_data[_p]["4h"].index < ts]) >= 20
                }
                _rs_early = compute_regime_score(
                    df_h4=hist_4h, recent_trades=trades, h4_slices=_h4_sl,
                )
                _et_bar = ts_utc.astimezone(_ET_TZ)
                _is_thu_pm = _et_bar.weekday() == 3 and _et_bar.hour > 9
                _is_fri    = _et_bar.weekday() == 4
                if (_is_thu_pm or _is_fri) and _rs_early.total < adaptive_threshold:
                    _adaptive_time_blocks += 1
                    log_gap(ts_utc, pair, "ENTER", "BLOCKED", "ADAPTIVE_TIME_GATE",
                            f"Thu PM/Fri AND regime_score={_rs_early.total:.2f} < "
                            f"{adaptive_threshold:.2f} — below quality threshold")
                    continue
                # Adaptive weekly cap: relax to 2/wk when regime is strong
                _adaptive_wk_cap = 2 if _rs_early.total >= 0.65 else 1
                _iso_key_adp = ts_utc.isocalendar()[:2]
                if _weekly_trade_counts.get(_iso_key_adp, 0) >= _adaptive_wk_cap:
                    _weekly_limit_blocks += 1
                    log_gap(ts_utc, pair, "ENTER", "BLOCKED", "ADAPTIVE_WEEKLY_CAP",
                            f"weekly cap={_adaptive_wk_cap} (regime={_rs_early.total:.2f}) "
                            f"already used ISO {_iso_key_adp[0]}-W{_iso_key_adp[1]:02d}")
                    continue

            # ── Gate 4a (strict_protrend): ALL three HTFs must agree with direction ──
            # Requires W+D+4H all agree.  Usually overconstrained at entry since 4H
            # is often mid-retracement — prefer Gate 4b (wd_protrend_htf) instead.
            if strict_protrend_htf:
                _sp_htf = alex_policy.htf_aligned(
                    decision.direction or "",
                    decision.trend_weekly,
                    decision.trend_daily,
                    decision.trend_4h,
                )
                if _sp_htf is not True:
                    _strict_htf_blocks += 1
                    _sp_reason = (
                        "HTF unknown (missing trend data)"
                        if _sp_htf is None
                        else f"W/D/4H not all {decision.direction} — strict protrend required"
                    )
                    log_gap(ts_utc, pair, "ENTER", "BLOCKED", "STRICT_PROTREND_HTF",
                            f"{_sp_reason} (direction={decision.direction})")
                    continue

            # ── Gate 4b (wd_protrend_htf): Weekly AND Daily must agree, 4H exempt ──
            # Alex enters during 4H retracements; W+D bias sets the macro direction.
            # The 1H engulfing at a key level IS the 4H flip/confirmation — requiring
            # 4H to already agree means you miss the entry entirely.
            # Gate: block if W or D opposes direction; allow 4H mixed/neutral/counter.
            if wd_protrend_htf:
                _wd_flag = alex_policy.htf_aligned_wd(
                    decision.direction or "",
                    decision.trend_weekly,
                    decision.trend_daily,
                )
                if _wd_flag is not True:
                    _wd_htf_blocks += 1
                    _wd_reason = (
                        "W/D trend data missing"
                        if _wd_flag is None
                        else f"W/D not both {decision.direction} (4H exempt)"
                    )
                    log_gap(ts_utc, pair, "ENTER", "BLOCKED", "WD_PROTREND_HTF",
                            f"{_wd_reason} (direction={decision.direction})")
                    continue

            # ── W==D alignment tracker (always, for all entered trades) ──────────
            # Tracks % of entered trades that had W+D protrend — useful diagnostic
            # independent of whether any gate is active.
            _wd_check = alex_policy.htf_aligned_wd(
                decision.direction or "",
                decision.trend_weekly,
                decision.trend_daily,
            )
            if _wd_check is True:
                _wd_aligned_entries += 1

            strat = strategies[pair]

            # ── Target: use exec_target from strategy (single source of truth) ──
            # The strategy's select_target() already chose the best qualifying
            # target (4H structure → measured_move → T2) and gated on MIN_RR.
            # We use decision.exec_target verbatim — no override, no divergence.
            # If exec_target is missing (legacy path), this is a hard bug, not a
            # graceful fallback — log it and skip so we don't enter at 0.3R.
            _target = decision.exec_target
            if _target is None:
                log_gap(ts_utc, pair, "ENTER", "BLOCKED", "no_exec_target",
                        f"decision.exec_target is None — strategy should have blocked this. "
                        f"Skipping to avoid uncontrolled RR. pattern={decision.pattern.pattern_type if decision.pattern else '?'}")
                continue

            # ── Regime score at entry ─────────────────────────────────────
            # Compute once at entry (cheap: ~17 times in Alex window).
            # All 4 components use data already loaded; no new OANDA calls.
            # If adaptive_gates=True we already computed _rs_early above — reuse it.
            if _rs_early is not None:
                _rs = _rs_early   # reuse from adaptive gate (avoid double compute)
            else:
                _h4_slices_rs = {
                    _p: candle_data[_p]["4h"][candle_data[_p]["4h"].index < ts]
                    for _p in candle_data
                    if candle_data[_p].get("4h") is not None
                    and len(candle_data[_p]["4h"][candle_data[_p]["4h"].index < ts]) >= 20
                }
                _rs = compute_regime_score(
                    df_h4        = hist_4h,
                    recent_trades = trades,      # closed trades so far
                    h4_slices    = _h4_slices_rs,
                )

            open_pos[pair] = {
                "entry_price":  decision.entry_price,
                "stop_loss":    decision.stop_loss,
                "direction":    decision.direction,
                "units":        units,
                "bar_idx":      bar_idx,
                "entry_ts":     ts_utc,
                "pattern":      decision.pattern.pattern_type if decision.pattern else "?",
                "notes":        decision.reason[:80],
                "be_moved":     False,
                "trail_max":    decision.entry_price,   # running max-favourable price for ratchet
                "trail_locked": False,                  # True after stage-1 lock fires (Arm C)
                "streak_at_entry": consecutive_losses,  # for per-trade risk audit
                "macro_theme":  f"{active_theme.currency}_{active_theme.direction}" if _is_theme_pair and active_theme else None,
                "regime_score": _rs.total,              # RegimeScore at entry
                "risk_mode_at_entry": (force_risk_mode or (_rms_entry.mode.value if _rms_entry else "MEDIUM")),
                "target_price": _target,
                "target_type":  decision.exec_target_type,
                "stop_type":    decision.stop_type,
                "initial_risk": abs(decision.entry_price - decision.stop_loss),
                "initial_stop_pips": decision.initial_stop_pips,
                "base_risk_pct":     _base_rpct,
                "final_risk_pct":    rpct,
                "entry_equity":      balance,
                "entry_risk_dollars": balance * rpct / 100,
                "dd_flag":           _dd_flag,
                "signal_type":   (decision.entry_signal.signal_type
                                  if decision.entry_signal else "unknown"),
                "planned_rr":    decision.exec_rr,
                "mfe_r":  0.0,
                "mae_r":  0.0,
            }
            # Record this week's trade (ISO week key)
            _weekly_trade_counts[_iso_key] = _weekly_trade_counts.get(_iso_key, 0) + 1

            strat.register_open_position(
                pair=pair,
                entry_price=decision.entry_price,
                stop_loss=decision.stop_loss,
                direction=decision.direction,
                pattern_type=decision.pattern.pattern_type if decision.pattern else None,
                neckline_ref=decision.neckline_ref,
                risk_pct=rpct,
            )

            theme_tag = (
                f"  🎯 MACRO THEME: {active_theme.currency} {active_theme.direction.upper()}"
                f" (score={active_theme.score:.1f}, {active_theme.trade_count} stacked,"
                f" size={active_theme.position_fraction:.0%} of normal)"
                if _is_theme_pair and active_theme else ""
            )
            print(f"  📈 {ts_utc.strftime('%Y-%m-%d %H:%M')} "
                  f"| ENTER {pair} {decision.direction.upper()}"
                  f" @ {decision.entry_price:.5f}  SL={decision.stop_loss:.5f}"
                  f"  conf={decision.confidence:.0%}  [{decision.reason[:55]}]"
                  f"  📊{pe:.0f}p{theme_tag}")
            # ── Risk mode audit print (every trade — proves multiplier wired) ──
            _pos_now = open_pos[pair]
            _mode_label = _pos_now.get("risk_mode_at_entry", "MEDIUM")
            _base_pct_now = _pos_now.get("base_risk_pct", 0)
            _eff_pct_now  = _pos_now.get("final_risk_pct", 0)
            _risk_usd_now = _pos_now.get("entry_risk_dollars", 0)
            _mode_mult_now = (_eff_pct_now / _base_pct_now) if _base_pct_now else 1.0
            _mode_icon = {"LOW": "🔵", "HIGH": "🟠", "EXTREME": "🔴"}.get(_mode_label, "⚪")
            print(f"       {_mode_icon} mode={_mode_label:<8}"
                  f"  base={_base_pct_now:.2f}%  eff={_eff_pct_now:.2f}%"
                  f"  mult≈{_mode_mult_now:.2f}×"
                  f"  risk=${_risk_usd_now:,.0f}"
                  f"  streak={consecutive_losses}"
                  f"  dd={_dd_pct_entry:.1f}%")

    # ── Last-resort close (runout period exhausted) ───────────────────
    # Normally positions close via target or stop during the runout window.
    # This block only fires if RUNOUT_DAYS wasn't long enough — rare edge case.
    last_ts = all_1h[-1] if all_1h else None
    if open_pos:
        print(f"\n  ⚠ {len(open_pos)} position(s) still open after {RUNOUT_DAYS}-day runout "
              f"— closing at last bar price (edge case).")
    for pair, pos in list(open_pos.items()):
        if pair not in candle_data: continue
        df_1h_p = candle_data[pair]["1h"]
        close_p = float(df_1h_p["close"].iloc[-1])
        entry   = pos["entry_price"]
        stop    = pos["stop_loss"]
        direction = pos["direction"]
        units   = pos["units"]
        delta       = (close_p - entry) if direction == "long" else (entry - close_p)
        spread_cost = _spread_deduction(pair, units)
        pnl         = delta * units - spread_cost
        r = (pnl / pos.get("entry_risk_dollars", 1)
             if pos.get("entry_risk_dollars") else
             delta / abs(entry - stop) if abs(entry - stop) else 0)
        balance    += pnl
        peak_balance = max(peak_balance, balance)  # DD circuit breaker HWM
        trades.append({
            "pair": pair, "direction": direction,
            "entry": entry, "exit": close_p,
            "pnl": pnl, "r": r, "reason": "runout_expired",
            "spread_cost": round(spread_cost, 2),
            "signal_type":  pos.get("signal_type", "unknown"),
            "planned_rr":   pos.get("planned_rr", 0.0),
            "stop_type":    pos.get("stop_type", "unknown"),
            "initial_stop_pips": pos.get("initial_stop_pips", 0),
            "mfe_r":        pos.get("mfe_r", 0.0),
            "mae_r":        pos.get("mae_r", 0.0),
            "dd_flag":      pos.get("dd_flag", ""),
            "entry_ts": pos["entry_ts"].isoformat(),
            "exit_ts":  last_ts.isoformat() if last_ts else "",
            "bars_held": len(all_1h) - pos["bar_idx"],
            "pattern": pos.get("pattern", "?"),
            "notes": pos.get("notes", ""),
            "macro_theme": pos.get("macro_theme"),
            "target_1": pos.get("target_price"),
            "regime_score_at_entry": pos.get("regime_score", 0.0),
                        "risk_mode_at_entry": pos.get("risk_mode_at_entry", "MEDIUM"),
        })

    # ── Results ───────────────────────────────────────────────────────
    net_pnl  = balance - starting_bal
    ret_pct  = net_pnl / starting_bal * 100
    # ── 3-bucket classification: Win / Loss / Scratch ─────────────────────
    # Scratch = |R| < SCRATCH_THRESHOLD (BE exits, tiny weekend partials).
    # Win/Loss buckets exclude scratches for conditional WR reporting.
    SCRATCH_R = 0.10   # ≤0.10R in either direction = scratch
    wins      = [t for t in trades if t["r"] >  SCRATCH_R]
    losses    = [t for t in trades if t["r"] < -SCRATCH_R]
    scratches = [t for t in trades if abs(t["r"]) <= SCRATCH_R]
    directional = wins + losses      # excludes scratches
    wr       = len(wins) / len(trades)       * 100 if trades     else 0
    cond_wr  = len(wins) / len(directional)  * 100 if directional else 0
    avg_r    = (sum(t["r"] for t in trades)   / len(trades))      if trades else 0.0
    avg_r_w  = (sum(t["r"] for t in wins)     / len(wins))        if wins   else 0.0
    avg_r_l  = (sum(t["r"] for t in losses)   / len(losses))      if losses else 0.0
    avg_r_s  = (sum(t["r"] for t in scratches)/ len(scratches))   if scratches else 0.0
    max_r    = max((t["r"] for t in trades), default=0.0)
    # Max drawdown: track running equity peak and worst trough
    _eq = starting_bal
    _pk = starting_bal
    _max_dd_pct = 0.0
    for t in trades:
        _eq += t["pnl"]
        _pk  = max(_pk, _eq)
        _dd  = (_pk - _eq) / _pk * 100 if _pk > 0 else 0
        _max_dd_pct = max(_max_dd_pct, _dd)
    # Trigger type counts
    from collections import Counter
    _trigger_counts = Counter(
        t.get("signal_type", "unknown") for t in trades
    )
    # DD cap usage
    _dd_cap_trades = [t for t in trades if t.get("dd_flag")]

    # ── Time-in-mode: H4 bar sampling (computed once, used in print + result) ─
    # Samples compute_risk_mode() at every H4 bar in the window.
    # Gives % of CALENDAR TIME in each mode — independent of entry frequency.
    #
    # Real W/D bias: uses actual daily ("d") and weekly ("w") candle series from
    # the representative pair.  Daily trend cached per calendar date; weekly trend
    # cached per ISO week.  This eliminates the "H4 proxy = wd_aligned always True"
    # artifact that inflated HIGH/EXTREME time in previous runs.
    _tim_counts = Counter({"LOW": 0, "MEDIUM": 0, "HIGH": 0, "EXTREME": 0})
    try:
        from src.strategy.forex.pattern_detector import PatternDetector as _PD
        _pd_inst  = _PD()
        _rep_pair = next(iter(candle_data.keys()), None)
        _rep_h4   = candle_data[_rep_pair].get("4h") if _rep_pair else None
        _rep_d    = candle_data[_rep_pair].get("d")  if _rep_pair else None
        _rep_w    = candle_data[_rep_pair].get("w")  if _rep_pair else None
        if _rep_h4 is not None and len(_rep_h4) > 0:
            _start_ts = pd.Timestamp(start_dt).replace(tzinfo=None)
            _end_ts   = (pd.Timestamp(end_dt).replace(tzinfo=None)
                         if end_dt else _rep_h4.index[-1])
            _h4_window = _rep_h4[(_rep_h4.index >= _start_ts) &
                                  (_rep_h4.index <  _end_ts)]

            # ── Bias caches: compute once per day/week, not once per H4 bar ─
            _daily_bias_cache:  dict = {}   # date_str  → trend str
            _weekly_bias_cache: dict = {}   # week_str  → trend str

            def _get_daily_bias(ts: "pd.Timestamp") -> str:
                """Daily trend from D candles sliced to < ts; cached per date."""
                key = str(ts.date())
                if key not in _daily_bias_cache:
                    if _rep_d is not None:
                        _d_sl = _rep_d[_rep_d.index < ts]
                        if len(_d_sl) >= 21:
                            try:
                                _daily_bias_cache[key] = _pd_inst.detect_trend(_d_sl).value
                                return _daily_bias_cache[key]
                            except Exception:
                                pass
                    _daily_bias_cache[key] = "neutral"
                return _daily_bias_cache[key]

            def _get_weekly_bias(ts: "pd.Timestamp") -> str:
                """Weekly trend from W candles sliced to < ts; cached per ISO week."""
                iso = ts.isocalendar()
                key = f"{iso[0]}-W{iso[1]:02d}"
                if key not in _weekly_bias_cache:
                    if _rep_w is not None:
                        _w_sl = _rep_w[_rep_w.index < ts]
                        if len(_w_sl) >= 21:
                            try:
                                _weekly_bias_cache[key] = _pd_inst.detect_trend(_w_sl).value
                                return _weekly_bias_cache[key]
                            except Exception:
                                pass
                    _weekly_bias_cache[key] = "neutral"
                return _weekly_bias_cache[key]

            _tim_consec  = 0   # consecutive qualifying H4 bars
            _tim_demote  = 0   # consecutive failing bars while in HIGH zone
            for _hi, (_h4_ts, _) in enumerate(_h4_window.iterrows()):
                _h4_slice = _h4_window.iloc[max(0, _hi - 80): _hi + 1]
                if len(_h4_slice) < 21:
                    continue

                # Real W/D trends (accurate wd_aligned — no H4 proxy)
                _trend_w_str = _get_weekly_bias(_h4_ts)
                _trend_d_str = _get_daily_bias(_h4_ts)

                _h4_ts_str = str(_h4_ts)
                _recent_closed = [t for t in trades
                                  if (t.get("exit_ts") or "0") < _h4_ts_str]
                _recent5 = _recent_closed[-5:]

                # Loss streak at this H4 bar from most recent closed trades
                _tim_streak = 0
                for _rt in reversed(_recent_closed):
                    if _rt.get("r", 0.0) < 0:
                        _tim_streak += 1
                    else:
                        break
                try:
                    _rms_h4 = compute_risk_mode(
                        trend_weekly            = _trend_w_str,
                        trend_daily             = _trend_d_str,
                        df_h4                   = _h4_slice,
                        recent_trades           = _recent5,
                        loss_streak             = _tim_streak,
                        dd_pct                  = 0.0,   # no per-bar balance in sampling
                        consecutive_high_bars   = _tim_consec,
                        demotion_streak         = _tim_demote,
                        streak_demotion_thresh  = streak_demotion_thresh,
                    )
                    _tim_consec = _rms_h4.consecutive_high_bars
                    _tim_demote = _rms_h4.demotion_streak
                    _tim_counts[_rms_h4.mode.value] += 1
                except Exception:
                    _tim_consec = 0
                    _tim_demote = 0
                    _tim_counts["MEDIUM"] += 1
    except Exception:
        pass   # time-in-mode is diagnostic — never crash the backtest

    _tim_total       = sum(_tim_counts.values()) or 1
    _tim_pct_low     = _tim_counts["LOW"]     / _tim_total * 100
    _tim_pct_medium  = _tim_counts["MEDIUM"]  / _tim_total * 100
    _tim_pct_high    = _tim_counts["HIGH"]    / _tim_total * 100
    _tim_pct_extreme = _tim_counts["EXTREME"] / _tim_total * 100

    print(f"\n{'='*65}")
    print(f"RESULTS — v2 (Real Strategy Code)")
    print(f"{'='*65}")
    print(f"  Trades:       {len(trades)}  "
          f"({len(wins)}W / {len(losses)}L / {len(scratches)}S)")
    print(f"  Win rate:     {wr:.0f}%  (all)    "
          f"Conditional (excl scratch): {cond_wr:.0f}%")
    _scratch_pct = len(scratches) / len(trades) * 100 if trades else 0.0
    print(f"  Scratch rate: {_scratch_pct:.0f}%  "
          f"({len(scratches)} of {len(trades)} trades)")
    print(f"  Avg R:        {avg_r:>+.2f}R  "
          f"(W: {avg_r_w:>+.1f}R  L: {avg_r_l:>+.1f}R  S: {avg_r_s:>+.1f}R)")
    print(f"  Best R:       {max_r:>+.1f}R")
    print(f"  Max DD:       {_max_dd_pct:.1f}%")
    print(f"  Starting:     ${starting_bal:>10,.2f}")
    print(f"  Net P&L:      ${net_pnl:>+10,.2f}")
    print(f"  Final:        ${balance:>10,.2f}")
    print(f"  Return:       {ret_pct:>+.1f}%")

    # ── Per-trade postmortem ──────────────────────────────────────────
    _low_rr    = [t for t in trades if t.get("planned_rr", 0) < 2.8 and t.get("planned_rr", 0) > 0]
    _zero_rr   = [t for t in trades if t.get("planned_rr", 0) == 0]
    _exit_counts = Counter(t["reason"] for t in trades)
    _avg_mfe   = sum(t.get("mfe_r", 0) for t in trades) / len(trades) if trades else 0
    _avg_mae   = sum(t.get("mae_r", 0) for t in trades) / len(trades) if trades else 0
    _avg_plan_rr = sum(t.get("planned_rr", 0) for t in trades if t.get("planned_rr", 0) > 0)
    _plan_rr_n   = sum(1 for t in trades if t.get("planned_rr", 0) > 0)
    _avg_plan_rr = _avg_plan_rr / _plan_rr_n if _plan_rr_n else 0

    # exec_rr distribution
    _rr_vals = sorted(t.get("planned_rr", 0) for t in trades if t.get("planned_rr", 0) > 0)
    _rr_p50  = _rr_vals[len(_rr_vals)//2] if _rr_vals else 0
    _rr_p95  = _rr_vals[int(len(_rr_vals)*0.95)] if _rr_vals else 0

    print(f"\n  ── Per-trade postmortem ───────────────────────────────────")
    if trades:
        print(f"    Exec RR avg:          {_avg_plan_rr:.2f}R  (n={_plan_rr_n})")
        print(f"    Exec RR p50/p95:      {_rr_p50:.2f}R / {_rr_p95:.2f}R")
        print(f"    Exec RR < 2.8:        {len(_low_rr)} trades ({len(_low_rr)/len(trades)*100:.0f}%)  ← should be ~0 after fix")
        print(f"    Exec RR = 0 (N/A):    {len(_zero_rr)} trades  ← no qualifying target at entry")
        print(f"    Avg MFE:              {_avg_mfe:+.2f}R")
        print(f"    Avg MAE:              {_avg_mae:+.2f}R")
    else:
        print("    (no trades — all entries blocked by active gates)")
    print(f"\n  ── Exit reason counts ────────────────────────────────────")
    _reason_order = ["target_reached", "weekend_proximity",
                     "ratchet_stop_hit", "stop_hit", "stall_exit", "max_hold", "runout_expired"]
    _all_reasons  = set(_exit_counts.keys())
    _sorted_reasons = ([r for r in _reason_order if r in _all_reasons] +
                       sorted(_all_reasons - set(_reason_order)))
    for reason in _sorted_reasons:
        cnt = _exit_counts[reason]
        avg_r = (sum(t["r"] for t in trades if t["reason"] == reason) / cnt
                 if cnt else 0.0)
        _epct = cnt / len(trades) * 100 if trades else 0.0
        print(f"    {reason:<30} {cnt:>3}  ({_epct:.0f}%)  "
              f"avg {avg_r:+.2f}R")

    if _trigger_counts:
        print(f"\n  ── Trigger types ──────────────────────────────────────")
        for sig_type, count in sorted(_trigger_counts.items(), key=lambda x: -x[1]):
            pct = count / len(trades) * 100 if trades else 0
            print(f"    {sig_type:<30} {count:>3}  ({pct:.0f}%)")

    # Stop type distribution — tracks structural vs atr_fallback usage
    _stop_types    = Counter(pos.get("stop_type", "unknown") for pos in trades)
    _st_total      = sum(_stop_types.values())
    _atr_fb_count  = _stop_types.get("atr_fallback", 0)
    _atr_fb_pct    = _atr_fb_count / _st_total * 100 if _st_total else 0.0
    _sp_vals       = sorted(t.get("initial_stop_pips", 0) for t in trades
                            if t.get("initial_stop_pips", 0) > 0)
    _sp_p50        = _sp_vals[len(_sp_vals) // 2] if _sp_vals else 0.0
    _sp_p90        = _sp_vals[int(len(_sp_vals) * 0.9)] if _sp_vals else 0.0

    _FLAG_ATR = "  ⚠ >40% atr_fallback — structural stop regression!" if _atr_fb_pct > 40 else ""
    print(f"\n  ── Stop type distribution{_FLAG_ATR} ────────────────────────")
    if _stop_types:
        for stype, cnt in _stop_types.most_common():
            pct = cnt / len(trades) * 100 if trades else 0
            _sp_by_type = sorted(t.get("initial_stop_pips", 0) for t in trades
                                 if t.get("stop_type") == stype and t.get("initial_stop_pips", 0) > 0)
            _p50_t = _sp_by_type[len(_sp_by_type)//2] if _sp_by_type else 0
            print(f"    {stype:<35} {cnt:>3}  ({pct:.0f}%)  p50={_p50_t:.0f}p")
        if _sp_vals:
            print(f"    Overall stop pips:  p50={_sp_p50:.0f}p  p90={_sp_p90:.0f}p  "
                  f"min={_sp_vals[0]:.0f}p  max={_sp_vals[-1]:.0f}p")
        print(f"    atr_fallback rate: {_atr_fb_pct:.0f}%{_FLAG_ATR}")
    else:
        print("    (no trades)")

    # ── Regime score distribution ─────────────────────────────────────────
    _rs_vals = [t.get("regime_score_at_entry", 0.0) for t in trades]
    if _rs_vals:
        _rs_by_band = {"LOW(<2)": 0, "MED(2-3)": 0, "HIGH(3-3.5)": 0, "EXTREME(3.5+)": 0}
        for _rsv in _rs_vals:
            if _rsv >= 3.5:   _rs_by_band["EXTREME(3.5+)"] += 1
            elif _rsv >= 3.0: _rs_by_band["HIGH(3-3.5)"] += 1
            elif _rsv >= 2.0: _rs_by_band["MED(2-3)"] += 1
            else:              _rs_by_band["LOW(<2)"] += 1
        _rs_high_pct  = sum(1 for v in _rs_vals if v >= 3.0) / len(_rs_vals) * 100
        _rs_ext_pct   = sum(1 for v in _rs_vals if v >= 3.5) / len(_rs_vals) * 100
        _rs_avg       = sum(_rs_vals) / len(_rs_vals)
        print(f"\n  ── Regime score @ entry ──────────────────────────────────")
        print(f"    Avg score:            {_rs_avg:.2f}")
        print(f"    % score ≥ 3.0 (HIGH): {_rs_high_pct:.0f}%")
        print(f"    % score ≥ 3.5 (EXTR): {_rs_ext_pct:.0f}%")
        _band_bounds = {"LOW(<2)": (0,2), "MED(2-3)": (2,3), "HIGH(3-3.5)": (3,3.5), "EXTREME(3.5+)": (3.5,99)}
        for _band, _cnt in _rs_by_band.items():
            if _cnt == 0: continue
            _lo, _hi = _band_bounds[_band]
            _brs    = [t["r"] for t in trades if _lo <= t.get("regime_score_at_entry", 0) < _hi]
            _avg_r  = sum(_brs) / len(_brs) if _brs else 0
            print(f"    {_band:<18}  {_cnt:>2} trades  avg R={_avg_r:+.2f}")

    # ── Risk mode distribution ────────────────────────────────────────────
    _rm_at_entry = [t.get("risk_mode_at_entry", "MEDIUM") for t in trades]
    if _rm_at_entry:
        # Build time-in-mode from local Counter for display
        _local_tim = Counter({"LOW": 0, "MEDIUM": 0, "HIGH": 0, "EXTREME": 0})
        _local_tim.update(_tim_counts)   # already computed above
        _local_tim_total = sum(_local_tim.values()) or 1

        print(f"\n  ── Risk Mode distribution ───────────────────────────────")
        print(f"    {'Mode':<10}  {'N':>4}  {'%entries':>9}  {'%h4time':>8}  "
              f"{'AvgR':>6}  {'BestR':>6}  config")
        print(f"    {'─'*10}  {'─'*4}  {'─'*9}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*30}")
        for _rm in ["LOW", "MEDIUM", "HIGH", "EXTREME"]:
            _rm_trades = [t for t in trades if t.get("risk_mode_at_entry") == _rm]
            _h4_pct    = _local_tim[_rm] / _local_tim_total * 100
            _mult      = RISK_MODE_PARAMS.get(_rm, {}).get("risk_mult", 1.0)
            _wk_cap    = RISK_MODE_PARAMS.get(_rm, {}).get("weekly_cap_std", 2)
            _dd_l1     = RISK_MODE_PARAMS.get(_rm, {}).get("dd_l1_cap", 10)
            _dd_l2     = RISK_MODE_PARAMS.get(_rm, {}).get("dd_l2_cap", 6)
            if not _rm_trades and _h4_pct < 1:
                continue
            if _rm_trades:
                _cnt  = len(_rm_trades)
                _pct  = _cnt / len(trades) * 100
                _avgr = sum(t["r"] for t in _rm_trades) / _cnt
                _bestr= max(t["r"] for t in _rm_trades)
                print(f"    {_rm:<10}  {_cnt:>4}  {_pct:>8.0f}%  {_h4_pct:>7.0f}%  "
                      f"{_avgr:>+5.2f}R  {_bestr:>+5.2f}R  "
                      f"mult={_mult}×  wk_cap={_wk_cap}  dd={_dd_l1}/{_dd_l2}%")
            else:
                print(f"    {_rm:<10}  {'—':>4}  {'—':>8}   {_h4_pct:>7.0f}%  "
                      f"{'—':>6}  {'—':>6}  "
                      f"mult={_mult}×  wk_cap={_wk_cap}  dd={_dd_l1}/{_dd_l2}%")

    # ── Spread model summary ──────────────────────────────────────────────
    import src.strategy.forex.strategy_config as _sc_ref
    if _sc_ref.SPREAD_MODEL_ENABLED and trades:
        _total_spread  = sum(t.get("spread_cost", 0.0) for t in trades)
        _avg_spread    = _total_spread / len(trades)
        _spread_pct_pnl = (_total_spread / abs(net_pnl) * 100) if net_pnl != 0 else 0
        print(f"\n  ── Spread model (bid/ask round-trip) ──────────────────")
        print(f"    Total spread drag:  ${_total_spread:>+9,.2f}  ({_spread_pct_pnl:.1f}% of gross P&L)")
        print(f"    Avg per trade:      ${_avg_spread:>+9,.2f}")
        _spread_by_pair = {}
        for t in trades:
            p = t["pair"]
            _spread_by_pair[p] = _spread_by_pair.get(p, 0.0) + t.get("spread_cost", 0.0)
        for _p, _sc_val in sorted(_spread_by_pair.items(), key=lambda x: -x[1]):
            print(f"    {_p:<12}  ${_sc_val:>+7,.2f}")

    print(f"\n  ── Risk controls summary ──────────────────────────────")
    # Dollar loss range
    _losses_dollar = sorted([t["pnl"] for t in trades if t["pnl"] < 0])
    _max_dollar_loss = _losses_dollar[0] if _losses_dollar else 0.0
    print(f"    Max single-trade $ loss: ${_max_dollar_loss:,.2f}")
    print(f"    Avg loss $:              ${sum(_losses_dollar)/len(_losses_dollar):,.2f}"
          if _losses_dollar else "    Avg loss $:              n/a")

    # Worst-3-losses DD — sum of 3 biggest individual losses as % of starting balance.
    # Answers "if these 3 hit back-to-back from start, what DD?". Useful for comparing
    # LOW_mult variants: higher mult → bigger individual losses → higher Worst3L DD.
    _w3 = _losses_dollar[:3]   # already sorted most-negative first
    _w3_sum_pct = abs(sum(_w3)) / starting_bal * 100 if _w3 else 0.0
    _w3_pnl_str = "  ".join(f"${p:,.0f}" for p in _w3) if _w3 else "n/a"
    print(f"    Worst-3-losses DD (vs start): {_w3_sum_pct:.1f}%  [{_w3_pnl_str}]")

    # DD killswitch
    if dd_killswitch_blocks > 0:
        print(f"    DD_KILLSWITCH blocks:    {dd_killswitch_blocks}  "
              f"(≥{_sc.DD_KILLSWITCH_PCT:.0f}% DD — entries hard-blocked)")
    else:
        print(f"    DD_KILLSWITCH blocks:    0  (40% threshold never breached)")

    # DD caps on entered trades
    if _dd_cap_trades:
        _cap_counts = Counter(t["dd_flag"] for t in _dd_cap_trades)
        for flag, cnt in sorted(_cap_counts.items()):
            print(f"    {flag}: {cnt} trade(s) entered at reduced risk")

    # Loss streak at entry distribution
    _streak_vals = [t.get("streak_at_entry", 0) for t in trades]
    _streak_counts = Counter(_streak_vals)
    _streak_capped = sum(1 for v in _streak_vals if v >= 2)
    if max(_streak_vals, default=0) > 0:
        print(f"    Loss-streak distribution at entry:")
        for s in sorted(_streak_counts):
            label = "" if s < 2 else f" ← streak cap {'6%' if s < 3 else '3%'}"
            print(f"      streak={s}: {_streak_counts[s]} trade(s){label}")
        if _streak_capped:
            print(f"    Entries affected by streak brake: {_streak_capped}")

    # ── 1R audit: stop_hit trades ── prove abs(pnl) ≈ entry_risk_dollars ─────
    # Mathematical proof (JPY included): units = risk_usd/dist, pnl = dist×units = risk_usd
    # Sizing errors and P&L conversion errors cancel exactly for all pair types.
    # Any overrun > 1% indicates a sizing/P&L mismatch in the code.
    _sh_trades = [t for t in trades if t.get("reason") in ("stop_hit",)
                  and t.get("entry_risk_dollars", 0) > 0]
    if _sh_trades:
        print(f"\n  ── 1R audit (stop_hit trades) ──────────────────────────")
        _overruns = []
        for t in _sh_trades:
            expected    = t["entry_risk_dollars"]
            actual      = abs(t["pnl"])
            # Spread is a legitimate cost on top of 1R — subtract it before checking
            _sc_cost    = t.get("spread_cost", 0.0)
            overrun     = actual - expected - _sc_cost   # residual after spread
            overrun_pct = overrun / expected * 100 if expected else 0
            _overruns.append((overrun_pct, t))
        _max_op, _max_ot = max(_overruns, key=lambda x: abs(x[0]))
        _clean = all(abs(op) < 2.0 for op, _ in _overruns)
        print(f"    Stop-hit count:    {len(_sh_trades)}")
        print(f"    All within ±2%:    {'✅ YES' if _clean else '❌ NO — check sizing!'}")
        if not _clean:
            for op, t in sorted(_overruns, key=lambda x: -abs(x[0]))[:5]:
                print(f"      {t['pair']} {t['direction']} {t['entry_ts'][:10]}: "
                      f"expected ${t['entry_risk_dollars']:,.0f}  "
                      f"actual ${abs(t['pnl']):,.0f}  overrun={op:+.1f}%")
        print(f"    Max overrun:       {_max_op:+.1f}%  "
              f"({_max_ot['pair']} {_max_ot['entry_ts'][:10]}: "
              f"expected ${_max_ot['entry_risk_dollars']:,.0f}  "
              f"actual ${abs(_max_ot['pnl']):,.0f}  "
              f"equity@entry ${_max_ot.get('entry_equity',0):,.0f}  "
              f"rpct={_max_ot.get('final_risk_pct',0):.2f}%  "
              f"dd_flag={_max_ot.get('dd_flag','')!r})")

    # ── Alex small-account rules summary ──────────────────────────────────
    print(f"\n  ── Alex small-account rules ─────────────────────────────────")
    print(f"    MIN_RR_ALIGN blocks (non-protrend/mixed → 3.0R, protrend → 2.5R): {_min_rr_small_blocks}x")
    print(f"    Weekly limit blocks (punch-card):                {_weekly_limit_blocks}x")
    # Extract time blocks and HTF blocks from all_decisions
    _all_blocked  = [d for d in all_decisions if d.get("decision") == "BLOCKED"]
    _all_filters  = [f for d in all_decisions
                     for f in (d.get("failed_filters") or []) + [d.get("event","")]]
    _time_block_n  = sum(1 for d in all_decisions
                         for f in (d.get("failed_filters") or [])
                         if f in ("NO_SUNDAY_TRADES", "NO_THU_FRI_TRADES", "MONDAY_WICK_GUARD"))
    _htf_block_n   = sum(1 for d in all_decisions
                         for f in (d.get("failed_filters") or [])
                         if f == "COUNTERTREND_HTF")
    _indes_block_n = sum(1 for d in all_decisions
                         for f in (d.get("failed_filters") or [])
                         if f == "INDECISION_DOJI")
    _gap_blocks    = [d for d in all_decisions if d.get("event") == "GAP"]
    _gap_weekly    = sum(1 for d in _gap_blocks if d.get("gap_type","") == "WEEKLY_TRADE_LIMIT")
    _gap_rr_sm     = sum(1 for d in _gap_blocks if d.get("gap_type","") == "MIN_RR_ALIGN")
    print(f"    Time blocks (NO_SUNDAY/NO_THU_FRI):              {_time_block_n}x")
    print(f"    HTF alignment blocks (COUNTERTREND_HTF):         {_htf_block_n}x")
    print(f"    Indecision doji blocks (INDECISION_DOJI):        {_indes_block_n}x")
    # Trades per week distribution
    if _weekly_trade_counts:
        _week_vals = sorted(_weekly_trade_counts.values())
        _week_hist: dict = {}
        for v in _week_vals:
            _week_hist[v] = _week_hist.get(v, 0) + 1
        _hist_str = "  ".join(f"{cnt}trade×{wks}wk" for cnt, wks in sorted(_week_hist.items()))
        print(f"    Trades/week distribution:  {_hist_str}"
              f"  (avg {sum(_week_vals)/len(_week_vals):.1f}/week  max {max(_week_vals)})")
    # HTF alignment pass rate
    _htf_checked = sum(1 for d in all_decisions
                       if any(f in ("COUNTERTREND_HTF", "no_pattern", "trend_alignment")
                              for f in (d.get("failed_filters") or [])))
    _n_enter_raw = len([d for d in all_decisions if d.get("decision") == "ENTER"])
    if (_n_enter_raw + _htf_block_n) > 0:
        _htf_pass_rate = _n_enter_raw / (_n_enter_raw + _htf_block_n) * 100
        print(f"    HTF alignment pass rate:   {_htf_pass_rate:.0f}%"
              f"  ({_n_enter_raw} passed / {_htf_block_n} blocked by HTF gate)")

    # ── Funnel: rejection reasons from WAIT decisions ─────────────────
    _wait_decisions = [d for d in all_decisions if d.get("decision") == "WAIT"]
    _wait_reasons   = [
        f for d in _wait_decisions
        for f in d.get("failed_filters", [])
        if f
    ]
    _n_wait = len(set((d["ts"], d["pair"]) for d in _wait_decisions))
    _n_enter = len([d for d in all_decisions if d.get("decision") == "ENTER"])
    print(f"\n  ── Entry funnel ({_n_enter} entered / {_n_wait} unique setups blocked) ─────")
    if _wait_reasons:
        for reason, count in Counter(_wait_reasons).most_common(15):
            print(f"    {reason:<45} {count:>5}x")
    else:
        print("    (no WAIT reasons recorded — check failed_filters population)")

    # exec_rr_min breakdown: which candidate type was the bottleneck
    _rr_blocked = [d for d in _wait_decisions if "exec_rr_min" in d.get("failed_filters", [])]
    if _rr_blocked:
        import re as _re
        _cand_fail_types: list = []
        for d in _rr_blocked:
            # Parse "Tried: 4h_structure=1.20R (rr_too_low); measured_move=0.80R (rr_too_low)"
            for m in _re.finditer(r'(\w+)=([\d.]+)R \((\w+)\)', d.get("reason", "")):
                _cand_fail_types.append(f"{m.group(1)}:{m.group(3)}")
        if _cand_fail_types:
            print(f"\n  ── exec_rr_min detail ({len(_rr_blocked)} blocks, MIN_RR={_sc.MIN_RR}) ──────────")
            for ct, cnt in Counter(_cand_fail_types).most_common():
                print(f"    {ct:<40} {cnt:>5}x")

    print(f"\n  Trade log:")
    for t in trades:
        r_sign   = "+" if t["r"] >= 0 else ""
        pnl_sign = "+" if t["pnl"] >= 0 else ""
        status   = "✅" if t["pnl"] >= 0 else "❌"
        theme_tag = f"  🎯 {t['macro_theme']}" if t.get("macro_theme") else ""
        print(f"  {status} {t['pair']:<10} {t['direction']:<6} "
              f"entry={t['entry']:.5f}  exit={t['exit']:.5f}  "
              f"{r_sign}{t['r']:.1f}R  ${pnl_sign}{t['pnl']:,.2f}  "
              f"[{t['reason']}]  {t['notes'][:40]}{theme_tag}")

    # ── Gap summary ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"GAP ANALYSIS — v2 vs v1")
    print(f"{'='*65}")
    if not gap_log:
        print("  No gaps detected — v1 and v2 in full agreement.")
    else:
        by_type: dict = {}
        for g in gap_log:
            gt = g["gap_type"]
            by_type.setdefault(gt, []).append(g)
        for gt, items in by_type.items():
            print(f"\n  [{gt}] — {len(items)} occurrence(s):")
            for g in items[:5]:   # show first 5 of each type
                print(f"    {g['ts'][:16]}  {g['pair']:<10}  v1={g['v1']}  v2={g['v2']}")
                print(f"      → {g['detail'][:90]}")
            if len(items) > 5:
                print(f"    ... and {len(items)-5} more. See {GAP_LOG_PATH}")

    # ── Save outputs ──────────────────────────────────────────────────
    GAP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GAP_LOG_PATH, "w") as f:
        for g in gap_log:
            f.write(json.dumps(g) + "\n")

    with open(DECISION_LOG, "w") as f:
        json.dump({
            "run_dt":    datetime.now(timezone.utc).isoformat(),
            "start_dt":  start_dt.isoformat(),
            "starting_bal": starting_bal,
            "final_bal": balance,
            "trades":    trades,
            "decisions": all_decisions,
            "gaps":      gap_log,
        }, f, indent=2)

    print(f"\n  Decision log: {DECISION_LOG}")
    print(f"  Gap log:      {GAP_LOG_PATH}")

    # ── Backtest results log (append) ─────────────────────────────────
    # Every run appends one JSON line to logs/backtest_results.jsonl.
    # Includes git commit + dirty flag so we always know EXACTLY what code
    # produced a given result. No more "where did +66% come from?" mysteries.
    import subprocess as _sp
    try:
        _commit = _sp.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent.parent,
            stderr=_sp.DEVNULL).decode().strip()
        _dirty  = bool(_sp.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).parent.parent,
            stderr=_sp.DEVNULL).decode().strip())
    except Exception:
        _commit, _dirty = "unknown", False

    # ── Pairs hash — reproducibility fingerprint ──────────────────────────
    import hashlib as _hl
    _active_pairs  = sorted(candle_data.keys())
    _pairs_hash    = _hl.md5(",".join(_active_pairs).encode()).hexdigest()[:6]

    _result_record = {
        "run_dt":       datetime.now(timezone.utc).isoformat(),
        "commit":       _commit + ("-dirty" if _dirty else ""),
        "notes":        notes,
        # model_tags: all active levers + trail arm + pairs fingerprint.
        # Always a list[str] — never contains dicts or None (defensive str() cast
        # inside get_model_tags guarantees this).
        "model_tags":   _sc.get_model_tags(trail_arm=trail_arm_key or "?",
                                           pairs_hash=_pairs_hash)
                        + ([f"policy={policy_tag}"]       if policy_tag else [])
                        + ([f"risk={flat_risk_pct:.0f}pct"] if flat_risk_pct else [])
                        + ([f"forced_mode={force_risk_mode}"] if force_risk_mode else []),
        "trail_arm":    trail_arm_key or "?",
        "pairs":        _active_pairs,
        "pairs_hash":   _pairs_hash,
        "window_start": start_dt.isoformat(),
        "window_end":   (end_dt or datetime.now(timezone.utc)).isoformat(),
        "config": {
            "starting_bal":                      starting_bal,
            "ATR_MIN_MULTIPLIER":                _sc.ATR_MIN_MULTIPLIER,
            "ATR_STOP_MULTIPLIER":               _sc.ATR_STOP_MULTIPLIER,
            "MIN_CONFIDENCE":                    _sc.MIN_CONFIDENCE,
            "MIN_RR":                            _sc.MIN_RR,
            "MAX_CONCURRENT_TRADES_BACKTEST":    _sc.MAX_CONCURRENT_TRADES_BACKTEST,
            "MAX_CONCURRENT_TRADES_LIVE":        _sc.MAX_CONCURRENT_TRADES_LIVE,
            "BLOCK_ENTRY_WHILE_WINNER_RUNNING":  _sc.BLOCK_ENTRY_WHILE_WINNER_RUNNING,
            "WINNER_THRESHOLD_R":                _sc.WINNER_THRESHOLD_R,
            "ENTRY_TRIGGER_MODE":                _sc.ENTRY_TRIGGER_MODE,
            "LEVEL_ALLOW_FINE_INCREMENT":        _sc.LEVEL_ALLOW_FINE_INCREMENT,
            "STRUCTURAL_LEVEL_MIN_SCORE":        _sc.STRUCTURAL_LEVEL_MIN_SCORE,
            "ALLOW_BREAK_RETEST":                _sc.ALLOW_BREAK_RETEST,
            "OVEREXTENSION_CHECK":               _sc.OVEREXTENSION_CHECK,
            "OVEREXTENSION_THRESHOLD":           _sc.OVEREXTENSION_THRESHOLD,
            "ALLOW_TIER3_REVERSALS":             _sc.ALLOW_TIER3_REVERSALS,
            "REQUIRE_THEME_GATE":                _sc.REQUIRE_THEME_GATE,
            # Alex small-account gates
            "NO_SUNDAY_TRADES_ENABLED":          _sc.NO_SUNDAY_TRADES_ENABLED,
            "NO_THU_FRI_TRADES_ENABLED":         _sc.NO_THU_FRI_TRADES_ENABLED,
            "REQUIRE_HTF_TREND_ALIGNMENT":       _sc.REQUIRE_HTF_TREND_ALIGNMENT,
            "MAX_TRADES_PER_WEEK_SMALL":         _sc.MAX_TRADES_PER_WEEK_SMALL,
            "MAX_TRADES_PER_WEEK_STANDARD":      _sc.MAX_TRADES_PER_WEEK_STANDARD,
            "MIN_RR_ALIGN":              _sc.MIN_RR_SMALL_ACCOUNT,
            "MIN_RR_STANDARD":                   _sc.MIN_RR_STANDARD,
            "INDECISION_FILTER_ENABLED":         _sc.INDECISION_FILTER_ENABLED,
            # Extra run_backtest param gates (not in strategy_config)
            "adaptive_gates":                    adaptive_gates,
            "adaptive_threshold":                adaptive_threshold,
            "strict_protrend_htf":               strict_protrend_htf,
            "wd_protrend_htf":                   wd_protrend_htf,
            "dynamic_pip_equity":                dynamic_pip_equity,
            "policy_tag":                        policy_tag,
            "force_risk_mode":                   force_risk_mode,
            "streak_demotion_thresh":             streak_demotion_thresh,
        },
        "results": {
            "n_trades":   len(trades),
            "n_wins":     len(wins),
            "n_losses":   len(losses),
            "win_rate":   round(wr, 1),
            "net_pnl":    round(net_pnl, 2),
            "final_bal":  round(balance, 2),
            "return_pct": round(ret_pct, 2),
        },
        "trades": [
            {
                "pair":                  t["pair"],
                "direction":             t["direction"],
                "entry":                 t["entry"],
                "exit":                  t["exit"],
                "r":                     round(t["r"], 2),
                "pnl":                   round(t["pnl"], 2),
                "reason":                t["reason"],
                "pattern":               t.get("pattern", ""),
                "macro_theme":           t.get("macro_theme", ""),
                "entry_ts":              t.get("entry_ts", ""),
                "exit_ts":               t.get("exit_ts", ""),
                # Risk mode audit (added 2026-02-28 — always present from now on)
                "risk_mode_at_entry":    t.get("risk_mode_at_entry", "MEDIUM"),
                "entry_risk_dollars":    round(t.get("entry_risk_dollars", 0), 2),
                "base_risk_pct":         round(t.get("base_risk_pct", 0), 4),
                "final_risk_pct":        round(t.get("final_risk_pct", 0), 4),
                "entry_equity":          round(t.get("entry_equity", 0), 2),
                "streak_at_entry":       t.get("streak_at_entry", 0),
            }
            for t in trades
        ],
        "gap_summary": {
            gt: len(items)
            for gt, items in {
                g["gap_type"]: [x for x in gap_log if x["gap_type"] == g["gap_type"]]
                for g in gap_log
            }.items()
        },
    }

    _results_log = Path(__file__).parent.parent / "logs" / "backtest_results.jsonl"
    _results_log.parent.mkdir(parents=True, exist_ok=True)
    with open(_results_log, "a") as _f:
        _f.write(json.dumps(_result_record) + "\n")

    print(f"  Results log:  {_results_log}  [{_commit}{'-dirty' if _dirty else ''}]")

    # ── Chop Shield summary (Arm D) ────────────────────────────────────────
    if chop_shield:
        print(f"\n  ── Chop Shield (Arm D) ──────────────────────────────────────")
        print(f"    Auto-pauses fired (streak≥{getattr(_sc,'CHOP_SHIELD_STREAK_THRESH',3)}): {_bt_chop_auto_pauses}x")
        print(f"    Entries blocked during 48h pause:  {_bt_chop_paused_blocks}x")
        print(f"    Entries blocked by recovery gates: {_bt_chop_recovery_blocks}x")
        print(f"    chopped={_bt_chop_auto_pauses}p/{_bt_chop_paused_blocks}b/{_bt_chop_recovery_blocks}r")

    # ── Compact comparison line (useful when diffing mult variants) ────────
    try:
        _cmp_low_n  = sum(1 for m in _rm_at_entry if m == "LOW")
        _cmp_med_n  = sum(1 for m in _rm_at_entry if m == "MEDIUM")
        _cmp_high_n = sum(1 for m in _rm_at_entry if m == "HIGH")
        _cmp_ext_n  = sum(1 for m in _rm_at_entry if m == "EXTREME")
        _low_m = RISK_MODE_PARAMS["LOW"]["risk_mult"]
        _chop_tag = (f"  chopped={_bt_chop_auto_pauses}p/{_bt_chop_paused_blocks}b"
                     f"/{_bt_chop_recovery_blocks}r") if chop_shield else ""
        print(f"\n  ▶ SUMMARY  low_mult={_low_m}×"
              f"  ret={ret_pct:+.1f}%  maxDD={_max_dd_pct:.1f}%"
              f"  worst3L={_w3_sum_pct:.1f}%"
              f"  LOW={_cmp_low_n}  MED={_cmp_med_n}"
              f"  HIGH={_cmp_high_n}  EXT={_cmp_ext_n}"
              f"{_chop_tag}")
    except Exception:
        pass  # compact summary is optional — never crash the run

    # ── Auto-run miss analyzer on Alex window ─────────────────────────
    # Only fires when running the Jul 15 – Oct 31 2024 window so we
    # always get an up-to-date Alex vs bot scorecard.
    _is_alex_window = (
        start_dt.strftime("%Y-%m")[:7] == "2024-07"
        and end_dt is not None
        and end_dt.strftime("%Y-%m")[:7] == "2024-10"
    )
    if _is_alex_window:
        try:
            from backtesting.miss_analyzer import analyze as _miss_analyze
            print()
            _miss_analyze(verbose=True)
        except Exception as _me:
            print(f"  [miss analyzer error: {_me}]")

    # ── Return result dict (used by multi-arm comparison) ─────────────
    _wins   = [t for t in trades if t["r"] >  0.10]
    _losses = [t for t in trades if t["r"] < -0.10]
    _ec     = Counter(t["reason"] for t in trades)
    _rrs    = sorted([t.get("planned_rr", 0) for t in trades if t.get("planned_rr", 0) > 0])

    _regime_scores = [t.get("regime_score_at_entry", 0.0) for t in trades]

    # ── Risk mode distribution (% of TRADES entered in each mode) ────────
    # _tim_counts / _tim_pct_* already computed above in the print section
    _rm_vals = [t.get("risk_mode_at_entry", "MEDIUM") for t in trades]
    _n = len(trades) or 1
    _rm_pct_low     = sum(1 for m in _rm_vals if m == "LOW")     / _n * 100
    _rm_pct_medium  = sum(1 for m in _rm_vals if m == "MEDIUM")  / _n * 100
    _rm_pct_high    = sum(1 for m in _rm_vals if m == "HIGH")    / _n * 100
    _rm_pct_extreme = sum(1 for m in _rm_vals if m == "EXTREME") / _n * 100

    return BacktestResult(
        # ── Core performance (canonical names) ───────────────────────
        return_pct     = (balance - starting_bal) / starting_bal * 100,
        max_dd_pct     = _max_dd_pct,
        win_rate       = len(_wins) / len(trades) if trades else 0,
        avg_r          = sum(t["r"] for t in trades) / len(trades) if trades else 0,
        best_r         = max((t["r"] for t in trades), default=0),
        worst_r        = min((t["r"] for t in trades), default=0),
        n_trades       = len(trades),
        balance        = balance,
        # ── Trade breakdown ───────────────────────────────────────────
        avg_r_win      = sum(t["r"] for t in _wins)   / len(_wins)   if _wins   else 0,
        avg_r_loss     = sum(t["r"] for t in _losses) / len(_losses) if _losses else 0,
        n_target       = _ec.get("target_reached", 0) + _ec.get("weekend_proximity", 0),
        n_ratchet      = _ec.get("ratchet_stop_hit", 0),
        n_sl           = _ec.get("stop_hit", 0),
        exec_rr_p50    = _rrs[len(_rrs) // 2] if _rrs else 0,
        max_dollar_loss= min((t["pnl"] for t in trades), default=0.0),
        # ── Gate hit counts (all 6 Alex rules + adaptive) ────────────
        time_blocks              = _time_block_n,
        countertrend_htf_blocks  = _htf_block_n,
        weekly_limit_blocks      = _weekly_limit_blocks,
        min_rr_small_blocks      = _min_rr_small_blocks,
        indecision_doji_blocks   = _indes_block_n,
        dd_killswitch_blocks     = dd_killswitch_blocks,
        adaptive_time_blocks     = _adaptive_time_blocks,
        strict_htf_blocks        = _strict_htf_blocks,
        wd_htf_blocks            = _wd_htf_blocks,
        dyn_pip_eq_blocks        = _dyn_pip_eq_blocks,
        wd_alignment_pct         = (_wd_aligned_entries / len(trades) * 100
                                    if trades else 0.0),
        # ── Stop quality ──────────────────────────────────────────────
        stop_type_counts         = dict(_stop_types),
        atr_fallback_pct         = _atr_fb_pct,
        stop_pips_p50            = _sp_p50,
        # ── Regime analytics ─────────────────────────────────────────
        regime_scores      = _regime_scores,
        regime_avg         = sum(_regime_scores) / len(_regime_scores) if _regime_scores else 0.0,
        regime_pct_high    = (sum(1 for s in _regime_scores if s >= 3.0)
                              / len(_regime_scores) * 100 if _regime_scores else 0.0),
        regime_pct_extreme = (sum(1 for s in _regime_scores if s >= 3.5)
                              / len(_regime_scores) * 100 if _regime_scores else 0.0),
        # ── Risk mode: % of trades in each mode ──────────────────────
        risk_mode_pct_low     = _rm_pct_low,
        risk_mode_pct_medium  = _rm_pct_medium,
        risk_mode_pct_high    = _rm_pct_high,
        risk_mode_pct_extreme = _rm_pct_extreme,
        # ── Risk mode: % of calendar time in each mode ───────────────
        time_in_mode_pct_low     = _tim_pct_low,
        time_in_mode_pct_medium  = _tim_pct_medium,
        time_in_mode_pct_high    = _tim_pct_high,
        time_in_mode_pct_extreme = _tim_pct_extreme,
        # ── Chop Shield ──────────────────────────────────────────────
        chop_auto_pauses     = _bt_chop_auto_pauses,
        chop_paused_blocks   = _bt_chop_paused_blocks,
        chop_recovery_blocks = _bt_chop_recovery_blocks,
        # ── Profiling ────────────────────────────────────────────────
        api_calls     = _api_call_count,
        eval_calls    = _eval_calls,
        eval_ms_avg   = (_eval_ms / _eval_calls) if _eval_calls else 0.0,
        # ── Raw lists ────────────────────────────────────────────────
        trades         = trades,
        trades_per_week= dict(_weekly_trade_counts),
        gap_log        = gap_log,
        candle_data    = candle_data,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OANDA Backtest v2 — Real Strategy Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Lever system — override any strategy_config constant without editing source:

  --lever KEY=VALUE       Set one lever (repeat for multiple)
  --profile NAME          Load profiles/<name>.json (applied before --lever flags)

Built-in window shortcuts:
  --window alex           2024-07-01 → 2024-10-31  (Alex's $100→$1M challenge)
  --window jan            2026-01-01 → 2026-01-31  (Jan 2026 live window)

Examples:
  python3 -m backtesting.oanda_backtest_v2 --window alex
  python3 -m backtesting.oanda_backtest_v2 --window alex --lever LEVEL_ALLOW_FINE_INCREMENT=False
  python3 -m backtesting.oanda_backtest_v2 --window jan  --profile core_reversals
  python3 -m backtesting.oanda_backtest_v2 --window alex --lever ALLOW_BREAK_RETEST=False --lever MIN_CONFIDENCE=0.70
""")
    parser.add_argument("--start",   default="2025-07-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=None,         help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--balance", type=float, default=STARTING_BAL, help="Starting balance")
    parser.add_argument("--notes",   default="",           help="Description of what changed (logged with results)")
    parser.add_argument("--window",  default=None,         help="Shortcut: alex=Jul-Oct 2024 | jan=Jan 2026")
    parser.add_argument("--profile", default=None,         help="Load profiles/<name>.json lever profile")
    parser.add_argument("--lever",   action="append",      default=[],
                        metavar="KEY=VALUE",
                        help="Override a strategy_config lever (e.g. --lever MIN_CONFIDENCE=0.70)")
    parser.add_argument("--news-filter", action="store_true", default=False,
                        help="Enable historical news filter (CSV from data/news/). Alex never used this.")
    parser.add_argument("--arm",   default="A",
                        help="Trail arm: A | B | C | C1 | C2 | C3 | D | all  (default: A). "
                             "D=Arm C + Chop Shield. all=run every arm.")
    parser.add_argument("--chop-shield", action="store_true", default=False,
                        help="Enable Chop Shield gates (auto-enabled for --arm D).")
    parser.add_argument("--cache", action="store_true", default=False,
                        help="No-op (kept for backward compat). Cache is ON by default. "
                             "Use --no-cache to disable.")
    parser.add_argument("--no-cache", action="store_true", default=False,
                        help="Disable per-pair disk cache — always fetch from OANDA. "
                             f"Cache dir: {CACHE_DIR}")
    parser.add_argument("--max-trades", type=int, default=None,
                        help="Override MAX_CONCURRENT_TRADES_BACKTEST. Default=1 (parity with live). "
                             "Values >1 are tagged EXPERIMENTAL in results — never use as baseline.")
    parser.add_argument("--timing", action="store_true", default=False,
                        help="Print performance stats after each run: total bars, evaluate() calls, "
                             "avg ms per call, total runtime.")
    parser.add_argument("--protrend-only", action="store_true", default=False,
                        help="Gate entries: require Weekly AND Daily bias to agree with trade direction. "
                             "4H is exempt (Alex enters during 4H retracements). "
                             "Wires PROTREND_ONLY=True. Use for W2 regime vs bug diagnostic.")
    parser.add_argument("--force-mode", default=None,
                        metavar="MODE",
                        help="Pin risk mode for every entry: LOW | MEDIUM | HIGH | EXTREME. "
                             "None = AUTO (compute dynamically per entry — default). "
                             "Use for mode comparison runs.")
    parser.add_argument("--low-mult", type=float, default=None,
                        metavar="MULT",
                        help="Override LOW risk-mode risk_mult (default 0.5). "
                             "e.g. --low-mult 0.7 to reduce drag vs MEDIUM in bad markets. "
                             "Only applies in AUTO mode when entries fall into LOW mode.")
    parser.add_argument("--demote-after-losses", type=int, default=None,
                        metavar="N",
                        dest="demote_after_losses",
                        help="Consecutive losses needed to flip streak_clear=False (default 1). "
                             "1=current: any single loss demotes score; "
                             "2=relax: need 2+ losses; keeps 1-loss trades in MEDIUM not LOW. "
                             "Only affects mode assignment — HIGH gate uses loss_streak<=1 unchanged.")
    args = parser.parse_args()

    # ── Window shortcuts ──────────────────────────────────────────────────────
    if args.window == "alex":
        args.start = "2024-07-01"
        args.end   = "2024-10-31"
    elif args.window == "jan":
        args.start = "2026-01-01"
        args.end   = "2026-01-31"
    elif args.window is not None:
        parser.error(f"Unknown --window '{args.window}'. Use: alex | jan")

    # ── Apply lever profile (before individual --lever flags) ─────────────────
    if args.profile:
        try:
            applied = _sc.load_profile(args.profile)
            print(f"  📋 Profile '{args.profile}' loaded: {applied}")
        except FileNotFoundError as e:
            parser.error(str(e))

    # ── News filter ───────────────────────────────────────────────────────────
    if args.news_filter:
        _sc.apply_levers({"NEWS_FILTER_ENABLED": True})
        print("  📰 News filter ENABLED — using data/news/high_impact_events.csv")

    # ── Apply individual --lever overrides ────────────────────────────────────
    if args.lever:
        overrides = {}
        for kv in args.lever:
            if "=" not in kv:
                parser.error(f"--lever must be KEY=VALUE, got: '{kv}'")
            k, v = kv.split("=", 1)
            overrides[k.strip()] = v.strip()
        try:
            applied = _sc.apply_levers(overrides)
            print(f"  🔧 Levers applied: {applied}")
        except ValueError as e:
            parser.error(str(e))

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end else None

    # ── LOW multiplier override ────────────────────────────────────────────────
    # Patches RISK_MODE_PARAMS["LOW"]["risk_mult"] for this run only.
    # Allows quantifying drag/protection tradeoff without touching source constants.
    if args.low_mult is not None:
        _low_mult_val = round(float(args.low_mult), 3)
        RISK_MODE_PARAMS["LOW"]["risk_mult"] = _low_mult_val
        print(f"  ⚡ LOW risk_mult overridden → {_low_mult_val}×  (default: 0.5×)")

    # ── Streak demotion threshold override ────────────────────────────────────
    # Controls how many consecutive losses flip streak_clear=False in compute_risk_mode.
    # 1 (default): any loss → LOW eligible; 2: need 2+ losses → 1-loss stays MEDIUM.
    _streak_demote_thresh = int(args.demote_after_losses) if args.demote_after_losses is not None else 1
    if args.demote_after_losses is not None:
        print(f"  ⚡ streak_demotion_thresh overridden → {_streak_demote_thresh}  (default: 1)")

    # Build notes string that captures any lever overrides
    notes = args.notes
    extra = []
    if args.news_filter:
        extra.append("news_filter=on")
    if args.profile:
        extra.append(f"profile={args.profile}")
    if args.lever:
        extra.extend(args.lever)
    if args.low_mult is not None:
        extra.append(f"low_mult={args.low_mult}")
    if args.demote_after_losses is not None:
        extra.append(f"demote_after={_streak_demote_thresh}")
    if extra:
        notes = (notes + " [levers: " + ", ".join(extra) + "]").strip()

    # ── Validate --arm ────────────────────────────────────────────────────────
    _arm_arg = args.arm.upper()
    _valid_arms = set(TRAIL_ARMS.keys()) | {"ALL"}
    if _arm_arg not in _valid_arms:
        parser.error(f"--arm must be one of {sorted(_valid_arms)}, got '{args.arm}'")
    _arms_to_run = list(TRAIL_ARMS.keys()) if _arm_arg == "ALL" else [_arm_arg]

    # ── Cache mode ────────────────────────────────────────────────────────────
    _use_cache = not getattr(args, "no_cache", False)
    if not _use_cache:
        print("  ⚠ Cache disabled (--no-cache) — all data fetched from OANDA")

    # ── Concurrency override (EXPERIMENTAL) ───────────────────────────────────
    # Default = 1 (parity with live). --max-trades N lets you test multi-position
    # behaviour explicitly. Any run with N > 1 is tagged EXPERIMENTAL in results
    # so it can never be confused with a parity baseline.
    _max_trades_override = getattr(args, "max_trades", None)
    if _max_trades_override is not None:
        if _max_trades_override < 1:
            parser.error("--max-trades must be ≥ 1")
        _sc.apply_levers({"MAX_CONCURRENT_TRADES_BACKTEST": _max_trades_override})
        if _max_trades_override > 1:
            _exp_tag = f"EXPERIMENTAL:max_trades={_max_trades_override}"
            notes = (f"{_exp_tag} " + notes).strip()
            print(f"  ⚠ EXPERIMENTAL: MAX_CONCURRENT_TRADES_BACKTEST={_max_trades_override} "
                  f"(default=1, parity with live). Tagged in results.")
        else:
            print(f"  ✓ MAX_CONCURRENT_TRADES_BACKTEST={_max_trades_override} (explicit parity)")

    # ── Run arms ──────────────────────────────────────────────────────────────
    # For multi-arm runs, candle data is fetched once then shared in-memory
    # across subsequent arms via preloaded_candle_data — no redundant I/O.
    _arm_results: dict = {}
    _shared_candles: Optional[dict] = None   # populated after first arm's fetch

    for _arm_key in _arms_to_run:
        _tcfg      = TRAIL_ARMS[_arm_key]
        _arm_notes = f"{notes} [arm={_arm_key}]".strip()

        _t_run = time.perf_counter()
        _protrend_flag = getattr(args, "protrend_only", False)
        _force_mode_raw = getattr(args, "force_mode", None)
        _force_mode = _force_mode_raw.upper() if _force_mode_raw else None
        if _force_mode and _force_mode not in ("LOW", "MEDIUM", "HIGH", "EXTREME"):
            parser.error(f"--force-mode must be LOW|MEDIUM|HIGH|EXTREME, got '{_force_mode_raw}'")
        _chop_flag = getattr(args, "chop_shield", False) or bool(_tcfg.get("_chop_shield"))
        result = run_backtest(
            start_dt=start, end_dt=end, starting_bal=args.balance,
            notes=_arm_notes, trail_cfg=_tcfg, trail_arm_key=_arm_key,
            preloaded_candle_data=_shared_candles,
            use_cache=_use_cache,
            wd_protrend_htf=_protrend_flag,
            force_risk_mode=_force_mode,
            streak_demotion_thresh=_streak_demote_thresh,
            chop_shield=_chop_flag,
        )
        _t_run_elapsed = time.perf_counter() - _t_run

        if args.timing and result:
            _ec  = result.get("eval_calls", 0)
            _ems = result.get("eval_ms_avg", 0)
            _ac  = result.get("api_calls", 0)
            print(f"\n  ── Timing ──────────────────────────────────────────")
            print(f"    Total runtime:        {_t_run_elapsed:.1f}s")
            print(f"    evaluate() calls:     {_ec:,}")
            print(f"    avg ms / evaluate():  {_ems:.2f} ms")
            print(f"    OANDA API calls:      {_ac}")
            _n_pairs = len(result.get("candle_data", {}))
            print(f"    Pairs simulated:      {_n_pairs}")

        # Share candle data from first arm with all subsequent arms in-process.
        # Bypasses cache overhead entirely for arms 2+ in a single run.
        if _shared_candles is None and result is not None:
            _shared_candles = result.get("candle_data")

        if result:
            _arm_results[_arm_key] = result

    # ── Multi-arm comparison table ────────────────────────────────────────────
    if len(_arm_results) > 1:
        print(f"\n{'='*75}")
        print("MULTI-ARM COMPARISON SUMMARY")
        print(f"{'='*75}")
        hdr = f"{'Metric':<28}" + "".join(f"{'Arm '+k:>14}" for k in _arm_results)
        print(hdr)
        print("-" * len(hdr))

        def _row(label, key, fmt=lambda x: str(x)):
            vals = [_arm_results[k].get(key) for k in _arm_results]
            row = f"  {label:<26}" + "".join(
                f"{fmt(v) if v is not None else 'n/a':>14}" for v in vals
            )
            print(row)

        _row("Trades",        "n_trades",      lambda x: str(x))
        _row("Win rate",      "win_rate",       lambda x: f"{x:.0%}")
        _row("Avg R",         "avg_r",          lambda x: f"{x:+.2f}R")
        _row("Best R",        "best_r",         lambda x: f"{x:+.1f}R")
        _row("Avg win R",     "avg_r_win",      lambda x: f"{x:+.2f}R")
        _row("Avg loss R",    "avg_r_loss",     lambda x: f"{x:+.2f}R")
        _row("Target reached","n_target",       lambda x: str(x))
        _row("Ratchet stops", "n_ratchet",      lambda x: str(x))
        _row("Stop hits",     "n_sl",           lambda x: str(x))
        _row("Exec RR p50",   "exec_rr_p50",    lambda x: f"{x:.1f}R")
        _row("Max DD",             "max_dd",                lambda x: f"{x:.1f}%")
        _row("Return",             "ret_pct",               lambda x: f"{x:+.1f}%")
        _row("Kill-switch blocks", "dd_killswitch_blocks",  lambda x: str(x))
        _row("Max $ loss",         "max_dollar_loss",       lambda x: f"${x:,.0f}")
        print(f"{'='*75}")
