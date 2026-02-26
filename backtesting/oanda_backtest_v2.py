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
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List

# ── Real strategy imports ─────────────────────────────────────────────────────
from src.exchange.oanda_client import OandaClient, INSTRUMENT_MAP
from src.strategy.forex.set_and_forget   import SetAndForgetStrategy, Decision
from src.strategy.forex.news_filter      import NewsFilter
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
                 preloaded_candle_data: Optional[dict] = None,
                 use_cache: bool = True):
    end_naive = end_dt.replace(tzinfo=None) if end_dt else None
    # Extend data fetch so open positions can run to natural close after the entry window.
    # Entries stop at end_dt; monitoring continues up to end_dt + RUNOUT_DAYS.
    RUNOUT_DAYS = 180
    runout_dt  = (end_dt + pd.Timedelta(days=RUNOUT_DAYS)).replace(tzinfo=timezone.utc) if end_dt else None

    _tcfg  = trail_cfg or TRAIL_ARMS[_DEFAULT_ARM]
    _tlabel = _tcfg.get("label", "")

    print(f"\n{'='*65}")
    print(f"OANDA 1H BACKTEST v2 — Real Strategy Code")
    print(f"Start: {start_dt.date()}  |  End: {end_dt.date() if end_dt else 'today'}  |  Capital: ${starting_bal:,.2f}")
    if _tlabel:
        print(f"Trail: {_tlabel}")
    print(f"{'='*65}")

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
    journal = TradeJournal()
    risk    = ForexRiskManager(journal=journal)

    # ── State ─────────────────────────────────────────────────────────
    balance       = starting_bal
    peak_balance  = starting_bal    # tracks equity high-water mark for DD circuit breaker
    open_pos: Dict[str, dict] = {}   # pair → position dict
    trades        = []
    all_decisions = []
    v1_decisions  = _load_v1_decisions()
    consecutive_losses: int = 0     # streak counter: reset on win, incremented on loss
    dd_killswitch_blocks: int = 0   # count of entries blocked by 40% DD killswitch

    def _risk_pct(bal):
        """DD-aware risk: applies graduated caps when equity is below peak."""
        pct, _dd_flag = risk.get_risk_pct_with_dd(
            bal, peak_equity=peak_balance, consecutive_losses=consecutive_losses)
        return pct

    def _risk_pct_with_flag(bal):
        """Returns (pct, dd_flag) — use for per-trade diagnostics."""
        return risk.get_risk_pct_with_dd(
            bal, peak_equity=peak_balance, consecutive_losses=consecutive_losses)

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
                decision = strat.evaluate(
                    pair        = pair,
                    df_weekly   = hist_w  if len(hist_w)  >= 4  else pd.DataFrame(),
                    df_daily    = hist_d  if len(hist_d)  >= 10 else pd.DataFrame(),
                    df_4h       = hist_4h,
                    df_1h       = hist_1h,
                    current_dt  = ts_utc,
                    macro_theme = active_theme if _is_theme_pair else None,
                )
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
            if not _is_theme_pair and not _is_cb_pattern and pe < _sc.MIN_PIP_EQUITY:
                log_gap(ts_utc, pair, "ENTER", "BLOCKED", "low_pip_equity",
                        f"Pip equity {pe:.0f}p < min {_sc.MIN_PIP_EQUITY:.0f}p — "
                        f"setup too small to consume a slot")
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

    print(f"\n{'='*65}")
    print(f"RESULTS — v2 (Real Strategy Code)")
    print(f"{'='*65}")
    print(f"  Trades:       {len(trades)}  "
          f"({len(wins)}W / {len(losses)}L / {len(scratches)}S)")
    print(f"  Win rate:     {wr:.0f}%  (all)    "
          f"Conditional (excl scratch): {cond_wr:.0f}%")
    print(f"  Scratch rate: {len(scratches)/len(trades)*100:.0f}%  "
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
    print(f"    Exec RR avg:          {_avg_plan_rr:.2f}R  (n={_plan_rr_n})")
    print(f"    Exec RR p50/p95:      {_rr_p50:.2f}R / {_rr_p95:.2f}R")
    print(f"    Exec RR < 2.8:        {len(_low_rr)} trades ({len(_low_rr)/len(trades)*100:.0f}%)  ← should be ~0 after fix")
    print(f"    Exec RR = 0 (N/A):    {len(_zero_rr)} trades  ← no qualifying target at entry")
    print(f"    Avg MFE:              {_avg_mfe:+.2f}R")
    print(f"    Avg MAE:              {_avg_mae:+.2f}R")
    print(f"\n  ── Exit reason counts ────────────────────────────────────")
    _reason_order = ["target_reached", "weekend_proximity",
                     "ratchet_stop_hit", "stop_hit", "max_hold", "runout_expired"]
    _all_reasons  = set(_exit_counts.keys())
    _sorted_reasons = ([r for r in _reason_order if r in _all_reasons] +
                       sorted(_all_reasons - set(_reason_order)))
    for reason in _sorted_reasons:
        cnt = _exit_counts[reason]
        avg_r = (sum(t["r"] for t in trades if t["reason"] == reason) / cnt
                 if cnt else 0.0)
        print(f"    {reason:<30} {cnt:>3}  ({cnt/len(trades)*100:.0f}%)  "
              f"avg {avg_r:+.2f}R")

    if _trigger_counts:
        print(f"\n  ── Trigger types ──────────────────────────────────────")
        for sig_type, count in sorted(_trigger_counts.items(), key=lambda x: -x[1]):
            pct = count / len(trades) * 100 if trades else 0
            print(f"    {sig_type:<30} {count:>3}  ({pct:.0f}%)")

    # Stop type distribution
    _stop_types = Counter(pos.get("stop_type", "unknown") for pos in trades)
    if _stop_types:
        print(f"\n  ── Stop type distribution ─────────────────────────────")
        for stype, cnt in _stop_types.most_common():
            pct = cnt / len(trades) * 100
            print(f"    {stype:<35} {cnt:>3}  ({pct:.0f}%)")
        _sp_vals = [t.get("initial_stop_pips", 0) for t in trades if t.get("initial_stop_pips", 0) > 0]
        if _sp_vals:
            _sp_vals.sort()
            print(f"    Stop pips:  min={_sp_vals[0]:.0f}p  p50={_sp_vals[len(_sp_vals)//2]:.0f}p  "
                  f"p90={_sp_vals[int(len(_sp_vals)*0.9)]:.0f}p  max={_sp_vals[-1]:.0f}p")

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

    _result_record = {
        "run_dt":       datetime.now(timezone.utc).isoformat(),
        "commit":       _commit + ("-dirty" if _dirty else ""),
        "notes":        notes,
        "model_tags":   _sc.get_model_tags(),   # reads live module state → captures lever overrides
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
                "pair":        t["pair"],
                "direction":   t["direction"],
                "entry":       t["entry"],
                "exit":        t["exit"],
                "r":           round(t["r"], 2),
                "pnl":         round(t["pnl"], 2),
                "reason":      t["reason"],
                "pattern":     t.get("pattern", ""),
                "macro_theme": t.get("macro_theme", ""),
                "entry_ts":    t.get("entry_ts", ""),
                "exit_ts":     t.get("exit_ts", ""),
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

    return {
        "trades":        trades,
        "balance":       balance,
        "gap_log":       gap_log,
        "candle_data":   candle_data,
        "n_trades":      len(trades),
        "win_rate":      len(_wins) / len(trades) if trades else 0,
        "avg_r":         sum(t["r"] for t in trades) / len(trades) if trades else 0,
        "best_r":        max((t["r"] for t in trades), default=0),
        "avg_r_win":     sum(t["r"] for t in _wins)   / len(_wins)   if _wins   else 0,
        "avg_r_loss":    sum(t["r"] for t in _losses) / len(_losses) if _losses else 0,
        "n_target":              _ec.get("target_reached", 0) + _ec.get("weekend_proximity", 0),
        "n_ratchet":             _ec.get("ratchet_stop_hit", 0),
        "n_sl":                  _ec.get("stop_hit", 0),
        "exec_rr_p50":           _rrs[len(_rrs) // 2] if _rrs else 0,
        "max_dd":                _max_dd_pct,
        "ret_pct":               (balance - starting_bal) / starting_bal * 100,
        "dd_killswitch_blocks":  dd_killswitch_blocks,
        "max_dollar_loss":       min((t["pnl"] for t in trades), default=0.0),
        "api_calls":             _api_call_count,   # 0 on full cache hit
    }


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
                        help="Trail arm: A | B | C | all  (default: A). "
                             "A=activate1R trail0.5R  B=activate1.5R  C=2-stage 2R+3R")
    parser.add_argument("--cache", action="store_true", default=False,
                        help="No-op (kept for backward compat). Cache is ON by default. "
                             "Use --no-cache to disable.")
    parser.add_argument("--no-cache", action="store_true", default=False,
                        help="Disable per-pair disk cache — always fetch from OANDA. "
                             f"Cache dir: {CACHE_DIR}")
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

    # Build notes string that captures any lever overrides
    notes = args.notes
    extra = []
    if args.news_filter:
        extra.append("news_filter=on")
    if args.profile:
        extra.append(f"profile={args.profile}")
    if args.lever:
        extra.extend(args.lever)
    if extra:
        notes = (notes + " [levers: " + ", ".join(extra) + "]").strip()

    # ── Validate --arm ────────────────────────────────────────────────────────
    _arm_arg = args.arm.upper()
    if _arm_arg not in ("A", "B", "C", "ALL"):
        parser.error(f"--arm must be A | B | C | all, got '{args.arm}'")
    _arms_to_run = list(TRAIL_ARMS.keys()) if _arm_arg == "ALL" else [_arm_arg]

    # ── Cache mode ────────────────────────────────────────────────────────────
    # Cache is ON by default (per-pair/TF disk cache in ~/.cache/forge_backtester/).
    # --no-cache disables disk reads/writes and always fetches from OANDA.
    # --cache is a legacy no-op kept for backward compat.
    _use_cache = not getattr(args, "no_cache", False)
    if not _use_cache:
        print("  ⚠ Cache disabled (--no-cache) — all data fetched from OANDA")

    # ── Run arms ──────────────────────────────────────────────────────────────
    # For multi-arm runs, candle data is fetched once then shared in-memory
    # across subsequent arms via preloaded_candle_data — no redundant I/O.
    _arm_results: dict = {}
    _shared_candles: Optional[dict] = None   # populated after first arm's fetch

    for _arm_key in _arms_to_run:
        _tcfg      = TRAIL_ARMS[_arm_key]
        _arm_notes = f"{notes} [arm={_arm_key}]".strip()

        result = run_backtest(
            start_dt=start, end_dt=end, starting_bal=args.balance,
            notes=_arm_notes, trail_cfg=_tcfg,
            preloaded_candle_data=_shared_candles,
            use_cache=_use_cache,
        )

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
