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
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List

# ── Real strategy imports ─────────────────────────────────────────────────────
from src.exchange.oanda_client import OandaClient, INSTRUMENT_MAP
from src.strategy.forex.set_and_forget   import SetAndForgetStrategy, Decision
from src.strategy.forex.news_filter      import NewsFilter
from src.execution.risk_manager_forex    import ForexRiskManager
from src.execution.trade_journal         import TradeJournal
from src.strategy.forex.currency_strength import CurrencyStrengthAnalyzer, CurrencyTheme, STACK_MAX

# ── Shared execution config (single source of truth) ─────────────────────────
# These constants are also used by the live orchestrator. Change them here,
# they apply everywhere. Do NOT hardcode these values anywhere else.
from src.strategy.forex.strategy_config import (
    MIN_CONFIDENCE,
    MIN_RR,
    ATR_STOP_MULTIPLIER,
    ATR_MIN_MULTIPLIER,
    ATR_LOOKBACK,
    MAX_CONCURRENT_TRADES,
    LONDON_SESSION_START_UTC,
    LONDON_SESSION_END_UTC,
    STOP_COOLDOWN_DAYS,
    progressive_confluence_check,
    NECKLINE_CLUSTER_PCT,
    DRY_RUN_PAPER_BALANCE,
)

# ── Backtest-only config ──────────────────────────────────────────────────────
BACKTEST_START  = datetime(2025, 7, 1, tzinfo=timezone.utc)   # ~7 months of history
STARTING_BAL    = 8_000.0
MAX_HOLD_BARS   = 365 * 24         # effectively no cap — strategy has no TP or hold limit
                                   # (live bot runs until stop hit or Mike manually closes)
GAP_LOG_PATH    = Path.home() / "trading-bot" / "logs" / "backtest_gap_log.jsonl"
DECISION_LOG    = Path.home() / "trading-bot" / "logs" / "backtest_v2_decisions.json"
V1_DECISION_LOG = Path.home() / "trading-bot" / "logs" / "backtest_decisions.json"

WATCHLIST = [
    # USD-based majors
    "GBP/USD", "EUR/USD", "USD/JPY", "USD/CHF", "USD/CAD", "NZD/USD", "AUD/USD",
    # GBP crosses
    "GBP/JPY", "GBP/CHF", "GBP/NZD",
    # EUR crosses (no USD/GBP — survive GBP+USD lockout)
    "EUR/GBP", "EUR/AUD", "EUR/NZD", "EUR/CAD",
    # Commodity crosses (no USD/GBP/JPY/CHF — survive most lockout scenarios)
    "AUD/CAD", "AUD/NZD", "NZD/CAD", "NZD/JPY",
]

# ── Disable-news-filter sentinel ──────────────────────────────────────────────
class _NoOpNewsFilter(NewsFilter):
    """News filter that never blocks — used in backtest since historical
    ForexFactory data is unavailable. Gap logged separately."""
    def is_entry_blocked(self, dt_utc, post_news_candle=None):
        return False, ""
    def is_news_candle(self, candle_dt_utc):
        return False
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


# ── Main backtest ─────────────────────────────────────────────────────────────
def run_backtest(start_dt: datetime = BACKTEST_START, end_dt: datetime = None, starting_bal: float = STARTING_BAL):
    end_naive = end_dt.replace(tzinfo=None) if end_dt else None
    print(f"\n{'='*65}")
    print(f"OANDA 1H BACKTEST v2 — Real Strategy Code")
    print(f"Start: {start_dt.date()}  |  End: {end_dt.date() if end_dt else 'today'}  |  Capital: ${starting_bal:,.2f}")
    print(f"{'='*65}")

    # ── Fetch candle data ─────────────────────────────────────────────
    print("\nFetching OANDA historical candles...")
    candle_data = {}
    # Use date-range fetch when start_dt is more than 200 days ago (beyond count=5000 H1 window)
    # Always fetch 6 months of daily context before start for pattern/trend detection
    data_start = (start_dt - pd.Timedelta(days=180)).replace(tzinfo=timezone.utc)
    data_end   = end_naive   # may be None for live use
    use_range  = (datetime.now(tz=timezone.utc) - start_dt).days > 200

    for pair in WATCHLIST:
        if use_range:
            # Paginated date-range fetch — handles any historical window
            df_1h = _fetch_range(pair, "H1", from_dt=data_start, to_dt=end_dt)
            df_4h = _fetch_range(pair, "H4", from_dt=data_start, to_dt=end_dt)
            df_d  = _fetch_range(pair, "D",  from_dt=(start_dt - pd.Timedelta(days=730)).replace(tzinfo=timezone.utc), to_dt=end_dt)
        else:
            df_1h = _fetch(pair, "H1", 5000)   # ~208 days of hourly bars
            df_4h = _fetch(pair, "H4", 1500)   # ~250 days of 4H bars
            df_d  = _fetch(pair, "D",  500)    # ~500 trading days (~2 years)
        if df_1h is None or len(df_1h) < 50:
            print(f"  ✗ {pair}: insufficient data, skipping")
            continue
        df_w = _resample_weekly(df_d) if df_d is not None else None
        candle_data[pair] = {"1h": df_1h, "4h": df_4h, "d": df_d, "w": df_w}
        time.sleep(0.3)   # OANDA rate limit
        print(f"  ✓ {pair}: {len(df_1h)} 1H | {len(df_4h) if df_4h is not None else 0} 4H"
              f" | {len(df_d) if df_d is not None else 0} D"
              f" | {len(df_w) if df_w is not None else 0} W")

    if not candle_data:
        print("No data loaded. Exiting.")
        return

    # Build unified 1H timeline from start_dt
    # Candle timestamps are tz-naive UTC from OandaClient — strip tz from start_dt for comparison
    start_naive = start_dt.replace(tzinfo=None)
    all_1h = sorted(set(
        ts for pdata in candle_data.values()
        for ts in pdata["1h"].index
        if ts >= start_naive and (end_naive is None or ts <= end_naive)
    ))
    print(f"\nPairs loaded: {len(candle_data)}")
    print(f"Backtesting {len(all_1h)} hourly bars: "
          f"{all_1h[0].date()} → {all_1h[-1].date()}")
    print(f"Starting balance: ${starting_bal:,.2f}\n")

    # ── Strategy instances (one per pair for independent state) ───────
    strategies: Dict[str, SetAndForgetStrategy] = {}
    for pair in candle_data:
        s = SetAndForgetStrategy(account_balance=starting_bal, risk_pct=15.0)
        s.news_filter = _NoOpNewsFilter()   # historical news data unavailable
        strategies[pair] = s
        log_gap(start_dt, pair, "N/A", "N/A", "news_filter_skipped",
                "NewsFilter disabled in backtest — historical ForexFactory data unavailable."
                " Live bot may block some of these entries.")

    # ── Risk manager ──────────────────────────────────────────────────
    journal = TradeJournal()
    risk    = ForexRiskManager(journal=journal)

    # ── State ─────────────────────────────────────────────────────────
    balance       = starting_bal
    open_pos: Dict[str, dict] = {}   # pair → position dict
    trades        = []
    all_decisions = []
    v1_decisions  = _load_v1_decisions()

    def _risk_pct(bal):
        return risk.get_risk_pct(bal)

    def _calc_units(pair, bal, rpct, entry, stop):
        pip    = 0.01 if "JPY" in pair else 0.0001
        dist   = abs(entry - stop)
        if dist == 0: return 0
        risk_usd = bal * rpct / 100
        return int(risk_usd / dist)

    def _daily_atr(pair) -> Optional[float]:
        """14-day ATR for the pair (in price terms, not pips)."""
        df_d = candle_data[pair].get("d")
        if df_d is None or len(df_d) < ATR_LOOKBACK + 1:
            return None
        recent = df_d.tail(ATR_LOOKBACK + 1)
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
        Min: ≥ 0.15 × 14-day ATR — lowered from 0.75 to allow Alex's 4H H&S stops.
             Original 0.75× was blocking valid 87-307 pip stops on 200-pip-ATR pairs.
             0.15× still blocks sub-20-pip micro-stops (noise entries) while
             allowing any stop that genuinely places behind a pattern extreme.
             Example: GBP/JPY ATR=214p → min=32p; passes 87p+ H&S stops.
        """
        atr = _daily_atr(pair)
        if atr is None:
            return True
        dist = abs(entry - stop)
        max_dist = atr * ATR_STOP_MULTIPLIER
        min_dist = atr * ATR_MIN_MULTIPLIER
        return min_dist <= dist <= max_dist

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

    def _entry_eligible(pair, macro_theme: Optional[CurrencyTheme] = None):
        """3-layer eligibility gate matching live risk_manager_forex logic.

        Macro theme exception: when a currency theme is active and this pair
        is one of the suggested stacked trades, we:
          (a) raise the concurrent cap to STACK_MAX (default 4), and
          (b) waive the currency-overlap check — correlated exposure is intentional.
        This is exactly how Alex stacked 4 JPY shorts in Week 7-8 for $70K.
        """
        _is_theme = (
            macro_theme is not None
            and pair in [p for p, _ in macro_theme.suggested_trades]
        )
        max_concurrent = STACK_MAX if _is_theme else MAX_CONCURRENT_TRADES

        # Layer 1: max concurrent
        if len(open_pos) >= max_concurrent:
            return False, "max_concurrent"
        # Layer 2: currency overlap (waived for macro theme pairs — intentionally correlated)
        if not _is_theme and (_pair_currencies(pair) & _currencies_in_use()):
            return False, "currency_overlap"
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

            # Breakeven at 1:1
            risk_dist = abs(entry - stop)
            if not pos.get("be_moved"):
                if direction == "long"  and high  >= entry + risk_dist:
                    stop = entry; pos["stop_loss"] = stop; pos["be_moved"] = True
                if direction == "short" and low   <= entry - risk_dist:
                    stop = entry; pos["stop_loss"] = stop; pos["be_moved"] = True

            # Stop hit?
            stopped = (direction == "long"  and low  <= stop) or \
                      (direction == "short" and high >= stop)
            max_hold = bars_held >= MAX_HOLD_BARS

            if stopped or max_hold:
                exit_p   = stop if stopped else close
                delta    = (exit_p - entry) if direction == "long" else (entry - exit_p)
                pnl      = delta * units
                balance += pnl
                reason   = "stop_hit" if stopped else "max_hold"
                risk_r   = delta / risk_dist if risk_dist else 0

                trades.append({
                    "pair": pair, "direction": direction,
                    "entry": entry, "exit": exit_p,
                    "pnl": pnl, "r": risk_r, "reason": reason,
                    "entry_ts": pos["entry_ts"].isoformat(),
                    "exit_ts": ts_utc.isoformat(),
                    "bars_held": bars_held,
                    "pattern": pos.get("pattern", "?"),
                    "notes": pos.get("notes", ""),
                    "macro_theme": pos.get("macro_theme"),
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
        # Macro theme: computed once per calendar day (daily data doesn't change intraday)
        active_theme = _get_theme(ts)

        for pair, pdata in candle_data.items():
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

            # Slice history up to (not including) current bar
            hist_1h = df_1h_p[df_1h_p.index < ts]
            hist_4h = df_4h_p[df_4h_p.index < ts] if df_4h_p is not None else pd.DataFrame()
            hist_d  = df_d_p[df_d_p.index < ts]   if df_d_p  is not None else pd.DataFrame()
            hist_w  = df_w_p[df_w_p.index < ts]   if df_w_p  is not None else pd.DataFrame()

            if len(hist_1h) < 20 or len(hist_4h) < 10:
                continue

            # Update strategy risk tier for current balance
            strat = strategies[pair]
            strat.account_balance = balance
            strat.risk_pct = _risk_pct(balance)

            # ── Run real strategy evaluation ──────────────────────────
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
            except Exception as e:
                continue

            dec_record = {
                "ts":         ts_utc.isoformat(),
                "pair":       pair,
                "decision":   decision.decision.value,
                "confidence": decision.confidence,
                "reason":     decision.reason,
                "direction":  decision.direction,
                "entry_price": decision.entry_price,
                "stop_loss":  decision.stop_loss,
                "filters_failed": decision.failed_filters,
                "balance":    balance,
            }
            all_decisions.append(dec_record)

            # ── Gap analysis vs v1 ────────────────────────────────────
            v1_key  = (pair, ts_utc.isoformat())
            v1_dec  = v1_decisions.get(v1_key)
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

            # ── Macro theme direction gate ────────────────────────────
            # If a macro theme is active and this pair is one of its suggested
            # trades, block any entry whose direction CONTRADICTS the theme.
            # E.g. GBP_weak theme → GBP/JPY SHORT is the play.
            #      If the pattern detector finds a bullish IH&S and wants to go
            #      LONG, that directly opposes the macro view → skip it.
            # We only enter if the pattern AGREES with (or is neutral to) the theme.
            if (decision.decision == Decision.ENTER
                    and _is_theme_pair and active_theme):
                theme_dir_map = dict(active_theme.suggested_trades)
                theme_dir     = theme_dir_map.get(pair)
                if theme_dir and theme_dir != decision.direction:
                    log_gap(ts_utc, pair, "ENTER", "BLOCKED",
                            "theme_direction_conflict",
                            f"Theme={active_theme.currency}_{active_theme.direction} "
                            f"wants {theme_dir}, pattern wants {decision.direction}. "
                            f"Blocked to avoid contradicting macro view.")
                    continue

            # ── Confidence gate ───────────────────────────────────────
            if (decision.decision == Decision.ENTER
                    and decision.confidence < MIN_CONFIDENCE):
                log_gap(ts_utc, pair, "ENTER", "BLOCKED",
                        "low_confidence",
                        f"Confidence {decision.confidence:.0%} < {MIN_CONFIDENCE:.0%} threshold  "
                        f"entry={decision.entry_price:.5f}  reason={decision.reason[:60]}")
                continue  # skip silently (high-frequency; print only on debug)

            # ── Progressive confluence gate ───────────────────────────
            # Second trade must clear a higher bar: 75%+ confidence,
            # structural pattern required (no break-retests as 2nd entry).
            if decision.decision == Decision.ENTER:
                pattern_type = (
                    decision.pattern.pattern_type if decision.pattern else ""
                )
                prog_blocked, prog_reason = progressive_confluence_check(
                    n_open=len(open_pos),
                    confidence=decision.confidence,
                    pattern_type=pattern_type,
                )
                if prog_blocked:
                    log_gap(ts_utc, pair, "ENTER", "BLOCKED",
                            "progressive_confluence", prog_reason)
                    continue

            # ── Stop-distance guard ───────────────────────────────────
            if (decision.decision == Decision.ENTER
                    and decision.entry_price and decision.stop_loss
                    and not _stop_ok(pair, decision.entry_price, decision.stop_loss)):
                atr  = _daily_atr(pair)
                dist = abs(decision.entry_price - decision.stop_loss)
                pip  = 0.01 if "JPY" in pair else 0.0001
                too_wide  = atr is not None and dist > atr * ATR_STOP_MULTIPLIER
                gap_type  = "stop_too_wide" if too_wide else "stop_too_tight"
                if too_wide:
                    msg = (f"Stop too wide: {dist/pip:.0f}p > max {atr*ATR_STOP_MULTIPLIER/pip:.0f}p "
                           f"({ATR_STOP_MULTIPLIER:.0f}×ATR={atr/pip:.0f}p)")
                else:
                    msg = (f"Stop too tight: {dist/pip:.0f}p < min {atr*ATR_MIN_MULTIPLIER/pip:.0f}p "
                           f"({ATR_MIN_MULTIPLIER:.2f}×ATR={atr/pip:.0f}p) — micro-stop, noise will hit it")
                log_gap(ts_utc, pair, "ENTER", "BLOCKED", gap_type,
                        f"{msg}  entry={decision.entry_price:.5f}  sl={decision.stop_loss:.5f}")
                print(f"  ⚠ {ts_utc.strftime('%Y-%m-%d %H:%M')} | SKIP {pair} — {msg}  sl={decision.stop_loss:.5f}")
                continue

            # ── Execute entry ─────────────────────────────────────────
            if (decision.decision == Decision.ENTER
                    and decision.entry_price
                    and decision.stop_loss):

                rpct  = _risk_pct(balance)
                # Macro theme stacking: each position gets fraction of normal risk
                # so TOTAL exposure across all stacked trades = one normal trade.
                # This is how Alex ran 4 JPY shorts at 1/4 size each = same $ risk as 1 trade.
                if _is_theme_pair and active_theme:
                    rpct = rpct * active_theme.position_fraction
                units = _calc_units(pair, balance, rpct,
                                    decision.entry_price, decision.stop_loss)
                if units <= 0:
                    continue

                risk_dist = abs(decision.entry_price - decision.stop_loss)
                risk_usd    = balance * rpct / 100

                open_pos[pair] = {
                    "entry_price": decision.entry_price,
                    "stop_loss":   decision.stop_loss,
                    "direction":   decision.direction,
                    "units":       units,
                    "bar_idx":     bar_idx,
                    "entry_ts":    ts_utc,
                    "pattern":     decision.pattern.pattern_type if decision.pattern else "?",
                    "notes":       decision.reason[:80],
                    "be_moved":    False,
                    "macro_theme": f"{active_theme.currency}_{active_theme.direction}" if _is_theme_pair and active_theme else None,
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
                      f"  conf={decision.confidence:.0%}"
                      f"  [{decision.reason[:60]}]{theme_tag}")

    # ── Close any still-open positions at last bar price ──────────────
    last_ts = all_1h[-1] if all_1h else None
    for pair, pos in list(open_pos.items()):
        if pair not in candle_data: continue
        df_1h_p = candle_data[pair]["1h"]
        close_p = float(df_1h_p["close"].iloc[-1])
        entry   = pos["entry_price"]
        stop    = pos["stop_loss"]
        direction = pos["direction"]
        units   = pos["units"]
        delta   = (close_p - entry) if direction == "long" else (entry - close_p)
        pnl     = delta * units
        r       = delta / abs(entry - stop) if abs(entry - stop) else 0
        balance += pnl
        trades.append({
            "pair": pair, "direction": direction,
            "entry": entry, "exit": close_p,
            "pnl": pnl, "r": r, "reason": "open_at_end",
            "entry_ts": pos["entry_ts"].isoformat(),
            "exit_ts":  last_ts.isoformat() if last_ts else "",
            "bars_held": len(all_1h) - pos["bar_idx"],
            "pattern": pos.get("pattern", "?"),
            "notes": pos.get("notes", ""),
            "macro_theme": pos.get("macro_theme"),
        })

    # ── Results ───────────────────────────────────────────────────────
    net_pnl  = balance - starting_bal
    ret_pct  = net_pnl / starting_bal * 100
    wins     = [t for t in trades if t["pnl"] > 0]
    losses   = [t for t in trades if t["pnl"] <= 0]
    wr       = len(wins) / len(trades) * 100 if trades else 0

    print(f"\n{'='*65}")
    print(f"RESULTS — v2 (Real Strategy Code)")
    print(f"{'='*65}")
    print(f"  Trades:       {len(trades)}  ({len(wins)} wins / {len(losses)} losses)")
    print(f"  Win rate:     {wr:.0f}%")
    print(f"  Starting:     ${starting_bal:>10,.2f}")
    print(f"  Net P&L:      ${net_pnl:>+10,.2f}")
    print(f"  Final:        ${balance:>10,.2f}")
    print(f"  Return:       {ret_pct:>+.1f}%")

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

    return trades, balance, gap_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OANDA Backtest v2 — Real Strategy Code")
    parser.add_argument("--start",   default="2025-07-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=None,         help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--balance", type=float, default=STARTING_BAL, help="Starting balance")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end else None
    run_backtest(start_dt=start, end_dt=end, starting_bal=args.balance)
