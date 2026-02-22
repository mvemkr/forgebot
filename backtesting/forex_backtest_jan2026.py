"""
Set & Forget Forex Backtest — Jan 1, 2026 → Feb 21, 2026

Uses actual historical OHLC data (yfinance) and the real strategy code.
Simulates week-by-week setup selection and day-by-day trade management.

Exit rules (mirroring position_monitor.py):
  - Hard stop at stop_loss level
  - Move SL to breakeven at 1:1
  - Exit on daily reversal signal (engulfing/pin bar against trade, strength > 0.5)
  - Max hold: 25 trading days (one month cap)

Starting capital: $4,000 (Mike's planned OANDA funding)
Risk scaling: 5% (balance < $8K), auto-scales with balance
"""
import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parents[1]))

import yfinance as yf
from src.strategy.forex.pattern_detector import PatternDetector, Trend
from src.strategy.forex.level_detector import LevelDetector
from src.strategy.forex.entry_signal import EntrySignalDetector

# ── Config ─────────────────────────────────────────────────────────────
START_DATE     = "2025-10-01"   # fetch extra history for indicator warmup
BACKTEST_START = datetime(2026, 1, 1)
BACKTEST_END   = datetime(2026, 2, 21)
STARTING_BAL   = 4_000.0

WATCHLIST = {
    "USD/JPY": "JPY=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/CHF": "CHF=X",
    "USD/CAD": "CAD=X",
    "GBP/JPY": "GBPJPY=X",
    "GBP/CHF": "GBPCHF=X",
    "GBP/NZD": "GBPNZD=X",
    "NZD/USD": "NZDUSD=X",
    "AUD/USD": "AUDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "NZD/JPY": "NZDJPY=X",
}

RISK_TIERS = [
    (8_000,       5.0),
    (15_000,     10.0),
    (30_000,     15.0),
    (float("inf"), 20.0),
]

MIN_LEVEL_SCORE     = 2
MIN_PATTERN_CLARITY = 0.35
MIN_SIGNAL_STRENGTH = 0.40
MAX_HOLD_DAYS       = 25
BREAKEVEN_AT_1R     = True
EXIT_SIGNAL_MIN     = 0.50

# ── Helpers ─────────────────────────────────────────────────────────────

def get_risk_pct(balance):
    for max_bal, pct in RISK_TIERS:
        if balance < max_bal:
            return pct
    return 20.0

def is_jpy(pair):
    return "JPY" in pair

def pip_mult(pair):
    return 100 if is_jpy(pair) else 10000

def calc_units(balance, risk_pct, stop_pips):
    risk_dollars = balance * (risk_pct / 100)
    pip_val = 0.01 if True else 0.0001   # simplified: $0.01/unit/pip for majors
    units = int(risk_dollars / max(stop_pips * 0.0001, 0.0001))
    return max(1000, min(units, 100_000))

def pnl_dollars(pair, direction, entry, exit_price, units):
    if direction == "long":
        delta = exit_price - entry
    else:
        delta = entry - exit_price
    # Simplified P&L: delta * units (works for USD-quoted pairs)
    # For JPY pairs: delta is in JPY, divide by exit rate — simplified here
    if is_jpy(pair):
        return round(delta * units / exit_price, 2)
    return round(delta * units, 2)

# ── Data Fetching ───────────────────────────────────────────────────────

print("Fetching historical Forex data from yfinance...")
data = {}
failed = []
for pair, ticker in WATCHLIST.items():
    try:
        df = yf.download(ticker, start=START_DATE, end="2026-02-22",
                         interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            failed.append(pair)
            continue
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        data[pair] = df
        print(f"  ✓ {pair}: {len(df)} days")
    except Exception as e:
        failed.append(pair)
        print(f"  ✗ {pair}: {e}")

if failed:
    print(f"\n  Skipped (no data): {failed}")

if not data:
    print("No data fetched — check internet connection")
    sys.exit(1)

# ── Strategy Components ──────────────────────────────────────────────────
pattern_det = PatternDetector(swing_window=5, tolerance_pct=0.3)
level_det   = LevelDetector(min_confluence=MIN_LEVEL_SCORE)
signal_det  = EntrySignalDetector(min_body_ratio=0.45, lookback_candles=2)

# ── Backtest Engine ──────────────────────────────────────────────────────
balance  = STARTING_BAL
trades   = []
open_pos = {}   # pair → position dict
be_moved = set()

# Trading days in backtest window
all_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq="B")

print(f"\nRunning backtest: {BACKTEST_START.date()} → {BACKTEST_END.date()}")
print(f"Pairs: {len(data)} | Starting balance: ${balance:,.2f}\n")

def get_trend(df_slice):
    if len(df_slice) < 20:
        return Trend.NEUTRAL
    return pattern_det.detect_trend(df_slice)

def evaluate_entry(pair, df_history, current_date):
    """
    Check if this pair has a valid entry signal on current_date.
    Uses df_history = all data up to (not including) current_date.
    """
    if len(df_history) < 60:
        return None

    current_price = df_history["close"].iloc[-1]

    # Trend alignment (use last 60/30/15 bars as weekly/daily/4H proxy)
    trend_w = get_trend(df_history.iloc[-60:])
    trend_d = get_trend(df_history.iloc[-30:])
    trend_4 = get_trend(df_history.iloc[-15:])

    bullish = sum(1 for t in [trend_w, trend_d, trend_4]
                  if t in (Trend.BULLISH, Trend.STRONG_BULLISH))
    bearish = sum(1 for t in [trend_w, trend_d, trend_4]
                  if t in (Trend.BEARISH, Trend.STRONG_BEARISH))

    if bullish >= 2:
        direction = "long"
    elif bearish >= 2:
        direction = "short"
    else:
        return None  # no trend alignment

    # Counter-trend block: weekly must agree
    if direction == "long" and trend_w in (Trend.BEARISH, Trend.STRONG_BEARISH):
        return None
    if direction == "short" and trend_w in (Trend.BULLISH, Trend.STRONG_BULLISH):
        return None

    # Key levels
    levels = level_det.detect(df_history.iloc[-100:], current_price)
    matching = [l for l in levels
                if l.distance_pct <= 0.5
                and l.score >= MIN_LEVEL_SCORE
                and ((direction == "long" and l.level_type == "support")
                     or (direction == "short" and l.level_type == "resistance"))]
    if not matching:
        return None
    level = matching[0]

    # Pattern
    patterns = pattern_det.detect_all(df_history.iloc[-60:])
    exp_dir = "bullish" if direction == "long" else "bearish"
    matching_pat = next(
        (p for p in patterns if p.direction == exp_dir and p.clarity >= MIN_PATTERN_CLARITY),
        None
    )
    if not matching_pat:
        return None

    # Entry signal (engulfing candle on last 3 days)
    has_signal, signal = signal_det.has_signal(df_history.iloc[-5:], direction)
    if not has_signal or signal is None or signal.strength < MIN_SIGNAL_STRENGTH:
        return None

    # Stop loss
    if direction == "short":
        stop_loss = max(matching_pat.stop_loss, level.price * 1.003)
    else:
        stop_loss = min(matching_pat.stop_loss, level.price * 0.997)

    stop_pips = abs(current_price - stop_loss) * pip_mult(pair)
    if stop_pips < 5:   # too tight
        return None

    return {
        "direction":      direction,
        "entry_price":    current_price,
        "stop_loss":      stop_loss,
        "stop_pips":      stop_pips,
        "pattern":        matching_pat.pattern_type,
        "pattern_clarity": matching_pat.clarity,
        "signal_strength": signal.strength,
        "level_score":    level.score,
        "trend_weekly":   trend_w.value,
        "trend_daily":    trend_d.value,
        "trend_4h":       trend_4.value,
        "confidence":     (
            0.3 * min(1.0, level.score / 4)
            + 0.3 * matching_pat.clarity
            + 0.25 * signal.strength
            + 0.15
        ),
    }


for current_date in all_dates:
    current_str = current_date.strftime("%Y-%m-%d")

    # ── Monitor open positions ──────────────────────────────────────
    for pair in list(open_pos.keys()):
        pos = open_pos[pair]
        df_pair = data.get(pair)
        if df_pair is None:
            continue

        # Today's candle
        day_data = df_pair[df_pair.index <= current_date]
        if len(day_data) == 0:
            continue
        today = day_data.iloc[-1]
        high  = today["high"]
        low   = today["low"]
        close = today["close"]

        entry     = pos["entry_price"]
        stop      = pos["stop_loss"]
        direction = pos["direction"]
        risk      = abs(entry - stop)
        pos_key   = f"{pair}_{pos['entry_date']}"

        # Stop hit?
        stop_hit = (direction == "long" and low <= stop) or \
                   (direction == "short" and high >= stop)

        if stop_hit:
            exit_price = stop
            pnl = pnl_dollars(pair, direction, entry, exit_price, pos["units"])
            balance += pnl
            trades.append({**pos,
                "exit_date": current_str,
                "exit_price": exit_price,
                "exit_reason": "stop_hit",
                "pnl": pnl,
                "rr": -1.0,
                "bars_held": (current_date - pd.Timestamp(pos["entry_date"])).days,
            })
            del open_pos[pair]
            be_moved.discard(pos_key)
            continue

        # Breakeven move at 1:1
        if BREAKEVEN_AT_1R and pos_key not in be_moved:
            moved_1r = (direction == "long" and close >= entry + risk) or \
                       (direction == "short" and close <= entry - risk)
            if moved_1r:
                pos["stop_loss"] = entry
                be_moved.add(pos_key)

        # Exit signal: reversal candle against trade
        recent = day_data.iloc[-5:]
        opp_dir = "short" if direction == "long" else "long"
        has_exit, exit_signal = signal_det.has_signal(recent, opp_dir)
        if has_exit and exit_signal and exit_signal.strength >= EXIT_SIGNAL_MIN:
            exit_price = close
            pnl = pnl_dollars(pair, direction, entry, exit_price, pos["units"])
            rr = (abs(exit_price - entry) / risk) * (1 if pnl > 0 else -1)
            balance += pnl
            trades.append({**pos,
                "exit_date": current_str,
                "exit_price": exit_price,
                "exit_reason": "exit_signal",
                "pnl": pnl,
                "rr": round(rr, 2),
                "bars_held": (current_date - pd.Timestamp(pos["entry_date"])).days,
            })
            del open_pos[pair]
            be_moved.discard(pos_key)
            continue

        # Max hold
        bars_held = (current_date - pd.Timestamp(pos["entry_date"])).days
        if bars_held >= MAX_HOLD_DAYS:
            exit_price = close
            pnl = pnl_dollars(pair, direction, entry, exit_price, pos["units"])
            rr = (abs(exit_price - entry) / risk) * (1 if pnl > 0 else -1)
            balance += pnl
            trades.append({**pos,
                "exit_date": current_str,
                "exit_price": exit_price,
                "exit_reason": "max_hold",
                "pnl": pnl,
                "rr": round(rr, 2),
                "bars_held": bars_held,
            })
            del open_pos[pair]
            be_moved.discard(pos_key)

    # ── Look for new entries (only if no open position) ─────────────
    if open_pos:
        continue   # one trade at a time

    for pair, df_pair in data.items():
        df_history = df_pair[df_pair.index < current_date]
        if len(df_history) < 60:
            continue

        entry_data = evaluate_entry(pair, df_history, current_date)
        if not entry_data:
            continue

        risk_pct  = get_risk_pct(balance)
        stop_pips = entry_data["stop_pips"]
        units     = calc_units(balance, risk_pct, stop_pips)
        risk_dollars = balance * (risk_pct / 100)

        pos = {
            **entry_data,
            "pair":         pair,
            "entry_date":   current_str,
            "risk_pct":     risk_pct,
            "risk_dollars": risk_dollars,
            "units":        units,
        }
        open_pos[pair] = pos
        break   # one trade at a time — take the first qualifying setup

# Close any still-open positions at the last available price
for pair, pos in open_pos.items():
    df_pair = data.get(pair)
    if df_pair is None:
        continue
    last_price = df_pair[df_pair.index <= BACKTEST_END]["close"].iloc[-1]
    entry = pos["entry_price"]
    stop  = pos["stop_loss"]
    risk  = abs(entry - stop)
    direction = pos["direction"]
    pnl = pnl_dollars(pair, direction, entry, last_price, pos["units"])
    rr = (abs(last_price - entry) / risk) * (1 if pnl > 0 else -1) if risk > 0 else 0
    balance += pnl
    trades.append({**pos,
        "exit_date": BACKTEST_END.strftime("%Y-%m-%d"),
        "exit_price": last_price,
        "exit_reason": "still_open_at_end",
        "pnl": pnl,
        "rr": round(rr, 2),
        "bars_held": (BACKTEST_END - pd.Timestamp(pos["entry_date"])).days,
    })

# ── Results ─────────────────────────────────────────────────────────────
print("=" * 65)
print(f"BACKTEST RESULTS — Jan 1 → Feb 21, 2026 (~7.5 weeks)")
print(f"Strategy: Set & Forget | Starting capital: ${STARTING_BAL:,.2f}")
print("=" * 65)

if not trades:
    print("\nNo trades taken in this period.")
    print("This is actually normal — the strategy is highly selective.")
    print("Alex typically took 1-2 trades per week maximum.")
    sys.exit(0)

wins   = [t for t in trades if t["pnl"] > 0]
losses = [t for t in trades if t["pnl"] <= 0]
total_pnl = sum(t["pnl"] for t in trades)
win_rate  = len(wins) / len(trades) * 100

print(f"\n📊 TRADE SUMMARY")
print(f"  Total trades:     {len(trades)}")
print(f"  Wins / Losses:    {len(wins)} / {len(losses)}")
print(f"  Win rate:         {win_rate:.0f}%")

if wins:
    avg_rr_win = np.mean([t["rr"] for t in wins])
    print(f"  Avg R on wins:    +{avg_rr_win:.1f}R")
if losses:
    avg_rr_loss = np.mean([t["rr"] for t in losses])
    print(f"  Avg R on losses:  {avg_rr_loss:.1f}R")

print(f"\n💰 P&L SUMMARY")
print(f"  Starting balance: ${STARTING_BAL:>10,.2f}")
print(f"  Total P&L:        ${total_pnl:>+10,.2f}")
print(f"  Final balance:    ${balance:>10,.2f}")
print(f"  Return:           {(balance/STARTING_BAL - 1)*100:>+.1f}%")

print(f"\n📋 INDIVIDUAL TRADES")
print(f"  {'#':<3} {'Date':<12} {'Pair':<10} {'Dir':<6} {'Entry':>9} {'Exit':>9} {'R:R':>6} {'P&L':>10} {'Reason':<18}")
print(f"  {'-'*90}")
for i, t in enumerate(trades, 1):
    rr_str = f"{t['rr']:+.1f}R" if t['rr'] is not None else "?"
    pnl_str = f"${t['pnl']:+,.2f}"
    print(
        f"  {i:<3} {t['entry_date']:<12} {t['pair']:<10} {t['direction']:<6} "
        f"{t['entry_price']:>9.4f} {t['exit_price']:>9.4f} "
        f"{rr_str:>6} {pnl_str:>10} {t['exit_reason']:<18}"
        f"  [{t.get('pattern','?')[:20]}]"
    )

print(f"\n🔍 WHAT DROVE THE RESULTS")
by_reason = {}
for t in trades:
    r = t["exit_reason"]
    by_reason.setdefault(r, []).append(t["pnl"])
for reason, pnls in by_reason.items():
    print(f"  {reason:<22} {len(pnls)} trades | Total: ${sum(pnls):+,.2f} | Avg: ${np.mean(pnls):+,.2f}")

print(f"\n⚠️  NOTES")
print(f"  • Daily candles used (no intraday — slightly less precise entries)")
print(f"  • Exit signal: daily reversal candle, strength ≥ {EXIT_SIGNAL_MIN}")
print(f"  • Breakeven: moved to entry after 1:1 profit")
print(f"  • Max hold: {MAX_HOLD_DAYS} days")
print(f"  • One trade at a time enforced")
print(f"  • Counter-trend trades blocked (weekly trend must agree)")
print()
