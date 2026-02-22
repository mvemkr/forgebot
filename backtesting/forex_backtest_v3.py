"""
Set & Forget Forex Backtest — v3 CORRECT ARCHITECTURE
Jan 1, 2026 → Feb 21, 2026

CRITICAL: This is PATTERN-FIRST, not trend-first.
The trader's edge comes from counter-trend REVERSALS at psychological levels.

Correct decision flow:
  1. Find major psychological levels (round numbers: 1.3500, 157.00, 1.1750, etc.)
  2. Is there an H&S or double top/bottom pattern AT that level?
  3. Has the neckline broken and retested?
  4. 1H engulfing candle on the retest → ENTER
  5. Stop behind the pattern's structural extreme
  6. Hold until daily candle body closes AGAINST trade at a major level → exit
     Otherwise: set and forget, let it run

NOT trend-following. Pattern quality at psych level > trend direction.
Counter-trend trades at psych levels are valid (Alex's best trades were these).

Verified v3 trades from prior session:
  - Jan 20: GBP/USD LONG, double bottom at 1.3350, entry 1.34403 → exit 1.37479 (+$300)
  - Feb 17: EUR/USD LONG, double bottom at 1.1750, stopped out (-$100)
  Net: +$200 (+4%) on $5,000
"""
import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parents[1]))

import yfinance as yf
from src.strategy.forex.entry_signal import EntrySignalDetector

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
    (8_000,        5.0),
    (15_000,      10.0),
    (30_000,      15.0),
    (float("inf"), 20.0),
]

# ── v3 Tuning ──────────────────────────────────────────────────────────
# Pattern detection
MIN_DOUBLE_PATTERN_SIMILARITY = 0.008   # tops/bottoms must be within 0.8% of each other
MIN_PATTERN_BARS              = 5       # min bars between the two tops/bottoms
MAX_PATTERN_BARS              = 40      # max bars between the two tops/bottoms
PSYCH_LEVEL_TOLERANCE         = 0.008  # price must be within 0.8% of a round number

# Entry
MIN_SIGNAL_STRENGTH = 0.40

# Exit (set & forget — be generous, only exit on strong evidence)
EXIT_SIGNAL_STRENGTH = 0.65    # higher threshold = hold longer (Alex held through noise)
EXIT_ONLY_AT_MAJOR_LEVEL = True # only exit if strong candle AND price at a major level
MAX_HOLD_DAYS = 30

# ── Helpers ────────────────────────────────────────────────────────────

def get_risk_pct(balance):
    for max_bal, pct in RISK_TIERS:
        if balance < max_bal:
            return pct
    return 20.0

def is_jpy(pair):
    return "JPY" in pair

def pip_size(pair):
    return 0.01 if is_jpy(pair) else 0.0001

def calc_units(balance, risk_pct, entry, stop):
    risk_dollars = balance * (risk_pct / 100)
    stop_distance = abs(entry - stop)
    if stop_distance == 0:
        return 1000
    # units = risk_dollars / stop_distance (for USD-quoted pairs)
    if is_jpy(str(entry)):
        units = int(risk_dollars / (stop_distance / entry))
    else:
        units = int(risk_dollars / stop_distance)
    return max(1000, min(units, 100_000))

def pnl_dollars(pair, direction, entry, exit_price, units):
    delta = (exit_price - entry) if direction == "long" else (entry - exit_price)
    if is_jpy(pair):
        return round(delta * units / exit_price, 2)
    return round(delta * units, 2)

def get_round_levels(price, tolerance=PSYCH_LEVEL_TOLERANCE):
    """
    Find nearby psychological round number levels.
    For most pairs: every 0.0100, 0.0050
    For JPY pairs: every 1.00, 0.50
    Returns list of round prices near current price.
    """
    levels = []
    if price > 50:  # JPY pairs
        increments = [1.0, 0.5]
    else:
        increments = [0.0100, 0.0050]

    for inc in increments:
        rounded = round(round(price / inc) * inc, 5)
        if abs(rounded - price) / price <= tolerance:
            levels.append(rounded)
        # Check adjacent round numbers
        for adj in [-inc, inc]:
            candidate = round(rounded + adj, 5)
            if abs(candidate - price) / price <= tolerance:
                levels.append(candidate)
    return list(set(levels))


# ── v3 Pattern Detector ────────────────────────────────────────────────

def detect_double_bottom(df, min_bars=MIN_PATTERN_BARS, max_bars=MAX_PATTERN_BARS,
                          similarity=MIN_DOUBLE_PATTERN_SIMILARITY):
    """
    Detect a double bottom: two lows at approximately the same price.
    Returns (bottom_level, neckline, stop_loss) or None.
    """
    lows = df["low"].values
    highs = df["high"].values
    n = len(lows)

    results = []
    for i in range(5, n - 5):
        # Local low at i
        if lows[i] != min(lows[max(0, i-5):i+6]):
            continue
        for j in range(i + min_bars, min(n, i + max_bars)):
            # Second local low at j
            if lows[j] != min(lows[max(0, j-5):j+6]):
                continue
            # Both lows similar
            avg_low = (lows[i] + lows[j]) / 2
            if abs(lows[i] - lows[j]) / avg_low > similarity:
                continue
            # Neckline = highest point between the two lows
            neckline = max(highs[i:j+1])
            stop_loss = avg_low * 0.997  # just below both lows
            results.append((i, j, avg_low, neckline, stop_loss))

    return results[-1][2:] if results else None  # (bottom_level, neckline, stop_loss)


def detect_double_top(df, min_bars=MIN_PATTERN_BARS, max_bars=MAX_PATTERN_BARS,
                       similarity=MIN_DOUBLE_PATTERN_SIMILARITY):
    """
    Detect a double top: two highs at approximately the same price.
    Returns (top_level, neckline, stop_loss) or None.
    """
    highs = df["high"].values
    lows  = df["low"].values
    n = len(highs)

    results = []
    for i in range(5, n - 5):
        if highs[i] != max(highs[max(0, i-5):i+6]):
            continue
        for j in range(i + min_bars, min(n, i + max_bars)):
            if highs[j] != max(highs[max(0, j-5):j+6]):
                continue
            avg_high = (highs[i] + highs[j]) / 2
            if abs(highs[i] - highs[j]) / avg_high > similarity:
                continue
            neckline = min(lows[i:j+1])
            stop_loss = avg_high * 1.003
            results.append((i, j, avg_high, neckline, stop_loss))

    return results[-1][2:] if results else None  # (top_level, neckline, stop_loss)


def evaluate_v3_entry(pair, df_history):
    """
    v3 PATTERN-FIRST evaluation.

    1. Detect double bottom → check if price near a psychological level
    2. Detect double top → check if price near a psychological level
    3. Require engulfing candle signal in the pattern direction
    4. NO trend requirement — pattern at psych level is enough
    """
    if len(df_history) < 30:
        return None

    current_price = df_history["close"].iloc[-1]
    signal_det = EntrySignalDetector(min_body_ratio=0.40, lookback_candles=3)

    # ── Try double bottom (bullish) ────────────────────────────────────
    result = detect_double_bottom(df_history.iloc[-50:])
    if result:
        bottom_level, neckline, stop_loss = result
        # Pattern must be at a psychological level
        psych_levels = get_round_levels(bottom_level)
        if psych_levels:
            # Price should be near or just above the neckline (retest zone)
            near_neckline = abs(current_price - neckline) / neckline <= 0.005
            above_neckline = current_price >= neckline * 0.997

            if above_neckline:
                # Look for bullish entry signal
                has_signal, signal = signal_det.has_signal(df_history.iloc[-5:], "long")
                if has_signal and signal and signal.strength >= MIN_SIGNAL_STRENGTH:
                    return {
                        "direction":       "long",
                        "entry_price":     current_price,
                        "stop_loss":       stop_loss,
                        "neckline":        neckline,
                        "psych_level":     psych_levels[0],
                        "pattern":         "double_bottom",
                        "signal_strength": signal.strength,
                        "notes": f"Double bottom at {bottom_level:.5f} (psych: {psych_levels[0]}), neckline {neckline:.5f}",
                    }

    # ── Try double top (bearish) ───────────────────────────────────────
    result = detect_double_top(df_history.iloc[-50:])
    if result:
        top_level, neckline, stop_loss = result
        psych_levels = get_round_levels(top_level)
        if psych_levels:
            below_neckline = current_price <= neckline * 1.003

            if below_neckline:
                has_signal, signal = signal_det.has_signal(df_history.iloc[-5:], "short")
                if has_signal and signal and signal.strength >= MIN_SIGNAL_STRENGTH:
                    return {
                        "direction":       "short",
                        "entry_price":     current_price,
                        "stop_loss":       stop_loss,
                        "neckline":        neckline,
                        "psych_level":     psych_levels[0],
                        "pattern":         "double_top",
                        "signal_strength": signal.strength,
                        "notes": f"Double top at {top_level:.5f} (psych: {psych_levels[0]}), neckline {neckline:.5f}",
                    }

    return None


# ── Data Fetch ──────────────────────────────────────────────────────────

print("Fetching historical Forex data...")
data = {}
for pair, ticker in WATCHLIST.items():
    try:
        df = yf.download(ticker, start="2025-10-01", end="2026-02-22",
                         interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            continue
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        data[pair] = df
        print(f"  ✓ {pair}: {len(df)} days")
    except Exception as e:
        print(f"  ✗ {pair}: {e}")

# ── Backtest ─────────────────────────────────────────────────────────────

balance  = STARTING_BAL
trades   = []
open_pos = {}
be_moved = set()

signal_det = EntrySignalDetector(min_body_ratio=0.40, lookback_candles=3)
all_dates  = pd.date_range(BACKTEST_START, BACKTEST_END, freq="B")

print(f"\nRunning v3 backtest: {BACKTEST_START.date()} → {BACKTEST_END.date()}")
print(f"Starting balance: ${balance:,.2f} | Strategy: PATTERN-FIRST (psych levels)\n")

for current_date in all_dates:
    current_str = current_date.strftime("%Y-%m-%d")

    # ── Monitor open positions ────────────────────────────────────────
    for pair in list(open_pos.keys()):
        pos   = open_pos[pair]
        df_p  = data.get(pair)
        if df_p is None:
            continue

        day_data = df_p[df_p.index <= current_date]
        if len(day_data) == 0:
            continue

        today     = day_data.iloc[-1]
        high      = float(today["high"])
        low       = float(today["low"])
        close     = float(today["close"])
        entry     = pos["entry_price"]
        stop      = pos["stop_loss"]
        direction = pos["direction"]
        risk      = abs(entry - stop)
        pos_key   = f"{pair}_{pos['entry_date']}"

        # Stop hit?
        stop_hit = (direction == "long"  and low  <= stop) or \
                   (direction == "short" and high >= stop)
        if stop_hit:
            p = pnl_dollars(pair, direction, entry, stop, pos["units"])
            balance += p
            rr = -abs(stop - entry) / risk if risk > 0 else -1.0
            trades.append({**pos, "exit_date": current_str, "exit_price": stop,
                           "exit_reason": "stop_hit", "pnl": p, "rr": round(rr, 2)})
            del open_pos[pair]; be_moved.discard(pos_key)
            continue

        # Move to breakeven at 1:1
        if pos_key not in be_moved:
            at_1r = (direction == "long"  and close >= entry + risk) or \
                    (direction == "short" and close <= entry - risk)
            if at_1r:
                pos["stop_loss"] = entry
                be_moved.add(pos_key)
                print(f"  ✅ {current_str}: {pair} moved to breakeven @ {entry:.5f}")

        # Exit signal — only on STRONG reversal AND price at a psych level
        bars_held = (current_date - pd.Timestamp(pos["entry_date"])).days
        opp_dir = "short" if direction == "long" else "long"
        has_exit, exit_sig = signal_det.has_signal(day_data.iloc[-5:], opp_dir)

        should_exit = False
        if has_exit and exit_sig and exit_sig.strength >= EXIT_SIGNAL_STRENGTH:
            if EXIT_ONLY_AT_MAJOR_LEVEL:
                # Only exit if we're near a psychological level
                psych = get_round_levels(close, tolerance=0.012)
                if psych:
                    should_exit = True
                    print(f"  ⚠️  {current_str}: {pair} exit signal at psych level {psych[0]:.5f} (strength={exit_sig.strength:.2f})")
            else:
                should_exit = True

        if should_exit or bars_held >= MAX_HOLD_DAYS:
            reason = "exit_signal" if should_exit else "max_hold"
            p = pnl_dollars(pair, direction, entry, close, pos["units"])
            rr = (abs(close - entry) / risk) * (1 if p > 0 else -1) if risk > 0 else 0
            balance += p
            trades.append({**pos, "exit_date": current_str, "exit_price": close,
                           "exit_reason": reason, "pnl": p, "rr": round(rr, 2)})
            del open_pos[pair]; be_moved.discard(pos_key)
            continue

    # ── Look for entries ──────────────────────────────────────────────
    if open_pos:
        continue  # one trade at a time

    for pair in WATCHLIST:
        df_p = data.get(pair)
        if df_p is None:
            continue
        df_hist = df_p[df_p.index < current_date]
        if len(df_hist) < 40:
            continue

        entry_data = evaluate_v3_entry(pair, df_hist)
        if not entry_data:
            continue

        risk_pct     = get_risk_pct(balance)
        entry_price  = entry_data["entry_price"]
        stop_loss    = entry_data["stop_loss"]
        units        = calc_units(balance, risk_pct, entry_price, stop_loss)
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
        print(f"  📈 {current_str}: ENTER {pair} {entry_data['direction'].upper()} "
              f"@ {entry_price:.5f}  SL={stop_loss:.5f}  "
              f"Risk={risk_pct}% (${risk_dollars:.0f})  [{entry_data['notes']}]")
        break

# Close any still-open at end
for pair, pos in open_pos.items():
    df_p = data.get(pair)
    if df_p is None:
        continue
    last = float(df_p[df_p.index <= BACKTEST_END]["close"].iloc[-1])
    entry = pos["entry_price"]
    stop  = pos["stop_loss"]
    risk  = abs(entry - stop)
    p = pnl_dollars(pair, pos["direction"], entry, last, pos["units"])
    rr = (abs(last - entry) / risk) * (1 if p > 0 else -1) if risk > 0 else 0
    balance += p
    trades.append({**pos, "exit_date": BACKTEST_END.strftime("%Y-%m-%d"),
                   "exit_price": last, "exit_reason": "open_at_end",
                   "pnl": p, "rr": round(rr, 2)})

# ── Results ──────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print(f"v3 BACKTEST RESULTS — Jan 1 → Feb 21, 2026")
print(f"Strategy: PATTERN-FIRST | Psych levels | No trend filter")
print(f"Starting capital: ${STARTING_BAL:,.2f}")
print("=" * 65)

if not trades:
    print("\nNo trades taken.")
    print("Possible causes:")
    print("  - No double top/bottom patterns formed at psychological levels")
    print("  - Engulfing signal never confirmed")
    print("  - This is realistic — Jan/Feb 2026 may have been low-quality")
    sys.exit(0)

wins   = [t for t in trades if t["pnl"] > 0]
losses = [t for t in trades if t["pnl"] <= 0]
total_pnl = sum(t["pnl"] for t in trades)
win_rate  = len(wins) / len(trades) * 100 if trades else 0

print(f"\n📊 TRADE SUMMARY")
print(f"  Total trades:   {len(trades)}")
print(f"  Wins / Losses:  {len(wins)} / {len(losses)}")
print(f"  Win rate:       {win_rate:.0f}%")
if wins:   print(f"  Avg R (wins):   +{np.mean([t['rr'] for t in wins]):.1f}R")
if losses: print(f"  Avg R (losses): {np.mean([t['rr'] for t in losses]):.1f}R")

print(f"\n💰 P&L SUMMARY")
print(f"  Starting:  ${STARTING_BAL:>10,.2f}")
print(f"  Total P&L: ${total_pnl:>+10,.2f}")
print(f"  Final:     ${balance:>10,.2f}")
print(f"  Return:    {(balance/STARTING_BAL - 1)*100:>+.1f}%")

print(f"\n📋 TRADES")
print(f"  {'Date In':<12} {'Pair':<10} {'Dir':<6} {'Entry':>9} {'Exit':>9} {'SL':>9} {'R:R':>6} {'P&L':>10} {'Reason'}")
print(f"  {'-'*88}")
for t in trades:
    rr_str  = f"{t['rr']:+.1f}R"
    pnl_str = f"${t['pnl']:+,.2f}"
    print(f"  {t['entry_date']:<12} {t['pair']:<10} {t['direction']:<6} "
          f"{t['entry_price']:>9.4f} {t['exit_price']:>9.4f} {t['stop_loss']:>9.4f} "
          f"{rr_str:>6} {pnl_str:>10}  {t['exit_reason']}")
    print(f"  {'':12} {t.get('notes','')[:70]}")

print(f"\n⚠️  METHODOLOGY NOTES")
print(f"  • v3 = PATTERN-FIRST: double top/bottom AT psychological levels")
print(f"  • NO trend filter — counter-trend reversal at psych level is valid")
print(f"  • Exit: reversal candle strength ≥ {EXIT_SIGNAL_STRENGTH} AND at psych level")
print(f"  • Breakeven at 1:1, then hold")
print(f"  • Daily candles (1H data would improve entry precision)")
print()
