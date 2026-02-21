"""
Compare three strategy variants on the same 7-day SOL 5-min data:
  1. STRICT  — RSI crossover + MACD flip + EMA alignment (same-bar)
  2. LOOSE   — RSI crossover + MACD direction + EMA position
  3. TRAILING— LOOSE entry + trailing stop (activates after initial TP level)
"""

import os, sys, json, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

import pandas as pd
import numpy as np
from src.exchange.coinbase_client import CoinbaseClient

PRODUCT_ID  = "SLP-20DEC30-CDE"
GRANULARITY = "FIVE_MINUTE"
FEE_RATE    = 0.0005
CAPITAL     = 1000.0
RISK_PCT    = 0.02
MAX_POS_PCT = 0.40
MIN_RR      = 3.0
CANDLES_NEEDED = 50

# Trailing stop params
TRAIL_ACTIVATION_MULT = 1.0   # Activate trail once price hits 1x the original TP distance
TRAIL_DISTANCE_MULT   = 1.0   # Trail at same distance as original stop


# ------------------------------------------------------------------ #
# Indicators                                                           #
# ------------------------------------------------------------------ #

def calc_indicators(candles):
    df = pd.DataFrame(candles).sort_values('timestamp').reset_index(drop=True)
    closes = df['close'].astype(float)
    highs  = df['high'].astype(float)
    lows   = df['low'].astype(float)

    # RSI
    delta    = closes.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = (100 - (100 / (1 + rs))).fillna(50)

    # MACD
    ema12      = closes.ewm(span=12, adjust=False).mean()
    ema26      = closes.ewm(span=26, adjust=False).mean()
    macd_hist  = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    # EMA20
    ema20 = closes.ewm(span=20, adjust=False).mean()

    df['rsi']       = rsi
    df['macd_hist'] = macd_hist
    df['ema20']     = ema20
    df['close_f']   = closes
    df['high_f']    = highs
    df['low_f']     = lows
    return df


# ------------------------------------------------------------------ #
# Signal functions                                                     #
# ------------------------------------------------------------------ #

def signal_strict(df, i):
    """RSI cross + MACD flip + EMA align — all same bar."""
    r, p = df.iloc[i], df.iloc[i-1]
    close, ema20 = r.close_f, r.ema20
    lows  = df['low_f'].iloc[max(0,i-5):i+1]
    highs = df['high_f'].iloc[max(0,i-5):i+1]

    if (p.rsi < 50 and r.rsi >= 50 and r.rsi < 65
            and p.macd_hist < 0 and r.macd_hist >= 0
            and close > ema20):
        sl = lows.min() * 0.998
        return 'BUY', close, sl

    if (p.rsi > 50 and r.rsi <= 50 and r.rsi > 35
            and p.macd_hist > 0 and r.macd_hist <= 0
            and close < ema20):
        sl = highs.max() * 1.002
        return 'SELL', close, sl

    return 'NONE', 0, 0


def signal_loose(df, i):
    """RSI cross + MACD direction + EMA position."""
    r, p = df.iloc[i], df.iloc[i-1]
    close, ema20 = r.close_f, r.ema20
    lows  = df['low_f'].iloc[max(0,i-5):i+1]
    highs = df['high_f'].iloc[max(0,i-5):i+1]

    if (p.rsi < 50 and r.rsi >= 50 and r.rsi < 65
            and r.macd_hist > 0
            and close > ema20):
        sl = lows.min() * 0.998
        return 'BUY', close, sl

    if (p.rsi > 50 and r.rsi <= 50 and r.rsi > 35
            and r.macd_hist < 0
            and close < ema20):
        sl = highs.max() * 1.002
        return 'SELL', close, sl

    return 'NONE', 0, 0


# ------------------------------------------------------------------ #
# Core backtest runner                                                  #
# ------------------------------------------------------------------ #

def run(df, signal_fn, use_trailing=False):
    cash     = CAPITAL
    position = None
    trades   = []

    for i in range(CANDLES_NEEDED, len(df)):
        row   = df.iloc[i]
        price = row.close_f
        high  = row.high_f
        low   = row.low_f
        ts    = datetime.fromtimestamp(int(row['timestamp']), tz=timezone.utc)

        # ---- Manage open position ----
        if position:
            sl, tp = position['sl'], position['tp']

            # Update trailing stop
            if use_trailing and position.get('trailing_active'):
                trail_dist = position['trail_dist']
                if position['side'] == 'BUY':
                    new_sl = high - trail_dist
                    if new_sl > position['sl']:
                        position['sl'] = new_sl
                        sl = new_sl
                else:
                    new_sl = low + trail_dist
                    if new_sl < position['sl']:
                        position['sl'] = new_sl
                        sl = new_sl

            # Activate trailing once price clears initial TP level
            if use_trailing and not position.get('trailing_active'):
                if position['side'] == 'BUY'  and high >= tp:
                    position['trailing_active'] = True
                    position['sl'] = tp - position['trail_dist']  # lock in at TP
                    sl = position['sl']
                elif position['side'] == 'SELL' and low <= tp:
                    position['trailing_active'] = True
                    position['sl'] = tp + position['trail_dist']
                    sl = position['sl']

            # Check exits (use intrabar high/low for realism)
            if position['side'] == 'BUY':
                hit_sl = low  <= sl
                hit_tp = not use_trailing and high >= tp
            else:
                hit_sl = high >= sl
                hit_tp = not use_trailing and low  <= tp

            if hit_tp or hit_sl:
                if hit_tp:
                    exit_price  = tp
                    exit_reason = 'TP'
                else:
                    exit_price  = sl
                    exit_reason = 'SL' if not position.get('trailing_active') else 'TRAIL'

                exit_fee = position['notional'] * FEE_RATE
                if position['side'] == 'BUY':
                    gross = (exit_price - position['entry']) * position['size']
                else:
                    gross = (position['entry'] - exit_price) * position['size']
                net = gross - exit_fee - position['entry_fee']
                cash += position['notional'] + gross - exit_fee

                trades.append({
                    'ts_open':  position['open_ts'],
                    'ts_close': ts.strftime('%m-%d %H:%M'),
                    'side':     position['side'],
                    'entry':    round(position['entry'], 3),
                    'exit':     round(exit_price, 3),
                    'sl_final': round(sl, 3),
                    'tp_orig':  round(tp, 3),
                    'net_pnl':  round(net, 4),
                    'fees':     round(position['entry_fee'] + exit_fee, 4),
                    'result':   'WIN' if net > 0 else 'LOSS',
                    'reason':   exit_reason,
                    'dur_min':  round((int(row['timestamp']) - position['open_ts_ts']) / 60),
                    'trailing': position.get('trailing_active', False),
                })
                position = None
                continue  # check for new signal this bar

        # ---- New signal ----
        if position:
            continue

        side, entry, sl = signal_fn(df, i)
        if side == 'NONE':
            continue

        stop_d = abs(entry - sl) / entry
        if stop_d < 0.003:
            continue

        risk_budget   = cash * RISK_PCT
        pos_size      = min(risk_budget / stop_d, cash * MAX_POS_PCT)
        entry_fee     = pos_size * FEE_RATE
        total_fees    = entry_fee * 2
        target_d      = stop_d * MIN_RR
        gross_tp      = pos_size * target_d
        net_tp        = gross_tp - total_fees
        actual_risk   = pos_size * stop_d + total_fees

        if net_tp / actual_risk < MIN_RR * 0.9:
            continue
        if total_fees / gross_tp > 0.20:
            continue
        if cash < pos_size + entry_fee:
            continue

        tp         = entry * (1 + target_d) if side == 'BUY' else entry * (1 - target_d)
        contracts  = max(1, int(pos_size / entry))
        notional   = contracts * entry
        entry_fee  = notional * FEE_RATE
        cash      -= notional + entry_fee

        position = {
            'side':       side,
            'entry':      entry,
            'sl':         sl,
            'tp':         tp,
            'size':       float(contracts),
            'notional':   notional,
            'entry_fee':  entry_fee,
            'open_ts':    ts.strftime('%m-%d %H:%M'),
            'open_ts_ts': int(row['timestamp']),
            'trail_dist': abs(entry - sl) * TRAIL_DISTANCE_MULT,
            'trailing_active': False,
        }

    # Close open position at last price
    if position:
        last_price = float(df.iloc[-1].close_f)
        exit_fee   = position['notional'] * FEE_RATE
        gross = (last_price - position['entry']) * position['size'] if position['side'] == 'BUY' \
                else (position['entry'] - last_price) * position['size']
        net = gross - exit_fee - position['entry_fee']
        cash += position['notional'] + gross - exit_fee
        trades.append({
            'ts_open': position['open_ts'], 'ts_close': 'OPEN',
            'side': position['side'], 'entry': round(position['entry'], 3),
            'exit': round(last_price, 3), 'sl_final': 0, 'tp_orig': round(position['tp'], 3),
            'net_pnl': round(net, 4), 'fees': round(position['entry_fee'] + exit_fee, 4),
            'result': 'WIN' if net > 0 else 'LOSS', 'reason': 'OPEN_END',
            'dur_min': 0, 'trailing': False,
        })

    return trades, cash


# ------------------------------------------------------------------ #
# Reporting                                                            #
# ------------------------------------------------------------------ #

def summarize(label, trades, final_cash):
    closed = [t for t in trades if t['reason'] != 'OPEN_END']
    wins   = [t for t in closed if t['result'] == 'WIN']
    losses = [t for t in closed if t['result'] == 'LOSS']
    net    = sum(t['net_pnl'] for t in closed)
    fees   = sum(t['fees']    for t in closed)
    aw = sum(t['net_pnl'] for t in wins)  / len(wins)  if wins  else 0
    al = sum(t['net_pnl'] for t in losses) / len(losses) if losses else 0
    pf = abs(aw * len(wins)) / abs(al * len(losses)) if losses and al != 0 else float('inf')

    print(f"\n{'─'*56}")
    print(f"  {label}")
    print(f"{'─'*56}")
    print(f"  Trades: {len(closed)}  |  {len(wins)}W / {len(losses)}L  |  "
          f"Win rate: {len(wins)/len(closed)*100:.0f}%" if closed else "  No trades")
    print(f"  Avg win: ${aw:+.2f}   Avg loss: ${al:+.2f}   PF: {pf:.2f}")
    print(f"  Net P&L: ${net:+.2f}   Fees: ${fees:.2f}   "
          f"Return: {net/CAPITAL*100:+.2f}%")
    print()
    for t in trades:
        trail_tag = " 📈trail" if t.get('trailing') else ""
        open_tag  = " (still open)" if t['reason'] == 'OPEN_END' else ""
        print(f"  {t['result']:4s} | {t['side']:4s} | {t['ts_open']}→{t['ts_close']} | "
              f"{t['entry']:.2f}→{t['exit']:.2f} | "
              f"net={t['net_pnl']:+.2f} | {t['reason']}{trail_tag}{open_tag}")
    return net


def main():
    # Load or fetch candles
    cache = "data/historical/sol_5min_7day.json"
    if os.path.exists(cache):
        with open(cache) as f:
            candles = json.load(f)
        print(f"Loaded {len(candles)} cached candles.")
    else:
        client = CoinbaseClient()
        print("Fetching candles...")
        now = int(time.time())
        chunk = 290 * 5 * 60
        start = now - 7 * 24 * 3600
        all_c = {}
        cursor = start
        while cursor < now:
            batch = client.get_candles(PRODUCT_ID, GRANULARITY, cursor, min(cursor+chunk, now))
            for c in batch:
                all_c[c['timestamp']] = c
            cursor += chunk
            time.sleep(0.2)
        candles = sorted(all_c.values(), key=lambda x: x['timestamp'])
        with open(cache, 'w') as f:
            json.dump(candles, f)
        print(f"Fetched {len(candles)} candles.")

    df = calc_indicators(candles)

    print(f"\nSOL 5-min Backtest — Past 7 Days ({len(candles)} candles)")
    print(f"Capital: ${CAPITAL:,.0f}  |  Risk/trade: {RISK_PCT*100:.0f}%  |  "
          f"Fees: {FEE_RATE*100:.2f}%/side  |  Min R:R: {MIN_RR:.0f}:1 after fees")

    t1, c1 = run(df, signal_strict,  use_trailing=False)
    t2, c2 = run(df, signal_loose,   use_trailing=False)
    t3, c3 = run(df, signal_loose,   use_trailing=True)

    n1 = summarize("STRICT  — fixed TP",      t1, c1)
    n2 = summarize("LOOSE   — fixed TP",      t2, c2)
    n3 = summarize("LOOSE + TRAILING STOP",   t3, c3)

    print(f"\n{'═'*56}")
    print(f"  WINNER: ", end="")
    best = max([(n1,'STRICT'),(n2,'LOOSE'),(n3,'TRAILING')], key=lambda x: x[0])
    print(f"{best[1]} with {best[0]:+.2f} net P&L")
    print(f"{'═'*56}")

    with open("backtesting/results/comparison_3way.json", "w") as f:
        json.dump({'strict': t1, 'loose': t2, 'trailing': t3}, f, indent=2)


if __name__ == "__main__":
    main()
