"""
4-way strategy comparison on 7-day SOL 5-min data:
  1. STRICT    — RSI cross + MACD flip + EMA
  2. LOOSE     — RSI cross + MACD direction + EMA
  3. TRAILING  — LOOSE entry + trailing stop
  4. VOLUME    — Volume surge + VWAP cross + price breakout (fast entry)

Volume strategy capitalizes on moves as they happen rather than waiting
for lagging indicators to confirm after the move is already underway.
"""

import os, sys, json, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

import pandas as pd
import numpy as np

PRODUCT_ID  = "SLP-20DEC30-CDE"
FEE_RATE    = 0.0005
CAPITAL     = 1000.0
RISK_PCT    = 0.02
MAX_POS_PCT = 0.40
MIN_RR      = 3.0
CANDLES_NEEDED = 50

VOL_SURGE_MULT  = 1.5   # Volume must be 1.5x rolling avg to qualify
VOL_AVG_PERIOD  = 20    # Rolling average period
BREAKOUT_PERIOD = 10    # Look back N bars for range breakout
TRAIL_DIST_MULT = 1.0   # Trail distance = original stop distance


# ──────────────────────────────────────────────────────────
# Indicator builder
# ──────────────────────────────────────────────────────────

def build_df(candles):
    df = pd.DataFrame(candles).sort_values('timestamp').reset_index(drop=True)
    c = df['close'] = df['close'].astype(float)
    h = df['high']  = df['high'].astype(float)
    l = df['low']   = df['low'].astype(float)
    v = df['volume']= df['volume'].astype(float)

    # RSI(14)
    delta = c.diff()
    ag = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['rsi'] = (100 - 100 / (1 + ag / al.replace(0, np.nan))).fillna(50)

    # MACD histogram
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df['macd_hist'] = macd - macd.ewm(span=9, adjust=False).mean()

    # EMA20
    df['ema20'] = c.ewm(span=20, adjust=False).mean()

    # Volume rolling average + surge flag
    df['vol_avg']   = v.rolling(VOL_AVG_PERIOD, min_periods=5).mean()
    df['vol_surge'] = v >= df['vol_avg'] * VOL_SURGE_MULT

    # VWAP (rolling intraday — resets every 96 bars ≈ 8h on 5-min)
    # Simple cumulative VWAP using all available history (good enough for signals)
    tp = (h + l + c) / 3
    df['vwap'] = (tp * v).cumsum() / v.cumsum()

    # Rolling high/low for breakout detection
    df['range_high'] = h.rolling(BREAKOUT_PERIOD).max().shift(1)
    df['range_low']  = l.rolling(BREAKOUT_PERIOD).min().shift(1)

    return df


# ──────────────────────────────────────────────────────────
# Signal functions
# ──────────────────────────────────────────────────────────

def sig_strict(df, i):
    r, p = df.iloc[i], df.iloc[i-1]
    lows  = df['low'].iloc[max(0,i-5):i+1]
    highs = df['high'].iloc[max(0,i-5):i+1]
    if (p.rsi < 50 and r.rsi >= 50 and r.rsi < 65
            and p.macd_hist < 0 and r.macd_hist >= 0
            and r.close > r.ema20):
        return 'BUY', r.close, lows.min() * 0.998
    if (p.rsi > 50 and r.rsi <= 50 and r.rsi > 35
            and p.macd_hist > 0 and r.macd_hist <= 0
            and r.close < r.ema20):
        return 'SELL', r.close, highs.max() * 1.002
    return 'NONE', 0, 0


def sig_loose(df, i):
    r, p = df.iloc[i], df.iloc[i-1]
    lows  = df['low'].iloc[max(0,i-5):i+1]
    highs = df['high'].iloc[max(0,i-5):i+1]
    if (p.rsi < 50 and r.rsi >= 50 and r.rsi < 65
            and r.macd_hist > 0 and r.close > r.ema20):
        return 'BUY', r.close, lows.min() * 0.998
    if (p.rsi > 50 and r.rsi <= 50 and r.rsi > 35
            and r.macd_hist < 0 and r.close < r.ema20):
        return 'SELL', r.close, highs.max() * 1.002
    return 'NONE', 0, 0


def sig_volume(df, i):
    """
    Volume-surge + VWAP cross + breakout entry.
    Fires DURING the move, not after lagging indicators catch up.

    BUY conditions:
      - Volume surge (≥1.5x avg)
      - Price closing above the 10-bar range high (breakout)
      - Price crossed above VWAP this bar
      - MACD histogram positive (momentum direction confirm)

    SELL conditions: mirror image.
    """
    r, p = df.iloc[i], df.iloc[i-1]

    if pd.isna(r.range_high) or pd.isna(r.vwap) or pd.isna(r.vol_avg):
        return 'NONE', 0, 0

    vol_ok  = r.vol_surge
    vwap_cross_up   = p.close <= p.vwap and r.close > r.vwap
    vwap_cross_down = p.close >= p.vwap and r.close < r.vwap
    breakout_up     = r.close > r.range_high
    breakout_down   = r.close < r.range_low
    macd_pos = r.macd_hist > 0
    macd_neg = r.macd_hist < 0

    # Don't enter if RSI already overbought/oversold (chase filter)
    rsi_ok_buy  = r.rsi < 70
    rsi_ok_sell = r.rsi > 30

    lows  = df['low'].iloc[max(0,i-5):i+1]
    highs = df['high'].iloc[max(0,i-5):i+1]

    if vol_ok and (vwap_cross_up or breakout_up) and macd_pos and rsi_ok_buy:
        sl = lows.min() * 0.998
        return 'BUY', r.close, sl

    if vol_ok and (vwap_cross_down or breakout_down) and macd_neg and rsi_ok_sell:
        sl = highs.max() * 1.002
        return 'SELL', r.close, sl

    return 'NONE', 0, 0


# ──────────────────────────────────────────────────────────
# Backtest runner
# ──────────────────────────────────────────────────────────

def run(df, signal_fn, use_trailing=False):
    cash, position, trades = CAPITAL, None, []

    for i in range(CANDLES_NEEDED, len(df)):
        row   = df.iloc[i]
        price = float(row.close)
        high  = float(row.high)
        low   = float(row.low)
        ts    = datetime.fromtimestamp(int(row['timestamp']), tz=timezone.utc)

        if position:
            sl, tp = position['sl'], position['tp']

            # Trailing stop update
            if use_trailing:
                if position.get('trail_active'):
                    td = position['trail_dist']
                    if position['side'] == 'BUY':
                        ns = high - td
                        if ns > sl: position['sl'] = sl = ns
                    else:
                        ns = low + td
                        if ns < sl: position['sl'] = sl = ns

                # Activate trail once price clears initial TP
                if not position.get('trail_active'):
                    if position['side'] == 'BUY'  and high >= tp:
                        position['trail_active'] = True
                        position['sl'] = sl = tp - position['trail_dist']
                    elif position['side'] == 'SELL' and low <= tp:
                        position['trail_active'] = True
                        position['sl'] = sl = tp + position['trail_dist']

            hit_sl = low  <= sl if position['side'] == 'BUY'  else high >= sl
            hit_tp = (not use_trailing) and (
                high >= tp if position['side'] == 'BUY' else low <= tp
            )

            if hit_tp or hit_sl:
                exit_p  = (tp if hit_tp else sl)
                x_fee   = position['notional'] * FEE_RATE
                gross   = ((exit_p - position['entry']) * position['size']
                           if position['side'] == 'BUY'
                           else (position['entry'] - exit_p) * position['size'])
                net     = gross - x_fee - position['entry_fee']
                cash   += position['notional'] + gross - x_fee
                reason  = ('TP' if hit_tp else
                           'TRAIL' if position.get('trail_active') else 'SL')
                trades.append({
                    'open': position['open_ts'],
                    'close': ts.strftime('%m-%d %H:%M'),
                    'side': position['side'],
                    'entry': round(position['entry'], 3),
                    'exit': round(exit_p, 3),
                    'net': round(net, 4),
                    'fees': round(position['entry_fee'] + x_fee, 4),
                    'result': 'WIN' if net > 0 else 'LOSS',
                    'reason': reason,
                    'dur': round((int(row['timestamp']) - position['open_ts_ts']) / 60),
                    'vol_entry': position.get('vol_ratio', 0),
                })
                position = None

        if position:
            continue

        side, entry, sl = signal_fn(df, i)
        if side == 'NONE' or entry <= 0:
            continue

        stop_d = abs(entry - sl) / entry
        if stop_d < 0.003:
            continue

        pos_size   = min(cash * RISK_PCT / stop_d, cash * MAX_POS_PCT)
        e_fee      = pos_size * FEE_RATE
        total_fees = e_fee * 2
        tgt_d      = stop_d * MIN_RR
        gross_tp   = pos_size * tgt_d
        net_tp     = gross_tp - total_fees
        act_risk   = pos_size * stop_d + total_fees

        if net_tp / act_risk < MIN_RR * 0.9: continue
        if total_fees / gross_tp > 0.20:     continue
        if cash < pos_size + e_fee:          continue

        tp        = entry * (1 + tgt_d) if side == 'BUY' else entry * (1 - tgt_d)
        contracts = max(1, int(pos_size / entry))
        notional  = contracts * entry
        e_fee     = notional * FEE_RATE
        cash     -= notional + e_fee

        vol_ratio = float(row.volume / row.vol_avg) if row.vol_avg > 0 else 0

        position = {
            'side': side, 'entry': entry, 'sl': sl, 'tp': tp,
            'size': float(contracts), 'notional': notional, 'entry_fee': e_fee,
            'open_ts': ts.strftime('%m-%d %H:%M'), 'open_ts_ts': int(row['timestamp']),
            'trail_dist': abs(entry - sl) * TRAIL_DIST_MULT, 'trail_active': False,
            'vol_ratio': round(vol_ratio, 2),
        }

    # Close remaining position at last bar
    if position:
        lp = float(df.iloc[-1].close)
        xf = position['notional'] * FEE_RATE
        gross = ((lp - position['entry']) * position['size'] if position['side'] == 'BUY'
                 else (position['entry'] - lp) * position['size'])
        net = gross - xf - position['entry_fee']
        cash += position['notional'] + gross - xf
        trades.append({
            'open': position['open_ts'], 'close': 'OPEN',
            'side': position['side'], 'entry': round(position['entry'], 3),
            'exit': round(lp, 3), 'net': round(net, 4),
            'fees': round(position['entry_fee'] + xf, 4),
            'result': 'WIN' if net > 0 else 'LOSS',
            'reason': 'OPEN_END', 'dur': 0, 'vol_entry': 0,
        })

    return trades, cash


# ──────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────

def report(label, trades):
    closed = [t for t in trades if t['reason'] != 'OPEN_END']
    if not closed:
        print(f"\n  {label}: NO TRADES")
        return 0

    wins   = [t for t in closed if t['result'] == 'WIN']
    losses = [t for t in closed if t['result'] == 'LOSS']
    net    = sum(t['net'] for t in closed)
    fees   = sum(t['fees'] for t in closed)
    aw = sum(t['net'] for t in wins)  / len(wins)   if wins   else 0
    al = sum(t['net'] for t in losses)/ len(losses)  if losses else 0
    pf = abs(aw*len(wins)) / abs(al*len(losses)) if losses and al != 0 else float('inf')

    print(f"\n{'─'*58}")
    print(f"  {label}")
    print(f"{'─'*58}")
    print(f"  {len(closed)} trades  |  {len(wins)}W/{len(losses)}L  |  "
          f"WR: {len(wins)/len(closed)*100:.0f}%  |  "
          f"PF: {pf:.2f}  |  Avg W/L: ${aw:+.2f}/${al:+.2f}")
    print(f"  Net: ${net:+.2f}  |  Fees: ${fees:.2f}  |  Return: {net/CAPITAL*100:+.2f}%")
    print()

    for t in trades:
        vol = f" vol={t['vol_entry']}x" if t.get('vol_entry', 0) > 0 else ""
        tag = " 📈" if t.get('reason') == 'TRAIL' else ""
        oe  = " (open)" if t['reason'] == 'OPEN_END' else ""
        print(f"  {'✅' if t['result']=='WIN' else '❌'}  {t['side']:4s}  "
              f"{t['open']}→{t['close']:14s}  "
              f"{t['entry']:.2f}→{t['exit']:.2f}  "
              f"net={t['net']:+.2f}  {t['reason']}{tag}{vol}{oe}")
    return net


def main():
    cache = "data/historical/sol_5min_7day.json"
    with open(cache) as f:
        candles = json.load(f)

    df = build_df(candles)
    prices = df['close']
    print(f"\nSOL 5-min  |  7 days  |  {len(candles)} candles")
    print(f"SOL: ${prices.iloc[0]:.2f} → ${prices.iloc[-1]:.2f}  "
          f"(range ${prices.min():.2f}–${prices.max():.2f})")
    print(f"Capital: ${CAPITAL:,.0f}  Risk: {RISK_PCT*100:.0f}%/trade  "
          f"Fee: {FEE_RATE*100:.3f}%/side  Min R:R: {MIN_RR:.0f}:1 after fees")

    t1, _ = run(df, sig_strict,  use_trailing=False)
    t2, _ = run(df, sig_loose,   use_trailing=False)
    t3, _ = run(df, sig_loose,   use_trailing=True)
    t4, _ = run(df, sig_volume,  use_trailing=False)
    t5, _ = run(df, sig_volume,  use_trailing=True)

    n1 = report("1. STRICT      — RSI cross + MACD flip + EMA (fixed TP)", t1)
    n2 = report("2. LOOSE       — RSI cross + MACD direction + EMA (fixed TP)", t2)
    n3 = report("3. LOOSE+TRAIL — Loose entry + trailing stop", t3)
    n4 = report("4. VOLUME      — Vol surge + VWAP cross + breakout (fixed TP)", t4)
    n5 = report("5. VOL+TRAIL   — Volume entry + trailing stop", t5)

    results = {'strict': n1, 'loose': n2, 'loose_trail': n3,
               'volume': n4, 'vol_trail': n5}
    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'═'*58}")
    print("  RANKING (by net P&L):")
    for rank, (name, val) in enumerate(ranked, 1):
        print(f"  {rank}. {name:15s}  ${val:+.2f}  ({val/CAPITAL*100:+.2f}%)")
    print(f"{'═'*58}")

    with open("backtesting/results/volume_comparison.json", "w") as f:
        json.dump({'trades': {'strict': t1,'loose': t2,'loose_trail': t3,
                              'volume': t4,'vol_trail': t5},
                   'summary': results}, f, indent=2)


if __name__ == "__main__":
    main()
