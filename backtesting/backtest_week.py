"""
One-week backtest of SOL momentum strategy on 5-minute candles.
Uses realistic fee modeling (0.05%/side futures) and swing-based stops.
"""

import os, sys, time, json
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

from src.exchange.coinbase_client import CoinbaseClient
from src.strategy.sol_momentum import SOLMomentumStrategy

PRODUCT_ID  = "SLP-20DEC30-CDE"
GRANULARITY = "FIVE_MINUTE"
FEE_RATE    = 0.0005   # 0.05% per side (futures)
CAPITAL     = 1000.0
RISK_PCT    = 0.02     # 2% per trade
MAX_POS_PCT = 0.40     # 40% of capital per position
MIN_RR      = 3.0      # 1:3 after fees
CANDLES_PER_REQUEST = 290   # Coinbase max is 300; use 290 for safety
CANDLES_NEEDED = 50


def fetch_week_candles(client: CoinbaseClient) -> list[dict]:
    """Fetch one full week of 5-min candles, paginating in 290-candle chunks."""
    end   = int(time.time())
    start = end - 7 * 24 * 3600
    chunk = CANDLES_PER_REQUEST * 5 * 60  # seconds per chunk

    all_candles = {}
    cursor = start
    requests = 0
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        batch = client.get_candles(PRODUCT_ID, GRANULARITY, cursor, chunk_end)
        for c in batch:
            all_candles[c['timestamp']] = c
        requests += 1
        cursor = chunk_end
        if cursor < end:
            time.sleep(0.2)   # gentle rate limiting

    candles = sorted(all_candles.values(), key=lambda x: x['timestamp'])
    print(f"Fetched {len(candles)} candles in {requests} requests "
          f"({len(candles)*5/60:.1f} hours of data)")
    return candles


def run_backtest(candles: list[dict]) -> dict:
    """Walk forward through candles, applying the strategy at each bar close."""
    strategy = SOLMomentumStrategy()

    capital      = CAPITAL
    peak_capital = CAPITAL
    cash         = CAPITAL

    trades      = []
    position    = None   # dict when open
    equity_curve = []

    for i in range(CANDLES_NEEDED, len(candles)):
        window  = candles[: i + 1]   # everything up to and including bar i
        current = candles[i]
        price   = float(current['close'])
        ts      = datetime.fromtimestamp(current['timestamp'], tz=timezone.utc)

        # --- Check open position exits first ---
        if position:
            if position['side'] == 'BUY':
                hit_sl = price <= position['sl']
                hit_tp = price >= position['tp']
            else:
                hit_sl = price >= position['sl']
                hit_tp = price <= position['tp']

            if hit_tp or hit_sl:
                exit_price = position['tp'] if hit_tp else position['sl']
                exit_fee   = position['notional'] * FEE_RATE
                if position['side'] == 'BUY':
                    gross = (exit_price - position['entry']) * position['size']
                else:
                    gross = (position['entry'] - exit_price) * position['size']
                net = gross - exit_fee - position['entry_fee']
                cash += position['notional'] + gross - exit_fee

                trades.append({
                    'open_ts':  position['open_ts'].isoformat(),
                    'close_ts': ts.isoformat(),
                    'side':     position['side'],
                    'entry':    round(position['entry'], 4),
                    'exit':     round(exit_price, 4),
                    'sl':       round(position['sl'], 4),
                    'tp':       round(position['tp'], 4),
                    'size':     position['size'],
                    'contracts': position['contracts'],
                    'gross_pnl': round(gross, 4),
                    'fees':     round(position['entry_fee'] + exit_fee, 4),
                    'net_pnl':  round(net, 4),
                    'result':   'WIN' if net > 0 else 'LOSS',
                    'exit_reason': 'TP' if hit_tp else 'SL',
                    'duration_min': round((current['timestamp'] - position['open_ts_ts']) / 60, 1),
                })
                position = None
                if cash > peak_capital:
                    peak_capital = cash

        # --- Skip if already in a position ---
        if position:
            equity_curve.append({'ts': ts.isoformat(), 'value': round(cash, 2)})
            continue

        # --- Generate signal ---
        signal = strategy.analyze(window)
        if signal.side == 'NONE':
            equity_curve.append({'ts': ts.isoformat(), 'value': round(cash, 2)})
            continue

        entry  = signal.entry_price
        sl     = signal.stop_loss
        stop_d = abs(entry - sl) / entry
        if stop_d <= 0:
            continue

        # Position sizing (2% risk)
        risk_budget   = cash * RISK_PCT
        position_size = min(risk_budget / stop_d, cash * MAX_POS_PCT)

        # Fee calc
        entry_fee  = position_size * FEE_RATE
        exit_fee   = position_size * FEE_RATE
        total_fees = entry_fee + exit_fee

        # Target = 3x stop after fees
        target_d   = stop_d * MIN_RR
        gross_tp   = position_size * target_d
        net_tp     = gross_tp - total_fees
        actual_risk = position_size * stop_d + total_fees

        if net_tp / actual_risk < MIN_RR * 0.9:  # slight tolerance
            continue
        if total_fees / gross_tp > 0.20:
            continue

        if cash < position_size + entry_fee:
            continue

        # Compute TP price
        if signal.side == 'BUY':
            tp = entry * (1 + target_d)
        else:
            tp = entry * (1 - target_d)

        contracts = max(1, int(position_size / entry))
        notional  = contracts * entry
        entry_fee = notional * FEE_RATE

        cash -= notional + entry_fee
        position = {
            'side':      signal.side,
            'entry':     entry,
            'sl':        sl,
            'tp':        tp,
            'size':      float(contracts),
            'contracts': contracts,
            'notional':  notional,
            'entry_fee': entry_fee,
            'open_ts':   ts,
            'open_ts_ts': current['timestamp'],
        }

        equity_curve.append({'ts': ts.isoformat(), 'value': round(cash + notional, 2)})

    # Close any still-open position at last price
    if position and candles:
        last = candles[-1]
        last_price = float(last['close'])
        exit_fee   = position['notional'] * FEE_RATE
        if position['side'] == 'BUY':
            gross = (last_price - position['entry']) * position['size']
        else:
            gross = (position['entry'] - last_price) * position['size']
        net = gross - exit_fee - position['entry_fee']
        cash += position['notional'] + gross - exit_fee
        trades.append({
            'open_ts': position['open_ts'].isoformat(),
            'close_ts': 'OPEN',
            'side': position['side'],
            'entry': round(position['entry'], 4),
            'exit': round(last_price, 4),
            'sl': round(position['sl'], 4),
            'tp': round(position['tp'], 4),
            'size': position['size'],
            'contracts': position['contracts'],
            'gross_pnl': round(gross, 4),
            'fees': round(position['entry_fee'] + exit_fee, 4),
            'net_pnl': round(net, 4),
            'result': 'WIN' if net > 0 else 'LOSS',
            'exit_reason': 'OPEN_AT_END',
            'duration_min': 0,
        })

    # --- Stats ---
    closed = [t for t in trades if t['exit_reason'] != 'OPEN_AT_END']
    wins   = [t for t in closed if t['result'] == 'WIN']
    losses = [t for t in closed if t['result'] == 'LOSS']

    total_net   = sum(t['net_pnl'] for t in closed)
    total_fees  = sum(t['fees']    for t in closed)
    total_gross = sum(t['gross_pnl'] for t in closed)
    final_value = CAPITAL + total_net

    avg_win  = sum(t['net_pnl'] for t in wins)  / len(wins)  if wins  else 0
    avg_loss = sum(t['net_pnl'] for t in losses) / len(losses) if losses else 0
    pf = abs(avg_win * len(wins)) / abs(avg_loss * len(losses)) if losses and avg_loss != 0 else float('inf')

    max_dd = 0.0
    peak   = CAPITAL
    for pt in equity_curve:
        v = pt['value']
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > max_dd: max_dd = dd

    return {
        'period':       '7 days',
        'granularity':  '5-min',
        'product':      PRODUCT_ID,
        'candles':      len(candles),
        'starting_capital': CAPITAL,
        'final_value':  round(final_value, 2),
        'total_return_pct': round((final_value - CAPITAL) / CAPITAL * 100, 2),
        'total_trades': len(closed),
        'wins':         len(wins),
        'losses':       len(losses),
        'win_rate_pct': round(len(wins)/len(closed)*100, 1) if closed else 0,
        'avg_win':      round(avg_win, 2),
        'avg_loss':     round(avg_loss, 2),
        'profit_factor': round(pf, 2),
        'total_net_pnl': round(total_net, 2),
        'total_gross_pnl': round(total_gross, 2),
        'total_fees_paid': round(total_fees, 2),
        'fees_as_pct_of_capital': round(total_fees / CAPITAL * 100, 2),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'trades': trades,
    }


def print_report(results: dict):
    t = results
    print("\n" + "="*60)
    print("BACKTEST RESULTS — SOL Momentum, 5-min, Past 7 Days")
    print("="*60)
    print(f"  Candles analyzed:  {t['candles']:,}")
    print(f"  Starting capital:  ${t['starting_capital']:,.2f}")
    print(f"  Final value:       ${t['final_value']:,.2f}")
    print(f"  Total return:      {t['total_return_pct']:+.2f}%")
    print()
    print(f"  Trades (closed):   {t['total_trades']}")
    print(f"  Wins / Losses:     {t['wins']} / {t['losses']}")
    print(f"  Win rate:          {t['win_rate_pct']}%")
    print(f"  Avg win:           ${t['avg_win']:+.2f}")
    print(f"  Avg loss:          ${t['avg_loss']:+.2f}")
    print(f"  Profit factor:     {t['profit_factor']}")
    print()
    print(f"  Gross P&L:         ${t['total_gross_pnl']:+.2f}")
    print(f"  Fees paid:         ${t['total_fees_paid']:.2f} ({t['fees_as_pct_of_capital']:.2f}% of capital)")
    print(f"  Net P&L:           ${t['total_net_pnl']:+.2f}")
    print(f"  Max drawdown:      {t['max_drawdown_pct']:.2f}%")
    print()
    print("  Trade log:")
    for tr in t['trades']:
        pnl_str = f"${tr['net_pnl']:+.2f}"
        dur = f"{tr['duration_min']:.0f}min" if tr['duration_min'] else 'open'
        print(f"    {tr['result']:4s} | {tr['side']:4s} | entry={tr['entry']:.3f} "
              f"exit={tr['exit']:.3f} | net={pnl_str:>8s} | fees=${tr['fees']:.3f} "
              f"| {tr['exit_reason']:12s} | {dur}")
    print("="*60)


if __name__ == "__main__":
    client = CoinbaseClient()
    print(f"Fetching 7 days of 5-min candles for {PRODUCT_ID}...")
    candles = fetch_week_candles(client)

    # Save raw candles
    os.makedirs("data/historical", exist_ok=True)
    with open("data/historical/sol_5min_7day.json", "w") as f:
        json.dump(candles, f)

    print("Running backtest...")
    results = run_backtest(candles)

    print_report(results)

    # Save results
    os.makedirs("backtesting/results", exist_ok=True)
    with open("backtesting/results/sol_5min_7day.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to backtesting/results/sol_5min_7day.json")
