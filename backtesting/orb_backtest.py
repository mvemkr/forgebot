"""
Opening Range Breakout (ORB) Backtest — SPY / ES Strategy

Rules (from video):
  1. Mark the 8:00AM ET 15-min candle high/low (the "zone")
  2. Wait for 9:30AM ET NYSE open
  3. If price breaks + closes ABOVE zone high → LONG
     If price breaks + closes BELOW zone low  → SHORT
  4. Confirm price is respecting midpoint or zone boundary on 5-min
  5. Stop: 5 ES points ≈ $0.50 SPY
  6. Target: 1:3 R:R = 15 ES points ≈ $1.50 SPY (or London high)
  7. One trade per day max. Exit by 4PM ET if not triggered.

Capital: $2,500 allocated (half of $5,000 split)
Position sizing: risk 2% of allocation per trade = $50
  → at $0.50 stop = 100 shares per trade (tracked in $ P&L)
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time
import pytz
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────
SYMBOL           = "SPY"
ALLOCATION       = 2500.0       # capital allocated to this strategy
RISK_PCT         = 0.02         # 2% risk per trade
STOP_POINTS      = 0.50         # $0.50/share stop (≈5 ES pts on SPY scale)
RR_RATIO         = 3.0          # 1:3 R:R
TARGET_POINTS    = STOP_POINTS * RR_RATIO   # $1.50
LOOKBACK_DAYS    = 60           # yfinance 5m limit
ET               = pytz.timezone("America/New_York")
ZONE_TIME        = time(8, 0)   # 8:00AM ET candle
ENTRY_TIME       = time(9, 30)  # NYSE open
MAX_ENTRY_TIME   = time(10, 30) # stop looking for entry after this
EXIT_TIME        = time(16, 0)  # hard close all positions


def load_data(symbol: str, interval: str = "5m") -> pd.DataFrame:
    print(f"Fetching {symbol} {interval} data ({LOOKBACK_DAYS}d, including pre-market)...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{LOOKBACK_DAYS}d", interval=interval, auto_adjust=True, prepost=True)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(ET)
    df = df.sort_index()
    return df[['open','high','low','close','volume']]


def get_zone_candle(day_df: pd.DataFrame) -> tuple:
    """Get the 8:00AM 15-min candle (use first two 5-min candles as proxy)."""
    zone_candles = day_df.between_time('08:00', '08:14')
    if zone_candles.empty:
        # Fall back to pre-market first candle available
        pre = day_df.between_time('07:00', '09:29')
        zone_candles = pre.iloc[:3] if len(pre) >= 3 else pre
    if zone_candles.empty:
        return None, None, None
    zone_high = zone_candles['high'].max()
    zone_low  = zone_candles['low'].min()
    zone_mid  = (zone_high + zone_low) / 2
    return zone_high, zone_low, zone_mid


def run_backtest(df: pd.DataFrame) -> dict:
    trades = []
    balance = ALLOCATION
    risk_per_trade = ALLOCATION * RISK_PCT
    shares = round(risk_per_trade / STOP_POINTS)  # shares per trade

    trading_days = df.groupby(df.index.date)

    for date, day_df in trading_days:
        day_df = day_df.sort_index()

        # Need pre-market data for zone
        zone_high, zone_low, zone_mid = get_zone_candle(day_df)
        if zone_high is None:
            continue

        zone_range = zone_high - zone_low
        if zone_range < 0.10:  # filter micro-range days (no setup)
            continue

        # Get 9:30AM onwards for entry
        session = day_df.between_time('09:30', '16:00')
        if len(session) < 2:
            continue

        direction = None
        entry_price = None
        stop_price  = None
        target_price = None
        entry_time_actual = None
        confirmed = False

        # Filter: zone must not be too wide (noisy days → skip)
        max_zone = session.iloc[0]['close'] * 0.005   # max 0.5% of price
        if zone_range > max_zone:
            trades.append({'date': date, 'direction': None, 'result': 'zone_too_wide',
                           'pnl': 0, 'balance': balance, 'entry': None,
                           'zone_high': zone_high, 'zone_low': zone_low})
            continue

        # Scan candles from 9:30 → max entry time for breakout + confirmation
        # Stricter: need BREAKOUT candle + next candle HOLDS (pullback respects level)
        broke_high = False
        broke_low  = False
        break_price = None

        for i, (ts, candle) in enumerate(session.iterrows()):
            if ts.time() > MAX_ENTRY_TIME:
                break
            if direction is not None:
                break

            c_close = candle['close']
            c_open  = candle['open']
            c_low   = candle['low']
            c_high  = candle['high']

            # Stage 1: detect initial break with body close outside zone
            if not broke_high and not broke_low:
                if c_close > zone_high and c_open > zone_mid:
                    broke_high  = True
                    break_price = c_close
                    continue
                elif c_close < zone_low and c_open < zone_mid:
                    broke_low   = True
                    break_price = c_close
                    continue

            # Stage 2: after break — next candle must HOLD (respect) the level
            # i.e. pullback to zone boundary but NOT back inside the zone
            if broke_high:
                pullback_low = c_low
                if pullback_low >= zone_high * 0.9995 and c_close > zone_high:
                    # Held above zone — confirmed long entry
                    direction     = 'long'
                    entry_price   = c_close
                    stop_price    = zone_low   # stop below ENTIRE zone, not just $0.50
                    # Cap stop at STOP_POINTS to preserve R:R
                    stop_price    = max(stop_price, entry_price - STOP_POINTS)
                    target_price  = entry_price + (entry_price - stop_price) * RR_RATIO
                    entry_time_actual = ts
                    confirmed = True
                elif c_close < zone_high:
                    broke_high = False  # failed — back inside zone, reset

            elif broke_low:
                pullback_high = c_high
                if pullback_high <= zone_low * 1.0005 and c_close < zone_low:
                    direction     = 'short'
                    entry_price   = c_close
                    stop_price    = zone_high
                    stop_price    = min(stop_price, entry_price + STOP_POINTS)
                    target_price  = entry_price - (stop_price - entry_price) * RR_RATIO
                    entry_time_actual = ts
                    confirmed = True
                elif c_close > zone_low:
                    broke_low = False  # back inside zone, reset

        if not confirmed or direction is None:
            trades.append({
                'date': date, 'direction': None, 'result': 'no_setup',
                'pnl': 0, 'balance': balance, 'entry': None,
                'zone_high': zone_high, 'zone_low': zone_low,
            })
            continue

        # Simulate trade: scan remaining candles for SL or TP hit
        remaining = session[session.index > entry_time_actual]
        result = 'open'
        exit_price = None
        exit_time = None

        for ts2, candle2 in remaining.iterrows():
            if ts2.time() >= EXIT_TIME:
                # Hard close at EOD
                exit_price = candle2['close']
                result = 'eod_close'
                exit_time = ts2
                break

            if direction == 'long':
                if candle2['low'] <= stop_price:
                    exit_price = stop_price
                    result = 'stop_loss'
                    exit_time = ts2
                    break
                elif candle2['high'] >= target_price:
                    exit_price = target_price
                    result = 'take_profit'
                    exit_time = ts2
                    break
            else:  # short
                if candle2['high'] >= stop_price:
                    exit_price = stop_price
                    result = 'stop_loss'
                    exit_time = ts2
                    break
                elif candle2['low'] <= target_price:
                    exit_price = target_price
                    result = 'take_profit'
                    exit_time = ts2
                    break

        if exit_price is None:
            exit_price = remaining.iloc[-1]['close'] if not remaining.empty else entry_price
            result = 'eod_close'

        # P&L
        if direction == 'long':
            pnl = (exit_price - entry_price) * shares
        else:
            pnl = (entry_price - exit_price) * shares

        balance += pnl
        trades.append({
            'date':       date,
            'direction':  direction,
            'result':     result,
            'entry':      entry_price,
            'exit':       exit_price,
            'stop':       stop_price,
            'target':     target_price,
            'zone_high':  zone_high,
            'zone_low':   zone_low,
            'zone_range': zone_range,
            'shares':     shares,
            'pnl':        round(pnl, 2),
            'balance':    round(balance, 2),
            'entry_time': entry_time_actual,
        })

    return {'trades': trades, 'final_balance': balance}


def print_results(results: dict, symbol: str):
    trades = pd.DataFrame(results['trades'])
    if trades.empty or 'direction' not in trades.columns:
        print("No trades generated. Check data availability.")
        return
    actual = trades[trades['direction'].notna()].copy()
    no_setup = trades[trades['direction'].isna()]

    wins  = actual[actual['result'] == 'take_profit']
    losses = actual[actual['result'] == 'stop_loss']
    eod   = actual[actual['result'] == 'eod_close']

    total_pnl   = actual['pnl'].sum()
    win_rate    = len(wins) / len(actual) * 100 if len(actual) else 0
    avg_win     = wins['pnl'].mean() if len(wins) else 0
    avg_loss    = losses['pnl'].mean() if len(losses) else 0
    profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() != 0 else float('inf')
    max_dd      = _max_drawdown(actual['balance'].tolist(), ALLOCATION)
    sharpe      = _sharpe(actual['pnl'].tolist())

    print("\n" + "="*65)
    print(f"  ORB BACKTEST — {symbol}  |  {LOOKBACK_DAYS}-day window")
    print("="*65)
    print(f"  Capital allocated : ${ALLOCATION:,.0f}")
    print(f"  Risk per trade    : ${ALLOCATION*RISK_PCT:.0f} ({RISK_PCT*100:.0f}%)")
    print(f"  Shares per trade  : {actual['shares'].iloc[0] if len(actual) else 0}")
    print(f"  Stop / Target     : ${STOP_POINTS:.2f} / ${TARGET_POINTS:.2f}  (1:{RR_RATIO:.0f})")
    print("-"*65)
    print(f"  Trading days      : {len(trades)}")
    print(f"  Days with setup   : {len(actual)}  ({len(no_setup)} no-setup days)")
    print(f"  Total trades      : {len(actual)}")
    print(f"    ✅ Take profit  : {len(wins)}")
    print(f"    ❌ Stop loss    : {len(losses)}")
    print(f"    🔄 EOD close    : {len(eod)}")
    print("-"*65)
    print(f"  Win rate          : {win_rate:.1f}%")
    print(f"  Avg win           : ${avg_win:,.2f}")
    print(f"  Avg loss          : ${avg_loss:,.2f}")
    print(f"  Profit factor     : {profit_factor:.2f}")
    print(f"  Sharpe (daily)    : {sharpe:.2f}")
    print(f"  Max drawdown      : {max_dd:.1f}%")
    print("-"*65)
    print(f"  Starting balance  : ${ALLOCATION:,.2f}")
    print(f"  Final balance     : ${results['final_balance']:,.2f}")
    print(f"  Net P&L           : ${total_pnl:+,.2f}  ({total_pnl/ALLOCATION*100:+.1f}%)")
    print("="*65)

    # Monthly breakdown
    if len(actual):
        actual['month'] = pd.to_datetime(actual['date']).dt.to_period('M')
        monthly = actual.groupby('month').agg(
            trades=('pnl','count'),
            pnl=('pnl','sum'),
            wins=('result', lambda x: (x=='take_profit').sum())
        )
        monthly['win_rate'] = (monthly['wins']/monthly['trades']*100).round(1)
        print("\n  Monthly Breakdown:")
        print(f"  {'Month':<10} {'Trades':>7} {'Wins':>6} {'WR%':>6} {'P&L':>10}")
        print("  " + "-"*45)
        for period, row in monthly.iterrows():
            print(f"  {str(period):<10} {row['trades']:>7} {row['wins']:>6} {row['win_rate']:>5.1f}% {row['pnl']:>+9.2f}")
        print()

    # Worst trades
    if len(actual) >= 5:
        print("  5 Worst Trades:")
        worst = actual.nsmallest(5, 'pnl')[['date','direction','result','entry','exit','pnl']]
        for _, r in worst.iterrows():
            print(f"    {r['date']} {r['direction']:5} {r['result']:12} entry={r['entry']:.2f} exit={r['exit']:.2f}  P&L=${r['pnl']:+.2f}")
    print()


def _max_drawdown(balances: list, start: float) -> float:
    if not balances:
        return 0
    peak = start
    max_dd = 0
    for b in balances:
        if b > peak:
            peak = b
        dd = (peak - b) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe(pnls: list, rf: float = 0) -> float:
    if len(pnls) < 2:
        return 0
    arr = np.array(pnls)
    return (arr.mean() - rf) / arr.std() * np.sqrt(252) if arr.std() > 0 else 0


if __name__ == "__main__":
    df = load_data(SYMBOL, "5m")
    print(f"Loaded {len(df)} 5-min candles spanning {df.index[0].date()} → {df.index[-1].date()}")
    results = run_backtest(df)
    print_results(results, SYMBOL)
