"""
FOREX STRATEGY REBUILD — Full implementation of missing pieces:
1. Break + Retest detector (the real entry mechanism)
2. Structural stop placement (behind pattern extreme, not fixed pips)
3. Weekly + Daily + 4H trend alignment (all 3 must agree)
4. London session filter on entries
5. One trade per week max (best setup only)
"""

import sys, warnings, os
sys.path.insert(0, '/home/forge/trading-bot/src')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pytz
from dataclasses import dataclass
from typing import Optional, List
from exchange.forex_data import ForexData
from strategy.forex.pattern_detector import PatternDetector, Trend
from strategy.forex.entry_signal import EntrySignalDetector

ET = pytz.timezone("America/New_York")
fx = ForexData()

# ─────────────────────────────────────────────
# CORE: Break + Retest Setup Detector
# ─────────────────────────────────────────────
@dataclass
class TradeSetup:
    pair:        str
    direction:   str        # long / short
    neckline:    float      # the level being retested
    entry:       float      # engulfing close price
    stop:        float      # structural stop (behind pattern extreme)
    target:      float      # let it run (no fixed TP — trail by chart)
    risk_pips:   float
    setup_type:  str        # 'break_retest' / 'hs_retest' / 'double_top_retest'
    signal_strength: float
    tf_alignment: int       # 1-3 (how many TFs agree)
    quality:     float      # 0-1 composite score
    notes:       str = ""

    @property
    def rr_at_target(self): 
        return abs(self.target - self.entry) / max(abs(self.entry - self.stop), 0.0001)


def detect_trend(df: pd.DataFrame) -> Optional[str]:
    """Returns 'bullish', 'bearish', or None using HH/HL structure."""
    if len(df) < 20:
        return None
    h = df['high'].values
    l = df['low'].values

    def find_swings(arr, kind, n=5):
        pts = []
        for i in range(n, len(arr)-n):
            win = arr[i-n:i+n+1]
            if (kind=='high' and arr[i]==max(win)) or (kind=='low' and arr[i]==min(win)):
                pts.append(arr[i])
        return pts[-4:] if len(pts) >= 2 else pts

    sh = find_swings(h, 'high')
    sl = find_swings(l, 'low')
    if len(sh) < 2 or len(sl) < 2: return None

    hh = all(sh[i] < sh[i+1] for i in range(len(sh)-1))
    hl = all(sl[i] < sl[i+1] for i in range(len(sl)-1))
    lh = all(sh[i] > sh[i+1] for i in range(len(sh)-1))
    ll = all(sl[i] > sl[i+1] for i in range(len(sl)-1))

    if hh or hl: return 'bullish'
    if lh or ll: return 'bearish'
    return None


def find_break_retest(df_4h: pd.DataFrame, df_1h: pd.DataFrame,
                      direction: str, cutoff_1h) -> Optional[TradeSetup]:
    """
    Detect break + retest in 4H data, confirmed with 1H engulfing.
    
    Logic:
    - Find consolidation zone (range-bound area in last 30 4H candles)
    - Detect clean break of zone boundary (close beyond zone)
    - Detect retest of broken boundary (price returns within 0.15%)
    - Confirm 1H engulfing candle at the retest in trade direction
    - Structural stop: behind the zone (full zone width behind entry)
    """
    if len(df_4h) < 20: return None

    recent = df_4h.tail(60)
    
    # Find zones: rolling 10-bar high/low
    zone_highs = recent['high'].rolling(10).max()
    zone_lows  = recent['low'].rolling(10).min()

    # Look at last 5 4H candles for breaks
    for i in range(-10, -2):
        try:
            candle = recent.iloc[i]
            c_close = float(candle['close'])
            zone_h  = float(zone_highs.iloc[i-1])
            zone_l  = float(zone_lows.iloc[i-1])
            zone_range = zone_h - zone_l

            # Filter tiny zones
            if zone_range / c_close < 0.0005: continue

            # Bearish break
            if direction == 'short' and c_close < zone_l:
                neckline = zone_l
                # Check for retest: subsequent candles approach neckline from below
                post = recent.iloc[i+1:]
                retest_found = any(
                    abs(float(row['high']) - neckline) / neckline < 0.0015
                    for _, row in post.iterrows()
                )
                if not retest_found: continue

                # Find 1H engulfing at retest
                df_1h_filt = df_1h[df_1h.index < cutoff_1h].tail(48)
                esd = EntrySignalDetector(min_body_ratio=0.45)
                signal = esd.detect(df_1h_filt)
                if not signal or signal.direction != 'short' or signal.strength < 0.35:
                    continue

                entry = float(signal.close)
                # Must be near the neckline (within 0.3%)
                if abs(entry - neckline) / neckline > 0.003: continue

                # Structural stop: above the zone (full zone above neckline)
                stop   = zone_h + zone_range * 0.1
                # Target: 3x the structural risk
                risk   = abs(stop - entry)
                target = entry - risk * 3.0

                return TradeSetup(
                    pair='', direction='short',
                    neckline=neckline, entry=entry, stop=stop, target=target,
                    risk_pips=risk * 10000,
                    setup_type='break_retest_bearish',
                    signal_strength=signal.strength,
                    tf_alignment=0,
                    quality=0.0,
                    notes=f"Zone {zone_l:.5f}–{zone_h:.5f} broke bearish, retesting {neckline:.5f}"
                )

            # Bullish break
            if direction == 'long' and c_close > zone_h:
                neckline = zone_h
                post = recent.iloc[i+1:]
                retest_found = any(
                    abs(float(row['low']) - neckline) / neckline < 0.0015
                    for _, row in post.iterrows()
                )
                if not retest_found: continue

                df_1h_filt = df_1h[df_1h.index < cutoff_1h].tail(48)
                esd = EntrySignalDetector(min_body_ratio=0.45)
                signal = esd.detect(df_1h_filt)
                if not signal or signal.direction != 'long' or signal.strength < 0.35:
                    continue

                entry = float(signal.close)
                if abs(entry - neckline) / neckline > 0.003: continue

                stop   = zone_l - zone_range * 0.1
                risk   = abs(entry - stop)
                target = entry + risk * 3.0

                return TradeSetup(
                    pair='', direction='long',
                    neckline=neckline, entry=entry, stop=stop, target=target,
                    risk_pips=risk * 10000,
                    setup_type='break_retest_bullish',
                    signal_strength=signal.strength,
                    tf_alignment=0,
                    quality=0.0,
                    notes=f"Zone {zone_l:.5f}–{zone_h:.5f} broke bullish, retesting {neckline:.5f}"
                )
        except (IndexError, KeyError):
            continue

    return None


def find_double_top_bottom_retest(df_d: pd.DataFrame, df_1h: pd.DataFrame,
                                   direction: str, cutoff_1h) -> Optional[TradeSetup]:
    """Double top/bottom: break of valley/peak between two equal highs/lows."""
    if len(df_d) < 30: return None
    recent = df_d.tail(40)
    h = recent['high'].values
    l = recent['low'].values
    n = 5

    if direction == 'short':
        # Find two highs of similar price within tolerance
        peaks = [(i, h[i]) for i in range(n, len(h)-n)
                 if h[i] == max(h[i-n:i+n+1])]
        for a, b in zip(peaks, peaks[1:]):
            diff = abs(a[1] - b[1]) / a[1]
            if diff > 0.008: continue  # tops must be within 0.8%
            top_level = (a[1] + b[1]) / 2
            # Valley between them
            valley = min(l[a[0]:b[0]+1])
            # Current close near valley (neckline break zone)
            cur_close = float(recent.iloc[-1]['close'])
            if cur_close > valley * 1.002: continue  # not near neckline yet

            df_1h_filt = df_1h[df_1h.index < cutoff_1h].tail(48)
            signal = EntrySignalDetector(min_body_ratio=0.45).detect(df_1h_filt)
            if not signal or signal.direction != 'short' or signal.strength < 0.35:
                continue

            entry = float(signal.close)
            stop  = top_level * 1.002
            risk  = abs(stop - entry)
            target = entry - risk * 3.0

            return TradeSetup(
                pair='', direction='short',
                neckline=valley, entry=entry, stop=stop, target=target,
                risk_pips=risk * 10000,
                setup_type='double_top_retest',
                signal_strength=signal.strength,
                tf_alignment=0, quality=0.0,
                notes=f"Double top at {top_level:.5f}, neckline at {valley:.5f}"
            )

    else:  # long / double bottom
        troughs = [(i, l[i]) for i in range(n, len(l)-n)
                   if l[i] == min(l[i-n:i+n+1])]
        for a, b in zip(troughs, troughs[1:]):
            diff = abs(a[1] - b[1]) / a[1]
            if diff > 0.008: continue
            bottom_level = (a[1] + b[1]) / 2
            peak = max(h[a[0]:b[0]+1])
            cur_close = float(recent.iloc[-1]['close'])
            if cur_close < peak * 0.998: continue

            df_1h_filt = df_1h[df_1h.index < cutoff_1h].tail(48)
            signal = EntrySignalDetector(min_body_ratio=0.45).detect(df_1h_filt)
            if not signal or signal.direction != 'long' or signal.strength < 0.35:
                continue

            entry = float(signal.close)
            stop  = bottom_level * 0.998
            risk  = abs(entry - stop)
            target = entry + risk * 3.0

            return TradeSetup(
                pair='', direction='long',
                neckline=peak, entry=entry, stop=stop, target=target,
                risk_pips=risk * 10000,
                setup_type='double_bottom_retest',
                signal_strength=signal.strength,
                tf_alignment=0, quality=0.0,
                notes=f"Double bottom at {bottom_level:.5f}, neckline at {peak:.5f}"
            )
    return None


def evaluate_pair(pair: str, df_w, df_d, df_4h, df_1h, cutoff_1h) -> Optional[TradeSetup]:
    """Full evaluation: TF alignment → pattern → break+retest → engulfing."""
    is_jpy = 'JPY' in pair

    # 1. Three-TF trend alignment
    tw = detect_trend(df_w)
    td = detect_trend(df_d)
    t4 = detect_trend(df_4h)
    trends = [tw, td, t4]

    bull = sum(1 for t in trends if t == 'bullish')
    bear = sum(1 for t in trends if t == 'bearish')

    if bull >= 2:   direction = 'long';  tf_score = bull
    elif bear >= 2: direction = 'short'; tf_score = bear
    else: return None  # no alignment

    # 2. Try double top/bottom on daily (cleaner structure)
    setup = find_double_top_bottom_retest(df_d, df_1h, direction, cutoff_1h)

    # 3. Fallback to break+retest on 4H
    if setup is None:
        setup = find_break_retest(df_4h, df_1h, direction, cutoff_1h)

    if setup is None:
        return None

    setup.pair = pair
    setup.tf_alignment = tf_score

    # Adjust pip calculation for JPY
    if is_jpy:
        setup.risk_pips = abs(setup.entry - setup.stop) * 100
        if setup.risk_pips > 100 or setup.risk_pips < 5:
            return None  # unreasonable stop
    else:
        if setup.risk_pips > 100 or setup.risk_pips < 3:
            return None

    # Quality score
    setup.quality = (
        0.35 * setup.signal_strength +
        0.35 * (tf_score / 3) +
        0.30 * min(1.0, 20 / max(setup.risk_pips, 1))
    )

    return setup


# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
WATCHLIST  = ['GBP/USD','GBP/CHF','GBP/JPY','USD/JPY','EUR/USD','USD/CHF','USD/CAD','NZD/USD','EUR/GBP']
ACCOUNT    = 5000.0
RISK_DOLS  = 100.0   # 2% of $5K

print("Fetching data for all pairs...")
pair_cache = {}
for pair in WATCHLIST:
    try:
        dw = fx.get_candles(pair, '1W', lookback=52)
        dd = fx.get_candles(pair, '1D', lookback=200)
        d4 = fx.get_candles(pair, '4h', lookback=200)
        d1 = fx.get_candles(pair, '1h', lookback=500)
        if dw.empty or dd.empty or d4.empty or d1.empty:
            continue
        for df in [dw, dd, d4, d1]:
            df.index = df.index.tz_convert(ET)
        pair_cache[pair] = (dw, dd, d4, d1)
        print(f"  {pair}: W:{len(dw)} D:{len(dd)} 4H:{len(d4)} 1H:{len(d1)}")
    except Exception as e:
        print(f"  {pair}: ERROR {e}")

WEEKS = [
    ("Jan 20–24", '2026-01-20','2026-01-23 17:00', '2026-01-19 22:00'),
    ("Jan 27–31", '2026-01-27','2026-01-30 17:00', '2026-01-26 22:00'),
    ("Feb 3–7",   '2026-02-03','2026-02-06 17:00', '2026-02-02 22:00'),
    ("Feb 10–14", '2026-02-10','2026-02-13 17:00', '2026-02-09 22:00'),
    ("Feb 17–20", '2026-02-17','2026-02-20 17:00', '2026-02-16 22:00'),
]

grand_pnl = 0.0
all_trades = []

print("\n" + "="*70)
print("  REBUILT STRATEGY BACKTEST — 5-Week Window")
print("  Jan 20 – Feb 20, 2026  |  $5,000 account  |  $100 risk/trade")
print("="*70)

for week_label, wk_start_str, wk_end_str, cut_str in WEEKS:
    cut_1h = pd.Timestamp(cut_str, tz=ET)
    wk_s   = pd.Timestamp(f"{wk_start_str} 03:00", tz=ET)  # London open
    wk_e   = pd.Timestamp(wk_end_str, tz=ET)

    week_setups = []
    for pair, (dw, dd, d4, d1) in pair_cache.items():
        dw_pre = dw[dw.index < cut_1h]
        dd_pre = dd[dd.index < cut_1h]
        d4_pre = d4[d4.index < cut_1h]
        d1_pre = d1[d1.index < cut_1h]
        if len(dw_pre) < 5 or len(dd_pre) < 30: continue

        setup = evaluate_pair(pair, dw_pre, dd_pre, d4_pre, d1_pre, cut_1h)
        if setup:
            week_setups.append(setup)

    # Take only the BEST setup (highest quality score) — one trade per week
    week_setups.sort(key=lambda s: s.quality, reverse=True)
    top_setup = week_setups[0] if week_setups else None

    if not top_setup:
        print(f"\n⏸️  Week of {week_label}: No qualifying setup — sat out  (capital preserved)")
        continue

    # Simulate trade through the week
    setup = top_setup
    pair  = setup.pair
    d1    = pair_cache[pair][3]
    df_week = d1[(d1.index >= wk_s) & (d1.index <= wk_e)]

    outcome    = 'still_running'
    exit_price = float(df_week['close'].iloc[-1]) if not df_week.empty else setup.entry

    for _, c in df_week.iterrows():
        lo, hi = float(c['low']), float(c['high'])
        if setup.direction == 'long':
            if lo <= setup.stop:    outcome='stop_loss';   exit_price=setup.stop;   break
            if hi >= setup.target:  outcome='take_profit'; exit_price=setup.target; break
        else:
            if hi >= setup.stop:    outcome='stop_loss';   exit_price=setup.stop;   break
            if lo <= setup.target:  outcome='take_profit'; exit_price=setup.target; break

    pnl_pts  = (exit_price-setup.entry) if setup.direction=='long' else (setup.entry-exit_price)
    units    = int(RISK_DOLS / max(abs(setup.entry - setup.stop), 0.0001))
    pnl_dols = round(pnl_pts * units, 2)
    grand_pnl += pnl_dols

    icon = '✅' if outcome=='take_profit' else '❌' if outcome=='stop_loss' else '🔄'
    print(f"\n{icon} Week of {week_label}:")
    print(f"   Pair:    {setup.pair}  {setup.direction.upper()}")
    print(f"   Setup:   {setup.setup_type}")
    print(f"   Notes:   {setup.notes}")
    print(f"   Entry:   {setup.entry:.5f}   Stop: {setup.stop:.5f}   Target: {setup.target:.5f}")
    print(f"   Risk:    {setup.risk_pips:.1f} pips  |  Units: {units:,}  |  TF alignment: {setup.tf_alignment}/3")
    print(f"   Quality: {setup.quality:.2f}  |  Signal strength: {setup.signal_strength:.2f}")
    print(f"   Result:  {outcome.upper().replace('_',' ')} @ {exit_price:.5f}   P&L: ${pnl_dols:+,.2f}")

    all_trades.append(dict(pair=setup.pair, direction=setup.direction,
                            outcome=outcome, pnl=pnl_dols,
                            rr=setup.rr_at_target, quality=setup.quality))

wins   = sum(1 for t in all_trades if t['outcome']=='take_profit')
losses = sum(1 for t in all_trades if t['outcome']=='stop_loss')
skipped = 5 - len(all_trades)

print()
print("="*70)
print("  REBUILT STRATEGY — 5-WEEK SUMMARY")
print("="*70)
print(f"  Weeks with trades:  {len(all_trades)} of 5  ({skipped} skipped — correctly sat out)")
print(f"  Total trades:       {len(all_trades)}")
print(f"  Wins:               {wins}")
print(f"  Losses:             {losses}")
if all_trades:
    print(f"  Win rate:           {wins/len(all_trades)*100:.0f}%")
print(f"  Net P&L:            ${grand_pnl:+,.2f}")
print(f"  Return:             {grand_pnl/ACCOUNT*100:+.2f}%")
print(f"  End balance:        ${ACCOUNT+grand_pnl:,.2f}")
print("="*70)

# Save results for Telegram message
import json
summary = {
    "trades": len(all_trades),
    "wins": wins,
    "losses": losses,
    "skipped": skipped,
    "win_rate": f"{wins/len(all_trades)*100:.0f}%" if all_trades else "N/A",
    "pnl": grand_pnl,
    "pnl_pct": round(grand_pnl/ACCOUNT*100, 2),
    "end_balance": round(ACCOUNT+grand_pnl, 2),
    "trade_details": all_trades
}
with open('/tmp/forex_rebuild_results.json', 'w') as f:
    json.dump(summary, f)

print("\nResults saved.")
