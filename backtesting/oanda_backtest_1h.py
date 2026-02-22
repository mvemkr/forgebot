"""
OANDA 1H Backtester — Set & Forget v3 Strategy
Jan 1, 2026 → Feb 21, 2026

Uses real OANDA 1H candle data (same source the live bot uses).
This is the proper backtester — same data, same logic, same exits.

v3 RULES (pattern-first, not trend-first):
  1. Double bottom/top at psychological level
  2. Neckline broken + price retesting
  3. 1H engulfing candle at the retest → ENTER
  4. Stop behind pattern structural extreme
  5. Move to breakeven at 1:1
  6. Exit: strong daily reversal at psych level OR max 30 days
  7. NO take profit ever
  8. One trade at a time
"""
import sys, json, time, requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.exchange.oanda_client import OandaClient, INSTRUMENT_MAP
from src.strategy.forex.entry_signal import EntrySignalDetector

# ── Config ──────────────────────────────────────────────────────────────
BACKTEST_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
BACKTEST_END   = datetime(2026, 2, 21, tzinfo=timezone.utc)
STARTING_BAL   = 4_000.0

WATCHLIST = [
    "GBP/USD", "USD/JPY", "EUR/USD", "USD/CHF",
    "USD/CAD", "GBP/JPY", "GBP/CHF", "GBP/NZD",
    "NZD/USD", "AUD/USD", "EUR/GBP", "NZD/JPY",
]

RISK_TIERS = [
    (8_000,        10.0),
    (15_000,       15.0),
    (30_000,       20.0),
    (float("inf"), 25.0),
]

# Detection thresholds
DOUBLE_SIMILARITY    = 0.006   # tops/bottoms within 0.6% of each other
MIN_PATTERN_BARS_4H  = 4       # min 4H bars between pattern peaks
MAX_PATTERN_BARS_4H  = 40
PSYCH_TOLERANCE      = 0.010   # within 1% of a round number
MIN_SIGNAL_STRENGTH  = 0.38

# Exit (set & forget — Alex held through noise, only exited on DAILY exhaustion)
EXIT_STRENGTH_MIN     = 0.75     # raised from 0.55 — requires a strong daily reversal candle
EXIT_PSYCH_TOLERANCE  = 0.010    # exit signal must be near a round number
EXIT_LEVEL_MIN_SEP    = 0.015    # exit level must be ≥1.5% away from entry level (different major level)
MIN_EXCURSION_FOR_EXIT = 1.5     # price must have traveled at least 1.5R before any exit fires
MAX_HOLD_BARS_1H      = 30 * 24  # 30 days hard cap
MIN_HOLD_HOURS        = 72       # raised from 48h — give the trade 3 days to establish
EXIT_CHECK_DAILY_ONLY = True     # only fire exit signals when a new DAILY candle closes

# ── Helpers ──────────────────────────────────────────────────────────────

def get_risk_pct(balance):
    for cap, pct in RISK_TIERS:
        if balance < cap:
            return pct
    return 20.0

def is_jpy(pair):
    return "JPY" in pair

def calc_units(balance, risk_pct, entry, stop):
    risk_dollars = balance * (risk_pct / 100)
    dist = abs(entry - stop)
    if dist == 0: return 1000
    units = int(risk_dollars / dist)
    if is_jpy(pair if (pair := "") else ""):
        pass  # simplified
    return max(1000, min(units, 100_000))

def trade_pnl(pair, direction, entry, exit_p, units):
    delta = (exit_p - entry) if direction == "long" else (entry - exit_p)
    if is_jpy(pair):
        return round(delta * units / exit_p, 2)
    return round(delta * units, 2)

def get_round_levels(price, tolerance=PSYCH_TOLERANCE):
    levels = []
    incs = [1.0, 0.5] if price > 50 else [0.0100, 0.0050]
    for inc in incs:
        r = round(round(price / inc) * inc, 5)
        if abs(r - price) / price <= tolerance:
            levels.append(r)
    return levels

def is_london_or_ny(dt_utc):
    """London: 08:00-17:00 UTC. NY: 13:00-22:00 UTC."""
    h = dt_utc.hour
    return 7 <= h <= 21

def detect_double_bottom(df, min_bars=MIN_PATTERN_BARS_4H, max_bars=MAX_PATTERN_BARS_4H):
    lows = df["low"].values; highs = df["high"].values; n = len(lows)
    best = None
    for i in range(5, n - 5):
        if lows[i] != min(lows[max(0,i-5):i+6]): continue
        for j in range(i+min_bars, min(n, i+max_bars)):
            if lows[j] != min(lows[max(0,j-5):j+6]): continue
            avg = (lows[i]+lows[j])/2
            if abs(lows[i]-lows[j])/avg > DOUBLE_SIMILARITY: continue
            neckline = max(highs[i:j+1])
            stop_loss = avg * (0.993 if not is_jpy("X") else 0.998)
            best = (avg, neckline, round(stop_loss, 5))
    return best

def detect_double_top(df, min_bars=MIN_PATTERN_BARS_4H, max_bars=MAX_PATTERN_BARS_4H):
    highs = df["high"].values; lows = df["low"].values; n = len(highs)
    best = None
    for i in range(5, n - 5):
        if highs[i] != max(highs[max(0,i-5):i+6]): continue
        for j in range(i+min_bars, min(n, i+max_bars)):
            if highs[j] != max(highs[max(0,j-5):j+6]): continue
            avg = (highs[i]+highs[j])/2
            if abs(highs[i]-highs[j])/avg > DOUBLE_SIMILARITY: continue
            neckline = min(lows[i:j+1])
            stop_loss = avg * 1.007
            best = (avg, neckline, round(stop_loss, 5))
    return best

# ── Data Fetch ────────────────────────────────────────────────────────────

oanda = OandaClient()

def fetch_candles_from(pair, granularity, count=2000):
    """Fetch candles from OANDA as a DataFrame."""
    instrument = INSTRUMENT_MAP.get(pair, pair.replace("/","_"))
    try:
        resp = requests.get(
            f"{oanda.base}/v3/instruments/{instrument}/candles",
            headers=oanda.headers,
            params={"granularity": granularity, "count": count, "price": "M"},
            timeout=15,
        )
        data = resp.json().get("candles", [])
        rows = []
        for c in data:
            mid = c.get("mid", {})
            rows.append({
                "time":   pd.Timestamp(c["time"]).tz_localize(None),
                "open":   float(mid.get("o", 0)),
                "high":   float(mid.get("h", 0)),
                "low":    float(mid.get("l", 0)),
                "close":  float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })
        df = pd.DataFrame(rows).set_index("time")
        return df
    except Exception as e:
        print(f"  ✗ {pair}/{granularity}: {e}")
        return None

print("Fetching OANDA historical candles (this takes ~30s)...")
candle_data = {}   # pair → {"1h": df, "4h": df, "d": df}

for pair in WATCHLIST:
    df_1h = fetch_candles_from(pair, "H1", 2000)
    df_4h = fetch_candles_from(pair, "H4", 800)
    df_d  = fetch_candles_from(pair, "D",  200)
    if df_1h is not None and len(df_1h) > 100:
        candle_data[pair] = {"1h": df_1h, "4h": df_4h, "d": df_d}
        print(f"  ✓ {pair}: {len(df_1h)} 1H | {len(df_4h) if df_4h is not None else 0} 4H | {len(df_d) if df_d is not None else 0} D")
    else:
        print(f"  ✗ {pair}: insufficient data")
    time.sleep(0.3)  # rate limit

print(f"\nPairs loaded: {len(candle_data)}\n")

# ── Backtester ──────────────────────────────────────────────────────────

signal_det = EntrySignalDetector(min_body_ratio=0.40, lookback_candles=3)
balance    = STARTING_BAL
trades     = []
open_pos   = {}
be_moved   = set()
decision_log = []  # full audit trail

# Pattern memory: keyed by (pair, pattern_type, rounded_neckline)
# Value: 'active' or 'exhausted'. Once a pattern is traded and exits,
# it's marked exhausted so we never re-enter the same formation.
# A new pattern at a materially different level is a fresh key → allowed.
NECKLINE_CLUSTER_PCT   = 0.003   # necklines within 0.3% = same pattern
CONSECUTIVE_EXITS_REQ  = 2       # require N consecutive daily reversal closes before exiting
traded_patterns: dict = {}     # key → 'exhausted'

def pattern_key(pair, pattern_type, neckline):
    """Round neckline to nearest cluster bucket so nearby levels merge."""
    bucket = round(neckline / (neckline * NECKLINE_CLUSTER_PCT)) * (neckline * NECKLINE_CLUSTER_PCT)
    return f"{pair}|{pattern_type}|{bucket:.5f}"

def already_traded(pair, pattern_type, neckline):
    key = pattern_key(pair, pattern_type, neckline)
    return traded_patterns.get(key) == 'exhausted'

def mark_pattern_exhausted(pair, pattern_type, neckline):
    key = pattern_key(pair, pattern_type, neckline)
    traded_patterns[key] = 'exhausted'
    print(f"  🔒 Pattern locked: {key}")

# Build hourly timeline from backtest window
all_1h_times = sorted(set(
    ts for pair, pdata in candle_data.items()
    for ts in pdata["1h"].index
    if BACKTEST_START.replace(tzinfo=None) <= ts <= BACKTEST_END.replace(tzinfo=None)
))

print(f"Backtesting {len(all_1h_times)} hourly bars: "
      f"{BACKTEST_START.date()} → {BACKTEST_END.date()}")
print(f"Starting balance: ${balance:,.2f}\n")

def evaluate_entry_1h(pair, ts):
    """Evaluate a potential v3 entry at timestamp ts using data up to ts."""
    pdata  = candle_data[pair]
    df_4h  = pdata["4h"]
    df_1h  = pdata["1h"]
    df_d   = pdata["d"]

    # Slice history up to (not including) this timestamp
    hist_4h = df_4h[df_4h.index < ts]
    hist_1h = df_1h[df_1h.index < ts]
    hist_d  = df_d[df_d.index < ts] if df_d is not None else None

    if len(hist_4h) < 20 or len(hist_1h) < 20:
        return None

    current_price = hist_1h["close"].iloc[-1]

    # Pattern detection on 4H (better resolution than daily)
    # Try double bottom (long)
    result_db = detect_double_bottom(hist_4h.iloc[-80:])
    pip = 0.01 if is_jpy(pair) else 0.0001

    if result_db:
        bottom_level, neckline, _orig_stop = result_db
        psych = get_round_levels(bottom_level, PSYCH_TOLERANCE)
        if psych:
            # LONG: entry must be ABOVE neckline (confirmed neckline break + retest from above)
            # Price bounced off neckline → enter long, stop below neckline
            above_neck = neckline <= current_price <= neckline * 1.015
            if above_neck:
                has_sig, sig = signal_det.has_signal(hist_1h.iloc[-5:], "long")
                if has_sig and sig and sig.strength >= MIN_SIGNAL_STRENGTH:
                    tight_stop = round(neckline - pip * 15, 5)  # 15 pips BELOW neckline
                    # Sanity check: stop must be below entry
                    if tight_stop >= current_price:
                        return None
                    return {
                        "direction": "long", "entry_price": current_price,
                        "stop_loss": tight_stop, "neckline": neckline,
                        "pattern": "double_bottom", "psych_level": psych[0],
                        "signal_strength": sig.strength, "pattern_level": bottom_level,
                        "confidence": 0.4 + sig.strength * 0.35 + 0.25 * (len(psych) / 2),
                        "notes": f"DB @ {bottom_level:.5f} (psych {psych[0]}) neckline {neckline:.5f}",
                        "consecutive_exit_signals": 0,
                    }

    # Try double top (short)
    result_dt = detect_double_top(hist_4h.iloc[-80:])
    if result_dt:
        top_level, neckline, _orig_stop = result_dt
        psych = get_round_levels(top_level, PSYCH_TOLERANCE)
        if psych:
            # SHORT: entry must be BELOW neckline (confirmed neckline break + retest from below)
            # Price rejected at neckline from below → enter short, stop above neckline
            below_neck = neckline * 0.985 <= current_price <= neckline
            if below_neck:
                has_sig, sig = signal_det.has_signal(hist_1h.iloc[-5:], "short")
                if has_sig and sig and sig.strength >= MIN_SIGNAL_STRENGTH:
                    tight_stop = round(neckline + pip * 15, 5)  # 15 pips ABOVE neckline
                    # Sanity check: stop must be above entry
                    if tight_stop <= current_price:
                        return None
                    return {
                        "direction": "short", "entry_price": current_price,
                        "stop_loss": tight_stop, "neckline": neckline,
                        "pattern": "double_top", "psych_level": psych[0],
                        "signal_strength": sig.strength, "pattern_level": top_level,
                        "confidence": 0.4 + sig.strength * 0.35 + 0.25 * (len(psych) / 2),
                        "notes": f"DT @ {top_level:.5f} (psych {psych[0]}) neckline {neckline:.5f}",
                        "consecutive_exit_signals": 0,
                    }
    return None

prev_pct = -1
for bar_idx, ts in enumerate(all_1h_times):
    pct = int(bar_idx / len(all_1h_times) * 100)
    if pct != prev_pct and pct % 10 == 0:
        print(f"  {pct}%... (balance ${balance:,.2f}, trades {len(trades)})")
        prev_pct = pct

    ts_utc = ts.to_pydatetime().replace(tzinfo=timezone.utc)

    # ── Monitor open positions ────────────────────────────────────────
    for pair in list(open_pos.keys()):
        pdata = candle_data.get(pair)
        if pdata is None: continue
        df_1h = pdata["1h"]

        row_mask = df_1h.index == ts
        if not row_mask.any(): continue
        bar = df_1h.loc[row_mask].iloc[0]
        high = float(bar["high"]); low = float(bar["low"]); close = float(bar["close"])

        pos       = open_pos[pair]
        entry     = pos["entry_price"]
        stop      = pos["stop_loss"]
        direction = pos["direction"]
        risk      = abs(entry - stop)
        pos_key   = f"{pair}_{pos['entry_ts']}"
        bars_held = bar_idx - pos["bar_idx"]

        # Stop hit?
        if (direction == "long" and low <= stop) or (direction == "short" and high >= stop):
            p = trade_pnl(pair, direction, entry, stop, pos["units"])
            balance += p
            rr = -1.0
            trades.append({**pos, "exit_ts": str(ts), "exit_price": stop,
                           "exit_reason": "stop_hit", "pnl": p, "rr": rr,
                           "bars_held": bars_held})
            decision_log.append({"ts": str(ts), "pair": pair, "event": "STOP_HIT",
                                  "price": stop, "pnl": p})
            # Pattern stopped out → this formation is exhausted, don't retry it
            mark_pattern_exhausted(pair, pos.get("pattern",""), pos.get("neckline", stop))
            del open_pos[pair]; be_moved.discard(pos_key)
            continue

        # Breakeven at 1:1
        if pos_key not in be_moved and risk > 0:
            at_1r = (direction=="long" and close >= entry+risk) or \
                    (direction=="short" and close <= entry-risk)
            if at_1r:
                pos["stop_loss"] = entry
                be_moved.add(pos_key)
                decision_log.append({"ts": str(ts), "pair": pair, "event": "BREAKEVEN",
                                     "price": entry})

        # Exit signal: DAILY candle only — Alex watched daily closes, not hourly noise
        # Only check once per day (when hour == 0, the previous daily candle just closed)
        # Also enforce minimum hold period so we don't exit before the trade is established
        is_daily_close = (ts.hour == 0)
        past_min_hold  = (bars_held >= MIN_HOLD_HOURS)

        if is_daily_close and past_min_hold:
            pdata_d = candle_data[pair]["d"]
            hist_d  = pdata_d[pdata_d.index < ts]
            opp     = "short" if direction=="long" else "long"

            has_exit, exit_sig = signal_det.has_signal(hist_d.iloc[-2:], opp)

            # Exit level must be at a DIFFERENT major level than our entry
            # (can't exit at the same level we entered from — that's just noise)
            entry_psych = pos.get("psych_level", 0)
            exit_psych_levels = get_round_levels(close, EXIT_PSYCH_TOLERANCE)
            at_different_level = any(
                abs(lvl - entry_psych) / max(entry_psych, 0.0001) >= EXIT_LEVEL_MIN_SEP
                for lvl in exit_psych_levels
            ) if exit_psych_levels else False

            # Price must have traveled at least MIN_EXCURSION_FOR_EXIT R from entry
            # before we'll even consider an exit — let winners run
            current_excursion_r = abs(close - entry) / risk if risk > 0 else 0
            traveled_enough = current_excursion_r >= MIN_EXCURSION_FOR_EXIT

            if (has_exit and exit_sig
                    and exit_sig.strength >= EXIT_STRENGTH_MIN
                    and at_different_level
                    and traveled_enough):
                # Increment consecutive exit signal counter
                pos["consecutive_exit_signals"] = pos.get("consecutive_exit_signals", 0) + 1
            else:
                # No qualifying signal today — reset the streak (trade still healthy)
                pos["consecutive_exit_signals"] = 0

            # Only exit after CONSECUTIVE_EXITS_REQ consecutive daily reversal closes
            if pos.get("consecutive_exit_signals", 0) >= CONSECUTIVE_EXITS_REQ:
                p = trade_pnl(pair, direction, entry, close, pos["units"])
                rr_val = (abs(close-entry)/risk) * (1 if p>0 else -1) if risk>0 else 0
                balance += p
                trades.append({**pos, "exit_ts": str(ts), "exit_price": close,
                               "exit_reason": f"exit_{CONSECUTIVE_EXITS_REQ}d_reversal",
                               "pnl": p, "rr": round(rr_val,2), "bars_held": bars_held})
                decision_log.append({"ts": str(ts), "pair": pair, "event": "EXIT_SIGNAL",
                                     "price": close, "strength": exit_sig.strength if exit_sig else 0, "pnl": p})
                mark_pattern_exhausted(pair, pos.get("pattern",""), pos.get("neckline", close))
                del open_pos[pair]; be_moved.discard(pos_key)
                continue

        # Max hold
        if bars_held >= MAX_HOLD_BARS_1H:
            p = trade_pnl(pair, direction, entry, close, pos["units"])
            rr_val = (abs(close-entry)/risk)*(1 if p>0 else -1) if risk>0 else 0
            balance += p
            trades.append({**pos, "exit_ts": str(ts), "exit_price": close,
                           "exit_reason": "max_hold", "pnl": p, "rr": round(rr_val,2),
                           "bars_held": bars_held})
            mark_pattern_exhausted(pair, pos.get("pattern",""), pos.get("neckline", close))
            del open_pos[pair]; be_moved.discard(pos_key)

    # ── Entry scan (London/NY only, no open positions) ─────────────────
    if open_pos or not is_london_or_ny(ts_utc):
        continue

    for pair in WATCHLIST:
        if pair not in candle_data: continue
        entry_data = evaluate_entry_1h(pair, ts)
        if not entry_data: continue

        # Skip if this exact pattern formation has already been traded
        if already_traded(pair, entry_data["pattern"], entry_data["neckline"]):
            continue

        risk_pct     = get_risk_pct(balance)
        e_price      = entry_data["entry_price"]
        s_loss       = entry_data["stop_loss"]
        units        = max(1000, min(int(balance*(risk_pct/100)/max(abs(e_price-s_loss),0.0001)), 100_000))
        risk_dollars = balance * (risk_pct/100)

        pos = {**entry_data, "pair": pair, "entry_ts": str(ts),
               "bar_idx": bar_idx, "risk_pct": risk_pct,
               "risk_dollars": risk_dollars, "units": units}
        open_pos[pair] = pos

        decision_log.append({"ts": str(ts), "pair": pair, "event": "TRADE_ENTERED",
                              "direction": entry_data["direction"], "price": e_price,
                              "stop": s_loss, "units": units, "notes": entry_data["notes"]})
        print(f"  📈 {ts} | ENTER {pair} {entry_data['direction'].upper()} "
              f"@ {e_price:.5f}  SL={s_loss:.5f}  "
              f"conf={entry_data['confidence']:.0%}  [{entry_data['notes']}]")
        break

# Close any still-open positions at last price
for pair, pos in open_pos.items():
    df = candle_data[pair]["1h"]
    last = float(df[df.index <= BACKTEST_END.replace(tzinfo=None)]["close"].iloc[-1])
    p = trade_pnl(pair, pos["direction"], pos["entry_price"], last, pos["units"])
    r = abs(pos["entry_price"]-pos["stop_loss"])
    rr = (abs(last-pos["entry_price"])/r)*(1 if p>0 else -1) if r>0 else 0
    balance += p
    trades.append({**pos, "exit_ts": str(BACKTEST_END.date()), "exit_price": last,
                   "exit_reason": "open_at_end", "pnl": p, "rr": round(rr,2), "bars_held": 0})

# ── Save decision log ────────────────────────────────────────────────────
log_path = Path.home() / "trading-bot/logs/backtest_decisions.json"
log_path.parent.mkdir(exist_ok=True)
log_path.write_text(json.dumps(decision_log, indent=2))

# ── Results ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"OANDA 1H BACKTEST — {BACKTEST_START.date()} → {BACKTEST_END.date()}")
print(f"Data: LIVE OANDA 1H candles | Capital: ${STARTING_BAL:,.2f}")
print("="*65)

if not trades:
    print("\nNo trades taken in this period.")
    print("This can mean: no double top/bottom patterns formed at")
    print("psychological levels with confirmed 1H engulfing entries.")
    print("Try widening DOUBLE_SIMILARITY or PSYCH_TOLERANCE above.\n")
else:
    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total  = sum(t["pnl"] for t in trades)
    wr     = len(wins)/len(trades)*100

    print(f"\n📊 SUMMARY")
    print(f"  Trades:        {len(trades)}")
    print(f"  Wins/Losses:   {len(wins)}/{len(losses)}  ({wr:.0f}% WR)")
    if wins:   print(f"  Avg win R:R:   +{np.mean([t['rr'] for t in wins]):.1f}R")
    if losses: print(f"  Avg loss R:R:  {np.mean([t['rr'] for t in losses]):.1f}R")
    print(f"\n💰 P&L")
    print(f"  Starting:  ${STARTING_BAL:>10,.2f}")
    print(f"  Net P&L:   ${total:>+10,.2f}")
    print(f"  Final:     ${balance:>10,.2f}")
    print(f"  Return:    {(balance/STARTING_BAL-1)*100:>+.1f}%")

    print(f"\n📋 TRADES")
    hdr = f"  {'Pair':<10} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'R:R':>6} {'P&L':>10}  Reason / Pattern"
    print(hdr); print("  "+"-"*80)
    for t in trades:
        et = t['entry_ts'][:16] if len(t['entry_ts'])>10 else t['entry_ts']
        xt = t['exit_ts'][:10]
        print(f"  {t['pair']:<10} {t['direction']:<6} {t['entry_price']:>10.5f} "
              f"{t['exit_price']:>10.5f} {t['rr']:>+5.1f}R {t['pnl']:>+10.2f}  "
              f"{t['exit_reason']:<14} {t.get('notes','')[:40]}")

# R:R distribution vs Alex's profile
if trades:
    wins_all   = [t for t in trades if t.get("pnl",0) > 0]
    losses_all = [t for t in trades if t.get("pnl",0) <= 0]
    rr_vals    = [t.get("rr",0) for t in trades]
    print(f"\n📐 R:R DISTRIBUTION vs ALEX'S PROFILE")
    buckets      = [("< 0  (loss)", lambda r: r < 0),
                    ("0 – 1R     ", lambda r: 0 <= r < 1),
                    ("1 – 3R     ", lambda r: 1 <= r < 3),
                    ("3 – 5R     ", lambda r: 3 <= r < 5),
                    ("5R+        ", lambda r: r >= 5)]
    alex_pcts    = ["~35%", "~10%", "~25%", "~20%", "~10%"]
    print(f"  {'Bucket':<14}  {'Ours':>10}  {'Alex':>8}")
    print(f"  {'-'*36}")
    for (label, fn), ap in zip(buckets, alex_pcts):
        n = sum(1 for r in rr_vals if fn(r))
        pct = n / len(rr_vals) * 100
        print(f"  {label}   {n:>3} ({pct:>4.0f}%)  {ap:>8}")
    avg_win_rr = np.mean([t["rr"] for t in wins_all]) if wins_all else 0
    print(f"\n  Our avg winning R:R:  {avg_win_rr:+.2f}R")
    print(f"  Alex avg winning R:R: ~+5.00R  (range 3.4R – 11R+)")
    print(f"  Gap source: exit timing (daily vs hourly resolution)")

print(f"\n✅ Full decision log saved: {log_path}")
