#!/usr/bin/env python3
"""
Controlled comparison: ENTRY_TRIGGER_MODE="engulf_only" vs "engulf_or_pin"

Window  : Jul 15 – Oct 31 2024 (Alex window, mid-month start per Mike spec)
Arm     : C (2-stage trail — best performer in ab9)
Spread  : ON
Concurrency : 1 (BACKTEST must mirror LIVE — parity rule)
MIN_RR  : 2.5 (from strategy_config)
Whitelist   : logs/whitelist_backtest.json (Alex 7 pairs, enabled=true)

Parity assertions fire before any backtest runs.
Outputs: clean summary table + per-trade list per mode.

Usage:
    cd ~/trading-bot
    PYTHONPATH=/home/forge/trading-bot venv/bin/python backtesting/compare_trigger_modes.py
"""
import sys
from datetime import datetime, timezone
from collections import Counter

sys.path.insert(0, "/home/forge/trading-bot")

import src.strategy.forex.strategy_config as _sc
from backtesting.oanda_backtest_v2 import run_backtest, TRAIL_ARMS

# ── Parity assertions ────────────────────────────────────────────────────────
def _assert_parity():
    errors = []
    if _sc.MAX_CONCURRENT_TRADES_BACKTEST != _sc.MAX_CONCURRENT_TRADES_LIVE:
        errors.append(
            f"  MAX_CONCURRENT_TRADES_BACKTEST={_sc.MAX_CONCURRENT_TRADES_BACKTEST} "
            f"≠ LIVE={_sc.MAX_CONCURRENT_TRADES_LIVE}"
        )
    if getattr(_sc, "MIN_RR", None) != 2.5:
        errors.append(
            f"  MIN_RR={getattr(_sc,'MIN_RR','<missing>')} ≠ required 2.5"
        )
    if errors:
        print("╔══ PARITY VIOLATION — aborting ══════════════════════════")
        for e in errors: print(e)
        print("╚══ Fix strategy_config.py before running comparison ════")
        sys.exit(1)

_assert_parity()
print(f"✓ Parity OK: concurrent LIVE={_sc.MAX_CONCURRENT_TRADES_LIVE} "
      f"BT={_sc.MAX_CONCURRENT_TRADES_BACKTEST}  |  MIN_RR={_sc.MIN_RR}")

ARM_KEY      = "C"   # 2-stage trail — best performer in ab9
ALEX_START   = datetime(2024, 7, 15, tzinfo=timezone.utc)  # mid-month per Mike spec
ALEX_END     = datetime(2024, 10, 31, tzinfo=timezone.utc)
STARTING_BAL = 8_000.0

MODES = [
    ("engulf_only",   "Engulf only  (production)"),
    ("engulf_or_pin", "Engulf + pin (experiment)"),
]


def _patch_trigger(mode: str):
    """Patch ENTRY_TRIGGER_MODE + derived ENGULFING_ONLY on the live module.
    entry_signal.py imports strategy_config as _cfg and reads _cfg.ENTRY_TRIGGER_MODE
    at call time, so patching the module attribute is sufficient."""
    _sc.ENTRY_TRIGGER_MODE = mode
    _sc.ENGULFING_ONLY     = (mode == "engulf_only")


results    = {}
candle_cache = None  # share across runs — saves second OANDA fetch

for mode, label in MODES:
    print(f"\n  Running: {label}  (arm {ARM_KEY})  …", end="", flush=True)
    _patch_trigger(mode)

    r = run_backtest(
        start_dt=ALEX_START,
        end_dt=ALEX_END,
        starting_bal=STARTING_BAL,
        notes=f"trigger_compare/{mode}/arm{ARM_KEY}/concurrent{_sc.MAX_CONCURRENT_TRADES_BACKTEST}",
        trail_cfg=TRAIL_ARMS[ARM_KEY],
        preloaded_candle_data=candle_cache,
        quiet=True,         # suppress verbose output — comparison table printed below
    )
    if candle_cache is None:
        candle_cache = r.get("candle_data")
    results[mode] = r
    n = len(r.get("trades", []))
    print(f"  {n} trades  ret={r.get('ret_pct',0):+.1f}%")

# Restore default
_patch_trigger("engulf_only")


# ── Build enriched summary per mode ───────────────────────────────────────
def summarise(r: dict) -> dict:
    trades  = r.get("trades", [])
    wins    = [t for t in trades if t["r"] >  0.10]
    losses  = [t for t in trades if t["r"] < -0.10]
    scratch = [t for t in trades if -0.10 <= t["r"] <= 0.10]
    ec      = Counter(t.get("reason", "") for t in trades)
    sc      = Counter(t.get("signal_type", "") for t in trades)
    n       = len(trades)

    pin_types    = {"pin_bar_bearish", "pin_bar_bullish"}
    engulf_types = {"bearish_engulfing", "bullish_engulfing"}
    n_pin    = sum(v for k, v in sc.items() if k in pin_types)
    n_engulf = sum(v for k, v in sc.items() if k in engulf_types)
    n_other  = n - n_pin - n_engulf

    # Exit counts + avg_r per exit type
    def _exit_avg(reason):
        ts = [t["r"] for t in trades if t.get("reason") == reason]
        return (sum(ts)/len(ts) if ts else 0.0, len(ts))

    n_target,  avg_r_target  = _exit_avg("target_reached")[1], _exit_avg("target_reached")[0]
    n_ratchet, avg_r_ratchet = _exit_avg("ratchet_stop_hit")[1], _exit_avg("ratchet_stop_hit")[0]
    n_sl,      avg_r_sl      = _exit_avg("stop_hit")[1], _exit_avg("stop_hit")[0]
    n_runout,  avg_r_runout  = _exit_avg("runout_expired")[1], _exit_avg("runout_expired")[0]
    n_weekend, avg_r_weekend = _exit_avg("weekend_proximity")[1], _exit_avg("weekend_proximity")[0]

    total_spread = sum(t.get("spread_cost", 0) for t in trades)

    return {
        "n_trades":     n,
        "n_wins":       len(wins),
        "n_losses":     len(losses),
        "n_scratch":    len(scratch),
        "win_rate":     r.get("win_rate", 0),
        "avg_r":        r.get("avg_r", 0),
        "avg_r_win":    r.get("avg_r_win", 0),
        "avg_r_loss":   r.get("avg_r_loss", 0),
        "ret_pct":      r.get("ret_pct", 0),
        "max_dd":       r.get("max_dd", 0),
        "exec_rr_p50":  r.get("exec_rr_p50", 0),
        "n_target":     n_target,   "avg_r_target":   avg_r_target,
        "n_ratchet":    n_ratchet,  "avg_r_ratchet":  avg_r_ratchet,
        "n_sl":         n_sl,       "avg_r_sl":       avg_r_sl,
        "n_runout":     n_runout,   "avg_r_runout":   avg_r_runout,
        "n_weekend":    n_weekend,  "avg_r_weekend":  avg_r_weekend,
        "n_pin":        n_pin,
        "n_engulf":     n_engulf,
        "n_other":      n_other,
        "spread_drag":  total_spread,
        "ks_blocks":    r.get("dd_killswitch_blocks", 0),
    }

sums = {mode: summarise(results[mode]) for mode, _ in MODES}


# ── Per-trade list ─────────────────────────────────────────────────────────
def _trade_table(r: dict, mode: str):
    trades = r.get("trades", [])
    label  = {"engulf_only": "ENGULF ONLY", "engulf_or_pin": "ENGULF + PIN"}[mode]
    print(f"\n  ── Trade list: {label} ──────────────────────────────────────────")
    print(f"  {'Pair':<10} {'Dir':<6} {'Entry':>10} {'Exit':>10} "
          f"{'R':>6} {'Reason':<22} {'Signal':<25}")
    print(f"  {'-'*95}")
    for t in sorted(trades, key=lambda x: x.get("entry_ts", "")):
        flag = "✓" if t["r"] > 0.10 else ("~" if t["r"] > -0.10 else "✗")
        sig  = t.get("signal_type", "?")[:24]
        print(f"  {t['pair']:<10} {t['direction']:<6} {t['entry']:>10.5f} {t['exit']:>10.5f} "
              f"{t['r']:>+6.2f}R {t.get('reason','?'):<22} {sig:<25} {flag}")
    print(f"  Total: {len(trades)} trades | "
          f"Wins: {sum(1 for t in trades if t['r']>0.10)} | "
          f"Losses: {sum(1 for t in trades if t['r']<-0.10)}")


# ── Print comparison table ─────────────────────────────────────────────────
print(f"\n\n{'═'*72}")
print(f"ENTRY TRIGGER COMPARISON — Jul 15–Oct 31 2024 (Alex window)")
print(f"{'Arm C  |  Spread ON  |  concurrent=1  |  MIN_RR=2.5  |  $8,000 start':>72}")
print(f"{'═'*72}")

col_w = 30
val_w = 21
hdr   = f"{'Metric':<{col_w}}" + "".join(f"{MODES[i][1]:>{val_w}}" for i in range(len(MODES)))
print(hdr)
print("─" * (col_w + val_w * len(MODES)))

def pct(v):    return f"{v*100:.1f}%"
def r2(v):     return f"{v:+.2f}R"
def r2u(v):    return f"{v:.2f}R"
def p1(v):     return f"{v:+.1f}%"
def p1u(v):    return f"{v:.1f}%"
def n(v):      return str(int(v))
def dollar(v): return f"${abs(v):,.0f}"
def nr(cnt_k, avgr_k):
    """Return a formatter that prints 'N (avg ±X.XXR)'."""
    def _fmt(v, _s=None, _ck=cnt_k, _ak=avgr_k):
        # v is ignored; called with sums[mode][cnt_k]
        return None
    return _fmt

def _row(label, key, fmt):
    if key == "":
        print(f"\n{label}")
        return
    vals = []
    for mode, _ in MODES:
        s = sums[mode]
        v = s.get(key, "—")
        try:
            vals.append(fmt(v) if fmt and v != "—" else str(v))
        except Exception:
            vals.append(str(v))
    print(f"{label:<{col_w}}" + "".join(f"{v:>{val_w}}" for v in vals))

def _exit_row(label, cnt_key, avgr_key):
    """Print exit row: 'N  avg ±X.XXR'"""
    vals = []
    for mode, _ in MODES:
        s = sums[mode]
        cnt  = s.get(cnt_key, 0)
        avgr = s.get(avgr_key, 0.0)
        if cnt:
            vals.append(f"{cnt}  ({avgr:+.2f}R avg)")
        else:
            vals.append("0")
    print(f"{label:<{col_w}}" + "".join(f"{v:>{val_w}}" for v in vals))

_row("Trades total",             "n_trades",     n)
print("")
_row("Wins",                     "n_wins",        n)
_row("Losses",                   "n_losses",      n)
_row("Scratches",                "n_scratch",     n)
_row("Win rate",                 "win_rate",      pct)
print("\n── Returns ──────────────────────────────────────")
_row("Return",                   "ret_pct",       p1)
_row("Max drawdown",             "max_dd",        p1u)
_row("Avg R (all)",              "avg_r",         r2)
_row("Avg R (wins)",             "avg_r_win",     r2)
_row("Avg R (losses)",           "avg_r_loss",    r2u)
_row("Exec RR p50",              "exec_rr_p50",   r2u)
print("\n── Exit reasons (count + avg R per bucket) ──────")
_exit_row("  Target reached",    "n_target",   "avg_r_target")
_exit_row("  Ratchet stop (win)","n_ratchet",  "avg_r_ratchet")
_exit_row("  Stop hit (loss)",   "n_sl",       "avg_r_sl")
_exit_row("  Weekend proximity", "n_weekend",  "avg_r_weekend")
_exit_row("  Runout expired",    "n_runout",   "avg_r_runout")
print("\n── Entry signals ─────────────────────────────────")
_row("  Engulf entries",         "n_engulf",      n)
_row("  Pin bar entries",        "n_pin",         n)
_row("  Other/unknown",          "n_other",       n)
print("\n── Cost / Risk controls ──────────────────────────")
_row("  Spread drag ($)",        "spread_drag",   dollar)
_row("  KS blocks",              "ks_blocks",     n)

# ── Verdict ───────────────────────────────────────────────────────────────
s0 = sums["engulf_only"]
s1 = sums["engulf_or_pin"]
d_ret  = s1["ret_pct"]  - s0["ret_pct"]
d_dd   = s1["max_dd"]   - s0["max_dd"]
d_wr   = (s1["win_rate"] - s0["win_rate"]) * 100
d_tr   = s1["n_trades"] - s0["n_trades"]
n_pin  = s1["n_pin"]

print(f"\n{'═'*72}")
print("VERDICT")
print(f"  {'engulf_only':14}: {s0['ret_pct']:+.1f}% return  "
      f"{s0['max_dd']:.1f}% DD  "
      f"{s0['n_trades']} trades  "
      f"{s0['win_rate']*100:.1f}% WR")
print(f"  {'engulf_or_pin':14}: {s1['ret_pct']:+.1f}% return  "
      f"{s1['max_dd']:.1f}% DD  "
      f"{s1['n_trades']} trades  "
      f"{s1['win_rate']*100:.1f}% WR  "
      f"({n_pin} pin entries)")
print(f"\n  Return delta : {d_ret:+.1f}%  |  DD delta: {d_dd:+.1f}%  |  "
      f"WR delta: {d_wr:+.1f}pp  |  "
      f"{'+' if d_tr >= 0 else ''}{d_tr} extra trades")

if   d_ret > 10 and d_dd < 5:
    verdict = "✅ Pin bars add meaningful edge — consider enabling for research runs"
elif d_ret > 5  and d_dd < 3:
    verdict = "↗ Marginal improvement — not enough to override engulf-only default"
elif d_ret < -3:
    verdict = "❌ Pin bars hurt — keep engulf_only"
elif abs(d_ret) <= 3 and n_pin == 0:
    verdict = "ℹ No pin bars triggered — no difference in this window"
elif abs(d_ret) <= 3:
    verdict = "⟷ Neutral — pin bars entered but no meaningful edge difference"
else:
    verdict = "⚠ Mixed — more data needed before changing default"

print(f"  Recommendation: {verdict}")
print(f"{'═'*72}\n")

# ── Per-trade lists ────────────────────────────────────────────────────────
for mode, _ in MODES:
    _trade_table(results[mode], mode)
