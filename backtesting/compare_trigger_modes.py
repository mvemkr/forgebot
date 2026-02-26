#!/usr/bin/env python3
"""
Controlled comparison: ENTRY_TRIGGER_MODE="engulf_only" vs "engulf_or_pin"

Window  : Jul–Oct 2024 (Alex window)
Arm     : C (2-stage trail — best performer)
Spread  : ON
Concurrency: 1 (parity rule — backtest must mirror live)

NOTE: A previous run (Feb 26 AM) used BACKTEST concurrency=4, which
contaminated the result (+15% vs -14.1%). That result is INVALID.
This script asserts parity before running; aborts if concurrency ≠ 1.

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

# ── Parity assertion — abort if backtest concurrency ≠ live ──────────────────
assert _sc.MAX_CONCURRENT_TRADES_BACKTEST == _sc.MAX_CONCURRENT_TRADES_LIVE, (
    f"PARITY VIOLATION: MAX_CONCURRENT_TRADES_BACKTEST={_sc.MAX_CONCURRENT_TRADES_BACKTEST} "
    f"≠ LIVE={_sc.MAX_CONCURRENT_TRADES_LIVE}. "
    f"Fix strategy_config.py or use --max-trades CLI flag for explicit experiments."
)
print(f"✓ Parity OK: concurrent LIVE={_sc.MAX_CONCURRENT_TRADES_LIVE} "
      f"BT={_sc.MAX_CONCURRENT_TRADES_BACKTEST}")

ARM_KEY      = "C"   # 2-stage trail — best performer in ab9
ALEX_START   = datetime(2024, 7,  1, tzinfo=timezone.utc)
ALEX_END     = datetime(2024, 10, 31, tzinfo=timezone.utc)
STARTING_BAL = 8_000.0

MODES = [
    ("engulf_only",   "Engulf only  (production)"),
    ("engulf_or_pin", "Engulf + pin (strict spec)"),
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
    print(f"\n{'─'*65}")
    print(f"MODE: {label}")
    print(f"{'─'*65}")
    _patch_trigger(mode)

    r = run_backtest(
        start_dt=ALEX_START,
        end_dt=ALEX_END,
        starting_bal=STARTING_BAL,
        notes=f"trigger_compare/{mode}/arm{ARM_KEY}/concurrent{_sc.MAX_CONCURRENT_TRADES_BACKTEST}",
        trail_cfg=TRAIL_ARMS[ARM_KEY],
        preloaded_candle_data=candle_cache,
    )
    if candle_cache is None:
        candle_cache = r.get("candle_data")
    results[mode] = r

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

    pin_types   = {"pin_bar_bearish", "pin_bar_bullish"}
    engulf_types = {"bearish_engulfing", "bullish_engulfing"}
    n_pin    = sum(v for k, v in sc.items() if k in pin_types)
    n_engulf = sum(v for k, v in sc.items() if k in engulf_types)
    n_other  = n - n_pin - n_engulf

    n_target  = ec.get("target_reached", 0) + ec.get("weekend_proximity", 0)
    n_ratchet = ec.get("ratchet_stop_hit", 0)
    n_sl      = ec.get("stop_hit", 0)
    n_runout  = ec.get("runout_expired", 0)

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
        "n_target":     n_target,
        "n_ratchet":    n_ratchet,
        "n_sl":         n_sl,
        "n_runout":     n_runout,
        "n_pin":        n_pin,
        "n_engulf":     n_engulf,
        "n_other":      n_other,
        "spread_drag":  total_spread,
        "ks_blocks":    r.get("dd_killswitch_blocks", 0),
    }

sums = {mode: summarise(results[mode]) for mode, _ in MODES}

# ── Print table ───────────────────────────────────────────────────────────
print(f"\n\n{'═'*68}")
print(f"{'ENTRY TRIGGER MODE COMPARISON — Jul–Oct 2024 (Alex window)'}")
print(f"{f'Arm {ARM_KEY}, Spread ON, concurrent=1, $8,000 start':>68}")
print(f"{'═'*68}")

col_w = 26
val_w = 20
hdr   = f"{'Metric':<{col_w}}" + "".join(f"{MODES[i][1]:>{val_w}}" for i in range(len(MODES)))
print(hdr)
print("─" * (col_w + val_w * len(MODES)))

def pct(v): return f"{v*100:.1f}%"
def r2(v):  return f"{v:+.2f}R"
def r2u(v): return f"{v:.2f}R"
def p1(v):  return f"{v:+.1f}%"
def p1u(v): return f"{v:.1f}%"
def n(v):   return str(int(v))
def dollar(v): return f"${abs(v):,.0f}"

rows = [
    ("Trades total",         "n_trades",    n),
    ("Wins / Losses / Scratch","",          None),
    ("  Wins",               "n_wins",      n),
    ("  Losses",             "n_losses",    n),
    ("  Scratches",          "n_scratch",   n),
    ("Win rate",             "win_rate",    pct),
    ("─── Returns ───",      "",            None),
    ("Return",               "ret_pct",     p1),
    ("Max drawdown",         "max_dd",      p1u),
    ("Avg R (all)",          "avg_r",       r2),
    ("Avg R (wins)",         "avg_r_win",   r2),
    ("Avg R (losses)",       "avg_r_loss",  r2u),
    ("Exec RR p50",          "exec_rr_p50", r2u),
    ("─── Exits ───",        "",            None),
    ("Target reached",       "n_target",    n),
    ("Ratchet stop",         "n_ratchet",   n),
    ("Stop hit",             "n_sl",        n),
    ("Runout",               "n_runout",    n),
    ("─── Entries ───",      "",            None),
    ("Engulf entries",       "n_engulf",    n),
    ("Pin bar entries",      "n_pin",       n),
    ("Other/unknown",        "n_other",     n),
    ("─── Cost ───",         "",            None),
    ("Spread drag ($)",      "spread_drag", dollar),
    ("KS blocks",            "ks_blocks",   n),
]

for label_str, key, fmt in rows:
    if not key:
        print(f"\n{label_str}")
        continue
    vals = []
    for mode, _ in MODES:
        s = sums[mode]
        v = s.get(key, "—")
        try:
            vals.append(fmt(v) if fmt and v != "—" else str(v))
        except Exception:
            vals.append(str(v))
    print(f"{label_str:<{col_w}}" + "".join(f"{v:>{val_w}}" for v in vals))

# ── Verdict ───────────────────────────────────────────────────────────────
s0 = sums["engulf_only"]
s1 = sums["engulf_or_pin"]
d_ret  = s1["ret_pct"]  - s0["ret_pct"]
d_dd   = s1["max_dd"]   - s0["max_dd"]
d_wr   = (s1["win_rate"] - s0["win_rate"]) * 100
d_tr   = s1["n_trades"] - s0["n_trades"]
n_pin  = s1["n_pin"]

print(f"\n{'═'*68}")
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
      f"WR delta: {d_wr:+.1f}pp  |  +{d_tr} extra trades")

if   d_ret > 5  and d_dd < 5:
    verdict = "✅ Pin bars add edge — consider enabling for research runs"
elif d_ret > 0  and d_dd < 3:
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
print(f"{'═'*68}\n")
