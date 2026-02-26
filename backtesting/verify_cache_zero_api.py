#!/usr/bin/env python3
"""
Two-run cache verification: confirms second run hits 0 OANDA API calls.

Usage:
    cd ~/trading-bot
    PYTHONPATH=/home/forge/trading-bot venv/bin/python backtesting/verify_cache_zero_api.py

Run 1: fetches from OANDA (or cache if already warm), saves per-pair files.
Run 2: all pairs/TFs from cache → api_calls == 0.
"""
import sys
from datetime import datetime, timezone

sys.path.insert(0, "/home/forge/trading-bot")

from backtesting.oanda_backtest_v2 import run_backtest, TRAIL_ARMS, _DEFAULT_ARM, CACHE_DIR

ALEX_START   = datetime(2024, 7,  1, tzinfo=timezone.utc)
ALEX_END     = datetime(2024, 10, 31, tzinfo=timezone.utc)
STARTING_BAL = 8_000.0

print("=" * 65)
print("CACHE VERIFICATION: two runs, same window, use_cache=True")
print("=" * 65)

for run_n in (1, 2):
    print(f"\n{'─'*65}")
    print(f"RUN {run_n}")
    print(f"{'─'*65}")
    r = run_backtest(
        start_dt=ALEX_START,
        end_dt=ALEX_END,
        starting_bal=STARTING_BAL,
        notes=f"cache_verify/run{run_n}",
        trail_cfg=TRAIL_ARMS[_DEFAULT_ARM],
        use_cache=True,
    )
    api = r.get("api_calls", -1)
    n   = r.get("n_trades", 0)
    ret = r.get("ret_pct", 0)
    print(f"\n  ✔ Run {run_n}: {n} trades  {ret:+.1f}%  OANDA API calls: {api}")
    if run_n == 2:
        if api == 0:
            print("  ✅ PASS — second run made 0 OANDA API calls (full cache hit)")
        else:
            print(f"  ❌ FAIL — second run made {api} API calls (expected 0)")

print(f"\nCache dir: {CACHE_DIR}")
import os
files = list(CACHE_DIR.glob("*.pkl")) if CACHE_DIR.exists() else []
print(f"Cached files: {len(files)}")
for f in sorted(files):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<45} {size_kb:6.0f} KB")
