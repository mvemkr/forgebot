# Trading Bot Development Guidelines

## The Most Important Rule

> **If it's in the backtester, it must be in the live bot. If it's in the live bot, it must be in the backtester. They must behave identically.**

This rule exists because we learned the hard way. During Feb 2026, the backtester had:
- `MIN_CONFIDENCE = 0.65` gate
- Session filter blocking non-London entries
- `ATR_MIN_MULTIPLIER` stop check

The live orchestrator had none of these. The bot entered a trade that the backtester would have blocked, produced a false kill-switch trigger from a JPY P&L bug, and cost hours of debugging.

---

## Shared Constants — Never Hardcode Thresholds

All execution parameters live in **one file**:

```
src/strategy/forex/strategy_config.py
```

Both `backtesting/oanda_backtest_v2.py` and `src/execution/orchestrator.py` **import from it**. They do not define their own copies.

**If you change a threshold, change it in `strategy_config.py` only.** The backtester and live bot inherit it automatically.

### What lives in strategy_config.py

| Constant | What it does |
|---|---|
| `MIN_CONFIDENCE` | Minimum signal confidence to execute |
| `MIN_RR` | Minimum R:R ratio (geometric quality gate) |
| `ATR_STOP_MULTIPLIER` | Max stop width = N × daily ATR |
| `ATR_MIN_MULTIPLIER` | Min stop width = N × daily ATR |
| `ATR_LOOKBACK` | Days for ATR calculation |
| `MAX_CONCURRENT_TRADES` | Hard cap on open positions |
| `LONDON_SESSION_START/END_UTC` | Session window for auto-execution |
| `STOP_COOLDOWN_DAYS` | Re-entry lockout after stop-out |
| `NECKLINE_CLUSTER_PCT` | Pattern deduplication tolerance |
| `DRY_RUN_PAPER_BALANCE` | Virtual balance when OANDA unfunded |

---

## Checklist: Adding a New Filter

When you add ANY new filter, gate, or threshold:

- [ ] Define the constant in `strategy_config.py`
- [ ] Import it in `oanda_backtest_v2.py`
- [ ] Import it in `orchestrator.py`
- [ ] Apply it in BOTH places with identical logic
- [ ] Run a short backtest to confirm the filter fires where expected
- [ ] Verify the live bot log shows the filter firing on the next run cycle
- [ ] Commit all three files together (config + backtester + orchestrator)

**Do not commit a filter to the backtester without wiring it to the live bot in the same commit.**

---

## The Backtester Is Production Code

The backtester is not a sandbox. It calls `SetAndForgetStrategy.evaluate()` directly — the exact same function the live bot calls. This is intentional.

**Do not reimplement strategy logic inside the backtester.** If you find yourself copy-pasting pattern detection or level scoring into `oanda_backtest_v2.py`, stop. Fix the strategy code instead.

The only acceptable backtester-only code:
- Data fetching for historical ranges
- P&L simulation (stop monitoring, trade sizing)
- Gap logging (v1 vs v2 comparison)
- `_NoOpNewsFilter` (historical ForexFactory data unavailable)

---

## Known Gaps (Acceptable Backtester/Live Differences)

These are intentional, documented, and do not violate the parity rule:

| Gap | Reason | Impact |
|---|---|---|
| News filter disabled | No historical ForexFactory data | Logged per-bar in gap_log.jsonl |
| Weekly candles resampled from daily | OANDA API limitation in backtest | Minor edge differences at week boundaries |
| Exit logic | Live bot notifies Mike; backtest auto-exits on stop | P&L identical — only execution path differs |

---

## P&L Calculations

JPY pairs require currency conversion. The pip value formula:

```python
# WRONG — gives JPY, not USD
pnl_dollars = pnl_pips * units * 0.01

# CORRECT — converts JPY to USD (approx rate)
pnl_dollars = pnl_pips * units * (0.01 / 150 if "JPY" in pair else 0.0001)
```

Using 150 as the USD/JPY rate is an approximation. Good enough for paper P&L display and kill-switch math. If you need exact settlement, pass the actual rate.

---

## Dry Run Rules

When `dry_run=True` and the OANDA account is unfunded ($0 real balance):

1. Use `DRY_RUN_PAPER_BALANCE` (from strategy_config) as the virtual balance
2. Apply it in `__init__` AND the run loop (the run loop refreshes the balance every tick)
3. Kill-switch math must use the paper balance — `$loss / $0 = 100%` is not a real drawdown
4. Never let the real $0 OANDA balance bleed into risk sizing or mode transitions

---

## Session Rules

Auto-execution only during London session (`LONDON_SESSION_START/END_UTC`).

Outside London:
- Detect and log the signal
- Send Telegram notification to Mike
- Do NOT auto-execute

This matches Alex's explicit rule: always enters London session, sets alarm, goes to sleep.

---

## When Something Goes Wrong

Before debugging the strategy, check the gap log:

```
~/trading-bot/logs/backtest_gap_log.jsonl
```

It records every bar where the backtester blocked an entry and why. Categories:
- `theme_direction_conflict` — macro theme opposing the signal
- `stop_too_wide` — stop > 8× ATR
- `low_confidence` — signal below 65%
- `news_filter_skipped` — expected gap (historical data unavailable)

If the live bot fired a trade the backtester would have blocked, or vice versa, that's a parity violation. Fix it by checking which file is missing the filter and adding it.
