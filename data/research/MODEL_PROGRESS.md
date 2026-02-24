# Model Progress Notes
_Updated: 2026-02-23 night session_

---

## Objective
Replicate Alex Mill's 12-week forex challenge results using an automated bot.
Alex turned $100 → $332,588 in 12 weeks (Jul–Oct 2024).
We are NOT trying to copy his exact trades — we are trying to prove his
**strategy** (round number + pattern + EMA + engulfing) is mechanically sound
enough to automate.

**Reference:** `data/research/ALEX_RUBRIC.md` — trade-by-trade breakdown.

---

## Strategy Rubric (Alex's 5 Rules)
1. **Round psychological level** — always explicitly named (157.5, 1.125, 0.889, 1.35, 192)
2. **Pattern AT the level** — H&S neckline, double top, or break/retest anchored to that round number
3. **EMA confluence** — weekly or 4H EMA at the same level almost every time
4. **ONE trigger** — 1H or 4H engulfing candle body-closing through the level
5. **No take profit** — stop only, trail to BE at 1:1

---

## Baseline History

| Commit    | Window        | Return  | Notes |
|-----------|---------------|---------|-------|
| 9195e2d   | Jul-Oct 2024  | -6.2%   | Official baseline, engulfing_only=True |
| 9195e2d   | Jan 2026      | +109%   | Official baseline |
| bffe330   | Jul-Oct 2024  | +61.5%  | conf=65%, start Jul 1 — GBP/JPY enters |
| 074fc1a   | Jul-Oct 2024  | +92.9%  | conf=75%, start Jul 1 — USD/JPY anchor |
| 074fc1a   | Jul-Oct 2024  | +50.3%  | conf=75%, start Jul 15 — 2 full matches |
| 074fc1a   | Jan 2026      | +39.8%  | conf=75%, 4 trades |

**Key insight (Mike, 2026-02-23):** Start window Jul 15 (not Jul 1) — Alex's
trading began Jul 16. Pre-Jul-15 noise trades filled slots before his real
setups could fire.

---

## Scorecard vs Alex (current best — 074fc1a, conf=75%, start Jul 15)

| Week | Alex Trade | Alex Result | Bot | Bot Result |
|------|-----------|-------------|-----|------------|
| Wk1  | GBP/JPY SHORT @~205 | WIN +119p | ✅ @204.99 H&S | +$3,712 open |
| Wk2  | USD/JPY SHORT @157.5 | WIN +150p | ❌ EUR/USD noise blocks slot | — |
| Wk3  | USD/CHF SHORT @0.889 | WIN +140p | ✅ @0.88076 | +$386 open |
| Wk4  | EUR/USD SHORT | LOSS -50p (rule viol) | ✅ Correctly skipped | — |
| Wk5  | No trades | — | ⚠️ EUR/USD LONG noise | -$1,200 |
| Wk6  | GBP/CHF SHORT @1.125 | WIN +350p | ❌ Bot finds 1.148 not 1.125 | — |
| Wk7  | GBP/CHF SHORT cont. | WIN +30p | ❌ Missed | — |
| Wk8  | USD/JPY SHORT @144 | WIN +380p | ⚠️ BE stop @145.2 | $0 |
| Wk9  | NZD/CAD SHORT | LOSS -70p | ✅ Correctly skipped | — |
| Wk10 | USD/CAD SHORT @1.35 | WIN +280p | ❌ Not detecting | — |
| Wk11 | USD/CAD SHORT @1.35 | WIN +160p | ❌ Not detecting | — |
| Wk12a| USD/JPY SHORT | LOSS -80p (Diddy) | ✅ Correctly blocked | — |
| Wk12b| GBP/CHF SHORT | WIN +600p | ❌ Missed | — |
| Wk13 | USD/CHF LONG | WIN +500p | ❌ Missed | — |

**Direct matches: 2/11 ✅ | Correct skips: 3/11 ✅ | Missed: 6/11 ❌**

---

## What's Working
- `ENGULFING_ONLY = True` — Alex's #1 rule enforced, single biggest fix
- `_neckline_at_level` — H&S/double-top necklines must be at a round number
- `PATTERN_PREFER_PROXIMITY` — prioritizes nearest pattern to current price
- `MIN_CONFIDENCE = 0.75` — kills low-quality noise signals
- Theme direction gate — correctly blocked 3 of Alex's losers
- GBP/JPY Wk1 now detecting correctly (H&S neckline at 204, enters Jul 15)
- USD/CHF Wk3 matching at 102% pip capture

## What's Broken / Next Fixes
1. **EUR/USD LONG false positive (Jul 15)** — `break_retest_bullish` fires same day
   as GBP/JPY, eats a slot, blocks USD/JPY Wk2. Fix: block EUR/USD break_retest_bullish
   when EUR theme is not weak / pattern doesn't match Alex's watchlist context.
   → Target: 3rd match unlocked (USD/JPY Wk2 @ 157.5)

2. **GBP/CHF 1.148 vs 1.125** — bot finds double_top at 1.148 (Jul) not 1.125 (Aug).
   The 1.125 is where the weekly EMA and round number meet in August.
   → EMA-at-level check should naturally prefer the Aug setup over Jul

3. **USD/CAD Wk10-11 @1.35** — break/retest at 1.35 not detecting. Tier 6 approach
   identified it correctly but added too much noise. Needs targeted reapplication.

4. **USD/CHF LONG Wk13** — bullish break/retest. Bot is structurally short-biased.
   Bullish break_retest detection working but not finding this specific one.

5. **GBP/CHF Wk12b** — 4H bearish engulfing at consolidation break. May need 4H
   pattern detection improvement.

---

## Key Architecture Decisions
- `set_and_forget.py` — main strategy: pattern detection → context gate → entry
- `pattern_detector.py` — H&S, double top/bottom, break_retest, engulfing
- `_mtf_context()` — multi-timeframe gate (Tier 1-6): trend alignment required
- `strategy_config.py` — all levers (ATR, confidence, risk tiers, etc.)
- Backtester: `backtesting/oanda_backtest_v2.py` — uses LIVE strategy code directly
- Decision log: `logs/backtest_v2_decisions.json` — every evaluated signal
- Gap log: `logs/backtest_gap_log.jsonl` — v1 vs v2 differences

## Important: Backtest Uses Real Code
The backtester imports and calls `SetAndForgetStrategy.evaluate()` directly.
Any change to `src/strategy/forex/` immediately affects backtest results.
No separate backtest-only logic. What backtests = what runs live.

---

## Commit Log (relevant)
| Hash    | Change |
|---------|--------|
| 27cd440 | ENGULFING_ONLY=True — single biggest quality fix |
| 9195e2d | Official baseline stamp — parity audit |
| f517243 | neckline_at_level + weekly_candle_agreement filters |
| 8a8a7a1 | PATTERN_PREFER_PROXIMITY sort + neckline string fix |
| bffe330 | Added analysis scripts (no strategy change) |
| 074fc1a | MIN_CONFIDENCE 0.65→0.75, ALEX_RUBRIC.md added |
