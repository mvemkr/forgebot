# Model Progress Notes
_Last updated: 2026-02-23 — commit c0236f5_

---

## Objective
Replicate Alex Mill's 12-week forex challenge results using an automated bot.
Alex turned $100 → $332,588 in 12 weeks (Jul–Oct 2024).
Goal: prove his strategy (round number + pattern + EMA + engulfing) is mechanically
sound enough to automate profitably.

Reference: `data/research/ALEX_RUBRIC.md`

---

## Alex's 5-Rule Model
1. **Round psychological level** — always named explicitly (157.5, 1.125, 0.889, 1.35, 205)
2. **Pattern AT the level** — H&S neckline, double top, or break/retest anchored to round number
3. **EMA confluence** — weekly or 4H EMA at the same level almost every time
4. **ONE trigger** — 1H or 4H engulfing candle body-closing through the level
5. **No take profit** — stop only, trail to BE at 1:1

---

## Return Baseline History

| Commit  | Window       | Start   | Conf | Return  | Notes |
|---------|--------------|---------|------|---------|-------|
| 9195e2d | Jul–Oct 2024 | Jul 1   | 65%  | -6.2%   | Official baseline |
| 9195e2d | Jan 2026     | Jan 1   | 65%  | +109%   | Official baseline |
| bffe330 | Jul–Oct 2024 | Jul 1   | 65%  | +61.5%  | GBP/JPY enters correctly |
| 074fc1a | Jul–Oct 2024 | Jul 1   | 75%  | +92.9%  | Noise reduced |
| 074fc1a | Jul–Oct 2024 | Jul 15  | 75%  | +50.3%  | 2 matches — EUR/USD noise blocks slot |
| c0236f5 | Jul–Oct 2024 | Jul 15  | 77%  | +68.2%  | EUR/USD blocked, 2 clean matches |
| c0236f5 | Jan 2026     | Jan 1   | 77%  | TBD     | Run pending |

**Key insight (Mike):** Start window Jul 15 — Alex began trading Jul 16. Pre-Jul-15
noise was filling slots before his real setups fired.

---

## Scorecard vs Alex — Current Best (c0236f5, conf=77%, start Jul 15)

### Alex's Trades
| Week | Alex | Result | Bot | Bot P&L |
|------|------|--------|-----|---------|
| Wk1  | GBP/JPY SHORT @~205 | WIN +119p | ✅ @204.99 H&S | +$3,712 open |
| Wk2  | USD/JPY SHORT @157.5 | WIN +150p | ❌ Pattern detected @158, no engulfing in retest window | — |
| Wk3  | USD/CHF SHORT @0.889 | WIN +140p | ✅ @0.881 break_retest | +$386 open |
| Wk4  | EUR/USD SHORT | LOSS -50p | ✅ Correctly skipped | — |
| Wk5  | No trades | — | ✅ Matched | — |
| Wk6  | GBP/CHF SHORT @1.125 | WIN +350p | ❌ Bot finds 1.148 not 1.125 | — |
| Wk7  | GBP/CHF SHORT cont. | WIN +30p | ❌ Missed | — |
| Wk8  | USD/JPY SHORT @144 | WIN +380p | ⚠️ H&S @145.2 → BE stop | $0 |
| Wk9  | NZD/CAD SHORT | LOSS -70p | ✅ Correctly skipped | — |
| Wk10 | USD/CAD SHORT @1.35 | WIN +280p | ❌ Not detecting | — |
| Wk11 | USD/CAD SHORT @1.35 | WIN +160p | ❌ Not detecting | — |
| Wk12a | USD/JPY SHORT | LOSS -80p | ✅ Correctly blocked | — |
| Wk12b | GBP/CHF SHORT | WIN +600p | ❌ Missed | — |
| Wk13 | USD/CHF LONG | WIN +500p | ❌ Missed | — |

**Matches: 2/11 ✅ | Correct skips: 3/11 ✅ | Missed: 6/11 ❌**

### Bot-Only Trades (not Alex's)
| Trade | P&L | Notes |
|-------|-----|-------|
| USD/JPY SHORT @145.2 → BE | $0 | Right pair, right direction — too late |
| NZD/JPY SHORT @94.8 | +$820 open | Valid setup, not Alex's pair |
| USD/JPY LONG @148.8 | +$534 open | USD_strong theme continuation |

---

## What's Working
- `ENGULFING_ONLY = True` — single biggest fix (commit 27cd440)
- `_neckline_at_level` — necklines must be at a round number
- `PATTERN_PREFER_PROXIMITY` — nearest valid pattern wins
- `MIN_CONFIDENCE = 0.77` — kills low-quality and borderline noise
- Theme direction gate — correctly skips 3 of Alex's losers
- GBP/JPY Wk1: H&S at 204.000, enters Jul 15 London ✅
- USD/CHF Wk3: 102% pip capture ✅

## What's Next (Priority Order)
1. **Wk2 USD/JPY @157.5** — Pattern detected correctly (double_top at 158.000),
   2 WAIT windows, but no 1H engulfing fires on OANDA data during retest.
   Fix: check if H&S @158.258 produces engulfing when double_top at 158 doesn't.
   May need to lower neckline proximity tolerance for the specific 1H retest hour.

2. **Wk6 GBP/CHF @1.125 vs 1.148** — Bot finds double_top at 1.148 (July).
   Alex's level was 1.125 (August, when weekly EMA arrived at that level).
   Fix: EMA-at-level should naturally score 1.125 higher in August.
   Needs EMA confluence in confidence formula targeting pattern_level not current_price.

3. **Wk10-11 USD/CAD @1.35** — Break/retest at 1.35 not detecting.
   Root cause identified: `_mtf_context` has no tier for neutral-trend break_retest.
   Tier 6 approach was correct but added noise. Needs cleaner reapplication with
   tighter round-number anchoring.

4. **Wk13 USD/CHF LONG** — Bullish break/retest not firing.
   Bot is structurally short-biased. Need to verify bullish break_retest
   detection for the Oct consolidation breakout.

5. **Wk12b GBP/CHF** — 4H engulfing at consolidation break. Needs
   better 4H pattern source detection.

---

## Architecture Notes
- `set_and_forget.py` — main strategy, all logic lives here
- `pattern_detector.py` — pattern detection (H&S, double top/bottom, break_retest)
- `_mtf_context()` — multi-TF gate (Tiers 1-6): trend must align
- `strategy_config.py` — all levers (ATR, MIN_CONFIDENCE, risk, etc.)
- **Backtest = live code**: `oanda_backtest_v2.py` imports `SetAndForgetStrategy` directly.
  No separate logic. Any code change affects both instantly.
- Decision log: `logs/backtest_v2_decisions.json` — full signal audit trail
- Gap log: `logs/backtest_gap_log.jsonl` — every blocked signal with reason

## Commit Log
| Hash    | Change |
|---------|--------|
| 27cd440 | ENGULFING_ONLY=True — #1 fix |
| 9195e2d | Official baseline stamp |
| f517243 | neckline_at_level + weekly_candle_agreement |
| 8a8a7a1 | PATTERN_PREFER_PROXIMITY + neckline fix |
| bffe330 | Analysis scripts (no strategy change) |
| 074fc1a | MIN_CONFIDENCE 0.65→0.75 + ALEX_RUBRIC.md |
| c0236f5 | MIN_CONFIDENCE 0.75→0.77 + MODEL_PROGRESS.md |
