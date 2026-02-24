# Alex's Trading Rubric — Extracted from Full Transcript

## The 5-Rule Model (Every Winning Trade)

1. **Round psychological level** — named every time. 157.5, 1.125, 0.889, 1.35, 144, 192. Never a random price.
2. **Pattern AT the level** — H&S neckline, double top peak, or break/retest — always anchored to that round number
3. **EMA confluence** — weekly or 4H EMA is at the same level almost every time
4. **ONE trigger** — 1H or 4H bearish/bullish engulfing candle body-closing through/at the level
5. **No take profit** — set alarm at next structure, close manually or trail to BE

That's the whole model. No trend gates. No Z-scores. No overextension tiers.
**Round number + pattern + EMA + engulfing = enter.**

---

## Trade-by-Trade Breakdown

### Wk1 — GBP/JPY SHORT @ ~192
> "we had the retest exactly how I wanted it… 1H bearish engulfing… entered on the little quick pullback of the next candle"
- Pattern: Price pulled back into round level **192** + 4H EMA rejection
- Trigger: 1H bearish engulfing body-closed below structure
- Entry: London session 3 AM EST, Jul 16

### Wk2 — USD/JPY SHORT @ 157.5
> "clean left head right shoulder… retesting the neckline along with the 1H EMA… Maru tweezer top rejection bearish engulfing… I keep telling you, you need an engulfing candle to enter"
- Pattern: H&S, neckline at round level **157.5**
- Trigger: 1H bearish engulfing at neckline + EMA rejection
- Entry: London session, ~Jul 22

### Wk3 — USD/CHF SHORT @ ~0.889
- Pattern: Double top or H&S at **0.889** (0.875 or 0.9 range)
- Trigger: Engulfing at neckline + EMA confluence
- Bot matched this ✅ at 102% pip capture

### Wk4 — EUR/USD SHORT (Alex's rule violation loss)
- Alex entered counter-trend without proper setup, lost
- Bot should SKIP — no valid pattern at round number
- Bot correctly skipped in baseline ✅

### Wk5 — No trades
- Alex waited, no setup
- Bot should match ✅

### Wk6 — GBP/CHF SHORT @ 1.125
> "ginormous double top right here… neckline of the double top happens to also be at the weekly EMA and the round psychological level 1.12500"
- Pattern: Double top neckline explicitly at **1.125**
- Trigger: Engulfing at neckline + weekly EMA confluence
- Entry: ~August, London session

### Wk7 — GBP/CHF SHORT (continuation ~30p)
- Second entry after Wk6, smaller move
- Same level, continuation engulfing

### Wk8 — USD/JPY SHORT @ 144
> "left head right shoulder… alarm at the shift of structure… once we body close under it we get the retest and then we sell"
- Pattern: H&S at **144** (round level)
- Trigger: Body close under neckline → retest → engulfing
- Entry: ~Aug 23

### Wk9 — NZD/CAD SHORT (Alex's loss)
- Alex took it, lost -70p
- Bot should ideally skip (our rules correctly blocked it in baseline ✅)

### Wk10 — USD/CAD SHORT @ 1.35
> "break and retest to then sell… we got the break of the structure… EMA rejection… re-entered at bearish engulfing confirmation, Evening Star formation under the EMA"
- Pattern: Break of **1.35** → retest from below
- Trigger: Bearish engulfing at retested 1.35 level + EMA
- Entry: ~Sep 22-26, multiple attempts

### Wk11 — USD/CAD SHORT (continuation)
> "entered USD CAD on the sales once again… I entered this trade where I was originally entered… re-enter the trade where I was originally entered"
- Same 1.35 setup, second entry after BE stop
- Same trigger: engulfing at level

### Wk12a — USD/JPY SHORT (Diddy loss)
- Alex entered, got stopped out ("Diddy did the did")
- Bot should correctly block ✅

### Wk12b — GBP/CHF SHORT (600p, +12R)
> "4H bearish engulfing how I wanted from that area… entered at the breakout of the consolidation… second entry on 15min pullback engulfing… third entry at the bottom of the rejection candle"
- Pattern: Consolidation break + double top → break + retest at level
- Trigger: 4H bearish engulfing at break, then 15m engulfing on pullbacks
- Entry: ~Oct (Week 12)

### Wk13 — USD/CHF LONG @ ~0.86 area
> "broke out of that consolidation zone… waiting for the retest… we had the most important thing — our entry signal"
- Pattern: Bullish break + retest at round number (consolidation breakout LONG)
- Trigger: Bullish engulfing at retest (same logic, opposite direction)
- Alex hints at a special entry signal on the live stream

---

## Key Quotes to Keep

- *"No engulfing candle = no trade"* — said in EVERY video
- *"I keep telling you, you need an engulfing Candlestick to enter a short or a bullish engulfing to enter a buy"*
- *"round psychological level… weekly EMA… area of Interest — so I just have three four things that simply make sense at this area"*
- *"set and forget… stop loss only, no take profit"*
- *"I never put a take profit, I always put a stop loss to minimize losses"*
- *"London session 3 AM EST — set alarm, go to sleep"*
- *"one at a time… less the better"*

---

## What the Bot's MTF Gates Are Getting Wrong

The current `_mtf_context` requires 2+ bearish TFs for a short (Tier 1) or specific reversal pattern types (Tiers 2-4). But Alex never says this. He says:
- If the round number is there ✅
- If the pattern is there ✅
- If the EMA is at the level ✅
- If the engulfing fires ✅
→ Enter. Period.

The trend TFs (weekly/daily/4H) are used to IDENTIFY direction bias (looking for shorts vs longs) and to AVOID taking a trade against the obvious multi-week trend — but they are NOT a gate that requires 2+ TF alignment. Alex frequently enters when daily is neutral or even slightly against (Wk10 USD/CAD daily was neutral).

---

## Implementation Priority

1. Wk1 GBP/JPY @ 192 — H&S at round level, EMA confluence, 1H engulfing
2. Wk2 USD/JPY @ 157.5 — H&S neckline retest, EMA, 1H engulfing
3. Wk6 GBP/CHF @ 1.125 — Double top at exact round number
4. Wk10-11 USD/CAD @ 1.35 — Break + retest at round number
5. Wk12b GBP/CHF — 4H engulfing at consolidation break
6. Wk13 USD/CHF LONG — Bullish break + retest
