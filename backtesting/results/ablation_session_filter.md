# Session Filter Ablation Study

Generated: 2026-03-11 11:52 UTC  |  Branch: `feat/session-filter-ablation`

## Setup

| | |
|---|---|
| Capital | $8,000 |
| Stop logic | C8 (structural + 3×ATR_1H ceiling + 8-pip floor) |
| Trigger mode | `engulf_or_strict_pin_at_level` (B-Prime) |
| ENGULF_CONFIRM_LOOKBACK_BARS | 2 |
| STRICT_PIN_PATTERN_WHITELIST | head_and_shoulders, inverted_head_and_shoulders |
| MIN_RR_STANDARD | 2.5 |
| MIN_CONFIDENCE | 0.77 |

### Variants

| Variant | THU_CUTOFF_ET | MON_HARD_BLOCK_END | Description |
|---------|:-------------:|:------------------:|-------------|
| **A** | 09:00 | 08:00 (production) | Baseline — Thu cutoff 09:00 ET, Mon hard-block ends 08:00 ET |
| **B** | 12:00 | 08:00 | Extended Thu — cutoff 12:00 ET (noon), Mon unchanged |
| **C** | 12:00 | 07:00 | Thu+Mon — cutoff 12:00 ET + Mon hard-block ends 07:00 ET |

### Windows

- **Q1-2025**: 2025-01-01 → 2025-03-31
- **Q2-2025**: 2025-04-01 → 2025-06-30
- **Q3-2025**: 2025-07-01 → 2025-09-30
- **Q4-2025**: 2025-10-01 → 2025-12-31
- **Jan-Feb-2026**: 2026-01-01 → 2026-02-28
- **W1**: 2026-02-17 → 2026-02-21
- **W2**: 2026-02-24 → 2026-02-28
- **live-parity**: 2026-03-02 → 2026-03-08

---

## 1. Q3 2025 Dead Zone Diagnosis

Q3 2025 (Jul–Sep 2025) produced **0 trades** across every prior ablation study.
This section diagnoses why.

### Decision Funnel — Q3 2025 Baseline

| Stage | Count |
|-------|------:|
| Total decisions logged | 0 |
| BLOCKED (inside evaluate) | 0 |
| WAIT / CANDIDATE_WAIT | 0 |
| ENTER (past evaluate) | 0 |
| Actual trades | 0 |

### Top Block Filters

| Filter | Count |
|--------|------:|

### Post-evaluate Block Distribution (Gap Log)

| Gap type | Count |
|----------|------:|
| (none) | — |

**Session blocks**: 0  |  **HTF blocks**: 0

### Verdict

> MARKET_CONDITION: No patterns detected at all (market too choppy / no structure)

---

## 2. Per-Window Breakdown

| Window | Var | T | WR | SumR | AvgR | MaxDD | Worst3L |
|--------|-----|:--:|:--:|:----:|:----:|:-----:|---------|
| Q1-2025 | A | 12 | 1% | +10.50R | +0.875R | 6.4% | -1.17R, -1.07R, -1.02R |
|  | B | 12 | 1% | +10.50R | +0.875R | 6.4% | -1.17R, -1.07R, -1.02R |
|  | C | 12 | 1% | +8.42R | +0.702R | 6.4% | -1.07R, -1.07R, -1.04R |
| | | | | | | | |
| Q2-2025 | A | 12 | 1% | -0.47R | -0.039R | 13.6% | -1.15R, -1.06R, -1.05R |
|  | B | 12 | 1% | -0.47R | -0.039R | 13.6% | -1.15R, -1.06R, -1.05R |
|  | C | 12 | 0% | -1.93R | -0.161R | 19.0% | -1.15R, -1.06R, -1.05R |
| | | | | | | | |
| Q3-2025 | A | 0 | 0% | +0.00R | +0.000R | 0.0% | — |
|  | B | 0 | 0% | +0.00R | +0.000R | 0.0% | — |
|  | C | 0 | 0% | +0.00R | +0.000R | 0.0% | — |
| | | | | | | | |
| Q4-2025 | A | 12 | 1% | +5.16R | +0.430R | 11.8% | -1.08R, -1.07R, +0.39R |
|  | B | 12 | 1% | +5.16R | +0.430R | 11.8% | -1.08R, -1.07R, +0.39R |
|  | C | 12 | 1% | +4.13R | +0.344R | 11.8% | -1.08R, -1.07R, +0.39R |
| | | | | | | | |
| Jan-Feb-2026 | A | 8 | 1% | +3.98R | +0.497R | 6.8% | -1.14R, +0.46R, +0.50R |
|  | B | 8 | 1% | +3.98R | +0.497R | 6.8% | -1.14R, +0.46R, +0.50R |
|  | C | 8 | 1% | +3.98R | +0.497R | 6.8% | -1.14R, +0.46R, +0.50R |
| | | | | | | | |
| W1 | A | 1 | 1% | +0.62R | +0.622R | 0.0% | +0.62R |
|  | B | 1 | 1% | +0.62R | +0.622R | 0.0% | +0.62R |
|  | C | 1 | 1% | +0.62R | +0.622R | 0.0% | +0.62R |
| | | | | | | | |
| W2 | A | 1 | 1% | +0.40R | +0.403R | 0.0% | +0.40R |
|  | B | 1 | 1% | +0.40R | +0.403R | 0.0% | +0.40R |
|  | C | 1 | 1% | +0.40R | +0.403R | 0.0% | +0.40R |
| | | | | | | | |
| live-parity | A | 1 | 1% | +0.46R | +0.464R | 0.0% | +0.46R |
|  | B | 1 | 1% | +0.46R | +0.464R | 0.0% | +0.46R |
|  | C | 1 | 1% | +0.46R | +0.464R | 0.0% | +0.46R |
| | | | | | | | |

## 3. Aggregate Summary

| Variant | Threshold | Trades | WR | SumR | AvgR | Avg MaxDD |
|---------|:---------:|:------:|:--:|:----:|:----:|:---------:|
| **A** | Thu≤9h/Mon≥8h | 47 | 77% | +20.66R | +0.440R | 4.8% |
| **B** | Thu≤12h/Mon≥8h | 47 | 77% | +20.66R (+0.00R vs A) | +0.440R | 4.8% |
| **C** | Thu≤12h/Mon≥7h | 47 | 72% | +16.08R (-4.58R vs A) | +0.342R | 5.5% |

## 4. Unlocked Trade Analysis

### Variant B vs A — Newly Unlocked Trades

No unlocked trades for Variant B.

### Variant C vs A — Newly Unlocked Trades

#### Q1-2025 — 4 unlocked trade(s)

| Pair | Pattern | Dir | Stop(p) | Entry(ET) | Session | R | MAE | MFE | W/L |
|------|---------|-----|:-------:|:---------:|:-------:|:--:|:---:|:---:|:---:|
| GBP/CHF | inverted_head_and_shoulders | long | 34p | Mon 08:00 | LONDON_NY_OVERLAP(1.0) | +0.49R | -0.76R | +1.12R | ✅ W |
| EUR/USD | inverted_head_and_shoulders | long | 14p | Mon 07:00 | LONDON(0.8) | -1.07R | -1.07R | +0.00R | ❌ L |
| USD/CAD | head_and_shoulders | short | 48p | Tue 03:00 | LONDON(0.8) | -1.04R | -1.01R | +0.55R | ❌ L |
| GBP/JPY | inverted_head_and_shoulders | long | 67p | Mon 07:00 | LONDON(0.8) | +0.53R | -0.56R | +1.07R | ✅ W |

#### Q2-2025 — 1 unlocked trade(s)

| Pair | Pattern | Dir | Stop(p) | Entry(ET) | Session | R | MAE | MFE | W/L |
|------|---------|-----|:-------:|:---------:|:-------:|:--:|:---:|:---:|:---:|
| GBP/JPY | inverted_head_and_shoulders | long | 62p | Mon 07:00 | LONDON(0.8) | -1.05R | -1.01R | +0.12R | ❌ L |
> ⚠️  HIGH_CONCENTRATION: GBP/JPY inverted_head_and_shoulders = 54% of window SumR (-1.05R / -1.93R)

#### Q4-2025 — 2 unlocked trade(s)

| Pair | Pattern | Dir | Stop(p) | Entry(ET) | Session | R | MAE | MFE | W/L |
|------|---------|-----|:-------:|:---------:|:-------:|:--:|:---:|:---:|:---:|
| USD/CAD | consolidation_breakout_bullish | long | 22p | Mon 07:00 | LONDON(0.8) | +0.42R | -0.16R | +1.01R | ✅ W |
| USD/CAD | inverted_head_and_shoulders | long | 14p | Mon 07:00 | LONDON(0.8) | +0.43R | -0.33R | +1.08R | ✅ W |

**All windows — 7 unlocked trade(s)**: WR=57%  SumR=-1.29R  AvgR=-0.184R

## 5. Cascade Displacement Analysis

Baseline (A) trades displaced by weekly cap when unlocked trades consumed the slot.

### Variant B displacements vs A

| Window | Displaced trade | Displaced R | Replacement | Replacement R | Net ΔR |
|--------|----------------|:-----------:|-------------|:-------------:|:------:|
| (none) | — | — | — | — | — |

**Net displacement delta (all windows): +0.00R**

### Variant C displacements vs A

| Window | Displaced trade | Displaced R | Replacement | Replacement R | Net ΔR |
|--------|----------------|:-----------:|-------------|:-------------:|:------:|
| Q1-2025 | GBP/JPY head_and_shoulders | +0.58R | GBP/CHF inverted_head_and_shoulders | +0.49R | -0.10R |
| Q1-2025 | USD/CHF head_and_shoulders | -1.17R | EUR/USD inverted_head_and_shoulders | -1.07R | +0.10R |
| Q1-2025 | EUR/USD inverted_head_and_shoulders | +0.61R | USD/CAD head_and_shoulders | -1.04R | -1.66R |
| Q1-2025 | GBP/JPY inverted_head_and_shoulders | +0.95R | GBP/JPY inverted_head_and_shoulders | +0.53R | -0.43R |
| Q2-2025 | GBP/JPY inverted_head_and_shoulders | +0.42R | GBP/JPY inverted_head_and_shoulders | -1.05R | -1.46R |
| Q4-2025 | USD/CAD consolidation_breakout_bullish | +0.97R | USD/CAD consolidation_breakout_bullish | +0.42R | -0.55R |
| Q4-2025 | USD/CAD inverted_head_and_shoulders | +0.92R | USD/CAD inverted_head_and_shoulders | +0.43R | -0.49R |

**Net displacement delta (all windows): -4.58R**

## 6. ATR Floor Check (C8 8-pip floor)

| Window | A violations | B violations | C violations |
|--------|:-----------:|:-----------:|:-----------:|
| Q1-2025 | 0 ✅ | 0 ✅ | 0 ✅ |
| Q2-2025 | 0 ✅ | 0 ✅ | 0 ✅ |
| Q3-2025 | 0 ✅ | 0 ✅ | 0 ✅ |
| Q4-2025 | 0 ✅ | 0 ✅ | 0 ✅ |
| Jan-Feb-2026 | 0 ✅ | 0 ✅ | 0 ✅ |
| W1 | 0 ✅ | 0 ✅ | 0 ✅ |
| W2 | 0 ✅ | 0 ✅ | 0 ✅ |
| live-parity | 0 ✅ | 0 ✅ | 0 ✅ |
| **Total** | **0 ✅** | **0 ✅** | **0 ✅** |

## 7. Verdict

### Summary comparison

| Variant | Thu/Mon config | Trades | SumR | vs A |
|---------|:---------------:|:------:|:----:|:----:|
| A | Thu≤9h / Mon≥8h | 47 | +20.66R | — |
| B | Thu≤12h / Mon≥8h | 47 | +20.66R | +0.00R |
| C | Thu≤12h / Mon≥7h | 47 | +16.08R | -4.58R |

### Decision

**Variant A (baseline)**: +20.66R across 47 trades — production default.
**Variant B: NEUTRAL** — SumR within ±0.5R of baseline (+0.00R). No strong signal either way.

**Variant C: REJECT** — SumR regressed vs baseline (-4.58R).

_Report generated by `scripts/ablation_session_filter.py`._
_Offline replay only — no live changes, no master merge._