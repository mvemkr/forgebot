# Variant C Sanity Confirmation
Generated: 2026-03-10 15:22 ET

Offline only. Compares Baseline (A) vs Variant C (struct + 3×ATR_1H ceil) vs Variant C8 (C + 8-pip hard floor).

## 1. Per-Window Summary

| Window | Var | T | WR | SumR | Ret% | MaxDD | AvgR |
|--------|-----|:-:|:--:|:----:|:----:|:-----:|:----:|
| Q1-2025 | A | 11 | 55% | +1.53R | +12.7% | +26.7% | +0.14R |
|  | **C** | 13 | 69% | +23.80R | +418.8% | +16.5% | +1.83R |
|  | **C8** | 13 | 69% | +17.11R | +215.0% | +16.5% | +1.32R |
| | | | | | | | |
| Q2-2025 | A | 12 | 58% | -0.74R | +0.3% | +14.3% | -0.06R |
|  | **C** | 12 | 58% | -0.08R | -16.7% | +30.3% | -0.01R |
|  | **C8** | 12 | 58% | -0.08R | -16.7% | +30.3% | -0.01R |
| | | | | | | | |
| Q3-2025 | A | 0 | 0% | +0.00R | +0.0% | +0.0% | +0.00R |
|  | **C** | 0 | 0% | +0.00R | +0.0% | +0.0% | +0.00R |
|  | **C8** | 0 | 0% | +0.00R | +0.0% | +0.0% | +0.00R |
| | | | | | | | |
| Q4-2025 | A | 12 | 67% | +1.65R | +3.5% | +12.1% | +0.14R |
|  | **C** | 13 | 85% | +5.53R | +55.4% | +17.2% | +0.43R |
|  | **C8** | 13 | 85% | +5.37R | +44.9% | +17.2% | +0.41R |
| | | | | | | | |
| Jan-Feb-2026 | A | 7 | 86% | +2.64R | +15.2% | +1.6% | +0.38R |
|  | **C** | 8 | 88% | +4.15R | +17.1% | +7.1% | +0.52R |
|  | **C8** | 8 | 88% | +4.15R | +17.1% | +7.1% | +0.52R |
| | | | | | | | |
| W1 | A | 1 | 100% | +0.62R | +3.7% | +0.0% | +0.62R |
|  | **C** | 1 | 100% | +0.81R | +4.9% | +0.0% | +0.81R |
|  | **C8** | 1 | 100% | +0.81R | +4.9% | +0.0% | +0.81R |
| | | | | | | | |
| W2 | A | 1 | 100% | +0.40R | +0.3% | +0.0% | +0.40R |
|  | **C** | 1 | 100% | +0.62R | +0.5% | +0.0% | +0.62R |
|  | **C8** | 1 | 100% | +0.62R | +0.5% | +0.0% | +0.62R |
| | | | | | | | |
| live-parity | A | 1 | 100% | +0.46R | +1.4% | +0.0% | +0.46R |
|  | **C** | 1 | 100% | +0.56R | +1.7% | +0.0% | +0.56R |
|  | **C8** | 1 | 100% | +0.56R | +1.7% | +0.0% | +0.56R |
| | | | | | | | |
## 2. Aggregate Totals

| Var | Trades | WR | SumR | Total Ret% | AvgDD | AvgR |
|-----|:------:|:--:|:----:|:----------:|:-----:|:----:|
| **A** | 45 | 67% | +6.56R | +37.1% | +6.8% | +0.15R |
| **C** | 49 | 76% | +35.39R | +481.6% | +8.9% | +0.72R |
| **C8** | 49 | 76% | +28.53R | +267.2% | +8.9% | +0.58R |

## 3. Outlier Audit — Top 10 Unlocked C Trades by R

| # | Date | Pair | Dir | Pattern | Stop | Target | R | MAE | MFE | Stop<8p? | StopType |
|---|------|------|-----|---------|:----:|:------:|:-:|:---:|:---:|:--------:|----------|
| 1 | 2025-03-18 | GBP/USD | short | head_and_shoulders | 3.2p | 100.7p | +10.66R | -3.06R | +11.62R | 🚩 YES | c_struct_raw |
| 2 | 2025-01-16 | GBP/JPY | short | head_and_shoulders | 9.9p | 190.0p | +9.70R | -4.68R | +10.51R | no | c_struct_raw |
| 3 | 2026-01-13 | GBP/CHF | long | consolidation_breakout_bullish | 20.9p | 95.8p | +1.18R | -0.71R | +1.90R | no | c_struct_capped |
| 4 | 2025-03-12 | GBP/USD | long | double_bottom | 34.7p | 87.7p | +1.10R | -0.11R | +1.64R | no | c_struct_capped |
| 5 | 2025-11-03 | USD/CAD | long | consolidation_breakout_bullish | 22.8p | 62.9p | +0.97R | -0.74R | +1.56R | no | c_struct_capped |
| 6 | 2025-03-25 | USD/CAD | short | consolidation_breakout_bearish | 27.6p | 73.4p | +0.95R | -0.75R | +1.52R | no | c_struct_capped |
| 7 | 2026-02-23 | USD/JPY | long | inverted_head_and_shoulders | 27.5p | 153.6p | +0.82R | -0.52R | +1.36R | no | c_struct_raw |
| 8 | 2025-02-24 | EUR/USD | long | inverted_head_and_shoulders | 24.9p | 314.7p | +0.75R | -0.86R | +1.29R | no | c_struct_raw |
| 9 | 2026-02-03 | GBP/USD | short | double_top | 37.3p | 128.7p | +0.71R | -0.93R | +1.25R | no | c_struct_capped |
| 10 | 2025-06-10 | GBP/JPY | long | inverted_head_and_shoulders | 80.6p | 222.9p | +0.69R | -0.92R | +1.23R | no | c_struct_capped |

## 4. Tiny-Stop Audit (All C Trades)

| Threshold | Count | % of C trades |
|-----------|:-----:|:-------------:|
| stop ≤ 3p | 0 | 0% |
| stop ≤ 5p | 1 | 2% 🚩 |
| stop ≤ 8p | 3 | 6% |
| stop ≤ 10p | 5 | 10% |

### C Trades with Stop < 8 Pips (potential realism risk)

| Date | Pair | Dir | Pattern | Stop | R | MAE | MFE |
|------|------|-----|---------|:----:|:-:|:---:|:---:|
| 2025-03-18 | GBP/USD | short | head_and_shoulders | 3.2p | +10.66R | -3.06R | +11.62R |
| 2025-11-17 | USD/CAD | long | inverted_head_and_shoulders | 7.1p | +0.94R | -0.25R | +1.72R |
| 2026-01-20 | GBP/USD | short | head_and_shoulders | 8.0p | -1.19R | -1.80R | +0.55R |

## 5. Realism Audit (Per Window, Unlocked Trades)

**Q1-2025**
  - ⚠ EXTREME_R GBP/JPY 2025-01-16 head_and_shoulders: R=9.70R  stop=9.9p
  - ⚠ CONCENTRATION GBP/JPY 2025-01-16 head_and_shoulders: single trade = 41% of window SumR=23.80R
  - ⚠ TINY_STOP GBP/USD 2025-03-18 head_and_shoulders: stop=3.2p < 8p floor
  - ⚠ EXTREME_R GBP/USD 2025-03-18 head_and_shoulders: R=10.66R  stop=3.2p
  - ⚠ CONCENTRATION GBP/USD 2025-03-18 head_and_shoulders: single trade = 45% of window SumR=23.80R

**Jan-Feb-2026**
  - ⚠ CONCENTRATION GBP/CHF 2026-01-13 consolidation_breakout_bullish: single trade = 28% of window SumR=4.15R
  - ⚠ TINY_STOP GBP/USD 2026-01-20 head_and_shoulders: stop=8.0p < 8p floor

## 6. Robustness Check

| Scenario | SumR | Beats A? |
|----------|:----:|:--------:|
| **A baseline** | +6.56R | — |
| **C full** | +35.39R | ✅ Yes |
| **C ex top-1** (GBP/USD 2025-03-18 3p +10.66R) | +24.73R | ✅ Yes |
| **C ex top-2** (also GBP/JPY 2025-01-16 10p +9.70R) | +15.03R | ✅ Yes |

### Top-2 Outlier Concentration

- Top-2 trades contribute **+20.36R** out of **+35.39R** total C SumR (58%)
- Top-2 trades contribute **71%** of the C vs A SumR delta (+28.83R)
- ⚠ HIGH concentration — delta is largely driven by two outlier trades

## 7. Variant C8 — 8-Pip Hard Floor Applied

*Same as Variant C but enforces stop ≥ 8 pips. Trades whose structural stop is < 8 pips are bumped to 8 pips. The backtester ATR-floor gate is still bypassed to isolate the effect.*

- **Total trades**: 49
- **Win rate**: 76%
- **SumR**: +28.53R
- **Avg R**: +0.58R
- **Total Ret%**: +267.2%
- **Avg MaxDD**: +8.9%
- **Unlocked vs A**: 17 trades
- **Beats A?**: ✅ Yes  (A=+6.56R  C8=+28.53R)

### C8 Per-Window

| Window | T | WR | SumR | Ret% | MaxDD | vs A |
|--------|:-:|:--:|:----:|:----:|:-----:|:----:|
| Q1-2025 | 13 | 69% | +17.11R | +215.0% | +16.5% | ✅ +15.58R |
| Q2-2025 | 12 | 58% | -0.08R | -16.7% | +30.3% | ✅ +0.66R |
| Q3-2025 | 0 | 0% | +0.00R | +0.0% | +0.0% | ✅ +0.00R |
| Q4-2025 | 13 | 85% | +5.37R | +44.9% | +17.2% | ✅ +3.72R |
| Jan-Feb-2026 | 8 | 88% | +4.15R | +17.1% | +7.1% | ✅ +1.51R |
| W1 | 1 | 100% | +0.81R | +4.9% | +0.0% | ✅ +0.19R |
| W2 | 1 | 100% | +0.62R | +0.5% | +0.0% | ✅ +0.22R |
| live-parity | 1 | 100% | +0.56R | +1.7% | +0.0% | ✅ +0.09R |

## 8. Recommendation

| Check | Result |
|-------|--------|
| C full beats A | ✅ |
| C ex top-1 beats A | ✅ |
| C ex top-2 beats A | ✅ |
| C8 (8-pip floor) beats A | ✅ |
| Sub-8-pip stops present | 🚩 Yes |
| Top-2 outliers >60% of delta | 🚩 Yes |

### Verdict: **⚠ PROMOTE WITH GUARDRAIL + MONITOR — top-2 concentration is elevated**

C8 beats A robustly, but top-2 outliers explain >60% of the delta. Promote C8 (8-pip floor) and monitor first 20 live trades closely.

---
*Promotion decision deferred to Mike.*