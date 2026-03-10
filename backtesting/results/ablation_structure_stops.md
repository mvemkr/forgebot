# Structural Stop Ablation Study
Generated: 2026-03-10 11:16 ET

## Config
| Setting | Value |
|---------|-------|
| ENTRY_TRIGGER_MODE | `engulf_or_strict_pin_at_level` |
| STRICT_PIN_PATTERN_WHITELIST | `['head_and_shoulders', 'inverted_head_and_shoulders']` |
| ENGULF_CONFIRM_LOOKBACK_BARS | `2` |
| MIN_CONFIDENCE | `0.77` |
| MIN_RR (unchanged) | `2.5` |
| RECOVERY_MIN_RR (unchanged) | `3.0` |
| ORIG ATR_MIN_MULTIPLIER | `0.15` |

## Variants
| ID | Label | Stop Logic | ATR Floor Bypassed |
|----|-------|------------|--------------------|
| A | Baseline (production stop) | see spec | No |
| B | Structural Pivot (no buffer, no bounds) | see spec | Yes ⚠ |
| C | Structural + ATR Ceiling (3×ATR_1H cap) | see spec | Yes ⚠ |
| D | Structural + ATR Noise Buffer (+0.5×ATR) | see spec | No |

## Per-Window Results

| Window | Var | Trades | WR | AvgR | SumR | Ret% | MaxDD | Expectancy | Worst3 |
|--------|-----|:------:|:--:|:----:|:----:|:----:|:-----:|:----------:|--------|
| **Q1-2025** | **A** | 11 | 55% | +0.14R | +1.53R | +12.7% | +26.7% | +0.14R | -1.17R  -1.08R  -1.03R |
| Q1-2025 | **B** | 11 | 73% | +0.78R | +8.55R | +140.1% | +19.3% | +0.78R | -1.24R  -1.05R  -1.03R |
| Q1-2025 | **C** | 13 | 69% | +1.83R | +23.80R | +418.8% | +16.5% | +1.83R | -1.24R  -1.10R  -1.03R |
| Q1-2025 | **D** | 11 | 73% | +0.50R | +5.46R | +36.5% | +19.0% | +0.50R | -1.15R  -1.04R  -1.02R |
| | | | | | | | | | |
| **Q2-2025** | **A** | 12 | 58% | -0.06R | -0.74R | +0.3% | +14.3% | -0.06R | -1.15R  -1.06R  -1.05R |
| Q2-2025 | **B** | 12 | 58% | -0.03R | -0.33R | -15.2% | +28.8% | -0.03R | -1.20R  -1.07R  -1.06R |
| Q2-2025 | **C** | 12 | 58% | -0.01R | -0.08R | -16.7% | +30.3% | -0.01R | -1.20R  -1.07R  -1.06R |
| Q2-2025 | **D** | 12 | 67% | +0.04R | +0.52R | +3.5% | +11.7% | +0.04R | -1.11R  -1.10R  -1.05R |
| | | | | | | | | | |
| **Q3-2025** | **A** | 0 | 0% | +0.00R | +0.00R | +0.0% | +0.0% | +0.00R | — |
| Q3-2025 | **B** | 0 | 0% | +0.00R | +0.00R | +0.0% | +0.0% | +0.00R | — |
| Q3-2025 | **C** | 0 | 0% | +0.00R | +0.00R | +0.0% | +0.0% | +0.00R | — |
| Q3-2025 | **D** | 0 | 0% | +0.00R | +0.00R | +0.0% | +0.0% | +0.00R | — |
| | | | | | | | | | |
| **Q4-2025** | **A** | 12 | 67% | +0.14R | +1.65R | +3.5% | +12.1% | +0.14R | -1.08R  -1.06R  -1.05R |
| Q4-2025 | **B** | 13 | 77% | +0.29R | +3.81R | +26.8% | +9.1% | +0.29R | -1.10R  -1.07R  -1.05R |
| Q4-2025 | **C** | 13 | 85% | +0.43R | +5.53R | +55.4% | +17.2% | +0.43R | -1.10R  -1.07R  +0.42R |
| Q4-2025 | **D** | 12 | 67% | +0.10R | +1.19R | +1.4% | +13.4% | +0.10R | -1.08R  -1.06R  -1.04R |
| | | | | | | | | | |
| **Jan-Feb-2026** | **A** | 7 | 86% | +0.38R | +2.64R | +15.2% | +1.6% | +0.38R | -1.05R  +0.46R  +0.46R |
| Jan-Feb-2026 | **B** | 7 | 57% | -0.14R | -0.97R | -9.5% | +13.9% | -0.14R | -1.53R  -1.19R  -1.11R |
| Jan-Feb-2026 | **C** | 8 | 88% | +0.52R | +4.15R | +17.1% | +7.1% | +0.52R | -1.19R  +0.55R  +0.58R |
| Jan-Feb-2026 | **D** | 6 | 50% | -0.21R | -1.28R | -4.3% | +9.2% | -0.21R | -1.06R  -1.04R  -1.04R |
| | | | | | | | | | |
| **W1** | **A** | 1 | 100% | +0.62R | +0.62R | +3.7% | +0.0% | +0.62R | +0.62R |
| W1 | **B** | 1 | 100% | +0.81R | +0.81R | +4.9% | +0.0% | +0.81R | +0.81R |
| W1 | **C** | 1 | 100% | +0.81R | +0.81R | +4.9% | +0.0% | +0.81R | +0.81R |
| W1 | **D** | 1 | 100% | +0.78R | +0.78R | +4.7% | +0.0% | +0.78R | +0.78R |
| | | | | | | | | | |
| **W2** | **A** | 1 | 100% | +0.40R | +0.40R | +0.3% | +0.0% | +0.40R | +0.40R |
| W2 | **B** | 1 | 100% | +0.62R | +0.62R | +0.5% | +0.0% | +0.62R | +0.62R |
| W2 | **C** | 1 | 100% | +0.62R | +0.62R | +0.5% | +0.0% | +0.62R | +0.62R |
| W2 | **D** | 1 | 0% | -1.08R | -1.08R | -0.8% | +0.8% | -1.08R | -1.08R |
| | | | | | | | | | |
| **live-parity** | **A** | 1 | 100% | +0.46R | +0.46R | +1.4% | +0.0% | +0.46R | +0.46R |
| live-parity | **B** | 1 | 100% | +0.56R | +0.56R | +1.7% | +0.0% | +0.56R | +0.56R |
| live-parity | **C** | 1 | 100% | +0.56R | +0.56R | +1.7% | +0.0% | +0.56R | +0.56R |
| live-parity | **D** | 1 | 100% | +0.65R | +0.65R | +2.0% | +0.0% | +0.65R | +0.65R |
| | | | | | | | | | |
## ATR Floor Telemetry (Variants B / C / D)

| Window | Var | Stop Candidates | Floor Rejections | Fallback to ATR | Floor Rej % |
|--------|-----|:---------------:|:----------------:|:---------------:|:-----------:|
| Q1-2025 | B | 144 | 6 | 0 | 4% |
| Q1-2025 | C | 119 | 7 | 0 | 6% |
| Q1-2025 | D | 134 | 4 | 0 | 3% |
| Q2-2025 | B | 141 | 4 | 0 | 3% |
| Q2-2025 | C | 114 | 2 | 0 | 2% |
| Q2-2025 | D | 131 | 3 | 0 | 2% |
| Q3-2025 | B | 0 | 0 | 0 | n/a |
| Q3-2025 | C | 0 | 0 | 0 | n/a |
| Q3-2025 | D | 0 | 0 | 0 | n/a |
| Q4-2025 | B | 134 | 17 | 8 | 13% |
| Q4-2025 | C | 131 | 16 | 8 | 12% |
| Q4-2025 | D | 132 | 17 | 8 | 13% |
| Jan-Feb-2026 | B | 112 | 6 | 0 | 5% |
| Jan-Feb-2026 | C | 91 | 5 | 0 | 5% |
| Jan-Feb-2026 | D | 107 | 7 | 0 | 7% |
| W1 | B | 12 | 0 | 0 | 0% |
| W1 | C | 12 | 0 | 0 | 0% |
| W1 | D | 12 | 0 | 0 | 0% |
| W2 | B | 9 | 0 | 0 | 0% |
| W2 | C | 9 | 0 | 0 | 0% |
| W2 | D | 9 | 0 | 0 | 0% |
| live-parity | B | 8 | 0 | 0 | 0% |
| live-parity | C | 8 | 0 | 0 | 0% |
| live-parity | D | 8 | 0 | 0 | 0% |

## Stop Width Distribution (pips)

| Window | Var | Trades | Stop p50 | Min | Max | Type Distribution |
|--------|-----|:------:|:--------:|:---:|:---:|-------------------|
| Q1-2025 | A | 11 | 39p | 11p | 158p | neckline_retest_swing:10  structural_anchor:1 |
| Q1-2025 | B | 11 | 29p | 8p | 153p | b_struct_raw:11 |
| Q1-2025 | C | 13 | 25p | 3p | 142p | c_struct_capped:4  c_struct_raw:9 |
| Q1-2025 | D | 11 | 39p | 13p | 173p | d_struct_buffered:11 |
| Q2-2025 | A | 12 | 55p | 20p | 116p | neckline_retest_swing:12 |
| Q2-2025 | B | 12 | 51p | 15p | 111p | b_struct_raw:12 |
| Q2-2025 | C | 12 | 51p | 15p | 107p | c_struct_capped:3  c_struct_raw:9 |
| Q2-2025 | D | 12 | 71p | 28p | 139p | d_struct_buffered:12 |
| Q3-2025 | A | 0 | —p | —p | —p | — |
| Q3-2025 | B | 0 | —p | —p | —p | — |
| Q3-2025 | C | 0 | —p | —p | —p | — |
| Q3-2025 | D | 0 | —p | —p | —p | — |
| Q4-2025 | A | 12 | 26p | 14p | 42p | neckline_retest_swing:12 |
| Q4-2025 | B | 13 | 22p | 7p | 37p | b_struct_raw:13 |
| Q4-2025 | C | 13 | 23p | 7p | 55p | c_struct_capped:3  c_struct_raw:10 |
| Q4-2025 | D | 12 | 29p | 11p | 48p | d_struct_buffered:12 |
| Jan-Feb-2026 | A | 7 | 33p | 21p | 55p | atr_fallback:1  neckline_retest_swing:6 |
| Jan-Feb-2026 | B | 7 | 18p | 4p | 50p | b_struct_raw:7 |
| Jan-Feb-2026 | C | 8 | 28p | 8p | 50p | c_struct_capped:2  c_struct_raw:6 |
| Jan-Feb-2026 | D | 6 | 39p | 20p | 67p | d_struct_buffered:6 |
| W1 | A | 1 | 34p | 34p | 34p | neckline_retest_swing:1 |
| W1 | B | 1 | 29p | 29p | 29p | b_struct_raw:1 |
| W1 | C | 1 | 29p | 29p | 29p | c_struct_raw:1 |
| W1 | D | 1 | 41p | 41p | 41p | d_struct_buffered:1 |
| W2 | A | 1 | 15p | 15p | 15p | neckline_retest_swing:1 |
| W2 | B | 1 | 12p | 12p | 12p | b_struct_raw:1 |
| W2 | C | 1 | 12p | 12p | 12p | c_struct_raw:1 |
| W2 | D | 1 | 18p | 18p | 18p | d_struct_buffered:1 |
| live-parity | A | 1 | 56p | 56p | 56p | neckline_retest_swing:1 |
| live-parity | B | 1 | 51p | 51p | 51p | b_struct_raw:1 |
| live-parity | C | 1 | 51p | 51p | 51p | c_struct_raw:1 |
| live-parity | D | 1 | 72p | 72p | 72p | d_struct_buffered:1 |

## Aggregate Totals (all windows)

| Var | Total Trades | Total SumR | Wins | Losses | WR | AvgR | AvgDD | Total Ret% |
|-----|:------------:|:----------:|:----:|:------:|:--:|:----:|:-----:|:----------:|
| **A** | 45 | +6.56R | 30 | 15 | 67% | +0.15R | +6.8% | +37.1% |
| **B** | 46 | +13.05R | 32 | 14 | 70% | +0.28R | +8.9% | +149.2% |
| **C** | 49 | +35.39R | 37 | 12 | 76% | +0.72R | +8.9% | +481.6% |
| **D** | 44 | +6.24R | 29 | 15 | 66% | +0.14R | +6.8% | +43.1% |

## Unlock Analysis: A → B

Total unlocked (A → B): **7** trades
- WR: 57%
- AvgR: +0.07R
- SumR: +0.52R
- Avg MAE: -1.44R
- Avg MFE: +1.14R

| Window | Pair | Dir | Pattern | StopPips | TargetPips | R | MAE | MFE | StopType |
|--------|------|-----|---------|:--------:|:----------:|:-:|:---:|:---:|----------|
| 2025-03-18 | GBP/USD | short | head_and_shoulders | 15p | 5.9R target | +1.97R | -0.97R | +2.57R | b_struct_raw |
| 2025-03-24 | GBP/JPY | long | inverted_head_and_shoulders | 41p | 16.8R target | +1.13R | -0.46R | +1.70R | b_struct_raw |
| 2025-12-10 | USD/CHF | short | head_and_shoulders | 24p | 2.9R target | +0.42R | -0.73R | +1.01R | b_struct_raw |
| 2026-01-14 | USD/CHF | long | inverted_head_and_shoulders | 4p | 24.6R target | -1.53R | -3.63R | +0.00R | b_struct_raw |
| 2026-01-20 | GBP/USD | short | head_and_shoulders | 8p | 16.8R target | -1.19R | -1.80R | +0.55R | b_struct_raw |
| 2026-02-03 | USD/JPY | short | head_and_shoulders | 11p | 12.3R target | -1.11R | -1.94R | +0.75R | b_struct_raw |
| 2026-02-23 | USD/JPY | long | inverted_head_and_shoulders | 28p | 5.6R target | +0.82R | -0.52R | +1.36R | b_struct_raw |

### Displacement Table (A → B)

| Displaced Trade | Displaced R | Replacement Trade | Replacement R | Net ΔR |
|-----------------|:-----------:|-------------------|:-------------:|:------:|
| GBP/JPY long 2025-03-19 | -1.03R | GBP/USD short 2025-03-18 | +1.97R | +3.00R |
| USD/CAD short 2025-03-26 | -1.08R | GBP/JPY long 2025-03-24 | +1.13R | +2.22R |
| GBP/CHF short 2026-01-15 | +0.61R | USD/CHF short 2025-12-10 | +0.42R | -0.19R |
| EUR/USD short 2026-01-20 | +0.46R | USD/CHF long 2026-01-14 | -1.53R | -1.99R |
| USD/JPY short 2026-02-10 | +1.03R | GBP/USD short 2026-01-20 | -1.19R | -2.22R |
| GBP/USD short 2026-02-23 | -1.05R | USD/JPY short 2026-02-03 | -1.11R | -0.07R |
| — | +0.00R | USD/JPY long 2026-02-23 | +0.82R | +0.82R |
| **TOTAL** | | | | **+1.58R** |

## Unlock Analysis: A → C

Total unlocked (A → C): **17** trades
- WR: 76%
- AvgR: +1.45R
- SumR: +24.69R
- Avg MAE: -1.35R
- Avg MFE: +2.30R

| Window | Pair | Dir | Pattern | StopPips | TargetPips | R | MAE | MFE | StopType |
|--------|------|-----|---------|:--------:|:----------:|:-:|:---:|:---:|----------|
| 2025-01-02 | GBP/JPY | short | head_and_shoulders | 136p | 3.1R target | -1.02R | -1.09R | +0.25R | c_struct_capped |
| 2025-01-16 | GBP/JPY | short | head_and_shoulders | 10p | 19.2R target | +9.70R | -4.68R | +10.51R | c_struct_raw |
| 2025-01-20 | USD/JPY | long | inverted_head_and_shoulders | 12p | 14.9R target | -1.10R | -3.70R | +0.44R | c_struct_raw |
| 2025-02-24 | EUR/USD | long | inverted_head_and_shoulders | 25p | 12.6R target | +0.75R | -0.86R | +1.29R | c_struct_raw |
| 2025-03-12 | GBP/USD | long | double_bottom | 35p | 2.5R target | +1.10R | -0.11R | +1.64R | c_struct_capped |
| 2025-03-18 | GBP/USD | short | head_and_shoulders | 3p | 31.5R target | +10.66R | -3.06R | +11.62R | c_struct_raw |
| 2025-03-25 | USD/CAD | short | consolidation_breakout_bearish | 28p | 2.7R target | +0.95R | -0.75R | +1.52R | c_struct_capped |
| 2025-05-06 | USD/CAD | short | consolidation_breakout_bearish | 44p | 2.7R target | -1.05R | -1.17R | +0.83R | c_struct_capped |
| 2025-06-10 | GBP/JPY | long | inverted_head_and_shoulders | 81p | 2.8R target | +0.69R | -0.92R | +1.23R | c_struct_capped |
| 2025-10-01 | GBP/JPY | short | head_and_shoulders | 55p | 3.2R target | +0.54R | -0.64R | +1.09R | c_struct_capped |
| 2025-11-03 | USD/CAD | long | consolidation_breakout_bullish | 23p | 2.8R target | +0.97R | -0.74R | +1.56R | c_struct_capped |
| 2025-11-12 | USD/JPY | long | consolidation_breakout_bullish | 49p | 2.7R target | +0.56R | -0.56R | +1.09R | c_struct_capped |
| 2025-12-10 | USD/CHF | short | head_and_shoulders | 24p | 2.9R target | +0.42R | -0.73R | +1.01R | c_struct_raw |
| 2026-01-13 | GBP/CHF | long | consolidation_breakout_bullish | 21p | 4.6R target | +1.18R | -0.71R | +1.90R | c_struct_capped |
| 2026-01-20 | GBP/USD | short | head_and_shoulders | 8p | 16.8R target | -1.19R | -1.80R | +0.55R | c_struct_raw |
| 2026-02-03 | GBP/USD | short | double_top | 37p | 3.5R target | +0.71R | -0.93R | +1.25R | c_struct_capped |
| 2026-02-23 | USD/JPY | long | inverted_head_and_shoulders | 28p | 5.6R target | +0.82R | -0.52R | +1.36R | c_struct_raw |

### Displacement Table (A → C)

| Displaced Trade | Displaced R | Replacement Trade | Replacement R | Net ΔR |
|-----------------|:-----------:|-------------------|:-------------:|:------:|
| GBP/JPY short 2025-01-09 | +0.88R | GBP/JPY short 2025-01-02 | -1.02R | -1.90R |
| USD/JPY long 2025-01-16 | +2.09R | GBP/JPY short 2025-01-16 | +9.70R | +7.61R |
| GBP/JPY short 2025-02-24 | -1.03R | USD/JPY long 2025-01-20 | -1.10R | -0.07R |
| GBP/USD long 2025-03-12 | -1.03R | EUR/USD long 2025-02-24 | +0.75R | +1.78R |
| USD/CAD short 2025-03-26 | -1.08R | GBP/USD long 2025-03-12 | +1.10R | +2.18R |
| GBP/JPY long 2025-05-06 | -1.05R | GBP/USD short 2025-03-18 | +10.66R | +11.71R |
| GBP/JPY long 2025-06-10 | +0.55R | USD/CAD short 2025-03-25 | +0.95R | +0.40R |
| USD/CAD long 2025-10-01 | +0.64R | USD/CAD short 2025-05-06 | -1.05R | -1.69R |
| USD/JPY long 2025-11-04 | -1.04R | GBP/JPY long 2025-06-10 | +0.69R | +1.73R |
| USD/JPY long 2025-11-13 | -1.05R | GBP/JPY short 2025-10-01 | +0.54R | +1.58R |
| GBP/CHF short 2026-01-15 | +0.61R | USD/CAD long 2025-11-03 | +0.97R | +0.36R |
| EUR/USD short 2026-01-20 | +0.46R | USD/JPY long 2025-11-12 | +0.56R | +0.10R |
| GBP/USD short 2026-02-23 | -1.05R | USD/CHF short 2025-12-10 | +0.42R | +1.47R |
| — | +0.00R | GBP/CHF long 2026-01-13 | +1.18R | +1.18R |
| — | +0.00R | GBP/USD short 2026-01-20 | -1.19R | -1.19R |
| — | +0.00R | GBP/USD short 2026-02-03 | +0.71R | +0.71R |
| — | +0.00R | USD/JPY long 2026-02-23 | +0.82R | +0.82R |
| **TOTAL** | | | | **+26.79R** |

## Unlock Analysis: A → D

Total unlocked (A → D): **4** trades
- WR: 75%
- AvgR: +0.40R
- SumR: +1.61R
- Avg MAE: -0.72R
- Avg MFE: +1.18R

| Window | Pair | Dir | Pattern | StopPips | TargetPips | R | MAE | MFE | StopType |
|--------|------|-----|---------|:--------:|:----------:|:-:|:---:|:---:|----------|
| 2025-03-18 | GBP/USD | short | head_and_shoulders | 20p | 4.5R target | +1.39R | -0.74R | +1.97R | d_struct_buffered |
| 2025-03-24 | GBP/JPY | long | inverted_head_and_shoulders | 56p | 12.3R target | +0.70R | -0.34R | +1.25R | d_struct_buffered |
| 2025-05-20 | USD/JPY | short | head_and_shoulders | 94p | 3.6R target | +0.59R | -0.76R | +1.10R | d_struct_buffered |
| 2026-02-03 | USD/JPY | short | head_and_shoulders | 20p | 6.7R target | -1.06R | -1.06R | +0.41R | d_struct_buffered |

### Displacement Table (A → D)

| Displaced Trade | Displaced R | Replacement Trade | Replacement R | Net ΔR |
|-----------------|:-----------:|-------------------|:-------------:|:------:|
| GBP/JPY long 2025-03-19 | -1.03R | GBP/USD short 2025-03-18 | +1.39R | +2.42R |
| USD/CAD short 2025-03-26 | -1.08R | GBP/JPY long 2025-03-24 | +0.70R | +1.78R |
| EUR/USD short 2025-05-19 | -1.02R | USD/JPY short 2025-05-20 | +0.59R | +1.60R |
| GBP/CHF short 2026-01-15 | +0.61R | USD/JPY short 2026-02-03 | -1.06R | -1.67R |
| USD/JPY short 2026-02-10 | +1.03R | — | +0.00R | -1.03R |
| **TOTAL** | | | | **+3.11R** |

## Per-Window Unlock Summary

| Window | B_unlocked | B_SumR | B_FloorRej | C_unlocked | C_SumR | C_FloorRej | D_unlocked | D_SumR | D_FloorRej |
|--------|:----------:|:------:|:----------:|:----------:|:------:|:----------:|:----------:|:------:|:----------:|
| Q1-2025 | 2 | +3.10R | 6 | 7 | +21.03R | 7 | 2 | +2.09R | 4 |
| Q2-2025 | 0 | +0.00R | 4 | 2 | -0.36R | 2 | 1 | +0.59R | 3 |
| Q3-2025 | 0 | +0.00R | 0 | 0 | +0.00R | 0 | 0 | +0.00R | 0 |
| Q4-2025 | 1 | +0.42R | 17 | 4 | +2.49R | 16 | 0 | +0.00R | 17 |
| Jan-Feb-2026 | 4 | -3.01R | 6 | 4 | +1.52R | 5 | 1 | -1.06R | 7 |
| W1 | 0 | +0.00R | 0 | 0 | +0.00R | 0 | 0 | +0.00R | 0 |
| W2 | 0 | +0.00R | 0 | 0 | +0.00R | 0 | 0 | +0.00R | 0 |
| live-parity | 0 | +0.00R | 0 | 0 | +0.00R | 0 | 0 | +0.00R | 0 |

### B-Unlocked: Pair Distribution
  - GBP/USD: 2 trades, SumR +0.79R
  - USD/CHF: 2 trades, SumR -1.10R
  - USD/JPY: 2 trades, SumR -0.29R
  - GBP/JPY: 1 trades, SumR +1.13R

### B-Unlocked: Pattern Distribution
  - head_and_shoulders: 4
  - inverted_head_and_shoulders: 3

### B-Unlocked: Stop Width vs Baseline
  - Baseline trades stop p50: 33p
  - B-unlocked stop p50: 15p

### C-Unlocked: Pair Distribution
  - GBP/JPY: 4 trades, SumR +9.91R
  - GBP/USD: 4 trades, SumR +11.28R
  - USD/JPY: 3 trades, SumR +0.28R
  - USD/CAD: 3 trades, SumR +0.87R
  - EUR/USD: 1 trades, SumR +0.75R
  - USD/CHF: 1 trades, SumR +0.42R
  - GBP/CHF: 1 trades, SumR +1.18R

### C-Unlocked: Pattern Distribution
  - head_and_shoulders: 6
  - inverted_head_and_shoulders: 4
  - consolidation_breakout_bullish: 3
  - consolidation_breakout_bearish: 2
  - double_bottom: 1
  - double_top: 1

### C-Unlocked: Stop Width vs Baseline
  - Baseline trades stop p50: 33p
  - C-unlocked stop p50: 28p

### D-Unlocked: Pair Distribution
  - USD/JPY: 2 trades, SumR -0.47R
  - GBP/USD: 1 trades, SumR +1.39R
  - GBP/JPY: 1 trades, SumR +0.70R

### D-Unlocked: Pattern Distribution
  - head_and_shoulders: 3
  - inverted_head_and_shoulders: 1

### D-Unlocked: Stop Width vs Baseline
  - Baseline trades stop p50: 33p
  - D-unlocked stop p50: 56p

## Summary & Recommendation

- Variant B vs A: total SumR +13.05R (delta +6.49R)
- Variant C vs A: total SumR +35.39R (delta +28.83R)
- Variant D vs A: total SumR +6.24R (delta -0.33R)

*(Promotion decision deferred to Mike — this is a report-only study.)*