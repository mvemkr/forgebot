# Offensive Plan — Expansion & Promotion Roadmap
_Planning document only. No implementation until explicitly authorized._
_Last updated: 2026-03-01_

---

## 1. Expansion Frequency Engineering Plan

### 1.1 Metrics Logged Per Scan (already available, no code change needed)

Every `SCAN_HEARTBEAT` entry in `decision_log.jsonl` carries a `regime_score` block.
The fields relevant to expansion detection are:

| Field | Source | Description |
|-------|--------|-------------|
| `wd_aligned` | `regime_score.wd_aligned` | Weekly & Daily trend in same direction |
| `atr_ratio` | `regime_score.atr_ratio` | H4 ATR / 20-bar avg H4 ATR |
| `last5_sum_r` | `regime_score.last5_sum_r` | Sum of R on last 5 closed trades |
| `last10_sum_r` | `regime_score.last10_sum_r` (derived) | Sum of R on last 10 closed trades |
| `loss_streak` | `regime_score.loss_streak` | Consecutive losses since last win |
| `dd_pct` | `stats.drawdown_pct` in `bot_state.json` | Peak-to-current equity drawdown % |
| `risk_mode` | `regime_score.risk_mode` | AUTO-computed mode at scan time |
| `eligible_high` | `regime_score.eligible_high` | Whether HIGH gate cleared |
| `eligible_extreme` | `regime_score.eligible_extreme` | Whether EXTREME gate cleared |

**Logging requirement (no code change):** all fields above are already written per scan.
Two-week measurement reads them from `decision_log.jsonl` and `logs/forex_orchestrator.log`.

---

### 1.2 EXPANSION Regime — Explicit Numeric Thresholds

EXPANSION is the state in which the engine is allowed to promote to HIGH or EXTREME.
All four conditions below must be simultaneously true at entry time (instantaneous evaluation):

| # | Condition | Threshold | Rationale |
|---|-----------|-----------|-----------|
| 1 | `wd_aligned` | `True` | Weekly and Daily trend agree — directional clarity |
| 2 | `atr_ratio` | `≥ 1.10` | H4 ATR expanding vs 20-bar avg — volatility present |
| 3 | `last5_sum_r` | `> 0.0` | Edge is positive over recent window |
| 4 | `loss_streak` | `≤ 1` | Not in a drawdown spiral |

HIGH promotion fires when ALL 4 are met.

EXTREME adds two further gates on top of HIGH:

| # | Condition | Threshold |
|---|-----------|-----------|
| 5 | `last10_sum_r` | `≥ 1.5` | Sustained edge over broader window |
| 6 | `dd_pct` | `< 10.0%` | Near peak equity — not recovering from hole |

A score of 4/4 (all HIGH conditions) is required before EXTREME is evaluated.
EXTREME implicitly requires `loss_streak == 0` (score == 4 forces it).

### 1.3 Confirmation Persistence Requirements

- **HIGH promotion**: ALL-4 conditions must be true at the moment of entry evaluation
  (instantaneous, `compute_risk_mode(..., instantaneous=True)`).
  The 2-bar hysteresis applies only inside the H4 time-sampling loop — not at entry level.
- **EXTREME promotion**: Same instantaneous rule; no additional bars required.
- **Demotion to MEDIUM**: Immediate if `loss_streak ≥ 2` OR `atr_ratio < 1.00` (hard reset).
  Otherwise 2 consecutive failing H4 bars trigger demotion (hysteresis).
- **Demotion to LOW**: Immediate on `loss_streak ≥ 3` (Chop Shield territory).

---

## 2. Risk Escalation Policy in Expansion

### 2.1 Mode → Risk Parameters

| Mode | Trigger | Base Risk % | Multiplier | Effective Risk % | Notes |
|------|---------|-------------|------------|-----------------|-------|
| LOW | score < 2 OR streak ≥ 3 | 6.0% | 0.5× | 3.0% | Chop / cold streak |
| MEDIUM | Default (score 1–3 not qualifying HIGH) | 6.0% | 1.0× | 6.0% | Normal operation |
| HIGH | ALL-4 met | 6.0% | 1.5× | 9.0% | Expansion active |
| EXTREME | ALL-4 + last10≥1.5 + dd<10% | 6.0% | 2.0× | 12.0% | Peak expansion |

_Base 6.0% applies at the ≤$8K tier. Tier steps at $8K→$15K→$30K→$30K+ raise the base._

### 2.2 When HIGH Activates

HIGH activates at entry time when the instantaneous check clears all four conditions:
- `wd_aligned == True`
- `atr_ratio ≥ 1.10`
- `last5_sum_r > 0.0`
- `loss_streak ≤ 1`

It is suppressed immediately (reverts to MEDIUM) on any hard-reset event:
- A new loss that pushes `loss_streak ≥ 2`
- `atr_ratio` dropping below `1.00` on the H4 sampling loop

### 2.3 When EXTREME Activates

EXTREME requires HIGH conditions PLUS:
- `last10_sum_r ≥ 1.5` — extended run of positive R
- `dd_pct < 10.0%` — near equity peak (not clawing back)
- `loss_streak == 0` (implied by score == 4)

Both conditions are checked instantaneously at entry. EXTREME is not pinnable from the
dashboard — it must be earned by the regime. A manual pin to HIGH is the ceiling for
operator-forced escalation.

### 2.4 Trade/Week Caps During Expansion

These caps govern NEW entries taken per calendar week (Monday 00:00 UTC → Sunday 23:59 UTC):

| Mode | Weekly Cap (new entries) |
|------|--------------------------|
| LOW | 1 |
| MEDIUM | 2 |
| HIGH | 3 |
| EXTREME | 4 |

Caps are enforced by `regime_weekly_caps` in `control_state.py`.
During Chop Shield recovery, an additional hard cap of 1 trade/week is applied
regardless of mode (recovery selectivity gate).

### 2.5 Hard Cutoffs — Expansion Blocked Regardless of Score

Any of the following conditions forces mode ≤ MEDIUM immediately:

| Cutoff | Value | Action |
|--------|-------|--------|
| Consecutive losses | ≥ 3 | Chop Shield fires: 48h AUTO_PAUSE then recovery mode |
| Weekly DD | ≥ 40% in 7 days | REGROUP mode (14-day cooldown) |
| `atr_ratio` | < 1.00 | Hard reset hysteresis; demotes to MEDIUM within 2 bars |
| Manual pin | dashboard override | `risk_mode_pinned` wins over AUTO in live; backtest ignores pin |

---

## 3. Two-Week LIVE_PAPER Measurement Plan

**Window:** Monday 2026-03-02 through Sunday 2026-03-15 (two full forex weeks).
**Goal:** Establish a live-conditions baseline before any PRACTICE_REAL promotion.

### 3.1 Metrics to Track

| Metric | How Collected | Cadence |
|--------|--------------|---------|
| **Total return %** | `paper_account.json` equity vs $8,000 baseline | EOD |
| **Max DD %** | `peak_equity` vs current equity in paper_account | EOD |
| **Mode time %** | Count `SCAN_HEARTBEAT` entries per `risk_mode` bucket | Weekly |
| **Entries taken** | `enter_count` sum in `decision_log.jsonl` | Per week |
| **Entries blocked** | `blocked_count` sum + top block reasons | Per week |
| **Chop Shield triggers** | `AUTO_PAUSE_STREAK3` events in journal | Per event |
| **Chop Shield pauses** | Hours in AUTO_PAUSE state | Per event |
| **Avg R per mode** | `rr_achieved` from `paper_journal.jsonl` grouped by `risk_mode_at_entry` | Weekly |
| **Win rate per mode** | Wins / total per mode bucket | Weekly |
| **Control drift events** | `control_drift != null` in `/api/status` polls | Per occurrence |
| **Broker order calls** | grep for ORDER CREATE/PLACE pattern in logs | Continuous |
| **Session filter blocks** | `session_reason` non-null blocked scans | Weekly |

### 3.2 Logging Requirements (no code change — fields already exist)

All required fields are already written to disk:

**`logs/decision_log.jsonl`** — per scan:
- `enter_count`, `blocked_count`, `top_watching`
- `regime_score` block: `risk_mode`, `wd_aligned`, `atr_ratio`, `last5_sum_r`, `loss_streak`
- `notes` (block reason string)

**`runtime_state/paper_journal.jsonl`** — per trade:
- `risk_mode_at_entry` (risk mode when trade was entered)
- `planned_risk_dollars`, `effective_risk_pct`, `base_risk_pct`
- `equity_before`, `equity_after`
- `rr_achieved` (on exit)
- `streak_at_entry`

**`runtime_state/paper_account.json`** — equity snapshot:
- `equity`, `peak_equity`, `realized_session_pnl`
- `saved_at` timestamp

**`logs/forex_orchestrator.log`** — narrative:
- `RiskMode:` lines for mode transitions
- `AUTO_PAUSE_STREAK3` for Chop Shield fires
- `[DRY RUN]` confirmation on every simulated order

### 3.3 End-of-Week Reporting Template

```
Week N (YYYY-MM-DD → YYYY-MM-DD)
  Equity:        $X,XXX.XX  (±X.X%)
  Max DD:        X.X%
  Trades:        N  (W wins / L losses, WR = X%)
  Mode time:     LOW X%  MED X%  HIGH X%  EXT X%
  Avg R / mode:  LOW X.XXR  MED X.XXR  HIGH X.XXR  EXT X.XXR
  Entries taken: N  |  Blocked: N
  Top block reasons:
    [Nx] MARKET_CLOSED / NO_SUNDAY_TRADES
    [Nx] WAIT: <pair> <pattern> <conf>%
    [Nx] session: non-London/NY
  Chop Shield:   N triggers  (N hours paused)
  Control drift: N events  (must be 0)
  Broker orders: 0  (must remain 0)
```

---

## 4. Promotion Criteria — LIVE_PAPER → PRACTICE_REAL

All of the following must be satisfied simultaneously. No partial passes.

### 4.1 Return & Drawdown

| Metric | Minimum Threshold | Notes |
|--------|------------------|-------|
| Two-week return | `≥ 0%` (no net loss) | Capital preservation floor |
| Max DD over two weeks | `≤ 25%` | Matches Scenario B risk tier design |
| No single-week loss | `> -15%` per week | Week-level guardrail |

### 4.2 Trade Quality

| Metric | Minimum Threshold | Notes |
|--------|------------------|-------|
| Win rate | `≥ 40%` | Alex's observed range was 29–67%; floor at survivable level |
| Avg R across all trades | `≥ +0.2R` | Positive expected value required |
| Min trades taken | `≥ 4` | Enough sample to evaluate edge |
| Avg R in MEDIUM mode | `≥ 0.0R` | The default mode must not be losing |

### 4.3 Control Plane Integrity

| Metric | Requirement |
|--------|-------------|
| `control_drift` events | `== 0` across full two-week window |
| Broker ORDER CREATE/PLACE calls | `== 0` (absolute, zero tolerance) |
| Paper journal entries include required risk fields | `planned_risk_dollars`, `risk_mode_at_entry`, `effective_risk_pct`, `equity_before` all present on every entry |
| No unplanned service restarts during sessions | Zero crashes during London/NY hours |

### 4.4 Regime Behavior

| Metric | Requirement |
|--------|-------------|
| HIGH or EXTREME activated at least once | Confirms regime scoring is working in live conditions |
| Chop Shield fire-and-recovery cycle observed at least once OR `loss_streak` never reached 3 | Either the shield worked correctly, or market was clean enough it didn't need to |
| Mode time LOW `< 60%` | Not permanently stuck in low-confidence mode |

### 4.5 PRACTICE_REAL Starting Parameters

Once all promotion criteria are met, PRACTICE_REAL begins with:

| Parameter | Value |
|-----------|-------|
| Account | OANDA practice (101-001-38598947-001, $100K play-money) |
| Starting notional risk per trade | 1.0% of $100K = $1,000 |
| `dry_run` | `False` (real OANDA demo orders placed) |
| `account_mode` | `LIVE_REAL` (equity sourced from broker) |
| Max concurrent | 1 |
| Mode | AUTO (no pin) |
| First week cap | 2 trades/week regardless of mode |
| Abort condition | Any single week DD > 20% on demo account |

PRACTICE_REAL is not a milestone — it is a separate two-week measurement window
on a live order book (demo). Promotion to funded live trading requires a separate
sign-off after PRACTICE_REAL results are reviewed.

---

_End of offensive_plan.md_
