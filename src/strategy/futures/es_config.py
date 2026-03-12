# ES Futures Strategy Configuration
# Single source of truth — mirrors strategy_config.py pattern

# ── Session times (ET) ────────────────────────────────────────────────────────
RTH_OPEN_ET         = 9
RTH_CLOSE_ET        = 16
PREMARKET_START_ET  = 8

# ── Opening Range Breakout (ORB) ──────────────────────────────────────────────
ORB_CANDLE_MINUTES      = 15          # capture 8:00-8:15 AM ET candle
ORB_BREAKOUT_BUFFER_PTS = 1.0         # must clear high/low by this many points
ORB_STOP_POINTS         = 4.0         # fixed stop distance from entry
ORB_TARGET_MULTIPLIER   = 2.0         # target = stop × this
ORB_MAX_RANGE_PTS       = 15.0        # skip if opening range too wide (choppy open)
ORB_MIN_RANGE_PTS       = 2.0         # skip if opening range too narrow (no energy)

# ── Session High/Low Fade ──────────────────────────────────────────────────────
FADE_LOOKBACK_DAYS      = 5           # rolling window for session high/low
FADE_ENTRY_BUFFER_PTS   = 1.0         # must touch within this many points of extreme
FADE_STOP_POINTS        = 5.0         # fixed stop distance from entry
FADE_TARGET_MULTIPLIER  = 1.5         # target = stop × this
FADE_VALID_START_ET     = 10          # fade window open hour (ET)
FADE_VALID_END_ET       = 15          # fade window close hour (ET)

# ── Contract / Instrument ─────────────────────────────────────────────────────
ES_POINT_VALUE   = 50.0               # $ per point per contract (/ES)
MES_POINT_VALUE  = 5.0                # $ per point per contract (/MES)
SYMBOL           = "/ESM26"
MES_SYMBOL       = "/MESM26"
USE_MES          = True               # default to Micro ES for paper trading

# ── Risk ──────────────────────────────────────────────────────────────────────
MAX_DAILY_LOSS_DOLLARS   = 500        # hard stop for the day
MAX_CONCURRENT_CONTRACTS = 2          # max open contracts at once
MAX_RISK_PCT             = 2.0        # % of account equity risked per trade

# ── Execution ─────────────────────────────────────────────────────────────────
DRY_RUN = True                        # always True until --live flag used

# ── Scan ──────────────────────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 60
