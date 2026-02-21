# Trading Bot 🔨

Autonomous cryptocurrency trading bot operating on the Coinbase Advanced Trade API.

**Operator:** Mike  
**Status:** Phase 0 — Machine Assessment  
**Started:** 2026-02-20

## Architecture

- **Exchange Layer**: Coinbase Advanced API (spot + derivatives)
- **Risk Management**: Fee-aware position sizing, hard kill switches
- **Liquidity Engine**: Pre-trade order book validation
- **Strategy Layer**: Technical + LLM-assisted signals
- **Local LLM**: RTX 5090 GPU-accelerated inference via Ollama
- **Monitoring**: Continuous portfolio tracking + Mike alerts

## Setup

```bash
cp .env.template .env
chmod 600 .env
# Fill in your Coinbase API credentials

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Current Status

See `data/research/` for API study notes and latency baselines.
See `logs/` for operational logs.

## Hard Rules

- Maximum single trade risk: 2% of portfolio
- Minimum R:R: 1:3 AFTER FEES
- No market orders (limit only, except emergency exits)
- No leverage (spot only until derivatives are approved)
- No trading if portfolio drops below $300
