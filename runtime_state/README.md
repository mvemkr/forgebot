# runtime_state/

Runtime state files. All `.json` files (except `*.example.json`) are **git-ignored** — never committed.

## Bootstrap (fresh machine)

```bash
bash scripts/bootstrap_runtime_state.sh
```

Copies example templates into place without overwriting existing files.
Use this on every fresh clone before starting the bot.

## LIVE_REAL Policy — Fail-Closed

In `LIVE_REAL` mode the bot **will NOT auto-create** `control.json` if it is missing:

- New entries are **blocked** (fail-closed) until the file is manually restored.
- Existing open positions continue to be managed normally.
- Log will show: `CRITICAL: [LIVE_REAL] control.json MISSING`

**To recover:** copy `control.example.json` → `control.json`, review it,
then restart the bot and resume via the dashboard.

In `LIVE_PAPER` mode a missing `control.json` is created automatically from
the example template (with `pause_new_entries: true`) so paper trading can
start without manual steps.

## Files

| File                        | Tracked? | Description                              |
|-----------------------------|----------|------------------------------------------|
| `control.json`              | ❌ git-ignored | Live control plane (pause/resume, risk mode pin) |
| `paper_account.json`        | ❌ git-ignored | Simulated equity state for LIVE_PAPER mode |
| `control.example.json`      | ✅ tracked    | Bootstrap template — safe defaults, paused |
| `paper_account.example.json`| ✅ tracked    | Bootstrap template — $8K sim equity      |
| `README.md`                 | ✅ tracked    | This file                                |
