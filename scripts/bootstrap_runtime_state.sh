#!/usr/bin/env bash
# bootstrap_runtime_state.sh
# --------------------------
# Set up runtime_state/ on a fresh clone.
# Safe to re-run: never overwrites existing files.
#
# Usage:
#   bash scripts/bootstrap_runtime_state.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME="$REPO_ROOT/runtime_state"

echo "==> Bootstrapping runtime_state at: $RUNTIME"
mkdir -p "$RUNTIME"

# ── control.json ─────────────────────────────────────────────────────────────
if [ -f "$RUNTIME/control.json" ]; then
    echo "  [skip]    control.json already exists"
else
    if [ ! -f "$RUNTIME/control.example.json" ]; then
        echo "  [ERROR]   control.example.json not found — cannot bootstrap" >&2
        exit 1
    fi
    cp "$RUNTIME/control.example.json" "$RUNTIME/control.json"
    echo "  [created] control.json (copied from control.example.json)"
fi

# ── paper_account.json ────────────────────────────────────────────────────────
if [ -f "$RUNTIME/paper_account.json" ]; then
    echo "  [skip]    paper_account.json already exists"
else
    if [ ! -f "$RUNTIME/paper_account.example.json" ]; then
        echo "  [ERROR]   paper_account.example.json not found — cannot bootstrap" >&2
        exit 1
    fi
    cp "$RUNTIME/paper_account.example.json" "$RUNTIME/paper_account.json"
    echo "  [created] paper_account.json (copied from paper_account.example.json)"
fi

echo ""
echo "==> Done. Files are local-only (git-ignored)."
echo ""
echo "NOTE (LIVE_REAL): If deploying in LIVE_REAL mode, do NOT rely solely on"
echo "     this script. The bot will REFUSE new entries if control.json was"
echo "     not manually reviewed. Confirm pause_new_entries and resume via"
echo "     the dashboard only after verifying system state."
