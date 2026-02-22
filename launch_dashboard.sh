#!/bin/bash
# Forge Bot Dashboard Launcher
# Opens the dashboard in the browser. If not running, starts it first.

VENV="$HOME/trading-bot/venv/bin/python"
APP="$HOME/trading-bot/dashboard/app.py"
PORT=5001
URL="http://localhost:$PORT"
LOG="$HOME/trading-bot/logs/dashboard.log"

mkdir -p "$HOME/trading-bot/logs"

is_running() {
    curl -s --max-time 2 "$URL" > /dev/null 2>&1
}

if is_running; then
    echo "Dashboard already running at $URL"
else
    echo "Starting dashboard..."
    nohup "$VENV" "$APP" >> "$LOG" 2>&1 &
    # Wait up to 8 seconds for it to start
    for i in $(seq 1 8); do
        sleep 1
        if is_running; then
            echo "Dashboard started (took ${i}s)"
            break
        fi
    done
fi

# Open browser
if command -v xdg-open &>/dev/null; then
    xdg-open "$URL" &
elif command -v google-chrome &>/dev/null; then
    google-chrome --app="$URL" &
elif command -v firefox &>/dev/null; then
    firefox "$URL" &
fi

echo "Done — $URL"
