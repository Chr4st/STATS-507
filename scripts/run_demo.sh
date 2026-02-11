#!/usr/bin/env bash
# Run the full GeoAg Arb Terminal demo
# ======================================
#
# This script starts the Python API server and then launches the C++ terminal.
# Press 'q' in the terminal to exit. The server will be stopped automatically.
#
# Prerequisites:
#   make setup
#
# Usage:
#   ./scripts/run_demo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== GeoAg Arb Terminal — Demo ==="
echo "DISCLAIMER: For research only; not investment advice."
echo ""

# Check if server is already running
if curl -sf http://localhost:8777/health > /dev/null 2>&1; then
    echo "✓ API server already running on port 8777"
    SERVER_PID=""
else
    echo "→ Starting API server..."
    cd "$REPO_ROOT/python"
    python3 -m geoag.api.server &
    SERVER_PID=$!
    cd "$REPO_ROOT"

    # Wait for server to be ready
    echo -n "  Waiting for server"
    for i in $(seq 1 30); do
        if curl -sf http://localhost:8777/health > /dev/null 2>&1; then
            echo " ✓"
            break
        fi
        echo -n "."
        sleep 1
    done
    echo ""
fi

# Check if terminal binary exists
TERMINAL_BIN="$REPO_ROOT/cpp/build/terminal"
if [ ! -f "$TERMINAL_BIN" ]; then
    echo "✗ Terminal binary not found. Run 'make setup-cpp' first."
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
    exit 1
fi

# Start terminal
echo "→ Launching terminal UI..."
echo "  (Press 'q' to quit)"
echo ""

"$TERMINAL_BIN" --url ws://localhost:8777/ws

# Cleanup
if [ -n "${SERVER_PID:-}" ]; then
    echo "→ Stopping API server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
fi

echo ""
echo "Demo complete."
echo "DISCLAIMER: For research only; not investment advice."
