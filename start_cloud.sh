#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
MINUTES="${1:-350}"
echo "☁️ Triggering remote GitHub Cloud Scanner for ${MINUTES} minutes..."
gh workflow run btcc_cloud_scanner.yml -f duration_minutes="$MINUTES" -f timeframe="4h" -f send_summary="true"
echo "✅ Cloud session started on GitHub! You can now safely shut down your PC."
echo "   View status anytime at: https://github.com/mitosiskain/btcc-trading-agent/actions"
