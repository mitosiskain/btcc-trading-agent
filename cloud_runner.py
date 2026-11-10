"""
Headless Cloud Runner for the BTCC AI Trading Agent System.
Uses DashboardController so remote cloud runs produce the exact same Unified Trade Plans,
Confidence %, Safe Entry / TP / SL prices, and natural-language Telegram chat responses.
"""

from __future__ import annotations
import argparse
import subprocess
import time
from gui_dashboard import DashboardController


def sync_state_to_git() -> None:
    """Commit and push updated portfolio, alert cache & static phone dashboard to GitHub."""
    try:
        subprocess.run(
            [
                "git",
                "add",
                "state/portfolio_state.json",
                "state/cloud_alert_cache.json",
                "docs/index.html",
                "docs/state.json",
                "btcc/index.html",
                "btcc/state.json",
                "index.html",
                "state.json",
            ],
            check=False,
            capture_output=True,
            timeout=10,
        )
        diff = subprocess.run(["git", "diff", "--staged", "--quiet"], check=False, timeout=5)
        if diff.returncode != 0:
            subprocess.run(
                ["git", "commit", "-m", "chore(state): sync live Telegram positions & alert cache [skip ci]"],
                check=False,
                capture_output=True,
                timeout=10,
            )
            subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=False, capture_output=True, timeout=20)
            subprocess.run(["git", "push", "origin", "main"], check=False, capture_output=True, timeout=20)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCC AI Trading Agent Cloud Runner")
    parser.add_argument("--timeframe", default="4h", help="Timeframe to scan (default: 4h)")
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=0.0,
        help="0 = single cloud pass & exit; >0 = continuous live polling for N minutes",
    )
    parser.add_argument(
        "--send-summary",
        action="store_true",
        help="Force-send the full simplified signal sheet to Telegram on startup",
    )
    parser.add_argument(
        "--git-sync",
        action="store_true",
        help="Auto-commit and push state changes back to GitHub during long cloud sessions",
    )
    args = parser.parse_args()

    controller = DashboardController(port=8765)
    controller.timeframe = args.timeframe

    print(
        f"☁️ Starting BTCC Cloud Scanner | Timeframe: {controller.timeframe} | "
        f"Alert Threshold: {controller.alert_confidence_threshold}% | Duration: {args.duration_minutes}m"
    )

    # 1. Process any pending Telegram messages/commands first
    if controller.orchestrator.telegram.token:
        if controller.orchestrator.telegram.poll_commands(controller.handle_telegram_command):
            controller._save_alert_cache()
            if args.git_sync:
                sync_state_to_git()

    # 2. Run full market scan (auto-sends high-confidence / active alerts & checks open positions)
    controller.run_single_scan()

    # 3. Poll again in case a message arrived during scan, and save cache
    if controller.orchestrator.telegram.token:
        controller.orchestrator.telegram.poll_commands(controller.handle_telegram_command)
    controller._save_alert_cache()
    if args.git_sync:
        sync_state_to_git()

    if args.send_summary and controller.orchestrator.telegram.is_configured:
        controller.orchestrator.telegram.send_message(controller.build_simple_telegram_sheet())

    if args.duration_minutes <= 0:
        print(f"✅ Single cloud scan pass completed ({len(controller.asset_cards)} assets).")
        return

    end_time = time.time() + (args.duration_minutes * 60.0)
    last_scan = time.time()
    while time.time() < end_time:
        time.sleep(4)
        if controller.orchestrator.telegram.token:
            if controller.orchestrator.telegram.poll_commands(controller.handle_telegram_command):
                controller._save_alert_cache()
                if args.git_sync:
                    sync_state_to_git()
        if time.time() - last_scan >= 60.0:
            last_scan = time.time()
            controller.run_single_scan()
            if args.git_sync:
                sync_state_to_git()

    print("☁️ Cloud session completed cleanly.")


if __name__ == "__main__":
    main()
