"""
Headless Cloud Runner for the BTCC AI Trading Agent System.
Uses DashboardController so remote cloud runs produce the exact same Unified Trade Plans,
Confidence %, Safe Entry / TP / SL prices, and Telegram command responses as the local GUI.
"""

from __future__ import annotations
import argparse
import time
from gui_dashboard import DashboardController


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
    args = parser.parse_args()

    controller = DashboardController(port=8765)
    controller.timeframe = args.timeframe

    print(f"☁️ Starting BTCC Cloud Scanner | Timeframe: {controller.timeframe} | Duration: {args.duration_minutes}m")

    # 1. Process any pending Telegram commands first
    if controller.orchestrator.telegram.token:
        controller.orchestrator.telegram.poll_commands(controller.handle_telegram_command)

    # 2. Run full market scan (auto-sends high-confidence / active alerts & checks open positions)
    controller.run_single_scan()

    # 3. Poll again in case a command arrived during scan, and save cache
    if controller.orchestrator.telegram.token:
        controller.orchestrator.telegram.poll_commands(controller.handle_telegram_command)
    controller._save_alert_cache()

    if args.send_summary and controller.orchestrator.telegram.is_configured:
        controller.orchestrator.telegram.send_message(controller.build_simple_telegram_sheet())

    if args.duration_minutes <= 0:
        print(f"✅ Single cloud scan pass completed ({len(controller.asset_cards)} assets).")
        return

    end_time = time.time() + (args.duration_minutes * 60.0)
    last_scan = time.time()
    while time.time() < end_time:
        time.sleep(5)
        if controller.orchestrator.telegram.token:
            controller.orchestrator.telegram.poll_commands(controller.handle_telegram_command)
            controller._save_alert_cache()
        if time.time() - last_scan >= 60.0:
            last_scan = time.time()
            controller.run_single_scan()

    print("☁️ Cloud session completed cleanly.")


if __name__ == "__main__":
    main()
