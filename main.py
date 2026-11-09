"""
Main Orchestrator for the BTCC High-Leverage AI Trading Agent System.
Combines:
1. Deterministic 'Impulse + Flatline -> Reversal' Scanner & Active Position 'Profit-Flatline Stall' Exit Monitor
2. Gemini Sentiment & Catalyst Analyst Agent (with Google Search Grounding)
3. Automated Backtest Researcher Agent
4. Full-Automation Paper/Live Execution Engine + Telegram Alert/Command Bot
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Set, Dict

from config import AppConfig, StrategyParams
from market_data import MarketDataFeed
from detector import ImpulseFlatlineDetector
from sentiment_agent import SentimentAnalystAgent
from execution_engine import ExecutionEngine
from backtest_researcher import AutomatedBacktestResearcher
from telegram_bot import TelegramNotifier


class BTCCTradingOrchestrator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.feed = MarketDataFeed()
        self.detector = ImpulseFlatlineDetector(config.strategy, config.risk)
        self.sentiment_agent = SentimentAnalystAgent(config)
        self.execution = ExecutionEngine(config)
        self.researcher = AutomatedBacktestResearcher(config)
        self.telegram = TelegramNotifier(config)
        self.alerted_bar_keys: Set[str] = set()
        self.state_dir = Path(__file__).resolve().parent / "state"

    def get_detector_for_symbol(self, symbol: str, timeframe: str) -> ImpulseFlatlineDetector:
        """Load per-asset optimized StrategyParams from state/ if available."""
        opt_file = self.state_dir / f"optimized_{symbol.lower()}_{timeframe}.json"
        if opt_file.exists():
            try:
                data = json.loads(opt_file.read_text())
                params_dict = data.get("best_metrics", {}).get("params")
                if params_dict:
                    return ImpulseFlatlineDetector(StrategyParams(**params_dict), self.config.risk)
            except Exception:
                pass
        return self.detector

    def scan_and_manage_once(self) -> None:
        """
        Single pass across watchlist:
        1. Checks open positions for Profit-Flatline Stalls, TP1/TP2, or Stop-Loss.
        2. Scans for new 'Impulse + Flatline -> Reversal' setups and runs Gemini Sentiment validation.
        """
        tf = self.config.strategy.timeframe
        for symbol in self.config.watchlist:
            sym_detector = self.get_detector_for_symbol(symbol, tf)
            try:
                df = self.feed.fetch_ohlcv(symbol=symbol, interval=tf, limit=120)
            except Exception as e:
                print(f"[WARN] Could not fetch {symbol}: {e}")
                continue

            last_idx = len(df) - 1
            curr_price = float(df.iloc[last_idx]["close"])
            curr_ts = str(df.iloc[last_idx]["timestamp"])

            # 1. Monitor Open Position for this symbol (Exit & Profit-Flatline Stall Monitor)
            if symbol in self.execution.open_positions:
                pos = self.execution.open_positions[symbol]
                pos.current_price = curr_price
                tf_sec_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800, "1M": 2592000}
                bar_sec = tf_sec_map.get(tf, 14400)
                try:
                    from datetime import datetime, timezone
                    opened_dt = datetime.fromisoformat(pos.opened_at)
                    pos.bars_held = max(0, int((datetime.now(timezone.utc) - opened_dt).total_seconds() // bar_sec))
                except Exception:
                    pos.bars_held = 0
                exit_sig = sym_detector.check_position_exit_or_stall(
                    df=df,
                    idx=last_idx,
                    symbol=symbol,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    stop_loss=pos.stop_loss,
                    tp1_price=pos.tp1_price,
                    tp2_price=pos.tp2_price,
                    leverage=pos.leverage,
                    margin_usd=pos.remaining_margin_usd,
                    bars_held=pos.bars_held,
                    tp1_already_hit=pos.tp1_hit,
                )
                if exit_sig is not None:
                    closed_rec = self.execution.process_exit_signal(exit_sig)
                    self.telegram.send_exit_or_stall_alert(exit_sig, closed_rec)
                continue

            # 2. Scan for new Impulse + Flatline Reversal Setup (check latest closed bar and current bar)
            for check_idx in (last_idx - 1, last_idx):
                if check_idx < 0:
                    continue
                bar_ts = str(df.iloc[check_idx]["timestamp"])
                dedup_key = f"{symbol}:{tf}:{bar_ts}"
                if dedup_key in self.alerted_bar_keys:
                    continue

                signal = sym_detector.detect_setup_at_index(df, check_idx, symbol=symbol)
                if signal is not None:
                    self.alerted_bar_keys.add(dedup_key)
                    verdict = self.sentiment_agent.evaluate_setup(signal)
                    opened_pos = self.execution.execute_reversal_entry(signal, verdict)
                    self.telegram.send_entry_setup_alert(signal, verdict, opened_pos)
                    break
            else:
                print(
                    f"[SCAN] {symbol:8s} ({tf}) | Price: {curr_price:<10.4f} | "
                    f"RSI: {float(df.iloc[last_idx]['rsi']):5.1f} | ATR: {float(df.iloc[last_idx]['atr']):.4f} | No flatline reversal trigger"
                )

    def handle_telegram_command(self, cmd_text: str) -> str:
        parts = cmd_text.strip().split()
        cmd = parts[0].lower()

        if cmd == "/status":
            return (
                f"📊 *BTCC Agent Bot Status*\n"
                f"• *Account Reserve*: `${self.execution.account_balance_usd:,.2f}`\n"
                f"• *Daily Realized PnL*: `${self.execution.daily_realized_pnl_usd:+,.2f}`\n"
                f"• *Open Positions*: `{len(self.execution.open_positions)}`\n"
                f"• *Closed Trades*: `{len(self.execution.closed_trades)}`\n"
                f"• *Watchlist*: `{', '.join(self.config.watchlist)}`"
            )

        if cmd == "/positions":
            if not self.execution.open_positions:
                return "📭 No open positions."
            lines = ["📈 *Open High-Leverage Positions*:"]
            for sym, p in self.execution.open_positions.items():
                lines.append(
                    f"• *{sym}* `{p.side}` ({p.leverage}x | ${p.remaining_margin_usd:.0f} margin) "
                    f"@ `{p.entry_price}` → `{p.current_price}` | "
                    f"PnL: `${p.unrealized_pnl_usd():+.2f}` (`{p.unrealized_roe_pct():+.1f}% ROE`) | SL: `{p.stop_loss}`"
                )
            return "\n".join(lines)

        if cmd == "/backtest":
            sym = parts[1].upper() if len(parts) > 1 else "ETHUSDT"
            raw_tf = parts[2] if len(parts) > 2 else "5m"
            tf = "1M" if raw_tf == "1M" else raw_tf.lower()
            res = self.researcher.run_research_sweep(symbol=sym, interval=tf, limit=1000)
            return f"🔬 *Backtest Research ({sym} {tf})*\n{res.get('ai_researcher_verdict')}"

        if cmd == "/scan":
            self.scan_and_manage_once()
            return "✅ Manual scan completed."

        if cmd in ("/prices", "/sheet"):
            tf = self.config.strategy.timeframe
            lines = [f"🎯 *BTCC LIMIT / TRIGGER ORDER PRICE SHEET ({tf.upper()})*\n_Tap any price in `code` on your phone to copy it into BTCC:_"]
            for sym in self.config.watchlist:
                try:
                    sym_det = self.get_detector_for_symbol(sym, tf)
                    df = self.feed.fetch_ohlcv(symbol=sym, interval=tf, limit=60)
                    sig = sym_det.detect_setup_at_index(df, len(df) - 1, symbol=sym)
                    radar = sym_det.inspect_mtf_breakout_radar(df, None, symbol=sym, closeup_tf="15m")
                    plan = radar.get("pending_order_plan") or {}
                    if sig:
                        lines.append(
                            f"⚡ *{sym}* (*{sig.trade_side}* @ `{sig.recommended_leverage}x` — ACTIVE):\n"
                            f"   Entry: `{sig.entry_price}` | SL: `{sig.stop_loss}` | TP1: `{sig.tp1_price}` | TP2: `{sig.tp2_price}`"
                        )
                    else:
                        lines.append(
                            f"⏳ *{sym}* (*{plan.get('side')}* @ `{plan.get('recommended_leverage', 5)}x` — TRIGGER):\n"
                            f"   Trigger: `{plan.get('trigger_entry_price')}` | SL: `{plan.get('stop_loss')}` | TP1: `{plan.get('tp1_price')}` | TP2: `{plan.get('tp2_price')}`"
                        )
                except Exception:
                    continue
            lines.append("\n👉 _Reply `/track SUIUSDT` (or any symbol) after placing the order on BTCC so I monitor it for Profit-Flatline Stalls!_")
            return "\n".join(lines)

        if cmd == "/track" and len(parts) >= 2:
            # Supports short form `/track SUIUSDT` OR full form `/track SUIUSDT SHORT 0.8412 5`
            sym = parts[1].upper()
            tf = self.config.strategy.timeframe
            sym_det = self.get_detector_for_symbol(sym, tf)
            df = self.feed.fetch_ohlcv(symbol=sym, interval=tf, limit=60)
            sig = sym_det.detect_setup_at_index(df, len(df) - 1, symbol=sym)
            radar = sym_det.inspect_mtf_breakout_radar(df, None, symbol=sym, closeup_tf="15m")
            plan = radar.get("pending_order_plan") or {}

            if len(parts) >= 4:
                side = parts[2].upper()
                entry_p = float(parts[3])
                lev = int(parts[4]) if len(parts) >= 5 else self.config.risk.default_leverage
            elif sig is not None:
                side = sig.trade_side
                entry_p = sig.entry_price
                lev = sig.recommended_leverage
            else:
                side = str(plan.get("side", "SHORT")).upper()
                entry_p = float(plan.get("trigger_entry_price") or df.iloc[-1]["close"])
                lev = int(plan.get("recommended_leverage") or self.config.risk.default_leverage)

            if sig is not None and abs(entry_p - sig.entry_price) / max(entry_p, 1e-9) < 0.01:
                sl = sig.stop_loss
                tp1 = sig.tp1_price
                tp2 = sig.tp2_price
                liq = sig.estimated_liq_price
            elif plan and abs(entry_p - float(plan.get("trigger_entry_price", 0))) / max(entry_p, 1e-9) < 0.01:
                sl = float(plan["stop_loss"])
                tp1 = float(plan["tp1_price"])
                tp2 = float(plan["tp2_price"])
                liq = float(plan["estimated_liq_price"])
            else:
                atr = float(df.iloc[-1]["atr"])
                if side == "LONG":
                    sl = round(entry_p - 1.5 * atr, 4)
                    tp1 = round(entry_p + 1.8 * atr, 4)
                    tp2 = round(entry_p + 3.0 * atr, 4)
                    liq = round(entry_p * (1.0 - 0.90 / lev), 4)
                else:
                    sl = round(entry_p + 1.5 * atr, 4)
                    tp1 = round(entry_p - 1.8 * atr, 4)
                    tp2 = round(entry_p - 3.0 * atr, 4)
                    liq = round(entry_p * (1.0 + 0.90 / lev), 4)

            from execution_engine import OpenPosition
            from datetime import datetime, timezone

            margin = self.config.risk.margin_per_trade_usd
            pos = OpenPosition(
                position_id=f"manual-{sym.lower()}-{int(time.time())}",
                symbol=sym,
                side=side,
                entry_price=entry_p,
                current_price=entry_p,
                stop_loss=sl,
                tp1_price=tp1,
                tp2_price=tp2,
                leverage=lev,
                initial_margin_usd=margin,
                remaining_margin_usd=margin,
                notional_usd=margin * lev,
                liq_price=liq,
                opened_at=datetime.now(timezone.utc).isoformat(),
                sentiment_decision="MANUAL_BTCC_TRADE",
                sentiment_reasoning="Manually opened on BTCC; monitored for Profit-Flatline Stall",
            )
            self.execution.open_positions[sym] = pos
            self.execution._save_state()
            return (
                f"🎯 *Now Monitoring Manual BTCC Trade: {sym} {side} ({lev}x)*\n"
                f"• *Entry*: `{entry_p}` | *Stop*: `{sl}`\n"
                f"• *TP1*: `{tp1}` | *TP2*: `{tp2}`\n"
                f"• _Exit monitor will alert you immediately when this trade hits TP or flatlines in profit!_"
            )

        if cmd == "/close" and len(parts) >= 2:
            sym = parts[1].upper()
            if sym in self.execution.open_positions:
                del self.execution.open_positions[sym]
                self.execution._save_state()
                return f"✅ Closed/removed `{sym}` from active exit monitoring."
            return f"ℹ️ No active position found for `{sym}`."

        if cmd == "/sentiment":
            lines = ["🧠 *Live Statistical Sentiment (50 TV + X/Social Posts/Asset)*:"]
            for sym in self.config.watchlist:
                ctx = self.sentiment_agent.fetch_derivatives_context(sym)
                sig_tag = f"✅ *STAT SIG ({ctx.confidence_pct:.1f}%)*" if ctx.is_statistically_significant else "⚪"
                ls_tag = f" | L/S `{ctx.long_short_ratio}x`" if ctx.long_short_ratio != 1.0 else ""
                lines.append(
                    f"• *{sym}*: `{ctx.news_bias_label}` (`{ctx.bull_posts}B/{ctx.bear_posts}S`, "
                    f"`z={ctx.sentiment_z_score:+.2f}`, `p={ctx.sentiment_p_value:.3f}`){ls_tag} {sig_tag}"
                )
            return "\n".join(lines)

        return (
            "Commands:\n"
            "• `/status` — Account & bot status\n"
            "• `/positions` — Open trades & live ROE%\n"
            "• `/sentiment` — Live 50-post Statistical Sentiment & Crowd L/S table\n"
            "• `/track ETHUSDT SHORT 2245.5 200` — Monitor a manual BTCC trade for profit-flatline exits\n"
            "• `/close ETHUSDT` — Stop monitoring a trade\n"
            "• `/scan` — Scan watchlist now\n"
            "• `/backtest SOLUSDT 5m` — Run backtest researcher"
        )

    def run_daemon(self, poll_seconds: int = 20) -> None:
        print(
            f"🚀 Starting BTCC AI Trading Agent Daemon | Watchlist: {self.config.watchlist} | "
            f"Timeframe: {self.config.strategy.timeframe} | Reserve: ${self.execution.account_balance_usd:,.2f}"
        )
        while True:
            self.telegram.poll_commands(self.handle_telegram_command)
            self.scan_and_manage_once()
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCC AI Trading Bot (Impulse + Flatline Reversal)")
    parser.add_argument(
        "--mode",
        choices=["backtest", "scan-once", "daemon", "gui"],
        default="gui",
        help="Execution mode (gui launches the live on-screen control center)",
    )
    parser.add_argument("--symbol", default="ETHUSDT", help="Symbol for backtest (e.g., ETHUSDT, XAUUSDT, BTCUSDT)")
    parser.add_argument("--timeframe", default="5m", help="Candle timeframe (1m, 5m, 15m)")
    parser.add_argument("--limit", type=int, default=1000, help="Historical bars to fetch for backtesting")
    parser.add_argument("--poll-seconds", type=int, default=20, help="Polling interval in daemon mode")
    args = parser.parse_args()

    cfg = AppConfig()
    cfg.strategy.timeframe = args.timeframe
    orchestrator = BTCCTradingOrchestrator(cfg)

    if args.mode == "backtest":
        print(f"🔬 Running Automated Backtest Researcher on {args.symbol} ({args.timeframe}, {args.limit} bars)...")
        report = orchestrator.researcher.run_research_sweep(
            symbol=args.symbol,
            interval=args.timeframe,
            limit=args.limit,
        )
        print(json.dumps(report, indent=2))
    elif args.mode == "scan-once":
        orchestrator.scan_and_manage_once()
    elif args.mode == "daemon":
        orchestrator.run_daemon(poll_seconds=args.poll_seconds)
    elif args.mode == "gui":
        from gui_dashboard import main as gui_main
        import sys
        sys.argv = [sys.argv[0]]
        gui_main()


if __name__ == "__main__":
    main()
