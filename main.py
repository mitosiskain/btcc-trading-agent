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
from carbon_v2_engine import CarbonLiveBridge
from krypton_engine import KryptonLiveBridge
from sentiment_agent import SentimentAnalystAgent
from execution_engine import ExecutionEngine
from backtest_researcher import AutomatedBacktestResearcher
from telegram_bot import TelegramNotifier


class BTCCTradingOrchestrator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.feed = MarketDataFeed()
        self.detector = ImpulseFlatlineDetector(config.strategy, config.risk)
        self.state_dir = Path(__file__).resolve().parent / "state"
        self.carbon_bridge = CarbonLiveBridge(config.risk, state_dir=self.state_dir)
        self.krypton_bridge = KryptonLiveBridge(config.risk, state_dir=self.state_dir)
        # Active live engine (ENGINE_VERSION env / LeverageRiskConfig.engine_version): KRYPTON_V3 (default) or CARBON_V2
        self.engine_version = str(getattr(config.risk, "engine_version", "KRYPTON_V3")).upper()
        self.is_krypton = self.engine_version.startswith("KRYPTON")
        self.engine_bridge = self.krypton_bridge if self.is_krypton else self.carbon_bridge
        self.engine_label = "Krypton V3" if self.is_krypton else "Carbon V2"
        self.sentiment_agent = SentimentAnalystAgent(config)
        self.execution = ExecutionEngine(config)
        self.researcher = AutomatedBacktestResearcher(config)
        self.telegram = TelegramNotifier(config)
        self.alerted_bar_keys: Set[str] = set()

    def scan_bars(self, timeframe: str) -> int:
        """History length the active engine needs (Krypton's 200/800-bar EMAs need far more than Carbon's 240)."""
        return KryptonLiveBridge.required_bars(timeframe) if self.is_krypton else 240

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
        Single pass across watchlist using the active engine (Krypton V3 by default; Carbon V2 via ENGINE_VERSION):
        1. Checks open positions for Post-TP1 Chandelier Trails, Mature Profit Stalls, TP1/TP2, or Stop-Loss.
        2. Scans 12-column perpetual orderflow + BTC beta-residuals + PurgedMetaEnsemble for Maker-Limit entries.
        """
        tf = self.config.strategy.timeframe
        for symbol in self.config.watchlist:
            sym_detector = self.get_detector_for_symbol(symbol, tf)
            try:
                df, btc_df = self.feed.fetch_enriched_ohlcv(symbol=symbol, interval=tf, limit=self.scan_bars(tf))
            except Exception as e:
                print(f"[WARN] Could not fetch {symbol}: {e}")
                continue

            last_idx = len(df) - 1
            curr_price = float(df.iloc[last_idx]["close"])

            # 1. Monitor Open Position for this symbol (Chandelier Trail, Mature Stall & TP/SL Monitor)
            if symbol in self.execution.open_positions:
                pos = self.execution.open_positions[symbol]
                pos.current_price = curr_price
                if pos.side == "LONG":
                    pos.peak_favorable_price = max(float(getattr(pos, "peak_favorable_price", 0.0) or curr_price), curr_price)
                else:
                    prev_pk = float(getattr(pos, "peak_favorable_price", 0.0) or curr_price)
                    pos.peak_favorable_price = min(prev_pk, curr_price) if prev_pk > 0 else curr_price

                tf_sec_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800, "1M": 2592000}
                bar_sec = tf_sec_map.get(tf, 14400)
                try:
                    from datetime import datetime, timezone
                    opened_dt = datetime.fromisoformat(pos.opened_at)
                    pos.bars_held = max(0, int((datetime.now(timezone.utc) - opened_dt).total_seconds() // bar_sec))
                except Exception:
                    pos.bars_held = 0
                if self.is_krypton:
                    exit_sig, trail_upd = self.krypton_bridge.check_position_exit(df, pos, tf)
                    if trail_upd is not None:
                        self.execution._save_state()
                        self.telegram.send_message(
                            f"🔒 *{symbol} {pos.side}*: Krypton trail raised — move your SL "
                            f"`{trail_upd['old_sl']}` → `{trail_upd['new_sl']}`"
                        )
                else:
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
                        trail_atr_mult=float(getattr(pos, "trail_atr_mult", 1.75)),
                        peak_favorable_price=float(getattr(pos, "peak_favorable_price", 0.0)),
                    )
                if exit_sig is not None:
                    closed_rec = self.execution.process_exit_signal(exit_sig)
                    self.telegram.send_exit_or_stall_alert(exit_sig, closed_rec)
                continue

            # 2. Scan with the active engine (Krypton V3 trend breakout by default, Carbon V2 optional)
            scan_res = self.engine_bridge.evaluate_symbol_live(df, btc_df, symbol=symbol, timeframe=tf)
            tel = scan_res.telemetry
            if scan_res.active_signal is not None and scan_res.signal_bars_ago in (0, 1):
                signal = scan_res.active_signal
                cand = scan_res.active_candidate
                dedup_key = f"{symbol}:{tf}:{signal.timestamp}"
                if dedup_key not in self.alerted_bar_keys:
                    self.alerted_bar_keys.add(dedup_key)
                    verdict = self.sentiment_agent.evaluate_setup(signal)
                    verdict.reasoning = (
                        f"{self.engine_label} {tel['archetype']} (E[{tel['meta_pred_r']:+.2f}R], "
                        f"Taker Δ {tel['delta_norm_pct']:+.1f}%, Whale {tel['whale_print_ratio']:.2f}x) • {verdict.reasoning}"
                    )
                    opened_pos = self.execution.execute_reversal_entry(
                        signal,
                        verdict,
                        archetype=cand.archetype if cand else "ABSORPTION_REVERSAL",
                        partial_frac=cand.partial_frac if cand else 0.48,
                        trail_atr_mult=cand.trail_atr_mult if cand else 1.75,
                        meta_pred_r=cand.meta_pred_r if cand else 0.0,
                    )
                    self.telegram.send_entry_setup_alert(signal, verdict, opened_pos)
                    continue

            print(
                f"[{self.engine_label.upper()}] {symbol:8s} ({tf}) | Price: {curr_price:<10.4f} | "
                f"Taker Δ: {tel['delta_norm_pct']:+5.1f}% | Whale: {tel['whale_print_ratio']:4.2f}x | "
                f"VWAP-Z: {tel['z_vwap']:+5.2f} | Meta E[R]: {tel['meta_pred_r']:+5.2f}R | Trap {scan_res.pending_order_plan['side']} @ {scan_res.pending_order_plan['limit_entry_price']}"
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
            lines = [f"🎯 *{self.engine_label.upper()} ORDER SHEET ({tf.upper()})*\n_Tap any price to copy:_"]
            for sym in self.config.watchlist:
                try:
                    df, btc_df = self.feed.fetch_enriched_ohlcv(symbol=sym, interval=tf, limit=self.scan_bars(tf))
                    scan_res = self.engine_bridge.evaluate_symbol_live(df, btc_df, symbol=sym, timeframe=tf)
                    sig = scan_res.active_signal
                    plan = scan_res.pending_order_plan
                    tel = scan_res.telemetry
                    if sig:
                        lines.append(
                            f"⚡ *{sym}* (*{sig.trade_side}* @ `{sig.recommended_leverage}x` — MAKER ENTER NOW • `E[{tel['meta_pred_r']:+.2f}R]`):\n"
                            f"   Limit Entry: `{sig.entry_price}` | SL: `{sig.stop_loss}` | TP1: `{sig.tp1_price}` | TP2: `{sig.tp2_price}`"
                        )
                    else:
                        lines.append(
                            f"⏳ *{sym}* (*{plan.get('side')}* @ `{plan.get('recommended_leverage', 5)}x` — MAKER TRAP):\n"
                            f"   Limit Trap: `{plan.get('limit_entry_price')}` | SL: `{plan.get('stop_loss')}` | TP1: `{plan.get('tp1_price')}` | TP2: `{plan.get('tp2_price')}`"
                        )
                except Exception:
                    continue
            lines.append(f"\n👉 _Reply `/track SUIUSDT` (or any symbol) after placing the order on BTCC so {self.engine_label} monitors your stop & exit rules!_")
            return "\n".join(lines)

        if cmd == "/track" and len(parts) >= 2:
            # Supports short form `/track SUIUSDT` OR full form `/track SUIUSDT SHORT 0.8412 5`
            sym = parts[1].upper()
            tf = self.config.strategy.timeframe
            df, btc_df = self.feed.fetch_enriched_ohlcv(symbol=sym, interval=tf, limit=self.scan_bars(tf))
            scan_res = self.engine_bridge.evaluate_symbol_live(df, btc_df, symbol=sym, timeframe=tf)
            sig = scan_res.active_signal
            plan = scan_res.pending_order_plan

            if len(parts) >= 4:
                side = parts[2].upper()
                entry_p = float(parts[3])
                lev = int(parts[4]) if len(parts) >= 5 else self.config.risk.default_leverage
            elif sig is not None:
                side = sig.trade_side
                entry_p = sig.entry_price
                lev = sig.recommended_leverage
            else:
                side = str(plan.get("side", "LONG")).upper()
                entry_p = float(plan.get("limit_entry_price") or df.iloc[-1]["close"])
                lev = int(plan.get("recommended_leverage") or self.config.risk.default_leverage)

            if sig is not None and abs(entry_p - sig.entry_price) / max(entry_p, 1e-9) < 0.015:
                sl = sig.stop_loss
                tp1 = sig.tp1_price
                tp2 = sig.tp2_price
                liq = sig.estimated_liq_price
            elif plan and abs(entry_p - float(plan.get("limit_entry_price", 0))) / max(entry_p, 1e-9) < 0.015:
                sl = float(plan["stop_loss"])
                tp1 = float(plan["tp1_price"])
                tp2 = float(plan["tp2_price"])
                liq = float(plan["estimated_liq_price"])
            else:
                prec = self.engine_bridge._price_precision(entry_p)
                atr = float(df.iloc[-1]["atr"]) or (entry_p * 0.015)
                is_rwa = any(k in sym for k in ("XAU", "XAG"))
                min_stop_pct = 0.0080 if is_rwa else 0.0155
                sl_dist = max(1.35 * atr, min_stop_pct * entry_p)
                if side == "LONG":
                    sl = round(entry_p - sl_dist, prec)
                    tp1 = round(entry_p + 1.30 * sl_dist, prec)
                    tp2 = round(entry_p + 2.65 * sl_dist, prec)
                    liq = round(entry_p * (1.0 - 0.90 / max(lev, 1)), prec)
                else:
                    sl = round(entry_p + sl_dist, prec)
                    tp1 = round(entry_p - 1.30 * sl_dist, prec)
                    tp2 = round(entry_p - 2.65 * sl_dist, prec)
                    liq = round(entry_p * (1.0 + 0.90 / max(lev, 1)), prec)

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
                archetype="KRYPTON_TREND_BREAKOUT" if self.is_krypton else "CARBON_ABSORPTION_REVERSAL",
            )
            self.execution.open_positions[sym] = pos
            self.execution.touch_positions_version()
            return (
                f"🟠 *Now Monitoring Manual BTCC Trade: {sym} {side} ({lev}x)*\n"
                f"• *Entry*: `{entry_p}` | *Stop*: `{sl}`\n"
                f"• *TP1*: `{tp1}` | *TP2*: `{tp2}`\n"
                f"• _Exit monitor will alert you immediately when this trade hits TP or flatlines in profit!_"
            )

        if cmd == "/close" and len(parts) >= 2:
            self.execution._load_state()
            sym = parts[1].upper()
            if sym == "ALL":
                recs = self.execution.close_all_manual(reason="MANUAL_CLOSE_ALL")
                if recs:
                    return f"✅ Closed all tracked positions (`{', '.join(r.symbol for r in recs)}`)."
                return "ℹ️ No active positions currently tracked."
            rec = self.execution.close_position_manual(sym, reason="MANUAL_CLOSE")
            if rec is not None:
                return f"✅ Closed `{sym}` (`${rec.net_pnl_usd:+.2f}` / `{rec.roe_pct:+.1f}% ROE`) — removed from active exit monitoring."
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
    parser.add_argument("--timeframe", default="4h", help="Candle timeframe (default: 4h Krypton V3 trend)")
    parser.add_argument("--limit", type=int, default=1200, help="Historical bars to fetch for backtesting")
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
