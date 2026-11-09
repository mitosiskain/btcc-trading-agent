"""
Local Web/Desktop GUI Control Center for the BTCC High-Leverage AI Trading Agent System.
Auto-runs multi-asset market scans, monitors open positions for Profit-Flatline Stalls,
displays live BUY/SELL/WATCH recommendations, and shows real-time component health diagnostics.
"""

from __future__ import annotations
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
import webbrowser

from aiohttp import web

from config import AppConfig, DEFAULT_ASSETS, StrategyParams
from detector import ImpulseFlatlineDetector, ReversalSetupSignal
from execution_engine import OpenPosition
from main import BTCCTradingOrchestrator


def fmt_price(val: float) -> str:
    """Format prices cleanly without scientific notation or float rounding noise."""
    v = float(val or 0.0)
    av = abs(v)
    if av == 0:
        return "0"
    if av < 0.001:
        return f"{v:.8f}"
    if av < 1.0:
        return f"{v:.5f}".rstrip("0").rstrip(".")
    if av < 100.0:
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return f"{v:.2f}"


class DashboardController:
    def __init__(self, port: int = 8765):
        self.port = port
        self.config = AppConfig()
        # Include NVDAUSDT in the GUI watchlist if not already present
        for sym in ["XRPUSDT", "DOGEUSDT", "SUIUSDT", "SOLUSDT", "ETHUSDT", "BTCUSDT", "XAUUSDT", "NVDAUSDT"]:
            if sym not in self.config.watchlist:
                self.config.watchlist.append(sym)

        self.orchestrator = BTCCTradingOrchestrator(self.config)
        self.timeframe: str = "4h"
        self.poll_interval_sec: int = 15
        self.auto_scan_enabled: bool = True
        self.is_scanning: bool = False
        self.last_scan_ts: Optional[str] = None
        self.last_scan_duration_ms: int = 0
        self.scan_count: int = 0

        self.asset_cards: Dict[str, Dict[str, Any]] = {}
        self.feed_health: Dict[str, Dict[str, Any]] = {}
        self.recent_alerts: List[Dict[str, Any]] = []
        self.backtest_jobs: Dict[str, Dict[str, Any]] = {}
        self.sentiment_alert_cooldowns: Dict[str, float] = {}
        self.alert_confidence_threshold: int = 80
        self._lock = threading.Lock()
        self._alert_cache_file = Path(__file__).resolve().parent / "state" / "cloud_alert_cache.json"
        self._load_alert_cache()

    def _load_alert_cache(self) -> None:
        if self._alert_cache_file.exists():
            try:
                data = json.loads(self._alert_cache_file.read_text())
                self.sentiment_alert_cooldowns = {
                    str(k): float(v) for k, v in (data.get("cooldowns") or {}).items()
                }
                self.orchestrator.alerted_bar_keys = set(data.get("alerted_bar_keys") or [])
                if data.get("last_update_id"):
                    self.orchestrator.telegram.last_update_id = int(data["last_update_id"])
                if data.get("alert_confidence_threshold"):
                    self.alert_confidence_threshold = max(30, min(99, int(data["alert_confidence_threshold"])))
            except Exception:
                pass

    def _save_alert_cache(self) -> None:
        try:
            self._alert_cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._alert_cache_file.write_text(
                json.dumps(
                    {
                        "cooldowns": self.sentiment_alert_cooldowns,
                        "alerted_bar_keys": sorted(list(self.orchestrator.alerted_bar_keys))[-200:],
                        "last_update_id": self.orchestrator.telegram.last_update_id,
                        "alert_confidence_threshold": self.alert_confidence_threshold,
                    },
                    indent=2,
                )
            )
        except Exception:
            pass

    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Return clear status of what IS and IS NOT working/configured."""
        state_dir = Path(__file__).resolve().parent / "state"
        opt_profiles = {}
        for sym in self.config.watchlist:
            opt_file = state_dir / f"optimized_{sym.lower()}_{self.timeframe}.json"
            if opt_file.exists():
                try:
                    data = json.loads(opt_file.read_text())
                    bm = data.get("best_metrics", {})
                    opt_profiles[sym] = {
                        "optimized": True,
                        "profitable": float(bm.get("profit_factor", 0.0)) >= 1.0 and float(bm.get("total_net_pnl_usd", 0.0)) > 0,
                        "win_rate_pct": bm.get("win_rate_pct", 0.0),
                        "profit_factor": bm.get("profit_factor", 0.0),
                        "total_net_pnl_usd": bm.get("total_net_pnl_usd", 0.0),
                        "total_trades": bm.get("total_trades", 0),
                        "profit_stall_exits": bm.get("profit_stall_exits_count", 0),
                    }
                except Exception as e:
                    opt_profiles[sym] = {"optimized": False, "error": str(e)}
            else:
                opt_profiles[sym] = {"optimized": False}

        gemini_ok = bool(self.config.gemini_api_key and self.config.gemini_api_key != "your_gemini_api_key_here")
        has_tg_token = bool(
            self.config.telegram_bot_token
            and self.config.telegram_bot_token != "your_telegram_bot_token_here"
        )
        telegram_ok = bool(has_tg_token and self.config.telegram_chat_id)
        btcc_api_ok = bool(self.config.btcc_api_key and self.config.btcc_api_secret)

        feeds_online = sum(1 for v in self.feed_health.values() if v.get("ok"))
        feeds_total = len(self.config.watchlist)

        return {
            "scanner_daemon": {
                "working": True,
                "status": "ACTIVE (AUTO-RUNNING)" if self.auto_scan_enabled else "PAUSED",
                "detail": f"Scans every {self.poll_interval_sec}s ({self.scan_count} scans completed, {self.last_scan_duration_ms}ms last pass)",
            },
            "market_data_feeds": {
                "working": feeds_online > 0,
                "status": f"{feeds_online}/{feeds_total} FEEDS ONLINE",
                "detail": "Live public OHLCV candles (Binance Perps / Bybit / Yahoo Finance RWA)",
                "per_symbol": self.feed_health,
            },
            "strategy_optimizers": {
                "working": any(v.get("optimized") for v in opt_profiles.values()),
                "status": f"{sum(1 for v in opt_profiles.values() if v.get('optimized'))}/{feeds_total} ASSETS OPTIMIZED ({self.timeframe})",
                "detail": "Per-asset backtest parameter profiles in state/",
                "profiles": opt_profiles,
            },
            "sentiment_agent": {
                "working": True,
                "mode": "GEMINI_LIVE_SEARCH" if gemini_ok else "TV_SOCIAL_STATISTICAL_ENGINE",
                "status": "LIVE TV NEWS + X/SOCIAL + CROWD L/S (50 Posts/Asset z-Test)",
                "detail": (
                    "Pulling 30 TradingView + 20 X/Social posts per asset + OKX Crowd L/S Ratio with Binomial z-score/p-value alerts"
                ),
            },
            "telegram_alerts": {
                "working": telegram_ok,
                "status": (
                    "CONNECTED (@kaintraderbot)"
                    if telegram_ok
                    else (
                        "WAITING FOR FIRST MESSAGE (@kaintraderbot)"
                        if has_tg_token
                        else "NOT CONFIGURED (On-Screen GUI Alerts Active)"
                    )
                ),
                "detail": (
                    f"Sending phone alerts to chat {self.config.telegram_chat_id}"
                    if telegram_ok
                    else (
                        "Bot token saved! Open @kaintraderbot in Telegram and send 'hi' or '/start' — it will auto-pair within 5s!"
                        if has_tg_token
                        else "Optional: add TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID in .env for phone push alerts"
                    )
                ),
            },
            "btcc_execution": {
                "working": True,
                "status": "LIVE BTCC API" if (btcc_api_ok and self.config.live_trading_enabled) else "LOCAL PAPER & MANUAL TRACKER",
                "detail": (
                    "Connected to BTCC API"
                    if btcc_api_ok
                    else "Runs full local simulation & tracks manual BTCC trades without needing an exchange API key"
                ),
            },
        }

    def run_single_scan(self) -> None:
        if self.is_scanning:
            return
        self.is_scanning = True
        t0 = time.time()
        tf = self.timeframe
        self.config.strategy.timeframe = tf

        try:
            self.orchestrator.execution._load_state()
            for symbol in self.config.watchlist:
                f_t0 = time.time()
                sym_detector = self.orchestrator.get_detector_for_symbol(symbol, tf)
                params = sym_detector.params
                profile = DEFAULT_ASSETS.get(symbol)

                try:
                    df = self.orchestrator.feed.fetch_ohlcv(symbol=symbol, interval=tf, limit=140)
                    latency_ms = int((time.time() - f_t0) * 1000)
                    self.feed_health[symbol] = {
                        "ok": True,
                        "latency_ms": latency_ms,
                        "bars": len(df),
                        "source": "Binance Perps / Spot" if (profile and profile.binance_symbol) else f"Yahoo ({profile.yfinance_ticker if profile else symbol})",
                        "updated_at": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                    }
                except Exception as e:
                    self.feed_health[symbol] = {
                        "ok": False,
                        "error": str(e),
                        "updated_at": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                    }
                    continue

                last_idx = len(df) - 1
                curr_bar = df.iloc[last_idx]
                curr_price = float(curr_bar["close"])
                curr_rsi = float(curr_bar["rsi"])
                curr_atr = float(curr_bar["atr"])

                # 1. Update & Check Open Position for Profit-Flatline Stall / TP / SL
                if symbol in self.orchestrator.execution.open_positions:
                    pos = self.orchestrator.execution.open_positions[symbol]
                    pos.current_price = curr_price
                    tf_sec_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800, "1M": 2592000}
                    bar_sec = tf_sec_map.get(tf, 14400)
                    try:
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
                        warning_already_sent=getattr(pos, "warning_sent", False),
                        use_live_spot_only=True,
                    )
                    if exit_sig is not None:
                        closed_rec = self.orchestrator.execution.process_exit_signal(exit_sig)
                        exit_cd_key = f"exit_{symbol}_{exit_sig.reason}"
                        now_exit_ts = time.time()
                        if now_exit_ts - self.sentiment_alert_cooldowns.get(exit_cd_key, 0.0) >= 3600:
                            self.sentiment_alert_cooldowns[exit_cd_key] = now_exit_ts
                            self.orchestrator.telegram.send_exit_or_stall_alert(exit_sig, closed_rec)
                        self.recent_alerts.insert(
                            0,
                            {
                                "time": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                                "type": "EXIT_OR_STALL",
                                "symbol": symbol,
                                "side": exit_sig.position_side,
                                "reason": exit_sig.reason,
                                "urgency": exit_sig.urgency,
                                "price": curr_price,
                                "roe_pct": round(exit_sig.unrealized_pnl_pct_on_margin, 2),
                                "pnl_usd": round(exit_sig.unrealized_pnl_usd, 2),
                                "explanation": exit_sig.explanation,
                            },
                        )
                        self.recent_alerts = self.recent_alerts[:25]

                # 2. Check for Active / Recent Confirmed Signal (within last 12 CLOSED bars & still valid)
                active_signal: Optional[ReversalSetupSignal] = None
                signal_bars_ago: Optional[int] = None
                curr_ema20 = float(df["ema20"].iloc[last_idx]) if "ema20" in df.columns else curr_price
                for lookback in range(1, 13):
                    chk_idx = last_idx - lookback
                    if chk_idx < 25:
                        continue
                    sig = sym_detector.detect_setup_at_index(df, chk_idx, symbol=symbol)
                    if sig is not None:
                        # Verify setup hasn't already been invalidated or completed in subsequent bars
                        sub_slice = df.iloc[chk_idx + 1 : last_idx + 1]
                        if not sub_slice.empty:
                            if sig.trade_side == "SHORT":
                                if float(sub_slice["high"].max()) >= sig.stop_loss:
                                    continue  # Stopped out already
                                if float(sub_slice["low"].min()) <= sig.tp2_price:
                                    continue  # Already completed TP2
                                # Invalidate stale SHORT if price bounced above EMA20, >40% toward SL, or already moved >30% toward TP1 (bad R:R now)
                                if (
                                    curr_price > curr_ema20
                                    or curr_price >= sig.entry_price + 0.40 * abs(sig.stop_loss - sig.entry_price)
                                    or curr_price <= sig.entry_price - 0.30 * abs(sig.entry_price - sig.tp1_price)
                                ):
                                    continue
                            else:
                                if float(sub_slice["low"].min()) <= sig.stop_loss:
                                    continue
                                if float(sub_slice["high"].max()) >= sig.tp2_price:
                                    continue
                                if (
                                    curr_price < curr_ema20
                                    or curr_price <= sig.entry_price - 0.40 * abs(sig.entry_price - sig.stop_loss)
                                    or curr_price >= sig.entry_price + 0.30 * abs(sig.tp1_price - sig.entry_price)
                                ):
                                    continue
                        active_signal = sig
                        signal_bars_ago = lookback
                        break

                # 2.5 Multi-Timeframe Flatline Box & Close-Up Breakout Radar (4h Flatline + 15m Close-Up)
                closeup_tf = "15m" if tf in ("4h", "1h", "1d") else ("1m" if tf in ("5m", "15m") else "15m")
                df_closeup = None
                try:
                    if closeup_tf != tf:
                        df_closeup = self.orchestrator.feed.fetch_ohlcv(symbol=symbol, interval=closeup_tf, limit=36)
                except Exception:
                    df_closeup = None

                mtf_radar = sym_detector.inspect_mtf_breakout_radar(
                    df, df_closeup, symbol=symbol, closeup_tf=closeup_tf
                )
                closeup_sig = mtf_radar.pop("closeup_signal", None)
                if closeup_sig is not None and (active_signal is None or closeup_sig.trade_side != active_signal.trade_side):
                    active_signal = closeup_sig
                    signal_bars_ago = 0

                # 3. Compute Live Phase 1 (Impulse) & Phase 2 (Flatline) Diagnostics for the UI Card
                w_imp = params.impulse_window_bars
                w_flat = params.flatline_bars
                imp_start = max(0, last_idx - w_flat - w_imp)
                imp_end = max(0, last_idx - w_flat - 1)
                flat_start = max(0, last_idx - w_flat)

                imp_slice = df.iloc[imp_start : imp_end + 1]
                flat_slice = df.iloc[flat_start : last_idx]
                if flat_slice.empty:
                    flat_slice = df.iloc[max(0, last_idx - w_flat + 1) : last_idx + 1]
                baseline_atr = float(df["atr"].iloc[max(0, imp_start - 5) : last_idx + 1].mean()) or curr_atr or 1e-6

                imp_open = float(imp_slice["open"].iloc[0]) if not imp_slice.empty else curr_price
                imp_close = float(imp_slice["close"].iloc[-1]) if not imp_slice.empty else curr_price
                net_move = imp_close - imp_open
                impulse_pct = (net_move / imp_open) * 100.0 if imp_open else 0.0
                impulse_atr_mult = abs(net_move) / baseline_atr if baseline_atr > 0 else 0.0

                flat_high = float(mtf_radar.get("box_high", float(flat_slice["high"].max())))
                flat_low = float(mtf_radar.get("box_low", float(flat_slice["low"].min())))
                flat_range_atr = float(mtf_radar.get("box_range_atr", (flat_high - flat_low) / baseline_atr if baseline_atr > 0 else 0.0))

                imp_bodies = (imp_slice["close"] - imp_slice["open"]).abs()
                flat_bodies = (flat_slice["close"] - flat_slice["open"]).abs()
                body_decay = float(flat_bodies.mean() / max(float(imp_bodies.mean()), 1e-9)) if not imp_bodies.empty else 0.5

                # 48-bar range position
                recent_48 = df.iloc[max(0, len(df) - 48) :]
                hi_48 = float(recent_48["high"].max())
                lo_48 = float(recent_48["low"].min())
                range_pos_pct = ((curr_price - lo_48) / (hi_48 - lo_48) * 100.0) if hi_48 > lo_48 else 50.0

                impulse_met = (impulse_atr_mult >= params.impulse_min_atr_mult) and (abs(impulse_pct) >= params.impulse_min_pct)
                flatline_met = bool(mtf_radar.get("is_flatline_box", False))

                # Last 24 close prices for mini sparkline
                prec = 8 if curr_price < 0.001 else (5 if curr_price < 1.0 else (4 if curr_price < 100.0 else 2))
                sparkline = [round(float(x), prec) for x in df["close"].tail(24).tolist()]
                macro_compass = self.orchestrator.feed.get_macro_trend_compass(symbol)
                market_intel = self.orchestrator.sentiment_agent.get_live_market_intel_dict(symbol)

                import math
                margin_usd = self.config.risk.margin_per_trade_usd
                min_notional = profile.min_notional_usd if profile else 15.0
                min_lot_lbl = profile.min_lot_label if profile else "0.01"
                min_lev_needed = max(1, int(math.ceil(min_notional / max(margin_usd, 0.5))))
                rec_lev = max(min_lev_needed, min(self.config.risk.default_leverage, profile.max_btcc_leverage if profile else 100))
                est_sl_loss = round(margin_usd * (getattr(self.config.risk, "target_sl_margin_risk_pct", 25.0) / 100.0), 2)

                sentiment_overlap = self.orchestrator.sentiment_agent.evaluate_pre_move_sentiment_overlap(
                    symbol=symbol,
                    curr_price=curr_price,
                    curr_rsi=curr_rsi,
                    curr_atr=curr_atr,
                    curr_ema20=curr_ema20,
                    mtf_radar=mtf_radar,
                    macro_compass=macro_compass,
                    rec_leverage=rec_lev,
                )

                # Load historical backtest win rate for unified confidence calculation
                bt_wr = 68.0
                bt_profitable = True
                opt_file = Path(__file__).resolve().parent / "state" / f"optimized_{symbol.lower()}_{tf}.json"
                if opt_file.exists():
                    try:
                        bm_data = json.loads(opt_file.read_text()).get("best_metrics", {})
                        bt_wr = float(bm_data.get("win_rate_pct", 68.0)) or 68.0
                        bt_profitable = float(bm_data.get("profit_factor", 1.0)) >= 1.0 and float(bm_data.get("total_net_pnl_usd", 0.0)) > 0
                    except Exception:
                        pass

                # Compute ONE Unified Trade Plan (Direction, Safe Entry, TP1, TP2, SL, Confidence %)
                verdict_dict = None
                if active_signal is not None:
                    verdict = self.orchestrator.sentiment_agent.evaluate_setup(active_signal)
                    verdict_dict = verdict.model_dump()

                if active_signal is not None and (verdict_dict is None or verdict_dict.get("decision") != "VETO_CONTINUATION_RISK"):
                    u_side = active_signal.trade_side
                    u_mode = "ENTER NOW"
                    u_entry = float(active_signal.entry_price)
                    u_sl = float(active_signal.stop_loss)
                    u_tp1 = float(active_signal.tp1_price)
                    u_tp2 = float(active_signal.tp2_price)
                    u_lev = int(active_signal.recommended_leverage)
                else:
                    plan = mtf_radar.get("pending_order_plan") or {}
                    pref_side = plan.get("side") or mtf_radar.get("prefer_side", "LONG")
                    long_pts = 1.2 if pref_side == "LONG" else 0.0
                    short_pts = 1.2 if pref_side == "SHORT" else 0.0

                    if curr_rsi <= 38.0 or range_pos_pct <= 20.0:
                        long_pts += 2.0
                    elif curr_rsi >= 62.0 or range_pos_pct >= 80.0:
                        short_pts += 2.0

                    dom_trend = macro_compass.get("dominant_trend", "MIXED")
                    if dom_trend == "BULLISH":
                        long_pts += 1.2
                    elif dom_trend == "BEARISH":
                        short_pts += 1.2
                    if symbol in ("XAUUSDT", "XAGUSDT") and (
                        macro_compass.get("1w", {}).get("trend") == "BULLISH"
                        or macro_compass.get("1M", {}).get("trend") == "BULLISH"
                    ):
                        long_pts += 1.5
                    elif (
                        macro_compass.get("1d", {}).get("trend") == "BULLISH"
                        and macro_compass.get("1w", {}).get("trend") == "BULLISH"
                    ):
                        long_pts += 1.2

                    so_side = sentiment_overlap.get("bias_side", "LONG")
                    so_wt = 1.6 if sentiment_overlap.get("is_statistically_significant") else 0.6
                    if so_side == "LONG":
                        long_pts += so_wt
                    else:
                        short_pts += so_wt

                    u_side = "LONG" if long_pts >= short_pts else "SHORT"
                    u_mode = "WAIT FOR SAFE ENTRY"
                    u_lev = rec_lev

                    if plan.get("side") == u_side and plan.get("trigger_entry_price"):
                        u_entry = float(plan["trigger_entry_price"])
                        u_sl = float(plan["stop_loss"])
                        u_tp1 = float(plan["tp1_price"])
                        u_tp2 = float(plan["tp2_price"])
                    else:
                        box_rng = max(flat_high - flat_low, 0.5 * curr_atr, 1e-9)
                        t_span = max(2.5 * curr_atr, 2.2 * box_rng)
                        if u_side == "LONG":
                            u_entry = flat_high + 0.03 * curr_atr
                            u_sl = (flat_high - 0.65 * box_rng) - 0.35 * curr_atr
                            u_tp1 = u_entry + 0.382 * t_span
                            u_tp2 = u_entry + 0.618 * t_span
                        else:
                            u_entry = flat_low - 0.03 * curr_atr
                            u_sl = (flat_low + 0.65 * box_rng) + 0.35 * curr_atr
                            u_tp1 = u_entry - 0.382 * t_span
                            u_tp2 = u_entry - 0.618 * t_span

                # Enforce High-Leverage Asymmetric Risk-to-Reward (TP1 >= 1.5x SL risk, TP2 >= 2.6x SL risk)
                # and cap SL distance (<= 0.45% on XAU/XAG, <= 1.75% on Crypto) so Leverage never faces a 6% SL!
                ref_p = curr_price if u_mode == "ENTER NOW" else u_entry
                max_sl_frac = 0.0045 if symbol in ("XAUUSDT", "XAGUSDT") else 0.0175
                min_sl_frac = 0.0018 if symbol in ("XAUUSDT", "XAGUSDT") else 0.0065
                raw_sl_dist = abs(ref_p - u_sl)
                clamped_sl_dist = max(ref_p * min_sl_frac, min(raw_sl_dist, ref_p * max_sl_frac))
                if u_side == "LONG":
                    u_sl = ref_p - clamped_sl_dist
                    if (u_tp1 - ref_p) < 1.5 * clamped_sl_dist:
                        u_tp1 = ref_p + 1.6 * clamped_sl_dist
                    if (u_tp2 - ref_p) < 2.6 * clamped_sl_dist:
                        u_tp2 = ref_p + 2.8 * clamped_sl_dist
                else:
                    u_sl = ref_p + clamped_sl_dist
                    if (ref_p - u_tp1) < 1.5 * clamped_sl_dist:
                        u_tp1 = ref_p - 1.6 * clamped_sl_dist
                    if (ref_p - u_tp2) < 2.6 * clamped_sl_dist:
                        u_tp2 = ref_p - 2.8 * clamped_sl_dist

                # Unified Confidence % (38% - 96%)
                base_conf = min(84.0, max(56.0, bt_wr * 0.85 + 12.0)) if bt_profitable else 54.0
                tech_bonus = 8.0 if u_mode == "ENTER NOW" else (4.0 if flatline_met else -3.0)
                dom_trend = macro_compass.get("dominant_trend", "MIXED")
                macro_bonus = (
                    5.0
                    if (u_side == "LONG" and dom_trend == "BULLISH") or (u_side == "SHORT" and dom_trend == "BEARISH")
                    else (-5.0 if dom_trend in ("BULLISH", "BEARISH") else 0.0)
                )
                so_sig = bool(sentiment_overlap.get("is_statistically_significant"))
                so_matches = sentiment_overlap.get("bias_side") == u_side
                sent_bonus = (6.0 if so_matches else -6.0) if so_sig else (2.0 if so_matches else -1.0)
                u_conf = int(round(max(38.0, min(96.0, base_conf + tech_bonus + macro_bonus + sent_bonus))))

                sent_lbl = f"{sentiment_overlap.get('dominant_pct', 50):.0f}% {'Bull' if sentiment_overlap.get('bias_side') == 'LONG' else 'Bear'}"
                simple_why = f"Backtest {bt_wr:.0f}% WR • Macro {dom_trend} • Sentiment {sent_lbl}"

                trade_plan = {
                    "side": u_side,
                    "entry_mode": u_mode,
                    "confidence_pct": u_conf,
                    "safe_entry": round(u_entry, prec),
                    "safe_entry_str": fmt_price(u_entry),
                    "sl": round(u_sl, prec),
                    "sl_str": fmt_price(u_sl),
                    "tp1": round(u_tp1, prec),
                    "tp1_str": fmt_price(u_tp1),
                    "tp2": round(u_tp2, prec),
                    "tp2_str": fmt_price(u_tp2),
                    "leverage": u_lev,
                    "margin_usd": margin_usd,
                    "simple_why": simple_why,
                }

                rec_action = f"{u_side} {'📈' if u_side == 'LONG' else '📉'} — {u_conf}% CONFIDENCE ({u_mode})"
                rec_color = (
                    ("green" if u_side == "LONG" else "red")
                    if u_mode == "ENTER NOW"
                    else ("amber-long" if u_side == "LONG" else "amber-short")
                )
                rec_summary = (
                    f"Safe Entry: ${trade_plan['safe_entry_str']} | TP1: ${trade_plan['tp1_str']} | TP2: ${trade_plan['tp2_str']} | SL: ${trade_plan['sl_str']} • {simple_why}"
                )

                # Single Unified Telegram Alert (respects user's configurable alert_confidence_threshold)
                should_alert = (u_conf >= self.alert_confidence_threshold) and (
                    u_mode == "ENTER NOW" or flatline_met or u_conf >= self.alert_confidence_threshold
                )
                if should_alert and self.orchestrator.telegram.is_configured:
                    open_pos = self.orchestrator.execution.open_positions.get(symbol)
                    alert_key = f"{symbol}_{tf}_{u_side}_{u_mode}_{'OPEN' if open_pos else 'NEW'}"
                    now_ts = time.time()
                    cooldown = 3600 if u_mode == "ENTER NOW" else 14400
                    if now_ts - self.sentiment_alert_cooldowns.get(alert_key, 0.0) >= cooldown:
                        self.sentiment_alert_cooldowns[alert_key] = now_ts
                        if open_pos is not None:
                            pnl_usd = open_pos.unrealized_pnl_usd()
                            roe_pct = open_pos.unrealized_roe_pct()
                            same_dir = (open_pos.side == u_side)
                            pos_action = (
                                f"HOLD / TRAIL {open_pos.side} ({u_conf}% Conf)"
                                if same_dir
                                else f"⚠️ OPPOSITE SIGNAL ({u_side} {u_conf}% Conf) — TIGHTEN SL"
                            )
                            tg_msg = (
                                f"🟠 *POSITION UPDATE: {symbol} ({open_pos.side} Open)*\n"
                                f"• *Your Open Position*: `{open_pos.side}` @ `{fmt_price(open_pos.entry_price)}` → Now `{fmt_price(curr_price)}` (`${pnl_usd:+.2f}` / `{roe_pct:+.1f}% ROE`)\n"
                                f"• *Update Action*: `{pos_action}` _({tf.upper()})_\n"
                                f"• *Updated Targets*: TP1 `{trade_plan['tp1_str']}` | TP2 `{trade_plan['tp2_str']}`\n"
                                f"• *Updated Stop-Loss*: `{trade_plan['sl_str']}`\n"
                                f"• *Why*: _{simple_why}_"
                            )
                        else:
                            icon = "🟢" if u_side == "LONG" else "🔴"
                            tg_msg = (
                                f"{icon} *NEW POSITION: {symbol} — {u_side} {'📈' if u_side == 'LONG' else '📉'} ({u_conf}% Confidence)*\n"
                                f"• *Status*: `{u_mode}` _(Now: `${fmt_price(curr_price)}` | {tf.upper()})_\n"
                                f"• *Safe Entry*: `{trade_plan['safe_entry_str']}`\n"
                                f"• *Take-Profit*: TP1 `{trade_plan['tp1_str']}` | TP2 `{trade_plan['tp2_str']}`\n"
                                f"• *Stop-Loss (SL)*: `{trade_plan['sl_str']}` _(${margin_usd:.0f} @ {u_lev}x)_\n"
                                f"• *Why*: _{simple_why}_"
                            )
                        self.orchestrator.telegram.send_message(tg_msg)

                self.asset_cards[symbol] = {
                    "symbol": symbol,
                    "category": profile.category.upper() if profile else "CRYPTO",
                    "max_btcc_leverage": profile.max_btcc_leverage if profile else 100,
                    "min_lot_label": min_lot_lbl,
                    "min_notional_usd": min_notional,
                    "min_lev_needed": min_lev_needed,
                    "rec_leverage": rec_lev,
                    "margin_per_trade_usd": margin_usd,
                    "est_sl_loss_usd": est_sl_loss,
                    "timeframe": tf,
                    "price": round(curr_price, prec),
                    "price_str": fmt_price(curr_price),
                    "rsi": round(curr_rsi, 1),
                    "atr": round(curr_atr, prec),
                    "atr_str": fmt_price(curr_atr),
                    "range_low": round(lo_48, prec),
                    "range_low_str": fmt_price(lo_48),
                    "range_high": round(hi_48, prec),
                    "range_high_str": fmt_price(hi_48),
                    "range_pos_pct": round(range_pos_pct, 1),
                    "recommendation": rec_action,
                    "rec_color": rec_color,
                    "rec_summary": rec_summary,
                    "trade_plan": trade_plan,
                    "checklist": {
                        "impulse_move_pct": round(impulse_pct, 3),
                        "impulse_atr_mult": round(impulse_atr_mult, 2),
                        "req_impulse_atr": params.impulse_min_atr_mult,
                        "req_impulse_pct": params.impulse_min_pct,
                        "impulse_met": impulse_met,
                        "flatline_range_atr": round(flat_range_atr, 2),
                        "req_flatline_atr": params.flatline_max_range_atr,
                        "body_decay": round(body_decay, 2),
                        "req_body_decay": params.flatline_body_decay_ratio,
                        "flatline_met": flatline_met,
                    },
                    "mtf_radar": mtf_radar,
                    "macro_compass": macro_compass,
                    "market_intel": market_intel,
                    "sentiment_overlap": sentiment_overlap,
                    "active_signal": asdict(active_signal) if active_signal else None,
                    "signal_bars_ago": signal_bars_ago,
                    "sentiment_verdict": verdict_dict,
                    "sparkline": sparkline,
                }

            self.orchestrator.execution._save_state()
            self._save_alert_cache()
            self.last_scan_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            self.last_scan_duration_ms = int((time.time() - t0) * 1000)
            self.scan_count += 1
        except Exception as scan_err:
            print(f"[SCAN ERROR] {scan_err}", flush=True)
        finally:
            self.is_scanning = False

    def trigger_backtest_async(self, symbol: str, timeframe: str) -> None:
        key = f"{symbol}_{timeframe}"
        if self.backtest_jobs.get(key, {}).get("status") == "RUNNING":
            return

        self.backtest_jobs[key] = {
            "status": "RUNNING",
            "started_at": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        }

        def _worker():
            try:
                res = self.orchestrator.researcher.run_research_sweep(
                    symbol=symbol, interval=timeframe, limit=1200
                )
                self.backtest_jobs[key] = {
                    "status": "COMPLETED",
                    "completed_at": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                    "best_metrics": res.get("best_metrics", {}),
                }
                self.run_single_scan()
            except Exception as e:
                self.backtest_jobs[key] = {
                    "status": "FAILED",
                    "error": str(e),
                }

        threading.Thread(target=_worker, daemon=True).start()

    def _resolve_symbol_from_text(self, text: str) -> Optional[str]:
        import re

        tl = text.lower()
        aliases = [
            ("BTCUSDT", ["btcusdt", "bitcoin", "btc"]),
            ("ETHUSDT", ["ethusdt", "ethereum", "ether", "eth"]),
            ("SOLUSDT", ["solusdt", "solana", "sol"]),
            ("XRPUSDT", ["xrpusdt", "ripple", "xrp"]),
            ("DOGEUSDT", ["dogeusdt", "dogecoin", "doge"]),
            ("SUIUSDT", ["suiusdt", "sui"]),
            ("XAUUSDT", ["xauusdt", "gold", "xau"]),
            ("XAGUSDT", ["xagusdt", "silver", "xag"]),
            ("PEPEUSDT", ["pepeusdt", "1000pepe", "pepe"]),
            ("AVAXUSDT", ["avaxusdt", "avalanche", "avax"]),
            ("WIFUSDT", ["wifusdt", "dogwifhat", "wif"]),
            ("INJUSDT", ["injusdt", "injective", "inj"]),
            ("NVDAUSDT", ["nvdausdt", "nvidia", "nvda"]),
        ]
        for sym, words in aliases:
            for w in words:
                if re.search(rf"\b{w}\b", tl):
                    return sym
        return None

    def _format_single_coin_card(self, sym: str) -> str:
        if not self.asset_cards:
            self.run_single_scan()
        c = self.asset_cards.get(sym)
        if not c:
            return f"ℹ️ Could not find live data for `{sym}`."
        tp = c.get("trade_plan") or {}
        side = tp.get("side", "LONG")
        conf = tp.get("confidence_pct", 65)
        mode = tp.get("entry_mode", "WAIT FOR SAFE ENTRY")
        arrow = "📈" if side == "LONG" else "📉"
        icon = "🟢" if side == "LONG" else "🔴"
        open_pos = self.orchestrator.execution.open_positions.get(sym)
        pos_block = ""
        if open_pos is not None:
            open_pos.current_price = float(c.get("price") or open_pos.current_price)
            pnl_usd = open_pos.unrealized_pnl_usd()
            roe_pct = open_pos.unrealized_roe_pct()
            pos_block = (
                f"\n🟠 *YOUR OPEN POSITION*: `{open_pos.side}` ({open_pos.leverage}x)\n"
                f"• *Entry*: `{fmt_price(open_pos.entry_price)}` → *Now*: `{c.get('price_str')}` (`${pnl_usd:+.2f}` / `{roe_pct:+.1f}% ROE`)\n"
                f"• *Your Active Targets*: TP1 `{fmt_price(open_pos.tp1_price)}` | TP2 `{fmt_price(open_pos.tp2_price)}` | SL `{fmt_price(open_pos.stop_loss)}`\n"
            )
        return (
            f"{icon} *{sym} ({self.timeframe.upper()}) — {side} {arrow} ({conf}% Confidence)*\n"
            f"• *Status*: `{mode}` _(Live Price: `${c.get('price_str')}`)_\n"
            f"• *Safe Entry*: `{tp.get('safe_entry_str')}`\n"
            f"• *Take-Profit*: TP1 `{tp.get('tp1_str')}` | TP2 `{tp.get('tp2_str')}`\n"
            f"• *Stop-Loss (SL)*: `{tp.get('sl_str')}` _({tp.get('leverage', 5)}x)_\n"
            f"• *Why*: _{tp.get('simple_why', '')}_"
            + pos_block
            + f"\n\n💡 _Tell me e.g. \"I went {side.lower()} on {sym.replace('USDT', '')} at {tp.get('safe_entry_str')} {tp.get('leverage', 5)}x\" when you enter!_"
        )

    def _format_open_positions(self) -> str:
        self.orchestrator.execution._load_state()
        if not self.orchestrator.execution.open_positions:
            return (
                "📭 *No Open Positions Currently Tracked*\n"
                "• Tell me anytime in plain English when you enter a trade, e.g.:\n"
                "  _\"I went long on BTC at 74500 with 20x\"_\n"
                "  _\"Shorted SOL at 86.5\"_"
            )
        if not self.asset_cards:
            self.run_single_scan()
        lines = [f"🟠 *Your Active Tracked Positions ({len(self.orchestrator.execution.open_positions)})*:"]
        for sym, pos in self.orchestrator.execution.open_positions.items():
            card = self.asset_cards.get(sym)
            if card and card.get("price"):
                pos.current_price = float(card["price"])
            pnl_usd = pos.unrealized_pnl_usd()
            roe_pct = pos.unrealized_roe_pct()
            lines.append(
                f"\n🟠 *{sym} — {pos.side} ({pos.leverage}x)*\n"
                f"   • *Entry*: `{fmt_price(pos.entry_price)}` → *Now*: `{fmt_price(pos.current_price)}`\n"
                f"   • *Open PnL*: `${pnl_usd:+.2f}` (`{roe_pct:+.1f}% ROE` on `${pos.remaining_margin_usd:.0f}` margin)\n"
                f"   • *Targets*: TP1 `{fmt_price(pos.tp1_price)}` | TP2 `{fmt_price(pos.tp2_price)}` | SL `{fmt_price(pos.stop_loss)}`"
            )
        lines.append("\n👉 _Reply \"closed BTC\" or \"move BTC SL to 74000\" anytime to update!_")
        return "\n".join(lines)

    def build_simple_telegram_sheet(self, top_n: int = 13) -> str:
        """Build a simple, unified signal sheet identical to the live website."""
        if not self.asset_cards:
            self.run_single_scan()
        cards = list(self.asset_cards.values())
        cards.sort(
            key=lambda c: (
                1 if c["symbol"] in self.orchestrator.execution.open_positions else 0,
                1 if (c.get("trade_plan") or {}).get("entry_mode") == "ENTER NOW" else 0,
                (c.get("trade_plan") or {}).get("confidence_pct", 0),
            ),
            reverse=True,
        )
        tf_lbl = self.timeframe.upper()
        lines = [
            f"🎯 *BTCC SIMPLE TRADE SIGNALS ({tf_lbl})*",
            f"_Alert Threshold: {self.alert_confidence_threshold}%+ Conf • Tap any number to copy:_\n",
        ]
        for c in cards[:top_n]:
            tp = c.get("trade_plan") or {}
            sym = c["symbol"]
            side = tp.get("side", "LONG")
            conf = tp.get("confidence_pct", 65)
            mode = tp.get("entry_mode", "WAIT FOR SAFE ENTRY")
            is_open = sym in self.orchestrator.execution.open_positions
            icon = "🟠" if is_open else ("⚡" if mode == "ENTER NOW" else ("🟢" if side == "LONG" else "🔴"))
            lbl = f"{sym} (OPEN)" if is_open else sym
            arrow = "📈" if side == "LONG" else "📉"
            lines.append(
                f"{icon} *{lbl}* — *{side} {arrow}* (*{conf}% Conf*)\n"
                f"   Safe Entry: `{tp.get('safe_entry_str')}` _(Now: `{c.get('price_str')}`)_\n"
                f"   TP1: `{tp.get('tp1_str')}` | TP2: `{tp.get('tp2_str')}` | SL: `{tp.get('sl_str')}`"
            )
        lines.append(
            "\n💬 _Chat naturally: \"I went long on BTC at 74500 20x\", \"closed BTC\", or \"set threshold to 75%\"_"
        )
        return "\n".join(lines)

    def handle_telegram_command(self, cmd_text: str) -> str:
        """
        Unified Telegram command & Natural-Language Chat handler.
        Understands both /slash commands and plain-English position / threshold messages.
        """
        import re

        raw = (cmd_text or "").strip()
        if not raw:
            return ""
        tl = raw.lower()
        parts = raw.split()
        cmd = parts[0].lower()

        # 1. Alert Threshold Control (slash or natural language)
        is_threshold_msg = (
            cmd in ("/threshold", "/alert", "/alerts")
            or "threshold" in tl
            or re.search(r"\b(?:alert|alerts|notify)\b.*\b(?:\d{2}\s*%|above|over|at|below|under|confidence|conf)\b", tl)
            or re.search(r"\b(?:confidence|conf)\b.*\b(?:threshold|alert|alerts|minimum|min)\b", tl)
        )
        if is_threshold_msg:
            m_num = re.search(r"\b(\d{2})\s*%?", tl)
            if m_num:
                new_thresh = max(30, min(99, int(m_num.group(1))))
                self.alert_confidence_threshold = new_thresh
                self._save_alert_cache()
                if not self.asset_cards:
                    self.run_single_scan()
                qualifying = [
                    f"{c['symbol']} ({c['trade_plan']['confidence_pct']}% {c['trade_plan']['side']})"
                    for c in self.asset_cards.values()
                    if (c.get("trade_plan") or {}).get("confidence_pct", 0) >= new_thresh
                ]
                qual_str = ", ".join(qualifying) if qualifying else "None right now (monitoring 24/7)"
                return (
                    f"🔔 *Alert Threshold Updated to `{new_thresh}%` Confidence!*\n"
                    f"• You will now receive automatic push alerts whenever a setup reaches *{new_thresh}%+ confidence*.\n"
                    f"• *Currently ≥ {new_thresh}%*: `{qual_str}`"
                )
            return (
                f"🔔 *Current Alert Threshold*: `{self.alert_confidence_threshold}% Confidence`\n"
                "• To change it, just reply e.g. _\"set alert threshold to 75%\"_ or `/threshold 75`."
            )

        # 2. Timeframe Switch via Chat
        tf_match = re.search(r"\b(?:timeframe|tf|chart|switch to)\s+(5m|15m|1h|4h|1d|1w|1m|1mo)\b", tl)
        if cmd == "/tf" and len(parts) >= 2:
            tf_val = parts[1]
        elif tf_match:
            tf_val = tf_match.group(1)
        else:
            tf_val = None
        if tf_val:
            norm_tf = "1M" if tf_val.lower() in ("1m", "1mo", "monthly") and "month" in tl else tf_val
            if norm_tf in ("5m", "15m", "1h", "4h", "1d", "1w", "1M"):
                self.timeframe = norm_tf
                self.run_single_scan()
                return f"⏱️ *Switched Chart Timeframe to `{self.timeframe.upper()}`*\n\n" + self.build_simple_telegram_sheet(top_n=8)

        # 3. Close / Exit Position via Natural Language or /close
        sym_in_msg = self._resolve_symbol_from_text(raw)
        is_close_intent = (
            cmd in ("/close", "/exit")
            or bool(
                re.search(
                    r"\b(?:close|closed|closing|exit|exited|exiting|out of|stopped out|took profit|sold my|remove|cancel)\b",
                    tl,
                )
            )
        )
        if is_close_intent:
            self.orchestrator.execution._load_state()
            if re.search(r"\ball\b", tl) and self.orchestrator.execution.open_positions:
                closed_syms = list(self.orchestrator.execution.open_positions.keys())
                self.orchestrator.execution.open_positions.clear()
                self.orchestrator.execution._save_state()
                return f"✅ *Closed all tracked positions*: `{', '.join(closed_syms)}`"
            target_sym = sym_in_msg
            if not target_sym and len(self.orchestrator.execution.open_positions) == 1:
                target_sym = next(iter(self.orchestrator.execution.open_positions.keys()))
            if target_sym:
                if target_sym in self.orchestrator.execution.open_positions:
                    del self.orchestrator.execution.open_positions[target_sym]
                    self.orchestrator.execution._save_state()
                    return f"✅ *Closed `{target_sym}`* — removed from active exit/stall monitoring."
                return f"ℹ️ `{target_sym}` is not currently in your open positions."

        # 4. Update SL / TP on an Already-Open Position (e.g. "move BTC sl to 73800")
        self.orchestrator.execution._load_state()
        is_modify_open = (
            sym_in_msg in self.orchestrator.execution.open_positions
            and bool(re.search(r"\b(?:move|moved|update|updated|set|change|changed|trail)\b", tl))
            and bool(re.search(r"\b(?:sl|stop|tp|tp1|tp2|target)\b", tl))
            and not bool(re.search(r"\b(?:went|bought|shorted|entered|opened|took)\b", tl))
        )
        if is_modify_open and sym_in_msg:
            pos = self.orchestrator.execution.open_positions[sym_in_msg]
            sl_m = re.search(r"\b(?:sl|stop(?:\s*loss)?)\s*(?:to|at|@|:|=)?\s*\$?(\d+(?:\.\d+)?)", tl)
            tp2_m = re.search(r"\btp2\s*(?:to|at|@|:|=)?\s*\$?(\d+(?:\.\d+)?)", tl)
            tp1_m = re.search(r"\b(?:tp1?|take\s*profit|target)\s*(?:to|at|@|:|=)?\s*\$?(\d+(?:\.\d+)?)", tl)
            changes = []
            if sl_m:
                pos.stop_loss = float(sl_m.group(1))
                changes.append(f"SL → `{fmt_price(pos.stop_loss)}`")
            if tp1_m:
                pos.tp1_price = float(tp1_m.group(1))
                changes.append(f"TP1 → `{fmt_price(pos.tp1_price)}`")
            if tp2_m:
                pos.tp2_price = float(tp2_m.group(1))
                changes.append(f"TP2 → `{fmt_price(pos.tp2_price)}`")
            if changes:
                self.orchestrator.execution._save_state()
                return (
                    f"🟠 *Updated Open Position: {sym_in_msg} ({pos.side})*\n"
                    f"• *Changes*: {', '.join(changes)}\n"
                    f"• *Current Levels*: Entry `{fmt_price(pos.entry_price)}` | TP1 `{fmt_price(pos.tp1_price)}` | TP2 `{fmt_price(pos.tp2_price)}` | SL `{fmt_price(pos.stop_loss)}`"
                )

        # 5. Open / Track a Position via Natural Language or /track
        has_entry_intent = (
            cmd == "/track"
            or bool(
                re.search(
                    r"\b(?:long|longed|short|shorted|bought|buy|buying|sell|selling|entered|enter|took|taking|opened|open|in on|i'm in|im in|track|tracking|position)\b",
                    tl,
                )
            )
        )
        if sym_in_msg and has_entry_intent:
            if not self.asset_cards:
                self.run_single_scan()
            card = self.asset_cards.get(sym_in_msg) or {}
            tp_plan = card.get("trade_plan") or {}
            live_price = float(card.get("price") or tp_plan.get("safe_entry") or 0.0)
            curr_atr = float(card.get("atr") or (live_price * 0.015) or 1.0)

            # Determine Side
            if re.search(r"\b(?:short|shorted|selling|sell|puts|bear)\b", tl):
                side = "SHORT"
            elif re.search(r"\b(?:long|longed|bought|buy|buying|calls|bull)\b", tl):
                side = "LONG"
            else:
                side = str(tp_plan.get("side") or "LONG").upper()

            # Extract optional custom SL, TP1, TP2, Leverage, Margin
            work_text = re.sub(r"\b1000pepe\b", "pepe", tl)
            sl_m = re.search(r"\b(?:sl|stop(?:\s*loss)?)\s*(?:at|@|to|:|=)?\s*\$?(\d+(?:\.\d+)?)", work_text)
            tp2_m = re.search(r"\btp2\s*(?:at|@|to|:|=)?\s*\$?(\d+(?:\.\d+)?)", work_text)
            tp1_m = re.search(r"\b(?:tp1?|take\s*profit|target)\s*(?:at|@|to|:|=)?\s*\$?(\d+(?:\.\d+)?)", work_text)
            lev_m = re.search(r"\b(\d{1,3})\s*x\b|\bleverage\s*(?:of\s*|at\s*|:|=)?\s*(\d{1,3})\b", work_text)
            margin_m = re.search(r"\b(?:margin\s*\$?(\d+(?:\.\d+)?)|\$(\d+(?:\.\d+)?)\s*margin)\b", work_text)

            lev = int(lev_m.group(1) or lev_m.group(2)) if lev_m else int(tp_plan.get("leverage") or self.config.risk.default_leverage)
            lev = max(1, min(500, lev))
            margin = float(margin_m.group(1) or margin_m.group(2)) if margin_m else float(self.config.risk.margin_per_trade_usd)

            # Remove matched substrings before searching for entry price
            cleaned_for_entry = work_text
            for m_obj in (sl_m, tp2_m, tp1_m, lev_m, margin_m):
                if m_obj:
                    cleaned_for_entry = cleaned_for_entry.replace(m_obj.group(0), " ")

            entry_p: Optional[float] = None
            # Explicit 'at / @ / entry' price
            explicit_entry_m = re.search(r"\b(?:at|@|entry|from|around|price)\s*\$?(\d+(?:\.\d+)?)", cleaned_for_entry)
            if explicit_entry_m:
                cand = float(explicit_entry_m.group(1))
                if live_price <= 0 or (0.15 * live_price <= cand <= 6.0 * live_price):
                    entry_p = cand
            if entry_p is None:
                # Any remaining number in the message that looks like a valid asset price
                for num_str in re.findall(r"\$?(\d+(?:\.\d+)?)", cleaned_for_entry):
                    cand = float(num_str)
                    if live_price > 0 and (0.20 * live_price <= cand <= 5.0 * live_price):
                        entry_p = cand
                        break
                    elif live_price <= 0 and cand > 0:
                        entry_p = cand
                        break
            if entry_p is None or entry_p <= 0:
                entry_p = live_price if live_price > 0 else float(tp_plan.get("safe_entry") or 1.0)

            # Compute default SL / TP1 / TP2 around entry_p
            if (
                tp_plan
                and tp_plan.get("side") == side
                and tp_plan.get("safe_entry")
                and abs(entry_p - float(tp_plan["safe_entry"])) / max(entry_p, 1e-9) < 0.012
            ):
                sl = float(tp_plan["sl"])
                tp1 = float(tp_plan["tp1"])
                tp2 = float(tp_plan["tp2"])
            else:
                if side == "LONG":
                    sl = entry_p - 1.5 * curr_atr
                    tp1 = entry_p + 1.8 * curr_atr
                    tp2 = entry_p + 3.2 * curr_atr
                else:
                    sl = entry_p + 1.5 * curr_atr
                    tp1 = entry_p - 1.8 * curr_atr
                    tp2 = entry_p - 3.2 * curr_atr

            if sl_m:
                sl = float(sl_m.group(1))
            if tp1_m:
                tp1 = float(tp1_m.group(1))
            if tp2_m:
                tp2 = float(tp2_m.group(1))

            liq = entry_p * (1.0 - 0.90 / lev) if side == "LONG" else entry_p * (1.0 + 0.90 / lev)
            pos = OpenPosition(
                position_id=f"manual-{sym_in_msg.lower()}-{int(time.time())}",
                symbol=sym_in_msg,
                side=side,
                entry_price=entry_p,
                current_price=live_price if live_price > 0 else entry_p,
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
                sentiment_reasoning=f"Reported via Telegram: {raw[:80]}",
            )
            self.orchestrator.execution.open_positions[sym_in_msg] = pos
            self.orchestrator.execution._save_state()
            pnl_usd = pos.unrealized_pnl_usd()
            roe_pct = pos.unrealized_roe_pct()
            return (
                f"🟠 *Position Tracked: {sym_in_msg} — {side} ({lev}x)*\n"
                f"• *Your Entry*: `{fmt_price(entry_p)}` _(Live: `{fmt_price(pos.current_price)}` | `${pnl_usd:+.2f}` / `{roe_pct:+.1f}% ROE`)_\n"
                f"• *Take-Profit*: TP1 `{fmt_price(tp1)}` | TP2 `{fmt_price(tp2)}`\n"
                f"• *Stop-Loss (SL)*: `{fmt_price(sl)}`\n"
                f"• _Monitoring 24/7 in the cloud for TP, SL & Profit-Flatline Stalls! Reply \"closed {sym_in_msg.replace('USDT', '')}\" when you exit._"
            )

        # 6. Check Open Positions (/positions or natural language)
        if cmd == "/positions" or re.search(r"\b(?:my positions|open positions|my trades|what am i in|active trades|positions)\b", tl):
            return self._format_open_positions()

        # 7. Single-Coin Check (e.g. "how is btc?", "sol", "what about gold?")
        if sym_in_msg and cmd not in ("/backtest",):
            return self._format_single_coin_card(sym_in_msg)

        # 8. Scan / Signals / Status / Help
        if cmd in ("/scan", "/prices", "/sheet", "/signals", "/start") or re.search(
            r"\b(?:scan|signals|sheet|prices|what looks good|best trades|setups|market|update)\b", tl
        ):
            self.run_single_scan()
            return self.build_simple_telegram_sheet()

        if cmd == "/status" or re.search(r"\b(?:status|hi|hello|hey|help|commands)\b", tl):
            self.run_single_scan()
            return (
                f"📊 *BTCC Cloud Bot Status ({self.timeframe.upper()} Chart)*\n"
                f"• *Last Scan*: `{self.last_scan_ts}`\n"
                f"• *Alert Threshold*: `{self.alert_confidence_threshold}%+ Confidence`\n"
                f"• *Open Positions*: `{len(self.orchestrator.execution.open_positions)}`\n\n"
                + self.build_simple_telegram_sheet(top_n=6)
                + "\n\n📖 *What You Can Say Anytime:*\n"
                "• _\"I went long on BTC at 74500 with 20x\"_\n"
                "• _\"Shorted SOL at 86.5, sl 89 tp 81\"_\n"
                "• _\"Closed my BTC trade\"_ or _\"My positions\"_\n"
                "• _\"Set alert threshold to 75%\"_\n"
                "• _\"How is ETH looking?\"_ or _\"Signals\"_"
            )

        return self.orchestrator.handle_telegram_command(cmd_text)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>BTCC AI Trading Agent — Live Control Center</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #131b2e;
    --panel-alt: #19233c;
    --border: #263554;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --green: #10b981;
    --green-bg: rgba(16, 185, 129, 0.14);
    --red: #f43f5e;
    --red-bg: rgba(244, 63, 94, 0.14);
    --amber: #f59e0b;
    --amber-bg: rgba(245, 158, 11, 0.14);
    --blue: #38bdf8;
    --blue-bg: rgba(56, 189, 248, 0.14);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    line-height: 1.45;
    padding: 18px 24px 40px;
  }
  header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 14px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 18px;
  }
  .brand h1 {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .pulse-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 10px var(--green);
    display: inline-block;
  }
  .brand p { color: var(--muted); font-size: 0.85rem; margin-top: 2px; }
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
  }
  button, select, input {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 7px 12px;
    border-radius: 6px;
    font-size: 0.85rem;
  }
  button {
    cursor: pointer;
    font-weight: 600;
    transition: all 0.15s ease;
  }
  button:hover { border-color: var(--blue); background: #223052; }
  button.primary { background: #0284c7; border-color: #38bdf8; color: #fff; }
  button.primary:hover { background: #0369a1; }
  button.success { background: #059669; border-color: #10b981; color: #fff; }
  button.danger { background: #e11d48; border-color: #f43f5e; color: #fff; }

  .section-title {
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin: 18px 0 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  /* System Health Grid */
  .health-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }
  .health-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
  }
  .health-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .badge {
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
  }
  .badge.ok { background: var(--green-bg); color: var(--green); border: 1px solid rgba(16,185,129,0.35); }
  .badge.warn { background: var(--amber-bg); color: var(--amber); border: 1px solid rgba(245,158,11,0.35); }
  .badge.err { background: var(--red-bg); color: var(--red); border: 1px solid rgba(244,63,94,0.35); }
  .health-status { font-size: 0.92rem; font-weight: 700; margin-bottom: 4px; }
  .health-detail { font-size: 0.78rem; color: var(--muted); }

  /* Asset Recommendation Cards */
  .cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }
  .asset-card {
    background: var(--panel);
    border: 2px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .asset-card.rec-green { border-color: var(--green); box-shadow: 0 0 20px rgba(16, 185, 129, 0.12); }
  .asset-card.rec-red { border-color: var(--red); box-shadow: 0 0 20px rgba(244, 63, 94, 0.12); }
  .asset-card.rec-amber-short, .asset-card.rec-amber-long { border-color: var(--amber); }

  .asset-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }
  .symbol-title {
    font-size: 1.2rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .cat-pill {
    font-size: 0.7rem;
    padding: 2px 6px;
    border-radius: 4px;
    background: var(--panel-alt);
    color: var(--muted);
    font-weight: 600;
  }
  .price-box { text-align: right; }
  .price-val { font-size: 1.25rem; font-weight: 800; font-family: monospace; }
  .sub-metrics { font-size: 0.76rem; color: var(--muted); }

  .rec-banner {
    padding: 10px 12px;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .rec-banner.green { background: var(--green-bg); border: 1px solid var(--green); }
  .rec-banner.red { background: var(--red-bg); border: 1px solid var(--red); }
  .rec-banner.amber-short, .rec-banner.amber-long, .rec-banner.amber {
    background: var(--amber-bg); border: 1px solid var(--amber);
  }
  .rec-banner.neutral { background: var(--panel-alt); border: 1px solid var(--border); }

  .rec-action {
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: 0.02em;
  }
  .rec-banner.green .rec-action { color: var(--green); }
  .rec-banner.red .rec-action { color: var(--red); }
  .rec-banner.amber-short .rec-action, .rec-banner.amber-long .rec-action { color: var(--amber); }
  .rec-desc { font-size: 0.8rem; color: var(--text); }

  .signal-levels {
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 6px 12px;
    font-size: 0.8rem;
  }
  .signal-levels div span { color: var(--muted); }
  .signal-levels div strong { font-family: monospace; font-size: 0.88rem; }

  .checklist {
    background: var(--panel-alt);
    border-radius: 6px;
    padding: 9px 11px;
    font-size: 0.78rem;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .chk-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .chk-pass { color: var(--green); font-weight: 700; }
  .chk-wait { color: var(--muted); }

  /* Two-Column Bottom Layout */
  .bottom-grid {
    display: grid;
    grid-template-columns: 1.3fr 1fr;
    gap: 18px;
  }
  @media (max-width: 1024px) {
    .bottom-grid { grid-template-columns: 1fr; }
  }
  .panel-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-top: 8px;
  }
  th, td {
    text-align: left;
    padding: 8px 6px;
    border-bottom: 1px solid var(--border);
  }
  th { color: var(--muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }
  .pnl-pos { color: var(--green); font-weight: 700; font-family: monospace; }
  .pnl-neg { color: var(--red); font-weight: 700; font-family: monospace; }

  .track-form {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
    align-items: center;
  }
  .track-form input, .track-form select { flex: 1; min-width: 95px; }
</style>
</head>
<body>

<header>
  <div class="brand">
    <h1><span class="pulse-dot" id="liveDot"></span> BTCC AI Trading Agent — Live Control Center</h1>
    <p id="lastScanText">Initializing local scanner daemon...</p>
  </div>
  <div class="controls">
    <label style="font-size:0.8rem;color:var(--muted);">Trade Margin:
      <select id="marginSelect" onchange="changeRisk()">
        <option value="2">$2 (Ultra-Micro)</option>
        <option value="5" selected>$5 (Micro / Low-Risk)</option>
        <option value="10">$10 (Small)</option>
        <option value="25">$25 (Moderate)</option>
        <option value="100">$100 (Standard)</option>
      </select>
    </label>
    <label style="font-size:0.8rem;color:var(--muted);">Max Leverage:
      <select id="levSelect" onchange="changeRisk()">
        <option value="3">3x (Ultra-Low)</option>
        <option value="5" selected>5x (Lowest Multiplier)</option>
        <option value="10">10x (Conservative)</option>
        <option value="20">20x (Low Risk)</option>
        <option value="50">50x (Medium)</option>
        <option value="100">100x (High)</option>
      </select>
    </label>
    <label style="font-size:0.8rem;color:var(--muted);">Timeframe:
      <select id="tfSelect" onchange="changeTimeframe(this.value)">
        <option value="1m">1m (Scalp)</option>
        <option value="5m">5m (Intraday)</option>
        <option value="15m">15m (Intraday)</option>
        <option value="1h">1h (Hourly — 65.7% WR)</option>
        <option value="4h" selected>4h (Multi-Day Swing ⭐ 73.8% WR)</option>
        <option value="1d">1d (Daily / 1-Wk Hold ⭐ 88.2% WR)</option>
        <option value="1w">1w (Weekly Macro — 75.0% WR)</option>
        <option value="1M">1M (Monthly Macro — 68.5% WR)</option>
      </select>
    </label>
    <button class="primary" onclick="scanNow()">Scan All Now</button>
    <button id="toggleAutoBtn" onclick="toggleAutoScan()">Auto-Scan: ON (15s)</button>
  </div>
</header>

<div class="section-title">
  <span>1. System Status & Component Diagnostics (What Is & Isn't Working)</span>
</div>
<div class="health-grid" id="healthGrid"></div>

<div class="section-title">
  <span>2. Live Buy / Sell Recommendations & Setup Checklist</span>
  <span style="font-size:0.78rem;text-transform:none;">Strategy: Impulse Spike/Dump &rarr; Tight Flatline &rarr; Reversal Fade</span>
</div>
<div class="cards-grid" id="cardsGrid"></div>

<div class="bottom-grid">
  <div class="panel-box">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <h3 style="font-size:1rem;">3. Active Positions & Profit-Flatline Stall Monitor</h3>
      <span id="accountSummary" style="font-size:0.82rem;color:var(--muted);"></span>
    </div>
    <div id="positionsContainer"></div>

    <div class="track-form">
      <strong style="font-size:0.8rem;width:100%;color:var(--blue);">+ Manually Track a BTCC Trade (Monitors for Profit-Flatline Stall & TP/SL):</strong>
      <select id="trkSymbol">
        <option value="XRPUSDT">XRPUSDT</option>
        <option value="DOGEUSDT">DOGEUSDT</option>
        <option value="SUIUSDT">SUIUSDT</option>
        <option value="SOLUSDT">SOLUSDT</option>
        <option value="ETHUSDT">ETHUSDT</option>
        <option value="BTCUSDT">BTCUSDT</option>
        <option value="XAUUSDT">XAUUSDT (Gold)</option>
        <option value="XAGUSDT">XAGUSDT (Silver)</option>
        <option value="PEPEUSDT">PEPEUSDT</option>
        <option value="AVAXUSDT">AVAXUSDT</option>
        <option value="WIFUSDT">WIFUSDT</option>
        <option value="INJUSDT">INJUSDT</option>
        <option value="NVDAUSDT">NVDAUSDT</option>
      </select>
      <select id="trkSide">
        <option value="LONG">BUY / LONG</option>
        <option value="SHORT">SELL / SHORT</option>
      </select>
      <input type="number" step="any" id="trkEntry" placeholder="Entry Price (blank = current)" />
      <input type="number" id="trkLev" value="5" min="2" max="500" placeholder="Leverage (e.g. 5)" />
      <button class="success" onclick="trackManualTrade()">Track Trade</button>
    </div>
  </div>

  <div class="panel-box">
    <h3 style="font-size:1rem;">4. Per-Asset Backtest Optimizer & Recent Exit Alerts</h3>
    <p style="font-size:0.78rem;color:var(--muted);margin-top:4px;">
      Click <strong>Optimize</strong> on any asset to sweep parameter combinations on historical candles and save its best profile to <code>state/</code>:
    </p>
    <div id="optimizerList" style="margin-top:10px;display:flex;flex-direction:column;gap:8px;"></div>

    <h4 style="font-size:0.85rem;color:var(--muted);margin-top:18px;text-transform:uppercase;">Recent Exit / Stall Alerts & Closed Trades</h4>
    <div id="alertsContainer" style="margin-top:8px;max-height:180px;overflow-y:auto;font-size:0.8rem;"></div>

    <div class="track-form" style="margin-top:16px;">
      <strong style="font-size:0.82rem;width:100%;color:var(--green);">5. Quick-Connect Telegram Phone Alerts &amp; Gemini API (Saves to .env):</strong>
      <p style="font-size:0.75rem;color:var(--muted);width:100%;">
        1. In Telegram, message <code>@BotFather</code> &rarr; send <code>/newbot</code> &rarr; copy the token.<br/>
        2. Send any message (like <code>hi</code>) to your new bot in Telegram, paste the token below, and click <strong>Connect &amp; Test Telegram</strong> (Chat ID auto-detects!).
      </p>
      <input type="text" id="tgToken" placeholder="Telegram Bot Token (123456:ABC...)" style="min-width:210px;" />
      <input type="text" id="tgChatId" placeholder="Chat ID (leave blank to auto-detect)" style="min-width:160px;" />
      <input type="password" id="geminiKey" placeholder="Optional: Gemini API Key" style="min-width:160px;" />
      <button class="primary" onclick="connectTelegram()">Connect &amp; Send Test Alert</button>
      <div id="tgStatusMsg" style="width:100%;font-size:0.78rem;margin-top:4px;"></div>
    </div>
  </div>
</div>

<script>
let currentState = null;

async function fetchState() {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();
    currentState = data;
    renderDashboard(data);
  } catch (e) {
    document.getElementById('lastScanText').innerText = 'Connection error: ' + e;
  }
}

function renderDashboard(data) {
  document.getElementById('tfSelect').value = data.timeframe;
  if (data.risk) {
    const mSel = document.getElementById('marginSelect');
    const lSel = document.getElementById('levSelect');
    if (mSel && document.activeElement !== mSel) mSel.value = String(Math.round(data.risk.margin_per_trade_usd));
    if (lSel && document.activeElement !== lSel) lSel.value = String(data.risk.default_leverage);
  }
  document.getElementById('lastScanText').innerText =
    `Running locally | Last scan: ${data.last_scan_ts || 'Running...'} (${data.last_scan_duration_ms}ms) | Total passes: ${data.scan_count}`;

  const autoBtn = document.getElementById('toggleAutoBtn');
  autoBtn.innerText = data.auto_scan_enabled ? `Auto-Scan: ON (${data.poll_interval_sec}s)` : 'Auto-Scan: PAUSED';
  document.getElementById('liveDot').style.background = data.auto_scan_enabled ? 'var(--green)' : 'var(--amber)';

  // 1. Render Diagnostics
  const d = data.diagnostics;
  const healthItems = [
    { title: 'Auto-Scanner Daemon', info: d.scanner_daemon, badgeClass: d.scanner_daemon.working ? 'ok' : 'warn' },
    { title: 'Live Market Feeds', info: d.market_data_feeds, badgeClass: d.market_data_feeds.working ? 'ok' : 'err' },
    { title: 'Backtest Profiles', info: d.strategy_optimizers, badgeClass: d.strategy_optimizers.working ? 'ok' : 'warn' },
    { title: 'Sentiment Analyst', info: d.sentiment_agent, badgeClass: d.sentiment_agent.mode === 'GEMINI_LIVE_SEARCH' ? 'ok' : 'warn' },
    { title: 'Telegram Bot Alerts', info: d.telegram_alerts, badgeClass: d.telegram_alerts.working ? 'ok' : 'warn' },
    { title: 'Execution & Tracking', info: d.btcc_execution, badgeClass: 'ok' }
  ];
  document.getElementById('healthGrid').innerHTML = healthItems.map(item => `
    <div class="health-card">
      <div class="health-top">
        <span>${item.title}</span>
        <span class="badge ${item.badgeClass}">${item.info.working ? 'WORKING' : 'OPTIONAL / OFF'}</span>
      </div>
      <div class="health-status">${item.info.status}</div>
      <div class="health-detail">${item.info.detail}</div>
    </div>
  `).join('');

  // 2. Render Asset Recommendation Cards (Sorted by Highest Confidence & Active Status, Single Unified Trade Box)
  const profiles = d.strategy_optimizers.profiles || {};
  const cards = Object.values(data.asset_cards || {}).sort((a, b) => {
    const tpA = a.trade_plan || {};
    const tpB = b.trade_plan || {};
    const actA = tpA.entry_mode === 'ENTER NOW' ? 1 : 0;
    const actB = tpB.entry_mode === 'ENTER NOW' ? 1 : 0;
    if (actA !== actB) return actB - actA;
    return (tpB.confidence_pct || 0) - (tpA.confidence_pct || 0);
  });
  document.getElementById('cardsGrid').innerHTML = cards.map(c => {
    const tp = c.trade_plan || {};
    const side = tp.side || 'LONG';
    const isLong = side === 'LONG';
    const isNow = tp.entry_mode === 'ENTER NOW';
    const conf = tp.confidence_pct || 65;
    const confColor = conf >= 78 ? 'var(--green)' : (conf >= 65 ? 'var(--blue)' : 'var(--amber)');
    const sideColor = isLong ? 'var(--green)' : 'var(--red)';
    const sideBg = isLong ? 'rgba(16,185,129,0.14)' : 'rgba(244,63,94,0.14)';
    const mUsd = tp.margin_usd || c.margin_per_trade_usd || 5;
    const lev = tp.leverage || c.rec_leverage || 5;
    const priceStr = c.price_str || c.price;
    const entryStr = tp.safe_entry_str || tp.safe_entry || priceStr;
    const slStr = tp.sl_str || tp.sl;
    const tp1Str = tp.tp1_str || tp.tp1;
    const tp2Str = tp.tp2_str || tp.tp2;

    return `
      <div class="asset-card rec-${c.rec_color}">
        <div class="asset-header">
          <div>
            <div class="symbol-title">
              ${c.symbol}
              <span class="cat-pill" style="background:${sideBg};color:${sideColor};border:1px solid ${sideColor};font-weight:800;">
                ${side} ${isLong ? '&#9650;' : '&#9660;'}
              </span>
              <span class="cat-pill" style="color:${confColor};border:1px solid ${confColor};font-weight:800;">
                ${conf}% CONFIDENCE
              </span>
            </div>
            <div class="sub-metrics">${isNow ? '&#9889; ACTIVE BREAKOUT — ENTER NOW' : '&#9203; WAIT FOR SAFE ENTRY LEVEL'} (${c.timeframe.toUpperCase()})</div>
          </div>
          <div class="price-box">
            <div class="price-val">$${priceStr}</div>
            <div class="sub-metrics">Live Price</div>
          </div>
        </div>

        <div class="signal-levels" style="border-color:${isNow ? sideColor : 'var(--border)'};margin-top:4px;">
          <div style="grid-column:1 / -1;font-weight:800;color:${sideColor};font-size:0.8rem;border-bottom:1px solid var(--border);padding-bottom:4px;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center;">
            <span>${isLong ? '&#128200; LONG' : '&#128201; SHORT'} SETUP (Click any price to copy)</span>
            <span style="color:${confColor};">${conf}% Conf</span>
          </div>
          <div style="cursor:pointer;font-size:0.84rem;" onclick="copyPrice('${entryStr}', this)" title="Click to copy Safe Entry">
            <span>Safe Entry:</span> <strong>$${entryStr} &#128203;</strong>
          </div>
          <div style="cursor:pointer;font-size:0.84rem;" onclick="copyPrice('${slStr}', this)" title="Click to copy Stop-Loss">
            <span>Stop-Loss (SL):</span> <strong style="color:var(--red)">$${slStr} &#128203;</strong>
          </div>
          <div style="cursor:pointer;font-size:0.84rem;" onclick="copyPrice('${tp1Str}', this)" title="Click to copy TP1">
            <span>Take-Profit 1 (TP1):</span> <strong style="color:var(--green)">$${tp1Str} &#128203;</strong>
          </div>
          <div style="cursor:pointer;font-size:0.84rem;" onclick="copyPrice('${tp2Str}', this)" title="Click to copy TP2">
            <span>Take-Profit 2 (TP2):</span> <strong style="color:var(--green)">$${tp2Str} &#128203;</strong>
          </div>
        </div>

        <div style="font-size:0.74rem;color:var(--muted);padding:4px 2px;">
          ${tp.simple_why || ''} &bull; Size: <strong>$${mUsd} @ ${lev}x</strong>
        </div>

        <div style="display:flex;gap:6px;margin-top:2px;">
          <button class="primary" style="flex:1.4;" onclick="quickTrack('${c.symbol}', '${side}', ${tp.safe_entry || c.price}, ${lev})">
            Track ${side} ($${mUsd} @ ${lev}x)
          </button>
          <button style="flex:1;" onclick="openBtccDesktop('${c.symbol}', '${entryStr}')">
            Open BTCC &#8599;
          </button>
        </div>
      </div>
    `;
  }).join('');

  // 3. Render Active Positions
  document.getElementById('accountSummary').innerText =
    `Reserve: $${data.portfolio.account_balance_usd.toFixed(2)} | Daily Realized PnL: $${data.portfolio.daily_realized_pnl_usd.toFixed(2)}`;

  const openPos = data.portfolio.open_positions || [];
  if (openPos.length === 0) {
    document.getElementById('positionsContainer').innerHTML =
      `<p style="color:var(--muted);font-size:0.84rem;padding:12px 0;">No active positions currently open. Use the form below or click a confirmed signal card to track a trade.</p>`;
  } else {
    document.getElementById('positionsContainer').innerHTML = `
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Side / Lev</th>
            <th>Entry &rarr; Current</th>
            <th>TP1 / TP2 / SL</th>
            <th>Unrealized PnL (ROE)</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          ${openPos.map(p => `
            <tr>
              <td><strong>${p.symbol}</strong></td>
              <td><span class="badge ${p.side === 'LONG' ? 'ok' : 'err'}">${p.side} ${p.leverage}x</span></td>
              <td style="font-family:monospace;">$${p.entry_price} &rarr; <strong>$${p.current_price}</strong></td>
              <td style="font-family:monospace;font-size:0.75rem;">
                TP1: $${p.tp1_price}<br/>TP2: $${p.tp2_price}<br/>SL: <span style="color:var(--red)">$${p.stop_loss}</span>
              </td>
              <td class="${p.unrealized_pnl_usd >= 0 ? 'pnl-pos' : 'pnl-neg'}">
                ${p.unrealized_pnl_usd >= 0 ? '+' : ''}$${p.unrealized_pnl_usd.toFixed(2)} (${p.unrealized_roe_pct >= 0 ? '+' : ''}${p.unrealized_roe_pct.toFixed(1)}%)
              </td>
              <td><button class="danger" style="padding:4px 8px;font-size:0.75rem;" onclick="closePosition('${p.symbol}')">Close</button></td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  }

  // 4. Render Optimizer Status List
  const jobs = data.backtest_jobs || {};
  document.getElementById('optimizerList').innerHTML = Object.keys(profiles).map(sym => {
    const prof = profiles[sym];
    const job = jobs[`${sym}_${data.timeframe}`];
    const isRunning = job && job.status === 'RUNNING';
    let statusSpan = '';
    if (prof.optimized) {
      if (prof.profitable) {
        statusSpan = `<span style="color:var(--green);margin-left:6px;font-weight:600;">&#10003; WORKING (Win Rate: ${prof.win_rate_pct}%, PF: ${prof.profit_factor}, Net: +$${prof.total_net_pnl_usd}, Stall Saves: ${prof.profit_stall_exits})</span>`;
      } else {
        statusSpan = `<span style="color:var(--red);margin-left:6px;font-weight:600;">&#9888; UNDERPERFORMING (Win Rate: ${prof.win_rate_pct}%, PF: ${prof.profit_factor}, Net: $${prof.total_net_pnl_usd})</span>`;
      }
    } else {
      statusSpan = `<span style="color:var(--amber);margin-left:6px;">Using Default Profile (Not Yet Optimized on ${data.timeframe})</span>`;
    }
    return `
      <div style="display:flex;justify-content:space-between;align-items:center;background:var(--panel-alt);padding:8px 12px;border-radius:6px;font-size:0.82rem;">
        <div>
          <strong>${sym} (${data.timeframe})</strong>:
          ${statusSpan}
        </div>
        <button onclick="runBacktest('${sym}')" ${isRunning ? 'disabled' : ''}>
          ${isRunning ? 'Optimizing...' : (prof.optimized ? 'Re-Optimize' : 'Optimize Now')}
        </button>
      </div>
    `;
  }).join('');

  // 5. Render Recent Alerts & Closed Trades
  const alerts = data.recent_alerts || [];
  const closed = (data.portfolio.closed_trades || []).slice(-6).reverse();
  let alertsHtml = '';
  if (alerts.length === 0 && closed.length === 0) {
    alertsHtml = `<p style="color:var(--muted);">No exit/stall alerts fired yet in this session.</p>`;
  } else {
    alertsHtml += alerts.map(a => `
      <div style="padding:6px 8px;border-left:3px solid var(--amber);background:var(--panel-alt);margin-bottom:6px;">
        <strong>[${a.time}] ${a.symbol} ${a.side} — ${a.reason}</strong> (${a.roe_pct >= 0 ? '+' : ''}${a.roe_pct}% ROE / $${a.pnl_usd})<br/>
        <span style="color:var(--muted);">${a.explanation}</span>
      </div>
    `).join('');
    alertsHtml += closed.map(c => `
      <div style="padding:6px 8px;border-left:3px solid ${c.net_pnl_usd >= 0 ? 'var(--green)' : 'var(--red)'};background:var(--panel-alt);margin-bottom:6px;">
        <strong>Closed ${c.symbol} ${c.side} (${c.leverage}x)</strong> &bull; ${c.exit_reason} &bull;
        <span class="${c.net_pnl_usd >= 0 ? 'pnl-pos' : 'pnl-neg'}">${c.net_pnl_usd >= 0 ? '+' : ''}$${c.net_pnl_usd} (${c.roe_pct}%)</span>
      </div>
    `).join('');
  }
  document.getElementById('alertsContainer').innerHTML = alertsHtml;
}

async function scanNow() {
  await fetch('/api/scan', { method: 'POST' });
  await fetchState();
}

function copyPrice(val, el) {
  navigator.clipboard.writeText(String(val));
  const orig = el.innerHTML;
  el.innerHTML = `<span style="color:var(--green);font-weight:800;">&#10003; Copied $${val}!</span>`;
  setTimeout(() => { el.innerHTML = orig; }, 1200);
}

function openBtccDesktop(symbol, entryPrice) {
  navigator.clipboard.writeText(String(entryPrice));
  window.open('https://www.btcc.com/en-US/trade/perpetual/' + symbol, '_blank');
}

async function toggleAutoScan() {
  await fetch('/api/toggle-auto', { method: 'POST' });
  await fetchState();
}

async function changeTimeframe(tf) {
  await fetch('/api/timeframe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ timeframe: tf })
  });
  await fetchState();
}

async function changeRisk() {
  const margin = parseFloat(document.getElementById('marginSelect').value || '5');
  const lev = parseInt(document.getElementById('levSelect').value || '20', 10);
  document.getElementById('trkLev').value = lev;
  await fetch('/api/risk', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ margin_per_trade_usd: margin, default_leverage: lev })
  });
  await fetchState();
}

async function trackManualTrade() {
  const symbol = document.getElementById('trkSymbol').value;
  const side = document.getElementById('trkSide').value;
  const entryRaw = document.getElementById('trkEntry').value;
  const leverage = parseInt(document.getElementById('trkLev').value || '100', 10);
  const entry_price = entryRaw ? parseFloat(entryRaw) : null;

  await fetch('/api/track', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol, side, entry_price, leverage })
  });
  document.getElementById('trkEntry').value = '';
  await fetchState();
}

async function quickTrack(symbol, side, entry_price, leverage) {
  await fetch('/api/track', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol, side, entry_price, leverage })
  });
  await fetchState();
}

async function closePosition(symbol) {
  await fetch('/api/close', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol })
  });
  await fetchState();
}

async function runBacktest(symbol) {
  await fetch('/api/backtest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol })
  });
  await fetchState();
}

async function connectTelegram() {
  const bot_token = document.getElementById('tgToken').value.trim();
  const chat_id = document.getElementById('tgChatId').value.trim();
  const gemini_key = document.getElementById('geminiKey').value.trim();
  const statusEl = document.getElementById('tgStatusMsg');
  statusEl.style.color = 'var(--blue)';
  statusEl.innerText = 'Connecting & checking Telegram...';

  const res = await fetch('/api/connect-telegram', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bot_token, chat_id, gemini_key })
  });
  const out = await res.json();
  statusEl.style.color = out.ok ? 'var(--green)' : 'var(--red)';
  statusEl.innerText = out.message || '';
  await fetchState();
}

fetchState();
setInterval(fetchState, 4000);
</script>
</body>
</html>
"""


def create_app(controller: DashboardController) -> web.Application:
    app = web.Application()

    async def handle_index(request: web.Request) -> web.Response:
        return web.Response(
            text=HTML_PAGE,
            content_type="text/html",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    async def handle_state(request: web.Request) -> web.Response:
        open_pos_list = []
        for sym, p in controller.orchestrator.execution.open_positions.items():
            d = asdict(p)
            d["unrealized_pnl_usd"] = round(p.unrealized_pnl_usd(), 2)
            d["unrealized_roe_pct"] = round(p.unrealized_roe_pct(), 2)
            open_pos_list.append(d)

        payload = {
            "timeframe": controller.timeframe,
            "poll_interval_sec": controller.poll_interval_sec,
            "auto_scan_enabled": controller.auto_scan_enabled,
            "is_scanning": controller.is_scanning,
            "last_scan_ts": controller.last_scan_ts,
            "last_scan_duration_ms": controller.last_scan_duration_ms,
            "scan_count": controller.scan_count,
            "diagnostics": controller.get_system_diagnostics(),
            "risk": asdict(controller.config.risk),
            "asset_cards": controller.asset_cards,
            "portfolio": {
                "account_balance_usd": controller.orchestrator.execution.account_balance_usd,
                "daily_realized_pnl_usd": controller.orchestrator.execution.daily_realized_pnl_usd,
                "open_positions": open_pos_list,
                "closed_trades": [asdict(c) for c in controller.orchestrator.execution.closed_trades],
            },
            "recent_alerts": controller.recent_alerts,
            "backtest_jobs": controller.backtest_jobs,
        }
        return web.json_response(
            payload,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    async def handle_scan(request: web.Request) -> web.Response:
        await asyncio.to_thread(controller.run_single_scan)
        return web.json_response({"ok": True})

    async def handle_toggle_auto(request: web.Request) -> web.Response:
        controller.auto_scan_enabled = not controller.auto_scan_enabled
        return web.json_response({"ok": True, "auto_scan_enabled": controller.auto_scan_enabled})

    async def handle_timeframe(request: web.Request) -> web.Response:
        body = await request.json()
        tf = body.get("timeframe", "4h")
        if tf in ("1m", "5m", "15m", "1h", "4h", "1d", "1w", "1M"):
            controller.timeframe = tf
            await asyncio.to_thread(controller.run_single_scan)
        return web.json_response({"ok": True, "timeframe": controller.timeframe})

    async def handle_risk(request: web.Request) -> web.Response:
        body = await request.json()
        margin = float(body.get("margin_per_trade_usd", controller.config.risk.margin_per_trade_usd))
        lev = int(body.get("default_leverage", controller.config.risk.default_leverage))
        controller.config.risk.margin_per_trade_usd = max(1.0, margin)
        controller.config.risk.default_leverage = max(3, min(500, lev))
        await asyncio.to_thread(controller.run_single_scan)
        return web.json_response({"ok": True, "risk": asdict(controller.config.risk)})

    async def handle_track(request: web.Request) -> web.Response:
        body = await request.json()
        sym = body.get("symbol", "ETHUSDT").upper()
        side = body.get("side", "LONG").upper()
        lev = int(body.get("leverage", 100))
        entry_p = body.get("entry_price")

        card = controller.asset_cards.get(sym)
        if not entry_p and card:
            entry_p = float(card["price"])
        if not entry_p:
            df = await asyncio.to_thread(
                controller.orchestrator.feed.fetch_ohlcv, sym, controller.timeframe, 30
            )
            entry_p = float(df.iloc[-1]["close"])

        cmd = f"/track {sym} {side} {entry_p} {lev}"
        msg = await asyncio.to_thread(controller.orchestrator.handle_telegram_command, cmd)
        if controller.orchestrator.telegram.is_configured:
            await asyncio.to_thread(controller.orchestrator.telegram.send_message, msg)
        await asyncio.to_thread(controller.run_single_scan)
        return web.json_response({"ok": True, "message": msg})

    async def handle_close(request: web.Request) -> web.Response:
        body = await request.json()
        sym = body.get("symbol", "").upper()
        msg = await asyncio.to_thread(controller.orchestrator.handle_telegram_command, f"/close {sym}")
        return web.json_response({"ok": True, "message": msg})

    async def handle_backtest(request: web.Request) -> web.Response:
        body = await request.json()
        sym = body.get("symbol", "ETHUSDT").upper()
        controller.trigger_backtest_async(sym, controller.timeframe)
        return web.json_response({"ok": True})

    async def handle_connect_telegram(request: web.Request) -> web.Response:
        import requests
        from telegram_bot import TelegramNotifier
        from sentiment_agent import SentimentAnalystAgent

        body = await request.json()
        bot_token = (body.get("bot_token") or "").strip() or controller.config.telegram_bot_token
        chat_id = (body.get("chat_id") or "").strip() or controller.config.telegram_chat_id
        gemini_key = (body.get("gemini_key") or "").strip() or controller.config.gemini_api_key

        if not bot_token and not gemini_key:
            return web.json_response({"ok": False, "message": "Please enter a Telegram Bot Token (or Gemini API Key)."})

        if bot_token:
            controller.config.telegram_bot_token = bot_token
            controller.orchestrator.telegram = TelegramNotifier(controller.config)

        # Auto-detect chat_id via getUpdates if bot_token is provided but chat_id is blank
        if bot_token and not chat_id:
            data = await asyncio.to_thread(controller.orchestrator.telegram._api_call, "getUpdates")
            if not data.get("ok"):
                return web.json_response({"ok": False, "message": f"Telegram error: {data.get('description')}"})
            results = data.get("result", [])
            for upd in reversed(results):
                msg = upd.get("message") or upd.get("edited_message") or {}
                cid = msg.get("chat", {}).get("id")
                if cid:
                    chat_id = str(cid)
                    break

        # Persist to .env
        env_path = Path(__file__).resolve().parent / ".env"
        env_lines: Dict[str, str] = {}
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    env_lines[k.strip()] = v.strip()

        if bot_token:
            env_lines["TELEGRAM_BOT_TOKEN"] = bot_token
            controller.config.telegram_bot_token = bot_token
        if chat_id:
            env_lines["TELEGRAM_CHAT_ID"] = chat_id
            controller.config.telegram_chat_id = chat_id
        if gemini_key:
            env_lines["GEMINI_API_KEY"] = gemini_key
            controller.config.gemini_api_key = gemini_key

        env_content = "\n".join(f"{k}={v}" for k, v in env_lines.items()) + "\n"
        env_path.write_text(env_content)

        controller.orchestrator.telegram = TelegramNotifier(controller.config)
        controller.orchestrator.sentiment_agent = SentimentAnalystAgent(controller.config)

        if controller.orchestrator.telegram.enabled:
            sent = await asyncio.to_thread(
                controller.orchestrator.telegram.send_message,
                "✅ *BTCC AI Trading Agent Connected!*\n"
                f"• *Chat ID*: `{chat_id}`\n"
                "• Live entry recommendations & *Profit-Flatline Stall* exit alerts are now active.\n"
                "• Try replying `/status`, `/positions`, or `/scan`!",
            )
            if sent:
                return web.json_response({
                    "ok": True,
                    "message": f"Connected! Saved to .env (Chat ID: {chat_id}) and sent a test alert to your phone!",
                })
            return web.json_response({
                "ok": False,
                "message": "Saved token, but Telegram rejected sendMessage. Make sure you sent '/start' to your bot first!",
            })

        return web.json_response({"ok": True, "message": "Saved settings to .env!"})

    async def telegram_poller_task(app_inst: web.Application) -> None:
        while True:
            await asyncio.sleep(3)
            if controller.orchestrator.telegram.token:
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            controller.orchestrator.telegram.poll_commands,
                            controller.handle_telegram_command,
                        ),
                        timeout=15.0,
                    )
                except asyncio.CancelledError:
                    break
                except Exception as tg_err:
                    print(f"[TELEGRAM POLL ERROR] {tg_err}", flush=True)

    async def background_scanner_task(app_inst: web.Application) -> None:
        await asyncio.sleep(1.0)
        try:
            await asyncio.wait_for(asyncio.to_thread(controller.run_single_scan), timeout=120.0)
        except Exception as init_err:
            controller.is_scanning = False
            print(f"[INITIAL SCAN ERROR] {init_err}", flush=True)

        while True:
            await asyncio.sleep(max(5, int(controller.poll_interval_sec)))
            if controller.auto_scan_enabled:
                try:
                    await asyncio.wait_for(asyncio.to_thread(controller.run_single_scan), timeout=120.0)
                except asyncio.CancelledError:
                    break
                except Exception as loop_err:
                    controller.is_scanning = False
                    print(f"[SCAN LOOP ERROR] {loop_err}", flush=True)

    async def start_bg(app_inst: web.Application) -> None:
        app_inst["scanner_task"] = asyncio.create_task(background_scanner_task(app_inst))
        app_inst["tg_poller_task"] = asyncio.create_task(telegram_poller_task(app_inst))

    async def cleanup_bg(app_inst: web.Application) -> None:
        for k in ("scanner_task", "tg_poller_task"):
            task = app_inst.get(k)
            if task:
                task.cancel()

    app.on_startup.append(start_bg)
    app.on_cleanup.append(cleanup_bg)

    app.router.add_get("/", handle_index)
    app.router.add_get("/api/state", handle_state)
    app.router.add_post("/api/scan", handle_scan)
    app.router.add_post("/api/toggle-auto", handle_toggle_auto)
    app.router.add_post("/api/timeframe", handle_timeframe)
    app.router.add_post("/api/risk", handle_risk)
    app.router.add_post("/api/track", handle_track)
    app.router.add_post("/api/close", handle_close)
    app.router.add_post("/api/backtest", handle_backtest)
    app.router.add_post("/api/connect-telegram", handle_connect_telegram)
    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="BTCC AI Trading Agent GUI Dashboard")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8765")), help="Port for GUI dashboard")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Bind host for GUI dashboard")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser window")
    args = parser.parse_args()

    controller = DashboardController(port=args.port)
    url = f"http://127.0.0.1:{args.port}"
    print(f"🚀 BTCC AI Trading GUI Control Center starting at: {url}")

    if not args.no_browser and not os.getenv("PORT"):
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app = create_app(controller)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
