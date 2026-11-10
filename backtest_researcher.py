"""
Automated Backtest Researcher Agent.
1. Fetches historical intraday (1m/5m) data for ETH, BTC, SOL, and Tokenized RWAs (Gold/XAU, NVDA).
2. Simulates the 'Impulse + Flatline -> Reversal' strategy with high-leverage isolated margin (20x-500x).
3. Quantifies the exact benefit of the 'Profit-Flatline Stall Exit' (preventing missed profits when a reversal stalls).
4. Sweeps parameter combinations and uses Gemini as an AI Quant Researcher to critique results and recommend optimal settings.
"""

from __future__ import annotations
import copy
import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from config import AppConfig, StrategyParams, LeverageRiskConfig
from detector import ImpulseFlatlineDetector
from market_data import MarketDataFeed


@dataclass
class BacktestMetrics:
    symbol: str
    timeframe: str
    total_bars: int
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    total_net_pnl_usd: float
    avg_roe_per_trade_pct: float
    max_drawdown_usd: float
    profit_stall_exits_count: int
    tp1_hits_count: int
    tp2_hits_count: int
    stop_loss_count: int
    params: Dict[str, Any]


class AutomatedBacktestResearcher:
    """
    Runs single backtests, parameter grid sweeps, and AI-guided strategy optimization.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.feed = MarketDataFeed()
        self.state_dir = Path(__file__).resolve().parent / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def run_backtest_on_df(
        self,
        df: pd.DataFrame,
        symbol: str,
        params: StrategyParams,
        risk_cfg: LeverageRiskConfig,
    ) -> tuple[BacktestMetrics, List[Dict[str, Any]]]:
        detector = ImpulseFlatlineDetector(params, risk_cfg)
        trades: List[Dict[str, Any]] = []

        in_pos = False
        pos_side = ""
        entry_price = 0.0
        stop_loss = 0.0
        tp1_price = 0.0
        tp2_price = 0.0
        leverage = risk_cfg.default_leverage
        initial_margin = risk_cfg.margin_per_trade_usd
        rem_margin = initial_margin
        tp1_hit = False
        partial_pnl = 0.0
        bars_held = 0
        entry_ts = ""

        stall_exits = 0
        tp1_count = 0
        tp2_count = 0
        sl_count = 0

        equity_curve = [0.0]

        for idx in range(len(df)):
            if in_pos:
                bars_held += 1
                exit_sig = detector.check_position_exit_or_stall(
                    df=df,
                    idx=idx,
                    symbol=symbol,
                    side=pos_side,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    tp1_price=tp1_price,
                    tp2_price=tp2_price,
                    leverage=leverage,
                    margin_usd=rem_margin,
                    bars_held=bars_held,
                    tp1_already_hit=tp1_hit,
                )
                if exit_sig is not None:
                    if exit_sig.urgency == "TAKE_PARTIAL_AND_TRAIL" and not tp1_hit:
                        # Lock in 50% at TP1 and move Stop-Loss to breakeven
                        close_m = rem_margin * 0.5
                        ret = (
                            (exit_sig.current_price - entry_price) / entry_price
                            if pos_side == "LONG"
                            else (entry_price - exit_sig.current_price) / entry_price
                        )
                        fee = close_m * leverage * (risk_cfg.btcc_taker_fee_pct / 100.0)
                        partial_pnl += (close_m * leverage * ret) - fee
                        rem_margin -= close_m
                        stop_loss = entry_price
                        tp1_hit = True
                        tp1_count += 1
                        continue

                    # Full exit
                    ret = (
                        (exit_sig.current_price - entry_price) / entry_price
                        if pos_side == "LONG"
                        else (entry_price - exit_sig.current_price) / entry_price
                    )
                    gross = max(-rem_margin, rem_margin * leverage * ret)
                    entry_Exit_fees = (initial_margin + rem_margin) * leverage * (risk_cfg.btcc_taker_fee_pct / 100.0)
                    net_pnl = partial_pnl + gross - entry_Exit_fees
                    roe_pct = (net_pnl / initial_margin) * 100.0

                    if exit_sig.reason == "PROFIT_FLATLINE_STALL":
                        stall_exits += 1
                    elif exit_sig.reason == "TP2_FULL_REVERSAL_TARGET":
                        tp2_count += 1
                    elif "STOP_LOSS" in exit_sig.reason:
                        sl_count += 1

                    trades.append(
                        {
                            "symbol": symbol,
                            "side": pos_side,
                            "leverage": leverage,
                            "entry_time": entry_ts,
                            "exit_time": str(df.iloc[idx]["timestamp"]),
                            "entry_price": round(entry_price, 4),
                            "exit_price": round(exit_sig.current_price, 4),
                            "net_pnl_usd": round(net_pnl, 2),
                            "roe_pct": round(roe_pct, 2),
                            "exit_reason": exit_sig.reason,
                            "bars_held": bars_held,
                        }
                    )
                    equity_curve.append(equity_curve[-1] + net_pnl)
                    in_pos = False
                continue

            # Look for new Impulse + Flatline Reversal setup
            sig = detector.detect_setup_at_index(df, idx, symbol=symbol)
            if sig is not None:
                in_pos = True
                pos_side = sig.trade_side
                entry_price = sig.entry_price
                stop_loss = sig.stop_loss
                tp1_price = sig.tp1_price
                tp2_price = sig.tp2_price
                leverage = sig.recommended_leverage
                initial_margin = risk_cfg.margin_per_trade_usd
                rem_margin = initial_margin
                tp1_hit = False
                partial_pnl = 0.0
                bars_held = 0
                entry_ts = sig.timestamp

        # Compute summary metrics
        total_trades = len(trades)
        if total_trades == 0:
            metrics = BacktestMetrics(
                symbol=symbol,
                timeframe=params.timeframe,
                total_bars=len(df),
                total_trades=0,
                win_rate_pct=0.0,
                profit_factor=0.0,
                total_net_pnl_usd=0.0,
                avg_roe_per_trade_pct=0.0,
                max_drawdown_usd=0.0,
                profit_stall_exits_count=0,
                tp1_hits_count=0,
                tp2_hits_count=0,
                stop_loss_count=0,
                params=asdict(params),
            )
            return metrics, trades

        pnls = [t["net_pnl_usd"] for t in trades]
        roes = [t["roe_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p <= 0]

        win_rate = (len(wins) / total_trades) * 100.0
        gross_win = sum(wins)
        gross_loss = sum(losses)
        pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

        eq = np.array(equity_curve)
        peaks = np.maximum.accumulate(eq)
        max_dd = float(np.max(peaks - eq))

        metrics = BacktestMetrics(
            symbol=symbol,
            timeframe=params.timeframe,
            total_bars=len(df),
            total_trades=total_trades,
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(pf, 2),
            total_net_pnl_usd=round(float(sum(pnls)), 2),
            avg_roe_per_trade_pct=round(float(np.mean(roes)), 2),
            max_drawdown_usd=round(max_dd, 2),
            profit_stall_exits_count=stall_exits,
            tp1_hits_count=tp1_count,
            tp2_hits_count=tp2_count,
            stop_loss_count=sl_count,
            params=asdict(params),
        )
        return metrics, trades

    def run_research_sweep(
        self,
        symbol: str = "ETHUSDT",
        interval: str = "4h",
        limit: int = 1200,
        use_ai_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluates Carbon V2 Purged Meta-Ensemble (with honest next-bar/maker fills, real fees, spread/slippage,
        and pessimistic intrabar wicks) when engine_version == 'CARBON_V2', saving verified metrics to state/.
        """
        if getattr(self.config.risk, "engine_version", "CARBON_V2") == "CARBON_V2":
            return self._run_carbon_v2_honest_eval(symbol=symbol, interval=interval, limit=max(limit, 1200))

        df = self.feed.fetch_ohlcv(symbol=symbol, interval=interval, limit=limit)

        if interval == "1M":
            grid_impulse_bars = [2, 3]
            grid_flatline_bars = [2, 3]
            grid_flatline_range_atr = [1.05, 1.35, 1.65]
            grid_impulse_atr = [1.1, 1.4, 1.8]
        elif interval == "1w":
            grid_impulse_bars = [2, 3, 4]
            grid_flatline_bars = [2, 3]
            grid_flatline_range_atr = [1.05, 1.35]
            grid_impulse_atr = [1.4, 1.8, 2.2]
        elif interval in ("1h", "4h", "1d"):
            grid_impulse_bars = [3, 5, 7]
            grid_flatline_bars = [3, 4]
            grid_flatline_range_atr = [0.95, 1.25]
            grid_impulse_atr = [1.4, 1.8, 2.2]
        else:
            grid_impulse_bars = [4, 6, 8]
            grid_flatline_bars = [2, 3]
            grid_flatline_range_atr = [1.05, 1.35]
            grid_impulse_atr = [1.4, 1.8, 2.2]

        grid_body_decay = [0.55, 0.78]
        grid_retracement = [0.38, 0.50]
        grid_sl_buffer = [0.35, 0.55]

        sym_u = symbol.upper()
        if interval == "1M":
            if "XAU" in sym_u or "GOLD" in sym_u:
                grid_min_pct = [2.50, 5.00]
            elif "BTC" in sym_u or "NVDA" in sym_u:
                grid_min_pct = [5.00, 9.00]
            else:
                grid_min_pct = [7.00, 12.00]
        elif interval == "1w":
            if "XAU" in sym_u or "GOLD" in sym_u:
                grid_min_pct = [0.75, 1.50]
            elif "BTC" in sym_u or "NVDA" in sym_u:
                grid_min_pct = [2.20, 4.50]
            else:
                grid_min_pct = [3.00, 6.00]
        elif interval == "1d":
            if "XAU" in sym_u or "GOLD" in sym_u:
                grid_min_pct = [0.40, 0.85]
            elif "BTC" in sym_u or "NVDA" in sym_u:
                grid_min_pct = [0.95, 2.00]
            else:
                grid_min_pct = [1.80, 3.40]
        elif interval == "4h":
            if "XAU" in sym_u or "GOLD" in sym_u:
                grid_min_pct = [0.45, 0.85]
            elif "BTC" in sym_u or "NVDA" in sym_u:
                grid_min_pct = [0.80, 1.50]
            else:
                grid_min_pct = [1.10, 2.00]
        elif interval == "1h":
            if "XAU" in sym_u or "GOLD" in sym_u:
                grid_min_pct = [0.25, 0.45]
            elif "BTC" in sym_u or "NVDA" in sym_u:
                grid_min_pct = [0.40, 0.75]
            else:
                grid_min_pct = [0.55, 1.00]
        else:
            if "XAU" in sym_u or "GOLD" in sym_u:
                grid_min_pct = [0.12, 0.20]
            elif "BTC" in sym_u:
                grid_min_pct = [0.20, 0.30] if interval == "5m" else [0.15, 0.25]
            elif "NVDA" in sym_u:
                grid_min_pct = [0.15, 0.25]
            else:
                grid_min_pct = [0.22, 0.32] if interval == "1m" else [0.28, 0.42]

        results: List[BacktestMetrics] = []
        best_trades: List[Dict[str, Any]] = []
        best_score = -1e9

        for imp_b, imp_atr, min_pct, flat_b, flat_rng, body_dec, retr, sl_buf in itertools.product(
            grid_impulse_bars,
            grid_impulse_atr,
            grid_min_pct,
            grid_flatline_bars,
            grid_flatline_range_atr,
            grid_body_decay,
            grid_retracement,
            grid_sl_buffer,
        ):
            p = copy.deepcopy(self.config.strategy)
            p.timeframe = interval
            p.impulse_window_bars = imp_b
            p.impulse_min_atr_mult = imp_atr
            p.impulse_min_pct = min_pct
            p.flatline_bars = flat_b
            p.flatline_max_range_atr = flat_rng
            p.flatline_body_decay_ratio = body_dec
            p.flatline_max_retracement_ratio = retr
            p.stop_loss_atr_buffer = sl_buf
            p.exit_on_profit_flatline = True
            if interval == "1d":
                p.max_hold_bars = 10
            elif interval == "1w":
                p.max_hold_bars = 8
            elif interval == "1M":
                p.max_hold_bars = 6

            m, tr = self.run_backtest_on_df(df, symbol, p, self.config.risk)
            min_req_trades = 1 if (interval == "1M" and len(df) < 60) else 2
            if m.total_trades >= min_req_trades:
                results.append(m)
                margin_u = max(self.config.risk.margin_per_trade_usd, 1.0)
                # Rank by margin-normalized net PnL, win rate (success rate), profit factor, and low drawdown
                score = (
                    ((m.total_net_pnl_usd / margin_u) * 28.0)
                    - ((m.max_drawdown_usd / margin_u) * 14.0)
                    + (m.win_rate_pct * 0.45)
                    + min(m.profit_factor, 8.0) * 4.0
                )
                if score > best_score:
                    best_score = score
                    best_trades = tr

        margin_u = max(self.config.risk.margin_per_trade_usd, 1.0)
        results.sort(
            key=lambda x: (
                ((x.total_net_pnl_usd / margin_u) * 28.0)
                - ((x.max_drawdown_usd / margin_u) * 14.0)
                + (x.win_rate_pct * 0.45)
                + min(x.profit_factor, 8.0) * 4.0,
                x.win_rate_pct,
            ),
            reverse=True,
        )

        top_metric = results[0] if results else None
        ab_comparison = None

        # Run A/B test on the top configuration: Profit-Flatline Stall Exit ON vs OFF
        if top_metric is not None:
            best_p = StrategyParams(**top_metric.params)
            best_p_no_stall = copy.deepcopy(best_p)
            best_p_no_stall.exit_on_profit_flatline = False
            m_no_stall, _ = self.run_backtest_on_df(df, symbol, best_p_no_stall, self.config.risk)
            ab_comparison = {
                "with_profit_flatline_exit": {
                    "net_pnl_usd": top_metric.total_net_pnl_usd,
                    "win_rate_pct": top_metric.win_rate_pct,
                    "profit_factor": top_metric.profit_factor,
                    "stall_saves_count": top_metric.profit_stall_exits_count,
                },
                "without_profit_flatline_exit": {
                    "net_pnl_usd": m_no_stall.total_net_pnl_usd,
                    "win_rate_pct": m_no_stall.win_rate_pct,
                    "profit_factor": m_no_stall.profit_factor,
                },
            }
            # Persist best parameters
            out_file = self.state_dir / f"optimized_{symbol.lower()}_{interval}.json"
            out_file.write_text(
                json.dumps(
                    {
                        "best_metrics": asdict(top_metric),
                        "ab_stall_exit_comparison": ab_comparison,
                        "sample_trades": best_trades[-10:],
                    },
                    indent=2,
                )
            )

        ai_analysis = (
            self._generate_ai_research_summary(symbol, interval, results[:5], ab_comparison)
            if use_ai_summary
            else "Optimization completed."
        )

        return {
            "symbol": symbol,
            "interval": interval,
            "bars_tested": len(df),
            "configs_tested": len(grid_impulse_bars)
            * len(grid_impulse_atr)
            * len(grid_min_pct)
            * len(grid_flatline_bars)
            * len(grid_flatline_range_atr)
            * len(grid_sl_buffer),
            "top_result": asdict(top_metric) if top_metric else None,
            "ab_stall_exit_comparison": ab_comparison,
            "recent_trades": best_trades[-6:],
            "ai_researcher_verdict": ai_analysis,
        }

    def _generate_ai_research_summary(
        self,
        symbol: str,
        interval: str,
        top_results: List[BacktestMetrics],
        ab_comparison: Optional[Dict[str, Any]],
    ) -> str:
        if not top_results:
            return "No trades triggered in the sample window; consider lowering impulse_min_atr_mult."

        best = top_results[0]
        if not self.config.gemini_api_key:
            ab_note = ""
            if ab_comparison:
                w_on = ab_comparison["with_profit_flatline_exit"]
                w_off = ab_comparison["without_profit_flatline_exit"]
                ab_note = (
                    f" Enabling Profit-Flatline Auto-Exit improved Win Rate from {w_off['win_rate_pct']:.1f}% "
                    f"to {w_on['win_rate_pct']:.1f}% and Net PnL from ${w_off['net_pnl_usd']:+.2f} to ${w_on['net_pnl_usd']:+.2f} "
                    f"({w_on['stall_saves_count']} stall exits locked in before reversals faded)."
                )
            return (
                f"Optimal {symbol} ({interval}) setup: {best.params['impulse_window_bars']}-bar impulse "
                f"(>= {best.params['impulse_min_atr_mult']}x ATR) followed by {best.params['flatline_bars']}-bar flatline "
                f"(<= {best.params['flatline_max_range_atr']}x ATR) with {best.params['stop_loss_atr_buffer']}x ATR stop buffer. "
                f"Achieved {best.win_rate_pct}% win rate, {best.profit_factor} profit factor, and "
                f"${best.total_net_pnl_usd:+.2f} net PnL on ${self.config.risk.margin_per_trade_usd:.0f}/trade isolated margin.{ab_note}"
            )

        try:
            from google import genai

            client = genai.Client(api_key=self.config.gemini_api_key)
            prompt = f"""
You are a Quantitative Crypto & RWA Backtest Researcher optimizing an 'Impulse + Flatline -> Reversal' strategy
for a trader using 20x-500x isolated leverage on BTCC (${self.config.risk.margin_per_trade_usd} margin per trade,
${self.config.risk.total_account_balance_usd} account reserve).

Top 3 Parameter Grid Results on {symbol} ({interval}):
{json.dumps([asdict(r) for r in top_results[:3]], indent=2)}

A/B Comparison of 'Profit-Flatline Stall Auto-Exit' (exiting automatically when the reversal stalls in profit):
{json.dumps(ab_comparison, indent=2)}

Provide a concise, high-signal 4-bullet quantitative takeaway covering:
1. Best impulse + flatline window settings for {symbol}
2. Impact of the Profit-Flatline Stall Exit on protecting unrealized gains
3. Recommended stop-loss ATR buffer and safe leverage ceiling to avoid wick liquidations
"""
            resp = client.models.generate_content(model=self.config.gemini_model, contents=prompt)
            return (resp.text or "").strip()
        except Exception:
            return (
                f"Best {symbol} ({interval}): {best.win_rate_pct}% WR, PF {best.profit_factor}, "
                f"Net PnL ${best.total_net_pnl_usd:+.2f}."
            )

    def _run_carbon_v2_honest_eval(
        self,
        symbol: str,
        interval: str = "4h",
        limit: int = 1200,
    ) -> Dict[str, Any]:
        from carbon_v2_engine import (
            CarbonFeatureEngine,
            CarbonCandidateGenerator,
            CarbonLiveBridge,
        )
        from benchmark_argon_vs_carbon import simulate_trade_honest

        df, btc_df = self.feed.fetch_enriched_ohlcv(symbol=symbol, interval=interval, limit=min(limit, 1500))
        feat_df = CarbonFeatureEngine.compute_features(df, btc_df=btc_df)

        is_swing = interval in ("4h", "1d", "1w", "1M")
        bridge = CarbonLiveBridge(self.config.risk, state_dir=self.state_dir)
        model = bridge.models.get("4h" if is_swing else "1h")
        gen = CarbonCandidateGenerator(timeframe="4h" if is_swing else "1h")

        cands = gen.extract_candidates(feat_df, symbol=symbol, start_idx=100, end_idx=len(feat_df) - 2)
        margin_u = float(self.config.risk.margin_per_trade_usd)
        base_risk_u = margin_u * (float(getattr(self.config.risk, "target_sl_margin_risk_pct", 20.0)) / 100.0)

        trades: List[Dict[str, Any]] = []
        eq = [0.0]
        peak_eq = 0.0
        max_dd = 0.0
        tp1_cnt = 0
        tp2_cnt = 0
        sl_cnt = 0
        trail_cnt = 0
        busy_until = -1

        for idx in sorted(cands.keys()):
            if idx <= busy_until:
                continue
            sig = cands[idx]
            if model is not None and model.is_fitted_:
                if not model.score_and_filter(sig):
                    continue
                if is_swing and sig.archetype == "FRESH_CHANNEL_BREAKOUT":
                    if sig.meta_pred_r < max(model.threshold_r_ + 0.07, 0.13) or sig.meta_win_prob < 0.50 or float(sig.features[3]) < 1.12:
                        continue

            sim = simulate_trade_honest(
                df=feat_df,
                symbol=symbol,
                signal_idx=idx,
                side=sig.side,
                stop_dist=sig.stop_dist,
                tp1_dist=sig.tp1_r * sig.stop_dist,
                tp2_dist=sig.tp2_r * sig.stop_dist,
                partial_frac=sig.partial_frac,
                max_hold_bars=sig.max_hold_bars,
                margin_usd=margin_u,
                base_risk_usd=max(0.50, base_risk_u),
                risk_scale=sig.risk_scale,
                model_mode="CARBON_V2",
                trail_atr_mult=sig.trail_atr_mult,
            )
            if sim is None:
                continue
            busy_until = int(sim["exit_idx"])
            trades.append(sim)
            eq.append(eq[-1] + float(sim["net_pnl_usd"]))
            peak_eq = max(peak_eq, eq[-1])
            max_dd = max(max_dd, peak_eq - eq[-1])
            rsn = str(sim["exit_reason"])
            if "TP2" in rsn:
                tp2_cnt += 1
                tp1_cnt += 1
            elif "TRAIL" in rsn or "BREAKEVEN" in rsn:
                trail_cnt += 1
                tp1_cnt += 1
            elif "STOP" in rsn:
                sl_cnt += 1

        wins = [t["net_pnl_usd"] for t in trades if t["net_pnl_usd"] > 0]
        losses = [abs(t["net_pnl_usd"]) for t in trades if t["net_pnl_usd"] <= 0]
        n_tr = len(trades)
        wr = (len(wins) / n_tr * 100.0) if n_tr > 0 else 0.0
        pf = (sum(wins) / max(sum(losses), 1e-6)) if wins else 0.0
        net_pnl = sum(t["net_pnl_usd"] for t in trades)
        avg_roe = float(np.mean([t["roe_pct"] for t in trades])) if trades else 0.0

        bm = BacktestMetrics(
            symbol=symbol,
            timeframe=interval,
            total_bars=len(feat_df),
            total_trades=n_tr,
            win_rate_pct=round(wr, 2),
            profit_factor=round(min(pf, 99.0), 2),
            total_net_pnl_usd=round(net_pnl, 2),
            avg_roe_per_trade_pct=round(avg_roe, 2),
            max_drawdown_usd=round(max_dd, 2),
            profit_stall_exits_count=trail_cnt,
            tp1_hits_count=tp1_cnt,
            tp2_hits_count=tp2_cnt,
            stop_loss_count=sl_cnt,
            params={"engine": "CARBON_V2_PURGED_META", "timeframe": interval},
        )

        out_file = self.state_dir / f"optimized_{symbol.lower()}_{interval}.json"
        out_file.write_text(
            json.dumps(
                {
                    "engine": "CARBON_V2_PURGED_META",
                    "best_metrics": asdict(bm),
                    "sample_trades": trades[-10:],
                },
                indent=2,
            )
        )

        verdict = (
            f"Carbon V2 Honest Evaluation ({symbol} {interval}, {len(feat_df)} bars, full fees+slippage+pessimistic wicks): "
            f"{n_tr} trades | {bm.win_rate_pct}% WR | PF {bm.profit_factor} | "
            f"Net PnL ${bm.total_net_pnl_usd:+.2f} | MaxDD ${bm.max_drawdown_usd:.2f} "
            f"({trail_cnt} Chandelier runner locks)."
        )
        return {
            "symbol": symbol,
            "interval": interval,
            "bars_tested": len(feat_df),
            "configs_tested": 1,
            "best_metrics": asdict(bm),
            "top_result": asdict(bm),
            "recent_trades": trades[-6:],
            "ai_researcher_verdict": verdict,
        }

