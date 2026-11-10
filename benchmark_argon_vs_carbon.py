"""
Head-to-Head Out-of-Sample Walk-Forward Benchmark: Argon V1 vs. Carbon V2
=========================================================================
Evaluates across 8 Perpetual Futures Assets (BTC, ETH, SOL, XRP, DOGE, SUI, AVAX, XAU/PAXG)
on both 1h (Intraday) and 4h (Swing) multi-month / multi-year datasets using:
  1. Strictly Causal Feature & Signal Verification (automated truncation lookahead test)
  2. Purged Rolling Walk-Forward Cross-Validation (zero overlap between train exits and test entries)
  3. Identical Realistic BTCC Perpetual Execution Simulator:
     - Next-bar-open entry (`open[idx + 1]`)
     - BTCC taker fees (0.045% per side)
     - Realistic half-spread + market/stop slippage (2.0 bps majors, 3.5 bps alts + vol shock on stops)
     - Actual 8-hour perpetual funding rate accrual
     - Pessimistic intrabar wick resolution (if both SL and TP touch in the same bar, SL fills first!)
     - Matched $25 baseline 1R risk per $100 isolated-margin trade across both models
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import asdict
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import AppConfig, StrategyParams, LeverageRiskConfig, DEFAULT_ASSETS
from detector import ImpulseFlatlineDetector
from market_data import MarketDataFeed
from backtest_researcher import AutomatedBacktestResearcher
from carbon_v2_engine import (
    CarbonFeatureEngine,
    CarbonCandidateGenerator,
    CarbonCandidateSignal,
    PurgedMetaEnsemble,
)


BENCHMARK_SYMBOLS: List[Tuple[str, str]] = [
    ("BTCUSDT", "BTCUSDT"),
    ("ETHUSDT", "ETHUSDT"),
    ("SOLUSDT", "SOLUSDT"),
    ("XRPUSDT", "XRPUSDT"),
    ("DOGEUSDT", "DOGEUSDT"),
    ("SUIUSDT", "SUIUSDT"),
    ("AVAXUSDT", "AVAXUSDT"),
    ("XAUUSDT", "PAXGUSDT"),
]

MAJOR_SYMBOLS = {"BTCUSDT", "ETHUSDT", "XAUUSDT"}
CACHE_DIR = Path(__file__).resolve().parent / "state" / "benchmark_cache"


def fetch_paginated_perp_data(
    canonical_sym: str,
    fapi_sym: str,
    interval: str,
    total_bars: int,
) -> pd.DataFrame:
    """
    Fetch multi-page 12-column perpetual futures klines + historical 8h funding rates
    from Binance FAPI and cache locally for deterministic, repeatable benchmarking.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{canonical_sym.lower()}_{interval}_{total_bars}.pkl"
    if cache_path.exists():
        try:
            df_cached = pd.read_pickle(cache_path)
            if len(df_cached) >= int(total_bars * 0.7):
                return df_cached
        except Exception:
            pass

    sess = requests.Session()
    sess.headers.update({"User-Agent": "BTCC-Quant-Benchmark/2.0"})

    end_ms: Optional[int] = None
    chunks: List[list] = []
    bars_fetched = 0

    while bars_fetched < total_bars:
        req_limit = min(1500, total_bars - bars_fetched)
        params: Dict[str, Any] = {"symbol": fapi_sym, "interval": interval, "limit": req_limit}
        if end_ms is not None:
            params["endTime"] = end_ms - 1
        resp = sess.get("https://fapi.binance.com/fapi/v1/klines", params=params, timeout=12)
        if resp.status_code != 200:
            break
        raw = resp.json()
        if not isinstance(raw, list) or not raw:
            break
        chunks.append(raw)
        bars_fetched += len(raw)
        end_ms = int(raw[0][0])
        if len(raw) < req_limit:
            break

    all_rows = []
    for ch in reversed(chunks):
        for r in ch:
            all_rows.append(
                {
                    "open_ms": int(r[0]),
                    "timestamp": pd.to_datetime(int(r[0]), unit="ms", utc=True),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                    "quote_vol": float(r[7]),
                    "num_trades": float(r[8]),
                    "taker_buy_vol": float(r[9]),
                    "taker_buy_quote_vol": float(r[10]),
                }
            )

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["open_ms"]).sort_values("open_ms").reset_index(drop=True)
    # Drop the final unclosed bar so dataset contains strictly completed historical candles
    if len(df) > 2:
        df = df.iloc[:-1].reset_index(drop=True)

    # Fetch paginated 8h historical funding rates covering `[start_ms .. last_ms]`
    if len(df) > 0:
        start_ms = int(df["open_ms"].iloc[0]) - 8 * 3600 * 1000
        last_ms = int(df["open_ms"].iloc[-1])
        f_rows = []
        cursor_ms = start_ms
        for _ in range(6):
            if cursor_ms > last_ms:
                break
            fr = sess.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": fapi_sym, "startTime": cursor_ms, "limit": 1000},
                timeout=10,
            )
            if fr.status_code != 200:
                break
            fdata = fr.json()
            if not isinstance(fdata, list) or not fdata:
                break
            for item in fdata:
                f_rows.append(
                    {
                        "timestamp": pd.to_datetime(int(item["fundingTime"]), unit="ms", utc=True),
                        "funding_rate": float(item["fundingRate"]),
                    }
                )
            cursor_ms = int(fdata[-1]["fundingTime"]) + 1
            if len(fdata) < 1000:
                break

        if f_rows:
            fdf = pd.DataFrame(f_rows).sort_values("timestamp").drop_duplicates("timestamp")
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                fdf,
                on="timestamp",
                direction="backward",
            )
            df["funding_rate"] = df["funding_rate"].fillna(0.0001)
        else:
            df["funding_rate"] = 0.0001

    df.to_pickle(cache_path)
    return df


def verify_strict_causality(raw_dfs: Dict[str, pd.DataFrame], timeframe: str) -> Dict[str, Any]:
    """
    Truncates future bars at multiple random timestamps and verifies that Carbon V2's
    features and candidate vectors match full-series evaluation to machine precision.
    """
    btc_raw = raw_dfs["BTCUSDT"]
    gen = CarbonCandidateGenerator(timeframe=timeframe)
    max_feat_diff = 0.0
    checked_points = 0

    for sym in ("ETHUSDT", "SOLUSDT"):
        df_full = CarbonFeatureEngine.compute_features(raw_dfs[sym], btc_df=btc_raw)
        cands_full = gen.extract_candidates(df_full, symbol=sym, start_idx=90)
        test_indices = [450, 1250, 2400, min(3500, len(df_full) - 5)]
        for t_idx in test_indices:
            if t_idx >= len(df_full) - 2:
                continue
            cutoff_ts = raw_dfs[sym]["timestamp"].iloc[t_idx]
            sub_sym = raw_dfs[sym].iloc[: t_idx + 1].copy()
            sub_btc = btc_raw[btc_raw["timestamp"] <= cutoff_ts].copy()
            df_trunc = CarbonFeatureEngine.compute_features(sub_sym, btc_df=sub_btc)

            cols_to_check = [
                "atr14", "rsi14", "ker20", "z_vwap", "delta_norm",
                "delta_z", "cvd_6", "whale_print_ratio", "funding_z",
                "btc_trend_score", "btc_residual_z",
            ]
            v_full = df_full.loc[t_idx, cols_to_check].to_numpy(dtype=float)
            v_trunc = df_trunc.loc[t_idx, cols_to_check].to_numpy(dtype=float)
            diff = float(np.max(np.abs(v_full - v_trunc)))
            max_feat_diff = max(max_feat_diff, diff)
            assert diff < 1e-10, f"Causality violation on {sym} at idx={t_idx}: diff={diff}"

            # Check candidate equivalence at `t_idx - 1` (since extract_candidates stops at len-1)
            chk_i = t_idx - 1
            c_trunc = gen.extract_candidates(df_trunc, symbol=sym, start_idx=max(80, chk_i - 2), end_idx=t_idx)
            assert (chk_i in cands_full) == (chk_i in c_trunc), f"Candidate mismatch at {chk_i}"
            if chk_i in cands_full:
                f_diff = float(np.max(np.abs(cands_full[chk_i].features - c_trunc[chk_i].features)))
                max_feat_diff = max(max_feat_diff, f_diff)
                assert f_diff < 1e-10, f"Candidate feature leak at {chk_i}: {f_diff}"
            checked_points += 1

    return {"passed": True, "points_verified": checked_points, "max_abs_diff": max_feat_diff}


def simulate_trade_honest(
    df: pd.DataFrame,
    symbol: str,
    signal_idx: int,
    side: str,
    stop_dist: float,
    tp1_dist: float,
    tp2_dist: float,
    partial_frac: float,
    max_hold_bars: int,
    margin_usd: float = 100.0,
    base_risk_usd: float = 25.0,
    risk_scale: float = 1.0,
    model_mode: str = "CARBON_V2",     # "ARGON_V1" or "CARBON_V2"
    trail_atr_mult: float = 1.65,
    argon_stall_bars: int = 3,
    argon_min_stall_r: float = 0.8,
) -> Optional[Dict[str, Any]]:
    """
    Identical, leak-free perpetual futures execution simulator for BOTH Argon V1 and Carbon V2:
    - Enters strictly at `open[signal_idx + 1]` +/- half-spread/slippage
    - Applies BTCC taker fee (4.5 bps per side) + funding payments every 8h UTC
    - Enforces pessimistic intrabar wick resolution (Stop-Loss ALWAYS beats TP if both hit in same bar)
    - Fills TP1 and TP2 at exact limit prices (fixing Argon V1's close-price TP1 bug)
    - Sizes integer isolated leverage so 1R structural stop risks `base_risk_usd * risk_scale`
    """
    entry_idx = signal_idx + 1
    n = len(df)
    if entry_idx >= n or stop_dist <= 0:
        return None

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    fr = df["funding_rate"].to_numpy(dtype=float) if "funding_rate" in df.columns else np.full(n, 0.0001)
    cvd6 = df["cvd_6"].to_numpy(dtype=float) if "cvd_6" in df.columns else np.zeros(n)
    ts = df["timestamp"]

    raw_open = float(o[entry_idx])
    if raw_open <= 0:
        return None

    # Realistic half-spread + market entry slippage
    base_slip_frac = 0.00020 if symbol in MAJOR_SYMBOLS else 0.00035
    taker_fee_frac = 0.00045  # BTCC 0.045% taker fee per side

    entry_price = raw_open * (1.0 + base_slip_frac) if side == "LONG" else raw_open * (1.0 - base_slip_frac)
    stop_pct = stop_dist / entry_price
    if stop_pct <= 0.0008:
        return None

    profile = DEFAULT_ASSETS.get(symbol)
    max_asset_lev = min(100, profile.max_btcc_leverage if profile else 100)
    intended_risk_usd = base_risk_usd * risk_scale
    # Leverage chosen so stop_pct * margin_usd * lev == intended_risk_usd
    raw_lev = int(round(intended_risk_usd / max(margin_usd * stop_pct, 1e-9)))
    leverage = max(3, min(max_asset_lev, raw_lev))

    notional_init = margin_usd * leverage
    rem_margin = margin_usd

    if side == "LONG":
        curr_sl = entry_price - stop_dist
        tp1_price = entry_price + tp1_dist
        tp2_price = entry_price + tp2_dist
        liq_price = entry_price * (1.0 - 0.90 / leverage)
    else:
        curr_sl = entry_price + stop_dist
        tp1_price = entry_price - tp1_dist
        tp2_price = entry_price - tp2_dist
        liq_price = entry_price * (1.0 + 0.90 / leverage)

    # Entry taker fee
    realized_pnl = -(notional_init * taker_fee_frac)
    tp1_hit = False
    tp1_bar = -1
    peak_favorable = entry_price
    exit_idx = entry_idx
    exit_price = entry_price
    exit_reason = "TIME_STOP"

    last_bar = min(n - 1, entry_idx + max_hold_bars - 1)

    for k in range(entry_idx, last_bar + 1):
        bars_held = k - entry_idx + 1
        hi_k = float(h[k])
        lo_k = float(l[k])
        cl_k = float(c[k])
        atr_k = max(float(atr[k]), 1e-9)

        # 1. Funding Rate Accrual at 00:00, 08:00, 16:00 UTC (for bars > entry_idx)
        if k > entry_idx:
            t_k = ts.iloc[k]
            if t_k.minute == 0 and (t_k.hour % 8 == 0):
                rem_notional = rem_margin * leverage
                # Positive funding: LONG pays SHORT; Negative funding: SHORT pays LONG
                fund_sign = -1.0 if side == "LONG" else 1.0
                realized_pnl += fund_sign * rem_notional * float(fr[k])

        # 2. Isolated Exchange Liquidation Check (Worst-Case Priority #1)
        if (side == "LONG" and lo_k <= liq_price) or (side == "SHORT" and hi_k >= liq_price):
            realized_pnl -= rem_margin
            rem_margin = 0.0
            exit_idx = k
            exit_price = liq_price
            exit_reason = "LIQUIDATED"
            break

        # 3. Pessimistic Intrabar Stop-Loss Check (Priority #2: SL ALWAYS beats TP on same candle!)
        hit_sl = (lo_k <= curr_sl) if side == "LONG" else (hi_k >= curr_sl)
        if hit_sl:
            # Add volatility-proportional stop-market slippage
            tr_frac = (hi_k - lo_k) / max(cl_k, 1e-9)
            stop_slip = base_slip_frac + min(0.0008, 0.04 * tr_frac)
            fill_sl = curr_sl * (1.0 - stop_slip) if side == "LONG" else curr_sl * (1.0 + stop_slip)
            ret = (fill_sl - entry_price) / entry_price if side == "LONG" else (entry_price - fill_sl) / entry_price
            leg_gross = max(-rem_margin, rem_margin * leverage * ret)
            leg_fee = (rem_margin * leverage) * taker_fee_frac
            realized_pnl += leg_gross - leg_fee
            rem_margin = 0.0
            exit_idx = k
            exit_price = fill_sl
            exit_reason = "TRAIL_OR_BE_STOP" if tp1_hit else "STOP_LOSS"
            break

        # 4. TP1 Partial Limit Fill (only if SL was NOT touched on bar k)
        hit_tp1 = (hi_k >= tp1_price) if side == "LONG" else (lo_k <= tp1_price)
        if (not tp1_hit) and hit_tp1:
            close_m = rem_margin * partial_frac
            ret_1 = (tp1_price - entry_price) / entry_price if side == "LONG" else (entry_price - tp1_price) / entry_price
            leg1_gross = close_m * leverage * ret_1
            leg1_fee = (close_m * leverage) * taker_fee_frac
            realized_pnl += leg1_gross - leg1_fee
            rem_margin -= close_m
            tp1_hit = True
            tp1_bar = k
            peak_favorable = hi_k if side == "LONG" else lo_k

            if model_mode == "ARGON_V1":
                curr_sl = entry_price  # Argon exact breakeven move
            else:
                # Carbon V2: fee-covered breakeven buffer
                curr_sl = (entry_price + 0.12 * stop_dist) if side == "LONG" else (entry_price - 0.12 * stop_dist)

        # 5. TP2 Full Runner Limit Fill (if SL was NOT touched on bar k)
        hit_tp2 = (hi_k >= tp2_price) if side == "LONG" else (lo_k <= tp2_price)
        if hit_tp2 and rem_margin > 0:
            ret_2 = (tp2_price - entry_price) / entry_price if side == "LONG" else (entry_price - tp2_price) / entry_price
            leg2_gross = rem_margin * leverage * ret_2
            leg2_fee = (rem_margin * leverage) * taker_fee_frac
            realized_pnl += leg2_gross - leg2_fee
            rem_margin = 0.0
            exit_idx = k
            exit_price = tp2_price
            exit_reason = "TP2_TARGET"
            break

        # 6. End-of-Bar Active Management (Stall Exit for Argon V1 vs. Orderflow/Chandelier Trail for Carbon V2)
        if model_mode == "ARGON_V1":
            raw_move = (cl_k - entry_price) if side == "LONG" else (entry_price - cl_k)
            r_mult = raw_move / stop_dist
            roe_now = (raw_move / entry_price) * 100.0 * leverage
            if bars_held >= argon_stall_bars and (r_mult >= argon_min_stall_r or roe_now >= 18.0) and k >= argon_stall_bars:
                s_idx = k - argon_stall_bars + 1
                stall_rng = float(np.max(h[s_idx : k + 1]) - np.min(l[s_idx : k + 1]))
                fav_prog = (cl_k - float(o[s_idx])) if side == "LONG" else (float(o[s_idx]) - cl_k)
                if fav_prog <= 0.25 * atr_k and stall_rng <= 1.85 * atr_k:
                    fill_c = cl_k * (1.0 - base_slip_frac) if side == "LONG" else cl_k * (1.0 + base_slip_frac)
                    ret_s = (fill_c - entry_price) / entry_price if side == "LONG" else (entry_price - fill_c) / entry_price
                    realized_pnl += max(-rem_margin, rem_margin * leverage * ret_s) - (rem_margin * leverage * taker_fee_frac)
                    rem_margin = 0.0
                    exit_idx = k
                    exit_price = fill_c
                    exit_reason = "PROFIT_STALL_EXIT"
                    break
        else:
            # Carbon V2 Runner Trail (only activates on bars strictly AFTER TP1 bar)
            if tp1_hit and k > tp1_bar:
                if side == "LONG":
                    peak_favorable = max(peak_favorable, hi_k)
                    chandelier_sl = peak_favorable - trail_atr_mult * atr_k
                    curr_sl = max(curr_sl, chandelier_sl)
                    signed_flow = float(cvd6[k])
                else:
                    peak_favorable = min(peak_favorable, lo_k)
                    chandelier_sl = peak_favorable + trail_atr_mult * atr_k
                    curr_sl = min(curr_sl, chandelier_sl)
                    signed_flow = -float(cvd6[k])

                # Orderflow Reversal Exit on Runner: if held >= 3 bars post-TP1 and aggressive flow flips hard against us
                if (k - tp1_bar) >= 3 and signed_flow <= -0.22:
                    fill_c = cl_k * (1.0 - base_slip_frac) if side == "LONG" else cl_k * (1.0 + base_slip_frac)
                    ret_o = (fill_c - entry_price) / entry_price if side == "LONG" else (entry_price - fill_c) / entry_price
                    realized_pnl += max(-rem_margin, rem_margin * leverage * ret_o) - (rem_margin * leverage * taker_fee_frac)
                    rem_margin = 0.0
                    exit_idx = k
                    exit_price = fill_c
                    exit_reason = "ORDERFLOW_DECAY_EXIT"
                    break

        # 7. Time Stop at end of max_hold_bars
        if k == last_bar and rem_margin > 0:
            fill_t = cl_k * (1.0 - base_slip_frac) if side == "LONG" else cl_k * (1.0 + base_slip_frac)
            ret_t = (fill_t - entry_price) / entry_price if side == "LONG" else (entry_price - fill_t) / entry_price
            realized_pnl += max(-rem_margin, rem_margin * leverage * ret_t) - (rem_margin * leverage * taker_fee_frac)
            rem_margin = 0.0
            exit_idx = k
            exit_price = fill_t
            exit_reason = "TIME_STOP"
            break

    # Actual 1R dollar risk at structural stop (before risk_scale adjustment, baseline = base_risk_usd)
    actual_1r_usd = max(margin_usd * leverage * stop_pct, 1.0)
    net_r = realized_pnl / actual_1r_usd

    return {
        "symbol": symbol,
        "side": side,
        "leverage": leverage,
        "signal_idx": signal_idx,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "entry_time": str(ts.iloc[entry_idx]),
        "exit_time": str(ts.iloc[exit_idx]),
        "entry_price": round(entry_price, 6),
        "exit_price": round(exit_price, 6),
        "net_pnl_usd": round(float(realized_pnl), 4),
        "roe_pct": round(float((realized_pnl / margin_usd) * 100.0), 3),
        "net_r": round(float(net_r), 4),
        "risk_scale": round(float(risk_scale), 3),
        "exit_reason": exit_reason,
        "bars_held": exit_idx - entry_idx + 1,
    }


def optimize_argon_on_train_fold(args: Tuple[str, pd.DataFrame, str]) -> StrategyParams:
    """
    Runs Argon V1's parameter grid strictly on a training fold DataFrame and returns
    Argon's top-scoring StrategyParams (exact scoring formula from `backtest_researcher.py`).
    Caches deterministic fold result by (symbol, interval, start_ms, length).
    """
    symbol, train_df_raw, interval = args
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    c_key = f"argon_opt_{symbol.lower()}_{interval}_{len(train_df_raw)}_{int(train_df_raw['open_ms'].iloc[0])}.json"
    c_file = CACHE_DIR / c_key
    if c_file.exists():
        try:
            return StrategyParams(**json.loads(c_file.read_text()))
        except Exception:
            pass

    df_ind = MarketDataFeed.add_indicators(train_df_raw)
    cfg = AppConfig()
    cfg.risk.margin_per_trade_usd = 100.0
    cfg.risk.default_leverage = 25
    researcher = AutomatedBacktestResearcher(cfg)

    sym_u = symbol.upper()
    if interval == "4h":
        min_pcts = [0.45, 0.85] if "XAU" in sym_u else ([0.80, 1.50] if "BTC" in sym_u else [1.10, 2.00])
    else:
        min_pcts = [0.25, 0.45] if "XAU" in sym_u else ([0.40, 0.75] if "BTC" in sym_u else [0.55, 1.00])

    best_score = -1e9
    best_params = StrategyParams(timeframe=interval)

    for imp_b, imp_atr, min_pct, flat_b, flat_rng, body_dec, sl_buf in itertools.product(
        [3, 5, 7],
        [1.4, 1.8, 2.2],
        min_pcts,
        [3, 4],
        [0.95, 1.25],
        [0.55, 0.78],
        [0.35, 0.55],
    ):
        p = StrategyParams(
            timeframe=interval,
            impulse_window_bars=imp_b,
            impulse_min_atr_mult=imp_atr,
            impulse_min_pct=min_pct,
            flatline_bars=flat_b,
            flatline_max_range_atr=flat_rng,
            flatline_body_decay_ratio=body_dec,
            flatline_max_retracement_ratio=0.45,
            stop_loss_atr_buffer=sl_buf,
            exit_on_profit_flatline=True,
        )
        m, _ = researcher.run_backtest_on_df(df_ind, symbol, p, cfg.risk)
        if m.total_trades >= 2:
            score = (
                ((m.total_net_pnl_usd / 100.0) * 28.0)
                - ((m.max_drawdown_usd / 100.0) * 14.0)
                + (m.win_rate_pct * 0.45)
                + min(m.profit_factor, 8.0) * 4.0
            )
            if score > best_score:
                best_score = score
                best_params = p

    try:
        c_file.write_text(json.dumps(asdict(best_params)))
    except Exception:
        pass
    return best_params


def run_argon_oos_fold(
    df_argon_ind: pd.DataFrame,
    symbol: str,
    params: StrategyParams,
    test_start: int,
    test_end: int,
) -> List[Dict[str, Any]]:
    risk_cfg = LeverageRiskConfig(margin_per_trade_usd=100.0, default_leverage=25, target_sl_margin_risk_pct=25.0)
    detector = ImpulseFlatlineDetector(params, risk_cfg)
    trades: List[Dict[str, Any]] = []
    busy_until = -1

    for idx in range(test_start, min(test_end, len(df_argon_ind) - 1)):
        if idx <= busy_until:
            continue
        sig = detector.detect_setup_at_index(df_argon_ind, idx, symbol=symbol)
        if sig is None:
            continue
        stop_dist = abs(sig.entry_price - sig.stop_loss)
        tp1_dist = abs(sig.tp1_price - sig.entry_price)
        tp2_dist = abs(sig.tp2_price - sig.entry_price)
        tr = simulate_trade_honest(
            df=df_argon_ind,
            symbol=symbol,
            signal_idx=idx,
            side=sig.trade_side,
            stop_dist=stop_dist,
            tp1_dist=tp1_dist,
            tp2_dist=tp2_dist,
            partial_frac=0.50,
            max_hold_bars=params.max_hold_bars,
            margin_usd=100.0,
            base_risk_usd=25.0,
            risk_scale=1.0,
            model_mode="ARGON_V1",
            argon_stall_bars=params.exit_flatline_bars,
            argon_min_stall_r=params.min_profit_r_for_stall_exit,
        )
        if tr is not None:
            trades.append(tr)
            busy_until = tr["exit_idx"]
    return trades


def apply_portfolio_concurrency_cap(
    trades: List[Dict[str, Any]],
    max_concurrent: int = 3,
    max_same_side: int = 2,
) -> List[Dict[str, Any]]:
    """
    Enforces realistic BTCC portfolio-level concurrency (`LeverageRiskConfig.max_concurrent_positions = 3`)
    and correlation guardrails (`max_same_side = 2`) chronologically by `entry_idx` across all symbols,
    preventing 8 correlated crypto positions from opening simultaneously on the same candle.
    """
    sorted_tr = sorted(
        trades,
        key=lambda t: (int(t["entry_idx"]), -float(t.get("meta_pred_r", 0.0)), t["symbol"]),
    )
    accepted: List[Dict[str, Any]] = []
    active: List[Dict[str, Any]] = []

    for tr in sorted_tr:
        e_idx = int(tr["entry_idx"])
        active = [a for a in active if int(a["exit_idx"]) >= e_idx]
        if any(a["symbol"] == tr["symbol"] for a in active):
            continue
        if len(active) >= max_concurrent:
            continue
        same_dir = sum(1 for a in active if a["side"] == tr["side"])
        if same_dir >= max_same_side:
            continue
        accepted.append(tr)
        active.append(tr)
    return accepted


def compute_portfolio_metrics(
    trades: List[Dict[str, Any]],
    oos_days: float,
    starting_capital: float = 1000.0,
) -> Dict[str, Any]:
    if not trades:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "expectancy_usd": 0.0,
            "total_net_pnl_usd": 0.0,
            "max_drawdown_usd": 0.0,
            "return_over_dd": 0.0,
            "sharpe_annualized": 0.0,
            "sortino_annualized": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "payoff_ratio": 0.0,
            "equity_curve": [],
        }

    ordered = sorted(trades, key=lambda x: x["exit_time"])
    pnls = np.array([float(t["net_pnl_usd"]) for t in ordered], dtype=float)
    rs = np.array([float(t["net_r"]) for t in ordered], dtype=float)

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_win = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = float(np.sum(np.abs(losses))) if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 1e-9 else (99.0 if gross_win > 0 else 0.0)

    cum_pnl = np.concatenate([[0.0], np.cumsum(pnls)])
    peaks = np.maximum.accumulate(cum_pnl)
    dd_series = peaks - cum_pnl
    max_dd = float(np.max(dd_series))
    net_pnl = float(cum_pnl[-1])

    # Daily mark-to-realized portfolio returns for Annualized Sharpe & Sortino (365d crypto calendar)
    df_tr = pd.DataFrame(ordered)
    df_tr["exit_day"] = pd.to_datetime(df_tr["exit_time"]).dt.floor("D")
    min_day = pd.to_datetime(ordered[0]["entry_time"]).floor("D")
    max_day = pd.to_datetime(ordered[-1]["exit_time"]).floor("D")
    all_days = pd.date_range(min_day, max_day, freq="D", tz="UTC")
    daily_pnl = df_tr.groupby("exit_day")["net_pnl_usd"].sum().reindex(all_days, fill_value=0.0).to_numpy(dtype=float)

    daily_ret = daily_pnl / starting_capital
    mean_d = float(np.mean(daily_ret))
    std_d = float(np.std(daily_ret))
    downside = daily_ret[daily_ret < 0.0]
    down_std = float(math.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 1e-9

    sharpe = (mean_d / std_d) * math.sqrt(365.0) if std_d > 1e-9 else 0.0
    sortino = (mean_d / down_std) * math.sqrt(365.0) if down_std > 1e-9 else 0.0

    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss = float(np.mean(np.abs(losses))) if len(losses) else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 1e-9 else 0.0

    # Downsample equity curve by day for compact UI plotting
    cum_daily = np.cumsum(daily_pnl)
    step = max(1, len(all_days) // 90)
    eq_points = [
        {"date": str(all_days[i].date()), "pnl": round(float(cum_daily[i]), 2)}
        for i in range(0, len(all_days), step)
    ]
    if eq_points[-1]["date"] != str(all_days[-1].date()):
        eq_points.append({"date": str(all_days[-1].date()), "pnl": round(float(cum_daily[-1]), 2)})

    return {
        "total_trades": len(ordered),
        "win_rate_pct": round((len(wins) / len(ordered)) * 100.0, 2),
        "profit_factor": round(pf, 3),
        "expectancy_r": round(float(np.mean(rs)), 4),
        "expectancy_usd": round(float(np.mean(pnls)), 2),
        "total_net_pnl_usd": round(net_pnl, 2),
        "max_drawdown_usd": round(max_dd, 2),
        "return_over_dd": round(net_pnl / max(max_dd, 1.0), 2),
        "sharpe_annualized": round(sharpe, 2),
        "sortino_annualized": round(sortino, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "payoff_ratio": round(payoff, 2),
        "oos_days": round(oos_days, 1),
        "equity_curve": eq_points,
    }


def run_walk_forward_for_timeframe(
    interval: str,
    total_bars: int,
    train_bars: int,
    fold_bars: int,
) -> Dict[str, Any]:
    print(f"\n{'='*78}")
    print(f"FETCHING & PREPARING {interval.upper()} PERPETUAL FUTURES PANEL ({total_bars} bars x {len(BENCHMARK_SYMBOLS)} assets)")
    print(f"{'='*78}")

    t0 = time.time()
    raw_dfs: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {
            sym: pool.submit(fetch_paginated_perp_data, sym, fapi_sym, interval, total_bars)
            for sym, fapi_sym in BENCHMARK_SYMBOLS
        }
        for sym, fut in futs.items():
            raw_dfs[sym] = fut.result()

    min_len = min(len(df) for df in raw_dfs.values())
    # Align all symbols to common bar count from tail so walk-forward folds match chronologically
    for sym in list(raw_dfs.keys()):
        raw_dfs[sym] = raw_dfs[sym].iloc[-min_len:].reset_index(drop=True)

    causality_res = verify_strict_causality(raw_dfs, timeframe=interval)
    print(
        f"[PASS] Zero-Lookahead Causality Verification ({interval}): "
        f"{causality_res['points_verified']} truncation checkpoints, max_abs_diff={causality_res['max_abs_diff']:.2e}"
    )

    # Precompute Argon indicators & Carbon V2 causal features + candidate pools
    btc_raw = raw_dfs["BTCUSDT"]
    argon_dfs: Dict[str, pd.DataFrame] = {}
    carbon_dfs: Dict[str, pd.DataFrame] = {}
    carbon_cand_Outcomes: Dict[str, List[Tuple[CarbonCandidateSignal, Dict[str, Any]]]] = {}

    gen = CarbonCandidateGenerator(timeframe=interval)
    for sym, _ in BENCHMARK_SYMBOLS:
        df_a = MarketDataFeed.add_indicators(raw_dfs[sym])
        df_a["funding_rate"] = raw_dfs[sym]["funding_rate"].to_numpy()
        argon_dfs[sym] = df_a

        df_c = CarbonFeatureEngine.compute_features(raw_dfs[sym], btc_df=btc_raw)
        carbon_dfs[sym] = df_c

        cands_map = gen.extract_candidates(df_c, symbol=sym, start_idx=80)
        sim_pairs: List[Tuple[CarbonCandidateSignal, Dict[str, Any]]] = []
        for idx_k in sorted(cands_map.keys()):
            sig = cands_map[idx_k]
            tr = simulate_trade_honest(
                df=df_c,
                symbol=sym,
                signal_idx=idx_k,
                side=sig.side,
                stop_dist=sig.stop_dist,
                tp1_dist=sig.tp1_r * sig.stop_dist,
                tp2_dist=sig.tp2_r * sig.stop_dist,
                partial_frac=sig.partial_frac,
                max_hold_bars=sig.max_hold_bars,
                margin_usd=100.0,
                base_risk_usd=25.0,
                risk_scale=1.0,
                model_mode="CARBON_V2",
                trail_atr_mult=sig.trail_atr_mult,
            )
            if tr is not None:
                sim_pairs.append((sig, tr))
        carbon_cand_Outcomes[sym] = sim_pairs

    # 0. Record Argon V1's Illusory In-Sample Claim (on a 1,000-bar slice using Argon's buggy simulator)
    illusory_wrs = []
    illusory_pfs = []
    illusory_pnl_sum = 0.0
    illusory_trades = 0
    for sym, _ in BENCHMARK_SYMBOLS:
        slice_1000 = raw_dfs[sym].iloc[-1000:].reset_index(drop=True)
        best_p = optimize_argon_on_train_fold((sym, slice_1000, interval))
        cfg_tmp = AppConfig()
        cfg_tmp.risk.margin_per_trade_usd = 100.0
        cfg_tmp.risk.default_leverage = 25
        m_ill, _ = AutomatedBacktestResearcher(cfg_tmp).run_backtest_on_df(
            MarketDataFeed.add_indicators(slice_1000), sym, best_p, cfg_tmp.risk
        )
        if m_ill.total_trades > 0:
            illusory_wrs.append(m_ill.win_rate_pct)
            illusory_pfs.append(min(m_ill.profit_factor, 25.0))
            illusory_pnl_sum += m_ill.total_net_pnl_usd
            illusory_trades += m_ill.total_trades

    # Build Walk-Forward Folds
    folds: List[Tuple[int, int]] = []
    cur_test = train_bars
    while cur_test + 100 < min_len:
        folds.append((cur_test, min(min_len - 1, cur_test + fold_bars)))
        cur_test += fold_bars

    argon_wf_trades: List[Dict[str, Any]] = []
    argon_static_trades: List[Dict[str, Any]] = []
    carbon_wf_trades: List[Dict[str, Any]] = []

    # Load Argon's existing static optimized params per symbol if present
    state_dir = Path(__file__).resolve().parent / "state"
    static_params_map: Dict[str, StrategyParams] = {}
    for sym, _ in BENCHMARK_SYMBOLS:
        p_file = state_dir / f"optimized_{sym.lower()}_{interval}.json"
        if p_file.exists():
            try:
                d_json = json.loads(p_file.read_text())
                static_params_map[sym] = StrategyParams(**d_json["best_metrics"]["params"])
                continue
            except Exception:
                pass
        sp = StrategyParams(timeframe=interval)
        static_params_map[sym] = sp

    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as proc_pool:
        for f_i, (test_start, test_end) in enumerate(folds):
            train_start = max(0, test_start - train_bars)

            # 1A. Argon Static Deployed Params on unseen [test_start .. test_end)
            for sym, _ in BENCHMARK_SYMBOLS:
                argon_static_trades.extend(
                    run_argon_oos_fold(argon_dfs[sym], sym, static_params_map[sym], test_start, test_end)
                )

            # 1B. Argon Walk-Forward Re-Optimized strictly on [train_start .. test_start)
            opt_jobs = [
                (sym, raw_dfs[sym].iloc[train_start:test_start].reset_index(drop=True), interval)
                for sym, _ in BENCHMARK_SYMBOLS
            ]
            wf_best_params = list(proc_pool.map(optimize_argon_on_train_fold, opt_jobs))
            for (sym, _), b_param in zip(BENCHMARK_SYMBOLS, wf_best_params):
                argon_wf_trades.extend(
                    run_argon_oos_fold(argon_dfs[sym], sym, b_param, test_start, test_end)
                )

            # 2. Carbon V2 Purged Walk-Forward Meta-Ensemble strictly on non-overlapping exits < test_start
            X_tr_list = []
            y_tr_list = []
            ts_tr_list = []
            for sym, _ in BENCHMARK_SYMBOLS:
                busy_tr = -1
                for sig, tr_out in carbon_cand_Outcomes[sym]:
                    if sig.bar_idx <= busy_tr:
                        continue
                    # STRICT PURGE CONDITION: both signal AND final trade exit must occur strictly BEFORE test_start!
                    if sig.bar_idx >= train_start and tr_out["exit_idx"] < test_start:
                        X_tr_list.append(sig.features)
                        y_tr_list.append(tr_out["net_r"])
                        ts_tr_list.append(sig.bar_idx)
                        busy_tr = tr_out["exit_idx"]

            meta = PurgedMetaEnsemble(ridge_alpha=180.0, n_trees=20, learning_rate=0.06, min_leaf=25)
            if len(y_tr_list) >= 30:
                meta.fit_with_internal_calibration(
                    np.vstack(X_tr_list),
                    np.array(y_tr_list, dtype=float),
                    np.array(ts_tr_list, dtype=int),
                )

            # Execute Carbon V2 on unseen [test_start .. test_end) with non-overlapping per-symbol positions
            for sym, _ in BENCHMARK_SYMBOLS:
                busy_until_c = -1
                df_c = carbon_dfs[sym]
                for sig, _ in carbon_cand_Outcomes[sym]:
                    if sig.bar_idx < test_start or sig.bar_idx >= test_end:
                        continue
                    if sig.bar_idx <= busy_until_c:
                        continue
                    sig_eval = copy.copy(sig)
                    if not meta.score_and_filter(sig_eval):
                        continue
                    tr_live = simulate_trade_honest(
                        df=df_c,
                        symbol=sym,
                        signal_idx=sig_eval.bar_idx,
                        side=sig_eval.side,
                        stop_dist=sig_eval.stop_dist,
                        tp1_dist=sig_eval.tp1_r * sig_eval.stop_dist,
                        tp2_dist=sig_eval.tp2_r * sig_eval.stop_dist,
                        partial_frac=sig_eval.partial_frac,
                        max_hold_bars=sig_eval.max_hold_bars,
                        margin_usd=100.0,
                        base_risk_usd=25.0,
                        risk_scale=sig_eval.risk_scale,
                        model_mode="CARBON_V2",
                        trail_atr_mult=sig_eval.trail_atr_mult,
                    )
                    if tr_live is not None:
                        tr_live["archetype"] = sig_eval.archetype
                        tr_live["meta_pred_r"] = round(sig_eval.meta_pred_r, 3)
                        tr_live["meta_win_prob"] = round(sig_eval.meta_win_prob, 3)
                        carbon_wf_trades.append(tr_live)
                        busy_until_c = tr_live["exit_idx"]

            print(
                f"  Fold {f_i+1}/{len(folds)} [{interval}] bars [{test_start}..{test_end}) | "
                f"Purged Train N={len(y_tr_list)} (tau*={meta.threshold_r_:.2f}R) | "
                f"Cum OOS Trades -> Argon-WF: {len(argon_wf_trades)}, Carbon-V2: {len(carbon_wf_trades)}"
            )

    bar_hours = 4.0 if interval == "4h" else 1.0
    oos_bars = (folds[-1][1] - folds[0][0]) if folds else 0
    oos_days = (oos_bars * bar_hours) / 24.0
    oos_start_ts = str(raw_dfs["BTCUSDT"]["timestamp"].iloc[folds[0][0]]) if folds else ""
    oos_end_ts = str(raw_dfs["BTCUSDT"]["timestamp"].iloc[folds[-1][1]]) if folds else ""

    argon_static_trades = apply_portfolio_concurrency_cap(argon_static_trades, max_concurrent=3, max_same_side=2)
    argon_wf_trades = apply_portfolio_concurrency_cap(argon_wf_trades, max_concurrent=3, max_same_side=2)
    carbon_wf_trades = apply_portfolio_concurrency_cap(carbon_wf_trades, max_concurrent=3, max_same_side=2)

    m_argon_static = compute_portfolio_metrics(argon_static_trades, oos_days)
    m_argon_wf = compute_portfolio_metrics(argon_wf_trades, oos_days)
    m_carbon = compute_portfolio_metrics(carbon_wf_trades, oos_days)

    # Per-symbol breakdown
    per_symbol: Dict[str, Any] = {}
    for sym, _ in BENCHMARK_SYMBOLS:
        a_sym = [t for t in argon_wf_trades if t["symbol"] == sym]
        c_sym = [t for t in carbon_wf_trades if t["symbol"] == sym]
        ma = compute_portfolio_metrics(a_sym, oos_days)
        mc = compute_portfolio_metrics(c_sym, oos_days)
        per_symbol[sym] = {
            "argon_wf": {k: v for k, v in ma.items() if k != "equity_curve"},
            "carbon_v2": {k: v for k, v in mc.items() if k != "equity_curve"},
            "argon_curve": ma["equity_curve"],
            "carbon_curve": mc["equity_curve"],
        }

    # Archetype breakdown for Carbon V2
    arch_breakdown: Dict[str, Any] = {}
    seen_archs = sorted(set(t.get("archetype", "UNKNOWN") for t in carbon_wf_trades))
    for arch_name in seen_archs:
        sub_t = [t for t in carbon_wf_trades if t.get("archetype") == arch_name]
        m_sub = compute_portfolio_metrics(sub_t, oos_days)
        arch_breakdown[arch_name] = {k: v for k, v in m_sub.items() if k != "equity_curve"}

    elapsed = time.time() - t0
    print(f"Completed {interval} Walk-Forward Benchmark in {elapsed:.1f}s.")

    return {
        "interval": interval,
        "total_bars_per_asset": min_len,
        "oos_bars_per_asset": oos_bars,
        "oos_asset_bars_total": oos_bars * len(BENCHMARK_SYMBOLS),
        "oos_days": round(oos_days, 1),
        "oos_window": f"{oos_start_ts[:10]} -> {oos_end_ts[:10]}",
        "causality_check": causality_res,
        "argon_illusory_insample": {
            "avg_claimed_win_rate_pct": round(float(np.mean(illusory_wrs)) if illusory_wrs else 0.0, 2),
            "avg_claimed_profit_factor": round(float(np.mean(illusory_pfs)) if illusory_pfs else 0.0, 2),
            "total_claimed_pnl_usd": round(illusory_pnl_sum, 2),
            "total_trades_across_8_assets": illusory_trades,
        },
        "argon_static_oos": m_argon_static,
        "argon_walkforward_oos": m_argon_wf,
        "carbon_v2_walkforward_oos": m_carbon,
        "per_symbol": per_symbol,
        "carbon_archetypes": arch_breakdown,
    }


def main() -> None:
    res_4h = run_walk_forward_for_timeframe(interval="4h", total_bars=4500, train_bars=1500, fold_bars=500)
    res_1h = run_walk_forward_for_timeframe(interval="1h", total_bars=6000, train_bars=2000, fold_bars=800)

    summary = {"4h": res_4h, "1h": res_1h}
    out_path = Path(__file__).resolve().parent / "state" / "benchmark_argon_vs_carbon_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved full benchmark results -> {out_path}")

    for tf, r in summary.items():
        ill = r["argon_illusory_insample"]
        ast = r["argon_static_oos"]
        awf = r["argon_walkforward_oos"]
        cbn = r["carbon_v2_walkforward_oos"]
        print(f"\n================ SUMMARY: {tf.upper()} ({r['oos_window']} | {r['oos_days']} OOS Days | {r['oos_asset_bars_total']:,} Asset-Bars) ================")
        print(
            f"  [0] Argon V1 (Illusory In-Sample Claim): Trades={ill['total_trades_across_8_assets']:4d} | "
            f"Claimed WR={ill['avg_claimed_win_rate_pct']:5.1f}% | Claimed PF={ill['avg_claimed_profit_factor']:5.2f} | "
            f"Claimed PnL=${ill['total_claimed_pnl_usd']:+8.2f}"
        )
        print(
            f"  [1] Argon V1 (Deployed Static, OOS)    : Trades={ast['total_trades']:4d} | "
            f"WR={ast['win_rate_pct']:5.1f}% | PF={ast['profit_factor']:5.2f} | Exp={ast['expectancy_r']:+.3f}R | "
            f"Sharpe={ast['sharpe_annualized']:+5.2f} | MaxDD=${ast['max_drawdown_usd']:6.2f} | NetPnL=${ast['total_net_pnl_usd']:+8.2f}"
        )
        print(
            f"  [2] Argon V1 (Walk-Forward Opt, OOS)   : Trades={awf['total_trades']:4d} | "
            f"WR={awf['win_rate_pct']:5.1f}% | PF={awf['profit_factor']:5.2f} | Exp={awf['expectancy_r']:+.3f}R | "
            f"Sharpe={awf['sharpe_annualized']:+5.2f} | MaxDD=${awf['max_drawdown_usd']:6.2f} | NetPnL=${awf['total_net_pnl_usd']:+8.2f}"
        )
        print(
            f"  [3] Carbon V2 (Purged Meta-WF, OOS)    : Trades={cbn['total_trades']:4d} | "
            f"WR={cbn['win_rate_pct']:5.1f}% | PF={cbn['profit_factor']:5.2f} | Exp={cbn['expectancy_r']:+.3f}R | "
            f"Sharpe={cbn['sharpe_annualized']:+5.2f} | MaxDD=${cbn['max_drawdown_usd']:6.2f} | NetPnL=${cbn['total_net_pnl_usd']:+8.2f}"
        )
        print("  Carbon V2 Archetypes (OOS):")
        for aname, am in r["carbon_archetypes"].items():
            print(
                f"    - {aname:22s}: Trades={am['total_trades']:3d} | WR={am['win_rate_pct']:5.1f}% | "
                f"PF={am['profit_factor']:5.2f} | Exp={am['expectancy_r']:+.3f}R | NetPnL=${am['total_net_pnl_usd']:+7.2f}"
            )


if __name__ == "__main__":
    main()
