"""
Krypton V5 — Dual-Engine Trend + CTA Multi-Speed TSMOM + Turtle Pyramiding Engine for BTCC Perpetuals
=====================================================================================================

Why Krypton V5 over V4/V3?
--------------------------
Krypton V4 replaced V3's sluggish 50/200 EMA with a responsive 20/80 EMA + slope + 24h CVD orderflow
confirmation + asymmetric causal Extreme Fear short gate (F&G <= 25 blocks fresh crypto shorts while
never capping bull-run longs).

Krypton V5 upgrades V4 with four strictly causal, cross-timeframe-verified profit multipliers
(validated on 2019-2025 development and 2025-12 -> 2026-11 OOS across both 4h and 1h, including 2x costs):
1. V-Bottom Regime-Flip Entry: catches explosive post-capitulation rebounds where price reclaims the
   20-bar channel before EMA20 crosses EMA80, then EMA20 flips bullish while price holds near highs.
2. CTA Multi-Speed TSMOM Sub-Engine (Carver / AHL): volatility-normalized 8/32 + 16/64 + 32/128 EMA
   risk-adjusted trend composite (+0.30 LONG / -0.55 SHORT cross, CVD & F&G gated) that enters strong
   multi-day trends cleanly and boosts confluence score when aligned with Donchian breakouts.
3. Causal Turtle Pyramiding (+2.0R -> +0.75R): adds a second 0.75R leg once an open winner reaches
   +2.0R and simultaneously ratchets the stop to breakeven (+0.10R), compounding fat right-tail runners
   without adding initial-entry whipsaw risk.
4. Dynamic Equity Compounding (2.0% of live equity per 1R) & Delayed-Read Execution Guidance:
   scales position size automatically as account equity grows while keeping liquidation safely distant,
   and publishes exact Maker Limit / Safe Entry Zone / Max Chase Ceiling / Time Estimations on every alert.

Everything is strictly causal: a signal on bar `i` uses data <= close[i]; orders fill at open[i+1]
(with spread/slippage), and trailing-stop / pyramid-breakeven levels computed on bar `k`'s close are
only active from bar `k+1`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MAJOR_SYMBOLS = {"BTCUSDT", "ETHUSDT", "XAUUSDT"}
TAKER_FEE = 0.00045


# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------
@dataclass
class KryptonParams:
    # Defaults = Krypton V5 4h preset (Dual-Engine: 20/10 Donchian + V-Bottom Flip + CTA Multi-Speed TSMOM
    # + responsive 20/80 EMA + slope + 24h CVD confirmation + F&G > 25 short gate + +2.0R -> +0.75R Turtle Pyramid).
    timeframe: str = "4h"
    entry_lookback: int = 20          # Donchian breakout channel (bars)
    exit_lookback: int = 10           # Donchian exit channel (bars)
    ema_fast: int = 20                # 3.3 days on 4h (0.4x of V3's 50/200 to eliminate multi-week short lag)
    ema_slow: int = 80                # 13.3 days on 4h
    atr_period: int = 20
    init_stop_atr: float = 2.0        # initial stop distance in ATR (= 1R)
    trail_atr: float = 8.0            # catastrophic chandelier trail in ATR (exit channel handles most exits)
    max_hold_bars: int = 180          # time stop (30 days on 4h)
    use_trend_filter: bool = True
    require_price_vs_emaf: bool = True
    require_emaf_slope: bool = True
    slope_lookback: int = 6           # 24h slope window on 4h (24 bars on 1h)
    allow_shorts: bool = True
    short_fng_min: Optional[float] = 25.0  # block fresh crypto shorts in Extreme Fear (F&G <= 25); never cap longs
    use_cvd_confirm: bool = True      # require 24h cumulative taker delta to confirm breakout direction
    cvd_window: int = 6               # 24h CVD window (6 bars on 4h, 24 bars on 1h)
    btc_filter: bool = False          # alts: require BTC slow trend to agree (hurt in development -> off)
    min_stop_pct: float = 0.004       # stop must be >= 0.4% of price (>> ~0.15% round-trip friction)
    # Krypton V5 Dual-Engine + Turtle Pyramiding parameters
    enable_state_flip: bool = True    # catch V-bottom EMA20>EMA80 flips while holding within 0.5 ATR of Donchian high
    enable_tsmom: bool = True         # CTA multi-speed volatility-normalized trend cross
    tsmom_long_thr: float = 0.30      # LONG cross threshold on multi-speed TSMOM composite
    tsmom_short_thr: float = -0.55    # stricter SHORT cross threshold (crypto rallies faster than sustained grinds down)
    pyramid_trigger_r: float = 2.0    # add second leg once open winner reaches +2.0R
    pyramid_add_scale: float = 0.75   # +0.75R second leg + move stop to breakeven (+0.10R)

    @classmethod
    def for_timeframe(cls, tf: str) -> "KryptonParams":
        if tf == "1h":
            # Calendar-equivalent horizons (x4 bars) of the 4h Krypton V5 engine.
            return cls(
                timeframe="1h",
                entry_lookback=160,
                exit_lookback=80,
                ema_fast=80,
                ema_slow=320,
                atr_period=80,
                init_stop_atr=3.0,
                trail_atr=12.0,
                max_hold_bars=720,
                slope_lookback=24,
                cvd_window=24,
            )
        return cls(timeframe=tf)

    @classmethod
    def v4_for_timeframe(cls, tf: str) -> "KryptonParams":
        """Krypton V4 baseline (responsive 20/80 EMA + CVD + F&G > 25, without V5 TSMOM/Flip/Pyramiding)."""
        base = cls.for_timeframe(tf)
        base.enable_state_flip = False
        base.enable_tsmom = False
        base.pyramid_trigger_r = 0.0
        base.pyramid_add_scale = 0.0
        return base

    @classmethod
    def v3_for_timeframe(cls, tf: str) -> "KryptonParams":
        """Legacy Krypton V3 baseline parameters preserved for head-to-head benchmarking."""
        if tf == "1h":
            return cls(
                timeframe="1h",
                entry_lookback=160,
                exit_lookback=80,
                ema_fast=200,
                ema_slow=800,
                atr_period=80,
                init_stop_atr=3.0,
                trail_atr=12.0,
                max_hold_bars=720,
                require_price_vs_emaf=False,
                require_emaf_slope=False,
                short_fng_min=None,
                use_cvd_confirm=False,
                enable_state_flip=False,
                enable_tsmom=False,
                pyramid_trigger_r=0.0,
                pyramid_add_scale=0.0,
            )
        return cls(
            timeframe=tf,
            ema_fast=50,
            ema_slow=200,
            require_price_vs_emaf=False,
            require_emaf_slope=False,
            short_fng_min=None,
            use_cvd_confirm=False,
            enable_state_flip=False,
            enable_tsmom=False,
            pyramid_trigger_r=0.0,
            pyramid_add_scale=0.0,
        )


@dataclass
class KryptonSignal:
    symbol: str
    bar_idx: int
    timestamp: str
    side: str
    stop_dist: float
    score: float          # breakout strength in ATR (used only as tie-break under the portfolio cap)
    archetype: str = "TREND_BREAKOUT"


# --------------------------------------------------------------------------------------
# Features (strictly causal)
# --------------------------------------------------------------------------------------
def _rma(x: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(x).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()


def compute_features(df: pd.DataFrame, p: KryptonParams, btc_df: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    prev_c = np.concatenate([[o[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr = np.maximum(_rma(tr, p.atr_period), 1e-12)

    cs = pd.Series(c)
    ema_f = cs.ewm(span=p.ema_fast, adjust=False).mean().to_numpy()
    ema_s = cs.ewm(span=p.ema_slow, adjust=False).mean().to_numpy()

    hs, ls = pd.Series(h), pd.Series(l)
    # Prior-N extremes EXCLUDING the current bar (shift(1)) -> breakout = close beyond yesterday's channel
    don_hi = hs.rolling(p.entry_lookback).max().shift(1).to_numpy()
    don_lo = ls.rolling(p.entry_lookback).min().shift(1).to_numpy()
    ex_hi = hs.rolling(p.exit_lookback).max().shift(1).to_numpy()
    ex_lo = ls.rolling(p.exit_lookback).min().shift(1).to_numpy()

    feats = {
        "open": o, "high": h, "low": l, "close": c, "atr": atr,
        "ema_f": ema_f, "ema_s": ema_s,
        "don_hi": don_hi, "don_lo": don_lo, "ex_hi": ex_hi, "ex_lo": ex_lo,
    }

    # Strictly causal multi-speed CTA TSMOM (Carver / AHL volatility-normalized trend composite)
    mult = 4 if p.timeframe == "1h" else 1
    e8 = cs.ewm(span=8 * mult, adjust=False).mean().to_numpy()
    e16 = cs.ewm(span=16 * mult, adjust=False).mean().to_numpy()
    e32 = cs.ewm(span=32 * mult, adjust=False).mean().to_numpy()
    e64 = cs.ewm(span=64 * mult, adjust=False).mean().to_numpy()
    e128 = cs.ewm(span=128 * mult, adjust=False).mean().to_numpy()
    feats["tsmom"] = (
        0.35 * np.clip((e8 - e32) / np.maximum(2.0 * atr, 1e-12), -2.0, 2.0)
        + 0.35 * np.clip((e16 - e64) / np.maximum(3.0 * atr, 1e-12), -2.0, 2.0)
        + 0.30 * np.clip((e32 - e128) / np.maximum(4.5 * atr, 1e-12), -2.0, 2.0)
    )

    # Strictly causal 24h cumulative taker volume delta (CVD) for orderflow breakout confirmation
    has_of_attr = df.attrs.get("has_true_orderflow", None)
    if "volume" in df.columns and "taker_buy_vol" in df.columns:
        vol_arr = df["volume"].to_numpy(float)
        tbv_arr = df["taker_buy_vol"].to_numpy(float)
        if has_of_attr is not None:
            has_true_of = bool(has_of_attr)
        else:
            nz = vol_arr > 0
            has_true_of = bool(
                np.any(nz)
                and float(np.nanmean(np.abs(tbv_arr[nz] - 0.5 * vol_arr[nz]) / np.maximum(vol_arr[nz], 1e-12))) > 1e-4
            )
        delta_arr = 2.0 * tbv_arr - vol_arr
        cvd_w = max(1, int(p.cvd_window))
        cvd_num = pd.Series(delta_arr).rolling(cvd_w, min_periods=1).sum().to_numpy()
        cvd_den = np.maximum(pd.Series(vol_arr).rolling(cvd_w, min_periods=1).sum().to_numpy(), 1e-9)
        feats["cvd"] = np.clip(cvd_num / cvd_den, -1.0, 1.0)
        feats["has_true_orderflow"] = has_true_of
    else:
        feats["cvd"] = np.zeros(len(df), dtype=float)
        feats["has_true_orderflow"] = False

    # Strictly causal 1-day-lagged Crypto Fear & Greed series if present (research / backtests)
    if "fng_causal" in df.columns:
        feats["fng_causal"] = df["fng_causal"].to_numpy(float)
    else:
        feats["fng_causal"] = np.full(len(df), np.nan, dtype=float)

    if btc_df is not None and len(btc_df):
        b = btc_df[["timestamp", "close"]].sort_values("timestamp").copy()
        bc = b["close"].astype(float)
        b["btc_trend"] = np.sign(bc.ewm(span=p.ema_fast, adjust=False).mean() - bc.ewm(span=p.ema_slow, adjust=False).mean())
        m = pd.merge_asof(df[["timestamp"]].sort_values("timestamp"), b[["timestamp", "btc_trend"]],
                          on="timestamp", direction="backward").sort_index()
        feats["btc_trend"] = m["btc_trend"].fillna(0.0).to_numpy()
    else:
        feats["btc_trend"] = np.sign(ema_f - ema_s)

    ts = pd.to_datetime(df["timestamp"], utc=True)
    feats["funding_bar"] = ((ts.dt.minute == 0) & (ts.dt.hour % 8 == 0)).to_numpy()
    feats["funding_rate"] = df["funding_rate"].to_numpy(float) if "funding_rate" in df.columns else np.full(len(df), 1e-4)
    feats["ts_str"] = ts.astype(str).to_numpy()
    feats["ts_ns"] = ts.dt.as_unit("ns").astype("int64").to_numpy()  # true nanoseconds (pandas 3 defaults to us)
    return feats


# --------------------------------------------------------------------------------------
# Signal generation
# --------------------------------------------------------------------------------------
def _is_crypto_symbol(symbol: str) -> bool:
    su = symbol.upper()
    return not any(
        k in su
        for k in ("XAU", "XAG", "PAXG", "GOLD", "SILVER", "GOOGL", "NVDA", "AAPL", "SPACEX", "SPCX", "TSLA")
    )


def generate_signals(f: Dict[str, np.ndarray], symbol: str, p: KryptonParams,
                     start_idx: int = 0, end_idx: Optional[int] = None,
                     live_fng: Optional[float] = None) -> Dict[int, KryptonSignal]:
    c, atr = f["close"], f["atr"]
    ef, es = f["ema_f"], f["ema_s"]
    cvd = f.get("cvd")
    has_of = bool(f.get("has_true_orderflow", False))
    fng_arr = f.get("fng_causal")
    tsmom = f.get("tsmom")
    n = len(c)
    slope_lag = max(1, int(p.slope_lookback))
    warm = max(p.ema_slow, p.entry_lookback + 1, p.atr_period * 3, slope_lag + 1)
    # Bars in [start_idx, end_idx) are evaluated; end_idx=None includes the final (closed) bar for live use.
    stop_i = n if end_idx is None else min(n, end_idx)
    out: Dict[int, KryptonSignal] = {}
    is_btc = symbol.upper().startswith("BTC")
    is_crypto = _is_crypto_symbol(symbol)

    def _trend_up(idx: int) -> bool:
        if not p.use_trend_filter:
            return True
        if idx < slope_lag:
            return False
        return bool(
            ef[idx] > es[idx]
            and c[idx] > es[idx]
            and (not p.require_price_vs_emaf or c[idx] > ef[idx])
            and (not p.require_emaf_slope or ef[idx] > ef[idx - slope_lag])
        )

    def _trend_dn(idx: int) -> bool:
        if not p.use_trend_filter:
            return True
        if idx < slope_lag:
            return False
        return bool(
            ef[idx] < es[idx]
            and c[idx] < es[idx]
            and (not p.require_price_vs_emaf or c[idx] < ef[idx])
            and (not p.require_emaf_slope or ef[idx] < ef[idx - slope_lag])
        )

    for i in range(max(start_idx, warm), stop_i):
        dh, dl = f["don_hi"][i], f["don_lo"][i]
        if not (np.isfinite(dh) and np.isfinite(dl)):
            continue
        pdh, pdl = f["don_hi"][i - 1], f["don_lo"][i - 1]
        a = atr[i]
        up_now = _trend_up(i)
        dn_now = _trend_dn(i)
        cvd_ok_long = (not p.use_cvd_confirm) or (not has_of) or (cvd is None) or (cvd[i] > 0.0)
        cvd_ok_short = (not p.use_cvd_confirm) or (not has_of) or (cvd is None) or (cvd[i] < 0.0)
        btc_ok_long = (not p.btc_filter) or is_btc or f["btc_trend"][i] > 0
        btc_ok_short = (not p.btc_filter) or is_btc or f["btc_trend"][i] < 0

        # Asymmetric causal sentiment gate: never fire fresh crypto shorts into Extreme Fear capitulation
        fng_val = (
            float(fng_arr[i])
            if (fng_arr is not None and i < len(fng_arr) and np.isfinite(fng_arr[i]))
            else (float(live_fng) if live_fng is not None else np.nan)
        )
        fng_ok_short = (
            (not is_crypto)
            or (p.short_fng_min is None)
            or (not np.isfinite(fng_val))
            or (fng_val > float(p.short_fng_min))
        )

        # Sub-engine 1: Classic Krypton Donchian Breakout
        don_long = bool(c[i] > dh and c[i - 1] <= pdh and up_now and cvd_ok_long and btc_ok_long)
        don_short = bool(
            p.allow_shorts and c[i] < dl and c[i - 1] >= pdl and dn_now and cvd_ok_short and fng_ok_short and btc_ok_short
        )

        # Sub-engine 2: V-Bottom Regime Flip (EMA20 flips above EMA80 while holding near Donchian highs)
        flip_long = bool(
            p.enable_state_flip
            and up_now
            and (not _trend_up(i - 1))
            and c[i] >= dh - 0.5 * a
            and cvd_ok_long
            and btc_ok_long
        )

        # Sub-engine 3: CTA Multi-Speed TSMOM Cross
        ts_ok = bool(p.enable_tsmom and tsmom is not None and np.isfinite(tsmom[i]) and np.isfinite(tsmom[i - 1]))
        ts_long = bool(
            ts_ok
            and tsmom[i] > p.tsmom_long_thr
            and tsmom[i - 1] <= p.tsmom_long_thr
            and cvd_ok_long
            and btc_ok_long
        )
        ts_short = bool(
            ts_ok
            and p.allow_shorts
            and tsmom[i] < p.tsmom_short_thr
            and tsmom[i - 1] >= p.tsmom_short_thr
            and cvd_ok_short
            and fng_ok_short
            and btc_ok_short
        )

        side = ""
        archetype = "TREND_BREAKOUT"
        if don_long or flip_long or ts_long:
            side = "LONG"
            if (don_long or flip_long) and ts_long:
                archetype = "DUAL_ENGINE_CONFLUENCE"
            elif don_long:
                archetype = "TREND_BREAKOUT"
            elif flip_long:
                archetype = "V_BOTTOM_REGIME_FLIP"
            else:
                archetype = "CTA_TSMOM_TREND"
        elif don_short or ts_short:
            side = "SHORT"
            if don_short and ts_short:
                archetype = "DUAL_ENGINE_CONFLUENCE"
            elif don_short:
                archetype = "TREND_BREAKOUT"
            else:
                archetype = "CTA_TSMOM_TREND"

        if not side:
            continue
        stop = max(p.init_stop_atr * a, p.min_stop_pct * c[i])
        brk_score = max(0.0, ((c[i] - dh) if side == "LONG" else (dl - c[i])) / a)
        ts_score = abs(float(tsmom[i])) if (tsmom is not None and np.isfinite(tsmom[i])) else 0.0
        score = max(brk_score, ts_score) + (1.5 if archetype == "DUAL_ENGINE_CONFLUENCE" else 0.0)
        out[i] = KryptonSignal(symbol, i, str(f["ts_str"][i]), side, float(stop), float(score), archetype=archetype)
    return out


# --------------------------------------------------------------------------------------
# Execution simulator (friction model identical to benchmark_argon_vs_carbon.simulate_trade_honest)
# --------------------------------------------------------------------------------------
def simulate_trade_fast(
    f: Dict[str, np.ndarray],
    symbol: str,
    signal_idx: int,
    side: str,
    stop_dist: float,
    max_hold_bars: int,
    *,
    mode: str = "KRYPTON",            # "KRYPTON" (trail from entry, no TP) | "CARBON_V2" (verification)
    trail_atr_mult: float = 4.0,
    exit_channel: bool = True,
    tp1_dist: float = 0.0,
    tp2_dist: float = 0.0,
    partial_frac: float = 0.0,
    cvd6: Optional[np.ndarray] = None,
    atr_override: Optional[np.ndarray] = None,
    margin_usd: float = 100.0,
    base_risk_usd: float = 25.0,
    risk_scale: float = 1.0,
    max_btcc_leverage: int = 100,
    cost_mult: float = 1.0,           # stress-test multiplier on fees + spread + slippage (1.0 = benchmark)
) -> Optional[Dict[str, Any]]:
    fee = TAKER_FEE * cost_mult
    o, h, l, c = f["open"], f["high"], f["low"], f["close"]
    atr = atr_override if atr_override is not None else f["atr"]
    fr, fbar = f["funding_rate"], f["funding_bar"]
    n = len(c)
    entry_idx = signal_idx + 1
    if entry_idx >= n or stop_dist <= 0:
        return None
    raw_open = float(o[entry_idx])
    if raw_open <= 0:
        return None
    is_long = side == "LONG"
    slip = (0.00020 if symbol in MAJOR_SYMBOLS else 0.00035) * cost_mult
    entry = raw_open * (1.0 + slip) if is_long else raw_open * (1.0 - slip)
    stop_pct = stop_dist / entry
    if stop_pct <= 0.0008:
        return None

    max_lev = min(100, max_btcc_leverage)
    lev = max(3, min(max_lev, int(round((base_risk_usd * risk_scale) / max(margin_usd * stop_pct, 1e-9)))))
    rem = margin_usd
    if is_long:
        sl = entry - stop_dist
        tp1, tp2 = entry + tp1_dist, entry + tp2_dist
        liq = entry * (1.0 - 0.90 / lev)
    else:
        sl = entry + stop_dist
        tp1, tp2 = entry - tp1_dist, entry - tp2_dist
        liq = entry * (1.0 + 0.90 / lev)

    pnl = -(margin_usd * lev * fee)
    use_tp = mode == "CARBON_V2"
    tp1_hit, tp1_bar = False, -1
    peak = entry if mode == "CARBON_V2" else (h[entry_idx] if is_long else l[entry_idx])
    exit_idx, exit_px, reason = entry_idx, entry, "TIME_STOP"
    last = min(n - 1, entry_idx + max_hold_bars - 1)
    mfe_r = 0.0

    def close_leg(px: float) -> float:
        ret = (px - entry) / entry if is_long else (entry - px) / entry
        return max(-rem, rem * lev * ret) - rem * lev * fee

    for k in range(entry_idx, last + 1):
        hi, lo, cl = float(h[k]), float(l[k]), float(c[k])
        a = max(float(atr[k]), 1e-9)
        if k > entry_idx and fbar[k]:
            pnl += (-1.0 if is_long else 1.0) * rem * lev * float(fr[k])
        if (is_long and lo <= liq) or ((not is_long) and hi >= liq):
            pnl -= rem
            rem = 0.0
            exit_idx, exit_px, reason = k, liq, "LIQUIDATED"
            break
        if (lo <= sl) if is_long else (hi >= sl):
            tr_frac = (hi - lo) / max(cl, 1e-9)
            sslip = slip + min(0.0008, 0.04 * tr_frac) * cost_mult
            fill = sl * (1.0 - sslip) if is_long else sl * (1.0 + sslip)
            ret = (fill - entry) / entry if is_long else (entry - fill) / entry
            pnl += max(-rem, rem * lev * ret) - rem * lev * fee
            rem = 0.0
            exit_idx, exit_px = k, fill
            if mode == "CARBON_V2":
                reason = "TRAIL_OR_BE_STOP" if tp1_hit else "STOP_LOSS"
            else:
                reason = "TRAIL_STOP" if ((fill > entry) if is_long else (fill < entry)) else "STOP_LOSS"
            break
        mfe_r = max(mfe_r, ((hi - entry) if is_long else (entry - lo)) / stop_dist)

        if use_tp:
            if (not tp1_hit) and ((hi >= tp1) if is_long else (lo <= tp1)):
                cm = rem * partial_frac
                ret1 = (tp1 - entry) / entry if is_long else (entry - tp1) / entry
                pnl += cm * lev * ret1 - cm * lev * fee
                rem -= cm
                tp1_hit, tp1_bar = True, k
                peak = hi if is_long else lo
                sl = (entry + 0.12 * stop_dist) if is_long else (entry - 0.12 * stop_dist)
            if ((hi >= tp2) if is_long else (lo <= tp2)) and rem > 0:
                ret2 = (tp2 - entry) / entry if is_long else (entry - tp2) / entry
                pnl += rem * lev * ret2 - rem * lev * fee
                rem = 0.0
                exit_idx, exit_px, reason = k, tp2, "TP2_TARGET"
                break
            if tp1_hit and k > tp1_bar:
                if is_long:
                    peak = max(peak, hi)
                    sl = max(sl, peak - trail_atr_mult * a)
                    flow = float(cvd6[k]) if cvd6 is not None else 0.0
                else:
                    peak = min(peak, lo)
                    sl = min(sl, peak + trail_atr_mult * a)
                    flow = -float(cvd6[k]) if cvd6 is not None else 0.0
                if (k - tp1_bar) >= 3 and flow <= -0.22:
                    fill = cl * (1.0 - slip) if is_long else cl * (1.0 + slip)
                    pnl += close_leg(fill)
                    rem = 0.0
                    exit_idx, exit_px, reason = k, fill, "ORDERFLOW_DECAY_EXIT"
                    break
        else:
            # KRYPTON: Chandelier trail from entry; level computed at close k, active from bar k+1
            if is_long:
                peak = max(peak, hi)
                sl = max(sl, peak - trail_atr_mult * a)
            else:
                peak = min(peak, lo)
                sl = min(sl, peak + trail_atr_mult * a)
            if exit_channel and k > entry_idx:
                if (is_long and cl < f["ex_lo"][k]) or ((not is_long) and cl > f["ex_hi"][k]):
                    fill = cl * (1.0 - slip) if is_long else cl * (1.0 + slip)
                    pnl += close_leg(fill)
                    rem = 0.0
                    exit_idx, exit_px, reason = k, fill, "CHANNEL_EXIT"
                    break

        if k == last and rem > 0:
            fill = cl * (1.0 - slip) if is_long else cl * (1.0 + slip)
            pnl += close_leg(fill)
            rem = 0.0
            exit_idx, exit_px, reason = k, fill, "TIME_STOP"
            break

    one_r = max(margin_usd * lev * stop_pct, 1.0)
    return {
        "symbol": symbol,
        "side": side,
        "leverage": lev,
        "signal_idx": signal_idx,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "entry_time": str(f["ts_str"][entry_idx]),
        "exit_time": str(f["ts_str"][exit_idx]),
        "entry_ns": int(f["ts_ns"][entry_idx]),
        "exit_ns": int(f["ts_ns"][exit_idx]),
        "entry_price": round(entry, 6),
        "exit_price": round(float(exit_px), 6),
        "net_pnl_usd": round(float(pnl), 4),
        "roe_pct": round(float(pnl / margin_usd * 100.0), 3),
        "net_r": round(float(pnl / one_r), 4),
        "mfe_r": round(float(mfe_r), 3),
        "risk_scale": round(float(risk_scale), 3),
        "exit_reason": reason,
        "bars_held": exit_idx - entry_idx + 1,
    }


def simulate_trade_v5(
    f: Dict[str, np.ndarray],
    symbol: str,
    signal_idx: int,
    side: str,
    stop_dist: float,
    max_hold_bars: int,
    *,
    trail_atr_mult: float = 8.0,
    exit_channel: bool = True,
    atr_override: Optional[np.ndarray] = None,
    margin_usd: float = 100.0,
    base_risk_usd: float = 25.0,
    risk_scale: float = 1.0,
    max_btcc_leverage: int = 100,
    cost_mult: float = 1.0,
    pyramid_trigger_r: float = 2.0,
    pyramid_add_scale: float = 0.75,
) -> Optional[Dict[str, Any]]:
    """
    Strictly causal Krypton V5 simulator with Turtle Pyramiding (+2.0R -> +0.75R):
      * Leg 1 fills at open[signal_idx + 1] with identical friction/funding/liquidation rules as `simulate_trade_fast`.
      * When bar `k` closes at >= +pyramid_trigger_r * R (evaluated strictly on close[k]), Leg 2 (+pyramid_add_scale
        of Leg 1 margin at the same leverage) is queued to fill at open[k + 1] + spread + taker fee, and the shared
        stop-loss ratchets to at least breakeven (entry1 +/- 0.10R) active from bar k + 1.
      * If open[k + 1] gaps back past the stop, Leg 2 is NOT added. Both active legs accrue funding and are checked
        for isolated liquidation before stop fills.
    """
    if pyramid_trigger_r <= 0.0 or pyramid_add_scale <= 0.0:
        return simulate_trade_fast(
            f, symbol, signal_idx, side, stop_dist, max_hold_bars,
            trail_atr_mult=trail_atr_mult, exit_channel=exit_channel, atr_override=atr_override,
            margin_usd=margin_usd, base_risk_usd=base_risk_usd, risk_scale=risk_scale,
            max_btcc_leverage=max_btcc_leverage, cost_mult=cost_mult,
        )

    fee = TAKER_FEE * cost_mult
    o, h, l, c = f["open"], f["high"], f["low"], f["close"]
    atr = atr_override if atr_override is not None else f["atr"]
    fr, fbar = f["funding_rate"], f["funding_bar"]
    n = len(c)
    entry_idx = signal_idx + 1
    if entry_idx >= n or stop_dist <= 0:
        return None
    raw_open = float(o[entry_idx])
    if raw_open <= 0:
        return None
    is_long = side == "LONG"
    slip = (0.00020 if symbol in MAJOR_SYMBOLS else 0.00035) * cost_mult
    entry1 = raw_open * (1.0 + slip) if is_long else raw_open * (1.0 - slip)
    stop_pct = stop_dist / entry1
    if stop_pct <= 0.0008:
        return None

    max_lev = min(100, max_btcc_leverage)
    lev = max(3, min(max_lev, int(round((base_risk_usd * risk_scale) / max(margin_usd * stop_pct, 1e-9)))))
    rem1 = margin_usd
    sl = entry1 - stop_dist if is_long else entry1 + stop_dist
    liq1 = entry1 * (1.0 - 0.90 / lev) if is_long else entry1 * (1.0 + 0.90 / lev)

    # Optional Leg 2 state
    pyr_pending = False
    pyr_active = False
    entry2 = 0.0
    rem2 = 0.0
    liq2 = 0.0
    pyr_bar = -1
    r2_bars_to_hit = -1

    pnl = -(rem1 * lev * fee)
    peak = h[entry_idx] if is_long else l[entry_idx]
    exit_idx, exit_px, reason = entry_idx, entry1, "TIME_STOP"
    last = min(n - 1, entry_idx + max_hold_bars - 1)
    mfe_r = 0.0
    sgn = 1.0 if is_long else -1.0

    def _leg_close(rem_m: float, ent: float, px: float) -> float:
        if rem_m <= 0:
            return 0.0
        ret = (px - ent) / ent if is_long else (ent - px) / ent
        return max(-rem_m, rem_m * lev * ret) - rem_m * lev * fee

    for k in range(entry_idx, last + 1):
        op, hi, lo, cl = float(o[k]), float(h[k]), float(l[k]), float(c[k])
        a = max(float(atr[k]), 1e-9)

        # Execute queued Pyramid Leg 2 at open[k] (strictly causal after close[k-1] >= +2R)
        if pyr_pending and not pyr_active:
            pyr_pending = False
            be_sl = entry1 + sgn * 0.10 * stop_dist
            sl = max(sl, be_sl) if is_long else min(sl, be_sl)
            cand_e2 = op * (1.0 + slip) if is_long else op * (1.0 - slip)
            if (is_long and op > sl and cand_e2 > sl) or ((not is_long) and op < sl and cand_e2 < sl):
                entry2 = cand_e2
                rem2 = margin_usd * pyramid_add_scale
                liq2 = entry2 * (1.0 - 0.90 / lev) if is_long else entry2 * (1.0 + 0.90 / lev)
                pnl -= rem2 * lev * fee
                pyr_active = True
                pyr_bar = k

        # Funding on open legs
        if k > entry_idx and fbar[k]:
            pnl += (-1.0 if is_long else 1.0) * rem1 * lev * float(fr[k])
        if pyr_active and k > pyr_bar and fbar[k]:
            pnl += (-1.0 if is_long else 1.0) * rem2 * lev * float(fr[k])

        # Isolated liquidation checks (checked conservatively before stop-loss)
        if pyr_active and ((is_long and lo <= liq2) or ((not is_long) and hi >= liq2)):
            pnl -= rem2
            rem2 = 0.0
            pyr_active = False
        if (is_long and lo <= liq1) or ((not is_long) and hi >= liq1):
            pnl -= rem1
            rem1 = 0.0
            if pyr_active and rem2 > 0:
                pnl += _leg_close(rem2, entry2, liq1)
                rem2 = 0.0
                pyr_active = False
            exit_idx, exit_px, reason = k, liq1, "LIQUIDATED"
            break

        # Shared Stop-Loss / Trail / Breakeven Stop check
        if (lo <= sl) if is_long else (hi >= sl):
            tr_frac = (hi - lo) / max(cl, 1e-9)
            sslip = slip + min(0.0008, 0.04 * tr_frac) * cost_mult
            base_stop = min(op, sl) if is_long else max(op, sl)
            fill = base_stop * (1.0 - sslip) if is_long else base_stop * (1.0 + sslip)
            pnl += _leg_close(rem1, entry1, fill)
            if pyr_active and rem2 > 0:
                pnl += _leg_close(rem2, entry2, fill)
                rem2 = 0.0
                pyr_active = False
            rem1 = 0.0
            exit_idx, exit_px = k, fill
            reason = "TRAIL_STOP" if ((fill > entry1) if is_long else (fill < entry1)) else "STOP_LOSS"
            break

        fav_r = ((hi - entry1) if is_long else (entry1 - lo)) / stop_dist
        mfe_r = max(mfe_r, fav_r)
        if r2_bars_to_hit < 0 and fav_r >= pyramid_trigger_r:
            r2_bars_to_hit = k - entry_idx + 1

        # Chandelier trail computed at close[k], active from k+1
        if is_long:
            peak = max(peak, hi)
            sl = max(sl, peak - trail_atr_mult * a)
        else:
            peak = min(peak, lo)
            sl = min(sl, peak + trail_atr_mult * a)

        # Exit-channel close
        if exit_channel and k > entry_idx:
            if (is_long and cl < f["ex_lo"][k]) or ((not is_long) and cl > f["ex_hi"][k]):
                fill = cl * (1.0 - slip) if is_long else cl * (1.0 + slip)
                pnl += _leg_close(rem1, entry1, fill)
                if pyr_active and rem2 > 0:
                    pnl += _leg_close(rem2, entry2, fill)
                    rem2 = 0.0
                    pyr_active = False
                rem1 = 0.0
                exit_idx, exit_px, reason = k, fill, "CHANNEL_EXIT"
                break

        # Queue causal Pyramid Add for next bar's open once close[k] reaches +pyramid_trigger_r * R
        if (not pyr_active) and (not pyr_pending) and k < last:
            close_r = ((cl - entry1) if is_long else (entry1 - cl)) / stop_dist
            if close_r >= pyramid_trigger_r:
                pyr_pending = True

        if k == last and rem1 > 0:
            fill = cl * (1.0 - slip) if is_long else cl * (1.0 + slip)
            pnl += _leg_close(rem1, entry1, fill)
            if pyr_active and rem2 > 0:
                pnl += _leg_close(rem2, entry2, fill)
                rem2 = 0.0
                pyr_active = False
            rem1 = 0.0
            exit_idx, exit_px, reason = k, fill, "TIME_STOP"
            break

    one_r = max(margin_usd * lev * stop_pct, 1.0)
    return {
        "symbol": symbol,
        "side": side,
        "leverage": lev,
        "signal_idx": signal_idx,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "entry_time": str(f["ts_str"][entry_idx]),
        "exit_time": str(f["ts_str"][exit_idx]),
        "entry_ns": int(f["ts_ns"][entry_idx]),
        "exit_ns": int(f["ts_ns"][exit_idx]),
        "entry_price": round(entry1, 6),
        "exit_price": round(float(exit_px), 6),
        "net_pnl_usd": round(float(pnl), 4),
        "roe_pct": round(float(pnl / margin_usd * 100.0), 3),
        "net_r": round(float(pnl / one_r), 4),
        "mfe_r": round(float(mfe_r), 3),
        "risk_scale": round(float(risk_scale), 3),
        "pyramided": bool(pyr_bar >= 0),
        "r2_bars_to_hit": int(r2_bars_to_hit),
        "exit_reason": reason,
        "bars_held": exit_idx - entry_idx + 1,
    }


# --------------------------------------------------------------------------------------
# Per-symbol runner and timestamp-based portfolio cap
# --------------------------------------------------------------------------------------
def run_symbol(f: Dict[str, np.ndarray], symbol: str, p: KryptonParams, start_idx: int, end_idx: int,
               max_btcc_leverage: int = 100, cost_mult: float = 1.0) -> List[Dict[str, Any]]:
    """Non-overlapping per-symbol execution of Krypton signals whose signal bar lies in [start_idx, end_idx)."""
    sigs = generate_signals(f, symbol, p, start_idx=start_idx, end_idx=end_idx)
    trades: List[Dict[str, Any]] = []
    busy = -1
    use_pyr = (p.pyramid_trigger_r > 0.0 and p.pyramid_add_scale > 0.0)
    for i in sorted(sigs):
        if i <= busy:
            continue
        s = sigs[i]
        if use_pyr:
            tr = simulate_trade_v5(
                f, symbol, i, s.side, s.stop_dist, p.max_hold_bars,
                trail_atr_mult=p.trail_atr, max_btcc_leverage=max_btcc_leverage,
                cost_mult=cost_mult,
                pyramid_trigger_r=p.pyramid_trigger_r,
                pyramid_add_scale=p.pyramid_add_scale,
            )
        else:
            tr = simulate_trade_fast(f, symbol, i, s.side, s.stop_dist, p.max_hold_bars,
                                     trail_atr_mult=p.trail_atr, max_btcc_leverage=max_btcc_leverage,
                                     cost_mult=cost_mult)
        if tr is None:
            continue
        tr["archetype"] = s.archetype
        tr["meta_pred_r"] = round(s.score, 3)
        trades.append(tr)
        busy = tr["exit_idx"]
    return trades


def portfolio_cap_by_time(
    trades: List[Dict[str, Any]],
    max_concurrent: int = 3,
    max_same_side: int = 2,
    max_long: Optional[int] = None,
    max_short: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Timestamp-based concurrency cap supporting both legacy symmetric and live asymmetric (3L/1S) limits."""
    acc, active = [], []
    side_lim = {
        "LONG": max_same_side if max_long is None else int(max_long),
        "SHORT": max_same_side if max_short is None else int(max_short),
    }
    for tr in sorted(trades, key=lambda t: (t["entry_ns"], -float(t.get("meta_pred_r", 0.0)), t["symbol"])):
        e = tr["entry_ns"]
        active = [a for a in active if a["exit_ns"] >= e]
        if any(a["symbol"] == tr["symbol"] for a in active) or len(active) >= max_concurrent:
            continue
        lim = side_lim.get(str(tr["side"]).upper(), max_same_side)
        if sum(1 for a in active if a["side"] == tr["side"]) >= lim:
            continue
        acc.append(tr)
        active.append(tr)
    return acc


# ======================================================================================
# Live bridge (scanner / cloud runner / dashboard / Telegram)
# ======================================================================================
_TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
               "1d": 86400, "1w": 604800, "1M": 2592000}


def live_params_for(tf: str) -> KryptonParams:
    """Validated presets exist for 4h and 1h; other timeframes reuse the closest preset by bar count."""
    if tf in ("1h", "4h"):
        return KryptonParams.for_timeframe(tf)
    base = KryptonParams.for_timeframe("1h" if tf in ("1m", "5m", "15m", "30m") else "4h")
    base.timeframe = tf
    return base


def closed_bars_only(df: pd.DataFrame, tf: str, now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Drop the still-forming candle so live signals use exactly the same closed-bar data as the backtest."""
    if df is None or len(df) == 0 or "timestamp" not in df.columns:
        return df
    now = now or pd.Timestamp.now(tz="UTC")
    secs = _TF_SECONDS.get(tf, 14400)
    last_open = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
    if last_open + pd.Timedelta(seconds=secs) > now:
        return df.iloc[:-1].reset_index(drop=True)
    return df.reset_index(drop=True)


@dataclass
class KryptonLiveScanResult:
    symbol: str
    timeframe: str
    active_signal: Any                    # Optional[detector.ReversalSetupSignal]
    signal_bars_ago: Optional[int]
    active_candidate: Optional[Any]
    pending_order_plan: Dict[str, Any]
    telemetry: Dict[str, Any]
    oos_benchmark_stats: Dict[str, Any]


@dataclass
class KryptonCandidateInfo:
    """Minimal candidate object consumed by execution_engine.execute_reversal_entry / main.py."""
    archetype: str = "KRYPTON_TREND_BREAKOUT"
    partial_frac: float = 0.25            # paper engine floor; Krypton itself takes no fixed partial
    trail_atr_mult: float = 8.0
    meta_pred_r: float = 0.0
    meta_win_prob: float = 0.3
    risk_scale: float = 1.0


class KryptonLiveBridge:
    """
    Drop-in replacement for `CarbonLiveBridge` (same public surface) that runs Krypton V5 live:
      * `evaluate_symbol_live` -> active Dual-Engine signal (Donchian Breakout / V-Bottom Regime Flip /
        CTA Multi-Speed TSMOM / Confluence), extreme-fear short pause, or breakout *watch* plan,
        complete with Dynamic Equity Compounding sizing + delayed-read Price Ladder & Time Estimations.
      * `check_position_exit` -> Krypton exit & scaling management for tracked positions
        (initial stop, +2.0R Turtle Pyramid Add + breakeven SL, chandelier trail ratchet, Donchian
        exit-channel close, time stop).
    """

    engine_name = "KRYPTON_V5"
    display_name = "Krypton V5"

    def __init__(self, risk_cfg: Any, state_dir: Optional[Any] = None):
        import json
        from pathlib import Path

        self.risk_cfg = risk_cfg
        self.state_dir = Path(state_dir) if state_dir else (Path(__file__).resolve().parent / "state")
        self.bench_path = self.state_dir / "benchmark_krypton_results.json"
        self.models: Dict[str, Any] = {}
        self.oos_stats: Dict[str, Any] = {}
        if self.bench_path.exists():
            try:
                self.oos_stats = json.loads(self.bench_path.read_text())
            except Exception:
                self.oos_stats = {}

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def required_bars(tf: str) -> int:
        return 1500 if tf in ("1m", "5m", "15m", "30m", "1h") else 1000

    @staticmethod
    def _price_precision(price: float) -> int:
        ap = abs(price)
        if ap < 0.001:
            return 8
        if ap < 1.0:
            return 5
        if ap < 100.0:
            return 4
        return 2

    def _tf_bucket(self, tf: str) -> str:
        return "1h" if tf in ("1m", "5m", "15m", "30m", "1h") else "4h"

    def get_oos_asset_summary(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        blk = self.oos_stats.get(self._tf_bucket(timeframe), {}) or {}
        port = blk.get("krypton_v5_oos") or blk.get("krypton_v4_oos") or blk.get("krypton_v3_oos") or {}
        sym = (blk.get("krypton_per_symbol") or {}).get(symbol.upper())
        has_asset_oos = bool(sym and int(sym.get("total_trades", 0)) > 0)
        base = {
            "portfolio_pf": float(port.get("profit_factor", 0.0)),
            "absorption_pf": float(port.get("profit_factor", 0.0)),
            "expectancy_r": float(port.get("expectancy_r", 0.0)) if has_asset_oos else 0.0,
            "validated": has_asset_oos,
        }
        if has_asset_oos and sym is not None:
            base.update(
                {
                    "scope": "ASSET_OOS",
                    "win_rate_pct": float(sym.get("win_rate_pct", 0.0)),
                    "profit_factor": float(sym.get("profit_factor", 0.0)),
                    "total_net_pnl_usd": float(sym.get("total_net_pnl_usd", 0.0)),
                    "max_drawdown_usd": float(sym.get("max_drawdown_usd", 0.0)),
                    "total_trades": int(sym.get("total_trades", 0)),
                }
            )
        else:
            base.update(
                {
                    "scope": "UNVALIDATED",
                    "win_rate_pct": 0.0,
                    "profit_factor": 0.0,
                    "total_net_pnl_usd": 0.0,
                    "max_drawdown_usd": 0.0,
                    "total_trades": 0,
                }
            )
        return base

    def compute_position_sizing(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_dist_pct: float,
        account_balance_usd: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Dynamic Equity Compounding + Equal-1R Risk-Parity isolated-margin sizing:
        - Risks `compound_risk_pct` (default 2.0%) of live account equity per 1R trade (floored at base micro-risk
          `base_margin * target_sl_margin_risk_pct / 100`), automatically compounding winners as equity grows.
        - Respects BTCC min-lot notionals WITHOUT forcing near-liquidation leverage, keeping isolated liquidation
          safely distant from the stop-loss, and pre-computes the +2.0R Turtle Pyramid add-on size (+0.75R).
        """
        from config import DEFAULT_ASSETS

        profile = DEFAULT_ASSETS.get(symbol.upper())
        asset_max = profile.max_btcc_leverage if profile else self.risk_cfg.max_leverage
        base_margin = max(1.0, float(self.risk_cfg.margin_per_trade_usd))
        equity = (
            float(account_balance_usd)
            if (account_balance_usd is not None and float(account_balance_usd) > 0)
            else float(getattr(self.risk_cfg, "total_account_balance_usd", 100.0))
        )
        base_target_pct = float(getattr(self.risk_cfg, "target_sl_margin_risk_pct", 15.0))
        floor_risk_usd = max(0.05, base_margin * (base_target_pct / 100.0))
        compound_pct = max(0.0, float(getattr(self.risk_cfg, "compound_risk_pct", 2.0)))
        compound_risk_usd = equity * (compound_pct / 100.0) if compound_pct > 0 else 0.0
        compound_mode = bool(compound_pct > 0 and compound_risk_usd >= floor_risk_usd)
        target_risk_usd = max(floor_risk_usd, compound_risk_usd)

        # Keep liquidation well outside the stop-loss (~90%/lev vs stop_dist_pct)
        sd_pct = max(stop_dist_pct, 0.15)
        liq_safe_max = max(1, int(math.floor(65.0 / sd_pct)))
        max_allowed = max(1, min(self.risk_cfg.max_leverage, asset_max, liq_safe_max))
        safe_ceiling = max(1, min(self.risk_cfg.default_leverage, max_allowed))
        min_lev = min(self.risk_cfg.min_leverage, max_allowed)

        # Ideal notional to risk exact target_risk_usd (2% of equity) at stop_dist_pct
        ideal_notional = target_risk_usd / (sd_pct / 100.0)
        max_margin_usd = max(base_margin, equity * 0.25)

        if ideal_notional <= base_margin * safe_ceiling:
            rec_margin = base_margin
            raw_lev = int(round(ideal_notional / base_margin))
            lev = max(min_lev, min(safe_ceiling, max(1, raw_lev)))
        else:
            lev = safe_ceiling
            rec_margin = min(max_margin_usd, ideal_notional / max(lev, 1))
            rec_margin = float(int(round(rec_margin))) if rec_margin >= 10.0 else round(rec_margin, 2)
            rec_margin = max(base_margin, rec_margin)

        min_notional = float(profile.min_notional_usd) if profile else 0.0
        hit_min_notional = False
        if min_notional > rec_margin * lev:
            hit_min_notional = True
            if min_notional <= rec_margin * safe_ceiling:
                lev = max(lev, int(math.ceil(min_notional / rec_margin)))
                lev = min(lev, safe_ceiling)
            else:
                lev = safe_ceiling
                rec_margin = float(int(math.ceil(min_notional / max(lev, 1))))

        notional = rec_margin * lev
        actual_sl_loss = notional * (sd_pct / 100.0)
        risk_mult = actual_sl_loss / target_risk_usd
        liq = entry_price * (1.0 - 0.90 / lev) if side == "LONG" else entry_price * (1.0 + 0.90 / lev)
        liq_dist_pct = 90.0 / max(lev, 1)

        # Pre-compute Turtle Pyramid (+2.0R -> +0.75R) second-leg size
        pyr_scale = max(0.0, float(getattr(self.risk_cfg, "pyramid_add_scale", 0.75)))
        pyr_margin = round(max(base_margin, rec_margin * pyr_scale), 2) if pyr_scale > 0 else 0.0
        if pyr_scale > 0 and min_notional > pyr_margin * lev:
            pyr_margin = float(int(math.ceil(min_notional / max(lev, 1))))
        pyr_risk_usd = round(pyr_margin * lev * (sd_pct / 100.0), 2) if pyr_margin > 0 else 0.0

        sizing_note = ""
        if hit_min_notional and (rec_margin > base_margin + 0.01 or risk_mult >= 1.35):
            sizing_note = (
                f"BTCC min lot (${min_notional:.0f}): use ${rec_margin:.0f} @ {lev}x "
                f"(SL ~${actual_sl_loss:.2f} / {risk_mult:.1f}R • liq {liq_dist_pct:.0f}% away)"
            )
        elif compound_mode and rec_margin > base_margin + 0.01:
            sizing_note = (
                f"Dynamic {compound_pct:.1f}% equity risk (${equity:,.0f} bal): "
                f"${rec_margin:.2f} @ {lev}x (SL ~${actual_sl_loss:.2f} = 1R • liq {liq_dist_pct:.0f}% away)"
            )

        return {
            "margin_usd": round(rec_margin, 2),
            "leverage": int(lev),
            "liq_price": float(liq),
            "actual_sl_loss_usd": round(actual_sl_loss, 2),
            "target_1r_usd": round(target_risk_usd, 2),
            "risk_multiple_of_1r": round(risk_mult, 2),
            "liq_dist_pct": round(liq_dist_pct, 1),
            "equity_usd": round(equity, 2),
            "compound_risk_pct": round(compound_pct, 2),
            "compound_mode": compound_mode,
            "pyramid_add_margin_usd": round(pyr_margin, 2),
            "pyramid_add_risk_usd": round(pyr_risk_usd, 2),
            "sizing_note": sizing_note,
        }

    def compute_safe_leverage(self, symbol: str, side: str, entry_price: float, stop_dist_pct: float) -> Tuple[int, float]:
        """Backwards-compatible wrapper returning (leverage, liq_price) from risk-parity sizing."""
        sz = self.compute_position_sizing(symbol, side, entry_price, stop_dist_pct)
        return int(sz["leverage"]), float(sz["liq_price"])

    def build_price_and_time_guidance(
        self,
        *,
        side: str,
        signal_price: float,
        live_price: float,
        stop_loss: float,
        atr: float,
        timeframe: str,
        prec: int,
        is_active: bool,
        is_daily: bool = False,
        bars_ago: Optional[int] = None,
        max_age_bars: int = 2,
        now_utc: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """
        Delayed-read-friendly Price Suggestions Ladder & Time Estimations:
        Gives the trader exact Maker Limit, Safe Entry Zone, Max Chase Ceiling (+0.50R),
        Pyramid Add (+2.0R) + Breakeven Stop, exact candle/daily close countdown, entry validity
        window, ATR-based watch trigger ETA, and empirical 2019-2026 Krypton V5 hold-time windows.
        """
        now = now_utc or pd.Timestamp.now(tz="UTC")
        is_long = str(side).upper() == "LONG"
        sgn = 1.0 if is_long else -1.0
        r = max(abs(signal_price - stop_loss), signal_price * 0.002)

        # ---- 1. Price Suggestions Ladder ----
        max_chase = round(signal_price + sgn * 0.50 * r, prec)
        pullback_bound = signal_price - sgn * 0.25 * r
        sl_floor = stop_loss + sgn * 0.35 * r
        inner_bound = max(pullback_bound, sl_floor) if is_long else min(pullback_bound, sl_floor)
        safe_low = round(min(inner_bound, max_chase), prec)
        safe_high = round(max(inner_bound, max_chase), prec)

        if is_active:
            retest_anchor = signal_price - sgn * 0.05 * r
            sl_maker_floor = stop_loss + sgn * 0.45 * r
            if is_long:
                ideal_maker = min(live_price * 0.9995, max(retest_anchor, sl_maker_floor))
            else:
                ideal_maker = max(live_price * 1.0005, min(retest_anchor, sl_maker_floor))
        else:
            ideal_maker = signal_price * (0.9995 if is_long else 1.0005)
        ideal_maker = round(ideal_maker, prec)

        pyr_r = float(getattr(self.risk_cfg, "pyramid_trigger_r", 2.0))
        pyr_price = round(signal_price + sgn * pyr_r * r, prec)
        pyr_be_stop = round(signal_price + sgn * 0.10 * r, prec)

        # Current status relative to ladder
        run_r = ((live_price - signal_price) if is_long else (signal_price - live_price)) / max(r, 1e-12)
        if not is_active:
            entry_status = "WAITING_FOR_CLOSE_TRIGGER"
        elif run_r <= 0.10:
            entry_status = "NEAR_IDEAL_MAKER"
        elif run_r <= 0.35:
            entry_status = "IN_SAFE_ZONE"
        else:
            entry_status = "NEAR_CHASE_LIMIT"

        # ---- 2. Next Candle / Daily Close Countdown ----
        if is_daily:
            dow = int(now.dayofweek)
            today_floor = now.floor("D")
            if dow < 5:  # Mon..Fri -> today's weekday bar closes at next 00:00 UTC
                next_close = today_floor + pd.Timedelta(days=1)
            elif dow == 5:  # Sat -> Monday bar closes Tue 00:00 UTC
                next_close = today_floor + pd.Timedelta(days=3)
            else:  # Sun -> Monday bar closes Tue 00:00 UTC
                next_close = today_floor + pd.Timedelta(days=2)
            rem_secs = max(60, int((next_close - now).total_seconds()))
        else:
            secs = _TF_SECONDS.get(timeframe, 14400)
            epoch = int(now.timestamp())
            rem_secs = secs - (epoch % secs)
            next_close = pd.Timestamp(epoch + rem_secs, unit="s", tz="UTC")

        def _fmt_dur(total_s: int) -> str:
            mins = max(1, int(round(total_s / 60.0)))
            days, rem_m = divmod(mins, 1440)
            hrs, rem_m = divmod(rem_m, 60)
            if days > 0:
                return f"{days}d {hrs}h"
            if hrs > 0:
                return f"{hrs}h {rem_m}m"
            return f"{rem_m}m"

        next_close_eta_str = _fmt_dur(rem_secs)
        next_close_utc_str = next_close.strftime("%Y-%m-%d %H:%M UTC") if is_daily else next_close.strftime("%H:%M UTC")

        # ---- 3. Entry Validity Window (for delayed Telegram reads) ----
        age = int(bars_ago or 0)
        rem_bars = max(1, max_age_bars - age + 1)
        if is_daily:
            valid_hrs = min(48, rem_bars * 24)
        else:
            bar_hrs = max(1.0, _TF_SECONDS.get(timeframe, 14400) / 3600.0)
            valid_hrs = int(max(2, round(rem_bars * bar_hrs)))
        valid_until = now + pd.Timedelta(hours=valid_hrs)
        if valid_hrs >= 36:
            entry_valid_str = f"~{valid_hrs}h (until {valid_until.strftime('%b %d %H:%M UTC')})"
        else:
            entry_valid_str = f"~{valid_hrs}h (until {valid_until.strftime('%H:%M UTC')})"

        # ---- 4. ATR-based Trigger ETA (for Watch cards) ----
        dist = abs(signal_price - live_price)
        atr_safe = max(float(atr), live_price * 0.002)
        bars_dist = dist / atr_safe
        if is_active:
            trigger_eta_str = f"ACTIVE NOW — valid {entry_valid_str} if price stays inside safe zone"
        elif is_daily:
            if bars_dist <= 0.45:
                trigger_eta_str = f"Near trigger — check 00:00 UTC close (in {next_close_eta_str})"
            else:
                d_lo = max(1, int(round(bars_dist * 0.75)))
                d_hi = max(d_lo + 1, int(round(max(bars_dist * 1.85, d_lo + 2))))
                if d_hi <= 15:
                    trigger_eta_str = f"~{d_lo}–{d_hi} trading days (at current volatility)"
                else:
                    trigger_eta_str = f"~{max(2, d_lo // 5)}–{max(3, (d_hi + 4) // 5)} weeks away"
        else:
            bar_hrs = max(1.0, _TF_SECONDS.get(timeframe, 14400) / 3600.0)
            if bars_dist <= 0.50:
                trigger_eta_str = f"Near trigger — watch next close in {next_close_eta_str} ({next_close_utc_str})"
            else:
                h_lo = max(int(bar_hrs), int(round(bars_dist * bar_hrs * 0.8)))
                h_hi = max(h_lo + int(bar_hrs), int(round(bars_dist * bar_hrs * 2.2)))
                if h_hi < 48:
                    trigger_eta_str = f"~{h_lo}h–{h_hi}h (next close in {next_close_eta_str})"
                else:
                    trigger_eta_str = f"~{h_lo / 24.0:.1f}–{h_hi / 24.0:.1f} days (next close in {next_close_eta_str})"

        # ---- 5. Empirical Hold & Milestone Windows (2019-2026 Krypton V5 / Metals / Equities medians) ----
        if is_daily:
            expected_pyr_eta = "~2–4 weeks to +2R"
            expected_winner_hold = "~4–12 weeks (multi-week trend)"
            expected_stop_win = "~1–3 weeks if invalidated"
        elif self._tf_bucket(timeframe) == "1h":
            expected_pyr_eta = "~24h–40h (median ~30h to +2R)"
            expected_winner_hold = "~6–11 days (median ~9d)"
            expected_stop_win = "~1–2 days (median ~35h)"
        else:
            expected_pyr_eta = "~36h–56h (median ~44h to +2R)"
            expected_winner_hold = "~5–9 days (median ~6.5d)"
            expected_stop_win = "~1.5–2.5 days (median ~48h)"

        return {
            "signal_price": round(signal_price, prec),
            "ideal_maker_entry": ideal_maker,
            "safe_entry_low": safe_low,
            "safe_entry_high": safe_high,
            "max_chase_price": max_chase,
            "pyramid_trigger_r": pyr_r,
            "pyramid_add_price": pyr_price,
            "pyramid_be_stop": pyr_be_stop,
            "entry_status_now": entry_status,
            "next_close_utc": next_close_utc_str,
            "next_close_eta_str": next_close_eta_str,
            "entry_valid_until_utc": valid_until.strftime("%Y-%m-%d %H:%M UTC"),
            "entry_valid_window_str": entry_valid_str,
            "trigger_eta_str": trigger_eta_str,
            "expected_pyramid_eta_str": expected_pyr_eta,
            "expected_hold_winner_str": expected_winner_hold,
            "expected_stop_window_str": expected_stop_win,
        }

    def _check_pyramid_add(
        self,
        pos: Any,
        live: float,
        prec: int,
    ) -> Optional[Any]:
        """
        Shared +2.0R Turtle Pyramid check (inherited by Krypton V5, Metals Trend, and Equities Trend).
        When an open winner reaches +pyramid_trigger_r * initial_1R and hasn't been pyramided yet:
          1. Ratchets `pos.stop_loss` to at least breakeven (`entry +/- 0.10R`).
          2. Returns an `ExitAlertSignal(reason="PYRAMID_ADD_2R", urgency="PYRAMID_ADD", ...)` with
             exact Maker Add limit, Max Chase ceiling, +0.75R add margin, and breakeven SL.
        """
        from detector import ExitAlertSignal
        from config import DEFAULT_ASSETS

        pyr_r = float(getattr(self.risk_cfg, "pyramid_trigger_r", 2.0))
        pyr_scale = float(getattr(self.risk_cfg, "pyramid_add_scale", 0.75))
        if pyr_r <= 0.0 or pyr_scale <= 0.0 or bool(getattr(pos, "pyramid_added", False)):
            return None

        is_long = str(pos.side).upper() == "LONG"
        sgn = 1.0 if is_long else -1.0
        entry = float(pos.entry_price)
        sl = float(pos.stop_loss)
        lev = int(pos.leverage)
        margin = float(pos.remaining_margin_usd)

        init_risk = float(getattr(pos, "initial_risk_dist", 0.0) or 0.0)
        if init_risk <= 0.0:
            if is_long and sl < entry:
                init_risk = entry - sl
            elif (not is_long) and sl > entry:
                init_risk = sl - entry
            else:
                init_risk = entry * 0.04  # conservative fallback for legacy positions with trailed SL

        mv = (live - entry) if is_long else (entry - live)
        r_mult = mv / max(init_risk, 1e-12)
        if r_mult < pyr_r:
            return None

        be_sl = round(entry + sgn * 0.10 * init_risk, prec)
        new_sl = max(sl, be_sl) if is_long else min(sl, be_sl)
        if abs(new_sl - sl) > 0:
            pos.stop_loss = round(new_sl, prec)

        profile = DEFAULT_ASSETS.get(str(pos.symbol).upper())
        min_notional = float(profile.min_notional_usd) if profile else 0.0
        base_margin = max(1.0, float(self.risk_cfg.margin_per_trade_usd))
        add_margin = round(max(base_margin, margin * pyr_scale), 2)
        if min_notional > add_margin * max(lev, 1):
            add_margin = float(int(math.ceil(min_notional / max(lev, 1))))

        maker_add = round(live * (0.9995 if is_long else 1.0005), prec)
        max_add_chase = round(entry + sgn * (pyr_r + 0.50) * init_risk, prec)
        roe = mv / max(entry, 1e-12) * 100.0 * lev

        return ExitAlertSignal(
            symbol=pos.symbol,
            position_side=pos.side,
            reason="PYRAMID_ADD_2R",
            urgency="PYRAMID_ADD",
            entry_price=entry,
            current_price=round(live, prec),
            unrealized_pnl_pct_on_margin=round(roe, 2),
            unrealized_pnl_usd=round(margin * roe / 100.0, 2),
            r_multiple=round(r_mult, 2),
            explanation=(
                f"Winner reached +{r_mult:.2f}R! Turtle Pyramid rule: "
                f"(1) Move existing Stop-Loss to breakeven {new_sl:.{prec}f} (+0.1R). "
                f"(2) Add second {pos.side} leg of ${add_margin:.2f} @ {lev}x "
                f"(Ideal Maker limit {maker_add:.{prec}f}, do NOT chase past {max_add_chase:.{prec}f}) "
                f"with shared SL at {new_sl:.{prec}f}."
            ),
        )

    @staticmethod
    def _orderflow_telemetry(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Market-microstructure context for the dashboard only (Krypton's rules do not use it)."""
        try:
            from carbon_v2_engine import CarbonFeatureEngine

            row = CarbonFeatureEngine.compute_features(df.tail(400), btc_df=btc_df.tail(400) if btc_df is not None else None).iloc[-1]
            g = lambda k, d=0.0: float(row[k]) if k in row and np.isfinite(row[k]) else d  # noqa: E731
            return {
                "delta_norm_pct": round(g("delta_norm") * 100.0, 1),
                "delta_z": round(g("delta_z"), 2),
                "cvd_6_pct": round(g("cvd_6") * 100.0, 1),
                "whale_print_ratio": round(g("whale_print_ratio", 1.0), 2),
                "vol_ratio": round(g("vol_ratio", 1.0), 2),
                "vol_regime": round(g("vol_regime", 1.0), 2),
                "ker20": round(g("ker20"), 2),
                "vwap_anchor": g("vwap24"),
                "z_vwap": round(g("z_vwap"), 2),
                "funding_z": round(g("funding_z"), 2),
                "btc_trend_score": round(g("btc_trend_score"), 2),
                "btc_residual_z": round(g("btc_residual_z"), 2),
                "htf_trend_atr": round(g("htf_trend_atr"), 2),
                "rsi14": round(g("rsi14", 50.0), 1),
            }
        except Exception:
            return {k: 0.0 for k in ("delta_norm_pct", "delta_z", "cvd_6_pct", "vol_regime", "ker20", "vwap_anchor", "z_vwap",
                                     "funding_z", "btc_trend_score", "btc_residual_z", "htf_trend_atr")} | {
                "whale_print_ratio": 1.0, "vol_ratio": 1.0, "rsi14": 50.0}

    # ---------------------------------------------------------------- entries
    def evaluate_symbol_live(
        self,
        df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame],
        symbol: str,
        timeframe: str = "4h",
        fear_greed_value: Optional[float] = None,
        market_intel: Optional[Dict[str, Any]] = None,
        account_balance_usd: Optional[float] = None,
    ) -> KryptonLiveScanResult:
        from detector import ReversalSetupSignal

        p = live_params_for(timeframe)
        live_price = float(df["close"].iloc[-1])
        dfc = closed_bars_only(df, timeframe)
        bdfc = closed_bars_only(btc_df, timeframe) if btc_df is not None else None
        prec = self._price_precision(live_price)
        oos = self.get_oos_asset_summary(symbol, timeframe)
        from config import DEFAULT_ASSETS as _DEF_ASSETS

        prof = _DEF_ASSETS.get(symbol.upper())
        reliable_feed = bool(df.attrs.get("is_reliable_24_7_feed", True))
        asset_validated = bool(getattr(prof, "validated_for_alerts", False)) and bool(oos.get("validated", False))
        fng_live = (
            float(fear_greed_value)
            if fear_greed_value is not None
            else (
                float(market_intel["fear_greed_value"])
                if isinstance(market_intel, dict) and market_intel.get("fear_greed_value") is not None
                else None
            )
        )
        is_crypto = _is_crypto_symbol(symbol)
        shorts_blocked_by_fear = bool(
            is_crypto
            and p.short_fng_min is not None
            and fng_live is not None
            and fng_live <= float(p.short_fng_min)
        )

        tel: Dict[str, Any] = {
            "engine": self.engine_name,
            "execution_style": "MARKET_OR_MAKER_ON_CLOSE",
            "preset_validated": timeframe in ("1h", "4h") and asset_validated,
            "asset_validated": asset_validated,
            "is_reliable_24_7_feed": reliable_feed,
            "data_source": str(df.attrs.get("data_source", "Unknown")),
            "meta_pred_r": round(float(oos.get("expectancy_r", 0.0)), 3),
            "meta_win_prob_pct": round(float(oos.get("win_rate_pct", 30.0)), 1),
            "meta_threshold_r": 0.0,
            "kelly_risk_scale": 1.0,
            "has_true_orderflow": bool(df.attrs.get("has_true_orderflow", False)),
            "fear_greed_value": int(round(fng_live)) if fng_live is not None else None,
            "shorts_blocked_by_fear": shorts_blocked_by_fear,
            "short_fng_min": p.short_fng_min,
        }
        tel.update(self._orderflow_telemetry(df, btc_df))

        slope_lag = max(1, int(p.slope_lookback))
        warm = max(p.ema_slow, p.entry_lookback + 1, p.atr_period * 3, slope_lag + 1) + 5
        if dfc is None or len(dfc) < warm:
            atr_fb = float((df["high"] - df["low"]).tail(20).mean()) if len(df) else live_price * 0.02
            tel.update({"archetype": "INSUFFICIENT_HISTORY", "trend": "NONE", "atr14": round(atr_fb, prec), "dist_to_trap_pct": 99.0})
            sl = live_price - 2.0 * atr_fb
            plan = self._plan(
                "LONG", live_price, sl, live_price, prec, symbol,
                order_type="NO_TRADE_INSUFFICIENT_HISTORY", expected_r=0.0,
                account_balance_usd=account_balance_usd,
            )
            return KryptonLiveScanResult(symbol, timeframe, None, None, None, plan, tel, oos)

        f = compute_features(dfc, p, btc_df=bdfc)
        n = len(dfc)
        last = n - 1
        atr_now = float(f["atr"][last])
        c, h, l = f["close"], f["high"], f["low"]
        ef, es = f["ema_f"], f["ema_s"]
        cvd_now = float(f["cvd"][last]) if "cvd" in f else 0.0
        tsmom_now = float(f["tsmom"][last]) if "tsmom" in f and np.isfinite(f["tsmom"][last]) else 0.0
        trend_up = bool(
            ef[last] > es[last]
            and c[last] > es[last]
            and (not p.require_price_vs_emaf or c[last] > ef[last])
            and (not p.require_emaf_slope or ef[last] > ef[last - slope_lag])
        )
        trend_dn = bool(
            ef[last] < es[last]
            and c[last] < es[last]
            and (not p.require_price_vs_emaf or c[last] < ef[last])
            and (not p.require_emaf_slope or ef[last] < ef[last - slope_lag])
        )
        brk_hi = float(np.max(h[n - p.entry_lookback:]))   # next-bar breakout levels (incl. last closed bar)
        brk_lo = float(np.min(l[n - p.entry_lookback:]))
        ex_lo_next = float(np.min(l[n - p.exit_lookback:]))
        ex_hi_next = float(np.max(h[n - p.exit_lookback:]))
        tel.update(
            {
                "atr14": round(atr_now, prec),
                "trend": "UP" if trend_up else ("DOWN" if trend_dn else "NONE"),
                "ema_fast": round(float(ef[last]), prec),
                "ema_slow": round(float(es[last]), prec),
                "cvd_window_pct": round(cvd_now * 100.0, 1),
                "tsmom_now": round(tsmom_now, 3),
                "breakout_long_level": round(brk_hi, prec),
                "breakout_short_level": round(brk_lo, prec),
                "exit_channel_long": round(ex_lo_next, prec),
                "exit_channel_short": round(ex_hi_next, prec),
                "entry_lookback": p.entry_lookback,
                "exit_lookback": p.exit_lookback,
                "init_stop_atr": p.init_stop_atr,
                "trail_atr": p.trail_atr,
                "last_closed_bar": str(f["ts_str"][last]),
            }
        )

        # ---- 1. Active signal: fresh V5 breakout / V-bottom flip / CTA TSMOM within the last few closed bars
        # Never synthesize live 24/7 perpetual entries from gappy Yahoo fallback feeds or unvalidated symbols.
        max_age = 2 if self._tf_bucket(timeframe) == "4h" else 3
        sigs = (
            generate_signals(f, symbol, p, start_idx=max(0, n - 1 - max_age), live_fng=fng_live)
            if (reliable_feed and asset_validated)
            else {}
        )
        active_sig, bars_ago, cand_info = None, None, None
        active_sizing: Optional[Dict[str, Any]] = None
        active_guidance: Optional[Dict[str, Any]] = None
        active_archetype: str = "TREND_BREAKOUT"
        for i in sorted(sigs, reverse=True):
            s = sigs[i]
            age = last - i
            is_long = s.side == "LONG"
            sl0 = c[i] - s.stop_dist if is_long else c[i] + s.stop_dist
            # still alive? replay Krypton management over the bars after the signal
            alive, trail = True, sl0
            peak = c[i]
            for k in range(i + 1, n):
                if (is_long and l[k] <= trail) or ((not is_long) and h[k] >= trail):
                    alive = False
                    break
                if k > i + 1 and ((is_long and c[k] < f["ex_lo"][k]) or ((not is_long) and c[k] > f["ex_hi"][k])):
                    alive = False
                    break
                peak = max(peak, h[k]) if is_long else min(peak, l[k])
                trail = max(trail, peak - p.trail_atr * f["atr"][k]) if is_long else min(trail, peak + p.trail_atr * f["atr"][k])
            if not alive:
                continue
            # chase guard: do not enter if price already ran > 0.5R past the signal close or sits within 0.3R of the stop
            r = s.stop_dist
            run = (live_price - c[i]) if is_long else (c[i] - live_price)
            room = (live_price - trail) if is_long else (trail - live_price)
            if run > 0.5 * r or room < 0.3 * r:
                continue
            stop_pct = abs(live_price - trail) / live_price * 100.0
            active_sizing = self.compute_position_sizing(
                symbol, s.side, live_price, stop_pct, account_balance_usd=account_balance_usd
            )
            active_guidance = self.build_price_and_time_guidance(
                side=s.side,
                signal_price=float(c[i]),
                live_price=live_price,
                stop_loss=float(trail),
                atr=atr_now,
                timeframe=timeframe,
                prec=prec,
                is_active=True,
                is_daily=False,
                bars_ago=age,
                max_age_bars=max_age,
            )
            lev, liq = int(active_sizing["leverage"]), float(active_sizing["liq_price"])
            m1 = live_price + 3.0 * r if is_long else live_price - 3.0 * r
            m2 = live_price + 6.0 * r if is_long else live_price - 6.0 * r
            active_archetype = s.archetype
            active_sig = ReversalSetupSignal(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=s.timestamp,
                impulse_direction=f"KRYPTON_V5_{s.archetype}_{'UP' if is_long else 'DOWN'}",
                trade_side=s.side,
                entry_price=round(live_price, prec),
                stop_loss=round(trail, prec),
                tp1_price=round(m1, prec),
                tp2_price=round(m2, prec),
                impulse_pct=round((c[i] / c[max(0, i - p.entry_lookback)] - 1.0) * 100.0, 3),
                impulse_atr_mult=round(s.score, 2),
                flatline_bars=0,
                flatline_range_pct=round(r / live_price * 100.0, 3),
                flatline_range_atr=round(r / max(atr_now, 1e-12), 2),
                body_decay_ratio=0.0,
                rsi_at_flatline=float(tel.get("rsi14", 50.0)),
                volume_decay_ratio=float(tel.get("vol_ratio", 1.0)),
                risk_reward_tp1=3.0,
                risk_reward_tp2=6.0,
                recommended_leverage=lev,
                estimated_liq_price=round(liq, prec),
            )
            bars_ago = age
            cand_info = KryptonCandidateInfo(
                archetype=s.archetype,
                trail_atr_mult=p.trail_atr,
                meta_pred_r=tel["meta_pred_r"],
                meta_win_prob=tel["meta_win_prob_pct"] / 100.0,
            )
            break

        # ---- 2. Watch plan: where the next breakout entry would trigger (on candle close)
        if trend_up or (tsmom_now >= 0.15 and not trend_dn):
            w_side, trig = "LONG", brk_hi
            order_type = "ENTER_ON_CLOSE_BEYOND_LEVEL"
            watch_arch = "BREAKOUT_WATCH"
        elif trend_dn and not shorts_blocked_by_fear:
            w_side, trig = "SHORT", brk_lo
            order_type = "ENTER_ON_CLOSE_BEYOND_LEVEL"
            watch_arch = "BREAKOUT_WATCH"
        elif trend_dn and shorts_blocked_by_fear:
            # Capitulation guard active: never display a SHORT trigger when Fear & Greed <= 25
            w_side, trig = "LONG", brk_hi
            order_type = "SHORTS_PAUSED_EXTREME_FEAR"
            watch_arch = "SHORTS_PAUSED_EXTREME_FEAR"
        else:
            # Rebound / mixed regime (e.g. price reclaimed EMA20 while EMA20 < EMA80): stand aside instead of shorting
            w_side, trig = "LONG", brk_hi
            order_type = "NO_TREND_STAND_ASIDE"
            watch_arch = "NO_TREND"

        w_sl = trig - p.init_stop_atr * atr_now if w_side == "LONG" else trig + p.init_stop_atr * atr_now
        w_sl = min(w_sl, trig * (1 - p.min_stop_pct)) if w_side == "LONG" else max(w_sl, trig * (1 + p.min_stop_pct))
        watch_guidance = self.build_price_and_time_guidance(
            side=w_side,
            signal_price=trig,
            live_price=live_price,
            stop_loss=w_sl,
            atr=atr_now,
            timeframe=timeframe,
            prec=prec,
            is_active=False,
            is_daily=False,
            max_age_bars=max_age,
        )
        plan = self._plan(
            w_side, trig, w_sl, live_price, prec, symbol,
            order_type=order_type, expected_r=tel["meta_pred_r"],
            exit_channel=ex_lo_next if w_side == "LONG" else ex_hi_next,
            account_balance_usd=account_balance_usd,
            guidance=watch_guidance,
        )
        tel["dist_to_trap_pct"] = round(abs(live_price - trig) / live_price * 100.0, 2)
        tel["watch_guidance"] = watch_guidance
        if active_sig is not None:
            tel["archetype"] = active_archetype
            tel["signal_archetype"] = active_archetype
            tel["active_stop"] = active_sig.stop_loss
            if active_sizing is not None:
                tel["active_sizing"] = active_sizing
            if active_guidance is not None:
                tel["active_guidance"] = active_guidance
            tel["exit_rule"] = (f"Close if a {timeframe} candle closes {'below' if active_sig.trade_side == 'LONG' else 'above'} "
                                f"{(ex_lo_next if active_sig.trade_side == 'LONG' else ex_hi_next):.{prec}f}")
        else:
            tel["archetype"] = watch_arch
        return KryptonLiveScanResult(symbol, timeframe, active_sig, bars_ago, cand_info, plan, tel, oos)

    def _plan(
        self,
        side: str,
        entry: float,
        sl: float,
        live_price: float,
        prec: int,
        symbol: str,
        *,
        order_type: str,
        expected_r: float,
        exit_channel: Optional[float] = None,
        account_balance_usd: Optional[float] = None,
        guidance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        r = abs(entry - sl) or entry * 0.01
        stop_pct = r / max(entry, 1e-12) * 100.0
        sz = self.compute_position_sizing(
            symbol, side, entry, stop_pct, account_balance_usd=account_balance_usd
        )
        lev, liq = int(sz["leverage"]), float(sz["liq_price"])
        sgn = 1.0 if side == "LONG" else -1.0
        maker_px = (
            guidance["ideal_maker_entry"]
            if (guidance and "ideal_maker_entry" in guidance)
            else round(entry * (1.0 - 0.0005) if side == "LONG" else entry * (1.0 + 0.0005), prec)
        )
        g = guidance or {}
        return {
            "side": side,
            "order_type": order_type,
            "trigger_entry_price": round(entry, prec),
            "limit_entry_price": round(entry, prec),
            "maker_limit_price": maker_px,
            "safe_entry_low": g.get("safe_entry_low"),
            "safe_entry_high": g.get("safe_entry_high"),
            "max_chase_price": g.get("max_chase_price", round(entry + sgn * 0.50 * r, prec)),
            "pyramid_add_price": g.get("pyramid_add_price", round(entry + sgn * 2.0 * r, prec)),
            "pyramid_be_stop": g.get("pyramid_be_stop", round(entry + sgn * 0.10 * r, prec)),
            "pyramid_add_margin_usd": sz.get("pyramid_add_margin_usd", 0.0),
            "next_close_utc": g.get("next_close_utc", ""),
            "next_close_eta_str": g.get("next_close_eta_str", ""),
            "entry_valid_window_str": g.get("entry_valid_window_str", ""),
            "trigger_eta_str": g.get("trigger_eta_str", ""),
            "expected_pyramid_eta_str": g.get("expected_pyramid_eta_str", ""),
            "expected_hold_winner_str": g.get("expected_hold_winner_str", ""),
            "expected_stop_window_str": g.get("expected_stop_window_str", ""),
            "stop_loss": round(sl, prec),
            "tp1_price": round(entry + sgn * 3.0 * r, prec),   # milestones only: Krypton has no fixed take-profit
            "tp2_price": round(entry + sgn * 6.0 * r, prec),
            "exit_channel": round(exit_channel, prec) if exit_channel is not None else None,
            "recommended_leverage": lev,
            "recommended_margin_usd": sz["margin_usd"],
            "actual_sl_loss_usd": sz["actual_sl_loss_usd"],
            "risk_multiple_of_1r": sz["risk_multiple_of_1r"],
            "compound_mode": sz.get("compound_mode", False),
            "compound_risk_pct": sz.get("compound_risk_pct", 2.0),
            "sizing_note": sz["sizing_note"],
            "estimated_liq_price": round(liq, prec),
            "expected_r": round(expected_r, 3),
            "win_prob": None,
        }

    # ---------------------------------------------------------------- exits
    def check_position_exit(self, df: pd.DataFrame, pos: Any, timeframe: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Krypton V5 exit & scaling management for a tracked position. Returns (exit_signal, trail_update):
          exit_signal  -> detector.ExitAlertSignal when the position should be closed (stop hit, exit-channel close,
                          time stop) OR pyramided (`urgency == "PYRAMID_ADD"` at +2.0R)
          trail_update -> {"old_sl", "new_sl"} when the chandelier trail ratchets the stop by >= 0.5 ATR (pos.stop_loss is updated)
        """
        from detector import ExitAlertSignal

        p = live_params_for(timeframe)
        is_long = pos.side == "LONG"
        live = float(df["close"].iloc[-1])
        dfc = closed_bars_only(df, timeframe)
        prec = self._price_precision(live)
        entry, lev, margin = float(pos.entry_price), int(pos.leverage), float(pos.remaining_margin_usd)
        sl = float(pos.stop_loss)
        init_risk = float(getattr(pos, "initial_risk_dist", 0.0) or 0.0)
        risk = max(init_risk if init_risk > 0 else abs(entry - sl), entry * 0.002)

        def mk(reason: str, urgency: str, px: float, why: str) -> ExitAlertSignal:
            mv = (px - entry) if is_long else (entry - px)
            roe = mv / entry * 100.0 * lev
            return ExitAlertSignal(symbol=pos.symbol, position_side=pos.side, reason=reason, urgency=urgency, entry_price=entry,
                                   current_price=round(px, prec), unrealized_pnl_pct_on_margin=round(roe, 2),
                                   unrealized_pnl_usd=round(margin * roe / 100.0, 2), r_multiple=round(mv / max(risk, 1e-12), 2),
                                   explanation=why)

        # 1. Stop (initial or trailed) breached by live price
        if (is_long and live <= sl) or ((not is_long) and live >= sl):
            in_profit = (sl > entry) if is_long else (sl < entry)
            return mk("TRAIL_STOP_HIT" if in_profit else "STOP_LOSS_HIT", "STOPPED_OUT", sl,
                      f"Krypton {'trailing ' if in_profit else ''}stop {sl:.{prec}f} crossed — close the position."), None

        if dfc is None or len(dfc) < max(p.exit_lookback, 30) + 2:
            pyr_sig = self._check_pyramid_add(pos, live, prec)
            return pyr_sig, None
        f = compute_features(dfc, p)
        n = len(dfc)
        last = n - 1
        try:
            opened = pd.to_datetime(pos.opened_at, utc=True)
        except Exception:
            opened = pd.to_datetime(dfc["timestamp"].iloc[-1], utc=True)
        ts = pd.to_datetime(dfc["timestamp"], utc=True)
        secs = _TF_SECONDS.get(timeframe, 14400)
        closed_since = (ts + pd.Timedelta(seconds=secs)) > opened  # bars that closed after the position was opened
        idx_since = np.where(closed_since.to_numpy())[0]

        # 2. Exit-channel close (only on bars that closed after entry)
        if len(idx_since):
            k = last
            if closed_since.iloc[k] and ((is_long and f["close"][k] < f["ex_lo"][k]) or ((not is_long) and f["close"][k] > f["ex_hi"][k])):
                lvl = f["ex_lo"][k] if is_long else f["ex_hi"][k]
                return mk("KRYPTON_CHANNEL_EXIT", "IMMEDIATE_CLOSE", live,
                          f"{timeframe} candle closed {'below' if is_long else 'above'} the {p.exit_lookback}-bar exit channel "
                          f"({lvl:.{prec}f}) — trend over, close now."), None

        # 3. Time stop
        if len(idx_since) >= p.max_hold_bars:
            return mk("TIME_STOP", "IMMEDIATE_CLOSE", live, f"Held {len(idx_since)} bars (max {p.max_hold_bars}) — close per Krypton rules."), None

        # 4. Turtle Pyramid Add (+2.0R winner -> +0.75R second leg + move SL to breakeven)
        pyr_sig = self._check_pyramid_add(pos, live, prec)
        if pyr_sig is not None:
            return pyr_sig, None

        # 5. Chandelier trail ratchet (computed on closed bars only)
        if len(idx_since):
            hs, ls_ = f["high"][idx_since], f["low"][idx_since]
            atr_k = float(f["atr"][last])
            new_sl = max(sl, float(hs.max()) - p.trail_atr * atr_k) if is_long else min(sl, float(ls_.min()) + p.trail_atr * atr_k)
            if abs(new_sl - sl) >= 0.5 * atr_k:   # only notify on meaningful (>= 0.5 ATR) ratchets
                pos.stop_loss = round(new_sl, prec)
                return None, {"old_sl": round(sl, prec), "new_sl": round(new_sl, prec)}
        return None, None
