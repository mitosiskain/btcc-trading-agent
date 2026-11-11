"""
Krypton V3 — Volatility-Normalised Trend-Following Engine for BTCC Perpetuals
=============================================================================

Why a trend engine?
-------------------
Argon V1 (impulse -> flatline fade) and Carbon V2 (climax-absorption fade + channel breakout with a
fixed 2.65R-3.6R take-profit and an ML meta-filter) are both, at heart, *short-horizon* setups that
cap the winners. In crypto perpetuals the most persistent, best-documented edge is the opposite:
time-series momentum with fat right tails. Krypton therefore:

1. Enters on a *fresh* Donchian channel breakout (first close beyond the prior N-bar extreme),
   only in the direction of the slow trend (EMA-fast vs EMA-slow, and price on the right side of
   EMA-slow). No ML, no per-symbol fitting: one parameter set for every asset.
2. Uses a volatility-normalised initial stop (k_init * ATR) so each trade risks exactly 1R ($25).
3. Has NO fixed take-profit. It rides the move with a Chandelier trail (k_trail * ATR from the
   highest high / lowest low since entry), plus a Donchian exit channel (close through the M-bar
   opposite extreme) and a generous time stop. Winners of 5R-15R pay for many small losers.

Everything is strictly causal: a signal on bar `i` uses data <= close[i]; the order fills at
open[i+1] (with spread/slippage), and trailing-stop levels computed on bar `k`'s close are only
active from bar `k+1`.

`simulate_trade_fast` re-implements the repo's `simulate_trade_honest` friction model
*bit-for-bit* (verified in `benchmark_krypton.py`): BTCC taker fee 4.5 bps/side, 2.0/3.5 bps
half-spread, volatility-scaled stop slippage, 8h funding accrual, pessimistic SL-first intrabar
resolution, isolated-margin liquidation, integer leverage sized so 1R = $25 on $100 margin.
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
    # Defaults = 4h preset, selected on development data only (2019-09 -> 2025-11-30).
    # It is effectively the classic Turtle "System 1" (20-bar breakout / 10-bar exit / 2N stop)
    # plus a 50/200 EMA trend filter; all 288 neighbouring 4h variants were profitable in development.
    timeframe: str = "4h"
    entry_lookback: int = 20        # Donchian breakout channel (bars)
    exit_lookback: int = 10         # Donchian exit channel (bars)
    ema_fast: int = 50
    ema_slow: int = 200
    atr_period: int = 20
    init_stop_atr: float = 2.0      # initial stop distance in ATR (= 1R)
    trail_atr: float = 8.0          # catastrophic chandelier trail in ATR (the exit channel does most exits)
    max_hold_bars: int = 180        # time stop (30 days on 4h)
    use_trend_filter: bool = True
    allow_shorts: bool = True
    btc_filter: bool = False        # alts: require BTC slow trend to agree (hurt in development -> off)
    min_stop_pct: float = 0.004     # stop must be >= 0.4% of price (>> ~0.15% round-trip friction)

    @classmethod
    def for_timeframe(cls, tf: str) -> "KryptonParams":
        if tf == "1h":
            # Calendar-equivalent horizons (x4 bars) of the 4h engine; stop/trail re-selected on 1h development data.
            return cls(timeframe="1h", entry_lookback=160, exit_lookback=80, ema_fast=200, ema_slow=800,
                       atr_period=80, init_stop_atr=3.0, trail_atr=12.0, max_hold_bars=720)
        return cls(timeframe=tf)


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
def generate_signals(f: Dict[str, np.ndarray], symbol: str, p: KryptonParams,
                     start_idx: int = 0, end_idx: Optional[int] = None) -> Dict[int, KryptonSignal]:
    c, atr = f["close"], f["atr"]
    n = len(c)
    warm = max(p.ema_slow, p.entry_lookback + 1, p.atr_period * 3)
    # Bars in [start_idx, end_idx) are evaluated; end_idx=None includes the final (closed) bar for live use.
    stop_i = n if end_idx is None else min(n, end_idx)
    out: Dict[int, KryptonSignal] = {}
    is_btc = symbol.upper().startswith("BTC")
    for i in range(max(start_idx, warm), stop_i):
        dh, dl = f["don_hi"][i], f["don_lo"][i]
        if not (np.isfinite(dh) and np.isfinite(dl)):
            continue
        pdh, pdl = f["don_hi"][i - 1], f["don_lo"][i - 1]
        a = atr[i]
        side = ""
        if c[i] > dh and c[i - 1] <= pdh:
            if (not p.use_trend_filter) or (f["ema_f"][i] > f["ema_s"][i] and c[i] > f["ema_s"][i]):
                if (not p.btc_filter) or is_btc or f["btc_trend"][i] > 0:
                    side = "LONG"
        elif p.allow_shorts and c[i] < dl and c[i - 1] >= pdl:
            if (not p.use_trend_filter) or (f["ema_f"][i] < f["ema_s"][i] and c[i] < f["ema_s"][i]):
                if (not p.btc_filter) or is_btc or f["btc_trend"][i] < 0:
                    side = "SHORT"
        if not side:
            continue
        stop = max(p.init_stop_atr * a, p.min_stop_pct * c[i])
        score = ((c[i] - dh) if side == "LONG" else (dl - c[i])) / a
        out[i] = KryptonSignal(symbol, i, str(f["ts_str"][i]), side, float(stop), float(score))
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


# --------------------------------------------------------------------------------------
# Per-symbol runner and timestamp-based portfolio cap
# --------------------------------------------------------------------------------------
def run_symbol(f: Dict[str, np.ndarray], symbol: str, p: KryptonParams, start_idx: int, end_idx: int,
               max_btcc_leverage: int = 100, cost_mult: float = 1.0) -> List[Dict[str, Any]]:
    """Non-overlapping per-symbol execution of Krypton signals whose signal bar lies in [start_idx, end_idx)."""
    sigs = generate_signals(f, symbol, p, start_idx=start_idx, end_idx=end_idx)
    trades: List[Dict[str, Any]] = []
    busy = -1
    for i in sorted(sigs):
        if i <= busy:
            continue
        s = sigs[i]
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


def portfolio_cap_by_time(trades: List[Dict[str, Any]], max_concurrent: int = 3, max_same_side: int = 2) -> List[Dict[str, Any]]:
    """Same rule as benchmark `apply_portfolio_concurrency_cap`, but keyed on timestamps (ragged histories)."""
    acc, active = [], []
    for tr in sorted(trades, key=lambda t: (t["entry_ns"], -float(t.get("meta_pred_r", 0.0)), t["symbol"])):
        e = tr["entry_ns"]
        active = [a for a in active if a["exit_ns"] >= e]
        if any(a["symbol"] == tr["symbol"] for a in active) or len(active) >= max_concurrent:
            continue
        if sum(1 for a in active if a["side"] == tr["side"]) >= max_same_side:
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
    Drop-in replacement for `CarbonLiveBridge` (same public surface) that runs Krypton V3 live:
      * `evaluate_symbol_live` -> active trend-breakout signal (ENTER NOW) or a breakout *watch* plan
        ("enter if the candle closes beyond X"), plus telemetry for the dashboard.
      * `check_position_exit` -> Krypton exit management for tracked positions
        (initial stop, chandelier trail ratchet, Donchian exit-channel close, time stop).
    """

    engine_name = "KRYPTON_V3"
    display_name = "Krypton V3"

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
        port = blk.get("krypton_v3_oos", {}) or {}
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

    def compute_safe_leverage(self, symbol: str, side: str, entry_price: float, stop_dist_pct: float) -> Tuple[int, float]:
        """Same isolated-margin rule as Carbon's live bridge: SL loses ~target_sl_margin_risk_pct of margin, capped safely below liquidation."""
        from config import DEFAULT_ASSETS

        profile = DEFAULT_ASSETS.get(symbol.upper())
        asset_max = profile.max_btcc_leverage if profile else self.risk_cfg.max_leverage
        # Never allow leverage where isolated liquidation (~90%/lev) sits inside or right on top of the stop-loss.
        liq_safe_max = max(1, int(math.floor(70.0 / max(stop_dist_pct, 0.15))))
        max_allowed = max(1, min(self.risk_cfg.max_leverage, asset_max, liq_safe_max))
        min_lot_lev = min(self.risk_cfg.min_leverage, max_allowed)
        if profile and self.risk_cfg.margin_per_trade_usd > 0:
            need = int(math.ceil(profile.min_notional_usd / self.risk_cfg.margin_per_trade_usd))
            min_lot_lev = max(min_lot_lev, min(need, max_allowed))
        target_pct = float(getattr(self.risk_cfg, "target_sl_margin_risk_pct", 15.0))
        raw_lev = int(round(target_pct / max(stop_dist_pct, 0.15)))
        ceiling = max(self.risk_cfg.default_leverage, min_lot_lev)
        lev = max(min_lot_lev, min(ceiling, max_allowed, max(1, raw_lev)))
        liq = entry_price * (1.0 - 0.90 / lev) if side == "LONG" else entry_price * (1.0 + 0.90 / lev)
        return lev, liq

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
    def evaluate_symbol_live(self, df: pd.DataFrame, btc_df: Optional[pd.DataFrame], symbol: str, timeframe: str = "4h") -> KryptonLiveScanResult:
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
        tel: Dict[str, Any] = {
            "engine": self.engine_name,
            "execution_style": "MARKET_ON_CANDLE_CLOSE",
            "preset_validated": timeframe in ("1h", "4h") and asset_validated,
            "asset_validated": asset_validated,
            "is_reliable_24_7_feed": reliable_feed,
            "data_source": str(df.attrs.get("data_source", "Unknown")),
            "meta_pred_r": round(float(oos.get("expectancy_r", 0.0)), 3),
            "meta_win_prob_pct": round(float(oos.get("win_rate_pct", 30.0)), 1),
            "meta_threshold_r": 0.0,
            "kelly_risk_scale": 1.0,
            "has_true_orderflow": bool(df.attrs.get("has_true_orderflow", False)),
        }
        tel.update(self._orderflow_telemetry(df, btc_df))

        warm = max(p.ema_slow, p.entry_lookback + 1, p.atr_period * 3) + 5
        if dfc is None or len(dfc) < warm:
            atr_fb = float((df["high"] - df["low"]).tail(20).mean()) if len(df) else live_price * 0.02
            tel.update({"archetype": "INSUFFICIENT_HISTORY", "trend": "NONE", "atr14": round(atr_fb, prec), "dist_to_trap_pct": 99.0})
            sl = live_price - 2.0 * atr_fb
            plan = self._plan("LONG", live_price, sl, live_price, prec, symbol, order_type="NO_TRADE_INSUFFICIENT_HISTORY", expected_r=0.0)
            return KryptonLiveScanResult(symbol, timeframe, None, None, None, plan, tel, oos)

        f = compute_features(dfc, p, btc_df=bdfc)
        n = len(dfc)
        last = n - 1
        atr_now = float(f["atr"][last])
        c, h, l = f["close"], f["high"], f["low"]
        trend_up = bool(f["ema_f"][last] > f["ema_s"][last] and c[last] > f["ema_s"][last])
        trend_dn = bool(f["ema_f"][last] < f["ema_s"][last] and c[last] < f["ema_s"][last])
        brk_hi = float(np.max(h[n - p.entry_lookback:]))   # next-bar breakout levels (incl. last closed bar)
        brk_lo = float(np.min(l[n - p.entry_lookback:]))
        ex_lo_next = float(np.min(l[n - p.exit_lookback:]))
        ex_hi_next = float(np.max(h[n - p.exit_lookback:]))
        tel.update(
            {
                "atr14": round(atr_now, prec),
                "trend": "UP" if trend_up else ("DOWN" if trend_dn else "NONE"),
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

        # ---- 1. Active signal: fresh breakout within the last few closed bars that is still alive
        # Never synthesize live 24/7 perpetual breakout entries from gappy Yahoo fallback feeds or unvalidated symbols.
        max_age = 2 if self._tf_bucket(timeframe) == "4h" else 3
        sigs = generate_signals(f, symbol, p, start_idx=max(0, n - 1 - max_age)) if (reliable_feed and asset_validated) else {}
        active_sig, bars_ago, cand_info = None, None, None
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
            lev, liq = self.compute_safe_leverage(symbol, s.side, live_price, stop_pct)
            m1 = live_price + 3.0 * r if is_long else live_price - 3.0 * r
            m2 = live_price + 6.0 * r if is_long else live_price - 6.0 * r
            active_sig = ReversalSetupSignal(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=s.timestamp,
                impulse_direction=f"KRYPTON_V3_TREND_BREAKOUT_{'UP' if is_long else 'DOWN'}",
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
            cand_info = KryptonCandidateInfo(trail_atr_mult=p.trail_atr, meta_pred_r=tel["meta_pred_r"],
                                             meta_win_prob=tel["meta_win_prob_pct"] / 100.0)
            break

        # ---- 2. Watch plan: where the next breakout entry would trigger (on candle close)
        if trend_dn and not trend_up:
            w_side, trig = "SHORT", brk_lo
        else:
            w_side, trig = "LONG", brk_hi
        w_sl = trig - p.init_stop_atr * atr_now if w_side == "LONG" else trig + p.init_stop_atr * atr_now
        w_sl = min(w_sl, trig * (1 - p.min_stop_pct)) if w_side == "LONG" else max(w_sl, trig * (1 + p.min_stop_pct))
        order_type = "ENTER_ON_CLOSE_BEYOND_LEVEL" if (trend_up or trend_dn) else "NO_TREND_STAND_ASIDE"
        plan = self._plan(w_side, trig, w_sl, live_price, prec, symbol, order_type=order_type, expected_r=tel["meta_pred_r"],
                          exit_channel=ex_lo_next if w_side == "LONG" else ex_hi_next)
        tel["dist_to_trap_pct"] = round(abs(live_price - trig) / live_price * 100.0, 2)
        if active_sig is not None:
            tel["archetype"] = "TREND_BREAKOUT"
            tel["active_stop"] = active_sig.stop_loss
            tel["exit_rule"] = (f"Close if a {timeframe} candle closes {'below' if active_sig.trade_side == 'LONG' else 'above'} "
                                f"{(ex_lo_next if active_sig.trade_side == 'LONG' else ex_hi_next):.{prec}f}")
        else:
            tel["archetype"] = "BREAKOUT_WATCH" if (trend_up or trend_dn) else "NO_TREND"
        return KryptonLiveScanResult(symbol, timeframe, active_sig, bars_ago, cand_info, plan, tel, oos)

    def _plan(self, side: str, entry: float, sl: float, live_price: float, prec: int, symbol: str, *, order_type: str,
              expected_r: float, exit_channel: Optional[float] = None) -> Dict[str, Any]:
        r = abs(entry - sl) or entry * 0.01
        stop_pct = r / max(entry, 1e-12) * 100.0
        lev, liq = self.compute_safe_leverage(symbol, side, entry, stop_pct)
        sgn = 1.0 if side == "LONG" else -1.0
        return {
            "side": side,
            "order_type": order_type,
            "trigger_entry_price": round(entry, prec),
            "limit_entry_price": round(entry, prec),
            "stop_loss": round(sl, prec),
            "tp1_price": round(entry + sgn * 3.0 * r, prec),   # milestones only: Krypton has no fixed take-profit
            "tp2_price": round(entry + sgn * 6.0 * r, prec),
            "exit_channel": round(exit_channel, prec) if exit_channel is not None else None,
            "recommended_leverage": lev,
            "estimated_liq_price": round(liq, prec),
            "expected_r": round(expected_r, 3),
            "win_prob": None,
        }

    # ---------------------------------------------------------------- exits
    def check_position_exit(self, df: pd.DataFrame, pos: Any, timeframe: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Krypton exit management for a tracked position. Returns (exit_signal, trail_update):
          exit_signal  -> detector.ExitAlertSignal when the position should be closed (stop hit, exit-channel close, time stop)
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
        risk = max(abs(entry - sl), entry * 0.002)

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
            return None, None
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

        # 4. Chandelier trail ratchet (computed on closed bars only)
        if len(idx_since):
            hs, ls_ = f["high"][idx_since], f["low"][idx_since]
            atr_k = float(f["atr"][last])
            new_sl = max(sl, float(hs.max()) - p.trail_atr * atr_k) if is_long else min(sl, float(ls_.min()) + p.trail_atr * atr_k)
            if abs(new_sl - sl) >= 0.5 * atr_k:   # only notify on meaningful (>= 0.5 ATR) ratchets
                pos.stop_loss = round(new_sl, prec)
                return None, {"old_sl": round(sl, prec), "new_sl": round(new_sl, prec)}
        return None, None
