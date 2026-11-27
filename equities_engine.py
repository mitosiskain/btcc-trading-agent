"""
Equities Trend V1 — dedicated tokenized-stock engine (GOOGLUSDT)
================================================================

Why a separate engine?
  Tokenized US equities on BTCC / Binance (e.g. GOOGLUSDT, listed 2026-03-26) trade 24/7 as perpetual swaps,
  whereas 90%+ of price discovery and liquidity occur during US cash hours (13:30–20:00 UTC Mon–Fri; weekend hourly
  volume is ~11x lower). Running crypto bar-count engines directly on 24/7 equity perpetuals collapses Donchian/VWAP
  channels onto dead overnight and weekend flatlines and shorts secular equity compounders:
    * Krypton V3 on GOOGLUSDT 24/7 perp: 4h PF 0.19 (-$428), 1h PF 0.33 (-$349)
    * Carbon V2  on GOOGLUSDT 24/7 perp: 4h PF 0.50 (-$233), 1h PF 0.33 (-$230)

Calendar-consistent & gap-honest design (identical signals on Nasdaq session data and on 24/7 perpetual data):
  * Every decision is taken once per WEEKDAY at the UTC daily close (00:00 UTC), from a weekday-only daily close
    series. Split-adjusted Nasdaq history (GOOGL since 2004) warms up the slow trend filter and breakout channel
    prior to the first 24/7 perpetual bar without ever synthesizing live intraday triggers from Yahoo data.
  * Execution runs on 24/7 perpetual bars with BTCC taker fees (4.5 bps/side), 3.5 bps half-spread, real 8h funding,
    isolated-margin liquidation checks, and TRUE overnight/earnings gap-through-stop fills (`min(open, sl)`).

Rules (selected strictly on pooled 2000–2015 US Mega-Cap Tech Equities — 100% of long-only trend variants were
profitable in both 2000–2015 dev and 2016–2025 validation; shorts collapsed GOOGL validation profitability to 28%):
  ENTRY  long when the weekday daily close breaks above the prior 40-weekday closing high (first cross) while above
         the 150-weekday simple moving average. Long-only by default (`EQUITIES_ALLOW_SHORTS=true` adds mirror shorts).
  STOP   2.5 x ATR(20 weekdays) initial stop (= 1R), then a Chandelier trail 4.0 x ATR below the highest high since entry.
  EXIT   weekday daily close below the prior 20-weekday closing low.
Evidence: `benchmark_equities.py` (GOOGL 2004–2026 PF 2.19 / +$891 / 0 liquidations; 24/7 GOOGLUSDT perp 4h PF 2.48, 1h PF 2.49).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from krypton_engine import (
    KryptonCandidateInfo,
    KryptonLiveBridge,
    KryptonLiveScanResult,
    TAKER_FEE,
)
from metals_engine import stitch_history, weekday_daily

EQUITIES_SYMBOLS: Dict[str, str] = {
    "GOOGLUSDT": "GOOGL",
}


@dataclass
class EquitiesParams:
    entry_days: int = 40          # breakout of the prior 40-weekday closing high
    exit_days: int = 20           # exit on a weekday close below the prior 20-weekday closing low
    sma_days: int = 150           # macro uptrend filter (150-weekday SMA)
    atr_days: int = 20
    init_stop_atr: float = 2.5    # wider initial stop for daily equity earnings/session noise (= 1R)
    trail_atr: float = 4.0        # Chandelier trail from highest high since entry
    allow_shorts: bool = False    # long-only: shorting compounder equities destroyed validation PF


# --------------------------------------------------------------------------------------
# Shared causal indicators & gap-honest backtest core
# --------------------------------------------------------------------------------------
def add_indicators(daily: pd.DataFrame, p: EquitiesParams) -> pd.DataFrame:
    out = daily.copy()
    c = out["close"].to_numpy(float)
    h = out["high"].to_numpy(float)
    l = out["low"].to_numpy(float)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    out["atr"] = pd.Series(tr, index=out.index).ewm(alpha=1.0 / p.atr_days, adjust=False).mean()
    out["sma"] = out["close"].rolling(p.sma_days).mean()
    cs = out["close"]
    out["hi_entry"] = cs.rolling(p.entry_days).max().shift(1)
    out["lo_entry"] = cs.rolling(p.entry_days).min().shift(1)
    out["lo_exit"] = cs.rolling(p.exit_days).min().shift(1)
    out["hi_exit"] = cs.rolling(p.exit_days).max().shift(1)
    return out


def entry_signals(d: pd.DataFrame, p: EquitiesParams, start: int = 0) -> List[Tuple[int, str]]:
    c = d["close"].to_numpy(float)
    sma = d["sma"].to_numpy(float)
    hi = d["hi_entry"].to_numpy(float)
    lo = d["lo_entry"].to_numpy(float)
    out: List[Tuple[int, str]] = []
    warm = max(start, p.sma_days + 1, p.entry_days + 1)
    for i in range(warm, len(d)):
        if not (np.isfinite(hi[i]) and np.isfinite(sma[i]) and np.isfinite(hi[i - 1])):
            continue
        if c[i] > hi[i] and c[i - 1] <= hi[i - 1] and c[i] > sma[i]:
            out.append((i, "LONG"))
        elif p.allow_shorts and np.isfinite(lo[i]) and np.isfinite(lo[i - 1]):
            if c[i] < lo[i] and c[i - 1] >= lo[i - 1] and c[i] < sma[i]:
                out.append((i, "SHORT"))
    return out


def simulate_equity_trade(
    ex: pd.DataFrame,
    symbol: str,
    signal_idx: int,
    side: str,
    stop_dist: float,
    trail_atr_mult: float,
    atr_arr: np.ndarray,
    exit_flags: np.ndarray,
    *,
    max_btcc_leverage: int = 50,
    margin_usd: float = 100.0,
    base_risk_usd: float = 25.0,
    cost_mult: float = 1.0,
    funding_rates: Optional[np.ndarray] = None,
    funding_bars: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """
    Causal, gap-honest & isolated-liquidation-honest perpetual trade simulator:
      * Signal on bar `signal_idx` fills at `open[signal_idx + 1]` + half-spread + BTCC 4.5 bps taker fee.
      * Overnight / earnings gap-through-stop fills at `min(open[k], sl)` for longs (`max(open[k], sl)` for shorts)
        minus volatility-scaled stop slippage, rather than optimistically filling at `sl`.
      * Isolated liquidation (`entry * (1 -/+ 0.90/lev)`) checked on both `open[k]` and intrabar `low[k]`/`high[k]`.
      * Accrues real 8h perpetual funding (or elapsed 0.01%/8h on historical session bars).
    """
    n = len(ex)
    entry_idx = signal_idx + 1
    if entry_idx >= n or stop_dist <= 0:
        return None

    o = ex["open"].to_numpy(float)
    h = ex["high"].to_numpy(float)
    l = ex["low"].to_numpy(float)
    c = ex["close"].to_numpy(float)
    ts_s = pd.to_datetime(ex["timestamp"], utc=True)
    ts_ns = ts_s.dt.as_unit("ns").astype("int64").to_numpy()
    ts_str = ts_s.astype(str).to_numpy()

    raw_open = float(o[entry_idx])
    if raw_open <= 0:
        return None
    fee = TAKER_FEE * cost_mult
    slip = 0.00035 * cost_mult
    is_long = side == "LONG"
    entry = raw_open * (1.0 + slip) if is_long else raw_open * (1.0 - slip)
    stop_pct = stop_dist / entry
    if stop_pct <= 0.001:
        return None

    max_lev = min(100, max_btcc_leverage)
    lev = max(2, min(max_lev, int(round(base_risk_usd / max(margin_usd * stop_pct, 1e-9)))))
    sl = entry - stop_dist if is_long else entry + stop_dist
    liq = entry * (1.0 - 0.90 / lev) if is_long else entry * (1.0 + 0.90 / lev)

    pnl = -(margin_usd * lev * fee)
    peak = h[entry_idx] if is_long else l[entry_idx]
    exit_idx, exit_px, reason = entry_idx, entry, "OPEN_AT_END"
    mfe_r = 0.0

    for k in range(entry_idx, n):
        op, hi, lo, cl = float(o[k]), float(h[k]), float(l[k]), float(c[k])
        a = max(float(atr_arr[k]), 1e-9)
        if k > entry_idx and funding_bars is not None and funding_rates is not None and funding_bars[k]:
            pnl += (-1.0 if is_long else 1.0) * margin_usd * lev * float(funding_rates[k])

        if (is_long and (op <= liq or lo <= liq)) or ((not is_long) and (op >= liq or hi >= liq)):
            pnl -= margin_usd
            exit_idx, exit_px, reason = k, liq, "LIQUIDATED"
            break

        if (is_long and (op <= sl or lo <= sl)) or ((not is_long) and (op >= sl or hi >= sl)):
            tr_frac = (hi - lo) / max(cl, 1e-9)
            sslip = slip + min(0.0008, 0.04 * tr_frac) * cost_mult
            base_stop_px = min(op, sl) if is_long else max(op, sl)
            fill = base_stop_px * (1.0 - sslip) if is_long else base_stop_px * (1.0 + sslip)
            ret = (fill - entry) / entry if is_long else (entry - fill) / entry
            pnl += max(-margin_usd, margin_usd * lev * ret) - margin_usd * lev * fee
            exit_idx, exit_px = k, fill
            reason = "TRAIL_STOP" if ((fill > entry) if is_long else (fill < entry)) else "STOP_LOSS"
            break

        mfe_r = max(mfe_r, ((hi - entry) if is_long else (entry - lo)) / stop_dist)

        if trail_atr_mult > 0:
            if is_long:
                peak = max(peak, hi)
                sl = max(sl, peak - trail_atr_mult * a)
            else:
                peak = min(peak, lo)
                sl = min(sl, peak + trail_atr_mult * a)

        if k > entry_idx and exit_flags[k]:
            fill = cl * (1.0 - slip) if is_long else cl * (1.0 + slip)
            ret = (fill - entry) / entry if is_long else (entry - fill) / entry
            pnl += max(-margin_usd, margin_usd * lev * ret) - margin_usd * lev * fee
            exit_idx, exit_px, reason = k, fill, "CHANNEL_EXIT"
            break

        if k == n - 1:
            fill = cl * (1.0 - slip) if is_long else cl * (1.0 + slip)
            ret = (fill - entry) / entry if is_long else (entry - fill) / entry
            pnl += max(-margin_usd, margin_usd * lev * ret) - margin_usd * lev * fee
            exit_idx, exit_px, reason = k, fill, "END_OF_SAMPLE"
            break

    one_r = max(margin_usd * lev * stop_pct, 1.0)
    return {
        "symbol": symbol,
        "side": side,
        "leverage": lev,
        "signal_idx": signal_idx,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "entry_time": str(ts_str[entry_idx]),
        "exit_time": str(ts_str[exit_idx]),
        "entry_ns": int(ts_ns[entry_idx]),
        "exit_ns": int(ts_ns[exit_idx]),
        "entry_price": round(entry, 4),
        "exit_price": round(float(exit_px), 4),
        "net_pnl_usd": round(float(pnl), 4),
        "roe_pct": round(float(pnl / margin_usd * 100.0), 3),
        "net_r": round(float(pnl / one_r), 4),
        "mfe_r": round(float(mfe_r), 3),
        "exit_reason": reason,
        "bars_held": exit_idx - entry_idx + 1,
    }


def backtest(
    ex: pd.DataFrame,
    symbol: str,
    p: EquitiesParams = EquitiesParams(),
    *,
    daily_hist: Optional[pd.DataFrame] = None,
    funding_mode: str = "elapsed",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    max_btcc_leverage: int = 50,
    cost_mult: float = 1.0,
    margin_usd: float = 100.0,
    base_risk_usd: float = 25.0,
) -> List[Dict[str, Any]]:
    """Run Equities Trend V1 on `ex` bars (session daily or 24/7 4h/1h perp) with gap-honest execution."""
    dl = weekday_daily(ex)
    d, off = stitch_history(dl, daily_hist)
    d = add_indicators(d, p)
    n_ex = len(ex)
    n_d = len(d)
    li = d["last_idx"].to_numpy(int)
    atr_d = d["atr"].to_numpy(float)

    atr_ex = np.full(n_ex, np.nan)
    for j in range(off, n_d):
        lo_i = li[j] + 1
        hi_i = li[j + 1] + 1 if j + 1 < n_d else n_ex
        if 0 <= lo_i < n_ex:
            atr_ex[lo_i:hi_i] = atr_d[j]
    atr_ex = pd.Series(atr_ex).bfill().ffill().to_numpy()

    ex_ts = pd.to_datetime(ex["timestamp"], utc=True)
    if funding_mode == "real" and "funding_rate" in ex.columns:
        f_bars = ((ex_ts.dt.minute == 0) & (ex_ts.dt.hour % 8 == 0)).to_numpy()
        f_rates = ex["funding_rate"].to_numpy(float)
    else:
        ns = ex_ts.dt.as_unit("ns").astype("int64").to_numpy()
        hrs = np.diff(ns, prepend=ns[0]) / 3.6e12
        f_bars = np.ones(n_ex, dtype=bool)
        f_rates = 0.0001 * hrs / 8.0

    c = d["close"].to_numpy(float)
    ex_long = c < d["lo_exit"].to_numpy(float)
    ex_short = c > d["hi_exit"].to_numpy(float)

    trades: List[Dict[str, Any]] = []
    busy = -1
    for i, side in entry_signals(d, p, start=off):
        sidx = int(li[i])
        if sidx < 0 or sidx <= busy or sidx + 1 >= n_ex:
            continue
        ts = ex_ts.iloc[sidx]
        if (start is not None and ts < start) or (end is not None and ts >= end):
            continue
        exit_flags = np.zeros(n_ex, dtype=bool)
        cond = ex_long if side == "LONG" else ex_short
        for j in np.where(cond & (np.arange(n_d) > i))[0]:
            if 0 <= li[j] < n_ex:
                exit_flags[li[j]] = True
        t = simulate_equity_trade(
            ex,
            symbol,
            sidx,
            side,
            p.init_stop_atr * atr_d[i],
            p.trail_atr,
            atr_ex,
            exit_flags,
            max_btcc_leverage=max_btcc_leverage,
            margin_usd=margin_usd,
            base_risk_usd=base_risk_usd,
            cost_mult=cost_mult,
            funding_rates=f_rates,
            funding_bars=f_bars,
        )
        if t is not None and t["exit_reason"] != "END_OF_SAMPLE":
            t["archetype"] = "EQUITIES_TREND"
            trades.append(t)
            busy = t["exit_idx"]
    return trades


# --------------------------------------------------------------------------------------
# Live bridge (drop-in surface matching KryptonLiveBridge & MetalsTrendBridge)
# --------------------------------------------------------------------------------------
class EquitiesTrendBridge(KryptonLiveBridge):
    engine_name = "EQUITIES_TREND"
    display_name = "Equities Trend"

    def __init__(self, risk_cfg: Any, state_dir: Optional[Any] = None, feed: Any = None):
        super().__init__(risk_cfg, state_dir=state_dir)
        import json

        self.p = EquitiesParams(allow_shorts=os.getenv("EQUITIES_ALLOW_SHORTS", "false").lower() == "true")
        self.feed = feed
        self._daily_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self.equities_stats: Dict[str, Any] = {}
        path = self.state_dir / "benchmark_equities_results.json"
        if path.exists():
            try:
                self.equities_stats = json.loads(path.read_text())
            except Exception:
                self.equities_stats = {}

    def get_oos_asset_summary(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        s = (self.equities_stats.get("live_summary") or {}).get(symbol.upper()) or {}
        ok = int(s.get("total_trades", 0)) > 0
        return {
            "scope": "EQUITIES_NASDAQ_2004_2026" if ok else "UNVALIDATED",
            "validated": ok,
            "win_rate_pct": float(s.get("win_rate_pct", 0.0)),
            "profit_factor": float(s.get("profit_factor", 0.0)),
            "total_net_pnl_usd": float(s.get("total_net_pnl_usd", 0.0)),
            "max_drawdown_usd": float(s.get("max_drawdown_usd", 0.0)),
            "total_trades": int(s.get("total_trades", 0)),
            "expectancy_r": float(s.get("expectancy_r", 0.0)),
            "portfolio_pf": float(s.get("profit_factor", 0.0)),
            "absorption_pf": float(s.get("profit_factor", 0.0)),
        }

    # ---------------------------------------------------------------- data
    def _daily(self, symbol: str, live_df: pd.DataFrame) -> pd.DataFrame:
        """Closed weekday daily bars from the 24/7 perpetual feed, warmed up with split-adjusted Nasdaq history."""
        now = time.time()
        hit = self._daily_cache.get(symbol)
        if hit and now - hit[0] < 900:
            base = hit[1]
        else:
            base = pd.DataFrame()
            if self.feed is not None:
                try:
                    dd = self.feed.fetch_ohlcv(symbol, interval="1d", limit=600)
                    if bool(dd.attrs.get("is_reliable_24_7_feed", False)):
                        base = weekday_daily(dd)
                except Exception:
                    base = pd.DataFrame()
                if len(base) < self.p.sma_days + self.p.entry_days + 5:
                    try:
                        ticker = EQUITIES_SYMBOLS.get(symbol.upper(), symbol.upper().removesuffix("USDT"))
                        yh = self.feed._fetch_yahoo(ticker, "1d", 900)
                        if not yh.empty:
                            hist = weekday_daily(yh)
                            base, _ = stitch_history(base, hist) if not base.empty else (hist, 0)
                    except Exception:
                        pass
            self._daily_cache[symbol] = (now, base)
        if base.empty:
            return base
        today = pd.Timestamp.now(tz="UTC").floor("D")
        return base[base.index < today]

    # ---------------------------------------------------------------- entries
    def evaluate_symbol_live(
        self,
        df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame],
        symbol: str,
        timeframe: str = "4h",
        fear_greed_value: Optional[float] = None,
        market_intel: Optional[Dict[str, Any]] = None,
    ) -> KryptonLiveScanResult:
        from detector import ReversalSetupSignal

        p = self.p
        live_price = float(df["close"].iloc[-1])
        prec = self._price_precision(live_price)
        oos = self.get_oos_asset_summary(symbol, timeframe)
        reliable = bool(df.attrs.get("is_reliable_24_7_feed", True))
        tel: Dict[str, Any] = {
            "engine": self.engine_name,
            "execution_style": "MARKET_AFTER_DAILY_CLOSE",
            "decision_label": "daily close (00:00 UTC)",
            "preset_validated": bool(oos.get("validated")),
            "asset_validated": bool(oos.get("validated")),
            "is_reliable_24_7_feed": reliable,
            "data_source": str(df.attrs.get("data_source", "Unknown")),
            "meta_pred_r": round(float(oos.get("expectancy_r", 0.0)), 3),
            "meta_win_prob_pct": round(float(oos.get("win_rate_pct", 43.5)), 1),
            "meta_threshold_r": 0.0,
            "kelly_risk_scale": 1.0,
            "has_true_orderflow": bool(df.attrs.get("has_true_orderflow", False)),
        }
        tel.update(self._orderflow_telemetry(df, btc_df))
        d = self._daily(symbol, df)
        if d.empty or len(d) < p.sma_days + p.entry_days + 2:
            atr_fb = float((df["high"] - df["low"]).tail(20).mean()) if len(df) else live_price * 0.02
            tel.update(
                {
                    "archetype": "INSUFFICIENT_HISTORY",
                    "trend": "NONE",
                    "atr14": round(atr_fb, prec),
                    "dist_to_trap_pct": 99.0,
                }
            )
            plan = self._plan(
                "LONG",
                live_price,
                live_price - p.init_stop_atr * atr_fb,
                live_price,
                prec,
                symbol,
                order_type="NO_TRADE_INSUFFICIENT_HISTORY",
                expected_r=0.0,
            )
            return KryptonLiveScanResult(symbol, timeframe, None, None, None, plan, tel, oos)

        d = add_indicators(d, p)
        n = len(d)
        last = n - 1
        c = d["close"].to_numpy(float)
        h = d["high"].to_numpy(float)
        l = d["low"].to_numpy(float)
        atr = d["atr"].to_numpy(float)
        sma = float(d["sma"].iloc[-1])

        brk_next = float(np.max(c[n - p.entry_days :]))
        exit_next = float(np.min(c[n - p.exit_days :]))
        trend_up = bool(c[last] > sma)
        atr_now = float(atr[last])
        tel.update(
            {
                "atr14": round(atr_now, prec),
                "trend": "UP" if trend_up else "DOWN",
                "breakout_long_level": round(brk_next, prec),
                "breakout_short_level": round(float(np.min(c[n - p.entry_days :])), prec),
                "exit_channel_long": round(exit_next, prec),
                "exit_channel_short": round(float(np.max(c[n - p.exit_days :])), prec),
                "entry_lookback": p.entry_days,
                "exit_lookback": p.exit_days,
                "init_stop_atr": p.init_stop_atr,
                "trail_atr": p.trail_atr,
                "last_closed_bar": str(d.index[last].date()),
                "sma150": round(sma, prec),
            }
        )

        active, bars_ago, cand = None, None, None
        if reliable and oos.get("validated"):
            for i, side in sorted(entry_signals(d, p, start=max(0, n - 3)), reverse=True):
                is_long = side == "LONG"
                r = p.init_stop_atr * atr[i]
                trail = c[i] - r if is_long else c[i] + r
                peak = c[i]
                alive = True
                for k in range(i + 1, n):
                    if (is_long and l[k] <= trail) or ((not is_long) and h[k] >= trail):
                        alive = False
                        break
                    if (is_long and c[k] < d["lo_exit"].iloc[k]) or ((not is_long) and c[k] > d["hi_exit"].iloc[k]):
                        alive = False
                        break
                    peak = max(peak, h[k]) if is_long else min(peak, l[k])
                    trail = (
                        max(trail, peak - p.trail_atr * atr[k])
                        if is_long
                        else min(trail, peak + p.trail_atr * atr[k])
                    )
                run = (live_price - c[i]) if is_long else (c[i] - live_price)
                room = (live_price - trail) if is_long else (trail - live_price)
                if not alive or run > 0.5 * r or room < 0.3 * r:
                    continue
                active_sizing = self.compute_position_sizing(
                    symbol, side, live_price, abs(live_price - trail) / live_price * 100.0
                )
                lev, liq = int(active_sizing["leverage"]), float(active_sizing["liq_price"])
                tel["active_sizing"] = active_sizing
                sg = 1.0 if is_long else -1.0
                active = ReversalSetupSignal(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=str(d.index[i]),
                    impulse_direction=f"EQUITIES_TREND_BREAKOUT_{'UP' if is_long else 'DOWN'}",
                    trade_side=side,
                    entry_price=round(live_price, prec),
                    stop_loss=round(trail, prec),
                    tp1_price=round(live_price + sg * 3 * r, prec),
                    tp2_price=round(live_price + sg * 6 * r, prec),
                    impulse_pct=round((c[i] / c[max(0, i - p.entry_days)] - 1) * 100, 3),
                    impulse_atr_mult=round(
                        ((c[i] - d["hi_entry"].iloc[i]) if is_long else (d["lo_entry"].iloc[i] - c[i]))
                        / max(atr[i], 1e-12),
                        2,
                    ),
                    flatline_bars=0,
                    flatline_range_pct=round(r / live_price * 100, 3),
                    flatline_range_atr=p.init_stop_atr,
                    body_decay_ratio=0.0,
                    rsi_at_flatline=float(tel.get("rsi14", 50.0)),
                    volume_decay_ratio=float(tel.get("vol_ratio", 1.0)),
                    risk_reward_tp1=3.0,
                    risk_reward_tp2=6.0,
                    recommended_leverage=lev,
                    estimated_liq_price=round(liq, prec),
                )
                bars_ago = last - i
                cand = KryptonCandidateInfo(
                    archetype="EQUITIES_TREND_BREAKOUT",
                    partial_frac=0.25,
                    trail_atr_mult=p.trail_atr,
                    meta_pred_r=tel["meta_pred_r"],
                    meta_win_prob=tel["meta_win_prob_pct"] / 100.0,
                )
                break

        if trend_up or not p.allow_shorts:
            w_side, trig = "LONG", brk_next
            sl_w = trig - p.init_stop_atr * atr_now
            w_exit = exit_next
            order_type = "ENTER_ON_CLOSE_BEYOND_LEVEL" if trend_up else "NO_TREND_STAND_ASIDE"
        else:
            w_side, trig = "SHORT", float(np.min(c[n - p.entry_days :]))
            sl_w = trig + p.init_stop_atr * atr_now
            w_exit = float(np.max(c[n - p.exit_days :]))
            order_type = "ENTER_ON_CLOSE_BEYOND_LEVEL"
        plan = self._plan(
            w_side,
            trig,
            sl_w,
            live_price,
            prec,
            symbol,
            order_type=order_type,
            expected_r=tel["meta_pred_r"],
            exit_channel=w_exit,
        )
        tel["dist_to_trap_pct"] = round(abs(live_price - trig) / live_price * 100.0, 2)
        if active is not None:
            tel["archetype"] = "TREND_BREAKOUT"
            tel["active_stop"] = active.stop_loss
            ex_lvl = exit_next if active.trade_side == "LONG" else float(np.max(c[n - p.exit_days :]))
            tel["exit_rule"] = (
                f"Close if a daily close (00:00 UTC) is {'below' if active.trade_side == 'LONG' else 'above'} "
                f"{ex_lvl:.{prec}f}"
            )
        else:
            tel["archetype"] = "BREAKOUT_WATCH" if (trend_up or p.allow_shorts) else "NO_TREND"
        return KryptonLiveScanResult(symbol, timeframe, active, bars_ago, cand, plan, tel, oos)

    # ---------------------------------------------------------------- exits
    def check_position_exit(
        self, df: pd.DataFrame, pos: Any, timeframe: str
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        from detector import ExitAlertSignal

        p = self.p
        is_long = pos.side == "LONG"
        live = float(df["close"].iloc[-1])
        prec = self._price_precision(live)
        entry, lev, margin = float(pos.entry_price), int(pos.leverage), float(pos.remaining_margin_usd)
        sl = float(pos.stop_loss)
        risk = max(abs(entry - sl), entry * 0.002)

        def mk(reason: str, urgency: str, px: float, why: str) -> ExitAlertSignal:
            mv = (px - entry) if is_long else (entry - px)
            roe = mv / entry * 100.0 * lev
            return ExitAlertSignal(
                symbol=pos.symbol,
                position_side=pos.side,
                reason=reason,
                urgency=urgency,
                entry_price=entry,
                current_price=round(px, prec),
                unrealized_pnl_pct_on_margin=round(roe, 2),
                unrealized_pnl_usd=round(margin * roe / 100.0, 2),
                r_multiple=round(mv / max(risk, 1e-12), 2),
                explanation=why,
            )

        if (is_long and live <= sl) or ((not is_long) and live >= sl):
            in_profit = (sl > entry) if is_long else (sl < entry)
            return (
                mk(
                    "TRAIL_STOP_HIT" if in_profit else "STOP_LOSS_HIT",
                    "STOPPED_OUT",
                    sl,
                    f"Equities Trend {'trailing ' if in_profit else ''}stop {sl:.{prec}f} crossed — close the position.",
                ),
                None,
            )

        d = self._daily(pos.symbol, df)
        if d.empty or len(d) < p.exit_days + p.atr_days + 2:
            return None, None
        d = add_indicators(d, p)
        try:
            opened = pd.to_datetime(pos.opened_at, utc=True)
        except Exception:
            opened = d.index[-1]
        closed_after = d[(d.index + pd.Timedelta(days=1)) > opened]
        if len(closed_after):
            k = d.index[-1]
            c_k = float(d["close"].iloc[-1])
            lvl = float(d["lo_exit"].iloc[-1]) if is_long else float(d["hi_exit"].iloc[-1])
            if k in closed_after.index and ((is_long and c_k < lvl) or ((not is_long) and c_k > lvl)):
                return (
                    mk(
                        "EQUITIES_CHANNEL_EXIT",
                        "IMMEDIATE_CLOSE",
                        live,
                        f"Daily close {c_k:.{prec}f} went {'below' if is_long else 'above'} the {p.exit_days}-day closing "
                        f"{'low' if is_long else 'high'} ({lvl:.{prec}f}) — trend over, close now.",
                    ),
                    None,
                )
            atr_k = float(d["atr"].iloc[-1])
            ts = pd.to_datetime(df["timestamp"], utc=True)
            since = df[ts >= opened]
            hi_since = max(float(closed_after["high"].max()), float(since["high"].max()) if len(since) else -np.inf)
            lo_since = min(float(closed_after["low"].min()), float(since["low"].min()) if len(since) else np.inf)
            new_sl = (
                max(sl, hi_since - p.trail_atr * atr_k)
                if is_long
                else min(sl, lo_since + p.trail_atr * atr_k)
            )
            if abs(new_sl - sl) >= 0.5 * atr_k:
                pos.stop_loss = round(new_sl, prec)
                return None, {"old_sl": round(sl, prec), "new_sl": round(new_sl, prec)}
        return None, None
