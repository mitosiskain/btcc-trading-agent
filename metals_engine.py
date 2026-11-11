"""
Metals Trend V1 — dedicated gold & silver engine (XAUUSDT / XAGUSDT)
=====================================================================

Why a separate engine?
  Gold and silver only have long price histories as *session* data (COMEX futures ~23h x 5 days); BTCC trades them
  as 24/7 perpetuals. Krypton V3's bar-count rules (tuned on crypto) behave differently on those two kinds of data,
  and in testing Krypton was weak on metals (silver 24/7 perp PF 0.81, gold futures PF 1.11).

Calendar-consistent design (identical on session data and on 24/7 data):
  * Every decision is taken once per WEEKDAY at the UTC daily close (00:00 UTC), from a weekday-only daily close
    series. On 24/7 feeds weekend drift simply shows up in Monday's change, so the signal series is the same.
  * Only execution differs with 24/7 data: entries/exits/stops can also happen on weekends.

Rules (chosen on 2000-2015 COMEX futures only; 97% of 36 trend variants were profitable there; mean-reversion
variants were not, 17%):
  ENTRY  long when the daily close breaks above the prior 40-weekday closing high (first cross) and is above
         the 200-day average. Long-only: shorts reduced profit factor in development.
  STOP   2 x ATR(20 days) initial stop (= 1R), then a chandelier trail 3 x ATR below the highest high since entry.
  EXIT   daily close below the prior 20-weekday closing low.
Evidence: benchmark_metals.py (futures 2000-2026, PAXG 24/7 2020-2026, Binance XAU/XAG perps 2025-26).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from krypton_engine import (
    KryptonLiveBridge,
    KryptonLiveScanResult,
    KryptonCandidateInfo,
    simulate_trade_fast,
    compute_features,
    KryptonParams,
)

METALS_SYMBOLS = {"XAUUSDT": "GC=F", "XAGUSDT": "SI=F"}


@dataclass
class MetalsParams:
    entry_days: int = 40        # breakout of the prior 40-weekday closing high
    exit_days: int = 20         # exit on a close below the prior 20-weekday closing low
    sma_days: int = 200         # trend filter
    atr_days: int = 20
    init_stop_atr: float = 2.0
    trail_atr: float = 3.0
    allow_shorts: bool = False


# --------------------------------------------------------------------------------------
# Shared core (used by both the live bridge and benchmark_metals.py)
# --------------------------------------------------------------------------------------
def weekday_daily(ex: pd.DataFrame) -> pd.DataFrame:
    """UTC daily OHLC from any bar size (weekday days only) + index of the execution bar that closes each day."""
    d = ex[["timestamp", "open", "high", "low", "close"]].copy()
    d["date"] = pd.to_datetime(d["timestamp"], utc=True).dt.floor("D")
    d["row"] = np.arange(len(d))
    g = d.groupby("date")
    daily = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                          "close": g["close"].last(), "last_idx": g["row"].last()})
    return daily[daily.index.dayofweek < 5].copy()


def add_indicators(daily: pd.DataFrame, p: MetalsParams) -> pd.DataFrame:
    out = daily.copy()
    c = out["close"].to_numpy(float); h = out["high"].to_numpy(float); l = out["low"].to_numpy(float)
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


def entry_signals(d: pd.DataFrame, p: MetalsParams, start: int = 0) -> List[Tuple[int, str]]:
    c = d["close"].to_numpy(float); sma = d["sma"].to_numpy(float)
    hi = d["hi_entry"].to_numpy(float); lo = d["lo_entry"].to_numpy(float)
    out = []
    for i in range(max(start, p.sma_days + 1), len(d)):
        if not (np.isfinite(hi[i]) and np.isfinite(sma[i]) and np.isfinite(hi[i - 1])):
            continue
        if c[i] > hi[i] and c[i - 1] <= hi[i - 1] and c[i] > sma[i]:
            out.append((i, "LONG"))
        elif p.allow_shorts and c[i] < lo[i] and c[i - 1] >= lo[i - 1] and c[i] < sma[i]:
            out.append((i, "SHORT"))
    return out


def stitch_history(daily_live: pd.DataFrame, daily_hist: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, int]:
    """Prepend level-matched session history (e.g. COMEX futures) before the first 24/7 day for indicator warm-up."""
    if daily_hist is None or daily_hist.empty or daily_live.empty:
        return daily_live, 0
    first = daily_live.index[0]
    ov = daily_hist.index.intersection(daily_live.index)
    ratio = float(daily_live.loc[ov[:5], "close"].mean() / daily_hist.loc[ov[:5], "close"].mean()) if len(ov) else 1.0
    hist = daily_hist[daily_hist.index < first][["open", "high", "low", "close"]] * ratio
    hist = hist[hist.index.dayofweek < 5].copy()
    hist["last_idx"] = -1
    return pd.concat([hist, daily_live]), len(hist)


def backtest(ex: pd.DataFrame, symbol: str, p: MetalsParams = MetalsParams(), *, daily_hist: Optional[pd.DataFrame] = None,
             funding_mode: str = "elapsed", start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None,
             max_btcc_leverage: int = 150) -> List[Dict[str, Any]]:
    """Execution on `ex` bars (any size, 24/7 or session) with the benchmark friction model."""
    dl = weekday_daily(ex)
    d, off = stitch_history(dl, daily_hist)
    d = add_indicators(d, p)
    ex2 = ex.copy()
    if "funding_rate" not in ex2.columns:
        ex2["funding_rate"] = 0.0001
    f = compute_features(ex2, KryptonParams())
    if funding_mode == "elapsed":  # 0.01% per 8h of real elapsed time (handles nights/weekends correctly)
        ns = pd.to_datetime(ex2["timestamp"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
        hrs = np.diff(ns, prepend=ns[0]) / 3.6e12
        f["funding_bar"] = np.ones(len(ex2), dtype=bool)
        f["funding_rate"] = 0.0001 * hrs / 8.0
    n_ex = len(ex2)
    li = d["last_idx"].to_numpy(int); atr_d = d["atr"].to_numpy(float)
    atr_ex = np.full(n_ex, np.nan)
    for j in range(off, len(li)):
        lo = li[j] + 1; hi = li[j + 1] + 1 if j + 1 < len(li) else n_ex
        atr_ex[lo:hi] = atr_d[j]
    atr_ex = pd.Series(atr_ex).bfill().ffill().to_numpy()
    c = d["close"].to_numpy(float)
    ex_long = c < d["lo_exit"].to_numpy(float); ex_short = c > d["hi_exit"].to_numpy(float)
    trades, busy = [], -1
    for i, side in entry_signals(d, p, start=off):
        sidx = int(li[i]); ts = pd.to_datetime(ex2["timestamp"].iloc[sidx], utc=True)
        if sidx <= busy or (start is not None and ts < start) or (end is not None and ts >= end):
            continue
        f["ex_lo"] = np.full(n_ex, -np.inf); f["ex_hi"] = np.full(n_ex, np.inf)
        flags = ex_long if side == "LONG" else ex_short
        for j in np.where(flags & (np.arange(len(li)) > i))[0]:
            if side == "LONG":
                f["ex_lo"][li[j]] = np.inf
            else:
                f["ex_hi"][li[j]] = -np.inf
        t = simulate_trade_fast(f, symbol, sidx, side, p.init_stop_atr * atr_d[i], max_hold_bars=10 ** 7,
                                trail_atr_mult=p.trail_atr, atr_override=atr_ex, max_btcc_leverage=max_btcc_leverage)
        if t is not None:
            t["archetype"] = "METALS_TREND"
            trades.append(t)
            busy = t["exit_idx"]
    return trades


# --------------------------------------------------------------------------------------
# Live bridge (same public surface as KryptonLiveBridge)
# --------------------------------------------------------------------------------------
class MetalsTrendBridge(KryptonLiveBridge):
    engine_name = "METALS_TREND"
    display_name = "Metals Trend"

    def __init__(self, risk_cfg: Any, state_dir: Optional[Any] = None, feed: Any = None):
        super().__init__(risk_cfg, state_dir=state_dir)
        import json

        import os

        # Long-only by default (shorts lost money on COMEX 2000-2015 and silver 2016-2025, although they were positive
        # in the 2025-12 -> now hold-out). Opt in with METALS_ALLOW_SHORTS=true.
        self.p = MetalsParams(allow_shorts=os.getenv("METALS_ALLOW_SHORTS", "false").lower() == "true")
        self.feed = feed
        self._daily_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self.metals_stats: Dict[str, Any] = {}
        path = self.state_dir / "benchmark_metals_results.json"
        if path.exists():
            try:
                self.metals_stats = json.loads(path.read_text())
            except Exception:
                self.metals_stats = {}

    def get_oos_asset_summary(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        s = (self.metals_stats.get("live_summary") or {}).get(symbol.upper()) or {}
        ok = int(s.get("total_trades", 0)) > 0
        return {
            "scope": "METALS_FUTURES_2000_2026" if ok else "UNVALIDATED",
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
        """Closed weekday daily bars from the 24/7 feed, warmed up with Yahoo session history when it is too short."""
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
                        yh = self.feed._fetch_yahoo(METALS_SYMBOLS.get(symbol.upper(), ""), "1d", 900)
                        if not yh.empty:
                            hist = weekday_daily(yh)
                            base, _ = stitch_history(base, hist) if not base.empty else (hist, 0)
                    except Exception:
                        pass
            self._daily_cache[symbol] = (now, base)
        if base.empty:
            return base
        today = pd.Timestamp.now(tz="UTC").floor("D")
        return base[base.index < today]  # closed days only

    # ---------------------------------------------------------------- entries
    def evaluate_symbol_live(self, df: pd.DataFrame, btc_df: Optional[pd.DataFrame], symbol: str, timeframe: str = "4h") -> KryptonLiveScanResult:
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
            "meta_win_prob_pct": round(float(oos.get("win_rate_pct", 40.0)), 1),
            "meta_threshold_r": 0.0,
            "kelly_risk_scale": 1.0,
            "has_true_orderflow": bool(df.attrs.get("has_true_orderflow", False)),
        }
        tel.update(self._orderflow_telemetry(df, btc_df))
        d = self._daily(symbol, df)
        if d.empty or len(d) < p.sma_days + p.entry_days + 2:
            atr_fb = float((df["high"] - df["low"]).tail(20).mean()) if len(df) else live_price * 0.02
            tel.update({"archetype": "INSUFFICIENT_HISTORY", "trend": "NONE", "atr14": round(atr_fb, prec), "dist_to_trap_pct": 99.0})
            plan = self._plan("LONG", live_price, live_price - 2 * atr_fb, live_price, prec, symbol,
                              order_type="NO_TRADE_INSUFFICIENT_HISTORY", expected_r=0.0)
            return KryptonLiveScanResult(symbol, timeframe, None, None, None, plan, tel, oos)

        d = add_indicators(d, p)
        n = len(d); last = n - 1
        c = d["close"].to_numpy(float); h = d["high"].to_numpy(float); l = d["low"].to_numpy(float)
        atr = d["atr"].to_numpy(float); sma = float(d["sma"].iloc[-1])
        # Levels that apply to the NEXT daily close (include the last closed day)
        brk_next = float(np.max(c[n - p.entry_days:]))
        exit_next = float(np.min(c[n - p.exit_days:]))
        trend_up = bool(c[last] > sma)
        atr_now = float(atr[last])
        tel.update({
            "atr14": round(atr_now, prec),
            "trend": "UP" if trend_up else "DOWN",
            "breakout_long_level": round(brk_next, prec),
            "breakout_short_level": round(float(np.min(c[n - p.entry_days:])), prec),
            "exit_channel_long": round(exit_next, prec),
            "exit_channel_short": round(float(np.max(c[n - p.exit_days:])), prec),
            "entry_lookback": p.entry_days, "exit_lookback": p.exit_days,
            "init_stop_atr": p.init_stop_atr, "trail_atr": p.trail_atr,
            "last_closed_bar": str(d.index[last].date()),
            "sma200": round(sma, prec),
        })

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
                        alive = False; break
                    if (is_long and c[k] < d["lo_exit"].iloc[k]) or ((not is_long) and c[k] > d["hi_exit"].iloc[k]):
                        alive = False; break
                    peak = max(peak, h[k]) if is_long else min(peak, l[k])
                    trail = max(trail, peak - p.trail_atr * atr[k]) if is_long else min(trail, peak + p.trail_atr * atr[k])
                run = (live_price - c[i]) if is_long else (c[i] - live_price)
                room = (live_price - trail) if is_long else (trail - live_price)
                if not alive or run > 0.5 * r or room < 0.3 * r:
                    continue
                lev, liq = self.compute_safe_leverage(symbol, side, live_price, abs(live_price - trail) / live_price * 100.0)
                sg = 1.0 if is_long else -1.0
                active = ReversalSetupSignal(
                    symbol=symbol, timeframe=timeframe, timestamp=str(d.index[i]),
                    impulse_direction=f"METALS_TREND_BREAKOUT_{'UP' if is_long else 'DOWN'}", trade_side=side,
                    entry_price=round(live_price, prec), stop_loss=round(trail, prec),
                    tp1_price=round(live_price + sg * 3 * r, prec), tp2_price=round(live_price + sg * 6 * r, prec),
                    impulse_pct=round((c[i] / c[max(0, i - p.entry_days)] - 1) * 100, 3),
                    impulse_atr_mult=round(((c[i] - d["hi_entry"].iloc[i]) if is_long else (d["lo_entry"].iloc[i] - c[i])) / max(atr[i], 1e-12), 2),
                    flatline_bars=0, flatline_range_pct=round(r / live_price * 100, 3), flatline_range_atr=p.init_stop_atr,
                    body_decay_ratio=0.0, rsi_at_flatline=float(tel.get("rsi14", 50.0)), volume_decay_ratio=float(tel.get("vol_ratio", 1.0)),
                    risk_reward_tp1=3.0, risk_reward_tp2=6.0, recommended_leverage=lev, estimated_liq_price=round(liq, prec),
                )
                bars_ago = last - i
                cand = KryptonCandidateInfo(archetype="METALS_TREND_BREAKOUT", partial_frac=0.25, trail_atr_mult=p.trail_atr,
                                            meta_pred_r=tel["meta_pred_r"], meta_win_prob=tel["meta_win_prob_pct"] / 100.0)
                break

        if trend_up or not p.allow_shorts:
            w_side, trig = "LONG", brk_next
            sl_w = trig - p.init_stop_atr * atr_now
            w_exit = exit_next
            order_type = "ENTER_ON_CLOSE_BEYOND_LEVEL" if trend_up else "NO_TREND_STAND_ASIDE"
        else:
            w_side, trig = "SHORT", float(np.min(c[n - p.entry_days:]))
            sl_w = trig + p.init_stop_atr * atr_now
            w_exit = float(np.max(c[n - p.exit_days:]))
            order_type = "ENTER_ON_CLOSE_BEYOND_LEVEL"
        plan = self._plan(w_side, trig, sl_w, live_price, prec, symbol, order_type=order_type, expected_r=tel["meta_pred_r"],
                          exit_channel=w_exit)
        tel["dist_to_trap_pct"] = round(abs(live_price - trig) / live_price * 100.0, 2)
        if active is not None:
            tel["archetype"] = "TREND_BREAKOUT"
            tel["active_stop"] = active.stop_loss
            ex_lvl = exit_next if active.trade_side == "LONG" else float(np.max(c[n - p.exit_days:]))
            tel["exit_rule"] = f"Close if a daily close (00:00 UTC) is {'below' if active.trade_side == 'LONG' else 'above'} {ex_lvl:.{prec}f}"
        else:
            tel["archetype"] = "BREAKOUT_WATCH" if (trend_up or p.allow_shorts) else "NO_TREND"
        return KryptonLiveScanResult(symbol, timeframe, active, bars_ago, cand, plan, tel, oos)

    # ---------------------------------------------------------------- exits
    def check_position_exit(self, df: pd.DataFrame, pos: Any, timeframe: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
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
            return ExitAlertSignal(symbol=pos.symbol, position_side=pos.side, reason=reason, urgency=urgency, entry_price=entry,
                                   current_price=round(px, prec), unrealized_pnl_pct_on_margin=round(roe, 2),
                                   unrealized_pnl_usd=round(margin * roe / 100.0, 2), r_multiple=round(mv / max(risk, 1e-12), 2),
                                   explanation=why)

        if (is_long and live <= sl) or ((not is_long) and live >= sl):
            in_profit = (sl > entry) if is_long else (sl < entry)
            return mk("TRAIL_STOP_HIT" if in_profit else "STOP_LOSS_HIT", "STOPPED_OUT", sl,
                      f"Metals Trend {'trailing ' if in_profit else ''}stop {sl:.{prec}f} crossed — close the position."), None

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
                return mk("METALS_CHANNEL_EXIT", "IMMEDIATE_CLOSE", live,
                          f"Daily close {c_k:.{prec}f} went {'below' if is_long else 'above'} the {p.exit_days}-day closing "
                          f"{'low' if is_long else 'high'} ({lvl:.{prec}f}) — trend over, close now."), None
            atr_k = float(d["atr"].iloc[-1])
            ts = pd.to_datetime(df["timestamp"], utc=True)
            since = df[ts >= opened]
            hi_since = max(float(closed_after["high"].max()), float(since["high"].max()) if len(since) else -np.inf)
            lo_since = min(float(closed_after["low"].min()), float(since["low"].min()) if len(since) else np.inf)
            new_sl = max(sl, hi_since - p.trail_atr * atr_k) if is_long else min(sl, lo_since + p.trail_atr * atr_k)
            if abs(new_sl - sl) >= 0.5 * atr_k:
                pos.stop_loss = round(new_sl, prec)
                return None, {"old_sl": round(sl, prec), "new_sl": round(new_sl, prec)}
        return None, None
