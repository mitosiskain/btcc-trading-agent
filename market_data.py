"""
Market Data Engine for Crypto (ETH, BTC, SOL) and Tokenized RWAs (Gold/XAU, NVDA, TSLA).
Fetches 1m / 5m / 15m intraday OHLCV data from public futures/spot APIs (Binance Perps, Bybit,
Yahoo Finance intraday for tokenized RWAs) and computes deterministic technical metrics
(ATR, RSI, Volume Z-score, Candle Body & Wick ratios).
"""

from __future__ import annotations
import time
from typing import Optional
import numpy as np
import pandas as pd
import requests

from config import DEFAULT_ASSETS, AssetProfile


class MarketDataFeed:
    """Unified OHLCV fetcher + indicator calculator for Crypto & Tokenized RWA pairs."""

    BINANCE_FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"
    BINANCE_FAPI_PREMIUM = "https://fapi.binance.com/fapi/v1/premiumIndex"
    BINANCE_SPOT_KLINES = "https://api.binance.com/api/v3/klines"
    BYBIT_KLINES = "https://api.bybit.com/v5/market/kline"
    YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    def __init__(self, timeout: int = 10):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BTCC-AI-Trading-Agent/2.0-Carbon"})
        self.timeout = timeout
        self._macro_cache: dict[str, tuple[float, dict]] = {}
        self._funding_cache: dict[str, tuple[float, float]] = {}
        self._btc_ref_cache: dict[str, tuple[float, pd.DataFrame]] = {}

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "5m",
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV DataFrame with columns:
        ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        profile: Optional[AssetProfile] = DEFAULT_ASSETS.get(symbol.upper())
        binance_sym = profile.binance_symbol if profile else symbol.upper()
        yf_ticker = profile.yfinance_ticker if profile else symbol.upper()

        df = pd.DataFrame()

        # 1. Try Bybit Linear Perps or Binance Futures/Spot for Crypto & PAXG (Tokenized Gold)
        if binance_sym:
            df = self._fetch_binance(binance_sym, interval, limit)
            if df.empty:
                df = self._fetch_bybit(binance_sym, interval, limit)

        # 2. Fallback to Yahoo Finance (or deeper macro history when 1M limit exceeds Spot history)
        if yf_ticker and (df.empty or (interval == "1M" and len(df) < limit)):
            df_yh = self._fetch_yahoo(yf_ticker, interval, limit)
            if len(df_yh) > len(df):
                df = df_yh

        if df.empty:
            raise RuntimeError(f"Could not fetch market data for {symbol} ({interval})")

        if "funding_rate" not in df.columns:
            df["funding_rate"] = self.fetch_funding_rate(binance_sym) if binance_sym else 0.0001

        return self.add_indicators(df)

    def fetch_funding_rate(self, binance_sym: str, ttl_seconds: int = 600) -> float:
        """Fetch live 8h perpetual funding rate with in-memory TTL cache."""
        if not binance_sym:
            return 0.0001
        now = time.time()
        cached = self._funding_cache.get(binance_sym)
        if cached and (now - cached[0]) < ttl_seconds:
            return cached[1]
        for cand in (binance_sym, f"1000{binance_sym}"):
            try:
                resp = self.session.get(
                    self.BINANCE_FAPI_PREMIUM,
                    params={"symbol": cand},
                    timeout=min(4, self.timeout),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    fr = float(data.get("lastFundingRate", 0.0001))
                    self._funding_cache[binance_sym] = (now, fr)
                    return fr
            except Exception:
                continue
        self._funding_cache[binance_sym] = (now, 0.0001)
        return 0.0001

    def fetch_btc_reference(self, interval: str = "4h", limit: int = 240, ttl_seconds: int = 45) -> Optional[pd.DataFrame]:
        """Fetch synchronized BTCUSDT reference klines (cached per scan cycle) for cross-asset beta residual."""
        cache_key = f"{interval}:{limit}"
        now = time.time()
        cached = self._btc_ref_cache.get(cache_key)
        if cached and (now - cached[0]) < ttl_seconds:
            return cached[1]
        try:
            btc_df = self.fetch_ohlcv("BTCUSDT", interval=interval, limit=limit)
            self._btc_ref_cache[cache_key] = (now, btc_df)
            return btc_df
        except Exception:
            return None

    def fetch_enriched_ohlcv(
        self,
        symbol: str,
        interval: str = "4h",
        limit: int = 240,
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Fetch 12-column perpetual OHLCV + Orderflow + Funding + synchronized BTC reference
        ready for CarbonFeatureEngine.compute_features().
        """
        df = self.fetch_ohlcv(symbol=symbol, interval=interval, limit=limit)
        btc_df = df if symbol.upper() == "BTCUSDT" else self.fetch_btc_reference(interval=interval, limit=limit)
        return df, btc_df

    def _fetch_binance(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        urls = (
            (self.BINANCE_SPOT_KLINES, self.BINANCE_FAPI_KLINES)
            if interval in ("1w", "1M")
            else (self.BINANCE_FAPI_KLINES, self.BINANCE_SPOT_KLINES)
        )
        sym_candidates = [symbol]
        if not symbol.startswith("1000") and symbol in ("PEPEUSDT", "SHIBUSDT", "BONKUSDT", "FLOKIUSDT"):
            sym_candidates.insert(0, f"1000{symbol}")

        for base_url in urls:
            for cand_sym in sym_candidates:
                if base_url == self.BINANCE_SPOT_KLINES and cand_sym.startswith("1000"):
                    continue
                scale = 0.001 if (cand_sym.startswith("1000") and not symbol.startswith("1000")) else 1.0
                try:
                    resp = self.session.get(
                        base_url,
                        params={"symbol": cand_sym, "interval": interval, "limit": min(limit, 1500)},
                        timeout=self.timeout,
                    )
                    if resp.status_code != 200:
                        continue
                    raw = resp.json()
                    if not isinstance(raw, list) or not raw:
                        continue
                    rows = [
                        {
                            "timestamp": pd.to_datetime(int(r[0]), unit="ms", utc=True),
                            "open": float(r[1]) * scale,
                            "high": float(r[2]) * scale,
                            "low": float(r[3]) * scale,
                            "close": float(r[4]) * scale,
                            "volume": float(r[5]) / scale,
                            "quote_vol": float(r[7]) if len(r) > 7 else (float(r[4]) * float(r[5])),
                            "num_trades": float(r[8]) if len(r) > 8 else 1.0,
                            "taker_buy_vol": (float(r[9]) / scale) if len(r) > 9 else (0.5 * float(r[5]) / scale),
                            "taker_buy_quote_vol": float(r[10]) if len(r) > 10 else (0.5 * float(r[4]) * float(r[5])),
                        }
                        for r in raw
                    ]
                    out_df = pd.DataFrame(rows)
                    out_df.attrs["has_true_orderflow"] = True
                    return out_df
                except Exception:
                    continue
        return pd.DataFrame()

    def _fetch_bybit(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        bybit_interval_map = {
            "1m": "1",
            "3m": "3",
            "5m": "5",
            "15m": "15",
            "1h": "60",
            "4h": "240",
            "1d": "D",
            "1w": "W",
            "1M": "M",
        }
        bybit_iv = bybit_interval_map.get(interval, "5")
        try:
            resp = self.session.get(
                self.BYBIT_KLINES,
                params={
                    "category": "linear",
                    "symbol": symbol,
                    "interval": bybit_iv,
                    "limit": min(limit, 1000),
                },
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return pd.DataFrame()
            data = resp.json()
            lst = data.get("result", {}).get("list", [])
            if not lst:
                return pd.DataFrame()
            rows = []
            for r in reversed(lst):
                o_v, h_v, l_v, c_v, vol_v = float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
                qv_v = float(r[6]) if len(r) > 6 else (c_v * vol_v)
                clv = ((2.0 * c_v - h_v - l_v) / max(h_v - l_v, 1e-9)) if h_v > l_v else 0.0
                tb_frac = max(0.18, min(0.82, 0.50 + 0.28 * clv))
                rows.append(
                    {
                        "timestamp": pd.to_datetime(int(r[0]), unit="ms", utc=True),
                        "open": o_v,
                        "high": h_v,
                        "low": l_v,
                        "close": c_v,
                        "volume": vol_v,
                        "quote_vol": qv_v,
                        "num_trades": max(1.0, vol_v * 0.1),
                        "taker_buy_vol": vol_v * tb_frac,
                        "taker_buy_quote_vol": qv_v * tb_frac,
                    }
                )
            out_df = pd.DataFrame(rows)
            out_df.attrs["has_true_orderflow"] = False
            return out_df
        except Exception:
            return pd.DataFrame()

    def _fetch_yahoo(self, ticker: str, interval: str, limit: int) -> pd.DataFrame:
        resample_4h = interval == "4h"
        if resample_4h:
            yh_interval = "1h"
        elif interval == "1w":
            yh_interval = "1wk"
        elif interval == "1M":
            yh_interval = "1mo"
        else:
            yh_interval = interval

        if yh_interval in ("1m", "5m"):
            range_str = "5d" if yh_interval == "1m" else "1mo"
        elif yh_interval == "15m":
            range_str = "1mo"
        elif yh_interval == "1h":
            range_str = "2y"
        elif yh_interval == "1mo":
            range_str = "max"
        elif yh_interval == "1wk":
            range_str = "10y"
        else:
            range_str = "5y"
        try:
            resp = self.session.get(
                self.YAHOO_CHART_URL.format(ticker=ticker),
                params={"interval": yh_interval, "range": range_str},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return pd.DataFrame()
            res = resp.json().get("chart", {}).get("result", [])
            if not res:
                return pd.DataFrame()
            # Some Yahoo crypto pairs downgrade 'max' + '1mo' to '1wk'; fallback to '10y' for true 1mo bars
            if yh_interval == "1mo" and res[0].get("meta", {}).get("dataGranularity") != "1mo":
                resp = self.session.get(
                    self.YAHOO_CHART_URL.format(ticker=ticker),
                    params={"interval": yh_interval, "range": "10y"},
                    timeout=self.timeout,
                )
                if resp.status_code != 200:
                    return pd.DataFrame()
                res = resp.json().get("chart", {}).get("result", [])
                if not res:
                    return pd.DataFrame()
            timestamps = res[0].get("timestamp", [])
            quote = res[0].get("indicators", {}).get("quote", [{}])[0]
            rows = []
            for i, ts in enumerate(timestamps):
                o = quote.get("open", [None])[i]
                h = quote.get("high", [None])[i]
                l = quote.get("low", [None])[i]
                c = quote.get("close", [None])[i]
                v = quote.get("volume", [0])[i]
                if None in (o, h, l, c):
                    continue
                o_v, h_v, l_v, c_v, vol_v = float(o), float(h), float(l), float(c), float(v or 0.0)
                qv_v = c_v * vol_v
                clv = ((2.0 * c_v - h_v - l_v) / max(h_v - l_v, 1e-9)) if h_v > l_v else 0.0
                tb_frac = max(0.18, min(0.82, 0.50 + 0.28 * clv))
                rows.append(
                    {
                        "timestamp": pd.to_datetime(int(ts), unit="s", utc=True),
                        "open": o_v,
                        "high": h_v,
                        "low": l_v,
                        "close": c_v,
                        "volume": vol_v,
                        "quote_vol": qv_v,
                        "num_trades": max(1.0, vol_v * 0.05),
                        "taker_buy_vol": vol_v * tb_frac,
                        "taker_buy_quote_vol": qv_v * tb_frac,
                    }
                )
            df = pd.DataFrame(rows)
            if df.empty:
                return df
            if resample_4h:
                df = (
                    df.set_index("timestamp")
                    .resample("4h")
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                            "quote_vol": "sum",
                            "num_trades": "sum",
                            "taker_buy_vol": "sum",
                            "taker_buy_quote_vol": "sum",
                        }
                    )
                    .dropna(subset=["open", "high", "low", "close"])
                    .reset_index()
                )
            out_df = df.tail(limit).reset_index(drop=True)
            out_df.attrs["has_true_orderflow"] = False
            return out_df
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def add_indicators(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
        """Compute strictly causal Wilder's RMA ATR/RSI and trend geometry."""
        orig_attrs = dict(df.attrs)
        df = df.copy().reset_index(drop=True)
        df.attrs.update(orig_attrs)
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1).fillna(df["open"])

        # True Range & Wilder's RMA ATR
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1.0 / float(max(1, atr_period)), adjust=False).mean()

        # Candle geometry
        df["body"] = (df["close"] - df["open"]).abs()
        df["range"] = (df["high"] - df["low"]).replace(0, 1e-9)
        df["body_ratio"] = df["body"] / df["range"]
        df["ret_pct"] = close.pct_change() * 100.0

        # True Wilder's RMA RSI (14)
        delta = close.diff().fillna(0.0)
        gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        rs = gain / loss.replace(0, 1e-9)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # Relative volume (vs 24-bar EMA)
        vol_ma = df["volume"].ewm(span=24, adjust=False).mean().replace(0, 1.0)
        df["vol_ratio"] = df["volume"] / vol_ma

        # Exponential Moving Averages for Macro Trend detection (Days, Weeks, Months)
        df["ema20"] = close.ewm(span=20, adjust=False).mean()
        df["ema50"] = close.ewm(span=50, adjust=False).mean()

        return df

    def get_macro_trend_compass(self, symbol: str, ttl_seconds: int = 900) -> dict:
        """
        Evaluates Daily (1d), Weekly (1w), and Monthly (1M) trend direction for `symbol`
        using EMA20, EMA50, and RSI(14). Cached in-memory for `ttl_seconds` (5 min default).
        """
        now = time.time()
        cached = self._macro_cache.get(symbol)
        if cached and (now - cached[0]) < ttl_seconds:
            return cached[1]

        compass = {}
        bull_score = 0
        bear_score = 0

        for tf, limit_n, weight in (("1d", 90, 1), ("1w", 60, 1), ("1M", 36, 1)):
            try:
                df_tf = self.fetch_ohlcv(symbol, interval=tf, limit=limit_n)
                if len(df_tf) >= 5:
                    last = df_tf.iloc[-1]
                    c = float(last["close"])
                    ema20 = float(last["ema20"])
                    ema50 = float(last["ema50"])
                    rsi = float(last["rsi"])
                    ret_prev = float(last["ret_pct"]) if not pd.isna(last["ret_pct"]) else 0.0
                    if c > ema20 and rsi >= 48.0:
                        state = "BULLISH"
                        bull_score += weight
                    elif c < ema20 and rsi <= 52.0:
                        state = "BEARISH"
                        bear_score += weight
                    else:
                        state = "NEUTRAL"
                    compass[tf] = {
                        "trend": state,
                        "rsi": round(rsi, 1),
                        "vs_ema20_pct": round(((c - ema20) / max(ema20, 1e-9)) * 100.0, 2),
                        "last_bar_pct": round(ret_prev, 2),
                    }
                    if tf == "1d" and len(df_tf) >= 10:
                        b_atr = float(last["atr"]) or (c * 0.02)
                        b_hi = float(df_tf["high"].iloc[-5:-1].max())
                        b_lo = float(df_tf["low"].iloc[-5:-1].min())
                        b_rng = max(b_hi - b_lo, 1e-9)
                        t_span = max(2.5 * b_atr, 2.2 * b_rng)
                        is_gold = any(k in symbol.upper() for k in ("XAU", "GOLD", "XAG", "SILVER"))
                        side_1d = "LONG" if (is_gold or abs(b_hi - c) <= abs(c - b_lo)) else "SHORT"
                        prec = 8 if c < 0.001 else (5 if c < 1.0 else 4)
                        if side_1d == "LONG":
                            e_1d = round(b_hi + 0.03 * b_atr, prec)
                            sl_1d = round((b_hi - 0.65 * b_rng) - 0.35 * b_atr, prec)
                            tp1_1d = round(e_1d + 0.382 * t_span, prec)
                            tp2_1d = round(e_1d + 0.618 * t_span, prec)
                        else:
                            e_1d = round(b_lo - 0.03 * b_atr, prec)
                            sl_1d = round((b_lo + 0.65 * b_rng) + 0.35 * b_atr, prec)
                            tp1_1d = round(e_1d - 0.382 * t_span, prec)
                            tp2_1d = round(e_1d - 0.618 * t_span, prec)
                        compass["1d_plan"] = {
                            "side": side_1d,
                            "trigger_entry_price": e_1d,
                            "stop_loss": sl_1d,
                            "tp1_price": tp1_1d,
                            "tp2_price": tp2_1d,
                        }
                else:
                    compass[tf] = {"trend": "NEUTRAL", "rsi": 50.0, "vs_ema20_pct": 0.0, "last_bar_pct": 0.0}
            except Exception:
                compass[tf] = {"trend": "NEUTRAL", "rsi": 50.0, "vs_ema20_pct": 0.0, "last_bar_pct": 0.0}

        if bull_score >= 2 and bull_score > bear_score:
            dom = "BULLISH"
        elif bear_score >= 2 and bear_score > bull_score:
            dom = "BEARISH"
        else:
            dom = "MIXED"

        d_str = compass.get("1d", {}).get("trend", "NEUTRAL")
        w_str = compass.get("1w", {}).get("trend", "NEUTRAL")
        m_str = compass.get("1M", {}).get("trend", "NEUTRAL")
        result = {
            "symbol": symbol,
            "1d": compass.get("1d", {}),
            "1w": compass.get("1w", {}),
            "1M": compass.get("1M", {}),
            "1d_plan": compass.get("1d_plan"),
            "dominant_trend": dom,
            "summary": f"1D: {d_str} • 1W: {w_str} • 1M: {m_str}",
        }
        self._macro_cache[symbol] = (now, result)
        return result
