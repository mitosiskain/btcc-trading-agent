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
    BINANCE_SPOT_KLINES = "https://api.binance.com/api/v3/klines"
    BYBIT_KLINES = "https://api.bybit.com/v5/market/kline"
    YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    def __init__(self, timeout: int = 10):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BTCC-AI-Trading-Agent/1.0"})
        self.timeout = timeout
        self._macro_cache: dict[str, tuple[float, dict]] = {}

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

        return self.add_indicators(df)

    def _fetch_binance(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        urls = (
            (self.BINANCE_SPOT_KLINES, self.BINANCE_FAPI_KLINES)
            if interval in ("1w", "1M")
            else (self.BINANCE_FAPI_KLINES, self.BINANCE_SPOT_KLINES)
        )
        for base_url in urls:
            try:
                resp = self.session.get(
                    base_url,
                    params={"symbol": symbol, "interval": interval, "limit": min(limit, 1500)},
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
                        "open": float(r[1]),
                        "high": float(r[2]),
                        "low": float(r[3]),
                        "close": float(r[4]),
                        "volume": float(r[5]),
                    }
                    for r in raw
                ]
                return pd.DataFrame(rows)
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
            rows = [
                {
                    "timestamp": pd.to_datetime(int(r[0]), unit="ms", utc=True),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                }
                for r in reversed(lst)
            ]
            return pd.DataFrame(rows)
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
                rows.append(
                    {
                        "timestamp": pd.to_datetime(int(ts), unit="s", utc=True),
                        "open": float(o),
                        "high": float(h),
                        "low": float(l),
                        "close": float(c),
                        "volume": float(v or 0.0),
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
                        }
                    )
                    .dropna(subset=["open", "high", "low", "close"])
                    .reset_index()
                )
            return df.tail(limit).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def add_indicators(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
        """Compute deterministic indicators needed for Impulse + Flatline detection."""
        df = df.copy().reset_index(drop=True)
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        # True Range & ATR
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=atr_period, min_periods=1).mean()

        # Candle geometry
        df["body"] = (df["close"] - df["open"]).abs()
        df["range"] = (df["high"] - df["low"]).replace(0, 1e-9)
        df["body_ratio"] = df["body"] / df["range"]
        df["ret_pct"] = close.pct_change() * 100.0

        # RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0.0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.clip(upper=0.0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-9)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # Relative volume (vs 20-bar mean)
        vol_ma = df["volume"].rolling(window=20, min_periods=1).mean().replace(0, 1.0)
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
