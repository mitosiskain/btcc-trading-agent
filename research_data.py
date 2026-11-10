"""
Full-history Binance USDT-M perpetual data loader for out-of-sample research.

Fetches the complete 12-column kline history (from listing date) plus the 8h funding-rate history
for every benchmark symbol and caches it under `state/research_cache/`. The schema is identical to
`benchmark_argon_vs_carbon.fetch_paginated_perp_data`, so every engine / simulator in this repo
can consume it unchanged.

Research protocol used with this data (see `benchmark_krypton.py`):
  * DEVELOPMENT  : every bar strictly before `DEV_CUTOFF` (2025-12-01 UTC)
  * HOLD-OUT TEST: the exact Argon/Carbon benchmark OOS windows (2025-12-02 onwards), never used for design.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent / "state" / "research_cache"
DEV_CUTOFF = pd.Timestamp("2025-12-01", tz="UTC")

RESEARCH_SYMBOLS = [
    ("BTCUSDT", "BTCUSDT"),
    ("ETHUSDT", "ETHUSDT"),
    ("SOLUSDT", "SOLUSDT"),
    ("XRPUSDT", "XRPUSDT"),
    ("DOGEUSDT", "DOGEUSDT"),
    ("SUIUSDT", "SUIUSDT"),
    ("AVAXUSDT", "AVAXUSDT"),
    ("XAUUSDT", "PAXGUSDT"),
]

_INTERVAL_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def _get(sess: requests.Session, url: str, params: Dict) -> Optional[list]:
    for attempt in range(5):
        try:
            r = sess.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429):
                time.sleep(2.0 * (attempt + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(1.0 * (attempt + 1))
    return None


def fetch_full_history(canonical: str, fapi_sym: str, interval: str, refresh: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{canonical.lower()}_{interval}_full.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)

    sess = requests.Session()
    sess.headers.update({"User-Agent": "BTCC-Krypton-Research/1.0"})
    step = _INTERVAL_MS[interval]
    cursor = 1_567_296_000_000  # 2019-09-01, before any USDT-M listing
    rows: List[list] = []
    while True:
        raw = _get(sess, "https://fapi.binance.com/fapi/v1/klines",
                   {"symbol": fapi_sym, "interval": interval, "startTime": cursor, "limit": 1500})
        if not raw:
            break
        rows.extend(raw)
        nxt = int(raw[-1][0]) + step
        if len(raw) < 1500 or nxt <= cursor:
            break
        cursor = nxt

    df = pd.DataFrame(
        {
            "open_ms": [int(r[0]) for r in rows],
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [float(r[5]) for r in rows],
            "quote_vol": [float(r[7]) for r in rows],
            "num_trades": [float(r[8]) for r in rows],
            "taker_buy_vol": [float(r[9]) for r in rows],
            "taker_buy_quote_vol": [float(r[10]) for r in rows],
        }
    )
    df = df.drop_duplicates("open_ms").sort_values("open_ms").reset_index(drop=True)
    df.insert(1, "timestamp", pd.to_datetime(df["open_ms"], unit="ms", utc=True))
    if len(df) > 2:
        df = df.iloc[:-1].reset_index(drop=True)  # drop the still-open candle

    # Funding history
    f_rows = []
    fcur = int(df["open_ms"].iloc[0]) - 8 * 3_600_000
    last_ms = int(df["open_ms"].iloc[-1])
    while fcur <= last_ms:
        fd = _get(sess, "https://fapi.binance.com/fapi/v1/fundingRate",
                  {"symbol": fapi_sym, "startTime": fcur, "limit": 1000})
        if not fd:
            break
        f_rows.extend(fd)
        fcur = int(fd[-1]["fundingTime"]) + 1
        if len(fd) < 1000:
            break
    if f_rows:
        fdf = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([int(x["fundingTime"]) for x in f_rows], unit="ms", utc=True),
                "funding_rate": [float(x["fundingRate"]) for x in f_rows],
            }
        ).sort_values("timestamp").drop_duplicates("timestamp")
        # Funding timestamps carry a few ms of jitter; floor to the second so 00:00 prints align.
        fdf["timestamp"] = fdf["timestamp"].dt.floor("s")
        df = pd.merge_asof(df.sort_values("timestamp"), fdf, on="timestamp", direction="backward")
        df["funding_rate"] = df["funding_rate"].fillna(0.0001)
    else:
        df["funding_rate"] = 0.0001

    df.to_pickle(path)
    return df


def load_panel(interval: str, refresh: bool = False) -> Dict[str, pd.DataFrame]:
    return {c: fetch_full_history(c, f, interval, refresh=refresh) for c, f in RESEARCH_SYMBOLS}


if __name__ == "__main__":
    import sys

    for tf in (sys.argv[1:] or ["4h", "1h"]):
        for c, f in RESEARCH_SYMBOLS:
            d = fetch_full_history(c, f, tf)
            print(f"{c:9s} {tf}: {len(d):6d} bars  {d['timestamp'].iloc[0]} -> {d['timestamp'].iloc[-1]}")
