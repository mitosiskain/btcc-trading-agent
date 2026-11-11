"""
Metals Trend V1 benchmark (gold & silver)
=========================================
1. Development grid on COMEX futures 2000-2015 (GC=F / SI=F daily), shown with 2016-2025/11 validation.
2. Chosen preset (metals_engine.MetalsParams) on:
     a) futures daily 2000 -> today and the 2025-12 -> today hold-out
     b) 24/7 execution: PAXG spot 4h (2020-09 ->), Binance XAU/XAG perps 4h (2025-12 / 2026-01 ->), with the futures
        history used only to warm up the 200-day average, vs. the same span traded on session-only futures data
     c) Krypton V3 on the same 24/7 data (current crypto engine) as the baseline
Writes state/benchmark_metals_results.json (the live bridge reads `live_summary` from it).
Run: .venv/bin/python benchmark_metals.py
"""
from __future__ import annotations

import itertools, json, math, sys, time
from dataclasses import replace, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_argon_vs_carbon import compute_portfolio_metrics  # noqa: E402
from krypton_engine import KryptonParams, compute_features, run_symbol  # noqa: E402
from metals_engine import MetalsParams, backtest, weekday_daily  # noqa: E402
from research_data import CACHE_DIR, _get, fetch_full_history  # noqa: E402

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
DEV_END, HOLD_START = T("2016-01-01"), T("2025-12-01")
STATE = Path(__file__).resolve().parent / "state"


def yahoo_daily(ticker: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"yahoo_{ticker.replace('=', '_')}_1d.pkl"
    if path.exists() and time.time() - path.stat().st_mtime < 6 * 3600:
        return pd.read_pickle(path)
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                     params={"interval": "1d", "period1": 0, "period2": int(time.time()), "includeAdjustedClose": "true"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=20).json()["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({"timestamp": pd.to_datetime(r["timestamp"], unit="s", utc=True), "open": q["open"], "high": q["high"],
                       "low": q["low"], "close": q["close"], "volume": q["volume"]}).dropna(subset=["open", "high", "low", "close"])
    df = df.reset_index(drop=True); df["volume"] = df["volume"].fillna(0.0)
    df.to_pickle(path)
    return df


def binance_spot_4h(sym: str) -> pd.DataFrame:
    path = CACHE_DIR / f"spot_{sym.lower()}_4h_full.pkl"
    if path.exists() and time.time() - path.stat().st_mtime < 6 * 3600:
        return pd.read_pickle(path)
    sess, cur, rows = requests.Session(), 1_598_000_000_000, []
    while True:
        raw = _get(sess, "https://api.binance.com/api/v3/klines", {"symbol": sym, "interval": "4h", "startTime": cur, "limit": 1000})
        if not raw:
            break
        rows += raw; cur = int(raw[-1][0]) + 14_400_000
        if len(raw) < 1000:
            break
    df = pd.DataFrame({"timestamp": pd.to_datetime([int(x[0]) for x in rows], unit="ms", utc=True), "open": [float(x[1]) for x in rows],
                       "high": [float(x[2]) for x in rows], "low": [float(x[3]) for x in rows], "close": [float(x[4]) for x in rows],
                       "volume": [float(x[5]) for x in rows]}).drop_duplicates("timestamp").iloc[:-1].reset_index(drop=True)
    df.to_pickle(path)
    return df


def stats(tr):
    if not tr:
        return {"total_trades": 0}
    df = pd.DataFrame(tr); r = df["net_r"].to_numpy()
    days = max(30.0, (pd.to_datetime(df.exit_time.max()) - pd.to_datetime(df.entry_time.min())).days)
    m = compute_portfolio_metrics(tr, days); m.pop("equity_curve", None)
    m["t_stat"] = round(float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))), 2) if len(r) > 2 else None
    m["weekend_exits"] = int(sum(1 for t in tr if pd.Timestamp(t["exit_time"]).dayofweek >= 5))
    m["yearly_pnl_usd"] = {int(k): round(float(v), 2) for k, v in df.groupby(pd.to_datetime(df.exit_time, utc=True).dt.year).net_pnl_usd.sum().items()}
    return m


def krypton_on(ex, sym, lev, elapsed_funding):
    p = KryptonParams.for_timeframe("4h")
    d = ex.copy()
    if "funding_rate" not in d.columns:
        d["funding_rate"] = 0.0001
    f = compute_features(d, p)
    if elapsed_funding:
        ns = pd.to_datetime(d["timestamp"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
        f["funding_bar"] = np.ones(len(d), bool); f["funding_rate"] = 0.0001 * (np.diff(ns, prepend=ns[0]) / 3.6e12) / 8.0
    return run_symbol(f, sym, p, 0, len(d) - 1, max_btcc_leverage=lev)


def main():
    gc, si = yahoo_daily("GC=F"), yahoo_daily("SI=F")
    fut = {"XAUUSDT": (gc, 200), "XAGUSDT": (si, 150)}
    res = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "params": asdict(MetalsParams()), "grid": [], "tests": {}}

    # 1) Development grid (trend family) — 2000-2015 only drives the choice
    for N, k, trail, shorts in itertools.product([20, 40, 60], [2.0, 3.0], [3.0, 5.0, 8.0], [False, True]):
        p = MetalsParams(entry_days=N, exit_days=max(5, N // 2), init_stop_atr=k, trail_atr=trail, allow_shorts=shorts)
        row = {"entry_days": N, "init_stop_atr": k, "trail_atr": trail, "shorts": shorts}
        for per, (a, b) in (("dev_2000_2015", (None, DEV_END)), ("valid_2016_2025", (DEV_END, HOLD_START))):
            tr = []
            for sym, (d, lev) in fut.items():
                tr += backtest(d, sym, p, start=a, end=b, max_btcc_leverage=lev)
            s = stats(tr)
            row[per] = {k2: s.get(k2) for k2 in ("total_trades", "profit_factor", "expectancy_r", "total_net_pnl_usd", "max_drawdown_usd", "t_stat")}
        res["grid"].append(row)

    P = MetalsParams()
    tests = res["tests"]
    live_summary = {}
    for sym, (d, lev) in fut.items():
        tests[f"{sym}_futures_2000_now"] = stats(backtest(d, sym, P, max_btcc_leverage=lev))
        tests[f"{sym}_futures_holdout"] = stats(backtest(d, sym, P, start=HOLD_START, max_btcc_leverage=lev))
        tests[f"{sym}_futures_krypton_daily"] = stats(krypton_on(d, sym, lev, True))
        live_summary[sym] = {k: tests[f"{sym}_futures_2000_now"].get(k) for k in
                             ("total_trades", "win_rate_pct", "profit_factor", "expectancy_r", "total_net_pnl_usd", "max_drawdown_usd")}
    res["live_summary"] = live_summary

    paxg = binance_spot_4h("PAXGUSDT")
    s0 = paxg["timestamp"].iloc[0] + pd.Timedelta(days=1)
    tests["XAUUSDT_247_paxg_spot_4h"] = stats(backtest(paxg, "XAUUSDT", P, daily_hist=weekday_daily(gc), max_btcc_leverage=200))
    tests["XAUUSDT_session_same_span"] = stats(backtest(gc, "XAUUSDT", P, start=s0, max_btcc_leverage=200))
    tests["XAUUSDT_247_paxg_krypton"] = stats(krypton_on(paxg, "XAUUSDT", 200, True))
    for sym, perp, d, lev in (("XAUUSDT", "XAUUSDT", gc, 200), ("XAGUSDT", "XAGUSDT", si, 150)):
        ex = fetch_full_history(f"{perp}_COMMODITY", perp, "4h")
        st = ex["timestamp"].iloc[0] + pd.Timedelta(days=1)
        tests[f"{sym}_247_perp_4h"] = stats(backtest(ex, sym, P, daily_hist=weekday_daily(d), funding_mode="real", max_btcc_leverage=lev))
        tests[f"{sym}_perp_session_same_span"] = stats(backtest(d, sym, P, start=st, max_btcc_leverage=lev))
        tests[f"{sym}_247_perp_krypton"] = stats(krypton_on(ex, sym, lev, False))

    (STATE / "benchmark_metals_results.json").write_text(json.dumps(res, indent=1, default=str))
    g = pd.DataFrame([{**{k: r[k] for k in ("entry_days", "init_stop_atr", "trail_atr", "shorts")},
                       **{f"dev_{k}": v for k, v in r["dev_2000_2015"].items()}, **{f"val_{k}": v for k, v in r["valid_2016_2025"].items()}}
                      for r in res["grid"]])
    pd.set_option("display.width", 250)
    print(f"Grid: {100 * (g.dev_total_net_pnl_usd > 0).mean():.0f}% of variants profitable 2000-2015, "
          f"{100 * (g.val_total_net_pnl_usd > 0).mean():.0f}% in 2016-2025/11")
    print(g.sort_values("dev_profit_factor", ascending=False).head(8).to_string(index=False))
    print()
    for k, s in tests.items():
        if s.get("total_trades"):
            print(f"{k:34s} n={s['total_trades']:3d} WR={s['win_rate_pct']:5.1f}% PF={s['profit_factor']:6.2f} expR={s['expectancy_r']:+.3f} "
                  f"PnL=${s['total_net_pnl_usd']:+7.0f} DD=${s['max_drawdown_usd']:5.0f} t={s['t_stat']} wkndExits={s['weekend_exits']}")
        else:
            print(f"{k:34s} no trades")


if __name__ == "__main__":
    main()
