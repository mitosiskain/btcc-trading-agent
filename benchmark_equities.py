"""
Equities Trend V1 benchmark & out-of-sample validation (Google / GOOGLUSDT)
===========================================================================
1. Pooled US Mega-Cap Tech Equities development grid on 2000–2015 ONLY (GOOGL, AAPL, MSFT, AMZN, NVDA, META),
   verified out-of-sample on 2016–2025/11 (gap-honest stops + isolated liquidation + funding accrual).
2. Chosen preset (`equities_engine.EquitiesParams`) evaluated on:
     a) GOOGL Nasdaq daily 2004 -> today, plus DEV (2004–2015), VAL (2016–2025/11), and HOLD-OUT (2025/12 -> today)
     b) 24/7 perpetual execution: Binance GOOGLUSDT 4h & 1h (2026-03-26 -> today) with real 8h funding rates,
        1x and 2x friction stress tests, warmed up with Nasdaq GOOGL history
     c) Baselines on the exact same 24/7 GOOGLUSDT data: Krypton V3 (4h & 1h) and Carbon V2 (4h & 1h)
3. Causality check: truncated daily indicator & signal series match full-history series to < 1e-9.
Writes `state/benchmark_equities_results.json` (the live bridge reads `live_summary` from it).
Run: `.venv/bin/python benchmark_equities.py`
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_argon_vs_carbon import compute_portfolio_metrics, simulate_trade_honest  # noqa: E402
from carbon_v2_engine import CarbonCandidateGenerator, CarbonFeatureEngine, CarbonLiveBridge  # noqa: E402
from config import LeverageRiskConfig  # noqa: E402
from equities_engine import EquitiesParams, add_indicators, backtest, entry_signals  # noqa: E402
from krypton_engine import KryptonParams, compute_features, run_symbol  # noqa: E402
from metals_engine import weekday_daily  # noqa: E402
from research_data import CACHE_DIR, fetch_full_history  # noqa: E402

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
DEV_END, HOLD_START = T("2016-01-01"), T("2025-12-01")
STATE = Path(__file__).resolve().parent / "state"
POOLED_EQUITIES = ("GOOGL", "AAPL", "MSFT", "AMZN", "NVDA", "META")


def yahoo_daily(ticker: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"yahoo_raw_{ticker}_1d.pkl"
    if path.exists() and time.time() - path.stat().st_mtime < 6 * 3600:
        return pd.read_pickle(path)
    r = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        params={"interval": "1d", "period1": 0, "period2": int(time.time()), "includeAdjustedClose": "true"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    ).json()["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(r["timestamp"], unit="s", utc=True),
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "volume": q["volume"],
        }
    ).dropna(subset=["open", "high", "low", "close"])
    df = df.reset_index(drop=True)
    df["volume"] = df["volume"].fillna(0.0)
    df.to_pickle(path)
    return df


def stats(tr: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not tr:
        return {"total_trades": 0}
    df = pd.DataFrame(tr)
    r = df["net_r"].to_numpy()
    days = max(30.0, (pd.to_datetime(df.exit_time.max()) - pd.to_datetime(df.entry_time.min())).days)
    m = compute_portfolio_metrics(tr, days)
    m.pop("equity_curve", None)
    m["t_stat"] = round(float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))), 2) if len(r) > 2 and r.std(ddof=1) > 0 else None
    m["liquidations"] = int(sum(1 for t in tr if t.get("exit_reason") == "LIQUIDATED"))
    m["weekend_exits"] = int(sum(1 for t in tr if pd.Timestamp(t["exit_time"]).day_of_week >= 5))
    m["yearly_pnl_usd"] = {
        int(k): round(float(v), 2)
        for k, v in df.groupby(pd.to_datetime(df.exit_time, utc=True).dt.year).net_pnl_usd.sum().items()
    }
    return m


def check_causality(googl_df: pd.DataFrame, p: EquitiesParams) -> Dict[str, Any]:
    wd = weekday_daily(googl_df)
    full = add_indicators(wd, p)
    sig_full = dict(entry_signals(full, p))
    max_diff = 0.0
    pts = 0
    for cut in np.linspace(int(len(wd) * 0.25), len(wd) - 5, 8).astype(int):
        sub = add_indicators(wd.iloc[: cut + 1], p)
        for col in ("atr", "sma", "hi_entry", "lo_entry", "lo_exit", "hi_exit"):
            a, b = full[col].iloc[cut], sub[col].iloc[cut]
            if np.isfinite(a) or np.isfinite(b):
                max_diff = max(max_diff, abs(float(a) - float(b)))
        sig_sub = dict(entry_signals(sub, p, start=max(0, cut - 50)))
        for idx in range(max(p.sma_days + 2, cut - 50), cut + 1):
            assert sig_full.get(idx) == sig_sub.get(idx), f"causality mismatch at {idx}"
        pts += 1
    assert max_diff < 1e-9, f"indicator leak {max_diff}"
    return {"passed": True, "points_verified": pts, "max_abs_diff": max_diff}


def krypton_baseline(ex: pd.DataFrame, sym: str, tf: str, lev: int, elapsed_funding: bool) -> List[Dict[str, Any]]:
    kp = KryptonParams.for_timeframe(tf)
    d = ex.copy()
    if "funding_rate" not in d.columns:
        d["funding_rate"] = 0.0001
    f = compute_features(d, kp)
    if elapsed_funding:
        ns = pd.to_datetime(d["timestamp"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
        f["funding_bar"] = np.ones(len(d), bool)
        f["funding_rate"] = 0.0001 * (np.diff(ns, prepend=ns[0]) / 3.6e12) / 8.0
    return run_symbol(f, sym, kp, 0, len(d) - 1, max_btcc_leverage=lev)


def carbon_baseline(ex: pd.DataFrame, btc_df: pd.DataFrame, sym: str, tf: str) -> List[Dict[str, Any]]:
    c_bridge = CarbonLiveBridge(LeverageRiskConfig(), state_dir=STATE)
    cdf = CarbonFeatureEngine.compute_features(ex, btc_df=btc_df)
    gen = CarbonCandidateGenerator(timeframe=tf)
    mod = c_bridge.models.get(tf)
    cands = gen.extract_candidates(cdf, symbol=sym, start_idx=80, end_idx=len(cdf) - 2)
    trades: List[Dict[str, Any]] = []
    busy = -1
    for idx in sorted(cands):
        if idx <= busy:
            continue
        sig = cands[idx]
        if mod is not None and mod.is_fitted_ and not mod.score_and_filter(sig):
            continue
        sim = simulate_trade_honest(
            df=cdf,
            symbol=sym,
            signal_idx=idx,
            side=sig.side,
            stop_dist=sig.stop_dist,
            tp1_dist=sig.tp1_r * sig.stop_dist,
            tp2_dist=sig.tp2_r * sig.stop_dist,
            partial_frac=sig.partial_frac,
            max_hold_bars=sig.max_hold_bars,
            model_mode="CARBON_V2",
            trail_atr_mult=sig.trail_atr_mult,
        )
        if sim is not None:
            sim["net_r"] = sim.get("realized_r", sim["net_pnl_usd"] / 25.0)
            trades.append(sim)
            busy = sim["exit_idx"]
    return trades


def main() -> None:
    panel = {sym: yahoo_daily(sym) for sym in POOLED_EQUITIES}
    googl = panel["GOOGL"]
    P = EquitiesParams()

    res: Dict[str, Any] = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "params": asdict(P),
        "causality_check": check_causality(googl, P),
        "grid": [],
        "tests": {},
    }

    # 1) Pooled 6-stock US Mega-Cap Tech development grid (2000-2015 drives choice; 2016-2025/11 validates)
    for N, sma, k_sl, trail, shorts in itertools.product([20, 40, 55], [150, 200], [2.0, 2.5], [3.0, 4.0, 5.0], [False, True]):
        p = EquitiesParams(entry_days=N, exit_days=max(8, N // 2), sma_days=sma, init_stop_atr=k_sl, trail_atr=trail, allow_shorts=shorts)
        row: Dict[str, Any] = {"entry_days": N, "sma_days": sma, "init_stop_atr": k_sl, "trail_atr": trail, "shorts": shorts}
        for per, (a, b) in (("dev_2000_2015", (None, DEV_END)), ("valid_2016_2025", (DEV_END, HOLD_START))):
            tr_all: List[Dict[str, Any]] = []
            for sym, d in panel.items():
                tr_all += backtest(d, f"{sym}USDT", p, start=a, end=b, max_btcc_leverage=50)
            s = stats(tr_all)
            row[per] = {
                k2: s.get(k2)
                for k2 in ("total_trades", "profit_factor", "expectancy_r", "total_net_pnl_usd", "max_drawdown_usd", "t_stat", "liquidations")
            }
        res["grid"].append(row)

    # 2) GOOGL Nasdaq 2004-now + splits + 24/7 GOOGLUSDT perp (4h & 1h) vs Krypton V3 & Carbon V2 baselines
    tests = res["tests"]
    tests["GOOGLUSDT_nasdaq_2004_now"] = stats(backtest(googl, "GOOGLUSDT", P, max_btcc_leverage=50))
    tests["GOOGLUSDT_nasdaq_dev_2004_2015"] = stats(backtest(googl, "GOOGLUSDT", P, end=DEV_END, max_btcc_leverage=50))
    tests["GOOGLUSDT_nasdaq_valid_2016_2025"] = stats(backtest(googl, "GOOGLUSDT", P, start=DEV_END, end=HOLD_START, max_btcc_leverage=50))
    tests["GOOGLUSDT_nasdaq_holdout_2025_now"] = stats(backtest(googl, "GOOGLUSDT", P, start=HOLD_START, max_btcc_leverage=50))
    tests["GOOGLUSDT_nasdaq_krypton_daily"] = stats(krypton_baseline(googl, "GOOGLUSDT", "4h", 50, True))

    perp_4h = fetch_full_history("GOOGLUSDT", "GOOGLUSDT", "4h")
    perp_1h = fetch_full_history("GOOGLUSDT", "GOOGLUSDT", "1h")
    btc_4h = fetch_full_history("BTCUSDT", "BTCUSDT", "4h")
    btc_1h = fetch_full_history("BTCUSDT", "BTCUSDT", "1h")
    wd_googl = weekday_daily(googl)
    st_perp = perp_4h["timestamp"].iloc[0] + pd.Timedelta(days=1)

    tests["GOOGLUSDT_247_perp_4h"] = stats(backtest(perp_4h, "GOOGLUSDT", P, daily_hist=wd_googl, funding_mode="real", max_btcc_leverage=50))
    tests["GOOGLUSDT_247_perp_4h_2x_cost"] = stats(
        backtest(perp_4h, "GOOGLUSDT", P, daily_hist=wd_googl, funding_mode="real", max_btcc_leverage=50, cost_mult=2.0)
    )
    tests["GOOGLUSDT_247_perp_1h"] = stats(backtest(perp_1h, "GOOGLUSDT", P, daily_hist=wd_googl, funding_mode="real", max_btcc_leverage=50))
    tests["GOOGLUSDT_perp_session_same_span"] = stats(backtest(googl, "GOOGLUSDT", P, start=st_perp, max_btcc_leverage=50))
    tests["GOOGLUSDT_247_perp_4h_krypton"] = stats(krypton_baseline(perp_4h, "GOOGLUSDT", "4h", 50, False))
    tests["GOOGLUSDT_247_perp_1h_krypton"] = stats(krypton_baseline(perp_1h, "GOOGLUSDT", "1h", 50, False))
    tests["GOOGLUSDT_247_perp_4h_carbon_v2"] = stats(carbon_baseline(perp_4h, btc_4h, "GOOGLUSDT", "4h"))
    tests["GOOGLUSDT_247_perp_1h_carbon_v2"] = stats(carbon_baseline(perp_1h, btc_1h, "GOOGLUSDT", "1h"))

    s_full = tests["GOOGLUSDT_nasdaq_2004_now"]
    res["live_summary"] = {
        "GOOGLUSDT": {
            k: s_full.get(k)
            for k in ("total_trades", "win_rate_pct", "profit_factor", "expectancy_r", "total_net_pnl_usd", "max_drawdown_usd")
        }
    }

    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "benchmark_equities_results.json").write_text(json.dumps(res, indent=1, default=str))

    g = pd.DataFrame(
        [
            {
                **{k: r[k] for k in ("entry_days", "sma_days", "init_stop_atr", "trail_atr", "shorts")},
                **{f"dev_{k}": v for k, v in r["dev_2000_2015"].items()},
                **{f"val_{k}": v for k, v in r["valid_2016_2025"].items()},
            }
            for r in res["grid"]
        ]
    )
    ns = g[~g["shorts"]]
    pd.set_option("display.width", 260)
    print(
        f"Pooled 6-Stock Long-Only Grid: {100 * (ns.dev_total_net_pnl_usd > 0).mean():.0f}% profitable in 2000-2015 DEV, "
        f"{100 * (ns.val_total_net_pnl_usd > 0).mean():.0f}% profitable in 2016-2025/11 VAL"
    )
    print(g.sort_values("dev_t_stat", ascending=False).head(8).to_string(index=False))
    print()
    for k, s in tests.items():
        if s.get("total_trades"):
            print(
                f"{k:36s} n={s['total_trades']:3d} WR={s['win_rate_pct']:5.1f}% PF={s['profit_factor']:6.2f} "
                f"expR={s['expectancy_r']:+.3f} PnL=${s['total_net_pnl_usd']:+8.1f} DD=${s['max_drawdown_usd']:6.1f} "
                f"t={s['t_stat']} liq={s['liquidations']}"
            )
        else:
            print(f"{k:36s} no trades")


if __name__ == "__main__":
    main()
