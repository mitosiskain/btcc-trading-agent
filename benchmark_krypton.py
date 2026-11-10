"""
Krypton V3 vs. Argon V1 vs. Carbon V2 — Hold-Out Benchmark
==========================================================

Protocol (pre-registered before touching any hold-out bar):
  * Krypton's rules & parameters were chosen ONLY on development data that ends 2025-12-01 UTC
    (Binance USDT-M history from listing, see `research_data.py`). The parameter surface was flat
    (all 288 4h variants profitable in development) and the chosen preset is the classic
    Turtle "System 1" 20/10 Donchian + 2-ATR stop with a 50/200 EMA trend filter.
  * The hold-out test re-uses the *exact* panels, OOS windows, execution simulator friction model,
    $25-per-1R sizing, portfolio concurrency cap (3 positions, <= 2 same side) and metric code of
    `benchmark_argon_vs_carbon.py`, so the numbers are directly comparable with Argon / Carbon.
  * Krypton has nothing to (re)train, so it simply runs causally over each OOS window.

Extra checks:
  1. Simulator equivalence: Krypton's fast simulator re-prices every Carbon candidate trade and must
     match `simulate_trade_honest` to the cent.
  2. Causality: features/signals recomputed on truncated history must equal the full-history values.
  3. Cost stress: 2x fees+spread+slippage.
  4. Bootstrap confidence interval on per-trade expectancy.
  5. Extended hold-out on the full-history panel (2025-12-01 -> today) and the development record.

Run:  .venv/bin/python benchmark_krypton.py [--rerun-reference]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_argon_vs_carbon import (  # noqa: E402
    BENCHMARK_SYMBOLS,
    apply_portfolio_concurrency_cap,
    compute_portfolio_metrics,
    fetch_paginated_perp_data,
    simulate_trade_honest,
)
from carbon_v2_engine import CarbonCandidateGenerator, CarbonFeatureEngine  # noqa: E402
from config import DEFAULT_ASSETS  # noqa: E402
from krypton_engine import (  # noqa: E402
    KryptonParams,
    compute_features,
    generate_signals,
    portfolio_cap_by_time,
    run_symbol,
    simulate_trade_fast,
)

STATE = Path(__file__).resolve().parent / "state"
REF_PATH = STATE / "benchmark_argon_vs_carbon_results.json"
OUT_PATH = STATE / "benchmark_krypton_results.json"

# Identical to benchmark_argon_vs_carbon.main()
TF_SPECS = {
    "4h": dict(total_bars=4500, train_bars=1500, fold_bars=500),
    "1h": dict(total_bars=6000, train_bars=2000, fold_bars=800),
}


def _max_lev(sym: str) -> int:
    prof = DEFAULT_ASSETS.get(sym)
    return prof.max_btcc_leverage if prof else 100


def load_benchmark_panel(tf: str) -> Dict[str, pd.DataFrame]:
    spec = TF_SPECS[tf]
    raw = {s: fetch_paginated_perp_data(s, f, tf, spec["total_bars"]) for s, f in BENCHMARK_SYMBOLS}
    m = min(len(d) for d in raw.values())
    return {s: d.iloc[-m:].reset_index(drop=True) for s, d in raw.items()}


def build_folds(min_len: int, train_bars: int, fold_bars: int) -> List[Tuple[int, int]]:
    folds, cur = [], train_bars
    while cur + 100 < min_len:
        folds.append((cur, min(min_len - 1, cur + fold_bars)))
        cur += fold_bars
    return folds


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------
def check_simulator_equivalence(raw: Dict[str, pd.DataFrame], tf: str) -> Dict[str, Any]:
    gen = CarbonCandidateGenerator(timeframe=tf)
    p = KryptonParams.for_timeframe(tf)
    n_tr, max_diff = 0, 0.0
    for s, _ in BENCHMARK_SYMBOLS:
        dfc = CarbonFeatureEngine.compute_features(raw[s], btc_df=raw["BTCUSDT"])
        f = compute_features(raw[s], p)
        for i, sig in gen.extract_candidates(dfc, symbol=s, start_idx=80).items():
            kw = dict(stop_dist=sig.stop_dist, tp1_dist=sig.tp1_r * sig.stop_dist, tp2_dist=sig.tp2_r * sig.stop_dist,
                      partial_frac=sig.partial_frac, max_hold_bars=sig.max_hold_bars, trail_atr_mult=sig.trail_atr_mult)
            a = simulate_trade_honest(df=dfc, symbol=s, signal_idx=i, side=sig.side, model_mode="CARBON_V2", **kw)
            b = simulate_trade_fast(f, s, i, sig.side, mode="CARBON_V2", cvd6=dfc["cvd_6"].to_numpy(),
                                    atr_override=dfc["atr"].to_numpy(), max_btcc_leverage=_max_lev(s), **kw)
            if a is None or b is None:
                assert a is None and b is None, f"existence mismatch {s}@{i}"
                continue
            assert a["exit_idx"] == b["exit_idx"] and a["exit_reason"] == b["exit_reason"], f"path mismatch {s}@{i}"
            max_diff = max(max_diff, abs(a["net_pnl_usd"] - b["net_pnl_usd"]))
            n_tr += 1
    assert max_diff < 1e-6, f"simulator drift {max_diff}"
    return {"passed": True, "carbon_trades_repriced": n_tr, "max_abs_pnl_diff_usd": max_diff}


def check_causality(raw: Dict[str, pd.DataFrame], tf: str) -> Dict[str, Any]:
    p = KryptonParams.for_timeframe(tf)
    pts, max_diff = 0, 0.0
    cols = ("atr", "ema_f", "ema_s", "don_hi", "don_lo", "ex_hi", "ex_lo", "btc_trend")
    for s in ("ETHUSDT", "SOLUSDT", "XAUUSDT"):
        full = compute_features(raw[s], p, btc_df=raw["BTCUSDT"])
        sig_full = generate_signals(full, s, p)
        n = len(raw[s])
        for t in np.linspace(n * 0.35, n - 3, 6).astype(int):
            sub = raw[s].iloc[: t + 1]
            bsub = raw["BTCUSDT"][raw["BTCUSDT"]["timestamp"] <= sub["timestamp"].iloc[-1]]
            tr = compute_features(sub, p, btc_df=bsub)
            for c in cols:
                a, b = full[c][t], tr[c][t]
                if np.isfinite(a) or np.isfinite(b):
                    max_diff = max(max_diff, abs(float(a) - float(b)))
            sig_tr = generate_signals(tr, s, p, start_idx=t - 40, end_idx=t + 1)  # includes bar t (n-1 of trunc is excluded)
            for i in range(t - 40, t):
                assert (i in sig_full) == (i in sig_tr), f"signal leak {s}@{i}"
            pts += 1
    assert max_diff < 1e-9, f"feature leak {max_diff}"
    return {"passed": True, "points_verified": pts, "max_abs_diff": max_diff}


# --------------------------------------------------------------------------------------
# Krypton OOS run on the benchmark panel
# --------------------------------------------------------------------------------------
def run_krypton_oos(raw: Dict[str, pd.DataFrame], tf: str, folds: List[Tuple[int, int]], cost_mult: float = 1.0,
                    p: KryptonParams | None = None) -> List[Dict[str, Any]]:
    p = p or KryptonParams.for_timeframe(tf)
    oos_start, oos_end = folds[0][0], folds[-1][1]
    trades: List[Dict[str, Any]] = []
    for s, _ in BENCHMARK_SYMBOLS:
        f = compute_features(raw[s], p, btc_df=raw["BTCUSDT"])
        trades += run_symbol(f, s, p, oos_start, oos_end, max_btcc_leverage=_max_lev(s), cost_mult=cost_mult)
    return apply_portfolio_concurrency_cap(trades, max_concurrent=3, max_same_side=2)


def bootstrap_ci(trades: List[Dict[str, Any]], n_boot: int = 5000, seed: int = 7) -> Dict[str, float]:
    if len(trades) < 5:
        return {}
    pnl = np.array([t["net_pnl_usd"] for t in trades])
    r = np.array([t["net_r"] for t in trades])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), size=(n_boot, len(pnl)))
    tot = pnl[idx].sum(axis=1)
    er = r[idx].mean(axis=1)
    return {
        "p_total_pnl_gt_0": round(float(np.mean(tot > 0)), 3),
        "pnl_ci90_low": round(float(np.quantile(tot, 0.05)), 2),
        "pnl_ci90_high": round(float(np.quantile(tot, 0.95)), 2),
        "exp_r_ci90_low": round(float(np.quantile(er, 0.05)), 3),
        "exp_r_ci90_high": round(float(np.quantile(er, 0.95)), 3),
        "t_stat_mean_r": round(float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))), 2),
    }


def strip(m: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in m.items() if k != "equity_curve"}


def breakdown(trades: List[Dict[str, Any]], key: str, oos_days: float) -> Dict[str, Any]:
    out = {}
    for v in sorted(set(t[key] for t in trades)):
        out[v] = strip(compute_portfolio_metrics([t for t in trades if t[key] == v], oos_days))
    return out


# --------------------------------------------------------------------------------------
# Extended (full-history) evaluation
# --------------------------------------------------------------------------------------
def run_full_history(tf: str, start: pd.Timestamp, end: pd.Timestamp | None = None) -> Dict[str, Any]:
    from research_data import load_panel

    p = KryptonParams.for_timeframe(tf)
    panel = load_panel(tf)
    btc = panel["BTCUSDT"]
    trades: List[Dict[str, Any]] = []
    for s, d in panel.items():
        ts = d["timestamp"]
        lo = int(ts.searchsorted(start, side="left"))
        hi = len(d) - 1 if end is None else int(ts.searchsorted(end, side="left"))
        if end is not None:
            d = d.iloc[: hi + 1].reset_index(drop=True)  # never let a dev trade see bars beyond `end`
            hi = len(d) - 1
        f = compute_features(d, p, btc_df=btc)
        trades += run_symbol(f, s, p, lo, hi, max_btcc_leverage=_max_lev(s))
    trades = portfolio_cap_by_time(trades)
    if not trades:
        return {}
    days = (max(t["exit_ns"] for t in trades) - min(t["entry_ns"] for t in trades)) / 86400e9
    m = compute_portfolio_metrics(trades, days)
    df = pd.DataFrame(trades)
    df["yr"] = pd.to_datetime(df["exit_time"]).dt.year
    yearly = {int(y): round(float(v), 2) for y, v in df.groupby("yr")["net_pnl_usd"].sum().items()}
    return {"window": f"{str(start)[:10]} -> {str(end)[:10] if end is not None else 'latest'}", "metrics": m, "yearly_pnl_usd": yearly,
            "bootstrap": bootstrap_ci(trades)}


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-reference", action="store_true", help="re-run Argon/Carbon walk-forward instead of loading stored results")
    args = ap.parse_args()

    ref = json.loads(REF_PATH.read_text()) if REF_PATH.exists() else {}
    results: Dict[str, Any] = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "params": {}}

    for tf, spec in TF_SPECS.items():
        t0 = time.time()
        raw = load_benchmark_panel(tf)
        min_len = len(raw["BTCUSDT"])
        folds = build_folds(min_len, spec["train_bars"], spec["fold_bars"])
        bar_h = 4.0 if tf == "4h" else 1.0
        oos_days = (folds[-1][1] - folds[0][0]) * bar_h / 24.0
        window = f"{str(raw['BTCUSDT']['timestamp'].iloc[folds[0][0]])[:10]} -> {str(raw['BTCUSDT']['timestamp'].iloc[folds[-1][1]])[:10]}"

        sim_chk = check_simulator_equivalence(raw, tf)
        cau_chk = check_causality(raw, tf)

        k_trades = run_krypton_oos(raw, tf, folds)
        k_stress = run_krypton_oos(raw, tf, folds, cost_mult=2.0)
        m_k = compute_portfolio_metrics(k_trades, oos_days)
        m_ks = compute_portfolio_metrics(k_stress, oos_days)

        if args.rerun_reference:
            from benchmark_argon_vs_carbon import run_walk_forward_for_timeframe

            ref_tf = run_walk_forward_for_timeframe(interval=tf, **spec)
        else:
            ref_tf = ref.get(tf, {})

        results["params"][tf] = KryptonParams.for_timeframe(tf).__dict__
        results[tf] = {
            "oos_window": window,
            "oos_days": round(oos_days, 1),
            "simulator_equivalence": sim_chk,
            "causality_check": cau_chk,
            "krypton_v3_oos": m_k,
            "krypton_v3_oos_2x_costs": strip(m_ks),
            "krypton_bootstrap": bootstrap_ci(k_trades),
            "krypton_per_symbol": breakdown(k_trades, "symbol", oos_days),
            "krypton_per_side": breakdown(k_trades, "side", oos_days),
            "krypton_exit_reasons": {r: sum(1 for t in k_trades if t["exit_reason"] == r) for r in sorted(set(t["exit_reason"] for t in k_trades))},
            "krypton_trades": k_trades,
            "reference": {
                "source": "re-run" if args.rerun_reference else str(REF_PATH.name),
                "oos_window": ref_tf.get("oos_window"),
                "argon_static_oos": strip(ref_tf.get("argon_static_oos", {})),
                "argon_walkforward_oos": strip(ref_tf.get("argon_walkforward_oos", {})),
                "carbon_v2_walkforward_oos": strip(ref_tf.get("carbon_v2_walkforward_oos", {})),
                "carbon_v2_equity_curve": ref_tf.get("carbon_v2_walkforward_oos", {}).get("equity_curve", []),
            },
        }
        print(f"[{tf}] done in {time.time() - t0:.1f}s  window {window}")

    from research_data import DEV_CUTOFF

    results["extended"] = {}
    for tf in ("4h", "1h"):
        results["extended"][tf] = {
            "development_2019_to_2025_11": run_full_history(tf, pd.Timestamp("2019-01-01", tz="UTC"), DEV_CUTOFF),
            "holdout_2025_12_to_latest": run_full_history(tf, DEV_CUTOFF),
        }
        for k in results["extended"][tf].values():
            if k and "metrics" in k:
                k["metrics"].pop("equity_curve", None)

    OUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved -> {OUT_PATH}\n")

    hdr = f"{'Model':38s} {'Trades':>6s} {'WR%':>6s} {'PF':>6s} {'Exp R':>7s} {'Net PnL':>9s} {'MaxDD':>8s} {'Sharpe':>7s}"
    for tf in TF_SPECS:
        r = results[tf]
        print(f"=== {tf.upper()} hold-out {r['oos_window']} ({r['oos_days']} days) ===")
        print(hdr)
        rows = [
            ("Argon V1 (static, OOS)", r["reference"]["argon_static_oos"]),
            ("Argon V1 (walk-forward, OOS)", r["reference"]["argon_walkforward_oos"]),
            ("Carbon V2 (purged meta-WF, OOS)", r["reference"]["carbon_v2_walkforward_oos"]),
            ("Krypton V3 (trend, OOS)", r["krypton_v3_oos"]),
            ("Krypton V3 (trend, OOS, 2x costs)", r["krypton_v3_oos_2x_costs"]),
        ]
        for name, m in rows:
            if not m:
                continue
            print(f"{name:38s} {m['total_trades']:6d} {m['win_rate_pct']:6.1f} {m['profit_factor']:6.2f} {m['expectancy_r']:+7.3f} "
                  f"{m['total_net_pnl_usd']:+9.2f} {m['max_drawdown_usd']:8.2f} {m['sharpe_annualized']:+7.2f}")
        print(f"  bootstrap: {r['krypton_bootstrap']}\n")
    for tf, e in results["extended"].items():
        for name, blk in e.items():
            if blk:
                m = blk["metrics"]
                print(f"[extended {tf}] {name:28s} trades={m['total_trades']:4d} PF={m['profit_factor']:.2f} expR={m['expectancy_r']:+.3f} "
                      f"PnL=${m['total_net_pnl_usd']:+.0f} DD=${m['max_drawdown_usd']:.0f} Sharpe={m['sharpe_annualized']:+.2f} yearly={blk['yearly_pnl_usd']}")


if __name__ == "__main__":
    main()
