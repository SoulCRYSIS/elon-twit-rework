#!/usr/bin/env python3
"""
Train a grid of ML variants (artifact-tag) and backtest each; write summary CSV/JSON.

Usage:
  python scripts/run_ml_experiments.py
  python scripts/run_ml_experiments.py --skip-train   # only backtest existing artifacts
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_run_backtest():
    path = ROOT / "scripts" / "run_backtest.py"
    spec = importlib.util.spec_from_file_location("run_backtest_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

ARTIFACT_DIR = ROOT / "data" / "ml_artifacts"
RESULTS_DIR = ROOT / "results"

# (tag, classifier, extra train_cli args). Tags become xgb_model_<tag> / meta_<tag> or hgb_<tag> / meta_hgb_<tag>.
EXPERIMENTS: list[tuple[str, str, list[str]]] = [
    ("xgb_baseline", "xgboost", []),
    ("xgb_ev_prec", "xgboost", ["--ev-weighted", "--tune-on", "precision"]),
    ("xgb_ev_pnl", "xgboost", ["--ev-weighted", "--tune-on", "pnl"]),
    ("xgb_pnl_only", "xgboost", ["--tune-on", "pnl"]),
    ("xgb_ev_pnl_m08", "xgboost", ["--ev-weighted", "--tune-on", "pnl", "--max-buy-rate", "0.08"]),
    ("xgb_ev_pnl_m12", "xgboost", ["--ev-weighted", "--tune-on", "pnl", "--max-buy-rate", "0.12"]),
    ("hist_baseline", "hgb", ["--tune-on", "precision"]),
    ("hist_ev_prec", "hgb", ["--ev-weighted", "--tune-on", "precision"]),
    ("hist_ev_pnl", "hgb", ["--ev-weighted", "--tune-on", "pnl"]),
]


def _closed_events_month_span(events: pd.DataFrame) -> float:
    """Fractional months min(start)→max(end) over closed events (~30.437 d/mo)."""
    ev = events[events["closed"] == True]
    if ev.empty:
        return 1.0
    start = pd.to_datetime(ev["start_date"], utc=True).min()
    end = pd.to_datetime(ev["end_date"], utc=True).max()
    days = max((end - start).total_seconds() / 86400.0, 1.0)
    return max(days / 30.437, 1e-6)


def artifact_paths(tag: str, clf: str) -> tuple[Path, Path]:
    if clf == "hgb":
        return ARTIFACT_DIR / f"hgb_{tag}.joblib", ARTIFACT_DIR / f"meta_hgb_{tag}.json"
    return ARTIFACT_DIR / f"xgb_model_{tag}.joblib", ARTIFACT_DIR / f"meta_{tag}.json"


def train_one(tag: str, clf: str, extra: list[str]) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_ml_model.py"),
        "--classifier",
        clf,
        "--artifact-tag",
        tag,
        *extra,
    ]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="Only run backtests (artifacts must exist).")
    parser.add_argument("--max-events", type=int, default=0, help="0=all closed events in backtest.")
    args = parser.parse_args()

    from approaches.ml_artifact_signal import make_get_signal

    rb = _load_run_backtest()
    load_data = rb.load_data
    run_backtest_for_approach = rb.run_backtest_for_approach

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        for tag, clf, extra in EXPERIMENTS:
            train_one(tag, clf, extra)

    events, markets, prices = load_data()
    if args.max_events > 0:
        closed_ids = events[events["closed"] == True]["event_id"].head(args.max_events).tolist()
        events = events[events["event_id"].isin(closed_ids)]
        markets = markets[markets["event_id"].isin(closed_ids)]

    month_span = _closed_events_month_span(events)

    rows = []
    for tag, clf, extra in EXPERIMENTS:
        mp, jp = artifact_paths(tag, clf)
        if not mp.exists() or not jp.exists():
            rows.append(
                {
                    "experiment": tag,
                    "classifier": clf,
                    "train_extra": " ".join(extra),
                    "error": "missing_artifacts",
                    "total_pnl": None,
                    "avg_profit_per_trade": None,
                    "total_trades": 0,
                    "win_rate": None,
                    "accuracy_pct": None,
                    "trades_per_month": None,
                    "sharpe": None,
                    "max_drawdown": None,
                }
            )
            continue
        with open(jp) as f:
            meta = json.load(f)
        get_signal = make_get_signal(mp, jp)
        r = run_backtest_for_approach(
            tag,
            events,
            markets,
            prices,
            get_signal=get_signal,
        )
        rows.append(
            {
                "experiment": tag,
                "classifier": clf,
                "train_extra": " ".join(extra),
                "ev_weighted": meta.get("ev_weighted"),
                "tune_on": meta.get("tune_on"),
                "edge_threshold": meta.get("edge_threshold"),
                "error": None,
                "total_pnl": round(r["total_pnl"], 4),
                "avg_profit_per_trade": round(r["avg_profit_per_trade"], 6),
                "total_trades": r["total_trades"],
                "win_rate": round(r["win_rate"], 4),
                "accuracy_pct": round(r["win_rate"] * 100, 2),
                "trades_per_month": round(r["total_trades"] / month_span, 4),
                "sharpe": round(r["sharpe"], 4),
                "max_drawdown": round(r["max_drawdown"], 4),
            }
        )

    df = pd.DataFrame(rows)
    df_ok = df[df["error"].isna() & (df["total_trades"] > 0)].copy()
    df_ok = df_ok.sort_values("avg_profit_per_trade", ascending=False)

    out_csv = RESULTS_DIR / "ml_experiment_results.csv"
    out_json = RESULTS_DIR / "ml_experiment_summary.json"
    df.to_csv(out_csv, index=False)
    best_avg = df_ok.iloc[0].to_dict() if len(df_ok) else None
    best_total = df_ok.sort_values("total_pnl", ascending=False).iloc[0].to_dict() if len(df_ok) else None
    summary = {
        "n_experiments": len(EXPERIMENTS),
        "backtest_month_span": round(month_span, 4),
        "best_by_avg_profit_per_trade": best_avg,
        "best_by_total_pnl": best_total,
        "ranking_avg_profit": df_ok["experiment"].tolist() if len(df_ok) else [],
        "ranking_total_pnl": df_ok.sort_values("total_pnl", ascending=False)["experiment"].tolist()
        if len(df_ok)
        else [],
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    table_path = RESULTS_DIR / "ml_summary_table.csv"
    if len(df_ok) and "accuracy_pct" in df_ok.columns:
        tbl = df_ok[
            [
                "experiment",
                "classifier",
                "accuracy_pct",
                "avg_profit_per_trade",
                "trades_per_month",
                "total_trades",
            ]
        ].sort_values("avg_profit_per_trade", ascending=False)
        tbl = tbl.rename(columns={"avg_profit_per_trade": "average_profit_usd"})
        tbl.to_csv(table_path, index=False)

    print(f"\nBacktest month span (closed events): {month_span:.3f} fractional months\n")
    print("=== Summary table: accuracy | avg profit | trades/mo ===\n")
    if len(df_ok) and table_path.exists():
        print(tbl.to_string(index=False))
        print(f"\nWrote {table_path}")
    print("\n=== Full metrics (sorted by avg_profit_per_trade) ===\n")
    if len(df_ok):
        cols = [
            "experiment",
            "avg_profit_per_trade",
            "total_pnl",
            "total_trades",
            "accuracy_pct",
            "trades_per_month",
            "sharpe",
            "edge_threshold",
            "ev_weighted",
            "tune_on",
        ]
        print(df_ok[[c for c in cols if c in df_ok.columns]].to_string(index=False))
    else:
        print(df.to_string(index=False))
    print(f"\nWrote {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
