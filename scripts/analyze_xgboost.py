#!/usr/bin/env python3
"""
Analyze backtest: AVERAGE PROFIT PER TRADE ($1 per trade) + SELL ANALYSIS.
Optimization target: maximize avg PnL per trade, ignore win rate.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared import progress_log  # noqa: E402

RESULTS_DIR = ROOT / "results"
AN_PHASES = 7


def analyze_selling(df: pd.DataFrame, name: str, *, step: int) -> None:
    """Analyze how positions are sold."""
    progress_log("analyze_xgboost", f"sell analysis — {name}", step=step, total=AN_PHASES)

    resolved = df[df["resolved"] == True]
    early = df[df["resolved"] == False]
    progress_log("analyze_xgboost", "  exit type:")
    progress_log(
        "analyze_xgboost",
        f"    resolution (hold to end): {len(resolved)} ({100 * len(resolved) / len(df):.1f}%)",
    )
    progress_log(
        "analyze_xgboost",
        f"    early sell: {len(early)} ({100 * len(early) / len(df):.1f}%)",
    )

    progress_log("analyze_xgboost", "  sell price distribution:")
    progress_log("analyze_xgboost", f"    resolved @1.0 (win): {(resolved['sell_price'] == 1.0).sum()}")
    progress_log("analyze_xgboost", f"    resolved @0.0 (loss): {(resolved['sell_price'] == 0.0).sum()}")
    if len(early) > 0:
        progress_log(
            "analyze_xgboost",
            f"    early sell price min/max/mean={early['sell_price'].min():.3f}/{early['sell_price'].max():.3f}/{early['sell_price'].mean():.3f}",
        )
        progress_log("analyze_xgboost", f"    early sell avg PnL ${early['pnl'].mean():.2f}/trade, total ${early['pnl'].sum():.2f}")

    wins = resolved[resolved["sell_price"] == 1.0]
    losses = resolved[resolved["sell_price"] == 0.0]
    progress_log("analyze_xgboost", "  resolved exits:")
    progress_log(
        "analyze_xgboost",
        f"    wins @1.0: {len(wins)} trades, avg PnL ${wins['pnl'].mean():.2f}" if len(wins) > 0 else "    wins: 0",
    )
    progress_log(
        "analyze_xgboost",
        f"    losses @0.0: {len(losses)} trades, avg PnL ${losses['pnl'].mean():.2f}" if len(losses) > 0 else "    losses: 0",
    )

    if len(early) > 0:
        bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        early = early.copy()
        early["sell_bucket"] = pd.cut(early["sell_price"], bins=bins)
        sell_bucket = early.groupby("sell_bucket", observed=True).agg(
            trades=("pnl", "count"), avg_pnl=("pnl", "mean")
        ).sort_values("avg_pnl", ascending=False)
        progress_log("analyze_xgboost", "  early sell by sell-price bucket (table):")
        print(sell_bucket.to_string())


def main():
    path = RESULTS_DIR / "backtest_xgboost.csv"
    if not path.exists():
        progress_log("analyze_xgboost", "missing results/backtest_xgboost.csv — run scripts/run_backtest.py first.")
        sys.exit(1)

    progress_log(
        "analyze_xgboost",
        "start | phases: (1) load+overview (2) buy buckets (3) bootstrap (4) brackets/events/time (5–6) sells (7) takeaways",
    )
    df = pd.read_csv(path)

    progress_log(
        "analyze_xgboost",
        f"loaded {path.name} | trades={len(df)} total_pnl=${df['pnl'].sum():.2f} avg_profit_per_trade=${df['pnl'].mean():.2f}",
        step=1,
        total=AN_PHASES,
    )

    progress_log("analyze_xgboost", "by buy price — sorted by avg profit (table):", step=2, total=AN_PHASES)
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0]
    df["price_bucket"] = pd.cut(df["buy_price"], bins=bins)
    bucket_stats = df.groupby("price_bucket", observed=True).agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    ).sort_values("avg_pnl", ascending=False)
    print(bucket_stats.to_string())

    progress_log("analyze_xgboost", "bootstrap 95% CI (N=1000) avg $/trade:", step=3, total=AN_PHASES)
    np.random.seed(42)
    n_boot = 1000
    for bucket in bucket_stats.index:
        sub = df[df["price_bucket"] == bucket]["pnl"]
        n = len(sub)
        if n < 2:
            continue
        boot_means = [sub.sample(n, replace=True).mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        avg = sub.mean()
        warn = " (wide CI - noisy)" if (hi - lo) > 5 else ""
        progress_log("analyze_xgboost", f"  {bucket}: avg=${avg:.2f}, 95% CI [${lo:.2f}, ${hi:.2f}], N={n}{warn}")

    progress_log("analyze_xgboost", "by bracket (min 3 trades, table):", step=4, total=AN_PHASES)
    bracket_stats = df.groupby("bracket").agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        avg_buy_price=("buy_price", "mean"),
    )
    bracket_stats = bracket_stats[bracket_stats["trades"] >= 3].sort_values("avg_pnl", ascending=False)
    print(bracket_stats.to_string())

    progress_log("analyze_xgboost", "by event top 15 (min 3 trades, table):", step=4, total=AN_PHASES)
    event_stats = df.groupby("event_id").agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    )
    event_stats = event_stats[event_stats["trades"] >= 3].sort_values("avg_pnl", ascending=False)
    print(event_stats.head(15).to_string())

    if "buy_time" in df.columns:
        events_path = ROOT / "data" / "events.parquet"
        if events_path.exists():
            progress_log("analyze_xgboost", "by time from event start (tables):", step=4, total=AN_PHASES)
            events = pd.read_parquet(events_path)
            df_merge = df.copy()
            df_merge["event_id"] = df_merge["event_id"].astype(str)
            events_merge = events[["event_id", "start_date", "end_date"]].copy()
            events_merge["event_id"] = events_merge["event_id"].astype(str)
            merged = df_merge.merge(events_merge, on="event_id")
            merged["start_ts"] = pd.to_datetime(merged["start_date"]).astype("int64") // 10**9
            merged["hours_from_start"] = (merged["buy_time"] - merged["start_ts"]) / 3600
            merged["event_duration"] = (pd.to_datetime(merged["end_date"]).astype("int64") // 10**9 - merged["start_ts"]) / 3600
            merged["pct_elapsed"] = 100 * merged["hours_from_start"] / merged["event_duration"]
            # Time buckets
            time_bins = [0, 24, 48, 72, 96, 120, 168, 9999]
            merged["hours_bucket"] = pd.cut(merged["hours_from_start"], bins=time_bins)
            time_stats = merged.groupby("hours_bucket", observed=True).agg(
                trades=("pnl", "count"),
                avg_pnl=("pnl", "mean"),
            ).sort_values("avg_pnl", ascending=False)
            print(time_stats.to_string())
            pct_bins = [0, 20, 40, 60, 80, 100]
            merged["pct_bucket"] = pd.cut(merged["pct_elapsed"], bins=pct_bins)
            pct_stats = merged.groupby("pct_bucket", observed=True).agg(
                trades=("pnl", "count"),
                avg_pnl=("pnl", "mean"),
            ).sort_values("avg_pnl", ascending=False)
            progress_log("analyze_xgboost", "  by % event elapsed:")
            print(pct_stats.to_string())

    analyze_selling(df, "XGBoost", step=5)

    momentum_path = RESULTS_DIR / "backtest_momentum.csv"
    if momentum_path.exists():
        df_mom = pd.read_csv(momentum_path)
        analyze_selling(df_mom, "Momentum", step=6)

    progress_log("analyze_xgboost", "takeaways (maximize avg $ per $1 trade):", step=7, total=AN_PHASES)
    best_bucket = bucket_stats.index[0] if len(bucket_stats) > 0 else "N/A"
    best_bracket = bracket_stats.index[0] if len(bracket_stats) > 0 else "N/A"
    progress_log(
        "analyze_xgboost",
        f"  best buy-price bucket: {best_bucket} (avg ${bucket_stats.iloc[0]['avg_pnl']:.2f}/trade)"
        if len(bucket_stats) > 0
        else "  best buy-price bucket: N/A",
    )
    progress_log(
        "analyze_xgboost",
        f"  best bracket: {best_bracket} (avg ${bracket_stats.iloc[0]['avg_pnl']:.2f}/trade)"
        if len(bracket_stats) > 0
        else "  best bracket: N/A",
    )
    progress_log("analyze_xgboost", "  XGBoost: typically hold to resolution (sell 1.0 or 0.0).")
    progress_log("analyze_xgboost", "  Low-N buckets: trust bootstrap CI; training is 7-day-only (train_ml_model.py).")
    progress_log("analyze_xgboost", "done.")


if __name__ == "__main__":
    main()
