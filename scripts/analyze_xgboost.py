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
RESULTS_DIR = ROOT / "results"


def analyze_selling(df: pd.DataFrame, name: str) -> None:
    """Analyze how positions are sold."""
    print("\n" + "=" * 60)
    print(f"SELL ANALYSIS - {name}")
    print("=" * 60)

    resolved = df[df["resolved"] == True]
    early = df[df["resolved"] == False]
    print(f"\nExit type:")
    print(f"  Resolution (hold to end): {len(resolved)} ({100*len(resolved)/len(df):.1f}%)")
    print(f"  Early sell:              {len(early)} ({100*len(early)/len(df):.1f}%)")

    # Sell price distribution
    print(f"\nSell price distribution:")
    print(f"  Resolved at 1.0 (win): {(resolved['sell_price']==1.0).sum()}")
    print(f"  Resolved at 0.0 (loss): {(resolved['sell_price']==0.0).sum()}")
    if len(early) > 0:
        print(f"  Early sell price: min={early['sell_price'].min():.3f}, max={early['sell_price'].max():.3f}, mean={early['sell_price'].mean():.3f}")
        print(f"  Early sell avg PnL: ${early['pnl'].mean():.2f}/trade")
        print(f"  Early sell total PnL: ${early['pnl'].sum():.2f}")

    # Resolved sell analysis
    wins = resolved[resolved["sell_price"] == 1.0]
    losses = resolved[resolved["sell_price"] == 0.0]
    print(f"\nResolved exits:")
    print(f"  Wins (sell@1.0):  {len(wins)} trades, avg PnL ${wins['pnl'].mean():.2f}" if len(wins) > 0 else "  Wins: 0")
    print(f"  Losses (sell@0.0): {len(losses)} trades, avg PnL ${losses['pnl'].mean():.2f}" if len(losses) > 0 else "  Losses: 0")

    if len(early) > 0:
        # Early sell by sell_price bucket
        bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        early = early.copy()
        early["sell_bucket"] = pd.cut(early["sell_price"], bins=bins)
        sell_bucket = early.groupby("sell_bucket", observed=True).agg(
            trades=("pnl", "count"), avg_pnl=("pnl", "mean")
        ).sort_values("avg_pnl", ascending=False)
        print(f"\nEarly sell by sell price bucket:")
        print(sell_bucket.to_string())


def main():
    path = RESULTS_DIR / "backtest_xgboost.csv"
    if not path.exists():
        print("Run backtest first: python scripts/run_backtest.py")
        sys.exit(1)

    df = pd.read_csv(path)
    # PnL is already per $1 position (shares = 1/buy_price, pnl = (sell-buy)*shares)

    print("=" * 60)
    print("XGBoost Analysis: AVERAGE PROFIT PER TRADE ($1 per trade)")
    print("=" * 60)
    print(f"\nTotal trades: {len(df)}")
    print(f"Total PnL: ${df['pnl'].sum():.2f}")
    print(f"AVG PROFIT PER TRADE: ${df['pnl'].mean():.2f}")

    # --- By price bucket (sorted by avg PnL) ---
    print("\n" + "-" * 50)
    print("BY BUY PRICE (cents) - sorted by avg profit per trade")
    print("-" * 50)
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0]
    df["price_bucket"] = pd.cut(df["buy_price"], bins=bins)
    bucket_stats = df.groupby("price_bucket", observed=True).agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    ).sort_values("avg_pnl", ascending=False)
    print(bucket_stats.to_string())

    # --- Bootstrap 95% CI for small-sample buckets (esp 0-5¢) ---
    print("\n" + "-" * 50)
    print("BOOTSTRAP 95% CI (N=1000) - avg $ per trade")
    print("-" * 50)
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
        print(f"  {bucket}: avg=${avg:.2f}, 95% CI [${lo:.2f}, ${hi:.2f}], N={n}{warn}")

    # --- By bracket (sorted by avg PnL, min 3 trades) ---
    print("\n" + "-" * 50)
    print("BY BRACKET - sorted by avg profit per trade (min 3 trades)")
    print("-" * 50)
    bracket_stats = df.groupby("bracket").agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        avg_buy_price=("buy_price", "mean"),
    )
    bracket_stats = bracket_stats[bracket_stats["trades"] >= 3].sort_values("avg_pnl", ascending=False)
    print(bracket_stats.to_string())

    # --- By event ---
    print("\n" + "-" * 50)
    print("BY EVENT - sorted by avg profit per trade (min 3 trades)")
    print("-" * 50)
    event_stats = df.groupby("event_id").agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    )
    event_stats = event_stats[event_stats["trades"] >= 3].sort_values("avg_pnl", ascending=False)
    print(event_stats.head(15).to_string())

    # --- Time analysis ---
    if "buy_time" in df.columns:
        events_path = ROOT / "data" / "events.parquet"
        if events_path.exists():
            print("\n" + "-" * 50)
            print("BY TIME (hours from event start) - avg profit per trade")
            print("-" * 50)
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
            # % elapsed buckets
            pct_bins = [0, 20, 40, 60, 80, 100]
            merged["pct_bucket"] = pd.cut(merged["pct_elapsed"], bins=pct_bins)
            pct_stats = merged.groupby("pct_bucket", observed=True).agg(
                trades=("pnl", "count"),
                avg_pnl=("pnl", "mean"),
            ).sort_values("avg_pnl", ascending=False)
            print("\nBy % event elapsed:")
            print(pct_stats.to_string())

    # --- SELL ANALYSIS ---
    analyze_selling(df, "XGBoost")

    # Also analyze momentum (has early sells)
    momentum_path = RESULTS_DIR / "backtest_momentum.csv"
    if momentum_path.exists():
        df_mom = pd.read_csv(momentum_path)
        analyze_selling(df_mom, "Momentum")

    # --- Top takeaways ---
    print("\n" + "=" * 60)
    print("TOP TAKEAWAYS (maximize avg $ per $1 trade)")
    print("=" * 60)
    best_bucket = bucket_stats.index[0] if len(bucket_stats) > 0 else "N/A"
    best_bracket = bracket_stats.index[0] if len(bracket_stats) > 0 else "N/A"
    print(f"Best price range: {best_bucket} (avg ${bucket_stats.iloc[0]['avg_pnl']:.2f}/trade)")
    print(f"Best bracket: {best_bracket} (avg ${bracket_stats.iloc[0]['avg_pnl']:.2f}/trade)")
    print("\nXGBoost sell behavior: holds to resolution only (no early sell).")
    print("  sell_price = 1.0 (win) or 0.0 (loss) always.")
    print("\nNote: 0-5¢ bucket has few trades; one big win can dominate. Bootstrap CI shows uncertainty.")
    print("  XGBoost is trained on 7-day events only (see scripts/train_ml_model.py).")
    print("=" * 60)


if __name__ == "__main__":
    main()
