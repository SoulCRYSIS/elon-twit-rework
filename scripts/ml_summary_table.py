#!/usr/bin/env python3
"""
Build a compact summary table from ML experiment backtest CSV:

- accuracy: % of trades with positive PnL (same as backtest win_rate)
- average_profit: mean $ PnL per trade ($1 stake engine)
- trades_per_month: total_trades / months spanned by closed events in the backtest window

Reads: results/ml_experiment_results.csv, data/events.parquet
Writes: results/ml_summary_table.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def closed_events_month_span(events: pd.DataFrame) -> tuple[float, str]:
    """
    Months between earliest closed-event start and latest closed-event end
    (fractional, ~30.44 days/month). At least 1e-6 to avoid div-by-zero.
    """
    ev = events[events["closed"] == True]
    if ev.empty:
        return 1.0, "(no closed events)"
    start = pd.to_datetime(ev["start_date"], utc=True).min()
    end = pd.to_datetime(ev["end_date"], utc=True).max()
    days = max((end - start).total_seconds() / 86400.0, 1.0)
    months = max(days / 30.437, 1e-6)
    note = f"{start.date()} → {end.date()} ({days:.0f}d ≈ {months:.2f} mo)"
    return months, note


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "ml_experiment_results.csv",
        help="Experiment results CSV from run_ml_experiments.py",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="If >0, match run_ml_experiments subset (first N closed event ids).",
    )
    args = parser.parse_args()

    if not (DATA_DIR / "events.parquet").exists():
        print("Missing data/events.parquet", file=sys.stderr)
        sys.exit(1)
    if not args.input.exists():
        print(f"Missing {args.input} — run scripts/run_ml_experiments.py first.", file=sys.stderr)
        sys.exit(1)

    events = pd.read_parquet(DATA_DIR / "events.parquet")
    if args.max_events > 0:
        closed_ids = events[events["closed"] == True]["event_id"].head(args.max_events).tolist()
        events = events[events["event_id"].isin(closed_ids)]

    months, span_note = closed_events_month_span(events)

    df = pd.read_csv(args.input)
    ok = df["error"].isna() & (df["total_trades"] > 0)
    df = df[ok].copy()
    df["accuracy_pct"] = (df["win_rate"] * 100).round(2)
    df["average_profit_usd"] = df["avg_profit_per_trade"].round(4)
    df["trades_per_month"] = (df["total_trades"] / months).round(2)

    out = df[
        [
            "experiment",
            "classifier",
            "accuracy_pct",
            "average_profit_usd",
            "trades_per_month",
            "total_trades",
        ]
    ].sort_values("average_profit_usd", ascending=False)

    out_path = RESULTS_DIR / "ml_summary_table.csv"
    out.to_csv(out_path, index=False)

    print(f"Backtest window (closed events): {span_note}")
    print(f"Month span used: {months:.3f}\n")
    print(out.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
