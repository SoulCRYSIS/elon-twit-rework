#!/usr/bin/env python3
"""
Summary table for the three XGB stacks: xgboost (base), xgboost_pick, xgboost_ev_m08.

Expects ``results/backtest_summary.json`` from::

  python scripts/run_backtest.py --approaches xgboost,xgboost_pick,xgboost_ev_m08

Writes ``results/xgb_models_summary_table.csv`` and prints the table.
Metrics align with ``ml_summary_table.py`` (accuracy %, avg $/trade, trades/month).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DEFAULT_MODELS = ("xgboost", "xgboost_pick", "xgboost_ev_m08")


def closed_events_month_span(events: pd.DataFrame) -> tuple[float, str]:
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
        "--summary-json",
        type=Path,
        default=RESULTS_DIR / "backtest_summary.json",
        help="Output from run_backtest.py",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="If >0, match run_backtest subset (first N closed event ids).",
    )
    args = parser.parse_args()

    if not (DATA_DIR / "events.parquet").exists():
        print("Missing data/events.parquet", file=sys.stderr)
        sys.exit(1)
    if not args.summary_json.exists():
        print(f"Missing {args.summary_json} — run run_backtest.py first.", file=sys.stderr)
        sys.exit(1)

    events = pd.read_parquet(DATA_DIR / "events.parquet")
    if args.max_events > 0:
        closed_ids = events[events["closed"] == True]["event_id"].head(args.max_events).tolist()
        events = events[events["event_id"].isin(closed_ids)]

    months, span_note = closed_events_month_span(events)

    with open(args.summary_json) as f:
        summary = json.load(f)

    rows = []
    for name in DEFAULT_MODELS:
        if name not in summary:
            rows.append(
                {
                    "model": name,
                    "accuracy_pct": None,
                    "average_profit_usd": None,
                    "trades_per_month": None,
                    "total_trades": 0,
                    "note": "missing in backtest_summary.json",
                }
            )
            continue
        s = summary[name]
        n = int(s.get("total_trades") or 0)
        wr = float(s.get("win_rate") or 0.0)
        avg = float(s.get("avg_profit_per_trade") or 0.0)
        rows.append(
            {
                "model": name,
                "accuracy_pct": round(wr * 100, 2),
                "average_profit_usd": round(avg, 4),
                "trades_per_month": round(n / months, 2) if months > 0 else 0.0,
                "total_trades": n,
                "total_pnl": round(float(s.get("total_pnl", 0)), 4),
                "sharpe": round(float(s.get("sharpe", 0)), 4),
                "note": "",
            }
        )

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "xgb_models_summary_table.csv"
    out.to_csv(out_path, index=False)

    print(f"Backtest window (closed events): {span_note}")
    print(f"Month span used: {months:.3f}\n")
    disp = out.drop(columns=["note"], errors="ignore") if "note" in out.columns else out
    print(disp.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
