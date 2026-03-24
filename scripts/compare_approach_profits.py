#!/usr/bin/env python3
"""
Run backtests for many approaches and rank by **profit** (avg $/trade, total PnL), not accuracy.

Uses the same engine as run_backtest.py. Excludes `lstm` by default (slow / torch).

Example:
  python scripts/compare_approach_profits.py
  python scripts/compare_approach_profits.py --max-events 15
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Curated suite: entry-only vs *_exit (tiered TP/SL from buy_price)
DEFAULT_APPROACHES = [
    "xgboost",
    "xgb_exit",
    "ml",
    "ml_exit",
    "historical",
    "historical_exit",
    "negbin",
    "negbin_exit",
    "momentum",
    "momentum_tier",
    "rules",
    "rules_exit",
    "price_filter",
    "price_filter_exit",
    "kelly",
    "kelly_exit",
    "regime",
    "regime_exit",
    "ranking",
    "ranking_exit",
    "bracket_spread",
    "bracket_spread_exit",
    "time_weighted",
    "time_weighted_exit",
    "ensemble",
    "ensemble_exit",
]


def _load_run_backtest():
    path = ROOT / "scripts" / "run_backtest.py"
    spec = importlib.util.spec_from_file_location("run_backtest_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument(
        "--approaches",
        type=str,
        default="",
        help="Comma-separated; default = built-in profit-comparison suite",
    )
    parser.add_argument("--skip", type=str, default="lstm", help="Comma names to never run")
    args = parser.parse_args()

    rb = _load_run_backtest()
    events, markets, prices = rb.load_data()
    if args.max_events > 0:
        closed_ids = events[events["closed"] == True]["event_id"].head(args.max_events).tolist()
        events = events[events["event_id"].isin(closed_ids)]
        markets = markets[markets["event_id"].isin(closed_ids)]
        print(f"Limited to {len(closed_ids)} events")

    names = (
        [x.strip() for x in args.approaches.split(",") if x.strip()]
        if args.approaches
        else DEFAULT_APPROACHES
    )
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}
    names = [n for n in names if n not in skip]

    rows = []
    for name in names:
        print(f"Running {name}...", flush=True)
        try:
            r = rb.run_backtest_for_approach(name, events, markets, prices)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            rows.append(
                {
                    "approach": name,
                    "error": str(e),
                    "total_trades": 0,
                    "avg_profit_per_trade": None,
                    "total_pnl": None,
                }
            )
            continue
        trades = r.get("trades") or []
        early = sum(1 for t in trades if not t.get("resolved", True))
        resolved = len(trades) - early
        rows.append(
            {
                "approach": name,
                "total_trades": r["total_trades"],
                "avg_profit_per_trade": round(r["avg_profit_per_trade"], 4),
                "total_pnl": round(r["total_pnl"], 2),
                "win_rate": round(r["win_rate"], 4),
                "max_drawdown": round(r["max_drawdown"], 2),
                "early_sell_trades": early,
                "resolved_trades": resolved,
            }
        )

    df = pd.DataFrame(rows)
    df_ok = df[df["avg_profit_per_trade"].notna()].copy()
    df_ok = df_ok.sort_values("avg_profit_per_trade", ascending=False)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "approach_profit_comparison.json"
    df.to_json(out_json, orient="records", indent=2)
    print(f"\nWrote {out_json}")

    print("\n## Ranked by avg profit per $1 trade (higher is better)\n")
    if len(df_ok) == 0:
        print("No successful runs.")
        return
    print(df_ok.to_string(index=False))

    best = df_ok.iloc[0]
    print("\n---")
    if int(best["total_trades"]) < 10:
        print(
            f"(Note: top approach has only {int(best['total_trades'])} trades — "
            "avg $/trade is very noisy; prefer total PnL or re-run on full data.)\n"
        )
    print(
        f"Best by avg $/trade: **{best['approach']}** "
        f"(avg ${best['avg_profit_per_trade']:.4f}/trade, "
        f"total PnL ${best['total_pnl']:.2f}, "
        f"n={int(best['total_trades'])} trades, "
        f"{int(best['early_sell_trades'])} early exits / {int(best['resolved_trades'])} at resolution)"
    )
    by_total = df_ok.sort_values("total_pnl", ascending=False).iloc[0]
    if by_total["approach"] != best["approach"]:
        print(
            f"Best by **total PnL**: **{by_total['approach']}** (${by_total['total_pnl']:.2f}, "
            f"n={int(by_total['total_trades'])} trades)"
        )
    print(
        "\nWhy: *_exit variants add **tiered TP/SL/time-stop** as a function of **avg buy_price** "
        "(see `approaches/exit_policy.py`); base models mostly hold to resolution. "
        "Momentum sell only when `position` is set. For joint ML on exit, extend training rows with "
        "`buy_price` + path features (future work) or tune `data/ml_artifacts/exit_tiers.json`."
    )


if __name__ == "__main__":
    main()
