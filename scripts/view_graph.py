#!/usr/bin/env python3
"""
View balance and P&L graph from bot state and trade history.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared import progress_log  # noqa: E402

VIEW_PHASES = 3
STATE_DRY = ROOT / "bot" / "state_dry.json"
STATE_LEGACY = ROOT / "bot" / "state.json"
INITIAL_BALANCE = 100.0


def load_state() -> dict:
    path = STATE_DRY if STATE_DRY.exists() else STATE_LEGACY
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"balance": INITIAL_BALANCE, "trade_history": []}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="Save only, don't display")
    args = parser.parse_args()

    progress_log(
        "view_graph",
        "start | phases: (1) load state (2) build series (3) save/show plot",
    )
    state = load_state()
    history = state.get("trade_history", [])

    if not history:
        progress_log("view_graph", "no trade history — run the bot first.")
        return

    # Build cumulative P&L over time
    cum_pnl = 0
    timestamps = []
    cum_pnls = []
    trade_pnls = []

    for t in history:
        ts = t.get("ts", "")
        pnl = t.get("pnl", 0)
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        cum_pnl += pnl
        timestamps.append(dt)
        cum_pnls.append(cum_pnl)
        trade_pnls.append(pnl)

    if not timestamps:
        progress_log("view_graph", "no valid timestamps in trade_history.")
        return

    progress_log(
        "view_graph",
        f"loaded {len(timestamps)} trades with valid timestamps | balance=${state.get('balance', INITIAL_BALANCE):.2f}",
        step=1,
        total=VIEW_PHASES,
    )

    progress_log("view_graph", "rendering cumulative + per-trade panels…", step=2, total=VIEW_PHASES)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Cumulative P&L over time
    ax1 = axes[0]
    ax1.plot(timestamps, cum_pnls, "b-", linewidth=2)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    ax1.set_ylabel("Cumulative P&L ($)")
    ax1.set_title("Cumulative P&L Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))

    # Per-trade P&L
    ax2 = axes[1]
    colors = ["green" if p >= 0 else "red" for p in trade_pnls]
    ax2.bar(range(len(trade_pnls)), trade_pnls, color=colors, alpha=0.7)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("P&L ($)")
    ax2.set_xlabel("Trade #")
    ax2.set_title("Per-Trade P&L")
    ax2.grid(True, alpha=0.3)

    total_pnl = sum(trade_pnls)
    fig.suptitle(f"Trading Bot Performance | Total P&L: ${total_pnl:.2f} | Current Balance: ${state.get('balance', INITIAL_BALANCE):.2f}")
    plt.tight_layout()
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pnl_graph.png"
    plt.savefig(out_path, dpi=150)
    progress_log("view_graph", f"wrote {out_path} — done.", step=3, total=VIEW_PHASES)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
