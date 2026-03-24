#!/usr/bin/env python3
"""
Plot bot equity (cash balance) and closed-trade P&L sorted by buy date.

Reads bot/state_dry.json or bot/state_live.json (--live).

Outputs results/bot_equity_<mode>.png and results/bot_trades_by_buy_<mode>.png
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared import progress_log  # noqa: E402

STATE_DRY = ROOT / "bot" / "state_dry.json"
PLOT_BOT_PHASES = 3
STATE_LIVE = ROOT / "bot" / "state_live.json"
LEGACY = ROOT / "bot" / "state.json"
RESULTS = ROOT / "results"


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use state_live.json")
    parser.add_argument("--state", type=str, default="", help="Override state JSON path")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.state:
        path = Path(args.state)
    elif args.live:
        path = STATE_LIVE
    else:
        path = STATE_DRY if STATE_DRY.exists() else LEGACY

    progress_log(
        "plot_bot_performance",
        "start | phases: (1) load state (2) build figures (3) save PNGs",
    )
    state = load_state(path)
    if not state:
        progress_log("plot_bot_performance", f"no state at {path} — run the bot first.")
        return

    mode = "live" if args.live or "live" in path.name else "dry"
    th = state.get("trade_history", [])
    balance = float(state.get("balance", 0))
    initial = float(state.get("initial_balance", 100))

    RESULTS.mkdir(parents=True, exist_ok=True)
    progress_log(
        "plot_bot_performance",
        f"loaded {path.name} | mode={mode} trades={len(th)} balance=${balance:.2f}",
        step=1,
        total=PLOT_BOT_PHASES,
    )

    # --- Equity over time: reconstruct from trade_history cumulative + show current balance ---
    rows = []
    for t in th:
        ts = parse_ts(t.get("ts"))
        if ts is None:
            continue
        rows.append(
            {
                "event_ts": ts,
                "pnl": float(t.get("pnl", 0)),
                "action": t.get("action", ""),
                "bracket": t.get("bracket", ""),
            }
        )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        progress_log(
            "plot_bot_performance",
            "no trades with timestamps — single balance plot only.",
            step=2,
            total=PLOT_BOT_PHASES,
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axhline(initial, color="gray", linestyle="--", label="Start $100")
        ax.axhline(balance, color="blue", linewidth=2, label=f"Cash now ${balance:.2f}")
        ax.set_title(f"Bot cash [{mode}] — no trade history yet")
        ax.legend()
        ax.set_ylabel("USD")
        out = RESULTS / f"bot_equity_{mode}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        progress_log("plot_bot_performance", f"wrote {out} — done.", step=3, total=PLOT_BOT_PHASES)
        if not args.no_show:
            plt.show()
        return

    progress_log("plot_bot_performance", "building equity + closed-trade panels…", step=2, total=PLOT_BOT_PHASES)
    df = df.sort_values("event_ts")
    df["cum_pnl"] = df["pnl"].cumsum()
    df["equity_from_trades"] = initial + df["cum_pnl"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=False)

    ax0 = axes[0]
    ax0.plot(df["event_ts"], df["equity_from_trades"], color="steelblue", linewidth=2, label="Cash implied by sum(trade pnl)")
    ax0.axhline(initial, color="gray", linestyle="--", alpha=0.7, label=f"Start ${initial:.0f}")
    ax0.scatter(
        [df["event_ts"].iloc[-1]],
        [balance],
        color="darkgreen",
        s=120,
        zorder=5,
        label=f"Recorded balance ${balance:.2f}",
    )
    ax0.set_ylabel("USD")
    ax0.set_title(f"Bot equity [{mode}] — path from trade log + current balance (open positions not marked)")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.3)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

    # Closed trades with P&L: resolve + sell, sorted by buy_ts
    closed = [t for t in th if t.get("action") in ("resolve", "sell") and t.get("pnl") is not None]
    if not closed:
        axes[1].set_visible(False)
    else:
        recs = []
        for t in closed:
            buy_ts = parse_ts(t.get("buy_ts")) or parse_ts(t.get("ts"))
            recs.append(
                {
                    "buy_ts": buy_ts or datetime.min.replace(tzinfo=None),
                    "pnl": float(t.get("pnl", 0)),
                    "action": t.get("action"),
                    "bracket": t.get("bracket", ""),
                }
            )
        tdf = pd.DataFrame(recs).sort_values("buy_ts")
        ax1 = axes[1]
        x = range(len(tdf))
        colors = ["#2ca02c" if p >= 0 else "#d62728" for p in tdf["pnl"]]
        ax1.bar(x, tdf["pnl"], color=colors, alpha=0.85)
        ax1.axhline(0, color="black", linewidth=0.6)
        ax1.set_ylabel("P&L ($)")
        ax1.set_xlabel("Trade index (sorted by buy time)")
        ax1.set_title("Closed trade P&L (resolve + sell), oldest buy → newest")
        ax1.grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        f"Polymarket bot [{mode}] | state={path.name} | trades logged={len(th)} | open_pos={len(state.get('positions', []))}",
        fontsize=10,
    )
    plt.tight_layout()
    out_eq = RESULTS / f"bot_equity_{mode}.png"
    plt.savefig(out_eq, dpi=150)
    progress_log("plot_bot_performance", f"wrote {out_eq} — done.", step=3, total=PLOT_BOT_PHASES)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
