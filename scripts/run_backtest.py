#!/usr/bin/env python3
"""
Backtest all prediction approaches on historical 7-day tweet count events.
"""

import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from approaches.position_context import aggregate_bracket_position

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
STEP_HOURS = 6
POSITION_SIZE_USD = 1.0
COOLDOWN_HOURS = 6  # Don't re-buy same bracket for N hours
MIN_PRICE_CHANGE = 1.0  # Require 2x price move (100%) to add to position (vs last buy)


def load_data():
    """Load cached events, markets, price history."""
    events = pd.read_parquet(DATA_DIR / "events.parquet")
    markets = pd.read_parquet(DATA_DIR / "markets.parquet")
    prices = pd.read_parquet(DATA_DIR / "price_history.parquet")
    return events, markets, prices


def get_price_at(prices_df: pd.DataFrame, token_id: str, ts: float) -> float | None:
    """Get YES price at timestamp (closest before or at ts)."""
    sub = prices_df[prices_df["token_id"] == token_id]
    if len(sub) == 0:
        return None
    before = sub[sub["timestamp"] <= ts]
    if len(before) == 0:
        after = sub[sub["timestamp"] >= ts]
        return float(after.iloc[0]["price"]) if len(after) > 0 else None
    return float(before.iloc[-1]["price"])


def get_price_history_up_to(prices_df: pd.DataFrame, token_id: str, ts: float) -> list[tuple[float, float]]:
    """Get (timestamp, price) list up to ts."""
    sub = prices_df[(prices_df["token_id"] == token_id) & (prices_df["timestamp"] <= ts)]
    if len(sub) == 0:
        return []
    sub = sub.sort_values("timestamp")
    return [(r["timestamp"], r["price"]) for _, r in sub.iterrows()]


def run_backtest_for_approach(
    approach_name: str,
    events: pd.DataFrame,
    markets: pd.DataFrame,
    prices: pd.DataFrame,
    cooldown_hours: float = COOLDOWN_HOURS,
    min_price_change: float = MIN_PRICE_CHANGE,
) -> dict:
    """Run backtest for one approach, return metrics and trade log."""
    from approaches import get_approach

    get_signal = get_approach(approach_name)

    closed = events[events["closed"] == True]
    trades = []
    balance = 0.0

    for _, ev in closed.iterrows():
        eid = ev["event_id"]
        start = pd.to_datetime(ev["start_date"]).timestamp()
        end = pd.to_datetime(ev["end_date"]).timestamp()

        mkts = markets[markets["event_id"] == eid]
        if len(mkts) == 0:
            continue

        # Get winning bracket for resolution
        winner = mkts[mkts["resolved_yes"] == 1]
        winning_bracket = winner.iloc[0]["bracket_range"] if len(winner) > 0 else None

        positions = []  # [{bracket, token_id, buy_price, amount_usd, shares, buy_time}]
        last_buy_times = {}  # (event_id, bracket) -> timestamp
        last_buy_prices = {}  # (event_id, bracket) -> price

        t = start
        while t < end:
            # Pre-compute all bracket prices and sorted brackets for ranking/spread
            all_bracket_prices = {}
            all_brackets_sorted = []
            for _, m in mkts.iterrows():
                bracket = m["bracket_range"]
                p = get_price_at(prices, m["yes_token_id"], t)
                if p is not None and p >= 0.001:
                    all_bracket_prices[bracket] = p
            if all_bracket_prices:
                from approaches.utils import parse_bracket
                all_brackets_sorted = sorted(
                    [(b, parse_bracket(b)) for b in all_bracket_prices],
                    key=lambda x: (x[1][0] + x[1][1]) / 2,
                )

            # For ranking: pre-compute all edges via historical
            all_bracket_edges = {}
            if approach_name == "ranking" and all_bracket_prices:
                from approaches.approach_historical import get_signal as hist_sig
                for _, m in mkts.iterrows():
                    bracket = m["bracket_range"]
                    if bracket not in all_bracket_prices:
                        continue
                    ph = get_price_history_up_to(prices, m["yes_token_id"], t)
                    h = hist_sig(event_id=eid, bracket=bracket, current_time=t,
                                 price_history=ph, market_data=m.to_dict(),
                                 current_price=all_bracket_prices[bracket], end_time=end)
                    if h.edge is not None:
                        all_bracket_edges[bracket] = h.edge

            for _, m in mkts.iterrows():
                bracket = m["bracket_range"]
                token_id = m["yes_token_id"]
                ph = get_price_history_up_to(prices, token_id, t)
                curr_price = get_price_at(prices, token_id, t)
                if curr_price is None or curr_price < 0.001:
                    continue

                market_data = m.to_dict()
                pos_ctx = aggregate_bracket_position(positions, bracket)
                kwargs = {
                    "event_id": eid,
                    "bracket": bracket,
                    "current_time": t,
                    "start_time": start,
                    "price_history": ph,
                    "market_data": market_data,
                    "current_price": curr_price,
                    "end_time": end,
                    "position": pos_ctx,
                }
                if approach_name == "ranking":
                    kwargs["all_bracket_edges"] = all_bracket_edges
                if approach_name == "bracket_spread":
                    kwargs["all_bracket_prices"] = all_bracket_prices
                    kwargs["all_brackets_sorted"] = all_brackets_sorted
                signal = get_signal(**kwargs)

                # Check sell for existing positions
                if signal.sell:
                    to_remove = [i for i, pos in enumerate(positions) if pos["bracket"] == bracket]
                    for i in reversed(to_remove):
                        pos = positions[i]
                        pnl = (curr_price - pos["buy_price"]) * pos["shares"]
                        balance += pnl
                        trades.append(
                            {
                                "event_id": eid,
                                "bracket": bracket,
                                "buy_price": pos["buy_price"],
                                "buy_time": pos.get("buy_time", t),
                                "sell_price": curr_price,
                                "pnl": pnl,
                                "resolved": False,
                                "approach": approach_name,
                            }
                        )
                        positions.pop(i)

                # Check buy (with cooldown and min price change for re-entry)
                if signal.buy:
                    key = (eid, bracket)
                    last_t = last_buy_times.get(key, 0)
                    last_p = last_buy_prices.get(key, curr_price)
                    cooldown_ok = (t - last_t) >= cooldown_hours * 3600
                    price_change_ok = abs(curr_price - last_p) / max(0.001, last_p) >= min_price_change
                    # First buy: no cooldown. Re-entry: need both cooldown and price change
                    can_buy = cooldown_ok and (last_t == 0 or price_change_ok)
                    if can_buy:
                        if signal.kelly_fraction is not None and signal.kelly_fraction > 0:
                            size = min(POSITION_SIZE_USD * 2, POSITION_SIZE_USD * (1 + signal.kelly_fraction))
                            size = max(POSITION_SIZE_USD * 0.5, min(size, POSITION_SIZE_USD * 2))
                        else:
                            size = POSITION_SIZE_USD
                        shares = size / curr_price
                        positions.append(
                            {
                                "bracket": bracket,
                                "token_id": token_id,
                                "buy_price": curr_price,
                                "amount_usd": size,
                                "shares": shares,
                                "buy_time": t,
                            }
                        )
                        last_buy_times[key] = t
                        last_buy_prices[key] = curr_price

            t += STEP_HOURS * 3600

        # Resolve remaining positions at end (all positions, including multiple per bracket)
        for pos in positions:
            resolved_price = 1.0 if pos["bracket"] == winning_bracket else 0.0
            pnl = (resolved_price - pos["buy_price"]) * pos["shares"]
            balance += pnl
            trades.append(
                {
                    "event_id": eid,
                    "bracket": pos["bracket"],
                    "buy_price": pos["buy_price"],
                    "buy_time": pos.get("buy_time", end),
                    "sell_price": resolved_price,
                    "pnl": pnl,
                    "resolved": True,
                    "approach": approach_name,
                }
            )

    if not trades:
        return {
            "approach": approach_name,
            "total_trades": 0,
            "win_rate": 0,
            "avg_profit_per_trade": 0,
            "total_pnl": 0,
            "max_drawdown": 0,
            "sharpe": 0,
            "trades": [],
        }

    df = pd.DataFrame(trades)
    total_pnl = df["pnl"].sum()
    wins = (df["pnl"] > 0).sum()
    n = len(df)
    avg_profit = total_pnl / n

    # Max drawdown
    cum = df["pnl"].cumsum()
    peak = cum.cummax()
    drawdown = peak - cum
    max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0

    # Sharpe-like (annualized)
    std = df["pnl"].std()
    sharpe = (avg_profit / std * (52 ** 0.5)) if std > 0 else 0  # ~weekly trades

    return {
        "approach": approach_name,
        "total_trades": n,
        "win_rate": wins / n,
        "avg_profit_per_trade": avg_profit,
        "total_pnl": total_pnl,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": trades,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=0, help="Max events to backtest (0=all)")
    parser.add_argument("--cooldown-hours", type=float, default=COOLDOWN_HOURS, help="Hours before re-buying same bracket")
    parser.add_argument("--min-price-change", type=float, default=MIN_PRICE_CHANGE, help="Min price change (0.03=3%%) to add to position")
    parser.add_argument("--approaches", type=str, default="", help="Comma-separated subset (default: all)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not (DATA_DIR / "events.parquet").exists():
        print("Run scripts/fetch_data.py --refresh first to fetch data.")
        sys.exit(1)

    print("Loading data...")
    events, markets, prices = load_data()
    if args.max_events > 0:
        closed_ids = events[events["closed"] == True]["event_id"].head(args.max_events).tolist()
        events = events[events["event_id"].isin(closed_ids)]
        markets = markets[markets["event_id"].isin(closed_ids)]
        print(f"  Limited to {len(closed_ids)} events")
    print(f"  Events: {len(events)}, Markets: {len(markets)}, Price rows: {len(prices)}")

    from approaches import APPROACHES

    names = [x.strip() for x in args.approaches.split(",") if x.strip()] if args.approaches else list(APPROACHES)
    for n in names:
        if n not in APPROACHES:
            print(f"Unknown approach: {n}")
            sys.exit(1)

    summary = {}
    for name in names:
        print(f"\nBacktesting {name}...")
        result = run_backtest_for_approach(
            name, events, markets, prices,
            cooldown_hours=args.cooldown_hours,
            min_price_change=args.min_price_change,
        )
        summary[name] = {
            "total_trades": result["total_trades"],
            "win_rate": result["win_rate"],
            "avg_profit_per_trade": result["avg_profit_per_trade"],
            "total_pnl": result["total_pnl"],
            "max_drawdown": result["max_drawdown"],
            "sharpe": result["sharpe"],
        }
        # Save per-approach trade log
        pd.DataFrame(result["trades"]).to_csv(RESULTS_DIR / f"backtest_{name}.csv", index=False)
        print(f"  Trades: {result['total_trades']}, Avg PnL: {result['avg_profit_per_trade']:.4f}, Win rate: {result['win_rate']:.2%}")

    with open(RESULTS_DIR / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    for name, s in sorted(summary.items(), key=lambda x: -x[1]["avg_profit_per_trade"]):
        print(f"  {name}: avg_profit={s['avg_profit_per_trade']:.4f}, trades={s['total_trades']}, win_rate={s['win_rate']:.2%}")
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
