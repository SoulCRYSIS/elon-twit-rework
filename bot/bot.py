#!/usr/bin/env python3
"""
Live trading bot for Elon Musk 7-day tweet count markets.

- Default: xgboost, dry-run, $100 start, state in bot/state_dry.json
- Live: --live, state in bot/state_live.json (requires Polymarket CLOB env vars)
- Retrain: fetch_data + train_ml_model once per local calendar day (first loop after midnight)
- Max 20 open positions; state persists positions/balance/history across restarts

Run 24/7: use systemd, pm2, or `nohup python bot/bot.py &` (see bot/README.md).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from approaches.position_context import aggregate_bracket_position
from shared import progress_log

DATA_DIR = ROOT / "data"
BOT_DIR = ROOT / "bot"
STATE_DRY_PATH = BOT_DIR / "state_dry.json"
STATE_LIVE_PATH = BOT_DIR / "state_live.json"
LEGACY_STATE_PATH = BOT_DIR / "state.json"

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

INITIAL_BALANCE = 100.0
AVG_POSITION_USD = 1.0
POLL_INTERVAL_SEC = 300  # 5 min
MAX_OPEN_POSITIONS = 20
COOLDOWN_HOURS = 6
MIN_PRICE_CHANGE = 1.0
DEFAULT_APPROACH = "xgboost"


def state_path(live: bool) -> Path:
    return STATE_LIVE_PATH if live else STATE_DRY_PATH


def load_state(live: bool) -> dict:
    path = state_path(live)
    if path.exists():
        with open(path) as f:
            state = json.load(f)
        _ensure_state_shape(state, live)
        return state

    # Migrate legacy single state.json into dry-run file once
    if not live and LEGACY_STATE_PATH.exists():
        with open(LEGACY_STATE_PATH) as f:
            state = json.load(f)
        _ensure_state_shape(state, live=False)
        save_state(False, state)
        progress_log("bot", f"migrated {LEGACY_STATE_PATH.name} → {STATE_DRY_PATH.name}")
        return state

    state = _fresh_state(live)
    return state


def _fresh_state(live: bool) -> dict:
    return {
        "balance": INITIAL_BALANCE,
        "positions": [],
        "trade_history": [],
        "last_daily_train_date": None,
        "approach": DEFAULT_APPROACH,
        "last_buy_times": {},
        "last_buy_prices": {},
        "live": live,
        "initial_balance": INITIAL_BALANCE,
    }


def _ensure_state_shape(state: dict, live: bool) -> None:
    state.setdefault("balance", INITIAL_BALANCE)
    state.setdefault("positions", [])
    state.setdefault("trade_history", [])
    state.setdefault("last_daily_train_date", None)
    state.setdefault("approach", DEFAULT_APPROACH)
    state.setdefault("last_buy_times", {})
    state.setdefault("last_buy_prices", {})
    state.setdefault("live", live)
    state.setdefault("initial_balance", INITIAL_BALANCE)


def save_state(live: bool, state: dict) -> None:
    BOT_DIR.mkdir(parents=True, exist_ok=True)
    state["live"] = live
    path = state_path(live)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(path)


def run_daily_training() -> bool:
    """Fetch latest data + retrain XGBoost artifacts."""
    progress_log(
        "bot",
        "daily_train | steps: (1) scripts/fetch_data.py (2) scripts/train_ml_model.py — child logs follow",
    )
    try:
        progress_log("bot", "daily_train 1/2: fetch_data.py …")
        r1 = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "fetch_data.py")],
            cwd=str(ROOT),
            timeout=900,
        )
        if r1.returncode != 0:
            progress_log("bot", "daily_train 1/2: fetch_data.py failed (non-zero exit).")
            return False
        progress_log("bot", "daily_train 1/2: fetch_data.py finished OK.")
        progress_log("bot", "daily_train 2/2: train_ml_model.py …")
        r2 = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_ml_model.py")],
            cwd=str(ROOT),
            timeout=600,
        )
        if r2.returncode != 0:
            progress_log("bot", "daily_train 2/2: train_ml_model.py failed (non-zero exit).")
            return False
        try:
            from approaches.approach_xgboost import clear_cache as xgb_clear_cache

            xgb_clear_cache()
        except Exception:
            pass
        progress_log("bot", "daily_train 2/2: train_ml_model.py finished OK — xgboost cache cleared.")
        progress_log("bot", "daily_train: complete.")
        return True
    except Exception as e:
        progress_log("bot", f"daily_train: error — {e}")
        return False


def should_run_daily_train(state: dict) -> bool:
    today = date.today().isoformat()
    return state.get("last_daily_train_date") != today


def fetch_events(closed: bool = False) -> list[dict]:
    all_events = []
    offset = 0
    limit = 50

    while True:
        resp = requests.get(
            f"{GAMMA_BASE}/events",
            params={"tag_slug": "elon-musk", "closed": str(closed).lower(), "limit": limit, "offset": offset},
            timeout=30,
        )
        resp.raise_for_status()
        events = resp.json()
        if not events:
            break
        all_events.extend(events)
        if len(events) < limit:
            break
        offset += limit
        time.sleep(0.2)

    filtered = []
    for e in all_events:
        slug = e.get("slug", "").lower()
        title = e.get("title", "").lower()
        if "tweet" not in slug and "tweet" not in title:
            continue
        start = e.get("startDate", "")
        end = e.get("endDate", "")
        try:
            from datetime import datetime as dt

            start_dt = dt.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = dt.fromisoformat(end.replace("Z", "+00:00"))
            if (end_dt - start_dt).days < 6:
                continue
        except (ValueError, TypeError):
            continue
        filtered.append(e)
    return filtered


def fetch_active_events() -> list[dict]:
    return fetch_events(closed=False)


def parse_outcome_prices(raw) -> list[float]:
    if isinstance(raw, str):
        import json as j

        raw = j.loads(raw) if raw.startswith("[") else [0, 0]
    return [float(x) for x in raw] if raw else [0, 0]


def _append_trade(
    state: dict,
    *,
    action: str,
    event_id,
    bracket: str,
    price: float,
    pnl: float,
    buy_ts: str | None = None,
    extra: dict | None = None,
) -> None:
    row = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event_id": event_id,
        "bracket": bracket,
        "action": action,
        "price": price,
        "pnl": pnl,
        "buy_ts": buy_ts,
        "live": state.get("live", False),
    }
    if extra:
        row.update(extra)
    state.setdefault("trade_history", []).append(row)


def run_loop(live: bool, approach: str) -> None:
    from approaches import get_approach

    if live:
        from bot.live_executor import live_configured, missing_live_env

        if not live_configured():
            progress_log("bot", "ERROR: --live requires environment variables:")
            for k in missing_live_env():
                progress_log("bot", f"  {k}")
            progress_log("bot", "See bot/live_executor.py docstring and bot/README.md")
            sys.exit(1)

    get_signal = get_approach(approach)
    state = load_state(live)
    state["approach"] = approach

    mode = "LIVE" if live else "DRY-RUN"
    progress_log(
        "bot",
        f"loop start [{mode}] approach={approach} balance=${float(state['balance']):.2f} "
        f"open_positions={len(state.get('positions', []))}/{MAX_OPEN_POSITIONS} state={state_path(live)}",
    )

    while True:
        now = time.time()
        now_utc = datetime.now(timezone.utc)

        if should_run_daily_train(state):
            if run_daily_training():
                state["last_daily_train_date"] = date.today().isoformat()
            save_state(live, state)

        # Resolve closed events
        if state["positions"]:
            position_eids = {str(p["event_id"]) for p in state["positions"]}
            closed_events = [e for e in fetch_events(closed=True) if str(e["id"]) in position_eids]
            closed_ids = {e["id"] for e in closed_events}
        else:
            closed_ids = set()
            closed_events = []

        closed_ids = {str(x) for x in closed_ids}
        for pos in list(state["positions"]):
            if str(pos.get("event_id")) not in closed_ids:
                continue
            ev = next((e for e in closed_events if str(e["id"]) == str(pos["event_id"])), None)
            if not ev:
                continue
            for m in ev.get("markets", []):
                from shared import extract_bracket_range

                if extract_bracket_range(m.get("question", "")) == pos.get("bracket"):
                    prices = parse_outcome_prices(m.get("outcomePrices", "[]"))
                    resolved_price = 1.0 if (prices and prices[0] >= 0.99) else 0.0
                    pnl = (resolved_price - pos["buy_price"]) * pos["shares"]
                    state["balance"] = float(state["balance"]) + pnl
                    buy_ts = pos.get("buy_time_iso") or pos.get("buy_ts")
                    _append_trade(
                        state,
                        action="resolve",
                        event_id=pos["event_id"],
                        bracket=pos["bracket"],
                        price=resolved_price,
                        pnl=pnl,
                        buy_ts=buy_ts,
                    )
                    state["positions"].remove(pos)
                    progress_log("bot", f"  resolved {pos['bracket']} @ {resolved_price:.2f} PnL=${pnl:.2f}")
                    break

        events = fetch_active_events()
        if not events:
            progress_log("bot", f"poll: no active 7-day tweet events — sleep {POLL_INTERVAL_SEC}s …")
            save_state(live, state)
            time.sleep(POLL_INTERVAL_SEC)
            continue

        price_df = None
        if (DATA_DIR / "price_history.parquet").exists():
            price_df = pd.read_parquet(DATA_DIR / "price_history.parquet")

        n_pos = len(state["positions"])

        for event in events:
            eid = event["id"]
            start_dt = datetime.fromisoformat(event["startDate"].replace("Z", "+00:00"))
            start_ts = start_dt.timestamp()
            end_dt = datetime.fromisoformat(event["endDate"].replace("Z", "+00:00"))
            end_ts = end_dt.timestamp()

            for m in event.get("markets", []):
                question = m.get("question", "")
                from shared import extract_bracket_range

                bracket = extract_bracket_range(question)
                if not bracket:
                    continue

                cids_raw = m.get("clobTokenIds") or "[]"
                if isinstance(cids_raw, str):
                    import json as j

                    cids = j.loads(cids_raw) if cids_raw.startswith("[") else []
                else:
                    cids = cids_raw
                if len(cids) < 2:
                    continue
                token_id = cids[0]

                prices_raw = m.get("outcomePrices") or '["0.05","0.95"]'
                prices = parse_outcome_prices(prices_raw)
                current_price = prices[0] if prices else 0.05

                ph = []
                if price_df is not None:
                    sub = price_df[price_df["token_id"] == token_id]
                    if len(sub) > 0:
                        sub = sub.sort_values("timestamp")
                        ph = [(r["timestamp"], r["price"]) for _, r in sub.iterrows()]

                pos_ctx = aggregate_bracket_position(state["positions"], bracket, eid)
                signal = get_signal(
                    event_id=eid,
                    bracket=bracket,
                    current_time=now,
                    start_time=start_ts,
                    price_history=ph,
                    market_data=m,
                    current_price=current_price,
                    end_time=end_ts,
                    position=pos_ctx,
                )

                # Early sell (e.g. *_exit approaches)
                if signal.sell:
                    to_remove = [
                        i
                        for i, p in enumerate(state["positions"])
                        if p.get("bracket") == bracket and p.get("event_id") == eid
                    ]
                    for i in reversed(to_remove):
                        pos = state["positions"][i]
                        pnl = (current_price - pos["buy_price"]) * pos["shares"]
                        buy_ts = pos.get("buy_time_iso")
                        if live:
                            from bot.live_executor import market_sell_yes

                            res = market_sell_yes(
                                str(pos["token_id"]),
                                float(current_price),
                                float(pos["shares"]),
                            )
                            if not res.ok:
                                progress_log("bot", f"  LIVE SELL failed {bracket}: {res.message}")
                                continue
                        state["balance"] = float(state["balance"]) + pnl
                        _append_trade(
                            state,
                            action="sell",
                            event_id=eid,
                            bracket=bracket,
                            price=current_price,
                            pnl=pnl,
                            buy_ts=buy_ts,
                            extra={"live_fill": live},
                        )
                        state["positions"].pop(i)
                        progress_log("bot", f"  sold {bracket} @ {current_price:.3f} PnL=${pnl:.2f}")

                if signal.buy and float(state["balance"]) >= AVG_POSITION_USD and current_price >= 0.01:
                    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
                        continue
                    key = f"{eid}:{bracket}"
                    last_ts = state.get("last_buy_times", {}).get(key)
                    last_p = state.get("last_buy_prices", {}).get(key, current_price)
                    if last_ts:
                        last_t = datetime.fromisoformat(last_ts.replace("Z", "+00:00")).timestamp()
                    else:
                        last_t = 0
                    cooldown_ok = (now - last_t) >= COOLDOWN_HOURS * 3600
                    price_change_ok = abs(current_price - last_p) / max(0.001, last_p) >= MIN_PRICE_CHANGE
                    can_buy = cooldown_ok and (last_t == 0 or price_change_ok)
                    if not can_buy:
                        continue

                    size = min(AVG_POSITION_USD, float(state["balance"]) * 0.5)
                    shares = size / current_price
                    buy_iso = now_utc.isoformat().replace("+00:00", "Z")

                    if live:
                        from bot.live_executor import market_buy_yes

                        res = market_buy_yes(str(token_id), float(current_price), float(shares))
                        if not res.ok:
                            progress_log("bot", f"  LIVE BUY failed {bracket}: {res.message}")
                            continue

                    state["balance"] = float(state["balance"]) - size
                    state["positions"].append(
                        {
                            "event_id": eid,
                            "bracket": bracket,
                            "token_id": token_id,
                            "buy_price": current_price,
                            "amount_usd": size,
                            "shares": shares,
                            "buy_time": now,
                            "buy_time_iso": buy_iso,
                        }
                    )
                    state.setdefault("last_buy_times", {})[key] = buy_iso
                    state.setdefault("last_buy_prices", {})[key] = current_price
                    _append_trade(
                        state,
                        action="buy",
                        event_id=eid,
                        bracket=bracket,
                        price=current_price,
                        pnl=-size,
                        buy_ts=buy_iso,
                    )
                    progress_log("bot", f"  bought {bracket} @ {current_price:.3f} size=${size:.2f}")

        save_state(live, state)
        progress_log(
            "bot",
            f"poll end [{mode}] balance=${float(state['balance']):.2f} "
            f"open_positions={len(state['positions'])}/{MAX_OPEN_POSITIONS} — sleep {POLL_INTERVAL_SEC}s",
        )
        time.sleep(POLL_INTERVAL_SEC)


def main():
    global POLL_INTERVAL_SEC
    default_poll = POLL_INTERVAL_SEC
    parser = argparse.ArgumentParser(description="Polymarket Elon tweet bot (xgboost default)")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Real CLOB orders (requires POLYMARKET_* env vars). Default is dry-run.",
    )
    parser.add_argument("--approach", default=DEFAULT_APPROACH, help="Signal approach (default: xgboost)")
    parser.add_argument(
        "--poll-sec",
        type=int,
        default=default_poll,
        help=f"Seconds between loops (default {default_poll})",
    )
    args = parser.parse_args()

    POLL_INTERVAL_SEC = max(60, args.poll_sec)

    live = args.live
    if live:
        progress_log("bot", f"WARNING: LIVE mode — real funds at risk. State: {STATE_LIVE_PATH}")
    run_loop(live=live, approach=args.approach)


if __name__ == "__main__":
    main()
