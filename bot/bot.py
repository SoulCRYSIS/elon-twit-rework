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
import re
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
from shared import is_tweet_count_event, progress_log

DATA_DIR = ROOT / "data"
BOT_DIR = ROOT / "bot"
STATE_DRY_PATH = BOT_DIR / "state_dry.json"
STATE_LIVE_PATH = BOT_DIR / "state_live.json"
LEGACY_STATE_PATH = BOT_DIR / "state.json"

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
# Live discovery matches Polymarket UI: public-search indexes active Elon tweet windows
# even when tag_slug=elon-musk /events omits them.
GAMMA_SEARCH_Q = "elon musk tweets"
GAMMA_SEARCH_LIMIT_PER_TYPE = 50
MIN_EVENT_MARKETS = 5  # drop 2‑day / sparse stubs (working bot parity)

# Monthly tweet-count slugs (e.g. elon-musk-of-tweets-may-2026) — out of scope vs ~weekly windows.
_MONTH_YEAR_TWEET_SLUG = re.compile(
    r"-(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)-(20\d{2})$",
    re.I,
)


def _is_monthly_scope_tweet_slug(slug: str) -> bool:
    """True if slug is a calendar-month tweet market (…-may-2026), not a ~7d window."""
    if not slug:
        return False
    return bool(_MONTH_YEAR_TWEET_SLUG.search(slug.strip()))


def _event_log_suffix(eid, event: dict | None) -> str:
    """Short event context for trade logs (id + slug when available)."""
    parts = [f"event_id={eid}"]
    if event is not None and str(event.get("id")) == str(eid):
        slug = (event.get("slug") or "").strip()
        if slug:
            if len(slug) > 96:
                slug = slug[:93] + "…"
            parts.append(f"slug={slug}")
    return " ".join(parts)


INITIAL_BALANCE = 100.0
AVG_POSITION_USD = 1.0
POLL_INTERVAL_SEC = 300  # 5 min
MAX_OPEN_POSITIONS = 50
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


def _gamma_fetch_events_paginated(base_params: dict) -> list[dict]:
    """GET /events with limit/offset until empty (used for closed events / resolution)."""
    out: list[dict] = []
    limit = 100
    offset = 0
    closed = base_params.get("closed", "false")
    while True:
        params = {**base_params, "limit": limit, "offset": offset, "closed": closed}
        resp = requests.get(f"{GAMMA_BASE}/events", params=params, timeout=30)
        resp.raise_for_status()
        chunk = resp.json()
        if not chunk:
            break
        out.extend(chunk)
        if len(chunk) < limit:
            break
        offset += limit
        time.sleep(0.2)
    return out


def _event_duration_days(e: dict) -> int | None:
    try:
        start_dt = datetime.fromisoformat(e["startDate"].replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(e["endDate"].replace("Z", "+00:00"))
        return (end_dt - start_dt).days
    except (ValueError, TypeError, KeyError):
        return None


def _hydrate_event_by_slug(e: dict, min_markets: int) -> dict | None:
    """Full /events?slug= payload when search hits are missing enough markets."""
    slug = e.get("slug")
    if not slug:
        return None
    mkts = e.get("markets") or []
    if len(mkts) >= min_markets:
        return e
    resp = requests.get(f"{GAMMA_BASE}/events", params={"slug": slug}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict) and data.get("id"):
        return data
    return None


def _fetch_open_events_public_search(*, log_if_empty: bool) -> list[dict]:
    """
    Active Elon tweet windows: Gamma public-search + optional /events?slug= hydrate.
    Same pattern as working Polymarket bots (public-search indexes what /events tags omit).
    """
    resp = requests.get(
        f"{GAMMA_BASE}/public-search",
        params={
            "q": GAMMA_SEARCH_Q,
            "limit_per_type": GAMMA_SEARCH_LIMIT_PER_TYPE,
            "events_status": "active",
        },
        timeout=30,
    )
    resp.raise_for_status()
    raw_events = resp.json().get("events") or []
    n_search = len(raw_events)

    seen_ids: set = set()
    hydrated: list[dict] = []
    for e in raw_events:
        full = _hydrate_event_by_slug(e, MIN_EVENT_MARKETS)
        if not full:
            continue
        eid = full.get("id")
        if eid is None or eid in seen_ids:
            continue
        seen_ids.add(eid)
        hydrated.append(full)

    filtered: list[dict] = []
    for e in hydrated:
        if len(e.get("markets") or []) < MIN_EVENT_MARKETS:
            continue
        slug = e.get("slug", "") or ""
        title = e.get("title", "") or ""
        if not is_tweet_count_event(slug, title):
            continue
        if _is_monthly_scope_tweet_slug(slug):
            continue
        days = _event_duration_days(e)
        if days is None or days < 6:
            continue
        filtered.append(e)

    if log_if_empty and not filtered:
        progress_log(
            "bot",
            f"gamma public-search active q={GAMMA_SEARCH_Q!r}: API returned {n_search} event(s), "
            f"{len(hydrated)} after hydrate/dedupe → 0 passed "
            f"(≥{MIN_EVENT_MARKETS} markets, Elon tweet-count, ≥6d window, exclude monthly slugs)",
        )

    return filtered


def fetch_events(closed: bool = False, *, log_if_empty: bool = False) -> list[dict]:
    """
    Open: Gamma public-search (active) + /events?slug= hydrate — matches live Polymarket discovery.
    Closed: merged /events feeds (elon-musk + twitter) for broad resolution coverage.
    """
    if not closed:
        return _fetch_open_events_public_search(log_if_empty=log_if_empty)

    flag = str(closed).lower()
    raw: list[dict] = []
    for tag in ["elon-musk", "twitter"]:
        raw.extend(_gamma_fetch_events_paginated({"tag_slug": tag, "closed": flag}))

    seen: set = set()
    unique: list[dict] = []
    for e in raw:
        eid = e.get("id")
        if eid is None or eid in seen:
            continue
        seen.add(eid)
        unique.append(e)

    filtered: list[dict] = []
    for e in unique:
        slug = e.get("slug", "") or ""
        title = e.get("title", "") or ""
        if not is_tweet_count_event(slug, title):
            continue
        days = _event_duration_days(e)
        if days is None or days < 6:
            continue
        filtered.append(e)

    if log_if_empty and not filtered:
        progress_log(
            "bot",
            f"gamma closed: merged {len(unique)} unique event(s) (elon-musk + twitter) → 0 after filters",
        )

    return filtered


def fetch_active_events() -> list[dict]:
    return fetch_events(closed=False, log_if_empty=True)


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
                    progress_log(
                        "bot",
                        f"  resolved {_event_log_suffix(pos['event_id'], ev)} bracket={pos['bracket']} "
                        f"@ {resolved_price:.2f} PnL=${pnl:.2f}",
                    )
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
                                progress_log(
                                    "bot",
                                    f"  LIVE SELL failed {_event_log_suffix(eid, event)} bracket={bracket}: {res.message}",
                                )
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
                        progress_log(
                            "bot",
                            f"  sold {_event_log_suffix(eid, event)} bracket={bracket} "
                            f"@ {current_price:.3f} PnL=${pnl:.2f}",
                        )

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
                            progress_log(
                                "bot",
                                f"  LIVE BUY failed {_event_log_suffix(eid, event)} bracket={bracket}: {res.message}",
                            )
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
                    progress_log(
                        "bot",
                        f"  bought {_event_log_suffix(eid, event)} bracket={bracket} "
                        f"@ {current_price:.3f} size=${size:.2f}",
                    )

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
