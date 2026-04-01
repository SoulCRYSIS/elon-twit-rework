#!/usr/bin/env python3
"""
Live trading bot for Elon Musk 7-day tweet count markets.

- Default: xgboost, dry-run, $100 start, state in bot/state_dry_xgboost.json (legacy bot/state_dry.json migrates once)
- Multi dry-run: --approach xgboost,xgboost_pick,xgboost_ev_m08 → separate state_dry_<name>.json per model
- Live: --live, state in bot/state_live.json (requires Polymarket CLOB env vars)
- Retrain: fetch_data + train_ml_model at most every 7 local calendar days (from last successful run)
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
from shared import in_event_buy_warmup, is_tweet_count_event, progress_log

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

# Throttle resolve debug lines so polls do not spam logs.
_last_resolve_missing_gamma_log_ts = 0.0
_RESOLVE_MISSING_GAMMA_LOG_INTERVAL_SEC = 600.0
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
POLL_INTERVAL_SEC = 1800  # 15 min
MAX_OPEN_POSITIONS = 50
COOLDOWN_HOURS = 12
RETRAIN_INTERVAL_DAYS = 7
MIN_PRICE_CHANGE = 2.0
DEFAULT_APPROACH = "xgboost_live"


def _approach_state_slug(approach: str) -> str:
    s = approach.strip()
    if not s:
        return DEFAULT_APPROACH
    return re.sub(r"[^a-zA-Z0-9._-]", "_", s)


def state_path(live: bool, approach: str) -> Path:
    slug = _approach_state_slug(approach)
    if live:
        return BOT_DIR / f"state_live_{slug}.json"
    return BOT_DIR / f"state_dry_{slug}.json"


def load_state(live: bool, approach: str) -> dict:
    path = state_path(live, approach)
    if path.exists():
        with open(path) as f:
            state = json.load(f)
        _ensure_state_shape(state, live)
        return state

    # One-time: legacy flat dry file → per-approach file (default approach only)
    if (
        not live
        and _approach_state_slug(approach) == _approach_state_slug(DEFAULT_APPROACH)
        and STATE_DRY_PATH.exists()
    ):
        with open(STATE_DRY_PATH) as f:
            state = json.load(f)
        _ensure_state_shape(state, live=False)
        save_state(False, approach, state)
        progress_log("bot", f"migrated {STATE_DRY_PATH.name} → {path.name}")
        return state

    # Migrate legacy single state.json into dry-run file once
    if not live and LEGACY_STATE_PATH.exists():
        with open(LEGACY_STATE_PATH) as f:
            state = json.load(f)
        _ensure_state_shape(state, live=False)
        save_state(False, approach, state)
        progress_log("bot", f"migrated {LEGACY_STATE_PATH.name} → {path.name}")
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


def save_state(live: bool, approach: str, state: dict) -> None:
    BOT_DIR.mkdir(parents=True, exist_ok=True)
    state["live"] = live
    state["approach"] = approach
    path = state_path(live, approach)
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
        progress_log(
            "bot",
            "daily_train 2/2: train_ml_model.py (baseline + xgb_ev_pnl_m08, same snapshot) …",
        )
        r2 = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "train_ml_model.py"),
                "--also-xgb-ev-m08",
            ],
            cwd=str(ROOT),
            timeout=900,
        )
        if r2.returncode != 0:
            progress_log("bot", "daily_train 2/2: train_ml_model.py failed (non-zero exit).")
            return False
        for _mod in (
            "approaches.approach_xgboost",
            "approaches.approach_xgboost_pick",
            "approaches.approach_xgboost_ev",
            "approaches.approach_xgboost_ev_m08",
        ):
            try:
                m = __import__(_mod, fromlist=["clear_cache"])
                if hasattr(m, "clear_cache"):
                    m.clear_cache()
            except Exception:
                pass
        progress_log("bot", "daily_train 2/2: train_ml_model.py finished OK — ML caches cleared.")
        progress_log("bot", "daily_train: complete.")
        return True
    except Exception as e:
        progress_log("bot", f"daily_train: error — {e}")
        return False


def should_run_daily_train(state: dict) -> bool:
    """True if we never trained or last successful train was >= RETRAIN_INTERVAL_DAYS ago (local date)."""
    last = state.get("last_daily_train_date")
    if not last:
        return True
    try:
        last_d = date.fromisoformat(str(last))
    except ValueError:
        return True
    return (date.today() - last_d).days >= RETRAIN_INTERVAL_DAYS


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


def fetch_gamma_event_by_id(event_id) -> dict | None:
    """Single event from Gamma (works for active or closed; avoids missing tag-paginated closed feeds)."""
    try:
        resp = requests.get(
            f"{GAMMA_BASE}/events",
            params={"id": str(event_id)},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict) and data.get("id") is not None:
            return data
    except Exception as e:
        progress_log("bot", f"warning: Gamma GET /events?id={event_id} failed: {e}")
    return None


def yes_payoff_if_settled(outcome_prices: list[float]) -> float | None:
    """
    Polymarket lists YES first in tweet-count outcomePrices.
    Return 1.0 / 0.0 when resolved; None while UMA/oracle still ambiguous (do not cash out early).
    """
    if not outcome_prices:
        return None
    p_yes = float(outcome_prices[0])
    p_no = float(outcome_prices[1]) if len(outcome_prices) > 1 else max(0.0, 1.0 - p_yes)
    if p_yes >= 0.99 or (p_yes >= 0.97 and p_no <= 0.03):
        return 1.0
    if p_yes <= 0.01 or p_no >= 0.99:
        return 0.0
    return None


def legacy_yes_payoff(outcome_prices: list[float]) -> float:
    """Original rule: YES wins only if first outcome ≥ 0.99, else treat as loss."""
    if not outcome_prices:
        return 0.0
    return 1.0 if float(outcome_prices[0]) >= 0.99 else 0.0


def _event_end_timestamp(ev: dict) -> float | None:
    raw = ev.get("endDate") or ev.get("end_date")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _is_closed_flag(x) -> bool:
    if x is True:
        return True
    if isinstance(x, str) and x.lower() in ("true", "1", "yes"):
        return True
    return False


# After event end, allow legacy 0/1 rule if prices stay stuck (Polymarket delay).
RESOLVE_LEGACY_GRACE_SEC = 36 * 3600


def choose_resolve_payoff(
    prices: list[float],
    *,
    ev: dict,
    market: dict,
    now_ts: float,
) -> float | None:
    """
    Prefer clear settlement from prices; if still ambiguous, use legacy rule once event/market
    is closed or we are past endDate + grace (so positions do not hang forever).
    """
    settled = yes_payoff_if_settled(prices)
    if settled is not None:
        return settled
    end_ts = _event_end_timestamp(ev)
    past_grace = end_ts is not None and now_ts >= end_ts + RESOLVE_LEGACY_GRACE_SEC
    if _is_closed_flag(ev.get("closed")) or _is_closed_flag(market.get("closed")) or past_grace:
        return legacy_yes_payoff(prices)
    return None


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


def run_loop(live: bool, approaches: list[str]) -> None:
    from approaches import get_approach

    if live and len(approaches) > 1:
        progress_log("bot", "ERROR: multiple approaches are dry-run only; use a single --approach with --live.")
        sys.exit(1)

    if live:
        from bot.live_executor import live_configured, missing_live_env

        if not live_configured():
            progress_log("bot", "ERROR: --live requires environment variables:")
            for k in missing_live_env():
                progress_log("bot", f"  {k}")
            progress_log("bot", "See bot/live_executor.py docstring and bot/README.md")
            sys.exit(1)

    mode = "LIVE" if live else "DRY-RUN"
    multi = len(approaches) > 1
    state_names = ", ".join(state_path(live, a).name for a in approaches)
    progress_log(
        "bot",
        f"loop start [{mode}] approach(es)={', '.join(approaches)} | state file(s): {state_names}",
    )

    while True:
        now = time.time()
        now_utc = datetime.now(timezone.utc)

        states = {a: load_state(live, a) for a in approaches}
        for a in approaches:
            states[a]["approach"] = a

        if any(should_run_daily_train(states[a]) for a in approaches):
            if run_daily_training():
                tday = date.today().isoformat()
                for a in approaches:
                    states[a]["last_daily_train_date"] = tday
            for a in approaches:
                save_state(live, a, states[a])

        pending_eids: set[str] = set()
        for a in approaches:
            for p in states[a].get("positions", []):
                pending_eids.add(str(p["event_id"]))

        events_by_id: dict[str, dict] = {}
        for eid in sorted(pending_eids):
            ev = fetch_gamma_event_by_id(eid)
            if ev:
                events_by_id[eid] = ev
            time.sleep(0.08)

        resolve_ts = time.time()
        missing_gamma = pending_eids - set(events_by_id.keys())
        global _last_resolve_missing_gamma_log_ts
        if missing_gamma and (
            resolve_ts - _last_resolve_missing_gamma_log_ts >= _RESOLVE_MISSING_GAMMA_LOG_INTERVAL_SEC
        ):
            _last_resolve_missing_gamma_log_ts = resolve_ts
            progress_log(
                "bot",
                "resolve: Gamma /events?id returned no event for "
                f"{len(missing_gamma)} id(s): {', '.join(sorted(missing_gamma)[:12])}"
                + (" …" if len(missing_gamma) > 12 else "")
                + " — positions cannot clear until IDs match API (or network works).",
            )

        for approach in approaches:
            state = states[approach]
            pfx = f"[{approach}] " if multi else ""
            nomatch_log = state.setdefault("_resolve_nomatch_log_ts", {})
            from shared import extract_bracket_range

            for pos in list(state["positions"]):
                eid = str(pos.get("event_id"))
                ev = events_by_id.get(eid)
                if not ev:
                    continue
                br_pos = pos.get("bracket")
                matched_market = None
                for m in ev.get("markets", []):
                    if extract_bracket_range(m.get("question", "")) != br_pos:
                        continue
                    matched_market = m
                    break

                if matched_market is None:
                    lk = f"{eid}:{br_pos}"
                    last = float(nomatch_log.get(lk, 0.0))
                    if resolve_ts - last >= 3600.0:
                        nomatch_log[lk] = resolve_ts
                        sample = [
                            extract_bracket_range(x.get("question", ""))
                            for x in (ev.get("markets") or [])[:24]
                        ]
                        sample = [x for x in sample if x]
                        progress_log(
                            "bot",
                            f"  {pfx}resolve: event {eid} no market for bracket={br_pos!r} "
                            f"(parsed brackets sample: {sample[:10]}{'…' if len(sample) > 10 else ''})",
                        )
                    continue

                prices = parse_outcome_prices(matched_market.get("outcomePrices", "[]"))
                resolved_price = choose_resolve_payoff(
                    prices,
                    ev=ev,
                    market=matched_market,
                    now_ts=resolve_ts,
                )
                if resolved_price is None:
                    continue
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
                    f"  {pfx}resolved {_event_log_suffix(pos['event_id'], ev)} bracket={pos['bracket']} "
                    f"@ {resolved_price:.2f} PnL=${pnl:.2f}",
                )
            save_state(live, approach, state)

        events = fetch_active_events()
        if not events:
            progress_log("bot", f"poll: no active 7-day tweet events — sleep {POLL_INTERVAL_SEC}s …")
            for a in approaches:
                save_state(live, a, states[a])
            time.sleep(POLL_INTERVAL_SEC)
            continue

        price_df = None
        if (DATA_DIR / "price_history.parquet").exists():
            price_df = pd.read_parquet(DATA_DIR / "price_history.parquet")

        for approach in approaches:
            state = states[approach]
            get_signal = get_approach(approach)
            pfx = f"[{approach}] " if multi else ""

            for event in events:
                eid = event["id"]
                start_dt = datetime.fromisoformat(event["startDate"].replace("Z", "+00:00"))
                start_ts = start_dt.timestamp()
                end_dt = datetime.fromisoformat(event["endDate"].replace("Z", "+00:00"))
                end_ts = end_dt.timestamp()

                markets_list = event.get("markets", [])
                pick_ctx = None
                if approach == "xgboost_pick":
                    from approaches.approach_xgboost_pick import build_pick_context
                    from shared import extract_bracket_range as _ebr

                    pick_rows = []
                    for m2 in markets_list:
                        q2 = m2.get("question", "")
                        br2 = _ebr(q2)
                        if not br2:
                            continue
                        cids_raw2 = m2.get("clobTokenIds") or "[]"
                        if isinstance(cids_raw2, str):
                            import json as _j

                            cids2 = _j.loads(cids_raw2) if cids_raw2.startswith("[") else []
                        else:
                            cids2 = cids_raw2
                        if len(cids2) < 2:
                            continue
                        tid2 = cids2[0]
                        pr2 = m2.get("outcomePrices") or '["0.05","0.95"]'
                        px2 = parse_outcome_prices(pr2)
                        cp2 = px2[0] if px2 else 0.05
                        if cp2 < 0.01:
                            continue
                        ph2 = []
                        if price_df is not None:
                            sub2 = price_df[price_df["token_id"] == tid2]
                            if len(sub2) > 0:
                                sub2 = sub2.sort_values("timestamp")
                                ph2 = [(r["timestamp"], r["price"]) for _, r in sub2.iterrows()]
                        pick_rows.append(
                            {
                                "bracket": br2,
                                "price_history": ph2,
                                "current_price": cp2,
                                "market_data": m2,
                            }
                        )
                    if pick_rows:
                        pick_ctx = build_pick_context(pick_rows, start_ts, end_ts, now)

                for m in markets_list:
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
                    sig_kw = {}
                    if approach == "xgboost_pick":
                        sig_kw["pick_context"] = pick_ctx
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
                        **sig_kw,
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
                            f"  {pfx}sold {_event_log_suffix(eid, event)} bracket={bracket} "
                            f"@ {current_price:.3f} PnL=${pnl:.2f}",
                        )

                    if (
                        signal.buy
                        and not in_event_buy_warmup(start_ts, now)
                        and float(state["balance"]) >= AVG_POSITION_USD
                        and current_price >= 0.01
                    ):
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
                            f"  {pfx}bought {_event_log_suffix(eid, event)} bracket={bracket} "
                            f"@ {current_price:.3f} size=${size:.2f}",
                        )

            save_state(live, approach, state)

        if multi:
            parts = [
                f"{a} bal=${float(states[a]['balance']):.2f} pos={len(states[a]['positions'])}"
                for a in approaches
            ]
            progress_log(
                "bot",
                f"poll end [{mode}] " + " | ".join(parts) + f" — sleep {POLL_INTERVAL_SEC}s",
            )
        else:
            a0 = approaches[0]
            st = states[a0]
            progress_log(
                "bot",
                f"poll end [{mode}] balance=${float(st['balance']):.2f} "
                f"open_positions={len(st['positions'])}/{MAX_OPEN_POSITIONS} — sleep {POLL_INTERVAL_SEC}s",
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
    parser.add_argument(
        "--approach",
        default=DEFAULT_APPROACH,
        help="One approach, or comma-separated for parallel dry-run "
        "(e.g. xgboost,xgboost_pick,xgboost_ev_m08). Each gets its own state file under bot/.",
    )
    parser.add_argument(
        "--poll-sec",
        type=int,
        default=default_poll,
        help=f"Seconds between loops (default {default_poll})",
    )
    args = parser.parse_args()

    POLL_INTERVAL_SEC = max(60, args.poll_sec)

    approaches = [x.strip() for x in args.approach.split(",") if x.strip()]
    if not approaches:
        approaches = [DEFAULT_APPROACH]

    live = args.live
    if live and len(approaches) > 1:
        progress_log("bot", "ERROR: --live allows only one approach (no comma-separated list).")
        sys.exit(1)
    if live:
        progress_log(
            "bot",
            f"WARNING: LIVE mode — real funds at risk. State: {state_path(True, approaches[0])}",
        )
    run_loop(live=live, approaches=approaches)


if __name__ == "__main__":
    main()
