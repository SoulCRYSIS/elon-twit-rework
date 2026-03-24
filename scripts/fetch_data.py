#!/usr/bin/env python3
"""
Fetch and cache Polymarket data for Elon Musk 7-day tweet count markets.
Reusable across all prediction approaches.
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def is_tweet_count_event(slug: str, title: str = "") -> bool:
    """Filter for Elon Musk tweet count events only."""
    text = f"{slug} {title}".lower()
    if "tweet" not in text:
        return False
    # Must be Elon Musk tweet count (exclude Trump, CZ, etc.)
    return (
        "elon-musk-of-tweets" in slug
        or "of-elon-musk-tweets" in slug
        or ("elon" in text and "musk" in text and ("of-tweets" in slug or "tweets" in slug))
    )


def is_7day_event(start_date: str, end_date: str) -> bool:
    """Filter for 7-day events (exclude 2-day)."""
    try:
        start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        days = (end - start).days
        return days >= 6
    except (ValueError, TypeError):
        return False


def is_short_event(start_date: str, end_date: str, min_days: int = 2) -> bool:
    """Filter for events of min_days or more (e.g. 2-day, 3-day, 7-day)."""
    try:
        start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        days = (end - start).days
        return days >= min_days
    except (ValueError, TypeError):
        return False


def extract_bracket_range(question: str) -> str | None:
    """Extract bracket range from market question, e.g. '180-199' or '450+'."""
    # Pattern: "Will Elon Musk post 180-199 tweets..." or "Will Elon tweet 60-74 times?"
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", question)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"less than (\d+)", question, re.I)
    if m:
        return f"0-{int(m.group(1))-1}"
    m = re.search(r"(\d+)\+", question)
    if m:
        return f"{m.group(1)}+"
    m = re.search(r"more than (\d+)", question, re.I)
    if m:
        return f"{int(m.group(1))+1}+"
    return None


def fetch_events(refresh: bool = False) -> list[dict]:
    """Fetch all Elon Musk tweet events from Gamma API with pagination.
    Uses multiple tags (elon-musk, twitter) and recurrence=weekly to maximize
    coverage. Gamma API returns ~400 events per tag; 7-day Elon tweet events
    since 2024 typically yield 70-115 depending on filters.
    """
    all_events = []
    limit = 100

    # Fetch from both tags - tweet count markets appear under elon-musk and twitter
    for tag in ["elon-musk", "twitter"]:
        offset = 0
        while True:
            url = f"{GAMMA_BASE}/events"
            params = {
                "tag_slug": tag,
                "limit": limit,
                "offset": offset,
            }
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            events = resp.json()
            if not events:
                break
            all_events.extend(events)
            if len(events) < limit:
                break
            offset += limit
            time.sleep(0.2)

    # Also fetch recurrence=weekly (7-day tweet markets) - can surface different events
    offset = 0
    while True:
        url = f"{GAMMA_BASE}/events"
        params = {
            "tag_slug": "elon-musk",
            "recurrence": "weekly",
            "limit": limit,
            "offset": offset,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        events = resp.json()
        if not events:
            break
        all_events.extend(events)
        if len(events) < limit:
            break
        offset += limit
        time.sleep(0.2)

    # Fetch closed events explicitly - maximizes historical coverage for backtest
    for tag in ["elon-musk", "twitter"]:
        offset = 0
        while True:
            url = f"{GAMMA_BASE}/events"
            params = {
                "tag_slug": tag,
                "closed": "true",
                "limit": limit,
                "offset": offset,
            }
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            events = resp.json()
            if not events:
                break
            all_events.extend(events)
            if len(events) < limit:
                break
            offset += limit
            time.sleep(0.2)

    # Deduplicate by id
    seen = set()
    unique = []
    for e in all_events:
        if e["id"] not in seen:
            seen.add(e["id"])
            unique.append(e)

    return unique


def fetch_price_history(
    token_id: str,
    interval: str = "all",
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> list[dict]:
    """Fetch price history for a token from CLOB API."""
    url = f"{CLOB_BASE}/prices-history"
    params = {"market": token_id, "interval": interval}
    if start_ts is not None:
        params["startTs"] = start_ts
    if end_ts is not None:
        params["endTs"] = end_ts
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("history", [])


def main(refresh: bool = False, max_tokens: int = 0, include_shorter: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cache_meta_path = DATA_DIR / "last_updated.json"
    if not refresh and cache_meta_path.exists() and (DATA_DIR / "events.parquet").exists():
        with open(cache_meta_path) as f:
            meta = json.load(f)
        print(f"Using cached data from {meta.get('last_updated', 'unknown')}")
        print("Use --refresh to force re-fetch.")
        return

    print("Fetching events from Gamma API...")
    events = fetch_events(refresh=refresh)

    # Filter: tweet count + duration (7-day only or include 2-day+ for more backtest range)
    filtered = []
    for e in events:
        if not is_tweet_count_event(e.get("slug", ""), e.get("title", "")):
            continue
        if include_shorter:
            if not is_short_event(e.get("startDate", ""), e.get("endDate", ""), min_days=2):
                continue
        else:
            if not is_7day_event(e.get("startDate", ""), e.get("endDate", "")):
                continue
        filtered.append(e)

    label = "2+ day" if include_shorter else "7-day"
    print(f"Found {len(filtered)} {label} tweet count events (from {len(events)} total)")

    events_rows = []
    markets_rows = []
    token_ids = set()

    for event in filtered:
        eid = event["id"]
        slug = event.get("slug", "")
        start = event.get("startDate", "")
        end = event.get("endDate", "")
        try:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            duration = (end_dt - start_dt).days
        except (ValueError, TypeError):
            duration = 7

        events_rows.append(
            {
                "event_id": eid,
                "slug": slug,
                "start_date": start,
                "end_date": end,
                "duration_days": duration,
                "closed": event.get("closed", False),
            }
        )

        for m in event.get("markets", []):
            question = m.get("question", "")
            bracket = extract_bracket_range(question)
            if not bracket:
                continue

            raw_cids = m.get("clobTokenIds") or "[]"
            if isinstance(raw_cids, str):
                cids = json.loads(raw_cids) if raw_cids.startswith("[") else []
            else:
                cids = raw_cids
            if len(cids) < 2:
                continue

            yes_token = cids[0]
            no_token = cids[1]
            token_ids.add(yes_token)

            raw = m.get("outcomePrices") or '["0","0"]'
            if isinstance(raw, str):
                prices = json.loads(raw) if raw.startswith("[") else ["0", "0"]
            else:
                prices = raw
            yes_price = float(prices[0]) if prices else 0.0
            resolved_yes = 1 if (event.get("closed") and yes_price >= 0.99) else 0

            markets_rows.append(
                {
                    "event_id": eid,
                    "bracket_range": bracket,
                    "condition_id": m.get("conditionId", ""),
                    "yes_token_id": yes_token,
                    "no_token_id": no_token,
                    "resolved_yes": resolved_yes,
                    "question": question[:200],
                }
            )

    events_df = pd.DataFrame(events_rows)
    markets_df = pd.DataFrame(markets_rows)

    events_df.to_parquet(DATA_DIR / "events.parquet", index=False)
    markets_df.to_parquet(DATA_DIR / "markets.parquet", index=False)

    print(f"Saved {len(events_rows)} events, {len(markets_rows)} markets")

    # Build (token_id, start_ts, end_ts) for closed events
    event_times = {e["event_id"]: (e["start_date"], e["end_date"]) for e in events_rows}
    token_fetches = []
    for r in markets_rows:
        if not event_times.get(r["event_id"]):
            continue
        start_str, end_str = event_times[r["event_id"]]
        try:
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())
        except (ValueError, TypeError):
            continue
        token_fetches.append((r["yes_token_id"], start_ts, end_ts))

    # Dedupe by token (keep first event's range)
    seen = set()
    unique_fetches = []
    for tid, st, et in token_fetches:
        if tid not in seen:
            seen.add(tid)
            unique_fetches.append((tid, st, et))
    if max_tokens > 0:
        unique_fetches = unique_fetches[:max_tokens]
        print(f"Limiting to {max_tokens} tokens for price history")

    print("Fetching price history from CLOB API...")
    price_rows = []
    for i, (tid, start_ts, end_ts) in enumerate(unique_fetches):
        try:
            hist = fetch_price_history(tid, start_ts=start_ts, end_ts=end_ts)
            for h in hist:
                price_rows.append(
                    {"token_id": tid, "timestamp": h["t"], "price": h["p"]}
                )
        except Exception as ex:
            print(f"  Warning: failed for token {tid[:20]}...: {ex}")
        time.sleep(0.2)
        if (i + 1) % 20 == 0:
            print(f"  Fetched {i + 1}/{len(unique_fetches)} tokens")

    if price_rows:
        price_df = pd.DataFrame(price_rows)
        price_df.to_parquet(DATA_DIR / "price_history.parquet", index=False)
        print(f"Saved {len(price_rows)} price history rows")
    else:
        print("No price history fetched (API may have limits)")

    meta = {"last_updated": datetime.now().replace(microsecond=0).isoformat() + "Z", "event_count": len(events_rows)}
    with open(cache_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch all data")
    parser.add_argument("--max-tokens", type=int, default=0, help="Max tokens to fetch price history for (0=all)")
    parser.add_argument("--include-shorter", action="store_true", help="Include 2-5 day events (more backtest range, ~30 extra events)")
    args = parser.parse_args()
    main(refresh=args.refresh, max_tokens=args.max_tokens, include_shorter=args.include_shorter)
