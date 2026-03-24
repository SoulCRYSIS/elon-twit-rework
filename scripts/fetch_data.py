#!/usr/bin/env python3
"""
Fetch and cache Polymarket data for Elon Musk 7-day tweet count markets.
Reusable across all prediction approaches.
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from shared import is_tweet_count_event, progress_log  # noqa: E402

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
# /prices-history returns 400 "startTs and endTs interval is too long" for windows > ~15 days.
CLOB_PRICES_MAX_SPAN_SEC = 14 * 86400
CLOB_CHUNK_OVERLAP_SEC = 3600
DATA_DIR = _ROOT / "data"
STEPS_TOTAL = 5


def log(msg: str, *, step: int | None = None) -> None:
    progress_log("fetch_data", msg, step=step, total=STEPS_TOTAL if step is not None else None)


def _gamma_paginate(
    phase_label: str,
    base_params: dict,
    *,
    verbose: bool,
) -> list[dict]:
    """GET /events with limit/offset until empty; log each page."""
    out: list[dict] = []
    limit = 100
    offset = 0
    page = 0
    while True:
        params = {**base_params, "limit": limit, "offset": offset}
        resp = requests.get(f"{GAMMA_BASE}/events", params=params, timeout=30)
        resp.raise_for_status()
        chunk = resp.json()
        if not chunk:
            break
        out.extend(chunk)
        page += 1
        if verbose:
            log(
                f"Gamma «{phase_label}» page {page}: +{len(chunk)} events (subtotal {len(out)})",
                step=1,
            )
        if len(chunk) < limit:
            break
        offset += limit
        time.sleep(0.2)
    return out


def fetch_events(verbose: bool = True) -> list[dict]:
    """Fetch all Elon-adjacent events from Gamma (open + closed + weekly)."""
    all_events: list[dict] = []

    if verbose:
        log("Downloading Gamma event lists (5 API passes, paginated)…", step=1)

    for tag in ["elon-musk", "twitter"]:
        all_events.extend(
            _gamma_paginate(f"open tag={tag}", {"tag_slug": tag}, verbose=verbose)
        )

    all_events.extend(
        _gamma_paginate(
            "open weekly recurrence",
            {"tag_slug": "elon-musk", "recurrence": "weekly"},
            verbose=verbose,
        )
    )

    for tag in ["elon-musk", "twitter"]:
        all_events.extend(
            _gamma_paginate(f"closed tag={tag}", {"tag_slug": tag, "closed": "true"}, verbose=verbose)
        )

    seen: set = set()
    unique: list[dict] = []
    for e in all_events:
        if e["id"] not in seen:
            seen.add(e["id"])
            unique.append(e)

    if verbose:
        log(f"Gamma done: {len(all_events)} raw rows → {len(unique)} unique event IDs", step=1)

    return unique


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


def _clob_prices_history_request(token_id: str, params: dict) -> list[dict]:
    url = f"{CLOB_BASE}/prices-history"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("history", [])


def fetch_price_history(
    token_id: str,
    interval: str = "all",
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> list[dict]:
    """
    Fetch price history for a token from CLOB API.

    When ``start_ts`` and ``end_ts`` are set, the API rejects ranges longer than ~15 days;
    we split into overlapping chunks and merge (dedupe by timestamp).
    Bounded requests use ``interval=max`` (``interval=all`` + explicit bounds often 400s).
    """
    if start_ts is not None and end_ts is not None:
        if end_ts <= start_ts:
            return []
        span = end_ts - start_ts
        if span <= CLOB_PRICES_MAX_SPAN_SEC:
            return _clob_prices_history_request(
                token_id,
                {
                    "market": token_id,
                    "startTs": start_ts,
                    "endTs": end_ts,
                    "interval": "max",
                },
            )
        merged: dict[int, float] = {}
        cur = start_ts
        while cur < end_ts:
            chunk_end = min(end_ts, cur + CLOB_PRICES_MAX_SPAN_SEC)
            hist = _clob_prices_history_request(
                token_id,
                {
                    "market": token_id,
                    "startTs": cur,
                    "endTs": chunk_end,
                    "interval": "max",
                },
            )
            for h in hist:
                merged[int(h["t"])] = float(h["p"])
            if chunk_end >= end_ts:
                break
            cur = chunk_end - CLOB_CHUNK_OVERLAP_SEC
        return [{"t": t, "p": merged[t]} for t in sorted(merged)]
    params: dict = {"market": token_id, "interval": interval}
    if start_ts is not None:
        params["startTs"] = start_ts
    if end_ts is not None:
        params["endTs"] = end_ts
    return _clob_prices_history_request(token_id, params)


def _event_ts_bounds(events_rows: list[dict]) -> dict[str, tuple[int, int]]:
    """event_id -> (start_ts, end_ts)."""
    out: dict[str, tuple[int, int]] = {}
    for e in events_rows:
        eid = str(e["event_id"])
        try:
            start_dt = datetime.fromisoformat(e["start_date"].replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(e["end_date"].replace("Z", "+00:00"))
            out[eid] = (int(start_dt.timestamp()), int(end_dt.timestamp()))
        except (ValueError, TypeError, KeyError):
            continue
    return out


def token_time_ranges(markets_rows: list[dict], event_bounds: dict[str, tuple[int, int]]) -> list[tuple[str, int, int]]:
    """Per YES token: min(start) and max(end) across all markets using that token."""
    agg: dict[str, list[int]] = defaultdict(lambda: [10**18, 0])
    for r in markets_rows:
        tid = r.get("yes_token_id")
        eid = str(r.get("event_id", ""))
        if not tid or eid not in event_bounds:
            continue
        st, et = event_bounds[eid]
        lo, hi = agg[tid]
        agg[tid] = [min(lo, st), max(hi, et)]
    return [(tid, bounds[0], bounds[1]) for tid, bounds in agg.items() if bounds[1] > 0]


def fetch_and_merge_prices(
    token_ranges: list[tuple[str, int, int]],
    existing_prices: pd.DataFrame | None,
    full_refresh: bool,
    max_tokens: int,
    overlap_sec: int = 3600,
) -> pd.DataFrame:
    """
    Fetch CLOB history. If not full_refresh and we already have rows for a token,
    only request timestamps after max(existing_ts) - overlap_sec up to end_ts.
    """
    old = existing_prices if existing_prices is not None and len(existing_prices) > 0 else None
    if not token_ranges:
        return old.copy() if old is not None else pd.DataFrame(columns=["token_id", "timestamp", "price"])

    available_tokens = len(token_ranges)
    if max_tokens > 0:
        token_ranges = token_ranges[:max_tokens]
        log(
            f"CLOB: fetching {len(token_ranges)} token(s) (--max-tokens={max_tokens}, {available_tokens} available).",
            step=4,
        )
    n_tokens = len(token_ranges)
    mode = "full (--refresh)" if full_refresh else "incremental (merge with cache)"
    log(
        f"CLOB: {n_tokens} YES token(s) in this run | mode: {mode}",
        step=4,
    )

    price_rows: list[dict] = []
    n_full = 0
    n_incr = 0
    n_skip = 0
    last_logged_done = 0
    progress_stride = max(1, len(token_ranges) // 25)

    def _maybe_progress(done: int, total: int) -> None:
        nonlocal last_logged_done
        if total <= 0:
            return
        if done == total or done == 1 or done - last_logged_done >= progress_stride:
            last_logged_done = done
            log(
                f"CLOB progress {done}/{total} ({100 * done / total:.1f}%) | "
                f"tokens: full={n_full} incremental={n_incr} skipped_up_to_date={n_skip}",
                step=4,
            )

    for i, (tid, start_ts, end_ts) in enumerate(token_ranges):
        fetch_start = start_ts
        fetch_end = end_ts
        if not full_refresh and old is not None:
            sub = old[old["token_id"] == tid]
            if len(sub) > 0:
                last_t = int(sub["timestamp"].max())
                if last_t >= end_ts - 60:
                    n_skip += 1
                    _maybe_progress(i + 1, len(token_ranges))
                    continue
                fetch_start = max(start_ts, last_t - overlap_sec)
                n_incr += 1
            else:
                n_full += 1
        else:
            n_full += 1

        try:
            hist = fetch_price_history(tid, start_ts=fetch_start, end_ts=fetch_end)
            for h in hist:
                price_rows.append({"token_id": tid, "timestamp": h["t"], "price": h["p"]})
        except Exception as ex:
            log(f"warning: token {tid[:16]}… failed: {ex}", step=4)
        time.sleep(0.2)
        _maybe_progress(i + 1, len(token_ranges))

    log(
        f"CLOB done | summary: full_window={n_full} incremental_window={n_incr} skipped_current={n_skip}",
        step=4,
    )

    new_df = pd.DataFrame(price_rows) if price_rows else pd.DataFrame(columns=["token_id", "timestamp", "price"])
    if old is not None and len(new_df) > 0:
        merged = pd.concat([old, new_df], ignore_index=True)
    elif len(new_df) > 0:
        merged = new_df
    elif old is not None:
        merged = old
    else:
        return pd.DataFrame(columns=["token_id", "timestamp", "price"])

    merged = merged.drop_duplicates(subset=["token_id", "timestamp"], keep="last")
    merged = merged.sort_values(["token_id", "timestamp"]).reset_index(drop=True)
    return merged


def main(refresh: bool = False, max_tokens: int = 0, include_shorter: bool = False) -> None:
    t0 = time.perf_counter()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cache_meta_path = DATA_DIR / "last_updated.json"
    has_price_cache = (DATA_DIR / "price_history.parquet").exists()
    has_events_cache = (DATA_DIR / "events.parquet").exists()

    log(
        "start | steps: (1) Gamma → (2) filter → (3) events/markets parquet → "
        "(4) CLOB prices → (5) price parquet + last_updated.json",
    )
    existing_prices: pd.DataFrame | None = None
    if not refresh and has_price_cache:
        existing_prices = pd.read_parquet(DATA_DIR / "price_history.parquet")
        log(
            f"cache: incremental CLOB — will merge into existing price_history ({len(existing_prices):,} rows).",
        )
    elif not refresh and not has_price_cache:
        log("cache: cold start — no price_history.parquet; full CLOB window per token.")
    if refresh:
        log("flag --refresh: CLOB ignores existing price file (Gamma rebuild unchanged).")
    if not has_events_cache and not has_price_cache:
        log("cache: no events or price files — full Gamma + CLOB run.")

    events = fetch_events(verbose=True)

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
    log(
        f"filter: kept {len(filtered)} {label} Elon tweet-count events (from {len(events)} Gamma events).",
        step=2,
    )

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

    log(
        f"wrote data/events.parquet + data/markets.parquet | "
        f"{len(events_rows)} events, {len(markets_rows)} markets, {len(token_ids)} unique YES tokens.",
        step=3,
    )

    event_bounds = _event_ts_bounds(events_rows)
    token_ranges = token_time_ranges(markets_rows, event_bounds)
    if not token_ranges:
        log("warning: no token time ranges from markets; price file unchanged if present.", step=3)
    price_df = fetch_and_merge_prices(
        token_ranges,
        existing_prices=None if refresh else existing_prices,
        full_refresh=refresh,
        max_tokens=max_tokens,
    )

    if len(price_df) > 0:
        price_df.to_parquet(DATA_DIR / "price_history.parquet", index=False)
        log(
            f"wrote data/price_history.parquet | {len(price_df):,} rows after merge/dedupe.",
            step=5,
        )
    else:
        log("no price rows to write (empty merge or API limits).", step=5)

    meta = {"last_updated": datetime.now().replace(microsecond=0).isoformat() + "Z", "event_count": len(events_rows)}
    with open(cache_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.perf_counter() - t0
    log(f"finished in {elapsed:.1f}s ({elapsed / 60:.2f} min).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Gamma events/markets (always refreshed) and CLOB prices (incremental by default)."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Full re-download of price history for every token (ignore existing parquet). Events/markets still rebuilt from Gamma.",
    )
    parser.add_argument("--max-tokens", type=int, default=0, help="Max tokens to fetch price history for (0=all)")
    parser.add_argument("--include-shorter", action="store_true", help="Include 2-5 day events (more backtest range, ~30 extra events)")
    args = parser.parse_args()
    main(refresh=args.refresh, max_tokens=args.max_tokens, include_shorter=args.include_shorter)
