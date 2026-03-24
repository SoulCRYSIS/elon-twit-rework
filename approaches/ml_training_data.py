"""Build training matrices for tabular YES models (7-day events only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from approaches.ml_features import ALL_FEATURE_KEYS, build_ml_features
from approaches.utils import parse_bracket

SEVEN_DAY_MIN_DURATION_DAYS = 6


def filter_seven_day_events(events: pd.DataFrame) -> pd.DataFrame:
    ev = events[events["closed"] == True].copy()
    if "duration_days" in ev.columns:
        ev = ev[ev["duration_days"] >= SEVEN_DAY_MIN_DURATION_DAYS]
    return ev


def hist_winning_mid(events: pd.DataFrame, markets: pd.DataFrame) -> float:
    mids = []
    for _, row in events.iterrows():
        mkt = markets[(markets["event_id"] == row["event_id"]) & (markets["resolved_yes"] == 1)]
        if len(mkt) == 0:
            continue
        br = mkt.iloc[0]["bracket_range"]
        lo, hi = parse_bracket(br)
        mids.append((lo + hi) / 2)
    return float(np.mean(mids)) if mids else 350.0


def build_training_dataframe(
    events: pd.DataFrame,
    markets: pd.DataFrame,
    prices: pd.DataFrame,
    sample_fracs: np.ndarray,
    hmid: float,
) -> pd.DataFrame:
    rows = []
    for _, ev in events.iterrows():
        eid = ev["event_id"]
        start = pd.to_datetime(ev["start_date"]).timestamp()
        end = pd.to_datetime(ev["end_date"]).timestamp()
        end_iso = ev["end_date"]
        mkts = markets[markets["event_id"] == eid]
        for _, m in mkts.iterrows():
            tid = m["yes_token_id"]
            br = m["bracket_range"]
            low, high = parse_bracket(br)
            y = int(m["resolved_yes"])
            ph = prices[prices["token_id"] == tid]
            if len(ph) < 3:
                continue
            for frac in sample_fracs:
                t = start + float(frac) * (end - start)
                before = ph[ph["timestamp"] <= t]
                if len(before) < 2:
                    continue
                hist = [(r["timestamp"], r["price"]) for _, r in before.iterrows()]
                feats = build_ml_features(hist, t, start, end, low, high, hmid)
                rows.append(
                    {
                        "event_id": eid,
                        "end_date": end_iso,
                        "end_ts": end,
                        "resolved_yes": y,
                        **feats,
                    }
                )
    return pd.DataFrame(rows)


def temporal_event_ids(df: pd.DataFrame, train_frac: float, val_frac: float):
    ev_meta = df.groupby("event_id").agg(end_ts=("end_ts", "max")).reset_index()
    ev_meta = ev_meta.sort_values("end_ts")
    n = len(ev_meta)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    ids = ev_meta["event_id"].tolist()
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val :])
    return train_ids, val_ids, test_ids
