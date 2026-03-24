"""Shared utilities for approaches."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_bracket(bracket_range: str) -> tuple[float, float]:
    m = re.match(r"(\d+)-(\d+)", bracket_range)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"(\d+)\+", bracket_range)
    if m:
        return float(m.group(1)), float(m.group(1)) + 50
    m = re.match(r"0-(\d+)", bracket_range)
    if m:
        return 0.0, float(m.group(1))
    return 0.0, 0.0


def get_historical_mean_std() -> tuple[float, float]:
    """Load historical mean and std of winning tweet counts."""
    events_path = DATA_DIR / "events.parquet"
    markets_path = DATA_DIR / "markets.parquet"
    if not events_path.exists() or not markets_path.exists():
        return 350.0, 80.0
    events = pd.read_parquet(events_path)
    markets = pd.read_parquet(markets_path)
    closed = events[events["closed"] == True]
    winning_counts = []
    for _, ev in closed.iterrows():
        mkt = markets[(markets["event_id"] == ev["event_id"]) & (markets["resolved_yes"] == 1)]
        if len(mkt) > 0:
            br = mkt.iloc[0]["bracket_range"]
            low, high = parse_bracket(br)
            winning_counts.append((low + high) / 2)
    if not winning_counts:
        return 350.0, 80.0
    mean = float(np.mean(winning_counts))
    std = float(np.std(winning_counts))
    return mean, max(20.0, std)
