"""Historical average approach: use resolved tweet counts to fit distribution."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from approaches.base import Signal

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EDGE_THRESHOLD = 0.05  # Buy when model prob exceeds market by 5%


def _parse_bracket(bracket_range: str) -> tuple[float, float]:
    """Parse bracket to (low, high) for midpoint."""
    m = re.match(r"(\d+)-(\d+)", bracket_range)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"(\d+)\+", bracket_range)
    if m:
        return float(m.group(1)), float(m.group(1)) + 50  # arbitrary upper
    m = re.match(r"0-(\d+)", bracket_range)
    if m:
        return 0.0, float(m.group(1))
    return 0.0, 0.0


def _load_historical_stats() -> tuple[float, float, dict]:
    """Load events + markets, compute mean/std of winning counts, return (mean, std, bracket_probs)."""
    events_path = DATA_DIR / "events.parquet"
    markets_path = DATA_DIR / "markets.parquet"
    if not events_path.exists() or not markets_path.exists():
        return 350.0, 80.0, {}  # fallback defaults

    events = pd.read_parquet(events_path)
    markets = pd.read_parquet(markets_path)

    # Only closed events
    closed = events[events["closed"] == True]
    winning_counts = []

    for _, ev in closed.iterrows():
        eid = ev["event_id"]
        mkt = markets[(markets["event_id"] == eid) & (markets["resolved_yes"] == 1)]
        if len(mkt) == 0:
            continue
        row = mkt.iloc[0]
        br = row["bracket_range"]
        low, high = _parse_bracket(br)
        mid = (low + high) / 2
        winning_counts.append(mid)

    if not winning_counts:
        return 350.0, 80.0, {}

    mean = float(np.mean(winning_counts))
    std = float(np.std(winning_counts))
    if std < 20:
        std = 20.0

    # Precompute P(bracket) for each bracket in markets
    bracket_probs = {}
    all_brackets = markets["bracket_range"].unique()
    for br in all_brackets:
        low, high = _parse_bracket(br)
        mid = (low + high) / 2
        # Normal approx: P(low <= X <= high)
        from scipy import stats

        z_low = (low - mean) / std if std > 0 else 0
        z_high = (high - mean) / std if std > 0 else 0
        p = float(stats.norm.cdf(z_high) - stats.norm.cdf(z_low))
        p = max(0.001, min(0.99, p))
        bracket_probs[br] = p

    return mean, std, bracket_probs


# Module-level cache
_stats_cache: tuple[float, float, dict] | None = None


def get_signal(
    event_id: int,
    bracket: str,
    current_time: float,
    price_history: list[tuple[float, float]],
    market_data: dict,
    current_price: float | None = None,
    current_count: float | None = None,
    end_time: float | None = None,
    **kwargs,
) -> Signal:
    """
    Get trading signal from historical average model.
    price_history: list of (timestamp, price)
    """
    global _stats_cache
    if _stats_cache is None:
        _stats_cache = _load_historical_stats()
    mean, std, bracket_probs = _stats_cache

    model_prob = bracket_probs.get(bracket, 0.05)
    market_price = current_price if current_price is not None else (price_history[-1][1] if price_history else 0.05)
    edge = model_prob - market_price

    buy = edge >= EDGE_THRESHOLD and market_price < 0.5
    confidence = min(1.0, abs(edge) / 0.2) if edge > 0 else 0.0

    return Signal(
        buy=buy,
        sell=False,  # Historical doesn't suggest early sell
        confidence=confidence,
        model_prob=model_prob,
        market_price=market_price,
        edge=edge,
        metadata={"mean": mean, "std": std},
    )
