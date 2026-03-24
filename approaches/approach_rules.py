"""
Interpretable rule baseline for YES longshots: cheap price + mild momentum + not at deadline.

Complements ML — no training, avoids overfitting to sparse labels; tune rules from correlation tables.
"""

from __future__ import annotations

from approaches.base import Signal
from approaches.ml_features import build_ml_features
from approaches.utils import parse_bracket

MIN_PRICE = 0.02
MAX_PRICE = 0.14
MIN_RET_6H = -0.03
MAX_PCT_ELAPSED = 0.92


def get_signal(
    event_id: int,
    bracket: str,
    current_time: float,
    price_history: list[tuple[float, float]],
    market_data: dict,
    current_price: float | None = None,
    end_time: float | None = None,
    start_time: float | None = None,
    **kwargs,
) -> Signal:
    market_price = current_price if current_price is not None else (
        price_history[-1][1] if price_history else 0.05
    )
    low, high = parse_bracket(bracket)
    end = end_time if end_time is not None else current_time + 7 * 24 * 3600
    start = start_time if start_time is not None else end - 7 * 24 * 3600
    feats = build_ml_features(price_history, current_time, start, end, low, high, None)
    rsi = feats["rsi_14"]
    if rsi != rsi:  # NaN → neutral
        rsi = 50.0

    buy = (
        MIN_PRICE <= market_price <= MAX_PRICE
        and feats["ret_6h"] >= MIN_RET_6H
        and feats["pct_elapsed"] <= MAX_PCT_ELAPSED
        and rsi < 62
    )
    edge = 0.05 if buy else 0.0
    return Signal(
        buy=buy,
        sell=False,
        confidence=0.4 if buy else 0.0,
        model_prob=market_price + edge,
        market_price=market_price,
        edge=edge,
        metadata=feats,
    )
