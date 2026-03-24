"""Regime detection: calm vs volatile periods, adjust thresholds."""

from pathlib import Path

import numpy as np
import pandas as pd

from approaches.base import Signal
from approaches.approach_historical import get_signal as historical_signal
from approaches.utils import get_historical_mean_std, parse_bracket

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Calm: std of winning counts low. Volatile: high.
REGIME_CACHE = None


def _get_regime() -> str:
    """Classify regime from historical std of winning counts."""
    global REGIME_CACHE
    if REGIME_CACHE is not None:
        return REGIME_CACHE
    mean, std = get_historical_mean_std()
    # Calm if std < 50, volatile if std > 90
    if std < 50:
        REGIME_CACHE = "calm"
    elif std > 90:
        REGIME_CACHE = "volatile"
    else:
        REGIME_CACHE = "normal"
    return REGIME_CACHE


def get_signal(
    event_id: int,
    bracket: str,
    current_time: float,
    price_history: list[tuple[float, float]],
    market_data: dict,
    current_price: float | None = None,
    end_time: float | None = None,
    **kwargs,
) -> Signal:
    """Historical + regime: stricter in volatile, looser in calm."""
    base = historical_signal(
        event_id=event_id,
        bracket=bracket,
        current_time=current_time,
        price_history=price_history,
        market_data=market_data,
        current_price=current_price,
        end_time=end_time,
        **kwargs,
    )
    regime = _get_regime()
    if regime == "volatile":
        edge_thresh = 0.05  # Slightly stricter
    elif regime == "calm":
        edge_thresh = 0.04  # Looser
    else:
        edge_thresh = 0.045
    if base.buy and (base.edge is None or base.edge < edge_thresh):
        return Signal(
            buy=False,
            sell=base.sell,
            confidence=base.confidence,
            model_prob=base.model_prob,
            market_price=base.market_price,
            edge=base.edge,
            metadata={**(base.metadata or {}), "regime": regime, "edge_thresh": edge_thresh},
        )
    return Signal(
        buy=base.buy,
        sell=base.sell,
        confidence=base.confidence,
        model_prob=base.model_prob,
        market_price=base.market_price,
        edge=base.edge,
        metadata={**(base.metadata or {}), "regime": regime},
    )
