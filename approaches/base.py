"""Common interface for prediction approaches."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Signal:
    """Trading signal from a prediction approach."""

    buy: bool
    sell: bool
    confidence: float  # 0.0 to 1.0
    model_prob: float | None = None  # Model's probability bracket wins
    market_price: float | None = None
    edge: float | None = None  # model_prob - market_price
    kelly_fraction: float | None = None  # Suggested position size as fraction of bankroll (0-1)
    metadata: dict[str, Any] | None = None
