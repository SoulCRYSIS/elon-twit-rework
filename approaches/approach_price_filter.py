"""Price-level filter: only buy when price is in 2-15 cent range."""

from approaches.base import Signal
from approaches.approach_historical import get_signal as historical_signal

MIN_PRICE = 0.02  # 2 cents
MAX_PRICE = 0.15  # 15 cents


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
    """Historical + price filter: avoid junk (too cheap) and low-upside (too expensive)."""
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
    market_price = base.market_price or 0.05
    if base.buy and (market_price < MIN_PRICE or market_price > MAX_PRICE):
        return Signal(
            buy=False,
            sell=base.sell,
            confidence=base.confidence,
            model_prob=base.model_prob,
            market_price=market_price,
            edge=base.edge,
            metadata={**(base.metadata or {}), "price_filtered": True},
        )
    return base
