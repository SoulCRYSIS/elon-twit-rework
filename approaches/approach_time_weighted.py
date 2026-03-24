"""Time-to-close weighting: stronger signals closer to resolution."""

from approaches.base import Signal
from approaches.approach_historical import get_signal as historical_signal

# Weight by inverse of time to close - more weight when closer
# Buy only when we have enough "effective" edge after time weighting
MIN_HOURS_TO_CLOSE = 24  # Don't buy in last 24h (too late)
MAX_HOURS_FOR_FULL_WEIGHT = 72  # Full weight when 72+ hours left


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
    """
    Historical + time weighting. Discount edge when far from close.
    Full confidence when < 72h to close; reduced when > 72h.
    """
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
    if end_time is None:
        return base
    hours_to_close = max(0, (end_time - current_time) / 3600)
    if hours_to_close < MIN_HOURS_TO_CLOSE:
        return Signal(buy=False, sell=base.sell, confidence=0, model_prob=base.model_prob,
                     market_price=base.market_price, edge=base.edge,
                     metadata={**(base.metadata or {}), "too_late": True})
    # Time weight: 1.0 when hours_to_close <= 72, else 0.5 + 0.5 * 72/hours
    if hours_to_close <= MAX_HOURS_FOR_FULL_WEIGHT:
        time_weight = 1.0
    else:
        time_weight = 0.5 + 0.5 * (MAX_HOURS_FOR_FULL_WEIGHT / hours_to_close)
    effective_edge = (base.edge or 0) * time_weight
    buy = base.buy and effective_edge >= 0.04  # Slightly stricter
    return Signal(
        buy=buy,
        sell=base.sell,
        confidence=base.confidence * time_weight,
        model_prob=base.model_prob,
        market_price=base.market_price,
        edge=effective_edge,
        metadata={**(base.metadata or {}), "time_weight": time_weight, "hours_to_close": hours_to_close},
    )
