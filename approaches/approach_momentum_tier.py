"""Momentum entry signals; exits use buy-price-tiered TP/SL (not momentum flip)."""

from approaches.approach_momentum import get_signal as _mom
from approaches.base import Signal
from approaches.exit_policy import exit_signal_tiered


def get_signal(
    event_id: int,
    bracket: str,
    current_time: float,
    price_history: list[tuple[float, float]],
    market_data: dict,
    current_price: float | None = None,
    end_time: float | None = None,
    position: dict | None = None,
    **kwargs,
) -> Signal:
    base = _mom(
        event_id=event_id,
        bracket=bracket,
        current_time=current_time,
        price_history=price_history,
        market_data=market_data,
        current_price=current_price,
        end_time=end_time,
        position=position,
        **kwargs,
    )
    mp = base.market_price if base.market_price is not None else (current_price or 0.05)
    et = end_time if end_time is not None else current_time + 7 * 24 * 3600
    sell = exit_signal_tiered(position, float(mp), float(current_time), float(et))
    return Signal(
        buy=base.buy,
        sell=sell,
        confidence=base.confidence,
        model_prob=base.model_prob,
        market_price=base.market_price,
        edge=base.edge,
        metadata={**(base.metadata or {}), "exit": "tiered"},
    )
