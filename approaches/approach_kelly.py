"""Kelly criterion: position sizing based on edge and odds."""

from approaches.base import Signal
from approaches.approach_historical import get_signal as historical_signal

KELLY_FRACTION = 0.25  # Quarter-Kelly for safety
MIN_EDGE = 0.03


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
    Historical + Kelly sizing. kelly_fraction = (bp - q) / b where b = odds, p = win prob, q = 1-p.
    For binary: b = (1 - price) / price (odds to win $1 per $1 bet). kelly = p - q/b = p - (1-p)*price/(1-price).
    Simplified: kelly = edge / (1 - price) when price is cost to buy YES.
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
    market_price = base.market_price or 0.05
    if not base.buy or base.edge is None or base.edge < MIN_EDGE:
        return base
    # Kelly: f = (p*b - q) / b = p - q/b. For YES at price: b = (1-price)/price. So f = p - (1-p)*price/(1-price)
    p = base.model_prob or market_price
    if market_price >= 0.99:
        kelly_frac = 0.0
    else:
        b = (1 - market_price) / market_price  # odds
        kelly_frac = (p * b - (1 - p)) / b if b > 0 else 0
        kelly_frac = max(0, min(0.5, kelly_frac * KELLY_FRACTION))  # cap at 50%
    return Signal(
        buy=base.buy,
        sell=base.sell,
        confidence=base.confidence,
        model_prob=base.model_prob,
        market_price=market_price,
        edge=base.edge,
        kelly_fraction=kelly_frac,
        metadata={**(base.metadata or {}), "kelly": kelly_frac},
    )
