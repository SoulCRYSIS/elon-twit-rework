"""Price momentum approach: buy when momentum positive and price below threshold."""

from approaches.base import Signal

EDGE_THRESHOLD = 0.03
MOMENTUM_6H_THRESHOLD = 0.02  # Price up 2% in 6h
MOMENTUM_24H_THRESHOLD = 0.05  # Price up 5% in 24h
MAX_BUY_PRICE = 0.35  # Don't buy above 35 cents
SELL_MOMENTUM_REVERSE = -0.03  # Sell if momentum turns negative


def _get_momentum(price_history: list[tuple[float, float]], current_time: float, hours: float) -> float:
    """Price change over last N hours."""
    if not price_history or len(price_history) < 2:
        return 0.0
    cutoff = current_time - hours * 3600
    past = [p for t, p in price_history if t <= cutoff]
    if not past:
        return 0.0
    old_price = past[-1]
    current = price_history[-1][1] if price_history else 0.0
    if old_price < 0.001:
        return 0.0
    return (current - old_price) / old_price


def get_signal(
    event_id: int,
    bracket: str,
    current_time: float,
    price_history: list[tuple[float, float]],
    market_data: dict,
    current_price: float | None = None,
    current_count: float | None = None,
    end_time: float | None = None,
    position: dict | None = None,
    **kwargs,
) -> Signal:
    """
    Buy when momentum positive and price low; sell when momentum reverses.
    """
    market_price = current_price if current_price is not None else (price_history[-1][1] if price_history else 0.05)

    mom_6h = _get_momentum(price_history, current_time, 6)
    mom_24h = _get_momentum(price_history, current_time, 24)

    buy = (
        market_price <= MAX_BUY_PRICE
        and (mom_6h >= MOMENTUM_6H_THRESHOLD or mom_24h >= MOMENTUM_24H_THRESHOLD)
        and market_price >= 0.01
    )
    # Sell only when actually long (buy and sell are paired)
    sell = (
        position is not None
        and mom_6h <= SELL_MOMENTUM_REVERSE
        and market_price > 0.05
    )

    confidence = min(1.0, (mom_6h + mom_24h) / 0.2) if buy else 0.0

    return Signal(
        buy=buy,
        sell=sell,
        confidence=confidence,
        model_prob=None,
        market_price=market_price,
        edge=mom_6h,
        metadata={"momentum_6h": mom_6h, "momentum_24h": mom_24h},
    )
