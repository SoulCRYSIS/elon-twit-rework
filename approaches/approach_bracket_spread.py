"""Adjacent-bracket spread: exploit mispricing between neighboring brackets."""

from approaches.base import Signal
from approaches.approach_historical import get_signal as historical_signal
from approaches.utils import parse_bracket

SPREAD_EDGE_THRESHOLD = 0.05  # Buy when implied prob gap is inconsistent


def get_signal(
    event_id: int,
    bracket: str,
    current_time: float,
    price_history: list[tuple[float, float]],
    market_data: dict,
    current_price: float | None = None,
    end_time: float | None = None,
    all_bracket_prices: dict[str, float] | None = None,
    all_brackets_sorted: list[tuple[str, tuple[float, float]]] | None = None,
    **kwargs,
) -> Signal:
    """
    Compare this bracket's price to neighbors. If our bracket is underpriced vs
    adjacent (sum of adjacent probs suggests we're too low), buy.
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
    if all_bracket_prices is None or all_brackets_sorted is None:
        return base

    low, high = parse_bracket(bracket)
    # Find adjacent brackets (by midpoint)
    my_mid = (low + high) / 2
    idx = -1
    for i, (br, (bl, bh)) in enumerate(all_brackets_sorted):
        if br == bracket:
            idx = i
            break
    if idx < 0:
        return base

    # Compare to prev/next
    prev_price = all_bracket_prices.get(all_brackets_sorted[idx - 1][0], 0) if idx > 0 else 0
    next_price = all_bracket_prices.get(all_brackets_sorted[idx + 1][0], 0) if idx < len(all_brackets_sorted) - 1 else 0
    my_price = all_bracket_prices.get(bracket, base.market_price or 0.05)

    # If prev and next are both high but we're low, we might be mispriced
    spread_edge = 0.0
    if prev_price > 0 and next_price > 0:
        avg_adj = (prev_price + next_price) / 2
        if my_price < avg_adj * 0.7 and base.edge and base.edge > 0:
            spread_edge = avg_adj - my_price

    buy = base.buy or (spread_edge >= SPREAD_EDGE_THRESHOLD and my_price < 0.2)
    return Signal(
        buy=buy,
        sell=base.sell,
        confidence=max(base.confidence, min(1.0, spread_edge / 0.1)),
        model_prob=base.model_prob,
        market_price=base.market_price,
        edge=(base.edge or 0) + spread_edge * 0.5,
        metadata={**(base.metadata or {}), "spread_edge": spread_edge},
    )
