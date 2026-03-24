"""Cross-sectional ranking: only buy top N brackets by edge per event."""

from approaches.base import Signal
from approaches.approach_historical import get_signal as historical_signal
from approaches.utils import parse_bracket

TOP_N = 2  # Only buy top 2 brackets by edge
MIN_EDGE = 0.03


def get_signal(
    event_id: int,
    bracket: str,
    current_time: float,
    price_history: list[tuple[float, float]],
    market_data: dict,
    current_price: float | None = None,
    end_time: float | None = None,
    all_bracket_edges: dict[str, float] | None = None,
    **kwargs,
) -> Signal:
    """
    Use historical model but only buy if this bracket is in top N by edge.
    all_bracket_edges: {bracket: edge} for all brackets in event (caller must provide).
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
    if not base.buy:
        return base
    if all_bracket_edges is None:
        return base
    # Rank: only buy if in top N
    sorted_brackets = sorted(all_bracket_edges.items(), key=lambda x: -x[1])
    top_brackets = {b for b, _ in sorted_brackets[:TOP_N]}
    if bracket not in top_brackets or base.edge is None or base.edge < MIN_EDGE:
        return Signal(
            buy=False,
            sell=base.sell,
            confidence=base.confidence,
            model_prob=base.model_prob,
            market_price=base.market_price,
            edge=base.edge,
            metadata={**(base.metadata or {}), "rank_filtered": True},
        )
    return base
