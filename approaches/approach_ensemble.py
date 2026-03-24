"""Ensemble: weighted combination of historical, ml, momentum."""

from approaches.base import Signal
from approaches.approach_historical import get_signal as historical_signal
from approaches.approach_ml import get_signal as ml_signal
from approaches.approach_momentum import get_signal as momentum_signal

# Weights (tuned for avg profit - can be learned)
W_HISTORICAL = 0.5
W_ML = 0.35
W_MOMENTUM = 0.15
EDGE_THRESHOLD = 0.04
VOTE_THRESHOLD = 0.5  # Buy if weighted vote > 0.5


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
    """Ensemble: weighted vote from historical, ml, momentum."""
    h = historical_signal(event_id=event_id, bracket=bracket, current_time=current_time,
                         price_history=price_history, market_data=market_data,
                         current_price=current_price, end_time=end_time, **kwargs)
    m = ml_signal(event_id=event_id, bracket=bracket, current_time=current_time,
                  price_history=price_history, market_data=market_data,
                  current_price=current_price, end_time=end_time, **kwargs)
    mom = momentum_signal(
        event_id=event_id,
        bracket=bracket,
        current_time=current_time,
        price_history=price_history,
        market_data=market_data,
        current_price=current_price,
        end_time=end_time,
        **kwargs,
    )

    market_price = h.market_price or m.market_price or 0.05
    vote = (1.0 if h.buy else 0.0) * W_HISTORICAL + (1.0 if m.buy else 0.0) * W_ML + (1.0 if mom.buy else 0.0) * W_MOMENTUM
    avg_edge = (h.edge or 0) * W_HISTORICAL + (m.edge or 0) * W_ML + (mom.edge or 0) * W_MOMENTUM
    avg_prob = (h.model_prob or market_price) * W_HISTORICAL + (m.model_prob or market_price) * W_ML + (mom.model_prob or market_price) * W_MOMENTUM

    buy = vote >= VOTE_THRESHOLD and avg_edge >= EDGE_THRESHOLD and market_price < 0.5
    confidence = min(1.0, vote)

    return Signal(
        buy=buy,
        sell=mom.sell,
        confidence=confidence,
        model_prob=avg_prob,
        market_price=market_price,
        edge=avg_edge,
        metadata={"vote": vote, "h_buy": h.buy, "m_buy": m.buy, "mom_buy": mom.buy},
    )
