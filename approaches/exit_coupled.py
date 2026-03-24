"""Compose an entry-only signal with tiered exits keyed off avg buy_price (requires `position` kwarg)."""

from __future__ import annotations

from collections.abc import Callable

from approaches.base import Signal
from approaches.exit_policy import exit_signal_tiered

SignalFn = Callable[..., Signal]


def wrap_tiered_exit(entry_get_signal: SignalFn) -> SignalFn:
    """
    `entry_get_signal` should implement buy logic (sell may be ignored).
    Sell is True only when `position` is not None and tiered TP/SL/time stop fires.
    """

    def get_signal(
        position: dict | None = None,
        current_time: float = 0.0,
        end_time: float | None = None,
        current_price: float | None = None,
        **kwargs,
    ) -> Signal:
        base = entry_get_signal(
            position=position,
            current_time=current_time,
            end_time=end_time,
            current_price=current_price,
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
            kelly_fraction=base.kelly_fraction,
            metadata={
                **(base.metadata or {}),
                "tier_exit": sell,
                "has_position": position is not None,
            },
        )

    return get_signal
