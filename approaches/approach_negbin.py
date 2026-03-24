"""Negative Binomial approach (TweetCast-style): market-implied rate + NegBin uncertainty."""

import re
from pathlib import Path

import numpy as np

from approaches.base import Signal

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EDGE_THRESHOLD = 0.05


def _parse_bracket(bracket_range: str) -> tuple[float, float]:
    m = re.match(r"(\d+)-(\d+)", bracket_range)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"(\d+)\+", bracket_range)
    if m:
        return float(m.group(1)), float(m.group(1)) + 100
    m = re.match(r"0-(\d+)", bracket_range)
    if m:
        return 0.0, float(m.group(1))
    return 0.0, 0.0


def _negbin_cdf(k: float, r: float, p: float) -> float:
    """P(X <= k) for NegBin. Simplified via scipy."""
    from scipy.stats import nbinom

    k = max(0, int(k))
    return float(nbinom.cdf(k, r, p))


def _p_bracket_negbin(low: float, high: float, lam: float, r: float = 5.0) -> float:
    """P(low <= X <= high) for NegBin with mean lam, dispersion r."""
    from scipy.stats import nbinom

    p = r / (r + lam) if (r + lam) > 0 else 0.5
    low_i = max(0, int(low))
    high_i = max(low_i, int(high))
    cdf_high = nbinom.cdf(high_i, r, p)
    cdf_low = nbinom.cdf(low_i - 1, r, p) if low_i > 0 else 0
    return max(0.001, min(0.99, float(cdf_high - cdf_low)))


def _market_implied_mean(market_prices: dict[str, float], brackets: list[tuple[str, tuple[float, float]]]) -> float:
    """Compute market-implied mean count from bracket prices (weighted by midpoint)."""
    total_prob = 0.0
    weighted_sum = 0.0
    for br, (low, high) in brackets:
        p = market_prices.get(br, 0.02)
        mid = (low + high) / 2
        total_prob += p
        weighted_sum += p * mid
    if total_prob < 0.01:
        return 350.0
    return weighted_sum / total_prob


def get_signal(
    event_id: int,
    bracket: str,
    current_time: float,
    price_history: list[tuple[float, float]],
    market_data: dict,
    current_price: float | None = None,
    current_count: float | None = None,
    end_time: float | None = None,
    all_bracket_prices: dict[str, float] | None = None,
    all_brackets: list[tuple[str, tuple[float, float]]] | None = None,
    **kwargs,
) -> Signal:
    """
    NegBin: rate = (market_mean - current_count) / hours_left, R ~ NegBin(λ, r).
    """
    market_price = current_price if current_price is not None else (price_history[-1][1] if price_history else 0.05)

    # Need end_time and current_count for live; for backtest we approximate
    if end_time is None or current_count is None:
        # Fallback: use market price as proxy, no edge
        return Signal(
            buy=False,
            sell=False,
            confidence=0.0,
            model_prob=market_price,
            market_price=market_price,
            edge=0.0,
        )

    hours_left = max(0.1, (end_time - current_time) / 3600)
    market_mean = 350.0
    if all_bracket_prices and all_brackets:
        market_mean = _market_implied_mean(all_bracket_prices, all_brackets)

    rate = (market_mean - current_count) / hours_left if hours_left > 0 else 0
    rate = max(0, min(100, rate))
    lam = rate * hours_left
    r = 5.0  # dispersion

    low, high = _parse_bracket(bracket)
    model_prob = _p_bracket_negbin(low, high, lam, r)
    edge = model_prob - market_price

    buy = edge >= EDGE_THRESHOLD and market_price < 0.5
    confidence = min(1.0, abs(edge) / 0.2) if edge > 0 else 0.0

    return Signal(
        buy=buy,
        sell=False,
        confidence=confidence,
        model_prob=model_prob,
        market_price=market_price,
        edge=edge,
        metadata={"rate": rate, "lam": lam, "market_mean": market_mean},
    )
