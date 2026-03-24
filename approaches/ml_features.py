"""
Feature engineering for YES-bracket models (price path + event timeline).

Uses actual event start/end (no fixed 7d assumption for elapsed time).
Technical indicators are computed on the time-ordered price series up to current_time.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from approaches.utils import parse_bracket


def _price_at_or_before(history: list[tuple[float, float]], ts: float) -> float | None:
    """Last price at timestamp <= ts."""
    best = None
    best_t = -1.0
    for t, p in history:
        if t <= ts and t >= best_t:
            best_t = t
            best = p
    return best


def _ema_series(prices: np.ndarray, span: int) -> np.ndarray:
    """EMA aligned with prices (same length); first values use expanding mean."""
    if len(prices) == 0:
        return prices
    alpha = 2.0 / (span + 1)
    out = np.empty_like(prices, dtype=float)
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
    return out


def rsi_from_prices(prices: np.ndarray, period: int = 14) -> float:
    """Wilder's RSI on closing prices (last value). Neutral 50 if insufficient data."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.clip(deltas, 0, None)
    losses = np.clip(-deltas, 0, None)
    # Wilder smoothing
    avg_g = gains[-period:].mean()
    avg_l = losses[-period:].mean()
    if avg_l < 1e-12:
        return 100.0 if avg_g > 0 else 50.0
    rs = avg_g / avg_l
    return float(100.0 - (100.0 / (1.0 + rs)))


def build_ml_features(
    price_history: list[tuple[float, float]],
    current_time: float,
    start_time: float,
    end_time: float,
    bracket_low: float,
    bracket_high: float,
    hist_winning_mid: float | None = None,
) -> dict[str, float]:
    """
    Point-in-time features (only info available at current_time).

    hist_winning_mid: mean of (low+high)/2 of winning brackets across past resolved events;
    optional context feature (same for all brackets at a time — use with care).
    """
    span_sec = max(end_time - start_time, 3600.0)
    pct_elapsed = float(np.clip((current_time - start_time) / span_sec, 0.0, 1.0))
    hours_elapsed = max(0.0, (current_time - start_time) / 3600.0)
    hours_to_close = max(0.0, (end_time - current_time) / 3600.0)

    hist = [(t, p) for t, p in price_history if t <= current_time]
    hist.sort(key=lambda x: x[0])
    prices_arr = np.array([p for _, p in hist], dtype=float)
    n_ticks = len(prices_arr)
    current_price = float(prices_arr[-1]) if n_ticks else 0.05

    # Returns vs reference times (true lookback)
    t_6h = current_time - 6 * 3600
    t_24h = current_time - 24 * 3600
    p_6h = _price_at_or_before(hist, t_6h)
    p_24h = _price_at_or_before(hist, t_24h)
    ret_6h = (current_price - p_6h) / max(p_6h, 1e-6) if p_6h is not None else 0.0
    ret_24h = (current_price - p_24h) / max(p_24h, 1e-6) if p_24h is not None else 0.0

    vol_ticks = float(np.std(prices_arr)) if n_ticks > 2 else 0.0
    # Log return over full observed path
    log_ret_path = (
        float(math.log((current_price + 1e-6) / (prices_arr[0] + 1e-6))) if n_ticks > 1 else 0.0
    )

    rsi_14 = rsi_from_prices(prices_arr, 14)
    if n_ticks >= 2:
        ema12 = _ema_series(prices_arr, 12)
        ema26 = _ema_series(prices_arr, 26)
        ema_ratio = float(ema12[-1] / max(ema26[-1], 1e-6))
        macd = float(ema12[-1] - ema26[-1])
    else:
        ema_ratio = 1.0
        macd = 0.0

    bracket_mid = (bracket_low + bracket_high) / 2.0
    # Normalize bracket level vs historical winning center (scale ~ tweet counts)
    hmid = hist_winning_mid if hist_winning_mid is not None else 350.0
    bracket_z = float((bracket_mid - hmid) / max(80.0, abs(hmid) * 0.25))

    hist_context = 0.0
    if hist_winning_mid is not None and hmid > 0:
        hist_context = float(math.tanh((bracket_mid - hmid) / max(50.0, hmid * 0.15)))

    return {
        "pct_elapsed": pct_elapsed,
        "hours_elapsed": hours_elapsed,
        "log1p_hours_elapsed": float(math.log1p(hours_elapsed)),
        "hours_to_close": hours_to_close,
        "log1p_hours_to_close": float(math.log1p(hours_to_close)),
        "bracket_low": float(bracket_low),
        "bracket_high": float(bracket_high),
        "bracket_mid": float(bracket_mid),
        "bracket_z_vs_hist": bracket_z,
        "hist_win_context": hist_context,
        "current_price": float(current_price),
        "volatility_ticks": vol_ticks,
        "n_ticks_log1p": float(math.log1p(n_ticks)),
        "ret_6h": float(ret_6h),
        "ret_24h": float(ret_24h),
        "log_ret_path": float(log_ret_path),
        "rsi_14": float(rsi_14),
        "ema_ratio_12_26": float(ema_ratio),
        "macd": float(macd),
    }


def build_features_for_bracket(
    price_history: list[tuple[float, float]],
    current_time: float,
    start_time: float,
    end_time: float,
    bracket: str,
    hist_winning_mid: float | None = None,
) -> dict[str, float]:
    low, high = parse_bracket(bracket)
    return build_ml_features(
        price_history, current_time, start_time, end_time, low, high, hist_winning_mid
    )


ALL_FEATURE_KEYS: Sequence[str] = (
    "pct_elapsed",
    "hours_elapsed",
    "log1p_hours_elapsed",
    "hours_to_close",
    "log1p_hours_to_close",
    "bracket_low",
    "bracket_high",
    "bracket_mid",
    "bracket_z_vs_hist",
    "hist_win_context",
    "current_price",
    "volatility_ticks",
    "n_ticks_log1p",
    "ret_6h",
    "ret_24h",
    "log_ret_path",
    "rsi_14",
    "ema_ratio_12_26",
    "macd",
)
