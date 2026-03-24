"""
Exit rules coupled to entry: TP/SL as return from avg buy_price; tiers by buy_price.

Optional: data/ml_artifacts/exit_tiers.json
{"tiers": [[buy_max, tp_return, sl_return], ...], "time_stop_hours": 10, "min_profit_time_stop": 0.03}
"""

from __future__ import annotations

import json
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parent.parent / "data" / "ml_artifacts" / "exit_tiers.json"

_DEFAULT_TIERS: list[tuple[float, float, float]] = [
    (0.04, 3.0, 0.40),
    (0.08, 1.8, 0.42),
    (0.12, 1.2, 0.45),
    (0.20, 0.75, 0.48),
    (0.35, 0.50, 0.50),
    (1.0, 0.35, 0.52),
]

_policy_cache: tuple[list[tuple[float, float, float]], float, float] | None = None


def clear_exit_policy_cache():
    global _policy_cache
    _policy_cache = None


def _policy() -> tuple[list[tuple[float, float, float]], float, float]:
    global _policy_cache
    if _policy_cache is not None:
        return _policy_cache
    if not ARTIFACT.exists():
        _policy_cache = (list(_DEFAULT_TIERS), 10.0, 0.03)
        return _policy_cache
    with open(ARTIFACT) as f:
        raw = json.load(f)
    tiers = [tuple(float(x) for x in row) for row in raw.get("tiers", _DEFAULT_TIERS)]
    th = float(raw.get("time_stop_hours", 10))
    mp = float(raw.get("min_profit_time_stop", 0.03))
    _policy_cache = (tiers, th, mp)
    return _policy_cache


def tp_sl_for_buy_price(buy_price: float) -> tuple[float, float]:
    tiers, _, _ = _policy()
    for hi, tp, sl in sorted(tiers, key=lambda x: x[0]):
        if buy_price < hi:
            return tp, sl
    return tiers[-1][1], tiers[-1][2]


def exit_signal_tiered(
    position: dict | None,
    market_price: float,
    current_time: float,
    end_time: float,
) -> bool:
    """True = sell long. Caller must only act when position is open."""
    if not position:
        return False
    bp = float(position["buy_price"])
    if bp < 1e-6:
        return False
    ret = (market_price - bp) / bp
    tp, sl = tp_sl_for_buy_price(bp)
    if ret >= tp or ret <= -sl:
        return True
    _, time_stop_h, min_profit = _policy()
    hours_left = max(0.0, (end_time - current_time) / 3600.0)
    if hours_left < time_stop_h and ret >= min_profit:
        return True
    return False
