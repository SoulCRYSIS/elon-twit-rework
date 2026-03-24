"""
Price-agnostic bracket choice: each step, buy only the bracket with highest model P(YES).

Uses the same artifacts as ``approach_xgboost`` (``xgb_model.joblib`` + ``meta.json``).
Entry ignores edge vs market and max buy price; backtest still settles at market prices and reports PnL.

Train: ``python scripts/train_ml_model.py``
Backtest / bot: ``--approach xgboost_pick``

Optional in ``meta.json``: ``winner_min_prob`` (float, default 0) — require max P(YES) >= this to buy.
"""

from __future__ import annotations

import pandas as pd

from approaches.approach_xgboost import _load_bundle
from approaches.base import Signal
from approaches.ml_features import build_ml_features
from approaches.utils import parse_bracket


def _model_prob_for_row(
    bundle: dict,
    meta: dict,
    bracket: str,
    price_history: list[tuple[float, float]],
    current_time: float,
    start: float,
    end: float,
) -> float:
    low, high = parse_bracket(bracket)
    hmid = float(meta.get("hist_winning_mid", 350.0))
    feats = build_ml_features(price_history, current_time, start, end, low, high, hmid)
    names: list[str] = meta["feature_names"]
    row = pd.DataFrame([[feats[n] for n in names]], columns=names)
    X = bundle["scaler"].transform(row)
    return float(bundle["model"].predict_proba(X)[0][1])


def build_pick_context(
    rows: list[dict],
    start_time: float,
    end_time: float,
    current_time: float,
) -> dict | None:
    """
    ``rows``: dicts with ``bracket``, ``price_history`` (and optional ``market_data``).

    Returns ``{"probs": {bracket: p}, "winner": bracket, "winner_p": float}`` or None if no model/rows.
    """
    b = _load_bundle()
    if b.get("missing") or not rows:
        return None
    meta = b["meta"]
    scored: list[tuple[str, float]] = []
    for r in rows:
        br = r["bracket"]
        ph = r.get("price_history") or []
        p = _model_prob_for_row(b, meta, br, ph, current_time, start_time, end_time)
        scored.append((br, p))
    winner_bracket, winner_p = max(scored, key=lambda x: (x[1], x[0]))
    return {"probs": dict(scored), "winner": winner_bracket, "winner_p": winner_p}


def get_signal(
    event_id: int,
    bracket: str,
    current_time: float,
    price_history: list[tuple[float, float]],
    market_data: dict,
    current_price: float | None = None,
    end_time: float | None = None,
    start_time: float | None = None,
    position: dict | None = None,
    pick_context: dict | None = None,
    event_market_candidates: list[dict] | None = None,
    **kwargs,
) -> Signal:
    market_price = (
        current_price
        if current_price is not None
        else (price_history[-1][1] if price_history else 0.05)
    )
    end = end_time if end_time is not None else current_time + 7 * 24 * 3600
    if start_time is not None:
        start = start_time
    else:
        start = end - 7 * 24 * 3600

    b = _load_bundle()
    if b.get("missing"):
        return Signal(
            buy=False,
            sell=False,
            confidence=0.0,
            model_prob=market_price,
            market_price=market_price,
            edge=0.0,
            metadata={"error": "Run scripts/train_ml_model.py after fetch_data"},
        )

    meta = b["meta"]
    min_w = float(meta.get("winner_min_prob", 0.0))

    ctx = pick_context
    if ctx is None and event_market_candidates:
        ctx = build_pick_context(event_market_candidates, start, end, current_time)

    if ctx is None:
        prob = _model_prob_for_row(b, meta, bracket, price_history, current_time, start, end)
        return Signal(
            buy=False,
            sell=False,
            confidence=0.0,
            model_prob=prob,
            market_price=market_price,
            edge=prob - market_price,
            metadata={"error": "pass pick_context or event_market_candidates for xgboost_pick"},
        )

    probs: dict[str, float] = ctx["probs"]
    prob = float(probs.get(bracket, 0.0))
    winner = ctx["winner"]
    winner_p = float(ctx["winner_p"])
    buy = bracket == winner and winner_p >= min_w
    edge = prob - market_price
    confidence = min(1.0, max(0.0, winner_p)) if buy else 0.0
    return Signal(
        buy=buy,
        sell=False,
        confidence=confidence,
        model_prob=prob,
        market_price=market_price,
        edge=edge,
        metadata={"pick_winner": winner, "pick_winner_p": winner_p},
    )


def clear_cache():
    from approaches.approach_xgboost import clear_cache as _xgb_clear

    _xgb_clear()
