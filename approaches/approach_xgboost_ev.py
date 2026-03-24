"""
Experimental XGBoost YES model: EV-weighted training + optional PnL threshold tuning.

Train:
  python scripts/train_ml_model.py --ev-weighted --tune-on pnl
Writes: data/ml_artifacts/xgb_model_ev.joblib, meta_ev.json

Backtest / bot: --approach xgboost_ev
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from approaches.base import Signal
from approaches.ml_features import build_ml_features
from approaches.utils import parse_bracket

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "data" / "ml_artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgb_model_ev.joblib"
META_PATH = ARTIFACT_DIR / "meta_ev.json"

_bundle: dict | None = None


def _load_bundle():
    global _bundle
    if _bundle is not None:
        return _bundle
    if not MODEL_PATH.exists() or not META_PATH.exists():
        _bundle = {"missing": True}
        return _bundle
    with open(META_PATH) as f:
        meta = json.load(f)
    model = joblib.load(MODEL_PATH)
    scaler = StandardScaler()
    scaler.mean_ = np.array(meta["scaler_mean"], dtype=float)
    scaler.scale_ = np.array(meta["scaler_scale"], dtype=float)
    scaler.n_features_in_ = len(meta["feature_names"])
    scaler.feature_names_in_ = np.array(meta["feature_names"], dtype=object)
    _bundle = {"model": model, "scaler": scaler, "meta": meta, "missing": False}
    return _bundle


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
    **kwargs,
) -> Signal:
    market_price = current_price if current_price is not None else (
        price_history[-1][1] if price_history else 0.05
    )
    low, high = parse_bracket(bracket)
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
            metadata={"error": "Run: python scripts/train_ml_model.py --ev-weighted --tune-on pnl"},
        )

    meta = b["meta"]
    hmid = float(meta.get("hist_winning_mid", 350.0))
    feats = build_ml_features(price_history, current_time, start, end, low, high, hmid)
    names: list[str] = meta["feature_names"]
    row = pd.DataFrame([[feats[n] for n in names]], columns=names)
    X = b["scaler"].transform(row)
    prob = float(b["model"].predict_proba(X)[0][1])
    edge = prob - market_price
    edge_thr = float(meta.get("edge_threshold", 0.08))
    max_px = float(meta.get("max_market_price_buy", 0.5))
    buy = edge >= edge_thr and market_price < max_px
    confidence = min(1.0, max(0.0, edge / 0.2)) if edge > 0 else 0.0
    return Signal(
        buy=buy,
        sell=False,
        confidence=confidence,
        model_prob=prob,
        market_price=market_price,
        edge=edge,
        metadata=feats,
    )


def clear_cache():
    global _bundle
    _bundle = None
