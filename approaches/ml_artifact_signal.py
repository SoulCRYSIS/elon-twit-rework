"""Build get_signal from arbitrary model.joblib + meta.json (for experiment grids)."""

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


def make_get_signal(model_path: Path, meta_path: Path):
    """Return get_signal(...) identical to approach_xgboost but bound to paths."""
    model_path = Path(model_path)
    meta_path = Path(meta_path)
    _bundle: dict | None = None

    def _load_bundle():
        nonlocal _bundle
        if _bundle is not None:
            return _bundle
        if not model_path.exists() or not meta_path.exists():
            _bundle = {"missing": True}
            return _bundle
        with open(meta_path) as f:
            meta = json.load(f)
        model = joblib.load(model_path)
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
                metadata={"error": f"missing {model_path.name} or {meta_path.name}"},
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

    get_signal.__name__ = f"ml_{model_path.stem}"
    return get_signal
