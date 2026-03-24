"""Random Forest on tabular features (same engineering as XGBoost path; temporal train split)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from approaches.base import Signal
from approaches.ml_features import ALL_FEATURE_KEYS, build_ml_features
from approaches.ml_training_data import (
    build_training_dataframe,
    filter_seven_day_events,
    hist_winning_mid,
    temporal_event_ids,
)
from approaches.utils import parse_bracket

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EDGE_THRESHOLD = 0.08
N_SAMPLES_FRAC = 12

_model_cache: RandomForestClassifier | None = None
_scaler_cache: StandardScaler | None = None
_feature_cols: list[str] | None = None
_historical_mean_cache: float = 350.0


def _active_columns(train_df: pd.DataFrame, candidates: list[str]) -> list[str]:
    out = []
    for c in candidates:
        s = train_df[c].astype(float)
        if not np.isfinite(s).all():
            s = s.replace([np.inf, -np.inf], np.nan).fillna(s.median())
        if float(s.std()) < 1e-10:
            continue
        out.append(c)
    return out


def _train_model() -> tuple[RandomForestClassifier, StandardScaler, list[str], float]:
    global _model_cache, _scaler_cache, _feature_cols, _historical_mean_cache
    events_path = DATA_DIR / "events.parquet"
    markets_path = DATA_DIR / "markets.parquet"
    price_path = DATA_DIR / "price_history.parquet"
    if not all(p.exists() for p in [events_path, markets_path, price_path]):
        m = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
        sc = StandardScaler()
        return m, sc, list(ALL_FEATURE_KEYS)[:8], 350.0

    events = pd.read_parquet(events_path)
    markets = pd.read_parquet(markets_path)
    prices = pd.read_parquet(price_path)
    closed_7d = filter_seven_day_events(events)
    hmid = hist_winning_mid(closed_7d, markets)
    _historical_mean_cache = hmid
    fracs = np.linspace(0.08, 0.92, N_SAMPLES_FRAC)
    df = build_training_dataframe(closed_7d, markets, prices, fracs, hmid)
    if len(df) < 80:
        m = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
        sc = StandardScaler()
        return m, sc, list(ALL_FEATURE_KEYS)[:8], hmid

    train_ids, _, _ = temporal_event_ids(df, 0.75, 0.10)
    tr = df[df["event_id"].isin(train_ids)]
    cols = _active_columns(tr, list(ALL_FEATURE_KEYS))
    if len(cols) < 4:
        cols = [c for c in ALL_FEATURE_KEYS if c in tr.columns]

    sc = StandardScaler()
    X = sc.fit_transform(tr[cols])
    y = tr["resolved_yes"].values
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    cw = {0: 1.0, 1: float(neg / max(pos, 1))}
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=5,
        min_samples_leaf=8,
        class_weight=cw,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    _model_cache = model
    _scaler_cache = sc
    _feature_cols = cols
    return model, sc, cols, hmid


def get_signal(
    event_id: int,
    bracket: str,
    current_time: float,
    price_history: list[tuple[float, float]],
    market_data: dict,
    current_price: float | None = None,
    current_count: float | None = None,
    end_time: float | None = None,
    start_time: float | None = None,
    **kwargs,
) -> Signal:
    global _model_cache, _scaler_cache, _feature_cols, _historical_mean_cache
    if _model_cache is None:
        _model_cache, _scaler_cache, _feature_cols, _historical_mean_cache = _train_model()

    market_price = current_price if current_price is not None else (
        price_history[-1][1] if price_history else 0.05
    )
    low, high = parse_bracket(bracket)
    end = end_time if end_time is not None else current_time + 7 * 24 * 3600
    start = start_time if start_time is not None else end - 7 * 24 * 3600

    feats = build_ml_features(
        price_history, current_time, start, end, low, high, _historical_mean_cache
    )
    row = pd.DataFrame([[feats[c] for c in _feature_cols]], columns=_feature_cols)
    X = _scaler_cache.transform(row)
    prob = float(_model_cache.predict_proba(X)[0][1])
    edge = prob - market_price
    buy = edge >= EDGE_THRESHOLD and market_price < 0.5
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
