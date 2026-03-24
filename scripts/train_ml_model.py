#!/usr/bin/env python3
"""
Train XGBoost for YES resolution with ML best practices:

- **7-day events only** (duration_days >= 6): do not mix 2–5d with 7d.
- **Temporal split** by event end date (no random row split — avoids leakage).
- **Correlation analysis**: target correlation + prune highly redundant features.
- **Class imbalance**: scale_pos_weight.
- **Trade frequency**: tune edge threshold on validation to meet minimum buy rate.

Outputs to data/ml_artifacts/ (model.joblib, meta.json).
Run after: python scripts/fetch_data.py --refresh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
ARTIFACT_DIR = DATA_DIR / "ml_artifacts"
RESULTS_DIR = ROOT / "results"

from approaches.ml_features import ALL_FEATURE_KEYS  # noqa: E402
from approaches.ml_training_data import (  # noqa: E402
    build_training_dataframe,
    filter_seven_day_events,
    hist_winning_mid,
    temporal_event_ids,
)
from shared import progress_log  # noqa: E402

TRAIN_PHASES = 6


def target_correlations(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    y = df["resolved_yes"].astype(float)
    out = {c: float(df[c].corr(y)) for c in cols}
    return pd.Series(out).sort_values(key=abs, ascending=False)


def select_features_low_redundancy(
    df: pd.DataFrame,
    candidates: list[str],
    y: pd.Series,
    max_pairwise_abs_corr: float = 0.92,
) -> list[str]:
    cy = df[candidates].corrwith(y).abs().sort_values(ascending=False)
    picked: list[str] = []
    for name in cy.index:
        ok = True
        for g in picked:
            if abs(df[name].corr(df[g])) > max_pairwise_abs_corr:
                ok = False
                break
        if ok:
            picked.append(name)
    return picked


def tune_edge_threshold(
    probs: np.ndarray,
    market_prices: np.ndarray,
    y_true: np.ndarray,
    min_buy_rate: float,
    max_buy_rate: float,
    max_market_price: float,
) -> float:
    """
    Cap how often we signal buy on validation: among price<max_market_price rows,
    take the top-K edges where K = floor(max_buy_rate * N). Threshold = K-th largest edge.
    Then raise threshold (fewer buys) if precision is poor, down to min_buy_rate floor.
    """
    n = len(probs)
    eligible = market_prices < max_market_price
    edge = probs - market_prices
    e_elig = edge[eligible]
    if len(e_elig) < 10:
        return 0.12

    k_max = max(1, int(max_buy_rate * n))
    k_min = max(1, int(min_buy_rate * n))
    k_max = min(k_max, len(e_elig))
    k_min = min(k_min, k_max)

    best_t = 0.10
    best_prec = -1.0
    for k in range(k_max, k_min - 1, -1):
        order = np.argsort(-e_elig)
        t = float(e_elig[order[k - 1]])
        buy = (edge >= t) & eligible
        if buy.sum() < 15:
            continue
        prec = float(y_true[buy].mean())
        if prec > best_prec + 1e-6 or (abs(prec - best_prec) < 1e-6 and t < best_t):
            best_prec = prec
            best_t = t
    if best_prec < 0:
        order = np.argsort(-e_elig)
        return float(e_elig[order[k_max - 1]])
    return best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--min-buy-rate", type=float, default=0.01)
    parser.add_argument("--max-buy-rate", type=float, default=0.18)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--max-pairwise-corr", type=float, default=0.92)
    args = parser.parse_args()

    if not (DATA_DIR / "events.parquet").exists():
        progress_log("train_ml_model", "missing data/*.parquet — run scripts/fetch_data.py first.")
        sys.exit(1)

    progress_log(
        "train_ml_model",
        "start | phases: (1) load (2) build dataset (3) split (4) feature stats (5) train (6) write artifacts",
    )
    events = pd.read_parquet(DATA_DIR / "events.parquet")
    markets = pd.read_parquet(DATA_DIR / "markets.parquet")
    prices = pd.read_parquet(DATA_DIR / "price_history.parquet")

    events_7d = filter_seven_day_events(events)
    progress_log(
        "train_ml_model",
        f"7-day+ closed events: {len(events_7d)} (shorter horizons excluded)",
        step=1,
        total=TRAIN_PHASES,
    )

    hmid = hist_winning_mid(events_7d, markets)
    sample_fracs = np.linspace(0.08, 0.92, 12)
    df = build_training_dataframe(events_7d, markets, prices, sample_fracs, hmid)
    progress_log(
        "train_ml_model",
        f"training table: {len(df):,} rows (event×time×bracket)",
        step=2,
        total=TRAIN_PHASES,
    )

    if len(df) < 80:
        progress_log("train_ml_model", "too few rows for stable training (need ≥80).")
        sys.exit(1)

    train_ids, val_ids, test_ids = temporal_event_ids(df, args.train_frac, args.val_frac)
    tr = df[df["event_id"].isin(train_ids)]
    va = df[df["event_id"].isin(val_ids)]
    te = df[df["event_id"].isin(test_ids)]
    progress_log(
        "train_ml_model",
        f"temporal split | events train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
        f"| rows train={len(tr)} val={len(va)} test={len(te)}",
        step=3,
        total=TRAIN_PHASES,
    )

    candidates = list(ALL_FEATURE_KEYS)
    cy = target_correlations(tr, candidates)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cy.to_frame(name="corr_with_target").to_csv(RESULTS_DIR / "feature_target_correlation_train.csv")

    progress_log("train_ml_model", "feature vs target (Pearson, train), |corr| descending:", step=4, total=TRAIN_PHASES)
    print(cy.to_string())

    if args.analyze_only:
        cm = tr[candidates].corr().abs()
        np.fill_diagonal(cm.values, 0.0)
        cm.max(axis=1).sort_values(ascending=False).to_frame(name="max_abs_corr_other").to_csv(
            RESULTS_DIR / "feature_max_pairwise_corr_train.csv"
        )
        progress_log(
            "train_ml_model",
            "analyze-only: wrote feature_target_correlation_train.csv + feature_max_pairwise_corr_train.csv — done.",
            step=6,
            total=TRAIN_PHASES,
        )
        return

    y_tr = tr["resolved_yes"]
    selected = select_features_low_redundancy(tr, candidates, y_tr, args.max_pairwise_corr)
    # Drop zero-variance / NaN-only columns (breaks scaler and corr)
    active = []
    for c in selected:
        s = tr[c].astype(float)
        if not np.isfinite(s).all():
            s = s.replace([np.inf, -np.inf], np.nan).fillna(s.median())
        if float(s.std()) < 1e-10:
            continue
        active.append(c)
    selected = active
    progress_log(
        "train_ml_model",
        f"selected {len(selected)} features (redundancy + variance): {selected}",
        step=5,
        total=TRAIN_PHASES,
    )
    if len(selected) < 4:
        progress_log("train_ml_model", "too few features after filtering (need ≥4).")
        sys.exit(1)

    scaler = StandardScaler()
    scaler.fit(tr[selected])
    X_tr = scaler.transform(tr[selected])
    X_va = scaler.transform(va[selected])
    X_te = scaler.transform(te[selected])

    pos = (y_tr == 1).sum()
    neg = (y_tr == 0).sum()
    spw = float(neg / max(pos, 1))

    try:
        import xgboost as xgb
    except ImportError:
        progress_log("train_ml_model", "missing dependency: pip install xgboost")
        sys.exit(1)

    eval_set = [(X_va, va["resolved_yes"])] if len(va) >= 10 else None
    fit_kw: dict = {}
    if eval_set:
        fit_kw["eval_set"] = eval_set
        fit_kw["verbose"] = False

    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.06,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.1,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
    )
    progress_log("train_ml_model", "fitting XGBClassifier…", step=5, total=TRAIN_PHASES)
    model.fit(X_tr, y_tr, **fit_kw)

    def report(name: str, X, part: pd.DataFrame):
        y = part["resolved_yes"].values
        p = model.predict_proba(X)[:, 1]
        try:
            auc = roc_auc_score(y, p)
        except ValueError:
            auc = float("nan")
        try:
            ap = average_precision_score(y, p)
        except ValueError:
            ap = float("nan")
        br = brier_score_loss(y, p)
        progress_log("train_ml_model", f"  {name}: ROC-AUC={auc:.4f} PR-AUC={ap:.4f} Brier={br:.4f}")

    progress_log("train_ml_model", "metrics:", step=5, total=TRAIN_PHASES)
    report("train", X_tr, tr)
    report("val  ", X_va, va)
    report("test ", X_te, te)

    probs_va = model.predict_proba(X_va)[:, 1]
    edge_t = tune_edge_threshold(
        probs_va,
        va["current_price"].values,
        va["resolved_yes"].values,
        min_buy_rate=args.min_buy_rate,
        max_buy_rate=args.max_buy_rate,
        max_market_price=0.5,
    )
    buys_va = ((probs_va - va["current_price"].values) >= edge_t) & (va["current_price"].values < 0.5)
    progress_log(
        "train_ml_model",
        f"tuned edge threshold (val)={edge_t:.4f} | val buy rate={buys_va.mean():.2%} ({buys_va.sum()}/{len(va)} rows)",
        step=5,
        total=TRAIN_PHASES,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "xgb_model.joblib")

    meta = {
        "feature_names": selected,
        "edge_threshold": edge_t,
        "max_market_price_buy": 0.5,
        "hist_winning_mid": hmid,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "train_events": len(train_ids),
        "val_events": len(val_ids),
        "test_events": len(test_ids),
        "n_train_rows": len(tr),
        "scale_pos_weight": spw,
        "seven_day_min_duration_days": 6,
        "notes": "Temporal split by event end; 7d-only; redundancy-pruned features.",
    }
    with open(ARTIFACT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    progress_log(
        "train_ml_model",
        f"wrote data/ml_artifacts/xgb_model.joblib + meta.json — done. Next: python scripts/run_backtest.py",
        step=6,
        total=TRAIN_PHASES,
    )


if __name__ == "__main__":
    main()
