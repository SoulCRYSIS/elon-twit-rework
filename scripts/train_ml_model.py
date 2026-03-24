#!/usr/bin/env python3
"""
Train tabular YES models for ML best practices:

- **7-day events only** (duration_days >= 6): do not mix 2–5d with 7d.
- **Train/val split by event** (default: random 80/20). Optional chronological split via ``--split temporal``.
- **Correlation analysis**: target correlation + prune highly redundant features.
- **Class imbalance**: scale_pos_weight (XGB default path).
- **Trade frequency**: tune edge threshold on validation (precision or mean $/stake on buys).

**Default** (unchanged): XGBoost → `data/ml_artifacts/xgb_model.joblib` + `meta.json`.

**One process, same training matrix** (after a single ``fetch_data``): baseline + ``xgb_ev_pnl_m08`` share one parquet load and one ``build_training_dataframe``::

  python scripts/train_ml_model.py --also-xgb-ev-m08

**Experiments** (expected value / cheap-tail emphasis):
  python scripts/train_ml_model.py --ev-weighted --tune-on pnl
    → `xgb_model_ev.joblib`, `meta_ev.json` — backtest `--approaches xgboost_ev`
  python scripts/train_ml_model.py --classifier hgb --ev-weighted --tune-on pnl
    → `hgb_ev.joblib`, `meta_hgb_ev.json` — backtest `--approaches hgb_ev`

Run after: python scripts/fetch_data.py
"""

from __future__ import annotations

import argparse
import json
import re
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
    random_event_train_val_split,
    temporal_event_train_val_split,
)
from shared import progress_log  # noqa: E402

TRAIN_PHASES = 6


def _fit_xgb_tune_write(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
    tr: pd.DataFrame,
    va: pd.DataFrame,
    te: pd.DataFrame,
    selected: list[str],
    scaler: StandardScaler,
    hmid: float,
    train_ids: set,
    val_ids: set,
    spw: float,
    *,
    ev_weighted: bool,
    tune_on: str,
    min_buy_rate: float,
    max_buy_rate: float,
    model_path: Path,
    meta_path: Path,
    artifact_tag: str | None,
    classifier: str,
    split: str,
    train_frac: float,
    random_state: int | None,
    run_label: str,
) -> None:
    try:
        import xgboost as xgb
    except ImportError:
        progress_log("train_ml_model", "missing dependency: pip install xgboost")
        sys.exit(1)

    y_tr = tr["resolved_yes"]
    sw_tr: np.ndarray | None = None
    if ev_weighted:
        sw_tr = edge_focus_sample_weights(tr["current_price"].values, y_tr.values)

    eval_set = [(X_va, va["resolved_yes"])] if len(va) >= 10 else None
    fit_kw: dict = {}
    if eval_set:
        fit_kw["eval_set"] = eval_set
        fit_kw["verbose"] = False

    spw_fit = 1.0 if ev_weighted else spw
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.06,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.1,
        scale_pos_weight=spw_fit,
        random_state=42,
        eval_metric="logloss",
    )
    progress_log(
        "train_ml_model",
        f"fitting XGBClassifier {run_label} (ev_weighted={ev_weighted}, tune_on={tune_on})…",
        step=5,
        total=TRAIN_PHASES,
    )
    model.fit(X_tr, y_tr, sample_weight=sw_tr, **fit_kw)

    def report(name: str, X: np.ndarray, part: pd.DataFrame):
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

    progress_log("train_ml_model", f"metrics {run_label}:", step=5, total=TRAIN_PHASES)
    report("train", X_tr, tr)
    report("val  ", X_va, va)
    if len(te) > 0:
        report("test ", X_te, te)

    probs_va = model.predict_proba(X_va)[:, 1]
    if tune_on == "pnl":
        edge_t = tune_edge_threshold_pnl(
            probs_va,
            va["current_price"].values,
            va["resolved_yes"].values,
            min_buy_rate=min_buy_rate,
            max_buy_rate=max_buy_rate,
            max_market_price=0.5,
        )
    else:
        edge_t = tune_edge_threshold(
            probs_va,
            va["current_price"].values,
            va["resolved_yes"].values,
            min_buy_rate=min_buy_rate,
            max_buy_rate=max_buy_rate,
            max_market_price=0.5,
        )
    buys_va = ((probs_va - va["current_price"].values) >= edge_t) & (va["current_price"].values < 0.5)
    px_b = va["current_price"].values[buys_va]
    y_b = va["resolved_yes"].values[buys_va]
    mean_pnl_val = (
        float(np.mean(np.where(y_b == 1, (1.0 - px_b) / np.maximum(px_b, 1e-6), -1.0)))
        if buys_va.any()
        else 0.0
    )
    progress_log(
        "train_ml_model",
        f"{run_label} tuned edge (val)={edge_t:.4f} tune_on={tune_on} | val buys={buys_va.sum()}/{len(va)} "
        f"({buys_va.mean():.2%}) | mean $/stake on val buys={mean_pnl_val:.4f}",
        step=5,
        total=TRAIN_PHASES,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    tag_safe = artifact_tag
    meta = {
        "feature_names": selected,
        "edge_threshold": edge_t,
        "max_market_price_buy": 0.5,
        "hist_winning_mid": hmid,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "train_events": len(train_ids),
        "val_events": len(val_ids),
        "test_events": 0,
        "n_train_rows": len(tr),
        "scale_pos_weight": spw,
        "seven_day_min_duration_days": 6,
        "classifier": classifier,
        "ev_weighted": ev_weighted,
        "tune_on": tune_on,
        "model_file": model_path.name,
        "artifact_tag": tag_safe,
        "split": split,
        "train_frac": train_frac,
        "random_state": random_state if split == "random" else None,
        "notes": f"Train/val only ({split} split by event); 7d-only; redundancy-pruned features.",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    progress_log(
        "train_ml_model",
        f"wrote {model_path.name} + {meta_path.name} ({run_label})",
        step=5,
        total=TRAIN_PHASES,
    )


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


def edge_focus_sample_weights(
    market_prices: np.ndarray,
    y_true: np.ndarray,
    *,
    payoff_cap: float = 40.0,
    cheap_px: float = 0.15,
    cheap_neg_boost: float = 1.35,
) -> np.ndarray:
    """
    Upweight training rows where a correct YES at the observed market price would pay off heavily
    (cheap winners ≈ (1-p)/p), and slightly upweight cheap negatives so the model learns tail risk.
    """
    p = np.clip(market_prices.astype(float), 0.01, 0.99)
    y = y_true.astype(int)
    win_w = (1.0 - p) / p
    win_w = np.minimum(win_w, payoff_cap)
    w = np.where(y == 1, win_w, np.ones_like(p))
    cheap = p < cheap_px
    w = np.where(cheap & (y == 0), np.maximum(w, cheap_neg_boost), w)
    w = w * (len(w) / np.maximum(w.sum(), 1e-9))
    return w


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


def tune_edge_threshold_pnl(
    probs: np.ndarray,
    market_prices: np.ndarray,
    y_true: np.ndarray,
    min_buy_rate: float,
    max_buy_rate: float,
    max_market_price: float,
) -> float:
    """
    Pick edge threshold on validation to maximize mean realized $ return per $1 staked on buys
    (YES wins: (1-p)/p, losses: -1), subject to buy-rate bounds — aligns with cheap-tail edge.
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

    order_local = np.argsort(-e_elig)
    best_t = 0.10
    best_mean_pnl = -1e18
    for k in range(k_max, k_min - 1, -1):
        t = float(e_elig[order_local[k - 1]])
        buy = (edge >= t) & eligible
        if int(buy.sum()) < 15:
            continue
        px = market_prices[buy]
        yb = y_true[buy]
        pnl = np.where(yb == 1, (1.0 - px) / np.maximum(px, 1e-6), -1.0)
        mean_pnl = float(np.mean(pnl))
        if mean_pnl > best_mean_pnl + 1e-12 or (
            abs(mean_pnl - best_mean_pnl) < 1e-12 and t < best_t
        ):
            best_mean_pnl = mean_pnl
            best_t = t
    if best_mean_pnl < -1e17:
        order = np.argsort(-e_elig)
        return float(e_elig[order[k_max - 1]])
    return best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of distinct events in the training set (rest → validation). No test set.",
    )
    parser.add_argument(
        "--split",
        choices=("random", "temporal"),
        default="random",
        help="random: shuffle events then split (default). temporal: split by event end time order.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="RNG seed for --split random (event shuffle).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Deprecated: ignored. Validation size is 1 - train-frac.",
    )
    parser.add_argument("--min-buy-rate", type=float, default=0.01)
    parser.add_argument("--max-buy-rate", type=float, default=0.18)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--max-pairwise-corr", type=float, default=0.92)
    parser.add_argument(
        "--ev-weighted",
        action="store_true",
        help="Train with sample weights emphasizing cheap YES wins (payoff ~ (1-p)/p).",
    )
    parser.add_argument(
        "--tune-on",
        choices=("precision", "pnl"),
        default="precision",
        help="Validation threshold: maximize buy precision (default) or mean $/stake on buys (pnl).",
    )
    parser.add_argument(
        "--classifier",
        choices=("xgboost", "hgb"),
        default="xgboost",
        help="xgboost (default) or sklearn HistGradientBoostingClassifier (no extra pip deps).",
    )
    parser.add_argument(
        "--artifact-tag",
        type=str,
        default="",
        help="Non-empty: write xgb_model_<tag>.joblib + meta_<tag>.json (or hgb_<tag> / meta_hgb_<tag>).",
    )
    parser.add_argument(
        "--also-xgb-ev-m08",
        action="store_true",
        help="With default baseline XGB only: also write xgb_ev_pnl_m08 artifacts using the same train/val matrix.",
    )
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

    if args.split == "random":
        train_ids, val_ids = random_event_train_val_split(
            df, args.train_frac, random_state=args.random_state
        )
        split_note = f"random (seed={args.random_state})"
    else:
        train_ids, val_ids = temporal_event_train_val_split(df, args.train_frac)
        split_note = "temporal (by event end)"
    test_ids: set = set()
    tr = df[df["event_id"].isin(train_ids)]
    va = df[df["event_id"].isin(val_ids)]
    te = df[df["event_id"].isin(test_ids)]
    progress_log(
        "train_ml_model",
        f"split={split_note} | events train={len(train_ids)} val={len(val_ids)} "
        f"| rows train={len(tr)} val={len(va)}",
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
    X_te = scaler.transform(te[selected]) if len(te) > 0 else np.empty((0, len(selected)), dtype=float)

    pos = (y_tr == 1).sum()
    neg = (y_tr == 0).sum()
    spw = float(neg / max(pos, 1))

    tag_raw = (args.artifact_tag or "").strip()
    tag_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", tag_raw) if tag_raw else ""

    use_standard_artifacts = (
        args.classifier == "xgboost"
        and not args.ev_weighted
        and args.tune_on == "precision"
        and not tag_safe
    )
    if use_standard_artifacts:
        model_path = ARTIFACT_DIR / "xgb_model.joblib"
        meta_path = ARTIFACT_DIR / "meta.json"
    elif tag_safe:
        if args.classifier == "hgb":
            model_path = ARTIFACT_DIR / f"hgb_{tag_safe}.joblib"
            meta_path = ARTIFACT_DIR / f"meta_hgb_{tag_safe}.json"
        else:
            model_path = ARTIFACT_DIR / f"xgb_model_{tag_safe}.joblib"
            meta_path = ARTIFACT_DIR / f"meta_{tag_safe}.json"
    elif args.classifier == "hgb":
        model_path = ARTIFACT_DIR / "hgb_ev.joblib"
        meta_path = ARTIFACT_DIR / "meta_hgb_ev.json"
    else:
        model_path = ARTIFACT_DIR / "xgb_model_ev.joblib"
        meta_path = ARTIFACT_DIR / "meta_ev.json"

    sw_tr: np.ndarray | None = None
    if args.ev_weighted:
        sw_tr = edge_focus_sample_weights(tr["current_price"].values, y_tr.values)

    rs = args.random_state if args.split == "random" else None

    if args.classifier == "xgboost":
        if args.also_xgb_ev_m08 and not use_standard_artifacts:
            progress_log(
                "train_ml_model",
                "--also-xgb-ev-m08 only with the default baseline (xgboost, --tune-on precision, "
                "no --ev-weighted, no --artifact-tag).",
            )
            sys.exit(1)
        lbl = "[baseline]" if args.also_xgb_ev_m08 and use_standard_artifacts else ""
        _fit_xgb_tune_write(
            X_tr,
            X_va,
            X_te,
            tr,
            va,
            te,
            selected,
            scaler,
            hmid,
            train_ids,
            val_ids,
            spw,
            ev_weighted=args.ev_weighted,
            tune_on=args.tune_on,
            min_buy_rate=args.min_buy_rate,
            max_buy_rate=args.max_buy_rate,
            model_path=model_path,
            meta_path=meta_path,
            artifact_tag=tag_safe or None,
            classifier=args.classifier,
            split=args.split,
            train_frac=args.train_frac,
            random_state=rs,
            run_label=lbl,
        )
        if args.also_xgb_ev_m08 and use_standard_artifacts:
            _fit_xgb_tune_write(
                X_tr,
                X_va,
                X_te,
                tr,
                va,
                te,
                selected,
                scaler,
                hmid,
                train_ids,
                val_ids,
                spw,
                ev_weighted=True,
                tune_on="pnl",
                min_buy_rate=args.min_buy_rate,
                max_buy_rate=0.08,
                model_path=ARTIFACT_DIR / "xgb_model_xgb_ev_pnl_m08.joblib",
                meta_path=ARTIFACT_DIR / "meta_xgb_ev_pnl_m08.json",
                artifact_tag="xgb_ev_pnl_m08",
                classifier="xgboost",
                split=args.split,
                train_frac=args.train_frac,
                random_state=rs,
                run_label="[xgb_ev_pnl_m08]",
            )
        if use_standard_artifacts and args.also_xgb_ev_m08:
            bt = (
                "python scripts/run_backtest.py --approaches xgboost,xgboost_pick,xgboost_ev_m08"
            )
        elif use_standard_artifacts:
            bt = "python scripts/run_backtest.py --approaches xgboost"
        else:
            bt = "python scripts/run_backtest.py --approaches xgboost_ev"
        progress_log(
            "train_ml_model",
            f"done. Next: {bt}",
            step=6,
            total=TRAIN_PHASES,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier

        if args.also_xgb_ev_m08:
            progress_log("train_ml_model", "--also-xgb-ev-m08 applies only to --classifier xgboost.")
            sys.exit(1)

        progress_log(
            "train_ml_model",
            f"fitting HistGradientBoostingClassifier (ev_weighted={args.ev_weighted}, tune_on={args.tune_on})…",
            step=5,
            total=TRAIN_PHASES,
        )
        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            learning_rate=0.07,
            min_samples_leaf=40,
            l2_regularization=0.8,
            random_state=42,
        )
        model.fit(X_tr, y_tr, sample_weight=sw_tr)

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
        if len(te) > 0:
            report("test ", X_te, te)

        probs_va = model.predict_proba(X_va)[:, 1]
        if args.tune_on == "pnl":
            edge_t = tune_edge_threshold_pnl(
                probs_va,
                va["current_price"].values,
                va["resolved_yes"].values,
                min_buy_rate=args.min_buy_rate,
                max_buy_rate=args.max_buy_rate,
                max_market_price=0.5,
            )
        else:
            edge_t = tune_edge_threshold(
                probs_va,
                va["current_price"].values,
                va["resolved_yes"].values,
                min_buy_rate=args.min_buy_rate,
                max_buy_rate=args.max_buy_rate,
                max_market_price=0.5,
            )
        buys_va = ((probs_va - va["current_price"].values) >= edge_t) & (va["current_price"].values < 0.5)
        px_b = va["current_price"].values[buys_va]
        y_b = va["resolved_yes"].values[buys_va]
        mean_pnl_val = (
            float(np.mean(np.where(y_b == 1, (1.0 - px_b) / np.maximum(px_b, 1e-6), -1.0)))
            if buys_va.any()
            else 0.0
        )
        progress_log(
            "train_ml_model",
            f"tuned edge (val)={edge_t:.4f} tune_on={args.tune_on} | val buys={buys_va.sum()}/{len(va)} "
            f"({buys_va.mean():.2%}) | mean $/stake on val buys={mean_pnl_val:.4f}",
            step=5,
            total=TRAIN_PHASES,
        )

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        meta = {
            "feature_names": selected,
            "edge_threshold": edge_t,
            "max_market_price_buy": 0.5,
            "hist_winning_mid": hmid,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "train_events": len(train_ids),
            "val_events": len(val_ids),
            "test_events": 0,
            "n_train_rows": len(tr),
            "scale_pos_weight": spw,
            "seven_day_min_duration_days": 6,
            "classifier": args.classifier,
            "ev_weighted": args.ev_weighted,
            "tune_on": args.tune_on,
            "model_file": model_path.name,
            "artifact_tag": tag_safe or None,
            "split": args.split,
            "train_frac": args.train_frac,
            "random_state": rs,
            "notes": f"Train/val only ({args.split} split by event); 7d-only; redundancy-pruned features.",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        if use_standard_artifacts:
            bt = "python scripts/run_backtest.py --approaches xgboost"
        elif args.classifier == "hgb":
            bt = "python scripts/run_backtest.py --approaches hgb_ev"
        else:
            bt = "python scripts/run_backtest.py --approaches xgboost_ev"
        progress_log(
            "train_ml_model",
            f"wrote {model_path} + {meta_path.name} — done. Next: {bt}",
            step=6,
            total=TRAIN_PHASES,
        )


if __name__ == "__main__":
    main()
