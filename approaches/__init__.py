"""Prediction approaches for Elon tweet count markets."""

from approaches.base import Signal


def get_approach(name: str):
    """Lazy load approach by name."""
    _map = {
        "historical": "approach_historical",
        "negbin": "approach_negbin",
        "momentum": "approach_momentum",
        "ml": "approach_ml",
        "lstm": "approach_lstm",
        "ranking": "approach_ranking",
        "price_filter": "approach_price_filter",
        "ensemble": "approach_ensemble",
        "xgboost": "approach_xgboost",
        "xgboost_pick": "approach_xgboost_pick",
        "xgboost_ev": "approach_xgboost_ev",
        "xgboost_ev_m08": "approach_xgboost_ev_m08",
        "hgb_ev": "approach_hgb_ev",
        "kelly": "approach_kelly",
        "regime": "approach_regime",
        "bracket_spread": "approach_bracket_spread",
        "time_weighted": "approach_time_weighted",
        "rules": "approach_rules",
        "xgb_exit": "approach_xgb_exit",
        "ml_exit": "approach_ml_exit",
        "rules_exit": "approach_rules_exit",
        "historical_exit": "approach_historical_exit",
        "negbin_exit": "approach_negbin_exit",
        "price_filter_exit": "approach_price_filter_exit",
        "kelly_exit": "approach_kelly_exit",
        "regime_exit": "approach_regime_exit",
        "bracket_spread_exit": "approach_bracket_spread_exit",
        "time_weighted_exit": "approach_time_weighted_exit",
        "ranking_exit": "approach_ranking_exit",
        "momentum_tier": "approach_momentum_tier",
        "ensemble_exit": "approach_ensemble_exit",
    }
    if name not in _map:
        raise ValueError(f"Unknown approach: {name}")
    mod = __import__(f"approaches.{_map[name]}", fromlist=["get_signal"])
    return mod.get_signal


APPROACHES = [
    "historical",
    "negbin",
    "momentum",
    "ml",
    "lstm",
    "ranking",
    "price_filter",
    "ensemble",
    "xgboost",
    "xgboost_pick",
    "xgboost_ev",
    "xgboost_ev_m08",
    "hgb_ev",
    "kelly",
    "regime",
    "bracket_spread",
    "time_weighted",
    "rules",
    "xgb_exit",
    "ml_exit",
    "rules_exit",
    "historical_exit",
    "negbin_exit",
    "price_filter_exit",
    "kelly_exit",
    "regime_exit",
    "bracket_spread_exit",
    "time_weighted_exit",
    "ranking_exit",
    "momentum_tier",
    "ensemble_exit",
]
