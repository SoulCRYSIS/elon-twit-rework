"""XGBoost entry + tiered exits (TP/SL/time keyed off avg buy_price)."""

from approaches.approach_xgboost import get_signal as _entry
from approaches.exit_coupled import wrap_tiered_exit

get_signal = wrap_tiered_exit(_entry)
