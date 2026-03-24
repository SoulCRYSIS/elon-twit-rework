"""Price-filtered historical entry + tiered exits."""

from approaches.approach_price_filter import get_signal as _entry
from approaches.exit_coupled import wrap_tiered_exit

get_signal = wrap_tiered_exit(_entry)
