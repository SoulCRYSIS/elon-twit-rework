"""Kelly-sized historical entry + tiered exits."""

from approaches.approach_kelly import get_signal as _entry
from approaches.exit_coupled import wrap_tiered_exit

get_signal = wrap_tiered_exit(_entry)
