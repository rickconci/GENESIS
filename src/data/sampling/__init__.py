"""Which train perturbations and cells the main loader sees.

HBSS thins conditions. ``cell_subsampling`` thins cells inside a condition.
``eval_cache_cell_sets`` picks the cells for the cell-set loader in
``src.data.ablations``.
"""

from .hbss_sampling import (
    HBSSResult,
    build_tuple_filter,
    run_hbss,
)

__all__ = [
    "HBSSResult",
    "build_tuple_filter",
    "run_hbss",
]
