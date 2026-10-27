"""Caches built from plate h5ads: HVG mmap CSR (train) and per-condition NPZ (eval)."""

from src.data.cache.raw_hvg_cache import (
    HvgCacheCollection,
    RawHvgShardCache,
    open_cache,
)
from src.data.cache.raw_hvg_cache_build import (
    obs_keys_for_raw_hvg_cache,
    run_raw_hvg_cache_build,
)
from src.data.cache.eval_cache_build import run_eval_cache_build

__all__ = [
    "HvgCacheCollection",
    "RawHvgShardCache",
    "obs_keys_for_raw_hvg_cache",
    "open_cache",
    "run_raw_hvg_cache_build",
    "run_eval_cache_build",
]
