#!/usr/bin/env python3
"""Build the caches a dataset needs, from its h5ads and a PertGen config.

  train    HVG mmap CSR training cache (training also builds it when build_cache: true)
  eval     per-condition truth NPZs + control pools that every scorer reads
  summary  one row per condition with num_cells: the input to make_splits.py
  stats    per-condition log1p means and cell counts in two seeded halves, one pass over the
           train cache's shards (the copy_closest neighbour's input; src/baselines/condition_stats.py)

The config names the data (data_dir, files_glob, hvg_genes_path), the condition
columns (model_conditioning_variables), the control and its match keys, and the
split. Nothing here is dataset-specific.

  python scripts/data/build_cache.py train   --config <cfg>
  python scripts/data/build_cache.py eval    --config <cfg> [--all-conditions]
  python scripts/data/build_cache.py summary --config <cfg> --columns cell_name drug dose plate --out <parquet>
  python scripts/data/build_cache.py stats   --cache-dir <hvg_cache/geneset_...> --columns cell_name drug dose plate

Controls that follow a split (e.g. PBMC PBS wells, some held out as test):

  python scripts/data/build_cache.py eval --config <cfg> --all-conditions \\
      --control-split <split.yaml> --control-unit-column bc1_well

writes one control pool per well, pools each donor x cell type from its train
wells only, and writes held-out wells as test truth.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _config(args: argparse.Namespace, overrides: list[str] = ()):
    from src.config.loader import load_config

    config = load_config(config_path=Path(args.config), overrides=list(overrides))
    for key in ("data_dir", "files_glob"):  # e.g. point a build at a subset h5ad
        if getattr(args, key, None):
            setattr(config, key, str(getattr(args, key)))
    for key in ("data_dir", "hvg_genes_path"):
        if not getattr(config, key, None):
            raise SystemExit(f"{args.config}: {key} is required")
    return config


def _h5ads(config) -> list[Path]:
    h5ads = sorted(Path(config.data_dir).glob(config.files_glob or "*.h5ad"))
    if not h5ads:
        raise SystemExit(f"no h5ad files matching {config.files_glob or '*.h5ad'} under {config.data_dir}")
    return h5ads


def _keys(config) -> tuple[list[str], dict, list[str]]:
    from src.data.data_utils import resolve_control_match_keys

    mcv = list(config.model_conditioning_variables or ())
    ctrl = dict(config.control_condition or {})
    return mcv, ctrl, list(resolve_control_match_keys(mcv, ctrl, config.control_match_keys))


def _split(path_or_none, config, *, use_config: bool):
    from src.data.cache.eval_cache_build import resolve_split_config_path

    raw = path_or_none or (getattr(config, "split_config", None) if use_config else None)
    return resolve_split_config_path(raw, repo_root=ROOT) if raw else None


def cmd_train(args: argparse.Namespace, overrides: list[str]) -> None:
    from src.config.config import resolve_cache_dir
    from src.data.cache.raw_hvg_cache_build import (
        obs_keys_for_raw_hvg_cache, resolve_force_csr, run_raw_hvg_cache_build,
    )

    config = _config(args, overrides)
    mcv, ctrl, mk = _keys(config)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else resolve_cache_dir(config)
    if cache_dir is None:
        raise SystemExit("no cache dir: pass --cache-dir or set cache_dir in the config")
    summary = run_raw_hvg_cache_build(
        data_dir=Path(config.data_dir), hvg_genes_path=Path(config.hvg_genes_path), cache_dir=cache_dir,
        obs_keys=obs_keys_for_raw_hvg_cache(mcv), files_glob=config.files_glob or "*.h5ad",
        chunk_rows=args.chunk_rows, n_workers=args.n_workers,
        shards_filter={int(x) for x in args.shards.split(",") if x.strip()} or None,
        force=resolve_force_csr(force_flag=args.force, explicit_cache_dir=args.cache_dir is not None,
                                config_overwrite_cache=bool(getattr(config, "overwrite_cache", False))),
        ranges_per_shard=args.ranges_per_shard, condition_variables=mcv, control_condition=ctrl,
        control_match_keys=mk,
    )
    logging.info("train cache: %s", summary)


def cmd_eval(args: argparse.Namespace) -> None:
    from src.config.config import resolve_eval_cache_dir
    from src.data.cache.eval_cache_build import probe_h5ad, run_eval_cache_build
    from src.eval.cache.eval_cache_io import gene_set_hash

    config = _config(args)
    mcv, ctrl, mk = _keys(config)
    split = None if args.all_conditions else _split(args.split_config, config, use_config=True)
    control_split = _split(args.control_split, config, use_config=False)
    overwrite = bool(args.force or getattr(config, "overwrite_eval_cache", False))
    joins = list(getattr(config, "metadata_joins", None) or ())

    h5ads = _h5ads(config)
    hvg_idx, genes, norm_id = probe_h5ad(h5ads[0], Path(config.hvg_genes_path), args.normalization_id)
    out = Path(args.output_dir).resolve() if args.output_dir else resolve_eval_cache_dir(config, normalization_id=norm_id)
    review = run_eval_cache_build(
        data_dir=Path(config.data_dir), h5ad_paths=h5ads, output_dir=out, hvg_indices=hvg_idx,
        hvg_genes_path=Path(config.hvg_genes_path), condition_variables=mcv, control_condition=ctrl,
        match_keys=mk, gene_set_hash_value=gene_set_hash(genes), n_genes=len(genes), normalization_id=norm_id,
        overwrite=overwrite, max_workers=args.n_workers, split_config=split, metadata_joins=joins,
        condition_summary_path=args.condition_summary, min_cells=int(getattr(config, "eval_min_cells", 4)),
        dry_run=args.dry_run, control_unit_column=args.control_unit_column, control_split_config=control_split,
    )
    logging.info("eval cache: %s", {k: v for k, v in review.items() if not k.startswith("missing")})


def cmd_summary(args: argparse.Namespace) -> None:
    from src.data.cache.condition_summary import condition_summary_from_h5ads

    config = _config(args)
    out = condition_summary_from_h5ads(_h5ads(config), args.columns, metadata_config=config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    logging.info("condition summary: %d rows -> %s", len(out), args.out)


def cmd_stats(args: argparse.Namespace) -> None:
    from src.baselines.condition_stats import build_condition_stats

    build_condition_stats(args.cache_dir, args.columns, seed=args.seed, workers=args.n_workers)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="HVG mmap CSR training cache; extra key=value args override the config")
    t.add_argument("--config", required=True)
    t.add_argument("--cache-dir", type=Path, help="use this cache as-is (config overwrite_cache ignored)")
    t.add_argument("--n-workers", type=int, default=16)
    t.add_argument("--chunk-rows", type=int, default=32_768)
    t.add_argument("--ranges-per-shard", type=int, default=4)
    t.add_argument("--shards", default="", help="comma-separated shard ids (default: all)")
    t.add_argument("--force", action="store_true", help="rebuild CSR shards from scratch")

    e = sub.add_parser("eval", help="per-condition truth NPZs + control pools")
    e.add_argument("--config", required=True)
    e.add_argument("--output-dir", type=Path)
    e.add_argument("--data-dir", type=Path, help="override the config's data_dir")
    e.add_argument("--files-glob", help="override the config's files_glob")
    e.add_argument("--split-config", type=Path, help="default: the config's split_config")
    e.add_argument("--all-conditions", action="store_true", help="every condition, not just val/test")
    e.add_argument("--control-split", type=Path, help="controls follow this split (held-out controls -> truth)")
    e.add_argument("--control-unit-column", help="one control pool per unit of this column (e.g. a well)")
    e.add_argument("--normalization-id", help="default: probed from the data")
    e.add_argument("--condition-summary", type=Path, help="plan targets from this summary")
    e.add_argument("--n-workers", type=int, default=14)
    e.add_argument("--force", action="store_true")
    e.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("summary", help="cells per condition (+ metadata joins) over every h5ad")
    s.add_argument("--config", required=True)
    s.add_argument("--columns", nargs="+", required=True)
    s.add_argument("--out", type=Path, required=True)

    st = sub.add_parser("stats", help="per-condition log1p means in two seeded halves, one pass over the shards")
    st.add_argument("--cache-dir", type=Path, required=True, help="train cache root (manifest.json + shard_*/)")
    st.add_argument("--columns", nargs="+", required=True, help="the condition columns, e.g. the eval cache's")
    st.add_argument("--seed", type=int, default=0, help="seed of the per-cell half coin")
    st.add_argument("--n-workers", type=int, default=4, help="shards read at once")

    args, extra = p.parse_known_args()
    if extra and args.cmd != "train":
        p.error(f"unrecognised arguments: {extra}")
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s", datefmt="%H:%M:%S")
    try:
        mp.set_start_method("fork", force=False)
    except RuntimeError:
        pass
    if args.cmd == "train":
        cmd_train(args, extra)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "stats":
        cmd_stats(args)
    else:
        cmd_summary(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
