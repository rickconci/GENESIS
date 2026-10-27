#!/usr/bin/env python3
"""Persist a GENESIS checkpoint's generated cells, for arms scored outside the streaming eval.

  replicates  one .npy per (condition, replicate) of a scored pair table, at that real
              replicate's size (n_truth_i / n_truth_j); each file's noise is one draw of its own
              seed (src/eval/scoring.py:LiveClouds, the eval's generator), so files are
              reproducible alone and a killed run resumes by skipping what exists. Alone, but
              on the same GPU model: CUDA lays its random stream out by the device's launch grid.
              --pseudobulk (deterministic MSE checkpoints only) writes one (1, G) log1p
              row per file instead: the model's pseudobulk, scored with --fmt pseudobulk.
              --conditions + --eval-cache instead of --counts-from: any split's conditions (e.g.
              training repeats no pair table covers), sizes read from the truth replicate_pairs scores.
  conditions  one .npz per test condition of the run's split, at the measured cell count
              (fields: predictions, gene_names, normalization, one per condition column),
              e.g. the donor x cytokine arm that src/eval/cell_classifier.py splits by cell type.
              One seeded stream, consumed in output-filename order: the published clouds
              were drawn that way, so only the full ordered run reproduces them.

  python scripts/eval/generate.py replicates --dataset tahoe --run-dir <run> \\
      --counts-from model_vs_batch_gap__genesis.parquet --out-pred <dir>
  python scripts/eval/generate.py replicates --dataset tahoe --run-dir <run> \\
      --conditions conds.tsv --eval-cache <ns> --out-pred <dir>
  python scripts/eval/generate.py conditions --run-dir <run> --checkpoint last --out-dir <dir>
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.reference.pairs import (  # noqa: E402
    EvalNpzReader, prediction_path, read_perturbed, perturbed_keys,
)
from src.eval.reference.generation import (  # noqa: E402
    atomic_save, generate_condition, replicate_cell_counts, replicate_k_seed,
)
from src.eval.reference.replicate import _replicate_label, enumerate_pairs, replicate_design  # noqa: E402
from src.eval.reference.designs import CloudRequest  # noqa: E402
from src.eval.scoring import LiveClouds  # noqa: E402
from src.paths import expand_path  # noqa: E402

from src.model.pertgen import is_deterministic  # noqa: E402

log = logging.getLogger("generate")


def _bundle(run_dir: Path, checkpoint, device: str):
    from src.model.inference_output import PertGenInferenceBundle
    bundle = PertGenInferenceBundle.from_run(run_dir, checkpoint=checkpoint, device=device, use_compile=False)
    log.info("bundle loaded | genes=%d device=%s space=%s", len(bundle.gene_names), bundle.device, bundle.output_space)
    return bundle


def truth_counts(d, conditions: Path, eval_cache: str) -> dict[tuple, int]:
    """(condition values..., replicate) -> real cell count, read from the same truth replicate_pairs
    scores against, so each file lands on exactly that pair's n_truth."""
    reader = EvalNpzReader(Path(expand_path(eval_cache)).resolve(), cache_size=1)
    out: dict[tuple, int] = {}
    for cond, i, j in enumerate_pairs(d, pd.read_csv(conditions, sep="\t"), "plates"):
        for r in (i, j):
            if (key := (*(cond[c] for c in d.condition_cols), r)) not in out:
                out[key] = int(read_perturbed(reader, perturbed_keys(d, cond, r), d.replicate_col).shape[0])
    return out


def cmd_replicates(a: argparse.Namespace) -> None:
    """One .npy per (condition, replicate) at that real replicate's size; a file that exists is skipped."""
    ckpt = a.checkpoint or (a.run_dir / "checkpoints" / "last.ckpt")
    if (a.counts_from is None) == (a.conditions is None) or (a.conditions is None) != (a.eval_cache is None):
        raise SystemExit("sizes from --counts-from (a scored pair table) OR --conditions + --eval-cache")
    for f in (Path(ckpt), a.counts_from or a.conditions):
        if not f.is_file():
            raise FileNotFoundError(f)
    d = replicate_design(a.dataset)
    counts = (replicate_cell_counts(d, pd.read_parquet(a.counts_from)) if a.counts_from
              else truth_counts(d, a.conditions, a.eval_cache))
    jobs = sorted(counts.items())[: a.limit]
    a.out_pred.mkdir(parents=True, exist_ok=True)
    log.info("%d (condition, %s) files | draw=%d | out=%s", len(jobs), d.replicate_col, a.draw, a.out_pred)
    todo: list[tuple[Path, CloudRequest]] = []
    for key, n_cells in jobs:
        cond, rep = dict(zip(d.condition_cols, key[:-1])), key[-1]
        dest = prediction_path(a.out_pred, d, cond, rep, "npy")
        if not dest.is_file():  # a resumed run skips what exists, and never touches the GPU if nothing is left
            row = {**cond, d.replicate_col: _replicate_label(d, rep)}
            todo.append((dest, CloudRequest({c: row[c] for c in d.file_cols}, 1 if a.pseudobulk else n_cells,
                                            a.draw, replicate_k_seed(d, cond, rep, a.draw))))
    n_skip, n_cells_total, t0 = len(jobs) - len(todo), 0, time.time()
    if todo:
        bundle = _bundle(a.run_dir, Path(ckpt), a.device)
        if a.pseudobulk and not is_deterministic(bundle.config):
            raise SystemExit("--pseudobulk needs a deterministic (train_loss=mse) checkpoint: "
                             "a generative model's one draw is not its pseudobulk")
        if a.pseudobulk and a.draw != 0:
            raise SystemExit("--pseudobulk is one deterministic row; it has no other draws")
        source = LiveClouds(bundle, batch_size=a.batch_size, label="Generate replicates")
        source.open([req for _, req in todo])
        try:
            for k, (dest, req) in enumerate(todo, start=1):
                cells = source.fulfil([req])[req.key]
                with source.scoring_stream():  # waits for the cloud's generation before the copy
                    arr = cells.cpu().numpy()
                if float(np.nanmax(arr)) > 30.0:
                    raise ValueError(f"{dest.name}: max={float(np.nanmax(arr)):.2f} looks like raw counts")
                atomic_save(dest, arr)
                n_cells_total += req.n_cells
                if k % 25 == 0 or k == len(todo):
                    log.info("%d/%d | skip=%d cells=%d | %.0fs", k, len(todo), n_skip, n_cells_total, time.time() - t0)
        finally:
            source.close()
    log.info("done | skip=%d gen=%d cells=%d | %.0fs | out=%s", n_skip, len(todo), n_cells_total, time.time() - t0,
             a.out_pred)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def cmd_conditions(a: argparse.Namespace) -> None:
    """One .npz per --eval-split condition of the run's split, at its measured cell count, from one seeded stream."""
    from src.config import DatasetConditionSpec, load_config
    from src.config.config import resolve_condition_input_variables
    from src.data.data_utils import load_eval_manifest
    from src.eval.cache.eval_cache_io import load_work_queue

    config = load_config(config_path=a.run_dir / "config_resolved.yaml")
    manifest = load_eval_manifest(a.run_dir)             # heals paths of moved runs
    cols = tuple(manifest.group_cols)
    queue = load_work_queue(condition_summary_parquet=manifest.condition_summary,
                            split_yaml=manifest.split_config, eval_split=a.eval_split, condition_variables=cols,
                            control_condition=dict(DatasetConditionSpec.from_alias(config.dataset).control_condition),
                            min_cells=a.min_cells)
    jobs = sorted(("__".join(f"{c}__{_safe(str(r.condition_keys[c]))}" for c in cols) + ".npz",
                   {c: str(r.condition_keys[c]) for c in cols}, int(r.num_cells)) for r in queue)
    if len({j[0] for j in jobs}) != len(jobs):
        raise SystemExit("two conditions map to one file name; the queue has duplicate condition keys")
    log.info("%d %s conditions over %s%s | out=%s", len(jobs), a.eval_split, list(cols),
             f" (first {a.limit} only)" if a.limit else "", a.out_dir)
    jobs = jobs[: a.limit]
    bundle = _bundle(a.run_dir, a.checkpoint or "last", a.device)
    # A run may drop condition columns from the generator input (condition_input_variables): those columns
    # still name the files, but the model never sees them, so clouds differing only there are one distribution.
    inputs = set(bundle.encoder_idx_to_name)
    if inputs != set(resolve_condition_input_variables(config)) or not inputs <= set(cols):
        raise SystemExit(f"encoder conditions {sorted(inputs)} != the run's inputs "
                         f"{resolve_condition_input_variables(config)} within columns {cols}")
    if inputs != set(cols):
        log.warning("generator input omits %s: those columns name the files only", sorted(set(cols) - inputs))
    gen = torch.Generator(device=bundle.device).manual_seed(int(a.seed))
    names = np.array(list(bundle.gene_names), dtype=object)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for i, (fname, keys, n_cells) in enumerate(jobs, start=1):
        pred = generate_condition(bundle, {c: v for c, v in keys.items() if c in inputs}, n_cells=n_cells,
                                  batch_size=a.batch_size, generator=gen)
        np.savez(a.out_dir / fname, predictions=np.ascontiguousarray(pred, dtype=np.float32), gene_names=names,
                 normalization=np.array(bundle.output_space.value), **{c: np.array(v) for c, v in keys.items()})
        if i == 1 or i % 25 == 0 or i == len(jobs):
            log.info("%d/%d %s n=%d", i, len(jobs), keys, n_cells)
    log.info("wrote %d clouds -> %s", len(jobs), a.out_dir)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--run-dir", type=Path, required=True)
    shared.add_argument("--device", default="cuda")
    shared.add_argument("--batch-size", type=int, default=4096)
    shared.add_argument("--limit", type=int, default=None, help="first N files only (smoke test)")
    r = sub.add_parser("replicates", parents=[shared])
    r.add_argument("--dataset", required=True)
    r.add_argument("--checkpoint", type=Path, default=None, help="default: <run-dir>/checkpoints/last.ckpt")
    r.add_argument("--counts-from", type=Path, default=None, help="scored pair table with n_truth_i/j")
    r.add_argument("--conditions", type=Path, default=None,
                   help="replicate_pairs' TSV (condition columns + plates, e.g. 6,14); sizes from --eval-cache")
    r.add_argument("--eval-cache", default=None, help="truth namespace for --conditions; {data:<dataset>} allowed")
    r.add_argument("--out-pred", type=Path, required=True)
    r.add_argument("--draw", type=int, default=0,
                   help="which seeded draw (0 = replicate_seed, what the generated arms were drawn with); "
                        "K draws = K dirs, each scored, metrics averaged")
    r.add_argument("--pseudobulk", action="store_true",
                   help="one (1, G) log1p row per file: an MSE model's E[log1p x], not cells")
    c = sub.add_parser("conditions", parents=[shared])
    c.add_argument("--checkpoint", default=None, help="'last' (default) or a checkpoint path")
    c.add_argument("--eval-split", default="test", choices=("test", "validation"))
    c.add_argument("--min-cells", type=int, default=4)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    if a.cmd == "replicates" and a.draw < 0:
        raise SystemExit("--draw must be >= 0")
    (cmd_replicates if a.cmd == "replicates" else cmd_conditions)(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
