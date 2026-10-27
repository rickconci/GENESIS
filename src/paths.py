"""Where files live: machine paths, results folders, and split files that moved.

Machine-specific locations are written in two files only; everything else names them by template:
  {data:<dataset>}  that dataset's data_dir in core_util_data/datasets/<dataset>.yaml;
  {root:<name>}     that entry of configs/roots.yaml (model outputs made outside this repository).
Two environment variables move them to another machine: PERTGEN_ROOTS names a roots file that replaces
configs/roots.yaml, and PERTGEN_DATA_ROOT (read by the dataset registry) puts every dataset at
<PERTGEN_DATA_ROOT>/<dataset>.
"""
import os
import re
import sys
from pathlib import Path

import yaml

PERTGEN_REPO_ROOT = Path(__file__).resolve().parent.parent      # the parent of src/


# Machine paths: the {data:<dataset>} and {root:<name>} templates

ROOTS_YAML = PERTGEN_REPO_ROOT / "configs" / "roots.yaml"
_PATH_TOKEN = re.compile(r"\{(data|root):([A-Za-z0-9_]+)\}")


def expand_path(template: str) -> str:
    """Fill in every {data:<dataset>} and {root:<name>} of ``template``; the rest is kept as written."""
    def fill(m: re.Match) -> str:
        kind, name = m.groups()
        return _data_dir(name) if kind == "data" else _root(name)
    return _PATH_TOKEN.sub(fill, str(template))


def _data_dir(dataset: str) -> str:
    # Imported here, not at the top: the registry imports this module to expand its own {root:...}.
    from src.config.dataset_registry import load_dataset_registry
    base = load_dataset_registry(dataset).data_dir
    if not base:
        raise KeyError(f"dataset {dataset!r} has no data_dir in its registry YAML")
    return str(base).rstrip("/")


def _root(name: str) -> str:
    # PERTGEN_ROOTS replaces (never merges with) configs/roots.yaml, so a reader's file is the whole truth.
    src = Path(os.environ.get("PERTGEN_ROOTS") or ROOTS_YAML)
    if not src.is_file():
        raise FileNotFoundError(f"roots file not found: {src}")
    roots = yaml.safe_load(src.read_text()) or {}
    if name not in roots:
        raise KeyError(f"no root {name!r} in {src}")
    return str(roots[name]).rstrip("/")


# Results folders: results/<dataset>/split_half/{pertgen,noise_ceiling}/

def dataset_eval_results_root(dataset: str | None) -> Path:
    """``results/<dataset>/split_half/``; split-half is the only eval method, and no dataset means tahoe."""
    return (PERTGEN_REPO_ROOT / "results" / (dataset or "tahoe").strip() / "split_half").resolve()


def pertgen_model_eval_results_root(dataset: str | None) -> Path:
    """``results/<dataset>/split_half/pertgen/``: the model's scores."""
    return dataset_eval_results_root(dataset) / "pertgen"


def noise_ceiling_results_root(dataset: str | None) -> Path:
    """``results/<dataset>/split_half/noise_ceiling/``: the split-half reference."""
    return dataset_eval_results_root(dataset) / "noise_ceiling"


# Split files that moved
# Run manifests, configs and ceiling JSONs record a split's path as it was when they were written.
# The move table maps such a path to the same file's address now.

_SPLIT_MOVES_TSV = PERTGEN_REPO_ROOT / "core_util_data/splits/moved_splits.tsv"


def _split_moves() -> dict[str, str]:
    """{old absolute path: new absolute path} from the move table (its rows are repo-relative).

    The table ships with the repository; without it no recorded path can be followed, so a missing
    table raises rather than reading as "nothing moved".
    """
    if not _SPLIT_MOVES_TSV.is_file():
        raise FileNotFoundError(f"split move table not found: {_SPLIT_MOVES_TSV}")
    moves: dict[str, str] = {}
    for line in _SPLIT_MOVES_TSV.read_text().splitlines():
        if line and not line.startswith("#"):
            old, new = line.split("\t")
            moves[f"{PERTGEN_REPO_ROOT}/{old}"] = f"{PERTGEN_REPO_ROOT}/{new}"
    return moves


def relocated_split_path(path: str) -> str:
    """Where a recorded split path is now: unchanged if it exists or never moved."""
    if not path or os.path.exists(path):
        return path
    # Longest first, so a file's own row wins over a row for the folder it sits in.
    for old, new in sorted(_split_moves().items(), key=lambda kv: -len(kv[0])):
        if path == old or path.startswith(old + "/"):
            return new + path[len(old):]
    return path


if __name__ == "__main__":  # shell scripts: EVAL=$(python -m src.paths '{data:tahoe}/eval_cache/...')
    print(expand_path(sys.argv[1]))
