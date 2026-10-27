#!/usr/bin/env python3
"""Generate one split family from ``core_util_data/splits/dataset_split_policies.yaml``.

A family is one published split directory (e.g. ``tahoe_overlap_bio``); its policy
entry names the scheme (scaling | overlap_bio | overlap_pert) and every constant.
The shipped YAMLs are canonical -- they are hashed into eval-cache keys -- so this
refuses to write into a family's own directory unless ``--overwrite`` is given.
``--check`` regenerates into a temp dir and compares against the shipped files:
"same" is byte-identical below the header, "same*" parses to identical data.

  python scripts/data/make_splits.py --dataset tahoe --family tahoe_overlap_bio --check
  python scripts/data/make_splits.py --dataset parse --family parse_overlap_pert_fixed --out-dir /tmp/s
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.dataset_registry import load_dataset_registry  # noqa: E402
from src.data.splits.generation import (  # noqa: E402
    REPO_ROOT, load_condition_summary, load_split_policy, write_manifest, write_split_yaml,
)
from src.data.splits.schemes import overlap_bio_splits, overlap_pert_splits, scaling_splits  # noqa: E402


def generate(dataset: str, name: str):
    policy, registry = load_split_policy(dataset), load_dataset_registry(dataset)
    if name not in policy.families:
        raise SystemExit(f"no family {name!r} for {dataset}; have {sorted(policy.families)}")
    family = policy.families[name]
    summary = load_condition_summary(policy, registry)
    scheme = family["scheme"]
    if scheme == "overlap_pert":
        return family, overlap_pert_splits(summary, policy, registry, family, name)
    base = yaml.safe_load((REPO_ROOT / family["base_split"]).read_text(encoding="utf-8"))
    if scheme == "scaling":
        return family, scaling_splits(summary, policy, registry, family, base)
    if scheme == "overlap_bio":
        return family, overlap_bio_splits(summary, policy, registry, family, base, name)
    raise SystemExit(f"{name}: unknown scheme {scheme!r}")


def body(path: Path) -> str:
    """A split YAML minus its comment header: what the split engine reads."""
    return "".join(line for line in path.read_text(encoding="utf-8").splitlines(True) if not line.startswith("#"))


def write(splits, out_dir: Path) -> None:
    for s in splits:
        write_split_yaml(out_dir / s.rel_path, s.payload, header=s.header)
    write_manifest(out_dir / "manifest.json", [s.manifest for s in splits])


def check(splits, shipped_dir: Path) -> int:
    """Diff regenerated bodies against the shipped YAMLs. Returns the mismatch count."""
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        write(splits, Path(tmp))
        for s in splits:
            ours, theirs = Path(tmp) / s.rel_path, shipped_dir / s.rel_path
            if not theirs.is_file():
                status = "MISSING"
            elif body(ours) == body(theirs):
                status = "same"
            elif yaml.safe_load(body(ours)) == yaml.safe_load(body(theirs)):
                status = "same*"  # identical data, different YAML formatting (e.g. unicode escapes)
            else:
                status = "DIFF"
            bad += status not in ("same", "same*")
            print(f"{status:8s}{s.rel_path}")
    return bad


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--family", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="diff against the shipped YAMLs; write nothing")
    mode.add_argument("--out-dir", type=Path, help="write the family here")
    p.add_argument("--overwrite", action="store_true", help="allow --out-dir to be the shipped directory")
    args = p.parse_args()

    family, splits = generate(args.dataset, args.family)
    shipped = (REPO_ROOT / family["out_dir"]).resolve()
    if args.check:
        bad = check(splits, shipped)
        print(f"{len(splits) - bad}/{len(splits)} identical to {family['out_dir']}")
        return 1 if bad else 0
    if args.out_dir.resolve() == shipped and not args.overwrite:
        raise SystemExit(f"{shipped} holds the canonical splits; pass --overwrite to replace them")
    write(splits, args.out_dir)
    print(f"wrote {len(splits)} splits + manifest.json under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
