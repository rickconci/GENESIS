"""Draw the 20 core figures from core_deliverables/data/ and check each against the committed PNG.

    python -m scripts.figures.draw_all              # draw into a temporary folder, compare, keep nothing
    python -m scripts.figures.draw_all --out DIR    # keep the drawn PNGs and PDFs in DIR

CPU only, about a minute with 6 jobs. Exits non-zero if a draw fails, a PNG is missing or extra, or any pixel
differs. Pixels, not bytes: PNG metadata may differ by platform, the image may not.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.figures import CORE

DRAWS = ("figure1.draw.main.draw", "figure1.draw.ed_genesis_settings.draw",
         "figure2.draw.main.draw", "figure2.draw.ed_conditioning.draw", "figure2.draw.ed_objective.draw", "figure2.draw.ed_metrics.draw",
         "figure2.draw.ed_pbmc_wells.draw", "figure2.draw.ed_pert_generalization.draw", "figure2.draw.ed_split_coverage.draw",
         "figure3.draw.main.draw", "figure3.draw.ed_12cells_per_condition.draw", "figure3.draw.ed_scaling_pert_split.draw",
         "figure4.draw.main.draw", "figure4.draw.ed_program_summary.draw", "figure4.draw.ed_tahoe_hallmark_gsea.draw",
         "figure4.draw.ed_celllines_per_model.draw")


def draw(module: str, out: Path) -> str | None:
    """One draw in its own process, so no figure inherits another's matplotlib state. The error, or None."""
    code = f"from pathlib import Path; from scripts.figures.{module} import main; main(out=Path({str(out)!r}))"
    run = subprocess.run([sys.executable, "-c", code], cwd=CORE.parent, capture_output=True, text=True)
    return None if run.returncode == 0 else run.stderr.strip().splitlines()[-1]


def same_pixels(a: Path, b: Path) -> bool:
    x, y = (np.asarray(Image.open(p).convert("RGBA")) for p in (a, b))
    return x.shape == y.shape and np.array_equal(x, y)


def check(out: Path) -> list[str]:
    """Every committed PNG must be drawn, pixel for pixel, and nothing else."""
    want = {p.relative_to(CORE) for p in CORE.glob("figure*/**/*.png")}
    got = {p.relative_to(out) for p in out.glob("figure*/**/*.png")}
    bad = [f"missing  {p}" for p in sorted(want - got)] + [f"extra    {p}" for p in sorted(got - want)]
    bad += [f"differs  {p}" for p in sorted(want & got) if not same_pixels(CORE / p, out / p)]
    print(f"{len(want & got) - sum(b.startswith('differs') for b in bad)} of {len(want)} PNGs pixel-identical")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="keep the figures here (default: a temporary folder)")
    ap.add_argument("--jobs", type=int, default=6, help="draws run at once")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        out = (args.out or Path(tmp)).resolve()
        with ThreadPoolExecutor(args.jobs) as pool:
            errors = dict(zip(DRAWS, pool.map(lambda m: draw(m, out / m.split(".")[0]), DRAWS)))
        for m, e in errors.items():
            print(f"{'FAILED' if e else 'drawn':8s} {m}" + (f": {e}" if e else ""))
        bad = check(out)
        for b in bad:
            print(b)
    if bad or any(errors.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
