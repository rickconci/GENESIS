"""One palette for every arm-comparison figure (metric bars, condition curves).

Foundation models get the four vivid hues; non-learned baselines stay gray so the
models own the eye, with a hatch as the second channel for grayscale print.
"""
from __future__ import annotations

COL = {
    "GENESIS":    "#2a78d6",   # blue (our model)
    "GENESIS_nocell": "#8fb8e8",  # same family, no cell-type input
    # Ablation siblings: same blue family, darker = unnorm (log1p), lighter = norm.
    "GENESIS_unnorm": "#2a78d6",
    "GENESIS_norm":   "#9bb8de",
    "GENESIS_slicemed": "#9bb8de",  # trained in log1p_slice_med
    # Input-covariate ablation: darker = more labels in the generator.
    "CIA_all": "#1b4f8a",
    "CIA_bio": "#2a78d6",
    "CIA_pert": "#7aabdf",
    "CIA_plate": "#c5daf2",
    # n_repeats sweep: darker = more noise samples.
    "NRS4": "#d6e4f5",
    "NRS8": "#a9c7ea",
    "NRS16": "#6aa3e0",
    "NRS32": "#2a78d6",
    "NRS256": "#1b4f8a",
    "NRS1024": "#0e2f55",
    "PerturbDiff": "#eb6834",  # orange
    "STATE":      "#1baf7a",   # aqua
    "CellFlow":   "#4a3aa7",   # violet
    # Four grays from charcoal to near-paper; a narrower ramp reads as one gray.
    "Additive":   "#1f1e1c",
    "LinRidge":   "#6a665c",
    "PertMean":   "#b7b2a6",
    "CtxMean":    "#f0ebe0",
    # Copy-paste of the closest neighbour: brown, not one of the linear baselines.
    "copy_closest": "#8a5a2b",
}
HATCH = {
    "Additive": "///",
    "LinRidge": "xx",
    "PertMean": "...",
    "CtxMean":  "||",
    "copy_closest": "++",
}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
# The split-half line is the reference the eye should find first: dirty rose, not a gray.
SPLIT_HALF_C = "#B56B78"
BANNER_C = "#F2F2F0"


def hatch_ink(fill: str) -> str:
    """Light hatch on a dark fill, dark hatch on a light one. Same-ink hatch disappears."""
    h = fill.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#f7f4ee" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 150 else "#1c1b19"
