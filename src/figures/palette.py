"""One palette (Figure 2's, shared by its Extended Data twins).

The four deep models keep the published hues (src/eval/plots/style.py); the fixed-rule baselines carry no
hue at all, so colour alone says "learned model". A grey ramp (white to black) plus a hatch names each
baseline's bar, and the same grey plus a dash names its curve.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

# label -> (fill, hatch, curve dash)
ARMS = {
    "GENESIS":      ("#2a78d6", "", "solid"),
    "STATE":        ("#1baf7a", "", "solid"),
    "PerturbDiff":  ("#eb6834", "", "solid"),
    "CellFlow":     ("#4a3aa7", "", "solid"),
    "PertMean":     ("#d4d4d4", "....", (0, (1, 1))),
    "LinRidge":     ("#5a5a5a", "xxx", (0, (3, 1, 1, 1))),
    "Additive":     ("#1a1a1a", "///", (0, (5, 1.6))),
    "CtxMean":      ("#ffffff", "||", (0, (2, 1))),
    "Nearest-neighbor": ("#9e9e9e", "++", (0, (6, 1.5, 1, 1.5, 1, 1.5))),
}
FILL = {arm: fill for arm, (fill, _hatch, _dash) in ARMS.items()}
C_REPEAT, C_SPLIT = "#1f5f68", "#B56B78"     # experimental repeat teal, split-half rose
REF_DASH = (0, (4, 2.5))                      # both references are dashed lines
INK, MUTED, GRID, BANNER = "#1b1d1f", "#898781", "#e5e4de", "#f2f2f0"


def hatch_ink(fill: str) -> str:
    """Light hatch on a dark fill, dark on a light one; same-ink hatch disappears."""
    r, g, b = (int(fill.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return "#f7f4ee" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 150 else "#1c1b19"


def curve_color(fill: str) -> str:
    """A baseline's line grey: its bar's grey squeezed into the dark half, so a white bar
    is still a visible line, while the ramp keeps the bars' light-to-dark order."""
    g = 0.06 + 0.55 * float(np.dot(mpl.colors.to_rgb(fill), [0.2126, 0.7152, 0.0722]))
    return mpl.colors.to_hex((g, g, g))
