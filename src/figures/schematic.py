"""The studio's primitives: what Figure 1's schematics (scripts/figures/figure1/) and Figure 2's ED metrics page share.

The schematics were ported from an in-house layout tool, "the studio". Its colours and its cell-membrane maths
are ported bit-for-bit, so every membrane comes out the shape the studio drew. One drawing unit is one point.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from matplotlib.figure import Figure
from matplotlib.path import Path

MM = 72.0 / 25.4                      # mm -> pt

PALETTE = {                           # the studio's shared.colors
    "amber": "#d99a2b", "blue": "#3a6dbf", "blueSoft": "#a9c6f0",
    "brand": "#2456a6", "charcoal": "#52514e", "dustyRose": "#9c4d63",
    "green": "#3f9b52", "greenDark": "#0f6b3c", "grey": "#a3a8ab",
    "ink": "#1b1d1f", "inkSoft": "#3a3f42", "magenta": "#8d3a72",
    "magentaSoft": "#b96ea0", "orange": "#e3873c", "panel": "#ececee",
    "purple": "#6b4fbb", "red": "#d8453f", "teal": "#2e7c7f",
    "tealPale": "#cfe3e4", "tealSoft": "#9cc6c7",
}


def C(k: str) -> str:
    """A palette name, or any matplotlib colour passed through."""
    return PALETTE.get(k, k)


def bl(size: float) -> float:
    """Text centre -> baseline offset: the studio places text by its baseline."""
    return size * 0.355


def text_width(fig: Figure, s: str, size: float, weight: int = 400, **kw) -> float:
    """Advance width in pt of `s` as this process sets it. The studio used the browser's metrics; the
    renderer's own stand in, so layouts stay centred in whichever font is actually installed."""
    if not s:
        return 0.0
    t = fig.text(0, 0, s, fontsize=size, fontweight=("bold" if weight >= 600 else "normal"), **kw)
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    return bb.width * 72.0 / fig.dpi


# Cell membranes: a closed spline through a wobbly circle, seeded.

def rng(seed: int) -> Callable[[], float]:
    """The studio's xorshift32, ported bit-for-bit with uint32 wrapping: the cell
    membranes and the mRNA spike heights are this generator's output, so any
    deviation changes the drawing."""
    s = (int(seed) & 0xFFFFFFFF) or 2463534242

    def nxt() -> float:
        nonlocal s
        s ^= (s << 13) & 0xFFFFFFFF
        s &= 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        s &= 0xFFFFFFFF
        return s / 4294967296.0
    return nxt


def wobble_radius(r: Callable[[], float], lobes: float, amp: float) -> Callable[[float], float]:
    """1 + a few harmonics at random phase, so a boundary undulates instead of
    being a clean circle. The r() call order is load-bearing."""
    L = max(1, min(6, int(round(lobes))))
    terms = [(i + 2, amp / (i + 1), r() * np.pi * 2) for i in range(L)]

    def R(th: float) -> float:
        v = 1.0
        for k, a, ph in terms:
            v += a * np.cos(k * th + ph)
        return max(0.25, v)
    return R


def catmull_closed(pts: list, rnd: Callable[[float], float] | None = None) -> tuple[list, list]:
    """Closed Catmull-Rom through pts as cubic Beziers, exactly as the studio's
    catmullClosed() (tangents = (p2 - p0) / 6). Returns (verts, codes). `rnd`
    rounds every emitted coordinate; the energy panel passes the studio's 2-dp nf."""
    q = (lambda p: p) if rnd is None else (lambda p: (rnd(p[0]), rnd(p[1])))
    n = len(pts)
    verts, codes = [q(pts[0])], [Path.MOVETO]
    for i in range(n):
        p0, p1 = pts[(i - 1) % n], pts[i]
        p2, p3 = pts[(i + 1) % n], pts[(i + 2) % n]
        a = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        b = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        verts += [q(a), q(b), q(p2)]
        codes += [Path.CURVE4] * 3
    verts.append(q(pts[0]))
    codes.append(Path.CLOSEPOLY)
    return verts, codes


def blob_path(seed: int, r: float, wob: float, lobes: float, n: float,
              rnd: Callable[[float], float] | None = None) -> tuple[list, list]:
    """One membrane of radius r centred on the origin, as (verts, codes)."""
    R = wobble_radius(rng(seed), lobes, wob)
    N = max(8, min(48, int(round(n))))
    pts = []
    for i in range(N):
        th = (i / N) * np.pi * 2
        rad = r * R(th)
        pts.append((np.cos(th) * rad, np.sin(th) * rad))
    return catmull_closed(pts, rnd)
