"""Figure 1b: the energy score's attraction and repulsion, a schematic (no data). figure1.py calls
draw(ax) on an axes whose units are points with y running down, as in the studio (the in-house layout
tool) it was ported from.
"""
import numpy as np
import matplotlib
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Polygon

from src.figures.schematic import C, bl, text_width
from src.figures.schematic import blob_path as _blob

# shared.type.scale and shared.line.scale are both 1, so the studio's TS()/LW()
# wrappers are the identity here and every number passes straight through.
P = {
        # 63 mm, not the studio's 58: the key's '(Ŷ)' is drawn past 58 mm once
        # the label is set in DejaVu rather than Helvetica.
        "canvas": {"wmm": 63, "hmm": 38, "bg": "#ffffff"},

    # The cell field. x/y place the field; w/h are the field box the item
    # fractions are taken against, so cells move with the box, not in absolutes.
    "cells": {
        "x": 9.5, "y": 9.76, "w": 79, "h": 68,
        "r": 9, "fill": 0.18, "stroke": 0.65, "nucleus": 0.34,
        "nucOpacity": 0.85, "wobble": 0.085, "lobes": 3, "points": 20,
        "genColor": "blue", "genDashed": False,
        "trueColor": "red", "trueDashed": False, "trueR": 9, "trueSeed": 601,
        "truePos": {"x": 0.5, "y": 0.5, "nx": 0.06, "ny": -0.12},
        # The true population, faded and dashed at the very back: y is one
        # SAMPLE from it, which is why the loss also needs the repulsion term.
        # dx/dy are pt from the true cell's centre, spread over the whole field.
        "pop": {"fade": 0.35, "dash": [1.6, 1.0], "rMul": 0.75,
                "items": [{"dx": 0.5, "dy": -7.5, "nx": 0.05, "ny": -0.25, "seed": 733},
                          {"dx": -1.5, "dy": 8.5, "nx": -0.05, "ny": 0.25, "seed": 118},
                          {"dx": -24, "dy": -3, "nx": -0.1, "ny": 0.1, "seed": 241},
                          {"dx": 25, "dy": 5, "nx": 0.1, "ny": -0.1, "seed": 377},
                          {"dx": -13, "dy": 23, "nx": 0.08, "ny": 0.15, "seed": 519},
                          {"dx": 14, "dy": -22, "nx": -0.12, "ny": 0.05, "seed": 862},
                          {"dx": 15, "dy": 21, "nx": 0.1, "ny": 0.1, "seed": 954},
                          {"dx": -17, "dy": -20, "nx": -0.05, "ny": -0.1, "seed": 1093}]},
        "items": [
            {"x": 0.21, "y": 0.14, "nx": -0.14, "ny": -0.10, "rMul": 1,
             "fillMul": 1.15, "seed": 37},
            {"x": 0.94, "y": 0.06, "nx": 0.10, "ny": -0.14, "rMul": 1,
             "fillMul": 1.00, "seed": 214},
            {"x": 0.86, "y": 0.93, "nx": -0.08, "ny": 0.12, "rMul": 1,
             "fillMul": 0.70, "seed": 455},
            {"x": 0.07, "y": 0.77, "nx": 0.13, "ny": 0.06, "rMul": 1,
             "fillMul": 1.30, "seed": 602},
        ],
    },

    # Arrows are measured off the cells, never placed by hand: `gap` is the
    # clearance left outside each membrane, `reach` the fraction of the
    # remaining clear span the arrow fills, `place` where its middle sits.
    "forces": {
        "attract": {"color": "green", "from": [0, 1, 2, 3],
                    "gap": 2, "head": 2.4, "place": 0.5, "reach": 0.62,
                    "stroke": 0.55},
        "disperse": {"color": "amber",
                     "pairs": [[0, 1], [1, 2], [2, 3], [3, 0]],
                     "gap": 2.5, "head": 2.4, "offset": 0, "place": 0.5,
                     "reach": 0.8, "stroke": 0.55},
    },

    "key": {
        "x": 93, "y": 27.78, "glyph": 16, "glyphR": 6, "gap": 5, "rowGap": 12,
        "arrowLen": 12, "head": 2.2, "size": 5.6, "color": "ink",
        "entries": [
            {"kind": "gen", "sym": r"$\hat{Y}$", "text": "Generated cells"},
            {"kind": "pop", "text": "True cells"},
            {"kind": "true", "sym": r"$y$", "text": "Sampled true cell"},
            {"kind": "attract", "text": "Attraction"},
            {"kind": "disperse", "text": "Repulsion"},
        ],
    },

    # L = E‖Ŷ − y‖ − ½ E‖Ŷ − Ŷ′‖, Ŷ′ an independent draw (Gneiting & Raftery's
    # form). No i≠j subscripts: mathtext sets them at 70%, under the 5 pt floor.
    # \dfrac, not \frac or ½: \frac shrinks the digits the same way, and the
    # cm font has no ½ glyph (mathtext silently draws ρ). Each term is named
    # beneath it in its arrow colour.
    "eloss": {
        "x": 82.09, "y": 93.12, "size": 6.5, "color": "ink",
        "lhs": r"\mathcal{L}=",
        "t1": r"\mathbb{E}\,\Vert\hat{Y}-y\Vert",
        "t2": r"\dfrac{1}{2}\,\mathbb{E}\,\Vert\hat{Y}-\hat{Y}^{\prime}\Vert",
        "labelGap": 9, "labelSize": 5.6,
        "t1label": "Attraction", "t2label": "Repulsion",
    },
}


# ──────────────────────────── primitives ────────────────────────────

def nf(v, p=2):
    """The studio rounds every emitted coordinate to 2 dp (3 for stroke
    widths); keeping that keeps the geometry bit-comparable to its SVG."""
    m = 10.0 ** p
    return (np.floor(v * m + 0.5) if v >= 0 else -np.floor(-v * m + 0.5)) / m


_Z = [10]                                   # monotonic z, so draw order == call order


def _z():
    _Z[0] += 1
    return _Z[0]


def stroke(ax, pts, color, lw):
    """One polyline, stroked, never filled."""
    path = Path([(nf(p[0]), nf(p[1])) for p in pts], [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1))
    ax.add_patch(PathPatch(path, fill=False, edgecolor=color, linewidth=nf(lw, 3), joinstyle="miter",
                           capstyle="butt", zorder=_z()))


def fill_path(ax, verts_codes, color, opacity=1.0, lw=0.0, edge=None,
              join="miter", dashes=None):
    verts, codes = verts_codes
    face = matplotlib.colors.to_rgba(color, opacity)
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=face,
                           edgecolor=(edge if edge else "none"),
                           linewidth=nf(lw, 3), joinstyle=join,
                           linestyle=((0, dashes) if dashes else "solid"),
                           zorder=_z()))


# Maths in this panel only is set in Computer Modern; per-text, so panel d's
# tick labels in the composite keep the sans default.
MATH_FONT = "cm"


def text(ax, s, x, y, size, fill, anchor="start"):
    """Placed by BASELINE, so the studio's y values transfer unchanged."""
    ha = {"start": "left", "middle": "center", "end": "right"}[anchor]
    ax.text(nf(x), nf(y), s, fontsize=size, color=fill, ha=ha, va="baseline",
            style="normal", fontweight="normal", math_fontfamily=MATH_FONT, zorder=_z())


# ───────────────────────────── cell shape ─────────────────────────────

def blob_path(seed, r, wob, lobes, n):
    """The studio's membrane, every coordinate rounded as the studio emits it."""
    return _blob(seed, r, wob, lobes, n, rnd=nf)


def shift(verts_codes, dx, dy):
    verts, codes = verts_codes
    return [(nf(v[0] + dx), nf(v[1] + dy)) for v in verts], codes


def membrane_cell(ax, cx, cy, color, r, nucleus, stroke_w, dashed, fill,
                  seed, wobble, lobes, points, nx=0, ny=0, nucOpacity=0.85,
                  fillMul=1, fade=1.0):
    """One cell: a wobbly membrane rather than a circle, a solid nucleus set a
    little off centre, and a tinted body whose weight varies cell to cell.
    Geometry only, so it survives every export path. `dashed` is the list of
    dash lengths in pt (or falsy); `fade` scales every opacity in the cell."""
    col = C(color)
    f = fill * fillMul * fade
    fill_path(ax, shift(blob_path(seed, r, wobble, lobes, points), cx, cy),
              col, nf(f, 3), lw=stroke_w,
              edge=matplotlib.colors.to_rgba(col, nf(fade, 2)),
              join="round", dashes=(dashed or None))
    nop = nf(nucOpacity * fade, 2)
    fill_path(ax, shift(blob_path(seed + 91, r * nucleus, wobble * 0.75, 3, 14),
                        cx + nf(r * nx), cy + nf(r * ny)), col, nop)


# ───────────────────────────── arrows ─────────────────────────────

def head_poly(ax, x, y, ux, uy, head, color):
    px, py, w = -uy, ux, head * 0.46
    pts = [(nf(x), nf(y)),
           (nf(x - ux * head + px * w), nf(y - uy * head + py * w)),
           (nf(x - ux * head - px * w), nf(y - uy * head - py * w))]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none",
                         linewidth=0, zorder=_z()))


def arrow_nodes(ax, x1, y1, x2, y2, color, stroke_w, head, both):
    """A straight arrow between two points, single- or double-headed. The shaft
    stops short of each head so the two do not overprint."""
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    sx = x1 + ux * head * 0.75 if both else x1
    sy = y1 + uy * head * 0.75 if both else y1
    ex, ey = x2 - ux * head * 0.75, y2 - uy * head * 0.75
    stroke(ax, [(sx, sy), (ex, ey)], color, stroke_w)
    head_poly(ax, x2, y2, ux, uy, head, color)
    if both:
        head_poly(ax, x1, y1, -ux, -uy, head, color)


def span_arrow(x1, y1, x2, y2, reach, place):
    """An arrow laid inside the clear gap between two cells: `reach` sets how
    much of the gap it fills, `place` where its middle sits (0 against the
    first cell, 0.5 centred, 1 against the second). Both ends are clamped so
    the arrow can never run into either membrane."""
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    ux, uy = (dx / L, dy / L) if L else (1.0, 0.0)
    ln = L * max(0.05, min(1.0, 0.7 if reach is None else reach))
    half = ln / 2.0
    c = L * max(0.0, min(1.0, 0.5 if place is None else place))
    c = max(half, min(L - half, c))
    return (x1 + ux * (c - half), y1 + uy * (c - half),
            x1 + ux * (c + half), y1 + uy * (c + half))


def energy_geom():
    """Centres and radii in field coordinates: the arrows are derived from the
    cells rather than placed by hand, so they cannot drift out of register."""
    o = P["cells"]
    t = {"cx": o["truePos"]["x"] * o["w"], "cy": o["truePos"]["y"] * o["h"],
         "r": o["trueR"]}
    gens = [{"cx": it["x"] * o["w"], "cy": it["y"] * o["h"],
             "r": o["r"] * (1 if it.get("rMul") is None else it["rMul"]),
             "seed": it["seed"], "nx": it["nx"], "ny": it["ny"],
             "fillMul": it["fillMul"]} for it in o.get("items", [])]
    return o, t, gens


# ───────────────────────────── blocks ─────────────────────────────

def draw_cells(ax):
    o, t, gens = energy_geom()
    # population first, so every generated cell and arrow sits in front of it
    q = o["pop"]
    for c in q["items"]:
        membrane_cell(ax, o["x"] + t["cx"] + c["dx"], o["y"] + t["cy"] + c["dy"],
                      o["trueColor"], t["r"] * q["rMul"], o["nucleus"],
                      o["stroke"], q["dash"], o["fill"], c["seed"], o["wobble"],
                      o["lobes"], o["points"], c["nx"], c["ny"],
                      o["nucOpacity"], 1, q["fade"])
    for c in gens:
        # bodies are translucent: the underlay keeps the population behind them
        fill_path(ax, shift(blob_path(c["seed"], c["r"], o["wobble"], o["lobes"],
                                      o["points"]), o["x"] + c["cx"], o["y"] + c["cy"]),
                  P["canvas"]["bg"])
        membrane_cell(ax, o["x"] + c["cx"], o["y"] + c["cy"], o["genColor"],
                      c["r"], o["nucleus"], o["stroke"], o["genDashed"],
                      o["fill"], c["seed"], o["wobble"], o["lobes"],
                      o["points"], c["nx"], c["ny"], o["nucOpacity"],
                      c["fillMul"])
    # the true body is translucent, so without an opaque underlay the population
    # would show through it rather than sit behind it
    fill_path(ax, shift(blob_path(o["trueSeed"], t["r"], o["wobble"], o["lobes"],
                                  o["points"]), o["x"] + t["cx"], o["y"] + t["cy"]),
              P["canvas"]["bg"])
    membrane_cell(ax, o["x"] + t["cx"], o["y"] + t["cy"], o["trueColor"],
                  t["r"], o["nucleus"], o["stroke"], o["trueDashed"],
                  o["fill"], o["trueSeed"], o["wobble"], o["lobes"],
                  o["points"], o["truePos"]["nx"], o["truePos"]["ny"],
                  o["nucOpacity"], 1)


def draw_forces(ax):
    o, t, gens = energy_geom()
    F = P["forces"]
    A, D = F["attract"], F["disperse"]
    ox, oy = o["x"], o["y"]

    for i in A["from"]:
        c = gens[i]
        # measured off the gap between this cell and the true cell, so the
        # arrow never crosses either membrane
        dx, dy = t["cx"] - c["cx"], t["cy"] - c["cy"]
        L = np.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        x2, y2 = t["cx"] - ux * (t["r"] + A["gap"]), t["cy"] - uy * (t["r"] + A["gap"])
        x1, y1 = c["cx"] + ux * (c["r"] + A["gap"]), c["cy"] + uy * (c["r"] + A["gap"])
        ax1, ay1, bx1, by1 = span_arrow(x1, y1, x2, y2, A["reach"], A["place"])
        arrow_nodes(ax, ox + ax1, oy + ay1, ox + bx1, oy + by1,
                    C(A["color"]), A["stroke"], A["head"], False)

    for p in D["pairs"]:
        a, b = gens[p[0]], gens[p[1]]
        dx, dy = b["cx"] - a["cx"], b["cy"] - a["cy"]
        L = np.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        off = D["offset"]
        px, py = -uy * off, ux * off
        x1 = a["cx"] + ux * (a["r"] + D["gap"]) + px
        y1 = a["cy"] + uy * (a["r"] + D["gap"]) + py
        x2 = b["cx"] - ux * (b["r"] + D["gap"]) + px
        y2 = b["cy"] - uy * (b["r"] + D["gap"]) + py
        ax2, ay2, bx2, by2 = span_arrow(x1, y1, x2, y2, D["reach"], D["place"])
        arrow_nodes(ax, ox + ax2, oy + ay2, ox + bx2, oy + by2,
                    C(D["color"]), D["stroke"], D["head"], True)


def draw_key(ax):
    """Every glyph is drawn from the figure's own settings, so the key cannot
    fall out of step with what it describes."""
    o, cells, F = P["key"], P["cells"], P["forces"]
    cx = o["glyph"] / 2.0
    kx, ky = o["x"], o["y"]
    for i, e in enumerate(o["entries"]):
        y = i * o["rowGap"]
        if e["kind"] in ("gen", "true", "pop"):
            isT, isP = e["kind"] == "true", e["kind"] == "pop"
            pop = cells["pop"]
            membrane_cell(ax, kx + cx, ky + y,
                          cells["trueColor"] if isT or isP else cells["genColor"],
                          o["glyphR"], cells["nucleus"], cells["stroke"],
                          pop["dash"] if isP else
                          cells["trueDashed"] if isT else cells["genDashed"],
                          cells["fill"], 601 if isT or isP else 37,
                          cells["wobble"], cells["lobes"], cells["points"],
                          -0.16, -0.12, cells["nucOpacity"], 1,
                          pop["fade"] if isP else 1.0)
        else:
            spec = F["disperse"] if e["kind"] == "disperse" else F["attract"]
            half = o["arrowLen"] / 2.0
            arrow_nodes(ax, kx + cx - half, ky + y, kx + cx + half, ky + y,
                        C(spec["color"]), spec["stroke"], o["head"],
                        e["kind"] == "disperse")
        label = e["text"] + (" (%s)" % e["sym"] if e.get("sym") else "")
        text(ax, label, kx + o["glyph"] + o["gap"],
             ky + y + bl(o["size"]), o["size"], C(o["color"]))


def draw_loss(ax):
    """The energy score as one mathtext run, so TeX does the spacing, centred on
    the eloss anchor. Each label sits under its own term: a term's span is the
    run's width up to the term's end, less the term's own width."""
    o, F = P["eloss"], P["forces"]

    def W(m):
        return text_width(ax.figure, "$%s$" % m, o["size"], math_fontfamily=MATH_FONT)

    full = o["lhs"] + o["t1"] + "-" + o["t2"]
    x0 = o["x"] - W(full) / 2.0
    text(ax, "$%s$" % full, x0, o["y"], o["size"], C(o["color"]))
    end1 = W(o["lhs"] + o["t1"])
    spans = ((end1 - W(o["t1"]), end1, o["t1label"], "attract"),
             (W(full) - W(o["t2"]), W(full), o["t2label"], "disperse"))
    for a, b, lab, force in spans:
        text(ax, lab, x0 + (a + b) / 2.0, o["y"] + o["labelGap"],
             o["labelSize"], C(F[force]["color"]), anchor="middle")


def draw(ax):
    """The whole panel; z order follows call order."""
    draw_cells(ax)
    draw_forces(ax)
    draw_key(ax)
    draw_loss(ax)
