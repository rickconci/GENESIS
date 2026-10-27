"""Figure 1a: GENESIS's architecture, a schematic (no data). figure1.py calls draw(ax) on an axes
whose units are points with y running down, as in the studio (the in-house layout tool) it was ported from.
"""
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle, Rectangle, Polygon, FancyBboxPatch

from src.figures.schematic import C, bl, blob_path, rng, text_width

# shared.type / shared.line scales are both 1, so TS()/LW() are the identity
# and every size below is already in points.
# ── the figure, exactly as the studio stores it ────────────────────────────
P = {
    "canvas": {"wmm": 113, "hmm": 34, "bg": "#ffffff"},

    "covariates": {
        # y centres the rows (and so the column) on the network: 6.84 + 68 / 2 = -0.66 + 83 / 2
        "x": 3.67, "y": 6.84,
        "boxW": 120, "rowH": 14, "rowGap": 4, "radius": 1.5, "stroke": 0.5,
        "iconBox": 9, "iconGap": 4,
        "nameSize": 5.5, "nameWeight": 700, "valueSize": 5.5,
        # 3 tokens of 6 pt: stacked, a row's tokens are exactly one row pitch
        # (rowH + rowGap = 18), so the concatenated column lines up with the rows
        "tokens": 3, "tokenSize": 6, "tokenGap": 0, "tokenPad": 8,
        "rows": [
            {"icon": "cell", "name": "Biological context", "tone": "teal",
             "value": "cell line C32", "dashed": False},
            {"icon": "bolt", "name": "Perturbation", "tone": "orange",
             "value": "Amsacrine", "dashed": False},
            {"icon": "dish", "name": "Replication context", "tone": "magenta",
             "value": "plate 6", "dashed": False},
            {"icon": "uniform", "name": "Noise", "tone": "grey",
             "value": "Uniform sample per cell", "dashed": True,
             "learned": False},       # appended raw (pertgen.py:140), not embedded
        ],
    },

    # centred under tokens + column (131.67 .. 170.39); the input caption is
    # centred under its boxes
    "embedLabel": {"x": 151.0, "y": 87, "size": 5.6, "color": "ink",
                   "anchor": "middle", "text": "Learned embeddings"},

    "concat": {"x": 164.39, "stroke": 0.45, "strokeColor": "ink",
               "feedStroke": 0.45, "sampleLabel": "sampled"},

    # Input layer = the concatenated column; these are the layers after it, x of
    # the first hidden layer to x of the output layer. y and h are the studio's:
    # node pitch h / (n + 1), so a wider layer is drawn taller.
    "network": {"x": 188.32, "w": 35.87, "y": -0.66, "h": 83,
                "layers": [8, 16, 8], "r": 1.9, "edgeOpacity": 0.18,
                "midColor": "ink", "outColor": "ink"},

    "cells4": {
        "x": 236.15, "y": 11.62, "w": 86, "h": 45,
        "color": "blue", "r": 8.5, "stroke": 0.9, "fill": 0.13,
        "nucleus": 0.33, "nucOpacity": 0.85,
        "lobes": 3, "points": 20, "wobble": 0.085,
        "cells": [{"cx": 0.30, "cy": 0.26, "r": 1.00},
                  {"cx": 0.64, "cy": 0.22, "r": 1.14},
                  {"cx": 0.36, "cy": 0.76, "r": 0.88},
                  {"cx": 0.69, "cy": 0.74, "r": 1.06}],
    },

    # the callout opens from its left edge onto the strip's top edge
    "zoom": {"x": 258.24, "y": 38.47, "w": 16.5, "h": 14.5,
             "color": "grey", "stroke": 0.4, "fromEdge": "left", "toEdge": "top"},

    "genes": {"x": 247.01, "y": 8.56, "w": 64.56, "h": 14.5, "rotate": 90,
              "bars": 90, "density": 0.07, "color": "blue"},

    "genesBrace": {"x": 232.31, "y": 74.6, "w": 14, "depth": 5,
                   "rotate": 0, "stroke": 0.5, "color": "ink"},

    "capInput": {"x": 63.67, "y": 87, "size": 5.8, "color": "ink", "anchor": "middle",
                 "text": "Input: condition covariates + noise"},
    # centred under the network (188.32 .. 224.19): "MLP" alone is too short to
    # start left-aligned without looking off-centre
    "capMLP": {"x": 206.25, "y": 87, "size": 5.8, "color": "ink",
               "anchor": "middle", "text": "MLP"},
    # one line is ~118 pt; the column under the cells is ~89 pt, so it
    # breaks after "expression" and stays left of the edge.
    "capOutput": {"x": 231.65, "y": 87, "size": 5.8, "color": "ink",
                  "text": "Output: mRNA expression\n"
                          "of a single cell"},
}

# the studio's spike seed for the mRNA strip (drawGenes calls spectrum(41, ...))
GENES_SEED = 41


# ── maths helpers, ported from the studio ─────────────────────────────────
def spectrum_segments(seed, bars, w, h, density):
    """The mRNA spike vector: `density` of the bars are tall, the rest are a
    low floor. Returns (segments, bar_width) in the strip's local frame."""
    r = rng(seed)
    bw = max(0.35, min(0.9, (w / bars) * 0.55))
    segs = []
    for i in range(bars):
        u = r()
        if u < density:
            v = 0.30 + 0.70 * (r() ** 1.5)
        else:
            v = 0.015 + 0.075 * r()
        x = (i + 0.5) * (w / bars)
        segs.append(((x, h), (x, h - v * h)))
    return segs, bw


def rot_pt(x, y, deg):
    """SVG rotate(deg): clockwise on a y-down canvas."""
    a = np.deg2rad(deg)
    ca, sa = np.cos(a), np.sin(a)
    return (x * ca - y * sa, x * sa + y * ca)


# ── drawing primitives on a y-down axes ───────────────────────────────────
_Z = [1]


def z():
    _Z[0] += 1
    return _Z[0]


def path_from_segments(segs):
    verts, codes = [], []
    for a, b in segs:
        verts += [a, b]
        codes += [Path.MOVETO, Path.LINETO]
    return Path(verts, codes)


def dash(stroke, on, off):
    """The studio writes stroke-dasharray in user units; matplotlib wants the
    pattern scaled by the line width it is drawn with."""
    return (0, (on / stroke, off / stroke))


def stroke_path(ax, path, color, lw, dashes=None, join="miter", cap="butt",
                alpha=None):
    pp = PathPatch(path, fill=False, edgecolor=color, linewidth=lw,
                   joinstyle=join, capstyle=cap, alpha=alpha, zorder=z())
    if dashes:
        pp.set_linestyle(dashes)
    ax.add_patch(pp)
    return pp


def fill_path(ax, path, color, alpha=1.0):
    pp = PathPatch(path, facecolor=color, edgecolor="none", alpha=alpha,
                   lw=0, zorder=z())
    ax.add_patch(pp)
    return pp


def rrect(ax, x, y, w, h, rx, facecolor="none", edgecolor=None, lw=0,
          dashes=None, alpha=None):
    """The studio's <rect rx=...>: FancyBboxPatch keeps the bbox exact."""
    if rx and rx > 0:
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0,rounding_size=%g" % rx,
                           mutation_scale=1, facecolor=facecolor,
                           edgecolor=("none" if edgecolor is None else edgecolor),
                           linewidth=lw, alpha=alpha, zorder=z())
    else:
        p = Rectangle((x, y), w, h, facecolor=facecolor,
                      edgecolor=("none" if edgecolor is None else edgecolor),
                      linewidth=lw, alpha=alpha, zorder=z())
    if dashes:
        p.set_linestyle(dashes)
    ax.add_patch(p)
    return p


_ANCHOR = {"middle": "center", "end": "right", "start": "left"}


def text(ax, s, x, y, size, color, anchor="start", weight=400, rotate=0):
    if not s:
        return None
    return ax.text(x, y, s, fontsize=size, color=color,
                   ha=_ANCHOR.get(anchor, "left"), va="baseline",
                   fontweight=("bold" if weight >= 600 else "normal"),
                   rotation=-rotate, rotation_mode="anchor", zorder=z())


def multiline(ax, s, x, y, size, color, anchor="start", weight=400):
    lh = size * 1.28
    for i, line in enumerate(str(s).split("\n")):
        text(ax, line, x, y + i * lh, size, color, anchor, weight)


# ══════════════════════ framework (architecture) ══════════════════════

def covariate_icon(ax, kind, box, color, ox, oy, bg):
    """Small glyphs that say what each covariate is, drawn rather than set."""
    c = box / 2.0
    st = 0.55
    col = C(color)

    def X(v): return ox + v

    def Y(v): return oy + v

    def poly(pts):                           # an open polyline, in the icon's box
        return Path([(X(p[0]), Y(p[1])) for p in pts], [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1))

    if kind == "cell":
        ax.add_patch(Circle((X(c), Y(c)), box * 0.33, facecolor=col,
                            alpha=0.14, edgecolor="none", zorder=z()))
        ax.add_patch(Circle((X(c), Y(c)), box * 0.33, facecolor="none",
                            edgecolor=col, linewidth=st, zorder=z()))
        ax.add_patch(Circle((X(c), Y(c)), box * 0.12, facecolor=col,
                            edgecolor="none", zorder=z()))
    elif kind == "bolt":
        w, h = box * 0.20, box * 0.34
        pts = [(c + w * 0.2, c - h), (c - w, c + h * 0.12),
               (c - w * 0.1, c + h * 0.12), (c - w * 0.2, c + h),
               (c + w, c - h * 0.15), (c + w * 0.1, c - h * 0.15)]
        ax.add_patch(Polygon([(X(p[0]), Y(p[1])) for p in pts], closed=True,
                             facecolor=col, edgecolor="none", zorder=z()))
    elif kind == "dish":
        rx, ry, dp = box * 0.34, box * 0.34 * 0.34, box * 0.20
        top = c - ry * 0.4
        # body + bowl: the SVG arc is exactly a half ellipse bulging downwards
        th = np.linspace(np.pi, 0.0, 48)
        arc = [(c + rx * np.cos(t), top + dp + ry * np.sin(t)) for t in th]
        pth = poly([(c - rx, top), (c - rx, top + dp)] + arc + [(c + rx, top)])
        fill_path(ax, pth, col, alpha=0.14)
        stroke_path(ax, pth, col, st)
        # the rim sits on top, filled with the canvas colour so the body
        # does not show through
        ep = poly([(c + rx * np.cos(t), top + ry * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 72)])
        ax.add_patch(PathPatch(ep, facecolor=bg, edgecolor=col, linewidth=st,
                               zorder=z()))
    else:                                    # uniform: a boxcar, flat top and vertical shoulders
        w3, h3 = box * 0.36, box * 0.30
        base = c + h3 * 0.8
        top = base - 0.95 * h3
        pth = poly([(c - w3, base), (c - w3 * 0.70, base), (c - w3 * 0.70, top),
                    (c + w3 * 0.70, top), (c + w3 * 0.70, base), (c + w3, base)])
        stroke_path(ax, pth, col, st, join="round", cap="round")


def cov_row_tops(cv):
    """Row tops, shared by the covariate boxes and the concatenated vector so
    the feed lines always meet the right block."""
    return [i * (cv["rowH"] + cv["rowGap"]) for i in range(len(cv["rows"]))]


def draw_covariates(ax, bg):
    """Covariate boxes in the loss figure's idiom, each carrying an example,
    with the covariate's icon set inline just after it."""
    o = P["covariates"]
    tops = cov_row_tops(o)
    sp = text_width(ax.figure, " ", o["valueSize"], 400)
    for ri, row in enumerate(o["rows"]):
        y = o["y"] + tops[ri]
        x = o["x"]
        tone = C(row["tone"])
        rrect(ax, x, y, o["boxW"], o["rowH"], o["radius"], facecolor="none",
              edgecolor=C("ink"), lw=o["stroke"],
              dashes=(dash(o["stroke"], 1.6, 1.2) if row["dashed"] else None))
        name_txt = row["name"] + ":"
        wN = text_width(ax.figure, name_txt, o["nameSize"], o["nameWeight"])
        wV = text_width(ax.figure, row["value"], o["valueSize"], 400)
        wI = o["iconGap"] + o["iconBox"]
        # the whole name + value + icon run is centred in the box
        x0 = (o["boxW"] - (wN + sp + wV + wI)) / 2.0
        base = y + o["rowH"] / 2.0 + bl(o["nameSize"])
        text(ax, name_txt, x + x0, base, o["nameSize"], C("ink"),
             weight=o["nameWeight"])
        text(ax, row["value"], x + x0 + wN + sp, base, o["valueSize"], C("ink"))
        covariate_icon(ax, row["icon"], o["iconBox"], row["tone"],
                       x + x0 + wN + sp + wV + o["iconGap"],
                       y + o["rowH"] / 2.0 - o["iconBox"] / 2.0, bg)
        # the per-row embedding tokens, alternating full / 45% shade; the noise
        # row has none: it is drawn, not looked up (see draw_concat)
        if not row.get("learned", True):
            continue
        tzx = o["boxW"] + o["tokenPad"]
        for i in range(o["tokens"]):
            rrect(ax, x + tzx + i * (o["tokenSize"] + o["tokenGap"]),
                  y + o["rowH"] / 2.0 - o["tokenSize"] / 2.0,
                  o["tokenSize"], o["tokenSize"], 0.4,
                  facecolor=tone, alpha=(0.45 if i % 2 else 1.0))


def draw_embed_label(ax):
    o = P["embedLabel"]
    text(ax, o["text"], o["x"], o["y"], o["size"], C(o["color"]), anchor=o["anchor"])


# A noise draw, fixed so rebuilds are identical: grey levels that read as random
# values rather than the learned rows' regular full / light alternation.
NOISE_SHADES = (0.85, 0.25, 0.55)


def concat_cells():
    """The concatenated vector, one (x, y, size, colour, alpha) per square, top
    down. Each covariate's block is centred on its row, and blocks are exactly
    one row pitch tall, so the column is contiguous AND every feed is level."""
    o, cv = P["concat"], P["covariates"]
    sz, n, tops = cv["tokenSize"], cv["tokens"], cov_row_tops(cv)
    if abs(n * sz - (cv["rowH"] + cv["rowGap"])) > 1e-9:
        raise ValueError("concat column needs tokens * tokenSize == rowH + rowGap")
    cells = []
    for row, top in zip(cv["rows"], tops):
        y0 = cv["y"] + top + cv["rowH"] / 2.0 - n * sz / 2.0
        learned = row.get("learned", True)
        for i in range(n):
            alpha = (0.45 if i % 2 else 1.0) if learned else NOISE_SHADES[i]
            cells.append((o["x"], y0 + i * sz, sz, C(row["tone"]), alpha))
    return cells


def draw_concat(ax):
    """The concatenated vector, which is also the MLP's input layer. A learned
    row's squares are copies of its tokens (same size, colour, shade, order),
    so the reader sees the embeddings stacked; the noise row has no tokens, so
    its feed starts at its box and its squares are a fresh draw per cell."""
    o, cv = P["concat"], P["covariates"]
    cells = concat_cells()
    for x, y, sz, col, alpha in cells:
        rrect(ax, x, y, sz, sz, 0, facecolor=col, alpha=alpha)
    top, sz = cells[0][1], cv["tokenSize"]
    rrect(ax, o["x"], top, sz, cells[-1][1] + sz - top, 0.4, facecolor="none",
          edgecolor=C(o["strokeColor"]), lw=o["stroke"])
    for row, top in zip(cv["rows"], cov_row_tops(cv)):
        y = cv["y"] + top + cv["rowH"] / 2.0
        learned = row.get("learned", True)
        x0 = cv["x"] + cv["boxW"] + (cv["tokenPad"] + cv["tokens"] * cv["tokenSize"] if learned else 0) + 1.5
        stroke_path(ax, Path([(x0, y), (o["x"] - 2.6, y)], [Path.MOVETO, Path.LINETO]),
                    C(row["tone"]), o["feedStroke"])
        ax.add_patch(Polygon([(o["x"], y), (o["x"] - 2.8, y - 1.2), (o["x"] - 2.8, y + 1.2)],
                             closed=True, facecolor=C(row["tone"]), edgecolor="none", zorder=z()))
        if not learned:     # sits where the tokens would be: "not an embedding"
            text(ax, o["sampleLabel"], (x0 + o["x"]) / 2.0, y - 1.6, 5.0, C("charcoal"),
                 anchor="middle")


def draw_network(ax):
    """The MLP after its input layer. The input layer is the concatenated column
    (draw_concat), so its edges leave the squares' right edges."""
    n = P["network"]
    cells = concat_cells()
    ys = [y + sz / 2.0 for _x, y, sz, _c, _a in cells]
    nl = len(n["layers"])
    cols = [[(cells[0][0] + cells[0][2], y) for y in ys]]
    for li, cnt in enumerate(n["layers"]):
        x = n["x"] + li * (n["w"] / (nl - 1))
        cols.append([(x, n["y"] + n["h"] / (cnt + 1) * (i + 1)) for i in range(cnt)])
    # all edges of a layer pair in one faint path, as the studio does
    for a, b in zip(cols, cols[1:]):
        stroke_path(ax, path_from_segments([(p, q) for p in a for q in b]),
                    C("ink"), 0.25, alpha=n["edgeOpacity"])
    for li, pts in enumerate(cols[1:]):
        out = li == nl - 1                    # hidden layers read as open rings
        f = C(n["outColor"] if out else n["midColor"])
        for p in pts:
            ax.add_patch(Circle(p, n["r"] if out else n["r"] * 0.85,
                                facecolor=(f if out else "#ffffff"),
                                edgecolor=f, linewidth=0.45, zorder=z()))


def membrane_cell(ax, ox, oy, o, r, seed):
    """One cell: a wobbly membrane rather than a circle, a solid nucleus at its
    centre, and a tinted body. Fill and stroke are separate patches, as in the studio."""
    col = C(o["color"])
    v, codes = blob_path(seed, r, o["wobble"], o["lobes"], o["points"])
    bp = Path(np.asarray(v, float) + np.array([ox, oy]), codes)
    ax.add_patch(PathPatch(bp, facecolor=col, alpha=o["fill"], edgecolor="none",
                           lw=0, zorder=z()))
    ax.add_patch(PathPatch(bp, facecolor="none", edgecolor=col, lw=o["stroke"],
                           joinstyle="round", zorder=z()))
    v, codes = blob_path(seed + 91, r * o["nucleus"], o["wobble"] * 0.75, 3, 14)
    np_ = Path(np.asarray(v, float) + np.array([ox, oy]), codes)
    ax.add_patch(PathPatch(np_, facecolor=col, alpha=o["nucOpacity"], edgecolor="none",
                           lw=0, zorder=z()))


def draw_cells4(ax):
    """Exactly four cells, positioned as fractions of the panel."""
    o = P["cells4"]
    for i, c in enumerate(o["cells"]):
        membrane_cell(ax, o["x"] + c["cx"] * o["w"], o["y"] + c["cy"] * o["h"],
                      o, o["r"] * c["r"], (i + 1) * 37)


def genes_corners():
    """Where the spike vector actually is, in the figure's own coordinates: its
    four corners after whatever rotation the strip carries. Read by the callout,
    so the callout's tips cannot come away from the strip."""
    g = P["genes"]
    def pt(px, py):
        rx, ry = rot_pt(px, py, g.get("rotate") or 0)
        return (g["x"] + rx, g["y"] + ry)
    # leading edge first: the two ends of the vector the callout opens onto
    return [pt(0, 0), pt(g["w"], 0), pt(g["w"], g["h"]), pt(0, g["h"])]


def edge_pair(c, which):
    """The two corners of one edge of a box given as [TL, TR, BR, BL]."""
    if which == "top":
        return [c[0], c[1]]
    if which == "right":
        return [c[1], c[2]]
    if which == "bottom":
        return [c[3], c[2]]
    return [c[0], c[3]]                       # left


def seg_cross(p1, p2, p3, p4):
    """Do p1p2 and p3p4 cross? Used to pick the pairing of corners that keeps
    the two leaders from scissoring over each other."""
    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
    d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def draw_zoom(ax):
    """A callout: mark a patch on the cells and open it onto the strip."""
    o = P["zoom"]
    col = C(o["color"])
    dashes = dash(o["stroke"], 1.3, 1.1)
    rrect(ax, o["x"], o["y"], o["w"], o["h"], 0, facecolor="none",
          edgecolor=col, lw=o["stroke"], dashes=dashes)
    # the patch marks a place on the cells; the callout opens onto the strip itself
    q = [(p[0] - o["x"], p[1] - o["y"]) for p in genes_corners()]

    # one named edge at each end, and whichever pairing of their corners does
    # not cross -- so changing either edge never needs the other rethought
    A = edge_pair([(0, 0), (o["w"], 0), (o["w"], o["h"]), (0, o["h"])], o["fromEdge"])
    B = edge_pair(q, o["toEdge"])
    if seg_cross(A[0], B[0], A[1], B[1]) and not seg_cross(A[0], B[1], A[1], B[0]):
        B = [B[1], B[0]]
    for L in ((A[0], B[0]), (A[1], B[1])):
        pth = Path([(o["x"] + L[0][0], o["y"] + L[0][1]),
                    (o["x"] + L[1][0], o["y"] + L[1][1])],
                   [Path.MOVETO, Path.LINETO])
        stroke_path(ax, pth, col, o["stroke"], dashes=dashes)
    pth = Path([(o["x"] + p[0], o["y"] + p[1]) for p in q] + [(0, 0)],
               [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
                Path.CLOSEPOLY])
    stroke_path(ax, pth, col, o["stroke"], dashes=dashes)


def draw_genes(ax):
    """The spike vector, on its own origin and its own angle. Three blocks, not
    one: the bracket and the caption next door carry their own x, y and angle."""
    o = P["genes"]
    rot = o.get("rotate") or 0
    segs, bw = spectrum_segments(GENES_SEED, o["bars"], o["w"], o["h"],
                                 o["density"])
    out = []
    for a, b in segs:
        ra, rb = rot_pt(*a, rot), rot_pt(*b, rot)
        out.append(((o["x"] + ra[0], o["y"] + ra[1]),
                    (o["x"] + rb[0], o["y"] + rb[1])))
    stroke_path(ax, path_from_segments(out), C(o["color"]), bw, cap="butt")


def draw_genes_brace(ax):
    """The bracket under the strip, on its own origin and its own angle."""
    o = P["genesBrace"]
    rot = o.get("rotate") or 0
    d = o["depth"]
    pts = [(0, 0), (0, d), (o["w"], d), (o["w"], 0)]
    verts = []
    for p in pts:
        r = rot_pt(p[0], p[1], rot)
        verts.append((o["x"] + r[0], o["y"] + r[1]))
    stroke_path(ax, Path(verts, [Path.MOVETO] + [Path.LINETO] * 3),
                C(o["color"]), o["stroke"])


def draw_text_block(ax, o):
    multiline(ax, o["text"], o["x"], o["y"], o["size"], C(o["color"]),
              anchor=o.get("anchor", "start"), weight=o.get("weight", 400))


# ══════════════════════════ assemble ══════════════════════════

def draw(ax):
    """The whole panel, in the studio's BUILDERS order for `framework` (z order follows call order)."""
    draw_covariates(ax, P["canvas"]["bg"])
    draw_embed_label(ax)
    draw_concat(ax)
    draw_network(ax)
    draw_cells4(ax)
    draw_zoom(ax)
    draw_genes(ax)
    draw_genes_brace(ax)
    for cap in ("capInput", "capMLP", "capOutput"):
        draw_text_block(ax, P[cap])
