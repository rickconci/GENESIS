"""Extended Data: how the three evaluation metrics read one experiment.

The cells on the left are the control, the perturbed cells, and a GENESIS prediction.
Each row is one metric: what it extracts, then what it scores. Grey is unperturbed,
red is measured, blue dashed is the prediction. The pseudobulk strips, the Δ strips
and the scatter read one seeded gene table. The DEG row splits the volcano's true DEGs
into down (light red) and up (dark red) and draws one Venn per direction: only DEGs
called in the same direction count, as in the eval's direction-matched overlap.

The geometry is the GENESIS Schematic Studio figure, with the saved layout applied.
The energy distance is set the way Figure 1b sets the energy score: Computer Modern,
blackboard-bold E, double bars, a prime on the second draw. Those glyphs are wider
than the studio's drawn E, so the page grows a few millimetres past the studio's
154 mm. It stays under Nature's 180 mm, and the row bands follow the formula.

No data: every mark is drawn from fixed seeds.

    python -m scripts.figures.figure2.draw.ed_metrics.draw
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

from scripts.figures import CORE  # noqa: E402
from src.figures import ed_page  # noqa: E402
from src.figures.schematic import C, blob_path, rng, text_width  # noqa: E402

MM = 72.0 / 25.4                              # one drawing unit is one point
MATH = "cm"                                   # Figure 1b's energy-score face
# True DEGs by direction: two shades of the true red, so both still read as true.
UP, DOWN = "#a3302b", "#eb8f8a"
# E = 2E‖X−Y‖ − E‖X−X′‖ − E‖Y−Y′‖. X is the prediction, Y the truth.
# \mathbb and \Vert are Figure 1b's. No subscripts: mathtext draws them under 5 pt.
FORMULA = [
    (r"E=", ""),
    (r"2\,\mathbb{E}\,\Vert X-Y\Vert", "Predicted\nvs True"),
    ("-", ""),
    (r"\mathbb{E}\,\Vert X-X^{\prime}\Vert", "Predicted\nvs Predicted"),
    ("-", ""),
    (r"\mathbb{E}\,\Vert Y-Y^{\prime}\Vert", "True\nvs True"),
]


def nf(v: float, p: int = 2) -> float:
    """The studio rounds every emitted coordinate to 2 dp."""
    m = 10.0 ** p
    return (np.floor(v * m + 0.5) if v >= 0 else -np.floor(-v * m + 0.5)) / m


def gauss(r) -> float:
    u, v = max(1e-9, r()), r()
    return float(np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * np.pi * v))


def cloud(d: dict) -> list[tuple[float, float]]:
    """A wobbly disc, uniform in area, the edge softened by gaussian scatter."""
    rnd = rng(d["seed"])
    L = max(1, min(6, int(round(d.get("lobes", 4)))))
    terms = [(i + 2, d.get("wobble", 0.17) / (i + 1), rnd() * np.pi * 2) for i in range(L)]

    def radius(th: float) -> float:
        v = 1.0 + sum(a * np.cos(k * th + ph) for k, a, ph in terms)
        return max(0.25, v)

    pts = []
    for _ in range(d["n"]):
        th = rnd() * np.pi * 2
        rad = np.sqrt(rnd()) * radius(th)
        pts.append((d["cx"] + np.cos(th) * d["rx"] * rad + gauss(rnd) * d["sigma"],
                    d["cy"] + np.sin(th) * d["ry"] * rad + gauss(rnd) * d["sigma"]))
    return pts


def gene_table() -> list[dict]:
    """One control mean per gene, and the true and predicted shifts away from it."""
    rnd, out = rng(7), []
    for _ in range(16):
        ctrl = 0.25 + 0.55 * rnd()
        pert = min(1.0, max(0.04, ctrl + gauss(rnd) * 0.3))
        pred = min(1.0, max(0.04, ctrl + 0.9 * (pert - ctrl) + gauss(rnd) * 0.1))
        out.append(dict(ctrl=ctrl, pert=pert, dTrue=pert - ctrl, predPert=pred, dPred=pred - ctrl))
    return out


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den else 0.0


def volcano() -> tuple[list[dict], int]:
    rnd, pts, n_sig = rng(5), [], 0
    for _ in range(180):
        x = gauss(rnd) * 0.22
        if rnd() < 0.16:
            x += (-1 if rnd() < 0.5 else 1) * (0.35 + 0.55 * rnd())
        x = min(0.97, max(-0.97, x))
        y = min(0.97, max(0.0, abs(x) ** 1.3 * 0.95 * (0.55 + 0.7 * rnd()) + 0.04 * rnd()))
        sig = abs(x) > 0.35 and y > 0.42
        n_sig += int(sig)
        pts.append(dict(x=x, y=y, sig=sig))
    return pts, n_sig


def width(fig, s: str, size: float, weight: int = 400, math: bool = False) -> float:
    body = f"${s}$" if math else s
    kw = {"math_fontfamily": MATH} if math else {}
    return text_width(fig, body, size, weight, **kw)


def space(fig, size: float) -> float:
    """A space's advance. Measuring the space alone drops it."""
    return width(fig, "a a", size) - width(fig, "aa", size)


# ----------------------------------------------------------------------------
# Marks. y grows downward, as in the studio and in Figure 1's schematics.
# ----------------------------------------------------------------------------

def text(ax, s, x, y, size, fill, anchor="start", weight=400, rot=0, math=False):
    ha = {"start": "left", "middle": "center", "end": "right"}[anchor]
    ax.text(x, y, f"${s}$" if math else s, fontsize=size, color=fill, ha=ha, va="baseline",
            fontweight=("bold" if weight >= 600 else "normal"), rotation=rot,
            rotation_mode="anchor", math_fontfamily=MATH, clip_on=False, zorder=5)


def stroke(ax, pts, color, lw, dash=None):
    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1)
    ax.add_patch(PathPatch(MPath([(nf(p[0]), nf(p[1])) for p in pts], codes), fill=False,
                           edgecolor=color, lw=nf(lw, 3), joinstyle="miter", capstyle="butt",
                           linestyle=(0, dash) if dash else "solid", zorder=3, clip_on=False))


def arrow(ax, x1, y1, x2, y2, color, lw, head):
    dx, dy = x2 - x1, y2 - y1
    L = float(np.hypot(dx, dy) or 1.0)
    ux, uy = dx / L, dy / L
    stroke(ax, [(x1, y1), (x2 - ux * head * 0.75, y2 - uy * head * 0.75)], color, lw)
    px, py, w = -uy, ux, head * 0.46
    ax.add_patch(Polygon([(nf(x2), nf(y2)),
                          (nf(x2 - ux * head + px * w), nf(y2 - uy * head + py * w)),
                          (nf(x2 - ux * head - px * w), nf(y2 - uy * head - py * w))],
                         closed=True, facecolor=color, edgecolor="none", lw=0, zorder=3, clip_on=False))


def cell(ax, cx, cy, color, r, seed, dashed, nx, ny):
    """Membrane plus a nucleus shifted by (nx, ny) in units of the radius."""
    col = C(color)
    body, codes = blob_path(seed, r, 0.085, 3, 18, rnd=nf)
    ax.add_patch(PathPatch(MPath([(nf(v[0] + cx), nf(v[1] + cy)) for v in body], codes),
                           facecolor=matplotlib.colors.to_rgba(col, 0.2), edgecolor=col, lw=0.5,
                           joinstyle="round", linestyle=((0, (1.4, 1.1)) if dashed else "solid"),
                           zorder=3, clip_on=False))
    nuc, nc = blob_path(seed + 91, r * 0.34, 0.085 * 0.75, 3, 14, rnd=nf)
    ax.add_patch(PathPatch(MPath([(nf(v[0] + cx + r * nx), nf(v[1] + cy + r * ny)) for v in nuc], nc),
                           facecolor=matplotlib.colors.to_rgba(col, 0.85), edgecolor="none", lw=0,
                           zorder=3, clip_on=False))


def bars(ax, vals, x, y, w, frac, color, base, scale):
    pitch = w / len(vals)
    bw = pitch * frac
    for i, v in enumerate(vals):
        x0 = x + i * pitch + (pitch - bw) / 2.0
        y0, y1 = y + base, y + base - v * scale
        ax.add_patch(Rectangle((x0, min(y0, y1)), bw, abs(y1 - y0), facecolor=color,
                               edgecolor="none", lw=0, zorder=2, clip_on=False))


def role_color(word: str) -> str:
    bare = word.strip("′'’,.:;()").lower()
    if bare in {"genesis", "x", "predicted", "generated"}:
        return C("blue")
    if bare in {"true", "y", "measured"}:
        return C("red")
    return C("ink")


def caption_width(fig, s: str, size: float) -> float:
    sp = space(fig, size)
    return max(sum(width(fig, w, size) for w in line.split()) + sp * (len(line.split()) - 1)
               for line in s.split("\n"))


def caption(ax, fig, s, cx, y, size, lead) -> None:
    """A centred caption. Each word takes the colour of the group it names."""
    sp = space(fig, size)
    for li, line in enumerate(s.split("\n")):
        words = line.split()
        ws = [width(fig, w, size) for w in words]
        x = cx - (sum(ws) + sp * (len(words) - 1)) / 2.0
        for w, ww in zip(words, ws):
            text(ax, w, x, y + li * lead, size, role_color(w))
            x += ww + sp


def formula_columns(fig) -> list[tuple[str, str, float, float]]:
    """Each term, centred in a column as wide as the term or the name under it."""
    cols = []
    for expr, lab in FORMULA:
        tw = width(fig, expr, 6, math=True)
        lw = caption_width(fig, lab, 5) if lab else 0.0
        cols.append((expr, lab, tw, max(tw, lw)))
    return cols


def y_title(ax, fig, s, x, top, h, size, fill):
    """An axis title reading upward, centred on an axis of height h whose top is `top`."""
    text(ax, s, x, top + h / 2.0 + width(fig, s, size) / 2.0, size, fill, rot=90)


def draw(fig, ax, band_w: float) -> None:
    ink, grey, red, blue = C("ink"), C("grey"), C("red"), C("blue")
    genes, (vol, _) = gene_table(), volcano()

    # Alternate rows, so each metric reads as one line. The band runs out to the formula.
    for y, h in ((5, 57), (117.5, 52)):
        ax.add_patch(FancyBboxPatch((110, y), band_w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                    mutation_scale=1, facecolor=matplotlib.colors.to_rgba(C("panel"), 0.75),
                                    edgecolor="none", lw=0, zorder=0, clip_on=False))

    # The three populations. A prediction's membrane is dashed.
    lay = [(0, 0), (1, 0), (2, 0), (0.5, 0.87), (1.5, 0.87)]
    groups = [("Control \n(unperturbed) \ncells", "grey", False),
              ("Perturbed\ncells", "red", False),
              ("GENESIS\nprediction", "blue", True)]
    r, s = 3.6, 3.6 * 2.3
    for i, (lab, role, dashed) in enumerate(groups):
        y0 = 51 + i * 36
        for k, (px, py) in enumerate(lay):
            cell(ax, 10 + r + px * s, y0 + (py - 0.435) * s, role, r, 37 + i * 211 + k * 101,
                 dashed, 0.1 * ((k % 3) - 1), -0.1)
        lines = lab.split("\n")
        lx = 10 + r * 2 + 2 * s + 5
        for j, line in enumerate(lines):
            text(ax, line, lx, y0 + 5.8 * 0.355 + (j - (len(lines) - 1) / 2.0) * 7, 5.8, ink)

    # One bracket, then an arrow into each row.
    stroke(ax, [(80, 37), (84, 37), (84, 137), (80, 137)], ink, 0.6)
    stroke(ax, [(84, 87), (94, 87)], ink, 0.6)
    stroke(ax, [(94, 33), (94, 141)], ink, 0.6)
    for ry in (33, 87, 141):
        arrow(ax, 94, ry, 112, ry, ink, 0.6, 2.4)

    # Two lines where a name is wider than the gap before the row's first panel.
    for i, name in enumerate(["Direction-matched\nDEG overlap", "Pearson Δ", "Energy distance"]):
        lines = name.split("\n")
        for j, line in enumerate(lines):
            text(ax, line, 117, 33 + i * 54 + 6.5 * 0.355 + (j - (len(lines) - 1) / 2.0) * 8,
                 6.5, ink, weight=700)

    # Volcano. Points past both thresholds are the DEGs the Venns count, shaded by direction.
    vx, vy, vw, vh = 196, 14, 84, 36

    def Xv(v):
        return vw / 2 + v * (vw / 2) * 0.94

    def Yv(v):
        return vh - v * vh

    for x in (-0.35, 0.35):
        stroke(ax, [(vx + Xv(x), vy), (vx + Xv(x), vy + vh)], grey, 0.35, (1.4, 1.1))
    stroke(ax, [(vx, vy + Yv(0.42)), (vx + vw, vy + Yv(0.42))], grey, 0.35, (1.4, 1.1))
    for p in vol:
        tone = (UP if p["x"] > 0 else DOWN) if p["sig"] else grey
        ax.add_patch(Circle((vx + Xv(p["x"]), vy + Yv(p["y"])), 0.7, facecolor=tone,
                            edgecolor="none", lw=0, alpha=0.9, zorder=3, clip_on=False))
    stroke(ax, [(vx, vy), (vx, vy + vh), (vx + vw, vy + vh)], ink, 0.4)
    text(ax, "log2 fold change", vx + vw / 2, vy + vh + 5 + 5.3 * 0.72, 5.3, ink, "middle")
    y_title(ax, fig, "−log10 p", vx - 4, vy, vh, 5.3, ink)
    text(ax, "Down ↓", vx + Xv(-0.66), vy - 3, 5, ink, "middle")
    text(ax, "Up ↑", vx + Xv(0.66), vy - 3, 5, ink, "middle")
    arrow(ax, 286, 34, 304, 34, ink, 0.6, 2.4)

    # One Venn per direction, each reading its half of the volcano: shared = round(0.7 · n),
    # plus a few the prediction alone calls. A gene predicted in the wrong direction lands in
    # the other Venn's predicted-only lobe, so the score, shared down + shared up, is the eval's
    # direction-matched count: one top-100 set per arm, split by sign afterwards.
    vr, vd, vy0 = 11.0, 9.0, 33.0
    for cx, side, tone, title, pred_only in ((330.0, -1, DOWN, "Down ↓", 3), (395.0, 1, UP, "Up ↑", 2)):
        n = sum(p["sig"] and np.sign(p["x"]) == side for p in vol)
        shared = int(np.floor(n * 0.7 + 0.5))
        ax.add_patch(Circle((cx - vd / 2, vy0), vr, facecolor=matplotlib.colors.to_rgba(tone, 0.15),
                            edgecolor=tone, lw=0.6, zorder=3, clip_on=False))
        ax.add_patch(Circle((cx + vd / 2, vy0), vr, facecolor=matplotlib.colors.to_rgba(blue, 0.12),
                            edgecolor=blue, lw=0.6, linestyle=(0, (1.5, 1.1)), zorder=3, clip_on=False))
        for x, k in ((-11, n - shared), (0, shared), (11, pred_only)):
            text(ax, str(k), cx + x, vy0 + 5.3 * 0.355, 5.3, ink, "middle")
        text(ax, title, cx, vy0 - vr - 3, 5.3, ink, "middle", 700)
    text(ax, "+", 362.5, vy0 + 6.5 * 0.355, 6.5, ink, "middle")
    text(ax, "True DEGs", 360.5, vy0 + vr + 9, 5.3, red, "end")
    text(ax, "Predicted DEGs", 364.5, vy0 + vr + 9, 5.3, blue)

    # One control, forked into the true and the predicted mean, then each shift.
    mx = max(max(abs(g["dTrue"]), abs(g["dPred"])) for g in genes) or 1.0
    sc, ox, oy = (14 / 2) / mx, 196, 72
    cy, pX, dX, rs = oy + 11, ox + 76, ox + 128, 22
    bars(ax, [g["ctrl"] for g in genes], ox, cy, 36, 0.6, grey, 11, 11)
    stroke(ax, [(ox, cy + 11), (ox + 36, cy + 11)], ink, 0.35)
    text(ax, "mean, control", ox + 18, cy - 4, 5, ink, "middle")
    hy = oy - 5.5
    text(ax, "mean, perturbed", pX + 18, hy, 5, ink, "middle")
    text(ax, "Δ = perturbed − control", dX + 18, hy, 5, ink, "middle")
    sx = ox + 42
    stroke(ax, [(ox + 38, cy + 5.5), (sx, cy + 5.5)], ink, 0.5)
    stroke(ax, [(sx, oy + 5.5), (sx, oy + rs + 5.5)], ink, 0.5)
    for y0, col, lab, key, dkey in ((oy, red, "True", "pert", "dTrue"),
                                    (oy + rs, blue, "Predicted", "predPert", "dPred")):
        mid = y0 + 5.5
        bars(ax, [g[key] for g in genes], pX, y0, 36, 0.6, col, 11, 11)
        stroke(ax, [(pX, y0 + 11), (pX + 36, y0 + 11)], ink, 0.35)
        arrow(ax, sx, mid, pX - 2, mid, ink, 0.5, 2.2)
        text(ax, lab, (sx + pX - 2) / 2, mid - 2.5, 5.3, col, "middle", 700)
        arrow(ax, pX + 39, mid, dX - 3, mid, ink, 0.5, 2.2)
        bars(ax, [g[dkey] for g in genes], dX, mid, 36, 0.6, col, 0, sc)
        stroke(ax, [(dX, mid), (dX + 36, mid)], ink, 0.35)
    m0, m1, bx = oy + 5.5, oy + rs + 5.5, dX + 39
    stroke(ax, [(bx, m0), (bx + 3, m0), (bx + 3, m1), (bx, m1)], ink, 0.5)
    stroke(ax, [(bx + 3, (m0 + m1) / 2), (bx + 7, (m0 + m1) / 2)], ink, 0.5)
    arrow(ax, 370, 88.5, 386, 88.5, ink, 0.6, 2.4)

    # The same genes, true shift against predicted shift. r is the correlation of these points.
    a = [g["dTrue"] for g in genes]
    b = [g["dPred"] for g in genes]
    lim = max(max(abs(v) for v in a + b) * 1.12, 1.0)
    sx0, sy0, sw, sh = 396, 71, 34, 34

    def Xs(v):
        return sw / 2 + (v / lim) * (sw / 2)

    def Ys(v):
        return sh / 2 - (v / lim) * (sh / 2)

    stroke(ax, [(sx0 + Xs(-lim), sy0 + Ys(-lim)), (sx0 + Xs(lim), sy0 + Ys(lim))], grey, 0.4, (1.4, 1.1))
    for u, v in zip(a, b):
        ax.add_patch(Circle((sx0 + Xs(u), sy0 + Ys(v)), 0.95, facecolor=C("inkSoft"),
                            edgecolor="none", lw=0, zorder=3, clip_on=False))
    stroke(ax, [(sx0, sy0), (sx0, sy0 + sh), (sx0 + sw, sy0 + sh)], ink, 0.4)
    text(ax, "Δ True", sx0 + sw / 2, sy0 + sh + 5 + 5 * 0.72, 5, red, "middle")
    y_title(ax, fig, "Δ Predicted", sx0 - 4, sy0, sh, 5, blue)
    text(ax, f"r = {pearson(a, b):.2f}", sx0 + 2.5, sy0 + 5.3, 5.3, ink, weight=700)

    # Every cell, kept. A between-pair is split red and blue; a within-pair is one colour.
    ox, oy, cw, ch = 196, 120, 96, 44
    true = cloud(dict(n=70, cx=0.42, cy=0.56, rx=0.24, ry=0.30, wobble=0.25, lobes=3, sigma=0.045, seed=41))
    pred = cloud(dict(n=70, cx=0.60, cy=0.54, rx=0.24, ry=0.30, wobble=0.25, lobes=3, sigma=0.045, seed=57))
    stroke(ax, [(ox, oy), (ox, oy + ch), (ox + cw, oy + ch)], ink, 0.4)

    def inside(p):
        return 0.02 < p[0] < 0.98 and 0.04 < p[1] < 0.97

    for pts, col in ((true, red), (pred, blue)):
        for p in pts:
            if inside(p):
                ax.add_patch(Circle((ox + p[0] * cw, oy + p[1] * ch), 0.6, facecolor=col,
                                    edgecolor="none", lw=0, alpha=0.85, zorder=3, clip_on=False))

    def near(pts, x, y):
        keep = [p for p in pts if inside(p)]
        return min(keep, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)

    for src, axy, dst, bxy, split in ((true, (0.34, 0.42), pred, (0.68, 0.40), True),
                                      (true, (0.36, 0.76), pred, (0.70, 0.78), True),
                                      (true, (0.20, 0.36), true, (0.24, 0.82), False),
                                      (pred, (0.82, 0.34), pred, (0.78, 0.82), False)):
        p, q = near(src, *axy), near(dst, *bxy)
        P = (ox + p[0] * cw, oy + p[1] * ch)
        Q = (ox + q[0] * cw, oy + q[1] * ch)
        c0 = red if src is true else blue
        c1 = blue if dst is pred else red
        if split:
            M = ((P[0] + Q[0]) / 2, (P[1] + Q[1]) / 2)
            stroke(ax, [P, M], red, 0.55)
            stroke(ax, [M, Q], blue, 0.55)
        else:
            stroke(ax, [P, Q], c0, 0.55)
        ax.add_patch(Circle(P, 1.1, facecolor=c0, edgecolor="none", lw=0, zorder=4, clip_on=False))
        ax.add_patch(Circle(Q, 1.1, facecolor=c1, edgecolor="none", lw=0, zorder=4, clip_on=False))
    text(ax, "Y: True", ox + 3, oy + 6, 5, red)
    text(ax, "X: Predicted", ox + cw - 1, oy + 6, 5, blue, "end")
    arrow(ax, 298, 143, 312, 143, ink, 0.6, 2.4)

    # The energy distance, in Figure 1b's face. Each term is named under itself.
    x, gap = 316.0, 3.5
    for expr, lab, tw, bw in formula_columns(fig):
        text(ax, expr, x + (bw - tw) / 2.0, 143, 6, ink, math=True)
        if lab:
            caption(ax, fig, lab, x + bw / 2.0, 150, 5, 6)
        x += bw + gap
    if x - gap > ax.get_xlim()[1] - 2:
        raise RuntimeError(f"the energy-distance formula ends at {x - gap:.0f} pt, past the page")


def main(data: Path = CORE / "data" / "figure2", out: Path = CORE / "figure2") -> None:
    """``data`` is unused (no inputs); kept so every draw script is called the same way."""
    ed_page.style()
    probe = plt.figure(figsize=(2, 2))
    probe.canvas.draw()
    cols = formula_columns(probe)
    plt.close(probe)
    right = 316.0 + sum(c[3] for c in cols) + 3.5 * (len(cols) - 1)
    # The studio's 154 mm page held its drawn E. Computer Modern needs the rest.
    band_right = max(110.0 + 323.0, right + 2.0)
    page_pt = max(154.0 * MM, band_right + 4.0)
    if page_pt > 180.0 * MM + 0.1:
        raise RuntimeError(f"the formula needs {page_pt / MM:.1f} mm; Nature allows 180")
    w_mm, h_mm = page_pt / MM, 61.0
    fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm * MM)
    ax.set_ylim(h_mm * MM, 0)
    ax.set_axis_off()
    ax.patch.set_linewidth(0)
    draw(fig, ax, band_right - 110.0)
    ed_page.save(fig, out / "ed" / "ED_metrics")
    plt.close(fig)


if __name__ == "__main__":
    main()
