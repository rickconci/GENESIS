"""Figure 1c: the experimental repeat vs the split-half, a schematic (no data). figure1.py calls
draw(ax) on an axes whose units are points with y running down, as in the studio (the in-house layout
tool) it was ported from.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle, Rectangle, Polygon, FancyBboxPatch

from src.figures.schematic import C, MM, text_width

# ───────── figure params, key names exactly as in the studio ─────────
P = {
    "canvas": {"wmm": 50, "hmm": 47, "bg": "#ffffff"},

    # each plate is cropped to the treated block: its wells at `pitch`, padded by the block's padding
    "plate": {
        "pitch": 7.5, "wellSize": 0.72, "controlColor": "grey",
        "block": {"cols": 4, "rows": 4, "treatRows": 2, "pad": 1.3,
                  "radius": 1.2, "color": "grey", "stroke": 0.4,
                  "fillColor": "panel", "fill": 0.5},
    },

    "pairExp": {
        "x": 8, "y": 18, "gap": 50,
        "leftHeader": "Experiment\nRepeat A", "leftColor": "red",
        "rightHeader": "Experiment\nRepeat B", "rightColor": "teal",
        "ghost": {"n": 0, "dx": 2.2, "dy": -2.2, "fade": 0.55},
        "headerSize": 6, "headerWeight": 700, "headerColor": "ink",
        "headerGap": 2,
        "label": "Experimental\nreplication", "labelSize": 5.3,
        "labelGap": 4, "labelColor": "teal", "labelWeight": 700,
        "arrowColor": "ink", "arrowStroke": 0.6, "arrowHead": 2.6, "inset": 4,
    },

    "pairPred": {
        "x": 8, "y": 78, "gap": 50,          # = pairExp's, so each half sits under its repeat
        "leftHeader": "", "leftColor": "red",
        "rightHeader": "", "rightColor": "red",
        # Split-half is a random partition of Repeat A's cells, not its left vs
        # right half: 1 = left plate, 0 = right plate, one string per block row.
        # Hard-coded so rebuilds stay byte-identical; each half gets 4 perturbed
        # (rows 0-1) and 4 control (rows 2-3) wells.
        "split": ["1101", "0100", "1000", "0111"],
        "ghost": {"n": 2, "dx": 2.2, "dy": -2.2, "fade": 0.55},
        "headerSize": 6, "headerWeight": 700, "headerColor": "ink",
        "headerGap": 5,
        "label": "Split-half", "labelSize": 5.6, "labelGap": 5,
        "labelColor": "dustyRose", "labelWeight": 700,
        "arrowColor": "ink", "arrowStroke": 0.6, "arrowHead": 2.6, "inset": 4,
        "labelBelow": True,
    },

    # x = left plate centre (8 + 30.5 / 2); the run and the legs are read off pairPred
    "splitBus": {"x": 23.25, "y": 51, "drop": 10, "radius": 1.5, "stroke": 0.6,
                 "head": 2.8, "color": "ink"},

    # x = midway between the plate columns (8 + 30.5 + 50 / 2), under both arrows
    "replKey": {"x": 63.5, "y": 118, "size": 5.2, "color": "ink",
                "gap": 2.25, "dotGap": 0.5, "rowGap": 6.5,
                "dotScale": 0.55, "treatedText": "Perturbed cells",
                "controlText": "Control cells"},
}


# ════════════════════════ svg-path / text helpers ════════════════════════
# The studio writes the split-bus corner as an SVG path string with an elliptical
# arc. Rather than re-deriving it by hand, port the path string verbatim and
# convert the arc to cubics here.

def _arc_to_cubics(x1, y1, rx, ry, phi_deg, fa, fs, x2, y2):
    """SVG F.6.5 endpoint -> centre parameterisation, then cubic segments."""
    if rx == 0 or ry == 0 or (x1 == x2 and y1 == y2):
        return [("L", (x2, y2))]
    phi = np.radians(phi_deg)
    cp, sp = np.cos(phi), np.sin(phi)
    dx2, dy2 = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cp * dx2 + sp * dy2
    y1p = -sp * dx2 + cp * dy2
    rx, ry = abs(rx), abs(ry)
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1:
        s = np.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    co = np.sqrt(max(0.0, num / den)) * (-1 if fa == fs else 1)
    cxp, cyp = co * rx * y1p / ry, -co * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2.0
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2.0
    th1 = np.arctan2((y1p - cyp) / ry, (x1p - cxp) / rx)
    th2 = np.arctan2((-y1p - cyp) / ry, (-x1p - cxp) / rx)
    dth = th2 - th1
    if fs == 0 and dth > 0:
        dth -= 2 * np.pi
    elif fs == 1 and dth < 0:
        dth += 2 * np.pi
    nseg = max(1, int(np.ceil(abs(dth) / (np.pi / 2) - 1e-9)))
    out, d = [], dth / nseg
    k = 4.0 / 3.0 * np.tan(d / 4.0)
    for i in range(nseg):
        a, b = th1 + i * d, th1 + (i + 1) * d

        def pt(t):
            return (cx + rx * np.cos(t) * cp - ry * np.sin(t) * sp,
                    cy + rx * np.cos(t) * sp + ry * np.sin(t) * cp)

        def dpt(t):
            return (-rx * np.sin(t) * cp - ry * np.cos(t) * sp,
                    -rx * np.sin(t) * sp + ry * np.cos(t) * cp)

        p0, p3 = pt(a), pt(b)
        d0, d3 = dpt(a), dpt(b)
        out.append(("C", (p0[0] + k * d0[0], p0[1] + k * d0[1],
                          p3[0] - k * d3[0], p3[1] - k * d3[1], p3[0], p3[1])))
    return out


def svg_path(d):
    """Parse the M / L / A subset the studio emits into a mpl Path."""
    toks = d.replace(",", " ").split()
    verts, codes, i = [], [], 0
    cur = (0.0, 0.0)
    while i < len(toks):
        c = toks[i]
        i += 1
        if c in ("M", "L"):
            cur = (float(toks[i]), float(toks[i + 1]))
            i += 2
            verts.append(cur)
            codes.append(Path.MOVETO if c == "M" else Path.LINETO)
        elif c == "A":
            rx, ry, rot = float(toks[i]), float(toks[i + 1]), float(toks[i + 2])
            fa, fs = int(float(toks[i + 3])), int(float(toks[i + 4]))
            x2, y2 = float(toks[i + 5]), float(toks[i + 6])
            i += 7
            for kind, a in _arc_to_cubics(cur[0], cur[1], rx, ry, rot, fa, fs, x2, y2):
                if kind == "L":
                    verts.append(a)
                    codes.append(Path.LINETO)
                else:
                    verts += [(a[0], a[1]), (a[2], a[3]), (a[4], a[5])]
                    codes += [Path.CURVE4] * 3
            cur = (x2, y2)
        else:
            raise ValueError(f"svg_path: unsupported command {c!r}")
    return Path(verts, codes)


def rgba(color, opacity=1.0):
    return matplotlib.colors.to_rgba(color, opacity)


def rrect(ax, x, y, w, h, rx, **kw):
    """The studio's <rect rx=...>, rx > 0."""
    pa = FancyBboxPatch((x + rx, y + rx), max(0.0, w - 2 * rx),
                        max(0.0, h - 2 * rx),
                        boxstyle="round,pad=%g,rounding_size=%g" % (rx, rx),
                        mutation_scale=1, **kw)
    ax.add_patch(pa)
    return pa


def tx(ax, s, x, y, size, color, weight=400, anchor="start"):
    """Studio <text>: placed by BASELINE, so y transfers unchanged."""
    ha = {"start": "left", "middle": "center", "end": "right"}[anchor]
    return ax.text(x, y, s, fontsize=size, color=color,
                   fontweight=("bold" if weight >= 600 else "normal"), fontstyle="normal",
                   ha=ha, va="baseline")


# ════════════════════════ shared drawing helpers ════════════════════════
def headPoly(ax, x, y, ux, uy, head, color):
    px, py = -uy, ux
    w = head * 0.46
    ax.add_patch(Polygon([(x, y),
                          (x - ux * head + px * w, y - uy * head + py * w),
                          (x - ux * head - px * w, y - uy * head - py * w)],
                         closed=True, facecolor=color, edgecolor="none"))


def arrowNodes(ax, x1, y1, x2, y2, color, stroke, head, both):
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    sx = x1 + ux * head * 0.75 if both else x1
    sy = y1 + uy * head * 0.75 if both else y1
    ex, ey = x2 - ux * head * 0.75, y2 - uy * head * 0.75
    ax.add_line(plt.Line2D([sx, ex], [sy, ey], color=color, lw=stroke,
                           solid_capstyle="butt"))
    headPoly(ax, x2, y2, ux, uy, head, color)
    if both:
        headPoly(ax, x1, y1, -ux, -uy, head, color)


# ════════════════════════ plate geometry ════════════════════════
def plateSize():
    """How big one plate is drawn: the treated block's own outline, the wells
    at `pitch` padded by the block's padding. The pair layout, the arrows and
    the headers all size themselves off this."""
    pl = P["plate"]
    B = pl["block"]
    r = pl["pitch"] * pl["wellSize"] / 2.0
    e = 2 * (r + B["pad"])
    return {"w": max(1, B["cols"] - 1) * pl["pitch"] + e,
            "h": max(1, B["rows"] - 1) * pl["pitch"] + e,
            "r": r}


def croppedPlate(ax, ox, oy, color, keep=None):
    """Just the treated block: the small rectangle and its wells. `keep`, if
    given, is the set of (row, col) wells drawn; the rest stay blank."""
    pl, Z = P["plate"], plateSize()
    B = pl["block"]
    rrect(ax, ox, oy, Z["w"], Z["h"], B["radius"],
          facecolor=rgba(C(B["fillColor"]), B["fill"]),
          linewidth=B["stroke"], edgecolor=C(B["color"]))
    o0 = Z["r"] + B["pad"]
    for rr in range(B["rows"]):
        for cc in range(B["cols"]):
            if keep is not None and (rr, cc) not in keep:
                continue
            fill = C(color) if rr < B["treatRows"] else C(pl["controlColor"])
            ax.add_patch(Circle((ox + o0 + cc * pl["pitch"],
                                 oy + o0 + rr * pl["pitch"]), Z["r"],
                                facecolor=fill, edgecolor="none"))


def ghostFrames(ax, ox, oy, o):
    """The same cut taken again and again: empty frames of the block stacked
    behind a plate, one per extra resample, each fainter than the one in
    front.  Filled with the canvas, so a nearer frame hides the one behind."""
    G = o["ghost"]
    B, Z = P["plate"]["block"], plateSize()
    for i in range(G["n"], 0, -1):
        rrect(ax, ox + i * G["dx"], oy + i * G["dy"],
              Z["w"], Z["h"], B["radius"],
              facecolor=P["canvas"]["bg"],
              edgecolor=rgba(C(B["color"]), G["fade"] ** i),
              linewidth=B["stroke"])


# ════════════════════════ blocks ════════════════════════
def drawPlatePair(ax, gid):
    """Two plates side by side and the double arrow between them.  Everything
    that joins them is read off where the plates actually ended up."""
    o, Z = P[gid], plateSize()
    W, H = Z["w"], Z["h"]
    L, R = [0, 0], [W + o["gap"], 0]
    gx, gy = o["x"], o["y"]            # the pair's own group translation

    # each plate's share of a split: left gets the 1s, right the 0s, so the two
    # are disjoint and together cover the block
    split = o.get("split")
    B = P["plate"]["block"]
    if split and (len(split) != B["rows"] or set("".join(split)) - set("01")
                  or any(len(row) != B["cols"] for row in split)):
        raise ValueError("%s split must be %d rows of %d '0'/'1': %r"
                         % (gid, B["rows"], B["cols"], split))
    keeps = ([{(r, c) for r, row in enumerate(split)
               for c, v in enumerate(row) if v == bit} for bit in "10"]
             if split else [None, None])

    # each plate, over its own stack of resamples
    for pos, col, keep in ((L, o["leftColor"], keeps[0]), (R, o["rightColor"], keeps[1])):
        px, py = gx + pos[0], gy + pos[1]
        ghostFrames(ax, px, py, o)
        croppedPlate(ax, px, py, col, keep)

    # headers stack upward from their own plate
    hLead = o["headerSize"] * 1.2
    for pos, head in ((L, o["leftHeader"]), (R, o["rightHeader"])):
        if not head:
            continue
        hl = str(head).split("\n")
        for i, t in enumerate(hl):
            tx(ax, t, gx + pos[0] + W / 2.0,
               gy + pos[1] - o["headerGap"] - (len(hl) - 1 - i) * hLead,
               o["headerSize"], C(o["headerColor"]),
               o["headerWeight"], "middle")

    # the double arrow, level with the plates' middles
    y = H / 2.0
    arrowNodes(ax, gx + W + o["inset"], gy + y, gx + R[0] - o["inset"], gy + y,
               C(o["arrowColor"]), o["arrowStroke"], o["arrowHead"], True)

    # the label over the arrow, or under it (labelBelow)
    below = o.get("labelBelow", False)
    mx = (W + R[0]) / 2.0
    ly = y + o["labelGap"] if below else y - o["labelGap"]
    lines = o["label"].split("\n")
    lead = o["labelSize"] * 1.25
    ltop = ly + o["labelSize"] if below else ly - (len(lines) - 1) * lead
    for i, t in enumerate(lines):
        tx(ax, t, gx + mx, gy + ltop + i * lead, o["labelSize"],
           C(o["labelColor"]), o["labelWeight"], "middle")


def splitBusGeom():
    """Where the fork's two heads land: read off the pair it feeds (the top
    centre of each of pairPred's plates, clear of their resample stacks), so
    the run and the legs retime themselves."""
    o, Pp = P["splitBus"], P["pairPred"]
    Z = plateSize()
    clear = 3
    up = max(0.0, -(Pp["ghost"]["n"] * Pp["ghost"]["dy"]))
    top = Pp["y"] - up - clear
    q2x = Z["w"] + Pp["gap"]                 # the right plate's x within the pair
    run = (Pp["x"] + q2x + Z["w"] / 2.0) - o["x"]
    return run, max(1.0, top - o["y"] - o["drop"])


def drawSplitBus(ax):
    o = P["splitBus"]
    col = C(o["color"])
    run, leg = splitBusGeom()
    gx, gy = o["x"], o["y"]
    yB = o["drop"]
    sgn = -1 if run < 0 else 1
    r = min(abs(o["radius"]), abs(run) / 2.0, abs(leg) / 2.0)

    # the trunk, carrying straight on into the first box
    arrowNodes(ax, gx, gy, gx, gy + yB + leg, col, o["stroke"],
               o["head"], False)

    # the branch: across, round the corner, then down into the second box
    tipY = yB + leg
    d = "M 0 %g L %g %g" % (yB, run - sgn * r, yB)
    d += " A %g %g 0 0 %s %g %g" % (r, r, "1" if sgn > 0 else "0",
                                    run, yB + r)
    d += " L %g %g" % (run, tipY - o["head"] * 0.75)
    pth = svg_path(d).transformed(
        matplotlib.transforms.Affine2D().translate(gx, gy))
    ax.add_patch(PathPatch(pth, facecolor="none", edgecolor=col,
                           linewidth=o["stroke"]))
    headPoly(ax, gx + run, gy + tipY, 0, 1, o["head"], col)


def replColours():
    """The distinct colours the plates are actually drawn in."""
    out = []
    for k in ("pairExp", "pairPred"):
        for c in (P[k]["leftColor"], P[k]["rightColor"]):
            if c not in out:
                out.append(c)
    return out


def drawReplKey(ax):
    """The colour key, read off the figure, at the plates' own well size: one
    entry per row (control first), each centred on the key's x."""
    o, pl = P["replKey"], P["plate"]
    r = plateSize()["r"] * o["dotScale"]
    cy = -o["size"] * 0.36

    def entry(cols, text):
        """One entry's dots and text, as offsets, plus its own width."""
        dots, x = [], 0.0
        for i, c in enumerate(cols):
            dots.append((x + r, cy, C(c)))
            x += 2 * r + (o["dotGap"] if i < len(cols) - 1 else 0)
        x += o["gap"]
        return {"dots": dots, "tx": x, "text": text,
                "w": x + text_width(ax.figure, text, o["size"], 400)}

    parts = [entry([pl["controlColor"]], o["controlText"]),
             entry(replColours(), o["treatedText"])]
    gx, gy = o["x"], o["y"]
    for i, p in enumerate(parts):
        dx, dy = -p["w"] / 2.0, i * o["rowGap"]
        for cxx, cyy, col in p["dots"]:
            ax.add_patch(Circle((gx + dx + cxx, gy + dy + cyy), r,
                                facecolor=col, edgecolor="none"))
        tx(ax, p["text"], gx + dx + p["tx"], gy + dy, o["size"], C(o["color"]))


# ════════════════════════ draw ════════════════════════
def draw(ax):
    """The whole panel, in the studio's BUILDERS["replication"] order."""
    c = P["canvas"]
    ax.add_patch(Rectangle((0, 0), c["wmm"] * MM, c["hmm"] * MM, facecolor=c["bg"],
                           edgecolor="none", zorder=-10))
    drawPlatePair(ax, "pairExp")
    drawSplitBus(ax)
    drawPlatePair(ax, "pairPred")
    drawReplKey(ax)
