"""A print-size page: millimetres from the top left, one type style, files that rebuild byte for
byte, and checks that refuse what Nature would (text under 5 pt, overlapping labels, drawing off the canvas).

State, said once: ``W`` is the page width in mm. Main figures are 183 (Nature's double column); an Extended
Data page sets ``page.W = 180`` (src.figures.ed_page). Everything here, and every panel, reads it at call time.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib as mpl
import matplotlib.text  # noqa: F401  (mpl.text.Text, for the text check)
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from PIL import Image

from src.figures.palette import BANNER, INK

W = 183.0                                    # page width, mm (see the module docstring)
# Empty dates, so a rebuild is byte-identical. Callers that save svg/pdf/png themselves.
NO_DATE = {"svg": {"Date": None}, "pdf": {"CreationDate": None}, "png": {}}


def style(label: float = 6.0, tick: float = 5.5) -> None:
    """Nature's print sizes: 6 pt type, 6.5 pt titles, 0.5 pt axes. ``label`` sizes axis labels and ``tick``
    tick labels and keys (Figure 2 sets 6.5 and 6). Fonts embed as TrueType, so the PDF text stays editable.
    Starts from a fresh process's settings, so figures drawn in one process don't leak style into each other."""
    global W
    W = 183.0
    mpl.rc_file_defaults()
    mpl.rcParams.update({
        # matplotlib ships DejaVu Sans, so every machine draws the same pixels; naming Helvetica or Arial
        # first would pick them up on macOS or Windows and move every glyph
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 6.0, "axes.labelsize": label, "axes.titlesize": 6.5,
        "xtick.labelsize": tick, "ytick.labelsize": tick, "legend.fontsize": tick,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0, "hatch.linewidth": 0.35,
        "axes.edgecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })


def axes_mm(fig, x: float, y: float, w: float, h: float, **kw):
    fh = fig.get_figheight() * 25.4
    return fig.add_axes([x / W, 1 - (y + h) / fh, w / W, h / fh], **kw)


def letter(fig, lab: str, x: float, y: float) -> None:
    fh = fig.get_figheight() * 25.4
    fig.text(x / W, 1 - y / fh, lab, fontsize=8, fontweight="bold", color=INK,
             ha="left", va="top")


def banner(fig, x: float, y: float, w: float, h: float, lines: list) -> None:
    """Grey header; each line is [(text, colour, weight), ...] so one line can mix weights."""
    bx = axes_mm(fig, x, y, w, h, facecolor=BANNER)
    bx.set_xticks([])
    bx.set_yticks([])
    for s in bx.spines.values():
        s.set_visible(False)
    rows = [HPacker(children=[TextArea(t, textprops=dict(color=c, fontweight=wt, fontsize=6.5))
                              for t, c, wt in line], align="baseline", pad=0, sep=0)
            for line in lines]
    bx.add_artist(AnchoredOffsetbox(loc="center", frameon=False, pad=0, borderpad=0,
                                    child=VPacker(children=rows, align="center", pad=0, sep=2)))


def save(fig, stem: Path, dpi: int = 600) -> None:
    """<stem>.pdf and <stem>.png (RGB, ``dpi``), refused if anything is drawn off the canvas. No timestamps, so
    a rebuild is byte-identical."""
    check_canvas(fig)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.pdf", dpi=dpi, facecolor="white", metadata={"CreationDate": None})
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", metadata={"Software": None})
    Image.open(buf).convert("RGB").save(f"{stem}.png", dpi=(dpi, dpi))     # Nature wants RGB; matplotlib writes RGBA
    print(f"wrote {stem}.{{pdf,png}}")


def check_canvas(fig) -> None:
    """Nothing drawn may leave the canvas (a clipped label is silent otherwise). The tight
    box counts only what is rendered, e.g. not tick labels outside an axis's view."""
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    over = np.r_[-tight.x0, -tight.y0, tight.x1 - fig.get_figwidth(), tight.y1 - fig.get_figheight()] * 25.4
    if (over > 0.25).any():
        raise RuntimeError(f"drawing runs off the canvas by {over.round(2)} mm (left, bottom, right, top)")


def check_text(fig) -> tuple[float, float]:
    """No drawn label under Nature's 5 pt, no two labels overlapping, no bar label outside
    its axis. A refreshed input moves labels, and nothing else would report a collision.
    Returns the (min, max) font size drawn."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axes = [(ax, axis) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis)]
    # Matplotlib keeps undrawn labels for ticks outside the view; only _update_ticks knows
    # which ticks a render shows (the same call Axis.get_tightbbox makes).
    every = {id(lab) for _ax, axis in axes for t in axis.get_major_ticks() + axis.get_minor_ticks()
             for lab in (t.label1, t.label2)}
    shown = [lab for _ax, axis in axes for t in axis._update_ticks() for lab in (t.label1, t.label2)
             if lab.get_visible()]
    texts = [t for t in fig.findobj(mpl.text.Text) if id(t) not in every] + shown
    texts = [t for t in texts if t.get_visible() and t.get_text().strip()]
    sizes = [t.get_fontsize() for t in texts]
    if min(sizes) < 5:
        raise RuntimeError(f"text under 5 pt: {[t.get_text() for t in texts if t.get_fontsize() < 5]}")
    boxes, tol = [t.get_window_extent(r) for t in texts], 0.5 * fig.dpi / 72
    for i, j in zip(*np.triu_indices(len(texts), k=1)):
        a, b = boxes[i], boxes[j]
        if min(a.x1, b.x1) - max(a.x0, b.x0) > tol and min(a.y1, b.y1) - max(a.y0, b.y0) > tol:
            raise RuntimeError(f"labels collide: {texts[i].get_text()!r} and {texts[j].get_text()!r}")
    for ax in fig.axes:
        box = ax.get_window_extent(r).padded(tol)
        for t in ax.texts:
            e = t.get_window_extent(r)
            if not (box.x0 <= e.x0 and e.x1 <= box.x1 and box.y0 <= e.y0 and e.y1 <= box.y1):
                raise RuntimeError(f"label {t.get_text()!r} leaves its axis")
    return min(sizes), max(sizes)
