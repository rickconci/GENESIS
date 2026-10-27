"""Nature's Extended Data rules, as code: one font, millimetre axes, and the checks a figure must pass
before it is saved.

Nature does not restyle Extended Data figures (its Extended Data guide), so the rules are
enforced here rather than trusted to the eye:
    page        <= 180 mm wide x 170 mm tall (the legend is set below)
    text        Helvetica or Arial, 5-7 pt; panel letters 8 pt bold lowercase
    lines       0.25-1 pt
    upload      TIFF/JPEG/EPS, RGB, 300 dpi, <= 10 MB (we also write a PDF for the LaTeX draft)

Each Extended Data draw saves its page with ``save`` to core_deliverables/figureN/ed/<name> (or a scratch folder
when checked). The font is the vendored Liberation Sans in src/figures/fonts/ (FONTS). The only state is
src.figures.page.W, which main_figure_style sets to this page's 180 mm for the main figures' panels.
The palette is Figure 2's (src.figures.palette).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines  # noqa: F401  (mpl.lines.Line2D, the stroke check)
import matplotlib.patches  # noqa: F401  (mpl.patches.Patch, the stroke check)
import matplotlib.pyplot as plt
import matplotlib.text  # noqa: F401  (mpl.text.Text, the text check)
import numpy as np
from matplotlib import font_manager

from src.figures import page
from src.figures.palette import INK

# Page order: the order of first citation in the main text (results.md), which is Nature's
# numbering. Builder name -> its folder; the number is the folder's prefix.
ED = {"ed_pbmc_wells": "ED_pbmc_wells",     # Figure 2's; no number
      "ed_pert_generalization": "ED02_pert_generalization",
      "ed_response_geometry": "ED03_response_geometry",
      "ed_data_efficiency": "ED_scaling_pert_split",  # folder and files share this name
      "ed_program_summary": "ED_program_summary",  # folder and files share this name
      "ed_conditioning": "ED08_conditioning",
      "ed_objective": "ED09_objective",          # the loss: ablations 3-4
      "ed_cells_per_combination": "ED_12cells_per_condition",  # folder and files share this name
      "ed_split_coverage": "ED_split_coverage",  # the four splits' grids; no number
      "ed_cells_per_condition": "ED15_cells_per_condition",  # Fig. 3a's training sets, cells per condition
      "ed_pathway_jakstat_tnfa": "ED16_pathway_jakstat_tnfa",
      "ed_pathway_mapk_pi3k": "ED17_pathway_mapk_pi3k",
      "ed_p53_relatedness": "ED18_p53_relatedness",
      "ed_tahoe_hallmark_gsea": "ED_tahoe_hallmark_gsea",  # Figure 4's; no number
      "ed_tahoe_hallmark_heatmap": "ED_tahoe_hallmark_heatmap",  # Figure 4's; four pages, no number
      "ed_tahoe_seriation": "ED_tahoe_seriation"}  # Figure 4's; two pages, no number
W, H_MAX = 180.0, 170.0                       # mm: Nature's Extended Data page (a constant; page.W is the state)
DPI = 300
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

# ----------------------------------------------------------------------------
# The font: Liberation Sans has Arial's metrics (SIL OFL, vendored in FONTS).
# It lacks only the superscript minus and zero (U+207B, U+2070), so those two glyphs fall back to
# DejaVu Sans; everything else is Liberation Sans. Registered when a figure is styled, not at import.
# ----------------------------------------------------------------------------
FONT = "Liberation Sans"


FONTS = Path(__file__).resolve().parent / "fonts"


def _load_font() -> None:
    fonts = sorted(FONTS.glob("LiberationSans-*.ttf"))
    if not fonts:
        raise FileNotFoundError(f"no LiberationSans-*.ttf in {FONTS}")
    for f in fonts:
        font_manager.fontManager.addfont(str(f))
    path = font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
    if "LiberationSans" not in path:
        raise RuntimeError(f"{FONT} did not load from {FONTS} (got {path})")


def style() -> None:
    """An Extended Data figure of its own (no main-figure twin). Starts from a fresh process's settings."""
    _load_font()
    page.W = W
    mpl.rc_file_defaults()
    plt.rcParams.update({
        "font.family": [FONT, "DejaVu Sans"],
        "font.size": 6.0, "axes.labelsize": 6.0, "axes.titlesize": 6.5,
        "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0, "hatch.linewidth": 0.35,
        "axes.edgecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42,
    })


def main_figure_style(style) -> None:
    """A main figure's look (``style``, e.g. src.figures.page.style) in Liberation Sans, Nature's Arial for
    Extended Data, with the main figures' panels drawn at this page's width (page.W)."""
    _load_font()
    style()
    plt.rcParams.update({"font.family": [FONT, "DejaVu Sans"]})
    page.W = W


# ----------------------------------------------------------------------------
# Placing a main-figure panel on this page (Fig. 4's functions make their own figure)
# ----------------------------------------------------------------------------

class _Panel:
    """A main-figure panel's own figure, mapped onto a region of the page: the fractions it passes to
    add_axes and text are of ITS size, so they are rescaled into the page's. Everything else (colorbar,
    dpi transforms) is the page's. (A matplotlib SubFigure will not do: its add_axes takes fractions
    of the whole page.)"""

    def __init__(self, page_fig, x: float, y: float, w: float, h: float):
        self._page, self.size_mm = page_fig, (w, h)
        ph = page_fig.get_figheight() * 25.4
        self._x0, self._y0 = x / W, (ph - y - h) / ph          # the panel's bottom-left, in page fractions
        self._sx, self._sy = w / W, h / ph

    def _xy(self, x: float, y: float) -> tuple[float, float]:
        return self._x0 + x * self._sx, self._y0 + y * self._sy

    def add_axes(self, rect, **kw):
        l, b, w, h = rect
        return self._page.add_axes([*self._xy(l, b), w * self._sx, h * self._sy], **kw)

    def text(self, x: float, y: float, s: str, **kw):
        if "transform" in kw:
            raise ValueError("a placed panel's fig.text must use figure fractions")
        return self._page.text(*self._xy(x, y), s, **kw)

    def get_figheight(self) -> float:          # helpers that lay out in mm ask the figure its size
        return self.size_mm[1] / 25.4

    def get_figwidth(self) -> float:
        return self.size_mm[0] / 25.4

    def __getattr__(self, name):
        return getattr(self._page, name)


@contextlib.contextmanager
def placed(page_fig, x: float, y: float):
    """Draw a main-figure panel that makes its own plt.figure(figsize=...) (Fig. 4's functions do)
    into this page instead, at its own size with its top-left corner at (x, y) mm."""
    made, real = [], plt.figure

    def figure(*args, figsize=None, **kw):
        if made:
            raise RuntimeError("placed() expects one figure per panel")
        if figsize is None:
            raise ValueError("placed() needs the panel's figsize")
        w, h = figsize[0] * 25.4, figsize[1] * 25.4
        ph = page_fig.get_figheight() * 25.4
        if x + w > W + 0.01 or y + h > ph + 0.01:
            raise ValueError(f"a {w:.0f} x {h:.0f} mm panel at ({x}, {y}) leaves the {W} x {ph:.0f} mm page")
        made.append(_Panel(page_fig, x, y, w, h))
        return made[0]
    plt.figure = figure
    try:
        yield made
    finally:
        plt.figure = real


# ----------------------------------------------------------------------------
# Small drawing helpers (millimetres, measured from the top left)
# ----------------------------------------------------------------------------

def figure(h: float) -> plt.Figure:
    if h > H_MAX:
        raise ValueError(f"figure is {h} mm tall; Nature allows {H_MAX} mm")
    return plt.figure(figsize=(W / 25.4, h / 25.4), facecolor="white")


def axes_mm(fig, x: float, y: float, w: float, h: float, **kw):
    fh = fig.get_figheight() * 25.4
    return fig.add_axes([x / W, 1 - (y + h) / fh, w / W, h / fh], **kw)


def letter(fig, lab: str, x: float, y: float) -> None:
    fh = fig.get_figheight() * 25.4
    fig.text(x / W, 1 - y / fh, lab, fontsize=8, fontweight="bold", ha="left", va="top")


def banner_line(fig, y: float, name: str, task: str, ref: str = "", ref_color: str = INK,
                x: float = 4.0, w: float = 172.0) -> None:
    """Fig. 2's banner on one line: dataset (bold) · task · reference (bold, in its colour), so a row of another
    dataset or split reads apart at a glance."""
    parts = [(name, INK, "bold"), (f"  ·  {task}" + ("  ·  " if ref else ""), INK, "normal")]
    page.banner(fig, x, y, w, 5.0, [parts + ([(ref, ref_color, "bold")] if ref else [])])


def p_text(p: float) -> str:
    """0.034, 0.11, 5×10⁻⁸; a P that underflows double precision is a bound, not 0."""
    if p < 1e-300:
        return "< 10⁻³⁰⁰"
    if p >= 0.001:
        return f"{p:.2g}"
    mant, exp = f"{p:.0e}".split("e")
    return f"{mant}×10{str(int(exp)).translate(SUPERSCRIPT)}"


# ----------------------------------------------------------------------------
# The checks: Nature will print exactly what we send, so a violation is an error, not a warning
# ----------------------------------------------------------------------------

def _texts(fig) -> list:
    """Every visible, non-empty text the renderer draws, tick labels included."""
    fig.canvas.draw()
    # Matplotlib keeps tick labels for ticks out of view, and for axes switched off; only
    # _update_ticks knows which a render shows (the same call Axis.get_tightbbox makes).
    axes = [axis for ax in fig.axes for axis in (ax.xaxis, ax.yaxis)]
    every = {id(lab) for axis in axes for t in axis.get_major_ticks() + axis.get_minor_ticks()
             for lab in (t.label1, t.label2)}
    shown = [lab for axis in axes if axis.axes.axison and axis.get_visible()
             for t in axis._update_ticks() for lab in (t.label1, t.label2) if lab.get_visible()]
    texts = [t for t in fig.findobj(mpl.text.Text) if id(t) not in every] + shown
    return [t for t in texts if t.get_visible() and t.get_text().strip()]


def check(fig) -> None:
    r = fig.canvas.get_renderer()
    texts = _texts(fig)
    for t in texts:
        s, txt = t.get_fontsize(), t.get_text()
        is_letter = len(txt) == 1 and txt.islower() and s == 8 and t.get_fontweight() in ("bold", 700)
        if not (5.0 <= s <= 7.0 or is_letter):
            raise RuntimeError(f"text {txt!r} is {s} pt; Nature allows 5-7 pt (panel letters 8 pt bold)")
    boxes, tol = [t.get_window_extent(r) for t in texts], 0.5 * fig.dpi / 72
    for i, j in zip(*np.triu_indices(len(texts), k=1)):
        a, b = boxes[i], boxes[j]
        if min(a.x1, b.x1) - max(a.x0, b.x0) > tol and min(a.y1, b.y1) - max(a.y0, b.y0) > tol:
            raise RuntimeError(f"labels collide: {texts[i].get_text()!r} and {texts[j].get_text()!r}")
    for art in fig.findobj(lambda a: isinstance(a, (mpl.lines.Line2D, mpl.patches.Patch)) and a.get_visible()):
        lw = art.get_linewidth()
        stroked = lw > 0 and (not isinstance(art, mpl.patches.Patch) or mpl.colors.to_rgba(art.get_edgecolor())[3] > 0)
        if isinstance(art, mpl.lines.Line2D) and art.get_linestyle() in ("None", "", " "):
            stroked = False
        if isinstance(art, mpl.lines.Line2D) and mpl.colors.to_hex(art.get_color()) == "#ffffff":
            stroked = False                     # a white mask (Fig. 2's axis-break gap), not a visible stroke
        if stroked and not 0.25 <= lw <= 1.0 and art is not fig.patch:
            raise RuntimeError(f"{type(art).__name__} line weight {lw} pt; Nature allows 0.25-1 pt")
    # fig.get_tightbbox does not report text past the canvas edge, so test every text and axes box
    canvas = fig.bbox
    for art, box in [(t.get_text(), b) for t, b in zip(texts, boxes)] + [(ax, ax.get_window_extent(r)) for ax in fig.axes]:
        over = np.r_[canvas.x0 - box.x0, canvas.y0 - box.y0, box.x1 - canvas.x1, box.y1 - canvas.y1] / fig.dpi * 25.4
        if (over > 0.1).any():
            raise RuntimeError(f"{art!r} runs off the canvas by {over.round(2)} mm (left, bottom, right, top)")


def clamp_strokes(fig) -> None:
    """The main figures draw a few strokes at 1.1 pt (Fig. 2b's GENESIS and repeat curves); Nature caps
    ED lines at 1 pt and does not restyle ED figures. Clamp visible strokes to 1 pt, saying so, rather
    than edit the main-figure code: a 0.1 pt difference no reader can see."""
    for art in fig.findobj(mpl.lines.Line2D):
        drawn = art.get_visible() and art.get_linestyle() not in ("None", "", " ")
        if drawn and art.get_linewidth() > 1.0 and mpl.colors.to_hex(art.get_color()) != "#ffffff":
            print(f"  clamped a {art.get_linewidth()} pt line to 1 pt (Nature ED maximum)")
            art.set_linewidth(1.0)


# ----------------------------------------------------------------------------
# Writing the files: each figure in its numbered folder, its plotted numbers beside it
# ----------------------------------------------------------------------------

def file_stem(name: str) -> str:
    """ED05 for a numbered folder. A folder without an ED number is its own file name (ED_split_coverage),
    so a figure whose number is not assigned is not labelled with a slice of that name."""
    if name not in ED:
        raise KeyError(f"{name} has no Extended Data number; add it to ED in src/figures/ed_page.py")
    prefix = ED[name][:4]
    return prefix if prefix[:2] == "ED" and prefix[2:4].isdigit() else ED[name]


def folder(name: str, root: Path) -> Path:
    """A figure's folder, e.g. root/ED09_objective."""
    if name not in ED:
        raise KeyError(f"{name} has no Extended Data number; add it to ED in src/figures/ed_page.py")
    return root / ED[name]


def locate(name: str, ed_root: Path) -> Path:
    """A figure's folder anywhere in the whole Extended Data tree: ed_root/FigureN/EDnn_x or ed_root/EDnn_x.
    Exactly one, or raise."""
    hits = [p for p in (ed_root / ED[name], *ed_root.glob(f"Figure*/{ED[name]}")) if p.is_dir()]
    if len(hits) != 1:
        raise FileNotFoundError(f"{ED[name]}: {len(hits)} folders under {ed_root}")
    return hits[0]


def save(fig, stem: Path) -> None:
    """<stem>.pdf and <stem>.png at Nature's 300 dpi, checked first: a violation is never written."""
    clamp_strokes(fig)
    check(fig)
    page.save(fig, stem, DPI)


def write_table(df, name: str, root: Path, dest: Path | None = None) -> None:
    """Every plotted number, beside the figure: <folder>/source_data.csv."""
    out = dest if dest is not None else folder(name, root)
    df.to_csv(out / "source_data.csv", index=False, float_format="%.6g")
