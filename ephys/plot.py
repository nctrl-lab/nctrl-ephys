"""Paper style for matplotlib, with an exact axes size.

Sizes are in millimeters. mystyle and mysubplots pin the *axes* size and grow the
figure around it by a fixed margin, so panels made with the same ratio come out
the same size and line up in Illustrator.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.transforms import ScaledTranslation

MM_PER_INCH = 25.4

# Full-page panel width. Cell 174, Nature 180, Science 184.
FULL_WIDTH_MM = 174.0

# Room for the labels: left, right, bottom, top.
MARGINS_MM = (13.0, 4.0, 10.0, 4.0)

# Okabe-Ito, color-blind safe. https://jfly.uni-koeln.de/color/
COLOR_CYCLE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
]

FONT_STACK = [
    "Helvetica",
    "Helvetica Neue LT Std",
    "Helvetica Neue",
    "Arial",
    "Liberation Sans",
    "Nimbus Sans",
]


def _axes_size_mm(width_ratio, height_ratio, width_mm, height_mm):
    return (
        FULL_WIDTH_MM * width_ratio if width_mm is None else width_mm,
        FULL_WIDTH_MM * height_ratio if height_mm is None else height_mm,
    )


def _pin_font():
    path = font_manager.findfont(font_manager.FontProperties(family=["sans-serif"]))
    found = font_manager.FontProperties(fname=path).get_name()
    mpl.rcParams["font.family"] = found
    print(f"Font: {found} ({path})")


def mystyle(
    width_ratio=1, height_ratio=1, *, width_mm=None, height_mm=None, margins_mm=None
):
    """Set the paper style and the axes size.

    width_ratio and height_ratio give the axes size in units of FULL_WIDTH_MM
    (equal values make a square axes); width_mm and height_mm set it in mm
    instead. The figure then grows by a fixed margin for the labels. For a grid
    of panels, use mysubplots. Examples:

        mystyle(0.2, 0.2); fig, ax = plt.subplots()      # square 34.8 mm axes
        mystyle(width_mm=40, height_mm=25)               # 40 x 25 mm axes
        mystyle(0.2, 0.2, margins_mm=(16, 4, 16, 4))     # more room left/bottom

    * Do not use tight_layout() after this. It will change the axes size and
      misalign panels.
    """
    plt.rcdefaults()

    axes_w, axes_h = _axes_size_mm(width_ratio, height_ratio, width_mm, height_mm)
    ml, mr, mb, mt = MARGINS_MM if margins_mm is None else margins_mm
    fig_w, fig_h = ml + axes_w + mr, mb + axes_h + mt

    mpl.rcParams["figure.figsize"] = [fig_w / MM_PER_INCH, fig_h / MM_PER_INCH]
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["figure.autolayout"] = False
    mpl.rcParams["figure.subplot.left"] = ml / fig_w
    mpl.rcParams["figure.subplot.right"] = (ml + axes_w) / fig_w
    mpl.rcParams["figure.subplot.bottom"] = mb / fig_h
    mpl.rcParams["figure.subplot.top"] = (mb + axes_h) / fig_h

    SMALL, MEDIUM, BIGGER = 6, 6, 7
    mpl.rcParams.update(
        {
            # Naming one font here would skip the list below.
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": MEDIUM,
            "figure.labelsize": BIGGER,
            "axes.titlesize": MEDIUM,
            "axes.labelsize": MEDIUM,
            "xtick.labelsize": SMALL,
            "ytick.labelsize": SMALL,
            "legend.fontsize": SMALL,
            "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "xtick.minor.size": 1,
            "ytick.minor.size": 1,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            "axes.labelpad": 2,
            "axes.titlepad": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 0.75,
            "lines.markersize": 3,
            "lines.markeredgewidth": 0,
            # U+2212 comes out wrong in Illustrator.
            "axes.unicode_minus": False,
            "savefig.transparent": True,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    _pin_font()


def mysubplots(
    width_ratio=1,
    height_ratio=1,
    nrows=1,
    ncols=1,
    *,
    width_mm=None,
    height_mm=None,
    width_ratios=None,
    height_ratios=None,
    gap_mm=2.5,
    margins_mm=None,
    **kwargs,
):
    """Do mystyle and plt.subplots together, with exact sizing.

    width_ratio and height_ratio set the size of the whole block of panels. The
    panels share this block: a gap_mm is left between them, and the rest is split
    by width_ratios/height_ratios. Extra kwargs (sharey, ...) go to plt.subplots.
    Returns (fig, ax) or (fig, axs). Example:

        fig, axs = mysubplots(0.3, 0.15, ncols=3, width_ratios=[2, 2, 1], sharey=True)

    * gap_mm is pure white space. Panels with their own y axis need a gap wide
      enough for the labels.
    """
    old = {"gap": "gap_mm", "margins": "margins_mm"}.keys() & kwargs.keys()
    if old:
        raise TypeError(
            f"{', '.join(sorted(old))}: now given in mm, not inches. Use "
            f"{', '.join(sorted(k + '_mm' for k in old))} and multiply by 25.4"
        )

    mystyle(
        width_ratio,
        height_ratio,
        width_mm=width_mm,
        height_mm=height_mm,
        margins_mm=margins_mm,
    )
    wr = width_ratios or [1] * ncols
    hr = height_ratios or [1] * nrows
    kwargs.pop("figsize", None)
    gridspec = {
        **kwargs.pop("gridspec_kw", {}),
        "width_ratios": wr,
        "height_ratios": hr,
    }
    fig, axs = plt.subplots(nrows, ncols, gridspec_kw=gridspec, **kwargs)

    # wspace is a fraction of the average panel width, so the gap stays exact.
    block_w, block_h = _axes_size_mm(width_ratio, height_ratio, width_mm, height_mm)
    fig.subplots_adjust(
        wspace=gap_mm * ncols / (block_w - (ncols - 1) * gap_mm) if ncols > 1 else 0,
        hspace=gap_mm * nrows / (block_h - (nrows - 1) * gap_mm) if nrows > 1 else 0,
    )
    return fig, axs


def panel_label(label, ax=None, *, dx_mm=-5.0, dy_mm=1.0, **kwargs):
    """Put a panel letter at a fixed offset from the top left of the axes.

    The offset is in mm, not in data or axes fractions, so the letters sit in the
    same place on every panel whatever its size. Extra kwargs go to ax.text.
    Returns the Text. Example:

        fig, axs = mysubplots(0.4, 0.2, ncols=2, gap_mm=12)
        for ax, letter in zip(axs, "AB"):
            panel_label(letter, ax)
    """
    if ax is None:
        ax = plt.gca()
    kwargs.setdefault("fontsize", 8)
    kwargs.setdefault("fontweight", "bold")
    kwargs.setdefault("ha", "right")
    kwargs.setdefault("va", "bottom")
    transform = ax.transAxes + ScaledTranslation(
        dx_mm / MM_PER_INCH, dy_mm / MM_PER_INCH, ax.figure.dpi_scale_trans
    )
    return ax.text(0, 1, label, transform=transform, **kwargs)
