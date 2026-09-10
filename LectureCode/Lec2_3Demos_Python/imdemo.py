"""
imdemo.py
Shared helpers for the Lecture 2 and 3 demos.

What is in here:
    sfigure      -> a numbered figure that does not steal focus
    imagesc      -> display an array, one cell per pixel, axis image
    demosaic     -> Malvar-He-Cutler gradient-corrected interpolation
    show_numbers -> a small patch drawn as cells with the value written in

Dependencies: numpy, scipy, matplotlib, pillow.  Everything here is a few
lines of array work.

Set the environment variable IMDEMO_SAVE to a directory to write every figure to
PNG instead of opening windows:
    IMDEMO_SAVE=figs python x01raw_image_demo.py
"""

import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
from scipy.ndimage import convolve

HERE = Path(__file__).resolve().parent
_SAVE_DIR = os.environ.get("IMDEMO_SAVE")
if _SAVE_DIR:
    mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Dark background, to match the lecture slides.  The plots below use yellow
# lines for the same reason.
plt.style.use("dark_background")
mpl.rcParams.update({
    "figure.figsize": (7.5, 5.5),
    "axes.grid": False,
    "image.interpolation": "nearest",
    "font.size": 10,
})

YEL = (1.0, 0.85, 0.15)


# ----------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------
def read_raw(path, w, h, dtype):
    """Headerless, row-major, little-endian raw file -> (h, w) array.

    A relative path is taken relative to this file's directory, not the
    working directory, so the demos run the same from anywhere.

    numpy reshapes in row-major order, so the (h, w) reshape below puts each
    row of the file on a row of the array with no transpose needed.
    """
    path = Path(path)
    if not path.is_absolute():
        path = HERE / path
    dt = np.dtype(dtype).newbyteorder("<")
    a = np.fromfile(path, dtype=dt)
    expect = w * h
    if a.size != expect:
        raise ValueError(f"{path}: got {a.size} samples, expected {expect}")
    return a.reshape(h, w)


# ----------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------
def sfigure(n, figsize=None):
    """Numbered figure, cleared and ready to draw on.  Returns (fig, ax)."""
    fig = plt.figure(n, figsize=figsize)
    fig.clf()
    return fig, fig.add_subplot(111)


def imagesc(ax, img, clim=None, cmap="gray"):
    """imagesc + axis image.  Handles 2D (with colormap) and HxWx3 (as colour).

    2D data is scaled to the display range (the full range of the data, or
    clim if given); colour data is assumed to already be in [0 1].
    """
    img = np.asarray(img)
    if img.ndim == 3:
        h = ax.imshow(np.clip(img, 0, 1), aspect="equal")
    else:
        vmin, vmax = (None, None) if clim is None else clim
        h = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    return h


def stem(ax, x, y, color=YEL, markersize=3, marker=True, label=None):
    """ax.stem with the colours set, since a StemContainer has no set_color."""
    c = ax.stem(x, y, linefmt="-", markerfmt="o", basefmt=" ", label=label)
    c.stemlines.set(color=color, linewidth=1.0)
    if marker:
        c.markerline.set(color=color, markerfacecolor=color,
                         markersize=markersize)
    else:
        c.markerline.set_visible(False)
    return c


def show():
    """End of script: open the windows, or write PNGs if IMDEMO_SAVE is set."""
    if not _SAVE_DIR:
        plt.show()
        return
    os.makedirs(_SAVE_DIR, exist_ok=True)
    for n in plt.get_fignums():
        plt.figure(n).savefig(
            os.path.join(_SAVE_DIR, f"fig{n:02d}.png"), dpi=110)
    plt.close("all")


# ----------------------------------------------------------------------
# Colour filter array
# ----------------------------------------------------------------------
def _cfa_offsets(pattern):
    """(row, col) of the R, G1, G2, B sites within the 2x2 tile, 0-based."""
    return {
        "rggb": ((0, 0), (0, 1), (1, 0), (1, 1)),
        "bggr": ((1, 1), (0, 1), (1, 0), (0, 0)),
        "grbg": ((0, 1), (0, 0), (1, 1), (1, 0)),
        "gbrg": ((1, 0), (0, 0), (1, 1), (0, 1)),
    }[pattern.lower()]


def cfa_masks(h, w, pattern="rggb"):
    """Boolean R, G, B site masks for an h-by-w sensor."""
    (rr, rc), (g1r, g1c), (g2r, g2c), (br, bc) = _cfa_offsets(pattern)
    R = np.zeros((h, w), bool)
    G = np.zeros((h, w), bool)
    B = np.zeros((h, w), bool)
    R[rr::2, rc::2] = True
    G[g1r::2, g1c::2] = True
    G[g2r::2, g2c::2] = True
    B[br::2, bc::2] = True
    return R, G, B


def cfa_mask(h, w, pattern="rggb"):
    """Unit-scale RGB colour of the filter sitting over each pixel."""
    R, G, B = cfa_masks(h, w, pattern)
    return np.stack([R, G, B], axis=-1).astype(float)


# Malvar-He-Cutler gradient-corrected linear interpolation.  Four 5x5 kernels,
# applied to the mosaic itself rather than to the sparse per-channel planes.
_GR_GB = np.array([
    [0, 0, -1, 0, 0],
    [0, 0, 2, 0, 0],
    [-1, 2, 4, 2, -1],
    [0, 0, 2, 0, 0],
    [0, 0, -1, 0, 0]], float) / 8.0

_Rg_RB = np.array([
    [0, 0, 0.5, 0, 0],
    [0, -1, 0, -1, 0],
    [-1, 4, 5, 4, -1],
    [0, -1, 0, -1, 0],
    [0, 0, 0.5, 0, 0]], float) / 8.0

_Rg_BR = _Rg_RB.T

_Rb_BB = np.array([
    [0, 0, -1.5, 0, 0],
    [0, 2, 0, 2, 0],
    [-1.5, 0, 6, 0, -1.5],
    [0, 2, 0, 2, 0],
    [0, 0, -1.5, 0, 0]], float) / 8.0


def demosaic(bayer, pattern="rggb"):
    """Malvar-He-Cutler demosaic.  Returns HxWx3 in the input's dtype range.

    Integer input comes back as the same integer type, clipped to that type's range; float input is left in float.
    """
    a = np.asarray(bayer)
    dt = a.dtype
    cfa = a.astype(np.float64)
    h, w = cfa.shape

    Rm, Gm, Bm = cfa_masks(h, w, pattern)
    R = cfa * Rm
    G = cfa * Gm
    B = cfa * Bm

    # Rows and columns that carry R sites, and likewise for B.
    R_r = np.tile(Rm.any(axis=1)[:, None], (1, w))
    R_c = np.tile(Rm.any(axis=0)[None, :], (h, 1))
    B_r = np.tile(Bm.any(axis=1)[:, None], (1, w))
    B_c = np.tile(Bm.any(axis=0)[None, :], (h, 1))

    conv = lambda k: convolve(cfa, k, mode="mirror")  # noqa: E731

    G = np.where(Rm | Bm, conv(_GR_GB), G)

    RB_gRB = conv(_Rg_RB)   # green site, on a row that runs R G R G
    RB_gBR = conv(_Rg_BR)   # green site, on a row that runs B G B G
    RB_diag = conv(_Rb_BB)  # opposite-colour site, diagonal neighbours

    R = np.where(R_r & B_c, RB_gRB, R)
    R = np.where(B_r & R_c, RB_gBR, R)
    B = np.where(B_r & R_c, RB_gRB, B)
    B = np.where(R_r & B_c, RB_gBR, B)
    R = np.where(B_r & B_c, RB_diag, R)
    B = np.where(R_r & R_c, RB_diag, B)

    rgb = np.stack([R, G, B], axis=-1)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return np.clip(np.round(rgb), info.min, info.max).astype(dt)
    return rgb


# ----------------------------------------------------------------------
# Pixels as numbers
# ----------------------------------------------------------------------
def show_numbers(ax, patch, tint=None, r0=0, c0=0, maxval=255):
    """Draw a small patch as coloured cells with the pixel value written in.

    tint is None for greyscale, or an MxNx3 unit-scale colour per cell.
    """
    patch = np.asarray(patch)
    m, n = patch.shape
    v = patch.astype(float) / maxval
    img = np.repeat(v[:, :, None], 3, axis=2) if tint is None \
        else np.asarray(tint, float) * v[:, :, None]

    ax.imshow(np.clip(img, 0, 1), aspect="equal")
    for i in range(m):
        for j in range(n):
            lum = 0.3 * img[i, j, 0] + 0.6 * img[i, j, 1] + 0.1 * img[i, j, 2]
            ax.text(j, i, f"{patch[i, j]:d}", color="k" if lum > 0.45 else "w",
                    ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(n), [str(c) for c in range(c0, c0 + n)])
    ax.set_yticks(range(m), [str(r) for r in range(r0, r0 + m)])
    ax.set_xlabel("column")
    ax.set_ylabel("row")
