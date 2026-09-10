"""
dftdemo.py
Shared helpers for the Lecture 4 (2D DFT) Python demos.

What is in here:
    sfigure          -> a numbered figure that does not steal focus
    imagesc          -> display an array, one cell per pixel, axis image
    nice3            -> a shaded surface plot of a 2D array
    load_image       -> read a JPEG as float in [0, 1], like im2double
    freq_lpf_2d      -> Gaussian lowpass filter built in the frequency domain
    freq_line_filt_2d-> Gaussian passband along the f_u axis
    dft_ticks        -> 0-based frequency axes, with and without fftshift
    drawnow          -> let the figures repaint inside a loop

Dependencies: numpy, matplotlib, pillow.  xPulseDet.py also needs opencv
(pip install opencv-python) to read the video.

Set the environment variable IMDEMO_SAVE to a directory to write every figure
to PNG instead of opening windows:
    IMDEMO_SAVE=figs python xCommonShapes_simple.py
Animated demos then run to completion and save their final frame.
"""

import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
_SAVE_DIR = os.environ.get("IMDEMO_SAVE")
if _SAVE_DIR:
    mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Dark background, to match the lecture slides.
plt.style.use("dark_background")
mpl.rcParams.update({
    "figure.figsize": (7.5, 5.5),
    "axes.grid": False,
    "image.interpolation": "nearest",
    "font.size": 12,
})

YEL = (1.0, 0.85, 0.15)


# ----------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------
def load_image(name, channel=None):
    """Read an image next to this file as float64 in [0, 1] (like im2double).

    channel=None returns HxWx3; channel=0/1/2 returns that plane (R/G/B).
    """
    path = Path(name)
    if not path.is_absolute():
        path = HERE / path
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return a if channel is None else a[:, :, channel]


# ----------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------
def sfigure(n, figsize=None, projection=None):
    """Numbered figure, cleared and ready to draw on.  Returns (fig, ax).

    projection="3d" gives a 3D axes, for nice3().
    """
    fig = plt.figure(n, figsize=figsize)
    fig.clf()
    return fig, fig.add_subplot(111, projection=projection)


def imagesc(ax, img, clim=None, cmap="gray", **kw):
    """imagesc + axis image.  Handles 2D (with colormap) and HxWx3 (as colour).

    Complex input (straight out of ifft2) is shown as its real part.
    """
    img = np.asarray(img)
    if np.iscomplexobj(img):
        img = img.real
    if img.ndim == 3:
        h = ax.imshow(np.clip(img, 0, 1), aspect="equal", **kw)
    else:
        vmin, vmax = (None, None) if clim is None else clim
        h = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    return h


def nice3(ax, a, cmap="viridis"):
    """Shaded surface plot of a 2D array.

    ax must be a 3D axes, e.g. from sfigure(n, projection="3d").  The view
    angle survives between calls, so a loop can spin the plot by hand.
    """
    a = np.asarray(a)
    if np.iscomplexobj(a):
        a = a.real
    elev, azim = ax.elev, ax.azim
    ax.cla()
    m, n = a.shape
    X, Y = np.meshgrid(np.arange(n), np.arange(m))
    surf = ax.plot_surface(X, Y, a, cmap=cmap, linewidth=0, antialiased=False,
                           rstride=max(1, m // 100), cstride=max(1, n // 100))
    ax.view_init(elev, azim)
    return surf


def dft_ticks(ax, shape, shifted=False, step=8):
    """Label a DFT image with 0-based frequency indices.

    Call after imagesc(ax, X, origin="lower").  With shifted=True the tick
    labels run from -N/2, matching fftshift.
    """
    rows, cols = shape[:2]
    for axis, n in (("x", cols), ("y", rows)):
        locs = np.r_[np.arange(0, n - step + 1, step), n - 1]
        labs = locs - n // 2 if shifted else locs
        if axis == "x":
            ax.set_xticks(locs, [str(v) for v in labs])
        else:
            ax.set_yticks(locs, [str(v) for v in labs])
    ax.set_xlabel("$f_u$")
    ax.set_ylabel("$f_v$")


def drawnow(dt=0.01):
    """Repaint all figures mid-loop.  No-op when saving."""
    if not _SAVE_DIR:
        plt.pause(dt)


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
# Frequency-domain filters
# ----------------------------------------------------------------------
def _freq_grid(shape):
    """f_v (rows) and f_u (cols) in cycles/sample, in fft2 order (DC at [0,0])."""
    rows, cols = shape[:2]
    fv = np.fft.fftfreq(rows)
    fu = np.fft.fftfreq(cols)
    return np.meshgrid(fv, fu, indexing="ij")


def freq_lpf_2d(shape, bw):
    """2D Gaussian lowpass filter in the frequency domain.

    bw is the 3 dB bandwidth in cycles/sample (0.5 is Nyquist).  H is real,
    the same size as the image and in fft2 order, so apply it as
        xf = ifft2(fft2(x) * H)
    """
    FV, FU = _freq_grid(shape)
    return 2.0 ** (-(FV**2 + FU**2) / (2 * bw**2))   # H = 1/sqrt(2) at |f| = bw


def freq_line_filt_2d(shape, bw):
    """Passband along the line f_v = 0: all horizontal frequencies, only
    vertical frequencies within bw (3 dB) of zero.  Same layout as
    freq_lpf_2d.
    """
    FV, _ = _freq_grid(shape)
    return 2.0 ** (-(FV**2) / (2 * bw**2))


# ----------------------------------------------------------------------
# Picking single waves out of a DFT
# ----------------------------------------------------------------------
def conjugate_partner(row, col, shape):
    """Index of the DFT bin that pairs with (row, col) for a real image."""
    rows, cols = shape[:2]
    return (-row) % rows, (-col) % cols


def single_wave(X, row, col):
    """The real 2D cosine that bin (row, col) of DFT X (and its mirror) encode.

    Returns (wave, magnitude).  Zeroing everything except the bin and its
    conjugate partner, then inverse-transforming, gives that one wave.
    """
    row2, col2 = conjugate_partner(row, col, X.shape)
    W = np.zeros_like(X)
    for r, c in ((row, col), (row2, col), (row2, col2), (row, col2)):
        W[r, c] = X[r, c]
    return np.fft.ifft2(W).real, abs(X[row, col])


# ----------------------------------------------------------------------
# Small shape helpers
# ----------------------------------------------------------------------
def disk_kernel(r, oversample=8):
    """Unit-sum disk of radius r on a (2r+1)^2 grid, edge pixels area-weighted."""
    n = 2 * r + 1
    s = oversample
    c = np.arange(n * s) / s - r - 0.5 + 0.5 / s     # sub-pixel centres
    X, Y = np.meshgrid(c, c)
    fine = (X**2 + Y**2) <= r**2
    d = fine.reshape(n, s, n, s).mean(axis=(1, 3))
    return d / d.sum()


def rotate_image(x, angle_deg):
    """Rotate about the centre with bilinear interpolation, same size out."""
    from scipy.ndimage import rotate
    return rotate(x, angle_deg, reshape=False, order=1)
