"""
imaging.py -- MECH5720 shared plumbing.

Loading, display and metrics helpers used across the lab sequence.
You should not need to modify anything in this file.

Conventions
-----------
Images are float64 in nominal range [0, 1], linear in scene radiance.
Image stacks have shape (N, H, W, C), where C is 1 for monochrome / raw
data and 3 for colour. Everything downstream must work for either.

Display gamma is applied only inside the display functions. The data
itself is never gamma-corrected, because the demultiplexing maths
assumes a linear relationship between light and pixel value.
"""

import os
import re
import glob

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from PIL import Image


# ----------------------------------------------------------------------
# Figure style
# ----------------------------------------------------------------------

def dark_style():
    """Dark figure backgrounds, as colordef black does in MATLAB.

    Applied on import. Saved figures match what is on screen, which
    matters because savefig otherwise falls back to a white background
    on older matplotlib and the labels come out white on white.
    """
    plt.style.use("dark_background")
    matplotlib.rcParams.update({
        "figure.facecolor": "#2b2b2b",
        "axes.facecolor": "#2b2b2b",
        "savefig.facecolor": "#2b2b2b",
        "savefig.edgecolor": "#2b2b2b",
    })


dark_style()


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def _numeric_key(path):
    """Sort key that orders Frame2 before Frame10 (lexical sort does not)."""
    name = os.path.basename(path)
    digits = re.findall(r"\d+", name)
    return (int(digits[-1]) if digits else -1, name)


def load_image(path, black_level=0.0):
    """Load a single image as float64 in [0, 1], shape (H, W, C).

    black_level is given in the same normalised units and is subtracted
    before returning. It is NOT clamped at zero: a pixel that reads below
    the black level is a legitimate negative noise excursion, and
    discarding it biases every measurement made downstream.
    """
    im = np.asarray(Image.open(path))

    if im.dtype == np.uint8:
        im = im.astype(np.float64) / 255.0
    elif im.dtype == np.uint16:
        im = im.astype(np.float64) / 65535.0
    else:
        im = im.astype(np.float64)

    if im.ndim == 2:
        im = im[:, :, None]

    return im - black_level


EXTENSIONS = ("png", "tif", "tiff")
def load_stack(path, prefix, n_frames=None, black_level=0.0):
    """Load a numbered set of frames as a stack of shape (N, H, W, C).

    Frames are matched as <path>/<prefix>*.<ext> and ordered by the
    trailing number in the filename, so Multiplexed2 precedes
    Multiplexed10 regardless of zero padding.

    Only the extensions in EXTENSIONS are read. Other formats, raw DNG
    among them, need a loader of their own.
    """
    files = []
    for ext in EXTENSIONS:
        files += glob.glob(os.path.join(path, prefix + "*." + ext))
        files += glob.glob(os.path.join(path, prefix + "*." + ext.upper()))
    files = sorted(set(files), key=_numeric_key)

    if not files:
        others = [f for f in glob.glob(os.path.join(path, prefix + "*"))
                  if not os.path.isdir(f)]
        detail = ""
        if others:
            seen = sorted({os.path.splitext(f)[1] or "(none)" for f in others})
            detail = (". Found %d file(s) with unsupported extension(s) %s; "
                      "load_stack reads %s only"
                      % (len(others), ", ".join(seen), ", ".join(EXTENSIONS)))
        raise FileNotFoundError(
            "No frames matching '%s*' in %s%s" % (prefix, path, detail))

    if n_frames is not None:
        if len(files) < n_frames:
            raise FileNotFoundError(
                "Expected %d frames matching '%s*' in %s, found %d"
                % (n_frames, prefix, path, len(files)))
        files = files[:n_frames]

    return np.stack([load_image(f, black_level) for f in files])


def green_sublattice(raw, offset=(0, 1)):
    """Extract one green sub-lattice from a Bayer raw frame.

    As used in Lab 1. offset=(0, 1) selects green for a BGGR sensor;
    try the other three offsets if the result looks structurally wrong.
    """
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    r, c = offset
    return raw[r::2, c::2][:, :, None]


# ----------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------

def _for_display(image, gamma):
    """Prepare an array for display: squeeze, clamp, apply display gamma.

    The clamp to [0, 1] happens HERE and only here. Negative values are
    meaningful in the data (they are noise) but cannot be displayed.
    """
    image = np.squeeze(image)
    image = np.clip(image, 0.0, 1.0)
    return image ** (1.0 / gamma)


def show(image, title="", gamma=2.0, fignum=None):
    """Display a single image. Returns the figure."""
    fig = plt.figure(fignum) if fignum is not None else plt.figure()
    fig.clf()
    ax = fig.add_subplot(111)
    ax.imshow(_for_display(image, gamma), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def show_stack(stack, title="", gamma=2.0, fignum=None):
    """Display a stack of frames with a slider to scrub between them.

    Accepts (N, H, W, C). Returns the figure. Keep a reference to the
    returned figure or the slider will be garbage collected and stop
    responding.
    """
    fig = plt.figure(fignum) if fignum is not None else plt.figure()
    fig.clf()
    ax = fig.add_axes([0.05, 0.15, 0.90, 0.78])
    im = ax.imshow(_for_display(stack[0], gamma), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if len(stack) > 1:
        ax_slider = fig.add_axes([0.15, 0.05, 0.70, 0.04])
        slider = Slider(ax_slider, "frame", 1, len(stack),
                        valinit=1, valstep=1)

        def update(val):
            im.set_data(_for_display(stack[int(val) - 1], gamma))
            fig.canvas.draw_idle()

        slider.on_changed(update)
        fig._slider = slider      # keep alive

    return fig


def side_by_side(*images):
    """Concatenate images or stacks horizontally for comparison display.

    Works on single frames (H, W, C) or stacks (N, H, W, C), as long as
    all arguments have the same shape.
    """
    return np.concatenate(images, axis=-2)


def save_figure(fig, name, outdir="out"):
    """Write a figure to <outdir>/<name>.png for inclusion in your report."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path



