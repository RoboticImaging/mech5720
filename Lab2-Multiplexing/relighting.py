"""
MECH5720 Lab 2 -- Multiplexing for Relighting.

Relight a scene from a stack of images captured under multiplexed
illumination, and measure the signal-to-noise advantage that
multiplexing buys you over illuminating one site at a time.

Run this by stepping through it in your debugger rather than executing
it end to end. Set a breakpoint at the first section and step forward,
watching each figure as it appears. Every step you complete should
produce a visible change in the figures.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard

import imaging as im

# ======================================================================
# Configuration
# ======================================================================
# Everything you need to change to switch between the provided data and
# your own capture lives in this block.

# directory holding the image files
DATA_PATH = "/home/don/Data.local/2026_Mux/provided"

BLACK_LEVEL = 0.0               # sensor black level, normalised (see Lab 1)

N_SITES = 15                    # illumination sites == multiplexed frames
N_AMBIENT = 16                  # frames with all sites off
N_ALLON = 16                    # frames with all sites on
N_FIRSTON = 16                  # frames with only site 1 on

GAMMA_DISPLAY = 2.0             # display only; never applied to the data
GAMMA_ERROR = 3.3               # stronger gamma, to see small error values

plt.ion()                       # figures update as the script runs


# ======================================================================
# 1. The multiplexing matrix
# ======================================================================
# An "S-matrix" built from a Hadamard matrix, following Schechner et al.
# 2007, "Multiplexing for Optimal Lighting", Appendix A. Row i gives the
# illumination pattern used for multiplexed frame i: a 1 means that site
# was on. Dropping the first row and column of the Hadamard matrix
# removes the all-on pattern, which carries no differential information.
# H must match the matrix used at capture time.

H = hadamard(N_SITES + 1)
H = H[1:, 1:]
H = (1 - H) / 2

fig = im.show(H[:, :, None], "Multiplexing matrix H", gamma=1.0, fignum=1)
im.save_figure(fig, "01_multiplexing_matrix")

print("Each multiplexed frame has %d of %d sites on."
      % (H[0].sum(), N_SITES))


# ======================================================================
# 2. Load and inspect the multiplexed frames
# ======================================================================

mux = im.load_stack(DATA_PATH, "Multiplexed", N_SITES, BLACK_LEVEL)
print("Multiplexed stack:", mux.shape)

fig = im.show_stack(mux, "Multiplexed input frames", GAMMA_DISPLAY, fignum=2)
im.save_figure(fig, "02_multiplexed_input")


# ======================================================================
# 3. Estimate the ambient illumination
# ======================================================================
# The ambient frames were captured with every illumination site off, so
# they record whatever light reaches the scene that you do not control.
# A single ambient frame is as noisy as any other frame; averaging the
# stack gives a much lower-noise estimate.

ambient_stack = im.load_stack(DATA_PATH, "Ambient", N_AMBIENT, BLACK_LEVEL)

fig = im.show(ambient_stack[0], "Ambient, single frame",
              GAMMA_DISPLAY, fignum=3)

# TODO 1: average the ambient frames to get a low-noise ambient estimate.
#         Result should have shape (H, W, C), i.e. one image, not a stack.
ambient = np.zeros_like(mux[0])  # <-- your code here

fig = im.show(ambient, "Ambient, averaged over %d frames" % N_AMBIENT,
              GAMMA_DISPLAY, fignum=4)
im.save_figure(fig, "03_ambient_averaged")


# ======================================================================
# 4. Remove the ambient contribution
# ======================================================================
# The multiplexing model describes only the light you control. Ambient
# light is a constant offset present in every frame, so subtract it.
#
# Note what we do NOT do here. It is tempting to clamp the result at
# zero, on the grounds that there is no such thing as negative light.
# Do not. Where the controlled illumination is weaker than the noise,
# roughly half the pixels legitimately read below ambient, and clamping
# them discards the negative half of a symmetric noise distribution.
# That leaves a positive bias in every frame, which propagates into
# every measurement you make later. Negative values here are noise, not
# an error, and they must be allowed to cancel. Clamping is for display
# only, and imaging.show() already handles it.

# TODO 2: subtract the ambient estimate from the multiplexed stack.
mux = mux  # <-- your code here

fig = im.show_stack(mux, "Multiplexed input, ambient removed",
                    GAMMA_DISPLAY, fignum=5)
im.save_figure(fig, "04_multiplexed_ambient_removed")


# ======================================================================
# 5. Demultiplex
# ======================================================================
# Each multiplexed frame is a sum of the single-site images selected by
# one row of H:   mux = H @ single_site.  Recovering the single-site
# images is therefore a linear solve, applied identically to every pixel
# and every colour channel. Flatten so that each row is one frame.

original_shape = mux.shape
mux_flat = mux.reshape(N_SITES, -1)
print("Flattened for solving:", mux_flat.shape)

# TODO 3: solve for the single-site images.
#         Use np.linalg.solve rather than forming H^-1 explicitly: it is
#         better conditioned and faster. The answer is the same either
#         way, so this is a numerical choice, not a mathematical one.
demux_flat = mux_flat  # <-- your code here

demux = demux_flat.reshape(original_shape)

fig = im.show_stack(im.side_by_side(mux, demux),
                    "Multiplexed (left) and demultiplexed (right)",
                    GAMMA_DISPLAY, fignum=6)
im.save_figure(fig, "05_mux_vs_demux")


# ======================================================================
# 6. Relight the scene
# ======================================================================
# You now hold the scene's appearance under each illumination site
# separately. Any illumination pattern you like is a weighted sum of
# those images. Add the ambient back in to get a physically plausible
# image.

fig = im.show_stack(demux + ambient,
                    "Single site illuminated, with ambient",
                    GAMMA_DISPLAY, fignum=7)

# TODO 4: synthesise the scene as it would appear with all sites on.
allon_est_demux = ambient  # <-- your code here

fig = im.show(allon_est_demux, "Synthesised: all sites on",
              GAMMA_DISPLAY, fignum=8)
im.save_figure(fig, "06_synth_allon")

# TODO 5: synthesise the scene lit by a checkerboard pattern.
#         Choose which sites to switch on. Which indices form a
#         checkerboard depends on how the sites were laid out on the
#         display at capture time, so work out the layout for the
#         dataset you are using before you pick them.

relit = ambient  # <-- your code here

fig = im.show(relit, "Synthesised: checkerboard illumination",
              GAMMA_DISPLAY, fignum=9)
im.save_figure(fig, "07_synth_checkerboard")

# --- Free experimentation -------------------------------------------
# Try: several sites at once to accentuate shadows; different weights
# per colour channel; scaling the ambient down to make the relighting
# more dramatic.

creative = np.zeros_like(demux[0])
if creative.shape[-1] == 3:
    creative[:, :, 0] = demux[0:4, :, :, 0].sum(axis=0)
    creative[:, :, 1] = demux[12:15, :, :, 1].sum(axis=0)
    creative[:, :, 2] = 2.0 * demux[3::4, :, :, 2].sum(axis=0)
else:
    creative = demux[0::3].sum(axis=0)
creative = creative + ambient / 4.0

fig = im.show(creative, "Creative relighting", GAMMA_DISPLAY, fignum=10)
im.save_figure(fig, "08_creative")


# ======================================================================
# 7. Qualitative comparison against single-site capture
# ======================================================================
# The impulse frames were captured with one site on at a time: the
# direct alternative to multiplexing. They cost the same number of
# exposures, so this is a fair comparison.

impulse = im.load_stack(DATA_PATH, "Impulse", N_SITES, BLACK_LEVEL)

fig = im.show_stack(im.side_by_side(impulse, demux + ambient),
                    "Impulse (left) vs demultiplexed (right)",
                    GAMMA_DISPLAY, fignum=100)
im.save_figure(fig, "09_impulse_vs_demux")
# Zoom in to better see the noise levels.

# The all-on frames, averaged, are the gold standard: a direct
# measurement of what both methods are trying to predict.
allon_gold_standard = im.load_stack(DATA_PATH, "AllOn", N_ALLON, BLACK_LEVEL).mean(axis=0)

# TODO 6: estimate the all-on image from the impulse frames.
#         Think carefully about the ambient. 
allon_est_impulse = ambient  # <-- your code here

fig = im.show_stack(
    im.side_by_side(allon_gold_standard, allon_est_impulse, allon_est_demux)[None],
    "All sites on:  gold standard  /  "
    "impulse estimate  /  demultiplexed estimate",
    GAMMA_DISPLAY, fignum=101)
im.save_figure(fig, "10_allon_three_way")


# ======================================================================
# 8. Quantitative evaluation
# ======================================================================

error_demux = allon_est_demux - allon_gold_standard
error_impulse = allon_est_impulse - allon_gold_standard

fig = im.show_stack(
    im.side_by_side(error_impulse ** 2, error_demux ** 2)[None],
    "Squared error vs the all-on gold standard:  "
    "impulse (left)  /  demultiplexed (right)",
    GAMMA_ERROR, fignum=102)
im.save_figure(fig, "11_error_images")


# TODO 7: find the mean squared error of each estimate against the gold
#         standard, then the peak signal-to-noise ratio of each, and the
#         advantage multiplexing gives you. Peak value here is 1.0, since
#         the images are on [0, 1]. Find PSNR and PSNR advantage in dB.

mse_demux = 1.0  # <-- your code here
mse_impulse = 1.0  # <-- your code here

psnr_demux_db = 0.0  # <-- your code here
psnr_impulse_db = 0.0  # <-- your code here
psnr_advantage_db = 0.0  # <-- your code here


# ======================================================================
# 9. The same comparison for a single illuminant
# ======================================================================
# The advantage above is for the all-on image, which is the easiest case
# for multiplexing. Repeat the measurement for one illumination site on
# its own, the hardest case, using the FirstOn frames as the gold
# standard: those were captured with site 1 on and nothing else.

first_gold_standard = im.load_stack(DATA_PATH, "FirstOn", N_FIRSTON,
                                    BLACK_LEVEL).mean(axis=0)

# TODO 8: estimate the site-1 image by each method. The impulse method
#         measured it directly, in one frame. For the demultiplexed
#         estimate don't forget the impact of ambient light.
first_est_impulse = impulse[0]
first_est_demux = ambient  # <-- your code here

fig = im.show_stack(
    im.side_by_side(first_gold_standard, first_est_impulse,
                    first_est_demux)[None],
    "Site 1 only:  gold standard  /  "
    "impulse estimate  /  demultiplexed estimate",
    GAMMA_DISPLAY, fignum=103)
im.save_figure(fig, "12_single_site_three_way")

mse_first_impulse = 1.0  # <-- your code here
mse_first_demux = 1.0  # <-- your code here

psnr_first_impulse_db = 0.0  # <-- your code here
psnr_first_demux_db = 0.0  # <-- your code here
psnr_first_advantage_db = 0.0  # <-- your code here


# ======================================================================
# 10. Report these numbers
# ======================================================================
def row(label, value, unit=""):
    print(("  %-32s %s %s" % (label, value, unit)).rstrip())

print("")
print("=" * 58)
row("ambient mean level", "%.5f" % ambient.mean())
row("ambient std dev", "%.5f" % ambient_stack.std(axis=0).mean())
print("-" * 58)
row("ALL-ON IMAGE", "")
row("  MSE, impulse estimate", "%.3e" % mse_impulse)
row("  MSE, demultiplexed estimate", "%.3e" % mse_demux)
row("  PSNR, impulse estimate", "%.3f" % psnr_impulse_db, "dB")
row("  PSNR, demultiplexed estimate", "%.3f" % psnr_demux_db, "dB")
row("  PSNR ADVANTAGE", "%.3f" % psnr_advantage_db, "dB")
print("-" * 58)
row("SINGLE-SITE IMAGE (site 1)", "")
row("  PSNR, impulse estimate", "%.3f" % psnr_first_impulse_db, "dB")
row("  PSNR, demultiplexed estimate", "%.3f" % psnr_first_demux_db, "dB")
row("  PSNR ADVANTAGE", "%.3f" % psnr_first_advantage_db, "dB")
print("=" * 58)
print("")


# Cleanup / close
plt.show(block=False)
plt.pause(0.2)          # let every figure draw before we block
try:
    input("press enter to close the figures ")
except EOFError:        # no interactive stdin: fall back to blocking
    plt.show(block=True)
plt.close("all")
