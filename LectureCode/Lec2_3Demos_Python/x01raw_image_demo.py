"""
x01raw_image_demo.py
Live tutorial: what a raw image actually is.
  1) monochrome raw as a picture
  2) the same pixels as an array of numbers
  3) a Bayer raw file, shown as plain numbers / grey pixels
  4) the same Bayer pixels with the CFA colours applied
  5) sub-lattice extraction
  6) demosaiced result
"""

import numpy as np

from imdemo import (cfa_mask, demosaic, imagesc, read_raw, sfigure, show,
                    show_numbers)

monFile, monW, monH = "mono_384x302_8bit.raw", 384, 302
bayFile, bayW, bayH = "bayer_512x512_rggb_8bit.raw", 512, 512
pattern = "rggb"

# --- Load: raw files are just bytes, row-major, no header ---------------
mono = read_raw(monFile, monW, monH, np.uint8)
bayer = read_raw(bayFile, bayW, bayH, np.uint8)

# --- 1. Monochrome raw, displayed as an image ---------------------------
fig, ax = sfigure(1)
imagesc(ax, mono, [0, 255])
ax.set_title("monochrome raw, 8-bit, one number per pixel")

# --- 2. The same data as an array of numbers ----------------------------
r0, c0, n = 120, 200, 8                    # small patch across an edge
patch = mono[r0:r0 + n, c0:c0 + n]

fig, ax = sfigure(2)
show_numbers(ax, patch, None, r0, c0)
ax.set_title("the image is an array of numbers")

# --- 3. Bayer raw, displayed as-is: still one number per pixel ----------
fig, ax = sfigure(3)
imagesc(ax, bayer, [0, 255])
ax.set_title("bayer raw, displayed with no knowledge of the CFA")

# --- 4. Bayer pixels as numbers, no colour ------------------------------
# Indices are 0-based here.  Keep both even so the top-left cell lands on an
# R site of the RGGB tile; get the parity wrong and the next figure lies.
rb, cb, m = 232, 264, 8
bpatch = bayer[rb:rb + m, cb:cb + m]

fig, ax = sfigure(4)
show_numbers(ax, bpatch, None, rb, cb)
ax.set_title("bayer pixels as numbers: which is red, which is blue?")

# --- 5. Same numbers, now with the CFA colours applied ------------------
cfa = cfa_mask(bayer.shape[0], bayer.shape[1], pattern)
cpatch = cfa[rb:rb + m, cb:cb + m, :]

fig, ax = sfigure(5)
show_numbers(ax, bpatch, cpatch, rb, cb)
ax.set_title("same numbers, coloured by the filter over each pixel")

# --- 6. Whole raw frame with the CFA colours applied --------------------
bayerRGB = cfa * bayer[:, :, None].astype(float) / 255

fig, ax = sfigure(6)
imagesc(ax, bayerRGB)
ax.set_title("every raw pixel, coloured by its filter "
             "(2/3 of each channel missing)")

# --- 7. Sub-lattice extraction: pull out the blue pixels only -----------
# RGGB, so blue sits at odd row, odd column in 0-based indexing.
blue = bayer[1::2, 1::2]                   # half the height, half the width

fig, ax = sfigure(7)
imagesc(ax, blue, [0, 255])
ax.set_title("the blue sub-lattice alone: a complete image at half resolution")

# --- 8. Demosaiced: the missing colour samples interpolated -------------
rgb = demosaic(bayer, pattern)

fig, ax = sfigure(8)
imagesc(ax, rgb.astype(float) / 255)
ax.set_title("demosaiced: three numbers per pixel, two of them interpolated")

show()
