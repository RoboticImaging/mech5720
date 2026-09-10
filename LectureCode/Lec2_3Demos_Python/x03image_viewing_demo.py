"""
x03image_viewing_demo.py
Ways of looking at an image beyond just displaying it.
  line profiles                (figs 1-6)
  histograms                   (figs 7-9)
  saturation                   (figs 10-13)
  dark scenes, nonzero black level, and how to display them (figs 14-20)

All files are headerless, row-major, little-endian.
"""

import numpy as np

from imdemo import YEL, imagesc, read_raw, sfigure, show

coinFile, coinW, coinH = "coins_384x302_8bit.raw", 384, 302
flatFile, flatW, flatH = "flatfield_640x480_8bit.raw", 640, 480
satFile, satW, satH = "saturated_512x512_8bit.raw", 512, 512
darkFile, darkW, darkH = "darkscene_900x600_12bit_in_16.raw", 900, 600

coins = read_raw(coinFile, coinW, coinH, np.uint8)
flat = read_raw(flatFile, flatW, flatH, np.uint8)
sat = read_raw(satFile, satW, satH, np.uint8)
dark = read_raw(darkFile, darkW, darkH, np.uint16)


# ----------------------------------------------------------------------
#  LINE PROFILES
# ----------------------------------------------------------------------

# --- 1. The image, with the row we are about to plot drawn on it -------
rowIdx = 120

fig, ax = sfigure(1)
imagesc(ax, coins, [0, 255])
ax.plot([0, coinW - 1], [rowIdx, rowIdx], "r-", linewidth=1.5)
ax.set_title(f"bright objects on a dark background, row {rowIdx} marked")

# --- 2. That row as a plot: the image is a signal ----------------------
profile = coins[rowIdx, :].astype(float)

fig, ax = sfigure(2)
ax.plot(profile, "-", color=YEL)
ax.set_xlim(0, coinW - 1)
ax.set_ylim(0, 255)
ax.set_xlabel("column")
ax.set_ylabel("pixel value")
ax.grid(True)
ax.set_title("a line profile: each coin is a plateau, the background is the floor")

# --- 4. A vertical profile through the same image, for comparison ------
colIdx = 200
vprofile = coins[:, colIdx].astype(float)

fig, ax = sfigure(4)
ax.plot(vprofile, "-", color=YEL)
ax.set_xlim(0, coinH - 1)
ax.set_ylim(0, 255)
ax.set_xlabel("row")
ax.set_ylabel("pixel value")
ax.grid(True)
ax.set_title(f"vertical profile, column {colIdx}")

# --- 5. A flat field: the scene is uniform, the image is not -----------
midRow = round(flatH / 2)

fig, ax = sfigure(5)
imagesc(ax, flat, [0, 255])
ax.plot([0, flatW - 1], [midRow, midRow], "r-", linewidth=1.5)
ax.set_title("flat field: a uniformly lit blank wall")

# --- 6. The profile shows vignetting the eye barely notices ------------
fprofile = flat[midRow, :].astype(float)
edgeFrac = 100 * np.mean(np.r_[fprofile[:5], fprofile[-5:]]) / fprofile.max()

fig, ax = sfigure(6)
ax.plot(fprofile, "-", color=YEL)
ax.set_xlim(0, flatW - 1)
ax.set_ylim(0, 255)
ax.set_xlabel("column")
ax.set_ylabel("pixel value")
ax.grid(True)
ax.set_title(f"vignetting: edges are {edgeFrac:.0f}% of the centre")

# Arbitrary lines, not just rows and columns: sample along any path with
#   scipy.ndimage.map_coordinates(flat, [rr, cc], order=1)
# where rr, cc are linspace arrays of row and column coordinates.


# ----------------------------------------------------------------------
#  HISTOGRAMS
# ----------------------------------------------------------------------

# --- 7. Histogram of the coins image: two populations ------------------
fig, ax = sfigure(7)
ax.hist(coins.ravel().astype(float), 256, color=YEL)
ax.set_xlim(0, 255)
ax.set_xlabel("pixel value")
ax.set_ylabel("count")
ax.set_title("coins histogram: dark background peak, brighter coin material")

# --- 8. Histogram of the flat field: one narrow peak, but not a spike ---
fig, ax = sfigure(8)
ax.hist(flat.ravel().astype(float), 256, color=YEL)
ax.set_xlim(0, 255)
ax.set_xlabel("pixel value")
ax.set_ylabel("count")
ax.set_title("flat field histogram: spread is vignetting plus noise, "
             "not scene content")

# --- 9. Histogram of a small central patch: now it is just noise -------
midCol = round(flatW / 2)
patch = flat[midRow - 15:midRow + 16, midCol - 15:midCol + 16]

fig, ax = sfigure(9)
ax.hist(patch.ravel().astype(float), 32, color=YEL)
ax.set_xlim(0, 255)
ax.set_xlabel("pixel value")
ax.set_ylabel("count")
ax.set_title("a 31x31 patch: the width of this peak is the noise")


# ----------------------------------------------------------------------
#  SATURATION
# ----------------------------------------------------------------------

# --- 10. An overexposed scene ------------------------------------------
satRow = 100

fig, ax = sfigure(10)
imagesc(ax, sat, [0, 255])
ax.plot([0, satW - 1], [satRow, satRow], "r-", linewidth=1.5)
ax.set_title("overexposed: the sky has no detail left")

# --- 11. The giveaway: a spike piled up at the top of the range --------
clipped = 100 * np.mean(sat == 255)

fig, ax = sfigure(11)
ax.hist(sat.ravel().astype(float), 32, color=YEL)
ax.set_xlim(0, 255)
ax.set_xlabel("pixel value")
ax.set_ylabel("count")
ax.set_title(f"{clipped:.0f}% of pixels sit at 255: clipped, not bright")

# --- 12. Where the clipped pixels are ----------------------------------
fig, ax = sfigure(12)
imagesc(ax, (sat == 255).astype(float), [0, 1])
ax.set_title("saturation map: everything white here is lost, not recoverable")

# --- 13. A profile through the clipping: flat tops instead of structure ---
sprofile = sat[satRow, :].astype(float)

fig, ax = sfigure(13)
ax.plot(sprofile, "-", color=YEL)
ax.plot([0, satW - 1], [255, 255], "r--")
ax.set_xlim(0, satW - 1)
ax.set_ylim(0, 270)
ax.set_xlabel("column")
ax.set_ylabel("pixel value")
ax.grid(True)
ax.set_title("clipped regions are dead flat at 255")


# ----------------------------------------------------------------------
#  A DARK SCENE WITH A NONZERO BLACK LEVEL
# ----------------------------------------------------------------------

# --- 14. Displayed honestly over the full 12-bit range: almost nothing ---
fig, ax = sfigure(14)
imagesc(ax, dark, [0, 4095])
ax.set_title("12-bit data shown over [0 4095]: the scene is faint "
             "but it is there")

# --- 15. The histogram explains why, and shows the black level ---------
fig, ax = sfigure(15)
ax.hist(dark.ravel().astype(float), 256, color=YEL)
ax.set_xlim(0, 4095)
ax.set_xlabel("pixel value")
ax.set_ylabel("count")
ax.set_title("everything is crammed into the bottom few percent of the range")

# Rough estimate, good enough for a mostly-empty frame.  It will run a little
# high, because any real background glow is included.  The proper way is a
# dark frame with the lens capped, or the sensor's optical black pixels.
blackLevel = np.median(dark.astype(float))

# --- 17. A profile across empty sky: it oscillates about the black level ---
darkRow = 300
dprofile = dark[darkRow, :].astype(float)

fig, ax = sfigure(17)
ax.plot(dprofile, "-", color=YEL)
ax.plot([0, darkW - 1], [blackLevel, blackLevel], "r--")
ax.set_xlim(0, darkW - 1)
ax.set_xlabel("column")
ax.set_ylabel("pixel value")
ax.grid(True)
ax.set_title(f"the dark level sits near {blackLevel:.0f} counts, "
             "with noise straddling it")


# ----------------------------------------------------------------------
#  WAYS OF DISPLAYING A DARK IMAGE
# ----------------------------------------------------------------------

# --- 18. Subtract the black level, convert to float, normalise to [0 1] ---
#     Subtracting first is what makes the numbers proportional to light.
d = dark.astype(float) - blackLevel
d = np.maximum(d, 0)           # noise takes some pixels below black
d = d / 4095                   # normalise against the sensor's full range

fig, ax = sfigure(18)
imagesc(ax, d, [0, 1])
ax.set_title("black level subtracted, normalised by full scale: "
             "still nearly black")

# --- 19. Normalise by the data instead of by full scale ----------------
dn = d / d.max()

fig, ax = sfigure(19)
imagesc(ax, dn, [0, 1])
ax.set_title("normalised by the brightest pixel: better, but one hot pixel "
             "sets the scale")

# --- 20. Percentile clipping, chosen by hand ---------------------------
#     Pick the low and high levels yourself, then map that window to [0 1].
#     Anything outside the window is clipped, which is a choice, not an error.
lo, hi = 0.00, 0.25
dc = np.clip((dn - lo) / (hi - lo), 0, 1)

fig, ax = sfigure(20)
imagesc(ax, dc, [0, 1])
ax.set_title(f"manual clip: [{lo:.2f} {hi:.2f}] mapped to the full "
             "display range")

show()
