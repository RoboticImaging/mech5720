"""
x05bit_depth_demo.py
A file with 16-bit samples is not a 16-bit image.
This one came off a 10-bit ADC and was left-justified into a 16-bit
container, so it fills the full 0..65535 range using 1024 distinct codes.
Digital gain rescales those codes.  It does not create new ones.
"""

import numpy as np

from imdemo import YEL, imagesc, read_raw, sfigure, show, stem

imgFile = "mono_10in16bit_512x512.raw"
W, H = 512, 512
gain = 2.5

img = read_raw(imgFile, W, H, np.uint16)

x = img.ravel().astype(float)
nLevels = np.unique(img).size

# --- 1. It looks like any other 16-bit image ---------------------------
fig, ax = sfigure(1)
imagesc(ax, img, [0, 65535])
ax.set_title(f"claims 16 bits per sample, uses {nLevels} distinct values")

# --- 2. Coarsely binned histogram: nothing looks wrong -----------------
fig, ax = sfigure(2)
ax.hist(x, 256, color=YEL)
ax.set_xlim(0, 65535)
ax.set_xlabel("code value")
ax.set_ylabel("count")
ax.set_title("histogram at 256 bins: smooth, unremarkable")

# --- 3. One bin per code, zoomed in: the data is a comb ----------------
lo, hi = 20000, 20400
edges = np.arange(lo - 0.5, hi + 1.5)
counts, _ = np.histogram(x, edges)

fig, ax = sfigure(3)
stem(ax, np.arange(lo, hi + 1), counts)
ax.set_xlim(lo, hi)
ax.set_xlabel("code value")
ax.set_ylabel("count")
ax.set_title(f"one bin per code: occupied every 64, so {nLevels} real "
             "levels = 10 bits")

# --- 4. Apply digital gain.  No clipping, no rounding, same measurements ---
xg = x * gain
nLevelsG = np.unique(xg).size

fig, ax = sfigure(4)
ax.hist(xg, 256, color=YEL)
ax.set_xlim(0, 65535 * gain)
ax.set_xlabel("code value")
ax.set_ylabel("count")
ax.set_title(f"after {gain:.1f}x digital gain: brighter, still "
             f"{nLevelsG} levels")

# --- 5. Zoomed to the same data, now spread further apart --------------
loG, hiG = lo * gain, hi * gain
edgesG = np.arange(loG - 0.5 * gain, hiG + 0.5 * gain + 1e-9, gain)
countsG, _ = np.histogram(xg, edgesG)

fig, ax = sfigure(5)
stem(ax, edgesG[:-1] + 0.5 * gain, countsG)
ax.set_xlim(loG, hiG)
ax.set_xlabel("code value")
ax.set_ylabel("count")
ax.set_title("gain widened the gaps between codes, it did not fill them in")

# --- 6. The two combs on one axis, gain undone, to show nothing was added ---
fig, ax = sfigure(6)
stem(ax, np.arange(lo, hi + 1), counts, label="original")
stem(ax, edgesG[:-1] / gain + 0.5, countsG, color=(0.4, 0.7, 1.0),
     marker=False, label="gained then divided back")
ax.set_xlim(lo, hi)
ax.set_xlabel("code value")
ax.set_ylabel("count")
ax.legend()
ax.set_title("digital gain is invertible bookkeeping, not information")

show()
