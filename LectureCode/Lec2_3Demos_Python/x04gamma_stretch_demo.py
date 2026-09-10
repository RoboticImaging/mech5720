"""
x04gamma_stretch_demo.py
Power-law stretching: making a dim, linear capture look right.

Needs a scene with continuous tone.  Something bimodal, like a star field,
has nothing to redistribute and a stretch there just lifts the noise floor.
This is a night launch pad: bright vehicle, gantry towers and ground haze
sitting just above the noise floor.  Peak signal is about 280 counts out of
4095, linear, with a black level of 64.

Headerless, row-major, little-endian.
"""

import numpy as np

from imdemo import YEL, imagesc, plt, read_raw, sfigure, show

lowFile, lowW, lowH = "rocket_640x427_12bit_in_16.raw", 640, 427
lowLevel = 64                                  # black level, in counts
maxCode = 4095                                 # sensor full scale

low = read_raw(lowFile, lowW, lowH, np.uint16)
l = np.maximum(low.astype(float) - lowLevel, 0)   # black level subtracted

# --- 1. As captured, against the sensor's full range: it really is dark ---
fig, ax = sfigure(1)
imagesc(ax, l / maxCode, [0, 1])
ax.set_title("a dim capture: brightest pixel reaches "
             f"{100 * l.max() / maxCode:.0f}% of full scale")

# --- 2. Rescaled so the brightest pixel is white: still muddy ----------
#     This fixes the exposure, not the tone curve.  Separate problems.
ln = l / l.max()

fig, ax = sfigure(2)
imagesc(ax, ln, [0, 1])
ax.set_title("rescaled to fill the range: linear, correct, and flat looking")

# --- 3. Its histogram: one broad population, bunched low ---------------
fig, ax = sfigure(3)
ax.hist(ln.ravel() * 255, 256, color=YEL)
ax.set_xlim(0, 255)
ax.set_xlabel("normalised value, scaled to 0-255")
ax.set_ylabel("count")
ax.set_title("continuous tone, crowded into the lower half of the range")

# --- 4. Power-law stretch, exponent 0.5 --------------------------------
#     Compresses the bright end and expands the dark end.  Nothing is thrown
#     away, unlike clipping.  This is close to what a camera does for you.
fig, ax = sfigure(4)
imagesc(ax, ln ** 0.5, [0, 1])
ax.set_title("power-law stretch, gamma 0.5")

# --- 5. Exponent 0.25: too far, the midtones flatten out ---------------
fig, ax = sfigure(5)
imagesc(ax, ln ** 0.25, [0, 1])
ax.set_title("gamma 0.25: shadows open up, but contrast is gone")

# --- 6. The three side by side -----------------------------------------
fig = plt.figure(6, figsize=(13, 4))
fig.clf()
for k, (im, ttl) in enumerate([(ln, "linear"),
                               (ln ** 0.5, "gamma 0.5"),
                               (ln ** 0.25, "gamma 0.25")]):
    ax = fig.add_subplot(1, 3, k + 1)
    imagesc(ax, im, [0, 1])
    ax.set_title(ttl)

# --- 7. What the stretch does to the histogram -------------------------
fig, ax = sfigure(7)
ax.hist((ln ** 0.5).ravel() * 255, 256, color=YEL)
ax.set_xlim(0, 255)
ax.set_xlabel("displayed value")
ax.set_ylabel("count")
ax.set_title("after gamma 0.5: the same pixels, spread across the range")

# --- 8. What each stretch does to the numbers --------------------------
#     The clip curve is here for contrast: it discards everything outside
#     the window, where the power laws keep every pixel distinct.
lo, hi = 0.00, 0.25
x = np.linspace(0, 1, 512)

fig, ax = sfigure(8)
ax.plot(x, x, "-", color=YEL, label="linear")
ax.plot(x, np.clip((x - lo) / (hi - lo), 0, 1), "b-", label="manual clip")
ax.plot(x, x ** 0.5, "r-", label="gamma 0.5")
ax.plot(x, x ** 0.25, "m-", label="gamma 0.25")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("input")
ax.set_ylabel("displayed")
ax.legend(loc="lower right")
ax.grid(True)
ax.set_title("the transfer curves: what each display choice does to the numbers")

show()
