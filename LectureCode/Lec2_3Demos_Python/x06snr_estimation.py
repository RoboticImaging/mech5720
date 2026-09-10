"""
x06snr_estimation.py
Burst averaging versus a single long exposure, and what an uncalibrated
gain and black level do to the result.

The reference file is headerless, row-major, little-endian.
"""

import numpy as np

from imdemo import YEL, imagesc, read_raw, sfigure, show

cleanFile, cleanW, cleanH = "clean_451x300_12bit_in_16.raw", 451, 300
S = read_raw(cleanFile, cleanW, cleanH, np.uint16).astype(float)

BL = 64          # black level of a short frame
sigmaRd = 60     # read noise, counts rms
nFrames = 20     # burst length
gain = 8.24      # what the long exposure actually collected
gainNom = 7      # what we asked for, and what we will assume
BLlong = 180     # its real pedestal, raised by dark current over a long
                 # integration.  We never measured it, so we assume BL.

rng = np.random.default_rng(0)   # fixed seed so the figures repeat in class

scene = BL + S   # the gold standard: a noiseless short frame

# SNR here is signal power over the power of what is left after subtracting
# the gold standard.  Calibration error counts as error, which is the point:
# a bias is just as damaging as noise, and only a reference frame reveals it.
sigRms = np.sqrt(np.mean(S ** 2))


def snr_db(im):
    return 20 * np.log10(sigRms / np.sqrt(np.mean((im - scene) ** 2)))


# --- 1. The gold standard ----------------------------------------------
fig, ax = sfigure(1)
imagesc(ax, scene, [BL, BL + 320])
ax.set_title("gold standard: the scene with no noise")

# --- 2. One short exposure ---------------------------------------------
frame1 = scene + sigmaRd * rng.standard_normal(S.shape)

fig, ax = sfigure(2)
imagesc(ax, frame1, [BL, BL + 320])
ax.set_title(f"single short exposure, SNR {snr_db(frame1):.1f} dB")

# --- 3. Twenty frames added, then divided by twenty --------------------
#     Adding is what reduces the noise.  Dividing puts the result back on the
#     scale of one frame, so the black level and the signal are unchanged.
burst = np.zeros(S.shape)
for _ in range(nFrames):
    burst += scene + sigmaRd * rng.standard_normal(S.shape)
burst /= nFrames

fig, ax = sfigure(3)
imagesc(ax, burst, [BL, BL + 320])
ax.set_title(f"burst of {nFrames}, SNR {snr_db(burst):.1f} dB")

# --- 4. One long exposure, with its own gain and black level -----------
long = BLlong + gain * S + sigmaRd * rng.standard_normal(S.shape)
# Put it on the short-frame scale using the numbers we think we have.
#    The gain is 3 percent off nominal, and the pedestal is 116 counts higher
#    than a short frame's.  Both are the kind of error you get from not
#    measuring, rather than from doing anything obviously wrong.
# longAdj = (long - BL) / gainNom + BL
longAdj = long / gainNom

fig, ax = sfigure(4)
imagesc(ax, longAdj)
ax.set_title(f"long exposure, SNR {snr_db(longAdj):.1f} dB")

# --- 5. One line trace through all three -------------------------------
row = 240

fig, ax = sfigure(5)
ax.plot(scene[row, :], "g-", linewidth=1.5, label="gold standard")
ax.plot(burst[row, :], "-", color=YEL, label="burst")
ax.plot(longAdj[row, :], "m-", label="long, gain/black level imperfect")
ax.set_xlim(0, cleanW - 1)
ax.set_xlabel("column")
ax.set_ylabel("counts")
ax.legend(loc="upper left")
ax.grid(True)
ax.set_title("the burst is noisy about the truth, the long exposure is biased")

show()
