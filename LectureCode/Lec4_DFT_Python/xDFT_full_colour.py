"""
xDFT_full_colour.py
One DFT bin, one 2D wave: pick a single frequency (row, col) out of each
colour channel's DFT and inverse-transform it on its own.  Also shows the
full magnitude spectrum per channel, as a colour image.

try:
  a different (row, col)
  row = 0 or col = 0, for a purely horizontal or vertical wave
"""

import numpy as np

from dftdemo import imagesc, load_image, sfigure, show, single_wave

I = load_image("sunset_128.jpg")
ImSize = I.shape[0]

row, col = 31, 21          # 0-based DFT bin: 31 cycles down, 21 across

Rebuild = np.zeros_like(I)
NiceDisp = np.zeros_like(I)

for ColChan in range(3):
    IF = np.fft.fft2(I[:, :, ColChan])
    CurWave, WaveMag = single_wave(IF, row, col)

    fig, ax = sfigure(2)
    imagesc(ax, CurWave)
    ax.set_title(f"Current 2D wave, channel {ColChan}, bin ({row}, {col})")

    Rebuild[:, :, ColChan] = CurWave
    NiceDisp[:, :, ColChan] = np.abs(IF)

fig, ax = sfigure(1)
imagesc(ax, I)
ax.set_title("Original image")

fig, ax = sfigure(3)
imagesc(ax, Rebuild / Rebuild.max() * 0.5 + 0.5)
ax.set_title("That one wave, all three channels")

fig, ax = sfigure(4)
imagesc(ax, (NiceDisp / NiceDisp.max()) ** 0.1, origin="lower")
ax.set_xlabel("fx")
ax.set_ylabel("fy")
ax.set_xlim(-0.5, 63.5)
ax.set_ylim(-0.5, 63.5)
ax.set_title("Magnitude spectrum per colour channel (top-left quadrant)")

show()
