"""
xCommonShapes_simple.py
A single 2D cosine and where it lands in the 2D DFT.

try:
  DC only
  Horizontal frequency only
  Horizontal frequency with DC
  Vertical frequency only
  Mixed frequencies
  Creeping frequency up through 0:ImgSize and beyond
"""

import numpy as np

from dftdemo import dft_ticks, imagesc, sfigure, show

ImgSize = 32
X, Y = np.meshgrid(np.arange(ImgSize), np.arange(ImgSize))
phasex = 0
phasey = 0

DC = 1 / 4
fx = 2
fy = 0
fx2 = 8
fy2 = 6

x = DC + np.cos(2 * np.pi * fx / ImgSize * X + phasex) \
       * np.cos(2 * np.pi * fy / ImgSize * Y + phasey)
# x = x + np.cos(2*np.pi*fx2/ImgSize*X + phasex) * np.cos(2*np.pi*fy2/ImgSize*Y + phasey)  # add another wave

fig, ax = sfigure(1)
imagesc(ax, x)
ax.set_title("Image")

Xf = np.fft.fft2(x)

fig, ax = sfigure(2)
imagesc(ax, np.abs(Xf), origin="lower")
dft_ticks(ax, Xf.shape)
ax.set_title("DFT with 0-based indexing")

fig, ax = sfigure(3)
imagesc(ax, np.fft.fftshift(np.abs(Xf)), origin="lower")
dft_ticks(ax, Xf.shape, shifted=True)
ax.set_title("fftshift(DFT), 0-based indexing")

show()
