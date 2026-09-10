"""
xCommonShapes_images.py
A real photo and its 2D DFT.  Compressed with a fourth root so the huge DC
term does not swamp everything else.
"""

import numpy as np

from dftdemo import imagesc, load_image, sfigure, show

xc = load_image("dirt.jpg")          # colour, for display
x = xc[:, :, 1]                      # green channel, for the transform

fig, ax = sfigure(1)
imagesc(ax, xc)
ax.set_title("Image")

Xf = np.fft.fft2(x)

fig, ax = sfigure(2)
imagesc(ax, np.abs(Xf) ** 0.25, origin="lower")
ax.set_title("DFT with 0-based indexing")

fig, ax = sfigure(3)
imagesc(ax, np.fft.fftshift(np.abs(Xf)) ** 0.25, origin="lower")
ax.set_title("fftshift(DFT), 0-based indexing")

show()
