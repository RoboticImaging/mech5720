"""
xCommonShapes_shapes.py
Simple shapes and their 2D DFTs.

try:
  Horz / vert bars of different width
  box of different width
  a rotated box
  a disk
"""

import numpy as np

from dftdemo import dft_ticks, disk_kernel, imagesc, rotate_image, sfigure, show

ImgSize = 32
x = np.zeros((ImgSize, ImgSize))

# box
w = ImgSize // 4
start = ImgSize // 2 - w // 2
# x[:, start:start + w] = 1                       # vertical bar
# x[start:start + w, start:start + w] = 1         # square
# x[start:start + 2*w, start:start + w] = 1       # tall box
x[start:start + w, start:start + 2 * w] = 1       # wide box

# rot
# x = rotate_image(x, 65)

# disk
# d = disk_kernel(w - 1)
# w = d.shape[0]
# start = ImgSize // 2 - w // 2
# x[start:start + w, start:start + w] = d

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
