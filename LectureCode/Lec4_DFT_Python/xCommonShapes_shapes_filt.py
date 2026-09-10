"""
xCommonShapes_shapes_filt.py
A shape, its DFT, a frequency-domain filter, and the filtered result.

try:
  the box instead of the disk
  the lowpass filter instead of the line filter
  H instead of 1-H
"""

import numpy as np

from dftdemo import (dft_ticks, disk_kernel, freq_line_filt_2d, freq_lpf_2d,
                     imagesc, sfigure, show)

ImgSize = 32
x = np.zeros((ImgSize, ImgSize))

# box
w = ImgSize // 4
start = ImgSize // 2 - w // 2
# x[start:start + w, start:start + 2 * w] = 1

# disk
d = disk_kernel(w - 1)
w = d.shape[0]
start = ImgSize // 2 - w // 2
x[start:start + w, start:start + w] = d

# filter: a Gaussian passband along f_v = 0, then inverted, so it REMOVES
# the low vertical frequencies and keeps everything else
H = freq_line_filt_2d(x.shape, 0.05)
# H = freq_lpf_2d(x.shape, 0.1)
H = 1 - H

Xf = np.fft.fft2(x)
Xff = Xf * H
xf = np.fft.ifft2(Xff).real

fig, _ = sfigure(1, figsize=(12, 7))
fig.clf()

ax = fig.add_subplot(2, 3, 1)
imagesc(ax, x)
ax.set_title("Image")

ax = fig.add_subplot(2, 3, 2)
imagesc(ax, np.fft.fftshift(np.abs(Xf)), origin="lower")
dft_ticks(ax, Xf.shape, shifted=True)
ax.set_title("DFT(Image)")

ax = fig.add_subplot(2, 3, 3)
imagesc(ax, np.fft.fftshift(H), origin="lower")
dft_ticks(ax, H.shape, shifted=True)
ax.set_title("Filter")

ax = fig.add_subplot(2, 3, 4)
imagesc(ax, xf)
ax.set_title("Filtered Image")

ax = fig.add_subplot(2, 3, 5)
imagesc(ax, np.fft.fftshift(np.abs(Xff)), origin="lower")
dft_ticks(ax, Xff.shape, shifted=True)
ax.set_title("Filtered DFT(Image)")

fig.tight_layout()
show()
