"""
xCommonShapes_images_filt.py
Filter a photo in the frequency domain: remove the low vertical frequencies
(the horizontal structure) and see what survives.

try:
  the lowpass filter instead of the line filter
  H instead of 1-H, i.e. keep only what the first version removed
  a wider or narrower bandwidth
"""

import numpy as np

from dftdemo import (freq_line_filt_2d, freq_lpf_2d, imagesc, load_image,
                     sfigure, show)

x = load_image("dirt.jpg", channel=1)     # green channel

H = freq_line_filt_2d(x.shape, 0.1)
# H = freq_lpf_2d(x.shape, 0.1)
H = 1 - H

Xf = np.fft.fft2(x)
Xff = Xf * H
xf = np.fft.ifft2(Xff).real

fig, _ = sfigure(1, figsize=(12, 6))
fig.clf()

ax = fig.add_subplot(2, 3, 1)
imagesc(ax, x)
ax.set_title("Image")

ax = fig.add_subplot(2, 3, 2)
imagesc(ax, np.fft.fftshift(np.abs(Xf)) ** 0.2, origin="lower")
ax.set_title("DFT(Image)")

ax = fig.add_subplot(2, 3, 3)
imagesc(ax, np.fft.fftshift(H), origin="lower")
ax.set_title("Filter")

ax = fig.add_subplot(2, 3, 4)
imagesc(ax, xf)
ax.set_title("Filtered Image")

ax = fig.add_subplot(2, 3, 5)
imagesc(ax, np.fft.fftshift(np.abs(Xff)) ** 0.2, origin="lower")
ax.set_title("Filtered DFT(Image)")

fig.tight_layout()
show()
