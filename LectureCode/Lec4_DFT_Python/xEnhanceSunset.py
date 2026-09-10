"""
xEnhanceSunset.py
Sharpening as a frequency-domain filter: keep everything, and boost the high
frequencies (1 + 2*(1 - LPF)).

try:
  a different bandwidth
  a different boost factor
  a plain lowpass filter, for blur
"""

import numpy as np
from PIL import Image

from dftdemo import HERE, freq_lpf_2d, imagesc, sfigure, show

im = Image.open(HERE / "sunset_128.jpg")
im = im.resize((im.width * 2, im.height * 2), Image.BICUBIC)   # imresize(x, 2)
x = np.asarray(im, dtype=float)[:, :, 1] / 255.0                # green channel

H = freq_lpf_2d(x.shape, 0.1)
H = 1 - H            # highpass
H = 1 + 2 * H        # all of the image, plus twice its high frequencies

Xf = np.fft.fft2(x)
xf = np.fft.ifft2(Xf * H).real

fig, _ = sfigure(1, figsize=(11, 5.5))
fig.clf()

ax = fig.add_subplot(1, 2, 1)
imagesc(ax, x, [0, 1])
ax.set_title("Image")

ax = fig.add_subplot(1, 2, 2)
imagesc(ax, xf, [0, 1])
ax.set_title("Filtered Image")

show()
