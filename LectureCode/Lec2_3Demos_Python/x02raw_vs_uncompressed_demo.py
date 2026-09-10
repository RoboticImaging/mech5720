"""
x02raw_vs_uncompressed_demo.py
"Raw" and "uncompressed" are different claims.
  raw          = sensor samples, one number per pixel, linear, unprocessed
  uncompressed = every pixel stored exactly, but processed first
  compressed   = pixels approximated to save bytes
The middle one is uncompressed but not raw.
"""

import os

import numpy as np
from PIL import Image

from imdemo import HERE, demosaic, imagesc, read_raw, sfigure, show

rawFile = "bayer_linear_512x512_rggb_12in16bit.raw"
W, H, pattern, maxVal = 512, 512, "rggb", 4095   # 12 bits in a 16-bit container
wbGains = (0.65, 1.0, 1.25)              # illuminant cast baked into the capture

# --- Load the raw: one 16-bit number per pixel, linear in scene radiance ---
bayer = read_raw(rawFile, W, H, np.uint16)

# --- 1. The raw as it actually sits on the card -------------------------
fig, ax = sfigure(1)
imagesc(ax, bayer, [0, maxVal])
ax.set_title("raw: one linear sample per pixel, mosaicked, "
             "no white balance, no gamma")

# --- 2. Simulate an on-camera image signal processor (ISP) --------------
#     white balance, demosaic, gamma
wb = bayer.astype(float) / maxVal
wb[0::2, 0::2] /= wbGains[0]                        # R sites
wb[1::2, 1::2] /= wbGains[2]                        # B sites
wb = (np.minimum(wb, 1) * 65535).astype(np.uint16)

rgbLin = demosaic(wb, pattern).astype(float) / 65535
isp = (255 * rgbLin ** (1 / 2.2)).astype(np.uint8)  # tone curve (gamma)

fig, ax = sfigure(2)
imagesc(ax, isp.astype(float) / 255)
ax.set_title("post-ISP: three numbers per pixel, white balanced, gamma encoded")

# --- 3. Write that ISP output with every pixel stored exactly, and as JPEG ---
ispFile = HERE / "isp_512x512_rgb_8bit.raw"     # written next to this script
jpegFile = HERE / "isp_q8.jpg"

isp.tofile(ispFile)                        # interleaved RGB, row-major
Image.fromarray(isp).save(jpegFile, quality=8)
jpg = np.asarray(Image.open(jpegFile))

fig, ax = sfigure(3)
imagesc(ax, jpg.astype(float) / 255)
ax.set_title("same picture, JPEG quality 8: "
             "still three numbers per pixel, wrong ones")

# --- 4. Bytes on disk: uncompressed is the largest file and the least raw ---
files = [HERE / rawFile, ispFile, jpegFile]
names = ["raw (12-bit mosaic)", "uncompressed RGB", "JPEG q8"]
byts = [os.path.getsize(f) for f in files]

print(f"\n{'file':<22} {'kB':>10} {'x raw':>8}")
for name, b in zip(names, byts):
    print(f"{name:<22} {b / 1024:>10.1f} {b / byts[0]:>8.2f}")
print()

show()
