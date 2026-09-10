"""
xPulseDet.py
Pulse detection from a video of a face, with a 1D DFT along time.

Sum each frame to one number per colour channel: three noisy signals over
time.  Their DFTs have a peak at the heart rate.  Then amplify that one
frequency in every pixel and play the video back: the face pulses.

Video from https://github.com/AnirudhBHarish/Pulse_Detection_From_Videos
Needs opencv-python to read the mp4.
"""

import numpy as np
from scipy import fft            # keeps single precision; numpy.fft would upcast to complex128

from dftdemo import HERE, drawnow, imagesc, sfigure, show

Shrink = 2      # spatial downsampling of the video. 301 frames at full size is
                # 1.1 GB in float32 and twice that inside the FFT; halving each
                # axis cuts it to a quarter and the pulse is still plain to see.

# --- load the video into an array, frames x rows x cols x 3 ---------------
try:
    import cv2
except ImportError as e:
    raise SystemExit("this demo needs opencv:  pip install opencv-python") from e

cap = cv2.VideoCapture(str(HERE / "face.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    if Shrink > 1:
        bgr = cv2.resize(bgr, None, fx=1 / Shrink, fy=1 / Shrink, interpolation=cv2.INTER_AREA)
    frames.append(bgr[:, :, ::-1])            # OpenCV gives BGR; flip to RGB
cap.release()

LF = np.stack(frames).astype(np.float32) / 255.0
NFrames = LF.shape[0]
print(f"{NFrames} frames at {fps:.0f} fps, {LF.shape[2]}x{LF.shape[1]} after {Shrink}x shrink")

# --- one number per frame per colour channel ------------------------------
x = LF.sum(axis=(1, 2))                        # NFrames x 3

fig, ax = sfigure(1)
for c, col in enumerate("rgb"):
    ax.plot(x[:, c], col)
ax.set_xlabel("frame")
ax.set_ylabel("sum over the frame")
ax.set_title("brightness of each colour channel over time")

# normalise each colour channel: remove the mean, so DC does not dominate
x = x - x.mean(axis=0)

fig, ax = sfigure(2)
for c, col in enumerate("rgb"):
    ax.plot(x[:, c], col)
ax.set_xlabel("frame")
ax.set_title("the same, mean removed")

# --- DFT along time -------------------------------------------------------
Xf = fft.fft(x, axis=0)
freqs = np.arange(NFrames) * fps / NFrames     # bin k is k*fps/N hertz

fig, ax = sfigure(3)
for c, col in enumerate("rgb"):
    ax.plot(np.abs(Xf[:, c]), col)
ax.set_xlim(0, NFrames - 1)
ax.set_xlabel("DFT bin")
ax.set_ylabel("|X|")
ax.set_title("DFT of each channel: symmetric, so only the left half matters")

fig, ax = sfigure(4)
half = NFrames // 2
for c, col in enumerate("rgb"):
    ax.plot(freqs[:half] * 60, np.abs(Xf[:half, c]), col)
ax.set_xlim(0, 200)
ax.set_xlabel("beats per minute")
ax.set_ylabel("|X|")
ax.set_title("zoomed in, axis in bpm: the peak is the heart rate")

# --- filter: amplify one temporal frequency in every pixel ----------------
Idx = 9                    # 0-based DFT bin along time; see the bpm plot
# Idx = 18                 # harmonic of the heartbeat
# Idx = 75                 # ??? powerline?
Idx2 = NFrames - Idx       # its conjugate partner
print(f"amplifying bin {Idx}: {freqs[Idx]:.2f} Hz = {60 * freqs[Idx]:.0f} bpm")

Filt = np.ones(NFrames, dtype=np.float32)
Filt[Idx] = 20
Filt[Idx2] = 20

fig, ax = sfigure(5)
ax.stem(Filt, basefmt=" ")
ax.set_xlabel("DFT bin")
ax.set_title("the filter: pass everything, boost one frequency")

# apply it along the time axis, one FFT per pixel per channel
LFF = fft.fft(LF, axis=0, overwrite_x=True)     # complex64, frames x rows x cols x 3
LFF *= Filt[:, None, None, None]
LFf = fft.ifft(LFF, axis=0, overwrite_x=True).real
del LFF
LFf = np.clip(LFf, 0, 1)

# --- play it back ---------------------------------------------------------
fig, ax = sfigure(10)
h = imagesc(ax, LFf[0])
ax.set_title("filtered video: the pulse, amplified")
for k in range(NFrames):
    h.set_data(LFf[k])
    ax.set_title(f"filtered video, frame {k}")
    drawnow(1 / fps)

show()
