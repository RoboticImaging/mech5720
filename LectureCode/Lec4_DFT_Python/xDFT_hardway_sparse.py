"""
xDFT_hardway_sparse.py
Rebuild an image one 2D wave at a time, largest DFT coefficient first.
Greedy: take the biggest remaining bin (and its conjugate partner), turn it
into a wave, add it to the reconstruction, zero it, repeat.

Runs as an animation.  Ctrl-C to stop early.
"""

import numpy as np

from dftdemo import drawnow, imagesc, load_image, sfigure, show, single_wave

NumIters = 1000         # lower it if it drags

I = load_image("sunset_128.jpg", channel=1)
ImSize = I.shape[0]

Rebuild = np.zeros_like(I)
IF = np.fft.fft2(I)
NiceDisp = np.zeros(IF.shape)

fig2, ax2 = sfigure(2)
fig3, ax3 = sfigure(3)
fig4, ax4 = sfigure(4)

for it in range(NumIters):
    idx = np.argmax(np.abs(IF))
    row, col = np.unravel_index(idx, IF.shape)
    row2, col2 = (-row) % ImSize, (-col) % ImSize

    CurWave, WaveMag = single_wave(IF, row, col)

    ax2.cla()
    imagesc(ax2, CurWave)
    ax2.set_title(f"Current 2D wave: bin ({row}, {col}), iteration {it + 1}")

    NiceDisp[min(row, row2), min(col, col2)] = WaveMag

    ax4.cla()
    imagesc(ax4, (NiceDisp / NiceDisp.max()) ** 0.1, origin="lower")
    ax4.set_xlim(-0.5, 31.5)
    ax4.set_ylim(-0.5, 31.5)
    ax4.set_xlabel("fx")
    ax4.set_ylabel("fy")
    ax4.set_title("Tracking magnitude vs frequencies")

    Rebuild += CurWave

    ax3.cla()
    imagesc(ax3, Rebuild, [0, 1])
    ax3.set_title(f"Reconstructed image so far, {it + 1} waves")

    drawnow()

    # remove the wave we just used, so the next argmax finds the next one
    for r, c in ((row, col), (row2, col), (row2, col2), (row, col2)):
        IF[r, c] = 0

show()
