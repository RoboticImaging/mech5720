"""
xDFT_hardway_sparse_colour.py
The same greedy one-wave-at-a-time reconstruction, per colour channel.
Here the wave is subtracted from the image each time rather than zeroed in
the DFT, which comes to the same thing.

Runs as an animation.  Ctrl-C to stop early.
"""

import numpy as np

from dftdemo import drawnow, imagesc, load_image, sfigure, show, single_wave

NumIters = 1000

I = load_image("sunset_128.jpg")
ImSize = I.shape[0]

Rebuild = np.zeros_like(I)
NiceDisp = np.zeros_like(I)

fig2, ax2 = sfigure(2)
fig3, ax3 = sfigure(3)
fig4, ax4 = sfigure(4)

for it in range(NumIters):
    for ColChan in range(3):
        IF = np.fft.fft2(I[:, :, ColChan])

        idx = np.argmax(np.abs(IF))
        row, col = np.unravel_index(idx, IF.shape)
        row2, col2 = (-row) % ImSize, (-col) % ImSize

        CurWave, WaveMag = single_wave(IF, row, col)

        ax2.cla()
        imagesc(ax2, CurWave)
        ax2.set_title(f"Current 2D wave, channel {ColChan}: bin ({row}, {col})")

        NiceDisp[min(row, row2), min(col, col2), ColChan] = WaveMag

        Rebuild[:, :, ColChan] += CurWave
        I[:, :, ColChan] -= CurWave

    ax3.cla()
    imagesc(ax3, Rebuild)
    ax3.set_title(f"Reconstructed image so far, {it + 1} waves per channel")

    ax4.cla()
    imagesc(ax4, (NiceDisp / NiceDisp.max()) ** 0.1, origin="lower")
    ax4.set_xlim(-0.5, 31.5)
    ax4.set_ylim(-0.5, 31.5)
    ax4.set_xlabel("fx")
    ax4.set_ylabel("fy")
    ax4.set_title("Tracking magnitude vs frequencies")

    drawnow()

show()
