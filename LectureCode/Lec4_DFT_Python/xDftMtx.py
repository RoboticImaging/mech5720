"""
xDftMtx.py
The rows of the DFT matrix are the basis functions: sampled complex
exponentials, one per frequency.  Here are the real parts of the first few.
"""

import numpy as np

from dftdemo import sfigure, show

N = 128
k = np.arange(N)
# The DFT matrix: F[j, k] = exp(-2*pi*i*j*k/N).  Not orthonormal; divide
# by sqrt(N) if you want F^-1 == F^H.
F = np.exp(-2j * np.pi * np.outer(k, k) / N)

fig, _ = sfigure(1)
fig.clf()
for i in range(6):
    ax = fig.add_subplot(6, 1, i + 1)
    ax.plot(k, F[i + 1, :].real)            # row i+1: i+1 cycles across N samples
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    ax.set_xlim(0, N - 1)
    ax.set_ylim(-1, 1)

show()
