"""
xSum2DSins.py
Any image is a sum of 2D cosines.  Add random ones together and watch a
"picture" appear.  Each wave has a frequency pair (f_x, f_y), a phase pair
and an amplitude; the two stem plots keep track of them.

Runs as an animation.
"""

import numpy as np

from dftdemo import drawnow, imagesc, nice3, sfigure, show

NumIters = 20
ImSize = 200

X, Y = np.meshgrid(np.linspace(0, 1, ImSize), np.linspace(0, 1, ImSize))
I = np.zeros((ImSize, ImSize))

rng = np.random.default_rng()

fig1, ax1 = sfigure(1, projection="3d")
fig2, ax2 = sfigure(2, projection="3d")
fig3, ax3 = sfigure(3, projection="3d")
fig4, ax4 = sfigure(4, projection="3d")
fig5, ax5 = sfigure(5)

for i in range(NumIters):
    fX = 10 * rng.random()
    fY = 10 * rng.random()
    phasex = 2 * np.pi * rng.random()
    phasey = 2 * np.pi * rng.random()
    amp = rng.random()

    ax1.stem([fY], [fX], [amp], linefmt="C1-", markerfmt="C1.", basefmt=" ")
    ax1.set_xlabel("$f_x$")
    ax1.set_ylabel("$f_y$")
    ax1.set_title("Magnitudes and frequencies of 2D waves")

    ax2.stem([phasey], [phasex], [amp], linefmt="C1-", markerfmt="C1.", basefmt=" ")
    ax2.set_xlabel(r"$\phi_x$")
    ax2.set_ylabel(r"$\phi_y$")
    ax2.set_title("Magnitudes and phases of 2D waves")

    fval = amp * np.cos(2 * np.pi * fX * X + phasex) * np.cos(2 * np.pi * fY * Y + phasey)
    I += fval

    nice3(ax3, fval)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("single 2D wave")

    nice3(ax4, I)
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_title("Adding them together")

    ax5.cla()
    imagesc(ax5, I)
    ax5.set_title("Adding them together, 2D display")

    drawnow(0.2)

show()
