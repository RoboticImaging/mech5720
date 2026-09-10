"""
x2DSin.py
Interactive 2D cosine explorer: drag the sliders for f_x, f_y and the two
phases, and watch the image (figure 1) and the surface (figure 2) follow.
"""

import numpy as np
from matplotlib.widgets import Slider

from dftdemo import imagesc, nice3, sfigure, show

X, Y = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 128))


def wave(fx, fy, phasex, phasey):
    return np.cos(2 * np.pi * fx * X + phasex) * np.cos(2 * np.pi * fy * Y + phasey)


# defaults
fx, fy = 1.0, 1.0
phasex, phasey = np.pi / 9, np.pi / 9

# --- figure 1: the image, with sliders underneath -------------------------
fig, ax = sfigure(1, figsize=(6, 7.5))
fig.subplots_adjust(bottom=0.32)
img = imagesc(ax, wave(fx, fy, phasex, phasey), [-1, 1])
ax.set_title("2D cosine wave explorer")

sliders = {}
for k, (name, lo, hi, v0) in enumerate([
        ("$f_x$", 0, 10, fx),
        ("$f_y$", 0, 10, fy),
        (r"$\phi_x$", -np.pi, np.pi, phasex),
        (r"$\phi_y$", -np.pi, np.pi, phasey)]):
    sax = fig.add_axes([0.2, 0.22 - 0.055 * k, 0.65, 0.03])
    sliders[name] = Slider(sax, name, lo, hi, valinit=v0)

# --- figure 2: the same wave as a surface ---------------------------------
fig2, ax2 = sfigure(2, projection="3d")
nice3(ax2, wave(fx, fy, phasex, phasey))
ax2.set_xlabel("x")
ax2.set_ylabel("y")


def update(_=None):
    vals = [s.val for s in sliders.values()]
    f = wave(*vals)
    img.set_data(f)
    nice3(ax2, f)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.canvas.draw_idle()
    fig2.canvas.draw_idle()


for s in sliders.values():
    s.on_changed(update)

show()
