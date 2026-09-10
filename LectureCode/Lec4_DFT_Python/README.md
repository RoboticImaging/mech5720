# MECH5720 Lecture 4: the 2D DFT, live coding demos

Short Python scripts, one idea each, in the order they come up in the
lecture. They are meant to be run a section at a time and fiddled with: most
have a "try:" list in their header, and the parameters worth changing sit at
the top of the file.

## Running

Dependencies: `numpy`, `scipy`, `matplotlib`, `pillow`. The pulse demo also
needs `opencv-python` to read the video.

    pip install numpy scipy matplotlib pillow opencv-python

Run any script from this folder, or from anywhere (image paths are resolved
relative to the script, not the working directory):

    python xCommonShapes_simple.py

Figures open as numbered windows. To run without opening windows and write
the figures to PNG instead:

    IMDEMO_SAVE=figs python xCommonShapes_simple.py

Animated demos then run to completion and save their final frame.

All scripts import `dftdemo.py`, which holds the small shared helpers:
`sfigure` and `imagesc` (numbered figures and one-cell-per-pixel image
display), `nice3` (surface plot), `freq_lpf_2d` and `freq_line_filt_2d`
(Gaussian filters built directly in the frequency domain), `single_wave`
(pull one bin and its conjugate partner out of a DFT) and `dft_ticks`
(0-based frequency axes). It is short and worth reading; nothing in it is
more than a few lines of array work.

## The demos, in lecture order

| Script | What it shows |
| --- | --- |
| `xDftMtx` | The rows of the DFT matrix are sampled complex exponentials. Plots the real part of the first few. |
| `xCommonShapes_simple` | A single 2D cosine and where it lands in the 2D DFT, with and without `fftshift`. Try the suggestions in the header: DC only, horizontal only, vertical only, mixed, and a frequency that creeps past N. |
| `xCommonShapes_shapes` | Bars, boxes, a rotated box and a disk, and their DFTs. |
| `xCommonShapes_shapes_filt` | A shape, its DFT, a frequency-domain filter, and the filtered result, all in one window. |
| `xCommonShapes_images` | A real photo (`dirt.jpg`) and its DFT, compressed with a fourth root so DC does not swamp everything. |
| `xCommonShapes_images_filt` | Filtering the photo in the frequency domain: remove low vertical frequencies and see what survives. |
| `xEnhanceSunset` | Sharpening as a frequency-domain filter: keep everything and boost the high frequencies. |
| `xSum2DSins` | Any image is a sum of 2D cosines. Adds random waves one at a time. Animated. |
| `x2DSin` | Interactive 2D cosine explorer with sliders for frequency and phase. |
| `xDFT_full_colour` | One DFT bin is one 2D wave: pull a single frequency out of each colour channel and look at it. |
| `xDFT_hardway_sparse` | Rebuild an image one wave at a time, largest coefficient first. Animated, Ctrl-C to stop. |
| `xDFT_hardway_sparse_colour` | The same, per colour channel. Animated. |
| `xPulseDet` | Pulse detection from a video of a face using a 1D DFT along time. Sums each frame to one number per channel, finds the heart-rate peak, then amplifies that frequency in every pixel and plays the video back. |

## Indexing

The frequency axes on every DFT plot are labelled 0-based, because that is
how the DFT is defined: bin 0 is DC, bin k is k cycles across the image, and
numpy's array indexing matches. So `X[0, 0]` is DC and `X[3, 5]` is the wave
with 3 cycles down and 5 across.

For a real image the DFT is conjugate symmetric: the partner of bin `(r, c)`
in an N-by-N transform is `((-r) % N, (-c) % N)`. A single real cosine
therefore lives in two bins, and `single_wave` in `dftdemo.py` keeps both
when it pulls one wave out. Note that Python's `%` returns a non-negative
result for a negative left operand, which is exactly what this needs.

## Notes on particular demos

`xPulseDet` downsamples the video 2x before the per-pixel FFT. At full
resolution the 301 frames are over 1 GB in float32 and twice that inside the
transform, which is more than a laptop wants to hold; the pulse is just as
visible at half size. The bpm plot (figure 4) shows which DFT bin to amplify
and why.

`x2DSin` uses matplotlib slider widgets. The 3D surface in figure 2 is the
slow part; if it lags, reduce the grid size at the top of the script.

`xDFT_hardway_sparse` and its colour version run 1000 iterations by default
and redraw three figures each time. Ctrl-C stops them early; `NumIters` at
the top sets the count.

## Data files

`dirt.jpg` and `sunset_128.jpg` are the test images. `face.mp4` is from
https://github.com/AnirudhBHarish/Pulse_Detection_From_Videos and is used
only by `xPulseDet`. Keep them in this folder; the scripts look for them
next to themselves.
