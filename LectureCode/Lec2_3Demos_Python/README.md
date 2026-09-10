# MECH5720 Lectures 2 and 3: images as data, live coding demos

Short Python scripts, one idea each, in the order they come up in the
lectures. They are meant to be run a section at a time and fiddled with: the
parameters worth changing (which row to profile, which patch to zoom, the
gamma, the black level) sit near the top of each section.

## Running

Dependencies: `numpy`, `scipy`, `matplotlib`, `pillow`.

    pip install numpy scipy matplotlib pillow

Run any script from this folder, or from anywhere (file paths are resolved
relative to the script, not the working directory):

    python x01raw_image_demo.py

Figures open as numbered windows. To run without opening windows and write
the figures to PNG instead:

    IMDEMO_SAVE=figs python x01raw_image_demo.py

All scripts import `imdemo.py`, which holds the small shared helpers:
`read_raw` (headerless raw file to array), `sfigure` and `imagesc`
(numbered figures and one-cell-per-pixel image display), `show_numbers` (a
patch drawn as cells with the value written in), `cfa_mask` (which colour
filter sits over each pixel) and `demosaic` (Malvar-He-Cutler interpolation).
It is short and worth reading; the demosaic in particular is a few 5x5
kernels and nothing more.

## The demos, in lecture order

| Script | What it shows |
| --- | --- |
| `x01raw_image_demo` | What a raw image actually is. A monochrome raw as a picture, then as an array of numbers; a Bayer raw as plain numbers, then with the colour filter over each pixel shown; the blue sub-lattice pulled out on its own; the demosaiced result. |
| `x02raw_vs_uncompressed_demo` | "Raw" and "uncompressed" are different claims. A linear 12-bit mosaic, a simulated in-camera ISP (white balance, demosaic, gamma), the result written uncompressed and as a low-quality JPEG, and the byte counts of all three. |
| `x03image_viewing_demo` | Ways of looking at an image beyond displaying it: line profiles, histograms, spotting saturation, and a dark scene with a nonzero black level and how to display it. |
| `x04gamma_stretch_demo` | Power-law stretching of a dim linear capture: why rescaling is not enough, what gamma 0.5 and 0.25 do to the image and its histogram, and the transfer curves side by side. |
| `x05bit_depth_demo` | A file with 16-bit samples is not a 16-bit image. Finding the real bit depth from the histogram, and why digital gain does not add information. |
| `x06snr_estimation` | Burst averaging versus one long exposure, and what an uncalibrated gain and black level do to the result. |

## The raw files

Every `.raw` file is headerless: just the samples, row-major, little-endian,
with no size or type information inside. The filename carries what you need
to read it, for example `darkscene_900x600_12bit_in_16.raw` is 900 columns by
600 rows of 16-bit little-endian integers holding 12-bit data. `read_raw`
takes the width, height and dtype and checks the byte count matches.

| File | Contents |
| --- | --- |
| `mono_384x302_8bit.raw` | Monochrome, 8-bit. |
| `bayer_512x512_rggb_8bit.raw` | Bayer mosaic, RGGB, 8-bit. |
| `bayer_linear_512x512_rggb_12in16bit.raw` | Bayer mosaic, RGGB, linear 12-bit in a 16-bit container. |
| `coins_384x302_8bit.raw` | Bright objects on a dark background. |
| `flatfield_640x480_8bit.raw` | A uniformly lit blank wall. |
| `saturated_512x512_8bit.raw` | An overexposed scene. |
| `darkscene_900x600_12bit_in_16.raw` | A faint scene with a nonzero black level. |
| `rocket_640x427_12bit_in_16.raw` | A dim, continuous-tone capture for the gamma demo. |
| `mono_10in16bit_512x512.raw` | 10-bit data left-justified in 16-bit samples. |
| `clean_451x300_12bit_in_16.raw` | A noiseless reference for the SNR demo. |

`x02raw_vs_uncompressed_demo` writes two files next to itself when it runs,
`isp_512x512_rgb_8bit.raw` and `isp_q8.jpg`. They are outputs, and are
regenerated every time.

## Indexing

Python indexes from 0, and so do all the row and column numbers printed on
the figures. Where a demo pulls out a patch of a Bayer image, the top-left
corner must land on an even row and even column to sit on the R site of the
RGGB tile; get the parity wrong and the coloured-numbers figure lies.
