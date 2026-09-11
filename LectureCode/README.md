# MECH5720 lecture code

Jupyter notebooks for the lecture demos, one notebook per lecture. Each
notebook is meant to be run a section at a time and fiddled with: the
parameters worth changing sit at the top of each cell.

| Notebook | Topic |
| --- | --- |
| `Lec2_3_ImagesAsData.ipynb` | Images as data: raw vs uncompressed, line profiles, histograms, saturation, black level, gamma, bit depth, SNR. |
| `Lec4_DFT.ipynb` | The 2D DFT: basis functions, single cosines, shapes, frequency-domain filtering, sharpening, greedy reconstruction, and pulse detection from a face video (with an inline video player). |
| `Lec5_CodedAperture.ipynb` | Coded aperture imaging: PSF and MTF of a disc vs a coded aperture, naive inversion, and Wiener deblurring. Ported from MATLAB (see `Lec5_matlab_original/`). |

Data files live in `assets/<lecture>/`. The original MATLAB source for
lecture 5, plus its reference figures, is in `Lec5_matlab_original/`.

## Building the environment

Everything runs on Python 3.10+ with the packages in `requirements.txt`
(`numpy`, `scipy`, `matplotlib`, `Pillow`, `opencv-python` and
`imageio-ffmpeg` for the Lecture 4 video demo, plus `jupyterlab` and
`ipykernel`).

### Option A: a virtual environment (pip)

From this `LectureCode/` folder:

**Linux / macOS**

```bash
python3 -m venv MECH_LECTURES
source MECH_LECTURES/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv MECH_LECTURES
MECH_LECTURES\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option B: system packages (Debian / Ubuntu)

```bash
sudo apt install python3-numpy python3-scipy python3-matplotlib python3-pil \
                 python3-opencv jupyter
```

## Launching the notebooks

With the environment active, from this folder:

```bash
jupyter lab
```

then open a notebook from the file browser, or go straight to one:

```bash
jupyter lab Lec4_DFT.ipynb
```

Run cells with **Shift-Enter** (run and advance) or **Ctrl-Enter** (run
in place). Always run the **Setup** cell at the top of a notebook first;
after that the sections are mostly independent.

The paths in the notebooks (`assets/Lec4/...` and so on) are relative to
this folder, so start Jupyter from here.

## Stepping the animations

Some Lecture 4 demos (sum of cosines, greedy reconstruction) were
animations in the original scripts. In the notebook each is split into an
**init** cell that resets the shared state and a **step** cell that
advances it by `STEP` iterations and redraws. Put the cursor in the step
cell and press **Ctrl-Enter** repeatedly to watch it evolve; re-run the
init cell to start over. This is the notebook version of a `for` loop
that draws every iteration: the loop body is a cell and the loop counter
is a global variable.

## Registering the kernel (only if Jupyter does not see it)

`ipykernel` in `requirements.txt` usually means the venv shows up
automatically. If it does not:

```bash
python -m ipykernel install --user --name mech5720
```

then pick the `mech5720` kernel from the notebook's kernel menu.
