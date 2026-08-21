#!/usr/bin/env python3
"""
capture_relighting.py -- MECH5720 Lab 2

Capture a scene under multiplexed illumination.

A monitor acts as the illumination source: it is divided into a grid of
patches ("illumination sites") that are switched on and off in a known
sequence, while the camera records the scene's appearance under each
pattern. The resulting image stacks are what relighting.py consumes.

Five stacks are produced:

    Multiplexed*.dng    the multiplexing patterns from H  (needed)
    Ambient*.dng        all sites off                     (needed)
    Impulse*.dng        one site on at a time             (comparison)
    AllOn*.dng          all sites on                      (comparison)
    FirstOn*.dng        only site 1 on                    (comparison)

Frames are written as DNG (untouched sensor raw, plus the metadata --
black level, colour gains, Bayer pattern -- needed to interpret it),
not PNG: no bit-shifting or scaling happens on the way to disk.

Run this on the Raspberry Pi, with the illumination monitor connected.
The illumination window goes fullscreen. To start the run, press ENTER
either on the illumination screen itself (if a keyboard is plugged into
the Pi) or in the SSH terminal you launched the script from -- type
'q' + ENTER in the terminal to abort instead. You never need physical
access to the Pi once the run starts.

Over SSH you must also tell it which display to draw on:

    DISPLAY=:0 python3 capture_relighting.py

Run with --preflight first. It checks that your exposure, gain and
ambient stability are good enough to be worth capturing 78 frames.
"""

import os
import sys
import time
import queue
import argparse
import threading

import numpy as np
from scipy.linalg import hadamard

import tkinter as tk


# ======================================================================
# Configuration
# ======================================================================

OUT_PATH = "data/captured"

GRID = (5, 3)              # illumination sites: (columns, rows) == 15
N_SITES = GRID[0] * GRID[1]

N_AMBIENT = 16
N_ALLON = 16
N_FIRSTON = 16

# Camera. Short exposure and high gain, deliberately: multiplexing pays
# off when the noise is signal-independent (read/thermal), and loses its
# advantage when photon noise dominates. A single-site frame SHOULD look
# grainy. If it looks clean, the lab has nothing to show.
ANALOGUE_GAIN = 8.0
EXPOSURE_US = None         # None = set automatically from the ambient level
AMBIENT_TARGET = 0.30      # aim ambient frames at this fraction of range

SETTLE_S = 0.25            # wait after a pattern change
DISCARD_FRAMES = 2         # frames thrown away after each pattern change

RAW_BLACK_LEVEL = 64       # OV5647, in 10-bit counts. See report at end.
RAW_MAX = 1023

# Disk writing. Frames are saved as DNG, built from a plain numpy array
# plus the frame's metadata (black level, colour gains, Bayer pattern),
# NOT from a live camera "request" object. That distinction matters on
# a memory-constrained board: a request holds one of the camera's raw
# buffers until released, and those buffers come out of a small pool
# of GPU/display-shared memory (CMA), not ordinary RAM. An earlier
# version of this held requests open for the duration of the (slow)
# DNG write and raised CAMERA_BUFFER_COUNT to compensate -- which ate
# into the memory the Wayland/X compositor needs for its own fullscreen
# surface and crashed the display session. Copying the array out and
# releasing the request immediately avoids that entirely: the slow
# part (the actual write) then happens on ordinary heap memory,
# completely decoupled from camera buffers, so the buffer count can
# stay as low as the original PNG version's.
CAMERA_BUFFER_COUNT = 2
WRITE_WORKERS = 3          # Pi 3B has 4 cores; leave one for capture/Tk
WRITE_QUEUE_DEPTH = 4


# ======================================================================
# Illumination patterns
# ======================================================================

def s_matrix(n):
    """S-matrix multiplexing patterns, one row per multiplexed frame.

    Built from a normalised Hadamard matrix of order n+1 by deleting the
    first row and column and mapping to {0, 1}. Valid for n = 4t-1.
    Follows Schechner et al. 2007, "Multiplexing for Optimal Lighting".
    """
    h = hadamard(n + 1)
    return (1 - h[1:, 1:]) / 2


def build_schedule(H):
    """Build the full capture schedule.

    Returns (patterns, labels): an (N_total, N_SITES) array of on/off
    patterns, and a matching list of stack names.

    The ambient frames are spread evenly through the sequence rather
    than captured as one block. Ambient light drifts over the several
    minutes a capture takes, and a block captured at one moment is a
    biased estimate of the ambient during the rest of the run. Spreading
    them means the average tracks the middle of the sequence, and lets
    you measure the drift rather than merely suffer it.
    """
    core = []
    for i in range(N_SITES):                       # impulse
        p = np.zeros(N_SITES); p[i] = 1
        core.append((p, "Impulse"))
    for i in range(len(H)):                        # multiplexed
        core.append((H[i], "Multiplexed"))
    for _ in range(N_ALLON):                       # all on
        core.append((np.ones(N_SITES), "AllOn"))
    for _ in range(N_FIRSTON):                     # first site only
        p = np.zeros(N_SITES); p[0] = 1
        core.append((p, "FirstOn"))

    off = (np.zeros(N_SITES), "Ambient")
    step = len(core) / float(N_AMBIENT)

    schedule, placed = [], 0
    for k, item in enumerate(core):
        while placed < N_AMBIENT and k >= placed * step:
            schedule.append(off)
            placed += 1
        schedule.append(item)
    while placed < N_AMBIENT:                      # any remainder, at the end
        schedule.append(off)
        placed += 1

    patterns = np.array([p for p, _ in schedule])
    labels = [n for _, n in schedule]
    return patterns, labels

def grab_mean(cam, n=8):
    """Grab n frames and compute their mean without allocating a large stack."""
    acc = None
    for _ in range(n):
        frame = grab(cam, discard=0).astype(np.float32)
        if acc is None:
            acc = frame
        else:
            acc += frame
    return acc / n


# ======================================================================
# Illumination display
# ======================================================================

class LightStage:
    """Fullscreen grid of illumination patches on the chosen display."""

    def __init__(self, grid=GRID):
        cols, rows = grid
        self.cols, self.rows = cols, rows

        self.root = tk.Tk()
        self.root.title("MECH5720 light stage")
        self.root.configure(bg="black")
        w = self.root.winfo_screenwidth()
        h = self.root.winfo_screenheight()
        self.root.geometry("%dx%d+0+0" % (w, h))
        self.root.attributes("-fullscreen", True)
        self.root.config(cursor="none")

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.update()

        cw = self.canvas.winfo_width() / cols
        ch = self.canvas.winfo_height() / rows
        self.cells = []
        for r in range(rows):
            for c in range(cols):
                self.cells.append(self.canvas.create_rectangle(
                    c * cw, r * ch, (c + 1) * cw, (r + 1) * ch,
                    fill="black", outline=""))

        # Message text, created last so it draws over the patches. It is
        # hidden during capture: anything drawn here is light falling on
        # the scene, so it must be off while frames are being recorded.
        self.text = self.canvas.create_text(
            self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
            text="", fill="#808080", font=("DejaVu Sans", 28),
            justify="center", state="hidden")

        self.root.focus_force()

    def show(self, pattern):
        for cell, on in zip(self.cells, pattern):
            self.canvas.itemconfig(cell, fill="white" if on else "black")
        self.root.update()

    def message(self, text):
        """Show text over the illumination area. Not during capture."""
        self.canvas.itemconfig(self.text, text=text, state="normal")
        self.canvas.tag_raise(self.text)
        self.root.update()

    def clear_message(self):
        self.canvas.itemconfig(self.text, state="hidden")
        self.root.update()

    def wait_for_key(self, prompt):
        """Block until start is signalled, with the prompt shown on screen.

        The signal can come from either of two places:

          - the illumination window itself, via Tk key bindings, for
            when a keyboard is plugged directly into the Pi; or
          - the SSH terminal the script was launched from, which is the
            normal case: the fullscreen illumination window usually
            doesn't have keyboard focus over SSH (no window manager is
            giving it focus, and the operator often has no physical
            keyboard at the Pi at all).

        Terminal input is read on a background daemon thread (a blocking
        line read on stdin) and handed to the Tk main loop through a
        queue, which the main loop drains with root.after() polling.
        Tk objects must only be touched from the Tk thread, so the
        reader thread never calls into Tk directly -- it just posts
        strings.

        At the terminal: press ENTER (a blank line) to start, or type
        'q' then ENTER to abort. On the illumination window: Return /
        KP_Enter to start, Escape to abort.

        Returns False if aborted.
        """
        state = {"go": False}
        done = {"flag": False}
        stdin_q = queue.Queue()

        def start(_event=None):
            if done["flag"]:
                return
            done["flag"] = True
            state["go"] = True
            self.root.quit()

        def abort(_event=None):
            if done["flag"]:
                return
            done["flag"] = True
            self.root.quit()

        def read_stdin():
            # Blocking readline loop in a daemon thread. If the user
            # never types anything this thread just sits here forever,
            # which is harmless -- it's killed when the process exits.
            try:
                for line in sys.stdin:
                    stdin_q.put(line.strip())
                    if done["flag"]:
                        return
            except Exception:
                pass

        def poll_stdin():
            if done["flag"]:
                return
            try:
                line = stdin_q.get_nowait()
            except queue.Empty:
                self.root.after(100, poll_stdin)
                return
            if line.lower() in ("q", "esc", "abort", "escape"):
                abort()
            else:
                start()

        self.message(prompt)
        self.root.bind("<Return>", start)
        self.root.bind("<KP_Enter>", start)
        self.root.bind("<Escape>", abort)
        self.root.focus_force()

        reader = threading.Thread(target=read_stdin, daemon=True)
        reader.start()
        self.root.after(100, poll_stdin)

        self.root.mainloop()
        self.root.unbind("<Return>")
        self.root.unbind("<KP_Enter>")
        self.root.unbind("<Escape>")
        self.clear_message()
        return state["go"]

    def close(self):
        self.root.destroy()


# ======================================================================
# Camera
# ======================================================================

def open_camera(exposure_us, gain):
    """Configure the camera for raw capture with everything held fixed.

    Nothing may adapt between frames. An auto-exposure or white-balance
    excursion partway through the sequence breaks the linear relationship
    between frames that demultiplexing depends on, and the failure is
    invisible until the error image shows scene structure.

    Returns None if no camera is available. The script then runs as a
    rehearsal: it steps through every illumination pattern at the real
    timing but records nothing, which is how you check the patch layout,
    the monitor position and the framing before committing to a capture.
    """
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
    except Exception as err:
        print("\n*** No camera available: %s" % err)
        print("*** Running as a rehearsal: patterns will be displayed,")
        print("*** but no frames will be captured or saved.\n")
        return None

    cfg = cam.create_still_configuration(
        raw={"format": "SBGGR10", "size": cam.sensor_resolution},
        buffer_count=CAMERA_BUFFER_COUNT)
    cam.configure(cfg)
    cam.set_controls({
        "AeEnable": False,
        "AwbEnable": False,
        "AnalogueGain": float(gain),
        "ExposureTime": int(exposure_us),
    })
    cam.start()
    time.sleep(2.0)          # let the pipeline settle on the fixed values
    return cam


def grab(cam, discard=DISCARD_FRAMES):
    """Capture one raw frame, discarding the first few after a change.

    The display, the camera pipeline and the sensor's own latency mean
    the first frame or two after a pattern change may still show the
    previous pattern. A stack silently offset by one row demultiplexes
    to something that looks almost right, which is worse than an
    obvious failure.
    """
    if cam is None:
        return None
    for _ in range(discard):
        cam.capture_array("raw")
    raw = cam.capture_array("raw")
    return raw.view(np.uint16) if raw.dtype == np.uint8 else raw


def grab_capture(cam, discard=DISCARD_FRAMES):
    """Capture one frame, ready to be saved as DNG later.

    Extracts the pixel array and metadata (black level, colour gains,
    Bayer pattern, timestamp) from the camera's request and releases
    the request immediately -- before any of the slow DNG-writing work
    happens. See the comment above CAMERA_BUFFER_COUNT for why this
    matters: holding a request open for the duration of a disk write
    ties up scarce camera/display-shared memory. .copy() is required
    on the array: make_array() gives a view straight onto the request's
    buffer, which is only valid until release(), not a value the array
    can safely outlive on its own.

    Same discard logic as grab(), for the same reason -- the first
    frame or two after a pattern change can still show the previous
    pattern.
    """
    if cam is None:
        return None
    for _ in range(discard):
        cam.capture_array("raw")
    request = cam.capture_request()
    try:
        array = request.make_array("raw").copy()
        metadata = request.get_metadata()
    finally:
        request.release()
    return array, metadata


def meter(cam, stage, target=AMBIENT_TARGET, gain=ANALOGUE_GAIN):
    """Choose an exposure so the ambient-only frame sits at `target`.

    Exposure is set from the AMBIENT level, not from the illuminated
    level. The screen contributes whatever it contributes on top; you
    are not trying to fill the sensor with it.
    """
    if cam is None:
        return None, None

    stage.show(np.zeros(N_SITES))
    time.sleep(SETTLE_S)

    exposure = 5000
    for _ in range(8):
        cam.set_controls({"ExposureTime": int(exposure),
                          "AnalogueGain": float(gain)})
        time.sleep(0.4)
        level = grab(cam).mean() / RAW_MAX
        level = max(level - RAW_BLACK_LEVEL / RAW_MAX, 1e-4)
        if abs(level - target) / target < 0.15:
            break
        exposure = int(np.clip(exposure * target / level, 200, 200000))
    return exposure, level


# ======================================================================
# Preflight
# ======================================================================

def preflight(cam, stage, H):
    """Check the capture is worth doing before committing to 78 frames.

    Measures the per-site signal, the noise, and how much the ambient
    moves over the course of a short run.
    """
    print("\n--- preflight ---")

    if cam is None:
        print("  no camera, nothing to measure")
        return True

    stage.show(np.zeros(N_SITES))
    time.sleep(SETTLE_S)
    amb_a = grab_mean(cam, 8) #np.stack([grab(cam, 0) for _ in range(8)]).astype(np.float64)

    stage.show(np.ones(N_SITES))
    time.sleep(SETTLE_S)
    allon = grab(cam).astype(np.float64)

    time.sleep(20.0)                       # let some time pass
    stage.show(np.zeros(N_SITES))
    time.sleep(SETTLE_S)
    amb_b = grab_mean(cam, 8) #np.stack([grab(cam, 0) for _ in range(8)]).astype(np.float64)

    scale = float(RAW_MAX)
    ambient = amb_a.mean(axis=0)
    sigma = amb_a.std(axis=0).mean() / scale
    per_site = (allon - ambient).mean() / scale / N_SITES
    drift = (amb_b.mean() - amb_a.mean()) / scale

    snr = per_site / sigma if sigma > 0 else 0.0
    drift_ratio = abs(drift) / per_site if per_site > 0 else np.inf

    print("  ambient level          %8.4f  of full scale" % (ambient.mean() / scale))
    print("  per-frame noise sigma  %8.5f" % sigma)
    print("  per-site signal        %8.5f" % per_site)
    print("  per-site SNR           %8.2f   (want > 5)" % snr)
    print("  ambient drift          %+8.5f" % drift)
    print("  drift / signal         %8.2f   (want < 0.2)" % drift_ratio)

    ok = snr > 5.0 and drift_ratio < 0.2
    print("  ---> %s" % ("PASS" if ok else "FAIL, see above"))
    if not ok:
        print("\n  Low SNR: raise gain, lengthen exposure, or move the")
        print("  monitor closer. Large drift: close curtains, stop anyone")
        print("  walking past the scene, and check nothing is auto-dimming.")
    return ok


# ======================================================================
# Capture
# ======================================================================

def save_frame(cam, array, metadata, raw_config, out_path, name, index):
    """Write a captured frame as a DNG raw file.

    Builds the DNG from a plain array + metadata + stream config via
    Picamera2's helpers.save_dng() -- picamera2's documented way to
    write a DNG once you're no longer holding the request that produced
    it (that's the whole point: see the comment above
    CAMERA_BUFFER_COUNT for why we don't hold requests open here). No
    bit-shifting or scaling happens: the DNG carries the sensor's own
    raw counts and its own metadata, and a reader (rawpy, dcraw, etc.)
    works them out itself.

    NOTE: `cam.helpers` is the picamera2 API this relies on. If your
    installed picamera2 version doesn't have it, `dir(cam.helpers)` on
    the Pi will show what's actually available -- the fix then is
    substituting whatever your version's equivalent free function is,
    the array/metadata/config extraction in grab_capture() stays the
    same either way.
    """
    cam.helpers.save_dng(
        array, metadata, raw_config,
        os.path.join(out_path, "%s%03d.dng" % (name, index)))


class AsyncFrameWriter:
    """Background DNG writer so capture() never blocks on disk I/O.

    capture() hands off an already-copied (array, metadata) pair with
    put() and moves straight on to the next pattern; a small pool of
    worker threads drains the queue and does the DNG write concurrently.
    Nothing here touches camera buffers -- grab_capture() has already
    copied the pixels out and released the request before this ever
    sees the frame -- so the queue can be as deep as memory allows
    without needing more camera buffers. It's still bounded, so a slow
    writer applies backpressure (put() blocks) rather than letting
    frames pile up unboundedly.
    """

    def __init__(self, cam, raw_config, out_path,
                n_workers=WRITE_WORKERS, depth=WRITE_QUEUE_DEPTH):
        os.makedirs(out_path, exist_ok=True)
        self.cam = cam
        self.raw_config = raw_config
        self.out_path = out_path
        self.q = queue.Queue(maxsize=depth)
        self.errors = []
        self._lock = threading.Lock()
        self.workers = [threading.Thread(target=self._run, daemon=True)
                        for _ in range(n_workers)]
        for w in self.workers:
            w.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            array, metadata, name, index = item
            try:
                save_frame(self.cam, array, metadata, self.raw_config,
                          self.out_path, name, index)
            except Exception as err:
                with self._lock:
                    self.errors.append("%s%03d: %s" % (name, index, err))
            finally:
                self.q.task_done()

    def put(self, array, metadata, name, index):
        """Queue a captured frame for writing. Blocks if the queue is full."""
        self.q.put((array, metadata, name, index))

    def pending(self):
        return self.q.qsize()

    def close(self):
        """Wait for every queued frame to be written, then stop the workers."""
        self.q.join()
        for _ in self.workers:
            self.q.put(None)
        for w in self.workers:
            w.join()
        if self.errors:
            print("\n  *** %d frame(s) failed to write:" % len(self.errors))
            for e in self.errors[:10]:
                print("      %s" % e)


def capture(cam, stage, patterns, labels, out_path):
    counters = {}
    t0 = time.time()
    raw_config = cam.camera_configuration()["raw"] if cam is not None else None
    writer = AsyncFrameWriter(cam, raw_config, out_path) if cam is not None else None

    for k, (pattern, name) in enumerate(zip(patterns, labels)):
        stage.show(pattern)
        time.sleep(SETTLE_S)
        captured = grab_capture(cam)

        counters[name] = counters.get(name, 0) + 1
        if captured is not None:
            array, metadata = captured
            writer.put(array, metadata, name, counters[name])

        if (k + 1) % 10 == 0 or k + 1 == len(patterns):
            print("  %3d / %3d frames captured  (%.0f s elapsed, %d queued for writing)"
                  % (k + 1, len(patterns), time.time() - t0, writer.pending() if writer else 0))

    if writer is not None:
        t1 = time.time()
        print("  capture done in %.0f s, flushing %d queued frame(s) to disk..."
              % (t1 - t0, writer.pending()))
        writer.close()
        print("  writing finished in %.0f s (total %.0f s)"
              % (time.time() - t1, time.time() - t0))

    return counters


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true",
                    help="run the checks only, capture nothing")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--gain", type=float, default=ANALOGUE_GAIN)
    ap.add_argument("--exposure", type=int, default=EXPOSURE_US)
    args = ap.parse_args()

    H = s_matrix(N_SITES)
    patterns, labels = build_schedule(H)

    print("Illumination grid   %d x %d = %d sites"
          % (GRID[0], GRID[1], N_SITES))
    print("Sites per mux frame %d" % int(H[0].sum()))
    print("Total frames        %d" % len(patterns))
    print("Sites are numbered from the top-left patch, left to right")
    print("along each row, then down. Watch the monitor to confirm.")

    cam = open_camera(args.exposure or 10000, args.gain)
    stage = LightStage(GRID)

    if cam is not None and args.exposure is None:
        print("\nMetering from the ambient level...")
        exposure, level = meter(cam, stage, gain=args.gain)
        cam.set_controls({"ExposureTime": int(exposure),
                          "AnalogueGain": float(args.gain)})
        time.sleep(1.0)
        print("  exposure %d us, gain %.1f -> ambient %.3f of range"
              % (exposure, args.gain, level))

    # The only "keystroke" this script asks for. It can be sent from the
    # illumination window (if a keyboard is plugged into the Pi) or,
    # normally, from this SSH terminal: press ENTER here to start, or
    # type 'q' + ENTER to abort. Everything after it runs unattended,
    # so the rig can be covered and left alone.
    print("\nCover the rig, then press ENTER here in the terminal")
    print("(or on the illumination screen) to start. Type 'q' + ENTER to abort.")
    if not stage.wait_for_key(
            "Cover the rig, then press ENTER --\n"
            "on this screen, or in your SSH terminal -- to start.\n\n"
            "%d frames, about %d minutes.\n"
            "Type 'q' + ENTER in the terminal, or press ESC here, to abort."
            % (len(patterns), max(1, round(len(patterns) * SETTLE_S / 60.0)))):
        print("Aborted.")
        stage.close()
        if cam is not None:
            cam.stop()
        return 1

    ok = preflight(cam, stage, H)

    if args.preflight:
        stage.close()
        if cam is not None:
            cam.stop()
        return 0 if ok else 1

    if not ok:
        print("\nPreflight failed. Capturing anyway; the results may not be")
        print("usable, so check the numbers above before relying on them.")

    print("\n--- capturing ---")
    counters = capture(cam, stage, patterns, labels, args.out)

    stage.show(np.zeros(N_SITES))
    if cam is not None:
        stage.message("Capture complete.\n\n%d frames written to\n%s"
                      % (len(patterns), args.out))
    else:
        stage.message("Rehearsal complete.\n\nNothing was saved.")
    time.sleep(4.0)
    stage.close()
    if cam is not None:
        cam.stop()
        print("\nWrote to %s:" % args.out)
        for name in ("Multiplexed", "Ambient", "Impulse", "AllOn", "FirstOn"):
            print("  %-12s %d frames" % (name, counters.get(name, 0)))
        print("\nIn relighting.py set:")
        print("  DATA_PATH = %r" % args.out)
        print("  N_SITES   = %d" % N_SITES)
        print("\nFrames are now .dng, not .png -- load_stack will need a raw")
        print("reader (e.g. rawpy) instead of PIL, and the black level no")
        print("longer needs setting by hand: it's in each DNG's own tags.")
    else:
        print("\nRehearsal complete: %d patterns displayed, nothing saved."
              % len(patterns))
    return 0


if __name__ == "__main__":
    sys.exit(main())