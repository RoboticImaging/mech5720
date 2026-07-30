"""
Raspberry Pi camera HTTP server.

Run this ON THE PI.

Endpoints:
  GET  /video_feed          -> MJPEG stream (open in browser, VLC, or <img src=...>)
  GET  /config              -> current camera configuration (JSON)
  POST /config              -> change camera configuration (JSON body)
  GET  /capture_raw         -> single frame, bare raw bayer bytes (not openable by rawpy)
  GET  /capture_dng         -> single frame as a real Adobe DNG file (openable by rawpy)
  GET  /                    -> simple HTML page that shows the live stream

Example POST body to /config:
  {"width": 1280, "height": 720, "format": "RGB888"}

Requires: picamera2, flask
  sudo apt install -y python3-picamera2
  pip install flask --break-system-packages
"""

import io
import json
import os
import tempfile
import threading
import time

from flask import Flask, Response, jsonify, request, render_template_string
from picamera2 import Picamera2
import libcamera

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Camera state
# ---------------------------------------------------------------------------

picam2 = Picamera2()
config_lock = threading.Lock()

current_config = {
    "width": 640,
    "height": 480,
    "format": "RGB888",   # RGB888, YUV420, RGB565, etc.
}

# Last-known set of controls the user explicitly requested (e.g. ExposureTime,
# AnalogueGain, AeEnable, ColourGains, ...). picamera2 resets controls to their
# defaults every time the camera is stopped/reconfigured (which happens on
# every /config change), so we keep this around to re-apply them afterward.
# It's also used as the source of truth when reporting "current" values back,
# since capture_metadata() lags a couple of frames behind a control you just set.
manual_controls = {}

VALID_FORMATS = {"RGB888", "YUV420", "RGB565", "BGR888"}


def build_and_apply_config(width, height, fmt):
    """Stop the camera, reconfigure it, and restart it. Blocks briefly."""
    cfg = picam2.create_video_configuration(
        main={"format": fmt, "size": (width, height)},
        raw={"size": (width, height)}
    )
    picam2.configure(cfg)


def start_camera():
    with config_lock:
        build_and_apply_config(
            current_config["width"], current_config["height"], current_config["format"]
        )
        picam2.start()
        if manual_controls:
            picam2.set_controls(manual_controls)


def reconfigure_camera(width, height, fmt):
    with config_lock:
        picam2.stop()
        build_and_apply_config(width, height, fmt)
        picam2.start()
        current_config.update({"width": width, "height": height, "format": fmt})
        # Re-apply any manual controls the user had set, since stop()/configure()
        # resets everything (exposure, gain, AE/AWB toggles, etc.) to defaults.
        if manual_controls:
            picam2.set_controls(manual_controls)


# ---------------------------------------------------------------------------
# MJPEG streaming
# ---------------------------------------------------------------------------

def mjpeg_generator():
    """Yields multipart JPEG frames. Encodes each frame to JPEG on the fly."""
    from PIL import Image

    while True:
        with config_lock:
            frame = picam2.capture_array()
            fmt = current_config["format"]

        # Convert whatever format we're in to something PIL can save as JPEG.
        if fmt in ("RGB888", "BGR888"):
            img = Image.fromarray(frame)
            if fmt == "BGR888":
                img = Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
        elif fmt == "YUV420":
            # capture_array on YUV420 returns a single-plane buffer; for simplicity
            # request RGB888 if you want easy JPEG encoding, or add proper YUV->RGB
            # conversion here (e.g. via OpenCV cv2.cvtColor) if you need raw YUV.
            import cv2
            import numpy as np
            h = current_config["height"]
            w = current_config["width"]
            yuv = frame.reshape((h * 3 // 2, w))
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            img = Image.fromarray(rgb)
        else:
            img = Image.fromarray(frame)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        jpg_bytes = buf.getvalue()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(
        """
        <html>
          <head><title>Pi Camera</title></head>
          <body style="background:#111;color:#eee;font-family:sans-serif;text-align:center">
            <h2>Live Feed</h2>
            <img src="/video_feed" style="max-width:90%;border:2px solid #444"/>
            <p>Current config: <span id="cfg"></span></p>
            <script>
              fetch('/config').then(r => r.json()).then(d => {
                document.getElementById('cfg').innerText = JSON.stringify(d);
              });
            </script>
          </body>
        </html>
        """
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/capture_raw", methods=["GET"])
def capture_raw():
    """
    Captures ONE frame and returns it completely uncompressed (no JPEG/PNG).
    Provides the raw bayer sensor data straight off the sensor prior to debayering.
    """
    with config_lock:
        frame = picam2.capture_array("raw")
        stream_config = picam2.camera_configuration()["raw"]
        stride = stream_config["stride"]
        fmt = str(stream_config["format"])
        width = stream_config["size"][0]
        height = stream_config["size"][1]
        shape_str = ",".join(map(str, frame.shape))
        dtype_str = str(frame.dtype)

        raw_bytes = frame.tobytes()

    headers = {
        "X-Width": str(width),
        "X-Height": str(height),
        "X-Format": fmt,
        "X-Stride": str(stride),
        "X-Shape": shape_str,
        "X-Dtype": dtype_str,
        "Content-Type": "application/octet-stream",
        "Content-Disposition": "attachment; filename=frame.raw",
    }
    return Response(raw_bytes, headers=headers)


@app.route("/capture_dng", methods=["GET"])
def capture_dng():
    """
    Captures ONE frame and returns it as an Adobe DNG (Digital Negative) file:
    a real photographic raw container (still the same raw bayer sensor data
    as /capture_raw) but wrapped with the Bayer pattern, black level, white
    balance, and colour matrix tags a raw file needs. This is what rawpy
    (and Lightroom/darktable/dcraw/etc.) actually expect to open - a bare
    .raw dump of pixel bytes has none of that, which is why rawpy rejects it.
    """
    with config_lock:
        cam_request = picam2.capture_request()
        try:
            raw_buffer = cam_request.make_buffer("raw")
            metadata = cam_request.get_metadata()
            raw_stream_config = picam2.camera_configuration()["raw"]
        finally:
            cam_request.release()

    # picamera2's helper only writes to a path, so stage it in a temp file
    # and stream the bytes back rather than saving anything on the Pi itself.
    fd, tmp_path = tempfile.mkstemp(suffix=".dng")
    os.close(fd)
    try:
        picam2.helpers.save_dng(raw_buffer, metadata, raw_stream_config, tmp_path)
        with open(tmp_path, "rb") as f:
            dng_bytes = f.read()
    finally:
        os.remove(tmp_path)

    headers = {
        "Content-Type": "image/x-adobe-dng",
        "Content-Disposition": "attachment; filename=frame.dng",
    }
    return Response(dng_bytes, headers=headers)


@app.route("/controls", methods=["GET"])
def get_controls():
    """Returns all available camera controls and their current values."""
    with config_lock:
        controls_limits = picam2.camera_controls
        try:
            metadata = picam2.capture_metadata()
        except Exception:
            metadata = {}

    response_data = {}
    for name, limits in controls_limits.items():
        # Prefer the value we know we asked for; capture_metadata() reflects
        # the last captured frame and can lag a couple of frames behind a
        # control that was just set, which otherwise makes the UI look like
        # the change "reverted" right after you made it.
        if name in manual_controls:
            curr_val = manual_controls[name]
        else:
            curr_val = metadata.get(name, limits[2] if len(limits) > 2 else None)
        try:
            min_val = limits[0]
            max_val = limits[1]
            def_val = limits[2] if len(limits) > 2 else None

            # Test JSON serializability
            json.dumps(min_val)
            json.dumps(max_val)
            json.dumps(def_val)
            json.dumps(curr_val)

            response_data[name] = {
                "min": min_val,
                "max": max_val,
                "default": def_val,
                "current": curr_val,
            }
        except Exception:
            # Fallback to string representations for non-serializable objects (like Enums or Tuples)
            response_data[name] = {
                "min": str(limits[0]),
                "max": str(limits[1]),
                "default": str(limits[2]) if len(limits) > 2 else None,
                "current": str(curr_val) if curr_val is not None else None,
            }

    return jsonify(response_data)


@app.route("/controls", methods=["POST"])
def set_controls():
    """Updates camera controls dynamically, converting incoming values to correct types."""
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "invalid or missing JSON body"}), 400

    print(f"Received controls request: {data}")
    with config_lock:
        try:
            controls_limits = picam2.camera_controls
            typed_data = {}
            for k, v in data.items():
                if k in controls_limits:
                    limit_val = controls_limits[k][2] if len(controls_limits[k]) > 2 else controls_limits[k][0]
                    if isinstance(limit_val, bool):
                        if isinstance(v, str):
                            typed_data[k] = v.lower() in ("true", "1")
                        else:
                            typed_data[k] = bool(v)
                    elif isinstance(limit_val, int):
                        typed_data[k] = int(v)
                    elif isinstance(limit_val, float):
                        typed_data[k] = float(v)
                    else:
                        typed_data[k] = v
                else:
                    typed_data[k] = v

            # Automatically disable AE/AWB if manual exposure/gain/color gains are set
            if "ExposureTime" in typed_data or "AnalogueGain" in typed_data:
                if "AeEnable" not in typed_data and "AeEnable" in controls_limits:
                    typed_data["AeEnable"] = False
                    print("Auto-disabling AeEnable for manual exposure/gain control")
            if "ColourGains" in typed_data:
                if "AwbEnable" not in typed_data and "AwbEnable" in controls_limits:
                    typed_data["AwbEnable"] = False
                    print("Auto-disabling AwbEnable for manual color gains control")

            print(f"Applying controls to picamera2: {typed_data}")
            picam2.set_controls(typed_data)
            manual_controls.update(typed_data)
            return jsonify({"status": "applied", "controls": {k: str(v) for k, v in typed_data.items()}})
        except Exception as e:
            print(f"Error setting controls: {e}")
            return jsonify({"error": str(e)}), 500


@app.route("/config", methods=["GET"])
def get_config():
    return jsonify(current_config)


@app.route("/config", methods=["POST"])
def set_config():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "invalid or missing JSON body"}), 400

    width = data.get("width", current_config["width"])
    height = data.get("height", current_config["height"])
    fmt = data.get("format", current_config["format"])

    if fmt not in VALID_FORMATS:
        return jsonify({"error": f"format must be one of {sorted(VALID_FORMATS)}"}), 400
    if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0):
        return jsonify({"error": "width/height must be positive integers"}), 400

    try:
        reconfigure_camera(width, height, fmt)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "reconfigured", "config": current_config})


if __name__ == "__main__":
    start_camera()
    # threaded=True lets /config requests be handled while /video_feed is streaming
    app.run(host="0.0.0.0", port=8000, threaded=True)
