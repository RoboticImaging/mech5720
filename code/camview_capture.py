"""
Example receiver client. Run this on your PC / laptop.

- Displays the MJPEG stream from the Pi using OpenCV inside a Dear ImGui window.
- Lets you change camera configurations via floating drop-down menus.
- Supports capturing raw uncompressed frames.

Requires: opencv-python, requests, glfw, pyopengl, imgui-bundle
  pip install opencv-python requests glfw pyopengl imgui-bundle
"""

import argparse
import threading
import time

import cv2
import numpy as np
import requests

import glfw
import OpenGL.GL as gl
# --- CORRECTED IMPORTS ---
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

# Bytes-per-pixel for the pixel formats the server supports.
BYTES_PER_PIXEL = {
    "RGB888": 3,
    "BGR888": 3,
    "RGB565": 2,
    "YUV420": 1,
}

# Shared UI status state
status_message = "Ready"
status_color = (0.5, 0.7, 1.0, 1.0)


def set_status(msg, color=(0.5, 0.7, 1.0, 1.0)):
    global status_message, status_color
    status_message = msg
    status_color = color


def get_config(base_url):
    r = requests.get(f"{base_url}/config", timeout=5)
    r.raise_for_status()
    return r.json()


def change_config_thread(base_url, width, height, fmt):
    set_status(f"Changing config to {width}x{height} {fmt}...", (0.9, 0.6, 0.2, 1.0))
    try:
        r = requests.post(
            f"{base_url}/config",
            json={"width": width, "height": height, "format": fmt},
            timeout=10,
        )
        if r.status_code == 200:
            set_status("Config updated successfully!", (0.2, 0.8, 0.2, 1.0))
        else:
            set_status(f"Error changing config: {r.status_code}", (0.9, 0.2, 0.2, 1.0))
        print("Config change response:", r.status_code, r.json())
    except Exception as e:
        set_status(f"Error: {e}", (0.9, 0.2, 0.2, 1.0))


# Cache for dynamic camera controls
camera_controls = {}


def fetch_camera_controls(base_url):
    global camera_controls
    try:
        r = requests.get(f"{base_url}/controls", timeout=5)
        if r.status_code == 200:
            camera_controls = r.json()
            print("Fetched camera controls from server:", len(camera_controls))
    except Exception as e:
        print("Failed to fetch camera controls:", e)


def set_camera_control_thread(base_url, name, value):
    try:
        r = requests.post(
            f"{base_url}/controls",
            json={name: value},
            timeout=5
        )
        if r.status_code == 200:
            if name in camera_controls:
                camera_controls[name]["current"] = value
            set_status(f"Updated {name} to {value}", (0.2, 0.8, 0.2, 1.0))
            fetch_camera_controls(base_url)
        else:
            set_status(f"Error setting {name}: {r.status_code}", (0.9, 0.2, 0.2, 1.0))
    except Exception as e:
        set_status(f"Error: {e}", (0.9, 0.2, 0.2, 1.0))


def capture_raw_frame(base_url, save_path=None):
    r = requests.get(f"{base_url}/capture_raw", timeout=10)
    r.raise_for_status()

    width = int(r.headers["X-Width"])
    height = int(r.headers["X-Height"])
    fmt = r.headers["X-Format"]
    raw_bytes = r.content

    if "X-Shape" in r.headers and "X-Dtype" in r.headers:
        shape = tuple(map(int, r.headers["X-Shape"].split(",")))
        dtype = np.dtype(r.headers["X-Dtype"])
        arr = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
        print(f"Got raw bayer sensor frame: shape={shape}, dtype={dtype}, size={len(raw_bytes)} bytes")
    else:
        stride = int(r.headers["X-Stride"])
        if fmt == "YUV420":
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            print(f"Got YUV420 raw frame: {len(raw_bytes)} bytes, {width}x{height}, stride={stride}")
            if save_path:
                arr.tofile(save_path)
            return arr, (width, height, fmt)

        bpp = BYTES_PER_PIXEL[fmt]
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        rows = len(raw_bytes) // stride
        arr = arr.reshape((rows, stride))
        arr = arr[:, : width * bpp].reshape((height, width, bpp))
        print(f"Got raw frame: {width}x{height} {fmt}, {len(raw_bytes)} bytes (stride={stride})")

    if save_path:
        # Save as a plain raw binary dump (no header/metadata), regardless of
        # source format. Shape/dtype/format are printed to the console above
        # so the raw bytes can be reinterpreted later if needed.
        arr.tofile(save_path)
        print(f"Saved raw bytes to {save_path} (shape={arr.shape}, dtype={arr.dtype})")

    return arr, (width, height, fmt)


def capture_dng_frame(base_url, save_path=None):
    """
    Downloads a single frame as a real Adobe DNG file from the server's
    /capture_dng endpoint. Unlike /capture_raw (bare pixel bytes), a .dng
    file is a complete, self-describing photographic raw container that
    rawpy, Lightroom, darktable, dcraw, etc. can open directly.
    """
    r = requests.get(f"{base_url}/capture_dng", timeout=15)
    r.raise_for_status()
    dng_bytes = r.content

    if save_path:
        with open(save_path, "wb") as f:
            f.write(dng_bytes)
        print(f"Saved DNG to {save_path} ({len(dng_bytes)} bytes)")

    return dng_bytes


class FrameReader:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.frame = None
        self.new_frame_available = False
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.fps = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        fps_count = 0
        fps_start_time = time.time()
        
        while self.running:
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                with self.lock:
                    self.connected = False
                time.sleep(1.0)
                continue
            
            with self.lock:
                self.connected = True
                
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    with self.lock:
                        self.connected = False
                    break
                
                with self.lock:
                    self.frame = frame
                    self.new_frame_available = True
                    self.connected = True
                
                fps_count += 1
                now = time.time()
                if now - fps_start_time >= 1.0:
                    self.fps = fps_count / (now - fps_start_time)
                    fps_count = 0
                    fps_start_time = now
            
            cap.release()
            time.sleep(0.5)

    def get_frame(self):
        with self.lock:
            is_new = self.new_frame_available
            if is_new:
                self.new_frame_available = False
            return self.frame, is_new

    def get_status(self):
        with self.lock:
            return self.connected

    def stop(self):
        self.running = False


def apply_theme():
    """Sets a beautiful, modern slate-purple dark theme on ImGui."""
    style = imgui.get_style()
    
    style.window_rounding = 10.0
    style.frame_rounding = 6.0
    style.grab_rounding = 6.0
    style.scrollbar_rounding = 6.0
    style.window_border_size = 1.0
    
    # Notice the change to imgui.Col_ here:
    theme_colors = {
        imgui.Col_.text: (0.90, 0.90, 0.95, 1.0),
        imgui.Col_.text_disabled: (0.50, 0.50, 0.55, 1.0),
        imgui.Col_.window_bg: (0.08, 0.08, 0.11, 0.95),
        imgui.Col_.child_bg: (0.08, 0.08, 0.11, 1.0),
        imgui.Col_.popup_bg: (0.12, 0.12, 0.16, 0.98),
        imgui.Col_.border: (0.25, 0.25, 0.35, 0.6),
        imgui.Col_.border_shadow: (0.00, 0.00, 0.00, 0.0),
        imgui.Col_.frame_bg: (0.14, 0.14, 0.20, 1.0),
        imgui.Col_.frame_bg_hovered: (0.20, 0.20, 0.28, 1.0),
        imgui.Col_.frame_bg_active: (0.28, 0.28, 0.38, 1.0),
        imgui.Col_.title_bg: (0.12, 0.12, 0.16, 1.0),
        imgui.Col_.title_bg_active: (0.18, 0.18, 0.24, 1.0),
        imgui.Col_.title_bg_collapsed: (0.08, 0.08, 0.11, 0.50),
        imgui.Col_.menu_bar_bg: (0.14, 0.14, 0.18, 1.0),
        imgui.Col_.scrollbar_bg: (0.10, 0.10, 0.14, 1.0),
        imgui.Col_.scrollbar_grab: (0.25, 0.25, 0.35, 1.0),
        imgui.Col_.scrollbar_grab_hovered: (0.35, 0.35, 0.45, 1.0),
        imgui.Col_.scrollbar_grab_active: (0.45, 0.45, 0.55, 1.0),
        imgui.Col_.check_mark: (0.50, 0.55, 0.90, 1.0),
        imgui.Col_.slider_grab: (0.40, 0.45, 0.80, 1.0),
        imgui.Col_.slider_grab_active: (0.50, 0.55, 0.90, 1.0),
        imgui.Col_.button: (0.20, 0.25, 0.45, 1.0),
        imgui.Col_.button_hovered: (0.28, 0.35, 0.60, 1.0),
        imgui.Col_.button_active: (0.35, 0.45, 0.75, 1.0),
        imgui.Col_.header: (0.20, 0.25, 0.45, 1.0),
        imgui.Col_.header_hovered: (0.28, 0.35, 0.60, 1.0),
        imgui.Col_.header_active: (0.35, 0.45, 0.75, 1.0),
    }

    # Set each color using imgui_bundle's set_color_ function
    for col_enum, col_val in theme_colors.items():
        style.set_color_(col_enum, imgui.ImVec4(*col_val))

ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def process_frame_for_display(frame_bgr, color_order_idx, rotation_deg):
    """
    Takes a raw BGR frame (as decoded by OpenCV's VideoCapture) and returns a
    frame ready to be uploaded to an RGB OpenGL texture, with:
      - color_order_idx == 0 ("RGB"): normal BGR->RGB conversion (correct colors)
      - color_order_idx == 1 ("BGR"): channels left as-is, so red/blue are
        swapped in the displayed image (useful for spotting/undoing a wrong
        RGB/BGR assumption somewhere in the pipeline)
      - rotated by rotation_deg (0/90/180/270) so the stream can be made upright
    """
    if color_order_idx == 0:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        img = frame_bgr.copy()

    rot_code = ROTATIONS.get(rotation_deg)
    if rot_code is not None:
        img = cv2.rotate(img, rot_code)

    return np.ascontiguousarray(img)


def upload_texture(texture_id, rgb_frame):
    h, w, _ = rgb_frame.shape

    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl.GL_RGB,
        w,
        h,
        0,
        gl.GL_RGB,
        gl.GL_UNSIGNED_BYTE,
        rgb_frame
    )
    return w, h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.100")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    
    initial_cfg = None
    try:
        initial_cfg = get_config(base_url)
        print("Connected to server. Current config:", initial_cfg)
        fetch_camera_controls(base_url)
    except Exception as e:
        print("Warning: Could not connect to Pi camera server on start. Will retry in background.")
        
    resolutions = ["640x480", "1280x720", "1920x1080"]
    formats = ["RGB888", "BGR888", "RGB565", "YUV420"]
    
    current_res_idx = 0
    current_fmt_idx = 0
    
    if initial_cfg:
        w = initial_cfg.get("width")
        h = initial_cfg.get("height")
        fmt = initial_cfg.get("format")
        
        res_str = f"{w}x{h}"
        if res_str in resolutions:
            current_res_idx = resolutions.index(res_str)
        else:
            resolutions.append(res_str)
            current_res_idx = len(resolutions) - 1
            
        if fmt in formats:
            current_fmt_idx = formats.index(fmt)
        else:
            formats.append(fmt)
            current_fmt_idx = len(formats) - 1

    if not glfw.init():
        print("Error: Could not initialize GLFW")
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    window = glfw.create_window(1280, 720, "Pi Camera Control Client", None, None)
    if not window:
        glfw.terminate()
        print("Error: Could not create GLFW window")
        return

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # --- RENDERER BINDING SETUP ---
    imgui.create_context()
    impl = GlfwRenderer(window)
    apply_theme()

    # Create texture placeholder
    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, 1, 1, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, b'\x00\x00\x00')

    stream_url = f"{base_url}/video_feed"
    reader = FrameReader(stream_url)

    tex_w, tex_h = 1, 1
    has_texture = False

    # Display processing options
    color_order_options = ["RGB", "BGR"]
    color_order_idx = 0
    rotation_deg = 0
    settings_dirty = False  # forces reprocessing of the last frame when True
    last_display_frame = None  # last processed (color+rotation) RGB-ordered frame, for saving

    global status_message, status_color
    status_message = "Ready"
    status_color = (0.5, 0.7, 1.0, 1.0)

    # Main Render Loop
    while not glfw.window_should_close(window):
        glfw.poll_events()
        
        # Get frame from reader thread
        frame, is_new = reader.get_frame()
        if frame is not None:
            if is_new or not has_texture or settings_dirty:
                try:
                    processed = process_frame_for_display(frame, color_order_idx, rotation_deg)
                    last_display_frame = processed
                    tex_w, tex_h = upload_texture(texture_id, processed)
                    has_texture = True
                    settings_dirty = False
                except Exception as e:
                    print("OpenGL texture upload failed:", e)

        # --- PROCESS HANDLES FOR FRAMES ---
        impl.process_inputs()
        imgui.new_frame()

        # --- Window 1: Video stream window ---
        imgui.set_next_window_size((800, 600), imgui.Cond_.first_use_ever)
        imgui.set_next_window_pos((50, 50), imgui.Cond_.first_use_ever)
        
        imgui.begin("Pi Camera Feed")
        if has_texture and reader.get_status():
            avail = imgui.get_content_region_avail()
            avail_w = avail.x
            avail_h = avail.y
            
            scale = min(avail_w / tex_w, avail_h / tex_h)
            draw_w = tex_w * scale
            draw_h = tex_h * scale
            
            cursor_pos = imgui.get_cursor_pos()
            cursor_x = cursor_pos.x + (avail_w - draw_w) / 2
            cursor_y = cursor_pos.y + (avail_h - draw_h) / 2
            imgui.set_cursor_pos((cursor_x, cursor_y))
            imgui.image(imgui.ImTextureRef(texture_id), imgui.ImVec2(draw_w, draw_h))
        else:
            avail = imgui.get_content_region_avail()
            avail_w = avail.x
            avail_h = avail.y
            
            imgui.set_cursor_pos((avail_w / 2 - 100, avail_h / 2 - 10))
            if reader.get_status():
                imgui.text("Decoding stream...")
            else:
                imgui.text("Connecting to camera stream...")
        imgui.end()

        # --- Window 2: Control Panel ---
        imgui.set_next_window_size((380, 580), imgui.Cond_.first_use_ever)
        imgui.set_next_window_pos((900, 50), imgui.Cond_.first_use_ever)
        
        imgui.begin("Stream Settings")
        imgui.text("Camera Settings")
        imgui.separator()
        imgui.spacing()

        changed_res, new_res_idx = imgui.combo("Resolution", current_res_idx, resolutions)
        if changed_res:
            current_res_idx = new_res_idx
            res_str = resolutions[new_res_idx]
            w, h = map(int, res_str.split('x'))
            fmt_str = formats[current_fmt_idx]
            threading.Thread(target=change_config_thread, args=(base_url, w, h, fmt_str), daemon=True).start()

        changed_fmt, new_fmt_idx = imgui.combo("Format", current_fmt_idx, formats)
        if changed_fmt:
            current_fmt_idx = new_fmt_idx
            res_str = resolutions[current_res_idx]
            w, h = map(int, res_str.split('x'))
            fmt_str = formats[new_fmt_idx]
            threading.Thread(target=change_config_thread, args=(base_url, w, h, fmt_str), daemon=True).start()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text("Display Processing")

        changed_color, new_color_idx = imgui.combo("Color Order", color_order_idx, color_order_options)
        if changed_color:
            color_order_idx = new_color_idx
            settings_dirty = True

        imgui.text(f"Rotation: {rotation_deg}\u00b0")
        if imgui.button("\u25c0 Rotate Left"):
            rotation_deg = (rotation_deg - 90) % 360
            settings_dirty = True
        imgui.same_line()
        if imgui.button("Rotate Right \u25b6"):
            rotation_deg = (rotation_deg + 90) % 360
            settings_dirty = True
        imgui.same_line()
        if imgui.button("Reset"):
            rotation_deg = 0
            settings_dirty = True

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if imgui.collapsing_header("Advanced Camera Controls"):
            if imgui.button("Refresh Controls"):
                threading.Thread(target=fetch_camera_controls, args=(base_url,), daemon=True).start()
            
            imgui.spacing()
            
            if camera_controls:
                imgui.begin_child("ControlsScroll", (0, 200))
                for name in sorted(camera_controls.keys()):
                    ctrl = camera_controls[name]
                    curr = ctrl.get("current")
                    min_v = ctrl.get("min")
                    max_v = ctrl.get("max")

                    if isinstance(curr, bool):
                        changed, new_val = imgui.checkbox(name, curr)
                        if changed:
                            threading.Thread(target=set_camera_control_thread, args=(base_url, name, new_val), daemon=True).start()
                    elif isinstance(curr, (int, float)):
                        is_int = isinstance(curr, int)
                        has_simple_limits = (min_v is not None and max_v is not None and 
                                             ((is_int and (max_v - min_v) <= 100) or 
                                              (not is_int and (max_v - min_v) <= 20.0)))
                        
                        if has_simple_limits:
                            if is_int:
                                changed, ctrl["current"] = imgui.slider_int(name, curr, min_v, max_v)
                            else:
                                changed, ctrl["current"] = imgui.slider_float(name, curr, min_v, max_v)
                            
                            if imgui.is_item_deactivated_after_edit():
                                threading.Thread(target=set_camera_control_thread, args=(base_url, name, ctrl["current"]), daemon=True).start()
                        else:
                            if is_int:
                                changed, ctrl["current"] = imgui.input_int(name, curr)
                            else:
                                changed, ctrl["current"] = imgui.input_float(name, curr)
                                
                            if imgui.is_item_deactivated_after_edit():
                                threading.Thread(target=set_camera_control_thread, args=(base_url, name, ctrl["current"]), daemon=True).start()
                    else:
                        imgui.text(f"{name}: {curr}")
                imgui.end_child()
            else:
                imgui.text("No controls available (click Refresh)")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        
        imgui.text("Actions")
        
        if imgui.button("Capture Raw Frame"):
            def grab():
                set_status("Downloading raw frame...", (0.9, 0.6, 0.2, 1.0))
                try:
                    raw_path = f"captured_frame_{int(time.time())}.raw"
                    arr, meta = capture_raw_frame(base_url, save_path=raw_path)
                    set_status(f"Saved {raw_path}: {meta[0]}x{meta[1]} {meta[2]}", (0.2, 0.8, 0.2, 1.0))
                except Exception as e:
                    set_status(f"Capture error: {str(e)}", (0.9, 0.2, 0.2, 1.0))
            
            threading.Thread(target=grab, daemon=True).start()

        imgui.same_line()

        if imgui.button("Capture DNG"):
            def grab_dng():
                set_status("Downloading DNG frame...", (0.9, 0.6, 0.2, 1.0))
                try:
                    dng_path = f"captured_frame_{int(time.time())}.dng"
                    capture_dng_frame(base_url, save_path=dng_path)
                    set_status(f"Saved {dng_path}", (0.2, 0.8, 0.2, 1.0))
                except Exception as e:
                    set_status(f"DNG capture error: {str(e)}", (0.9, 0.2, 0.2, 1.0))

            threading.Thread(target=grab_dng, daemon=True).start()

        imgui.spacing()

        if imgui.button("Save Current"):
            def save_current(frame_to_save):
                if frame_to_save is None:
                    set_status("No frame to save yet", (0.9, 0.2, 0.2, 1.0))
                    return
                try:
                    # frame_to_save is RGB-ordered (as uploaded to the GL texture);
                    # cv2.imwrite expects BGR, so reverse the channel order to save
                    # a PNG that matches exactly what's currently on screen.
                    bgr_for_save = frame_to_save[:, :, ::-1]
                    png_path = f"frame_{int(time.time())}.png"
                    cv2.imwrite(png_path, bgr_for_save)
                    set_status(f"Saved {png_path}", (0.2, 0.8, 0.2, 1.0))
                except Exception as e:
                    set_status(f"Save error: {str(e)}", (0.9, 0.2, 0.2, 1.0))

            threading.Thread(target=save_current, args=(last_display_frame,), daemon=True).start()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text("Status: ")
        imgui.same_line()
        if reader.get_status():
            imgui.text_colored((0.2, 0.8, 0.2, 1.0), "Connected")
        else:
            imgui.text_colored((0.9, 0.2, 0.2, 1.0), "Disconnected")

        imgui.text("Activity: ")
        imgui.same_line()
        imgui.text_colored(status_color, status_message)

        imgui.spacing()
        imgui.separator()
        imgui.text(f"UI Frame Rate: {imgui.get_io().framerate:.1f} FPS")
        imgui.text(f"Stream Frame Rate: {reader.fps:.1f} FPS")

        imgui.end()

        # Render ImGui scene
        imgui.render()
        
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glClearColor(0.06, 0.06, 0.08, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    # Cleanup
    reader.stop()
    impl.shutdown()
    imgui.destroy_context()
    gl.glDeleteTextures([texture_id])
    glfw.terminate()


if __name__ == "__main__":
    main()
