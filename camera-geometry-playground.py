#!/usr/bin/env python3
"""
Interactive Camera Geometrry Playground
==================================

The input image is treated as a planar object lying in 3D space at Z = Z_PLANE.
A virtual pinhole camera (intrinsics K, extrinsics R|T, radial distortion k1..k3)
"photographs" that plane, and the result is shown live while you move the sliders.

Pipeline executed every time a slider changes
---------------------------------------------
  1. Build the intrinsic matrix K from fx, fy, cx, cy.
  2. Build the rotation matrix R from the Euler angles (Pitch/Yaw/Roll) via cv2.Rodrigues.
  3. Build the plane-to-image homography:
         H = K * [r1 | r2 | Z*r3 + T] * M
     where r1, r2, r3 are the columns of R and M maps source pixels to plane coordinates.
  4. Warp the image with cv2.warpPerspective(H)  -> ideal (undistorted) pinhole image.
  5. Inject radial distortion (k1, k2, k3) with a meshgrid + cv2.remap.

Camera model:   X_cam = R * X_world + T      (world = the plane, camera at the origin)
  - T = 0  -> pure camera rotation (H = K R K^-1 when Z_PLANE = 1)
  - Tz > 0 -> the plane moves away from the camera, Tz < 0 -> closer.

Default slider positions are chosen so that the output equals the input image
(identity: R = I, T = 0, f = max(w, h), principal point at the image center, k = 0).

Keys (focus the image window)
-----------------------------
  r        reset all sliders
  s        save the current output as 'calibration_lab_output.png' + print K, R, T, dist
  h        toggle the on-screen parameter overlay
  q / ESC  quit

Usage
-----
  python camera_calibration_lab.py [path/to/image.jpg]

Requirements: pip install opencv-python numpy
"""

import os
import sys

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_IMAGE_PATH = "img/scacchiera.jpg"  # used when no CLI argument is given
MAX_DIM = 900          # the image is downscaled so its longest side is <= MAX_DIM (keeps UI fluid)
Z_PLANE = 1.0          # depth of the image plane in world coordinates
EPS_DEPTH = 1e-3       # minimum allowed depth for plane corners (avoids points behind the camera)
MAX_COND = 1e10        # maximum allowed condition number of H (avoids numerically singular H)
DIST_ITERS = 8         # fixed-point iterations used to invert the radial distortion model
IMG_WIN = "Camera Lab - Output"
CTRL_WIN = "Camera Lab - Controls"
OUTPUT_FILE = "calibration_lab_output.png"


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def make_synthetic_chessboard(width=800, height=600, squares=(9, 7)):
    """Fallback test image (chessboard with a colored border) used if the file is missing."""
    img = np.full((height, width, 3), 255, np.uint8)
    cols, rows = squares
    sq = min(width // (cols + 1), height // (rows + 1))
    x0 = (width - cols * sq) // 2
    y0 = (height - rows * sq) // 2
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                cv2.rectangle(img, (x0 + c * sq, y0 + r * sq),
                              (x0 + (c + 1) * sq - 1, y0 + (r + 1) * sq - 1), (0, 0, 0), -1)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (0, 0, 255), 6)  # red frame shows image borders
    return img


def load_image(path):
    """Load the image from disk, downscale it if huge; fall back to a synthetic chessboard."""
    img = cv2.imread(path, cv2.IMREAD_COLOR) if os.path.isfile(path) else None
    if img is None:
        print(f"[WARN] Cannot read '{path}'. Using a synthetic chessboard instead.")
        img = make_synthetic_chessboard()
    h, w = img.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------------------------- #
# Slider definitions
# --------------------------------------------------------------------------- #
def make_slider_specs(w, h):
    """
    OpenCV trackbars are integers in [0, count]. Each spec maps a real range onto that:
        real_value = lo + position * step
    Returns an ordered list of (key, trackbar_label, lo, hi, step, default_real_value).
    Ranges are limited to values where the math stays well-conditioned.
    """
    f0 = float(max(w, h))  # reference focal length: ~53 deg FOV on the longest side
    return [
        # ---- Extrinsics: translation (in plane-depth units, plane sits at depth 1.0) ----
        ("tx",    "Tx",            -2.0,      2.0,      0.01, 0.0),
        ("ty",    "Ty",            -2.0,      2.0,      0.01, 0.0),
        ("tz",    "Tz",            -0.9,      4.0,      0.01, 0.0),   # depth = 1 + Tz >= 0.1
        # ---- Extrinsics: Euler angles in degrees (limited to avoid an edge-on plane) ----
        ("pitch", "Pitch X (deg)", -85.0,     85.0,     0.5,  0.0),
        ("yaw",   "Yaw Y (deg)",   -85.0,     85.0,     0.5,  0.0),
        ("roll",  "Roll Z (deg)",  -180.0,    180.0,    0.5,  0.0),
        # ---- Intrinsics (pixels) ----
        ("fx",    "fx (px)",       0.25 * f0, 4.0 * f0, 1.0,  f0),
        ("fy",    "fy (px)",       0.25 * f0, 4.0 * f0, 1.0,  f0),
        ("cx",    "cx (px)",       0.0,       float(w), 1.0,  w / 2.0),
        ("cy",    "cy (px)",       0.0,       float(h), 1.0,  h / 2.0),
        # ---- Radial distortion ----
        ("k1",    "k1",            -1.0,      1.0,      0.01, 0.0),
        ("k2",    "k2",            -1.0,      1.0,      0.01, 0.0),
        ("k3",    "k3",            -1.0,      1.0,      0.01, 0.0),
    ]


def real_to_pos(spec, value):
    _, _, lo, _, step, _ = spec
    return int(round((value - lo) / step))


def read_params(specs):
    """Read all trackbars and convert integer positions into real-world parameter values."""
    params = {}
    for spec in specs:
        key, label, lo, hi, step, _ = spec
        pos = cv2.getTrackbarPos(label, CTRL_WIN)
        params[key] = min(max(lo + pos * step, lo), hi)  # clamp for safety
    return params


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def build_intrinsics(fx, fy, cx, cy):
    """Pinhole intrinsic matrix K (zero skew)."""
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def euler_to_rotation(pitch_deg, yaw_deg, roll_deg):
    """
    Euler angles (degrees) -> 3x3 rotation matrix, built with cv2.Rodrigues per axis.
    Composition order: R = Rz(roll) @ Ry(yaw) @ Rx(pitch).
    """
    def axis_rot(axis, deg):
        rvec = np.zeros(3, dtype=np.float64)
        rvec[axis] = np.radians(deg)          # Rodrigues vector = axis * angle
        R, _ = cv2.Rodrigues(rvec)
        return R

    Rx = axis_rot(0, pitch_deg)
    Ry = axis_rot(1, yaw_deg)
    Rz = axis_rot(2, roll_deg)
    return Rz @ Ry @ Rx


def build_homography(K, R, T, w, h, f0):
    """
    Homography mapping SOURCE image pixels -> OUTPUT image pixels.

    1) M: source pixel (u, v) -> plane point (x, y, Z) in world coordinates.
          The image is centered on the optical axis and scaled so that, with the default
          camera (f = f0), the plane projects exactly onto the original pixels.
    2) Plane points are (x, y, Z_PLANE), so for X_cam = R*X + T:
          X_cam = r1*x + r2*y + (Z*r3 + T)   ->   P = [r1 | r2 | Z*r3 + T]
    3) K projects the camera-frame point to pixels.
    """
    s = Z_PLANE / f0
    M = np.array([[s, 0.0, -s * w / 2.0],
                  [0.0, s, -s * h / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    P = np.column_stack((R[:, 0], R[:, 1], Z_PLANE * R[:, 2] + T.reshape(3)))
    return K @ P @ M


def homography_is_valid(H, w, h):
    """
    Reject degenerate cases so the app never crashes or shows a mirrored image:
      - non-finite entries,
      - (near-)singular H,
      - any image corner with depth <= EPS_DEPTH (plane crosses/behind the camera plane).
    Since K's last row is [0 0 1], the third component of H @ [u v 1] is the true depth.
    """
    if not np.all(np.isfinite(H)):
        return False
    if np.linalg.cond(H) > MAX_COND:
        return False
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
    depth = (H @ corners)[2]
    return bool(np.all(np.isfinite(depth)) and np.all(depth > EPS_DEPTH))


# --------------------------------------------------------------------------- #
# Radial distortion (meshgrid + remap)
# --------------------------------------------------------------------------- #
def apply_radial_distortion(img, K, k1, k2, k3, grid_u, grid_v):
    """
    Simulate the OpenCV radial model:  x_d = x_u * (1 + k1 r^2 + k2 r^4 + k3 r^6).

    cv2.remap needs, for each DESTINATION (distorted) pixel, the SOURCE (undistorted) location.
    That requires inverting the polynomial, which we do with a fixed-point iteration
        x_u <- x_d / factor(x_u)
    (same idea as cv2.undistortPoints). The factor is clamped to stay positive and bounded so
    extreme coefficients can never produce NaN/inf or division by zero.
    """
    if k1 == 0.0 and k2 == 0.0 and k3 == 0.0:
        return img

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Normalized distorted coordinates for every output pixel
    xd = (grid_u - cx) / fx
    yd = (grid_v - cy) / fy
    xu, yu = xd.copy(), yd.copy()

    for _ in range(DIST_ITERS):
        r2 = xu * xu + yu * yu
        factor = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        factor = np.clip(np.nan_to_num(factor, nan=1.0, posinf=50.0, neginf=0.05), 0.05, 50.0)
        xu = xd / factor
        yu = yd / factor

    map_x = np.clip(np.nan_to_num(xu * fx + cx), -1e5, 1e5).astype(np.float32)
    map_y = np.clip(np.nan_to_num(yu * fy + cy), -1e5, 1e5).astype(np.float32)

    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


# --------------------------------------------------------------------------- #
# Full rendering pipeline
# --------------------------------------------------------------------------- #
def render(src, p, grid_u, grid_v):
    """
    Run the complete pipeline for the given parameter dict.
    Returns (output_image, valid_flag, (K, R, T, dist_coeffs)).
    """
    h, w = src.shape[:2]
    f0 = float(max(w, h))

    K = build_intrinsics(p["fx"], p["fy"], p["cx"], p["cy"])                 # step 1
    R = euler_to_rotation(p["pitch"], p["yaw"], p["roll"])                    # step 2
    T = np.array([p["tx"], p["ty"], p["tz"]], dtype=np.float64)
    H = build_homography(K, R, T, w, h, f0)                                   # step 3
    dist = np.array([p["k1"], p["k2"], 0.0, 0.0, p["k3"]], dtype=np.float64)  # OpenCV order

    if not homography_is_valid(H, w, h):
        # Safe fallback: black frame, the caller draws a warning on it
        return np.zeros_like(src), False, (K, R, T, dist)

    warped = cv2.warpPerspective(src, H, (w, h), flags=cv2.INTER_LINEAR,      # step 4
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    out = apply_radial_distortion(warped, K, p["k1"], p["k2"], p["k3"],       # step 5
                                  grid_u, grid_v)
    return out, True, (K, R, T, dist)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def put_text(img, text, org, color=(255, 255, 255), scale=0.5):
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_overlay(img, p, valid):
    """Draw the current parameter values on a copy of the output image."""
    vis = img.copy()
    lines = [
        f"T  = ({p['tx']:+.2f}, {p['ty']:+.2f}, {p['tz']:+.2f})",
        f"RPY = pitch {p['pitch']:+.1f}  yaw {p['yaw']:+.1f}  roll {p['roll']:+.1f} deg",
        f"f  = ({p['fx']:.0f}, {p['fy']:.0f})   c = ({p['cx']:.0f}, {p['cy']:.0f})",
        f"k  = ({p['k1']:+.2f}, {p['k2']:+.2f}, {p['k3']:+.2f})",
    ]
    for i, line in enumerate(lines):
        put_text(vis, line, (10, 20 + 20 * i), (0, 255, 255))
    if not valid:
        msg = "INVALID POSE: plane behind / crossing the camera"
        put_text(vis, msg, (10, vis.shape[0] // 2), (0, 0, 255), 0.7)
    return vis


def make_control_panel_image():
    """Small canvas shown in the control window (trackbars are attached below it)."""
    panel = np.full((100, 520, 3), 30, np.uint8)
    put_text(panel, "Camera parameters - move the sliders", (10, 25), (0, 255, 255), 0.55)
    put_text(panel, "Focus the image window:", (10, 50), (200, 200, 200), 0.45)
    put_text(panel, "[r] reset  [s] save  [h] overlay  [q/ESC] quit", (10, 70), (200, 200, 200), 0.45)
    return panel


def print_calibration(K, R, T, dist):
    np.set_printoptions(precision=5, suppress=True)
    print("\n=== Current virtual camera ===")
    print("K =\n", K)
    print("R =\n", R)
    print("T =", T)
    print("distCoeffs (k1, k2, p1, p2, k3) =", dist)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    src = load_image(path)
    h, w = src.shape[:2]

    # Pixel meshgrid reused by every distortion pass (float64 for precision)
    grid_u, grid_v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))

    specs = make_slider_specs(w, h)

    # Shared state: the trackbar callback only flags that a redraw is needed.
    # The heavy work runs once per frame in the main loop, so resetting 13 sliders at once
    # (which fires 13 callbacks) triggers a single re-render instead of 13.
    state = {"dirty": True, "overlay": True}

    def on_change(_value):
        state["dirty"] = True

    # Output window
    cv2.namedWindow(IMG_WIN, cv2.WINDOW_AUTOSIZE)

    # Control window with all the trackbars
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.imshow(CTRL_WIN, make_control_panel_image())
    for spec in specs:
        key, label, lo, hi, step, default = spec
        count = max(1, int(round((hi - lo) / step)))
        cv2.createTrackbar(label, CTRL_WIN, 0, count, on_change)
        cv2.setTrackbarPos(label, CTRL_WIN, real_to_pos(spec, default))
    cv2.resizeWindow(CTRL_WIN, 540, 620)
    cv2.moveWindow(CTRL_WIN, 20, 20)
    cv2.moveWindow(IMG_WIN, 600, 20)

    last_state = None  # (K, R, T, dist) of the last render, used by the 's' key
    output = src.copy()
    valid = True
    params = read_params(specs)

    print("Camera Calibration Lab started. Keys: [r] reset  [s] save  [h] overlay  [q/ESC] quit")

    while True:
        if state["dirty"]:
            state["dirty"] = False
            params = read_params(specs)
            try:
                output, valid, last_state = render(src, params, grid_u, grid_v)
            except cv2.error as exc:  # last line of defense: never crash on odd parameter combos
                print(f"[WARN] OpenCV error ignored: {exc}")
                output, valid = np.zeros_like(src), False
            shown = draw_overlay(output, params, valid) if state["overlay"] or not valid else output
            cv2.imshow(IMG_WIN, shown)

        key = cv2.waitKey(15) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord("r"):
            for spec in specs:
                cv2.setTrackbarPos(spec[1], CTRL_WIN, real_to_pos(spec, spec[5]))
            state["dirty"] = True
        elif key == ord("h"):
            state["overlay"] = not state["overlay"]
            state["dirty"] = True
        elif key == ord("s"):
            cv2.imwrite(OUTPUT_FILE, output)
            print(f"Saved '{OUTPUT_FILE}'")
            if last_state is not None:
                print_calibration(*last_state)

        # Exit if the user closes either window with the mouse
        try:
            if (cv2.getWindowProperty(IMG_WIN, cv2.WND_PROP_VISIBLE) < 1 or
                    cv2.getWindowProperty(CTRL_WIN, cv2.WND_PROP_VISIBLE) < 1):
                break
        except cv2.error:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
