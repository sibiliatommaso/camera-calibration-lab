# Interactive Camera Calibration Lab

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A small, dependency-light playground to **see** what camera parameters do to an image, in real time.

The input image is treated as a flat object placed in 3D space. A *virtual pinhole camera* looks at it, and you control that camera with sliders: position, orientation, focal length, optical center and lens distortion. Every slider movement re-renders the result immediately.

It is meant as a teaching and exploration tool for anyone learning camera geometry, homographies and lens distortion, or as a visual sanity check when reading OpenCV calibration results (`K`, `R`, `T`, `distCoeffs`).

![Example outputs](docs/examples.png)

*Example outputs produced by the same pipeline the app uses (default parameters, a pose change, a translation + roll, barrel and pincushion distortion, and an intrinsics change).*

---

## Table of contents

- [Features](#features)
- [Quick start](#quick-start)
- [Controls](#controls)
- [The math behind it](#the-math-behind-it)
- [Code structure](#code-structure)
- [Robustness: what happens at the slider extremes](#robustness-what-happens-at-the-slider-extremes)
- [Limitations and design notes](#limitations-and-design-notes)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Experiments to try](#experiments-to-try)
- [License](#license)

---

## Features

- **13 live parameters** on a dedicated control window: translation (`Tx`, `Ty`, `Tz`), Euler rotation (pitch, yaw, roll), intrinsics (`fx`, `fy`, `cx`, `cy`) and radial distortion (`k1`, `k2`, `k3`).
- **Identity by default**: with the sliders at their initial positions the output equals the input image, so every change you see is caused by exactly the parameter you moved.
- **Real geometry**: a plane-to-image homography built from `K`, `R`, `T`, applied with `cv2.warpPerspective`, followed by radial distortion applied with `cv2.remap`.
- **Does not crash at the extremes**: ranges are bounded, degenerate poses are detected and reported on screen, and the distortion math is clamped against NaN/inf.
- **Export**: press `s` to save the current image and print `K`, `R`, `T` and `distCoeffs` in OpenCV order.
- **Single file, two dependencies** (`numpy`, `opencv-python`). Heavily commented.

---

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/camera-calibration-lab.git
cd camera-calibration-lab

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Run
python camera_calibration_lab.py
```

By default the script loads `img/scacchiera.jpg` (a sample chessboard included in the repo). To use your own picture:

```bash
python camera_calibration_lab.py path/to/your_image.jpg
```

If the file cannot be read, the program prints a warning and falls back to a synthetic chessboard, so it always starts.

> **Note:** install `opencv-python`, **not** `opencv-python-headless`. The headless build has no GUI support and cannot open windows or trackbars.

Two windows will open: **Camera Lab - Output** (the image) and **Camera Lab - Controls** (the sliders).

---

## Controls

### Keyboard (focus the image window)

| Key | Action |
|-----|--------|
| `r` | Reset all sliders to their defaults |
| `s` | Save the current output to `calibration_lab_output.png` and print `K`, `R`, `T`, `distCoeffs` in the terminal |
| `h` | Show / hide the parameter overlay |
| `q` or `ESC` | Quit (closing either window also quits) |

### Sliders

OpenCV trackbars only support integers, so each slider is mapped linearly onto a real range (`value = min + position * step`). The current real values are shown in the overlay on the image. In the table, `f₀ = max(width, height)` of the loaded image.

| Group | Slider | Range | Step | Default | Meaning |
|-------|--------|-------|------|---------|---------|
| Translation | `Tx` | -2 … 2 | 0.01 | 0 | Shift along the camera X axis (plane depth units) |
| | `Ty` | -2 … 2 | 0.01 | 0 | Shift along the camera Y axis |
| | `Tz` | -0.9 … 4 | 0.01 | 0 | Shift along the optical axis. Plane depth becomes `1 + Tz` |
| Rotation | `Pitch X` | -85° … 85° | 0.5° | 0 | Rotation around X |
| | `Yaw Y` | -85° … 85° | 0.5° | 0 | Rotation around Y |
| | `Roll Z` | -180° … 180° | 0.5° | 0 | Rotation around Z (optical axis) |
| Intrinsics | `fx` | 0.25 f₀ … 4 f₀ | 1 px | f₀ | Focal length (x), in pixels |
| | `fy` | 0.25 f₀ … 4 f₀ | 1 px | f₀ | Focal length (y), in pixels |
| | `cx` | 0 … width | 1 px | width / 2 | Optical center x |
| | `cy` | 0 … height | 1 px | height / 2 | Optical center y |
| Distortion | `k1` | -1 … 1 | 0.01 | 0 | Radial coefficient r² |
| | `k2` | -1 … 1 | 0.01 | 0 | Radial coefficient r⁴ |
| | `k3` | -1 … 1 | 0.01 | 0 | Radial coefficient r⁶ |

Sign conventions in the OpenCV distortion model: `k1 < 0` gives **barrel** distortion, `k1 > 0` gives **pincushion** distortion.

---

## The math behind it

### 1. Scene model

The image is a rectangle lying on the plane `Z = Z₀` (here `Z₀ = 1`) in a world frame whose origin is the camera center. A world point `X` is expressed in the camera frame as

$$
\mathbf{X}_c = R\,\mathbf{X} + T
$$

so `R` and `T` describe the pose of the plane relative to the camera. With `T = 0` the effect is a pure camera rotation about its optical center.

### 2. Intrinsic matrix

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

### 3. Rotation from Euler angles

Each elementary rotation is built from a Rodrigues vector (`axis * angle`) with `cv2.Rodrigues`, then composed as

$$
R = R_z(\text{roll})\,R_y(\text{yaw})\,R_x(\text{pitch})
$$

### 4. Plane-to-image homography

A point on the plane is `X = (x, y, Z₀)ᵀ`, therefore

$$
\mathbf{X}_c = x\,\mathbf{r}_1 + y\,\mathbf{r}_2 + (Z_0\,\mathbf{r}_3 + T)
$$

where `r₁, r₂, r₃` are the columns of `R`. Projecting with `K` gives the homography from plane coordinates to output pixels:

$$
H_{\text{plane}\to\text{img}} = K \,[\,\mathbf{r}_1 \;|\; \mathbf{r}_2 \;|\; Z_0\mathbf{r}_3 + T\,]
$$

The source pixels `(u, v)` are mapped to plane coordinates by

$$
M = \begin{bmatrix} Z_0/f_0 & 0 & -Z_0 w/(2f_0) \\ 0 & Z_0/f_0 & -Z_0 h/(2f_0) \\ 0 & 0 & 1 \end{bmatrix}
$$

(the image is centered on the optical axis and scaled by the reference focal length `f₀`). The matrix used by `cv2.warpPerspective` is the product

$$
H = K\,[\,\mathbf{r}_1 \;|\; \mathbf{r}_2 \;|\; Z_0\mathbf{r}_3 + T\,]\;M
$$

**Sanity checks**

- With the default values (`R = I`, `T = 0`, `f = f₀`, `c = (w/2, h/2)`), `H` is the identity.
- With `T = 0`, `H = K R K₀⁻¹`, where `K₀` is the default intrinsic matrix, i.e. the classic *pure rotation* homography.

### 5. Radial distortion

The OpenCV radial model maps ideal normalized coordinates to distorted ones:

$$
\mathbf{x}_d = \mathbf{x}_u \,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6), \qquad r^2 = x_u^2 + y_u^2
$$

`cv2.remap` is a *backward* mapping: for every **destination** (distorted) pixel it needs the **source** (undistorted) position. That requires inverting the polynomial, which has no closed form. The code uses a fixed-point iteration (the same idea as `cv2.undistortPoints`):

$$
\mathbf{x}_u \leftarrow \frac{\mathbf{x}_d}{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}\bigg|_{r = \lVert \mathbf{x}_u \rVert}
$$

started from `x_u = x_d` and run for 8 iterations. Normalization uses the same `K` (`x = (u - cx)/fx`, `y = (v - cy)/fy`), so the coefficients you see here have the same meaning as the ones returned by `cv2.calibrateCamera`.

---

## Code structure

Everything lives in `camera_calibration_lab.py`.

| Function | Role |
|----------|------|
| `load_image` | Loads and (if needed) downscales the image; falls back to a synthetic chessboard |
| `make_slider_specs` | Single source of truth for every slider: name, range, step, default |
| `read_params` | Reads trackbar positions and converts them to real values |
| `build_intrinsics` | Builds `K` |
| `euler_to_rotation` | Euler angles → `R` using `cv2.Rodrigues` |
| `build_homography` | Builds `H = K [r1 r2 Z₀ r3 + T] M` |
| `homography_is_valid` | Rejects singular / behind-the-camera configurations |
| `apply_radial_distortion` | Meshgrid + iterative inversion + `cv2.remap` |
| `render` | Runs the whole pipeline for one parameter set |
| `main` | Window creation, trackbars, event loop |

Rendering is **event-driven**: the trackbar callback only marks the state as "dirty", and the main loop re-renders once per frame. This means resetting all 13 sliders at once triggers a single render instead of 13.

---

## Robustness: what happens at the slider extremes

| Risk | Protection |
|------|-----------|
| Plane edge-on or behind the camera (mirrored / exploding image) | Rotation limited to ±85° for pitch/yaw; `Tz ≥ -0.9`; all four image corners must have depth > `1e-3`, otherwise a black frame with an **INVALID POSE** message is shown |
| Singular or ill-conditioned `H` | Checked with `np.linalg.cond(H)` and `np.isfinite` before warping |
| Division by zero or negative factor in the distortion model | Radial factor clamped to `[0.05, 50]`; `NaN` / `inf` replaced; remap coordinates clipped |
| Unexpected OpenCV error | Caught in the main loop; the app keeps running |

The core was stress-tested headlessly with thousands of random parameter combinations including every slider at its minimum and maximum.

---

## Limitations and design notes

- **Pinhole + radial distortion only.** There is no skew and no tangential distortion (`p1 = p2 = 0` in the exported `distCoeffs`).
- **The whole plane must be in front of the camera.** If any corner of the image crosses the camera plane, the pose is flagged invalid instead of showing a partial render.
- **Large `k` values.** For strong coefficients the radial polynomial is not monotonic, so the fixed-point inversion may not converge everywhere. The output stays finite and safe, but it is no longer geometrically exact in those regions.
- **Image is downscaled** to at most 900 px on the longest side (`MAX_DIM`) to keep the UI responsive. `f₀`, `cx` and `cy` ranges refer to the downscaled size.
- **Rotation happens about the camera center.** With a plane at depth 1 and `f = f₀`, even a 30° rotation moves the plane far off-screen. That is physically correct. Add some `Tz` or reduce `f` to keep it in view.

---

## Customization

The main constants are at the top of the file:

| Constant | Default | Effect |
|----------|---------|--------|
| `MAX_DIM` | `900` | Maximum image size before downscaling |
| `Z_PLANE` | `1.0` | Depth of the image plane |
| `EPS_DEPTH` | `1e-3` | Minimum corner depth for a pose to be valid |
| `MAX_COND` | `1e10` | Maximum condition number of `H` |
| `DIST_ITERS` | `8` | Iterations for the distortion inversion (more = more accurate for large `k`) |

To change a slider range, edit its row in `make_slider_specs()`. Nothing else needs to change.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `cv2.error: The function is not implemented ... Rebuild the library with Windows, GTK+ 2.x or Cocoa support` | You installed `opencv-python-headless`. Run `pip uninstall opencv-python-headless` and then `pip install opencv-python` |
| The control window shows no sliders or they look squashed | Resize the **Controls** window by dragging its border |
| The window does not open on a remote server / WSL / Docker | The app needs a display. Use a local desktop session, X forwarding, or WSLg |
| `[WARN] Cannot read 'img/scacchiera.jpg'` | Run the script from the repository root, or pass the image path explicitly |
| Black image with INVALID POSE | The plane is behind the camera or edge-on. Reduce rotation or increase `Tz` |
| Qt / Wayland errors on Linux | Try `QT_QPA_PLATFORM=xcb python camera_calibration_lab.py` |

---

## Experiments to try

1. **Zooming vs. moving closer.** For a frontal plane, `u = f·x/(1 + Tz) + cx`, so what matters is the ratio `f / (1 + Tz)`: raising `fx = fy` and raising `Tz` by the same factor gives an identical image. Now tilt the plane with `Yaw` and repeat the comparison: the foreshortening is no longer the same, which is exactly why focal length and distance are not interchangeable.
2. **Optical center.** Move `cx`, `cy` and notice the image shifts without any perspective change, unlike a translation.
3. **Barrel vs. pincushion.** Sweep `k1` from negative to positive and watch how straight chessboard lines bend.
4. **Rotation vs. translation.** Compare `Yaw` with `Tx`. A pure rotation has no parallax and is exactly a homography `K R K⁻¹`.
5. **Match a real calibration.** Take the `K` and `distCoeffs` returned by `cv2.calibrateCamera` for your own camera and reproduce them with the sliders to understand what they mean.

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

If you use this tool in teaching material or a publication, a link back to this repository is mandatory.
