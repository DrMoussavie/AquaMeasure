# AquaMeasure

**Stereo vision fish measurement tool**

AquaMeasure is a desktop application for calibrating a stereo camera rig and measuring distances (e.g. fish length) from synchronized video pairs using epipolar geometry and DLT triangulation.

---

## Features

### Tab 1 — Calibration
- Load left and right video files recorded simultaneously
- Automatic extraction of ~80 synchronized frames
- Auto-detection of checkerboard orientation
- Intrinsic calibration for each camera (reprojection error in pixels)
- Stereo calibration (R, T, F matrices)
- Post-calibration corner verification grid with:
  - 4× zoom panel on hover
  - Draggable corner handles for manual adjustment
  - One-click recalibration with adjusted points

### Tab 2 — Measure
- Synchronized video playback (play/pause, seek slider, Space key)
- Click to place 4 measurement points (A & B on left, A & B on right)
- Draggable handles to refine point positions at any time
- 4× zoom overlay follows cursor/handle for sub-pixel precision
- DLT triangulation → distance in mm
- 3D point cloud viewer (Open3D)

---

## Installation

```bash
pip install opencv-python numpy scipy PyQt6 open3d
```

## Usage

```bash
python aquameasure.py
```

### Calibration workflow
1. Click **Select Left Video** and **Select Right Video**
2. Set checkerboard rows/columns and square size (mm)
3. Click **Run Calibration** — wait for the progress bar
4. Inspect the verification grid; drag corners if needed and recalibrate

### Measurement workflow
1. Open **Measure** tab → **Load Videos** (auto-loads from `camera_parameters/videos.txt`)
2. Navigate to the desired frame (slider or play/pause)
3. Pause the video
4. Click point **A** then **B** on the LEFT image (red handles)
5. Click point **A** then **B** on the RIGHT image (blue handles)
6. Distance is displayed immediately; drag any handle to refine

---

## Camera parameters

Saved automatically to `camera_parameters/` after calibration:

| File | Content |
|------|---------|
| `mtx1.npy` / `mtx2.npy` | Camera intrinsic matrices |
| `dist1.npy` / `dist2.npy` | Distortion coefficients |
| `R.npy` / `T.npy` | Stereo rotation & translation |
| `F.npy` | Fundamental matrix |
| `videos.txt` | Paths to calibration videos |
| `verify/` | Annotated corner detection frames |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Calibration, video, disparity |
| `numpy` | Array maths |
| `scipy` | SVD for DLT |
| `PyQt6` | GUI |
| `open3d` *(optional)* | 3D point cloud viewer |
