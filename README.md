# Fish Picking Demo

A desktop demo for picking fish with a TM collaborative robot. A USB camera feed
is run through a YOLO fish detector (ONNX Runtime), and the detected fish
position is turned into X/Y offsets that can be sent to the robot over the TMSVR
protocol.

The GUI is a pygame window with two tabs:

- **Controls** — camera feed, arbitrary robot variable read/write, detection
  statistics, and tuning sliders for `g_diffX` / `g_diffY` / `g_diffZ` /
  `g_diffRz`.
- **Fish** — camera feed, the fish position relative to the frame centre, a
  preview of the offsets that will be sent, and an **Execute** button that runs
  the move handshake.

## Requirements

- Python 3.13 (tested on Windows)
- A USB camera
- Network access to the robot's TMSVR port
- `git` on `PATH` (required because `pg-widgets` is installed from GitHub)

Install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` pulls `pg-widgets` from the newest `main` branch, not the
PyPI release. Pin a commit in that file if you need an exact, reproducible
install.

## Running

```powershell
.venv\Scripts\python main.py
```

Without a camera or robot the window still opens; the affected panels report
that vision or the robot is unavailable.

Useful standalone entry points:

| Command | What it does |
| --- | --- |
| `python main.py` | Full GUI demo |
| `python Vision.py` | Camera + detector with an OpenCV preview (press `q` to quit) |
| `python Communication.py` | Writes `g_testInt` and reads `g_testFloat` to check the robot link |

Close the GUI with the window close button or `Esc`.

## Configuration

### Robot address

`Communication.py` reads these environment variables, falling back to the lab
robot:

| Variable | Default |
| --- | --- |
| `TM_ROBOT_HOST` | `158.38.140.99` |
| `TM_ROBOT_PORT` | `5891` |
| `TM_ROBOT_TIMEOUT` | `3.0` seconds |

```powershell
$env:TM_ROBOT_HOST = "192.168.1.50"
.venv\Scripts\python main.py
```

### Camera and detector

The defaults live at the top of `Vision.py`:

- `DEFAULT_CAMERA_INDICES = (1, 2, 3, 4)` — the first index that delivers a
  frame is used, so set this to your camera's index if probing is slow.
- `DEFAULT_WIDTH` / `DEFAULT_HEIGHT` / `DEFAULT_FPS` — `1280x720` at `15` fps.
- `DEFAULT_CONFIDENCE` / `DEFAULT_IOU_THRESHOLD` — `0.25` / `0.45`.
- `DEFAULT_RECONNECT_INTERVAL` — `1.0` s between camera reconnect attempts.

### Model

`model/best.onnx` is an Ultralytics export with `nms=False`, exported at
`640x640` for a single class (`fish`). Decoding and non-maximum suppression
happen in `Vision.py`, and the input size and class names are read from the
model metadata where available. To use another model, replace the file or pass
`model_path=...` to `Vision` / `OnnxFishDetector`.

`model/best.pt` is the training checkpoint; it is not used at runtime.

### Motion mapping

The **Fish** tab sends offsets once per click of **Execute**:

1. `g_diffX` and `g_diffY` are written.
2. Wait `EXECUTE_WAIT_SECONDS` (1 s).
3. `g_toComputer = 0`.
4. Wait 1 s, then `g_toRobot = 1`.

The pixel-to-robot mapping is defined in `executeFishMove()` in `main.py`:

```python
diffX = float(targetDiff["x"]) * -0.5
diffY = float(targetDiff["y"]) *  0.5
```

Offsets smaller than `FISH_DEADBAND_PIXELS` (10 px) from the frame centre are
sent as `0`. **Verify the signs and scale on the real robot** before relying on
them; the axis directions depend on how the camera is mounted.

The sliders on the **Controls** tab are only transmitted while that tab is
active, so they cannot overwrite the offsets written by **Execute**.

### Training images

`main.py` can save unannotated camera frames to `training_images/` at 5 Hz while
the demo runs. It is off by default; set `SAVE_TRAINING_IMAGES = True` in
`main()` to enable it. Saving happens on a background thread and frames are
dropped rather than queued if the disk cannot keep up.

## Project layout

| File | Purpose |
| --- | --- |
| `main.py` | GUI entry point; wires vision, robot and widgets together |
| `Vision.py` | USB camera capture, ONNX inference, annotation, snapshots |
| `Communication.py` | TMSVR client, both blocking and threaded/non-blocking |
| `ImageView.py` | `pg-widgets` widget that displays a scaled image |
| `model/` | `best.onnx` (runtime) and `best.pt` (training checkpoint) |
| `training_images/` | Captured frames for retraining (ignored by git) |

## Notes and limitations

- Vision and communication failures do not stop the program. Check the status
  fields in the GUI (`Vision not available`, `Waiting for camera`) and the
  `last_error` properties when something looks stuck.
- The robot link reconnects automatically on the next command after a failure.
