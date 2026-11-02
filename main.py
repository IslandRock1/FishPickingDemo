"""Fish picking demo GUI.

The window is split into four equally sized quadrants:

* top left: camera feed from Vision (aspect ratio preserved)
* top right: write and read arbitrary robot variables, four rows each
* bottom left: detection stats (status, position, size, confidence)
* bottom right: sliders for g_diffX, g_diffY, g_diffZ and g_diffRz

Unannotated camera frames are also saved to ``training_images/`` at 5 Hz so
they can be used to train a new model later.
"""

import pygame as pg
import pg_widgets as pw
import numpy as np
import queue
import threading
import time
from pathlib import Path

from ImageView import ImageView
from Vision import Vision
from Communication import ThreadedRobotClient

IMAGE_SAVE_DIRECTORY = Path(__file__).resolve().parent / "training_images"
IMAGE_SAVE_INTERVAL = 0.2  # 5 images per second
IMAGE_JPEG_QUALITY = 95


class ImageSaver:
    """Save camera frames in the background so disk I/O cannot stall the GUI."""

    def __init__(
        self,
        directory: Path = IMAGE_SAVE_DIRECTORY,
        *,
        interval: float = IMAGE_SAVE_INTERVAL,
        jpeg_quality: int = IMAGE_JPEG_QUALITY,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.jpeg_quality = jpeg_quality

        self._last_save = 0.0
        self._last_sequence: int | None = None
        self._queue: queue.Queue[tuple[Path, np.ndarray] | None] = queue.Queue(
            maxsize=2
        )
        self._thread = threading.Thread(
            target=self._run,
            name="ImageSaver",
            daemon=True,
        )
        self._thread.start()

    def offer(self, frame: np.ndarray, sequence: int) -> None:
        """Queue one frame if the save interval has passed."""
        now = time.monotonic()
        if sequence == self._last_sequence or now - self._last_save < self.interval:
            return

        self._last_save = now
        self._last_sequence = sequence
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.directory / f"frame_{timestamp}_{sequence:06d}.jpg"

        try:
            self._queue.put_nowait((path, frame.copy()))
        except queue.Full:
            # Saving is falling behind; skip this frame rather than build a
            # backlog of stale images.
            pass

    def close(self) -> None:
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            import cv2
        except ImportError:
            print("Image saving disabled: OpenCV is not installed")
            return

        while True:
            item = self._queue.get()
            if item is None:
                return

            path, frame = item
            try:
                saved = cv2.imwrite(
                    str(path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not saved:
                    print(f"Could not save image: {path}")
            except Exception as error:
                print(f"Error saving image {path}: {error}")


def _numpy_bgr_to_surface(frame: np.ndarray) -> pg.Surface:
    """Convert BGR numpy frame (from OpenCV) to pygame Surface."""
    rgb = frame[:, :, ::-1]
    surface = pg.surfarray.make_surface(rgb.swapaxes(0, 1))
    return surface

def getVision() -> Vision | None:
    vision = Vision(annotate=True)
    try:
        vision.start()
    except Exception as e:
        print(f"Warning: Could not start Vision: {e}")
        vision = None

    return vision

def getRobot() -> ThreadedRobotClient | None:
    robot = None
    try:
        robot = ThreadedRobotClient()
        robot.start()
        print("Robot communication started")
    except Exception as e:
        print(f"Warning: Could not start robot communication: {e}")
        robot = None

    return robot

def _parse_value(text: str):
    """Convert text from a value box to int/float/bool, falling back to str."""
    text = text.strip()
    if not text:
        return ""

    try:
        if any(character in text for character in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        pass

    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    return text


def _write_variable(robot, name: str, value_text: str) -> bool:
    """Write one value to the robot, converting the text to a useful type."""
    name = name.strip()
    if not name:
        print("Write skipped: variable name is empty")
        return False

    if robot is None:
        print("Write skipped: robot not connected")
        return False

    try:
        value = _parse_value(value_text)
        robot.write(name, value)
        print(f"Sent: {name} = {value!r}")
        return True
    except Exception as error:
        print(f"Error sending {name!r}: {error}")
        return False


def main():
    control = pw.ControlManager()

    vision = getVision()
    robot = getRobot()

    # ----- Top left: camera feed -----
    topLeft = pw.UIGroup((0.0, 0.0), (0.5, 0.5))
    imageView = ImageView((0.0, 0.0), (1.0, 1.0))
    topLeft["image"] = imageView
    control["topLeft"] = topLeft

    # ----- Bottom left: detection statistics -----
    botLeft = pw.UIGroup((0.0, 0.5), (0.5, 0.5))

    statsLabels = pw.TextBoxes((0.0, 0.0), (0.5, 1.0), labels=[
        "Status:",
        "Pos:",
        "Size:",
        "Confidence:",
    ])
    statsLabels.setAlignments(
        horizontal=pw.TextBox.AlignmentHorizontal.LEFT,
        vertical=pw.TextBox.AlignmentVertical.MIDDLE,
    )

    statsValues = pw.TextBoxes((0.5, 0.0), (0.5, 1.0), labels=[
        "Initializing..",
        "-- | --",
        "-- x --",
        "--",
    ])
    statsValues.setAlignments(
        horizontal=pw.TextBox.AlignmentHorizontal.LEFT,
        vertical=pw.TextBox.AlignmentVertical.MIDDLE,
    )

    for i in range(4):
        statsLabels[f"text{i}"].disableSelection()
        statsValues[f"text{i}"].disableSelection()

    botLeft["statsLabels"] = statsLabels
    botLeft["statsValues"] = statsValues
    control["botLeft"] = botLeft

    # ----- Top right: write and read arbitrary variables -----
    topRight = pw.UIGroup((0.5, 0.0), (0.5, 0.5))

    # Write side (left half)
    writeVar = pw.UIGroup((0.0, 0.0), (0.5, 1.0))

    writeLabel = pw.TextBox((0.0, 0.0), (1.0, 0.2))
    writeLabel.setText("Write")
    writeLabel.disableSelection()
    writeVar["writeLabel"] = writeLabel

    writeNameLabel = pw.TextBox((0.0, 0.2), (0.40, 0.2))
    writeNameLabel.setText("Name")
    writeNameLabel.disableSelection()
    writeVar["nameLabel"] = writeNameLabel

    writeValueLabel = pw.TextBox((0.40, 0.2), (0.40, 0.2))
    writeValueLabel.setText("Value")
    writeValueLabel.disableSelection()
    writeVar["valueLabel"] = writeValueLabel

    writeButtonLabel = pw.TextBox((0.8, 0.2), (0.2, 0.2))
    writeButtonLabel.setText("Send")
    writeButtonLabel.disableSelection()
    writeVar["writeButtonLabel"] = writeButtonLabel

    writeNameBoxes = pw.TextBoxes((0.0, 0.4), (0.4, 0.6), labels=[""] * 4)
    writeValueBoxes = pw.TextBoxes((0.4, 0.4), (0.4, 0.6), labels=[""] * 4)
    writeNameBoxes.setText("g_testVar", 0)
    writeValueBoxes.setText("0", 0)

    for i in range(4):
        writeNameBoxes[f"text{i}"].setAlignment(
            horizontal=pw.TextBox.AlignmentHorizontal.LEFT,
            vertical=pw.TextBox.AlignmentVertical.MIDDLE,
        )
        writeValueBoxes[f"text{i}"].setAlignment(
            horizontal=pw.TextBox.AlignmentHorizontal.LEFT,
            vertical=pw.TextBox.AlignmentVertical.MIDDLE,
        )

    writeVar["writeNameBoxes"] = writeNameBoxes
    writeVar["writeValueBoxes"] = writeValueBoxes

    for i in range(4):
        sendButton = pw.Button(
            (0.8, 0.4 + i * 0.6 / 4.0),
            (0.2, 0.6 / 4.0),
        )
        sendButton.setText("Send")
        sendButton.setFunction(
            lambda _=None, row=i: _write_variable(
                robot,
                writeNameBoxes[f"text{row}"].getText(),
                writeValueBoxes[f"text{row}"].getText(),
            )
        )
        writeVar[f"writeSendButtons{i}"] = sendButton

    # Read side (right half)
    readVar = pw.UIGroup((0.5, 0.0), (0.5, 1.0))

    readLabel = pw.TextBox((0.0, 0.0), (1.0, 0.2))
    readLabel.setText("Read")
    readLabel.disableSelection()
    readVar["readLabel"] = readLabel

    readNameLabel = pw.TextBox((0.0, 0.2), (0.45, 0.2))
    readNameLabel.setText("Name")
    readNameLabel.disableSelection()
    readVar["readNameLabel"] = readNameLabel

    readValueLabel = pw.TextBox((0.45, 0.2), (0.55, 0.2))
    readValueLabel.setText("Value")
    readValueLabel.disableSelection()
    readVar["readValueLabel"] = readValueLabel

    readNameBoxes = pw.TextBoxes((0.0, 0.4), (0.45, 0.6), labels=[""] * 4)
    readValueBoxes = pw.TextBoxes((0.45, 0.4), (0.55, 0.6), labels=["--"] * 4)
    readNameBoxes.setText("g_testFloat", 0)

    for i in range(4):
        readNameBoxes[f"text{i}"].setAlignment(
            horizontal=pw.TextBox.AlignmentHorizontal.LEFT,
            vertical=pw.TextBox.AlignmentVertical.MIDDLE,
        )
        readValueBoxes[f"text{i}"].setAlignment(
            horizontal=pw.TextBox.AlignmentHorizontal.LEFT,
            vertical=pw.TextBox.AlignmentVertical.MIDDLE,
        )
        readValueBoxes[f"text{i}"].disableSelection()

    readVar["readNameBoxes"] = readNameBoxes
    readVar["readValueBoxes"] = readValueBoxes

    topRight["writeVar"] = writeVar
    topRight["readVar"] = readVar
    control["topRight"] = topRight

    # ----- Bottom right: diff sliders -----
    botRight = pw.UIGroup((0.5, 0.5), (0.5, 0.5))

    diffLabel = pw.TextBox((0.02, 0.01), (0.96, 0.08))
    diffLabel.setText("Diff Control")
    diffLabel.setAlignment(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    diffLabel.disableSelection()
    botRight["diffLabel"] = diffLabel

    diffLabels = ["g_diffX", "g_diffY", "g_diffZ", "g_diffRz"]
    diffSliders = pw.TuningSliders(
        (0.02, 0.10),
        (0.96, 0.86),
        labels=diffLabels,
        lower_bounds=[-250.0] * 4,
        upper_bounds=[250.0] * 4,
        current_values=[0.0] * 4,
    )
    for label in diffLabels:
        diffSliders[f"value{label}"].disableSelection()
        diffSliders[f"text{label}"].disableSelection()

    botRight["diffSliders"] = diffSliders
    control["botRight"] = botRight

    last_diff_values: dict[str, float] = {}
    last_diff_error = None
    last_read_poll = 0.0
    read_poll_interval = 1.0
    requested_read_names: set[str] = set()

    SAVE_TRAINING_IMAGES = False
    if (SAVE_TRAINING_IMAGES):
        try:
            imageSaver = ImageSaver()
            print(
                f"Saving unannotated images to {imageSaver.directory} "
                f"at {1.0 / IMAGE_SAVE_INTERVAL:.0f} Hz"
            )
        except Exception as error:
            print(f"Warning: could not start image saving: {error}")
            imageSaver = None
        else: imageSaver = None

    try:
        while control.isRunning():
            now = time.monotonic()

            # Camera feed and detection statistics.
            if vision is not None:
                snapshot = vision.latest()
                if snapshot is not None:
                    if imageSaver is not None:
                        imageSaver.offer(snapshot.frame, snapshot.sequence)

                    frame = (
                        snapshot.annotated_frame
                        if snapshot.annotated_frame is not None
                        else snapshot.frame
                    )

                    if frame is not None:
                        try:
                            imageView.setImage(_numpy_bgr_to_surface(frame))
                        except Exception:
                            pass

                    best = snapshot.best
                    if best is None:
                        statsValues.setTexts([
                            "No fish detected",
                            "-- | --",
                            "-- x --",
                            "--",
                        ])
                    else:
                        frameHeight, frameWidth = frame.shape[:2]
                        centerX = frameWidth / 2.0
                        centerY = frameHeight / 2.0
                        fishX, fishY = best.center
                        relativeX = fishX - centerX
                        relativeY = fishY - centerY

                        statsValues.setTexts([
                            f"Fish detected ({len(snapshot.detections)} total)",
                            f"{relativeX:.1f} | {relativeY:.1f}",
                            f"{best.width:.1f} x {best.height:.1f} px",
                            f"{best.confidence:.3f}",
                        ])
                else:
                    statsValues.setTexts([
                        "Waiting for camera",
                        "-- | --",
                        "-- x --",
                        "--",
                    ])
            else:
                statsValues.setTexts([
                    "Vision not available",
                    "-- | --",
                    "-- x --",
                    "--",
                ])

            # Refresh the read rows once per second.
            if now - last_read_poll >= read_poll_interval:
                readNames = [
                    readNameBoxes[f"text{i}"].getText().strip()
                    for i in range(4)
                ]
                activeNames = {name for name in readNames if name}

                if robot is not None:
                    for name in requested_read_names - activeNames:
                        robot.forget(name)
                    for name in activeNames - requested_read_names:
                        try:
                            robot.request(name)
                        except Exception as error:
                            print(f"Error requesting {name!r}: {error}")
                requested_read_names = activeNames

                for i, name in enumerate(readNames):
                    if robot is None or not name:
                        value = "--"
                    elif robot.is_ready(name):
                        value = robot.get_value(name)
                        if value is None:
                            value = "--"
                    else:
                        value = "--"
                    readValueBoxes[f"text{i}"].setText(str(value))

                last_read_poll = now

            # Send only the diff sliders that changed since the last update.
            # ThreadedRobotClient.write() is non-blocking and coalesces writes
            # per name, so dragging cannot build up a backlog.
            if robot is not None:
                # A queued write may be dropped while the connection is down.
                # After recovery, resend every setpoint once.
                diff_error = robot.last_error
                if last_diff_error is not None and diff_error is None:
                    last_diff_values.clear()
                last_diff_error = diff_error

                currentDiffValues = dict(
                    zip(diffLabels, diffSliders.getValue())
                )
                for label, value in currentDiffValues.items():
                    value = float(value)
                    if last_diff_values.get(label) == value:
                        continue
                    try:
                        robot.write(label, value)
                        last_diff_values[label] = value
                    except Exception as error:
                        print(f"Error sending {label}: {error}")

            control.update()
    finally:
        if imageSaver is not None:
            imageSaver.close()

        if vision is not None:
            try:
                vision.stop()
            except Exception:
                pass

        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass

        control.close()

if __name__ == "__main__":
    main()
