"""Fish picking demo GUI.

The window has two tabs.

* **Controls**: the original four-quadrant dashboard

  - top left: camera feed from Vision (aspect ratio preserved)
  - top right: write and read arbitrary robot variables, four rows each
  - bottom left: detection stats (status, position, size, confidence)
  - bottom right: sliders for g_diffX, g_diffY, g_diffZ and g_diffRz

* **Fish**: camera feed, the fish position relative to the frame centre, a
  preview of the g_diffX/g_diffY values that will be sent, and an Execute
  button that runs the move handshake.

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

# Robot variables used by the Execute handshake.
DIFF_X_VARIABLE = "g_diffX"
DIFF_Y_VARIABLE = "g_diffY"
TO_COMPUTER_VARIABLE = "g_toComputer"
TO_ROBOT_VARIABLE = "g_toRobot"
CALIBRATION_DONE_VARIABLE = "g_calibrationDone"

# Fish positions closer than this (in pixels from the frame centre) are treated
# as already aligned and are sent as 0.
FISH_DEADBAND_PIXELS = 3.0
EXECUTE_WAIT_SECONDS = 1.0


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

    # Content of the first tab. It stops at 95% height so the tab buttons in
    # the top-right corner do not cover any of the existing widgets.
    mainTab = pw.UIGroup((0.0, 0.0), (1.0, 0.95))

    # ----- Top left: camera feed -----
    topLeft = pw.UIGroup((0.0, 0.0), (0.5, 0.5))
    imageView = ImageView((0.0, 0.0), (1.0, 1.0))
    topLeft["image"] = imageView
    mainTab["topLeft"] = topLeft

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
    mainTab["botLeft"] = botLeft

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
    mainTab["topRight"] = topRight

    # ----- Bottom right: diff sliders -----
    botRight = pw.UIGroup((0.5, 0.5), (0.5, 0.5))

    diffLabel = pw.TextBox((0.02, 0.01), (0.96, 0.08))
    diffLabel.setText("Diff Control")
    diffLabel.setAlignment(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    diffLabel.disableSelection()
    botRight["diffLabel"] = diffLabel

    diffLabels = [
        DIFF_X_VARIABLE,
        DIFF_Y_VARIABLE,
        "g_diffZ",
        "g_diffRz",
    ]
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
    mainTab["botRight"] = botRight

    # ----- Second tab: fish targeting -----
    fishTab = pw.UIGroup((0.0, 0.0), (1.0, 0.95))

    fishCamera = pw.UIGroup((0.0, 0.0), (0.62, 1.0))
    fishImageView = ImageView((0.0, 0.0), (1.0, 1.0))
    fishCamera["image"] = fishImageView
    fishTab["camera"] = fishCamera

    fishPanel = pw.UIGroup((0.62, 0.0), (0.38, 1.0))

    fishTitle = pw.TextBox((0.04, 0.02), (0.92, 0.10))
    fishTitle.setText("Fish Targeting")
    fishTitle.setAlignment(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    fishTitle.disableSelection()
    fishPanel["title"] = fishTitle

    fishStatsLabels = pw.TextBoxes(
        (0.04, 0.14),
        (0.48, 0.36),
        labels=[
            "Fish X:",
            "Fish Y:",
            "Send g_diffX:",
            "Send g_diffY:",
        ],
    )
    fishStatsLabels.setAlignments(
        horizontal=pw.TextBox.AlignmentHorizontal.LEFT,
        vertical=pw.TextBox.AlignmentVertical.MIDDLE,
    )

    fishStatsValues = pw.TextBoxes(
        (0.52, 0.14),
        (0.44, 0.36),
        labels=["--", "--", "--", "--"],
    )
    fishStatsValues.setAlignments(
        horizontal=pw.TextBox.AlignmentHorizontal.LEFT,
        vertical=pw.TextBox.AlignmentVertical.MIDDLE,
    )

    for i in range(4):
        fishStatsLabels[f"text{i}"].disableSelection()
        fishStatsValues[f"text{i}"].disableSelection()

    fishPanel["statsLabels"] = fishStatsLabels
    fishPanel["statsValues"] = fishStatsValues

    fishDeadbandLabel = pw.TextBox((0.04, 0.52), (0.92, 0.08))
    fishDeadbandLabel.setText(
        f"Deadband: {FISH_DEADBAND_PIXELS:.0f} px from centre"
    )
    fishDeadbandLabel.setAlignment(
        horizontal=pw.TextBox.AlignmentHorizontal.LEFT
    )
    fishDeadbandLabel.disableSelection()
    fishPanel["deadbandLabel"] = fishDeadbandLabel

    executeButton = pw.Button((0.04, 0.63), (0.92, 0.12))
    executeButton.setText("Execute")
    fishPanel["executeButton"] = executeButton

    calibrationLabel = pw.TextBox((0.04, 0.77), (0.64, 0.11))
    calibrationLabel.setText("Calibration done")
    calibrationLabel.setAlignment(
        horizontal=pw.TextBox.AlignmentHorizontal.LEFT
    )
    calibrationLabel.disableSelection()
    fishPanel["calibrationLabel"] = calibrationLabel

    calibrationToggle = pw.ToggleButton((0.72, 0.77), (0.24, 0.11))
    calibrationToggle.disableSelection()
    fishPanel["calibrationToggle"] = calibrationToggle

    executeStatus = pw.TextBox((0.04, 0.90), (0.92, 0.09))
    executeStatus.setText("Idle")
    executeStatus.setAlignment(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    executeStatus.disableSelection()
    fishPanel["executeStatus"] = executeStatus

    fishTab["panel"] = fishPanel

    tabs = pw.Tab(
        (0.0, 0.0),
        (1.0, 1.0),
        uiElements=[mainTab, fishTab],
        names=["Controls", "Fish"],
    )
    control["tabs"] = tabs

    # Values shown on the Fish tab and used by the Execute button. Updated from
    # the newest vision snapshot in the main loop.
    targetDiff = {"x": 0.0, "y": 0.0, "hasFish": False}
    executionState = {"running": False, "status": "Idle"}

    def executeFishMove(_=None) -> None:
        if executionState["running"]:
            return
        if robot is None:
            executionState["status"] = "Robot not connected"
            return
        if not targetDiff["hasFish"]:
            executionState["status"] = "No fish detected"
            return

        diffX = float(targetDiff["x"]) * -0.5
        diffY = float(targetDiff["y"]) *  0.5
        executionState["running"] = True
        executionState["status"] = f"Sending {diffX:.1f} | {diffY:.1f} ..."

        def run() -> None:
            try:
                robot.write(DIFF_X_VARIABLE, diffX)
                robot.write(DIFF_Y_VARIABLE, diffY)
                print(
                    f"Execute: {DIFF_X_VARIABLE}={diffX:.2f}, "
                    f"{DIFF_Y_VARIABLE}={diffY:.2f}"
                )

                time.sleep(EXECUTE_WAIT_SECONDS)
                robot.write(TO_COMPUTER_VARIABLE, 0)
                print(f"Execute: {TO_COMPUTER_VARIABLE}=0")

                time.sleep(EXECUTE_WAIT_SECONDS)
                robot.write(TO_ROBOT_VARIABLE, 1)
                print(f"Execute: {TO_ROBOT_VARIABLE}=1")
                executionState["status"] = "Execute sequence sent"
            except Exception as error:
                executionState["status"] = f"Execute failed: {error}"
            finally:
                executionState["running"] = False

        threading.Thread(
            target=run,
            name="FishExecute",
            daemon=True,
        ).start()

    executeButton.setFunction(executeFishMove)

    last_calibration_state = False
    last_diff_values: dict[str, float] = {}
    last_diff_error = None
    last_read_poll = 0.0
    read_poll_interval = 1.0
    requested_read_names: set[str] = set()
    last_execute_status = executionState["status"]

    SAVE_TRAINING_IMAGES = False
    imageSaver = None
    if SAVE_TRAINING_IMAGES:
        try:
            imageSaver = ImageSaver()
            print(
                f"Saving unannotated images to {imageSaver.directory} "
                f"at {1.0 / IMAGE_SAVE_INTERVAL:.0f} Hz"
            )
        except Exception as error:
            print(f"Warning: could not start image saving: {error}")
            imageSaver = None

    try:
        while control.isRunning():
            now = time.monotonic()

            # Camera feed and detection statistics.
            snapshot = vision.latest() if vision is not None else None

            if vision is None:
                statsValues.setTexts([
                    "Vision not available",
                    "-- | --",
                    "-- x --",
                    "--",
                ])
                fishStatsValues.setTexts(["--", "--", "--", "--"])
                targetDiff["hasFish"] = False
            elif snapshot is None:
                statsValues.setTexts([
                    "Waiting for camera",
                    "-- | --",
                    "-- x --",
                    "--",
                ])
                fishStatsValues.setTexts(["--", "--", "--", "--"])
                targetDiff["hasFish"] = False
            else:
                if imageSaver is not None:
                    imageSaver.offer(snapshot.frame, snapshot.sequence)

                frame = (
                    snapshot.annotated_frame
                    if snapshot.annotated_frame is not None
                    else snapshot.frame
                )

                if frame is not None:
                    try:
                        surface = _numpy_bgr_to_surface(frame)
                        imageView.setImage(surface)
                        fishImageView.setImage(surface)
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
                    fishStatsValues.setTexts(["--", "--", "--", "--"])
                    targetDiff["hasFish"] = False
                else:
                    frameHeight, frameWidth = frame.shape[:2]
                    centerX = frameWidth / 2.0
                    centerY = frameHeight / 2.0
                    fishX, fishY = best.center
                    relativeX = fishX - centerX
                    relativeY = fishY - centerY

                    # Only move an axis when the fish is outside the deadband.
                    targetX = (
                        round(relativeX, 2)
                        if abs(relativeX) > FISH_DEADBAND_PIXELS
                        else 0.0
                    )
                    targetY = (
                        round(relativeY, 2)
                        if abs(relativeY) > FISH_DEADBAND_PIXELS
                        else 0.0
                    )

                    targetDiff["x"] = targetX
                    targetDiff["y"] = targetY
                    targetDiff["hasFish"] = True

                    statsValues.setTexts([
                        f"Fish detected ({len(snapshot.detections)} total)",
                        f"{relativeX:.1f} | {relativeY:.1f}",
                        f"{best.width:.1f} x {best.height:.1f} px",
                        f"{best.confidence:.3f}",
                    ])
                    fishStatsValues.setTexts([
                        f"{relativeX:+.1f} px",
                        f"{relativeY:+.1f} px",
                        f"{targetX:+.1f}",
                        f"{targetY:+.1f}",
                    ])

            calibration_state = bool(calibrationToggle.getValue())
            if calibration_state != last_calibration_state:
                last_calibration_state = calibration_state
                if robot is None:
                    executionState["status"] = "Robot not connected"
                else:
                    try:
                        robot.write(
                            CALIBRATION_DONE_VARIABLE,
                            calibration_state,
                        )
                    except Exception as error:
                        executionState["status"] = (
                            f"Calibration write failed: {error}"
                        )
                    else:
                        valueText = "true" if calibration_state else "false"
                        executionState["status"] = (
                            f"{CALIBRATION_DONE_VARIABLE} = {valueText}"
                        )
                        print(
                            f"Sent: {CALIBRATION_DONE_VARIABLE} = {valueText}"
                        )

            if executionState["status"] != last_execute_status:
                executeStatus.setText(executionState["status"])
                last_execute_status = executionState["status"]

            executeText = "Executing..." if executionState["running"] else "Execute"
            if executeButton.getText() != executeText:
                executeButton.setText(executeText)

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

            # Send only the diff sliders that changed since the last update,
            # and only while the Controls tab is visible. Otherwise the
            # sliders (usually 0) would overwrite values written by Execute on
            # the Fish tab. ThreadedRobotClient.write() is non-blocking and
            # coalesces writes per name, so dragging cannot build up a backlog.
            if robot is not None:
                # Track recovery even while another tab is active, so switching
                # back to Controls does not trigger a stale resend.
                diff_error = robot.last_error
                recovered = last_diff_error is not None and diff_error is None
                last_diff_error = diff_error

                if tabs.getActiveTab() == 0:
                    # A queued write may be dropped while the connection is
                    # down. After recovery, resend every setpoint once.
                    if recovered:
                        last_diff_values.clear()

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
