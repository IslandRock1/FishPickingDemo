"""Fish picking demo GUI.

Top left: camera feed from Vision (60% width, aspect ratio preserved).
Below that: AI detection stats (position relative to center, size; handles no-detection).
Right side: communication interface to send and read arbitrary variables, plus sliders for g_diffX/Y/Z.
"""

import pygame as pg
import pg_widgets as pw
import numpy as np
import time

from ImageView import ImageView
from Vision import Vision
from Communication import ThreadedRobotClient


def _numpy_bgr_to_surface(frame: np.ndarray) -> pg.Surface:
    """Convert BGR numpy frame (from OpenCV) to pygame Surface."""
    rgb = frame[:, :, ::-1]
    surface = pg.surfarray.make_surface(rgb.swapaxes(0, 1))
    return surface


def main() -> None:
    control = pw.ControlManager()

    # Start vision system
    vision = Vision(annotate=True)
    try:
        vision.start()
    except Exception as e:
        print(f"Warning: Could not start Vision: {e}")
        vision = None

    # Start robot communication (non-blocking for GUI)
    robot = None
    try:
        robot = ThreadedRobotClient()
        robot.start()
        print("Robot communication started")
    except Exception as e:
        print(f"Warning: Could not start robot communication: {e}")
        robot = None

    # ----- Left side: Camera feed (60% width) and stats below -----
    # Camera feed takes top portion of left side
    camera_group = pw.UIGroup((0.0, 0.0), (0.6, 0.7))
    image_view = ImageView((0.02, 0.02), (0.96, 0.96))
    camera_group["image"] = image_view

    control["camera"] = camera_group

    # Stats below camera feed
    stats_group = pw.UIGroup((0.0, 0.72), (0.6, 0.26))
    stats_label = pw.TextBoxes((0.02, 0.05), (0.96, 0.9), labels=[
        "Status: Initializing...",
        "Position rel. center: --",
        "Size (W x H): --",
        "Area: --",
        "Confidence: --",
    ])
    stats_label.setAlignments(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    stats_group["stats"] = stats_label
    control["stats"] = stats_group

    # ----- Right side: Communication interface -----
    comm_group = pw.UIGroup((0.62, 0.0), (0.36, 1.0))

    # Diff sliders section
    diff_label = pw.TextBox((0.02, 0.01), (0.96, 0.05))
    diff_label.setText("Diff Control (g_diffX/Y/Z)")
    diff_label.setAlignment(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    diff_label.disableSelection()
    comm_group["diffLabel"] = diff_label

    diff_sliders = pw.TuningSliders(
        (0.02, 0.07),
        (0.96, 0.22),
        labels=["g_diffX", "g_diffY", "g_diffZ"],
        lower_bounds=[-250.0, -250.0, -250.0],
        upper_bounds=[250.0, 250.0, 250.0],
        current_values=[0.0, 0.0, 0.0],
    )
    for label in ("g_diffX", "g_diffY", "g_diffZ"):
        diff_sliders[f"value{label}"].disableSelection()
        diff_sliders[f"text{label}"].disableSelection()
    comm_group["diffSliders"] = diff_sliders

    # Write section
    write_label = pw.TextBox((0.02, 0.31), (0.96, 0.05))
    write_label.setText("WRITE")
    write_label.setAlignment(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    write_label.disableSelection()
    comm_group["writeLabel"] = write_label

    # Variable name input
    name_label = pw.TextBox((0.02, 0.37), (0.96, 0.05))
    name_label.setText("Variable Name:")
    name_label.setAlignment(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    name_label.disableSelection()
    comm_group["nameLabel"] = name_label

    var_name_input = pw.TextBox((0.02, 0.43), (0.96, 0.07))
    var_name_input.setText("g_testVar")
    var_name_input.setAlignment(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    comm_group["varName"] = var_name_input

    # Variable value input
    value_label = pw.TextBox((0.02, 0.51), (0.96, 0.05))
    value_label.setText("Variable Value:")
    value_label.setAlignment(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    value_label.disableSelection()
    comm_group["valueLabel"] = value_label

    var_value_input = pw.TextBox((0.02, 0.57), (0.96, 0.07))
    var_value_input.setText("0")
    var_value_input.setAlignment(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    comm_group["varValue"] = var_value_input

    # Send button
    send_button = pw.Button((0.02, 0.65), (0.96, 0.08))
    send_button.setText("Send")

    # Callback setup
    def send_callback(variable=None):
        var_name = var_name_input.getText()
        var_value = var_value_input.getText()
        print(f"Send requested: {var_name} = {var_value}")
        if robot is not None:
            try:
                # Try to convert to appropriate type
                try:
                    if '.' in var_value or 'e' in var_value.lower():
                        val = float(var_value)
                    else:
                        val = int(var_value)
                except ValueError:
                    if var_value.lower() in ('true', 'false'):
                        val = var_value.lower() == 'true'
                    else:
                        val = var_value
                robot.write(var_name, val)
                print(f"Sent via robot: {var_name} = {val}")
            except Exception as e:
                print(f"Error sending: {e}")
        else:
            print("Robot not connected")

    try:
        send_button.setFunction(send_callback)
        send_button.setVariable(None)
    except Exception:
        pass

    try:
        send_button.overrideOnLeftClick(send_callback)
    except Exception:
        pass

    comm_group["sendButton"] = send_button

    # Read section
    read_label = pw.TextBox((0.02, 0.75), (0.96, 0.05))
    read_label.setText("READ")
    read_label.setAlignment(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    read_label.disableSelection()
    comm_group["readLabel"] = read_label

    read_name_label = pw.TextBox((0.02, 0.81), (0.96, 0.05))
    read_name_label.setText("Variable Name:")
    read_name_label.setAlignment(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    read_name_label.disableSelection()
    comm_group["readNameLabel"] = read_name_label

    read_var_name_input = pw.TextBox((0.02, 0.87), (0.96, 0.07))
    read_var_name_input.setText("g_testFloat")
    read_var_name_input.setAlignment(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    comm_group["readVarName"] = read_var_name_input

    read_button = pw.Button((0.02, 0.95), (0.47, 0.04))
    read_button.setText("Read")

    read_value_display = pw.TextBox((0.51, 0.95), (0.47, 0.04))
    read_value_display.setText("--")
    read_value_display.setAlignment(horizontal=pw.TextBox.AlignmentHorizontal.LEFT)
    read_value_display.disableSelection()
    comm_group["readValueDisplay"] = read_value_display

    def read_callback(variable=None):
        var_name = read_var_name_input.getText()
        print(f"Read requested: {var_name}")
        if robot is not None:
            try:
                robot.request(var_name)
                print(f"Requested {var_name} for polling")
            except Exception as e:
                print(f"Error requesting: {e}")
        else:
            print("Robot not connected")

    try:
        read_button.setFunction(read_callback)
        read_button.setVariable(None)
    except Exception:
        pass

    try:
        read_button.overrideOnLeftClick(read_callback)
    except Exception:
        pass

    comm_group["readButton"] = read_button
    control["comm"] = comm_group

    # Store references
    control._var_name_input = var_name_input
    control._var_value_input = var_value_input
    control._send_callback = send_callback
    control._read_var_name_input = read_var_name_input
    control._read_value_display = read_value_display
    control._robot = robot
    control._diff_sliders = diff_sliders

    # Timer for sending diff values
    last_diff_send = time.time()
    diff_send_interval = 1.0  # Send every second

    clock = pg.time.Clock()

    while control.isRunning():
        clock.tick(60)
        current_time = time.time()

        # Send diff values periodically
        if robot is not None and current_time - last_diff_send >= diff_send_interval:
            try:
                diff_x, diff_y, diff_z = diff_sliders.getValue()
                robot.write("g_diffX", float(diff_x))
                robot.write("g_diffY", float(diff_y))
                robot.write("g_diffZ", float(diff_z))
                print(f"Sent diffs: g_diffX={diff_x:.2f}, g_diffY={diff_y:.2f}, g_diffZ={diff_z:.2f}")
            except Exception as e:
                print(f"Error sending diffs: {e}")
            last_diff_send = current_time

        # Update camera feed from vision
        if vision is not None:
            snapshot = vision.latest()
            if snapshot is not None:
                frame = snapshot.annotated_frame if snapshot.annotated_frame is not None else snapshot.frame
                if frame is not None:
                    try:
                        surf = _numpy_bgr_to_surface(frame)
                        image_view.setImage(surf)
                    except Exception:
                        pass

                best = snapshot.best
                if best is None or len(snapshot.detections) == 0:
                    stats_label.setTexts([
                        "Status: No fish detected",
                        "Position rel. center: --",
                        "Size (W x H): --",
                        "Area: --",
                        "Confidence: --",
                    ])
                else:
                    frame_h, frame_w = frame.shape[:2]
                    center_x = frame_w / 2.0
                    center_y = frame_h / 2.0
                    fish_cx, fish_cy = best.center
                    rel_x = fish_cx - center_x
                    rel_y = fish_cy - center_y

                    stats_label.setTexts([
                        f"Status: Fish detected ({len(snapshot.detections)} total)",
                        f"Position rel. center: ({rel_x:.1f}, {rel_y:.1f}) px",
                        f"Size (W x H): ({best.width:.1f}, {best.height:.1f}) px",
                        f"Area: {best.area:.1f} px²",
                        f"Confidence: {best.confidence:.3f}",
                    ])
            else:
                stats_label.setTexts([
                    "Status: Waiting for camera...",
                    "Position rel. center: --",
                    "Size (W x H): --",
                    "Area: --",
                    "Confidence: --",
                ])
        else:
            stats_label.setTexts([
                "Status: Vision not available",
                "Position rel. center: --",
                "Size (W x H): --",
                "Area: --",
                "Confidence: --",
            ])

        # Update read value display if we have robot and requested variable
        if robot is not None:
            try:
                var_name = read_var_name_input.getText()
                if var_name and robot.is_ready(var_name):
                    val = robot.get_value(var_name)
                    if val is not None:
                        read_value_display.setText(str(val))
            except Exception:
                pass

        control.update()

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
