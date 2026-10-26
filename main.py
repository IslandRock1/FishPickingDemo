"""A small, self-contained demo of the pg-widgets library.

Run with ``python main.py``.  Slide the vertical sliders to change the
amplitude, frequency and noise of the plotted signal, use the colour picker to
change the line colour, and use the buttons/toggle to pause, clear, take a
screenshot or show a second trace.

The tabs at the bottom right of the left panel switch between the live plot and
the image view.  The image tab shows a placeholder until Vision.py supplies
frames; call ``image_view.setImage(surface)`` to display a real image.
"""

import math
import random

import pygame as pg
import pg_widgets as pw

from ImageView import ImageView


def _make_placeholder(size: tuple[int, int] = (640, 480)) -> pg.Surface:
    """Return a simple 4:3 test image so the Vision tab is not empty."""
    surface = pg.Surface(size)
    surface.fill((28, 32, 38))

    for x in range(0, size[0], 40):
        pg.draw.line(surface, (42, 50, 60), (x, 0), (x, size[1]))
    for y in range(0, size[1], 40):
        pg.draw.line(surface, (42, 50, 60), (0, y), (size[0], y))

    centre = (size[0] // 2, size[1] // 2)
    pg.draw.circle(surface, (0, 200, 180), centre, 90, 3)
    pg.draw.line(
        surface,
        (0, 200, 180),
        (centre[0] - 130, centre[1]),
        (centre[0] + 130, centre[1]),
        2,
    )
    pg.draw.line(
        surface,
        (0, 200, 180),
        (centre[0], centre[1] - 130),
        (centre[0], centre[1] + 130),
        2,
    )

    font = pg.font.SysFont("Arial", 28)
    text = font.render("Vision feed (placeholder)", True, (200, 210, 220))
    surface.blit(text, ((size[0] - text.get_width()) // 2, size[1] - 54))
    return surface


def main() -> None:
    control = pw.ControlManager()

    # ----- Left side: live plot and vision feed in tabs --------------------
    signal_tab = pw.UIGroup((0.0, 0.0), (1.0, 0.95))
    vision_tab = pw.UIGroup((0.0, 0.0), (1.0, 0.95))

    plot = pw.Plot((0.02, 0.02), (0.96, 0.96))
    plot.setTitle("Live signal")
    plot.setXLabel("time [s]")
    plot.setYLabel("amplitude")
    plot.setBounds(x_low=0.0, x_high=8.0, y_low=-1.5, y_high=1.5)
    plot.setColor("textColor", (225, 225, 225))
    plot.setColor("plot1", (0, 220, 255))
    signal_tab["plot"] = plot

    image_view = ImageView((0.02, 0.02), (0.96, 0.96))
    image_view.setImage(_make_placeholder())
    # Replace the placeholder when Vision is ready, e.g.:
    # image_view.setImage(surface_from_vision)
    vision_tab["image"] = image_view

    tabs = pw.Tab(
        (0.02, 0.04),
        (0.63, 0.79),
        uiElements=[vision_tab, signal_tab],
        names=["Vision", "Plot"],
    )
    control["tabs"] = tabs

    # ----- Sliders that drive the signal -----------------------------------
    sliders = pw.TuningSliders(
        (0.67, 0.04),
        (0.31, 0.38),
        labels=["Amp", "Freq", "Noise"],
        lower_bounds=[0.0, 0.2, 0.0],
        upper_bounds=[1.2, 3.0, 0.3],
        current_values=[0.8, 0.8, 0.05],
    )
    # Keep the value boxes read-only; the sliders themselves remain usable.
    for label in ("Amp", "Freq", "Noise"):
        sliders[f"value{label}"].disableSelection()
        sliders[f"text{label}"].disableSelection()
    control["sliders"] = sliders

    # ----- Buttons ---------------------------------------------------------
    pause_button = pw.Button((0.67, 0.44), (0.097, 0.07))
    pause_button.setText("Pause")

    clear_button = pw.Button((0.777, 0.44), (0.097, 0.07))
    clear_button.setText("Clear")

    snapshot_button = pw.Button((0.884, 0.44), (0.097, 0.07))
    snapshot_button.setText("Snapshot")

    control["pauseButton"] = pause_button
    control["clearButton"] = clear_button
    control["snapshotButton"] = snapshot_button

    # ----- Toggle for the second trace -------------------------------------
    trace_label = pw.TextBox((0.67, 0.515), (0.20, 0.05))
    trace_label.setText("Second trace")
    trace_label.disableSelection()
    control["traceLabel"] = trace_label

    show_second = pw.ToggleButton((0.89, 0.515), (0.09, 0.05))
    # UIElement.__new__ only forwards its own border keywords, so subclass
    # keywords such as ``state=True`` cannot be passed to the constructor.
    show_second.setValue(True)
    control["showSecond"] = show_second

    # ----- Colour picker for the main trace --------------------------------
    picker = pw.ColorPicker((0.67, 0.57), (0.31, 0.17))
    picker["sliderHue"].setValue(0.33)
    picker["sliderSat"].setValue(0.75)
    picker["sliderVal"].setValue(1.0)
    for name in ("Hue", "Sat", "Val"):
        picker[f"text{name}"].disableSelection()
    control["picker"] = picker

    # ----- Buffer / status readouts ----------------------------------------
    progress = pw.ProgressBar((0.67, 0.75), (0.31, 0.035))
    control["progress"] = progress

    status = pw.TextBoxes(
        (0.67, 0.79),
        (0.31, 0.19),
        labels=["time", "render", "samples", "last"],
    )
    control["status"] = status

    # ----- Demo state and callbacks ----------------------------------------
    MAX_POINTS = 600
    WINDOW = 8.0

    paused = False
    sim_time = 0.0
    samples = 0
    last_action = "ready"
    was_showing_second = False
    last_color = None
    clock = pg.time.Clock()

    def set_action(text: str) -> None:
        nonlocal last_action
        last_action = text

    def toggle_pause() -> None:
        nonlocal paused
        paused = not paused
        pause_button.setText("Resume" if paused else "Pause")
        set_action("paused" if paused else "running")

    def clear_plot() -> None:
        nonlocal sim_time, samples
        sim_time = 0.0
        samples = 0
        plot.setValue([], [], plotIx=0)
        plot.setValue([], [], plotIx=1)
        plot.setBounds(x_low=0.0, x_high=WINDOW)
        set_action("plot cleared")

    def take_screenshot() -> None:
        control.screenshot()
        set_action(f"saved frame {control.getIteration()}")

    pause_button.overrideOnLeftClick(toggle_pause)
    clear_button.overrideOnLeftClick(clear_plot)
    snapshot_button.overrideOnLeftClick(take_screenshot)

    # ----- Main loop -------------------------------------------------------
    while control.isRunning():
        # 60 FPS keeps the time axis stable and avoids burning a CPU core.
        dt = min(clock.tick(60) / 1000.0, 0.1)  # cap after a long stall

        # Clear the second trace when it is switched off.
        show_second_now = show_second.getValue()
        if show_second_now and not was_showing_second:
            was_showing_second = True
        elif not show_second_now and was_showing_second:
            plot.setValue([], [], plotIx=1)
            was_showing_second = False

        if not paused:
            sim_time += dt
            amplitude, frequency, noise = sliders.getValue()
            omega = 2.0 * math.pi * frequency

            first = amplitude * math.sin(omega * sim_time)
            second = amplitude * math.cos(omega * sim_time)
            second += noise * random.uniform(-1.0, 1.0)

            plot.addValue(sim_time, first, plotIx=0, maxLength=MAX_POINTS)
            if show_second_now:
                plot.addValue(sim_time, second, plotIx=1, maxLength=MAX_POINTS)

            samples = min(MAX_POINTS, samples + 1)
            if sim_time < WINDOW:
                plot.setBounds(x_low=0.0, x_high=WINDOW)
            else:
                plot.setBounds(x_low=sim_time - WINDOW, x_high=sim_time)

        color = tuple(picker.getValue())
        if color != last_color:
            plot.setColor("plot0", color)
            last_color = color

        progress.setValue(samples / MAX_POINTS)

        render_time = control.getRenderTime()
        if render_time is None or render_time < 0:
            render_text = "render      --"
        else:
            render_text = f"render {render_time * 1000:6.2f} ms"

        status.setTexts([
            f"time    {sim_time:6.2f} s",
            render_text,
            f"samples {samples:3d} / {MAX_POINTS}",
            f"last    {last_action}",
        ])

        control.update()

    control.close()


if __name__ == "__main__":
    main()
