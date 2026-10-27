"""USB camera capture and YOLO fish detection on a background thread.

This module pairs a USB camera with the ONNX fish detector in
``model/best.onnx`` and runs capture, inference and annotation in one worker
thread.  The caller only ever reads an immutable snapshot, so a GUI can keep a
steady frame rate even when inference is slow::

    from Vision import Vision

    with Vision() as vision:
        while gui.is_running():
            snapshot = vision.latest()          # never blocks
            if snapshot is not None and snapshot.best is not None:
                fish = snapshot.best
                print(fish.center, fish.confidence)
            gui.update()

Turn on the OpenCV preview while bringing the camera up::

    with Vision(show_preview=True) as vision:
        ...

A preview window must be pumped from the same thread that created it on some
platforms, so it is off by default.  Prefer reading ``snapshot.annotated_frame``
and showing it in your own GUI.

Dependencies
------------
The detector uses ONNX Runtime and OpenCV, not Ultralytics::

    pip install numpy opencv-python onnxruntime

Both are imported lazily, so ``import Vision`` works before they are installed
and raises :class:`VisionDependencyError` with the install command when the
camera or model is actually started.

Model
-----
``model/best.onnx`` is an Ultralytics export with ``nms=False``: it returns the
raw detection head, so boxes are decoded and non-maximum suppressed here.  It
was exported at 640x640 for a single class, ``fish``; class names and input size
are read from the model metadata where available.  Pass ``model_path=...`` to
use another export.

Threading
---------
:meth:`Vision.start` loads the model and opens the camera, then starts a daemon
worker that repeatedly reads a frame, runs inference and publishes a
:class:`VisionSnapshot`.  :meth:`Vision.latest` and :meth:`Vision.wait_ready`
are the only read paths and are safe to call from the GUI thread.  The optional
``on_result`` callback runs *on the worker thread* and must not block; use it
for logging or pushing into a queue, not for drawing.

A camera that returns no frame is released and reopened every
``reconnect_interval`` seconds, so unplugging and replugging the USB camera does
not kill the process.  Inference errors are reported through
:attr:`Vision.last_error` and do not stop the worker either.
"""

from __future__ import annotations

import ast
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_CAMERA_INDICES",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_FPS",
    "DEFAULT_HEIGHT",
    "DEFAULT_IOU_THRESHOLD",
    "DEFAULT_MAX_DETECTIONS",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_RECONNECT_INTERVAL",
    "DEFAULT_WIDTH",
    "PREVIEW_WINDOW",
    "CameraError",
    "Detection",
    "ModelError",
    "OnnxFishDetector",
    "UsbCamera",
    "Vision",
    "VisionDependencyError",
    "VisionError",
    "VisionSnapshot",
    "draw_detections",
]

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "best.onnx"

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MAX_DETECTIONS = 100
DEFAULT_INPUT_SIZE = (640, 640)

DEFAULT_CAMERA_INDICES = (1, 2, 3, 4)
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 15
DEFAULT_RECONNECT_INTERVAL = 1.0

PREVIEW_WINDOW = "Vision preview"

_PAD_COLOR = 114
_MAX_CANDIDATES = 30_000
_MAX_READ_FAILURES = 10
_FALLBACK_CLASS_NAMES = ("fish",)


class VisionError(Exception):
    """Base exception for camera and model failures."""


class VisionDependencyError(VisionError):
    """Raised when OpenCV or ONNX Runtime is not installed."""


class ModelError(VisionError):
    """Raised when the ONNX model cannot be loaded or interpreted."""


class CameraError(VisionError):
    """Raised when no usable USB camera could be opened."""


def _import_cv2() -> Any:
    """Import OpenCV lazily so the module imports without the dependency."""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - depends on the machine
        raise VisionDependencyError(
            "Vision needs OpenCV for camera capture and drawing. "
            "Install it with: pip install opencv-python"
        ) from error
    return cv2


def _import_onnxruntime() -> Any:
    """Import ONNX Runtime lazily so the module imports without the dependency."""
    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - depends on the machine
        raise VisionDependencyError(
            "Vision needs ONNX Runtime for inference. "
            "Install it with: pip install onnxruntime"
        ) from error
    return ort


@dataclass(frozen=True)
class Detection:
    """One fish bounding box in pixel coordinates of the captured frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Bounding box as ``(x1, y1, x2, y2)`` in pixels."""
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Box centre as ``(x, y)`` in pixels; the usual pick point."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def as_dict(self) -> dict[str, Any]:
        """Plain, JSON-friendly representation for logging or the robot API."""
        centre_x, centre_y = self.center
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "center_x": centre_x,
            "center_y": centre_y,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass(frozen=True, eq=False, repr=False)
class VisionSnapshot:
    """One processed frame, frozen at the moment the worker finished it.

    ``frame`` is the raw BGR frame as read from the camera and is never
    modified.  ``annotated_frame`` is a separate BGR copy with boxes drawn on
    it, or ``None`` when annotation is disabled.  Both arrays belong to the
    worker; treat them as read-only (copy before drawing on them).
    """

    sequence: int
    captured_at: float
    processed_at: float
    frame: NDArray[np.uint8]
    annotated_frame: NDArray[np.uint8] | None
    detections: tuple[Detection, ...]
    inference_seconds: float
    fps: float
    camera_index: int | None

    @property
    def best(self) -> Detection | None:
        """Most confident fish in this frame, or ``None`` when empty.

        This is only a confidence ranking.  If the robot should pick the fish
        nearest the gripper or the image centre, rank ``detections`` yourself.
        """
        return max(
            self.detections,
            key=lambda detection: detection.confidence,
            default=None,
        )

    @property
    def age(self) -> float:
        """Seconds since the frame was captured."""
        return time.monotonic() - self.captured_at

    @property
    def is_empty(self) -> bool:
        """Whether no fish was detected in this frame."""
        return not self.detections

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(sequence={self.sequence}, "
            f"detections={len(self.detections)}, "
            f"inference={self.inference_seconds * 1000:.1f} ms, "
            f"fps={self.fps:.1f})"
        )


def _nms(
    boxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    iou_threshold: float,
) -> list[int]:
    """Greedy non-maximum suppression on ``(N, 4)`` xyxy boxes."""
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(boxes[index, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[index, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[index, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[index, 3], boxes[rest, 3])

        intersection = np.clip(xx2 - xx1, 0.0, None) * np.clip(yy2 - yy1, 0.0, None)
        area = (boxes[index, 2] - boxes[index, 0]) * (
            boxes[index, 3] - boxes[index, 1]
        )
        other_area = (boxes[rest, 2] - boxes[rest, 0]) * (
            boxes[rest, 3] - boxes[rest, 1]
        )
        union = area + other_area - intersection
        iou = intersection / np.maximum(union, 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def draw_detections(
    frame: NDArray[np.uint8],
    detections: Sequence[Detection],
    best: Detection | None = None,
) -> NDArray[np.uint8]:
    """Return a BGR copy of ``frame`` with boxes, labels and centres drawn.

    ``best`` (usually :attr:`VisionSnapshot.best`) is highlighted in green with
    a ``PICK`` label.  All other detections are drawn in orange.
    """
    cv2 = _import_cv2()
    canvas = frame.copy()

    for detection in detections:
        is_best = best is not None and detection == best
        colour = (60, 220, 60) if is_best else (60, 180, 255)
        thickness = 3 if is_best else 2

        x1 = int(round(detection.x1))
        y1 = int(round(detection.y1))
        x2 = int(round(detection.x2))
        y2 = int(round(detection.y2))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

        label = f"{detection.class_name} {detection.confidence:.2f}"
        if is_best:
            label = f"PICK {label}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_top = y1 - text_height - baseline - 6
        if label_top < 0:
            label_top = y1 + 4
        cv2.rectangle(
            canvas,
            (x1, label_top),
            (x1 + text_width + 6, label_top + text_height + baseline + 4),
            colour,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 3, label_top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

        centre_x, centre_y = detection.center
        cv2.circle(
            canvas,
            (int(round(centre_x)), int(round(centre_y))),
            5,
            colour,
            -1,
            cv2.LINE_AA,
        )

    return canvas


class OnnxFishDetector:
    """Decode ``model/best.onnx`` into :class:`Detection` objects.

    The session is created by :meth:`load`; :meth:`detect` calls it on demand,
    so a detector can also be used directly outside :class:`Vision`.  This
    object is not thread-safe: share one detector between threads only if the
    calls are serialized.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        confidence: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        class_names: Sequence[str] | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        if not 0.0 < confidence <= 1.0:
            raise ValueError(f"confidence must be in (0, 1], got {confidence}")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        if max_detections <= 0:
            raise ValueError(f"max_detections must be positive, got {max_detections}")

        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.providers = tuple(providers) if providers else None
        self._configured_class_names = (
            tuple(str(name) for name in class_names) if class_names else None
        )

        self._session: Any | None = None
        self._cv2: Any | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._input_size = DEFAULT_INPUT_SIZE
        self._class_names: tuple[str, ...] = (
            self._configured_class_names or _FALLBACK_CLASS_NAMES
        )
        self._load_lock = threading.Lock()

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "not loaded"
        return f"{type(self).__name__}({str(self.model_path)!r}, {state})"

    @property
    def is_loaded(self) -> bool:
        """Whether an ONNX session has been created."""
        return self._session is not None

    @property
    def class_names(self) -> tuple[str, ...]:
        """Class names in model output order."""
        return self._class_names

    @property
    def input_size(self) -> tuple[int, int]:
        """Model input size as ``(height, width)``."""
        return self._input_size

    def load(self) -> None:
        """Create the ONNX session.  Calling this twice is harmless.

        Raises:
            ModelError: if the file is missing or ONNX Runtime rejects it.
            VisionDependencyError: if ONNX Runtime or OpenCV is missing.
        """
        with self._load_lock:
            if self._session is not None:
                return

            if not self.model_path.is_file():
                raise ModelError(f"Model file not found: {self.model_path}")

            cv2 = _import_cv2()
            ort = _import_onnxruntime()

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            kwargs: dict[str, Any] = {"sess_options": options}
            if self.providers:
                kwargs["providers"] = list(self.providers)

            try:
                session = ort.InferenceSession(str(self.model_path), **kwargs)
            except Exception as error:
                raise ModelError(
                    f"Could not load ONNX model {self.model_path}: {error}"
                ) from error

            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if not inputs or not outputs:
                raise ModelError(
                    f"ONNX model {self.model_path} has no inputs or no outputs"
                )

            input_meta = inputs[0]
            shape = list(input_meta.shape)
            if len(shape) == 4 and all(
                isinstance(value, int) and value > 0 for value in shape[2:]
            ):
                self._input_size = (int(shape[2]), int(shape[3]))

            metadata = self._metadata(session)
            metadata_size = _parse_size(metadata.get("imgsz"))
            if metadata_size is not None and not (
                len(shape) == 4
                and all(isinstance(value, int) and value > 0 for value in shape[2:])
            ):
                self._input_size = metadata_size

            if self._configured_class_names is None:
                names = _parse_class_names(metadata.get("names"))
                if names:
                    self._class_names = names

            self._session = session
            self._cv2 = cv2
            self._input_name = input_meta.name
            self._output_name = outputs[0].name

            logger.info(
                "Loaded %s (input %dx%d, classes=%s)",
                self.model_path,
                self._input_size[1],
                self._input_size[0],
                ", ".join(self._class_names),
            )

    def detect(
        self, frame: NDArray[np.uint8]
    ) -> tuple[tuple[Detection, ...], float]:
        """Run inference on one BGR frame.

        Returns the detections in original-frame pixel coordinates and the
        inference time in seconds.
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with shape (H, W, 3)")
        if not self.is_loaded:
            self.load()

        assert self._cv2 is not None
        assert self._session is not None
        assert self._input_name is not None
        assert self._output_name is not None

        height, width = frame.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(self._cv2, frame)

        started = time.perf_counter()
        outputs = self._session.run(
            [self._output_name], {self._input_name: blob}
        )
        elapsed = time.perf_counter() - started

        detections = self._postprocess(
            outputs[0],
            width=width,
            height=height,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        return detections, elapsed

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _metadata(session: Any) -> dict[str, str]:
        try:
            metadata = session.get_modelmeta().custom_metadata_map
        except Exception:  # noqa: BLE001 - metadata is optional
            return {}
        return dict(metadata or {})

    def _preprocess(
        self, cv2: Any, frame: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float32], float, float, float]:
        input_height, input_width = self._input_size
        height, width = frame.shape[:2]
        scale = min(input_width / width, input_height / height)
        scaled_width = max(1, int(round(width * scale)))
        scaled_height = max(1, int(round(height * scale)))

        resized = cv2.resize(
            frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR
        )
        canvas = np.full(
            (input_height, input_width, 3), _PAD_COLOR, dtype=np.uint8
        )
        pad_x = (input_width - scaled_width) // 2
        pad_y = (input_height - scaled_height) // 2
        canvas[pad_y : pad_y + scaled_height, pad_x : pad_x + scaled_width] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
        return blob, scale, float(pad_x), float(pad_y)

    def _postprocess(
        self,
        output: Any,
        *,
        width: int,
        height: int,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> tuple[Detection, ...]:
        prediction = np.asarray(output)

        # Raw Ultralytics detect output is (batch, 4 + classes, anchors).
        if prediction.ndim == 3:
            if prediction.shape[0] != 1:
                raise ModelError(
                    f"Expected a batch size of 1, got {prediction.shape[0]}"
                )
            prediction = prediction[0]
        if prediction.ndim != 2:
            raise ModelError(f"Unexpected model output shape: {prediction.shape}")
        if prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T

        if prediction.shape[1] < 5:
            raise ModelError(
                "Model output does not look like a YOLO detection head "
                f"(shape {prediction.shape})"
            )

        boxes = prediction[:, :4]
        scores = prediction[:, 4:]
        if scores.shape[1] != len(self._class_names):
            raise ModelError(
                f"Model outputs {scores.shape[1]} classes but "
                f"{len(self._class_names)} names are configured; pass "
                "class_names=... to OnnxFishDetector"
            )

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]
        keep = confidences >= self.confidence
        if not np.any(keep):
            return ()

        boxes = boxes[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        if boxes.shape[0] > _MAX_CANDIDATES:
            top = np.argsort(confidences)[-_MAX_CANDIDATES:]
            boxes = boxes[top]
            class_ids = class_ids[top]
            confidences = confidences[top]

        centre_x, centre_y, box_width, box_height = boxes.T
        xyxy = np.stack(
            [
                (centre_x - box_width / 2.0 - pad_x) / scale,
                (centre_y - box_height / 2.0 - pad_y) / scale,
                (centre_x + box_width / 2.0 - pad_x) / scale,
                (centre_y + box_height / 2.0 - pad_y) / scale,
            ],
            axis=1,
        )
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0.0, float(width))
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0.0, float(height))

        valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        if not np.any(valid):
            return ()
        xyxy = xyxy[valid]
        class_ids = class_ids[valid]
        confidences = confidences[valid]

        # Class-aware NMS: YOLO boxes can overlap across classes, but boxes of
        # the same class should not.
        kept: list[int] = []
        for class_id in np.unique(class_ids):
            indices = np.flatnonzero(class_ids == class_id)
            local = _nms(xyxy[indices], confidences[indices], self.iou_threshold)
            kept.extend(int(indices[index]) for index in local)

        kept.sort(key=lambda index: float(confidences[index]), reverse=True)
        kept = kept[: self.max_detections]

        return tuple(
            Detection(
                x1=float(xyxy[index, 0]),
                y1=float(xyxy[index, 1]),
                x2=float(xyxy[index, 2]),
                y2=float(xyxy[index, 3]),
                confidence=float(confidences[index]),
                class_id=int(class_ids[index]),
                class_name=self._class_names[int(class_ids[index])],
            )
            for index in kept
        )


class UsbCamera:
    """A USB camera that probes ``indices`` until one delivers a frame.

    On Windows, DirectShow is used by default because it starts faster and is
    more forgiving than Media Foundation with cheap USB webcams.  Pass
    ``backend=...`` to override it, for example ``cv2.CAP_MSMF``.
    """

    def __init__(
        self,
        indices: Sequence[int] | int = DEFAULT_CAMERA_INDICES,
        *,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        backend: int | None = None,
    ) -> None:
        if isinstance(indices, int):
            indices = (indices,)
        self.indices = tuple(int(index) for index in indices)
        if not self.indices:
            raise ValueError("indices must contain at least one camera index")
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("width, height and fps must be positive")

        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.backend = backend

        self.index: int | None = None
        self.width: int | None = None
        self.height: int | None = None
        self.fps: float | None = None

        self._capture: Any | None = None
        self._cv2: Any | None = None

    def __repr__(self) -> str:
        state = "open" if self.is_open else "closed"
        return f"{type(self).__name__}(index={self.index!r}, {state})"

    def __enter__(self) -> "UsbCamera":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    @property
    def is_open(self) -> bool:
        """Whether a capture object is currently open."""
        return self._capture is not None

    def open(self) -> None:
        """Open the first camera that returns a frame.

        Raises:
            CameraError: if none of the configured indices works.
            VisionDependencyError: if OpenCV is not installed.
        """
        if self._capture is not None:
            return

        cv2 = self._cv2 or _import_cv2()
        self._cv2 = cv2
        backend = self.backend
        if backend is None:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

        for index in self.indices:
            capture = cv2.VideoCapture(index, backend)
            if not capture.isOpened():
                capture.release()
                continue

            # MJPG is much faster than raw YUYV on most USB webcams at 720p.
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.requested_width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.requested_height))
            capture.set(cv2.CAP_PROP_FPS, float(self.requested_fps))
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            frame = self._read_first_frame(capture)
            if frame is None:
                capture.release()
                continue

            self._capture = capture
            self.index = index
            self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
            self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
            self.fps = float(capture.get(cv2.CAP_PROP_FPS))
            logger.info(
                "Opened camera %d at %dx%d (requested %dx%d @ %d fps)",
                index,
                self.width,
                self.height,
                self.requested_width,
                self.requested_height,
                self.requested_fps,
            )
            return

        raise CameraError(
            "Could not open a USB camera at any of "
            f"{', '.join(str(index) for index in self.indices)}"
        )

    def read(self) -> NDArray[np.uint8] | None:
        """Read one frame, or ``None`` if the camera dropped it."""
        capture = self._capture
        if capture is None:
            return None
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        """Release the camera.  Safe to call when already closed."""
        capture, self._capture = self._capture, None
        self.index = None
        self.width = None
        self.height = None
        self.fps = None
        if capture is not None:
            capture.release()
            logger.debug("Camera released")

    def reopen(self) -> None:
        """Release and probe the indices again."""
        self.release()
        self.open()

    @staticmethod
    def _read_first_frame(capture: Any) -> NDArray[np.uint8] | None:
        """Discard a few warm-up frames and return the first usable one."""
        for _ in range(5):
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.02)
        return None


class Vision:
    """Background USB camera + fish detector for a GUI thread.

    Example:
        vision = Vision()
        vision.start()
        try:
            while running:
                snapshot = vision.latest()
                ...
        finally:
            vision.stop()

    The worker publishes a new :class:`VisionSnapshot` for every processed
    frame.  Consumers that fall behind simply miss intermediate frames; the
    latest snapshot is always the newest one.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        camera_indices: Sequence[int] | int = DEFAULT_CAMERA_INDICES,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        confidence: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        class_names: Sequence[str] | None = None,
        annotate: bool = True,
        show_preview: bool = False,
        reconnect_interval: float = DEFAULT_RECONNECT_INTERVAL,
        on_result: Callable[[VisionSnapshot], None] | None = None,
        detector: OnnxFishDetector | None = None,
        camera: UsbCamera | None = None,
    ) -> None:
        self._detector = detector or OnnxFishDetector(
            model_path,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            class_names=class_names,
        )
        self._camera = camera or UsbCamera(
            camera_indices, width=width, height=height, fps=fps
        )
        self.annotate = annotate
        self.show_preview = show_preview
        self.reconnect_interval = max(0.0, reconnect_interval)
        self.on_result = on_result

        self._condition = threading.Condition()
        self._latest: VisionSnapshot | None = None
        self._last_error: VisionError | None = None
        self._sequence = 0

        self._start_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._worker_finished = threading.Event()
        self._worker_finished.set()
        self._preview_created = False

    def __repr__(self) -> str:
        state = "running" if self.is_running else "stopped"
        return f"{type(self).__name__}({str(self._detector.model_path)!r}, {state})"

    def __enter__(self) -> "Vision":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the worker thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def last_error(self) -> VisionError | None:
        """Most recent camera or inference failure, cleared after a success."""
        with self._condition:
            return self._last_error

    @property
    def detector(self) -> OnnxFishDetector:
        """The detector in use."""
        return self._detector

    @property
    def camera(self) -> UsbCamera:
        """The camera in use."""
        return self._camera

    def start(self) -> None:
        """Load the model, open the camera and start the worker.

        This fails fast: :class:`ModelError` or :class:`CameraError` is raised
        immediately instead of being deferred to the worker.
        """
        with self._start_lock:
            if self.is_running:
                return

            self._detector.load()
            if not self._camera.is_open:
                self._camera.open()

            with self._condition:
                stop = threading.Event()
                finished = threading.Event()
                self._stop = stop
                self._worker_finished = finished
                self._thread = threading.Thread(
                    target=self._run,
                    args=(stop, finished),
                    name=f"{type(self).__name__}-worker",
                    daemon=True,
                )
                self._thread.start()

    def stop(self, timeout: float | None = 2.0) -> None:
        """Stop the worker and release the camera.

        Waits up to ``timeout`` seconds.  If the worker is still inside a
        blocking camera read, releasing the camera aborts it, like closing a
        socket in :class:`Communication.ThreadedRobotClient`.
        """
        with self._start_lock:
            stop, self._stop = self._stop, None
            if stop is not None:
                stop.set()

            thread, self._thread = self._thread, None
            if thread is None or thread is threading.current_thread():
                self._camera.release()
                self._close_preview()
                return

            thread.join(timeout)
            if thread.is_alive():
                logger.warning("Vision worker did not stop within %s s", timeout)
                self._camera.release()
                thread.join(timeout)

        self._close_preview()

    def close(self) -> None:
        """Alias for :meth:`stop`."""
        self.stop()

    # -- snapshots (never block) ------------------------------------------

    def latest(self) -> VisionSnapshot | None:
        """Newest processed frame, or ``None`` until the first one arrives."""
        with self._condition:
            return self._latest

    def wait_ready(self, timeout: float | None = None) -> VisionSnapshot:
        """Block until a snapshot exists.  For scripts and tests, not the GUI."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._latest is not None
                or self._worker_finished.is_set(),
                timeout,
            )
            snapshot = self._latest

        if snapshot is not None:
            return snapshot
        if self._worker_finished.is_set():
            raise VisionError("Vision worker stopped before producing a snapshot")
        raise TimeoutError(f"No snapshot within {timeout} seconds")

    # -- worker ------------------------------------------------------------

    def _run(self, stop: threading.Event, finished: threading.Event) -> None:
        logger.debug("Vision worker started")
        last_processed_at: float | None = None
        fps_estimate = 0.0
        read_failures = 0
        try:
            while not stop.is_set():
                frame = self._camera.read()
                if frame is None:
                    if stop.is_set():
                        break
                    read_failures += 1
                    self._record_error(CameraError("Camera did not return a frame"))
                    # One dropped frame is normal; only reopen after a burst.
                    if read_failures < _MAX_READ_FAILURES:
                        stop.wait(0.01)
                        continue
                    read_failures = 0
                    self._reconnect(stop)
                    continue
                read_failures = 0

                captured_at = time.monotonic()
                try:
                    detections, inference_seconds = self._detector.detect(frame)
                except Exception as error:  # noqa: BLE001 - keep the loop alive
                    self._record_error(ModelError(f"Inference failed: {error}"))
                    logger.exception("Inference failed")
                    stop.wait(0.05)
                    continue

                processed_at = time.monotonic()
                if last_processed_at is not None:
                    interval = processed_at - last_processed_at
                    if interval > 0:
                        instant = 1.0 / interval
                        fps_estimate = (
                            instant
                            if fps_estimate == 0.0
                            else 0.9 * fps_estimate + 0.1 * instant
                        )
                last_processed_at = processed_at

                best = max(
                    detections,
                    key=lambda detection: detection.confidence,
                    default=None,
                )
                annotated = (
                    draw_detections(frame, detections, best) if self.annotate else None
                )
                snapshot = self._publish(
                    captured_at=captured_at,
                    processed_at=processed_at,
                    frame=frame,
                    annotated=annotated,
                    detections=detections,
                    inference_seconds=inference_seconds,
                    fps=fps_estimate,
                )
                self._note_success()

                if self.show_preview and not stop.is_set():
                    preview_frame = annotated if annotated is not None else frame
                    try:
                        self._show_preview(stop, preview_frame)
                    except Exception:  # noqa: BLE001 - preview cannot kill capture
                        logger.exception("OpenCV preview failed; disabling it")
                        self.show_preview = False

                if self.on_result is not None:
                    try:
                        self.on_result(snapshot)
                    except Exception:  # noqa: BLE001 - callback cannot kill capture
                        logger.exception("Vision on_result callback failed")
        except Exception:  # noqa: BLE001 - a crash here must not pass silently
            logger.exception("Vision worker stopped unexpectedly")
            raise
        finally:
            self._camera.release()
            self._close_preview()
            finished.set()
            with self._condition:
                self._condition.notify_all()
            logger.debug("Vision worker stopped")

    def _publish(
        self,
        *,
        captured_at: float,
        processed_at: float,
        frame: NDArray[np.uint8],
        annotated: NDArray[np.uint8] | None,
        detections: tuple[Detection, ...],
        inference_seconds: float,
        fps: float,
    ) -> VisionSnapshot:
        with self._condition:
            self._sequence += 1
            snapshot = VisionSnapshot(
                sequence=self._sequence,
                captured_at=captured_at,
                processed_at=processed_at,
                frame=frame,
                annotated_frame=annotated,
                detections=detections,
                inference_seconds=inference_seconds,
                fps=fps,
                camera_index=self._camera.index,
            )
            self._latest = snapshot
            self._condition.notify_all()
            return snapshot

    def _reconnect(self, stop: threading.Event) -> None:
        self._camera.release()
        if stop.wait(self.reconnect_interval):
            return
        if stop.is_set():
            return
        try:
            self._camera.open()
        except VisionError as error:
            self._record_error(error)

    def _show_preview(
        self, stop: threading.Event, frame: NDArray[np.uint8]
    ) -> None:
        cv2 = _import_cv2()
        if not self._preview_created:
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
            self._preview_created = True
        cv2.imshow(PREVIEW_WINDOW, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            logger.info("Preview requested shutdown")
            stop.set()

    def _close_preview(self) -> None:
        if not self._preview_created:
            return
        self._preview_created = False
        try:
            cv2 = _import_cv2()
            cv2.destroyWindow(PREVIEW_WINDOW)
            cv2.waitKey(1)
        except Exception:  # noqa: BLE001 - closing a window is best effort
            pass

    def _record_error(self, error: VisionError) -> None:
        with self._condition:
            first_failure = self._last_error is None
            self._last_error = error
            self._condition.notify_all()
        if first_failure:
            logger.warning("Vision failed: %s", error)

    def _note_success(self) -> None:
        with self._condition:
            if self._last_error is None:
                return
            logger.info("Vision recovered")
            self._last_error = None
            self._condition.notify_all()


def _parse_class_names(raw: str | None) -> tuple[str, ...] | None:
    """Parse the ``names`` metadata dict, e.g. ``{0: 'fish'}``."""
    if not raw:
        return None
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None

    if isinstance(value, dict) and value:
        try:
            items = sorted(value.items(), key=lambda item: int(item[0]))
        except (TypeError, ValueError):
            items = list(value.items())
        return tuple(str(name) for _, name in items)
    if isinstance(value, (list, tuple)) and value:
        return tuple(str(name) for name in value)
    return None


def _parse_size(raw: str | None) -> tuple[int, int] | None:
    """Parse the ``imgsz`` metadata value, e.g. ``[640, 640]``."""
    if not raw:
        return None
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None

    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def main() -> None:
    """Run the camera with a preview window; press ``q`` or Ctrl+C to stop."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    with Vision(show_preview=True) as vision:
        try:
            last_sequence = 0
            while vision.is_running:
                snapshot = vision.latest()
                if snapshot is not None and snapshot.sequence != last_sequence:
                    last_sequence = snapshot.sequence
                    best = snapshot.best
                    if best is None:
                        print(f"{snapshot.sequence:5d}  no fish")
                    else:
                        centre_x, centre_y = best.center
                        print(
                            f"{snapshot.sequence:5d}  "
                            f"{len(snapshot.detections)} fish  "
                            f"best=({centre_x:.0f}, {centre_y:.0f})  "
                            f"conf={best.confidence:.2f}  "
                            f"{snapshot.inference_seconds * 1000:.1f} ms"
                        )
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            if vision.last_error is not None:
                print(f"last error: {vision.last_error}")


if __name__ == "__main__":
    main()
