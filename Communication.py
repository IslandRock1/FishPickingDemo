"""TMSVR communication with a TM collaborative robot.

This module provides :class:`RobotClient`, a reusable client for the TMSVR
protocol used by TM/Omron robots. It handles packet framing (length prefix and
XOR checksum), sending commands, and parsing responses.

Example (blocking, for scripts and tests)::

    from Communication import RobotClient

    with RobotClient("158.38.140.99", 5891) as robot:
        robot.write("g_testInt", 200)
        response = robot.read("g_testFloat")
        print(response.as_float())

The TCP connection is opened lazily on the first command and can be reused for
many commands. Use the client as a context manager to ensure the socket is
always closed.

Example (non-blocking, for a GUI at 60 fps)::

    from Communication import ThreadedRobotClient

    with ThreadedRobotClient("158.38.140.99", 5891) as robot:
        robot.write("g_velocity", 0.2)          # queued, returns immediately
        robot.request("g_poseX", "g_robotStatus")

        # ... in your GUI update loop, every frame:
        if robot.is_ready("g_poseX"):
            x = robot.get_float("g_poseX")
        if robot.last_error is not None:
            show_disconnected_banner()

Threading
---------

A single :class:`RobotClient` may be shared between threads: :meth:`send`,
:meth:`write` and :meth:`read` are serialized by a lock, so concurrent callers
never interleave requests or corrupt the receive buffer. Each call blocks for
up to ``timeout`` seconds, so it must not run on a GUI thread that needs to
keep a steady frame rate.

:class:`ThreadedRobotClient` wraps it in a background worker and exposes a
non-blocking API: :meth:`~ThreadedRobotClient.write` only queues a command,
:meth:`~ThreadedRobotClient.request` registers an item for polling, and
:meth:`~ThreadedRobotClient.get` / :meth:`~ThreadedRobotClient.is_ready` are
plain lookups into a dict guarded by a short lock. None of them touch the
network, so they are safe to call from the GUI thread every frame.

:meth:`RobotClient.close` deliberately does not take the send lock: calling it
from another thread is a supported way to abort a blocked call (the worker uses
this when shutting down). A failed or timed-out command closes the socket, so
the next command reconnects and starts from a clean protocol state instead of
reusing a desynchronized stream.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_TIMEOUT",
    "MAX_FRAME_SIZE",
    "MODE_READ",
    "MODE_WRITE",
    "RobotClient",
    "RobotCommunicationError",
    "RobotConnectionError",
    "RobotProtocolError",
    "RobotResponse",
    "ThreadedRobotClient",
    "build_packet",
    "parse_response",
]

logger = logging.getLogger(__name__)

# Defaults can be overridden per instance or through the environment, which
# keeps the lab address out of call sites that only pass a host.
DEFAULT_HOST = os.environ.get("TM_ROBOT_HOST", "158.38.140.99")
DEFAULT_PORT = int(os.environ.get("TM_ROBOT_PORT", "5891"))
DEFAULT_TIMEOUT = float(os.environ.get("TM_ROBOT_TIMEOUT", "3.0"))

PROTOCOL_NAME = "TMSVR"
ENCODING = "utf-8"
RECV_SIZE = 4096
FRAME_TERMINATOR = b"\r\n"
MAX_FRAME_SIZE = 64 * 1024

# Transaction IDs are echoed by the robot and used to match a response to the
# request that produced it. Auto-generated IDs cycle through "T1".."T255".
TRANSACTION_ID_PREFIX = "T"
TRANSACTION_ID_MAX = 255

# Modes used in the data section of a TMSVR packet (see the TMFlow
# communication manual, "TMSVR" / listen node section).
MODE_WRITE = 2
MODE_READ = 12

# Characters that would corrupt the comma separated frame layout.
_FORBIDDEN_FIELD_CHARS = (",", "*", "\r", "\n")
_HEX_DIGITS = b"0123456789abcdefABCDEF"
_TRUE_VALUES = frozenset({"1", "true", "on", "yes"})
_FALSE_VALUES = frozenset({"0", "false", "off", "no"})


class RobotCommunicationError(Exception):
    """Base exception for communication with the robot."""


class RobotConnectionError(RobotCommunicationError):
    """Raised when a connection cannot be opened or is unexpectedly closed."""


class RobotProtocolError(RobotCommunicationError):
    """Raised when a response packet is malformed or unexpected."""


@dataclass(frozen=True)
class RobotResponse:
    """A parsed TMSVR response packet."""

    transaction_id: str
    mode: int
    payload: str
    raw: str

    @property
    def name(self) -> str | None:
        """Item name from a ``name=value`` payload, if present."""
        name, separator, _ = self.payload.partition("=")
        return name if separator else None

    @property
    def value(self) -> str | None:
        """Item value from a ``name=value`` payload, if present."""
        _, separator, value = self.payload.partition("=")
        return value if separator else None

    def as_float(self) -> float:
        """Return the value as a float, raising if it is missing or malformed."""
        text = self._require_value()
        try:
            return float(text)
        except ValueError as error:
            raise RobotProtocolError(
                f"Cannot interpret {text!r} as a float"
            ) from error

    def as_int(self) -> int:
        """Return the value as an int, accepting integral text such as ``"200.0"``."""
        text = self._require_value()
        try:
            return int(text)
        except ValueError:
            pass
        try:
            number = float(text)
        except ValueError as error:
            raise RobotProtocolError(
                f"Cannot interpret {text!r} as an integer"
            ) from error
        if not number.is_integer():
            raise RobotProtocolError(f"Cannot interpret {text!r} as an integer")
        return int(number)

    def as_bool(self) -> bool:
        """Return the value as a bool, accepting ``1/0``, ``true/false`` and numbers."""
        text = self._require_value().strip().lower()
        if text in _TRUE_VALUES:
            return True
        if text in _FALSE_VALUES:
            return False
        try:
            return float(text) != 0.0
        except ValueError as error:
            raise RobotProtocolError(
                f"Cannot interpret {text!r} as a boolean"
            ) from error

    def _require_value(self) -> str:
        value = self.value
        if value is None:
            raise RobotProtocolError(
                f"Response payload {self.payload!r} does not contain a value"
            )
        return value

    def __str__(self) -> str:
        return self.raw


def _xor_checksum(data: bytes) -> int:
    checksum = 0
    for byte in data:
        checksum ^= byte
    return checksum


def _format_value(value: Any) -> str:
    """Format a Python value the way the robot expects to receive it."""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _validate_field(value: Any, field: str, *, allow_empty: bool) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string, got {type(value).__name__}")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    for character in _FORBIDDEN_FIELD_CHARS:
        if character in value:
            raise ValueError(
                f"{field} must not contain {character!r}: {value!r}"
            )
    return value


def _decode_text(value: bytes, field: str) -> str:
    try:
        return value.decode(ENCODING)
    except UnicodeDecodeError as error:
        raise RobotProtocolError(
            f"{field} is not valid {ENCODING}: {value!r}"
        ) from error


def build_packet(transaction_id: str, mode: int, payload: str) -> bytes:
    """Build a complete TMSVR packet.

    Frame layout::

        $TMSVR,<length>,<transaction_id>,<mode>,<payload>,*<checksum>\\r\\n

    ``transaction_id`` and ``payload`` must not contain ``','``, ``'*'`` or
    line breaks, since those characters delimit the frame.

    Raises:
        ValueError: if a field would corrupt the frame layout.
    """
    _validate_field(transaction_id, "transaction_id", allow_empty=False)
    _validate_field(payload, "payload", allow_empty=True)
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise ValueError(f"mode must be an int, got {type(mode).__name__}")
    if mode <= 0:
        raise ValueError(f"mode must be positive, got {mode}")

    data = f"{transaction_id},{mode},{payload}"
    length = len(data.encode(ENCODING))
    body = f"{PROTOCOL_NAME},{length},{data},"
    checksum = _xor_checksum(body.encode(ENCODING))
    return f"${body}*{checksum:02X}\r\n".encode(ENCODING)


def parse_response(frame: bytes) -> RobotResponse:
    """Parse and validate a TMSVR response packet.

    ``frame`` is the raw frame including ``$`` and the trailing ``*<checksum>``.
    The checksum is verified on the received bytes, before any decoding, so a
    corrupted or non-UTF-8 frame is reported as a checksum error rather than a
    phantom length error.
    """
    if not isinstance(frame, (bytes, bytearray)):
        raise TypeError(f"frame must be bytes, got {type(frame).__name__}")

    message = bytes(frame).strip()
    if not message.startswith(b"$"):
        raise RobotProtocolError(
            f"Response is missing '$' prefix: {message!r}"
        )
    if b"*" not in message:
        raise RobotProtocolError(f"Response is missing checksum: {message!r}")

    body, _, checksum_text = message[1:].rpartition(b"*")
    if len(checksum_text) != 2 or any(
        byte not in _HEX_DIGITS for byte in checksum_text
    ):
        raise RobotProtocolError(
            f"Invalid checksum {checksum_text!r} in response"
        )
    received_checksum = int(checksum_text, 16)

    expected_checksum = _xor_checksum(body)
    if received_checksum != expected_checksum:
        raise RobotProtocolError(
            "Checksum mismatch: "
            f"received {received_checksum:02X}, expected {expected_checksum:02X}"
        )

    parts = body.split(b",", 2)
    if len(parts) != 3:
        raise RobotProtocolError(f"Malformed response header: {message!r}")

    protocol, length_text, data = parts
    if protocol != PROTOCOL_NAME.encode(ENCODING):
        raise RobotProtocolError(f"Unexpected protocol {protocol!r}")
    if not data.endswith(b","):
        raise RobotProtocolError(
            f"Malformed response data section: {message!r}"
        )

    data = data[:-1]
    try:
        length = int(length_text)
    except ValueError as error:
        raise RobotProtocolError(
            f"Invalid length {length_text!r} in response"
        ) from error

    if len(data) != length:
        raise RobotProtocolError(
            f"Length mismatch: header says {length}, data is {len(data)} bytes"
        )

    data_parts = data.split(b",", 2)
    if len(data_parts) != 3:
        raise RobotProtocolError(f"Malformed response data: {data!r}")

    transaction_id, mode_text, payload = data_parts
    try:
        mode = int(mode_text)
    except ValueError as error:
        raise RobotProtocolError(
            f"Invalid mode {mode_text!r} in response"
        ) from error

    return RobotResponse(
        transaction_id=_decode_text(transaction_id, "transaction ID"),
        mode=mode,
        payload=_decode_text(payload, "payload"),
        raw=_decode_text(message, "response frame"),
    )


class RobotClient:
    """Client for sending commands to a robot using the TMSVR protocol.

    Instances are safe to share between threads; see the module docstring.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: socket.socket | None = None
        self._buffer: bytearray = bytearray()
        self._lock = threading.Lock()
        self._next_transaction_number = 1

    def __repr__(self) -> str:
        state = "connected" if self.is_connected else "disconnected"
        return f"{type(self).__name__}({self.host!r}, {self.port!r}, {state})"

    def __enter__(self) -> "RobotClient":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def is_connected(self) -> bool:
        """Whether a socket is open (not a guarantee that the robot is alive)."""
        return self._socket is not None

    def connect(self) -> None:
        """Open the TCP connection if it is not already open."""
        if self._socket is not None:
            return

        logger.debug("Connecting to %s:%d", self.host, self.port)
        try:
            sock = socket.create_connection(
                (self.host, self.port),
                timeout=self.timeout,
            )
        except OSError as error:
            raise RobotConnectionError(
                f"Could not connect to {self.host}:{self.port}: {error}"
            ) from error

        sock.settimeout(self.timeout)
        self._socket = sock
        self._buffer.clear()
        logger.debug("Connected to %s:%d", self.host, self.port)

    def close(self) -> None:
        """Close the TCP connection and discard any buffered data.

        Safe to call from another thread while a command is blocked; the
        blocked call fails with :class:`RobotConnectionError`.
        """
        sock, self._socket = self._socket, None
        self._buffer.clear()
        if sock is None:
            return

        try:
            sock.close()
        except OSError:
            logger.debug("Error while closing the socket", exc_info=True)

    def send(
        self,
        mode: int,
        payload: str,
        *,
        transaction_id: str | None = None,
    ) -> RobotResponse:
        """Send one command and return the parsed response.

        ``mode`` and ``payload`` form the data section of the TMSVR packet.
        ``transaction_id`` is echoed by the robot and is used to match the
        response to the request; when omitted, a unique ID is generated.
        """
        with self._lock:
            if transaction_id is None:
                transaction_id = self._reserve_transaction_id()
            packet = build_packet(transaction_id, mode, payload)
            sock = self._require_socket()
            logger.debug(
                "TX: %s", packet.decode(ENCODING, errors="replace").rstrip()
            )

            try:
                sock.sendall(packet)
            except OSError as error:
                self.close()
                raise RobotConnectionError(
                    f"Failed to send command to {self.host}:{self.port}: {error}"
                ) from error

            response = parse_response(self._read_frame(sock))
            logger.debug("RX: %s", response.raw)

            if response.transaction_id != transaction_id:
                # The stream is out of phase; reconnect rather than pairing
                # later responses with the wrong requests.
                self.close()
                raise RobotProtocolError(
                    f"Unexpected transaction ID {response.transaction_id!r}; "
                    f"expected {transaction_id!r}"
                )

        return response

    def write(
        self,
        name: str,
        value: str | int | float | bool,
        *,
        transaction_id: str | None = None,
    ) -> RobotResponse:
        """Write a value to a robot item (for example ``g_testInt``).

        Booleans are sent as ``1``/``0`` because the robot does not accept
        Python's ``True``/``False`` spelling.
        """
        return self.send(
            MODE_WRITE,
            f"{_validate_field(name, 'name', allow_empty=False)}="
            f"{_format_value(value)}",
            transaction_id=transaction_id,
        )

    def read(
        self,
        name: str,
        *,
        transaction_id: str | None = None,
    ) -> RobotResponse:
        """Read a value from a robot item (for example ``g_testFloat``)."""
        return self.send(
            MODE_READ,
            _validate_field(name, "name", allow_empty=False),
            transaction_id=transaction_id,
        )

    def _reserve_transaction_id(self) -> str:
        """Return the next transaction ID, cycling 1..255. Caller holds the lock."""
        number = self._next_transaction_number
        self._next_transaction_number = number % TRANSACTION_ID_MAX + 1
        return f"{TRANSACTION_ID_PREFIX}{number}"

    def _require_socket(self) -> socket.socket:
        self.connect()
        if self._socket is None:  # pragma: no cover - connect() guarantees this
            raise RobotConnectionError("Not connected to the robot")
        return self._socket

    def _read_frame(self, sock: socket.socket) -> bytes:
        """Read one complete frame from ``sock``. Caller holds the lock."""
        while FRAME_TERMINATOR not in self._buffer:
            if len(self._buffer) > MAX_FRAME_SIZE:
                self.close()
                raise RobotProtocolError(
                    f"No frame terminator within {MAX_FRAME_SIZE} bytes"
                )
            try:
                chunk = sock.recv(RECV_SIZE)
            except OSError as error:
                self.close()
                raise RobotConnectionError(
                    f"Failed to receive response from {self.host}:{self.port}: "
                    f"{error}"
                ) from error

            if not chunk:
                self.close()
                raise RobotConnectionError(
                    "Connection closed before a complete response was received"
                )

            self._buffer.extend(chunk)

        end = self._buffer.index(FRAME_TERMINATOR) + len(FRAME_TERMINATOR)
        frame = bytes(self._buffer[:end])
        del self._buffer[:end]
        return frame


@dataclass(frozen=True)
class _WriteCommand:
    """A write that has been queued by the caller and not yet sent."""

    name: str
    payload: str
    transaction_id: str | None


@dataclass(frozen=True)
class _Reading:
    """The most recent successful read of one item."""

    response: RobotResponse
    received_at: float


class ThreadedRobotClient:
    """Non-blocking TMSVR client backed by a single background worker thread.

    The worker owns a :class:`RobotClient` and is the only thread that talks to
    the network. Callers never block on I/O::

        with ThreadedRobotClient() as robot:
            robot.request("g_poseX", "g_robotStatus")   # start polling
            robot.write("g_velocity", 0.2)              # fire and forget

            while gui.is_running():                     # 60 fps
                if robot.is_ready("g_poseX"):
                    x = robot.get_float("g_poseX")
                gui.update()

    Writes are coalesced per item: if the same name is written again before the
    worker sends it, only the newest value goes out. That keeps a GUI pushing a
    setpoint at 60 fps from building an unbounded backlog of stale commands.

    Reads are polled in a loop, one round trip per requested item per cycle, at
    roughly ``poll_interval`` seconds between cycles. Values are cached, so
    :meth:`get` returns the newest value the worker has received, or ``None`` if
    nothing has arrived yet. Use :meth:`age` to grey out stale values.

    A command that fails to send is dropped rather than retried, because a late
    setpoint is usually worse than a missing one; check :attr:`last_error` and
    :attr:`pending_writes` to surface problems in the GUI. Connection failures
    never stop the worker: it keeps retrying, and :attr:`last_error` is cleared
    on the first success afterwards.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
        poll_interval: float = 0.05,
        *,
        client: RobotClient | None = None,
    ) -> None:
        self._client = client or RobotClient(host, port, timeout)
        self.poll_interval = poll_interval

        self._condition = threading.Condition()
        self._pending: dict[str, _WriteCommand] = {}
        self._inflight = 0
        self._requested: dict[str, None] = {}
        self._readings: dict[str, _Reading] = {}
        self._last_error: RobotCommunicationError | None = None

        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._wake = threading.Event()
        self._run_id = 0

    def __repr__(self) -> str:
        state = "running" if self.is_running else "stopped"
        return (
            f"{type(self).__name__}({self._client.host!r}, "
            f"{self._client.port!r}, {state})"
        )

    def __enter__(self) -> "ThreadedRobotClient":
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
        """Whether the background worker thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Start the worker thread. Calling this twice is harmless."""
        with self._condition:
            if self.is_running:
                return
            # Each run gets its own stop event, so a worker that is still
            # finishing after a timed-out stop() cannot be revived by a restart.
            stop = threading.Event()
            self._stop = stop
            self._run_id += 1
            self._thread = threading.Thread(
                target=self._run,
                args=(stop, self._run_id),
                name=f"{type(self).__name__}-worker",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float | None = 2.0) -> None:
        """Stop the worker and close the connection.

        Queued writes are **not** flushed first: call :meth:`flush` before
        :meth:`stop` if the last commands must reach the robot. The worker gets
        a short grace period to finish the command it is on; after that the
        socket is closed to abort a blocking read, so shutdown stays responsive.
        Waits at most ``timeout`` seconds in total.
        """
        stop, self._stop = self._stop, None
        if stop is not None:
            stop.set()
        self._wake.set()

        thread, self._thread = self._thread, None
        if thread is None or thread is threading.current_thread():
            self._client.close()
            return

        grace = 0.25 if timeout is None else min(timeout, 0.25)
        thread.join(grace)
        if thread.is_alive():
            self._client.close()
            remaining = None if timeout is None else max(timeout - grace, 0.0)
            thread.join(remaining)
        self._client.close()

    def close(self) -> None:
        """Alias for :meth:`stop`."""
        self.stop()

    # -- writes (never block) ---------------------------------------------

    def write(
        self,
        name: str,
        value: str | int | float | bool,
        *,
        transaction_id: str | None = None,
    ) -> None:
        """Queue a write; the worker sends it as soon as it can.

        Repeated writes to the same ``name`` overwrite each other, so the robot
        receives the newest value rather than a backlog of old ones. The call
        itself does no I/O and returns immediately.
        """
        payload = (
            f"{_validate_field(name, 'name', allow_empty=False)}="
            f"{_format_value(value)}"
        )
        self._ensure_started()
        with self._condition:
            self._pending[name] = _WriteCommand(name, payload, transaction_id)
            self._condition.notify_all()
        self._wake.set()

    @property
    def pending_writes(self) -> int:
        """Number of queued or in-flight write commands."""
        with self._condition:
            return len(self._pending) + self._inflight

    def flush(self, timeout: float | None = None) -> bool:
        """Block until queued writes have been sent.

        Returns ``False`` if the worker stopped or ``timeout`` expired first.
        This does block, so do not call it from the GUI thread.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._pending or self._inflight:
                if not self.is_running:
                    return False
                if deadline is None:
                    self._condition.wait(0.1)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    # -- reads (never block) ----------------------------------------------

    def request(self, *names: str) -> None:
        """Register items to poll continuously. Calling again is harmless."""
        for name in names:
            _validate_field(name, "name", allow_empty=False)
        self._ensure_started()
        with self._condition:
            for name in names:
                self._requested[name] = None
        self._wake.set()

    def forget(self, *names: str) -> None:
        """Stop polling ``names`` and drop their cached values."""
        with self._condition:
            for name in names:
                self._requested.pop(name, None)
                self._readings.pop(name, None)
            self._condition.notify_all()

    @property
    def requested_names(self) -> tuple[str, ...]:
        """Items currently being polled, in polling order."""
        with self._condition:
            return tuple(self._requested)

    def is_ready(self, name: str) -> bool:
        """Whether at least one successful read of ``name`` has arrived."""
        with self._condition:
            return name in self._readings

    def get(self, name: str) -> RobotResponse | None:
        """Newest response for ``name``, or ``None`` if not received yet."""
        reading = self._reading(name)
        return None if reading is None else reading.response

    def get_value(self, name: str) -> str | None:
        """Newest raw value for ``name``, or ``None`` if not received yet."""
        response = self.get(name)
        return None if response is None else response.value

    def get_float(self, name: str, default: float = 0.0) -> float:
        """Newest value of ``name`` as a float.

        Returns ``default`` until a value has arrived; raises
        :class:`RobotProtocolError` if the value cannot be parsed.
        """
        response = self.get(name)
        return default if response is None else response.as_float()

    def get_int(self, name: str, default: int = 0) -> int:
        """Newest value of ``name`` as an int. See :meth:`get_float`."""
        response = self.get(name)
        return default if response is None else response.as_int()

    def get_bool(self, name: str, default: bool = False) -> bool:
        """Newest value of ``name`` as a bool. See :meth:`get_float`."""
        response = self.get(name)
        return default if response is None else response.as_bool()

    def age(self, name: str) -> float | None:
        """Seconds since ``name`` last updated, or ``None`` if never received."""
        reading = self._reading(name)
        if reading is None:
            return None
        return time.monotonic() - reading.received_at

    def wait_ready(
        self,
        name: str,
        timeout: float | None = None,
    ) -> RobotResponse:
        """Block until ``name`` has arrived and return it.

        Intended for scripts and tests, not for the GUI thread.
        """
        with self._condition:
            if name not in self._requested:
                raise ValueError(
                    f"{name!r} is not being polled; call request() first"
                )
            self._condition.wait_for(
                lambda: name in self._readings or not self.is_running,
                timeout,
            )
            reading = self._readings.get(name)

        if reading is not None:
            return reading.response
        if not self.is_running:
            raise RobotConnectionError(
                f"Worker stopped before {name!r} was received"
            )
        raise TimeoutError(f"No value for {name!r} within {timeout} seconds")

    @property
    def last_error(self) -> RobotCommunicationError | None:
        """Most recent communication failure, cleared after a success."""
        with self._condition:
            return self._last_error

    # -- worker ------------------------------------------------------------

    def _ensure_started(self) -> None:
        if not self.is_running:
            self.start()

    def _reading(self, name: str) -> _Reading | None:
        with self._condition:
            return self._readings.get(name)

    def _run(self, stop: threading.Event, run_id: int) -> None:
        logger.debug("Communication worker started")
        try:
            while not stop.is_set():
                try:
                    self._send_pending(stop)
                    self._poll_once(stop)
                except RobotCommunicationError as error:
                    self._record_failure(error)
                    stop.wait(self.poll_interval)
                    continue
                self._wake.wait(self.poll_interval)
                self._wake.clear()
        except Exception:  # noqa: BLE001 - a crash here must not pass silently
            logger.exception("Communication worker stopped unexpectedly")
            raise
        finally:
            # Only close the connection if a later run has not taken it over.
            if self._run_id == run_id:
                self._client.close()
            logger.debug("Communication worker stopped")

    def _send_pending(self, stop: threading.Event) -> None:
        with self._condition:
            if not self._pending:
                return
            commands = list(self._pending.values())
            self._pending.clear()
            self._inflight = len(commands)

        try:
            for command in commands:
                if stop.is_set():
                    break
                self._client.send(
                    MODE_WRITE,
                    command.payload,
                    transaction_id=command.transaction_id,
                )
                self._note_success()
        finally:
            with self._condition:
                self._inflight = 0
                self._condition.notify_all()

    def _poll_once(self, stop: threading.Event) -> None:
        with self._condition:
            names = tuple(self._requested)

        for name in names:
            if stop.is_set():
                return
            response = self._client.read(name)
            with self._condition:
                self._readings[name] = _Reading(response, time.monotonic())
                self._condition.notify_all()
            self._note_success()

    def _note_success(self) -> None:
        with self._condition:
            if self._last_error is None:
                return
            logger.info("Robot communication recovered")
            self._last_error = None
            self._condition.notify_all()

    def _record_failure(self, error: RobotCommunicationError) -> None:
        with self._condition:
            first_failure = self._last_error is None
            self._last_error = error
            self._condition.notify_all()
        if first_failure:
            logger.warning("Robot communication failed: %s", error)


def main() -> None:
    """Example usage that mirrors the original test script."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    value = 200
    try:
        with RobotClient() as robot:
            robot.write("g_testInt", value)
            response = robot.read("g_testFloat")
    except RobotCommunicationError as error:
        logger.error("Communication failed: %s", error)
        raise SystemExit(1) from error

    print(f"Value sent: {value}")
    print(f"Raw response: {response.raw}")
    print(f"Value read: {response.value}")


if __name__ == "__main__":
    main()
