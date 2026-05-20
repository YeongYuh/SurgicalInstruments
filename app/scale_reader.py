import logging
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class BaseScaleReader:
    def read_weight(self) -> Optional[float]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MockScaleReader(BaseScaleReader):
    def __init__(self, mock_weight: float):
        self.mock_weight = mock_weight

    def read_weight(self) -> Optional[float]:
        return self.mock_weight


class SerialScaleReader(BaseScaleReader):
    """
    Reads weight from a USB serial device (e.g. Arduino + load cell).

    The port is opened once and kept open between calls.  This prevents
    the Arduino from resetting on every read (DTR toggle), which avoids
    re-reading the startup debug messages on each call.

    Each read_weight() drains up to `retries` buffered lines and returns
    the most recent valid numeric value found.  If no new valid data
    arrives, the last cached value is returned.  Returns None only when
    no valid weight has ever been received.

    Valid line formats:
        "50"          -> 50.0    (bare integer)
        "50.2"        -> 50.2   (bare decimal)
        "-1.5"        -> -1.5   (signed)
        "WEIGHT:50.2" -> 50.2   (key:value, for future firmware variants)

    Silently ignored (logged at DEBUG):
        ""                         (empty)
        "Initializing the scale"   (no digits)
        "Before setting up the scale:"  (no digits after colon)
        any other non-numeric text
    """

    # Entire stripped line is a number
    _BARE_NUMBER = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
    # "LABEL: 50.2" or "WEIGHT:50.2" — label, colon, optional space, number
    _KV_NUMBER   = re.compile(r"^[A-Za-z_][\w\s]*:\s*([+-]?\d+(?:\.\d+)?)$")

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: float = 2.0,
        retries: int = 5,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.retries = retries
        self._ser = None
        self._lock = threading.Lock()
        self._latest: Optional[float] = None

    # ── connection management ─────────────────────────────────────────

    def _ensure_connected(self) -> bool:
        """Open the port if not already open. Returns True when ready."""
        if self._ser is not None and self._ser.is_open:
            return True
        try:
            import serial as _serial
            # Configure before opening so dtr=False is set as early as possible.
            # On boards with a reset-on-DTR circuit (e.g. Arduino Uno) this
            # may prevent the Arduino from resetting when the port opens.
            # It is not guaranteed on all platforms; if it doesn't help on your
            # board the startup messages will simply be skipped gracefully.
            ser = _serial.Serial()
            ser.port = self.port
            ser.baudrate = self.baudrate
            ser.timeout = self.timeout
            ser.dtr = False
            ser.open()
            self._ser = ser
            logger.info("[ScaleReader] Connected to %s at %d baud.", self.port, self.baudrate)
            return True
        except Exception as exc:
            logger.warning(
                "[ScaleReader] Cannot open %s: %s  "
                "(run: sudo usermod -aG dialout $USER  then re-login)",
                self.port, exc,
            )
            self._ser = None
            return False

    # ── line parser ───────────────────────────────────────────────────

    @classmethod
    def _parse_line(cls, raw_line: str) -> Optional[float]:
        """
        Return a float if the line encodes a valid weight, else None.
        Empty lines are silently ignored; non-numeric text is DEBUG-logged.
        """
        line = raw_line.strip()
        if not line:
            return None

        # Bare number: "50", "50.2", "-1.5"
        if cls._BARE_NUMBER.match(line):
            try:
                return float(line)
            except ValueError:
                pass

        # Key:value: "WEIGHT:50.2"
        m = cls._KV_NUMBER.match(line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

        # Everything else (debug text, garbage) — debug only, never error
        logger.debug("[ScaleReader] Skipping non-numeric line: %r", line)
        return None

    # ── public API ────────────────────────────────────────────────────

    def read_weight(self) -> Optional[float]:
        """
        Drain up to `retries` buffered serial lines.  Update and return
        the most recent valid weight.  Falls back to the cached value
        (or None) if no new valid line is found.  Never raises.
        """
        with self._lock:
            if not self._ensure_connected():
                return self._latest  # return cached value while disconnected

            try:
                for _ in range(self.retries):
                    try:
                        raw = self._ser.readline()
                    except Exception as exc:
                        logger.warning(
                            "[ScaleReader] Read error: %s — will reconnect on next call.", exc
                        )
                        self._ser = None
                        break

                    if not raw:
                        break  # readline() timed out — buffer is empty

                    line = raw.decode("utf-8", errors="ignore")
                    value = self._parse_line(line)
                    if value is not None:
                        self._latest = value  # always take the most recent valid value

            except Exception as exc:
                logger.warning("[ScaleReader] Unexpected error: %s", exc)

            return self._latest

    def close(self) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None


def create_scale_reader(
    mode: str,
    mock_weight: float,
    port: str,
    baudrate: int,
    timeout: float,
    retries: int,
) -> BaseScaleReader:
    if mode == "mock":
        return MockScaleReader(mock_weight=mock_weight)
    if mode == "serial":
        return SerialScaleReader(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            retries=retries,
        )
    raise ValueError(f"Unsupported scale reader mode: '{mode}'. Choose 'mock' or 'serial'.")
