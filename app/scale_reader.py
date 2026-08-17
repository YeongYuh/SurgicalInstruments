import logging
import re
import threading
import time
from collections import deque
from statistics import median
from typing import Optional

from app.scale_sample import ScaleSample, StabilityTracker, constant_sample

logger = logging.getLogger(__name__)


class BaseScaleReader:
    def read_weight(self) -> Optional[float]:
        raise NotImplementedError

    def get_latest_weight(self) -> Optional[float]:
        """Return the most recently cached weight without any I/O or blocking.
        Returns None if no valid weight has been received yet.
        Camera inference uses this to avoid blocking on serial retries."""
        return None

    def get_latest_sample(self) -> ScaleSample:
        """Cached weight *with* timestamp, freshness, and stability.

        Preferred over get_latest_weight(): a bare float cannot tell a caller
        whether the scale is still connected or whether the reading has
        settled, and weight verification needs both.
        """
        return ScaleSample()

    def close(self) -> None:
        pass


class MockScaleReader(BaseScaleReader):
    """Fixed weight, reported as a genuinely fresh and stable sample.

    A constant really is stable, which is what makes mock mode useful for
    exercising the full PASS/FAIL path with no hardware attached.
    """

    def __init__(self, mock_weight: float, stable: bool = True):
        self.mock_weight = mock_weight
        self.stable = stable

    def read_weight(self) -> Optional[float]:
        return self.mock_weight

    def get_latest_weight(self) -> Optional[float]:
        return self.mock_weight

    def get_latest_sample(self) -> ScaleSample:
        sample = constant_sample(self.mock_weight)
        if not self.stable:
            sample.stable = False
            sample.reason = "unstable"
        return sample


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
        zero_threshold: float = 2.0,
        zero_confirm_samples: int = 3,
        filter_window: int = 3,
        transition_threshold: float = 5.0,
        debug: bool = False,
        stable_window_sec: float = 2.5,
        stable_range_grams: float = 1.0,
        stable_min_samples: int = 3,
        max_sample_age_sec: float = 2.0,
        stable_min_coverage_ratio: float = 0.5,
        reconnect_backoff_initial: float = 0.5,
        reconnect_backoff_max: float = 5.0,
        reconnect_log_interval_sec: float = 30.0,
        max_plausible_grams: float = 20000.0,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.retries = retries
        self._ser = None

        # Reconnect pacing.  The background drain calls read_weight() every
        # SCALE_BG_POLL_INTERVAL (100 ms); without this, an unplugged scale
        # means 10 open() syscalls and 10 log lines per second forever.
        self._reconnect_backoff_initial = max(0.0, reconnect_backoff_initial)
        self._reconnect_backoff_max     = max(self._reconnect_backoff_initial,
                                              reconnect_backoff_max)
        self._reconnect_log_interval    = max(0.0, reconnect_log_interval_sec)
        # Anything heavier than this is not a surgical tray, it is a bad frame.
        self._max_plausible_grams = float(max_plausible_grams)
        self._retry_delay  = self._reconnect_backoff_initial
        self._next_retry_at = 0.0    # monotonic deadline; 0 = try immediately
        self._down_since    = None   # monotonic time the current outage began
        self._last_fail_log = 0.0
        self._lock = threading.Lock()        # guards serial I/O (long-held)
        self._cache_lock = threading.Lock()  # guards _latest only (never held during I/O)
        self._latest: Optional[float] = None

        # Zero-rejection debounce + transition-aware median filter
        self._zero_threshold       = zero_threshold
        self._zero_confirm_samples = zero_confirm_samples
        self._transition_threshold = transition_threshold
        self._zero_candidate_count: int = 0          # consecutive near-zero readings seen
        self._raw_window: deque = deque(maxlen=max(1, filter_window))  # for median
        self._scale_debug          = debug

        # Timestamped history of accepted (filtered) values.  Drives freshness
        # and stability; without it a cached value from before an unplug would
        # keep looking like a live measurement forever.
        self._tracker = StabilityTracker(
            window_sec=stable_window_sec,
            range_grams=stable_range_grams,
            min_samples=stable_min_samples,
            max_age_sec=max_sample_age_sec,
            min_coverage_ratio=stable_min_coverage_ratio,
        )
        self._was_connected = False

    # ── connection management ─────────────────────────────────────────

    def _ensure_connected(self) -> bool:
        """Open the port if not already open. Returns True when ready.

        While the device is absent this is called at the background drain rate
        (10 Hz).  Retries are therefore paced by an exponential backoff and the
        failures are logged once per outage plus a heartbeat, so a scale left
        unplugged overnight costs a couple of log lines a minute instead of
        ~36 000 an hour drowning every real error on the box.
        """
        if self._ser is not None and self._ser.is_open:
            return True

        now = time.monotonic()
        if now < self._next_retry_at:
            return False          # still backing off — don't touch the port

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
            # A freshly enumerated CH340 can hold a partial frame from before
            # the unplug.  Parsed as a number that becomes a nonsense weight
            # (observed: 133938 g), so drop whatever is already buffered.
            try:
                ser.reset_input_buffer()
            except Exception:      # noqa: BLE001 - not fatal, the filter copes
                pass
            self._ser = ser
            if self._was_connected:
                # Recovery is logged at WARNING on purpose: it is the line that
                # closes an outage, and the kiosk has no console anyone reads at
                # INFO.  A lost/restored pair must be greppable together.
                outage = "" if self._down_since is None else \
                    " after %.1f s" % (now - self._down_since)
                logger.warning("[ScaleReader] Reconnected to %s at %d baud%s.",
                               self.port, self.baudrate, outage)
            else:
                logger.info("[ScaleReader] Connected to %s at %d baud.",
                            self.port, self.baudrate)
            self._was_connected = True
            self._retry_delay   = self._reconnect_backoff_initial
            self._next_retry_at = 0.0
            self._down_since    = None
            return True
        except Exception as exc:
            first = self._down_since is None
            if first:
                self._down_since = now
            # ENOENT means the device is simply not plugged in; the dialout
            # group hint only makes sense for a permission failure and is
            # actively misleading otherwise.
            hint = ("  (run: sudo usermod -aG dialout $USER  then re-login)"
                    if isinstance(exc, PermissionError) else "")
            if first or (self._reconnect_log_interval > 0
                         and now - self._last_fail_log >= self._reconnect_log_interval):
                logger.warning("[ScaleReader] Cannot open %s: %s%s  "
                               "(retrying, backoff up to %.1f s)",
                               self.port, exc, hint, self._reconnect_backoff_max)
                self._last_fail_log = now
            else:
                logger.debug("[ScaleReader] Still cannot open %s: %s", self.port, exc)
            self._ser = None
            self._next_retry_at = now + self._retry_delay
            self._retry_delay = min(self._retry_delay * 2.0,
                                    self._reconnect_backoff_max)
            return False

    def _drop_connection(self, why: str) -> None:
        """Close and discard a broken handle so the next read reconnects.

        Called inside self._lock.  The sample history is deliberately NOT
        cleared here: the tracker's freshness rule ages the last reading out on
        its own, which is what makes a disconnect visible as fresh=false rather
        than as a sudden "no reading".
        """
        ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception:  # noqa: BLE001 - the handle is already broken
                pass
        # A new outage always gets one immediate retry before the backoff grows,
        # so a momentary glitch recovers on the very next poll.
        self._retry_delay   = self._reconnect_backoff_initial
        self._next_retry_at = 0.0
        self._down_since    = None
        self._last_fail_log = 0.0
        logger.warning("[ScaleReader] Connection dropped (%s) — will reconnect on next read.",
                       why)

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

    def get_latest_weight(self) -> Optional[float]:
        """Non-blocking: return the last valid weight received, or None.
        Never touches the serial port.  Thread-safe."""
        with self._cache_lock:
            return self._latest

    def get_latest_sample(self) -> ScaleSample:
        """Non-blocking: last reading plus age, freshness, and stability.

        Never touches the serial port.  Once the scale stops sending (cable
        pulled, board reset) the reported age grows and fresh flips to False,
        so a stale cached value can never be verified as if it were live.
        """
        return self._tracker.snapshot()

    def _is_plausible(self, value: float) -> bool:
        """Reject readings no surgical instrument tray could produce."""
        if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
            return False
        return abs(value) <= self._max_plausible_grams

    def _apply_sample(self, raw_value: float) -> None:
        """Apply transition-aware zero-rejection debounce and median filter.

        Called inside self._lock — must NOT acquire self._lock again.
        May acquire self._cache_lock briefly to read/write _latest.

        Values outside the plausible range are dropped here rather than
        filtered downstream.  Re-plugging the USB re-powers the scale, and its
        first frame after boot is often garbled — 133938 g and 95588 g were
        both observed on the bench.  The stability gate already stops such a
        reading from producing a verdict, but letting it into the window puts a
        95 kg number on the kiosk and inflates the spread, delaying the point
        at which a real measurement can settle.

        State machine (action per sample):

          near-zero sample + current is non-zero
            → zero_candidate: count consecutive near-zeros; don't touch window
            → confirm_zero_reset (count >= threshold): flush window to [0]

          near-zero sample + current is already near-zero / None
            → accept_near_zero: append to window normally

          non-zero sample + current is None
            → init: clear window, start with this sample

          non-zero sample + current is near-zero (zero → non-zero transition)
            → zero_to_nonzero_reset: flush stale zero window, start with this sample

          non-zero sample + abs(raw - current) >= transition_threshold
            → transition_reset: flush stale same-weight window, start with this sample
            Prevents old 38g samples from mixing with new 50g readings.

          non-zero sample + small change (same stable weight)
            → accept_same_weight: append to window, recompute median

        Key invariant: the rolling window never contains samples from a different
        stable weight than the current one.  Any confirmed transition clears it.
        """
        if not self._is_plausible(raw_value):
            logger.warning("[ScaleReader] Discarding implausible reading %.1f g "
                           "(limit ±%.0f g) — likely a garbled frame.",
                           raw_value, self._max_plausible_grams)
            return

        is_near_zero = abs(raw_value) <= self._zero_threshold

        with self._cache_lock:
            current = self._latest

        if self._scale_debug:
            window_before = list(self._raw_window)

        if is_near_zero:
            # ── Near-zero sample ─────────────────────────────────────────
            if current is not None and abs(current) > self._zero_threshold:
                # Current filtered is non-zero → require consecutive zeros.
                self._zero_candidate_count += 1
                if self._zero_candidate_count >= self._zero_confirm_samples:
                    # Confirmed zero: flush ALL stale non-zero samples from window.
                    self._raw_window.clear()
                    self._raw_window.append(0.0)
                    with self._cache_lock:
                        self._latest = 0.0
                    self._tracker.add(0.0)
                    self._zero_candidate_count = 0
                    if self._scale_debug:
                        logger.debug(
                            "[ScaleReader] action=confirm_zero_reset  raw=%.2f"
                            "  before=%.2f  window_before=%s  window_after=%s  filtered=0.0",
                            raw_value, current, window_before, list(self._raw_window),
                        )
                else:
                    if self._scale_debug:
                        logger.debug(
                            "[ScaleReader] action=zero_candidate %d/%d  raw=%.2f"
                            "  filtered=%.2f  (window unchanged)",
                            self._zero_candidate_count, self._zero_confirm_samples,
                            raw_value, current,
                        )
                return  # don't update window or _latest until confirmed
            else:
                # Already near zero or first reading — accept normally.
                action = "accept_near_zero"
                self._zero_candidate_count = 0
        else:
            # ── Non-zero sample ──────────────────────────────────────────
            # Cancel any pending zero confirmation: a non-zero reading interrupts it.
            self._zero_candidate_count = 0

            if current is None:
                action = "init"
                self._raw_window.clear()
            elif abs(current) <= self._zero_threshold:
                # Transition: zero → non-zero.  Clear stale zero samples.
                action = "zero_to_nonzero_reset"
                self._raw_window.clear()
            elif abs(raw_value - current) >= self._transition_threshold:
                # Transition: significant weight change (e.g. 38g → 50g).
                # Flush the window so old samples cannot dominate the new median.
                action = "transition_reset"
                self._raw_window.clear()
            else:
                action = "accept_same_weight"

        # Common update: append accepted sample and recompute median.
        self._raw_window.append(raw_value)
        filtered = float(median(self._raw_window))
        with self._cache_lock:
            self._latest = filtered
        self._tracker.add(filtered)

        if self._scale_debug:
            logger.debug(
                "[ScaleReader] action=%s  raw=%.2f  before=%s  window_before=%s"
                "  window_after=%s  filtered=%.2f",
                action, raw_value,
                f"{current:.2f}" if current is not None else "None",
                window_before, list(self._raw_window), filtered,
            )

    def read_weight(self) -> Optional[float]:
        """
        Drain up to `retries` buffered serial lines.  Apply zero-rejection
        debouncing and median filtering before updating the cached value.
        Falls back to the cached value (or None) if no new valid line is found.
        Never raises.
        """
        with self._lock:
            if not self._ensure_connected():
                return self._latest

            try:
                for _ in range(self.retries):
                    try:
                        raw = self._ser.readline()
                    except Exception as exc:
                        self._drop_connection("read error: %s" % exc)
                        break

                    if not raw:
                        break  # readline() timed out — buffer is empty

                    line = raw.decode("utf-8", errors="ignore")
                    value = self._parse_line(line)
                    if value is not None:
                        self._apply_sample(value)

            except Exception as exc:
                logger.warning("[ScaleReader] Unexpected error: %s", exc)
                self._drop_connection("unexpected error: %s" % exc)

            return self._latest

    def close(self) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
        self._tracker.reset()


def create_scale_reader(
    mode: str,
    mock_weight: float,
    port: str,
    baudrate: int,
    timeout: float,
    retries: int,
    zero_threshold: float = 2.0,
    zero_confirm_samples: int = 3,
    filter_window: int = 3,
    transition_threshold: float = 5.0,
    debug: bool = False,
    stable_window_sec: float = 2.5,
    stable_range_grams: float = 1.0,
    stable_min_samples: int = 3,
    max_sample_age_sec: float = 2.0,
    stable_min_coverage_ratio: float = 0.5,
    reconnect_backoff_initial: float = 0.5,
    reconnect_backoff_max: float = 5.0,
    reconnect_log_interval_sec: float = 30.0,
    max_plausible_grams: float = 20000.0,
) -> BaseScaleReader:
    if mode == "mock":
        return MockScaleReader(mock_weight=mock_weight)
    if mode == "serial":
        return SerialScaleReader(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            retries=retries,
            zero_threshold=zero_threshold,
            zero_confirm_samples=zero_confirm_samples,
            filter_window=filter_window,
            transition_threshold=transition_threshold,
            debug=debug,
            stable_window_sec=stable_window_sec,
            stable_range_grams=stable_range_grams,
            stable_min_samples=stable_min_samples,
            max_sample_age_sec=max_sample_age_sec,
            stable_min_coverage_ratio=stable_min_coverage_ratio,
            reconnect_backoff_initial=reconnect_backoff_initial,
            reconnect_backoff_max=reconnect_backoff_max,
            reconnect_log_interval_sec=reconnect_log_interval_sec,
            max_plausible_grams=max_plausible_grams,
        )
    raise ValueError(f"Unsupported scale reader mode: '{mode}'. Choose 'mock' or 'serial'.")
