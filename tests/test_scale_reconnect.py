"""Reconnect pacing for the serial scale.

Found by physically unplugging the CH340 on the Jetson: the background drain
polls read_weight() every 100 ms, so an absent device produced ~10 failed
open() calls and ~10 WARNING lines per second — 523 lines in 53 seconds of
real measurement — with no backoff and a "join the dialout group" hint that is
actively wrong when the errno is ENOENT.

These tests pin the pacing without any hardware: a fake pyserial module is
injected so the reader's open() can be made to fail, succeed, or blow up
mid-read on demand.
"""

import logging
import sys
import types

import pytest

from app.scale_reader import SerialScaleReader


class FakeSerial:
    """Minimal stand-in for serial.Serial with scriptable failures."""

    #: set by each test — callable() -> None to open OK, or raises
    open_hook = None
    #: lines readline() should hand back, oldest first
    lines = []
    #: number of open() calls that actually reached the "device"
    open_calls = 0
    reset_calls = 0

    def __init__(self):
        self.port = None
        self.baudrate = None
        self.timeout = None
        self.dtr = None
        self.is_open = False

    def open(self):
        type(self).open_calls += 1
        if type(self).open_hook is not None:
            type(self).open_hook()
        self.is_open = True

    def reset_input_buffer(self):
        type(self).reset_calls += 1

    def readline(self):
        if type(self).lines:
            return type(self).lines.pop(0)
        return b""

    def close(self):
        self.is_open = False


@pytest.fixture
def fake_serial(monkeypatch):
    """Install a fake `serial` module for the duration of one test."""
    FakeSerial.open_hook = None
    FakeSerial.lines = []
    FakeSerial.open_calls = 0
    FakeSerial.reset_calls = 0
    module = types.ModuleType("serial")
    module.Serial = FakeSerial
    module.SerialException = OSError
    monkeypatch.setitem(sys.modules, "serial", module)
    return FakeSerial


def make_reader(**kw):
    kw.setdefault("port", "/dev/ttyUSB-test")
    kw.setdefault("reconnect_backoff_initial", 0.5)
    kw.setdefault("reconnect_backoff_max", 5.0)
    kw.setdefault("reconnect_log_interval_sec", 30.0)
    return SerialScaleReader(**kw)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr("app.scale_reader.time.monotonic", c)
    return c


# ── backoff ──────────────────────────────────────────────────────────────────

def test_absent_device_is_not_hammered_at_the_poll_rate(fake_serial, clock):
    """The original bug: 10 open() syscalls per second, forever."""
    def boom():
        raise OSError(2, "No such file or directory")
    fake_serial.open_hook = boom

    reader = make_reader()
    # Simulate 5 seconds of the 100 ms background drain.
    for _ in range(50):
        reader.read_weight()
        clock.advance(0.1)

    # Unpaced this would be 50.  With 0.5s doubling to a 5s cap the schedule is
    # t=0, 0.5, 1.5, 3.5, and then every 5s — well under ten attempts.
    assert fake_serial.open_calls <= 6, fake_serial.open_calls
    assert fake_serial.open_calls >= 3, "backoff must still keep retrying"


def test_backoff_doubles_up_to_the_cap(fake_serial, clock):
    def boom():
        raise OSError(2, "No such file or directory")
    fake_serial.open_hook = boom

    reader = make_reader(reconnect_backoff_initial=0.5, reconnect_backoff_max=2.0)
    attempts = []
    for _ in range(400):
        before = fake_serial.open_calls
        reader.read_weight()
        if fake_serial.open_calls > before:
            attempts.append(clock.now)
        clock.advance(0.1)

    gaps = [round(b - a, 3) for a, b in zip(attempts, attempts[1:])]
    assert gaps[:3] == [0.5, 1.0, 2.0], gaps[:5]
    # A retry fires on the first 100 ms drain tick at or after the deadline, so
    # an observed gap may run one poll interval past the cap.
    assert all(g <= 2.0 + 0.1 + 1e-6 for g in gaps), gaps


def test_recovery_is_immediate_after_a_drop(fake_serial, clock):
    """A momentary glitch must recover on the very next poll, not after 5 s."""
    reader = make_reader()
    assert reader.read_weight() is None or True
    assert fake_serial.open_calls == 1

    # Blow up mid-read the way a yanked USB does.
    def explode():
        raise OSError("device disconnected")
    fake_serial.lines = []
    reader._ser.readline = lambda: (_ for _ in ()).throw(OSError("disconnected"))
    reader.read_weight()          # triggers _drop_connection
    assert reader._ser is None

    # Next poll, 100 ms later, must retry straight away.
    clock.advance(0.1)
    reader.read_weight()
    assert fake_serial.open_calls == 2


# ── logging volume ───────────────────────────────────────────────────────────

def test_repeated_failures_do_not_flood_the_log(fake_serial, clock, caplog):
    def boom():
        raise OSError(2, "No such file or directory")
    fake_serial.open_hook = boom

    reader = make_reader(reconnect_log_interval_sec=30.0)
    with caplog.at_level(logging.WARNING, logger="app.scale_reader"):
        for _ in range(600):      # 60 seconds of 10 Hz polling
            reader.read_weight()
            clock.advance(0.1)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    # One for the start of the outage, plus ~one heartbeat per 30 s.
    assert len(warnings) <= 4, [r.getMessage() for r in warnings]
    assert len(warnings) >= 1


def test_missing_device_does_not_blame_the_dialout_group(fake_serial, clock, caplog):
    def boom():
        raise OSError(2, "No such file or directory")
    fake_serial.open_hook = boom

    reader = make_reader()
    with caplog.at_level(logging.WARNING, logger="app.scale_reader"):
        reader.read_weight()

    text = " ".join(r.getMessage() for r in caplog.records)
    assert "dialout" not in text, text


def test_permission_error_still_suggests_dialout(fake_serial, clock, caplog):
    def boom():
        raise PermissionError(13, "Permission denied")
    fake_serial.open_hook = boom

    reader = make_reader()
    with caplog.at_level(logging.WARNING, logger="app.scale_reader"):
        reader.read_weight()

    text = " ".join(r.getMessage() for r in caplog.records)
    assert "dialout" in text, text


def test_reconnect_is_logged_where_an_operator_will_see_it(fake_serial, clock, caplog):
    """The kiosk has no console at INFO; a closed outage must be greppable."""
    reader = make_reader()
    reader.read_weight()                      # first connect (INFO)

    reader._ser.readline = lambda: (_ for _ in ()).throw(OSError("disconnected"))
    reader.read_weight()                      # drop
    clock.advance(0.1)

    fake_serial.lines = [b"123.4\n"]
    with caplog.at_level(logging.WARNING, logger="app.scale_reader"):
        reader.read_weight()                  # reconnect

    messages = " ".join(r.getMessage() for r in caplog.records
                        if r.levelno >= logging.WARNING)
    assert "Reconnected" in messages, messages


# ── stale frame on reconnect ─────────────────────────────────────────────────

def test_input_buffer_is_flushed_on_connect(fake_serial, clock):
    """A partial CH340 frame parsed as a weight produced 133938 g in the field."""
    reader = make_reader()
    reader.read_weight()
    assert fake_serial.reset_calls == 1

    reader._ser.readline = lambda: (_ for _ in ()).throw(OSError("disconnected"))
    reader.read_weight()
    clock.advance(0.1)
    reader.read_weight()
    assert fake_serial.reset_calls == 2, "reconnect must also discard stale bytes"


# ── the safety property the whole thing exists for ───────────────────────────

def test_cached_weight_never_stays_fresh_across_a_disconnect(fake_serial, clock,
                                                             monkeypatch):
    """Measured on hardware: fresh=false 2.25 s after the plug left the socket."""
    # StabilityTracker binds time.monotonic as a default argument at import,
    # so patching the module attribute cannot reach it — drive its clock directly.
    wall = FakeClock()
    reader = make_reader(max_sample_age_sec=2.0)
    reader._tracker._monotonic = wall
    reader._tracker._wallclock = wall
    fake_serial.lines = [b"500.0\n"]
    reader.read_weight()
    assert reader.get_latest_sample().fresh is True

    def boom():
        raise OSError(2, "No such file or directory")
    fake_serial.open_hook = boom
    reader._ser.readline = lambda: (_ for _ in ()).throw(OSError("disconnected"))
    reader.read_weight()

    wall.advance(3.0)
    clock.advance(3.0)
    sample = reader.get_latest_sample()
    assert sample.fresh is False
    assert sample.stable is False
    assert sample.value == 500.0, "the last reading may still be displayed"


# ── implausible readings ─────────────────────────────────────────────────────

def test_garbled_frame_after_replug_is_discarded(fake_serial, clock, caplog):
    """Both 133938 g and 95588 g were seen as the first frame after a replug.

    Re-plugging the USB re-powers the scale, so its boot-time output can land
    mid-frame.  reset_input_buffer() cannot catch this — the bytes arrive after
    the port is already open — so the value has to be rejected on its merits.
    """
    reader = make_reader(max_plausible_grams=20000.0)
    fake_serial.lines = [b"500.0\n"]
    reader.read_weight()
    assert reader.get_latest_weight() == 500.0

    with caplog.at_level(logging.WARNING, logger="app.scale_reader"):
        fake_serial.lines = [b"95588\n", b"133938\n"]
        reader.read_weight()

    assert reader.get_latest_weight() == 500.0, "garbage must not become the reading"
    assert "implausible" in " ".join(r.getMessage() for r in caplog.records).lower()


def test_implausible_reading_never_enters_the_stability_window(fake_serial, clock):
    """A 95 kg spike would otherwise inflate the spread and delay settling."""
    reader = make_reader(max_plausible_grams=20000.0)
    for line in (b"500.0\n", b"500.0\n", b"500.0\n"):
        fake_serial.lines = [line]
        reader.read_weight()
    before = reader.get_latest_sample()

    fake_serial.lines = [b"95588\n"]
    reader.read_weight()
    after = reader.get_latest_sample()

    assert after.sample_count == before.sample_count
    assert after.spread == before.spread


def test_a_real_heavy_tray_is_still_accepted(fake_serial, clock):
    """The bound must reject garbage without rejecting a genuinely heavy tray."""
    reader = make_reader(max_plausible_grams=20000.0)
    fake_serial.lines = [b"4820.5\n"]
    reader.read_weight()
    assert reader.get_latest_weight() == 4820.5


# ── stability threshold, set from bench measurement ──────────────────────────

def test_stable_range_has_headroom_over_one_scale_count():
    """Bench: ~1091 g load, 196 samples at 2.06 Hz, worst window spread 1.0 g.

    The scale reports whole grams, so one count of dither is 1.0 g.  The
    threshold must sit strictly above that (otherwise there is no margin) and
    stay well below the transition threshold, which is what a human still
    loading the tray looks like.
    """
    import app.config as config
    assert config.SCALE_STABLE_RANGE_GRAMS > 1.0, "no headroom over one scale count"
    assert config.SCALE_STABLE_RANGE_GRAMS < config.SCALE_TRANSITION_THRESHOLD_GRAMS


def test_one_count_of_dither_still_settles(fake_serial, clock, monkeypatch):
    """1090/1091 alternating — exactly what the bench load did — must settle."""
    wall = FakeClock()
    reader = make_reader(stable_window_sec=2.5, stable_range_grams=2.0,
                         stable_min_samples=3, max_sample_age_sec=2.0)
    reader._tracker._monotonic = wall
    reader._tracker._wallclock = wall

    for i in range(8):
        fake_serial.lines = [b"1091\n" if i % 3 else b"1090\n"]
        reader.read_weight()
        wall.advance(0.485)          # the measured 2.06 Hz cadence

    sample = reader.get_latest_sample()
    assert sample.stable is True, (sample.reason, sample.spread, sample.sample_count)
    assert sample.spread <= 1.0
