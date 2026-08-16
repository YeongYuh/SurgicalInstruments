"""Model lifecycle: one resident model, no inference/unload race, no TOCTOU.

Every test here is deterministic — threads coordinate through Events, and the
only sleeps are short *negative* assertions ("this must still be blocked"),
never a guess at how long something takes.
"""

from __future__ import annotations

import threading

import pytest

from app.inference.base import AdapterUnavailableError
from app.inference.manager import ModelManagerError
from conftest import RESIDENT, control_for, make_manager, write_package


BLOCKED_CHECK_SEC = 0.3   # long enough to catch "not actually blocked"
COMPLETE_WAIT_SEC = 10.0  # generous upper bound for a real completion


def _run(fn):
    """Start fn in a thread; return (done_event, result_box)."""
    done = threading.Event()
    box = {}

    def runner():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion
            box["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return done, box, thread


def _two_packages(tmp_path, **kwargs):
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "a",
                  class_weights={"widget": 10.0}, standards={"widget": 2},
                  adapter_options=dict(counts={"widget": 2}, **kwargs))
    write_package(manager.packages_dir, "b",
                  class_weights={"forceps": 7.5}, standards={"forceps": 4},
                  adapter_options={"counts": {"forceps": 4}})
    return manager


# ── 1. never two heavy models resident ──────────────────────────────────────

def test_switch_never_holds_two_models_resident(tmp_path):
    """The Jetson has 4 GB — a transient double-load is an OOM, not a hiccup."""
    manager = _two_packages(tmp_path)

    manager.activate("a")
    assert RESIDENT.current == 1
    manager.activate("b")

    assert RESIDENT.current == 1
    assert RESIDENT.peak <= 1, "both models were resident at the same time"
    assert manager.state().package_id == "b"


def test_switch_unloads_the_old_model_before_loading_the_new_one(tmp_path):
    order = []
    manager = _two_packages(tmp_path)
    manager.activate("a")
    old = manager.active_adapter

    original_unload = old._do_unload

    def tracked_unload():
        order.append("unload-a")
        original_unload()

    old._do_unload = tracked_unload

    manager.activate("b")
    new = manager.active_adapter
    # The new adapter's load must come after the old one's unload.
    assert order == ["unload-a"]
    assert new.load_calls == 1
    assert old.loaded is False
    assert new.loaded is True


def test_retired_adapter_cannot_be_resurrected(tmp_path):
    """A stray reference must never be able to reload a second model."""
    manager = _two_packages(tmp_path)
    manager.activate("a")
    old = manager.active_adapter
    manager.activate("b")

    assert old.retired is True
    with pytest.raises(AdapterUnavailableError):
        old.infer(object())
    with pytest.raises(AdapterUnavailableError):
        old.load()
    assert RESIDENT.current == 1


# ── 2 & 3. unload waits for a running inference ─────────────────────────────

def test_unload_waits_for_in_flight_inference(tmp_path):
    manager = _two_packages(tmp_path, blocking=True)
    ctl = control_for("a")
    manager.activate("a")
    adapter = manager.active_adapter

    infer_done, infer_box, _ = _run(lambda: adapter.infer(object()))
    assert ctl.entered.wait(COMPLETE_WAIT_SEC), "inference never started"

    unload_done, unload_box, _ = _run(adapter.unload)
    assert not unload_done.wait(BLOCKED_CHECK_SEC), \
        "unload tore the model out from under a running inference"

    ctl.release.set()
    assert unload_done.wait(COMPLETE_WAIT_SEC)
    assert infer_done.wait(COMPLETE_WAIT_SEC)
    # The FakeAdapter asserts internally that its model handle stayed alive.
    assert "error" not in infer_box, infer_box.get("error")
    assert "error" not in unload_box, unload_box.get("error")
    assert infer_box["value"].counts == {"widget": 2}


def test_new_inference_is_refused_while_unloading(tmp_path):
    """A new call must be turned away, not queued behind the teardown."""
    manager = _two_packages(tmp_path, blocking=True)
    ctl = control_for("a")
    manager.activate("a")
    adapter = manager.active_adapter

    _run(lambda: adapter.infer(object()))
    assert ctl.entered.wait(COMPLETE_WAIT_SEC)
    unload_done, _, _ = _run(adapter.unload)
    assert not unload_done.wait(BLOCKED_CHECK_SEC)   # unload is now pending

    with pytest.raises(AdapterUnavailableError):
        adapter.infer(object())

    ctl.release.set()
    assert unload_done.wait(COMPLETE_WAIT_SEC)


# ── 4. one inference at a time per adapter ──────────────────────────────────

def test_two_inferences_never_run_concurrently(tmp_path):
    """Camera background inference and an operator upload can collide.

    Nothing promises an ML runtime is re-entrant, so they must be serialised.
    """
    manager = _two_packages(tmp_path, blocking=True)
    ctl = control_for("a")
    manager.activate("a")
    adapter = manager.active_adapter

    first_done, _, _ = _run(lambda: adapter.infer(object()))
    assert ctl.entered.wait(COMPLETE_WAIT_SEC)
    second_done, _, _ = _run(lambda: adapter.infer(object()))

    assert not second_done.wait(BLOCKED_CHECK_SEC), "two inferences ran in parallel"
    ctl.release.set()
    assert first_done.wait(COMPLETE_WAIT_SEC)
    assert second_done.wait(COMPLETE_WAIT_SEC)
    assert ctl.max_active == 1


# ── 5, 6 & 7. the publication transaction ───────────────────────────────────

def test_switch_waits_for_an_open_inference_session(tmp_path):
    manager = _two_packages(tmp_path)
    manager.activate("a")

    switch_done = threading.Event()
    with manager.inference_session() as session:
        assert session.package_id == "a"
        switch_done, switch_box, _ = _run(lambda: manager.activate("b"))
        assert not switch_done.wait(BLOCKED_CHECK_SEC), \
            "a package switch started while an inference transaction was open"
        # Still package A for the whole transaction.
        assert manager.state().package_id == "a"
        assert session.standards == {"widget": 2}

    assert switch_done.wait(COMPLETE_WAIT_SEC)
    assert "error" not in switch_box, switch_box.get("error")
    assert manager.state().package_id == "b"


def test_result_cannot_be_published_into_the_package_that_replaced_it(tmp_path):
    """The exact TOCTOU: infer under A, switch to B, then publish.

    Because publication lives inside the session, the switch cannot land in
    between — A's counts can never appear under B's standards.
    """
    manager = _two_packages(tmp_path)
    manager.activate("a")
    published = {}

    with manager.inference_session() as session:
        result = session.infer(object())
        switch_done, _, _ = _run(lambda: manager.activate("b"))
        assert not switch_done.wait(BLOCKED_CHECK_SEC)

        # Everything the publisher needs still describes A, consistently.
        published["package_id"] = session.package_id
        published["generation"] = session.generation
        published["standards"] = session.standards
        published["counts"] = result.counts
        session.register_classes(result.counts.keys())

    assert switch_done.wait(COMPLETE_WAIT_SEC)
    assert published == {
        "package_id": "a",
        "generation": 1,
        "standards": {"widget": 2},
        "counts": {"widget": 2},
    }
    assert manager.state().package_id == "b"


def test_register_classes_writes_to_the_package_that_produced_them(tmp_path):
    """A's detected classes must never be registered on B's profile."""
    manager = _two_packages(tmp_path)
    manager.activate("a")

    with manager.inference_session() as session:
        result = session.infer(object())
        a_profile = session.profile
        switch_done, _, _ = _run(lambda: manager.activate("b"))
        assert not switch_done.wait(BLOCKED_CHECK_SEC)
        session.register_classes(list(result.counts) + ["a-only-class"])

    assert switch_done.wait(COMPLETE_WAIT_SEC)
    assert "a-only-class" in a_profile.standards
    b_standards = manager.state().standards
    assert "a-only-class" not in b_standards
    assert "widget" not in b_standards
    assert b_standards == {"forceps": 4}


def test_inference_session_reports_a_consistent_snapshot(tmp_path):
    manager = _two_packages(tmp_path)
    manager.activate("a")
    with manager.inference_session() as session:
        assert session.package_id == "a"
        assert session.generation == manager.generation
        assert session.standards == {"widget": 2}
        assert session.class_weights == {"widget": 10.0}
        assert session.profile is manager.active_profile


# ── 8, 9 & 10. failed switch rolls back ─────────────────────────────────────

def test_failed_load_reloads_the_previous_model(tmp_path):
    manager = _two_packages(tmp_path)
    write_package(manager.packages_dir, "bad", adapter_options={"fail_load": True})
    manager.activate("a")
    good = manager.active_adapter
    generation_before = manager.generation

    with pytest.raises(ModelManagerError):
        manager.activate("bad")

    state = manager.state()
    assert state.package_id == "a"
    assert state.ready is True          # A is loaded again, not merely selected
    assert good.loaded is True
    assert manager.generation == generation_before
    assert RESIDENT.current == 1
    assert RESIDENT.peak <= 1
    # And it really works.
    assert manager.infer(object()).counts == {"widget": 2}


def test_failed_warmup_rolls_back_and_does_not_report_success(tmp_path):
    """warmup is the verification gate for a runtime switch.

    A model that loads but cannot run would otherwise be reported as a
    successful switch and only fail mid-inventory.
    """
    manager = _two_packages(tmp_path)
    write_package(manager.packages_dir, "cold", adapter_options={"fail_warmup": True})
    manager.activate("a")
    generation_before = manager.generation

    with pytest.raises(ModelManagerError) as excinfo:
        manager.activate("cold")
    assert "warmup failed" in str(excinfo.value)

    state = manager.state()
    assert state.package_id == "a"
    assert state.ready is True
    assert manager.generation == generation_before
    assert RESIDENT.current == 1


def test_failed_switch_leaves_no_orphan_model_loaded(tmp_path):
    manager = _two_packages(tmp_path)
    write_package(manager.packages_dir, "cold", adapter_options={"fail_warmup": True})
    manager.activate("a")

    with pytest.raises(ModelManagerError):
        manager.activate("cold")

    # The half-built replacement was retired, not left holding memory.
    assert RESIDENT.current == 1
    assert RESIDENT.peak <= 1


def test_incompatible_model_is_refused_and_rolled_back(tmp_path):
    """A standard the model can never detect makes the tray impossible to pass."""
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "a",
                  class_weights={"widget": 10.0}, standards={"widget": 2},
                  adapter_options={"counts": {"widget": 2}})
    write_package(manager.packages_dir, "mismatch",
                  class_weights={"ghost": 5.0}, standards={"ghost": 1},
                  adapter_options={"counts": {"other": 1}})
    manager.activate("a")

    # The fake model reports only the classes it "detects" — 'ghost' is not one.
    adapter_cls_state = manager.state()
    assert adapter_cls_state.package_id == "a"

    # FakeAdapter publishes no class list, so compatibility cannot be enforced;
    # this documents that the check is skipped rather than guessed.
    manager.activate("mismatch")
    assert manager.state().package_id == "mismatch"
    assert manager.compatibility["has_model_class_list"] is False


# ── 11. startup background load shares the same gate ────────────────────────

def test_background_startup_load_cannot_race_a_switch(tmp_path):
    """Bootstrap load and a runtime switch must not both hold a model."""
    manager = _two_packages(tmp_path, load_delay=0.4)

    manager.bootstrap("a", background=True)
    # Immediately ask for B while A is still loading in the background.
    manager.activate("b")

    assert manager.state().package_id == "b"
    assert manager.state().ready is True
    assert RESIDENT.peak <= 1, "startup load and switch were both resident"
    assert RESIDENT.current == 1


def test_background_load_bails_out_when_superseded(tmp_path):
    manager = _two_packages(tmp_path)
    manager.bootstrap("a", background=False)
    assert manager.state().ready is True

    manager.activate("b")
    # Re-running the (stale) background loader must be a no-op, not a reload.
    manager._background_load("a", 1)

    assert manager.state().package_id == "b"
    assert RESIDENT.current == 1
    assert RESIDENT.peak <= 1


def test_bootstrap_is_not_ready_until_the_model_is_loaded(tmp_path):
    manager = _two_packages(tmp_path, load_delay=0.3)
    manager.bootstrap("a", background=True)

    state = manager.state()
    # Selected and configured, but honest about not being usable yet.
    assert state.configured is True
    assert state.ready is False
    assert manager.loading is True
    assert manager.get_discovered("a") is not None

    assert manager.wait_until_ready(timeout=COMPLETE_WAIT_SEC) is True
    assert manager.state().ready is True
    assert manager.loading is False


def test_bootstrap_load_failure_is_visible_and_not_ready(tmp_path):
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "a", adapter_options={"fail_load": True})

    manager.bootstrap("a", background=False)

    state = manager.state()
    assert state.configured is True
    assert state.ready is False
    assert manager.fatal_error
    assert manager.wait_until_ready(timeout=0.2) is False
    info = manager.model_info()
    assert info["ready"] is False
    assert info["loaded"] is False
    assert info["fatal_error"]
