"""ModelManager: activation, safe switching, per-package isolation, generations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.inference.manager import ModelManagerError
from conftest import make_manager, write_package, write_raw_manifest


def _packages_dir(manager):
    return manager.packages_dir


# ── 6. load package A ───────────────────────────────────────────────────────

def test_activate_loads_package_and_profile(tmp_path):
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "ortho",
                  class_weights={"widget": 10.0}, standards={"widget": 2})

    state = manager.activate("ortho")

    assert state.ready
    assert state.package_id == "ortho"
    assert state.adapter.loaded
    assert state.standards == {"widget": 2}
    assert state.class_weights == {"widget": 10.0}
    assert manager.generation == 1


def test_activate_seeds_profile_on_disk(tmp_path):
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "ortho", standards={"widget": 4})

    manager.activate("ortho")

    profile_file = tmp_path / "profiles" / "ortho" / "standards.json"
    assert profile_file.exists()
    assert json.loads(profile_file.read_text(encoding="utf-8")) == {"widget": 4}


def test_activate_unknown_package_raises(tmp_path):
    manager = make_manager(tmp_path)
    with pytest.raises(ModelManagerError) as excinfo:
        manager.activate("nope")
    assert "not found" in str(excinfo.value)


def test_template_package_cannot_be_activated(tmp_path):
    manager = make_manager(tmp_path)
    write_raw_manifest(_packages_dir(manager), "demo", {
        "schema_version": 1, "id": "demo", "display_name": "Demo",
        "adapter": "fake", "template": True, "model_file": "m.bin",
        "inventory": {"class_weights": "cw.json"},
    })
    with pytest.raises(ModelManagerError) as excinfo:
        manager.activate("demo")
    assert "template" in str(excinfo.value)


def test_invalid_package_cannot_be_activated(tmp_path):
    manager = make_manager(tmp_path)
    write_raw_manifest(_packages_dir(manager), "broken", {"schema_version": 1, "id": "broken"})
    with pytest.raises(ModelManagerError) as excinfo:
        manager.activate("broken")
    assert "invalid" in str(excinfo.value)


def test_activating_the_active_package_is_a_no_op(tmp_path):
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "ortho")
    first = manager.activate("ortho")
    adapter = first.adapter

    second = manager.activate("ortho")

    assert second.adapter is adapter          # not reloaded
    assert manager.generation == 1            # generation unchanged


# ── 7. switch A -> B ────────────────────────────────────────────────────────

def test_switch_replaces_package_profile_and_adapter(tmp_path):
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "ortho",
                  class_weights={"widget": 10.0}, standards={"widget": 2})
    write_package(_packages_dir(manager), "obgyn",
                  class_weights={"forceps": 7.5}, standards={"forceps": 5})

    manager.activate("ortho")
    old_adapter = manager.active_adapter

    state = manager.activate("obgyn")

    assert state.package_id == "obgyn"
    assert state.standards == {"forceps": 5}
    assert state.class_weights == {"forceps": 7.5}
    assert manager.generation == 2
    # Jetson Nano has 4 GB: exactly one model stays resident.
    assert old_adapter.loaded is False
    assert old_adapter.unload_calls == 1
    assert manager.active_adapter.loaded is True


# ── 8. a failed switch must not break the working package ───────────────────

def test_failed_switch_keeps_previous_package_serving(tmp_path):
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "ortho", standards={"widget": 2})
    write_package(_packages_dir(manager), "broken_load",
                  adapter_options={"fail_load": True})

    manager.activate("ortho")
    good_adapter = manager.active_adapter
    generation_before = manager.generation

    with pytest.raises(ModelManagerError) as excinfo:
        manager.activate("broken_load")
    assert "failed to activate package" in str(excinfo.value)

    state = manager.state()
    assert state.package_id == "ortho"           # still the old package
    assert state.adapter is good_adapter
    assert state.adapter.loaded is True          # and still usable
    assert state.standards == {"widget": 2}
    assert manager.generation == generation_before   # no half-switch
    assert manager.infer(object()).counts        # inference still works


def test_failed_switch_to_missing_package_keeps_previous(tmp_path):
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "ortho")
    manager.activate("ortho")

    with pytest.raises(ModelManagerError):
        manager.activate("does_not_exist")

    assert manager.state().package_id == "ortho"
    assert manager.generation == 1


# ── 9. standards are isolated per package ───────────────────────────────────

def test_standards_edits_do_not_leak_between_packages(tmp_path):
    """Orthopaedic standards must never end up applied to an obstetric tray."""
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "ortho",
                  class_weights={"widget": 10.0}, standards={"widget": 2})
    write_package(_packages_dir(manager), "obgyn",
                  class_weights={"forceps": 7.5}, standards={"forceps": 5})

    manager.activate("ortho")
    manager.active_profile.update_standards({"widget": 99})
    assert manager.state().standards == {"widget": 99}

    manager.activate("obgyn")
    assert manager.state().standards == {"forceps": 5}      # untouched by ortho's edit
    manager.active_profile.update_standards({"forceps": 1})

    manager.activate("ortho")
    assert manager.state().standards == {"widget": 99}      # ortho's edit survived

    manager.activate("obgyn")
    assert manager.state().standards == {"forceps": 1}


def test_package_default_standards_stay_read_only(tmp_path):
    manager = make_manager(tmp_path)
    root = write_package(_packages_dir(manager), "ortho", standards={"widget": 2})

    manager.activate("ortho")
    manager.active_profile.update_standards({"widget": 42})

    factory = json.loads((root / "standards.json").read_text(encoding="utf-8"))
    assert factory == {"widget": 2}      # the package's factory default is intact
    profile = json.loads(
        (tmp_path / "profiles" / "ortho" / "standards.json").read_text(encoding="utf-8"))
    assert profile == {"widget": 42}


def test_legacy_global_standards_seed_only_the_legacy_package(tmp_path):
    """An in-service unit keeps its site configuration across the upgrade."""
    legacy = tmp_path / "output" / "standards.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({"widget": 7}), encoding="utf-8")

    manager = make_manager(tmp_path, legacy_package_id="ortho", legacy_standards_path=legacy)
    write_package(_packages_dir(manager), "ortho", standards={"widget": 2})
    write_package(_packages_dir(manager), "obgyn",
                  class_weights={"forceps": 1.0}, standards={"forceps": 3})

    manager.activate("ortho")
    assert manager.state().standards == {"widget": 7}     # live site config wins

    manager.activate("obgyn")
    assert manager.state().standards == {"forceps": 3}    # legacy did not leak


# ── 10. class weights are per package ───────────────────────────────────────

def test_class_weights_come_from_the_active_package(tmp_path):
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "ortho", class_weights={"widget": 10.0},
                  standards={"widget": 1})
    write_package(_packages_dir(manager), "obgyn", class_weights={"forceps": 7.5},
                  standards={"forceps": 1})

    manager.activate("ortho")
    assert manager.state().class_weights == {"widget": 10.0}

    manager.activate("obgyn")
    assert manager.state().class_weights == {"forceps": 7.5}
    assert "widget" not in manager.state().class_weights


def test_class_weights_are_not_editable_through_the_profile(tmp_path):
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "ortho", class_weights={"widget": 10.0},
                  standards={"widget": 1})
    manager.activate("ortho")

    weights = manager.state().class_weights
    weights["widget"] = 999.0                 # mutating the copy must not stick

    assert manager.state().class_weights == {"widget": 10.0}


# ── 11. generation ──────────────────────────────────────────────────────────

def test_generation_increments_only_on_successful_activation(tmp_path):
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "a")
    write_package(_packages_dir(manager), "b")
    write_package(_packages_dir(manager), "bad", adapter_options={"fail_load": True})

    assert manager.generation == 0
    manager.activate("a")
    assert manager.generation == 1
    manager.activate("b")
    assert manager.generation == 2
    with pytest.raises(ModelManagerError):
        manager.activate("bad")
    assert manager.generation == 2       # failure does not advance the generation


def test_infer_stamps_the_generation_it_ran_under(tmp_path):
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "a")
    manager.activate("a")

    result = manager.infer(object())

    assert result.metadata["generation"] == 1
    assert result.metadata["package_id"] == "a"


# ── 12. stale results are detectable and discarded ──────────────────────────

def test_result_from_a_superseded_package_is_detected_as_stale(tmp_path):
    """The exact race a camera worker hits: infer under A, switch to B, publish.

    The worker holds generation N from before the switch; the manager is now at
    N+1, so the result must be discarded rather than written into B's state.
    """
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "ortho",
                  class_weights={"widget": 10.0}, standards={"widget": 2},
                  adapter_options={"counts": {"widget": 2}})
    write_package(_packages_dir(manager), "obgyn",
                  class_weights={"forceps": 7.5}, standards={"forceps": 5},
                  adapter_options={"counts": {"forceps": 5}})

    manager.activate("ortho")
    worker_state = manager.state()                  # worker snapshots the package
    result = manager.infer(object())                # ...and runs inference
    assert result.counts == {"widget": 2}

    manager.activate("obgyn")                       # user switches mid-inference

    # The worker's publish-time check:
    assert manager.generation != worker_state.generation      # -> discard

    # And nothing from ortho is visible under obgyn.
    assert manager.state().standards == {"forceps": 5}
    assert manager.state().class_weights == {"forceps": 7.5}


def test_snapshot_pairs_package_profile_and_adapter_atomically(tmp_path):
    """A snapshot never mixes one package's model with another's standards."""
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "ortho",
                  class_weights={"widget": 10.0}, standards={"widget": 2})
    write_package(_packages_dir(manager), "obgyn",
                  class_weights={"forceps": 7.5}, standards={"forceps": 5})

    manager.activate("ortho")
    snapshot = manager.state()

    manager.activate("obgyn")

    # The old snapshot still describes ortho consistently — model, standards and
    # weights all from the same activation.
    assert snapshot.package_id == "ortho"
    assert snapshot.standards == {"widget": 2}
    assert snapshot.class_weights == {"widget": 10.0}
    assert snapshot.generation == 1


# ── listing / introspection ─────────────────────────────────────────────────

def test_list_packages_marks_active_valid_and_activatable(tmp_path):
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "ortho")
    write_raw_manifest(_packages_dir(manager), "demo", {
        "schema_version": 1, "id": "demo", "display_name": "Demo",
        "adapter": "fake", "template": True,
    })
    write_raw_manifest(_packages_dir(manager), "broken", {"schema_version": 1})
    manager.activate("ortho")

    listing = {entry["id"]: entry for entry in manager.list_packages()}

    assert listing["ortho"]["active"] is True
    assert listing["ortho"]["activatable"] is True
    assert listing["demo"]["template"] is True
    assert listing["demo"]["activatable"] is False
    assert listing["broken"]["valid"] is False
    assert listing["broken"]["error"]


def test_infer_without_an_active_package_raises(tmp_path):
    manager = make_manager(tmp_path)
    with pytest.raises(ModelManagerError) as excinfo:
        manager.infer(object())
    assert "no active model package" in str(excinfo.value)


def test_model_info_reports_identity_and_generation(tmp_path):
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "ortho", display_name="骨科")
    manager.activate("ortho")

    info = manager.model_info()

    assert info["active"] is True
    assert info["package_id"] == "ortho"
    assert info["display_name"] == "骨科"
    assert info["adapter"] == "fake"
    assert info["generation"] == 1
    assert info["loaded"] is True


def test_bootstrap_publishes_synchronously_then_loads(tmp_path):
    manager = make_manager(tmp_path)
    write_package(_packages_dir(manager), "ortho", standards={"widget": 2})

    state = manager.bootstrap("ortho", background=False)

    assert state is not None
    assert state.package_id == "ortho"
    assert state.standards == {"widget": 2}
    assert manager.active_adapter.loaded is True


def test_bootstrap_of_an_unusable_package_returns_none(tmp_path):
    manager = make_manager(tmp_path)
    assert manager.bootstrap("missing", background=False) is None
    assert manager.state().ready is False
    assert manager.last_error
