"""Startup surgery preset — the tray the unit boots into.

A package profile restores whichever preset was active when the unit was last
shut down.  A theatre that always begins a session on one tray needs the
opposite: a known state on every boot, regardless of where the previous
operator left it.  STARTUP_INVENTORY_PRESET forces that, and must do so before
anything is published — otherwise the kiosk shows the restored tray and then
visibly jumps, and worse, an inference could run against the wrong standards.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import app.config as config
from conftest import make_manager, write_package


SHIPPED_ROOT = Path(config.PROJECT_ROOT) / "model_packages" / "demo"
SHIPPED_PRESETS = SHIPPED_ROOT / "surgery_instruments.json"


def _shipped_presets():
    return json.loads(SHIPPED_PRESETS.read_text(encoding="utf-8"))


def _demo_like_package(tmp_path, package_id="demo"):
    """A package carrying the real shipped presets, on a fake adapter."""
    presets = _shipped_presets()
    classes = sorted({cls for tray in presets.values() for cls in tray})
    packages_dir = tmp_path / "model_packages"
    root = write_package(
        packages_dir, package_id,
        class_weights={cls: 10.0 for cls in classes},
        standards={cls: 0 for cls in classes},
        adapter_options={"counts": {}},
        manifest_overrides={"inventory": {
            "class_weights": "class_weight.json",
            "default_standards": "standards.json",
            "presets": "presets.json",
        }},
    )
    (root / "presets.json").write_text(
        json.dumps(presets, ensure_ascii=False), encoding="utf-8")
    return packages_dir, root


def _preset_file(manager, package_id="demo"):
    return Path(manager.profiles_dir) / package_id / "preset.json"


def _seed_disk_preset(manager, preset_id, package_id="demo"):
    """Pretend the unit was shut down while this preset was selected."""
    path = _preset_file(manager, package_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"active_preset": preset_id}), encoding="utf-8")


# ── 1-2. the shipped defaults ────────────────────────────────────────────────

def test_run_jetson_defaults_to_the_demo_package():
    """Bare ./run_jetson.sh must bring up demo, not ortho_tka."""
    script = (Path(config.PROJECT_ROOT) / "run_jetson.sh").read_text(encoding="utf-8")
    assert 'ACTIVE_MODEL_PACKAGE="${ACTIVE_MODEL_PACKAGE:-demo}"' in script


def test_run_jetson_default_preset_is_surgeryb_and_paired_to_demo():
    """The default preset must be conditional on the default package.

    SurgeryB is a tray of `demo`; forcing it onto ortho_tka would be a
    misconfiguration, so the default may only apply when demo is active.
    """
    script = (Path(config.PROJECT_ROOT) / "run_jetson.sh").read_text(encoding="utf-8")
    assert 'STARTUP_INVENTORY_PRESET="SurgeryB"' in script
    assert '${STARTUP_INVENTORY_PRESET+isset}' in script, \
        "must distinguish unset from explicitly empty"
    block = script[script.index("${STARTUP_INVENTORY_PRESET+isset}"):]
    block = block[:block.index("export STARTUP_INVENTORY_PRESET")]
    assert 'ACTIVE_MODEL_PACKAGE" = "demo"' in block
    assert 'STARTUP_INVENTORY_PRESET=""' in block, "non-demo must get no preset"


def test_env_example_matches_the_shipped_default():
    text = (Path(config.PROJECT_ROOT) / ".env.example").read_text(encoding="utf-8")
    assert "ACTIVE_MODEL_PACKAGE=demo" in text
    assert "STARTUP_INVENTORY_PRESET=SurgeryB" in text


def test_surgeryb_is_a_real_preset_of_the_demo_package():
    """Guards the exact id — Surgent B / Surgen B are not it."""
    presets = _shipped_presets()
    assert "SurgeryB" in presets, sorted(presets)
    assert presets["SurgeryB"] == {
        "Adson-Smooth-Tissue-Forceps": 2,
        "Mayo Scissors Cvd": 2,
        "Towel-Clamp": 1,
    }


# ── 3-5. the persisted preset must lose to the startup preset ────────────────

@pytest.mark.parametrize("stored", ["SurgeryA", "SurgeryC", "SurgeryD"])
def test_startup_preset_overrides_whatever_was_left_on_disk(tmp_path, stored):
    packages_dir, _root = _demo_like_package(tmp_path)
    manager = make_manager(tmp_path)
    manager.packages_dir = packages_dir
    manager.discover(force=True)
    _seed_disk_preset(manager, stored)

    manager.bootstrap("demo", background=False, startup_preset="SurgeryB")

    assert manager.active_profile.active_preset == "SurgeryB"


def test_startup_preset_replaces_standards_with_the_canonical_tray(tmp_path):
    packages_dir, _root = _demo_like_package(tmp_path)
    manager = make_manager(tmp_path)
    manager.packages_dir = packages_dir
    manager.discover(force=True)
    _seed_disk_preset(manager, "SurgeryD")

    manager.bootstrap("demo", background=False, startup_preset="SurgeryB")

    profile = manager.active_profile
    # Canonical form zero-fills every class the package knows, so an instrument
    # belonging to another tray shows up as EXTRA rather than going unnoticed.
    assert profile.standards == profile.canonical_standards_for("SurgeryB")
    non_zero = {k: v for k, v in profile.standards.items() if v}
    assert non_zero == _shipped_presets()["SurgeryB"], \
        "standards must be REPLACED, not merged with SurgeryD's tray"


def test_startup_preset_is_persisted(tmp_path):
    """The forced choice is written through, so the two sources agree."""
    packages_dir, _root = _demo_like_package(tmp_path)
    manager = make_manager(tmp_path)
    manager.packages_dir = packages_dir
    manager.discover(force=True)
    _seed_disk_preset(manager, "SurgeryA")

    manager.bootstrap("demo", background=False, startup_preset="SurgeryB")

    on_disk = json.loads(_preset_file(manager).read_text(encoding="utf-8"))
    assert on_disk["active_preset"] == "SurgeryB"


# ── 6. no extra model work ───────────────────────────────────────────────────

def test_startup_preset_does_not_cost_an_extra_load(tmp_path):
    """A preset changes expectations only; it must not reload the model."""
    packages_dir, _root = _demo_like_package(tmp_path)

    plain = make_manager(tmp_path / "a")
    plain.packages_dir = packages_dir
    plain.discover(force=True)
    plain.bootstrap("demo", background=False)

    forced = make_manager(tmp_path / "b")
    forced.packages_dir = packages_dir
    forced.discover(force=True)
    forced.bootstrap("demo", background=False, startup_preset="SurgeryB")

    assert forced.generation == plain.generation, \
        "forcing a preset must not bump the model generation"
    assert forced.state().ready is True


# ── 7-8. runtime switching still works, and the next boot still resets ───────

def test_operator_can_still_switch_presets_after_startup(tmp_path):
    packages_dir, _root = _demo_like_package(tmp_path)
    manager = make_manager(tmp_path)
    manager.packages_dir = packages_dir
    manager.discover(force=True)
    manager.bootstrap("demo", background=False, startup_preset="SurgeryB")

    generation_before = manager.generation
    manager.apply_preset("SurgeryA")

    profile = manager.active_profile
    assert profile.active_preset == "SurgeryA"
    assert profile.standards == profile.canonical_standards_for("SurgeryA")
    assert {k: v for k, v in profile.standards.items() if v} == \
        _shipped_presets()["SurgeryA"]
    assert manager.generation == generation_before, "preset switch must not reload"


def test_next_boot_returns_to_the_startup_preset(tmp_path):
    """The acceptance case: operator leaves it on SurgeryA, reboot resets it."""
    packages_dir, _root = _demo_like_package(tmp_path)
    manager = make_manager(tmp_path)
    manager.packages_dir = packages_dir
    manager.discover(force=True)
    manager.bootstrap("demo", background=False, startup_preset="SurgeryB")
    manager.apply_preset("SurgeryA")
    assert manager.active_profile.active_preset == "SurgeryA"

    # Same profiles dir — a new process over the same on-disk state.
    reborn = make_manager(tmp_path)
    reborn.packages_dir = packages_dir
    reborn.profiles_dir = manager.profiles_dir
    reborn.discover(force=True)
    reborn.bootstrap("demo", background=False, startup_preset="SurgeryB")

    profile = reborn.active_profile
    assert profile.active_preset == "SurgeryB"
    assert profile.standards == profile.canonical_standards_for("SurgeryB")
    assert {k: v for k, v in profile.standards.items() if v} == \
        _shipped_presets()["SurgeryB"]


def test_empty_startup_preset_keeps_the_restored_one(tmp_path):
    """Explicitly empty means 'don't force' — the old behaviour, on request."""
    packages_dir, _root = _demo_like_package(tmp_path)
    manager = make_manager(tmp_path)
    manager.packages_dir = packages_dir
    manager.discover(force=True)
    _seed_disk_preset(manager, "SurgeryD")

    manager.bootstrap("demo", background=False, startup_preset="")

    assert manager.active_profile.active_preset == "SurgeryD"


# ── 9. a package with no presets must not be handed demo's tray ──────────────

def test_a_package_without_presets_is_not_given_surgeryb(tmp_path, caplog):
    packages_dir = tmp_path / "model_packages"
    write_package(packages_dir, "ortho_tka",
                  class_weights={"patellar-reamer": 12.0},
                  standards={"patellar-reamer": 1},
                  adapter_options={"counts": {}})
    manager = make_manager(tmp_path)
    manager.packages_dir = packages_dir
    manager.discover(force=True)

    with caplog.at_level(logging.ERROR, logger="app.inference.manager"):
        state = manager.bootstrap("ortho_tka", background=False, startup_preset=None)

    assert state is not None and state.ready is True
    assert manager.active_profile.active_preset is None
    assert manager.active_profile.standards == {"patellar-reamer": 1}
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], \
        "no preset requested — nothing to complain about"


def test_ortho_startup_is_unaffected_by_the_demo_default(tmp_path):
    """ACTIVE_MODEL_PACKAGE=ortho_tka must still boot cleanly."""
    packages_dir = tmp_path / "model_packages"
    write_package(packages_dir, "ortho_tka",
                  class_weights={"patellar-reamer": 12.0},
                  standards={"patellar-reamer": 2},
                  adapter_options={"counts": {}})
    manager = make_manager(tmp_path)
    manager.packages_dir = packages_dir
    manager.discover(force=True)

    state = manager.bootstrap("ortho_tka", background=False, startup_preset="")

    assert state.ready is True
    assert state.package_id == "ortho_tka"


# ── 10. misconfiguration must be loud, never a silent fallback ───────────────

def test_unknown_startup_preset_fails_loudly(tmp_path, caplog):
    packages_dir, _root = _demo_like_package(tmp_path)
    manager = make_manager(tmp_path)
    manager.packages_dir = packages_dir
    manager.discover(force=True)
    _seed_disk_preset(manager, "SurgeryD")

    with caplog.at_level(logging.ERROR, logger="app.inference.manager"):
        state = manager.bootstrap("demo", background=False,
                                  startup_preset="SomethingNotExist")

    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "an unknown startup preset must be reported, not swallowed"
    joined = " ".join(errors)
    assert "SomethingNotExist" in joined
    assert "STARTUP_INVENTORY_PRESET" in joined
    assert "SurgeryB" in joined, "the error must list what IS available"
    # Startup still completes — a kiosk that will not boot is worse — but the
    # tray is left exactly as it was, never silently swapped for a guess.
    assert state is not None and state.ready is True
    assert manager.active_profile.active_preset == "SurgeryD"


def test_startup_preset_on_a_package_without_presets_is_reported(tmp_path, caplog):
    packages_dir = tmp_path / "model_packages"
    write_package(packages_dir, "ortho_tka",
                  class_weights={"patellar-reamer": 12.0},
                  standards={"patellar-reamer": 1},
                  adapter_options={"counts": {}})
    manager = make_manager(tmp_path)
    manager.packages_dir = packages_dir
    manager.discover(force=True)

    with caplog.at_level(logging.ERROR, logger="app.inference.manager"):
        manager.bootstrap("ortho_tka", background=False, startup_preset="SurgeryB")

    joined = " ".join(r.getMessage() for r in caplog.records
                      if r.levelno >= logging.ERROR)
    assert "no surgery presets" in joined, joined
    assert manager.active_profile.active_preset is None


# ── 11-12. what the kiosk actually sees on its first load ────────────────────

@pytest.fixture
def booted_api(tmp_path, monkeypatch):
    """A Flask client over a manager bootstrapped exactly like the unit does."""
    import app.web as web_pkg
    from app.scale_reader import MockScaleReader
    from app.web import history as hist

    packages_dir, _root = _demo_like_package(tmp_path)
    write_package(packages_dir, "ortho_tka",
                  display_name="骨科 TKA",
                  class_weights={"patellar-reamer": 12.0},
                  standards={"patellar-reamer": 1},
                  adapter_options={"counts": {}})
    manager = make_manager(tmp_path)
    manager.packages_dir = packages_dir
    manager.discover(force=True)
    # The unit was shut down on a different tray — the startup preset must win.
    _seed_disk_preset(manager, "SurgeryD")
    manager.bootstrap("demo", background=False, startup_preset="SurgeryB")

    monkeypatch.setattr(web_pkg, "model_manager", manager)
    monkeypatch.setattr(web_pkg, "scale_reader", MockScaleReader(222.0))
    monkeypatch.setattr(hist, "HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(web_pkg, "camera_thread", None, raising=False)
    web_pkg.reset_latest_state()
    web_pkg.app.config["TESTING"] = True
    with web_pkg.app.test_client() as client:
        yield client, manager


def _json(response):
    return json.loads(response.data.decode("utf-8"))


def test_status_reports_demo_and_surgeryb(booted_api):
    client, _manager = booted_api
    payload = _json(client.get("/status"))
    assert payload["active_package"] == "demo"
    assert payload["active_preset"] == "SurgeryB"
    assert payload["has_presets"] is True


def test_inventory_presets_endpoint_reports_surgeryb_active(booted_api):
    client, _manager = booted_api
    payload = _json(client.get("/api/inventory-presets"))
    assert payload["active_preset"] == "SurgeryB"
    assert sorted(p["id"] for p in payload["presets"]) == [
        "SurgeryA", "SurgeryB", "SurgeryC", "SurgeryD"]


def test_standards_endpoint_serves_the_surgeryb_tray(booted_api):
    client, manager = booted_api
    standards = _json(client.get("/standards"))
    assert standards == manager.active_profile.canonical_standards_for("SurgeryB")
    assert {k: v for k, v in standards.items() if v} == _shipped_presets()["SurgeryB"]


def test_a_selected_preset_alone_is_not_a_pass(booted_api):
    """Choosing a tray is not counting one — the first load has no result."""
    client, _manager = booted_api
    payload = _json(client.get("/status"))
    assert payload["has_result"] is False
    assert payload["result_status"] == "no_result"
    assert payload["counts"] == {}
    assert payload["timestamp"] == ""
    assert payload["weight_verification"] is None, \
        "no inference yet — there is nothing to issue a verdict about"
