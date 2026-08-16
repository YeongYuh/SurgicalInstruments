"""Surgery presets: named standard trays inside one model package.

A preset changes only WHAT IS EXPECTED.  It is not a package switch: no model
is loaded, no adapter is touched, and the generation does not advance.  The one
thing it must do is REPLACE the standards — merging one surgery's tray over
another's leaves instruments expected that this surgery never uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.config as config
import app.web as web_pkg
from app.inference.manager import (
    PackageMismatchError,
    ProfileValidationError,
)
from app.inference.package import ModelPackageError, load_manifest
from app.scale_reader import MockScaleReader
from app.web import history as hist
from conftest import FakeCamera, make_manager, write_package


SHIPPED_PRESETS = Path(config.PROJECT_ROOT) / "model_packages" / "demo" / "surgery_instruments.json"
SHIPPED_WEIGHTS = Path(config.PROJECT_ROOT) / "model_packages" / "demo" / "class_weight.json"

#: Computed from the shipped data, never hard-coded into production files.
EXPECTED_TOTALS = {"SurgeryA": 309, "SurgeryB": 222, "SurgeryC": 107, "SurgeryD": 307}


def _json(response):
    return json.loads(response.data.decode("utf-8"))


def _shipped(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ── 1-4. the shipped preset data ────────────────────────────────────────────

def test_surgery_instruments_file_parses():
    assert SHIPPED_PRESETS.exists()
    data = _shipped(SHIPPED_PRESETS)
    assert isinstance(data, dict)


def test_all_four_surgeries_are_defined():
    data = _shipped(SHIPPED_PRESETS)
    assert sorted(data) == ["SurgeryA", "SurgeryB", "SurgeryC", "SurgeryD"]


def test_every_quantity_is_a_positive_whole_number():
    for name, items in _shipped(SHIPPED_PRESETS).items():
        assert items, "%s defines no instruments" % name
        for cls, qty in items.items():
            assert isinstance(cls, str) and cls.strip()
            assert isinstance(qty, int) and not isinstance(qty, bool), \
                "%s/%s quantity %r is not an integer" % (name, cls, qty)
            assert qty > 0


def test_every_preset_instrument_has_a_class_weight():
    """An unpriced instrument would silently lower that tray's expected total."""
    weights = _shipped(SHIPPED_WEIGHTS)
    for name, items in _shipped(SHIPPED_PRESETS).items():
        missing = sorted(c for c in items if c not in weights)
        assert not missing, "%s expects unpriced instruments: %s" % (name, missing)
        for cls in items:
            assert weights[cls] > 0


def test_demo_package_exposes_its_presets():
    pkg = load_manifest(Path(config.PROJECT_ROOT) / "model_packages/demo/manifest.json",
                        project_root=config.PROJECT_ROOT, expected_id="demo")
    assert pkg.has_presets is True
    presets = pkg.load_presets()
    assert sorted(presets) == ["SurgeryA", "SurgeryB", "SurgeryC", "SurgeryD"]


# ── 5-8. expected weights, computed not asserted from a constant ────────────

@pytest.mark.parametrize("preset_id", ["SurgeryA", "SurgeryB", "SurgeryC", "SurgeryD"])
def test_expected_weight_of_each_surgery(preset_id):
    presets = _shipped(SHIPPED_PRESETS)
    weights = _shipped(SHIPPED_WEIGHTS)
    total = sum(qty * weights[cls] for cls, qty in presets[preset_id].items())
    assert total == EXPECTED_TOTALS[preset_id], (
        "%s totals %s g, expected %s g" % (preset_id, total, EXPECTED_TOTALS[preset_id]))


def test_surgery_a_breakdown_is_what_the_weights_imply():
    """Spelled out once so a weight change cannot pass unnoticed."""
    weights = _shipped(SHIPPED_WEIGHTS)
    assert weights["Adson-Smooth-Tissue-Forceps"] == 23
    assert weights["Adson-Teeth-Tissue-Forceps"] == 21   # 21, not 22 — see report
    assert weights["Mayo Scissors Cvd"] == 71
    assert weights["Patten Retractor"] == 30
    assert weights["Ring-Forceps"] == 75
    assert weights["Smooth-Tissue-Forceps"] == 21
    assert weights["Towel-Clamp"] == 34
    assert 23 + 21 + 71 + 30 + 75 + 21 + 34 * 2 == EXPECTED_TOTALS["SurgeryA"]


# ── preset file validation ──────────────────────────────────────────────────

def _package_with_presets(tmp_path, presets, class_weights=None, standards=None):
    packages_dir = tmp_path / "model_packages"
    root = write_package(
        packages_dir, "surg",
        class_weights=class_weights or {"a": 10.0, "b": 5.0},
        standards=standards if standards is not None else {"a": 1},
        adapter_options={"counts": {"a": 1}},
        manifest_overrides={"inventory": {
            "class_weights": "class_weight.json",
            "default_standards": "standards.json",
            "presets": "presets.json",
        }},
    )
    (root / "presets.json").write_text(json.dumps(presets), encoding="utf-8")
    return packages_dir, root


@pytest.mark.parametrize("presets,fragment", [
    ({"S": {"a": 1.5}}, "whole number"),
    ({"S": {"a": True}}, "must be a number"),
    ({"S": {"a": -1}}, "must not be negative"),
    ({"S": {"a": "1"}}, "must be a number"),
    ({"S": {"": 1}}, "empty class name"),
    ({"S": 3}, "must map class names"),
    ({}, "defines no presets"),
])
def test_invalid_preset_data_is_rejected(tmp_path, presets, fragment):
    _packages_dir, root = _package_with_presets(tmp_path, presets)
    with pytest.raises(ModelPackageError) as excinfo:
        load_manifest(root / "manifest.json", project_root=tmp_path, expected_id="surg")
    assert fragment in str(excinfo.value)


def test_preset_expecting_an_unpriced_instrument_is_rejected(tmp_path):
    _packages_dir, root = _package_with_presets(
        tmp_path, {"S": {"a": 1, "ghost": 2}}, class_weights={"a": 10.0})
    with pytest.raises(ModelPackageError) as excinfo:
        load_manifest(root / "manifest.json", project_root=tmp_path, expected_id="surg")
    assert "ghost" in str(excinfo.value)


def test_zero_quantity_in_a_preset_needs_no_weight(tmp_path):
    _packages_dir, root = _package_with_presets(
        tmp_path, {"S": {"a": 1, "spare": 0}}, class_weights={"a": 10.0})
    pkg = load_manifest(root / "manifest.json", project_root=tmp_path, expected_id="surg")
    assert pkg.load_presets()["S"] == {"a": 1, "spare": 0}


# ── 9-13. applying a preset REPLACES the standards ──────────────────────────

@pytest.fixture
def surgery_manager(tmp_path):
    """A package carrying the real shipped surgery presets and weights."""
    manager = make_manager(tmp_path)
    packages_dir = manager.packages_dir
    weights = _shipped(SHIPPED_WEIGHTS)
    root = write_package(
        packages_dir, "demo",
        display_name="Demo 手術器械組",
        class_weights=weights,
        standards={},
        adapter_options={"counts": {"Towel-Clamp": 2}},
        manifest_overrides={"inventory": {
            "class_weights": "class_weight.json",
            "default_standards": "standards.json",
            "presets": "presets.json",
        }},
    )
    shutil_copy = SHIPPED_PRESETS.read_text(encoding="utf-8")
    (root / "presets.json").write_text(shutil_copy, encoding="utf-8")
    write_package(packages_dir, "ortho_tka",
                  display_name="骨科 TKA",
                  class_weights={"patellar-reamer": 12.0},
                  standards={"patellar-reamer": 1},
                  adapter_options={"counts": {"patellar-reamer": 1}})
    manager.activate("demo")
    return manager


def test_selecting_a_preset_sets_the_expected_quantities(surgery_manager):
    manager = surgery_manager
    manager.apply_preset("SurgeryA", package_id="demo")

    standards = manager.state().standards
    assert standards["Towel-Clamp"] == 2
    assert standards["Ring-Forceps"] == 1
    assert standards["Patten Retractor"] == 1
    assert sum(standards.values()) == 8       # SurgeryA has 8 instruments


def test_switching_surgery_replaces_rather_than_merges(surgery_manager):
    """SurgeryB must not inherit SurgeryA's Ring-Forceps and Patten Retractor.

    A merge would leave the tray reported incomplete for instruments this
    surgery does not use at all.
    """
    manager = surgery_manager
    manager.apply_preset("SurgeryA", package_id="demo")
    after_a = manager.state().standards
    assert after_a["Ring-Forceps"] == 1
    assert after_a["Patten Retractor"] == 1

    manager.apply_preset("SurgeryB", package_id="demo")
    after_b = manager.state().standards

    assert after_b["Adson-Smooth-Tissue-Forceps"] == 2
    assert after_b["Mayo Scissors Cvd"] == 2
    assert after_b["Towel-Clamp"] == 1
    # SurgeryA-only instruments are present as 0, never as a leftover quantity.
    assert after_b["Ring-Forceps"] == 0
    assert after_b["Patten Retractor"] == 0
    assert after_b["Smooth-Tissue-Forceps"] == 0
    assert sum(after_b.values()) == 5


def test_switching_from_b_to_d_is_also_a_replace(surgery_manager):
    manager = surgery_manager
    manager.apply_preset("SurgeryB", package_id="demo")
    manager.apply_preset("SurgeryD", package_id="demo")

    standards = manager.state().standards
    assert standards["Towel-Clamp"] == 2
    assert standards["Knife-Handle-No.3"] == 1
    assert standards["Suction-Tube-Fr.12"] == 1
    assert standards["Mayo Scissors Cvd"] == 0     # SurgeryB-only quantity gone
    assert sum(standards.values()) == 10


def test_expected_weight_follows_the_selected_surgery(surgery_manager):
    from app.weight_verification import expected_weight

    manager = surgery_manager
    for preset_id, total in EXPECTED_TOTALS.items():
        manager.apply_preset(preset_id, package_id="demo")
        state = manager.state()
        assert expected_weight(state.standards, state.class_weights) == total


def test_replace_standards_is_not_a_merge(surgery_manager):
    profile = surgery_manager.active_profile
    profile.update_standards({"Towel-Clamp": 9, "Ring-Forceps": 4})
    assert profile.standards["Ring-Forceps"] == 4

    profile.replace_standards({"Towel-Clamp": 1})

    assert profile.standards == {"Towel-Clamp": 1}
    assert "Ring-Forceps" not in profile.standards


# ── 14 & 15. manual edits ───────────────────────────────────────────────────

def test_a_manual_edit_marks_the_preset_as_modified(surgery_manager):
    manager = surgery_manager
    manager.apply_preset("SurgeryA", package_id="demo")
    assert manager.preset_overview()["preset_modified"] is False

    manager.update_active_profile("standards", {"Towel-Clamp": 5}, package_id="demo")

    overview = manager.preset_overview()
    assert overview["active_preset"] == "SurgeryA"   # still the chosen surgery
    assert overview["preset_modified"] is True       # ...but no longer factory


def test_reselecting_the_same_preset_discards_manual_edits(surgery_manager):
    manager = surgery_manager
    manager.apply_preset("SurgeryA", package_id="demo")
    manager.update_active_profile("standards", {"Towel-Clamp": 5}, package_id="demo")
    assert manager.state().standards["Towel-Clamp"] == 5

    manager.apply_preset("SurgeryA", package_id="demo")

    assert manager.state().standards["Towel-Clamp"] == 2
    assert manager.preset_overview()["preset_modified"] is False


def test_a_manual_edit_is_not_silently_overwritten(surgery_manager):
    """Only an explicit preset selection replaces the operator's numbers."""
    manager = surgery_manager
    manager.apply_preset("SurgeryA", package_id="demo")
    manager.update_active_profile("standards", {"Towel-Clamp": 5}, package_id="demo")

    # Merely reading the overview must not reset anything.
    manager.preset_overview()
    manager.preset_overview()

    assert manager.state().standards["Towel-Clamp"] == 5


# ── 16. persistence ─────────────────────────────────────────────────────────

def test_the_selected_surgery_survives_a_profile_reload(surgery_manager, tmp_path):
    manager = surgery_manager
    manager.apply_preset("SurgeryC", package_id="demo")

    stored = json.loads((tmp_path / "profiles" / "demo" / "preset.json")
                        .read_text(encoding="utf-8"))
    assert stored == {"active_preset": "SurgeryC"}

    # Simulate a restart: rebuild the profile from disk.
    manager.activate("ortho_tka")
    manager.activate("demo")

    assert manager.active_profile.active_preset == "SurgeryC"
    assert manager.state().standards["Knife-Handle-No.3"] == 1


def test_a_stored_preset_that_no_longer_exists_resolves_to_none(surgery_manager, tmp_path):
    manager = surgery_manager
    preset_file = tmp_path / "profiles" / "demo" / "preset.json"
    preset_file.parent.mkdir(parents=True, exist_ok=True)
    preset_file.write_text(json.dumps({"active_preset": "SurgeryZ"}), encoding="utf-8")

    manager.activate("ortho_tka")
    manager.activate("demo")

    assert manager.active_profile.active_preset is None   # no crash
    assert manager.preset_overview()["active_preset"] is None


# ── 17 & 18. rejections ─────────────────────────────────────────────────────

def test_unknown_preset_is_rejected(surgery_manager):
    with pytest.raises(ProfileValidationError) as excinfo:
        surgery_manager.apply_preset("SurgeryZ", package_id="demo")
    assert "unknown preset" in str(excinfo.value)


def test_preset_for_the_wrong_package_is_rejected(surgery_manager):
    with pytest.raises(PackageMismatchError):
        surgery_manager.apply_preset("SurgeryA", package_id="ortho_tka")


def test_a_package_without_presets_refuses_preset_selection(surgery_manager):
    manager = surgery_manager
    manager.activate("ortho_tka")
    with pytest.raises(ProfileValidationError) as excinfo:
        manager.apply_preset("SurgeryA", package_id="ortho_tka")
    assert "no surgery presets" in str(excinfo.value)


# ── 19 & 20. a preset is NOT a package switch ───────────────────────────────

def test_selecting_a_preset_does_not_touch_the_model(surgery_manager):
    manager = surgery_manager
    adapter = manager.active_adapter
    generation = manager.generation
    loads = adapter.load_calls
    unloads = adapter.unload_calls

    manager.apply_preset("SurgeryA", package_id="demo")
    manager.apply_preset("SurgeryD", package_id="demo")

    assert manager.generation == generation      # no new activation
    assert manager.active_adapter is adapter     # same adapter object
    assert adapter.load_calls == loads           # not reloaded
    assert adapter.unload_calls == unloads       # not unloaded
    assert adapter.loaded is True


# ── 21-24. results and history ──────────────────────────────────────────────

@pytest.fixture
def api(surgery_manager, tmp_path, monkeypatch):
    monkeypatch.setattr(web_pkg, "model_manager", surgery_manager)
    monkeypatch.setattr(web_pkg, "scale_reader", MockScaleReader(309.0))
    monkeypatch.setattr(hist, "HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(web_pkg, "camera_thread", None, raising=False)
    web_pkg.reset_latest_state()
    web_pkg.app.config["TESTING"] = True
    with web_pkg.app.test_client() as client:
        yield client, surgery_manager


def test_preset_switch_invalidates_the_previous_inference(api):
    """SurgeryA's counts must not be re-judged against SurgeryB's tray.

    The model did not change, so nothing else would have marked them stale.
    """
    client, manager = api
    manager.apply_preset("SurgeryA", package_id="demo")

    with manager.inference_session() as session:
        result = session.infer(object())
    with web_pkg.state_lock:
        web_pkg.latest_state.update({
            "timestamp": "2026-05-21 10:00:00",
            "counts": result.counts,
            "weight": 309.0,
            "package_id": session.package_id,
            "model_generation": session.generation,
            "weight_verification": {"ready": True, "passed": True},
        })
    assert _json(client.get("/status"))["has_result"] is True

    response = client.post("/api/inventory-preset",
                           json={"id": "SurgeryB", "package_id": "demo"})
    assert response.status_code == 200

    status = _json(client.get("/status"))
    assert status["has_result"] is False
    assert status["counts"] == {}
    assert status["active_preset"] == "SurgeryB"


def test_preset_switch_invalidates_the_camera_result(api, monkeypatch):
    client, manager = api
    manager.apply_preset("SurgeryA", package_id="demo")
    cam = FakeCamera(running=True, recognizing=True, result={
        "timestamp": "2026-05-21 10:00:00", "counts": {"Towel-Clamp": 2},
        "weight": 309.0, "package_id": "demo", "package_display_name": "Demo",
        "model_generation": manager.generation, "annotated_b64": None,
    })
    monkeypatch.setattr(web_pkg, "camera_thread", cam, raising=False)

    client.post("/api/inventory-preset", json={"id": "SurgeryC", "package_id": "demo"})

    assert cam.invalidated == 1
    assert _json(client.get("/camera/result"))["has_result"] is False


def test_a_history_record_stores_the_selected_surgery(api, tmp_path):
    client, manager = api
    manager.apply_preset("SurgeryB", package_id="demo")

    with manager.inference_session() as session:
        result = session.infer(object())
        hist.append_record(hist.make_record(
            source="upload", counts=result.counts, weight=222.0,
            standards_snapshot=session.standards,
            package_id=session.package_id,
            package_display_name=session.display_name,
            preset_id=session.preset_id))

    stored = hist.load_history()[-1]
    assert stored["preset_id"] == "SurgeryB"
    assert stored["package_id"] == "demo"
    assert stored["standards_snapshot"]["Mayo Scissors Cvd"] == 2


def test_old_history_keeps_its_own_standards_after_a_preset_change(api):
    """A past inventory is read with the tray it was taken under."""
    from app.web import report as rpt

    client, manager = api
    manager.apply_preset("SurgeryA", package_id="demo")
    hist.append_record(hist.make_record(
        "webcam", {"Ring-Forceps": 1}, 309.0, {"Ring-Forceps": 1},
        package_id="demo", package_display_name="Demo 手術器械組",
        preset_id="SurgeryA"))

    manager.apply_preset("SurgeryB", package_id="demo")   # Ring-Forceps now 0

    csv_text = rpt.generate_csv(hist.load_history(), manager.state().standards)
    rows = [line.split(",") for line in csv_text.splitlines()
            if "Ring-Forceps" in line and not line.startswith("timestamp")]

    assert rows
    assert rows[0][6] == "1"       # judged against SurgeryA's standard, not 0
    assert rows[0][7] == "正常"
    assert rows[0][9] == "SurgeryA"


def test_report_header_keeps_preset_last_for_column_stability():
    from app.web import report as rpt

    header = rpt.generate_csv([]).splitlines()[0].split(",")
    assert header[:9] == ["timestamp", "source", "package_id", "package_name",
                          "class_name", "detected", "standard", "status", "weight_g"]
    assert header[9] == "preset_id"


def test_a_record_without_a_preset_still_exports(api):
    from app.web import report as rpt

    hist.append_record(hist.make_record("upload", {"a": 1}, 1.0, {"a": 1}))
    csv_text = rpt.generate_csv(hist.load_history())
    assert csv_text.count("\n") >= 2      # header + at least one row


# ── 25 & 26. the API surface ────────────────────────────────────────────────

def test_get_presets_returns_all_four_with_weights(api):
    client, _ = api
    payload = _json(client.get("/api/inventory-presets"))

    assert payload["ok"] is True
    assert payload["package_id"] == "demo"
    by_id = {p["id"]: p for p in payload["presets"]}
    assert sorted(by_id) == ["SurgeryA", "SurgeryB", "SurgeryC", "SurgeryD"]
    for preset_id, total in EXPECTED_TOTALS.items():
        assert by_id[preset_id]["expected_weight"] == total
    assert by_id["SurgeryA"]["instrument_count"] == 8
    assert by_id["SurgeryB"]["instrument_count"] == 5


def test_a_package_without_presets_reports_an_empty_list(api):
    client, manager = api
    manager.activate("ortho_tka")

    payload = _json(client.get("/api/inventory-presets"))

    assert payload["ok"] is True
    assert payload["package_id"] == "ortho_tka"
    assert payload["active_preset"] is None
    assert payload["presets"] == []
    assert _json(client.get("/status"))["has_presets"] is False


def test_post_preset_applies_and_reports_the_new_state(api):
    client, manager = api

    payload = _json(client.post("/api/inventory-preset",
                                json={"id": "SurgeryD", "package_id": "demo"}))

    assert payload["ok"] is True
    assert payload["active_preset"] == "SurgeryD"
    assert payload["model_reloaded"] is False
    assert payload["standards"]["Towel-Clamp"] == 2
    assert payload["standards"]["Mayo Scissors Cvd"] == 0
    assert manager.state().standards["Suction-Tube-Fr.12"] == 1


def test_post_preset_rejects_a_stale_package_identity(api):
    client, _ = api
    response = client.post("/api/inventory-preset",
                           json={"id": "SurgeryA", "package_id": "ortho_tka"})
    assert response.status_code == 409


def test_post_preset_rejects_an_unknown_preset(api):
    client, _ = api
    response = client.post("/api/inventory-preset",
                           json={"id": "SurgeryZ", "package_id": "demo"})
    assert response.status_code == 400
    assert "unknown preset" in _json(response)["error"]


def test_demo_to_ortho_to_demo_restores_the_surgery(api):
    """Preset choice is per package and survives a package round trip."""
    client, manager = api
    client.post("/api/inventory-preset", json={"id": "SurgeryC", "package_id": "demo"})
    assert _json(client.get("/api/inventory-presets"))["active_preset"] == "SurgeryC"

    manager.activate("ortho_tka")
    ortho = _json(client.get("/api/inventory-presets"))
    assert ortho["presets"] == []
    assert ortho["active_preset"] is None

    manager.activate("demo")
    restored = _json(client.get("/api/inventory-presets"))
    assert restored["active_preset"] == "SurgeryC"
    assert manager.state().standards["Knife-Handle-No.3"] == 1


# ── the model must actually be able to see what a preset expects ────────────

def test_preset_naming_classes_the_model_cannot_detect_is_refused(tmp_path):
    """The single biggest risk when the real demo model arrives.

    The preset class names must match the model's class names exactly.  If they
    do not, the tray can never be completed, so selecting it is refused at the
    moment of selection rather than producing a permanently "missing" inventory.
    """
    manager = make_manager(tmp_path)
    root = write_package(
        manager.packages_dir, "demo",
        adapter="class_listing",
        class_weights={"Towel-Clamp": 34.0, "Ring-Forceps": 75.0},
        standards={"Towel-Clamp": 1},
        adapter_options={"model_classes": ["Towel-Clamp"],   # Ring-Forceps unknown
                         "counts": {"Towel-Clamp": 1}},
        manifest_overrides={"inventory": {
            "class_weights": "class_weight.json",
            "default_standards": "standards.json",
            "presets": "presets.json",
        }},
    )
    (root / "presets.json").write_text(json.dumps({
        "Known": {"Towel-Clamp": 2},
        "Unknown": {"Towel-Clamp": 1, "Ring-Forceps": 1},
    }), encoding="utf-8")
    manager.activate("demo")

    # A tray the model can see is fine.
    manager.apply_preset("Known", package_id="demo")
    assert manager.state().standards["Towel-Clamp"] == 2

    with pytest.raises(ProfileValidationError) as excinfo:
        manager.apply_preset("Unknown", package_id="demo")
    assert "Ring-Forceps" in str(excinfo.value)
    assert "無法辨識" in str(excinfo.value)
    # The refused selection changed nothing.
    assert manager.state().standards["Towel-Clamp"] == 2
    assert manager.active_profile.active_preset == "Known"


# ── alignment with the real demo model's class list ─────────────────────────

DEMO_DATA_YAML = Path(config.PROJECT_ROOT) / "models" / "demo" / "data_0923.yaml"


def _demo_model_classes():
    import re

    text = DEMO_DATA_YAML.read_text(encoding="utf-8")
    names = re.search(r"names:\s*\[(.*?)\]", text, re.S).group(1)
    return [n.strip().strip("'\"") for n in names.split(",")]


@pytest.mark.skipif(not DEMO_DATA_YAML.exists(),
                    reason="demo dataset config not present on this machine")
def test_class_weights_match_the_demo_model_class_list_exactly():
    """The one mismatch that would make every preset unusable.

    Preset and weight keys must equal the model's class names character for
    character; anything else and selecting that surgery is refused because the
    tray could never be completed.
    """
    model_classes = _demo_model_classes()
    weights = _shipped(SHIPPED_WEIGHTS)

    assert len(model_classes) == 17
    assert set(weights) == set(model_classes), (
        "unpriced model classes: %s | weights for unknown classes: %s"
        % (sorted(set(model_classes) - set(weights)),
           sorted(set(weights) - set(model_classes))))


@pytest.mark.skipif(not DEMO_DATA_YAML.exists(),
                    reason="demo dataset config not present on this machine")
def test_every_surgery_only_expects_instruments_the_model_can_detect():
    model_classes = set(_demo_model_classes())
    for name, items in _shipped(SHIPPED_PRESETS).items():
        unknown = sorted(set(items) - model_classes)
        assert not unknown, "%s expects classes the model has no output for: %s" % (
            name, unknown)
