"""Demo model package: the shipped template, and the integration path it feeds.

The demo model binary is not present on this machine, so nothing here loads a
real demo model.  What IS tested is everything that must already be correct for
the drop-in to work the moment the binary arrives:

* the shipped template is structurally sound and declares a registered adapter
* filling it in produces a valid, activatable package
* switching ortho -> demo -> ortho keeps the two packages' data apart
* a demo inventory is recorded and reported as a demo inventory

The adapter name is deliberately NOT asserted to be "ultralytics": that field in
the template is a placeholder, and the real demo model decides it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import app.config as config
from app.inference.package import ModelPackageError, discover_packages, load_manifest
from app.inference.registry import available_adapters, has_adapter
from app.web import history as hist
from conftest import make_manager, write_package


DEMO_DIR = Path(config.PROJECT_ROOT) / "model_packages" / "demo"
DEMO_MANIFEST = DEMO_DIR / "manifest.json"


def _raw_demo_manifest():
    return json.loads(DEMO_MANIFEST.read_text(encoding="utf-8"))


# ── 1-4. the shipped template ───────────────────────────────────────────────

def test_demo_manifest_exists_and_parses():
    assert DEMO_MANIFEST.exists(), "the demo package template is missing"
    data = _raw_demo_manifest()
    assert data["id"] == "demo"
    assert data["schema_version"] == 1


def test_demo_manifest_loads_as_a_package():
    pkg = load_manifest(DEMO_MANIFEST, project_root=config.PROJECT_ROOT,
                        expected_id="demo")
    assert pkg.id == "demo"
    assert pkg.display_name


def test_demo_is_still_a_template_until_real_data_arrives():
    """It must stay a template while the model and its data are absent.

    Flipping template=false without the real files would put a package in the
    operator's dropdown that cannot possibly work.
    """
    pkg = load_manifest(DEMO_MANIFEST, project_root=config.PROJECT_ROOT,
                        expected_id="demo")
    assert pkg.is_template is True
    assert pkg.model_available() is False


def test_demo_declares_a_registered_adapter():
    """Whatever adapter the demo ends up using, it must exist in the registry.

    The name is not pinned here on purpose — the template's value is a
    placeholder, and the real model determines it.  Guessing it from the file
    extension is exactly what the platform refuses to do.
    """
    data = _raw_demo_manifest()
    adapter = data["adapter"]
    assert has_adapter(adapter), (
        "demo manifest names adapter %r, which is not registered; available: %s"
        % (adapter, ", ".join(available_adapters())))


def test_demo_declares_where_its_data_will_live():
    data = _raw_demo_manifest()
    assert data["model_file"], "no model_file path for the operator to drop into"
    inventory = data["inventory"]
    assert inventory["class_weights"]
    assert inventory["default_standards"]


def test_demo_drop_in_directory_exists():
    """The place to put the binary must already be there, and be git-ignored."""
    drop_in = Path(config.PROJECT_ROOT) / "models" / "demo"
    assert drop_in.is_dir(), "models/demo/ should exist as the drop-in location"


def test_demo_is_listed_but_not_activatable_yet():
    packages = discover_packages(Path(config.MODEL_PACKAGES_DIR),
                                 project_root=config.PROJECT_ROOT)
    assert "demo" in packages, "the demo package must be visible to the operator"
    entry = packages["demo"].to_dict()
    assert entry["valid"] is True          # the manifest itself is fine
    assert entry["template"] is True
    assert entry["activatable"] is False   # ...but it cannot be selected yet


# ── 5-8. what happens once the real data is dropped in ──────────────────────

def _filled_in_demo(tmp_path, *, adapter="fake", class_weights=None,
                    standards=None, with_binary=True):
    """A copy of the shipped template with real data filled in.

    Mirrors exactly what the operator will do: same manifest shape, same file
    names, template flipped to false.
    """
    packages_dir = tmp_path / "model_packages"
    root = packages_dir / "demo"
    root.mkdir(parents=True)

    data = _raw_demo_manifest()
    data["template"] = False
    data["adapter"] = adapter
    data["display_name"] = "Demo 手術器械組"
    data["model_file"] = "models/demo/model.bin"
    data.pop("fallback_model_file", None)
    data["adapter_options"] = {"counts": {"demo-forceps": 2}}
    data["inventory"] = {"class_weights": "class_weight.json",
                         "default_standards": "standards.json"}

    (root / "manifest.json").write_text(json.dumps(data, ensure_ascii=False),
                                        encoding="utf-8")
    (root / "class_weight.json").write_text(
        json.dumps(class_weights if class_weights is not None
                   else {"demo-forceps": 12.0, "demo-clamp": 5.0}), encoding="utf-8")
    (root / "standards.json").write_text(
        json.dumps(standards if standards is not None
                   else {"demo-forceps": 2}), encoding="utf-8")
    if with_binary:
        binary = tmp_path / "models" / "demo"
        binary.mkdir(parents=True, exist_ok=True)
        (binary / "model.bin").write_bytes(b"demo-model")
    return packages_dir


def test_filled_in_demo_package_is_valid_and_activatable(tmp_path):
    packages_dir = _filled_in_demo(tmp_path)

    entry = discover_packages(packages_dir, project_root=tmp_path)["demo"]
    payload = entry.to_dict()

    assert payload["valid"] is True
    assert payload["template"] is False
    assert payload["model_available"] is True
    assert payload["activatable"] is True
    assert payload["error"] is None
    assert payload["model_error"] is None


def test_filled_in_demo_without_the_binary_is_not_activatable(tmp_path):
    """The manifest can be complete while the model file is still missing."""
    packages_dir = _filled_in_demo(tmp_path, with_binary=False)

    payload = discover_packages(packages_dir, project_root=tmp_path)["demo"].to_dict()

    assert payload["valid"] is True         # manifest fine
    assert payload["model_available"] is False
    assert payload["activatable"] is False
    assert "model file not found" in payload["model_error"]


def test_demo_standards_must_be_priced(tmp_path):
    """An expected demo instrument with no unit weight fails validation.

    This is what stops a half-configured demo package from reaching the ward
    and quietly under-reporting the expected total.
    """
    packages_dir = _filled_in_demo(
        tmp_path,
        class_weights={"demo-forceps": 12.0},
        standards={"demo-forceps": 2, "demo-scissors": 1},   # scissors unpriced
    )
    with pytest.raises(ModelPackageError) as excinfo:
        load_manifest(packages_dir / "demo" / "manifest.json",
                      project_root=tmp_path, expected_id="demo")
    assert "demo-scissors" in str(excinfo.value)


def test_recognition_only_demo_with_zero_standards_is_allowed(tmp_path):
    """Before the real weights exist, standards=0 keeps the package usable.

    Recognition works; weight verification simply reports that the standard
    weight is not configured, instead of inventing one.
    """
    packages_dir = _filled_in_demo(
        tmp_path,
        class_weights={"demo-forceps": 12.0},
        standards={"demo-forceps": 0, "demo-clamp": 0},
    )
    pkg = load_manifest(packages_dir / "demo" / "manifest.json",
                        project_root=tmp_path, expected_id="demo")
    assert pkg.load_default_standards() == {"demo-forceps": 0, "demo-clamp": 0}

    from app.weight_verification import compute_weight_verification
    from app.scale_sample import constant_sample

    wv = compute_weight_verification(pkg.load_default_standards(),
                                     pkg.load_class_weights(),
                                     constant_sample(24.0), 0.5)
    assert wv["ready"] is False
    assert wv["passed"] is None
    assert wv["reason"] == "no_standard_weight"


# ── 9-11. ortho -> demo -> ortho ────────────────────────────────────────────

@pytest.fixture
def two_package_manager(tmp_path):
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "ortho_tka",
                  display_name="骨科 TKA 器械組",
                  class_weights={"patellar-reamer": 12.0}, standards={"patellar-reamer": 1},
                  adapter_options={"counts": {"patellar-reamer": 1}})
    write_package(manager.packages_dir, "demo",
                  display_name="Demo 手術器械組",
                  class_weights={"demo-forceps": 12.0, "demo-clamp": 5.0},
                  standards={"demo-forceps": 2},
                  adapter_options={"counts": {"demo-forceps": 2}})
    manager.activate("ortho_tka")
    return manager


def test_ortho_to_demo_to_ortho_keeps_the_packages_apart(two_package_manager):
    manager = two_package_manager

    assert manager.state().standards == {"patellar-reamer": 1}
    assert manager.state().class_weights == {"patellar-reamer": 12.0}
    generation = manager.generation

    manager.activate("demo")
    assert manager.state().package_id == "demo"
    assert manager.state().standards == {"demo-forceps": 2}
    assert manager.state().class_weights == {"demo-forceps": 12.0, "demo-clamp": 5.0}
    assert "patellar-reamer" not in manager.state().class_weights
    assert manager.generation == generation + 1

    # An operator edit under demo.
    manager.update_active_profile("standards", {"demo-forceps": 5}, package_id="demo")

    manager.activate("ortho_tka")
    assert manager.state().standards == {"patellar-reamer": 1}   # untouched
    assert "demo-forceps" not in manager.state().standards
    assert manager.generation == generation + 2

    manager.activate("demo")
    assert manager.state().standards == {"demo-forceps": 5}      # edit survived
    assert manager.recovery_required is False


def test_demo_profile_is_a_separate_file(two_package_manager, tmp_path):
    manager = two_package_manager
    manager.activate("demo")
    manager.update_active_profile("standards", {"demo-forceps": 7}, package_id="demo")

    demo_profile = tmp_path / "profiles" / "demo" / "standards.json"
    ortho_profile = tmp_path / "profiles" / "ortho_tka" / "standards.json"

    assert json.loads(demo_profile.read_text(encoding="utf-8")) == {"demo-forceps": 7}
    assert json.loads(ortho_profile.read_text(encoding="utf-8")) == {"patellar-reamer": 1}


def test_demo_inference_is_recorded_as_a_demo_inventory(two_package_manager,
                                                        tmp_path, monkeypatch):
    manager = two_package_manager
    monkeypatch.setattr(hist, "HISTORY_FILE", tmp_path / "history.json")
    manager.activate("demo")

    with manager.inference_session() as session:
        result = session.infer(object())
        record = hist.make_record(
            source="upload",
            counts=result.counts,
            weight=24.0,
            standards_snapshot=session.standards,
            package_id=session.package_id,
            package_display_name=session.display_name,
            model_identity=(result.model_info.identity() if result.model_info else None),
            class_weights_snapshot=session.class_weights,
        )
        hist.append_record(record)

    manager.activate("ortho_tka")

    stored = hist.load_history()
    assert len(stored) == 1
    saved = stored[0]
    assert saved["package_id"] == "demo"
    assert saved["package_display_name"] == "Demo 手術器械組"
    assert saved["counts"] == {"demo-forceps": 2}
    assert saved["standards_snapshot"] == {"demo-forceps": 2}
    assert "patellar-reamer" not in saved["standards_snapshot"]


def test_report_reads_a_demo_record_with_demo_standards(two_package_manager,
                                                        tmp_path, monkeypatch):
    """Exported after switching back to ortho, the demo row must still be demo."""
    from app.web import report as rpt

    manager = two_package_manager
    monkeypatch.setattr(hist, "HISTORY_FILE", tmp_path / "history.json")
    manager.activate("demo")
    hist.append_record(hist.make_record(
        "webcam", {"demo-forceps": 2}, 24.0, {"demo-forceps": 2},
        package_id="demo", package_display_name="Demo 手術器械組"))
    manager.activate("ortho_tka")

    csv_text = rpt.generate_csv(hist.load_history(), manager.state().standards)

    rows = [line.split(",") for line in csv_text.splitlines()
            if line.startswith("2") and "demo-forceps" in line]
    assert rows, "the demo record vanished from the report"
    row = rows[0]
    assert row[2] == "demo"
    assert row[6] == "2"        # judged against demo's standard, not ortho's
    assert row[7] == "正常"
    assert "patellar" not in csv_text
