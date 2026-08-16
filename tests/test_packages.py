"""Model package discovery, manifest validation, and the adapter registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.inference.package import (
    ModelPackageError,
    discover_packages,
    load_manifest,
)
from app.inference.registry import (
    UnknownAdapterError,
    available_adapters,
    get_adapter_class,
    has_adapter,
    register_adapter,
    unregister_adapter,
)
from conftest import FakeAdapter, write_package, write_raw_manifest


# ── 1. discovery ────────────────────────────────────────────────────────────

def test_discovery_finds_all_packages(tmp_path):
    packages_dir = tmp_path / "model_packages"
    write_package(packages_dir, "ortho")
    write_package(packages_dir, "obgyn")
    (packages_dir / "not_a_package").mkdir()           # no manifest → ignored
    (packages_dir / "_hidden").mkdir()                 # underscore → ignored

    found = discover_packages(packages_dir, project_root=tmp_path)

    assert set(found) == {"ortho", "obgyn"}
    assert all(entry.valid for entry in found.values())


def test_discovery_on_missing_directory_returns_empty(tmp_path):
    found = discover_packages(tmp_path / "nope", project_root=tmp_path)
    assert found == {}


# ── 2. valid manifest ───────────────────────────────────────────────────────

def test_valid_manifest_exposes_every_field(tmp_path):
    packages_dir = tmp_path / "model_packages"
    root = write_package(
        packages_dir, "ortho",
        display_name="骨科",
        department="orthopedics",
        class_weights={"widget": 12.5},
        standards={"widget": 3},
        adapter_options={"task": "segment"},
        confidence=0.4,
        image_size=416,
    )

    pkg = load_manifest(root / "manifest.json", project_root=tmp_path, expected_id="ortho")

    assert pkg.id == "ortho"
    assert pkg.display_name == "骨科"
    assert pkg.department == "orthopedics"
    assert pkg.adapter == "fake"
    assert pkg.adapter_options == {"task": "segment"}
    assert pkg.confidence == 0.4
    assert pkg.image_size == 416
    assert pkg.model_file_exists()
    assert pkg.load_class_weights() == {"widget": 12.5}
    assert pkg.load_default_standards() == {"widget": 3}


def test_template_package_skips_data_file_checks(tmp_path):
    packages_dir = tmp_path / "model_packages"
    root = write_raw_manifest(packages_dir, "demo", {
        "schema_version": 1,
        "id": "demo",
        "display_name": "Demo",
        "adapter": "fake",
        "template": True,
        "model_file": "models/does/not/exist.onnx",
        "inventory": {"class_weights": "class_weight.json"},
    })
    pkg = load_manifest(root / "manifest.json", project_root=tmp_path, expected_id="demo")
    assert pkg.is_template is True
    assert pkg.model_file_exists() is False


# ── 3. invalid manifest rejection ───────────────────────────────────────────

@pytest.mark.parametrize("payload,fragment", [
    ({"id": "x", "display_name": "X", "adapter": "fake"}, "schema_version"),
    ({"schema_version": 99, "id": "x", "display_name": "X", "adapter": "fake"},
     "unsupported schema_version"),
    ({"schema_version": 1, "display_name": "X", "adapter": "fake"}, "'id'"),
    ({"schema_version": 1, "id": "x", "adapter": "fake"}, "display_name"),
    ({"schema_version": 1, "id": "Bad Id!", "display_name": "X", "adapter": "fake"}, "'id'"),
    ({"schema_version": 1, "id": "x", "display_name": "X"}, "adapter"),
    ({"schema_version": 1, "id": "x", "display_name": "X", "adapter": "fake",
      "model_file": "m.bin", "inference": {"confidence": 5.0},
      "inventory": {"class_weights": "cw.json"}}, "confidence"),
    ({"schema_version": 1, "id": "x", "display_name": "X", "adapter": "fake",
      "model_file": "m.bin", "inference": {"image_size": -1},
      "inventory": {"class_weights": "cw.json"}}, "image_size"),
    ({"schema_version": 1, "id": "x", "display_name": "X", "adapter": "fake",
      "inventory": {"class_weights": "cw.json"}}, "model_file"),
    ({"schema_version": 1, "id": "x", "display_name": "X", "adapter": "fake",
      "model_file": "m.bin"}, "class_weights"),
    ("{not json", "not valid JSON"),
    ([1, 2, 3], "JSON object"),
])
def test_invalid_manifest_is_rejected(tmp_path, payload, fragment):
    root = write_raw_manifest(tmp_path / "model_packages", "x", payload)
    with pytest.raises(ModelPackageError) as excinfo:
        load_manifest(root / "manifest.json", project_root=tmp_path)
    assert fragment in str(excinfo.value)


def test_manifest_id_must_match_directory_name(tmp_path):
    packages_dir = tmp_path / "model_packages"
    root = write_package(packages_dir, "ortho")
    # Rename the directory so the manifest id no longer matches.
    renamed = packages_dir / "obgyn"
    root.rename(renamed)

    with pytest.raises(ModelPackageError) as excinfo:
        load_manifest(renamed / "manifest.json", project_root=tmp_path, expected_id="obgyn")
    assert "does not match its directory name" in str(excinfo.value)


def test_invalid_package_is_reported_not_hidden(tmp_path):
    """A broken package must be listed with its error, never silently dropped.

    Silently skipping it would leave the operator staring at a package that is
    'missing' with no explanation.
    """
    packages_dir = tmp_path / "model_packages"
    write_package(packages_dir, "good")
    write_raw_manifest(packages_dir, "broken", {"schema_version": 1, "id": "broken"})

    found = discover_packages(packages_dir, project_root=tmp_path)

    assert set(found) == {"good", "broken"}
    assert found["good"].valid is True
    assert found["broken"].valid is False
    assert found["broken"].error
    assert found["broken"].to_dict()["activatable"] is False


# ── 4. unknown adapter rejection ────────────────────────────────────────────

def test_unknown_adapter_is_rejected_at_validation(tmp_path):
    root = write_package(tmp_path / "model_packages", "x", adapter="no_such_runtime")
    with pytest.raises(ModelPackageError) as excinfo:
        load_manifest(root / "manifest.json", project_root=tmp_path, expected_id="x")
    message = str(excinfo.value)
    assert "unknown adapter" in message
    assert "no_such_runtime" in message
    # The error must list what IS available, or the operator cannot fix it.
    assert "fake" in message


def test_onnx_extension_does_not_imply_an_adapter(tmp_path):
    """A .onnx file must still name its adapter explicitly.

    ONNX is a serialization format, not an inference contract — two ONNX files
    can need completely different post-processing, so guessing from the
    extension would silently produce wrong counts.
    """
    root = write_raw_manifest(tmp_path / "model_packages", "x", {
        "schema_version": 1,
        "id": "x",
        "display_name": "X",
        "adapter": "onnx",              # not a registered adapter name
        "model_file": "model.onnx",
        "inventory": {"class_weights": "class_weight.json"},
    })
    with pytest.raises(ModelPackageError) as excinfo:
        load_manifest(root / "manifest.json", project_root=tmp_path, expected_id="x")
    assert "unknown adapter" in str(excinfo.value)


# ── 5. adapter registry ─────────────────────────────────────────────────────

def test_registry_registers_resolves_and_unregisters():
    assert has_adapter("fake")
    assert "fake" in available_adapters()
    assert get_adapter_class("fake") is FakeAdapter

    class TempAdapter(FakeAdapter):
        name = "temp_adapter"

    register_adapter("temp_adapter", TempAdapter)
    try:
        assert get_adapter_class("temp_adapter") is TempAdapter
        with pytest.raises(ValueError):
            register_adapter("temp_adapter", TempAdapter)   # duplicate without replace
        register_adapter("temp_adapter", TempAdapter, replace=True)
    finally:
        unregister_adapter("temp_adapter")
    assert not has_adapter("temp_adapter")


def test_registry_resolves_lazy_loaders():
    loaded = []

    def loader():
        loaded.append(True)
        return FakeAdapter

    register_adapter("lazy_adapter", loader, replace=True)
    try:
        assert loaded == []                       # not imported just by registering
        assert has_adapter("lazy_adapter")        # ...nor by checking existence
        assert loaded == []
        assert get_adapter_class("lazy_adapter") is FakeAdapter
        assert loaded == [True]
        get_adapter_class("lazy_adapter")         # memoised — loader not called again
        assert loaded == [True]
    finally:
        unregister_adapter("lazy_adapter")


def test_unknown_adapter_class_lookup_raises():
    with pytest.raises(UnknownAdapterError):
        get_adapter_class("definitely_not_registered")


# ── 28. path safety and file error handling ─────────────────────────────────

@pytest.mark.parametrize("field,value", [
    ("model_file", "../../etc/passwd"),
    ("inventory", {"class_weights": "../../../secrets.json"}),
])
def test_path_traversal_is_rejected(tmp_path, field, value):
    payload = {
        "schema_version": 1,
        "id": "x",
        "display_name": "X",
        "adapter": "fake",
        "model_file": "model.bin",
        "inventory": {"class_weights": "class_weight.json"},
    }
    payload[field] = value
    root = write_raw_manifest(tmp_path / "model_packages", "x", payload)
    with pytest.raises(ModelPackageError) as excinfo:
        load_manifest(root / "manifest.json", project_root=tmp_path, expected_id="x")
    assert "path traversal" in str(excinfo.value)


def test_missing_class_weights_file_is_rejected(tmp_path):
    root = write_package(tmp_path / "model_packages", "x")
    (root / "class_weight.json").unlink()
    with pytest.raises(ModelPackageError) as excinfo:
        load_manifest(root / "manifest.json", project_root=tmp_path, expected_id="x")
    assert "class_weights file not found" in str(excinfo.value)


@pytest.mark.parametrize("content,fragment", [
    ("{ broken", "not valid JSON"),
    ('[1,2,3]', "must contain a JSON object"),
    ('{"widget": "heavy"}', "must be a number"),
    ('{"widget": -1}', "must not be negative"),
])
def test_malformed_class_weights_are_rejected(tmp_path, content, fragment):
    root = write_package(tmp_path / "model_packages", "x")
    (root / "class_weight.json").write_text(content, encoding="utf-8")
    with pytest.raises(ModelPackageError) as excinfo:
        load_manifest(root / "manifest.json", project_root=tmp_path, expected_id="x")
    assert fragment in str(excinfo.value)


def test_manifest_paths_resolve_against_project_root_when_not_in_package(tmp_path):
    """A manifest may point at a shared model outside its own directory."""
    shared = tmp_path / "models"
    shared.mkdir()
    (shared / "shared.onnx").write_bytes(b"x")

    root = write_package(tmp_path / "model_packages", "x",
                         model_file="models/shared.onnx", create_model_file=False)
    pkg = load_manifest(root / "manifest.json", project_root=tmp_path, expected_id="x")

    assert pkg.model_file == tmp_path / "models" / "shared.onnx"
    assert pkg.model_file_exists()
