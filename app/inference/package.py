"""Instrument Model Package — manifest parsing, validation, discovery.

A package bundles everything the platform needs to recognise one family of
surgical instruments:

    model_packages/<id>/manifest.json     what model, which adapter, defaults
    model_packages/<id>/standards.json    factory-default expected quantities

The model binary itself is usually NOT inside the package (and not in git):
manifests reference it by a project-relative path such as ``models/best.onnx``.

Adding support for a new instrument family means adding a package directory —
not editing camera, routes, history, or report code.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from app.inference.registry import (
    adapter_requires_model_file,
    available_adapters,
    has_adapter,
)

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
MANIFEST_FILENAME = "manifest.json"

# Package ids become directory names under output/profiles/, so they are
# restricted to a safe slug alphabet.  This is what stops a manifest from
# writing outside the profiles directory.
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ModelPackageError(Exception):
    """Manifest is missing, malformed, or references unusable data."""


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ModelPackageError(message)


def _as_str(data: Mapping[str, Any], key: str, *, required: bool = True,
            default: str = "") -> str:
    if key not in data or data[key] is None:
        _require(not required, "missing required field '%s'" % key)
        return default
    value = data[key]
    _require(isinstance(value, str), "field '%s' must be a string, got %s"
             % (key, type(value).__name__))
    value = value.strip()
    _require(not required or bool(value), "field '%s' must not be empty" % key)
    return value


def _safe_relative(raw: str, field: str) -> None:
    parts = Path(raw).parts
    _require(".." not in parts,
             "path traversal is not allowed in '%s': %r" % (field, raw))


class ModelPackage:
    """A validated manifest plus the paths it resolves to."""

    def __init__(
        self,
        *,
        package_id: str,
        display_name: str,
        department: str,
        adapter: str,
        root: Path,
        manifest_path: Path,
        project_root: Path,
        schema_version: int = 1,
        model_file: Optional[Path] = None,
        fallback_model_file: Optional[Path] = None,
        adapter_options: Optional[Dict[str, Any]] = None,
        confidence: float = 0.25,
        image_size: Optional[int] = None,
        class_weights_path: Optional[Path] = None,
        default_standards_path: Optional[Path] = None,
        default_unit_weights_path: Optional[Path] = None,
        presets_path: Optional[Path] = None,
        is_template: bool = False,
        raw: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.id = package_id
        self.display_name = display_name
        self.department = department
        self.adapter = adapter
        self.root = root
        self.manifest_path = manifest_path
        self.project_root = project_root
        self.schema_version = schema_version
        self.model_file = model_file
        self.fallback_model_file = fallback_model_file
        self.adapter_options = adapter_options or {}
        self.confidence = confidence
        self.image_size = image_size
        self.class_weights_path = class_weights_path
        self.default_standards_path = default_standards_path
        self.default_unit_weights_path = default_unit_weights_path
        self.presets_path = presets_path
        self.is_template = is_template
        self.raw = raw or {}

    # ── model file helpers ────────────────────────────────────────────────

    def model_file_exists(self) -> bool:
        return bool(self.model_file and self.model_file.exists())

    def fallback_exists(self) -> bool:
        return bool(self.fallback_model_file and self.fallback_model_file.exists())

    def resolved_model_file(self) -> Optional[Path]:
        """Primary model if present, else the declared fallback, else None."""
        if self.model_file_exists():
            return self.model_file
        if self.fallback_exists():
            return self.fallback_model_file
        return None

    # ── inventory data ────────────────────────────────────────────────────

    @property
    def requires_model_file(self) -> bool:
        return adapter_requires_model_file(self.adapter)

    def model_available(self) -> bool:
        """Is there actually a model on disk to load?

        A manifest can be perfectly valid while its model binary is absent —
        the two are different failures and are reported separately, so the
        operator is not offered a package that is guaranteed to fail on click.
        """
        if not self.requires_model_file:
            return True
        return self.model_file_exists() or self.fallback_exists()

    def model_error(self) -> Optional[str]:
        if self.is_template or self.model_available():
            return None
        parts = [str(self.model_file)] if self.model_file else []
        if self.fallback_model_file:
            parts.append(str(self.fallback_model_file))
        return "model file not found: %s" % (" / ".join(parts) or "(none declared)")

    def load_class_weights(self) -> Dict[str, float]:
        """grams per single instrument, keyed by class name (read-only)."""
        if self.class_weights_path is None:
            return {}
        return _load_number_map(self.class_weights_path, "class_weights", cast=float)

    def load_default_standards(self) -> Dict[str, int]:
        """Factory-default expected quantities, keyed by class name."""
        if self.default_standards_path is None:
            return {}
        raw = _load_number_map(self.default_standards_path, "default_standards",
                               cast=float, require_int=True)
        return {k: int(v) for k, v in raw.items()}

    def load_default_unit_weights(self) -> Dict[str, float]:
        if self.default_unit_weights_path is None:
            return {}
        return _load_number_map(self.default_unit_weights_path, "default_unit_weights",
                                cast=float)

    @property
    def has_presets(self) -> bool:
        return bool(self.presets_path and self.presets_path.exists())

    def load_presets(self) -> Dict[str, Dict[str, int]]:
        """Named standard trays this package offers, e.g. one per surgery type.

        Optional: a package with a single fixed tray simply declares none, and
        the UI shows no preset selector for it.  Read-only factory data — the
        operator's edits go to the profile, never back into the package.
        """
        if self.presets_path is None:
            return {}
        return _load_presets(self.presets_path)

    # ── misc ──────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "department": self.department,
            "adapter": self.adapter,
            "schema_version": self.schema_version,
            "model_file": str(self.model_file) if self.model_file else "",
            "model_file_exists": self.model_file_exists(),
            "fallback_model_file": (str(self.fallback_model_file)
                                    if self.fallback_model_file else ""),
            "model_available": self.model_available(),
            "requires_model_file": self.requires_model_file,
            "confidence": self.confidence,
            "image_size": self.image_size,
            "class_weights": (str(self.class_weights_path)
                              if self.class_weights_path else ""),
            "default_standards": (str(self.default_standards_path)
                                  if self.default_standards_path else ""),
            "template": self.is_template,
            "manifest": str(self.manifest_path),
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<ModelPackage %s adapter=%s>" % (self.id, self.adapter)


def _load_number_map(path: Path, field: str, cast=float,
                     require_int: bool = False) -> Dict[str, float]:
    if not path.exists():
        raise ModelPackageError("%s file not found: %s" % (field, path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelPackageError("%s file is not valid JSON (%s): %s" % (field, path, exc))
    if not isinstance(data, dict):
        raise ModelPackageError("%s file must contain a JSON object: %s" % (field, path))
    out: Dict[str, float] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key.strip():
            raise ModelPackageError("%s: class names must be non-empty strings (%s)"
                                    % (field, path))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ModelPackageError("%s: value for '%s' must be a number, got %r"
                                    % (field, key, value))
        if not math.isfinite(float(value)):
            raise ModelPackageError("%s: value for '%s' must be finite, got %r"
                                    % (field, key, value))
        if value < 0:
            raise ModelPackageError("%s: value for '%s' must not be negative" % (field, key))
        if require_int and float(value) != int(value):
            # Silently truncating 1.7 to 1 would quietly change how many
            # instruments the tray is expected to hold.
            raise ModelPackageError(
                "%s: value for '%s' must be a whole number, got %r" % (field, key, value))
        out[key] = cast(value)
    return out


def _load_presets(path: Path) -> Dict[str, Dict[str, int]]:
    """Parse a preset file: {preset name: {class name: quantity}}.

    Held to the same strictness as standards anywhere else — a fractional or
    non-finite quantity is a mistake, not something to round.
    """
    if not path.exists():
        raise ModelPackageError("presets file not found: %s" % path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelPackageError("presets file is not valid JSON (%s): %s" % (path, exc))
    if not isinstance(data, dict):
        raise ModelPackageError("presets file must contain a JSON object: %s" % path)

    presets: Dict[str, Dict[str, int]] = {}
    for name, items in data.items():
        if not isinstance(name, str) or not name.strip():
            raise ModelPackageError("presets: preset names must be non-empty strings (%s)"
                                    % path)
        if not isinstance(items, dict):
            raise ModelPackageError(
                "presets: '%s' must map class names to quantities (%s)" % (name, path))
        parsed: Dict[str, int] = {}
        for cls, qty in items.items():
            if not isinstance(cls, str) or not cls.strip():
                raise ModelPackageError(
                    "presets: '%s' has an empty class name (%s)" % (name, path))
            if isinstance(qty, bool) or not isinstance(qty, (int, float)):
                raise ModelPackageError(
                    "presets: '%s'/'%s' quantity must be a number, got %r"
                    % (name, cls, qty))
            if not math.isfinite(float(qty)):
                raise ModelPackageError(
                    "presets: '%s'/'%s' quantity must be finite, got %r" % (name, cls, qty))
            if qty < 0:
                raise ModelPackageError(
                    "presets: '%s'/'%s' quantity must not be negative" % (name, cls))
            if float(qty) != int(qty):
                raise ModelPackageError(
                    "presets: '%s'/'%s' quantity must be a whole number, got %r"
                    % (name, cls, qty))
            parsed[cls.strip()] = int(qty)
        presets[name.strip()] = parsed
    if not presets:
        raise ModelPackageError("presets file defines no presets: %s" % path)
    return presets


def _cross_validate_presets(presets: Mapping[str, Mapping[str, int]],
                            class_weights: Mapping[str, float],
                            where: str) -> None:
    """Every instrument a preset expects must have a usable unit weight.

    Same rule as the package's default standards: a preset that silently omits
    an instrument's weight would under-report the expected total for that tray.
    """
    problems = []
    for name, items in presets.items():
        missing = sorted(cls for cls, qty in items.items()
                         if int(qty or 0) > 0 and not _usable_class_weight(
                             class_weights.get(cls)))
        if missing:
            problems.append("%s: %s" % (name, ", ".join(missing)))
    if problems:
        raise ModelPackageError(
            "%s: preset instruments have no usable class weight — %s"
            % (where, "; ".join(problems)))


def _usable_class_weight(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) > 0


def _cross_validate_inventory(standards: Mapping[str, int],
                              class_weights: Mapping[str, float],
                              where: str) -> None:
    """Every expected instrument must have a usable unit weight.

    Catching this at package-validation time keeps a half-configured package
    from reaching the ward, where a missing weight would otherwise silently
    lower the expected total and let an incomplete tray pass.
    """
    missing = []
    for cls, qty in standards.items():
        if int(qty or 0) <= 0:
            continue
        weight = class_weights.get(cls)
        if weight is None:
            missing.append("%s (no class weight)" % cls)
        elif not math.isfinite(float(weight)) or float(weight) <= 0:
            missing.append("%s (class weight %r is not a positive number)" % (cls, weight))
    if missing:
        raise ModelPackageError(
            "%s: expected instruments have no usable class weight — %s"
            % (where, "; ".join(sorted(missing))))


def _resolve_path(raw: str, field: str, package_root: Path, project_root: Path) -> Path:
    _safe_relative(raw, field)
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    in_package = (package_root / candidate)
    if in_package.exists():
        return in_package
    return project_root / candidate


def load_manifest(
    manifest_path: Path,
    *,
    project_root: Path,
    expected_id: Optional[str] = None,
) -> ModelPackage:
    """Parse and validate one manifest.  Raises ModelPackageError on any problem.

    Validation deliberately never falls back to a different package: a broken
    manifest must be loud, because silently activating the wrong instrument set
    would produce a confidently wrong inventory result.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise ModelPackageError("manifest not found: %s" % manifest_path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelPackageError("manifest is not valid JSON (%s): %s" % (manifest_path, exc))
    if not isinstance(data, dict):
        raise ModelPackageError("manifest must contain a JSON object: %s" % manifest_path)

    package_root = manifest_path.parent

    # schema_version
    version = data.get("schema_version")
    _require(isinstance(version, int) and not isinstance(version, bool),
             "'schema_version' must be an integer")
    _require(version in SUPPORTED_SCHEMA_VERSIONS,
             "unsupported schema_version %r (supported: %s)"
             % (version, sorted(SUPPORTED_SCHEMA_VERSIONS)))

    # id
    package_id = _as_str(data, "id")
    _require(bool(_ID_RE.match(package_id)),
             "'id' must match %s (lowercase letters, digits, '-', '_'), got %r"
             % (_ID_RE.pattern, package_id))
    if expected_id is not None:
        _require(package_id == expected_id,
                 "manifest id %r does not match its directory name %r"
                 % (package_id, expected_id))

    display_name = _as_str(data, "display_name")
    department = _as_str(data, "department", required=False)

    is_template = bool(data.get("template", False))

    # adapter
    adapter = _as_str(data, "adapter")
    _require(has_adapter(adapter),
             "unknown adapter %r — registered adapters: %s"
             % (adapter, ", ".join(available_adapters()) or "(none)"))

    adapter_options = data.get("adapter_options", {})
    _require(isinstance(adapter_options, dict), "'adapter_options' must be an object")

    # model files — only required by adapters that actually read one
    needs_model_file = adapter_requires_model_file(adapter)
    model_file: Optional[Path] = None
    fallback_file: Optional[Path] = None
    raw_model = data.get("model_file")
    if raw_model is None:
        _require(is_template or not needs_model_file,
                 "missing required field 'model_file' (adapter '%s' needs one)" % adapter)
    else:
        _require(isinstance(raw_model, str) and raw_model.strip(),
                 "'model_file' must be a non-empty string")
        model_file = _resolve_path(raw_model.strip(), "model_file", package_root, project_root)
    raw_fallback = data.get("fallback_model_file")
    if raw_fallback is not None:
        _require(isinstance(raw_fallback, str) and raw_fallback.strip(),
                 "'fallback_model_file' must be a non-empty string")
        fallback_file = _resolve_path(raw_fallback.strip(), "fallback_model_file",
                                      package_root, project_root)

    # inference settings
    inference = data.get("inference", {})
    _require(isinstance(inference, dict), "'inference' must be an object")
    confidence = inference.get("confidence", 0.25)
    _require(isinstance(confidence, (int, float)) and not isinstance(confidence, bool),
             "'inference.confidence' must be a number")
    _require(math.isfinite(float(confidence)),
             "'inference.confidence' must be finite, got %r" % (confidence,))
    _require(0.0 < float(confidence) <= 1.0,
             "'inference.confidence' must be in (0, 1], got %r" % (confidence,))
    image_size = inference.get("image_size")
    if image_size is not None:
        _require(isinstance(image_size, int) and not isinstance(image_size, bool),
                 "'inference.image_size' must be an integer")
        _require(image_size > 0, "'inference.image_size' must be positive")

    # inventory data
    inventory = data.get("inventory", {})
    _require(isinstance(inventory, dict), "'inventory' must be an object")

    class_weights_path: Optional[Path] = None
    raw_cw = inventory.get("class_weights")
    if raw_cw is None:
        _require(is_template, "missing required field 'inventory.class_weights'")
    else:
        _require(isinstance(raw_cw, str) and raw_cw.strip(),
                 "'inventory.class_weights' must be a non-empty string")
        class_weights_path = _resolve_path(raw_cw.strip(), "inventory.class_weights",
                                           package_root, project_root)

    default_standards_path: Optional[Path] = None
    raw_std = inventory.get("default_standards")
    if raw_std is not None:
        _require(isinstance(raw_std, str) and raw_std.strip(),
                 "'inventory.default_standards' must be a non-empty string")
        default_standards_path = _resolve_path(raw_std.strip(), "inventory.default_standards",
                                               package_root, project_root)

    default_unit_weights_path: Optional[Path] = None
    raw_uw = inventory.get("default_unit_weights")
    if raw_uw is not None:
        _require(isinstance(raw_uw, str) and raw_uw.strip(),
                 "'inventory.default_unit_weights' must be a non-empty string")
        default_unit_weights_path = _resolve_path(
            raw_uw.strip(), "inventory.default_unit_weights", package_root, project_root)

    # Optional: named standard trays (e.g. one per surgery type).  A package
    # without them simply has one fixed tray and no preset selector.
    presets_path: Optional[Path] = None
    raw_presets = inventory.get("presets")
    if raw_presets is not None:
        _require(isinstance(raw_presets, str) and raw_presets.strip(),
                 "'inventory.presets' must be a non-empty string")
        presets_path = _resolve_path(raw_presets.strip(), "inventory.presets",
                                     package_root, project_root)

    package = ModelPackage(
        package_id=package_id,
        display_name=display_name,
        department=department,
        adapter=adapter,
        root=package_root,
        manifest_path=manifest_path,
        project_root=project_root,
        schema_version=int(version),
        model_file=model_file,
        fallback_model_file=fallback_file,
        adapter_options=dict(adapter_options),
        confidence=float(confidence),
        image_size=image_size,
        class_weights_path=class_weights_path,
        default_standards_path=default_standards_path,
        default_unit_weights_path=default_unit_weights_path,
        presets_path=presets_path,
        is_template=is_template,
        raw=data,
    )

    # A template is a fill-in-the-blanks skeleton: its referenced data files are
    # not expected to exist yet, so they are not read.  It can be listed but
    # never activated.
    # A template's referenced files may not exist yet, so they are not required.
    # Any that DO exist are still validated: half-filled demo data should fail
    # here, while it is being prepared, not on the ward.
    def _present(path: Optional[Path]) -> bool:
        return path is not None and path.exists()

    check_class_weights = not is_template or _present(class_weights_path)
    class_weights: Dict[str, float] = {}
    if check_class_weights:
        class_weights = package.load_class_weights()   # raises if missing / malformed

    if (not is_template or _present(default_standards_path)) \
            and default_standards_path is not None:
        default_standards = package.load_default_standards()  # raises if malformed
        if check_class_weights:
            _cross_validate_inventory(default_standards, class_weights,
                                      "inventory.default_standards")

    if (not is_template or _present(default_unit_weights_path)) \
            and default_unit_weights_path is not None:
        package.load_default_unit_weights()

    if (not is_template or _present(presets_path)) and presets_path is not None:
        presets = package.load_presets()               # raises if malformed
        if check_class_weights:
            _cross_validate_presets(presets, class_weights, "inventory.presets")

    return package


class DiscoveredPackage:
    """One directory under model_packages/ — valid or not."""

    __slots__ = ("id", "package", "error", "path")

    def __init__(self, package_id: str, path: Path,
                 package: Optional[ModelPackage] = None,
                 error: Optional[str] = None) -> None:
        self.id = package_id
        self.path = path
        self.package = package
        self.error = error

    @property
    def valid(self) -> bool:
        return self.package is not None

    @property
    def is_template(self) -> bool:
        return bool(self.package is not None and self.package.is_template)

    @property
    def model_available(self) -> bool:
        return bool(self.package is not None and self.package.model_available())

    @property
    def activatable(self) -> bool:
        """Can the operator actually select this right now?

        A valid manifest is not enough: without a model binary the activation
        is guaranteed to fail, and offering it only to error out on click is a
        worse experience than greying it out with the reason.
        """
        if self.package is None or self.package.is_template:
            return False
        return self.model_available

    def to_dict(self, *, active: bool = False) -> Dict[str, Any]:
        pkg = self.package
        return {
            "id": self.id,
            "display_name": pkg.display_name if pkg else self.id,
            "department": pkg.department if pkg else "",
            "adapter": pkg.adapter if pkg else "",
            "active": active,
            # 'valid' is about the manifest; 'model_available' is about the
            # binary on disk.  They are different problems with different fixes.
            "valid": self.valid,
            "template": self.is_template,
            "model_available": self.model_available,
            "activatable": self.activatable,
            "model_file": str(pkg.model_file) if (pkg and pkg.model_file) else "",
            "model_file_exists": pkg.model_file_exists() if pkg else False,
            "requires_model_file": pkg.requires_model_file if pkg else True,
            "model_error": pkg.model_error() if pkg else None,
            "error": self.error,
        }


def discover_packages(packages_dir: Path, *, project_root: Path) -> Dict[str, DiscoveredPackage]:
    """Scan a directory for model packages.  Never raises for a bad package."""
    packages_dir = Path(packages_dir)
    found: Dict[str, DiscoveredPackage] = {}
    if not packages_dir.is_dir():
        logger.warning("[packages] directory not found: %s", packages_dir)
        return found

    for entry in sorted(packages_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        manifest = entry / MANIFEST_FILENAME
        if not manifest.exists():
            logger.debug("[packages] skipping %s — no %s", entry, MANIFEST_FILENAME)
            continue
        try:
            pkg = load_manifest(manifest, project_root=project_root, expected_id=entry.name)
            found[pkg.id] = DiscoveredPackage(pkg.id, entry, package=pkg)
        except ModelPackageError as exc:
            logger.warning("[packages] invalid package '%s': %s", entry.name, exc)
            found[entry.name] = DiscoveredPackage(entry.name, entry, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - discovery must never crash startup
            logger.warning("[packages] failed to read package '%s': %s", entry.name, exc)
            found[entry.name] = DiscoveredPackage(entry.name, entry, error=str(exc))
    return found
