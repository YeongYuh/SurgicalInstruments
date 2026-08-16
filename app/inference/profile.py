"""Package-scoped inventory configuration.

Every model package owns its own standards (expected quantities) and unit
weights.  They live under::

    output/profiles/<package_id>/standards.json
    output/profiles/<package_id>/unit_weights.json

Orthopaedic standards must never leak into an obstetrics package, so nothing
here is global.  The package's own ``standards.json`` is a read-only factory
default; the profile copy is what the operator edits at runtime.

``class_weights`` (grams per instrument) is loaded from the package manifest
and is *not* editable — it is the single source of truth for standard weight.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from app.inference.package import ModelPackage

logger = logging.getLogger(__name__)


def save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a corrupt profile must not brick startup
        logger.warning("[profile] %s is not valid JSON (%s) — ignoring", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("[profile] %s does not contain an object — ignoring", path)
        return None
    return data


def _coerce_ints(data: Mapping[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key, value in data.items():
        try:
            out[str(key)] = max(0, int(value))
        except (TypeError, ValueError):
            logger.warning("[profile] dropping non-integer standard %r=%r", key, value)
    return out


def _coerce_floats(data: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in data.items():
        try:
            out[str(key)] = max(0.0, float(value))
        except (TypeError, ValueError):
            logger.warning("[profile] dropping non-numeric unit weight %r=%r", key, value)
    return out


class PackageProfile:
    """Mutable, persisted, per-package inventory configuration."""

    def __init__(
        self,
        package_id: str,
        *,
        standards: Dict[str, int],
        unit_weights: Dict[str, float],
        class_weights: Dict[str, float],
        standards_path: Path,
        unit_weights_path: Path,
    ) -> None:
        self.package_id = package_id
        self.standards_path = standards_path
        self.unit_weights_path = unit_weights_path
        self._standards = dict(standards)
        self._unit_weights = dict(unit_weights)
        self._class_weights = dict(class_weights)  # read-only after construction
        self._lock = threading.Lock()

    # ── reads (always return copies; callers must not hold a live reference) ──

    @property
    def standards(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._standards)

    @property
    def unit_weights(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._unit_weights)

    @property
    def class_weights(self) -> Dict[str, float]:
        return dict(self._class_weights)

    # ── writes ────────────────────────────────────────────────────────────

    def update_standards(self, values: Mapping[str, Any]) -> Dict[str, int]:
        parsed = _coerce_ints(values)
        with self._lock:
            self._standards.update(parsed)
            snapshot = dict(self._standards)
        save_json_atomic(self.standards_path, snapshot)
        return snapshot

    def update_unit_weights(self, values: Mapping[str, Any]) -> Dict[str, float]:
        parsed = _coerce_floats(values)
        with self._lock:
            self._unit_weights.update(parsed)
            snapshot = dict(self._unit_weights)
        save_json_atomic(self.unit_weights_path, snapshot)
        return snapshot

    def register_classes(self, class_names) -> bool:
        """Add newly seen classes with neutral defaults.  Returns True if changed.

        A model can output a class the operator never configured.  Rather than
        silently dropping it, it is registered with standard=0 / unit weight=0
        so it shows up in the UI as "多出" and can be configured.
        """
        changed_std = changed_uw = False
        with self._lock:
            for name in class_names:
                key = str(name)
                if key not in self._standards:
                    self._standards[key] = 0
                    changed_std = True
                if key not in self._unit_weights:
                    self._unit_weights[key] = 0.0
                    changed_uw = True
            std_snapshot = dict(self._standards)
            uw_snapshot = dict(self._unit_weights)
        if changed_std:
            save_json_atomic(self.standards_path, std_snapshot)
        if changed_uw:
            save_json_atomic(self.unit_weights_path, uw_snapshot)
        return changed_std or changed_uw


def load_profile(
    package: ModelPackage,
    profiles_dir: Path,
    *,
    legacy_package_id: Optional[str] = None,
    legacy_standards_path: Optional[Path] = None,
    legacy_unit_weights_path: Optional[Path] = None,
) -> PackageProfile:
    """Load (or seed) the on-disk profile for ``package``.

    Seeding precedence, highest first:

    1. an existing ``output/profiles/<id>/*.json`` — what the operator edited
    2. the pre-package global ``output/standards.json`` — only for the package
       declared as the legacy one, so an in-service Jetson keeps its site
       configuration across the upgrade
    3. the package's own factory defaults
    4. empty
    """
    profile_dir = Path(profiles_dir) / package.id
    standards_path = profile_dir / "standards.json"
    unit_weights_path = profile_dir / "unit_weights.json"
    is_legacy = legacy_package_id is not None and package.id == legacy_package_id

    class_weights = package.load_class_weights() if not package.is_template else {}

    # ── standards ────────────────────────────────────────────────────────
    standards_source = "profile"
    raw_standards = _read_json_dict(standards_path)
    if raw_standards is None and is_legacy and legacy_standards_path is not None:
        raw_standards = _read_json_dict(Path(legacy_standards_path))
        if raw_standards is not None:
            standards_source = "legacy-global"
    if raw_standards is None:
        try:
            raw_standards = package.load_default_standards()
            standards_source = "package-default"
        except Exception as exc:  # noqa: BLE001 - fall back to empty rather than fail
            logger.warning("[profile] %s: cannot read package default standards: %s",
                           package.id, exc)
            raw_standards = {}
            standards_source = "empty"
    standards = _coerce_ints(raw_standards)

    # ── unit weights ─────────────────────────────────────────────────────
    uw_source = "profile"
    raw_uw = _read_json_dict(unit_weights_path)
    if raw_uw is None and is_legacy and legacy_unit_weights_path is not None:
        raw_uw = _read_json_dict(Path(legacy_unit_weights_path))
        if raw_uw is not None:
            uw_source = "legacy-global"
    if raw_uw is None:
        try:
            raw_uw = package.load_default_unit_weights()
            uw_source = "package-default"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[profile] %s: cannot read package default unit weights: %s",
                           package.id, exc)
            raw_uw = {}
            uw_source = "empty"
    if not raw_uw and class_weights:
        # Nothing configured anywhere: start from the model-defined weights so
        # the BOM report is meaningful on first use.  Editable afterwards.
        raw_uw = dict(class_weights)
        uw_source = "class-weights"
    unit_weights = _coerce_floats(raw_uw)

    profile = PackageProfile(
        package.id,
        standards=standards,
        unit_weights=unit_weights,
        class_weights=class_weights,
        standards_path=standards_path,
        unit_weights_path=unit_weights_path,
    )

    # Persist whatever we seeded so the next start reads it back from the profile.
    if not standards_path.exists():
        save_json_atomic(standards_path, standards)
    if not unit_weights_path.exists():
        save_json_atomic(unit_weights_path, unit_weights)

    logger.info(
        "[profile] %s ready — standards=%d (%s)  unit_weights=%d (%s)  class_weights=%d",
        package.id, len(standards), standards_source,
        len(unit_weights), uw_source, len(class_weights),
    )
    return profile
