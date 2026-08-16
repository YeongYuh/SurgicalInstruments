"""ModelManager — owns the one active model package at a time.

Responsibilities
----------------
* discover and validate packages
* build the right adapter from the adapter registry
* keep exactly ONE model loaded (Jetson Nano has 4 GB — preloading every
  package is not an option)
* switch packages atomically, so the platform is never in a half-switched
  "new standards + old model" state
* hand out a monotonically increasing ``generation`` so background workers can
  discard results produced by a package that is no longer active
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.inference.base import ModelAdapter
from app.inference.package import (
    DiscoveredPackage,
    ModelPackage,
    ModelPackageError,
    discover_packages,
)
from app.inference.profile import PackageProfile, load_profile
from app.inference.registry import UnknownAdapterError, get_adapter_class
from app.inference.types import InferenceResult, ModelInfo

logger = logging.getLogger(__name__)


class ModelManagerError(RuntimeError):
    """Raised when a package cannot be activated or no model is active."""


class ActiveState:
    """Immutable snapshot of what is active right now.

    Handing callers a snapshot (rather than letting them read three separate
    attributes) is what prevents a package switch between two reads from
    pairing one package's counts with another package's standards.
    """

    __slots__ = ("generation", "package", "profile", "adapter")

    def __init__(self, generation: int, package: Optional[ModelPackage],
                 profile: Optional[PackageProfile], adapter: Optional[ModelAdapter]) -> None:
        self.generation = generation
        self.package = package
        self.profile = profile
        self.adapter = adapter

    @property
    def ready(self) -> bool:
        return self.package is not None and self.profile is not None and self.adapter is not None

    @property
    def package_id(self) -> str:
        return self.package.id if self.package else ""

    @property
    def display_name(self) -> str:
        return self.package.display_name if self.package else ""

    @property
    def standards(self) -> Dict[str, int]:
        return self.profile.standards if self.profile else {}

    @property
    def class_weights(self) -> Dict[str, float]:
        return self.profile.class_weights if self.profile else {}

    @property
    def unit_weights(self) -> Dict[str, float]:
        return self.profile.unit_weights if self.profile else {}


class ModelManager:
    def __init__(
        self,
        packages_dir: Path,
        profiles_dir: Path,
        *,
        project_root: Path,
        legacy_package_id: Optional[str] = None,
        legacy_standards_path: Optional[Path] = None,
        legacy_unit_weights_path: Optional[Path] = None,
    ) -> None:
        self.packages_dir = Path(packages_dir)
        self.profiles_dir = Path(profiles_dir)
        self.project_root = Path(project_root)
        self.legacy_package_id = legacy_package_id
        self.legacy_standards_path = legacy_standards_path
        self.legacy_unit_weights_path = legacy_unit_weights_path

        self._state_lock = threading.RLock()   # guards the active tuple
        self._switch_lock = threading.Lock()   # serialises activations
        self._generation = 0
        self._package: Optional[ModelPackage] = None
        self._profile: Optional[PackageProfile] = None
        self._adapter: Optional[ModelAdapter] = None
        self._packages: Dict[str, DiscoveredPackage] = {}
        self._last_error: Optional[str] = None
        self._loading = False

    # ── discovery ─────────────────────────────────────────────────────────

    def discover(self, force: bool = True) -> Dict[str, DiscoveredPackage]:
        if force or not self._packages:
            found = discover_packages(self.packages_dir, project_root=self.project_root)
            with self._state_lock:
                self._packages = found
            logger.info("[models] discovered %d package(s) in %s: %s",
                        len(found), self.packages_dir, ", ".join(sorted(found)) or "(none)")
        return dict(self._packages)

    def list_packages(self) -> List[Dict[str, Any]]:
        packages = self.discover(force=True)
        active_id = self.state().package_id
        return [entry.to_dict(active=(entry.id == active_id))
                for entry in sorted(packages.values(), key=lambda e: e.id)]

    def get_discovered(self, package_id: str) -> Optional[DiscoveredPackage]:
        """Look up a package, rescanning if it is not already known.

        The rescan matters operationally: dropping a new package directory onto
        a running kiosk should make it selectable, without restarting the unit.
        Discovery is JSON-only, so the extra scan is cheap.
        """
        if not self._packages or package_id not in self._packages:
            self.discover(force=True)
        return self._packages.get(package_id)

    # ── active state ──────────────────────────────────────────────────────

    def state(self) -> ActiveState:
        with self._state_lock:
            return ActiveState(self._generation, self._package, self._profile, self._adapter)

    @property
    def generation(self) -> int:
        with self._state_lock:
            return self._generation

    @property
    def active_package(self) -> Optional[ModelPackage]:
        with self._state_lock:
            return self._package

    @property
    def active_profile(self) -> Optional[PackageProfile]:
        with self._state_lock:
            return self._profile

    @property
    def active_adapter(self) -> Optional[ModelAdapter]:
        with self._state_lock:
            return self._adapter

    @property
    def last_error(self) -> Optional[str]:
        with self._state_lock:
            return self._last_error

    # ── activation ────────────────────────────────────────────────────────

    def _resolve_for_activation(self, package_id: str) -> ModelPackage:
        entry = self.get_discovered(package_id)
        if entry is None:
            available = ", ".join(sorted(self._packages)) or "(none)"
            raise ModelManagerError(
                "model package '%s' not found in %s — available: %s"
                % (package_id, self.packages_dir, available))
        if entry.package is None:
            raise ModelManagerError(
                "model package '%s' is invalid: %s" % (package_id, entry.error))
        if entry.package.is_template:
            raise ModelManagerError(
                "model package '%s' is a template — fill in its manifest "
                "(model_file, adapter, inventory.class_weights) before activating it"
                % package_id)
        return entry.package

    def _build(self, package: ModelPackage) -> Tuple[ModelAdapter, PackageProfile]:
        """Create + load a fresh adapter and profile.  Raises on failure."""
        try:
            adapter_cls = get_adapter_class(package.adapter)
        except UnknownAdapterError as exc:
            raise ModelManagerError(str(exc))

        profile = load_profile(
            package,
            self.profiles_dir,
            legacy_package_id=self.legacy_package_id,
            legacy_standards_path=self.legacy_standards_path,
            legacy_unit_weights_path=self.legacy_unit_weights_path,
        )

        adapter = adapter_cls(package)
        try:
            adapter.load()
        except Exception as exc:
            try:
                adapter.unload()
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
            raise ModelManagerError(
                "failed to load model for package '%s' (%s): %s"
                % (package.id, package.adapter, exc))
        return adapter, profile

    def _publish(self, package: ModelPackage, profile: PackageProfile,
                 adapter: ModelAdapter) -> int:
        """Atomically swap in the new triple and return the new generation."""
        with self._state_lock:
            previous = self._adapter
            self._package = package
            self._profile = profile
            self._adapter = adapter
            self._generation += 1
            self._last_error = None
            generation = self._generation
        # Release the old model outside the lock — unload can be slow.
        if previous is not None and previous is not adapter:
            try:
                previous.unload()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[models] error releasing previous adapter: %s", exc)
        return generation

    def activate(self, package_id: str, *, warmup: bool = True) -> ActiveState:
        """Fully synchronous package switch.

        Everything that can fail (validation, adapter construction, model load,
        profile load) happens BEFORE the swap, so a failure leaves the
        previously working package untouched and still serving.
        """
        with self._switch_lock:
            current = self.state()
            if current.ready and current.package_id == package_id:
                logger.info("[models] package '%s' already active", package_id)
                return current

            package = self._resolve_for_activation(package_id)
            logger.info("[models] activating package '%s' (adapter=%s)",
                        package.id, package.adapter)
            with self._state_lock:
                self._loading = True
            try:
                adapter, profile = self._build(package)
                if warmup:
                    adapter.warmup()   # best effort; error recorded in model_info
                generation = self._publish(package, profile, adapter)
            except ModelManagerError as exc:
                with self._state_lock:
                    self._last_error = str(exc)
                logger.error("[models] activation of '%s' failed: %s", package_id, exc)
                raise
            finally:
                with self._state_lock:
                    self._loading = False

            logger.info("[models] package '%s' active — generation=%d", package.id, generation)
            return self.state()

    def bootstrap(self, package_id: str, *, background: bool = True) -> Optional[ActiveState]:
        """Startup activation.

        The package manifest and profile are published synchronously (cheap —
        JSON only) so that /standards and the UI have data immediately, while
        the expensive model load runs in the background exactly like the old
        detector warmup thread did.  ``infer()`` still works during that window
        because adapters load lazily under their own lock.

        This split is safe only because there is no previously active package
        at startup.  Runtime switches always go through ``activate()``, which
        is fully synchronous.
        """
        with self._switch_lock:
            try:
                package = self._resolve_for_activation(package_id)
            except ModelManagerError as exc:
                with self._state_lock:
                    self._last_error = str(exc)
                logger.error("[models] startup package '%s' unusable: %s", package_id, exc)
                return None

            try:
                adapter_cls = get_adapter_class(package.adapter)
            except UnknownAdapterError as exc:
                with self._state_lock:
                    self._last_error = str(exc)
                logger.error("[models] startup package '%s': %s", package_id, exc)
                return None

            try:
                profile = load_profile(
                    package,
                    self.profiles_dir,
                    legacy_package_id=self.legacy_package_id,
                    legacy_standards_path=self.legacy_standards_path,
                    legacy_unit_weights_path=self.legacy_unit_weights_path,
                )
            except Exception as exc:  # noqa: BLE001
                with self._state_lock:
                    self._last_error = str(exc)
                logger.error("[models] startup profile for '%s' failed: %s", package_id, exc)
                return None

            adapter = adapter_cls(package)
            generation = self._publish(package, profile, adapter)
            logger.info("[models] startup package '%s' published — generation=%d",
                        package.id, generation)

        if background:
            threading.Thread(
                target=self._background_load,
                args=(package.id, generation),
                name="model-warmup",
                daemon=True,
            ).start()
        else:
            self._background_load(package.id, generation)
        return self.state()

    def _background_load(self, package_id: str, generation: int) -> None:
        adapter = self.active_adapter
        if adapter is None or self.generation != generation:
            return
        try:
            adapter.load()
        except Exception as exc:  # noqa: BLE001 - a cold load failure is not fatal here
            logger.warning("[models] background load of '%s' failed: %s — will retry on "
                           "first inference", package_id, exc)
            with self._state_lock:
                self._last_error = str(exc)
            return
        if self.generation != generation:
            return
        adapter.warmup()

    # ── inference ─────────────────────────────────────────────────────────

    def infer(self, image: Any, conf: Optional[float] = None,
              imgsz: Optional[int] = None) -> InferenceResult:
        """Run inference with the active model.

        Returns a result whose ``metadata['generation']`` is the generation the
        inference actually ran under.  Background workers compare that against
        the current generation and discard the result if it changed.
        """
        state = self.state()
        if not state.ready or state.adapter is None:
            raise ModelManagerError(
                "no active model package (%s)" % (self.last_error or "not configured"))
        result = state.adapter.infer(image, conf=conf, imgsz=imgsz)
        result.metadata.setdefault("generation", state.generation)
        result.metadata.setdefault("package_id", state.package_id)
        return result

    # ── introspection ─────────────────────────────────────────────────────

    def model_info(self) -> Dict[str, Any]:
        state = self.state()
        if not state.ready or state.package is None:
            return {
                "active": False,
                "generation": state.generation,
                "package_id": None,
                "error": self.last_error,
                "packages_dir": str(self.packages_dir),
            }
        adapter = state.adapter
        info: ModelInfo = (adapter.model_info if adapter is not None
                           else ModelInfo(package_id=state.package_id))
        payload = info.to_dict()
        payload.update({
            "active": True,
            "generation": state.generation,
            "loading": self._loading,
            "package": state.package.summary(),
            "standards_count": len(state.standards),
            "class_weights_count": len(state.class_weights),
            "error": self.last_error,
        })
        return payload

    def shutdown(self) -> None:
        with self._state_lock:
            adapter = self._adapter
            self._adapter = None
            self._package = None
            self._profile = None
        if adapter is not None:
            try:
                adapter.unload()
            except Exception:  # noqa: BLE001
                pass
