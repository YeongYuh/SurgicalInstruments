"""ModelManager — owns the one active model package at a time.

Responsibilities
----------------
* discover and validate packages
* build the right adapter from the adapter registry
* keep exactly ONE model resident at any instant, including *during* a switch
  (Jetson Nano has 4 GB — two heavy models at once is an OOM, not a hiccup)
* switch packages as an all-or-nothing transaction, rolling the previous
  package back into service if anything fails
* pin a whole inference transaction — model, package, profile, standards,
  class weights, generation — so results can never be published against a
  package that replaced the one they were produced under

Lifecycle gate
--------------
One lock, ``_exec_lock``, serialises everything that touches a loaded model:

    inference session   (infer + publish, held for the whole transaction)
    package switch      (unload old -> load new -> warm -> publish)
    startup background load

Holding it for the whole inference *transaction* — not just the ``infer()``
call — is what closes the publication race: a switch cannot land between the
generation check and the write, because the switch cannot start until the
transaction has finished.

Lock order is always ``_switch_lock`` -> ``_exec_lock`` -> ``_state_lock``,
and callers layer their own locks (web state, camera thread) *below* the
session, never above it.
"""

from __future__ import annotations

import logging
import math
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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

#: How long an inference session waits for the lifecycle gate before giving up.
#: Comfortably longer than a slow Jetson inference (~2 s) plus a package switch
#: (~10 s), so a legitimate wait never trips it.
DEFAULT_SESSION_TIMEOUT = 120.0


class ModelManagerError(RuntimeError):
    """Raised when a package cannot be activated or no model is active."""


class PackageMismatchError(ModelManagerError):
    """A request targeted a package that is no longer the active one."""

    def __init__(self, requested: Optional[str], active: Optional[str]) -> None:
        self.requested = requested
        self.active = active
        super().__init__(
            "request targets model package %r but %r is active"
            % (requested, active))


class ModelBusyError(ModelManagerError):
    """The lifecycle gate could not be acquired in time."""


class ModelNotReadyError(ModelManagerError):
    """The active model is not loaded, warmed, and verified yet.

    ``loading`` distinguishes "wait a moment" from "this is broken", which the
    web layer turns into different messages for the operator.
    """

    def __init__(self, message: str, *, loading: bool = False,
                 detail: Optional[str] = None) -> None:
        self.loading = loading
        self.detail = detail
        super().__init__(message)


class ProfileValidationError(ModelManagerError):
    """A profile edit the active model could never satisfy."""

    def __init__(self, message: str, *, invalid: Optional[List[str]] = None) -> None:
        self.invalid = list(invalid or [])
        super().__init__(message)


def _usable_weight(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) > 0


class ActiveState:
    """Immutable snapshot of what is active right now.

    Handing callers a snapshot (rather than letting them read three separate
    attributes) is what prevents a package switch between two reads from
    pairing one package's counts with another package's standards.
    """

    __slots__ = ("generation", "package", "profile", "adapter", "model_error")

    def __init__(self, generation: int, package: Optional[ModelPackage],
                 profile: Optional[PackageProfile], adapter: Optional[ModelAdapter],
                 model_error: Optional[str] = None) -> None:
        self.generation = generation
        self.package = package
        self.profile = profile
        self.adapter = adapter
        self.model_error = model_error

    @property
    def configured(self) -> bool:
        """A package is selected and its inventory data is loaded.

        Enough to serve /standards and render the UI; NOT enough to promise
        that inference will work.
        """
        return self.package is not None and self.profile is not None \
            and self.adapter is not None

    @property
    def ready(self) -> bool:
        """The model is loaded, verified, and can run right now.

        A loaded model is not automatically a usable one: startup also has to
        clear the compatibility check and the warmup, and either failing leaves
        ``model_error`` set.  Reporting ready on "the file loaded" alone is how
        a kiosk ends up claiming it can count instruments it cannot see.
        """
        return (self.configured
                and bool(self.adapter and self.adapter.loaded
                         and not self.adapter.retired)
                and not self.model_error)

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


class InferenceSession:
    """A pinned inference transaction.

    While the session is open the active package cannot change, so everything
    reached through it — model, standards, class weights, profile, generation —
    belongs to the same activation, from the frame going in to the history
    record coming out.
    """

    __slots__ = ("_manager", "state")

    def __init__(self, manager: "ModelManager", state: ActiveState) -> None:
        self._manager = manager
        self.state = state

    # identity of the pinned activation
    @property
    def generation(self) -> int:
        return self.state.generation

    @property
    def package(self) -> Optional[ModelPackage]:
        return self.state.package

    @property
    def profile(self) -> Optional[PackageProfile]:
        return self.state.profile

    @property
    def package_id(self) -> str:
        return self.state.package_id

    @property
    def display_name(self) -> str:
        return self.state.display_name

    @property
    def standards(self) -> Dict[str, int]:
        return self.state.standards

    @property
    def class_weights(self) -> Dict[str, float]:
        return self.state.class_weights

    def infer(self, image: Any, conf: Optional[float] = None,
              imgsz: Optional[int] = None) -> InferenceResult:
        adapter = self.state.adapter
        if adapter is None:
            raise ModelManagerError("no active model package")
        result = adapter.infer(image, conf=conf, imgsz=imgsz)
        result.metadata.setdefault("generation", self.state.generation)
        result.metadata.setdefault("package_id", self.state.package_id)
        return result

    def register_classes(self, class_names: Iterable[str]) -> bool:
        """Register newly seen classes on the package that produced them.

        Uses the profile captured by this session, never "whichever profile is
        active now" — otherwise an orthopaedic detection finishing just after a
        switch would add its classes to the obstetric package.
        """
        profile = self.state.profile
        if profile is None:
            return False
        return profile.register_classes(class_names)


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
        session_timeout: float = DEFAULT_SESSION_TIMEOUT,
    ) -> None:
        self.packages_dir = Path(packages_dir)
        self.profiles_dir = Path(profiles_dir)
        self.project_root = Path(project_root)
        self.legacy_package_id = legacy_package_id
        self.legacy_standards_path = legacy_standards_path
        self.legacy_unit_weights_path = legacy_unit_weights_path
        self.session_timeout = session_timeout

        self._state_lock = threading.RLock()   # guards the active tuple (fast reads)
        self._switch_lock = threading.Lock()   # serialises activations
        self._exec_lock = threading.RLock()    # THE model lifecycle gate
        self._ready_event = threading.Event()

        self._generation = 0
        self._package: Optional[ModelPackage] = None
        self._profile: Optional[PackageProfile] = None
        self._adapter: Optional[ModelAdapter] = None
        self._packages: Dict[str, DiscoveredPackage] = {}
        self._last_error: Optional[str] = None
        # Is the ACTIVE model unusable right now?  Distinct from the reason the
        # last switch attempt failed: after a successful rollback the previous
        # model is perfectly healthy, and reporting the failed target's error as
        # the current model's error would read as a broken system.
        self._model_error: Optional[str] = None
        self._last_switch_error: Optional[str] = None
        self._loading = False
        self._compatibility: Dict[str, Any] = {}
        # Called inside the lifecycle gate right after a successful publish, so
        # application state derived from the old package is cleared before any
        # reader can observe "new package + old counts".
        self._switch_listeners: List[Callable[["ActiveState"], None]] = []

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
        a running kiosk should make it selectable without restarting the unit.
        Discovery is JSON-only, so the extra scan is cheap.
        """
        if not self._packages or package_id not in self._packages:
            self.discover(force=True)
        return self._packages.get(package_id)

    # ── active state ──────────────────────────────────────────────────────

    def state(self) -> ActiveState:
        with self._state_lock:
            return ActiveState(self._generation, self._package, self._profile,
                               self._adapter, self._model_error)

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

    @property
    def model_error(self) -> Optional[str]:
        """Why the ACTIVE model cannot be used — None when it is healthy."""
        with self._state_lock:
            return self._model_error

    #: kept for callers written against the previous name
    @property
    def fatal_error(self) -> Optional[str]:
        return self.model_error

    @property
    def last_switch_error(self) -> Optional[str]:
        """Why the most recent switch ATTEMPT failed — survives a rollback."""
        with self._state_lock:
            return self._last_switch_error

    def add_switch_listener(self, listener: Callable[["ActiveState"], None]) -> None:
        """Register a callback run inside the gate on every successful switch."""
        with self._state_lock:
            self._switch_listeners.append(listener)

    def _notify_switch(self, state: "ActiveState") -> None:
        with self._state_lock:
            listeners = list(self._switch_listeners)
        for listener in listeners:
            try:
                listener(state)
            except Exception as exc:  # noqa: BLE001 - a listener must not undo a switch
                logger.warning("[models] switch listener failed: %s", exc)

    @property
    def loading(self) -> bool:
        with self._state_lock:
            return self._loading

    @property
    def compatibility(self) -> Dict[str, Any]:
        with self._state_lock:
            return dict(self._compatibility)

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Block until the active model is loaded, or the timeout expires."""
        if self.state().ready:
            return True
        self._ready_event.wait(timeout=timeout)
        return self.state().ready

    # ── the lifecycle gate ────────────────────────────────────────────────

    @contextmanager
    def _gate(self, timeout: Optional[float] = None, what: str = "operation"):
        wait = self.session_timeout if timeout is None else timeout
        acquired = self._exec_lock.acquire(timeout=wait)
        if not acquired:
            raise ModelBusyError(
                "model is busy — %s timed out after %.0fs waiting for the "
                "lifecycle gate" % (what, wait))
        try:
            yield
        finally:
            self._exec_lock.release()

    @contextmanager
    def inference_session(self, timeout: Optional[float] = None):
        """Pin the active package for a whole inference transaction.

        Everything model-dependent — running the model, computing counts,
        publishing state, registering classes, writing history — belongs inside
        the ``with`` block.  A package switch cannot begin until it closes, so
        no result can ever be published against a package that replaced the one
        it came from.

        Requires a READY model.  Accepting a merely-configured one would let an
        upload arriving during startup grab the gate and lazily load the model
        itself, skipping the compatibility and warmup checks that decide whether
        this package may be used at all.

        Readiness is checked BEFORE taking the gate as well as after: during a
        startup load the gate is held for many seconds, and a request should be
        told "still loading" straight away rather than queueing behind it and
        then succeeding as if nothing had happened.
        """
        self._require_ready()
        with self._gate(timeout, what="inference"):
            state = self._require_ready()
            yield InferenceSession(self, state)

    def _require_ready(self) -> ActiveState:
        """Raise unless the active model is loaded, verified, and usable."""
        state = self.state()
        if not state.configured:
            raise ModelNotReadyError(
                "no active model package (%s)" % (self.last_error or "not configured"),
                loading=False, detail=self.last_error)
        if state.model_error:
            raise ModelNotReadyError("model is not usable: %s" % state.model_error,
                                     loading=False, detail=state.model_error)
        if not state.ready:
            loading = self.loading
            raise ModelNotReadyError(
                "model is still loading" if loading else "model is not loaded",
                loading=loading, detail=self.last_error)
        return state

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
        if not entry.package.model_available():
            raise ModelManagerError(
                "model package '%s' has no model file to load: %s"
                % (package_id, entry.package.model_error()))
        return entry.package

    def _load_profile_for(self, package: ModelPackage) -> PackageProfile:
        return load_profile(
            package,
            self.profiles_dir,
            legacy_package_id=self.legacy_package_id,
            legacy_standards_path=self.legacy_standards_path,
            legacy_unit_weights_path=self.legacy_unit_weights_path,
        )

    def _build_compatibility(self, profile: PackageProfile,
                             adapter: ModelAdapter) -> Dict[str, Any]:
        """Compare what the model can recognise with what the package expects."""
        info: ModelInfo = adapter.model_info
        model_classes = [str(c) for c in (info.class_names or [])]
        has_list = bool(model_classes)
        model_set = set(model_classes)
        standards = profile.standards
        class_weights = profile.class_weights
        positive = {cls for cls, qty in standards.items() if int(qty or 0) > 0}

        return {
            "has_model_class_list": has_list,
            "model_classes": len(model_classes),
            "model_classes_missing_weight": (
                sorted(model_set - set(class_weights)) if has_list else []),
            "standards_not_in_model": (
                sorted(positive - model_set) if has_list else []),
            "unused_class_weights": (
                sorted(set(class_weights) - model_set) if has_list else []),
            "standards_missing_class_weight": sorted(
                cls for cls in positive if not _usable_weight(class_weights.get(cls))),
        }

    def _assert_compatible(self, package: ModelPackage, diagnostics: Dict[str, Any]) -> None:
        """Refuse an activation the model can never satisfy.

        A standard that names a class the model has no output for means the
        tray can never be reported complete — better to fail the switch loudly
        than to run an inventory that is structurally incapable of passing.
        Only enforced when the adapter published a full class list.
        """
        if not diagnostics.get("has_model_class_list"):
            return
        missing = diagnostics.get("standards_not_in_model") or []
        if missing:
            raise ModelManagerError(
                "model for package '%s' cannot recognise expected instrument(s): %s"
                % (package.id, ", ".join(missing)))

    def _publish(self, package: ModelPackage, profile: PackageProfile,
                 adapter: ModelAdapter, compatibility: Dict[str, Any]) -> int:
        with self._state_lock:
            self._package = package
            self._profile = profile
            self._adapter = adapter
            self._compatibility = dict(compatibility)
            self._generation += 1
            self._last_error = None
            self._model_error = None
            generation = self._generation
        if adapter.loaded:
            self._ready_event.set()
        else:
            self._ready_event.clear()
        return generation

    def _restore(self, package: Optional[ModelPackage], profile: Optional[PackageProfile],
                 adapter: Optional[ModelAdapter]) -> None:
        """Bring the previous package back into service after a failed switch.

        Called with the lifecycle gate held.  If the old model refuses to load
        again the manager goes into an explicit unusable state rather than
        reporting an operational system that cannot actually run.
        """
        if adapter is None or package is None:
            return
        try:
            adapter.load()
            adapter.warmup()      # lenient: the model already loaded once
            with self._state_lock:
                self._package = package
                self._profile = profile
                self._adapter = adapter
                # The failed target's error belongs to the switch, not to this
                # model — it is healthy again and must not look broken.
                self._model_error = None
            self._ready_event.set()
            logger.info("[models] rolled back to package '%s'", package.id)
        except Exception as exc:  # noqa: BLE001
            message = ("rollback failed: package '%s' could not be reloaded: %s"
                       % (package.id, exc))
            with self._state_lock:
                self._model_error = message
                self._last_error = message
            self._ready_event.clear()
            logger.error("[models] %s", message)

    def activate(self, package_id: str, *, warmup: bool = True,
                 strict_warmup: bool = True, timeout: Optional[float] = None) -> ActiveState:
        """Switch the active package.  All-or-nothing.

        The old model is released BEFORE the new one is loaded, so the two are
        never resident at the same time.  If anything then fails, the old model
        is loaded again and stays in service.
        """
        with self._switch_lock:
            current = self.state()
            if current.ready and current.package_id == package_id:
                logger.info("[models] package '%s' already active", package_id)
                return current

            # Validation and profile loading are pure JSON work — done outside
            # the gate so a slow inference does not delay reporting a bad id.
            package = self._resolve_for_activation(package_id)
            try:
                adapter_cls = get_adapter_class(package.adapter)
            except UnknownAdapterError as exc:
                raise ModelManagerError(str(exc))
            profile = self._load_profile_for(package)
            new_adapter = adapter_cls(package)   # object only; loads nothing

            logger.info("[models] activating package '%s' (adapter=%s)",
                        package.id, package.adapter)

            with self._gate(timeout, what="package switch"):
                old_adapter = self._adapter
                old_package = self._package
                old_profile = self._profile
                with self._state_lock:
                    self._loading = True
                self._ready_event.clear()
                # 1. Free the old model FIRST — never two heavy models resident.
                #    Strict: if teardown fails the weights may still be held, so
                #    loading a replacement on top of them is not a risk worth
                #    taking on a 4 GB board.  Abort without touching the target.
                if old_adapter is not None:
                    try:
                        old_adapter.unload(strict=True)
                    except Exception as exc:  # noqa: BLE001
                        message = ("cannot switch to '%s': releasing the current model "
                                   "failed (%s) — refusing to load a second model"
                                   % (package.id, exc))
                        with self._state_lock:
                            self._last_error = message
                            self._last_switch_error = message
                            self._loading = False
                        if old_adapter.loaded:
                            self._ready_event.set()
                        logger.error("[models] %s", message)
                        raise ModelManagerError(message)

                try:
                    # 2. load the replacement
                    new_adapter.load()
                    # 3. warmup is the verification gate for a runtime switch:
                    #    reporting success for a model that cannot run would be
                    #    discovered mid-inventory instead of now.
                    if warmup:
                        new_adapter.warmup(strict=strict_warmup)
                    # 4. the model must be able to recognise what is expected
                    compatibility = self._build_compatibility(profile, new_adapter)
                    self._assert_compatible(package, compatibility)
                except Exception as exc:  # noqa: BLE001 - any failure rolls back
                    message = ("failed to activate package '%s' (%s): %s"
                               % (package.id, package.adapter, exc))
                    try:
                        new_adapter.retire()
                    except Exception:  # noqa: BLE001 - best-effort cleanup
                        pass
                    self._restore(old_package, old_profile, old_adapter)
                    with self._state_lock:
                        self._last_error = message
                        self._last_switch_error = message
                        self._loading = False
                    logger.error("[models] %s", message)
                    raise ModelManagerError(message)

                generation = self._publish(package, profile, new_adapter, compatibility)
                with self._state_lock:
                    self._loading = False
                    self._last_switch_error = None
                # The outgoing adapter is retired, not merely unloaded, so a
                # stray reference can never resurrect a second resident model.
                if old_adapter is not None and old_adapter is not new_adapter:
                    try:
                        old_adapter.retire()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[models] error retiring previous adapter: %s", exc)
                # Still inside the gate: application state derived from the old
                # package is cleared before any reader can see the new package.
                self._notify_switch(self.state())

            logger.info("[models] package '%s' active — generation=%d", package.id, generation)
            return self.state()

    def bootstrap(self, package_id: str, *, background: bool = True) -> Optional[ActiveState]:
        """Startup activation.

        The manifest and profile are published synchronously (cheap — JSON
        only) so /standards and the UI have data immediately, while the model
        load runs in the background exactly like the old detector warmup thread.
        The background load takes the same lifecycle gate as everything else,
        so it can never end up loading in parallel with a runtime switch.

        Until the load finishes ``ActiveState.ready`` is False and /status
        reports ``model_loading``, so nothing claims the model is usable yet.
        """
        with self._switch_lock:
            try:
                package = self._resolve_for_activation(package_id)
                adapter_cls = get_adapter_class(package.adapter)
                profile = self._load_profile_for(package)
            except (ModelManagerError, UnknownAdapterError) as exc:
                with self._state_lock:
                    self._last_error = str(exc)
                logger.error("[models] startup package '%s' unusable: %s", package_id, exc)
                return None
            except Exception as exc:  # noqa: BLE001
                with self._state_lock:
                    self._last_error = str(exc)
                logger.error("[models] startup profile for '%s' failed: %s", package_id, exc)
                return None

            adapter = adapter_cls(package)
            with self._state_lock:
                self._loading = True
            generation = self._publish(package, profile, adapter, {})
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
        """Load the startup model under the same gate as everything else.

        Startup applies exactly the same correctness bar as a runtime switch:
        load, then compatibility, then warmup.  Any of them failing leaves the
        package *configured* (so the UI can show which one is broken) but not
        *ready*, and nothing may infer against it.  Two different standards for
        the same package would mean a unit that boots into a state a switch
        would have refused.
        """
        def _fail(message: str, *, warmup: bool = False) -> None:
            logger.error("[models] %s", message)
            with self._state_lock:
                self._last_error = message
                self._model_error = message
            self._ready_event.clear()

        try:
            with self._gate(what="startup load"):
                # A runtime switch may have overtaken us while we waited for the
                # gate; loading now would resurrect a package that is no longer
                # active and put two models in memory.
                if self.generation != generation:
                    logger.info("[models] startup load of '%s' superseded by generation %d",
                                package_id, self.generation)
                    return
                adapter = self.active_adapter
                package = self.active_package
                profile = self.active_profile
                if adapter is None or package is None or profile is None:
                    return

                try:
                    adapter.load()
                except Exception as exc:  # noqa: BLE001
                    _fail("startup load of '%s' failed: %s" % (package_id, exc))
                    return

                compatibility = self._build_compatibility(profile, adapter)
                with self._state_lock:
                    self._compatibility = compatibility
                try:
                    self._assert_compatible(package, compatibility)
                except ModelManagerError as exc:
                    _fail("startup package '%s' is incompatible: %s" % (package_id, exc))
                    return

                # Warmup runs a real inference at the production image size.  If
                # that fails the model demonstrably cannot run, so claiming
                # model_ready would be a lie the operator finds out mid-inventory.
                warmup_error = adapter.warmup(strict=False)
                if warmup_error:
                    _fail("startup warmup of '%s' failed: %s" % (package_id, warmup_error),
                          warmup=True)
                    return

                self._ready_event.set()
                logger.info("[models] startup package '%s' ready", package_id)
        finally:
            with self._state_lock:
                self._loading = False

    # ── profile edits ─────────────────────────────────────────────────────

    def _validate_standards_edit(self, values: Dict[str, Any],
                                 profile: PackageProfile,
                                 adapter: Optional[ModelAdapter]) -> None:
        """Reject an expectation the active model could never satisfy.

        Checked at edit time rather than at the next switch: an operator who
        asks for an instrument this model cannot detect, or one with no unit
        weight, has configured a tray that can never be reported complete, and
        should find out while they are still looking at the screen.

        A quantity of 0 is always allowed — that is how an unused class is
        turned off.
        """
        class_weights = profile.class_weights
        model_classes = None
        if adapter is not None:
            names = adapter.model_info.class_names
            if names:
                model_classes = set(str(n) for n in names)

        unknown: List[str] = []
        unpriced: List[str] = []
        for cls, qty in values.items():
            try:
                quantity = int(qty or 0)
            except (TypeError, ValueError):
                continue
            if quantity <= 0:
                continue
            if model_classes is not None and cls not in model_classes:
                unknown.append(cls)
            elif not _usable_weight(class_weights.get(cls)):
                unpriced.append(cls)

        problems = []
        if unknown:
            problems.append("目前模型無法辨識：%s" % "、".join(sorted(unknown)))
        if unpriced:
            problems.append("尚未設定單重：%s" % "、".join(sorted(unpriced)))
        if problems:
            raise ProfileValidationError(
                "無法設定標準數量 — " + "；".join(problems),
                invalid=sorted(set(unknown) | set(unpriced)))

    def update_active_profile(self, kind: str, values: Dict[str, Any], *,
                              package_id: Optional[str] = None) -> Dict[str, Any]:
        """Edit the active package's profile, refusing cross-package writes.

        ``package_id`` is what the client believed was active when it composed
        the edit.  A debounced standards edit can easily arrive after the
        operator has already switched packages; without this check the
        orthopaedic numbers would land in the obstetric profile.
        """
        with self._state_lock:
            active_id = self._package.id if self._package else None
            if package_id is not None and package_id != active_id:
                raise PackageMismatchError(package_id, active_id)
            profile = self._profile
            if profile is None:
                raise ModelManagerError("no active model package — cannot edit %s" % kind)
            if kind == "standards":
                self._validate_standards_edit(values, profile, self._adapter)
                return profile.update_standards(values)
            if kind == "unit_weights":
                return profile.update_unit_weights(values)
            raise ValueError("unknown profile section %r" % kind)

    # ── inference (legacy convenience) ────────────────────────────────────

    def infer(self, image: Any, conf: Optional[float] = None,
              imgsz: Optional[int] = None) -> InferenceResult:
        """One-shot inference.

        Prefer ``inference_session()`` whenever the result is going to be
        published anywhere — this helper pins the package only for the model
        call itself, not for the publication that follows.
        """
        with self.inference_session() as session:
            return session.infer(image, conf=conf, imgsz=imgsz)

    # ── introspection ─────────────────────────────────────────────────────

    def model_info(self) -> Dict[str, Any]:
        state = self.state()
        if not state.configured or state.package is None:
            return {
                "active": False,
                "ready": False,
                "loading": self.loading,
                "generation": state.generation,
                "package_id": None,
                "error": self.last_error,
                "model_error": self.model_error,
                "last_switch_error": self.last_switch_error,
                "fatal_error": self.model_error,   # legacy alias
                "packages_dir": str(self.packages_dir),
            }
        adapter = state.adapter
        info: ModelInfo = (adapter.model_info if adapter is not None
                           else ModelInfo(package_id=state.package_id))
        payload = info.to_dict()
        payload.update({
            "active": True,
            "ready": state.ready,
            "loading": self.loading,
            "generation": state.generation,
            "package": state.package.summary(),
            "standards_count": len(state.standards),
            "class_weights_count": len(state.class_weights),
            "compatibility": self.compatibility,
            "error": self.last_error,
            # Whether THIS model is unusable, versus why the last switch failed.
            "model_error": self.model_error,
            "last_switch_error": self.last_switch_error,
            "fatal_error": self.model_error,   # legacy alias
        })
        return payload

    def shutdown(self) -> None:
        with self._gate(timeout=10.0, what="shutdown"):
            with self._state_lock:
                adapter = self._adapter
                self._adapter = None
                self._package = None
                self._profile = None
            self._ready_event.clear()
            if adapter is not None:
                try:
                    adapter.retire()
                except Exception:  # noqa: BLE001
                    pass
