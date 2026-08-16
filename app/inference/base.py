"""ModelAdapter — the boundary between the platform and an ML runtime.

Everything runtime-specific (ultralytics, onnxruntime session handling, a
TensorRT engine, a bespoke post-processor) lives inside an adapter subclass.
The platform only ever sees InferenceResult.

Concurrency contract
--------------------
An adapter holds a real model.  Nothing here assumes a runtime is thread-safe,
so ``load`` / ``infer`` / ``unload`` are serialised on one execution lock:

* two ``infer()`` calls never run concurrently on the same adapter (the camera
  worker and an operator upload can arrive at the same moment)
* ``unload()`` waits for an in-flight inference rather than pulling the model
  out from under it
* once unloading has begun, a new ``infer()`` is refused instead of queuing
  behind the teardown and then silently reloading the model

The ModelManager adds a second, transaction-level gate on top of this; see
``ModelManager.inference_session``.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from app.inference.types import InferenceResult, ModelInfo

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.inference.package import ModelPackage

logger = logging.getLogger(__name__)


class AdapterError(RuntimeError):
    """Raised when an adapter cannot load or run its model."""


class AdapterUnavailableError(AdapterError):
    """Raised when the adapter is being torn down or has been retired."""


class ModelAdapter(ABC):
    """Base class for all model adapters.

    Lifecycle::

        adapter = SomeAdapter(package)
        adapter.load()      # may be slow (disk + runtime session init)
        adapter.warmup()    # verification for a runtime switch
        adapter.infer(img)  # many times
        adapter.unload()    # reversible — the model can be loaded again
        adapter.retire()    # permanent — this adapter will never run again
    """

    #: registry name; must match the ``adapter`` field of a manifest
    name: str = "base"

    #: set False by adapters that do not read a model file from disk
    requires_model_file: bool = True

    def __init__(self, package: "ModelPackage") -> None:
        self.package = package
        self.options = dict(package.adapter_options)
        self._loaded = False
        # Guards the model itself: load / infer / unload never overlap.
        self._exec_lock = threading.RLock()
        self._closing = False
        self._retired = False
        self._warmup_error: Optional[str] = None
        self._inference_count = 0

    # ── subclass hooks ────────────────────────────────────────────────────

    @abstractmethod
    def _do_load(self) -> None:
        """Load the model.  Raise AdapterError (or anything) on failure."""

    @abstractmethod
    def _do_infer(self, image: Any, conf: float, imgsz: Optional[int]) -> InferenceResult:
        """Run inference on one BGR frame / image path and return a result."""

    def _do_unload(self) -> None:
        """Release runtime resources.  Default: nothing to do."""

    def _do_warmup(self) -> None:
        """Optional cheap dummy inference to pay the cold-start cost early."""

    def _describe(self) -> ModelInfo:
        """Adapter-specific model identity.  Override to add backend details."""
        pkg = self.package
        return ModelInfo(
            package_id=pkg.id,
            display_name=pkg.display_name,
            department=pkg.department,
            adapter=self.name,
            model_file=str(pkg.model_file) if pkg.model_file else "",
            model_file_exists=pkg.model_file_exists(),
            loaded=self._loaded,
            warmup_error=self._warmup_error,
        )

    # ── state ─────────────────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def retired(self) -> bool:
        return self._retired

    @property
    def warmup_error(self) -> Optional[str]:
        return self._warmup_error

    @property
    def inference_count(self) -> int:
        return self._inference_count

    def _check_available(self) -> None:
        if self._retired:
            raise AdapterUnavailableError(
                "adapter for package '%s' has been retired" % self.package.id)
        if self._closing:
            raise AdapterUnavailableError(
                "adapter for package '%s' is being unloaded" % self.package.id)

    # ── public API ────────────────────────────────────────────────────────

    def load(self) -> None:
        self._check_available()
        with self._exec_lock:
            self._check_available()
            if self._loaded:
                return
            t0 = time.perf_counter()
            self._do_load()
            self._loaded = True
            logger.info(
                "[adapter:%s] loaded package=%s in %.0f ms",
                self.name, self.package.id, (time.perf_counter() - t0) * 1000,
            )

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def warmup(self, strict: bool = False) -> Optional[str]:
        """Run a dummy inference to pay the cold-start cost.

        ``strict=True`` re-raises the failure.  A runtime package switch uses
        strict mode: reporting a switch as successful when the model cannot
        actually run would be a lie the operator only discovers mid-inventory.
        Startup warmup stays lenient — the model already loaded, and refusing to
        boot over a warmup hiccup is worse than a slow first inference.
        """
        try:
            with self._exec_lock:
                self._check_available()
                self._ensure_loaded()
                t0 = time.perf_counter()
                self._do_warmup()
            self._warmup_error = None
            logger.info(
                "[adapter:%s] warmup done package=%s in %.0f ms",
                self.name, self.package.id, (time.perf_counter() - t0) * 1000,
            )
            return None
        except Exception as exc:  # noqa: BLE001 - lenient mode must not crash startup
            self._warmup_error = str(exc)
            logger.warning(
                "[adapter:%s] warmup failed package=%s: %s",
                self.name, self.package.id, exc,
            )
            if strict:
                raise AdapterError(
                    "warmup failed for package '%s': %s" % (self.package.id, exc))
            return str(exc)

    def infer(self, image: Any, conf: Optional[float] = None,
              imgsz: Optional[int] = None) -> InferenceResult:
        # Checked before AND after acquiring: refusing up front avoids queuing
        # behind a teardown, and re-checking closes the window where unload()
        # set the flag while this thread was already waiting for the lock.
        self._check_available()
        with self._exec_lock:
            self._check_available()
            self._ensure_loaded()
            effective_conf = self.package.confidence if conf is None else conf
            effective_imgsz = self.package.image_size if imgsz is None else imgsz
            t0 = time.perf_counter()
            result = self._do_infer(image, effective_conf, effective_imgsz)
            self._inference_count += 1
        if not result.inference_ms:
            result.inference_ms = (time.perf_counter() - t0) * 1000
        if result.model_info is None:
            result.model_info = self.model_info
        return result

    def unload(self) -> None:
        """Release the model.  Reversible — ``load()`` can bring it back.

        Setting ``_closing`` before taking the lock turns away new inference
        immediately, then the lock acquisition waits for any in-flight one.
        """
        self._closing = True
        try:
            with self._exec_lock:
                if not self._loaded:
                    return
                try:
                    self._do_unload()
                except Exception as exc:  # noqa: BLE001 - teardown must never raise
                    logger.warning("[adapter:%s] unload error: %s", self.name, exc)
                finally:
                    self._loaded = False
                    logger.info("[adapter:%s] unloaded package=%s",
                                self.name, self.package.id)
        finally:
            self._closing = False

    def retire(self) -> None:
        """Unload permanently.  A retired adapter can never run again.

        Used for the outgoing adapter of a completed switch, so a stray
        reference can never resurrect a second resident model.
        """
        self.unload()
        self._retired = True

    # alias — some callers think in terms of file handles
    close = unload

    @property
    def model_info(self) -> ModelInfo:
        return self._describe()
