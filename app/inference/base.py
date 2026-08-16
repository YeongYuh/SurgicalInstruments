"""ModelAdapter — the boundary between the platform and an ML runtime.

Everything runtime-specific (ultralytics, onnxruntime session handling, a
TensorRT engine, a bespoke post-processor) lives inside an adapter subclass.
The platform only ever sees InferenceResult.
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


class ModelAdapter(ABC):
    """Base class for all model adapters.

    Lifecycle::

        adapter = SomeAdapter(package)
        adapter.load()      # may be slow (disk + runtime session init)
        adapter.warmup()    # optional, best-effort
        adapter.infer(img)  # many times
        adapter.unload()

    ``infer()`` must be safe to call without an explicit ``load()`` — the base
    class provides ``_ensure_loaded()`` which loads lazily under a lock so a
    cold first request works exactly like the pre-refactor detector did.
    """

    #: registry name; must match the ``adapter`` field of a manifest
    name: str = "base"

    #: set False by adapters that do not read a model file from disk
    requires_model_file: bool = True

    def __init__(self, package: "ModelPackage") -> None:
        self.package = package
        self.options = dict(package.adapter_options)
        self._loaded = False
        self._load_lock = threading.RLock()
        self._warmup_error: Optional[str] = None

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

    # ── public API ────────────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        with self._load_lock:
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

    def warmup(self) -> Optional[str]:
        """Best-effort warmup.

        Returns None on success, or the error string on failure.  A warmup
        failure is deliberately NOT fatal: the model already loaded, and
        refusing to activate over a warmup hiccup would be worse than a slow
        first inference.  The error is surfaced through ``model_info``.
        """
        try:
            self._ensure_loaded()
            t0 = time.perf_counter()
            self._do_warmup()
            self._warmup_error = None
            logger.info(
                "[adapter:%s] warmup done package=%s in %.0f ms",
                self.name, self.package.id, (time.perf_counter() - t0) * 1000,
            )
            return None
        except Exception as exc:  # noqa: BLE001 - warmup must never crash startup
            self._warmup_error = str(exc)
            logger.warning(
                "[adapter:%s] warmup failed package=%s: %s",
                self.name, self.package.id, exc,
            )
            return str(exc)

    def infer(self, image: Any, conf: Optional[float] = None,
              imgsz: Optional[int] = None) -> InferenceResult:
        self._ensure_loaded()
        effective_conf = self.package.confidence if conf is None else conf
        effective_imgsz = self.package.image_size if imgsz is None else imgsz
        t0 = time.perf_counter()
        result = self._do_infer(image, effective_conf, effective_imgsz)
        if not result.inference_ms:
            result.inference_ms = (time.perf_counter() - t0) * 1000
        if result.model_info is None:
            result.model_info = self.model_info
        return result

    def unload(self) -> None:
        with self._load_lock:
            if not self._loaded:
                return
            try:
                self._do_unload()
            except Exception as exc:  # noqa: BLE001 - teardown must never raise
                logger.warning("[adapter:%s] unload error: %s", self.name, exc)
            finally:
                self._loaded = False
                logger.info("[adapter:%s] unloaded package=%s", self.name, self.package.id)

    # alias — some callers think in terms of file handles
    close = unload

    @property
    def model_info(self) -> ModelInfo:
        return self._describe()
