"""Adapter registry.

A manifest names its adapter explicitly (``"adapter": "ultralytics"``).  The
registry maps that name to a class.

Why names and not file extensions: ".onnx" is a serialization format, not an
inference contract.  Two ONNX models can have completely different output
tensors and post-processing.  Guessing an adapter from a file extension would
silently produce garbage counts, so it is never done.

Adapters are registered as *lazy loaders* so that listing or validating
packages never imports a heavy runtime (ultralytics pulls in torch).  The few
facts package validation needs — does this adapter read a model file? — are
registered as plain metadata alongside the loader, so they are answerable
without importing anything.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List, Optional, Type, Union

from app.inference.base import ModelAdapter

logger = logging.getLogger(__name__)

AdapterLoader = Union[Type[ModelAdapter], Callable[[], Type[ModelAdapter]]]


class UnknownAdapterError(KeyError):
    """Raised when a manifest names an adapter that is not registered."""

    def __init__(self, name: str, available: List[str]) -> None:
        self.name = name
        self.available = available
        super().__init__(
            "unknown adapter %r — registered adapters: %s"
            % (name, ", ".join(sorted(available)) or "(none)")
        )

    def __str__(self) -> str:  # KeyError repr-quotes its message otherwise
        return self.args[0]


class _AdapterSpec:
    """Loader plus the metadata that must be readable without importing it."""

    __slots__ = ("loader", "requires_model_file")

    def __init__(self, loader: AdapterLoader, requires_model_file: bool = True) -> None:
        self.loader = loader
        self.requires_model_file = requires_model_file


_lock = threading.RLock()
_registry: Dict[str, _AdapterSpec] = {}


def _normalise(name: str) -> str:
    return str(name).strip().lower()


def register_adapter(name: str, loader: AdapterLoader, *, replace: bool = False,
                     requires_model_file: Optional[bool] = None) -> None:
    """Register an adapter class, or a zero-arg callable returning one.

    ``requires_model_file`` defaults to the class attribute when a class is
    given, and to True for a lazy loader (which cannot be inspected without
    importing it).  Pass it explicitly for a lazy loader that needs no file.
    """
    key = _normalise(name)
    if not key:
        raise ValueError("adapter name must be a non-empty string")
    if requires_model_file is None:
        if isinstance(loader, type) and issubclass(loader, ModelAdapter):
            requires_model_file = bool(loader.requires_model_file)
        else:
            requires_model_file = True
    with _lock:
        if key in _registry and not replace:
            raise ValueError("adapter %r is already registered" % key)
        _registry[key] = _AdapterSpec(loader, requires_model_file)


def unregister_adapter(name: str) -> None:
    """Remove an adapter.  Used by tests; harmless if the name is absent."""
    with _lock:
        _registry.pop(_normalise(name), None)


def has_adapter(name: str) -> bool:
    """True if the name is registered.  Never imports the adapter module."""
    with _lock:
        return _normalise(name) in _registry


def available_adapters() -> List[str]:
    with _lock:
        return sorted(_registry)


def adapter_requires_model_file(name: str) -> bool:
    """Whether this adapter reads a model file from disk.

    Answered from registry metadata, so package validation and listing never
    import a runtime just to find out.  Unknown adapters are assumed to need
    one, which is the stricter answer.
    """
    with _lock:
        spec = _registry.get(_normalise(name))
    return True if spec is None else spec.requires_model_file


def get_adapter_class(name: str) -> Type[ModelAdapter]:
    """Resolve a name to a class, importing the runtime only now."""
    key = _normalise(name)
    with _lock:
        spec = _registry.get(key)
        if spec is None:
            raise UnknownAdapterError(name, list(_registry))
        loader = spec.loader
    if isinstance(loader, type) and issubclass(loader, ModelAdapter):
        return loader
    cls = loader()  # type: ignore[operator]
    if not (isinstance(cls, type) and issubclass(cls, ModelAdapter)):
        raise TypeError("adapter loader for %r did not return a ModelAdapter subclass" % name)
    with _lock:
        spec = _registry.get(key)
        if spec is not None:
            spec.loader = cls  # memoise the resolved class
            spec.requires_model_file = bool(cls.requires_model_file)
    return cls


def _load_ultralytics_adapter() -> Type[ModelAdapter]:
    from app.inference.adapters.ultralytics_adapter import UltralyticsAdapter

    return UltralyticsAdapter


def register_builtin_adapters() -> None:
    """Register the adapters shipped with the platform.  Idempotent."""
    with _lock:
        if "ultralytics" not in _registry:
            _registry["ultralytics"] = _AdapterSpec(_load_ultralytics_adapter, True)


register_builtin_adapters()
