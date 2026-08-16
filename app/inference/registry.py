"""Adapter registry.

A manifest names its adapter explicitly (``"adapter": "ultralytics"``).  The
registry maps that name to a class.

Why names and not file extensions: ".onnx" is a serialization format, not an
inference contract.  Two ONNX files can have completely different output
tensors and post-processing.  Guessing an adapter from a file extension would
silently produce garbage counts, so it is never done.

Adapters are registered as *lazy loaders* so that listing or validating
packages never imports a heavy runtime (ultralytics pulls in torch).
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List, Type, Union

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


_lock = threading.RLock()
_registry: Dict[str, AdapterLoader] = {}


def register_adapter(name: str, loader: AdapterLoader, *, replace: bool = False) -> None:
    """Register an adapter class, or a zero-arg callable returning one."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("adapter name must be a non-empty string")
    with _lock:
        if key in _registry and not replace:
            raise ValueError("adapter %r is already registered" % key)
        _registry[key] = loader


def unregister_adapter(name: str) -> None:
    """Remove an adapter.  Used by tests; harmless if the name is absent."""
    with _lock:
        _registry.pop(str(name).strip().lower(), None)


def has_adapter(name: str) -> bool:
    """True if the name is registered.  Never imports the adapter module."""
    with _lock:
        return str(name).strip().lower() in _registry


def available_adapters() -> List[str]:
    with _lock:
        return sorted(_registry)


def get_adapter_class(name: str) -> Type[ModelAdapter]:
    """Resolve a name to a class, importing the runtime only now."""
    key = str(name).strip().lower()
    with _lock:
        loader = _registry.get(key)
        if loader is None:
            raise UnknownAdapterError(name, list(_registry))
    if isinstance(loader, type) and issubclass(loader, ModelAdapter):
        return loader
    cls = loader()  # type: ignore[operator]
    if not (isinstance(cls, type) and issubclass(cls, ModelAdapter)):
        raise TypeError("adapter loader for %r did not return a ModelAdapter subclass" % name)
    with _lock:
        _registry[key] = cls  # memoise the resolved class
    return cls


def _load_ultralytics_adapter() -> Type[ModelAdapter]:
    from app.inference.adapters.ultralytics_adapter import UltralyticsAdapter

    return UltralyticsAdapter


def register_builtin_adapters() -> None:
    """Register the adapters shipped with the platform.  Idempotent."""
    with _lock:
        if "ultralytics" not in _registry:
            _registry["ultralytics"] = _load_ultralytics_adapter


register_builtin_adapters()
