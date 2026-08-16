"""Model-agnostic inference layer.

Import order matters only in that ``registry`` must be importable without
pulling in any ML runtime — package validation depends on that.
"""

from app.inference.types import (  # noqa: F401
    BBox,
    Detection,
    InferenceResult,
    ModelInfo,
    counts_from_detections,
)
from app.inference.base import AdapterError, ModelAdapter  # noqa: F401
from app.inference.registry import (  # noqa: F401
    UnknownAdapterError,
    available_adapters,
    get_adapter_class,
    has_adapter,
    register_adapter,
    unregister_adapter,
)
from app.inference.package import (  # noqa: F401
    DiscoveredPackage,
    ModelPackage,
    ModelPackageError,
    discover_packages,
    load_manifest,
)
from app.inference.profile import PackageProfile, load_profile  # noqa: F401
from app.inference.manager import (  # noqa: F401
    ActiveState,
    ModelManager,
    ModelManagerError,
)
