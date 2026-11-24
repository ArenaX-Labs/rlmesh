"""RLMesh Python SDK."""

import sys as _sys

# Wire encoding is little-endian; native-endian frombuffer on a big-endian host
# would byteswap every tensor leaf.
if _sys.byteorder != "little":
    raise RuntimeError(
        "rlmesh requires a little-endian host: the wire value encoding is "
        "little-endian, so a big-endian host would silently byteswap tensors."
    )

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from ._describe import describe as describe
    from ._describe import describe_json as describe_json

from . import _rlmesh as _rlmesh
from . import adapters as adapters
from . import managed as managed
from . import params as params
from . import spaces as spaces
from . import specs as specs
from . import types as types
from ._authoring import EnvFactory, trial_index
from ._editions import WorkflowEditionWarning, current_workflow_edition
from ._metadata import sanitize_metadata
from ._models import (
    NO_ADAPTER,
    RANDOM_SAMPLE,
    BatchOptions,
    Binding,
    EpisodeResult,
    EvalReport,
    FrameSources,
    Interrupted,
    Reader,
    RunContext,
    RunHooks,
    RunResult,
    StepEvent,
    TelemetryRow,
    View,
    run,
)
from ._models import (
    session as session,
)
from ._native import (
    Model,
    RemoteEnv,
    RemoteVectorEnv,
    SandboxEnv,
    SandboxModel,
    SandboxVectorEnv,
)
from ._native import (
    RemoteModel as RemoteModel,
)
from ._peer_info import register_python_peer_info as _register_python_peer_info
from ._rlmesh import (
    DESCRIBE_METADATA_KEY,
    DESCRIBE_SCHEMA_VERSION,
    ENV_RESET_OPTIONS_KEY,
    BuildInfo,
    EnvironmentException,
    ModelException,
    ProtocolException,
    RecoverableEnvironmentException,
    RLMeshException,
    ServeOptions,
    Tensor,
    build_info,
    predict_seed,
)
from ._sandbox import SandboxBuild, SandboxRuntime
from ._server import EnvServer
from ._upgrade import Session as Session
from ._variants import Variant
from .params import Param, ParamSpec, Vector
from .recorder import Recorder

try:
    __version__ = _package_version("rlmesh")
except _PackageNotFoundError:
    __version__ = str(getattr(_rlmesh, "__version__", "0+unknown"))

__doc__ = _rlmesh.__doc__

_register_python_peer_info()


# Lazy: an eager import puts rlmesh._describe in sys.modules before
# `python -m rlmesh._describe` runs, so runpy would execute a second copy.
def __getattr__(name: str) -> object:
    if name in ("describe", "describe_json"):
        from . import _describe

        return getattr(_describe, name)
    if name == "__build__":
        raise AttributeError(
            "rlmesh.__build__ is gone in rlmesh 0.2: "
            "use rlmesh.build_info().workflow_edition"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "DESCRIBE_METADATA_KEY",
    "DESCRIBE_SCHEMA_VERSION",
    "ENV_RESET_OPTIONS_KEY",
    "NO_ADAPTER",
    "RANDOM_SAMPLE",
    "BatchOptions",
    "Binding",
    "BuildInfo",
    "EnvFactory",
    "EnvServer",
    "EnvironmentException",
    "EpisodeResult",
    "EvalReport",
    "FrameSources",
    "Interrupted",
    "Model",
    "ModelException",
    "Param",
    "ParamSpec",
    "ProtocolException",
    "RLMeshException",
    "Reader",
    "Recorder",
    "RecoverableEnvironmentException",
    "RemoteEnv",
    "RemoteVectorEnv",
    "RunContext",
    "RunHooks",
    "RunResult",
    "SandboxBuild",
    "SandboxEnv",
    "SandboxModel",
    "SandboxRuntime",
    "SandboxVectorEnv",
    "ServeOptions",
    "StepEvent",
    "TelemetryRow",
    "Tensor",
    "Variant",
    "Vector",
    "View",
    "WorkflowEditionWarning",
    "__version__",
    "adapters",
    "build_info",
    "current_workflow_edition",
    "describe",
    "describe_json",
    "params",
    "predict_seed",
    "run",
    "sanitize_metadata",
    "spaces",
    "specs",
    "trial_index",
    "types",
]
