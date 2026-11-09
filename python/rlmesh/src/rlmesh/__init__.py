"""RLMesh Python SDK."""

import sys as _sys

# The wire value encoding is little-endian and numpy/torch `frombuffer` are
# native-endian (torch/dlpack admit no byte-order override), so a big-endian host
# would silently byteswap every tensor leaf. Fail fast rather than corrupt.
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
    session,  # noqa: F401 -- a 0.1 name, kept only to raise
)
from ._native import (
    Model,
    RemoteEnv,
    RemoteModel,  # noqa: F401 -- a 0.1 name, kept only to raise
    RemoteVectorEnv,
    SandboxEnv,
    SandboxModel,
    SandboxVectorEnv,
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
from ._upgrade import Session  # noqa: F401 -- a 0.1 name, kept only to raise
from ._variants import Variant
from .params import Param, ParamSpec, Vector
from .recorder import Recorder

try:
    __version__ = _package_version("rlmesh")
except _PackageNotFoundError:
    __version__ = str(getattr(_rlmesh, "__version__", "0+unknown"))

__doc__ = _rlmesh.__doc__

# Stamp this Python runtime's identity onto the native handshake PeerInfo so a
# python-hosted env/model peer reports its real runtime for debugging. Advisory
# only and best-effort; never raises.
_register_python_peer_info()


# describe/describe_json load lazily (PEP 562): an eager `from ._describe
# import ...` here puts rlmesh._describe in sys.modules during the package
# import that `python -m rlmesh._describe` performs first, so runpy would then
# execute a second copy as __main__ and warn about unpredictable behaviour.
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


# __getattr__ alone leaves the lazy names out of dir(), so a REPL or IDE never
# completes them; __all__ is the public list, globals() the eagerly bound one.
# The 0.1 names 0.2 removed (Session, session, RemoteModel) stay importable,
# out of __all__, only to raise an error naming their replacement.
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
