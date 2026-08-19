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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._describe import describe as describe
    from ._describe import describe_json as describe_json

from . import _rlmesh as _rlmesh
from . import adapters as adapters
from . import params as params
from . import spaces as spaces
from . import specs as specs
from . import types as types
from ._authoring import EnvFactory
from ._metadata import sanitize_metadata
from ._models import (
    NO_ADAPTER,
    RANDOM_SAMPLE,
    EpisodeResult,
    Reader,
    RunHooks,
    RunResult,
    Session,
    StepEvent,
    View,
    run,
    session,
)
from ._native import (
    Model,
    RemoteEnv,
    RemoteModel,
    RemoteVectorEnv,
    SandboxEnv,
    SandboxModel,
    SandboxVectorEnv,
)
from ._peer_info import register_python_peer_info as _register_python_peer_info
from ._rlmesh import (
    DESCRIBE_METADATA_KEY,
    DESCRIBE_SCHEMA_VERSION,
    ServeOptions,
    Tensor,
)
from ._sandbox import SandboxBuild, SandboxRuntime
from ._server import EnvServer
from ._variants import Variant
from .params import Param, ParamSpec, Vector
from .recorder import Recorder

try:
    __version__ = _package_version("rlmesh")
except _PackageNotFoundError:
    __version__ = str(getattr(_rlmesh, "__version__", "0+unknown"))

__build__ = str(getattr(_rlmesh, "__build__", "unknown"))

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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DESCRIBE_METADATA_KEY",
    "DESCRIBE_SCHEMA_VERSION",
    "NO_ADAPTER",
    "RANDOM_SAMPLE",
    "EnvFactory",
    "EnvServer",
    "EpisodeResult",
    "Model",
    "Param",
    "ParamSpec",
    "Reader",
    "Recorder",
    "RemoteEnv",
    "RemoteModel",
    "RemoteVectorEnv",
    "RunHooks",
    "RunResult",
    "SandboxBuild",
    "SandboxEnv",
    "SandboxModel",
    "SandboxRuntime",
    "SandboxVectorEnv",
    "ServeOptions",
    "Session",
    "StepEvent",
    "Tensor",
    "Variant",
    "Vector",
    "View",
    "__build__",
    "__version__",
    "adapters",
    "describe",
    "describe_json",
    "params",
    "run",
    "sanitize_metadata",
    "session",
    "spaces",
    "specs",
    "types",
]
