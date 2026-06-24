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

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from . import _rlmesh as _rlmesh
from . import adapters as adapters
from . import spaces as spaces
from . import types as types
from ._models import DELEGATED, EpisodeResult, RunResult, hf_load
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
from ._rlmesh import ServeOptions, Tensor
from ._server import EnvServer

try:
    __version__ = package_version("rlmesh")
except PackageNotFoundError:
    __version__ = str(getattr(_rlmesh, "__version__", "0+unknown"))

__doc__ = _rlmesh.__doc__

# Stamp this Python runtime's identity onto the native handshake PeerInfo so a
# python-hosted env/model peer reports its real runtime for debugging. Advisory
# only and best-effort; never raises.
_register_python_peer_info()


__all__ = [
    "DELEGATED",
    "EnvServer",
    "EpisodeResult",
    "Model",
    "RemoteEnv",
    "RemoteModel",
    "RemoteVectorEnv",
    "RunResult",
    "SandboxEnv",
    "SandboxModel",
    "SandboxVectorEnv",
    "ServeOptions",
    "Tensor",
    "__version__",
    "adapters",
    "hf_load",
    "spaces",
    "types",
]
