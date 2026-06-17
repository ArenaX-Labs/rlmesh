"""RLMesh Python SDK."""

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
from ._rlmesh import ServeOptions, Tensor
from ._server import EnvServer

try:
    __version__ = package_version("rlmesh")
except PackageNotFoundError:
    __version__ = str(getattr(_rlmesh, "__version__", "0+unknown"))

__doc__ = _rlmesh.__doc__


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
