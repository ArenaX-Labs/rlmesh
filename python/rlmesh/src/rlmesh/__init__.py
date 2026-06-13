"""RLMesh Python SDK."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from . import _rlmesh as _rlmesh
from . import serving as serving
from . import spaces as spaces
from . import types as types
from ._native import Model, RemoteEnv, RemoteVectorEnv
from ._rlmesh import ServeOptions, Tensor
from .server import EnvServer

try:
    __version__ = package_version("rlmesh")
except PackageNotFoundError:
    __version__ = str(getattr(_rlmesh, "__version__", "0+unknown"))

__doc__ = _rlmesh.__doc__

__all__ = [
    "EnvServer",
    "Model",
    "RemoteEnv",
    "RemoteVectorEnv",
    "ServeOptions",
    "Tensor",
    "__version__",
    "serving",
    "spaces",
    "types",
]
