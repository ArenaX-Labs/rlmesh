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
    # The distribution metadata is not visible (e.g. a PyInstaller/zipapp
    # bundle or a vendored site-packages copy without dist-info). The compiled
    # extension does not expose a version, so report an explicit "unknown"
    # marker instead of a misleading concrete version that downstream version
    # gating could trust.
    __version__ = "0+unknown"

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
