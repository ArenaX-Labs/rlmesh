"""RLMesh Python SDK."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from . import _rlmesh as _rlmesh
from . import recipes as recipes
from . import serving as serving
from . import spaces as spaces
from . import types as types
from ._native import Model, RemoteEnv, RemoteVectorEnv
from ._rlmesh import ServeOptions, Tensor
from .recipes import EnvRecipe, Recipe, make, register
from .server import EnvServer

try:
    __version__ = package_version("rlmesh")
except PackageNotFoundError:
    __version__ = str(getattr(_rlmesh, "__version__", "0+unknown"))

__doc__ = _rlmesh.__doc__

__all__ = [
    "EnvRecipe",
    "EnvServer",
    "Model",
    "Recipe",
    "RemoteEnv",
    "RemoteVectorEnv",
    "ServeOptions",
    "Tensor",
    "__version__",
    "make",
    "recipes",
    "register",
    "serving",
    "spaces",
    "types",
]
