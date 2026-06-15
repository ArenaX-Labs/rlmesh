"""RLMesh Python SDK."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

from . import _rlmesh as _rlmesh
from . import adapters as adapters
from . import models as models
from . import recipes as recipes
from . import serving as serving
from . import spaces as spaces
from . import types as types
from ._native import (
    Model,
    RemoteEnv,
    RemoteVectorEnv,
    SandboxEnv,
    SandboxModel,
    SandboxVectorEnv,
)
from ._rlmesh import ServeOptions, Tensor
from .models import ModelRecipe
from .recipes import EnvRecipe, Recipe
from .recipes import make as make
from .recipes import register as _register_env
from .recipes.authoring.model import is_model_recipe as _is_model_recipe
from .sandbox import ExportResult, export
from .server import EnvServer

try:
    __version__ = package_version("rlmesh")
except PackageNotFoundError:
    __version__ = str(getattr(_rlmesh, "__version__", "0+unknown"))

__doc__ = _rlmesh.__doc__


def register(source: Any, **kwargs: Any) -> Any:
    """Register an env or a model by name -- the single entry, dispatched by kind.

    Routes to the env registry (``EnvRecipe``/``Recipe``/``gym=``/``factory=``) or
    the model registry (``ModelRecipe``/``hf=``/``load=``/``spec=``). The keyword
    sets are disjoint, so dispatch is unambiguous.
    """
    is_model = _is_model_recipe(source) or any(
        key in kwargs for key in ("hf", "load", "spec")
    )
    if is_model:
        return models.register(source, **kwargs)
    # Dynamic kind dispatch: source/kwargs are Any, so the overloaded env register
    # cannot be matched statically.
    return _register_env(source, **kwargs)  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType]


__all__ = [
    "EnvRecipe",
    "EnvServer",
    "ExportResult",
    "Model",
    "ModelRecipe",
    "Recipe",
    "RemoteEnv",
    "RemoteVectorEnv",
    "SandboxEnv",
    "SandboxModel",
    "SandboxVectorEnv",
    "ServeOptions",
    "Tensor",
    "__version__",
    "adapters",
    "export",
    "make",
    "models",
    "recipes",
    "register",
    "serving",
    "spaces",
    "types",
]
