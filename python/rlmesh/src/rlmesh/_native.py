"""Dependency-free native RLMesh SDK classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypeVar, final

from ._client import RemoteEnvBase, RemoteModelBase, RemoteVectorEnvBase
from ._models.base import ModelBase
from ._sandbox import SandboxEnvBase, SandboxVectorEnvBase
from ._sandbox._model import SandboxModel
from ._value_conversion import ValueBridge, identity_bridge
from .types import Value

if TYPE_CHECKING:
    from typing_extensions import TypeVar as _DefaultTypeVar

    _ObsT = _DefaultTypeVar("_ObsT", default=Value)
    _ActT = _DefaultTypeVar("_ActT", default=Value)
else:
    _ObsT = TypeVar("_ObsT")
    _ActT = TypeVar("_ActT")


@final
class RemoteEnv(RemoteEnvBase[Value, Value]):
    """Dependency-free client for one remote environment endpoint.

    Observations and actions stay RLMesh-native values; see
    :class:`rlmesh.numpy.RemoteEnv` for NumPy leaves.
    """

    _bridge: ClassVar[ValueBridge] = identity_bridge


@final
class RemoteModel(RemoteModelBase[Value, Value]):
    """Dependency-free client for a model already served on an endpoint.

    Bind it to an environment with :func:`rlmesh.run` or :func:`rlmesh.session`.
    """

    _bridge: ClassVar[ValueBridge] = identity_bridge


@final
class RemoteVectorEnv(RemoteVectorEnvBase[Value, Value]):
    """Dependency-free client for a remote vector-environment endpoint.

    Observations and actions stay RLMesh-native values; see
    :class:`rlmesh.numpy.RemoteVectorEnv` for NumPy leaves.
    """

    _bridge: ClassVar[ValueBridge] = identity_bridge


class Model(ModelBase[_ObsT, _ActT]):
    """Dependency-free model over RLMesh-native values.

    Wrap a prediction function, or subclass it and implement one predict corner.
    """

    _bridge: ClassVar[ValueBridge] = identity_bridge
    # Without this, run(address) falls back to the numpy RemoteEnv (forcing the
    # optional numpy dep and decoding observations as ndarrays instead of Values).
    _remote_env_cls = RemoteEnv


@final
class SandboxEnv(SandboxEnvBase[Value, Value]):
    """Owned Docker-backed session for one environment (experimental).

    Builds or pulls the image, connects a :class:`RemoteEnv`, and stops the
    container on close.
    """

    _bridge: ClassVar[ValueBridge] = identity_bridge


@final
class SandboxVectorEnv(SandboxVectorEnvBase[Value, Value]):
    """Owned Docker-backed session for a vector environment (experimental).

    Builds or pulls the image, connects a :class:`RemoteVectorEnv`, and stops
    the container on close.
    """

    _bridge: ClassVar[ValueBridge] = identity_bridge


__all__ = [
    "Model",
    "RemoteEnv",
    "RemoteModel",
    "RemoteVectorEnv",
    "SandboxEnv",
    "SandboxModel",
    "SandboxVectorEnv",
]
