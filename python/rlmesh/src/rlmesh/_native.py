"""Dependency-free native RLMesh SDK classes."""

from __future__ import annotations

from typing import ClassVar, final

from ._client import RemoteEnvBase, RemoteModelBase, RemoteVectorEnvBase
from ._models.base import ModelBase
from ._sandbox import SandboxEnvBase, SandboxVectorEnvBase
from ._sandbox._model import SandboxModel
from ._value_conversion import ValueBridge, identity_bridge
from .types import Value


@final
class RemoteEnv(RemoteEnvBase[Value, Value]):
    _bridge: ClassVar[ValueBridge] = identity_bridge


@final
class RemoteModel(RemoteModelBase[Value, Value]):
    _bridge: ClassVar[ValueBridge] = identity_bridge


@final
class RemoteVectorEnv(RemoteVectorEnvBase[Value, Value]):
    _bridge: ClassVar[ValueBridge] = identity_bridge


class Model(ModelBase[Value, Value]):
    _bridge: ClassVar[ValueBridge] = identity_bridge
    # Without this, run(address) falls back to the numpy RemoteEnv (forcing the
    # optional numpy dep and decoding observations as ndarrays instead of Values).
    _remote_env_cls = RemoteEnv


@final
class SandboxEnv(SandboxEnvBase[Value, Value]):
    _bridge: ClassVar[ValueBridge] = identity_bridge


@final
class SandboxVectorEnv(SandboxVectorEnvBase[Value, Value]):
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
