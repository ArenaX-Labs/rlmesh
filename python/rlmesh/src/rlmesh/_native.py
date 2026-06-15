"""Dependency-free native RLMesh SDK classes."""

from __future__ import annotations

from typing import ClassVar, final

from ._framework_bridge import ValueBridge, identity_bridge
from .client import RemoteEnvBase, RemoteVectorEnvBase
from .models.base import ModelBase
from .sandbox import SandboxEnvBase, SandboxVectorEnvBase
from .sandbox._model import SandboxModel
from .types import Value


@final
class RemoteEnv(RemoteEnvBase[Value, Value]):
    _bridge: ClassVar[ValueBridge] = identity_bridge


@final
class RemoteVectorEnv(RemoteVectorEnvBase[Value, Value]):
    _bridge: ClassVar[ValueBridge] = identity_bridge


@final
class Model(ModelBase[Value, Value]):
    _bridge: ClassVar[ValueBridge] = identity_bridge


@final
class SandboxEnv(SandboxEnvBase[Value, Value]):
    _remote_env_cls = RemoteEnv


@final
class SandboxVectorEnv(SandboxVectorEnvBase[Value, Value]):
    _remote_env_cls = RemoteVectorEnv


__all__ = [
    "Model",
    "RemoteEnv",
    "RemoteVectorEnv",
    "SandboxEnv",
    "SandboxModel",
    "SandboxVectorEnv",
]
