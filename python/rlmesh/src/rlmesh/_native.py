"""Dependency-free native RLMesh SDK classes."""

from __future__ import annotations

from typing import ClassVar, final

from ._values import ValueAdapter, identity_adapter
from .client import RemoteEnvBase, RemoteVectorEnvBase
from .model import ModelBase
from .sandbox import SandboxEnvBase, SandboxInfo, SandboxVectorEnvBase
from .types import Value


@final
class RemoteEnv(RemoteEnvBase[Value, Value]):
    _adapter: ClassVar[ValueAdapter] = identity_adapter


@final
class RemoteVectorEnv(RemoteVectorEnvBase[Value, Value]):
    _adapter: ClassVar[ValueAdapter] = identity_adapter


@final
class Model(ModelBase[Value, Value]):
    _adapter: ClassVar[ValueAdapter] = identity_adapter


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
    "SandboxInfo",
    "SandboxVectorEnv",
]
