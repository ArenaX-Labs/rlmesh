"""Dependency-free native RLMesh SDK classes."""

from __future__ import annotations

from typing import ClassVar, final

from ._values import ValueBridge, identity_bridge
from .client import RemoteEnvBase, RemoteVectorEnvBase
from .model import ModelBase
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


__all__ = [
    "Model",
    "RemoteEnv",
    "RemoteVectorEnv",
]
