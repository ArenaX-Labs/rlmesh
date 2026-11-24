# pyright: reportUnnecessaryTypeIgnoreComment=error

from __future__ import annotations

from typing import Any

import rlmesh
from rlmesh.specs import EnvContract
from rlmesh.types import EnvTarget


class ContractOnlyEnv:
    @property
    def env_contract(self) -> EnvContract:
        raise NotImplementedError


def accepts_env_target(env: EnvTarget) -> None:
    pass


accepts_env_target(ContractOnlyEnv())


def _local_model_rejects_contract_only(
    local_model: rlmesh.Model[Any, Any], env: ContractOnlyEnv
) -> None:
    local_model.bind(env)  # pyright: ignore[reportArgumentType]
