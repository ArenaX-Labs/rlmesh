# pyright: reportUnnecessaryTypeIgnoreComment=error
"""Pins the env-target split between served and local models.

A contract-only env handle (``env_contract`` without ``reset``/``step``) is a
valid :data:`rlmesh.types.EnvTarget` -- a served model binds to it -- but is
rejected at the type level by a local ``Model.bind``, whose
:data:`rlmesh.types.LocalEnvTarget` deliberately excludes it. The file-level
directive turns the suppressed rejection into a regression alarm: if
``Model.bind`` ever starts accepting the handle, the ignore comment below
becomes unnecessary and this fixture fails.
"""

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
