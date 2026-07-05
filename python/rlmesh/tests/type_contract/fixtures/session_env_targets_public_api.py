# pyright: reportUnnecessaryTypeIgnoreComment=error
"""Pins the env-target split between served and local model sessions.

A contract-only env handle (``env_contract`` without ``reset``/``step``) is a
valid :data:`rlmesh.types.EnvTarget` -- served models bind to it via
:func:`rlmesh.session` -- but is rejected at the type level by a local
``Model.session``, whose :data:`rlmesh.types.LocalEnvTarget` deliberately
excludes it. The file-level directive turns the suppressed rejection into a
regression alarm: if ``Model.session`` ever starts accepting the handle, the
ignore comment below becomes unnecessary and this fixture fails.
"""

from __future__ import annotations

from typing import Any, cast

import rlmesh
from rlmesh.specs import EnvContract
from rlmesh.types import EnvTarget
from typing_extensions import assert_type


class ContractOnlyEnv:
    @property
    def env_contract(self) -> EnvContract:
        raise NotImplementedError


def accepts_env_target(env: EnvTarget) -> None:
    pass


accepts_env_target(ContractOnlyEnv())


def _served_model_accepts_contract_only(
    remote_model: rlmesh.RemoteModel, env: ContractOnlyEnv
) -> None:
    assert_type(remote_model.session(env), rlmesh.Session[Any, Any])
    assert_type(rlmesh.session(remote_model, env), rlmesh.Session[Any, Any])


def _local_model_rejects_contract_only(
    local_model: rlmesh.Model[Any, Any], env: ContractOnlyEnv
) -> None:
    local_model.session(env)  # pyright: ignore[reportArgumentType]
    rlmesh.session(local_model, cast(EnvTarget, env))
