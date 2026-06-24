from __future__ import annotations

from typing import Any, cast


def test_remote_handle_protocols_share_a_common_base() -> None:
    """The two remote-handle protocols are deduplicated via a shared base.

    The common surface lives on _RemoteHandleBase so it is declared once
    instead of being restated in both protocols.
    """
    from rlmesh._sandbox.session import (
        RemoteEnvHandle,
        RemoteVectorEnvHandle,
        _RemoteHandleBase,
    )

    env_handle_mro: tuple[type, ...] = cast(Any, RemoteEnvHandle).__mro__
    vector_handle_mro: tuple[type, ...] = cast(Any, RemoteVectorEnvHandle).__mro__
    assert _RemoteHandleBase in env_handle_mro
    assert _RemoteHandleBase in vector_handle_mro

    shared = (
        "env_contract",
        "spec",
        "render_mode",
        "metadata",
        "observation_space_spec",
        "action_space_spec",
        "render",
        "close",
    )
    # Shared members are defined on the base, not on the leaf protocols.
    for name in shared:
        assert name in vars(_RemoteHandleBase)
        assert name not in vars(RemoteEnvHandle)
        assert name not in vars(RemoteVectorEnvHandle)

    # Vector-only members stay on the vector protocol.
    for name in ("single_observation_space", "single_action_space", "num_envs"):
        assert name in vars(RemoteVectorEnvHandle)
        assert name not in vars(RemoteEnvHandle)
