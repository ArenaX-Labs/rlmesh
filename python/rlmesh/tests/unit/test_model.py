from __future__ import annotations

import pytest


def test_env_address_accepts_private_address_attr() -> None:
    from rlmesh.model import _env_address

    class Remote:
        _address = "tcp://127.0.0.1:5555"

    assert _env_address(Remote()) == "tcp://127.0.0.1:5555"


def test_env_address_accepts_address_property() -> None:
    from rlmesh.model import _env_address

    class ServerLike:
        @property
        def address(self) -> str:
            return "tcp://127.0.0.1:6000"

    assert _env_address(ServerLike()) == "tcp://127.0.0.1:6000"


def test_env_address_accepts_plain_address_attr() -> None:
    from rlmesh.model import _env_address

    class Holder:
        def __init__(self) -> None:
            self.address = "tcp://127.0.0.1:7000"

    assert _env_address(Holder()) == "tcp://127.0.0.1:7000"


def test_env_address_accepts_callable_address() -> None:
    from rlmesh.model import _env_address

    class Native:
        def address(self) -> str:
            return "tcp://127.0.0.1:8000"

    assert _env_address(Native()) == "tcp://127.0.0.1:8000"


def test_env_address_rejects_unsupported() -> None:
    from rlmesh.model import _env_address

    with pytest.raises(TypeError, match="remote env object or address string"):
        _env_address(object())


def test_shutdown_env_falls_back_to_no_arg_shutdown() -> None:
    from rlmesh.model import _shutdown_env

    calls: list[tuple[object, ...]] = []

    class ServerLike:
        def shutdown(self) -> None:
            calls.append(())

    _shutdown_env(ServerLike(), "tcp://127.0.0.1:6000")

    assert calls == [()]
