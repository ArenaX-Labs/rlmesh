"""How an in-process host carries boards' realtime parts.

- By default it listens for the datagram path on UDP, its port + 1.
- ``RLMESH_BOARD_DATA=tcp`` or ``Host(board_data="tcp")`` binds no UDP
  socket there, so a sandbox that forbids UDP sets one variable.
- A bad value is refused with ``rlmesh host``'s message.
"""

from __future__ import annotations

import socket

import pytest
from rlmesh import realtime

POSSIBLE = "[possible values: tcp, udp, udp-only]"


def _udp_taken(host: realtime.Host) -> bool:
    """Whether the host holds UDP on its base port + 1."""
    ip, port = host.address.split("/")[0].split("#")[0].rsplit(":", 1)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.bind((ip, int(port) + 1))
        except OSError:
            return True
    return False


@pytest.fixture(autouse=True)
def _no_board_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RLMESH_BOARD_DATA", raising=False)


def test_the_default_listens_on_udp() -> None:
    with realtime.Host() as host:
        assert _udp_taken(host)


def test_the_variable_set_to_tcp_binds_no_udp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLMESH_BOARD_DATA", "tcp")
    with realtime.Host() as host:
        assert not _udp_taken(host)


@pytest.mark.parametrize("board_data", ["tcp", realtime.BoardData.TCP])
def test_host_board_data_tcp_binds_no_udp(
    board_data: realtime.BoardData | str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLMESH_BOARD_DATA", "udp")  # the argument wins
    with realtime.Host(board_data=board_data) as host:  # pyright: ignore[reportArgumentType]
        assert not _udp_taken(host)


def test_a_bad_variable_is_refused_as_the_cli_refuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RLMESH_BOARD_DATA", "quic")
    with pytest.raises(realtime.SessionException) as raised:
        realtime.Host()
    message = str(raised.value)
    assert "invalid value 'quic' for 'RLMESH_BOARD_DATA'" in message, message
    assert POSSIBLE in message, message


def test_a_bad_argument_is_refused() -> None:
    with pytest.raises(
        ValueError, match="invalid value 'quic' for 'board_data'"
    ) as raised:
        realtime.Host(board_data="quic")  # pyright: ignore[reportArgumentType]
    assert POSSIBLE in str(raised.value)
