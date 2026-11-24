from __future__ import annotations

import socket
import threading
from collections.abc import Iterator

import pytest
import rlmesh
from rlmesh import spaces
from rlmesh._client._endpoint import env_address
from rlmesh._rlmesh import parse_address


class Env:
    observation_space = spaces.Discrete(2)
    action_space = spaces.Discrete(2)

    def reset(self, *, seed: int | None = None, options: object = None):
        return 0, {}

    def step(self, action: object):
        return 0, 1.0, True, False, {}

    def close(self) -> None:
        pass


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_a_bare_host_and_port_serves_and_reaches_the_env_there() -> None:
    bare = f"127.0.0.1:{free_port()}"
    server = rlmesh.EnvServer(Env(), bare)
    server.start()
    try:
        assert server.address.endswith("/env")
        for address in (bare, f"{bare}/env"):
            with rlmesh.RemoteEnv(address) as remote:
                assert remote.address == f"{bare}/env"
                remote.reset(seed=0)
                assert remote.step(0)[1] == 1.0
        result = rlmesh.Model(lambda observation: 0).run(bare)
        assert result.num_episodes == 1
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    ("short", "full"),
    [
        ("127.0.0.1:5555", "127.0.0.1:5555/env"),
        (":0", ":0/env"),
        ("[::1]:5555", "[::1]:5555/env"),
        ("lab:7447/env", "lab:7447/env"),
        ("lab:7447/bench/arm", "lab:7447/bench/arm"),
        ("cartpole", "cartpole"),
    ],
)
def test_the_bare_form_is_the_env_participant(short: str, full: str) -> None:
    assert env_address(short) == full


@pytest.mark.parametrize(
    "address", ["tcp://127.0.0.1:1", "unix:///tmp/env.sock", "https://127.0.0.1:1"]
)
def test_v1_schemes_are_still_refused(address: str) -> None:
    with pytest.raises(ValueError, match=r"v1 endpoints need RLMesh 0\.1"):
        rlmesh.RemoteEnv(address)
    with pytest.raises(ValueError, match=r"v1 endpoints need RLMesh 0\.1"):
        rlmesh.EnvServer(Env(), address)


def test_a_bare_port_names_the_host_form() -> None:
    for refused in (
        lambda: rlmesh.RemoteEnv("5555"),
        lambda: rlmesh.EnvServer(Env(), "5555"),
        lambda: rlmesh.Model(lambda observation: 0).run("5555"),
    ):
        with pytest.raises(ValueError, match=r'write ":5555" for this machine'):
            refused()


def test_a_served_model_still_names_itself() -> None:
    with pytest.raises(ValueError, match="participant address"):
        rlmesh.Model("127.0.0.1:5555")


def test_a_session_address_keeps_host_port_as_that_hosts_session() -> None:
    session = parse_address("127.0.0.1:5555", session=True)
    assert session["host"] == "127.0.0.1"
    assert session["port"] == 5555
    assert session["session"] is None
    assert session["participant"] is None


@pytest.fixture
def grpc_like_server() -> Iterator[int]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    stop = threading.Event()

    def answer(conn: socket.socket) -> None:
        with conn:
            conn.sendall(bytes([0, 0, 0, 4, 0, 0, 0, 0, 0]))
            try:
                conn.recv(256)
            except OSError:
                pass

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=answer, args=(conn,), daemon=True).start()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield listener.getsockname()[1]
    finally:
        stop.set()
        listener.close()


def test_a_01_grpc_server_is_named(grpc_like_server: int) -> None:
    address = f"127.0.0.1:{grpc_like_server}"
    expected = (
        rf"the server at {address} doesn't speak the RLMesh session protocol: "
        r"it answers HTTP/2, as an RLMesh 0\.1 \(gRPC\) server does"
    )
    with pytest.raises(rlmesh.ProtocolException, match=expected):
        rlmesh.RemoteEnv(address, connect_timeout_seconds=20)
