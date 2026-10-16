"""Remote clients over TLS: https:// endpoints, api_key, and latency hiding.

A plaintext model server runs behind a TLS-terminating TCP proxy (ALPN h2)
whose certificate comes from a throwaway test CA, as a deployment behind a
TLS load balancer looks to a client. The proxy can delay each direction to
stand in for a long round trip.
"""

from __future__ import annotations

import asyncio
import os
import socket
import ssl
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

trustme = pytest.importorskip("trustme")


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])
    finally:
        sock.close()


class TlsProxy:
    """Terminate TLS on a loopback port and forward bytes to ``backend_port``.

    ``delay_ms`` holds every chunk in each direction that long before it is
    forwarded (a round trip costs twice it), keeping order and overlap.
    """

    def __init__(
        self, backend_port: int, context: ssl.SSLContext, delay_ms: float = 0.0
    ) -> None:
        self.backend_port = backend_port
        self.context = context
        self.delay = delay_ms / 1000.0
        self.port = 0
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._ready.wait(5), "TLS proxy never started"

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start())
        self._ready.set()
        self._loop.run_forever()

    async def _start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle, "127.0.0.1", 0, ssl=self.context
        )
        self.port = self._server.sockets[0].getsockname()[1]

    async def _handle(
        self, client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter
    ) -> None:
        try:
            backend_r, backend_w = await asyncio.open_connection(
                "127.0.0.1", self.backend_port
            )
        except OSError:
            client_w.close()
            return
        await asyncio.gather(
            self._pipe(client_r, backend_w),
            self._pipe(backend_r, client_w),
            return_exceptions=True,
        )

    async def _pipe(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        queue: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue()

        async def forward() -> None:
            while True:
                due, data = await queue.get()
                if not data:
                    break
                wait = due - self._loop.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                writer.write(data)
                await writer.drain()

        sender = asyncio.ensure_future(forward())
        try:
            while True:
                data = await reader.read(65536)
                await queue.put((self._loop.time() + self.delay, data))
                if not data:
                    break
            await sender
        finally:
            sender.cancel()
            writer.close()

    async def _shutdown(self) -> None:
        self._server.close()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def close(self) -> None:
        asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)
        self._loop.close()


@pytest.fixture(scope="module")
def ca() -> Any:
    return trustme.CA()


@pytest.fixture(scope="module")
def ca_file(ca: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tls") / "ca.pem"
    ca.cert_pem.write_to_path(str(path))
    return path


def _server_context(ca: Any) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ca.issue_cert("localhost", "127.0.0.1").configure_cert(context)
    context.set_alpn_protocols(["h2"])
    return context


def _serve_policy(model: Any) -> int:
    import rlmesh

    port = _free_port()

    def run() -> None:
        model.serve(
            f"127.0.0.1:{port}",
            options=rlmesh.ServeOptions(allow_remote_shutdown=True),
        )

    threading.Thread(target=run, daemon=True).start()
    _wait_for_port(port)
    return port


def _wait_for_port(port: int) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise AssertionError(f"nothing listens on 127.0.0.1:{port}")


class CountEnv:
    """Box observations counting steps; ends after ``length`` steps."""

    def __init__(self, length: int = 3) -> None:
        import gymnasium as gym
        import numpy as np

        self.observation_space = gym.spaces.Box(-1e6, 1e6, (2,), np.float32)
        self.action_space = gym.spaces.Box(-1e6, 1e6, (2,), np.float32)
        self.length = length
        self.n = 0

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
        import numpy as np

        _ = seed, options
        self.n = 0
        return np.zeros(2, np.float32), {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        import numpy as np

        self.n += 1
        return np.full(2, self.n, np.float32), 1.0, self.n >= self.length, False, {}

    def close(self) -> None:
        return None


def _echo_policy() -> Any:
    import numpy as np
    from rlmesh.numpy import Model

    class Echo(Model):
        def predict(self, observation: Any) -> Any:
            return np.asarray(observation, np.float32)

    return Echo()


@pytest.fixture(scope="module")
def tls_model(ca: Any) -> Iterator[TlsProxy]:
    proxy = TlsProxy(_serve_policy(_echo_policy()), _server_context(ca))
    yield proxy
    proxy.close()


def _run_one(model: Any, env: Any) -> Any:
    import rlmesh

    with rlmesh.session(model, env) as sess:
        return sess.run(episodes=1)


def test_https_endpoint_with_a_ca_file_drives_a_local_env(
    tls_model: TlsProxy, ca_file: Path
) -> None:
    import rlmesh

    model = rlmesh.RemoteModel(
        f"https://localhost:{tls_model.port}", tls_ca_file=ca_file
    )
    result = _run_one(model, CountEnv())
    assert result.episodes[0].steps == 3


def test_an_untrusted_certificate_is_a_connection_error(
    tls_model: TlsProxy, tmp_path: Path
) -> None:
    import rlmesh

    other = tmp_path / "other-ca.pem"
    trustme.CA().cert_pem.write_to_path(str(other))
    model = rlmesh.RemoteModel(f"https://localhost:{tls_model.port}", tls_ca_file=other)
    with pytest.raises(
        ConnectionError, match="TLS handshake with localhost:"
    ) as raised:
        _run_one(model, CountEnv())
    assert "invalid peer certificate" in str(raised.value)

    # The system roots do not trust the test CA either.
    with pytest.raises(ConnectionError, match="TLS handshake with localhost:"):
        _run_one(rlmesh.RemoteModel(f"https://localhost:{tls_model.port}"), CountEnv())


def test_https_to_a_plaintext_server_is_a_connection_error(ca_file: Path) -> None:
    import rlmesh

    port = _serve_policy(_echo_policy())
    model = rlmesh.RemoteModel(f"https://127.0.0.1:{port}", tls_ca_file=ca_file)
    with pytest.raises(ConnectionError, match=f"TLS handshake with 127.0.0.1:{port}"):
        _run_one(model, CountEnv())


def test_an_https_address_with_a_path_is_a_value_error() -> None:
    import rlmesh

    with pytest.raises(ValueError, match="cannot have a path"):
        rlmesh.RemoteModel("https://models.example/v1")


def test_an_unusable_ca_file_is_a_value_error(tmp_path: Path) -> None:
    import rlmesh

    with pytest.raises(ValueError, match="cannot be read"):
        rlmesh.RemoteModel("https://localhost:1", tls_ca_file=tmp_path / "missing.pem")
    bad = tmp_path / "bad.pem"
    bad.write_text("not a certificate")
    with pytest.raises(ValueError, match="holds no PEM certificate"):
        rlmesh.RemoteModel("https://localhost:1", tls_ca_file=bad)
    with pytest.raises(ValueError, match="only to an https:// address"):
        rlmesh.RemoteModel("127.0.0.1:1", tls_ca_file=bad)


@pytest.fixture(scope="module")
def protected_tls_model(ca: Any) -> Iterator[TlsProxy]:
    """A model served with RLMESH_MODEL_ENDPOINT_TOKEN, behind the TLS proxy."""
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import rlmesh, numpy as np; from rlmesh.numpy import Model; "
            "Model(lambda observation: np.asarray(observation, np.float32))"
            f".serve('127.0.0.1:{port}')",
        ],
        env={**os.environ, "RLMESH_MODEL_ENDPOINT_TOKEN": "right-key"},
    )
    try:
        _wait_for_port(port)
        proxy = TlsProxy(port, _server_context(ca))
        yield proxy
        proxy.close()
    finally:
        process.kill()
        process.wait(10)


def test_the_right_api_key_is_accepted(
    protected_tls_model: TlsProxy, ca_file: Path
) -> None:
    import rlmesh

    model = rlmesh.RemoteModel(
        f"https://localhost:{protected_tls_model.port}",
        api_key="right-key",
        tls_ca_file=ca_file,
    )
    assert _run_one(model, CountEnv()).episodes[0].steps == 3


@pytest.mark.parametrize("api_key", [None, "wrong-key"])
def test_a_wrong_or_missing_api_key_is_a_connection_error(
    protected_tls_model: TlsProxy,
    ca_file: Path,
    api_key: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlmesh

    monkeypatch.delenv("RLMESH_API_KEY", raising=False)
    model = rlmesh.RemoteModel(
        f"https://localhost:{protected_tls_model.port}",
        api_key=api_key,
        tls_ca_file=ca_file,
    )
    with pytest.raises(ConnectionError) as raised:
        _run_one(model, CountEnv())
    assert "right-key" not in str(raised.value)
    if api_key is not None:
        assert api_key not in str(raised.value)


def test_rlmesh_api_key_is_sent_only_over_tls(
    protected_tls_model: TlsProxy, ca_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rlmesh

    monkeypatch.setenv("RLMESH_API_KEY", "right-key")
    model = rlmesh.RemoteModel(
        f"https://localhost:{protected_tls_model.port}", tls_ca_file=ca_file
    )
    assert _run_one(model, CountEnv()).episodes[0].steps == 3

    # The same key in the environment never reaches a plaintext address: the
    # protected backend, dialed directly, refuses the session.
    plaintext = rlmesh.RemoteModel(f"127.0.0.1:{protected_tls_model.backend_port}")
    with pytest.raises(ConnectionError):
        _run_one(plaintext, CountEnv())


def test_an_explicit_key_over_plaintext_warns_once() -> None:
    import rlmesh

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rlmesh.RemoteModel("10.255.255.1:5555", api_key="some-key")
        rlmesh.RemoteModel("127.0.0.1:5555", api_key="some-key")
    messages = [str(w.message) for w in caught]
    assert messages == [
        "api_key is sent unencrypted to tcp://10.255.255.1:5555; use an https:// "
        "address to keep it private"
    ], messages


def test_the_key_never_appears_in_repr_or_errors() -> None:
    import rlmesh

    model = rlmesh.RemoteModel("https://localhost:1", api_key="hidden-key")
    assert "hidden-key" not in repr(model)
    with pytest.raises(ValueError) as raised:
        rlmesh.RemoteModel("https://localhost:1", api_key="hidden-key\n")
    assert "hidden-key" not in str(raised.value)
    assert "trailing whitespace" in str(raised.value)


def test_chunked_execution_hides_round_trip_latency(ca: Any, ca_file: Path) -> None:
    """At ~40 ms injected RTT, execution_horizon=8 re-plans one step in eight,
    so the same episode runs several times faster than execution_horizon=1."""
    import numpy as np
    import rlmesh
    from rlmesh.numpy import Model

    class Chunker(Model):
        native_chunk = 8

        def predict_chunk(self, observation: Any) -> Any:
            return np.repeat(np.asarray(observation, np.float32)[None], 8, axis=0)

    proxy = TlsProxy(_serve_policy(Chunker()), _server_context(ca), delay_ms=20.0)
    try:
        model = rlmesh.RemoteModel(
            f"https://localhost:{proxy.port}", tls_ca_file=ca_file
        )

        def timed(horizon: int) -> float:
            with rlmesh.session(model, CountEnv(32), execution_horizon=horizon) as sess:
                start = time.perf_counter()
                result = sess.run(episodes=1)
                elapsed = time.perf_counter() - start
            assert result.episodes[0].steps == 32
            return elapsed

        one, eight = timed(1), timed(8)
    finally:
        proxy.close()
    # 32 round trips against 4: a loose bound that holds on a loaded host.
    assert one > 32 * 0.035, one
    assert eight * 2.5 < one, (one, eight)


def test_remote_env_dials_an_https_env_server(ca: Any, ca_file: Path) -> None:
    import rlmesh

    server = rlmesh.EnvServer(CountEnv(), address="tcp://127.0.0.1:0")
    server.start()
    proxy = TlsProxy(int(server.address.rsplit(":", 1)[1]), _server_context(ca))
    try:
        with rlmesh.numpy.RemoteEnv(
            f"https://localhost:{proxy.port}", tls_ca_file=ca_file
        ) as env:
            assert env.address == f"https://localhost:{proxy.port}"
            obs, _ = env.reset(seed=0)
            assert obs.tolist() == [0.0, 0.0]
            obs, *_ = env.step(obs)
            assert obs.tolist() == [1.0, 1.0]
    finally:
        proxy.close()
        server.shutdown()
