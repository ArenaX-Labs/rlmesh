"""Remote env clients over TLS: an https:// endpoint.

A plaintext env server runs behind a TLS-terminating TCP proxy (ALPN h2)
whose certificate comes from a throwaway test CA, as a deployment behind a
TLS load balancer looks to a client.
"""

from __future__ import annotations

import asyncio
import ssl
import threading
from pathlib import Path
from typing import Any

import pytest

trustme = pytest.importorskip("trustme")


class TlsProxy:
    """Terminate TLS on a loopback port bound to port 0 and forward bytes to
    ``backend`` (a Unix socket path, or a loopback TCP port).

    ``delay_ms`` holds every chunk in each direction that long before it is
    forwarded (a round trip costs twice it), keeping order and overlap.
    """

    def __init__(
        self, backend: str | int, context: ssl.SSLContext, delay_ms: float = 0.0
    ) -> None:
        self.backend = backend
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
            if isinstance(self.backend, str):
                backend_r, backend_w = await asyncio.open_unix_connection(self.backend)
            else:
                backend_r, backend_w = await asyncio.open_connection(
                    "127.0.0.1", self.backend
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
