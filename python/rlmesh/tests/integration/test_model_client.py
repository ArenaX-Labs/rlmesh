"""A Python eval loop driving a served model via PyModelClient.

This exercises the eval-driving symmetry: the model is *served* (here a Python
model on a background thread; the wire path is identical for a C/C++ model served
with rlmesh_model_serve) and the eval loop stays in Python, mapping observations
to actions over PyModelClient -> RemoteModel -> ModelService.
"""

from __future__ import annotations

import socket
import threading
import time


def _reserve_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def test_python_eval_drives_a_served_model() -> None:
    import pytest
    import rlmesh
    from rlmesh import spaces
    from rlmesh._rlmesh import PyModelClient

    observation_space = spaces.Discrete(8)
    action_space = spaces.Discrete(4)
    address = f"tcp://127.0.0.1:{_reserve_port()}"

    # A constant model: ignore the observation, always answer action 2.
    model = rlmesh.Model(lambda _observation: 2)
    server_error: list[BaseException] = []

    def _serve() -> None:
        try:
            model.serve(
                address,
                options=rlmesh.ServeOptions(allow_remote_shutdown=True),
            )
        except BaseException as error:
            server_error.append(error)

    threading.Thread(target=_serve, daemon=True).start()

    client = None
    last_error: BaseException | None = None
    for _ in range(100):
        if server_error:
            break
        try:
            client = PyModelClient(address, observation_space, action_space)
            break
        except Exception as error:
            last_error = error
            time.sleep(0.05)

    if server_error and "Operation not permitted" in str(server_error[0]):
        pytest.skip("local tcp bind is not permitted in this environment")
    assert client is not None, f"could not connect to served model: {last_error}"

    try:
        # First step of the first episode, then a fresh episode — the constant
        # model answers 2 either way.
        assert client.predict(3) == 2
        client.begin_episode()
        assert client.predict(0) == 2
    finally:
        client.shutdown("test complete")
