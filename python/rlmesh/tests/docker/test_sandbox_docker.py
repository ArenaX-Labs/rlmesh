"""SandboxEnv and SandboxModel against real containers on this Docker host.

The release gate for the sandbox (``mise run test:sandbox:docker``): every
start joins, however long the container takes to listen behind Docker's
port proxy; a wrong key is refused at once; a failing container reports its
own error. Repeats are set by ``RLMESH_DOCKER_GATE_STARTS`` (starts of each
sandbox, default 10) and ``RLMESH_DOCKER_GATE_RUNS`` (evaluations, default 3).

Addresses carry keys, so no test prints one.
"""

from __future__ import annotations

import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest
import rlmesh
from rlmesh._rlmesh import (
    sandbox_host_identity,
    sandbox_session_ports,
    sandbox_stop_env,
)
from rlmesh._sandbox import _model as sandbox_model
from rlmesh._sandbox import env as sandbox_env
from rlmesh._sandbox import session as sandbox_session
from sandbox_gate import LABEL, TIMINGS, GateImages, docker

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.docker

STARTS = int(os.environ.get("RLMESH_DOCKER_GATE_STARTS", "10"))
RUNS = int(os.environ.get("RLMESH_DOCKER_GATE_RUNS", "3"))

#: How long the join's probe waits for a port to hold a connection
#: (``HOLD_PROBE`` in ``rlmesh-session``'s hosting rule).
HOLD_PROBE = 0.25
#: What the exits image's make() raises.
EXITS_MARKER = "rlmesh-docker-gate: make() failed on purpose"


class Started:
    """When each container came up, so a join is timed from there."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, module: Any, name: str) -> None:
        self.at: list[float] = []
        inner: Callable[..., Any] = getattr(module, name)

        def timed(*args: Any, **kwargs: Any) -> Any:
            started = inner(*args, **kwargs)
            self.at.append(time.monotonic())
            return started

        monkeypatch.setattr(module, name, timed)


def _failures(errors: list[tuple[int, BaseException]], what: str, total: int) -> None:
    if errors:
        lines = "\n".join(f"  #{i}: {type(err).__name__}: {err}" for i, err in errors)
        pytest.fail(f"{len(errors)} of {total} {what} failed:\n{lines}")


def _rekeyed(address: str, key: str) -> str:
    """``address`` with its key replaced by ``key``."""
    head, _, name = address.partition("/")
    return f"{head.partition('#')[0]}#{key}/{name}"


def test_sandbox_env_starts_every_time(
    gate_images: GateImages, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = Started(monkeypatch, sandbox_env, "start_sandbox_container")
    errors: list[tuple[int, BaseException]] = []
    for i in range(STARTS):
        t0 = time.monotonic()
        try:
            env = rlmesh.SandboxEnv(gate_images.env)
        except Exception as err:  # counted, so one run reports every failure
            errors.append((i, err))
            continue
        joined = time.monotonic()
        TIMINGS["SandboxEnv start (docker run + join)"].append(joined - t0)
        TIMINGS["SandboxEnv join (after docker run)"].append(joined - started.at[-1])
        try:
            env.reset(seed=i)
            env.step(env.action_space.sample())
        except Exception as err:
            errors.append((i, err))
        finally:
            env.close()
    _failures(errors, "SandboxEnv starts", STARTS)


def test_sandbox_model_serves_every_time(
    gate_images: GateImages, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = Started(monkeypatch, sandbox_model, "start_prebuilt_container")
    errors: list[tuple[int, BaseException]] = []
    for i in range(STARTS):
        model = rlmesh.SandboxModel(gate_images.model)
        t0 = time.monotonic()
        try:
            model.serve()
        except Exception as err:
            errors.append((i, err))
            continue
        else:
            joined = time.monotonic()
            TIMINGS["SandboxModel serve (docker run + join)"].append(joined - t0)
            TIMINGS["SandboxModel join (after docker run)"].append(
                joined - started.at[-1]
            )
        finally:
            model.shutdown()
    _failures(errors, "SandboxModel serves", STARTS)


def test_model_image_evaluates_env_image(gate_images: GateImages) -> None:
    errors: list[tuple[int, BaseException]] = []
    for i in range(RUNS):
        t0 = time.monotonic()
        try:
            with rlmesh.SandboxModel(gate_images.model) as model:
                env = rlmesh.SandboxEnv(gate_images.env)
                try:
                    t1 = time.monotonic()
                    result = model.run(env, episodes=1, max_episode_steps=20)
                    TIMINGS["evaluation run (both up)"].append(time.monotonic() - t1)
                finally:
                    env.close()
            assert result.num_episodes == 1
            assert result.total_steps > 0
        except Exception as err:
            errors.append((i, err))
            continue
        TIMINGS["evaluation (start both, run, stop)"].append(time.monotonic() - t0)
    _failures(errors, "evaluations", RUNS)


def test_a_wrong_key_is_refused_at_once(gate_images: GateImages) -> None:
    _, other = sandbox_host_identity()
    with rlmesh.SandboxEnv(gate_images.env) as env:
        wrong = _rekeyed(env.sandbox.address, other)
        t0 = time.monotonic()
        with pytest.raises(Exception, match="Unauthenticated") as refused:
            rlmesh.RemoteEnv(wrong, connect_timeout_seconds=30)
        waited = time.monotonic() - t0
    TIMINGS["wrong key refused (host up)"].append(waited)
    assert waited < 2.0, f"refused only after {waited:.2f}s: {refused.value}"


def test_a_wrong_key_is_refused_once_a_starting_host_listens(
    gate_images: GateImages,
) -> None:
    """Dialled while the container starts: no refusal before the host
    listens (the proxy drops links until then), a final one soon after."""
    info = sandbox_session.start_prebuilt_container(
        gate_images.env, requested_source=gate_images.env, name="env", binding={}
    )
    try:
        _, other = sandbox_host_identity()
        t0 = time.monotonic()
        with pytest.raises(Exception, match="Unauthenticated"):
            rlmesh.RemoteEnv(_rekeyed(info.address, other), connect_timeout_seconds=30)
        waited = time.monotonic() - t0
    finally:
        sandbox_stop_env(container_id=info.container_id)
    TIMINGS["wrong key refused (dialled at docker run)"].append(waited)
    assert waited < 10.0, f"refused only after {waited:.2f}s"


def test_a_container_that_exits_reports_its_own_error(gate_images: GateImages) -> None:
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="exited before becoming ready") as failed:
        rlmesh.SandboxEnv(gate_images.exits)
    waited = time.monotonic() - t0
    TIMINGS["exited container reported"].append(waited)
    assert EXITS_MARKER in str(failed.value), str(failed.value)
    assert waited < 15.0, f"reported only after {waited:.2f}s"


def test_a_container_publishing_only_its_base_port_is_named(
    gate_images: GateImages,
) -> None:
    identity, key = sandbox_host_identity()
    with sandbox_session._PORTS_LOCK:  # pyright: ignore[reportPrivateUsage]
        base = sandbox_session_ports()
        container = docker(
            "run",
            "-d",
            "--label",
            f"{LABEL}={gate_images.run}",
            "-p",
            f"127.0.0.1:{base}:{base}",
            "-e",
            f"RLMESH_ADDRESS=0.0.0.0:{base}",
            "-e",
            "RLMESH_NAME=env",
            "-e",
            sandbox_session.HOST_IDENTITY_VAR,
            gate_images.env,
            env={sandbox_session.HOST_IDENTITY_VAR: identity},
        )
    # Past the join's 10s default, so the caller's own bound is the one kept.
    timeout = 12.0
    t0 = time.monotonic()
    try:
        with pytest.raises(Exception, match="did not answer") as unanswered:
            rlmesh.RemoteEnv(
                f"127.0.0.1:{base}#{key}/env", connect_timeout_seconds=timeout
            )
        waited = time.monotonic() - t0
    finally:
        docker("rm", "-f", container)
    TIMINGS["base port only, named (12s bound)"].append(waited)
    message = str(unanswered.value)
    for port in (base + 1, base + 2, base + 3):
        assert str(port) in message, message
    assert key not in message
    assert timeout - 1.0 < waited < timeout + 1.0, f"gave up after {waited:.2f}s"


def test_two_sandboxes_start_at_once(gate_images: GateImages) -> None:
    def env_up(_: int) -> float:
        t0 = time.monotonic()
        with rlmesh.SandboxEnv(gate_images.env) as env:
            joined = time.monotonic() - t0
            env.reset(seed=0)
        return joined

    def model_up(_: int) -> float:
        t0 = time.monotonic()
        with rlmesh.SandboxModel(gate_images.model):
            return time.monotonic() - t0

    for what, start in (("SandboxEnv", env_up), ("SandboxModel", model_up)):
        with ThreadPoolExecutor(2) as pool:
            for joined in pool.map(start, range(2)):
                TIMINGS[f"{what} start, two at once"].append(joined)
    with ThreadPoolExecutor(2) as pool:
        both = [pool.submit(env_up, 0), pool.submit(model_up, 0)]
        for future in both:
            TIMINGS["SandboxEnv + SandboxModel at once"].append(future.result())


def test_dockers_proxy_drops_an_unserved_link_fast(gate_images: GateImages) -> None:
    """The join tells a host from Docker's port proxy by whether a port holds
    a connection for HOLD_PROBE; the proxy, with nothing listening behind
    it, must close one well within that."""
    container = docker(
        "run",
        "-d",
        "--label",
        f"{LABEL}={gate_images.run}",
        "-p",
        "127.0.0.1::9",
        "--entrypoint",
        "sleep",
        gate_images.env,
        "120",
    )
    try:
        port = int(
            docker("port", container, "9/tcp").splitlines()[0].rpartition(":")[2]
        )
        held: list[float] = []
        for i in range(200):
            try:
                link = socket.create_connection(("127.0.0.1", port), timeout=2.0)
            except OSError:
                held.append(0.0)  # no userland proxy: refused outright
                continue
            t0 = time.monotonic()
            with link:
                link.settimeout(5.0)
                try:
                    if i % 2:
                        link.sendall(b"\0" * 16)
                    link.recv(1)
                except OSError:
                    pass
            held.append(time.monotonic() - t0)
    finally:
        docker("rm", "-f", container)
    TIMINGS["proxy holds an unserved link"].extend(held)
    assert max(held) < HOLD_PROBE / 2, f"proxy held a link {max(held):.3f}s"
