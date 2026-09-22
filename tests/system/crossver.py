#!/usr/bin/env python3
"""Cross-version compatibility matrix: the pinned published wheel vs this tree.

The "old" side of every cell is the immutable PyPI artifact pinned in
``tests/system/crossver.lock``; the "new" side is a release-cohort wheel built
from this tree (``RLMESH_RELEASE_BUILD=1``, so it stamps the edition the
published wheel advertises and the two land in one cohort).

What the six real cells assert:

* the **trace**: a Join across the two builds completes and reproduces the
  committed ``traces/counter-entrypoint.json``. This is the cross-version
  evidence -- the only assertion both builds take part in.
* the **server's handshake**, measured by the raw wire probe below: its
  ``compatible`` flag and its WANT (``preferred_workflow_edition`` -- this
  tree's servers declare the current edition, the published wheel declares
  nothing, which is the undeclared-peer case the negotiation must tolerate).
  The probe offers the runtime's CAN set but is not the runtime's own client,
  so these two columns characterise the *server* of a cell, not the pair.
* the **edition**. On the model leg this is the real pin: the runtime's
  ``ResolveAdapterRequest`` edition, read from the served model's
  ``model adapter pinned to runtime-selected edition`` log line, so it is
  negotiated between the two builds. On the env leg this tree's runtime pins
  the env with ``ConfigureEnv`` as its first Join message: cell 4 reads that
  pin from this tree's env server ``env pinned to runtime-selected edition``
  log line, and cell 1 (the published wheel serves the env and logs no pin)
  relies on the trace, since a refused pin aborts the session before Reset.
  The published wheel's runtime sends no pin, so cells 2-3 fall back to the
  two builds' CAN intersection, confirmed by pinning that same edition through
  the probe's ``ConfigureEnv`` and requiring an ack.

Cells 9 and 10 are forged refusals: they assert the refusal message, not a
trace or an edition.

Run it through ``mise run test:crossver``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomllib

if TYPE_CHECKING:
    # Only the matrix half of this script talks to the system runner; the probe
    # and the model-leg driver run inside venvs that do not have it installed.
    from rlmesh_system.support.manifest import ScenarioSpec
    from rlmesh_system.support.process import ManagedProcess

# The published fixture, as ONE variable. It is repointed to "0.1.0" once that
# tag ships and then never changes again: a sealed release is the permanent old
# side of this matrix. Keep it in step with tests/system/crossver.lock.
FIXTURE_VERSION = "0.1.0rc12"

ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
SYSTEM_DIR = ROOT / "tests" / "system"
PROFILE_DIR = SYSTEM_DIR / "profiles"
LOCK_PATH = SYSTEM_DIR / "crossver.lock"
FIXTURE_DIR = ROOT / "tools" / "rlmesh_system_fixtures"
WORK_DIR = ROOT / "target" / "python-validation"
OLD_VENV = WORK_DIR / "venvs" / "crossver-old"
LOGS = WORK_DIR / "logs"
TRACES = WORK_DIR / "crossver"

# The scenario every cell runs, taken from the basic profile so the matrix and
# its committed baseline can never drift apart.
SCENARIO_NAME = "counter-entrypoint"
BASELINE = SYSTEM_DIR / "traces" / f"{SCENARIO_NAME}.json"
MODEL_ENTRYPOINT = "rlmesh_system_fixtures.models.discrete:discrete_zero"

# An edition no build implements, and a generation no build speaks.
UNKNOWN_EDITION = "2099.01"
FORGED_GENERATION = "rlmesh-wire-v2"

ENV_SERVICE = "rlmesh.env.v1.EnvService"
MODEL_SERVICE = "rlmesh.model.v1.ModelService"
MODEL_SERVING_RE = re.compile(r"RLMesh serving model on (\S+)")
PINNED_EDITION_RE = re.compile(
    r"model adapter pinned to runtime-selected edition.*?selected_workflow_edition=(\S+)"
)
ENV_PINNED_EDITION_RE = re.compile(
    r"env pinned to runtime-selected edition.*?selected_workflow_edition=(\S+)"
)
# tracing-subscriber colorizes field names, so the log has to be de-styled
# before the pin can be read out of it.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# What the table's columns are evidence of, printed under it.
LEGEND = (
    "compatible: the server's own HandshakeResponse flag, read off the wire by the raw probe;",
    "            the probe carries the runtime's CAN but is not its client, so it rates the server.",
    "edition:    cells 5-6, the runtime's real ResolveAdapter pin; cells 1 and 4, the runtime's real",
    "            ConfigureEnv pin -- read off this tree's env server log in cell 4, proven by the",
    "            trace in cell 1 (the old server logs no pin; a refused pin aborts before Reset);",
    "            cells 2-3, the old runtime sends no pin, so the two builds' CAN intersection, acked",
    "            by a forged ConfigureEnv.",
    "trace:      the cross-version Join against tests/system/traces/counter-entrypoint.json.",
)


@dataclass
class Cell:
    number: int
    name: str
    compatible: bool | None
    edition: str | None
    trace: str
    status: str
    detail: str


# ──────────────────────────────────────────────
# Minimal protobuf + gRPC probe
#
# The probe forges handshake and ConfigureEnv fields no released client can
# send, so it cannot go through the rlmesh client. Three messages are needed and
# every field of them is a scalar, so they are hand-encoded rather than adding a
# protobuf toolchain to the profile venv.
# ──────────────────────────────────────────────


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _delimited(field: int, payload: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _text(field: int, value: str) -> bytes:
    return _delimited(field, value.encode())


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def _decode(data: bytes) -> dict[int, list[bytes | int]]:
    fields: dict[int, list[bytes | int]] = {}
    pos = 0
    while pos < len(data):
        key, pos = _read_varint(data, pos)
        field, wire = key >> 3, key & 7
        value: bytes | int
        if wire == 0:
            value, pos = _read_varint(data, pos)
        elif wire == 2:
            length, pos = _read_varint(data, pos)
            value, pos = data[pos : pos + length], pos + length
        elif wire in (1, 5):
            width = 8 if wire == 1 else 4
            value, pos = data[pos : pos + width], pos + width
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        fields.setdefault(field, []).append(value)
    return fields


def _message(
    fields: dict[int, list[bytes | int]], field: int
) -> dict[int, list[bytes | int]]:
    values = fields.get(field)
    return _decode(bytes(values[0])) if values else {}


def _strings(fields: dict[int, list[bytes | int]], field: int) -> list[str]:
    return [bytes(value).decode() for value in fields.get(field, [])]


def _string(fields: dict[int, list[bytes | int]], field: int) -> str:
    values = _strings(fields, field)
    return values[0] if values else ""


def run_probe(args: argparse.Namespace) -> int:
    import grpc

    service = ENV_SERVICE if args.peer == "env" else MODEL_SERVICE
    channel = grpc.insecure_channel(args.address.removeprefix("tcp://"))
    raw = (lambda payload: payload, lambda payload: payload)
    result: dict[str, object] = {}
    try:
        # core.v1.HandshakeRequest{1: generation, 4: CAN}, inside the service's
        # own HandshakeRequest{1: base}.
        core = _text(1, args.generation)
        for edition in args.can:
            core += _text(4, edition)
        handshake = channel.unary_unary(f"/{service}/Handshake", *raw)(
            _delimited(1, core), timeout=args.timeout
        )
        base = _message(_decode(handshake), 1)
        result["handshake"] = {
            "compatible": bool(base.get(1, [0])[0]),
            "supported_workflow_editions": _strings(base, 4),
            "preferred_workflow_edition": _string(base, 6),
            "error_message": _string(base, 5),
        }

        if args.configure is not None:
            # JoinRequest{5: request_id, 6: ConfigureEnvRequest{1: edition}}; the
            # reply is JoinResponse{7: ack} or JoinResponse{10: EnvError{2: msg}}.
            request = _delimited(6, _text(1, args.configure)) + _text(5, "crossver")
            stream = channel.stream_stream(f"/{service}/Join", *raw)(
                iter([request]), timeout=args.timeout
            )
            reply = _decode(next(iter(stream)))
            result["configure"] = {
                "accepted": 7 in reply,
                "message": _string(_message(reply, 10), 2),
            }
    finally:
        channel.close()

    print(json.dumps(result))
    return 0


# ──────────────────────────────────────────────
# Matrix
# ──────────────────────────────────────────────


class Harness:
    """Thin wrapper over the system runner's process helpers."""

    def __init__(self, *, verbose: bool) -> None:
        from rlmesh_system.support.rendering import Renderer

        self.renderer = Renderer(plain=True)
        self.verbose = verbose
        LOGS.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        label: str,
        log: str,
        timeout: float = 600.0,
    ) -> Path:
        from rlmesh_system.support.command import run_command

        log_path = LOGS / f"{log}.log"
        run_command(
            command,
            cwd=ROOT,
            env=env,
            log_path=log_path,
            label=label,
            verbose=self.verbose,
            renderer=self.renderer,
            timeout_seconds=timeout,
        )
        return log_path

    def start(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        label: str,
        log: str,
        pattern: re.Pattern[str],
    ) -> tuple[ManagedProcess, str]:
        from rlmesh_system.support.process import start_process_until

        def predicate(line: str) -> str | None:
            match = pattern.search(line)
            return match.group(1) if match else None

        return start_process_until(
            command,
            cwd=ROOT,
            env=env,
            log_path=LOGS / f"{log}.log",
            label=label,
            verbose=self.verbose,
            renderer=self.renderer,
            timeout_seconds=120.0,
            predicate=predicate,
        )


def expected_edition() -> str:
    manifest = tomllib.loads((ROOT / "rlmesh.toml").read_text())
    return str(manifest["workflow"]["current_edition"])


def assert_lock_pins_fixture() -> None:
    requirement = f"rlmesh=={FIXTURE_VERSION}"
    if requirement not in LOCK_PATH.read_text():
        raise SystemExit(
            f"{LOCK_PATH} does not pin {requirement}: the lock file and "
            "FIXTURE_VERSION must name the same published wheel"
        )


def read_build(python: Path, label: str) -> tuple[str, str]:
    """The installed package version and the edition its native build stamps."""
    code = "import rlmesh; print(rlmesh.__version__); print(rlmesh.__build__)"
    result = subprocess.run(
        [str(python), "-c", code], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(f"{label}: could not import rlmesh\n{result.stderr}")
    lines = result.stdout.split()
    if len(lines) != 2:
        raise SystemExit(
            f"{label}: unexpected `rlmesh` version output\n{result.stdout}"
        )
    version, build = lines
    return version, build


def wait_until_connectable(address: str, log: Path, timeout: float = 30.0) -> None:
    """Block until ``address`` accepts a connection.

    ``python -m rlmesh.serve`` prints its serving line before ``model.serve()``
    binds (python/rlmesh/src/rlmesh/serve.py), so the startup line alone does
    not mean the model is listening -- and it echoes the requested address, so
    a port lost between ``free_port`` and the bind surfaces here rather than as
    a deadline expiry inside a cell.
    """
    host, _, port = address.removeprefix("tcp://").rpartition(":")
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((host, int(port)), timeout=1.0):
                return
        except OSError:
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"model server never accepted a connection on {address}; see {log}"
                ) from None
            time.sleep(0.1)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def prepare_old_venv(new_python: Path, env: dict[str, str], harness: Harness) -> Path:
    """Build the fixture venv from the hash-pinned published wheel.

    ``RLMESH_CROSSVER_WHEEL_DIR`` resolves the wheel from a local directory
    instead of PyPI, for an offline run -- still through the lock, so a wheel
    that is not the published artifact (this tree's own build, say, which
    carries the same version) fails the install. Nothing is vendored in the
    repo: the pin is the contract and PyPI is the source of record.
    """
    if OLD_VENV.exists():
        shutil.rmtree(OLD_VENV)
    harness.run(
        ["uv", "venv", "--python", str(new_python), str(OLD_VENV)],
        env=env,
        label="create fixture venv",
        log="crossver-old-venv",
    )
    python = OLD_VENV / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    pip = ["uv", "--no-config", "pip", "install", "--python", str(python)]
    offline = os.environ.get("RLMESH_CROSSVER_WHEEL_DIR")
    source = (["--no-index", "--find-links", offline] if offline else []) + [
        "--require-hashes",
        "-r",
        str(LOCK_PATH),
    ]
    harness.run(
        [*pip, "--no-deps", *source],
        env=env,
        label=f"install rlmesh=={FIXTURE_VERSION}",
        log="crossver-old-rlmesh",
    )
    harness.run(
        [*pip, "numpy"],
        env=env,
        label="install fixture venv dependencies",
        log="crossver-old-deps",
    )
    harness.run(
        [*pip, "--reinstall", str(FIXTURE_DIR)],
        env=env,
        label="install system fixtures",
        log="crossver-old-fixtures",
    )
    return python


def probe(
    harness: Harness,
    new_python: Path,
    env: dict[str, str],
    *,
    address: str,
    peer: str,
    can: list[str],
    log: str,
    generation: str = "rlmesh-wire-v1",
    configure: str | None = None,
) -> dict[str, Any]:
    """Run the raw wire probe in the new venv, the one carrying grpcio."""
    command = [
        str(new_python),
        str(SELF),
        "probe",
        "--address",
        address,
        "--peer",
        peer,
        "--generation",
        generation,
    ]
    for edition in can:
        command.extend(["--can", edition])
    if configure is not None:
        command.extend(["--configure", configure])
    log_path = harness.run(
        command, env=env, label=f"probe {peer} ({log})", log=log, timeout=120.0
    )
    return json.loads(log_path.read_text().splitlines()[-1])


def start_env_server(
    harness: Harness,
    python: Path,
    scenario: ScenarioSpec,
    env: dict[str, str],
    log: str,
) -> tuple[ManagedProcess, str]:
    from rlmesh_system.runner import SERVER_ADDRESS_RE, env_server_command

    return harness.start(
        env_server_command(python, scenario),
        env=env,
        label=f"start env server ({log})",
        log=log,
        pattern=SERVER_ADDRESS_RE,
    )


def compare_trace(current: Path, slug: str) -> str:
    from rlmesh_system.runner import assert_trace_matches

    assert_trace_matches(BASELINE, current, LOGS / f"{slug}-trace.log")
    return "match"


def env_leg_cell(
    harness: Harness,
    *,
    number: int,
    name: str,
    env_python: Path,
    client_python: Path,
    client_edition: str,
    server_want: str,
    env_logs_pin: bool,
    runtime_pins: bool,
    scenario: ScenarioSpec,
    env: dict[str, str],
    new_python: Path,
    expected: str,
) -> Cell:
    """One env-leg cell.

    ``runtime_pins``: the runtime is this tree's build, so it sends the real
    ``ConfigureEnv`` pin (a refusal aborts before Reset, so the trace proves it
    was accepted); otherwise the old runtime sends none and the probe forges the
    pin to show the server would accept it. ``env_logs_pin``: the env server is
    this tree's build, so its log names every pin it accepted -- exactly the
    runtime's in cell 4, exactly the probe's in cell 2.
    """
    slug = f"cell{number}"
    env_log = LOGS / f"{slug}-env.log"
    server, address = start_env_server(
        harness,
        env_python,
        scenario,
        dict(env, RUST_LOG="rlmesh=info") if env_logs_pin else env,
        f"{slug}-env",
    )
    try:
        wire = probe(
            harness,
            new_python,
            env,
            address=address,
            peer="env",
            can=[client_edition],
            configure=None if runtime_pins else expected,
            log=f"{slug}-probe",
        )
        handshake = wire["handshake"]
        compatible = bool(handshake["compatible"])
        if not compatible:
            raise AssertionError(
                f"{name}: handshake refused: {handshake['error_message']}"
            )
        assert_want(name, "env", handshake, server_want)
        served = list(handshake["supported_workflow_editions"])
        negotiated = sorted(set(served) & {client_edition})
        if negotiated != [expected]:
            raise AssertionError(
                f"{name}: env CAN {served} and runtime CAN [{client_edition!r}] share "
                f"{negotiated or 'nothing'}, expected [{expected!r}]"
            )
        if not runtime_pins and not wire["configure"]["accepted"]:
            raise AssertionError(
                f"{name}: env refused a pin of {expected!r}: {wire['configure']['message']}"
            )

        trace_path = TRACES / f"{slug}.trace.json"
        harness.run(
            trace_command(client_python, scenario, address, trace_path),
            env=env,
            label=f"run {name}",
            log=f"{slug}-driver",
            timeout=300.0,
        )
        trace = compare_trace(trace_path, slug)
    finally:
        server.stop()

    if env_logs_pin:
        pinned = ENV_PINNED_EDITION_RE.findall(ANSI_RE.sub("", env_log.read_text()))
        source = (
            "the runtime" if runtime_pins else "the probe (the old runtime sends none)"
        )
        if pinned != [expected]:
            raise AssertionError(
                f"{name}: the env server logged pins {pinned}, expected exactly "
                f"[{expected!r}] from {source}; see {env_log}"
            )
    return Cell(number, name, compatible, expected, trace, "pass", "")


def assert_want(name: str, peer: str, handshake: dict[str, Any], expected: str) -> None:
    """The server's own declared WANT, straight off its HandshakeResponse."""
    want = str(handshake["preferred_workflow_edition"])
    if want != expected:
        raise AssertionError(
            f"{name}: {peer} server declared WANT {want!r}, expected {expected!r}"
        )


def trace_command(
    python: Path,
    scenario: ScenarioSpec,
    address: str,
    output: Path,
    *,
    model_address: str | None = None,
) -> list[str]:
    """The fixture driver, in-process model by default, served model with an
    address -- one driver, so both legs record the same trace shape."""
    source = (
        ["--model-address", model_address]
        if model_address is not None
        else ["--model", str(scenario.model)]
    )
    return [
        str(python),
        "-m",
        "rlmesh_system_fixtures.driver",
        "trace",
        "--scenario",
        SCENARIO_NAME,
        "--address",
        address,
        "--client",
        scenario.client,
        *source,
        "--steps",
        str(scenario.steps),
        "--seed",
        str(scenario.seed),
        "--output",
        str(output),
    ]


def model_leg_cell(
    harness: Harness,
    *,
    number: int,
    name: str,
    model_python: Path,
    runtime_python: Path,
    env_python: Path,
    client_edition: str,
    server_want: str,
    scenario: ScenarioSpec,
    env: dict[str, str],
    new_python: Path,
    expected: str,
) -> Cell:
    slug = f"cell{number}"
    server, env_address = start_env_server(
        harness, env_python, scenario, env, f"{slug}-env"
    )
    model_log = LOGS / f"{slug}-model.log"
    model, model_address = harness.start(
        [
            str(model_python),
            "-u",
            "-m",
            "rlmesh.serve",
            MODEL_ENTRYPOINT,
            "--address",
            f"127.0.0.1:{free_port()}",
        ],
        # The served model logs the runtime's edition pin at debug; that line is
        # where this cell reads the negotiated edition from.
        env=dict(env, RUST_LOG="rlmesh=debug"),
        label=f"start model server ({slug})",
        log=f"{slug}-model",
        pattern=MODEL_SERVING_RE,
    )
    try:
        wait_until_connectable(model_address, model_log)
        handshake = probe(
            harness,
            new_python,
            env,
            address=model_address,
            peer="model",
            can=[client_edition],
            log=f"{slug}-probe",
        )["handshake"]
        compatible = bool(handshake["compatible"])
        if not compatible:
            raise AssertionError(
                f"{name}: handshake refused: {handshake['error_message']}"
            )
        assert_want(name, "model", handshake, server_want)

        trace_path = TRACES / f"{slug}.trace.json"
        harness.run(
            trace_command(
                runtime_python,
                scenario,
                env_address,
                trace_path,
                model_address=model_address,
            ),
            env=env,
            label=f"run {name}",
            log=f"{slug}-driver",
            timeout=300.0,
        )
        trace = compare_trace(trace_path, slug)
    finally:
        model.stop()
        server.stop()

    pinned = set(PINNED_EDITION_RE.findall(ANSI_RE.sub("", model_log.read_text())))
    if not pinned:
        raise AssertionError(
            f"{name}: the model server never logged the runtime's edition pin; see {model_log}"
        )
    if pinned != {expected}:
        raise AssertionError(
            f"{name}: model leg negotiated {sorted(pinned)}, expected [{expected!r}]"
        )
    return Cell(number, name, compatible, expected, trace, "pass", "")


def refusals(
    harness: Harness,
    *,
    python: Path,
    build: str,
    scenario: ScenarioSpec,
    env: dict[str, str],
    new_python: Path,
    expected: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Forge both refusals against one env server of ``build``."""
    slug = f"refusal-{build}"
    server, address = start_env_server(harness, python, scenario, env, f"{slug}-env")
    try:
        pin = probe(
            harness,
            new_python,
            env,
            address=address,
            peer="env",
            can=[expected],
            configure=UNKNOWN_EDITION,
            log=f"{slug}-pin",
        )["configure"]
        generation = probe(
            harness,
            new_python,
            env,
            address=address,
            peer="env",
            can=[expected],
            generation=FORGED_GENERATION,
            log=f"{slug}-generation",
        )["handshake"]
    finally:
        server.stop()

    if pin["accepted"]:
        raise AssertionError(f"{build} env accepted a pin of {UNKNOWN_EDITION!r}")
    message = str(pin["message"])
    if UNKNOWN_EDITION not in message or expected not in message:
        raise AssertionError(
            f"{build} env refused the pin without naming both sets: {message!r}"
        )
    if generation["compatible"]:
        raise AssertionError(f"{build} env accepted generation {FORGED_GENERATION!r}")
    error = str(generation["error_message"])
    if FORGED_GENERATION not in error:
        raise AssertionError(
            f"{build} env refused the generation unhelpfully: {error!r}"
        )
    return pin, generation


def run_matrix(args: argparse.Namespace) -> int:
    from rlmesh_system.support.manifest import load_specs

    assert_lock_pins_fixture()
    expected = expected_edition()
    scenario = load_specs(PROFILE_DIR).scenarios[SCENARIO_NAME]

    raw_python = os.environ.get("RLMESH_SYSTEM_PYTHON")
    if not raw_python:
        raise SystemExit(
            "RLMESH_SYSTEM_PYTHON is unset: run the matrix through "
            "`mise run test:crossver`, which builds the release wheel and lets "
            "the system runner create the new-side venv"
        )
    new_python = Path(raw_python)
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(WORK_DIR / "uv-cache"))
    env.setdefault("RUST_LOG", "warn")
    env["PYTHONUNBUFFERED"] = "1"
    TRACES.mkdir(parents=True, exist_ok=True)

    harness = Harness(verbose=args.verbose)
    new_version, new_build = read_build(new_python, "new wheel")
    if new_build != expected:
        raise SystemExit(
            f"the new wheel stamps edition {new_build!r}, not rlmesh.toml's "
            f"{expected!r}: build it with RLMESH_RELEASE_BUILD=1 so it lands in "
            "the published wheel's cohort (`mise run test:crossver`)"
        )

    old_python = prepare_old_venv(new_python, env, harness)
    old_version, old_build = read_build(old_python, "fixture wheel")
    if old_version != FIXTURE_VERSION:
        raise SystemExit(
            f"fixture venv installed rlmesh {old_version}, not {FIXTURE_VERSION}"
        )

    print(f"new: rlmesh {new_version} edition {new_build}")
    print(f"old: rlmesh {old_version} edition {old_build} (pinned by {LOCK_PATH.name})")

    # A server's WANT is its own declared edition: this tree declares the
    # current one, the published wheel declares nothing at all -- the
    # undeclared-peer case the negotiation has to tolerate (plan section A.2).
    old_want = ""
    shared = {
        "scenario": scenario,
        "env": env,
        "new_python": new_python,
        "expected": expected,
    }
    cells = [
        env_leg_cell(
            harness,
            number=3,
            name="control: old env server + old runtime",
            env_python=old_python,
            client_python=old_python,
            client_edition=old_build,
            server_want=old_want,
            env_logs_pin=False,
            runtime_pins=False,
            **shared,
        ),
        env_leg_cell(
            harness,
            number=4,
            name="control: new env server + new runtime",
            env_python=new_python,
            client_python=new_python,
            client_edition=new_build,
            server_want=expected,
            env_logs_pin=True,
            runtime_pins=True,
            **shared,
        ),
        env_leg_cell(
            harness,
            number=1,
            name="old env server + new runtime",
            env_python=old_python,
            client_python=new_python,
            client_edition=new_build,
            server_want=old_want,
            env_logs_pin=False,
            runtime_pins=True,
            **shared,
        ),
        env_leg_cell(
            harness,
            number=2,
            name="new env server + old runtime",
            env_python=new_python,
            client_python=old_python,
            client_edition=old_build,
            server_want=expected,
            env_logs_pin=True,
            runtime_pins=False,
            **shared,
        ),
        model_leg_cell(
            harness,
            number=5,
            name="old model server + new runtime",
            model_python=old_python,
            runtime_python=new_python,
            env_python=new_python,
            client_edition=new_build,
            server_want=old_want,
            **shared,
        ),
        model_leg_cell(
            harness,
            number=6,
            name="new model server + old runtime",
            model_python=new_python,
            runtime_python=old_python,
            env_python=old_python,
            client_edition=old_build,
            server_want=expected,
            **shared,
        ),
    ]

    new_pin, new_generation = refusals(
        harness, python=new_python, build="new", **shared
    )
    old_pin, old_generation = refusals(
        harness, python=old_python, build="old", **shared
    )
    cells += [
        Cell(
            9,
            f"pin of {UNKNOWN_EDITION} refused by both builds",
            None,
            None,
            "n/a",
            "pass",
            f"new: {new_pin['message']} || old: {old_pin['message']}",
        ),
        Cell(
            10,
            f"forged {FORGED_GENERATION} refused by both builds",
            False,
            None,
            "n/a",
            "pass",
            f"new: {new_generation['error_message']} || old: {old_generation['error_message']}",
        ),
        # Cells 7 and 8 exercise selection *between* editions. They are listed,
        # never silently dropped, and become real once a second edition is sealed
        # and the explicit pin surfaces land.
        Cell(
            7,
            "old env + a model rebuilt here, authored for the sealed edition",
            None,
            None,
            "pending",
            "pending",
            "needs a second sealed edition; with one edition it is cell 1",
        ),
        Cell(
            8,
            "both peers explicitly pinned to the older of two editions",
            None,
            None,
            "pending",
            "pending",
            "needs a second sealed edition; the explicit pin surfaces exist",
        ),
    ]

    cells.sort(key=lambda cell: cell.number)
    print_results(cells)
    return 0


def print_results(cells: list[Cell]) -> None:
    """Print the per-cell table, and leave a copy where the mise task can show it.

    The runner captures an external scenario's output into its log, so without
    the copy a green run says nothing about what it proved.
    """
    header = ("#", "cell", "compatible", "edition", "trace", "status")
    rows = [
        (
            str(cell.number),
            cell.name,
            "-" if cell.compatible is None else str(cell.compatible).lower(),
            cell.edition or "-",
            cell.trace,
            cell.status,
        )
        for cell in cells
    ]
    widths = [
        max(len(row[index]) for row in (header, *rows)) for index in range(len(header))
    ]

    def line(row: tuple[str, ...]) -> str:
        return "  ".join(
            value.ljust(width) for value, width in zip(row, widths, strict=True)
        )

    report = [line(header), "  ".join("-" * width for width in widths)]
    report.extend(line(row) for row in rows)
    report.append("")
    report.extend(f"cell {cell.number}: {cell.detail}" for cell in cells if cell.detail)
    report.append("")
    report.extend(LEGEND)
    text = "\n".join(report) + "\n"
    (TRACES / "matrix.txt").write_text(text)
    print()
    print(text, end="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RLMesh cross-version compatibility matrix."
    )
    parser.add_argument("--verbose", action="store_true", help="print command output")
    subparsers = parser.add_subparsers(dest="command")
    parser.set_defaults(command="matrix")
    _ = subparsers.add_parser("matrix", help="run the whole cross-version matrix")

    probe_parser = subparsers.add_parser("probe", help="raw wire handshake/pin probe")
    _ = probe_parser.add_argument("--address", required=True)
    _ = probe_parser.add_argument("--peer", choices=["env", "model"], required=True)
    _ = probe_parser.add_argument("--generation", default="rlmesh-wire-v1")
    _ = probe_parser.add_argument("--can", action="append", default=[])
    _ = probe_parser.add_argument("--configure")
    _ = probe_parser.add_argument("--timeout", type=float, default=30.0)

    args = parser.parse_args(argv)
    if args.command == "probe":
        return run_probe(args)
    return run_matrix(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130) from None
