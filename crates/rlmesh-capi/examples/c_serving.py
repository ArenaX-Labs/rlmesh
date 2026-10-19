"""Probe the C foreground/start envelopes and model signal modes over gRPC."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import tomllib

import grpc

ROOT = Path(__file__).resolve().parents[3]
# Reuse the system harness's minimal protobuf reader, with no generated stubs.
sys.path.insert(0, str(ROOT / "tests/system"))
from crossver import _decode, _delimited, _message, _string, _text  # noqa: E402


def read_checked(process: subprocess.Popen[str]) -> None:
    """Wait for the native signal assertion without hanging on a failed child."""
    assert process.stdout is not None
    output = b""
    while b"checked signals\n" not in output:
        ready, _, _ = select.select([process.stdout], [], [], 10)
        assert ready, "C consumer did not finish its signal assertions"
        chunk = os.read(process.stdout.fileno(), 4096)
        assert chunk, "C consumer exited before checking signals"
        output += chunk


def probe(binary: str, mode: str) -> None:
    """Run one fresh process so signal dispositions cannot leak between cases."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{listener.getsockname()[1]}"
    is_env = mode in {
        "foreground",
        "start",
        "legacy-foreground",
        "legacy-start",
        "env-server",
    }
    process = subprocess.Popen(
        [binary, mode, f"tcp://{address}"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    channel = grpc.insecure_channel(address)
    try:
        grpc.channel_ready_future(channel).result(timeout=15)
        service = (
            "rlmesh.env.v1.EnvService" if is_env else "rlmesh.model.v1.ModelService"
        )
        generation = tomllib.loads((ROOT / "rlmesh.toml").read_text())["protocol"][
            "current_generation"
        ]
        reply = channel.unary_unary(
            f"/{service}/Handshake",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )(_delimited(1, _text(1, generation)), timeout=5)
        base = _message(_decode(reply), 1)
        assert base[1] == [1], base
        if is_env:
            peer = _message(base, 2)
            extra = {
                _string(_decode(bytes(v)), 1): _string(_decode(bytes(v)), 2)
                for v in peer.get(15, [])
            }
            envelope = json.loads(extra["rlmesh.describe.v1"])
            legacy = mode.startswith("legacy-")
            language, version = ("c", "") if legacy else ("csharp", "9.0")
            frameworks = {
                _string(_decode(bytes(v)), 1): _string(_decode(bytes(v)), 2)
                for v in peer.get(8, [])
            }
            assert _string(peer, 3) == language, peer
            assert _string(peer, 4) == version, peer
            assert frameworks == ({} if legacy else {"unity": "6"}), peer
            assert envelope["runtime"]["language"] == language, envelope
            if not legacy:
                assert envelope["runtime"]["language_version"] == version, envelope
                assert envelope["runtime"]["framework_versions"] == frameworks, envelope
        assert process.stdin is not None
        if "foreground" not in mode:
            process.stdin.write("i\n")
            process.stdin.flush()
            read_checked(process)
        if mode not in {"signals", "old-signals"}:
            process.stdin.write("q\n")
            process.stdin.flush()
        stdout, _ = process.communicate(timeout=15)
        assert process.returncode == 0, (mode, stdout)
    finally:
        channel.close()
        if process.poll() is None:
            process.kill()
            process.wait()


if __name__ == "__main__":
    for case in (
        "foreground",
        "start",
        "legacy-foreground",
        "legacy-start",
        "env-server",
        "no-signals",
        "signals",
        "old-signals",
    ):
        probe(sys.argv[1], case)
        print(f"C serving probe: {case} OK", flush=True)
