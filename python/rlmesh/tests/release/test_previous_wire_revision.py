"""A live session between this tree and the last release, across a wire revision change."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
VERSION_RS = "crates/rlmesh-session-core/src/wire/version.rs"
REVISION = re.compile(r"pub const WIRE_REVISION: u32 = (\d+);")

SERVE = """
import sys
import gymnasium as gym
import rlmesh

server = rlmesh.EnvServer(gym.make("CartPole-v1"), ":0/env")
server.start()
print("ADDRESS", server.address, flush=True)
sys.stdin.read()
server.shutdown()
"""

DIAL = """
import sys
import rlmesh.numpy as rn

try:
    rn.RemoteEnv(sys.argv[1], connect_timeout_seconds=20)
except Exception as exc:
    print("REFUSED", type(exc).__name__, exc, flush=True)
else:
    print("CONNECTED", flush=True)
"""


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def _pep440(version: str) -> str:
    for tag, short in (("-alpha.", "a"), ("-beta.", "b"), ("-rc.", "rc")):
        if tag in version:
            base, number = version.split(tag, 1)
            return f"{base}{short}{number}"
    return version


@pytest.fixture(scope="module")
def previous(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, int, int]:
    tag = _git("describe", "--tags", "--abbrev=0", "--match", "v*").strip()
    match = REVISION.search(_git("show", f"{tag}:{VERSION_RS}"))
    assert match, f"{tag} has no WIRE_REVISION"
    released = int(match.group(1))
    current_match = REVISION.search((ROOT / VERSION_RS).read_text(encoding="utf-8"))
    assert current_match
    current = int(current_match.group(1))
    if released == current:
        pytest.skip(f"{tag} speaks this tree's wire revision {current}")
    venv = tmp_path_factory.mktemp("previous") / "venv"
    subprocess.run(["uv", "venv", "-q", "-p", "3.11", str(venv)], check=True)
    python = venv / "bin" / "python"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-q",
            "-p",
            str(python),
            f"rlmesh[gymnasium,numpy]=={_pep440(tag.removeprefix('v'))}",
        ],
        check=True,
    )
    return python, released, current


def _session(host: Path | str, client: Path | str, runtime: Path) -> str:
    env = {**os.environ, "RLMESH_RUNTIME_DIR": str(runtime), "RLMESH_STALE_CHECK": "0"}
    server = subprocess.Popen(
        [str(host), "-c", SERVE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert server.stdout is not None
        address = ""
        for line in server.stdout:
            if line.startswith("ADDRESS "):
                address = line.split(" ", 1)[1].strip()
                break
        assert address, "the host printed no address"
        dial = subprocess.run(
            [str(client), "-c", DIAL, address],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        return dial.stdout.strip()
    finally:
        if server.stdin is not None:
            server.stdin.close()
        server.wait(timeout=30)


def test_a_previous_release_and_this_tree_refuse_each_other_by_name(
    previous: tuple[Path, int, int], tmp_path: Path
) -> None:
    old, released, current = previous
    for host, client in ((old, sys.executable), (sys.executable, old)):
        outcome = _session(host, client, tmp_path)
        assert outcome.startswith("REFUSED"), outcome
        assert "Incompatible" in outcome, outcome
        assert f"revision {released}" in outcome, outcome
        assert f"revision {current}" in outcome, outcome
