from __future__ import annotations

import sys
from pathlib import Path

import pytest
import rlmesh
from sandbox_gate import LABEL, ROOT, GateImages, docker

pytestmark = pytest.mark.docker

EXAMPLE = ROOT / "examples/python/llm_agent"

FAKE_HARNESS = """\
from fake_llm import scripted
from harness import LLMHarness


class FakeHarness(LLMHarness):
    def __init__(self) -> None:
        super().__init__(model="fake", chat=scripted("", "5"))
"""

IMAGES = """\
ARG HARNESS
FROM ${HARNESS} AS model
COPY fake_harness.py .
ENV RLMESH_SERVE_TARGET=fake_harness:FakeHarness

FROM ${HARNESS} AS env
ENV RLMESH_OUTPUT_DIR=/rlmesh/output/env
ENTRYPOINT ["python", "-m", "rlmesh.serve", "--env", "chat_env:AdditionTask"]
"""


def test_harness_image_solves_addition_env_image(
    gate_images: GateImages,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = f"{gate_images.run}-llm-harness:gate"
    docker(
        "build",
        "--quiet",
        "--build-arg",
        f"BASE={gate_images.model}",
        "--build-arg",
        f"RLMESH_VERSION={rlmesh.__version__}",
        "--label",
        f"{LABEL}={gate_images.run}",
        "-t",
        harness,
        "-f",
        str(EXAMPLE / "docker/harness.Dockerfile"),
        str(EXAMPLE),
    )
    context = tmp_path / "context"
    context.mkdir()
    (context / "fake_harness.py").write_text(FAKE_HARNESS)
    (context / "Dockerfile").write_text(IMAGES)
    tags = {}
    for target in ("model", "env"):
        tags[target] = f"{gate_images.run}-llm-{target}:gate"
        docker(
            "build",
            "--quiet",
            "--target",
            target,
            "--build-arg",
            f"HARNESS={harness}",
            "--label",
            f"{LABEL}={gate_images.run}",
            "-t",
            tags[target],
            str(context),
        )

    monkeypatch.syspath_prepend(str(EXAMPLE))
    import run_docker

    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_docker.py",
            "--env-image",
            tags["env"],
            "--harness-image",
            tags["model"],
            "--output",
            str(output),
        ],
    )
    run_docker.main()

    assert "seed=0 steps=1 reward=1.000 success=True" in capsys.readouterr().out
    assert len(list((output / "transcripts").glob("*.jsonl"))) == 1
    assert len(list((output / "env").glob("*.jsonl"))) == 1
