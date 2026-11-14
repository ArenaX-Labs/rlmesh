"""A gym-shaped sandbox source that also names a local Docker image, against
a real Docker daemon (``mise run test:sandbox:docker``)."""

from __future__ import annotations

import os

import pytest
from rlmesh._sandbox import _sources
from sandbox_gate import LABEL, docker, docker_missing, remove_labelled, run_id

pytestmark = pytest.mark.docker


def test_a_local_image_named_like_a_gym_id_runs_as_the_image(
    tmp_path: os.PathLike[str],
) -> None:
    if missing := docker_missing():
        pytest.skip(f"needs a Docker daemon: {missing}")
    run = f"{run_id()}-sources"
    # Gym's -v<N> suffix and no tag: before, this always built from source.
    image = f"{run}-v1"
    assert _sources.looks_like_gym_id(image)
    assert _sources.resolve_source_kind(image) == ("build", image)
    context = os.fspath(tmp_path)
    with open(os.path.join(context, "Dockerfile"), "w", encoding="utf-8") as file:
        file.write(f'FROM scratch\nLABEL {LABEL}="{run}"\nCMD ["serve"]\n')
    try:
        docker("build", "-q", "-t", image, context)
        assert _sources.resolve_source_kind(image) == ("prebuilt", image)
        with pytest.warns(UserWarning, match=f"docker://{image}"):
            assert _sources.resolve_source_kind(image, prefer_build=True) == (
                "build",
                image,
            )
    finally:
        remove_labelled(run)
