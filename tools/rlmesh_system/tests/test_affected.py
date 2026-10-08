from __future__ import annotations

from pathlib import PurePosixPath

from scripts.dev.affected import Plan, Workspace, plan

# spaces <- runtime <- grpc <- python; viewer stands alone.
WORKSPACE = Workspace(
    dirs={
        "rlmesh-spaces": PurePosixPath("crates/rlmesh-spaces"),
        "rlmesh-runtime": PurePosixPath("crates/rlmesh-runtime"),
        "rlmesh-grpc": PurePosixPath("crates/rlmesh-grpc"),
        "rlmesh-viewer": PurePosixPath("crates/rlmesh-viewer"),
        "rlmesh-python": PurePosixPath("python/rlmesh/rust"),
    },
    dependents={
        "rlmesh-spaces": {"rlmesh-runtime"},
        "rlmesh-runtime": {"rlmesh-grpc"},
        "rlmesh-grpc": {"rlmesh-python"},
        "rlmesh-viewer": set(),
        "rlmesh-python": set(),
    },
)


def test_leaf_crate_change_selects_only_that_crate() -> None:
    assert plan(["crates/rlmesh-viewer/src/lib.rs"], WORKSPACE) == Plan(
        rust=["rlmesh-viewer"], python=False
    )


def test_change_reaches_every_dependent_and_the_extension() -> None:
    assert plan(["crates/rlmesh-spaces/src/box.rs"], WORKSPACE) == Plan(
        rust=["rlmesh-grpc", "rlmesh-python", "rlmesh-runtime", "rlmesh-spaces"],
        python=True,
    )


def test_python_only_change_skips_rust() -> None:
    assert plan(["python/rlmesh/src/rlmesh/env.py"], WORKSPACE) == Plan(
        rust=[], python=True
    )


def test_workspace_inputs_select_everything() -> None:
    assert plan(["Cargo.lock"], WORKSPACE) == Plan(rust=None, python=True)


def test_docs_only_change_selects_nothing() -> None:
    assert plan(["docs/testing.md", "README.md"], WORKSPACE) == Plan(
        rust=[], python=False
    )
