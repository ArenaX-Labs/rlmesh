from __future__ import annotations

from pathlib import Path, PurePosixPath

from scripts.dev.affected import Plan, Workspace, plan, write_github_output

# spaces <- runtime <- grpc <- python; viewer stands alone.
WORKSPACE = Workspace(
    dirs={
        "rlmesh-spaces": PurePosixPath("crates/rlmesh-spaces"),
        "rlmesh-runtime": PurePosixPath("crates/rlmesh-runtime"),
        "rlmesh-grpc": PurePosixPath("crates/rlmesh-grpc"),
        "rlmesh-viewer": PurePosixPath("crates/rlmesh-viewer"),
        "rlmesh-python": PurePosixPath("python/rlmesh/rust"),
        "rlmesh": PurePosixPath("crates/rlmesh"),
        "rlmesh-capi": PurePosixPath("crates/rlmesh-capi"),
    },
    dependents={
        "rlmesh-spaces": {"rlmesh-runtime"},
        "rlmesh-runtime": {"rlmesh-grpc", "rlmesh"},
        "rlmesh": {"rlmesh-capi"},
        "rlmesh-capi": set(),
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
        rust=[
            "rlmesh",
            "rlmesh-capi",
            "rlmesh-grpc",
            "rlmesh-python",
            "rlmesh-runtime",
            "rlmesh-spaces",
        ],
        python=True,
        csharp=True,
    )


def test_python_only_change_skips_rust() -> None:
    assert plan(["python/rlmesh/src/rlmesh/env.py"], WORKSPACE) == Plan(
        rust=[], python=True
    )


def test_workspace_inputs_select_everything() -> None:
    assert plan(["Cargo.lock"], WORKSPACE) == Plan(rust=None, python=True, csharp=True)


def test_docs_only_change_selects_nothing() -> None:
    assert plan(["docs/testing.md", "README.md"], WORKSPACE) == Plan(
        rust=[], python=False
    )


def test_csharp_and_c_header_select_csharp() -> None:
    for path in ("csharp/src/RLMesh/Env.cs", "crates/rlmesh-capi/include/rlmesh.h"):
        assert plan([path], WORKSPACE).csharp
    assert not plan(["docs/testing.md"], WORKSPACE).csharp


def test_capi_implementation_selects_csharp() -> None:
    assert plan(["crates/rlmesh-capi/src/env.rs"], WORKSPACE) == Plan(
        rust=["rlmesh-capi"], python=False, csharp=True
    )


def test_core_change_reaches_csharp_through_capi() -> None:
    assert plan(["crates/rlmesh/src/env/sync_wire.rs"], WORKSPACE) == Plan(
        rust=["rlmesh", "rlmesh-capi"], python=False, csharp=True
    )


def test_scoped_csharp_tools_do_not_select_other_suites() -> None:
    for path in ("csharp/mise.toml", "csharp/mise.lock"):
        assert plan([path], WORKSPACE) == Plan(rust=[], python=False, csharp=True)


def test_ci_output_uses_the_same_selection_and_appends(tmp_path: Path) -> None:
    output = tmp_path / "github-output"
    output.write_text("existing=value\n")
    write_github_output(plan(["crates/rlmesh/src/env.rs"], WORKSPACE), output)
    write_github_output(plan(["crates/rlmesh-viewer/src/lib.rs"], WORKSPACE), output)
    assert output.read_text() == "existing=value\ncsharp=true\ncsharp=false\n"
