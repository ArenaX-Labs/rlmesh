from __future__ import annotations

from pathlib import Path, PurePosixPath

from scripts.dev.affected import (
    Plan,
    Workspace,
    csharp_dependency_snapshot,
    csharp_group_changed,
    plan,
    write_github_output,
)

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
        csharp=False,
    )
    assert plan(["crates/rlmesh-spaces/src/box.rs"], WORKSPACE, ci=True).csharp


def test_python_only_change_skips_rust() -> None:
    assert plan(["python/rlmesh/src/rlmesh/env.py"], WORKSPACE) == Plan(
        rust=[], python=True
    )


def test_workspace_inputs_select_everything() -> None:
    assert plan(["Cargo.lock"], WORKSPACE) == Plan(rust=None, python=True)
    assert plan(["Cargo.lock"], WORKSPACE, ci=True) == Plan(
        rust=None, python=True, csharp=True
    )


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


def test_unity_changes_select_standalone_csharp_tests() -> None:
    for path in (
        "integrations/unity/Runtime/Core/MainThreadDispatcher.cs",
        "integrations/unity/Editor/ReloadPreflight.cs",
        "integrations/unity/Tests~/client.py",
    ):
        assert plan([path], WORKSPACE) == Plan(rust=[], python=False, csharp=True)


def test_core_change_reaches_csharp_through_capi_only_in_ci() -> None:
    assert plan(["crates/rlmesh/src/env/sync_wire.rs"], WORKSPACE) == Plan(
        rust=["rlmesh", "rlmesh-capi"], python=False
    )
    assert plan(["crates/rlmesh/src/env/sync_wire.rs"], WORKSPACE, ci=True).csharp


def test_scoped_csharp_tools_do_not_select_other_suites() -> None:
    for path in ("csharp/mise.toml", "csharp/mise.lock"):
        assert plan([path], WORKSPACE) == Plan(rust=[], python=False, csharp=True)


def test_ci_output_uses_broad_selection_and_appends(tmp_path: Path) -> None:
    output = tmp_path / "github-output"
    output.write_text("existing=value\n")
    write_github_output(plan(["crates/rlmesh/src/env.rs"], WORKSPACE, ci=True), output)
    write_github_output(plan(["crates/rlmesh-viewer/src/lib.rs"], WORKSPACE), output)
    assert output.read_text() == "existing=value\ncsharp=true\ncsharp=false\n"


def test_ci_only_inputs_do_not_require_local_dotnet() -> None:
    for path in (
        "pyproject.toml",
        "uv.lock",
        "python/rlmesh/src/rlmesh/env.py",
        ".mise/tasks/release/cxx/package",
        ".github/workflows/ci.yml",
    ):
        assert not plan([path], WORKSPACE).csharp, path
        assert plan([path], WORKSPACE, ci=True).csharp, path


def test_direct_inputs_select_csharp_locally_and_in_ci() -> None:
    for path in (
        "csharp/src/RLMesh/Env.cs",
        "integrations/unity/Tests~/step_client.cpp",
        "tools/rlmesh_csharp_surface/Program.cs",
        "crates/rlmesh-capi/src/env.rs",
        "crates/rlmesh-capi/include/rlmesh.h",
        "scripts/dev/affected.py",
    ):
        for ci in (False, True):
            assert plan([path], WORKSPACE, ci=ci).csharp, path


def test_csharp_dependency_group_change_selects_local_csharp() -> None:
    assert plan(["uv.lock"], WORKSPACE, csharp_dependencies_changed=True).csharp


PROJECT = '\n[dependency-groups]\ncsharp = ["client[image]"]\ndev = ["unrelated"]\n'
LOCK = "\n".join(
    (
        "[[package]]",
        'name = "workspace"',
        'source = {virtual = "."}',
        "[package.dev-dependencies]",
        'csharp = [{name = "client", extra = ["image"]}]',
        "[[package]]",
        'name = "client"',
        'version = "1.0"',
        'dependencies = [{name = "transitive"}]',
        "[package.optional-dependencies]",
        'image = [{name = "pixels"}]',
        'other = [{name = "unrelated"}]',
        "[[package]]",
        'name = "transitive"',
        'version = "2.0"',
        "[[package]]",
        'name = "pixels"',
        'version = "3.0"',
        "[[package]]",
        'name = "unrelated"',
        'version = "4.0"',
    )
)


def test_group_snapshot_ignores_unrelated_groups_and_unused_extras() -> None:
    before = csharp_dependency_snapshot(PROJECT, LOCK)
    assert before == csharp_dependency_snapshot(
        PROJECT.replace('dev = ["unrelated"]', 'dev = ["more-tools"]'),
        LOCK.replace('version = "4.0"', 'version = "4.1"'),
    )


def test_group_snapshot_tracks_transitive_and_selected_extra_packages() -> None:
    before = csharp_dependency_snapshot(PROJECT, LOCK)
    for version in ("1.0", "2.0", "3.0"):
        assert before != csharp_dependency_snapshot(
            PROJECT, LOCK.replace(f'version = "{version}"', 'version = "9.0"')
        )
    assert before != csharp_dependency_snapshot(
        PROJECT.replace("client[image]", "client"), LOCK
    )


def test_group_snapshot_tracks_included_dependency_groups() -> None:
    project = '[dependency-groups]\ncsharp = [{include-group = "interop"}]\ninterop = ["client"]'
    assert csharp_dependency_snapshot(project, LOCK) != csharp_dependency_snapshot(
        project.replace('interop = ["client"]', 'interop = ["other"]'), LOCK
    )


def test_group_comparison_reads_merge_base_and_current_files(
    tmp_path, monkeypatch
) -> None:
    from scripts.dev import affected

    monkeypatch.setattr(affected, "ROOT", tmp_path)
    previous = {"pyproject.toml": PROJECT, "uv.lock": LOCK}
    for name, value in previous.items():
        (tmp_path / name).write_text(value)

    def fake_git(*args):
        if args == ("merge-base", "HEAD", "base"):
            return "merge-base\n"
        assert args[0] == "show"
        assert args[1].startswith("merge-base:")
        return previous[args[1].split(":")[1]]

    monkeypatch.setattr(affected, "git", fake_git)
    assert not csharp_group_changed("base")
    (tmp_path / "uv.lock").write_text(
        LOCK.replace('version = "2.0"', 'version = "2.1"')
    )
    assert csharp_group_changed("base")
