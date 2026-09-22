"""Where a workflow edition is declared, and which declaration wins."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import rlmesh
from rlmesh import _editions
from rlmesh._editions import (
    WORKFLOW_EDITION_ENV_VAR,
    current_workflow_edition,
    resolve_workflow_edition,
    serve_options_declaring,
)
from rlmesh._rlmesh import ServeOptions

if TYPE_CHECKING:
    from collections.abc import Iterator

#: A well-formed `YYYY.MM` no build implements, so it parses but never resolves.
UNKNOWN_EDITION = "2099.01"

#: The bare base this build runs at -- the stateVersion value every build of the
#: edition accepts, whether it offers the sealed name or only a cohort of it.
EDITION = current_workflow_edition()

#: The exact spelling this build offers: the sealed base on a release, the
#: cohort (`2026.06-dev.<sha>`) on a prerelease or source build.
COHORT = rlmesh.build_info().workflow_edition


@pytest.fixture(autouse=True)
def _isolated_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    """Resolve against nothing: no env var, no project manifest, warning re-armed.

    The suite-wide conftest sets ``RLMESH_WORKFLOW_EDITION`` to the empty string
    (the documented quiet switch), and the repo's own ``pyproject.toml`` is an
    ancestor of the working directory -- both would mask what these tests are
    asserting.
    """
    monkeypatch.delenv(WORKFLOW_EDITION_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_editions, "_warned", False)
    monkeypatch.setattr(_editions, "_pyproject_cache", {})
    yield


def _two_step_env() -> Any:
    """A minimal Box-spaced env, enough for EnvServer to publish a contract."""
    gym = pytest.importorskip("gymnasium")
    np = pytest.importorskip("numpy")

    class TwoStepEnv:
        observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

        def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
            _ = seed, options
            return np.zeros(2, np.float32), {}

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
            _ = action
            return np.zeros(2, np.float32), 0.0, True, False, {}

        def close(self) -> None:
            return None

    return TwoStepEnv()


def write_pyproject(directory: Path, edition: str) -> None:
    """Put a ``[tool.rlmesh] workflow_edition`` manifest in ``directory``."""
    _ = (directory / "pyproject.toml").write_text(
        textwrap.dedent(f"""\
            [tool.rlmesh]
            workflow_edition = "{edition}"
            """),
        encoding="utf-8",
    )


def test_current_workflow_edition_is_the_bare_base_on_every_build() -> None:
    # The documented paste-into-a-declaration value is the contract's name, not
    # this build's spelling of it: the same on a release and on a dev build, and
    # accepted by both.
    edition = current_workflow_edition()
    assert rlmesh.current_workflow_edition() == edition
    assert "-" not in edition
    assert COHORT.partition("-")[0] == edition
    assert resolve_workflow_edition(call=edition) == edition


def test_undeclared_public_session_preserves_a_legacy_binder() -> None:
    sentinel = object()

    class LegacyHandle:
        def session(
            self,
            env: object,
            *,
            instruction: str | None,
            close_env: bool,
            trust_entrypoints: bool | None,
            execution_horizon: int,
            view: object,
        ) -> object:
            return sentinel

    assert rlmesh.session(LegacyHandle(), "tcp://127.0.0.1:1") is sentinel


def test_sandbox_refuses_an_invalid_call_pin_before_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_serve(self: object) -> None:
        pytest.fail("invalid edition started a container")

    monkeypatch.setattr(rlmesh.SandboxModel, "serve", unexpected_serve)
    handle = rlmesh.SandboxModel("image://unused:latest")
    with pytest.raises(ValueError, match=UNKNOWN_EDITION):
        rlmesh.session(handle, "tcp://127.0.0.1:1", workflow_edition=UNKNOWN_EDITION)


class TestPrecedence:
    """Each rung of the A.5 table, and that the rung above it wins."""

    def test_undeclared_resolves_to_nothing(self) -> None:
        with pytest.warns(UserWarning, match="no workflow edition declared"):
            assert resolve_workflow_edition() is None

    def test_pyproject_is_the_lowest_declaration(self, tmp_path: Path) -> None:
        write_pyproject(tmp_path, EDITION)
        assert resolve_workflow_edition() == EDITION

    def test_pyproject_is_found_from_a_subdirectory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_pyproject(tmp_path, EDITION)
        nested = tmp_path / "src" / "pkg"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)
        assert resolve_workflow_edition() == EDITION

    def test_class_declaration_beats_pyproject(self, tmp_path: Path) -> None:
        write_pyproject(tmp_path, UNKNOWN_EDITION)
        assert resolve_workflow_edition(declared=EDITION) == EDITION

    def test_serve_options_beat_the_class_declaration(self) -> None:
        assert (
            resolve_workflow_edition(option=EDITION, declared=UNKNOWN_EDITION)
            == EDITION
        )

    def test_env_var_beats_the_class_declaration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(WORKFLOW_EDITION_ENV_VAR, EDITION)
        assert resolve_workflow_edition(declared=UNKNOWN_EDITION) == EDITION

    def test_env_var_beats_serve_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The operator's process-wide override outranks what the program passed:
        # a deployment can pin an endpoint it did not author.
        monkeypatch.setenv(WORKFLOW_EDITION_ENV_VAR, EDITION)
        assert resolve_workflow_edition(option=UNKNOWN_EDITION) == EDITION

    def test_call_keyword_beats_the_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(WORKFLOW_EDITION_ENV_VAR, UNKNOWN_EDITION)
        assert resolve_workflow_edition(call=EDITION) == EDITION

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_declaration_is_no_declaration(self, blank: str) -> None:
        with pytest.warns(UserWarning, match="no workflow edition declared"):
            assert resolve_workflow_edition(call=blank, option=blank) is None


class TestRefusal:
    """A value this build cannot run is refused where it is typed."""

    @pytest.mark.parametrize("surface", ["call", "option", "declared"])
    def test_unknown_edition_is_refused_at_construction(self, surface: str) -> None:
        with pytest.raises(ValueError, match=UNKNOWN_EDITION) as refusal:
            _ = resolve_workflow_edition(**{surface: UNKNOWN_EDITION})
        # The message names both halves of the mismatch: what was asked for and
        # what this build actually implements.
        assert current_workflow_edition() in str(refusal.value)

    def test_unknown_edition_is_refused_by_serve_options(self) -> None:
        with pytest.raises(ValueError, match=UNKNOWN_EDITION):
            _ = ServeOptions(workflow_edition=UNKNOWN_EDITION)

    def test_the_base_and_this_builds_cohort_are_both_accepted(self) -> None:
        # The bare base is a base-level ceiling that admits every cohort of
        # itself, so it is declarable on every build; the build's own cohort
        # spelling pins to that exact moving build and is accepted too.
        assert resolve_workflow_edition(call=EDITION) == EDITION
        assert resolve_workflow_edition(call=COHORT) == COHORT

    def test_a_cohort_below_everything_this_build_offers_is_refused(self) -> None:
        # A stale prerelease cohort of this base admits nothing a newer build
        # offers, so declaring it would refuse every session; it is refused
        # where it is typed, naming the base to declare instead. A sealed
        # release offers the bare base, which any cohort of it admits.
        stale = f"{EDITION}-0.0.0"
        if COHORT == EDITION:
            assert resolve_workflow_edition(call=stale) == stale
            return
        with pytest.raises(ValueError, match=EDITION) as refusal:
            _ = resolve_workflow_edition(call=stale)
        assert stale in str(refusal.value)
        assert COHORT in str(refusal.value)


class TestOneTimeWarning:
    """The undeclared warning fires once per process, not once per call."""

    def test_warns_once(self, recwarn: pytest.WarningsRecorder) -> None:
        assert resolve_workflow_edition() is None
        assert resolve_workflow_edition() is None
        assert (
            len(
                [w for w in recwarn if "no workflow edition declared" in str(w.message)]
            )
            == 1
        )

    def test_names_the_edition_it_floated_to_and_how_to_declare(self) -> None:
        with pytest.warns(UserWarning) as records:
            _ = resolve_workflow_edition()
        message = str(records[0].message)
        assert current_workflow_edition() in message
        assert "workflow_edition" in message
        assert WORKFLOW_EDITION_ENV_VAR in message

    def test_the_empty_env_var_floats_deliberately_and_stays_quiet(
        self, monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
    ) -> None:
        monkeypatch.setenv(WORKFLOW_EDITION_ENV_VAR, "")
        assert resolve_workflow_edition() is None
        assert [
            w for w in recwarn if "no workflow edition declared" in str(w.message)
        ] == []

    def test_the_empty_env_var_short_circuits_every_lower_rung(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # "Deliberately undeclared" has to mean undeclared: a project or class
        # declaration below the variable must not float back up through it.
        write_pyproject(tmp_path, EDITION)
        monkeypatch.setenv(WORKFLOW_EDITION_ENV_VAR, "")
        assert resolve_workflow_edition() is None
        assert resolve_workflow_edition(option=EDITION) is None
        assert resolve_workflow_edition(declared=EDITION) is None
        # Only the call keyword outranks it.
        assert resolve_workflow_edition(call=EDITION) == EDITION

    def test_the_warning_is_attributed_to_the_caller(self) -> None:
        # Not to rlmesh's own resolver: every surface reaches it through a
        # different number of internal hops, and a user filtering by module has
        # to be able to name their own.
        with pytest.warns(UserWarning, match="no workflow edition declared") as records:
            _ = serve_options_declaring()
        assert Path(records[0].filename) == Path(__file__)

    def test_the_warning_names_the_env_server_construction_site(self) -> None:
        # One hop deeper than the resolver's direct callers: EnvServer.__init__.
        with pytest.warns(UserWarning, match="no workflow edition declared") as records:
            server = rlmesh.EnvServer(_two_step_env(), "127.0.0.1:0")
        server.shutdown()
        assert Path(records[0].filename) == Path(__file__)


class TestPyprojectScan:
    """The project rung is read once per working directory, not per server."""

    def test_the_scan_is_memoized_per_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_pyproject(tmp_path, EDITION)
        scans: list[Path] = []
        real = _editions._pyproject_edition  # pyright: ignore[reportPrivateUsage]

        def counting(start: Path) -> str | None:
            scans.append(start)
            return real(start)

        monkeypatch.setattr(_editions, "_pyproject_edition", counting)
        assert resolve_workflow_edition() == EDITION
        assert resolve_workflow_edition() == EDITION
        assert scans == [tmp_path]
        # A different working directory is a different answer.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert resolve_workflow_edition() == EDITION
        assert scans == [tmp_path, elsewhere]

    def test_a_higher_rung_never_consults_the_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scans: list[Path] = []
        monkeypatch.setattr(
            _editions, "_pyproject_edition", lambda start: scans.append(start)
        )
        assert resolve_workflow_edition(option=EDITION) == EDITION
        assert resolve_workflow_edition(declared=EDITION) == EDITION
        monkeypatch.setenv(WORKFLOW_EDITION_ENV_VAR, EDITION)
        assert resolve_workflow_edition() == EDITION
        assert scans == []


class TestServeOptionsDeclaring:
    """The declaration reaches the serve options a served peer hands the server."""

    def test_empty_environment_clears_an_existing_option_pin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(WORKFLOW_EDITION_ENV_VAR, "")
        given = ServeOptions(workflow_edition=EDITION, idle_timeout_seconds=2.5)
        options = serve_options_declaring(given)
        assert options is not None
        assert options.workflow_edition is None
        assert options.idle_timeout_seconds == 2.5
        assert given.workflow_edition == EDITION

    def test_stamps_the_resolved_declaration_onto_fresh_options(self) -> None:
        options = serve_options_declaring(declared=EDITION)
        assert options is not None
        assert options.workflow_edition == EDITION

    def test_preserves_the_other_lifecycle_fields(self) -> None:
        given = ServeOptions(allow_remote_shutdown=True, idle_timeout_seconds=2.5)
        options = serve_options_declaring(given, declared=EDITION)
        assert options is not None
        assert options.workflow_edition == EDITION
        assert options.allow_remote_shutdown
        assert options.idle_timeout_seconds == 2.5

    def test_an_explicit_option_outranks_the_class_declaration(self) -> None:
        given = ServeOptions(workflow_edition=EDITION)
        options = serve_options_declaring(given, declared=UNKNOWN_EDITION)
        assert options is not None
        assert options.workflow_edition == EDITION

    def test_declaring_nothing_leaves_the_options_untouched(self) -> None:
        given = ServeOptions(allow_remote_shutdown=True)
        with pytest.warns(UserWarning, match="no workflow edition declared"):
            assert serve_options_declaring(given) is given

    def test_an_already_resolved_call_declaration_is_adopted_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The loopback env server `run()` stands up is part of that one call, so
        # it takes the run's decision rather than re-resolving to another rung.
        monkeypatch.setenv(WORKFLOW_EDITION_ENV_VAR, UNKNOWN_EDITION)
        options = serve_options_declaring(call=EDITION)
        assert options is not None
        assert options.workflow_edition == EDITION

    def test_declaring_nothing_without_options_stays_none(self) -> None:
        # Byte-identical to a build without the field: no declaration goes out.
        with pytest.warns(UserWarning, match="no workflow edition declared"):
            assert serve_options_declaring() is None
