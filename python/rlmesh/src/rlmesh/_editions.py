"""Where a workflow edition is declared, and which declaration wins.

A workflow edition is a *sticky declaration*, not a version ceiling: a
participant states the edition it was authored against and keeps it until its
author deliberately bumps it, so upgrading the rlmesh package never changes how
that participant behaves. This module is the single Python-side resolver for
that declaration -- every surface that can carry one funnels through
:func:`resolve_workflow_edition`, so the precedence is written down once.

Precedence, highest first:

1. an explicit call keyword (``run(workflow_edition=...)`` /
   ``session(workflow_edition=...)``);
2. the ``RLMESH_WORKFLOW_EDITION`` environment variable (per process);
3. ``ServeOptions(workflow_edition=...)`` / ``--workflow-edition`` (per served
   peer);
4. the :attr:`rlmesh.EnvFactory.workflow_edition` / :attr:`rlmesh.Model.workflow_edition`
   class declaration (source-resident);
5. ``[tool.rlmesh] workflow_edition`` in the ``pyproject.toml`` of the project
   that contains the authored class's module, else of the working directory
   (project-resident);
6. nothing -- the peer floats to this build's newest edition.

Running and serving resolve quietly. A float is reported where an author
reviews a participant before an upgrade: ``rlmesh check``, ``rlmesh
check-image`` and ``rlmesh doctor`` (see :func:`float_nudge`), and a session
refused because its peers share no edition.

An empty ``RLMESH_WORKFLOW_EDITION`` short-circuits the chain at rung 2: it
declares nothing *deliberately*, so rungs 3-5 are not consulted and no check
reports the float.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ._load_native import load_native

if TYPE_CHECKING:
    from collections.abc import Callable

    from rlmesh._rlmesh import ServeOptions

__all__ = [
    "WorkflowEditionWarning",
    "current_workflow_edition",
    "resolve_workflow_edition",
]


class WorkflowEditionWarning(UserWarning):
    """The category of rlmesh's workflow-edition warnings.

    Running and serving an undeclared participant do not warn; ``rlmesh
    check``, ``check-image`` and ``doctor`` report the float instead. Filter
    it by category (``warnings.filterwarnings("ignore",
    category=rlmesh.WorkflowEditionWarning)``) rather than by message.
    """


#: Process-wide declaration, above every in-code surface but a call keyword.
#: Set it to the empty string to declare nothing *deliberately*: resolution stops
#: there -- no lower surface is consulted -- the peer floats to this build's
#: newest edition, and no check reports it. That is what this repo's own test
#: suites do.
WORKFLOW_EDITION_ENV_VAR: Final = "RLMESH_WORKFLOW_EDITION"

#: The ``[tool.rlmesh]`` table and key read from the project's ``pyproject.toml``.
_PYPROJECT_TABLE: Final = "[tool.rlmesh]"
_PYPROJECT_KEY: Final = "workflow_edition"

#: This package's own directory: a class defined in rlmesh (a ``Model``
#: wrapping a callable) names no user project.
_PACKAGE_DIR: Final = Path(__file__).resolve().parent


def current_workflow_edition() -> str:
    """The bare ``YYYY.MM`` edition this build runs at -- what to paste into a declaration.

    The same value on every build of that edition (``"2026.10"``): a declaration
    names the contract a participant was authored against, and a bare base
    selects whichever spelling of it both sides offer -- the sealed name on a
    release, this build's cohort (``"2026.10-<version>"`` on a prerelease,
    ``"2026.10-dev.<sha>"`` on a local build: what
    ``rlmesh.build_info().workflow_edition`` reports) on a prerelease or local
    build. That cohort spelling is also accepted as a
    declaration, and pins to that exact moving build.
    """
    return str(load_native("current_workflow_edition")())


def validate_workflow_edition(edition: str) -> str:
    """Return ``edition`` trimmed, or raise :class:`ValueError` naming what this build offers.

    Accepts exactly the editions this build can actually run a session at: the
    bare base of any edition it retains, or a cohort spelling that admits one it
    offers. A declaration is a ceiling, so a cohort below everything in
    ``SUPPORTED_WORKFLOW_EDITIONS`` is refused here rather than deadlocking
    every negotiation.
    """
    return str(load_native("validate_workflow_edition")(edition))


def resolve_workflow_edition(
    *,
    call: str | None = None,
    option: str | None = None,
    declared: str | None = None,
    origin: object | None = None,
) -> str | None:
    """The workflow edition this participant declares, or ``None`` when it declares none.

    Args:
        call: An explicit ``workflow_edition=`` keyword on this call.
        option: A served peer's ``ServeOptions`` / ``--workflow-edition`` value.
        declared: The class-level declaration (``EnvFactory.workflow_edition`` /
            ``Model.workflow_edition``).
        origin: The authored class (or an instance or function of it) whose
            project's ``pyproject.toml`` is the project rung. Without one, or
            when its module lies in no project, the working directory's.

    Returns:
        The winning declaration, validated and trimmed, or ``None`` when no
        surface declares one, in which case the peer floats to this build's
        newest edition. An empty ``RLMESH_WORKFLOW_EDITION`` ends resolution
        without consulting ``option`` / ``declared`` / ``pyproject.toml``.

    Raises:
        ValueError: If the winning value names an edition this build cannot run.
    """
    return resolve_declaration(
        call=call, option=option, declared=declared, origin=origin
    )[0]


def float_nudge() -> str:
    """What a check says about a participant that floats: where to pin, and why."""
    edition = current_workflow_edition()
    return (
        f"no workflow edition declared: it floats to {edition!r}, which moves when "
        f'rlmesh upgrades. Pin it before you upgrade: workflow_edition = "{edition}" '
        "on the class or under [tool.rlmesh] in pyproject.toml "
        f"({WORKFLOW_EDITION_ENV_VAR}='' floats deliberately)"
    )


def resolve_declaration(
    *,
    call: str | None = None,
    option: str | None = None,
    declared: str | None = None,
    origin: object | None = None,
) -> tuple[str | None, bool]:
    """:func:`resolve_workflow_edition`, and whether it floats for want of a declaration.

    The flag is ``False`` for a declaration and for the deliberate float (an
    empty ``RLMESH_WORKFLOW_EDITION``): only an undeclared float is reported.
    """
    # Rungs 1-4 are the native resolver's, shared with every language (COR-6);
    # Python layers its project-file rung on top (D-7).
    resolved, deliberate = load_native("resolve_workflow_edition_rungs")(
        call=call, option=option, declared=declared
    )
    if resolved is not None:
        return str(resolved), False
    if deliberate:
        return None, False
    project = _from_pyproject(origin)
    if project is not None and project.strip():
        return validate_workflow_edition(project), False
    return None, True


#: The project-manifest rung, scanned once per starting directory: every
#: server or session a process opens would otherwise walk the same ancestors.
_pyproject_cache: dict[Path, str | None] = {}


def _from_pyproject(origin: object | None = None) -> str | None:
    """``[tool.rlmesh] workflow_edition`` from the project that holds ``origin``.

    The project is the nearest ``pyproject.toml`` above ``origin``'s module
    file, so the answer does not change with where the process starts; the
    working directory's is the fallback for an origin with no file or no
    enclosing project.
    """
    manifest = _origin_manifest(origin)
    if manifest is not None:
        return _cached(manifest.parent, lambda: _manifest_edition(manifest))
    start = Path.cwd()
    return _cached(start, lambda: _pyproject_edition(start))


def _cached(key: Path, scan: Callable[[], str | None]) -> str | None:
    if key not in _pyproject_cache:
        _pyproject_cache[key] = scan()
    return _pyproject_cache[key]


def _origin_manifest(origin: object | None) -> Path | None:
    """The ``pyproject.toml`` of the project ``origin``'s module lies in, if any."""
    if origin is None:
        return None
    module_name = getattr(origin, "__module__", None)
    if not isinstance(module_name, str):
        module_name = getattr(type(origin), "__module__", None)
    module = sys.modules.get(module_name) if isinstance(module_name, str) else None
    filename = getattr(module, "__file__", None)
    if not isinstance(filename, str):
        return None
    directory = Path(filename).resolve().parent
    if directory == _PACKAGE_DIR or _PACKAGE_DIR in directory.parents:
        return None
    for candidate in (directory, *directory.parents):
        manifest = candidate / "pyproject.toml"
        if manifest.is_file():
            return manifest
    return None


def _pyproject_edition(start: Path) -> str | None:
    for directory in (start, *start.parents):
        manifest = directory / "pyproject.toml"
        if manifest.is_file():
            return _manifest_edition(manifest)
    return None


def _manifest_edition(manifest: Path) -> str | None:
    try:
        return _scan_table_string(manifest.read_text(encoding="utf-8"))
    except OSError:
        # An unreadable manifest is not this resolver's error to raise: the
        # project is simply not declaring an edition here.
        return None


def _scan_table_string(text: str) -> str | None:
    """Read ``workflow_edition`` out of a ``[tool.rlmesh]`` table by line scan.

    A declaration is one quoted string on its own line, so this reads it without
    a TOML parser -- ``tomllib`` is 3.11+ and this package supports 3.10. The
    same single-key scan is what ``crates/rlmesh-proto/build.rs`` does to read
    ``rlmesh.toml`` before any TOML crate is available to it.
    """
    in_table = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("["):
            in_table = line == _PYPROJECT_TABLE
            continue
        if not in_table or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() != _PYPROJECT_KEY:
            continue
        value = value.strip()
        for quote in ('"', "'"):
            if len(value) >= 2 and value.startswith(quote) and value.endswith(quote):
                return value[1:-1]
        return None
    return None


def serve_options_declaring(
    options: ServeOptions | None = None,
    *,
    call: str | None = None,
    option: str | None = None,
    declared: str | None = None,
    origin: object | None = None,
) -> ServeOptions | None:
    """Return ``options`` carrying this peer's resolved edition declaration.

    ``option`` is the served-peer rung -- ``--workflow-edition``, or whatever
    ``options`` already carries. ``call`` is a declaration this call already
    resolved and that the server must adopt verbatim (the loopback env server
    ``run(workflow_edition=...)`` stands up is part of that one call, so it
    cannot be allowed to re-resolve to a different edition). The resolved winner
    is stamped onto a copy of ``options``; resolving to ``None`` clears any
    lower-priority declaration already carried by the options.
    """
    if option is None and options is not None:
        option = options.workflow_edition
    resolved = resolve_workflow_edition(
        call=call, option=option, declared=declared, origin=origin
    )
    if (resolved is None and options is None) or (
        options is not None and resolved == options.workflow_edition
    ):
        return options
    native: type[ServeOptions] = load_native("ServeOptions")
    if options is None:
        return native(workflow_edition=resolved)
    return native(
        close_timeout_seconds=options.close_timeout_seconds,
        workflow_edition=resolved,
    )
