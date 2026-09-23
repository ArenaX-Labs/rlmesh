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
5. ``[tool.rlmesh] workflow_edition`` in the project's ``pyproject.toml``
   (project-resident);
6. nothing -- the peer floats to this build's newest edition. A participant
   with a home for the declaration (an authored ``EnvFactory``/``Model``
   subclass, or anything served) says so once per process as a
   :class:`WorkflowEditionWarning`; an ad hoc callable in a local run floats
   quietly.

An empty ``RLMESH_WORKFLOW_EDITION`` short-circuits the chain at rung 2: it
declares nothing *deliberately*, so rungs 3-5 are not consulted and the float is
not reported.
"""

from __future__ import annotations

import contextvars
import inspect
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ._load_native import load_native

if TYPE_CHECKING:
    from rlmesh._rlmesh import ServeOptions

__all__ = [
    "WorkflowEditionWarning",
    "current_workflow_edition",
    "resolve_workflow_edition",
]


class WorkflowEditionWarning(UserWarning):
    """An authored or served participant declared no workflow edition.

    Filter it by category (``warnings.filterwarnings("ignore",
    category=rlmesh.WorkflowEditionWarning)``) rather than by message.
    """


#: Set by a caller that already resolved this participant's declaration for
#: the servers it stands up itself (the loopback env server in ``run``), so
#: they do not nudge a second time on the caller's behalf.
resolved_by_caller: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "rlmesh_editionresolved_by_caller", default=False
)

#: Process-wide declaration, above every in-code surface but a call keyword.
#: Set it to the empty string to declare nothing *deliberately*: resolution stops
#: there -- no lower surface is consulted -- the peer floats to this build's
#: newest edition, and the undeclared warning stays quiet. That is what this
#: repo's own test suites do.
WORKFLOW_EDITION_ENV_VAR: Final = "RLMESH_WORKFLOW_EDITION"

#: The ``[tool.rlmesh]`` table and key read from the project's ``pyproject.toml``.
_PYPROJECT_TABLE: Final = "[tool.rlmesh]"
_PYPROJECT_KEY: Final = "workflow_edition"

_warned = False


def current_workflow_edition() -> str:
    """The bare ``YYYY.MM`` edition this build runs at -- what to paste into a declaration.

    The same value on every build of that edition (``"2026.06"``): a declaration
    names the contract a participant was authored against, and a bare base
    selects whichever spelling of it both sides offer -- the sealed name on a
    release, this build's cohort (``"2026.06-0.1.0-rc.12"``,
    ``"2026.06-dev.<sha>"``, what ``rlmesh.build_info().workflow_edition``
    reports) on a
    prerelease or local build. That cohort spelling is also accepted as a
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
    authored: bool = False,
) -> str | None:
    """The workflow edition this participant declares, or ``None`` when it declares none.

    Args:
        call: An explicit ``workflow_edition=`` keyword on this call.
        option: A served peer's ``ServeOptions`` / ``--workflow-edition`` value.
        declared: The class-level declaration (``EnvFactory.workflow_edition`` /
            ``Model.workflow_edition``).
        authored: Whether this participant has a home for a declaration -- an
            authored subclass, or a served peer. Only such a participant is
            nudged when it floats; an ad hoc callable in a local run is not.

    Returns:
        The winning declaration, validated and trimmed, or ``None`` when no
        surface declares one -- in which case the peer floats to this build's
        newest edition and, when ``authored``, a one-time
        :class:`WorkflowEditionWarning` says so. An empty
        ``RLMESH_WORKFLOW_EDITION`` is the deliberate form of that: it ends
        resolution without consulting ``option`` / ``declared`` /
        ``pyproject.toml``, and without the warning.

    Raises:
        ValueError: If the winning value names an edition this build cannot run.
    """
    if call is not None and call.strip():
        return validate_workflow_edition(call)
    environment = os.environ.get(WORKFLOW_EDITION_ENV_VAR)
    if environment is not None and not environment.strip():
        # Set-but-empty is the documented "deliberately undeclared" switch: it
        # ends resolution here rather than falling through to the surfaces
        # below it, and suppresses the float warning.
        return None
    for candidate in (environment, option, declared):
        if candidate is not None and candidate.strip():
            return validate_workflow_edition(candidate)
    project = _from_pyproject()
    if project is not None and project.strip():
        return validate_workflow_edition(project)
    if authored and not resolved_by_caller.get():
        _warn_undeclared()
    return None


#: The project-manifest rung, scanned once per working directory: a long-lived
#: process can chdir between sessions, but within one directory the answer is
#: stable and every server or session it opens would otherwise walk the same
#: ancestors again.
_pyproject_cache: dict[Path, str | None] = {}


def _from_pyproject() -> str | None:
    """``[tool.rlmesh] workflow_edition`` from the nearest enclosing ``pyproject.toml``."""
    start = Path.cwd()
    if start not in _pyproject_cache:
        _pyproject_cache[start] = _pyproject_edition(start)
    return _pyproject_cache[start]


def _pyproject_edition(start: Path) -> str | None:
    for directory in (start, *start.parents):
        manifest = directory / "pyproject.toml"
        if not manifest.is_file():
            continue
        try:
            return _scan_table_string(manifest.read_text(encoding="utf-8"))
        except OSError:
            # An unreadable manifest is not this resolver's error to raise: the
            # project is simply not declaring an edition here.
            return None
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


def _caller_stacklevel() -> int:
    """Frames from :func:`warnings.warn` out to the first caller outside rlmesh.

    Every declaration surface reaches the resolver through a different number of
    internal hops, so a fixed ``stacklevel`` points at rlmesh's own files on most
    of them -- which is both unhelpful and wrong for a module-scoped warning
    filter. ``skip_file_prefixes=`` would say this directly, but it is 3.12+ and
    this package supports 3.10.
    """
    package = str(Path(__file__).parent)
    level = 1
    frame = inspect.currentframe()
    frame = frame.f_back if frame is not None else None
    while frame is not None and frame.f_code.co_filename.startswith(package):
        frame = frame.f_back
        level += 1
    return level


def _warn_undeclared() -> None:
    """Say once, per process, which edition an undeclared participant floated to."""
    global _warned
    if _warned:
        return
    _warned = True
    stacklevel = _caller_stacklevel()
    edition = current_workflow_edition()
    warnings.warn(
        f"no workflow edition declared: floating to {edition!r}, which moves when "
        f'rlmesh upgrades. Pin it with workflow_edition = "{edition}" on the class '
        f"or [tool.rlmesh] in pyproject.toml; {WORKFLOW_EDITION_ENV_VAR}='' floats "
        "deliberately.",
        WorkflowEditionWarning,
        stacklevel=stacklevel,
    )


def serve_options_declaring(
    options: ServeOptions | None = None,
    *,
    call: str | None = None,
    option: str | None = None,
    declared: str | None = None,
    authored: bool = True,
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
        call=call, option=option, declared=declared, authored=authored
    )
    if (resolved is None and options is None) or (
        options is not None and resolved == options.workflow_edition
    ):
        return options
    native: type[ServeOptions] = load_native("ServeOptions")
    if options is None:
        return native(workflow_edition=resolved)
    return native(
        allow_remote_shutdown=options.allow_remote_shutdown,
        idle_timeout_seconds=options.idle_timeout_seconds,
        drain_timeout_seconds=options.drain_timeout_seconds,
        close_timeout_seconds=options.close_timeout_seconds,
        workflow_edition=resolved,
    )
