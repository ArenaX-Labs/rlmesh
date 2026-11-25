"""Where a workflow edition is declared, and which declaration wins."""

from __future__ import annotations

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
    """The category of rlmesh's workflow-edition warnings.

    Filter it by category (``warnings.filterwarnings("ignore",
    category=rlmesh.WorkflowEditionWarning)``) rather than by message.
    """


#: Process-wide declaration, above every in-source one. Blank declares nothing.
WORKFLOW_EDITION_ENV_VAR: Final = "RLMESH_WORKFLOW_EDITION"


def current_workflow_edition() -> str:
    """The bare ``YYYY.MM`` edition this build runs at when nothing declares one.

    The value to paste into a declaration (``"2026.10"``), the same on a release,
    prerelease and source build.
    """
    return str(load_native("current_workflow_edition")())


def validate_workflow_edition(edition: str) -> str:
    """Return ``edition`` trimmed, or raise :class:`ValueError` for one this build does not know."""
    return str(load_native("validate_workflow_edition")(edition))


def resolve_workflow_edition(declared: str | None = None) -> str:
    """The workflow edition a participant runs at.

    Args:
        declared: The in-source declaration (``EnvFactory.workflow_edition`` /
            ``Model.workflow_edition``).

    Returns:
        ``RLMESH_WORKFLOW_EDITION`` when set, else ``declared``, else
        :func:`current_workflow_edition`; validated and trimmed.

    Raises:
        ValueError: If the winner names an edition this build does not know.
    """
    return str(load_native("resolve_workflow_edition")(declared))


def serve_options_declaring(
    options: ServeOptions | None = None, *, declared: str | None = None
) -> ServeOptions:
    """Return ``options`` carrying the edition the served participant runs at.

    ``RLMESH_WORKFLOW_EDITION``, else ``options``' own ``workflow_edition``, else
    ``declared``, else the current edition.
    """
    own = options.workflow_edition if options is not None else None
    resolved = resolve_workflow_edition(own or declared)
    if options is not None and resolved == options.workflow_edition:
        return options
    native: type[ServeOptions] = load_native("ServeOptions")
    if options is None:
        return native(workflow_edition=resolved)
    return native(
        close_timeout_seconds=options.close_timeout_seconds,
        workflow_edition=resolved,
    )
