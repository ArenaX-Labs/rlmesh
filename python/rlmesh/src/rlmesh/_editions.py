"""Where a workflow edition is declared, and which declaration wins."""

from __future__ import annotations

import contextlib
import contextvars
from typing import TYPE_CHECKING

from ._load_native import load_native

if TYPE_CHECKING:
    from collections.abc import Iterator

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
        ``declared``, else :func:`current_workflow_edition`; validated and
        trimmed.

    Raises:
        ValueError: If the winner names an edition this build does not know.
    """
    return str(load_native("resolve_workflow_edition")(declared))


_declared: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "rlmesh_declared_workflow_edition", default=None
)


@contextlib.contextmanager
def declaring(declared: object) -> Iterator[None]:
    """Make ``declared`` the edition an :class:`~rlmesh.EnvServer` built inside declares."""
    token = _declared.set(declared if isinstance(declared, str) else None)
    try:
        yield
    finally:
        _declared.reset(token)


def declared_edition() -> str | None:
    """The edition the innermost :func:`declaring` names, if any."""
    return _declared.get()
