"""The lobby and scene runs: unreleased, fenced behind the ``fleet`` feature.

Only an extension built with ``fleet`` has them; a release wheel does not,
and importing this module there raises :class:`ImportError`. Nothing here is
part of the public API.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Any

from .._rlmesh import V2Robot, V2Scene  # type: ignore[attr-defined]
from . import Robot

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["Scene", "leaving", "run_scene", "wait_in_lobby"]


def wait_in_lobby(
    name: str,
    parts: Mapping[str, Any],
    *,
    home: str | None = None,
    timeout: float | None = None,
    on_handover: Callable[[str, str, str], None] | None = None,
) -> Robot:
    """Wait in the home rlmeshd's lobby until a scene's run assigns this robot.

    ``home`` names the daemon (``"rig-pi.local"``; ``None``: this machine's).
    The robot then joins the run and declares ``parts``; a refused join (its
    pin, another map) is reported to the daemon and the wait goes on. Run
    until :func:`leaving` is set, close, and wait again.
    """
    robot = Robot.__new__(Robot)
    robot._lockstep = None
    robot._native = V2Robot.from_lobby(home, name, json.dumps(parts), timeout)
    robot._adopt(on_handover)
    return robot


def leaving(robot: Robot) -> str | None:
    """Why ``robot`` should leave its scene's run, or ``None`` while it is in it.

    ``scene_ended`` or ``scene_failed`` when the run closes, ``robot_lost``
    when the host lost it (wait in the lobby again: its run re-sends its
    join), ``superseded`` when another process took its name over (stop).
    Always ``None`` outside a scene.
    """
    if robot._lockstep is not None:
        return None
    return robot._native.leaving  # type: ignore[attr-defined, no-any-return]


class Scene:
    """A scene's run on the home rlmeshd, started by :func:`run_scene`.

    Its members join :attr:`address` (this script's own policy among them).
    Use it as a context manager, or ``close()``: either stops the run unless
    it ended. States are dicts: ``run``, ``stage``, ``members`` (each
    ``name``, ``stage``, ``error``) and ``error`` (each ``status`` and
    ``message``).
    """

    def __init__(self, native: V2Scene) -> None:
        self._native = native

    @property
    def run(self) -> str:
        """The run's id (``r-7k2m``), as ``rlmesh stop`` takes it."""
        return self._native.run

    @property
    def address(self) -> str | None:
        """Where the run's members join, once its host is up."""
        return self._native.address()

    def status(self) -> dict[str, Any]:
        """The run's state now."""
        return self._native.status()

    def wait_until(self, stage: str, *, timeout: float = 30.0) -> dict[str, Any]:
        """Block until the run is at ``stage`` (``"running"``); its state then.

        A run that ends first raises :class:`SessionRefused` with its error;
        ``timeout`` raises :class:`TimeoutError`.
        """
        return self._native.wait_until(stage, timeout)

    def arm(self) -> dict[str, Any]:
        """Arm a run that waits for it (``arm="manual"``, or after a lost member rejoined)."""
        return self._native.arm()

    def stop(self, *, timeout: float = 10.0) -> dict[str, Any]:
        """Stop the run and wait for it to end; this process's policies' runs return."""
        return self._native.stop(timeout)

    def close(self) -> None:
        """Stop the run unless it ended, and leave the lobby."""
        self._native.close()

    def __repr__(self) -> str:
        return f"Scene({self.run!r})"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def run_scene(
    path: str | os.PathLike[str],
    *,
    home: str | None = None,
    arm: str = "auto",
    timeout: float = 10.0,
) -> Scene:
    """Run the scene file at ``path`` on its robots' home rlmeshd (``home``; ``None``: this machine's).

    The daemon checks it, claims its robots, assigns them from its lobby and
    arms once every member declared and bound (``arm="manual"``: on
    :meth:`Scene.arm`). A refused scene raises :class:`SessionRefused` with its
    code (``INVALID_REQUEST``, ``NOT_AUTHORIZED``, ``ROBOT_BUSY``).
    """
    if arm not in ("auto", "manual"):
        raise ValueError(f"arm {arm!r}: auto or manual")
    with open(path, encoding="utf-8") as file:
        text = file.read()
    return Scene(V2Scene(home, text, arm == "manual", timeout))
