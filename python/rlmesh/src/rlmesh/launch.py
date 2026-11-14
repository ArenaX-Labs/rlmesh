"""Start a session host in another process and dial it.

A process that serves an env or a model (``python -m rlmesh.serve`` in a
container) hosts its session beyond loopback only under keys its launcher
passed it: the launcher makes a host identity, passes it to the process in
``RLMESH_HOST_IDENTITY``, and dials the keyed address the identity's key
forms::

    import os, subprocess
    from rlmesh import launch

    identity, _key = launch.launch_identity()
    address = launch.launched_address(identity, "127.0.0.1:7600", "env")
    subprocess.Popen(
        [
            "docker",
            "run",
            "-e",
            "RLMESH_HOST_IDENTITY",
            "-e",
            "RLMESH_ADDRESS=0.0.0.0:7600",
            "-p",
            "127.0.0.1:7600-7603:7600-7603",
            "my-env:latest",
        ],
        env={**os.environ, "RLMESH_HOST_IDENTITY": identity},
    )
    # rlmesh.RemoteEnv(address) once it serves.

The identity is a secret: pass it through the environment, never argv, and
keep it out of logs. The keyed address is a credential too: its holder joins
the session, from this machine or any other that reaches the host's ports.
``rlmesh host-identity`` does the same from a shell.
"""

from __future__ import annotations

from ._rlmesh import identity_key as _identity_key
from ._rlmesh import launch_identity as _launch_identity
from ._rlmesh import launched_address as _launched_address

__all__ = ["HOST_IDENTITY_VAR", "identity_key", "launch_identity", "launched_address"]

#: The variable a launched process reads its host identity from.
HOST_IDENTITY_VAR = "RLMESH_HOST_IDENTITY"


def launch_identity() -> tuple[str, str]:
    """Fresh host keys: ``(identity, key)``.

    ``identity`` is the ``RLMESH_HOST_IDENTITY`` value to start the host with
    (a secret); ``key`` is the address key its participants join by.
    """
    return _launch_identity()


def identity_key(identity: str) -> str:
    """The address key of the host identity ``identity``.

    The key :func:`launch_identity` returned with it, for a launcher that keeps
    only the identity. Raises ``ValueError`` on text that is not an identity,
    without repeating it.
    """
    return _identity_key(identity)


def launched_address(
    identity: str, session: str, participant: str | None = None
) -> str:
    """The keyed address of a session on a host run under ``identity``.

    ``session`` is where participants reach the host, ``host:port`` (its base
    port, the default session) or ``host:port/session``; with ``participant``,
    that participant's address in it, as ``RemoteEnv`` / ``Model`` take it.
    """
    return _launched_address(identity, session, participant)
