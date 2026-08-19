#!/usr/bin/env python3
"""Thin Python entrypoint that forwards to the Rust RLMesh CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, MutableMapping
from typing import cast

from ._cli.main import find_repo_root

_DISTRIBUTION_ENV = "RLMESH_CLI_DISTRIBUTION"


def _load_extension_cli() -> Callable[[list[str]], int]:
    """Import and return the native ``run_cli``, or raise ``ImportError``.

    Import/lookup only -- the returned callable runs outside any ImportError
    handler, so an ImportError raised *by* the CLI itself propagates instead of
    being misreported as a missing native module. ``run_cli`` only exists in
    builds with the 'cli' cargo feature; lean wheels omit it (and the embedded
    CLI) entirely.
    """
    import rlmesh._rlmesh as _rlmesh

    run_cli = cast(
        "Callable[[list[str]], int] | None", getattr(_rlmesh, "run_cli", None)
    )
    if run_cli is None:
        raise ImportError(
            "the rlmesh native module was built without the 'cli' feature"
        )
    return run_cli


def main(argv: list[str] | None = None) -> int:
    """Forward the Python entrypoint directly to the Rust CLI."""
    argv = sys.argv[1:] if argv is None else argv
    repo_root = find_repo_root()
    _ensure_distribution_marker(
        os.environ,
        "python-source" if repo_root is not None else "python-wheel",
    )

    try:
        run_cli = _load_extension_cli()
    except ImportError:
        run_cli = None
    if run_cli is not None:
        return int(run_cli(argv))

    cargo = shutil.which("cargo")
    if repo_root is not None and cargo is not None:
        return subprocess.run(
            [cargo, "run", "-p", "rlmesh-cli", "--bin", "rlmesh", "--", *argv],
            check=False,
            cwd=repo_root,
        ).returncode

    print(
        (
            "Error: RLMesh native module could not be imported. "
            "Install the package wheel or build the workspace extension."
        ),
        file=sys.stderr,
    )
    return 1


def _ensure_distribution_marker(
    environ: MutableMapping[str, str],
    distribution: str,
) -> None:
    environ.setdefault(_DISTRIBUTION_ENV, distribution)


def credential_helper_main(argv: list[str] | None = None) -> int:
    """Docker credential-helper entrypoint (installed as docker-credential-rlmesh)."""
    argv = sys.argv[1:] if argv is None else argv
    return main(["registry", "credential-helper", *argv])


if __name__ == "__main__":
    sys.exit(main())
