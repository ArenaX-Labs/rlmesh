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


def _run_extension_cli(argv: list[str]) -> int:
    import rlmesh._rlmesh as _rlmesh

    # run_cli only exists in builds with the 'viewer' cargo feature; lean
    # wheels omit it (and the GUI stack) entirely.
    run_cli = cast(
        "Callable[[list[str]], int] | None", getattr(_rlmesh, "run_cli", None)
    )
    if run_cli is None:
        raise ImportError(
            "the rlmesh native module was built without the 'viewer' feature"
        )
    return int(run_cli(argv))


def main(argv: list[str] | None = None) -> int:
    """Forward the Python entrypoint directly to the Rust CLI."""
    argv = sys.argv[1:] if argv is None else argv
    repo_root = find_repo_root()
    _ensure_distribution_marker(
        os.environ,
        "python-source" if repo_root is not None else "python-wheel",
    )

    try:
        return _run_extension_cli(argv)
    except ImportError:
        pass

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


if __name__ == "__main__":
    sys.exit(main())
