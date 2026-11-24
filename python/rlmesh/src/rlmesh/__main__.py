#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, MutableMapping
from typing import cast

from ._cli.main import find_repo_root

_DISTRIBUTION_ENV = "RLMESH_CLI_DISTRIBUTION"
#: Interpreter `rlmesh check` / `rlmesh describe` run `rlmesh._describe` in.
_PYTHON_ENV = "RLMESH_PYTHON"


def _load_extension_cli() -> Callable[[list[str]], int]:
    """Import and return the native ``run_cli``, or raise ``ImportError``."""
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
    os.environ.setdefault(_PYTHON_ENV, sys.executable)

    try:
        run_cli = _load_extension_cli()
    except ImportError:
        run_cli = None
    if run_cli is not None:
        try:
            return int(run_cli(argv))
        except KeyboardInterrupt:
            # A command that stops on Ctrl-C has returned; Python's own handler raises afterwards.
            return 130

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


def daemon_main(argv: list[str] | None = None) -> int:
    """The rlmeshd entrypoint: ``rlmesh host`` under its own name."""
    argv = sys.argv[1:] if argv is None else argv
    return main(["host", *argv])


def credential_helper_main(argv: list[str] | None = None) -> int:
    """Docker credential-helper entrypoint (installed as docker-credential-rlmesh)."""
    argv = sys.argv[1:] if argv is None else argv
    return main(["registry", "credential-helper", *argv])


if __name__ == "__main__":
    sys.exit(main())
