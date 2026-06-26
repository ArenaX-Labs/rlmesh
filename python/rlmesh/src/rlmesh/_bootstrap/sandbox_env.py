"""Sandbox environment bootstrap entrypoint."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, cast

from .loaders import load_env_from_spec
from .spec_resolution import BootstrapUsageError, resolve_bootstrap_spec

if TYPE_CHECKING:
    from rlmesh._server import EnvLike, VectorServerEnvLike


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "python -m rlmesh._bootstrap.sandbox_env",
) -> int:
    """Serve a sandbox environment from a bootstrap payload.

    The bootstrap spec is resolved from inline ``RLMESH_BOOTSTRAP_JSON`` (or a
    ``bootstrap.json`` path argument). ``RLMESH_NUM_ENVS``/
    ``RLMESH_VECTORIZATION_MODE`` set the eval shape at run time.
    """
    argv = sys.argv[1:] if argv is None else argv

    try:
        spec = dict(resolve_bootstrap_spec(argv, prog=prog))
    except BootstrapUsageError as exc:
        print(exc, file=sys.stderr)
        return 2

    # Flat eval knobs override whatever the spec carried (gym/hf both read
    # num_envs/vectorization_mode from the spec).
    num_envs = os.environ.get("RLMESH_NUM_ENVS")
    if num_envs:
        try:
            spec["num_envs"] = int(num_envs)
        except ValueError:
            print(
                f"RLMESH_NUM_ENVS must be an integer, got {num_envs!r}",
                file=sys.stderr,
            )
            return 2
    vectorization_mode = os.environ.get("RLMESH_VECTORIZATION_MODE")
    if vectorization_mode:
        spec["vectorization_mode"] = vectorization_mode

    try:
        from rlmesh import EnvServer

        raw_env = load_env_from_spec(spec)
        env = cast("EnvLike[Any, Any] | VectorServerEnvLike", raw_env)
        # Canonical bind contract: RLMESH_ADDRESS (a full bind address) wins, then
        # RLMESH_PORT (default 50051); RLMESH_ENV_ADDRESS/RLMESH_ENV_PORT remain
        # deprecated aliases read after the new names. EnvServer auto-detects the
        # vectorized shape, so one construction path serves scalar and vector envs.
        address = os.environ.get("RLMESH_ADDRESS") or os.environ.get(
            "RLMESH_ENV_ADDRESS"
        )
        if address:
            server = EnvServer(env, address)
        else:
            port = int(
                os.environ.get("RLMESH_PORT")
                or os.environ.get("RLMESH_ENV_PORT")
                or "50051"
            )
            server = EnvServer(env, host="0.0.0.0", port=port)
        print(f"RLMesh sandbox serving {server.address}", flush=True)
        server.serve()
        return 0
    except Exception as exc:  # pragma: no cover - exercised through container runs
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
