"""Sandbox environment bootstrap entrypoint."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, cast

from .._client._endpoint import redact_address
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
        framework = os.environ.get("RLMESH_FRAMEWORK") or None
        device = os.environ.get("RLMESH_DEVICE") or None
        # RLMESH_DEVICE only applies to torch/jax; on a numpy/gym env EnvServer rejects it.
        # Gym vectorization concatenates with numpy and breaks framework tensors, so serve those scalar.
        if framework not in ("torch", "jax"):
            device = None
        elif int(cast("str | int", spec.get("num_envs", 1) or 1)) > 1:
            raise ValueError(
                f"a {framework} env cannot be served with RLMESH_NUM_ENVS>1: gym "
                "vectorization concatenates observations with numpy, which discards "
                "the framework tensors. Serve scalar (RLMESH_NUM_ENVS=1)."
            )
        address = os.environ.get("RLMESH_ADDRESS") or "0.0.0.0:7447"
        name = os.environ.get("RLMESH_NAME") or "env"
        server = EnvServer(
            env, f"{address.rstrip('/')}/{name}", framework=framework, device=device
        )
        # The bound address carries the session key; don't print it.
        print(f"RLMesh sandbox serving {redact_address(server.address)}", flush=True)
        server.serve()
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
