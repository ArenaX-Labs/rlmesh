"""Sandbox environment bootstrap entrypoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import cast

from .env import expect_mapping, load_env_from_spec


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "python -m rlmesh._bootstrap.sandbox_env",
) -> int:
    """Serve a sandbox environment from a bootstrap payload.

    The payload is supplied either inline via the ``RLMESH_BOOTSTRAP_JSON``
    environment variable (the sandbox runner delivers runtime-only parameters
    this way so they never need to be baked into the image) or, for backward
    compatibility, as a path to a JSON file passed as the sole argument.
    """
    argv = sys.argv[1:] if argv is None else argv

    inline = os.environ.get("RLMESH_BOOTSTRAP_JSON")
    if inline is not None:
        if argv:
            print(
                f"usage: {prog} (set RLMESH_BOOTSTRAP_JSON, no arguments)",
                file=sys.stderr,
            )
            return 2
        raw = inline
    else:
        if len(argv) != 1:
            print(
                f"usage: {prog} <bootstrap.json> (or set RLMESH_BOOTSTRAP_JSON)",
                file=sys.stderr,
            )
            return 2
        raw = Path(argv[0]).read_text(encoding="utf-8")

    payload_data = cast(object, json.loads(raw))
    payload = expect_mapping(payload_data, "bootstrap payload")
    spec = expect_mapping(payload.get("spec"), "bootstrap spec")

    try:
        from rlmesh import EnvServer
        from rlmesh.server import EnvLike as ServedEnv

        env = cast(ServedEnv, load_env_from_spec(spec))
        # Canonical bind contract: RLMESH_ENV_ADDRESS (a full bind address) takes
        # precedence; RLMESH_ENV_PORT remains a port-only fallback on 0.0.0.0.
        address = os.environ.get("RLMESH_ENV_ADDRESS")
        if address:
            server = EnvServer(env, address)
        else:
            port = int(os.environ.get("RLMESH_ENV_PORT", "50051"))
            server = EnvServer(env, host="0.0.0.0", port=port)
        print(f"RLMesh sandbox serving {server.address}", flush=True)
        server.serve()
        return 0
    except Exception as exc:  # pragma: no cover - exercised through container runs
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
