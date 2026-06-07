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
    """Serve a sandbox environment from a bootstrap JSON file."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(
            f"usage: {prog} <bootstrap.json>",
            file=sys.stderr,
        )
        return 2

    payload_data = cast(object, json.loads(Path(argv[0]).read_text(encoding="utf-8")))
    payload = expect_mapping(payload_data, "bootstrap payload")
    spec = expect_mapping(payload.get("spec"), "bootstrap spec")

    try:
        from rlmesh import EnvServer
        from rlmesh.server import EnvLike as ServedEnv

        env = cast(ServedEnv, load_env_from_spec(spec))
        port = int(os.environ.get("RLMESH_ENV_PORT", "50051"))
        server = EnvServer(env, host="0.0.0.0", port=port)
        print(f"RLMesh sandbox serving {server.address()}", flush=True)
        server.serve()
        return 0
    except Exception as exc:  # pragma: no cover - exercised through container runs
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
