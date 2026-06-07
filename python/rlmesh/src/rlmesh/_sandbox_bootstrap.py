"""Compatibility entrypoint for containerized RLMesh sandbox environments."""

from __future__ import annotations

from rlmesh._bootstrap.env import (
    import_gym_modules,
    load_gym_env,
    load_hf_env,
    load_module_from_path,
    looks_like_env,
    normalize_hf_env,
    select_mapping_item,
)
from rlmesh._bootstrap.env import (
    load_env_from_spec as load_env,
)
from rlmesh._bootstrap.env import (
    make_gym_environment as make_gym_env,
)
from rlmesh._bootstrap.sandbox_env import main as _sandbox_env_main

__all__ = [
    "import_gym_modules",
    "load_env",
    "load_gym_env",
    "load_hf_env",
    "load_module_from_path",
    "looks_like_env",
    "main",
    "make_gym_env",
    "normalize_hf_env",
    "select_mapping_item",
]


def main(argv: list[str] | None = None) -> int:
    """Run the legacy sandbox bootstrap entrypoint."""
    return _sandbox_env_main(argv, prog="python -m rlmesh._sandbox_bootstrap")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
