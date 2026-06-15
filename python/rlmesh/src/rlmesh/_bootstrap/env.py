"""Back-compat shim: the bootstrap env API now lives in three focused modules.

``rlmesh._bootstrap.env`` stays a stable import path; the implementation is split
across :mod:`.loaders`, :mod:`.spec_resolution`, and :mod:`.gym_support`.
"""

from __future__ import annotations

from .gym_support import (
    import_gym_modules,
    make_gym_environment,
)
from .loaders import (
    RecipeConstructionError,
    import_packages,
    is_env_lookup_error,
    load_env_entrypoint,
    load_env_from_spec,
    load_environment,
    load_gym_env,
    load_hf_env,
    load_module_from_path,
    load_recipe_env,
    looks_like_env,
    normalize_hf_env,
)
from .spec_resolution import (
    BootstrapUsageError,
    apply_member_params,
    expect_mapping,
    member_params_from_env,
    resolve_bootstrap_spec,
    select_mapping_item,
)

__all__ = [
    "BootstrapUsageError",
    "RecipeConstructionError",
    "apply_member_params",
    "expect_mapping",
    "import_gym_modules",
    "import_packages",
    "is_env_lookup_error",
    "load_env_entrypoint",
    "load_env_from_spec",
    "load_environment",
    "load_gym_env",
    "load_hf_env",
    "load_module_from_path",
    "load_recipe_env",
    "looks_like_env",
    "make_gym_environment",
    "member_params_from_env",
    "normalize_hf_env",
    "resolve_bootstrap_spec",
    "select_mapping_item",
]
