"""Sphinx configuration for the RLMesh documentation."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = ROOT / "python" / "rlmesh" / "src"
REPOSITORY_URL = "https://github.com/ArenaX-Labs/rlmesh"

sys.path.insert(0, str(PYTHON_SRC))


def _native_type(name: str, bases: tuple[type[object], ...] = (object,)) -> type:
    return type(
        name,
        bases,
        {
            "__doc__": "Native RLMesh type. See the API reference for details.",
            "__module__": "rlmesh._rlmesh",
        },
    )


def _native_unavailable(*_args: object, **_kwargs: object) -> Any:
    raise RuntimeError("rlmesh._rlmesh is not available during documentation builds")


def _install_native_docs_stub() -> None:
    native = types.ModuleType("rlmesh._rlmesh")
    native.__doc__ = "Documentation stub for the RLMesh native extension."
    native.__file__ = str(PYTHON_SRC / "rlmesh" / "_rlmesh.pyi")
    native.__version__ = "0.0.0"

    for name in (
        "EnvironmentException",
        "ProtocolException",
        "RLMeshException",
    ):
        setattr(native, name, _native_type(name, (RuntimeError,)))

    for name in (
        "EnvContract",
        "PyEnvClient",
        "PyEnvServer",
        "PyModel",
        "PyVectorEnvClient",
        "ServeOptions",
        "Space",
        "SpaceSpec",
        "Tensor",
    ):
        setattr(native, name, _native_type(name))

    for name in (
        "box_space_spec",
        "dict_space_spec",
        "discrete_space_spec",
        "multi_binary_space_spec",
        "multi_discrete_space_spec",
        "run_cli",
        "sandbox_start_env",
        "sandbox_stop_env",
        "space_spec_from_gym_space",
        "text_space_spec",
        "tuple_space_spec",
    ):
        setattr(native, name, _native_unavailable)

    native.PrimitiveValue = object
    native.Value = object
    native.ResetInfo = dict[str, object]
    native.StepInfo = dict[str, object]
    native.RenderBundle = dict[str, object]
    native.SandboxRunInfo = dict[str, object]
    native.__all__ = sorted(name for name in vars(native) if not name.startswith("_"))
    sys.modules["rlmesh._rlmesh"] = native


def _source_ref() -> str:
    for env_name in ("VERCEL_GIT_COMMIT_SHA", "GITHUB_SHA"):
        value = os.environ.get(env_name)
        if value:
            return value

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "main"
    return result.stdout.strip() or "main"


SOURCE_REF = _source_ref()

_install_native_docs_stub()

project = "RLMesh"
author = "ArenaX Labs"
copyright = "2026 ArenaX Labs, Inc."

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "local-dev.md",
    "release.md",
    "testing.md",
]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

autoclass_content = "both"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

extlinks = {
    "source": (f"{REPOSITORY_URL}/blob/{SOURCE_REF}/%s", "%s"),
}

html_theme = "furo"
html_title = "RLMesh"
html_show_sphinx = False
templates_path = ["_templates"]
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": f"{REPOSITORY_URL}/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#31545f",
        "color-brand-content": "#31545f",
        "color-api-name": "#243a43",
        "color-background-primary": "#fbfbf8",
        "color-background-secondary": "#f1f3ee",
        "color-background-hover": "#e7ece7",
        "color-sidebar-background": "#f4f5f1",
        "color-sidebar-background-border": "#d9ded7",
        "color-sidebar-caption-text": "#68736d",
        "color-sidebar-link-text--top-level": "#1f2a2d",
        "color-highlight-on-target": "#eef2dc",
        "color-inline-code-background": "#eef1ec",
    },
    "dark_css_variables": {
        "color-brand-primary": "#9eb8bd",
        "color-brand-content": "#9eb8bd",
        "color-api-name": "#c0d5d8",
        "color-background-primary": "#111414",
        "color-background-secondary": "#191d1d",
        "color-sidebar-background": "#151919",
        "color-sidebar-background-border": "#2a3030",
        "color-sidebar-caption-text": "#a7b0ad",
        "color-sidebar-link-text--top-level": "#ecefed",
        "color-highlight-on-target": "#283024",
        "color-inline-code-background": "#222827",
    },
}


def _resolve_object(module_name: str, fullname: str) -> object | None:
    module = sys.modules.get(module_name)
    if module is None:
        return None

    obj: object = module
    for part in fullname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None

    if isinstance(obj, property):
        obj = obj.fget
    return inspect.unwrap(obj) if obj is not None else None


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    if domain != "py":
        return None

    module_name = info.get("module")
    fullname = info.get("fullname")
    if not module_name or not fullname:
        return None

    obj = _resolve_object(module_name, fullname)
    if obj is None:
        return None

    try:
        source_file = inspect.getsourcefile(obj)
        source_lines, line_number = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return None

    if source_file is None:
        return None

    try:
        relative_path = Path(source_file).resolve().relative_to(ROOT)
    except ValueError:
        return None

    end_line = line_number + len(source_lines) - 1
    path = relative_path.as_posix()
    return f"{REPOSITORY_URL}/blob/{SOURCE_REF}/{path}#L{line_number}-L{end_line}"
