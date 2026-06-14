"""Migration scaffolder: generate a recipe (and a factory stub) from a project.

Migration-first tooling, orthogonal to the schema -- it ships regardless. The
scaffolder transcribes the load-bearing 80% of a port (the ``build.pip`` steps
from a project's dependencies and ``[tool.uv]`` indices, the base/gpu guess, the
``PyMake`` entrypoint) and leaves explicit ``TODO`` markers where it cannot infer
(the apt runtime list, Isaac's ``SimulationApp`` pre-init order). The core
operates on already-parsed data so it is fully testable without a TOML parser;
``scaffold_from_pyproject`` is the convenience wrapper that parses the text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

__all__ = [
    "ScaffoldResult",
    "scaffold_from_pyproject",
    "scaffold_recipe",
]

_GPU_MARKERS = ("isaacsim", "isaaclab", "cuda", "torch")
_DEFAULT_GPU_BASE = "nvidia/cuda:12.4.1-runtime-ubuntu22.04"
_PKG_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


@dataclass(frozen=True)
class ScaffoldResult:
    """The generated artifacts for a recipe migration."""

    recipe_source: str
    factory_source: str
    todos: tuple[str, ...] = field(default_factory=tuple)


def _canonical_pkg_name(dependency: str) -> str:
    """Extract the distribution name from a PEP 508 dependency string."""
    match = _PKG_NAME.match(dependency.strip())
    if match is None:
        return dependency.strip()
    return match.group(1).lower().replace("_", "-")


def _looks_gpu(dependencies: Sequence[str]) -> bool:
    blob = " ".join(dependencies).lower()
    return any(marker in blob for marker in _GPU_MARKERS)


def _index_urls(uv_index: Sequence[Mapping[str, object]]) -> dict[str, str]:
    """Map ``[tool.uv.index]`` names to their URLs."""
    urls: dict[str, str] = {}
    for entry in uv_index:
        name = entry.get("name")
        url = entry.get("url")
        if isinstance(name, str) and isinstance(url, str):
            urls[name] = url
    return urls


def _source_index(uv_sources: Mapping[str, object]) -> dict[str, str]:
    """Map a package name to the index name it is pinned to in ``[tool.uv.sources]``."""
    pinned: dict[str, str] = {}
    for package, spec in uv_sources.items():
        spec_map = _as_map(spec)
        if spec_map is not None:
            index = spec_map.get("index")
            if isinstance(index, str):
                pinned[_canonical_pkg_name(package)] = index
    return pinned


def scaffold_recipe(
    name: str,
    entrypoint: str,
    *,
    dependencies: Sequence[str] = (),
    uv_sources: Mapping[str, object] | None = None,
    uv_index: Sequence[Mapping[str, object]] | None = None,
    detect_assets: bool = False,
    gpu: bool | None = None,
) -> ScaffoldResult:
    """Generate a recipe (and factory stub) from project metadata.

    Args:
        name: The recipe name (``namespace/name``).
        entrypoint: The ``module:callable`` factory entrypoint.
        dependencies: The project's PEP 508 dependency strings.
        uv_sources: The ``[tool.uv.sources]`` table (package -> {index: name}).
        uv_index: The ``[tool.uv.index]`` array (each {name, url, explicit}).
        detect_assets: Whether ``__file__``-relative asset access was detected
            (emits a ``ProjectInstall`` carrying ``assets/**``).
        gpu: Force the gpu flag; ``None`` guesses it from the dependency markers.

    Returns:
        The generated recipe source, a factory stub, and the unresolved TODOs.
    """
    uv_sources = uv_sources or {}
    uv_index = uv_index or []
    index_urls = _index_urls(uv_index)
    pinned = _source_index(uv_sources)

    is_gpu = _looks_gpu(dependencies) if gpu is None else gpu
    base = _DEFAULT_GPU_BASE if is_gpu else None

    # Each package pinned to an explicit index gets its own PipInstall; the rest
    # collapse into one PyPI step (the faithful transcription of per-package
    # [tool.uv.sources]).
    indexed: list[tuple[str, str]] = []
    plain: list[str] = []
    todos: list[str] = []
    for dependency in dependencies:
        package = _canonical_pkg_name(dependency)
        index_name = pinned.get(package)
        if index_name is not None:
            url = index_urls.get(index_name)
            if url is None:
                todos.append(
                    f"package {dependency!r} is pinned to unknown index {index_name!r}"
                )
                plain.append(dependency)
            else:
                indexed.append((dependency, url))
        else:
            plain.append(dependency)

    pip_steps: list[str] = []
    for dependency, url in indexed:
        pip_steps.append(
            f"            PipInstall(packages=[{dependency!r}], index_url={url!r}),"
        )
    if plain:
        joined = ", ".join(repr(dep) for dep in plain)
        pip_steps.append(f"            PipInstall(packages=[{joined}]),")

    todos.append(
        "fill in build.system_runtime (apt runtime libs) -- not inferable from pyproject"
    )
    if is_gpu:
        todos.append(
            "if this is an Isaac env, the factory must create SimulationApp before "
            "importing isaaclab.* (see the factory stub)"
        )

    recipe_source = _render_recipe_source(
        name=name,
        entrypoint=entrypoint,
        base=base,
        gpu=is_gpu,
        pip_steps=pip_steps,
        detect_assets=detect_assets,
        todos=tuple(todos),
    )
    factory_source = _render_factory_stub(entrypoint, is_gpu)
    return ScaffoldResult(
        recipe_source=recipe_source,
        factory_source=factory_source,
        todos=tuple(todos),
    )


def _render_recipe_source(
    *,
    name: str,
    entrypoint: str,
    base: str | None,
    gpu: bool,
    pip_steps: Sequence[str],
    detect_assets: bool,
    todos: Sequence[str],
) -> str:
    imports = ["PipInstall", "PyMake", "Recipe", "register"]
    if detect_assets:
        imports.insert(0, "ProjectInstall")
    imports.insert(0, "Build")
    import_line = f"from rlmesh.recipes import {', '.join(sorted(set(imports)))}"

    pip_block = "\n".join(pip_steps) if pip_steps else ""
    project_line = (
        '        project=ProjectInstall(src=".", include=("assets/**",)),\n'
        if detect_assets
        else ""
    )
    todo_block = "".join(f"# TODO: {todo}\n" for todo in todos)

    return (
        f'"""Generated recipe for {name}.\n\n'
        f"Scaffolded from project metadata; review the TODOs below before shipping.\n"
        f'"""\n\n'
        f"from __future__ import annotations\n\n"
        f"{import_line}\n\n"
        f"{todo_block}"
        f"RECIPE = Recipe(\n"
        f"    name={name!r},\n"
        f"    make=PyMake(entrypoint={entrypoint!r}),\n"
        f"    build=Build(\n"
        f"        base={base!r},\n"
        f"        gpu={gpu!r},\n"
        f"        pip=[\n"
        f"{pip_block}\n"
        f"        ],\n"
        f"{project_line}"
        f"    ),\n"
        f")\n\n"
        f"register(RECIPE)\n"
    )


def _render_factory_stub(entrypoint: str, gpu: bool) -> str:
    module, _, callable_name = entrypoint.partition(":")
    isaac_note = (
        "    # TODO (Isaac): create the SimulationApp BEFORE importing isaaclab.*\n"
        "    #   from isaacsim import SimulationApp\n"
        "    #   _app = SimulationApp({'headless': True})\n"
        "    #   import isaaclab.envs  # only now safe to import\n"
        if gpu
        else ""
    )
    return (
        f'"""Generated factory stub for {module!r}."""\n\n'
        f"from __future__ import annotations\n\n\n"
        f"def {callable_name or 'make'}(**kwargs: object) -> object:\n"
        f'    """Construct and return the environment (reset/step/close)."""\n'
        f"{isaac_note}"
        f"    raise NotImplementedError('port the construction logic here')\n"
    )


def scaffold_from_pyproject(
    name: str,
    entrypoint: str,
    pyproject_text: str,
    *,
    detect_assets: bool = False,
    gpu: bool | None = None,
) -> ScaffoldResult:
    """Parse a ``pyproject.toml`` and scaffold a recipe from it.

    Raises:
        RuntimeError: If no TOML parser is available (Python < 3.11 without
            ``tomli`` installed).
    """
    data = _parse_toml(pyproject_text)
    project = _as_map(data.get("project"))
    dependencies: list[str] = []
    if project is not None:
        deps = project.get("dependencies")
        if isinstance(deps, Sequence) and not isinstance(deps, (str, bytes)):
            dependencies = [str(dep) for dep in cast("Sequence[object]", deps)]
    uv = _as_map((_as_map(data.get("tool")) or {}).get("uv"))
    uv_sources: Mapping[str, object] = {}
    uv_index: list[Mapping[str, object]] = []
    if uv is not None:
        uv_sources = _as_map(uv.get("sources")) or {}
        index = uv.get("index")
        if isinstance(index, Sequence) and not isinstance(index, (str, bytes)):
            uv_index = [
                entry
                for entry in (_as_map(item) for item in cast("Sequence[object]", index))
                if entry is not None
            ]
    return scaffold_recipe(
        name,
        entrypoint,
        dependencies=dependencies,
        uv_sources=uv_sources,
        uv_index=uv_index,
        detect_assets=detect_assets,
        gpu=gpu,
    )


def _as_map(value: object) -> Mapping[str, object] | None:
    """Narrow a parsed-TOML value to a string-keyed mapping, or None."""
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return None


def _parse_toml(text: str) -> Mapping[str, object]:
    # Imported dynamically: ``tomllib`` is stdlib only on 3.11+, and the shipped
    # package floor is 3.10, so a static import would not resolve there.
    import importlib

    for module_name in ("tomllib", "tomli"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        return cast("Mapping[str, object]", module.loads(text))
    raise RuntimeError(  # pragma: no cover - environment dependent
        "scaffold_from_pyproject needs a TOML parser; use Python 3.11+ or install "
        "'tomli'. Alternatively call scaffold_recipe with parsed data."
    )
