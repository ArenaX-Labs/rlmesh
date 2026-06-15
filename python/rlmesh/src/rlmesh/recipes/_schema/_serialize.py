"""Recipe schema serialization: canonical JSON to_dict/from_dict + JSON primitives.

These free functions move a recipe between its typed dataclass form (see
:mod:`.definitions`) and the canonical JSON shape consumed by the Rust serde
core (snake_case keys, a ``kind``-tagged ``make`` union).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from ._definitions import (
    ArtifactInput,
    Build,
    Fetch,
    FileWrite,
    GymMake,
    HfMake,
    Make,
    PipInstall,
    ProjectInstall,
    PyMake,
    RecipeKind,
    RecipeValidationError,
    RuntimeReserved,
    Setup,
)

if TYPE_CHECKING:
    from rlmesh.adapters import EnvTags, ModelSpec


def _recipe_kind_from(value: object) -> RecipeKind:
    """Coerce a JSON ``kind`` to the sealed literal; absent/None defaults to ``env``."""
    if value is None:
        return "env"
    if value not in ("env", "model"):
        raise RecipeValidationError(
            f"Recipe.kind must be 'env' or 'model', got {value!r}"
        )
    return value


def _artifact_list(value: object) -> tuple[ArtifactInput, ...]:
    """Coerce a JSON ``inputs`` array (or absent) to a tuple of artifact mounts."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(_artifact_from_dict(item) for item in cast("Sequence[object]", value))


def _adapter_to_dict(
    adapter: EnvTags | ModelSpec | Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Serialize ``Recipe.adapter`` to a BARE JSON dict.

    A raw Mapping passes through verbatim; a dataclass (``EnvTags``/``ModelSpec``)
    serializes via its ``to_dict()`` -- NOT ``to_metadata()`` (that double-nests under
    the wire key; the recipe envelope carries the bare spec).
    """
    if adapter is None:
        return None
    if isinstance(adapter, Mapping):
        return dict(adapter)
    return adapter.to_dict()


def _artifact_to_dict(artifact: ArtifactInput) -> dict[str, object]:
    """Serialize one runtime weight mount; omit None/empty optionals."""
    data: dict[str, object] = {
        "name": artifact.name,
        "target_path": artifact.target_path,
        "required": artifact.required,
    }
    if artifact.uri is not None:
        data["uri"] = artifact.uri
    if artifact.local_dir is not None:
        data["local_dir"] = artifact.local_dir
    if artifact.include:
        data["include"] = list(artifact.include)
    return data


def _artifact_from_dict(value: object) -> ArtifactInput:
    data = _require_map(value, "ArtifactInput")
    return ArtifactInput(
        name=_expect_str(data.get("name"), "ArtifactInput.name"),
        target_path=_expect_str(data.get("target_path"), "ArtifactInput.target_path"),
        uri=_opt_str(data.get("uri"), "ArtifactInput.uri"),
        local_dir=_opt_str(data.get("local_dir"), "ArtifactInput.local_dir"),
        include=_str_list(data.get("include")),
        required=_expect_bool(data.get("required"), "ArtifactInput.required", True),
    )


def _runtime_to_dict(runtime: RuntimeReserved) -> dict[str, object] | None:
    """The reserved-features struct serializes to ``None`` when inert."""
    return runtime.to_dict()


def _make_to_dict(make: Make | None) -> dict[str, object] | None:
    if make is None:
        return None
    if isinstance(make, GymMake):
        return {"kind": "gym", "env_id": make.env_id, "kwargs": dict(make.kwargs)}
    if isinstance(make, PyMake):
        return {
            "kind": "py",
            "entrypoint": make.entrypoint,
            "kwargs": dict(make.kwargs),
        }
    return {
        "kind": "hf",
        "repo": make.repo,
        "revision": make.revision,
        "suite": make.suite,
        "task": make.task,
        "kwargs": dict(make.kwargs),
    }


def _make_from_dict(value: object) -> Make | None:
    if value is None:
        return None
    value = _require_map(value, "make")
    kind = _expect_str(value.get("kind"), "make.kind")
    kwargs = _opt_map(value.get("kwargs")) or {}
    if kind == "gym":
        return GymMake(
            env_id=_expect_str(value.get("env_id"), "make.env_id"), kwargs=kwargs
        )
    if kind == "py":
        return PyMake(
            entrypoint=_expect_str(value.get("entrypoint"), "make.entrypoint"),
            kwargs=kwargs,
        )
    if kind == "hf":
        return HfMake(
            repo=_expect_str(value.get("repo"), "make.repo"),
            revision=_opt_str(value.get("revision"), "make.revision"),
            suite=_opt_str(value.get("suite"), "make.suite"),
            task=_opt_str(value.get("task"), "make.task"),
            kwargs=kwargs,
        )
    raise RecipeValidationError(f"unknown make.kind {kind!r}")


def _pip_to_dict(step: PipInstall) -> dict[str, object]:
    return {
        "packages": list(step.packages),
        "index_url": step.index_url,
        "extra_index_urls": list(step.extra_index_urls),
        "no_deps": step.no_deps,
        "pre": step.pre,
        "requirements": step.requirements,
    }


def _fetch_to_dict(step: Fetch) -> dict[str, object]:
    return {
        "kind": step.kind,
        "repo": step.repo,
        "ref": step.ref,
        "dest": step.dest,
        "pip_install": step.pip_install,
        "pip_requirements": step.pip_requirements,
        "url": step.url,
        "sha256": step.sha256,
    }


def _project_to_dict(project: ProjectInstall | None) -> dict[str, object] | None:
    if project is None:
        return None
    return {
        "src": project.src,
        "dest": project.dest,
        "editable": project.editable,
        "include": list(project.include),
    }


def _build_to_dict(build: Build) -> dict[str, object]:
    return {
        "base": build.base,
        "from_recipe": build.from_recipe,
        "system": list(build.system),
        "system_runtime": list(build.system_runtime),
        "pip": [_pip_to_dict(step) for step in build.pip],
        "project": _project_to_dict(build.project),
        "fetch": [_fetch_to_dict(step) for step in build.fetch],
        "env": dict(build.env),
        "pythonpath": list(build.pythonpath),
        "gpu": build.gpu,
        "installer": build.installer,
        "run_as": build.run_as,
        "toolchain": build.toolchain,
        "commands": list(build.commands),
        "dockerfile": build.dockerfile,
    }


def _setup_to_dict(setup: Setup) -> dict[str, object]:
    return {
        "env": dict(setup.env),
        "files": [
            {"path": fw.path, "contents": fw.contents, "if_absent": fw.if_absent}
            for fw in setup.files
        ],
        "params": list(setup.params),
    }


def _build_from_dict(value: object) -> Build:
    if value is None:
        return Build()
    value = _require_map(value, "build")
    return Build(
        base=_opt_str(value.get("base"), "build.base"),
        from_recipe=_opt_str(value.get("from_recipe"), "build.from_recipe"),
        system=_str_list(value.get("system")),
        system_runtime=_str_list(value.get("system_runtime")),
        pip=tuple(_pip_from_dict(item) for item in _seq(value.get("pip"))),
        project=_project_from_dict(value.get("project")),
        fetch=tuple(_fetch_from_dict(item) for item in _seq(value.get("fetch"))),
        env=_str_str_map(value.get("env")),
        pythonpath=_str_list(value.get("pythonpath")),
        gpu=_expect_bool(value.get("gpu"), "build.gpu", False),
        installer=_opt_str(value.get("installer"), "build.installer") or "pip",
        run_as=_opt_int(value.get("run_as"), "build.run_as"),
        toolchain=_opt_bool(value.get("toolchain"), "build.toolchain"),
        commands=_str_list(value.get("commands")),
        dockerfile=_opt_str(value.get("dockerfile"), "build.dockerfile"),
    )


def _pip_from_dict(value: object) -> PipInstall:
    value = _require_map(value, "build.pip[]")
    return PipInstall(
        packages=_str_list(value.get("packages")),
        index_url=_opt_str(value.get("index_url"), "pip.index_url"),
        extra_index_urls=_str_list(value.get("extra_index_urls")),
        no_deps=_expect_bool(value.get("no_deps"), "pip.no_deps", False),
        pre=_expect_bool(value.get("pre"), "pip.pre", False),
        requirements=_opt_str(value.get("requirements"), "pip.requirements"),
    )


def _fetch_from_dict(value: object) -> Fetch:
    value = _require_map(value, "build.fetch[]")
    return Fetch(
        kind=_expect_str(value.get("kind"), "fetch.kind"),
        repo=_opt_str(value.get("repo"), "fetch.repo"),
        ref=_opt_str(value.get("ref"), "fetch.ref"),
        dest=_opt_str(value.get("dest"), "fetch.dest") or "",
        pip_install=_expect_bool(value.get("pip_install"), "fetch.pip_install", False),
        pip_requirements=_opt_str(
            value.get("pip_requirements"), "fetch.pip_requirements"
        ),
        url=_opt_str(value.get("url"), "fetch.url"),
        sha256=_opt_str(value.get("sha256"), "fetch.sha256"),
    )


def _project_from_dict(value: object) -> ProjectInstall | None:
    if value is None:
        return None
    value = _require_map(value, "build.project")
    return ProjectInstall(
        src=_opt_str(value.get("src"), "project.src") or ".",
        dest=_opt_str(value.get("dest"), "project.dest") or "",
        editable=_expect_bool(value.get("editable"), "project.editable", True),
        include=_str_list(value.get("include")),
    )


def _setup_from_dict(value: object) -> Setup:
    if value is None:
        return Setup()
    value = _require_map(value, "setup")
    files: list[FileWrite] = []
    for raw in _seq(value.get("files")):
        item = _require_map(raw, "setup.files[]")
        files.append(
            FileWrite(
                path=_expect_str(item.get("path"), "setup.files[].path"),
                contents=_expect_str(item.get("contents"), "setup.files[].contents"),
                if_absent=_expect_bool(
                    item.get("if_absent"), "setup.files[].if_absent", False
                ),
            )
        )
    return Setup(
        env=_str_str_map(value.get("env")),
        files=tuple(files),
        params=_str_list(value.get("params")),
    )


def _expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise RecipeValidationError(f"{label} must be a string")
    return value


def _opt_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _expect_str(value, label)


def _expect_int(value: object, label: str, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise RecipeValidationError(f"{label} must be an integer")
    return value


def _opt_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _expect_int(value, label, 0)


def _expect_bool(value: object, label: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise RecipeValidationError(f"{label} must be a boolean")
    return value


def _opt_bool(value: object, label: str) -> bool | None:
    if value is None:
        return None
    return _expect_bool(value, label, False)


def _require_map(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecipeValidationError(f"{label} must be a JSON object")
    # JSON object keys are always strings once parsed by json.loads.
    return cast("Mapping[str, object]", value)


def _seq(value: object) -> Sequence[object]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RecipeValidationError("expected a JSON array")
    return cast("Sequence[object]", value)


def _str_list(value: object) -> list[str]:
    return [_expect_str(item, "array item") for item in _seq(value)]


def _str_str_map(value: object) -> dict[str, str]:
    if value is None:
        return {}
    return {
        key: _expect_str(item, "map value")
        for key, item in _require_map(value, "object of strings").items()
    }


def _opt_map(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    return dict(_require_map(value, "object"))


def _get(payload: Mapping[str, object], outer: str, inner: str) -> object:
    section = payload.get(outer)
    if section is None:
        return None
    return _require_map(section, outer).get(inner)
