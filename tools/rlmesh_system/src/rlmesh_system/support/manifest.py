from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from rlmesh_system.support.env import InstalledEnvironmentSpec

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover - runner is expected on 3.11+
    raise SystemExit("rlmesh-system requires Python 3.11 or newer") from exc


SCENARIO_KINDS = {"trace", "artifact", "external"}
SpecT = TypeVar("SpecT")


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    kind: str
    description: str
    env: dict[str, Any]
    model: str | None
    client: str
    seed: int | None
    steps: int
    trace: str | None
    artifact: str | None
    command_env: str | None
    timeout_seconds: float | None
    metadata: dict[str, str]


@dataclass(frozen=True)
class EnvironmentSpec(InstalledEnvironmentSpec):
    tier: str
    scenarios: tuple[str, ...]
    warmups: int
    samples: int
    processes: int
    timeout_seconds: float | None
    rlmesh: dict[str, Any]


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    description: str
    environments: tuple[str, ...]


@dataclass(frozen=True)
class SystemSpec:
    profiles: dict[str, ProfileSpec]
    environments: dict[str, EnvironmentSpec]
    scenarios: dict[str, ScenarioSpec]


def load_specs(path: Path) -> SystemSpec:
    """Load one profile TOML file or a directory of profile TOML files."""
    files = sorted(path.glob("*.toml")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"no profile files found in {path}")

    profiles: dict[str, ProfileSpec] = {}
    environments: dict[str, EnvironmentSpec] = {}
    scenarios: dict[str, ScenarioSpec] = {}
    for file in files:
        data = tomllib.loads(file.read_text())
        file_scenarios = parse_scenarios(data)
        merge_specs(scenarios, file_scenarios, file)

        file_environments = parse_environments(data)
        merge_specs(environments, file_environments, file)

        profile = parse_profile(data, file, tuple(file_environments))
        if profile.name in profiles:
            raise ValueError(f"duplicate profile {profile.name!r} in {file}")
        profiles[profile.name] = profile

    spec = SystemSpec(
        profiles=profiles,
        environments=environments,
        scenarios=scenarios,
    )
    validate_spec(spec)
    return spec


def parse_profile(
    data: dict[str, object], file: Path, environment_names: tuple[str, ...]
) -> ProfileSpec:
    raw = as_mapping(data.get("profile", {}), context=f"{file}: profile")
    name = str(raw.get("name") or file.stem)
    environments = tuple(str(value) for value in raw.get("environments", ()))
    if not environments:
        environments = environment_names
    return ProfileSpec(
        name=name,
        description=str(raw.get("description", "")),
        environments=environments,
    )


def parse_environments(data: dict[str, object]) -> dict[str, EnvironmentSpec]:
    environments = {}
    for name, value in as_mapping(data.get("environments", {})).items():
        raw = as_mapping(value, context=f"environment {name!r}")
        env = string_mapping(raw.get("env", {}), context=f"environment {name!r} env")
        platform_env = as_mapping(
            raw.get("platform_env", {}),
            context=f"environment {name!r} platform_env",
        )
        for platform_key in platform_env_keys():
            if platform_key in platform_env:
                env.update(
                    string_mapping(
                        platform_env[platform_key],
                        context=f"environment {name!r} platform_env.{platform_key}",
                    )
                )
        environments[str(name)] = EnvironmentSpec(
            name=str(name),
            python=str(raw["python"]),
            dependencies=tuple(str(item) for item in raw.get("dependencies", ())),
            dependency_args=tuple(str(item) for item in raw.get("dependency_args", ())),
            env=env,
            tier=str(raw.get("tier", "basic")),
            scenarios=tuple(str(item) for item in raw.get("scenarios", ())),
            warmups=int(raw.get("warmups", 1)),
            samples=int(raw.get("samples", 5)),
            processes=int(raw.get("processes", 1)),
            timeout_seconds=optional_float(raw.get("timeout_seconds")),
            rlmesh=dict(as_mapping(raw.get("rlmesh", {}))),
        )
    return environments


def parse_scenarios(data: dict[str, object]) -> dict[str, ScenarioSpec]:
    scenarios = {}
    for name, value in as_mapping(data.get("scenarios", {})).items():
        raw = as_mapping(value, context=f"scenario {name!r}")
        metadata = {
            str(key): str(value)
            for key, value in as_mapping(raw.get("metadata", {})).items()
        }
        scenarios[str(name)] = ScenarioSpec(
            name=str(name),
            kind=str(raw["kind"]),
            description=str(raw.get("description", "")),
            env=dict(as_mapping(raw.get("env", {}))),
            model=(str(raw["model"]) if raw.get("model") is not None else None),
            client=str(raw.get("client", "numpy")),
            seed=(int(raw["seed"]) if raw.get("seed") is not None else None),
            steps=int(raw.get("steps", 1)),
            trace=(str(raw["trace"]) if raw.get("trace") is not None else None),
            artifact=(
                str(raw["artifact"]) if raw.get("artifact") is not None else None
            ),
            command_env=(
                str(raw["command_env"]) if raw.get("command_env") is not None else None
            ),
            timeout_seconds=optional_float(raw.get("timeout_seconds")),
            metadata=metadata,
        )
    return scenarios


def string_mapping(value: object, *, context: str) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in as_mapping(value, context=context).items()
    }


def platform_env_keys() -> tuple[str, ...]:
    if sys.platform.startswith("linux"):
        return ("linux", sys.platform) if sys.platform != "linux" else ("linux",)
    if sys.platform == "darwin":
        return ("darwin",)
    if sys.platform.startswith("win"):
        return ("windows", sys.platform)
    return (sys.platform,)


def validate_spec(spec: SystemSpec) -> None:
    for scenario in spec.scenarios.values():
        if scenario.kind not in SCENARIO_KINDS:
            raise ValueError(f"unknown scenario kind {scenario.kind!r}")
        if scenario.kind == "trace":
            if not scenario.env:
                raise ValueError(f"trace scenario {scenario.name!r} needs env")
            validate_trace_env(scenario)
            if scenario.model is None:
                raise ValueError(f"trace scenario {scenario.name!r} needs model")
        if scenario.kind == "artifact" and scenario.artifact is None:
            raise ValueError(f"artifact scenario {scenario.name!r} needs artifact")
        if scenario.kind == "external" and not scenario.command_env:
            raise ValueError(f"external scenario {scenario.name!r} needs command_env")
        if scenario.steps <= 0:
            raise ValueError(f"scenario {scenario.name!r} needs positive steps")

    for profile in spec.profiles.values():
        for environment_name in profile.environments:
            if environment_name not in spec.environments:
                raise ValueError(
                    f"profile {profile.name!r} references {environment_name!r}"
                )

    for environment in spec.environments.values():
        for scenario_name in environment.scenarios:
            if scenario_name not in spec.scenarios:
                raise ValueError(
                    f"environment {environment.name!r} references {scenario_name!r}"
                )
        if environment.warmups < 0:
            raise ValueError(f"environment {environment.name!r} has negative warmups")
        if environment.samples <= 0:
            raise ValueError(f"environment {environment.name!r} needs positive samples")
        if environment.processes <= 0:
            raise ValueError(
                f"environment {environment.name!r} needs positive processes"
            )


def select_environments(
    profile_names: list[str],
    environment_names: list[str],
    spec: SystemSpec,
    *,
    default_profile: str = "basic",
) -> list[EnvironmentSpec]:
    if not profile_names and not environment_names:
        profile_names = [default_profile]

    selected_names: list[str] = []
    for profile_name in profile_names:
        try:
            profile = spec.profiles[profile_name]
        except KeyError as exc:
            raise SystemExit(f"unknown profile {profile_name!r}") from exc
        selected_names.extend(profile.environments)

    selected_names.extend(environment_names)

    selected: list[EnvironmentSpec] = []
    seen: set[str] = set()
    for environment_name in selected_names:
        if environment_name in seen:
            continue
        seen.add(environment_name)
        try:
            selected.append(spec.environments[environment_name])
        except KeyError as exc:
            raise SystemExit(f"unknown environment {environment_name!r}") from exc
    return selected


def filter_scenarios(
    environment: EnvironmentSpec,
    spec: SystemSpec,
    *,
    kinds: set[str] | None,
    scenario_names: set[str] | None = None,
) -> list[ScenarioSpec]:
    scenarios = [spec.scenarios[name] for name in environment.scenarios]
    if kinds:
        scenarios = [scenario for scenario in scenarios if scenario.kind in kinds]
    if scenario_names:
        scenarios = [
            scenario for scenario in scenarios if scenario.name in scenario_names
        ]
    return scenarios


def profile_names_or_default(names: list[str]) -> list[str]:
    return names or ["basic"]


def validate_trace_env(scenario: ScenarioSpec) -> None:
    env = scenario.env
    selectors = [
        key
        for key in ("fixture", "gym", "entrypoint", "id")
        if env.get(key) is not None
    ]
    if not selectors:
        raise ValueError(
            f"trace scenario {scenario.name!r} env needs fixture, gym, entrypoint, or id"
        )

    fixture_or_gym_keys = {"fixture", "gym"}
    if len(fixture_or_gym_keys.intersection(selectors)) > 1:
        raise ValueError(
            f"trace scenario {scenario.name!r} env must choose one fixture or gym"
        )
    if "fixture" in selectors and ({"entrypoint", "id"} & set(selectors)):
        raise ValueError(
            f"trace scenario {scenario.name!r} env fixture cannot be mixed with "
            "entrypoint or id"
        )
    if "gym" in selectors and ({"entrypoint", "id"} & set(selectors)):
        raise ValueError(
            f"trace scenario {scenario.name!r} env gym cannot be mixed with "
            "entrypoint or id"
        )

    mode = str(env.get("mode", "entrypoint"))
    if "entrypoint" in selectors and mode != "entrypoint":
        raise ValueError(
            f"trace scenario {scenario.name!r} env entrypoint requires mode entrypoint"
        )
    if "id" in selectors and mode != "gym":
        raise ValueError(f"trace scenario {scenario.name!r} env id requires mode gym")


def merge_specs(target: dict[str, SpecT], source: dict[str, SpecT], file: Path) -> None:
    for name, value in source.items():
        if name in target and target[name] != value:
            raise ValueError(f"conflicting definition for {name!r} in {file}")
        target[name] = value


def as_mapping(value: object, *, context: str = "value") -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise TypeError(f"{context} must be a TOML table")


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
