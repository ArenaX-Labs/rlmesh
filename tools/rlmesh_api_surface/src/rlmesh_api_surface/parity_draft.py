"""Produce a *draft* mapping for human review, never used by parity:check.

Inputs are the committed Python snapshot, area parity tables, and extracted
native surfaces (``python -m rlmesh_api_surface.parity --dump ...``). Unresolved
mappings are reported; this command deliberately cannot overwrite the reviewed
manifest. Re-run only to compare a draft, then review the relevant source.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rlmesh_api_surface.parity import LANGUAGES, native_name, python_entries, snake


def table_hints(directory: Path) -> dict[str, list[tuple[str, list[str]]]]:
    hints: dict[str, list[tuple[str, list[str]]]] = {}
    for file in sorted(directory.glob("*.md")):
        if file.stem == "rules":
            continue
        for line in file.read_text().splitlines():
            if not line.startswith("|"):
                continue
            cells = re.split(r"(?<!\\)\|", line)[1:-1]
            if len(cells) != 4:
                continue
            tokens = [re.findall(r"`([^`]+)`", c) for c in cells]
            for token in tokens[0]:
                hints.setdefault(token, []).append(
                    (file.stem, [" ".join(t) for t in tokens[1:]])
                )
    return hints


def generate(root: Path, surfaces: dict) -> dict:
    snapshot = json.loads(
        (
            root / "python/rlmesh/tests/api_surface/snapshots/api_surface.json"
        ).read_text()
    )
    expected = python_entries(snapshot)
    hints = table_hints(root / "docs/specs/language-parity")
    result = {}
    for path, obj in expected.items():
        module, _, symbol = path.partition(":")
        owner, _, member = symbol.partition(".")
        area = (
            "adapters"
            if module.endswith("adapters")
            else "spaces"
            if module.endswith("spaces") or owner == "Tensor"
            else "remote"
            if owner.startswith("Remote")
            else "env"
            if owner in ("EnvFactory", "EnvServer", "ServeOptions", "Variant")
            else "running"
            if owner
            in (
                "run",
                "session",
                "Session",
                "RunResult",
                "RunHooks",
                "RunContext",
                "StepEvent",
                "EpisodeResult",
                "TelemetryRow",
                "Reader",
                "FrameSources",
                "View",
                "Recorder",
                "EpisodeRecord",
                "MediaRef",
                "WorkloadRecord",
                "RANDOM_SAMPLE",
            )
            or module.endswith("recorder")
            else "model"
            if owner
            in ("Model", "NO_ADAPTER", "SpecArg", "PredictContext", "predict_seed")
            else "core"
        )
        row = {
            "area": area,
            "rule": ["R-4"],
            "parameters": {
                p: {"default": d} if d is not None else {}
                for p, d in obj["params"].items()
            },
        }
        bridge = (
            module in ("rlmesh.types", "rlmesh.numpy", "rlmesh.jax", "rlmesh.torch")
            or "gym" in symbol
        )
        exempt = owner == "__build__"
        if bridge:
            row["rule"] = ["R-24"]
        if exempt:
            row["rule"] = ["D-8"]
        prefix = "rlmesh::" + (
            module.removeprefix("rlmesh.").replace(".", "::") + "::"
            if module != "rlmesh"
            else ""
        )
        if module == "rlmesh.recorder":
            prefix = "rlmesh::recorder::"
        cname = "RLMesh" + (
            "AdaptersText" if module == "rlmesh.adapters" and owner == "Text" else owner
        )
        for lang in LANGUAGES:
            if bridge or exempt:
                row[lang] = {"status": "bridge" if bridge else "exempt"}
                continue
            candidates = []
            if lang == "c":
                if member:
                    candidates = [
                        cname + "::" + native_name(member, lang),
                        "rlmesh_" + snake(owner) + "_" + member,
                    ]
                    if owner in ("Box", "Discrete", "Dict"):
                        candidates += ["rlmesh_space_" + member]
                    if (owner == "Model" and member.startswith(("predict", "on_"))) or (
                        owner == "Model" and member in ("device", "load")
                    ):
                        candidates += ["RLMeshModelVtable::" + member]
                    if owner == "EnvFactory":
                        candidates += [
                            "RLMeshEnvFactoryAttributes::" + member,
                            "RLMeshEnvFactoryVtable::" + member,
                        ]
                else:
                    candidates = [
                        cname,
                        "rlmesh_"
                        + (
                            "adapters_"
                            if module.endswith("adapters")
                            else "spaces_"
                            if module.endswith("spaces")
                            else "recorder_"
                            if module.endswith("recorder")
                            else ""
                        )
                        + owner,
                        "rlmesh_" + owner,
                        "RLMESH_"
                        + (
                            "ADAPTERS_"
                            if module.endswith("adapters")
                            else "RECORDER_"
                            if module.endswith("recorder")
                            else ""
                        )
                        + owner,
                    ]
                    if owner == "session":
                        candidates.insert(0, "rlmesh_session_new")
                    if owner == "__version__":
                        candidates = ["RLMESH_VERSION"]
                if owner.endswith("Exception"):
                    errors = {
                        "RLMeshException": "INTERNAL",
                        "ProtocolException": "PROTOCOL",
                        "EnvironmentException": "ENVIRONMENT",
                        "RecoverableEnvironmentException": "ENVIRONMENT",
                        "ModelException": "MODEL",
                    }
                    candidates += (
                        ["RLMeshErrorClass"]
                        if not member
                        else ["rlmesh_last_error_" + member, "rlmesh_last_error_code"]
                    )
                    if owner in errors and not member:
                        candidates = ["RLMeshErrorClass"]
            else:
                native_owner = prefix + owner
                if lang == "cpp" and module == "rlmesh.recorder" and owner != "SCHEMA":
                    native_owner = "rlmesh::" + owner
                if owner == "SpaceSpec":
                    native_owner = (
                        "rlmesh::spaces::SpaceSpec"
                        if lang == "rust"
                        else "rlmesh::specs::SpaceSpec"
                    )
                if owner == "EnvContract":
                    native_owner = "rlmesh::specs::EnvContract"
                candidates = (
                    [native_owner + "::" + native_name(member, lang)]
                    if member
                    else [native_owner]
                )
                if member and owner in ("Box", "Discrete", "Dict"):
                    candidates += ["rlmesh::spaces::Space::" + member]
                if member and owner == "RecoverableEnvironmentException":
                    candidates += ["rlmesh::EnvironmentException::" + member]
                if member and owner == "Model":
                    candidates += ["rlmesh::Policy::" + member]
                if member and owner == "EnvFactory":
                    candidates += [
                        "rlmesh::EnvFactoryExt::" + member,
                        "rlmesh::EnvFactoryBase::" + member,
                    ]
                if owner == "__version__":
                    candidates = ["rlmesh::VERSION"]
                if owner.endswith("Exception") and lang == "rust":
                    variant = {
                        "RLMeshException": "Internal",
                        "ProtocolException": "Protocol",
                        "EnvironmentException": "Environment",
                        "RecoverableEnvironmentException": "Environment",
                        "ModelException": "Model",
                    }.get(owner, "Internal")
                    candidates = (
                        ["rlmesh::Error::" + variant]
                        if not member
                        else ["rlmesh::Error::" + member]
                    )
                if owner == "WorkflowEditionWarning":
                    candidates = ["rlmesh::WarningCategory::WorkflowEditionWarning"]
            # Exact native paths mentioned in the matching area table are a
            # second source of candidates; no fuzzy match is silently accepted.
            for key in (symbol, owner, "." + member if member else owner):
                for _, texts in hints.get(key, []):
                    text = texts[LANGUAGES.index(lang)]
                    for token in re.findall(
                        r"(?:rlmesh::)?\b\w+(?:::\w+)+|rlmesh_\w+|RLMesh\w+", text
                    ):
                        candidates += [token, "rlmesh::" + token]
            candidates += [
                p
                for p in surfaces.get(lang, {})
                if any(
                    p.endswith("::" + candidate.removeprefix("rlmesh::"))
                    for candidate in candidates
                    if candidate.startswith("rlmesh::") and candidate.count("::") <= 2
                )
            ]
            target = next(
                (p for p in candidates if p in surfaces.get(lang, {})),
                "REVIEW:" + (candidates[0] if candidates else symbol),
            )
            mapping = {"status": "present", "path": target, "params": {}}
            if obj["params"]:
                types = [target, target.rsplit("::", 1)[0]]
                # Options are discoverable on the callable's native parameter.
                functions = [target]
                if lang == "c" and owner == "session":
                    functions += ["rlmesh_session_new"]
                if obj["kind"] == "class":
                    functions += [
                        target + "::new",
                        target + "::" + owner,
                        "rlmesh_" + snake(owner) + "_new",
                    ]
                for function in functions:
                    for ty in (
                        surfaces.get(lang, {})
                        .get(function, {})
                        .get("params", {})
                        .values()
                    ):
                        for token in re.findall(r"(?:\w+::)*\w+Options", ty):
                            types += [
                                token,
                                "rlmesh::" + token,
                                prefix + token,
                                target.rsplit("::", 1)[0] + "::" + token,
                            ]
                # Common options ownership follows R-6.
                options = (
                    owner
                    + ("" if not member else member.title().replace("_", ""))
                    + "Options"
                )
                types += [
                    prefix + options,
                    "RLMesh" + options,
                    target.rsplit("::", 1)[0] + "::" + options,
                ]
                if owner in ("run", "session"):
                    types += [
                        "rlmesh::" + owner.title() + "Options",
                        "RLMesh" + owner.title() + "Options",
                    ]
                for p in obj["params"]:
                    name = native_name(p, lang)
                    mapped = next(
                        (
                            t + "#" + name
                            for t in [*functions, *types]
                            if name
                            in surfaces.get(lang, {}).get(t, {}).get("params", {})
                            or surfaces.get(lang, {})
                            .get(t + "::" + name, {})
                            .get("kind")
                            == "field"
                        ),
                        None,
                    )
                    if mapped:
                        mapping["params"][p] = mapped
                    elif (
                        p in ("bridge", "input_bridge", "framework", "device")
                        and owner == "EnvServer"
                    ) or p == "bridge":
                        mapping.setdefault("bridges", []).append(p)
                        row["rule"].append("R-24")
                    else:
                        mapping["params"][p] = "REVIEW:#" + name
            row[lang] = mapping
        result[path] = row
    return result


def write_toml(data: dict) -> str:
    lines = [
        "# Reviewed language parity mapping. Edit intentionally; parity:check never regenerates it."
    ]

    def quote(value):
        return json.dumps(value, ensure_ascii=False)

    def value(v):
        if isinstance(v, dict):
            return (
                "{ "
                + ", ".join(quote(k) + " = " + value(x) for k, x in v.items())
                + " }"
            )
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        return quote(v)

    for path, row in data.items():
        lines += ["", "[" + quote(path) + "]"]
        for key, val in row.items():
            if key in LANGUAGES:
                continue
            lines.append(key + " = " + value(val))
        for lang in LANGUAGES:
            if lang not in row:
                continue
            lines += ["", "[" + quote(path) + "." + lang + "]"]
            for key, val in row[lang].items():
                lines.append(key + " = " + value(val))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surfaces", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", default=Path.cwd(), type=Path)
    args = parser.parse_args()
    if (
        args.output.resolve()
        == (args.root / "docs/specs/language-parity/manifest.toml").resolve()
    ):
        parser.error("write a separate draft, then review it against the code")
    draft = generate(args.root, json.loads(args.surfaces.read_text()))
    args.output.write_text(write_toml(draft))
    print(
        f"Wrote {len(draft)} draft entries; resolve every REVIEW marker before adoption"
    )


if __name__ == "__main__":
    main()
