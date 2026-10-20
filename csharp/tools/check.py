"""Build and inspect both assemblies without executing library code."""

import argparse
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIB = ROOT / "csharp/src/RLMesh"
TOOL = ROOT / "tools/rlmesh_csharp_surface"


def run(*args):
    """Run one checked command from the repository root."""
    scratch = pathlib.Path.home() / ".cache/rlmesh-csharp/tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        env=dict(os.environ, TMPDIR=str(scratch), MSBUILDDISABLENODEREUSE="1"),
    )


def public_text(surface):
    """Render the metadata snapshot as a readable declared-member inventory."""
    lines = [
        "RLMesh Experimental public API",
        "R-28 v0.2 environment-serving foundation.",
        "Declared members; nullability and custom attributes are recorded in v0.2.json.",
        "",
    ]
    for key, item in surface.items():
        if not key.startswith("T:"):
            continue
        name = key[2:]
        base = item["base"]
        suffix = (
            " : " + base
            if base not in ("", "System.Object", "System.Enum", "System.ValueType")
            else ""
        )
        lines.append(item["kind"] + " " + name + suffix)
        for protocol in item["protocols"]:
            lines.append("  implements " + protocol)
        for member, entry in surface.items():
            if not member[2:].startswith(name + "."):
                continue
            short = member[3 + len(name) :]
            if member.startswith("M:"):
                if short.startswith(("get_", "set_")):
                    continue
                lines.append("  " + short + " : " + entry["signature"])
            elif member.startswith("P:"):
                access = "get; " if entry["get"] else ""
                access += "set; " if entry["set"] else ""
                lines.append(
                    "  " + short + " : " + entry["signature"] + " { " + access + "}"
                )
            elif member.startswith("F:"):
                lines.append("  " + short + " : " + str(entry["value"]))
        lines.append("")
    return "\n".join(lines)


def main():
    """Run the task with the pinned repository tools."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update", action="store_true", help="regenerate the reviewed public inventory"
    )
    args = parser.parse_args()
    for project in (LIB / "RLMesh.csproj", TOOL / "rlmesh_csharp_surface.csproj"):
        run(
            "dotnet",
            "build",
            str(project),
            "--disable-build-servers",
            "-p:UseSharedCompilation=false",
            "--nologo",
        )
    dumps = []
    for target in ("netstandard2.1", "net10.0"):
        output = ROOT / f"target/csharp-surface-{target}.json"
        run(
            "dotnet",
            str(TOOL / "bin/Debug/net10.0/rlmesh_csharp_surface.dll"),
            str(LIB / f"bin/Debug/{target}/RLMesh.dll"),
            str(output),
        )
        dumps.append(json.loads(output.read_text()))
    if dumps[0] != dumps[1]:
        keys = sorted(
            k
            for k in dumps[0].keys() | dumps[1].keys()
            if dumps[0].get(k) != dumps[1].get(k)
        )
        raise SystemExit("C# target surface differs: " + ", ".join(keys))
    snapshot = ROOT / "csharp/surface/v0.2.json"
    text_snapshot = ROOT / "csharp/surface/public-api.txt"
    rendered = public_text(dumps[0])
    if args.update:
        snapshot.write_text(json.dumps(dumps[0], indent=2) + "\n")
        text_snapshot.write_text(rendered)
    if text_snapshot.read_text() != rendered:
        raise SystemExit("C# readable public inventory is stale")
    if not snapshot.exists() or json.loads(snapshot.read_text()) != dumps[0]:
        raise SystemExit(
            "C# surface differs from R-28's v0.2 inventory (csharp/surface/v0.2.json)"
        )
    print(
        f"C# assembly checks: {len(dumps[0])} public items; both targets match the v0.2 inventory"
    )


if __name__ == "__main__":
    main()
