"""Generate the loader version from the Cargo workspace during every .NET build."""

import argparse
import json
from pathlib import Path

import tomllib


def main():
    """Write only when changed, including for direct dotnet builds after a bump."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    version = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["package"][
        "version"
    ]
    content = (
        "// Generated from Cargo.toml; do not edit.\n"
        "namespace RLMesh.Interop;\n"
        "internal static unsafe partial class Initializers\n{\n"
        f"    internal const string PackageVersion = {json.dumps(version)};\n"
        "}\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != content:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
