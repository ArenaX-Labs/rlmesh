"""Check a native example's describe runtime (source and installed-package gates)."""

import json
import re
import sys


def _check(condition, runtime):
    if not condition:
        sys.exit(f"unexpected native runtime: {runtime!r}")


language, version = sys.argv[1:]
runtime = json.load(sys.stdin)["runtime"]
_check(runtime.get("language") == language, runtime)
if language == "c++":
    _check("language_version" not in runtime, runtime)
else:
    _check(runtime.get("language_version") == version, runtime)
_check(runtime.get("component") == "rlmesh-capi", runtime)
_check("framework_versions" not in runtime, runtime)
toolchain = runtime.get("toolchain")
_check(isinstance(toolchain, dict) and set(toolchain) == {"name", "version"}, runtime)
_check(toolchain["name"] in {"clang", "gcc", "msvc"}, runtime)
_check(isinstance(toolchain["version"], str), runtime)
_check(re.fullmatch(r"[0-9]+\.[0-9]+", toolchain["version"]), runtime)
print(json.dumps(runtime, sort_keys=True))
