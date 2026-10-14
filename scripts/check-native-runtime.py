"""Assert a native example's describe runtime (source and installed-package gates)."""

import json
import sys

language, version = sys.argv[1:]
runtime = json.load(sys.stdin)["runtime"]
assert runtime["language"] == language, runtime
assert runtime["language_version"] == version, runtime
assert runtime["component"] == "rlmesh-capi", runtime
compiler = runtime["framework_versions"]
assert len(compiler) == 1 and set(compiler) <= {"clang", "gcc", "msvc"}, runtime
assert all(isinstance(value, str) and value for value in compiler.values()), runtime
print(json.dumps(runtime, sort_keys=True))
