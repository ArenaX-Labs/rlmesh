#!/usr/bin/env bash
# Load and link the extracted distribution, including its final code signature.
set -euo pipefail
[[ "$(uname -s)" == Darwin ]] || { echo 'macOS package smoke requires Darwin' >&2; exit 1; }
scratch_root="${RUNNER_TEMP:-$HOME/.cache/rlmesh-csharp/tmp}"
mkdir -p "$scratch_root"
scratch="$(mktemp -d "$scratch_root/rlmesh-macos-smoke.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT
version="$(python -c 'import tomllib; print(tomllib.load(open("Cargo.toml", "rb"))["workspace"]["package"]["version"])')"
name="rlmesh-$version-universal-apple-darwin"
tar -xzf "target/capi-pkg/$name.tar.gz" -C "$scratch"
package="$scratch/$name"
python - "$package/lib/librlmesh_capi.dylib" <<'PY'
import ctypes
import pathlib
import re
import sys
library = ctypes.CDLL(sys.argv[1])
library.rlmesh_abi_version.restype = ctypes.c_uint32
header = pathlib.Path("crates/rlmesh-capi/include/rlmesh.h").read_text()
expected = int(re.search(r"^#define RLMESH_ABI_VERSION (\d+)$", header, re.M)[1])
assert library.rlmesh_abi_version() == expected
PY
cat > "$scratch/smoke.c" <<'C'
#include <rlmesh.h>
int main(void) { return rlmesh_abi_version() == RLMESH_ABI_VERSION ? 0 : 1; }
C
export PKG_CONFIG_PATH="$package/lib/pkgconfig"
# pkg-config deliberately emits shell-separated compiler arguments.
# shellcheck disable=SC2046
cc "$scratch/smoke.c" $(pkg-config --cflags --libs rlmesh) -Wl,-rpath,"$package/lib" -o "$scratch/smoke"
"$scratch/smoke"
