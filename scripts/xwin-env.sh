# Sourced (bash) by every task that cross-builds for x86_64-pc-windows-msvc
# with cargo-xwin: check:windows, release:js:native win32-x64 and
# release:cxx:package --target windows. Run it from the repo root under mise, so
# rustc is the pinned toolchain.
#
# cargo-xwin compiles C with clang-cl, archives with llvm-lib and links with
# lld-link, and runs each from PATH. Only clang has to come from the host
# (cargo-xwin links clang-cl to it). The archiver and linker come from the
# pinned Rust toolchain: rust-lld ships with it, and llvm-ar from its llvm-tools
# component acts as lib.exe when run as llvm-lib. They go first on PATH, so every
# machine archives and links with the same tools whatever LLVM it has, or lacks.
# cargo-xwin would link them only when PATH has none of its own, and only once
# llvm-tools is installed.

xwin_triple=x86_64-pc-windows-msvc
xwin_toolchain="$(mise current rust)"

rustup target list --installed --toolchain "$xwin_toolchain" | grep -qx "$xwin_triple" ||
  rustup target add --toolchain "$xwin_toolchain" "$xwin_triple"
rustup component list --installed --toolchain "$xwin_toolchain" | grep -q '^llvm-tools' ||
  rustup component add --toolchain "$xwin_toolchain" llvm-tools

if ! command -v clang-cl >/dev/null && ! command -v clang >/dev/null; then
  echo "error: cargo-xwin compiles C dependencies with clang-cl: put clang on PATH (e.g. apt install clang)" >&2
  exit 1
fi

xwin_rust_bin="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin"
for tool in llvm-ar rust-lld; do
  if [ ! -x "$xwin_rust_bin/$tool" ]; then
    echo "error: $xwin_rust_bin/$tool is missing from Rust $xwin_toolchain" >&2
    exit 1
  fi
done
xwin_bin="${CARGO_TARGET_DIR:-$PWD/target}/xwin-bin"
mkdir -p "$xwin_bin"
xwin_bin="$(cd "$xwin_bin" && pwd)" # build scripts run from their crate dirs
ln -sfn "$xwin_rust_bin/llvm-ar" "$xwin_bin/llvm-lib"
ln -sfn "$xwin_rust_bin/llvm-ar" "$xwin_bin/llvm-dlltool"
ln -sfn "$xwin_rust_bin/rust-lld" "$xwin_bin/lld-link"
export PATH="$xwin_bin:$PATH"
# cargo-xwin fetches the MSVC CRT and Windows SDK, accepting Microsoft's license
# for them as maturin does for the Windows wheel.
export XWIN_ACCEPT_LICENSE=1
