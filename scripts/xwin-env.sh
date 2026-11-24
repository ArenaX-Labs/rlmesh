
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
xwin_bin="$(cd "$xwin_bin" && pwd)"
ln -sfn "$xwin_rust_bin/llvm-ar" "$xwin_bin/llvm-lib"
ln -sfn "$xwin_rust_bin/llvm-ar" "$xwin_bin/llvm-dlltool"
ln -sfn "$xwin_rust_bin/rust-lld" "$xwin_bin/lld-link"
export PATH="$xwin_bin:$PATH"
export XWIN_ACCEPT_LICENSE=1
