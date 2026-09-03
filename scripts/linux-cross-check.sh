#!/bin/sh
# Local Linux cross-check: compiles mahbot (lib + all test targets, which pulls
# in src/tools/computer/linux.rs and its Linux-gated tests) for
# x86_64-unknown-linux-gnu. Manual release-gate tool — not wired into any CI.
#
# Requires: zig, cargo-zigbuild (`brew install cargo-zigbuild`) and the target's
# rust std (`rustup target add x86_64-unknown-linux-gnu`).
#
# Why not a plain `cargo zigbuild` run: cargo-zigbuild only supports build-like
# subcommands (no `check`), and a full build links test/binary targets — which
# would need real Linux shared libs (alsa, …) that don't exist on this machine.
# So: `cargo check` with zig as the cross C compiler covers lib + test targets
# (type/borrow-check, no final link), and `cargo zigbuild --lib` adds full
# codegen for the lib target (an rlib needs no final link).
set -eu

# Repo root, regardless of the caller's cwd (script lives in scripts/).
cd "$(dirname -- "$0")/.." || exit 1

TARGET=x86_64-unknown-linux-gnu
command -v zig >/dev/null 2>&1 || { echo "zig not found" >&2; exit 1; }
command -v cargo-zigbuild >/dev/null 2>&1 \
  || { echo "cargo-zigbuild not found: brew install cargo-zigbuild" >&2; exit 1; }

# alsa-sys (via cpal) probes alsa through pkg-config but ships pregenerated
# bindings and compiles no C, so a stub .pc satisfies the probe. Kept under
# target/ so it stays gitignored and regenerates on every run.
STUB_DIR="$PWD/target/cross-pkgconfig/alsa-stub"
mkdir -p "$STUB_DIR/include" "$STUB_DIR/lib"
cat > "$STUB_DIR/alsa.pc" <<EOF
prefix=$STUB_DIR
Name: alsa
Description: stub for Linux cross-check (alsa-sys bindings are pregenerated)
Version: 1.2.0
Cflags: -I$STUB_DIR/include
Libs: -L$STUB_DIR/lib -lasound
EOF
export PKG_CONFIG_ALLOW_CROSS=1
export PKG_CONFIG_PATH="$STUB_DIR"

# Route cc-crate build scripts to zig for the Linux target. cc-rs passes the
# rust-style `--target=x86_64-unknown-linux-gnu` (rust "unknown" OS is not a
# zig triple — what cargo-zigbuild normally shims), so wrap zig cc to drop it
# and substitute zig's own spelling.
ZIGCC="$PWD/target/cross-pkgconfig/zigcc"
cat > "$ZIGCC" <<'EOF'
#!/bin/sh
n=$#
i=0
while [ "$i" -lt "$n" ]; do
  i=$((i + 1))
  arg="$1"; shift
  case "$arg" in
    --target=x86_64-unknown-linux-gnu) ;;
    *) set -- "$@" "$arg" ;;
  esac
done
exec zig cc -target x86_64-linux-gnu "$@"
EOF
chmod +x "$ZIGCC"
sed 's/zig cc/zig c++/' "$ZIGCC" > "$ZIGCC.cxx"
chmod +x "$ZIGCC.cxx"
export CC_x86_64_unknown_linux_gnu="$ZIGCC"
export CXX_x86_64_unknown_linux_gnu="$ZIGCC.cxx"
export AR_x86_64_unknown_linux_gnu="zig ar"

cargo check --target "$TARGET" --lib --tests
cargo zigbuild --target "$TARGET" --lib
