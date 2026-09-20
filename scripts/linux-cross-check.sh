#!/bin/sh
# Local Linux cross-check: compiles mahbot for x86_64-unknown-linux-gnu — the
# lib, all test targets (which pull in src/tools/computer/linux.rs and its
# Linux-gated tests) and the benches under the `voice-tests` marker. Manual
# release-gate tool — not wired into any CI.
#
# Requires: zig, cargo-zigbuild (`brew install cargo-zigbuild`) and the target's
# rust std (`rustup target add x86_64-unknown-linux-gnu`).
#
# Why not a plain `cargo zigbuild` run: cargo-zigbuild only supports build-like
# subcommands (no `check`), and the check lane is what type-checks the test
# targets. So: `cargo check` with zig as the cross C compiler covers lib + test
# targets (type/borrow-check, no final link), and `cargo zigbuild --lib` adds
# full codegen for the lib target (an rlib needs no final link).
set -eu

# Repo root, regardless of the caller's cwd (script lives in scripts/).
cd "$(dirname -- "$0")/.." || exit 1

TARGET=x86_64-unknown-linux-gnu
command -v zig >/dev/null 2>&1 || { echo "zig not found" >&2; exit 1; }
command -v cargo-zigbuild >/dev/null 2>&1 \
  || { echo "cargo-zigbuild not found: brew install cargo-zigbuild" >&2; exit 1; }

# Everything this script generates lives under target/ (gitignored): the tool
# shims in a scratch dir of their own.
BIN="$PWD/target/linux-cross-check/bin"
mkdir -p "$BIN"

# Route cc-crate build scripts to zig for the Linux target. cc-rs passes the
# rust-style `--target=x86_64-unknown-linux-gnu` (rust "unknown" OS is not a
# zig triple — what cargo-zigbuild normally shims), so wrap zig cc to drop it
# and substitute zig's own spelling.
ZIGCC="$BIN/zigcc"
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
sed 's/zig cc/zig c++/' "$ZIGCC" > "$BIN/zigcc.cxx"
chmod +x "$BIN/zigcc.cxx"
export CC_x86_64_unknown_linux_gnu="$ZIGCC"
export CXX_x86_64_unknown_linux_gnu="$BIN/zigcc.cxx"
export AR_x86_64_unknown_linux_gnu="zig ar"

cargo check --target "$TARGET" --lib --tests
# Audio benches and the `voice-tests` dev marker: the feature cannot be
# platform-gated, so this lane proves the marker enables nothing here and the
# wake-word bench compiles to its inert stub instead of reaching for the
# macOS-only audio subsystem.
cargo check --target "$TARGET" --benches --features voice-tests
cargo zigbuild --target "$TARGET" --lib
