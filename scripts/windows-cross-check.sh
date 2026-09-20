#!/bin/sh
# Local Windows cross-check: type-checks mahbot (lib + bins + the unit-test
# target + the benches under the `voice-tests` marker), runs the workspace lint
# gate with warnings denied, and compiles the lib to a Windows rlib — all for
# the Windows target. Manual release-gate tool, not wired into any CI, hooks or
# pipeline. Compile-only: nothing is linked into a runnable binary and nothing
# of the result is executed.
#
# Its purpose is to keep the Windows-only breakage from creeping back: the
# project is regularly tested on macOS/Linux only, so without this check the
# `cfg(windows)` branches of our own source are never compiled by anything.
#
# Target: x86_64-pc-windows-gnu. The project's platform gating is by operating
# system (no code consults `target_env`), so the GNU and MSVC ABIs reach the
# same branches; the GNU one is chosen because zig ships a mingw-w64 toolchain
# and can therefore serve as the cross C compiler.
#
# Requires: zig, cargo-zigbuild (`brew install cargo-zigbuild`) and the target's
# rust std (`rustup target add x86_64-pc-windows-gnu`).
#
# Why the shims below: `cargo check` still runs the dependency build scripts,
# which compile C for the target (onig_sys, the tree-sitter grammars, zstd-sys,
# ring, turso_sdk_kit). Zig's C compiler is routed in via the cc-crate env vars
# the way scripts/linux-cross-check.sh does it, and `windres` (which
# turso_sdk_kit's build script calls for its DLL version resource) is adapted to
# `zig rc`. cargo-zigbuild supplies the equivalent wrappers for `cargo
# zigbuild`, but it has no `check` subcommand, so the check lane needs its own.
#
# No dependency stand-in is needed: the local audio subsystem (and the speech
# engine behind it) is macOS-only, so nothing audio-shaped is part of a Windows
# build in the first place.
set -eu

# Repo root, regardless of the caller's cwd (script lives in scripts/).
cd "$(dirname -- "$0")/.." || exit 1

TARGET=x86_64-pc-windows-gnu
command -v zig >/dev/null 2>&1 || { echo "zig not found" >&2; exit 1; }
command -v cargo-zigbuild >/dev/null 2>&1 \
  || { echo "cargo-zigbuild not found: brew install cargo-zigbuild" >&2; exit 1; }

# `rustc --print target-libdir` succeeds even for a target whose std is not
# installed, so the directory it reports is what has to exist.
[ -d "$(rustc --print target-libdir --target "$TARGET")" ] \
  || { echo "target std missing: rustup target add $TARGET" >&2; exit 1; }

# Everything this script generates lives under target/ (gitignored): the tool
# shims and the scratch build dir. The working tree, its manifest and its lock
# file are never touched.
WORK="$PWD/target/windows-cross-check"
BIN="$WORK/bin"
mkdir -p "$BIN"

# ── C-toolchain shims ─────────────────────────────────────────────────────

# cc-rs passes the rust-style `--target=x86_64-pc-windows-gnu` (rust's "unknown"
# vendor is not a zig triple), so strip it and substitute zig's own spelling.
ZIGCC="$BIN/zigcc"
cat > "$ZIGCC" <<'EOF'
#!/bin/sh
n=$#
i=0
while [ "$i" -lt "$n" ]; do
  i=$((i + 1))
  arg="$1"; shift
  case "$arg" in
    --target=x86_64-pc-windows-gnu) ;;
    *) set -- "$@" "$arg" ;;
  esac
done
exec zig cc -target x86_64-windows-gnu "$@"
EOF
chmod +x "$ZIGCC"
sed 's/zig cc/zig c++/' "$ZIGCC" > "$BIN/zigcc.cxx"
chmod +x "$BIN/zigcc.cxx"

# turso_sdk_kit's build script compiles a VERSIONINFO resource for its cdylib by
# shelling out to `windres` (`<in> -O coff -o <out>`). zig ships an `rc.exe`
# drop-in, so translate the windres spelling and let zig do the work.
cat > "$BIN/windres" <<'EOF'
#!/bin/sh
in=""; out=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    -i) in="$2"; shift 2 ;;
    -O) shift 2 ;;
    -*) shift ;;
    *) in="$1"; shift ;;
  esac
done
[ -n "$in" ] && [ -n "$out" ] || { echo "windres shim: missing input or output" >&2; exit 1; }
exec zig rc /nologo /fo "$out" "$in"
EOF
chmod +x "$BIN/windres"

export PATH="$BIN:$PATH"
# A scratch build dir, so the host's target/ is not invalidated by the cross
# build and a rerun recompiles only what changed.
export CARGO_TARGET_DIR="$WORK/target"
# cc-rs reads these for the target's C toolchain; only the check/clippy lanes
# need them (`cargo zigbuild` configures its own).
export CC_x86_64_pc_windows_gnu="$ZIGCC"
export CXX_x86_64_pc_windows_gnu="$BIN/zigcc.cxx"
export AR_x86_64_pc_windows_gnu="zig ar"

# The real tree, not a copy: nothing here patches the manifest (the audio
# subsystem and its dependencies are not part of a Windows build at all).
# Audio benches and the `voice-tests` dev marker: the feature cannot be
# platform-gated, so these lanes prove the marker enables nothing here and the
# wake-word bench compiles to its inert stub instead of reaching for the
# macOS-only audio subsystem.
echo "==> type-check (lib + bins + tests + audio benches) for $TARGET"
cargo check --target "$TARGET" --lib --bins --tests
cargo check --target "$TARGET" --benches --features voice-tests

echo "==> lint (lib + bins + tests + audio benches), warnings denied, for $TARGET"
cargo clippy --target "$TARGET" --lib --bins --tests -- -D warnings
cargo clippy --target "$TARGET" --benches --features voice-tests -- -D warnings

echo "==> codegen (lib) for $TARGET"
cargo zigbuild --target "$TARGET" --lib
