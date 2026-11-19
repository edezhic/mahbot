#!/bin/sh
# Local Linux cross-check: compiles mahbot for both Linux targets the release ships
# — the lib, the bin, all test targets (which pull in src/tools/computer/linux.rs
# and its Linux-gated tests), and each target's lint gate with warnings denied.
# Manual release-gate tool — not wired into any CI, and not part of a release.
#
# Both targets, not one: aarch64 has its own blind spots, and a surface that x86_64
# accepts can fail there — only a real compile of both targets shows a generated
# surface standing on its own. The audio shim commits its bindings, so this needs no
# ALSA headers, no pkg-config file and no binding toolchain.
#
# Requires: zig, cargo-zigbuild (`brew install cargo-zigbuild`) and both targets'
# rust std (`rustup target add x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu`).
#
# Why not a plain `cargo zigbuild` run: cargo-zigbuild only supports build-like
# subcommands (no `check`), and the check lane is what type-checks the test
# targets. So: `cargo check` with zig as the cross C compiler covers lib + bin +
# test targets (type/borrow-check, no final link), and `cargo zigbuild --lib` adds
# full codegen for the lib target (an rlib needs no final link).
set -eu

# Repo root, regardless of the caller's cwd (script lives in scripts/).
cd "$(dirname -- "$0")/.." || exit 1

command -v zig >/dev/null 2>&1 || { echo "zig not found" >&2; exit 1; }
command -v cargo-zigbuild >/dev/null 2>&1 \
  || { echo "cargo-zigbuild not found: brew install cargo-zigbuild" >&2; exit 1; }

# Everything this script generates lives under target/ (gitignored): the tool
# shims in a scratch dir of their own.
BIN="$PWD/target/linux-cross-check/bin"
mkdir -p "$BIN"

for TARGET in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu; do
  case "$TARGET" in
    x86_64-*) ZIG_TARGET=x86_64-linux-gnu ;;
    aarch64-*) ZIG_TARGET=aarch64-linux-gnu ;;
  esac
  # cc-crate environment variables name the target with underscores.
  TARGET_ENV=$(printf '%s' "$TARGET" | tr '-' '_')

  # Route cc-crate build scripts to zig for the Linux target. cc-rs passes the
  # rust-style `--target=<triple>` (rust's "unknown" OS is not a zig triple — what
  # cargo-zigbuild normally shims), so wrap zig cc to drop it and substitute zig's
  # own spelling.
  ZIGCC="$BIN/zigcc-$ZIG_TARGET"
  cat > "$ZIGCC" <<EOF
#!/bin/sh
n=\$#
i=0
while [ "\$i" -lt "\$n" ]; do
  i=\$((i + 1))
  arg="\$1"; shift
  case "\$arg" in
    --target=$TARGET) ;;
    *) set -- "\$@" "\$arg" ;;
  esac
done
exec zig cc -target $ZIG_TARGET "\$@"
EOF
  chmod +x "$ZIGCC"
  sed 's/zig cc/zig c++/' "$ZIGCC" > "$ZIGCC.cxx"
  chmod +x "$ZIGCC.cxx"
  export "CC_$TARGET_ENV=$ZIGCC"
  export "CXX_$TARGET_ENV=$ZIGCC.cxx"
  export "AR_$TARGET_ENV=zig ar"

  echo "==> type-check (lib + bins + tests) for $TARGET"
  cargo check --target "$TARGET" --lib --bins --tests
  # This target's lint gate, matching the Windows lane: the host lint never sees
  # Linux-gated code.
  echo "==> lint (lib + bins + tests), warnings denied, for $TARGET"
  cargo clippy --target "$TARGET" --lib --bins --tests -- -D warnings
  echo "==> codegen (lib) for $TARGET"
  cargo zigbuild --target "$TARGET" --lib
done
