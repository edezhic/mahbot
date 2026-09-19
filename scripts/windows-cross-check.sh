#!/bin/sh
# Local Windows cross-check: type-checks mahbot (lib + bins + the unit-test
# target), runs the workspace lint gate with warnings denied, and compiles the
# lib to a Windows rlib — all for the Windows target. Manual release-gate tool,
# not wired into any CI, hooks or pipeline. Compile-only: nothing is linked into
# a runnable binary and nothing of the result is executed.
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
# The stand-in for the speech dependency: see the comment above the crate below.
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
# shims, the scratch copy of the tree, the dependency stand-in and the scratch
# build dir. The working tree, its manifest and its lock file are never touched.
WORK="$PWD/target/windows-cross-check"
TREE="$WORK/tree"
STANDIN="$WORK/qwen-asr-standin"
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

# ── Speech-dependency stand-in ────────────────────────────────────────────
#
# qwen-asr (local speech recognition) does not compile for Windows: it drives
# mmap/`write_all_at` through libc, which has no Windows implementation. That is
# its own decision and out of scope here, so the check substitutes an
# API-compatible stand-in for it — in the scratch copy of the tree only, so
# nothing that ships and nothing that runs on macOS/Linux is affected.
#
# Fidelity limit: the stand-in mirrors the upstream signatures this crate calls
# (context/config/encoder/audio/transcribe), so our call sites are type-checked
# against the same API, but it reproduces none of the behaviour and its own
# internals are never compiled for Windows either. It is a check-only
# scaffold: it must be deleted once qwen-asr builds for the target itself.

mkdir -p "$STANDIN/src"
cat > "$STANDIN/Cargo.toml" <<'EOF'
[package]
name = "qwen-asr"
version = "0.11.0"
edition = "2021"

# Only the feature names the shipping manifest requests have to exist here:
# cargo resolves the macOS target section's `blas`/`vdsp` even when building for
# another target, and errors if the patch does not declare them. Nothing else is
# requested of this dependency anywhere (`default-features = false` throughout).
[features]
default = []
blas = []
vdsp = []
EOF

cat > "$STANDIN/src/lib.rs" <<'EOF'
//! Check-only stand-in for the real `qwen-asr` crate (see the script header).

pub mod audio {
    pub fn mel_spectrogram(samples: &[f32]) -> Option<(Vec<f32>, usize)> {
        let _ = samples;
        None
    }

    pub fn parse_wav_buffer(data: &[u8]) -> Option<Vec<f32>> {
        let _ = data;
        None
    }

    pub fn resample(samples: &[f32], from_rate: i32, to_rate: i32) -> Vec<f32> {
        let _ = (from_rate, to_rate);
        samples.to_vec()
    }
}

pub mod config {
    pub const SAMPLE_RATE: i32 = 16000;
    pub const HOP_LENGTH: usize = 160;

    pub struct QwenConfig {
        pub enc_output_dim: usize,
    }
}

pub mod encoder {
    use super::config::QwenConfig;

    pub struct EncoderBuffers;

    pub struct Encoder;

    impl Encoder {
        pub fn forward(
            &self,
            cfg: &QwenConfig,
            mel: &[f32],
            mel_frames: usize,
            enc_bufs: Option<&mut EncoderBuffers>,
        ) -> Option<(Vec<f32>, usize)> {
            let _ = (cfg, mel, mel_frames, enc_bufs);
            None
        }
    }
}

pub mod context {
    use super::config::QwenConfig;
    use super::encoder::Encoder;
    use std::sync::Arc;

    pub struct QwenModel {
        pub model_dir: String,
        pub config: QwenConfig,
        pub encoder: Encoder,
    }

    pub struct QwenCtx {
        pub model: Arc<QwenModel>,
        pub want_language_detection: bool,
        pub segment_sec: f32,
    }

    impl QwenCtx {
        pub fn load(model_dir: &str) -> Option<Self> {
            let _ = model_dir;
            None
        }
    }
}

pub mod transcribe {
    use super::context::QwenCtx;

    pub fn transcribe_audio(ctx: &mut QwenCtx, samples: &[f32]) -> Option<String> {
        let _ = (ctx, samples);
        None
    }
}
EOF

# ── Scratch tree ──────────────────────────────────────────────────────────

# tar (not cp) so modification times survive: the scratch build dir is reused
# between runs and only files that actually changed recompile.
rm -rf "$TREE"
mkdir -p "$TREE"
tar -cf - --exclude=./target --exclude=./.git . | (cd "$TREE" && tar -xf -)
cat >> "$TREE/Cargo.toml" <<'EOF'

[patch.crates-io]
qwen-asr = { path = "../qwen-asr-standin" }
EOF

export PATH="$BIN:$PATH"
export CARGO_TARGET_DIR="$WORK/target"
# cc-rs reads these for the target's C toolchain; only the check/clippy lanes
# need them (`cargo zigbuild` configures its own).
export CC_x86_64_pc_windows_gnu="$ZIGCC"
export CXX_x86_64_pc_windows_gnu="$BIN/zigcc.cxx"
export AR_x86_64_pc_windows_gnu="zig ar"

cd "$TREE"

echo "==> type-check (lib + bins + tests) for $TARGET"
cargo check --target "$TARGET" --lib --bins --tests

echo "==> lint (lib + bins + tests), warnings denied, for $TARGET"
cargo clippy --target "$TARGET" --lib --bins --tests -- -D warnings

echo "==> codegen (lib) for $TARGET"
cargo zigbuild --target "$TARGET" --lib
