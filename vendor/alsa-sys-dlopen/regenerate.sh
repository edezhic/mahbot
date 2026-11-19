#!/bin/sh
# Rebuilds `src/generated.rs`: the published `alsa-sys` 0.4.0 bindings, every `extern "C"`
# declaration replaced by a `dlopen` wrapper.
#
# Run by hand, never by the build. The build compiles the committed file as it stands, and that is
# what makes the compiled surface the same on every machine: a build that read `<alsa/asoundlib.h>`
# compiled whatever vintage that header happened to be, and the release image's is older than the
# `alsa` crate expects. Regenerate when `alsa` moves to another `alsa-sys` release, then prove the
# result on both Linux targets with `scripts/linux-cross-check.sh`.
#
#   vendor/alsa-sys-dlopen/regenerate.sh [<path to alsa-sys-0.4.0/src/generated.rs>]
#
# The input is the crate's own pre-generated bindings, not a header, so this needs no ALSA
# development package, no pkg-config and no libclang — only the crate source and `rustc`. Without an
# argument the source is taken from the cargo registry (`cargo fetch` first).
set -eu

cd "$(dirname -- "$0")" || exit 1

# The exact bytes the frozen surface was derived from: `alsa-sys` 0.4.0 as crates.io publishes it. A
# different input is a different surface, so it is refused rather than wrapped silently.
DIGEST=c8c3a3b54577345f81b147a70af6fa3146826032460c1c9ca1e02250177fad4b

input=${1:-}
if [ -z "$input" ]; then
    for candidate in "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/alsa-sys-0.4.0/src/generated.rs; do
        if [ -f "$candidate" ]; then
            input=$candidate
            break
        fi
    done
fi
if [ -z "$input" ] || [ ! -f "$input" ]; then
    echo "no alsa-sys 0.4.0 source found: pass the path to its src/generated.rs as the" >&2
    echo "first argument — taken from a crates.io download of alsa-sys 0.4.0, because this" >&2
    echo "repository's [patch.crates-io] means cargo never fetches the registry crate" >&2
    exit 1
fi

if command -v shasum >/dev/null 2>&1; then
    actual=$(shasum -a 256 "$input" | cut -d' ' -f1)
else
    actual=$(sha256sum "$input" | cut -d' ' -f1)
fi
if [ "$actual" != "$DIGEST" ]; then
    echo "unexpected input: $input" >&2
    echo "  expected sha256 $DIGEST" >&2
    echo "  got      sha256 $actual" >&2
    exit 1
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/alsa-sys-dlopen-regenerate.XXXXXX")
trap 'rm -rf "$work"; rm -f src/.generated.rs.new' 0
rustc --edition 2021 -O -o "$work/transform" regenerate.rs
# Written beside the committed file and renamed over it once the transform has produced a whole
# surface: a redirect straight onto it would leave the crate's bindings empty if the transform
# failed, and a rename inside the directory is atomic where a move across filesystems is not.
"$work/transform" "$input" > src/.generated.rs.new
mv src/.generated.rs.new src/generated.rs
echo "wrote src/generated.rs from $input"
