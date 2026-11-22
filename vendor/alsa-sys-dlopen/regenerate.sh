#!/bin/sh
# Rebuilds `src/generated.rs`: the part of the published `alsa-sys` 0.4.0 bindings that `alsa` 0.11
# reaches, with every `extern "C"` declaration replaced by a `dlopen` wrapper.
#
# Run by hand, never by the build. The build compiles the committed file as it stands, and that is
# what makes the compiled surface the same on every machine: a build that read `<alsa/asoundlib.h>`
# compiled whatever vintage that header happened to be, and the release image's is older than the
# `alsa` crate expects. Regenerate when either input moves — the `alsa` crate to another version, or
# that crate to another `alsa-sys` release — because both decide the surface: one is what gets
# wrapped, the other what is kept. A move also refreshes that crate's pin below, and the output
# pin in `src/audio/mod.rs` moves with the result. Then prove it on both Linux targets with
# `scripts/linux-cross-check.sh` — a host that is not Linux never compiles this crate at all, so a
# green local `cargo check` says nothing about it.
#
#   vendor/alsa-sys-dlopen/regenerate.sh [<path to alsa-sys-0.4.0/src/generated.rs> [<path to
#                                         alsa-0.11.0/src>]]
#
# The inputs are the two crates' own sources, not a header, so this needs no ALSA development
# package, no pkg-config and no libclang — only the crate sources and `rustc`. Without an argument
# each is taken from the cargo registry (`cargo fetch` first).
#
# The surface is narrowed to what `alsa` reaches, and nothing else: `cpal` 0.18's off-by-default
# `realtime` feature names `snd_pcm_type` and the `SND_PCM_TYPE_*` constants, which that narrowing
# drops — enabling the feature means regenerating from a wider keep-set, not flipping a flag.
set -eu
LC_ALL=C
export LC_ALL

cd "$(dirname -- "$0")" || exit 1

# The exact bytes each input is expected to be. `alsa-sys` 0.4.0 is the surface as crates.io
# publishes it; `alsa` 0.11.0 is the layer whose references decide what is kept, hashed over its
# `src/*.rs` concatenated in name order (`LC_ALL=C cat alsa-0.11.0/src/*.rs | shasum -a 256`, which
# is what the glob below does under the C locale this script sets). A different input is a different
# surface, so it is refused rather than wrapped silently.
DIGEST=c8c3a3b54577345f81b147a70af6fa3146826032460c1c9ca1e02250177fad4b
LAYER_DIGEST=b970cf4258f0638e5f6b41224c0e513a7cdfe68dc8e088c14323d912b51d8ed8

digest_of() {
    if command -v shasum >/dev/null 2>&1; then
        cat "$@" | shasum -a 256 | cut -d' ' -f1
    else
        cat "$@" | sha256sum | cut -d' ' -f1
    fi
}

from_registry() {
    for candidate in "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/"$1"; do
        if [ -e "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

bindings=${1:-}
if [ -z "$bindings" ]; then
    bindings=$(from_registry alsa-sys-0.4.0/src/generated.rs) || bindings=
fi
if [ -z "$bindings" ] || [ ! -f "$bindings" ]; then
    echo "no alsa-sys 0.4.0 source found: pass the path to its src/generated.rs as the" >&2
    echo "first argument — taken from a crates.io download of alsa-sys 0.4.0, because this" >&2
    echo "repository's [patch.crates-io] means cargo never fetches the registry crate" >&2
    exit 1
fi

layer=${2:-}
if [ -z "$layer" ]; then
    layer=$(from_registry alsa-0.11.0/src) || layer=
fi
if [ -z "$layer" ] || [ ! -d "$layer" ]; then
    echo "no alsa 0.11.0 source found: pass the path to its src directory as the second" >&2
    echo "argument, taken from a crates.io download of alsa 0.11.0" >&2
    exit 1
fi

actual=$(digest_of "$bindings")
if [ "$actual" != "$DIGEST" ]; then
    echo "unexpected input: $bindings" >&2
    echo "  expected sha256 $DIGEST" >&2
    echo "  got      sha256 $actual" >&2
    exit 1
fi
actual=$(digest_of "$layer"/*.rs)
if [ "$actual" != "$LAYER_DIGEST" ]; then
    echo "unexpected keep-set input: $layer" >&2
    echo "  expected sha256 $LAYER_DIGEST over its src/*.rs" >&2
    echo "  got      sha256 $actual" >&2
    exit 1
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/alsa-sys-dlopen-regenerate.XXXXXX")
trap 'rm -rf "$work"; rm -f src/.generated.rs.new' 0
rustc --edition 2021 -O -o "$work/transform" regenerate.rs
# Written beside the committed file and renamed over it once the transform has produced a whole
# surface: a redirect straight onto it would leave the crate's bindings empty if the transform
# failed, and a rename inside the directory is atomic where a move across filesystems is not.
"$work/transform" "$bindings" "$layer" > src/.generated.rs.new
mv src/.generated.rs.new src/generated.rs
echo "wrote src/generated.rs from $bindings and $layer"
