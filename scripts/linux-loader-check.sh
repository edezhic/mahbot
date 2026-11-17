#!/bin/sh
# Prove what a Linux release artefact needs at load. Invoked by
# `.github/workflows/release.yml` on every Linux leg, on the built file itself.
#
#   linux-loader-check.sh <binary> <target-triple>
#
# The product must start on a server that has no audio stack and no compression
# library: the audio library is loaded on first use (`vendor/alsa-sys-dlopen`) and
# zlib is built from its own source into the file. Both are asserted here, so a
# release that names either as a loader dependency is refused instead of shipped.
#
# Nothing is built here. The artefact is the one the leg already built, and the
# graph is the one its lockfile resolves — `cargo tree --locked` reports the same
# resolution the build used, so there is no second build to drift from the first.
set -eu

if [ $# -lt 2 ]; then
    echo "usage: linux-loader-check.sh <binary> <target-triple>" >&2
    exit 1
fi

binary=$1
triple=$2

fail() {
    echo "::error::$*" >&2
    exit 1
}

[ -f "$binary" ] || fail "no artefact at $binary"
command -v readelf >/dev/null 2>&1 ||
    fail "readelf was not found; binutils is what reads the artefact's NEEDED list"

# ── The replacement is in the graph ──────────────────────────────────

# `[patch.crates-io]` points `alsa-sys` at the vendored shim. When the patch stops
# applying — the crate it replaces moved to a version it no longer matches — cargo
# only warns and builds the registry crate instead, whose build script links
# `libasound` at load. The resolution says which one this build got: the shim's own
# path, or a registry crate. `--invert` names it either way.
#
# A graph with no ALSA crate at all needs no replacement: nothing is left to link
# the library, and the file's own NEEDED list below is then the whole answer.
tree=$(mktemp "${TMPDIR:-/tmp}/alsa-tree.XXXXXX")
trap 'rm -f "$tree"' 0
if cargo tree --locked --target "$triple" --invert alsa-sys >"$tree" 2>&1; then
    # The resolution's own line, never the warning above it: that one names the patch's path
    # whether or not the patch applied.
    if grep -qE '^alsa-sys v[^ ]+ \([^)]*alsa-sys-dlopen\)' "$tree"; then
        : # the shim is in this build
    elif grep -q 'nothing to print' "$tree" || ! grep -q '^alsa-sys ' "$tree"; then
        # Cargo answers an empty tree either way, depending on the version.
        echo "this graph has no ALSA crate, so there is no replacement to apply"
    else
        cause=$(grep 'was not used in the crate graph' "$tree" | head -n 1)
        fail "alsa-sys resolved from the registry, not from vendor/alsa-sys-dlopen, so the file loads libasound at start: the [patch.crates-io] replacement is not in this build${cause:+ ($cause)}"
    fi
elif grep -q 'did not match any packages' "$tree"; then
    echo "this graph has no ALSA crate, so there is no replacement to apply"
else
    fail "cargo tree could not resolve alsa-sys for $triple: $(head -n 3 "$tree")"
fi

# ── The file itself ──────────────────────────────────────────────────

dynamic=$(readelf -d "$binary" 2>&1) || fail "readelf could not read $binary: $dynamic"
needed=$(printf '%s\n' "$dynamic" | awk '/NEEDED/ { sub(/.*\[/, ""); sub(/\].*/, ""); print }')
[ -n "$needed" ] || fail "readelf found no NEEDED entries in $binary"
echo "loader dependencies of $binary:"
printf '%s\n' "$needed" | sed 's/^/  /'
for library in libasound libz.so; do
    if printf '%s\n' "$needed" | grep -q "^$library"; then
        fail "$binary loads $library at start; it must be loaded when it is used, or built in from its own source"
    fi
done
echo "no load-time dependency on the audio library or on zlib"
