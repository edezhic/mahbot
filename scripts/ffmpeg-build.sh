#!/bin/sh
# Build the pair the product downloads — `ffmpeg` and `ffprobe` — linked against
# the codec, image and TLS libraries the product's video jobs need. Invoked by
# `.github/workflows/ffmpeg-release.yml` on every leg of the release.
#
#   ffmpeg-build.sh <sources-dir> <deps-prefix> <recipe-file> [<macos-floor>]
#
# <sources-dir> holds the unpacked trees of ffmpeg, x264, x265, libvpx, opus,
# zimg, zlib and (on unix) openssl, each in a directory named after the
# component. <deps-prefix> is where every dependency is installed: the unix legs
# pass their own scratch prefix, the Windows legs the MSYS2 toolchain's own
# (`$MINGW_PREFIX`), which is where its pkg-config already looks. <recipe-file>
# receives every command this script runs, together with what it ran them with —
# that file is the leg's section of the published BUILD-RECIPE.txt.
# <macos-floor> is the macOS legs' `-mmacosx-version-min` flag, empty elsewhere.
#
# The build's own identities come from the environment, because they are what
# COMPONENTS.txt, SOURCE.txt and the recipe record: the upstream tag, the version
# its tree reports and this recipe's revision; the seven library pins, each named
# as its source archive is; and the commit and clone host the upstream tag
# resolved to. `.github/workflows/ffmpeg-release.yml`'s `env:` block carries them
# for every release leg, and a re-run by hand has to supply the same ones.
#
# `CMAKE_GENERATOR` picks the build tool cmake drives; the release legs have
# Ninja, and a machine without it can name `Unix Makefiles` to re-run the recipe.
#
# The build is GPLv3. `--enable-gpl` is what `--enable-libx264` and
# `--enable-libx265` require, and `--enable-version3` is what the OpenSSL 3
# backend requires beside it — configure refuses that pair otherwise. The
# product itself is unchanged: it runs this pair as a separate program.
#
# Nothing is taken from the image: `--disable-autodetect`, and every library
# enabled here was built here from the pinned sources in <sources-dir>.
set -eu

if [ $# -lt 3 ]; then
    echo "usage: ffmpeg-build.sh <sources-dir> <deps-prefix> <recipe-file> [<macos-floor>]" >&2
    exit 1
fi

: "${FFMPEG_TAG:?the upstream tag the pair is built from}"
: "${FFMPEG_VERSION:?the version the upstream tree reports}"
: "${BUILD_REVISION:?the revision of this build recipe}"
: "${SOURCE_COMMIT:?the commit the upstream tag resolved to}"
: "${SOURCE_HOST:?the host the upstream tag was cloned from}"
: "${X264_REF:?the x264 pin, as its source archive is named}"
: "${X265_REF:?the x265 pin, as its source archive is named}"
: "${VPX_REF:?the libvpx pin, as its source archive is named}"
: "${OPUS_REF:?the opus pin, as its source archive is named}"
: "${ZIMG_REF:?the zimg pin, as its source archive is named}"
: "${ZLIB_REF:?the zlib pin, as its source archive is named}"
: "${OPENSSL_REF:?the openssl pin, as its source archive is named}"

sources=$(cd "$1" && pwd)
prefix=$2
recipe=$3
floor=${4:-}
if [ -z "$prefix" ]; then
    echo "an install prefix is required; the Windows legs pass \$MINGW_PREFIX" >&2
    exit 1
fi

os=$(uname -s)
case $os in
Darwin*) platform=macos ;;
Linux*) platform=linux ;;
# Every MSYS2 environment reports `<MSYSTEM>_NT-<build>`: MINGW64, UCRT64, CLANG64 and
# CLANGARM64 alike, so the suffix is what says Windows, not the environment's name.
*_NT-*) platform=windows ;;
*)
    echo "unsupported host: $os" >&2
    exit 1
    ;;
esac

# The MSYS2 environments share one kernel: the environment, not `uname`, is what
# says which machine the result runs on.
if [ -n "${MSYSTEM:-}" ]; then
    case $MSYSTEM in
    CLANGARM64) arch=aarch64 ;;
    MINGW32) arch=i686 ;;
    *) arch=x86_64 ;;
    esac
else
    case $(uname -m) in
    arm64 | aarch64) arch=aarch64 ;;
    x86_64 | amd64) arch=x86_64 ;;
    *)
        echo "unsupported machine: $(uname -m)" >&2
        exit 1
        ;;
    esac
fi

jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 2)

# The Windows legs are clang: MSYS2's CLANGARM64 environment carries no gcc at all,
# and one compiler for every library is one recipe. `ar`, `ranlib` and `strip` come
# from the `binutils` package both environments install.
if [ "$platform" = windows ]; then
    export CC=clang CXX=clang++
fi

# Every object, the dependencies included, has to run on the oldest macOS the
# product itself does. Clang takes the floor from here, whichever build system
# invokes it.
if [ -n "$floor" ]; then
    export MACOSX_DEPLOYMENT_TARGET=${floor#-mmacosx-version-min=}
fi

# The unix legs' libraries go to a prefix of their own, so pkg-config has to be
# told where. On Windows they go into the toolchain's prefix, which pkg-config
# already searches.
if [ "$platform" != windows ]; then
    export PKG_CONFIG_PATH="$prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
fi

# ffmpeg links the C++ libraries (x265, zimg) with the C driver, so the C++ runtime is
# named on the link line, and which one it is belongs to the toolchain: Apple's clang
# always links libc++ and ships no static libc++.a for a probe to find, llvm-mingw
# carries one, and gcc's toolchains link libstdc++.
cxx=${CXX:-c++}
if [ "$platform" = macos ] || [ "$($cxx -print-file-name=libc++.a)" != libc++.a ]; then
    cxx_runtime=-lc++
else
    cxx_runtime=-lstdc++
fi

: >"$recipe"

# A command line, verbatim, in the recipe; the command itself follows in the log.
record() {
    printf '%s\n' "$*" >>"$recipe"
}

run_in() {
    dir=$1
    shift
    record "( cd $dir && $* )"
    ( cd "$dir" && "$@" ) || {
        status=$?
        echo "failed: $*" >&2
        exit "$status"
    }
}

cmark() {
    dir=$1
    build=$2
    shift 2
    run_in "$dir" cmake -S "$dir" -B "$build" -G "${CMAKE_GENERATOR:-Ninja}" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" "$@"
    run_in "$dir" cmake --build "$build" --parallel "$jobs"
    run_in "$dir" cmake --install "$build"
}

# ── What travels with the pair ──────────────────────────────────────

# The licence material of the build: the text of the licence the whole is
# distributed under, the notices of the libraries linked into it, and a note naming
# the source of every one of them. The packaging step copies these three files into
# the archive, and the product's installer puts them beside the binaries.

components() {
    echo "  libx264   $X264_REF (GPL-2.0-or-later)  x264-$X264_REF-source.tar.gz"
    echo "  libx265   $X265_REF (GPL-2.0-or-later)  x265-$X265_REF-source.tar.gz"
    echo "  libvpx    $VPX_REF (BSD-3-Clause)  libvpx-$VPX_REF-source.tar.gz"
    echo "  libopus   $OPUS_REF (BSD-3-Clause)  opus-$OPUS_REF-source.tar.gz"
    echo "  zimg      $ZIMG_REF (WTFPL)  zimg-$ZIMG_REF-source.tar.gz"
    echo "  zlib      $ZLIB_REF (Zlib)  zlib-$ZLIB_REF-source.tar.gz"
    if [ "$platform" = windows ]; then
        echo "  TLS       Windows SChannel, part of the system (nothing to ship)"
    else
        echo "  OpenSSL   $OPENSSL_REF (Apache-2.0)  openssl-$OPENSSL_REF-source.tar.gz"
    fi
}
components >"$sources/COMPONENTS.txt"

# `--enable-gpl --enable-version3` is GPLv3, and that is the text that travels.
cp "$sources/ffmpeg/COPYING.GPLv3" "$sources/LICENCE.txt"

{
    for component in x264:COPYING x265:COPYING libvpx:LICENSE opus:COPYING \
        zimg:COPYING zlib:LICENSE; do
        name=${component%%:*}
        notice=${component#*:}
        printf '── %s ──\n' "$name"
        cat "$sources/$name/$notice"
        echo
    done
    if [ "$platform" != windows ]; then
        printf '── OpenSSL %s ──\n' "$OPENSSL_REF"
        cat "$sources/openssl/LICENSE.txt"
        echo
    fi
} >"$sources/LICENCES.txt"

{
    printf 'FFmpeg %s - upstream project: FFmpeg (https://ffmpeg.org/), tag %s\n' \
        "$FFMPEG_VERSION" "$FFMPEG_TAG"
    printf 'Upstream repository clone: %s\n' "$SOURCE_HOST"
    printf 'Resolved commit: %s\n' "$SOURCE_COMMIT"
    printf 'Source archive: ffmpeg-%s-mahbot.%s-source.tar.gz, published with this release\n' \
        "$FFMPEG_VERSION" "$BUILD_REVISION"
    printf 'This build is licensed under the GNU General Public License, version 3: LICENCE.txt.\n'
    printf 'It links these libraries, each built from the source archive named beside it and\n'
    printf 'published with this release — or, where the line says so, part of the system:\n'
    cat "$sources/COMPONENTS.txt"
    printf 'Their notices are collected in LICENCES.txt.\n'
    printf 'FFmpeg is invoked by the product as a separate program, never linked into it.\n'
} >"$sources/SOURCE.txt"

# ── The recipe's own header ─────────────────────────────────────────

{
    echo "── $platform-$arch ──"
    echo "runner: ${ImageOS:-unknown} ${ImageVersion:-unknown}"
    echo "system: $(uname -srm)"
    if [ "$platform" = macos ]; then
        echo "macOS: $(sw_vers -productVersion 2>/dev/null || echo unknown)"
    elif [ "$platform" = linux ]; then
        echo "libc: $(ldd --version 2>&1 | sed -n '1p')"
    fi
    if [ "$platform" = linux ]; then
        echo "compiler: $(cc --version 2>&1 | sed -n '1p')"
    else
        echo "compiler: $(clang --version 2>&1 | sed -n '1p')"
    fi
    if command -v nasm >/dev/null 2>&1; then
        echo "assembler: $(nasm -v 2>&1 | sed -n '1p')"
    fi
    echo "source: ffmpeg $FFMPEG_TAG, commit $SOURCE_COMMIT (cloned from $SOURCE_HOST)"
    echo "licence: GNU GPL version 3 (LICENCE.txt); notices of the libraries in LICENCES.txt"
    echo "linked libraries, each built here from the source archive of its name:"
    cat "$sources/COMPONENTS.txt"
    if [ -n "${MACOSX_DEPLOYMENT_TARGET:-}" ]; then
        echo "MACOSX_DEPLOYMENT_TARGET: $MACOSX_DEPLOYMENT_TARGET"
    fi
    if [ -n "${PKG_CONFIG_PATH:-}" ]; then
        echo "PKG_CONFIG_PATH: $PKG_CONFIG_PATH"
    fi
    echo
} >>"$recipe"

# ── zlib: PNG ───────────────────────────────────────────────────────

run_in "$sources/zlib" ./configure --static --prefix="$prefix"
run_in "$sources/zlib" make -j"$jobs"
run_in "$sources/zlib" make install

# ── x264: H.264 ─────────────────────────────────────────────────────

set -- --prefix="$prefix" --enable-static --disable-cli --disable-opencl
if [ "$platform" = windows ]; then
    set -- "$@" --host="$arch-w64-mingw32"
fi
run_in "$sources/x264" ./configure "$@"
run_in "$sources/x264" make -j"$jobs"
run_in "$sources/x264" make install

# ── x265: HEVC ──────────────────────────────────────────────────────

# The cli, the shared library and libnuma are all off: only the static library is
# installed, and libnuma would be one more thing the owner's system must carry.
# x265's aarch64 assembly is written for the ELF assembler and not for Windows on
# ARM, so that leg builds the portable C++ (MSYS2's own build does the same) — the
# encoder is the same, only slower.
x265_assembly=ON
if [ "$platform" = windows ] && [ "$arch" != x86_64 ]; then
    x265_assembly=OFF
fi
cmark "$sources/x265/source" "$sources/x265/build" \
    -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_LIBNUMA=OFF \
    -DENABLE_HDR10_PLUS=OFF -DHIGH_BIT_DEPTH=OFF -DENABLE_ASSEMBLY="$x265_assembly"

# ── libvpx: VP9 ─────────────────────────────────────────────────────

# libvpx names its macOS targets by Darwin major, and calls the ISA `arm64`: the
# unversioned `arm64-darwin-gcc` is its iOS target, and only the versioned form builds
# against the macOS SDK. 1.17 knows darwin20 to darwin25; a newer kernel builds with the
# newest name it has, because the number only picks the `-arch` and the SDK path.
darwin=$(uname -r | cut -d. -f1)
if [ "$darwin" -gt 25 ]; then
    darwin=25
fi
case "$platform-$arch" in
macos-aarch64) vpx_target="arm64-darwin$darwin-gcc" ;;
macos-x86_64) vpx_target="x86_64-darwin$darwin-gcc" ;;
linux-aarch64) vpx_target=arm64-linux-gcc ;;
linux-x86_64) vpx_target=x86_64-linux-gcc ;;
windows-aarch64) vpx_target=arm64-win64-gcc ;;
windows-x86_64) vpx_target=x86_64-win64-gcc ;;
*)
    echo "no libvpx target for $platform-$arch" >&2
    exit 1
    ;;
esac
run_in "$sources/libvpx" ./configure --target="$vpx_target" --prefix="$prefix" \
    --disable-shared --enable-static --disable-examples --disable-tools \
    --disable-unit-tests --disable-docs
run_in "$sources/libvpx" make -j"$jobs"
run_in "$sources/libvpx" make install

# ── opus: Opus ──────────────────────────────────────────────────────

cmark "$sources/opus" "$sources/opus/build" \
    -DOPUS_BUILD_SHARED_LIBRARY=OFF -DOPUS_BUILD_PROGRAMS=OFF -DOPUS_BUILD_TESTING=OFF

# ── zimg: zscale ────────────────────────────────────────────────────

# zimg's tree carries no generated `configure`: `autogen.sh` runs autoreconf, and
# the workflow installs the autotools it needs.
run_in "$sources/zimg" ./autogen.sh
run_in "$sources/zimg" ./configure --prefix="$prefix" --enable-static --disable-shared
run_in "$sources/zimg" make -j"$jobs"
run_in "$sources/zimg" make install

# ── OpenSSL: https, on the unix legs ────────────────────────────────

# Windows takes its TLS from SChannel, which is part of the system and carries
# its own trust store; the unix legs carry OpenSSL, built here so that the pair
# needs no libssl on the owner's machine.
if [ "$platform" != windows ]; then
    case "$platform-$arch" in
    macos-aarch64) openssl_target=darwin64-arm64-cc ;;
    macos-x86_64) openssl_target=darwin64-x86_64-cc ;;
    *) openssl_target="linux-$arch" ;;
    esac
    # The compiled-in default certificate store is the platform's own `/etc/ssl`, not
    # `<deps-prefix>/ssl`: a static libcrypto must never point at a directory that exists
    # only on the machine that built it. This FFmpeg reads CAs from `ca_file` alone, and
    # whatever consults that store afterwards finds the owner's own. `install_sw` installs
    # no configuration, so this writes nothing outside the prefix.
    run_in "$sources/openssl" ./Configure "$openssl_target" --prefix="$prefix" \
        --libdir=lib --openssldir=/etc/ssl no-shared no-tests no-apps no-docs
    run_in "$sources/openssl" make -j"$jobs"
    run_in "$sources/openssl" make install_sw
fi

# ── ffmpeg ──────────────────────────────────────────────────────────

# One flag list per option, assembled before the configure line is built: the recipe
# publishes the line the build ran with, so the prefix's include and lib paths and
# each leg's own additions belong in the same `--extra-cflags`/`--extra-ldflags` and
# not in a second one after them.
cflags=-I"$prefix"/include
ldflags=-L"$prefix"/lib
libs=$cxx_runtime
if [ "$platform" = windows ]; then
    # The mingw C++ runtimes forward their threads to winpthreads, which the C driver
    # does not add by itself; `-static` folds it in, so no DLL travels beside the exe.
    ldflags="$ldflags -static"
    libs="$libs -lwinpthread"
fi
if [ -n "$floor" ]; then
    cflags="$cflags $floor"
    ldflags="$ldflags $floor"
fi

set -- --enable-gpl --enable-version3 --disable-nonfree \
    --disable-autodetect --disable-shared --enable-static \
    --disable-doc --disable-ffplay --disable-debug \
    --enable-zlib --enable-libx264 --enable-libx265 --enable-libvpx \
    --enable-libopus --enable-libzimg \
    --extra-cflags="$cflags" --extra-ldflags="$ldflags" --extra-libs="$libs"
if [ "$platform" = windows ]; then
    set -- "$@" --cc=clang --arch="$arch" --target-os=mingw32 --enable-schannel
else
    set -- "$@" --enable-openssl
fi
if [ -n "$floor" ]; then
    set -- "$@" --cc=clang
fi
run_in "$sources/ffmpeg" ./configure "$@"
run_in "$sources/ffmpeg" make -j"$jobs"

# Linux and Windows are stripped. macOS is not: the linker's ad-hoc signature is
# what lets an arm64 file execute, and `strip` would invalidate it.
case $platform in
linux) run_in "$sources/ffmpeg" strip --strip-unneeded ffmpeg ffprobe ;;
windows)
    if command -v strip >/dev/null 2>&1; then
        run_in "$sources/ffmpeg" strip ffmpeg.exe ffprobe.exe
    else
        run_in "$sources/ffmpeg" llvm-strip ffmpeg.exe ffprobe.exe
    fi
    ;;
esac
