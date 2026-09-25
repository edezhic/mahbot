#!/bin/sh
#
# MahBot's own installer for macOS and Linux.
#
# It downloads this system's ready-made release file, puts it in place, makes
# the location visible to the owner's own shell, and starts the product. It
# never builds anything.
#
# The release contract it mirrors — one contract in four places: this script, the
# Windows one (`install.ps1`), the product's own updater (`src/self_update.rs`)
# and the workflow that publishes the files (`.github/workflows/release.yml`):
#
#   newest release:  {base}/releases/latest/download/version.txt, body `<version>\n`
#                    — the newest release that is not a test release, and nothing
#                    at all while only test releases exist
#   exact version:   {base}/releases/download/v{version}/{asset}
#   version:         the newest release above, or one named as this script's own
#                    first argument, or in MAHBOT_INSTALL_VERSION — either
#                    spelling, `0.7.0` or the release mark's own `v0.7.0`
#   asset:           mahbot-{version}-{os}-{arch}.{ext}, os in macos/linux,
#                    arch in x86_64/aarch64, ext tar.gz, holding exactly one
#                    file named `mahbot` at its root
#   install:         ~/.local/bin/mahbot (mode 0755) — the per-user programs
#                    directory `src/util/managed_bin.rs::mahbot_install_dir`
#                    names on unix
#
# The search-path block is the product's own (`src/util/owner_path.rs`): its two
# markers, one leading newline and one trailing one, and between the markers exactly
# one body line of the shape the product reads back out. The body line is this
# script's own — the one directory it puts a file in — and the product's own sync
# replaces it with its own line naming every directory it manages; the markers and
# the one-body-line shape are what must not drift, because the product strips a run
# between the markers only when exactly one body line sits inside it, so any other
# shape would leave a block it could never take out again — and a second block beside
# the first would sit on the owner's search path for good. The file it goes into
# is the one the shell he actually runs reads, resolved the way the product
# resolves it — his account's record first, then `$SHELL` — because that is the
# file the product itself rewrites, and any other choice would leave the two of
# us keeping two different files.
#
# MAHBOT_RELEASE_BASE_URL is a test-only hook, the one `src/self_update.rs` also
# honours: it moves the release base off the constant below, which is how the
# published files can be exercised end to end against another release host. With it
# unset nothing differs.
#
# MAHBOT_INSTALL_VERSION names the version to install when the script arrives down a
# pipe (`curl … | sh`), where no argument of its own can reach it — the same hook the
# Windows script honours.

set -eu

RELEASE_BASE="${MAHBOT_RELEASE_BASE_URL:-https://github.com/edezhic/mahbot}"

# The floors the two published files are built against, and nothing is installed
# below them: the macOS files carry a deployment target of 12.3, and the Linux ones
# are built on the release's own base (`ubuntu-22.04` in the workflow), which is
# glibc 2.35. Below either, the file cannot load at all, which is the same refusal as
# having none — and a loader error is not what the owner is told about it.
MACOS_FLOOR_MAJOR=12
MACOS_FLOOR_MINOR=3
GLIBC_FLOOR_MAJOR=2
GLIBC_FLOOR_MINOR=35

# Printed for every system the project publishes no file for: one prefix, the plain
# refusal built from it, and the sentences naming which systems do have one — the two
# floors, and the Linux host with no glibc at all.
NO_FILE_PREFIX='MahBot has no file for this system'
NO_FILE="$NO_FILE_PREFIX, so nothing was installed."
NO_MACOS="$NO_FILE_PREFIX: macOS $MACOS_FLOOR_MAJOR.$MACOS_FLOOR_MINOR or newer is required, so nothing was installed."
NO_GLIBC="$NO_FILE_PREFIX: Linux with glibc $GLIBC_FLOOR_MAJOR.$GLIBC_FLOOR_MINOR or newer is required, so nothing was installed."
NO_LIBC="$NO_FILE_PREFIX: the released Linux files are built for glibc, and no glibc was found on this system, so nothing was installed."

# The product's own block (`src/util/owner_path.rs`): its markers, the opening and
# closing each of its two body lines has there (the very things that make a line a
# body line the product reads back out), and the body line per shell family that
# names this script's install directory.
BLOCK_START='# >>> mahbot managed binaries >>>'
BLOCK_END='# <<< mahbot managed binaries <<<'
SH_BODY_PREFIX='export PATH="$PATH:'
SH_BODY_SUFFIX='"'
FISH_BODY_PREFIX='set -gx PATH $PATH '
SH_BODY='export PATH="$PATH:$HOME/.local/bin"'
FISH_BODY='set -gx PATH $PATH $HOME/.local/bin'

# ── What the script says ──────────────────────────────────────────────────

say() { printf '%s\n' "$1"; }
warn() { printf '%s\n' "$1" >&2; }
fail() { printf '%s\n' "$1" >&2; exit 1; }

# ── Helpers ───────────────────────────────────────────────────────────────

# curl or wget, whichever this host has: nothing else is assumed. Which one it
# is matters from the version lookup on.
CURL=''
WGET=''
if command -v curl >/dev/null 2>&1; then
    CURL=curl
elif command -v wget >/dev/null 2>&1; then
    WGET=wget
fi

# Download `url` to `dest`, or to standard output when `dest` is `-`. Quiet on
# purpose: the script's own sentence is the only thing a failure prints.
fetch() {
    if [ -n "$CURL" ]; then
        "$CURL" -fsL -o "$2" "$1"
    else
        "$WGET" -q -O "$2" "$1"
    fi
}

# Whether the product's own lock file is provably held: the lock is an exclusive
# lock on the whole file, taken for its whole run (`crate::util::lock`), so `flock`
# is refused it and `lsof` sees the open file. A refusal is raised only on evidence,
# and never on a lock file under another home.
#
# `flock` is asked first, because what it answers is about the lock itself: refused
# without a word about it is the product holding it. A `flock` that could not do its
# job at all says so on its error stream, and that is no evidence either way — then
# `lsof` is asked, and a process it names holding the file open is evidence too. With
# neither tool able to answer, nothing is refused on a guess: the copy already
# running is then updated by the product's own path, and it says so itself when the
# second instance comes up.
lock_held() {
    if command -v flock >/dev/null 2>&1; then
        FLOCK_SAID=$(flock -n "$1" true 2>&1 >/dev/null) && return 1
        [ -z "$FLOCK_SAID" ] && return 0
    fi
    if command -v lsof >/dev/null 2>&1 && [ -n "$(lsof -t -- "$1" 2>/dev/null)" ]; then
        return 0
    fi
    return 1
}

# Whether the directory `$1` is already an entry of the search path. A trailing
# separator names the same directory, and the first and last entries are entries
# like any other, so the path is compared wrapped in separators.
path_has_dir() {
    case ":${PATH:-}:" in
        *":$1:"* | *":$1/:"*) return 0 ;;
    esac
    return 1
}

# Whether the version `$1` (`major[.minor…]`) reaches `$2.$3`: each part is compared
# as a whole number, so no version is read as a decimal — a minor of 7 is not seven
# tenths — and a version with no minor part counts as zero. A part that is not there
# or is not a number is a version this cannot compare, so it is refused rather than
# handed to the numeric tests.
reaches_floor() {
    case "$1" in
        *.*)
            MAJOR=${1%%.*}
            MINOR=${1#*.}
            ;;
        *)
            MAJOR=$1
            MINOR=0
            ;;
    esac
    MINOR=${MINOR%%.*}
    case "$MAJOR" in '' | *[!0-9]*) return 1 ;; esac
    case "$MINOR" in '' | *[!0-9]*) return 1 ;; esac
    [ "$MAJOR" -gt "$2" ] && return 0
    [ "$MAJOR" -eq "$2" ] && [ "$MINOR" -ge "$3" ]
}

# The shell the owner's account record names, or nothing when neither `getent`
# nor `dscl` can be asked for it. The product reads that record directly
# (`crate::shell_env::owner_shell_and_home`); these are the tools that print it.
recorded_shell() {
    if command -v getent >/dev/null 2>&1; then
        getent passwd "$(id -u)" 2>/dev/null | awk -F: '{ print $7; exit }'
    elif command -v dscl >/dev/null 2>&1; then
        dscl . -read "/Users/$(id -un)" UserShell 2>/dev/null |
            awk '/^UserShell:/ { print $2; exit }'
    fi
}

# The startup file the block belongs in, or nothing when there is none: a shell
# whose file the product cannot name gets no block at all — never one invented
# for it, whose mere presence would change which files that shell reads.
target_file() {
    case "$1" in
        zsh)
            if [ -n "${ZDOTDIR:-}" ]; then
                case "$ZDOTDIR" in
                    /*)
                        printf '%s\n' "$ZDOTDIR/.zshrc"
                        return 0
                        ;;
                esac
            fi
            printf '%s\n' "$HOME/.zshrc"
            ;;
        bash)
            if [ "$OS" = macos ]; then
                # Whatever exists, not whatever is a regular file: the product picks
                # the same file (`crate::util::owner_path::login_profile_file`) and
                # asks `exists()`, so a file of another kind must not send the two of
                # us to different files.
                for file in "$HOME/.bash_profile" "$HOME/.bash_login" "$HOME/.profile"; do
                    if [ -e "$file" ]; then
                        printf '%s\n' "$file"
                        return 0
                    fi
                done
                printf '%s\n' "$HOME/.bash_profile"
            else
                printf '%s\n' "$HOME/.bashrc"
            fi
            ;;
        fish)
            if [ -n "${XDG_CONFIG_HOME:-}" ]; then
                case "$XDG_CONFIG_HOME" in
                    /*)
                        printf '%s\n' "$XDG_CONFIG_HOME/fish/config.fish"
                        return 0
                        ;;
                esac
            fi
            printf '%s\n' "$HOME/.config/fish/config.fish"
            ;;
        sh | dash | ash | ksh | ksh93 | mksh | pdksh)
            if [ "$OS" = macos ]; then
                printf '%s\n' "$HOME/.profile"
            else
                # The one startup file an interactive POSIX shell reads here is
                # the file `$ENV` names: any other would be a guess, and writing
                # the wrong one puts the block where his terminal never looks. A
                # relative value is refused like an empty one.
                if [ -n "${ENV:-}" ]; then
                    case "$ENV" in
                        /*)
                            if [ -e "$ENV" ]; then
                                printf '%s\n' "$ENV"
                            fi
                            ;;
                    esac
                fi
            fi
            ;;
    esac
}

# Whether the startup file already holds a block the product would recognise as its
# own: one of its start markers, then exactly one line of the shape it writes, then
# its end marker. That is precisely what the product itself counts as its block
# (`crate::util::owner_path::block_regions` and `is_product_body`), and the very same
# shape is what makes the location visible, so the same block is what this writes
# there too. The rule for a run that never closes is the product's own as well: it is
# not evidence for a block and the scan goes on past it, so a later block is still
# found. Anything else in that state — a pair with no body line between them, a body
# line of another shape, a run holding any second line — is a block the product would
# not take out, so it is not this block and this script writes the product's own
# beside it rather than leaving the location unnamed.
holds_block() {
    [ -f "$1" ] || return 1
    awk -v start="$BLOCK_START" -v end="$BLOCK_END" \
        -v sh_open="$SH_BODY_PREFIX" -v sh_close="$SH_BODY_SUFFIX" \
        -v fish_open="$FISH_BODY_PREFIX" '
        # The body-line shapes the product itself reads back out
        # (`owner_path::body_entries`): the posix form, closing quote included, or the
        # fish form with anything after its opening.
        function is_body(line) {
            return (index(line, sh_open) == 1 && substr(line, length(line)) == sh_close) ||
                   index(line, fish_open) == 1
        }
        # The state machine the product itself reads a file with
        # (`owner_path::block_regions`): a start marker opens a run, the run closes only
        # when the end marker immediately follows its one body line, and a run that
        # never closes is dropped while the scan goes on, so a later block is found
        # where the product finds one.
        $0 == start { state = 1; next }
        state == 1 { state = is_body($0) ? 2 : 3; next }
        state == 2 { if ($0 == end) { closed = 1; exit }; state = 3; next }
        END { exit !closed }
    ' "$1"
}

# Append the product's block. Every other byte of the file stays where it is: a
# last line without a newline is closed first, so the block's own leading
# newline never lands on one of the owner's own lines, and a file that does not
# exist yet is created by the append itself. A write that fails says nothing of
# its own: the caller states it.
add_block() {
    if [ -s "$1" ] && [ -n "$(tail -c 1 "$1" 2>/dev/null)" ]; then
        printf '\n' >>"$1" 2>/dev/null || return 1
    fi
    printf '\n%s\n%s\n%s\n' "$BLOCK_START" "$2" "$BLOCK_END" >>"$1" 2>/dev/null
}

# ── 1. Refuse while the product is running ────────────────────────────────

[ -n "${HOME:-}" ] ||
    fail "HOME is not set, so there is no place to install MahBot into."
INSTALL_DIR="$HOME/.local/bin"
DEST="$INSTALL_DIR/mahbot"

# Scoped to the data directory this home's product uses, so a product running
# under another home is not this script's business.
if [ -e "$HOME/.mahbot/mahbot.lock" ] && lock_held "$HOME/.mahbot/mahbot.lock"; then
    fail "MahBot is already running; updating a running installation is MahBot's own update path, so nothing was installed."
fi

# ── 2. Work out the system ────────────────────────────────────────────────

case "$(uname -s)" in
    Darwin) OS=macos ;;
    Linux) OS=linux ;;
    *) fail "$NO_FILE" ;;
esac
case "$(uname -m)" in
    x86_64 | amd64) ARCH=x86_64 ;;
    arm64 | aarch64) ARCH=aarch64 ;;
    *) fail "$NO_FILE" ;;
esac
# The floors the files are built against. Below either, the published file cannot
# load at all, so the refusal is the same plain one rather than a loader error after
# the running copy has been replaced. Each check refuses only on what it can actually
# read: a host that cannot be asked is not refused for that.
if [ "$OS" = macos ]; then
    MACOS_VERSION=$(sw_vers -productVersion 2>/dev/null || true)
    if [ -n "$MACOS_VERSION" ] &&
        ! reaches_floor "$MACOS_VERSION" "$MACOS_FLOOR_MAJOR" "$MACOS_FLOOR_MINOR"; then
        fail "$NO_MACOS"
    fi
else
    GLIBC_REPORT=$(getconf GNU_LIBC_VERSION 2>/dev/null || true)
    case "$GLIBC_REPORT" in
        glibc\ *)
            # What this host answers with is the evidence that decides: at or above
            # the floor the published file loads, below it there is none for it.
            if ! reaches_floor "${GLIBC_REPORT#glibc }" "$GLIBC_FLOOR_MAJOR" "$GLIBC_FLOOR_MINOR"; then
                fail "$NO_GLIBC"
            fi
            ;;
        *)
            # No glibc named. The musl loader's presence is then the check the
            # product makes too (`crate::util::managed_bin::linux_host_is_musl`):
            # `/lib` is a link to `/usr/lib` on Debian, where musl can sit beside
            # glibc, so it is only evidence where no glibc answered — but a host
            # that names none has none of its own to load the file with, which is
            # the sentence the product refuses that host with as well.
            if [ -e /lib/ld-musl-x86_64.so.1 ] || [ -e /lib/ld-musl-aarch64.so.1 ]; then
                fail "$NO_LIBC"
            fi
            ;;
    esac
fi

# ── 3. The version to install ─────────────────────────────────────────────

# The version file is fetched, and a named version is still downloaded, so
# nothing can go on without one of the two tools.
[ -n "$CURL$WGET" ] ||
    fail "neither curl nor wget is available to download MahBot, so nothing was installed."

if [ -n "${1:-}" ]; then
    # A version named on the command line is how one particular release is
    # installed; nothing is looked up then.
    VERSION=$1
elif [ -n "${MAHBOT_INSTALL_VERSION:-}" ]; then
    # The same version, named through the environment for the form that arrives
    # down a pipe, where no argument of the script's own can reach it.
    VERSION=$MAHBOT_INSTALL_VERSION
else
    # The newest release there is: the host's own newest-release file, which names
    # the newest release that is not a test release — and nothing at all while only
    # test releases exist, where one is installed by naming its version instead.
    RAW=$(fetch "$RELEASE_BASE/releases/latest/download/version.txt" -) ||
        fail "the newest MahBot release could not be looked up, so nothing was installed (install.sh <version>, or MAHBOT_INSTALL_VERSION, installs one particular release)."
    VERSION=$(printf '%s' "$RAW" | tr -d '\r\n')
fi
# A version named by hand — as this script's own argument or through
# MAHBOT_INSTALL_VERSION — may be spelled the way the release's own mark is
# (`v0.7.0`), while the file's name and the address it is fetched from are built
# from the version itself: the mark's leading `v` is dropped here rather than
# landing in either.
VERSION=${VERSION#v}
[ -n "$VERSION" ] ||
    fail "the MahBot release to install could not be determined, so nothing was installed."

# A pre-release part says plainly what it is: such a release is not what the
# owner is given normally.
case "$VERSION" in
    *-*)
        say "The MahBot $VERSION release is a test release, not a normal one; installing it."
        ;;
esac

# ── 4. Download the file and put it in place ──────────────────────────────

# Nothing here prints anything of its own: the sentence after each step is the
# whole of what the owner is told, so the tool's own line is dropped rather than
# left beside it.
ASSET="mahbot-$VERSION-$OS-$ARCH.tar.gz"
TMP=$(mktemp -d 2>/dev/null) ||
    fail "a temporary directory could not be created, so nothing was installed."
STAGED=''
cleanup() {
    if [ -n "$STAGED" ]; then
        rm -f "$STAGED"
        STAGED=''
    fi
    if [ -n "$TMP" ]; then
        rm -rf "$TMP"
        TMP=''
    fi
}
trap cleanup EXIT
trap 'cleanup; exit 1' HUP INT TERM

fetch "$RELEASE_BASE/releases/download/v$VERSION/$ASSET" "$TMP/$ASSET" ||
    fail "the MahBot $VERSION file for this system could not be downloaded, so nothing was installed."
tar -xzf "$TMP/$ASSET" -C "$TMP" mahbot 2>/dev/null ||
    fail "the downloaded MahBot file could not be opened, so nothing was installed."
# A half-written command is never left in place: the file is staged beside its
# destination and moved onto it, which is a rename.
mkdir -p "$INSTALL_DIR" 2>/dev/null ||
    fail "MahBot's own directory could not be created, so nothing was installed."
STAGED="$INSTALL_DIR/mahbot.new"
cp "$TMP/mahbot" "$STAGED" 2>/dev/null ||
    fail "MahBot could not be put in place at $DEST, so nothing was installed."
chmod 755 "$STAGED" 2>/dev/null ||
    fail "MahBot could not be put in place at $DEST, so nothing was installed."
mv -f "$STAGED" "$DEST" 2>/dev/null ||
    fail "MahBot could not be put in place at $DEST, so nothing was installed."
STAGED=''

# ── 5. Make the location visible ──────────────────────────────────────────

# The shell the owner actually runs: the one his account record names, else the
# one `$SHELL` names.
SHELL_PATH=$(recorded_shell)
if [ -z "$SHELL_PATH" ]; then
    SHELL_PATH=${SHELL:-}
fi
SHELL_NAME=${SHELL_PATH##*/}
TARGET=$(target_file "$SHELL_NAME")
if [ "$SHELL_NAME" = fish ]; then
    BODY=$FISH_BODY
else
    BODY=$SH_BODY
fi

# Two reasons to write nothing: the directory is already an entry of his own
# search path, or a block of the product's own is already in the file, where a
# second one must never go. Everything else that stops the location from being
# made visible is said plainly — no startup file his shell reads, and a write that
# fails — and no tool's own line is left beside that sentence.
if ! path_has_dir "$INSTALL_DIR"; then
    if [ -z "$TARGET" ]; then
        warn "the startup file the owner's own shell reads could not be resolved, so $INSTALL_DIR was not added to his search path."
    elif ! holds_block "$TARGET"; then
        if ! mkdir -p "$(dirname "$TARGET")" 2>/dev/null; then
            warn "the directory for $TARGET could not be created, so the location was not made visible to the owner's own shell."
        elif ! add_block "$TARGET" "$BODY"; then
            warn "the block naming $INSTALL_DIR could not be written to $TARGET."
        fi
    fi
fi

# ── 6. Remove the copy the old way of installing left behind ──────────────

# Installing from the package registry (`cargo install mahbot`) puts the command
# in the toolchain's own directory. That copy is the owner's file, so a removal
# that fails is said plainly and the install goes on. The two locations are compared
# as the files they are, not as text: a `$CARGO_HOME` spelled with a redundant
# separator names the very file just installed, and removing that would leave the
# owner with nothing at all — `-ef` asks the filesystem instead of the strings.
LEGACY="${CARGO_HOME:-$HOME/.cargo}/bin/mahbot"
if [ -f "$LEGACY" ] && [ ! "$LEGACY" -ef "$DEST" ]; then
    if ! rm -f "$LEGACY" 2>/dev/null; then
        warn "the copy the old way of installing left at $LEGACY could not be removed; MahBot is installed at $DEST."
    fi
fi

# ── 7. Start the product ──────────────────────────────────────────────────

# The trap cannot clean up after an `exec` — it replaces this shell, and no trap
# runs then — so the temporary directory goes now.
cleanup
exec "$DEST"
