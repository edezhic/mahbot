#!/bin/sh
#
# MahBot's own installer for macOS and Linux: it downloads this system's ready-made release file,
# puts it in place, makes the location visible to the owner's shell, gives the speech models their
# machine-wide home, and starts the product — as a system service on a Linux server, at login
# everywhere else. It never builds anything.
#
# Which of the two registrations a host gets is decided from a display manager's own files, never
# from the systemd default target: stock Ubuntu points that at `graphical.target` with no desktop at
# all, which is how servers came to be installed with no service and the product left in the
# foreground of the installing session. A host with no display manager has no graphical login for the
# product's own entry (`src/autostart.rs`) to run in, so it gets a unit instead.
# `MAHBOT_INSTALL_REGISTRATION=service|login` asks for the other registration, refused before
# anything is downloaded when this host cannot take it; `MAHBOT_INSTALL_DRY_RUN=1` prints what the
# run would do and touches nothing.
#
# The release contract, mirrored by `install.ps1`, `src/self_update.rs` and the release workflow:
#   newest release:  {base}/releases/latest/download/version.txt, body `<version>\n` — the newest
#                    release that is not a test release, and nothing while only test releases exist
#   exact version:   {base}/releases/download/v{version}/{asset}
#   version:         the newest release above, or one named as this script's first argument or in
#                    MAHBOT_INSTALL_VERSION, spelled `0.7.0` or `v0.7.0`
#   asset:           mahbot-{version}-{os}-{arch}.tar.gz, os in macos/linux, arch in
#                    x86_64/aarch64, holding exactly one file named `mahbot` at its root
#   install:         ~/.local/bin/mahbot (mode 0755), the per-user programs directory
#                    `src/util/managed_bin.rs::mahbot_install_dir` names on unix
#
# The search-path block is the product's own (`src/util/owner_path.rs`): its two markers and, between
# them, exactly one body line of the shape it reads back out — the shape must not drift, or the
# product can never take the block out again. The file is the one the owner's shell actually reads,
# resolved the way the product resolves it (his account's record, then `$SHELL`).
#
# MAHBOT_RELEASE_BASE_URL is the test-only hook `src/self_update.rs` also honours.
# MAHBOT_INSTALL_VERSION names the version for the form that arrives down a pipe (`curl … | sh`).

set -eu

RELEASE_BASE="${MAHBOT_RELEASE_BASE_URL:-https://github.com/edezhic/mahbot}"

# The floors the two published files are built against, and nothing is installed
# below them: the macOS files carry a deployment target of 14.0, and the Linux ones
# are built on the release's own base (`ubuntu-22.04` in the workflow), which is
# glibc 2.35. Below either, the file cannot load at all, which is the same refusal as
# having none — and a loader error is not what the owner is told about it.
MACOS_FLOOR_MAJOR=14
MACOS_FLOOR_MINOR=0
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

# How this run acts as root: nothing when it is root already, `sudo` when that is the way in. An empty
# `AS_ROOT` and no privilege at all are different states, which `PRIVILEGED` tells apart — a root
# install on a host without the sudo package is privileged, which `IS_ROOT` says — and `NO_ROOT_WHY`
# names what is missing when it is neither.
AS_ROOT=''
PRIVILEGED=''
IS_ROOT=''
NO_ROOT_WHY=''
if [ "$(id -u)" -eq 0 ]; then
    IS_ROOT=1
    PRIVILEGED=1
elif command -v sudo >/dev/null 2>&1; then
    PRIVILEGED=1
    AS_ROOT='sudo '
else
    NO_ROOT_WHY='sudo was not found'
fi

# How a sentence hands a command that needs root to the owner: `with: sudo …` where that is the way in,
# `as root with: …` where this run cannot elevate and the owner must run it themselves.
AS_ROOT_SAYS='with: '
[ -n "$PRIVILEGED" ] || AS_ROOT_SAYS='as root with: '

# Run a command as root: directly when this run is root already, through sudo when that is the way in.
as_root() {
    if [ -n "$AS_ROOT" ]; then
        sudo "$@"
    else
        "$@"
    fi
}

# Put back a unit this run stopped to replace its file, whichever step then failed: it starts the
# file that is now in place, and is a no-op when this install stopped nothing. Defined before `fail`,
# and read with `:-`, because a refusal from before the variable is set reaches it too.
restore_stopped_unit() {
    [ -n "${UNIT_STOPPED:-}" ] || return 0
    as_root systemctl start "$UNIT_NAME" >/dev/null 2>&1 || true
}
fail() {
    restore_stopped_unit
    printf '%s\n' "$1" >&2
    exit 1
}

# What this run has to undo when it is cut short: the temporary directory and the staged file it may
# have created, and the service it may have stopped. Armed here, before anything on the host is
# touched, because an interrupt between the stop and the rest of the run would otherwise leave the
# owner's product down — the one thing this install never does.
cleanup() {
    if [ -n "${STAGED:-}" ]; then
        rm -f "$STAGED"
        STAGED=''
    fi
    if [ -n "${TMP:-}" ]; then
        rm -rf "$TMP"
        TMP=''
    fi
}
trap cleanup EXIT
trap 'cleanup; restore_stopped_unit; exit 1' HUP INT TERM

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

# The version this run was told to install — the script's own argument, or MAHBOT_INSTALL_VERSION for
# the documented pipe — with the release mark's leading `v` dropped. Empty means the newest release,
# which only the run that installs looks up: the dry run names it without a download.
version_named() {
    printf '%s\n' "${1:-${MAHBOT_INSTALL_VERSION:-}}" | sed 's/^v//'
}

# Whether the product's own lock file is provably held, and only then: `flock` refused without a
# word about it is the product holding it, a `flock` that could not run at all is no evidence, and
# `lsof` naming a process with the file open is evidence too. With neither tool able to answer
# nothing is refused on a guess.
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

# The processes holding `$1` open, space-separated, so the refusal can name the exact `kill`: `lsof`
# on both platforms, `/proc` on Linux when it is missing.
lock_holders() {
    if command -v lsof >/dev/null 2>&1; then
        PIDS=$(lsof -t -- "$1" 2>/dev/null | tr '\n' ' ' || true)
        printf '%s' "${PIDS% }"
        return 0
    fi
    if [ -d /proc ]; then
        PIDS=''
        for dir in /proc/[0-9]*; do
            for fd in "$dir"/fd/*; do
                if [ "$(readlink "$fd" 2>/dev/null)" = "$1" ]; then
                    PIDS="$PIDS ${dir#/proc/}"
                    break
                fi
            done
        done
        printf '%s' "${PIDS# }"
    fi
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

# Whether the startup file already holds the block the product counts as its own
# (`crate::util::owner_path::block_regions`, `is_product_body`): a start marker, exactly one body
# line of the shape it writes, an end marker. Anything else — no body line, another shape, a second
# line — is a block the product would never take out, so this writes its own beside it.
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

# ── The registration: evidence, the unit, and the service manager ─────────

# Whether a display manager is present: the alias systemd's own documentation gives one
# (`display-manager.service`, which a package aliases and `systemctl enable` links), or a unit file
# the packaging of a display manager installs. This — never the default target — is the evidence
# that a graphical login exists for the product's login entry to run in.
have_display_manager() {
    for unit in display-manager.service gdm.service gdm3.service lightdm.service sddm.service \
        xdm.service lxdm.service ly.service greetd.service slim.service nodm.service \
        plasmalogin.service; do
        for dir in /etc/systemd/system /usr/local/lib/systemd/system /usr/lib/systemd/system \
            /lib/systemd/system; do
            if [ -e "$dir/$unit" ]; then
                DISPLAY_MANAGER="$dir/$unit"
                return 0
            fi
        done
    done
    return 1
}

# Whether the service registration is one this host can take at all: a Linux host running systemd,
# and a way to put a unit under `/etc`. `SERVICE_UNAVAILABLE` names the missing piece when it cannot,
# so the explicit ask and the switch line read from one predicate.
can_register_service() {
    SERVICE_UNAVAILABLE=''
    if [ "$OS" != linux ]; then
        SERVICE_UNAVAILABLE="this platform has no systemd unit to register"
    elif [ -z "$HAVE_SYSTEMD" ]; then
        SERVICE_UNAVAILABLE="no service manager runs on this host"
    elif [ -z "$PRIVILEGED" ]; then
        SERVICE_UNAVAILABLE="$NO_ROOT_WHY"
    fi
    [ -z "$SERVICE_UNAVAILABLE" ]
}

# Whether the unit file `$1` is one systemd will load, with the first line of its refusal in
# `UNIT_CHECK_SAID` when it is not. `systemd-analyze verify` is the service manager's own reading of
# the file, so what it refuses is what the manager would refuse.
unit_loads() {
    UNIT_CHECK_SAID=''
    command -v systemd-analyze >/dev/null 2>&1 || return 0
    if UNIT_CHECK_SAID=$(systemd-analyze verify "$1" 2>&1); then
        UNIT_CHECK_SAID=''
        return 0
    fi
    UNIT_CHECK_SAID=$(printf '%s\n' "$UNIT_CHECK_SAID" | awk 'NF { print; exit }')
    return 1
}

# The service manager's own word for a state it reported — `is-enabled` for a unit's registration,
# `ActiveState` for its run — and `unknown` when it said nothing at all: every message that reports a
# state prints this, never a paraphrase and never an empty pair of quotes.
unit_state_word() {
    printf '%s' "${1:-unknown}"
}

# The one command that takes off the mask a state names: a persistent mask is a link beside the unit,
# a runtime one lives under `/run`, where only `--runtime` removes it, a host with no service manager
# to ask is told to remove the link itself, and a link to nothing — which the manager calls
# `not-found`, and `systemctl unmask` leaves where it is — is removed as the file it is. Nothing for a
# unit that is not masked. One place, so a message naming the mask and the command in front of
# another cannot disagree about its form.
unmask_command() {
    case "${1:-}" in
        masked)
            if [ -n "$HAVE_SYSTEMD" ]; then
                printf '%s' "${AS_ROOT}systemctl unmask $UNIT_NAME"
            else
                printf '%s' "${AS_ROOT}rm -f $UNIT_PATH"
            fi
            ;;
        masked-runtime) printf '%s' "${AS_ROOT}systemctl unmask --runtime $UNIT_NAME" ;;
        severed) printf '%s' "${AS_ROOT}rm -f $UNIT_PATH" ;;
    esac
}

# The same command in front of one that has to reach the unit, `&&`-joined: nothing at all for a unit
# that is not masked.
unmask_prefix() {
    _unmask=$(unmask_command "${1:-}")
    [ -z "$_unmask" ] || printf '%s' "$_unmask && "
}

# The mask this host carries, in the service manager's own word where it reports one and `severed`
# where the unit's place is a link to nothing — a state the manager calls `not-found`, and one
# `systemctl unmask` leaves where it is. Empty for a host with no mask.
unit_mask_form() {
    case "${UNIT_ENABLED_BEFORE:-}" in
        masked | masked-runtime) printf '%s' "$UNIT_ENABLED_BEFORE" ;;
        *)
            if [ -L "$UNIT_PATH" ] && [ ! -e "$UNIT_PATH" ]; then
                printf '%s' severed
            fi
            ;;
    esac
}

# Whether the host's unit is masked — the owner's decision that nothing starts it. The product reads
# the same states as a mask (`src/autostart.rs`), so the two cannot disagree.
unit_masked() {
    [ -n "$(unit_mask_form)" ]
}

# The one command that changes the answer for the two states where a command does — a mask removed and
# the enable that follows, and a disabled unit enabled — and nothing for every other state, where no
# command would change it. Printed only by a message that reports the state.
unit_remedy() {
    case "${1:-}" in
        masked | masked-runtime) printf '%s' "$(unmask_prefix "$1")${AS_ROOT}systemctl enable --now $UNIT_NAME" ;;
        disabled) printf '%s' "${AS_ROOT}systemctl enable --now $UNIT_NAME" ;;
        *) return 1 ;;
    esac
}

# The unit as systemd reads it. `Environment=` and `ExecStart=` are quoted because the manager
# strips quotes there; `WorkingDirectory=` is not, because it does not — a quoted value is not an
# absolute path to it, and it refuses the whole unit. `KillMode=mixed` sends SIGTERM to the main
# process alone, which is what lets the drain it runs finish the shells it is waiting on.
# `TimeoutStopSec` is the product's own drain cap (`jobs::DRAIN_CAP_SECS`) plus an allowance for the
# closing steps after it; a closing path slower than that is ended by the manager, which is the
# accepted limit of a hard stop.
render_unit() {
    cat <<EOF
[Unit]
Description=MahBot
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
User=$(id -un)
Environment="HOME=$HOME"
WorkingDirectory=$HOME
ExecStart="$DEST"
Restart=on-failure
RestartSec=5
KillMode=mixed
TimeoutStopSec=$STOP_TIMEOUT_SECS

[Install]
WantedBy=multi-user.target
EOF
}

unit_by_hand() {
    say "Save this as $UNIT_PATH ${AS_ROOT_SAYS}${AS_ROOT}tee $UNIT_PATH, then run:"
    say "  ${AS_ROOT}systemctl daemon-reload && ${AS_ROOT}systemctl enable --now $UNIT_NAME"
    render_unit
}

no_registration() {
    warn "MahBot was not registered as a service: $1."
    unit_by_hand
    fail "This install is not finished: MahBot was not registered to start at boot on this host."
}

login_entry_on_disk() {
    for entry in "$AUTOSTART_ENTRY" "$AUTOSTART_ENTRY_HOME"; do
        if [ -e "$entry" ]; then
            printf '%s\n' "$entry"
            return 0
        fi
    done
    return 1
}

remove_login_entries() {
    for entry in "$AUTOSTART_ENTRY" "$AUTOSTART_ENTRY_HOME"; do
        [ -e "$entry" ] || continue
        rm -f "$entry" 2>/dev/null ||
            fail "the login entry $entry could not be removed, so this host would keep two registrations; remove it as its owner."
    done
}

# Whether the unit has settled: `active` with no restart for longer than its own `RestartSec`, so a
# service that starts and dies on its feet has already been restarted — or given up on — inside the
# window. `Type=simple` reports success the moment the process is forked, which is why the state and
# the restart count are read rather than the start command's exit status. `UNIT_STATE` carries what
# the manager said when this returns false.
wait_for_unit() {
    SETTLE_WAITED=0
    SETTLE_STABLE=0
    SETTLE_RESTARTS=''
    UNIT_STATE=''
    while [ "$SETTLE_WAITED" -lt "$SETTLE_DEADLINE_SECS" ]; do
        UNIT_STATE=$(systemctl show -p ActiveState "$UNIT_NAME" 2>/dev/null | sed 's/^ActiveState=//')
        RESTARTS=$(systemctl show -p NRestarts "$UNIT_NAME" 2>/dev/null | sed 's/^NRestarts=//')
        case "$UNIT_STATE" in
            active)
                if [ "$RESTARTS" = "$SETTLE_RESTARTS" ]; then
                    SETTLE_STABLE=$((SETTLE_STABLE + 1))
                    if [ "$SETTLE_STABLE" -ge "$SETTLED_SECS" ]; then
                        return 0
                    fi
                else
                    SETTLE_RESTARTS=$RESTARTS
                    SETTLE_STABLE=0
                fi
                ;;
            failed | inactive | deactivating)
                return 1
                ;;
            *)
                SETTLE_RESTARTS=$RESTARTS
                SETTLE_STABLE=0
                ;;
        esac
        sleep 1
        SETTLE_WAITED=$((SETTLE_WAITED + 1))
    done
    return 1
}

# Start again the run this install stopped to replace the file, and say plainly when it does not come
# back: the host is never left worse than this install found it. Returns 1 when this install stopped
# nothing, so the caller says what it left alone instead.
put_back_stopped_run() {
    [ -n "$UNIT_STOPPED" ] || return 1
    restore_stopped_unit
    if ! wait_for_unit; then
        START_HOW="$(unmask_prefix "$UNIT_ENABLED_BEFORE")${AS_ROOT}systemctl start $UNIT_NAME"
        warn "MahBot was stopped to replace its file and is not running (the service manager reports '$(unit_state_word "$UNIT_STATE")'). Start it ${AS_ROOT_SAYS}$START_HOW"
        fail "This install is not finished: the host's service was stopped for the file swap and is not running."
    fi
    say "It was stopped to replace its file and is running again."
}

# The registration this install will use and the one line saying why — one place, so the read-only
# mode and the run that changes the host cannot disagree.
choose_registration() {
    REGISTRATION=''
    REGISTRATION_WHY=''
    UNIT_KEPT_AT=''
    UNIT_KEPT_WHY=''
    UNIT_REASON=''
    UNIT_ENABLED_BEFORE=''
    DISPLAY_MANAGER=''
    if [ "$OS" = linux ]; then
        have_display_manager || true
    fi
    if [ "$OS" = linux ] && [ -n "$HAVE_SYSTEMD" ]; then
        # `is-enabled` first: a masked unit is a link to `/dev/null` (or one under `/run`), so there is
        # no unit text to read, and a mask is the owner's decision rather than a unit that would not
        # load. `UNIT_KEPT_AT` is the path that actually holds the registration — a runtime mask lives
        # under `/run`, not beside the unit — and is the one flag saying the host has one.
        UNIT_ENABLED_BEFORE=$(systemctl is-enabled "$UNIT_NAME" 2>/dev/null || true)
        case "$UNIT_ENABLED_BEFORE" in
            masked)
                UNIT_KEPT_AT=$UNIT_PATH
                UNIT_KEPT_WHY="the owner masked"
                ;;
            masked-runtime)
                UNIT_KEPT_AT="/run/systemd/system/$UNIT_NAME"
                UNIT_KEPT_WHY="the owner masked for this boot only"
                ;;
            *)
                if [ -e "$UNIT_PATH" ] || [ -L "$UNIT_PATH" ]; then
                    if unit_loads "$UNIT_PATH"; then
                        UNIT_KEPT_AT=$UNIT_PATH
                        UNIT_KEPT_WHY="loads"
                    else
                        UNIT_REASON=$UNIT_CHECK_SAID
                    fi
                fi
                ;;
        esac
    elif [ "$OS" = linux ] && [ -L "$UNIT_PATH" ] && [ "$(readlink "$UNIT_PATH")" = /dev/null ]; then
        # No manager to ask, and the link is the owner's mask all the same. The state is recorded so
        # that nothing below can mistake a mask for a unit file this install may remove; it is not a
        # registration, because nothing on this host starts the product, so `UNIT_KEPT_AT` stays empty.
        UNIT_ENABLED_BEFORE=masked
    fi
    case "$REGISTRATION_ASKED" in
        '') ;;
        login)
            REGISTRATION=login
            REGISTRATION_WHY="asked for through MAHBOT_INSTALL_REGISTRATION"
            ;;
        service)
            # An ask this host cannot satisfy is refused before anything is downloaded, rather than
            # after the binary is in place and the registration fails — and never answered silently.
            # A unit the host already has is answered first, so its owner is told about the unit
            # rather than about installing one, and the ask keeps refusing every unit that is not
            # enabled: the two states a command changes get that command, the rest get the state
            # itself, because for those there is no command that would change the answer.
            if [ -n "$UNIT_KEPT_AT" ] && [ "$UNIT_ENABLED_BEFORE" != enabled ]; then
                UNIT_SAID=$(unit_state_word "$UNIT_ENABLED_BEFORE")
                if REMEDY=$(unit_remedy "$UNIT_ENABLED_BEFORE"); then
                    fail "$SERVICE_ASKED the unit at $UNIT_KEPT_AT is '$UNIT_SAID' on this host, so it does not start at boot; nothing was installed. Start it at boot ${AS_ROOT_SAYS}$REMEDY"
                fi
                fail "$SERVICE_ASKED the unit at $UNIT_KEPT_AT is '$UNIT_SAID' on this host; nothing was installed, and this install did not start or enable it."
            fi
            if ! can_register_service; then
                if [ -n "$UNIT_KEPT_AT" ]; then
                    fail "$SERVICE_ASKED the unit at $UNIT_KEPT_AT is installed on this host, and this install cannot replace its file and restart it without root ($SERVICE_UNAVAILABLE); nothing was installed. Re-run this install as root."
                fi
                fail "$SERVICE_ASKED $SERVICE_UNAVAILABLE; nothing was installed."
            fi
            REGISTRATION=service
            REGISTRATION_WHY="asked for through MAHBOT_INSTALL_REGISTRATION"
            ;;
        *)
            fail "MAHBOT_INSTALL_REGISTRATION must name one of 'service' and 'login', not '$REGISTRATION_ASKED', so nothing was installed."
            ;;
    esac
    if [ -z "$REGISTRATION" ] && [ -n "$UNIT_KEPT_AT" ]; then
        REGISTRATION=service
        REGISTRATION_WHY="the unit at $UNIT_KEPT_AT, which $UNIT_KEPT_WHY"
    fi
    if [ -z "$REGISTRATION" ] && [ "$OS" != linux ]; then
        REGISTRATION=login
        REGISTRATION_WHY="the only registration this platform has"
    fi
    if [ -z "$REGISTRATION" ] && [ -z "$HAVE_SYSTEMD" ]; then
        REGISTRATION=login
        REGISTRATION_WHY="no service manager to register a unit with"
    fi
    if [ -z "$REGISTRATION" ] && [ -n "$DISPLAY_MANAGER" ]; then
        REGISTRATION=login
        REGISTRATION_WHY="the display manager at $DISPLAY_MANAGER"
    fi
    if [ -z "$REGISTRATION" ]; then
        REGISTRATION=service
        REGISTRATION_WHY="no display manager was found"
    fi
}

# The one line that names the other registration and the switch that asks for it: the environment
# the install is run with, never the systemd default target, which decides nothing here. Either
# direction is offered only where this host can carry it out, so no host is told to re-run into a
# refusal — a masked unit refuses both, and the run has already named what undoes it.
registration_switch_line() {
    if unit_masked; then
        return 0
    fi
    if [ "$REGISTRATION" = service ]; then
        # Login means no unit on this host, and taking one off needs root.
        if { [ -e "$UNIT_PATH" ] || [ -L "$UNIT_PATH" ]; } && [ -z "$PRIVILEGED" ]; then
            return 0
        fi
        say "To start MahBot at login instead, re-run this install with MAHBOT_INSTALL_REGISTRATION=login in front of sh."
    elif can_register_service &&
        { [ -z "$UNIT_KEPT_AT" ] || [ "$UNIT_ENABLED_BEFORE" = enabled ]; }; then
        # The ask is offered only where the follow-up would honour it: a unit the owner disabled or
        # masked would make the service registration refuse.
        say "To run MahBot as a service that starts at boot instead, re-run this install with MAHBOT_INSTALL_REGISTRATION=service in front of sh."
    fi
}

# A unit the host already has is its registration, and this install never replaces it. Three
# outcomes: in service, so the new file is put under it and it is restarted; not in service, so it is
# left exactly as it is in one sentence; and, on either path, a run this install stopped for the file
# swap is started again.
keep_existing_unit() {
    REGISTERED="MahBot is already registered as a system service at $UNIT_KEPT_AT"
    if [ "$UNIT_ENABLED_BEFORE" = enabled ]; then
        if [ -n "$PRIVILEGED" ]; then
            as_root systemctl daemon-reload >/dev/null 2>&1 || true
            # The restart's own failure and a restart that does not settle are told apart: a word the
            # manager was never asked for is never put in its mouth.
            STARTED=''
            if ! as_root systemctl restart "$UNIT_NAME" 2>/dev/null; then
                warn "$REGISTERED, but it could not be restarted on the new file; see: systemctl status $UNIT_NAME"
            elif wait_for_unit; then
                STARTED=1
                say "$REGISTERED; this install left the registration as it is and brought it up on the new file."
            else
                warn "$REGISTERED, but it did not come up under it (the service manager reports '$(unit_state_word "$UNIT_STATE")'); see: systemctl status $UNIT_NAME"
            fi
            if [ -z "$STARTED" ]; then
                say "Start it ${AS_ROOT_SAYS}${AS_ROOT}systemctl start $UNIT_NAME"
                fail "This install is not finished: the host's service is in place but nothing is running."
            fi
        else
            say "$REGISTERED and is enabled on this host, but this install cannot restart it without root; restart it ${AS_ROOT_SAYS}${AS_ROOT}systemctl restart $UNIT_NAME"
        fi
    elif ! put_back_stopped_run; then
        UNIT_SAID=$(unit_state_word "$UNIT_ENABLED_BEFORE")
        if REMEDY=$(unit_remedy "$UNIT_ENABLED_BEFORE"); then
            say "$REGISTERED and is '$UNIT_SAID' on this host, so it does not start at boot; this install left it as it is. Start it at boot ${AS_ROOT_SAYS}$REMEDY"
        else
            say "$REGISTERED and is '$UNIT_SAID' on this host; this install left it as it is and did not start or enable it."
        fi
    fi
    # The unit is the registration in use whatever state it is in, so the entry that would start a
    # second copy goes — it is the owner's own file and needs no root.
    remove_login_entries
    registration_switch_line
}

# ── 1. This host, and where it starts MahBot ──────────────────────────────

[ -n "${HOME:-}" ] ||
    fail "HOME is not set, so there is no place to install MahBot into."
INSTALL_DIR="$HOME/.local/bin"
DEST="$INSTALL_DIR/mahbot"

# The two registrations a host can have: the system unit this script writes on a Linux server,
# and the login entry the product writes everywhere else — on Linux the XDG desktop entry below
# (macOS writes a LaunchAgent plist and Windows a registry value, neither of which can be paired
# with a systemd unit). The unit's path is the product's own (`src/autostart.rs`), which reads it
# to know the installer made the choice.
UNIT_NAME=mahbot.service
UNIT_PATH="/etc/systemd/system/$UNIT_NAME"
# The entry's directory is the product's own (`directories::BaseDirs::config_dir`, read by
# `src/autostart.rs`): `$XDG_CONFIG_HOME` when that is absolute, `$HOME/.config` otherwise — a
# relative value is ignored there, so it is ignored here, and every trailing separator is dropped there,
# so it is dropped here too and the entry joins onto what is left with one separator. The home path is
# what a service start computes, and the one an earlier release wrote whatever this shell holds: both
# are removed when the unit goes into service, or a host could keep two registrations.
AUTOSTART_DIR="$HOME/.config"
case "${XDG_CONFIG_HOME:-}" in
    /*)
        AUTOSTART_DIR=$XDG_CONFIG_HOME
        while [ "${AUTOSTART_DIR%/}" != "$AUTOSTART_DIR" ]; do
            AUTOSTART_DIR=${AUTOSTART_DIR%/}
        done
        ;;
esac
AUTOSTART_ENTRY="$AUTOSTART_DIR/autostart/mahbot.desktop"
AUTOSTART_ENTRY_HOME="$HOME/.config/autostart/mahbot.desktop"
# What this run has done to the host's service, for the paths that must put it back: `UNIT_STOPPED`
# when the lock block stopped it, `UNIT_STATE` as the manager last reported it.
UNIT_STOPPED=''
UNIT_STATE=''

# The unit's stop timeout: the product's own drain cap (`jobs::DRAIN_CAP_SECS`) plus an allowance
# for the closing steps after it. One variable, so the unit and the sentence printed while the
# service stops cannot disagree.
DRAIN_CAP_SECS=600
STOP_TIMEOUT_SECS=$((DRAIN_CAP_SECS + 300))

# How the liveness proof reads the service manager: a unit that stays active this long without a
# restart has come up and stayed up — longer than the unit's own `RestartSec`, so a service that
# dies on its feet has already been restarted, or given up on, inside the window. The deadline is
# how long the manager gets to settle on a state before the install reports a failure.
SETTLED_SECS=10
SETTLE_DEADLINE_SECS=60

# The registration asked for in this run's environment, and the read-only mode: both arrive this
# way because the documented install is a pipe (`curl … | sh`), where the script's own arguments
# are not reachable. A value that is neither empty nor the documented one is refused, so a mistyped
# `MAHBOT_INSTALL_DRY_RUN=yes` cannot quietly become a real install.
REGISTRATION_ASKED=${MAHBOT_INSTALL_REGISTRATION:-}
DRY_RUN=${MAHBOT_INSTALL_DRY_RUN:-}
case "$DRY_RUN" in
    '' | 1) ;;
    *)
        fail "MAHBOT_INSTALL_DRY_RUN must be 1 to ask for a dry run, not '$DRY_RUN', so nothing was installed."
        ;;
esac

# The host's platform, read once: everything below — systemd, the login entry, the floors, the
# file this install fetches — is decided from it.
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

# Whether systemd runs this host. A `systemctl` binary alone is not the answer: it exists
# inside containers that run no systemd, where a unit would never be started — the running
# manager's own directory is what says otherwise.
HAVE_SYSTEMD=''
if [ "$OS" = linux ] && [ -d /run/systemd/system ] &&
    command -v systemctl >/dev/null 2>&1; then
    HAVE_SYSTEMD=1
fi

# The explicit ask's refusals, so the three of them read as one sentence.
SERVICE_ASKED="MAHBOT_INSTALL_REGISTRATION=service was asked for, but"

choose_registration

# ── 2. What this run would do, when it is asked only to say ───────────────

# Read-only by contract: the decision, the registration and the unit the run would write, with no
# download, no sudo and no write anywhere. It answers the question a server owner actually has —
# which registration this host would get, and why — without needing the release to exist.
if [ "$DRY_RUN" = 1 ]; then
    say "MahBot install, dry run: nothing on this machine is changed."
    say "  system:       $OS $ARCH"
    say "  installs to:  $DEST"
    NAMED=$(version_named "${1:-}")
    if [ -n "$NAMED" ]; then
        say "  version:      $NAMED"
    else
        say "  version:      the newest release at $RELEASE_BASE"
    fi
    if [ -n "$UNIT_KEPT_AT" ]; then
        say "  registered:   the unit at $UNIT_KEPT_AT, '$(unit_state_word "$UNIT_ENABLED_BEFORE")'"
    elif unit_masked; then
        # A mask the manager does not report as one (a link to nothing) or a host with no manager to
        # ask: the owner's decision either way, and this run leaves it in place — so it is named here
        # rather than reported as a unit that would not load or as no registration at all.
        say "  registered:   nothing that starts: the unit at $UNIT_PATH is masked"
    elif [ -n "$UNIT_REASON" ]; then
        say "  registered:   nothing that works: the unit at $UNIT_PATH would not load ($UNIT_REASON)"
    elif ENTRY=$(login_entry_on_disk); then
        say "  registered:   the login entry at $ENTRY"
    else
        say "  registered:   nothing yet"
    fi
    say "  registration: $REGISTRATION ($REGISTRATION_WHY)"
    if [ "$REGISTRATION" = service ]; then
        if [ -n "$UNIT_KEPT_AT" ]; then
            say "  unit:         $UNIT_KEPT_AT is kept as it is; this run would write no unit"
        elif ! can_register_service; then
            # The run would refuse, so the dry run says that instead of a unit it would not write.
            say "  unit:         none: the run would refuse, because $SERVICE_UNAVAILABLE"
            exit 1
        else
            say "  unit path:    $UNIT_PATH"
            say ""
            render_unit
        fi
    elif [ "$OS" = linux ]; then
        if unit_masked; then
            # The product writes no entry beside a mask, so the dry run must not promise one.
            say "  login entry:  $AUTOSTART_ENTRY (not written while the unit is masked)"
        else
            say "  login entry:  $AUTOSTART_ENTRY (MahBot writes it at its first start)"
        fi
    else
        say "  login start:  MahBot's own login registration, which it writes at its first start"
    fi
    say ""
    registration_switch_line
    exit 0
fi

# ── 3. Take over from a copy already running ──────────────────────────────

# Scoped to the data directory this home's product uses, so a product running
# under another home is not this script's business.
if [ -e "$HOME/.mahbot/mahbot.lock" ] && lock_held "$HOME/.mahbot/mahbot.lock"; then
    # A server's unit owns the copy holding the lock, and that copy is exactly what this
    # install replaces — so it is stopped, not refused. `is-active` answers without root; stopping
    # needs it, and a host without root is told the one command that does it rather than being sent
    # after a pid that belongs to its own service.
    if [ -n "$HAVE_SYSTEMD" ] && systemctl is-active --quiet "$UNIT_NAME" 2>/dev/null; then
        if [ -z "$PRIVILEGED" ]; then
            fail "MahBot is running as the system service $UNIT_NAME, and this install cannot stop it without root, so nothing was installed. Stop it ${AS_ROOT_SAYS}systemctl stop $UNIT_NAME, then run this installer again."
        fi
        say "Stopping the running MahBot service so its file can be replaced. Its in-flight work is drained first, so this can take up to $STOP_TIMEOUT_SECS seconds — the unit's own stop timeout."
        # Recorded before the stop returns: the manager waits out the drain above, and an interrupt
        # inside that window must still put the service back. Starting a service that never stopped
        # is harmless.
        UNIT_STOPPED=1
        as_root systemctl stop "$UNIT_NAME" 2>/dev/null ||
            fail "the running MahBot service could not be stopped, so nothing was installed."
    else
        # A copy no unit of this host owns: what an install that never registered a service left
        # running, or one started by hand. It holds the instance lock, so the file cannot be replaced
        # under it — and on the hosts the shipped releases broke it is all that stands between them
        # and the fix. The command that clears it is named, with the pid when a tool can say which.
        LEFTOVER=$(lock_holders "$HOME/.mahbot/mahbot.lock")
        if [ -n "$LEFTOVER" ]; then
            fail "MahBot is already running as a process no service unit of this host owns — a copy an earlier install left behind, or one started by hand — so nothing was installed. Stop it with: kill $LEFTOVER, then run this installer again."
        fi
        fail "MahBot is already running as a process no service unit of this host owns, so nothing was installed. Stop the process holding $HOME/.mahbot/mahbot.lock, then run this installer again."
    fi
fi

# ── 4. The floors the published files are built against ───────────────────

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

# ── 5. The version to install ─────────────────────────────────────────────

# The version file is fetched, and a named version is still downloaded, so
# nothing can go on without one of the two tools.
[ -n "$CURL$WGET" ] ||
    fail "neither curl nor wget is available to download MahBot, so nothing was installed."

VERSION=$(version_named "${1:-}")
if [ -z "$VERSION" ]; then
    # The newest release there is: the host's own newest-release file, which names
    # the newest release that is not a test release — and nothing at all while only
    # test releases exist, where one is installed by naming its version instead.
    RAW=$(fetch "$RELEASE_BASE/releases/latest/download/version.txt" -) ||
        fail "the newest MahBot release could not be looked up, so nothing was installed (install.sh <version>, or MAHBOT_INSTALL_VERSION, installs one particular release)."
    # The file's name and the address it is fetched from are built from the version itself, never
    # from the release mark's own spelling.
    VERSION=$(printf '%s' "$RAW" | tr -d '\r\n')
    VERSION=${VERSION#v}
fi
[ -n "$VERSION" ] ||
    fail "the MahBot release to install could not be determined, so nothing was installed."

# A pre-release part says plainly what it is: such a release is not what the
# owner is given normally.
case "$VERSION" in
    *-*)
        say "The MahBot $VERSION release is a test release, not a normal one; installing it."
        ;;
esac

# ── 6. Download the file and put it in place ──────────────────────────────

# Nothing here prints anything of its own: the sentence after each step is the
# whole of what the owner is told, so the tool's own line is dropped rather than
# left beside it.
ASSET="mahbot-$VERSION-$OS-$ARCH.tar.gz"
TMP=$(mktemp -d 2>/dev/null) ||
    fail "a temporary directory could not be created, so nothing was installed."
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

# ── 7. Make the location visible ──────────────────────────────────────────

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

# ── 8. Remove the copy the old way of installing left behind ──────────────

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

# ── 9. Give the speech models their machine-wide home ─────────────────────

# Both speech engines read one fixed folder per system and never create it: a
# missing one is an error naming the path, never a substitute location. The folder
# is outside what an ordinary user may create on Linux, so that one is created with
# elevation, once, and handed to the account the product runs under; on macOS
# `/Users/Shared` is world-writable and the folder is created as the owner.
#
# A failure here is said plainly and does not fail the install: the product still
# runs, and only speech is refused (naming the path) until the folder exists.
if [ "$OS" = macos ]; then
    MODELS_DIR='/Users/Shared/MahBot/models'
    if [ ! -d "$MODELS_DIR" ] && ! mkdir -p "$MODELS_DIR" 2>/dev/null; then
        warn "the speech model folder $MODELS_DIR could not be created; speech will refuse to run until it exists."
    fi
else
    MODELS_DIR='/var/lib/mahbot/models'
    # Absent, or there and not this account's to write — an earlier install by another
    # account leaves it that way — is the same one elevated step.
    if [ -d "$MODELS_DIR" ] && [ -w "$MODELS_DIR" ]; then
        :
    elif [ -z "$PRIVILEGED" ]; then
        warn "$MODELS_DIR could not be created without root ($NO_ROOT_WHY); speech will refuse to run until it exists."
    else
        if [ -n "$AS_ROOT" ]; then
            say "Creating $MODELS_DIR (your administrator password is needed once)."
        fi
        if ! as_root mkdir -p "$MODELS_DIR" 2>/dev/null ||
            ! as_root chown "$(id -u):$(id -g)" "$MODELS_DIR" 2>/dev/null; then
            warn "$MODELS_DIR could not be created and given to this account; speech will refuse to run until it exists."
        fi
    fi
fi

# ── 10. What MahBot needs from this host ──────────────────────────────────

# Said once, on every host: these hold however the product is started. The audio library is
# named apart from them because only voice needs it — a server with no sound card runs
# everything else.
if [ "$OS" = macos ]; then
    OS_FLOOR="macOS $MACOS_FLOOR_MAJOR.$MACOS_FLOOR_MINOR or newer"
else
    OS_FLOOR="glibc $GLIBC_FLOOR_MAJOR.$GLIBC_FLOOR_MINOR or newer"
fi
say ""
say "MahBot needs from this host:"
say "  - $OS_FLOOR"
say "  - a local filesystem for $HOME/.mahbot; network storage is not supported"
say "  - one operating-system user per machine"
say "  - no inbound network ports: MahBot talks out, and Telegram delivers its messages"
say "  - for voice only, the audio library (libasound2 on Linux) — the service does not need it"

# ── 11. Register the service, or start at login ───────────────────────────

if [ "$REGISTRATION" = service ]; then
    # A unit the host already has is the host's registration: nothing is replaced, and nothing is
    # asked of a host whose owner cannot act on it. The rest of this section writes a new unit, which
    # needs root.
    if [ -n "$UNIT_KEPT_AT" ]; then
        keep_existing_unit
        exit 0
    fi
    if [ -z "$PRIVILEGED" ]; then
        no_registration "this install cannot write a unit without root ($NO_ROOT_WHY)"
    fi

    # Rendered and checked before anything on the host is replaced: a unit the manager would refuse
    # must never be left behind, and the registration the host already had — a working unit, or the
    # product's own login entry — stays exactly as it is when the check says no.
    render_unit >"$TMP/$UNIT_NAME" 2>/dev/null ||
        fail "the service unit could not be prepared, so nothing was registered."
    if ! unit_loads "$TMP/$UNIT_NAME"; then
        fail "the service unit this install would write would not load: $UNIT_CHECK_SAID; nothing was registered and $UNIT_PATH was not changed."
    fi
    if [ -n "$UNIT_REASON" ]; then
        say "The unit already at $UNIT_PATH would not load ($UNIT_REASON), so it is replaced by one that does."
    fi

    say "Registering MahBot as a system service ($REGISTRATION_WHY)."
    # A root install is legitimate — a bare server, a container — but the unit then runs the product
    # as root, which is not the shape it is meant for.
    if [ -n "$IS_ROOT" ]; then
        warn "This install runs as root, so the service will run as root as well; installing as an ordinary user is the recommended way."
    fi
    if ! as_root tee "$UNIT_PATH" <"$TMP/$UNIT_NAME" >/dev/null 2>&1; then
        no_registration "$UNIT_PATH could not be written"
    fi
    # The file on disk is checked too, not only the text that was prepared: what is enabled below is
    # what the manager will read at boot, and a write that landed wrong must not survive as a unit.
    if ! unit_loads "$UNIT_PATH"; then
        as_root rm -f "$UNIT_PATH" 2>/dev/null || true
        fail "the unit written to $UNIT_PATH would not load: $UNIT_CHECK_SAID; it was removed again and nothing is registered."
    fi

    if [ "$UNIT_ENABLED_BEFORE" = disabled ]; then
        # The owner disabled the unit an earlier install left, and this install does not undo that —
        # the command that would is the remedy its state earns, and it is named, not run.
        as_root systemctl daemon-reload >/dev/null 2>&1 || true
        REMEDY=$(unit_remedy disabled)
        say "The service unit is in place but is 'disabled' on this host, so this install did not enable it. Start it at boot ${AS_ROOT_SAYS}$REMEDY"
        put_back_stopped_run || say "Start it ${AS_ROOT_SAYS}${AS_ROOT}systemctl start $UNIT_NAME"
        remove_login_entries
        registration_switch_line
        exit 0
    fi
    as_root systemctl daemon-reload >/dev/null 2>&1 || true
    as_root systemctl enable --now "$UNIT_NAME" >/dev/null 2>&1 || true
    # The proof is the manager's own settled state, read whether or not the commands above succeeded,
    # so a failure names what the manager reports rather than nothing.
    if ! wait_for_unit; then
        warn "$UNIT_PATH was written and loads, but MahBot did not come up under it (the service manager reports '$(unit_state_word "$UNIT_STATE")'); see: systemctl status $UNIT_NAME"
        say "Start it ${AS_ROOT_SAYS}${AS_ROOT}systemctl daemon-reload && ${AS_ROOT}systemctl enable --now $UNIT_NAME"
        fail "This install is not finished: the unit is in place but nothing is running."
    fi
    UNIT_SAID=$(unit_state_word "$(systemctl is-enabled "$UNIT_NAME" 2>/dev/null || true)")
    if [ "$UNIT_SAID" != enabled ]; then
        # The state is the manager's own word, so a host that starts the unit some other way is not
        # told it starts nothing; only the two states a command changes get that command.
        warn "$UNIT_PATH was written and loads, and MahBot is running under it, but the service manager reports it as '$UNIT_SAID', not 'enabled'."
        if REMEDY=$(unit_remedy "$UNIT_SAID"); then
            say "Start it at boot ${AS_ROOT_SAYS}$REMEDY"
        fi
        fail "This install is not finished: the unit is in place but the service manager does not report it as enabled."
    fi

    # The unit is in service, so the login entry would start a second copy of the product: it goes.
    remove_login_entries

    say "MahBot is registered as a system service at $UNIT_PATH and starts at boot; it was running when this install finished."
    say "See it with: systemctl status $UNIT_NAME"
    registration_switch_line
    exit 0
fi

# ── 12. Start at login ────────────────────────────────────────────────────

# A graphical host starts the product at login, and the entry that does it is the product's own
# (`src/autostart.rs`): this install writes none and reports where it goes. A mask starts nothing —
# it is the owner's decision, so it is left where it is and named with the manager's own word where it
# reports one and the path that holds it, a runtime mask living under `/run` rather than beside the
# unit and a link to nothing being the same statement the manager calls `not-found`. A unit an
# earlier install registered would start a second copy, so it goes — and a removal that fails is said
# plainly rather than hidden. The mask is the one state left where it is, and the login consequence is
# said once, below.
if [ -e "$UNIT_PATH" ] && ! unit_masked; then
    if [ -n "$HAVE_SYSTEMD" ]; then
        if [ -n "$PRIVILEGED" ]; then
            as_root systemctl disable --now "$UNIT_NAME" >/dev/null 2>&1 || true
        fi
        # Checked after the disable, the way the lock block above does: a copy the unit owns
        # that is still running would be started twice, once by the login start beside it. The state
        # is a read, so it is asked of the manager directly rather than through the privileged helper.
        if systemctl is-active --quiet "$UNIT_NAME" 2>/dev/null; then
            fail "This install is not finished: $UNIT_NAME is still running; stop it ${AS_ROOT_SAYS}${AS_ROOT}systemctl stop $UNIT_NAME"
        fi
    fi
    # The one command that takes a unit an earlier install left off this host, as this host can run
    # it: the reload only means something where a service manager is running.
    REMOVE_HOW="${AS_ROOT}rm -f $UNIT_PATH"
    [ -z "$HAVE_SYSTEMD" ] || REMOVE_HOW="$REMOVE_HOW && ${AS_ROOT}systemctl daemon-reload"
    if [ -z "$PRIVILEGED" ]; then
        warn "The system service an earlier install registered is still at $UNIT_PATH, and this install cannot remove it without root ($NO_ROOT_WHY), so this host would keep two registrations; remove it ${AS_ROOT_SAYS}$REMOVE_HOW"
        fail "This install is not finished: the service unit an earlier install registered is still in place."
    fi
    if ! as_root rm -f "$UNIT_PATH" 2>/dev/null; then
        warn "$UNIT_PATH could not be removed, so this host would keep two registrations; remove it ${AS_ROOT_SAYS}$REMOVE_HOW"
        fail "This install is not finished: the service unit an earlier install registered is still in place."
    fi
    # The file is gone either way, and a manager that would not reload still holds the unit it read:
    # the run says that rather than calling a removal that happened a failed one.
    if [ -n "$HAVE_SYSTEMD" ] && ! as_root systemctl daemon-reload >/dev/null 2>&1; then
        warn "$UNIT_PATH was removed, but the service manager would not reload its unit files, so it still holds the unit this install took off; run ${AS_ROOT_SAYS}${AS_ROOT}systemctl daemon-reload"
        fail "This install is not finished: the service manager still holds the unit this install took off."
    fi
    say "The system service an earlier install registered was removed; MahBot starts at login."
fi

if [ "$OS" = linux ]; then
    if unit_masked; then
        # The product writes no entry beside a mask, so the promise this line makes on every other
        # host would be false here: the owner is told what will happen instead, and the one command
        # that gets the entry.
        say "The unit at ${UNIT_KEPT_AT:-$UNIT_PATH} is masked on this host, so nothing starts MahBot at login until the mask goes."
        say "The login entry is $AUTOSTART_ENTRY; MahBot does not write it while the mask stays."
        say "Undo the mask ${AS_ROOT_SAYS}$(unmask_command "$(unit_mask_form)")"
    else
        say "MahBot starts when you log in ($REGISTRATION_WHY)."
        say "The login entry is $AUTOSTART_ENTRY; MahBot writes it at its first start."
    fi
else
    say "MahBot starts when you log in."
fi
registration_switch_line

# The product detaches itself only when its standard output is a terminal (`src/detach.rs`), so
# handing it a pipe's session — the documented install is a pipe — would make it the foreground
# process of the installing session, which is how a server came to hang its SSH session. The terminal
# is handed over only where there is one, and what this run created goes first: `exec` replaces this
# shell, so the exit trap never runs for it. The product then keeps the terminal until it is up
# (bounded by `READY_TIMEOUT`), says so, and hands the prompt back; what it prints meanwhile goes to
# `$HOME/.mahbot/daemon.log`.
if [ -t 1 ]; then
    say "This install hands your terminal back to you once MahBot is up."
    cleanup
    exec "$DEST"
fi
say "Start it with: $DEST"
