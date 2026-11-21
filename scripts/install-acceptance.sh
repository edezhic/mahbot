#!/bin/sh
#
# install-acceptance.sh — the release gate for install.sh's registration: it installs the *published*
# release (never a local build) into throwaway Ubuntu 24.04 containers, one of them deliberately
# running no service manager, and asserts which registration the installer chose and what the host
# ended up with. Every other leg boots a real systemd as PID 1. `usage()` lists the legs; the
# assertions themselves are the specification.
#
# Two things about the evidence. A container restart (`docker restart`) stands in for a reboot — a
# container has no kernel boot, so the harness says so rather than claiming one. The display-manager
# evidence is read out of `install.sh` itself, so a manager added there cannot silently weaken these
# assertions.
#
# Prerequisites: Docker with a daemon this shell can reach, network access to the release host
# (https://github.com/edezhic/mahbot, or MAHBOT_RELEASE_BASE_URL), three to ten minutes for a full run
# (each install leg downloads the ~65 MB asset, and every install is bounded by a 10-minute hang
# timeout, so a slow release host stretches the run rather than failing it), and about 1 GB for the
# three images.
#
# Nothing on the host is touched: no $HOME, no host systemd, no MahBot instance. Everything happens in
# containers this script creates and, unless --keep is given, removes on exit — including on failure.
# Temp files live under mktemp -d.
#
set -eu

BASE_IMAGE=mahbot-acceptance-base
DESKTOP_IMAGE=mahbot-acceptance-desktop
PLAIN_IMAGE=mahbot-acceptance-plain
SERVER=mahbot-acceptance-server
MASKED=mahbot-acceptance-masked
LEFTOVER=mahbot-acceptance-leftover
NOSUDO=mahbot-acceptance-nosudo
NOSYSTEMD=mahbot-acceptance-no-systemd
DYING=mahbot-acceptance-dying
GRAPHICAL=mahbot-acceptance-graphical
DRYRUN=mahbot-acceptance-dry-run
DRYRUN_DESKTOP=mahbot-acceptance-dry-run-desktop
BAD_UNIT=mahbot-acceptance-bad-unit
ROOT=mahbot-acceptance-root

# The legs, in the order they run: `usage()` prints them, `--only` accepts them and the run below
# walks them, so a leg added here is a leg everywhere.
LEGS='clean-server reboot re-run broken-unit masked switch leftover no-sudo no-systemd dying-service root bad-unit graphical dry-run'

# The release to install: named with --version, else the newest published one, resolved below. A
# pinned default would quietly test a release the host has long moved past.
VERSION=''
RELEASE_BASE=${MAHBOT_RELEASE_BASE_URL:-https://github.com/edezhic/mahbot}
RELEASE_ENV=''
if [ -n "${MAHBOT_RELEASE_BASE_URL:-}" ]; then
    RELEASE_ENV="-e MAHBOT_RELEASE_BASE_URL=$MAHBOT_RELEASE_BASE_URL"
fi

# A download of the ~65 MB asset has taken 20–300 s here, so this bounds a hang
# (an installer waiting for something that never comes), not the network.
INSTALL_TIMEOUT=600

KEEP=0
REBUILD=0
ONLY=''
CONTAINERS=''
LEGS_DONE=''
LEGS_COUNT=0
CURRENT_LEG=''
INSTALL_OUT=''
INSTALL_STATUS=0
WORK=''

usage() {
    cat <<EOF
usage: scripts/install-acceptance.sh [--keep] [--rebuild] [--version <v>] [--only <leg>]

Installs the released MahBot into throwaway Ubuntu 24.04 containers — one of them
running no service manager at all, the rest a real systemd — and asserts what the
installer did. No arguments runs every leg.

  --keep            leave the containers behind for inspection (default: remove them)
  --rebuild         rebuild the three acceptance images even if they exist
  --version <v>     the release to install (default: the newest published release)
  --only <leg>      run one leg (no argument runs them all):
                      $(printf '%s' "$LEGS" | sed 's/ /, /g').
                    reboot, re-run, broken-unit and switch continue the clean-server
                    container and need it to exist.

The release host is https://github.com/edezhic/mahbot, or MAHBOT_RELEASE_BASE_URL.
EOF
}

say() { printf '%s\n' "$1"; }
warn() { printf '%s\n' "$1" >&2; }
ok() { printf 'ok: %s\n' "$1"; }

fail() {  # fail <what was expected> <what was seen>
    printf 'FAILED: %s — got: %s\n' "$1" "${2:-nothing}" >&2
    if [ -n "$INSTALL_OUT" ] && [ -s "$INSTALL_OUT" ]; then
        printf '%s\n' '── container output of the last run ──' >&2
        tail -n 40 "$INSTALL_OUT" >&2
        printf '%s\n' '──────────────────────────────────────' >&2
    fi
    exit 1
}

check() {  # check <description> <command...>
    _desc=$1
    shift
    if _seen=$("$@" 2>&1); then
        ok "$_desc"
    else
        fail "$_desc" "${_seen:-the command exited non-zero}"
    fi
}

check_eq() {  # check_eq <description> <expected> <actual>
    if [ "$2" = "$3" ]; then
        ok "$1"
    else
        fail "$1 (expected '$2')" "$3"
    fi
}

check_contains() {  # check_contains <description> <file> <text>
    if grep -qF -- "$3" "$2"; then
        ok "$1"
    else
        fail "$1 (expected to find '$3')" "$(tail -n 40 "$2")"
    fi
}

check_lacks() {  # check_lacks <description> <file> <text>
    if [ ! -r "$2" ]; then
        fail "$1" "the file to search is missing: $2"
    elif _seen=$(grep -F -- "$3" "$2"); then
        fail "$1 (must not contain '$3')" "$_seen"
    else
        ok "$1"
    fi
}

check_empty() {  # check_empty <description> <command...>
    _desc=$1
    shift
    if _seen=$("$@" 2>&1) && [ -z "$_seen" ]; then
        ok "$_desc"
    else
        fail "$_desc (expected nothing)" "${_seen:-the command exited non-zero}"
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --keep) KEEP=1 ;;
        --rebuild) REBUILD=1 ;;
        --version)
            [ $# -ge 2 ] || { usage >&2; exit 2; }
            VERSION=$2
            shift
            ;;
        --only)
            [ $# -ge 2 ] || { usage >&2; exit 2; }
            ONLY=$2
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
    shift
done

VERSION=${VERSION#v}

if [ -n "$ONLY" ]; then
    case " $LEGS " in
        *" $ONLY "*) ;;
        *)
            warn "install-acceptance: unknown leg '$ONLY'."
            usage >&2
            exit 2
            ;;
    esac
fi

need_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        warn "install-acceptance: docker is not installed or not on PATH, and this harness needs a reachable Docker daemon."
        exit 2
    fi
    if ! docker info >/dev/null 2>&1; then
        warn "install-acceptance: the Docker daemon is not reachable, so no container can be started."
        exit 2
    fi
}

need_docker

# The release under test, named or the newest one published — never a local build, and never a
# version this script remembers.
if [ -z "$VERSION" ]; then
    command -v curl >/dev/null 2>&1 ||
        {
            warn "install-acceptance: curl is needed to look the newest release up; name one with --version <v>."
            exit 2
        }
    VERSION=$(curl -fsSL "$RELEASE_BASE/releases/latest/download/version.txt" 2>/dev/null | tr -d '\r\n' || true)
    VERSION=${VERSION#v}
    if [ -z "$VERSION" ]; then
        warn "install-acceptance: the newest release could not be looked up at $RELEASE_BASE; name one with --version <v>."
        exit 2
    fi
    say "the newest published release is v$VERSION (name another with --version <v>)"
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/mahbot-acceptance.XXXXXX")
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
INSTALLER=$REPO/install.sh

finish() {
    _status=$?
    trap - EXIT HUP INT TERM
    if [ "$KEEP" -eq 1 ] && [ -n "$CONTAINERS" ]; then
        say ""
        say "containers kept for inspection:$CONTAINERS"
        say "remove them with: docker rm -f$CONTAINERS"
    else
        for _c in $CONTAINERS; do
            docker rm -f "$_c" >/dev/null 2>&1 || true
        done
    fi
    rm -rf "$WORK"
    say ""
    if [ "$_status" -eq 0 ]; then
        say "summary: $LEGS_COUNT leg(s) passed:$LEGS_DONE"
    elif [ -n "$CURRENT_LEG" ]; then
        say "summary: FAILED in leg '$CURRENT_LEG' after $LEGS_COUNT leg(s) passed:$LEGS_DONE"
    else
        say "summary: FAILED after $LEGS_COUNT leg(s) passed:$LEGS_DONE"
    fi
    exit "$_status"
}
trap finish EXIT HUP INT TERM

cat >"$WORK/hold-lock.sh" <<'EOF'
#!/bin/sh
exec 9>>"$HOME/.mahbot/mahbot.lock"
flock 9
exec sleep 600
EOF

[ -f "$INSTALLER" ] || fail "the installer under test is at $INSTALLER" "no such file"

# ── Images ────────────────────────────────────────────────────────────────

# The service manager and the two tools the installer needs, and an ordinary account with sudo: the
# unit runs as the account that installs, so the legs install as `owner` — the `root` leg installs as
# root on purpose, and asserts the installer says what that means.
build_base_image() {
    docker build -t "$BASE_IMAGE" - <<'EOF'
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -o Acquire::ForceIPv4=true update \
 && apt-get -o Acquire::ForceIPv4=true install -y systemd systemd-sysv curl ca-certificates sudo \
 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash owner \
 && printf 'owner ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/owner \
 && chmod 0440 /etc/sudoers.d/owner
EOF
}

# The same host with a display manager: greetd is one small package, needs no X, and brings the
# `display-manager.service` alias the installer reads as its evidence.
build_desktop_image() {
    docker build -t "$DESKTOP_IMAGE" - <<EOF
FROM $BASE_IMAGE
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -o Acquire::ForceIPv4=true update \\
 && apt-get -o Acquire::ForceIPv4=true install -y --no-install-recommends greetd \\
 && rm -rf /var/lib/apt/lists/*
EOF
}

# A host with no service manager at all — the one the installer must not start the product in the
# foreground of.
build_plain_image() {
    docker build -t "$PLAIN_IMAGE" - <<'EOF'
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -o Acquire::ForceIPv4=true update \
 && apt-get -o Acquire::ForceIPv4=true install -y curl ca-certificates sudo \
 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash owner \
 && printf 'owner ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/owner \
 && chmod 0440 /etc/sudoers.d/owner
EOF
}

ensure_image() {  # ensure_image <name> <what it is> <builder>
    if [ "$REBUILD" -eq 0 ] && docker image inspect "$1" >/dev/null 2>&1; then
        say "reusing image $1 (pass --rebuild to rebuild it)"
        return 0
    fi
    say "building image $1 ($2)"
    if ! "$3" >"$WORK/build-$1.log" 2>&1; then
        cat "$WORK/build-$1.log" >&2
        fail "the image $1 builds" "the docker build output above"
    fi
    say "built image $1"
}

ensure_images() {
    ensure_image "$BASE_IMAGE" "the service manager; a minute or two" build_base_image
    ensure_image "$DESKTOP_IMAGE" "base plus greetd; a minute" build_desktop_image
    ensure_image "$PLAIN_IMAGE" "no service manager; a minute" build_plain_image
}

# ── Containers ────────────────────────────────────────────────────────────

# Poll `$@` once a second until it succeeds, or `$1` seconds pass; the status is its own.
wait_until() {  # wait_until <seconds> <command...>
    _tries=$1
    shift
    _i=0
    while [ "$_i" -lt "$_tries" ]; do
        if "$@" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        _i=$((_i + 1))
    done
    return 1
}

wait_systemd() {  # wait_systemd <container>
    if wait_until 120 docker exec "$1" systemctl is-system-running --quiet; then
        ok "systemd reached 'running' in $1"
        return 0
    fi
    fail "systemd in $1 reaches 'running' within 120 s" \
        "$(docker exec "$1" systemctl is-system-running 2>&1 || true)"
}

new_container() {  # new_container <name> <image> <service-manager: yes|no>
    docker rm -f "$1" >/dev/null 2>&1 || true
    if [ "$3" = yes ]; then
        _init='exec /lib/systemd/systemd'
    else
        _init='exec sleep infinity'
    fi
    if ! docker run -d --name "$1" --privileged --cgroupns=host \
        -v /sys/fs/cgroup:/sys/fs/cgroup:rw "$2" \
        /bin/bash -c "$_init" >/dev/null; then
        fail "a container from $2 starts" "docker run failed"
    fi
    CONTAINERS="$CONTAINERS $1"
    if [ "$3" = yes ]; then
        wait_systemd "$1"
    else
        say "$1 runs no service manager: its image has none, which is the point of this leg."
    fi
}

require_container() {  # require_container <name>
    if ! docker container inspect "$1" >/dev/null 2>&1; then
        fail "the container $1 exists (leg '$CURRENT_LEG' continues clean-server; run --only clean-server --keep first)" \
            "no container named $1"
    fi
}

# The host most legs start from: a throwaway container with a real service manager and the installer
# staged in it.
new_installed_container() {  # new_installed_container <name> <image>
    new_container "$1" "$2" yes
    stage_installer "$1"
}

leg_start() {
    CURRENT_LEG=$1
    INSTALL_OUT="$WORK/install-$1.out"
    : >"$INSTALL_OUT"
    printf '\n── leg: %s ──\n' "$1"
}

wait_active() {  # wait_active <container> <seconds>
    wait_until "$2" docker exec "$1" systemctl is-active --quiet mahbot.service
}

wait_lock() {  # wait_lock <container> <seconds>
    wait_until "$2" lock_is_held "$1"
}

sessions_on() {
    docker exec "$1" loginctl list-sessions --no-legend
}

# The display-manager evidence, read out of the installer rather than copied: a manager added there
# must not silently weaken the "no display manager is present" assertion here.
DM_EVIDENCE=$(sed -n '/^have_display_manager()/,/^}/p' "$INSTALLER")
DM_UNITS=$(printf '%s\n' "$DM_EVIDENCE" | grep -oE '[A-Za-z0-9_-]+\.service' | sort -u || true)
DM_DIRS=$(printf '%s\n' "$DM_EVIDENCE" | grep -oE '(/[A-Za-z0-9_.-]+)+/systemd/system' | sort -u || true)

display_manager_on() {
    docker exec "$1" sh -c '
        for unit in $1; do
            for dir in $2; do
                if [ -e "$dir/$unit" ]; then
                    printf "%s\n" "$dir/$unit"
                fi
            done
        done
    ' sh "$DM_UNITS" "$DM_DIRS"
}

container_arch() {
    case "$(docker exec "$1" uname -m)" in
        x86_64 | amd64) printf '%s\n' x86_64 ;;
        *) printf '%s\n' aarch64 ;;
    esac
}

unit_hash() {
    docker exec "$1" sha256sum /etc/systemd/system/mahbot.service | cut -d' ' -f1
}

lock_is_free() {
    # A lock file that does not exist yet is a lock nothing holds: `flock` would fail on it and read
    # as held.
    docker exec -u owner -e HOME=/home/owner "$1" sh -c \
        'test ! -e "$HOME/.mahbot/mahbot.lock" || flock -n "$HOME/.mahbot/mahbot.lock" true 2>/dev/null'
}

lock_is_held() {
    ! lock_is_free "$1"
}

stage_installer() {  # stage_installer <container>
    docker cp "$INSTALLER" "$1:/tmp/install.sh" >/dev/null
    docker exec "$1" chmod 0644 /tmp/install.sh
}

# A login entry at each directory named, as an earlier release left it: the product's own content is
# not what these legs read, only where the entry sits.
write_login_entries() {  # write_login_entries <container> <dir>...
    _c=$1
    shift
    docker exec "$_c" sh -c '
        for dir in "$@"; do
            mkdir -p "$dir/autostart"
            printf "[Desktop Entry]\nType=Application\nName=MahBot\n" > "$dir/autostart/mahbot.desktop"
        done
        chown -R owner:owner "$@"
    ' sh "$@"
}

# The unit the installer itself would write on this host, read out of its own dry run: a fixture taken
# from the script cannot drift from the unit under test.
installer_unit() {  # installer_unit <container>
    docker exec -u owner -e HOME=/home/owner -w /home/owner -e MAHBOT_INSTALL_DRY_RUN=1 "$1" \
        sh /tmp/install.sh 2>/dev/null | sed -n '/^\[Unit\]$/,/^WantedBy=/p'
}

# The shipped release's defect, reproduced on a unit that loads: a `WorkingDirectory=` the manager reads
# as a relative path, so it refuses the unit — while `systemctl enable` still succeeds.
break_unit() {  # break_unit <container>
    docker exec "$1" sh -c '
        sed "s|^WorkingDirectory=.*|WorkingDirectory=\"/home/owner\"|" \
            /etc/systemd/system/mahbot.service > /tmp/broken.service
        cat /tmp/broken.service > /etc/systemd/system/mahbot.service
        systemctl daemon-reload
    '
}

# Run the staged installer the way the documented one-liner runs it, as the ordinary account that
# installs, with the version under test named the way a pipe cannot: through the environment. `$1` is
# the container, `$2` says whether the run gets a terminal, and anything after it is one more
# `docker exec` flag or environment assignment — the last of each wins there, which is how the legs
# that need another account or home override the two below.
run_installer() {
    _c=$1
    _pty=$2
    shift 2
    _flags=''
    if [ "$_pty" = yes ]; then
        _flags=-t
    fi
    # shellcheck disable=SC2086
    if _out=$(docker exec $_flags -u owner -e HOME=/home/owner -w /home/owner \
        $RELEASE_ENV -e MAHBOT_INSTALL_VERSION="$VERSION" "$@" "$_c" \
        timeout "$INSTALL_TIMEOUT" sh -c 'cat /tmp/install.sh | sh' 2>&1); then
        INSTALL_STATUS=0
    else
        INSTALL_STATUS=$?
    fi
    printf '%s\n' "$_out" | tr -d '\r' >"$INSTALL_OUT"
}

# ── Legs ──────────────────────────────────────────────────────────────────

leg_clean_server() {
    leg_start clean-server
    new_installed_container "$SERVER" "$BASE_IMAGE"

    check_eq "systemctl get-default is graphical.target, the misclassification trigger" \
        graphical.target "$(docker exec "$SERVER" systemctl get-default)"
    check_empty "no display manager is present (the installer's own evidence check)" \
        display_manager_on "$SERVER"

    run_installer "$SERVER" yes
    check_eq "the installer exits 0" 0 "$INSTALL_STATUS"
    # The pinned release is the one on the machine, not whatever `releases/latest` answered at some
    # other moment: a gate that installs a different release than the one it names proves nothing.
    check_eq "the binary is the pinned release" "mahbot $VERSION" \
        "$(docker exec -u owner -e HOME=/home/owner "$SERVER" /home/owner/.local/bin/mahbot --version 2>/dev/null)"
    check_contains "it names the system service it registered" "$INSTALL_OUT" \
        "MahBot is registered as a system service at /etc/systemd/system/mahbot.service"
    check_contains "it names the evidence the choice was made on" "$INSTALL_OUT" \
        "no display manager was found"
    check_contains "it names how to ask for the login registration instead" "$INSTALL_OUT" \
        "MAHBOT_INSTALL_REGISTRATION=login"
    check "the unit is at /etc/systemd/system/mahbot.service" \
        docker exec "$SERVER" test -f /etc/systemd/system/mahbot.service
    check "systemd-analyze verify accepts the unit" \
        docker exec "$SERVER" systemd-analyze verify /etc/systemd/system/mahbot.service
    check_eq "WorkingDirectory= is unquoted and names the owner's home" \
        'WorkingDirectory=/home/owner' \
        "$(docker exec "$SERVER" grep '^WorkingDirectory=' /etc/systemd/system/mahbot.service)"
    check_eq "the unit is enabled" enabled \
        "$(docker exec "$SERVER" systemctl is-enabled mahbot.service)"
    check "the service is active" wait_active "$SERVER" 30
    check "no login entry was written" \
        docker exec "$SERVER" test ! -e /home/owner/.config/autostart/mahbot.desktop
    check_empty "nobody is logged in" sessions_on "$SERVER"
}

leg_reboot() {
    leg_start reboot
    require_container "$SERVER"
    say "This is a container restart (docker restart), the accepted stand-in for a reboot: a container has no kernel boot."
    docker restart "$SERVER" >/dev/null
    wait_systemd "$SERVER"
    check "the service is active again after the restart" wait_active "$SERVER" 60
    check_empty "still nobody is logged in after the restart" sessions_on "$SERVER"
}

leg_re_run() {
    leg_start re-run
    require_container "$SERVER"
    stage_installer "$SERVER"
    _before=$(unit_hash "$SERVER")
    run_installer "$SERVER" yes
    check_eq "the second install exits 0" 0 "$INSTALL_STATUS"
    check_contains "it reports the registration is already there" "$INSTALL_OUT" \
        "already registered as a system service at /etc/systemd/system/mahbot.service"
    check_eq "the unit file is byte-for-byte unchanged" "$_before" "$(unit_hash "$SERVER")"
    check "the service is active afterwards" wait_active "$SERVER" 30
}

leg_broken_unit() {
    leg_start broken-unit
    require_container "$SERVER"
    stage_installer "$SERVER"

    break_unit "$SERVER"
    check_eq "the shipped broken unit is in place" 'WorkingDirectory="/home/owner"' \
        "$(docker exec "$SERVER" grep '^WorkingDirectory=' /etc/systemd/system/mahbot.service)"
    check "systemd-analyze verify refuses the broken unit" \
        docker exec "$SERVER" sh -c '! systemd-analyze verify /etc/systemd/system/mahbot.service >/dev/null 2>&1'

    run_installer "$SERVER" yes
    check_eq "the install exits 0" 0 "$INSTALL_STATUS"
    say "── the installer's own words, verbatim ──"
    cat "$INSTALL_OUT"
    say "─────────────────────────────────────────"
    check_contains "it names the unit that would not load" "$INSTALL_OUT" "would not load"
    check "the repaired unit passes systemd-analyze verify" \
        docker exec "$SERVER" systemd-analyze verify /etc/systemd/system/mahbot.service
    check "the service is active on the repaired unit" wait_active "$SERVER" 30

    # The same repair on a unit the owner disabled while it ran: the file does not load, so a new one
    # is written, and the run the lock block stopped for the swap has to be put back — a run never
    # leaves the host worse than it found it.
    say "Disabling the unit while it runs, then breaking its file again."
    docker exec "$SERVER" systemctl disable mahbot.service >/dev/null 2>&1
    break_unit "$SERVER"
    check_eq "the unit is disabled while it runs" disabled \
        "$(docker exec "$SERVER" systemctl is-enabled mahbot.service 2>/dev/null || true)"
    check "the disabled unit is still running" wait_active "$SERVER" 10
    wait_lock "$SERVER" 30 || true

    run_installer "$SERVER" no
    check_eq "the re-run on a disabled unit exits 0" 0 "$INSTALL_STATUS"
    check_contains "it says it did not enable the unit" "$INSTALL_OUT" \
        "so this install did not enable it"
    check_contains "it says the run it stopped is back" "$INSTALL_OUT" \
        "was stopped to replace its file and is running again"
    check_eq "the unit is still disabled" disabled \
        "$(docker exec "$SERVER" systemctl is-enabled mahbot.service 2>/dev/null || true)"
    check "the unit is running again" wait_active "$SERVER" 10
}

leg_masked() {
    leg_start masked
    new_installed_container "$MASKED" "$BASE_IMAGE"

    # A mask is a link to `/dev/null`, and `[ -f ]` says no for it: an install that probes for a unit
    # file that way never sees the owner's mask, writes through it and deletes it on the way out. A
    # mask cannot replace a file that is already there, so the owner's route to one is to remove the
    # unit and mask its name — which is also the state that has to survive a re-run.
    run_installer "$MASKED" no
    check_eq "the first install exits 0" 0 "$INSTALL_STATUS"
    docker exec "$MASKED" sh -c '
        systemctl disable --now mahbot.service >/dev/null 2>&1
        rm -f /etc/systemd/system/mahbot.service
        systemctl mask mahbot.service
    ' >/dev/null 2>&1
    check_eq "the unit is a mask now" /dev/null \
        "$(docker exec "$MASKED" readlink /etc/systemd/system/mahbot.service)"
    check_eq "the manager reports it masked" masked \
        "$(docker exec "$MASKED" systemctl is-enabled mahbot.service 2>/dev/null || true)"

    run_installer "$MASKED" no
    check_eq "the re-run exits 0" 0 "$INSTALL_STATUS"
    check_contains "it reports the state it found" "$INSTALL_OUT" "is 'masked' on this host"
    check_contains "it says it left it as it is" "$INSTALL_OUT" "left it as it is"
    check_eq "the mask is still the owner's mask" /dev/null \
        "$(docker exec "$MASKED" readlink /etc/systemd/system/mahbot.service)"
    check_eq "the manager still reports it masked" masked \
        "$(docker exec "$MASKED" systemctl is-enabled mahbot.service 2>/dev/null || true)"

    # An explicit ask is never answered silently: a mask is the owner's decision, so the run that is
    # told to register the service must say it cannot, non-zero, and leave the mask alone.
    run_installer "$MASKED" no -e MAHBOT_INSTALL_REGISTRATION=service
    check "the explicit ask a mask blocks is an error" test "$INSTALL_STATUS" -ne 0
    check_contains "it names the mask that blocks it" "$INSTALL_OUT" "is 'masked' on this host"
    check_contains "it names how to satisfy the ask" "$INSTALL_OUT" \
        "sudo systemctl unmask mahbot.service"
    check_eq "the mask survived the refusal" /dev/null \
        "$(docker exec "$MASKED" readlink /etc/systemd/system/mahbot.service)"

    # A runtime mask shadows a unit that is not in `/etc` — a unit file there takes precedence — so the
    # owner's route to this state is to remove the unit and mask the name for this boot only. The
    # install must name the path that actually holds the mask, never one where no file exists.
    say "Removing the unit and masking its name for this boot only."
    docker exec "$MASKED" systemctl unmask mahbot.service >/dev/null 2>&1
    run_installer "$MASKED" no
    docker exec "$MASKED" sh -c '
        systemctl disable --now mahbot.service >/dev/null 2>&1
        rm -f /etc/systemd/system/mahbot.service
        systemctl daemon-reload
        systemctl mask --runtime mahbot.service
    ' >/dev/null 2>&1
    check_eq "the manager reports a runtime mask" masked-runtime \
        "$(docker exec "$MASKED" systemctl is-enabled mahbot.service 2>/dev/null || true)"
    check "the mask is under /run" \
        docker exec "$MASKED" test -L /run/systemd/system/mahbot.service
    run_installer "$MASKED" no
    check_eq "the re-run with a runtime mask exits 0" 0 "$INSTALL_STATUS"
    check_contains "it names the path that holds the runtime mask" "$INSTALL_OUT" \
        "at /run/systemd/system/mahbot.service"
    check_contains "it names the runtime state" "$INSTALL_OUT" "is 'masked-runtime' on this host"
    check_contains "its remedy removes the runtime mask, not the persistent one" "$INSTALL_OUT" \
        "systemctl unmask --runtime mahbot.service"
    check "the runtime mask is still there" \
        docker exec "$MASKED" test -L /run/systemd/system/mahbot.service
    docker exec "$MASKED" rm -f /run/systemd/system/mahbot.service
    docker exec "$MASKED" systemctl daemon-reload >/dev/null 2>&1

    # The owner's other decision: a unit that is disabled but running. The lock block stops it to swap
    # the file, and the install must put it back running without enabling it — and must say so only
    # once the manager agrees it is up.
    say "Unmasking, installing again, then disabling the unit while it runs."
    run_installer "$MASKED" no
    check_eq "the install after the unmask exits 0" 0 "$INSTALL_STATUS"
    docker exec "$MASKED" systemctl disable mahbot.service >/dev/null 2>&1
    check_eq "the unit is disabled" disabled \
        "$(docker exec "$MASKED" systemctl is-enabled mahbot.service 2>/dev/null || true)"
    check "the unit is still running" wait_active "$MASKED" 10
    wait_lock "$MASKED" 30 || true
    check "the disabled unit holds the instance lock" lock_is_held "$MASKED"

    # The login entry is the owner's own file, and it goes on every path where the unit is the
    # registration in use — a disabled unit included — or the host keeps two registrations.
    write_login_entries "$MASKED" /home/owner/.config

    run_installer "$MASKED" no
    check_eq "the re-run exits 0" 0 "$INSTALL_STATUS"
    check_contains "it says the unit was put back running" "$INSTALL_OUT" \
        "was stopped to replace its file and is running again"
    check_eq "the unit is still disabled" disabled \
        "$(docker exec "$MASKED" systemctl is-enabled mahbot.service 2>/dev/null || true)"
    check "the unit is running again" wait_active "$MASKED" 10
    check "the login entry beside the kept unit is gone" \
        docker exec "$MASKED" test ! -e /home/owner/.config/autostart/mahbot.desktop
}

leg_switch() {
    leg_start switch
    require_container "$SERVER"
    stage_installer "$SERVER"

    # The other registration, asked for explicitly: the unit a working host has goes, and nothing of
    # it is left enabled. No terminal here, so the product is not started by the install itself.
    run_installer "$SERVER" no -e MAHBOT_INSTALL_REGISTRATION=login
    check_eq "the switch to login exits 0" 0 "$INSTALL_STATUS"
    check "the unit is gone" \
        docker exec "$SERVER" test ! -e /etc/systemd/system/mahbot.service
    check "nothing of it is enabled" \
        docker exec "$SERVER" sh -c '! systemctl is-enabled mahbot.service >/dev/null 2>&1'
    check_contains "it names the login entry it leaves to the product" "$INSTALL_OUT" \
        "The login entry is /home/owner/.config/autostart/mahbot.desktop"

    # And back: the login entry goes and the unit is written, enabled and running.
    run_installer "$SERVER" no -e MAHBOT_INSTALL_REGISTRATION=service
    check_eq "the switch back to the service exits 0" 0 "$INSTALL_STATUS"
    check_eq "the unit is enabled again" enabled \
        "$(docker exec "$SERVER" systemctl is-enabled mahbot.service)"
    check "the service is active again" wait_active "$SERVER" 30
    check "the login entry is gone" \
        docker exec "$SERVER" test ! -e /home/owner/.config/autostart/mahbot.desktop
}

leg_leftover() {
    leg_start leftover
    new_installed_container "$LEFTOVER" "$BASE_IMAGE"

    check "no unit is installed yet" \
        docker exec "$LEFTOVER" test ! -e /etc/systemd/system/mahbot.service
    say "Putting the published binary at ~/.local/bin/mahbot and starting it with no unit, as the broken release left it."
    if ! docker exec -u owner -e HOME=/home/owner -w /home/owner "$LEFTOVER" sh -c '
        set -e
        mkdir -p "$HOME/.local/bin" "$HOME/.mahbot"
        curl -fsSL -o /tmp/mahbot.tar.gz "$1"
        tar -xzf /tmp/mahbot.tar.gz -C "$HOME/.local/bin" mahbot
        chmod 755 "$HOME/.local/bin/mahbot"
    ' sh "$RELEASE_BASE/releases/download/v$VERSION/mahbot-$VERSION-linux-$(container_arch "$LEFTOVER").tar.gz" >/dev/null 2>&1; then
        fail "the published binary can be fetched into $LEFTOVER" "the download or extraction failed"
    fi
    docker exec -u owner -e HOME=/home/owner -w /home/owner "$LEFTOVER" sh -c '
        setsid "$HOME/.local/bin/mahbot" >/tmp/leftover.log 2>&1 </dev/null &
        echo $!
    ' >"$WORK/leftover.pid"

    wait_lock "$LEFTOVER" 10 || true
    _kind='the published binary (setsid ~/.local/bin/mahbot)'
    if lock_is_free "$LEFTOVER"; then
        _kind='a flock(1) holder (setsid sh -c "exec 9>>~/.mahbot/mahbot.lock; flock 9; exec sleep 600")'
        docker cp "$WORK/hold-lock.sh" "$LEFTOVER:/tmp/hold-lock.sh" >/dev/null
        docker exec "$LEFTOVER" chmod 0755 /tmp/hold-lock.sh
        docker exec -u owner -e HOME=/home/owner "$LEFTOVER" sh -c \
            'setsid /tmp/hold-lock.sh >/dev/null 2>&1 </dev/null &'
        sleep 2
    fi
    say "The leftover process is $_kind."
    check "the leftover holds ~/.mahbot/mahbot.lock with no unit installed" lock_is_held "$LEFTOVER"

    run_installer "$LEFTOVER" no
    check "the install refuses a leftover process" test "$INSTALL_STATUS" -ne 0
    check "the refusal is not a timeout" test "$INSTALL_STATUS" -ne 124
    check_contains "it names the kill command" "$INSTALL_OUT" "kill "
    _pid=$(sed -n 's/.*kill \([0-9][0-9]*\).*/\1/p' "$INSTALL_OUT" | head -n 1)
    check "it names the pid to kill" test -n "$_pid"

    say "Killing $_pid and running the installer again."
    docker exec "$LEFTOVER" kill "$_pid"
    wait_until 60 lock_is_free "$LEFTOVER" || true
    check "the lock is released after the kill" lock_is_free "$LEFTOVER"

    # The broken release left the product's login entry on a host like this one. The installer must
    # clear it at the path this shell's config home names *and* at the one a service start computes,
    # or the host keeps two registrations until the product's first start clears the second.
    say "Placing the login entry the broken release left, at both paths the product can compute."
    write_login_entries "$LEFTOVER" /home/owner/.config /home/owner/.config-xdg

    run_installer "$LEFTOVER" yes -e XDG_CONFIG_HOME=/home/owner/.config-xdg
    check_eq "the re-run exits 0" 0 "$INSTALL_STATUS"
    check_eq "the service is registered and enabled" enabled \
        "$(docker exec "$LEFTOVER" systemctl is-enabled mahbot.service)"
    check "the service is active" wait_active "$LEFTOVER" 30
    check "the login entry at the XDG config home was removed" \
        docker exec "$LEFTOVER" test ! -e /home/owner/.config-xdg/autostart/mahbot.desktop
    check "the login entry at the home config path was removed too" \
        docker exec "$LEFTOVER" test ! -e /home/owner/.config/autostart/mahbot.desktop
}

leg_no_sudo() {
    leg_start no-sudo
    new_installed_container "$NOSUDO" "$BASE_IMAGE"
    # The unit fixture is taken while sudo is still there: on this host the dry run refuses to render
    # one, which is asserted below.
    installer_unit "$NOSUDO" >"$WORK/unit-nosudo"
    if ! docker exec "$NOSUDO" env SUDO_FORCE_REMOVE=yes DEBIAN_FRONTEND=noninteractive \
        apt-get remove -y sudo >/dev/null 2>&1; then
        fail "sudo can be removed from the no-sudo container" "apt-get remove -y sudo failed"
    fi
    check "sudo is not installed" \
        docker exec "$NOSUDO" sh -c '! command -v sudo >/dev/null 2>&1'

    run_installer "$NOSUDO" no
    check "the install exits non-zero" test "$INSTALL_STATUS" -ne 0
    check "it does not hit the timeout (it does not hang)" test "$INSTALL_STATUS" -ne 124
    check "no unit was written" \
        docker exec "$NOSUDO" test ! -e /etc/systemd/system/mahbot.service
    check_contains "it says sudo was not found" "$INSTALL_OUT" "sudo was not found"
    check_contains "it prints the unit for the owner to install, framed for this host" "$INSTALL_OUT" \
        "Save this as /etc/systemd/system/mahbot.service as root with: tee /etc/systemd/system/mahbot.service, then run:"
    check_contains "it names how to start the unit" "$INSTALL_OUT" \
        "systemctl daemon-reload && systemctl enable --now mahbot.service"
    check_contains "the printed unit has an unquoted WorkingDirectory=" "$INSTALL_OUT" \
        "WorkingDirectory=/home/owner"
    check_lacks "it prints no sudo command this host cannot run" "$INSTALL_OUT" "sudo systemctl"

    # The read-only mode answers the same question the run does: on a host that cannot write a unit,
    # it says the run would refuse instead of rendering a unit the run would not write.
    run_installer "$NOSUDO" no -e MAHBOT_INSTALL_DRY_RUN=1
    check "the dry run reports the refusal the run would hit" test "$INSTALL_STATUS" -ne 0
    check_contains "it says why the run would refuse" "$INSTALL_OUT" \
        "the run would refuse, because sudo was not found"
    check_lacks "it renders no unit the run would not write" "$INSTALL_OUT" "[Unit]"

    # The same host once a unit of its own is in service: the lock belongs to that service, and the
    # install cannot stop it without root — so it must name the service and the command, never send
    # the owner after a pid of their own service.
    say "Putting a working unit in place and starting it, then running the install again with no sudo."
    docker cp "$WORK/unit-nosudo" "$NOSUDO:/etc/systemd/system/mahbot.service" >/dev/null
    docker exec "$NOSUDO" sh -c '
        chown root:root /etc/systemd/system/mahbot.service
        systemctl daemon-reload
        systemctl enable --now mahbot.service
    ' >/dev/null 2>&1
    check "the unit is active" wait_active "$NOSUDO" 30
    wait_lock "$NOSUDO" 30 || true
    check "the service holds the instance lock" lock_is_held "$NOSUDO"

    run_installer "$NOSUDO" no
    check "the install refuses, because it cannot stop the service" test "$INSTALL_STATUS" -ne 0
    check_contains "it names the service that holds the lock" "$INSTALL_OUT" \
        "MahBot is running as the system service mahbot.service"
    check_contains "it names the command that stops it" "$INSTALL_OUT" \
        "Stop it as root with: systemctl stop mahbot.service"
    check_lacks "it does not send the owner after a pid of their own service" "$INSTALL_OUT" "kill "
    check_lacks "it prints no sudo command this host cannot run" "$INSTALL_OUT" "sudo systemctl"
    check "the service is still running" wait_active "$NOSUDO" 10

    # An explicit ask is answered with the unit this host already has and the step that fixes it, not
    # with a claim that no unit can be installed — and it is answered before anything is downloaded.
    run_installer "$NOSUDO" no -e MAHBOT_INSTALL_REGISTRATION=service
    check "the explicit ask on a host with a unit and no root is an error" test "$INSTALL_STATUS" -ne 0
    check_contains "it names the unit that is already installed" "$INSTALL_OUT" \
        "is installed on this host, and this install cannot replace its file and restart it without root"
    check_contains "it names the next step" "$INSTALL_OUT" "Re-run this install as root"
}

leg_no_systemd() {
    leg_start no-systemd
    new_container "$NOSYSTEMD" "$PLAIN_IMAGE" no
    stage_installer "$NOSYSTEMD"
    check "no service manager runs on this host" \
        docker exec "$NOSYSTEMD" sh -c '! systemctl is-system-running >/dev/null 2>&1'

    # No terminal and no unit: the shape the shipped installer hung an SSH session with, by running
    # the product in the foreground of the installing session.
    run_installer "$NOSYSTEMD" no
    check_eq "the install exits 0" 0 "$INSTALL_STATUS"
    check "it does not hit the timeout (it does not hang)" test "$INSTALL_STATUS" -ne 124
    check_contains "it says no service manager was found" "$INSTALL_OUT" \
        "no service manager to register a unit with"
    check_contains "it says how to start the product" "$INSTALL_OUT" \
        "Start it with: /home/owner/.local/bin/mahbot"
    check "the binary is installed" \
        docker exec "$NOSYSTEMD" test -x /home/owner/.local/bin/mahbot
    check "no unit was written" \
        docker exec "$NOSYSTEMD" test ! -e /etc/systemd/system/mahbot.service
    check_empty "the product was not left running as the service" \
        docker exec "$NOSYSTEMD" sh -c 'ps -u owner -o comm= | grep -x mahbot || true'
    check_lacks "it offers no switch to a registration this host cannot take" "$INSTALL_OUT" \
        "MAHBOT_INSTALL_REGISTRATION=service"

    # An explicit ask for the service on a host that has none is refused before anything is
    # downloaded, rather than after the binary is in place.
    run_installer "$NOSYSTEMD" no -e MAHBOT_INSTALL_REGISTRATION=service
    check "the explicit ask for the service is refused" test "$INSTALL_STATUS" -ne 0
    check_contains "the refusal names why" "$INSTALL_OUT" "no service manager runs on this host"

    # The read-only mode on this host: the login registration, and no unit for a host that could not
    # load one.
    run_installer "$NOSYSTEMD" no -e MAHBOT_INSTALL_DRY_RUN=1
    check_eq "the dry run exits 0" 0 "$INSTALL_STATUS"
    check_contains "it names the login registration and why" "$INSTALL_OUT" \
        "registration: login (no service manager to register a unit with)"
    check_lacks "it prints no unit for a host that cannot load one" "$INSTALL_OUT" "[Unit]"
    check_lacks "it offers no switch to the service" "$INSTALL_OUT" \
        "MAHBOT_INSTALL_REGISTRATION=service"
}

leg_dying_service() {
    leg_start dying-service
    new_installed_container "$DYING" "$BASE_IMAGE"

    # A unit that loads, is enabled, starts — and dies. `Type=simple` reports it as active the moment
    # the process forks, which is how the shipped installer's five one-second polls called a dying
    # service a started one. The installer's own liveness proof must wait for the manager to settle
    # and then report the failure.
    docker exec "$DYING" sh -c '
        printf "%s\n" \
            "[Unit]" "Description=MahBot" "StartLimitIntervalSec=300" "StartLimitBurst=5" \
            "[Service]" "Type=simple" "User=owner" "WorkingDirectory=/home/owner" \
            "ExecStart=/bin/false" "Restart=on-failure" "RestartSec=5" \
            "[Install]" "WantedBy=multi-user.target" \
            > /etc/systemd/system/mahbot.service
        systemctl daemon-reload
        systemctl enable mahbot.service
    '
    check "the planted unit loads" \
        docker exec "$DYING" systemd-analyze verify /etc/systemd/system/mahbot.service

    run_installer "$DYING" no
    check "the install reports the failure instead of success" test "$INSTALL_STATUS" -ne 0
    check "the refusal is not a timeout" test "$INSTALL_STATUS" -ne 124
    check_contains "it says the service did not come up" "$INSTALL_OUT" "did not come up under it"
    check "the manager agrees the unit is not running" \
        docker exec "$DYING" sh -c '! systemctl is-active --quiet mahbot.service'
}

leg_root() {
    leg_start root
    new_installed_container "$ROOT" "$BASE_IMAGE"
    if ! docker exec "$ROOT" env SUDO_FORCE_REMOVE=yes DEBIAN_FRONTEND=noninteractive \
        apt-get remove -y sudo >/dev/null 2>&1; then
        fail "sudo can be removed from the root container" "apt-get remove -y sudo failed"
    fi
    check "sudo is not installed" \
        docker exec "$ROOT" sh -c '! command -v sudo >/dev/null 2>&1'

    # The minimal server's shape: the one-liner run as root on a host with no sudo package. Root is
    # the privilege, so the unit is written, enabled and started — and the installer says what that
    # means for the service it just registered.
    say "Installing as root, with no sudo on the host."
    run_installer "$ROOT" no -u root -e HOME=/root
    check_eq "the install exits 0" 0 "$INSTALL_STATUS"
    check_contains "it names the service it registered" "$INSTALL_OUT" \
        "MahBot is registered as a system service at /etc/systemd/system/mahbot.service"
    check_contains "it says the service will run as root" "$INSTALL_OUT" \
        "the service will run as root as well; installing as an ordinary user is the recommended way"
    check_eq "the unit runs as root, as the install said" 'User=root' \
        "$(docker exec "$ROOT" grep '^User=' /etc/systemd/system/mahbot.service)"
    check "systemd-analyze verify accepts the unit" \
        docker exec "$ROOT" systemd-analyze verify /etc/systemd/system/mahbot.service
    check_eq "the unit is enabled" enabled \
        "$(docker exec "$ROOT" systemctl is-enabled mahbot.service)"
    check "the service is active" wait_active "$ROOT" 30
    check_empty "nobody is logged in" sessions_on "$ROOT"
    check_lacks "it prints no sudo command this host cannot run" "$INSTALL_OUT" "sudo systemctl"
}

leg_bad_unit() {
    leg_start bad-unit
    new_installed_container "$BAD_UNIT" "$BASE_IMAGE"

    # A unit that would not load is never left on the machine. A home the manager reads as a relative
    # path is a render it refuses, so the installer's own check must catch it — before anything is
    # written — and say why.
    say "Installing with a relative HOME, so the unit the installer renders is one systemd refuses."
    run_installer "$BAD_UNIT" no -e HOME=relative
    check "the install refuses the unit it rendered" test "$INSTALL_STATUS" -ne 0
    check "the refusal is not a timeout" test "$INSTALL_STATUS" -ne 124
    check_contains "it names the check that refused it" "$INSTALL_OUT" \
        "the service unit this install would write would not load"
    check_contains "it says nothing was registered" "$INSTALL_OUT" \
        "nothing was registered and /etc/systemd/system/mahbot.service was not changed"
    check "no unit was written" \
        docker exec "$BAD_UNIT" test ! -e /etc/systemd/system/mahbot.service
    check "nothing is enabled" \
        docker exec "$BAD_UNIT" sh -c '! systemctl is-enabled mahbot.service >/dev/null 2>&1'
}

leg_graphical() {
    leg_start graphical
    new_installed_container "$GRAPHICAL" "$DESKTOP_IMAGE"
    check "a display manager is present in the desktop image" \
        test -n "$(display_manager_on "$GRAPHICAL")"

    run_installer "$GRAPHICAL" yes -e XDG_CONFIG_HOME=/home/owner/.config-xdg
    check_eq "the install exits 0" 0 "$INSTALL_STATUS"
    check_contains "it names the login start" "$INSTALL_OUT" "MahBot starts when you log in"
    check_contains "it names the login entry path" "$INSTALL_OUT" \
        "The login entry is /home/owner/.config-xdg/autostart/mahbot.desktop"
    check_contains "it names how to ask for the service instead" "$INSTALL_OUT" \
        "MAHBOT_INSTALL_REGISTRATION=service"
    check "no unit was written" \
        docker exec "$GRAPHICAL" test ! -e /etc/systemd/system/mahbot.service
    check "nothing was enabled" \
        docker exec "$GRAPHICAL" sh -c '! systemctl is-enabled mahbot.service >/dev/null 2>&1'

    # The install handed its terminal to the product, which detaches into a session of its own and
    # writes the login entry at that first start. A headless container has no window to open, so the
    # start is repeated here to show the entry exists whether or not the first one stayed up; a
    # second copy refuses because the instance lock is held, which is expected and not asserted.
    say "Starting the product once as owner, so the entry it maintains is shown from a start of ours too."
    _prc=0
    docker exec -u owner -e HOME=/home/owner -w /home/owner \
        -e XDG_CONFIG_HOME=/home/owner/.config-xdg \
        "$GRAPHICAL" timeout 60 /home/owner/.local/bin/mahbot \
        >"$WORK/graphical-start.out" 2>&1 || _prc=$?
    say "That start exited $_prc (headless, so a window or a held lock is expected; the entry is what is asserted)."
    check "the entry is at the XDG config home" \
        docker exec "$GRAPHICAL" test -f /home/owner/.config-xdg/autostart/mahbot.desktop
    check "the home entry was not written" \
        docker exec "$GRAPHICAL" test ! -e /home/owner/.config/autostart/mahbot.desktop
    say "── /home/owner/.config-xdg/autostart/mahbot.desktop ──"
    docker exec "$GRAPHICAL" cat /home/owner/.config-xdg/autostart/mahbot.desktop
}

leg_dry_run() {
    leg_start dry-run
    new_installed_container "$DRYRUN" "$BASE_IMAGE"
    run_installer "$DRYRUN" no -e MAHBOT_INSTALL_DRY_RUN=1
    check_eq "the dry run exits 0" 0 "$INSTALL_STATUS"
    check_contains "it names the registration" "$INSTALL_OUT" \
        "registration: service (no display manager was found)"
    check_contains "it prints the unit it would write" "$INSTALL_OUT" \
        "WorkingDirectory=/home/owner"
    check "no unit was written" \
        docker exec "$DRYRUN" test ! -e /etc/systemd/system/mahbot.service
    check "no binary was installed" \
        docker exec "$DRYRUN" test ! -e /home/owner/.local/bin/mahbot
    check "no login entry was written" \
        docker exec "$DRYRUN" test ! -e /home/owner/.config/autostart/mahbot.desktop

    # The entry the product maintains is reported where it actually is, not at the path this shell's
    # config home happens to name.
    say "With the home entry in place and the config home redirected, the dry run names the entry on disk."
    write_login_entries "$DRYRUN" /home/owner/.config
    run_installer "$DRYRUN" no -e MAHBOT_INSTALL_DRY_RUN=1 -e XDG_CONFIG_HOME=/home/owner/.config-xdg
    check_eq "the redirected dry run exits 0" 0 "$INSTALL_STATUS"
    check_contains "it names the entry that is on disk" "$INSTALL_OUT" \
        "registered:   the login entry at /home/owner/.config/autostart/mahbot.desktop"
    check "it wrote nothing under the redirected config home" \
        docker exec "$DRYRUN" test ! -e /home/owner/.config-xdg/autostart/mahbot.desktop

    # A unit the host already has is kept, so the dry run must not advertise one it would not write.
    say "With a unit in place, the dry run says it would keep it and writes nothing."
    run_installer "$DRYRUN" no
    check_eq "the install that writes the unit exits 0" 0 "$INSTALL_STATUS"
    _hash=$(unit_hash "$DRYRUN")
    run_installer "$DRYRUN" no -e MAHBOT_INSTALL_DRY_RUN=1
    check_eq "the dry run with a unit in place exits 0" 0 "$INSTALL_STATUS"
    check_contains "it says it would keep the unit" "$INSTALL_OUT" \
        "is kept as it is; this run would write no unit"
    check_lacks "it prints no unit it would not write" "$INSTALL_OUT" "WorkingDirectory="
    check_eq "the unit is untouched" "$_hash" "$(unit_hash "$DRYRUN")"

    say "The same dry run on a graphical host names the login entry instead."
    new_installed_container "$DRYRUN_DESKTOP" "$DESKTOP_IMAGE"
    # A trailing separator names the same directory: the entry is printed the way the product computes
    # it, never with a doubled slash.
    run_installer "$DRYRUN_DESKTOP" no \
        -e MAHBOT_INSTALL_DRY_RUN=1 \
        -e XDG_CONFIG_HOME=/home/owner/.config-xdg/
    check_eq "the desktop dry run exits 0" 0 "$INSTALL_STATUS"
    check_contains "it names the login registration" "$INSTALL_OUT" \
        "registration: login (the display manager at"
    check_contains "it prints the entry it would write" "$INSTALL_OUT" \
        "login entry:  /home/owner/.config-xdg/autostart/mahbot.desktop"
    check "no unit was written on the desktop" \
        docker exec "$DRYRUN_DESKTOP" test ! -e /etc/systemd/system/mahbot.service
    check "no binary was installed on the desktop" \
        docker exec "$DRYRUN_DESKTOP" test ! -e /home/owner/.local/bin/mahbot
    check "no login entry was written on the desktop" \
        docker exec "$DRYRUN_DESKTOP" sh -c \
        'test ! -e /home/owner/.config-xdg/autostart/mahbot.desktop && test ! -e /home/owner/.config/autostart/mahbot.desktop'
}

run_leg() {
    if [ -n "$ONLY" ] && [ "$ONLY" != "$1" ]; then
        return 0
    fi
    "leg_$(printf '%s' "$1" | tr '-' '_')"
    LEGS_DONE="$LEGS_DONE $1"
    LEGS_COUNT=$((LEGS_COUNT + 1))
}

# ── The run ───────────────────────────────────────────────────────────────

say "MahBot install acceptance: release v$VERSION from $RELEASE_BASE"
say "installer under test: $INSTALLER"
say ""
# The display-manager evidence is read out of the installer, so an extraction that finds nothing would
# make the "no display manager is present" assertion vacuous rather than failing it.
if [ -z "$DM_UNITS" ] || [ -z "$DM_DIRS" ]; then
    fail "the display-manager evidence is readable out of $INSTALLER" \
        "no unit name or no directory was found in have_display_manager"
fi
ensure_images

for _leg in $LEGS; do
    run_leg "$_leg"
done
