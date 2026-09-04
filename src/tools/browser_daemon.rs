//! chrome-use daemon health monitoring and bounded auto-recovery.
//!
//! The chrome-use CLI talks to a per-session background daemon that drives
//! Chrome through the extension relay. When the daemon or relay dies, CLI
//! commands hang inside its own 5-retry loop (~152 s) instead of failing fast.
//! This module classifies health from a daemon-free `status` snapshot
//! (extension disabled, relay down, host missing, …) and auto-restarts the
//! daemon with bounded backoff and thrash protection. Real wedge detection is
//! per-call: a browser command that fails with the daemon-unavailable signature
//! marks the daemon unhealthy and wakes the watchdog, which recovers from that
//! stored classification — the daemon-free status cannot see a wedged daemon,
//! so the watchdog never re-evaluates over a fail-fast classification. It also
//! owns the chrome-use CLI invocation primitives (binary name, env setup,
//! `--version` check) so the browser tool depends on this module and not vice
//! versa.
//!
//! Verified tab-sweep: mahbot-owned session tab groups (`link-enricher-*`) are
//! closed through the CLI and verified by round-over-round re-enumeration. A
//! bare `session stop` is not enough — it SIGTERMs the daemon, waits ~1 s, then
//! force-kills, so when the extension relay is slow or wedged the daemon's
//! graceful tab close cannot finish inside that window and the scratch tab is
//! orphaned forever (no other mechanism ever reclaims it).
//!
//! Trade-offs:
//! - A genuine daemon wedge surfaces on the first real browser call, which pays
//!   the CLI's ~152 s internal retry before fail-fast marks it unhealthy (worst
//!   case, rare, and self-healing — the restart clears the wedge).
//! - `daemon restart` destroys all session state; recovery guidance notes that
//!   existing browser sessions are reset.

use crate::util::UnwrapPoison;
use futures_util::future::join_all;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::process::Command;
use tracing::{debug, error, info, warn};

// ── Bounds (pinned for deterministic recovery) ────────────────────────────
/// How long a CLI health/sweep command may take before it is considered
/// wedged. A healthy daemon answers in milliseconds; a wedged one hangs for
/// the CLI's internal 45s read timeouts × 5 retries.
const CLI_TIMEOUT: Duration = Duration::from_secs(8);
/// Cache TTL for a healthy evaluation (fresh enough for per-call checks).
const HEALTH_TTL: Duration = Duration::from_secs(10);
/// Longer TTL for a confirmed-down result, so repeated browser calls fail fast
/// instead of re-evaluating on every invocation.
const UNHEALTHY_TTL: Duration = Duration::from_mins(1);
/// Watchdog cadence between automatic health evaluations.
const WATCHDOG_INTERVAL: Duration = Duration::from_secs(30);
/// How often the watchdog re-verifies CLI presence on hosts where it was
/// found — the binary can be uninstalled while the daemon runs, but checking
/// every watchdog interval would spawn `--version` needlessly.
const CLI_RECHECK: Duration = Duration::from_mins(5);
/// Consecutive definitive-missing CLI probes before the watchdog stands down —
/// a single transient probe failure (spawn EAGAIN/EMFILE under process
/// pressure) must not take the watchdog out of service.
const CLI_MISSING_THRESHOLD: u32 = 2;
/// Consecutive failed restarts before auto-recovery halts (thrash protection).
const MAX_RESTART_ATTEMPTS: u32 = 3;
/// Sustained-health window: the restart-attempt counter resets only after the
/// daemon-free status has been healthy for this long (≥2 watchdog intervals).
/// A transient healthy right after a restart must not reopen a bounded cycle
/// early, or a runaway restart loop can never trip the halt. Daemon-free
/// status cannot see wedges — for a persistent wedge the budget keeps
/// resetting between sparse real calls, so the halt engages only for
/// service-level causes (accepted with per-call wedge detection).
const SUSTAINED_HEALTHY_WINDOW: Duration = Duration::from_mins(1);
/// Backoff between restart attempts (30s → 2min → 10min).
const RESTART_BACKOFF: [Duration; 3] = [
    Duration::from_secs(30),
    Duration::from_mins(2),
    Duration::from_mins(10),
];
/// Cooldown after the max restart attempts, before a fresh bounded cycle.
const HALT_COOLDOWN: Duration = Duration::from_mins(30);
/// How long recovery waits for the extension relay to republish after a
/// `daemon restart` on a relay-drop — the MV3 service worker revives on its
/// keepalive (~30 s) and only then writes the relay endpoint back.
const RELAY_REVIVE_WAIT: Duration = Duration::from_secs(40);

// ── Verified-close sweep bounds (pinned for deterministic recovery) ──────
/// Total budget for one sweep invocation, starting before the service-state
/// skip gate. Every CLI call checks the deadline before
/// spawning (one call may overshoot by at most [`CLI_TIMEOUT`] — the
/// in-flight bound). On expiry the sweep defers: leftover tabs are retried by
/// the next sweep/startup (self-healing), never a permanent orphan.
const SWEEP_TOTAL_BUDGET: Duration = Duration::from_secs(15);
/// Convergence rounds before a sweep gives up for this invocation. The budget
/// is the hard cap; this only bounds the number of enumerate/close/stop cycles
/// (a healthy host converges in 3 rounds; a retried failed close needs 4–5).
const SWEEP_MAX_ROUNDS: u32 = 5;

/// Classified cause for a failed health check. Drives cause-specific warnings
/// and decides whether auto-recovery can help at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeFailure {
    /// chrome-use extension or native host is not installed.
    NotInstalled,
    /// Native host manifest present but launcher/target broken.
    HostBroken,
    /// Extension installed but disabled (Chrome reports disable reasons).
    ExtensionDisabled,
    /// Extension enabled but the relay is down — transient, self-heals.
    RelayDown,
    /// The session's tab lost its debugger attach (orphaned) — the daemon and
    /// relay are up; only closing the tab by hand unblocks the session.
    UnreachableTab,
    /// The daemon socket hung or errored (daemon-side wedge).
    DaemonWedge,
}

/// Result of a health evaluation: healthy, or down with a classified cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeOutcome {
    Healthy,
    Down(ProbeFailure),
}

impl ProbeOutcome {
    fn is_healthy(self) -> bool {
        matches!(self, ProbeOutcome::Healthy)
    }

    fn failure(self) -> Option<ProbeFailure> {
        match self {
            ProbeOutcome::Healthy => None,
            ProbeOutcome::Down(f) => Some(f),
        }
    }
}

impl ProbeFailure {
    /// Causes a daemon restart cannot fix — reported with their concrete fix
    /// and never consume restart attempts.
    fn is_unfixable(self) -> bool {
        matches!(
            self,
            ProbeFailure::NotInstalled
                | ProbeFailure::HostBroken
                | ProbeFailure::ExtensionDisabled
                | ProbeFailure::UnreachableTab
        )
    }
}

#[derive(Default)]
struct DaemonHealth {
    healthy: Option<bool>,
    last_probe: Option<Instant>,
    restart_attempts: u32,
    next_restart_at: Option<Instant>,
    halted: bool,
    halted_until: Option<Instant>,
    /// Last classified failure — surfaces the cause in LLM-facing
    /// messages and drives transition-based warning logging.
    last_failure: Option<ProbeFailure>,
    /// The failure cause the last transition-based warning named — reset on
    /// recovery so the same cause warns again after a healthy spell.
    last_cause_warned: Option<ProbeFailure>,
    /// Start of the current sustained-healthy streak — the restart budget
    /// resets only once this reaches [`SUSTAINED_HEALTHY_WINDOW`]; any failure
    /// aborts the streak.
    healthy_since: Option<Instant>,
}

/// Decision from [`DaemonHealth::gate_restart`].
#[derive(Debug, PartialEq, Eq)]
enum RestartGate {
    /// Attempt N is allowed now.
    Allowed(u32),
    /// Backoff between attempts not yet elapsed.
    Backoff,
    /// Thrash halt cooldown in progress.
    Cooldown,
    /// Max consecutive attempts exhausted — auto-recovery just halted.
    Halted,
}

impl DaemonHealth {
    /// Decide whether a restart attempt is allowed, updating the bookkeeping
    /// in place. Pure (no I/O) so the bounded state machine is unit-testable.
    fn gate_restart(&mut self, now: Instant) -> RestartGate {
        if self.halted {
            if self.halted_until.is_some_and(|until| now < until) {
                return RestartGate::Cooldown;
            }
            // Cooldown expired — reset and allow a fresh bounded cycle.
            self.halted = false;
            self.restart_attempts = 0;
            self.halted_until = None;
            self.next_restart_at = None;
        }
        // Backoff is checked before the attempt cap, so the final 10-min grace
        // after the last restart is honored before the halt fires.
        if self.next_restart_at.is_some_and(|next| now < next) {
            return RestartGate::Backoff;
        }
        if self.restart_attempts >= MAX_RESTART_ATTEMPTS {
            self.halted = true;
            self.halted_until = Some(now + HALT_COOLDOWN);
            return RestartGate::Halted;
        }
        let attempt = self.restart_attempts + 1;
        self.restart_attempts = attempt;
        self.next_restart_at =
            Some(now + RESTART_BACKOFF[(attempt as usize - 1).min(RESTART_BACKOFF.len() - 1)]);
        RestartGate::Allowed(attempt)
    }

    /// Apply a health observation. A healthy result opens the sustained-healthy
    /// window ([`SUSTAINED_HEALTHY_WINDOW`]); the restart budget resets only
    /// after the window completes, so a transient healthy right after a restart
    /// (the post-restart verification — `seed_window = false` — or a single
    /// watchdog interval) cannot reopen a bounded cycle early. Any failure
    /// aborts the window.
    fn apply_outcome(&mut self, outcome: ProbeOutcome, now: Instant, seed_window: bool) {
        let healthy = outcome.is_healthy();
        if healthy {
            if self
                .healthy_since
                .is_some_and(|since| now.duration_since(since) >= SUSTAINED_HEALTHY_WINDOW)
            {
                // Sustained health — open a fresh bounded cycle.
                self.restart_attempts = 0;
                self.next_restart_at = None;
                self.halted = false;
                self.halted_until = None;
                self.last_cause_warned = None;
            }
            if seed_window && self.healthy_since.is_none() {
                self.healthy_since = Some(now);
            }
        } else {
            self.healthy_since = None;
        }
        // A cause change never resets the restart budget — flapping causes (e.g.
        // RelayDown ↔ DaemonWedge) must not evade the attempt halt. Only
        // sustained health (or the cooldown expiry in gate_restart) opens a
        // fresh cycle.
        self.last_failure = outcome.failure();
        self.healthy = Some(healthy);
        self.last_probe = Some(now);
    }
}

static HEALTH: OnceLock<Mutex<DaemonHealth>> = OnceLock::new();
static WAKE: OnceLock<tokio::sync::Notify> = OnceLock::new();

fn health() -> &'static Mutex<DaemonHealth> {
    HEALTH.get_or_init(|| Mutex::new(DaemonHealth::default()))
}

fn wake() -> &'static tokio::sync::Notify {
    WAKE.get_or_init(tokio::sync::Notify::new)
}

/// Detect the daemon-unavailable signature chrome-use produces when its
/// background daemon is dead or wedged: the CLI hangs in its own 5-retry loop
/// (EAGAIN / "Resource temporarily unavailable") and eventually reports
/// "daemon may be busy or unresponsive". Also covers the 1.5.8x-era texts
/// (stuck-daemon auto-stop, disappeared daemon endpoint, failed auto-launch).
pub(crate) fn is_daemon_unavailable_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("resource temporarily unavailable")
        || lower.contains("os error 35")
        || lower.contains("os error 11")
        || lower.contains("daemon may be busy or unresponsive")
        || lower.contains("session unresponsive")
        || lower.contains("cdp session is unresponsive after attaching")
        || lower.contains("daemon failed to start")
        || lower.contains("auto-launch failed")
        // Colon form only — "failed to connect to <host>" is a page-level
        // navigation failure, not a daemon socket problem.
        || lower.contains("failed to connect:")
}

/// Relay-side failure signature — the daemon is alive but cannot drive Chrome
/// through the extension relay. Distinct from a daemon wedge (restart clears a
/// wedge; a relay drop needs the extension to republish, then self-heals).
fn is_relay_unavailable_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("relay isn't connected")
        || lower.contains("relay is not")
        || lower.contains("relay dropped")
        || lower.contains("relay down")
        || lower.contains("could not drive your chrome")
}

/// Classify a fast CLI failure text into a health cause. Unreachable-tab
/// errors are their own state — the daemon and relay are up, only the
/// session's tab is orphaned, so recovery must NOT run for them. The signature
/// also appears wrapped inside the auto-connect envelope ('Could not drive your
/// Chrome…') and the daemon wrapper ('Auto-launch failed'), so it wins over
/// both. The relay signature is more specific than the daemon wrapper it is
/// wrapped in — both the watchdog and the fail-fast path must agree on the
/// cause.
fn classify_failure_text(msg: &str) -> Option<ProbeFailure> {
    if is_unreachable_tab_error(msg) {
        Some(ProbeFailure::UnreachableTab)
    } else if is_relay_unavailable_error(msg) {
        Some(ProbeFailure::RelayDown)
    } else if is_daemon_unavailable_error(msg) {
        Some(ProbeFailure::DaemonWedge)
    } else {
        None
    }
}

/// Stable error-envelope `code` values (v1.5.78+) that are unambiguously
/// daemon-side. The coarse `connection_failed` code is shared with page-level
/// navigation failures (the CLI classifies any "connection" text as such), so
/// the message-text matcher stays the source of truth for those.
pub(crate) fn is_daemon_unavailable_code(code: Option<&str>) -> bool {
    matches!(code, Some("browser_not_launched"))
}

/// Get the platform-appropriate chrome-use binary name.
pub(crate) const fn browser_bin() -> &'static str {
    if cfg!(target_os = "windows") {
        "chrome-use.exe"
    } else {
        "chrome-use"
    }
}

/// Result of a CLI availability probe — distinguishes definitive absence
/// from transient failures so callers never report "not installed" for a
/// resource-exhaustion or wedged-binary failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CliStatus {
    /// `chrome-use --version` ran successfully.
    Available,
    /// Binary definitively absent (not on PATH, not in common install
    /// locations, or the resolved binary vanished).
    Missing,
    /// Probe failed — the binary is present but could not be confirmed
    /// working. Structured so user messages distinguish a transient spawn
    /// failure from a deterministic broken-install or wedge.
    Transient(CliProbeFailure),
}

/// Why a CLI probe of a present binary failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CliProbeFailure {
    /// Spawn failed (EAGAIN/EMFILE/ENOMEM under process pressure, …) —
    /// temporary; retry rather than standing down.
    Spawn(String),
    /// `--version` ran but exited non-zero — the install is broken.
    BadVersion(String),
    /// The bounded probe timed out — the binary may be wedged.
    Timeout,
}

impl std::fmt::Display for CliProbeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliProbeFailure::Spawn(reason) => write!(f, "spawn failed ({reason})"),
            CliProbeFailure::BadVersion(status) => write!(f, "--version check failed ({status})"),
            CliProbeFailure::Timeout => write!(f, "probe timed out"),
        }
    }
}

/// GitHub repo whose releases host the chrome-use binary (single source of truth).
const CHROME_USE_RELEASE_REPO: &str = "leeguooooo/chrome-use";

/// Timeout for a chrome-use release download (archives are ~9 MB).
const CHROME_USE_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(300);

/// Timeout for resolving the latest chrome-use release tag.
const CHROME_USE_RELEASE_TIMEOUT: Duration = Duration::from_secs(30);

/// Timeout for a chrome-use native-host registration subprocess (`extension
/// install`): a hung registration must not block the Support agent's tool call
/// indefinitely.
const CHROME_USE_INSTALL_TIMEOUT: Duration = Duration::from_secs(120);

/// Install hint for the definitive not-found case — appended to every
/// user-facing message that names the chrome-use CLI as missing.
pub(crate) const CHROME_USE_INSTALL_HINT: &str = "To install it, the Support agent can run the \
     user-consented `install_chrome_use` tool (a direct, checksum-verified release download).";

/// Resolved absolute path of the chrome-use binary (managed dir first, then
/// PATH, then common install locations), cached after the first probe.
/// Re-resolves only when the cached path vanished or was never found, so a
/// late installation is picked up by the next probe.
static CLI_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();

/// Absolute path of the chrome-use binary, or `None` when definitively not
/// installed. Spawns must go through this (not the bare name) so PATH
/// mutations and non-PATH install locations cannot break them.
pub(crate) fn cli_path() -> Option<PathBuf> {
    let mut cache = CLI_PATH
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_poison();
    // Same executability predicate as the resolver, so a cached binary that
    // loses its execute bit mid-run is re-resolved (a non-executable path
    // would otherwise pin every probe in a permanent PermissionDenied).
    if let Some(path) = cache.as_ref().filter(|p| crate::util::is_executable(p)) {
        return Some(path.clone());
    }
    let found = find_cli_binary();
    cache.clone_from(&found);
    found
}

/// Clear the cached CLI path so a relocated binary is re-resolved on the next
/// probe — after a first install the binary lands at a fresh managed-dir path
/// that a stale cache would not see.
fn invalidate_cli_path() {
    *CLI_PATH
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_poison() = None;
}

/// Stable mahbot-owned install dir (`<storage root>/bin`) for the chrome-use
/// binary. Probed FIRST by [`find_cli_binary`] on every OS — on Windows this
/// is the only probe that reliably finds mahbot's own install — so the
/// first-install location and the auto-update swap location are always the
/// same path.
#[must_use]
fn managed_bin_dir() -> Option<PathBuf> {
    crate::config::CONFIG
        .try_storage_root()
        .map(|r| r.join("bin"))
}

/// Locate the chrome-use binary: the mahbot-managed install dir first
/// (`<storage root>/bin`, always resolved by [`managed_bin_dir`]), then a PATH
/// lookup, then the common install locations the old curl installer targeted
/// (`~/.local/bin`, `~/.cargo/bin`, `/usr/local/bin`, `/opt/homebrew/bin` —
/// the first two in the installer's order so a fresh curl install wins over a
/// stale cargo one). The managed dir is probed FIRST on every OS (on Windows
/// it is the only reliable mahbot-install probe, since the PATH there may not
/// include it) so the first-install location and the auto-update swap
/// location are always the same path. Home resolution
/// goes through [`crate::util::cargo_bin_dir`] and `directories::UserDirs`
/// (not `$HOME`) so the fallback still works on HOME-less hosts (docker); when
/// `CARGO_HOME` is set the literal `~/.cargo/bin` is probed too
/// (belt-and-suspenders, mirroring the shell module's
/// `extra_shell_path_prefixes`). Candidates must be executable (`execvp` would
/// skip a non-executable PATH entry, so we do too).
fn find_cli_binary() -> Option<PathBuf> {
    let name = browser_bin();
    if let Some(dir) = managed_bin_dir() {
        let candidate = dir.join(name);
        if crate::util::is_executable(&candidate) {
            return Some(candidate);
        }
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join(name);
            if crate::util::is_executable(&candidate) {
                return Some(candidate);
            }
        }
    }
    if !cfg!(target_os = "windows") {
        let home = directories::UserDirs::new().map(|d| d.home_dir().to_path_buf());
        let literal_cargo_bin = match (std::env::var_os("CARGO_HOME"), home.as_deref()) {
            (Some(cargo_home), Some(h)) if !cargo_home.is_empty() => Some(h.join(".cargo/bin")),
            _ => None,
        };
        for base in [
            home.as_deref().map(|h| h.join(".local/bin")),
            crate::util::cargo_bin_dir(),
            literal_cargo_bin,
            Some(PathBuf::from("/usr/local/bin")),
            Some(PathBuf::from("/opt/homebrew/bin")),
        ]
        .into_iter()
        .flatten()
        {
            let candidate = base.join(name);
            if crate::util::is_executable(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

/// Classify a `--version` spawn failure: only a genuinely missing binary
/// (`NotFound`) is definitive absence; every other spawn error (EAGAIN,
/// EMFILE, ENOMEM, …) is a transient failure.
fn classify_spawn_error(e: &std::io::Error) -> CliStatus {
    if e.kind() == std::io::ErrorKind::NotFound {
        CliStatus::Missing
    } else {
        debug!("chrome-use CLI probe spawn failed: {e}");
        CliStatus::Transient(CliProbeFailure::Spawn(e.to_string()))
    }
}

/// Probe the chrome-use CLI: run `--version` via the resolved absolute path,
/// bounded by [`CLI_TIMEOUT`], kill-on-drop, with the no-update-check browser
/// env. Only definitive absence reports [`CliStatus::Missing`]; spawn errors,
/// timeouts, and non-zero exits are [`CliStatus::Transient`].
pub(crate) async fn cli_probe() -> CliStatus {
    let Some(path) = cli_path() else {
        return CliStatus::Missing;
    };
    let mut cmd = Command::new(&path);
    ensure_browser_env(&mut cmd);
    cmd.arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let status = match tokio::time::timeout(CLI_TIMEOUT, cmd.status()).await {
        Ok(Ok(status)) => status,
        Ok(Err(e)) => return classify_spawn_error(&e),
        Err(_) => {
            debug!("chrome-use CLI probe timed out");
            return CliStatus::Transient(CliProbeFailure::Timeout);
        }
    };
    if status.success() {
        CliStatus::Available
    } else {
        debug!("chrome-use CLI probe failed: --version exited with {status}");
        CliStatus::Transient(CliProbeFailure::BadVersion(status.to_string()))
    }
}

/// Parse a chrome-use release version (`v1.5.100` or `1.5.100`) into a semver.
/// GitHub tags are `v`-prefixed; the CLI's `--version` output is bare — both
/// flow through here.
fn parse_release_version(s: &str) -> Option<semver::Version> {
    semver::Version::parse(s.strip_prefix('v').unwrap_or(s)).ok()
}

/// Extract the chrome-use version from `--version` stdout. The banner carries
/// extra lines (bug-report URL etc.), so scan every whitespace token for the
/// first parseable semver rather than assuming a position.
fn parse_cli_version(stdout: &str) -> Option<semver::Version> {
    stdout.split_whitespace().find_map(parse_release_version)
}

/// Release-asset platform tag for chrome-use archives, e.g. `darwin-arm64`
/// or `linux-musl-x64`. `None` on platform/arch combos the vendor does not
/// publish. Testable pure function; [`release_asset_name`] wraps it with
/// the compile-time target triple.
#[must_use]
fn release_asset_platform(os: &str, arch: &str, musl: bool) -> Option<String> {
    let asset = match (os, arch, musl) {
        ("macos", "x86_64", _) => "darwin-x64",
        ("macos", "aarch64", _) => "darwin-arm64",
        ("linux", "x86_64", false) => "linux-x64",
        ("linux", "aarch64", false) => "linux-arm64",
        ("linux", "x86_64", true) => "linux-musl-x64",
        ("linux", "aarch64", true) => "linux-musl-arm64",
        ("windows", "x86_64", _) => "win32-x64",
        _ => return None,
    };
    Some(asset.to_string())
}

/// Asset platform tag for THIS build, or `Err` naming the unsupported
/// platform. Uses the compile-time target triple (`cfg!`) so it is correct
/// regardless of the runtime host; on Linux, musl is detected by the presence
/// of a musl loader in `/lib` (mirrors the vendor installer — no `ldd`).
fn release_asset_name() -> Result<String, String> {
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        return Err(format!(
            "unsupported chrome-use platform: {}",
            std::env::consts::OS
        ));
    };
    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        return Err(format!(
            "unsupported chrome-use arch: {}",
            std::env::consts::ARCH
        ));
    };
    let musl = os == "linux"
        && (Path::new("/lib/ld-musl-x86_64.so.1").exists()
            || Path::new("/lib/ld-musl-aarch64.so.1").exists());
    release_asset_platform(os, arch, musl)
        .ok_or_else(|| format!("chrome-use has no release asset for {os}-{arch}"))
}

/// `(hash, filename)` from a `.sha256` sidecar body (`"<hash>  <filename>"`).
/// The hash must be 64 hex chars (normalized to lowercase to match the
/// computed digest) and a filename must be present — a bare-hash sidecar is
/// rejected so a cross-paired sidecar can never verify another asset.
#[must_use]
fn parse_sha256_sidecar(body: &str) -> Option<(String, String)> {
    let mut tokens = body.split_whitespace();
    let hash = tokens.next()?;
    let filename = tokens.next()?.to_string();
    let valid_hash = hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit());
    valid_hash.then(|| (hash.to_ascii_lowercase(), filename))
}

/// Parse the local chrome-use version from `--version` stdout, bounded by
/// [`CLI_TIMEOUT`]. `None` when the CLI is missing, times out, exits non-zero,
/// or its output is not a parseable semver — callers distinguish "not
/// installed" (via [`cli_path`]) from "unparseable version" (proceed assuming
/// outdated).
async fn cli_version() -> Option<semver::Version> {
    let path = cli_path()?;
    let mut cmd = Command::new(&path);
    ensure_browser_env(&mut cmd);
    cmd.arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let out = tokio::time::timeout(CLI_TIMEOUT, cmd.output())
        .await
        .ok()?
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_cli_version(&String::from_utf8_lossy(&out.stdout))
}

/// Set HOME, `CHROMIUM_FLAGS`, and default timeout env vars on the command
/// so that the Chromium spawned by chrome-use works in service/docker
/// environments.
pub(crate) fn ensure_browser_env(cmd: &mut Command) {
    if std::env::var_os("HOME").is_none() {
        cmd.env("HOME", "/tmp");
    }
    // Suppress Chromium's "--enable-crashes-dialog" and GPU-related flags
    // that cause issues in headless/service environments.
    if std::env::var_os("CHROMIUM_FLAGS").is_none() {
        cmd.env(
            "CHROMIUM_FLAGS",
            "--no-first-run --no-default-browser-check --disable-gpu",
        );
    }
    // Default 15-second timeout for all chrome-use actions (including
    // `wait --text` which would otherwise block much longer).
    cmd.env("AGENT_BROWSER_DEFAULT_TIMEOUT", "15000");
    // 5-minute idle timeout — the chrome-use daemon still stops after 5 idle
    // minutes, but chrome-use ≥1.5.101 PRESERVES external Chrome tabs on idle
    // (only an explicit close/session stop cleans them up), so mahbot closes
    // agent-opened sessions explicitly at run end.
    cmd.env("AGENT_BROWSER_IDLE_TIMEOUT_MS", "300000");
    // Enable human-like interaction speed for bot-detection avoidance.
    // chrome-use supports the same env vars as agent-browser for backward
    // compatibility.
    cmd.env("AGENT_BROWSER_HUMANIZE", "human");
    // Keep the upgrade-available banner out of every command's stderr.
    cmd.env("CHROME_USE_NO_UPDATE_CHECK", "1");
    cmd.env("AGENT_BROWSER_NO_UPDATE_CHECK", "1");
    // The watchdog owns recovery. Without these, a browser command issued while
    // the relay is down makes the CLI kill session daemons / the native host
    // and wait up to 45s for a relay revive — racing the watchdog's own
    // cause-aware recovery and turning a health check into a 45s stall.
    cmd.env("AGENT_BROWSER_NO_AUTO_RECONNECT", "1");
    cmd.env("AGENT_BROWSER_RELAY_REVIVE_SECS", "0");
}

/// Spawn a chrome-use CLI call with the browser env, `--json`, and an optional
/// `--session`, bounded by [`CLI_TIMEOUT`] — a wedged daemon hangs inside the
/// CLI's own ~152 s retry loop, so every call must be bounded.
async fn run_cli_bounded(args: &[&str], session: Option<&str>) -> Option<std::process::Output> {
    let mut cmd = Command::new(cli_path()?);
    ensure_browser_env(&mut cmd);
    cmd.args(args).arg("--json");
    if let Some(session) = session {
        cmd.args(["--session", session]);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    cmd.kill_on_drop(true);
    tokio::time::timeout(CLI_TIMEOUT, cmd.output())
        .await
        .ok()?
        .ok()
}

/// Run a chrome-use CLI command with `--json` (and optional `--session`),
/// bounded by [`CLI_TIMEOUT`] — a wedged daemon would otherwise hang the
/// CLI's own ~152 s retry loop inside the health/recovery path. `Ok(value)` on
/// success; `Err(Some(msg))` when the CLI answered with a structured error
/// (the message survives for signature detection); `Err(None)` on
/// timeout/spawn/parse failure.
async fn run_cli_json_opt(args: &[&str], session: Option<&str>) -> Result<Value, Option<String>> {
    let out = run_cli_bounded(args, session).await.ok_or(None)?;
    if !out.status.success() {
        return Err(extract_error(&out.stdout));
    }
    let v: Value = serde_json::from_slice(&out.stdout).map_err(|_| None)?;
    if envelope_verdict(&v) != Some(true) {
        return Err(extract_error(&out.stdout));
    }
    Ok(v)
}

/// Non-session variant for daemon-free commands (`status`, `extension
/// status`): errors are dropped — callers treat an unavailable status as
/// healthy (the per-call fail-fast path still catches wedges).
async fn run_cli_json(args: &[&str]) -> Option<Value> {
    run_cli_json_opt(args, None).await.ok()
}

/// Daemon-free service-state snapshot: `status --json` classifies extension /
/// native-host / relay problems without spawning a daemon or tab. Returns a
/// classified failure when the service is unusable; `None` when the status is
/// unavailable (old CLI or broken binary) or it looks healthy — wedge
/// detection then relies entirely on the per-call fail-fast path.
async fn service_state() -> Option<ProbeFailure> {
    let status = run_cli_json(&["status"]).await?;
    classify_service_state(&status).await
}

/// Classify a `status --json` snapshot (see [`service_state`]). Note:
/// pre-1.5.86 CLIs whose `status` lacks extension data fall through to `None`
/// (healthy).
async fn classify_service_state(status: &Value) -> Option<ProbeFailure> {
    let ext = status.get("data")?.get("extension")?;
    if ext.get("hostInstalled").and_then(Value::as_bool) == Some(false) {
        return Some(ProbeFailure::NotInstalled);
    }
    if ext.get("hostHealthy").and_then(Value::as_bool) == Some(false) {
        return Some(ProbeFailure::HostBroken);
    }
    if ext.get("relayUp").and_then(Value::as_bool) == Some(false) {
        // Distinguish extension-disabled (restart cannot fix) from a transient
        // relay drop (self-heals). Key on the extension's disable reasons, not
        // the unreliable active-bit signal.
        return Some(if extension_disabled().await {
            ProbeFailure::ExtensionDisabled
        } else {
            ProbeFailure::RelayDown
        });
    }
    None
}

/// Whether the chrome-use extension is disabled, from `extension status --json`
/// (daemon-free; reads Chrome's Secure Preferences). Non-empty `disableReasons`
/// means the extension is genuinely disabled.
async fn extension_disabled() -> bool {
    let Some(status) = run_cli_json(&["extension", "status"]).await else {
        return false; // Unknown → treat as a transient relay drop, not disabled.
    };
    status
        .get("data")
        .and_then(|d| d.get("chromeExtension"))
        .and_then(|c| c.get("disableReasons"))
        .and_then(Value::as_array)
        .is_some_and(|reasons| !reasons.is_empty())
}

/// Classify daemon health from the daemon-free `status` snapshot. Wedges are
/// invisible to this check by design — a real browser call that fails with the
/// daemon-unavailable signature marks the daemon unhealthy via [`note_unhealthy`]
/// (fail-fast) and wakes the watchdog, which recovers from that stored cause.
/// A healthy snapshot also drives the extension-skew advisory (single `status`
/// spawn per evaluation).
async fn evaluate_health() -> ProbeOutcome {
    let Some(status) = run_cli_json(&["status"]).await else {
        // Status unavailable (old CLI or broken binary) — treat as healthy;
        // the per-call fail-fast path still catches wedges.
        return ProbeOutcome::Healthy;
    };
    if let Some(failure) = classify_service_state(&status).await {
        return ProbeOutcome::Down(failure);
    }
    advise_extension_skew(&status);
    ProbeOutcome::Healthy
}

/// One tab in a session's tab group, from `tab list --json`. The `active` flag
/// is deliberately not tracked: a fresh daemon pins adopted leftovers exactly
/// like its own scratch, so it cannot tell the two apart (see the pinned
/// chrome-use behaviors below). Identity is `target_id` — the stable CDP id
/// that survives daemon restarts — while `tab_id` (`t<N>`) is the ref the
/// `close` command resolves.
struct SweepTab {
    tab_id: String,
    target_id: String,
}

// chrome-use CLI behaviors this sweep relies on (live-verified against
// 1.5.100; note the installed CLI auto-updates past that version, and that
// ≥1.5.101 idle keeps external tabs alive — the explicit close/stop the sweep
// uses still cleans up, so the verified behaviors below are unchanged):
// - `tab list --session <name>` enumerates only that session's tab group (the
//   relay scopes `Target.getTargets` per announced group, issue #40). When the
//   session has no daemon the CLI spawns one: an empty group makes it create a
//   fresh scratch tab; a non-empty group makes it ADOPT the existing tabs
//   without marking them created (`created_targets` stays empty).
// - Both the created scratch and adopted leftovers are pinned, so `active: true`
//   does NOT identify the daemon's own tab — the sweep tracks its own scratch
//   by stable `targetId` instead.
// - `close <ref>` closes one tab through the relay; it refuses to close the
//   last tab of a session ("Cannot close the last tab"), so the sweep creates
//   its own scratch first to keep the count ≥ 2 until every listed tab is
//   closed. The daemon discards the closeTarget result, so even a successful
//   JSON response is not proof of closure — only re-enumeration is, and a
//   failed close REAPPEARS in the next same-daemon `tab list` (resync adopts
//   still-open tabs again), which is the convergence loop.
// - `session stop` SIGTERMs the daemon (its shutdown handler closes its
//   created tabs best-effort through the relay), waits ≤ 1 s, then SIGKILLs.
//   Its exit code / JSON success are NOT proof of closure, and adopted tabs
//   are never closed at shutdown.
//
// Residual limits (accepted): the sweep's scratch tab is about:blank and the
// extension refuses to re-attach `about:` URLs (its `eligible()`/`SKIP_URL`
// filter). An orphan that lost its attach while the daemon kept a stale
// binding (relay blip, kill during an outage) fails every command with the
// unreachable-tab signatures — the sweep logs the 'close it by hand in
// Chrome' signal and keeps retrying; live orphans whose attach survived
// heal automatically. An orphan the extension fully dropped (Chrome
// service-worker restart unmarks ineligible about: tabs and never re-attaches
// them; the relay's group is fed only by attach announcements) is invisible
// to every CLI path: `tab list` succeeds with only the fresh scratch and the
// sweep converges to Clean with no log. That case is undetectable by design —
// no CLI path can enumerate a tab the extension no longer announces; it stays
// in Chrome until closed by hand. Dead-daemon link-enricher orphans are
// similarly not enumerable (their session names are per-message and the
// daemon inventory drops dead pids) — documented residual.
/// Close every tab in a mahbot-owned session's tab group except the sweep's own
/// scratch, verifying closure by round-over-round re-enumeration. Shared by the
/// startup sweep and the link-enricher per-fetch close.
pub(crate) async fn sweep_session(name: &str) {
    if !is_mahbot_session_name(name) {
        warn!(
            session = name,
            "tab sweep refused: not a mahbot-owned session (user/default/other-agent sessions are never touched)"
        );
        return;
    }
    // Skip on known service outage: no close is possible while the relay is
    // down, and every CLI call would cost the full step timeout for nothing.
    // Tabs stay until the browser is reachable again (next sweep/startup). The
    // deadline starts before the skip gate so the gate counts against the
    // total budget.
    let deadline = Instant::now() + SWEEP_TOTAL_BUDGET;
    if let Some(failure) = service_state().await {
        debug!(
            session = name,
            ?failure,
            "tab sweep skipped — browser service unavailable"
        );
        return;
    }
    // The sweep's own scratch tab — the ONE tab it creates via `tab new` and
    // tracks by stable targetId; everything else in the group is a leftover
    // that must be closed. `stopped` marks the round after a daemon stop: the
    // enumeration then spawned a fresh daemon, so a lone tab is provably that
    // daemon's own scratch (clean) unless it is our tracked scratch that
    // survived the stop (close it again — it was adopted, not owned).
    let mut scratch: Option<String> = None;
    let mut stopped = false;
    for _round in 1..=SWEEP_MAX_ROUNDS {
        if Instant::now() >= deadline {
            break;
        }
        let Some(tabs) = session_tab_list(name, deadline).await else {
            return; // warning already emitted by the enumerator
        };
        if tabs.is_empty() {
            // Live daemon whose tabs were all closed externally — nothing to
            // close; stop it so the next round spawns a fresh daemon (which
            // creates its own scratch).
            let _ = stop_session_daemon(name, deadline).await;
            stopped = true;
            scratch = None;
            continue;
        }
        if stopped {
            // Verification round: the previous stop either closed our scratch
            // (the fresh daemon created its own → clean) or failed to (our
            // scratch survives, now adopted → must be closed again).
            if tabs.len() == 1 && scratch.as_deref() != Some(tabs[0].target_id.as_str()) {
                // Clean: the group holds only the fresh daemon's own scratch.
                clear_sweep_warn();
                let _ = stop_session_daemon(name, deadline).await;
                return;
            }
            stopped = false;
            scratch = None; // adopted by the fresh daemon — no longer owned
        }
        if tabs.len() == 1 && scratch.as_deref() == Some(tabs[0].target_id.as_str()) {
            // Only our owned scratch remains — every listed leftover is closed
            // and verified (same-daemon re-enumeration). Stop closes it.
            let _ = stop_session_daemon(name, deadline).await;
            stopped = true;
            continue;
        }
        // Close cycle: ensure an owned scratch exists, then close every other
        // listed tab. Closing our own scratch is refused (last-tab rule) only
        // once all leftovers are gone — handled by the stop branch above.
        if scratch.is_none() {
            let Some(target_id) = session_tab_new_scratch(name, deadline).await else {
                return; // every None path already emitted its SweepWarn
            };
            scratch = Some(target_id);
        }
        for tab in &tabs {
            if tab.target_id == *scratch.as_deref().unwrap_or_default() {
                continue; // never close our own scratch
            }
            if Instant::now() >= deadline {
                break;
            }
            // Per-tab errors are swallowed by the CLI close path — the next
            // round's same-daemon enumeration is the only proof of closure.
            let _ = session_close_tab(name, &tab.tab_id, deadline).await;
        }
    }
    sweep_warn_transition(SweepWarn::Deferred);
}

/// Close the browser sessions one agent run's browser tooling used, via the
/// created-only close (`--session <name> close` — chrome-use drops the
/// session's own tabs without enumerating anything, so the user's own tabs
/// are never touched). Best-effort and strictly warn-only: a failed close
/// (daemon down, wedged CLI) leaks the session's tabs — an accepted edge, the
/// same class as the other cleanup paths. Called from the agent run end
/// (`run_agent`).
pub(crate) async fn close_run_sessions(sessions: &super::browser::BrowserRunSessions) {
    let names = sessions.snapshot();
    if names.is_empty() {
        return;
    }
    // Nothing can be closed while the relay/daemon is down; skip the
    // guaranteed-to-fail CLI round-trips (same skip gate as the sweep).
    if let Some(failure) = service_state().await {
        for name in &names {
            warn!(
                session = name,
                ?failure,
                "agent-run browser session close skipped — browser service unavailable"
            );
        }
        return;
    }
    let closes: Vec<_> = names.iter().map(|name| close_run_session(name)).collect();
    join_all(closes).await;
}

async fn close_run_session(name: &str) {
    match run_cli_bounded(&["close"], Some(name)).await {
        None => warn!(
            session = name,
            "agent-run browser session close timed out or daemon unavailable — tabs may leak until closed by hand"
        ),
        Some(out) if !out.status.success() => warn!(
            session = name,
            "agent-run browser session close failed: {}",
            // run_cli_bounded nulls stderr — the envelope error on stdout is
            // the only failure detail available.
            extract_error(&out.stdout).unwrap_or_else(|| {
                let status = out.status;
                format!("exit status {status}")
            })
        ),
        Some(_) => debug!(session = name, "agent-run browser session closed"),
    }
}

/// Only mahbot-owned session names may be swept — user, default, and other
/// agents' sessions must never be touched (strict-scope rule).
fn is_mahbot_session_name(name: &str) -> bool {
    name.starts_with("link-enricher-")
}

/// Causes the sweep warns about — warn once per cause transition so a
/// persistent orphan does not spam every sweep, and warn again after
/// a healthy sweep cleared the previous cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SweepWarn {
    /// Leftover tab the daemon can no longer re-drive (stale binding on an
    /// about:blank tab the extension never re-attaches) — manual intervention
    /// required. Only fires while a command still errors; an orphan the
    /// extension fully dropped is invisible (see the pinned-behaviors note).
    UnreachableTab,
    /// Relay/daemon unreachable mid-sweep; retried next sweep/startup.
    CannotEnumerate,
    /// Budget exhausted without convergence; retried next sweep/startup.
    Deferred,
}

/// Last-cause anti-spam state, global across sessions: a clean sweep in one
/// session clears it for all, so a persistent orphan in another session can
/// re-warn once after that convergence — acceptable tradeoff, no per-session
/// map needed.
static LAST_SWEEP_WARN: OnceLock<Mutex<Option<SweepWarn>>> = OnceLock::new();

fn sweep_warn_transition(cause: SweepWarn) {
    let mut last = LAST_SWEEP_WARN
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_poison();
    if *last == Some(cause) {
        return;
    }
    *last = Some(cause);
    match cause {
        SweepWarn::UnreachableTab => warn!(
            "tab sweep: a leftover tab is unreachable (the extension lost its debugger attach; \
             about:blank tabs are never re-attached) — close the leftover tab in Chrome to \
             unblock this session; the sweep keeps retrying"
        ),
        SweepWarn::CannotEnumerate => warn!(
            "tab sweep: cannot enumerate session tabs (relay/daemon unreachable or malformed \
             response) — deferring to the next sweep"
        ),
        SweepWarn::Deferred => {
            warn!("tab sweep: group not clean within budget — deferring to the next sweep");
        }
    }
}

fn clear_sweep_warn() {
    *LAST_SWEEP_WARN
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_poison() = None;
}

/// Map a session-CLI error to its [`SweepWarn`] and return `None` for the
/// caller's `Option<T>` (the error path never yields a value). Generic over
/// the Ok type so both `tab list` (`Vec<SweepTab>`) and `tab new` (`String`)
/// callers compile with the same one-liner; every error emits its warn, so a
/// deferral is never silent.
fn sweep_none_on_cli_error<T>(name: &str, err: Option<&str>) -> Option<T> {
    let msg = err.unwrap_or_default();
    if is_unreachable_tab_error(msg) {
        tracing::debug!(
            session = name,
            error = msg,
            "tab sweep: unreachable-tab detail"
        );
        sweep_warn_transition(SweepWarn::UnreachableTab);
    } else {
        sweep_warn_transition(SweepWarn::CannotEnumerate);
    }
    None
}

/// Bounded `tab list --json` on a session. `None` on timeout, CLI failure, an
/// error response, or a malformed entry (a missing tabId/targetId makes the
/// count unreliable — defer rather than risk a false-clean verdict). Every
/// `None` path emits its [`SweepWarn`], so a deferral is never silent.
async fn session_tab_list(name: &str, deadline: Instant) -> Option<Vec<SweepTab>> {
    if Instant::now() >= deadline {
        sweep_warn_transition(SweepWarn::Deferred);
        return None;
    }
    let v = match run_session_cli_json(&["tab", "list"], name).await {
        Ok(v) => v,
        Err(err) => return sweep_none_on_cli_error(name, err.as_deref()),
    };
    let Some(tabs) = v
        .get("data")
        .and_then(|d| d.get("tabs"))
        .and_then(Value::as_array)
    else {
        sweep_warn_transition(SweepWarn::CannotEnumerate);
        return None;
    };
    let parsed: Option<Vec<SweepTab>> = tabs
        .iter()
        .map(|t| {
            Some(SweepTab {
                tab_id: t.get("tabId")?.as_str()?.to_string(),
                target_id: t.get("targetId")?.as_str()?.to_string(),
            })
        })
        .collect();
    parsed.or_else(|| {
        sweep_warn_transition(SweepWarn::CannotEnumerate);
        None
    })
}

/// Error signatures of a leftover tab the daemon can no longer re-drive: its
/// binding went stale (relay blip, kill during an outage) while the extension
/// keeps the attach. The sweep's scratch tab is about:blank and the extension
/// never re-attaches `about:` URLs (its `eligible()` filter), so only closing
/// the tab by hand unblocks the session — the sweep logs this signal and keeps
/// retrying. An orphan the extension fully dropped (service-worker restart)
/// never produces these; it is invisible to every CLI path (see the sweep's
/// pinned-behaviors note). Real browser calls hitting this state fail fast
/// with the same guidance without marking the daemon unhealthy — see
/// [`unreachable_tab_message`]. The "or the relay lost it" variant is a
/// permanent orphan (the relay dropped the attach); "navigated across
/// processes" is a recoverable OAuth/SSO retarget and must stay OUT.
pub(crate) fn is_unreachable_tab_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("can no longer be resolved")
        || lower.contains("owns no resolvable tab")
        || lower.contains("no attached tab")
        || lower.contains("its tab is gone")
        || lower.contains("stale session")
        || lower.contains("unknown session")
        || lower.contains("the relay lost it")
}

/// Actionable error for a real call that hit an orphaned tab: the daemon and
/// relay are up, only the session's tab is unreachable (the extension never
/// re-attaches about:blank tabs). Fail fast with hand-close guidance instead of
/// paying the CLI's ~152 s retry loop, and do NOT mark the daemon unhealthy —
/// recovery cannot fix a Chrome-side orphan.
pub(crate) fn unreachable_tab_message(error: &str) -> String {
    format!(
        "{error}. The chrome-use extension lost its debugger attach to this tab and never \
         re-attaches about:blank tabs — close the leftover tab in Chrome to unblock this \
         session (the browser daemon itself is healthy)."
    )
}

/// Create a scratch tab and return its stable targetId, matched by the `t<N>`
/// ref from the `tab new` response in the next enumeration (same daemon, so
/// refs are stable until a stop). Every `None` path emits its [`SweepWarn`]
/// (or the re-enumeration's `session_tab_list` already did), so callers return
/// without re-warning and never override a more specific cause.
async fn session_tab_new_scratch(name: &str, deadline: Instant) -> Option<String> {
    if Instant::now() >= deadline {
        sweep_warn_transition(SweepWarn::Deferred);
        return None;
    }
    let resp = match run_session_cli_json(&["tab", "new"], name).await {
        Ok(v) => v,
        Err(err) => return sweep_none_on_cli_error(name, err.as_deref()),
    };
    let Some(tab_id) = resp
        .get("data")
        .and_then(|d| d.get("tabId"))
        .and_then(Value::as_str)
        .map(String::from)
    else {
        sweep_warn_transition(SweepWarn::CannotEnumerate);
        return None;
    };
    let after = session_tab_list(name, deadline).await?; // warns on None
    after
        .iter()
        .find(|t| t.tab_id == tab_id)
        .map(|t| t.target_id.clone())
        .or_else(|| {
            sweep_warn_transition(SweepWarn::CannotEnumerate);
            None
        })
}

async fn session_close_tab(name: &str, tab_id: &str, deadline: Instant) -> Option<()> {
    if Instant::now() >= deadline {
        return None;
    }
    run_session_cli_json(&["close", tab_id], name)
        .await
        .ok()
        .map(|_| ())
}

/// Bounded `session stop` — its exit code is never trusted as proof of closure
/// (re-enumeration is), and it is skipped when the sweep is already over
/// budget (the daemon idles out on its own and the next sweep retries). The
/// session is named by the helper's `--session` flag alone.
async fn stop_session_daemon(name: &str, deadline: Instant) -> Option<()> {
    if Instant::now() >= deadline {
        return None;
    }
    run_session_cli_json(&["session", "stop"], name)
        .await
        .ok()
        .map(|_| ())
}

/// Session-scoped variant — the structured error message survives for
/// signature detection.
async fn run_session_cli_json(args: &[&str], session: &str) -> Result<Value, Option<String>> {
    run_cli_json_opt(args, Some(session)).await
}

/// Extract the `error` message from a CLI error response, if any.
fn extract_error(stdout: &[u8]) -> Option<String> {
    let v: Value = serde_json::from_slice(stdout).unwrap_or_default();
    v.get("error")
        .and_then(Value::as_str)
        .map(String::from)
        .filter(|s| !s.is_empty())
}

/// Core envelope-success predicate with failure precedence: an explicit
/// `false` on either key loses over a contradicting success key (conservative
/// — the error text surfaces), and a payload with neither key is not a
/// success. Shared by [`envelope_verdict`] (Value-based) and
/// `BrowserResponse::is_success` (typed) so the two cannot drift.
pub(crate) fn envelope_success(success: Option<bool>, ok: Option<bool>) -> bool {
    !(success == Some(false) || ok == Some(false)) && (success == Some(true) || ok == Some(true))
}

/// Tri-state verdict of a chrome-use JSON envelope: `Some(true)` when either
/// `success` or `ok` reports success, `Some(false)` when either reports
/// failure, `None` when the payload carries no verdict key (callers decide
/// their own default). Newer chrome-use commands replaced `success` with `ok`
/// (e.g. `session list`).
pub(crate) fn envelope_verdict(v: &Value) -> Option<bool> {
    let success = v.get("success").and_then(Value::as_bool);
    let ok = v.get("ok").and_then(Value::as_bool);
    if success.is_none() && ok.is_none() {
        return None;
    }
    Some(envelope_success(success, ok))
}

fn set_health(outcome: ProbeOutcome) {
    let mut h = health().lock().unwrap_poison();
    h.apply_outcome(outcome, Instant::now(), true);
}

/// Health update for the verification right after a restart. Healthy here
/// must NOT seed the sustained-healthy window — the window counts consecutive
/// watchdog interval evaluations after recovery, not the immediate
/// verification.
fn set_health_after_restart(outcome: ProbeOutcome) {
    let mut h = health().lock().unwrap_poison();
    h.apply_outcome(outcome, Instant::now(), false);
}

/// Async availability for call paths: uses a fresh cached evaluation when
/// possible, otherwise re-evaluates the daemon-free status (bounded) and
/// caches the result. A fresh down-result wakes the watchdog so recovery
/// starts without waiting for the next interval.
pub(crate) async fn is_available() -> bool {
    let cached = {
        let h = health().lock().unwrap_poison();
        let ttl = if h.healthy == Some(false) {
            UNHEALTHY_TTL
        } else {
            HEALTH_TTL
        };
        h.last_probe
            .filter(|t| t.elapsed() < ttl)
            .map(|_| h.healthy)
    };
    if let Some(Some(healthy)) = cached {
        return healthy;
    }
    let outcome = evaluate_health().await;
    let healthy = outcome.is_healthy();
    set_health(outcome);
    if !healthy {
        wake().notify_one();
    }
    healthy
}

/// Sync availability for tool advertisement (never evaluates — uses the last
/// known state). Unknown → advertise optimistically; only a confirmed-down
/// evaluation hides the tool.
pub(crate) fn is_advertised() -> bool {
    health().lock().unwrap_poison().healthy != Some(false)
}

/// Mark the daemon unhealthy immediately (fail-fast path) with the cause the
/// error text points to, and wake the watchdog so recovery starts without
/// waiting for the next interval. Unreachable-tab errors never reach this path
/// — the browser tool's fail-fast guard bails with hand-close guidance first
/// (recovery cannot fix a Chrome-side orphan, so none is attempted).
pub(crate) fn note_unhealthy(error: &str) {
    // Same classification as the watchdog's health evaluation — the two
    // detection paths must agree on the cause.
    set_health(ProbeOutcome::Down(
        classify_failure_text(error).unwrap_or(ProbeFailure::DaemonWedge),
    ));
    wake().notify_one();
}

/// Actionable error shown when the daemon is down. Names the classified cause
/// with its concrete fix, and reflects whether auto-recovery is active or
/// frozen by thrash protection.
pub(crate) fn daemon_down_message() -> String {
    let h = health().lock().unwrap_poison();
    let cause = match h.last_failure {
        Some(ProbeFailure::NotInstalled) => {
            "The chrome-use extension or native host is not installed — the browser daemon \
             cannot run. Enable the chrome-use extension at chrome://extensions (or reinstall \
             the chrome-use CLI); health recovers automatically once it is installed."
        }
        Some(ProbeFailure::HostBroken) => {
            "The chrome-use native host launcher is broken — run `chrome-use doctor` (or \
             reinstall the chrome-use CLI); health recovers automatically once it is fixed."
        }
        Some(ProbeFailure::ExtensionDisabled) => {
            "The chrome-use extension is disabled — enable it at chrome://extensions. Daemon \
             restarts cannot fix a Chrome-side disable; health recovers automatically once \
             it is enabled."
        }
        Some(ProbeFailure::RelayDown) => {
            "The chrome-use extension relay is down (the extension itself is enabled). \
             Auto-recovery restarts the session daemons and waits for the extension to \
             reconnect."
        }
        Some(ProbeFailure::UnreachableTab) => {
            "A browser tab the session was driving is unreachable (the extension lost its \
             debugger attach; about:blank tabs are never re-attached) — close the leftover \
             tab in Chrome to unblock the session."
        }
        Some(ProbeFailure::DaemonWedge) | None => {
            "The chrome-use browser daemon is down or unresponsive."
        }
    };
    let recovery = if h.halted {
        " Auto-recovery exhausted its restart attempts and is in a 30-minute cooldown (thrash \
         protection); it will retry after the cooldown."
    } else if h.last_failure.is_some_and(ProbeFailure::is_unfixable) {
        " Auto-recovery is paused for this cause — no restart will be attempted; it resumes \
         automatically once the underlying issue is resolved."
    } else {
        " Auto-recovery was triggered and will restart it automatically — no manual action is \
         needed (note: the restart resets browser sessions)."
    };
    format!(
        "{cause}{recovery} While it's down, use web_search, or shell `curl` for page fetches, \
         instead of the browser tool."
    )
}

/// Background watchdog: evaluate daemon health from the daemon-free status,
/// auto-restart with bounded backoff when down, and halt after repeated crashes
/// to avoid a restart loop. Stands down on hosts without the chrome-use CLI
/// (nothing to monitor or restart) — but only after [`CLI_MISSING_THRESHOLD`]
/// consecutive definitive-missing probes, so a transient spawn failure (EAGAIN
/// under process pressure) never takes the watchdog out of service.
///
/// Probe cadence: healthy hosts re-verify CLI presence every [`CLI_RECHECK`]
/// (5 min, no per-interval `--version` spawns); unknown hosts re-probe every
/// [`WATCHDOG_INTERVAL`] (30 s); stood-down hosts re-check at [`CLI_RECHECK`]
/// (5 min) in the steady state, and every [`WATCHDOG_INTERVAL`] while a
/// transient persists — the deliberate price of never standing down on a
/// single transient, bounded by the probe timeout. A transient verdict implies
/// the binary resolved, so it resets the missing streak and re-enables
/// recovery even on a stood-down host (the stand-down premise is stale). A
/// deterministically broken install (`--version` exits non-zero) classifies as
/// transient and thus never stands down, re-probing at the applicable cadence.
pub async fn run_watchdog() {
    let mut cli_present: Option<bool> = None;
    let mut last_cli_check = Instant::now();
    let mut cli_missing: u32 = 0;
    // Last transient probe cause — warn only on change so a persistent
    // transient leaves a trail without spamming the log.
    let mut last_transient: Option<CliProbeFailure> = None;
    // One-time sweep of leaked mahbot-owned session artifacts from crashed runs
    // or older versions (see cleanup_stale_sessions).
    let mut cleaned = false;
    // Whether the last wait ended in an early wake from the fail-fast path —
    // recovery then consumes the stored classification instead of re-evaluating
    // (the daemon-free status cannot see a wedged daemon and would clobber it).
    let mut woken = false;
    loop {
        // How long to wait before the next iteration, and whether the health
        // evaluation is skipped: a CLI-less host cannot run commands, and the
        // status-unavailable-is-healthy fallback would mark its daemon Healthy.
        let mut sleep = WATCHDOG_INTERVAL;
        let mut skip_health = false;
        let cli_due = last_cli_check.elapsed() >= CLI_RECHECK;
        if cli_present != Some(true) || cli_due {
            last_cli_check = Instant::now();
            match cli_probe().await {
                CliStatus::Available => {
                    cli_present = Some(true);
                    cli_missing = 0;
                    last_transient = None;
                }
                CliStatus::Transient(failure) => {
                    // Not definitive absence — the watchdog stays in service.
                    // Healthy hosts re-probe at the CLI_RECHECK gate; unknown
                    // and stood-down hosts re-probe next interval. Warn on
                    // each distinct cause so a persistently wedged-but-present
                    // CLI leaves a trail without spamming the log.
                    if last_transient.as_ref() != Some(&failure) {
                        warn!("chrome-use CLI probe transient: {failure}");
                        last_transient = Some(failure);
                    }
                    cli_missing = 0;
                }
                CliStatus::Missing => {
                    cli_missing += 1;
                    last_transient = None;
                    if cli_missing < CLI_MISSING_THRESHOLD {
                        // First miss — confirm on the next interval before
                        // standing down (and re-probe: the cached verdict is
                        // no longer trustworthy).
                        cli_present = None;
                        skip_health = true;
                    } else {
                        if cli_present != Some(false) {
                            cli_present = Some(false);
                            warn!(
                                "chrome-use CLI not found — browser daemon watchdog standing down"
                            );
                        }
                        // Re-check rarely on CLI-less hosts so the watchdog
                        // doesn't spawn `--version` every interval; an early
                        // wake re-checks.
                        sleep = CLI_RECHECK;
                        skip_health = true;
                    }
                }
            }
        }
        if !skip_health {
            if !cleaned {
                cleaned = true;
                cleanup_stale_sessions().await;
            }
            // A fail-fast classification from a real call is the freshest signal —
            // recover from it directly (the daemon-free status cannot see a wedged
            // daemon and would clobber the cause). On interval ticks (or a wake
            // without a stored failure), run the daemon-free evaluation and recover
            // from a service-level failure it finds.
            let failure = if woken {
                health().lock().unwrap_poison().last_failure
            } else {
                None
            };
            if let Some(failure) = failure {
                attempt_recovery(failure).await;
            } else {
                let outcome = evaluate_health().await;
                set_health(outcome);
                if let ProbeOutcome::Down(failure) = outcome {
                    attempt_recovery(failure).await;
                }
            }
        }
        // Wait for the next interval or an early wake from the fail-fast path.
        // `woken` resets on every timeout, so a wake that is not consumed
        // before a CLI stand-down is dropped instead of replayed after
        // reinstall.
        let shutdown = crate::shutdown::shutdown_token();
        woken = tokio::select! {
            () = tokio::time::sleep(sleep) => false,
            () = wake().notified() => true,
            () = shutdown.cancelled() => break,
        };
    }
}

// ── Extension-skew advisory ───────────────────────────────────────────
/// Last (expected, live) extension-version pair that was advised on. Log the
/// extension-skew notice only on a transition (pair changed, or re-skewed
/// after an in-sync clear) — mirrors the sweep's anti-spam pattern.
static LAST_EXTENSION_SKEW: OnceLock<Mutex<Option<(String, String)>>> = OnceLock::new();

/// Extension-skew advisory from an already-fetched `status --json` snapshot:
/// the Chrome Web Store extension is NEVER force-updated — the Store
/// auto-updates it in the background — so a `liveVersion` behind the CLI's
/// `expectedVersion` only logs once per skew transition. Older CLIs without
/// these fields skip silently.
fn advise_extension_skew(status: &Value) {
    let ext = status.get("data").and_then(|d| d.get("extension"));
    let (Some(expected), Some(live)) = (
        ext.and_then(|e| e.get("expectedVersion"))
            .and_then(Value::as_str),
        ext.and_then(|e| e.get("liveVersion"))
            .and_then(Value::as_str),
    ) else {
        // Older CLIs without these fields — nothing to advise on.
        return;
    };
    if expected.is_empty() || live.is_empty() {
        return;
    }
    let mut last = LAST_EXTENSION_SKEW
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_poison();
    if expected == live {
        // In sync — clear the stored skew so a later re-skew re-logs.
        *last = None;
        return;
    }
    if *last == Some((expected.to_string(), live.to_string())) {
        return;
    }
    *last = Some((expected.to_string(), live.to_string()));
    info!(
        "chrome-use browser extension version mismatch: installed {live}, CLI expects {expected} — \
         Chrome updates Web Store extensions automatically in the background; reloading the \
         extension at chrome://extensions (or waiting for the Store auto-update) clears this"
    );
}

/// Download the chrome-use release archive for `tag` for this platform,
/// verify it against the published `.sha256` sidecar (missing/unreadable/
/// mismatching sidecar is a HARD failure), and extract the single binary into
/// a temp dir. Returns `(temp dir, binary path)` — the caller must keep the
/// guard alive until the binary has been moved into place.
async fn download_chrome_use_binary(tag: &str) -> Result<(tempfile::TempDir, PathBuf), String> {
    use crate::util::http::{DownloadSizeCheck, build_download_client, download_verified};

    let platform = release_asset_name()?;
    let asset = format!("chrome-use-{platform}.tar.gz");
    let base = format!("https://github.com/{CHROME_USE_RELEASE_REPO}/releases/download/{tag}");
    let tgz_url = format!("{base}/{asset}");
    let sha_url = format!("{tgz_url}.sha256");

    let client = build_download_client(CHROME_USE_DOWNLOAD_TIMEOUT)
        .map_err(|e| format!("failed to build download client: {e}"))?;

    // Fetch the `.sha256` sidecar with the same client; a missing/unreadable/
    // mismatching sidecar is a hard failure so a tampered or partial release is
    // never installed.
    let sidecar = client
        .get(&sha_url)
        .send()
        .await
        .map_err(|e| format!("failed to fetch sha256 sidecar {sha_url}: {e}"))?;
    if !sidecar.status().is_success() {
        return Err(format!(
            "failed to fetch sha256 sidecar {sha_url}: HTTP {}",
            sidecar.status()
        ));
    }
    let body = sidecar
        .text()
        .await
        .map_err(|e| format!("failed to read sha256 sidecar {sha_url}: {e}"))?;
    let (hash, sidecar_name) = parse_sha256_sidecar(&body).ok_or_else(|| {
        format!("sha256 sidecar {sha_url} is malformed (no `64-hex-hash  filename` pair)")
    })?;
    // The sidecar names the archive it was published for — a valid hash from a
    // cross-paired sidecar must not verify a different asset.
    if sidecar_name != asset {
        return Err(format!(
            "sha256 sidecar {sha_url} names '{sidecar_name}', expected '{asset}'"
        ));
    }

    let dir = tempfile::tempdir().map_err(|e| format!("failed to create temp dir: {e}"))?;
    let archive_path = dir.path().join("archive.tar.gz");
    download_verified(
        &client,
        &tgz_url,
        &archive_path,
        &hash,
        None,
        DownloadSizeCheck::None,
        |_, _| {},
    )
    .await
    .map_err(|e| format!("failed to download {tgz_url}: {e}"))?;

    let out_path = extract_chrome_use_binary(&archive_path, dir.path())?;
    Ok((dir, out_path))
}

/// Extract the single `<browser_bin()>` binary from a chrome-use release
/// archive into `dir` and return its path. The entry is matched by file name
/// at any depth but always written to `<dir>/<browser_bin()>`, so a nested
/// vendor layout still lands correctly; `Err` when the archive has no such
/// regular-file entry.
fn extract_chrome_use_binary(archive: &Path, dir: &Path) -> Result<PathBuf, String> {
    let file = fs::File::open(archive)
        .map_err(|e| format!("failed to open archive {}: {e}", archive.display()))?;
    let mut tar_archive = tar::Archive::new(flate2::read::GzDecoder::new(file));
    let out_path = dir.join(browser_bin());
    let mut unpacked = false;
    let entries = tar_archive
        .entries()
        .map_err(|e| format!("failed to read archive {}: {e}", archive.display()))?;
    for entry in entries {
        let mut entry = entry.map_err(|e| format!("failed to read archive entry: {e}"))?;
        let path = entry
            .path()
            .map_err(|e| format!("failed to read archive entry path: {e}"))?
            .into_owned();
        if !entry.header().entry_type().is_file() {
            continue;
        }
        if path
            .file_name()
            .is_some_and(|n| n == std::ffi::OsStr::new(browser_bin()))
        {
            // `unpack` writes exactly to `out_path` (the entry's own internal
            // path is ignored), keeping the returned path correct for any
            // archive layout.
            entry
                .unpack(&out_path)
                .map_err(|e| format!("failed to extract {}: {e}", out_path.display()))?;
            unpacked = true;
            break;
        }
    }
    if !unpacked {
        return Err(format!("archive contains no {} binary", browser_bin()));
    }

    // The freshly extracted binary must be executable.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&out_path, fs::Permissions::from_mode(0o755)).map_err(|e| {
            format!(
                "failed to set executable bit on {}: {e}",
                out_path.display()
            )
        })?;
    }
    Ok(out_path)
}

/// Replace the binary at `dest` with `fresh`, never leaving a broken install.
/// Unix: copy to a `<dest>.mahbot_tmp` sibling, preserve the old file's
/// permissions (default 0o755 when `dest` is new), then rename (atomic, safe
/// over a running binary). Windows: a running exe cannot be overwritten or
/// deleted but CAN be renamed — rename-aside `dest` → `dest.old`, rename the
/// temp copy in, restore the aside on failure, and best-effort remove the
/// aside afterwards (its removal fails while the old binary is still running;
/// the next successful swap clears it).
fn swap_binary_in_place(fresh: &Path, dest: &Path) -> Result<(), String> {
    let tmp = dest.with_extension("mahbot_tmp");
    let _ = fs::remove_file(&tmp);
    fs::copy(fresh, &tmp).map_err(|e| {
        format!(
            "failed to copy {} to {}: {e}",
            fresh.display(),
            tmp.display()
        )
    })?;
    // Unix: keep the old binary's mode (0o755 default on first install) so the
    // swap never drops the executable bit.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(dest).map_or(0o755, |m| m.permissions().mode() & 0o777);
        fs::set_permissions(&tmp, fs::Permissions::from_mode(mode))
            .map_err(|e| format!("failed to set permissions on {}: {e}", tmp.display()))?;
    }

    if cfg!(target_os = "windows") && dest.exists() {
        rename_aside_swap(&tmp, dest)
    } else {
        fs::rename(&tmp, dest).map_err(|e| {
            format!(
                "failed to rename {} to {}: {e}",
                tmp.display(),
                dest.display()
            )
        })
    }
}

/// The rename-aside swap used on Windows, where a running exe cannot be
/// overwritten or deleted but CAN be renamed: move `dest` aside, rename the
/// prepared `tmp` copy into place, and restore the aside if the final rename
/// fails. Aside removal is best-effort — while the old binary still runs it
/// cannot be removed; the next successful swap clears it. Platform-neutral
/// plain `fs::rename`/`remove_file`, so it is unit-testable on every OS.
fn rename_aside_swap(tmp: &Path, dest: &Path) -> Result<(), String> {
    let aside = dest.with_extension("old");
    let _ = fs::remove_file(&aside);
    fs::rename(dest, &aside)
        .map_err(|e| format!("failed to move {} aside: {e}", dest.display()))?;
    if let Err(e) = fs::rename(tmp, dest) {
        // Last-resort restore: if this also fails the install is genuinely
        // broken, so surface that instead of discarding the error.
        if let Err(restore) = fs::rename(&aside, dest) {
            return Err(format!(
                "failed to rename {} to {}: {e}; the restore also failed ({restore}) — \
                 {} is missing and chrome-use must be reinstalled",
                tmp.display(),
                dest.display(),
                dest.display()
            ));
        }
        return Err(format!(
            "failed to rename {} to {}: {e}",
            tmp.display(),
            dest.display()
        ));
    }
    let _ = fs::remove_file(&aside);
    Ok(())
}

/// First install of chrome-use: download the pinned chrome-use release directly
/// (SHA-256-verified against the published sidecar), place the single binary at
/// the stable managed dir that [`find_cli_binary`] always resolves, then do a
/// one-time native-host registration that never activates managed Chrome mode.
/// Called by the Support install tool (first install only — the auto-updater
/// swaps the binary in place and never re-registers). `Err` names the failing
/// step and carries truncated stdout/stderr for diagnosis.
pub(crate) async fn install_chrome_use() -> Result<(), String> {
    let tag = fetch_latest_tag(CHROME_USE_RELEASE_TIMEOUT).await?;
    let (_temp, fresh) = download_chrome_use_binary(&tag).await?;

    let dest = managed_bin_dir()
        .ok_or_else(|| "managed chrome-use bin dir unavailable (storage root not set)".to_string())?
        .join(browser_bin());
    let parent = dest
        .parent()
        .ok_or_else(|| format!("invalid chrome-use install path {}", dest.display()))?;
    fs::create_dir_all(parent)
        .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    let fresh_install = !dest.exists();
    swap_binary_in_place(&fresh, &dest)?;

    // The binary landed at the managed dir; clear any cached path so the next
    // probe re-resolves to the stable managed location.
    invalidate_cli_path();

    // Register the native-messaging host using the absolute freshly-downloaded
    // binary (never a stale PATH entry). `--no-profile` is REQUIRED on macOS to
    // avoid chrome-use's default of writing and queueing the
    // `ab-connect.mobileconfig` managed-configuration profile
    // (ExtensionInstallForcelist) that flips Chrome into "managed by your
    // organization" mode; mahbot never creates or re-queues that profile in any
    // flow. Supported since chrome-use v1.5.93; the binary is always freshly
    // downloaded so the flag is always available.
    let mut host = Command::new(&dest);
    host.args(["extension", "install", "--no-profile"]);
    match run_install_step("`chrome-use extension install --no-profile`", host).await {
        Ok(()) => Ok(()),
        Err(e) => {
            // A registration failure must not leave a half-installed state: a
            // freshly created binary is removed again (re-running the tool
            // re-downloads everything). When it replaced a previously working
            // install, the binary stays — the host registration still points
            // at the same absolute path, so the existing setup keeps working.
            if fresh_install {
                let _ = fs::remove_file(&dest);
                invalidate_cli_path();
                Err(format!(
                    "{e}\nThe chrome-use binary was placed at {} but the native-host \
                     registration failed, so it was removed to leave no half-installed state.",
                    dest.display()
                ))
            } else {
                Err(format!(
                    "{e}\nThe chrome-use binary at {} was updated, but the native-host \
                     re-registration failed — the previous registration still references this \
                     path and keeps working.",
                    dest.display()
                ))
            }
        }
    }
}

/// Run one install subprocess bounded by [`CHROME_USE_INSTALL_TIMEOUT`].
async fn run_install_step(label: &str, mut cmd: Command) -> Result<(), String> {
    let out = tokio::time::timeout(CHROME_USE_INSTALL_TIMEOUT, cmd.kill_on_drop(true).output())
        .await
        .map_err(|_| format!("{label} timed out"))?
        .map_err(|e| format!("{label} failed to spawn: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    Err(format!(
        "{label} failed ({}).\nstdout: {}\nstderr: {}",
        out.status,
        crate::util::truncate(&String::from_utf8_lossy(&out.stdout), 2048),
        crate::util::truncate(&String::from_utf8_lossy(&out.stderr), 2048),
    ))
}

// ── Once-per-boot auto-update ─────────────────────────────────────────
/// Best-effort once-per-boot auto-update of an existing chrome-use install.
/// Runs ~5 minutes after startup (never competes with boot), is skipped when
/// offline, and never first-installs — the first install stays behind the
/// Support consent flow. On all platforms the update is a binary-only,
/// checksum-verified release swap in place: the native-messaging launcher and
/// manifests reference the binary by its fixed absolute path and keep working
/// after the swap, so there is no re-registration. After success there is no
/// periodic re-check (once per boot).
pub async fn run_auto_update() {
    const RETRY_WAIT: Duration = Duration::from_mins(10);
    const MAX_RETRIES: u32 = 3;

    // Spawned task — sleep first so it never competes with service startup.
    tokio::time::sleep(Duration::from_mins(5)).await;

    if cli_path().is_none() {
        debug!("chrome-use not installed — first install stays user-confirmed (Support)");
        return;
    }
    let local = cli_version().await;

    // Resolve the latest release tag (follow the releases/latest redirect — no
    // api.github.com rate limit). Retry on network failure; a non-semver tag
    // gives up immediately (cannot compare). The raw tag string is captured so
    // the later download uses the SAME tag (no re-resolution race).
    let mut latest: Option<semver::Version> = None;
    let mut resolve_tag = String::new();
    for attempt in 0..MAX_RETRIES {
        match fetch_latest_tag(CHROME_USE_RELEASE_TIMEOUT).await {
            Ok(tag) => {
                if let Some(v) = parse_release_version(&tag) {
                    latest = Some(v);
                    resolve_tag = tag;
                    break;
                }
                info!(
                    "chrome-use auto-update: latest release tag '{tag}' is not a semver version; giving up"
                );
                return;
            }
            Err(e) => {
                debug!(
                    "chrome-use auto-update: release check failed (attempt {}/{}): {e}",
                    attempt + 1,
                    MAX_RETRIES
                );
                if attempt + 1 < MAX_RETRIES {
                    tokio::time::sleep(RETRY_WAIT).await;
                }
            }
        }
    }
    let Some(latest) = latest else {
        debug!("chrome-use auto-update skipped: release check kept failing (offline?)");
        return;
    };

    if let Some(local) = local.as_ref()
        && local >= &latest
    {
        debug!("chrome-use is up to date ({local})");
        return;
    }
    let local = local.map_or_else(|| "unknown version".to_string(), |v| v.to_string());
    debug!("chrome-use auto-update: updating {local} → {latest}");

    // The install path is the one the resolver found (the same path identity as
    // the first install — [`find_cli_binary`] always probes the managed dir
    // first). If it vanished, give up for this boot.
    let Some(dest) = cli_path() else {
        debug!("chrome-use auto-update skipped: CLI path vanished");
        return;
    };

    // Binary-only, checksum-verified release swap. The swap helper never leaves
    // a broken install, so on any error the previous binary stays untouched and
    // there is no same-boot retry (v1 policy). The native-messaging
    // launcher/manifests reference the binary by its fixed absolute path and
    // keep working after the swap — no `extension install`, no path
    // invalidation. The failing step is named in the error.
    // The temp-dir guard must be bound OUTSIDE the match: it owns the freshly
    // extracted binary, and dropping it at the end of a match arm would delete
    // the file before the swap runs.
    let (_temp, fresh) = match download_chrome_use_binary(&resolve_tag).await {
        Ok(pair) => pair,
        Err(e) => {
            info!(
                "chrome-use auto-update failed: {}",
                crate::util::truncate(&e, 1024)
            );
            return;
        }
    };
    if let Err(e) = swap_binary_in_place(&fresh, &dest) {
        info!(
            "chrome-use auto-update failed: {}",
            crate::util::truncate(&e, 1024)
        );
        return;
    }
    info!("chrome-use auto-updated to {latest} (binary in place)");
}

/// Follow the GitHub `releases/latest` redirect and return the last path
/// segment (the release tag) — avoids the api.github.com rate limit. reqwest
/// follows the redirect by default; the final response URL is the tag page.
async fn fetch_latest_tag(timeout: Duration) -> Result<String, String> {
    use crate::util::http::build_download_client;

    let url = format!("https://github.com/{CHROME_USE_RELEASE_REPO}/releases/latest");
    let client =
        build_download_client(timeout).map_err(|e| format!("release check client failed: {e}"))?;
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("release check request failed: {e}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "release check got HTTP {} from {url}",
            response.status()
        ));
    }
    let tag = response
        .url()
        .path_segments()
        .and_then(|mut segments| segments.next_back().map(str::to_string))
        .unwrap_or_default();
    if tag.is_empty() {
        return Err("release redirect resolved to an empty tag".to_string());
    }
    Ok(tag)
}

/// One-time cleanup of stale mahbot-owned browser-session artifacts at watchdog
/// start: leftover link-enricher sessions get swept so their tab groups don't
/// accumulate. Each sweep is verified (round-over-round convergence) and only
/// ever closes the target session's own tabs — sessions owned by other agents
/// or the user (explicit tabs, `default`, any non-mahbot name) are never
/// touched. Agent-run sessions are instead closed at run end, but a hard-killed
/// run's `agent-tab-*` sessions may leak (accepted residual, the same class as
/// dead-daemon link-enricher orphans — they are not enumerable here). Dead
/// link-enricher orphans stay until the tab is closed by hand — a documented
/// residual limit.
async fn cleanup_stale_sessions() {
    let Some(sessions) = registered_sessions().await else {
        return;
    };
    for name in sessions {
        if name.starts_with("link-enricher-") {
            sweep_session(&name).await;
        }
    }
}

/// Names of currently registered session daemons (from the daemon-free
/// `status --json` snapshot).
async fn registered_sessions() -> Option<Vec<String>> {
    let status = run_cli_json(&["status"]).await?;
    Some(
        status
            .get("data")?
            .get("sessions")?
            .as_array()?
            .iter()
            .filter_map(|s| s.get("name").and_then(Value::as_str).map(String::from))
            .collect(),
    )
}

/// Poll `status --json` (daemon-free) until the extension relay republishes or
/// the budget elapses. The MV3 service worker revives on its keepalive (~30 s).
async fn wait_for_relay(budget: Duration) {
    let deadline = Instant::now() + budget;
    while Instant::now() < deadline {
        if relay_up().await == Some(true) {
            return;
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

async fn relay_up() -> Option<bool> {
    let status = run_cli_json(&["status"]).await?;
    status
        .get("data")?
        .get("extension")?
        .get("relayUp")
        .and_then(Value::as_bool)
}

/// Warn once per cause transition — an ongoing failure does not spam every
/// watchdog interval, but the same cause warns again after a healthy spell.
fn warn_transition(failure: ProbeFailure) {
    let mut h = health().lock().unwrap_poison();
    if h.last_cause_warned == Some(failure) {
        return;
    }
    h.last_cause_warned = Some(failure);
    match failure {
        ProbeFailure::NotInstalled => warn!(
            "chrome-use extension or native host is not installed — the browser \
             daemon cannot run. Enable the chrome-use extension at \
             chrome://extensions (or reinstall the chrome-use CLI). Auto-recovery \
             paused until it is installed."
        ),
        ProbeFailure::HostBroken => warn!(
            "chrome-use native host launcher is broken — run `chrome-use doctor` or \
             reinstall the chrome-use CLI. Auto-recovery paused until it is fixed."
        ),
        ProbeFailure::ExtensionDisabled => warn!(
            "chrome-use extension is disabled — enable it at chrome://extensions. \
             Daemon restarts cannot fix a Chrome-side disable; auto-recovery paused \
             until it is enabled."
        ),
        ProbeFailure::RelayDown => warn!(
            "chrome-use extension relay is down (the extension is enabled) — waiting \
             for the extension to reconnect and restarting session daemons to clear \
             stale relay bindings."
        ),
        ProbeFailure::UnreachableTab => warn!(
            "a browser tab the session was driving is unreachable (the extension lost its \
             debugger attach; about:blank tabs are never re-attached) — close the leftover \
             tab in Chrome to unblock the session"
        ),
        ProbeFailure::DaemonWedge => {
            warn!("browser daemon is unresponsive — restarting it (bounded backoff).");
        }
    }
}

/// Bounded auto-recovery: restart session daemons with backoff between attempts
/// and a halt after MAX_RESTART_ATTEMPTS failures. Causes that a restart cannot
/// fix — extension disabled, not installed, broken host, unreachable tab — are
/// reported with their concrete fix and never consume restart attempts. A
/// transient relay drop is waited out first and consumes no attempt if it
/// self-heals.
async fn attempt_recovery(mut failure: ProbeFailure) {
    warn_transition(failure);
    // Unfixable causes stop here — they never consume restart attempts.
    if failure.is_unfixable() {
        return;
    }
    // While a recovery timer (restart backoff or halt cooldown) is pending, the
    // timer IS the wait — don't poll the relay for up to RELAY_REVIVE_WAIT on
    // top of it. The next watchdog cycle re-evaluates and re-enters recovery.
    let now = Instant::now();
    let throttled = {
        let h = health().lock().unwrap_poison();
        h.next_restart_at.is_some_and(|t| now < t) || h.halted_until.is_some_and(|t| now < t)
    };
    if throttled {
        return;
    }
    // A transient relay drop is waited out before any session-disrupting
    // restart: the MV3 worker republishes on its keepalive (~30 s). A drop
    // that self-heals consumes no restart attempt.
    if failure == ProbeFailure::RelayDown {
        wait_for_relay(RELAY_REVIVE_WAIT).await;
        let outcome = evaluate_health().await;
        set_health(outcome);
        match outcome {
            ProbeOutcome::Healthy => {
                info!("browser daemon: relay recovered without a restart");
                return;
            }
            ProbeOutcome::Down(f) => {
                // Re-classified (e.g. now a wedge) — re-warn and re-gate below.
                warn_transition(f);
                if f.is_unfixable() {
                    return;
                }
                failure = f;
            }
        }
    }

    // Decide whether a restart is allowed, and update the attempt bookkeeping,
    // entirely within a scoped lock so the MutexGuard is never held across
    // an await point.
    let gate = {
        let mut h = health().lock().unwrap_poison();
        h.gate_restart(Instant::now())
    };
    let RestartGate::Allowed(attempt) = gate else {
        match gate {
            RestartGate::Halted => error!(
                attempts = MAX_RESTART_ATTEMPTS,
                "browser daemon: {MAX_RESTART_ATTEMPTS} consecutive failed restarts; \
                 auto-recovery halted for 30 min (thrash protection)"
            ),
            RestartGate::Backoff => {
                debug!("browser daemon: still down; waiting out restart backoff");
            }
            RestartGate::Cooldown => {
                debug!("browser daemon: still down; thrash-protection cooldown in progress");
            }
            RestartGate::Allowed(_) => unreachable!(),
        }
        return;
    };

    info!(
        attempt,
        max = MAX_RESTART_ATTEMPTS,
        "browser daemon: attempting auto-recovery"
    );
    // Restart session daemons (session-less; closes their tabs, relay survives).
    // No `reconnect` — it can cold-restart the user's Chrome or open the Web
    // Store; a persistent relay drop self-heals on the extension's keepalive.
    let _ = run_cli(&["daemon", "restart"]).await;
    if failure == ProbeFailure::RelayDown {
        wait_for_relay(RELAY_REVIVE_WAIT).await;
    }

    let outcome = evaluate_health().await;
    // Post-restart verification must not seed the sustained-healthy window —
    // the restart budget resets only after consecutive watchdog intervals of
    // genuine health, so a run that keeps failing cannot reopen a fresh cycle.
    set_health_after_restart(outcome);
    if outcome.is_healthy() {
        info!("browser daemon: recovered after restart");
    } else {
        warn!(
            attempt,
            "browser daemon: restart attempt did not restore health"
        );
    }
}

async fn run_cli(args: &[&str]) -> bool {
    let Some(path) = cli_path() else {
        return false;
    };
    let mut cmd = Command::new(path);
    ensure_browser_env(&mut cmd);
    cmd.args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    cmd.kill_on_drop(true);
    tokio::time::timeout(Duration::from_mins(1), cmd.status())
        .await
        .is_ok_and(|r| r.is_ok_and(|st| st.success()))
}

/// Test-only lock serializing tests that mutate the global daemon health
/// state (cargo runs tests in parallel threads).
#[cfg(test)]
pub(crate) async fn with_health_test_lock() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

/// Test-only: restore the global health singleton to its pristine (unknown)
/// state so mutating tests don't leak state into later readers (e.g.
/// `Agent::new` filtering tools via `is_advertised`).
#[cfg(test)]
pub(crate) fn reset_health() {
    *health().lock().unwrap_poison() = DaemonHealth::default();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn advertisement_and_availability_reflect_daemon_state() {
        let _guard = with_health_test_lock().await;
        // Dead-daemon fixture: confirmed-down → not advertised, and the cached
        // result fails fast without re-probing.
        set_health(ProbeOutcome::Down(ProbeFailure::DaemonWedge));
        assert!(!is_advertised());
        assert!(!is_available().await);
        // Recovered: fresh healthy state → advertised and available.
        set_health(ProbeOutcome::Healthy);
        assert!(is_advertised());
        assert!(is_available().await);
        // Unknown (fresh boot) → advertised optimistically.
        reset_health();
        assert!(is_advertised());
    }

    #[test]
    fn sustained_health_resets_restart_attempts() {
        let now = Instant::now();
        let mut h = DaemonHealth {
            restart_attempts: 2,
            next_restart_at: Some(now),
            halted: true,
            halted_until: Some(now),
            ..DaemonHealth::default()
        };
        // A transient healthy result (e.g. the post-restart verification probe)
        // must not reset the budget — a runaway cycle would otherwise reopen a
        // fresh bounded cycle on every restart.
        h.apply_outcome(ProbeOutcome::Healthy, now, false);
        assert_eq!(h.restart_attempts, 2);
        assert!(h.halted);
        // The first watchdog healthy seeds the sustained-healthy window…
        h.apply_outcome(ProbeOutcome::Healthy, now + WATCHDOG_INTERVAL, true);
        assert_eq!(h.restart_attempts, 2);
        assert!(h.healthy_since.is_some());
        // …but a second healthy before the window elapses still does not reset.
        h.apply_outcome(ProbeOutcome::Healthy, now + WATCHDOG_INTERVAL * 2, true);
        assert_eq!(h.restart_attempts, 2);
        assert!(h.halted);
        // Only sustained health across the window opens a fresh bounded cycle.
        h.apply_outcome(ProbeOutcome::Healthy, now + WATCHDOG_INTERVAL * 3, true);
        assert_eq!(h.restart_attempts, 0);
        assert_eq!(h.next_restart_at, None);
        assert!(!h.halted);
        assert!(h.halted_until.is_none());
        assert_eq!(h.last_failure, None);
    }

    #[test]
    fn cause_flapping_and_transient_health_do_not_reset_restart_budget() {
        let now = Instant::now();
        let mut h = DaemonHealth {
            restart_attempts: 2,
            next_restart_at: Some(now),
            last_failure: Some(ProbeFailure::DaemonWedge),
            ..DaemonHealth::default()
        };
        // A cause flip (wedge → relay-down) must NOT reset the budget —
        // alternating causes must not evade the 3-attempt halt.
        h.apply_outcome(ProbeOutcome::Down(ProbeFailure::RelayDown), now, true);
        assert_eq!(h.last_failure, Some(ProbeFailure::RelayDown));
        assert_eq!(h.restart_attempts, 2);
        assert!(h.next_restart_at.is_some());
        // Flapping back and forth accumulates — never resets.
        h.apply_outcome(ProbeOutcome::Down(ProbeFailure::DaemonWedge), now, true);
        h.apply_outcome(ProbeOutcome::Down(ProbeFailure::RelayDown), now, true);
        assert_eq!(h.last_failure, Some(ProbeFailure::RelayDown));
        assert_eq!(h.restart_attempts, 2);
        // A transient healthy result does not reset either — only sustained
        // health across the window opens a fresh bounded cycle.
        h.apply_outcome(ProbeOutcome::Healthy, now, true);
        assert_eq!(h.restart_attempts, 2);
        assert!(h.next_restart_at.is_some());
        h.apply_outcome(ProbeOutcome::Healthy, now + SUSTAINED_HEALTHY_WINDOW, true);
        assert_eq!(h.last_failure, None);
        assert_eq!(h.restart_attempts, 0);
        assert_eq!(h.next_restart_at, None);
        assert!(!h.halted);
    }

    #[test]
    fn gate_honors_backoff_before_halt() {
        let now = Instant::now();
        let mut h = DaemonHealth::default();
        assert_eq!(h.gate_restart(now), RestartGate::Allowed(1));
        // 30s backoff before attempt 2.
        assert_eq!(h.gate_restart(now), RestartGate::Backoff);
        assert_eq!(
            h.gate_restart(now + RESTART_BACKOFF[0]),
            RestartGate::Allowed(2)
        );
        // 2min backoff before attempt 3.
        let t2 = now + RESTART_BACKOFF[0] + RESTART_BACKOFF[1];
        assert_eq!(h.gate_restart(t2), RestartGate::Allowed(3));
        // The final 10-min grace is honored before the halt fires.
        assert_eq!(h.gate_restart(t2), RestartGate::Backoff);
        let t3 = t2 + RESTART_BACKOFF[2];
        assert_eq!(h.gate_restart(t3), RestartGate::Halted);
        assert!(h.halted);
        assert_eq!(h.gate_restart(t3), RestartGate::Cooldown);
        // After the cooldown a fresh bounded cycle starts.
        assert_eq!(h.gate_restart(t3 + HALT_COOLDOWN), RestartGate::Allowed(1));
        assert_eq!(h.restart_attempts, 1);
        assert!(!h.halted);
    }

    #[test]
    fn unreachable_tab_error_signature_detected() {
        for msg in [
            "the tab this session was driving can no longer be resolved (it was closed, or a flaky relay dropped it)",
            "the tab this command was driving is gone — it may have been closed, or the relay lost it",
            "this session owns no resolvable tab in its group. Refusing to run on a tab this session does not drive",
            "stale sessionId ... its tab is gone",
            "unknown sessionId ...",
            "no attached tab ...",
        ] {
            assert!(is_unreachable_tab_error(msg), "should detect: {msg}");
        }
        for msg in [
            // Relay-side outage — owned by is_relay_unavailable_error, and the
            // sweep's service_state skip gate already covers it.
            "Auto-launch failed: Could not drive your Chrome through the ab-connect extension.",
            // Recoverable CLI retarget (OAuth/SSO navigation), NOT a permanent
            // orphan — must stay out of the unreachable-tab matcher.
            "the tab this command was driving is gone — it navigated across processes",
            // Daemon socket / page-level failures — not tab-attach problems.
            "Failed to read: Resource temporarily unavailable (os error 35)",
            "chrome-use error: Element not found",
        ] {
            assert!(!is_unreachable_tab_error(msg), "should NOT detect: {msg}");
        }
    }

    #[test]
    fn daemon_unavailable_error_signature_detected() {
        for msg in [
            "Failed to read: Resource temporarily unavailable (os error 35) (after 5 retries - daemon may be busy or unresponsive)",
            "Failed to connect: No such file or directory (os error 2) (after 5 retries - daemon may be busy or unresponsive)",
            "session unresponsive: no response within 45s",
            "Daemon failed to start (socket: /tmp/x.sock)",
            // 1.5.8x-era texts.
            "session unresponsive: the stuck '__mahbot_probe' daemon was stopped automatically",
            "Failed to connect: the daemon endpoint for session '__mahbot_probe' disappeared (/tmp/x.sock).",
            "CDP session is unresponsive after attaching (Connection reset).",
            "Auto-launch failed: Could not drive your Chrome through the ab-connect extension.",
        ] {
            assert!(is_daemon_unavailable_error(msg), "should detect: {msg}");
        }
        for msg in [
            "chrome-use error: Element not found",
            "chrome-use error: Evaluation error: ReferenceError",
            "chrome-use error: Navigation failed",
            // Page-level navigation failure — not a daemon socket problem.
            "Failed to connect to example.com: Connection timed out",
        ] {
            assert!(
                !is_daemon_unavailable_error(msg),
                "should NOT detect: {msg}"
            );
        }
    }

    #[test]
    fn relay_unavailable_signature_detected() {
        for msg in [
            "The chrome-use extension is installed, but its relay isn't connected.",
            "Could not drive your Chrome through the ab-connect extension.",
            "Chrome relay dropped — reconnecting…",
        ] {
            assert!(is_relay_unavailable_error(msg), "should detect: {msg}");
        }
        for msg in [
            "chrome-use error: Element not found",
            "Failed to read: Resource temporarily unavailable (os error 35)",
        ] {
            assert!(!is_relay_unavailable_error(msg), "should NOT detect: {msg}");
        }
    }

    #[test]
    fn daemon_unavailable_code_detected() {
        assert!(is_daemon_unavailable_code(Some("browser_not_launched")));
        // Page-level failures share the coarse `connection_failed` code with
        // daemon-socket problems — the code alone must not fail-fast the
        // daemon path (the message-text matcher disambiguates).
        assert!(!is_daemon_unavailable_code(Some("connection_failed")));
        assert!(!is_daemon_unavailable_code(Some("timeout")));
        assert!(!is_daemon_unavailable_code(Some("element_not_found")));
        assert!(!is_daemon_unavailable_code(None));
    }

    #[test]
    fn failure_text_classification_is_shared_between_detection_paths() {
        // The captured combined error (stale tab wrapped in the auto-connect
        // envelope AND the daemon wrapper) classifies as unreachable-tab, NOT
        // relay-down or wedge — recovery must not fire for an orphaned tab on
        // an otherwise-healthy relay.
        assert_eq!(
            classify_failure_text(
                "Auto-launch failed: Could not drive your Chrome through the ab-connect \
                 extension. The tab this session was driving can no longer be resolved (it \
                 was closed, or a flaky relay dropped it)"
            ),
            Some(ProbeFailure::UnreachableTab)
        );
        // Auto-connect failure alone names the relay as the cause (its body
        // points at `chrome-use extension connect`) — the relay signature wins
        // over the daemon wrapper it is wrapped in, in both the watchdog and
        // fail-fast paths, so they never disagree on the cause.
        assert_eq!(
            classify_failure_text(
                "Auto-launch failed: Could not drive your Chrome through the ab-connect extension."
            ),
            Some(ProbeFailure::RelayDown)
        );
        assert_eq!(
            classify_failure_text(
                "Failed to connect: the daemon endpoint for session '__mahbot_probe' \
                 disappeared (/tmp/x.sock)."
            ),
            Some(ProbeFailure::DaemonWedge)
        );
        assert_eq!(
            classify_failure_text("chrome-use error: Element not found"),
            None
        );
    }

    #[test]
    fn spawn_error_classification_distinguishes_missing_from_transient() {
        // Only a genuinely missing binary (NotFound) is definitive absence;
        // every other spawn error is transient and must never be reported as
        // "not installed".
        let not_found = std::io::Error::from(std::io::ErrorKind::NotFound);
        assert_eq!(classify_spawn_error(&not_found), CliStatus::Missing);
        for kind in [
            std::io::ErrorKind::WouldBlock,  // EAGAIN — process-table exhaustion
            std::io::ErrorKind::OutOfMemory, // ENOMEM
            std::io::ErrorKind::PermissionDenied, // EACCES
            std::io::ErrorKind::StorageFull, // ENOSPC
            std::io::ErrorKind::TimedOut,
        ] {
            let err = std::io::Error::from(kind);
            assert!(
                matches!(
                    classify_spawn_error(&err),
                    CliStatus::Transient(CliProbeFailure::Spawn(_))
                ),
                "kind {kind:?} must classify as transient, not missing"
            );
        }
    }

    #[test]
    fn release_version_parsing_strips_v_prefix() {
        // GitHub release tags are v-prefixed; `--version` output is bare.
        // Both must parse — a v-prefixed tag is NOT valid semver verbatim.
        assert_eq!(
            parse_release_version("v1.5.100"),
            Some(semver::Version::new(1, 5, 100))
        );
        assert_eq!(
            parse_release_version("1.5.100"),
            Some(semver::Version::new(1, 5, 100))
        );
        assert_eq!(parse_release_version("latest"), None);
        assert_eq!(parse_release_version(""), None);
    }

    #[test]
    fn cli_version_parsing_scans_the_real_banner() {
        // The --version banner ends with a bug-report URL — the version must
        // be found by scanning, not by assuming the last token.
        let banner = "chrome-use 1.5.100\n\
             report bugs / rough edges: https://github.com/leeguooooo/chrome-use/issues";
        assert_eq!(
            parse_cli_version(banner),
            Some(semver::Version::new(1, 5, 100))
        );
        assert_eq!(
            parse_cli_version("chrome-use v1.5.99"),
            Some(semver::Version::new(1, 5, 99))
        );
        assert_eq!(parse_cli_version("chrome-use\nno version here"), None);
    }

    #[test]
    fn envelope_verdict_covers_both_envelopes() {
        let v = |json: serde_json::Value| envelope_verdict(&json);
        assert_eq!(v(serde_json::json!({"success": true})), Some(true));
        assert_eq!(v(serde_json::json!({"ok": true})), Some(true));
        assert_eq!(
            v(serde_json::json!({"success": false, "error": "x"})),
            Some(false)
        );
        assert_eq!(
            v(serde_json::json!({"ok": false, "error": "x"})),
            Some(false)
        );
        // Failure beats an unrelated success key; no verdict key → None.
        assert_eq!(
            v(serde_json::json!({"success": false, "ok": true})),
            Some(false)
        );
        assert_eq!(v(serde_json::json!({"data": {}})), None);
    }

    #[test]
    fn release_asset_platform_maps_supported_combos() {
        assert_eq!(
            release_asset_platform("macos", "x86_64", false).as_deref(),
            Some("darwin-x64")
        );
        assert_eq!(
            release_asset_platform("macos", "aarch64", false).as_deref(),
            Some("darwin-arm64")
        );
        assert_eq!(
            release_asset_platform("linux", "x86_64", false).as_deref(),
            Some("linux-x64")
        );
        assert_eq!(
            release_asset_platform("linux", "aarch64", false).as_deref(),
            Some("linux-arm64")
        );
        assert_eq!(
            release_asset_platform("linux", "x86_64", true).as_deref(),
            Some("linux-musl-x64")
        );
        assert_eq!(
            release_asset_platform("linux", "aarch64", true).as_deref(),
            Some("linux-musl-arm64")
        );
        // The musl flag is ignored for non-linux platforms.
        assert_eq!(
            release_asset_platform("windows", "x86_64", true).as_deref(),
            Some("win32-x64")
        );
        assert_eq!(
            release_asset_platform("macos", "aarch64", true).as_deref(),
            Some("darwin-arm64")
        );
        // Unsupported platform/arch combos return None.
        assert_eq!(release_asset_platform("windows", "aarch64", false), None);
        assert_eq!(release_asset_platform("freebsd", "x86_64", false), None);
    }

    #[test]
    fn sha256_sidecar_parses_hash_and_filename() {
        let hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        // Real two-space sidecar format.
        assert_eq!(
            parse_sha256_sidecar(&format!("{hash}  chrome-use-linux-x64.tar.gz")),
            Some((hash.to_string(), "chrome-use-linux-x64.tar.gz".to_string()))
        );
        // Leading whitespace and trailing newline are tolerated; an uppercase
        // hash is normalized to lowercase (the computed digest is lowercase).
        assert_eq!(
            parse_sha256_sidecar(&format!("  {hash}  chrome-use-darwin-arm64.tar.gz\n")),
            Some((
                hash.to_string(),
                "chrome-use-darwin-arm64.tar.gz".to_string()
            ))
        );
        assert_eq!(
            parse_sha256_sidecar(&format!("{}  x.tar.gz", hash.to_uppercase())),
            Some((hash.to_string(), "x.tar.gz".to_string()))
        );
        // A bare-hash sidecar (no filename) is rejected — the filename is what
        // guards against cross-paired sidecars.
        assert_eq!(parse_sha256_sidecar(hash), None);
        // Rejects short, non-hex, and empty hashes.
        assert_eq!(parse_sha256_sidecar("abcd  x.tar.gz"), None);
        assert_eq!(
            parse_sha256_sidecar(&format!("{}  x.tar.gz", "g".repeat(64))),
            None
        );
        assert_eq!(parse_sha256_sidecar(""), None);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn swap_binary_in_place_preserves_permissions_and_leaves_no_temp() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("create temp dir");
        let dest = dir.path().join("chrome-use");
        let fresh = dir.path().join("fresh");

        fs::write(&dest, "old").expect("write old dest");
        fs::set_permissions(&dest, fs::Permissions::from_mode(0o755)).expect("chmod dest");
        fs::write(&fresh, "new").expect("write fresh");
        fs::set_permissions(&fresh, fs::Permissions::from_mode(0o755)).expect("chmod fresh");

        swap_binary_in_place(&fresh, &dest).expect("swap");
        assert_eq!(fs::read(&dest).expect("read dest"), b"new".as_slice());
        assert_eq!(
            fs::metadata(&dest).expect("stat dest").permissions().mode() & 0o777,
            0o755
        );
        assert!(
            !dest.with_extension("mahbot_tmp").exists(),
            "no temp sibling left"
        );

        // Second swap over an existing install keeps the running binary's mode.
        fs::set_permissions(&dest, fs::Permissions::from_mode(0o700)).expect("chmod dest");
        fs::write(&fresh, "newer").expect("write fresh 2");
        swap_binary_in_place(&fresh, &dest).expect("swap 2");
        assert_eq!(fs::read(&dest).expect("read dest"), b"newer".as_slice());
        assert_eq!(
            fs::metadata(&dest).expect("stat dest").permissions().mode() & 0o777,
            0o700
        );
        assert!(
            !dest.with_extension("mahbot_tmp").exists(),
            "no temp sibling left"
        );
    }

    #[test]
    fn rename_aside_swap_replaces_and_cleans_up() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("chrome-use");
        let tmp = dir.path().join("prepared");
        fs::write(&dest, "old").expect("write dest");
        fs::write(&tmp, "new").expect("write tmp");

        rename_aside_swap(&tmp, &dest).expect("swap");
        assert_eq!(fs::read(&dest).expect("read dest"), b"new");
        // The aside and the prepared copy are both gone after a clean swap.
        assert!(!dest.with_extension("old").exists());
        assert!(!tmp.exists());
    }

    #[test]
    fn rename_aside_swap_restores_dest_when_the_final_rename_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("chrome-use");
        // The prepared copy does not exist, so the final rename fails.
        let tmp = dir.path().join("missing");
        fs::write(&dest, "old").expect("write dest");

        assert!(rename_aside_swap(&tmp, &dest).is_err());
        // The previous binary was restored at its original location.
        assert_eq!(fs::read(&dest).expect("read dest"), b"old");
        assert!(!dest.with_extension("old").exists(), "aside was moved back");
    }

    /// Build a tar.gz archive in `dir` with the given (path, contents) entries.
    fn write_test_archive(dir: &std::path::Path, entries: &[(&str, &[u8])]) -> fs::File {
        let enc = flate2::write::GzEncoder::new(
            fs::File::create(dir.join("pkg.tar.gz")).expect("create archive"),
            flate2::Compression::default(),
        );
        let mut builder = tar::Builder::new(enc);
        for (path, contents) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o644);
            header.set_entry_type(tar::EntryType::Regular);
            builder
                .append_data(&mut header, path, *contents)
                .expect("append entry");
        }
        builder
            .into_inner()
            .expect("finish archive")
            .finish()
            .expect("finish gzip")
    }

    #[test]
    fn extract_chrome_use_binary_lands_at_the_dir_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_test_archive(dir.path(), &[(browser_bin(), b"BIN" as &[u8])]);

        let out =
            extract_chrome_use_binary(&dir.path().join("pkg.tar.gz"), dir.path()).expect("extract");
        assert_eq!(out, dir.path().join(browser_bin()));
        assert_eq!(fs::read(&out).expect("read extracted"), b"BIN");
        // The extracted binary must be executable (the exec-bit chmod is
        // unix-only; on Windows executability is the .exe extension).
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_ne!(
                fs::metadata(&out)
                    .expect("stat extracted")
                    .permissions()
                    .mode()
                    & 0o111,
                0
            );
        }
    }

    #[test]
    fn extract_chrome_use_binary_handles_a_nested_layout() {
        let dir = tempfile::tempdir().expect("tempdir");
        let entry = format!("pkg/bin/{}", browser_bin());
        write_test_archive(dir.path(), &[(entry.as_str(), b"NESTED" as &[u8])]);

        let out =
            extract_chrome_use_binary(&dir.path().join("pkg.tar.gz"), dir.path()).expect("extract");
        // Matched by file name at any depth, but always written to the dir root.
        assert_eq!(out, dir.path().join(browser_bin()));
        assert_eq!(fs::read(&out).expect("read extracted"), b"NESTED");
    }

    #[test]
    fn extract_chrome_use_binary_errors_without_the_binary() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_test_archive(dir.path(), &[("readme.txt", b"no bin" as &[u8])]);

        assert!(extract_chrome_use_binary(&dir.path().join("pkg.tar.gz"), dir.path()).is_err());
    }
}
