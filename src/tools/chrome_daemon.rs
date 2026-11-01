//! chrome-use daemon health monitoring and bounded auto-recovery.
//!
//! The chrome-use CLI talks to a per-session daemon driving Chrome through the
//! extension relay. When the daemon or relay dies its commands hang inside its
//! own 5-retry loop (~152 s) instead of failing fast, so this module classifies
//! health from a daemon-free `status` snapshot and auto-restarts with bounded
//! backoff and thrash protection. Wedge detection is per-call: a chrome command
//! failing with the daemon-unavailable signature marks the daemon unhealthy and
//! wakes the watchdog, which recovers from that stored classification (the
//! snapshot cannot see a wedge). It also owns the CLI invocation primitives
//! (binary name, `--version` check); command env lives in `crate::chrome::spawn`.
//!
//! Chrome is the owner's real, logged-in browser through that relay — no
//! `--launch` mode, no profile copies, no CDP. A down relay usually means Chrome
//! is closed, so the watchdog auto-launches the real Chrome (approved product
//! decision: a window may appear unasked, inheriting the real profile and
//! environment) under its own launch budget, separate from restarts. An absent
//! extension is unfixable by any launch or restart and pauses auto-recovery.
//!
//! Tab sweep: mahbot-owned groups (`link-enricher-*`) are closed through the CLI
//! and verified by re-enumeration, because a bare `session stop` never closes nor
//! reports tabs the session ADOPTED, and [`stop_session_daemon`] is bounded by
//! [`CLI_TIMEOUT`] — which expires as the daemon's shutdown grace does, so the
//! reclaim never runs and a missed scratch tab is orphaned with nothing left to
//! reclaim it.
//!
//! ## Readiness and the pre-action gate
//!
//! [`readiness`] reports what was ESTABLISHED about the connection from daemon-free
//! `status --json` and `browsers --json` plus the Chrome-process and display facts:
//! every fact tri-state, a missing one never defaulted to "fine", and a running
//! process only one fact, never a verdict. The display fact is needed only to LAUNCH,
//! so its absence blocks nothing while a browser stays reachable through the relay.
//!
//! - [`Readiness::ready_for_actions`] is the verdict [`Readiness::outcome`] reports
//!   `Healthy` for: host installed AND healthy, relay up, a real browser reachable. An
//!   unanswered fact leaves it false.
//! - [`Readiness::blocked`] is the gate's condition: a fact ESTABLISHED that the
//!   connection cannot work; a fact never obtained blocks nothing.
//! - [`ensure_ready_for_actions`] is the cache-backed gate: proven dispatches, an
//!   established cause runs one bounded cause-aware recovery pass and one re-probe,
//!   and anything else dispatches too, since refusing on a reporting gap would make
//!   both chrome surfaces unavailable. An own-browser fallback is a plain failure.
//!
//! The gate plus [`crate::chrome::spawn::pin_real_browser_env`] and chrome-use's own
//! browser-replacement note keep the tool's own-browser fallback from taking the work.
//!
//! [`ProbeOutcome::Unknown`] is neither health nor a down cause (no snapshot, no
//! extension data, or a decisive fact never obtained): nothing is reported as fine, no
//! recovery is driven by it — a restart destroys session state, open tabs included —
//! and advertisement stays optimistic, which is not a health claim.
//!
//! ## Recovery on every path
//!
//! [`recover_unresponsive_session`] serves the interactive tool's fail-fast and timeout
//! paths, the `mahbot chrome` verbs and their timeout, and the daemon-side session
//! calls: it records the wedge under the same `DaemonWedge` classification (waking the
//! watchdog) and stops that session's daemon within the product's bound, so the next
//! call gets a clean one. Recovery is bounded and thrash-proof, never turns a succeeded
//! call into a failure, and acts only on an established cause. An unreachable tab is its
//! own unfixable state.
//!
//! Trade-offs: a genuine wedge surfaces on the first real call, which runs to the full
//! clock (mahbot's plus the relay recovery window) before the wedge probe classifies it
//! and the restart clears it. `daemon restart` destroys all session state, which
//! guidance says. The auto-launch may open a window unasked and pauses on display-less hosts.

use crate::chrome::contract::{
    ChromeResponse, is_daemon_unavailable_error, is_relay_unavailable_error,
    is_session_unresponsive_error, is_unreachable_tab_error, parse_first,
};
use crate::chrome::spawn::{CliOutput, CliRun, CliSpawn, ensure_chrome_env, spawn_cli};
use crate::chrome::{CliRecovery, SESSION_STOP_TIMEOUT, probe_clocks};
use crate::util::UnwrapPoison;
use serde_json::Value;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::process::Command;
use tracing::{debug, error, info, warn};

/// How long a CLI health/sweep command may take before it counts as wedged: a
/// healthy daemon answers in milliseconds, a wedged one hangs for the CLI's
/// internal 45s read timeouts × 5 retries.
const CLI_TIMEOUT: Duration = Duration::from_secs(8);
/// Cache TTL for a healthy evaluation (fresh enough for per-call checks).
const HEALTH_TTL: Duration = Duration::from_secs(10);
/// Watchdog cadence between automatic health evaluations.
const WATCHDOG_INTERVAL: Duration = Duration::from_secs(30);
/// How often the watchdog re-verifies CLI presence where it was found — the
/// binary can be uninstalled while the daemon runs, but spawning `--version`
/// every interval would be waste.
const CLI_RECHECK: Duration = Duration::from_mins(5);
/// Consecutive definitive-missing CLI probes before the watchdog stands down —
/// one transient spawn failure (EAGAIN/EMFILE) must not take it out of service.
const CLI_MISSING_THRESHOLD: u32 = 2;
/// Consecutive failed restarts before auto-recovery halts (thrash protection).
const MAX_RESTART_ATTEMPTS: u32 = 3;
/// Consecutive failed Chrome auto-launches before the launch budget halts —
/// independent of the restart budget, since a closed browser is a different
/// problem than a wedged daemon.
const MAX_LAUNCH_ATTEMPTS: u32 = 3;
/// Sustained-health window: the restart counter resets only after the status has been
/// healthy this long (≥2 watchdog intervals), so a transient healthy right after a
/// restart cannot reopen a cycle and keep a runaway loop from tripping the halt. That
/// status cannot see wedges, so the halt engages only for service-level causes.
const SUSTAINED_HEALTHY_WINDOW: Duration = Duration::from_mins(1);
/// Backoff between restart attempts (30s → 2min → 10min).
const RESTART_BACKOFF: [Duration; 3] = [
    Duration::from_secs(30),
    Duration::from_mins(2),
    Duration::from_mins(10),
];
/// Backoff between Chrome auto-launch attempts, mirroring the restart backoff so
/// a host that keeps failing cannot spam launches.
const LAUNCH_BACKOFF: [Duration; 3] = [
    Duration::from_secs(30),
    Duration::from_mins(2),
    Duration::from_mins(10),
];
/// Cooldown after the max restart attempts, before a fresh bounded cycle.
const HALT_COOLDOWN: Duration = Duration::from_mins(30);
/// How long recovery waits for the relay to republish after a relay-drop
/// `daemon restart`: the MV3 service worker revives on its ~30 s keepalive and
/// only then writes the endpoint back.
const RELAY_REVIVE_WAIT: Duration = Duration::from_secs(40);
/// Bound on how long recovery's CALLER waits for `session stop`, whose whole legitimate
/// cost is ≈28 s (the shutdown grace plus the tab reclaim that closes the session's
/// group): 45 s covers it, so a working stop confirms instead of being cut off as
/// unanswered. The stop itself is never killed ([`session_stop_via`]). A sweep recovering
/// a wedge spends this whole bound and deliberately overshoots [`SWEEP_TOTAL_BUDGET`] —
/// an unrecovered wedge is worse, and leftovers are retried later.
pub(crate) const SESSION_RECOVERY_TIMEOUT: Duration = Duration::from_secs(45);

/// Total budget for one sweep, from before the skip gate. The deadline is checked before
/// each CLI call, so one in-flight call overshoots by at most [`CLI_TIMEOUT`], except a
/// wedge recovery, which spends up to [`SESSION_RECOVERY_TIMEOUT`]. On expiry leftover
/// tabs are retried by the next sweep or startup, never orphaned.
const SWEEP_TOTAL_BUDGET: Duration = Duration::from_secs(15);
/// Convergence rounds per sweep: the deadline decides when it stops, this bounds
/// the enumerate/close/stop cycles (a healthy host converges in 3, a retried
/// failed close needs 4–5).
const SWEEP_MAX_ROUNDS: u32 = 5;

/// Classified cause for a failed health check; decides whether auto-recovery can
/// help at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeFailure {
    NotInstalled,
    HostBroken,
    ExtensionDisabled,
    /// Absent from Chrome entirely — unfixable by any launch or restart; the user
    /// must install it from the store.
    ExtensionAbsent,
    /// Extension enabled but the relay is down — transient, self-heals.
    RelayDown,
    /// No browser process is running — NOT unfixable: the fix is launching the
    /// owner's real Chrome, never a daemon restart.
    ChromeNotRunning,
    /// The session's tab lost its debugger attach while daemon and relay are up;
    /// the product closes the session's tabs at run end and on stop/recovery, and
    /// the session cannot drive that tab again.
    UnreachableTab,
    /// The daemon socket hung or errored (daemon-side wedge).
    DaemonWedge,
}

/// Result of a health evaluation. `Unknown` — nothing decisive established — is neither
/// health nor a down cause: it fails [`ProbeOutcome::is_healthy`], never drives a recovery
/// (see [`ProbeOutcome::failure`]), and advertisement stays optimistic, since hiding the
/// tool would be a claim the probe never made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeOutcome {
    Healthy,
    Down(ProbeFailure),
    Unknown,
}

impl ProbeOutcome {
    fn is_healthy(self) -> bool {
        matches!(self, ProbeOutcome::Healthy)
    }

    /// The cause a fact ESTABLISHED, or `None` when none was. `Unknown` yields
    /// `None`, so a recovery reading this never spends the state-destroying
    /// restart on "could not find out" — it re-probes instead.
    fn failure(self) -> Option<ProbeFailure> {
        match self {
            ProbeOutcome::Healthy | ProbeOutcome::Unknown => None,
            ProbeOutcome::Down(f) => Some(f),
        }
    }
}

impl ProbeFailure {
    /// Causes a restart cannot fix: reported with their concrete fix and never
    /// consuming restart attempts. `ChromeNotRunning` is deliberately not here —
    /// a closed browser is fixed by launching it.
    fn is_unfixable(self) -> bool {
        matches!(
            self,
            ProbeFailure::NotInstalled
                | ProbeFailure::HostBroken
                | ProbeFailure::ExtensionDisabled
                | ProbeFailure::ExtensionAbsent
                | ProbeFailure::UnreachableTab
        )
    }
}

/// Outcome of a Chrome auto-launch, for honest down messaging (`None` on
/// [`DaemonHealth`] = never attempted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChromeLaunchOutcome {
    Launched,
    Failed,
    NoDisplay,
}

/// One bounded attempt budget: backoff between attempts, halt plus cooldown after
/// `max` failures. Pure state, so the state machine is unit-testable.
#[derive(Default)]
struct AttemptBudget {
    attempts: u32,
    next_at: Option<Instant>,
    halted: bool,
    halted_until: Option<Instant>,
}

impl AttemptBudget {
    /// Decide whether an attempt is allowed, updating the bookkeeping. Backoff is checked
    /// before the attempt cap, so the final grace after the last attempt is honored before
    /// the halt fires; cooldown expiry opens a fresh cycle.
    fn gate(
        &mut self,
        max: u32,
        backoff: &[Duration],
        cooldown: Duration,
        now: Instant,
    ) -> RecoveryGate {
        if self.halted {
            if self.halted_until.is_some_and(|until| now < until) {
                return RecoveryGate::Cooldown;
            }
            // Cooldown expired — reset and allow a fresh bounded cycle.
            self.halted = false;
            self.attempts = 0;
            self.halted_until = None;
            self.next_at = None;
        }
        // Backoff is checked before the attempt cap, so the final long grace
        // after the last attempt is honored before the halt fires.
        if self.next_at.is_some_and(|next| now < next) {
            return RecoveryGate::Backoff;
        }
        if self.attempts >= max {
            self.halted = true;
            self.halted_until = Some(now + cooldown);
            return RecoveryGate::Halted;
        }
        let attempt = self.attempts + 1;
        self.attempts = attempt;
        self.next_at = Some(now + backoff[(attempt as usize - 1).min(backoff.len() - 1)]);
        RecoveryGate::Allowed(attempt)
    }

    /// Whether a recovery timer is currently pending — when it is, the timer IS
    /// the wait (callers must not stack their own relay-revive polls on top of it).
    fn is_waiting(&self, now: Instant) -> bool {
        self.halted_until.is_some_and(|until| now < until)
            || self.next_at.is_some_and(|next| now < next)
    }

    /// Reset to a fresh bounded cycle (sustained health / cooldown expiry).
    fn reset(&mut self) {
        self.attempts = 0;
        self.next_at = None;
        self.halted = false;
        self.halted_until = None;
    }
}

#[derive(Default)]
struct DaemonHealth {
    /// Last verdict; `None` before any probe (advertised optimistically). Unknown
    /// carries no cause, so it is neither health nor a named-cause down message.
    verdict: Option<ProbeOutcome>,
    last_probe: Option<Instant>,
    /// The readiness snapshot behind the verdict and when it was taken — the one
    /// record [`probe_and_record`] writes, so the gate's [`HEALTH_TTL`]-bounded
    /// cache and [`daemon_down_message`] always read the same probe.
    readiness: Option<(Instant, Readiness)>,
    restart_budget: AttemptBudget,
    /// Chrome auto-launch cycle — separate from the restart budget, since a closed
    /// browser and a wedged daemon are independent failures.
    launch_budget: AttemptBudget,
    /// Last launch outcome (never attempted = `None`); failures survive until
    /// health or the sustained-health reset clears them, keeping messaging honest.
    launch_outcome: Option<ChromeLaunchOutcome>,
    /// The cause the last transition-based record named — reset on recovery so the
    /// same cause is reported again after a healthy spell.
    last_cause_reported: Option<ProbeFailure>,
    /// Start of the current sustained-healthy streak, which resets the budgets.
    healthy_since: Option<Instant>,
}

/// Decision from [`DaemonHealth::gate_restart`] / [`DaemonHealth::gate_launch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecoveryGate {
    Allowed(u32),
    Backoff,
    Cooldown,
    Halted,
}

impl DaemonHealth {
    /// Decide whether a daemon restart attempt is allowed.
    fn gate_restart(&mut self, now: Instant) -> RecoveryGate {
        self.restart_budget
            .gate(MAX_RESTART_ATTEMPTS, &RESTART_BACKOFF, HALT_COOLDOWN, now)
    }

    /// Decide whether a Chrome launch is allowed — its own budget, since a closed
    /// browser is not a wedged daemon.
    fn gate_launch(&mut self, now: Instant) -> RecoveryGate {
        self.launch_budget
            .gate(MAX_LAUNCH_ATTEMPTS, &LAUNCH_BACKOFF, HALT_COOLDOWN, now)
    }

    /// Apply a health observation. Both budgets reset only once a healthy verdict has held
    /// for [`SUSTAINED_HEALTHY_WINDOW`], so a transient healthy after a recovery (a
    /// `seed_window = false` verification, or one interval) cannot reopen a cycle. Any
    /// failure, and any `Unknown` as no evidence of health, aborts the window.
    fn apply_outcome(&mut self, outcome: ProbeOutcome, now: Instant, seed_window: bool) {
        let healthy = outcome.is_healthy();
        if healthy {
            if self
                .healthy_since
                .is_some_and(|since| now.duration_since(since) >= SUSTAINED_HEALTHY_WINDOW)
            {
                // Sustained health — open a fresh bounded cycle for both budgets.
                self.restart_budget.reset();
                self.launch_budget.reset();
                self.last_cause_reported = None;
            }
            if seed_window && self.healthy_since.is_none() {
                self.healthy_since = Some(now);
            }
            // Any healthy observation supersedes a stale launch record; failure
            // outcomes do not clear it, so the down message stays honest.
            self.launch_outcome = None;
        } else {
            self.healthy_since = None;
        }
        // A cause change never resets the restart budget — flapping causes (e.g.
        // RelayDown ↔ DaemonWedge) must not evade the attempt halt; only sustained
        // health or the cooldown expiry opens a fresh cycle.
        self.verdict = Some(outcome);
        self.last_probe = Some(now);
    }

    /// The cause of the last evaluation when it was a down one; an `Unknown`
    /// verdict reads as "nothing established" rather than as a cause.
    fn failure(&self) -> Option<ProbeFailure> {
        self.verdict.and_then(ProbeOutcome::failure)
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

/// Record the outcome of a Chrome auto-launch so down messaging stays honest about
/// what the watchdog did (survives until health clears it).
fn record_launch_outcome(outcome: ChromeLaunchOutcome) {
    health().lock().unwrap_poison().launch_outcome = Some(outcome);
}

/// Classify a fast CLI failure text into a health cause. Unreachable-tab is its own state
/// — daemon and relay are up, only the session's tab is orphaned, so no recovery runs for
/// it — and its signature wins over the auto-connect and daemon wrappers it may sit in.
/// The relay signature likewise beats the daemon wrapper, and both paths must agree.
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

/// Get the platform-appropriate chrome-use binary name.
pub(crate) const fn chrome_bin() -> &'static str {
    if cfg!(target_os = "windows") {
        "chrome-use.exe"
    } else {
        "chrome-use"
    }
}

/// Result of a CLI availability probe — distinguishes definitive absence from
/// transient failures so callers never report "not installed" for a
/// resource-exhaustion or wedged-binary failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CliStatus {
    Available,
    /// Definitively absent: no copy where the product installs the helper, or the
    /// resolved copy vanished.
    Missing,
    /// Present but not confirmed working.
    Transient(CliProbeFailure),
}

/// Why a CLI probe of a present binary failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CliProbeFailure {
    /// Spawn failed (EAGAIN/EMFILE/ENOMEM under process pressure, …) — retry
    /// rather than stand down.
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
/// install`): a hung registration must not block the background install task indefinitely.
const CHROME_USE_INSTALL_TIMEOUT: Duration = Duration::from_secs(120);

/// Install hint for the definitive not-found failure, on the surface that shows a
/// person the install path (the page rasteriser); the chrome tool's agent-facing
/// copy carries no install or repair steps. The helper is checked and updated on
/// every product start, so a missing one means that install failed.
pub(crate) const CHROME_USE_INSTALL_HINT: &str = "It is checked automatically every time the \
     product starts and updated when a newer release exists — if it is still \
     missing or broken the automatic install failed; check the logs and it will \
     retry on the next start.";

/// Cached path of the product's own chrome-use copy; re-resolved when the cached
/// path vanished or was never found, so a late install is picked up.
static CLI_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();

/// Absolute path of the chrome-use binary, or `None` when definitively not
/// installed. Spawns must go through this, never the bare name: the product
/// resolves its own copy, so the search path cannot change which helper runs.
pub(crate) fn cli_path() -> Option<PathBuf> {
    let mut cache = CLI_PATH
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_poison();
    // Same executability predicate as the resolver, so a cached binary that loses
    // its execute bit mid-run is re-resolved instead of pinning every probe in a
    // permanent PermissionDenied.
    if let Some(path) = cache.as_ref().filter(|p| crate::util::is_executable(p)) {
        return Some(path.clone());
    }
    let found = find_cli_binary();
    cache.clone_from(&found);
    found
}

/// Clear the cached path so a relocated binary is re-resolved on the next probe.
fn invalidate_cli_path() {
    *CLI_PATH
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_poison() = None;
}

/// Locate the product's own copy at [`crate::util::managed_bin::chrome_use_bin_path`] and
/// nothing else: the search path is never consulted, so a leftover copy in the old private
/// tools folder can never be preferred; [`crate::util::owner_path`] puts that directory on
/// the owner's path where it is missing.
fn find_cli_binary() -> Option<PathBuf> {
    let candidate = crate::util::managed_bin::chrome_use_bin_path(chrome_bin())?;
    crate::util::is_executable(&candidate).then_some(candidate)
}

/// Classify a `--version` spawn failure: only a genuinely missing binary
/// (`NotFound`) is definitive absence; every other spawn error is transient.
fn classify_spawn_error(e: &std::io::Error) -> CliStatus {
    if e.kind() == std::io::ErrorKind::NotFound {
        CliStatus::Missing
    } else {
        debug!("chrome-use CLI probe spawn failed: {e}");
        CliStatus::Transient(CliProbeFailure::Spawn(e.to_string()))
    }
}

/// Probe the CLI by running `--version` through the resolved absolute path,
/// bounded by [`CLI_TIMEOUT`], kill-on-drop, with the no-update-check env. Only
/// definitive absence reports [`CliStatus::Missing`]; spawn errors, timeouts and
/// non-zero exits are [`CliStatus::Transient`].
pub(crate) async fn cli_probe() -> CliStatus {
    let Some(path) = cli_path() else {
        return CliStatus::Missing;
    };
    let mut cmd = Command::new(&path);
    #[cfg(target_os = "windows")]
    cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    ensure_chrome_env(&mut cmd);
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

/// Extract the chrome-use version from `--version` stdout. The banner carries
/// extra lines, so scan every token for the first parseable semver rather than
/// assuming a position; the version may be `v1.5.99` or bare.
fn parse_cli_version(stdout: &str) -> Option<semver::Version> {
    stdout
        .split_whitespace()
        .find_map(crate::util::managed_bin::parse_version_token)
}

/// Release-asset platform tag for chrome-use archives, e.g. `darwin-arm64` or
/// `linux-musl-x64`; `None` on combos the vendor does not publish. Pure and
/// testable; [`release_asset_name`] adds the compile-time target triple.
#[must_use]
fn release_asset_platform(os: &str, arch: &str, musl: bool) -> Option<String> {
    let asset = match (os, arch, musl) {
        ("macos", "x86_64", _) => "darwin-x64",
        ("macos", "aarch64", _) => "darwin-arm64",
        ("linux", "x86_64", false) => "linux-x64",
        ("linux", "aarch64", false) => "linux-arm64",
        ("linux", "x86_64", true) => "linux-musl-x64",
        ("linux", "aarch64", true) => "linux-musl-arm64",
        // Windows on ARM takes the ordinary x64 asset: the vendor publishes no ARM Windows
        // build and its own `install.ps1:151-159` (v1.5.140) says so — only x64 is
        // published, and ARM64 takes `chrome-use-win32-x64` with a note rather than a
        // refusal — so that emulated build is what first install and every update must take.
        ("windows", "x86_64" | "aarch64", _) => "win32-x64",
        _ => return None,
    };
    Some(asset.to_string())
}

/// Asset platform tag for THIS build, or `Err` naming the unsupported platform.
/// Uses the compile-time target triple, and on Linux detects musl by a musl
/// loader in `/lib` (mirroring the vendor installer — no `ldd`).
fn release_asset_name() -> Result<String, String> {
    let (os, arch) = crate::util::managed_bin::host_os_arch()?;
    let musl = os == "linux" && crate::util::managed_bin::linux_host_is_musl();
    release_asset_platform(os, arch, musl)
        .ok_or_else(|| format!("chrome-use has no release asset for {os}-{arch}"))
}

/// Parse the local chrome-use version from `--version` stdout, bounded by
/// [`CLI_TIMEOUT`]. `None` when the CLI is missing, the probe fails (timeout or
/// non-zero exit), or the banner holds no parseable semver — a missing or
/// unusable copy, which the start-time policy treats as needing (re)install.
pub(crate) async fn cli_version() -> Option<semver::Version> {
    let path = cli_path()?;
    let mut cmd = Command::new(&path);
    #[cfg(target_os = "windows")]
    cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    ensure_chrome_env(&mut cmd);
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

/// A bounded CLI call through [`crate::chrome::spawn::spawn_cli`] (shared env, `--json`,
/// optional `--session`) against an already-resolved binary. `timeout` is the caller's
/// bound — the shared [`CLI_TIMEOUT`] where a call must fail fast — and the clocks come
/// from [`probe_clocks`], the one formula for a product probe rather than agent work: the
/// bound is the kill and also the clock chrome-use is given (capped at the longest
/// declaration mahbot makes for a verb it forwards no `--timeout` to). Relay self-heal is
/// suppressed and the kill armed before the child can apply its own clock, so a silent
/// probe ends at the product's bound (the "no answer at all" reading the wedge checks rest
/// on). The raw [`CliRun`] comes back so a caller can tell "nothing was asked" from "no
/// answer"; [`super::rasterize_html`] only logs which leg it took.
pub(crate) async fn run_cli_bounded_at(
    path: &Path,
    args: &[&str],
    session: Option<&str>,
    timeout: Duration,
) -> CliRun {
    let clocks = probe_clocks(timeout);
    spawn_cli(CliSpawn {
        path,
        args,
        session,
        json: true,
        capture_stderr: false,
        timeout: clocks.kill,
        cancel_kills: true,
        input: None,
        chrome_side: clocks.chrome_side,
        recovery: CliRecovery::Suppressed,
    })
    .await
}

/// The probed-path variant of [`run_cli_json_at`], bounded by [`CLI_TIMEOUT`],
/// keeping the `Option<String>` error vocabulary its callers already speak: the
/// verdict chrome-use itself reported, or `None` for every leg that carries no text.
async fn run_cli_json_opt(args: &[&str], session: Option<&str>) -> Result<Value, Option<String>> {
    let Some(path) = cli_path() else {
        return Err(None);
    };
    run_cli_json_at(&path, args, session, CLI_TIMEOUT)
        .await
        .map_err(|verdict| match verdict {
            NoVerdict::Reported(msg) => Some(msg),
            NoVerdict::TimedOut | NoVerdict::SpawnFailure | NoVerdict::Unreadable => None,
        })
}

/// Why a bounded `--json` call carried no verdict: a call that never produced a
/// child of ours was never asked anything, one killed at the product's bound stayed silent.
#[derive(Debug)]
pub(crate) enum NoVerdict {
    /// chrome-use answered with a structured error — its own text, for a caller
    /// detecting a signature in it.
    Reported(String),
    /// The child was killed at the product's own bound.
    TimedOut,
    /// Could not be spawned, or its exit could not be waited on (the two legs
    /// [`CliRun`] folds into `SpawnFailure`) — nothing was asked of the session.
    SpawnFailure,
    /// The answer carried no readable verdict: unparseable bytes, or a failure
    /// envelope naming no error.
    Unreadable,
}

impl NoVerdict {
    /// The one-line cause for a log line.
    pub(crate) fn text(&self) -> &str {
        match self {
            Self::Reported(msg) => msg,
            Self::TimedOut => "no answer within the attempt bound",
            Self::SpawnFailure => "no chrome-use child could be run",
            Self::Unreadable => "no readable envelope in the answer",
        }
    }
}

/// [`run_cli_bounded_at`] plus the `--json` envelope contract: `Ok(value)` on
/// success, else the [`NoVerdict`] leg saying why there is none.
pub(crate) async fn run_cli_json_at(
    path: &Path,
    args: &[&str],
    session: Option<&str>,
    timeout: Duration,
) -> Result<Value, NoVerdict> {
    match run_cli_bounded_at(path, args, session, timeout).await {
        CliRun::Output(out) => match json_outcome(&out) {
            Ok(v) => Ok(v),
            Err(Some(msg)) => Err(NoVerdict::Reported(msg)),
            Err(None) => Err(NoVerdict::Unreadable),
        },
        CliRun::TimedOut => Err(NoVerdict::TimedOut),
        CliRun::SpawnFailure => Err(NoVerdict::SpawnFailure),
    }
}

/// One bounded call's envelope verdict: the decoded JSON on a zero exit with
/// `success: true`, else `Err(Some(msg))` for the reported error and `Err(None)`
/// when none was named. The reading is [`envelope_outcome`].
fn json_outcome(out: &CliOutput) -> Result<Value, Option<String>> {
    envelope_outcome(out.status.success(), &out.stdout)
}

/// The verdict from the child's exit status and the bytes it wrote — pure so tests pin the
/// reading. The parse is the shared tolerant
/// [`crate::chrome::contract::parse_first`], so bytes a leftover process wrote into the
/// channel cannot turn an answered call into silence.
fn envelope_outcome(status_success: bool, stdout: &[u8]) -> Result<Value, Option<String>> {
    let Some(v) = parse_first::<Value>(stdout) else {
        return Err(None);
    };
    let env = ChromeResponse::from_value(&v);
    if !status_success || env.verdict() != Some(true) {
        return Err(env.error.filter(|e| !e.is_empty()));
    }
    Ok(v)
}

/// Non-session variant for daemon-free commands (`status`, `browsers`,
/// `extension status`): a structured error is dropped and the caller's fact
/// stays `None` — an unanswered snapshot establishes NOTHING, which is exactly
/// what readiness and the health evaluation report about it (never "healthy").
async fn run_cli_json(args: &[&str]) -> Option<Value> {
    run_cli_json_opt(args, None).await.ok()
}

/// What was established about the connection at one moment, from the daemon-free `status
/// --json` and `browsers --json` snapshots plus the Chrome-process fact. Every fact is
/// tri-state and a missing one is never defaulted to "fine": an action that runs without
/// proof went to a self-launched browser, which is a plain failure.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Readiness {
    pub(crate) answered: bool,
    pub(crate) cli_version: Option<String>,
    pub(crate) host_installed: Option<bool>,
    pub(crate) host_healthy: Option<bool>,
    /// Probed (as the health evaluation does) only when the relay is not up — the
    /// one verdict it changes.
    pub(crate) extension: Option<ExtensionState>,
    pub(crate) relay_up: Option<bool>,
    pub(crate) profile_email: Option<String>,
    pub(crate) profile_id: Option<String>,
    /// A non-empty `browsers` list — the fact a page action needs, not a file or a
    /// process being present.
    pub(crate) real_browser: Option<bool>,
    pub(crate) chrome_running: Option<bool>,
    /// A usable display session, which chrome-use needs to LAUNCH Chrome; reported
    /// when absent, but it blocks nothing — a running browser stays reachable.
    pub(crate) display: Option<bool>,
}

impl Readiness {
    /// Whether the connection is PROVEN — every decisive fact established and positive.
    /// [`Self::outcome`] reports `Healthy` for this alone, so an unanswered decisive fact
    /// (a future chrome-use renaming keys) is never health. Not the gate's rule: the gate
    /// dispatches what is neither proven nor blocked, and an unproven action that drives a
    /// self-launched browser is a plain failure
    /// ([`crate::chrome::contract::self_launched_browser_error`]).
    #[must_use]
    pub(crate) fn ready_for_actions(&self) -> bool {
        self.host_installed == Some(true)
            && self.host_healthy == Some(true)
            && self.relay_up == Some(true)
            && self.real_browser == Some(true)
    }

    /// The facts the `status --json` snapshot carries. A missing snapshot, or a
    /// missing key inside it, leaves its fact `None` — never a default.
    fn from_status(status: Option<&Value>) -> Self {
        let mut r = Self::default();
        let Some(status) = status else {
            return r;
        };
        r.answered = true;
        let data = status.get("data");
        r.cli_version = data
            .and_then(|d| d.get("cliVersion"))
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
            .map(str::to_string);
        let ext = data.and_then(|d| d.get("extension"));
        r.host_installed = ext
            .and_then(|e| e.get("hostInstalled"))
            .and_then(Value::as_bool);
        r.host_healthy = ext
            .and_then(|e| e.get("hostHealthy"))
            .and_then(Value::as_bool);
        r.relay_up = ext.and_then(|e| e.get("relayUp")).and_then(Value::as_bool);
        r.profile_email = ext
            .and_then(|e| e.get("profileEmail"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        r.profile_id = ext
            .and_then(|e| e.get("profileId"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        r
    }

    /// The `browsers --json` fact: `Some(true)` for a non-empty list of connected
    /// real profiles, `Some(false)` for an empty one, `None` when the snapshot did
    /// not answer or carries no list (a missing fact is never "reachable").
    fn real_browser_from(browsers: Option<&Value>) -> Option<bool> {
        let browsers = browsers?;
        let list = browsers
            .get("data")
            .and_then(|d| d.get("browsers"))
            .or_else(|| browsers.get("browsers"))?
            .as_array()?;
        Some(!list.is_empty())
    }

    /// The cause a fact ESTABLISHES, or `None` when none established anything wrong
    /// — nothing may be invented from a fact the probe never obtained.
    fn down_cause(&self) -> Option<ProbeFailure> {
        if self.host_installed == Some(false) {
            return Some(ProbeFailure::NotInstalled);
        }
        if self.host_healthy == Some(false) {
            return Some(ProbeFailure::HostBroken);
        }
        if self.relay_up == Some(false) {
            // The pinned precedence: an unfixable extension cause first (no
            // launch or restart can fix it), then the browser-running probe,
            // then the transient relay drop.
            return Some(classify_relay_down(
                self.extension.unwrap_or(ExtensionState::Unknown),
                self.chrome_running,
            ));
        }
        match self.extension {
            Some(ExtensionState::Disabled) => return Some(ProbeFailure::ExtensionDisabled),
            Some(ExtensionState::Absent) => return Some(ProbeFailure::ExtensionAbsent),
            _ => {}
        }
        if self.real_browser == Some(false) {
            // The relay answers but no browser profile is connected through it —
            // the relay's browser side dropped, and the extension republishes on
            // its own keepalive.
            return Some(ProbeFailure::RelayDown);
        }
        None
    }

    /// Whether a fact ESTABLISHED that the connection cannot work — the gate's
    /// blocking condition, narrower than "not proven": an unanswered snapshot
    /// neither blocks an action nor spends a restart.
    #[must_use]
    pub(crate) fn blocked(&self) -> bool {
        self.down_cause().is_some()
    }

    /// The health evaluation this snapshot implies: `Healthy` only for a PROVEN
    /// connection, a down cause when a fact established one, `Unknown` otherwise.
    fn outcome(&self) -> ProbeOutcome {
        if self.ready_for_actions() {
            ProbeOutcome::Healthy
        } else {
            self.down_cause()
                .map_or(ProbeOutcome::Unknown, ProbeOutcome::Down)
        }
    }

    /// The report's three lists from ONE derivation, so the report and the refusal
    /// can never disagree: facts established, established as NOT holding, and never
    /// established. An established negative counts as knowledge, not as a gap.
    fn facts(&self) -> (Vec<String>, Vec<String>, Vec<String>) {
        let mut held: Vec<String> = Vec::new();
        let mut negated: Vec<String> = Vec::new();
        let mut unknown: Vec<String> = Vec::new();
        if self.answered {
            held.push(match &self.cli_version {
                Some(v) => format!("the tool answered `status` (chrome-use {v})"),
                None => "the tool answered `status`".to_string(),
            });
        } else {
            unknown.push("whether the tool answers `status` at all".to_string());
        }
        tri_state(
            &mut held,
            &mut negated,
            &mut unknown,
            self.host_installed,
            "the native host is installed",
            "the native host is NOT installed",
            "whether the native host is installed",
        );
        tri_state(
            &mut held,
            &mut negated,
            &mut unknown,
            self.host_healthy,
            "the native host is healthy",
            "the native host is NOT healthy (broken launcher)",
            "whether the native host is healthy",
        );
        match self.extension {
            Some(ExtensionState::Present) => {
                held.push("the chrome-use extension is installed and enabled".to_string());
            }
            Some(ExtensionState::Disabled) => {
                negated.push("the chrome-use extension is installed but DISABLED".to_string());
            }
            Some(ExtensionState::Absent) => {
                negated.push("the chrome-use extension is NOT installed in Chrome".to_string());
            }
            Some(ExtensionState::Unknown) => {
                unknown.push("whether the chrome-use extension is installed".to_string());
            }
            None => {}
        }
        tri_state(
            &mut held,
            &mut negated,
            &mut unknown,
            self.relay_up,
            "the extension relay is up",
            "the extension relay is down",
            "whether the extension relay is up",
        );
        match (&self.profile_email, &self.profile_id) {
            (Some(email), Some(id)) => {
                held.push(format!("driving the real profile {email} ({id})"));
            }
            (Some(email), None) => held.push(format!("driving the real profile {email}")),
            (None, Some(id)) => held.push(format!("driving the real profile {id}")),
            (None, None) => unknown.push("which real profile is being driven".to_string()),
        }
        tri_state(
            &mut held,
            &mut negated,
            &mut unknown,
            self.real_browser,
            "a real browser is reachable through the relay",
            "no real browser profile is reachable through the relay",
            "whether a real browser is reachable through the relay",
        );
        // The process fact is deliberately phrased as a fact, never as health.
        match self.chrome_running {
            Some(true) => held.push(
                "a Chrome-family process is running (process presence is not by itself health)"
                    .to_string(),
            ),
            Some(false) => negated.push("no Chrome-family process is running".to_string()),
            None => unknown.push("whether a Chrome-family process is running".to_string()),
        }
        tri_state(
            &mut held,
            &mut negated,
            &mut unknown,
            self.display,
            "this host has a usable display (what chrome-use needs to launch the owner's Chrome)",
            "this host has NO usable display (chrome-use cannot launch the owner's Chrome here; a \
             browser already running is still reachable)",
            "whether this host has a usable display",
        );
        (held, negated, unknown)
    }

    /// The factual report: what was established, what was established as not holding, what
    /// was never established, and the verdict — never "fine" when an action cannot be
    /// carried out, never a process presence presented as health.
    #[must_use]
    pub(crate) fn report(&self) -> String {
        let (held, negated, unknown) = self.facts();
        let mut out = String::from("what was established about the owner's real browser:\n");
        out.push_str("  established:\n");
        push_fact_list(&mut out, &held);
        out.push_str("  established as NOT holding:\n");
        push_fact_list(&mut out, &negated);
        out.push_str("  not established:\n");
        push_fact_list(&mut out, &unknown);
        out.push_str("  verdict: ");
        if self.ready_for_actions() {
            out.push_str(
                "ready for actions — a page action goes to the owner's own logged-in browser\n",
            );
        } else if self.blocked() {
            out.push_str(
                "NOT ready for actions — the facts above rule out a connection to the owner's own \
                 browser, so an action is refused instead of running in a browser chrome-use would \
                 launch itself\n",
            );
        } else {
            out.push_str(
                "NOT proven ready for actions — nothing above rules the connection out, but every \
                 fact listed as not established could not be checked, so an action would still be \
                 attempted and a browser chrome-use launches itself is reported as a failure\n",
            );
        }
        out
    }

    /// The plain refusal for a snapshot a fact ruled out: what is missing and what
    /// would have happened otherwise. An established negative and an unestablished
    /// fact are both reasons the connection cannot be established.
    #[must_use]
    pub(crate) fn refusal(&self) -> String {
        let (_, negated, unknown) = self.facts();
        let missing: Vec<String> = negated.into_iter().chain(unknown).collect();
        let missing = if missing.is_empty() {
            "the connection to the owner's real browser".to_string()
        } else {
            missing.join("; ")
        };
        format!(
            "cannot establish a connection to the owner's real browser: {missing}. Without it the \
             work would have gone to a browser chrome-use launches itself — not the owner's own \
             logged-in Chrome — so the page state an action assumed would not be there and nothing \
             it read or wrote would land in the owner's session."
        )
    }
}

/// Push one group of the report's facts, or `- nothing` when the group is empty.
fn push_fact_list(out: &mut String, facts: &[String]) {
    if facts.is_empty() {
        out.push_str("    - nothing\n");
    }
    for line in facts {
        let _ = writeln!(out, "    - {line}");
    }
}

/// Push a tri-state fact onto its list: an explicit `Some(false)` is an ESTABLISHED
/// negative, and only `None` means nothing was established.
fn tri_state(
    held: &mut Vec<String>,
    negated: &mut Vec<String>,
    unknown: &mut Vec<String>,
    value: Option<bool>,
    held_text: &str,
    negated_text: &str,
    unknown_text: &str,
) {
    match value {
        Some(true) => held.push(held_text.to_string()),
        Some(false) => negated.push(negated_text.to_string()),
        None => unknown.push(unknown_text.to_string()),
    }
}

/// Probe readiness now (uncached) from the daemon-free snapshots: `status --json` carries
/// the version, native-host facts, relay state and driven profile, and `browsers --json`'s
/// non-empty profile list is what proves the owner's browser is reachable. The extension
/// probe runs only when the relay is not up. A snapshot that does not answer leaves its
/// facts `None`, but a structured ERROR is still an answer: `answered` stays true.
async fn probe_readiness() -> Readiness {
    let status = run_cli_json_opt(&["status"], None).await;
    let mut readiness = match &status {
        Ok(value) => Readiness::from_status(Some(value)),
        Err(Some(_)) => Readiness {
            answered: true,
            ..Readiness::default()
        },
        Err(None) => Readiness::default(),
    };
    if readiness.relay_up != Some(true) {
        readiness.extension = Some(extension_state().await);
    }
    readiness.chrome_running = chrome_running().await;
    readiness.real_browser =
        Readiness::real_browser_from(run_cli_json(&["browsers"]).await.as_ref());
    readiness.display = Some(display_available());
    if let Ok(status) = status.as_ref() {
        // The single `status` spawn also drives the extension-skew advisory.
        advise_extension_skew(status);
    }
    readiness
}

/// The last readiness snapshot while it is still fresh — the cache that keeps a
/// burst of per-call gates from re-probing, on the healthy-verdict window. It lives
/// in [`DaemonHealth`] beside the verdict derived from it, so one probe is one record.
fn cached_readiness() -> Option<Readiness> {
    health()
        .lock()
        .unwrap_poison()
        .readiness
        .as_ref()
        .filter(|(at, _)| at.elapsed() < HEALTH_TTL)
        .map(|(_, readiness)| readiness.clone())
}

/// The readiness snapshot for this moment: a fresh cached one when there is one,
/// otherwise a fresh probe, which also feeds the shared health verdict so
/// advertisement and the watchdog stay in step.
pub(crate) async fn readiness() -> Readiness {
    if let Some(cached) = cached_readiness() {
        return cached;
    }
    probe_and_record(true).await
}

/// The last snapshot taken, whatever its age — what [`daemon_down_message`] reports
/// on, which the TTL-bounded cache would drop.
fn last_readiness() -> Option<Readiness> {
    health()
        .lock()
        .unwrap_poison()
        .readiness
        .as_ref()
        .map(|(_, readiness)| readiness.clone())
}

/// The ONE place a probe's result is recorded: probe, store the snapshot and apply the
/// verdict to the same record, so they never come from different evaluations. `seeding`
/// decides whether a healthy result opens the sustained-healthy window — a probe of its own
/// does, the verification right after a recovery action ([`attempt_chrome_launch`],
/// [`attempt_recovery`]) must not. The watchdog is not woken: the caller's recovery already
/// acted, and the next tick re-evaluates.
async fn probe_and_record(seeding: bool) -> Readiness {
    let readiness = probe_readiness().await;
    let outcome = readiness.outcome();
    let now = Instant::now();
    let mut h = health().lock().unwrap_poison();
    h.readiness = Some((now, readiness.clone()));
    h.apply_outcome(outcome, now, seeding);
    readiness
}

/// The pre-action gate. A PROVEN snapshot returns without probing, so the cache
/// keeps it cheap enough to run on every call. An established cause runs ONE
/// bounded, cause-aware recovery pass (the watchdog's machinery and budgets: launch
/// a closed browser, wait out a relay drop, spend a restart on a wedge, recover
/// nothing for an unfixable cause), re-probes, and refuses only while a cause still
/// stands. Anything else — not proven, nothing ruled out — dispatches: no fact
/// established anything wrong, no destructive restart is justified without one, and
/// refusing on a reporting gap would make both chrome surfaces unavailable.
pub(crate) async fn ensure_ready_for_actions() -> Result<(), String> {
    let readiness = readiness().await;
    if readiness.ready_for_actions() {
        return Ok(());
    }
    let Some(cause) = readiness.down_cause() else {
        return Ok(());
    };
    attempt_recovery(cause).await;
    let readiness = probe_and_record(true).await;
    if readiness.blocked() {
        Err(readiness.refusal())
    } else {
        Ok(())
    }
}

/// For callers that only need "could this host drive a page at all" (the sweep's
/// skip gate): the cause a fact established, or `None`, from the shared cached view.
async fn service_state() -> Option<ProbeFailure> {
    readiness().await.down_cause()
}

/// Extension presence from `extension status --json` (daemon-free; reads Chrome's
/// Secure Preferences).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExtensionState {
    Present,
    Disabled,
    Absent,
    Unknown,
}

/// Parse extension presence from an `extension status --json` value. An explicit null
/// `chromeExtension` means the extension is not installed in Chrome at all — distinct from a
/// missing native host and from `Disabled`. A MISSING key on a successful envelope is
/// shape-uncertain (old CLI), so it never claims the unfixable absent cause; it fails open as
/// `Unknown`, as a FAILED status does. Shared with the run-end release's door check
/// (`crate::tools::chrome_tabs`), so the absence rule has one home.
pub(crate) fn extension_state_from(status: &Value) -> ExtensionState {
    match status.get("data").and_then(|d| d.get("chromeExtension")) {
        // A missing key on a SUCCESSFUL envelope is shape-uncertain (old CLI, or one that
        // omits it) — never claim the unfixable absent cause from it; it fails open.
        None => ExtensionState::Unknown,
        // The CLI's absence signal: an explicit null chromeExtension.
        Some(c) if c.is_null() => ExtensionState::Absent,
        Some(c) => {
            if c.get("disableReasons")
                .and_then(Value::as_array)
                .is_some_and(|r| !r.is_empty())
            {
                ExtensionState::Disabled
            } else {
                ExtensionState::Present
            }
        }
    }
}

/// Whether the extension is present, disabled or absent, from `extension status --json`. A
/// failed command is never "absent": unknown fails open to the transient-relay path, since a
/// CLI hiccup must not be reported as an uninstalled extension.
async fn extension_state() -> ExtensionState {
    run_cli_json_opt(&["extension", "status"], None)
        .await
        .map_or(ExtensionState::Unknown, |v| extension_state_from(&v))
}

/// Relay-down cause with deterministic precedence: unfixable Chrome-side causes first (no
/// recovery can fix them), then the browser-running probe, then the transient relay drop.
/// `chrome_running == None` is inconclusive — degrade to RelayDown.
fn classify_relay_down(ext: ExtensionState, chrome_running: Option<bool>) -> ProbeFailure {
    match ext {
        ExtensionState::Disabled => ProbeFailure::ExtensionDisabled,
        ExtensionState::Absent => ProbeFailure::ExtensionAbsent,
        ExtensionState::Present | ExtensionState::Unknown => {
            if chrome_running == Some(false) {
                ProbeFailure::ChromeNotRunning
            } else {
                ProbeFailure::RelayDown
            }
        }
    }
}

/// macOS app process names for Chrome/Chromium-family browsers.
#[cfg(not(target_os = "windows"))]
const CHROME_PROCESS_NAMES: [&str; 6] = [
    "Google Chrome",
    "Chromium",
    "Brave Browser",
    "chrome",
    "chromium",
    "brave",
];
/// Windows image names for the same browsers.
#[cfg(target_os = "windows")]
const WINDOWS_CHROME_PROCESS_NAMES: [&str; 3] = ["chrome.exe", "chromium.exe", "brave.exe"];

/// Whether a Chrome/Chromium-family browser is running: one `pgrep -x` spawn over the known
/// names on Unix, `tasklist` filters on Windows. A match → `Some(true)`, no match →
/// `Some(false)`, spawn error / other exit / timeout → `None`, which the classifier degrades
/// to RelayDown. A closed browser is recovered by launching it, not by restarting.
pub(crate) async fn chrome_running() -> Option<bool> {
    #[cfg(not(target_os = "windows"))]
    {
        pgrep_running(&CHROME_PROCESS_NAMES.join("|")).await
    }
    #[cfg(target_os = "windows")]
    {
        for name in WINDOWS_CHROME_PROCESS_NAMES {
            if tasklist_has(name).await? {
                return Some(true);
            }
        }
        Some(false)
    }
}

/// Single `pgrep -x <ERE alternation>` probe: exit 0 → running, 1 → none, anything
/// else → inconclusive.
#[cfg(not(target_os = "windows"))]
async fn pgrep_running(pattern: &str) -> Option<bool> {
    let mut cmd = Command::new("pgrep");
    cmd.args(["-x", pattern])
        // Only the exit code matters — a piped-but-undrained stdout kills a
        // matching pgrep with SIGPIPE before it can report its status.
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let status = tokio::time::timeout(CLI_TIMEOUT, cmd.status())
        .await
        .ok()?
        .ok()?;
    match status.code() {
        Some(0) => Some(true),
        Some(1) => Some(false), // no process matched
        _ => None,
    }
}

/// `tasklist` image-name probe (Windows): CSV output containing the image name
/// → running; empty → no match; invocation failure → inconclusive.
#[cfg(target_os = "windows")]
async fn tasklist_has(name: &str) -> Option<bool> {
    let mut cmd = Command::new("tasklist");
    cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    cmd.args(["/FI", &format!("IMAGENAME eq {name}"), "/NH", "/FO", "CSV"])
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
    Some(String::from_utf8_lossy(&out.stdout).contains(name))
}

/// One tab in a session's tab group, from `tab list --json`. Identity is
/// `target_id`, the stable CDP id that survives daemon restarts, while `tab_id`
/// (`t<N>`) is the ref `close` resolves. `active` is not tracked: a fresh daemon
/// pins adopted leftovers exactly like its own scratch.
struct SweepTab {
    tab_id: String,
    target_id: String,
}

// chrome-use CLI behaviors this sweep relies on (live-verified against 1.5.100; the
// installed copy is updated when a newer release exists, and ≥1.5.101 idle keeps
// external tabs alive — the explicit close/stop below still cleans up):
// - `tab list --session <name>` enumerates only that session's group (the relay scopes
//   `Target.getTargets` per announced group, leeguooooo/chrome-use#40). With no daemon it
//   spawns one: an empty group gets a fresh scratch tab, a non-empty one is ADOPTED
//   unmarked. Created and adopted tabs are both pinned, so `active: true` does not identify
//   the daemon's own tab — the sweep tracks its scratch by stable `targetId`.
// - `close <ref>` refuses to close the last tab of a session ("Cannot close the last
//   tab"), so the sweep creates its scratch first to keep the count ≥ 2. The daemon
//   discards the closeTarget result, so only re-enumeration proves closure: a failed close
//   REAPPEARS in the next same-daemon `tab list` — that resync is the loop.
// - `session stop` SIGTERMs the daemon (its shutdown handler closes created tabs
//   best-effort), waits out the shutdown grace (8 s on the installed CLI, 1 s before
//   leeguooooo/chrome-use#192), then SIGKILLs, and reclaims the session's PERSISTED
//   created-tab record under its own 20 s timeout — ≈28 s end to end, the whole of what one
//   stop may legitimately spend (why the ended-run release gives a stop that much of its
//   pass budget before its stop-by-name fallback) — dropping an id only when its close was
//   acknowledged. So a successful stop proves every CREATED tab is gone and nothing about
//   an ADOPTED one: hence the re-enumeration, not the exit code.
//
// Residual limits (accepted): the scratch is about:blank and the extension never re-attaches
// `about:` URLs (its `eligible()`/`SKIP_URL` filter). An orphan that lost its attach while
// the daemon kept a stale binding fails every command with the unreachable-tab signatures,
// which the sweep logs while retrying; one the extension fully dropped (a service-worker
// restart unmarks ineligible about: tabs) is invisible to every CLI path, so the sweep
// converges to Clean with no log — undetectable by design. An `agent-tab-*` orphan of that
// shape is still reached by the run-end release, which enumerates live tab groups by title
// (`tools::chrome_tabs`' honest limits); a link-enricher orphan stays until closed by hand.
/// Close every tab in a mahbot-owned session's group except the sweep's own scratch,
/// verifying by round-over-round re-enumeration. Shared by the startup sweep and the
/// link-enricher per-fetch close; the refusal below fails closed for a foreign name.
pub(crate) async fn sweep_session(name: &str) {
    if !is_mahbot_session_name(name) {
        warn!(
            session = name,
            "tab sweep refused: not a mahbot-owned session (user/default/other-agent sessions are never touched)"
        );
        return;
    }
    // No close is possible while the relay is down and every CLI call would cost the full
    // step timeout, so skip; the deadline starts here so the gate counts against the budget.
    let deadline = Instant::now() + SWEEP_TOTAL_BUDGET;
    if let Some(failure) = service_state().await {
        debug!(
            session = name,
            ?failure,
            "tab sweep skipped — chrome service unavailable"
        );
        return;
    }
    // The ONE tab the sweep creates via `tab new`, tracked by stable targetId; everything
    // else in the group is a leftover to close. `stopped` marks the round after a stop:
    // enumeration then spawned a fresh daemon, so a lone tab is provably that daemon's own
    // scratch (clean) unless it is our tracked scratch, which survived and was adopted.
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
            let _ = stop_session_daemon(name, deadline).await;
            stopped = true;
            scratch = None;
            continue;
        }
        if stopped {
            // Verification round: a lone tab that is not our scratch means the stop
            // closed ours and the fresh daemon made its own (clean).
            if tabs.len() == 1 && scratch.as_deref() != Some(tabs[0].target_id.as_str()) {
                clear_sweep_warn();
                let _ = stop_session_daemon(name, deadline).await;
                return;
            }
            stopped = false;
            scratch = None; // adopted by the fresh daemon — no longer owned
        }
        if tabs.len() == 1 && scratch.as_deref() == Some(tabs[0].target_id.as_str()) {
            // Only our scratch remains, so every leftover is closed and verified.
            let _ = stop_session_daemon(name, deadline).await;
            stopped = true;
            continue;
        }
        if scratch.is_none() {
            let Some(target_id) = session_tab_new_scratch(name, deadline).await else {
                return; // every None path already emitted its SweepWarn
            };
            scratch = Some(target_id);
        }
        for tab in &tabs {
            if tab.target_id == *scratch.as_deref().unwrap_or_default() {
                continue; // never close our own scratch (last-tab rule)
            }
            if Instant::now() >= deadline {
                break;
            }
            let _ = session_close_tab(name, &tab.tab_id, deadline).await;
        }
    }
    sweep_warn_transition(SweepWarn::Deferred);
}

/// Only mahbot-owned session names may be swept — link-enricher-* and ephemeral
/// `mahbot-chrome-ephemeral-*` sessions (orphan protection). `mahbot-chrome-<name>`
/// and user/default/agent-tab sessions are never touched. The legacy
/// `mahbot-browser-ephemeral-*` prefix still matches so older orphans do not leak.
pub(crate) fn is_mahbot_session_name(name: &str) -> bool {
    name.starts_with("link-enricher-")
        || name.starts_with(crate::chrome::CLI_EPHEMERAL_PREFIX)
        || name.starts_with("mahbot-browser-ephemeral-")
}

/// Causes the sweep warns about — once per cause transition, so a persistent orphan
/// does not spam every sweep and a healthy sweep clears the previous cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SweepWarn {
    /// Leftover tab the daemon can no longer re-drive (stale binding on an
    /// about:blank tab never re-attached); the session's own tab, closed by the
    /// product at run end or session stop. Only fires while a command still errors.
    UnreachableTab,
    /// Relay/daemon unreachable mid-sweep; retried next sweep/startup.
    CannotEnumerate,
    /// Budget exhausted without convergence; retried next sweep/startup.
    Deferred,
}

/// Last-cause anti-spam state, global across sessions: a clean sweep in one session
/// clears it for all, so a persistent orphan elsewhere may re-warn once after that
/// convergence — an accepted trade for not keeping a per-session map.
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
             about:blank tabs are never re-attached) — the tab belongs to the session and the \
             product closes the session's tabs itself when the run ends (or the session is \
             stopped); the sweep keeps retrying meanwhile"
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

/// Map a session-CLI error to its [`SweepWarn`] and return `None` for the caller's
/// `Option<T>`; generic over the Ok type so `tab list` and `tab new` share it, and
/// every error emits its warn. A session-unresponsive error is RECOVERED here
/// through the same shared helper as every other path — the daemon-side calls hit
/// the same signature, so the background sweep recovers the wedge instead of only
/// logging it, then defers (the next sweep sees the fresh daemon). This is the one
/// step that may take the sweep past [`SWEEP_TOTAL_BUDGET`] — deliberately, see
/// [`SESSION_RECOVERY_TIMEOUT`].
async fn sweep_none_on_cli_error<T>(name: &str, err: Option<&str>) -> Option<T> {
    let msg = err.unwrap_or_default();
    if is_unreachable_tab_error(msg) {
        tracing::debug!(
            session = name,
            error = msg,
            "tab sweep: unreachable-tab detail"
        );
        sweep_warn_transition(SweepWarn::UnreachableTab);
    } else {
        if is_session_unresponsive_error(msg) {
            let recovery = recover_unresponsive_session(name).await;
            info!(
                session = name,
                recovery = recovery.summary(),
                "tab sweep: wedged session recovered"
            );
        }
        sweep_warn_transition(SweepWarn::CannotEnumerate);
    }
    None
}

/// Bounded `tab list --json` on a session. `None` on timeout, CLI failure, an error
/// response, or a malformed entry (a missing id makes the count unreliable — defer
/// rather than risk a false clean). Every `None` path emits its [`SweepWarn`].
async fn session_tab_list(name: &str, deadline: Instant) -> Option<Vec<SweepTab>> {
    if Instant::now() >= deadline {
        sweep_warn_transition(SweepWarn::Deferred);
        return None;
    }
    let v = match run_session_cli_json(&["tab", "list"], name).await {
        Ok(v) => v,
        Err(err) => return sweep_none_on_cli_error(name, err.as_deref()).await,
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

/// Create a scratch tab and return its stable targetId, matched through the `t<N>`
/// ref from the `tab new` response in the next enumeration (refs are stable until a
/// stop). Every `None` path emits its [`SweepWarn`] already, so callers return
/// without overriding a more specific cause.
async fn session_tab_new_scratch(name: &str, deadline: Instant) -> Option<String> {
    if Instant::now() >= deadline {
        sweep_warn_transition(SweepWarn::Deferred);
        return None;
    }
    let resp = match run_session_cli_json(&["tab", "new"], name).await {
        Ok(v) => v,
        Err(err) => return sweep_none_on_cli_error(name, err.as_deref()).await,
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

/// Bounded `session stop` — its outcome is deliberately ignored, since the sweep
/// proves closure by re-enumeration. Skipped when already over budget (the daemon
/// idles out and the next sweep retries).
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

/// What the bounded session recovery did, for the caller's error text: the enum lets
/// a caller with a manual recovery of its own name that flow only when the automatic
/// one could not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionRecovery {
    /// The daemon was stopped within the bound — the next call gets a fresh one.
    Stopped,
    /// The stop was ISSUED but had not confirmed within the bound; it keeps running
    /// (it is what reclaims the session's tabs), so only a session that still does
    /// not answer afterwards needs the manual flow.
    Unanswered,
    /// No `session stop` could be STARTED (no binary, a spent bound, a spawn
    /// failure), so the session is in whatever state it was found in.
    NotStarted,
}

impl SessionRecovery {
    /// One plain sentence for the caller's error text.
    #[must_use]
    pub(crate) fn summary(self) -> &'static str {
        match self {
            Self::Stopped => {
                "The wedged session's daemon was stopped, so the next call starts a fresh one \
                 (cookies persist in the profile; open tabs do not)."
            }
            Self::Unanswered => {
                "The `session stop` for the wedged session was issued and had not confirmed \
                 within the bound — the stop itself keeps running and reclaims the session's \
                 tabs, so the session answers again once it lands."
            }
            Self::NotStarted => {
                "No `session stop` could be started for the wedged session, so the session is in \
                 whatever state you found it in."
            }
        }
    }
}

/// Record a session wedge under the same `DaemonWedge` classification the fail-fast
/// path uses, and wake the watchdog so the background health path recovers too.
fn note_session_wedge() {
    set_health(ProbeOutcome::Down(ProbeFailure::DaemonWedge));
    wake().notify_one();
}

/// Bounded `session stop` through an already-resolved binary — the session recovery the
/// daemon's unresponsive-session paths share, under [`SESSION_RECOVERY_TIMEOUT`]. A sweep
/// that recovers a wedge therefore overshoots [`SWEEP_TOTAL_BUDGET`] deliberately: an
/// unrecovered wedge is worse, and the sweep is self-healing. The stop is UNINTERRUPTIBLE
/// ([`CliSpawn::cancel_kills`] false), because the tail the bound would cut off is exactly
/// the tab reclaim that closes the session's group — so the bound decides only when THIS
/// CALLER stops waiting and reports [`SessionRecovery::Unanswered`], never whether the
/// reclaim happens.
async fn session_stop_via(path: &Path, session: &str, bound: Duration) -> SessionRecovery {
    if bound.is_zero() {
        // A spent budget must not spawn a child that is killed on arrival.
        return SessionRecovery::NotStarted;
    }
    let recovered = session_stop_outcome(path, session, bound).await;
    match recovered {
        SessionRecovery::Stopped => debug!(
            session,
            "chrome daemon: wedged session recovered (its daemon was stopped)"
        ),
        SessionRecovery::Unanswered => warn!(
            session,
            "chrome daemon: the `session stop` for a wedged session had not confirmed within the \
             bound (the stop keeps running and reclaims its tabs)"
        ),
        SessionRecovery::NotStarted => {}
    }
    recovered
}

/// The bounded, never-killed `session stop` spawn the session-level paths use (wedge
/// recovery, the rasterize drop, the command line's lifetime close):
/// [`run_cli_json_at`]'s judgement of the same `--json` contract on the same clocks
/// ([`probe_clocks`]), spawned so the call is never killed — the only difference is
/// [`CliSpawn::cancel_kills`], which fail-fast probes need and this stop must not
/// have. The answer distinguishes a confirmed stop from one issued but not confirmed
/// (the child ran) and from one that never started.
pub(super) async fn session_stop_outcome(
    path: &Path,
    session: &str,
    bound: Duration,
) -> SessionRecovery {
    let clocks = probe_clocks(bound);
    match spawn_cli(CliSpawn {
        path,
        args: &["session", "stop"],
        session: Some(session),
        json: true,
        capture_stderr: false,
        timeout: clocks.kill,
        cancel_kills: false,
        input: None,
        chrome_side: clocks.chrome_side,
        recovery: CliRecovery::Suppressed,
    })
    .await
    {
        CliRun::Output(out) if json_outcome(&out).is_ok() => SessionRecovery::Stopped,
        CliRun::Output(_) | CliRun::TimedOut => SessionRecovery::Unanswered,
        CliRun::SpawnFailure => SessionRecovery::NotStarted,
    }
}

/// Close one session this product's command line created, for the CLI session
/// lifetime ([`crate::chrome::cli_sessions`]): [`session_stop_outcome`] with the
/// product's own stop bound.
pub(crate) async fn stop_session_by_name(session: &str) -> SessionRecovery {
    let Some(path) = cli_path() else {
        return SessionRecovery::NotStarted;
    };
    session_stop_outcome(&path, session, SESSION_STOP_TIMEOUT).await
}

/// The shared recovery for a session that stopped answering: record the wedge (which
/// wakes the watchdog, so the background path recovers too), then stop that session's
/// daemon within [`SESSION_RECOVERY_TIMEOUT`] so the next call gets a clean one. The
/// wedge is recorded even when no binary resolves — the watchdog does not depend on
/// this caller's stop, which is then [`SessionRecovery::NotStarted`].
pub(crate) async fn recover_unresponsive_session(session: &str) -> SessionRecovery {
    note_session_wedge();
    match cli_path() {
        Some(path) => session_stop_via(&path, session, SESSION_RECOVERY_TIMEOUT).await,
        None => SessionRecovery::NotStarted,
    }
}

fn set_health(outcome: ProbeOutcome) {
    let mut h = health().lock().unwrap_poison();
    h.apply_outcome(outcome, Instant::now(), true);
}

/// Sync availability for tool advertisement, from the last known state (never
/// evaluates). Only a confirmed-down verdict hides the tool: `Unknown` is advertised
/// optimistically, since advertisement is not a health claim.
pub(crate) fn is_advertised() -> bool {
    !matches!(
        health().lock().unwrap_poison().verdict,
        Some(ProbeOutcome::Down(_))
    )
}

/// Mark the daemon unhealthy immediately (fail-fast path) with the cause the error
/// text points to, and wake the watchdog so recovery starts without waiting for the
/// next interval. Unreachable-tab errors never reach here — the chrome tool's
/// fail-fast guard bails first, since recovery cannot fix a Chrome-side orphan.
pub(crate) fn note_unhealthy(error: &str) {
    // Same classification as the watchdog's health evaluation — the two
    // detection paths must agree on the cause.
    set_health(ProbeOutcome::Down(
        classify_failure_text(error).unwrap_or(ProbeFailure::DaemonWedge),
    ));
    wake().notify_one();
}

/// Appended to every integration-level failure the agent reads, whatever the cause:
/// report it as it came, leave the repair to the owner (an agent cannot repair the
/// helper, host or Chrome), and try the browser-free workaround. Deliberately no
/// install/enable/launch steps — those live in [`report_cause`]'s copy.
pub(crate) const REPORT_AS_IT_CAME: &str = " Report it as it came rather than paraphrased, and leave the \
     repair to the owner — it is not something an agent can fix. Meanwhile `web_search` or \
     `shell` with `curl` can fetch a page instead.";

/// The down message for the agent driving the chrome tool: [`daemon_down_cause`]
/// plus what the agent does about it ([`REPORT_AS_IT_CAME`]).
pub(crate) fn daemon_down_message() -> String {
    format!("{}{REPORT_AS_IT_CAME}", daemon_down_cause())
}

/// The cause as established plus what auto-recovery is doing about it, with no report
/// tail — the shared readiness gate answers with this, and the page rasteriser reads
/// it too.
fn daemon_down_cause() -> String {
    let answered = last_readiness().filter(|r| r.answered);
    let h = health().lock().unwrap_poison();
    if h.verdict == Some(ProbeOutcome::Unknown) {
        return unknown_down_cause(answered.as_ref());
    }
    let failure = h.failure();
    let cause = match failure {
        Some(ProbeFailure::NotInstalled) => {
            "The chrome-use extension or its native host is not installed, so the browser \
             daemon cannot run."
        }
        Some(ProbeFailure::HostBroken) => {
            "The chrome-use native host launcher is broken, so the browser daemon cannot run."
        }
        Some(ProbeFailure::ExtensionDisabled) => {
            "The chrome-use extension is disabled in Chrome, so nothing can reach the browser. \
             No daemon restart fixes a Chrome-side disable."
        }
        Some(ProbeFailure::ExtensionAbsent) => {
            "The chrome-use extension is not installed in Chrome, so nothing can reach the \
             browser. Daemon restarts and launches cannot fix an absent extension."
        }
        Some(ProbeFailure::RelayDown)
            if h.launch_outcome == Some(ChromeLaunchOutcome::Launched) =>
        {
            "Chrome is running, but the extension is not republishing its relay — the usual \
             cause is the extension being installed only in a non-default profile."
        }
        Some(ProbeFailure::RelayDown) => {
            "The chrome-use extension is enabled but its relay is down, so nothing reaches the \
             browser."
        }
        Some(ProbeFailure::ChromeNotRunning) => match h.launch_outcome {
            None | Some(ChromeLaunchOutcome::Launched) => "Chrome is not running.",
            Some(ChromeLaunchOutcome::Failed) => {
                "Chrome is not running and the auto-launch failed on this host (no browser \
                 binary, or it could not start)."
            }
            Some(ChromeLaunchOutcome::NoDisplay) => {
                "Chrome is not running and this host has no display."
            }
        },
        Some(ProbeFailure::UnreachableTab) => {
            "A browser tab the session was driving is unreachable (the extension lost its \
             debugger attach; about:blank tabs are never re-attached). The tab belongs to the \
             session and the product closes the session's tabs itself when the run ends (or \
             the session is stopped/recovered), so no manual step is needed."
        }
        Some(ProbeFailure::DaemonWedge) | None => "The chrome daemon is down or unresponsive.",
    };
    let recovery = if matches!(failure, Some(ProbeFailure::ChromeNotRunning)) {
        // ChromeNotRunning has its own launch budget, so a halted RESTART state
        // must not produce restart-halt text here.
        match h.launch_outcome {
            Some(ChromeLaunchOutcome::NoDisplay) => {
                " Auto-recovery is paused on this host until a display is available."
            }
            _ if h.launch_budget.halted => {
                " Launch attempts are paused in a 30-minute cooldown after repeated failures; \
                 they resume automatically after it."
            }
            Some(ChromeLaunchOutcome::Failed) => " Auto-recovery retries the launch with backoff.",
            None | Some(ChromeLaunchOutcome::Launched) => {
                " Auto-recovery launches Chrome and keeps retrying, so a retry in a moment can \
                 work."
            }
        }
    } else if h.restart_budget.halted {
        " Restart attempts are paused in a 30-minute cooldown after repeated failures; they \
         resume automatically after it."
    } else if failure.is_some_and(ProbeFailure::is_unfixable) {
        " Auto-recovery is paused for this cause and resumes once it is resolved."
    } else {
        " Auto-recovery restarts it automatically and retries (the restart resets chrome \
         sessions), so retry once after that."
    };
    format!("{cause}{recovery}")
}

/// The down cause for an `Unknown` verdict: nothing is named as a cause and nothing
/// is restarted. When the tool DID answer (a fact inside its answer was missing), the
/// snapshot's refusal names what could not be established; only a silent tool is
/// reported as not having answered.
fn unknown_down_cause(answered: Option<&Readiness>) -> String {
    const TAIL: &str = " No cause was established, so nothing is restarted and re-checking \
                        continues automatically.";
    answered.map_or_else(
        || {
            format!(
                "The chrome-use CLI did not answer, so nothing could be established about the \
                 browser daemon — neither a working connection nor a classified failure.{TAIL}"
            )
        },
        |snapshot| format!("{}{TAIL}", snapshot.refusal()),
    )
}

/// The gate's refusal as a caller should see it: the readiness refusal, plus
/// [`daemon_down_cause`] when the stored verdict classified a failure (an `Unknown`
/// verdict has no cause, so nothing is appended). No report tail — the gate serves the
/// chrome tool and the page rasteriser alike.
#[must_use]
pub(crate) fn refusal_message(refusal: &str) -> String {
    let classified = health().lock().unwrap_poison().failure().is_some();
    if classified {
        format!("{refusal} {}", daemon_down_cause())
    } else {
        refusal.to_string()
    }
}

/// Bounded post-timeout health evaluation. `status` cannot see a wedged session daemon,
/// and mahbot's per-call bound cuts the CLI off before its ~152 s retry loop can surface
/// the daemon-unavailable signature — so after a healthy `status` the session's daemon is
/// probed with a trivial bounded command. A second consecutive hang is the wedge
/// signature: the session is recovered through the shared helper and the text says what
/// was done. `None` = healthy, a slow call rather than the daemon.
///
/// An UNESTABLISHED snapshot does not short-circuit this: keys a future chrome-use stops
/// answering must not disable the tool's own wedge recovery.
pub(crate) async fn health_after_call_timeout(session: &str) -> Option<String> {
    let readiness = readiness().await;
    if !readiness.answered || readiness.blocked() {
        return Some(daemon_down_message());
    }
    // Silence is the wedge signature: our bound cuts the CLI off before its ~152 s retry
    // loop can surface the daemon-unavailable text, so no answer means the daemon stopped
    // answering. A host that cannot spawn chrome-use has no browser to drive either.
    let probe = match cli_path() {
        Some(path) => run_cli_bounded_at(&path, &["get", "url"], Some(session), CLI_TIMEOUT).await,
        None => CliRun::SpawnFailure,
    };
    if !matches!(probe, CliRun::Output(_)) {
        let recovery = recover_unresponsive_session(session).await;
        return Some(format!("{} {}", daemon_down_message(), recovery.summary()));
    }
    None
}

/// Background watchdog: evaluate health from the daemon-free status, auto-restart with
/// bounded backoff, and halt after repeated crashes to avoid a restart loop. Stands down
/// on hosts without the CLI only after [`CLI_MISSING_THRESHOLD`] definitive-missing
/// probes, so a transient spawn failure never takes it out of service. Cadence: healthy
/// hosts re-verify CLI presence every [`CLI_RECHECK`], unknown hosts re-probe every
/// [`WATCHDOG_INTERVAL`], stood-down hosts at [`CLI_RECHECK`] and every interval while a
/// transient persists. A transient implies the binary resolved, so it resets the streak and
/// re-enables recovery; a broken install whose `--version` exits non-zero classifies as
/// transient and never stands down.
pub async fn run_watchdog() {
    let mut cli_present: Option<bool> = None;
    let mut last_cli_check = Instant::now();
    let mut cli_missing: u32 = 0;
    // Last transient probe cause — warn only on change, so a persistent transient
    // leaves a trail without spamming the log.
    let mut last_transient: Option<CliProbeFailure> = None;
    let mut cleaned = false;
    // Whether the last wait ended in an early wake from the fail-fast path: recovery
    // then consumes the stored classification instead of re-evaluating, which the
    // daemon-free status cannot see and would clobber.
    let mut woken = false;
    loop {
        // A CLI-less host cannot run commands, and an evaluation without a CLI would
        // only record Unknown for a binary that is not there.
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
                    // Not definitive absence, so the watchdog stays in service.
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
                        // First miss — confirm on the next interval, re-probing
                        // since the cached verdict is no longer trustworthy.
                        cli_present = None;
                        skip_health = true;
                    } else {
                        if cli_present != Some(false) {
                            cli_present = Some(false);
                            warn!(
                                "chrome-use CLI not found — chrome daemon watchdog standing down"
                            );
                        }
                        // Rare re-checks on CLI-less hosts; an early wake re-checks.
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
            // A fail-fast classification from a real call is the freshest signal, so
            // recover from it directly; on interval ticks run the daemon-free
            // evaluation and recover from a service-level failure only — an `Unknown`
            // verdict established nothing, and a restart may be spent on nothing.
            let failure = if woken {
                health().lock().unwrap_poison().failure()
            } else {
                None
            };
            if let Some(failure) = failure {
                attempt_recovery(failure).await;
            } else {
                let outcome = probe_and_record(true).await.outcome();
                if let Some(failure) = outcome.failure() {
                    attempt_recovery(failure).await;
                }
            }
            // The command-line sessions' lifetime tick: one session nobody used for its
            // lifetime is closed per pass. It rides the health pass, so a stood-down host
            // runs no tick — harmless, since the CLI's own pre-parse sweep covers it.
            crate::chrome::cli_sessions::sweep_expired().await;
        }
        // Wait for the next interval or an early wake from the fail-fast path. `woken`
        // resets on every timeout, so a wake not consumed before a CLI stand-down is
        // dropped instead of replayed after reinstall.
        let shutdown = crate::shutdown::shutdown_token();
        woken = tokio::select! {
            () = tokio::time::sleep(sleep) => false,
            () = wake().notified() => true,
            () = shutdown.cancelled() => break,
        };
    }
}

/// Last (expected, live) extension-version pair advised on — logged only on a
/// transition, mirroring the sweep's anti-spam pattern.
static LAST_EXTENSION_SKEW: OnceLock<Mutex<Option<(String, String)>>> = OnceLock::new();

/// Extension-skew advisory from an already-fetched `status --json`: the Store
/// extension is NEVER force-updated (the Store does that in the background), so a
/// `liveVersion` behind the CLI's `expectedVersion` only logs once per transition.
/// Older CLIs without these fields skip silently.
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

/// Download the chrome-use release archive for `tag`, verify it against the published
/// `.sha256` sidecar (missing, unreadable or mismatching is a HARD failure), and
/// extract the single binary into a temp dir. The caller must keep the guard alive
/// until the binary is in place.
async fn download_chrome_use_binary(tag: &str) -> Result<(tempfile::TempDir, PathBuf), String> {
    use crate::util::http::{DownloadSizeCheck, build_download_client, download_verified};

    let platform = release_asset_name()?;
    let asset = format!("chrome-use-{platform}.tar.gz");
    let base = format!("https://github.com/{CHROME_USE_RELEASE_REPO}/releases/download/{tag}");
    let tgz_url = format!("{base}/{asset}");
    let sha_url = format!("{tgz_url}.sha256");

    let client = build_download_client(CHROME_USE_DOWNLOAD_TIMEOUT)
        .map_err(|_| "the download client could not be built".to_string())?;

    // A missing, unreadable or mismatching sidecar is a hard failure, so a tampered
    // or partial release is never installed.
    let sidecar = client.get(&sha_url).send().await.map_err(|e| {
        format!(
            "the sha256 sidecar could not be fetched: {}",
            crate::util::managed_bin::request_reason(&e)
        )
    })?;
    if !sidecar.status().is_success() {
        return Err(format!(
            "the sha256 sidecar could not be fetched: HTTP {}",
            sidecar.status()
        ));
    }
    let body = sidecar.text().await.map_err(|e| {
        format!(
            "the sha256 sidecar could not be read: {}",
            crate::util::managed_bin::request_reason(&e)
        )
    })?;
    let (hash, sidecar_name) =
        crate::util::managed_bin::parse_sha256_sidecar(&body).ok_or_else(|| {
            "the sha256 sidecar is malformed (no `64-hex-hash  filename` pair)".to_string()
        })?;
    // The sidecar names the archive it was published for — a valid hash from a
    // cross-paired sidecar must not verify a different asset.
    if sidecar_name != asset {
        return Err(format!(
            "the sha256 sidecar names '{sidecar_name}', expected '{asset}'"
        ));
    }

    let dir = tempfile::tempdir().map_err(|e| {
        format!(
            "the download's temporary directory could not be created ({})",
            e.kind()
        )
    })?;
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
    .map_err(|e| {
        format!(
            "the release archive could not be downloaded: {}",
            crate::util::managed_bin::download_reason(&e)
        )
    })?;

    let out_path = crate::util::managed_bin::extract_single_file_tar_gz(
        &archive_path,
        dir.path(),
        chrome_bin(),
    )?;
    Ok((dir, out_path))
}

/// Install the newest chrome-use release over whatever sits at the destination —
/// reached only when that copy differs from the newest release, or is missing or
/// unusable. `Err` names the failing step with no local path and no value read from the
/// owner's environment, and never removes or invalidates the copy already there, so the
/// machine is never left without a working helper. Native-host registration is NOT part
/// of this: it binds a path an install never changes ([`run_chrome_use_management`]).
async fn install_chrome_use(
    release: Option<crate::util::managed_bin::Release>,
) -> Result<PathBuf, String> {
    // The location the helper goes to, named before anything is fetched, so an
    // unresolvable home costs no download.
    let dest = crate::util::managed_bin::chrome_use_bin_path(chrome_bin()).ok_or_else(|| {
        "the helper's install directory is unavailable (the owner's home could not be resolved)"
            .to_string()
    })?;
    // The tag the caller's comparison already resolved; an absent copy had none to
    // compare, so this start resolves the newest release itself.
    let tag = match release {
        Some(release) => release.tag,
        None => {
            crate::util::managed_bin::fetch_latest_tag(
                CHROME_USE_RELEASE_REPO,
                CHROME_USE_RELEASE_TIMEOUT,
            )
            .await?
        }
    };
    let (_temp, fresh) = download_chrome_use_binary(&tag).await?;
    let placed = crate::util::managed_bin::place_extracted(&fresh, &dest);
    // The copy belongs at the location the resolver looks in; clear any cached path
    // so the next probe re-resolves it.
    invalidate_cli_path();
    placed.map(|()| dest)
}

/// Register the installed binary as the native-messaging host. The registration binds
/// the helper's full path, the same for every install, so it runs on every start: one
/// that did not land is repaired even when versions match, since a launcher pointing at
/// a copy that is no longer installed is the classic silent relay failure. `--no-profile`
/// is REQUIRED on macOS: without it chrome-use writes and queues the
/// `ab-connect.mobileconfig` managed profile (ExtensionInstallForcelist), flipping Chrome
/// into "managed by your organization" mode — mahbot never creates or re-queues it. The
/// flag needs chrome-use v1.5.93+, and an older copy can only have had its update not
/// land, so the registration fails with its own reason and is retried next start. A
/// failure never removes the binary: the browser must always launch a copy that works.
async fn register_native_host(dest: &Path) -> Result<(), String> {
    let mut host = Command::new(dest);
    #[cfg(target_os = "windows")]
    host.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    host.args(["extension", "install", "--no-profile"]);
    run_install_step("`chrome-use extension install --no-profile`", host)
        .await
        .map_err(|e| format!("{e}\nThe installed copy is left in place."))
}

/// Run one install subprocess bounded by [`CHROME_USE_INSTALL_TIMEOUT`]. The helper's
/// output is dropped (it can name the owner's browser profile), so only `label` and the
/// exit status or spawn-error kind reach the message.
async fn run_install_step(label: &str, mut cmd: Command) -> Result<(), String> {
    let out = tokio::time::timeout(CHROME_USE_INSTALL_TIMEOUT, cmd.kill_on_drop(true).output())
        .await
        .map_err(|_| format!("{label} timed out"))?
        .map_err(|e| format!("{label} failed to spawn: {}", e.kind()))?;
    if out.status.success() {
        return Ok(());
    }
    Err(format!("{label} failed ({})", out.status))
}

/// Spawned one-shot task: install the product's own copy of chrome-use straight away
/// when it is absent or not executable, else check its version against the newest
/// release once boot has settled and update only when the two differ. Which copy the
/// owner's own terminal resolves first is his search path's order, which the product
/// neither knows nor changes. Native-host registration follows on every start, since
/// its bound path does not depend on the version and a registration that did not land
/// must be repaired even when nothing was downloaded. Failures are non-fatal, recorded
/// at WARN (the level the issues view shows) and retried next start.
pub async fn run_chrome_use_management() {
    crate::util::managed_bin::install_on_start(
        "chrome-use",
        || cli_path().is_some(),
        async || cli_version().await,
        async || {
            crate::util::managed_bin::latest_release(
                CHROME_USE_RELEASE_REPO,
                CHROME_USE_RELEASE_TIMEOUT,
            )
            .await
        },
        async |release| install_chrome_use(release).await,
        crate::util::managed_bin::WhenMissing::Install,
    )
    .await;
    // Only a working copy can be registered; with none, the install above already
    // reported why.
    let Some(dest) = cli_path() else {
        return;
    };
    if let Err(reason) = register_native_host(&dest).await {
        let reason = crate::util::truncate(&reason, 1024);
        tracing::warn!(
            "chrome-use's native-host registration could not be refreshed (retried on the next \
             start): {reason}"
        );
    }
}

/// One-time cleanup of stale mahbot-owned chrome-session artifacts at watchdog start,
/// so leftover link-enricher tab groups do not accumulate. Each sweep is verified and
/// only ever closes the target session's own tabs: user and other-agent sessions and
/// run-owned `agent-tab-*` ones are never touched. A NAMED command-line session never
/// reaches here, so the in-flight record guard protects a call that is running right
/// now — it is read per candidate, so a call that records itself mid-sweep is covered
/// too. Dead link-enricher orphans stay until the tab is closed by hand.
async fn cleanup_stale_sessions() {
    let Some(sessions) = registered_sessions().await else {
        return;
    };
    for name in sessions {
        if is_mahbot_session_name(&name)
            && !crate::chrome::cli_sessions::recorded_names().contains(&name)
        {
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

/// Poll `status --json` until the relay republishes or the budget elapses (the MV3
/// service worker revives on its ~30 s keepalive).
async fn wait_for_relay(budget: Duration) {
    let deadline = Instant::now() + budget;
    while Instant::now() < deadline {
        if relay_up().await == Some(true) {
            return;
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

pub(crate) async fn relay_up() -> Option<bool> {
    let status = run_cli_json(&["status"]).await?;
    status
        .get("data")?
        .get("extension")?
        .get("relayUp")
        .and_then(Value::as_bool)
}

/// Chrome launch flags for auto-launch: deliberately minimal, no daemon/profile/home
/// overrides, since the launch inherits the user's real profile and environment.
const CHROME_LAUNCH_FLAGS: [&str; 3] = [
    "--no-first-run",
    "--no-default-browser-check",
    "--silent-debugger-extension-api",
];

/// Launch the user's real Chrome when [`ProbeFailure::ChromeNotRunning`], under its own
/// bounded launch budget. Display-less hosts are paused with no budget consumed. The
/// launch inherits mahbot's real environment (NOT `ensure_chrome_env` — its HOME
/// override and daemon flags must never reach the user's browser) and waits for the relay.
async fn attempt_chrome_launch() {
    if !display_available() {
        record_launch_outcome(ChromeLaunchOutcome::NoDisplay);
        debug!("chrome daemon: headless host — Chrome launch skipped; start Chrome manually");
        return;
    }
    let gate = { health().lock().unwrap_poison().gate_launch(Instant::now()) };
    let RecoveryGate::Allowed(attempt) = gate else {
        log_gate_denied(gate, "chrome launch", MAX_LAUNCH_ATTEMPTS);
        return;
    };
    let Some(binary) = chrome_binary() else {
        record_launch_outcome(ChromeLaunchOutcome::Failed);
        warn!("no Chrome/Chromium binary found — start Chrome manually");
        return;
    };
    if let Err(e) = spawn_chrome_detached(&binary) {
        record_launch_outcome(ChromeLaunchOutcome::Failed);
        warn!(error = %e, "failed to launch Chrome — start Chrome manually");
        return;
    }
    info!(
        attempt,
        max = MAX_LAUNCH_ATTEMPTS,
        "launched the user's Chrome; waiting for the extension relay to come up"
    );
    wait_for_relay(RELAY_REVIVE_WAIT).await;
    // The post-launch verification must not seed the sustained-healthy window (same
    // rationale as the restart path): the launch budget resets only on sustained health.
    let outcome = probe_and_record(false).await.outcome();
    if outcome.is_healthy() {
        info!("chrome daemon: relay recovered after Chrome launch");
    } else {
        record_launch_outcome(ChromeLaunchOutcome::Launched);
        warn!(
            "Chrome launched but the extension relay is still down — if the ab-connect \
             extension is only installed in a non-default profile, install it in Chrome's \
             default profile"
        );
    }
}

/// Whether a Chrome window can actually appear: Linux needs a display session;
/// macOS/Windows attempt the launch and degrade through the failure path.
pub(crate) fn display_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::env::var_os("DISPLAY").is_some() || std::env::var_os("WAYLAND_DISPLAY").is_some()
    }
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
}

/// Resolve the user's real Chrome/Chromium-family binary (standard install locations,
/// in the spirit of chrome-use's own resolution); `None` fails the launch honestly.
fn chrome_binary() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        let home = directories::UserDirs::new().map(|d| d.home_dir().to_path_buf());
        let mut candidates = Vec::new();
        for app in ["Google Chrome", "Chromium", "Brave Browser"] {
            candidates.push(PathBuf::from(format!(
                "/Applications/{app}.app/Contents/MacOS/{app}"
            )));
            if let Some(home) = home.as_deref() {
                candidates.push(home.join(format!("Applications/{app}.app/Contents/MacOS/{app}")));
            }
        }
        candidates
            .into_iter()
            .find(|p| crate::util::is_executable(p))
    }
    #[cfg(target_os = "linux")]
    {
        let names = [
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
            "brave-browser",
            "brave",
        ];
        let mut dirs: Vec<PathBuf> = Vec::new();
        if let Some(paths) = std::env::var_os("PATH") {
            dirs.extend(std::env::split_paths(&paths));
        }
        dirs.extend([
            PathBuf::from("/usr/bin"),
            PathBuf::from("/usr/local/bin"),
            PathBuf::from("/snap/bin"),
        ]);
        for dir in dirs {
            for name in names {
                let candidate = dir.join(name);
                if crate::util::is_executable(&candidate) {
                    return Some(candidate);
                }
            }
        }
        None
    }
    #[cfg(target_os = "windows")]
    {
        let mut candidates = Vec::new();
        for base in ["ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"] {
            let Some(base) = std::env::var_os(base) else {
                continue;
            };
            let base = PathBuf::from(base);
            for rel in [
                Path::new("Google/Chrome/Application/chrome.exe"),
                Path::new("Chromium/Application/chrome.exe"),
                Path::new("BraveSoftware/Brave-Browser/Application/brave.exe"),
            ] {
                candidates.push(base.join(rel));
            }
        }
        candidates
            .into_iter()
            .find(|p| crate::util::is_executable(p))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

/// Spawn Chrome independent of mahbot: the child leads its own process group on Unix,
/// so terminal signals to mahbot's group must not kill the browser, and on Windows it
/// gets the same windowless-console flag as every other spawn (inert for this GUI image
/// — see the window guarantee in `tools::shell::tree`). Dropped without wait or
/// kill_on_drop, so Chrome survives mahbot restarts.
fn spawn_chrome_detached(binary: &Path) -> std::io::Result<()> {
    let mut cmd = std::process::Command::new(binary);
    cmd.args(CHROME_LAUNCH_FLAGS)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // Inherit the real env — do NOT touch HOME or add AGENT_BROWSER_*/CHROMIUM_FLAGS.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    }
    cmd.spawn().map(|_| ())
}

/// Report the cause once per transition — an ongoing failure does not spam every
/// interval, but the same cause is reported again after a healthy spell. The wedge
/// record is informational (its restart is automatic); every other cause warns.
fn report_cause(failure: ProbeFailure) {
    let mut h = health().lock().unwrap_poison();
    if h.last_cause_reported == Some(failure) {
        return;
    }
    h.last_cause_reported = Some(failure);
    match failure {
        ProbeFailure::NotInstalled => warn!(
            "chrome-use extension or native host is not installed — the browser \
             daemon cannot run. Enable the chrome-use extension at \
             chrome://extensions (the CLI is installed at the start of every run \
             of the product). Auto-recovery paused until it is installed."
        ),
        ProbeFailure::HostBroken => warn!(
            "chrome-use native host launcher is broken — run `chrome-use doctor` (the copy the \
             product installed; spell its full path when the bare name is not on this shell's \
             own search path). Auto-recovery paused until it is fixed."
        ),
        ProbeFailure::ExtensionDisabled => warn!(
            "chrome-use extension is disabled — enable it at chrome://extensions. \
             Daemon restarts cannot fix a Chrome-side disable; auto-recovery paused \
             until it is enabled."
        ),
        ProbeFailure::ExtensionAbsent => warn!(
            "chrome-use extension is not installed in Chrome — install/enable it from \
             the Chrome Web Store. Auto-recovery paused until it is installed."
        ),
        ProbeFailure::RelayDown => warn!(
            "chrome-use extension relay is down (the extension is enabled) — waiting \
             for the extension to reconnect and restarting session daemons to clear \
             stale relay bindings."
        ),
        ProbeFailure::ChromeNotRunning => {
            // Must match what attempt_recovery will actually do on this host — a
            // headless machine pauses recovery instead of launching.
            if !display_available() {
                warn!(
                    "Chrome is not running and no display is available (headless host) — \
                       start Chrome manually; auto-recovery paused."
                );
            } else if h.launch_budget.halted {
                warn!(
                    "Chrome is not running — launch attempts are paused in a thrash-protection \
                     cooldown; start Chrome manually."
                );
            } else {
                warn!(
                    "Chrome is not running — auto-recovery will launch it (bounded launch budget)."
                );
            }
        }
        ProbeFailure::UnreachableTab => warn!(
            "a browser tab the session was driving is unreachable (the extension lost its \
             debugger attach; about:blank tabs are never re-attached). The tab belongs to the \
             session and the product closes the session's tabs itself when the run ends (or \
             the session is stopped)"
        ),
        // Informational: the restart it announces is automatic and deduplicated per
        // transition — a failed restart and an exhausted budget still surface.
        ProbeFailure::DaemonWedge => {
            info!("chrome daemon is unresponsive — restarting it (bounded backoff).");
        }
    }
}

/// Log a blocked recovery-gate decision — the gated action does not run. `noun` names
/// the gated action ("restart" / "chrome launch").
fn log_gate_denied(gate: RecoveryGate, noun: &str, max: u32) {
    match gate {
        RecoveryGate::Halted => error!(
            attempts = max,
            "chrome daemon: {max} consecutive failed {noun} attempts; \
             auto-recovery halted for 30 min (thrash protection)"
        ),
        RecoveryGate::Backoff => {
            debug!("chrome daemon: still down; waiting out {noun} backoff");
        }
        RecoveryGate::Cooldown => {
            debug!("chrome daemon: still down; {noun} cooldown in progress (thrash protection)");
        }
        RecoveryGate::Allowed(_) => unreachable!(),
    }
}

/// Bounded auto-recovery for a cause a fact ESTABLISHED. `ChromeNotRunning` funnels
/// into a bounded Chrome launch — never a daemon restart, never the restart budget.
/// Restart causes restart session daemons with backoff and a halt after
/// [`MAX_RESTART_ATTEMPTS`]. Causes a restart cannot fix are reported with their
/// concrete fix and never consume attempts. A relay drop is waited out first and costs
/// no attempt if it self-heals; a wait that ends with nothing established stops there —
/// no restart is justified by "could not find out".
async fn attempt_recovery(mut failure: ProbeFailure) {
    report_cause(failure);
    // Unfixable causes stop here — they never consume restart attempts.
    if failure.is_unfixable() {
        return;
    }
    // ChromeNotRunning is handled by a bounded launch BEFORE the restart throttle
    // check: its budget is independent, so a halted restart must not block it.
    if failure == ProbeFailure::ChromeNotRunning {
        attempt_chrome_launch().await;
        return;
    }
    // While a recovery timer is pending, that timer IS the wait — don't poll the relay
    // on top of it; the next watchdog cycle re-enters recovery.
    let now = Instant::now();
    let throttled = health()
        .lock()
        .unwrap_poison()
        .restart_budget
        .is_waiting(now);
    if throttled {
        return;
    }
    // A relay drop is waited out before any session-disrupting restart, and a drop
    // that self-heals consumes no restart attempt.
    if failure == ProbeFailure::RelayDown {
        wait_for_relay(RELAY_REVIVE_WAIT).await;
        let outcome = probe_and_record(true).await.outcome();
        match outcome {
            ProbeOutcome::Healthy => {
                info!("chrome daemon: relay recovered without a restart");
                return;
            }
            ProbeOutcome::Down(f) => {
                // Re-classified (e.g. now a wedge or a closed browser) — re-report
                // the cause and re-gate below.
                report_cause(f);
                if f.is_unfixable() {
                    return;
                }
                if f == ProbeFailure::ChromeNotRunning {
                    attempt_chrome_launch().await;
                    return;
                }
                failure = f;
            }
            ProbeOutcome::Unknown => {
                // The wait ended with nothing established — no fact showed why the
                // relay never came back. A restart destroys session state, so it is
                // not justified by "could not find out": report it and leave the
                // next pass to re-probe.
                warn!(
                    "chrome daemon: nothing could be established after waiting for the relay — no \
                     restart attempted"
                );
                return;
            }
        }
    }

    // Decide the restart within a scoped lock, so the guard is never held across an
    // await point.
    let gate = {
        let mut h = health().lock().unwrap_poison();
        h.gate_restart(Instant::now())
    };
    let RecoveryGate::Allowed(attempt) = gate else {
        log_gate_denied(gate, "restart", MAX_RESTART_ATTEMPTS);
        return;
    };

    info!(
        attempt,
        max = MAX_RESTART_ATTEMPTS,
        "chrome daemon: attempting auto-recovery"
    );
    // Restart session daemons (session-less; closes their tabs, relay survives). No
    // `reconnect` — it can cold-restart the user's Chrome or open the Web Store.
    let _ = run_cli(&["daemon", "restart"]).await;
    if failure == ProbeFailure::RelayDown {
        wait_for_relay(RELAY_REVIVE_WAIT).await;
    }

    // Post-restart verification must not seed the sustained-healthy window: the budget
    // resets only after a sustained health spell, so a failing run cannot reopen it.
    let outcome = probe_and_record(false).await.outcome();
    if outcome.is_healthy() {
        info!("chrome daemon: recovered after restart");
    } else {
        warn!(
            attempt,
            "chrome daemon: restart attempt did not restore health"
        );
    }
}

async fn run_cli(args: &[&str]) -> bool {
    let Some(path) = cli_path() else {
        return false;
    };
    let mut cmd = Command::new(path);
    #[cfg(target_os = "windows")]
    cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    ensure_chrome_env(&mut cmd);
    cmd.args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    cmd.kill_on_drop(true);
    tokio::time::timeout(Duration::from_mins(1), cmd.status())
        .await
        .is_ok_and(|r| r.is_ok_and(|st| st.success()))
}

/// Test-only lock serializing tests that mutate the global daemon health state (cargo
/// runs tests in parallel threads).
#[cfg(test)]
pub(crate) async fn with_health_test_lock() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

/// Test-only: restore the global health singleton so mutating tests do not leak state
/// into later readers (e.g. `Agent::new` filtering tools via `is_advertised`).
#[cfg(test)]
pub(crate) fn reset_health() {
    *health().lock().unwrap_poison() = DaemonHealth::default();
}

/// Test-only: put a readiness snapshot behind the stored verdict (or clear it), so
/// [`daemon_down_message`]'s branches run without the CLI spawn a real probe needs.
#[cfg(test)]
pub(crate) fn set_readiness_snapshot(readiness: Option<Readiness>) {
    health().lock().unwrap_poison().readiness = readiness.map(|r| (Instant::now(), r));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chrome::contract::is_daemon_unavailable_code;

    /// The readiness/session probes read the answer tolerantly: bytes a leftover
    /// process wrote after the envelope must not turn an answered call into "no answer".
    #[test]
    fn envelope_outcome_ignores_bytes_written_after_the_answer() {
        let answered = br#"{"ok":true,"data":{"sessions":[]}}"#;
        let mut with_leftover = answered.to_vec();
        with_leftover.extend_from_slice(b"\nthe helper also wrote this\n");
        assert_eq!(
            envelope_outcome(true, &with_leftover).expect("trailing bytes ignored"),
            serde_json::json!({"ok": true, "data": {"sessions": []}})
        );

        // A structured error is still an answer: its own message survives.
        assert_eq!(
            envelope_outcome(
                false,
                br#"{"success":false,"error":"session unresponsive"}"#
            )
            .unwrap_err(),
            Some("session unresponsive".to_string())
        );
        // Only a call with no JSON value at all is "no answer".
        assert_eq!(
            envelope_outcome(true, b"chrome-use is not installed\n"),
            Err(None)
        );
    }

    #[tokio::test]
    async fn advertisement_reflects_daemon_state() {
        let _guard = with_health_test_lock().await;
        // Dead-daemon fixture: confirmed-down → not advertised.
        set_health(ProbeOutcome::Down(ProbeFailure::DaemonWedge));
        assert!(!is_advertised());
        // Recovered: fresh healthy state → advertised.
        set_health(ProbeOutcome::Healthy);
        assert!(is_advertised());
        // Unknown is neither health nor a cause: still advertised, never acted on.
        set_health(ProbeOutcome::Unknown);
        assert!(is_advertised());
        assert!(!ProbeOutcome::Unknown.is_healthy());
        assert_eq!(ProbeOutcome::Unknown.failure(), None);
        // Fresh boot (nothing probed yet) → advertised optimistically.
        reset_health();
        assert!(is_advertised());
    }

    /// A reporting gap is neither health nor a named cause: it promises no repair and
    /// spends no restart, while an action is still dispatched on it.
    #[tokio::test]
    async fn a_reporting_gap_is_neither_health_nor_a_cause() {
        {
            let case = "unknown is not health but is not a classified cause";
            // The tool did not answer: nothing established, so no health, no named
            // cause, no restart.
            let _guard = with_health_test_lock().await;
            reset_health();
            set_readiness_snapshot(None);
            set_health(ProbeOutcome::Unknown);
            let msg = daemon_down_message();
            assert!(msg.contains("did not answer"), "case {case}: got: {msg}");
            assert!(
                msg.contains("nothing could be established"),
                "case {case}: got: {msg}"
            );
            // The launch/restart tails belong to classified causes.
            assert!(!msg.contains("relay is down"), "case {case}: got: {msg}");
            assert!(!msg.contains("not installed"), "case {case}: got: {msg}");
            // An Unknown verdict appends no cause and no report tail to the gate's facts.
            let refusal = refusal_message("the facts");
            assert_eq!(refusal, "the facts", "case {case}");

            // The tool DID answer and a fact inside its answer was missing: the
            // snapshot's own refusal is accurate and must be what the caller reads —
            // never the false claim that the CLI stayed silent.
            let answered = Readiness {
                answered: true,
                relay_up: None,
                ..Readiness::default()
            };
            set_readiness_snapshot(Some(answered));
            let msg = daemon_down_message();
            assert!(!msg.contains("did not answer"), "case {case}: got: {msg}");
            assert!(
                msg.contains("whether the extension relay is up"),
                "case {case}: got: {msg}"
            );
            assert!(
                msg.contains("re-checking continues"),
                "case {case}: got: {msg}"
            );
            // …and a classified verdict's cause IS appended to the gate's facts, still
            // without the report tail.
            set_health(ProbeOutcome::Down(ProbeFailure::RelayDown));
            let msg = refusal_message("the facts");
            assert!(msg.starts_with("the facts "), "case {case}: got: {msg}");
            assert!(msg.contains("relay is down"), "case {case}: got: {msg}");
            assert!(
                !msg.contains("Report it as it came"),
                "case {case}: got: {msg}"
            );
            set_readiness_snapshot(None);
            reset_health();
        }
        {
            let case = "an unestablished snapshot dispatches and spends no restart";
            // The tool answered but a decisive key was missing: not proven, not blocked,
            // so the action dispatches and no restart is spent. The cached snapshot is
            // served as-is, so no CLI is spawned.
            let _guard = with_health_test_lock().await;
            reset_health();
            set_readiness_snapshot(Some(Readiness {
                answered: true,
                chrome_running: Some(true),
                display: Some(true),
                ..Readiness::default()
            }));
            assert!(ensure_ready_for_actions().await.is_ok(), "case {case}");
            let h = health().lock().unwrap_poison();
            assert_eq!(h.verdict, None, "case {case}");
            assert_eq!(h.restart_budget.attempts, 0, "case {case}");
            drop(h);
            set_readiness_snapshot(None);
            reset_health();
        }
    }

    /// The down message names the launch state it actually has: plain relay drop, the
    /// non-default-profile caveat, the launch promise, the failed-launch backoff and the
    /// paused budget.
    #[tokio::test]
    async fn daemon_down_message_reflects_launch_state() {
        let _guard = with_health_test_lock().await;
        // Each sub-case starts pristine so a recorded launch outcome never leaks.
        reset_health();

        // Plain relay drop — no launch was attempted, so no launch caveat, and every
        // integration cause ends with what the agent does about it.
        set_health(ProbeOutcome::Down(ProbeFailure::RelayDown));
        let msg = daemon_down_message();
        assert!(
            msg.contains("relay is down"),
            "the plain relay drop names the relay: {msg}"
        );
        assert!(
            !msg.contains("auto-launched"),
            "no launch was attempted, so no launch caveat: {msg}"
        );
        assert!(
            msg.contains("Report it as it came"),
            "the agent reports it as it came: {msg}"
        );
        assert!(
            msg.contains("leave the repair to the owner"),
            "the repair is the owner's: {msg}"
        );

        // Relay still down after a successful auto-launch: the durable non-default-profile
        // caveat surfaces instead of the plain text.
        reset_health();
        set_health(ProbeOutcome::Down(ProbeFailure::RelayDown));
        record_launch_outcome(ChromeLaunchOutcome::Launched);
        let msg = daemon_down_message();
        assert!(
            msg.contains("not republishing its relay"),
            "a launched browser names the relay caveat: {msg}"
        );
        assert!(
            msg.contains("non-default profile"),
            "the caveat names the non-default profile: {msg}"
        );

        // Chrome not running with no recorded launch — the tail promises a launch.
        reset_health();
        set_health(ProbeOutcome::Down(ProbeFailure::ChromeNotRunning));
        let msg = daemon_down_message();
        assert!(
            msg.contains("Chrome is not running."),
            "the cause is stated: {msg}"
        );
        assert!(
            msg.contains("launches Chrome and keeps retrying"),
            "recovery launches Chrome: {msg}"
        );

        // Failed launch with the budget un-halted: the failed-cause text and the backoff
        // tail appear, and no repair step is handed to the agent.
        reset_health();
        set_health(ProbeOutcome::Down(ProbeFailure::ChromeNotRunning));
        record_launch_outcome(ChromeLaunchOutcome::Failed);
        let msg = daemon_down_message();
        assert!(
            msg.contains("auto-launch failed on this host"),
            "a failed launch names the cause: {msg}"
        );
        assert!(
            msg.contains("retries the launch with backoff"),
            "the retry is a backoff: {msg}"
        );
        assert!(
            !msg.contains("start Chrome manually"),
            "no manual step is handed to the agent: {msg}"
        );

        // Same failure with the budget halted (thrash protection): the cooldown text
        // supersedes the backoff promise.
        reset_health();
        set_health(ProbeOutcome::Down(ProbeFailure::ChromeNotRunning));
        record_launch_outcome(ChromeLaunchOutcome::Failed);
        health().lock().unwrap_poison().launch_budget.halted = true;
        let msg = daemon_down_message();
        assert!(
            msg.contains("Launch attempts are paused"),
            "a halted budget is the cooldown: {msg}"
        );
        assert!(
            !msg.contains("retries the launch with backoff"),
            "the cooldown supersedes the backoff promise: {msg}"
        );

        // Leave the singleton pristine for sibling health tests.
        reset_health();
    }

    /// Sustained health resets the restart attempts; a flapping cause or a
    /// transient health reading does not.
    #[tokio::test]
    async fn only_sustained_health_resets_the_restart_budget() {
        {
            let case = "sustained health resets restart attempts";
            let now = Instant::now();
            let mut h = DaemonHealth {
                restart_budget: AttemptBudget {
                    attempts: 2,
                    next_at: Some(now),
                    halted: true,
                    halted_until: Some(now),
                },
                ..DaemonHealth::default()
            };
            // A transient healthy result (the post-recovery verification) must not reset
            // the budget.
            h.apply_outcome(ProbeOutcome::Healthy, now, false);
            assert_eq!(h.restart_budget.attempts, 2, "case {case}");
            assert!(h.restart_budget.halted, "case {case}");
            // The first watchdog healthy seeds the window…
            h.apply_outcome(ProbeOutcome::Healthy, now + WATCHDOG_INTERVAL, true);
            assert_eq!(h.restart_budget.attempts, 2, "case {case}");
            assert!(h.healthy_since.is_some(), "case {case}");
            // …a second healthy before it elapses still does not reset.
            h.apply_outcome(ProbeOutcome::Healthy, now + WATCHDOG_INTERVAL * 2, true);
            assert_eq!(h.restart_budget.attempts, 2, "case {case}");
            assert!(h.restart_budget.halted, "case {case}");
            // Only sustained health across the window opens a fresh bounded cycle.
            h.apply_outcome(ProbeOutcome::Healthy, now + WATCHDOG_INTERVAL * 3, true);
            assert_eq!(h.restart_budget.attempts, 0, "case {case}");
            assert_eq!(h.restart_budget.next_at, None, "case {case}");
            assert!(!h.restart_budget.halted, "case {case}");
            assert!(h.restart_budget.halted_until.is_none(), "case {case}");
            assert_eq!(h.failure(), None, "case {case}");
        }
        {
            let case = "cause flapping and transient health do not reset restart budget";
            let now = Instant::now();
            let mut h = DaemonHealth {
                restart_budget: AttemptBudget {
                    attempts: 2,
                    next_at: Some(now),
                    ..AttemptBudget::default()
                },
                verdict: Some(ProbeOutcome::Down(ProbeFailure::DaemonWedge)),
                ..DaemonHealth::default()
            };
            // A cause flip (wedge → relay-down) must NOT reset the budget, or alternating
            // causes would evade the halt.
            h.apply_outcome(ProbeOutcome::Down(ProbeFailure::RelayDown), now, true);
            assert_eq!(h.failure(), Some(ProbeFailure::RelayDown), "case {case}");
            assert_eq!(h.restart_budget.attempts, 2, "case {case}");
            assert!(h.restart_budget.next_at.is_some(), "case {case}");
            // Flapping back and forth accumulates — never resets.
            h.apply_outcome(ProbeOutcome::Down(ProbeFailure::DaemonWedge), now, true);
            h.apply_outcome(ProbeOutcome::Down(ProbeFailure::RelayDown), now, true);
            assert_eq!(h.failure(), Some(ProbeFailure::RelayDown), "case {case}");
            assert_eq!(h.restart_budget.attempts, 2, "case {case}");
            // Only sustained health across the window opens a fresh cycle.
            h.apply_outcome(ProbeOutcome::Healthy, now, true);
            assert_eq!(h.restart_budget.attempts, 2, "case {case}");
            assert!(h.restart_budget.next_at.is_some(), "case {case}");
            h.apply_outcome(ProbeOutcome::Healthy, now + SUSTAINED_HEALTHY_WINDOW, true);
            assert_eq!(h.failure(), None, "case {case}");
            assert_eq!(h.restart_budget.attempts, 0, "case {case}");
            assert_eq!(h.restart_budget.next_at, None, "case {case}");
            assert!(!h.restart_budget.halted, "case {case}");
        }
    }

    /// The two bounded cycles are independent: a halted restart budget does not block a
    /// launch, an exhausted launch budget leaves the restart gate open, and sustained
    /// health opens a fresh cycle for both while clearing a stale launch record.
    #[test]
    fn the_gates_bound_restarts_and_launches_independently() {
        {
            let case = "gate honors backoff before halt";
            let now = Instant::now();
            let mut h = DaemonHealth::default();
            assert_eq!(h.gate_restart(now), RecoveryGate::Allowed(1), "case {case}");
            // 30s backoff before attempt 2.
            assert_eq!(h.gate_restart(now), RecoveryGate::Backoff, "case {case}");
            assert_eq!(
                h.gate_restart(now + RESTART_BACKOFF[0]),
                RecoveryGate::Allowed(2),
                "case {case}"
            );
            // 2min backoff before attempt 3.
            let t2 = now + RESTART_BACKOFF[0] + RESTART_BACKOFF[1];
            assert_eq!(h.gate_restart(t2), RecoveryGate::Allowed(3), "case {case}");
            // The final 10-min grace is honored before the halt fires.
            assert_eq!(h.gate_restart(t2), RecoveryGate::Backoff, "case {case}");
            let t3 = t2 + RESTART_BACKOFF[2];
            assert_eq!(h.gate_restart(t3), RecoveryGate::Halted, "case {case}");
            assert!(h.restart_budget.halted, "case {case}");
            assert_eq!(h.gate_restart(t3), RecoveryGate::Cooldown, "case {case}");
            // After the cooldown a fresh bounded cycle starts.
            assert_eq!(
                h.gate_restart(t3 + HALT_COOLDOWN),
                RecoveryGate::Allowed(1),
                "case {case}"
            );
            assert_eq!(h.restart_budget.attempts, 1, "case {case}");
            assert!(!h.restart_budget.halted, "case {case}");
        }
        {
            let case = "launch budget is independent of restart budget";
            let now = Instant::now();
            // A halted restart budget must NOT block a launch.
            let mut h = DaemonHealth {
                restart_budget: AttemptBudget {
                    attempts: MAX_RESTART_ATTEMPTS,
                    halted: true,
                    halted_until: Some(now),
                    ..AttemptBudget::default()
                },
                ..DaemonHealth::default()
            };
            assert!(
                matches!(h.gate_launch(now), RecoveryGate::Allowed(_)),
                "case {case}"
            );
            assert_eq!(h.launch_budget.attempts, 1, "case {case}");
            // An exhausted LAUNCH budget still leaves the restart gate open.
            let mut h2 = DaemonHealth {
                launch_budget: AttemptBudget {
                    attempts: MAX_LAUNCH_ATTEMPTS,
                    ..AttemptBudget::default()
                },
                ..DaemonHealth::default()
            };
            assert!(
                !matches!(h2.gate_launch(now), RecoveryGate::Allowed(_)),
                "case {case}"
            );
            assert!(
                matches!(h2.gate_restart(now), RecoveryGate::Allowed(_)),
                "case {case}"
            );
            // Sustained health resets both budgets and clears a recorded launch outcome.
            let mut h3 = DaemonHealth {
                restart_budget: AttemptBudget {
                    attempts: 2,
                    next_at: Some(now),
                    halted: true,
                    halted_until: Some(now),
                },
                launch_budget: AttemptBudget {
                    attempts: 2,
                    next_at: Some(now),
                    halted: true,
                    halted_until: Some(now),
                },
                launch_outcome: Some(ChromeLaunchOutcome::Failed),
                ..DaemonHealth::default()
            };
            // The first healthy seeds the window without resetting either budget; a
            // healthy observation clears a stale launch record.
            h3.apply_outcome(ProbeOutcome::Healthy, now + WATCHDOG_INTERVAL, true);
            assert_eq!(h3.restart_budget.attempts, 2, "case {case}");
            assert_eq!(h3.launch_budget.attempts, 2, "case {case}");
            assert!(h3.restart_budget.halted, "case {case}");
            assert!(h3.launch_budget.halted, "case {case}");
            assert_eq!(h3.launch_outcome, None, "case {case}");
            // A healthy across the sustained window opens fresh bounded cycles.
            h3.apply_outcome(ProbeOutcome::Healthy, now + WATCHDOG_INTERVAL * 3, true);
            assert_eq!(h3.restart_budget.attempts, 0, "case {case}");
            assert_eq!(h3.launch_budget.attempts, 0, "case {case}");
            assert!(!h3.restart_budget.halted, "case {case}");
            assert!(!h3.launch_budget.halted, "case {case}");
        }
    }

    /// The answer's extension block parses into a state that classifies the relay-down
    /// cause: disabled beats absent beats ChromeNotRunning, an absent extension is
    /// unfixable, and only the process fact is recoverable.
    #[test]
    fn the_extension_state_and_the_cause_it_classifies() {
        {
            let case = "relay down classification precedence";
            // Disabled wins regardless of whether Chrome is running.
            assert_eq!(
                classify_relay_down(ExtensionState::Disabled, Some(true)),
                ProbeFailure::ExtensionDisabled,
                "case {case}"
            );
            assert_eq!(
                classify_relay_down(ExtensionState::Disabled, Some(false)),
                ProbeFailure::ExtensionDisabled,
                "case {case}"
            );
            assert_eq!(
                classify_relay_down(ExtensionState::Disabled, None),
                ProbeFailure::ExtensionDisabled,
                "case {case}"
            );
            // Absent beats ChromeNotRunning — launching can't fix an uninstalled extension.
            assert_eq!(
                classify_relay_down(ExtensionState::Absent, Some(false)),
                ProbeFailure::ExtensionAbsent,
                "case {case}"
            );
            assert_eq!(
                classify_relay_down(ExtensionState::Absent, None),
                ProbeFailure::ExtensionAbsent,
                "case {case}"
            );
            // Present + no Chrome running → ChromeNotRunning.
            assert_eq!(
                classify_relay_down(ExtensionState::Present, Some(false)),
                ProbeFailure::ChromeNotRunning,
                "case {case}"
            );
            // Present + Chrome running (or an inconclusive probe) → transient relay drop.
            assert_eq!(
                classify_relay_down(ExtensionState::Present, Some(true)),
                ProbeFailure::RelayDown,
                "case {case}"
            );
            assert_eq!(
                classify_relay_down(ExtensionState::Present, None),
                ProbeFailure::RelayDown,
                "case {case}"
            );
            // Unknown fails open to the probe; only an explicit no-Chrome proves not running.
            assert_eq!(
                classify_relay_down(ExtensionState::Unknown, Some(false)),
                ProbeFailure::ChromeNotRunning,
                "case {case}"
            );
            assert_eq!(
                classify_relay_down(ExtensionState::Unknown, None),
                ProbeFailure::RelayDown,
                "case {case}"
            );
            // ChromeNotRunning is recoverable (launch), ExtensionAbsent is not.
            assert!(
                !ProbeFailure::ChromeNotRunning.is_unfixable(),
                "case {case}"
            );
            assert!(ProbeFailure::ExtensionAbsent.is_unfixable(), "case {case}");
        }
        {
            let case = "extension state parsing";
            // No `data`, or no `chromeExtension` in it → shape-uncertain, never the absent
            // cause (fails open as unknown).
            assert_eq!(
                extension_state_from(&serde_json::json!({})),
                ExtensionState::Unknown,
                "case {case}"
            );
            assert_eq!(
                extension_state_from(&serde_json::json!({ "data": {} })),
                ExtensionState::Unknown,
                "case {case}"
            );
            // Explicit null → absent (the CLI reports the extension as not present).
            assert_eq!(
                extension_state_from(&serde_json::json!({ "data": { "chromeExtension": null } })),
                ExtensionState::Absent,
                "case {case}"
            );
            // Present with an empty disableReasons, or with no disableReasons at all.
            assert_eq!(
                extension_state_from(&serde_json::json!({
                    "data": { "chromeExtension": { "disableReasons": [] } }
                })),
                ExtensionState::Present,
                "case {case}"
            );
            assert_eq!(
                extension_state_from(&serde_json::json!({
                    "data": { "chromeExtension": { "enabled": true } }
                })),
                ExtensionState::Present,
                "case {case}"
            );
            // Disabled when disableReasons is non-empty.
            assert_eq!(
                extension_state_from(&serde_json::json!({
                    "data": { "chromeExtension": { "disableReasons": ["user"] } }
                })),
                ExtensionState::Disabled,
                "case {case}"
            );
        }
    }

    /// Readiness reports every fact it established and never defaults a missing one.
    #[tokio::test]
    #[expect(clippy::too_many_lines)]
    async fn readiness_reports_every_fact_and_never_defaults_a_missing_one() {
        {
            let case = "readiness reports every fact and refuses without a real browser";
            // The healthy snapshot: every fact established, ready for actions.
            let status = serde_json::json!({
                "data": {
                    "cliVersion": "1.5.141",
                    "extension": {
                        "hostInstalled": true,
                        "hostHealthy": true,
                        "relayUp": true,
                        "profileEmail": "owner@example.com",
                        "profileId": "abc-123"
                    }
                }
            });
            let browsers = serde_json::json!({
                "data": { "browsers": [{ "email": "owner@example.com", "id": "abc-123" }] }
            });
            let mut ready = Readiness::from_status(Some(&status));
            ready.chrome_running = Some(true);
            ready.display = Some(true);
            ready.real_browser = Readiness::real_browser_from(Some(&browsers));
            assert!(ready.ready_for_actions(), "case {case}");
            assert_eq!(ready.outcome(), ProbeOutcome::Healthy, "case {case}");
            let report = ready.report();
            assert!(report.contains("established:"), "case {case}");
            assert!(
                report.contains("established as NOT holding:\n    - nothing"),
                "case {case}"
            );
            assert!(
                report.contains("not established:\n    - nothing"),
                "case {case}"
            );
            assert!(
                report.contains("chrome-use 1.5.141"),
                "case {case}: got: {report}"
            );
            assert!(
                report.contains("driving the real profile owner@example.com (abc-123)"),
                "case {case}: got: {report}"
            );
            assert!(
                report.contains("a real browser is reachable through the relay"),
                "case {case}: got: {report}"
            );
            // Process presence is reported as a fact, never as health.
            assert!(
                report.contains("process presence is not by itself health"),
                "case {case}: got: {report}"
            );
            assert!(
                report.contains("this host has a usable display"),
                "case {case}: got: {report}"
            );
            assert!(
                report.contains("verdict: ready for actions"),
                "case {case}: got: {report}"
            );

            // An UNANSWERED `browsers` list (a shape change in a future chrome-use) is a
            // reporting gap: not proven, `Unknown` rather than health, and no invented cause.
            let mut unanswered_browsers = ready.clone();
            unanswered_browsers.real_browser = None;
            assert!(!unanswered_browsers.ready_for_actions(), "case {case}");
            assert_eq!(
                unanswered_browsers.outcome(),
                ProbeOutcome::Unknown,
                "case {case}"
            );
            assert!(!unanswered_browsers.blocked(), "case {case}");

            // The failure this gate exists for: relay up but no real browser reachable —
            // not ready, and the refusal says what would have happened instead. The
            // negative is reported under "established as NOT holding".
            let empty = serde_json::json!({ "data": { "browsers": [] } });
            let mut not_ready = Readiness::from_status(Some(&status));
            not_ready.chrome_running = Some(true);
            not_ready.display = Some(true);
            not_ready.real_browser = Readiness::real_browser_from(Some(&empty));
            assert!(!not_ready.ready_for_actions(), "case {case}");
            assert_eq!(
                not_ready.down_cause(),
                Some(ProbeFailure::RelayDown),
                "case {case}"
            );
            assert_eq!(
                not_ready.outcome(),
                ProbeOutcome::Down(ProbeFailure::RelayDown),
                "case {case}"
            );
            let not_ready_report = not_ready.report();
            assert!(
                not_ready_report.contains(
                    "established as NOT holding:\n    - no real browser profile is reachable through \
                     the relay"
                ),
                "case {case}: got: {not_ready_report}"
            );
            // An established cause is the one verdict that says the connection was ruled
            // out, so the action is refused rather than attempted.
            assert!(
                not_ready_report.contains("verdict: NOT ready for actions"),
                "case {case}: got: {not_ready_report}"
            );
            assert!(
                not_ready
                    .refusal()
                    .contains("the work would have gone to a browser chrome-use launches itself"),
                "case {case}: got: {}",
                not_ready.refusal()
            );

            // A host with NO display refuses nothing: a display is only needed to LAUNCH,
            // and a running browser stays reachable, so the connection is still proven.
            let mut headless = ready.clone();
            headless.display = Some(false);
            assert!(headless.ready_for_actions(), "case {case}");
            assert_eq!(headless.down_cause(), None, "case {case}");
            assert!(!headless.blocked(), "case {case}");
            let headless_report = headless.report();
            assert!(
                headless_report.contains("this host has NO usable display"),
                "case {case}: got: {headless_report}"
            );
            assert!(
                headless_report.contains("verdict: ready for actions"),
                "case {case}: got: {headless_report}"
            );
            // The display blocks nothing even when another DECISIVE fact is missing:
            // the connection is then not proven (`Unknown`), never ruled out.
            let mut headless_unproven = headless.clone();
            headless_unproven.real_browser = None;
            assert!(!headless_unproven.ready_for_actions(), "case {case}");
            assert_eq!(
                headless_unproven.outcome(),
                ProbeOutcome::Unknown,
                "case {case}"
            );
            assert!(!headless_unproven.blocked(), "case {case}");
        }
        {
            let case = "readiness never defaults a missing fact to fine";
            // Nothing answered: every fact stays unestablished, the verdict is Unknown,
            // and no cause is claimed — keeping an unanswerable probe out of both the
            // health verdict and the restart budget.
            let nothing = Readiness::default();
            assert!(!nothing.answered, "case {case}");
            assert!(!nothing.ready_for_actions(), "case {case}");
            assert_eq!(nothing.outcome(), ProbeOutcome::Unknown, "case {case}");
            assert_eq!(nothing.down_cause(), None, "case {case}");
            let report = nothing.report();
            assert!(
                report.contains("established:\n    - nothing"),
                "case {case}"
            );
            assert!(
                report.contains("established as NOT holding:"),
                "case {case}"
            );
            assert!(report.contains("not established:"), "case {case}");
            assert!(
                report.contains("whether the tool answers `status` at all"),
                "case {case}: got: {report}"
            );
            // Not proven and nothing ruled out: the verdict claims neither.
            assert!(
                report.contains("verdict: NOT proven ready for actions"),
                "case {case}: got: {report}"
            );
            assert!(
                !report.contains("verdict: NOT ready for actions"),
                "case {case}: got: {report}"
            );

            // A status that answered but carries no extension data: every decisive key is
            // missing, so nothing is established.
            let bare = serde_json::json!({ "data": { "cliVersion": "1.2.3" } });
            let mut r = Readiness::from_status(Some(&bare));
            r.chrome_running = Some(true);
            assert!(r.answered, "case {case}");
            assert_eq!(r.relay_up, None, "case {case}");
            assert_eq!(r.host_installed, None, "case {case}");
            assert!(!r.ready_for_actions(), "case {case}");
            assert_eq!(r.outcome(), ProbeOutcome::Unknown, "case {case}");
        }
        {
            let case = "readiness classifies only what a fact established";
            // Installed host + relay down + closed browser → ChromeNotRunning, so the gate
            // launches Chrome rather than restarting a daemon.
            let status = serde_json::json!({
                "data": { "extension": {
                    "hostInstalled": true,
                    "hostHealthy": true,
                    "relayUp": false
                } }
            });
            let mut relay_down = Readiness::from_status(Some(&status));
            relay_down.extension = Some(ExtensionState::Present);
            relay_down.chrome_running = Some(false);
            assert_eq!(
                relay_down.down_cause(),
                Some(ProbeFailure::ChromeNotRunning),
                "case {case}"
            );
            // The process fact is stated flatly in the established-as-NOT-holding list.
            let report = relay_down.report();
            assert!(
                report.contains("    - no Chrome-family process is running"),
                "case {case}: got: {report}"
            );
            // An absent extension is unfixable — no launch or restart recovers it.
            let mut absent = relay_down;
            absent.extension = Some(ExtensionState::Absent);
            assert_eq!(
                absent.down_cause(),
                Some(ProbeFailure::ExtensionAbsent),
                "case {case}"
            );
            let absent_cause = absent.down_cause();
            assert!(
                absent_cause.is_some_and(ProbeFailure::is_unfixable),
                "case {case}"
            );
            // The native-host facts classify before the relay does.
            let uninstalled =
                serde_json::json!({ "data": { "extension": { "hostInstalled": false } } });
            assert_eq!(
                Readiness::from_status(Some(&uninstalled)).down_cause(),
                Some(ProbeFailure::NotInstalled),
                "case {case}"
            );
        }
    }

    /// A daemon or relay outage is recognised from the error code and the failure
    /// text alike, and both detection paths share the same classification.
    #[tokio::test]
    async fn a_daemon_or_relay_outage_is_recognised_by_code_or_signature() {
        {
            let case = "daemon unavailable error signature detected";
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
                assert!(
                    is_daemon_unavailable_error(msg),
                    "case {case}: should detect: {msg}"
                );
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
                    "case {case}: should NOT detect: {msg}"
                );
            }
        }
        {
            let case = "relay unavailable signature detected";
            for msg in [
                "The chrome-use extension is installed, but its relay isn't connected.",
                "Could not drive your Chrome through the ab-connect extension.",
                "Chrome relay dropped — reconnecting…",
            ] {
                assert!(
                    is_relay_unavailable_error(msg),
                    "case {case}: should detect: {msg}"
                );
            }
            for msg in [
                "chrome-use error: Element not found",
                "Failed to read: Resource temporarily unavailable (os error 35)",
            ] {
                assert!(
                    !is_relay_unavailable_error(msg),
                    "case {case}: should NOT detect: {msg}"
                );
            }
        }
        {
            let case = "daemon unavailable code detected";
            assert!(
                is_daemon_unavailable_code(Some("browser_not_launched")),
                "case {case}"
            );
            // Page-level failures share the coarse `connection_failed` code with
            // daemon-socket problems, so the code alone must not fail fast.
            assert!(
                !is_daemon_unavailable_code(Some("connection_failed")),
                "case {case}"
            );
            assert!(!is_daemon_unavailable_code(Some("timeout")), "case {case}");
            assert!(
                !is_daemon_unavailable_code(Some("element_not_found")),
                "case {case}"
            );
            assert!(!is_daemon_unavailable_code(None), "case {case}");
        }
        {
            let case = "failure text classification is shared between detection paths";
            // A stale tab wrapped in the auto-connect envelope AND the daemon wrapper
            // classifies as unreachable-tab, NOT relay-down or wedge: recovery must not
            // fire for an orphaned tab on a healthy relay.
            assert_eq!(
                classify_failure_text(
                    "Auto-launch failed: Could not drive your Chrome through the ab-connect \
                     extension. The tab this session was driving can no longer be resolved (it \
                     was closed, or a flaky relay dropped it)"
                ),
                Some(ProbeFailure::UnreachableTab),
                "case {case}"
            );
            // Auto-connect failure alone names the relay: its signature wins over the
            // daemon wrapper it sits in, in both detection paths.
            assert_eq!(
                classify_failure_text(
                    "Auto-launch failed: Could not drive your Chrome through the ab-connect extension."
                ),
                Some(ProbeFailure::RelayDown),
                "case {case}"
            );
            assert_eq!(
                classify_failure_text(
                    "Failed to connect: the daemon endpoint for session '__mahbot_probe' \
                     disappeared (/tmp/x.sock)."
                ),
                Some(ProbeFailure::DaemonWedge),
                "case {case}"
            );
            assert_eq!(
                classify_failure_text("chrome-use error: Element not found"),
                None,
                "case {case}"
            );
        }
    }

    #[test]
    fn spawn_error_classification_distinguishes_missing_from_transient() {
        // Only a genuinely missing binary (NotFound) is definitive absence; every other
        // spawn error is transient and must never be reported as "not installed".
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
        // Windows on ARM takes the ordinary x64 asset (the helper publishes no
        // ARM Windows build); the musl flag is ignored there too.
        assert_eq!(
            release_asset_platform("windows", "aarch64", false).as_deref(),
            Some("win32-x64")
        );
        assert_eq!(
            release_asset_platform("windows", "aarch64", true).as_deref(),
            Some("win32-x64")
        );
        // Unsupported platform/arch combos return None.
        assert_eq!(release_asset_platform("freebsd", "x86_64", false), None);
        assert_eq!(release_asset_platform("freebsd", "aarch64", false), None);
    }
}
