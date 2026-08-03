//! chrome-use daemon health monitoring and bounded auto-recovery.
//!
//! The chrome-use CLI talks to a per-session background daemon that drives
//! Chrome through the extension relay. When the daemon or relay dies, CLI
//! commands hang inside its own 5-retry loop (~152 s) instead of failing fast.
//! This module tracks daemon health with a two-stage probe — a daemon-free
//! `status` snapshot that classifies the cause (extension disabled, relay down,
//! host missing, …) plus a bounded daemon round-trip that detects wedges — and
//! auto-restarts the daemon with bounded backoff and thrash protection. It also
//! owns the chrome-use CLI invocation primitives (binary name, env setup,
//! `--version` check) so the browser tool depends on this module and not vice
//! versa.
//!
//! Probe hygiene: the daemon round-trip runs on a dedicated `__mahbot_probe`
//! session and that session is stopped immediately afterwards (`session stop`
//! closes the daemon and the scratch tab it created). A healthy host therefore
//! has zero probe tabs, and repeated probes never accumulate any. This also
//! means the probe no longer keeps a daemon (or its Chrome instance) resident —
//! the old watchdog left its probe daemon (and scratch tab) running forever;
//! now every probe stops it.
//!
//! Trade-offs:
//! - Each probe briefly spawns the probe daemon and its background scratch tab
//!   (~1–2 s per probe, in the collapsed `__mahbot_probe` tab group) — the
//!   cost of a real socket round-trip with zero persistent tabs.
//! - `daemon restart` destroys all session state; recovery guidance notes that
//!   existing browser sessions are reset.
//! - Residual latency: a whole-daemon wedge is normally caught by the next
//!   probe (watchdog or per-call) at 8 s. The CLI's ~152 s retry loop is only
//!   paid when the daemon dies inside the 10 s healthy-probe cache window (the
//!   call's own run_command is the first to notice) or when the wedge is
//!   session-specific so the probe session stays healthy. Either way the error
//!   signature then marks the daemon unhealthy and subsequent calls fail fast.

use crate::util::UnwrapPoison;
use serde_json::Value;
use std::process::Stdio;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::process::Command;
use tracing::{debug, error, info, warn};

/// Fixed session name used by the liveness probe. The probe's daemon is stopped
/// after every probe so its scratch tab never persists.
const PROBE_SESSION: &str = "__mahbot_probe";

// ── Bounds (pinned for deterministic recovery) ────────────────────────────
/// How long a probe command may take before the daemon is considered wedged.
/// A healthy daemon answers in milliseconds; a wedged one hangs for the CLI's
/// internal 45s read timeouts × 5 retries.
const PROBE_TIMEOUT: Duration = Duration::from_secs(8);
/// Cache TTL for a healthy probe result (fresh enough for per-call checks).
const HEALTH_TTL: Duration = Duration::from_secs(10);
/// Longer TTL for a confirmed-down result, so repeated browser calls fail fast
/// instead of re-probing (and re-paying the 8s timeout) on every invocation.
const UNHEALTHY_TTL: Duration = Duration::from_mins(1);
/// Watchdog cadence between automatic health probes.
const WATCHDOG_INTERVAL: Duration = Duration::from_secs(30);
/// How often the watchdog re-verifies CLI presence on hosts where it was
/// found — the binary can be uninstalled while the daemon runs, but checking
/// every probe interval would spawn `--version` needlessly.
const CLI_RECHECK: Duration = Duration::from_mins(5);
/// Consecutive failed restarts before auto-recovery halts (thrash protection).
const MAX_RESTART_ATTEMPTS: u32 = 3;
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

/// Classified cause for a failed health probe. Drives cause-specific warnings
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
    /// The daemon socket hung or errored (daemon-side wedge).
    DaemonWedge,
}

/// Result of a health probe: healthy, or down with a classified cause.
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
            ProbeFailure::NotInstalled | ProbeFailure::HostBroken | ProbeFailure::ExtensionDisabled
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
    /// Last classified probe failure — surfaces the cause in LLM-facing
    /// messages and drives transition-based warning logging.
    last_failure: Option<ProbeFailure>,
    /// The failure cause the last transition-based warning named — reset on
    /// recovery so the same cause warns again after a healthy spell.
    last_cause_warned: Option<ProbeFailure>,
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
}

static HEALTH: OnceLock<Mutex<DaemonHealth>> = OnceLock::new();
static WAKE: OnceLock<tokio::sync::Notify> = OnceLock::new();
/// Serializes health probes — a probe ends by stopping the probe session's
/// daemon, which would tear the daemon out from under a concurrent in-flight
/// command and misclassify it as a wedge on a healthy host. Concurrent
/// callers (is_available, watchdog, recovery) queue on this one lock.
static PROBE_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

fn health() -> &'static Mutex<DaemonHealth> {
    HEALTH.get_or_init(|| Mutex::new(DaemonHealth::default()))
}

fn wake() -> &'static tokio::sync::Notify {
    WAKE.get_or_init(tokio::sync::Notify::new)
}

/// Serialize probe-session mutation — a `session stop` tears the probe daemon
/// out from under a concurrent in-flight probe command, which would misclassify
/// a healthy host as wedged. Held by probes and the startup cleanup.
async fn probe_session_lock() -> tokio::sync::MutexGuard<'static, ()> {
    PROBE_LOCK
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
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

/// Classify a fast CLI failure text into a probe cause. The relay signature is
/// the more specific symptom (auto-connect failures name the relay as the
/// cause), so it wins over the daemon wrapper it is wrapped in — both the
/// probe and the fail-fast path must agree on the cause.
fn classify_failure_text(msg: &str) -> Option<ProbeFailure> {
    if is_relay_unavailable_error(msg) {
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

/// Whether the chrome-use CLI binary is installed and runs (--version check).
pub(crate) async fn cli_available() -> bool {
    Command::new(browser_bin())
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .is_ok_and(|s| s.success())
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
    // 5-minute idle timeout — the chrome-use daemon shuts down after
    // 5 minutes of inactivity, closing the tabs it created. The watchdog no
    // longer keeps a probe daemon resident (it stops after every probe), so
    // browser sessions idle out on this bound naturally.
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
    // cause-aware recovery and turning a probe into a 45s stall.
    cmd.env("AGENT_BROWSER_NO_AUTO_RECONNECT", "1");
    cmd.env("AGENT_BROWSER_RELAY_REVIVE_SECS", "0");
}

/// Run a chrome-use CLI command that emits a `--json` response and parse it.
/// Bounded by [`PROBE_TIMEOUT`]; returns `None` on timeout, spawn failure,
/// non-zero exit, or unparseable output (callers degrade to the daemon
/// round-trip, which does not depend on the JSON commands).
async fn run_cli_json(args: &[&str]) -> Option<Value> {
    let mut cmd = Command::new(browser_bin());
    ensure_browser_env(&mut cmd);
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::null());
    cmd.kill_on_drop(true);
    let out = tokio::time::timeout(PROBE_TIMEOUT, cmd.output())
        .await
        .ok()?
        .ok()?;
    if !out.status.success() {
        return None;
    }
    serde_json::from_slice(&out.stdout).ok()
}

/// Daemon-free service-state snapshot: `status --json` classifies extension /
/// native-host / relay problems without spawning a daemon or tab. Returns a
/// classified failure when the service is unusable; `None` when it looks
/// healthy (the caller then runs the daemon round-trip).
async fn service_state() -> Option<ProbeFailure> {
    let Some(status) = run_cli_json(&["status", "--json"]).await else {
        // Status unavailable (old CLI or broken binary) — fall through to the
        // round-trip, which only needs the socket protocol.
        return None;
    };
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
    let Some(status) = run_cli_json(&["extension", "status", "--json"]).await else {
        return false; // Unknown → treat as a transient relay drop, not disabled.
    };
    status
        .get("data")
        .and_then(|d| d.get("chromeExtension"))
        .and_then(|c| c.get("disableReasons"))
        .and_then(Value::as_array)
        .is_some_and(|reasons| !reasons.is_empty())
}

/// Real daemon round-trip on the probe session. `get url` spawns a probe
/// daemon (with its background scratch tab) if none is running and proves the
/// socket answers within the bound. A wedged daemon — or one bound to a dead
/// relay — hangs here (the `url` action is a live CDP eval through the relay).
/// The caller stops the probe session afterwards so the scratch tab never
/// persists.
async fn round_trip_probe() -> ProbeOutcome {
    let mut cmd = Command::new(browser_bin());
    ensure_browser_env(&mut cmd);
    cmd.args(["get", "url", "--json"])
        .args(["--session", PROBE_SESSION]);
    cmd.kill_on_drop(true);
    match tokio::time::timeout(PROBE_TIMEOUT, cmd.output()).await {
        // Timed out (wedged or dead relay) or failed to spawn — daemon-side.
        Err(_) | Ok(Err(_)) => ProbeOutcome::Down(ProbeFailure::DaemonWedge),
        Ok(Ok(out)) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let combined = format!("{stdout} {stderr}");
            if out.status.success() {
                return ProbeOutcome::Healthy;
            }
            if let Some(failure) = classify_failure_text(&combined) {
                return ProbeOutcome::Down(failure);
            }
            // Any other fast response proves the daemon socket round-tripped.
            ProbeOutcome::Healthy
        }
    }
}

/// Full health probe: service-state classification first (daemon-free, no tab),
/// then a bounded daemon round-trip when the service looks healthy. The probe
/// session is stopped on every path so its scratch tab never persists.
async fn probe_daemon() -> ProbeOutcome {
    // Serialized with the startup cleanup and other probes (see probe_session_lock).
    let _guard = probe_session_lock().await;
    let outcome = match service_state().await {
        Some(failure) => ProbeOutcome::Down(failure),
        None => round_trip_probe().await,
    };
    // Stop the probe daemon (closing its scratch tab) on both success and
    // failure paths — a leftover from a previous probe is cleaned here too.
    let _ = run_cli(&["session", "stop", PROBE_SESSION]).await;
    outcome
}

fn set_health(outcome: ProbeOutcome) {
    let mut h = health().lock().unwrap_poison();
    let healthy = outcome.is_healthy();
    let failure = outcome.failure();
    if healthy {
        // Natural recovery — restart attempts count consecutive failures only.
        h.restart_attempts = 0;
        h.next_restart_at = None;
        h.halted = false;
        h.halted_until = None;
        h.last_cause_warned = None;
    }
    // A cause change never resets the restart budget — flapping causes (e.g.
    // RelayDown ↔ DaemonWedge) must not evade the attempt halt. Only a healthy
    // probe (or the cooldown expiry in gate_restart) opens a fresh cycle.
    h.last_failure = failure;
    h.healthy = Some(healthy);
    h.last_probe = Some(Instant::now());
}

/// Async availability for call paths: uses a fresh cached probe when possible,
/// otherwise runs a real probe (bounded) and caches the result. A fresh
/// down-result wakes the watchdog so recovery starts without waiting for the
/// next interval.
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
    let outcome = probe_daemon().await;
    let healthy = outcome.is_healthy();
    set_health(outcome);
    if !healthy {
        wake().notify_one();
    }
    healthy
}

/// Sync availability for tool advertisement (never probes — uses the last
/// known state). Unknown → advertise optimistically; only a confirmed-down
/// probe hides the tool.
pub(crate) fn is_advertised() -> bool {
    health().lock().unwrap_poison().healthy != Some(false)
}

/// Mark the daemon unhealthy immediately (fail-fast path) with the cause the
/// error text points to, and wake the watchdog so recovery starts without
/// waiting for the next interval.
pub(crate) fn note_unhealthy(error: &str) {
    // Same classification as the probe path — the two detection paths must
    // agree on the cause.
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

/// Background watchdog: probe daemon health, auto-restart with bounded backoff
/// when down, and halt after repeated crashes to avoid a restart loop. Stands
/// down on hosts without the chrome-use CLI (nothing to monitor or restart).
/// CLI presence is cached after the first success and re-verified every
/// [`CLI_RECHECK`] so healthy hosts don't spawn `--version` per interval.
pub async fn run_watchdog() {
    let mut cli_present: Option<bool> = None;
    let mut last_cli_check = Instant::now();
    // One-time sweep of leaked mahbot-owned session artifacts from crashed runs
    // or older versions (see cleanup_stale_sessions).
    let mut cleaned = false;
    loop {
        let cli_due = last_cli_check.elapsed() >= CLI_RECHECK;
        if cli_present != Some(true) || cli_due {
            last_cli_check = Instant::now();
            if !cli_available().await {
                if cli_present != Some(false) {
                    cli_present = Some(false);
                    warn!("chrome-use CLI not found — browser daemon watchdog standing down");
                }
                // Re-check rarely on CLI-less hosts so the watchdog doesn't
                // spawn `--version` every interval; an early wake re-checks.
                let shutdown = crate::shutdown::shutdown_token();
                tokio::select! {
                    () = tokio::time::sleep(CLI_RECHECK) => {}
                    () = wake().notified() => {}
                    () = shutdown.cancelled() => break,
                }
                continue;
            }
            cli_present = Some(true);
        }
        if !cleaned {
            cleaned = true;
            cleanup_stale_sessions().await;
        }
        let outcome = probe_daemon().await;
        set_health(outcome);
        if let ProbeOutcome::Down(failure) = outcome {
            attempt_recovery(failure).await;
        }
        // Wait for the next interval or an early wake from the fail-fast path.
        let shutdown = crate::shutdown::shutdown_token();
        tokio::select! {
            () = tokio::time::sleep(WATCHDOG_INTERVAL) => {}
            () = wake().notified() => {}
            () = shutdown.cancelled() => break,
        }
    }
}

/// One-time cleanup of stale mahbot-owned browser-session artifacts at watchdog
/// start: the probe session and leftover link-enricher sessions get stopped so
/// their tab groups don't accumulate. `session stop` is idempotent and
/// dead-session safe, and only ever closes the target session's own tabs —
/// sessions owned by other agents or the user (explicit tabs, `default`, any
/// non-mahbot name) are never touched.
async fn cleanup_stale_sessions() {
    // Serialized with probes — stopping the probe session tears its daemon out
    // from under a concurrent in-flight probe (see probe_session_lock).
    let _guard = probe_session_lock().await;
    let _ = run_cli(&["session", "stop", PROBE_SESSION]).await;
    let Some(sessions) = registered_sessions().await else {
        return;
    };
    for name in sessions {
        if name.starts_with("link-enricher-") {
            let _ = run_cli(&["session", "stop", &name]).await;
        }
    }
}

/// Names of currently registered session daemons (from the daemon-free
/// `status --json` snapshot).
async fn registered_sessions() -> Option<Vec<String>> {
    let status = run_cli_json(&["status", "--json"]).await?;
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
    let status = run_cli_json(&["status", "--json"]).await?;
    status
        .get("data")?
        .get("extension")?
        .get("relayUp")
        .and_then(Value::as_bool)
}

/// Warn once per cause transition — an ongoing failure does not spam every
/// probe interval, but the same cause warns again after a healthy spell.
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
        ProbeFailure::DaemonWedge => {
            warn!("browser daemon is unresponsive — restarting it (bounded backoff).");
        }
    }
}

/// Bounded auto-recovery: restart session daemons with backoff between attempts
/// and a halt after MAX_RESTART_ATTEMPTS failures. Causes that a restart cannot
/// fix — extension disabled, not installed, broken host — are reported with
/// their concrete fix and never consume restart attempts. A transient relay
/// drop is waited out first and consumes no attempt if it self-heals.
async fn attempt_recovery(mut failure: ProbeFailure) {
    warn_transition(failure);
    // Unfixable causes stop here — they never consume restart attempts.
    if failure.is_unfixable() {
        return;
    }
    // While a recovery timer (restart backoff or halt cooldown) is pending, the
    // timer IS the wait — don't poll the relay for up to RELAY_REVIVE_WAIT on
    // top of it. The next watchdog cycle re-probes and re-enters recovery.
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
        let outcome = probe_daemon().await;
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

    let outcome = probe_daemon().await;
    set_health(outcome);
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
    let mut cmd = Command::new(browser_bin());
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

    #[tokio::test]
    async fn healthy_probe_resets_restart_attempts() {
        let _guard = with_health_test_lock().await;
        {
            let mut h = health().lock().unwrap_poison();
            h.restart_attempts = 2;
            h.next_restart_at = Some(Instant::now());
            h.halted = true;
            h.halted_until = Some(Instant::now());
        }
        // Natural recovery (a healthy probe landing between two incidents)
        // resets the cycle so attempts count consecutive failures only.
        set_health(ProbeOutcome::Healthy);
        {
            let h = health().lock().unwrap_poison();
            assert_eq!(h.restart_attempts, 0);
            assert_eq!(h.next_restart_at, None);
            assert!(!h.halted);
            assert!(h.halted_until.is_none());
            assert_eq!(h.last_failure, None);
        }
        // Restore pristine state like the sibling tests.
        reset_health();
    }

    #[tokio::test]
    async fn cause_flapping_does_not_reset_restart_budget() {
        let _guard = with_health_test_lock().await;
        // A wedge failure with accumulated attempts.
        {
            let mut h = health().lock().unwrap_poison();
            h.restart_attempts = 2;
            h.next_restart_at = Some(Instant::now());
            h.last_failure = Some(ProbeFailure::DaemonWedge);
        }
        // A cause flip (wedge → relay-down) must NOT reset the budget —
        // alternating causes must not evade the 3-attempt halt.
        set_health(ProbeOutcome::Down(ProbeFailure::RelayDown));
        {
            let h = health().lock().unwrap_poison();
            assert_eq!(h.last_failure, Some(ProbeFailure::RelayDown));
            assert_eq!(h.restart_attempts, 2);
            assert!(h.next_restart_at.is_some());
        }
        // Flapping back and forth accumulates — never resets.
        set_health(ProbeOutcome::Down(ProbeFailure::DaemonWedge));
        set_health(ProbeOutcome::Down(ProbeFailure::RelayDown));
        {
            let h = health().lock().unwrap_poison();
            assert_eq!(h.last_failure, Some(ProbeFailure::RelayDown));
            assert_eq!(h.restart_attempts, 2);
        }
        // Only a healthy probe opens a fresh bounded cycle.
        set_health(ProbeOutcome::Healthy);
        {
            let h = health().lock().unwrap_poison();
            assert_eq!(h.last_failure, None);
            assert_eq!(h.restart_attempts, 0);
            assert_eq!(h.next_restart_at, None);
            assert!(!h.halted);
        }
        reset_health();
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
        // Auto-connect failure names the relay as the cause (its body points at
        // `chrome-use extension connect`) — the relay signature wins over the
        // daemon wrapper it is wrapped in, in both the probe and fail-fast
        // paths, so they never disagree on the cause.
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
}
