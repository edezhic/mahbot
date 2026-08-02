//! chrome-use daemon health monitoring and bounded auto-recovery.
//!
//! The chrome-use CLI talks to a per-session background daemon that drives
//! Chrome through the extension relay. When the daemon or relay dies, CLI
//! commands hang inside its own 5-retry loop (~152 s) instead of failing fast.
//! This module tracks daemon liveness with a real round-trip probe, gates the
//! browser tool's advertisement on it, and auto-restarts the daemon with
//! bounded backoff and thrash protection. It also owns the chrome-use CLI
//! invocation primitives (binary name, env setup, `--version` check) so the
//! browser tool depends on this module and not vice versa.
//!
//! Trade-offs:
//! - The 30 s watchdog probe keeps the daemon (and its Chrome instance)
//!   resident on hosts with chrome-use installed, defeating the daemon's
//!   5-minute idle shutdown. That is the cost of fast wedge detection and
//!   automatic recovery.
//! - `daemon restart` destroys all session state; recovery guidance notes that
//!   existing browser sessions are reset.
//! - Residual latency: a whole-daemon wedge is normally caught by the next
//!   probe (watchdog or per-call) at 8 s. The CLI's ~152 s retry loop is only
//!   paid when the daemon dies inside the 10 s healthy-probe cache window (the
//!   call's own run_command is the first to notice) or when the wedge is
//!   session-specific so the probe session stays healthy. Either way the error
//!   signature then marks the daemon unhealthy and subsequent calls fail fast.

use crate::util::UnwrapPoison;
use std::process::Stdio;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::process::Command;
use tracing::{debug, error, info, warn};

/// Fixed session name used by the liveness probe. Reusing one session keeps a
/// single probe daemon alive instead of spawning a new one per check.
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

#[derive(Default)]
struct DaemonHealth {
    healthy: Option<bool>,
    last_probe: Option<Instant>,
    restart_attempts: u32,
    next_restart_at: Option<Instant>,
    halted: bool,
    halted_until: Option<Instant>,
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

fn health() -> &'static Mutex<DaemonHealth> {
    HEALTH.get_or_init(|| Mutex::new(DaemonHealth::default()))
}

fn wake() -> &'static tokio::sync::Notify {
    WAKE.get_or_init(tokio::sync::Notify::new)
}

/// Detect the daemon-unavailable signature chrome-use produces when its
/// background daemon is dead or wedged: the CLI hangs in its own 5-retry loop
/// (EAGAIN / "Resource temporarily unavailable") and eventually reports
/// "daemon may be busy or unresponsive".
pub(crate) fn is_daemon_unavailable_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("resource temporarily unavailable")
        || lower.contains("os error 35")
        || lower.contains("os error 11")
        || lower.contains("daemon may be busy or unresponsive")
        || lower.contains("session unresponsive")
        // Colon form only — "failed to connect to <host>" is a page-level
        // navigation failure, not a daemon socket problem.
        || lower.contains("failed to connect:")
        || lower.contains("daemon failed to start")
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
    // 5 minutes of inactivity, cleaning up browser resources. The watchdog's
    // probe cadence defeats this while the daemon is healthy (see module doc).
    cmd.env("AGENT_BROWSER_IDLE_TIMEOUT_MS", "300000");
    // Enable human-like interaction speed for bot-detection avoidance.
    // chrome-use supports the same env vars as agent-browser for backward
    // compatibility.
    cmd.env("AGENT_BROWSER_HUMANIZE", "human");
}

/// Real daemon round-trip: `get url` on a fixed probe session. A wedged daemon
/// hangs here (the CLI's internal retry loop); our timeout turns that into a
/// fast unhealthy signal. A completed response — success or a fast error like
/// "no Chrome running" — proves the daemon socket round-tripped, unless the
/// error itself carries the daemon-unavailable signature (e.g. ENOENT socket).
async fn probe_daemon() -> bool {
    let mut cmd = Command::new(browser_bin());
    ensure_browser_env(&mut cmd);
    cmd.args(["get", "url", "--json"])
        .args(["--session", PROBE_SESSION]);
    cmd.kill_on_drop(true);
    match tokio::time::timeout(PROBE_TIMEOUT, cmd.output()).await {
        // Timed out (wedged) or failed to spawn — both mean unhealthy.
        Err(_) | Ok(Err(_)) => false,
        Ok(Ok(out)) => {
            if out.status.success() {
                return true;
            }
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            !is_daemon_unavailable_error(&format!("{stdout} {stderr}"))
        }
    }
}

fn set_health(healthy: bool) {
    let mut h = health().lock().unwrap_poison();
    h.healthy = Some(healthy);
    h.last_probe = Some(Instant::now());
    if healthy {
        // Natural recovery — restart attempts count consecutive failures only.
        h.restart_attempts = 0;
        h.next_restart_at = None;
        h.halted = false;
        h.halted_until = None;
    }
}

/// Async availability for call paths: uses a fresh cached probe when possible,
/// otherwise runs a real round-trip (bounded) and caches the result. A fresh
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
    let healthy = probe_daemon().await;
    set_health(healthy);
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

/// Mark the daemon unhealthy immediately (fail-fast path) and wake the
/// watchdog so recovery starts without waiting for the next interval.
pub(crate) fn note_unhealthy() {
    set_health(false);
    wake().notify_one();
}

/// Actionable error shown when the daemon is down. Reflects whether
/// auto-recovery is active or frozen by thrash protection.
pub(crate) fn daemon_down_message() -> String {
    if health().lock().unwrap_poison().halted {
        "The chrome-use browser daemon is down or unresponsive. Auto-recovery \
         exhausted its restart attempts and is in a 30-minute cooldown (thrash \
         protection); it will retry after the cooldown. While it's down, use \
         web_search, or shell `curl` for page fetches, instead of the browser tool."
            .to_string()
    } else {
        "The chrome-use browser daemon is down or unresponsive. Auto-recovery \
         was triggered and will restart it automatically — no manual action is \
         needed (note: the restart resets browser sessions). While it's down, \
         use web_search, or shell `curl` for page fetches, instead of the \
         browser tool."
            .to_string()
    }
}

/// Background watchdog: probe daemon health, auto-restart with bounded backoff
/// when down, and halt after repeated crashes to avoid a restart loop. Stands
/// down on hosts without the chrome-use CLI (nothing to monitor or restart).
/// CLI presence is cached after the first success and re-verified every
/// [`CLI_RECHECK`] so healthy hosts don't spawn `--version` per interval.
pub async fn run_watchdog() {
    let mut cli_present: Option<bool> = None;
    let mut last_cli_check = Instant::now();
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
        let healthy = probe_daemon().await;
        set_health(healthy);
        if !healthy {
            attempt_recovery().await;
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

/// Bounded auto-recovery: restart session daemons and re-bind the relay, with
/// backoff between attempts and a halt after MAX_RESTART_ATTEMPTS failures.
async fn attempt_recovery() {
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
    // Restart session daemons (next command spawns a fresh one), then re-bind
    // the extension relay in case it dropped.
    let _ = run_cli(&["daemon", "restart"]).await;
    let _ = run_cli(&["reconnect"]).await;

    let healthy = probe_daemon().await;
    set_health(healthy);
    if healthy {
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
        set_health(false);
        assert!(!is_advertised());
        assert!(!is_available().await);
        // Recovered: fresh healthy state → advertised and available.
        set_health(true);
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
        set_health(true);
        {
            let h = health().lock().unwrap_poison();
            assert_eq!(h.restart_attempts, 0);
            assert_eq!(h.next_restart_at, None);
            assert!(!h.halted);
            assert!(h.halted_until.is_none());
        }
        // Restore pristine state like the sibling tests.
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
}
