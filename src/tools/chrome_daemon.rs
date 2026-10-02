//! chrome-use daemon health monitoring and bounded auto-recovery.
//!
//! The chrome-use CLI talks to a per-session background daemon that drives
//! Chrome through the extension relay. When the daemon or relay dies, CLI
//! commands hang inside its own 5-retry loop (~152 s) instead of failing fast.
//! This module classifies health from a daemon-free `status` snapshot
//! (extension disabled, relay down, host missing, …) and auto-restarts the
//! daemon with bounded backoff and thrash protection. Real wedge detection is
//! per-call: a chrome command that fails with the daemon-unavailable signature
//! marks the daemon unhealthy and wakes the watchdog, which recovers from that
//! stored classification — the daemon-free status cannot see a wedged daemon,
//! so the watchdog never re-evaluates over a fail-fast classification. It also
//! owns the chrome-use CLI invocation primitives (binary name, `--version`
//! check) so the chrome tool depends on this module and not vice versa; the
//! shared command env/setup lives in [`crate::chrome::spawn`].
//!
//! mahbot drives the user's real, logged-in Chrome through the chrome-use
//! extension relay — no chrome-use `--launch` mode, no profile copies, no CDP
//! fallbacks. When the relay is down because Chrome is closed, the watchdog
//! silently auto-launches the user's real Chrome (approved product decision; a
//! window may appear unasked — the launch inherits the user's real profile/
//! environment) under its own bounded launch budget, separate from daemon
//! restarts. An absent extension is unfixable by any launch or restart and
//! pauses auto-recovery.
//!
//! Verified tab-sweep: mahbot-owned session tab groups (`link-enricher-*`) are
//! closed through the CLI and verified by round-over-round re-enumeration. A
//! bare `session stop` cannot settle them: a group can hold tabs the session
//! ADOPTED, which stop never closes and never reports, and this file's own stop
//! (`stop_session_daemon`) is bounded by `CLI_TIMEOUT` — which expires as the
//! daemon's shutdown grace does, so the reclaim that follows never runs and a
//! scratch tab it misses is orphaned forever (no other mechanism ever reclaims
//! it). What a stop proves and what it costs is stated by the live-verified
//! behaviours below.
//!
//! ## Readiness and the pre-action gate
//!
//! [`readiness`] reports what the product ESTABLISHED about the connection to
//! the owner's real browser, from its own daemon-free snapshots (`status --json`
//! and `browsers --json`) plus the Chrome-process and display facts — every fact
//! tri-state, and a missing fact is never defaulted to "fine". The display fact is
//! what chrome-use needs to LAUNCH the owner's Chrome: its absence is reported,
//! but it never blocks an action, because a browser already running is still
//! reachable through the relay. It is deliberately NOT a file/process health
//! check: a running Chrome process is one fact among the others, never a verdict.
//! A snapshot is read by two predicates and one gate, and they are deliberately
//! different:
//!
//! - [`Readiness::ready_for_actions`] is the VERDICT and the health claim —
//!   [`Readiness::outcome`] reports `Healthy` for it and nothing else. It is true
//!   only when every decisive fact was ESTABLISHED and positive: the native host
//!   installed AND healthy, the relay up, and a real browser reachable through it
//!   — the fact that a page action would go to the owner's own browser. A fact
//!   nobody answered leaves it false, never health.
//! - [`Readiness::blocked`] is the GATE's blocking condition: a fact ESTABLISHED
//!   that the connection cannot work ([`Readiness::down_cause`] is `Some`). A
//!   fact a probe never obtained blocks nothing.
//! - [`ensure_ready_for_actions`] is the pre-action gate, cache-backed so a burst
//!   of calls does not re-probe: PROVEN dispatches; an ESTABLISHED cause runs ONE
//!   bounded cause-aware recovery pass (the same machinery and budgets the
//!   watchdog uses), re-probes once, and refuses plainly only while a cause still
//!   stands; anything else — not proven, nothing ruled out — dispatches too,
//!   because refusing every action on a reporting gap would turn it into total
//!   unavailability of both chrome surfaces. An own-browser fallback that slips
//!   through is a plain failure, never a quiet success.
//!
//! That gate, plus [`crate::chrome::spawn::pin_real_browser_env`] (which removes
//! every environment switch that could divert a call onto a throwaway browser)
//! and chrome-use's own browser-replacement note (see
//! [`crate::chrome::contract::self_launched_browser_error`]), is what keeps the
//! tool's own-browser fallback from taking the work.
//!
//! [`ProbeOutcome::Unknown`] is the third health outcome: neither health nor a
//! classified down cause. An unavailable `status` snapshot, a snapshot that
//! carries no extension data at all, and a decisive fact the probe never obtained
//! (a `browsers` list that did not answer) all produce it — the product could not
//! establish anything, so nothing may be reported as fine, and no recovery
//! action may be driven by it: a daemon restart destroys session state (open tabs
//! included), so it may only ever be spent on a cause a fact established. It
//! fails [`ProbeOutcome::is_healthy`] and is never a cause a recovery may act on,
//! while advertisement stays optimistic — advertisement is not a health claim.
//!
//! ## Recovery on every path
//!
//! A session that stops answering is RECOVERED on every path, never merely
//! hinted at: the interactive tool's fail-fast and timeout paths, the `mahbot
//! chrome` CLI's action verbs and their timeout path, and the daemon-side session
//! calls. [`recover_unresponsive_session`] records the
//! wedge under the same `DaemonWedge` classification the fail-fast path uses —
//! which wakes the watchdog, so the BACKGROUND health path recovers too — and
//! stops that session's daemon within the product's own bound, so the next call
//! gets a clean one. Every recovery is bounded and thrash-proof (the existing
//! backoff/halt budgets apply), and a recovery that fails never turns a call
//! that succeeded into a failure. Recovery acts only on a cause a fact
//! ESTABLISHED, never on "the probe found out nothing": [`ProbeOutcome::Unknown`]
//! is re-probed, and a restart — the destructive lever — is spent only where a
//! fact established that it is needed. An unreachable tab stays its own unfixable
//! state: the daemon and relay are up and only the session's tab is orphaned, so
//! nothing is ever recovered for it.
//!
//! Trade-offs:
//! - A genuine daemon wedge surfaces on the first real chrome call, which runs
//!   to the product's own bound (the clock mahbot declares to chrome-use plus the
//!   relay recovery window) before the bounded wedge probe classifies it and wakes the
//!   watchdog. That is the deliberate price of never cutting a call off before the
//!   tool's own clock; the restart then clears the wedge.
//! - `daemon restart` destroys all session state; recovery guidance notes that
//!   existing chrome sessions are reset.
//! - The Chrome auto-launch shares the user's profile and environment, so it may
//!   open a window unasked; on display-less hosts it is paused rather than spend
//!   the launch budget (launching can never help there).

use crate::chrome::contract::{
    ChromeResponse, is_daemon_unavailable_error, is_relay_unavailable_error,
    is_session_unresponsive_error, is_unreachable_tab_error, parse_first,
};
use crate::chrome::spawn::{CliOutput, CliRun, CliSpawn, CliTimeout, ensure_chrome_env, spawn_cli};
use crate::chrome::{CliRecovery, probe_clocks};
use crate::util::UnwrapPoison;
use serde_json::Value;
use std::fmt::Write as _;
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
/// Consecutive failed Chrome auto-launches before the launch budget halts —
/// independent of the daemon-restart budget (a closed browser is a different
/// problem than a wedged daemon; fixing one must not consume the other's).
const MAX_LAUNCH_ATTEMPTS: u32 = 3;
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
/// Backoff between Chrome auto-launch attempts (30s → 2min → 10min), mirroring
/// the restart backoff so a host that keeps failing cannot spam launches.
const LAUNCH_BACKOFF: [Duration; 3] = [
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
/// Bound on how long the session-level recovery's CALLER waits for
/// `session stop`. A stop's whole legitimate cost is the ≈28 s the live-verified
/// behaviours of this module's doc state: the session daemon's own shutdown grace
/// plus the tab reclaim that follows it and closes the session's tab group. 45 s
/// covers that end to end with headroom, so a stop that works confirms here
/// instead of being reported as unanswered while the reclaim it was for is cut
/// off. The stop itself is never killed ([`session_stop_via`]) — only a caller's
/// patience is bounded, and every caller spends this bound as it is. A sweep that
/// recovers a wedged session
/// spends this whole bound, so such a sweep deliberately overshoots
/// [`SWEEP_TOTAL_BUDGET`]: an unrecovered wedge is worse than the overrun, and
/// the sweep is best-effort and self-healing (leftovers are retried by the next
/// sweep or startup).
pub(crate) const SESSION_RECOVERY_TIMEOUT: Duration = Duration::from_secs(45);

// ── Verified-close sweep bounds (pinned for deterministic recovery) ──────
/// Total budget for one sweep invocation, starting before the service-state
/// skip gate. The sweep checks the deadline before each of its own CLI calls, so
/// an in-flight call overshoots by at most that call's own bound
/// ([`CLI_TIMEOUT`]) plus its post-exit collection — one bounded drain window
/// covering both pipes together, which sits outside the per-call kill (see
/// [`crate::chrome::spawn`]); the one deliberate exception is recovering a
/// wedged session, which spends up to [`SESSION_RECOVERY_TIMEOUT`] instead (and
/// its own collection). On expiry the sweep defers: leftover tabs are retried by
/// the next sweep/startup (self-healing), never a permanent orphan.
const SWEEP_TOTAL_BUDGET: Duration = Duration::from_secs(15);
/// Convergence rounds before a sweep gives up for this invocation. The deadline
/// decides when the sweep stops; this only bounds the number of
/// enumerate/close/stop cycles (a healthy host converges in 3 rounds; a retried
/// failed close needs 4–5).
const SWEEP_MAX_ROUNDS: u32 = 5;

/// Classified cause for a failed health check. Drives cause-specific records
/// and decides whether auto-recovery can help at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeFailure {
    /// chrome-use extension or native host is not installed.
    NotInstalled,
    /// Native host manifest present but launcher/target broken.
    HostBroken,
    /// Extension installed but disabled (Chrome reports disable reasons).
    ExtensionDisabled,
    /// The chrome-use extension is absent from Chrome entirely — unfixable by
    /// any daemon restart or launch; the user must install it from the store.
    ExtensionAbsent,
    /// Extension enabled but the relay is down — transient, self-heals.
    RelayDown,
    /// No Chrome/Chromium-family browser process is running. NOT unfixable —
    /// auto-recovery launches the user's real Chrome (never a daemon restart).
    ChromeNotRunning,
    /// The session's tab lost its debugger attach (orphaned) — the daemon and
    /// relay are up and only that session's tab is orphaned. The tab belongs to
    /// the session: the product closes the session's tabs itself at run end (and
    /// when the session is stopped/recovered), so nothing is left for the user;
    /// the session cannot drive that tab again.
    UnreachableTab,
    /// The daemon socket hung or errored (daemon-side wedge).
    DaemonWedge,
}

/// Result of a health evaluation: healthy, down with a classified cause, or
/// unknown — the product could not establish anything decisive (an unavailable
/// `status` snapshot, one that carries no extension data, or a decisive fact the
/// probe never obtained). `Unknown` is deliberately neither health nor a down
/// cause: it fails [`ProbeOutcome::is_healthy`], it carries no invented cause (and
/// so never drives a recovery action — see [`ProbeOutcome::failure`]), and
/// advertisement stays optimistic (converting "could not establish" into a hidden
/// tool would be a claim the probe never made).
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

    /// The cause a fact ESTABLISHED, or `None` when none was. This is the input
    /// the recovery gate reads, so `Unknown` — the probe found out nothing —
    /// yields `None`: a daemon restart DESTROYS session state (open tabs
    /// included) and may only ever be spent on a cause a fact established, never
    /// on "could not find out". An unestablished state is re-probed instead.
    fn failure(self) -> Option<ProbeFailure> {
        match self {
            ProbeOutcome::Healthy | ProbeOutcome::Unknown => None,
            ProbeOutcome::Down(f) => Some(f),
        }
    }
}

impl ProbeFailure {
    /// Causes a daemon restart cannot fix — reported with their concrete fix
    /// and never consume restart attempts. `ChromeNotRunning` is deliberately
    /// NOT here: a closed browser is fixed by launching it, not by restarting
    /// the daemon.
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

/// Outcome of a Chrome auto-launch attempt, for honest down messaging about
/// what the watchdog actually did (never attempted = `None` on [`DaemonHealth`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChromeLaunchOutcome {
    /// Chrome was spawned (and the relay is still down afterward).
    Launched,
    /// No Chrome/Chromium binary found, or it could not be started.
    Failed,
    /// No display session on this host — launching would never help.
    NoDisplay,
}

/// One bounded attempt budget (backoff between attempts, halt + cooldown after
/// `max` failures). Pure state — the bounded state machine is unit-testable.
#[derive(Default)]
struct AttemptBudget {
    attempts: u32,
    next_at: Option<Instant>,
    halted: bool,
    halted_until: Option<Instant>,
}

impl AttemptBudget {
    /// Decide whether an attempt is allowed, updating the bookkeeping in place.
    /// Backoff is checked before the attempt cap, so the final long grace after
    /// the last attempt is honored before the halt fires. Cooldown expiry resets
    /// and opens a fresh bounded cycle.
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
    /// Last evaluated verdict: `None` before any probe (advertised
    /// optimistically), then Healthy, a classified Down cause, or Unknown —
    /// "the product could not establish anything". Unknown carries no cause, so
    /// it is neither health nor a down message that names an invented cause.
    verdict: Option<ProbeOutcome>,
    last_probe: Option<Instant>,
    /// The readiness snapshot behind the verdict with the moment it was taken —
    /// the ONE record [`probe_and_record`] writes, so the gate's
    /// [`HEALTH_TTL`]-bounded cache and the snapshot [`daemon_down_message`]
    /// reports on always come from the same probe and no lock order is needed
    /// between them.
    readiness: Option<(Instant, Readiness)>,
    /// Restart bounded cycle — backoff between attempts, halt + cooldown after
    /// [`MAX_RESTART_ATTEMPTS`] failures (thrash protection).
    restart_budget: AttemptBudget,
    /// Chrome auto-launch bounded cycle — separate from the restart budget (a
    /// closed browser and a wedged daemon are independent failures, so neither
    /// should consume the other's bounded cycle).
    launch_budget: AttemptBudget,
    /// Outcome of the last Chrome auto-launch — `None` when never attempted or
    /// superseded by health. Failure outcomes survive until health or the
    /// sustained-health reset clears them, so down messaging stays honest.
    launch_outcome: Option<ChromeLaunchOutcome>,
    /// The failure cause the last transition-based record named — reset on
    /// recovery so the same cause is reported again after a healthy spell.
    last_cause_reported: Option<ProbeFailure>,
    /// Start of the current sustained-healthy streak — the restart budget
    /// resets only once this reaches [`SUSTAINED_HEALTHY_WINDOW`]; any failure
    /// aborts the streak.
    healthy_since: Option<Instant>,
}

/// Decision from [`DaemonHealth::gate_restart`] / [`DaemonHealth::gate_launch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecoveryGate {
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
    /// Decide whether a daemon restart attempt is allowed.
    fn gate_restart(&mut self, now: Instant) -> RecoveryGate {
        self.restart_budget
            .gate(MAX_RESTART_ATTEMPTS, &RESTART_BACKOFF, HALT_COOLDOWN, now)
    }

    /// Decide whether a Chrome launch attempt is allowed — a separate bounded
    /// budget from the restart one (a closed browser is not a wedged daemon).
    fn gate_launch(&mut self, now: Instant) -> RecoveryGate {
        self.launch_budget
            .gate(MAX_LAUNCH_ATTEMPTS, &LAUNCH_BACKOFF, HALT_COOLDOWN, now)
    }

    /// Apply a health observation. A healthy result opens the sustained-healthy
    /// window ([`SUSTAINED_HEALTHY_WINDOW`]); the restart AND launch budgets
    /// reset only after the window completes, so a transient healthy right after
    /// a restart or launch (the post-recovery verification —
    /// `seed_window = false` — or a single watchdog interval) cannot reopen a
    /// bounded cycle early. Any failure — and any `Unknown`, which is no
    /// evidence of health — aborts the window.
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
            // Any healthy observation supersedes a stale launch record — the
            // relay is back up, so there is nothing honest to report about the
            // last launch. Failure outcomes do NOT clear it (it must survive so
            // the down message stays honest about what happened).
            self.launch_outcome = None;
        } else {
            self.healthy_since = None;
        }
        // A cause change never resets the restart budget — flapping causes (e.g.
        // RelayDown ↔ DaemonWedge) must not evade the attempt halt. Only
        // sustained health (or the cooldown expiry in gate_restart) opens a
        // fresh cycle.
        self.verdict = Some(outcome);
        self.last_probe = Some(now);
    }

    /// The classified cause of the last evaluation, when it was a down one — the
    /// down message and the transition-based records key on this, so an
    /// `Unknown` verdict (which has no cause) reads as "nothing established"
    /// rather than as a cause.
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

/// Record the outcome of a Chrome auto-launch attempt so down messaging stays
/// honest about what the watchdog actually did (survives until health clears it).
fn record_launch_outcome(outcome: ChromeLaunchOutcome) {
    health().lock().unwrap_poison().launch_outcome = Some(outcome);
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

/// Get the platform-appropriate chrome-use binary name.
pub(crate) const fn chrome_bin() -> &'static str {
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
    /// Binary definitively absent (no copy at the location the product installs
    /// the helper to, or the resolved copy vanished).
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
/// install`): a hung registration must not block the background install task
/// indefinitely.
const CHROME_USE_INSTALL_TIMEOUT: Duration = Duration::from_secs(120);

/// Install hint for the definitive not-found case — appended to every
/// user-facing message that names the chrome-use CLI as missing. The helper is
/// brought up to date automatically every time the product starts, so if it is
/// still missing the automatic install failed: check the logs and it will retry
/// on the next start.
pub(crate) const CHROME_USE_INSTALL_HINT: &str = "It is brought up to date automatically every \
     time the product starts — if it is still missing the automatic install \
     failed; check the logs and it will retry on the next start.";

/// Resolved absolute path of the chrome-use binary (the product's own copy
/// only, at [`crate::util::managed_bin::chrome_use_bin_path`]), cached after the
/// first probe. Re-resolves only when the cached path vanished or was never
/// found, so a late installation is picked up by the next probe.
static CLI_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();

/// Absolute path of the chrome-use binary, or `None` when definitively not
/// installed. Spawns must go through this (not the bare name): the product
/// resolves its own copy itself, so the search path and anything else on it can
/// never change which helper a spawn runs.
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
/// probe — an install lands at a path the resolver looks up fresh, which a
/// stale cache would not see.
fn invalidate_cli_path() {
    *CLI_PATH
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_poison() = None;
}

/// Locate the product's own chrome-use copy: the one at
/// [`crate::util::managed_bin::chrome_use_bin_path`] and nothing else. The search
/// path is never consulted, so a leftover copy in the product's old private tools
/// folder can never be preferred over the maintained one — and where that directory
/// is one the owner's own search path does not already hold, [`crate::util::owner_path`]
/// is what puts it there.
fn find_cli_binary() -> Option<PathBuf> {
    let candidate = crate::util::managed_bin::chrome_use_bin_path(chrome_bin())?;
    crate::util::is_executable(&candidate).then_some(candidate)
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
/// extra lines (bug-report URL etc.), so scan every whitespace token for the
/// first parseable semver rather than assuming a position; the version may be
/// spelled `v1.5.99` or bare.
fn parse_cli_version(stdout: &str) -> Option<semver::Version> {
    stdout
        .split_whitespace()
        .find_map(crate::util::managed_bin::parse_version_token)
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
        // Windows on ARM takes the ordinary x64 asset: the vendor publishes no ARM
        // Windows build, and its own `install.ps1` says exactly that in the section
        // that picks the release asset (`install.ps1:151-159` at the `v1.5.140` tag —
        // only x64 is published, and ARM64 takes `chrome-use-win32-x64` with a note
        // rather than a refusal) — so that build, which runs under x64 emulation, is
        // what both the first install and every later update must take.
        ("windows", "x86_64" | "aarch64", _) => "win32-x64",
        _ => return None,
    };
    Some(asset.to_string())
}

/// Asset platform tag for THIS build, or `Err` naming the unsupported
/// platform. Uses the compile-time target triple (`cfg!`) so it is correct
/// regardless of the runtime host; on Linux, musl is detected by the presence
/// of a musl loader in `/lib` (mirrors the vendor installer — no `ldd`).
fn release_asset_name() -> Result<String, String> {
    let (os, arch) = crate::util::managed_bin::host_os_arch()?;
    let musl = os == "linux" && crate::util::managed_bin::linux_host_is_musl();
    release_asset_platform(os, arch, musl)
        .ok_or_else(|| format!("chrome-use has no release asset for {os}-{arch}"))
}

/// Parse the local chrome-use version from `--version` stdout, bounded by
/// [`CLI_TIMEOUT`]. `None` when the CLI is missing, the probe fails (timeout or
/// non-zero exit), or the banner holds no parseable semver — the caller reports
/// that state rather than assuming the CLI is outdated.
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

/// A bounded chrome-use CLI call through [`crate::chrome::spawn::spawn_cli`]
/// (shared env, `--json`, optional `--session`) against an already-resolved binary,
/// so a caller that resolves the path itself does not pay for a second probe.
/// `timeout` is the caller's own bound: the shared [`CLI_TIMEOUT`] for everything
/// that must fail fast, a longer one for a call that waits on the CLI's own cleanup.
/// Its clocks are [`probe_clocks`] — the ONE formula for a product probe: `timeout`
/// is the kill, and that same bound (capped at the longest declaration mahbot makes
/// for a verb it forwards no `--timeout` to) is what chrome-use is given. Every call
/// here is a product-owned probe rather than agent work (`status`, `extension
/// status`, a daemon probe, a lifecycle stop), so chrome-use's relay self-heal is
/// suppressed and its kill is armed once the child is spawned — before that child
/// has booted far enough to apply the clock declared to it — so what ends a probe
/// that stays silent is the product's own bound, which is the "no answer at all"
/// reading the wedge checks are built on. The raw [`CliRun`] is handed back so a
/// caller that turns on the difference between "nothing was asked" and "no answer"
/// can tell them apart.
async fn run_cli_bounded_at(
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
        timeout: CliTimeout::Bounded(clocks.kill),
        cancel_kills: true,
        input: None,
        chrome_side: clocks.chrome_side,
        recovery: CliRecovery::Suppressed,
    })
    .await
}

/// The probed-path variant of [`run_cli_json_at`], bounded by [`CLI_TIMEOUT`],
/// keeping the `Option<String>` error vocabulary its callers already speak: the
/// verdict chrome-use itself reported, or `None` for every leg that carries no
/// text.
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

/// Why a bounded `--json` call carried no verdict — the distinction every caller
/// turns on: a call that never produced a child of ours was never asked anything,
/// while one killed at the product's own bound stayed silent.
#[derive(Debug)]
pub(crate) enum NoVerdict {
    /// chrome-use answered with a structured error — its own text, for a caller
    /// that detects a signature in it.
    Reported(String),
    /// The child was killed at the product's own bound: it did not answer in time.
    TimedOut,
    /// No chrome-use answer came from a child of ours: it could not be spawned, or
    /// its exit could not be waited on — the two legs [`CliRun`] folds into
    /// `SpawnFailure` — so nothing was asked of the session.
    SpawnFailure,
    /// The child's answer carried no readable verdict: bytes the envelope contract
    /// cannot parse, a failure envelope that named no error, or an envelope whose
    /// verdict is not `true` and which named no error either.
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
/// success, else the [`NoVerdict`] leg that says why there is none (the reported
/// message survives for signature detection). The path-and-timeout-taking entry
/// point, for a caller that has resolved the binary itself and knows what its own
/// bound must be.
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
/// `success: true`; otherwise `Err(Some(msg))` for the error chrome-use reported,
/// and `Err(None)` when it named no error — a failure envelope with an empty error,
/// or bytes carrying no readable verdict. The reading itself is [`envelope_outcome`],
/// the same one every other chrome-use path uses.
fn json_outcome(out: &CliOutput) -> Result<Value, Option<String>> {
    envelope_outcome(out.status.success(), &out.stdout)
}

/// The verdict from the child's own exit status and the bytes it wrote — pure so
/// tests pin the reading. The parse is the shared tolerant one
/// ([`crate::chrome::contract::parse_first`]) — the SAME reading every other
/// chrome-use path uses — so bytes a process the command left behind wrote into
/// the call's output channel cannot turn an answered call into silence: a
/// readiness fact would fall to `None` (and the report to "nothing established")
/// and a `session stop` that worked would be counted as unanswered.
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

// ── Readiness ─────────────────────────────────────────────────────────
/// What the product established about the connection to the owner's real browser
/// at one moment, from its own daemon-free snapshots (`status --json` and
/// `browsers --json`) plus the Chrome-process fact.
///
/// Every fact is tri-state and a missing one is NEVER defaulted to "fine": the
/// report and the refusal built from this must not claim a connection that was
/// not proved, because an action that runs without one is work that went to a
/// browser chrome-use launched itself (reported as a plain failure, never a quiet
/// success). This is deliberately not a file/process health check —
/// [`ready_for_actions`](Self::ready_for_actions) is true only when the native
/// host is installed AND healthy, the relay is up, and a real browser profile is
/// reachable through it. The display fact is a REPORTING one, not decisive: it is
/// what chrome-use needs to launch the owner's Chrome, and its absence never
/// blocks an action.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Readiness {
    /// Whether `status --json` answered at all.
    pub(crate) answered: bool,
    /// The tool's own version, when the status carried one.
    pub(crate) cli_version: Option<String>,
    /// The native host is installed (the status's `extension.hostInstalled`).
    pub(crate) host_installed: Option<bool>,
    /// The native host is healthy (the status's `extension.hostHealthy`).
    pub(crate) host_healthy: Option<bool>,
    /// Extension presence — probed (as the health evaluation does) only when the
    /// relay is not up, since that is the one verdict it changes.
    pub(crate) extension: Option<ExtensionState>,
    /// The extension relay is up (the status's `extension.relayUp`).
    pub(crate) relay_up: Option<bool>,
    /// Email of the real profile being driven, when the relay reported one.
    pub(crate) profile_email: Option<String>,
    /// Id of the real profile being driven, when the relay reported one.
    pub(crate) profile_id: Option<String>,
    /// A real Chrome profile is reachable through the relay: the `browsers`
    /// snapshot answered with a non-empty list. This — not a file or a process
    /// being present — is the fact a page action needs.
    pub(crate) real_browser: Option<bool>,
    /// A Chrome/Chromium-family process is running — ONE fact among the others,
    /// never a health verdict (a running process says nothing about whether the
    /// relay can reach the owner's real browser).
    pub(crate) chrome_running: Option<bool>,
    /// This host has a usable display session ([`display_available`]) — what
    /// chrome-use needs to LAUNCH the owner's Chrome. A `Some(false)` host is
    /// reported as such, but its absence blocks nothing: a browser already running
    /// is still reachable through the relay.
    pub(crate) display: Option<bool>,
}

impl Readiness {
    /// Whether the connection to the owner's own browser is PROVEN: every
    /// DECISIVE fact was ESTABLISHED and positive — the native host is installed
    /// and healthy, the relay is up, and a real browser profile is reachable
    /// through it. This is the VERDICT and the health claim:
    /// [`Self::outcome`] reports `Healthy` for this predicate and for nothing
    /// else, so a decisive fact nobody answered — a shape or subcommand change in
    /// a future chrome-use leaves the decisive keys missing — is never health.
    ///
    /// It is deliberately NOT the gate's rule. The gate asks whether a fact
    /// ESTABLISHED that the connection cannot work ([`Self::blocked`]) and
    /// dispatches whatever is neither proven nor blocked: refusing every action on
    /// a merely unanswered fact would turn a reporting gap into total
    /// unavailability of both chrome surfaces. An action let through without proof
    /// either reaches the real browser or drives a browser chrome-use launched
    /// itself, which is reported as a plain failure, never a quiet success
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
    /// real Chrome profiles, `Some(false)` for an empty one, and `None` when the
    /// snapshot did not answer or carries no list at all (a missing fact is never
    /// "reachable").
    fn real_browser_from(browsers: Option<&Value>) -> Option<bool> {
        let browsers = browsers?;
        let list = browsers
            .get("data")
            .and_then(|d| d.get("browsers"))
            .or_else(|| browsers.get("browsers"))?
            .as_array()?;
        Some(!list.is_empty())
    }

    /// The cause a fact ESTABLISHES, or `None` when no fact established anything
    /// wrong (an unanswered snapshot, a missing `browsers` list, a relay the
    /// snapshot never reported on). `None` is deliberately not a cause: nothing
    /// may be invented from a fact the probe never obtained.
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

    /// Whether a fact ESTABLISHED that the connection cannot work — the GATE's
    /// blocking condition, and deliberately narrower than "not proven": it is
    /// [`Self::down_cause`]'s answer, so an unanswered snapshot neither blocks an
    /// action nor spends a restart.
    #[must_use]
    pub(crate) fn blocked(&self) -> bool {
        self.down_cause().is_some()
    }

    /// The health evaluation this snapshot implies, by the one rule: `Healthy`
    /// only for a PROVEN connection ([`Self::ready_for_actions`]), a classified
    /// down cause when a fact established one, and `Unknown` otherwise — never
    /// health for a fact nobody answered.
    fn outcome(&self) -> ProbeOutcome {
        if self.ready_for_actions() {
            ProbeOutcome::Healthy
        } else {
            self.down_cause()
                .map_or(ProbeOutcome::Unknown, ProbeOutcome::Down)
        }
    }

    /// The report's three lists, from ONE derivation so the report and the
    /// refusal can never disagree: the facts that were ESTABLISHED, those
    /// established as NOT holding, and those never established at all. An
    /// established negative is a fact like any other — it belongs with what the
    /// product knows, not with what it failed to find out.
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

    /// The factual report: what was established, what was established as not
    /// holding, what was never established, and the verdict. It never says a
    /// connection is fine when an action cannot be carried out, and it never
    /// presents a file/process presence as health. The verdict has three states —
    /// PROVEN ready for actions ([`Self::ready_for_actions`]), NOT ready because a
    /// fact ruled the connection out ([`Self::down_cause`]), and not proven with
    /// nothing ruled out, which names the facts that could not be checked and says
    /// that an action would still be attempted.
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

    /// The plain refusal for a snapshot whose connection a fact ruled out: what
    /// is missing and what would have happened otherwise. Lists the established
    /// negatives AND the unestablished facts — both are reasons the connection
    /// cannot be established.
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

/// Push a tri-state fact onto the established, the established-as-NOT-holding, or
/// the not-established list: an explicit `Some(false)` is an ESTABLISHED negative
/// (it belongs with what the product knows, not with what it failed to find
/// out), and only `None` means the probe established nothing.
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

/// Probe readiness now (uncached) from the product's own daemon-free snapshots:
/// `status --json` carries the version, the native-host facts, the relay state
/// and the profile being driven, and `browsers --json` carries the connected
/// real Chrome profiles whose non-empty list is the fact that the owner's real,
/// logged-in browser is reachable through the relay. The extension probe runs
/// only when the relay is not up (the one verdict it changes — the health
/// evaluation classifies the same way), the Chrome-process probe and the display
/// fact are one fact each among the others. A snapshot that does not answer
/// leaves its facts `None`: the tool answering with a structured ERROR is still
/// an answer (a bad status is not silence), so `answered` stays true while the
/// facts stay unknown.
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
/// burst of per-call gates from re-probing, on the same window as a healthy
/// health verdict ([`HEALTH_TTL`]). It lives in [`DaemonHealth`], beside the
/// verdict derived from it, so one probe is one record.
fn cached_readiness() -> Option<Readiness> {
    health()
        .lock()
        .unwrap_poison()
        .readiness
        .as_ref()
        .filter(|(at, _)| at.elapsed() < HEALTH_TTL)
        .map(|(_, readiness)| readiness.clone())
}

/// The readiness snapshot for the current moment: a fresh cached one when there
/// is one, otherwise a fresh probe. The probe feeds the shared health verdict
/// too, so advertisement and the watchdog stay in step with what the probe saw.
pub(crate) async fn readiness() -> Readiness {
    if let Some(cached) = cached_readiness() {
        return cached;
    }
    probe_and_record(true).await
}

/// The last readiness snapshot taken, whatever its age — the snapshot behind the
/// stored health verdict that [`daemon_down_message`] reports on, which the
/// TTL-bounded cache would drop.
fn last_readiness() -> Option<Readiness> {
    health()
        .lock()
        .unwrap_poison()
        .readiness
        .as_ref()
        .map(|(_, readiness)| readiness.clone())
}

/// The ONE place a probe's result is recorded: probe, store the readiness
/// snapshot, and apply the health verdict to the same record — one probe, one
/// mutex, so the snapshot and the verdict can never come from different
/// evaluations or disagree about a cause. `seeding` decides whether a healthy
/// result opens the sustained-healthy window ([`SUSTAINED_HEALTHY_WINDOW`]): a
/// probe of its own is a genuine evaluation and seeds it, while the verification
/// right after a recovery action ([`attempt_chrome_launch`] and the post-restart
/// check in [`attempt_recovery`]) must not. The watchdog is deliberately not
/// woken here: a caller's own recovery already acted, and the next tick
/// re-evaluates.
async fn probe_and_record(seeding: bool) -> Readiness {
    let readiness = probe_readiness().await;
    let outcome = readiness.outcome();
    let now = Instant::now();
    let mut h = health().lock().unwrap_poison();
    h.readiness = Some((now, readiness.clone()));
    h.apply_outcome(outcome, now, seeding);
    readiness
}

/// The pre-action gate. A PROVEN snapshot ([`Readiness::ready_for_actions`])
/// returns without probing — the cache is what makes the gate cheap enough to run
/// on every call. A snapshot that established a cause ([`Readiness::blocked`])
/// instead runs ONE bounded, cause-aware recovery pass (the same machinery and
/// budgets the watchdog uses: a closed browser is launched, a relay drop is waited
/// out, a wedge consumes a restart, an unfixable cause recovers nothing),
/// re-probes, and refuses plainly only while a cause still stands. Anything else —
/// NOT proven, nothing ruled out — dispatches the action: no fact established
/// anything wrong, no destructive restart is justified without one, and refusing
/// every action on a reporting gap would turn it into total unavailability of both
/// chrome surfaces. An own-browser fallback reports itself as a plain failure.
///
/// This gate, plus [`crate::chrome::spawn::pin_real_browser_env`] and the
/// browser-replacement note the frontends report as a plain failure, is what
/// keeps the tool's own-browser fallback from taking the work.
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

/// Service-state snapshot for callers that only need "could this host drive a
/// page at all" (the tab sweep's skip gate): the cause a fact established, or
/// `None`. The cached readiness view — the sweep shares the snapshot with the
/// pre-action gate and the watchdog, so the three cannot disagree about a cause.
async fn service_state() -> Option<ProbeFailure> {
    readiness().await.down_cause()
}

/// Extension presence from `extension status --json` (daemon-free; reads
/// Chrome's Secure Preferences). Pure so parsing is unit-testable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExtensionState {
    Present,
    Disabled,
    Absent,
    Unknown,
}

/// Parse extension presence from a `extension status --json` value. An absent
/// `chromeExtension` (an explicit `null`) means the extension is not installed
/// in Chrome at all — distinct from `NotInstalled` (native host missing) and
/// `Disabled` (installed but disabled). A MISSING `chromeExtension` key on a
/// successful envelope is shape-uncertain (old CLI, or a CLI that omits the key
/// for other reasons) — never claim the unfixable absent cause from it; it
/// fails open through the transient-relay path as `Unknown`. A FAILED status
/// command also yields `Unknown` (see [`extension_state`]). Shared with the
/// run-end release's own door check (`crate::tools::chrome_tabs`), so the
/// absence rule has one home.
pub(crate) fn extension_state_from(status: &Value) -> ExtensionState {
    match status.get("data").and_then(|d| d.get("chromeExtension")) {
        // A missing key on a SUCCESSFUL envelope is shape-uncertain (old CLI, or a
        // CLI that omits the key for other reasons) — never claim the unfixable
        // absent cause from it; it fails open through the transient-relay path.
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

/// Whether the chrome-use extension is present, disabled, or absent, from
/// `extension status --json`. A FAILED status command (spawn error, timeout,
/// structured CLI error) is never "absent" — unknown fails open to the
/// transient-relay path, since a mere CLI hiccup must not be reported as an
/// uninstalled extension.
async fn extension_state() -> ExtensionState {
    run_cli_json_opt(&["extension", "status"], None)
        .await
        .map_or(ExtensionState::Unknown, |v| extension_state_from(&v))
}

/// Relay-down cause with deterministic precedence: unfixable Chrome-side
/// causes first (no recovery action can fix them), then the browser-running
/// probe, then the transient relay drop. `chrome_running == None` means the
/// probe was inconclusive or not applicable — degrade to RelayDown.
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

/// Whether a Chrome/Chromium-family browser process is running. Unix probes
/// `pgrep -x <ERE alternation>` over the known process names in a single spawn
/// (macOS bundle binaries plus Linux `comm` names); Windows probes `tasklist`
/// image-name filters. Any match → `Some(true)`; all no-match → `Some(false)`;
/// spawn error, other exit code, or timeout → `None` (inconclusive → the
/// classifier degrades to RelayDown). A closed browser is recovered by
/// launching, not by restarting the daemon.
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

/// Single `pgrep -x <ERE alternation>` probe: exit 0 → a Chrome/Chromium-family
/// browser process is running, exit 1 → none, any other exit code / spawn error /
/// timeout → inconclusive.
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
// 1.5.100; note the installed copy is replaced with the newest release on every
// product start, and that ≥1.5.101 idle keeps external tabs alive — the explicit
// close/stop the sweep uses still cleans up, so the verified behaviors below are
// unchanged):
// - `tab list --session <name>` enumerates only that session's tab group (the
//   relay scopes `Target.getTargets` per announced group, leeguooooo/chrome-use#40). When the
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
// - `session stop` SIGTERMs the daemon — whose shutdown handler closes its
//   created tabs best-effort through the relay — waits out the daemon's shutdown
//   grace (an upstream chrome-use figure, not a constant of this repo: 8 s on the
//   installed CLI, 1 s before leeguooooo/chrome-use#192), then SIGKILLs. That
//   shutdown close is best-effort and NOT proof of anything. What follows it IS:
//   chrome-use then reconnects to the browser endpoint and reclaims the session's
//   PERSISTED created-tab ownership record under its own 20 s timeout — ≈28 s end to
//   end on the installed CLI, the whole of what one stop may legitimately spend (which
//   is why the ended-run release gives a stop that much of its own pass budget before
//   its stop-by-name fallback, and why a stop it had to cut short there concludes
//   nothing) — dropping a tab's id only when its close was acknowledged, and exits
//   non-zero while the record still holds anything. So a successful stop proves that
//   every tab the session CREATED is gone, and proves nothing for a tab the session
//   ADOPTED (stop never closes adopted tabs) — which is why the sweep verifies by
//   round-over-round re-enumeration rather than by the stop's own exit code.
//
// Residual limits (accepted): the sweep's scratch tab is about:blank and the
// extension refuses to re-attach `about:` URLs (its `eligible()`/`SKIP_URL`
// filter). An orphan that lost its attach while the daemon kept a stale
// binding (relay blip, kill during an outage) fails every command with the
// unreachable-tab signatures — the sweep logs the unreachable-tab signal and
// keeps retrying; live orphans whose attach survived heal automatically. An
// orphan the extension fully dropped (Chrome
// service-worker restart unmarks ineligible about: tabs and never re-attaches
// them; the relay's group is fed only by attach announcements) is invisible
// to every CLI path: `tab list` succeeds with only the fresh scratch and the
// sweep converges to Clean with no log. That case is undetectable by design —
// no CLI path can enumerate a tab the extension no longer announces. An
// `agent-tab-*` orphan of that shape is still reached by the run-end release,
// which enumerates the live tab groups by title instead: it closes what the
// extension owns there and reports what it does not. A link-enricher orphan
// stays in Chrome until closed by hand.
/// Close every tab in a mahbot-owned session's tab group except the sweep's own
/// scratch, verifying closure by round-over-round re-enumeration. Shared by the
/// startup sweep and the link-enricher per-fetch close, which keep to mahbot-owned
/// names — a name that is not one is not ours to touch, so the refusal below is a
/// fail-closed backstop rather than a live path.
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
            "tab sweep skipped — chrome service unavailable"
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

/// Only mahbot-owned session names may be swept — link-enricher-* and ephemeral
/// `mahbot-chrome-ephemeral-*` CLI sessions (orphan protection). Named
/// `mahbot-chrome-<name>` sessions and user/default/agent-tab sessions are
/// never touched (strict-scope rule). The legacy pre-rename
/// `mahbot-browser-ephemeral-*` prefix still matches so orphans left by older
/// builds don't leak.
pub(crate) fn is_mahbot_session_name(name: &str) -> bool {
    name.starts_with("link-enricher-")
        || name.starts_with(crate::chrome::CLI_EPHEMERAL_PREFIX)
        || name.starts_with("mahbot-browser-ephemeral-")
}

/// Causes the sweep warns about — warn once per cause transition so a
/// persistent orphan does not spam every sweep, and warn again after
/// a healthy sweep cleared the previous cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SweepWarn {
    /// Leftover tab the daemon can no longer re-drive (stale binding on an
    /// about:blank tab the extension never re-attaches) — the tab belongs to the
    /// session and is closed by the product when the session's run ends (or the
    /// session itself is stopped), not by hand. Only fires while a command still
    /// errors; an orphan the extension fully dropped is invisible (see the
    /// pinned-behaviors note).
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

/// Map a session-CLI error to its [`SweepWarn`] and return `None` for the
/// caller's `Option<T>` (the error path never yields a value). Generic over
/// the Ok type so both `tab list` (`Vec<SweepTab>`) and `tab new` (`String`)
/// callers compile with the same one-liner; every error emits its warn, so a
/// deferral is never silent.
///
/// A session-unresponsive error is RECOVERED here through the same shared
/// helper every other path uses — the daemon-side session calls hit the same
/// signature, so the background sweep recovers the wedged session instead of
/// only logging it. The sweep then defers (its remaining attempts would hit the
/// wedged daemon again); the next sweep/startup sees the fresh one. This is the
/// one step that may take the sweep past its own [`SWEEP_TOTAL_BUDGET`] — see
/// [`SESSION_RECOVERY_TIMEOUT`] for why that is deliberate.
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

/// Bounded `session stop` — its outcome is deliberately ignored: the sweep
/// proves closure by round-over-round re-enumeration. Skipped when the sweep is
/// already over budget (the daemon idles out on its own and the next sweep
/// retries). The session is named by the helper's `--session` flag alone.
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

// ── Session recovery (every path) ─────────────────────────────────────
/// What the bounded session recovery did, for the caller's error text.
/// [`summary`](Self::summary) is the one sentence that reports it; the enum is
/// what lets a caller that has a manual recovery of its own name that flow ONLY
/// when the automatic one could not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionRecovery {
    /// The session's daemon was stopped within the bound — the next call gets a
    /// fresh one.
    Stopped,
    /// The stop was ISSUED but had not confirmed within the bound — the stop
    /// itself keeps running (it is what reclaims the session's tabs), so only a
    /// session that still does not answer afterwards needs the manual flow.
    Unanswered,
    /// No `session stop` could be STARTED — no binary to run, a spent bound, or a
    /// spawn failure — so no stop was issued and the session is in whatever state
    /// it was found in.
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

/// Record a session wedge under the SAME `DaemonWedge` classification the
/// fail-fast path uses, and wake the watchdog: the background health path then
/// recovers from it too, not just the caller that saw the symptom.
fn note_session_wedge() {
    set_health(ProbeOutcome::Down(ProbeFailure::DaemonWedge));
    wake().notify_one();
}

/// Bounded `session stop` through an already-resolved binary — the session-level
/// recovery the daemon's own unresponsive-session paths share, under a caller's
/// own bound ([`SESSION_RECOVERY_TIMEOUT`] for those paths). A sweep that
/// recovers a wedged session therefore overshoots its own [`SWEEP_TOTAL_BUDGET`] by that
/// bound — deliberately, because an unrecovered wedge is worse (the sweep is
/// best-effort and self-healing).
///
/// The stop is deliberately UNINTERRUPTIBLE: a stop still running keeps running
/// when the bound expires ([`CliSpawn::cancel_kills`] false), because the long
/// tail the bound would otherwise cut off is exactly the tab reclaim that closes
/// the session's tab group. The bound therefore decides only when THIS CALLER
/// stops waiting and reports [`SessionRecovery::Unanswered`] — never whether the
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

/// The recovery stop's envelope verdict: [`run_cli_json_at`]'s judgement of the
/// same `--json` contract, on the same clocks ([`probe_clocks`], the ONE formula
/// for a product probe: `bound` is the kill, and that same bound — capped at the
/// longest declaration mahbot makes for a verb it forwards no `--timeout` to — is
/// what chrome-use is given), but
/// spawned so the call is never killed — the only difference is
/// [`CliSpawn::cancel_kills`], which the fail-fast probes need and this stop must
/// not have (see [`session_stop_via`]). The answer distinguishes a stop that was
/// issued and confirmed from one issued but not confirmed (the child ran) and
/// from one that never started.
async fn session_stop_outcome(path: &Path, session: &str, bound: Duration) -> SessionRecovery {
    let args = ["session", "stop"];
    let clocks = probe_clocks(bound);
    match spawn_cli(CliSpawn {
        path,
        args: &args,
        session: Some(session),
        json: true,
        capture_stderr: false,
        timeout: CliTimeout::Bounded(clocks.kill),
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

/// The shared recovery for a session that stopped answering: record the wedge
/// (which wakes the watchdog, so the BACKGROUND health path recovers too), then
/// stop that session's daemon through the product's own chrome-use within
/// [`SESSION_RECOVERY_TIMEOUT`], so the next call gets a clean one. The wedge is
/// recorded even when no binary can be resolved — the watchdog's own recovery
/// does not depend on this caller's stop, which is then
/// [`SessionRecovery::NotStarted`]. Never turns a successful call into a
/// failure; the caller's error text carries [`SessionRecovery::summary`].
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

/// Sync availability for tool advertisement (never evaluates — uses the last
/// known state). Only a confirmed-down evaluation hides the tool: `Unknown` is
/// advertised optimistically, because advertisement is not a health claim.
pub(crate) fn is_advertised() -> bool {
    !matches!(
        health().lock().unwrap_poison().verdict,
        Some(ProbeOutcome::Down(_))
    )
}

/// Mark the daemon unhealthy immediately (fail-fast path) with the cause the
/// error text points to, and wake the watchdog so recovery starts without
/// waiting for the next interval. Unreachable-tab errors never reach this path
/// — the chrome tool's fail-fast guard bails with unreachable-tab guidance first
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
    let answered = last_readiness().filter(|r| r.answered);
    let h = health().lock().unwrap_poison();
    if h.verdict == Some(ProbeOutcome::Unknown) {
        return unknown_down_message(answered.as_ref());
    }
    let failure = h.failure();
    let cause = match failure {
        Some(ProbeFailure::NotInstalled) => {
            "The chrome-use extension or native host is not installed — the chrome daemon \
             cannot run. Enable the chrome-use extension at chrome://extensions (the CLI is \
             installed at the start of every run of the product); health recovers \
             automatically once it is installed."
        }
        Some(ProbeFailure::HostBroken) => {
            "The chrome-use native host launcher is broken — run `chrome-use doctor` to see why; \
             health recovers automatically once it is fixed."
        }
        Some(ProbeFailure::ExtensionDisabled) => {
            "The chrome-use extension is disabled — enable it at chrome://extensions. Daemon \
             restarts cannot fix a Chrome-side disable; health recovers automatically once \
             it is enabled."
        }
        Some(ProbeFailure::ExtensionAbsent) => {
            "The chrome-use extension is not installed in Chrome — install/enable the ab-connect \
             extension from the Chrome Web Store. Daemon restarts and launches cannot fix an \
             absent extension; health recovers automatically once it is installed."
        }
        Some(ProbeFailure::RelayDown)
            if h.launch_outcome == Some(ChromeLaunchOutcome::Launched) =>
        {
            "The extension relay is down even after Chrome was auto-launched — Chrome is \
             running, but the ab-connect extension is not republishing. If the extension is \
             only installed in a non-default profile, install it in Chrome's default profile."
        }
        Some(ProbeFailure::RelayDown) => {
            "The chrome-use extension relay is down (the extension itself is enabled). \
             Auto-recovery restarts the session daemons and waits for the extension to \
             reconnect."
        }
        Some(ProbeFailure::ChromeNotRunning) => match h.launch_outcome {
            None | Some(ChromeLaunchOutcome::Launched) => "Chrome is not running.",
            Some(ChromeLaunchOutcome::Failed) => {
                "Chrome is not running and the auto-launch attempt failed (binary not found or \
                 could not start) — start Chrome manually."
            }
            Some(ChromeLaunchOutcome::NoDisplay) => {
                "Chrome is not running and this host has no display — start Chrome manually."
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
                " Auto-recovery is paused for this cause — no launch will be attempted; it \
                 resumes automatically once a display is available."
            }
            _ if h.launch_budget.halted => {
                " Auto-recovery exhausted its launch attempts and is in a 30-minute cooldown \
                 (thrash protection); it will retry after the cooldown — start Chrome manually \
                 if Chrome stays down."
            }
            Some(ChromeLaunchOutcome::Failed) => {
                " Auto-recovery will retry the launch with backoff."
            }
            None | Some(ChromeLaunchOutcome::Launched) => {
                " Auto-recovery will launch Chrome, backing off between attempts — no manual \
                 action is needed."
            }
        }
    } else if h.restart_budget.halted {
        " Auto-recovery exhausted its restart attempts and is in a 30-minute cooldown (thrash \
         protection); it will retry after the cooldown."
    } else if failure.is_some_and(ProbeFailure::is_unfixable) {
        " Auto-recovery is paused for this cause — no restart will be attempted; it resumes \
         automatically once the underlying issue is resolved."
    } else {
        " Auto-recovery was triggered and will restart it automatically — no manual action is \
         needed (note: the restart resets chrome sessions)."
    };
    format!(
        "{cause}{recovery} While it's down, use web_search, or shell `curl` for page fetches, \
         instead of the chrome tool."
    )
}

/// The down message for an `Unknown` verdict: not health and not a classified
/// down cause, so nothing is named as one and nothing is restarted. When the
/// tool DID answer (a fact inside its answer was missing), the readiness
/// snapshot's own refusal names what could not be established and is what the
/// caller gets; only a silent tool — no answered snapshot behind the verdict —
/// is reported as not having answered.
fn unknown_down_message(answered: Option<&Readiness>) -> String {
    const TAIL: &str = " No cause was established, so no restart is attempted; re-checking \
                        continues automatically. While it's down, use web_search, or shell `curl` \
                        for page fetches, instead of the chrome tool.";
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
/// [`daemon_down_message`] when the stored verdict classified a failure — the
/// bare fact list says what is missing, and the classified cause names the
/// concrete remedy for it (which an `Unknown` verdict has none of, so nothing is
/// appended then).
#[must_use]
pub(crate) fn refusal_message(refusal: &str) -> String {
    let classified = health().lock().unwrap_poison().failure().is_some();
    if classified {
        format!("{refusal} {}", daemon_down_message())
    } else {
        refusal.to_string()
    }
}

/// Bounded post-timeout health evaluation, deciding what a timed-out chrome
/// call means. `status` is daemon-free and by design cannot see a wedged
/// session daemon (wedges are invisible to it), and the mahbot-side per-call
/// bound cuts the CLI off before its own ~152 s retry loop can surface the
/// daemon-unavailable signature — so after a healthy `status`, the session's
/// daemon itself is probed with a trivial bounded command. A second
/// consecutive hang is the wedge signature: the session is recovered through
/// the shared helper (which records the wedge — waking the watchdog, so the
/// background path recovers too — and stops that session's daemon) and the
/// returned text says what was done. `None` = healthy — the timeout was a slow
/// call, not the daemon.
///
/// An UNESTABLISHED snapshot does not short-circuit this path. A snapshot whose
/// `browsers` (or status) key a future chrome-use stops answering must not
/// silently disable the tool's own wedge recovery on a host whose tool can still
/// drive the real browser: only a cause a fact ESTABLISHED, or a tool that
/// answered nothing at all, leaves nothing to probe through.
///
/// A readiness snapshot that did not answer or is blocked is the whole
/// not-available signal here: the snapshot and the stored verdict come from one
/// record, so a further availability check could only repeat it, and
/// [`daemon_down_message`] is returned directly.
pub(crate) async fn health_after_call_timeout(session: &str) -> Option<String> {
    let readiness = readiness().await;
    if !readiness.answered || readiness.blocked() {
        return Some(daemon_down_message());
    }
    // The probe is bounded, and silence is the wedge signature it reads: the
    // mahbot-side bound cuts the CLI off before its own ~152 s retry loop can
    // surface the daemon-unavailable text, so no answer from any child of ours —
    // none spawned for it, or killed at that bound — means the session's daemon
    // stopped answering. Here a host that can spawn no chrome-use at all also has no
    // browser to drive, so the daemon guidance is what the caller needs either way.
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
        // evaluation is skipped: a CLI-less host cannot run commands, and an
        // evaluation without a CLI would only record Unknown for a binary that is
        // not there.
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
                                "chrome-use CLI not found — chrome daemon watchdog standing down"
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
            // from a service-level failure it finds — and only from one: an
            // `Unknown` verdict established nothing, and nothing is what a restart
            // may be spent on.
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
        .map_err(|_| "the download client could not be built".to_string())?;

    // Fetch the `.sha256` sidecar with the same client; a missing/unreadable/
    // mismatching sidecar is a hard failure so a tampered or partial release is
    // never installed.
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

/// Bring the product's own copy of chrome-use to the newest release.
///
/// The release is always fetched and whatever sits at the destination is replaced
/// — a copy the owner installed himself, one a package manager put there, or the
/// product's own from the previous start. No version is compared and nothing about
/// the existing copy is consulted. `Err` names the failing step with no local path
/// and no value read from the owner or his environment (see
/// [`crate::util::managed_bin::install_on_start`]); the copy already at the
/// destination is never removed or invalidated by a failure here, so the machine
/// is never left without a working helper.
///
/// The native-host registration is refreshed whether or not the swap landed: it
/// binds the helper's full path, and that path is the same either way — a copy the
/// product may not replace is still the copy the browser has to launch.
async fn install_chrome_use() -> Result<PathBuf, String> {
    // The location the helper goes to, named once here — before anything is fetched,
    // so an unresolvable home costs no download: the registration binds the same path
    // even when the swap does not land, because the copy the browser has to launch is
    // the one that location names either way.
    let dest = crate::util::managed_bin::chrome_use_bin_path(chrome_bin()).ok_or_else(|| {
        "the helper's install directory is unavailable (the owner's home could not be resolved)"
            .to_string()
    })?;
    let tag = crate::util::managed_bin::fetch_latest_tag(
        CHROME_USE_RELEASE_REPO,
        CHROME_USE_RELEASE_TIMEOUT,
    )
    .await?;
    let (_temp, fresh) = download_chrome_use_binary(&tag).await?;
    let placed = crate::util::managed_bin::place_extracted(&fresh, &dest);
    // The copy belongs at the location the resolver looks in; clear any cached path
    // so the next probe re-resolves it.
    invalidate_cli_path();
    let registered = register_native_host(&dest).await;
    // The first failing step's reason is the one reported — the swap's, when it did
    // not land, or the registration's when it did — and a copy left in place is what
    // the caller's own `present()` check reports afterwards.
    placed.and(registered).map(|()| dest)
}

/// Register the freshly placed binary as the native-messaging host.
///
/// Registration binds the helper's full path, so it is redone after every install
/// — the copy is placed on every start, so the registration runs on every start
/// too (a bounded subprocess). A launcher pointing at a copy that is no longer
/// the installed one is the classic silent relay failure. `--no-profile` is
/// REQUIRED on macOS to avoid chrome-use's default of writing and queueing the
/// `ab-connect.mobileconfig` managed-configuration profile
/// (ExtensionInstallForcelist) that flips Chrome into "managed by your
/// organization" mode; mahbot never creates or re-queues that profile in any
/// flow. Supported since chrome-use v1.5.93; the binary is always freshly
/// downloaded so the flag is always available.
///
/// A registration failure never removes the binary: the error says the installed
/// copy is left in place and the next start tries again, because the browser must
/// always launch a copy that exists and works.
async fn register_native_host(dest: &Path) -> Result<(), String> {
    let mut host = Command::new(dest);
    #[cfg(target_os = "windows")]
    host.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    host.args(["extension", "install", "--no-profile"]);
    run_install_step("`chrome-use extension install --no-profile`", host)
        .await
        .map_err(|e| format!("{e}\nThe installed copy is left in place."))
}

/// Run one install subprocess bounded by [`CHROME_USE_INSTALL_TIMEOUT`]. The
/// helper's own output is dropped: it can name the owner's own browser profile,
/// and the failure is recorded verbatim, so only `label` and the exit status (or
/// the spawn error's kind) reach the message.
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

// ── Managed install ───────────────────────────────────────────────────
/// Spawned one-shot task: install the product's own copy of chrome-use straight
/// away when it is missing, else bring it to the newest release once the boot has
/// settled — on every product start, deliberately without a version comparison and
/// without looking at what is already there. The copy in place is what the agents
/// run — which copy the owner's own terminal resolves first is his search path's own
/// order, which the product neither knows nor changes — and the native-host
/// registration is refreshed on every install so the browser launches the helper
/// that is actually installed. Failures are non-fatal, recorded at WARN (the level
/// the product's own issues view shows) and retried on the next start; every level
/// and the shape itself live in [`crate::util::managed_bin::install_on_start`].
pub async fn run_chrome_use_management() {
    crate::util::managed_bin::install_on_start(
        "chrome-use",
        || cli_path().is_some(),
        install_chrome_use(),
    )
    .await;
}

/// One-time cleanup of stale mahbot-owned chrome-session artifacts at watchdog
/// start: leftover link-enricher sessions get swept so their tab groups don't
/// accumulate. Each sweep is verified (round-over-round convergence) and only
/// ever closes the target session's own tabs — sessions owned by other agents
/// or the user (explicit tabs, `default`, any non-mahbot name) are never
/// touched, and neither are run-owned `agent-tab-*` ones, which are not mahbot
/// names at all (`chrome.rs`'s `AGENT_TAB_PREFIX`). Dead link-enricher orphans stay
/// until the tab is closed by hand — a documented residual limit.
async fn cleanup_stale_sessions() {
    let Some(sessions) = registered_sessions().await else {
        return;
    };
    for name in sessions {
        if is_mahbot_session_name(&name) {
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

pub(crate) async fn relay_up() -> Option<bool> {
    let status = run_cli_json(&["status"]).await?;
    status
        .get("data")?
        .get("extension")?
        .get("relayUp")
        .and_then(Value::as_bool)
}

/// Chrome launch flags applied on auto-launch. Deliberately minimal — no
/// daemon/profile/home overrides: the launch inherits the user's real
/// profile/environment, so only flags that suppress first-run friction belong.
const CHROME_LAUNCH_FLAGS: [&str; 3] = [
    "--no-first-run",
    "--no-default-browser-check",
    "--silent-debugger-extension-api",
];

/// Launch the user's real Chrome when [`ProbeFailure::ChromeNotRunning`], within
/// a bounded launch budget separate from daemon restarts. Display-less hosts
/// are unfixable here: paused, no budget consumed. The launch inherits mahbot's
/// real environment (NOT `ensure_chrome_env` — its HOME override and daemon
/// flags must never reach the user's browser) and waits for the relay.
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
    // The verification right after a launch must not seed the sustained-healthy
    // window (same rationale as the restart path) — the launch budget resets only
    // on sustained health.
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
/// macOS/Windows launch is attempted and degrades via the failure path (SSH/
/// service sessions surface there).
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

/// Resolve the user's real Chrome/Chromium-family binary (standard install
/// locations, mirroring chrome-use's own resolution in spirit). None → the
/// launch fails honestly.
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

/// Spawn Chrome independent of mahbot: the child leads its own process group on
/// Unix, so terminal signals to mahbot's group must not kill the browser, and on
/// Windows it gets the same windowless-console flag as every other spawn (inert
/// for this GUI image — see the window guarantee in `tools::shell::tree`). The
/// child is deliberately dropped without wait or kill_on_drop — Chrome must
/// survive mahbot restarts.
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
/// watchdog interval, but the same cause is reported again after a healthy
/// spell. Severity is per cause: the wedge record is informational (the restart
/// it announces is automatic), every other cause is a warning.
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
            "chrome-use extension is not installed in Chrome — install/enable the \
             ab-connect extension from the Chrome Web Store. Auto-recovery paused \
             until it is installed."
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
        // Informational: the restart it announces is automatic and this cause is
        // deduplicated per transition, so a self-cleared wedge should not sit in
        // the issues view — a failed restart and an exhausted budget still do.
        ProbeFailure::DaemonWedge => {
            info!("chrome daemon is unresponsive — restarting it (bounded backoff).");
        }
    }
}

/// Log a blocked recovery-gate decision — the gated action does not run.
/// `noun` names the gated action ("restart" / "chrome launch").
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
/// into a bounded Chrome launch — never a daemon restart, never the restart budget
/// (a closed browser cannot be fixed by restarting the daemon). Restart causes
/// restart session daemons with backoff between attempts and a halt after
/// MAX_RESTART_ATTEMPTS failures. Causes that a restart cannot fix — extension
/// absent/disabled, not installed, broken host, unreachable tab — are reported
/// with their concrete fix and never consume restart attempts. A transient relay
/// drop is waited out first and consumes no attempt if it self-heals; if that
/// wait ends with nothing established, the pass stops there — no restart is
/// justified by "could not find out".
async fn attempt_recovery(mut failure: ProbeFailure) {
    report_cause(failure);
    // Unfixable causes stop here — they never consume restart attempts.
    if failure.is_unfixable() {
        return;
    }
    // ChromeNotRunning is handled by a bounded launch, BEFORE the restart
    // throttle check: the launch budget is independent of restart backoff/halt,
    // so a halted restart state must not block launching the user's browser.
    if failure == ProbeFailure::ChromeNotRunning {
        attempt_chrome_launch().await;
        return;
    }
    // While a recovery timer is pending, the timer IS the wait — don't poll the
    // relay for up to RELAY_REVIVE_WAIT on top of it. The next watchdog cycle
    // re-evaluates and re-enters recovery.
    let now = Instant::now();
    let throttled = health()
        .lock()
        .unwrap_poison()
        .restart_budget
        .is_waiting(now);
    if throttled {
        return;
    }
    // A transient relay drop is waited out before any session-disrupting
    // restart: the MV3 worker republishes on its keepalive (~30 s). A drop
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

    // Decide whether a restart is allowed, and update the attempt bookkeeping,
    // entirely within a scoped lock so the MutexGuard is never held across
    // an await point.
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
    // Restart session daemons (session-less; closes their tabs, relay survives).
    // No `reconnect` — it can cold-restart the user's Chrome or open the Web
    // Store; a persistent relay drop self-heals on the extension's keepalive.
    let _ = run_cli(&["daemon", "restart"]).await;
    if failure == ProbeFailure::RelayDown {
        wait_for_relay(RELAY_REVIVE_WAIT).await;
    }

    // Post-restart verification must not seed the sustained-healthy window — the
    // restart budget resets only after consecutive watchdog intervals of genuine
    // health, so a run that keeps failing cannot reopen a fresh cycle.
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

/// Test-only: put a readiness snapshot behind the stored verdict — or clear it
/// with `None` — so [`daemon_down_message`]'s branches can be exercised without
/// the CLI spawn a real probe needs.
#[cfg(test)]
pub(crate) fn set_readiness_snapshot(readiness: Option<Readiness>) {
    health().lock().unwrap_poison().readiness = readiness.map(|r| (Instant::now(), r));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chrome::contract::is_daemon_unavailable_code;

    /// The readiness/session probes read the tool's answer the same tolerant way
    /// every other path does: bytes a process the command left behind wrote into
    /// the call's channel after chrome-use's own envelope must not turn an
    /// answered call into "no answer" (which would drop readiness facts to `None`
    /// and count a stop that worked as unanswered).
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
        // Unknown is neither health nor a down cause: still advertised
        // (advertisement is not a health claim), and it is never a cause a
        // recovery pass may act on.
        set_health(ProbeOutcome::Unknown);
        assert!(is_advertised());
        assert!(!ProbeOutcome::Unknown.is_healthy());
        assert_eq!(ProbeOutcome::Unknown.failure(), None);
        // Fresh boot (nothing probed yet) → advertised optimistically.
        reset_health();
        assert!(is_advertised());
    }

    #[tokio::test]
    async fn unknown_is_not_health_but_is_not_a_classified_cause() {
        // The tool did not answer, so nothing was established. That must not sit
        // in the health cache as "fine", must not be reported as a named cause,
        // and must not drive a restart.
        let _guard = with_health_test_lock().await;
        reset_health();
        set_readiness_snapshot(None);
        set_health(ProbeOutcome::Unknown);
        let msg = daemon_down_message();
        assert!(msg.contains("did not answer"), "got: {msg}");
        assert!(msg.contains("nothing could be established"), "got: {msg}");
        // The launch/restart tails belong to classified causes — an Unknown
        // must not promise a cause-specific repair.
        assert!(!msg.contains("relay is down"), "got: {msg}");
        assert!(!msg.contains("not installed"), "got: {msg}");
        // An Unknown verdict has no classified cause, so `refusal_message` adds
        // nothing to the gate's own facts.
        assert_eq!(refusal_message("the facts"), "the facts");

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
        assert!(!msg.contains("did not answer"), "got: {msg}");
        assert!(
            msg.contains("whether the extension relay is up"),
            "got: {msg}"
        );
        assert!(msg.contains("re-checking continues"), "got: {msg}");
        // …and a CLASSIFIED verdict's remedy IS appended to the gate's facts.
        set_health(ProbeOutcome::Down(ProbeFailure::RelayDown));
        let msg = refusal_message("the facts");
        assert!(msg.starts_with("the facts "), "got: {msg}");
        assert!(msg.contains("relay is down"), "got: {msg}");
        set_readiness_snapshot(None);
        reset_health();
    }

    #[tokio::test]
    async fn an_unestablished_snapshot_dispatches_and_spends_no_restart() {
        // The gate's other half: a snapshot that is neither proven nor blocked —
        // the tool answered but a decisive key was missing — is NOT a refusal.
        // The action is dispatched, because a reporting gap must not take both
        // chrome surfaces out of service, and no restart is spent on it: a restart
        // destroys session state, so only a cause a fact established may pay for
        // one. The cached snapshot is served as-is, so no CLI is spawned here.
        let _guard = with_health_test_lock().await;
        reset_health();
        set_readiness_snapshot(Some(Readiness {
            answered: true,
            chrome_running: Some(true),
            display: Some(true),
            ..Readiness::default()
        }));
        assert!(ensure_ready_for_actions().await.is_ok());
        let h = health().lock().unwrap_poison();
        assert_eq!(h.verdict, None);
        assert_eq!(h.restart_budget.attempts, 0);
        drop(h);
        set_readiness_snapshot(None);
        reset_health();
    }

    #[tokio::test]
    async fn daemon_down_message_reflects_launch_state() {
        let _guard = with_health_test_lock().await;
        // Each sub-case starts from the pristine (unknown) state so the launch
        // outcome recorded in one case never leaks into the next.
        reset_health();

        // Plain relay drop — no launch was attempted, so the auto-launch caveat
        // must not leak into the message (it names the relay and the restart).
        set_health(ProbeOutcome::Down(ProbeFailure::RelayDown));
        let msg = daemon_down_message();
        assert!(msg.contains("relay is down"));
        assert!(!msg.contains("auto-launched"));

        // Relay still down after a successful auto-launch — the durable caveat
        // (non-default profile) surfaces instead of the plain relay-drop text.
        reset_health();
        set_health(ProbeOutcome::Down(ProbeFailure::RelayDown));
        record_launch_outcome(ChromeLaunchOutcome::Launched);
        let msg = daemon_down_message();
        assert!(msg.contains("even after Chrome was auto-launched"));
        assert!(msg.contains("non-default profile"));

        // Chrome not running with no recorded launch — the tail promises a
        // launch, honest whether it fires this cycle or after backoff.
        reset_health();
        set_health(ProbeOutcome::Down(ProbeFailure::ChromeNotRunning));
        let msg = daemon_down_message();
        assert!(msg.contains("Chrome is not running."));
        assert!(msg.contains("will launch Chrome"));

        // Chrome not running, last launch failed — the failed-attempt cause and
        // the backoff retry tail both appear (the launch budget is not halted).
        reset_health();
        set_health(ProbeOutcome::Down(ProbeFailure::ChromeNotRunning));
        record_launch_outcome(ChromeLaunchOutcome::Failed);
        let msg = daemon_down_message();
        assert!(msg.contains("auto-launch attempt failed"));
        assert!(msg.contains("start Chrome manually"));
        assert!(msg.contains("retry the launch with backoff"));

        // Same failed launch but the budget is halted (thrash protection) — the
        // cooldown text supersedes the backoff retry promise.
        reset_health();
        set_health(ProbeOutcome::Down(ProbeFailure::ChromeNotRunning));
        record_launch_outcome(ChromeLaunchOutcome::Failed);
        health().lock().unwrap_poison().launch_budget.halted = true;
        let msg = daemon_down_message();
        assert!(msg.contains("exhausted its launch attempts"));
        assert!(!msg.contains("retry the launch with backoff"));

        // Chrome not running because there is no display — recovery is paused
        // for this cause instead of spending the launch budget.
        reset_health();
        set_health(ProbeOutcome::Down(ProbeFailure::ChromeNotRunning));
        record_launch_outcome(ChromeLaunchOutcome::NoDisplay);
        let msg = daemon_down_message();
        assert!(msg.contains("no display"));
        assert!(msg.contains("paused for this cause"));

        // Absent extension — the Web Store fix and the paused recovery tail.
        reset_health();
        set_health(ProbeOutcome::Down(ProbeFailure::ExtensionAbsent));
        let msg = daemon_down_message();
        assert!(msg.contains("Chrome Web Store"));
        assert!(msg.contains("no restart will be attempted"));

        // Wedged daemon — the generic down text plus the restart tail.
        reset_health();
        set_health(ProbeOutcome::Down(ProbeFailure::DaemonWedge));
        let msg = daemon_down_message();
        assert!(msg.contains("down or unresponsive"));
        assert!(msg.contains("restart it automatically"));
        // Leave the singleton pristine for sibling health tests.
        reset_health();
    }

    #[test]
    fn sustained_health_resets_restart_attempts() {
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
        // A transient healthy result (e.g. the post-restart verification probe)
        // must not reset the budget — a runaway cycle would otherwise reopen a
        // fresh bounded cycle on every restart.
        h.apply_outcome(ProbeOutcome::Healthy, now, false);
        assert_eq!(h.restart_budget.attempts, 2);
        assert!(h.restart_budget.halted);
        // The first watchdog healthy seeds the sustained-healthy window…
        h.apply_outcome(ProbeOutcome::Healthy, now + WATCHDOG_INTERVAL, true);
        assert_eq!(h.restart_budget.attempts, 2);
        assert!(h.healthy_since.is_some());
        // …but a second healthy before the window elapses still does not reset.
        h.apply_outcome(ProbeOutcome::Healthy, now + WATCHDOG_INTERVAL * 2, true);
        assert_eq!(h.restart_budget.attempts, 2);
        assert!(h.restart_budget.halted);
        // Only sustained health across the window opens a fresh bounded cycle.
        h.apply_outcome(ProbeOutcome::Healthy, now + WATCHDOG_INTERVAL * 3, true);
        assert_eq!(h.restart_budget.attempts, 0);
        assert_eq!(h.restart_budget.next_at, None);
        assert!(!h.restart_budget.halted);
        assert!(h.restart_budget.halted_until.is_none());
        assert_eq!(h.failure(), None);
    }

    #[test]
    fn cause_flapping_and_transient_health_do_not_reset_restart_budget() {
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
        // A cause flip (wedge → relay-down) must NOT reset the budget —
        // alternating causes must not evade the 3-attempt halt.
        h.apply_outcome(ProbeOutcome::Down(ProbeFailure::RelayDown), now, true);
        assert_eq!(h.failure(), Some(ProbeFailure::RelayDown));
        assert_eq!(h.restart_budget.attempts, 2);
        assert!(h.restart_budget.next_at.is_some());
        // Flapping back and forth accumulates — never resets.
        h.apply_outcome(ProbeOutcome::Down(ProbeFailure::DaemonWedge), now, true);
        h.apply_outcome(ProbeOutcome::Down(ProbeFailure::RelayDown), now, true);
        assert_eq!(h.failure(), Some(ProbeFailure::RelayDown));
        assert_eq!(h.restart_budget.attempts, 2);
        // A transient healthy result does not reset either — only sustained
        // health across the window opens a fresh bounded cycle.
        h.apply_outcome(ProbeOutcome::Healthy, now, true);
        assert_eq!(h.restart_budget.attempts, 2);
        assert!(h.restart_budget.next_at.is_some());
        h.apply_outcome(ProbeOutcome::Healthy, now + SUSTAINED_HEALTHY_WINDOW, true);
        assert_eq!(h.failure(), None);
        assert_eq!(h.restart_budget.attempts, 0);
        assert_eq!(h.restart_budget.next_at, None);
        assert!(!h.restart_budget.halted);
    }

    #[test]
    fn gate_honors_backoff_before_halt() {
        let now = Instant::now();
        let mut h = DaemonHealth::default();
        assert_eq!(h.gate_restart(now), RecoveryGate::Allowed(1));
        // 30s backoff before attempt 2.
        assert_eq!(h.gate_restart(now), RecoveryGate::Backoff);
        assert_eq!(
            h.gate_restart(now + RESTART_BACKOFF[0]),
            RecoveryGate::Allowed(2)
        );
        // 2min backoff before attempt 3.
        let t2 = now + RESTART_BACKOFF[0] + RESTART_BACKOFF[1];
        assert_eq!(h.gate_restart(t2), RecoveryGate::Allowed(3));
        // The final 10-min grace is honored before the halt fires.
        assert_eq!(h.gate_restart(t2), RecoveryGate::Backoff);
        let t3 = t2 + RESTART_BACKOFF[2];
        assert_eq!(h.gate_restart(t3), RecoveryGate::Halted);
        assert!(h.restart_budget.halted);
        assert_eq!(h.gate_restart(t3), RecoveryGate::Cooldown);
        // After the cooldown a fresh bounded cycle starts.
        assert_eq!(h.gate_restart(t3 + HALT_COOLDOWN), RecoveryGate::Allowed(1));
        assert_eq!(h.restart_budget.attempts, 1);
        assert!(!h.restart_budget.halted);
    }

    #[test]
    fn relay_down_classification_precedence() {
        // Disabled wins regardless of whether Chrome is running.
        assert_eq!(
            classify_relay_down(ExtensionState::Disabled, Some(true)),
            ProbeFailure::ExtensionDisabled
        );
        assert_eq!(
            classify_relay_down(ExtensionState::Disabled, Some(false)),
            ProbeFailure::ExtensionDisabled
        );
        assert_eq!(
            classify_relay_down(ExtensionState::Disabled, None),
            ProbeFailure::ExtensionDisabled
        );
        // Absent beats ChromeNotRunning — launching can't fix an uninstalled extension.
        assert_eq!(
            classify_relay_down(ExtensionState::Absent, Some(false)),
            ProbeFailure::ExtensionAbsent
        );
        assert_eq!(
            classify_relay_down(ExtensionState::Absent, None),
            ProbeFailure::ExtensionAbsent
        );
        // Present + no Chrome running → ChromeNotRunning.
        assert_eq!(
            classify_relay_down(ExtensionState::Present, Some(false)),
            ProbeFailure::ChromeNotRunning
        );
        // Present + Chrome running (or an inconclusive probe) → transient relay drop.
        assert_eq!(
            classify_relay_down(ExtensionState::Present, Some(true)),
            ProbeFailure::RelayDown
        );
        assert_eq!(
            classify_relay_down(ExtensionState::Present, None),
            ProbeFailure::RelayDown
        );
        // Unknown fails open to the probe; only an explicit no-Chrome proves not running.
        assert_eq!(
            classify_relay_down(ExtensionState::Unknown, Some(false)),
            ProbeFailure::ChromeNotRunning
        );
        assert_eq!(
            classify_relay_down(ExtensionState::Unknown, None),
            ProbeFailure::RelayDown
        );
        // ChromeNotRunning is recoverable (launch), ExtensionAbsent is not.
        assert!(!ProbeFailure::ChromeNotRunning.is_unfixable());
        assert!(ProbeFailure::ExtensionAbsent.is_unfixable());
    }

    #[test]
    fn extension_state_parsing() {
        // No `data` at all, or no `chromeExtension` within it → shape-uncertain,
        // never the unfixable absent cause (fails open as unknown).
        assert_eq!(
            extension_state_from(&serde_json::json!({})),
            ExtensionState::Unknown
        );
        assert_eq!(
            extension_state_from(&serde_json::json!({ "data": {} })),
            ExtensionState::Unknown
        );
        // Explicit null → absent (the CLI reports the extension as not present).
        assert_eq!(
            extension_state_from(&serde_json::json!({ "data": { "chromeExtension": null } })),
            ExtensionState::Absent
        );
        // Present with an empty disableReasons, or with no disableReasons at all.
        assert_eq!(
            extension_state_from(&serde_json::json!({
                "data": { "chromeExtension": { "disableReasons": [] } }
            })),
            ExtensionState::Present
        );
        assert_eq!(
            extension_state_from(&serde_json::json!({
                "data": { "chromeExtension": { "enabled": true } }
            })),
            ExtensionState::Present
        );
        // Disabled when disableReasons is non-empty.
        assert_eq!(
            extension_state_from(&serde_json::json!({
                "data": { "chromeExtension": { "disableReasons": ["user"] } }
            })),
            ExtensionState::Disabled
        );
    }

    #[test]
    fn readiness_reports_every_fact_and_refuses_without_a_real_browser() {
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
        assert!(ready.ready_for_actions());
        assert_eq!(ready.outcome(), ProbeOutcome::Healthy);
        let report = ready.report();
        assert!(report.contains("established:"));
        assert!(report.contains("established as NOT holding:\n    - nothing"));
        assert!(report.contains("not established:\n    - nothing"));
        assert!(report.contains("chrome-use 1.5.141"), "got: {report}");
        assert!(
            report.contains("driving the real profile owner@example.com (abc-123)"),
            "got: {report}"
        );
        assert!(
            report.contains("a real browser is reachable through the relay"),
            "got: {report}"
        );
        // Process presence is reported as a fact, never as health.
        assert!(
            report.contains("process presence is not by itself health"),
            "got: {report}"
        );
        assert!(
            report.contains("this host has a usable display"),
            "got: {report}"
        );
        assert!(
            report.contains("verdict: ready for actions"),
            "got: {report}"
        );

        // An UNANSWERED `browsers` list (`None`, e.g. a subcommand/shape change
        // in a future chrome-use) is a reporting gap: a decisive fact was never
        // established, so the connection is NOT proven, the outcome is `Unknown`
        // rather than health, and no cause is invented from it.
        let mut unanswered_browsers = ready.clone();
        unanswered_browsers.real_browser = None;
        assert!(!unanswered_browsers.ready_for_actions());
        assert_eq!(unanswered_browsers.outcome(), ProbeOutcome::Unknown);
        assert!(!unanswered_browsers.blocked());

        // The failure this gate exists for: the relay is up but no real browser
        // profile is reachable — not ready, and the refusal says plainly what
        // would have happened otherwise. The established negative is reported
        // with what the product KNOWS, under "established as NOT holding".
        let empty = serde_json::json!({ "data": { "browsers": [] } });
        let mut not_ready = Readiness::from_status(Some(&status));
        not_ready.chrome_running = Some(true);
        not_ready.display = Some(true);
        not_ready.real_browser = Readiness::real_browser_from(Some(&empty));
        assert!(!not_ready.ready_for_actions());
        assert_eq!(not_ready.down_cause(), Some(ProbeFailure::RelayDown));
        assert_eq!(
            not_ready.outcome(),
            ProbeOutcome::Down(ProbeFailure::RelayDown)
        );
        let not_ready_report = not_ready.report();
        assert!(
            not_ready_report.contains(
                "established as NOT holding:\n    - no real browser profile is reachable through \
                 the relay"
            ),
            "got: {not_ready_report}"
        );
        // An established cause is the one verdict that says the connection was
        // ruled out, so the action is refused rather than attempted.
        assert!(
            not_ready_report.contains("verdict: NOT ready for actions"),
            "got: {not_ready_report}"
        );
        assert!(
            not_ready
                .refusal()
                .contains("the work would have gone to a browser chrome-use launches itself"),
            "got: {}",
            not_ready.refusal()
        );

        // A host with NO display is reported as such but refuses nothing: a
        // usable display is what chrome-use needs to LAUNCH the owner's Chrome,
        // and a browser already running is still reachable through the relay. The
        // fact is established, but it is not a decisive one — the connection is
        // still proven.
        let mut headless = ready.clone();
        headless.display = Some(false);
        assert!(headless.ready_for_actions());
        assert_eq!(headless.down_cause(), None);
        assert!(!headless.blocked());
        let headless_report = headless.report();
        assert!(
            headless_report.contains("this host has NO usable display"),
            "got: {headless_report}"
        );
        assert!(
            headless_report.contains("verdict: ready for actions"),
            "got: {headless_report}"
        );
        // The display blocks nothing even when another DECISIVE fact is missing:
        // the connection is then not proven (`Unknown`), never ruled out.
        let mut headless_unproven = headless.clone();
        headless_unproven.real_browser = None;
        assert!(!headless_unproven.ready_for_actions());
        assert_eq!(headless_unproven.outcome(), ProbeOutcome::Unknown);
        assert!(!headless_unproven.blocked());
    }

    #[test]
    fn readiness_never_defaults_a_missing_fact_to_fine() {
        // Nothing answered: every fact stays unestablished, the verdict is
        // Unknown rather than Healthy, and no cause is claimed — which is what
        // keeps an unanswerable probe out of both the health verdict and the
        // restart budget.
        let nothing = Readiness::default();
        assert!(!nothing.answered);
        assert!(!nothing.ready_for_actions());
        assert_eq!(nothing.outcome(), ProbeOutcome::Unknown);
        assert_eq!(nothing.down_cause(), None);
        let report = nothing.report();
        assert!(report.contains("established:\n    - nothing"));
        assert!(report.contains("established as NOT holding:"));
        assert!(report.contains("not established:"));
        assert!(
            report.contains("whether the tool answers `status` at all"),
            "got: {report}"
        );
        // Not proven and nothing ruled out: the verdict must not claim the
        // connection is fine, and must not claim it is ruled out either.
        assert!(
            report.contains("verdict: NOT proven ready for actions"),
            "got: {report}"
        );
        assert!(
            !report.contains("verdict: NOT ready for actions"),
            "got: {report}"
        );

        // A status that answered but carries no extension data at all: every
        // decisive key is missing, so nothing is established.
        let bare = serde_json::json!({ "data": { "cliVersion": "1.2.3" } });
        let mut r = Readiness::from_status(Some(&bare));
        r.chrome_running = Some(true);
        assert!(r.answered);
        assert_eq!(r.relay_up, None);
        assert_eq!(r.host_installed, None);
        assert!(!r.ready_for_actions());
        assert_eq!(r.outcome(), ProbeOutcome::Unknown);
    }

    #[test]
    fn readiness_classifies_only_what_a_fact_established() {
        // Installed host + relay down + a closed browser → ChromeNotRunning, so
        // the gate launches the owner's Chrome rather than restarting a daemon.
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
            Some(ProbeFailure::ChromeNotRunning)
        );
        // The process fact is reported as the fact it is, stated flatly in the
        // established-as-NOT-holding list.
        let report = relay_down.report();
        assert!(
            report.contains("    - no Chrome-family process is running"),
            "got: {report}"
        );
        // An absent extension is unfixable — no launch or restart recovers it.
        let mut absent = relay_down;
        absent.extension = Some(ExtensionState::Absent);
        assert_eq!(absent.down_cause(), Some(ProbeFailure::ExtensionAbsent));
        let absent_cause = absent.down_cause();
        assert!(absent_cause.is_some_and(ProbeFailure::is_unfixable));
        // The native-host facts classify before the relay does.
        let uninstalled =
            serde_json::json!({ "data": { "extension": { "hostInstalled": false } } });
        assert_eq!(
            Readiness::from_status(Some(&uninstalled)).down_cause(),
            Some(ProbeFailure::NotInstalled)
        );
    }

    #[test]
    fn launch_budget_is_independent_of_restart_budget() {
        let now = Instant::now();
        // A halted/exhausted RESTART budget must NOT block a launch — the two
        // bounded cycles are independent.
        let mut h = DaemonHealth {
            restart_budget: AttemptBudget {
                attempts: MAX_RESTART_ATTEMPTS,
                halted: true,
                halted_until: Some(now),
                ..AttemptBudget::default()
            },
            ..DaemonHealth::default()
        };
        assert!(matches!(h.gate_launch(now), RecoveryGate::Allowed(_)));
        assert_eq!(h.launch_budget.attempts, 1);
        // An exhausted LAUNCH budget still leaves the restart gate open.
        let mut h2 = DaemonHealth {
            launch_budget: AttemptBudget {
                attempts: MAX_LAUNCH_ATTEMPTS,
                ..AttemptBudget::default()
            },
            ..DaemonHealth::default()
        };
        assert!(!matches!(h2.gate_launch(now), RecoveryGate::Allowed(_)));
        assert!(matches!(h2.gate_restart(now), RecoveryGate::Allowed(_)));
        // Sustained health resets both budgets and clears a recorded launch
        // outcome (a pending Failed launch must not survive recovery).
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
        // The first healthy seeds the sustained-healthy window but does not yet
        // reset either budget; a healthy observation clears a stale launch record.
        h3.apply_outcome(ProbeOutcome::Healthy, now + WATCHDOG_INTERVAL, true);
        assert_eq!(h3.restart_budget.attempts, 2);
        assert_eq!(h3.launch_budget.attempts, 2);
        assert!(h3.restart_budget.halted);
        assert!(h3.launch_budget.halted);
        assert_eq!(h3.launch_outcome, None);
        // A healthy across the sustained window opens fresh bounded cycles.
        h3.apply_outcome(ProbeOutcome::Healthy, now + WATCHDOG_INTERVAL * 3, true);
        assert_eq!(h3.restart_budget.attempts, 0);
        assert_eq!(h3.launch_budget.attempts, 0);
        assert!(!h3.restart_budget.halted);
        assert!(!h3.launch_budget.halted);
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
