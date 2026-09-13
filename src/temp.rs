//! Daemon temp lifecycle: one private `/tmp/mahbot` root for ALL daemon temp
//! artifacts, plus the periodic temp cleaner that reclaims abandoned
//! agent artifacts from the common OS temp folder.
//!
//! ## One root
//!
//! The daemon creates a single private root under `/tmp` (`/tmp/mahbot`,
//! mode 0700) with exclusive-create + ownership/mode verification, and fails
//! loudly on a squatted path. It then pins its own `TMPDIR` to that root at
//! the very start of startup (before config and any temp use), so every
//! `std::env::temp_dir()`-based consumer relocates automatically: shell spill
//! files, research run folders, background-mode output, voice/telegram temp.
//! The root is simply recreated/re-verified on every boot.
//!
//! The root-setup code itself performs NO startup reclamation of its own —
//! crash leftovers in the root are reclaimed by the periodic temp cleaner
//! below, by the OS's own temp sweep, and by each run's completion flow (see
//! [`crate::research_cleanup`]).
//!
//! The path is fixed (no per-user `-{uid}` suffix): this is a single-user
//! deployment, and the suffix was superfluous. The accepted multi-user
//! consequence: with a shared fixed path, a second OS user's daemon fails
//! loudly at boot via the ownership check below. Old suffixed leftovers
//! (`/tmp/mahbot-<uid>`) are reclaimed by the periodic temp cleaner like any
//! other old agent artifact.
//!
//! Shell children get `TMPDIR` set to the same root (see
//! [`crate::tools::shell::shell_tmpdir`]).
//!
//! ## Periodic cleaner (Sanitation role)
//!
//! A Sanitation-role agent ([`run_temp_cleanup_loop`]) keeps the common OS temp
//! area bounded. Its judgement and its scan roots live in its task prompt
//! (`src/prompt/sanitation/temp_cleanup.md`): there are no programmatic
//! exclusions and no shell-guard changes — the read-only shell's existing
//! TEMP_MUTATORS gate on temp roots is what permits deletion at all. This
//! module owns only WHEN the cleaner runs; the cadence rules are documented on
//! the constants and helpers below.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use crate::Workspace;
use crate::util::UnwrapPoison;
use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use futures_util::FutureExt;

/// The pinned private temp root, set once by [`init_temp_root`].
static TEMP_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// The pre-pin OS temp dir (e.g. `/var/folders/.../T` on macOS). Captured
/// before the `TMPDIR` pin so the readonly guard keeps it as an allowed root
/// (bare macOS `mktemp` ignores `TMPDIR` and lands there).
static LEGACY_TEMP_DIR: OnceLock<PathBuf> = OnceLock::new();

/// The pinned root path, if [`init_temp_root`] ran (production unix startup).
#[must_use]
fn temp_root() -> Option<&'static Path> {
    TEMP_ROOT.get().map(PathBuf::as_path)
}

/// The pre-pin OS temp dir (the legacy darwin dir on macOS), if captured.
#[must_use]
pub(crate) fn legacy_temp_dir() -> Option<&'static Path> {
    LEGACY_TEMP_DIR.get().map(PathBuf::as_path)
}

/// Initialize the private temp root and pin `TMPDIR` to it.
///
/// Must run at the very start of startup, BEFORE config and any temp use, and
/// AFTER the debug/`__grep-engine` subcommand dispatches (those must not
/// create the root). Unix-only: `/tmp/mahbot` is meaningless on Windows,
/// where the shell env uses `TEMP`/`TMP` instead.
///
/// Failure modes (fail loudly, never paper over):
/// - the root exists but is not a directory;
/// - the root exists and is owned by a different uid (squatting — with the
///   fixed shared path this is also the second-OS-user guard: their daemon
///   fails loudly here, the accepted multi-user consequence);
/// - the root's mode has group/other bits AND re-chmod fails — a loose mode on
///   OUR OWN path (uid verified) is self-healed to 0700 (a previous boot's
///   create-time chmod can fail on a race with the umask; bricking startup
///   forever over that would be worse).
#[cfg(unix)]
pub fn init_temp_root() -> anyhow::Result<()> {
    // Capture the legacy temp dir BEFORE the pin — on macOS this is the
    // darwin user temp dir (`/var/folders/.../T`) that bare `mktemp` uses.
    let legacy = std::env::temp_dir();
    let uid = unsafe { libc::geteuid() };
    // Fixed shared path (no `-{uid}` suffix): single-user deployment; the
    // ownership check below is what makes the shared path safe — a second OS
    // user's daemon fails loudly at boot instead of sharing the root.
    let root = PathBuf::from("/tmp/mahbot");

    // Exclusive create with explicit mode 0700. `create_dir` fails when the
    // path already exists (exclusive semantics); the mode is applied
    // explicitly because `create_dir` honors the process umask.
    match std::fs::create_dir(&root) {
        Ok(()) => {
            std::fs::set_permissions(&root, std::os::unix::fs::PermissionsExt::from_mode(0o700))
                .map_err(|e| {
                    anyhow::anyhow!("temp root {}: chmod 0700 failed: {e}", root.display())
                })?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            // Reuse after verification: ownership + mode + dir-ness.
            use std::os::unix::fs::MetadataExt;
            let meta = std::fs::symlink_metadata(&root)
                .map_err(|e| anyhow::anyhow!("temp root {}: stat failed: {e}", root.display()))?;
            if !meta.is_dir() {
                anyhow::bail!(
                    "temp root {} exists and is not a directory — refusing to use it",
                    root.display()
                );
            }
            if meta.uid() != uid {
                anyhow::bail!(
                    "temp root {} is owned by uid {} (expected {uid}) — refusing a squatted path",
                    root.display(),
                    meta.uid()
                );
            }
            let mode = meta.mode() & 0o777;
            if mode & 0o077 != 0 {
                // Self-heal: a loose mode on OUR OWN path is a previous
                // boot's create-time chmod failure (umask raced it), NOT a
                // squatter — the uid check above already proved ownership.
                // Re-chmod 0700 instead of bricking startup forever.
                std::fs::set_permissions(&root, std::os::unix::fs::PermissionsExt::from_mode(0o700))
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "temp root {} has group/other permissions (mode {mode:o}) and re-chmod 0700 failed: {e}",
                            root.display()
                        )
                    })?;
                tracing::warn!(
                    root = %root.display(),
                    mode = format_args!("{mode:o}"),
                    "Temp root had loose permissions — re-chmod 0700 (self-heal, path owned by self)"
                );
            }
        }
        Err(e) => anyhow::bail!("temp root {}: create failed: {e}", root.display()),
    }

    let _ = TEMP_ROOT.set(root.clone());
    let _ = LEGACY_TEMP_DIR.set(legacy);

    // Pin TMPDIR before any temp use. SAFETY: single-threaded startup (before
    // the tokio runtime / iced), no concurrent env access.
    unsafe { std::env::set_var("TMPDIR", &root) };
    tracing::info!(root = %root.display(), "Pinned daemon temp root");
    Ok(())
}

/// No-op on non-unix (the shell env there uses `TEMP`/`TMP`, not `TMPDIR`).
#[cfg(not(unix))]
pub fn init_temp_root() -> anyhow::Result<()> {
    Ok(())
}

/// The `TMPDIR` value for shell children: the pinned root when available,
/// otherwise the historical `"/tmp"` baseline (tests / non-unix).
#[must_use]
pub(crate) fn shell_tmpdir() -> String {
    temp_root().map_or_else(|| "/tmp".to_string(), |p| p.to_string_lossy().into_owned())
}

/// Where a BARE `mktemp -d` (no `-p`/template) actually lands on this
/// platform. On macOS bare mktemp ignores `TMPDIR` and uses
/// `_CS_DARWIN_USER_TEMP_DIR` (the legacy darwin dir); elsewhere mktemp
/// honors `TMPDIR`. The readonly guard's synthetic mktemp anchor must match
/// this, so `..` chains over it resolve like the real value.
#[must_use]
pub(crate) fn bare_mktemp_landing_root() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        if let Some(legacy) = legacy_temp_dir() {
            return legacy.to_path_buf();
        }
    }
    std::env::temp_dir()
}

// ── Adaptive cadence ──────────────────────────────────────────────────────

/// How often the cadence loop wakes to check whether the cleanup is due. The
/// due test is wall-clock based, so this only bounds how late a due run may
/// start.
const TEMP_CLEANUP_WAKE: Duration = Duration::from_mins(30);

const GIB: u64 = 1 << 30;

/// Absolute floors under the percentage thresholds (10 GiB / 15 GiB).
const TEMP_CLEANUP_LOW_FREE_FLOOR: u64 = 10 * GIB;
const TEMP_CLEANUP_RECOVERED_FREE_FLOOR: u64 = 15 * GIB;
/// Percentage thresholds of the volume's capacity.
const TEMP_CLEANUP_LOW_FREE_PCT: u64 = 10;
const TEMP_CLEANUP_RECOVERED_FREE_PCT: u64 = 15;

/// `config_kv` key holding the cleanup's last-run start (RFC 3339). Owned by
/// this subsystem — preserved by the config orphan purge.
pub(crate) const TEMP_CLEANUP_LAST_RUN_KV_KEY: &str = "temp_cleanup_last_run_at";
/// `config_kv` key holding the sticky hysteresis mode (`"daily"`/`"weekly"`).
pub(crate) const TEMP_CLEANUP_MODE_KV_KEY: &str = "temp_cleanup_mode";

/// In-memory mirror of the latest dispatch (wall clock). The persisted
/// `config_kv` timestamp is the cross-restart record; this mirror exists so a
/// failing config-store write cannot turn the cadence into a re-dispatch loop
/// within the same lifetime.
static LAST_DISPATCH: std::sync::Mutex<Option<DateTime<Utc>>> = std::sync::Mutex::new(None);

/// Free-space mode of the cleaner, with hysteresis between the two thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CleanupMode {
    /// Free space is tight: at most one cleanup per day.
    Daily,
    /// Free space is comfortable: at most one cleanup per week.
    Weekly,
}

impl CleanupMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Daily => "daily",
            Self::Weekly => "weekly",
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "daily" => Some(Self::Daily),
            "weekly" => Some(Self::Weekly),
            _ => None,
        }
    }

    fn interval(self) -> ChronoDuration {
        match self {
            Self::Daily => ChronoDuration::days(1),
            Self::Weekly => ChronoDuration::days(7),
        }
    }
}

/// Free space below which the cleaner switches to daily mode.
fn daily_entry_threshold(capacity: u64) -> u64 {
    (capacity.saturating_mul(TEMP_CLEANUP_LOW_FREE_PCT) / 100).max(TEMP_CLEANUP_LOW_FREE_FLOOR)
}

/// Free space above which daily mode is left for weekly mode.
fn weekly_return_threshold(capacity: u64) -> u64 {
    (capacity.saturating_mul(TEMP_CLEANUP_RECOVERED_FREE_PCT) / 100)
        .max(TEMP_CLEANUP_RECOVERED_FREE_FLOOR)
}

/// Apply the hysteresis thresholds to the current mode. Below the entry
/// threshold the mode is always daily; above the (higher) return threshold it
/// is always weekly; in between the current mode sticks, so the cadence cannot
/// flap around the line.
fn mode_after_free(current: CleanupMode, free: u64, capacity: u64) -> CleanupMode {
    if free < daily_entry_threshold(capacity) {
        CleanupMode::Daily
    } else if free > weekly_return_threshold(capacity) {
        CleanupMode::Weekly
    } else {
        current
    }
}

/// The sticky mode persisted in `config_kv`; unset/unparseable/read-error
/// falls back to weekly (the documented default where free space is unknown).
async fn stored_cleanup_mode(store: Option<&crate::config_db::ConfigStore>) -> CleanupMode {
    let Some(store) = store else {
        return CleanupMode::Weekly;
    };
    match store.get_kv(TEMP_CLEANUP_MODE_KV_KEY).await {
        Ok(Some(raw)) => CleanupMode::parse(&raw).unwrap_or_else(|| {
            tracing::warn!(value = %raw, "Unrecognized temp cleaner mode — using weekly");
            CleanupMode::Weekly
        }),
        Ok(None) => CleanupMode::Weekly,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to read temp cleaner mode — using weekly");
            CleanupMode::Weekly
        }
    }
}

/// Parse a persisted last-run timestamp. `None` (missing, or unparseable —
/// fail-open) makes the cleanup due.
fn parse_last_run(raw: &str) -> Option<DateTime<Utc>> {
    match crate::db::parse_utc_timestamp(raw) {
        Ok(last) => Some(last),
        Err(e) => {
            tracing::warn!(value = %raw, error = %e, "Unparseable temp cleaner last run — treating as due");
            None
        }
    }
}

/// Is the cleanup due? No recorded run means due; otherwise the wall-clock
/// elapsed time must cover the mode's interval.
fn temp_cleanup_due(
    last_run_at: Option<DateTime<Utc>>,
    mode: CleanupMode,
    now: DateTime<Utc>,
) -> bool {
    match last_run_at {
        None => true,
        Some(last) => now.signed_duration_since(last) >= mode.interval(),
    }
}

/// The cleaner's dedicated cadence loop, spawned once from the background-task
/// set. Decoupled from every other periodic loop so its frequency follows free
/// disk space alone. Panic posture matches the sibling periodic loops: the
/// task-level guard in `spawn_cancellable` logs and ends the loop, while the
/// agent run is guarded in [`run_temp_cleanup_and_finish`] so its jobs row is
/// terminalized on every exit path.
pub async fn run_temp_cleanup_loop() {
    loop {
        // Sleep BEFORE the first check: a cleaner run must never compete with
        // startup (store opens, provider warmup, boot recovery). The first tick
        // is far later than `jobs::recover_from_restart`, so a leftover
        // `temp_cleanup` row from a crash is already terminalized by then; even
        // an unnoticed one would be harmless — each run terminalizes its own
        // row.
        if !crate::shutdown::sleep_or_shutdown_or_drain(TEMP_CLEANUP_WAKE).await {
            break;
        }
        temp_cleanup_tick().await;
    }
}

/// One cadence tick: refresh the sticky mode from current free space, then
/// dispatch the cleaner when due.
async fn temp_cleanup_tick() {
    let store = crate::config_db::CONFIG_STORE.get();
    let current = stored_cleanup_mode(store).await;
    let mode = match crate::util::disk::free_and_capacity(&std::env::temp_dir()) {
        Some((free, capacity)) => mode_after_free(current, free, capacity),
        None => current,
    };
    if mode != current
        && let Some(store) = store
    {
        if let Err(e) = store.set_kv(TEMP_CLEANUP_MODE_KV_KEY, mode.as_str()).await {
            tracing::warn!(error = %e, "Failed to persist temp cleaner mode");
        }
        tracing::info!(mode = mode.as_str(), "Temp cleaner cadence mode changed");
    }

    let persisted = match store {
        Some(store) => match store.get_kv(TEMP_CLEANUP_LAST_RUN_KV_KEY).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to read temp cleaner last run — treating as due");
                None
            }
        },
        None => None,
    };
    // The persisted record is authoritative across restarts; the in-memory
    // mirror keeps a failed config write from re-dispatching every tick.
    let last_run = [
        persisted.as_deref().and_then(parse_last_run),
        *LAST_DISPATCH.lock().unwrap_poison(),
    ]
    .into_iter()
    .flatten()
    .max();
    if !temp_cleanup_due(last_run, mode, Utc::now()) {
        return;
    }
    if let Err(e) = dispatch_temp_cleanup().await {
        tracing::warn!(error = %e, "Temp cleaner dispatch failed");
    }
}

// ── Dispatch ──────────────────────────────────────────────────────────────

/// Prompt asset for the periodic temp cleaner task.
const TEMP_CLEANUP_PROMPT_KEY: &str = "sanitation/temp_cleanup.md";

/// Synthetic workspace name for the cleaner. Never registered in the
/// `workspaces` table — ephemeral, like research run roots. Explicit name: the
/// pinned temp root's last path component is "mahbot", which would collide
/// with the registered repo workspace via `Workspace::from_path`.
const TEMP_CLEANUP_WORKSPACE_NAME: &str = "tmp";

/// Dispatch the cleaner and run it to completion (Sanitation role).
///
/// The loop is the only dispatcher and awaits the run, so dispatch is
/// serialized and a crashed run can never strand the cadence. The last-run
/// timestamp is recorded right after the durable row exists (a failed dispatch
/// leaves the cleanup due, retried on the next tick) in BOTH the persisted
/// `config_kv` record and the in-memory mirror; the row is terminalized on
/// every exit path (including a panic inside the agent).
async fn dispatch_temp_cleanup() -> Result<()> {
    let conn = &crate::session::store().conn;
    let job_id = crate::generate_id();
    // Ephemeral workspace over the common OS temp dir: the shell cwd lands
    // under the scan roots and workspace-relative reads resolve there.
    let ws = Workspace::ephemeral_run(TEMP_CLEANUP_WORKSPACE_NAME, Path::new("/tmp"));
    let prompt = crate::prompt::load_prompt(TEMP_CLEANUP_PROMPT_KEY);

    crate::jobs::spawn_job(
        conn,
        &job_id,
        &prompt,
        &ws.name,
        "",
        "",
        crate::Role::Sanitation,
        &[crate::jobs::NewAgent {
            agent_id: crate::research_cleanup::cleanup_agent_id(&job_id),
            kind: crate::jobs::AgentKind::Sanitation,
            idx: None,
            task: prompt.clone(),
        }],
        &crate::jobs::SpawnChild::TempCleanup,
        None,
    )
    .await
    .map_err(|e| {
        tracing::error!(job = %job_id, error = %e, "Failed to spawn temp cleaner job");
        e
    })?;

    let now = Utc::now();
    *LAST_DISPATCH.lock().unwrap_poison() = Some(now);
    if let Some(store) = crate::config_db::CONFIG_STORE.get()
        && let Err(e) = store
            .set_kv(TEMP_CLEANUP_LAST_RUN_KV_KEY, &now.to_rfc3339())
            .await
    {
        tracing::warn!(error = %e, "Failed to record temp cleaner run start");
    }

    tracing::info!(job = %job_id, "Temp cleaner dispatched");
    run_temp_cleanup_and_finish(&job_id, &ws, &prompt).await;
    Ok(())
}

/// Run the cleaner agent and terminalize the job row on EVERY exit path,
/// including a panic inside the agent (so a crashed run never leaves a
/// stranded row behind).
///
/// No folder hold exists for this cleaner (unlike research cleanup) — the row
/// is simply deleted when the run finishes (success OR failure), so the next
/// due pass re-runs cleanly.
async fn run_temp_cleanup_and_finish(job_id: &str, ws: &Workspace, prompt: &str) {
    let agent_id = crate::research_cleanup::cleanup_agent_id(job_id);
    let run = std::panic::AssertUnwindSafe(crate::agent::run_default_agent(
        &agent_id,
        crate::Role::Sanitation,
        ws,
        prompt,
        false,
        None,
        None,
        None,
    ))
    .catch_unwind()
    .await;
    match run {
        Ok((agent, response)) => {
            let report = response.unwrap_or_else(|| {
                format!(
                    "Temp cleaner FAILED (job {job_id}): {}",
                    agent
                        .failure
                        .clone()
                        .unwrap_or_else(|| "no failure detail".to_string())
                )
            });
            tracing::info!(
                job = %job_id,
                agent = %agent_id,
                "Temp cleaner finished: {}",
                crate::util::scrub_credentials(&report)
            );
        }
        Err(payload) => tracing::error!(
            job = %job_id,
            agent = %agent_id,
            "Temp cleaner panicked: {}",
            crate::util::panic_message(&*payload)
        ),
    }
    let _ = crate::jobs::terminalize_job(&crate::session::store().conn, job_id).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn legacy_capture_and_pin_are_consistent() {
        // init_temp_root is a process-global singleton — not run in tests
        // (it would pin TMPDIR for the whole test process). Just verify the
        // accessors are coherent: unset → no root, no legacy.
        assert!(temp_root().is_none());
        assert!(legacy_temp_dir().is_none());
        assert_eq!(shell_tmpdir(), "/tmp");
    }

    #[test]
    fn mode_after_free_applies_hysteresis() {
        // 50 GiB volume: the 10/15 GiB floors dominate the percentages; 460 GiB
        // volume: the 10%/15% percentages dominate the floors.
        let small = 50 * GIB;
        let big = 460 * GIB;
        assert_eq!(daily_entry_threshold(small), 10 * GIB);
        assert_eq!(weekly_return_threshold(small), 15 * GIB);
        assert_eq!(daily_entry_threshold(big), 46 * GIB);
        assert_eq!(weekly_return_threshold(big), 69 * GIB);

        // Below the entry threshold the mode is always daily; above the return
        // threshold always weekly.
        assert_eq!(
            mode_after_free(CleanupMode::Weekly, 5 * GIB, small),
            CleanupMode::Daily
        );
        assert_eq!(
            mode_after_free(CleanupMode::Daily, 20 * GIB, small),
            CleanupMode::Weekly
        );
        // Between the thresholds the current mode sticks — no flapping.
        assert_eq!(
            mode_after_free(CleanupMode::Daily, 12 * GIB, small),
            CleanupMode::Daily
        );
        assert_eq!(
            mode_after_free(CleanupMode::Weekly, 12 * GIB, small),
            CleanupMode::Weekly
        );
    }

    #[test]
    fn due_uses_wall_clock_and_mode_interval() {
        let now = Utc::now();
        assert!(temp_cleanup_due(None, CleanupMode::Weekly, now));
        let two_days_ago = now - ChronoDuration::days(2);
        assert!(temp_cleanup_due(
            Some(two_days_ago),
            CleanupMode::Daily,
            now
        ));
        assert!(!temp_cleanup_due(
            Some(two_days_ago),
            CleanupMode::Weekly,
            now
        ));
        let eight_days_ago = now - ChronoDuration::days(8);
        assert!(temp_cleanup_due(
            Some(eight_days_ago),
            CleanupMode::Weekly,
            now
        ));
        let recent = now - ChronoDuration::hours(1);
        assert!(!temp_cleanup_due(Some(recent), CleanupMode::Daily, now));
    }
}
