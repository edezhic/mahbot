//! Daemon temp lifecycle: one private `/tmp/mahbot` root for ALL daemon temp
//! artifacts, plus the periodic OS temp-dir cleaner that reclaims abandoned
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
//! The root-setup code itself performs NO startup/periodic reclamation of its
//! own — crash leftovers in the root are the operating system's job (the OS
//! removes stale files on its periodic temp sweep), plus the run's own
//! completion flow (see [`crate::research_cleanup`]). The ONE exception is the
//! periodic temp-dir cleaner described below, which reclaims abandoned agent
//! artifacts that the OS sweep and run-completion flow leave behind.
//!
//! The path is fixed (no per-user `-{uid}` suffix): this is a single-user
//! deployment, and the suffix was superfluous. The accepted multi-user
//! consequence: with a shared fixed path, a second OS user's daemon fails
//! loudly at boot via the ownership check below — accepted for this
//! deployment. Old suffixed leftovers (`/tmp/mahbot-<uid>`) are left to the
//! OS sweep.
//!
//! Shell children get `TMPDIR` set to the same root (see
//! [`crate::tools::shell::shell_tmpdir`]).
//!
//! ## Periodic cleaner (Sanitation role)
//!
//! A Sanitation-role agent removes clearly-old, abandoned agent artifacts
//! from the common OS temp folder — including the pinned daemon temp root
//! (`/tmp/mahbot`) — on the nightly discovery pass cadence. Dispatched
//! fire-and-forget from the nightly-check loop's gated pass block
//! (`crate::workspace::run_nightly_check_loop`), it inherits the rolling
//! 7-day discovery cadence (at most one cleaner per 7 days); the gate's
//! pass-start timestamp also dedups the two 30-min wakes in the nightly
//! window. The durable `temp_cleanup` jobs row isolates the cleaner from the
//! discovery pass: a crash leaves the row `'launched'`, and the boot scan
//! (`crate::jobs::recover_from_restart`) terminalizes leftover rows WITHOUT
//! resuming — the cleanup is best-effort and the next scheduled pass simply
//! runs again.
//!
//! The cleaner's only safety mechanism is its task prompt
//! (`src/prompt/sanitation/temp_cleanup.md`) — deliberately over-specified
//! deletion criteria (age + artifact shape + daemon uid, with hard bans on
//! sockets/FIFOs, fresh files, and active research run folders). There are NO
//! programmatic exclusions and no shell-guard changes; the read-only shell's
//! existing TEMP_MUTATORS gate on temp roots is what permits deletion at all.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::Workspace;
use anyhow::Result;

/// The pinned private temp root, set once by [`init_temp_root`].
static TEMP_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// The pre-pin OS temp dir (e.g. `/var/folders/.../T` on macOS). Captured
/// before the `TMPDIR` pin so the readonly guard keeps it as an allowed root
/// (bare macOS `mktemp` ignores `TMPDIR` and lands there).
static LEGACY_TEMP_DIR: OnceLock<PathBuf> = OnceLock::new();

/// The pinned root path, if [`init_temp_root`] ran (production unix startup).
#[must_use]
pub(crate) fn temp_root() -> Option<&'static Path> {
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

/// Prompt asset for the periodic temp-dir cleaner task.
const TEMP_CLEANUP_PROMPT_KEY: &str = "sanitation/temp_cleanup.md";

/// Synthetic workspace name for the cleaner. Never registered in
/// the `workspaces` table — ephemeral, like research run roots. Explicit name: the
/// pinned temp root's last path component is "mahbot", which would collide
/// with the registered repo workspace via `Workspace::from_path`.
const TEMP_CLEANUP_WORKSPACE_NAME: &str = "tmp";

/// Does a `temp_cleanup` jobs row survive (status != 'done')? The dispatch
/// dedup marker: a surviving row means a previous cleaner never finished
/// (crash); the boot scan terminalizes leftover rows, so a row present at
/// dispatch time is an anomaly — skip rather than double-run.
pub(crate) async fn temp_cleanup_row_exists(conn: &crate::db::Connection) -> Result<bool> {
    Ok(conn
        .query_optional(
            "SELECT 1 FROM jobs WHERE kind = 'temp_cleanup' AND status != 'done' LIMIT 1",
            (),
            |_| Ok::<(), anyhow::Error>(()),
        )
        .await?
        .is_some())
}

/// Dispatch the periodic temp-dir cleaner (fire-and-forget, Sanitation role).
///
/// Called from the nightly-check loop's gated pass block so the cleaner
/// inherits the discovery cadence (at most once per 7 days). A failure here
/// only logs — the discovery pass continues (and vice versa).
pub(crate) async fn dispatch_temp_cleanup() -> Result<()> {
    let conn = &crate::session::store().conn;
    if temp_cleanup_row_exists(conn).await? {
        tracing::info!("Temp-dir cleanup already in flight — skipping dispatch");
        return Ok(());
    }

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
    )
    .await
    .map_err(|e| {
        tracing::error!(job = %job_id, error = %e, "Failed to spawn temp-dir cleanup job");
        e
    })?;

    // Fire-and-forget: raw tokio::spawn (NOT spawn_cancellable) — same
    // rationale as the research-run cleanup. Durability is the jobs row; the
    // boot scan terminalizes leftovers instead of resuming them.
    let ws = ws.clone();
    let job_id_log = job_id.clone();
    let agent_id = crate::research_cleanup::cleanup_agent_id(&job_id);
    let agent_id_log = agent_id.clone();
    tokio::spawn(async move {
        run_temp_cleanup_and_finish(&job_id, &ws, &prompt).await;
    });

    tracing::info!(job = %job_id_log, agent = %agent_id_log, "Temp-dir cleanup dispatched");
    Ok(())
}

/// Run the cleaner agent and terminalize the job row on EVERY exit path.
///
/// No folder hold exists for this cleaner (unlike research cleanup) — the row
/// is simply deleted when the run finishes (success OR failure), so the next
/// scheduled pass re-runs cleanly.
async fn run_temp_cleanup_and_finish(job_id: &str, ws: &Workspace, prompt: &str) {
    let agent_id = crate::research_cleanup::cleanup_agent_id(job_id);
    let (agent, response) = crate::agent::run_default_agent(
        &agent_id,
        crate::Role::Sanitation,
        ws,
        prompt,
        None,
        None,
        None,
    )
    .await;
    let report = response.unwrap_or_else(|| {
        format!(
            "Temp-dir cleanup FAILED (job {job_id}): {}",
            agent
                .failure
                .clone()
                .unwrap_or_else(|| "no failure detail".to_string())
        )
    });
    tracing::info!(
        job = %job_id,
        agent = %agent_id,
        "Temp-dir cleanup finished: {}",
        crate::util::scrub_credentials(&report)
    );
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

    async fn init_stores() {
        crate::util::test::init_management_test_stores().await;
    }

    /// Dispatch must be deduped by a surviving `temp_cleanup` jobs row (what
    /// a crash-interrupted cleaner leaves behind) — no second row, no agent
    /// spawn. Mirrors the research-cleanup dedup test. Serialized with the
    /// jobs boot-arm test on `reset_inflight`: both create `temp_cleanup`
    /// rows in the same process-global session store, and this test counts
    /// ALL `temp_cleanup` rows (a wrongly-run dispatch would create a fresh
    /// random id, so the count cannot be id-scoped like the research dedup
    /// test) — a concurrent row from the boot-arm test would inflate it.
    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // shares session-store rows with the jobs boot-arm test
    async fn dispatch_temp_cleanup_deduped_by_jobs_row() {
        init_stores().await;
        crate::jobs::spawn_job(
            &crate::session::store().conn,
            "tmpclean_dedup",
            "task",
            "tmp",
            "",
            "",
            crate::Role::Sanitation,
            &[],
            &crate::jobs::SpawnChild::TempCleanup,
        )
        .await
        .unwrap();
        assert!(
            temp_cleanup_row_exists(&crate::session::store().conn)
                .await
                .unwrap(),
            "the pre-created row is the dedup marker"
        );
        dispatch_temp_cleanup().await.unwrap();
        let rows = crate::session::store()
            .conn
            .query("SELECT COUNT(*) FROM jobs WHERE kind = 'temp_cleanup'", ())
            .await
            .unwrap();
        assert_eq!(
            rows[0].get::<i64>(0).unwrap(),
            1,
            "single temp_cleanup job row"
        );
        let sessions = crate::session::store()
            .conn
            .query(
                "SELECT COUNT(*) FROM session_metadata WHERE agent_id = 'cleanup_tmpclean_dedup'",
                (),
            )
            .await
            .unwrap();
        assert_eq!(
            sessions[0].get::<i64>(0).unwrap(),
            0,
            "deduped dispatch must not spawn the temp cleaner"
        );
        crate::jobs::terminalize_job(&crate::session::store().conn, "tmpclean_dedup")
            .await
            .unwrap();
    }
}
