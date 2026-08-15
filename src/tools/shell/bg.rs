//! Background shell sessions (Full shell roles only).
//!
//! An agent can launch a long-running non-interactive command that keeps
//! running after the initiating tool call returns. Output is written RAW
//! (no scrubbing, no profile transforms) to a file in the temp area's
//! `.agent` directory; the agent reads progress with the read tool and stops
//! the session via the shell tool's `stop` argument.
//!
//! Sessions are strictly agent-scoped: the registry lives inside the
//! [`crate::Agent`] and is force-killed on agent teardown ([`BackgroundSessions::terminate_all`]).
//!
//! # Watchdog (Unix)
//!
//! A child-side watcher process keeps the process group alive only while the
//! daemon is alive. A pipe is created with both ends `O_CLOEXEC`; the read
//! end becomes the watcher's stdin and the write end is held by the daemon
//! (in the session entry). The watcher blocks reading stdin; when the write
//! end closes — daemon crash/kill/exit, or the session finishing — the
//! watcher SIGKILLs the whole process group. The watcher is the group leader
//! (`process_group(0)`); the launched command joins its group, so a single
//! `kill(-pgid)` (or the watcher's `kill 0`) reaches the command and all its
//! descendants. This guarantees no orphaned background processes even on a
//! hard daemon crash, without any daemon-shutdown hook (destructors do not
//! run on SIGKILL/SIGSEGV/OOM/`process::exit`, and the platform lacks
//! parent-death signals).
//!
//! The same watchdog covers the launch window: if the launch future is
//! aborted between process spawn and session registration, the write end
//! (a local in the future) is closed on drop, the watcher fires, and the
//! group dies — no dedicated kill-on-drop guard is needed.
//!
//! # Windows
//!
//! No watchdog and no process-group kill: the direct child is killed on stop
//! and teardown, and grandchildren survive (accepted asymmetry — documented
//! per the spec).
//!
//! The launch-failure probe is Unix-shell-specific as well: cmd.exe exits 1
//! (not 126/127) for a missing or non-executable command, so on Windows a
//! failed launch surfaces as a successful background session whose output
//! file holds the cmd.exe error and `[exit status: 1]` — the synchronous
//! launch-error path is Unix-only (accepted deviation from the launch-failure
//! contract, mirroring the kill asymmetry above).

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
#[cfg(unix)]
use std::os::fd::FromRawFd;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::util::UnwrapPoison;

/// Watcher script: ignore SIGTERM (the daemon owns the graceful two-stage
/// stop), block on stdin until EOF, then SIGKILL the caller's process group
/// (`kill -s KILL 0` targets the caller's own group — the watcher is the
/// group leader, and the launched command joined its group).
#[cfg(unix)]
const WATCHER_SCRIPT: &str = "trap '' TERM; cat >/dev/null; kill -s KILL 0";

/// Bounded early-exit launch probe: command-not-found / not-executable exits
/// (126/127) within this window are treated as synchronous launch failures.
/// All other early exits — including legitimate non-zero exits (grep
/// no-match, diff differences, test failures) and exit 0 — are successful
/// launches surfaced via the unconditional exit-status annotation.
const LAUNCH_PROBE: Duration = Duration::from_millis(250);
/// Poll interval for the launch probe and the stop/teardown completion wait.
const POLL_INTERVAL: Duration = Duration::from_millis(25);
/// Waiter poll interval while waiting for the command child to exit.
const WAITER_POLL: Duration = Duration::from_millis(50);
/// Grace between SIGTERM and SIGKILL in the two-stage stop.
const STOP_GRACE: Duration = Duration::from_secs(5);
/// How long `stop()` waits for the waiter to append the exit annotation after
/// the SIGKILL before returning (best-effort determinism for the follow-up
/// read of the output file).
const STOP_ANNOTATION_WAIT: Duration = Duration::from_secs(2);
/// Output-file name attempts before giving up (create_new never overwrites).
const MAX_OUTPUT_NAME_ATTEMPTS: usize = 16;
/// How many leading bytes of the output file are included in the synchronous
/// launch-failure tool error (the shell's "command not found" message).
const FAILURE_OUTPUT_PREFIX_BYTES: usize = 400;

/// Result of [`BackgroundSessions::stop`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StopResult {
    /// The two-stage stop ran (or the process was already dead).
    Stopped,
    /// The session had already finished before the stop arrived — no-op.
    AlreadyFinished,
}

/// Agent-scoped registry of background shell sessions.
///
/// State is owned by the agent (dropped with it) and reachable from the
/// synchronous teardown path for the teardown kill; the per-call tool
/// context carries only an `Arc` handle re-scoped per tool-group invocation.
pub(crate) struct BackgroundSessions {
    inner: std::sync::Mutex<HashMap<PathBuf, SessionEntry>>,
}

impl Default for BackgroundSessions {
    fn default() -> Self {
        Self {
            inner: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

/// One live (or finished) background session, keyed by its output-file path.
struct SessionEntry {
    /// The launched command child, shared with the waiter task (which reaps
    /// it) and the stop/teardown paths (which signal it on Windows). Never
    /// taken out after launch — the waiter polls `try_wait` in place. A
    /// `std::sync::Mutex` is sufficient: every critical section (try_wait,
    /// start_kill) is a short synchronous call and is never held across an
    /// await, so blocking here cannot stall the runtime (and makes the
    /// synchronous teardown kill reliable — no try_lock to fail silently).
    command: Arc<std::sync::Mutex<tokio::process::Child>>,
    /// Watcher child (Unix), reaped by the waiter after the command exits.
    #[cfg(unix)]
    watcher: Option<tokio::process::Child>,
    /// Write end of the daemon→watcher lifeline pipe. Closed (dropped) by
    /// the waiter after the command exits, by teardown, or when the daemon
    /// dies — EOF on the watcher's stdin is what makes it fire.
    #[cfg(unix)]
    write_end: Option<std::os::fd::OwnedFd>,
    /// Process-group id == the watcher's PID (Unix), used for group kills.
    #[cfg(unix)]
    pgid: u32,
    /// The command already exited during the launch probe with a
    /// non-launch-failure status; the waiter reuses this instead of waiting.
    early_status: Option<std::process::ExitStatus>,
    /// Set by the waiter after the exit annotation is appended — the only
    /// completion signal; guards stop/teardown against PID reuse.
    finished: Arc<AtomicBool>,
}

impl BackgroundSessions {
    /// Launch `command` in the background. Returns the output-file path.
    ///
    /// The command is spawned detached from the tool call: stdout and stderr
    /// are redirected (RAW) into the output file, stdin is null (strictly
    /// non-interactive), and the command runs in its own process group whose
    /// lifetime is bounded by the agent (teardown kill) and the watchdog.
    ///
    /// Launch failures (command not found / not executable, detected via the
    /// bounded early-exit probe) are returned as `Err` synchronously — never
    /// a silent empty file.
    pub(crate) async fn launch(
        self: &Arc<Self>,
        command: &str,
        workspace_root: &Path,
    ) -> Result<PathBuf, String> {
        // ── Output file (create_new — never overwrite) ──
        let (output_path, out_file) = create_bg_output_file()
            .map_err(|e| format!("Failed to create background output file: {e}"))?;
        // Owner-deletes-at-end: attribute the output file to the calling agent
        // (the CURRENT_TOOL_AGENT_ID task-local is set during tool execution)
        // so `run_agent`'s end-of-run cleanup removes it alongside the spill
        // files — `terminate_all` kills the process group but leaves the file.
        super::record_spill_owner(output_path.clone());

        // ── Command child ──
        let mut cmd = super::build_shell_command(command, workspace_root);
        let stdout_file = out_file.try_clone().map_err(|e| {
            let _ = std::fs::remove_file(&output_path);
            format!("Failed to set up background output: {e}")
        })?;
        let stderr_file = out_file.try_clone().map_err(|e| {
            let _ = std::fs::remove_file(&output_path);
            format!("Failed to set up background output: {e}")
        })?;
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::from(stdout_file));
        cmd.stderr(Stdio::from(stderr_file));

        #[cfg(unix)]
        let watchdog = match WatchdogSetup::spawn(&mut cmd, &output_path) {
            Ok(w) => w,
            Err(e) => {
                let _ = std::fs::remove_file(&output_path);
                return Err(e);
            }
        };

        let mut cmd_child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                #[cfg(unix)]
                watchdog.abort().await;
                let _ = std::fs::remove_file(&output_path);
                return Err(format!("Failed to start background command: {e}"));
            }
        };

        // ── Bounded early-exit launch probe ──
        // Note: the 126/127 detection is Unix-shell-specific — cmd.exe on
        // Windows exits 1 for a missing/non-executable command, so a failed
        // launch there surfaces as a successful session whose file holds the
        // cmd.exe error and `[exit status: 1]` (documented deviation, see the
        // module docs).
        let early_status = match probe_command(&mut cmd_child).await {
            Ok(Some(status)) if matches!(status.code(), Some(126 | 127)) => {
                let prefix = read_output_prefix(&output_path, FAILURE_OUTPUT_PREFIX_BYTES);
                #[cfg(unix)]
                watchdog.abort().await;
                let _ = std::fs::remove_file(&output_path);
                let prefix_msg = prefix
                    .filter(|p| !p.is_empty())
                    .map(|p| format!("\noutput: {p}"))
                    .unwrap_or_default();
                return Err(format!(
                    "Failed to start background command.\n\
                     command: {command}\n\
                     reason: command not found or not executable (exit status {}).{prefix_msg}",
                    status.code().unwrap_or(-1)
                ));
            }
            Ok(Some(status)) => Some(status),
            Ok(None) => None,
            Err(e) => {
                #[cfg(unix)]
                watchdog.abort().await;
                let _ = std::fs::remove_file(&output_path);
                return Err(e);
            }
        };

        // ── Register the session and hand the reaping off to the waiter ──
        let entry = SessionEntry {
            command: Arc::new(std::sync::Mutex::new(cmd_child)),
            #[cfg(unix)]
            watcher: Some(watchdog.watcher_child),
            #[cfg(unix)]
            write_end: Some(watchdog.write_end),
            #[cfg(unix)]
            pgid: watchdog.pgid,
            early_status,
            finished: Arc::new(AtomicBool::new(false)),
        };
        self.inner
            .lock()
            .unwrap_poison()
            .insert(output_path.clone(), entry);
        self.spawn_waiter(&output_path);
        Ok(output_path)
    }

    /// Two-stage stop of the session whose output file is `output_path`:
    /// SIGTERM → grace (~5s) → SIGKILL to the process group. Stopping an
    /// already-finished session is a no-op guarded by the per-session
    /// finished flag (PID-reuse safety).
    pub(crate) async fn stop(self: &Arc<Self>, output_path: &Path) -> Result<StopResult, String> {
        #[cfg(unix)]
        {
            let (pgid, finished) = {
                let guard = self.inner.lock().unwrap_poison();
                let entry = guard.get(output_path).ok_or_else(|| {
                    format!(
                        "No background session found for output file: {}",
                        output_path.display()
                    )
                })?;
                (entry.pgid, entry.finished.clone())
            };
            if finished.load(Ordering::SeqCst) {
                return Ok(StopResult::AlreadyFinished);
            }
            super::kill_process_group(pgid, libc::SIGTERM);
            tokio::time::sleep(stop_grace()).await;
            // Re-check `finished` before the second stage: if the command
            // exited during the grace (e.g. it died from SIGTERM), the waiter
            // has already appended the annotation and set `finished`, and is
            // about to close the lifeline — the watcher (whose PID is the
            // PGID) will then exit and its PID could be recycled before the
            // SIGKILL lands. `finished == false` here implies the watcher is
            // still alive, so the PGID is still valid. Skipping the SIGKILL
            // also avoids a misleading kill(-pgid) failure warning when the
            // group is already gone.
            if !finished.load(Ordering::SeqCst) {
                super::kill_process_group(pgid, libc::SIGKILL);
            }
            wait_for_finished(&finished, stop_annotation_wait()).await;
            Ok(StopResult::Stopped)
        }
        #[cfg(not(unix))]
        {
            let (command, finished) = {
                let guard = self.inner.lock().unwrap_poison();
                let entry = guard.get(output_path).ok_or_else(|| {
                    format!(
                        "No background session found for output file: {}",
                        output_path.display()
                    )
                })?;
                (entry.command.clone(), entry.finished.clone())
            };
            if finished.load(Ordering::SeqCst) {
                return Ok(StopResult::AlreadyFinished);
            }
            // Windows: no SIGTERM equivalent — terminate the direct child
            // (grandchildren survive; accepted asymmetry).
            let mut g = command.lock().unwrap_poison();
            let _ = g.start_kill();
            drop(g);
            wait_for_finished(&finished, stop_annotation_wait()).await;
            Ok(StopResult::Stopped)
        }
    }

    /// Force-kill every live session (no grace). Called synchronously from
    /// the agent teardown path ([`crate::Agent::drop`]) — the agent's
    /// background processes must not outlive it. Finished sessions are
    /// skipped; their waiters already reaped the processes.
    pub(crate) fn terminate_all(&self) {
        let targets: Vec<_> = {
            let mut guard = self.inner.lock().unwrap_poison();
            guard
                .iter_mut()
                .filter(|(_, e)| !e.finished.load(Ordering::SeqCst))
                .map(|(_, e)| {
                    #[cfg(unix)]
                    {
                        (e.pgid, e.write_end.take())
                    }
                    #[cfg(not(unix))]
                    {
                        (e.command.clone(),)
                    }
                })
                .collect()
        };
        for target in targets {
            #[cfg(unix)]
            {
                super::kill_process_group(target.0, libc::SIGKILL);
                // Closing the lifeline makes the watcher exit (it may already
                // be dead from the group kill — closing is still harmless).
                drop(target.1);
            }
            #[cfg(not(unix))]
            {
                // Blocking lock: every critical section on this mutex is a
                // short synchronous try_wait/start_kill call (never held
                // across an await), so this cannot deadlock — and a try_lock
                // that fails while the waiter polls would silently leak the
                // direct child past teardown (no watchdog on Windows to cover
                // it).
                let mut g = target.0.lock().unwrap_poison();
                let _ = g.start_kill();
            }
        }
    }

    /// Test/observability accessor: whether a session with this output-file
    /// path is registered at all.
    #[cfg(test)]
    pub(crate) fn contains(&self, path: &Path) -> bool {
        self.inner.lock().unwrap_poison().contains_key(path)
    }

    /// Test/observability accessor: whether a registered session has finished
    /// (its exit annotation has been appended).
    #[cfg(test)]
    pub(crate) fn is_finished(&self, path: &Path) -> bool {
        self.inner
            .lock()
            .unwrap_poison()
            .get(path)
            .is_some_and(|e| e.finished.load(Ordering::SeqCst))
    }

    /// Spawn the per-session waiter task: poll the command child until it
    /// exits, append the unconditional exit-status annotation to the output
    /// file, set the finished flag, close the lifeline write end (the watcher
    /// then kills any leftover group members and exits), and reap the
    /// watcher child. Detached — it outlives the agent by a few instants on
    /// teardown to reap the children the teardown kill just signalled.
    fn spawn_waiter(self: &Arc<Self>, output_path: &Path) {
        let sessions = self.clone();
        let output_path = output_path.to_path_buf();
        let (command, finished, early_status) = {
            let guard = self.inner.lock().unwrap_poison();
            let entry = guard.get(&output_path).expect("session just registered");
            (
                entry.command.clone(),
                entry.finished.clone(),
                entry.early_status,
            )
        };
        tokio::spawn(async move {
            // Wait for the command to exit (or reuse the probe's early result).
            let mut status = early_status;
            if status.is_none() {
                loop {
                    let exited = {
                        let mut g = command.lock().unwrap_poison();
                        g.try_wait().ok().flatten()
                    };
                    if let Some(s) = exited {
                        status = Some(s);
                        break;
                    }
                    tokio::time::sleep(WAITER_POLL).await;
                }
            }
            let status = status.expect("background command status determined");

            // The exit-status annotation is the only completion signal —
            // append it unconditionally (including exit 0).
            append_exit_annotation(&output_path, status);
            finished.store(true, Ordering::SeqCst);

            // Close the lifeline and reap the watcher.
            #[cfg(unix)]
            {
                let (watcher, write_end) = {
                    let mut guard = sessions.inner.lock().unwrap_poison();
                    let entry = guard
                        .get_mut(&output_path)
                        .expect("session still registered");
                    (entry.watcher.take(), entry.write_end.take())
                };
                drop(write_end); // EOF → watcher kills leftover group members → exits
                if let Some(mut w) = watcher {
                    let _ = w.wait().await;
                }
            }
        });
    }
}

// ── Unix watchdog setup ──────────────────────────────────────────────

/// Spawned process-group topology for a background session (Unix):
///
/// ```text
/// daemon (mahbot)
///  ├── pipe write end (held by the session entry; CLOEXEC)
///  ├── [watcher child] sh -c WATCHER_SCRIPT   ← group leader (PGID = its PID)
///  │     └─ stdin = pipe read end (CLOEXEC); blocks until EOF, then SIGKILLs
///  │        the group via `kill -s KILL 0`
///  └── [command child] sh -c "command"        ← joins the watcher's group
///        └─ stdout/stderr → output file; stdin → /dev/null
/// ```
#[cfg(unix)]
struct WatchdogSetup {
    watcher_child: tokio::process::Child,
    write_end: std::os::fd::OwnedFd,
    pgid: u32,
}

#[cfg(unix)]
impl WatchdogSetup {
    /// Create the lifeline pipe and spawn the watcher as the group leader,
    /// then point `cmd` at the watcher's process group.
    fn spawn(cmd: &mut tokio::process::Command, output_path: &Path) -> Result<Self, String> {
        let (read_end, write_end) = make_lifeline_pipe().map_err(|e| {
            let _ = std::fs::remove_file(output_path);
            format!("Failed to create background lifeline pipe: {e}")
        })?;
        let mut watcher_cmd = tokio::process::Command::new("sh");
        watcher_cmd
            .arg("-c")
            .arg(WATCHER_SCRIPT)
            .process_group(0)
            .stdin(Stdio::from(read_end))
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let watcher_child = match watcher_cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let _ = std::fs::remove_file(output_path);
                return Err(format!("Failed to spawn background watchdog: {e}"));
            }
        };
        let pgid = watcher_child
            .id()
            .expect("watcher PID available after spawn");
        let pgid_signed: libc::pid_t = pgid.try_into().expect("PGID fits in pid_t");
        cmd.process_group(pgid_signed);
        Ok(Self {
            watcher_child,
            write_end,
            pgid,
        })
    }

    /// Kill the watcher's group, close the lifeline, and reap the watcher.
    /// Used on launch-failure paths (command spawn failure, probe failure).
    async fn abort(self) {
        super::kill_process_group(self.pgid, libc::SIGKILL);
        drop(self.write_end);
        let mut w = self.watcher_child;
        let _ = w.wait().await;
    }
}

/// Create a pipe with both ends `O_CLOEXEC`. The pipe is the daemon→watcher
/// lifeline: it must never leak into the command (or its descendants) or the
/// watcher's own children, or EOF would never arrive.
#[cfg(unix)]
fn make_lifeline_pipe() -> std::io::Result<(std::os::fd::OwnedFd, std::os::fd::OwnedFd)> {
    let mut fds = [0i32; 2];
    // SAFETY: pipe(2) writes two valid fds into the array on success.
    let ret = unsafe { libc::pipe(fds.as_mut_ptr()) };
    if ret != 0 {
        return Err(std::io::Error::last_os_error());
    }
    for fd in fds {
        // SAFETY: fds are valid pipe fds; F_GETFD/F_SETFD are standard.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        if flags < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let ret = unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) };
        if ret != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    // SAFETY: fds[0] and fds[1] are valid, owned pipe fds.
    Ok(unsafe {
        (
            std::os::fd::OwnedFd::from_raw_fd(fds[0]),
            std::os::fd::OwnedFd::from_raw_fd(fds[1]),
        )
    })
}

// ── Shared helpers ───────────────────────────────────────────────────

/// Grace between SIGTERM and SIGKILL in the two-stage stop. Env-overridable
/// for tests.
fn stop_grace() -> Duration {
    crate::util::env_duration_secs("MAHBOT_BG_STOP_GRACE_SECS", STOP_GRACE.as_secs())
}

/// How long `stop()` waits for the waiter to append the exit annotation
/// after the kill. Env-overridable for tests.
fn stop_annotation_wait() -> Duration {
    crate::util::env_duration_secs(
        "MAHBOT_BG_ANNOTATION_WAIT_SECS",
        STOP_ANNOTATION_WAIT.as_secs(),
    )
}

/// Create the output file with create_new semantics in the shared `.agent`
/// temp directory. The name is a distinct `bg_*` shape (flat, outside the
/// `spill_*` namespace) so the existing startup cleanup purges it and spill
/// detection never mistakes it for a spill file. Returns the path and an
/// open handle for the command's stdout/stderr.
fn create_bg_output_file() -> std::io::Result<(PathBuf, File)> {
    let dir = super::agent_temp_dir().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "agent temp dir unavailable")
    })?;
    for _ in 0..MAX_OUTPUT_NAME_ATTEMPTS {
        let path = dir.join(format!("bg_{:04x}.out", rand::random::<u16>()));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a unique background output file name",
    ))
}

/// Bounded early-exit probe: poll the command child for up to [`LAUNCH_PROBE`].
/// `Ok(Some(status))` = the command exited within the window (already
/// reaped); `Ok(None)` = still running after the window (normal launch).
async fn probe_command(
    cmd_child: &mut tokio::process::Child,
) -> Result<Option<std::process::ExitStatus>, String> {
    let deadline = Instant::now() + LAUNCH_PROBE;
    loop {
        match cmd_child.try_wait() {
            Ok(Some(status)) => return Ok(Some(status)),
            Ok(None) => {
                if Instant::now() >= deadline {
                    return Ok(None);
                }
                tokio::time::sleep(POLL_INTERVAL).await;
            }
            Err(e) => {
                return Err(format!("Failed to probe background command: {e}"));
            }
        }
    }
}

/// Append the unconditional exit-status annotation to the output file —
/// the only completion signal for background sessions. Matches the
/// foreground format: `[exit status: N]` / `[exit status: terminated by signal]`.
fn append_exit_annotation(output_path: &Path, status: std::process::ExitStatus) {
    let note = match status.code() {
        Some(c) => format!("[exit status: {c}]"),
        None => "[exit status: terminated by signal]".to_string(),
    };
    match OpenOptions::new().append(true).open(output_path) {
        Ok(mut f) => {
            use std::io::Write;
            let _ = writeln!(f, "\n{note}");
        }
        Err(e) => {
            tracing::warn!(
                path = %output_path.display(),
                err = %e,
                "Failed to append background exit annotation"
            );
        }
    }
}

/// Read up to `max_bytes` from the START of the output file (the shell's
/// launch-error message lives at the start), credential-scrubbed, for the
/// synchronous launch-failure tool error.
fn read_output_prefix(path: &Path, max_bytes: usize) -> Option<String> {
    use std::io::Read;
    let mut f = File::open(path).ok()?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 256];
    while buf.len() < max_bytes {
        let n = f.read(&mut chunk).ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    buf.truncate(max_bytes);
    Some(crate::util::scrub_credentials(&String::from_utf8_lossy(
        &buf,
    )))
}

/// Poll `finished` until it is set or the bound elapses.
async fn wait_for_finished(finished: &AtomicBool, bound: Duration) {
    let deadline = Instant::now() + bound;
    while !finished.load(Ordering::SeqCst) && Instant::now() < deadline {
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::set_env_var;
    use crate::workspace::test_ws;
    use tempfile::TempDir;

    /// Wait for a session's finished flag with a generous bound.
    async fn wait_finished(sessions: &BackgroundSessions, path: &Path, bound: Duration) -> bool {
        let deadline = Instant::now() + bound;
        loop {
            let f = sessions
                .inner
                .lock()
                .unwrap_poison()
                .get(path)
                .is_none_or(|e| e.finished.load(Ordering::SeqCst));
            if f || Instant::now() >= deadline {
                return f;
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    fn read_file(path: &Path) -> String {
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    }

    #[tokio::test]
    async fn launch_quick_exit_appends_annotation() {
        let dir = TempDir::new().expect("tempdir");
        let ws = test_ws(dir.path());
        let sessions = Arc::new(BackgroundSessions::default());

        let path = sessions
            .launch("echo hello-bg", ws.as_path())
            .await
            .expect("launch succeeds");
        assert!(
            path.parent()
                .and_then(|p| p.file_name())
                .is_some_and(|n| n == ".agent"),
            "bg output must be flat in the .agent temp dir: {}",
            path.display()
        );
        assert!(
            path.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("bg_")),
            "bg output must use the bg_* name shape: {}",
            path.display()
        );

        assert!(
            wait_finished(&sessions, &path, Duration::from_secs(10)).await,
            "quick command should finish"
        );
        let out = read_file(&path);
        assert!(out.contains("hello-bg"), "output: {out}");
        assert!(out.contains("[exit status: 0]"), "output: {out}");
    }

    #[tokio::test]
    async fn launch_running_has_no_annotation_until_exit() {
        let dir = TempDir::new().expect("tempdir");
        let ws = test_ws(dir.path());
        let sessions = Arc::new(BackgroundSessions::default());

        let path = sessions
            .launch("sleep 1", ws.as_path())
            .await
            .expect("launch succeeds");

        // Still running shortly after launch: no annotation, not finished.
        tokio::time::sleep(Duration::from_millis(150)).await;
        let out = read_file(&path);
        assert!(!out.contains("[exit status:"), "output before exit: {out}");
        assert!(
            !sessions
                .inner
                .lock()
                .unwrap_poison()
                .get(&path)
                .expect("session registered")
                .finished
                .load(Ordering::SeqCst),
            "should not be finished while running"
        );

        assert!(
            wait_finished(&sessions, &path, Duration::from_secs(10)).await,
            "sleep should finish"
        );
        let out = read_file(&path);
        assert!(out.contains("[exit status: 0]"), "output: {out}");
    }

    // The 126/127 launch-failure probe is Unix-shell-specific (cmd.exe exits 1
    // for these cases); both tests are Unix-gated.
    #[cfg(unix)]
    #[tokio::test]
    async fn launch_failure_command_not_found_errors() {
        let dir = TempDir::new().expect("tempdir");
        let ws = test_ws(dir.path());
        let sessions = Arc::new(BackgroundSessions::default());

        let err = sessions
            .launch("definitely_not_a_command_xyz_123", ws.as_path())
            .await
            .expect_err("unknown command must be a synchronous launch error");
        assert!(
            err.contains("not found or not executable"),
            "error message: {err}"
        );
        assert!(err.contains("127"), "error should mention exit 127: {err}");
        // No session registered, no stray output file left behind.
        assert!(
            sessions.inner.lock().unwrap_poison().is_empty(),
            "no session should be registered after a launch failure"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn launch_failure_not_executable_errors() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().expect("tempdir");
        let ws = test_ws(dir.path());
        let sessions = Arc::new(BackgroundSessions::default());

        // A non-executable script invoked directly → sh exits 126.
        let script = dir.path().join("not-exec.sh");
        std::fs::write(&script, "#!/bin/sh\necho hi\n").expect("write script");
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o644)).expect("chmod");
        let err = sessions
            .launch("./not-exec.sh", ws.as_path())
            .await
            .expect_err("non-executable script must be a synchronous launch error");
        assert!(
            err.contains("not found or not executable"),
            "error message: {err}"
        );
        assert!(err.contains("126"), "error should mention exit 126: {err}");
        assert!(sessions.inner.lock().unwrap_poison().is_empty());
    }

    #[tokio::test]
    async fn launch_legit_nonzero_early_exit_is_successful_launch() {
        let dir = TempDir::new().expect("tempdir");
        let ws = test_ws(dir.path());
        let sessions = Arc::new(BackgroundSessions::default());

        // A legitimate non-zero early exit (grep no-match semantics) is a
        // successful launch surfaced via the annotation.
        let path = sessions
            .launch("exit 3", ws.as_path())
            .await
            .expect("exit 3 is a successful launch");
        assert!(
            wait_finished(&sessions, &path, Duration::from_secs(10)).await,
            "should finish"
        );
        let out = read_file(&path);
        assert!(out.contains("[exit status: 3]"), "output: {out}");
    }

    #[tokio::test]
    async fn stop_kills_running_session() {
        // Grace 0: SIGTERM and SIGKILL land back-to-back — the test does not
        // need to wait out the default 5s grace.
        let _env = set_env_var("MAHBOT_BG_STOP_GRACE_SECS", Some("0"));
        let dir = TempDir::new().expect("tempdir");
        let ws = test_ws(dir.path());
        let sessions = Arc::new(BackgroundSessions::default());

        let path = sessions
            .launch("sleep 30", ws.as_path())
            .await
            .expect("launch succeeds");
        assert!(
            !sessions
                .inner
                .lock()
                .unwrap_poison()
                .get(&path)
                .expect("session")
                .finished
                .load(Ordering::SeqCst)
        );

        let result = sessions.stop(&path).await.expect("stop succeeds");
        assert_eq!(result, StopResult::Stopped);
        assert!(
            wait_finished(&sessions, &path, Duration::from_secs(10)).await,
            "stopped session should finish"
        );
        let out = read_file(&path);
        assert!(
            out.contains("[exit status: terminated by signal]"),
            "output: {out}"
        );
    }

    /// Two-stage stop must not SIGKILL after the command exited during the
    /// grace: a TERM-trapping command that exits on its own within the window
    /// is a clean stop — the post-grace SIGKILL is skipped (via the re-checked
    /// `finished` flag), so there is no recycled-PGID risk and no misleading
    /// kill(-pgid) failure warning.
    #[cfg(unix)]
    #[tokio::test]
    async fn stop_grace_skips_sigkill_after_early_exit() {
        // Short grace so the test does not wait out the default 5s; long
        // enough (1s) that the 0.5s command exits well inside the window and
        // well after the 250ms launch probe.
        let _env = set_env_var("MAHBOT_BG_STOP_GRACE_SECS", Some("1"));
        let dir = TempDir::new().expect("tempdir");
        let ws = test_ws(dir.path());
        let sessions = Arc::new(BackgroundSessions::default());

        // Traps SIGTERM (ignores it), exits 0 on its own after ~0.5s.
        let path = sessions
            .launch("trap '' TERM; sleep 0.5; exit 0", ws.as_path())
            .await
            .expect("launch succeeds");

        let result = sessions.stop(&path).await.expect("stop succeeds");
        assert_eq!(result, StopResult::Stopped);
        assert!(
            wait_finished(&sessions, &path, Duration::from_secs(10)).await,
            "TERM-trapping command should exit during the grace"
        );
        let out = read_file(&path);
        assert!(
            out.contains("[exit status: 0]"),
            "the command must be allowed to exit 0 during the grace, not SIGKILLed: {out}"
        );
        assert!(
            !out.contains("[exit status: terminated by signal]"),
            "the post-grace SIGKILL must be skipped once the waiter finished: {out}"
        );
    }

    #[tokio::test]
    async fn stop_already_finished_is_noop() {
        let dir = TempDir::new().expect("tempdir");
        let ws = test_ws(dir.path());
        let sessions = Arc::new(BackgroundSessions::default());

        let path = sessions
            .launch("echo quick", ws.as_path())
            .await
            .expect("launch succeeds");
        assert!(
            wait_finished(&sessions, &path, Duration::from_secs(10)).await,
            "should finish"
        );
        let result = sessions.stop(&path).await.expect("stop is a no-op");
        assert_eq!(result, StopResult::AlreadyFinished);
    }

    #[tokio::test]
    async fn stop_unknown_path_errors() {
        let dir = TempDir::new().expect("tempdir");
        let sessions = Arc::new(BackgroundSessions::default());

        let err = sessions
            .stop(&dir.path().join(".agent/bg_0000.out"))
            .await
            .expect_err("unknown session must error");
        assert!(
            err.contains("No background session found"),
            "error message: {err}"
        );
    }

    #[tokio::test]
    async fn terminate_all_kills_running_sessions() {
        let dir = TempDir::new().expect("tempdir");
        let ws = test_ws(dir.path());
        let sessions = Arc::new(BackgroundSessions::default());

        let p1 = sessions
            .launch("sleep 30", ws.as_path())
            .await
            .expect("launch 1");
        let p2 = sessions
            .launch("sleep 30", ws.as_path())
            .await
            .expect("launch 2");
        assert_eq!(sessions.inner.lock().unwrap_poison().len(), 2);

        sessions.terminate_all();

        assert!(
            wait_finished(&sessions, &p1, Duration::from_secs(10)).await,
            "session 1 killed by teardown"
        );
        assert!(
            wait_finished(&sessions, &p2, Duration::from_secs(10)).await,
            "session 2 killed by teardown"
        );
        let o1 = read_file(&p1);
        assert!(
            o1.contains("[exit status: terminated by signal]"),
            "output 1: {o1}"
        );
    }

    /// The watchdog: closing the daemon-side write end (simulating a hard
    /// daemon crash — SIGKILL/SIGSEGV leave no chance to run destructors)
    /// must make the watcher kill the whole process group.
    #[cfg(unix)]
    #[tokio::test]
    async fn watchdog_kills_group_when_write_end_closes() {
        let dir = TempDir::new().expect("tempdir");
        let ws = test_ws(dir.path());
        let sessions = Arc::new(BackgroundSessions::default());

        let path = sessions
            .launch("sleep 30", ws.as_path())
            .await
            .expect("launch succeeds");

        // Take the write end out of the session entry and drop it — the
        // daemon "died". The watcher gets EOF and must SIGKILL the group.
        let write_end = sessions
            .inner
            .lock()
            .unwrap_poison()
            .get_mut(&path)
            .expect("session registered")
            .write_end
            .take()
            .expect("write end present while running");
        drop(write_end);

        assert!(
            wait_finished(&sessions, &path, Duration::from_secs(10)).await,
            "watchdog should kill the command when the daemon side dies"
        );
        let out = read_file(&path);
        assert!(
            out.contains("[exit status: terminated by signal]"),
            "output: {out}"
        );
    }

    /// A command that exits while leaving a stray grandchild in the group
    /// (`sh -c 'sleep 5 &'` — the foreground path's drain-timeout scenario)
    /// must not leak the stray: the waiter closes the lifeline, the watcher
    /// SIGKILLs the group, and the stray dies with the session.
    #[cfg(unix)]
    #[tokio::test]
    async fn stray_grandchild_killed_when_command_exits() {
        let dir = TempDir::new().expect("tempdir");
        let ws = test_ws(dir.path());
        let sessions = Arc::new(BackgroundSessions::default());
        let pid_file = dir.path().join("stray.pid");
        let cmd = format!("sleep 5 & echo $! > {}", pid_file.display());

        let path = sessions
            .launch(&cmd, ws.as_path())
            .await
            .expect("launch succeeds — sh exits 0 even with a stray");

        assert!(
            wait_finished(&sessions, &path, Duration::from_secs(10)).await,
            "session finishes when the launched command exits"
        );
        let out = read_file(&path);
        assert!(out.contains("[exit status: 0]"), "output: {out}");

        // The stray must be dead (watcher SIGKILLed the group on lifeline EOF).
        let pid: i32 = std::fs::read_to_string(&pid_file)
            .expect("stray pid file")
            .trim()
            .parse()
            .expect("valid pid");
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut alive = true;
        while Instant::now() < deadline {
            // SAFETY: kill(pid, 0) checks existence without sending a signal.
            if unsafe { libc::kill(pid, 0) } != 0 {
                alive = false;
                break;
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
        assert!(
            !alive,
            "stray grandchild (pid={pid}) must die with the session"
        );
    }
}
