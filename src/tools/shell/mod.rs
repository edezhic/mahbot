use crate::{Tool, Workspace};
use async_trait::async_trait;
use directories::UserDirs;
use regex::RegexSet;
use serde_json::json;
use std::collections::HashSet;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
#[cfg(windows)]
use std::sync::OnceLock;
use std::time::Duration;

use super::listing::{ListingEntry, format_listing, human_readable_size};
use crate::util::TOOL_OUTPUT_BUDGET_BYTES;
use crate::util::UnwrapPoison;
use crate::util::scrub_credentials;
use crate::util::strip_ansi_escapes;

mod bg;
pub(crate) mod grep_engine;
mod profiles;
mod readonly;
mod scan;
mod tree;

pub(crate) use self::bg::BackgroundSessions;
use self::profiles::{CARGO_COMPILE_PREFIXES, GEN_FALLBACK, PROFILES, Profile};
pub use self::readonly::ShellMode;
use self::readonly::check_command;
use self::tree::{RunOwner, Tree};

/// The shell that runs a validated command string (`sh -c` on unix,
/// `cmd.exe /C` on Windows). Every platform rule in this module tree reads
/// this one value — the spawn side ([`build_shell_command`]), the read-only
/// guard's tables and the grep engine's command model — as a runtime value
/// rather than a `cfg` branch, so both platforms' behaviour is drivable from
/// any host's unit-test lane. Only the value's own definition branches on the
/// target ([`SHELL_PLATFORM`] is one `if cfg!(windows)` constant); every
/// consumer takes it as data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ShellPlatform {
    Unix,
    Windows,
}

/// The running process's [`ShellPlatform`]; its spawn side is
/// [`build_shell_command`] — the two must not drift.
pub(super) const SHELL_PLATFORM: ShellPlatform = if cfg!(windows) {
    ShellPlatform::Windows
} else {
    ShellPlatform::Unix
};

/// The Windows command interpreter, named once so the spawn side
/// ([`build_shell_command`]) and the GUI Shell page's terminal
/// (`gui::shell::terminal_program`) cannot start different programs.
pub(crate) const WINDOWS_COMMAND_INTERPRETER: &str = "cmd.exe";

/// Shell builtins/prefixes to skip when extracting the primary command.
/// NOTE: `su` is intentionally NOT in this list. It can be used to run
/// commands as another user (e.g., `su -c "rm -rf /"`), which would bypass
/// the read-only validation. This is an accepted gap — adding `su` would be
/// a half-measure since there are countless other privilege-escalation
/// vectors, and the read-only shell is a best-effort safety layer, not a
/// sandbox.
pub(super) const SHELL_PREFIXES: &[&str] = &[
    "cd",
    "pushd",
    "popd",
    "export",
    "source",
    ".",
    "sudo",
    "time",
    // `!` negation: needed so the raw first-command-word scan skips it
    // (`time ! rm -rf ./x` must dispatch `rm` to the blocklist). Side effect
    // on shared consumers: `canonical_command` routes `! ls` to the ls
    // profile — safe direction, profile selection only.
    "!",
    "command",
    "builtin",
    "env",
    "nohup",
    "exec",
    "nice",
    "noglob",
    "nocorrect",
    "eval",
    "npx",
];

/// Corresponding entries in [`SHELL_PREFIXES`] that do NOT forward their
/// arguments as a command — they change shell state internally. These are
/// excluded from delegation-based tests because they don't execute their
/// arguments.
#[cfg(test)]
pub(super) const NON_DELEGATING_PREFIXES: &[&str] =
    &["cd", "pushd", "popd", "export", "source", "."];

/// Git global flags that may appear between `git` and its subcommand.
///
/// **IMPORTANT**: Only include flags that take a space-separated value argument.
/// Boolean flags like `--bare` MUST NOT be listed here — `find_first_non_flag_index`
/// skips 2 words (flag + value) for each entry, causing boolean flags to consume
/// the subcommand as their "value" and bypass read-only validation entirely.
const GIT_GLOBAL_FLAGS: &[&str] = &["-C", "--git-dir", "--work-tree", "-c"];

/// Default maximum shell command execution time before kill.
const DEFAULT_SHELL_TIMEOUT_SECS: u64 = 600;
/// Absolute maximum allowed shell command timeout (1 hour).
/// Prevents agents from setting absurdly long timeouts.
const MAX_SHELL_TIMEOUT_SECS: u64 = 3600;
/// Default bound on the post-exit output drain (seconds). After the main
/// command exited, remaining output must drain within this bound — a leftover
/// backgrounded process holding the pipes open otherwise blocks EOF forever.
const DEFAULT_OUTPUT_DRAIN_TIMEOUT_SECS: u64 = 10;
/// Grace window given to pipe readers to notice cancellation and return
/// buffered partial output after a process-group kill.
const DRAIN_CANCEL_GRACE: Duration = Duration::from_secs(2);
/// Cap bytes collected from each pipe during command execution (including timeouts).
///
/// # Truncation safety
///
/// This cap (256 KB) is well below 1 MB, so the [`decode_and_strip_ansi`]
/// output never reaches a size where truncation would be necessary. If this
/// cap is ever raised above 1 MB, re-add truncation safeguards (e.g.,
/// [`truncate_sandwich`](crate::util::truncate_sandwich) after ANSI stripping)
/// to prevent unbounded output.
const SHELL_PIPE_READ_CAP: usize = 256 * 1024;
/// Max chars of partial output included in timeout error messages.
const TIMEOUT_OUTPUT_TAIL_CHARS: usize = 2_000;
/// Max chars of the engine's own stderr line quoted as the cause of a refused
/// Windows search (see [`engine_cause`]).
const ENGINE_FAILURE_DETAIL_CHARS: usize = 200;

/// Environment variables safe to pass to shell commands.
///
/// Only functional variables are included — never API keys or secrets. The
/// platform's temp variables are NOT listed here: they are bound to the daemon's
/// private temp root by [`apply_safe_env`] from
/// [`crate::temp::shell_temp_vars`].
#[cfg(not(target_os = "windows"))]
const SAFE_ENV_VARS: &[&str] = &[
    "PATH", "HOME", "TERM", "LANG", "LC_ALL", "LC_CTYPE", "USER", "SHELL",
];

/// Environment variables safe to pass to shell commands on Windows.
///
/// Includes Windows-specific variables needed for cmd.exe and program
/// resolution. The temp variables are not listed here: [`apply_safe_env`] binds
/// them to the daemon's private temp root.
#[cfg(target_os = "windows")]
const SAFE_ENV_VARS: &[&str] = &[
    "PATH",
    "PATHEXT",
    "HOME",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "TERM",
    "LANG",
    "USERNAME",
];

/// Clear the child's environment and re-populate it from [`SAFE_ENV_VARS`], plus
/// the platform's temp variables bound to the daemon's private temp root.
///
/// The temp names and value come from [`crate::temp::shell_temp_vars`] — the
/// same pair the read-only guard's temp model reads — so the scratch location a
/// child actually writes to and the location the guard accepts for a write
/// cannot drift apart, and a new temp name reaches both at once.
pub(crate) fn apply_safe_env(cmd: &mut tokio::process::Command) {
    cmd.env_clear();
    for &name in SAFE_ENV_VARS {
        if let Some(value) = baseline_env_value(name) {
            cmd.env(name, value);
        }
    }
    for (name, value) in crate::temp::shell_temp_vars() {
        cmd.env(name, value);
    }
}

/// Windows: create the child without a console window.
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Build a [`tokio::process::Command`] for executing a shell command in the
/// workspace root. The environment is cleared and re-populated from
/// [`SAFE_ENV_VARS`] only — no parent-process environment is inherited. This
/// prevents leaking API keys and other secrets into subprocesses (CWE-200).
///
/// **Note:** `$USER`/`$USERNAME` and the Windows system-path variables
/// (`%SystemRoot%`/`%WINDIR%`, `%SystemDrive%`, `%ComSpec%`) are intentional
/// exceptions — read from the parent process (usernames and system paths are
/// not secrets).
///
/// On Unix the child is made a process group leader (via
/// [`process_group(0)`](tokio::process::Command::process_group)), so that the
/// timeout handler in [`run_command_with_timeout`] can kill the entire
/// subprocess tree (child + grandchildren) with a single PGID signal.
/// Grandchildren (e.g., `cargo test` or long-running `sleep`) inherit the new
/// PGID from `sh`, preventing orphaned CPU-consuming process trees when a
/// shell command times out.
///
/// On Windows the child is what the runner puts under a job object right after
/// the spawn — the platform's own whole-tree mechanism ([`tree`]).
fn build_shell_command(command: &str, workspace_root: &Path) -> tokio::process::Command {
    // The spawn side of [`SHELL_PLATFORM`]; the two must not drift.
    #[cfg(not(target_os = "windows"))]
    let mut process = {
        let mut p = tokio::process::Command::new("sh");
        p.arg("-c").arg(command);
        // Make this child a process group leader so grandchildren inherit the
        // PGID and can be killed together on timeout (see run_command_with_timeout).
        #[cfg(unix)]
        {
            p.process_group(0);
        }
        p
    };

    #[cfg(target_os = "windows")]
    let mut process = {
        let mut p = tokio::process::Command::new(WINDOWS_COMMAND_INTERPRETER);
        // `raw_arg`, not `arg`: std's argument escaping belongs to the
        // `CommandLineToArgvW` convention `cmd.exe` does not follow (see
        // `Command::raw_arg`'s own doc). It would re-escape the command's quotes
        // to `\"`, which cmd.exe keeps literally — it has no backslash escape —
        // so a command whose first word is a quoted path (every engine rewrite
        // is, and an install under `C:\Program Files\…` needs it) would name a
        // program cmd.exe cannot resolve.
        //
        // The command goes inside the extra quote pair cmd.exe documents for
        // this hand-off (`cmd /?`, "the remainder of the command line after the
        // switch"): its quote processing strips that pair, so the command
        // arrives verbatim — whatever quotes it carries of its own.
        p.raw_arg(format!("/C \"{command}\""));
        p.creation_flags(CREATE_NO_WINDOW);
        p
    };

    finalize_command(&mut process, workspace_root);
    process
}

/// Build a [`tokio::process::Command`] that runs `program` with `args` as
/// argv — no shell in between, so no argument can ever be reinterpreted as
/// shell syntax (unlike [`build_shell_command`], whose string is parsed by
/// `sh -c`). Containment is otherwise identical: the workspace root as cwd,
/// a cleared environment re-populated from [`SAFE_ENV_VARS`], and (Unix) the
/// child leading its own process group; on Windows the runner's job is the
/// platform's side of that containment ([`tree`]).
fn build_program_command(
    program: &Path,
    args: &[String],
    workspace_root: &Path,
) -> tokio::process::Command {
    let mut process = tokio::process::Command::new(program);
    process.args(args);
    #[cfg(unix)]
    {
        process.process_group(0);
    }
    #[cfg(target_os = "windows")]
    process.creation_flags(CREATE_NO_WINDOW);
    finalize_command(&mut process, workspace_root);
    process
}

/// Shared command setup: working directory + sanitized environment.
fn finalize_command(process: &mut tokio::process::Command, workspace_root: &Path) {
    process.current_dir(workspace_root);
    apply_safe_env(process);
}

/// Outcome of a timed shell subprocess run.
#[derive(Debug)]
enum ShellRunResult {
    Completed {
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        status: std::process::ExitStatus,
        elapsed: Duration,
    },
    TimedOut {
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        pid: Option<u32>,
        elapsed: Duration,
    },
    /// The main process exited but leftover processes kept the output pipes
    /// open past the drain bound, so EOF never arrived. `pid` is the containment
    /// root's when the containment ended the tree, and `None` otherwise — the
    /// error names it as killed only in the former case.
    DrainTimedOut {
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        pid: Option<u32>,
        elapsed: Duration,
    },
    SpawnFailed(std::io::Error),
}

/// Read from an async stream up to `cap` bytes, then continue reading and
/// discarding any remaining data to drain the pipe (preventing back-pressure
/// on the child process from a full pipe buffer).
///
/// Stops early when `cancel` is signalled, returning whatever has been read
/// so far. This allows the timeout path to collect partial output even when
/// grandchild processes inherited the pipe write end and prevent EOF.
async fn read_stream_limited(
    reader: &mut (impl tokio::io::AsyncRead + Unpin),
    cap: usize,
    cancel: tokio_util::sync::CancellationToken,
) -> Vec<u8> {
    use tokio::io::AsyncReadExt;

    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        let to_read = if buf.len() >= cap {
            chunk.len() // drain mode — read and discard to prevent back-pressure
        } else {
            (cap - buf.len()).min(chunk.len())
        };

        tokio::select! {
            biased;
            result = reader.read(&mut chunk[..to_read]) => {
                match result {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if buf.len() < cap {
                            let take = n.min(cap - buf.len());
                            buf.extend_from_slice(&chunk[..take]);
                        }
                        // In drain mode (buf.len() >= cap): chunk data is discarded
                        // to prevent the child from blocking on a full pipe buffer.
                    }
                }
            }
            () = cancel.cancelled() => break,
        }
    }
    buf
}

/// Spawn a background task that reads from a pipe.
/// The task stops early when `cancel` is signalled and returns
/// whatever data has been buffered so far.
fn spawn_pipe_reader(
    pipe: impl tokio::io::AsyncRead + Unpin + Send + 'static,
    cancel: tokio_util::sync::CancellationToken,
) -> tokio::task::JoinHandle<Vec<u8>> {
    tokio::spawn(async move {
        let mut pipe = pipe;
        read_stream_limited(&mut pipe, SHELL_PIPE_READ_CAP, cancel).await
    })
}

/// Await a pipe reader task with a grace timeout, used after killing a child
/// process — the reader is given `cancellation_timeout` to notice cancellation
/// and return whatever data it has buffered so far.
async fn await_pipe_reader_with_cancellation_timeout(
    handle: tokio::task::JoinHandle<Vec<u8>>,
    label: &str,
    cancellation_timeout: Duration,
) -> Vec<u8> {
    tokio::time::timeout(cancellation_timeout, handle)
        .await
        .ok()
        .and_then(std::result::Result::ok)
        .unwrap_or_else(|| {
            tracing::warn!(
                "{label} reader did not respond to cancellation within {cancellation_timeout:?}"
            );
            Vec::new()
        })
}

/// Finish a pipe reader after a drain timeout: an already-completed reader
/// (its output is `Some`) keeps its data — never re-awaited, tokio panics on
/// JoinHandle re-poll — while a still-pending one gets the cancellation grace
/// bound.
async fn finish_partial_reader(
    partial: Option<Vec<u8>>,
    handle: tokio::task::JoinHandle<Vec<u8>>,
    label: &str,
) -> Vec<u8> {
    match partial {
        Some(data) => data,
        None => {
            await_pipe_reader_with_cancellation_timeout(handle, label, DRAIN_CANCEL_GRACE).await
        }
    }
}

/// Output-drain bound for [`run_command_with_timeout`]: after the main command
/// exits, remaining output must drain within this bound. Overridable via env
/// for tuning; tests pass explicit durations.
fn output_drain_timeout() -> Duration {
    crate::util::env_duration_secs(
        "MAHBOT_SHELL_DRAIN_TIMEOUT_SECS",
        DEFAULT_OUTPUT_DRAIN_TIMEOUT_SECS,
    )
}

/// Signal the process group of a spawned shell (PGID == the child PID after
/// `process_group(0)` in [`build_shell_command`]). Used to terminate the whole
/// subprocess tree when a leftover backgrounded process keeps the output pipes
/// open after the main process exited (drain timeout) or was killed (command
/// timeout) — prevents orphaned pipe-holding strays from accumulating.
///
/// Also used by [`self::bg`] for the two-stage background-session stop
/// (SIGTERM then SIGKILL) and the teardown kill; there the target is the
/// background watcher's PID (the group leader), which stays alive for the
/// whole session, so the group always exists when the signal is sent. The unix
/// arm of [`tree`]'s containment is this same call.
///
/// Note on ordering: the timeout path deliberately kills before reaping the
/// child (see [`run_command_with_timeout`]) to avoid a PID-reuse race; the
/// drain path kills after `child.wait()` already reaped it, so the PID could
/// in theory be recycled as a new group leader within the drain window —
/// accepted risk, the kill is best-effort.
#[cfg(unix)]
fn kill_process_group(pid: u32, signal: libc::c_int) {
    let pid_signed: libc::pid_t = pid.try_into().expect("PID fits in pid_t");
    // SAFETY: kill(-pgid, sig) is the standard POSIX way to signal an entire
    // process group. The target PGID is our own child's PID (also its
    // PGID after process_group(0)), so we own every process in the group.
    let ret = unsafe { libc::kill(-pid_signed, signal) };
    if ret != 0 {
        let err = std::io::Error::last_os_error();
        tracing::warn!(
            pid = pid,
            signal,
            err = %err,
            "kill(-pgid) failed — leftover processes may survive"
        );
    }
}

/// The lifecycle point of the grep engine's scratch spec files: Windows serves a
/// member's spec through a file (cmd.exe cannot carry the payload on its command
/// line), and every exit path from the serve decision — including the validation
/// `bail!`s and a cancellation drop — must clean it up. The removal itself, and
/// its best-effort policy, are the engine's ([`grep_engine::discard_spec_files`],
/// which the engine also calls inline when a hand-off abandons a rewrite); this
/// guard owns only the point. A killed daemon still leaves a file behind; the
/// temp cleaner is the backstop.
struct SpecFiles(Vec<PathBuf>);

impl Drop for SpecFiles {
    fn drop(&mut self) {
        grep_engine::discard_spec_files(&self.0);
    }
}

/// The refusal this shell call must fail with, if any: the serve decision's own
/// refusal ([`grep_engine::GrepServe::refusal`] — a Windows-only value), or a
/// produced-but-unapplied rewrite, i.e. one the read-only guard rejected.
/// Returns the short cause; the agent-facing message is rendered from it by
/// [`grep_engine::unserved_failure`] at the bail site, the way the engine's own
/// cause is.
///
/// `applied` is whether the rewrite reached the shell (`exec_str !=
/// command_str`). The platform is tested first: on unix neither case refuses —
/// the original command runs and its `grep` is the real one — while on Windows
/// an unserved search must not come back looking like an empty result (see
/// [`grep_engine::unserved_failure`]).
fn unserved_refusal(
    platform: ShellPlatform,
    grep_serve: &grep_engine::GrepServe,
    applied: bool,
) -> Option<String> {
    if platform != ShellPlatform::Windows {
        return None;
    }
    grep_serve.refusal.clone().or_else(|| {
        (!applied && grep_serve.rewritten.is_some()).then(|| GUARD_REJECTED_REASON.to_string())
    })
}

/// The grep telemetry cause for a rewrite that was produced but never applied
/// (the read-only guard rejected it), spelled once so the refusal's cause and
/// the stored row's reason cannot drift apart.
const GUARD_REJECTED_REASON: &str = "read-only guard rejected rewrite";

/// What a completed run's engine failure means to this platform.
enum EngineFailure {
    /// The engine could not serve and reported no account to render: the
    /// original command is re-run, so the agent gets the authentic answer (the
    /// platform with a real `grep` to re-run).
    ReRun,
    /// The engine's own short account of the failure (the platform that refuses
    /// the call instead of re-running it).
    Refused(String),
}

/// Classify a completed run: `Some` when it is the engine failing to serve —
/// the sentinel exit code, the stale-binary lock message, or, on the platform
/// that refuses rather than re-runs, the engine's refusal marker on a line of
/// its own, which survives a pipeline tail masking the exit status. `None` when
/// the run is not an engine failure.
///
/// The marker is recognised by line equality, never as a substring: a served
/// run's stderr may echo the token inside a matched line, and a result is not a
/// refusal. Only the refusing platform carries a cause; on the other one the
/// marker is not even scanned and [`EngineFailure::ReRun`] says all that
/// platform's caller needs.
fn engine_failure(
    status_code: Option<i32>,
    stderr: &[u8],
    platform: ShellPlatform,
) -> Option<EngineFailure> {
    let text = String::from_utf8_lossy(stderr);
    let marked = platform == ShellPlatform::Windows
        && text
            .lines()
            .any(|line| line.trim() == grep_engine::ENGINE_REFUSAL_MARKER);
    if status_code != Some(grep_engine::ENGINE_FAILED_EXIT)
        && !text.contains(grep_engine::STALE_BINARY_LOCK_MSG)
        && !marked
    {
        return None;
    }
    if platform != ShellPlatform::Windows {
        return Some(EngineFailure::ReRun);
    }
    Some(EngineFailure::Refused(engine_cause(&text, marked)))
}

/// The engine's own short account of a failure it reported: the detail it wrote
/// after the refusal marker, which it emits first — so this is the engine's own
/// line and never a served member's stderr earlier in the stream — else the
/// run's first non-blank line (a stale binary's lock message), else a generic
/// cause for a failure that carried none. Trimmed and length-capped;
/// [`grep_engine::unserved_failure`] renders the agent-facing message from it.
fn engine_cause(text: &str, marked: bool) -> String {
    let mut lines = text.lines().map(str::trim);
    let detail = if marked {
        lines
            .by_ref()
            .skip_while(|line| *line != grep_engine::ENGINE_REFUSAL_MARKER)
            .nth(1)
    } else {
        lines.find(|line| !line.is_empty())
    };
    detail.map_or_else(
        || "engine could not serve the search".to_string(),
        |line| crate::util::truncate(line, ENGINE_FAILURE_DETAIL_CHARS),
    )
}

/// Ends a run's whole process tree if the run is dropped before a stop path or
/// the successful-completion path has taken responsibility for it — an expired
/// drain cap aborting the task, a panic in a sibling tool, runtime teardown. The
/// tree is the platform's ([`Tree`]): the child's process group on unix, the job
/// on Windows.
///
/// It must be disarmed on every path where the child is reaped or the tree is
/// handed on: after a unix child is reaped a group kill risks PID reuse, and a
/// Windows job retained for the process lifetime must not be ended by this guard.
///
/// It holds only the tree, never the child, so it has no fallback when
/// [`Tree::terminate`] reports `false` — a Windows run with no job (fail-open,
/// see [`tree`]) kills nothing here, just as it did before this guard existed.
struct KillOnDrop {
    tree: Tree,
    armed: bool,
}

impl KillOnDrop {
    fn new(tree: Tree) -> Self {
        Self { tree, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.tree.terminate();
        }
    }
}

/// Result of draining both pipe readers within a bound.
enum DrainOutcome {
    /// Both readers hit EOF within the bound — full output.
    Both(Vec<u8>, Vec<u8>),
    /// The bound was exceeded (a leftover process holds a pipe open): each
    /// field is `Some` only when that reader already completed. The caller
    /// must not re-await a completed handle (tokio panics on re-poll); only
    /// the `None` side is still pending.
    Partial {
        stdout: Option<Vec<u8>>,
        stderr: Option<Vec<u8>>,
    },
}

/// Drain both pipe readers with a bound — see [`DrainOutcome`].
async fn drain_pipe_readers(
    mut stdout_handle: &mut tokio::task::JoinHandle<Vec<u8>>,
    mut stderr_handle: &mut tokio::task::JoinHandle<Vec<u8>>,
    drain_limit: Duration,
) -> DrainOutcome {
    let drain = tokio::time::sleep(drain_limit);
    tokio::pin!(drain);
    let mut stdout_done = None;
    let mut stderr_done = None;
    loop {
        tokio::select! {
            biased;
            r = &mut stdout_handle, if stdout_done.is_none() => {
                stdout_done = Some(r.unwrap_or_else(|e| {
                    tracing::warn!(%e, "stdout reader task panicked");
                    Vec::new()
                }));
            }
            r = &mut stderr_handle, if stderr_done.is_none() => {
                stderr_done = Some(r.unwrap_or_else(|e| {
                    tracing::warn!(%e, "stderr reader task panicked");
                    Vec::new()
                }));
            }
            () = &mut drain => break,
        }
        if stdout_done.is_some() && stderr_done.is_some() {
            break;
        }
    }
    match (stdout_done, stderr_done) {
        (Some(stdout), Some(stderr)) => DrainOutcome::Both(stdout, stderr),
        (stdout, stderr) => DrainOutcome::Partial { stdout, stderr },
    }
}

/// Spawn `cmd`, read stdout/stderr concurrently, and enforce `timeout`.
/// After the main process exits, remaining output must drain within
/// `drain_limit` — a leftover backgrounded process holding the pipes open
/// turns the drain into a bounded [`ShellRunResult::DrainTimedOut`] instead
/// of an indefinite hang.
///
/// `owner` decides the run's whole-tree containment ([`tree`]): unix contains
/// every run the same way — the process group the child leads — while Windows
/// gives a job object only to an agent's run.
async fn run_command_with_timeout(
    cmd: &mut tokio::process::Command,
    timeout: Duration,
    drain_limit: Duration,
    owner: RunOwner,
) -> ShellRunResult {
    let start = std::time::Instant::now();

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    // The run's containment, created before the spawn so the child can be put
    // under it the moment it exists; the window between the two is the accepted
    // limitation `tree` documents.
    let mut tree = Tree::new(owner);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return ShellRunResult::SpawnFailed(e),
    };

    let pid = child.id();
    if let Some(pid) = pid {
        tree.attach(pid);
    }
    // Kill-on-drop: an aborted task (drain-cap force-cancel, panic, runtime
    // teardown) must not orphan the run's tree. Disarmed on every path where
    // the child is reaped or the tree is handed on.
    let mut kill_guard = KillOnDrop::new(tree.clone());
    // Stdio::piped() was set above, so the handles are always present.
    let stdout_pipe = child.stdout.take().expect("stdout piped");
    let stderr_pipe = child.stderr.take().expect("stderr piped");

    let cancel = tokio_util::sync::CancellationToken::new();
    let mut stdout_handle = spawn_pipe_reader(stdout_pipe, cancel.clone());
    let mut stderr_handle = spawn_pipe_reader(stderr_pipe, cancel.clone());

    match tokio::time::timeout(timeout, child.wait()).await {
        Ok(Ok(status)) => {
            // Child reaped — the guard must not fire: on unix the group leader
            // is gone and a post-reap kill would risk PID reuse, while a Windows
            // job retained for the process lifetime must not be ended here.
            kill_guard.disarm();
            // Child exited naturally — drain remaining output with a bound.
            // A leftover backgrounded process that inherited the pipes prevents
            // EOF; without the bound the drain would hang the tool forever.
            match drain_pipe_readers(&mut stdout_handle, &mut stderr_handle, drain_limit).await {
                DrainOutcome::Both(stdout, stderr) => {
                    // The run ended on its own — not a stop, so nothing is killed
                    // here and the job is kept rather than closed (`tree`).
                    tree.retain_after_completion();
                    ShellRunResult::Completed {
                        stdout,
                        stderr,
                        status,
                        elapsed: start.elapsed(),
                    }
                }
                DrainOutcome::Partial { stdout, stderr } => {
                    // Drain bound exceeded: a leftover process still holds the
                    // pipes. End the tree the run is contained in — the process
                    // group on unix, the job on Windows — mirroring the timeout
                    // path, then cancel the readers and collect partial output so
                    // the caller gets a visible, recoverable error instead of
                    // a hang. Readers that already completed keep their output
                    // (a completed JoinHandle must not be re-awaited).
                    let ended = tree.terminate();
                    cancel.cancel();
                    let (stdout, stderr) = tokio::join!(
                        finish_partial_reader(stdout, stdout_handle, "stdout"),
                        finish_partial_reader(stderr, stderr_handle, "stderr"),
                    );
                    ShellRunResult::DrainTimedOut {
                        stdout,
                        stderr,
                        // Named as killed only when the tree really was ended (see
                        // the variant).
                        pid: pid.filter(|_| ended),
                        elapsed: start.elapsed(),
                    }
                }
            }
        }
        Ok(Err(e)) => ShellRunResult::SpawnFailed(e),
        Err(_) => {
            // Kill the entire process tree (child + grandchildren) on timeout:
            // the run's containment, plus a signal to the direct child itself
            // first, so the reap below cannot wait on a child the containment
            // termination failed to reach.
            //
            // Order matters on unix: `start_kill` (fire-and-forget SIGKILL)
            // precedes the group kill, because `child.kill().await` — which
            // includes `wait()` — would let the child's PID be reused before we
            // could signal the group it leads (`process_group(0)` in
            // [`build_shell_command`]). On Windows the direct kill is the
            // fallback for a run whose containment could not be established (see
            // `tree`) and for a job termination that failed.
            let _ = child.start_kill();
            tree.terminate();
            let _ = child.wait().await;
            // Reaped — disarm the guard (the explicit kill already ran).
            kill_guard.disarm();
            cancel.cancel();

            // Give readers a grace window to notice cancellation and return
            // their buffers. If a reader takes longer than 2 s (e.g. because
            // a grandchild keeps the pipe open), we still get partial data
            // from the buffer it returns after noticing cancellation.
            let stdout = await_pipe_reader_with_cancellation_timeout(
                stdout_handle,
                "stdout",
                DRAIN_CANCEL_GRACE,
            )
            .await;
            let stderr = await_pipe_reader_with_cancellation_timeout(
                stderr_handle,
                "stderr",
                DRAIN_CANCEL_GRACE,
            )
            .await;
            ShellRunResult::TimedOut {
                stdout,
                stderr,
                pid,
                elapsed: start.elapsed(),
            }
        }
    }
}

/// Outcome of a direct program run (argv — never a shell command string), used
/// by trigger-armed alarms to decide whether a check reported anything.
///
/// `output` is the combined stdout+stderr AFTER ANSI stripping, but WITHOUT
/// credential scrubbing and WITHOUT the shell pipeline's annotations (the
/// exit-status note, timing, spill hints). `has_output` is the wake/no-wake
/// signal and reads the streams themselves: output that is nothing but
/// whitespace or escape sequences is still output.
pub(crate) struct ProgramOutcome {
    /// Whether the program exited with status 0.
    pub success: bool,
    /// How the run ended, e.g. `exit status 2` or `timed out after 600s`.
    pub detail: String,
    /// Combined stdout+stderr, ANSI-stripped, unscrubbed.
    pub output: String,
    /// Whether the program emitted anything at all.
    pub has_output: bool,
}

/// Build the outcome for a run that produced (possibly partial) raw streams.
/// A run with no streams at all (spawn failure) yields empty, no-output.
fn program_outcome(success: bool, detail: String, stdout: &[u8], stderr: &[u8]) -> ProgramOutcome {
    ProgramOutcome {
        success,
        detail,
        output: decode_raw_streams(stdout, stderr),
        // The raw streams decide: only a run that emitted nothing at all
        // reported nothing.
        has_output: !stdout.is_empty() || !stderr.is_empty(),
    }
}

/// Run `program` with `args` (argv — never a shell command string) in `ws` and
/// return the raw run result. The single place that decides how a direct run is
/// bounded: the sanitized environment, [`DEFAULT_SHELL_TIMEOUT_SECS`], the
/// per-pipe output cap and the post-exit drain bound. `owner` travels to
/// [`run_command_with_timeout`] and decides the run's containment.
async fn run_program(
    ws: &Workspace,
    program: &Path,
    args: &[String],
    owner: RunOwner,
) -> ShellRunResult {
    let timeout = Duration::from_secs(DEFAULT_SHELL_TIMEOUT_SECS);
    let mut cmd = build_program_command(program, args, ws.as_path());
    run_command_with_timeout(&mut cmd, timeout, output_drain_timeout(), owner).await
}

/// Run `program` with `args` in `ws` and report the run as an outcome instead
/// of folding a non-zero exit into an error: a run that did not complete is a
/// failed outcome, never an `Err`, because the caller's rule is about what the
/// run reported rather than about this layer's error type.
///
/// The run is a service launch rather than an agent's work, so Windows gives it
/// no job ([`RunOwner::Service`] — see [`tree`]).
pub(crate) async fn run_program_outcome(
    ws: &Workspace,
    program: &Path,
    args: &[String],
) -> ProgramOutcome {
    match run_program(ws, program, args, RunOwner::Service).await {
        ShellRunResult::Completed {
            stdout,
            stderr,
            status,
            ..
        } => {
            let code = status.code();
            program_outcome(
                code == Some(0),
                match code {
                    Some(n) => format!("exit status {n}"),
                    None => "terminated by signal".to_string(),
                },
                &stdout,
                &stderr,
            )
        }
        ShellRunResult::TimedOut { stdout, stderr, .. } => program_outcome(
            false,
            format!("timed out after {DEFAULT_SHELL_TIMEOUT_SECS}s"),
            &stdout,
            &stderr,
        ),
        ShellRunResult::DrainTimedOut { stdout, stderr, .. } => program_outcome(
            false,
            "output drain overrun — a leftover process held the pipes".to_string(),
            &stdout,
            &stderr,
        ),
        ShellRunResult::SpawnFailed(e) => {
            program_outcome(false, format!("failed to start: {e}"), &[], &[])
        }
    }
}

/// Run `program` with `args` (argv — never a shell command string) in `ws`,
/// under the bounds [`run_program`] applies to a direct run, contained as an
/// agent's command (see [`RunOwner::Agent`]).
///
/// `label` names what the caller asked to run and is what the failure prose
/// talks about, so the model reads back the thing it called rather than the
/// interpreter's path. Program and argv are taken explicitly rather than
/// specialised to the script convention so that the argv contract — nothing in
/// an argument is ever parsed as shell syntax — can be exercised without the
/// managed runtime.
///
/// Returns the run's combined output, annotated the way the shell annotates a
/// failure: the `[exit status: …]` note is appended as its own paragraph for
/// anything but a clean exit. `Err` is a run that could not complete — spawn
/// failure, timeout or drain overrun — never a non-zero exit, so a script's
/// own failure stays the script's output.
pub(crate) async fn run_program_with_timeout(
    ws: &Workspace,
    program: &Path,
    args: &[String],
    label: &str,
) -> anyhow::Result<String> {
    match run_program(ws, program, args, RunOwner::Agent).await {
        ShellRunResult::Completed {
            stdout,
            stderr,
            status,
            ..
        } => {
            let code = status.code();
            let output = decode_raw_streams(&stdout, &stderr);
            if code == Some(0) {
                return Ok(output);
            }
            Ok(with_note(&output, &format_exit_status_note(code)))
        }
        // The same error class as the shell tool's timeout (a run that could
        // not complete), but a typed one-liner carrying the output tail rather
        // than the shell's structured block: a direct run has no per-call knob
        // to raise, so nothing here may advertise the `timeout_secs` escape
        // hatch, and it does not name what the kill ended the way the tool's
        // block does.
        ShellRunResult::TimedOut {
            stdout,
            stderr,
            elapsed,
            ..
        } => Err(anyhow::anyhow!(
            "timeout: {label} did not finish within {DEFAULT_SHELL_TIMEOUT_SECS}s and was killed\n{}",
            program_error_tail(elapsed, &stdout, &stderr),
        )),
        ShellRunResult::DrainTimedOut {
            stdout,
            stderr,
            elapsed,
            ..
        } => Err(anyhow::anyhow!(
            "timeout: {label} exited but a leftover process kept its output pipes open \
             past the drain limit — hint: keep any process the script launches inside its \
             own lifetime\n{}",
            program_error_tail(elapsed, &stdout, &stderr),
        )),
        ShellRunResult::SpawnFailed(e) => Err(anyhow::anyhow!(
            "io: cannot run {label}: {e} — hint: the program must exist and be executable"
        )),
    }
}

/// Append a bracketed note (`[exit status: 3]`, `[ignored arguments: x]`) under
/// `text` as its own paragraph: the text's trailing whitespace is trimmed so
/// the note is never preceded by a blank gap, and a run that produced nothing
/// is left as just the note.
pub(crate) fn with_note(text: &str, note: &str) -> String {
    let mut out = text.trim_end().to_string();
    if !out.is_empty() {
        out.push_str("\n\n");
    }
    out.push_str(note);
    out
}

/// Elapsed time plus the scrubbed tails of both streams, for a program run
/// that did not complete.
fn program_error_tail(elapsed: Duration, stdout: &[u8], stderr: &[u8]) -> String {
    let mut msg = format!("elapsed: {:.1}s", elapsed.as_secs_f64());
    append_output_tail(&mut msg, "stdout", stdout);
    append_output_tail(&mut msg, "stderr", stderr);
    msg
}

/// Decode both raw streams (lossy UTF-8 + ANSI strip) and combine stderr onto
/// its own line when non-blank (never a leading blank line when stdout is
/// empty). Never credential-scrubbed here: the `custom` tool's output is scrubbed
/// by the agent-level pass, while an alarm's trigger deliberately shows a check
/// exactly as the check produced it.
fn decode_raw_streams(stdout: &[u8], stderr: &[u8]) -> String {
    let stdout = decode_and_strip_ansi(stdout);
    let stderr = decode_and_strip_ansi(stderr);
    if stderr.trim().is_empty() {
        stdout
    } else if stdout.trim().is_empty() {
        stderr
    } else {
        format!("{stdout}\n{stderr}")
    }
}

fn tail_chars(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        return s.to_string();
    }
    s.chars().skip(char_count - max_chars).collect()
}

/// Appends the tail of an output buffer to `msg` with a label (e.g. "stdout" or "stderr").
///
/// The transformation chain: lossy UTF-8 decode → strip ANSI escapes → scrub credentials
/// → truncate to [`TIMEOUT_OUTPUT_TAIL_CHARS`] characters.
fn append_output_tail(msg: &mut String, label: &str, data: &[u8]) {
    if !data.is_empty() {
        let scrubbed = strip_and_scrub(data);
        let tail = tail_chars(&scrubbed, TIMEOUT_OUTPUT_TAIL_CHARS);
        let _ = write!(
            msg,
            "\n{label} (last {} chars): {tail}",
            tail.chars().count()
        );
    }
}

fn format_timeout_error(
    command: &str,
    elapsed: Duration,
    timeout_limit: Duration,
    pid: Option<u32>,
    stdout: &[u8],
    stderr: &[u8],
) -> String {
    let mut msg = format!(
        "Shell command timed out.\n\
         command: {command}\n\
         elapsed: {:.1}s\n\
         timeout_limit: {:.0}s",
        elapsed.as_secs_f64(),
        timeout_limit.as_secs_f64(),
    );
    if let Some(p) = pid {
        let _ = write!(msg, "\npid: {p}");
    }
    msg.push_str("\nreason: command was killed after exceeding the timeout");
    msg.push_str(
        "\nhint: for known-long commands, pass a larger per-call timeout via the `timeout_secs` tool argument (max 3600s).",
    );

    append_output_tail(&mut msg, "stdout", stdout);
    append_output_tail(&mut msg, "stderr", stderr);
    msg
}

fn format_drain_timeout_error(
    mode: ShellMode,
    command: &str,
    elapsed: Duration,
    drain_limit: Duration,
    pid: Option<u32>,
    stdout: &[u8],
    stderr: &[u8],
) -> String {
    let mut msg = format!(
        "Shell command output drain timed out.\n\
         command: {command}\n\
         elapsed: {:.1}s\n\
         drain_limit: {:.0}s\n\
         reason: the command exited but a leftover process kept the output \
         pipes open past the drain limit, so EOF never arrived",
        elapsed.as_secs_f64(),
        drain_limit.as_secs_f64(),
    );
    if let Some(p) = pid {
        // The pid is the run's containment root — what the drain timeout ended —
        // not the already-reaped command, and it is named in this platform's own
        // terms: a process group on unix, the process tree of the job on Windows.
        let scope = match SHELL_PLATFORM {
            ShellPlatform::Unix => "process group",
            ShellPlatform::Windows => "process tree",
        };
        let _ = write!(msg, "\nkilled {scope}: {p}");
    }
    msg.push_str(match mode {
        // Full mode has the mechanism this error is asking for — point at it
        // instead of telling the agent it does not exist.
        ShellMode::Full => {
            "\nhint: launch long-running processes with `background: true` and \
             stop them with `stop`, instead of letting a child outlive the command."
        }
        ShellMode::ReadOnly => {
            "\nhint: the tool does not support processes that outlive the command; \
             keep launched processes inside the command's lifetime. \
             If background execution is genuinely required, state that in your final response."
        }
    });

    append_output_tail(&mut msg, "stdout", stdout);
    append_output_tail(&mut msg, "stderr", stderr);
    msg
}

/// Shell command execution tool
pub struct ShellTool {
    /// Whether the shell runs in full or read-only mode.
    pub mode: ShellMode,
}

impl ShellTool {
    /// Create a new [`ShellTool`] with the given execution mode.
    #[must_use]
    pub const fn new(mode: ShellMode) -> Self {
        Self { mode }
    }

    /// Launch a command in the background (Full mode only): the command keeps
    /// running after this tool call returns, its raw output is written to a
    /// file in the temp area's `.agent` directory, and the returned message
    /// carries that file's path. The agent reads progress with the read tool
    /// and stops the session via [`Self::stop_background`].
    async fn launch_background(
        &self,
        ws: &Workspace,
        command: &str,
    ) -> anyhow::Result<(String, Option<i32>)> {
        let sessions = Self::background_sessions_handle()?;
        let path = sessions
            .launch(command, ws.as_path())
            .await
            .map_err(anyhow::Error::msg)?;
        Ok((
            format!(
                "Background session started.\n\
                 output file: {}\n\
                 command: {command}\n\
                 The command is running detached from this tool call — its raw output \
                 is written to the output file. Read the file with the read tool to follow \
                 progress. When the command exits, the line `[exit status: N]` is appended \
                 to the end of the file (including for exit 0) — its presence means the \
                 command finished. Stop the session with the shell tool's `stop` argument \
                 set to this output-file path.",
                path.display()
            ),
            Some(0),
        ))
    }

    /// Stop a background session by its output-file path (Full mode only):
    /// two-stage SIGTERM → grace → SIGKILL on unix, the session's whole process
    /// tree ended at once on Windows (see [`self::bg`] for the mechanism);
    /// stopping an already-finished session is a no-op.
    async fn stop_background(&self, stop_path: &str) -> anyhow::Result<(String, Option<i32>)> {
        let sessions = Self::background_sessions_handle()?;
        let path = PathBuf::from(stop_path);
        match sessions.stop(&path).await {
            Ok(self::bg::StopResult::Stopped) => Ok((
                format!(
                    "Background session stopped.\noutput file: {}",
                    path.display()
                ),
                Some(0),
            )),
            Ok(self::bg::StopResult::AlreadyFinished) => Ok((
                format!(
                    "Background session already finished — no action taken.\noutput file: {}",
                    path.display()
                ),
                Some(0),
            )),
            Err(e) => anyhow::bail!("{e}"),
        }
    }

    /// The agent-scoped background-session registry, read from the tool-batch
    /// context (set once per tool batch by the agent's `execute_tool_group`).
    /// `None` outside an agent run (management diagnostics, tests) —
    /// background mode is unavailable there.
    fn background_sessions_handle()
    -> anyhow::Result<std::sync::Arc<crate::tools::shell::BackgroundSessions>> {
        crate::agent::CURRENT_TOOL_BACKGROUND_SESSIONS
            .try_with(std::clone::Clone::clone)
            .unwrap_or(None)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Background shell mode is not available in this context \
                     (no agent session registry)."
                )
            })
    }

    /// Execute a command and return `(formatted_output, exit_code)`.
    ///
    /// `exit_code` is `Some(0)` for success, `Some(n)` for non-zero exit, or
    /// `None` when the process was terminated by a signal.
    ///
    /// The `formatted_output` includes the `[exit status: N]` annotation for
    /// non-zero exits and signal termination (same format as [`execute()`]).
    /// Credentials are already scrubbed from the returned output.
    #[expect(clippy::too_many_lines)] // orchestration: validation, engine hook, containment
    pub(crate) async fn execute_with_status(
        &self,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<(String, Option<i32>)> {
        // ── Background mode (Full shell roles only) ──
        // `stop`/`background` are checked before `command` is required, so a
        // stop-only invocation (no command) works. ReadOnly mode never reads
        // these arguments — its behavior is byte-identical to before.
        if self.mode == ShellMode::Full {
            let stop_path = super::get_opt_str(&args, "stop").filter(|s| !s.is_empty());
            let background = super::get_opt_bool(&args, "background")?.unwrap_or(false);
            if let Some(stop_path) = stop_path {
                if background {
                    anyhow::bail!(
                        "The `stop` and `background` arguments cannot be combined — \
                         pass only `stop` with the output-file path of a background session."
                    );
                }
                return self.stop_background(stop_path).await;
            }
            if background {
                let command_str = super::get_str(&args, "command")?;
                return self.launch_background(ws, command_str).await;
            }
        }

        let command_str = super::get_str(&args, "command")?;

        // Read-only mode: validate command before execution.
        // The grep engine interception runs in BOTH modes (read-only for the
        // validation path, full for the inherent read-only engine) — see below.
        let mut exec_str = command_str.to_string();

        // Capture the grep-engine serve decision once, before the mode branch;
        // reused by both branches for the rewrite and by the telemetry write
        // after execution.
        let mut grep_serve = grep_engine::try_serve_command(command_str, ws.as_path());
        // The rewrite may name scratch spec files (the Windows hand-off); the
        // guard removes them on every exit path below, `bail!`s included.
        let _spec_files = SpecFiles(std::mem::take(&mut grep_serve.spec_files));

        if self.mode == ShellMode::ReadOnly {
            let ctx = self::readonly::CheckContext::for_workspace(ws.as_path());
            if let Err(rejection) = check_command(command_str, &ctx) {
                anyhow::bail!("{rejection}");
            }
            if let Some(rewritten) = grep_serve.rewritten.as_deref() {
                // The engine verb is an unlisted literal and passes validation;
                // on the off chance it does not, keep the original command (on
                // Windows the refusal below reports it instead).
                if check_command(rewritten, &ctx).is_ok() {
                    exec_str = rewritten.to_string();
                }
            }
        } else {
            // Full mode: no read-only validation; the engine is inherently
            // read-only and preserves non-grep (incl. mutating) segments
            // verbatim. Background-mode Full greps are deliberately NOT served
            // (the early return above keeps them on the original command).
            if let Some(rewritten) = grep_serve.rewritten.as_deref() {
                exec_str = rewritten.to_string();
            }
        }

        // ── Windows: an unserved search is a tool error ──
        // The deliberate, documented exception to "a completed command is never
        // a tool error" (the engine module header states why): an unserved grep
        // member, or a produced rewrite the read-only guard rejected, would
        // otherwise reach the agent as interpreter noise under a status it
        // cannot tell from "no match". On unix `unserved_refusal` is `None` and
        // the original command runs.
        if let Some(cause) = unserved_refusal(SHELL_PLATFORM, &grep_serve, exec_str != command_str)
        {
            self.write_grep_telemetry(ws, command_str, &grep_serve, false, &cause, None)
                .await;
            anyhow::bail!("{}", grep_engine::unserved_failure(&cause));
        }

        // Execute with timeout to prevent hanging commands. `exec_str` may be
        // the grep-engine rewrite; the ORIGINAL `command_str` is what
        // `process_shell_output` sees below, so the grep output profile keeps
        // matching (and `cd … &&` chains still navigate the shell).
        let mut cmd = build_shell_command(&exec_str, ws.as_path());

        // Allow agent to override the default timeout via `timeout_secs`.
        // Capped at MAX_SHELL_TIMEOUT_SECS to prevent absurdly long runs.
        let timeout_secs = super::get_opt_u64(&args, "timeout_secs")?
            .map_or(DEFAULT_SHELL_TIMEOUT_SECS, |s| {
                s.min(MAX_SHELL_TIMEOUT_SECS)
            });
        let timeout = Duration::from_secs(timeout_secs);
        let drain_limit = output_drain_timeout();

        let mut result =
            run_command_with_timeout(&mut cmd, timeout, drain_limit, RunOwner::Agent).await;

        // Stream-size marker: the engine reports stdin-fed stream bytes
        // consumed via a stderr marker; strip it from the agent-visible stderr.
        // The strip removes any line containing the marker token. The pre-flight
        // `exec` check misses env-prefixed (`FOO=1 exec 2>&1`) and escaped-verb
        // (`\exec 2>&1`) redirects, so the marker can leak into the agent-visible
        // stdout or a file — beyond this strip's reach. Runs on timeout output
        // too — the marker is flushed before the engine exits, so a later-member
        // hang would otherwise surface it.
        match &mut result {
            ShellRunResult::Completed { stderr, .. }
            | ShellRunResult::TimedOut { stderr, .. }
            | ShellRunResult::DrainTimedOut { stderr, .. } => {
                if exec_str != command_str {
                    grep_engine::strip_stream_size_marker(stderr);
                }
            }
            ShellRunResult::SpawnFailed(_) => {}
        }

        // Grep-engine failure containment: the sentinel exit code means the
        // engine could neither serve nor exec the real grep, and a stale
        // self-update binary (one lacking the hidden subcommand) runs full
        // main() and dies at instance-lock with the lock message — its exit 1
        // is a legitimate grep no-match code, so the message is treated the
        // same. On unix the original command is re-run so the agent sees the
        // authentic result: in Full mode that re-executes the whole original
        // command, including any preserved mutation segments (mv/cp/mkdir/…)
        // kept verbatim in the rewrite — documented behavior: a mid-command
        // engine failure may repeat them. Residual: a mid-search panic after
        // output was streamed exits sentinel-3, but a pipe/chain member's exit
        // status masks it — aggregation tails (`grep | wc -l`, `grep | sort`)
        // are the worst case, turning the partial stream into authoritative-
        // looking wrong answers — so the agent sees the partial output.
        //
        // On Windows the same signatures are the engine failing to serve a
        // search the platform cannot run itself: telemetry is recorded and the
        // call is refused (the Windows-only exception above), with the engine's
        // own stderr line — where it wrote why it could not serve — as the
        // named cause. There the refusal marker covers the pipe/chain case the
        // exit status would otherwise mask. Timed-out/drain-timeout/spawn-failed
        // results are not sentinel cases and keep their own handling below.
        //
        // Accepted limit of reading exit 3 as the engine's on both platforms: on
        // Windows that refuses the call, so a SERVED multi-member command whose
        // last verbatim member legitimately exits 3 (`grep -rn x . &&
        // something-that-exits-3`, or `grep … | tail -1` with a tail that exits
        // 3) is refused even though the search ran — with the run's stderr read
        // as the cause, the generic line when it holds none. On unix the same
        // false trigger only re-runs the original command. The parent reads the
        // run's one composed status, never a per-member one, so a member-3 is
        // not told apart from an engine-3 here.
        let engine_failure = match &result {
            ShellRunResult::Completed { status, stderr, .. } if exec_str != command_str => {
                engine_failure(status.code(), stderr, SHELL_PLATFORM)
            }
            _ => None,
        };
        let mut sentinel_rerun = false;
        let result = match engine_failure {
            Some(EngineFailure::Refused(cause)) => {
                self.write_grep_telemetry(
                    ws,
                    command_str,
                    &grep_serve,
                    false,
                    &cause,
                    Some(&result),
                )
                .await;
                anyhow::bail!("{}", grep_engine::unserved_failure(&cause));
            }
            Some(EngineFailure::ReRun) => {
                sentinel_rerun = true;
                let mut original = build_shell_command(command_str, ws.as_path());
                run_command_with_timeout(&mut original, timeout, drain_limit, RunOwner::Agent).await
            }
            None => result,
        };

        // Record the served invocation's exit code at DEBUG (filtered from the
        // general log stream). The dedicated `grep_telemetry` table is the
        // source of truth for grep decisions; this line is observability only.
        if exec_str != command_str {
            let exit_code = match &result {
                ShellRunResult::Completed { status, .. } => status.code(),
                _ => None,
            };
            tracing::debug!(
                command = command_str,
                ?exit_code,
                "grep engine: served exit"
            );
        }

        // Grep-engine telemetry: one row per greppable call lands in the
        // dedicated `grep_telemetry` table, keeping the general `logs` stream
        // clean of grep detail. Best-effort/fail-open — a telemetry failure
        // never affects the shell result. `applied` is the ground truth of
        // whether the shell was rewritten to the engine: the analysis outcomes
        // can report served even when no engine ran (engine unavailable, spec
        // too large, ReadOnly rejection) — and a sentinel re-run replaced the
        // engine with a real-grep run — so the row's served flag must reflect
        // the ACTUAL execution. The engine-internal `exec_grep` path (cwd
        // mismatch, matcher-build failure, version mismatch) re-execs real grep
        // in place, which is NOT observable from the parent; that residual is
        // documented rather than fixed.
        let applied = exec_str != command_str;
        let served = applied && !sentinel_rerun;
        let reason = if sentinel_rerun {
            "engine sentinel re-run (real grep)"
        } else if !applied && grep_serve.rewritten.is_some() {
            // The rewrite was produced but ReadOnly validation rejected it —
            // the whole command ran real grep. Per-member skip reasons
            // describe the analysis, not why real grep ran. (On Windows this
            // case never reaches the row: it is refused above, under the same
            // cause.)
            GUARD_REJECTED_REASON
        } else if applied {
            grep_serve
                .outcomes
                .iter()
                .find(|o| !o.served)
                .map(|o| o.reason.as_str())
                .unwrap_or_default()
        } else {
            // No rewrite at all (engine unavailable / spec too large / only
            // skipped members): the outcomes carry the concrete reason.
            grep_serve
                .outcomes
                .iter()
                .find(|o| !o.reason.is_empty())
                .map_or("no rewrite produced", |o| o.reason.as_str())
        };
        self.write_grep_telemetry(ws, command_str, &grep_serve, served, reason, Some(&result))
            .await;

        match result {
            ShellRunResult::Completed {
                stdout,
                stderr,
                status,
                elapsed,
            } => {
                let stdout = decode_and_strip_ansi(&stdout);
                let stderr = decode_and_strip_ansi(&stderr);

                let exit_code = status.code(); // Option<i32> — None means signal
                let exit_note = format_exit_status_note(exit_code);

                // All completed commands return output with exit info,
                // regardless of exit code. Only actual execution failures
                // (timeout, process launch failure) are tool errors — plus the
                // Windows unserved search refused above.
                let processed = process_shell_output(
                    command_str,
                    &stdout,
                    &stderr,
                    exit_code.unwrap_or(-1),
                    elapsed,
                );
                let combined = if exit_code == Some(0) {
                    processed
                } else {
                    with_note(&processed, &exit_note)
                };
                Ok((combined, exit_code))
            }
            ShellRunResult::TimedOut {
                stdout,
                stderr,
                pid,
                elapsed,
            } => {
                tracing::info!(
                    command = command_str,
                    elapsed_secs = elapsed.as_secs_f64(),
                    ?pid,
                    stdout_bytes = stdout.len(),
                    stderr_bytes = stderr.len(),
                    "Shell command timed out"
                );
                let msg =
                    format_timeout_error(command_str, elapsed, timeout, pid, &stdout, &stderr);
                anyhow::bail!("{msg}");
            }
            ShellRunResult::DrainTimedOut {
                stdout,
                stderr,
                pid,
                elapsed,
            } => {
                tracing::info!(
                    command = command_str,
                    elapsed_secs = elapsed.as_secs_f64(),
                    drain_limit_secs = drain_limit.as_secs_f64(),
                    ?pid,
                    stdout_bytes = stdout.len(),
                    stderr_bytes = stderr.len(),
                    "Shell command output drain timed out — leftover process held the pipes"
                );
                let msg = format_drain_timeout_error(
                    self.mode,
                    command_str,
                    elapsed,
                    drain_limit,
                    pid,
                    &stdout,
                    &stderr,
                );
                anyhow::bail!("{msg}");
            }
            ShellRunResult::SpawnFailed(e) => anyhow::bail!(
                "Failed to start shell command.\n\
                 command: {command_str}\n\
                 reason: {e}"
            ),
        }
    }

    /// Persist one grep-engine telemetry row, best-effort (fail-open: a
    /// telemetry failure never affects the shell result). One helper because
    /// the record is required on every platform and on every outcome — the
    /// post-execution decision as well as the two Windows failure paths, which
    /// bail with the row's `reason` before ever running a command.
    ///
    /// `served` is the ground truth of the actual engine execution and `reason`
    /// the decision's cause; `result` is the run the row describes (`None` for
    /// a call that bailed before executing, so its elapsed time and exit code
    /// are empty). A command the analysis never reached a grep member in writes
    /// no row — except a refused Windows search, which is recorded precisely
    /// because it must not be invisible (the shape fields stay empty).
    async fn write_grep_telemetry(
        &self,
        ws: &Workspace,
        command: &str,
        grep_serve: &grep_engine::GrepServe,
        served: bool,
        reason: &str,
        result: Option<&ShellRunResult>,
    ) {
        if grep_serve.outcomes.is_empty() && grep_serve.refusal.is_none() {
            return;
        }
        let shape = grep_serve.telemetry_shape(served);
        let mode = match self.mode {
            ShellMode::ReadOnly => "ReadOnly",
            ShellMode::Full => "Full",
        };
        let workspace = ws.as_path().to_string_lossy().into_owned();
        let (duration_ms, exit_code) = match result {
            Some(ShellRunResult::Completed {
                elapsed, status, ..
            }) => (
                Some(i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)),
                status.code(),
            ),
            Some(
                ShellRunResult::TimedOut { elapsed, .. }
                | ShellRunResult::DrainTimedOut { elapsed, .. },
            ) => (
                Some(i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)),
                None,
            ),
            // A spawn failure has an attempt behind it but no clock to read: the
            // duration is the placeholder zero, and the absent exit status is
            // what marks the row.
            Some(ShellRunResult::SpawnFailed(_)) => (Some(0), None),
            // A call that bailed before running has nothing to measure: the row
            // carries no timing at all, rather than a zero that would read as a
            // genuine zero-duration run.
            None => (None, None),
        };
        if let Some(store) = crate::logs::LOG_STORE.get() {
            let row = crate::logs::GrepTelemetryRow {
                command,
                served,
                reason,
                recursive: shape.recursive,
                piped: shape.piped,
                operand_count: shape.operand_count,
                flags: shape.flags.as_str(),
                mode,
                workspace: workspace.as_str(),
                grep_count: shape.grep_count,
                served_count: shape.served_count,
                skipped_count: shape.skipped_count,
                duration_ms,
                exit_code,
            };
            let _ = store.record_grep_telemetry(row).await;
        }
    }
}

/// Extra `PATH` entries prepended for shell subprocesses so developer tools
/// (`cargo`, Homebrew, npm global bins, the managed bun runtime, etc.) resolve
/// without reading the parent process `PATH`.
///
/// Always includes the cargo bin directory (via `$CARGO_HOME/bin` if set,
/// else `~/.cargo/bin`) plus commonly expected system tool directories.
///
/// # `$CARGO_HOME` belt-and-suspenders
///
/// When `$CARGO_HOME` is explicitly set, both `$CARGO_HOME/bin` (from
/// [`crate::util::cargo_bin_dir`]) AND `~/.cargo/bin` are added, so users
/// with a non-default `CARGO_HOME` still have their cargo-installed tools
/// found. Deduplication in [`prepend_path_entries`] handles the case when
/// both point to the same directory.
fn extra_shell_path_prefixes() -> Vec<PathBuf> {
    let mut v = Vec::new();

    // cargo_bin_dir() returns $CARGO_HOME/bin if CARGO_HOME is set,
    // else ~/.cargo/bin.
    if let Some(dir) = crate::util::cargo_bin_dir() {
        v.push(dir);
    }

    // Belt-and-suspenders: when CARGO_HOME is explicitly set, also add
    // ~/.cargo/bin so both paths are covered. Dedup by prepend_path_entries.
    if let Ok(cargo_home) = std::env::var("CARGO_HOME")
        && !cargo_home.is_empty()
        && let Some(dirs) = UserDirs::new()
    {
        v.push(dirs.home_dir().join(".cargo").join("bin"));
    }

    #[cfg(unix)]
    if let Some(dirs) = UserDirs::new() {
        v.push(dirs.home_dir().join(".npm-global").join("bin"));
    }
    #[cfg(target_os = "macos")]
    {
        v.push(PathBuf::from("/opt/homebrew/bin"));
        v.push(PathBuf::from("/usr/local/bin"));
    }

    // Managed bun runtime (~/.bun/bin) — appended LAST so a user-installed bun
    // (Homebrew, npm, official installer) keeps precedence over ours.
    if let Some(dir) = crate::util::managed_bin::bun_bin_dir() {
        v.push(dir);
    }
    v
}

#[cfg(unix)]
const fn default_search_path_without_parent_env() -> &'static str {
    "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
}

/// The system root (normally `C:\Windows`) from the system's own values — the
/// drive is never assumed, since Windows may be installed on another one.
///
/// Resolution order: [`GetWindowsDirectoryW`] (authoritative — it is what the
/// OS itself uses), then `%SystemRoot%`, then `%WINDIR%`. Cached because it is
/// read several times per spawned command. The empty string when all three
/// fail, in which case callers refuse to hand out a fabricated root.
///
/// [`GetWindowsDirectoryW`]: windows_sys::Win32::System::SystemInformation::GetWindowsDirectoryW
#[cfg(windows)]
#[must_use]
fn windows_system_root() -> &'static str {
    static ROOT: OnceLock<String> = OnceLock::new();
    ROOT.get_or_init(|| {
        system_root_from_api()
            .or_else(|| env_system_root("SystemRoot"))
            .or_else(|| env_system_root("WINDIR"))
            .unwrap_or_else(|| {
                tracing::error!(
                    "Windows system root unavailable: GetWindowsDirectoryW failed and \
                     %SystemRoot%/%WINDIR% are unset"
                );
                String::new()
            })
    })
}

/// The system root as the OS reports it, through `GetWindowsDirectoryW`'s
/// two-pass protocol: probe the required length with a null/short buffer, then
/// read into a buffer of exactly that length. Trailing separators are trimmed
/// so callers can append `\System32` to the result.
#[cfg(windows)]
#[must_use]
fn system_root_from_api() -> Option<String> {
    use windows_sys::Win32::System::SystemInformation::GetWindowsDirectoryW;

    // SAFETY: a null buffer with a zero size is the documented length probe —
    // with no room to write, the API only reports the required size.
    let needed = unsafe { GetWindowsDirectoryW(std::ptr::null_mut(), 0) };
    if needed == 0 {
        return None;
    }
    let mut buf = vec![0u16; needed as usize];
    // SAFETY: `buf` has room for exactly the `needed` UTF-16 units the probe
    // asked for; the API writes at most that many and reports how many.
    let written = unsafe { GetWindowsDirectoryW(buf.as_mut_ptr(), needed) };
    if written == 0 {
        return None;
    }
    // `written` excludes the terminator, but a root that grew between the probe
    // and the read returns the (larger) required size instead — clamp it.
    let root = String::from_utf16_lossy(&buf[..(written as usize).min(buf.len())]);
    let root = root.trim_end_matches(['\\', '/', '\0']);
    if root.is_empty() {
        return None;
    }
    Some(root.to_string())
}

/// A root taken from the named environment variable: unset, empty or
/// separator-only values count as absent, and a trailing separator is trimmed
/// so the value joins natively.
#[cfg(windows)]
#[must_use]
fn env_system_root(name: &str) -> Option<String> {
    let value = std::env::var(name).ok()?;
    let trimmed = value.trim_end_matches(['\\', '/']);
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(windows)]
fn default_search_path_without_parent_env() -> String {
    let root = windows_system_root();
    if root.is_empty() {
        // An empty baseline beats one fabricated around a drive we can't
        // confirm; the extra prefixes are still prepended onto it.
        return String::new();
    }
    format!(r"{root}\System32;{root};{root}\System32\Wbem;{root}\System32\WindowsPowerShell\v1.0")
}

fn prepend_path_entries(base: impl AsRef<str>, extras: &[PathBuf]) -> String {
    let base = base.as_ref();
    let sep = if cfg!(windows) { ";" } else { ":" };

    let mut seen = HashSet::<String>::new();
    let mut parts = Vec::new();

    let normalize = |s: &str| -> String {
        if cfg!(windows) {
            s.to_lowercase()
        } else {
            s.to_string()
        }
    };

    for p in extras {
        let s = p.to_string_lossy().to_string();
        if s.is_empty() {
            continue;
        }
        if seen.insert(normalize(&s)) {
            parts.push(s);
        }
    }

    for part in base.split(sep) {
        if part.is_empty() {
            continue;
        }
        if seen.insert(normalize(part)) {
            parts.push(part.to_string());
        }
    }

    parts.join(sep)
}

/// `PATH` for shell tools: built from a portable system baseline plus
/// [`extra_shell_path_prefixes`] (no parent `PATH` read).
fn resolved_shell_path() -> String {
    prepend_path_entries(
        default_search_path_without_parent_env(),
        &extra_shell_path_prefixes(),
    )
}

/// Baseline value of a sanitized session-environment variable. The temp
/// variables are not handled here: they come from
/// [`crate::temp::shell_temp_vars`] (see [`apply_safe_env`]).
fn baseline_env_value(name: &str) -> Option<String> {
    match name {
        "PATH" => Some(resolved_shell_path()),
        "HOME" | "USERPROFILE" => {
            UserDirs::new().map(|d| d.home_dir().to_string_lossy().into_owned())
        }
        // $USER is an explicit exception to the no-parent-process-env-reads
        // constraint — usernames are not secrets and this avoids a full crate dependency
        "USER" | "USERNAME" => std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .ok()
            .or_else(|| Some("user".into())),
        "TERM" => Some("dumb".into()),
        "LANG" | "LC_ALL" | "LC_CTYPE" => Some("C.UTF-8".into()),
        "SHELL" => Some("/bin/sh".into()),
        _ => {
            #[cfg(windows)]
            if let Some(val) = windows_baseline_env_value(name) {
                return Some(val);
            }
            None
        }
    }
}

/// Returns baseline values for Windows-specific environment variables.
///
/// These variables are only meaningful on Windows. Where the parent process
/// knows the value (`%SystemDrive%`, `%ComSpec%`), its own is preferred; the
/// rest are derived from the OS-reported system root rather than from a
/// hard-coded `C:`, so an install on another drive still gets a usable
/// environment.
#[cfg(windows)]
fn windows_baseline_env_value(name: &str) -> Option<String> {
    match name {
        "PATHEXT" => Some(".COM;.EXE;.BAT;.CMD;.VBS;.JS".into()),
        "HOMEDRIVE" | "HOMEPATH" => UserDirs::new().and_then(|d| {
            let s = d.home_dir().to_string_lossy().into_owned();
            // Note: Windows home paths always start with a drive letter
            // (e.g., "C:\Users\..."), so byte-index slicing at positions 0..2
            // is safe. We validate the drive-letter pattern by checking that
            // the second byte is b':' (colon). This relies on the ASCII
            // representation of drive letters (A-Z).
            if s.len() >= 2 && s.as_bytes().get(1) == Some(&b':') {
                match name {
                    "HOMEDRIVE" => Some(s[..2].to_string()),
                    // HOMEPATH: slice from byte 2 onward to skip "C:"
                    _ => Some(s[2..].to_string()),
                }
            } else {
                None
            }
        }),
        // None of these is ever fabricated around a `C:` assumption: an absent
        // system root leaves the variable unset rather than pointing a child at
        // a plausible-but-wrong drive.
        "SYSTEMROOT" | "WINDIR" => {
            let root = windows_system_root();
            (!root.is_empty()).then(|| root.to_string())
        }
        "SYSTEMDRIVE" => inherited_env_value("SystemDrive").or_else(|| {
            // Drive prefix of the system root — the drive letter plus its colon
            // (`C:`), derived rather than assumed: the second byte being `:`
            // proves the `X:` shape, and both bytes are ASCII, so slicing is
            // safe.
            let root = windows_system_root();
            (root.as_bytes().get(1) == Some(&b':')).then(|| root[..2].to_string())
        }),
        "COMSPEC" => inherited_env_value("ComSpec").or_else(|| {
            let root = windows_system_root();
            (!root.is_empty()).then(|| format!(r"{root}\System32\cmd.exe"))
        }),
        _ => None,
    }
}

/// The parent process's value for a Windows variable that is not a secret, when
/// set to something non-empty — the authoritative source when it exists.
#[cfg(windows)]
#[must_use]
fn inherited_env_value(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// The read-only banner: the shared skeleton with the daemon's temp root and the
/// session platform's check bullets substituted in. A literal path is the one
/// spelling every shell resolves the same way, so the skeleton carries it; the
/// platform's own tools and temp variables are named only in the selected
/// fragment, since one shared file would push one platform's spellings at the
/// other's sessions (the same reason `crate::temp`'s temp-cleanup renderer
/// selects its tool block).
fn render_readonly_banner() -> String {
    let platform_checks = crate::prompt::load_prompt(match SHELL_PLATFORM {
        ShellPlatform::Windows => "tool/shell_readonly_banner_windows.md",
        ShellPlatform::Unix => "tool/shell_readonly_banner_unix.md",
    })
    .trim()
    .to_owned();
    crate::prompt::substitute(
        &crate::prompt::load_prompt("tool/shell_readonly_banner.md"),
        &[
            ("{{temp_root}}", &crate::temp::shell_tmpdir()),
            ("{{platform_checks}}", &platform_checks),
        ],
    )
}

/// The grep-engine disclosure: the shared skeleton with the session platform's
/// fragment substituted in — the engine serves the same searches here as
/// everywhere, but what a platform does with the ones it cannot serve, and what
/// it has to fall back on, differs. Both renderers read [`SHELL_PLATFORM`], the
/// same value the guard and the engine's command model read, so a Windows
/// session is never told about a system `grep` and a unix one never about its
/// absence.
fn render_grep_notes() -> String {
    let platform_notes = crate::prompt::load_prompt(match SHELL_PLATFORM {
        ShellPlatform::Windows => "tool/shell_grep_notes_windows.md",
        ShellPlatform::Unix => "tool/shell_grep_notes_unix.md",
    })
    .trim()
    .to_owned();
    crate::prompt::substitute(
        &crate::prompt::load_prompt("tool/shell_grep_notes.md"),
        &[("{{platform_notes}}", &platform_notes)],
    )
}

/// The full-mode notes: the shared skeleton with this platform's stop semantics
/// substituted in. What stopping a session does is the one thing the two
/// platforms do differently, and a session must never be promised a mechanism
/// its platform does not have — [`tree`] is where the mechanisms themselves are.
fn render_full_mode_notes() -> String {
    let stop = stop_semantics();
    crate::prompt::substitute(
        &crate::prompt::load_prompt("tool/shell_full.md"),
        &[("{{stop_semantics}}", &stop)],
    )
}

/// What stopping a run does on this platform — one text for both the tool
/// description's stop bullet and the `stop` argument's schema entry, so the two
/// cannot promise different mechanisms.
fn stop_semantics() -> String {
    crate::prompt::load_prompt(match SHELL_PLATFORM {
        ShellPlatform::Windows => "tool/shell_full_stop_windows.md",
        ShellPlatform::Unix => "tool/shell_full_stop_unix.md",
    })
    .trim()
    .to_owned()
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &'static str {
        "shell"
    }

    fn description(&self) -> String {
        // The base description and the grep-engine disclosure are shared
        // verbatim between the modes (a single copy each, so the two
        // descriptions cannot drift); only the read-only banner, the full-mode
        // sections (stop semantics included) and the platform's grep notes are
        // mode-/platform-specific.
        let base = crate::prompt::load_prompt("tool/shell.md");
        let sections: [String; 3] = match self.mode {
            ShellMode::ReadOnly => [render_readonly_banner(), base, render_grep_notes()],
            ShellMode::Full => [base, render_full_mode_notes(), render_grep_notes()],
        };
        sections.map(|s| s.trim_end().to_owned()).join("\n\n")
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let timeout_secs = json!({
            "type": "integer",
            "description": "Optional custom timeout in seconds (default: 600, max: 3600). Use this for long-running commands that need more than the default 10-minute timeout.",
            "minimum": 1,
            "maximum": 3600
        });
        match self.mode {
            // ReadOnly: byte-identical to the pre-background literal. The
            // background capability is Full-only; the shared `timeout_secs`
            // binding below keeps the two schemas from drifting.
            ShellMode::ReadOnly => super::tool_params_schema(
                &json!({
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout_secs": timeout_secs,
                }),
                &["command"],
            ),
            // Full: adds the background/stop arguments on top of the same
            // command/timeout_secs literal. `command` is intentionally NOT in
            // `required` here: a stop-only invocation passes just `stop`, and
            // requiring `command` would push a schema-following model into
            // inventing a dummy command for stop calls (the daemon accepts
            // both shapes, but the documented stop-only shape must not be
            // rejected by provider-side validation).
            ShellMode::Full => super::tool_params_schema(
                &json!({
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute. Required for normal and background runs; not needed (and ignored) when `stop` is set."
                    },
                    "timeout_secs": timeout_secs,
                    "background": {
                        "type": "boolean",
                        "description": "When true, run the command in the background: it keeps running after this tool call returns and its raw output is written to a file in the temp area whose path is returned. Read that file with the read tool; when the command exits, the line `[exit status: N]` is appended to its end (including exit 0). `timeout_secs` is ignored in background mode. Default: false.",
                        "default": false
                    },
                    "stop": {
                        "type": "string",
                        // The platform's own stop text (`stop_semantics`) — the
                        // same text the description's stop bullet carries.
                        "description": format!(
                            "Output-file path of a background session (as returned by a \
                             background launch) to stop. {} Pass only `stop` with the exact \
                             path — a `command` is not needed and is ignored if present, and \
                             `background` must NOT be combined with `stop` (the tool rejects \
                             the combination). Stopping an already-finished session is a no-op.",
                            stop_semantics()
                        )
                    },
                }),
                &[],
            ),
        }
    }

    fn side_effects(&self) -> bool {
        // ReadOnly mode validates commands against a mutating-command blocklist
        // — best-effort guard, not a sandbox, but sufficient for grouping.
        self.mode != ShellMode::ReadOnly
    }

    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        false // shell pipeline already scrubs stdout and stderr once at pipeline entry
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        self.execute_with_status(ws, args)
            .await
            .map(|(output, _)| output)
    }
}

/// Get the shared temp directory for spill/full output logs.
///
/// NO startup purge here: crash-leftover `.agent` files are reclaimed by the
/// periodic temp cleaner (see [`crate::temp`]) or the OS temp sweep — the
/// daemon builds no startup reclamation. Within a run, owner-deletes-at-end
/// removes what the agent created.
pub(crate) fn agent_temp_dir() -> Option<std::path::PathBuf> {
    let dir = std::env::temp_dir().join(".agent");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Spill paths created during agent runs, keyed by the owning agent id: spill
/// files plus the read tool's per-call document-conversion directories.
/// Owner-deletes-at-end: [`cleanup_agent_spills`] removes them when the agent
/// run ends. Entries for a dead agent id are removed on cleanup; a daemon
/// crash leaves the files behind for the OS temp sweep and the periodic
/// temp cleaner (the daemon performs no startup purge).
static SPILL_OWNERS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, Vec<std::path::PathBuf>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Record a spill path (file or the read tool's conversion directory) under the
/// current tool's owning agent id (set during agent tool execution). Outside an
/// agent run (diagnostics runner, tests) it is recorded under
/// [`crate::agent::role::DIAGNOSTICS_ROLE`] (`"diagnostics"`) so the diagnostics
/// runner can clean up what it created. Agent ids are always prefixed
/// (`ticket_*`, `manager_*`, etc.) and never equal bare `"diagnostics"` so no
/// collision. Tests outside an agent also bucket there (acceptable).
pub(crate) fn record_spill_owner(path: std::path::PathBuf) {
    let agent = crate::agent::CURRENT_TOOL_AGENT_ID
        .try_with(Clone::clone)
        .unwrap_or(None);
    if let Some(ref agent_id) = agent {
        debug_assert_ne!(
            agent_id,
            crate::agent::role::DIAGNOSTICS_ROLE,
            "agent id must not collide with diagnostics spill owner"
        );
    }
    let key = agent.unwrap_or_else(|| crate::agent::role::DIAGNOSTICS_ROLE.to_string());
    let mut map = SPILL_OWNERS.lock().unwrap_poison();
    map.entry(key).or_default().push(path);
}

/// Delete the spill paths recorded for `agent_id` (owner-deletes-at-end): a
/// path that is a directory (the read tool's per-call document-conversion
/// artifacts) is removed whole with [`std::fs::remove_dir_all`], a spill file
/// with [`std::fs::remove_file`]. Also clears the registry entry so a later run
/// of the same agent id starts fresh. Callers: the agent run-end cleanup guard
/// (`RunEndCleanup`, which fires on every unwinding exit path of a run — see its
/// doc for the ones it cannot cover) and the diagnostics runner (which passes
/// [`crate::agent::role::DIAGNOSTICS_ROLE`]). The diagnostics spill owner is
/// [`crate::agent::role::DIAGNOSTICS_ROLE`] (`"diagnostics"`).
pub(crate) fn cleanup_agent_spills(agent_id: &str) {
    let mut map = SPILL_OWNERS.lock().unwrap_poison();
    let Some(paths) = map.remove(agent_id) else {
        return;
    };
    for p in paths {
        let _ = if p.is_dir() {
            std::fs::remove_dir_all(&p)
        } else {
            std::fs::remove_file(&p)
        };
    }
}

// ── Pipeline functions ────────────────────────────────────────────────

/// Quote-tracking state machine. Returns `true` when `c` is outside quotes
/// and should be examined for shell operators or redirect patterns.
///
/// This function tracks ONLY quote state — escape handling is the caller's
/// responsibility. Most callers should use [`track_char_context`] instead,
/// which combines both escape and quote tracking.
const fn check_outside_quotes(c: char, in_single: &mut bool, in_double: &mut bool) -> bool {
    match c {
        '\'' if !*in_double => {
            *in_single = !*in_single;
            false
        }
        '"' if !*in_single => {
            *in_double = !*in_double;
            false
        }
        _ => !*in_single && !*in_double,
    }
}

/// Combined escape and quote tracking for shell command scanning.
///
/// Handles backslash escaping (with the `escaped` flag) and quote state
/// transitions (via [`check_outside_quotes`]). Returns `true` when `c` is
/// a normal unescaped character outside quotes that the caller should
/// examine for shell operators or redirect patterns. Returns `false` when:
///
/// * `c` was preceded by an escape backslash (the `escaped` flag was set)
/// * `c` is itself a backslash starting an escape
/// * `c` is a quote character or inside quotes
///
/// After a `false` return, the caller may still need to push the character
/// to an output buffer (e.g., [`super::scan::strip_heredoc_bodies`]
/// preserves the command string for redirect scanning, while the token
/// classifier simply continues without pushing).
///
/// # Known limitation
///
/// Inside double quotes, `\` should only escape `\`, `$`, `` ` ``, `"`, and
/// newline in a real shell. This function treats any backslash inside double
/// quotes as an escape, which is acceptable for redirect detection: a quoted
/// redirect operator is harmless, and an escaped actual redirect would be a
/// false negative (allow), also harmless.
const fn track_char_context(
    c: char,
    in_single: &mut bool,
    in_double: &mut bool,
    escaped: &mut bool,
) -> bool {
    if *escaped {
        *escaped = false;
        return false;
    }
    if c == '\\' && !*in_single {
        *escaped = true;
        return false;
    }
    check_outside_quotes(c, in_single, in_double)
}

/// Consume a `$(...)` or backtick command substitution (delimiters included)
/// into `current` when `c` starts one; returns whether one was consumed.
/// Escape- and quote-aware via [`track_char_context`]: separators inside the
/// body are substitution content, not command separators — mis-splitting
/// flips `is_chained` in [`process_shell_output`] and disables
/// standalone-only output transforms. Paren depth mirrors
/// scan::find_paren_close.
fn consume_substitution(
    c: char,
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    current: &mut String,
) -> bool {
    if c == '$' && chars.peek() == Some(&'(') {
        current.push(c);
        current.push(chars.next().expect("peeked '('"));
        let mut depth = 1usize;
        let mut sub_single = false;
        let mut sub_double = false;
        let mut sub_escaped = false;
        for c2 in chars.by_ref() {
            current.push(c2);
            if !track_char_context(c2, &mut sub_single, &mut sub_double, &mut sub_escaped) {
                continue;
            }
            // Every unquoted paren nests — mirrors scan::find_paren_close.
            if c2 == '(' {
                depth += 1;
            } else if c2 == ')' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
        }
        return true;
    }
    if c == '`' {
        current.push(c);
        let mut sub_escaped = false;
        for c2 in chars.by_ref() {
            current.push(c2);
            if sub_escaped {
                sub_escaped = false;
            } else if c2 == '\\' {
                sub_escaped = true;
            } else if c2 == '`' {
                break;
            }
        }
        return true;
    }
    false
}

/// Split a shell command string into logical segments at shell operators
/// (quote- and substitution-aware, newlines separate, heredoc bodies
/// stripped first). Mis-splitting flips `is_chained` in
/// [`process_shell_output`] and disables standalone-only output transforms.
fn extract_command_segments(command: &str) -> Vec<String> {
    // Heredoc bodies are excluded from command scanning (see the read-only
    // shell guard contract): body text must never be segmented
    // as commands, and redirect operators inside bodies must not be scanned.
    let scan = scan::strip_heredoc_bodies(command);
    segment_command(&scan, SegmentMode::Profile)
        .expect("profile segmentation never errors")
        .into_iter()
        .map(|(seg, _)| seg)
        .collect()
}

/// Splitting policy of the shared segmenter core ([`segment_command`]): the
/// profile-selection and grep-interception splitters had already drifted
/// apart silently (backslash handling); both run this core, and these two
/// policies are the entire divergence. Unifying them is a separate decision.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SegmentMode {
    /// Empty segments silently skipped; backslash dropped before ordinary chars.
    Profile,
    /// Empty segments before a connector are syntax errors, except blank
    /// lines and `;;` case-arm terminators; backslash always preserved.
    Grep,
}

/// Empty-segment handling at a flush: skip (profile) or hard error (grep).
/// Grep's blank-line and case-`;;` allowances are structural, not policy
/// variants: the `\n` flush skips; the `;` arm consumes the second `;`.
#[derive(Clone, Copy)]
enum EmptySegPolicy {
    Skip,
    Error,
}

/// Escape-sensitive chars for the profile backslash policy: preserved so
/// downstream scans see escaped operators, not real ones.
const fn is_escape_sensitive(c: char) -> bool {
    matches!(
        c,
        '\\' | '\'' | '"' | '>' | '<' | '&' | '|' | ';' | '$' | '`'
    )
}

/// True when `keyword` (e.g. `case` when opening, `esac` when closing a
/// `case` statement) appears in command position: first word of the segment,
/// or second after a block keyword that introduces commands.
fn segment_command_word(segment: &str, keyword: &str) -> bool {
    let mut words = segment.split_whitespace();
    match words.next() {
        Some(w) if w == keyword => true,
        Some("do" | "then" | "else" | "elif" | "if") => words.next() == Some(keyword),
        _ => false,
    }
}

/// Shared quote/substitution-aware segmenter core: split an already-heredoc-
/// stripped command into (segment, connector) pairs. Returns `None` when a
/// connector follows an empty segment in a mode that errors on it (grep;
/// blank-line and case-arm `;;` exceptions apply).
#[expect(clippy::too_many_lines)] // quote/substitution state machine
pub(super) fn segment_command(command: &str, mode: SegmentMode) -> Option<Vec<(String, String)>> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut in_case = false;
    let mut chars = command.chars().peekable();

    let flush = |current: &mut String,
                 out: &mut Vec<(String, String)>,
                 conn: &str,
                 policy: EmptySegPolicy,
                 in_case: &mut bool|
     -> bool {
        let t = current.trim();
        let pushed = !t.is_empty();
        if pushed {
            out.push((t.to_string(), conn.to_string()));
            *in_case =
                segment_command_word(t, "case") || (*in_case && !segment_command_word(t, "esac"));
        }
        current.clear();
        pushed || matches!(policy, EmptySegPolicy::Skip)
    };

    let base = match mode {
        SegmentMode::Profile => EmptySegPolicy::Skip,
        SegmentMode::Grep => EmptySegPolicy::Error,
    };

    while let Some(c) = chars.next() {
        if c == '\\' && !in_single {
            match chars.next() {
                Some('\n') => continue,
                // Profile drops the backslash before ordinary chars; grep
                // always preserves both (dropping `\*` makes a shell glob).
                Some(next) if mode == SegmentMode::Profile && !is_escape_sensitive(next) => {
                    current.push(next);
                }
                Some(next) => {
                    current.push('\\');
                    current.push(next);
                }
                None => current.push('\\'),
            }
            continue;
        }

        if check_outside_quotes(c, &mut in_single, &mut in_double) {
            if consume_substitution(c, &mut chars, &mut current) {
                continue;
            }
            match c {
                '&' if chars.peek() == Some(&'&') => {
                    chars.next();
                    if !flush(&mut current, &mut out, "&&", base, &mut in_case) {
                        return None;
                    }
                    continue;
                }
                // `>|` is one compound redirect, not a pipe (bare `>` misread).
                '|' if current.trim_end().ends_with('>') => {
                    current.push(c);
                    continue;
                }
                '|' => {
                    // `|&` is one connector in grep mode; profile splits at `|`.
                    if mode == SegmentMode::Grep && chars.peek() == Some(&'&') {
                        chars.next();
                        if !flush(&mut current, &mut out, "|&", base, &mut in_case) {
                            return None;
                        }
                    } else if chars.peek() == Some(&'|') {
                        chars.next();
                        if !flush(&mut current, &mut out, "||", base, &mut in_case) {
                            return None;
                        }
                    } else if !flush(&mut current, &mut out, "|", base, &mut in_case) {
                        return None;
                    }
                    continue;
                }
                // Newlines separate commands; blank lines stay valid sh
                // (grep's blank-line exception: this flush always skips).
                '\n' => {
                    flush(
                        &mut current,
                        &mut out,
                        "\n",
                        EmptySegPolicy::Skip,
                        &mut in_case,
                    );
                    continue;
                }
                ';' => {
                    if !flush(&mut current, &mut out, ";", base, &mut in_case) {
                        return None;
                    }
                    // Case-arm `;;` (grep): consume the second `;` so no empty
                    // flush happens for it. Outside a case, `;;` errors in grep
                    // mode and is silently skipped in profile mode.
                    if mode == SegmentMode::Grep && in_case && chars.peek() == Some(&';') {
                        chars.next();
                    }
                    continue;
                }
                _ => {}
            }
        }
        current.push(c);
    }
    flush(
        &mut current,
        &mut out,
        "",
        EmptySegPolicy::Skip,
        &mut in_case,
    );
    // Trailing pipe/`&&`/`||` would silently drop an empty member into a
    // VALID executed pipeline — fail closed (grep policy only).
    if mode == SegmentMode::Grep
        && matches!(
            out.last().map(|(_, c)| c.as_str()),
            Some("|" | "|&" | "||" | "&&")
        )
    {
        return None;
    }
    Some(out)
}

/// Find the index of the first non-flag word in a slice, skipping:
/// - Git global flags (and their values) when `is_git` is true
/// - Cargo toolchain specifiers (`+nightly`-style) when not git
/// - Stderr-capture suffixes (`2>&1`, `1>&2`) — never a subcommand
/// - Any word starting with `-`
///
/// Shared helper used by [`canonical_command`] and `extract_git_subcommand`
/// to avoid duplicating the flag-skipping loop. The toolchain/stderr-capture
/// skipping fixes the subcommand-resolution bugs where `cargo +nightly build`
/// or `git --version 2>&1` parsed the toolchain/suffix as the subcommand
/// (guard contract, resolved).
pub(super) fn find_first_non_flag_index(words: &[&str], is_git: bool) -> Option<usize> {
    let mut i = 0;
    while i < words.len() {
        let w = words[i];
        if is_git && GIT_GLOBAL_FLAGS.contains(&w) {
            i += 2; // skip flag and its value (safe: loop condition checks len)
            continue;
        }
        // Toolchain specifiers (`+nightly`, `+stable`) — cargo only.
        if !is_git && w.starts_with('+') {
            i += 1;
            continue;
        }
        // Stderr-capture suffixes (`2>&1`, `1>&2`) are not subcommands.
        if w == "2>&1" || w == "1>&2" {
            i += 1;
            continue;
        }
        if w.starts_with('-') {
            i += 1;
            continue;
        }
        return Some(i);
    }
    None
}

/// Find the index of the first word that is a command (not a shell prefix,
/// flag, or environment variable assignment).
///
/// Shared helper used by [`first_command_word`] and [`canonical_command`]
/// to avoid duplicating the scanning logic. A balanced-quoted word (`"env"`,
/// `'sudo'`) names the same program — it is normalized before the prefix
/// match so `"env" "t"ouch` resolves to `touch` instead of being masked as
/// an unknown quoted `env` (fail-closed blocklist dispatch). Env assignments
/// are matched the same way (`"TMPDIR=/tmp"` is an assignment), so `export
/// "TMPDIR=/tmp"` has no command word and `"TMPDIR=/tmp" cmd` resolves to
/// `cmd`. Note this also feeds profile selection (`"env" ls` classifies as
/// the ls profile), which is directionally more correct for exotic
/// quoted-prefix spellings.
pub(super) fn find_first_command_word_index(words: &[&str]) -> Option<usize> {
    words.iter().position(|w| {
        let u = scan::strip_quoted_word(w);
        !SHELL_PREFIXES.contains(&u) && !w.starts_with('-') && !is_env_assignment(u)
    })
}

/// Extract the first command word index, basename, and the split words
/// from a shell segment.
///
/// Trims the segment, splits on whitespace keeping `$(...)`/backtick
/// substitutions whole (bash treats them as ONE word — a plain split would
/// shift the command word onto a substitution's inner word and skip the
/// mutator/git/cargo dispatch), finds the first non-prefix/non-flag/non-env
/// word index via [`find_first_command_word_index`], and extracts the
/// basename from it (see [`command_word_basename`]). Returns `None` when no
/// command word is found (e.g., only prefixes/flags/env assignments).
fn command_word_from_segment(segment: &str) -> Option<(usize, &str, Vec<&str>)> {
    let trimmed = segment.trim();
    let words = scan::split_words_keeping_substitutions(trimmed);
    let idx = find_first_command_word_index(&words)?;
    Some((idx, command_word_basename(words[idx]), words))
}

/// Basename of a command word, for the blocklist/git/profile dispatch.
///
/// Balanced quotes are not part of the name, so the split runs on the literal
/// spelling rather than the raw one — otherwise the split leaves a stray quote
/// in the basename (`'/opt/mahbot/bin/mahbot'` → `mahbot'`) and the verb
/// classifiers read the result as unprovable. An unprovable word keeps its raw
/// spelling: it is no literal name anyway, and the guard resolves a variable
/// command word by exact spelling (`"$BIN"` must stay `"$BIN"`).
pub(super) fn command_word_basename(word: &str) -> &str {
    // Unix-only, like the classifier arm this consumes (see
    // `classify_verb_word`): elsewhere the raw spelling is split, as before.
    #[cfg(unix)]
    let word = match readonly::classify_verb_word(word) {
        readonly::VerbClass::Literal(content) => content,
        readonly::VerbClass::Unprovable => word,
    };
    word.rsplit('/')
        .next()
        .expect("rsplit always yields at least one element")
}

/// Extract just the first command word (basename) from a shell segment.
///
/// Strips shell prefixes, environment variable assignments (`KEY=value`), and
/// absolute paths, but stops before any subcommand detection. This is the
/// lightweight alternative to [`canonical_command`] for callers that only need
/// the command name (e.g., the read-only guard's verb dispatch).
pub(super) fn first_command_word(segment: &str) -> &str {
    let Some((_, cmd, _)) = command_word_from_segment(segment) else {
        return "";
    };
    cmd
}

/// Extract a canonical command key from a shell segment for profile matching.
///
/// Strips shell prefixes, environment variable assignments (`KEY=value`),
/// absolute paths, and flags between the command and its subcommand. For git, also skips flag *values* for global flags like `-C` and
/// `-c`. For all other commands, only the flag token itself is skipped — flags
/// that take values (e.g., `cargo --profile release build`) will have their
/// value misidentified as the subcommand. This is a known limitation: the
/// returned key won't match any profile, falling through to generic filtering,
/// which is the same end state as the pre-fix behavior.
fn canonical_command(segment: &str) -> String {
    let Some((cmd_idx, cmd, words)) = command_word_from_segment(segment) else {
        return String::new();
    };

    let remaining = &words[cmd_idx + 1..];
    if remaining.is_empty() {
        return cmd.to_string();
    }

    // Skip flags between command and subcommand using shared helper
    // cmd is the bare basename (quotes stripped, then rsplit('/').next()),
    // so == comparison is safe.
    let is_git = cmd == "git";
    if let Some(sub_idx) = find_first_non_flag_index(remaining, is_git) {
        format!("{} {}", cmd, remaining[sub_idx])
    } else {
        cmd.to_string()
    }
}

/// Check if a word is a POSIX shell variable assignment (`VAR=value`).
///
/// The word must start with `[A-Za-z_]`, contain at least one `=`, and the
/// name portion (before `=`) must consist only of `[A-Za-z0-9_]`.
fn is_env_assignment(word: &str) -> bool {
    if let Some(eq_pos) = word.find('=')
        && eq_pos > 0
    {
        let prefix = &word[..eq_pos];
        return prefix
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && prefix
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_');
    }
    false
}

/// Select the first matching profile for a command string.
/// Returns on the first match in the first matching command segment
/// (breaks out of both segment and profile iteration early); chained
/// commands are handled by iterating over pre-parsed segments.
///
/// Profiles with `standalone_only` are skipped when `is_chained` is true,
/// so transforms that assume homogeneous output (e.g., `compact_ls`) are
/// not applied to chained commands. The command falls through to
/// `GEN_FALLBACK` with its sensible truncation defaults instead.
fn select_profile(segments: &[String], is_chained: bool) -> &'static Profile {
    for segment in segments {
        let canonical = canonical_command(segment);
        if canonical.is_empty() {
            continue;
        }
        for p in PROFILES.iter() {
            if is_chained && p.standalone_only {
                continue;
            }
            if p.match_command.is_match(&canonical) {
                return p;
            }
        }
    }
    &GEN_FALLBACK
}

/// Combine stdout with (filtered) stderr.
///
/// Returns stdout as-is when stderr is empty, regardless of exit code.
/// On success (exit 0) with `keep_stderr` patterns: only stderr lines
/// matching those patterns are appended. On success without patterns
/// (or when no stderr lines match), stderr is silently dropped.
/// On failure (non-zero exit) with `keep_stderr` patterns: the same
/// filtering is applied to avoid dumping command progress noise
/// (e.g. `Checking mahbot ...` from cargo). When no lines match,
/// the stderr section is omitted entirely. On failure without
/// patterns, all stderr is appended unconditionally.
fn combine_output(
    stdout: &str,
    stderr: &str,
    exit_code: i32,
    keep_stderr: Option<&RegexSet>,
) -> String {
    let stderr_trimmed = stderr.trim();
    if stderr_trimmed.is_empty() {
        return stdout.to_string();
    }
    // Apply keep_stderr filtering on both success and failure paths.
    let filtered = keep_stderr.and_then(|patterns| filter_keep_stderr(stderr, patterns));
    let with_stderr_section = |body: &str| {
        if stdout.is_empty() {
            format!("stderr:\n{body}")
        } else {
            format!("{stdout}\nstderr:\n{body}")
        }
    };
    match (exit_code == 0, filtered) {
        // Matching lines found — append them as a stderr section.
        (_, Some(relevant)) => with_stderr_section(&relevant),
        // Failure without keep_stderr: dump all stderr unconditionally.
        (false, None) if keep_stderr.is_none() => with_stderr_section(stderr_trimmed),
        // Success or failure with keep_stderr but no matching lines: omit stderr section.
        _ => stdout.to_string(),
    }
}

/// Filter stderr lines through `keep_stderr` patterns and join matching lines.
/// Returns `None` when no lines match.
fn filter_keep_stderr<'a>(stderr: &'a str, patterns: &RegexSet) -> Option<String> {
    let relevant: Vec<&'a str> = stderr.lines().filter(|l| patterns.is_match(l)).collect();
    if relevant.is_empty() {
        return None;
    }
    Some(relevant.join("\n"))
}

/// Shared tail: append elapsed timing (if ≥1s), then spill to file.
///
/// When `full_output_for_spill` is provided, it represents the full output
/// *before* head/tail truncation was applied. If it exceeds the spill
/// threshold, the full version is saved and a spill header is appended to
/// `combined` — this preserves complete content for agent review when
/// head/tail trimming reduces the inline view to a small snippet.
///
/// When no pre-head/tail output is available (or it's too small to spill),
/// the final `combined` output is checked against the threshold and spilled
/// if large enough, replacing the inline content with a preview.
fn finish_shell_output(
    mut combined: String,
    elapsed: Duration,
    full_output_for_spill: Option<&str>,
) -> String {
    // Both stdout and stderr are credential-scrubbed once at
    // `apply_profile_pipeline` entry, so no further scrubbing is needed
    // here.  ShellTool overrides `should_scrub_output` to return `false`,
    // disabling the agent-level scrub as redundant — the pipeline
    // guarantees scrubbing before any data reaches this function.
    if elapsed.as_secs_f64() >= 1.0 {
        let _ = write!(combined, "\n[took {:.1}s]", elapsed.as_secs_f64());
    }

    // When pre-head/tail output is available and large enough, spill the
    // fuller version so agents can `read` the complete output despite the
    // head/tail truncation reducing inline content to a snippet.
    // `full_output_for_spill` is already credential-scrubbed.
    // The `pre` value is only provided when output exceeds
    // `TOOL_OUTPUT_BUDGET_BYTES` (guaranteed by `apply_line_truncation`),
    // so no length re-check is needed here.
    if let Some(pre) = full_output_for_spill {
        debug_assert!(
            pre.len() > TOOL_OUTPUT_BUDGET_BYTES,
            "invariant: pre-truncation output ({}) must exceed threshold ({})",
            pre.len(),
            TOOL_OUTPUT_BUDGET_BYTES,
        );
        let byte_count = pre.len();
        let line_count = pre.lines().count();
        if let Some(path) = spill_output(pre) {
            let hint = format_spill_header(&path, byte_count, line_count);
            combined.push('\n');
            combined.push_str(&hint);
        }
        return combined;
    }

    // No pre-truncation spill — try spilling the final combined output
    try_spill_to_file(combined, TOOL_OUTPUT_BUDGET_BYTES)
}

/// Append `line` to `buf`, inserting a `'\n'` separator if `buf` is non-empty.
fn push_line(buf: &mut String, line: &str) {
    if !buf.is_empty() {
        buf.push('\n');
    }
    buf.push_str(line);
}

/// Collapse runs of blank lines to at most 2 consecutive.
fn collapse_blank_lines(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut blank_run = 0usize;
    for line in input.lines() {
        if line.trim().is_empty() {
            blank_run += 1;
            if blank_run > 2 {
                continue; // skip excess blank lines
            }
        } else {
            blank_run = 0;
        }
        push_line(&mut result, line);
    }
    result
}

/// State machine for processing cargo test output.
/// Drops passing test lines, captures only failure blocks + summary.
pub(super) fn filter_cargo_test_output(output: &str, exit_code: i32) -> String {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Section {
        Normal,
        InFailures,
    }

    struct CargoTestFilter {
        section: Section,
        has_failures: bool,
        has_compile_errors: bool,
        summary_lines: Vec<String>,
        output_lines: Vec<String>,
    }

    let exit_ok = exit_code == 0;

    let mut f = CargoTestFilter {
        section: Section::Normal,
        has_failures: false,
        has_compile_errors: false,
        summary_lines: Vec::new(),
        output_lines: Vec::new(),
    };

    for line in output.lines() {
        let trimmed = line.trim_start();

        // Skip compilation/download noise (Running is useful context in test output)
        if CARGO_COMPILE_PREFIXES
            .iter()
            .any(|p| *p != "Running" && trimmed.starts_with(p))
        {
            continue;
        }

        if trimmed.starts_with("test ") && trimmed.contains("... ok") {
            continue;
        }

        // Skip "running N" lines
        if trimmed.starts_with("running ") {
            continue;
        }

        // Track compile errors for fallback
        if trimmed.starts_with("error[") || trimmed.starts_with("error:") {
            f.has_compile_errors = true;
        }

        // "failures:" toggles section state
        if trimmed == "failures:" {
            f.section = Section::InFailures;
            continue;
        }

        // Capture test result summary — must remain unconditional by section.
        // The second "failures:" now stays in InFailures (no InFailureNames state),
        // so "test result:" after a failure-name list is caught here, not by the
        // InFailures block below. Adding a section guard here would break the output.
        if trimmed.starts_with("test result:") {
            f.summary_lines.push(line.to_string());
            f.section = Section::Normal;
            continue;
        }

        // In failure section — keep the block
        if f.section != Section::Normal {
            f.has_failures = true;
            f.output_lines.push(line.to_string());
            continue;
        }

        // Default: pass through (warnings, non-test output, etc.)
        f.output_lines.push(line.to_string());
    }

    // If we captured failure blocks, show them + summary
    if f.has_failures {
        let mut result = f.output_lines.join("\n");
        if !f.summary_lines.is_empty() {
            push_line(&mut result, &f.summary_lines.join("\n"));
        }
        return result;
    }

    // If there were compile errors but no test failures, show build error summary
    if f.has_compile_errors && !exit_ok {
        // Just show last 15 meaningful lines
        let lines: Vec<&str> = f
            .output_lines
            .iter()
            .map(String::as_str)
            .filter(|l| !l.trim().is_empty())
            .collect();
        let last = lines
            .iter()
            .rev()
            .take(15)
            .rev()
            .copied()
            .collect::<Vec<_>>();
        return last.join("\n");
    }

    // All passed — return just the summary
    if !f.summary_lines.is_empty() {
        return f.summary_lines.join("\n");
    }

    // Fallback: return raw output (shouldn't normally reach here)
    let result = output.to_string();
    if exit_ok && result.trim().is_empty() {
        // NOTE: This emptiness check runs before combine_output, so it only
        // considers stdout. The cargo test profile has keep_stderr: None, so
        // combine_output never appends stderr on success — the result is empty
        // too. If keep_stderr were ever added to the cargo test profile, this
        // check would produce [cargo test: ok] even when stderr contains
        // warnings, while combine_output would then append them after it.
        // That's arguably better behavior (tests passed, warnings are
        // secondary), but the coupling should be intentional.
        "[cargo test: ok]".to_string()
    } else {
        result
    }
}

/// Parse a single `ls -l` line into its listing entry. Returns `None` for the
/// `total` header, blank lines, the `.`/`..` entries, and anything that is not
/// an `ls -l` row.
fn parse_ls_line(line: &str) -> Option<ListingEntry> {
    if line.starts_with("total ") || line.trim().is_empty() {
        return None;
    }
    let mut parts = line.split_whitespace();
    let permissions = parts.next()?;
    if permissions.len() < 10
        || !(permissions.starts_with('-')
            || permissions.starts_with('d')
            || permissions.starts_with('l'))
    {
        return None;
    }
    // `l` (a link) is a file entry: a link is never grouped with directories.
    let is_dir = permissions.starts_with('d');
    parts.next(); // link count
    parts.next(); // owner
    parts.next(); // group
    // `ls -l` prints bytes, which the listing renders human-readable. A `-h`
    // listing's `1.0K` and the `?` of an entry `ls` itself could not stat are
    // already display text and pass through unchanged.
    let size = parts.next()?;
    let size = size
        .parse::<u64>()
        .ok()
        .map_or_else(|| size.to_string(), human_readable_size);
    parts.next(); // month
    parts.next(); // day
    parts.next(); // time or year
    let name = parts.collect::<Vec<_>>().join(" ").trim().to_string();
    if name.is_empty() || name == "." || name == ".." {
        return None;
    }
    // Strip symlink target
    let name = name
        .split(" -> ")
        .next()
        .expect("split always yields at least one element")
        .to_string();
    if name.is_empty() {
        return None;
    }
    Some(ListingEntry {
        name,
        is_dir,
        // A directory never carries a size (see `ListingEntry::size`).
        size: if is_dir { None } else { Some(size) },
    })
}

/// Compress `ls -l` output into the compact listing ([`format_listing`]).
///
/// Non-`-l` output (without a `total N` header) passes through unchanged.
pub(super) fn compact_ls(output: &str, _exit_code: i32) -> String {
    // If the output lacks a "total N" header, it's not in `-l` format.
    // `parse_ls_line` can only parse `-l` lines — passing non-`-l` output
    // through would trigger the "(empty)" false positive.
    if !output.lines().any(|line| line.starts_with("total ")) {
        return output.to_string();
    }

    let mut entries: Vec<ListingEntry> = Vec::new();
    let mut lines_seen = 0usize;

    for line in output.lines() {
        if line.starts_with("total ") || line.trim().is_empty() {
            continue;
        }
        lines_seen += 1;
        if let Some(entry) = parse_ls_line(line) {
            entries.push(entry);
        }
    }

    if lines_seen == 0 {
        // Only the header of an empty `-A` listing: it stays as printed.
        return output.to_string();
    }

    format_listing(&entries)
}

/// Main entry point for shell output processing.
///
/// Routes command output through the profile system: select profile → apply
/// profile pipeline → combine → finish. Custom output transforms for commands
/// like `cargo test` (state machine) and `ls` (compact parser) are handled
/// through the profile's `output_transform` field rather than as hardcoded
/// special cases.
fn process_shell_output(
    command: &str,
    stdout: &str,
    stderr: &str,
    exit_code: i32,
    elapsed: Duration,
) -> String {
    let segments = extract_command_segments(command);
    let is_chained = segments.len() > 1;
    let profile = select_profile(&segments, is_chained);
    apply_profile_pipeline(profile, stdout, stderr, exit_code, elapsed)
}

/// Apply strip_lines filter.
fn apply_strip_lines(output: &str, profile: &Profile) -> String {
    output
        .lines()
        .filter(|l| {
            if let Some(ref set) = profile.strip_lines
                && set.is_match(l)
            {
                return false;
            }
            true
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build a head/tail sandwich with an omission marker in the middle.
///
/// `marker_verb` is the verb shown in the marker line — "omitted" for head/tail
/// sandwiches, "truncated" for the `max_lines`-only cap. It is a plain string
/// rather than an enum; a typo would silently alter tool output, which is
/// acceptable for the fixed call sites given the existing tests.
///
/// Edge cases:
/// - `head=0` (tail-only, e.g., `ping`, `gh`, `helm` profiles, and the
///   `max_lines`-only cap, which calls this as `(output, max, 0, "truncated")`):
///   no leading newline before the omission marker. A future `max_lines = 0`
///   profile would hit this path too (unreachable today — profiles.rs minimum
///   is 20) and would format the marker without a leading newline.
/// - `tail=0` (head-only, e.g., `git log` profile): no trailing newline after
///   the omission marker.
fn format_sandwich(output: &str, head: usize, tail: usize, marker_verb: &str) -> String {
    let lines: Vec<&str> = output.lines().collect();
    let total = lines.len();
    if total <= head + tail {
        return output.to_string();
    }
    let omitted = total - head - tail;

    // Build the result directly from &str slices, avoiding intermediate Vec<String> copies.
    let mut result = lines[..head].join("\n");
    if result.is_empty() {
        // head=0: no leading newline before the omission marker
        let _ = write!(result, "... ({omitted} lines {marker_verb})");
    } else {
        let _ = write!(result, "\n... ({omitted} lines {marker_verb})");
    }
    if tail > 0 {
        let _ = write!(result, "\n{}", lines[total - tail..].join("\n"));
    }
    result
}

/// Apply line truncation: head/tail sandwich (byte-gated), `max_lines`-only absolute cap, or passthrough.
///
/// Returns `(truncated_output, pre_truncation_copy)` where the copy captures the
/// full output before truncation for potential spilling by `finish_shell_output`.
/// Head/tail is gated on `TOOL_OUTPUT_BUDGET_BYTES` — small outputs are shown in full
/// regardless of configured head/tail line counts.
fn apply_line_truncation(output: &str, profile: &Profile) -> (String, Option<String>) {
    let head = profile.head_lines.unwrap_or(0);
    let tail = profile.tail_lines.unwrap_or(0);
    let max = profile.max_lines;

    // Fast path: no truncation configured at all
    if head == 0 && tail == 0 && max.is_none() {
        return (output.to_string(), None);
    }

    let line_count = output.lines().count();

    // Capture pre-truncation output for spilling — only when head/tail
    // actually truncates (byte threshold exceeded + would reduce lines).
    let should_sandwich = (head > 0 || tail > 0)
        && line_count > head + tail
        && output.len() > TOOL_OUTPUT_BUDGET_BYTES;

    let pre_truncation = if should_sandwich {
        Some(output.to_string())
    } else {
        None
    };

    // Single-pass truncation:
    // 1. Head/tail sandwich (byte-gated), OR
    // 2. max_lines-only absolute cap (head-only sandwich with "truncated" verb), OR
    // 3. passthrough.
    let result = if should_sandwich {
        format_sandwich(output, head, tail, "omitted")
    } else if let Some(max) = max {
        format_sandwich(output, max, 0, "truncated")
    } else {
        output.to_string()
    };

    // Invariant: head+tail+1 <= max_lines — guaranteed by profile configs.
    debug_assert!(
        !should_sandwich || max.is_none_or(|m| result.lines().count() <= m),
        "sandwich result ({}) exceeds max_lines ({:?}) — profile invariant violated",
        result.lines().count(),
        max,
    );

    (result, pre_truncation)
}

/// Run the full profile-based processing pipeline on pre-processed output.
///
/// Upstream processing (`execute()`) handles ANSI stripping, and the pipe read
/// cap (`SHELL_PIPE_READ_CAP`, 256 KB) bounds output collected during execution.
/// Both stdout and stderr are scrubbed once at pipeline entry so all paths
/// (early-return and main) consistently receive clean data.  No further
/// scrubbing is needed anywhere downstream — all derived output (truncated,
/// pre-head/tail for spill, on_empty messages) originates
/// from this single scrubbed source.
///
/// Early-return stages (on_empty) call `combine_output` only.  The main path continues through `combine_output` →
/// `finish_shell_output` (timing, spill-to-file).  `TOOL_OUTPUT_BUDGET_BYTES`
/// (5 KB) gates the head/tail truncation and spill preview — it is a trigger,
/// not a truncation cutoff.  `finish_shell_output` → `try_spill_to_file`'s
/// use of `crate::util::truncate_tool_output` (5 KB head+tail) provides a final
/// safety net for output that still exceeds the threshold after all pipeline
/// stages.
///
/// Stages: line filters, collapse, truncate, line_truncation,
///         on_empty, output_transform.
fn apply_profile_pipeline(
    profile: &Profile,
    output: &str,
    stderr: &str,
    exit_code: i32,
    elapsed: Duration,
) -> String {
    // Scrub both stdout and stderr once at pipeline entry so all
    // downstream paths (on_empty, main) uniformly receive clean
    // data.  Scrubbing before keep_stderr filtering means credential
    // lines that matched keep_stderr patterns will be dropped entirely
    // because the redacted version no longer matches — this is acceptable
    // since credentials should never appear in output.  No further
    // credential scrubbing is needed anywhere in the pipeline; all
    // derived data (truncated output, spill output) originates from
    // this single scrubbed source.
    let stderr = scrub_credentials(stderr);
    let output = scrub_credentials(output);

    // Local closure capturing the trailing combine_output arguments (stderr,
    // exit_code, keep_stderr) to reduce repetition across all call sites.
    // Output is already scrubbed at pipeline entry.
    let combine =
        |output: &str| combine_output(output, &stderr, exit_code, profile.keep_stderr.as_ref());

    // Stage 1: strip lines
    let mut processed = apply_strip_lines(&output, profile);

    // Stage 2: collapse blank lines (runs >2 → 2).
    processed = collapse_blank_lines(&processed);

    // Stage 3: truncate long lines
    if let Some(max) = profile.max_line_len {
        processed = truncate_line_width(&processed, max);
    }

    // Stage 4: line truncation (head/tail + max_lines).
    // Returns pre-truncation output for spilling by finish_shell_output,
    // so the complete content remains accessible even when truncation
    // reduces the inline view to a snippet.
    let (truncated, pre_head_tail) = apply_line_truncation(&processed, profile);
    processed = truncated;

    // Stage 5: on_empty — fallback when all output stripped
    if processed.trim().is_empty()
        && let Some(msg) = profile.on_empty
    {
        // When on_fail_msg is set, use it (fully replacing the message
        // including the "(failed)" suffix) to avoid contradictory output
        // like "[cargo clippy: ok] (failed)".
        if exit_code != 0
            && let Some(fail_msg) = profile.on_fail_msg
        {
            let secs = elapsed.as_secs_f64();
            return combine(&format!("{fail_msg} ({secs:.1}s)"));
        }
        let exit_note = if exit_code == 0 { "" } else { " (failed)" };
        let secs = elapsed.as_secs_f64();
        return combine(&format!("{msg}{exit_note} ({secs:.1}s)"));
    }

    // Stage 6: output transform — replaces processed output before combine/finish.
    // This allows profiles to apply custom transformations (e.g., cargo test state
    // machine, ls compaction) that operate on the full output after standard
    // line-level processing has been applied.
    // `standalone_only` profiles are already skipped at profile-selection time
    // for chained commands, so the transform here is always applicable.
    if let Some(transform) = profile.output_transform {
        processed = transform(&processed, exit_code);
    }

    let combined = combine(&processed);
    finish_shell_output(combined, elapsed, pre_head_tail.as_deref())
}

/// Decode raw shell output bytes (lossy UTF-8) and strip ANSI escape sequences.
///
/// This is the first step before further processing such as credential scrubbing
/// ([`strip_and_scrub`]).
fn decode_and_strip_ansi(data: &[u8]) -> String {
    let decoded = String::from_utf8_lossy(data);
    strip_ansi_escapes(&decoded)
}

/// Format the tool-facing exit-status note: `[exit status: N]`, or
/// `[exit status: terminated by signal]` when the process was killed by a
/// signal (exit code `None`). Shared by the foreground and background paths
/// so the two renderings cannot drift.
pub(super) fn format_exit_status_note(exit_code: Option<i32>) -> String {
    match exit_code {
        Some(c) => format!("[exit status: {c}]"),
        None => "[exit status: terminated by signal]".to_string(),
    }
}

/// Decode raw shell output bytes (lossy UTF-8), strip ANSI escape sequences,
/// then scrub credentials.
///
/// Builds on [`decode_and_strip_ansi`] by additionally applying credential
/// scrubbing. Use this whenever you need to process raw shell bytes into
/// display-safe text.
fn strip_and_scrub(data: &[u8]) -> String {
    scrub_credentials(&decode_and_strip_ansi(data))
}

/// Truncate any single line exceeding `max_line_len` with a note.
fn truncate_line_width(input: &str, max_line_len: usize) -> String {
    let mut result = String::with_capacity(input.len());
    for line in input.lines() {
        if line.len() > max_line_len {
            let cut = line.floor_char_boundary(max_line_len);
            push_line(&mut result, &line[..cut]);
            let _ = write!(
                result,
                "\n... ({} more chars on this line)",
                line[cut..].chars().count()
            );
        } else {
            push_line(&mut result, line);
        }
    }
    result
}

/// Format the spill file header line shared by all spill hint producers.
fn format_spill_header(path: &Path, byte_count: usize, line_count: usize) -> String {
    format!(
        "[Output saved to {} ({} bytes, {} lines)]\n\
         [view with: read {}]\n",
        path.display(),
        byte_count,
        line_count,
        path.display(),
    )
}

/// Build a head/tail preview for large output that was spilled to a file.
fn format_spill_preview(output: &str, path: &Path) -> String {
    let line_count = output.lines().count();
    let byte_count = output.len();
    let header = format_spill_header(path, byte_count, line_count);
    format!("{header}{}", format_sandwich(output, 5, 5, "omitted"))
}

/// Write content to the agent temp directory with the given filename.
/// Content should already be credential-scrubbed. The path is recorded under
/// the owning agent id (owner-deletes-at-end).
fn write_to_spill(content: &str, filename: &str) -> Option<std::path::PathBuf> {
    let dir = agent_temp_dir()?;
    let path = dir.join(filename);
    std::fs::write(&path, content).ok()?;
    record_spill_owner(path.clone());
    Some(path)
}

/// Write pre-scrubbed output to a random spill file.
/// Caller is responsible for scrubbing credentials before calling this function.
fn spill_output(output: &str) -> Option<std::path::PathBuf> {
    let filename = crate::tools::path::format_spill_filename();
    write_to_spill(output, &filename)
}

/// If output exceeds threshold, spill to a temp file and return a preview.
/// The full output is saved to a file; the inline preview is a short summary.
///
/// Reused by read-only tools (e.g. [`crate::tools::mahbot_debug`]) to keep
/// large result sets out of the LLM context while still making them available
/// on demand via the read path.
pub(crate) fn try_spill_to_file(output: String, threshold_bytes: usize) -> String {
    if output.len() <= threshold_bytes {
        return output;
    }

    match spill_output(&output) {
        Some(path) => format_spill_preview(&output, &path),
        None => crate::util::truncate_tool_output(&output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::test_ws;
    use tempfile::TempDir;

    #[cfg(unix)]
    use crate::util::test::env_lock;
    use crate::util::test::set_env_var;

    // ── Table-driven test helpers ─────────────────────────────────────
    // These helpers reduce boilerplate for process_shell_output and
    // filter_cargo_test_output test groups. Each case carries its own
    // assertions (contains, not_contains, eq) with a descriptive name
    // so test failures show exactly which scenario broke.

    /// Shared assertion helper: verify that all `contains` strings are present
    /// and all `not_contains` strings are absent in `result`.
    fn assert_contains_not_contains(
        name: &str,
        result: &str,
        contains: &[&str],
        not_contains: &[&str],
    ) {
        for &s in contains {
            assert!(
                result.contains(s),
                "[{name}] expected contains {s:?}\n  got: {result:?}",
            );
        }
        for &s in not_contains {
            assert!(
                !result.contains(s),
                "[{name}] expected NOT contains {s:?}\n  got: {result:?}",
            );
        }
    }

    /// A test case for [`process_shell_output`] with multi-assertion support.
    #[derive(Default)]
    struct ShellOutputCase {
        /// Human-readable name for failure diagnostics.
        name: &'static str,
        /// Canonical/shell command string passed to select_profile.
        command: &'static str,
        stdout: &'static str,
        /// Stderr input. Default: `""`.
        stderr: &'static str,
        /// Exit code. Default: `0`.
        exit_code: i32,
        /// Elapsed time in seconds. Default: `0.0`.
        elapsed_secs: f64,
        /// Strings that must all be present in the output.
        contains: &'static [&'static str],
        /// Strings that must all be absent from the output.
        not_contains: &'static [&'static str],
        /// If set, asserts `result.trim() == eq` (leading/trailing whitespace
        /// is stripped before comparison).
        eq: Option<&'static str>,
    }

    #[derive(Default)]
    /// A test case for [`filter_cargo_test_output`].
    struct CargoTestFilterCase {
        /// Human-readable name for failure diagnostics.
        name: &'static str,
        output: &'static str,
        exit_code: i32,
        contains: &'static [&'static str],
        not_contains: &'static [&'static str],
    }

    /// Run [`process_shell_output`] for each case and assert expectations.
    ///
    /// `contains` and `not_contains` are checked against the raw result.
    /// `eq` (if set) compares against `result.trim()` — leading/trailing
    /// whitespace is stripped, consistent with how profiles produce
    /// on_empty messages (e.g. `"[cargo test: ok]"`).
    fn check_shell_output(cases: &[ShellOutputCase]) {
        for case in cases {
            let result = process_shell_output(
                case.command,
                case.stdout,
                case.stderr,
                case.exit_code,
                Duration::from_secs_f64(case.elapsed_secs),
            );
            assert_contains_not_contains(case.name, &result, case.contains, case.not_contains);
            if let Some(expected) = case.eq {
                assert_eq!(
                    result.trim(),
                    expected,
                    "[{}] expected eq {expected:?}",
                    case.name,
                );
            }
        }
    }

    /// Run [`filter_cargo_test_output`] for each case and assert expectations.
    fn check_cargo_test_filter(cases: &[CargoTestFilterCase]) {
        for case in cases {
            let result = filter_cargo_test_output(case.output, case.exit_code);
            assert_contains_not_contains(case.name, &result, case.contains, case.not_contains);
        }
    }

    // ── Consolidated table-driven tests ───────────────────────────────
    //
    // Each `_cases` function replaces multiple individual test functions
    // that followed the same pattern (call process_shell_output →
    // assert contains/not_contains). Adding a new scenario is a single
    // struct literal with a descriptive name.

    #[test]
    fn cargo_test_filter_cases() {
        // Cargo test output filter state machine — consolidated table.
        let cases: &[CargoTestFilterCase] = &[
            CargoTestFilterCase {
                name: "failure block captures failures and panic message",
                output: "\n\
                    Compiling foo v1.0.0\n\
                    test test1 ... ok\n\
                    test test2 ... FAILED\n\
                    \n\
                    failures:\n\
                    \n\
                    ---- test2 stdout ----\n\
                    thread 'test2' panicked at src/lib.rs:42:\n\
                    assertion failed\n\
                    \n\
                    \n\
                    failures:\n\
                    test2\n\
                    \n\
                    test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n\
                ",
                exit_code: 1,
                contains: &["test2 ... FAILED", "assertion failed", "test result:"],
                not_contains: &["Compiling", "test1 ... ok"],
            },
            CargoTestFilterCase {
                name: "all pass returns summary",
                output: "\
                    Compiling foo v1.0.0\n\
                    Checking bar v2.0.0\n\
                    test test1 ... ok\n\
                    test test2 ... ok\n\
                    \n\
                    test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n\
                ",
                exit_code: 0,
                contains: &["test result:"],
                not_contains: &["Compiling", "Checking", "test1 ... ok", "test2 ... ok"],
            },
            CargoTestFilterCase {
                name: "compile error fallback preserves errors",
                output: "\
                    Compiling foo v1.0.0\n\
                    error[E0425]: cannot find value `bar` in this scope\n\
                     --> src/lib.rs:1:5\n\
                    \n\
                    error: could not compile `foo` due to 1 previous error\n\
                ",
                exit_code: 1,
                contains: &["error[E0425]", "could not compile"],
                not_contains: &["Compiling"],
            },
            CargoTestFilterCase {
                name: "Running preserved in test output",
                output: "\
                    Compiling foo v1.0.0\n\
                    Running unittests src/lib.rs\n\
                    test test1 ... ok\n\
                    test test2 ... FAILED\n\
                    \n\
                    failures:\n\
                    \n\
                    ---- test2 stdout ----\n\
                    assertion failed\n\
                    \n\
                    failures:\n\
                    test2\n\
                    \n\
                    test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n\
                ",
                exit_code: 1,
                contains: &["Running unittests", "test2 ... FAILED", "test result:"],
                not_contains: &["Compiling"],
            },
        ];
        check_cargo_test_filter(cases);
    }

    #[expect(clippy::too_many_lines)]
    #[test]
    fn profile_selection_cases() {
        // Profile selection/dispatch tests — consolidated table.
        // Includes detection of correct profiles from canonical commands,
        // fallback behavior for unknown/builtin-only commands, and chained
        // command dispatch that selects the first matching segment.
        let cases: &[ShellOutputCase] = &[
            ShellOutputCase {
                name: "cargo --release test triggers state machine",
                command: "cargo --release test",
                eq: Some("[cargo test: ok]"),
                ..Default::default()
            },
            ShellOutputCase {
                name: "git -C /repo diff triggers git diff on_empty",
                command: "git -C /repo diff",
                contains: &["no changes"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "unknown tool falls through to generic",
                command: "some_obscure_tool --flag",
                stdout: "some\nrandom\noutput\n",
                contains: &["some", "output"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "empty command uses fallback",
                command: "",
                stdout: "hello world",
                contains: &["hello"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "builtins-only falls through to generic",
                command: "cd .. && cd /tmp",
                stdout: "some output",
                contains: &["some output"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "chained command selects first matching profile (pnpm install)",
                command: "cd frontend && pnpm install && pnpm build",
                stdout: "Already up to date\nsome output\n",
                not_contains: &["Already up to date"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "chained cargo test falls through to GEN_FALLBACK",
                command: "cd project && cargo test",
                stdout: "Compiling foo v1.0.0\ntest test1 ... ok\ntest test2 ... FAILED\n\nfailures:\n\n---- test2 stdout ----\npanic!\n\nfailures:\n    test2\n\ntest result: FAILED. 1 passed; 1 failed\n",
                exit_code: 1,
                contains: &["test2 ... FAILED", "Compiling", "test1 ... ok"],
                ..Default::default()
            },
            // Regression: chained cargo test with compile errors and exit_code=0
            // must NOT produce the misleading "[cargo test: ok]".
            ShellOutputCase {
                name: "chained cargo test compile error regression",
                command: "cargo test --lib || true",
                stdout: "",
                stderr: "error[E0425]: cannot find value `x` in this scope\n  --> src/lib.rs:2:21\n   |\n2 |     let y = x + 1;\n   |             ^ not found in this scope\n",
                exit_code: 0,
                not_contains: &["[cargo test: ok]"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "chained git log preserves content",
                command: "cd repo && git log --oneline",
                stdout: "commit abc123\nAuthor: test\nDate:   Mon Jan 1\n\n    initial commit\n",
                contains: &["commit", "Author"],
                ..Default::default()
            },
            // ── Multi-line commands (newline = command separator) ─────────
            // Regression: newline splitting in the shared
            // extract_command_segments changes is_chained, which disables
            // standalone_only output transforms (compact_ls, cargo test state
            // machine) for multi-line commands. Raw output must pass through.
            ShellOutputCase {
                name: "multi-line ls skips compact_ls (newline = chained)",
                command: "ls -la\ncat README.md",
                stdout: "total 8\n-rw-r--r--  1 user  group  2048 May 21 10:00 file.txt\n",
                contains: &["total 8", "file.txt"],
                not_contains: &["Summary:"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "multi-line cargo test skips state machine",
                command: "cargo test --lib\ncat notes.txt",
                stdout: "Compiling foo v1.0.0\ntest test1 ... ok\ntest result: ok. 1 passed; 1 failed\n",
                contains: &["Compiling foo", "test1 ... ok", "test result:"],
                not_contains: &["[cargo test: ok]"],
                ..Default::default()
            },
            // Heredoc bodies are removed before segmentation, so a heredoc
            // command stays a single segment and keeps its profile (the cargo
            // test state machine still applies — it is not skipped as chained).
            ShellOutputCase {
                name: "heredoc command stays single segment for profile selection",
                command: "cargo test --lib <<EOF\nbody\nEOF",
                stdout: "test test1 ... ok\ntest result: ok. 1 passed; 0 failed\n",
                eq: Some("test result: ok. 1 passed; 0 failed"),
                ..Default::default()
            },
            // Regression: a heredoc whose UNQUOTED body contains a
            // command substitution emits that substitution as its own segment
            // (it must remain scanned — bash executes `$()` in unquoted
            // bodies), so the command becomes chained and standalone-only
            // profiles (the cargo test state machine) are skipped.
            ShellOutputCase {
                name: "heredoc with substitution body becomes chained",
                command: "cargo test --lib <<EOF\n$(echo hi)\nEOF",
                stdout: "Compiling foo v1.0.0\ntest test1 ... ok\ntest result: ok. 1 passed; 0 failed\n",
                contains: &["Compiling foo", "test1 ... ok", "test result:"],
                not_contains: &["[cargo test: ok]"],
                ..Default::default()
            },
            // ── npx-wrapped tools (profile selection) ─────────────────
            ShellOutputCase {
                name: "npx eslint selects eslint profile",
                command: "npx eslint .",
                contains: &["[eslint: ok]"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "npx prettier selects prettier profile",
                command: "npx prettier --check file.js",
                stdout: "unchanged",
                contains: &["unchanged"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "npx tsc selects tsc profile",
                command: "npx tsc --noEmit",
                contains: &["[tsc: ok]"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "npx vitest selects vitest profile",
                command: "npx vitest --run",
                stdout: "stdout: Tests passed\nPASS src/test.ts\n",
                // Vitest profile strips "PASS" prefixed lines but keeps "stdout:" lines
                not_contains: &["PASS"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "npx with flags before subcommand selects eslint profile",
                command: "npx --yes eslint .",
                contains: &["[eslint: ok]"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "unknown npx tool falls through to generic",
                command: "npx some_obscure_tool --flag",
                stdout: "some\nrandom\noutput\n",
                contains: &["some", "output"],
                ..Default::default()
            },
        ];
        check_shell_output(cases);
    }

    #[test]
    fn mahbot_chrome_envelope_line_bypasses_the_line_cap() {
        // The chrome CLI emits ONE line of JSON; its profile must not apply
        // GEN_FALLBACK's 500-byte line cap, which would truncate the envelope
        // mid-JSON and drop the page content.
        let envelope = format!(
            r#"{{"schema":1,"action":"open","ok":true,"kind":"ok","content":"{}","url":"https://example.com/"}}"#,
            "x".repeat(600)
        );
        let result = process_shell_output(
            "mahbot chrome open https://example.com",
            &envelope,
            "",
            0,
            Duration::from_secs_f64(1.0),
        );
        assert_contains_not_contains(
            "mahbot chrome envelope",
            &result,
            &[r#""content":""#],
            &["more chars on this line"],
        );
        assert!(
            result.contains(&envelope),
            "envelope line must survive intact"
        );
    }

    #[expect(clippy::too_many_lines)]
    #[test]
    fn tool_profile_cases() {
        // Specific tool profile tests — consolidated table.
        // Covers strip, on_empty, and transform behaviors of named
        // tool profiles (git, docker, df, du, make, rsync, tsc, gh,
        // terraform, pytest, and the generic fallback pipeline).
        let cases: &[ShellOutputCase] = &[
            ShellOutputCase {
                name: "git diff no changes via on_empty",
                command: "git diff",
                contains: &["no changes"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "docker build success via on_empty",
                command: "docker build -t myimage .",
                stdout: "Step 1/3 : FROM alpine\n ---> abc123\nStep 2/3 : RUN echo hi\n ---> Using cache\nStep 3/3 : CMD [\"sh\"]\n ---> def456\nSuccessfully built abc123\nSuccessfully tagged myimage:latest\n",
                contains: &["[docker"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "git log preserves content",
                command: "git log --oneline",
                stdout: "commit abc123\nAuthor: test\nDate:   Mon Jan 1\n\n    initial commit\n\ncommit def456\nAuthor: test\nDate:   Tue Jan 2\n\n    second commit\n\n",
                contains: &["commit", "Author"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "generic pipeline: strips ANSI, preserves content",
                command: "unknown",
                stdout: "Compiling foo v1.0.0 (/tmp)\nCompiling bar v2.0.0 (/tmp)\nresult: ok\nline1\nline2\nline3\nline3\nline3\nline3\nline3\nline3\nline3\n",
                contains: &["Compiling", "result: ok"],
                not_contains: &["\x1B["],
                ..Default::default()
            },
            ShellOutputCase {
                name: "du strips blank lines",
                command: "du -sh",
                stdout: "1.0K\t./file1\n\n2.0K\t./file2\n\n\n3.0K\t./file3",
                not_contains: &["\n\n"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "make strips directory noise",
                command: "make",
                stdout: "make[1]: Entering directory `/tmp'\nmake[1]: Leaving directory `/tmp'\ncc -c file.c\nNothing to be done",
                not_contains: &["Entering directory", "Nothing to be done"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "rsync success shows transfer summary",
                command: "rsync -avz source/ dest/",
                stdout: "building file list ... done\nsent 100 bytes  received 50 bytes\n\ntotal size is 98765  speedup is 658.43\n",
                contains: &["building file list", "total size is", "98765"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "tsc on empty returns ok",
                command: "tsc --noEmit",
                eq: Some("[tsc: ok] (0.0s)"),
                ..Default::default()
            },
            ShellOutputCase {
                name: "tsc on empty shows timing",
                command: "tsc --noEmit",
                elapsed_secs: 3.2,
                contains: &["(3.2s)"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "docker strips build steps and shows on_empty",
                command: "docker build -t myapp .",
                stdout: "Step 1/10 : FROM node:18\nStep 2/10 : WORKDIR /app\n ---> Using cache\nSuccessfully built abc123\nSuccessfully tagged myapp:latest\n",
                contains: &["[docker: ok]"],
                not_contains: &["Step "],
                ..Default::default()
            },
            ShellOutputCase {
                name: "gh strips warning noise, preserves output",
                command: "gh pr create --fill",
                stdout: "  \n - some detail\nwarning: consider updating gh\n✓ Created pull request\n",
                contains: &["Created pull request"],
                not_contains: &["warning:"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "terraform shows no changes message",
                command: "terraform plan",
                stdout: "data.aws_region.current: Refreshing state...\nNo changes. Your infrastructure matches the configuration.\n",
                contains: &["No changes", "infrastructure matches"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "pytest strips collected count",
                command: "pytest",
                stdout: "============================= test session starts ==============================\ncollected 5 items\n\n.test..\n\n============================== 5 passed ==============================\n",
                not_contains: &["collected"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "python -m pytest falls through to generic (collected preserved)",
                command: "python -m pytest tests/",
                stdout: "============================= test session starts ==============================\ncollected 5 items\n\n.test..\n\n============================== 5 passed ==============================\n",
                contains: &["collected"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "poetry run pytest falls through to generic (collected preserved)",
                command: "poetry run pytest tests/",
                stdout: "============================= test session starts ==============================\ncollected 5 items\n\n.test..\n\n============================== 5 passed ==============================\n",
                contains: &["collected"],
                ..Default::default()
            },
        ];
        check_shell_output(cases);
    }

    #[test]
    fn compact_ls_cases() {
        // ls compaction profile tests — consolidated table.
        // compact_ls transforms `ls -la` output into a compact summary.
        // Plain `ls` (no -l) and chained/piped commands are excluded
        // from compaction via standalone_only and selection logic.
        let cases: &[ShellOutputCase] = &[
            ShellOutputCase {
                name: "empty directory shows (empty)",
                command: "ls -la",
                stdout: "total 0\ndrwxr-xr-x  2 user  group  64 May 21 10:00 .\ndrwxr-xr-x  3 user  group  96 May 21 10:00 ..\n",
                eq: Some("(empty)"),
                ..Default::default()
            },
            ShellOutputCase {
                name: "mixed files and dirs shows summary",
                command: "ls -la",
                stdout: "total 32\ndrwxr-xr-x  5 user  group   160 May 21 10:00 .\ndrwxr-xr-x  3 user  group    96 May 21 10:00 ..\n-rw-r--r--  1 user  group  2048 May 21 10:00 main.rs\n-rw-r--r--  1 user  group  4096 May 21 10:00 lib.rs\ndrwxr-xr-x  2 user  group    64 May 21 10:00 src\nlrwxr-xr-x  1 user  group     5 May 21 10:00 link -> target\n",
                contains: &["src/", "main.rs", "lib.rs", "Summary:"],
                not_contains: &["link -> target"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "dotless files classified as no ext",
                command: "ls -la",
                stdout: "total 16\n-rw-r--r--  1 user  group  1024 May 21 10:00 Makefile\n-rw-r--r--  1 user  group  2048 May 21 10:00 README\n-rw-r--r--  1 user  group   512 May 21 10:00 .gitignore\n-rw-r--r--  1 user  group  1024 May 21 10:00 main.rs\n",
                contains: &["Makefile", "README", "no ext", ".rs"],
                not_contains: &[".Makefile", ".README"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "plain ls passes through without compaction",
                command: "ls",
                stdout: "Cargo.toml\nCargo.lock\nsrc\ntarget\nREADME.md\n",
                contains: &["Cargo.toml", "src"],
                not_contains: &["(empty)", "Summary:"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "chained ls skips compact_ls",
                command: "ls -l && echo done",
                stdout: "total 8\n-rw-r--r--  1 user  group  1024 May 21 10:00 foo\n-rw-r--r--  1 user  group  2048 May 21 10:00 bar\ndone\n",
                contains: &["done"],
                not_contains: &["Summary:"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "piped ls skips compact_ls",
                command: "ls -l | head -5",
                stdout: "total 8\n-rw-r--r--  1 user  group  1024 May 21 10:00 foo\n-rw-r--r--  1 user  group  2048 May 21 10:00 bar\n",
                contains: &["total 8"],
                not_contains: &["Summary:"],
                ..Default::default()
            },
        ];
        check_shell_output(cases);
    }

    #[test]
    fn cargo_build_cases() {
        // Cargo build/check profile tests — consolidated table.
        let cases: &[ShellOutputCase] = &[
            ShellOutputCase {
                name: "cargo build strips Compiling, preserves errors",
                command: "cargo build",
                stdout: "Compiling foo v1.0.0 (/tmp)\nCompiling bar v2.0.0 (/tmp)\n   Compiling baz v3.0.0 (/tmp)\nerror[E0425]: cannot find value\n\nFor more information about this error, try `rustc --explain E0425`.\nerror: could not compile `foo` due to 1 previous error",
                exit_code: 1,
                contains: &["error[E0425]", "could not compile"],
                not_contains: &["Compiling foo"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "cargo check strips Checking lines",
                command: "cargo check",
                stdout: "    Checking foo v1.0.0\n    Checking bar v2.0.0\n    warning: unused import\n\nwarning: 1 warning emitted\n\n    Finished `dev` profile [unoptimized] target\n",
                not_contains: &["Checking"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "cargo build strips Compiling and Finished on success",
                command: "cargo build",
                stdout: "   Compiling foo v1.0.0\n   Compiling bar v2.0.0\n    Finished dev [unoptimized]\n",
                not_contains: &["Compiling", "Finished"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "absolute cargo check strips Compiling",
                command: "/usr/local/bin/cargo check",
                stdout: "   Compiling foo v1.0.0\nwarning: unused import\n",
                not_contains: &["Compiling"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "chained cargo build strips Compiling, preserves errors",
                command: "cd project && cargo build",
                stdout: "   Compiling foo v1.0.0\n   Compiling bar v2.0.0\nerror[E0425]: cannot find value\n",
                exit_code: 1,
                contains: &["error[E0425]"],
                not_contains: &["Compiling"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "cargo build keeps stderr warnings on success",
                command: "cargo build",
                stdout: "   Compiling foo v1.0.0\n    Finished\n",
                stderr: "warning: unused import: `std::fs`\n  --> src/main.rs:1:5\n",
                contains: &["warning:"],
                ..Default::default()
            },
            // ── on_fail_msg tests ────────────────────────────────────
            ShellOutputCase {
                name: "cargo clippy failure shows on_fail_msg (no (failed) suffix)",
                command: "cargo clippy",
                exit_code: 1,
                eq: Some("[cargo clippy: failed] (0.0s)"),
                ..Default::default()
            },
            ShellOutputCase {
                name: "cargo clippy failure filters progress lines from stderr",
                command: "cargo clippy",
                stderr: "    Checking mahbot v0.1.0 (/Users/user/mahbot)\nwarning: unused import: `std::fs`\n  --> src/main.rs:1:5\n",
                exit_code: 1,
                contains: &["warning:", "[cargo clippy: failed]"],
                not_contains: &["Checking mahbot"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "cargo clippy failure omits stderr when no keep_stderr match",
                command: "cargo clippy",
                stderr: "    Checking mahbot v0.1.0\n    Finished dev [unoptimized]\n",
                exit_code: 1,
                eq: Some("[cargo clippy: failed] (0.0s)"),
                not_contains: &["Checking", "stderr:"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "cargo build failure shows on_fail_msg (no (failed) suffix)",
                command: "cargo build",
                exit_code: 1,
                eq: Some("[cargo: failed] (0.0s)"),
                ..Default::default()
            },
            ShellOutputCase {
                name: "cargo build success still shows ok (backward compat)",
                command: "cargo build",
                eq: Some("[cargo: ok] (0.0s)"),
                ..Default::default()
            },
        ];
        check_shell_output(cases);
    }

    /// Create a minimal [`Workspace`] from a path for testing.

    #[test]
    fn shell_safe_env_vars() {
        for var in SAFE_ENV_VARS {
            let lower = var.to_lowercase();
            assert!(
                !lower.contains("key") && !lower.contains("secret") && !lower.contains("token")
            );
        }
        assert!(SAFE_ENV_VARS.contains(&"PATH"));
        assert!(SAFE_ENV_VARS.contains(&"HOME") || SAFE_ENV_VARS.contains(&"USERPROFILE"));
        assert!(SAFE_ENV_VARS.contains(&"TERM"));
    }

    /// `build_shell_command` clears the parent environment (except `$USER`
    /// /`$USERNAME`, see [`build_shell_command()`]) and only exposes
    /// [`SAFE_ENV_VARS`] with baseline values (CWE-200). Verify by running
    /// `env` through the built command.
    ///
    /// Acquires the shared [`env_lock()`] because `build_shell_command` →
    /// `resolved_shell_path` → `extra_shell_path_prefixes` reads `$CARGO_HOME`
    /// from the environment.
    #[cfg(unix)]
    #[tokio::test]
    async fn build_shell_command_isolates_environment() {
        let tmp = TempDir::new().expect("tempdir");
        // Acquire env_lock while building the command since extra_shell_path_prefixes
        // reads $CARGO_HOME — concurrent tests in other modules may write it, so
        // holding the shared lock prevents the theoretical data race.
        let mut cmd = {
            let _guard = env_lock().lock().unwrap_poison();
            build_shell_command("env", tmp.path())
        };

        // We can't inspect env vars on a Command directly; spawn it and check.
        let output = cmd.output().await.expect("env should run");
        let stdout = String::from_utf8_lossy(&output.stdout);

        // Safe vars with baseline values should be present.
        assert!(stdout.contains("HOME="), "HOME must be in safe env");
        assert!(stdout.contains("PATH="), "PATH must be in safe env");

        // Parent-process env vars not in SAFE_ENV_VARS must NOT leak.
        // CARGO_HOME is commonly set but NOT in SAFE_ENV_VARS.
        assert!(
            !stdout.contains("CARGO_HOME="),
            "CARGO_HOME must not leak into subprocess env"
        );
    }

    /// A direct program run takes its arguments as argv: nothing in them is
    /// parsed as shell syntax, a non-zero exit is a completed run (reported with
    /// the standard note), and the output is decoded like the shell's.
    #[cfg(unix)]
    #[tokio::test]
    async fn run_program_with_timeout_passes_argv_and_annotates_exit() {
        let tmp = TempDir::new().expect("tempdir");
        let ws = crate::workspace::test_ws(tmp.path());
        let output = run_program_with_timeout(
            &ws,
            Path::new("/bin/sh"),
            &[
                "-c".to_string(),
                // `$0` is the placeholder below; `$1` is the hostile-looking
                // value, which must arrive verbatim rather than be executed.
                "printf '%s' \"$1\"; exit 3".to_string(),
                "sh".to_string(),
                "a; echo pwned".to_string(),
            ],
            "the test program",
        )
        .await
        .expect("a completed run");

        // The hostile-looking argument arrives verbatim as the only output
        // line; the status note follows it as its own paragraph.
        assert_eq!(
            output.lines().collect::<Vec<_>>(),
            ["a; echo pwned", "", "[exit status: 3]"]
        );
    }

    /// A direct program run reported as an outcome: output is raw and
    /// un-annotated, and only the raw streams decide whether anything was
    /// reported.
    #[cfg(unix)]
    #[tokio::test]
    async fn run_program_outcome_reports_output_and_status_without_annotations() {
        let tmp = TempDir::new().expect("tempdir");
        let ws = crate::workspace::test_ws(tmp.path());

        // Success with stdout — raw, un-annotated output.
        let out =
            run_program_outcome(&ws, Path::new("/bin/sh"), &sh_args("echo hello-prog-out")).await;
        assert!(out.success);
        assert!(out.has_output);
        assert!(out.output.contains("hello-prog-out"), "got: {}", out.output);
        assert!(!out.output.contains("[exit status"), "got: {}", out.output);
        assert_eq!(out.detail, "exit status 0");

        // Non-zero exit is a failed outcome whose output is still just the
        // program's own text.
        let out =
            run_program_outcome(&ws, Path::new("/bin/sh"), &sh_args("echo bad; exit 3")).await;
        assert!(!out.success);
        assert!(out.has_output);
        assert!(out.output.contains("bad"), "got: {}", out.output);
        assert!(!out.output.contains("[exit status"), "got: {}", out.output);
        assert_eq!(out.detail, "exit status 3");

        // No output at all: nothing was reported.
        let out = run_program_outcome(&ws, Path::new("/bin/sh"), &sh_args("printf ''")).await;
        assert!(out.success);
        assert!(!out.has_output);
        assert!(out.output.is_empty());

        // Whitespace-only output still counts as reported — the raw streams
        // decide, not the trimmed text.
        let out = run_program_outcome(&ws, Path::new("/bin/sh"), &sh_args("printf '\\n'")).await;
        assert!(out.has_output);

        // ... and so does escape-sequence-only output — reported, even though
        // the ANSI strip leaves nothing readable behind.
        let out = run_program_outcome(
            &ws,
            Path::new("/bin/sh"),
            &sh_args("printf '\\033[32m\\033[0m'"),
        )
        .await;
        assert!(out.has_output);
        assert!(out.output.is_empty(), "got: {}", out.output);

        // stderr alone counts as output.
        let out = run_program_outcome(&ws, Path::new("/bin/sh"), &sh_args("echo hi >&2")).await;
        assert!(out.has_output);
        assert!(out.output.contains("hi"), "got: {}", out.output);

        // The literal is split so this source does not itself look like a
        // credential to output scrubbers; the control below proves the scrubber
        // rewrites it, and that it therefore reaches the alarm verbatim.
        let raw = concat!("API_KEY=", "abcd1234");
        assert!(
            scrub_credentials(raw).contains("*[REDACTED]"),
            "the scrubber must rewrite this value, or the pin below is vacuous"
        );
        let out =
            run_program_outcome(&ws, Path::new("/bin/sh"), &sh_args(&format!("echo {raw}"))).await;
        assert!(out.has_output);
        assert_eq!(
            out.output.trim(),
            raw,
            "the check's output must reach the alarm unredacted"
        );
    }

    #[cfg(unix)]
    fn sh_args(script: &str) -> Vec<String> {
        vec!["-c".to_string(), script.to_string()]
    }

    #[tokio::test]
    async fn shell_executes_allowed_command() {
        let tmp = TempDir::new().expect("tempdir");
        let result = ShellTool::new(ShellMode::Full)
            .execute(&test_ws(tmp.path()), json!({"command": "echo hello"}))
            .await;
        assert!(
            result.is_ok(),
            "echo command execution should succeed: {result:?}"
        );
        let result = result.unwrap();
        assert!(result.trim().contains("hello"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shell_nonzero_exit_with_stdout_counts_as_success() {
        // `test -f` exits 1 when file doesn't exist, with no stdout/stderr.
        // Combined with `echo`, this produces stdout + non-zero exit without
        // triggering any command-specific filter profiles.
        let tmp = TempDir::new().expect("tempdir");
        let result = ShellTool::new(ShellMode::Full)
            .execute(
                &test_ws(tmp.path()),
                json!({"command": "echo partial; test -f nonexistent_file_xyz"}),
            )
            .await;
        assert!(
            result.is_ok(),
            "shell should return Ok(String) when stdout present: {result:?}"
        );
        let result = result.unwrap();
        assert!(result.contains("partial"));
        assert!(
            result.contains("[exit status: 1]"),
            "model should still see real exit status, got {result:?}",
        );
    }

    #[tokio::test]
    async fn shell_captures_exit_code() {
        let tmp = TempDir::new().expect("tempdir");
        let result = ShellTool::new(ShellMode::Full)
            .execute(
                &test_ws(tmp.path()),
                json!({"command": "ls nonexistent_dir_xyz"}),
            )
            .await;
        assert!(
            result.is_ok(),
            "command with nonexistent path should return ok: {result:?}"
        );
        let output = result.unwrap();
        assert!(
            output.contains("[exit status: 1]"),
            "output should contain exit status: {output:?}",
        );
        assert!(
            output.contains("nonexistent_dir_xyz"),
            "output should contain the error: {output:?}"
        );
    }

    // ── Background mode (Full roles only) ─────────────────────────────

    /// Run `shell_tool.execute(...)` inside the agent tool-batch context with a
    /// background-session registry, so the background/stop arguments resolve.
    async fn execute_with_bg_registry(
        shell_tool: &ShellTool,
        ws: &crate::Workspace,
        args: serde_json::Value,
        sessions: &std::sync::Arc<crate::tools::shell::BackgroundSessions>,
    ) -> anyhow::Result<String> {
        crate::agent::CURRENT_TOOL_BACKGROUND_SESSIONS
            .scope(Some(sessions.clone()), async {
                shell_tool.execute(ws, args).await
            })
            .await
    }

    #[tokio::test]
    async fn background_launch_via_tool_registers_session_and_read_tool_reads_output() {
        let tmp = TempDir::new().expect("tempdir");
        let ws = test_ws(tmp.path());
        let sessions = std::sync::Arc::new(BackgroundSessions::default());

        let output = execute_with_bg_registry(
            &ShellTool::new(ShellMode::Full),
            &ws,
            json!({"command": "echo bg-via-tool", "background": true}),
            &sessions,
        )
        .await
        .expect("background launch succeeds");

        // The message carries the output-file path.
        let path_line = output
            .lines()
            .find(|l| l.starts_with("output file:"))
            .expect("launch message must name the output file");
        let path = PathBuf::from(path_line.trim_start_matches("output file:").trim());
        assert!(
            path.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("bg_")),
            "bg_* name shape: {path:?}"
        );

        // The session is registered in the agent-scoped registry.
        assert!(sessions.contains(&path), "session must be registered");

        // The read tool can read the output file (temp-area allowlist covers
        // the .agent directory) — no allowlist changes were needed.
        let content = crate::tools::ReadTool::general()
            .execute(&ws, json!({"path": path.to_string_lossy().to_string()}))
            .await
            .expect("read tool must read the bg output file");
        assert!(
            content.contains("bg-via-tool"),
            "raw output must be in the file: {content}"
        );

        // And the annotation eventually lands.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !sessions.is_finished(&path) && std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        assert!(sessions.is_finished(&path), "session should finish");
        let content = std::fs::read_to_string(&path).expect("bg output file readable");
        assert!(
            content.contains("[exit status: 0]"),
            "annotation must be appended: {content}"
        );
    }

    #[tokio::test]
    async fn background_stop_via_tool() {
        let _env = set_env_var("MAHBOT_BG_STOP_GRACE_SECS", Some("0"));
        let tmp = TempDir::new().expect("tempdir");
        let ws = test_ws(tmp.path());
        let sessions = std::sync::Arc::new(BackgroundSessions::default());
        let tool = ShellTool::new(ShellMode::Full);

        let launch_out = execute_with_bg_registry(
            &tool,
            &ws,
            json!({"command": "sleep 30", "background": true}),
            &sessions,
        )
        .await
        .expect("launch");
        let path = launch_out
            .lines()
            .find(|l| l.starts_with("output file:"))
            .expect("output file line")
            .trim_start_matches("output file:")
            .trim();

        // Stop via the tool's `stop` argument, addressing the output-file path.
        let stop_out = execute_with_bg_registry(&tool, &ws, json!({"stop": path}), &sessions)
            .await
            .expect("stop succeeds");
        assert!(
            stop_out.contains("Background session stopped"),
            "stop message: {stop_out}"
        );
        assert!(
            sessions.is_finished(Path::new(path)),
            "stopped session must be finished"
        );
    }

    #[tokio::test]
    async fn background_stop_and_background_conflict_errors() {
        let tmp = TempDir::new().expect("tempdir");
        let ws = test_ws(tmp.path());
        let sessions = std::sync::Arc::new(BackgroundSessions::default());

        let err = execute_with_bg_registry(
            &ShellTool::new(ShellMode::Full),
            &ws,
            json!({"command": "echo hi", "background": true, "stop": "/tmp/.agent/bg_0000.out"}),
            &sessions,
        )
        .await
        .expect_err("stop + background must be rejected");
        assert!(
            err.to_string().contains("cannot be combined"),
            "error message: {err}"
        );
    }

    #[tokio::test]
    async fn background_unavailable_without_agent_context() {
        // No task-local registry (management diagnostics / tests context):
        // background mode must fail loudly instead of leaking a process.
        let tmp = TempDir::new().expect("tempdir");
        let err = ShellTool::new(ShellMode::Full)
            .execute(
                &test_ws(tmp.path()),
                json!({"command": "sleep 30", "background": true}),
            )
            .await
            .expect_err("background without an agent registry must error");
        assert!(
            err.to_string().contains("not available in this context"),
            "error message: {err}"
        );
    }

    #[test]
    fn full_description_and_schema_cover_background_capability() {
        // The Full variant gets its own prompt asset and an extended argument
        // schema describing the background capability.
        let full = ShellTool::new(ShellMode::Full);
        let description = full.description();
        assert!(
            description.contains("Background mode"),
            "Full description must describe background mode"
        );
        assert!(
            description.contains("[exit status: N]"),
            "Full description must document the completion annotation"
        );

        let schema = full.parameters_schema();
        let props = schema["properties"].as_object().expect("schema properties");
        assert!(
            props.contains_key("background"),
            "Full schema must advertise the background argument"
        );
        assert!(
            props.contains_key("stop"),
            "Full schema must advertise the stop argument"
        );
        assert_eq!(
            props["background"]["type"], "boolean",
            "background must be a boolean"
        );
        assert_eq!(
            props["background"]["default"], false,
            "background must default to false"
        );
    }

    /// The read-only variant shares the base description and the grep-engine
    /// disclosure with Full, but must not advertise the full-only background
    /// capability (it has no `background` argument).
    #[test]
    fn read_only_description_covers_grep_notes_without_background_capability() {
        let description = ShellTool::new(ShellMode::ReadOnly).description();
        assert!(
            description.contains("## Grep notes"),
            "read-only description must carry the grep-engine disclosure"
        );
        assert!(
            !description.contains("Background mode"),
            "read-only description must not advertise the full-only background capability"
        );
    }

    /// A command that outlives its timeout must be killed: the run returns
    /// `ShellRunResult::TimedOut` with an elapsed time close to the limit
    /// (not the full `sleep` duration).
    ///
    /// This test is `#[ignore]` by default because it waits out a real 1 s
    /// command timeout against a live `sleep 10` child. Run it explicitly
    /// with:
    ///
    /// ```sh
    /// cargo test run_command_with_timeout_kills_long_sleep -- --ignored --nocapture
    /// ```
    #[ignore = "waits out real command timeouts against live processes (hardcoded waits); runs only when explicitly invoked"]
    #[cfg(unix)]
    #[tokio::test]
    async fn run_command_with_timeout_kills_long_sleep() {
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg("sleep 10");
        let result = run_command_with_timeout(
            &mut cmd,
            Duration::from_secs(1),
            Duration::from_secs(10),
            RunOwner::Agent,
        )
        .await;
        match result {
            ShellRunResult::TimedOut { elapsed, .. } => {
                assert!(
                    elapsed < Duration::from_secs(3),
                    "expected ~1s timeout, got {elapsed:?}"
                );
            }
            other => panic!("expected TimedOut, got {other:?}"),
        }
    }

    /// When a command times out, stdout written before the kill must be
    /// preserved in `ShellRunResult::TimedOut`.
    ///
    /// This test is `#[ignore]` by default because it waits out a real 2 s
    /// command timeout against a live `sleep 60` child. Run it explicitly
    /// with:
    ///
    /// ```sh
    /// cargo test run_command_with_timeout_captures_partial_stdout -- --ignored --nocapture
    /// ```
    #[ignore = "waits out real command timeouts against live processes (hardcoded waits); runs only when explicitly invoked"]
    #[cfg(unix)]
    #[tokio::test]
    async fn run_command_with_timeout_captures_partial_stdout() {
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg("echo started; sleep 60");
        let result = run_command_with_timeout(
            &mut cmd,
            Duration::from_secs(2),
            Duration::from_secs(10),
            RunOwner::Agent,
        )
        .await;
        match result {
            ShellRunResult::TimedOut { stdout, .. } => {
                let s = String::from_utf8_lossy(&stdout);
                assert!(
                    s.contains("started"),
                    "stdout should contain partial output: {s}"
                );
            }
            other => panic!("expected TimedOut, got {other:?}"),
        }
    }

    /// Tests that timeout error messages include partial output (stdout and stderr
    /// tails) and ANSI stripping works.
    ///
    /// The `sleep 30` grandchild is now terminated by the PGID kill, not orphaned.
    ///
    /// This test is `#[ignore]` by default because it waits out a real 1 s command timeout against a live `sleep 30` child. Run it
    /// explicitly with:
    ///
    /// ```sh
    /// cargo test shell_timeout_error_includes_diagnostics -- --ignored --nocapture
    /// ```
    #[ignore = "waits out real command timeouts against live processes (hardcoded waits); runs only when explicitly invoked"]
    #[cfg(unix)]
    #[tokio::test]
    async fn shell_timeout_error_includes_diagnostics() {
        let tmp = TempDir::new().expect("tempdir");
        let mut cmd = build_shell_command("echo before-timeout; sleep 30", tmp.path());
        let result = run_command_with_timeout(
            &mut cmd,
            Duration::from_secs(1),
            Duration::from_secs(10),
            RunOwner::Agent,
        )
        .await;
        let ShellRunResult::TimedOut {
            stdout,
            stderr,
            pid,
            elapsed,
        } = result
        else {
            panic!("expected timeout");
        };
        let msg = format_timeout_error(
            "echo test",
            elapsed,
            Duration::from_secs(1),
            pid,
            &stdout,
            &stderr,
        );
        assert!(msg.contains("elapsed:"), "msg: {msg}");
        assert!(msg.contains("timeout_limit:"), "msg: {msg}");
        assert!(msg.contains("timeout_secs"), "msg: {msg}");
        assert!(msg.contains("before-timeout"), "msg: {msg}");

        // Verify ANSI escape sequences are stripped from timeout error messages
        let ansi_stdout = b"\x1B[31mred error\x1B[0m";
        let ansi_stderr = b"\x1B[1mBOLD STUFF\x1B[22m";
        let ansi_msg = format_timeout_error(
            "test",
            elapsed,
            Duration::from_mins(5),
            Some(42),
            ansi_stdout,
            ansi_stderr,
        );
        assert!(
            ansi_msg.contains("red error"),
            "ANSI text content should survive stripping: {ansi_msg}"
        );
        assert!(
            !ansi_msg.contains("\x1B["),
            "ANSI escape codes should be stripped from timeout error: {ansi_msg}"
        );
        assert!(
            ansi_msg.contains("BOLD STUFF"),
            "ANSI text content should survive stripping: {ansi_msg}"
        );
    }

    /// Wait — briefly — for a pid to stop existing. `kill(pid, 0)` is the
    /// existence probe: it sends no signal.
    #[cfg(unix)]
    async fn wait_for_death(pid: i32) {
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            // SAFETY: signal 0 performs the permission/existence check only.
            if unsafe { libc::kill(pid, 0) } != 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("pid {pid} is still alive");
    }

    /// A run that can only end by being stopped: `sh` backgrounds a long sleep —
    /// which inherits sh's process group — then waits, writing the sleep's pid to
    /// `grandchild.pid` in `dir`. Returns that file and the prepared command.
    #[cfg(unix)]
    fn run_with_backgrounded_sleep(dir: &TempDir) -> (std::path::PathBuf, tokio::process::Command) {
        let pid_path = dir.path().join("grandchild.pid");
        let cmd_str = format!(
            "sleep 999 & echo $! > {}; wait",
            pid_path.to_str().expect("valid utf-8 path")
        );
        (pid_path, build_shell_command(&cmd_str, dir.path()))
    }

    /// The pid the shell wrote for that sleep, once the run has been stopped.
    #[cfg(unix)]
    fn stopped_grandchild(pid_path: &std::path::Path) -> i32 {
        std::fs::read_to_string(pid_path)
            .expect("grandchild PID file must exist — grandchild was launched")
            .trim()
            .parse()
            .expect("valid PID from file")
    }

    /// A stop ends the command's whole process tree, not only its direct child:
    /// the timeout path ends the run's containment, so a grandchild the command
    /// backgrounded dies with it. This is the unix arm of [`Tree::terminate`];
    /// Windows reaches the same outcome through its job object, which only a
    /// Windows host can exercise.
    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_kills_the_command_process_tree() {
        let dir = TempDir::new().expect("tempdir");
        let (pid_path, mut cmd) = run_with_backgrounded_sleep(&dir);

        let result = run_command_with_timeout(
            &mut cmd,
            Duration::from_secs(2),
            Duration::from_secs(5),
            RunOwner::Agent,
        )
        .await;
        assert!(
            matches!(result, ShellRunResult::TimedOut { .. }),
            "expected TimedOut, got {result:?}"
        );

        wait_for_death(stopped_grandchild(&pid_path)).await;
    }

    /// A run that is torn down (its future dropped — an abandoned tool call, a
    /// force-cancelled drain) takes its whole process tree with it: the
    /// kill-on-drop guard fires.
    #[cfg(unix)]
    #[tokio::test]
    async fn dropping_the_run_kills_the_command_process_tree() {
        let dir = TempDir::new().expect("tempdir");
        let (pid_path, mut cmd) = run_with_backgrounded_sleep(&dir);

        // The timeout drops the run's future, exactly as an aborted task does.
        let dropped = tokio::time::timeout(
            Duration::from_secs(2),
            run_command_with_timeout(
                &mut cmd,
                Duration::from_secs(30),
                Duration::from_secs(5),
                RunOwner::Agent,
            ),
        )
        .await;
        assert!(
            dropped.is_err(),
            "the run must still be live when it is dropped"
        );

        wait_for_death(stopped_grandchild(&pid_path)).await;
    }

    /// A leftover backgrounded process holding the output pipes open must not
    /// hang the drain after the main command exits — it errors within the
    /// drain bound, and the leftover process group is killed.
    #[cfg(unix)]
    #[tokio::test]
    async fn output_drain_times_out_when_background_process_holds_pipes() {
        let dir = TempDir::new().expect("tempdir");
        let pid_path = dir.path().join("bg.pid");
        let pid_path_str = pid_path.to_str().expect("valid utf-8 path");

        // Main command exits immediately; `sleep 999 &` inherits
        // stdout/stderr (no redirect), so EOF never arrives after sh exits.
        let cmd_str = format!("echo before-drain; sleep 999 & echo $! > {pid_path_str}");
        let mut cmd = build_shell_command(&cmd_str, dir.path());

        let result = run_command_with_timeout(
            &mut cmd,
            Duration::from_secs(30),
            Duration::from_millis(150),
            RunOwner::Agent,
        )
        .await;
        let ShellRunResult::DrainTimedOut {
            stdout,
            stderr,
            elapsed,
            ..
        } = result
        else {
            panic!("expected DrainTimedOut, got {result:?}");
        };
        let out = String::from_utf8_lossy(&stdout);
        assert!(out.contains("before-drain"), "partial stdout: {out}");
        assert!(
            elapsed < Duration::from_secs(5),
            "drain should error within the bound: {elapsed:?}"
        );

        let msg = format_drain_timeout_error(
            ShellMode::ReadOnly,
            "test",
            elapsed,
            Duration::from_millis(150),
            None,
            &stdout,
            &stderr,
        );
        assert!(msg.contains("drain"), "msg: {msg}");
        assert!(msg.contains("before-drain"), "msg: {msg}");

        // The leftover grandchild must be dead (the drain path ends the run's
        // containment). Poll briefly — SIGKILL delivery + reap is immediate, but
        // the kernel may lag under load.
        let pid_content = std::fs::read_to_string(&pid_path)
            .expect("grandchild PID file must exist — grandchild was launched");
        let pid: i32 = pid_content.trim().parse().expect("valid PID from file");
        wait_for_death(pid).await;
    }

    /// Short-lived backgrounded jobs that finish within the drain bound keep
    /// working — the drain completes normally.
    #[cfg(unix)]
    #[tokio::test]
    async fn output_drain_completes_for_short_lived_background_job() {
        let dir = TempDir::new().expect("tempdir");
        let mut cmd = build_shell_command("echo done; sleep 0.2 &", dir.path());
        let result = run_command_with_timeout(
            &mut cmd,
            Duration::from_secs(30),
            Duration::from_secs(5),
            RunOwner::Agent,
        )
        .await;
        let ShellRunResult::Completed { stdout, .. } = result else {
            panic!("expected Completed, got {result:?}");
        };
        assert!(String::from_utf8_lossy(&stdout).contains("done"));
    }

    /// A leftover process holding only ONE pipe (stdout EOFs, stderr hangs)
    /// still drains the completed side and errors without re-polling the
    /// completed reader (tokio panics on JoinHandle re-poll).
    #[cfg(unix)]
    #[tokio::test]
    async fn output_drain_timeout_keeps_completed_side() {
        let dir = TempDir::new().expect("tempdir");
        // The backgrounded sleep redirects stdout (releasing the shell's
        // stdout pipe → EOF) but inherits stderr, keeping it open forever.
        let mut cmd = build_shell_command("echo out; sleep 999 >/dev/null &", dir.path());
        let result = run_command_with_timeout(
            &mut cmd,
            Duration::from_secs(30),
            Duration::from_millis(150),
            RunOwner::Agent,
        )
        .await;
        let ShellRunResult::DrainTimedOut { stdout, .. } = result else {
            panic!("expected DrainTimedOut, got {result:?}");
        };
        assert!(
            String::from_utf8_lossy(&stdout).contains("out"),
            "completed stdout side must be preserved: {stdout:?}"
        );
    }

    // ── Shell compression pipeline tests ────────────────────────────

    #[test]
    fn pipeline_credential_scrubbing_cases() {
        // Pipeline entry scrubs both stdout and stderr; on_empty (Stage 5)
        // also verifies scrubbing in stderr.
        // Each case follows the same pattern: raw credentials not present,
        // redacted form present, pipeline-specific content preserved.
        let cases: &[ShellOutputCase] = &[
            ShellOutputCase {
                // Stage 5 (on-empty): credentials from stderr are scrubbed
                // (via git diff profile's on_empty message)
                name: "git diff on_empty scrubs credentials in stderr",
                command: "git diff",
                stderr: "api_key=abcdefghijklmnop12345678",
                exit_code: 1,
                not_contains: &["api_key=abcdefghijklmnop12345678"],
                contains: &["api_key=abcd*[REDACTED]", "no changes"],
                ..Default::default()
            },
            ShellOutputCase {
                // Stage 5 (on-empty): credentials from stderr are scrubbed.
                // Uses a `warning:` prefix so the line passes through the
                // keep_stderr filter even on the failure path.
                name: "on-empty scrubs credentials in stderr",
                command: "tsc --noEmit",
                stderr: "warning: api_key=abcdefghijklmnop12345678",
                exit_code: 1,
                not_contains: &["api_key=abcdefghijklmnop12345678"],
                contains: &["api_key=abcd*[REDACTED]", "[tsc: ok]"],
                ..Default::default()
            },
            ShellOutputCase {
                // Main pipeline path (all stages applied, no early-return):
                // stdout credentials are scrubbed at pipeline entry before any
                // downstream stage processes them.
                name: "main pipeline scrubs credentials in stdout",
                command: "echo test",
                stdout: "API_KEY=abcdefghijklmnop12345678",
                not_contains: &["abcdefghijklmnop12345678"],
                contains: &["API_KEY=abcd*[REDACTED]"],
                ..Default::default()
            },
        ];
        check_shell_output(cases);
    }

    #[test]
    fn truncate_line_width_short_and_long() {
        // Long lines truncated with continuation marker
        let long = "a".repeat(500);
        let result = truncate_line_width(&long, 100);
        assert!(result.len() < long.len() + 100, "should truncate");
        assert!(
            result.contains("more chars on this line"),
            "should show continuation marker"
        );
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 2, "truncated line + continuation marker");
        assert_eq!(
            lines[0].len(),
            100,
            "first line should be exactly max_chars"
        );
        assert!(
            !lines[0].contains("..."),
            "first line should not contain truncation marker"
        );

        // Short lines preserved unchanged
        let input = "hello\nworld";
        let result = truncate_line_width(input, 500);
        assert_eq!(result, input, "short lines should pass through");
    }

    #[test]
    fn try_spill_to_file_behavior() {
        // Small output passes through unchanged
        let short = "hello".to_string();
        let result = try_spill_to_file(short, TOOL_OUTPUT_BUDGET_BYTES);
        assert_eq!(result, "hello", "short output should pass through");

        // Large single-line output spills
        let large = "x".repeat(TOOL_OUTPUT_BUDGET_BYTES * 2);
        let result = try_spill_to_file(large, TOOL_OUTPUT_BUDGET_BYTES);
        assert!(
            result.contains("[Output saved to"),
            "should contain spill path"
        );
        assert!(
            result.contains("10000 bytes"),
            "should mention byte count: {result}"
        );

        // Multi-line large output shows head+tail preview
        let lines: Vec<String> = (0..800).map(|i| format!("line_{i:04}")).collect();
        let multi = lines.join("\n");
        let multi_len = multi.len();
        assert!(
            multi_len > TOOL_OUTPUT_BUDGET_BYTES,
            "test data {multi_len} must exceed spill threshold"
        );
        let result = try_spill_to_file(multi, TOOL_OUTPUT_BUDGET_BYTES);
        assert!(
            result.contains("[Output saved to"),
            "should contain spill path"
        );
        assert!(
            result.contains("[view with: read "),
            "should contain actionable read hint"
        );
        assert!(result.contains("line_0000"), "should show first line");
        assert!(result.contains("line_0799"), "should show last line");
        assert!(
            result.len() < multi_len,
            "inline preview should be truncated"
        );

        // Ensure the spill dir exists
        assert!(
            std::fs::read_dir(std::env::temp_dir().join(".agent")).is_ok(),
            "spill dir should exist"
        );
    }

    #[test]
    fn resolved_shell_path_covers_tool_dirs() {
        #[cfg(unix)]
        {
            let path = resolved_shell_path();
            assert!(
                path.contains(".npm-global/bin"),
                "PATH should include ~/.npm-global/bin for globally installed npm tools: {path}"
            );
            // Belt-and-suspenders means ~/.cargo/bin is always added regardless
            // of $CARGO_HOME, so it shows up in both CARGO_HOME cases below.
            assert!(
                path.contains(".cargo/bin"),
                "PATH should include ~/.cargo/bin: {path}"
            );
            {
                let _guard = set_env_var("CARGO_HOME", Some("/custom/cargo"));
                let path = resolved_shell_path();
                assert!(
                    path.contains("/custom/cargo/bin"),
                    "PATH should include $CARGO_HOME/bin when CARGO_HOME is set: {path}"
                );
                assert!(
                    path.contains(".cargo/bin"),
                    "PATH should still include ~/.cargo/bin when CARGO_HOME is set (belt-and-suspenders): {path}"
                );
            }
            // $CARGO_HOME/bin and ~/.cargo/bin must deduplicate when they point
            // to the same directory (skipped when no home directory exists).
            if let Some(dirs) = UserDirs::new() {
                let default_cargo_home = dirs
                    .home_dir()
                    .join(".cargo")
                    .to_string_lossy()
                    .into_owned();
                let _guard = set_env_var("CARGO_HOME", Some(&default_cargo_home));
                let path = resolved_shell_path();
                let count = path
                    .split(':')
                    .filter(|part| *part == format!("{default_cargo_home}/bin"))
                    .count();
                assert_eq!(
                    count, 1,
                    "$CARGO_HOME/bin and ~/.cargo/bin should deduplicate when they point to the same directory: {path}"
                );
            }
        }
        #[cfg(target_os = "macos")]
        {
            let path = resolved_shell_path();
            assert!(
                path.contains("/opt/homebrew/bin"),
                "PATH should include Homebrew bin on macOS: {path}"
            );
        }
    }

    // ── New feature tests ───────────────────────────────────────────

    #[test]
    fn collapse_blank_lines_cases() {
        let cases: &[(&str, &str)] = &[
            // 3+ consecutive blank lines collapse to 2
            ("a\n\n\n\nb\n\n\nc", "a\n\n\nb\n\n\nc"),
            // runs ≤2 left alone, longer runs collapse to 2
            ("a\n\nb\n\n\nc\n\n\n\nd", "a\n\nb\n\n\nc\n\n\nd"),
            // no blank lines → pass through unchanged
            ("a\nb\nc", "a\nb\nc"),
            // all-blank input collapses completely (no anchor lines)
            ("\n\n\n\n\n", ""),
        ];
        for (input, expected) in cases {
            let result = collapse_blank_lines(input);
            assert_eq!(result, *expected, "input: {input:?}");
        }
    }

    // ── Chained command and canonical command tests ─────────────────

    #[test]
    fn extract_segments_cases() {
        let cases: &[(&str, &[&str])] = &[
            // simple single command
            ("cargo build", &["cargo build"]),
            // chained with &&
            ("cd project && cargo build", &["cd project", "cargo build"]),
            // pipe (|) splits commands
            (
                "npm run build 2>&1 | tee build.log",
                &["npm run build 2>&1", "tee build.log"],
            ),
            // single quotes protect && and | from being treated as separators
            ("echo 'foo && bar' | cat", &["echo 'foo && bar'", "cat"]),
            // semicolon splits commands
            ("cargo build ; cargo test", &["cargo build", "cargo test"]),
            // single-quoted && preserved as one segment
            ("echo 'foo && bar'", &["echo 'foo && bar'"]),
            // double-quoted pipe preserved as one segment
            ("echo \"pipe | test\"", &["echo \"pipe | test\""]),
            // ── Newline as command separator ──────────────────────
            (
                "touch /tmp/a\necho hi > /tmp/b",
                &["touch /tmp/a", "echo hi > /tmp/b"],
            ),
            // Backslash-newline continuation joins the logical line.
            ("echo hello \\\nworld", &["echo hello world"]),
            // Heredoc bodies are removed before segmentation (never commands).
            ("cat <<EOF\nbody\nEOF", &["cat"]),
            (
                "cat <<EOF > /tmp/out\nbody\nEOF",
                &["cat   > /tmp/out"], // `<<EOF` marker replaced by a space
            ),
            // Herestrings have no body — the whole line stays one segment.
            ("cat <<< hi > /tmp/out", &["cat <<< hi > /tmp/out"]),
            // `>|` compound redirect is not split at the pipe.
            ("echo hi >| /tmp/force", &["echo hi >| /tmp/force"]),
            // ── Command substitutions stay whole ────────────────────
            // Separators inside `$(...)` are part of the substitution, not
            // command separators (mis-splitting flips is_chained and skips
            // standalone-only output transforms).
            ("echo $(echo hi; touch x)", &["echo $(echo hi; touch x)"]),
            ("echo $(echo hi) ; touch x", &["echo $(echo hi)", "touch x"]),
            ("echo `echo hi; touch x`", &["echo `echo hi; touch x`"]),
            (
                "cd /tmp && echo $(echo a && echo b)",
                &["cd /tmp", "echo $(echo a && echo b)"],
            ),
            // Nested substitution stays inside the outer one.
            ("echo $(echo $(ls; pwd))", &["echo $(echo $(ls; pwd))"]),
            // Nested `$(` adds exactly one paren; a real `;` after it splits.
            (
                "echo $(echo $(echo hi)) ; touch x",
                &["echo $(echo $(echo hi))", "touch x"],
            ),
            // Plain parens inside a substitution nest too: the `;` is body.
            (
                "echo $( (echo hi) ; echo more) tail",
                &["echo $( (echo hi) ; echo more) tail"],
            ),
            // Arithmetic with parens: `&&` inside `$((...))` is body content.
            (
                "echo $(( (a) && (b) )) tail",
                &["echo $(( (a) && (b) )) tail"],
            ),
            // `$((...))` nested inside `$(...)`.
            (
                "echo $(echo $((a+1)); echo x)",
                &["echo $(echo $((a+1)); echo x)"],
            ),
            // Substitution with newline inside stays whole.
            ("echo $(echo hi\necho bye)", &["echo $(echo hi\necho bye)"]),
            // Heredoc body with a substitution: the body substitution is
            // emitted as its own segment (it must remain scanned — bash
            // executes `$()` inside unquoted heredoc bodies).
            ("cat <<EOF\n$(touch ws)\nEOF", &["cat", "$(touch ws)"]),
            // Double-quoted substitutions stay whole too: a `;` inside
            // `"$(...)"` is part of the substitution, not a separator, and
            // bash executes the body, so segmentation must not fragment it.
            (
                "echo \"$(echo hi; touch x)\"",
                &["echo \"$(echo hi; touch x)\""],
            ),
            (
                "echo \"$(echo hi)\" ; touch x",
                &["echo \"$(echo hi)\"", "touch x"],
            ),
            (
                "echo \"`echo hi; touch x`\"",
                &["echo \"`echo hi; touch x`\""],
            ),
            // Escape-aware: `\)` is body content, not a substitution closer.
            (
                "echo $(echo \\)) ; touch x",
                &["echo $(echo \\))", "touch x"],
            ),
            // Escaped backticks inside a backtick substitution don't close it.
            (
                "echo `echo \\`hi\\`` ; touch x",
                &["echo `echo \\`hi\\``", "touch x"],
            ),
            // `\'` inside `$(...)`: the closing quote is not swallowed.
            (
                "echo $(echo 'a\\') ; touch x",
                &["echo $(echo 'a\\')", "touch x"],
            ),
        ];
        for (input, expected) in cases {
            let result = extract_command_segments(input);
            assert_eq!(
                result.iter().map(String::as_str).collect::<Vec<_>>(),
                *expected,
                "input: {input:?}"
            );
        }
    }

    #[test]
    fn canonical_command_cases() {
        // Each case: (input, expected_output) — organized by theme.
        // Adding a new case is one line; include an inline comment explaining why.
        let cases: &[(&str, &str)] = &[
            // ── Path stripping ──────────────────────────────────────
            ("/usr/local/bin/cargo build", "cargo build"),
            // ── Git global flags ────────────────────────────────────
            ("git -C /repo diff", "git diff"),
            ("git -c user.name=me log", "git log"),
            ("git -- diff", "git diff"), // -- is treated as a flag and skipped
            // ── Shell prefix: sudo ──────────────────────────────────
            ("sudo cargo build", "cargo build"),
            ("sudo -E cargo build", "cargo build"), // -E flag between prefix and command
            ("sudo --preserve-env cargo build", "cargo build"), // --preserve-env flag
            ("sudo -E git -C /repo diff", "git diff"), // compound: multiple flags + subcommand
            // ── Shell prefix: time ─────────────────────────────────
            ("time -v cargo test", "cargo test"), // -v flag skipped
            // ── cd (shell builtin) ─────────────────────────────────
            ("cd", ""),      // all shell prefixes, no command → empty
            ("cd ..", ".."), // path segment only — won't match anything but shouldn't crash
            // ── Package managers ───────────────────────────────────
            ("pnpm install", "pnpm install"),
            ("yarn add foo", "yarn add"),
            // ── Cargo flags ────────────────────────────────────────
            ("cargo test --lib", "cargo test"),
            ("cargo --release build", "cargo build"),
            // Only flags without values are skipped for non-git commands.
            // --release and --verbose don't take values, so both are skipped.
            ("cargo --release --verbose build", "cargo build"),
            // ── Toolchain specifiers and stderr-capture suffixes ───
            // `+nightly`-style toolchain specifiers and `2>&1`/`1>&2` suffixes
            // must never become the parsed subcommand.
            ("cargo +nightly build", "cargo build"),
            ("cargo +stable check", "cargo check"),
            ("cargo build 2>&1", "cargo build"),
            ("cargo --version 2>&1", "cargo"),
            ("git --version 2>&1", "git"),
            ("git status 2>&1", "git status"),
            // ── Environment variable assignments ───────────────────
            ("CC=gcc make", "make"), // env assignment before command
            ("VAR=val cargo check", "cargo check"),
            ("CC=gcc CXX=g++ make -j4", "make"), // multiple env assignments
            ("CC=gcc", ""),                      // only env assignments, no command
            ("sudo CC=gcc make", "make"),        // prefix + env assignment + command
            // ── pytest dead-branch documentation ──────────────────────
            // These cases document why `python -m pytest` and
            // `poetry run pytest` cannot match the pytest profile:
            // canonical_command strips `-m` (flag), producing "python pytest"
            // that doesn't start with "pytest"; and treats `run` as the
            // subcommand of `poetry`, producing "poetry run".
            ("python -m pytest tests/", "python pytest"),
            ("poetry run pytest tests/", "poetry run"),
            // ── npx (shell prefix) ─────────────────────────────────────
            // npx is a shell prefix, so it's stripped before command extraction.
            // Tool-specific profiles (eslint, prettier, vitest, etc.) now match
            // correctly instead of being shadowed by the generic npx catch-all.
            ("npx eslint", "eslint"),
            ("npx eslint .", "eslint ."),
            ("npx eslint --fix .", "eslint ."),
            ("npx --yes eslint .", "eslint ."),
            ("npx prettier --check file.js", "prettier file.js"),
            ("npx vitest --run", "vitest"),
            ("npx tsc --noEmit", "tsc"),
            (
                "npx --yes create-react-app my-app",
                "create-react-app my-app",
            ),
        ];
        for &(input, expected) in cases {
            assert_eq!(
                canonical_command(input),
                expected,
                "canonical_command({input:?})",
            );
        }
    }

    /// A quoted path spelling profiles like its unquoted twin. Profiles are
    /// selected from the original command string, so this is what gives an
    /// agent-written `'/usr/local/bin/cargo' build` the cargo output treatment.
    #[cfg(unix)]
    #[test]
    fn quoted_path_spelling_selects_the_same_profile_as_its_unquoted_twin() {
        let selected = |command: &str| {
            let segments = extract_command_segments(command);
            select_profile(&segments, false)
                .match_command
                .as_str()
                .to_owned()
        };
        let quoted = selected("'/usr/local/bin/cargo' build");
        assert_eq!(quoted, selected("/usr/local/bin/cargo build"));
        // Non-vacuity: an unrelated command must NOT select that same profile.
        assert_ne!(quoted, selected("'/usr/local/bin/true' build"));
    }

    #[test]
    fn first_command_word_consistent_with_canonical() {
        // Property: first_command_word returns the first word of canonical_command's
        // result, or empty when canonical_command is empty.
        let inputs: &[&str] = &[
            // Path stripping
            "/usr/local/bin/cargo build",
            // Git global flags
            "git -C /repo diff",
            "git -c user.name=me log",
            "git -- diff",
            // Shell prefix: sudo
            "sudo cargo build",
            "sudo -E cargo build",
            "sudo --preserve-env cargo build",
            "sudo -E git -C /repo diff",
            // Shell prefix: time
            "time -v cargo test",
            // cd (shell builtin)
            "cd",
            "cd ..",
            // Package managers
            "pnpm install",
            "yarn add foo",
            // Cargo flags
            "cargo test --lib",
            "cargo --release build",
            "cargo --release --verbose build",
            // Environment variable assignments
            "CC=gcc make",
            "VAR=val cargo check",
            "CC=gcc CXX=g++ make -j4",
            "CC=gcc",
            "sudo CC=gcc make",
            // Edge cases
            "",
            "   ",
            "ls",
            "cat file.txt",
            "/bin/echo hello",
        ];
        for &input in inputs {
            let canonical = canonical_command(input);
            let first = first_command_word(input);
            if canonical.is_empty() {
                assert!(
                    first.is_empty(),
                    "first_command_word({input:?}) should be empty when canonical_command is empty",
                );
            } else {
                let expected_first = canonical.split_whitespace().next().unwrap_or("");
                assert_eq!(
                    first, expected_first,
                    "first_command_word({input:?}) should match first word of canonical_command({input:?}) = {canonical:?}",
                );
            }
        }
    }

    #[test]
    fn test_all_profiles_have_valid_configs() {
        let profiles = PROFILES.iter().collect::<Vec<_>>();
        assert!(
            !profiles.is_empty(),
            "should have at least the generic fallback"
        );
        for p in &profiles {
            assert!(
                !p.match_command.as_str().is_empty(),
                "match_command should not be empty"
            );
            if let (Some(head), Some(tail), Some(max)) = (p.head_lines, p.tail_lines, p.max_lines) {
                assert!(
                    head + tail < max,
                    "head+tail ({head}+{tail}) should be strictly less than max_lines ({max}) — omission marker would overflow"
                );
            }
        }
    }

    #[test]
    fn profile_df_caps_at_20_lines() {
        let input = (0..50)
            .map(|i| format!("filesystem{i}  used  avail capacity mounted_on"))
            .collect::<Vec<_>>()
            .join("\n");
        let result = process_shell_output("df -h", &input, "", 0, Duration::ZERO);
        let lines = result.lines().count();
        // 20 lines + 1 truncated note line = 21 max
        assert!(lines <= 21, "df should cap at ~21 lines, got {lines}");
        assert!(lines >= 19, "df should have around 20 lines, got {lines}");
    }

    // ── check_outside_quotes ──────────────────────────────────────────
    // Pure quote-tracking state machine — escape handling is caller's concern.

    type QuoteStep = (char, bool, bool, bool);

    #[test]
    fn check_outside_quotes_cases() {
        // Each case: (name, &[(char, expected_return, in_single, in_double)])
        let cases: &[(&str, &[QuoteStep])] = &[
            ("normal char outside", &[('a', true, false, false)]),
            (
                "single quote blocks",
                &[
                    ('\'', false, true, false),
                    ('>', false, true, false),
                    ('\'', false, false, false),
                    ('>', true, false, false),
                ],
            ),
            (
                "double quote blocks",
                &[
                    ('"', false, false, true),
                    ('>', false, false, true),
                    ('"', false, false, false),
                    ('>', true, false, false),
                ],
            ),
            (
                "single inside double",
                &[('"', false, false, true), ('\'', false, false, true)],
            ),
            (
                "double inside single",
                &[('\'', false, true, false), ('"', false, true, false)],
            ),
        ];

        for (name, steps) in cases {
            let (mut s, mut d) = (false, false);
            for (i, &(ch, exp_out, exp_s, exp_d)) in steps.iter().enumerate() {
                let result = check_outside_quotes(ch, &mut s, &mut d);
                assert_eq!(
                    result, exp_out,
                    "{name} step {i}: check_outside_quotes({ch:?}) returned {result}, expected {exp_out}",
                );
                assert_eq!(
                    s, exp_s,
                    "{name} step {i}: after {ch:?}, in_single={s}, expected {exp_s}",
                );
                assert_eq!(
                    d, exp_d,
                    "{name} step {i}: after {ch:?}, in_double={d}, expected {exp_d}",
                );
            }
        }
    }

    // ── track_char_context ───────────────────────────────────────────
    // Combined escape + quote tracking state machine.

    type ContextStep = (char, bool, bool, bool, bool);

    #[test]
    fn track_char_context_cases() {
        // Each case: (char, expected_return, in_single, in_double, escaped)
        // The `escaped` column shows the flag AFTER processing the character.
        let cases: &[(&str, &[ContextStep])] = &[
            (
                "backslash escapes outside quotes",
                &[
                    ('\\', false, false, false, true), // backslash sets escaped
                    ('a', false, false, false, false), // escaped 'a' consumed, skip
                    ('a', true, false, false, false),  // normal 'a' outside quotes
                ],
            ),
            (
                "escaped backslash",
                &[
                    ('\\', false, false, false, true),  // first backslash sets escaped
                    ('\\', false, false, false, false), // second backslash: escaped flag was set, consume it; does NOT start new escape
                    ('a', true, false, false, false),   // normal 'a' outside quotes
                ],
            ),
            (
                "escaped quote inside double does not toggle",
                &[
                    ('"', false, false, true, false),  // double opens
                    ('\\', false, false, true, true),  // backslash inside double, sets escaped
                    ('"', false, false, true, false), // escaped quote consumed, doesn't toggle double
                    ('"', false, false, false, false), // unescaped quote closes double
                ],
            ),
            (
                "backslash inside single is literal",
                &[
                    ('\'', false, true, false, false),  // single opens
                    ('\\', false, true, false, false), // backslash inside single: not escape, still inside
                    ('a', false, true, false, false),  // inside single quotes
                    ('\'', false, false, false, false), // single closes
                    ('>', true, false, false, false),  // outside quotes again
                ],
            ),
        ];

        for (name, steps) in cases {
            let (mut s, mut d, mut e) = (false, false, false);
            for (i, &(ch, exp_out, exp_s, exp_d, exp_e)) in steps.iter().enumerate() {
                let result = track_char_context(ch, &mut s, &mut d, &mut e);
                assert_eq!(
                    result, exp_out,
                    "{name} step {i}: track_char_context({ch:?}) returned {result}, expected {exp_out}",
                );
                assert_eq!(
                    s, exp_s,
                    "{name} step {i}: after {ch:?}, in_single={s}, expected {exp_s}",
                );
                assert_eq!(
                    d, exp_d,
                    "{name} step {i}: after {ch:?}, in_double={d}, expected {exp_d}",
                );
                assert_eq!(
                    e, exp_e,
                    "{name} step {i}: after {ch:?}, escaped={e}, expected {exp_e}",
                );
            }
        }
    }

    // ── apply_line_truncation unit tests ──────────────────────────────

    struct TruncateCase {
        name: &'static str,
        head: usize,
        tail: usize,
        max: Option<usize>,
        output: &'static str,
        pre_is_some: bool,
        check_contains: &'static [&'static str],
    }

    fn check_truncate(cases: &[TruncateCase]) {
        for case in cases {
            let mut p = Profile::new("test");
            if case.head > 0 || case.tail > 0 {
                p = p.head(case.head).tail(case.tail);
            }
            if let Some(m) = case.max {
                p = p.max(m);
            }
            let (result, pre) = apply_line_truncation(case.output, &p);
            assert_eq!(
                pre.is_some(),
                case.pre_is_some,
                "[{}] pre.is_some mismatch. pre: {pre:?}",
                case.name
            );
            assert_contains_not_contains(case.name, &result, case.check_contains, &[]);
        }
    }

    #[test]
    fn truncate_simple_cases() {
        check_truncate(&[
            TruncateCase {
                name: "no config passthrough",
                head: 0,
                tail: 0,
                max: None,
                output: "line1\nline2\nline3",
                pre_is_some: false,
                check_contains: &["line1\nline2\nline3"],
            },
            TruncateCase {
                name: "head+tail small output no sandwich",
                head: 2,
                tail: 2,
                max: None,
                output: "line1\nline2\nline3\nline4\nline5",
                pre_is_some: false,
                check_contains: &["line1\nline2\nline3\nline4\nline5"],
            },
            TruncateCase {
                name: "max only caps at limit",
                head: 0,
                tail: 0,
                max: Some(3),
                output: "a\nb\nc\nd\ne",
                pre_is_some: false,
                check_contains: &["... (2 lines truncated)"],
            },
            TruncateCase {
                name: "max only fits no truncation",
                head: 0,
                tail: 0,
                max: Some(10),
                output: "a\nb\nc",
                pre_is_some: false,
                check_contains: &["a\nb\nc"],
            },
            TruncateCase {
                name: "head+tail fits when under limit",
                head: 5,
                tail: 3,
                max: None,
                output: "a\nb\nc\nd",
                pre_is_some: false,
                check_contains: &["a\nb\nc\nd"],
            },
            // head=2, tail=2, max=3 on 5-line output: small output bypasses sandwich,
            // so max cap is the only active truncation → 3 lines + marker = 4 total.
            TruncateCase {
                name: "head+tail+max small output no sandwich",
                head: 2,
                tail: 2,
                max: Some(3),
                output: "a\nb\nc\nd\ne",
                pre_is_some: false,
                check_contains: &["... (2 lines truncated)"],
            },
        ]);
    }

    #[test]
    fn truncate_head_tail_triggers_sandwich_large_output() {
        let p = Profile::new("test").head(2).tail(2);
        // Generate output large enough to exceed TOOL_OUTPUT_BUDGET_BYTES
        let lines: Vec<String> = (0..100)
            .map(|i| {
                format!(
                    "line {i} aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                )
            })
            .collect();
        let output = lines.join("\n");
        assert!(
            output.len() > TOOL_OUTPUT_BUDGET_BYTES,
            "test output must exceed threshold (got {} bytes)",
            output.len()
        );

        let (result, pre) = apply_line_truncation(&output, &p);
        assert!(pre.is_some(), "should capture pre-truncation output");
        assert!(
            result.contains("... (96 lines omitted)"),
            "should have omission marker"
        );
        assert!(
            result.starts_with("line 0 aaaaaaaa"),
            "should start with head"
        );
        assert!(
            result.ends_with("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            "should end with tail"
        );
        // Also verify that adding max=N doesn't change behavior when
        // head+tail+1 <= N (sandwich format is already smaller than cap).
        let p2 = Profile::new("test").head(2).tail(2).max(100);
        let (result2, pre2) = apply_line_truncation(&output, &p2);
        assert!(pre2.is_some(), "should capture pre-truncation output");
        assert!(
            result2.contains("... (96 lines omitted)"),
            "should have sandwich omission marker"
        );
        assert!(
            !result2.contains("lines truncated"),
            "sandwich should not be additionally truncated when head+tail+1 <= max"
        );
    }

    // ── format_sandwich tests ──────────────────────────────────────────

    #[test]
    fn format_sandwich_cases() {
        let cases: &[(&str, usize, usize, &str)] = &[
            // (input, head, tail, expected)
            ("a\nb\nc", 2, 2, "a\nb\nc"),
            (
                "a\nb\nc\nd\ne\nf\ng",
                2,
                2,
                "a\nb\n... (3 lines omitted)\nf\ng",
            ),
            ("a\nb\nc\nd\ne\nf\ng", 7, 0, "a\nb\nc\nd\ne\nf\ng"),
            ("a\nb\nc\nd\ne\nf\ng", 0, 7, "a\nb\nc\nd\ne\nf\ng"),
            // Head-only truncation (e.g., git log profile: head=20, no tail)
            (
                "a\nb\nc\nd\ne\nf\ng",
                3,
                0,
                "a\nb\nc\n... (4 lines omitted)",
            ),
            // Tail-only truncation (e.g., ping/gh/helm profiles: tail=4..10, no head)
            (
                "a\nb\nc\nd\ne\nf\ng",
                0,
                3,
                "... (4 lines omitted)\ne\nf\ng",
            ),
        ];
        for (input, head, tail, expected) in cases {
            let result = format_sandwich(input, *head, *tail, "omitted");
            assert_eq!(
                result, *expected,
                "format_sandwich({input:?}, {head}, {tail})"
            );
        }
    }

    // ── finish_shell_output unit tests ────────────────────────────────
    // Credential scrubbing is now performed upstream at pipeline entry in
    // `apply_profile_pipeline`, so `finish_shell_output` receives pre-scrubbed
    // input.  These tests verify that non-scrubbing behavior (timing, spill,
    // idempotence) remains correct.

    struct FinishCase {
        name: &'static str,
        combined: &'static str,
        elapsed: Duration,
        pre: Option<&'static str>,
        check: &'static [&'static str], // all must be contained in result
        not_check: &'static [&'static str], // none must be contained
        eq: Option<&'static str>,
    }

    fn check_finish(cases: &[FinishCase]) {
        for case in cases {
            // For pre-truncation spill path: repeat the string enough times to
            // exceed TOOL_OUTPUT_BUDGET_BYTES, triggering the spill-to-file branch
            // in finish_shell_output. The spill content is pre-scrubbed by
            // apply_profile_pipeline before being passed to finish_shell_output.
            // This test verifies that the spill hint is properly appended.
            let pre_owned = case.pre.map(|s| s.repeat(TOOL_OUTPUT_BUDGET_BYTES + 1));
            let result = finish_shell_output(
                case.combined.to_string(),
                case.elapsed,
                pre_owned.as_deref(),
            );
            assert_contains_not_contains(case.name, &result, case.check, case.not_check);
            if let Some(expected) = case.eq {
                assert_eq!(result.trim(), expected, "[{}] expected eq", case.name);
            }
        }
    }

    #[test]
    fn finish_shell_output_cases() {
        check_finish(&[
            FinishCase {
                name: "pre-scrubbed input passes through",
                combined: "API_KEY=abcd*[REDACTED]",
                elapsed: Duration::ZERO,
                pre: None,
                check: &["abcd*[REDACTED]"],
                not_check: &["abcdefghijklmnop"],
                eq: None,
            },
            FinishCase {
                name: "pre-scrubbed combined in spill path",
                combined: "SECRET=wxyz*[REDACTED]",
                elapsed: Duration::ZERO,
                pre: Some("x"),
                check: &["wxyz*[REDACTED]", "[Output saved to"],
                not_check: &["wxyz1234abcdefgh"],
                eq: None,
            },
            FinishCase {
                name: "preserves clean output",
                combined: "no credentials here",
                elapsed: Duration::ZERO,
                pre: None,
                check: &[],
                not_check: &[],
                eq: Some("no credentials here"),
            },
            FinishCase {
                name: "appends elapsed timing with pre-scrubbed input",
                combined: "API_KEY=abcd*[REDACTED]",
                elapsed: Duration::from_secs(5),
                pre: None,
                check: &["[took 5.0s]", "abcd*[REDACTED]"],
                not_check: &["abcdefghijklmnop"],
                eq: None,
            },
            FinishCase {
                name: "scrub idempotent",
                combined: "API_KEY=abcd*[REDACTED]",
                elapsed: Duration::ZERO,
                pre: None,
                check: &[],
                not_check: &[],
                eq: Some("API_KEY=abcd*[REDACTED]"),
            },
        ]);
    }

    /// Both platform fragments of the read-only banner and the grep notes must
    /// be embedded, and must carry no placeholder of their own: `substitute`
    /// does not rescan a replacement value. Each shared skeleton must keep
    /// exactly the keys its renderer supplies — a stray one would render
    /// literally into every agent's description. The renderers pick the
    /// fragment through `SHELL_PLATFORM`, so a typo in a key would otherwise
    /// panic on that platform's host only.
    #[test]
    fn platform_prompt_assets_are_embedded() {
        for (skeleton, keys, fragments) in [
            (
                "tool/shell_readonly_banner.md",
                ["{{temp_root}}", "{{platform_checks}}"].as_slice(),
                [
                    "tool/shell_readonly_banner_unix.md",
                    "tool/shell_readonly_banner_windows.md",
                ]
                .as_slice(),
            ),
            (
                "tool/shell_grep_notes.md",
                ["{{platform_notes}}"].as_slice(),
                [
                    "tool/shell_grep_notes_unix.md",
                    "tool/shell_grep_notes_windows.md",
                ]
                .as_slice(),
            ),
            (
                "tool/shell_full.md",
                ["{{stop_semantics}}"].as_slice(),
                [
                    "tool/shell_full_stop_unix.md",
                    "tool/shell_full_stop_windows.md",
                ]
                .as_slice(),
            ),
        ] {
            let shared = crate::prompt::load_prompt(skeleton);
            let mut rest = shared.clone();
            for key in keys {
                assert!(shared.contains(key), "{skeleton} lost {key}");
                rest = rest.replace(key, "");
            }
            assert!(!rest.contains("{{"), "{skeleton} carries an unrendered key");
            for asset in fragments {
                let text = crate::prompt::load_prompt(asset);
                assert!(!text.contains("{{"), "{asset} carries a placeholder");
            }
        }
    }

    /// The stop text is one text: the tool description's stop bullet and the `stop`
    /// argument's own schema entry must both carry this platform's sentence, and
    /// only this platform's — a session must never be promised a mechanism its
    /// platform does not have (see `tree`).
    #[test]
    fn stop_text_is_this_platforms_own() {
        let (this, other) = match SHELL_PLATFORM {
            ShellPlatform::Unix => (
                "tool/shell_full_stop_unix.md",
                "tool/shell_full_stop_windows.md",
            ),
            ShellPlatform::Windows => (
                "tool/shell_full_stop_windows.md",
                "tool/shell_full_stop_unix.md",
            ),
        };
        let this = crate::prompt::load_prompt(this).trim().to_owned();
        let other = crate::prompt::load_prompt(other).trim().to_owned();

        let full = ShellTool::new(ShellMode::Full);
        let description = full.description();
        assert!(
            description.contains(&this),
            "the description must carry this platform's stop text"
        );
        assert!(
            !description.contains(&other),
            "the description must not promise the other platform's stop text"
        );
        let schema = full.parameters_schema();
        let stop = schema["properties"]["stop"]["description"]
            .as_str()
            .expect("the stop argument carries a description");
        assert!(
            stop.contains(&this),
            "the stop schema must carry this platform's stop text"
        );
        assert!(
            !stop.contains(&other),
            "the stop schema must not promise the other platform's stop text"
        );
    }

    /// The Windows-only refusal decision, driven directly: the serve decision's
    /// own refusal, a produced rewrite the read-only guard rejected, and `None`
    /// everywhere else — on unix in particular, where the original command runs.
    #[test]
    fn unserved_refusal_is_windows_only_and_covers_both_cases() {
        let produced = grep_engine::GrepServe {
            rewritten: Some("engine".into()),
            outcomes: Vec::new(),
            spec_files: Vec::new(),
            refusal: None,
        };
        // An applied rewrite is never a refusal; on unix neither case refuses.
        assert!(unserved_refusal(ShellPlatform::Unix, &produced, true).is_none());
        assert!(unserved_refusal(ShellPlatform::Unix, &produced, false).is_none());
        assert!(unserved_refusal(ShellPlatform::Windows, &produced, true).is_none());

        // Produced but not applied (the read-only guard rejected it): refused
        // on Windows, with the guard rejection named as the cause.
        let guard = unserved_refusal(ShellPlatform::Windows, &produced, false)
            .expect("a rejected rewrite is refused on Windows");
        assert!(guard.contains("read-only guard"), "{guard}");

        // A serve decision that carries its own refusal reports that cause.
        let refused = grep_engine::GrepServe {
            rewritten: None,
            outcomes: Vec::new(),
            spec_files: Vec::new(),
            refusal: Some("nested grep".into()),
        };
        assert_eq!(
            unserved_refusal(ShellPlatform::Windows, &refused, false).as_deref(),
            Some("nested grep")
        );
    }

    /// An engine failure must be recognised from a completed run, and on
    /// Windows from the refusal marker alone: a pipeline tail (`grep … | tail
    /// -1`) carries the tail's exit status, so the sentinel code is masked
    /// there and the run would otherwise look like an empty match set. On unix
    /// the same run re-runs the original command, so the marker is ignored.
    #[test]
    fn engine_failure_covers_the_masked_windows_sentinel() {
        let marker = grep_engine::ENGINE_REFUSAL_MARKER;
        let reason = "grep: engine: working directory diverged from the analyzed command";
        // The engine's own emission order: the marker first, then the detail.
        let masked = format!("{marker}\n{reason}\n");

        let Some(EngineFailure::Refused(cause)) =
            engine_failure(Some(0), masked.as_bytes(), ShellPlatform::Windows)
        else {
            panic!("the marker refuses the Windows call");
        };
        assert!(cause.contains("diverged"), "{cause}");
        assert!(
            !cause.contains(marker),
            "the marker is not agent-facing: {cause}"
        );
        assert!(
            engine_failure(Some(0), masked.as_bytes(), ShellPlatform::Unix).is_none(),
            "unix re-runs the original command instead of refusing"
        );

        // Only a line that IS the marker counts: a served run whose matched
        // line merely echoes the token is a result, not a refusal.
        let echoed = format!("f.txt:{marker}\n");
        assert!(
            engine_failure(Some(0), echoed.as_bytes(), ShellPlatform::Windows).is_none(),
            "a matched line carrying the token is not a refusal: {echoed:?}"
        );

        // The cause is the line after the marker, never a served member's own
        // stderr line that happens to come first.
        let noisy = format!("f.txt: a line mentioning {reason}\n{marker}\nengine detail\n");
        assert!(
            matches!(
                engine_failure(Some(0), noisy.as_bytes(), ShellPlatform::Windows),
                Some(EngineFailure::Refused(cause)) if cause == "engine detail"
            ),
            "the marker's own detail line is the cause"
        );

        // A sentinel exit needs no marker, and a marker with no detail line
        // still yields a refusal with a generic cause.
        assert!(matches!(
            engine_failure(
                Some(grep_engine::ENGINE_FAILED_EXIT),
                reason.as_bytes(),
                ShellPlatform::Unix
            ),
            Some(EngineFailure::ReRun)
        ));
        let bare = engine_failure(
            Some(0),
            format!("{marker}\n").as_bytes(),
            ShellPlatform::Windows,
        );
        let Some(EngineFailure::Refused(bare)) = bare else {
            panic!("a marker alone is a refusal");
        };
        assert!(bare.contains("could not serve"), "{bare}");

        // A stale self-update binary is recognised by its own message on both
        // platforms (its exit 1 is a legitimate grep no-match code) — with its
        // message as the cause where the platform renders one.
        assert!(matches!(
            engine_failure(
                Some(1),
                grep_engine::STALE_BINARY_LOCK_MSG.as_bytes(),
                ShellPlatform::Unix
            ),
            Some(EngineFailure::ReRun)
        ));
        assert!(matches!(
            engine_failure(
                Some(1),
                grep_engine::STALE_BINARY_LOCK_MSG.as_bytes(),
                ShellPlatform::Windows
            ),
            Some(EngineFailure::Refused(cause)) if cause.contains(grep_engine::STALE_BINARY_LOCK_MSG)
        ));
        // A served run is not a refusal: exit 0/1 with no engine complaint.
        for code in [0, 1] {
            assert!(engine_failure(Some(code), b"", ShellPlatform::Windows).is_none());
        }
    }
}
