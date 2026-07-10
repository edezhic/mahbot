use crate::{Tool, ToolOutputPhase, Workspace, util::UnwrapPoison};
use async_trait::async_trait;
use directories::UserDirs;
use regex::{Regex, RegexSet};
use serde_json::json;
use std::collections::HashSet;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::util::scrub_credentials;

mod profiles;
mod readonly;

use self::profiles::{CARGO_COMPILE_PREFIXES, GEN_FALLBACK, PROFILES, Profile, ShortCircuit};
pub use self::readonly::ShellMode;
use self::readonly::check_command;

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
const DEFAULT_SHELL_TIMEOUT_SECS: u64 = 300;
/// Cap bytes collected from each pipe during command execution (including timeouts).
const SHELL_PIPE_READ_CAP: usize = 256 * 1024;
/// Max chars of partial output included in timeout error messages.
const TIMEOUT_OUTPUT_TAIL_CHARS: usize = 2_000;
/// Maximum output size in bytes (1MB).
const MAX_OUTPUT_BYTES: usize = 1_048_576;

/// Environment variables safe to pass to shell commands.
/// Only functional variables are included — never API keys or secrets.
#[cfg(not(target_os = "windows"))]
const SAFE_ENV_VARS: &[&str] = &[
    "PATH", "HOME", "TERM", "LANG", "LC_ALL", "LC_CTYPE", "USER", "SHELL", "TMPDIR",
];

/// Environment variables safe to pass to shell commands on Windows.
/// Includes Windows-specific variables needed for cmd.exe and program resolution.
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
    "TEMP",
    "TMP",
    "TERM",
    "LANG",
    "USERNAME",
];

pub(crate) fn apply_safe_env(cmd: &mut tokio::process::Command) {
    cmd.env_clear();
    for &name in SAFE_ENV_VARS {
        if let Some(value) = baseline_env_value(name) {
            cmd.env(name, value);
        }
    }
}

/// Build a [`tokio::process::Command`] for executing a shell command in the
/// workspace root. The environment is cleared and re-populated from
/// [`SAFE_ENV_VARS`] only — no parent-process environment is inherited. This
/// prevents leaking API keys and other secrets into subprocesses (CWE-200).
fn build_shell_command(command: &str, workspace_root: &Path) -> tokio::process::Command {
    // Platform-specific shell selection and arguments.
    #[cfg(not(target_os = "windows"))]
    let mut process = {
        let mut p = tokio::process::Command::new("sh");
        p.arg("-c").arg(command);
        p
    };

    #[cfg(target_os = "windows")]
    let mut process = {
        const CREATE_NO_WINDOW: u32 = 0x08000000;

        let mut p = tokio::process::Command::new("cmd.exe");
        p.arg("/C").arg(command).creation_flags(CREATE_NO_WINDOW);
        p
    };

    // Shared setup: set working directory and sanitize the environment.
    process.current_dir(workspace_root);
    apply_safe_env(&mut process);
    process
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
    SpawnFailed(std::io::Error),
}

/// Read from an async stream up to `cap` bytes, mirroring progress into `shared`.
async fn read_stream_limited(
    reader: &mut (impl tokio::io::AsyncRead + Unpin),
    cap: usize,
    shared: &Arc<Mutex<Vec<u8>>>,
) -> Vec<u8> {
    use tokio::io::AsyncReadExt;

    let mut chunk = [0u8; 8192];
    loop {
        let to_read = {
            let guard = shared.lock().unwrap_poison();
            if guard.len() >= cap {
                chunk.len()
            } else {
                (cap - guard.len()).min(chunk.len())
            }
        };
        match reader.read(&mut chunk[..to_read]).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let mut guard = shared.lock().unwrap_poison();
                if guard.len() < cap {
                    let take = n.min(cap - guard.len());
                    guard.extend_from_slice(&chunk[..take]);
                }
            }
        }
    }
    shared.lock().unwrap_poison().clone()
}

/// Spawn a background task that reads from an optional pipe into a shared buffer.
fn spawn_pipe_reader(
    pipe: Option<impl tokio::io::AsyncRead + Unpin + Send + 'static>,
    shared: Arc<Mutex<Vec<u8>>>,
) -> tokio::task::JoinHandle<Vec<u8>> {
    tokio::spawn(async move {
        if let Some(mut reader) = pipe {
            read_stream_limited(&mut reader, SHELL_PIPE_READ_CAP, &shared).await
        } else {
            Vec::new()
        }
    })
}

/// Spawn `cmd`, read stdout/stderr concurrently, and enforce `timeout`.
async fn run_command_with_timeout(
    cmd: &mut tokio::process::Command,
    timeout: Duration,
) -> ShellRunResult {
    let start = std::time::Instant::now();

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return ShellRunResult::SpawnFailed(e),
    };

    let pid = child.id();
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    let stdout_shared = Arc::new(Mutex::new(Vec::new()));
    let stderr_shared = Arc::new(Mutex::new(Vec::new()));

    let stdout_handle = spawn_pipe_reader(stdout_pipe, Arc::clone(&stdout_shared));
    let stderr_handle = spawn_pipe_reader(stderr_pipe, Arc::clone(&stderr_shared));

    match tokio::time::timeout(timeout, child.wait()).await {
        Ok(Ok(status)) => {
            let stdout = stdout_handle.await.unwrap_or_else(|e| {
                tracing::warn!(%e, "stdout reader task panicked");
                Vec::new()
            });
            let stderr = stderr_handle.await.unwrap_or_else(|e| {
                tracing::warn!(%e, "stderr reader task panicked");
                Vec::new()
            });
            ShellRunResult::Completed {
                stdout,
                stderr,
                status,
                elapsed: start.elapsed(),
            }
        }
        Ok(Err(e)) => ShellRunResult::SpawnFailed(e),
        Err(_) => {
            let _ = child.kill().await;
            let (_, _, _) = tokio::join!(
                tokio::time::timeout(Duration::from_secs(2), child.wait()),
                tokio::time::timeout(Duration::from_secs(2), stdout_handle),
                tokio::time::timeout(Duration::from_secs(2), stderr_handle),
            );
            let stdout = stdout_shared.lock().unwrap_poison().clone();
            let stderr = stderr_shared.lock().unwrap_poison().clone();
            ShellRunResult::TimedOut {
                stdout,
                stderr,
                pid,
                elapsed: start.elapsed(),
            }
        }
    }
}

fn tail_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let skip = s.chars().count().saturating_sub(max_chars);
    s.chars().skip(skip).collect()
}

/// Appends the tail of an output buffer to `msg` with a label (e.g. "stdout" or "stderr").
///
/// The transformation chain: lossy UTF-8 decode → strip ANSI escapes → scrub credentials
/// → truncate to [`TIMEOUT_OUTPUT_TAIL_CHARS`] characters.
fn append_output_tail(msg: &mut String, label: &str, data: &[u8]) {
    if !data.is_empty() {
        let decoded = String::from_utf8_lossy(data);
        let stripped = strip_ansi_escapes(&decoded);
        let scrubbed = scrub_credentials(&stripped);
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
    pid: Option<u32>,
    stdout: &[u8],
    stderr: &[u8],
) -> String {
    let mut msg = format!(
        "Shell command timed out.\n\
         command: {command}\n\
         elapsed: {:.1}s\n\
         timeout_limit: {DEFAULT_SHELL_TIMEOUT_SECS}s",
        elapsed.as_secs_f64()
    );
    if let Some(p) = pid {
        let _ = write!(msg, "\npid: {p}");
    }
    msg.push_str("\nreason: command was killed after exceeding the timeout");

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

    /// Execute a command and return `(formatted_output, exit_code)`.
    ///
    /// `exit_code` is `Some(0)` for success, `Some(n)` for non-zero exit, or
    /// `None` when the process was terminated by a signal.
    ///
    /// The `formatted_output` includes the `[exit status: N]` annotation for
    /// non-zero exits and signal termination (same format as [`execute()`]).
    /// Credentials are already scrubbed from the returned output.
    pub(crate) async fn execute_with_status(
        &self,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<(String, Option<i32>)> {
        let command_str = super::get_str(&args, "command")?;

        // Read-only mode: validate command before execution
        if self.mode == ShellMode::ReadOnly
            && let Err(rejection) = check_command(command_str)
        {
            anyhow::bail!("{rejection}");
        }

        // Execute with timeout to prevent hanging commands.
        // Use the ORIGINAL command string (not the stripped version) so that
        // `cd workspace/subdir && cargo build` actually navigates the shell.
        let mut cmd = build_shell_command(command_str, ws.as_path());

        let result =
            run_command_with_timeout(&mut cmd, Duration::from_secs(DEFAULT_SHELL_TIMEOUT_SECS))
                .await;

        match result {
            ShellRunResult::Completed {
                stdout,
                stderr,
                status,
                elapsed,
            } => {
                // Save raw output BEFORE any truncation so agents can access
                // the full output even when filtered/truncated to fit the context.
                let raw_hint = save_raw_output_if_large(&stdout, &stderr, command_str);

                let stdout = clean_truncate(&stdout, "output");
                let stderr = clean_truncate(&stderr, "stderr");

                let exit_code = status.code(); // Option<i32> — None means signal
                let exit_note = match exit_code {
                    Some(c) => format!("[exit status: {c}]"),
                    None => "[exit status: terminated by signal]".to_string(),
                };

                // All completed commands return output with exit info,
                // regardless of exit code. Only actual execution failures
                // (timeout, process launch failure) are tool errors.
                let processed = process_shell_output(
                    command_str,
                    &stdout,
                    &stderr,
                    exit_code.unwrap_or(-1),
                    elapsed,
                );
                let mut combined = processed;
                // Include raw output hint if truncation or spill occurred.
                if let Some(hint) = &raw_hint {
                    combined.push('\n');
                    combined.push_str(hint);
                }
                if exit_code != Some(0) {
                    combined.push_str("\n\n");
                    combined.push_str(&exit_note);
                }
                Ok((combined, exit_code))
            }
            ShellRunResult::TimedOut {
                stdout,
                stderr,
                pid,
                elapsed,
            } => {
                tracing::warn!(
                    command = command_str,
                    elapsed_secs = elapsed.as_secs_f64(),
                    ?pid,
                    stdout_bytes = stdout.len(),
                    stderr_bytes = stderr.len(),
                    "Shell command timed out"
                );
                let msg = format_timeout_error(command_str, elapsed, pid, &stdout, &stderr);
                anyhow::bail!("{msg}");
            }
            ShellRunResult::SpawnFailed(e) => anyhow::bail!(
                "Failed to start shell command.\n\
                 command: {command_str}\n\
                 reason: {e}"
            ),
        }
    }
}

/// Extra `PATH` entries prepended for shell subprocesses so developer tools
/// (`cargo`, Homebrew, npm global bins, etc.) resolve without reading the
/// parent process `PATH`.
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
    v
}

#[cfg(unix)]
const fn default_search_path_without_parent_env() -> &'static str {
    "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
}

#[cfg(windows)]
fn windows_system_root() -> String {
    r"C:\Windows".to_string()
}

#[cfg(windows)]
fn default_search_path_without_parent_env() -> String {
    let root = windows_system_root();
    format!(r"{root}\System32;{root};{root}\System32\Wbem;{root}\System32\WindowsPowerShell\v1.0")
}

fn prepend_path_entries(base: &str, extras: &[PathBuf]) -> String {
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
    let base = default_search_path_without_parent_env();
    prepend_path_entries(base, &extra_shell_path_prefixes())
}

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
        "TMPDIR" => Some("/tmp".into()),
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
/// These variables are only meaningful on Windows and provide sensible defaults
/// when they are not set in the parent process environment.
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
        "SYSTEMROOT" | "WINDIR" => Some(windows_system_root()),
        "SYSTEMDRIVE" => Some("C:".into()),
        "COMSPEC" => {
            let root = windows_system_root();
            Some(format!(r"{root}\System32\cmd.exe"))
        }
        "TEMP" | "TMP" => {
            let root = windows_system_root();
            Some(format!(r"{root}\Temp"))
        }
        _ => None,
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &'static str {
        "shell"
    }

    fn description(&self) -> String {
        match self.mode {
            ShellMode::ReadOnly => {
                const RESTRICTION: &str = "\
⚠️ READ-ONLY MODE: You are not permitted to modify the workspace. \
Commands that write files, delete files, or mutate git state will be rejected before execution. \
Writing to the OS temp directory is allowed. \
Use this tool only for inspection: reading files, listing directories, running cargo check/test/clippy, git status/log/diff, searching, etc.\n\n";
                let base = crate::prompt::load_prompt(&format!("tool/{}.md", self.name()));
                format!("{RESTRICTION}{base}")
            }
            ShellMode::Full => crate::prompt::load_prompt(&format!("tool/{}.md", self.name())),
        }
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
            }),
            &["command"],
        )
    }

    fn side_effects(&self) -> bool {
        // ReadOnly mode validates commands against a mutating-command blocklist
        // — best-effort guard, not a sandbox, but sufficient for grouping.
        self.mode != ShellMode::ReadOnly
    }

    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        false // shell pipeline already scrubs credentials internally at every output path
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        self.execute_with_status(ws, args)
            .await
            .map(|(output, _)| output)
    }

    fn debug_output(
        &self,
        phase: ToolOutputPhase,
        args: &serde_json::Value,
        outcome: Option<(&str, bool)>,
    ) -> Option<String> {
        match phase {
            ToolOutputPhase::Before => {
                let cmd = args.get("command").and_then(|v| v.as_str()).unwrap_or("?");
                Some(cmd.to_owned())
            }
            ToolOutputPhase::After => {
                let (output, _success) = outcome?;
                let trimmed = output.trim();
                if trimmed.is_empty() {
                    return None;
                }
                Some(crate::util::truncate_sandwich(trimmed, 2000, "debug"))
            }
        }
    }
}

const SPILL_THRESHOLD_BYTES: usize = 5_000;

/// Once-flag for cleaning up old spill files at daemon startup.
static SPILL_DIR_CLEANED: AtomicBool = AtomicBool::new(false);

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
/// to an output buffer (e.g., [`super::readonly::strip_heredoc_bodies`]
/// preserves the command string for redirect scanning, but
/// [`super::readonly::has_disallowed_redirect`] simply continues without
/// pushing).
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

/// Split a shell command string into logical segments at shell operators.
/// Respects single and double quotes.
fn extract_command_segments(command: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = command.chars().peekable();

    let mut flush = |current: &mut String| {
        if !current.trim().is_empty() {
            segments.push(current.trim().to_string());
        }
        current.clear();
    };

    while let Some(c) = chars.next() {
        // Handle backslash escape independently — must come before the shared
        // quote state machine to preserve the caller's peek-ahead behavior
        // and trailing-backslash push.  Because backslash never reaches
        // [`check_outside_quotes`], there is no need for an `escaped` flag
        // in the quote tracker; it only tracks single/double quote state.
        if c == '\\' && !in_single {
            if let Some(next) = chars.next() {
                current.push(next);
            } else {
                current.push(c); // trailing backslash preserved
            }
            continue;
        }

        if check_outside_quotes(c, &mut in_single, &mut in_double) {
            match c {
                '&' if chars.peek() == Some(&'&') => {
                    chars.next(); // consume second &
                    flush(&mut current);
                    continue;
                }
                '|' => {
                    if chars.peek() == Some(&'|') {
                        chars.next();
                    }
                    flush(&mut current);
                    continue;
                }
                ';' => {
                    flush(&mut current);
                    continue;
                }
                _ => {}
            }
        }
        current.push(c);
    }
    flush(&mut current);
    segments
}

/// Find the index of the first non-flag word in a slice, skipping:
/// - Git global flags (and their values) when `is_git` is true
/// - Any word starting with `-`
///
/// Shared helper used by [`canonical_command`] and `extract_git_subcommand`
/// to avoid duplicating the flag-skipping loop.
pub(super) fn find_first_non_flag_index(words: &[&str], is_git: bool) -> Option<usize> {
    let mut i = 0;
    while i < words.len() {
        let w = words[i];
        if is_git && GIT_GLOBAL_FLAGS.contains(&w) {
            i += 2; // skip flag and its value (safe: loop condition checks len)
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
/// to avoid duplicating the scanning logic.
pub(super) fn find_first_command_word_index(words: &[&str]) -> Option<usize> {
    words
        .iter()
        .position(|w| !SHELL_PREFIXES.contains(w) && !w.starts_with('-') && !is_env_assignment(w))
}

/// Shared helper that returns both the index and the basename of the first
/// command word from a bare list of words. Used by [`command_word_from_segment`]
/// to avoid duplicating index unwrap + basename extraction.
fn command_word_and_index<'a>(words: &[&'a str]) -> Option<(usize, &'a str)> {
    let cmd_idx = find_first_command_word_index(words)?;
    let basename = words[cmd_idx]
        .rsplit('/')
        .next()
        .expect("rsplit always yields at least one element");
    Some((cmd_idx, basename))
}

/// Extract the first command word index, basename, and the whitespace-split words
/// from a shell segment.
///
/// Trims the segment, splits on whitespace, and delegates to [`command_word_and_index`].
/// Returns `None` when no command word is found (e.g., only prefixes/flags/env assignments).
fn command_word_from_segment(segment: &str) -> Option<(usize, &str, Vec<&str>)> {
    let trimmed = segment.trim();
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let (idx, cmd) = command_word_and_index(&words)?;
    Some((idx, cmd, words))
}

/// Extract just the first command word (basename) from a shell segment.
///
/// Strips shell prefixes, environment variable assignments (`KEY=value`),
/// and absolute paths, but stops before any subcommand detection.
/// This is the lightweight alternative to [`canonical_command`] for callers
/// that only need the command name (e.g., `check_segment`).
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
pub(super) fn canonical_command(segment: &str) -> String {
    let Some((cmd_idx, cmd, words)) = command_word_from_segment(segment) else {
        return String::new();
    };

    // If no more words after the command, return just the command
    let remaining = &words[cmd_idx + 1..];
    if remaining.is_empty() {
        return cmd.to_string();
    }

    // Skip flags between command and subcommand using shared helper
    // cmd is the bare basename (rsplit('/').next()), so == comparison is safe.
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
/// (short-circuits on both segment and profile iteration); chained
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
/// On failure (non-zero exit): all stderr is appended unconditionally.
fn combine_output(
    stdout: &str,
    stderr: &str,
    exit_code: i32,
    keep_stderr: Option<&RegexSet>,
) -> String {
    let stderr_trimmed = stderr.trim();
    let exit_ok = exit_code == 0;
    if stderr_trimmed.is_empty() {
        return stdout.to_string();
    }
    if exit_ok {
        // Keep only warning/error lines from stderr on success
        if let Some(patterns) = keep_stderr {
            let relevant: Vec<&str> = stderr.lines().filter(|l| patterns.is_match(l)).collect();
            if !relevant.is_empty() {
                if stdout.is_empty() {
                    return format!("stderr:\n{}", relevant.join("\n"));
                }
                return format!("{stdout}\nstderr:\n{}", relevant.join("\n"));
            }
        }
        return stdout.to_string();
    }
    // Non-zero exit: show all stderr
    if stdout.is_empty() {
        return format!("stderr:\n{stderr_trimmed}");
    }
    format!("{stdout}\nstderr:\n{stderr_trimmed}")
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
    // Combined output is already credential-scrubbed upstream in the
    // `apply_profile_pipeline` combine closure, so no further scrubbing
    // is needed here.  ShellTool overrides `should_scrub_output` to return
    // `false`, disabling the agent-level scrub as redundant — the pipeline
    // guarantees scrubbing at every output path.  Stderr was already
    // scrubbed at `apply_profile_pipeline` entry.
    if elapsed.as_secs_f64() >= 1.0 {
        let _ = write!(combined, "\n[took {:.1}s]", elapsed.as_secs_f64());
    }

    // When pre-head/tail output is available and large enough, spill the
    // fuller version so agents can `read` the complete output despite the
    // head/tail truncation reducing inline content to a snippet.
    if let Some(pre) = full_output_for_spill
        && pre.len() > SPILL_THRESHOLD_BYTES
    {
        let scrubbed = scrub_credentials(pre);
        let byte_count = scrubbed.len();
        let line_count = scrubbed.lines().count();
        if let Some(path) = spill_output(&scrubbed) {
            let hint = format_spill_header(&path, byte_count, line_count);
            combined.push('\n');
            combined.push_str(&hint);
        }
        return combined;
    }

    // No pre-truncation spill — try spilling the final combined output
    try_spill_to_file(combined, SPILL_THRESHOLD_BYTES)
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

        // Skip passing test lines
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

/// Parse a single `ls -l` line. Returns `(file_type, size, name)` on success.
fn parse_ls_line(line: &str) -> Option<(char, String, String)> {
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
    let file_type = permissions.chars().next()?;
    parts.next(); // link count
    parts.next(); // owner
    parts.next(); // group
    let size = parts.next()?.to_string();
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
    Some((file_type, size, name))
}

/// Format a raw byte count into a human-readable size string.
#[allow(clippy::cast_precision_loss)]
fn human_readable_size(size: &str) -> String {
    if let Ok(bytes) = size.parse::<u64>() {
        if bytes >= 1_000_000_000 {
            format!("{:.1}G", bytes as f64 / 1_000_000_000.0)
        } else if bytes >= 1_000_000 {
            format!("{:.1}M", bytes as f64 / 1_000_000.0)
        } else if bytes >= 1_000 {
            format!("{:.1}K", bytes as f64 / 1_000.0)
        } else {
            format!("{bytes}B")
        }
    } else {
        size.to_string() // already human-readable
    }
}

/// Compress `ls -l` output into a compact directory listing.
/// Groups directories and files, shows sizes, and includes an extension summary.
///
/// Non-`-l` output (without a `total N` header) passes through unchanged.
pub(super) fn compact_ls(output: &str, _exit_code: i32) -> String {
    // If the output lacks a "total N" header, it's not in `-l` format.
    // `parse_ls_line` can only parse `-l` lines — passing non-`-l` output
    // through would trigger the "(empty)" false positive.
    if !output.lines().any(|line| line.starts_with("total ")) {
        return output.to_string();
    }

    let mut dirs: Vec<String> = Vec::new();
    let mut files: Vec<(String, String)> = Vec::new();
    let mut ext_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut lines_seen = 0usize;

    for line in output.lines() {
        if line.starts_with("total ") || line.trim().is_empty() {
            continue;
        }
        lines_seen += 1;

        let Some((file_type, size, name)) = parse_ls_line(line) else {
            continue;
        };

        if file_type == 'd' {
            dirs.push(name);
        } else {
            let ext = if let Some((_, e)) = name.rsplit_once('.') {
                format!(".{e}")
            } else {
                "no ext".to_string()
            };
            *ext_counts.entry(ext).or_insert(0) += 1;
            let human = human_readable_size(&size);
            files.push((name, human));
        }
    }

    if dirs.is_empty() && files.is_empty() {
        if lines_seen > 0 {
            return "(empty)\n".to_string();
        }
        return output.to_string();
    }

    let mut entries = String::new();

    for d in &dirs {
        let _ = writeln!(entries, "{d}/");
    }

    for (name, size) in &files {
        let _ = writeln!(entries, "{name}  {size}");
    }

    let _ = write!(
        entries,
        "Summary: {} files, {} dirs",
        files.len(),
        dirs.len()
    );
    if !ext_counts.is_empty() {
        let mut sorted: Vec<_> = ext_counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        let parts: Vec<String> = sorted
            .iter()
            .take(5)
            .map(|(ext, count)| format!("{count} {ext}"))
            .collect();
        let _ = write!(entries, " ({})", parts.join(", "));
        if sorted.len() > 5 {
            let _ = write!(entries, ", +{} more", sorted.len() - 5);
        }
    }
    entries.push('\n');

    entries
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
    apply_profile_pipeline(profile, stdout, stderr, exit_code, elapsed, is_chained)
}

/// Match short-circuit patterns against trimmed output.
fn match_short_circuit<'a>(output: &str, short_circuits: &'a [ShortCircuit]) -> Option<&'a str> {
    let blob = output.trim();
    for sc in short_circuits {
        if sc.pattern.is_match(blob)
            && !sc
                .unless
                .as_ref()
                .is_some_and(|unless_re| unless_re.is_match(blob))
        {
            return Some(sc.message);
        }
    }
    None
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

/// Split output into head lines, an omitted count, and tail lines.
///
/// When total lines ≤ head_count + tail_count, all lines are returned in `head`
/// with `omitted = 0` and an empty `tail`. Otherwise the output is split into
/// the first `head_count` lines and the last `tail_count` lines, with the
/// omitted line count recorded.
fn split_head_tail(
    output: &str,
    head_count: usize,
    tail_count: usize,
) -> (Vec<String>, usize, Vec<String>) {
    let lines: Vec<&str> = output.lines().collect();
    let total = lines.len();
    if total <= head_count + tail_count {
        (
            lines.iter().map(ToString::to_string).collect(),
            0,
            Vec::new(),
        )
    } else {
        let head = lines[..head_count.min(total)]
            .iter()
            .map(ToString::to_string)
            .collect();
        let tail = lines[total.saturating_sub(tail_count)..]
            .iter()
            .map(ToString::to_string)
            .collect();
        (head, total - head_count - tail_count, tail)
    }
}

/// Build a head/tail sandwich with an omitted-lines marker in the middle.
fn format_sandwich(output: &str, head: usize, tail: usize) -> String {
    let (head_lines, omitted, tail_lines) = split_head_tail(output, head, tail);
    if omitted == 0 {
        return output.to_string();
    }
    let mut v = head_lines;
    v.push(format!("... ({omitted} lines omitted)"));
    v.extend(tail_lines);
    v.join("\n")
}

/// Apply line truncation: head/tail sandwich (byte-gated), `max_lines`-only absolute cap, or passthrough.
///
/// Returns `(truncated_output, pre_truncation_copy)` where the copy captures the
/// full output before truncation for potential spilling by `finish_shell_output`.
/// Head/tail is gated on `SPILL_THRESHOLD_BYTES` — small outputs are shown in full
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
    let should_sandwich =
        (head > 0 || tail > 0) && line_count > head + tail && output.len() > SPILL_THRESHOLD_BYTES;

    let pre_truncation = if should_sandwich {
        Some(output.to_string())
    } else {
        None
    };

    // Single-pass truncation:
    // 1. Head/tail sandwich (byte-gated), OR
    // 2. max_lines-only absolute cap, OR
    // 3. passthrough.
    let result = if should_sandwich {
        format_sandwich(output, head, tail)
    } else if let Some(max) = max {
        cap_at_max_lines(output, max)
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

/// Cap `output` to at most `max` lines, appending a truncation marker
/// if lines were removed.
fn cap_at_max_lines(output: &str, max: usize) -> String {
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() > max {
        let truncated = lines.len() - max;
        let mut capped = lines[..max].join("\n");
        let _ = write!(capped, "\n... ({truncated} lines truncated)");
        capped
    } else {
        output.to_string()
    }
}

/// Run the full profile-based processing pipeline on pre-processed output.
///
/// Upstream processing (`execute()`) handles ANSI stripping and 1 MB truncation.
/// Stderr is scrubbed at entry so all paths (early-return and main) are consistent.
/// Stdout is scrubbed inside the `combine` closure so all output paths (JSON preview,
/// short-circuit, on_empty, main) consistently receive scrubbed output.
///
/// Early-return stages (JSON preview, short-circuit, on_empty) call
/// `combine_output` only.  The main path continues through `combine_output` →
/// `finish_shell_output` (timing, spill-to-file).  `SPILL_THRESHOLD_BYTES`
/// (5 KB) gates the head/tail truncation and spill preview — it is a trigger,
/// not a truncation cutoff.  `finish_shell_output` → `try_spill_to_file`'s
/// use of `crate::util::truncate_tool_output` (5 KB head+tail) provides a final
/// safety net for output that still exceeds the threshold after all pipeline
/// stages.
///
/// Stages: JSON preview, short-circuit, line filters, collapse, truncate,
///         line_truncation, on_empty, output_transform.
/// When `is_chained` is true (command has `&&`, `||`, `;`, or `|` segments),
/// short-circuit is skipped to avoid suppressing output from later segments.
fn apply_profile_pipeline(
    profile: &Profile,
    output: &str,
    stderr: &str,
    exit_code: i32,
    elapsed: Duration,
    is_chained: bool,
) -> String {
    // Scrub stderr once at the top of the pipeline so all early-return
    // paths (JSON preview, short-circuit, on_empty) consistently receive
    // scrubbed stderr.  Scrubbing before keep_stderr filtering means
    // credential lines that matched keep_stderr patterns will be dropped
    // entirely because the redacted version no longer matches — this is
    // acceptable since credentials should never appear in output.
    // Stdout is scrubbed inside the `combine` closure below so all output
    // paths uniformly receive scrubbed output at a single convergence point.
    let stderr = scrub_credentials(stderr);

    // Local closure capturing the trailing combine_output arguments (stderr,
    // exit_code, keep_stderr) to reduce repetition across all call sites.
    // Scrubbing stdout here ensures all output paths (JSON preview,
    // short-circuit, on_empty, main) consistently receive scrubbed output.
    let combine = |output: &str| {
        let scrubbed = scrub_credentials(output);
        combine_output(&scrubbed, &stderr, exit_code, profile.keep_stderr.as_ref())
    };

    // Stage 1: try JSON preview — early-return path; credentials are
    // scrubbed by the `combine` closure automatically.
    if let Some(json_preview) = try_json_preview(output) {
        return combine(&json_preview);
    }

    // Stage 2: short-circuit on success patterns — skip for chained commands
    // to avoid suppressing output from later segments (e.g., `cargo build && echo done`).
    if !is_chained && let Some(msg) = match_short_circuit(output, &profile.short_circuits) {
        return combine(msg);
    }

    let mut processed = output.to_string();

    // Stage 3: strip lines
    processed = apply_strip_lines(&processed, profile);

    // Stage 4: collapse blank lines (runs >2 → 2), then collapse consecutive
    // duplicate content lines (≥5 identical → [repeated N] marker).
    // collapse_consecutive_lines skips blank lines — otherwise [repeated]
    // markers on blank runs would prevent blank-line compression.
    // The blank-line skip makes the two passes order-independent.
    processed = collapse_blank_lines(&processed);
    processed = collapse_consecutive_lines(&processed);

    // Stage 5: truncate long lines
    if let Some(max) = profile.max_line_len {
        processed = truncate_line_width(&processed, max);
    }

    // Stage 6: line truncation (head/tail + max_lines).
    // Returns pre-truncation output for spilling by finish_shell_output,
    // so the complete content remains accessible even when truncation
    // reduces the inline view to a snippet.
    let (truncated, pre_head_tail) = apply_line_truncation(&processed, profile);
    processed = truncated;

    // Stage 7: on_empty — fallback when all output stripped
    if processed.trim().is_empty()
        && let Some(msg) = profile.on_empty
    {
        let exit_note = if exit_code == 0 { "" } else { " (failed)" };
        let secs = elapsed.as_secs_f64();
        return combine(&format!("{msg}{exit_note} ({secs:.1}s)"));
    }

    // Stage 8: output transform — replaces processed output before combine/finish.
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

/// Pass 1: Strip ANSI escape sequences.
fn strip_ansi_escapes(input: &str) -> String {
    static RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"\x1B\[[0-9;]*[a-zA-Z]|\x1B\][0-9;]*[^\x1B]*\x1B\\|\x1B[\(\)\[\]KM]|\x1B\][0-9;]*\x07",
        )
        .unwrap()
    });
    RE.replace_all(input, "").to_string()
}

/// Strip ANSI escape sequences first, then truncate to [`MAX_OUTPUT_BYTES`].
///
/// Applying [`truncate_sandwich`] after ANSI stripping guarantees that
/// truncation boundaries cannot split multi-character escape sequences
/// into garbled fragments.
fn clean_truncate(data: &[u8], label: &'static str) -> String {
    let s = String::from_utf8_lossy(data);
    let cleaned = strip_ansi_escapes(&s);
    crate::util::truncate_sandwich(&cleaned, MAX_OUTPUT_BYTES, label)
}

/// Try to parse as JSON/structured data and return a schema preview.
/// Returns `Some(preview)` if JSON was parsed, `None` otherwise.
fn try_json_preview(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }

    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(serde_json::Value::Array(arr)) => {
            let count = arr.len();
            let preview = if arr.is_empty() {
                String::from("[] (empty array)")
            } else {
                let sample = arr.iter().take(3).collect::<Vec<_>>();
                let types = infer_json_types(&sample);
                let entries = sample
                    .iter()
                    .map(|v| serde_json::to_string(v).unwrap_or_default())
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "[JSON array: {count} items, schema: {types}]\n{entries}\n(total: {} chars)",
                    input.len()
                )
            };
            Some(preview)
        }
        Ok(serde_json::Value::Object(obj)) => {
            let fields: Vec<String> = obj
                .iter()
                .map(|(k, v)| {
                    let t = json_value_type(v);
                    format!("  {k}: {t}")
                })
                .collect();
            let preview = format!(
                "[JSON object: {} fields]\n{}\n(total: {} chars)",
                fields.len(),
                fields.join("\n"),
                input.len()
            );
            Some(preview)
        }
        _ => None,
    }
}

/// Infer a schema from a list of JSON values (typically array elements).
fn infer_json_types(values: &[&serde_json::Value]) -> String {
    use std::collections::BTreeMap;
    let mut fields: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for v in values {
        if let Some(obj) = v.as_object() {
            for (k, val) in obj {
                fields
                    .entry(k)
                    .or_default()
                    .push(json_value_type(val).to_string());
            }
        }
    }
    if fields.is_empty() {
        return json_value_type(values.first().copied().unwrap_or(&serde_json::Value::Null))
            .to_string();
    }
    fields
        .iter()
        .map(|(k, types)| {
            let unique: Vec<&str> = {
                let mut v: Vec<&str> = types.iter().map(String::as_str).collect();
                v.sort_unstable();
                v.dedup();
                v
            };
            format!("{k}: {}", unique.join(" | "))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn json_value_type(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(n) => {
            if n.is_f64() {
                "float"
            } else {
                "int"
            }
        }
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Collapse consecutive identical non-blank lines (≥5 repetitions).
/// Blank/whitespace-only lines are passed through individually so
/// collapse_blank_lines can compress them without interference.
fn collapse_consecutive_lines(input: &str) -> String {
    const THRESHOLD: usize = 5;
    let mut result = String::with_capacity(input.len());
    let lines: Vec<&str> = input.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let current = lines[i];
        // Blank lines are handled by collapse_blank_lines — treat each
        // blank line individually to avoid [repeated N times] markers
        // that would prevent blank-line compression.
        if current.trim().is_empty() {
            push_line(&mut result, current);
            i += 1;
            continue;
        }
        let mut count = 1;
        while i + count < lines.len() && lines[i + count] == current {
            count += 1;
        }
        if count >= THRESHOLD {
            push_line(&mut result, current);
            let _ = write!(result, "\n[repeated {count} times]");
        } else {
            for _ in 0..count {
                push_line(&mut result, current);
            }
        }
        i += count;
    }
    result
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
                line.len() - cut
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
    format!("{header}{}", format_sandwich(output, 5, 5))
}

/// Get the shared temp directory for spill files and raw output logs.
/// On first call, purges any leftover files from previous sessions.
fn agent_temp_dir() -> Option<std::path::PathBuf> {
    let dir = std::env::temp_dir().join(".agent");

    // Clean up leftover spill files from previous daemon sessions once.
    if !SPILL_DIR_CLEANED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        let _ = cleanup_temp_dir(&dir);
    }

    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Remove all files in the given temp directory (spill files, raw logs, etc.).
///
/// These are purely ephemeral artifacts from the previous daemon session and are
/// safe to delete on startup — the current session will recreate them as needed.
fn cleanup_temp_dir(dir: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let _ = std::fs::remove_file(&path);
        }
    }
    Ok(())
}

/// Write content to the agent temp directory with the given filename.
/// Content should already be credential-scrubbed.
fn write_to_spill(content: &str, filename: &str) -> Option<std::path::PathBuf> {
    let dir = agent_temp_dir()?;
    let path = dir.join(filename);
    std::fs::write(&path, content).ok()?;
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
fn try_spill_to_file(output: String, threshold_bytes: usize) -> String {
    if output.len() <= threshold_bytes {
        return output;
    }

    match spill_output(&output) {
        Some(path) => format_spill_preview(&output, &path),
        None => crate::util::truncate_tool_output(&output),
    }
}

/// Save raw (pre-truncation) output to a temp file when 1MB truncation occurred,
/// so agents can access the full output even when filtering/truncation reduces the
/// inline view. Returns a spill header hint like "[Output saved to ...]" or None
/// if the save failed or no truncation occurred.
fn save_raw_output_if_large(
    stdout_bytes: &[u8],
    stderr_bytes: &[u8],
    command: &str,
) -> Option<String> {
    // Only save when actual truncation is needed — skip trivial commands
    if stdout_bytes.len() <= MAX_OUTPUT_BYTES && stderr_bytes.len() <= MAX_OUTPUT_BYTES {
        return None;
    }
    let slug: String = command
        .chars()
        .take(40)
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    let filename = format!("{epoch}_{slug}.raw.log");

    // Combine stdout + stderr with labels, strip ANSI escapes, scrub credentials
    // Order: strip first, then scrub — prevents ANSI-obfuscated credentials
    // from bypassing the regex-based scrubber, and matches append_output_tail.
    let raw = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(stdout_bytes),
        String::from_utf8_lossy(stderr_bytes)
    );
    let stripped = strip_ansi_escapes(&raw);
    let scrubbed = scrub_credentials(&stripped);
    let line_count = scrubbed.lines().count();
    let byte_count = scrubbed.len();
    let path = write_to_spill(&scrubbed, &filename)?;
    Some(format_spill_header(&path, byte_count, line_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::test_ws;
    use tempfile::TempDir;

    use crate::util::test::{env_lock, set_env_var};

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
        /// is stripped before comparison, consistent with how profiles
        /// trim output in short-circuit messages).
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
    /// short-circuit messages (e.g. `"[cargo test: ok]"`).
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

    #[allow(clippy::too_many_lines)]
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
                name: "git -C /repo diff triggers git diff short-circuit",
                command: "git -C /repo diff",
                eq: Some("[git diff: no changes]"),
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
                contains: &["prettier: ok"],
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

    #[allow(clippy::too_many_lines)]
    #[test]
    fn tool_profile_cases() {
        // Specific tool profile tests — consolidated table.
        // Covers short-circuit, strip, and transform behaviors of named
        // tool profiles (git, docker, df, du, make, rsync, tsc, gh,
        // terraform, pytest, and the generic fallback pipeline).
        let cases: &[ShellOutputCase] = &[
            ShellOutputCase {
                name: "git diff no changes short-circuits",
                command: "git diff",
                contains: &["no changes"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "docker build short-circuits on success",
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
                name: "generic pipeline: strips ANSI, collapses repeats, preserves content",
                command: "unknown",
                stdout: "Compiling foo v1.0.0 (/tmp)\nCompiling bar v2.0.0 (/tmp)\nresult: ok\nline1\nline2\nline3\nline3\nline3\nline3\nline3\nline3\nline3\n",
                contains: &["Compiling", "[repeated", "result: ok"],
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
                name: "rsync short-circuits on success",
                command: "rsync -avz source/ dest/",
                stdout: "building file list ... done\nsent 100 bytes  received 50 bytes\n\ntotal size is 98765  speedup is 658.43\n",
                eq: Some("ok (synced)"),
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
                name: "docker strips build steps and short-circuits",
                command: "docker build -t myapp .",
                stdout: "Step 1/10 : FROM node:18\nStep 2/10 : WORKDIR /app\n ---> Using cache\nSuccessfully built abc123\nSuccessfully tagged myapp:latest\n",
                contains: &["[docker build: ok]"],
                not_contains: &["Step "],
                ..Default::default()
            },
            ShellOutputCase {
                name: "gh strips warning noise",
                command: "gh pr create --fill",
                stdout: "  \n - some detail\nwarning: consider updating gh\n✓ Created pull request\n",
                contains: &["[gh: ok]"],
                not_contains: &["warning:"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "terraform short-circuits on no changes",
                command: "terraform plan",
                stdout: "data.aws_region.current: Refreshing state...\nNo changes. Your infrastructure matches the configuration.\n",
                contains: &["[terraform: no changes]"],
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

    /// `build_shell_command` clears the parent environment and only exposes
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

    #[tokio::test]
    async fn run_command_with_timeout_kills_long_sleep() {
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg("sleep 10");
        let result = run_command_with_timeout(&mut cmd, Duration::from_secs(1)).await;
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

    #[tokio::test]
    async fn run_command_with_timeout_captures_partial_stdout() {
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg("echo started; sleep 60");
        let result = run_command_with_timeout(&mut cmd, Duration::from_secs(2)).await;
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

    #[tokio::test]
    async fn shell_timeout_error_includes_diagnostics() {
        let tmp = TempDir::new().expect("tempdir");
        let mut cmd = build_shell_command("echo before-timeout; sleep 30", tmp.path());
        let result = run_command_with_timeout(&mut cmd, Duration::from_secs(1)).await;
        let ShellRunResult::TimedOut {
            stdout,
            stderr,
            pid,
            elapsed,
        } = result
        else {
            panic!("expected timeout");
        };
        let msg = format_timeout_error("echo test", elapsed, pid, &stdout, &stderr);
        assert!(msg.contains("elapsed:"), "msg: {msg}");
        assert!(msg.contains("timeout_limit:"), "msg: {msg}");
        assert!(msg.contains("before-timeout"), "msg: {msg}");

        // Verify ANSI escape sequences are stripped from timeout error messages
        let ansi_stdout = b"\x1B[31mred error\x1B[0m";
        let ansi_stderr = b"\x1B[1mBOLD STUFF\x1B[22m";
        let ansi_msg = format_timeout_error("test", elapsed, Some(42), ansi_stdout, ansi_stderr);
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

    // ── Shell compression pipeline tests ────────────────────────────

    #[test]
    fn ansi_escape_cases() {
        let cases: &[(&str, &str)] = &[
            ("\x1B[31mred\x1B[0m \x1B[1mbold\x1B[22m", "red bold"),
            ("hello world", "hello world"),
        ];
        for (input, expected) in cases {
            assert_eq!(strip_ansi_escapes(input), *expected, "input: {input:?}");
        }
    }

    #[test]
    fn try_json_preview_cases() {
        let cases: &[(&str, &[&str])] = &[
            // JSON array → detected with item count and schema inference
            (
                r#"[{"name": "alice", "age": 30}, {"name": "bob", "age": 25}]"#,
                &["2 items", "name: string", "age: int"],
            ),
            // JSON object → detected with field count and field names
            (
                r#"{"status": "ok", "count": 42}"#,
                &["2 fields", "status", "count"],
            ),
            // Non-JSON → None
            ("hello world\nthis is not json", &[]),
        ];
        for (input, expected_contains) in cases {
            let result = try_json_preview(input);
            if expected_contains.is_empty() {
                assert!(result.is_none(), "expected no JSON preview for: {input:?}");
            } else {
                assert!(result.is_some(), "expected JSON preview for: {input:?}");
                let output = result.unwrap();
                for s in *expected_contains {
                    assert!(output.contains(s), "expected {s:?} in: {output}");
                }
            }
        }
    }

    #[test]
    fn pipeline_credential_scrubbing_cases() {
        // Pipeline stages 1/2/7 all scrub credentials in stderr.
        // Each case follows the same pattern: raw credentials not present,
        // redacted form present, pipeline-specific content preserved.
        let cases: &[ShellOutputCase] = &[
            ShellOutputCase {
                // Stage 1 (JSON preview): credentials in stdout are scrubbed
                name: "json preview scrubs credentials in array",
                command: "echo test",
                stdout: r#"[{"token": "sk-abcdefghijklmnop12345678", "name": "test"}]"#,
                not_contains: &["sk-abcdefghijklmnop12345678"],
                contains: &["sk-a*[REDACTED]", "test"],
                ..Default::default()
            },
            ShellOutputCase {
                // Stage 1 (JSON preview): object field values are scrubbed, names preserved
                name: "json preview scrubs credentials in object",
                command: "curl api",
                stdout: r#"{"api_key": "abcdefghijklmnop12345678", "status": "ok"}"#,
                not_contains: &["abcdefghijklmnop12345678"],
                contains: &["api_key", "status", "[JSON object:"],
                ..Default::default()
            },
            ShellOutputCase {
                name: "json preview preserves clean json",
                command: "echo test",
                stdout: r#"[{"name": "alice", "age": 30}]"#,
                contains: &["alice", "age", "30", "[JSON array:"],
                ..Default::default()
            },
            ShellOutputCase {
                // Stage 1 (JSON preview): credentials from stderr are scrubbed
                name: "json preview scrubs credentials in stderr",
                command: "echo test",
                stdout: r#"[{"name": "alice"}]"#,
                stderr: "api_key=abcdefghijklmnop12345678",
                exit_code: 1,
                not_contains: &["api_key=abcdefghijklmnop12345678"],
                contains: &["api_key=abcd*[REDACTED]", "alice"],
                ..Default::default()
            },
            ShellOutputCase {
                // Stage 2 (short-circuit): credentials from stderr are scrubbed
                name: "short-circuit scrubs credentials in stderr",
                command: "git diff",
                stderr: "api_key=abcdefghijklmnop12345678",
                exit_code: 1,
                not_contains: &["api_key=abcdefghijklmnop12345678"],
                contains: &["api_key=abcd*[REDACTED]", "no changes"],
                ..Default::default()
            },
            ShellOutputCase {
                // Stage 7 (on-empty): credentials from stderr are scrubbed
                name: "on-empty scrubs credentials in stderr",
                command: "tsc --noEmit",
                stderr: "api_key=abcdefghijklmnop12345678",
                exit_code: 1,
                not_contains: &["api_key=abcdefghijklmnop12345678"],
                contains: &["api_key=abcd*[REDACTED]", "[tsc: ok]"],
                ..Default::default()
            },
        ];
        check_shell_output(cases);
    }

    #[test]
    fn collapse_consecutive_lines_cases() {
        let cases: &[(&str, &str)] = &[
            // 6 identical "b" lines → collapsed with [repeated N times] marker
            ("a\nb\nb\nb\nb\nb\nb\nc", "a\nb\n[repeated 6 times]\nc"),
            // 3 identical "b" lines → below threshold (5), pass through unchanged
            ("a\nb\nb\nb\nc", "a\nb\nb\nb\nc"),
        ];
        for (input, expected) in cases {
            let result = collapse_consecutive_lines(input);
            assert_eq!(result, *expected, "input: {input:?}");
        }
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
        assert_eq!(
            try_spill_to_file(short.clone(), 5_000),
            short,
            "short output should pass through"
        );

        // Large single-line output spills
        let large = "x".repeat(10_000);
        let result = try_spill_to_file(large, 5_000);
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
            multi_len > 5_000,
            "test data {multi_len} must exceed spill threshold"
        );
        let result = try_spill_to_file(multi, 5_000);
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

    #[cfg(unix)]
    #[test]
    fn resolved_shell_path_includes_npm_global_bin() {
        let path = resolved_shell_path();
        assert!(
            path.contains(".npm-global/bin"),
            "PATH should include ~/.npm-global/bin for globally installed npm tools: {path}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolved_shell_path_includes_cargo_bin() {
        // Belt-and-suspenders means ~/.cargo/bin is always added regardless
        // of $CARGO_HOME, so no env manipulation needed here.
        let path = resolved_shell_path();
        assert!(
            path.contains(".cargo/bin"),
            "PATH should include ~/.cargo/bin: {path}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolved_shell_path_includes_cargo_home_when_set() {
        let _guard = set_env_var("CARGO_HOME", Some("/custom/cargo"));
        let path = resolved_shell_path();

        assert!(
            path.contains("/custom/cargo/bin"),
            "PATH should include $CARGO_HOME/bin when CARGO_HOME is set: {path}"
        );
        // Default ~/.cargo/bin should also be present (belt-and-suspenders).
        assert!(
            path.contains(".cargo/bin"),
            "PATH should still include ~/.cargo/bin when CARGO_HOME is set (belt-and-suspenders): {path}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolved_shell_path_dedup_cargo_home_and_default() {
        let Some(dirs) = UserDirs::new() else {
            // No home directory — skip dedup test (cannot determine default path).
            return;
        };

        let default_cargo_home = dirs.home_dir().join(".cargo").to_string_lossy().to_string();
        let _guard = set_env_var("CARGO_HOME", Some(&default_cargo_home));
        let path = resolved_shell_path();

        // Count occurrences of the default cargo bin path.
        let sep = ":";
        let count = path
            .split(sep)
            .filter(|part| *part == format!("{default_cargo_home}/bin"))
            .count();
        assert_eq!(
            count, 1,
            "$CARGO_HOME/bin and ~/.cargo/bin should deduplicate when they point to the same directory"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn resolved_shell_path_includes_homebrew() {
        let path = resolved_shell_path();
        assert!(
            path.contains("/opt/homebrew/bin"),
            "PATH should include Homebrew bin on macOS: {path}"
        );
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

    /// 5+ consecutive blank lines should collapse to 2 without `[repeated]` markers,
    /// regardless of whether collapse_blank_lines or collapse_consecutive_lines runs first.
    #[test]
    fn collapse_blank_lines_then_consecutive_no_marker() {
        let input = "a\n\n\n\n\n\nb"; // 6 blank lines between a and b

        // Forward order: blank first, then consecutive
        let forward = collapse_blank_lines(input);
        let forward = collapse_consecutive_lines(&forward);
        assert_eq!(
            forward, "a\n\n\nb",
            "6 blank lines → 2 blanks, no [repeated] marker"
        );
        assert!(
            !forward.contains("[repeated"),
            "should not contain repeated marker for blank lines"
        );

        // Reverse order: consecutive first, then blank — should produce same result
        let reverse = collapse_consecutive_lines(input);
        let reverse = collapse_blank_lines(&reverse);
        assert_eq!(forward, reverse, "both orders produce same result");
        assert!(
            !reverse.contains("[repeated"),
            "no [repeated] marker in reverse order"
        );
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

    #[test]
    fn save_raw_output_if_large_skips_small_output() {
        let result = save_raw_output_if_large(b"hello", b"", "echo hello");
        assert!(result.is_none(), "should skip saving for small output");
    }

    #[test]
    fn save_raw_output_if_large_saves_large_output() {
        let large = vec![b'a'; MAX_OUTPUT_BYTES + 1];
        let result = save_raw_output_if_large(&large, b"", "large-test");
        assert!(result.is_some(), "should save for oversized output");
        let hint = result.unwrap();
        assert!(
            hint.contains("[Output saved to"),
            "should mention saved file"
        );
        assert!(
            hint.contains("[view with: read"),
            "should provide read hint"
        );
    }

    #[test]
    fn save_raw_output_if_large_strips_ansi_escapes() {
        // Build output with ANSI escape sequences above the spill threshold
        let ansi_green = "\x1B[0;32m";
        let ansi_reset = "\x1B[0m";
        let inner =
            format!("{ansi_green}output line{ansi_reset}\n{ansi_green}another line{ansi_reset}");
        // Pad to exceed MAX_OUTPUT_BYTES while preserving ANSI content
        let padding = " ".repeat(MAX_OUTPUT_BYTES.saturating_sub(inner.len()) + 1);
        let ansi_content = format!("{inner}{padding}");
        let stdout_bytes = ansi_content.as_bytes();

        let result = save_raw_output_if_large(stdout_bytes, b"", "ansi-test");
        assert!(
            result.is_some(),
            "should save for oversized output with ANSI"
        );
        let hint = result.unwrap();

        // Extract the spill file path from the hint
        let path_str = hint
            .strip_prefix("[Output saved to ")
            .and_then(|s| s.split_once(' '))
            .map(|(path, _)| path)
            .expect("should parse path from hint");
        let path = std::path::Path::new(path_str);
        assert!(path.exists(), "spill file should exist");

        let spill_content = std::fs::read_to_string(path).expect("should read spill file");

        // Verify ANSI escapes were stripped
        assert!(
            !spill_content.contains("\x1B["),
            "spill file should not contain ANSI escapes: {spill_content:?}"
        );
        assert!(
            !spill_content.contains(ansi_green),
            "spill file should not contain ANSI green code"
        );
        assert!(
            spill_content.contains("output line"),
            "spill file should contain the actual output text"
        );
        assert!(
            spill_content.contains("another line"),
            "spill file should contain all output text"
        );
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
        // Generate output large enough to exceed SPILL_THRESHOLD_BYTES
        let lines: Vec<String> = (0..100)
            .map(|i| {
                format!(
                    "line {i} aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                )
            })
            .collect();
        let output = lines.join("\n");
        assert!(
            output.len() > SPILL_THRESHOLD_BYTES,
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
        ];
        for (input, head, tail, expected) in cases {
            let result = format_sandwich(input, *head, *tail);
            assert_eq!(
                result, *expected,
                "format_sandwich({input:?}, {head}, {tail})"
            );
        }
    }

    // ── finish_shell_output unit tests ────────────────────────────────
    // Credential scrubbing is now performed upstream in the `apply_profile_pipeline`
    // combine closure, so `finish_shell_output` receives pre-scrubbed input.
    // These tests verify that non-scrubbing behavior (timing, spill, idempotence)
    // remains correct.

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
            // exceed SPILL_THRESHOLD_BYTES, triggering the spill-to-file branch
            // in finish_shell_output. The spill content scrubbing is done
            // separately in finish_shell_output (via line 1012) — this test
            // verifies that the spill hint is properly appended.
            let pre_owned = case.pre.map(|s| s.repeat(SPILL_THRESHOLD_BYTES + 1));
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
}
