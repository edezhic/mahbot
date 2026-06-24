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
];

/// Git global flags that may appear between `git` and its subcommand.
/// These flags take a value (the next word after the flag).
pub(super) const GIT_GLOBAL_FLAGS: &[&str] = &["-C", "--git-dir", "--work-tree", "--bare", "-c"];

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

fn apply_safe_env(cmd: &mut tokio::process::Command) {
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
    #[cfg(not(target_os = "windows"))]
    {
        let mut process = tokio::process::Command::new("sh");
        process.arg("-c").arg(command).current_dir(workspace_root);
        apply_safe_env(&mut process);
        process
    }

    #[cfg(target_os = "windows")]
    {
        const CREATE_NO_WINDOW: u32 = 0x08000000;

        let mut process = tokio::process::Command::new("cmd.exe");
        process
            .arg("/C")
            .arg(command)
            .current_dir(workspace_root)
            .creation_flags(CREATE_NO_WINDOW);
        apply_safe_env(&mut process);
        process
    }
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

    let stdout_buf = Arc::clone(&stdout_shared);
    let stdout_handle = tokio::spawn(async move {
        if let Some(mut out) = stdout_pipe {
            read_stream_limited(&mut out, SHELL_PIPE_READ_CAP, &stdout_buf).await
        } else {
            Vec::new()
        }
    });
    let stderr_buf = Arc::clone(&stderr_shared);
    let stderr_handle = tokio::spawn(async move {
        if let Some(mut err) = stderr_pipe {
            read_stream_limited(&mut err, SHELL_PIPE_READ_CAP, &stderr_buf).await
        } else {
            Vec::new()
        }
    });

    match tokio::time::timeout(timeout, child.wait()).await {
        Ok(Ok(status)) => {
            let stdout = stdout_handle.await.unwrap_or_default();
            let stderr = stderr_handle.await.unwrap_or_default();
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
            let _ = tokio::time::timeout(Duration::from_secs(2), child.wait()).await;
            let _ = tokio::time::timeout(Duration::from_secs(2), stdout_handle).await;
            let _ = tokio::time::timeout(Duration::from_secs(2), stderr_handle).await;
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

    if !stdout.is_empty() {
        let scrubbed = scrub_credentials(&String::from_utf8_lossy(stdout));
        let tail = tail_chars(&scrubbed, TIMEOUT_OUTPUT_TAIL_CHARS);
        let _ = write!(
            msg,
            "\nstdout (last {} chars): {tail}",
            tail.chars().count()
        );
    }
    if !stderr.is_empty() {
        let scrubbed = scrub_credentials(&String::from_utf8_lossy(stderr));
        let tail = tail_chars(&scrubbed, TIMEOUT_OUTPUT_TAIL_CHARS);
        let _ = write!(
            msg,
            "\nstderr (last {} chars): {tail}",
            tail.chars().count()
        );
    }
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
}

/// Extra `PATH` entries prepended for shell subprocesses so developer tools
/// (`cargo`, Homebrew, npm global bins, etc.) resolve without reading the
/// parent process `PATH`.
///
/// # `$CARGO_HOME` handling
///
/// Checks `$CARGO_HOME` first (if set and non-empty) so users with a non-default
/// `CARGO_HOME` get their cargo-installed tools visible. Always adds
/// `~/.cargo/bin` via `UserDirs` as a fallback for default installs. When both
/// point to the same directory, deduplication in [`prepend_path_entries`] handles
/// it — the belt-and-suspenders approach ensures tools installed via either path
/// are found.
///
/// This follows the same resolution order as
/// [`crate::self_update::resolve_cargo_bin_path`].
#[cfg(unix)]
fn extra_shell_path_prefixes() -> Vec<PathBuf> {
    let mut v = Vec::new();

    // Check $CARGO_HOME first — users with a non-default CARGO_HOME
    // need $CARGO_HOME/bin in PATH for cargo-installed tools.
    // Dedup with ~/.cargo/bin is handled by prepend_path_entries.
    if let Ok(cargo_home) = std::env::var("CARGO_HOME")
        && !cargo_home.is_empty()
    {
        v.push(PathBuf::from(cargo_home).join("bin"));
    }

    if let Some(dirs) = UserDirs::new() {
        let home = dirs.home_dir();
        v.push(home.join(".cargo").join("bin"));
        v.push(home.join(".npm-global").join("bin"));
    }
    #[cfg(target_os = "macos")]
    {
        v.push(PathBuf::from("/opt/homebrew/bin"));
        v.push(PathBuf::from("/usr/local/bin"));
    }
    v
}

/// Extra `PATH` entries prepended for shell subprocesses so developer tools
/// (`cargo`, etc.) resolve without reading the parent process `PATH`.
///
/// # `$CARGO_HOME` handling
///
/// Same belt-and-suspenders approach as the unix variant: checks `$CARGO_HOME`
/// first, then adds `~/.cargo/bin` via `UserDirs`.
#[cfg(windows)]
fn extra_shell_path_prefixes() -> Vec<PathBuf> {
    let mut v = Vec::new();

    // Check $CARGO_HOME first.
    if let Ok(cargo_home) = std::env::var("CARGO_HOME")
        && !cargo_home.is_empty()
    {
        v.push(PathBuf::from(cargo_home).join("bin"));
    }

    if let Some(dirs) = UserDirs::new() {
        v.push(dirs.home_dir().join(".cargo").join("bin"));
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
            // Safety: Windows home paths always start with a drive letter
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

    fn side_effects(&self, _args: &serde_json::Value) -> bool {
        // ReadOnly mode validates commands against a mutating-command blocklist
        // — best-effort guard, not a sandbox, but sufficient for grouping.
        self.mode != ShellMode::ReadOnly
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
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

                // Strip ANSI escape codes before truncation so that
                // truncation boundaries cannot split multi-character
                // escape sequences into garbled fragments.
                let stdout_str = String::from_utf8_lossy(&stdout);
                let cleaned_stdout = strip_ansi_escapes(&stdout_str);
                let stdout =
                    crate::util::truncate_sandwich(&cleaned_stdout, MAX_OUTPUT_BYTES, "output");
                let stderr_str = String::from_utf8_lossy(&stderr);
                let stderr =
                    crate::util::truncate_sandwich(&stderr_str, MAX_OUTPUT_BYTES, "stderr");

                let (exit_code, exit_note) = match status.code() {
                    Some(c) => (c, format!("[exit status: {c}]")),
                    None => (-1, "[exit status: terminated by signal]".to_string()),
                };

                // All completed commands return output with exit info,
                // regardless of exit code. Only actual execution failures
                // (timeout, process launch failure) are tool errors.
                let processed =
                    process_shell_output(command_str, &stdout, &stderr, exit_code, elapsed);
                let mut combined = processed;
                // Include raw output hint if truncation or spill occurred.
                if let Some(hint) = &raw_hint {
                    combined.push('\n');
                    combined.push_str(hint);
                }
                if exit_code != 0 {
                    combined.push_str("\n\n");
                    combined.push_str(&exit_note);
                }
                Ok(combined)
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

    fn debug_output(
        &self,
        phase: ToolOutputPhase,
        args: &serde_json::Value,
        outcome: Option<&crate::tools::ToolExecutionOutcome>,
    ) -> Option<String> {
        match phase {
            ToolOutputPhase::Before => {
                let cmd = args.get("command").and_then(|v| v.as_str()).unwrap_or("?");
                Some(cmd.to_owned())
            }
            ToolOutputPhase::After => {
                let outcome = outcome?;
                let trimmed = outcome.output.trim();
                if trimmed.is_empty() {
                    return None;
                }
                Some(crate::util::truncate_sandwich(trimmed, 2000, "debug"))
            }
        }
    }
}

// ── Shell output processing pipeline ──────────────────────────────────
//
// Single dispatch path through the profile system:
//
// Phase 1 — process_shell_output (select profile + dispatch):
//   NOTE: ANSI stripping and 1MB truncation now happen in execute(),
//   before process_shell_output is called.
//   1. Extract command segments
//   2. Select matching profile (or GEN_FALLBACK)
//   3. Dispatch to apply_profile_pipeline
//
// Phase 2 — apply_profile_pipeline (profile-driven stages):
//   Stage numbers below match inline comments in apply_profile_pipeline.
//   1.  json_preview           — if output is JSON, return a schema preview
//   2.  short_circuit          — match success patterns (skipped for chained
//                                commands to preserve later-segment output)
//   3.  strip lines            — drop lines via regex patterns
//   4a. collapse_blank_lines    — collapse blank lines first (must precede 4b)
//   4b. collapse_consecutive    — collapse consecutive identical content lines
//   5.  max_line_len           — cap individual line length
//   6.  line_truncation        — head/tail sandwich + max_lines cap
//   7.  on_empty               — fallback message when all output stripped
//   8.  output_transform       — custom transform (cargo test state machine,
//                                ls compact parser, etc.). `standalone_only`
//                                profiles are excluded by `select_profile` for
//                                chained commands, falling through to
//                                GEN_FALLBACK with its truncation defaults.
//
// The main pipeline path ends with combine_output + finish_shell_output.
// Early-return paths (json_preview, short_circuit, on_empty) call combine_output
// only, since their output is already short or includes its own timing:
//   combine_output         — merge stderr for non-zero exit; filter stderr
//                            warnings on success when profile has keep_stderr
//   finish_shell_output    — append elapsed timing, spill to file for large
//                            output. Only the main path reaches this stage;
//                            early-return paths handle their own timing or
//                            produce output that doesn't need spilling.
//
// Output size is controlled by the profile pipeline: head_lines/tail_lines,
// max_line_len, and max_lines reduce inline content. SPILL_THRESHOLD_BYTES
// gates stage-6 head/tail and the pre-truncation spill in finish_shell_output,
// but is not itself a truncation target. finish_shell_output() replaces output
// >5K with a short spill preview instead of the full content. The default
// format_output() (5K head+tail truncation) acts as a secondary safety net
// for any output that still exceeds 5K after all pipeline stages — no custom
// override needed.

const SPILL_THRESHOLD_BYTES: usize = 5_000;

/// Once-flag for cleaning up old spill files at daemon startup.
static SPILL_DIR_CLEANED: AtomicBool = AtomicBool::new(false);

// ── Pipeline functions ────────────────────────────────────────────────

/// Shared quote-tracking state machine used by both [`extract_command_segments`]
/// and `super::readonly::has_disallowed_redirect` to avoid duplicating shell
/// quoting logic. Returns `true` when `c` is outside quotes and should be
/// examined for shell operators or redirect patterns.
///
/// This function tracks ONLY quote state — escape handling is the caller's
/// responsibility. Callers that need backslash escape semantics must handle
/// `\\` detection before invoking this function.
///
/// # Known limitation
///
/// Inside double quotes, `\` should only escape `\`, `$`, `` ` ``, `"`, and
/// newline in a real shell. The escape handling in both callers treats any
/// backslash as an escape, which is acceptable for our use cases (segment
/// splitting and redirect detection), where over-escaping is safe (false
/// negative for redirect detection = allows command; no redirect inside
/// double-quoted strings is actually harmful since the redirect operator is
/// quoted).
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
pub(super) fn is_env_assignment(word: &str) -> bool {
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
    let exit_ok = exit_code == 0;
    if stderr.trim().is_empty() {
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
        return format!("stderr:\n{}", stderr.trim());
    }
    format!("{stdout}\nstderr:\n{}", stderr.trim())
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
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(line);
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
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(&f.summary_lines.join("\n"));
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

/// Apply line truncation in a single pass: head/tail sandwich (byte-gated) with a
/// defensive `max_lines` cap, `max_lines`-only absolute cap, or passthrough.
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
    // 1. Head/tail sandwich (byte-gated) with a defensive max cap, OR
    // 2. max_lines-only absolute cap, OR
    // 3. passthrough.
    let mut result = if should_sandwich {
        let (head_lines, omitted, tail_lines) = split_head_tail(output, head, tail);
        let mut v = head_lines;
        v.push(format!("... ({omitted} lines omitted)"));
        v.extend(tail_lines);
        v.join("\n")
    } else if let Some(max) = max {
        cap_at_max_lines(output, max)
    } else {
        output.to_string()
    };

    // Defensive max cap on the sandwich path: ensures the sandwich +
    // omission marker doesn't exceed max_lines even if the
    // head+tail+1 <= max invariant is violated.
    if should_sandwich
        && let Some(max) = max
        && result.lines().count() > max
    {
        result = cap_at_max_lines(&result, max);
    }

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
/// Stages: JSON preview, short-circuit, line filters, collapse, truncate,
///         line_truncation, on_empty.
/// When `is_chained` is true (command has `&&`, `||`, `;`, or `|` segments), short-circuit is
/// skipped to avoid suppressing output from later segments.
fn apply_profile_pipeline(
    profile: &Profile,
    output: &str,
    stderr: &str,
    exit_code: i32,
    elapsed: Duration,
    is_chained: bool,
) -> String {
    // Stage 1: try JSON preview — if output is JSON, return schema preview early
    if let Some(json_preview) = try_json_preview(output) {
        return combine_output(
            &json_preview,
            stderr,
            exit_code,
            profile.keep_stderr.as_ref(),
        );
    }

    // Stage 2: short-circuit on success patterns — skip for chained commands
    // to avoid suppressing output from later segments (e.g., `cargo build && echo done`).
    if !is_chained && let Some(msg) = match_short_circuit(output, &profile.short_circuits) {
        return combine_output(msg, stderr, exit_code, profile.keep_stderr.as_ref());
    }

    let mut processed = output.to_string();

    // Stage 3: strip lines
    processed = apply_strip_lines(&processed, profile);

    // Stage 4-4b: collapse blank lines first, then consecutive content lines.
    // Blank-line collapse must run first: otherwise collapse_consecutive_lines
    // (threshold ≥5) sees 5+ identical blank lines as a "run" and emits
    // [repeated N times] markers, which break the blank-line run and prevent
    // collapse_blank_lines from compressing them.
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
        return combine_output(
            &format!("{msg}{exit_note} ({secs:.1}s)"),
            stderr,
            exit_code,
            profile.keep_stderr.as_ref(),
        );
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

    let combined = combine_output(&processed, stderr, exit_code, profile.keep_stderr.as_ref());
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

/// Try to parse as JSON/structured data and return a schema preview.
/// Returns `Some(preview)` if JSON was parsed, `None` otherwise.
fn try_json_preview(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() || (!trimmed.starts_with('[') && !trimmed.starts_with('{')) {
        return None;
    }

    // Try top-level array
    if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(trimmed) {
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
        return Some(preview);
    }

    // Try top-level object
    if let Ok(obj) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(trimmed) {
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
        return Some(preview);
    }

    None
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

/// Collapse consecutive identical lines (≥5 repetitions).
fn collapse_consecutive_lines(input: &str) -> String {
    const THRESHOLD: usize = 5;
    let mut result = String::with_capacity(input.len());
    let lines: Vec<&str> = input.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let current = lines[i];
        let mut count = 1;
        while i + count < lines.len() && lines[i + count] == current {
            count += 1;
        }
        if count >= THRESHOLD {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(current);
            let _ = write!(result, "\n[repeated {count} times]");
            i += count;
        } else {
            for _ in 0..count {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str(current);
            }
            i += count;
        }
    }
    result
}

/// Truncate any single line exceeding `max_line_len` with a note.
fn truncate_line_width(input: &str, max_line_len: usize) -> String {
    let mut result = String::with_capacity(input.len());
    for line in input.lines() {
        if !result.is_empty() {
            result.push('\n');
        }
        if line.len() > max_line_len {
            let cut = line.floor_char_boundary(max_line_len);
            result.push_str(&line[..cut]);
            let _ = write!(
                result,
                "\n... ({} more chars on this line)",
                line.len() - cut
            );
        } else {
            result.push_str(line);
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

    let (head_lines, omitted, tail_lines) = split_head_tail(output, 5, 5);
    if omitted == 0 {
        format!("{header}{output}")
    } else {
        format!(
            "{}{}\n... ({} lines omitted)\n{}",
            header,
            head_lines.join("\n"),
            omitted,
            tail_lines.join("\n"),
        )
    }
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
    let filename = format!("spill_{:04x}.txt", rand::random::<u16>());
    write_to_spill(output, &filename)
}

/// If output exceeds threshold, spill to a temp file and return a preview.
/// The full output is saved to a file; the inline preview is a short summary.
fn try_spill_to_file(output: String, threshold_bytes: usize) -> String {
    if output.len() <= threshold_bytes {
        return output;
    }

    let scrubbed = scrub_credentials(&output);
    match spill_output(&scrubbed) {
        Some(path) => format_spill_preview(&scrubbed, &path),
        None => crate::util::format_tool_output(&scrubbed),
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

    // Combine stdout + stderr with labels, scrub credentials
    let raw = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(stdout_bytes),
        String::from_utf8_lossy(stderr_bytes)
    );
    let scrubbed = scrub_credentials(&raw);
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

    /// Mutex serializing env-var-modifying tests to prevent thread-safety
    /// issues with `std::env::set_var` (which is `unsafe` in Rust 2024).
    ///
    /// ## Cross-module coordination
    ///
    /// `self_update.rs` has its own `ENV_LOCK` — the two locks are independent.
    /// A shell test that reads `CARGO_HOME` (under this lock) could theoretically
    /// race with a `self_update` test that writes it (under the other lock).
    /// To mitigate this, tests that call into code reading `$CARGO_HOME` (via
    /// `extra_shell_path_prefixes`) hold this lock during the read, while
    /// write tests in both modules hold their respective locks for minimal
    /// durations. The race window is negligible in practice since env var
    /// accesses are instantaneous, but the lock pattern documents the
    /// synchronization boundary.
    static ENV_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    fn env_lock() -> &'static std::sync::Mutex<()> {
        ENV_LOCK.get_or_init(|| std::sync::Mutex::new(()))
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
    /// Acquires [`ENV_LOCK`] because `build_shell_command` → `resolved_shell_path`
    /// → `extra_shell_path_prefixes` reads `$CARGO_HOME` from the environment,
    /// which concurrent tests in `self_update.rs` may write under their own lock.
    #[cfg(unix)]
    #[tokio::test]
    async fn build_shell_command_isolates_environment() {
        let tmp = TempDir::new().expect("tempdir");
        // Acquire ENV_LOCK while building the command since extra_shell_path_prefixes
        // reads $CARGO_HOME — concurrent self_update tests write it under a different
        // lock, so holding our own prevents the theoretical data race.
        let mut cmd = {
            let _guard = env_lock().lock().unwrap();
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
    }

    // ── Shell compression pipeline tests ────────────────────────────

    #[test]
    fn ansi_escape_stripping() {
        let input = "\x1B[31mred\x1B[0m \x1B[1mbold\x1B[22m";
        assert_eq!(strip_ansi_escapes(input), "red bold");
    }

    #[test]
    fn ansi_escape_no_op_for_clean_input() {
        let input = "hello world";
        assert_eq!(strip_ansi_escapes(input), input);
    }

    #[test]
    fn json_array_preview() {
        let input = r#"[{"name": "alice", "age": 30}, {"name": "bob", "age": 25}]"#;
        let result = try_json_preview(input);
        assert!(result.is_some(), "should detect JSON array");
        let output = result.unwrap();
        assert!(output.contains("2 items"), "should show item count");
        assert!(
            output.contains("name: string"),
            "should infer string schema"
        );
        assert!(output.contains("age: int"), "should infer int schema");
    }

    #[test]
    fn json_object_preview() {
        let input = r#"{"status": "ok", "count": 42}"#;
        let result = try_json_preview(input);
        assert!(result.is_some(), "should detect JSON object");
        let output = result.unwrap();
        assert!(output.contains("2 fields"), "should show field count");
        assert!(output.contains("status"), "should show field name");
        assert!(output.contains("count"), "should show field name");
    }

    #[test]
    fn non_json_passes_through() {
        let input = "hello world\nthis is not json";
        let result = try_json_preview(input);
        assert!(result.is_none(), "should not detect JSON");
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
    fn cargo_build_compiling_lines_stripped() {
        let input = "Compiling foo v1.0.0 (/tmp)\nCompiling bar v2.0.0 (/tmp)\n   Compiling baz v3.0.0 (/tmp)\nerror[E0425]: cannot find value\n\nFor more information about this error, try `rustc --explain E0425`.\nerror: could not compile `foo` due to 1 previous error";
        let result = process_shell_output("cargo build", input, "", 1, Duration::ZERO);
        // Compiling lines should be stripped by cargo build profile
        assert!(
            !result.contains("Compiling foo"),
            "compiling lines should be stripped"
        );
        // Error info should be preserved
        assert!(result.contains("error[E0425]"), "error info preserved");
        assert!(
            result.contains("could not compile"),
            "build failure preserved"
        );
    }

    #[test]
    fn cargo_check_short_circuit_on_success() {
        let input = "    Checking foo v1.0.0\n    Checking bar v2.0.0\n    warning: unused import\n\nwarning: 1 warning emitted\n\n    Finished `dev` profile [unoptimized] target\n";
        let result = process_shell_output("cargo check", input, "", 0, Duration::ZERO);
        // No "0 errors" pattern, so it should go through the normal pipe
        assert!(
            !result.contains("Checking"),
            "checking lines should be stripped"
        );
    }

    #[test]
    fn long_lines_truncated() {
        let long = "a".repeat(500);
        let result = truncate_line_width(&long, 100);
        assert!(result.len() < long.len() + 100, "should truncate");
        assert!(
            result.contains("more chars on this line"),
            "should show continuation marker on separate line"
        );
        // Verify the original line boundary is preserved (no mid-line truncation)
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(
            lines.len(),
            2,
            "original truncated line + continuation marker"
        );
        assert_eq!(
            lines[0].len(),
            100,
            "first line should be exactly max_chars"
        );
        assert!(
            !lines[0].contains("..."),
            "first line should not contain truncation marker"
        );
    }

    #[test]
    fn short_lines_preserved() {
        let input = "hello\nworld";
        let result = truncate_line_width(input, 500);
        assert_eq!(result, input, "short lines should pass through");
    }

    #[test]
    fn spill_writes_file_for_large_output() {
        let large = "x".repeat(10_000);
        let result = try_spill_to_file(large, 5_000);
        assert!(
            result.contains("[Output saved to"),
            "should contain spill path"
        );
        assert!(
            result.contains("[view with: read "),
            "should contain actionable read hint"
        );
        assert!(result.contains("10000 bytes"), "should mention byte count");
        // Ensure the spill file was actually written
        assert!(
            std::fs::read_dir(std::env::temp_dir().join(".agent")).is_ok(),
            "spill dir should exist"
        );
    }

    #[test]
    fn spill_truncates_multi_line_large_output() {
        // Many lines totalling well over 5K chars should produce head+tail preview
        let lines: Vec<String> = (0..800).map(|i| format!("line_{i:04}")).collect();
        let large = lines.join("\n");
        let large_len = large.len();
        assert!(
            large_len > 5_000,
            "test data {large_len} must exceed spill threshold",
        );
        let result = try_spill_to_file(large, 5_000);
        assert!(
            result.contains("[Output saved to"),
            "should contain spill path"
        );
        assert!(
            result.contains("[view with: read "),
            "should contain actionable read hint"
        );
        // Preview should have head (first lines) and tail (last lines)
        assert!(
            result.contains("line_0000"),
            "should show first line {result:?}"
        );
        assert!(result.contains("line_0799"), "should show last line");
        assert!(
            result.len() < large_len,
            "inline preview should be truncated"
        );
    }

    #[test]
    fn spill_returns_short_output_as_is() {
        let short = "hello".to_string();
        let result = try_spill_to_file(short.clone(), 5_000);
        assert_eq!(result, short, "short output should pass through unchanged");
    }

    #[test]
    fn compress_shell_output_pipeline_full() {
        let input = "Compiling foo v1.0.0 (/tmp)\nCompiling bar v2.0.0 (/tmp)\nresult: ok\nline1\nline2\nline3\nline3\nline3\nline3\nline3\nline3\nline3\n";
        let result = process_shell_output("unknown", input, "", 0, Duration::ZERO);
        // Input is pre-stripped by the caller — verify no ANSI contamination
        assert!(
            !result.contains("\x1B["),
            "ANSI escapes should be stripped (input pre-stripped)"
        );
        // Generic fallback doesn't strip cargo lines — they remain (only profiled cargo commands strip them)
        assert!(
            result.contains("Compiling"),
            "generic fallback preserves cargo lines"
        );
        // Consecutive lines deduped
        assert!(
            result.contains("[repeated"),
            "repeated lines should be collapsed"
        );
        // Original data preserved
        assert!(
            result.contains("result: ok"),
            "non-pattern content preserved"
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
        // SAFETY: ENV_LOCK serializes all env writes in this module.
        let _guard = env_lock().lock().unwrap();

        // SAFETY: ENV_LOCK serializes all env writes in this module.
        unsafe {
            std::env::set_var("CARGO_HOME", "/custom/cargo");
        }
        let path = resolved_shell_path();
        // SAFETY: ENV_LOCK serializes all env writes in this module.
        unsafe {
            std::env::remove_var("CARGO_HOME");
        }

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

        // SAFETY: ENV_LOCK serializes all env writes in this module.
        let _guard = env_lock().lock().unwrap();

        let default_cargo_home = dirs.home_dir().join(".cargo").to_string_lossy().to_string();
        // SAFETY: ENV_LOCK serializes all env writes in this module.
        unsafe {
            std::env::set_var("CARGO_HOME", &default_cargo_home);
        }
        let path = resolved_shell_path();
        // SAFETY: ENV_LOCK serializes all env writes in this module.
        unsafe {
            std::env::remove_var("CARGO_HOME");
        }

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

    /// 5+ consecutive blank lines should collapse to 2, not produce `[repeated]` markers.
    #[test]
    fn collapse_blank_lines_then_consecutive_no_marker() {
        let input = "a\n\n\n\n\n\nb"; // 6 blank lines between a and b
        let result = collapse_blank_lines(input);
        let result = collapse_consecutive_lines(&result);
        assert_eq!(
            result, "a\n\n\nb",
            "6 blank lines → 2 blanks, no [repeated] marker"
        );
        assert!(
            !result.contains("[repeated"),
            "should not contain repeated marker for blank lines"
        );
    }

    /// 5+ identical non-blank lines should still collapse with `[repeated]`.
    #[test]
    fn collapse_consecutive_non_blank_still_collapses() {
        let input = "x\nx\nx\nx\nx\nx"; // 6 identical non-blank lines
        let result = collapse_blank_lines(input); // should be a no-op
        let result = collapse_consecutive_lines(&result);
        assert!(
            result.contains("[repeated 6 times]"),
            "6 identical non-blank lines should produce [repeated] marker"
        );
        assert!(
            !result.contains("x\nx\nx\nx\nx\nx"),
            "should not keep individual lines"
        );
    }

    #[test]
    fn cargo_test_failure_block_capture() {
        let output = "\n\
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
        ";
        let result = filter_cargo_test_output(output, 1);
        assert!(!result.contains("Compiling"), "compiling stripped");
        assert!(!result.contains("test1 ... ok"), "passing tests stripped");
        assert!(result.contains("test2 ... FAILED"), "failure preserved");
        assert!(
            result.contains("assertion failed"),
            "panic message preserved"
        );
        assert!(result.contains("test result:"), "summary preserved");
    }

    #[test]
    fn cargo_test_all_pass_returns_summary() {
        let output = "\
            Compiling foo v1.0.0\n\
            Checking bar v2.0.0\n\
            test test1 ... ok\n\
            test test2 ... ok\n\
            \n\
            test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n\
        ";
        let result = filter_cargo_test_output(output, 0);
        assert!(!result.contains("Compiling"), "compiling stripped");
        assert!(!result.contains("Checking"), "checking stripped");
        assert!(!result.contains("test1 ... ok"), "passing stripped");
        assert!(!result.contains("test2 ... ok"), "passing stripped");
        assert!(result.contains("test result:"), "summary preserved");
    }

    #[test]
    fn cargo_test_compile_error_fallback() {
        let output = "\
            Compiling foo v1.0.0\n\
            error[E0425]: cannot find value `bar` in this scope\n\
             --> src/lib.rs:1:5\n\
            \n\
            error: could not compile `foo` due to 1 previous error\n\
        ";
        let result = filter_cargo_test_output(output, 1);
        assert!(!result.contains("Compiling"), "compiling stripped");
        assert!(result.contains("error[E0425]"), "error preserved");
        assert!(
            result.contains("could not compile"),
            "build error preserved"
        );
    }

    #[test]
    fn cargo_test_running_preserved() {
        // `Running unittests src/lib.rs` is useful context and must NOT be stripped.
        // Use a failure scenario so the function returns the full output (not just summary).
        let output = "\
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
        ";
        let result = filter_cargo_test_output(output, 1);
        assert!(!result.contains("Compiling"), "compiling stripped");
        assert!(
            result.contains("Running unittests"),
            "Running preserved in test output (not cargo noise)"
        );
        assert!(result.contains("test2 ... FAILED"), "failure preserved");
        assert!(result.contains("test result:"), "summary preserved");
    }

    #[test]
    fn git_diff_no_changes() {
        let result = process_shell_output("git diff", "", "", 0, Duration::ZERO);
        assert!(result.contains("no changes"), "short-circuit on empty diff");
    }

    #[test]
    fn docker_build_ok_short_circuit() {
        let input = "Step 1/3 : FROM alpine\n ---> abc123\nStep 2/3 : RUN echo hi\n ---> Using cache\nStep 3/3 : CMD [\"sh\"]\n ---> def456\nSuccessfully built abc123\nSuccessfully tagged myimage:latest\n";
        let result =
            process_shell_output("docker build -t myimage .", input, "", 0, Duration::ZERO);
        assert!(
            result.contains("[docker"),
            "docker build should short-circuit to ok message"
        );
    }

    #[test]
    fn git_log_filter() {
        let input = "commit abc123\nAuthor: test\nDate:   Mon Jan 1\n\n    initial commit\n\ncommit def456\nAuthor: test\nDate:   Tue Jan 2\n\n    second commit\n\n";
        let result = process_shell_output("git log --oneline", input, "", 0, Duration::ZERO);
        // git log profile has head_lines: 20, max_lines: 50 — with small output it passes through
        assert!(result.contains("commit"), "git log content preserved");
        // No strip patterns beyond blank lines, so should be mostly preserved
        assert!(result.contains("Author"), "Author field preserved");
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
    fn select_profile_cargo_test_with_flags_triggers_state_machine() {
        // Regression: flags before 'test' used to bypass the cargo test state machine.
        // With empty output and exit code 0, the state machine produces [cargo test: ok].
        let result = process_shell_output("cargo --release test", "", "", 0, Duration::ZERO);
        assert_eq!(result.trim(), "[cargo test: ok]");
    }

    #[test]
    fn select_profile_chained_cargo_build_strips_compiling() {
        let output =
            "   Compiling foo v1.0.0\n   Compiling bar v2.0.0\nerror[E0425]: cannot find value\n";
        let result =
            process_shell_output("cd project && cargo build", output, "", 1, Duration::ZERO);
        assert!(
            !result.contains("Compiling"),
            "chained cargo build: compiling lines stripped"
        );
        assert!(
            result.contains("error[E0425]"),
            "chained cargo build: errors preserved"
        );
    }

    #[test]
    fn select_profile_absolute_cargo_strips_compiling() {
        let output = "   Compiling foo v1.0.0\nwarning: unused import\n";
        let result =
            process_shell_output("/usr/local/bin/cargo check", output, "", 0, Duration::ZERO);
        assert!(
            !result.contains("Compiling"),
            "absolute cargo: compiling lines stripped"
        );
    }

    #[test]
    fn select_profile_git_with_c_flag_triggers_git_diff() {
        let result = process_shell_output("git -C /repo diff", "", "", 0, Duration::ZERO);
        // git diff profile short-circuits on empty
        assert_eq!(result.trim(), "[git diff: no changes]");
    }

    #[test]
    fn select_profile_fallback_for_unknown_uses_generic() {
        let output = "some\nrandom\noutput\n";
        let result =
            process_shell_output("some_obscure_tool --flag", output, "", 0, Duration::ZERO);
        assert!(result.contains("some"), "generic: output passes through");
        assert!(result.contains("output"), "generic: output passes through");
    }

    #[test]
    fn chained_command_matches_correct_profile() {
        // Should match pnpm install (the first matching segment)
        let output = "Already up to date\nsome output\n";
        let result = process_shell_output(
            "cd frontend && pnpm install && pnpm build",
            output,
            "",
            0,
            Duration::ZERO,
        );
        // pnpm install strips "Already up to date" — that profile matched
        assert!(
            !result.contains("Already up to date"),
            "pnpm install profile matched and stripped noise line"
        );
    }

    #[test]
    fn empty_command_uses_fallback() {
        let result = process_shell_output("", "hello world", "", 0, Duration::ZERO);
        assert!(result.contains("hello"));
    }

    #[test]
    fn only_shell_builtins_use_fallback() {
        let result = process_shell_output("cd .. && cd /tmp", "some output", "", 0, Duration::ZERO);
        assert!(
            result.contains("some output"),
            "builtins-only falls through to generic"
        );
    }

    #[test]
    fn chained_cargo_test_uses_state_machine() {
        // Uses the cargo test state machine, not the generic profile
        let output = "Compiling foo v1.0.0\ntest test1 ... ok\ntest test2 ... FAILED\n\nfailures:\n\n---- test2 stdout ----\npanic!\n\nfailures:\n    test2\n\ntest result: FAILED. 1 passed; 1 failed\n";
        let result =
            process_shell_output("cd project && cargo test", output, "", 1, Duration::ZERO);
        assert!(
            !result.contains("Compiling"),
            "cargo test: compiling stripped"
        );
        assert!(!result.contains("test1 ... ok"), "passing tests stripped");
        assert!(
            result.contains("test2 ... FAILED"),
            "failures preserved in chained cargo test"
        );
    }

    #[test]
    fn chained_git_log_preserves_content() {
        let input = "commit abc123\nAuthor: test\nDate:   Mon Jan 1\n\n    initial commit\n";
        let result =
            process_shell_output("cd repo && git log --oneline", input, "", 0, Duration::ZERO);
        // Should match git log profile (after git global flags in canonical form:
        // "git log --oneline" → canonical "git log", matches ^git\s+log\b)
        assert!(result.contains("commit"), "git log content preserved");
        assert!(result.contains("Author"), "Author field preserved");
    }

    // ── New feature tests ──────────────────────────────────────────

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
        assert!(lines <= 50, "df should cap at ~21 lines, got {lines}");
        assert!(lines >= 19, "df should have around 20 lines, got {lines}");
    }

    #[test]
    fn profile_du_strips_blank_lines() {
        let input = "1.0K\t./file1\n\n2.0K\t./file2\n\n\n3.0K\t./file3";
        let result = process_shell_output("du -sh", input, "", 0, Duration::ZERO);
        assert!(
            !result.contains("\n\n"),
            "should not have consecutive blank lines"
        );
    }

    #[test]
    fn profile_make_strips_directory_noise() {
        let input = "make[1]: Entering directory `/tmp'\nmake[1]: Leaving directory `/tmp'\ncc -c file.c\nNothing to be done";
        let result = process_shell_output("make", input, "", 0, Duration::ZERO);
        assert!(
            !result.contains("Entering directory"),
            "make noise stripped"
        );
        assert!(
            !result.contains("Nothing to be done"),
            "'nothing to be done' stripped"
        );
    }

    #[test]
    fn profile_rsync_short_circuits_on_success() {
        let input = "building file list ... done\nsent 100 bytes  received 50 bytes\n\ntotal size is 98765  speedup is 658.43\n";
        let result = process_shell_output("rsync -avz source/ dest/", input, "", 0, Duration::ZERO);
        assert_eq!(
            result.trim(),
            "ok (synced)",
            "rsync should short-circuit on 'total size is'"
        );
    }

    #[test]
    fn profile_cargo_build_strips_noise() {
        let input =
            "   Compiling foo v1.0.0\n   Compiling bar v2.0.0\n    Finished dev [unoptimized]\n";
        let result = process_shell_output("cargo build", input, "", 0, Duration::ZERO);
        assert!(
            !result.contains("Compiling"),
            "cargo build strips Compiling lines"
        );
        assert!(
            !result.contains("Finished"),
            "cargo build strips Finished lines"
        );
    }

    #[test]
    fn profile_tsc_on_empty_returns_ok() {
        let result = process_shell_output("tsc --noEmit", "", "", 0, Duration::ZERO);
        assert_eq!(result.trim(), "[tsc: ok] (0.0s)");
    }

    #[test]
    fn profile_docker_strips_build_steps() {
        let input = "Step 1/10 : FROM node:18\nStep 2/10 : WORKDIR /app\n ---> Using cache\nSuccessfully built abc123\nSuccessfully tagged myapp:latest\n";
        let result = process_shell_output("docker build -t myapp .", input, "", 0, Duration::ZERO);
        assert!(!result.contains("Step "), "docker strips step lines");
        assert!(
            result.contains("[docker build: ok]"),
            "docker short-circuits on success"
        );
    }

    #[test]
    fn profile_pytest_strips_collected() {
        let input = "============================= test session starts ==============================\ncollected 5 items\n\n.test..\n\n============================== 5 passed ==============================\n";
        let result = process_shell_output("pytest", input, "", 0, Duration::ZERO);
        assert!(
            !result.contains("collected"),
            "pytest strips collected count"
        );
    }

    #[test]
    fn profile_pytest_python_m_falls_through_to_generic() {
        // Regression test: python -m pytest must NOT match the pytest profile
        // because canonical_command strips the `-m` flag, producing
        // "python pytest" which doesn't start with "^pytest\b".
        // The "collected" line should be preserved (GEN_FALLBACK doesn't strip it).
        let input = "============================= test session starts ==============================\ncollected 5 items\n\n.test..\n\n============================== 5 passed ==============================\n";
        let result = process_shell_output("python -m pytest tests/", input, "", 0, Duration::ZERO);
        assert!(
            result.contains("collected"),
            "python -m pytest falls through to GEN_FALLBACK (collected preserved)"
        );
    }

    #[test]
    fn profile_pytest_poetry_run_falls_through_to_generic() {
        // Regression test: poetry run pytest must NOT match the pytest profile
        // because canonical_command treats `run` as poetry's subcommand,
        // producing "poetry run" which doesn't match "^pytest\b".
        let input = "============================= test session starts ==============================\ncollected 5 items\n\n.test..\n\n============================== 5 passed ==============================\n";
        let result = process_shell_output("poetry run pytest tests/", input, "", 0, Duration::ZERO);
        assert!(
            result.contains("collected"),
            "poetry run pytest falls through to GEN_FALLBACK (collected preserved)"
        );
    }

    #[test]
    fn profile_keep_stderr_warnings_on_success() {
        let stderr = "warning: unused import: `std::fs`\n  --> src/main.rs:1:5\n";
        let result = process_shell_output(
            "cargo build",
            "   Compiling foo v1.0.0\n    Finished\n",
            stderr,
            0,
            Duration::ZERO,
        );
        assert!(
            result.contains("warning:"),
            "cargo build warnings shown on success"
        );
    }

    #[test]
    fn compact_ls_empty_directory() {
        let input = "total 0\ndrwxr-xr-x  2 user  group  64 May 21 10:00 .\ndrwxr-xr-x  3 user  group  96 May 21 10:00 ..\n";
        let result = process_shell_output("ls -la", input, "", 0, Duration::ZERO);
        assert_eq!(
            result.trim(),
            "(empty)",
            "empty ls output should show (empty)"
        );
    }

    #[test]
    fn compact_ls_mixed_files_and_dirs() {
        let input = "total 32\ndrwxr-xr-x  5 user  group   160 May 21 10:00 .\ndrwxr-xr-x  3 user  group    96 May 21 10:00 ..\n-rw-r--r--  1 user  group  2048 May 21 10:00 main.rs\n-rw-r--r--  1 user  group  4096 May 21 10:00 lib.rs\ndrwxr-xr-x  2 user  group    64 May 21 10:00 src\nlrwxr-xr-x  1 user  group     5 May 21 10:00 link -> target\n";
        let result = process_shell_output("ls -la", input, "", 0, Duration::ZERO);
        assert!(result.contains("src/"), "directory should end with slash");
        assert!(result.contains("main.rs"), "file name preserved");
        assert!(result.contains("lib.rs"), "file name preserved");
        assert!(result.contains("Summary:"), "should have summary");
        assert!(
            !result.contains("link -> target"),
            "symlink target stripped"
        );
    }

    #[test]
    fn compact_ls_dotless_files() {
        let input = "total 16\n-rw-r--r--  1 user  group  1024 May 21 10:00 Makefile\n-rw-r--r--  1 user  group  2048 May 21 10:00 README\n-rw-r--r--  1 user  group   512 May 21 10:00 .gitignore\n-rw-r--r--  1 user  group  1024 May 21 10:00 main.rs\n";
        let result = process_shell_output("ls -la", input, "", 0, Duration::ZERO);
        assert!(result.contains("Makefile"), "dotless file preserved");
        assert!(result.contains("README"), "dotless file preserved");
        assert!(
            !result.contains(".Makefile"),
            "dotless file should not get fake extension"
        );
        assert!(
            !result.contains(".README"),
            "dotless file should not get fake extension"
        );
        // The summary should classify these as "no ext", not as fake extensions
        assert!(
            result.contains("no ext"),
            "summary should include 'no ext' for dotless files"
        );
        // .gitignore is a dotfile — its stem is empty, extension is "gitignore"
        assert!(
            result.contains(".rs"),
            "main.rs should be classified as .rs"
        );
    }

    #[test]
    fn compact_ls_plain_ls_passes_through() {
        // Plain `ls` output (no `-l` flag) has no "total N" header.
        // compact_ls should pass it through unchanged instead of returning "(empty)".
        let input = "Cargo.toml\nCargo.lock\nsrc\ntarget\nREADME.md\n";
        let result = process_shell_output("ls", input, "", 0, Duration::ZERO);
        assert!(
            result.contains("Cargo.toml"),
            "plain ls output should show filenames unchanged"
        );
        assert!(
            result.contains("src"),
            "plain ls output should show filenames unchanged"
        );
        assert!(
            !result.contains("(empty)"),
            "plain ls output should NOT show (empty)"
        );
        assert!(
            !result.contains("Summary:"),
            "plain ls output should NOT be compacted — no Summary header"
        );
    }

    #[test]
    fn chained_ls_skips_compact_ls() {
        // Chained `ls -l && echo done` should NOT go through compact_ls —
        // the standalone_only flag ensures the transform is skipped for
        // chained commands, preserving output from later segments.
        let input = "total 8\n-rw-r--r--  1 user  group  1024 May 21 10:00 foo\n-rw-r--r--  1 user  group  2048 May 21 10:00 bar\ndone\n";
        let result = process_shell_output("ls -l && echo done", input, "", 0, Duration::ZERO);
        assert!(
            result.contains("done"),
            "chained ls: later segments' output preserved"
        );
        assert!(
            !result.contains("Summary:"),
            "chained ls: compact_ls should not be applied"
        );
    }

    #[test]
    fn save_raw_output_if_large_skips_small_output() {
        let result = save_raw_output_if_large(b"hello", b"", "echo hello");
        assert!(result.is_none(), "should skip saving for small output");
    }

    #[test]
    fn chained_ls_with_pipe_skips_compact_ls() {
        // Piped `ls -l | head -5` should NOT go through compact_ls —
        // the standalone_only flag causes select_profile to skip the ls
        // profile for chained commands, falling through to GEN_FALLBACK.
        let input = "total 8\n-rw-r--r--  1 user  group  1024 May 21 10:00 foo\n-rw-r--r--  1 user  group  2048 May 21 10:00 bar\n";
        let result = process_shell_output("ls -l | head -5", input, "", 0, Duration::ZERO);
        assert!(
            !result.contains("Summary:"),
            "piped ls: compact_ls should not be applied"
        );
        assert!(
            result.contains("total 8"),
            "piped ls: raw -l format should be preserved"
        );
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
    fn profile_on_empty_shows_timing() {
        let result = process_shell_output("tsc --noEmit", "", "", 0, Duration::from_secs_f64(3.2));
        // The on_empty message should include timing
        assert!(
            result.contains("(3.2s)"),
            "timing should appear in on_empty message"
        );
    }

    #[test]
    fn profile_gh_strips_noise() {
        let input = "  \n - some detail\nwarning: consider updating gh\n✓ Created pull request\n";
        let result = process_shell_output("gh pr create --fill", input, "", 0, Duration::ZERO);
        assert!(!result.contains("warning:"), "gh: warning stripped");
        assert!(result.contains("[gh: ok]"), "gh: short-circuit on success");
    }

    #[test]
    fn profile_terraform_short_circuits() {
        let input = "data.aws_region.current: Refreshing state...\nNo changes. Your infrastructure matches the configuration.\n";
        let result = process_shell_output("terraform plan", input, "", 0, Duration::ZERO);
        assert!(
            result.contains("[terraform: no changes]"),
            "terraform: short-circuits on 'No changes'"
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

    // ── apply_line_truncation unit tests ──────────────────────────────

    #[test]
    fn truncate_no_config_passthrough() {
        let p = Profile::new("passthrough");
        let output = "line1\nline2\nline3";
        let (result, pre) = apply_line_truncation(output, &p);
        assert_eq!(result, output);
        assert_eq!(pre, None);
    }

    #[test]
    fn truncate_head_tail_only_small_output_no_sandwich() {
        // Small output (< SPILL_THRESHOLD_BYTES) should not trigger sandwich
        let p = Profile::new("test").head(2).tail(2);
        let output = "line1\nline2\nline3\nline4\nline5";
        let (result, pre) = apply_line_truncation(output, &p);
        assert_eq!(result, output, "small output passes through");
        assert_eq!(pre, None, "no pre-truncation for small output");
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
    }

    #[test]
    fn truncate_max_only_caps_at_limit() {
        let p = Profile::new("test").max(3);
        let output = "a\nb\nc\nd\ne";
        let (result, pre) = apply_line_truncation(output, &p);
        assert_eq!(pre, None, "no pre-truncation for max-only");
        assert_eq!(
            result.lines().count(),
            4, // 3 lines + 1 truncation marker
            "should have max+1 lines (3 data + marker)"
        );
        assert!(result.contains("... (2 lines truncated)"));
    }

    #[test]
    fn truncate_max_only_fits_no_truncation() {
        let p = Profile::new("test").max(10);
        let output = "a\nb\nc";
        let (result, pre) = apply_line_truncation(output, &p);
        assert_eq!(result, output, "fits within max, passthrough");
        assert_eq!(pre, None);
    }

    #[test]
    fn truncate_head_tail_plus_max_byte_threshold_exceeded() {
        let p = Profile::new("test").head(2).tail(2).max(100);
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
        assert!(pre.is_some(), "should capture pre-truncation");
        // head+tail+1 = 5 <= max=100, so defensive cap is no-op
        assert!(
            result.contains("... (96 lines omitted)"),
            "should have sandwich omission marker"
        );
        assert!(
            !result.contains("lines truncated"),
            "defensive cap should not fire when head+tail+1 <= max"
        );
    }

    #[test]
    fn truncate_head_tail_plus_max_byte_threshold_not_exceeded() {
        let p = Profile::new("test").head(2).tail(2).max(3);
        // Small output: byte threshold NOT exceeded, so head/tail skipped,
        // but max cap should still apply.
        let output = "a\nb\nc\nd\ne";
        let (result, pre) = apply_line_truncation(output, &p);
        assert_eq!(pre, None, "no pre-truncation for small output");
        assert_eq!(
            result.lines().count(),
            4, // 3 lines + 1 truncation marker
            "max cap should apply"
        );
        assert!(result.contains("... (2 lines truncated)"));
    }

    #[test]
    fn truncate_head_tail_fits_when_under_limit() {
        let p = Profile::new("test").head(5).tail(3);
        // Only 4 lines total — fewer than head+tail (8), so no truncation
        let output = "a\nb\nc\nd";
        let (result, pre) = apply_line_truncation(output, &p);
        assert_eq!(result, output, "not enough lines to truncate");
        assert_eq!(pre, None);
    }
}
