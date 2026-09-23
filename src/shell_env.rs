//! The owner's own shell environment.
//!
//! An agent's shell command must run in the environment the owner's own
//! terminal would give it, not in one the product fabricates. Where that
//! environment comes from is platform-specific:
//!
//! - macOS: the shell the account records for itself is started the way
//!   Terminal.app starts it — a login AND interactive shell — and asked to run
//!   this binary's hidden `__env-dump` subcommand, so the values are the ones
//!   that shell's startup files actually export.
//! - Every other unix: the same, with an interactive (not login) shell — the
//!   convention those systems' terminals use.
//! - Windows: there is no shell startup to replay, so the OS's own assembly for
//!   a program is asked for directly.
//!
//! The read is a background task ([`run_reader_loop`]): it re-reads every
//! [`READ_INTERVAL`] and never blocks a command — a command reads the cached
//! [`snapshot`] and nothing else. Until a read has succeeded, and whenever one
//! cannot be obtained, the reduced environment in [`crate::tools::shell`] is
//! what the command runs with.
//!
//! Running the owner's shell startup files is deliberate: it is the only way to
//! obtain the environment they produce (a version manager's shim on `PATH`, a
//! toolchain's root, a compiler's own variable). Nothing is filtered out of the
//! result — the owner's secrets and tokens reach the agent's commands by design,
//! because those commands are the owner's own work. No value this module reads
//! is ever logged or stored anywhere in the product: the reader logs counts,
//! shape and timing only, and the snapshot lives in memory for the life of the
//! process.

use std::ffi::OsString;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwapOption;

#[cfg(unix)]
use std::ffi::OsStr;
#[cfg(unix)]
use std::path::{Path, PathBuf};

// ── Cadence ───────────────────────────────────────────────────────────────

/// How long the reader waits between reads while reads keep succeeding. The
/// environment a startup file produces is stable, so this is a re-check rather
/// than a poll.
const READ_INTERVAL: Duration = Duration::from_mins(10);

/// Ceiling on the retry backoff: [`READ_INTERVAL`] doubles per consecutive
/// failure up to this, so a shell that is gone for good costs one bounded read
/// every 80 minutes instead of a spin.
const READ_RETRY_MAX: Duration = Duration::from_mins(80);

/// Bound on one read's capture: the dumper's dump has to arrive inside it, and the
/// whole process group is killed when it does not. Neither of the two waits around
/// this one is what it covers — the shell's own exit belongs to the reaper, which
/// bounds its wait by this same value, and the shell's stderr is drained by a task
/// that lives only as long as the reaper does. A startup file can be slow (network
/// mounts, a version manager's first run), so this is generous rather than tight.
#[cfg(unix)]
const READ_TIMEOUT: Duration = Duration::from_secs(15);

/// Cap on the bytes collected from the dumper's stdout. A real environment is a
/// few kilobytes; the cap only keeps a runaway startup file from growing the
/// reader's buffer without bound.
#[cfg(unix)]
const DUMP_STDOUT_CAP: usize = 1024 * 1024;

/// Size of one read from a pipe.
///
/// Deliberately modest: each in-flight read future carries its own chunk inline,
/// and the whole read's futures (the dump's and the stderr drain's) must stay
/// well under the size clippy flags as a large future.
#[cfg(unix)]
const READ_CHUNK: usize = 4096;

/// The shell basenames [`shell_recipe`] knows how to start. A shell outside
/// this list is refused rather than guessed at — the flags that make a shell
/// read its startup files are shell-specific, and guessing them would run the
/// wrong files or none at all. A shell whose own convention the recipe cannot
/// express (csh/tcsh, whose login-ness is not a flag) belongs outside the list
/// on purpose: its owner keeps the fallback of [`crate::tools::shell`] rather
/// than a recipe that would run the wrong files.
#[cfg(unix)]
const KNOWN_SHELLS: &[&str] = &[
    "sh", "bash", "dash", "ash", "zsh", "fish", "ksh", "ksh93", "mksh", "pdksh",
];

// ── Failure ───────────────────────────────────────────────────────────────

/// Why a read produced no environment.
///
/// The variants carry basenames and short I/O descriptions only, never an
/// environment value: a failure is logged, and a value must never reach a log.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ReadFailure {
    /// Neither the account's recorded shell nor `$SHELL` yielded a path.
    #[cfg(unix)]
    NoShell,
    /// The recorded shell is not one [`shell_recipe`] knows how to start. The
    /// payload is the basename (which may be a locally installed program name,
    /// so nothing more than the name is kept).
    #[cfg(unix)]
    UnknownShell(String),
    /// The shell could not be started, or its exit could not be collected. The
    /// payload is a short I/O description.
    #[cfg(unix)]
    Spawn(String),
    /// The read did not finish within its bound. Together with the kills below,
    /// this is what keeps a stray startup-file process from holding the reader.
    #[cfg(unix)]
    Timeout,
    /// The shell started but produced no usable dump: a marker or the entries
    /// were missing, or an entry was mangled by another writer of the pipe. The
    /// payload is a short description in words.
    #[cfg(unix)]
    NoDump(String),
    /// The OS could not assemble the owner's environment (Windows, where there
    /// is no shell read at all).
    #[cfg(windows)]
    AssemblyFailed(String),
}

impl std::fmt::Display for ReadFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(unix)]
            Self::NoShell => write!(f, "no shell recorded for the account"),
            #[cfg(unix)]
            Self::UnknownShell(name) => write!(f, "unrecognized shell {name}"),
            #[cfg(unix)]
            Self::Spawn(detail) => write!(f, "could not start the shell ({detail})"),
            #[cfg(unix)]
            Self::Timeout => write!(f, "the shell did not finish in time"),
            #[cfg(unix)]
            Self::NoDump(detail) => write!(f, "the shell produced no environment dump ({detail})"),
            #[cfg(windows)]
            Self::AssemblyFailed(detail) => {
                write!(f, "the OS could not assemble the environment ({detail})")
            }
        }
    }
}

// ── Snapshot ──────────────────────────────────────────────────────────────

/// A successfully read environment, held as `NAME=VALUE` pairs.
///
/// Deliberately not `Debug`: the type exists to be handed to a command, and a
/// derived `Debug` would let one stray `{:?}` print the owner's secrets into a
/// log.
pub(crate) struct OwnerEnv {
    vars: Vec<(OsString, OsString)>,
}

impl OwnerEnv {
    /// Wrap the pairs a read recovered, in the order it recovered them.
    #[must_use]
    pub(crate) fn new(vars: Vec<(OsString, OsString)>) -> Self {
        Self { vars }
    }

    /// The pairs.
    #[must_use]
    pub(crate) fn vars(&self) -> &[(OsString, OsString)] {
        &self.vars
    }
}

/// The last successfully read environment, empty until one succeeds.
///
/// A lock-free holder on purpose: [`snapshot`] runs on the path of every
/// command, so it must never block, and a failed read must leave the previous
/// value untouched rather than clear it.
static SNAPSHOT: ArcSwapOption<OwnerEnv> = ArcSwapOption::const_empty();

/// The cached owner environment, `None` until a read has succeeded.
///
/// A cheap lock-free read — safe to call from `connect`-time code on a
/// command's path.
#[must_use]
pub(crate) fn snapshot() -> Option<Arc<OwnerEnv>> {
    SNAPSHOT.load_full()
}

/// Publish a freshly read environment; the next [`snapshot`] sees it.
fn publish(env: Arc<OwnerEnv>) {
    SNAPSHOT.store(Some(env));
}

/// Replace the cached environment from a test.
///
/// The only test writer: the production writer is the background reader, which
/// needs a real shell. Unix-only with its callers — the lanes that need a
/// snapshot drive a real shell's child, which only exists there.
#[cfg(all(test, unix))]
pub(crate) fn set_snapshot(env: Option<OwnerEnv>) {
    SNAPSHOT.store(env.map(Arc::new));
}

// ── Reader loop ───────────────────────────────────────────────────────────

/// Read the owner's environment now, then re-read it every [`READ_INTERVAL`],
/// until shutdown. Spawned as a background task; it never returns except on
/// shutdown.
///
/// A success is logged at INFO (the 8-hour INFO retention must never leave the
/// owner without a row — this is his only place to see whether his environment
/// was picked up), and carries counts, shape and timing only. A failure is
/// logged at WARN (which survives retention and is what the app's issue view
/// shows) with the consecutive-failure count, the next retry delay and the
/// failure's own short description. Neither line, and nothing else here, ever
/// carries a value out of the environment.
///
/// The regex-search parity verdict is re-measured after every read attempt (a
/// read that puts a new environment in place has already dropped the previous
/// verdict, in the step before it published) — an environment already measured
/// with a verdict is a no-op, and a run that produced none is attempted again by
/// the next read — and never before the first read: the read is the thing that has
/// to start early, and until a verdict for the environment in effect is published
/// the fast path is closed rather than assumed. When a refresh does measure, it
/// runs here on the reader's own path, so the period is `READ_INTERVAL` plus that
/// measurement — typically well under a second, and never on a command's path,
/// though a battery of ~40 rows against a pathologically slow program on the
/// owner's own `PATH` can stretch it by minutes, since each child is bounded only
/// by the battery's own 5 s deadline. The measurement's own line carries booleans,
/// counts and its probe's finding only.
pub async fn run_reader_loop() {
    let mut failures: u32 = 0;
    loop {
        if crate::shutdown::aborting() {
            return;
        }

        let started = Instant::now();
        #[cfg(unix)]
        let result = read_unix().await;
        #[cfg(windows)]
        let result = read_windows();

        match result {
            Ok(env) => {
                let names = env.vars().len();
                let path_entries = path_entry_count(env.vars());
                let elapsed_ms = started.elapsed().as_millis();
                // The verdict in effect was measured for the environment about to
                // be replaced, so it is dropped first: what commands will get is
                // computed once, here, and the same list is what the measurement
                // below is keyed on — no command is ever served the new environment
                // under the old verdict, and the gate cannot be filed under a pair
                // list a command does not actually get. The measurement itself
                // follows on the blocking pool.
                let pairs = crate::tools::shell::agent_env_pairs_from(&env);
                crate::tools::shell::grep_engine::invalidate_verdict_for(&pairs);
                publish(Arc::new(env));
                failures = 0;
                // How the served search stands under the environment just read: a
                // measured verdict on unix, where there is a real search to compare
                // against; on Windows asking is meaningless — its gate never
                // consults a verdict, the served path is the only search — so the
                // word says that instead of a `false` that would read as a closed
                // fast path.
                #[cfg(unix)]
                let search = if refresh_grep_parity(pairs).await {
                    "verdict measured for the environment in effect"
                } else {
                    "no verdict yet (the real search runs)"
                };
                #[cfg(windows)]
                let search = "not applicable (the served path is the only search)";
                tracing::info!(
                    names,
                    path_entries,
                    elapsed_ms,
                    search,
                    "owner environment read"
                );
            }
            Err(reason) => {
                failures = failures.saturating_add(1);
                // No publish here: a failed read leaves the last good snapshot
                // in place, so a command never loses an environment it had.
                let retry_secs = retry_delay(failures).as_secs();
                tracing::warn!(
                    consecutive_failures = failures,
                    retry_secs,
                    reason = %reason,
                    "owner environment read failed"
                );
                // The environment in effect is unchanged, so this measures only if
                // no verdict for it has been taken yet — the first read failed and
                // commands are still on the fallback. The list is the one commands
                // get now. On Windows there is nothing to measure, and the call
                // returns without a round trip.
                refresh_grep_parity(crate::tools::shell::agent_env_pairs()).await;
            }
        }

        if !crate::shutdown::sleep_or_shutdown_or_drain(retry_delay(failures)).await {
            return;
        }
    }
}

/// Re-measure the regex-search parity verdict if the environment an agent's
/// command runs under differs from the one the published verdict was measured
/// for.
///
/// The measurement is blocking — it runs the real search over a fixture — so it
/// rides the blocking pool. The loop awaits the pool's join, so the *async*
/// runtime is never blocked, but the reader's own next read does wait for the
/// measurement; nothing on an agent's command path does (`serve_allowed` reads
/// the published verdict lock-free). A shutdown that tears the pool down while
/// it runs is not fatal: the join result is dropped, and the loop's own abort
/// check ends it.
///
/// The verdict for the previous environment is dropped in the step that publishes
/// a new read (see [`crate::tools::shell::grep_engine::invalidate_verdict_for`]),
/// so the two never mix: while the battery runs, the fast path is closed and the
/// real search answers instead.
///
/// Returns whether a verdict for the environment in effect stands afterwards — an
/// environment already measured is a no-op, and a run that produced no verdict is
/// attempted again by the next read; the success arm records the flag on its INFO
/// line. `env` is the pairs that read produced (or, after a failed read, the ones
/// in effect), passed through so the reader derives them once.
///
/// The Windows arm below is the reader's own and the gate has the same one for
/// itself ([`refresh_if_environment_changed`] answers without measuring where there
/// is no real search): the gate's is the rule, this one keeps a failed read from
/// asking the pool for that `false`.
async fn refresh_grep_parity(env: Vec<(OsString, OsString)>) -> bool {
    if cfg!(windows) {
        return false;
    }
    tokio::task::spawn_blocking(move || {
        crate::tools::shell::grep_engine::refresh_if_environment_changed(&env)
    })
    .await
    .unwrap_or(false)
}

/// The wait before the next read after `failures` consecutive failures.
///
/// The counter resets to zero on success, so zero failures is [`READ_INTERVAL`]
/// itself; otherwise the interval doubles per consecutive failure — 10, 20, 40,
/// then 80 minutes — and stays at [`READ_RETRY_MAX`].
#[must_use]
fn retry_delay(failures: u32) -> Duration {
    let doublings = failures.saturating_sub(1).min(3);
    let secs = READ_INTERVAL.as_secs() << doublings;
    Duration::from_secs(secs.min(READ_RETRY_MAX.as_secs()))
}

/// The number of entries in a set's `PATH`, for the reader's log line. Only the
/// count is ever used — never the value. Case-insensitive because Windows
/// spells the variable `Path`.
#[must_use]
fn path_entry_count(vars: &[(OsString, OsString)]) -> usize {
    vars.iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("PATH"))
        .map_or(0, |(_, value)| {
            std::env::split_paths(value)
                .filter(|entry| !entry.as_os_str().is_empty())
                .count()
        })
}

// ── The dumper subcommand ─────────────────────────────────────────────────

/// The byte sequence opening a dump: `RS <nonce> US`.
#[must_use]
fn start_marker(nonce: &str) -> String {
    format!("\u{1e}{nonce}\u{1f}")
}

/// The byte sequence closing a dump: `RS <nonce> GS`.
#[must_use]
fn end_marker(nonce: &str) -> String {
    format!("\u{1e}{nonce}\u{1d}")
}

/// Body of the hidden `__env-dump` subcommand: write the environment this
/// process inherited to stdout in the dump protocol, then return `0`.
///
/// `args[0]` is the nonce the reader chose. The output is the start marker,
/// then every entry of [`std::env::vars_os`] as `NAME=VALUE` followed by a NUL,
/// then the end marker. On unix the raw `OsStr` bytes go out verbatim, so a
/// value containing a newline, an `=` or invalid UTF-8 survives exactly as the
/// shell exported it. On Windows the UTF-8 spelling is written — the Windows
/// read never runs this (it asks the OS directly), so the spelling there is only
/// for a manual inspection.
///
/// Write errors are ignored: a truncated dump reaches the reader as
/// [`ReadFailure::NoDump`] either way, and a closed stdout must not become a
/// panic in a process whose only job is to print its environment.
pub fn dump_environment(args: &[String]) -> i32 {
    use std::io::Write as _;

    let nonce = args.first().map_or("", String::as_str);
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let _ = out.write_all(start_marker(nonce).as_bytes());

    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt as _;
        for (name, value) in std::env::vars_os() {
            let _ = out.write_all(name.as_bytes());
            let _ = out.write_all(b"=");
            let _ = out.write_all(value.as_bytes());
            let _ = out.write_all(b"\0");
        }
    }
    #[cfg(not(unix))]
    {
        for (name, value) in std::env::vars_os() {
            let _ = out.write_all(name.to_string_lossy().as_bytes());
            let _ = out.write_all(b"=");
            let _ = out.write_all(value.to_string_lossy().as_bytes());
            let _ = out.write_all(b"\0");
        }
    }

    let _ = out.write_all(end_marker(nonce).as_bytes());
    let _ = out.flush();
    0
}

// ── Reading (unix) ────────────────────────────────────────────────────────

/// The flags the owner's shell is started with for a read, given its basename
/// and whether this is macOS.
///
/// Only the leading flags are returned; the caller appends `-c` and the dump
/// command. macOS starts the shell the way Terminal.app does — login AND
/// interactive — while every other unix starts it interactive only.
///
/// `None` for a basename outside [`KNOWN_SHELLS`]: the caller then fails the
/// read with [`ReadFailure::UnknownShell`] rather than running a shell with
/// flags it may not understand. Unix-only: the Windows read asks the OS, not a
/// shell.
#[cfg(unix)]
#[must_use]
fn shell_recipe(shell_basename: &str, macos: bool) -> Option<Vec<&'static str>> {
    if !KNOWN_SHELLS.contains(&shell_basename) {
        return None;
    }
    Some(if macos { vec!["-l", "-i"] } else { vec!["-i"] })
}

/// Parse a dumper's stdout into `(name, value)` pairs.
///
/// The protocol is the start marker, then `NAME=VALUE\0` per entry, then the end
/// marker. The LAST start marker wins, so a startup file's own output before it
/// is ignored rather than mistaken for the dump, and the dump ends at the FIRST
/// end marker after it, so whatever the shell (or a startup file's background
/// job) prints after the dump is ignored too.
///
/// Between the markers the bytes are split on NUL, and a chunk that is not
/// `NAME=VALUE` with a non-empty name is skipped. A write from another holder of
/// the pipe that lands between two entries glues to the next entry's name, and
/// the one shape that is caught is a name carrying a control byte — a stray
/// line's newline, which is what a shell's or a startup file's own output between
/// entries leaves. No shell exports such a name, so that dump is
/// [`ReadFailure::NoDump`] rather than a set that is the owner's minus whatever
/// the stray write swallowed: the fallback is better than an approximation. A
/// stray write that carries no control byte cannot be told from a name and is not
/// detected — the read is no more exact than the bytes it is handed.
///
/// A missing end marker, or no entries at all, is [`ReadFailure::NoDump`].
#[cfg(unix)]
fn parse_dump(bytes: &[u8], nonce: &str) -> Result<Vec<(OsString, OsString)>, ReadFailure> {
    use std::os::unix::ffi::OsStrExt as _;

    let start = start_marker(nonce);
    let end = end_marker(nonce);
    let at = bytes
        .windows(start.len())
        .rposition(|window| window == start.as_bytes())
        .ok_or_else(|| ReadFailure::NoDump("no start marker".into()))?;
    let body = &bytes[at + start.len()..];
    let dump_end = body
        .windows(end.len())
        .position(|window| window == end.as_bytes())
        .ok_or_else(|| ReadFailure::NoDump("no end marker".into()))?;

    let mut vars = Vec::new();
    for chunk in body[..dump_end].split(|&byte| byte == 0) {
        let Some(eq) = chunk.iter().position(|&byte| byte == b'=') else {
            continue;
        };
        let name = &chunk[..eq];
        if name.is_empty() {
            continue;
        }
        if name.iter().any(|&byte| byte < b' ') {
            return Err(ReadFailure::NoDump("a mangled entry in the dump".into()));
        }
        vars.push((
            OsStr::from_bytes(name).to_os_string(),
            OsStr::from_bytes(&chunk[eq + 1..]).to_os_string(),
        ));
    }

    if vars.is_empty() {
        return Err(ReadFailure::NoDump("no variables".into()));
    }
    Ok(vars)
}

/// Read at most `cap` bytes from `reader`, stopping early once `stop_at` appears
/// in what has been read (or at EOF, whichever comes first).
///
/// The early stop is what keeps the read from waiting on a startup file's own
/// background process: such a process inherits the reader's pipes, so EOF may
/// never arrive even though the dump is complete. A complete dump is complete —
/// what the shell prints after it is not part of the read.
#[cfg(unix)]
async fn read_capped(
    mut reader: impl tokio::io::AsyncRead + Unpin,
    cap: usize,
    stop_at: Option<&[u8]>,
) -> Vec<u8> {
    use tokio::io::AsyncReadExt as _;

    let mut buf = Vec::new();
    let mut chunk = [0u8; READ_CHUNK];
    while buf.len() < cap {
        match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let room = cap - buf.len();
                buf.extend_from_slice(&chunk[..n.min(room)]);
                if let Some(marker) = stop_at
                    && !marker.is_empty()
                    && buf.windows(marker.len()).any(|window| window == marker)
                {
                    break;
                }
            }
        }
    }
    buf
}

/// Read `reader` to EOF, discarding it.
///
/// The shell's stderr is never inspected, never logged and never attached to a
/// failure; it is drained only so a talkative startup file cannot block the
/// shell on a full pipe.
#[cfg(unix)]
async fn drain(mut reader: impl tokio::io::AsyncRead + Unpin) {
    use tokio::io::AsyncReadExt as _;

    let mut chunk = [0u8; READ_CHUNK];
    while let Ok(n) = reader.read(&mut chunk).await {
        if n == 0 {
            break;
        }
    }
}

/// Releases a read's reaper when its capture is over.
///
/// The release is a `Drop` rather than a plain call because the capture can end
/// by the read being left early — the reader is cancelled at shutdown, or a panic
/// unwinds between the spawn and the capture — and the reaper has no other way out
/// of its wait: a shell left unreaped would park it, and the stderr drainer it
/// owns, for the life of the process.
#[cfg(unix)]
#[derive(Clone)]
struct CaptureDone(Arc<tokio::sync::Notify>);

#[cfg(unix)]
impl CaptureDone {
    /// A fresh, unreleased guard.
    fn new() -> Self {
        CaptureDone(Arc::new(tokio::sync::Notify::new()))
    }

    /// Resolve once the capture is over.
    async fn wait(&self) {
        self.0.notified().await;
    }
}

#[cfg(unix)]
impl Drop for CaptureDone {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}

/// SIGKILL a read shell's whole process group.
///
/// The shell leads its own session and group (`setsid` in the spawn below), so
/// the group id is the pid it was spawned with, and grandchildren (a startup
/// file's background job) are in that group and die with it.
///
/// Must be called while the shell is still unreaped: a group whose last member has
/// been reaped can have its id reused, and the signal would then reach a group the
/// read never started. [`run_shell`] holds the shell's reap until after this call
/// for exactly that reason.
#[cfg(unix)]
fn kill_group(pgid: Option<u32>) {
    let Some(pgid) = pgid else {
        return;
    };
    let Ok(pgid) = libc::pid_t::try_from(pgid) else {
        return;
    };
    // SAFETY: `kill` with a negative pid signals the process group `-pgid`. The
    // caller's guarantee that the shell is unreaped is what makes that group the
    // read's own: the shell leads it, and its descendants are its only other
    // members.
    let ret = unsafe { libc::kill(-pgid, libc::SIGKILL) };
    if ret != 0 {
        let err = std::io::Error::last_os_error();
        // ESRCH: the group is already gone (this kill follows the capture's own,
        // or the shell died on its own) — nothing can be in it to stray, so the
        // warning would claim a stray that does not exist.
        if err.raw_os_error() != Some(libc::ESRCH) {
            tracing::warn!(pgid, err = %err, "kill(-pgid) failed — a read stray may survive");
        }
    }
}

/// One read through a shell: start `shell` the platform's way, run `dump_cmd`
/// in it, and parse what it prints.
///
/// `nonce` is the marker nonce [`parse_dump`] must look for, and its end marker
/// is also what stops the capture (see [`run_shell`]).
#[cfg(unix)]
async fn read_with_shell(
    shell: &Path,
    dump_cmd: &str,
    nonce: &str,
    cwd: Option<&Path>,
    timeout: Duration,
) -> Result<OwnerEnv, ReadFailure> {
    let end = end_marker(nonce);
    let bytes = run_shell(shell, dump_cmd, cwd, timeout, Some(end.as_bytes())).await?;
    Ok(OwnerEnv::new(parse_dump(&bytes, nonce)?))
}

/// Start `shell` the platform's way and run `cmd` in it, returning what it wrote
/// to stdout.
///
/// The shell inherits this process's environment exactly as it is — deliberately
/// no `env_clear`, no completion: the point of the read is the environment a
/// startup file produces on top of the daemon's own. Only the working directory
/// is set (`cwd`, the owner's home when it is one; the daemon's own when it is
/// not, which is all a shell needs).
///
/// The capture ends as soon as `stop_at` has been read (or at EOF), and is
/// bounded by `timeout` either way: on expiry the whole group is killed, so a
/// startup file that hangs cannot stretch the read. The shell is not waited for
/// once its output is in: a detached task reaps it under the same bound — killing
/// the group first if the shell itself outlives it — and discards its stderr (a
/// talkative startup file must not block the shell, and nothing it prints is ever
/// inspected or recorded). A startup file's own background processes therefore
/// outlive the read holding the shell's own pipes, not a terminal's: one that
/// writes to the captured stdout after the capture stopped gets EPIPE, and one
/// still holding the shell's pipes when the bound expires loses the shell to the
/// reaper's kill.
#[cfg(unix)]
async fn run_shell(
    shell: &Path,
    cmd: &str,
    cwd: Option<&Path>,
    timeout: Duration,
    stop_at: Option<&[u8]>,
) -> Result<Vec<u8>, ReadFailure> {
    let basename = shell_basename(shell);
    let recipe = shell_recipe(&basename, cfg!(target_os = "macos"))
        .ok_or(ReadFailure::UnknownShell(basename))?;

    let mut child = {
        let mut cmd_builder = tokio::process::Command::new(shell);
        cmd_builder.args(recipe);
        cmd_builder.arg("-c").arg(cmd);
        cmd_builder
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        if let Some(dir) = cwd {
            cmd_builder.current_dir(dir);
        }
        // A session of its own, with no controlling terminal: the read is
        // deliberately not a terminal. This is also what makes an *interactive*
        // shell usable here — such a shell does job-control setup on startup and
        // calls `tcsetpgrp`, which stops a process-group leader that does not own
        // the controlling terminal (SIGTTOU); the shell would sit stopped before
        // it ever ran the command. `setsid` also makes the child its own group
        // leader, which is what lets a timed-out read kill the whole group.
        // SAFETY: `setsid` is async-signal-safe, and the child is freshly forked,
        // so it is not a group leader and the call succeeds.
        unsafe {
            cmd_builder.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        cmd_builder
            .spawn()
            .map_err(|e| ReadFailure::Spawn(e.to_string()))?
    };
    let pgid = child.id();
    let Some(stdout) = child.stdout.take() else {
        return Err(ReadFailure::Spawn("no stdout pipe".into()));
    };
    let Some(stderr) = child.stderr.take() else {
        return Err(ReadFailure::Spawn("no stderr pipe".into()));
    };

    // The reaper owns the child from here. It drains stderr from the moment the
    // shell is spawned — a talkative startup file must not block the shell on a
    // full pipe — but it does not reap the shell until the capture below is over:
    // `kill_group` signals the shell's own process group, and a group whose last
    // member has been reaped can have its id reused, so the signal must never race
    // the reap.
    //
    // Its wait for the shell is bounded by the read's own timeout, and a shell
    // that outlives it is killed as a group (still unreaped, so the id is still its
    // own) and collected under the same bound. A stray holder of the shell's pipes
    // therefore cannot keep the task alive, and neither can an abandoned read: the
    // guard below releases the reaper on the way out however this function is left.
    let capture_done = CaptureDone::new();
    tokio::spawn({
        let capture_done = capture_done.clone();
        async move {
            let draining = tokio::spawn(drain(stderr));
            capture_done.wait().await;
            if tokio::time::timeout(timeout, child.wait()).await.is_err() {
                // The shell outlived its own bound: kill its group — it is still
                // unreaped, so the id is still its own — and give the kill the same
                // bound. A shell that survives even SIGKILL (uninterruptible I/O is
                // the case the bound exists for) is then left to the OS: the task
                // ends here, and its pipes with it.
                kill_group(pgid);
                let _ = tokio::time::timeout(timeout, child.wait()).await;
            }
            draining.abort();
        }
    });

    let captured =
        tokio::time::timeout(timeout, read_capped(stdout, DUMP_STDOUT_CAP, stop_at)).await;
    // The child is still unreaped here, so its group id is still its own.
    if captured.is_err() {
        kill_group(pgid);
    }
    // Only now may the reaper reap the shell.
    drop(capture_done);
    captured.map_err(|_elapsed| ReadFailure::Timeout)
}

/// The file name of a shell path — what [`shell_recipe`] is keyed on. The
/// directory is deliberately skipped: the same shell is often reachable by
/// several paths.
#[cfg(unix)]
#[must_use]
fn shell_basename(shell: &Path) -> String {
    shell
        .file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
}

/// A fresh nonce for one dump — 16 hex characters, from the process RNG, so a
/// stray line from a startup file can never coincide with a marker.
#[cfg(unix)]
#[must_use]
fn new_nonce() -> String {
    format!("{:016x}", rand::random::<u64>())
}

/// The command the shell runs to dump its environment: this binary's hidden
/// `__env-dump` verb with the read's nonce. The dumper runs as a CHILD of the
/// shell (no `exec`), so it inherits the environment the startup files exported
/// — the whole point of the read.
#[cfg(unix)]
#[must_use]
fn dump_command(exe: &Path, nonce: &str) -> String {
    format!(
        "{} __env-dump {nonce}",
        crate::tools::path::shell_quote(&exe.to_string_lossy())
    )
}

/// One unix read: resolve the owner's shell, start it the platform's way, and
/// let the child dumper print the environment it inherited.
#[cfg(unix)]
async fn read_unix() -> Result<OwnerEnv, ReadFailure> {
    let (shell, home) = owner_shell_and_home();
    let shell = shell.ok_or(ReadFailure::NoShell)?;
    let exe = std::env::current_exe().map_err(|e| ReadFailure::Spawn(e.to_string()))?;
    let nonce = new_nonce();
    let dump_cmd = dump_command(&exe, &nonce);
    read_with_shell(&shell, &dump_cmd, &nonce, home.as_deref(), READ_TIMEOUT).await
}

/// The owner's shell and home directory, from one lookup of the account record:
/// the shell the system records for it (falling back to a non-empty `$SHELL`), and
/// the home a non-empty `$HOME` names when it is a directory, else the passwd home
/// when that is one.
///
/// The home may resolve to nothing, in which case the read sets no working
/// directory at all — the shell then starts wherever the daemon is, which is all a
/// shell needs. (A home that does not exist would fail the spawn outright, leaving
/// the agent on the fallback for good.)
#[cfg(unix)]
#[must_use]
fn owner_shell_and_home() -> (Option<PathBuf>, Option<PathBuf>) {
    let (recorded_shell, recorded_home) = passwd_record().unwrap_or((None, None));
    let shell = recorded_shell.or_else(|| {
        std::env::var_os("SHELL")
            .filter(|shell| !shell.is_empty())
            .map(PathBuf::from)
    });
    let home = std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
        .filter(|home| home.is_dir())
        .or_else(|| recorded_home.filter(|home| home.is_dir()));
    (shell, home)
}

/// The shell and home directory the system records for the effective uid, or
/// `None` when there is no entry.
///
/// `getpwuid_r` rather than `getpwuid`: the reentrant form keeps the entry in a
/// caller-owned buffer instead of returning shared mutable state.
#[cfg(unix)]
#[must_use]
fn passwd_record() -> Option<(Option<PathBuf>, Option<PathBuf>)> {
    // One passwd entry is a few hundred bytes; 16 KiB is the customary
    // _SC_GETPW_R_SIZE_MAX with room to spare, and a fixed size keeps this from
    // needing a retry loop for ERANGE.
    let mut buf = vec![0; 16 * 1024];
    let mut entry: libc::passwd = unsafe { std::mem::zeroed() };
    let mut found: *mut libc::passwd = std::ptr::null_mut();
    // SAFETY: the buffer is ours and 16 KiB, the result pointer is ours, and
    // `getpwuid_r` has no other precondition. The entry it fills points into
    // `buf`, which outlives the reads below.
    let ret = unsafe {
        libc::getpwuid_r(
            libc::geteuid(),
            &raw mut entry,
            buf.as_mut_ptr(),
            buf.len(),
            &raw mut found,
        )
    };
    if ret != 0 || found.is_null() {
        return None;
    }
    // SAFETY: both pointers are fields of the entry the call just filled, valid
    // until the next `getpw*` call.
    Some(unsafe { (c_path(entry.pw_shell), c_path(entry.pw_dir)) })
}

/// The path a NUL-terminated C string pointer spells, or `None` for a null
/// pointer or an empty string.
///
/// # Safety
///
/// `ptr` must be null, or point to a valid NUL-terminated C string that stays
/// live for the duration of the call.
#[cfg(unix)]
#[must_use]
unsafe fn c_path(ptr: *const libc::c_char) -> Option<PathBuf> {
    use std::os::unix::ffi::OsStrExt as _;

    if ptr.is_null() {
        return None;
    }
    // SAFETY: the caller guarantees a valid NUL-terminated string.
    let bytes = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_bytes();
    (!bytes.is_empty()).then(|| PathBuf::from(OsStr::from_bytes(bytes)))
}

// ── Reading (Windows) ─────────────────────────────────────────────────────

/// One Windows read: the environment the OS assembles for the owner's own
/// programs.
///
/// `CreateEnvironmentBlock` builds a fresh block for the user named by the token
/// it is given, from that user's own settings and the machine's — the same
/// assembly the system performs for a user at logon, so references are expanded,
/// the machine's search path and the user's are both there (in whatever order the
/// system itself puts them: the block is taken as the OS returned it, never
/// re-merged here), and settings added after this process started are included.
/// The token passed is the calling account's own (`OpenProcessToken`): *not* a
/// null token, which returns the system's variables only, and *not* the current
/// process's environment, which `bInherit = FALSE` deliberately keeps out of the
/// block. Where the product runs as a service account rather than as the owner at
/// the keyboard, that account's environment is what this reads — which is what
/// the token says.
///
/// `%USERPROFILE%` and its family are present because a desktop launch runs with
/// the user's profile loaded; the empty block is refused rather than published,
/// since a read that yields nothing would leave commands with nothing but the
/// pinned values.
#[cfg(windows)]
fn read_windows() -> Result<OwnerEnv, ReadFailure> {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::Security::{TOKEN_DUPLICATE, TOKEN_QUERY};
    use windows_sys::Win32::System::Environment::{
        CreateEnvironmentBlock, DestroyEnvironmentBlock,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    let mut token: HANDLE = 0;
    // SAFETY: the pseudo-handle needs no cleanup and this out-pointer is ours.
    let process = unsafe { GetCurrentProcess() };
    // SAFETY: `process` is the (always valid) current-process pseudo-handle and
    // the out-pointer is ours. The token `GetCurrentProcess` stands behind is a
    // *primary* token, and `CreateEnvironmentBlock` requires one with
    // `TOKEN_DUPLICATE` alongside `TOKEN_QUERY` — asking for both is what the API
    // documents, and asking for less would fail every read on a system that
    // enforces it (leaving the agent on the fallback with only a WARN row).
    if unsafe { OpenProcessToken(process, TOKEN_QUERY | TOKEN_DUPLICATE, &raw mut token) } == 0 {
        return Err(ReadFailure::AssemblyFailed("OpenProcessToken".into()));
    }

    let mut block: *mut std::ffi::c_void = std::ptr::null_mut();
    // SAFETY: `token` was just opened and outlives the call; `block` is ours to
    // receive the allocation, which `DestroyEnvironmentBlock` frees below.
    let created = unsafe { CreateEnvironmentBlock(&raw mut block, token, 0) };
    // The block is a copy: the token is not needed once it exists.
    // SAFETY: `token` came from `OpenProcessToken` and is closed exactly once.
    unsafe { CloseHandle(token) };
    if created == 0 || block.is_null() {
        return Err(ReadFailure::AssemblyFailed("CreateEnvironmentBlock".into()));
    }
    // SAFETY: `block` is the double-NUL-terminated UTF-16 block the call just
    // filled and is still owned here.
    let vars = unsafe { walk_block(block.cast()) };
    // SAFETY: `block` came from `CreateEnvironmentBlock` and is freed exactly
    // once; it is not used afterwards.
    unsafe { DestroyEnvironmentBlock(block) };
    if vars.is_empty() {
        return Err(ReadFailure::AssemblyFailed("empty block".into()));
    }
    Ok(OwnerEnv::new(vars))
}

/// Walk the UTF-16, double-NUL-terminated block `CreateEnvironmentBlock`
/// returns into `(name, value)` pairs.
///
/// The `=X:` per-drive current-directory pseudo entries carry their whole
/// spelling in the value position with an empty name, which `Command::env`
/// cannot express (a name containing `=` is rejected), so they are skipped. A
/// child that needs a per-drive current directory is not something this
/// product's commands do.
///
/// # Safety
///
/// `block` must be the block `CreateEnvironmentBlock` returned and must still be
/// owned by the caller.
#[cfg(windows)]
unsafe fn walk_block(block: *const u16) -> Vec<(OsString, OsString)> {
    use std::os::windows::ffi::OsStringExt as _;

    let mut vars = Vec::new();
    let mut cursor = block;
    // SAFETY: the block ends with two NULs, so a NUL here is the end of it.
    while unsafe { *cursor } != 0 {
        let mut len = 0usize;
        // SAFETY: within an entry, the scan stops at that entry's own NUL.
        while unsafe { *cursor.add(len) } != 0 {
            len += 1;
        }
        // SAFETY: `len` is the length of the entry `cursor` points at.
        let entry = unsafe { std::slice::from_raw_parts(cursor, len) };
        if let Some(eq) = entry.iter().position(|&unit| unit == u16::from(b'='))
            && eq > 0
        {
            vars.push((
                OsString::from_wide(&entry[..eq]),
                OsString::from_wide(&entry[eq + 1..]),
            ));
        }
        // SAFETY: step past the entry and its NUL.
        cursor = unsafe { cursor.add(len + 1) };
    }
    vars
}

/// The unit tests: all unix-only (the Windows read is a single OS call with no
/// shell to drive), so the whole module is gated on unix and needs no per-test
/// `cfg`.
#[cfg(all(test, unix))]
mod tests {
    use super::*;

    use std::os::unix::ffi::OsStrExt as _;

    #[test]
    fn parse_dump_round_trips_exotic_bytes() {
        let nonce = "0123456789abcdef";
        // A newline, an `=`, a space and an invalid-UTF-8 byte: each must come
        // back byte for byte, which is what the raw-`OsStr` protocol buys.
        let invalid = OsStr::from_bytes(b"b\xffd");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(start_marker(nonce).as_bytes());
        for (name, value) in [
            ("PLAIN", OsStr::new("value")),
            ("EQUALS", OsStr::new("a=b=c")),
            ("NEWLINE", OsStr::new("line one\nline two")),
            ("SPACE", OsStr::new("a b")),
            ("INVALID", invalid),
        ] {
            bytes.extend_from_slice(name.as_bytes());
            bytes.push(b'=');
            bytes.extend_from_slice(value.as_bytes());
            bytes.push(0);
        }
        bytes.extend_from_slice(end_marker(nonce).as_bytes());

        let vars = parse_dump(&bytes, nonce).expect("the round trip parses");
        assert_eq!(
            vars,
            vec![
                (OsString::from("PLAIN"), OsString::from("value")),
                (OsString::from("EQUALS"), OsString::from("a=b=c")),
                (
                    OsString::from("NEWLINE"),
                    OsString::from("line one\nline two")
                ),
                (OsString::from("SPACE"), OsString::from("a b")),
                (OsString::from("INVALID"), invalid.to_os_string()),
            ]
        );
    }

    #[test]
    fn parse_dump_ignores_output_around_the_dump() {
        let nonce = "feedfacefeedface";
        let mut bytes = Vec::new();
        // A startup file's own output before the dump.
        bytes.extend_from_slice(b"nvm: warning\n");
        bytes.extend_from_slice(start_marker(nonce).as_bytes());
        bytes.extend_from_slice(b"NAME=value");
        bytes.push(0);
        bytes.extend_from_slice(b"OTHER=x");
        bytes.push(0);
        bytes.extend_from_slice(end_marker(nonce).as_bytes());
        // A background job's line after the dump — a complete dump is complete.
        bytes.extend_from_slice(b"stray noise\n");

        let vars = parse_dump(&bytes, nonce).expect("stray output around the dump is tolerated");
        assert_eq!(
            vars,
            vec![
                (OsString::from("NAME"), OsString::from("value")),
                (OsString::from("OTHER"), OsString::from("x")),
            ]
        );
    }

    #[test]
    fn parse_dump_refuses_an_interleaved_write() {
        let nonce = "deadbeefdeadbeef";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(start_marker(nonce).as_bytes());
        bytes.extend_from_slice(b"NAME=value");
        bytes.push(0);
        // Another holder of the pipe writing between two entries: its bytes glue
        // to the next entry's name, which no environment variable's name could
        // carry — the read must fail rather than hand over a mangled set.
        bytes.extend_from_slice(b"stray noise\n");
        bytes.extend_from_slice(b"OTHER=x");
        bytes.push(0);
        bytes.extend_from_slice(end_marker(nonce).as_bytes());

        assert!(matches!(
            parse_dump(&bytes, nonce),
            Err(ReadFailure::NoDump(_))
        ));
    }

    #[test]
    fn parse_dump_requires_markers_and_entries() {
        let nonce = "cafebabecafebabe";
        // No marker at all.
        assert!(matches!(
            parse_dump(b"just some output\n", nonce),
            Err(ReadFailure::NoDump(_))
        ));
        // Start marker, one entry, no end marker.
        let mut unterminated = Vec::new();
        unterminated.extend_from_slice(start_marker(nonce).as_bytes());
        unterminated.extend_from_slice(b"NAME=value\0");
        assert!(matches!(
            parse_dump(&unterminated, nonce),
            Err(ReadFailure::NoDump(_))
        ));
        // Both markers, but nothing between them.
        let empty = format!("{}{}", start_marker(nonce), end_marker(nonce));
        assert!(matches!(
            parse_dump(empty.as_bytes(), nonce),
            Err(ReadFailure::NoDump(_))
        ));
    }

    /// The read through a real shell, bounded by the timeout it is given.
    ///
    /// `/bin/sh` is driven with the reader's own recipe, so on macOS these lanes run
    /// the host's startup files (`/etc/profile`, `~/.profile`) — the shells the
    /// product itself runs every ten minutes. The 10 s timeout is what keeps a slow
    /// one from hanging the lane.
    #[tokio::test]
    async fn read_with_shell_parses_a_synthetic_dump() {
        const NONCE: &str = "0123456789abcdef";
        // `\036`/`\037`/`\000`/`\035` are the markers and the separator as
        // `printf` understands them, so the shell emits the exact protocol
        // bytes without this test owning a binary to dump with.
        let dump = format!("printf '\\036{NONCE}\\037NAME=VALUE\\000\\036{NONCE}\\035'");

        let env = read_with_shell(
            Path::new("/bin/sh"),
            &dump,
            NONCE,
            None,
            Duration::from_secs(10),
        )
        .await
        .expect("the synthetic dump parses");
        assert_eq!(
            env.vars(),
            &[(OsString::from("NAME"), OsString::from("VALUE"))]
        );
    }

    #[tokio::test]
    async fn read_with_shell_reports_no_dump_when_the_shell_is_silent() {
        let failure = read_with_shell(
            Path::new("/bin/sh"),
            ":",
            "0123456789abcdef",
            None,
            Duration::from_secs(10),
        )
        .await
        .err()
        .expect("a silent shell has no dump");
        assert!(matches!(failure, ReadFailure::NoDump(_)), "{failure}");
    }

    #[tokio::test]
    async fn read_with_shell_times_out_on_a_hanging_shell() {
        let failure = read_with_shell(
            Path::new("/bin/sh"),
            "sleep 30",
            "0123456789abcdef",
            None,
            Duration::from_millis(250),
        )
        .await
        .err()
        .expect("a hanging shell times out");
        assert_eq!(failure, ReadFailure::Timeout);
    }

    #[tokio::test]
    async fn read_with_shell_refuses_an_unknown_shell() {
        let failure = read_with_shell(
            Path::new("/bin/pwsh"),
            ":",
            "0123456789abcdef",
            None,
            Duration::from_secs(10),
        )
        .await
        .err()
        .expect("an unknown shell is refused");
        assert_eq!(failure, ReadFailure::UnknownShell("pwsh".to_string()));
    }
}

/// The evidence harness for the owner's shell environment: one manual, ignored run
/// that compares — by name, count and shape only — the environment an agent's
/// command actually receives with the one this machine's own login shell produces,
/// what the fallback gives before a read has succeeded, and what each way a read can
/// fail records.
///
/// Deliberately outside the suite: the unit tests above run `/bin/sh` alone —
/// which on macOS still reads the system and user profile, bounded by the read's
/// own timeout — while this one runs the owner's own shell over his real startup
/// files. Run it with the binary built:
///
/// ```text
/// cargo build --bin mahbot
/// cargo test --lib -- --ignored --nocapture evidence
/// ```
///
/// It prints no value of the environment — only names, counts, timings and the
/// product's own recorded words.
#[cfg(all(test, unix))]
mod evidence {
    use super::*;
    use std::collections::BTreeSet;

    /// The names in a newline-separated `NAME=VALUE` listing.
    fn listed_names(bytes: &[u8]) -> BTreeSet<String> {
        String::from_utf8_lossy(bytes)
            .lines()
            .filter_map(|line| line.split_once('=').map(|(name, _)| name.to_string()))
            .collect()
    }

    /// The raw value of `name` in an `env` listing, if it is there.
    fn env_value(bytes: &[u8], name: &str) -> Option<Vec<u8>> {
        let prefix = format!("{name}=");
        bytes
            .split(|&b| b == b'\n')
            .find_map(|line| line.strip_prefix(prefix.as_bytes()))
            .map(<[u8]>::to_vec)
    }

    /// The names a shell rewrites in the environment it hands its own child.
    /// Needed because the command under test is a shell itself, so its own
    /// bookkeeping must not read as the product changing the owner's environment.
    fn shell_bookkeeping(name: &str) -> bool {
        matches!(name, "HOME" | "LOGNAME" | "OLDPWD" | "PWD" | "SHLVL" | "_")
    }

    /// The names a recovered environment carries.
    fn recovered_names(env: &OwnerEnv) -> BTreeSet<String> {
        env.vars()
            .iter()
            .map(|(name, _)| name.to_string_lossy().into_owned())
            .filter(|name| !name.is_empty())
            .collect()
    }

    /// The path this test's build put the `mahbot` binary at, so the read can ask
    /// the real binary to dump: `target/debug/mahbot` beside the test harness'
    /// `target/debug/deps/…`.
    fn built_binary() -> Option<PathBuf> {
        let harness = std::env::current_exe().ok()?;
        let candidate = harness.parent()?.parent()?.join("mahbot");
        candidate.is_file().then_some(candidate)
    }

    /// The same shell the read uses, listing its own environment between two
    /// markers its own command prints — so the capture ends with the listing, not
    /// with whatever the startup files left holding the pipe.
    async fn marked_env(shell: &Path, home: Option<&Path>) -> Vec<u8> {
        let nonce = new_nonce();
        let start = start_marker(&nonce);
        let end = end_marker(&nonce);
        let cmd = format!(
            "printf '%s\\n' {}; env; printf '%s\\n' {}",
            crate::tools::path::shell_quote(&start),
            crate::tools::path::shell_quote(&end)
        );
        let bytes = run_shell(shell, &cmd, home, READ_TIMEOUT, Some(end.as_bytes()))
            .await
            .expect("the same shell lists its environment");
        // Keep only what sits between the markers: a startup file's own output
        // before them is not the listing.
        let text = String::from_utf8_lossy(&bytes).into_owned();
        match (text.find(&start), text.find(&end)) {
            (Some(from), Some(to)) if from < to => text.as_bytes()[from + start.len()..to].to_vec(),
            _ => Vec::new(),
        }
    }

    /// What a read recovers, next to what the owner's own login shell prints, what
    /// an agent's command then receives, what the fallback gives before a read has
    /// succeeded, and what each way a read can fail records.
    ///
    /// The closest available instrument for "the terminal's result": the same
    /// recipe the reader uses (a login and interactive shell on macOS, an
    /// interactive one on Linux), started the same way [`run_shell`] starts it —
    /// without a terminal, so nothing the terminal itself would add is reproduced
    /// and nothing is invented to stand in for it.
    ///
    /// The assertions are the properties the product promises — the read recovers
    /// the owner's set, an agent's command receives it, the two pinned exceptions
    /// win, the fallback is the reduced set, a recorded reason carries no value — so
    /// a host where the contract broke fails the lane instead of only printing
    /// differently. The login-shell comparison itself is printed, not asserted: a
    /// startup file may legitimately behave differently under `-c`.
    ///
    /// `#[serial]`: the snapshot is process-global.
    #[serial_test::serial(shell_env)]
    #[tokio::test]
    #[ignore = "manual evidence run: needs `cargo build --bin mahbot` and the owner's shell"]
    async fn owner_environment_end_to_end() {
        let Some(binary) = built_binary() else {
            println!("EVIDENCE SKIPPED: build the binary first (cargo build --bin mahbot)");
            return;
        };
        let (shell, home) = owner_shell_and_home();
        let shell = shell.expect("the account records a shell");
        println!(
            "shell: {} | home resolved: {}",
            shell_basename(&shell),
            home.is_some()
        );

        // ── The read, next to the same shell's own listing ────────────────────
        let nonce = new_nonce();
        let started = Instant::now();
        let recovered = read_with_shell(
            &shell,
            &dump_command(&binary, &nonce),
            &nonce,
            home.as_deref(),
            READ_TIMEOUT,
        )
        .await
        .expect("the read succeeds on this host");
        let elapsed_ms = started.elapsed().as_millis();
        let read_names = recovered_names(&recovered);

        let listed = marked_env(&shell, home.as_deref()).await;
        let shell_names = listed_names(&listed);
        let missing: Vec<&String> = shell_names.difference(&read_names).collect();
        let extra: Vec<&String> = read_names.difference(&shell_names).collect();
        println!(
            "read: {} names, {} PATH entries, {elapsed_ms} ms",
            read_names.len(),
            path_entry_count(recovered.vars())
        );
        println!(
            "login shell: {} names | names the read missed: {} | names the read added: {}",
            shell_names.len(),
            missing.len(),
            extra.len()
        );
        assert!(
            !read_names.is_empty(),
            "a read that recovered nothing is not a read"
        );

        // ── What an agent's command receives ──────────────────────────────────
        // The product's own environment application, on the product's own command
        // builder path. The read's own values for the two exceptions are replaced by
        // markers first, so the assertions below show the product's values winning
        // rather than coinciding: the read inherits the daemon's own pinned temp
        // root, so a presence check alone would hold with no override at all.
        let marker = "/nonexistent-owner-value-evidence";
        let temp_vars = crate::temp::shell_temp_vars();
        let mut replaced: Vec<OsString> = vec![OsString::from("HOME")];
        replaced.extend(temp_vars.iter().map(|(name, _)| OsString::from(name)));
        let mut vars = recovered.vars().to_vec();
        vars.retain(|(name, _)| !replaced.contains(name));
        for name in &replaced {
            vars.push((name.clone(), OsString::from(marker)));
        }
        set_snapshot(Some(OwnerEnv::new(vars)));
        let mut command = {
            let mut cmd = tokio::process::Command::new("sh");
            cmd.arg("-c").arg("env");
            crate::tools::shell::apply_agent_env(&mut cmd);
            cmd
        };
        let command_started = Instant::now();
        let command_out = command.output().await.expect("the command runs");
        let command_ms = command_started.elapsed().as_millis();
        let command_names = listed_names(&command_out.stdout);
        let command_home = env_value(&command_out.stdout, "HOME");
        let imposed_home = crate::tools::shell::agent_env_pairs()
            .into_iter()
            .find_map(|(name, value)| (name == "HOME").then_some(value));
        let imposed_home = imposed_home
            .as_deref()
            .map(std::ffi::OsStr::as_encoded_bytes);
        // Each pinned temp name must carry the product's value, not the marker the
        // read's own set now holds.
        let temp_names: Vec<(String, bool)> = temp_vars
            .iter()
            .map(|(name, pinned)| {
                let value = env_value(&command_out.stdout, name);
                let wins = value.as_deref() == Some(pinned.as_bytes())
                    && value.as_deref() != Some(marker.as_bytes());
                (name.clone(), wins)
            })
            .collect();
        set_snapshot(None);

        let added: Vec<&String> = command_names.difference(&read_names).collect();
        let lost: Vec<&String> = read_names.difference(&command_names).collect();
        // The command under test is itself a shell, and a shell rewrites its own
        // bookkeeping names on the way to its child — `/bin/sh` here does not carry
        // the owner's own `OLDPWD` over — so a raw name diff would blame the product
        // for the child interpreter's own doing.
        let (bookkeeping, lost): (Vec<&String>, Vec<&String>) = lost
            .into_iter()
            .partition(|name| shell_bookkeeping(name.as_str()));
        let (bookkeeping_added, added): (Vec<&String>, Vec<&String>) = added
            .into_iter()
            .partition(|name| shell_bookkeeping(name.as_str()));
        let rewritten: Vec<&String> = bookkeeping.into_iter().chain(bookkeeping_added).collect();
        println!(
            "agent command: {} names, started in {command_ms} ms",
            command_names.len()
        );
        println!("names the command's own shell rewrote: {rewritten:?}");
        println!("owner names it lost: {lost:?} | names it gained: {added:?}");
        println!("temp variables: {temp_names:?}");
        // The data-location home is the one value the product imposes on top of the
        // owner's own environment; compared, never printed.
        println!(
            "HOME: the command's value is the product's, not the read's own: {}",
            command_home.as_deref() == imposed_home
                && command_home.as_deref() != Some(marker.as_bytes())
        );
        assert!(
            lost.is_empty(),
            "the owner's own names must reach the command: {lost:?}"
        );
        assert!(
            added.is_empty(),
            "nothing but the pinned values may be added: {added:?}"
        );
        assert!(
            command_home.as_deref() == imposed_home,
            "the data-location home must win over the read's own"
        );
        assert!(
            temp_names.iter().all(|(_, pinned)| *pinned),
            "the pinned temp names must win: {temp_names:?}"
        );

        // ── The fallback, before any read has succeeded ───────────────────────
        let fallback: BTreeSet<String> = crate::tools::shell::agent_env_pairs()
            .iter()
            .map(|(name, _)| name.to_string_lossy().into_owned())
            .collect();
        let fallback_path_entries = path_entry_count(&crate::tools::shell::agent_env_pairs());
        println!(
            "fallback (no read yet): {} names, {fallback_path_entries} PATH entries, HOME pinned: {}, TERM pinned: {}",
            fallback.len(),
            fallback.contains("HOME"),
            fallback.contains("TERM")
        );
        assert!(
            fallback.contains("HOME") && fallback.contains("PATH"),
            "the fallback is the reduced environment: {fallback:?}"
        );
        assert!(
            !fallback.contains("OLDPWD"),
            "the fallback must not carry the owner's own environment: {fallback:?}"
        );

        // ── Every way a read can fail, and what it records ────────────────────
        // `OwnerEnv` is deliberately not `Debug` (a derived one would let a stray
        // `{:?}` print the owner's secrets), so the failures are taken by match.
        let failure_of = |result: Result<OwnerEnv, ReadFailure>| match result {
            Ok(_) => panic!("this read was expected to fail"),
            Err(failure) => failure,
        };
        let nonce = "0123456789abcdef";
        let failures = [
            (
                "shell the product does not know",
                failure_of(
                    read_with_shell(
                        Path::new("/bin/pwsh"),
                        ":",
                        nonce,
                        None,
                        Duration::from_secs(1),
                    )
                    .await,
                ),
            ),
            (
                "shell that produces no dump",
                failure_of(
                    read_with_shell(
                        Path::new("/bin/sh"),
                        ":",
                        nonce,
                        None,
                        Duration::from_secs(1),
                    )
                    .await,
                ),
            ),
            (
                "shell that hangs",
                failure_of(
                    read_with_shell(
                        Path::new("/bin/sh"),
                        "sleep 30",
                        nonce,
                        None,
                        Duration::from_millis(250),
                    )
                    .await,
                ),
            ),
        ];
        for (case, failure) in failures {
            let recorded = failure.to_string();
            println!("{case}: recorded as \"{recorded}\"");
            // The recorded reason is a description, never a value out of the
            // environment: an `=` in it would be the shape of one.
            assert!(
                !recorded.is_empty() && !recorded.contains('='),
                "a recorded reason names no value: {recorded}"
            );
        }
        let delays: Vec<u64> = (1..=5).map(|n| retry_delay(n).as_secs()).collect();
        println!("retry delay after 1..5 consecutive failures: {delays:?} seconds");
        assert!(
            delays[0] == READ_INTERVAL.as_secs()
                && delays.windows(2).all(|pair| pair[0] <= pair[1]),
            "a repeated failure is retried no more often than the interval: {delays:?}"
        );
    }
}
