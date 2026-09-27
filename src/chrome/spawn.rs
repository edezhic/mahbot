//! The single chrome-use CLI spawn path.
//!
//! A wedged daemon hangs inside the CLI's own ~152 s retry loop, so every
//! health/watchdog/sweep call is deadline-bounded (the shutdown close path is
//! instead bounded by its outer total-budget timeout); the interactive tool
//! bounds its dispatch itself per call through [`crate::chrome::clocks`] (the
//! derivation of the pair for a step that declares its own chrome-side clock),
//! and on a timeout runs a bounded health
//! evaluation — failing fast with daemon guidance when the daemon is down or
//! wedged (a second consecutive hang on a session-daemon probe), since the
//! mahbot-side bound cuts off the CLI's own wedge signature. One helper,
//! [`spawn_cli`], plus the per-call timeout and cancellation policies.
//!
//! Two clocks and the relay-recovery policy ride every call — [`crate::chrome`]
//! is the one statement of both. The environment is pinned so nothing the
//! service happens to be started with can divert the call onto a browser
//! chrome-use launches itself.
//!
//! The child's EXIT decides the call, never pipe EOF: on Windows a program the
//! command leaves behind keeps the output channel open, so pipe EOF cannot stand
//! in for the child's exit. The exit status plus whatever bytes the pipes
//! delivered are therefore the result — the exited child's own answer keeps being
//! drained while it arrives (one [`PIPE_DRAIN_MAX`] window over BOTH pipes
//! together) — and a pipe a leftover holder keeps open past [`PIPE_DRAIN_GRACE`]
//! is reported instead ([`CliOutput::leftover_pipes`]). The per-call deadline
//! bounds the CHILD's lifetime — spawning and waiting for its exit — only: the
//! collection that follows is that one bounded window, so a child that answers at
//! the very end of its deadline is still reported by its answer, never as a
//! timeout.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::process::Command;
use tracing::debug;

use crate::chrome::{CliRecovery, RELAY_RECOVERY_BUDGET};
use crate::util::UnwrapPoison;

/// Per-call timeout policy for a chrome-use CLI call. The deadline bounds the
/// CHILD's lifetime — spawning and waiting for its exit — only: collecting the
/// bytes it wrote is a separate step bounded by one [`PIPE_DRAIN_MAX`] window
/// over BOTH pipes together, renewed while bytes keep arriving, so an answer the
/// child already produced is never cut off by this clock.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CliTimeout {
    /// No per-call bound — only the shutdown close path uses it (its own outer
    /// total-budget timeout bounds the whole cleanup sequence; see module doc).
    Unbounded,
    /// Kill the child after the deadline (`kill_on_drop` makes dropping the
    /// child fatal). See the type doc: only the child's lifetime is bounded.
    Bounded(Duration),
}

/// Everything one chrome-use CLI invocation needs. The binary path is
/// resolved by the caller so call-site-specific missing-CLI handling (and,
/// for the tool, session tracking after the path resolves) stays at the call
/// site. Argument ORDER is the caller's business: the helper folds
/// `--json` / `--session <s>` in last, or before a caller `--` marker (see
/// [`build_argv`]).
pub(crate) struct CliSpawn<'a> {
    pub(crate) path: &'a Path,
    pub(crate) args: &'a [&'a str],
    pub(crate) session: Option<&'a str>,
    /// Add `--json` and pipe stdout (JSON envelopes are read from stdout).
    /// Without it stdout is nulled — non-JSON callers judge by exit status.
    pub(crate) json: bool,
    /// Pipe stderr (needed when the caller reads failure details from it);
    /// otherwise null it.
    pub(crate) capture_stderr: bool,
    /// Kill the child when the awaiting future is dropped (task cancellation
    /// or a `Bounded` timeout) instead of letting it run to completion: letting a
    /// timed-out call's chrome-use child keep retrying in the background has no
    /// upside. Every caller passes `true` except the session-recovery stop, which
    /// passes `false` so a stop nothing is waiting on any more still runs on and
    /// reclaims the session's tabs (`chrome_daemon::session_stop_via`).
    pub(crate) cancel_kills: bool,
    /// Optional stdin payload (e.g. `fill --stdin`). When set, stdin is piped
    /// and the payload is written concurrently with output collection.
    pub(crate) input: Option<Vec<u8>>,
    /// The product's own bound on this call: `Bounded(kill)` is the caller's own
    /// kill — computed with [`crate::chrome::kill_bound`] so it rides above the
    /// declared `chrome_side` plus whatever recovery the call allows, or with
    /// [`crate::chrome::probe_clocks`], where the one bound IS both the kill and the
    /// declaration — while `Unbounded` leaves the child with no product kill at all
    /// (a lifecycle call bounded only by the caller's outer budget).
    pub(crate) timeout: CliTimeout,
    /// The chrome-side deadline this call DECLARES to chrome-use (always
    /// applied as `AGENT_BROWSER_DEFAULT_TIMEOUT`, in ms) — the clock
    /// chrome-use itself works to.
    pub(crate) chrome_side: Duration,
    /// Whether this call may run chrome-use's own relay self-heal.
    pub(crate) recovery: CliRecovery,
}

/// The result of one chrome-use CLI invocation. `status` is the child's own
/// exit status and decides the call (never pipe EOF; see the module doc);
/// `stdout`/`stderr` are the bytes its pipes delivered within that one
/// [`PIPE_DRAIN_MAX`] window (`stderr` is empty unless `capture_stderr` was set);
/// `leftover_pipes` is true when the child exited but a process it left behind
/// still held an output channel open past that window — the verdict is still the
/// child's, and the leftover is reported separately by the caller.
pub(crate) struct CliOutput {
    pub(crate) status: std::process::ExitStatus,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
    pub(crate) leftover_pipes: bool,
    /// The product cut off what chrome-use wrote, so what the caller parses may
    /// be a truncated envelope rather than the whole answer. Either cause counts:
    /// a stream reached [`PIPE_READ_CAP`], or the collection window ended while
    /// bytes were still arriving ([`PipeDrain::held_after_exit`]).
    pub(crate) truncated: bool,
}

pub(crate) enum CliRun {
    Output(CliOutput),
    /// Nothing came back from a child of ours: it could not be spawned (the `path`
    /// the caller resolved is gone, or the OS refused the spawn) or its exit could
    /// not be waited on — the two legs [`run_child`] folds here.
    SpawnFailure,
    /// Bounded call exceeded its deadline (child killed).
    TimedOut,
}

/// Set HOME, `CHROMIUM_FLAGS`, and the fixed chrome-use knobs on the command
/// so that the Chromium spawned by chrome-use works in service/docker
/// environments, and pin the call to the owner's real browser
/// ([`pin_real_browser_env`]). The whole inherited browser-control namespace —
/// chrome-use's own `AGENT_BROWSER_*` switches and its `CHROME_USE_*` ones — is
/// cleared FIRST, before any of the product's own values are set, so no switch
/// the service happened to be started with can survive into chrome-use. The two
/// per-call policies — the declared chrome-side deadline and the relay-recovery
/// allowance — are applied separately ([`apply_chrome_side`],
/// [`apply_recovery_env`]).
pub(crate) fn ensure_chrome_env(cmd: &mut Command) {
    clear_browser_env(cmd);
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
    // 5-minute idle timeout — the chrome-use daemon still stops after 5 idle
    // minutes, but chrome-use ≥1.5.101 PRESERVES external Chrome tabs on idle
    // (only an explicit close/session stop cleans them up), so mahbot releases
    // the sessions a run opened when it ends, via `crate::tools::chrome_release`.
    cmd.env("AGENT_BROWSER_IDLE_TIMEOUT_MS", "300000");
    // Enable human-like interaction speed for bot-detection avoidance.
    // chrome-use supports the same env vars as agent-browser for backward
    // compatibility.
    cmd.env("AGENT_BROWSER_HUMANIZE", "human");
    // Keep the upgrade-available banner out of every command's stderr.
    cmd.env("CHROME_USE_NO_UPDATE_CHECK", "1");
    cmd.env("AGENT_BROWSER_NO_UPDATE_CHECK", "1");
    pin_real_browser_env(cmd);
}

/// Clear the WHOLE inherited browser-control namespace from the child's
/// environment — both chrome-use's own `AGENT_BROWSER_*` switches and its
/// `CHROME_USE_*` ones — so only the values the product sets afterwards reach
/// chrome-use. An enumerated remove-list only covers the switches known today: a
/// switch nobody enumerated (`AGENT_BROWSER_CONFIG`, `AGENT_BROWSER_BROWSER`,
/// `CHROME_USE_PROFILE`, …) would still divert the call. The match is
/// case-insensitive because Windows' environment is. Names only — the values are
/// never read or logged.
///
/// The relay-dir name of each namespace is the only one spared:
/// `AGENT_BROWSER_RELAY_DIR`/`CHROME_USE_RELAY_DIR` has to agree with the
/// registered native host (see [`pin_real_browser_env`]), so it rides the
/// service's environment through untouched.
fn clear_browser_env(cmd: &mut Command) {
    for (name, _) in std::env::vars_os() {
        if let Some(name) = name.to_str() {
            let upper = name.to_ascii_uppercase();
            if (upper.starts_with("AGENT_BROWSER_") || upper.starts_with("CHROME_USE_"))
                && !matches!(
                    upper.as_str(),
                    "AGENT_BROWSER_RELAY_DIR" | "CHROME_USE_RELAY_DIR"
                )
            {
                cmd.env_remove(name);
            }
        }
    }
}

/// Pin the browser this call drives to the OWNER'S REAL one, whatever the
/// environment the service happens to be started with says: nothing present in
/// that environment may move a call onto a throwaway browser chrome-use starts
/// itself. That is why the whole browser-control namespace — chrome-use's own
/// `AGENT_BROWSER_*` switches and its `CHROME_USE_*` ones — is CLEARED
/// ([`clear_browser_env`]) rather than the switches found so far removed: only
/// the product's own values then reach chrome-use. `CI` presents the
/// throwaway-profile case, and `AGENT_BROWSER_AUTO_CONNECT` is pinned on so the
/// call attaches to the browser that is already running.
///
/// `CHROME_USE_RELAY_DIR`/`AGENT_BROWSER_RELAY_DIR` is deliberately NOT
/// touched — not cleared with the rest of the namespace, never set: it has to
/// agree with the registered native host. A per-call guarantee that the real
/// browser was actually used is established higher up (the pre-action readiness
/// check and chrome-use's own browser-replacement warning), not here. The
/// per-call knobs ([`apply_chrome_side`], [`apply_recovery_env`]) are applied
/// afterwards.
fn pin_real_browser_env(cmd: &mut Command) {
    cmd.env_remove("CI");
    cmd.env("AGENT_BROWSER_AUTO_CONNECT", "1");
}

/// Apply the relay-recovery policy ([`CliRecovery`]): `Allowed` leaves
/// chrome-use's own self-heal on and pins its revive budget to the number the
/// product's kill rides above — [`clear_browser_env`] has already removed any
/// inherited auto-reconnect switch, so nothing has to be un-set here;
/// `Suppressed` disables the self-heal outright (`0` seconds, no
/// auto-reconnect), so a probe returns "the tool did not answer" instead of
/// spending the recovery window.
fn apply_recovery_env(cmd: &mut Command, recovery: CliRecovery) {
    match recovery {
        CliRecovery::Allowed => {
            cmd.env(
                "AGENT_BROWSER_RELAY_REVIVE_SECS",
                RELAY_RECOVERY_BUDGET.as_secs().to_string(),
            );
        }
        CliRecovery::Suppressed => {
            cmd.env("AGENT_BROWSER_NO_AUTO_RECONNECT", "1");
            cmd.env("AGENT_BROWSER_RELAY_REVIVE_SECS", "0");
        }
    }
}

/// Declare `chrome_side` as this call's chrome-use-side deadline by setting
/// `AGENT_BROWSER_DEFAULT_TIMEOUT` (ms). Applied to EVERY call — the declaration
/// is what lets chrome-use's own verdict, rather than the product's kill,
/// normally reach the caller.
pub(crate) fn apply_chrome_side(cmd: &mut Command, chrome_side: Duration) {
    cmd.env(
        "AGENT_BROWSER_DEFAULT_TIMEOUT",
        chrome_side.as_millis().to_string(),
    );
}

/// Caller args with the global `--json` / `--session <s>` flags folded in.
/// The flags go before a `--` end-of-options marker when the caller args
/// contain one (a fill/type leading-dash text shield): chrome-use's arg
/// preprocessor drops everything after `--`, so appended flags would be
/// swallowed into the text value (verified against chrome-use 1.5.111).
/// Without a `--` the flags go last.
fn build_argv(args: &[&str], json: bool, session: Option<&str>) -> Vec<String> {
    let mut global: Vec<String> = Vec::new();
    if json {
        global.push("--json".to_string());
    }
    if let Some(s) = session {
        global.extend(["--session".to_string(), s.to_string()]);
    }
    if global.is_empty() {
        return args.iter().map(|a| (*a).to_string()).collect();
    }
    let insert_at = args.iter().position(|a| *a == "--").unwrap_or(args.len());
    let mut folded: Vec<String> = args[..insert_at].iter().map(|a| (*a).to_string()).collect();
    folded.extend(global);
    folded.extend(args[insert_at..].iter().map(|a| (*a).to_string()));
    folded
}

/// Spawn a chrome-use CLI invocation per [`CliSpawn`]: apply
/// [`ensure_chrome_env`], the declared chrome-side deadline and the recovery
/// policy, the caller's args with the global `--json` / `--session <s>` flags
/// folded in, pipe stdout and stderr per the `json` / `capture_stderr` flags,
/// and kill-on-drop. A bounded call that hits its deadline reports
/// [`CliRun::TimedOut`]; any spawn IO error is folded into
/// [`CliRun::SpawnFailure`].
pub(crate) async fn spawn_cli(spec: CliSpawn<'_>) -> CliRun {
    let mut cmd = Command::new(spec.path);
    #[cfg(windows)]
    cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    ensure_chrome_env(&mut cmd);
    apply_chrome_side(&mut cmd, spec.chrome_side);
    apply_recovery_env(&mut cmd, spec.recovery);
    cmd.args(build_argv(spec.args, spec.json, spec.session));
    if spec.json {
        cmd.stdout(std::process::Stdio::piped());
    } else {
        cmd.stdout(std::process::Stdio::null());
    }
    if spec.capture_stderr {
        cmd.stderr(std::process::Stdio::piped());
    } else {
        cmd.stderr(std::process::Stdio::null());
    }
    // Null stdin unless a payload is coming: a spawned child must never read
    // the service's own stdin.
    if spec.input.is_some() {
        cmd.stdin(std::process::Stdio::piped());
    } else {
        cmd.stdin(std::process::Stdio::null());
    }
    cmd.kill_on_drop(spec.cancel_kills);

    run_child(&mut cmd, spec.input, spec.timeout).await
}

/// Spawn `cmd`, wait for its exit under `timeout`, and collect its output. The
/// child's EXIT decides the call, not pipe EOF (see the module doc): two reader
/// tasks drain the pipes, the child is waited concurrently, and once it has
/// exited — with both readers already running, so the grace measures a genuinely
/// pending pipe — each reader is given [`PIPE_DRAIN_GRACE`] to deliver its bytes,
/// renewed while they keep arriving and capped by [`PIPE_DRAIN_MAX`]. The two
/// streams are collected in ONE such window, awaited together, so a leftover
/// holding both pipes costs one bounded window, not one per stream. The deadline
/// bounds the CHILD's lifetime only; the collection that follows is that one
/// window, never that clock. Each reader is handed to the bounded retained-channel
/// set ([`crate::util::leftover_channels`]) when the call ends ([`PipeDrain`]), so
/// a retained reader always belongs to a call that has already returned: a reader
/// that reached EOF delivers all its bytes, and a pipe a leftover still holds past
/// the grace keeps its read end open — the leftover is never blocked by a full
/// pipe nor faulted by a closed read end, so the call is reported with a leftover
/// holder rather than allowed to turn a finished command into a timeout.
async fn run_child(cmd: &mut Command, input: Option<Vec<u8>>, timeout: CliTimeout) -> CliRun {
    let Ok(mut child) = cmd.spawn() else {
        return CliRun::SpawnFailure;
    };
    // A stdin payload is written concurrently: a full stdin write must not
    // deadlock against a full stdout pipe. The task is detached — the payload
    // is handed to the pipe, and a child that exited leaves it to fail on
    // EPIPE.
    if let (Some(data), Some(mut stdin)) = (input, child.stdin.take()) {
        tokio::task::spawn(async move {
            use tokio::io::AsyncWriteExt;
            let _ = stdin.write_all(&data).await;
            let _ = stdin.shutdown().await;
        });
    }
    let stdout = child.stdout.take().map(PipeDrain::start);
    let stderr = child.stderr.take().map(PipeDrain::start);
    // The post-exit grace below must measure a genuinely pending pipe, never a
    // reader the runtime has not polled yet: a child that is already gone while
    // its readers have not been scheduled would otherwise be reported as a
    // leftover with a truncated envelope.
    if let Some(drain) = stdout.as_ref() {
        drain.started().await;
    }
    if let Some(drain) = stderr.as_ref() {
        drain.started().await;
    }

    // The deadline bounds the CHILD's lifetime only. Collecting the bytes it wrote
    // is a separate step that must not be charged to it: every byte a child wrote
    // before exiting is already in its pipes, so a child that answers at the very
    // end of its deadline would otherwise be reported as a timeout with its answer
    // thrown away. The collection bounds itself ([`PipeDrain::held_after_exit`]).
    let status = match timeout {
        CliTimeout::Unbounded => match child.wait().await {
            Ok(s) => s,
            Err(_) => return CliRun::SpawnFailure,
        },
        CliTimeout::Bounded(deadline) => match tokio::time::timeout(deadline, child.wait()).await {
            Ok(Ok(status)) => status,
            Ok(Err(_)) => return CliRun::SpawnFailure,
            // The child is dropped as this returns (`kill_on_drop`), and the drains
            // with it — a pipe a leftover holds stays readable for it (see
            // [`crate::util::leftover_channels`]).
            Err(_) => return CliRun::TimedOut,
        },
    };
    // The command's own answer is already written by the time it exits, so the
    // wait below only covers the last bytes of a pipe still being closed, and a
    // pipe a leftover holder keeps open is reported rather than waited for. Both
    // streams share ONE window ([`tokio::join!`]), so a leftover holding both
    // costs one [`PIPE_DRAIN_MAX`], not one per stream.
    let (stdout_held, stderr_held) = tokio::join!(
        held_after_exit(stdout.as_ref()),
        held_after_exit(stderr.as_ref())
    );
    let leftover_pipes = stdout_held || stderr_held;
    let truncated = [stdout.as_ref(), stderr.as_ref()]
        .into_iter()
        .flatten()
        .any(PipeDrain::truncated);
    let output = CliOutput {
        status,
        stdout: stdout.map_or_else(Vec::new, |drain| drain.buf.take()),
        stderr: stderr.map_or_else(Vec::new, |drain| drain.buf.take()),
        leftover_pipes,
        truncated,
    };
    if output.leftover_pipes {
        debug!(
            "chrome-use exited but a process it left behind still holds its output channel — \
             the exit status and the bytes drained so far are this call's result"
        );
    }
    CliRun::Output(output)
}

/// Whether the pipe of `drain` is still held after the child exited; `false` for
/// a stream that was nulled (no drain). A free function so both streams can be
/// awaited together in one [`tokio::join!`] — which is what bounds the whole
/// post-exit collection by one [`PIPE_DRAIN_MAX`] window
/// ([`PipeDrain::held_after_exit`]).
async fn held_after_exit(drain: Option<&PipeDrain>) -> bool {
    match drain {
        Some(drain) => drain.held_after_exit().await,
        None => false,
    }
}

/// How long a finished call's output pipes are given to deliver their last
/// bytes before a leftover holder is reported — and the bound on waiting for a
/// freshly spawned reader to be polled at all.
const PIPE_DRAIN_GRACE: Duration = Duration::from_secs(2);

/// Ceiling on the post-exit collection: ONE window covering BOTH pipes together,
/// renewed while bytes keep arriving. A reader that keeps receiving output is
/// draining the exited child's own answer (a large envelope takes longer than one
/// grace), while the pipe a leftover holder keeps open is left open and reported.
/// Five graces — long enough for a multi-megabyte answer, short enough that a
/// leftover writing forever cannot hold the call.
const PIPE_DRAIN_MAX: Duration = Duration::from_secs(10);

/// Byte chunk a pipe drain reads at a time.
const PIPE_DRAIN_CHUNK_BYTES: usize = 8 * 1024;

/// Bytes one output stream may keep — a memory bound, not a policy: far above
/// any envelope a chrome-use command returns (a page's own text) and small enough
/// that a leftover writing at full speed cannot grow memory. The reader keeps
/// draining past it, so the writer is never blocked by a full pipe; a reader that
/// drops bytes because of this cap marks the drain ([`PipeDrain::truncated`]), so
/// the frontends can name the truncation instead of reading the cut-off envelope
/// as malformed output.
pub(crate) const PIPE_READ_CAP: usize = 64 * 1024 * 1024;

/// A drain of one of the child's output pipes: the buffer the reader fills plus
/// the signals that mark the reader polled and done.
///
/// The reader task is handed to the bounded retained-channel set when this drain
/// is dropped ([`crate::util::leftover_channels`], which prunes a finished reader
/// on its next pass) — i.e. when the CALL ends, never while it is still in
/// flight. That is what lets the set's cap release the oldest channel without
/// ever truncating a call still awaiting its reader: a reader that reached EOF
/// contributes all its bytes, and one still pending when the call ends is a
/// genuine leftover holder whose read end must stay open.
struct PipeDrain {
    /// Bytes drained so far (see [`PipeBuf`]).
    buf: Arc<PipeBuf>,
    /// Signalled by the reader task immediately before its read loop, so
    /// [`Self::started`] can tell a genuinely pending pipe from a reader the
    /// runtime has not polled yet.
    started: Arc<tokio::sync::Notify>,
    /// Signalled when the read loop ends (EOF or a read error).
    finished: Arc<tokio::sync::Notify>,
    /// Bytes read so far, for [`Self::held_after_exit`]: whether the reader is
    /// still receiving output is what separates a slow drain from a pipe a
    /// thing that outlived the command keeps open.
    progress: Arc<std::sync::atomic::AtomicU64>,
    /// Set by the reader when it had to drop bytes at [`PIPE_READ_CAP`] (see
    /// [`Self::truncated`]).
    truncated: Arc<std::sync::atomic::AtomicBool>,
    /// The reader task, handed to the bounded retained-channel set when this
    /// drain is dropped — after the post-exit grace on the normal path, and from
    /// the dropped future on the timeout path.
    task: Option<tokio::task::JoinHandle<()>>,
}

impl PipeDrain {
    fn start<R>(mut pipe: R) -> Self
    where
        R: tokio::io::AsyncRead + Unpin + Send + 'static,
    {
        let buf = Arc::new(PipeBuf(Mutex::new(Some(Vec::new()))));
        let started = Arc::new(tokio::sync::Notify::new());
        let finished = Arc::new(tokio::sync::Notify::new());
        let progress = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let truncated = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task = tokio::task::spawn({
            let buf = Arc::clone(&buf);
            let started = Arc::clone(&started);
            let finished = Arc::clone(&finished);
            let progress = Arc::clone(&progress);
            let truncated = Arc::clone(&truncated);
            async move {
                use tokio::io::AsyncReadExt;
                started.notify_one();
                let mut chunk = [0u8; PIPE_DRAIN_CHUNK_BYTES];
                while let Ok(read) = pipe.read(&mut chunk).await {
                    if read == 0 {
                        break;
                    }
                    // Every byte read is counted — the post-exit renewal below
                    // keys on bytes still arriving — while only what fits under
                    // [`PIPE_READ_CAP`] is kept.
                    progress.fetch_add(read as u64, std::sync::atomic::Ordering::Relaxed);
                    if let Some(bytes) = buf.0.lock().unwrap_poison().as_mut() {
                        let room = PIPE_READ_CAP.saturating_sub(bytes.len()).min(read);
                        if room < read {
                            truncated.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        bytes.extend_from_slice(&chunk[..room]);
                    }
                }
                finished.notify_one();
            }
        });
        Self {
            buf,
            started,
            finished,
            progress,
            truncated,
            task: Some(task),
        }
    }

    /// Wait until the reader task has actually been polled. A grace measured
    /// before that would count a reader the runtime simply had not reached yet
    /// as a pending pipe. The wait is bounded by [`PIPE_DRAIN_GRACE`]: when it
    /// expires the pipe is treated as pending, so the call is reported with a
    /// leftover holder rather than hanging or spinning on a reader the runtime
    /// never reached.
    async fn started(&self) {
        let _ = tokio::time::timeout(PIPE_DRAIN_GRACE, self.started.notified()).await;
    }

    /// Wait out the post-exit grace and report whether a thing that OUTLIVED the
    /// command still holds this pipe.
    ///
    /// Progress — not the mere absence of EOF — is what tells the two apart: a
    /// reader still receiving bytes is draining the exited child's own answer (a
    /// large envelope outlives one grace), while a reader that saw neither bytes
    /// nor EOF across a whole grace is waiting on a holder the call left behind.
    /// The wait is renewed while bytes keep arriving, capped by
    /// [`PIPE_DRAIN_MAX`], so a leftover writing forever is reported instead of
    /// holding the call — but when that cap ends the collection while bytes were
    /// still arriving, the buffer may be cut mid-envelope, so the cap case is also
    /// reported as the product's own cut-off ([`PipeDrain::truncated`]) rather than
    /// only as a leftover holder.
    async fn held_after_exit(&self) -> bool {
        let mut last = self.bytes();
        let mut waited = Duration::ZERO;
        loop {
            if tokio::time::timeout(PIPE_DRAIN_GRACE, self.finished.notified())
                .await
                .is_ok()
            {
                return false;
            }
            waited = waited.saturating_add(PIPE_DRAIN_GRACE);
            let now = self.bytes();
            if now == last {
                return true;
            }
            if waited >= PIPE_DRAIN_MAX {
                // Bytes were still arriving when the cap ended the collection, so
                // what the caller reads may be cut mid-envelope: mark it here.
                self.truncated
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                return true;
            }
            last = now;
        }
    }

    /// Bytes the reader has received so far.
    fn bytes(&self) -> u64 {
        self.progress.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether the reader had to drop bytes at [`PIPE_READ_CAP`], so the
    /// envelope the caller reads is the product's own cut-off answer rather than
    /// what chrome-use wrote.
    fn truncated(&self) -> bool {
        self.truncated.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for PipeDrain {
    /// Discard the buffer and hand the reader to the bounded retained-channel set
    /// as the call ends. On the normal path the post-exit grace has run, so an
    /// unfinished reader here is a genuine leftover holder; on the timeout path
    /// the call's dropped future lands here too — which is why retention exists
    /// at all. Retaining at the END rather than at spawn is what keeps the cap
    /// from ever truncating a call still in flight.
    fn drop(&mut self) {
        // The bytes go first: on the timeout path nothing takes the envelope, so
        // the buffer would stay armed, and a reader retained to keep a leftover's
        // pipe open must not keep growing a buffer nobody will read.
        self.buf.take();
        if let Some(task) = self.task.take() {
            crate::util::leftover_channels::retain_channel(task);
        }
    }
}

/// Bytes drained from one of the child's output pipes.
///
/// The reader appends as it goes, under the lock, up to [`PIPE_READ_CAP`] bytes:
/// whatever arrived before the drain grace expired survives even when the pipe
/// never reaches EOF, and a leftover process writing into the pipe can neither
/// block (the reader keeps draining past the cap) nor grow this buffer past that
/// bound. Once the collector has taken the bytes (`None`) the reader appends
/// nothing more at all. Whatever the cap cut off is marked on the drain, not
/// here ([`PipeDrain::truncated`]).
struct PipeBuf(Mutex<Option<Vec<u8>>>);

impl PipeBuf {
    /// Take everything drained so far and stop collecting.
    fn take(&self) -> Vec<u8> {
        self.0.lock().unwrap_poison().take().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chrome::{
        CHROME_USE_DECLARED_BUDGET, KILL_SLACK, RELAY_RECOVERY_BUDGET, kill_bound,
    };
    use crate::util::test::set_env_var;
    use tokio::process::Command;

    /// Read an env var explicitly set on the command.
    fn cmd_env(cmd: &Command, key: &str) -> Option<String> {
        cmd.as_std()
            .get_envs()
            .find(|(k, _)| *k == std::ffi::OsStr::new(key))
            .and_then(|(_, v)| v.map(|v| v.to_string_lossy().into_owned()))
    }

    /// Whether the command explicitly REMOVES `key` from the child's
    /// environment (`env_remove` lands as a `(key, None)` entry).
    fn cmd_removes(cmd: &Command, key: &str) -> bool {
        cmd.as_std()
            .get_envs()
            .any(|(k, v)| v.is_none() && k == std::ffi::OsStr::new(key))
    }

    /// Whether the command says nothing about `key`, leaving whatever the
    /// parent has (if anything) inherited.
    fn cmd_inherits(cmd: &Command, key: &str) -> bool {
        cmd.as_std()
            .get_envs()
            .all(|(k, _)| k != std::ffi::OsStr::new(key))
    }

    #[test]
    fn build_argv_folds_global_flags_before_dash_shield() {
        assert_eq!(
            build_argv(
                &["fill", "#q", "--", "-tail"],
                true,
                Some("mahbot-chrome-abc")
            ),
            [
                "fill",
                "#q",
                "--json",
                "--session",
                "mahbot-chrome-abc",
                "--",
                "-tail"
            ]
        );
        assert_eq!(
            build_argv(&["open", "https://x"], true, Some("s")),
            ["open", "https://x", "--json", "--session", "s"]
        );
        assert_eq!(
            build_argv(&["session", "list"], false, None),
            ["session", "list"]
        );
        assert_eq!(
            build_argv(&["session", "list"], false, Some("s")),
            ["session", "list", "--session", "s"]
        );
    }

    #[test]
    fn ensure_chrome_env_defaults_home_only_when_missing() {
        {
            let _guard = set_env_var("HOME", None);
            let mut cmd = Command::new("true");
            ensure_chrome_env(&mut cmd);
            assert_eq!(cmd_env(&cmd, "HOME").as_deref(), Some("/tmp"));
        }
        {
            let _guard = set_env_var("HOME", Some("/home/user"));
            let mut cmd = Command::new("true");
            ensure_chrome_env(&mut cmd);
            assert_eq!(cmd_env(&cmd, "HOME"), None);
        }
    }

    #[test]
    fn ensure_chrome_env_defaults_chromium_flags_only_when_missing() {
        {
            let _guard = set_env_var("CHROMIUM_FLAGS", None);
            let mut cmd = Command::new("true");
            ensure_chrome_env(&mut cmd);
            assert_eq!(
                cmd_env(&cmd, "CHROMIUM_FLAGS").as_deref(),
                Some("--no-first-run --no-default-browser-check --disable-gpu")
            );
        }
        {
            let _guard = set_env_var("CHROMIUM_FLAGS", Some("--headless"));
            let mut cmd = Command::new("true");
            ensure_chrome_env(&mut cmd);
            assert_eq!(cmd_env(&cmd, "CHROMIUM_FLAGS"), None);
        }
    }

    #[test]
    fn ensure_chrome_env_sets_fixed_env_vars() {
        let mut cmd = Command::new("true");
        ensure_chrome_env(&mut cmd);
        assert_eq!(
            cmd_env(&cmd, "AGENT_BROWSER_IDLE_TIMEOUT_MS").as_deref(),
            Some("300000")
        );
        assert_eq!(
            cmd_env(&cmd, "AGENT_BROWSER_HUMANIZE").as_deref(),
            Some("human")
        );
        assert_eq!(
            cmd_env(&cmd, "AGENT_BROWSER_NO_UPDATE_CHECK").as_deref(),
            Some("1")
        );
        // The two per-call policies are NOT fixed here: a call declares its own
        // chrome-side deadline and its own recovery allowance.
        assert!(cmd_inherits(&cmd, "AGENT_BROWSER_DEFAULT_TIMEOUT"));
        assert!(cmd_inherits(&cmd, "AGENT_BROWSER_RELAY_REVIVE_SECS"));
        assert!(cmd_inherits(&cmd, "AGENT_BROWSER_NO_AUTO_RECONNECT"));
    }

    #[test]
    fn browser_targeting_env_cannot_divert_a_call_off_the_owners_browser() {
        // A switch nobody enumerated is cleared from the child, because BOTH
        // whole namespaces — chrome-use's own `AGENT_BROWSER_*` switches and its
        // `CHROME_USE_*` ones — are cleared rather than the switches found so
        // far.
        for key in [
            "CI",
            "AGENT_BROWSER_CONFIG",
            "AGENT_BROWSER_BROWSER",
            "AGENT_BROWSER_NO_AUTO_CONNECT",
            "AGENT_BROWSER_PROVIDER",
            "AGENT_BROWSER_EXECUTABLE_PATH",
            "AGENT_BROWSER_PROFILE",
            "AGENT_BROWSER_ENGINE",
            "AGENT_BROWSER_FORCE_LAUNCH",
            "CHROME_USE_PROFILE",
            "CHROME_USE_EXECUTABLE_PATH",
            // The match is case-insensitive, as Windows' environment is.
            "chrome_use_profile",
        ] {
            let _guard = set_env_var(key, Some("hostile"));
            let mut cmd = Command::new("true");
            ensure_chrome_env(&mut cmd);
            assert!(
                cmd_removes(&cmd, key),
                "{key} must be removed from the child's environment"
            );
        }

        // The product's own values are set AFTER the clear, so a hostile value
        // under the same name never survives into chrome-use.
        for (key, expected) in [
            ("AGENT_BROWSER_IDLE_TIMEOUT_MS", "300000"),
            ("AGENT_BROWSER_HUMANIZE", "human"),
            ("AGENT_BROWSER_NO_UPDATE_CHECK", "1"),
            ("AGENT_BROWSER_AUTO_CONNECT", "1"),
            ("CHROME_USE_NO_UPDATE_CHECK", "1"),
        ] {
            let _guard = set_env_var(key, Some("hostile"));
            let mut cmd = Command::new("true");
            ensure_chrome_env(&mut cmd);
            assert_eq!(
                cmd_env(&cmd, key).as_deref(),
                Some(expected),
                "{key} must carry the product's own value, not the inherited one"
            );
        }

        // The relay dir is the one name spared the clear in each namespace: it
        // has to ride the service's environment through to agree with the
        // registered native host.
        for key in ["AGENT_BROWSER_RELAY_DIR", "CHROME_USE_RELAY_DIR"] {
            let mut cmd = Command::new("true");
            {
                let _guard = set_env_var(key, Some("/tmp/relay"));
                ensure_chrome_env(&mut cmd);
            }
            assert!(
                cmd_inherits(&cmd, key),
                "{key} must ride the service's environment through"
            );
        }
    }

    #[test]
    fn recovery_policy_gates_the_relay_self_heal() {
        // Allowed: an inherited auto-reconnect-OFF switch never reaches the
        // child — the namespace clear removes it — and the revive budget is
        // pinned to the number the product's kill rides above.
        let _guard = set_env_var("AGENT_BROWSER_NO_AUTO_RECONNECT", Some("1"));
        let mut allowed = Command::new("true");
        ensure_chrome_env(&mut allowed);
        apply_recovery_env(&mut allowed, CliRecovery::Allowed);
        assert!(cmd_removes(&allowed, "AGENT_BROWSER_NO_AUTO_RECONNECT"));
        assert_eq!(
            cmd_env(&allowed, "AGENT_BROWSER_RELAY_REVIVE_SECS").as_deref(),
            Some("45")
        );

        let mut suppressed = Command::new("true");
        ensure_chrome_env(&mut suppressed);
        apply_recovery_env(&mut suppressed, CliRecovery::Suppressed);
        assert_eq!(
            cmd_env(&suppressed, "AGENT_BROWSER_NO_AUTO_RECONNECT").as_deref(),
            Some("1")
        );
        assert_eq!(
            cmd_env(&suppressed, "AGENT_BROWSER_RELAY_REVIVE_SECS").as_deref(),
            Some("0")
        );
    }

    #[test]
    fn apply_chrome_side_declares_the_deadline_on_every_call() {
        let mut cmd = Command::new("true");
        ensure_chrome_env(&mut cmd);
        apply_chrome_side(&mut cmd, Duration::from_secs(18));
        assert_eq!(
            cmd_env(&cmd, "AGENT_BROWSER_DEFAULT_TIMEOUT").as_deref(),
            Some("18000")
        );

        // Always applied — even for the smallest step, because the declaration
        // is what makes chrome-use's own verdict reach the caller.
        let mut cmd = Command::new("true");
        ensure_chrome_env(&mut cmd);
        apply_chrome_side(&mut cmd, Duration::from_millis(250));
        assert_eq!(
            cmd_env(&cmd, "AGENT_BROWSER_DEFAULT_TIMEOUT").as_deref(),
            Some("250")
        );
    }

    #[test]
    fn kill_bound_rides_above_the_declared_deadline_and_the_recovery() {
        // The arithmetic's one home is `chrome::kill_bound`, and its literal pin
        // lives in `chrome/mod.rs`; here the invariants are the relations, not the
        // numbers. The argument is the domain's own declaration — the deadline
        // handed to chrome-use for a verb mahbot forwards no `--timeout` to.
        let allowed = kill_bound(CHROME_USE_DECLARED_BUDGET, CliRecovery::Allowed);
        let suppressed = kill_bound(CHROME_USE_DECLARED_BUDGET, CliRecovery::Suppressed);
        assert_eq!(allowed, suppressed + RELAY_RECOVERY_BUDGET);
        // The declared deadline is always chrome-use's to hit first: the kill
        // sits KILL_SLACK above it, and the recovery window above that.
        assert_eq!(suppressed, CHROME_USE_DECLARED_BUDGET + KILL_SLACK);
        assert_eq!(
            allowed,
            CHROME_USE_DECLARED_BUDGET + KILL_SLACK + RELAY_RECOVERY_BUDGET
        );
        assert!(kill_bound(Duration::from_secs(6), CliRecovery::Allowed) > Duration::from_secs(6));
        assert!(
            kill_bound(Duration::from_secs(6), CliRecovery::Suppressed) > Duration::from_secs(6)
        );
    }

    /// The child's exit decides the call: a process it leaves behind holding
    /// the output channel must not turn a finished command into a timeout, and
    /// the leftover is reported instead of waited for.
    #[cfg(unix)]
    #[tokio::test]
    async fn child_exit_decides_the_call_when_a_leftover_holds_the_pipe() {
        // `sh` exits as soon as the script ends; the background `sleep` inherits
        // stdout and keeps the pipe open well past the drain grace.
        let run = spawn_cli(CliSpawn {
            path: Path::new("/bin/sh"),
            args: &["-c", "sleep 5 & echo answered"],
            session: None,
            json: true,
            capture_stderr: true,
            cancel_kills: true,
            input: None,
            timeout: CliTimeout::Bounded(Duration::from_secs(10)),
            chrome_side: Duration::from_secs(8),
            recovery: CliRecovery::Suppressed,
        })
        .await;
        let CliRun::Output(out) = run else {
            panic!("expected output");
        };
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "answered");
        assert!(out.leftover_pipes, "the background sleep holds the channel");

        // With no leftover the grace is not consumed and nothing is reported.
        let run = spawn_cli(CliSpawn {
            path: Path::new("/bin/sh"),
            args: &["-c", "echo answered"],
            session: None,
            json: true,
            capture_stderr: true,
            cancel_kills: true,
            input: None,
            timeout: CliTimeout::Bounded(Duration::from_secs(10)),
            chrome_side: Duration::from_secs(8),
            recovery: CliRecovery::Suppressed,
        })
        .await;
        let CliRun::Output(out) = run else {
            panic!("expected output");
        };
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "answered");
        assert!(!out.leftover_pipes);
    }
}
