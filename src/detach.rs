//! Detaching the service from the terminal it was started from.
//!
//! A launch whose standard output is a terminal — a bare `mahbot`, `mahbot paused`, and
//! the installer's own final `exec` — is not the service itself: it opens
//! `<storage root>/daemon.log` and a readiness pipe, starts this same image again in a
//! session of its own (`setsid`: no controlling terminal, none of the shell's streams,
//! not a child of its session), then prints one line and exits. The prompt comes back at
//! once while the service keeps running.
//!
//! The word the child sends back is the instance lock: taken, the service is up; refused,
//! the reason travels back and the launcher exits non-zero — or says the product is
//! already running, where a copy that is up could be reached to raise its window. A child
//! that dies before then closes the pipe, and the launcher points at the log either way.
//!
//! Everything the service prints before the log store is running, a bootstrap failure and
//! a panic message land in that file, which sits beside `error.log` and `update.log` —
//! both keep their own roles.
//!
//! A launch that is not from a terminal is left exactly as it was: a pipe or a redirect,
//! a supervisor, the login registration and the update hand-off all keep their streams
//! and their exit code.

use std::path::Path;

/// Hand the terminal back when this launch came from one: the service is started again
/// in a session of its own, the launcher prints one line and exits, and this function
/// returns only in that service.
///
/// What it returns is the failure that leaves no child to report for itself — no log
/// file, no pipe, no resolvable image, no child — and main's own error path prints it on
/// the terminal the launch still holds, non-zero, never silently. A child that fails in
/// its own start-up says so through the pipe instead, and the launcher prints that.
///
/// Windows is a no-op: that launch is built windowed and already starts console-free.
pub fn detach_from_terminal(root: &Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        // Standard output on a terminal is the launcher itself: it starts the service again
        // and exits, and this process never goes on as the service.
        if std::io::stdout().is_terminal() {
            return relaunch(root);
        }
        // Otherwise this is the service the launcher started — no terminal, the log for its
        // streams, its own session — and it keeps the pipe it reports through.
        if let Some(pipe) = adopted_pipe() {
            *READY.lock().unwrap_poison() = Some(pipe);
        }
        Ok(())
    }

    #[cfg(not(unix))]
    {
        let _ = root;
        Ok(())
    }
}

/// Tell the waiting launcher the service is up — the instance lock is taken.
pub fn report_running() {
    send(RUNNING);
}

/// Tell the waiting launcher the product is already running, and that this launch did
/// what it came to do.
pub fn report_already_running() {
    send(ALREADY_RUNNING);
}

/// Tell the waiting launcher the launch failed, naming the reason it prints.
pub fn report_failure(reason: &anyhow::Error) {
    let reason = format!("{reason:#}").replace('\n', " ");
    send(&format!("{FAILED_PREFIX}{reason}"));
}

/// The service's word that it is up.
const RUNNING: &str = "running";

/// The service's word that a copy is already up.
const ALREADY_RUNNING: &str = "already running";

/// The prefix of the word that carries a launch failure's reason.
const FAILED_PREFIX: &str = "failed: ";

// ── The launcher and the service ─────────────────────────────────────────

#[cfg(unix)]
use std::io::{IsTerminal as _, Read as _, Write as _};
#[cfg(unix)]
use std::os::fd::{AsRawFd as _, FromRawFd as _, RawFd};
#[cfg(unix)]
use std::os::unix::process::CommandExt as _;

#[cfg(unix)]
use anyhow::{Context as _, anyhow};

#[cfg(unix)]
use crate::util::{UnwrapPoison as _, print_stderr, print_stdout};

/// The file a detached service's own output goes to, beside `error.log` and
/// `update.log` in the storage root.
#[cfg(unix)]
const DAEMON_LOG: &str = "daemon.log";

/// The environment variable carrying the readiness pipe's descriptor into the service;
/// read once there and removed, so nothing the service starts inherits a number that
/// means nothing to it.
#[cfg(unix)]
const READY_FD_ENV: &str = "MAHBOT_LAUNCH_READY_FD";

/// How long the launcher waits for the service's word. It arrives when the instance lock
/// does, a moment after the launch; the bound only keeps a start that never gets that far
/// from holding the terminal.
#[cfg(unix)]
const READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// The readiness pipe's writing end, held by the service until it says how the launch
/// went. `None` in a launch that was never detached.
#[cfg(unix)]
static READY: std::sync::Mutex<Option<std::fs::File>> = std::sync::Mutex::new(None);

/// What the launcher heard from the service.
#[cfg(unix)]
enum Ready {
    /// Its word.
    Said(String),
    /// Nothing, with the pipe still open: it is alive but has not reached the lock yet.
    Silent,
    /// Nothing, and the pipe closed: it ended before it could say anything.
    Gone,
}

/// The readiness pipe this process was handed, taken out of the environment.
#[cfg(unix)]
fn adopted_pipe() -> Option<std::fs::File> {
    let value = std::env::var(READY_FD_ENV).ok()?;
    // SAFETY: single-threaded start-up, before the runtime and before any child exists.
    unsafe { std::env::remove_var(READY_FD_ENV) };
    let fd: RawFd = value.parse().ok()?;
    // A descriptor the launcher did not hand over has no business being adopted: a closed
    // one must not be written to, a pipe is the only thing this reports through, and a
    // number something else now holds is not ours.
    // SAFETY: `fstat` on an open descriptor only fills the struct, and `fcntl` with F_GETFD
    // only asks whether it is open.
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if fd < 3
        || unsafe { libc::fcntl(fd, libc::F_GETFD) } == -1
        || unsafe { libc::fstat(fd, &raw mut stat) } == -1
        || stat.st_mode & libc::S_IFMT != libc::S_IFIFO
    {
        return None;
    }
    // Close-on-exec from here on, so a child of this process can never be handed the
    // launcher's pipe.
    // SAFETY: `fcntl` with F_SETFD on an open descriptor.
    unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
    // SAFETY: the descriptor is open, is a pipe and nothing else owns it; the `File` closes
    // it once.
    Some(unsafe { std::fs::File::from_raw_fd(fd) })
}

/// Start the service as a detached child, wait for its word, say how the launch went and
/// exit. Never returns.
#[cfg(unix)]
fn relaunch(root: &Path) -> anyhow::Result<()> {
    let log_path = root.join(DAEMON_LOG);
    std::fs::create_dir_all(root)
        .with_context(|| format!("could not create {}", root.display()))?;
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("could not open {}", log_path.display()))?;

    let (reader, writer) = ready_pipe()?;
    let exe = std::env::current_exe().context("could not resolve the product's own path")?;
    let mut command = std::process::Command::new(&exe);
    command
        .args(std::env::args_os().skip(1))
        .env(READY_FD_ENV, writer.as_raw_fd().to_string())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(
            log.try_clone()
                .context("could not hand the log to the service")?,
        ))
        .stderr(std::process::Stdio::from(log));
    // SAFETY: `setsid` is async-signal-safe and is the only thing run between the fork and
    // the exec — the service leaves the shell's session and process group here, so it
    // holds no controlling terminal and no hangup aimed at the shell reaches it.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    // Nothing waits on it: the service is this launch's continuation, and this process's
    // exit is what makes it nobody's child.
    command
        .spawn()
        .with_context(|| format!("could not start the service from {}", exe.display()))?;
    // The writing end lives in the service now; this copy would keep the launcher from
    // ever reading the end of the pipe.
    drop(writer);
    announce(read_word(reader), &log_path)
}

/// A pipe whose writing end survives the exec into the service and whose reading end does
/// not.
#[cfg(unix)]
fn ready_pipe() -> anyhow::Result<(std::fs::File, std::fs::File)> {
    let mut fds = [0; 2];
    // SAFETY: `pipe` writes two open descriptors into the array.
    if unsafe { libc::pipe(fds.as_mut_ptr()) } == -1 {
        return Err(anyhow!(
            "could not create the readiness pipe: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: both descriptors are open and owned by this process from here on.
    let (reader, writer) = unsafe {
        (
            std::fs::File::from_raw_fd(fds[0]),
            std::fs::File::from_raw_fd(fds[1]),
        )
    };
    // SAFETY: `fcntl` with F_SETFD on an open descriptor.
    unsafe { libc::fcntl(fds[0], libc::F_SETFD, libc::FD_CLOEXEC) };
    Ok((reader, writer))
}

/// The service's word, read within [`READY_TIMEOUT`].
#[cfg(unix)]
fn read_word(mut reader: std::fs::File) -> Ready {
    let deadline = std::time::Instant::now() + READY_TIMEOUT;
    let mut buffer = [0_u8; 1024];
    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        let mut poll = libc::pollfd {
            fd: reader.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one valid descriptor, and the array outlives the call.
        let ready = unsafe { libc::poll(&raw mut poll, 1, remaining_ms(remaining)) };
        if ready == 0 {
            return Ready::Silent;
        }
        if ready < 0 {
            // A hangup or a suspend of this process landing mid-wait interrupts the call,
            // and is neither an answer nor a reason to give up while the bound lasts.
            let interrupted =
                std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted;
            if interrupted && std::time::Instant::now() < deadline {
                continue;
            }
            return Ready::Silent;
        }
        match reader.read(&mut buffer) {
            // The pipe's end: the service ended without a word.
            Ok(0) => return Ready::Gone,
            Ok(read) => {
                return match String::from_utf8_lossy(&buffer[..read]).lines().next() {
                    Some(word) => Ready::Said(word.trim().to_string()),
                    None => Ready::Gone,
                };
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return Ready::Gone,
        }
    }
}

/// A remaining time as `poll`'s millisecond argument, never past its range.
#[cfg(unix)]
fn remaining_ms(remaining: std::time::Duration) -> libc::c_int {
    libc::c_int::try_from(remaining.as_millis()).unwrap_or(libc::c_int::MAX)
}

/// Say how the launch went, in one line on the terminal the launcher held, and end it.
#[cfg(unix)]
fn announce(ready: Ready, log_path: &Path) -> ! {
    let path = log_path.display();
    match ready {
        Ready::Said(word) if word == RUNNING => {
            print_stdout(&format!("MahBot is running — its log is {path}"));
        }
        Ready::Said(word) if word == ALREADY_RUNNING => {
            print_stdout("MahBot is already running.");
        }
        Ready::Said(word) => {
            let reason = word.strip_prefix(FAILED_PREFIX).unwrap_or(word.as_str());
            print_stderr(&format!("MahBot could not start: {reason}"));
            std::process::exit(1);
        }
        // The bound passed with the pipe still open: it is alive but has not taken the lock
        // yet, and a launcher cannot watch further — "starting" is all it can honestly say,
        // and what the service does next is in the log.
        Ready::Silent => print_stdout(&format!("MahBot is starting — its log is {path}")),
        Ready::Gone => {
            print_stderr(&format!("MahBot could not start — its log is {path}"));
            std::process::exit(1);
        }
    }
    std::process::exit(0);
}

/// Write the word and close the pipe: the launcher reads one line and then its end.
#[cfg(unix)]
fn send(word: &str) {
    let mut held = READY.lock().unwrap_poison();
    if let Some(mut pipe) = held.take() {
        let _ = pipe.write_all(format!("{word}\n").as_bytes());
    }
}

/// Nothing to say: no launch detaches from a terminal on this platform.
#[cfg(not(unix))]
fn send(_word: &str) {}
