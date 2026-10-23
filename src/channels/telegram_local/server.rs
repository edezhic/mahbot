//! The server process: where its pieces live, how it is started, probed,
//! adopted, stopped and kept from piling up on disk.
//!
//! One server per bot, always. A process that outlived the daemon — a crash, a
//! killed terminal, an update handoff that exits without running destructors —
//! is found through its record file. The record's process start identity is what
//! proves it still names the process this product started: a pid alone is reused
//! across a reboot, so acting on it without the start time would signal a
//! stranger. A port proves even less — any process of this user can hold one — and
//! there is no portable way to ask the kernel which process owns a listening
//! socket, so that ownership is not proven: a port is only ever taken from a
//! record whose process has been proven, and a fresh one is chosen immediately
//! before each spawn, which keeps the window in which something else could take it
//! down to the spawn itself. A process that
//! proves itself and answers is adopted; one that proves itself and does not
//! answer is stopped; one that cannot be proven is left alone and its record
//! dropped. Starting a second poller on the same token splits the bot's updates,
//! which is the one failure this module exists to prevent.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// How long a freshly started process may take to answer on its port.
const READY_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a stop waits for the process to go away before it is killed.
const STOP_GRACE: Duration = Duration::from_secs(10);

/// The cheapest possible health request: a connection and one HTTP line with a
/// short bound. A path outside `/bot…` never creates a bot client, so probing
/// costs the platform nothing and cannot log the bot in anywhere.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// The server's own log is bounded here rather than at its 2 GB default.
const LOG_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// Files the server leaves in its own directories are its business, but they are
/// swept by age so they cannot grow without bound. A day is far longer than any
/// ingest takes.
const STATE_FILE_RETENTION: Duration = Duration::from_hours(24);

/// What a running server is, as the record file keeps it across restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Instance {
    pub(super) pid: u32,
    pub(super) port: u16,
    /// The version the artifact reports for itself.
    #[serde(default)]
    pub(super) version: Option<String>,
    /// A fingerprint of the credentials it was started with: a changed
    /// `api_id`/`api_hash` must reach a restarted process, because the server
    /// reads them once at launch and caches a failed login for half an hour.
    #[serde(default)]
    pub(super) credentials: String,
    /// The platform's start time for the process, captured right after spawn.
    /// A pid names whatever holds that number now; this is what says the process
    /// is still the one this record was written for.
    #[serde(default)]
    pub(super) start: u64,
}

/// The process this daemon started, when it started one.
struct Owned {
    child: tokio::process::Child,
    pid: u32,
}

static OWNED: std::sync::OnceLock<tokio::sync::Mutex<Option<Owned>>> = std::sync::OnceLock::new();

fn owned() -> &'static tokio::sync::Mutex<Option<Owned>> {
    OWNED.get_or_init(|| tokio::sync::Mutex::new(None))
}

static ENSURE: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();

fn ensure_lock() -> &'static tokio::sync::Mutex<()> {
    ENSURE.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// `~/.mahbot/telegram-bot-api`.
pub(super) fn root() -> Option<PathBuf> {
    Some(
        crate::config::default_config_dir()
            .ok()?
            .join("telegram-bot-api"),
    )
}

/// The server's working directory: TDLib's own state, kept stable across binary
/// updates so the bot's session survives one.
pub(super) fn state_dir() -> Option<PathBuf> {
    Some(root()?.join("state"))
}

/// Where the server keeps the payloads it downloads (`--files-dir`), one directory
/// per bot token under it.
fn files_dir() -> Option<PathBuf> {
    Some(root()?.join("files"))
}

/// Test hook: the directory a local hand-over must come from, in place of the
/// server's own.
const FILES_DIR_ENV: &str = "MAHBOT_TGAPI_FILES_DIR";

/// The only place a local hand-over may come from: the directory the server is
/// given for its own files. Nothing else, so a malformed or misbehaving answer can
/// neither copy nor delete anything that is not the server's.
pub(super) fn download_root() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os(FILES_DIR_ENV) {
        return Some(PathBuf::from(dir));
    }
    files_dir()
}

/// The directories the server leaves payloads in, swept by age so that a file an
/// interrupted or refused transfer left behind cannot accumulate: downloads land
/// under `--files-dir`, uploads in the temporary one (`--temp-dir`). The working
/// directory beside them — its TDLib database — is never walked.
fn payload_dirs() -> Vec<PathBuf> {
    [files_dir(), temp_dir()].into_iter().flatten().collect()
}

/// The server's own temporary directory, likewise swept by age.
fn temp_dir() -> Option<PathBuf> {
    Some(root()?.join("tmp"))
}

fn log_path() -> Option<PathBuf> {
    Some(root()?.join("server.log"))
}

/// The child's stderr: everything it prints before `--log` is even opened, which
/// is where a launch failure says why. Truncated at every start, so it holds the
/// last launch only and cannot grow — the server's own log has its own bound.
fn stderr_path() -> Option<PathBuf> {
    Some(root()?.join("server.stderr.log"))
}

fn record_path() -> Option<PathBuf> {
    Some(root()?.join("server.json"))
}

/// The installed process, as the record file describes it.
pub(super) fn read_record() -> Option<Instance> {
    let body = std::fs::read_to_string(record_path()?).ok()?;
    serde_json::from_str(&body).ok()
}

/// Write the record atomically: a torn record is one no later start can read,
/// which is the state that lets a second poller start beside a live one.
fn write_record(instance: &Instance) -> Result<(), String> {
    let Some(path) = record_path() else {
        return Err("the server's record path is unavailable".to_string());
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("the server's directory could not be created ({})", e.kind()))?;
    }
    let body = serde_json::to_vec(instance)
        .map_err(|_| "the server's record could not be encoded".to_string())?;
    let tmp = path.with_extension("json.new");
    std::fs::write(&tmp, body)
        .map_err(|e| format!("the server's record could not be written ({})", e.kind()))?;
    std::fs::rename(&tmp, &path)
        .map_err(|e| format!("the server's record could not be placed ({})", e.kind()))
}

fn clear_record() {
    if let Some(path) = record_path() {
        let _ = std::fs::remove_file(path);
    }
}

/// A fingerprint of the credentials a process was started with.
pub(super) fn credentials_fingerprint(api_id: &str, api_hash: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(api_id.as_bytes());
    hasher.update(b":");
    hasher.update(api_hash.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Ask a port whether the server answers there. A connection and one HTTP
/// request line: the answer says "something speaking HTTP is listening", which
/// is all a liveness check needs.
pub(super) async fn probe(port: u16) -> bool {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let Ok(Ok(mut stream)) = tokio::time::timeout(
        PROBE_TIMEOUT,
        tokio::net::TcpStream::connect(("127.0.0.1", port)),
    )
    .await
    else {
        return false;
    };
    let request = b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    if tokio::time::timeout(PROBE_TIMEOUT, stream.write_all(request))
        .await
        .is_err()
    {
        return false;
    }
    let mut head = [0u8; 5];
    matches!(
        tokio::time::timeout(PROBE_TIMEOUT, stream.read_exact(&mut head)).await,
        Ok(Ok(_))
    ) && &head == b"HTTP/"
}

/// A fresh loopback port for the next spawn.
///
/// The listener only reserves the number and is dropped before the server binds it,
/// so this proves the port was free at this moment — not that the server ends up
/// owning it: no portable check names the owner of a listening socket, and the
/// window in which something else could take it is the spawn itself. The server
/// fails to start rather than silently serving on another port, and the record's
/// process identity is what the rest of this module trusts.
fn pick_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))
        .map_err(|e| format!("no loopback port could be reserved ({})", e.kind()))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("the reserved port could not be read ({})", e.kind()))?
        .port();
    drop(listener);
    Ok(port)
}

/// The version the installed artifact reports for itself: the Bot API version
/// it implements, which is what makes a mismatch between what the product
/// expects and what runs diagnosable. `None` when the binary cannot run at all.
pub(super) async fn reported_version() -> Option<String> {
    reported_version_at(&super::install::binary_path()?).await
}

/// The version the binary at `path` reports, or `None` when it cannot run there.
///
/// Also the only proof that a copied file is a usable executable: a checksum-valid
/// artifact can still fail on a missing library, and the answer it prints is the
/// one thing that tells the two apart before it replaces a working copy.
pub(super) async fn reported_version_at(path: &std::path::Path) -> Option<String> {
    let mut command = tokio::process::Command::new(path);
    #[cfg(target_os = "windows")]
    command.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    command
        .arg("--version")
        .stdin(std::process::Stdio::null())
        // The banner is printed through tdlib's plain log, which writes to
        // stderr — capturing only stdout reads nothing at all.
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(10), command.output())
        .await
        .ok()?
        .ok()?;
    if !output.status.success() {
        return None;
    }
    // The banner goes through tdlib's plain log, which writes to stderr.
    let mut text = String::from_utf8_lossy(&output.stderr).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stdout));
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    lines
        .iter()
        .find(|line| line.starts_with("Bot API "))
        .or(lines.first())
        .map(|line| (*line).to_string())
}

/// Make sure a process this daemon owns is running against `credentials`, and
/// answer with its record.
///
/// The order is what keeps the cloud listener untouched while this happens: an
/// absent binary is fetched, an orphan is adopted, and only then is a new
/// process started — on a port nothing else holds, in a stable state directory,
/// with the credentials passed in the environment so they never appear in the
/// process list.
pub(super) async fn ensure(credentials: &str) -> Result<Instance, String> {
    // Serialized: the boot path and the supervisor can ask at the same time, and
    // two answers to "is it running" are how a second process gets started beside
    // the first.
    let _guard = ensure_lock().lock().await;
    if let Some(instance) = read_record()
        && instance.credentials == credentials
        && verify(&instance)
        && probe(instance.port).await
    {
        tracing::info!(
            port = instance.port,
            "telegram-bot-api: adopting the running server"
        );
        return Ok(instance);
    }

    // A process of ours that no longer answers, or one started with the previous
    // credentials: gone before a second one can exist.
    stop().await;
    if let Some(left) = read_record() {
        // `stop` keeps the record only when the process outlived it. Starting
        // beside one would put two pollers on the same bot token — the one outcome
        // this module exists to prevent — so the start waits instead.
        return Err(format!(
            "the previous server process (pid {}) is still running, so a second one was not started",
            left.pid
        ));
    }

    if !super::install::present() {
        super::install::install(super::install::latest().await?).await?;
    }
    let version = if let Some(version) = reported_version().await {
        version
    } else {
        // A copy that checksums right but cannot run is not an installed server:
        // reinstalling is the only repair, and without it the update check could
        // never replace the copy either — it has no version to compare, which is
        // exactly the "stranded on a version that cannot run" state.
        tracing::warn!("telegram-bot-api: the installed server cannot run; reinstalling");
        super::install::install(super::install::latest().await?).await?;
        reported_version()
            .await
            .ok_or_else(|| "the server binary could not be started".to_string())?
    };

    let port = pick_port()?;
    let (pid, start) = spawn(port).await?;
    // Recorded before it is ready: an exit that skips destructors between the
    // spawn and the ready answer would otherwise leave a process no later start
    // could find, and the next one would start a second poller beside it.
    if let Err(reason) = write_record(&Instance {
        pid,
        port,
        version: Some(version.clone()),
        credentials: credentials.to_string(),
        start,
    }) {
        // Without the record the process could outlive this daemon un-found, and
        // the next start would put a second poller beside it.
        stop().await;
        return Err(reason);
    }

    let deadline = tokio::time::Instant::now() + READY_TIMEOUT;
    loop {
        // Both facts, in this order: the process must still be ours when its port
        // answers, otherwise something else took the port and this is not a
        // running server of ours.
        if probe(port).await && exit_status().await.is_none() {
            break;
        }
        if let Some(status) = exit_status().await {
            let reason = startup_failure(&format!("it exited with {status}"));
            stop().await;
            return Err(reason);
        }
        if tokio::time::Instant::now() >= deadline {
            let reason = startup_failure("no answer on its port");
            stop().await;
            return Err(reason);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    Ok(Instance {
        pid,
        port,
        version: Some(version),
        credentials: credentials.to_string(),
        start,
    })
}

/// Start the server process on `port` and answer with its process id and start
/// time.
///
/// Its credentials come from the environment, so `api_hash` never appears in the
/// process list; its three directories are created here, because a missing one
/// must be reported as this side's failure rather than read out of its log.
async fn spawn(port: u16) -> Result<(u32, u64), String> {
    let Some(state) = state_dir() else {
        return Err("the server's working directory is unavailable".to_string());
    };
    let Some(files) = files_dir() else {
        return Err("the server's files directory is unavailable".to_string());
    };
    let Some(log) = log_path() else {
        return Err("the server's log path is unavailable".to_string());
    };
    for dir in [Some(state.clone()), Some(files.clone()), temp_dir()]
        .into_iter()
        .flatten()
    {
        tokio::fs::create_dir_all(&dir).await.map_err(|e| {
            format!(
                "the server's working directory could not be created ({})",
                e.kind()
            )
        })?;
    }
    if let Some(dir) = log.parent() {
        let _ = tokio::fs::create_dir_all(dir).await;
    }
    // Startup failures land here, so the reason can be quoted back to the owner
    // instead of leaving only "it did not start".
    let stderr = match stderr_path().and_then(|path| {
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)
            .ok()
    }) {
        Some(file) => std::process::Stdio::from(file),
        None => std::process::Stdio::null(),
    };

    let Some(root) = root() else {
        return Err("the install directory is unavailable".to_string());
    };
    let Some(binary) = super::install::binary_path() else {
        return Err("the install directory is unavailable".to_string());
    };
    let (api_id, api_hash) = super::credentials();
    let mut command = tokio::process::Command::new(binary);
    #[cfg(target_os = "windows")]
    command.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    command
        // Everything it writes stays under its own root, including whatever a
        // relative path would resolve against: the product's working directory is
        // not its business.
        .current_dir(&root)
        .arg("--local")
        .arg("--http-ip-address")
        .arg("127.0.0.1")
        .arg("--http-port")
        .arg(port.to_string())
        .arg("--dir")
        .arg(&state)
        // Its payloads go to a directory of their own, which is what the sweep
        // ages out and what a local hand-over is confined to.
        .arg("--files-dir")
        .arg(&files)
        .arg("--temp-dir")
        .arg(temp_dir().unwrap_or_else(|| state.clone()))
        .arg("--log")
        .arg(&log)
        .arg("--log-max-file-size")
        .arg(LOG_MAX_BYTES.to_string())
        // Warnings and errors in the log file, so a death after startup can be
        // quoted back instead of reported as "it stopped".
        .arg("--verbosity")
        .arg("1")
        // Read once at launch and cached by the server for half an hour. The
        // environment keeps them out of `ps` output (a same-user process can
        // still read `/proc/<pid>/environ`, which `ps` cannot show).
        .env("TELEGRAM_API_ID", &api_id)
        .env("TELEGRAM_API_HASH", &api_hash)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(stderr)
        .kill_on_drop(false);
    let child = command
        .spawn()
        .map_err(|e| format!("the server binary could not be started ({})", e.kind()))?;
    let pid = child
        .id()
        .ok_or_else(|| "the server reported no process id".to_string())?;
    *owned().lock().await = Some(Owned { child, pid });
    // Captured now, while the pid can only mean this process: it is the fact that
    // later says the record still names the process this side started. A process
    // that cannot be identified could never be proven ours again, and the next
    // start would put a second poller beside it — so it is stopped while it is
    // still ours to stop.
    let Some(start) = process_identity(pid) else {
        let child = owned().lock().await.take();
        if let Some(Owned { mut child, .. }) = child {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
        return Err("the server process could not be identified, so it was stopped".to_string());
    };
    Ok((pid, start))
}

/// The last lines the process wrote, when it died on its own — the only place
/// the real reason is recorded.
fn startup_failure(reason: &str) -> String {
    let tail = [stderr_path(), log_path()]
        .into_iter()
        .flatten()
        .find_map(|path| {
            let bytes = std::fs::read(path).ok()?;
            let start = bytes.len().saturating_sub(2048);
            let text = String::from_utf8_lossy(&bytes[start..]).trim().to_string();
            let line = text.lines().next_back()?.trim().to_string();
            (!line.is_empty()).then_some(line)
        })
        .unwrap_or_default();
    if tail.is_empty() {
        format!("the server did not become ready ({reason})")
    } else {
        format!("the server did not become ready ({reason}): {tail}")
    }
}

/// The child's exit status, when it has one.
///
/// `try_wait` reaps a child that has exited, which is what makes the status
/// final — and what frees its pid for reuse, so nothing may signal that pid
/// afterwards.
async fn exit_status() -> Option<std::process::ExitStatus> {
    let mut guard = owned().lock().await;
    let owned = guard.as_mut()?;
    owned.child.try_wait().unwrap_or_default()
}

/// Whether the recorded process is the one this record was written for.
///
/// The pid alone proves nothing: it is reused across a reboot, and the port it
/// names can be taken by anything. The start time recorded at spawn is what says
/// "this is still the process we started", and a record without one proves
/// nothing at all.
fn verify(instance: &Instance) -> bool {
    instance.start != 0 && process_identity(instance.pid) == Some(instance.start)
}

/// Whether the recorded process is still ours and answering on its port: one this
/// daemon started or one it adopted.
pub(super) async fn alive(instance: &Instance) -> bool {
    // The port, not only the process: a server that is alive but no longer answers
    // is as useless to the bot as one that died, and only the probe tells them
    // apart.
    verify(instance) && probe(instance.port).await
}

/// Boot-time sweep: adopt a server that proves itself and answers, stop one that
/// proves itself and does not, and leave no record pointing at either.
pub(super) async fn adopt_or_reap() {
    let Some(instance) = read_record() else {
        return;
    };
    if !verify(&instance) {
        // Whatever that pid names now, it is not this product's server: it must be
        // neither signalled nor handed the bot's token.
        tracing::debug!(
            pid = instance.pid,
            "telegram-bot-api: the record names no process this product started"
        );
        clear_record();
        return;
    }
    if probe(instance.port).await {
        tracing::info!(
            port = instance.port,
            "telegram-bot-api: adopting the running server"
        );
        return;
    }
    tracing::info!("telegram-bot-api: stopping a server left by a previous run");
    terminate_pid(instance.pid).await;
    clear_record();
}

/// Stop the server: the one this daemon started, or the one it adopted.
pub(super) async fn stop() {
    stop_process().await;
    // The endpoint names a port nothing serves any more: a status that kept it
    // would report a version for a server that is not installed, and a listener
    // could be built against it.
    super::clear_endpoint();
}

/// Stop the process and settle the record; the published endpoint is the caller's.
async fn stop_process() {
    if let Some(Owned { child, pid }) = owned().lock().await.take() {
        // A child this process started: its pid cannot be reused until it is
        // reaped, so this is the one path that may signal without asking the
        // record.
        stop_child(child, pid).await;
        clear_record();
        return;
    }
    let Some(instance) = read_record() else {
        return;
    };
    // Never signal a pid the record cannot prove: after a reboot it may name any
    // process of this user.
    if !verify(&instance) {
        tracing::debug!(
            pid = instance.pid,
            "telegram-bot-api: the record names no process this product started"
        );
        clear_record();
        return;
    }
    terminate_pid(instance.pid).await;
    // The record is what a later start adopts or reaps: one that outlived its own
    // process must not be left pointing at a port nobody answers on.
    if process_alive(instance.pid) {
        tracing::warn!(
            pid = instance.pid,
            "telegram-bot-api: the server outlived its stop"
        );
    } else {
        clear_record();
    }
}

/// Termination, graceful where the platform has a way to ask: Unix gets
/// `SIGTERM`, which the server handles by closing its TDLib state properly.
/// Windows has nothing gentler than `TerminateProcess` — `signal_terminate` is a
/// hard kill there. The timeout and the unconditional kill are the fallback.
async fn stop_child(mut child: tokio::process::Child, pid: u32) {
    // Only a child that has not exited may be signalled: reading its status reaps
    // it, and a reaped pid can already belong to an unrelated process.
    if matches!(child.try_wait(), Ok(None)) {
        signal_terminate(pid);
    }
    if tokio::time::timeout(STOP_GRACE, child.wait())
        .await
        .is_err()
    {
        tracing::warn!("telegram-bot-api: the server did not stop in time");
        let _ = child.start_kill();
        let _ = child.wait().await;
    }
}

/// Stop a process this daemon did not start (an adopted or orphaned server).
async fn terminate_pid(pid: u32) {
    if !process_alive(pid) {
        return;
    }
    signal_terminate(pid);
    if wait_until_gone(pid, STOP_GRACE).await {
        return;
    }
    signal_kill(pid);
    // A start that follows this must not put a second poller beside one that is
    // still shutting down, so the kill is waited out rather than assumed.
    if !wait_until_gone(pid, STOP_GRACE).await {
        tracing::warn!(pid, "telegram-bot-api: the server survived its kill");
    }
}

/// Wait up to `bound` for `pid` to stop existing; `true` when it is gone.
async fn wait_until_gone(pid: u32, bound: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + bound;
    loop {
        if !process_alive(pid) {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// The platform's start time for `pid`, the fact that tells one process apart
/// from a later one holding the same number.
///
/// A pid is reused across a reboot and a port can be taken by anything; the start
/// time is what says "this is still the process we started".
fn process_identity(pid: u32) -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        // `comm` (field 2) is a parenthesised name that may itself contain spaces
        // and parentheses, so everything after the last `)` is positional.
        let rest = stat.rsplit_once(')')?.1;
        // The remainder starts at `state` (field 3), which makes `starttime`
        // (field 22) the twentieth whitespace-separated item.
        rest.split_whitespace().nth(19)?.parse().ok()
    }
    #[cfg(target_os = "macos")]
    {
        let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
        let size = libc::c_int::try_from(std::mem::size_of::<libc::proc_bsdinfo>()).ok()?;
        let read = unsafe {
            libc::proc_pidinfo(
                libc::c_int::try_from(pid).ok()?,
                libc::PROC_PIDTBSDINFO,
                0,
                std::ptr::addr_of_mut!(info).cast::<libc::c_void>(),
                size,
            )
        };
        if read != size {
            return None;
        }
        Some(info.pbi_start_tvsec * 1_000_000 + info.pbi_start_tvusec)
    }
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
        use windows_sys::Win32::System::Threading::{
            GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };
        let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if handle == 0 {
            return None;
        }
        let mut creation = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut exit = creation;
        let mut kernel = creation;
        let mut user = creation;
        let asked =
            unsafe { GetProcessTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user) };
        unsafe { CloseHandle(handle) };
        let start = (u64::from(creation.dwHighDateTime) << 32) | u64::from(creation.dwLowDateTime);
        (asked != 0).then_some(start)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        let _ = pid;
        None
    }
}

/// Whether `pid` names a live process.
fn process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(unix)]
    {
        // `kill(pid, 0)` asks the platform whether the process exists without
        // touching it; EPERM still means it exists.
        let alive = unsafe { libc::kill(i32::try_from(pid).unwrap_or(i32::MAX), 0) } == 0;
        alive || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };
        let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if handle == 0 {
            return false;
        }
        unsafe { CloseHandle(handle) };
        true
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        false
    }
}

/// Ask the process to stop.
fn signal_terminate(pid: u32) {
    #[cfg(unix)]
    if let Ok(pid) = i32::try_from(pid) {
        unsafe { libc::kill(pid, libc::SIGTERM) };
    }
    #[cfg(not(unix))]
    signal_kill(pid);
}

/// Stop the process unconditionally.
fn signal_kill(pid: u32) {
    #[cfg(unix)]
    if let Ok(pid) = i32::try_from(pid) {
        unsafe { libc::kill(pid, libc::SIGKILL) };
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_TERMINATE, TerminateProcess,
        };
        let handle = unsafe { OpenProcess(PROCESS_TERMINATE, 0, pid) };
        if handle != 0 {
            unsafe {
                TerminateProcess(handle, 1);
                CloseHandle(handle);
            }
        }
    }
}

/// Remove everything the server owns — the binary, its state, the payloads it
/// left and its record. Used when the owner goes back to the cloud: none of it
/// means anything there, and the owner is told so before confirming.
pub(super) async fn remove_state() {
    stop().await;
    let Some(root) = root() else { return };
    match tokio::fs::remove_dir_all(&root).await {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => tracing::warn!(error = %e, "telegram-bot-api: could not remove its state"),
    }
}

/// Keep the server's own disk use bounded: the payloads it leaves behind are
/// dropped by age. Its log needs nothing here — `--log-max-file-size` is the
/// server's own bound, and rewriting a log a live process holds open would
/// corrupt it.
///
/// Only its download directory (`--files-dir`) and its temporary one
/// (`--temp-dir`) are walked ([`payload_dirs`]): its TDLib database is never
/// touched.
pub(super) async fn sweep_state_files() {
    for dir in payload_dirs() {
        let mut pending = vec![dir];
        while let Some(dir) = pending.pop() {
            let Ok(mut entries) = tokio::fs::read_dir(&dir).await else {
                continue;
            };
            while let Ok(Some(entry)) = entries.next_entry().await {
                let Ok(kind) = entry.file_type().await else {
                    continue;
                };
                if kind.is_dir() {
                    pending.push(entry.path());
                    continue;
                }
                let old = entry
                    .metadata()
                    .await
                    .and_then(|meta| meta.modified())
                    .ok()
                    .and_then(|modified| modified.elapsed().ok())
                    .is_some_and(|age| age > STATE_FILE_RETENTION);
                if old {
                    let _ = tokio::fs::remove_file(entry.path()).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The record is what a later start acts on, and only the process start time
    /// tells its pid apart from whatever holds that number now — the difference
    /// between signalling our server and signalling a stranger.
    #[test]
    fn a_record_proves_itself_only_with_the_start_time_of_its_process() {
        let pid = std::process::id();
        let start = process_identity(pid).expect("the current process has a start time");
        let record = Instance {
            pid,
            port: 1,
            version: None,
            credentials: String::new(),
            start,
        };
        assert!(verify(&record));

        // A record written before the start time was kept proves nothing: a pid is
        // reused across a reboot, so matching numbers say nothing about the process.
        assert!(!verify(&Instance {
            start: 0,
            ..record.clone()
        }));
        // A start time that belongs to another process is not this one.
        assert!(!verify(&Instance {
            start: start + 1,
            ..record
        }));
    }

    /// The update path places a downloaded artifact only after it has run it: a
    /// checksum-valid file that cannot start must be detected before it replaces a
    /// working copy, because the update check would have no version to compare and
    /// could never repair the copy it stranded.
    #[tokio::test]
    async fn an_artifact_that_cannot_run_reports_no_version() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("telegram-bot-api");
        tokio::fs::write(&path, b"not an executable")
            .await
            .expect("write");
        assert_eq!(reported_version_at(&path).await, None);
    }
}
