//! IPC query endpoint for the `mahbot debug` read-only CLI, plus the
//! command-line operation channel.
//!
//! After `multiprocess_wal` was removed, the debug CLI can no longer open a
//! second physical instance of a live store's database (single-process mode
//! holds the flock). Instead the running instance exposes a local IPC endpoint
//! (a Unix-domain socket on Unix, a named pipe on Windows) that accepts
//! read-only SQL queries and returns the rows. The debug CLI connects to it
//! instead of opening the database directly.
//!
//! The same socket carries two request shapes:
//! - the read-only diagnostic query it always did, and
//! - a command-line operation (`mahbot config …` / `mahbot workspace …`),
//!   carried in the SAME frame with an extra `call` field.
//!
//! Protocol (length-prefixed JSON over a local socket stream):
//! - request  → `[u32 LE payload_len][payload]`, payload is JSON `IpcRequest`
//!   (`call` present on the operation shape, absent on a diagnostic query).
//! - response → `[u32 LE payload_len][payload]`, payload is JSON `QueryResponse`
//!   for a diagnostic query and `CallReply` for an operation.
//!
//! The compatibility rule is why the operation rides the old frame instead of a
//! new one: an operation frame carries the magic `store` value [`CALL_STORE`],
//! so a product built before this channel existed parses the frame as an
//! ordinary diagnostic query and refuses it as an unknown store — a clear
//! refusal of the environment class ("this product does not serve that here")
//! rather than a protocol complaint about a frame it cannot read. A new frame
//! type, or any shape the old product could parse as something else, would
//! answer something misleading or nothing at all; a superset of the old frame
//! is the only shape that degrades honestly.
//!
//! The call channel is same-user-only, exactly like the diagnostic one, and it
//! is FAIL-CLOSED: when the socket could not be restricted to the instance's
//! own user, diagnostic queries keep working and operations are refused (the
//! listener computes this and passes it into each connection).
//!
//! An operation is never given up on by either side: the instance answers every
//! call it accepted (it waits for the product to finish starting before running
//! one, and holds a whole-flight guard the graceful drain waits for), and the
//! client reads that answer with no bound at all. The only bound is on the WAIT
//! for the product to come up — a wedged boot is answered rather than hung on.
//! Later operations that carry files the caller keeps, or that hand over without
//! an answer, extend [`CallReply`] — the payload is additive JSON, so neither
//! the framing nor the compatibility rule above has to change for them.
//!
//! The operation surface is the command line's own: a tool is served by name
//! only when one of [`crate::cli`]'s families fronts it, so the channel can
//! never reach the rest of the owner's toolset (see
//! [`crate::cli::fronts_tool`]).
//!
//! Write enforcement is the `PRAGMA query_only=1` guard in
//! [`crate::db::Connection::query_readonly`], which runs and resets the pragma
//! (back to `0`) under a single hold of the connection mutex.

use std::path::{Path, PathBuf};
use std::time::Duration;

use interprocess::local_socket::tokio::prelude::*;
use interprocess::local_socket::{GenericFilePath, ListenerOptions, Name};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tracing::{debug, warn};

use crate::db::{ReadonlyRows, Value};

/// Maximum number of result rows served per IPC query (LIMIT+1 semantics).
pub(crate) const IPC_ROW_LIMIT: usize = 10_000;

/// Socket file name for the debug IPC endpoint (a filesystem socket on Unix).
#[cfg(not(windows))]
const IPC_SOCKET_FILE_NAME: &str = "mahbot-debug.sock";
/// Prefix of the Windows named-pipe name. Named pipes live in a global
/// `\\.\pipe\` namespace with no per-directory scope, and the `GenericFilePath`
/// name type only accepts `\\.\pipe\`-prefixed paths there — the location
/// digest is appended below.
#[cfg(windows)]
const IPC_PIPE_PREFIX: &str = r"\\.\pipe\mahbot-debug-";

/// Total wall-clock bound for the instance-up-but-channel-not-bound retry
/// (lock held before the listener binds during boot / self-update hand-off).
const IPC_BOUND_TIMEOUT_SECS: u64 = 15;

/// [`IPC_BOUND_TIMEOUT_SECS`], overridable with `MAHBOT_IPC_BOUND_TIMEOUT_SECS`
/// (0 = give up after the first attempt) — the same env-override pattern the
/// other bounded waits use, and what lets a test exercise the refusal without
/// waiting out the default.
pub(crate) fn ipc_bound_timeout() -> std::time::Duration {
    crate::util::env_duration_secs("MAHBOT_IPC_BOUND_TIMEOUT_SECS", IPC_BOUND_TIMEOUT_SECS)
}

/// Retry backoff schedule (ms) for the IPC socket-not-yet-bound window.
const IPC_RETRY_BACKOFF_MS: [u64; 6] = [50, 100, 200, 400, 800, 1600];

/// Length (bytes) of the `u32 LE` length prefix on each IPC frame.
const FRAME_LEN: usize = std::mem::size_of::<u32>();

/// Upper bound (bytes) on a single IPC frame payload, enforced on read so a
/// same-user misbehaving client cannot trigger a multi-GiB allocation and OOM
/// the instance. Requests are a few KB; responses are bounded by
/// [`IPC_ROW_LIMIT`], so this is far beyond legitimate use.
const MAX_FRAME_LEN: usize = 64 * 1024 * 1024;

/// Resolve the IPC socket name.
///
/// On Unix this is a filesystem socket file under the storage root. On Windows
/// it is a named pipe scoped to that storage location: the pipe namespace is
/// global, so a machine-wide name would let one instance reach another
/// instance's data. Both the listener and the clients call this so they agree
/// on the endpoint name.
#[must_use]
fn socket_path(storage_root: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        use sha2::{Digest, Sha256};

        // A digest of the location, not the path itself: a pipe name may not
        // contain a backslash, must stay well under the 256-char limit, and is
        // case-insensitive. `canonicalize` gives the real spelling of an
        // existing directory (both sides derive the root from the same
        // `default_config_dir`), falling back to the given path when that
        // directory does not exist yet — no instance can be running there;
        // lowercasing removes case-only differences, and SHA-256 truncated to
        // 16 hex chars keeps the name short and collision-free.
        let resolved =
            std::fs::canonicalize(storage_root).unwrap_or_else(|_| storage_root.to_path_buf());
        let digest = crate::util::hex_string(&Sha256::digest(
            resolved.to_string_lossy().to_lowercase().as_bytes(),
        ));
        PathBuf::from(format!("{IPC_PIPE_PREFIX}{}", &digest[..16]))
    }
    #[cfg(not(windows))]
    {
        storage_root.join(IPC_SOCKET_FILE_NAME)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "t", content = "v", rename_all = "snake_case")]
pub(crate) enum WireValue {
    Null,
    Integer(i64),
    /// serde_json serializes non-finite floats as `null`; `de_real` maps it
    /// back to `NaN` so the IPC path round-trips them like the direct
    /// (no-instance) read path renders them ("NaN").
    Real(#[serde(deserialize_with = "de_real")] f64),
    Text(String),
    /// Base64-encoded blob.
    Blob(String),
}

/// Deserialize a `WireValue::Real` payload: a finite JSON number, or `null`
/// (which serde_json emits for NaN/Inf on the wire) mapped to `NaN`.
fn de_real<'de, D>(d: D) -> std::result::Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = <Option<f64> as serde::Deserialize>::deserialize(d)?;
    Ok(v.unwrap_or(f64::NAN))
}

/// Decode a base64-encoded blob, defaulting to empty on malformed input.
fn decode_blob(b: &str) -> Vec<u8> {
    crate::util::base64_decode(b).unwrap_or_default()
}

impl WireValue {
    pub(crate) fn from_turso(v: &Value) -> Self {
        match v {
            Value::Null => WireValue::Null,
            Value::Integer(i) => WireValue::Integer(*i),
            Value::Real(f) => WireValue::Real(*f),
            Value::Text(s) => WireValue::Text(s.clone()),
            Value::Blob(b) => WireValue::Blob(crate::util::base64_encode(b)),
        }
    }

    pub(crate) fn to_turso(&self) -> Value {
        match self {
            WireValue::Null => Value::Null,
            WireValue::Integer(i) => Value::Integer(*i),
            WireValue::Real(f) => Value::Real(*f),
            WireValue::Text(s) => Value::Text(s.clone()),
            WireValue::Blob(b) => Value::Blob(decode_blob(b)),
        }
    }

    /// Pipe/format display (NULL→empty, Blob→lowercase hex).
    #[must_use]
    pub(crate) fn format(&self) -> String {
        match self {
            WireValue::Null => String::new(),
            WireValue::Integer(i) => i.to_string(),
            WireValue::Real(f) => f.to_string(),
            WireValue::Text(s) => s.clone(),
            WireValue::Blob(b) => crate::util::hex_string(&decode_blob(b)),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct IpcRequest {
    pub store: String, // "core" or "logs" (physical store names), or CALL_STORE
    pub sql: String,
    #[serde(default)]
    pub params: Vec<WireValue>,
    /// Present only on the operation shape of a frame; `None` for a diagnostic
    /// query, and omitted from the wire entirely for one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call: Option<CallRequest>,
}

/// The store name an operation frame carries. Chosen so a product built
/// before this channel parses the frame as a diagnostic query and answers
/// "unknown store" — see the module doc.
pub(crate) const CALL_STORE: &str = "cli";

/// One command-line operation: the tool the frontend addressed and the
/// arguments it passed, plus the values the caller handed over as secrets.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct CallRequest {
    pub tool: String,
    pub args: serde_json::Value,
    /// The values the caller handed over as secrets: they are never echoed
    /// back, never recorded and never reported — the instance removes them from
    /// the answer, from the error text and from the recorded arguments.
    #[serde(default)]
    pub secrets: Vec<String>,
}

/// What became of an operation, as the caller reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CallOutcome {
    /// The tool carried it out.
    Done,
    /// The tool carried it out in part; its own answer names the part that did
    /// not succeed. A failure for the caller, whose answer is printed all the
    /// same.
    Partial,
    /// The product refused it: the tool returned an error and nothing was
    /// carried out.
    Refused,
    /// The product could not take it at all right now (still starting up, or
    /// shutting down) and nothing ran, so a caller MAY retry it safely — it MAY
    /// be served later.
    NotAccepted,
    /// The instance cannot take this operation on and nothing ran — either it
    /// can never serve operations (a socket it could not restrict to the owner,
    /// a boot that failed) or it is not ready yet. The client does not retry it
    /// inside one command invocation; the instance's own sentence says whether a
    /// later attempt may help.
    Unavailable,
}

/// The instance's answer to one operation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct CallReply {
    pub outcome: CallOutcome,
    pub answer: String,
}

impl CallReply {
    /// The operation was carried out.
    pub(crate) fn done(answer: impl Into<String>) -> Self {
        Self {
            outcome: CallOutcome::Done,
            answer: answer.into(),
        }
    }

    /// The operation was carried out in part; `answer` names what did not
    /// succeed.
    pub(crate) fn partial(answer: impl Into<String>) -> Self {
        Self {
            outcome: CallOutcome::Partial,
            answer: answer.into(),
        }
    }

    /// The product refused the operation; nothing ran.
    pub(crate) fn refused(answer: impl Into<String>) -> Self {
        Self {
            outcome: CallOutcome::Refused,
            answer: answer.into(),
        }
    }

    /// The product could not take the operation; nothing ran and a retry is
    /// safe.
    pub(crate) fn not_accepted(answer: impl Into<String>) -> Self {
        Self {
            outcome: CallOutcome::NotAccepted,
            answer: answer.into(),
        }
    }

    /// The product cannot take the operation on right now and nothing ran. Its
    /// own sentence says whether a later attempt may help: a socket it could not
    /// restrict to the owner, or a boot that failed, will never serve one, while
    /// an instance that has not finished starting may.
    pub(crate) fn unavailable(answer: impl Into<String>) -> Self {
        Self {
            outcome: CallOutcome::Unavailable,
            answer: answer.into(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct QueryResponse {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<WireValue>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default)]
    pub truncated: bool,
}

async fn write_frame<S>(stream: &mut S, payload: &[u8]) -> std::io::Result<()>
where
    S: AsyncWrite + Unpin,
{
    let len = u32::try_from(payload.len())
        .map_err(|_| std::io::Error::other("payload too large for IPC frame"))?;
    stream.write_all(&len.to_le_bytes()).await?;
    stream.write_all(payload).await?;
    stream.flush().await
}

async fn read_frame<S>(stream: &mut S) -> std::io::Result<Vec<u8>>
where
    S: AsyncRead + Unpin,
{
    let mut len_buf = [0u8; FRAME_LEN];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_FRAME_LEN {
        return Err(std::io::Error::other(format!(
            "IPC frame too large: {len} bytes (max {MAX_FRAME_LEN})"
        )));
    }
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload).await?;
    Ok(payload)
}

fn response_from_rows(rows: ReadonlyRows) -> QueryResponse {
    QueryResponse {
        columns: rows.columns,
        rows: rows
            .rows
            .into_iter()
            .map(|row| row.iter().map(WireValue::from_turso).collect())
            .collect(),
        error: None,
        truncated: rows.truncated,
    }
}

fn response_error(msg: &str) -> QueryResponse {
    QueryResponse {
        columns: Vec::new(),
        rows: Vec::new(),
        error: Some(msg.to_string()),
        truncated: false,
    }
}

/// Serialize one response — a `QueryResponse` or a `CallReply` — into an IPC
/// frame and write it. (serde_json emits `null` — not an error — for non-finite
/// floats, which `de_real` maps back to NaN, so serialization never fails for
/// the response types used.)
async fn write_json_frame<S, T>(stream: &mut S, message: &T) -> std::io::Result<()>
where
    S: AsyncWrite + Unpin,
    T: serde::Serialize,
{
    let payload = serde_json::to_vec(message).map_err(std::io::Error::other)?;
    write_frame(stream, &payload).await
}

/// The one shape of a malformed-frame answer, so every undecodable request —
/// whether the whole frame failed to decode or the operation store named one
/// that is not there — gets the same actionable wording instead of an
/// EOF/timeout.
fn malformed_response(e: impl std::fmt::Display) -> QueryResponse {
    response_error(&format!("malformed IPC request: {e}"))
}

/// The answer when the socket could not be restricted to the instance's own
/// user: the channel is fail-closed, so operations are refused while diagnostic
/// queries keep working (see the module doc).
const UNRESTRICTED_SOCKET_ANSWER: &str = concat!(
    "the command-line channel could not be restricted to this machine's own user, so ",
    "operations are not served through it (diagnostic queries are unaffected)"
);

async fn handle_connection(
    mut stream: LocalSocketStream,
    log_store: std::sync::Arc<crate::logs::LogStore>,
    serve_calls: bool,
) -> anyhow::Result<()> {
    let payload = read_frame(&mut stream).await?;
    let req: IpcRequest = match serde_json::from_slice(&payload) {
        Ok(req) => req,
        Err(e) => {
            // A malformed request must get an actionable error, not an EOF/timeout.
            let resp = malformed_response(e);
            write_json_frame(&mut stream, &resp).await?;
            return Ok(());
        }
    };

    // The operation shape: the magic store plus its `call`. A frame that names
    // the store and carries nothing to run is malformed — no client writes
    // `store = "cli"` for a diagnostic query — and is answered exactly like a
    // frame that failed to decode.
    if req.store == CALL_STORE {
        let Some(call) = req.call else {
            let resp = malformed_response(format!(
                "store \"{CALL_STORE}\" carries no `call` to operate"
            ));
            write_json_frame(&mut stream, &resp).await?;
            return Ok(());
        };
        // One guard covers the whole flight: it is taken BEFORE the operation is
        // served and lives to the END of this connection's handling, so a
        // graceful drain (or an update hand-off) cannot end the process with an
        // accepted operation unfinished or with its answer unwritten — the
        // caller is owed that answer.
        let _flight = serve_calls.then(crate::cli::instance::hold_flight);
        let reply = if serve_calls {
            crate::cli::instance::execute(call).await
        } else {
            CallReply::unavailable(UNRESTRICTED_SOCKET_ANSWER)
        };
        write_json_frame(&mut stream, &reply).await?;
        return Ok(());
    }

    let conn = match req.store.as_str() {
        "logs" => Some(log_store.conn.clone()),
        "core" => crate::db::DOMAIN_CONN.get().cloned(),
        other => {
            let resp = response_error(&format!(
                "unknown store '{other}' (expected \"core\" or \"logs\")"
            ));
            write_json_frame(&mut stream, &resp).await?;
            return Ok(());
        }
    };

    let resp = match conn {
        Some(conn) => {
            let params: Vec<Value> = req.params.iter().map(WireValue::to_turso).collect();
            match conn.query_readonly(&req.sql, params, IPC_ROW_LIMIT).await {
                Ok(rows) => response_from_rows(rows),
                Err(e) => response_error(&format!("{e}")),
            }
        }
        None => response_error("the requested store is not initialized"),
    };

    write_json_frame(&mut stream, &resp).await?;
    Ok(())
}

/// Spawn the instance-side IPC listener. Call after `DOMAIN_CONN` / `LOG_STORE`
/// are set. Exits (dropping the listener) on shutdown.
pub async fn run_ipc_listener(
    storage_root: &std::path::Path,
    log_store: std::sync::Arc<crate::logs::LogStore>,
) {
    let socket = socket_path(storage_root);

    let name: Name<'static> = match socket.as_path().to_fs_name::<GenericFilePath>() {
        Ok(name) => name.into_owned(),
        Err(e) => {
            warn!(
                error = %e,
                socket = %socket.display(),
                "ipc: cannot build socket name; debug IPC disabled"
            );
            return;
        }
    };

    let mut opts = ListenerOptions::new();
    opts = opts.name(name).try_overwrite(true);

    let listener = match opts.create_tokio() {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            warn!(
                error = %e,
                socket = %socket.display(),
                "ipc: socket already in use (another instance?); debug IPC disabled"
            );
            return;
        }
        Err(e) => {
            warn!(
                error = %e,
                socket = %socket.display(),
                "ipc: cannot bind debug socket; debug IPC disabled"
            );
            return;
        }
    };
    // Restrictive perms: only the instance's user can connect. `ListenerOptions::mode`
    // is NOT used — it calls `fchmod` on the socket fd, which returns
    // EINVAL→Unsupported on macOS UDS; `chmod` via the path works everywhere.
    //
    // Whether that restriction took is what decides whether this listener
    // serves command-line OPERATIONS: the channel can carry a side effect, so
    // an endpoint another user could connect to must not reach the tool layer.
    // Diagnostic queries are read-only and unaffected. On other platforms the
    // endpoint is user-scoped by construction, so operations are served.
    #[cfg(unix)]
    let serve_calls = {
        use std::os::unix::fs::PermissionsExt;
        match std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600)) {
            Ok(()) => true,
            Err(e) => {
                warn!(
                    error = %e,
                    socket = %socket.display(),
                    "ipc: failed to restrict debug socket perms — diagnostics keep working, \
                     command-line operations are refused"
                );
                false
            }
        }
    };
    #[cfg(not(unix))]
    let serve_calls = true;
    debug!(socket = %socket.display(), "ipc: debug query listener running");
    let shutdown = crate::shutdown::shutdown_token();

    loop {
        tokio::select! {
            () = shutdown.cancelled() => {
                debug!("ipc: shutdown — closing debug query listener");
                break;
            }
            accepted = listener.accept() => {
                match accepted {
                    Ok(stream) => {
                        let store = log_store.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, store, serve_calls).await {
                                debug!(error = %e, "ipc: connection handling error");
                            }
                        });
                    }
                    Err(e) => {
                        warn!(error = %e, "ipc: accept error");
                    }
                }
            }
        }
    }
}

/// Client helper: connect to the instance's IPC socket and run a read-only query.
pub(crate) async fn ipc_query(
    storage_root: &Path,
    req: IpcRequest,
) -> anyhow::Result<QueryResponse> {
    let socket = socket_path(storage_root);
    let name = socket
        .as_path()
        .to_fs_name::<GenericFilePath>()
        .map_err(|e| anyhow::anyhow!("instance IPC endpoint not reachable: {e}"))?;

    let result = tokio::time::timeout(Duration::from_secs(5), async {
        let mut stream = LocalSocketStream::connect(name).await?;
        let payload = serde_json::to_vec(&req)?;
        write_frame(&mut stream, &payload).await?;
        let payload = read_frame(&mut stream).await?;
        let resp: QueryResponse = serde_json::from_slice(&payload)?;
        Ok::<QueryResponse, anyhow::Error>(resp)
    })
    .await
    .map_err(|_| anyhow::anyhow!("instance IPC endpoint timed out"))??;

    Ok(result)
}

/// Backoff delay for retry `attempt` (0-based), clamped to the schedule.
pub(crate) fn retry_backoff(attempt: usize) -> Duration {
    Duration::from_millis(IPC_RETRY_BACKOFF_MS[attempt.min(IPC_RETRY_BACKOFF_MS.len() - 1)])
}

/// Refusal wording for the client side when an instance holds the storage
/// location but its debug channel cannot be reached. Callers only get here after
/// the probe reported the location held, so a bare OS error ("connection
/// refused", "no such file") would read as "the store is gone" instead of "an
/// instance is running and its store was deliberately left alone".
fn channel_unreachable(socket: &Path) -> String {
    format!(
        "a mahbot instance is running against this storage location and holds its stores, but its \
         debug channel at {} could not be reached — the live store was left untouched",
        socket.display()
    )
}

/// Bounded-retry async client used by `mahbot debug` when an instance holds the
/// storage location but the IPC socket is not yet bound (it is between
/// lock-acquire and listener-bind, e.g. boot or self-update hand-off). The
/// caller must only use this when [`crate::util::lock::instance_lock_state_settled`]
/// reports the location [`Held`](crate::util::lock::InstanceLockState::Held) —
/// otherwise the retry would mask a genuinely-down instance.
pub(crate) async fn ipc_query_with_wait(
    storage_root: &Path,
    req: &IpcRequest,
) -> anyhow::Result<QueryResponse> {
    let socket = socket_path(storage_root);
    let deadline = std::time::Instant::now() + ipc_bound_timeout();
    let mut attempt = 0usize;
    loop {
        match ipc_query(storage_root, req.clone()).await {
            Ok(resp) => return Ok(resp),
            Err(_) if std::time::Instant::now() < deadline => {
                tokio::time::sleep(retry_backoff(attempt)).await;
                attempt += 1;
            }
            Err(e) => return Err(e.context(channel_unreachable(&socket))),
        }
    }
}

/// Synchronous bounded-retry client used by `bench-openrouter`'s synchronous
/// config-resolution path (which cannot await the async [`ipc_query_with_wait`];
/// the call blocks the current thread). Like the async client, the caller must
/// only use it after the instance-lock probe reported the location held.
pub(crate) fn ipc_query_sync(
    storage_root: &Path,
    req: &IpcRequest,
) -> std::io::Result<QueryResponse> {
    let socket = socket_path(storage_root);
    let deadline = std::time::Instant::now() + ipc_bound_timeout();
    let mut attempt = 0usize;
    loop {
        match ipc_query_sync_once(storage_root, req) {
            Ok(resp) => return Ok(resp),
            Err(_) if std::time::Instant::now() < deadline => {
                std::thread::sleep(retry_backoff(attempt));
                attempt += 1;
            }
            // The bare OS text would read as "the store is gone"; our sentence
            // says the instance is running and its store was left alone. The
            // failure kind is not carried through — no caller distinguishes it.
            Err(e) => {
                return Err(std::io::Error::other(format!(
                    "{}: {e}",
                    channel_unreachable(&socket)
                )));
            }
        }
    }
}

fn ipc_query_sync_once(storage_root: &Path, req: &IpcRequest) -> std::io::Result<QueryResponse> {
    let socket = socket_path(storage_root);
    let name = socket
        .as_path()
        .to_fs_name::<GenericFilePath>()
        .map_err(|e| std::io::Error::other(format!("instance IPC endpoint not reachable: {e}")))?;
    // `Stream::connect` is an associated function on the sync `Stream` trait,
    // not the enum — call it fully-qualified so `Self` resolves to the enum.
    let mut stream =
        <interprocess::local_socket::Stream as interprocess::local_socket::traits::Stream>::connect(
            name,
        )?;
    let payload = serde_json::to_vec(req).map_err(std::io::Error::other)?;
    write_frame_sync(&mut stream, &payload)?;
    let payload = read_frame_sync(&mut stream)?;
    let resp: QueryResponse = serde_json::from_slice(&payload).map_err(std::io::Error::other)?;
    Ok(resp)
}

fn write_frame_sync(stream: &mut impl std::io::Write, payload: &[u8]) -> std::io::Result<()> {
    let len = u32::try_from(payload.len())
        .map_err(|_| std::io::Error::other("payload too large for IPC frame"))?;
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(payload)?;
    stream.flush()
}

fn read_frame_sync(stream: &mut impl std::io::Read) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; FRAME_LEN];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_FRAME_LEN {
        return Err(std::io::Error::other(format!(
            "IPC frame too large: {len} bytes (max {MAX_FRAME_LEN})"
        )));
    }
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;
    Ok(payload)
}

// ── Command-line operation client ───────────────────────────────

/// One attempt's outcome for a command-line operation (see `CallAttempt`).
pub(crate) enum CallAttempt {
    /// The instance answered the operation; its own outcome is in the reply.
    Answered(CallReply),
    /// The instance never took the operation on (it was not delivered, or the
    /// instance answered that it did not accept it): nothing ran, and it may
    /// be served a little later. Carries a COMPLETE caller-facing reason — one
    /// composed by [`call_not_served`] for an OS-level failure, or the
    /// instance's own wording — which the caller prints as this command's
    /// diagnostic; nothing re-wraps it.
    NotServed(String),
    /// The product answered in a form this command cannot read — i.e. it was
    /// built before the channel existed.
    TooOld,
    /// The exchange did not produce a decodable answer: either the frame went
    /// out and the exchange was severed before the product answered, or a reply
    /// arrived that this command cannot read. Which of the two happened is not
    /// known from here, and the caller-facing diagnostic stays neutral about it:
    /// the operation may or may not have been carried out, so it must never be
    /// sent again.
    Severed(anyhow::Error),
}

/// Classify one reply frame to an operation.
///
/// A `CallReply` is the channel's own answer — except for [`NotAccepted`],
/// which is the instance itself saying it did not take the operation on, so it
/// reads as "nothing ran, may be served later" rather than as an answer.
///
/// Anything else is examined for the one pre-channel shape: a product built
/// before this channel parses the operation frame as an ordinary diagnostic
/// query and refuses the magic store, so a `QueryResponse` whose error is
/// `unknown store 'cli'` is [`TooOld`] — a definite "this product predates the
/// channel", not an unreadable/severed answer.
///
/// [`NotAccepted`]: CallOutcome::NotAccepted
/// [`TooOld`]: CallAttempt::TooOld
fn classify_reply(payload: &[u8]) -> CallAttempt {
    if let Ok(reply) = serde_json::from_slice::<CallReply>(payload) {
        return match reply.outcome {
            CallOutcome::NotAccepted => CallAttempt::NotServed(reply.answer),
            _ => CallAttempt::Answered(reply),
        };
    }
    if let Ok(resp) = serde_json::from_slice::<QueryResponse>(payload)
        && resp
            .error
            .as_deref()
            .is_some_and(|e| e.contains(&format!("unknown store '{CALL_STORE}'")))
    {
        return CallAttempt::TooOld;
    }
    CallAttempt::Severed(anyhow::anyhow!(
        "the instance answered in a form this command cannot read"
    ))
}

/// Failure wording for the client side when an operation could not be handed
/// over: the caller only gets here after the lock probe reported the storage
/// location held, so a bare OS error ("connection refused", "no such file")
/// would read as a product that is not there. This says what happened instead —
/// nothing ran, so the operation may be tried again.
///
/// Used for the PRE-WRITE failures only, where the caller's own address of the
/// endpoint is all there is to say. Once a frame is out, the instance's own
/// answer is a complete sentence on its own and is passed through untouched.
fn call_not_served(socket: &Path, detail: &str) -> String {
    format!(
        "the running product's command-line channel at {} did not take the operation on: \
         {detail} — nothing ran, so it may be tried again once the instance is up",
        socket.display()
    )
}

/// One attempt at a command-line operation: connect to the instance's endpoint,
/// write the operation frame, then read the answer with NO timeout.
///
/// The write boundary is the point of no return. An incomplete frame is never
/// executed by the instance (it reads the length prefix and then the payload
/// whole), so a connect, serialization or write failure is `NotServed` —
/// nothing ran and the operation may be retried. Once the frame is out, the
/// answer is awaited with no bound at all — an operation the product is still
/// performing is never given up on — and the frame is never written again.
///
/// A `NotServed` here always carries a COMPLETE caller-facing sentence: the one
/// [`call_not_served`] composes for a pre-write failure, or the instance's own
/// answer text when it said it did not accept the operation. Nothing re-wraps
/// either.
async fn ipc_call_once(storage_root: &Path, call: &CallRequest) -> CallAttempt {
    let socket = socket_path(storage_root);
    let name = match socket.as_path().to_fs_name::<GenericFilePath>() {
        Ok(name) => name,
        Err(e) => {
            return CallAttempt::NotServed(call_not_served(&socket, &e.to_string()));
        }
    };
    let mut stream = match LocalSocketStream::connect(name).await {
        Ok(stream) => stream,
        Err(e) => return CallAttempt::NotServed(call_not_served(&socket, &e.to_string())),
    };
    let req = IpcRequest {
        store: CALL_STORE.to_string(),
        // Never reached on the operation shape — the store branch dispatches on
        // `call` before the SQL is looked at.
        sql: String::new(),
        params: Vec::new(),
        call: Some(call.clone()),
    };
    let payload = match serde_json::to_vec(&req) {
        Ok(payload) => payload,
        Err(e) => return CallAttempt::NotServed(call_not_served(&socket, &e.to_string())),
    };
    if let Err(e) = write_frame(&mut stream, &payload).await {
        return CallAttempt::NotServed(call_not_served(&socket, &e.to_string()));
    }
    match read_frame(&mut stream).await {
        Ok(payload) => classify_reply(&payload),
        Err(e) => CallAttempt::Severed(
            anyhow::Error::new(e).context("the answer could not be read off the exchange"),
        ),
    }
}

/// Client: run one command-line operation against the instance at
/// `storage_root`.
///
/// It retries ONLY while the operation is known not to have run: a connect or
/// frame-write failure (an incomplete frame is never executed, so nothing ran)
/// and a reply whose outcome is [`NotAccepted`] (the instance said so itself).
/// The retry window is [`ipc_bound_timeout()`] with the [`retry_backoff`]
/// schedule; when it expires the result is [`CallAttempt::NotServed`] — always a
/// COMPLETE caller-facing sentence, never a bare OS error: either the one
/// [`call_not_served`] composed for an OS-level failure, or the instance's own
/// wording, returned verbatim as it was received.
///
/// A reply with outcome [`Unavailable`] is returned AT ONCE as
/// [`CallAttempt::Answered`] — never retried: the instance cannot take this
/// operation on and said so itself (whether a later attempt may help is in its
/// own sentence), so another attempt here would only wait for an answer that is
/// already known.
///
/// Once the frame has been written in full, it is NEVER sent again in any
/// circumstance: it reads the answer with NO timeout (an operation the product
/// is still performing is never given up on), and a read error, EOF or an
/// undecodable reply is [`CallAttempt::Severed`] — or [`CallAttempt::TooOld`]
/// when it decodes as the pre-channel shape (a `QueryResponse` whose `error`
/// contains `unknown store 'cli'`).
///
/// This deliberately does NOT go through the diagnostic path's [`ipc_query`]:
/// that one puts a five-second bound on every attempt, which would give up on an
/// operation the instance is still performing.
///
/// [`NotAccepted`]: CallOutcome::NotAccepted
/// [`Unavailable`]: CallOutcome::Unavailable
pub(crate) async fn ipc_call_with_wait(storage_root: &Path, call: &CallRequest) -> CallAttempt {
    let deadline = std::time::Instant::now() + ipc_bound_timeout();
    let mut attempt = 0usize;
    loop {
        let reason = match ipc_call_once(storage_root, call).await {
            CallAttempt::NotServed(reason) => reason,
            other => return other,
        };
        if std::time::Instant::now() >= deadline {
            return CallAttempt::NotServed(reason);
        }
        tokio::time::sleep(retry_backoff(attempt)).await;
        attempt += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two request shapes ride the same frame: a diagnostic query omits
    /// `call` from the wire entirely (an old build's frame is byte-unchanged),
    /// while an operation frame names the magic store and carries its call.
    #[test]
    fn the_two_request_shapes_share_the_frame() {
        let query = serde_json::to_value(IpcRequest {
            store: "logs".to_string(),
            sql: "SELECT 1".to_string(),
            params: Vec::new(),
            call: None,
        })
        .unwrap();
        assert!(
            query.get("call").is_none(),
            "a diagnostic query must omit `call`: {query}"
        );

        let operation = serde_json::to_value(IpcRequest {
            store: CALL_STORE.to_string(),
            sql: String::new(),
            params: Vec::new(),
            call: Some(CallRequest {
                tool: "workspace".to_string(),
                args: serde_json::json!({ "action": "list" }),
                secrets: Vec::new(),
            }),
        })
        .unwrap();
        assert_eq!(operation["store"], CALL_STORE);
        assert_eq!(operation["call"]["tool"], "workspace");
    }

    /// A command-line operation no instance serves must fail with this
    /// command's own sentence, never with the bare OS error underneath it: the
    /// caller only gets this far with a running product, so "No such file or
    /// directory (os error 2)" would read as that product not being there.
    /// `serial(ipc_bound)`: shrinks `MAHBOT_IPC_BOUND_TIMEOUT_SECS`, the bound
    /// the listener-bound e2e tests above retry within.
    #[serial_test::serial(ipc_bound)]
    #[tokio::test]
    async fn call_refusal_names_the_channel_instead_of_the_os_error() {
        // 0 = give up after the first attempt, so the refusal is immediate.
        let _env = crate::util::test::set_env_var("MAHBOT_IPC_BOUND_TIMEOUT_SECS", Some("0"));
        let dir = tempfile::TempDir::new().unwrap();
        let call = CallRequest {
            tool: "workspace".to_string(),
            args: serde_json::json!({ "action": "list" }),
            secrets: Vec::new(),
        };
        match crate::db::ipc::ipc_call_with_wait(dir.path(), &call).await {
            CallAttempt::NotServed(reason) => {
                assert!(
                    reason.contains(&socket_path(dir.path()).display().to_string()),
                    "the refusal must name the endpoint: {reason}"
                );
                assert!(
                    reason.contains("command-line channel"),
                    "the refusal must name the channel that did not take it: {reason}"
                );
            }
            _ => panic!("a call to a non-existent endpoint must be NotServed"),
        }
    }

    /// The client's classification of one reply frame. A `not_accepted` reply
    /// is the retryable "nothing ran" and an `unavailable` one is final; a
    /// product built before the channel answers the operation frame as an
    /// ordinary diagnostic query whose store it refuses, and that exact shape
    /// must read as `TooOld` rather than as an unreadable/severed answer.
    #[test]
    fn replies_classify_as_answered_retryable_or_too_old() {
        let retryable = serde_json::to_vec(&CallReply::not_accepted("still starting")).unwrap();
        assert!(matches!(
            classify_reply(&retryable),
            CallAttempt::NotServed(reason) if reason == "still starting"
        ));

        let final_reply = serde_json::to_vec(&CallReply::unavailable("no channel")).unwrap();
        assert!(matches!(
            classify_reply(&final_reply),
            CallAttempt::Answered(reply) if reply.outcome == CallOutcome::Unavailable
        ));

        // The exact JSON an old build returns for `store = "cli"`.
        let old = serde_json::to_vec(&serde_json::json!({
            "columns": [],
            "rows": [],
            "error": "unknown store 'cli' (expected \"core\" or \"logs\")",
            "truncated": false,
        }))
        .unwrap();
        assert!(
            matches!(classify_reply(&old), CallAttempt::TooOld),
            "a pre-channel refusal is TooOld"
        );

        // Any other diagnostic error is an answer this command cannot read, and
        // the operation may have been carried out.
        let other_error = serde_json::to_vec(&serde_json::json!({
            "columns": [],
            "rows": [],
            "error": "the requested store is not initialized",
            "truncated": false,
        }))
        .unwrap();
        match classify_reply(&other_error) {
            CallAttempt::Severed(e) => assert!(
                e.to_string().contains("cannot read"),
                "the severed answer names the unreadable reply: {e}"
            ),
            _ => panic!("an unreadable reply must be severed"),
        }
    }

    /// The IPC `WireValue` round-trips through `from_turso`/`to_turso` and
    /// renders NULL as empty, integers as decimals, and blobs as lowercase hex.
    #[test]
    fn wire_value_round_trips() {
        use crate::db::Value;
        for (value, wire) in [
            (Value::Integer(42), WireValue::Integer(42)),
            (Value::Real(1.5), WireValue::Real(1.5)),
            (
                Value::Text("hi".to_string()),
                WireValue::Text("hi".to_string()),
            ),
            (Value::Null, WireValue::Null),
        ] {
            assert_eq!(WireValue::from_turso(&value), wire, "from_turso");
            assert_eq!(wire.to_turso(), value, "to_turso");
        }
        // Blob round-trips through the base64 wire form.
        let blob = vec![0x00u8, 0xDE, 0xAD];
        let wire = WireValue::from_turso(&Value::Blob(blob.clone()));
        assert_eq!(wire.to_turso(), Value::Blob(blob));
        assert_eq!(wire.format(), "00dead");
        assert_eq!(WireValue::Null.format(), "");
    }

    /// serde_json emits `null` for non-finite floats; `de_real` must map it back
    /// to `NaN` so a REAL NaN cell round-trips instead of failing deserialization.
    #[test]
    fn wire_value_round_trips_non_finite_real() {
        let payload = serde_json::to_vec(&WireValue::Real(f64::NAN)).unwrap();
        match serde_json::from_slice::<WireValue>(&payload).unwrap() {
            WireValue::Real(f) => {
                assert!(f.is_nan(), "non-finite REAL must round-trip as NaN");
            }
            other => panic!("expected Real, got {other:?}"),
        }
    }

    /// End-to-end: the instance-side listener serves a read-only query over the
    /// local socket, returning column names + rows (the IPC path `mahbot debug`
    /// uses when an instance holds the location). `serial(ipc_bound)`: this
    /// relies on the bounded retry (the listener task may bind after the first
    /// attempt) and must not overlap the tests that shrink that bound through
    /// `MAHBOT_IPC_BOUND_TIMEOUT_SECS`.
    #[serial_test::serial(ipc_bound)]
    #[tokio::test]
    async fn ipc_query_serves_readonly_queries_end_to_end() {
        let (store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        let store_arc = std::sync::Arc::new(store);
        // Insert a row so COUNT is non-zero (proves the query reads real data).
        store_arc
            .conn
            .execute_batch(
                "INSERT INTO logs (timestamp, level, target, message) \
                 VALUES ('2026-01-01T00:00:00Z', 'INFO', 'test', 'hello');",
            )
            .await
            .unwrap();

        let root = dir.path().to_path_buf();
        let listener_root = root.clone();
        let listener = tokio::spawn(async move {
            crate::db::ipc::run_ipc_listener(&listener_root, store_arc).await;
        });

        let req = IpcRequest {
            store: "logs".to_string(),
            sql: "SELECT COUNT(*) FROM logs".to_string(),
            params: Vec::new(),
            call: None,
        };
        let resp = crate::db::ipc::ipc_query_with_wait(&root, &req)
            .await
            .expect("IPC query must reach the listener");
        assert!(resp.error.is_none(), "no error expected: {:?}", resp.error);
        assert_eq!(resp.columns, vec!["COUNT(*)"]);
        assert_eq!(resp.rows, vec![vec![WireValue::Integer(1)]]);
        assert!(!resp.truncated);

        // Engine write enforcement: a write statement is rejected with
        // query_only=ON (defense-in-depth on top of the CLI blocklist).
        let write_req = IpcRequest {
            store: "logs".to_string(),
            sql: "CREATE TABLE nope (id INTEGER)".to_string(),
            params: Vec::new(),
            call: None,
        };
        let write_resp = crate::db::ipc::ipc_query_with_wait(&root, &write_req)
            .await
            .expect("IPC write attempt must reach the listener");
        assert!(
            write_resp.error.is_some(),
            "a write statement must be rejected by query_only=ON"
        );

        // The shared connection's query_only must be reset after the query so
        // the instance's writes are never left disabled.
        let reset_check = crate::db::ipc::ipc_query_with_wait(
            &root,
            &IpcRequest {
                store: "logs".to_string(),
                sql: "SELECT 1".to_string(),
                params: Vec::new(),
                call: None,
            },
        )
        .await
        .expect("subsequent query must succeed after reset");
        assert!(reset_check.error.is_none());

        listener.abort();
        let _ = listener.await;
    }

    /// End-to-end: the same listener serves the OPERATION shape of a frame —
    /// `handle_connection` reads the `call` off `store = CALL_STORE` and routes
    /// it through the instance-side executor, and the client reads the
    /// instance's own `CallReply` back. The tool here is one the owner's own
    /// toolset does not hold, so the operation is refused before any store
    /// access — enough to prove the frame is served as an operation and not as
    /// a diagnostic query. `serial(ipc_bound)`: shares the listener-bound retry
    /// with the diagnostic e2e test above.
    #[serial_test::serial(ipc_bound)]
    #[tokio::test]
    async fn ipc_call_serves_an_operation_frame_end_to_end() {
        // The executor waits for the instance to be ready before serving an
        // operation; this test's instance is a bare listener, so it declares
        // itself up (the same signal `bootstrap_mahbot` sends in production).
        crate::cli::mark_ready();
        let (store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        let root = dir.path().to_path_buf();
        let listener_root = root.clone();
        let listener = tokio::spawn(async move {
            crate::db::ipc::run_ipc_listener(&listener_root, std::sync::Arc::new(store)).await;
        });

        let call = CallRequest {
            tool: "no_such_tool".to_string(),
            args: serde_json::json!({}),
            secrets: Vec::new(),
        };
        match crate::db::ipc::ipc_call_with_wait(&root, &call).await {
            CallAttempt::Answered(reply) => {
                assert_eq!(
                    reply.outcome,
                    CallOutcome::Refused,
                    "answer: {}",
                    reply.answer
                );
                assert!(
                    reply.answer.contains("no_such_tool"),
                    "the refusal must name the tool: {}",
                    reply.answer
                );
            }
            _ => panic!("the operation frame must be served as an operation"),
        }

        listener.abort();
        let _ = listener.await;
    }
}
