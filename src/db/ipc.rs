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
//! Where the window can be hidden (macOS and Windows, see `crate::window_requests`),
//! the same socket carries a third request shape next to the two it always did — the
//! read-only diagnostic query and the command-line operation (any family [`crate::cli`]
//! publishes — `mahbot config …`, `mahbot send …`) that rides the query's SAME frame with
//! an extra `call` field:
//!
//! - an operation names its family's protocol value in `store` and carries its `call`;
//! - a second launch of the product names the window store to bring the window up, and is
//!   answered when the interface took it.
//!
//! Protocol (length-prefixed JSON over a local socket stream):
//! - request  → `[u32 LE payload_len][payload]`, payload is JSON `IpcRequest`
//!   (`call` present on the operation shape, absent on a diagnostic query and on the
//!   window raise).
//! - response → `[u32 LE payload_len][payload]`, payload is JSON `QueryResponse`
//!   for a diagnostic query and a window raise, `CallReply` for an operation.
//!
//! The compatibility rule is why the operation rides the old frame instead of a
//! new one: an operation frame carries a magic `store` value — the value of the
//! command-line surface generation its family declares ([`CALL_STORE`] /
//! [`CALL_STORE_BASE`]), never a store name — so a product that does not know it
//! parses the frame as an ordinary diagnostic query and refuses it as an unknown
//! store: a clear refusal of the environment class ("this product does not serve
//! that here") rather than a protocol complaint about a frame it cannot read. A
//! new frame type, or any shape the old product could parse as something else,
//! would answer something misleading or nothing at all; a superset of the old
//! frame is the only shape that degrades honestly.
//!
//! The value belongs to the FAMILY, because a family is what an older product
//! either serves or would misanswer: every family the channel has already served
//! carries the generation that served it, so it keeps working against a product
//! one build behind, and only a family added since carries a value that product
//! does not know. This build serves every generation at once ([`CALL_STORES`])
//! while its frontend sends the one its family declares, so the degradation runs
//! both ways: a product older than the frontend does not know what the frontend
//! sends for such a family, and a frontend older than the product sends the
//! value of the generation it was built in.
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
//! The two paths are bounded apart. A read-only query is bounded by
//! [`crate::db::read_limit`]: one command-wide budget that covers the wait to
//! reach the instance, the instance's answer and the retries alike, and whose
//! own failure is what a read that ran out is reported with. The operation
//! channel keeps its own window ([`ipc_bound_timeout`]) and its reply read is
//! deliberately unbounded (above), so widening one never changes the other.
//!
//! The operation surface is the command line's own: a tool is served by name
//! only when one of [`crate::cli`]'s families fronts it, so the channel can
//! never reach the rest of the owner's toolset (see
//! [`crate::cli::fronts_tool`]).
//!
//! Write enforcement for the read-only query channel is the store's own, and it
//! is the same per store. The statement forms a read-only connection refuses
//! ([`crate::db::reader_refusal`] — transaction control, `ATTACH`/`DETACH`, and
//! the pragmas that write without being classified as such) are refused on
//! whichever path serves the query: the reader entry refuses them on a reader,
//! and [`crate::db::Connection::query_readonly`] refuses them on the
//! writing-connection fallback. On a store that serves its reads (the core
//! store) the query then runs on a reader connection: one created read-only
//! (`PRAGMA query_only = 1`, kept on for its whole life), so nothing has to be
//! set, reset or carried over. On a store that does not (the log store) it runs
//! on the writing connection's path in
//! [`crate::db::Connection::query_readonly`], which sets the pragma, runs the
//! query and resets it under a single hold of that connection's mutex. Which of
//! the two serves a query is not the client's business.
//! The client side of the inspection path (`mahbot debug`, the `mahbot_debug`
//! tool) applies its own validator on top
//! ([`crate::db::debug::validate_read_only`]), which takes the same decision.

use std::path::{Path, PathBuf};
use std::time::Duration;

use interprocess::local_socket::tokio::prelude::*;
use interprocess::local_socket::traits::Stream as _;
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

/// Bounds the retry window of the command-line OPERATION channel
/// (`mahbot config …`, `mahbot send …`) while an instance holds the lock but
/// has not bound its listener yet (boot / self-update hand-off).
///
/// Every failed attempt that did not reach the instance is retried while the
/// window has time left — the window is not restricted to transport failures —
/// and an operation that reached the instance is never retried at all
/// ([`ipc_call_with_wait`]). The read-only query paths answer to
/// [`crate::db::read_limit`] instead — its own knob, so widening one never
/// changes the other's responsiveness.
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

    /// The cell as [`crate::db::format_cell`] prints it in a pipe table —
    /// delegates through [`Self::to_turso`], so the rules stay in one place.
    #[must_use]
    pub(crate) fn format(&self) -> String {
        crate::db::format_cell(&self.to_turso())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct IpcRequest {
    pub store: String, // "core" or "logs" (physical store names), or the call's protocol value (CALL_STORES)
    pub sql: String,
    #[serde(default)]
    pub params: Vec<WireValue>,
    /// Present only on the operation shape of a frame; `None` for a diagnostic
    /// query, and omitted from the wire entirely for one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call: Option<CallRequest>,
}

/// The CURRENT operation-frame PROTOCOL value: what a family this build adds
/// carries, so a product built before it refuses the frame as an unknown store —
/// the environment's class, and the truth (nothing ran) — instead of answering a
/// call it cannot carry out as a failed operation. Bump it when a family is
/// added that an older product would misanswer that way.
pub(crate) const CALL_STORE: &str = "cli2";

/// The value every family the channel has already served carries: any product
/// that knows the operation channel at all serves it, so those families keep
/// working against a product one build behind instead of being told to restart.
pub(crate) const CALL_STORE_BASE: &str = "cli";

/// Every protocol value this build serves — one per generation of the command
/// line's surface.
pub(crate) const CALL_STORES: &[&str] = &[CALL_STORE, CALL_STORE_BASE];

/// The store value a second launch's frame names: not a store, and nothing is
/// queried — the frame asks the interface to bring its window up (see
/// [`crate::window_requests`]) and is answered when the interface took it. A
/// product that does not know the value refuses the frame as an unknown store,
/// and so does one that cannot show a window, which the caller reads as "no
/// window came up".
#[cfg(any(target_os = "macos", target_os = "windows"))]
const SHOW_WINDOW_STORE: &str = "window";

/// The answer a window raise gets when the socket could not be restricted to the
/// instance's own user: the fail-closed rule the operations follow (see the module
/// doc), stated for this request.
#[cfg(any(target_os = "macos", target_os = "windows"))]
const SHOW_WINDOW_UNRESTRICTED_ANSWER: &str = concat!(
    "the command-line channel could not be restricted to this machine's own user, so a window ",
    "is not brought up through it"
);

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
    /// The directory the produced files must go into, resolved on the machine the
    /// command runs on by the caller that named it. It is the command line's own
    /// operand and never a tool argument, so no tool sees it in `args`; only the
    /// command line's producing families send one, and the tools that write files
    /// are the only readers (see [`crate::agent::CURRENT_TOOL_OUTPUT_DIR`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_dir: Option<String>,
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
    /// a boot that failed), it fronts no family for the named tool, or it is not
    /// ready yet. The client does not retry it inside one command invocation; the
    /// instance's own sentence says whether a later attempt may help.
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

/// The answer to a request that was taken on, with nothing to report.
#[cfg(any(target_os = "macos", target_os = "windows"))]
fn response_ok() -> QueryResponse {
    QueryResponse {
        columns: Vec::new(),
        rows: Vec::new(),
        error: None,
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

    // A second launch of the product: its one frame asks the interface to bring the window up.
    // Served only on a channel restricted to the instance's own user, and taken only while the
    // product can still show a window — a launch that finds a drain or an exit under way is
    // refused the way every second launch was, since no window will come up. The answer says
    // the request was taken, not that the window is up.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    if req.store == SHOW_WINDOW_STORE {
        let resp = if !serve_calls {
            response_error(SHOW_WINDOW_UNRESTRICTED_ANSWER)
        } else if crate::shutdown::aborting() {
            response_error("the product is shutting down, so no window came up")
        } else if !crate::window_requests::show() {
            response_error("the interface is not running, so no window came up")
        } else {
            response_ok()
        };
        write_json_frame(&mut stream, &resp).await?;
        return Ok(());
    }

    // The operation shape: the magic store plus its `call`. A frame that names
    // one of the protocol values this build serves and carries nothing to run is
    // malformed — no client writes an operation store for a diagnostic query —
    // and is answered exactly like a frame that failed to decode.
    if CALL_STORES.contains(&req.store.as_str()) {
        let Some(call) = req.call else {
            let resp = malformed_response(format!(
                "store \"{}\" carries no `call` to operate",
                req.store
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

/// One attempt at a read-only query, in the shape the retry loop can act on.
///
/// The distinction that matters is delivery: an attempt that never reached the
/// instance sent nothing, so it is the only one worth sending again — resending
/// one the instance took would run a second copy of the same work.
enum Attempt {
    /// The instance answered.
    Answered(QueryResponse),
    /// The connection was never established: nothing reached the instance.
    NotDelivered(String),
    /// The frame went out and no answer came back inside the wait.
    NoAnswer,
    /// The frame went out and the exchange broke before a decodable answer.
    Broken(String),
}

/// One attempt: connect, send, and wait what is left of the command's budget for
/// the answer.
///
/// The connect is deliberately OUTSIDE the wait: a connection that is refused is
/// "nothing was delivered" (retryable), which the timeout wrapper could not tell
/// apart from an instance that took the query and went quiet. The wait is read
/// AFTER the connect, so an attempt can never run past the budget by the
/// connect's own duration.
async fn ipc_query_once(
    root: &Path,
    req: &IpcRequest,
    read: crate::db::read_limit::Read<'_>,
) -> Attempt {
    let socket = socket_path(root);
    let name = match socket.as_path().to_fs_name::<GenericFilePath>() {
        Ok(name) => name,
        Err(e) => return Attempt::NotDelivered(e.to_string()),
    };
    let mut stream = match LocalSocketStream::connect(name).await {
        Ok(stream) => stream,
        Err(e) => return Attempt::NotDelivered(e.to_string()),
    };
    let payload = match serde_json::to_vec(req) {
        Ok(payload) => payload,
        Err(e) => return Attempt::Broken(e.to_string()),
    };
    // A budget spent while the connect ran cannot be waited on at all, and a
    // zero wait is not an immediate failure for the timer below (its own first
    // poll can let an answer through), so it is reported as the limit outright —
    // as `NoAnswer`, because a connect succeeded and the query must not be sent
    // again.
    let wait = read.remaining();
    if wait.is_zero() {
        return Attempt::NoAnswer;
    }
    match tokio::time::timeout(wait, async {
        write_frame(&mut stream, &payload).await?;
        let p = read_frame(&mut stream).await?;
        Ok::<QueryResponse, anyhow::Error>(serde_json::from_slice(&p)?)
    })
    .await
    {
        // A timeout that lands during the write is conservatively `NoAnswer`:
        // nothing is resent after a possible delivery.
        Err(_elapsed) => Attempt::NoAnswer,
        Ok(Err(e)) => Attempt::Broken(format!("{e:#}")),
        Ok(Ok(resp)) => Attempt::Answered(resp),
    }
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

/// [`channel_unreachable`] with the transport's own error appended — the shape
/// every terminal channel failure takes, whichever branch produced it.
fn unreachable(socket: &Path, detail: &str) -> anyhow::Error {
    anyhow::anyhow!("{}: {detail}", channel_unreachable(socket))
}

/// The failure for a budget that is spent before another attempt can be made:
/// while only never-delivered attempts ran, the instance is what could not be
/// reached; before the first attempt there is no such history, so the limit's own
/// failure (`spent`) stands — its smallest value means "fail now".
fn not_attempted(undelivered: Option<&str>, spent: anyhow::Error, socket: &Path) -> anyhow::Error {
    undelivered.map_or(spent, |detail| unreachable(socket, detail))
}

/// What a finished attempt means to the retry loop: one classification for both
/// clients, so they cannot drift on which failures are sent again or on what a
/// failure says.
enum Outcome {
    /// The instance answered.
    Answer(QueryResponse),
    /// The instance never took it: send the next attempt after the backoff. The
    /// transport's own error is carried along, for the case where the budget runs
    /// out before an attempt can be made after it.
    Resend(String),
    /// Report this; the attempt is not sent again.
    Report(anyhow::Error),
}

/// Classify one attempt for the retry loop that made it, framing everything but
/// the limit's own failure with [`channel_unreachable`]: the caller only gets
/// here after the probe reported the location held, so a bare OS text would read
/// as "the store is gone" instead of "an instance is running and its store was
/// deliberately left alone".
fn classify(attempt: Attempt, read: crate::db::read_limit::Read<'_>, socket: &Path) -> Outcome {
    match attempt {
        // An answer the budget was spent on is not an answer, however well the
        // exchange itself went.
        Attempt::Answered(resp) => {
            if read.remaining().is_zero() {
                Outcome::Report(read.timeout())
            } else {
                Outcome::Answer(resp)
            }
        }
        // A delivery that never happened is retryable only while there is
        // budget left to send it again.
        Attempt::NotDelivered(e) => {
            if read.remaining().is_zero() {
                Outcome::Report(unreachable(socket, &e))
            } else {
                Outcome::Resend(e)
            }
        }
        // The frame went out: the instance took the query, so it is never sent
        // again — this is the limit's own failure, not an unreachable channel.
        Attempt::NoAnswer => Outcome::Report(read.timeout()),
        Attempt::Broken(e) => Outcome::Report(unreachable(socket, &e)),
    }
}

/// Bounded-retry async client used by `mahbot debug` when an instance holds the
/// storage location but the IPC socket is not yet bound (it is between
/// lock-acquire and listener-bind, e.g. boot or self-update hand-off). The
/// caller must only use this when [`crate::util::lock::instance_lock_state_settled`]
/// reports the location [`Held`](crate::util::lock::InstanceLockState::Held) —
/// otherwise the retry would mask a genuinely-down instance.
///
/// The whole exchange runs under the command's read budget
/// ([`crate::db::read_limit`]): every attempt waiting what the budget has left —
/// never a per-statement window — and a spent budget reported as the limit's own
/// failure. Only an attempt that never reached the instance is sent again; one
/// the instance took is never resent, so a slow instance is waited out rather
/// than misreported as an unreachable channel.
///
/// The caller owns the limit's process-ending half ([`crate::db::read_limit::Read::arm`]),
/// which is what covers a wait that never comes back — armed ONCE for the route
/// that makes the reads, not per attempt or per round trip.
pub(crate) async fn ipc_query_with_wait(
    storage_root: &Path,
    req: &IpcRequest,
    read: crate::db::read_limit::Read<'_>,
) -> anyhow::Result<QueryResponse> {
    let socket = socket_path(storage_root);
    // The transport error of the last attempt that never reached the instance, for
    // the case where the budget runs out before another attempt can be made.
    let mut undelivered: Option<String> = None;
    let mut attempt = 0usize;
    loop {
        // Checked per attempt: a budget spent while the previous attempt was in
        // flight or backing off fails here, before another query goes out.
        if let Err(spent) = read.check() {
            return Err(not_attempted(undelivered.as_deref(), spent, &socket));
        }
        match classify(ipc_query_once(storage_root, req, read).await, read, &socket) {
            Outcome::Answer(resp) => return Ok(resp),
            Outcome::Resend(e) => {
                undelivered = Some(e);
                // Never sleep past the budget: the next attempt's own check
                // ends the read at the limit, not somewhere after it.
                tokio::time::sleep(retry_backoff(attempt).min(read.remaining())).await;
                attempt += 1;
            }
            Outcome::Report(e) => return Err(e),
        }
    }
}

/// Synchronous bounded-retry client used by `bench-openrouter`'s synchronous
/// config-resolution path (which cannot await the async [`ipc_query_with_wait`];
/// the call blocks the current thread). Like the async client, the caller must
/// only use it after the instance-lock probe reported the location held.
///
/// The whole exchange runs under the command's read budget
/// ([`crate::db::read_limit`]) exactly like the async client's: every attempt
/// checked at the top and waiting what is left, and a spent budget failing before
/// anything is sent. Only an attempt that never reached the instance is sent
/// again.
///
/// As on the async path, the caller owns the process-ending half
/// ([`crate::db::read_limit::Read::arm`]) — armed once for the route. Here it is
/// also the only end a wedged exchange has on a platform whose local sockets
/// carry no timeout (Windows named pipes).
pub(crate) fn ipc_query_sync(
    storage_root: &Path,
    req: &IpcRequest,
    read: crate::db::read_limit::Read<'_>,
) -> anyhow::Result<QueryResponse> {
    let socket = socket_path(storage_root);
    let mut undelivered: Option<String> = None;
    let mut attempt = 0usize;
    loop {
        // A budget spent before another query can go out fails here (see the
        // async client).
        if let Err(spent) = read.check() {
            return Err(not_attempted(undelivered.as_deref(), spent, &socket));
        }
        match classify(ipc_query_sync_once(storage_root, req, read), read, &socket) {
            Outcome::Answer(resp) => return Ok(resp),
            Outcome::Resend(e) => {
                undelivered = Some(e);
                // Never sleep past the budget (see the async client).
                std::thread::sleep(retry_backoff(attempt).min(read.remaining()));
                attempt += 1;
            }
            Outcome::Report(e) => return Err(e),
        }
    }
}

fn ipc_query_sync_once(
    storage_root: &Path,
    req: &IpcRequest,
    read: crate::db::read_limit::Read<'_>,
) -> Attempt {
    let mut stream = match connect_sync(storage_root) {
        Ok(stream) => stream,
        Err(e) => return Attempt::NotDelivered(e),
    };
    // Read AFTER the connect, like the async client's wait, so an attempt cannot
    // run past the budget by the connect's own duration; and a budget spent here
    // is reported as `NoAnswer` rather than waited on (a zero timeout means "no
    // timeout" to the OS, and the query must not be sent again after a connect).
    let wait = read.remaining();
    if wait.is_zero() {
        return Attempt::NoAnswer;
    }
    // Best-effort bound on the blocking exchange: the socket timeouts make a
    // read that would never return come back as the limit's own failure. Best
    // effort in both directions — the send/recv side may ignore them, and
    // Windows named pipes carry no timeout at all (`no_timeouts` error); either
    // way the read-limit watchdog the caller armed is what ends a wedged read.
    let _ = stream.set_recv_timeout(Some(wait));
    let _ = stream.set_send_timeout(Some(wait));
    match exchange_sync(&mut stream, req) {
        // A send or a read that ran out of time is the same event the async
        // client's timeout wrapper sees, and gets the same answer: the limit, not
        // a channel that could not be reached.
        Err(e) if timed_out(&e) => Attempt::NoAnswer,
        Err(e) => Attempt::Broken(e.to_string()),
        Ok(payload) => match serde_json::from_slice(&payload) {
            Ok(resp) => Attempt::Answered(resp),
            Err(e) => Attempt::Broken(e.to_string()),
        },
    }
}

/// Connect to the running instance's endpoint.
fn connect_sync(storage_root: &Path) -> Result<interprocess::local_socket::Stream, String> {
    let socket = socket_path(storage_root);
    let name = socket
        .as_path()
        .to_fs_name::<GenericFilePath>()
        .map_err(|e| e.to_string())?;
    // `Stream::connect` is an associated function on the sync `Stream` trait,
    // not the enum — call it fully-qualified so `Self` resolves to the enum.
    <interprocess::local_socket::Stream as interprocess::local_socket::traits::Stream>::connect(
        name,
    )
    .map_err(|e| e.to_string())
}

/// One request frame out, one answer frame in — the whole of the sync channel's
/// exchange, shared by the query client and the second-launch request.
fn exchange_sync(
    stream: &mut interprocess::local_socket::Stream,
    req: &IpcRequest,
) -> std::io::Result<Vec<u8>> {
    let payload = serde_json::to_vec(req).map_err(std::io::Error::other)?;
    write_frame_sync(stream, &payload)?;
    read_frame_sync(stream)
}

/// True for the error a socket timeout surfaces as on this platform (both kinds:
/// interprocess reports a receive timeout as one or the other).
fn timed_out(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
    )
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

// ── Second-launch request ────────────────────────────────────────

/// Ask the running instance to bring its window up — the one request a second launch of the
/// product makes (see `main`).
///
/// A single short attempt with no retry: this runs at startup, before any interface exists in
/// this process, and `false` means the instance has not bound the endpoint yet (a boot or an
/// update hand-off under way), stopped answering, is shutting down, or does not serve the
/// request at all. The caller then answers the launch the way a second launch always was.
#[cfg(any(target_os = "macos", target_os = "windows"))]
#[must_use]
pub fn request_window_show(storage_root: &Path) -> bool {
    /// Bound on the whole exchange: the instance either answers at once or is not there to,
    /// and a launch must not wait on one that never will — it has no window to show either way.
    const WAIT: Duration = Duration::from_secs(2);

    // The exchange runs on a thread of its own so the wait has a bound everywhere: a Windows
    // named pipe carries no I/O timeout at all (see `ipc_query_sync_once`), and an instance
    // that took the frame but stopped answering must not hang the launch.
    let (sent, answer) = std::sync::mpsc::channel();
    let root = storage_root.to_path_buf();
    let spawned = std::thread::Builder::new()
        .name("window-show".to_string())
        .spawn(move || {
            let _ = sent.send(ask_window_show(&root));
        });
    if spawned.is_err() {
        return false;
    }
    answer.recv_timeout(WAIT).unwrap_or(false)
}

/// One exchange with the running instance: the request frame out, the answer in.
#[cfg(any(target_os = "macos", target_os = "windows"))]
fn ask_window_show(storage_root: &Path) -> bool {
    let Ok(mut stream) = connect_sync(storage_root) else {
        return false;
    };
    let request = IpcRequest {
        store: SHOW_WINDOW_STORE.to_string(),
        sql: String::new(),
        params: Vec::new(),
        call: None,
    };
    let Ok(payload) = exchange_sync(&mut stream, &request) else {
        return false;
    };
    serde_json::from_slice::<QueryResponse>(&payload).is_ok_and(|response| response.error.is_none())
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
    /// The product does not know the protocol value this call carried: it was
    /// built before the channel existed, or before the generation that value
    /// names. Nothing ran.
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
/// Anything else is examined for the unknown-store shape: a product that does
/// not know the protocol value this call carried parses the operation frame as
/// an ordinary diagnostic query and refuses that store, so a `QueryResponse`
/// whose error is `unknown store '<the value the call carried>'` is [`TooOld`] —
/// a definite "this product cannot serve what this command line sends" (one
/// built before the channel, or before the generation that value names), not an
/// unreadable/severed answer.
///
/// [`NotAccepted`]: CallOutcome::NotAccepted
/// [`TooOld`]: CallAttempt::TooOld
fn classify_reply(payload: &[u8], protocol: &str) -> CallAttempt {
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
            .is_some_and(|e| e.contains(&format!("unknown store '{protocol}'")))
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
///
/// `protocol` is the value the call's family declares (see [`crate::cli`]) and is
/// what the frame's magic `store` carries.
async fn ipc_call_once(storage_root: &Path, call: &CallRequest, protocol: &str) -> CallAttempt {
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
        store: protocol.to_string(),
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
        Ok(payload) => classify_reply(&payload, protocol),
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
/// when it decodes as the unknown-store shape (a `QueryResponse` whose `error`
/// contains `unknown store '<the value this build sent>'`).
///
/// This deliberately does NOT go through the read-only path's
/// [`ipc_query_with_wait`]: that one bounds each attempt's wait by the command's
/// own read budget, which an operation must not inherit — an operation the
/// product is still performing is never given up on.
///
/// [`NotAccepted`]: CallOutcome::NotAccepted
/// [`Unavailable`]: CallOutcome::Unavailable
pub(crate) async fn ipc_call_with_wait(
    storage_root: &Path,
    call: &CallRequest,
    protocol: &str,
) -> CallAttempt {
    let deadline = std::time::Instant::now() + ipc_bound_timeout();
    let mut attempt = 0usize;
    loop {
        let reason = match ipc_call_once(storage_root, call, protocol).await {
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
            // The `workspace` family's own protocol value: the generation that
            // served it, not the current one the `send` family carries.
            store: CALL_STORE_BASE.to_string(),
            sql: String::new(),
            params: Vec::new(),
            call: Some(CallRequest {
                tool: "workspace".to_string(),
                args: serde_json::json!({ "action": "list" }),
                secrets: Vec::new(),
                output_dir: None,
            }),
        })
        .unwrap();
        assert_eq!(operation["store"], CALL_STORE_BASE);
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
            output_dir: None,
        };
        match crate::db::ipc::ipc_call_with_wait(dir.path(), &call, CALL_STORE_BASE).await {
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
    /// product that does not know this build's protocol value answers the
    /// operation frame as an ordinary diagnostic query whose store it refuses,
    /// and that exact shape must read as `TooOld` rather than as an
    /// unreadable/severed answer.
    #[test]
    fn replies_classify_as_answered_retryable_or_too_old() {
        let retryable = serde_json::to_vec(&CallReply::not_accepted("still starting")).unwrap();
        assert!(matches!(
            classify_reply(&retryable, CALL_STORE),
            CallAttempt::NotServed(reason) if reason == "still starting"
        ));

        let final_reply = serde_json::to_vec(&CallReply::unavailable("no channel")).unwrap();
        assert!(matches!(
            classify_reply(&final_reply, CALL_STORE),
            CallAttempt::Answered(reply) if reply.outcome == CallOutcome::Unavailable
        ));

        // The exact JSON a product that does not know the value the call carried
        // returns.
        let old = serde_json::to_vec(&serde_json::json!({
            "columns": [],
            "rows": [],
            "error": format!("unknown store '{CALL_STORE}' (expected \"core\" or \"logs\")"),
            "truncated": false,
        }))
        .unwrap();
        assert!(
            matches!(classify_reply(&old, CALL_STORE), CallAttempt::TooOld),
            "an unknown-store refusal is TooOld"
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
        match classify_reply(&other_error, CALL_STORE) {
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
    /// uses when an instance holds the location). `serial(read_limit)`: this
    /// relies on the bounded retry inside the read limit (the listener task may
    /// bind after the first attempt), so it must not overlap the tests that
    /// shrink that knob.
    #[serial_test::serial(read_limit)]
    #[tokio::test]
    async fn ipc_query_serves_readonly_queries_end_to_end() {
        let (store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        // The instance's own connection, kept out of the listener task: the
        // read-only query path must leave it able to write.
        let instance_conn = store.conn.clone();
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
        let budget = crate::db::read_limit::Budget::start();
        let read = crate::db::read_limit::Read::over_channel("logs", &budget);
        let resp = crate::db::ipc::ipc_query_with_wait(&root, &req, read)
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
        let write_resp = crate::db::ipc::ipc_query_with_wait(&root, &write_req, read)
            .await
            .expect("IPC write attempt must reach the listener");
        assert!(
            write_resp.error.is_some(),
            "a write statement must be rejected by query_only=ON"
        );

        // The forms a reader refuses are refused on this store too, where the
        // writing connection's path serves the query — the same verdict whichever
        // connection serves it. The CLI's own validator never lets such a
        // statement out, so this probe speaks the protocol directly.
        let checkpoint_req = IpcRequest {
            store: "logs".to_string(),
            sql: "PRAGMA wal_checkpoint(TRUNCATE)".to_string(),
            params: Vec::new(),
            call: None,
        };
        let checkpoint_resp = crate::db::ipc::ipc_query_with_wait(&root, &checkpoint_req, read)
            .await
            .expect("IPC checkpoint attempt must reach the listener");
        assert!(
            checkpoint_resp
                .error
                .as_deref()
                .is_some_and(|e| e.contains("refused on a read-only connection")),
            "the compaction must be refused on the writing-connection path too: {:?}",
            checkpoint_resp.error
        );

        // The pragma is set and reset around each query on this store's writing
        // connection (the log store serves no reads on readers of its own), so
        // the instance's own writes are never left disabled. Only the instance
        // side can see that: the check is an ordinary instance write.
        let instance_write = instance_conn
            .execute_batch(
                "INSERT INTO logs (timestamp, level, target, message) \
                 VALUES ('2026-01-01T00:00:01Z', 'INFO', 'test', 'after the read-only query');",
            )
            .await;
        assert!(
            instance_write.is_ok(),
            "the read-only query must leave the connection able to write: {instance_write:?}"
        );

        listener.abort();
        let _ = listener.await;
    }

    /// A hand-rolled listener for the client tests below: it accepts EVERY
    /// connection, reads one request frame, and then answers after
    /// `answer_after` — or never, when that is `None`, which is the instance
    /// that took the query and went quiet. Accepted streams are held (nothing
    /// closes under the client's read), connections are counted so a test can
    /// prove a taken query was not sent again, and the stop signal — or the
    /// listener's own drop — ends it.
    ///
    /// No store is involved: the socket path is the whole endpoint.
    fn slow_listener(
        root: &Path,
        answer_after: Option<Duration>,
    ) -> (
        std::thread::JoinHandle<()>,
        std::sync::mpsc::Sender<()>,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) {
        use interprocess::local_socket::traits::Listener as _;

        let socket = socket_path(root);
        let accepted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let accepted_in_server = accepted.clone();
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            let name = socket
                .as_path()
                .to_fs_name::<GenericFilePath>()
                .expect("socket name");
            let mut opts = ListenerOptions::new();
            opts = opts.name(name).try_overwrite(true);
            let listener = opts.create_sync().expect("bind the client-test listener");
            // Nonblocking accept: EVERY connection must be counted — a resend
            // would be a second one — and the loop must still notice the stop
            // signal. The accepted streams stay blocking, so reading the frame
            // below waits for it as the test means it to.
            listener
                .set_nonblocking(interprocess::local_socket::ListenerNonblockingMode::Accept)
                .expect("nonblocking accept");
            let mut held = Vec::new();
            loop {
                match listener.accept() {
                    Ok(mut stream) => {
                        accepted_in_server.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        let _ = read_frame_sync(&mut stream);
                        if let Some(after) = answer_after {
                            std::thread::sleep(after);
                            let body = serde_json::to_vec(&QueryResponse {
                                columns: vec!["COUNT(*)".to_string()],
                                rows: vec![vec![WireValue::Integer(1)]],
                                error: None,
                                truncated: false,
                            })
                            .expect("serialize the answer");
                            let _ = write_frame_sync(&mut stream, &body);
                        }
                        held.push(stream);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        if stop_rx.recv_timeout(Duration::from_millis(10)).is_ok() {
                            break;
                        }
                    }
                    Err(e) => panic!("accept failed: {e}"),
                }
            }
        });
        (server, stop_tx, accepted)
    }

    /// A spent budget is reported, never raced against: with the limit at zero the
    /// client must fail at once and must not even open an exchange. A zero wait is
    /// not an immediate failure on its own — the timer's own first poll can let a
    /// fast answer through — so the only deterministic reading of the smallest
    /// limit is to refuse before anything is sent. `serial(read_limit)`: this
    /// shrinks the read limit, so it must not overlap another test that reads the
    /// knob.
    #[serial_test::serial(read_limit)]
    #[tokio::test]
    async fn a_spent_budget_never_reaches_the_instance() {
        let _env = crate::util::test::set_env_var(crate::db::read_limit::ENV, Some("0"));
        let dir = tempfile::TempDir::new().expect("temp dir for the socket");
        let (server, stop, accepted) = slow_listener(dir.path(), None);

        let budget = crate::db::read_limit::Budget::start();
        let read = crate::db::read_limit::Read::over_channel("logs", &budget);
        let req = IpcRequest {
            store: "logs".to_string(),
            sql: "SELECT COUNT(*) FROM logs".to_string(),
            params: Vec::new(),
            call: None,
        };
        let err = crate::db::ipc::ipc_query_with_wait(dir.path(), &req, read)
            .await
            .expect_err("a spent budget must fail the read");
        let text = err.to_string();
        assert!(
            text.contains("'logs'"),
            "the failure names the target: {text}"
        );
        assert!(
            text.contains("0s read limit"),
            "the failure names the limit: {text}"
        );

        let _ = stop.send(());
        server.join().expect("the listener thread must end");
        assert_eq!(
            accepted.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a spent budget must not send the query at all"
        );
    }

    /// A service that is slow but alive answers inside the limit: the wait
    /// follows the command's own budget, so an instance that is merely occupied
    /// is waited out instead of being cut off as an unreachable channel — the
    /// wrong diagnosis the fixed channel-wide bound produced. `serial(read_limit)`:
    /// this shrinks the read limit, so it must not overlap another test that reads
    /// the knob.
    #[serial_test::serial(read_limit)]
    #[tokio::test]
    async fn a_slow_but_alive_instance_answers_inside_the_read_limit() {
        let _env = crate::util::test::set_env_var(crate::db::read_limit::ENV, Some("5"));
        let dir = tempfile::TempDir::new().expect("temp dir for the socket");
        let (server, stop, _accepted) = slow_listener(dir.path(), Some(Duration::from_millis(700)));

        let budget = crate::db::read_limit::Budget::start();
        let read = crate::db::read_limit::Read::over_channel("logs", &budget);
        let req = IpcRequest {
            store: "logs".to_string(),
            sql: "SELECT COUNT(*) FROM logs".to_string(),
            params: Vec::new(),
            call: None,
        };
        let resp = crate::db::ipc::ipc_query_with_wait(dir.path(), &req, read)
            .await
            .expect("an answer that arrives inside the limit must be returned");
        assert!(resp.error.is_none(), "no error expected: {:?}", resp.error);
        assert_eq!(resp.rows, vec![vec![WireValue::Integer(1)]]);

        let _ = stop.send(());
        server.join().expect("the listener thread must end");
    }

    /// A read-only query the instance TOOK but never answered must fail with the
    /// read limit's own wording, never as an unreachable channel — the wrong
    /// diagnosis the channel-wide 15 s bound produced (a merely slow instance
    /// reads as "the debug channel could not be reached"). The query is also
    /// never sent again: the instance has it, so a second copy would run the
    /// same work twice.
    ///
    /// `serial(read_limit)`: this shrinks the read limit, so it must not overlap
    /// another test that reads the knob.
    #[serial_test::serial(read_limit)]
    #[tokio::test]
    async fn a_query_a_silent_instance_took_fails_with_the_read_limit_and_is_not_resent() {
        let _env = crate::util::test::set_env_var(crate::db::read_limit::ENV, Some("1"));
        let dir = tempfile::TempDir::new().expect("temp dir for the socket");
        let (server, stop, accepted) = slow_listener(dir.path(), None);

        let budget = crate::db::read_limit::Budget::start();
        let read = crate::db::read_limit::Read::over_channel("logs", &budget);
        let req = IpcRequest {
            store: "logs".to_string(),
            sql: "SELECT COUNT(*) FROM logs".to_string(),
            params: Vec::new(),
            call: None,
        };
        let err = crate::db::ipc::ipc_query_with_wait(dir.path(), &req, read)
            .await
            .expect_err("a query the instance took but never answered must fail");
        let text = err.to_string();
        assert!(
            text.contains("logs"),
            "the failure names the target: {text}"
        );
        assert!(
            text.contains("1s read limit"),
            "the failure names the limit: {text}"
        );
        assert!(
            text.contains(crate::db::read_limit::ENV),
            "the failure names the knob: {text}"
        );
        assert!(
            !text.contains("could not be reached"),
            "a taken query must not be reported as an unreachable channel: {text}"
        );

        let _ = stop.send(());
        server.join().expect("the silent listener thread must end");
        assert_eq!(
            accepted.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "a query the instance took must never be sent again"
        );
    }

    /// The same limit bounds the SYNCHRONOUS client, which is the one
    /// `bench-openrouter` resolves its config through: a query the instance took
    /// and never answered fails with the limit's own wording — never as an
    /// unreachable channel — and is never sent again. `serial(read_limit)`: this
    /// shrinks the read limit, so it must not overlap another test that reads the
    /// knob.
    ///
    /// Unix only: the sync client's end here is its socket timeout, and named
    /// pipes carry none — on Windows the only end a wedged exchange has is the
    /// read-limit watchdog, which a test binary disarms, so this test could never
    /// finish there.
    #[cfg(unix)]
    #[serial_test::serial(read_limit)]
    #[test]
    fn a_silent_instance_fails_the_sync_config_read_with_the_read_limit() {
        let _env = crate::util::test::set_env_var(crate::db::read_limit::ENV, Some("1"));
        let dir = tempfile::TempDir::new().expect("temp dir for the socket");
        let (server, stop, accepted) = slow_listener(dir.path(), None);

        let budget = crate::db::read_limit::Budget::start();
        let read = crate::db::read_limit::Read::over_channel("config", &budget);
        let req = IpcRequest {
            store: "core".to_string(),
            sql: "SELECT value FROM config_kv WHERE key = 'worker_model'".to_string(),
            params: Vec::new(),
            call: None,
        };
        let err = crate::db::ipc::ipc_query_sync(dir.path(), &req, read)
            .expect_err("a silent instance must fail the config read");
        let text = format!("{err:#}");
        assert!(
            text.contains("'config'"),
            "the failure names the target: {text}"
        );
        assert!(
            text.contains("1s read limit"),
            "the failure names the limit: {text}"
        );
        assert!(
            !text.contains("could not be reached"),
            "a taken query must not be reported as an unreachable channel: {text}"
        );

        let _ = stop.send(());
        server.join().expect("the silent listener thread must end");
        assert_eq!(
            accepted.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "a query the instance took must never be sent again"
        );
    }

    /// End-to-end: the same listener serves the OPERATION shape of a frame —
    /// `handle_connection` reads the `call` off a protocol value it serves and
    /// routes it through the instance-side executor, and the client reads the
    /// instance's own `CallReply` back. The tool here is one no family fronts —
    /// the executor's own guard against a frame this build's value never carries
    /// in practice — so the frame is served as an operation and refused as
    /// something the product cannot take on at all, with the restart the
    /// environment's class carries — enough to prove the frame is served as an
    /// operation and not as a diagnostic query.
    /// `serial(ipc_bound)`: the call channel's wait is read from that knob (the
    /// listener may bind after the first attempt), so it must not overlap the
    /// test that shrinks it.
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
            output_dir: None,
        };
        match crate::db::ipc::ipc_call_with_wait(&root, &call, CALL_STORE).await {
            CallAttempt::Answered(reply) => {
                assert_eq!(
                    reply.outcome,
                    CallOutcome::Unavailable,
                    "answer: {}",
                    reply.answer
                );
                assert!(
                    reply.answer.contains("no_such_tool"),
                    "the refusal must name the tool: {}",
                    reply.answer
                );
                assert!(
                    reply.answer.contains("restart it from this build"),
                    "a product that cannot take the operation on says to restart: {}",
                    reply.answer
                );
            }
            _ => panic!("the operation frame must be served as an operation"),
        }

        listener.abort();
        let _ = listener.await;
    }

    /// End-to-end through the whole command line: the frame a `mahbot send`
    /// invocation writes is served by the instance-side executor, which hands the
    /// message to the owner's own assistant — the one-way send with no addressee
    /// named, which is the command's default address. What comes back is the
    /// product's own acknowledgement, the handover is durable in `pending_jobs`
    /// when the caller reads it, the message reaches the assistant's
    /// conversation, and it is persisted to the conversation's history as the
    /// owner's own incoming message. The store is shared with the parallel tests,
    /// so the test reclaims the durable row AND the conversation row it created —
    /// no other test can pick the handover up and route it into a live turn.
    /// `serial(ipc_bound, …)`: the call channel's wait is read from that knob,
    /// and this registers the owner's assistant consumer in the
    /// process-global router, which the dispatch tests hold too.
    #[serial_test::serial(ipc_bound, provider, channel_registry, drain)]
    #[tokio::test]
    async fn ipc_call_serves_the_one_way_send_end_to_end() {
        crate::util::test::init_management_test_stores().await;
        // The handover reaches the owner as an ordinary incoming message, so it
        // mirrors to Telegram on the way — the registry must exist for that leg.
        crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);
        crate::cli::mark_ready();
        let (store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        let root = dir.path().to_path_buf();
        let listener_root = root.clone();
        let listener = tokio::spawn(async move {
            crate::db::ipc::run_ipc_listener(&listener_root, std::sync::Arc::new(store)).await;
        });

        // Consume the routed envelope so the real agent run never starts.
        let assistant = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut rx = crate::agent::message_router::register_agent(&assistant);

        // The durable rows outstanding before this call: the store is shared with
        // the parallel tests, so what this call created is the DIFF it makes, never
        // a bare count another test could satisfy.
        let pending_before: std::collections::HashSet<String> =
            crate::jobs::list_pending_jobs(&crate::session::store().conn)
                .await
                .expect("list pending")
                .into_iter()
                .map(|row| row.id)
                .collect();

        let call = CallRequest {
            tool: "send".to_string(),
            args: serde_json::json!({ "message": "from the command line" }),
            secrets: Vec::new(),
            output_dir: None,
        };
        match crate::db::ipc::ipc_call_with_wait(&root, &call, CALL_STORE).await {
            CallAttempt::Answered(reply) => {
                assert_eq!(reply.outcome, CallOutcome::Done, "answer: {}", reply.answer);
                assert!(
                    reply.answer.contains("assistant"),
                    "the answer names the handover: {}",
                    reply.answer
                );
            }
            _ => panic!("the send frame must be served as an operation"),
        }

        let job = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("the handover must reach the assistant")
            .expect("the consumer must still be registered");
        assert_eq!(job.content, "from the command line");
        assert_eq!(job.channel, crate::cli::CHANNEL);

        let created: Vec<crate::jobs::PendingJobRow> =
            crate::jobs::list_pending_jobs(&crate::session::store().conn)
                .await
                .expect("list pending")
                .into_iter()
                .filter(|row| !pending_before.contains(&row.id))
                .collect();
        // The rows this call made, by the message only this call sent — a row a
        // parallel test added to the shared store in the meantime is not ours and
        // must not fail the assertion.
        let mine: Vec<&crate::jobs::PendingJobRow> = created
            .iter()
            .filter(|row| row.envelope.contains("from the command line"))
            .collect();
        assert_eq!(
            mine.len(),
            1,
            "exactly the handover's own durable row: {created:?}"
        );
        let row = mine[0];
        assert_eq!(
            row.target_agent_id, assistant,
            "the row is addressed to the assistant"
        );

        let workspace = crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME);
        let entries = crate::channels::chat_history::store()
            .load_conversation(crate::users::ADMIN_USER_NAME, &workspace)
            .await
            .unwrap();
        assert!(
            entries.iter().any(|entry| {
                entry.direction == crate::ChatDirection::User
                    && entry.content == "from the command line"
            }),
            "the message is persisted as an incoming user message: {entries:?}"
        );

        // Reclaim BOTH rows this call created before the test ends: the store is
        // shared with the parallel tests, and a leftover durable envelope can be
        // picked up by a parallel scan (or replayed at the next boot) and routed
        // into a live turn for the owner's assistant.
        crate::jobs::delete_pending_job(&crate::session::store().conn, &row.id)
            .await
            .expect("reclaim the handover's row");
        let message_id = entries
            .iter()
            .find(|entry| {
                entry.direction == crate::ChatDirection::User
                    && entry.content == "from the command line"
            })
            .expect("the conversation row's own entry")
            .message_id
            .clone();
        crate::channels::chat_history::store()
            .conn
            .execute(
                "DELETE FROM chat_history WHERE message_id = ?1",
                crate::db::params![message_id],
            )
            .await
            .expect("reclaim the conversation row");

        crate::agent::message_router::unregister_agent(&assistant);
        listener.abort();
        let _ = listener.await;
    }
}
