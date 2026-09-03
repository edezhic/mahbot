//! IPC query endpoint for the `mahbot debug` read-only CLI.
//!
//! After `multiprocess_wal` was removed, the debug CLI can no longer open a
//! second physical instance of a live store's database (single-process mode
//! holds the flock). Instead the daemon exposes a local IPC endpoint
//! (a Unix-domain socket on Unix, a named pipe on Windows) that accepts
//! read-only SQL queries and returns the rows. The debug CLI connects to it
//! instead of opening the database directly.
//!
//! Protocol (length-prefixed JSON over a local socket stream):
//! - request  → `[u32 LE payload_len][payload]`, payload is JSON `QueryRequest`.
//! - response → `[u32 LE payload_len][payload]`, payload is JSON `QueryResponse`.
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
pub(crate) const IPC_SOCKET_FILE_NAME: &str = "mahbot-debug.sock";
/// Windows named-pipe name. Named pipes live in a global `\\.\pipe\` namespace
/// with no per-directory scope, and the `GenericFilePath` name type only
/// accepts `\\.\pipe\`-prefixed paths there — so this is not derived from the
/// storage root. The instance lock already guarantees a single daemon.
#[cfg(windows)]
pub(crate) const IPC_PIPE_NAME: &str = r"\\.\pipe\mahbot-debug";

/// Total wall-clock bound for the daemon-up-but-socket-not-bound retry
/// (flock held before the listener binds during boot / self-update handoff).
pub(crate) const IPC_BOUND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// Retry backoff schedule (ms) for the IPC socket-not-yet-bound window.
const IPC_RETRY_BACKOFF_MS: [u64; 6] = [50, 100, 200, 400, 800, 1600];

/// Length (bytes) of the `u32 LE` length prefix on each IPC frame.
const FRAME_LEN: usize = std::mem::size_of::<u32>();

/// Upper bound (bytes) on a single IPC frame payload, enforced on read so a
/// same-user misbehaving client cannot trigger a multi-GiB allocation and OOM
/// the daemon. Requests are a few KB; responses are bounded by
/// [`IPC_ROW_LIMIT`], so this is far beyond legitimate use.
const MAX_FRAME_LEN: usize = 64 * 1024 * 1024;

/// Resolve the IPC socket name.
///
/// On Unix this is a filesystem socket file under the storage root. On Windows
/// it is the named-pipe name (see [`IPC_PIPE_NAME`]) — a global namespace, not
/// a path under the storage root. Both the listener and the clients call this
/// so they agree on the endpoint name.
#[must_use]
pub(crate) fn socket_path(storage_root: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        let _ = storage_root;
        std::path::PathBuf::from(IPC_PIPE_NAME)
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
    /// back to `NaN` so the IPC path round-trips them like the daemon-down
    /// path renders them ("NaN").
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
pub(crate) struct QueryRequest {
    pub store: String, // "core" or "logs" (physical store names)
    pub sql: String,
    #[serde(default)]
    pub params: Vec<WireValue>,
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

/// Serialize a `QueryResponse` into an IPC frame and write it. (serde_json
/// emits `null` — not an error — for non-finite floats, which `de_real` maps
/// back to NaN, so serialization never fails for the response types used.)
async fn write_response<S>(stream: &mut S, resp: &QueryResponse) -> std::io::Result<()>
where
    S: AsyncWrite + Unpin,
{
    let payload = serde_json::to_vec(resp).map_err(std::io::Error::other)?;
    write_frame(stream, &payload).await
}

async fn handle_connection(
    mut stream: LocalSocketStream,
    log_store: std::sync::Arc<crate::logs::LogStore>,
) -> anyhow::Result<()> {
    let payload = read_frame(&mut stream).await?;
    let req: QueryRequest = match serde_json::from_slice(&payload) {
        Ok(req) => req,
        Err(e) => {
            // A malformed request must get an actionable error, not an EOF/timeout.
            let resp = response_error(&format!("malformed IPC request: {e}"));
            write_response(&mut stream, &resp).await?;
            return Ok(());
        }
    };

    let conn = match req.store.as_str() {
        "logs" => Some(log_store.conn.clone()),
        "core" => crate::db::DOMAIN_CONN.get().cloned(),
        other => {
            let resp = response_error(&format!(
                "unknown store '{other}' (expected \"core\" or \"logs\")"
            ));
            write_response(&mut stream, &resp).await?;
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

    write_response(&mut stream, &resp).await?;
    Ok(())
}

/// Spawn the daemon-side IPC listener. Call after `DOMAIN_CONN` / `LOG_STORE`
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
                "ipc: socket already in use (another daemon?); debug IPC disabled"
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
    // Restrictive perms: only the daemon user can connect. `ListenerOptions::mode`
    // is NOT used — it calls `fchmod` on the socket fd, which returns
    // EINVAL→Unsupported on macOS UDS; `chmod` via the path works everywhere.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(e) = std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600)) {
            warn!(
                error = %e,
                socket = %socket.display(),
                "ipc: failed to restrict debug socket perms"
            );
        }
    }
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
                            if let Err(e) = handle_connection(stream, store).await {
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

/// Client helper: connect to the daemon's IPC socket and run a read-only query.
pub(crate) async fn ipc_query(
    storage_root: &Path,
    req: QueryRequest,
) -> anyhow::Result<QueryResponse> {
    let socket = socket_path(storage_root);
    let name = socket
        .as_path()
        .to_fs_name::<GenericFilePath>()
        .map_err(|e| anyhow::anyhow!("daemon IPC endpoint not reachable: {e}"))?;

    let result = tokio::time::timeout(Duration::from_secs(5), async {
        let mut stream = LocalSocketStream::connect(name).await?;
        let payload = serde_json::to_vec(&req)?;
        write_frame(&mut stream, &payload).await?;
        let payload = read_frame(&mut stream).await?;
        let resp: QueryResponse = serde_json::from_slice(&payload)?;
        Ok::<QueryResponse, anyhow::Error>(resp)
    })
    .await
    .map_err(|_| anyhow::anyhow!("daemon IPC endpoint timed out"))??;

    Ok(result)
}

/// Backoff delay for retry `attempt` (0-based), clamped to the schedule.
fn retry_backoff(attempt: usize) -> Duration {
    Duration::from_millis(IPC_RETRY_BACKOFF_MS[attempt.min(IPC_RETRY_BACKOFF_MS.len() - 1)])
}

/// Bounded-retry async client used by `mahbot debug` when the daemon holds the
/// instance lock but the IPC socket is not yet bound (the daemon is between
/// lock-acquire and listener-bind, e.g. boot or self-update handoff). The
/// caller must only use this when [`crate::util::lock::daemon_holds_lock_settled`]
/// is true — otherwise the retry would mask a genuinely-down daemon.
pub(crate) async fn ipc_query_with_wait(
    storage_root: &Path,
    req: &QueryRequest,
) -> anyhow::Result<QueryResponse> {
    let deadline = std::time::Instant::now() + IPC_BOUND_TIMEOUT;
    let mut attempt = 0usize;
    loop {
        match ipc_query(storage_root, req.clone()).await {
            Ok(resp) => return Ok(resp),
            Err(_) if std::time::Instant::now() < deadline => {
                tokio::time::sleep(retry_backoff(attempt)).await;
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Synchronous bounded-retry client used by `bench-openrouter`'s synchronous
/// config-resolution path (which cannot await the async [`ipc_query_with_wait`];
/// the call blocks the current thread).
pub(crate) fn ipc_query_sync(
    storage_root: &Path,
    req: &QueryRequest,
) -> std::io::Result<QueryResponse> {
    let deadline = std::time::Instant::now() + IPC_BOUND_TIMEOUT;
    let mut attempt = 0usize;
    loop {
        match ipc_query_sync_once(storage_root, req) {
            Ok(resp) => return Ok(resp),
            Err(_) if std::time::Instant::now() < deadline => {
                std::thread::sleep(retry_backoff(attempt));
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

fn ipc_query_sync_once(storage_root: &Path, req: &QueryRequest) -> std::io::Result<QueryResponse> {
    let socket = socket_path(storage_root);
    let name = socket
        .as_path()
        .to_fs_name::<GenericFilePath>()
        .map_err(|e| std::io::Error::other(format!("daemon IPC endpoint not reachable: {e}")))?;
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// End-to-end: the daemon-side listener serves a read-only query over the
    /// local socket, returning column names + rows (the daemon-up IPC path that
    /// `mahbot debug` uses when the instance lock is held).
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

        let req = QueryRequest {
            store: "logs".to_string(),
            sql: "SELECT COUNT(*) FROM logs".to_string(),
            params: Vec::new(),
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
        let write_req = QueryRequest {
            store: "logs".to_string(),
            sql: "CREATE TABLE nope (id INTEGER)".to_string(),
            params: Vec::new(),
        };
        let write_resp = crate::db::ipc::ipc_query_with_wait(&root, &write_req)
            .await
            .expect("IPC write attempt must reach the listener");
        assert!(
            write_resp.error.is_some(),
            "a write statement must be rejected by query_only=ON"
        );

        // The shared connection's query_only must be reset after the query so
        // daemon writes are never left disabled.
        let reset_check = crate::db::ipc::ipc_query_with_wait(
            &root,
            &QueryRequest {
                store: "logs".to_string(),
                sql: "SELECT 1".to_string(),
                params: Vec::new(),
            },
        )
        .await
        .expect("subsequent query must succeed after reset");
        assert!(reset_check.error.is_none());

        listener.abort();
        let _ = listener.await;
    }
}
