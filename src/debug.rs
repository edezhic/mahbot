//! Debug query tool — read-only SQL diagnostics for mahbot's databases.
//!
//! Invoked as `mahbot debug --db <name> "SQL query"` from the command line.
//! Skips tracing initialization, lock acquisition, and GUI startup.
//!
//! ## Genuinely read-only
//!
//! The CLI opens every store through `turso::core::Database::open_file_with_flags`
//! with `OpenFlags::ReadOnly | OpenFlags::NoLock` (the turso SDK `Builder` has
//! no read-only option and always passes `OpenFlags::Create`, which would
//! create a missing `-tshm` file — a filesystem mutation). The read-only open:
//!
//! - Opens the database file `O_RDONLY` — the DB and WAL files have no write
//!   path, so no statement (including `PRAGMA wal_checkpoint`) can mutate
//!   them.
//! - Reuses an existing `.tshm` coordination file when present (so live reads
//!   see the daemon's WAL frames); it never creates a missing `.tshm`
//!   (turso_core degrades to the legacy read-only WAL path instead).
//! - Applies `multiprocess_wal` + `index_method` experimental features, the
//!   same set the daemon uses.
//!
//! One nuance: turso_core opens a present `.tshm` with its own read-write
//! handle and memory-maps it `PROT_READ | PROT_WRITE`, and every read
//! transaction registers/unregisters a reader slot (writing reader-owner
//! fields and the reader bitmap into the mmap'd header). That header churn
//! stays in memory — repeated md5 comparisons confirm no inode, mtime, or size
//! change on the file — but it is a write path to the mmap. The strict "no
//! writes at the file level" claim therefore applies to the DB/WAL files, and
//! the strict no-change proof is guaranteed on snapshot copies (where the
//! `.tshm` is omitted and the legacy read-only WAL path is used).
//!
//! Two defense-in-depth layers remain on top of the read-only open: an
//! upfront file-existence check and a SQL validator (mutation-keyword
//! blocklist plus a PRAGMA allowlist — see [`validate_read_only`]).
//!
//! ## Live-instance artifact detection
//!
//! When a foreign standard-SQLite actor removes/recreates `-wal` files
//! under the running daemon, the daemon's WAL fd becomes orphaned: the `.tshm`
//! header advertises live frames while the on-disk `-wal` is empty, so reads
//! hit torn-frame errors. Before opening, the CLI parses the `.tshm` header
//! and, when it detects the artifact condition (`max_frame > 0` and on-disk
//! `-wal` is 0 bytes), fails immediately with an explicit, actionable error
//! ("live instance artifact: query a snapshot copy") instead of a cryptic
//! torn-frame error. Torn-frame errors that slip past the pre-check (race
//! windows, partial WALs) are recognized by signature and retried with bounded
//! backoff. If the artifact condition is confirmed after retries, the CLI
//! reports the explicit artifact error; otherwise it reports a
//! corruption/inconsistency error (raw torn-frame text is never printed — the
//! engine messages are themselves the cryptic output this CLI exists to
//! replace).

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use tokio::time::sleep;

use crate::turso as turso_mod;
use crate::wal_guard::{is_orphaned_wal, parse_tshm_header};

/// Row limit per query to prevent unbounded output.
const ROW_LIMIT: usize = 10_000;

/// Mutation keywords blocked by the read-only validator.
/// Case-insensitive whole-word match (not substring).
const BLOCKLIST: &[&str] = &[
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "BEGIN", "COMMIT",
    "ROLLBACK", "VACUUM", "REINDEX", "GRANT", "REVOKE", "ATTACH", "DETACH", "ANALYZE",
];

/// SQL punctuation characters used for word-boundary tokenization.
/// Splitting on these ensures whole-word matching — a column named
/// `created_at` is not blocked by the `CREATE` keyword.
const SQL_PUNCTUATION: &[char] = &[
    '(', ')', ';', ',', '.', '*', '+', '-', '/', '=', '<', '>', '!', '|', '&', '~', '\'', '"', '[',
    ']', '{', '}', ':',
];

/// PRAGMA names allowed by the read-only validator.
///
/// The connection is opened genuinely read-only (`O_RDONLY`), so no PRAGMA can
/// mutate the database file. This allowlist is defense-in-depth: mutating
/// PRAGMAs (`wal_checkpoint`, `journal_mode`, `synchronous`, `auto_vacuum`, …)
/// are rejected with a clear message even though the open would already refuse
/// them at the file layer.
const SAFE_PRAGMAS: &[&str] = &[
    "quick_check",
    "integrity_check",
    "table_info",
    "table_xinfo",
    "index_info",
    "index_list",
    "index_xinfo",
    "foreign_key_check",
    "database_list",
    "compile_options",
    "page_count",
    "freelist_count",
    "page_size",
    "encoding",
    "user_version",
    "schema_version",
    "collation_list",
    "function_list",
    "module_list",
    "pragma_list",
    "table_list",
    "stats",
];

/// Torn-frame / WAL-inconsistency error signatures (lowercased substring
/// match). Error variants are stringified at the SDK/engine boundary, so
/// signature matching is the robust detection mechanism.
const TORN_FRAME_SIGNATURES: &[&str] = &[
    "short read on wal frame",
    "short read on page",
    "invalid page type",
    "wal frame page mismatch",
    "checksum mismatch",
    "torn",
];

/// Maximum attempts (including the first) for a torn-frame open/query failure.
const MAX_OPEN_ATTEMPTS: usize = 5;

/// Backoff (seconds) between retry attempts, indexed by `attempt`.
const OPEN_RETRY_BACKOFF_SECS: [u64; MAX_OPEN_ATTEMPTS - 1] = [1, 2, 4, 8];

/// Run the debug subcommand. Parses `env::args()` for the debug invocation.
///
/// Returns `Ok(())` on success (exit code 0), `Err` on failure (exit code 1).
/// Prints usage to stderr and returns `Err` for invalid argument combinations.
pub async fn run_debug() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    run_debug_with_args(args, None).await
}

/// Testable core of [`run_debug`].
///
/// `args` has the same shape as `std::env::args()` (`args[0]` is the program
/// name, `args[1]` is `"debug"`). `home_override`, when provided, replaces the
/// `~/.mahbot` storage-root resolution (used by tests with temporary roots and
/// by the snapshot-copy procedure).
async fn run_debug_with_args(args: Vec<String>, home_override: Option<PathBuf>) -> Result<()> {
    // args[0] = "mahbot", args[1] = "debug"
    // Handle --help: mahbot debug --help
    if args.get(2).is_some_and(|a| a == "--help") {
        print_usage();
        return Ok(());
    }

    // Need at least: mahbot debug --db <name> <sql>
    if args.len() < 5 {
        print_usage();
        bail!("expected: mahbot debug --db <name> \"SQL query\"");
    }

    if args[2] != "--db" {
        eprintln!("Error: expected --db flag, got '{}'", args[2]);
        print_usage();
        bail!("expected --db flag");
    }

    let db_name = &args[3];
    let sql = &args[4];

    // Resolve ~/.mahbot/ (or the test/snapshot override).
    let mahbot_home = match home_override {
        Some(home) => home,
        None => crate::config::default_config_dir()?,
    };

    // Validate SQL is read-only before touching any database
    validate_read_only(sql)?;

    let db_list = resolve_db_list(db_name)?;

    let mut failures = 0usize;
    for (label, db_filename) in &db_list {
        if db_name == "all" {
            println!("=== {label} ===");
        }

        let file_path = mahbot_home.join(db_filename);

        // Pre-existence check: the read-only open fails on a non-existent
        // file, but we check existence upfront for a better error message.
        if !file_path.exists() {
            if db_name == "all" {
                eprintln!(
                    "Warning: database not found, skipping: {}",
                    file_path.display()
                );
                failures += 1;
                continue;
            }
            bail!("database file not found: {}", file_path.display());
        }

        match query_one_store(&file_path, sql).await {
            Ok(()) => {}
            Err(e) => {
                if db_name == "all" {
                    eprintln!("Error: {e:#}");
                    failures += 1;
                    continue;
                }
                return Err(e);
            }
        }
    }

    // `--db all` must not exit 0 when any store failed: scripted diagnostics
    // rely on the exit code to detect failures.
    if db_name == "all" && failures > 0 {
        bail!(
            "{failures} of {} store(s) failed — see the per-store errors above",
            db_list.len()
        );
    }

    Ok(())
}

/// Open one store read-only and run `sql`, retrying torn-frame failures with
/// bounded backoff.
///
/// The artifact pre-check lives here (not only in the caller) so the
/// final-failure path can re-check it: a race may start the daemon publishing
/// frames mid-retry, or the path may be a snapshot copy with no `.tshm` at all
/// — in which case a persistent torn/page failure means the copied data is
/// corrupt, not that a live artifact exists.
async fn query_one_store(file_path: &Path, sql: &str) -> Result<()> {
    // Pre-open artifact detection: `.tshm` advertises live frames while the
    // on-disk `-wal` is empty → the daemon's WAL fd is orphaned. Report the
    // explicit, actionable artifact error instead of letting the open surface
    // a raw torn-frame read.
    if let Some(artifact_msg) = detect_live_artifact(file_path) {
        bail!("{artifact_msg}");
    }

    // Snapshot copies have no `-tshm`; they are static, so a torn-frame
    // failure cannot be transient there. Only live stores (which the daemon
    // keeps writing to) get the bounded retry that spans write windows.
    let tshm_path = PathBuf::from(format!("{}-tshm", file_path.display()));
    let is_live = tshm_path.exists();

    // Open + query with bounded retry on torn-frame failures. Retrying
    // spans short write windows (e.g. the daemon mid-checkpoint); the
    // backoff total (~15s) is bounded so the CLI cannot hang for long.
    let mut attempt = 0usize;
    loop {
        match open_and_query_readonly(file_path, sql) {
            Ok(()) => return Ok(()),
            Err(e) => {
                if !is_torn_frame_error(&e) {
                    return Err(e);
                }
                if is_live && attempt < MAX_OPEN_ATTEMPTS - 1 {
                    sleep(Duration::from_secs(OPEN_RETRY_BACKOFF_SECS[attempt])).await;
                    attempt += 1;
                    continue;
                }
                // Final attempt failed with a torn-frame read. Re-check the
                // artifact condition — the pre-open check may have missed it
                // (the daemon can start publishing frames mid-retry). The raw
                // engine message is deliberately withheld: it is a torn-frame
                // signature and must never leak as raw output.
                if let Some(artifact_msg) = detect_live_artifact(file_path) {
                    eprintln!(
                        "Note: after {attempt} retries, the final attempt still hit a torn-frame read."
                    );
                    bail!("{artifact_msg}");
                }
                // Not the artifact: the on-disk data itself is corrupt or
                // internally inconsistent (a snapshot whose copied main DB/WAL
                // is bad, or a live store whose main file is corrupt). Report
                // the condition — never the raw torn-frame text.
                let msg = corruption_error_message(file_path, attempt);
                bail!("{msg}");
            }
        }
    }
}

/// Detect the live-instance WAL artifact before opening a store.
///
/// Returns a human-readable, actionable error message when `.tshm` advertises
/// live frames (`max_frame > 0`) but the on-disk `-wal` is empty — the
/// signature of an orphaned daemon WAL fd. Returns `None` when the condition
/// is absent (healthy stores, or affected stores in a quiet window right after
/// a checkpoint, where `max_frame` reads 0 and a read would actually succeed).
fn detect_live_artifact(db_path: &Path) -> Option<String> {
    let tshm_path = PathBuf::from(format!("{}-tshm", db_path.display()));
    let wal_path = PathBuf::from(format!("{}-wal", db_path.display()));
    let header = parse_tshm_header(&tshm_path);
    let wal_size = std::fs::metadata(&wal_path).map_or(0, |m| m.len());
    // Shared with the wal-guard (`wal_guard::is_orphaned_wal`) so the central
    // business rule cannot drift between the guard and the CLI.
    if !is_orphaned_wal(header, wal_size) {
        return None;
    }
    Some(artifact_error_message(db_path))
}

/// Build the explicit artifact error message for a store.
fn artifact_error_message(db_path: &Path) -> String {
    format!(
        "live instance artifact: cannot read '{}' safely.\n\
         The running daemon's WAL file descriptor is orphaned: its on-disk \
         `-wal` file is empty while the `.tshm` coordination header advertises \
         live WAL frames (foreign standard-SQLite activity likely removed or \
         replaced the `-wal` files under the daemon). Query a snapshot \
         copy instead — see docs/ops/wal-snapshots.md. Never delete or recreate \
         `-wal`/`-shm`/`-tshm` files while the daemon runs.",
        db_path.display()
    )
}

/// Build the error message for a persistent torn/page failure that is **not**
/// the live-instance artifact (the artifact re-check came back negative).
///
/// Distinguishes live stores (a `.tshm` is present — a snapshot copy may still
/// read cleanly) from snapshot copies (no `.tshm` — the copied data itself is
/// corrupt or was copied mid-checkpoint). `retries` is the number of bounded
/// retries actually performed (0 for snapshot copies, which are static and are
/// not retried). Never includes the raw engine error: its text is itself a
/// torn-frame signature.
fn corruption_error_message(db_path: &Path, retries: usize) -> String {
    let tshm_path = PathBuf::from(format!("{}-tshm", db_path.display()));
    if tshm_path.exists() {
        format!(
            "database corruption/inconsistency: cannot read '{}' — the read \
             produced a WAL/page error even after {retries} retries. This is a \
             live store: query a snapshot copy instead (see \
             docs/ops/wal-snapshots.md), or retry during a quiet window.",
            db_path.display(),
        )
    } else {
        format!(
            "database corruption/inconsistency: cannot read '{}' — the copied \
             main DB file (or its WAL) is corrupt or internally inconsistent. \
             No `-tshm` was present, so this is a snapshot copy, not a live \
             store. Re-copy the store files (a copy taken mid-checkpoint can be \
             inconsistent); if a fresh copy still fails, the store's on-disk \
             data itself is corrupt — see docs/ops/wal-snapshots.md.",
            db_path.display()
        )
    }
}

/// Open a store read-only and execute `sql`, printing pipe-delimited results.
///
/// Uses the low-level `turso::core` API because the SDK `Builder` cannot open
/// read-only: `OpenFlags::ReadOnly | OpenFlags::NoLock` guarantees the open
/// cannot create or mutate files, and reuses an existing `.tshm` when present
/// so live stores are read through the daemon's WAL coordination. The core IO
/// is synchronous (`FsIO`), so this function is intentionally blocking; the
/// debug CLI runs on a single-task current-thread runtime.
///
/// Query output is buffered and printed only on success: a mid-query failure
/// (e.g. a torn-frame read) must not leave a partial column header on stdout,
/// which would be re-printed by the caller's retry loop.
fn open_and_query_readonly(file_path: &Path, sql: &str) -> Result<()> {
    let path_str = file_path
        .to_str()
        .with_context(|| format!("database path must be UTF-8: {}", file_path.display()))?;

    let io: std::sync::Arc<dyn turso::core::IO> =
        std::sync::Arc::new(turso::core::PlatformIO::new()?);
    let db = turso::core::Database::open_file_with_flags(
        io.clone(),
        path_str,
        turso::core::OpenFlags::ReadOnly | turso::core::OpenFlags::NoLock,
        turso_mod::experimental_database_opts(),
        None,
    )
    .map_err(|e| {
        anyhow!(
            "failed to open database '{}' read-only: {e}",
            file_path.display()
        )
    })?;
    let conn = db.connect().map_err(|e| {
        anyhow!(
            "failed to connect to database '{}': {e}",
            file_path.display()
        )
    })?;

    let output = execute_query_readonly(&io, &conn, sql, file_path)?;
    print!("{output}");
    Ok(())
}

/// Execute a read-only query on a `turso::core` connection and return the
/// results as pipe-delimited text (same format as the previous SDK-based
/// executor).
///
/// Output is accumulated in memory and returned as a single string; the caller
/// prints it only on success, so a mid-query failure never leaks a partial
/// result set (e.g. a bare column header) to stdout.
fn execute_query_readonly(
    io: &std::sync::Arc<dyn turso::core::IO>,
    conn: &std::sync::Arc<turso::core::Connection>,
    sql: &str,
    db_path: &Path,
) -> Result<String> {
    let mut stmt = conn
        .query(sql)
        .map_err(|e| anyhow!("SQL query failed on '{}': {e}", db_path.display()))?
        .ok_or_else(|| anyhow!("query produced no statement on '{}'", db_path.display()))?;

    let col_count = stmt.num_columns();
    if col_count == 0 {
        return Ok(String::new());
    }

    let mut out = String::new();

    // Column header row.
    let column_names: Vec<String> = (0..col_count)
        .map(|i| stmt.get_column_name(i).into_owned())
        .collect();
    out.push_str(&column_names.join("|"));
    out.push('\n');

    let mut row_count = 0usize;
    let mut has_more = false;

    loop {
        match stmt
            .step()
            .map_err(|e| anyhow!("SQL query failed on '{}': {e}", db_path.display()))?
        {
            turso::core::StepResult::Done => break,
            turso::core::StepResult::IO | turso::core::StepResult::Yield => {
                io.step()
                    .map_err(|e| anyhow!("SQL query failed on '{}': {e}", db_path.display()))?;
            }
            turso::core::StepResult::Row => {
                if row_count >= ROW_LIMIT {
                    has_more = true;
                    break;
                }
                let row = stmt
                    .row()
                    .ok_or_else(|| anyhow!("row missing after StepResult::Row"))?;
                out.push_str(&format_core_row(row, col_count));
                out.push('\n');
                row_count += 1;
            }
            turso::core::StepResult::Interrupt => {
                bail!("query interrupted on '{}'", db_path.display())
            }
            turso::core::StepResult::Busy => {
                bail!("database busy on '{}'; try again later", db_path.display())
            }
        }
    }

    if has_more {
        out.push_str(&format_truncation_row(col_count));
        out.push('\n');
    }

    Ok(out)
}

/// True when the error chain looks like a torn-frame / WAL-inconsistency read.
///
/// Matches on the stringified error because engine variants are stringified at
/// the error boundary; the signatures mirror turso_core 0.7.0's
/// `CompletionError::ShortReadWalFrame` / `ShortRead` / `ChecksumMismatch` /
/// `WalFramePageMismatch` display texts.
fn is_torn_frame_error(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}").to_lowercase();
    TORN_FRAME_SIGNATURES.iter().any(|sig| msg.contains(sig))
}

/// Map a `--db` argument to a list of `(label, filename)` pairs.
fn resolve_db_list(name: &str) -> Result<Vec<(String, String)>> {
    let names = turso_mod::store_names();
    if name == "all" {
        Ok(names
            .iter()
            .map(|n| (n.to_string(), format!("db/{n}.db")))
            .collect())
    } else if names.contains(&name) {
        Ok(vec![(name.to_string(), format!("db/{name}.db"))])
    } else {
        let valid = names.join(", ");
        bail!("invalid database name '{name}'. Valid names: {valid}, all");
    }
}

/// Reject any SQL containing mutation keywords (whole-word, case-insensitive)
/// or a PRAGMA not on the read-only allowlist.
fn validate_read_only(sql: &str) -> Result<()> {
    let tokens = tokenize_sql(sql);
    for (idx, token) in tokens.iter().enumerate() {
        let upper = token.to_uppercase();
        if BLOCKLIST.contains(&upper.as_str()) {
            bail!("query rejected: contains blocked keyword '{token}'");
        }
        if upper == "PRAGMA" {
            let Some(name) = tokens.get(idx + 1) else {
                bail!("query rejected: incomplete PRAGMA statement");
            };
            if !SAFE_PRAGMAS.contains(&name.to_lowercase().as_str()) {
                bail!(
                    "query rejected: PRAGMA '{name}' is not on the read-only allowlist \
                     (mutating PRAGMAs are blocked; the connection is read-only)"
                );
            }
        }
    }
    Ok(())
}

/// Split SQL into whole-word tokens on whitespace and SQL punctuation.
///
/// Punctuation characters are discarded (they can never match a blocklist
/// keyword). Adjacent punctuation creates empty tokens which are filtered.
fn tokenize_sql(sql: &str) -> Vec<String> {
    sql.split(|c: char| c.is_whitespace() || SQL_PUNCTUATION.contains(&c))
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Format a single `turso::core` result row as pipe-delimited values.
fn format_core_row(row: &turso::core::Row, column_count: usize) -> String {
    let parts: Vec<String> = (0..column_count)
        .map(|idx| format_core_value(row.get_value(idx)))
        .collect();
    parts.join("|")
}

/// Convert a `turso::core` column value to its display representation.
///
/// - NULL → empty (no text between pipe delimiters → `||`)
/// - Integer → decimal string
/// - Real → default `f64::Display`
/// - Text → verbatim (no escaping for pipes or newlines)
/// - Blob → lowercase hex string
fn format_core_value(val: &turso::core::Value) -> String {
    match val {
        turso::core::Value::Null => String::new(),
        turso::core::Value::Numeric(turso::core::Numeric::Integer(i)) => i.to_string(),
        turso::core::Value::Numeric(turso::core::Numeric::Float(fl)) => {
            let f: f64 = (*fl).into();
            f.to_string()
        }
        turso::core::Value::Text(t) => t.as_str().to_string(),
        turso::core::Value::Blob(b) => hex_encode(b),
    }
}

/// Encode a byte slice as a lowercase hex string.
fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Format a truncation sentinel row that matches the column count.
///
/// For 1 column:  `truncated`
/// For 2 columns: `truncated|truncated`
/// For N≥3:       `...|truncated|truncated|...|...`
///                 (ellipsis at first and last, `truncated` in between)
fn format_truncation_row(column_count: usize) -> String {
    let parts: Vec<&str> = match column_count {
        1 => vec!["truncated"],
        2 => vec!["truncated", "truncated"],
        _ => {
            let mut parts = vec!["..."];
            parts.extend(std::iter::repeat_n("truncated", column_count - 2));
            parts.push("...");
            parts
        }
    };
    parts.join("|")
}

fn print_usage() {
    eprintln!("Usage: mahbot debug --db <name> \"SQL query\"");
    let names = turso_mod::store_names().join(" | ");
    eprintln!("  --db <name>  {names} | all");
    eprintln!("  SQL query    read-only SQL, quoted as a single argument");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  mahbot debug --db board \"SELECT phase, COUNT(*) FROM tickets GROUP BY phase\"");
    eprintln!("  mahbot debug --db all \"SELECT name FROM sqlite_master WHERE type='table'\"");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocklist_rejects_mutation_keywords() {
        for sql in [
            "DROP TABLE tickets",
            "DELETE FROM logs",
            "INSERT INTO logs VALUES (1)",
            "UPDATE users SET name='x'",
            "VACUUM",
            "BEGIN",
            "PRAGMA wal_checkpoint(TRUNCATE)",
        ] {
            assert!(validate_read_only(sql).is_err(), "should reject: {sql}");
        }
    }

    #[test]
    fn safe_queries_pass_validation() {
        for sql in [
            "SELECT * FROM tickets",
            "SELECT created_at FROM logs LIMIT 10",
            "PRAGMA quick_check",
            "PRAGMA integrity_check(1)",
            "PRAGMA table_info(tickets)",
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'",
        ] {
            assert!(validate_read_only(sql).is_ok(), "should accept: {sql}");
        }
    }

    #[test]
    fn mutating_pragmas_are_rejected() {
        for sql in [
            "PRAGMA wal_checkpoint",
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=OFF",
            "PRAGMA auto_vacuum=INCREMENTAL",
        ] {
            assert!(validate_read_only(sql).is_err(), "should reject: {sql}");
        }
    }

    #[test]
    fn tokenizer_splits_sql_punctuation() {
        let tokens = tokenize_sql("SELECT a, b FROM t WHERE x='y'");
        assert!(tokens.contains(&"SELECT".to_string()));
        assert!(tokens.contains(&"b".to_string()));
        assert!(tokens.contains(&"y".to_string()));
        assert!(!tokens.contains(&"".to_string()));
    }

    #[test]
    fn torn_frame_signatures_match_engine_errors() {
        let cases = [
            "I/O error: short read on WAL frame at offset 1466752: expected 4096 bytes, got 0",
            "I/O error: short read on page 12: expected 4096 bytes, got 512",
            "Invalid page type: 0",
            "WAL frame page mismatch at frame 3: expected page 5, got 9",
            "Checksum mismatch on page 7: expected 123, got 456",
        ];
        for msg in cases {
            let err = anyhow!("{msg}");
            assert!(is_torn_frame_error(&err), "should classify: {msg}");
        }
        let unrelated = anyhow!("SQL error: no such table: foo");
        assert!(!is_torn_frame_error(&unrelated));
    }

    #[test]
    fn artifact_error_message_is_actionable() {
        let msg = artifact_error_message(Path::new("/tmp/x/board.db"));
        assert!(msg.contains("live instance artifact"));
        assert!(msg.contains("snapshot"));
        assert!(msg.contains("wal-snapshots.md"));
    }

    /// End-to-end: `run_debug_with_args` opens a real (temporary) store
    /// read-only through the `turso::core` path and runs a query against it.
    #[tokio::test]
    async fn run_debug_queries_a_real_store_read_only() {
        let (_store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "--db".to_string(),
            "logs".to_string(),
            "SELECT COUNT(*) FROM logs".to_string(),
        ];
        let result = run_debug_with_args(args, Some(dir.path().to_path_buf())).await;
        assert!(
            result.is_ok(),
            "read-only query on a real store must succeed: {result:?}"
        );

        // The read-only open must not have created or modified any files in
        // the store directory (in particular no new -tshm beyond the daemon's).
        let db_dir = dir.path().join("db");
        let names: Vec<String> = std::fs::read_dir(&db_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            names.iter().any(|n| n == "logs.db"),
            "store db must exist: {names:?}"
        );
        // Exactly one tshm file (created by the store open itself) — the
        // read-only CLI open must not create another.
        let tshm_count = names.iter().filter(|n| n.ends_with("-tshm")).count();
        assert_eq!(tshm_count, 1, "no new -tshm file may be created: {names:?}");
    }

    /// End-to-end: a crafted artifact state (tshm advertises live frames while
    /// the on-disk WAL is empty) must produce the explicit artifact error, not
    /// a raw torn-frame read.
    #[tokio::test]
    async fn run_debug_reports_artifact_instead_of_torn_frame() {
        let (_store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        let db_dir = dir.path().join("db");
        let tshm_path = db_dir.join("logs.db-tshm");
        let wal_path = db_dir.join("logs.db-wal");

        // Truncate the on-disk WAL to 0 bytes and patch the tshm header's
        // max_frame field (byte offset 56, u64 LE) to advertise live frames —
        // the live-instance artifact state.
        std::fs::write(&wal_path, []).unwrap();
        let mut tshm = std::fs::read(&tshm_path).unwrap();
        assert!(tshm.len() >= 64, "tshm header must cover max_frame");
        tshm[56..64].copy_from_slice(&356u64.to_le_bytes());
        std::fs::write(&tshm_path, &tshm).unwrap();

        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "--db".to_string(),
            "logs".to_string(),
            "SELECT COUNT(*) FROM logs".to_string(),
        ];
        let err = run_debug_with_args(args, Some(dir.path().to_path_buf()))
            .await
            .expect_err("artifact state must fail with the explicit artifact error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("live instance artifact"),
            "must be the explicit artifact error, got: {msg}"
        );
        assert!(
            !msg.to_lowercase().contains("short read"),
            "must not leak raw torn-frame output: {msg}"
        );
    }

    /// The corruption message must distinguish a live store (tshm present) from
    /// a snapshot copy (no tshm) and must never claim a live artifact on a
    /// snapshot — the previous behavior told a user who was *already* querying
    /// a snapshot to "query a snapshot copy instead".
    #[test]
    fn corruption_message_distinguishes_live_and_snapshot() {
        let dir = std::env::temp_dir().join(format!("debug_corrupt_msg_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // No tshm → snapshot copy → message must be corruption, not artifact.
        let snap = dir.join("logs.db");
        std::fs::write(&snap, b"garbage").unwrap();
        let msg = corruption_error_message(&snap, 0);
        assert!(msg.contains("corruption"), "got: {msg}");
        assert!(
            !msg.contains("live instance artifact"),
            "a snapshot must not be reported as a live artifact: {msg}"
        );
        assert!(
            !msg.to_lowercase().contains("invalid page type"),
            "raw torn-frame text must not leak: {msg}"
        );

        // tshm present → live store branch; the retry count is reported.
        std::fs::write(dir.join("logs.db-tshm"), b"x").unwrap();
        let msg_live = corruption_error_message(&snap, 4);
        assert!(msg_live.contains("live store"), "got: {msg_live}");
        assert!(msg_live.contains("4 retries"), "got: {msg_live}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// End-to-end: `--db all` must exit non-zero (return an error) when any
    /// store fails — scripted diagnostics rely on the exit code, so per-store
    /// failures cannot be silently swallowed.
    #[tokio::test]
    async fn run_debug_all_reports_failure_summary() {
        let (_store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        let db_dir = dir.path().join("db");

        // Craft an artifact state on the one existing store (logs).
        let tshm_path = db_dir.join("logs.db-tshm");
        let wal_path = db_dir.join("logs.db-wal");
        std::fs::write(&wal_path, []).unwrap();
        let mut tshm = std::fs::read(&tshm_path).unwrap();
        assert!(tshm.len() >= 64, "tshm header must cover max_frame");
        tshm[56..64].copy_from_slice(&7u64.to_le_bytes());
        std::fs::write(&tshm_path, &tshm).unwrap();

        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "--db".to_string(),
            "all".to_string(),
            "SELECT COUNT(*) FROM logs".to_string(),
        ];
        let err = run_debug_with_args(args, Some(dir.path().to_path_buf()))
            .await
            .expect_err("--db all must fail when any store fails");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("store(s) failed"),
            "expected a failure summary, got: {msg}"
        );
        assert!(
            !msg.to_lowercase().contains("short read"),
            "summary must not leak raw torn-frame text: {msg}"
        );

        let _ = std::fs::remove_dir_all(dir.path());
    }
}
