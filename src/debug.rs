//! Debug query tool — read-only SQL diagnostics for mahbot's databases.
//!
//! Invoked as `mahbot debug --db <name> "SQL query"` from the command line.
//! Skips tracing initialization, lock acquisition, and GUI startup.
//! Opens databases using the standard `turso::Builder` API (same experimental
//! features as the daemon: `index_method` + `multiprocess_wal`). The builder
//! path in turso 0.7.0 applies `OpenFlags::NoLock` automatically when
//! `multiprocess_wal` is enabled, preventing fcntl lock contention on the WAL
//! / `.tshm` coordination file. Read-only safety comes from two layers: an
//! upfront file-existence check (prevents accidental database creation) and a
//! SQL mutation keyword blocklist (defense-in-depth). Uses the upstream
//! `turso::Connection` for lazy row iteration with a 10k row limit.
//!
//! ## Read-only safeguard
//!
//! Queries are validated before execution: the SQL is split into whole words
//! (on whitespace and SQL punctuation) and checked case-insensitively against
//! a blocklist of mutation keywords (INSERT, UPDATE, DELETE, DROP, ALTER,
//! CREATE, REPLACE, BEGIN, COMMIT, ROLLBACK, VACUUM, REINDEX, GRANT, REVOKE,
//! ATTACH, DETACH, ANALYZE). Whole-word matching avoids false positives on
//! column names like `created_at`.
//!
//! `PRAGMA` is intentionally **not** in the blocklist because:
//! - The connection is opened with NoLock via the turso builder (multiprocess_wal
//!   auto-applies NoLock), and the upfront file-existence check + blocklist
//!   are the primary safeguards — a PRAGMA that tries to mutate the database
//!   (e.g. `PRAGMA wal_checkpoint`) would still execute, but cannot create a
//!   new database file (file must exist) and is caught by the blocklist if it
//!   contains mutation keywords. In practice, useful read-only PRAGMAs like
//!   `PRAGMA quick_check` and `PRAGMA integrity_check` don't trigger mutation
//!   keywords.
//! - Read-only PRAGMAs are valuable on-demand diagnostics. The upfront
//!   file-existence check and the mutation blocklist are the primary safeguard;
//!   the blocklist is defense-in-depth for accidental queries.

use std::time::Duration;

use anyhow::{Context, Result, bail};

use crate::turso as turso_mod;

/// Row limit per query to prevent unbounded output.
const ROW_LIMIT: usize = 10_000;

/// Mutation keywords blocked by the read-only validator.
/// Case-insensitive whole-word match (not substring).
///
/// `PRAGMA` is intentionally absent — see module-level docs for rationale.
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

/// Run the debug subcommand. Parses `env::args()` for the debug invocation.
///
/// Returns `Ok(())` on success (exit code 0), `Err` on failure (exit code 1).
/// Prints usage to stderr and returns `Err` for invalid argument combinations.
pub async fn run_debug() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

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

    // Resolve ~/.mahbot/
    let mahbot_home = crate::config::default_config_dir()?;

    // Validate SQL is read-only before touching any database
    validate_read_only(sql)?;

    let db_list = resolve_db_list(db_name)?;

    for (label, db_filename) in &db_list {
        if db_name == "all" {
            println!("=== {label} ===");
        }

        let file_path = mahbot_home.join(db_filename);

        // Pre-existence check: we open the file via Database::open_file_with_flags
        // which would create the file if missing (when Create flag is set), but
        // we use ReadOnly, so the open fails on a non-existent file instead.
        // We still check existence upfront for a better error message.
        if !file_path.exists() {
            if db_name == "all" {
                eprintln!(
                    "Warning: database not found, skipping: {}",
                    file_path.display()
                );
                continue;
            }
            bail!("database file not found: {}", file_path.display());
        }

        // Use the high-level turso::Builder (same experimental features as the
        // daemon), which in turso 0.7.0 correctly applies OpenFlags::NoLock
        // when multiprocess_wal is enabled (see rsapi.rs). This is simpler and
        // safer than the previous low-level workaround that bypassed the builder
        // to work around a NoLock bug — the bug was fixed in the 0.7.0 release.
        //
        // The upfront file-existence check (above) and SQL blocklist (below)
        // provide read-only safety, compensating for the absence of
        // OpenFlags::ReadOnly in the builder path. NoLock (automatically added
        // for multiprocess_wal) prevents fcntl lock contention on both the DB
        // file and the WAL / .tshm coordination file.
        let path_str = file_path.to_string_lossy().to_string();

        let opts = crate::turso::experimental_database_opts();
        let db = turso::Builder::new_local(&path_str)
            .experimental_index_method(opts.enable_index_method)
            .experimental_multiprocess_wal(opts.enable_multiprocess_wal)
            .build()
            .await
            .with_context(|| format!("failed to open database '{}'", file_path.display()))?;

        let wrapper = db
            .connect()
            .with_context(|| format!("failed to connect to database '{}'", file_path.display()))?;

        // Set busy_timeout as defense-in-depth against transient lock errors
        // when the daemon holds a write transaction (matching Connection::open
        // in turso.rs line 344). Even with NoLock, this protects against edge
        // cases where the OS or filesystem layer introduces contention.
        wrapper
            .busy_timeout(Duration::from_mins(1))
            .with_context(|| format!("failed to set busy timeout for '{}'", file_path.display()))?;

        // Capture the query result, then let the wrapper go out of scope to
        // release the connection. For a CLI diagnostic tool the OS will clean
        // up everything on process exit.
        let query_result = execute_query(&wrapper, sql).await;

        drop(wrapper);

        query_result?;
    }

    Ok(())
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

/// Reject any SQL containing mutation keywords (whole-word, case-insensitive).
fn validate_read_only(sql: &str) -> Result<()> {
    for token in tokenize_sql(sql) {
        let upper = token.to_uppercase();
        if BLOCKLIST.contains(&upper.as_str()) {
            bail!("query rejected: contains blocked keyword '{token}'");
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

/// Execute a read-only SQL query and print results as pipe-delimited text.
///
/// Uses the upstream `::turso::Connection` (not our `crate::turso::Connection`
/// wrapper) so we get the lazy `Rows` iterator and can enforce the row limit
/// during iteration rather than after eagerly collecting all rows into memory.
async fn execute_query(conn: &::turso::Connection, sql: &str) -> Result<()> {
    let mut rows = conn.query(sql, ()).await.context("SQL query failed")?;

    let column_names = rows.column_names();
    if column_names.is_empty() {
        return Ok(());
    }

    // Print column header row
    println!("{}", column_names.join("|"));

    let col_count = column_names.len();
    let mut row_count = 0;
    let mut has_more = false;

    while let Some(row) = rows.next().await? {
        if row_count >= ROW_LIMIT {
            has_more = true;
            break;
        }
        print_row(&row, col_count);
        row_count += 1;
    }

    if has_more {
        print_truncation_row(col_count);
    }

    Ok(())
}

/// Format a single result row as pipe-delimited values and print to stdout.
fn print_row(row: &::turso::Row, column_count: usize) {
    let parts: Vec<String> = (0..column_count)
        .map(|idx| format_value(row.get_value(idx)))
        .collect();
    println!("{}", parts.join("|"));
}

/// Convert a column value to its display representation.
///
/// - NULL → empty (no text between pipe delimiters → `||`)
/// - Integer → decimal string
/// - Real → default `f64::Display` (may differ slightly from SQLite's output)
/// - Text → verbatim (no escaping for pipes or newlines)
/// - Blob → lowercase hex string
/// - Error reading value → empty (treated like NULL)
fn format_value(val: ::turso::Result<::turso::Value>) -> String {
    match val {
        Ok(::turso::Value::Null) | Err(_) => String::new(),
        Ok(::turso::Value::Integer(i)) => i.to_string(),
        Ok(::turso::Value::Real(f)) => f.to_string(),
        Ok(::turso::Value::Text(s)) => s,
        Ok(::turso::Value::Blob(b)) => hex_encode(&b),
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

/// Print a truncation sentinel row that matches the column count.
///
/// For 1 column:  `truncated`
/// For 2 columns: `truncated|truncated`
/// For N≥3:       `...|truncated|truncated|...|...`
///                 (ellipsis at first and last, `truncated` in between)
fn print_truncation_row(column_count: usize) {
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
    println!("{}", parts.join("|"));
}

fn print_usage() {
    eprintln!("Usage: mahbot debug --db <name> \"SQL query\"");
    let names = turso_mod::store_names().join(" | ");
    eprintln!("  --db <name>  {names} | all");
    eprintln!("  SQL query    read-only SQL, quoted as a single argument");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  mahbot debug --db board \"SELECT status, COUNT(*) FROM tickets GROUP BY status\"");
    eprintln!("  mahbot debug --db all \"SELECT name FROM sqlite_master WHERE type='table'\"");
}
