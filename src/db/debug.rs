//! Debug query tool — read-only SQL diagnostics for mahbot's databases.
//!
//! Invoked as `mahbot debug --db <name> ["SQL query"]` from the command line.
//! With a SQL argument it runs a read-only query (pipe-delimited output);
//! without one it prints a schema dump — per-table DDL blocks with row counts
//! (internal catalog artifacts excluded, see [`dump_schema`]). `--db all`
//! accepts both forms, dumping/querying every live store in per-store
//! sections. `-h`/`--help` prints usage (with the live database list) and
//! exits 0. Skips tracing initialization and GUI startup.
//!
//! ## Instance-up IPC routing
//!
//! Since `multiprocess_wal` was removed, an instance is the single-process
//! writer and holds the instance lock while it runs. `mahbot debug` must never
//! open a store an instance holds, so when the settled instance-lock probe
//! ([`crate::util::lock::instance_lock_state_settled`]) reports the location
//! [`Held`](crate::util::lock::InstanceLockState::Held) the CLI routes queries
//! through the instance's local IPC endpoint (`crate::db::ipc`) instead —
//! [`query_over_ipc`] for a query and [`dump_over_ipc`] for a schema dump. The
//! endpoint is scoped to this storage location, so two instances with different
//! locations never address each other. Write enforcement there is the store's
//! own read-only side — a reader connection refused for every change, or the
//! writing connection's `PRAGMA query_only` set around the query (see
//! [`crate::db::Connection::query_readonly`]).
//!
//! ## No-instance direct open
//!
//! When no instance holds the location ([`Free`](crate::util::lock::InstanceLockState::Free))
//! the CLI opens each store directly in single-process read-only mode via
//! `turso::core::Database::open_file_with_flags` with
//! `OpenFlags::ReadOnly | OpenFlags::NoLock` — no create and no whole-file
//! lock, so the open cannot create or mutate files. The connection layer is not
//! used here because it opens read-write and steps asynchronously, which is the
//! wrong shape for a caller that must never take the store's lock and steps its
//! statements on the current thread. The open reads committed WAL frames in
//! single-process mode.
//!
//! The routing is fail-closed ([`Unknown`](crate::util::lock::InstanceLockState::Unknown)
//! is never read as "no instance is running"): the CLI refuses outright and
//! opens nothing, so a live store is never opened directly while an instance may
//! hold it.
//!
//! Two defense-in-depth layers remain on top of the read-only open: an
//! upfront file-existence check and a SQL validator (mutation-keyword
//! blocklist plus a PRAGMA allowlist — see [`validate_read_only`]).
//!
//! ## Bounded reads
//!
//! One budget bounds the whole command, started where it starts reading (see
//! [`crate::db::read_limit`]). A read that spends it fails at once with the
//! limit's own message, naming the store or family the user asked about; a read
//! that never yields is ended by the watchdog, which exits the process with the
//! same message. Its own knob is [`crate::db::read_limit::ENV`].
//!
//! ## Other verbs
//!
//! - `mahbot debug families [--db <name>]` — list every forensic family in the
//!   store directory with its original store, artifact type, timestamp, total
//!   size, and a file-set/header classification. No engine is opened.
//! - `mahbot debug --family <id> "SQL query"` — query one forensic family
//!   (a static snapshot) with the same read-only guarantees as live stores.
//!   Families have no live-store layers; the engine opens single-process
//!   read-only. Engine panics on corrupt families are caught and reported as
//!   clean errors.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};

use crate::db as turso_mod;
use crate::db::ipc;
use crate::db::read_limit;
use crate::db::read_limit::{Budget, Read};
use crate::db::wal_guard;
use crate::util::lock::InstanceLockState;

/// Row limit per query to prevent unbounded output. Same LIMIT+1 semantic as
/// the instance-side IPC bound (`ipc::IPC_ROW_LIMIT`) so the two never drift.
const ROW_LIMIT: usize = ipc::IPC_ROW_LIMIT;

/// Mutation keywords blocked by the read-only validator.
/// Case-insensitive whole-word match (not substring).
const BLOCKLIST: &[&str] = &[
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "BEGIN", "COMMIT",
    "ROLLBACK", "VACUUM", "REINDEX", "GRANT", "REVOKE", "ATTACH", "DETACH", "ANALYZE",
];

/// SQL punctuation characters used for word-boundary tokenization.
/// Splitting on these ensures whole-word matching — a column named
/// `created_at` is not blocked by the `CREATE` keyword. Quote characters
/// (`'`, `"`, `[`, `]`) are NOT included: [`scan_sql`] consumes them in its
/// string/identifier arms before they reach the punctuation arm, and the `--`
/// `/*` comment starts are likewise handled before it.
const SQL_PUNCTUATION: &[char] = &[
    '(', ')', ';', ',', '.', '*', '+', '-', '/', '=', '<', '>', '!', '|', '&', '~', '{', '}', ':',
];

/// Run the debug subcommand. Parses `env::args()` for the debug invocation.
///
/// Returns `Ok(())` on success (exit code 0), `Err` on failure (exit code 1).
/// Prints usage to stderr and returns `Err` for invalid argument combinations.
pub async fn run_debug() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    run_debug_with_args(args, None).await
}

/// Resolve the mahbot storage root: the test/snapshot override or `~/.mahbot`.
fn resolve_home(home_override: Option<PathBuf>) -> Result<PathBuf> {
    match home_override {
        Some(home) => Ok(home),
        None => crate::config::default_config_dir(),
    }
}

/// Write `text` to stdout, mapping a closed pipe (EPIPE) to an ordinary
/// error instead of a panic — `println!`/`print!` panic on a closed stdout,
/// which would surface as a misleading failure (or crash) in piped
/// invocations. Single wording for every explicit stdout write.
fn write_stdout(text: &str) -> Result<()> {
    use std::io::Write as _;
    let mut out = std::io::stdout().lock();
    out.write_all(text.as_bytes())
        .and_then(|()| out.flush())
        .map_err(|e| anyhow!("{STDOUT_WRITE_ERROR_PREFIX}{e}"))
}

/// Print one line to stdout (EPIPE-safe, see [`write_stdout`]).
fn print_line(args: std::fmt::Arguments<'_>) -> Result<()> {
    write_stdout(&format!("{args}\n"))
}

/// Shared `[--db <name>]` flag parser for the families verb
/// (`mahbot debug <verb> --db <name>`); `None` means no flag was given.
fn parse_db_flag(args: &[String], subcommand: &str) -> Result<Option<String>> {
    match args.get(3).map(String::as_str) {
        Some("--db") => match args.get(4) {
            Some(name) => Ok(Some(name.clone())),
            None => bail!("expected: mahbot debug {subcommand} --db <name>"),
        },
        Some(other) => bail!("invalid {subcommand} argument '{other}'"),
        None => Ok(None),
    }
}

/// Validate a store name against the accepted set. `all_valid` appends the
/// literal `all` option to the hint — only callers that accept `--db all`
/// pass `true`.
///
/// The accepted set is [`turso_mod::debug_db_names`]: the logical store names
/// ([`turso_mod::store_names`]) plus the physical consolidated file name
/// ([`turso_mod::CONSOLIDATED_DB_NAME`], shown as the label in `--db all`
/// output).
pub(crate) fn validate_store_name(name: &str, all_valid: bool) -> Result<()> {
    let names = turso_mod::debug_db_names();
    if names.contains(&name) {
        return Ok(());
    }
    let hint = names.join(", ") + if all_valid { ", all" } else { "" };
    bail!("invalid database name '{name}'. Valid names: {hint}");
}

// ── Forensic families (quarantine / pre-reindex) ─────────────────────────

/// Artifact type of a forensic family. Variant order is the listing sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FamilyKind {
    /// Renamed-aside artifact family: the original family the verify-then-swap
    /// rebuild moved aside before putting the rebuilt store in place (never
    /// deleted).
    Quarantine,
    /// Pre-repair copy taken before an in-place REINDEX.
    PreReindex,
}

impl FamilyKind {
    #[must_use]
    fn label(self) -> &'static str {
        match self {
            Self::Quarantine => "quarantine",
            Self::PreReindex => "pre-reindex",
        }
    }
}

/// One forensic family discovered in the store directory.
#[derive(Debug)]
struct FamilyInfo {
    /// Base file name — the id fed back into `--family <id>`.
    id: String,
    /// Original store name (the family's `{store}.db` prefix).
    store: String,
    kind: FamilyKind,
    /// `%Y%m%dT%H%M%SZ` stamp from the family name.
    stamp: String,
    /// Total size in bytes of every present family file.
    size: u64,
    /// File-set/header classification (family-specific — never a live-store
    /// class, which is meaningless for static files).
    class: FamilyClass,
    /// Present members, comma-joined (`db,wal`).
    files: String,
}

/// State classification of one forensic family (snapshot semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FamilyClass {
    /// Main DB present with a valid header and its `-wal` beside it.
    Complete,
    /// Main DB present with a valid header, but its `-wal` is missing.
    Partial,
    /// Main DB present but its header is corrupt/empty — queryable, likely fails.
    BadHeader,
    /// No main DB file (a `-wal`-only quarantine) — not queryable.
    SidecarOnly,
}

impl FamilyClass {
    #[must_use]
    fn label(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Partial => "partial",
            Self::BadHeader => "bad-header",
            Self::SidecarOnly => "sidecar-only",
        }
    }
}

/// Parsed family-name components (see [`parse_family_name`]).
#[derive(Debug)]
pub(crate) struct FamilyMeta {
    pub(crate) store: String,
    pub(crate) kind: FamilyKind,
    pub(crate) stamp: String,
}

/// Parse a forensic-family base file name:
/// `{store}.db.quarantine-{stamp}-{pid}[-{seq}]` or
/// `{store}.db.pre-reindex-{stamp}-{pid}` (stamp = `%Y%m%dT%H%M%SZ`).
///
/// Rejects sidecar members (`-wal`-suffixed names), foreign files,
/// and any path-like name — the path-safety gate for `--family <id>`: only a
/// parsed family id reaches the store directory. The stamp is shape-checked
/// (16 chars, digits around `T`/`Z`), not calendar-validated; a store name of
/// `.` or `..` parses (flat filename — no traversal possible).
///
/// Every part is matched without ASCII case — the volumes the stores live on
/// compare file names that way, so a differently-cased spelling of a family name
/// is the same file — while the parsed fields keep the spelling they came in as.
///
/// The formats are written by `db::quarantine_family` (quarantine) and the
/// pre-reindex snapshot in `db::snapshot_store_via_engine` (REINDEX repair) —
/// keep both sides of the naming contract in sync; the writer round-trip test
/// locks the coupling.
pub(crate) fn parse_family_name(name: &str) -> Option<FamilyMeta> {
    let lower = name.to_ascii_lowercase();
    let (kind, marker) = if lower.contains(".quarantine-") {
        (FamilyKind::Quarantine, ".quarantine-")
    } else if lower.contains(".pre-reindex-") {
        (FamilyKind::PreReindex, ".pre-reindex-")
    } else {
        return None;
    };
    // The lowercased copy is the same length with the same ASCII offsets, so the
    // marker's position maps onto the caller's own spelling.
    let at = lower.find(marker)?;
    let (prefix, tail) = (&name[..at], &name[at + marker.len()..]);
    let store = crate::util::strip_suffix_ignoring_case(prefix, ".db")?;
    if store.is_empty() || store.contains('/') || store.contains('\\') {
        return None;
    }
    // tail = "{stamp}-{pid}[-{seq}]"
    let (stamp, pid_rest) = tail.split_once('-')?;
    if !is_family_stamp(stamp) {
        return None;
    }
    let (pid, seq) = match pid_rest.split_once('-') {
        Some((pid, seq)) => (pid, seq),
        None => (pid_rest, ""),
    };
    if pid.is_empty() || pid_rest.ends_with('-') || !pid.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // Pre-reindex names never carry a seq suffix; both reject non-digit tails.
    if kind == FamilyKind::PreReindex && !seq.is_empty() {
        return None;
    }
    if !seq.is_empty() && !seq.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(FamilyMeta {
        store: store.to_string(),
        kind,
        stamp: stamp.to_string(),
    })
}

/// True for the `%Y%m%dT%H%M%SZ` stamp shape (16 chars, digits around `T`/`Z`,
/// either case — the files are named on volumes that compare names without it).
fn is_family_stamp(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 16
        && b[8].eq_ignore_ascii_case(&b'T')
        && b[15].eq_ignore_ascii_case(&b'Z')
        && b[..8].iter().all(u8::is_ascii_digit)
        && b[9..15].iter().all(u8::is_ascii_digit)
}

/// Discover all forensic families under the store directory and classify them.
///
/// Families are found from their base file name or from a `-wal`-only member
/// (such a quarantine has no base file). Foreign files (`.DS_Store`, live
/// stores) are ignored. Sorted by (store, kind, stamp).
fn list_families(root: &Path) -> Result<Vec<FamilyInfo>> {
    let db_dir = super::store_dir(root);
    // Fresh install: no store directory yet — nothing to list, not an error.
    let entries = match std::fs::read_dir(&db_dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(e)
                .with_context(|| format!("failed to read store directory: {}", db_dir.display()));
        }
    };
    let names: Vec<String> = entries
        .map(|e| {
            e.map(|e| e.file_name().to_string_lossy().into_owned())
                .with_context(|| format!("failed to read entry in {}", db_dir.display()))
        })
        .collect::<Result<_>>()?;

    // Base names first; then sidecar-only families (entries whose base never
    // appeared — a quarantine can move only the `-wal`).
    let mut bases: std::collections::BTreeMap<String, FamilyMeta> =
        std::collections::BTreeMap::new();
    for name in &names {
        if let Some(meta) = parse_family_name(name) {
            bases.insert(name.clone(), meta);
        }
    }
    for name in &names {
        if let Some(base) = name.strip_suffix("-wal")
            && !bases.contains_key(base)
            && let Some(meta) = parse_family_name(base)
        {
            bases.insert(base.to_string(), meta);
        }
    }

    let mut families: Vec<FamilyInfo> = bases
        .into_iter()
        .map(|(id, meta)| classify_family(root, id, meta))
        .collect();
    families.sort_by(|a, b| (&a.store, a.kind, &a.stamp).cmp(&(&b.store, b.kind, &b.stamp)));
    Ok(families)
}

/// Expected members of a forensic family — one source of truth for both the
/// listing and the Complete check: the main file plus the engine's `-wal`. A
/// pre-reindex snapshot has the same shape: `VACUUM INTO` leaves an empty
/// `<dest>-wal` beside the copy (see `snapshot_store_via_engine`).
const FAMILY_MEMBERS: [(&str, &str); 2] = [("", "db"), ("-wal", "wal")];

/// Classify one family's file set — pure filesystem inspection, never opens.
fn classify_family(root: &Path, id: String, meta: FamilyMeta) -> FamilyInfo {
    let db_path = super::store_dir(root).join(&id);
    let mut members: Vec<&'static str> = Vec::new();
    let mut size: u64 = 0;
    for (suffix, label) in FAMILY_MEMBERS {
        let path = db_path.with_file_name(format!("{id}{suffix}"));
        if let Ok(md) = std::fs::metadata(&path)
            && md.is_file()
        {
            members.push(label);
            size += md.len();
        }
    }
    let complete = FAMILY_MEMBERS
        .iter()
        .all(|(_, label)| members.contains(label));
    let class = if !members.contains(&"db") {
        FamilyClass::SidecarOnly
    } else if !db_header_ok(&db_path) {
        FamilyClass::BadHeader
    } else if complete {
        FamilyClass::Complete
    } else {
        FamilyClass::Partial
    };
    FamilyInfo {
        id,
        store: meta.store,
        kind: meta.kind,
        stamp: meta.stamp,
        size,
        class,
        files: members.join(","),
    }
}

/// Main-DB header sanity without opening the database: SQLite magic plus a
/// valid page size (the same checks wal_guard applies to live stores).
fn db_header_ok(db_path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(db_path) else {
        return false;
    };
    if meta.len() < wal_guard::DB_HEADER_MIN_SIZE {
        return false; // empty or truncated header
    }
    wal_guard::read_db_header(db_path).is_some_and(|h| wal_guard::db_header_valid(&h))
}

/// `mahbot debug families [--db <name>]` — list all forensic families
/// (quarantine + pre-reindex) with a stat plus a header read per family, never
/// an engine open. Informational: exits 0 whether or not families exist.
fn run_debug_families(args: &[String], home_override: Option<PathBuf>) -> Result<()> {
    let mahbot_home = resolve_home(home_override)?;
    let mut families = list_families(&mahbot_home)?;
    // `--db <name>` filters by store name; a name matching nothing (canonical
    // store with no families, unknown/legacy name) prints nothing and exits 0
    // — the filter never depends on the current listing content.
    let filter = parse_db_flag(args, "families")?;
    if let Some(store) = filter {
        families.retain(|fam| fam.store == store);
    }
    for fam in &families {
        print_line(format_args!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            fam.id,
            fam.store,
            fam.kind.label(),
            fam.stamp,
            fam.size,
            fam.class.label(),
            fam.files
        ))?;
    }
    Ok(())
}

/// Open a forensic family read-only and execute `sql`, returning the rendered
/// output. Split from [`query_family`] so tests can assert result values
/// instead of only success.
///
/// Snapshot semantics: the family is static, so none of the live-store layers
/// apply — no IPC routing, no live heuristics. The engine opens the family in
/// place, read-only (`db` + `-wal`), on the shared read-only stack
/// ([`open_and_run_readonly`]) — so the read limit bounds a family read exactly
/// as it bounds a `--db` one, and a damaged family yields a clean error instead
/// of a CLI crash.
fn execute_family_query(read: Read<'_>, family_id: &str, sql: &str, root: &Path) -> Result<String> {
    parse_family_name(family_id).with_context(|| {
        format!("invalid family id '{family_id}' — list valid ids with `mahbot debug families`")
    })?;
    let db_path = super::store_dir(root).join(family_id);
    if !db_path.exists() {
        // A sidecar-only family (no base file — a quarantine that moved only
        // the `-wal`, or a failed pre-reindex copy) is a real family that
        // cannot be opened; a well-formed id with no member at all is a stale
        // or hand-constructed id — report the two distinctly.
        if turso_mod::wal_path(&db_path).exists() {
            bail!(
                "forensic family '{family_id}' has no main database file — \
                 the family cannot be queried"
            );
        }
        bail!(
            "forensic family '{family_id}' not found — list valid ids with \
             `mahbot debug families`"
        );
    }
    // Snapshot semantics: the main file must be a regular file — a symlink
    // could redirect the open to a live store, bypassing the read-only IPC
    // routing / no-instance validation the `--db` path applies.
    let md = std::fs::symlink_metadata(&db_path)
        .with_context(|| format!("cannot stat forensic family '{family_id}'"))?;
    if !md.file_type().is_file() {
        bail!("forensic family '{family_id}' main file is not a regular file — refusing to open");
    }
    let result = open_and_run_readonly(read, &db_path, |read, io, db, path| {
        connect_execute(read, io, db, sql, path)
    });
    match result {
        Ok(output) => Ok(output),
        // An engine panic is deterministic — classify it as a panic before the
        // generic error fallback. The arm is gated by `is_engine_panic_error`,
        // so `{e:#}` already starts with the prefix — a single message, no
        // double-named chain.
        Err(e) if is_engine_panic_error(&e) => {
            bail!("forensic family '{family_id}' could not be read — {e:#}")
        }
        Err(e) => {
            Err(e).with_context(|| format!("forensic family '{family_id}' could not be read"))
        }
    }
}

/// `mahbot debug --family <id> "SQL query"` — execute and print.
fn query_family(read: Read<'_>, family_id: &str, sql: &str, root: &Path) -> Result<()> {
    let output = execute_family_query(read, family_id, sql, root)?;
    // Write explicitly: a closed stdout (EPIPE) surfaces as an ordinary I/O
    // error, never a panic misreported as an engine crash.
    write_stdout(&output)
}

/// Testable core of [`run_debug`].
///
/// `args` has the same shape as `std::env::args()` (`args[0]` is the program
/// name, `args[1]` is `"debug"`). `home_override`, when provided, replaces the
/// `~/.mahbot` storage-root resolution (used by tests with temporary roots and
/// by the snapshot-copy procedure).
#[expect(clippy::too_many_lines)] // one linear per-store walk: route, read, per-store failure
async fn run_debug_with_args(args: Vec<String>, home_override: Option<PathBuf>) -> Result<()> {
    // args[0] = "mahbot", args[1] = "debug"
    let tail: Vec<&str> = args.iter().skip(2).map(String::as_str).collect();
    // Help at any verb position prints the same usage. An exact `--help`/`-h`
    // element anywhere in the tail (verb, flags, or trailing position) never
    // falls through to SQL: store/family ids never equal `--help`/`-h`, and a
    // SQL argument of exactly `--help` is a comment, not a statement, while a
    // lone `-h` is a syntax error (a single-dash token can never parse as SQL)
    // — intercepting either loses no valid query.
    if tail.contains(&"--help") || tail.contains(&"-h") {
        print_usage();
        return Ok(());
    }

    // `mahbot debug families [--db <name>]` — list forensic families, no opens.
    if args.get(2).is_some_and(|a| a == "families") {
        return run_debug_families(&args, home_override);
    }

    // `mahbot debug --family <id> "SQL query"` — read-only query on one
    // forensic family (snapshot semantics: no IPC routing, no live heuristics).
    if args.get(2).is_some_and(|a| a == "--family") {
        if args.len() < 5 {
            print_usage();
            bail!("expected: mahbot debug --family <id> \"SQL query\"");
        }
        let mahbot_home = resolve_home(home_override)?;
        let sql = &args[4];
        validate_read_only(sql)?;
        let budget = Budget::start();
        let read = Read::direct(&args[3], &budget);
        return query_family(read, &args[3], sql, &mahbot_home);
    }

    // Need at least: mahbot debug --db <name> [SQL]
    if args.len() < 4 {
        print_usage();
        bail!("expected: mahbot debug --db <name> [\"SQL query\"]");
    }

    if args[2] != "--db" {
        eprintln!("Error: expected --db flag, got '{}'", args[2]);
        print_usage();
        bail!("expected --db flag");
    }

    let db_name = &args[3];
    // A SQL argument runs a read-only query; without one the command prints a
    // schema dump (per-table DDL blocks + row counts — see [`dump_schema`]).
    let sql = args.get(4).map(String::as_str);

    // Resolve ~/.mahbot/ (or the test/snapshot override).
    let mahbot_home = resolve_home(home_override)?;

    // Validate SQL is read-only before touching any database. The dump mode
    // has no user SQL — its queries are internal and read-only by
    // construction.
    if let Some(sql) = sql {
        validate_read_only(sql)?;
    }

    let db_list = resolve_db_list(db_name, &mahbot_home)?;

    // Routing hinges on whether an instance is the single-process writer right
    // now: if it holds the instance lock the CLI must NOT open the live store
    // directly, so it goes through the debug IPC endpoint (scoped to this
    // storage location); if no instance holds it, a direct read-only open is
    // safe. `instance_lock_state_settled` re-checks over a short window so a
    // self-update handoff (the lock momentarily free between the old instance's
    // release and the new instance's re-acquire) never falls through to a
    // concurrent direct open. The third state refuses — see the module doc's
    // fail-closed rule.
    let route_via_channel = match crate::util::lock::instance_lock_state_settled(&mahbot_home) {
        InstanceLockState::Held => true,
        InstanceLockState::Free => false,
        InstanceLockState::Unknown(e) => bail!(
            "{}. Re-run when no instance is running.",
            crate::util::lock::lock_state_unknown(&mahbot_home, &e)
        ),
    };

    // One read budget for the whole command, spent by every read under it and
    // never reset per store (see [`crate::db::read_limit`]).
    let budget = Budget::start();
    let mut failures = 0usize;
    for (label, file_path) in &db_list {
        if db_name == "all" {
            print_line(format_args!("=== {label} ==="))?;
        }

        // Physical store name the IPC endpoint keys on (the consolidated
        // "core" file or "logs"), derived from the file stem.
        let physical = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .map_or_else(|| label.clone(), str::to_string);

        // The user's own label is the target, so a failure names it.
        let read = if route_via_channel {
            Read::over_channel(label, &budget)
        } else {
            Read::direct(label, &budget)
        };

        let result = if route_via_channel {
            match sql {
                Some(sql) => query_over_ipc(read, &mahbot_home, &physical, sql).await,
                None => dump_over_ipc(read, &mahbot_home, &physical, label).await,
            }
        } else {
            // No instance holds the location — direct single-process read-only
            // open. Pre-existence check: the read-only open fails on a
            // non-existent file, but we check existence upfront for a better
            // error message.
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
            match sql {
                Some(sql) => open_and_query_readonly(read, file_path, sql),
                None => open_and_dump_readonly(read, file_path, label),
            }
        };
        match result {
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

/// Run one read-only SQL query through the instance's debug IPC endpoint,
/// returning the response or the endpoint's error.
async fn ipc_query_readonly(
    read: Read<'_>,
    root: &Path,
    store: &str,
    sql: &str,
) -> Result<ipc::QueryResponse> {
    let req = ipc::IpcRequest {
        store: store.to_string(),
        sql: sql.to_string(),
        params: Vec::new(),
        call: None,
    };
    let resp = ipc::ipc_query_with_wait(root, &req, read).await?;
    if let Some(err) = &resp.error {
        bail!("{err}");
    }
    Ok(resp)
}

/// Query a live store through the instance's debug IPC endpoint (used when it
/// holds the storage location and the CLI must not open the store directly).
/// `physical` is the physical store name the endpoint keys on
/// ("core" or "logs").
async fn query_over_ipc(read: Read<'_>, root: &Path, physical: &str, sql: &str) -> Result<()> {
    // The guard spans the reading only, as on the direct route: the answer is
    // written after it is dropped, so a read that came back is printed as its
    // answer and is never cut.
    let out = {
        let _watchdog = read.arm()?;
        let resp = ipc_query_readonly(read, root, physical, sql).await?;
        let rows = resp
            .rows
            .iter()
            .map(|row| row.iter().map(ipc::WireValue::format).collect::<Vec<_>>())
            .collect();
        render_pipe_table(&resp.columns, rows, resp.truncated)
    };
    write_stdout(&out)
}

/// Schema dump of a live store through the instance's debug IPC endpoint. One
/// watchdog for the whole walk — the table list and every per-table count run
/// under the read's own process-ending half, not one per round trip — and the
/// dump is written after it is dropped, like the direct route's.
async fn dump_over_ipc(read: Read<'_>, root: &Path, physical: &str, label: &str) -> Result<()> {
    use std::fmt::Write as _;
    let dump = {
        let _watchdog = read.arm()?;
        let tables_sql = USER_TABLES_SQL.replace("{filter}", turso_mod::USER_OBJECT_FILTER);
        let resp = ipc_query_readonly(read, root, physical, &tables_sql).await?;
        let mut out = format!("== schema dump: {label} ==\n");
        for row in &resp.rows {
            // Many round trips: stop the walk on a spent budget instead of starting
            // another one.
            read.check()?;
            let name = row.first().map(ipc::WireValue::format).unwrap_or_default();
            let sql = row.get(1).map(ipc::WireValue::format).unwrap_or_default();
            if name.is_empty() {
                continue;
            }
            let count_sql = format!("SELECT COUNT(*) FROM {}", quote_ident(&name));
            let count_resp = ipc_query_readonly(read, root, physical, &count_sql).await?;
            let count = count_resp
                .rows
                .first()
                .and_then(|r| r.first())
                .map(ipc::WireValue::format)
                .unwrap_or_default();
            writeln!(out, "\n[table] {name}\n{sql}\nrows: {count}\n").expect("writing to a String");
        }
        out
    };
    write_stdout(&dump)
}

/// Prefix of [`guard_panics`] errors — shared with the classifier so a wording
/// change updates both sides (string-matching convention; a marker error type
/// would be more robust but heavier).
const ENGINE_PANIC_PREFIX: &str = "the database engine panicked while reading the store: ";

/// Prefix of stdout-write failures from [`write_stdout`] — the single wording
/// for every explicit stdout write (query results, listings, banners), so a
/// closed pipe surfaces consistently as an I/O error, never a panic.
const STDOUT_WRITE_ERROR_PREFIX: &str = "failed writing to stdout: ";

/// Run a blocking store open/query under [`crate::shutdown::contain_panics`],
/// converting a panic into a clean error — turso_core panics on specific
/// corruption shapes (pager/wal index OOB — the documented prod class), and
/// the debug CLI must never crash on a damaged store. `contain_panics` also
/// keeps the caught panic off stderr: the boot hook skips a panic raised
/// inside it.
fn guard_panics<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    match crate::shutdown::contain_panics(f) {
        Ok(r) => r,
        Err(payload) => Err(anyhow!(
            "{ENGINE_PANIC_PREFIX}{}",
            crate::util::panic_message(&*payload)
        )),
    }
}

/// True for errors produced by [`guard_panics`] (an engine panic, not an
/// ordinary query/open error).
fn is_engine_panic_error(err: &anyhow::Error) -> bool {
    format!("{err:#}").starts_with(ENGINE_PANIC_PREFIX)
}

/// Open a store file read-only with the given engine opts.
///
/// Also used by the `bench-openrouter` subcommand for read-only config-store
/// lookups (worker model / provider key fallbacks) — same guarantees, same
/// no-write path as the debug CLI.
///
/// Uses the low-level `turso::core` API because only it can express
/// `OpenFlags::ReadOnly | OpenFlags::NoLock`: no create and no whole-file lock,
/// so the open cannot create or mutate files. The core IO is synchronous
/// (`FsIO`), so this is intentionally blocking; the debug CLI runs on a
/// single-task current-thread runtime.
pub(crate) fn open_readonly(
    db_path: &Path,
    opts: turso::core::DatabaseOpts,
) -> Result<(
    std::sync::Arc<dyn turso::core::IO>,
    std::sync::Arc<turso::core::Database>,
)> {
    let path_str = db_path
        .to_str()
        .with_context(|| format!("database path must be UTF-8: {}", db_path.display()))?;
    let io: std::sync::Arc<dyn turso::core::IO> =
        std::sync::Arc::new(turso::core::PlatformIO::new()?);
    let db = turso::core::Database::open_file_with_flags(
        io.clone(),
        path_str,
        turso::core::OpenFlags::ReadOnly | turso::core::OpenFlags::NoLock,
        opts,
        None,
        std::sync::Arc::new(turso::core::SqliteDialect),
    )
    .map_err(|e| {
        anyhow!(
            "failed to open database '{}' read-only: {e}",
            db_path.display()
        )
    })?;
    Ok((io, db))
}

/// Connect to an opened store, applying the in-memory temp-store setting and
/// the per-statement limit.
///
/// Also used by the `bench-openrouter` subcommand for read-only config-store
/// lookups — see [`open_readonly`].
///
/// The limit is armed here because this connection serves caller-supplied SQL:
/// it is what stops one statement from holding the store for as long as it
/// likes. The engine terminates a statement that overruns
/// [`crate::db::store::statement_limit`] and the connection stays usable, so
/// the store answers the next query instead of being held until the statement
/// ends on its own.
///
/// Every turso connection the application opens — the service's opening path
/// (the writing connection and each reader connection, both of which set the
/// pragma as they open) and this debug CLI path (they do not share an opening
/// path) — runs with in-memory temp storage, so no statement or transaction can
/// ever fail because a temp directory is missing. turso_core's tempdir
/// creation has no parent-creation and no fallback chain, and resolves
/// through `$TMPDIR`, which the service pins to its private root; a missing
/// root used to fail every eager-temp statement (RETURNING buffers, ORDER
/// BY/LIMIT heap sorts, DISTINCT, subqueries, window functions, spills)
/// with "I/O error (tempdir): entity not found". The PRAGMA maps to the
/// per-connection `set_temp_store` (turso_core translate/pragma.rs); if a
/// future engine rejects it, the CLI fails loudly here instead of silently
/// regressing to disk-backed temp storage.
pub(crate) fn connect_readonly(
    db: &std::sync::Arc<turso::core::Database>,
    db_path: &Path,
) -> Result<std::sync::Arc<turso::core::Connection>> {
    let conn = db
        .connect()
        .map_err(|e| anyhow!("failed to connect to database '{}': {e}", db_path.display()))?;
    conn.set_query_timeout(crate::db::store::statement_limit());
    conn.execute("PRAGMA temp_store = MEMORY").map_err(|e| {
        anyhow!(
            "failed to set in-memory temp storage on '{}': {e}",
            db_path.display()
        )
    })?;
    Ok(conn)
}

/// Connect to an opened store and execute `sql`, returning the pipe-delimited
/// output. Output is buffered and returned only on success: a mid-query failure
/// must not leave a partial column header on stdout.
fn connect_execute(
    read: Read<'_>,
    io: &std::sync::Arc<dyn turso::core::IO>,
    db: &std::sync::Arc<turso::core::Database>,
    sql: &str,
    db_path: &Path,
) -> Result<String> {
    let conn = connect_readonly(db, db_path)?;
    execute_query_readonly(read, io, &conn, sql, db_path)
}

/// Open a store read-only and execute `sql`, printing pipe-delimited results.
/// See [`open_and_run_readonly`] for the shared open+run stack; the write
/// happens after it returns, so it is never inside the watchdog's scope.
fn open_and_query_readonly(read: Read<'_>, file_path: &Path, sql: &str) -> Result<()> {
    let output = open_and_run_readonly(read, file_path, |read, io, db, path| {
        connect_execute(read, io, db, sql, path)
    })?;
    write_stdout(&output)
}

/// Open a store read-only and run `runner` against it. Used by the query and
/// schema-dump paths and by the `--family` snapshot route, so every direct read
/// shares one open+limit stack. The watchdog is armed before the open and covers
/// the READ alone: the caller writes the output after this returns, so a read that
/// finishes inside the watchdog's grace is never cut mid-write (which is what the
/// channel route does too). The whole open+run is panic-guarded — a damaged store
/// must yield a clean error, never a CLI crash.
fn open_and_run_readonly<T>(
    read: Read<'_>,
    file_path: &Path,
    runner: impl FnOnce(
        Read<'_>,
        &std::sync::Arc<dyn turso::core::IO>,
        &std::sync::Arc<turso::core::Database>,
        &Path,
    ) -> Result<T>,
) -> Result<T> {
    guard_panics(|| {
        let _watchdog = read.arm()?;
        let (io, db) = open_readonly(file_path, turso_mod::experimental_database_opts())?;
        let out = runner(read, &io, &db, file_path)?;
        // A read that came back past the limit is not an answer.
        read.check()?;
        Ok(out)
    })
}

/// [`open_and_run_readonly`] runner for the schema dump: build the whole
/// store's dump text (buffered — a mid-dump failure never leaks a partial
/// dump to stdout, matching [`connect_execute`]'s all-or-nothing output),
/// then write it once, after the watchdog is disarmed.
fn open_and_dump_readonly(read: Read<'_>, file_path: &Path, label: &str) -> Result<()> {
    let dump = open_and_run_readonly(read, file_path, |read, io, db, path| {
        dump_schema(read, io, db, path, label)
    })?;
    write_stdout(&dump)
}

/// Execute a read-only query on a `turso::core` connection and return the
/// results as pipe-delimited text.
///
/// Output is accumulated in memory and returned as a single string; the caller
/// prints it only on success, so a mid-query failure never leaks a partial
/// result set (e.g. a bare column header) to stdout.
fn execute_query_readonly(
    read: Read<'_>,
    io: &std::sync::Arc<dyn turso::core::IO>,
    conn: &std::sync::Arc<turso::core::Connection>,
    sql: &str,
    db_path: &Path,
) -> Result<String> {
    let mut stmt = prepare_readonly(conn, sql, db_path)?;

    let col_count = stmt.num_columns();
    if col_count == 0 {
        return Ok(String::new());
    }

    let column_names: Vec<String> = (0..col_count)
        .map(|i| stmt.get_column_name(i).into_owned())
        .collect();
    let (rows, has_more) = step_rows(read, io, db_path, &mut stmt, Some(ROW_LIMIT), |row| {
        format_core_row(row, col_count)
    })?;

    Ok(render_pipe_table(&column_names, rows, has_more))
}

/// SQL selecting a store's user tables: `type='table'`, with real DDL, and not
/// an internal catalog artifact.
///
/// `sql IS NOT NULL` alone is NOT a sufficient filter in turso — the internal
/// sequence/shadow tables (`sqlite_sequence`, `__turso_internal_*`) carry real
/// DDL — so the name-prefix exclusion is required. The exclusion is the shared
/// [`turso_mod::USER_OBJECT_FILTER`] (the same predicate turso's integrity
/// scans use), so the debug CLI and the engine cannot drift on what counts as
/// internal. FTS5 shadow tables (NULL sql) and auto-indexes are excluded by
/// `sql IS NOT NULL` + `type='table'`.
const USER_TABLES_SQL: &str = "SELECT name, sql FROM sqlite_master \
     WHERE type = 'table' AND sql IS NOT NULL AND {filter} \
     ORDER BY name";

/// Build the schema-dump text for one store.
///
/// Format (block, not pipe-delimited — DDL contains newlines and pipes):
///
/// ```text
/// == schema dump: board ==
///
/// [table] ticket_comments
/// CREATE TABLE ticket_comments(...);
/// rows: 7
///
/// [table] tickets
/// CREATE TABLE tickets(...);
/// rows: 4213
/// ```
///
/// Tables are ordered by name. Internal catalog artifacts — `sqlite_%`
/// (`sqlite_sequence`, …) and `__turso_internal_%` (turso's sequence and FTS
/// directory tables) — are excluded: the dump shows only meaningful user
/// objects, and it reflects the LIVE stored schema (`sqlite_master`), not
/// source-code constants (live schemas drift from source over time).
fn dump_schema(
    read: Read<'_>,
    io: &std::sync::Arc<dyn turso::core::IO>,
    db: &std::sync::Arc<turso::core::Database>,
    db_path: &Path,
    label: &str,
) -> Result<String> {
    use std::fmt::Write as _;
    let conn = connect_readonly(db, db_path)?;
    let tables_sql = USER_TABLES_SQL.replace("{filter}", turso_mod::USER_OBJECT_FILTER);
    let tables = collect_rows(read, io, &conn, &tables_sql, db_path, |row| {
        let name = format_core_value(row.get_value(0));
        let sql = format_core_value(row.get_value(1));
        (name, sql)
    })?;

    let mut out = format!("== schema dump: {label} ==\n");
    for (name, sql) in tables {
        if name.is_empty() {
            continue;
        }
        let count_sql = format!("SELECT COUNT(*) FROM {}", quote_ident(&name));
        let counts = collect_rows(read, io, &conn, &count_sql, db_path, |row| {
            format_core_value(row.get_value(0))
        })?;
        let count = counts.first().ok_or_else(|| {
            anyhow!(
                "row count query returned no rows on '{}'",
                db_path.display()
            )
        })?;
        write!(out, "\n[table] {name}\n{sql}\nrows: {count}\n")
            .expect("writing to a String cannot fail");
    }
    Ok(out)
}

/// Prepare a read-only statement on a `turso::core` connection, mapping prepare
/// failures to the same error strings used across the debug tooling.
fn prepare_readonly(
    conn: &std::sync::Arc<turso::core::Connection>,
    sql: &str,
    db_path: &Path,
) -> Result<turso::core::Statement> {
    conn.query(sql)
        .map_err(|e| anyhow!("SQL query failed on '{}': {e}", db_path.display()))?
        .ok_or_else(|| anyhow!("query produced no statement on '{}'", db_path.display()))
}

/// Shared IO/Yield/Busy/Interrupt stepping loop for all read-only
/// `turso::core` queries.
///
/// `limit` stops stepping with `has_more = true` when a row would exceed the
/// cap; that row is NOT collected. Callers buffer output so a mid-query
/// failure never leaks partial results.
///
/// An interrupt is always [`crate::db::store::statement_limit`] terminating a
/// statement that overran it — nothing else interrupts these connections — so
/// it reports the shared caller-facing sentence ([`crate::db::store::too_heavy_message`]),
/// not the engine's bare "interrupted".
fn step_rows<T>(
    read: Read<'_>,
    io: &std::sync::Arc<dyn turso::core::IO>,
    db_path: &Path,
    stmt: &mut turso::core::Statement,
    limit: Option<usize>,
    mut collect: impl FnMut(&turso::core::Row) -> T,
) -> Result<(Vec<T>, bool)> {
    let mut rows = Vec::new();
    let mut has_more = false;
    loop {
        // Cooperative half of the command's read limit: end the read on a spent budget.
        read.check()?;
        match stmt
            .step()
            .map_err(|e| anyhow!("SQL query failed on '{}': {e}", db_path.display()))?
        {
            turso::core::StepResult::Done => break,
            turso::core::StepResult::IO
            | turso::core::StepResult::Yield
            | turso::core::StepResult::Sleep { .. } => {
                io.step()
                    .map_err(|e| anyhow!("SQL query failed on '{}': {e}", db_path.display()))?;
            }
            turso::core::StepResult::Row => {
                if limit.is_some_and(|l| rows.len() >= l) {
                    has_more = true;
                    break;
                }
                let row = stmt
                    .row()
                    .ok_or_else(|| anyhow!("row missing after StepResult::Row"))?;
                rows.push(collect(row));
            }
            turso::core::StepResult::Interrupt => bail!(
                "SQL query failed on '{}': {}",
                db_path.display(),
                crate::db::store::too_heavy_message()
            ),
            turso::core::StepResult::Busy => {
                bail!("database busy on '{}'; try again later", db_path.display())
            }
        }
    }
    Ok((rows, has_more))
}

/// Step a single statement to completion, collecting one owned value per row
/// (uncapped [`step_rows`]). A failure propagates and the caller reports it as
/// a clean error.
fn collect_rows<T>(
    read: Read<'_>,
    io: &std::sync::Arc<dyn turso::core::IO>,
    conn: &std::sync::Arc<turso::core::Connection>,
    sql: &str,
    db_path: &Path,
    collect: impl FnMut(&turso::core::Row) -> T,
) -> Result<Vec<T>> {
    let mut stmt = prepare_readonly(conn, sql, db_path)?;
    let (rows, _has_more) = step_rows(read, io, db_path, &mut stmt, None, collect)?;
    Ok(rows)
}

/// Double-quote a SQL identifier, escaping embedded double quotes.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// The physical database files for the debug CLI, each opened once.
///
/// After consolidation all the domain stores share ONE file, so `--db all`
/// opens each unique file once rather than once per logical domain name. Labels
/// are the physical store names from
/// [`turso_mod::iter_checkpoint_stores`].
fn physical_store_list(root: &Path) -> Vec<(String, PathBuf)> {
    turso_mod::iter_checkpoint_stores()
        .map(|(name, _)| (name.to_string(), turso_mod::store_db_path(root, name)))
        .collect()
}

/// Map a `--db` argument to a list of `(label, absolute db path)` pairs.
fn resolve_db_list(name: &str, root: &Path) -> Result<Vec<(String, PathBuf)>> {
    if name == "all" {
        return Ok(physical_store_list(root));
    }
    validate_store_name(name, true)?;
    Ok(vec![(
        name.to_string(),
        turso_mod::store_db_path(root, name),
    )])
}

/// Reject any SQL containing mutation keywords (whole-word, case-insensitive)
/// or a PRAGMA not on the read-only allowlist.
///
/// String literals, `--`/`/* */` comments, and quoted identifiers (`"x"`,
/// `[x]`, `` `x` ``) are stripped before keyword matching, so a mutation keyword
/// inside them is ignored rather than rejected. Anything surviving in SQL
/// position is matched fail-closed.
///
/// The allowlist is the reader guard's own ([`turso_mod::READER_PRAGMAS`]): the
/// connection this validator fronts is opened genuinely read-only (`O_RDONLY`),
/// so no PRAGMA could mutate the database file anyway — the list is
/// defense-in-depth on top of that, refusing a mutating PRAGMA with a clear
/// message instead of a silent no-op.
///
/// The reader entry's own rule comes first, taken from where it lives
/// ([`turso_mod::reader_refusal`]) rather than re-derived: a statement a reader
/// connection refuses is refused here too, so a query the inspection path
/// accepts and sends is never turned away by the reader that serves it. This
/// validator may refuse more (a blocked keyword anywhere, a form the tokenizer
/// cannot read, a later statement in a multi-statement string) — that direction
/// only keeps a query from leaving the client.
pub(crate) fn validate_read_only(sql: &str) -> Result<()> {
    if let Some(refusal) = turso_mod::reader_refusal(sql) {
        bail!("query rejected: {refusal}");
    }
    let tokens = scan_sql(sql).map_err(|e| anyhow!("query rejected: {e}"))?;
    for (idx, token) in tokens.iter().enumerate() {
        if !token.quoted {
            let upper = token.text.to_uppercase();
            if BLOCKLIST.contains(&upper.as_str()) {
                bail!("query rejected: contains blocked keyword '{}'", token.text);
            }
        }
        if token.text.eq_ignore_ascii_case("PRAGMA") {
            let Some(name) = tokens.get(idx + 1) else {
                bail!("query rejected: incomplete PRAGMA statement");
            };
            if !turso_mod::READER_PRAGMAS.contains(&name.text.to_lowercase().as_str()) {
                bail!(
                    "query rejected: PRAGMA '{}' is not on the read-only allowlist \
                     (mutating PRAGMAs are blocked; the connection is read-only)",
                    name.text
                );
            }
        }
    }
    Ok(())
}

/// A SQL token produced by [`scan_sql`]. `quoted` marks tokens that came from
/// a quoted identifier (`"x"`, `[x]`, `` `x` ``) — their content never matches
/// the mutation-keyword blocklist.
struct SqlToken {
    text: String,
    quoted: bool,
}

/// Split SQL into whole-word tokens, stripping string literals, comments, and
/// quoted identifiers before keyword matching.
///
/// Punctuation characters are discarded (they can never match a blocklist
/// keyword). A quoted identifier still yields one token (with `quoted: true`)
/// so PRAGMA names can be resolved through it.
fn scan_sql(sql: &str) -> Result<Vec<SqlToken>, String> {
    fn flush_word(tokens: &mut Vec<SqlToken>, word: &mut String) {
        if !word.is_empty() {
            tokens.push(SqlToken {
                text: std::mem::take(word),
                quoted: false,
            });
        }
    }

    let chars = sql.chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut word = String::new();

    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];

        // `--` line comment: skip to the next newline or end of input.
        if c == '-' && chars.get(i + 1) == Some(&'-') {
            flush_word(&mut tokens, &mut word);
            i += 2;
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        // `/* ... */` block comment.
        if c == '/' && chars.get(i + 1) == Some(&'*') {
            flush_word(&mut tokens, &mut word);
            i += 2;
            let mut closed = false;
            while i < chars.len() {
                if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                    i += 2;
                    closed = true;
                    break;
                }
                i += 1;
            }
            if !closed {
                return Err("unterminated /* comment".to_string());
            }
            continue;
        }

        // Single-quoted string literal — consumed entirely, no token.
        if c == '\'' {
            flush_word(&mut tokens, &mut word);
            i += 1;
            loop {
                if i >= chars.len() {
                    return Err("unterminated string literal".to_string());
                }
                if chars[i] == '\'' {
                    if chars.get(i + 1) == Some(&'\'') {
                        i += 2; // doubled quote == escaped quote
                        continue;
                    }
                    i += 1; // closing quote
                    break;
                }
                i += 1;
            }
            continue;
        }

        // Double-quoted / backtick quoted identifier — one quoted token.
        if c == '"' || c == '`' {
            flush_word(&mut tokens, &mut word);
            let quote = c;
            i += 1;
            let mut inner = String::new();
            loop {
                if i >= chars.len() {
                    return Err("unterminated quoted identifier".to_string());
                }
                if chars[i] == quote {
                    if chars.get(i + 1) == Some(&quote) {
                        inner.push(quote);
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                inner.push(chars[i]);
                i += 1;
            }
            tokens.push(SqlToken {
                text: inner,
                quoted: true,
            });
            continue;
        }

        // Bracket-quoted identifier — one quoted token, no escape doubling.
        if c == '[' {
            flush_word(&mut tokens, &mut word);
            i += 1;
            let mut inner = String::new();
            loop {
                if i >= chars.len() {
                    return Err("unterminated [bracket] identifier".to_string());
                }
                if chars[i] == ']' {
                    i += 1;
                    break;
                }
                inner.push(chars[i]);
                i += 1;
            }
            tokens.push(SqlToken {
                text: inner,
                quoted: true,
            });
            continue;
        }

        // Whitespace or SQL punctuation ends the current word.
        if c.is_whitespace() || SQL_PUNCTUATION.contains(&c) {
            flush_word(&mut tokens, &mut word);
            i += 1;
            continue;
        }

        // Anything else: append to the current word.
        word.push(c);
        i += 1;
    }
    flush_word(&mut tokens, &mut word);
    Ok(tokens)
}

/// Format a single `turso::core` result row as its pipe-table cells.
fn format_core_row(row: &turso::core::Row, column_count: usize) -> Vec<String> {
    (0..column_count)
        .map(|idx| format_core_value(row.get_value(idx)))
        .collect()
}

/// Convert a `turso::core` column value to its display representation through
/// [`crate::db::format_cell`], the one statement of the pipe-table rules. The
/// cell is copied once into the owned [`crate::db::Value`] shape that formatter
/// takes — the price of the shared rule on this path.
fn format_core_value(val: &turso::core::Value) -> String {
    crate::db::format_cell(&crate::db::Value::from(val.clone()))
}

/// Assemble the canonical `mahbot debug` pipe-table output: a column-header
/// line, one line per pre-formatted row, and a truncation sentinel when the
/// row limit was hit. Callers spell their own cells through
/// [`crate::db::format_cell`] (the value types differ per path) and delegate
/// the assembly here. An empty column list renders as empty output.
pub(crate) fn render_pipe_table(
    columns: &[String],
    rows: Vec<Vec<String>>,
    truncated: bool,
) -> String {
    if columns.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str(&columns.join("|"));
    out.push('\n');
    for row in rows {
        out.push_str(&row.join("|"));
        out.push('\n');
    }
    if truncated {
        out.push_str(&format_truncation_row(columns.len()));
        out.push('\n');
    }
    out
}

/// Format a truncation sentinel row that matches the column count.
///
/// For 1 column:  `truncated`
/// For 2 columns: `truncated|truncated`
/// For N≥3:       `...` + N−2 × `truncated` + `...`, pipe-joined
///                 (N=3 → `...|truncated|...`, N=4 → `...|truncated|truncated|...`)
pub(crate) fn format_truncation_row(column_count: usize) -> String {
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
    eprintln!("Usage: mahbot debug --db <name> [\"SQL query\"]");
    eprintln!("       mahbot debug families [--db <name>]");
    eprintln!("       mahbot debug --family <id> \"SQL query\"");
    let names = turso_mod::debug_db_names().join(" | ");
    eprintln!("  -h, --help  print this help and exit 0");
    eprintln!("  --db <name> {names} | all");
    eprintln!("              with a SQL argument: read-only query, pipe-delimited output");
    eprintln!("              without one: schema dump — one block per user table");
    eprintln!("              (`[table] <name>` / DDL / `rows: N`); `all` dumps every live");
    eprintln!("              database in per-store sections (per-store errors; exit 1 if");
    eprintln!("              any store failed)");
    eprintln!("  SQL query   read-only SQL, quoted as a single argument");
    eprintln!("  families    list quarantine/pre-reindex forensic families (--db filters by");
    eprintln!("               store name; a name matching nothing prints an empty list)");
    eprintln!("  --family <id>  read-only SQL against one forensic family (id from `families`)");
    eprintln!("  reads are bounded per command by {}", read_limit::ENV);
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  mahbot debug --db board");
    eprintln!("  mahbot debug --db all");
    eprintln!("  mahbot debug --db board \"SELECT phase, COUNT(*) FROM tickets GROUP BY phase\"");
    eprintln!("  mahbot debug --db all \"SELECT name FROM sqlite_master WHERE type='table'\"");
    eprintln!("  mahbot debug families");
    eprintln!(
        "  mahbot debug --family board.db.quarantine-20260812T120000Z-1234 \"SELECT COUNT(*) FROM tickets\""
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::UnwrapPoison;

    // Every test that lets a read start shares `serial(read_limit)`: a read takes
    // the process-global read-limit knob, and the read-limit tests below shrink it
    // to make a read fail at once.

    /// The entry names of a store directory — the no-new-files proof for a
    /// read-only open (sorted, so the comparison is order-independent).
    fn dir_names(dir: &Path) -> std::collections::BTreeSet<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn blocklist_rejects_mutation_keywords() {
        for sql in [
            "DROP TABLE tickets",
            "DELETE FROM logs",
            "INSERT INTO logs VALUES (1)",
            "UPDATE users SET name='x'",
            "VACUUM",
            "BEGIN",
            "ANALYZE",
            "analyze",
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
            "SELECT * FROM tool_calls WHERE tool_name = 'analyze'",
            "SELECT 'DROP TABLE tickets' AS note",
            "-- DROP TABLE tickets\nSELECT 1",
            "/* DELETE FROM logs */ SELECT 1",
            "SELECT \"UPDATE\" FROM t",
            "SELECT [drop] FROM t",
            "SELECT 'it''s analyze time' FROM t",
        ] {
            assert!(validate_read_only(sql).is_ok(), "should accept: {sql}");
        }
    }

    #[test]
    fn unterminated_literals_and_comments_are_rejected() {
        for sql in [
            "SELECT 'analyze",
            "SELECT 1 /* note",
            "SELECT \"drop",
            "SELECT [drop",
        ] {
            assert!(validate_read_only(sql).is_err(), "should reject: {sql}");
        }
    }

    #[test]
    fn quoted_pragma_names_still_resolve() {
        assert!(validate_read_only("PRAGMA [table_info](tickets)").is_ok());
        assert!(validate_read_only("PRAGMA [wal_checkpoint]").is_err());
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
        let tokens = scan_sql("SELECT a, b FROM t WHERE x='y'").expect("valid SQL");
        let unquoted: Vec<&str> = tokens
            .iter()
            .filter(|t| !t.quoted)
            .map(|t| t.text.as_str())
            .collect();
        for tok in ["SELECT", "a", "b", "FROM", "t", "WHERE", "x"] {
            assert!(unquoted.contains(&tok), "missing unquoted token: {tok}");
        }
        assert!(!unquoted.contains(&"y"), "string literal leaked as token");
        assert!(
            tokens.iter().all(|t| !t.text.is_empty()),
            "tokens must not be empty"
        );
    }

    /// A `mahbot debug` invocation over the given verb tail.
    fn debug_args(tail: &[&str]) -> Vec<String> {
        ["mahbot", "debug"]
            .into_iter()
            .chain(tail.iter().copied())
            .map(str::to_owned)
            .collect()
    }

    /// The direct route and the `--db all` walk read the store they name and
    /// create nothing beside it; the walk reports every store it could not read
    /// per store instead of stopping at the first.
    #[serial_test::serial(read_limit)]
    #[tokio::test]
    async fn run_debug_reads_one_store_and_the_whole_set_without_creating_anything() {
        let (_store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        let db_dir = dir.path().join("db");
        let before = dir_names(&db_dir);

        let result = run_debug_with_args(
            debug_args(&["--db", "logs", "SELECT COUNT(*) FROM logs"]),
            Some(dir.path().to_path_buf()),
        )
        .await;
        assert!(
            result.is_ok(),
            "direct route: a read-only query must succeed: {result:?}"
        );
        assert_eq!(
            dir_names(&db_dir),
            before,
            "direct route: the read-only open must leave the store directory unchanged"
        );

        // Computed from the physical store source (the single source of truth), so
        // the test does not hardcode the current store count — only the logs store
        // exists here, so the consolidated core file is missing.
        let err = run_debug_with_args(debug_args(&["--db", "all"]), Some(dir.path().to_path_buf()))
            .await
            .expect_err("--db all: a missing store must report a failure summary");
        let msg = format!("{err:#}");
        let total = physical_store_list(dir.path()).len();
        assert!(
            msg.contains(&format!("{} of {total} store(s) failed", total - 1)),
            "--db all: must summarize per-store failures, got: {msg}"
        );
    }

    /// The family route reads a pre-reindex family in place: the committed but
    /// never-checkpointed row lives only in the family's wal, so reading it
    /// proves the query covers both members — and touches neither.
    #[serial_test::serial(read_limit)]
    #[tokio::test]
    async fn run_debug_queries_a_pre_reindex_family_in_place() {
        let (store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        let db_dir = dir.path().join("db");
        store
            .flush_batch(
                "j1",
                "Engineer",
                "ws1",
                crate::stats::CallSource::Agent,
                &crate::db::now(),
                &[crate::ToolCallRecord {
                    tool_name: "read".to_string(),
                    arguments: "{}".to_string(),
                    duration_ms: 1,
                    success: true,
                    error_message: None,
                }],
            )
            .await
            .unwrap();
        let fam = "logs.db.pre-reindex-20260812T120000Z-4242";
        move_family_aside(&db_dir, &db_dir.join("logs.db"), fam);
        assert!(
            std::fs::metadata(db_dir.join(format!("{fam}-wal")))
                .unwrap()
                .len()
                > 0,
            "family route: the pre-reindex wal must hold committed frames"
        );
        let family_before: Vec<((u64, std::time::SystemTime), String)> = std::fs::read_dir(&db_dir)
            .unwrap()
            .map(|e| {
                let path = e.unwrap().path();
                (
                    file_state(&path),
                    path.file_name().unwrap().to_string_lossy().into_owned(),
                )
            })
            .collect();

        run_debug_with_args(
            debug_args(&["--family", fam, "SELECT COUNT(*) FROM tool_calls"]),
            Some(dir.path().to_path_buf()),
        )
        .await
        .expect("family route: a pre-reindex family must be queryable in place");

        let after_names: Vec<String> = std::fs::read_dir(&db_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            after_names.len(),
            family_before.len(),
            "family route: an in-place query must not create files beside the family"
        );
        for (state, name) in &family_before {
            assert_eq!(
                file_state(&db_dir.join(name)),
                *state,
                "family route: file must be unchanged: {name}"
            );
        }

        let budget = Budget::start();
        let read = Read::direct(fam, &budget);
        let out =
            execute_family_query(read, fam, "SELECT COUNT(*) FROM tool_calls", dir.path()).unwrap();
        assert_eq!(
            out, "COUNT(*)\n1\n",
            "family route: the in-place query must read the wal-only row"
        );
    }

    /// The held-location route: with the location held, the CLI must reach the
    /// instance's endpoint instead of opening the store — the CLI's own root
    /// holds no openable store, so a successful read can only have come from the
    /// running instance.
    #[serial_test::serial(read_limit)]
    #[tokio::test]
    async fn run_debug_reaches_a_held_location_over_the_channel() {
        let (instance_store, _instance_dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        instance_store
            .conn
            .execute_batch(
                "INSERT INTO logs (timestamp, level, target, message) \
                 VALUES ('2026-01-01T00:00:00Z', 'INFO', 'test', 'ipc-row')",
            )
            .await
            .expect("insert a committed row");
        let root = tempfile::TempDir::new().unwrap();
        let ipc_db_dir = root.path().join("db");
        std::fs::create_dir_all(&ipc_db_dir).unwrap();
        std::fs::write(ipc_db_dir.join("logs.db"), b"not a store").unwrap();
        let _held = hold_location(root.path());
        let listener_root = root.path().to_path_buf();
        let listener = tokio::spawn(async move {
            crate::db::ipc::run_ipc_listener(&listener_root, std::sync::Arc::new(instance_store))
                .await;
        });
        // No wait for the endpoint to be bound: the CLI's own channel-bound retry
        // covers a late bind, which is what makes this portable.
        run_debug_with_args(
            debug_args(&["--db", "logs", "SELECT COUNT(*) FROM logs"]),
            Some(root.path().to_path_buf()),
        )
        .await
        .expect("held location: must be queried through the instance's channel");
        listener.abort();
    }

    /// The schema dump (`--db <name>` with no SQL) reflects the LIVE stored
    /// schema: one block per user table with its DDL and row count, excluding
    /// internal catalog artifacts (`sqlite_%`, `__turso_internal_%`).
    #[serial_test::serial(read_limit)]
    #[tokio::test]
    async fn schema_dump_prints_user_tables_with_row_counts() {
        let (store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        // One non-trivial row so the dump's row count reflects live contents.
        store
            .conn
            .execute(
                "INSERT INTO logs (timestamp, level, target, message) \
                 VALUES ('2026-01-01T00:00:00Z', 'INFO', 'test', 'hello')",
                turso_mod::params![],
            )
            .await
            .expect("insert a log row");

        let db_path = dir.path().join("db").join("logs.db");
        let budget = Budget::start();
        let read = Read::direct("logs", &budget);
        let dump = open_and_run_readonly(read, &db_path, |read, io, db, path| {
            dump_schema(read, io, db, path, "logs")
        })
        .expect("dump must succeed on a real store");

        assert!(
            dump.starts_with("== schema dump: logs ==\n"),
            "dump must open with the store header: {dump}"
        );
        for table in ["logs", "tool_calls", "llm_requests"] {
            assert!(
                dump.contains(&format!("\n[table] {table}\n")),
                "user table block missing for '{table}': {dump}"
            );
        }
        assert!(
            dump.contains("CREATE TABLE logs ("),
            "table DDL must be included: {dump}"
        );
        assert!(
            dump.contains("\nrows: 1\n"),
            "row count must reflect the live row: {dump}"
        );
        assert!(
            !dump.contains("__turso_internal_"),
            "internal turso artifacts must be excluded: {dump}"
        );
        assert!(
            !dump.contains("sqlite_sequence"),
            "sqlite_sequence must be excluded: {dump}"
        );

        // Without a SQL argument the same dump is what `mahbot debug --db
        // <name>` prints: it exits 0 and leaves the store directory untouched,
        // like the query path.
        let db_dir = dir.path().join("db");
        let before = dir_names(&db_dir);
        let result = run_debug_with_args(
            debug_args(&["--db", "logs"]),
            Some(dir.path().to_path_buf()),
        )
        .await;
        assert!(
            result.is_ok(),
            "schema dump without SQL must succeed: {result:?}"
        );
        assert_eq!(
            dir_names(&db_dir),
            before,
            "the schema dump must leave the store directory unchanged"
        );
    }

    // ── read-limit tests ─────────────────────────────────────────────

    /// A spent budget fails every route before it opens anything: the error names
    /// the store or the family id it was given and the limit, the store directory
    /// stays byte-identical (nothing was opened), and every store the `--db all`
    /// walk reaches is counted as failed through its summary rather than silently
    /// skipped. The stack's end-of-read verdict — a read that comes back past the
    /// limit is the limit's failure — is pinned deterministically by
    /// `run_debug_read_limit_fails_a_read_that_finished_after_it`.
    #[serial_test::serial(read_limit)]
    #[tokio::test]
    async fn run_debug_read_limit_fails_every_route() {
        let _env = crate::util::test::set_env_var(read_limit::ENV, Some("0"));

        // Direct: the error names the store, the limit and the knob.
        let (_store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        let db_dir = dir.path().join("db");
        let before = dir_names(&db_dir);
        let err = run_debug_with_args(
            debug_args(&["--db", "logs", "SELECT COUNT(*) FROM logs"]),
            Some(dir.path().to_path_buf()),
        )
        .await
        .expect_err("a spent read budget must fail the read");
        let msg = format!("{err:#}");
        assert!(msg.contains("'logs'"), "error must name the store: {msg}");
        assert!(
            msg.contains("0s read limit"),
            "error must name the limit: {msg}"
        );
        assert!(
            msg.contains(read_limit::ENV),
            "error must name the knob: {msg}"
        );
        assert_eq!(
            dir_names(&db_dir),
            before,
            "a spent budget must fail before the store is opened"
        );

        // Family: refused before the snapshot file is opened, naming the id.
        let fam_dir = tempfile::TempDir::new().unwrap();
        let fam_db_dir = fam_dir.path().join("db");
        std::fs::create_dir_all(&fam_db_dir).unwrap();
        let fam = "logs.db.quarantine-20260812T120000Z-4242";
        std::fs::write(fam_db_dir.join(fam), valid_db_bytes()).unwrap();
        let err = run_debug_with_args(
            debug_args(&["--family", fam, "SELECT 1"]),
            Some(fam_dir.path().to_path_buf()),
        )
        .await
        .expect_err("a spent read budget must fail the family read");
        let msg = format!("{err:#}");
        assert!(msg.contains(fam), "error must name the family id: {msg}");
        assert!(
            msg.contains("0s read limit"),
            "error must name the limit: {msg}"
        );

        // `--db all`: never an empty result, never a silent skip. The per-store
        // "was not read" lines are stderr text, which a test cannot capture
        // in-process; what is pinned is that every store the walk reached is
        // counted as failed and the exit is non-zero.
        let err = run_debug_with_args(debug_args(&["--db", "all"]), Some(dir.path().to_path_buf()))
            .await
            .expect_err("a spent budget must fail --db all");
        let msg = format!("{err:#}");
        let total = physical_store_list(dir.path()).len();
        assert!(
            msg.contains(&format!("{total} of {total} store(s) failed")),
            "must summarize every store as failed, got: {msg}"
        );
    }

    /// The limit's verdict is final: a direct read that ran past the limit is the
    /// limit's failure even though it came back with a value. The runner here
    /// stands in for a real query — the same open+run stack, only slow enough to
    /// outlive the budget. In a test binary the watchdog is off, so what ends the
    /// read is the limit's own check at the end of it.
    ///
    /// The dump route shares that stack (`open_and_dump_readonly`), and a dump is
    /// walked table by table with the budget asked between tables, so its
    /// end-of-read verdict is the same check this test pins. `--db all` is where a
    /// spent budget across stores is pinned, by its summary.
    #[serial_test::serial(read_limit)]
    #[tokio::test]
    async fn run_debug_read_limit_fails_a_read_that_finished_after_it() {
        let _env = crate::util::test::set_env_var(read_limit::ENV, Some("1"));
        let (store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        // The direct read-only open is a second opener of this file: close the
        // fixture's writing connection first.
        drop(store);
        let db_path = dir.path().join("db").join("logs.db");

        let budget = Budget::start();
        let read = Read::direct("logs", &budget);
        let err = open_and_run_readonly(read, &db_path, |_read, _io, _db, _path| {
            std::thread::sleep(std::time::Duration::from_millis(1100));
            Ok("COUNT(*)\n1\n".to_string())
        })
        .expect_err("a read that finished after the limit must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("'logs'"), "error must name the store: {msg}");
        assert!(
            msg.contains("1s read limit"),
            "error must name the limit: {msg}"
        );
    }

    // ── forensic-family tests ──────────────────────────────────────────

    /// The listing's family-name parser must round-trip the writer's exact
    /// naming formats and reject sidecars, foreign files, and malformed names
    /// — a future writer change breaks this test instead of the listing.
    #[test]
    fn family_name_parser_round_trips() {
        let q = parse_family_name("board.db.quarantine-20260812T120000Z-1234").unwrap();
        assert_eq!(q.store, "board");
        assert_eq!(q.kind, FamilyKind::Quarantine);
        assert_eq!(q.stamp, "20260812T120000Z");

        // Seq-suffixed quarantine (process-local counter, multi-store boots).
        assert!(parse_family_name("board.db.quarantine-20260812T120000Z-1234-2").is_some());

        // Legacy/non-canonical stores are listed too.
        assert_eq!(
            parse_family_name("stats.db.quarantine-20260812T120000Z-7")
                .unwrap()
                .store,
            "stats"
        );

        let p = parse_family_name("logs.db.pre-reindex-20260812T120000Z-99").unwrap();
        assert_eq!(p.kind, FamilyKind::PreReindex);

        // A differently-cased spelling is the same family — the volumes compare
        // file names without case — while the fields keep the given spelling.
        let c = parse_family_name("CORE.DB.Quarantine-20260812t120000z-1234").unwrap();
        assert_eq!(c.store, "CORE");
        assert_eq!(c.kind, FamilyKind::Quarantine);
        assert_eq!(c.stamp, "20260812t120000z");

        // Format-focused rejections; the -wal sidecar / path-traversal /
        // live-store-name cases are covered end-to-end by
        // `run_debug_family_rejects_invalid_id` at the CLI layer.
        for bad in [
            "board.db.quarantine-20260812T120000Z",       // no pid
            "board.db.quarantine-20260812T120000Z-abc",   // non-digit pid
            "board.db.quarantine-20260812T120000Z-1234-", // trailing-dash seq
            "board.db.pre-reindex-20260812T120000Z-99-1", // seq on pre-reindex
            "board.db.quarantine-12T34-1",                // bad stamp
        ] {
            assert!(parse_family_name(bad).is_none(), "must reject: {bad}");
        }
    }

    /// Main-DB bytes with a valid SQLite header (magic + 4096-byte page size).
    fn valid_db_bytes() -> Vec<u8> {
        let mut b = vec![0u8; 128];
        b[..16].copy_from_slice(b"SQLite format 3\0");
        b[16] = 0x10; // page size 4096 (u16 BE)
        b[17] = 0x00;
        b
    }

    /// The listing discovers families from base names and sidecar-only
    /// members, classifies their file sets, ignores foreign files, and sorts
    /// by (store, kind, stamp).
    #[test]
    fn list_families_classifies_file_sets() {
        let dir = tempfile::TempDir::new().unwrap();
        let db_dir = dir.path().join("db");
        std::fs::create_dir_all(&db_dir).unwrap();

        // Complete quarantine: db + wal.
        let c = "board.db.quarantine-20260812T120000Z-100";
        std::fs::write(db_dir.join(c), valid_db_bytes()).unwrap();
        std::fs::write(db_dir.join(format!("{c}-wal")), b"x").unwrap();
        // Complete pre-reindex: db + wal (the `VACUUM INTO` snapshot shape).
        let p = "logs.db.pre-reindex-20260812T120000Z-200";
        std::fs::write(db_dir.join(p), valid_db_bytes()).unwrap();
        std::fs::write(db_dir.join(format!("{p}-wal")), b"x").unwrap();
        // Sidecar-only quarantine: no base file.
        let s = "sessions.db.quarantine-20260812T120000Z-300";
        std::fs::write(db_dir.join(format!("{s}-wal")), b"x").unwrap();
        // Partial quarantine: db without its wal.
        let pa = "users.db.quarantine-20260812T120000Z-400";
        std::fs::write(db_dir.join(pa), valid_db_bytes()).unwrap();
        // Bad-header quarantine: db present, corrupt header.
        let bh = "config.db.quarantine-20260812T120000Z-500";
        std::fs::write(db_dir.join(bh), vec![b'x'; 128]).unwrap();
        // Foreign files / live stores are ignored.
        std::fs::write(db_dir.join(".DS_Store"), b"").unwrap();
        std::fs::write(db_dir.join("board.db"), valid_db_bytes()).unwrap();

        let families = list_families(dir.path()).unwrap();
        let by_id: std::collections::BTreeMap<&str, &FamilyInfo> =
            families.iter().map(|f| (f.id.as_str(), f)).collect();
        assert_eq!(by_id.len(), 5, "families: {families:?}");
        assert_eq!(by_id[c].class, FamilyClass::Complete);
        assert_eq!(by_id[c].files, "db,wal");
        assert_eq!(by_id[p].class, FamilyClass::Complete);
        assert_eq!(by_id[p].files, "db,wal");
        assert_eq!(by_id[s].class, FamilyClass::SidecarOnly);
        assert_eq!(by_id[s].files, "wal");
        assert_eq!(by_id[pa].class, FamilyClass::Partial);
        assert_eq!(by_id[pa].files, "db");
        assert_eq!(by_id[bh].class, FamilyClass::BadHeader);
        // Sorted by (store, kind, stamp): board, config, logs, sessions, users.
        let stores: Vec<&str> = families.iter().map(|f| f.store.as_str()).collect();
        assert_eq!(stores, ["board", "config", "logs", "sessions", "users"]);
    }

    /// Rename a live store family into a forensic family name (the boot path
    /// renames, it does not copy; a missing member is skipped). Returns the
    /// moved member file names.
    fn move_family_aside(db_dir: &Path, base: &Path, fam: &str) -> Vec<String> {
        let mut moved = Vec::new();
        for (src, suffix) in [(base, ""), (&turso_mod::wal_path(base), "-wal")] {
            if src.exists() {
                std::fs::rename(src, db_dir.join(format!("{fam}{suffix}"))).unwrap();
                moved.push(format!("{fam}{suffix}"));
            }
        }
        moved
    }

    /// (size, mtime) of one file — the no-mutation proof snapshot.
    fn file_state(path: &Path) -> (u64, std::time::SystemTime) {
        let md = std::fs::metadata(path).unwrap();
        (md.len(), md.modified().unwrap())
    }

    /// Query-error paths report distinct, clear messages: a sidecar-only
    /// quarantine (no main DB) and a garbage main DB both fail without
    /// crashing, naming the family.
    #[serial_test::serial(read_limit)]
    #[tokio::test]
    async fn run_debug_family_error_paths_report_clear_errors() {
        let dir = tempfile::TempDir::new().unwrap();
        let db_dir = dir.path().join("db");
        std::fs::create_dir_all(&db_dir).unwrap();
        let fam = "logs.db.quarantine-20260812T120000Z-4242";
        let run = |sql: &str| {
            run_debug_with_args(
                debug_args(&["--family", fam, sql]),
                Some(dir.path().to_path_buf()),
            )
        };

        // Sidecar-only quarantine: listed but not queryable — a clear
        // sidecar-only message, not a missing-family one.
        std::fs::write(db_dir.join(format!("{fam}-wal")), b"x").unwrap();
        let err = run("SELECT 1").await.expect_err("sidecar-only must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("no main database file"), "got: {msg}");

        // Garbage main DB: fails turso's clean NotADB magic check (the engine
        // panic path is exercised separately by `guard_panics_converts_panic`).
        std::fs::remove_file(db_dir.join(format!("{fam}-wal"))).unwrap();
        std::fs::write(db_dir.join(fam), vec![0x42; 4096]).unwrap();
        let err = run("SELECT 1").await.expect_err("garbage db must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains(fam), "error must name the family: {msg}");
    }

    /// `guard_panics` converts a panic into a clean error — the containment
    /// path a damaged family can trigger inside turso_core.
    #[test]
    fn guard_panics_converts_panic_to_error() {
        let err = guard_panics(|| -> Result<()> { panic!("boom: pager index OOB") })
            .expect_err("a panic must surface as an error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("panicked while reading the store"),
            "got: {msg}"
        );
        assert!(msg.contains("boom: pager index OOB"), "got: {msg}");
        assert!(is_engine_panic_error(&err), "panic must be engine-panic");
    }

    /// `--family` rejects ids that are not family names (path-safety gate).
    #[tokio::test]
    async fn run_debug_family_rejects_invalid_id() {
        let dir = tempfile::TempDir::new().unwrap();
        for bad in [
            "../../etc/passwd",
            "board.db",
            "board.db.quarantine-20260812T120000Z-1-wal",
        ] {
            let err = run_debug_with_args(
                debug_args(&["--family", bad, "SELECT 1"]),
                Some(dir.path().to_path_buf()),
            )
            .await
            .expect_err("invalid family id must be rejected");
            let msg = format!("{err:#}");
            assert!(msg.contains("invalid family id"), "got: {msg}");
        }
    }

    /// Help at any verb position (`-h`/`--help` standalone, after a verb, after
    /// a store/family id, misordered, or with trailing args) prints usage and
    /// exits 0 instead of falling through to running the flag as a SQL query.
    #[tokio::test]
    async fn run_debug_help_after_verb_prints_usage() {
        let dir = tempfile::TempDir::new().unwrap();
        for tail in [
            vec!["--help"],
            vec!["-h"],
            vec!["families", "--help"],
            vec!["families", "-h"],
            vec!["--family", "--help"],
            vec!["--family", "-h"],
            vec!["families", "--db", "--help"],
            vec!["--db", "--help"],
            vec!["--db", "-h"],
            vec!["--family", "--help", "extra"],
            vec!["--family", "-h", "extra"],
            vec![
                "--family",
                "logs.db.quarantine-20260812T120000Z-4242",
                "--help",
            ],
            vec!["--db", "board", "--help"],
            vec!["--db", "board", "-h"],
            vec!["--db", "board", "--help", "extra"],
            vec!["--db", "board", "-h", "extra"],
            vec!["families", "--db", "board", "--help"],
        ] {
            run_debug_with_args(debug_args(&tail), Some(dir.path().to_path_buf()))
                .await
                .unwrap_or_else(|e| panic!("help must print usage and exit 0: {e:#}"));
        }
    }

    /// `families --db <name>` filters silently (a name matching nothing prints
    /// nothing, exit 0), and a well-formed but non-existent family id reports
    /// "not found" rather than the sidecar-only message.
    #[tokio::test]
    async fn run_debug_families_filters_and_reports_missing_family() {
        let dir = tempfile::TempDir::new().unwrap();

        // A name matching no family (canonical store without families,
        // unknown/legacy name) prints nothing and exits 0 — the filter never
        // depends on the current listing content.
        run_debug_with_args(
            debug_args(&["families", "--db", "nonexistent"]),
            Some(dir.path().to_path_buf()),
        )
        .await
        .expect("families --db <matching-nothing> must print nothing and exit 0");

        let err = run_debug_with_args(
            debug_args(&[
                "--family",
                "logs.db.quarantine-20260812T120000Z-4242",
                "SELECT 1",
            ]),
            Some(dir.path().to_path_buf()),
        )
        .await
        .expect_err("a well-formed but missing family must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("not found"), "got: {msg}");
        assert!(!msg.contains("sidecar-only"), "got: {msg}");

        // A legacy (non-canonical) store family appears in the listing and is
        // --db-filterable, even though store_names() does not know it.
        let db_dir = dir.path().join("db");
        std::fs::create_dir_all(&db_dir).unwrap();
        let legacy = "stats.db.quarantine-20260812T120000Z-7";
        std::fs::write(db_dir.join(legacy), valid_db_bytes()).unwrap();
        run_debug_with_args(
            debug_args(&["families", "--db", "stats"]),
            Some(dir.path().to_path_buf()),
        )
        .await
        .expect("legacy store family must be filterable by --db");
    }

    /// End-to-end no-instance path: a row committed to the WAL (the store is
    /// dropped without a checkpoint) is visible through the direct
    /// single-process read-only open — no IPC, no running instance.
    #[serial_test::serial(read_limit)]
    #[tokio::test]
    async fn run_debug_without_an_instance_reads_committed_wal() {
        let (store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        store
            .conn
            .execute_batch(
                "INSERT INTO logs (timestamp, level, target, message) \
                 VALUES ('2026-01-01T00:00:00Z', 'INFO', 'test', 'wal-row')",
            )
            .await
            .expect("insert a committed row");
        // Drop the store WITHOUT checkpointing, so the row lives only in the WAL.
        drop(store);

        assert!(
            matches!(
                crate::util::lock::instance_lock_state_settled(dir.path()),
                InstanceLockState::Free
            ),
            "temp root has no mahbot.lock — no instance must be considered to hold it"
        );
        run_debug_with_args(
            debug_args(&["--db", "logs", "SELECT COUNT(*) FROM logs"]),
            Some(dir.path().to_path_buf()),
        )
        .await
        .expect("direct read of committed WAL rows must succeed");

        // Prove the WAL-only row is actually visible via the direct open.
        let db_path = dir.path().join("db").join("logs.db");
        let budget = Budget::start();
        let read = Read::direct("logs", &budget);
        let out = open_and_run_readonly(read, &db_path, |read, io, db, path| {
            connect_execute(read, io, db, "SELECT COUNT(*) FROM logs", path)
        })
        .expect("direct read-only open must succeed");
        assert_eq!(
            out, "COUNT(*)\n1\n",
            "WAL-only committed row must be visible"
        );
    }

    /// Take the location for the duration of a test: an exclusive lock on the
    /// standard lock file path, exactly what the CLI's settled probe observes as
    /// [`InstanceLockState::Held`].
    fn hold_location(root: &Path) -> std::fs::File {
        let lock = std::fs::File::create(crate::util::lock::lock_file_path(root))
            .expect("create the lock file");
        assert!(
            crate::util::lock::try_flock(&lock).expect("lock the location"),
            "the test must hold the location"
        );
        lock
    }

    /// A held location the CLI cannot reach is refused, never opened: the store
    /// here is a real one it could read directly, so an `Ok` would mean it read
    /// the live store behind the instance's back. The probe's third state is a
    /// refusal too — a lock file that exists but cannot be opened must never be
    /// read as "no instance is running".
    #[serial_test::serial(read_limit)]
    #[tokio::test]
    async fn run_debug_refuses_a_held_location_it_cannot_reach() {
        // One second, not zero: a spent budget is reported as the limit's own
        // failure without ever reaching the channel, which is a different
        // sentence — this is about the channel that cannot be reached, so the
        // retries must still happen, within a second instead of the default.
        {
            let _env = crate::util::test::set_env_var(read_limit::ENV, Some("1"));
            let (_store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
            let _held = hold_location(dir.path());

            let err = run_debug_with_args(
                debug_args(&["--db", "logs", "SELECT COUNT(*) FROM logs"]),
                Some(dir.path().to_path_buf()),
            )
            .await
            .expect_err("a held location without a reachable channel must refuse");
            let msg = format!("{err:#}");
            assert!(msg.contains("instance is running"), "got: {msg}");
            assert!(msg.contains("left untouched"), "got: {msg}");
        }

        let (_store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        // A directory where the lock file belongs: the probe cannot open it.
        std::fs::create_dir(crate::util::lock::lock_file_path(dir.path()))
            .expect("occupy the lock file path");

        let err = run_debug_with_args(
            debug_args(&["--db", "logs", "SELECT COUNT(*) FROM logs"]),
            Some(dir.path().to_path_buf()),
        )
        .await
        .expect_err("an undecidable lock probe must refuse");
        let msg = format!("{err:#}");
        assert!(msg.contains("cannot tell whether"), "got: {msg}");
        assert!(msg.contains("left untouched"), "got: {msg}");
    }

    /// Marker for the spawned child of
    /// [`run_debug_read_limit_watchdog_ends_a_wedged_read`]: it runs the child
    /// test instead of the suite's tests. Set only by the driver below.
    const WATCHDOG_CHILD_ENV: &str = "MAHBOT_DEBUG_READ_WATCHDOG_CHILD";

    /// The read limit the child runs under: long enough that the timing assertions
    /// sit nowhere near the handshake's jitter, short enough to keep the test quick.
    const WATCHDOG_CHILD_LIMIT_SECS: u64 = 5;

    /// The grace the watchdog promises, as this test expects it. Spelled out
    /// rather than read from [`read_limit::WATCHDOG_GRACE_SECS`], because a bound
    /// derived from that constant moves with it — a watchdog tuned to cut at
    /// exactly the limit would still pass. Checked against it below.
    const EXPECTED_WATCHDOG_GRACE_SECS: u64 = 1;

    /// The child's fallback — the driver's only hang guard: a watchdog that never
    /// fires leaves a child that wakes up, panics and exits 101, which fails the
    /// driver's status assertion instead of hanging the suite. Derived from the
    /// wait it stands in for, so it can never be shorter than it.
    const WATCHDOG_CHILD_FALLBACK: std::time::Duration = std::time::Duration::from_secs(
        WATCHDOG_CHILD_LIMIT_SECS + EXPECTED_WATCHDOG_GRACE_SECS + 24,
    );

    /// The watchdog is the half of the read limit that does not depend on the
    /// read yielding: a read that never returns is ended by the process exiting
    /// with the limit's own failure. Proven across two processes — the only way
    /// to observe "the process is gone".
    ///
    /// The assertions are behaviour, not machine speed: the watchdog must still be
    /// asleep until the limit plus its own grace has elapsed, and it must end the
    /// child after that with its own status and sentence — which is what tells the
    /// watchdog firing apart from a child that gave up before arming. The child
    /// starts its budget only once released below, so this process's scheduling
    /// cannot fake an early cut; the residue is the other way: the child's startup
    /// before it reaches its read, plus this process's delay before the write, still
    /// hide a cut of that size.
    ///
    /// Load still reaches the test through the spawn itself (an `EAGAIN`/`ENOMEM`
    /// under memory pressure) and through a kill of the child; both fail here as
    /// what they are rather than as a timing verdict.
    ///
    /// The filter below names [`debug_read_limit_watchdog_child`]; if that test is
    /// renamed the child runs nothing, exits 0, and this driver fails on the
    /// status.
    #[test]
    fn run_debug_read_limit_watchdog_ends_a_wedged_read() {
        // The child inherits this process's environment, so hold the env lock for
        // the whole run: `util::test::set_env_var` callers take that same lock,
        // and a change set at the wrong instant would reach the child.
        let _env = crate::util::test::env_lock().lock().unwrap_poison();
        assert_eq!(
            read_limit::WATCHDOG_GRACE_SECS,
            EXPECTED_WATCHDOG_GRACE_SECS,
            "the grace is part of what this test asserts — re-derive the lower bound \
             before changing it"
        );
        let exe = std::env::current_exe().expect("test binary path");
        let mut child = std::process::Command::new(exe)
            .args([
                "--ignored",
                "--nocapture",
                "debug_read_limit_watchdog_child",
            ])
            .env(WATCHDOG_CHILD_ENV, "1")
            .env(read_limit::ENV, WATCHDOG_CHILD_LIMIT_SECS.to_string())
            // `spawn` inherits stdio (only `output` pipes it): the sentence is the
            // evidence, and stdin releases the child.
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn the watchdog child");
        let started = std::time::Instant::now();
        let mut release = child.stdin.take().expect("the child's stdin is piped");
        std::io::Write::write_all(&mut release, b"g").expect("release the watchdog child");
        drop(release);
        let output = child
            .wait_with_output()
            .expect("collect the watchdog child's output");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        assert_eq!(
            output.status.code(),
            Some(read_limit::WATCHDOG_EXIT_CODE),
            "the watchdog must end the child:\n{text}"
        );
        assert!(text.contains("'logs' was not read"), "got: {text}");
        assert!(
            text.contains(&format!("{WATCHDOG_CHILD_LIMIT_SECS}s read limit")),
            "got: {text}"
        );
        assert!(text.contains(read_limit::ENV), "got: {text}");
        let earliest_cut = std::time::Duration::from_secs(
            WATCHDOG_CHILD_LIMIT_SECS + EXPECTED_WATCHDOG_GRACE_SECS,
        );
        assert!(
            started.elapsed() >= earliest_cut,
            "the watchdog must not cut the read before the limit plus its grace \
             ({earliest_cut:?}) has elapsed: {:?}",
            started.elapsed()
        );
    }

    /// Child mode for [`run_debug_read_limit_watchdog_ends_a_wedged_read`]: hold
    /// a read that never yields and let the watchdog end the process.
    #[test]
    #[ignore = "driven by run_debug_read_limit_watchdog_ends_a_wedged_read"]
    fn debug_read_limit_watchdog_child() {
        if std::env::var_os(WATCHDOG_CHILD_ENV).is_none() {
            return;
        }
        // Released by the driver before its clock starts (see there). EOF — run by
        // hand — only means no clock is waiting on this process.
        let mut release = [0u8];
        let _ = std::io::Read::read(&mut std::io::stdin(), &mut release);
        let budget = Budget::start();
        let read = Read::direct("logs", &budget);
        // A test binary's watchdog does not end the process (see
        // [`read_limit::enable_process_exit`]); this child IS the process that
        // proves it does.
        read_limit::enable_process_exit();
        let _watchdog = read.arm().expect("the child's budget must be live");
        std::thread::sleep(WATCHDOG_CHILD_FALLBACK);
        panic!("the read limit's watchdog did not end the process");
    }
}
