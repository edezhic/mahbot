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
//! ## Daemon-up IPC routing
//!
//! Since `multiprocess_wal` was removed, the daemon is the single-process
//! writer and holds the instance flock while it runs. `mahbot debug` must
//! never open a second physical instance of a live store, so when the daemon
//! holds the lock ([`crate::util::lock::daemon_holds_lock_settled`]) the CLI
//! routes queries through the daemon's local IPC endpoint (`crate::db::ipc`)
//! instead — [`query_over_ipc`] for a query and [`dump_over_ipc`] for a schema
//! dump. The endpoint applies the same `PRAGMA query_only=ON` guard as the
//! daemon's own read-only path.
//!
//! ## Daemon-down direct open
//!
//! When the daemon is down (no lock held) the CLI opens each store directly in
//! single-process read-only mode via `turso::core::Database::open_file_with_flags`
//! with `OpenFlags::ReadOnly | OpenFlags::NoLock`. This cannot create or
//! mutate files (the SDK `Builder` has no read-only option and always passes
//! `OpenFlags::Create`, which would create a missing `-shm`/`-wal` pair), and
//! it reads committed WAL frames in single-process mode.
//!
//! Two defense-in-depth layers remain on top of the read-only open: an
//! upfront file-existence check and a SQL validator (mutation-keyword
//! blocklist plus a PRAGMA allowlist — see [`validate_read_only`]).
//!
//! ## Other verbs
//!
//! - `mahbot debug detect [--db <name>]` — classify store file sets without
//!   opening any database via [`wal_guard::inspect_store_at`], reporting
//!   `healthy`/`durable-b`/`structural` plus the on-disk `-wal` size and a
//!   stale `.tshm` flag. Exits 1 when any store is structurally corrupt.
//! - `mahbot debug families [--db <name>]` — list every quarantine and
//!   pre-reindex family in the store directory with its original store,
//!   artifact type, timestamp, total size, and a file-set/header
//!   classification. No database is opened.
//! - `mahbot debug --family <id> "SQL query"` — query one forensic family
//!   (a static snapshot) with the same read-only guarantees as live stores.
//!   Families have no live-store layers; the engine opens single-process
//!   read-only, and a family carrying a stale `.tshm` is copied to an OS temp
//!   dir first (the copy omits the `.tshm`). Engine panics on corrupt
//!   families are caught and reported as clean errors.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};

use crate::db as turso_mod;
use crate::db::ipc;
use crate::db::wal_guard;

/// Row limit per query to prevent unbounded output. Same LIMIT+1 semantic as
/// the daemon-side IPC bound (`ipc::IPC_ROW_LIMIT`) so the two never drift.
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

/// Shared `[--db <name>]` flag parser for the detect/families verbs
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

/// Validate a store name against the canonical list. `all_valid` appends the
/// literal `all` option to the hint — only callers that accept `--db all`
/// pass `true`.
///
/// Accepts both the LOGICAL store names ([`turso_mod::store_names`]) and the
/// PHYSICAL consolidated file name ([`turso_mod::CONSOLIDATED_DB_NAME`], shown
/// as the label in `--db all`/`detect` output). Both lists must stay in sync
/// with the debug CLI's `--db` surface.
pub(crate) fn validate_store_name(name: &str, all_valid: bool) -> Result<()> {
    let names = turso_mod::debug_db_names();
    if names.contains(&name) {
        return Ok(());
    }
    let hint = names.join(", ") + if all_valid { ", all" } else { "" };
    bail!("invalid database name '{name}'. Valid names: {hint}");
}

/// `mahbot debug detect [--db <name>]` — classify every store's file set
/// without opening any database. Prints one line per physical store
/// (`name\tclass\twal_size=N\tstale_tshm=B`). Exit 0 when all stores are
/// healthy; exit 1 (via `Err`) when a store is structurally corrupt (a bad
/// main-DB header — the only class the single-process classifier fails on).
fn run_debug_detect(args: &[String], home_override: Option<PathBuf>) -> Result<()> {
    let mahbot_home = resolve_home(home_override)?;
    let selected = match parse_db_flag(args, "detect")? {
        Some(name) if name == "all" => physical_store_list(&mahbot_home)
            .into_iter()
            .map(|(n, _)| n)
            .collect(),
        Some(name) => {
            validate_store_name(&name, false)?;
            vec![name]
        }
        // No `--db` → diagnose each PHYSICAL file once (core + logs), not
        // once per logical domain name (every logical name maps to one of the
        // physical files).
        None => physical_store_list(&mahbot_home)
            .into_iter()
            .map(|(n, _)| n)
            .collect(),
    };
    let mut failures = 0usize;
    for name in &selected {
        let status = wal_guard::inspect_store_at(&turso_mod::store_db_path(&mahbot_home, name));
        print_line(format_args!(
            "{}\t{}\twal_size={}\tstale_tshm={}",
            name,
            status.class.label(),
            status.wal_size,
            status.has_stale_tshm,
        ))?;
        if status.class == wal_guard::StoreClass::Structural {
            failures += 1;
        }
    }
    if failures > 0 {
        bail!("{failures} store(s) structurally corrupt — see above");
    }
    Ok(())
}

// ── Forensic families (quarantine / pre-reindex) ─────────────────────────

/// Artifact type of a forensic family. Variant order is the listing sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FamilyKind {
    /// Renamed-aside artifact family (the boot corruption record; never deleted).
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
    /// File-set/header classification (family-specific — never the live-store
    /// wal_guard coordination labels, which are meaningless for static files).
    class: FamilyClass,
    /// Present members, comma-joined (`db,wal,shm,tshm`).
    files: String,
}

/// State classification of one forensic family (snapshot semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FamilyClass {
    /// Main DB present with a valid header and every expected sidecar present.
    Complete,
    /// Main DB present with a valid header but some expected sidecars missing.
    Partial,
    /// Main DB present but its header is corrupt/empty — queryable, likely fails.
    BadHeader,
    /// No main DB file (coordination-sidecar-only quarantine) — not queryable.
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
/// Rejects sidecar members (`-wal`/`-shm`/`-tshm` suffixes), foreign files,
/// and any path-like name — the path-safety gate for `--family <id>`: only a
/// parsed family id reaches `root/db/<id>`. The stamp is shape-checked
/// (16 chars, digits around `T`/`Z`), not calendar-validated; a store name of
/// `.` or `..` parses (flat filename — no traversal possible).
///
/// The formats are written by `turso.rs::quarantine_family` (quarantine) and
/// the pre-reindex snapshot in `turso.rs` (REINDEX repair) — keep both sides
/// of the naming contract in sync; the writer round-trip test locks the
/// coupling.
pub(crate) fn parse_family_name(name: &str) -> Option<FamilyMeta> {
    let (kind, marker) = if name.contains(".quarantine-") {
        (FamilyKind::Quarantine, ".quarantine-")
    } else if name.contains(".pre-reindex-") {
        (FamilyKind::PreReindex, ".pre-reindex-")
    } else {
        return None;
    };
    let (prefix, tail) = name.split_once(marker)?;
    let store = prefix.strip_suffix(".db")?;
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

/// True for the `%Y%m%dT%H%M%SZ` stamp shape (16 chars, digits around `T`/`Z`).
fn is_family_stamp(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 16
        && b[8] == b'T'
        && b[15] == b'Z'
        && b[..8].iter().all(u8::is_ascii_digit)
        && b[9..15].iter().all(u8::is_ascii_digit)
}

/// Discover all forensic families under `root/db/` and classify them.
///
/// Families are found from their base file name or from sidecar-only members
/// (a coordination-sidecar quarantine has no base file). Foreign files
/// (`.DS_Store`, live stores) are ignored. Sorted by (store, kind, stamp).
fn list_families(root: &Path) -> Result<Vec<FamilyInfo>> {
    let db_dir = root.join("db");
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
    // appeared — a sidecar-only quarantine moves only `-wal`/`-shm`/`-tshm`).
    let mut bases: std::collections::BTreeMap<String, FamilyMeta> =
        std::collections::BTreeMap::new();
    for name in &names {
        if let Some(meta) = parse_family_name(name) {
            bases.insert(name.clone(), meta);
        }
    }
    for name in &names {
        for suffix in ["-wal", "-shm", "-tshm"] {
            if let Some(base) = name.strip_suffix(suffix) {
                if !bases.contains_key(base)
                    && let Some(meta) = parse_family_name(base)
                {
                    bases.insert(base.to_string(), meta);
                }
                break;
            }
        }
    }

    let mut families: Vec<FamilyInfo> = bases
        .into_iter()
        .map(|(id, meta)| classify_family(root, id, meta))
        .collect();
    families.sort_by(|a, b| (&a.store, a.kind, &a.stamp).cmp(&(&b.store, b.kind, &b.stamp)));
    Ok(families)
}

/// Classify one family's file set — pure filesystem inspection, never opens.
fn classify_family(root: &Path, id: String, meta: FamilyMeta) -> FamilyInfo {
    let db_path = root.join("db").join(&id);
    // Expected members per family kind — one source of truth for both the
    // listing and the Complete check. `-shm` is deliberately not expected:
    // the engine never creates it (turso_core 0.7.2 has no `-shm`
    // references), so a complete quarantine is db + wal + tshm — requiring
    // `-shm` would label every real full quarantine as `partial`. A leftover
    // foreign `-shm` still counts toward the listing below.
    let expected: &[(&str, &'static str)] = match meta.kind {
        FamilyKind::Quarantine => &[("", "db"), ("-wal", "wal"), ("-tshm", "tshm")],
        FamilyKind::PreReindex => &[("", "db"), ("-wal", "wal")],
    };
    let mut members: Vec<&'static str> = Vec::new();
    let mut size: u64 = 0;
    for (suffix, label) in expected {
        let path = db_path.with_file_name(format!("{id}{suffix}"));
        if let Ok(md) = std::fs::metadata(&path)
            && md.is_file()
        {
            members.push(label);
            size += md.len();
        }
    }
    let shm = db_path.with_file_name(format!("{id}-shm"));
    if let Ok(md) = std::fs::metadata(&shm)
        && md.is_file()
    {
        members.push("shm");
        size += md.len();
    }
    let class = if !members.contains(&"db") {
        FamilyClass::SidecarOnly
    } else if !db_header_ok(&db_path) {
        FamilyClass::BadHeader
    } else if expected.iter().all(|(_, label)| members.contains(label)) {
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
/// (quarantine + pre-reindex) without opening any database. Informational:
/// exits 0 whether or not families exist.
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

/// A temp-dir copy of a tshm-bearing family's `db` + `-wal` (the documented
/// snapshot read set). Created and removed around one query so the family's
/// own files — the forensic record — are never opened by the engine at all
/// (a possibly-corrupt `.tshm` would otherwise be probed/mapped by turso).
/// `tempfile` creates the dir 0700 with a collision-free random name and
/// `TempDir`'s Drop removes it on every path — early returns, copy failures,
/// and normal completion.
struct TempFamily {
    _dir: tempfile::TempDir,
    db_path: std::path::PathBuf,
}

impl TempFamily {
    fn create(db_path: &Path) -> Result<Self> {
        let name = db_path
            .file_name()
            .with_context(|| format!("family path must have a file name: {}", db_path.display()))?;
        let dir = tempfile::Builder::new()
            .prefix("mahbot-debug-family-")
            .tempdir()
            .with_context(|| "failed to create family query temp dir")?;
        let copy = dir.path().join(name);
        copy_file(db_path, &copy, "database")?;
        let wal = turso_mod::store_sidecars(db_path).wal;
        if wal.exists() {
            copy_file(
                &wal,
                &dir.path().join(format!("{}-wal", name.to_string_lossy())),
                "WAL",
            )?;
        }
        Ok(Self {
            db_path: copy,
            _dir: dir,
        })
    }

    #[must_use]
    fn db_path(&self) -> &Path {
        &self.db_path
    }
}

/// Copy one family file into the temp dir with private (0600) permissions.
fn copy_file(src: &Path, dst: &Path, what: &str) -> Result<()> {
    std::fs::copy(src, dst)
        .with_context(|| format!("failed to copy family {what} to {}", dst.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(dst, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// Open a forensic family read-only and execute `sql`, returning the rendered
/// output. Split from [`query_family`] so tests can assert result values
/// instead of only success.
///
/// Snapshot semantics: the family is static, so none of the live-store layers
/// apply — no IPC routing, no live heuristics. The engine opens in default
/// single-process mode (the read-only legacy path reads `db` + `-wal`
/// directly); a family carrying a stale `.tshm` leftover from a pre-removal
/// multiprocess run is copied to an OS temp dir first, omitting the
/// coordination file, so the no-touch guarantee comes from the copy, not the
/// opts. The whole open+query is panic-guarded: a damaged family yields a
/// clean error.
fn execute_family_query(family_id: &str, sql: &str, root: &Path) -> Result<String> {
    parse_family_name(family_id).with_context(|| {
        format!("invalid family id '{family_id}' — list valid ids with `mahbot debug families`")
    })?;
    let db_path = root.join("db").join(family_id);
    if !db_path.exists() {
        // A sidecar-only family (no base file — a quarantine that moved only
        // sidecars, or a failed pre-reindex copy) is a real family that cannot
        // be opened; a well-formed id with no member at all is a stale or
        // hand-constructed id — report the two distinctly.
        let any_member = ["-wal", "-shm", "-tshm"]
            .iter()
            .any(|s| db_path.with_file_name(format!("{family_id}{s}")).exists());
        if any_member {
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
    // routing / daemon-down validation the `--db` path applies.
    let md = std::fs::symlink_metadata(&db_path)
        .with_context(|| format!("cannot stat forensic family '{family_id}'"))?;
    if !md.file_type().is_file() {
        bail!("forensic family '{family_id}' main file is not a regular file — refusing to open");
    }
    let sidecars = turso_mod::store_sidecars(&db_path);
    // tshm-bearing families are copied to a temp dir (never opened in place).
    let temp = if sidecars.tshm.exists() {
        Some(TempFamily::create(&db_path)?)
    } else {
        None
    };
    let target = temp.as_ref().map_or(db_path.as_path(), |t| t.db_path());
    let result = guard_panics(|| {
        let (io, db) = open_readonly(target, &db_path, turso_mod::experimental_database_opts())?;
        connect_execute(&io, &db, sql, &db_path)
    });
    drop(temp);
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
fn query_family(family_id: &str, sql: &str, root: &Path) -> Result<()> {
    let output = execute_family_query(family_id, sql, root)?;
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

    // `mahbot debug detect [--db <name>]` — full predicate, no opens, no gate.
    if args.get(2).is_some_and(|a| a == "detect") {
        return run_debug_detect(&args, home_override);
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
        return query_family(&args[3], sql, &mahbot_home);
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
    // schema dump (per-table DDL blocks + row counts — see `dump_one_store`).
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

    // Routing hinges on whether the daemon is the single-process writer right
    // now: if it holds the instance flock the CLI must NOT open the live store
    // directly, so it goes through the debug IPC endpoint; if the daemon is
    // down, a direct read-only open is safe. `daemon_holds_lock_settled`
    // re-checks over a short window so a self-update handoff (flock momentarily
    // free between the old daemon's release and the new daemon's re-acquire)
    // never falls through to a concurrent direct open.
    let daemon_up = crate::util::lock::daemon_holds_lock_settled(&mahbot_home);

    // Single-process mode never creates a `.tshm`; remove any stale leftover
    // from a pre-removal multiprocess run once, before any store open. The
    // `-wal` is never touched (it may hold committed-but-uncheckpointed
    // frames; deleting it would silently lose commits).
    if !daemon_up {
        wal_guard::cleanup_stale_tshm(&mahbot_home);
    }

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

        let result = if daemon_up {
            match sql {
                Some(sql) => query_over_ipc(&mahbot_home, &physical, sql).await,
                None => dump_over_ipc(&mahbot_home, &physical, label).await,
            }
        } else {
            // Daemon down — direct single-process read-only open (the stale
            // `.tshm` cleanup for the whole root already ran above).
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
            match sql {
                Some(sql) => query_one_store(file_path, sql),
                None => dump_one_store(file_path, label),
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

/// Open one store read-only and run `sql` (pipe-delimited output) — the
/// daemon-down direct path (a single-process read-only open).
fn query_one_store(file_path: &Path, sql: &str) -> Result<()> {
    open_and_query_readonly(file_path, sql)
}

/// Open one store read-only and print its schema dump — a header line naming
/// the store, then one block per user table (`[table] <name>`, the table's
/// DDL, and its row count). Internal catalog artifacts are excluded (see
/// [`dump_schema`]). Same direct path as [`query_one_store`].
fn dump_one_store(file_path: &Path, label: &str) -> Result<()> {
    open_and_dump_readonly(file_path, label)
}

/// Query a live store through the daemon's debug IPC endpoint (used when the
/// daemon holds the instance lock and the CLI must not open the store
/// directly). `physical` is the physical store name the endpoint keys on
/// ("core" or "logs").
async fn query_over_ipc(root: &Path, physical: &str, sql: &str) -> Result<()> {
    let req = ipc::QueryRequest {
        store: physical.to_string(),
        sql: sql.to_string(),
        params: Vec::new(),
    };
    let resp = ipc::ipc_query_with_wait(root, &req).await?;
    if let Some(err) = &resp.error {
        bail!("{err}");
    }
    let mut out = String::new();
    out.push_str(&resp.columns.join("|"));
    out.push('\n');
    for row in &resp.rows {
        out.push_str(
            &row.iter()
                .map(ipc::WireValue::format)
                .collect::<Vec<_>>()
                .join("|"),
        );
        out.push('\n');
    }
    if resp.truncated {
        out.push_str(&format_truncation_row(resp.columns.len()));
        out.push('\n');
    }
    write_stdout(&out)
}

/// Schema dump of a live store through the daemon's debug IPC endpoint.
async fn dump_over_ipc(root: &Path, physical: &str, label: &str) -> Result<()> {
    use std::fmt::Write as _;
    let tables_sql = USER_TABLES_SQL.replace("{filter}", turso_mod::USER_OBJECT_FILTER);
    let req = ipc::QueryRequest {
        store: physical.to_string(),
        sql: tables_sql,
        params: Vec::new(),
    };
    let resp = ipc::ipc_query_with_wait(root, &req).await?;
    if let Some(err) = &resp.error {
        bail!("{err}");
    }
    let mut out = format!("== schema dump: {label} ==\n");
    for row in &resp.rows {
        let name = row.first().map(ipc::WireValue::format).unwrap_or_default();
        let sql = row.get(1).map(ipc::WireValue::format).unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let count_sql = format!("SELECT COUNT(*) FROM {}", quote_ident(&name));
        let count_req = ipc::QueryRequest {
            store: physical.to_string(),
            sql: count_sql,
            params: Vec::new(),
        };
        let count_resp = ipc::ipc_query_with_wait(root, &count_req).await?;
        if let Some(err) = &count_resp.error {
            bail!("{err}");
        }
        let count = count_resp
            .rows
            .first()
            .and_then(|r| r.first())
            .map(ipc::WireValue::format)
            .unwrap_or_default();
        writeln!(out, "\n[table] {name}\n{sql}\nrows: {count}\n").expect("writing to a String");
    }
    write_stdout(&out)
}

/// Prefix of [`guard_panics`] errors — shared with the classifier so a wording
/// change updates both sides (string-matching convention; a marker error type
/// would be more robust but heavier).
const ENGINE_PANIC_PREFIX: &str = "the database engine panicked while reading the store: ";

/// Prefix of stdout-write failures from [`write_stdout`] — the single wording
/// for every explicit stdout write (query results, listings, banners), so a
/// closed pipe surfaces consistently as an I/O error, never a panic.
const STDOUT_WRITE_ERROR_PREFIX: &str = "failed writing to stdout: ";

/// Run a blocking store open/query under `catch_unwind`, converting a panic
/// into a clean error — turso_core panics on specific corruption shapes
/// (pager/wal index OOB — the documented prod class), and the debug CLI must
/// never crash on a damaged store. The default panic hook is suppressed for
/// the duration so a caught panic does not spray stderr.
///
/// Safety: swapping the process-global panic hook is only sound because the
/// debug CLI is single-threaded (current-thread tokio runtime in `main.rs`);
/// a panic on any other thread during the window would be silently swallowed.
fn guard_panics<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(prev);
    match result {
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
/// Uses the low-level `turso::core` API because the SDK `Builder` cannot open
/// read-only: `OpenFlags::ReadOnly | OpenFlags::NoLock` guarantees the open
/// cannot create or mutate files. The core IO is synchronous (`FsIO`), so
/// this is intentionally blocking; the debug CLI runs on a single-task
/// current-thread runtime.
///
/// `open_path` is the file actually opened; `display_path` is the identity
/// named in errors (e.g. a forensic family opened through its temp copy
/// reports the family, not the OS temp dir).
pub(crate) fn open_readonly(
    open_path: &Path,
    display_path: &Path,
    opts: turso::core::DatabaseOpts,
) -> Result<(
    std::sync::Arc<dyn turso::core::IO>,
    std::sync::Arc<turso::core::Database>,
)> {
    let path_str = open_path
        .to_str()
        .with_context(|| format!("database path must be UTF-8: {}", open_path.display()))?;
    let io: std::sync::Arc<dyn turso::core::IO> =
        std::sync::Arc::new(turso::core::PlatformIO::new()?);
    let db = turso::core::Database::open_file_with_flags(
        io.clone(),
        path_str,
        turso::core::OpenFlags::ReadOnly | turso::core::OpenFlags::NoLock,
        opts,
        None,
    )
    .map_err(|e| {
        anyhow!(
            "failed to open database '{}' read-only: {e}",
            display_path.display()
        )
    })?;
    Ok((io, db))
}

/// Connect to an opened store, applying the in-memory temp-store setting.
///
/// Also used by the `bench-openrouter` subcommand for read-only config-store
/// lookups — see [`open_readonly`].
///
/// Every turso connection the application opens — the service connection
/// factory AND this debug CLI path (they do not share an opening path) —
/// runs with in-memory temp storage, so no statement or transaction can
/// ever fail because a temp directory is missing. turso_core's tempdir
/// creation has no parent-creation and no fallback chain, and resolves
/// through `$TMPDIR`, which the daemon pins to its private root; a missing
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
    io: &std::sync::Arc<dyn turso::core::IO>,
    db: &std::sync::Arc<turso::core::Database>,
    sql: &str,
    db_path: &Path,
) -> Result<String> {
    let conn = connect_readonly(db, db_path)?;
    execute_query_readonly(io, &conn, sql, db_path)
}

/// [`connect_execute`] plus a print to stdout. Write explicitly instead of
/// `print!`: a closed stdout (EPIPE) surfaces as an ordinary I/O error here
/// rather than a panic misreported as an engine crash by guard_panics.
fn connect_execute_print(
    io: &std::sync::Arc<dyn turso::core::IO>,
    db: &std::sync::Arc<turso::core::Database>,
    sql: &str,
    db_path: &Path,
) -> Result<()> {
    let output = connect_execute(io, db, sql, db_path)?;
    write_stdout(&output)
}

/// Open a store read-only and execute `sql`, printing pipe-delimited results.
/// See [`open_and_run_readonly`] for the shared open+run stack.
fn open_and_query_readonly(file_path: &Path, sql: &str) -> Result<()> {
    open_and_run_readonly(file_path, |io, db, path| {
        connect_execute_print(io, db, sql, path)
    })
}

/// Open a store read-only and run `runner` against it. Used by the query and
/// schema-dump paths so both share the same open stack. The whole open+run is
/// panic-guarded — a damaged store must yield a clean error, never a CLI crash.
fn open_and_run_readonly<T>(
    file_path: &Path,
    runner: impl FnOnce(
        &std::sync::Arc<dyn turso::core::IO>,
        &std::sync::Arc<turso::core::Database>,
        &Path,
    ) -> Result<T>,
) -> Result<T> {
    guard_panics(|| {
        let (io, db) = open_readonly(
            file_path,
            file_path,
            turso_mod::experimental_database_opts(),
        )?;
        runner(&io, &db, file_path)
    })
}

/// [`open_and_run_readonly`] runner for the schema dump: build the whole
/// store's dump text (buffered — a mid-dump failure never leaks a partial
/// dump to stdout, matching [`connect_execute`]'s all-or-nothing output),
/// then write it once.
fn open_and_dump_readonly(file_path: &Path, label: &str) -> Result<()> {
    open_and_run_readonly(file_path, |io, db, path| {
        let dump = dump_schema(io, db, path, label)?;
        write_stdout(&dump)
    })
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
    io: &std::sync::Arc<dyn turso::core::IO>,
    db: &std::sync::Arc<turso::core::Database>,
    db_path: &Path,
    label: &str,
) -> Result<String> {
    use std::fmt::Write as _;
    let conn = connect_readonly(db, db_path)?;
    let tables_sql = USER_TABLES_SQL.replace("{filter}", turso_mod::USER_OBJECT_FILTER);
    let tables = collect_rows(io, &conn, &tables_sql, db_path, |row| {
        let name = format_core_value(row.get_value(0));
        let sql = format_core_value(row.get_value(1));
        Ok((name, sql))
    })?;

    let mut out = format!("== schema dump: {label} ==\n");
    for (name, sql) in tables {
        let count_sql = format!("SELECT COUNT(*) FROM {}", quote_ident(&name));
        let counts = collect_rows(io, &conn, &count_sql, db_path, |row| {
            Ok(format_core_value(row.get_value(0)))
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

/// Step a single statement to completion, collecting one owned value per row.
/// Mirrors [`execute_query_readonly`]'s IO/Yield/Busy/Interrupt handling so the
/// dump queries share the same robustness (a failure propagates and the caller
/// reports it as a clean error).
fn collect_rows<T>(
    io: &std::sync::Arc<dyn turso::core::IO>,
    conn: &std::sync::Arc<turso::core::Connection>,
    sql: &str,
    db_path: &Path,
    mut collect: impl FnMut(&turso::core::Row) -> Result<T>,
) -> Result<Vec<T>> {
    let mut stmt = conn
        .query(sql)
        .map_err(|e| anyhow!("SQL query failed on '{}': {e}", db_path.display()))?
        .ok_or_else(|| anyhow!("query produced no statement on '{}'", db_path.display()))?;

    let mut rows = Vec::new();
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
                let row = stmt
                    .row()
                    .ok_or_else(|| anyhow!("row missing after StepResult::Row"))?;
                rows.push(collect(row)?);
            }
            turso::core::StepResult::Interrupt => {
                bail!("query interrupted on '{}'", db_path.display())
            }
            turso::core::StepResult::Busy => {
                bail!("database busy on '{}'; try again later", db_path.display())
            }
        }
    }
    Ok(rows)
}

/// Double-quote a SQL identifier, escaping embedded double quotes.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// The physical database files for the debug CLI, each opened once.
///
/// After consolidation all the domain stores share ONE file, so `--db all` and
/// `detect` (with no `--db`) open each unique file once rather than once per
/// logical domain name. Labels are the physical store names from
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
pub(crate) fn validate_read_only(sql: &str) -> Result<()> {
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
            if !SAFE_PRAGMAS.contains(&name.text.to_lowercase().as_str()) {
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
        turso::core::Value::Blob(b) => crate::util::hex_string(b),
    }
}

/// Format a truncation sentinel row that matches the column count.
///
/// For 1 column:  `truncated`
/// For 2 columns: `truncated|truncated`
/// For N≥3:       `...|truncated|truncated|...|...`
///                 (ellipsis at first and last, `truncated` in between)
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
    eprintln!("       mahbot debug detect [--db <name>]");
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
    eprintln!("  detect      classify single-process store health (structural/durable-b)");
    eprintln!("               without opening stores; reports stale .tshm leftovers");
    eprintln!("  families    list quarantine/pre-reindex forensic families (--db filters by");
    eprintln!("               store name; a name matching nothing prints an empty list)");
    eprintln!("  --family <id>  read-only SQL against one forensic family (id from `families`)");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  mahbot debug --db board");
    eprintln!("  mahbot debug --db all");
    eprintln!("  mahbot debug --db board \"SELECT phase, COUNT(*) FROM tickets GROUP BY phase\"");
    eprintln!("  mahbot debug --db all \"SELECT name FROM sqlite_master WHERE type='table'\"");
    eprintln!("  mahbot debug detect");
    eprintln!("  mahbot debug families");
    eprintln!(
        "  mahbot debug --family board.db.quarantine-20260812T120000Z-1234 \"SELECT COUNT(*) FROM tickets\""
    );
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

    /// End-to-end: `run_debug_with_args` opens a real (temporary) store
    /// read-only through the `turso::core` path and runs a query against it.
    /// Serialized: `guard_panics` swaps the process-global panic hook, so
    /// guard windows must not overlap other debug tests.
    #[tokio::test]
    #[serial_test::serial(family)]
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
        // the store directory (in particular no -tshm — single-process mode
        // uses the standard -shm and never creates a -tshm).
        let db_dir = dir.path().join("db");
        let names: Vec<String> = std::fs::read_dir(&db_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            names.iter().any(|n| n == "logs.db"),
            "store db must exist: {names:?}"
        );
        let tshm_count = names.iter().filter(|n| n.ends_with("-tshm")).count();
        assert_eq!(
            tshm_count, 0,
            "single-process mode must not create a -tshm file: {names:?}"
        );
    }

    /// The schema dump (`--db <name>` with no SQL) reflects the LIVE stored
    /// schema: one block per user table with its DDL and row count, excluding
    /// internal catalog artifacts (`sqlite_%`, `__turso_internal_%`).
    /// Serialized: `guard_panics` swaps the process-global panic hook, so
    /// guard windows must not overlap other debug tests.
    #[tokio::test]
    #[serial_test::serial(family)]
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
        let dump =
            open_and_run_readonly(&db_path, |io, db, path| dump_schema(io, db, path, "logs"))
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
    }

    /// `mahbot debug --db <name>` without a SQL argument dumps the store schema
    /// and exits 0 — and, like the query path, creates no extra `-tshm` files.
    /// Serialized: `guard_panics` swaps the process-global panic hook, so
    /// guard windows must not overlap other debug tests.
    #[tokio::test]
    #[serial_test::serial(family)]
    async fn run_debug_dumps_schema_without_sql() {
        let (_store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "--db".to_string(),
            "logs".to_string(),
        ];
        let result = run_debug_with_args(args, Some(dir.path().to_path_buf())).await;
        assert!(
            result.is_ok(),
            "schema dump without SQL must succeed: {result:?}"
        );

        let db_dir = dir.path().join("db");
        let names: Vec<String> = std::fs::read_dir(&db_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        // Single-process mode never creates a -tshm file, so the read-only CLI
        // dump must not have created one either.
        let tshm_count = names.iter().filter(|n| n.ends_with("-tshm")).count();
        assert_eq!(tshm_count, 0, "no -tshm file may be created: {names:?}");
    }

    /// `mahbot debug --db all` without a SQL argument dumps every present
    /// store and reports missing stores per-store with a failure summary
    /// (exit 1) — matching the query verb's `--db all` failure semantics.
    /// Serialized: `guard_panics` swaps the process-global panic hook, so
    /// guard windows must not overlap other debug tests.
    #[tokio::test]
    #[serial_test::serial(family)]
    async fn run_debug_dump_all_reports_missing_stores() {
        let (_store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "--db".to_string(),
            "all".to_string(),
        ];
        let err = run_debug_with_args(args, Some(dir.path().to_path_buf()))
            .await
            .expect_err("--db all with missing stores must report a failure summary");
        let msg = format!("{err:#}");
        // Compute the expectation from the physical store source (the single
        // source of truth) so the test does not hardcode the current store
        // count — only the logs store exists, so the consolidated core file
        // is missing.
        let total = physical_store_list(dir.path()).len();
        assert!(
            msg.contains(&format!("{} of {total} store(s) failed", total - 1)),
            "must summarize per-store failures, got: {msg}"
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

        // Complete quarantine: db + wal + tshm (the engine's real sidecar set
        // — `-shm` is never created by turso_core).
        let c = "board.db.quarantine-20260812T120000Z-100";
        std::fs::write(db_dir.join(c), valid_db_bytes()).unwrap();
        for s in ["-wal", "-tshm"] {
            std::fs::write(db_dir.join(format!("{c}{s}")), b"x").unwrap();
        }
        // Complete pre-reindex: db + wal (no shm/tshm by design).
        let p = "logs.db.pre-reindex-20260812T120000Z-200";
        std::fs::write(db_dir.join(p), valid_db_bytes()).unwrap();
        std::fs::write(db_dir.join(format!("{p}-wal")), b"x").unwrap();
        // Sidecar-only quarantine: no base file.
        let s = "sessions.db.quarantine-20260812T120000Z-300";
        std::fs::write(db_dir.join(format!("{s}-wal")), b"x").unwrap();
        std::fs::write(db_dir.join(format!("{s}-tshm")), b"x").unwrap();
        // Partial quarantine: db + wal, missing tshm.
        let pa = "users.db.quarantine-20260812T120000Z-400";
        std::fs::write(db_dir.join(pa), valid_db_bytes()).unwrap();
        std::fs::write(db_dir.join(format!("{pa}-wal")), b"x").unwrap();
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
        assert_eq!(by_id[c].files, "db,wal,tshm");
        assert_eq!(by_id[p].class, FamilyClass::Complete);
        assert_eq!(by_id[p].files, "db,wal");
        assert_eq!(by_id[s].class, FamilyClass::SidecarOnly);
        assert_eq!(by_id[s].files, "wal,tshm");
        assert_eq!(by_id[pa].class, FamilyClass::Partial);
        assert_eq!(by_id[bh].class, FamilyClass::BadHeader);
        // Sorted by (store, kind, stamp): board, config, logs, sessions, users.
        let stores: Vec<&str> = families.iter().map(|f| f.store.as_str()).collect();
        assert_eq!(stores, ["board", "config", "logs", "sessions", "users"]);
    }

    /// Rename a live store family into a forensic family name (the boot path
    /// renames, it does not copy; missing sidecars are skipped). Returns the
    /// moved member file names.
    fn move_family_aside(db_dir: &Path, base: &Path, fam: &str) -> Vec<String> {
        let sidecars = turso_mod::store_sidecars(base);
        let mut moved = Vec::new();
        for (src, suffix) in [
            (base, ""),
            (&sidecars.wal, "-wal"),
            (&sidecars.shm, "-shm"),
            (&sidecars.tshm, "-tshm"),
        ] {
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

    /// End-to-end: a quarantined family (moved db + sidecars, corrupt `.tshm`
    /// — the forensic case) is queryable via `--family` with the same
    /// read-only guarantees: the temp-copy path omits the family's `.tshm`,
    /// and the family files are unchanged with no new files beside them.
    ///
    /// Serialized with the other family tests: `guard_panics` swaps the
    /// process-global panic hook, so overlapping guard windows would lose
    /// stderr diagnostics from concurrent test panics.
    #[tokio::test]
    #[serial_test::serial(family)]
    async fn run_debug_queries_a_quarantined_family() {
        let (store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        let db_dir = dir.path().join("db");
        let fam = "logs.db.quarantine-20260812T120000Z-4242";
        // One committed tool-call row before the move: the query result below
        // must prove the temp-copy read returns real data, not just "some
        // query succeeded".
        store
            .flush_batch(
                "j1",
                "Engineer",
                "ws1",
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
        let _moved = move_family_aside(&db_dir, &db_dir.join("logs.db"), fam);
        // Corrupt the family's tshm (garbage — the record of the corruption).
        std::fs::write(db_dir.join(format!("{fam}-tshm")), vec![0xAB; 64]).unwrap();

        // Snapshot every file beside the family (including the corrupt tshm we
        // just wrote) so the no-mutation proof covers the whole family set.
        let before: Vec<((u64, std::time::SystemTime), String)> = std::fs::read_dir(&db_dir)
            .unwrap()
            .map(|e| {
                let path = e.unwrap().path();
                (
                    file_state(&path),
                    path.file_name().unwrap().to_string_lossy().into_owned(),
                )
            })
            .collect();

        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "--family".to_string(),
            fam.to_string(),
            "SELECT COUNT(*) FROM tool_calls".to_string(),
        ];
        run_debug_with_args(args, Some(dir.path().to_path_buf()))
            .await
            .expect("quarantined family (corrupt tshm) must be queryable via the temp copy");

        // The family must be untouched and no new files created beside it.
        for (state, name) in &before {
            assert_eq!(
                file_state(&db_dir.join(name)),
                *state,
                "family file must be unchanged: {name}"
            );
        }
        let after_names: Vec<String> = std::fs::read_dir(&db_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            after_names.len(),
            before.len(),
            "no new files beside the family: {after_names:?}"
        );

        // Data fidelity: the same open path returns the committed row count.
        let out = execute_family_query(fam, "SELECT COUNT(*) FROM tool_calls", dir.path()).unwrap();
        assert_eq!(
            out, "COUNT(*)\n1\n",
            "temp-copy query must return the committed row"
        );
    }

    /// Query-error paths report distinct, clear messages: a sidecar-only
    /// quarantine (no main DB) and a garbage main DB both fail without
    /// crashing, naming the family.
    #[tokio::test]
    #[serial_test::serial(family)]
    async fn run_debug_family_error_paths_report_clear_errors() {
        let dir = tempfile::TempDir::new().unwrap();
        let db_dir = dir.path().join("db");
        std::fs::create_dir_all(&db_dir).unwrap();
        let fam = "logs.db.quarantine-20260812T120000Z-4242";
        let run = |sql: &str| {
            let args = vec![
                "mahbot".to_string(),
                "debug".to_string(),
                "--family".to_string(),
                fam.to_string(),
                sql.to_string(),
            ];
            run_debug_with_args(args, Some(dir.path().to_path_buf()))
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
    #[serial_test::serial(family)]
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

    /// A tshm-less family (pre-reindex shape: db + wal) is queried **in
    /// place** — the legacy read-only WAL path must not create or modify any
    /// file beside the family.
    #[tokio::test]
    #[serial_test::serial(family)]
    async fn run_debug_queries_pre_reindex_family_in_place() {
        let (store, dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        let db_dir = dir.path().join("db");
        let fam = "logs.db.pre-reindex-20260812T120000Z-4242";
        // The committed row lives only in the wal (no checkpoint): the query
        // result below must show it, behaviorally proving the legacy path
        // reads db + wal — a db-only read would report COUNT 0.
        store
            .flush_batch(
                "j1",
                "Engineer",
                "ws1",
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

        // Move the whole family aside, then drop shm/tshm so the family has
        // the pre-reindex shape (db + wal, no coordination files).
        move_family_aside(&db_dir, &db_dir.join("logs.db"), fam);
        let _ = std::fs::remove_file(db_dir.join(format!("{fam}-shm")));
        let _ = std::fs::remove_file(db_dir.join(format!("{fam}-tshm")));

        // The moved wal holds the store's schema frames (the test store is
        // dropped without a checkpoint), so a successful query — with no
        // tshm to coordinate through — exercises the legacy db + wal read.
        assert!(
            std::fs::metadata(db_dir.join(format!("{fam}-wal")))
                .unwrap()
                .len()
                > 0,
            "pre-reindex wal must hold committed frames"
        );

        let before: Vec<((u64, std::time::SystemTime), String)> = std::fs::read_dir(&db_dir)
            .unwrap()
            .map(|e| {
                let path = e.unwrap().path();
                (
                    file_state(&path),
                    path.file_name().unwrap().to_string_lossy().into_owned(),
                )
            })
            .collect();

        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "--family".to_string(),
            fam.to_string(),
            "SELECT COUNT(*) FROM tool_calls".to_string(),
        ];
        run_debug_with_args(args, Some(dir.path().to_path_buf()))
            .await
            .expect("pre-reindex family must be queryable in place");

        // Same no-mutation proof as the temp-copy test: identical file set,
        // and every file's size + mtime unchanged.
        let after_names: Vec<String> = std::fs::read_dir(&db_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            after_names.len(),
            before.len(),
            "in-place family query must not create files beside the family"
        );
        for (state, name) in &before {
            assert_eq!(
                file_state(&db_dir.join(name)),
                *state,
                "family file must be unchanged: {name}"
            );
        }

        // Data fidelity: the wal-only committed row must be visible.
        let out = execute_family_query(fam, "SELECT COUNT(*) FROM tool_calls", dir.path()).unwrap();
        assert_eq!(
            out, "COUNT(*)\n1\n",
            "in-place query must read the wal-only row"
        );
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
            let args = vec![
                "mahbot".to_string(),
                "debug".to_string(),
                "--family".to_string(),
                bad.to_string(),
                "SELECT 1".to_string(),
            ];
            let err = run_debug_with_args(args, Some(dir.path().to_path_buf()))
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
            vec!["detect", "--help"],
            vec!["families", "-h"],
            vec!["detect", "-h"],
            vec!["--family", "--help"],
            vec!["--family", "-h"],
            vec!["families", "--db", "--help"],
            vec!["detect", "--db", "--help"],
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
            vec!["detect", "--db", "board", "--help"],
            vec!["detect", "--help", "extra"],
        ] {
            let mut args = vec!["mahbot".to_string(), "debug".to_string()];
            args.extend(tail.into_iter().map(str::to_owned));
            run_debug_with_args(args, Some(dir.path().to_path_buf()))
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
        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "families".to_string(),
            "--db".to_string(),
            "nonexistent".to_string(),
        ];
        run_debug_with_args(args, Some(dir.path().to_path_buf()))
            .await
            .expect("families --db <matching-nothing> must print nothing and exit 0");

        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "--family".to_string(),
            "logs.db.quarantine-20260812T120000Z-4242".to_string(),
            "SELECT 1".to_string(),
        ];
        let err = run_debug_with_args(args, Some(dir.path().to_path_buf()))
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
        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "families".to_string(),
            "--db".to_string(),
            "stats".to_string(),
        ];
        run_debug_with_args(args, Some(dir.path().to_path_buf()))
            .await
            .expect("legacy store family must be filterable by --db");
    }

    /// `debug detect` classifies synthetic file sets without opening the store
    /// through turso: a valid main DB reports `healthy` (succeeds), a garbage
    /// (structurally corrupt) main DB reports `structural` and fails.
    #[test]
    fn debug_detect_reports_non_healthy_state() {
        let dir = tempfile::TempDir::new().unwrap();
        let db_dir = dir.path().join("db");
        std::fs::create_dir_all(&db_dir).unwrap();

        // A structurally corrupt main DB (garbage header) + a healthy one.
        std::fs::write(db_dir.join("core.db"), vec![0x42; 128]).unwrap();
        std::fs::write(db_dir.join("logs.db"), valid_db_bytes()).unwrap();

        // Healthy store (logs) — detect succeeds.
        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "detect".to_string(),
            "--db".to_string(),
            "logs".to_string(),
        ];
        assert!(run_debug_detect(&args, Some(dir.path().to_path_buf())).is_ok());

        // Structurally corrupt store (board resolves to core.db) — detect fails.
        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "detect".to_string(),
            "--db".to_string(),
            "board".to_string(),
        ];
        let err = run_debug_detect(&args, Some(dir.path().to_path_buf()))
            .expect_err("structurally corrupt store must fail detect");
        assert!(
            format!("{err:#}").contains("structurally corrupt"),
            "got: {err:#}"
        );
    }

    /// `debug detect --db all` resolves to every physical store (core + logs),
    /// matching the `--db all` form of the main debug verb.
    #[test]
    fn detect_db_all_diagnoses_every_physical_store() {
        let dir = tempfile::TempDir::new().unwrap();
        let db_dir = dir.path().join("db");
        std::fs::create_dir_all(&db_dir).unwrap();
        std::fs::write(db_dir.join("core.db"), valid_db_bytes()).unwrap();
        std::fs::write(db_dir.join("logs.db"), valid_db_bytes()).unwrap();

        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "detect".to_string(),
            "--db".to_string(),
            "all".to_string(),
        ];
        assert!(
            run_debug_detect(&args, Some(dir.path().to_path_buf())).is_ok(),
            "detect --db all must succeed on healthy core + logs stores"
        );
    }

    /// The IPC `WireValue` round-trips through `from_turso`/`to_turso` and
    /// renders NULL as empty, integers as decimals, and blobs as lowercase hex.
    #[test]
    fn wire_value_round_trips() {
        use crate::db::Value;
        use crate::db::ipc::WireValue;
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

    /// End-to-end daemon-down path: a row committed to the WAL (the store is
    /// dropped without a checkpoint) is visible through the direct
    /// single-process read-only open — no IPC, no live daemon.
    #[tokio::test]
    #[serial_test::serial(family)]
    async fn run_debug_daemon_down_reads_committed_wal() {
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
            !crate::util::lock::daemon_holds_lock(dir.path()),
            "temp root has no mahbot.lock — daemon must be considered down"
        );
        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "--db".to_string(),
            "logs".to_string(),
            "SELECT COUNT(*) FROM logs".to_string(),
        ];
        run_debug_with_args(args, Some(dir.path().to_path_buf()))
            .await
            .expect("daemon-down read of committed WAL rows must succeed");

        // Prove the WAL-only row is actually visible via the direct open.
        let db_path = dir.path().join("db").join("logs.db");
        let out = open_and_run_readonly(&db_path, |io, db, path| {
            connect_execute(io, db, "SELECT COUNT(*) FROM logs", path)
        })
        .expect("direct read-only open must succeed");
        assert_eq!(
            out, "COUNT(*)\n1\n",
            "WAL-only committed row must be visible"
        );
    }
}
