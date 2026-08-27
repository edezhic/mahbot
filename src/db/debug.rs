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
//! Three further verbs: `detect` classifies store coordination state without
//! opening any database; `families` and `--family` target forensic artifact
//! families (quarantine and pre-reindex records — static snapshots of a store
//! family renamed aside at boot, never deleted):
//!
//! - `mahbot debug detect [--db <name>]` — classify coordination state without
//!   opening stores (exit 1 when any store is checkpoint-blocking).
//! - `mahbot debug families [--db <name>]` — list every quarantine and
//!   pre-reindex family in the store directory with its original store,
//!   artifact type, timestamp, total size, and a file-set/header
//!   classification. No database is opened.
//! - `mahbot debug --family <id> "SQL query"` — query one family with the
//!   same read-only guarantees as live stores. Families are static
//!   snapshots: none of the live-store layers apply (no flock gate, no
//!   artifact pre-check, no torn-frame retry, no lock-drop verification),
//!   and the engine opens without multiprocess WAL (legacy read-only WAL
//!   path). The legacy path still probes a present `.tshm`, so families that
//!   carry one are copied to an OS temp dir first (the copy omits the
//!   `.tshm`); the family files themselves are never modified and no new
//!   files appear beside them. Engine panics on corrupt families are caught
//!   and reported as clean errors instead of crashing the CLI.
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

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use tokio::time::sleep;

use crate::db as turso_mod;
use crate::db::wal_guard;

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
///
/// The list mirrors turso_core 0.7.2's user-facing error texts — the four
/// `CompletionError` display strings plus `Corrupt("Invalid page type: …")`.
/// (A bare "torn" word is deliberately absent: it appears in no turso_core
/// error text, only comments and test panics, and would misclassify user SQL
/// like "no such table: torn_table" as store corruption.)
const TORN_FRAME_SIGNATURES: &[&str] = &[
    "short read on wal frame",
    "short read on page",
    "invalid page type",
    "wal frame page mismatch",
    "checksum mismatch",
];

/// Maximum attempts (including the first) for a torn-frame open/query failure.
const MAX_OPEN_ATTEMPTS: usize = 5;

/// Backoff (seconds) between retry attempts, indexed by `attempt`.
const OPEN_RETRY_BACKOFF_SECS: [u64; MAX_OPEN_ATTEMPTS - 1] = [1, 2, 4, 8];

/// Run the debug subcommand. Parses `env::args()` for the debug invocation.
///
/// Returns `Ok(())` on success (exit code 0), `Err` on failure (exit code 1).
/// A gate refusal is a [`GateRefusal`] (mapped to exit code 2 by the caller).
/// Prints usage to stderr and returns `Err` for invalid argument combinations.
pub async fn run_debug() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    run_debug_with_args(args, None).await
}

/// Marker error for the CLI flock-gate refusal (fixed exit code 2 — 0=ok and
/// 1=error are already taken, and scripts/agents invoke `mahbot debug` while
/// the daemon is live).
#[derive(Debug)]
pub struct GateRefusal;

impl std::fmt::Display for GateRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "another process holds the mahbot instance lock; retry later"
        )
    }
}

impl std::error::Error for GateRefusal {}

/// Overall deadline for the flock-gate state machine (bounds the polling).
const GATE_TIMEOUT_SECS: u64 = 15;

/// Consecutive free observations required before taking the flock in
/// crash-recovery mode (narrows — not closes — the self-update handoff
/// window: in the saw-busy path `take_flock` races the incoming daemon's
/// fail-fast acquire — a lost race restarts the observations; the no-flock
/// path is covered by the post-open byte-0 verification).
const GATE_FREE_OBSERVATIONS: u32 = 2;

/// PID of the process holding a write lock on `.tshm` byte-0 (the lifetime-
/// lock byte) via `fcntl(F_GETLK)` with `F_WRLCK` semantics — a read-only
/// probe that never takes the lock. `None` = free or no `.tshm` (snapshot
/// copy → the gate does not apply). `F_RDLCK` probing would miss the daemon's
/// shared read locks (false "free"), so `F_WRLCK` is used. The PID lets the
/// gate detect a completed self-update handoff (the byte-0 holder changes)
/// without ever re-probing the flock.
#[cfg(unix)]
fn probe_tshm_byte0_pid(tshm_path: &Path) -> Option<i32> {
    use std::os::unix::io::AsRawFd;
    let file = std::fs::File::open(tshm_path).ok()?;
    let mut fl: libc::flock = unsafe { std::mem::zeroed() };
    fl.l_type = libc::F_WRLCK;
    fl.l_whence = 0; // SEEK_SET
    fl.l_start = 0;
    fl.l_len = 0;
    fl.l_pid = 0;
    // SAFETY: F_GETLK is a non-blocking, non-mutating probe; the fd is valid.
    let ret = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETLK, &mut fl) };
    (ret != -1 && fl.l_type != libc::F_UNLCK).then_some(fl.l_pid)
}

#[cfg(windows)]
fn probe_tshm_byte0_pid(_tshm_path: &Path) -> Option<i32> {
    None // Windows has no fcntl lifetime lock; the gate is unix-only
}

/// Probe whether the instance flock is currently free. flock() has no
/// F_GETLK equivalent, so this is a transient `LOCK_EX|LOCK_NB` acquire+
/// release. Callers must probe byte-0 **first** and skip this probe while
/// byte-0 is held (self-update handoff) — a transient acquire in that window
/// can kill a fail-fast incoming self-update daemon.
fn probe_flock_free(lock_path: &Path) -> bool {
    // Delegating wrapper: the acquired File is dropped immediately by the
    // `is_some()` expression, keeping this a transient observation (unlike
    // take-and-hold callers, which must keep the returned File alive).
    take_flock(lock_path).is_some()
}

/// Take (and hold) the instance flock — crash-recovery mode. The returned
/// `File` must be kept alive for the duration of the debug run; dropping it
/// releases the kernel lock.
fn take_flock(lock_path: &Path) -> Option<File> {
    use std::fs::OpenOptions;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)
        .ok()?;
    match crate::util::lock::try_flock(&file) {
        Ok(true) => Some(file),
        _ => None,
    }
}

/// Apply the CLI flock-gate before any turso open of a live store.
///
/// Returns `Some(file)` when the gate acquired the instance flock
/// (crash-recovery after a busy→free transition) — the caller must keep the
/// `File` alive for the run so a daemon cannot start between the gate's
/// observations and the turso open. Returns `Ok(None)` when no flock is held
/// by this process (no lock file at all, a live daemon holds it and the read
/// proceeds as MultiProcess, or no daemon was ever observed and the read
/// proceeds like the no-lock-file case).
///
/// State machine (validated by the research round):
/// - flock held + byte-0 held → proceed (MultiProcess read);
/// - flock held + byte-0 free → wait 1s (macOS lock-drop state; a live daemon
///   whose fcntl locks are gone must not be opened as Exclusive);
/// - flock free + byte-0 held on any store → wait, never take the flock
///   (self-update handoff: the old process is alive; taking the flock would
///   kill the incoming daemon). The handoff state itself never probes — the
///   only transient flock acquires are the observation that establishes it
///   and the exit when byte-0 goes free (the old daemon exited — normal flow
///   resumes), each a sub-ms risk of landing in the old→new daemon gap.
///   Outside the handoff the loop probes once per observation (failed
///   acquires while the flock is held are harmless). The exit is otherwise
///   detected without probing: a byte-0 holder PID change means the new
///   daemon opened stores (it holds the flock already — boot order: flock
///   before stores);
/// - flock free + all byte-0 free → after two consecutive observations: take
///   the flock (crash-recovery) and hold it — but only once a busy→free
///   transition was observed (a "both free" state from the start could be the
///   self-update handoff gap, where taking the flock would kill the incoming
///   fail-fast daemon; the post-open byte-0 verification covers the residual
///   handoff-gap window). With no busy observation ever, proceed without the
///   flock (fresh install / long-exited daemon — the no-lock-file case);
/// - timeout → refuse with a [`GateRefusal`] (exit code 2).
///
/// Unix-only: the lock-drop vector is process-scoped fcntl (macOS), and
/// Windows LockFileEx locks are handle-scoped — the byte-0 probe has no
/// F_GETLK equivalent there, so the gate is skipped (a live daemon would
/// otherwise misread as permanent lock-drop and refuse every invocation).
///
/// Rollout note: the bounded polling assumes the daemon carries the
/// persistent-fd fix — against a pre-fix daemon the lock-drop state is
/// near-permanent and every invocation polls up to the deadline before
/// refusing (exit 2). Daemons and their shell-tool agents share the deployed
/// binary version, so this only affects a mixed-version rollout; the refusal
/// is safe either way (never opens as Exclusive).
///
/// Only live stores (`.tshm` present) participate. Snapshot copies (db/-wal
/// only) never block — the documented snapshot-query workflow must keep
/// working while the daemon is live. When `mahbot.lock` does not exist at all
/// no daemon has ever started — proceed directly.
async fn flock_gate(root: &Path, db_names: &[String]) -> Result<Option<File>> {
    flock_gate_with_timeout(root, db_names, Duration::from_secs(GATE_TIMEOUT_SECS)).await
}

/// [`flock_gate`] with an explicit deadline (tests inject a short one).
async fn flock_gate_with_timeout(
    root: &Path,
    db_names: &[String],
    timeout: Duration,
) -> Result<Option<File>> {
    // Unix-only (see [`flock_gate`]): on other platforms every live daemon
    // would misread as permanent lock-drop (no fcntl byte-0 probe) and refuse.
    if cfg!(not(unix)) {
        return Ok(None);
    }
    let lock_path = crate::util::lock::lock_file_path(root);
    if !lock_path.exists() {
        return Ok(None);
    }
    let tshm_paths: Vec<PathBuf> = db_names
        .iter()
        .map(|n| turso_mod::store_sidecars(&turso_mod::store_db_path(root, n)).tshm)
        .filter(|p| p.exists())
        .collect();
    if tshm_paths.is_empty() {
        return Ok(None); // snapshot copies only
    }

    let deadline = tokio::time::Instant::now() + timeout;
    let mut consecutive_free = 0u32;
    // Self-update handoff state (flock free + byte-0 held by the old daemon).
    // Established by one transient flock probe; while set, the flock is never
    // probed again — the exit is detected via the byte-0 holder PID.
    let mut handoff = false;
    let mut handoff_pids: Vec<Option<i32>> = Vec::new();
    // A daemon's flock was observed held at least once. Crash-recovery takes
    // the flock only after that busy→free transition: without it, the "both
    // free" state could be the self-update handoff gap (old daemon released,
    // new daemon not yet through acquire_lock — macOS Gatekeeper validation
    // can stretch this window), and taking the flock would kill the fail-fast
    // incoming daemon.
    let mut saw_busy = false;
    loop {
        let byte0_pids: Vec<Option<i32>> =
            tshm_paths.iter().map(|p| probe_tshm_byte0_pid(p)).collect();
        let all_byte0_free = byte0_pids.iter().all(Option::is_none);
        // Probe byte-0 first. The flock probe is a transient acquire; while
        // the handoff state is established it is skipped — a transient
        // acquire in the old→new daemon gap would kill the fail-fast
        // incoming self-update daemon.
        let flock_free = if handoff {
            if all_byte0_free {
                handoff = false; // old daemon exited — normal flow resumes
                probe_flock_free(&lock_path)
            } else if byte0_pids
                .iter()
                .zip(&handoff_pids)
                .any(|(cur, prev)| cur.is_some() && cur != prev)
            {
                // The byte-0 holder changed — the new daemon opened stores.
                // It holds the flock already (boot order) → MultiProcess read.
                return Ok(None);
            } else {
                true // still the old daemon mid-handoff — never probe
            }
        } else {
            probe_flock_free(&lock_path)
        };
        if !flock_free {
            saw_busy = true;
            if all_byte0_free {
                // flock held + byte-0 free: a live daemon whose fcntl locks
                // are dropped (lock-drop state). Opening would classify
                // Exclusive — wait for the byte-0 lock to reappear.
                handoff = false;
                consecutive_free = 0;
            } else {
                // flock held + byte-0 held: normal MultiProcess read — safe.
                return Ok(None);
            }
        } else if all_byte0_free {
            consecutive_free += 1;
            if consecutive_free >= GATE_FREE_OBSERVATIONS {
                if saw_busy {
                    // Crash-recovery: the daemon we saw is gone. Take the
                    // flock and hold it for the rest of the run so a daemon
                    // cannot start between the observations and the open.
                    // Residual gap (acknowledged): a NEW self-update daemon
                    // starting in the busy→free window loses the race — its
                    // fail-fast acquire_lock fails after the old daemon
                    // exited (service outage). The post-open byte-0
                    // verification never runs here (guard is Some), so this
                    // is caught by nothing.
                    if let Some(guard) = take_flock(&lock_path) {
                        return Ok(Some(guard));
                    }
                } else {
                    // No daemon was ever observed (fresh install or one that
                    // exited before the first observation). Proceed without
                    // the flock — the no-lock-file equivalent; a daemon
                    // starting after the final observation is caught by the
                    // post-open byte-0 verification.
                    return Ok(None);
                }
                // Lost the race (a daemon started) — restart the observations.
                consecutive_free = 0;
            }
        } else {
            // flock free + byte-0 held: self-update handoff — never take the
            // flock; wait for the old process to finish.
            if !handoff {
                handoff = true;
                handoff_pids = byte0_pids;
            }
            consecutive_free = 0;
        }
        if tokio::time::Instant::now() >= deadline {
            // Single user-facing print: main.rs maps the GateRefusal to exit
            // code 2 and prints `{e:#}` — the context carries the timeout
            // diagnosis on top of the generic refusal cause.
            return Err(anyhow!(GateRefusal).context(format!(
                "timed out waiting for the mahbot instance lock ({}s)",
                timeout.as_secs()
            )));
        }
        sleep(Duration::from_secs(1)).await;
    }
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
fn validate_store_name(name: &str, all_valid: bool) -> Result<()> {
    let mut names = turso_mod::store_names();
    names.push(turso_mod::CONSOLIDATED_DB_NAME);
    if names.contains(&name) {
        return Ok(());
    }
    let hint = names.join(", ") + if all_valid { ", all" } else { "" };
    bail!("invalid database name '{name}'. Valid names: {hint}");
}

/// `mahbot debug detect [--db <name>]` — run the full coordination-state
/// predicate over the stores without opening them. Prints one line per store
/// (`name\tclass\twal_size=N\tblocking=B`). Exit 0 when all stores are healthy
/// or warn-only; exit 1 (via `Err`) only when a store is in a
/// checkpoint-blocking class — warn-only classes (oversized WAL, torn-pre
/// index, unreadable `.tshm`) are reported but are not failures, mirroring the
/// checkpoint loop's severity split.
fn run_debug_detect(args: &[String], home_override: Option<PathBuf>) -> Result<()> {
    let mahbot_home = resolve_home(home_override)?;
    let selected = match parse_db_flag(args, "detect")? {
        Some(name) => {
            validate_store_name(&name, false)?;
            vec![name]
        }
        // No `--db` → diagnose each PHYSICAL file once (core + logs), not
        // once per logical domain name (the 6 domain names all share the one
        // consolidated file).
        None => physical_store_list(&mahbot_home)
            .into_iter()
            .map(|(n, _)| n)
            .collect(),
    };
    let mut failures = 0usize;
    for name in &selected {
        let status = wal_guard::inspect_store_at(
            &turso_mod::store_db_path(&mahbot_home, name),
            wal_guard::StoreFds::none(),
        );
        let blocking = status.class.blocks_checkpoint();
        print_line(format_args!(
            "{}\t{}\twal_size={}\tblocking={}",
            name,
            status.class.label(),
            status.wal_size,
            blocking,
        ))?;
        // Exit 1 only for the checkpoint-blocking classes; warn-only classes
        // (oversized WAL, torn-pre index, unreadable .tshm) are reported but
        // are not failures — they mirror the checkpoint loop's severity split.
        if blocking {
            failures += 1;
        }
    }
    if failures > 0 {
        bail!("{failures} store(s) in a checkpoint-blocking coordination state — see above");
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
/// apply — no flock gate, no artifact pre-check, no torn-frame retry, no
/// post-open lock-drop verification. The engine opens without multiprocess
/// WAL (legacy read-only WAL path reads `db` + `-wal` directly — note it still
/// probes a present `.tshm` via `reject_live_multiprocess_wal_for_legacy_open`,
/// so a family carrying a `.tshm` is copied to an OS temp dir first, omitting
/// the coordination file; the no-touch guarantee comes from the copy, not the
/// opts). The whole open+query is panic-guarded: a damaged family yields a
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
    // could redirect the open to a live store, bypassing the flock gate and
    // live heuristics the `--db` path applies.
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
        let (io, db) = open_readonly(target, &db_path, turso_mod::family_database_opts())?;
        connect_execute(&io, &db, sql, &db_path)
    });
    drop(temp);
    match result {
        Ok(output) => Ok(output),
        // An engine panic takes precedence over the string-level torn-frame
        // classification (a panic payload could itself mention a torn frame).
        // The arm is gated by `is_engine_panic_error`, so `{e:#}` already
        // starts with the prefix — a single message, no double-named chain.
        Err(e) if is_engine_panic_error(&e) => {
            bail!("forensic family '{family_id}' could not be read — {e:#}")
        }
        // Static snapshot: a torn-frame/page read means the family's on-disk
        // data itself is corrupt, not a transient artifact. Never leak the raw
        // engine text (it is itself a torn-frame signature).
        Err(e) if is_torn_frame_error(&e) => {
            bail!(
                "forensic family '{family_id}' data is corrupt or internally inconsistent \
                 (the read produced a WAL/page error) — the family is a static snapshot; \
                 restore it from the original store if a healthy copy exists"
            )
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
    // forensic family (snapshot semantics: no flock gate, no live heuristics).
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

    // Flock-gate: refuse (exit code 2) rather than open a live store in a way
    // that could classify Exclusive and trigger repair. Runs before any turso
    // open (the open itself is what triggers the repair). The returned guard
    // (crash-recovery mode) is held for the whole run — dropping it would
    // release the instance lock before the queries finish.
    let flock_guard = flock_gate(
        &mahbot_home,
        &db_list.iter().map(|(l, _)| l.clone()).collect::<Vec<_>>(),
    )
    .await?;

    let mut failures = 0usize;
    for (label, file_path) in &db_list {
        if db_name == "all" {
            print_line(format_args!("=== {label} ==="))?;
        }

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

        let result = match sql {
            Some(sql) => query_one_store(file_path, sql, &mahbot_home, flock_guard.as_ref()).await,
            None => dump_one_store(file_path, label, &mahbot_home, flock_guard.as_ref()).await,
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

/// Run `open_fn` against one store with the full read-only safety stack,
/// retrying torn-frame failures with bounded backoff.
///
/// The artifact pre-check lives here (not only in the caller) so the
/// final-failure path can re-check it: a race may start the daemon publishing
/// frames mid-retry, or the path may be a snapshot copy with no `.tshm` at all
/// — in which case a persistent torn/page failure means the copied data is
/// corrupt, not that a live artifact exists. Shared by the query and
/// schema-dump paths so the two cannot drift apart.
///
/// `flock_guard` is the crash-recovery instance flock held by the caller
/// (`None` when the daemon holds the flock or no lock file exists); the
/// post-open byte-0 verification uses it to distinguish "we hold the flock"
/// (byte-0 free is expected) from "the daemon's flock is held but its byte-0
/// lock dropped" (the open classified Exclusive — abort).
///
/// `open_fn` re-opens the store on every attempt — a torn-frame failure can
/// leave a connection unusable, so each attempt is a fresh open+read.
async fn open_with_retry(
    file_path: &Path,
    root: &Path,
    flock_guard: Option<&File>,
    open_fn: impl Fn(&Path, &Path, Option<&File>) -> Result<()>,
) -> Result<()> {
    // Snapshot copies have no `-tshm`; they are static, so a torn-frame
    // failure cannot be transient there. Only live stores (which the daemon
    // keeps writing to) get the bounded retry that spans write windows.
    let is_live = turso_mod::store_sidecars(file_path).tshm.exists();

    // Pre-open artifact detection: `.tshm` advertises live frames while the
    // on-disk `-wal` is empty → the daemon's WAL fd is orphaned. Report the
    // explicit, actionable artifact error instead of letting the open surface
    // a raw torn-frame read. Live stores get the bounded wait-out first: the
    // daemon's TRUNCATE checkpoint truncates `-wal` to 32B before `.tshm`
    // max_frame resets — a transient window, not an artifact.
    if let Some(artifact_msg) = wait_out_artifact(file_path, is_live).await {
        bail!("{artifact_msg}");
    }

    // Open + read with bounded retry on torn-frame failures. Retrying
    // spans short write windows (e.g. the daemon mid-checkpoint); the
    // backoff total (~15s) is bounded so the CLI cannot hang for long.
    let mut attempt = 0usize;
    loop {
        match open_fn(file_path, root, flock_guard) {
            Ok(()) => return Ok(()),
            Err(e) => {
                // An engine panic is deterministic — never retry it, and
                // classify it as a panic before the string-level torn-frame
                // check (the payload could mention a torn frame). Gated by
                // `is_engine_panic_error`, so `{e:#}` already starts with the
                // prefix — a single message, no double-named chain.
                if is_engine_panic_error(&e) {
                    bail!(
                        "database '{}' could not be read — {e:#}",
                        file_path.display()
                    );
                }
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

/// Open one store read-only and run `sql` (pipe-delimited output) with the
/// full safety stack of [`open_with_retry`].
async fn query_one_store(
    file_path: &Path,
    sql: &str,
    root: &Path,
    flock_guard: Option<&File>,
) -> Result<()> {
    open_with_retry(file_path, root, flock_guard, |path, root_path, guard| {
        open_and_query_readonly(path, sql, root_path, guard)
    })
    .await
}

/// Open one store read-only and print its schema dump — a header line naming
/// the store, then one block per user table (`[table] <name>`, the table's
/// DDL, and its row count). Internal catalog artifacts are excluded (see
/// [`dump_schema`]). Same safety stack as [`query_one_store`].
async fn dump_one_store(
    file_path: &Path,
    label: &str,
    root: &Path,
    flock_guard: Option<&File>,
) -> Result<()> {
    open_with_retry(file_path, root, flock_guard, |path, root_path, guard| {
        open_and_dump_readonly(path, label, root_path, guard)
    })
    .await
}

/// Re-check the artifact condition with the bounded retry backoff — live
/// stores are waited out first: the daemon's TRUNCATE checkpoint truncates
/// `-wal` to 32B before `.tshm` max_frame resets, a transient window that
/// must not be refused as an artifact (asymmetric with the flock-gate's
/// wait-out-transients polling). Snapshot copies (static) bail immediately.
async fn wait_out_artifact(db_path: &Path, is_live: bool) -> Option<String> {
    let mut attempt = 0usize;
    while let Some(msg) = detect_live_artifact(db_path) {
        if is_live && attempt < MAX_OPEN_ATTEMPTS - 1 {
            sleep(Duration::from_secs(OPEN_RETRY_BACKOFF_SECS[attempt])).await;
            attempt += 1;
            continue;
        }
        return Some(msg);
    }
    None
}

/// Detect the live-instance WAL artifact before opening a store.
///
/// Returns a human-readable, actionable error message when the coordination
/// state blocks a safe read (orphaned/foreign/truncated WAL — the classes that
/// also block checkpoints). Returns `None` when the state is healthy or merely
/// warn-only (oversized WAL, torn-pre index).
fn detect_live_artifact(db_path: &Path) -> Option<String> {
    let status = wal_guard::inspect_store_at(db_path, wal_guard::StoreFds::none());
    if status.class.blocks_checkpoint() {
        Some(artifact_error_message(db_path, status.class.label()))
    } else {
        None
    }
}

/// Build the explicit artifact error message for a store.
fn artifact_error_message(db_path: &Path, class: &str) -> String {
    format!(
        "live instance artifact ({class}): cannot read '{}' safely.\n\
         The on-disk `-wal`/`.tshm` coordination state is inconsistent with the \
         daemon's live WAL (foreign standard-SQLite activity likely removed or \
         replaced the `-wal` files under the daemon). Query a snapshot \
         copy instead. Never delete or recreate `-wal`/`-shm`/`-tshm` files \
         while the daemon runs.",
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
    let tshm_path = turso_mod::store_sidecars(db_path).tshm;
    if tshm_path.exists() {
        format!(
            "database corruption/inconsistency: cannot read '{}' — the read \
             produced a WAL/page error even after {retries} retries. This is a \
             live store: query a snapshot copy instead (copy `db` + `-wal`, no \
             `-tshm`, while the daemon runs), or retry during a quiet window.",
            db_path.display(),
        )
    } else {
        format!(
            "database corruption/inconsistency: cannot read '{}' — the copied \
             main DB file (or its WAL) is corrupt or internally inconsistent. \
             No `-tshm` was present, so this is a snapshot copy, not a live \
             store. Re-copy the store files (a copy taken mid-checkpoint can be \
             inconsistent); if a fresh copy still fails, the store's on-disk \
             data itself is corrupt.",
            db_path.display()
        )
    }
}

/// Prefix of [`guard_panics`] errors — shared with the classifier so a wording
/// change updates both sides (string-matching matches the `is_torn_frame_error`
/// convention; a marker error type would be more robust but heavier).
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
/// ordinary query/open error). Checked before the string-level torn-frame
/// classification: a panic payload could itself mention a torn frame.
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
/// output. Output is buffered and returned only on success: a mid-query
/// failure (e.g. a torn-frame read) must not leave a partial column header on
/// stdout, which would be re-printed by the caller's retry loop.
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
/// See [`open_and_run_readonly`] for the shared open+verification stack.
fn open_and_query_readonly(
    file_path: &Path,
    sql: &str,
    root: &Path,
    flock_guard: Option<&File>,
) -> Result<()> {
    open_and_run_readonly(file_path, root, flock_guard, |io, db, path| {
        connect_execute_print(io, db, sql, path)
    })
}

/// Open a store read-only, run the post-open lock-drop verification, then run
/// `runner` against the connection. Used by the query and schema-dump paths so
/// both share the same open+verification stack.
///
/// Reuses an existing `.tshm` when present so live stores are read through the
/// daemon's WAL coordination. The whole open+run is panic-guarded — a damaged
/// store must yield a clean error, never a CLI crash.
fn open_and_run_readonly<T>(
    file_path: &Path,
    root: &Path,
    flock_guard: Option<&File>,
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

        // Post-open verification: when the caller does NOT hold the instance
        // flock (a live daemon does), a free byte-0 right after the open means
        // the open classified Exclusive — the lock-drop window between the
        // gate and this open — and the daemon's repair could run concurrently.
        // Abort. Unix-only: the fcntl byte-0 probe has no Windows equivalent
        // (the gate is skipped there too).
        #[cfg(unix)]
        if flock_guard.is_none() {
            let tshm = turso_mod::store_sidecars(file_path).tshm;
            if tshm.exists() && probe_tshm_byte0_pid(&tshm).is_none() {
                let lock_path = crate::util::lock::lock_file_path(root);
                if lock_path.exists() && !probe_flock_free(&lock_path) {
                    bail!(
                        "live daemon lock-drop detected after open on '{}' — the open \
                         classified Exclusive, aborting (retry later or query a snapshot copy)",
                        file_path.display()
                    );
                }
            }
        }

        runner(&io, &db, file_path)
    })
}

/// [`open_and_run_readonly`] runner for the schema dump: build the whole
/// store's dump text (buffered — a mid-dump failure never leaks a partial
/// dump to stdout, matching [`connect_execute`]'s all-or-nothing output),
/// then write it once.
fn open_and_dump_readonly(
    file_path: &Path,
    label: &str,
    root: &Path,
    flock_guard: Option<&File>,
) -> Result<()> {
    open_and_run_readonly(file_path, root, flock_guard, |io, db, path| {
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
/// dump queries share the same robustness (a torn-frame failure propagates and
/// the caller re-runs the whole dump via [`open_with_retry`]).
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

/// True when the error chain looks like a torn-frame / WAL-inconsistency read.
///
/// Matches on the stringified error because engine variants are stringified at
/// the error boundary; the signatures mirror turso_core 0.7.2's
/// `CompletionError::ShortReadWalFrame` / `ShortRead` / `ChecksumMismatch` /
/// `WalFramePageMismatch` display texts.
fn is_torn_frame_error(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}").to_lowercase();
    TORN_FRAME_SIGNATURES.iter().any(|sig| msg.contains(sig))
}

/// The physical database files for the debug CLI, each opened once.
///
/// After consolidation the 6 domain stores share ONE file, so `--db all` and
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
        turso::core::Value::Blob(b) => crate::util::hex_string(b),
    }
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
    eprintln!("Usage: mahbot debug --db <name> [\"SQL query\"]");
    eprintln!("       mahbot debug detect [--db <name>]");
    eprintln!("       mahbot debug families [--db <name>]");
    eprintln!("       mahbot debug --family <id> \"SQL query\"");
    let mut names = turso_mod::store_names();
    names.push(turso_mod::CONSOLIDATED_DB_NAME);
    let names = names.join(" | ");
    eprintln!("  -h, --help  print this help and exit 0");
    eprintln!("  --db <name> {names} | all");
    eprintln!("              with a SQL argument: read-only query, pipe-delimited output");
    eprintln!("              without one: schema dump — one block per user table");
    eprintln!("              (`[table] <name>` / DDL / `rows: N`); `all` dumps every live");
    eprintln!("              database in per-store sections (per-store errors; exit 1 if");
    eprintln!("              any store failed)");
    eprintln!("  SQL query   read-only SQL, quoted as a single argument");
    eprintln!("  detect      classify coordination state without opening stores");
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
        assert!(!tokens.contains(&String::new()));
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
        // A user table named "torn_*" must not be misclassified as corruption
        // (the bare "torn" signature was removed for exactly this reason).
        let unrelated = anyhow!("SQL error: no such table: torn_table");
        assert!(!is_torn_frame_error(&unrelated));
    }

    #[test]
    fn artifact_error_message_is_actionable() {
        let msg = artifact_error_message(Path::new("/tmp/x/board.db"), "orphaned-wal");
        assert!(msg.contains("live instance artifact"));
        assert!(msg.contains("snapshot"));
        assert!(msg.contains("Never delete or recreate"));
    }

    /// End-to-end: `run_debug_with_args` opens a real (temporary) store
    /// read-only through the `turso::core` path and runs a query against it.
    /// Serialized: `guard_panics` swaps the process-global panic hook, so
    /// guard windows must not overlap other debug tests.
    #[tokio::test]
    #[serial_test::serial(family, tshm_counter)]
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
    ///
    /// This test is `#[ignore]` by default because the artifact open burns the full 15 s torn-frame retry backoff before reporting. Run it
    /// explicitly with:
    ///
    /// ```sh
    /// cargo test run_debug_reports_artifact_instead_of_torn_frame -- --ignored --nocapture
    /// ```
    #[ignore = "burns the full 15 s torn-frame retry backoff on a crafted artifact state; runs only when explicitly invoked"]
    #[tokio::test]
    /// Serialized under `tshm_counter` too: this `--db` artifact path performs
    /// path-based coordination reads that increment `TSHM_OPEN_CLOSE_COUNT`, and
    /// that side effect persists under `--ignored` — so the key must not be
    /// dropped later as seemingly-dead.
    #[serial_test::serial(family, tshm_counter)]
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
    ///
    /// This test is `#[ignore]` by default because the artifact open burns the full 15 s torn-frame retry backoff before reporting. Run it
    /// explicitly with:
    ///
    /// ```sh
    /// cargo test run_debug_all_reports_failure_summary -- --ignored --nocapture
    /// ```
    #[ignore = "burns the full 15 s torn-frame retry backoff on a crafted artifact state; runs only when explicitly invoked"]
    #[tokio::test]
    /// Serialized under `tshm_counter` too: this `--db all` artifact path
    /// performs path-based coordination reads that increment
    /// `TSHM_OPEN_CLOSE_COUNT`, and that side effect persists under
    /// `--ignored` — so the key must not be dropped later as seemingly-dead.
    #[serial_test::serial(family, tshm_counter)]
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
        let dump = open_and_run_readonly(&db_path, dir.path(), None, |io, db, path| {
            dump_schema(io, db, path, "logs")
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
    }

    /// `mahbot debug --db <name>` without a SQL argument dumps the store schema
    /// and exits 0 — and, like the query path, creates no extra `-tshm` files.
    /// Serialized: `guard_panics` swaps the process-global panic hook, so
    /// guard windows must not overlap other debug tests.
    #[tokio::test]
    #[serial_test::serial(family, tshm_counter)]
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
        // Exactly one tshm file (created by the store open itself) — the
        // read-only CLI open must not create another.
        let tshm_count = names.iter().filter(|n| n.ends_with("-tshm")).count();
        assert_eq!(tshm_count, 1, "no new -tshm file may be created: {names:?}");
    }

    /// `mahbot debug --db all` without a SQL argument dumps every present
    /// store and reports missing stores per-store with a failure summary
    /// (exit 1) — matching the query verb's `--db all` failure semantics.
    /// Serialized: `guard_panics` swaps the process-global panic hook, so
    /// guard windows must not overlap other debug tests.
    #[tokio::test]
    #[serial_test::serial(family, tshm_counter)]
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
        let moved = move_family_aside(&db_dir, &db_dir.join("logs.db"), fam);
        // Corrupt the family's tshm (garbage — the record of the corruption).
        std::fs::write(db_dir.join(format!("{fam}-tshm")), vec![0xAB; 64]).unwrap();

        let before: Vec<((u64, std::time::SystemTime), String)> = moved
            .iter()
            .map(|name| (file_state(&db_dir.join(name)), name.clone()))
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
            moved.len(),
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
        assert!(!is_torn_frame_error(&err), "panic must not be torn-frame");
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

    // ── flock-gate tests ───────────────────────────────────────────────

    fn write_tshm_only(root: &Path, name: &str) {
        let db_path = turso_mod::store_db_path(root, name);
        std::fs::create_dir_all(db_path.parent().unwrap()).unwrap();
        let mut bytes = vec![0u8; 116];
        bytes[0..8].copy_from_slice(crate::db::wal_guard::TSHM_MAGIC.as_slice());
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&64u32.to_le_bytes());
        std::fs::write(turso_mod::store_sidecars(&db_path).tshm, bytes).unwrap();
    }

    /// Create `mahbot.lock` WITHOUT taking the flock (the gate's crash-recovery
    /// and handoff states require the file to exist).
    fn touch_lock(lock_path: &Path) {
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)
            .unwrap();
    }

    fn hold_flock(lock_path: &Path) -> std::fs::File {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)
            .unwrap();
        assert!(
            crate::util::lock::try_flock(&file).unwrap(),
            "test must hold the flock"
        );
        file
    }

    /// Spawn a perl child that holds `.tshm` byte-0 via fcntl F_SETLK — a
    /// real cross-process lock (macOS F_GETLK deliberately ignores locks held
    /// by the calling process itself). The child lives ~30s; callers kill it.
    #[cfg(unix)]
    fn hold_byte0_perl(tshm: &Path) -> std::process::Child {
        let perl = format!(
            "use Fcntl qw(F_SETLK F_WRLCK); open(my $f, \"+<\", $ARGV[0]) or die $!; \
             my $buf = pack(\"q< q< l< s< s<\", 0, 0, $$, {}, 0); \
             fcntl($f, F_SETLK, $buf) or die $!; sleep 30;",
            libc::F_WRLCK,
        );
        std::process::Command::new("perl")
            .args(["-e", &perl, tshm.to_str().unwrap()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn perl byte-0 holder")
    }

    /// The gate is a no-op when no `mahbot.lock` exists (no daemon ever
    /// started) — the snapshot-query and bare-store workflows stay unblocked.
    /// Serialized: the gate tests manipulate process-wide flock/fcntl state
    /// and spawn perl lock-holder children, so they must not run concurrently.
    ///
    /// This test is `#[ignore]` by default because it exercises real cross-process fcntl locks (spawned perl lock-holder children) with multi-second waits. Run it
    /// explicitly with:
    ///
    /// ```sh
    /// cargo test flock_gate_passes_without_lock_file -- --ignored --nocapture
    /// ```
    #[ignore = "exercises real cross-process fcntl locks with multi-second waits; runs only when explicitly invoked"]
    #[tokio::test]
    #[serial_test::serial(flock_gate)]
    async fn flock_gate_passes_without_lock_file() {
        let dir = tempfile::TempDir::new().unwrap();
        write_tshm_only(dir.path(), "board");
        assert!(
            flock_gate_with_timeout(dir.path(), &["board".into()], Duration::from_secs(1))
                .await
                .is_ok()
        );
    }

    /// Normal live-daemon state (flock held + byte-0 held) proceeds — the
    /// CLI's turso open sees MultiProcess and reads safely. The byte-0 lock
    /// must come from a real child process: on macOS `F_GETLK` deliberately
    /// ignores locks held by the calling process itself, so an in-process
    /// lock would read as "free" and the gate would (correctly) wait.
    ///
    /// This test is `#[ignore]` by default because it exercises real cross-process fcntl locks (spawned perl lock-holder children) with multi-second waits. Run it
    /// explicitly with:
    ///
    /// ```sh
    /// cargo test flock_gate_proceeds_when_daemon_alive -- --ignored --nocapture
    /// ```
    #[ignore = "exercises real cross-process fcntl locks with multi-second waits; runs only when explicitly invoked"]
    #[tokio::test]
    #[cfg(unix)]
    #[serial_test::serial(flock_gate)]
    async fn flock_gate_proceeds_when_daemon_alive() {
        let dir = tempfile::TempDir::new().unwrap();
        write_tshm_only(dir.path(), "board");
        let _flock = hold_flock(&crate::util::lock::lock_file_path(dir.path()));
        let tshm = turso_mod::store_sidecars(&turso_mod::store_db_path(dir.path(), "board")).tshm;
        let mut child = hold_byte0_perl(&tshm);
        // Wait for the child to place the lock before probing.
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(
            probe_tshm_byte0_pid(&tshm).is_some(),
            "child must hold the byte-0 lock"
        );
        let guard = flock_gate_with_timeout(dir.path(), &["board".into()], Duration::from_secs(1))
            .await
            .expect("flock held + byte-0 held must proceed");
        assert!(
            guard.is_none(),
            "a live daemon holds the flock — the gate must not take it"
        );
        let _ = child.kill();
        let _ = child.wait();
    }

    /// Crash-recovery: after a busy→free transition (the daemon's flock was
    /// observed held, then released), two consecutive free observations make
    /// the gate TAKE the flock and return the guard, which actually holds it
    /// (a second probe sees it busy).
    ///
    /// This test is `#[ignore]` by default because it exercises real cross-process fcntl locks (spawned perl lock-holder children) with multi-second waits. Run it
    /// explicitly with:
    ///
    /// ```sh
    /// cargo test flock_gate_takes_flock_in_crash_recovery -- --ignored --nocapture
    /// ```
    #[ignore = "exercises real cross-process fcntl locks with multi-second waits; runs only when explicitly invoked"]
    #[tokio::test]
    #[serial_test::serial(flock_gate)]
    async fn flock_gate_takes_flock_in_crash_recovery() {
        let dir = tempfile::TempDir::new().unwrap();
        write_tshm_only(dir.path(), "board");
        let lock_path = crate::util::lock::lock_file_path(dir.path());
        touch_lock(&lock_path);
        // The gate must first observe the flock busy (the daemon's release
        // precedes its exit), then see two consecutive free observations.
        let flock = hold_flock(&lock_path);
        let dir_path = dir.path().to_path_buf();
        let names = vec!["board".to_string()];
        let gate = tokio::spawn(async move {
            flock_gate_with_timeout(&dir_path, &names, Duration::from_secs(4)).await
        });
        // Let the gate's first observation land while the flock is held.
        tokio::time::sleep(Duration::from_millis(500)).await;
        drop(flock); // busy → free
        let guard = gate
            .await
            .expect("gate task must not panic")
            .expect("flock free + byte-0 free after busy must proceed");
        assert!(
            !probe_flock_free(&lock_path),
            "the returned guard must hold the flock (probe sees it busy)"
        );
        drop(guard);
        // The kernel releases the flock when the guard's fd closes; poll
        // briefly since close→flock-visible-release can lag under load on
        // macOS (the assertion's semantics are unchanged — the flock must
        // become free, not stay busy).
        let release_deadline = tokio::time::Instant::now() + Duration::from_millis(500);
        while !probe_flock_free(&lock_path) {
            assert!(
                tokio::time::Instant::now() < release_deadline,
                "dropping the guard must release the flock"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Self-update handoff: flock free + byte-0 held (old process finishing).
    /// The gate must wait (time out into a refusal) and must never take the
    /// flock — a transient acquire would kill a fail-fast incoming daemon.
    ///
    /// This test is `#[ignore]` by default because it exercises real cross-process fcntl locks (spawned perl lock-holder children) with multi-second waits. Run it
    /// explicitly with:
    ///
    /// ```sh
    /// cargo test flock_gate_never_takes_flock_during_handoff -- --ignored --nocapture
    /// ```
    #[ignore = "exercises real cross-process fcntl locks with multi-second waits; runs only when explicitly invoked"]
    #[tokio::test]
    #[cfg(unix)]
    #[serial_test::serial(flock_gate)]
    async fn flock_gate_never_takes_flock_during_handoff() {
        let dir = tempfile::TempDir::new().unwrap();
        write_tshm_only(dir.path(), "board");
        let lock_path = crate::util::lock::lock_file_path(dir.path());
        touch_lock(&lock_path);
        let tshm = turso_mod::store_sidecars(&turso_mod::store_db_path(dir.path(), "board")).tshm;
        let mut child = hold_byte0_perl(&tshm);
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(
            probe_tshm_byte0_pid(&tshm).is_some(),
            "child must hold byte-0"
        );
        let err = flock_gate_with_timeout(dir.path(), &["board".into()], Duration::from_secs(1))
            .await
            .expect_err("flock free + byte-0 held must wait and time out");
        assert!(
            err.downcast_ref::<GateRefusal>().is_some(),
            "must be a GateRefusal (exit code 2), got: {err:#}"
        );
        assert!(
            probe_flock_free(&lock_path),
            "the gate must never take the flock during the handoff"
        );
        let _ = child.kill();
        let _ = child.wait();
    }

    /// Self-update handoff completed: the byte-0 holder PID changes (the new
    /// daemon opened stores). The gate must exit the handoff and proceed as a
    /// MultiProcess read instead of timing out — the exit is detected via the
    /// PID, never by re-probing the flock.
    ///
    /// Two stores make the swap deterministic: the gate's first observation
    /// records `[Some(old), None]` (old holds `board`, `logs` unheld),
    /// and the new child acquires `logs` mid-run — uncontended (the
    /// old child locks a different file), so the new pid lands with wide
    /// margin before the gate's second observation (~1s) and the PID change
    /// fires. Under the consolidated layout `board` is the domain file
    /// (`core.db`) and `logs` is the one separate physical store, so the two
    /// names resolve to two distinct sidecar files.
    ///
    /// This test is `#[ignore]` by default because it exercises real cross-process fcntl locks (spawned perl lock-holder children) with multi-second waits. Run it
    /// explicitly with:
    ///
    /// ```sh
    /// cargo test flock_gate_proceeds_after_handoff_pid_change -- --ignored --nocapture
    /// ```
    #[ignore = "exercises real cross-process fcntl locks with multi-second waits; runs only when explicitly invoked"]
    #[tokio::test]
    #[cfg(unix)]
    #[serial_test::serial(flock_gate)]
    async fn flock_gate_proceeds_after_handoff_pid_change() {
        let dir = tempfile::TempDir::new().unwrap();
        write_tshm_only(dir.path(), "board");
        write_tshm_only(dir.path(), "logs");
        let lock_path = crate::util::lock::lock_file_path(dir.path());
        touch_lock(&lock_path);
        let tshm_a = turso_mod::store_sidecars(&turso_mod::store_db_path(dir.path(), "board")).tshm;
        let tshm_b = turso_mod::store_sidecars(&turso_mod::store_db_path(dir.path(), "logs")).tshm;
        let mut old = hold_byte0_perl(&tshm_a);
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(
            probe_tshm_byte0_pid(&tshm_a).is_some() && probe_tshm_byte0_pid(&tshm_b).is_none(),
            "old daemon holds board; logs starts unheld"
        );
        let dir_path = dir.path().to_path_buf();
        let names = vec!["board".to_string(), "logs".to_string()];
        // The first observation runs immediately at spawn (old holds board)
        // and establishes the handoff as [Some(old), None]. Swap children
        // early: logs is uncontended, so the new pid lands with wide
        // margin before the gate's second observation (~1s after spawn).
        let gate = tokio::spawn(async move {
            flock_gate_with_timeout(&dir_path, &names, Duration::from_secs(5)).await
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        let _ = old.kill();
        let _ = old.wait();
        let mut new = hold_byte0_perl(&tshm_b);
        // Poll (bounded) until the new child holds byte-0 — assert well
        // before the gate's second observation (~1s after start) so the
        // handoff never observes an all-free gap.
        let deadline = tokio::time::Instant::now() + Duration::from_millis(600);
        while probe_tshm_byte0_pid(&tshm_b).is_none() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "new daemon must hold byte-0 before the gate's next observation"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let guard = gate
            .await
            .expect("gate task must not panic")
            .expect("pid change must exit the handoff");
        assert!(
            guard.is_none(),
            "the new daemon holds the flock — the gate must not take it"
        );
        assert!(
            probe_flock_free(&lock_path),
            "the flock must never be taken"
        );
        let _ = new.kill();
        let _ = new.wait();
    }

    /// The macOS lock-drop state (flock held by a live daemon whose fcntl
    /// byte-0 lock is gone) must NOT proceed — opening would classify
    /// Exclusive and trigger repair. Times out into a [`GateRefusal`].
    ///
    /// This test is `#[ignore]` by default because it exercises real cross-process fcntl locks (spawned perl lock-holder children) with multi-second waits. Run it
    /// explicitly with:
    ///
    /// ```sh
    /// cargo test flock_gate_refuses_in_lock_drop_state -- --ignored --nocapture
    /// ```
    #[ignore = "exercises real cross-process fcntl locks with multi-second waits; runs only when explicitly invoked"]
    #[tokio::test]
    #[serial_test::serial(flock_gate)]
    async fn flock_gate_refuses_in_lock_drop_state() {
        let dir = tempfile::TempDir::new().unwrap();
        write_tshm_only(dir.path(), "board");
        let _flock = hold_flock(&crate::util::lock::lock_file_path(dir.path()));
        let err = flock_gate_with_timeout(dir.path(), &["board".into()], Duration::from_secs(1))
            .await
            .expect_err("flock held + byte-0 free must refuse");
        assert!(
            err.downcast_ref::<GateRefusal>().is_some(),
            "must be a GateRefusal (exit code 2), got: {err:#}"
        );
    }

    /// `debug detect` classifies synthetic states without opening the store
    /// through turso. It still performs path-based coordination reads (via
    /// `run_debug_detect`), so it must be serialized under `tshm_counter`
    /// alongside the counter-asserting test.
    #[test]
    #[serial_test::serial(tshm_counter)]
    fn debug_detect_reports_non_healthy_state() {
        let dir = tempfile::TempDir::new().unwrap();
        write_tshm_only(dir.path(), "board");
        // No max_frame in the fixture → healthy.
        let args = vec![
            "mahbot".to_string(),
            "debug".to_string(),
            "detect".to_string(),
            "--db".to_string(),
            "board".to_string(),
        ];
        assert!(run_debug_detect(&args, Some(dir.path().to_path_buf())).is_ok());

        // max_frame > 0 with no WAL → orphaned → non-zero exit.
        let tshm_path =
            turso_mod::store_sidecars(&turso_mod::store_db_path(dir.path(), "board")).tshm;
        let mut bytes = std::fs::read(&tshm_path).unwrap();
        bytes[56..64].copy_from_slice(&5u64.to_le_bytes());
        std::fs::write(&tshm_path, bytes).unwrap();
        let err = run_debug_detect(&args, Some(dir.path().to_path_buf()))
            .expect_err("orphaned state must fail detect");
        assert!(format!("{err:#}").contains("blocking"), "got: {err:#}");
    }
}
