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

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use tokio::time::sleep;

use crate::turso as turso_mod;
use crate::wal_guard;

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
    use std::fs::OpenOptions;
    let Ok(file) = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)
    else {
        return false;
    };
    crate::lock_utils::try_flock(&file).unwrap_or(false)
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
    match crate::lock_utils::try_flock(&file) {
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
    let lock_path = crate::lock_utils::lock_file_path(root);
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

/// `mahbot debug detect [--db <name>]` — run the full coordination-state
/// predicate over the stores without opening them. Prints one line per store
/// (`name\tclass\twal_size=N\tblocking=B`). Exit 0 when all stores are healthy
/// or warn-only; exit 1 (via `Err`) only when a store is in a
/// checkpoint-blocking class — warn-only classes (oversized WAL, torn-pre
/// index, unreadable `.tshm`) are reported but are not failures, mirroring the
/// checkpoint loop's severity split.
fn run_debug_detect(args: &[String], home_override: Option<PathBuf>) -> Result<()> {
    let mahbot_home = match home_override {
        Some(home) => home,
        None => crate::config::default_config_dir()?,
    };
    let selected = match args.get(3).map(String::as_str) {
        Some("--db") => match args.get(4) {
            Some(name) => vec![name.clone()],
            None => bail!("expected: mahbot debug detect --db <name>"),
        },
        Some(other) => bail!("invalid detect argument '{other}'"),
        None => turso_mod::store_names()
            .into_iter()
            .map(String::from)
            .collect(),
    };
    let mut failures = 0usize;
    for name in &selected {
        if !turso_mod::store_names().contains(&name.as_str()) {
            bail!("invalid database name '{name}'");
        }
        let status = wal_guard::inspect_store_at(
            &turso_mod::store_db_path(&mahbot_home, name),
            wal_guard::StoreFds::none(),
        );
        let blocking = status.class.blocks_checkpoint();
        println!(
            "{}\t{}\twal_size={}\tblocking={}",
            name,
            status.class.label(),
            status.wal_size,
            blocking,
        );
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

    // `mahbot debug detect [--db <name>]` — full predicate, no opens, no gate.
    if args.get(2).is_some_and(|a| a == "detect") {
        return run_debug_detect(&args, home_override);
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
            println!("=== {label} ===");
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

        match query_one_store(file_path, sql, &mahbot_home, flock_guard.as_ref()).await {
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
///
/// `flock_guard` is the crash-recovery instance flock held by the caller
/// (`None` when the daemon holds the flock or no lock file exists); the
/// post-open byte-0 verification uses it to distinguish "we hold the flock"
/// (byte-0 free is expected) from "the daemon's flock is held but its byte-0
/// lock dropped" (the open classified Exclusive — abort).
async fn query_one_store(
    file_path: &Path,
    sql: &str,
    root: &Path,
    flock_guard: Option<&File>,
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

    // Open + query with bounded retry on torn-frame failures. Retrying
    // spans short write windows (e.g. the daemon mid-checkpoint); the
    // backoff total (~15s) is bounded so the CLI cannot hang for long.
    let mut attempt = 0usize;
    loop {
        match open_and_query_readonly(file_path, sql, root, flock_guard) {
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
fn open_and_query_readonly(
    file_path: &Path,
    sql: &str,
    root: &Path,
    flock_guard: Option<&File>,
) -> Result<()> {
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

    // Post-open verification: when the caller does NOT hold the instance
    // flock (a live daemon does), a free byte-0 right after the open means the
    // open classified Exclusive — the lock-drop window between the gate and
    // this open — and the daemon's repair could run concurrently. Abort.
    // Unix-only: the fcntl byte-0 probe has no Windows equivalent (the gate
    // is skipped there too).
    #[cfg(unix)]
    if flock_guard.is_none() {
        let tshm = turso_mod::store_sidecars(file_path).tshm;
        if tshm.exists() && probe_tshm_byte0_pid(&tshm).is_none() {
            let lock_path = crate::lock_utils::lock_file_path(root);
            if lock_path.exists() && !probe_flock_free(&lock_path) {
                bail!(
                    "live daemon lock-drop detected after open on '{}' — the open \
                     classified Exclusive, aborting (retry later or query a snapshot copy)",
                    file_path.display()
                );
            }
        }
    }

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
/// the error boundary; the signatures mirror turso_core 0.7.2's
/// `CompletionError::ShortReadWalFrame` / `ShortRead` / `ChecksumMismatch` /
/// `WalFramePageMismatch` display texts.
fn is_torn_frame_error(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}").to_lowercase();
    TORN_FRAME_SIGNATURES.iter().any(|sig| msg.contains(sig))
}

/// Map a `--db` argument to a list of `(label, absolute db path)` pairs.
fn resolve_db_list(name: &str, root: &Path) -> Result<Vec<(String, PathBuf)>> {
    let names = turso_mod::store_names();
    if name == "all" {
        Ok(names
            .iter()
            .map(|n| (n.to_string(), turso_mod::store_db_path(root, n)))
            .collect())
    } else if names.contains(&name) {
        Ok(vec![(
            name.to_string(),
            turso_mod::store_db_path(root, name),
        )])
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
    eprintln!("Usage: mahbot debug --db <name> \"SQL query\"");
    eprintln!("       mahbot debug detect [--db <name>]");
    let names = turso_mod::store_names().join(" | ");
    eprintln!("  --db <name>  {names} | all");
    eprintln!("  SQL query    read-only SQL, quoted as a single argument");
    eprintln!("  detect       classify coordination state without opening stores");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  mahbot debug --db board \"SELECT phase, COUNT(*) FROM tickets GROUP BY phase\"");
    eprintln!("  mahbot debug --db all \"SELECT name FROM sqlite_master WHERE type='table'\"");
    eprintln!("  mahbot debug detect");
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
        let msg = artifact_error_message(Path::new("/tmp/x/board.db"), "orphaned-wal");
        assert!(msg.contains("live instance artifact"));
        assert!(msg.contains("snapshot"));
        assert!(msg.contains("Never delete or recreate"));
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

    // ── flock-gate tests ───────────────────────────────────────────────

    fn write_tshm_only(root: &Path, name: &str) {
        let db_path = turso_mod::store_db_path(root, name);
        std::fs::create_dir_all(db_path.parent().unwrap()).unwrap();
        let mut bytes = vec![0u8; 116];
        bytes[0..8].copy_from_slice(crate::wal_guard::TSHM_MAGIC.as_slice());
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
            crate::lock_utils::try_flock(&file).unwrap(),
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
    #[tokio::test]
    #[cfg(unix)]
    #[serial_test::serial(flock_gate)]
    async fn flock_gate_proceeds_when_daemon_alive() {
        let dir = tempfile::TempDir::new().unwrap();
        write_tshm_only(dir.path(), "board");
        let _flock = hold_flock(&crate::lock_utils::lock_file_path(dir.path()));
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
    #[tokio::test]
    #[serial_test::serial(flock_gate)]
    async fn flock_gate_takes_flock_in_crash_recovery() {
        let dir = tempfile::TempDir::new().unwrap();
        write_tshm_only(dir.path(), "board");
        let lock_path = crate::lock_utils::lock_file_path(dir.path());
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
    #[tokio::test]
    #[cfg(unix)]
    #[serial_test::serial(flock_gate)]
    async fn flock_gate_never_takes_flock_during_handoff() {
        let dir = tempfile::TempDir::new().unwrap();
        write_tshm_only(dir.path(), "board");
        let lock_path = crate::lock_utils::lock_file_path(dir.path());
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
    /// records `[Some(old), None]` (old holds `board`, `chat_history` unheld),
    /// and the new child acquires `chat_history` mid-run — uncontended (the
    /// old child locks a different file), so the new pid lands with wide
    /// margin before the gate's second observation (~1s) and the PID change
    /// fires.
    #[tokio::test]
    #[cfg(unix)]
    #[serial_test::serial(flock_gate)]
    async fn flock_gate_proceeds_after_handoff_pid_change() {
        let dir = tempfile::TempDir::new().unwrap();
        write_tshm_only(dir.path(), "board");
        write_tshm_only(dir.path(), "chat_history");
        let lock_path = crate::lock_utils::lock_file_path(dir.path());
        touch_lock(&lock_path);
        let tshm_a = turso_mod::store_sidecars(&turso_mod::store_db_path(dir.path(), "board")).tshm;
        let tshm_b =
            turso_mod::store_sidecars(&turso_mod::store_db_path(dir.path(), "chat_history")).tshm;
        let mut old = hold_byte0_perl(&tshm_a);
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(
            probe_tshm_byte0_pid(&tshm_a).is_some() && probe_tshm_byte0_pid(&tshm_b).is_none(),
            "old daemon holds board; chat_history starts unheld"
        );
        let dir_path = dir.path().to_path_buf();
        let names = vec!["board".to_string(), "chat_history".to_string()];
        // The first observation runs immediately at spawn (old holds board)
        // and establishes the handoff as [Some(old), None]. Swap children
        // early: chat_history is uncontended, so the new pid lands with wide
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
    #[tokio::test]
    #[serial_test::serial(flock_gate)]
    async fn flock_gate_refuses_in_lock_drop_state() {
        let dir = tempfile::TempDir::new().unwrap();
        write_tshm_only(dir.path(), "board");
        let _flock = hold_flock(&crate::lock_utils::lock_file_path(dir.path()));
        let err = flock_gate_with_timeout(dir.path(), &["board".into()], Duration::from_secs(1))
            .await
            .expect_err("flock held + byte-0 free must refuse");
        assert!(
            err.downcast_ref::<GateRefusal>().is_some(),
            "must be a GateRefusal (exit code 2), got: {err:#}"
        );
    }

    /// `debug detect` classifies synthetic states without opening anything.
    #[test]
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
