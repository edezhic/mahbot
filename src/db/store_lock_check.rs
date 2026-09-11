//! Cross-process verification of the store locks: that the running service holds
//! them, and that it refuses to run without them.
//!
//! The engine's whole-file record locks belong to the process that takes them,
//! so no process can observe its own — only a foreign client can (wal_guard's
//! lock rule). [`verify`] therefore boots the real stores, drives the real paths
//! that used to lose the lock (the periodic inspection round and the
//! engine-produced pre-reindex snapshot), and then has a stock `sqlite3` read and
//! write both live stores' main files: both must be refused as locked, and the
//! store directory's metadata fingerprint must be unchanged. That refusal is the
//! whole-file lock on the MAIN file, which a client's own byte-range lock
//! conflicts with — a lock left only on a `-wal` would not show up this way.
//!
//! [`tests::a_store_held_by_another_process_refuses_the_boot`] covers the other
//! half: while a foreign process holds a store's lock, the daemon's bring-up must
//! refuse — recorded on the boot-diagnostic channel — rather than open what it
//! cannot lock.
//!
//! [`tests::an_unusable_store_file_refuses_the_boot`] and
//! [`tests::both_unusable_stores_are_named_in_the_durable_record`] cover the
//! sibling rule: a store file that exists but is not usable refuses the boot
//! before either store is opened, leaving both stores untouched and recording a
//! refusal that names every unusable store.
//!
//! [`tests::an_unreadable_store_file_refuses_the_boot_as_an_environment_cause`]
//! adds the environment-caused half of that rule with a real permission failure:
//! the refusal is recorded as an environment condition, never as damage (skipped
//! when the test runs as root, where mode 000 denies nothing).
//!
//! Verified on macOS, runnable on Linux — the two platforms the project is tested
//! on (the module is Unix-only, so elsewhere the property is left unverified
//! rather than checked). Driven by this module's own test drivers, which
//! re-execute this test binary as the second process.
//!
//! The foreign probes need the stock `sqlite3` CLI and are skipped without it,
//! which the evidence line then says in place of their results: the project's
//! test run must never depend on a tool that may be absent from the machine
//! running it. The boot bring-up, the paths above and the fingerprint run either
//! way — only the foreign-client observation is left to a machine that has the
//! CLI (see [`sqlite3_available`]).

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, ensure};

/// The two foreign-client probes: a read (needs a SHARED lock to load the
/// schema) and a write (needs a RESERVED lock).
const PROBES: [(&str, &str); 2] = [
    ("read", "SELECT count(*) FROM sqlite_schema;"),
    (
        "write",
        "CREATE TABLE lock_probe (x); INSERT INTO lock_probe VALUES (1);",
    ),
];

/// Boot the real stores, exercise the real inspection/snapshot paths, then
/// prove from a foreign process that every store is locked. Returns the
/// evidence lines on success.
async fn verify() -> Result<String> {
    crate::config::load_or_init().await?;
    let root = crate::config::CONFIG
        .try_storage_root()
        .context("verify: config::load_or_init() did not set the storage root")?;
    // This check boots real stores — it migrates them and may quarantine family
    // files — so it must only ever be pointed at a fresh hermetic root. The `db`
    // directory is created by the first store open and by nothing else, so its
    // existence is what makes a root someone's install: refuse it rather than
    // touch it (path-based stats, like everything else here).
    let store_dir = root.join("db");
    ensure!(
        !store_dir.exists(),
        "refusing to run against an existing store root ({}) — this check writes to the \
         root it is given and must be pointed at a fresh hermetic $HOME",
        store_dir.display(),
    );
    // The daemon's own store bring-up, driven verbatim (see `boot::open_stores`)
    // so a future boot-order change cannot silently weaken this check.
    crate::boot::open_stores().await?;

    // The real snapshot path, with the live consolidated connection. It is here
    // to exercise the path (the probe below is what catches a regression): a
    // reintroduced in-process open would still let this succeed, it would just
    // cost the lock.
    let conn = super::DOMAIN_CONN
        .get()
        .context("the consolidated store connection is not initialized")?;
    let snapshot = super::snapshot_store_via_engine(
        conn,
        &super::store_db_path(&root, super::CONSOLIDATED_DB_NAME),
    )
    .await?;

    // Let the boot's background work settle: the log writer flushes on a 500 ms
    // timer and the CDC drainer polls, so the probes then read a store nothing
    // is writing to.
    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;

    // The real inspection round — the other path that historically dropped the
    // lock — runs last on purpose: it folds the settled WAL into the main files
    // immediately before the fingerprint below, so the engine's own
    // auto-checkpoint cannot move a store inside the probe window. Repeated,
    // because the rounds are not identical: the first folds a settled WAL and
    // the later ones inspect the already-folded state.
    for _ in 0..3 {
        crate::db::checkpoint::periodic_checkpoint_and_verify().await;
    }

    let before = dir_fingerprint(&store_dir)?;
    let probes = if sqlite3_available() {
        let mut refusals = Vec::new();
        for (name, _) in super::iter_checkpoint_stores() {
            let store = super::store_db_path(&root, name);
            for (kind, sql) in PROBES {
                refusals.push(format!("{name} {kind}: {}", probe_store(&store, sql)?));
            }
        }
        refusals.join("\n")
    } else {
        "foreign-client probes SKIPPED: no stock sqlite3 on PATH".to_string()
    };
    let after = dir_fingerprint(&store_dir)?;
    ensure!(
        before == after,
        "the store file set (or a main store file) changed under the foreign probes:\n  \
         before: {before:?}\n  after:  {after:?}",
    );

    Ok(format!(
        "store-lock check: ok on {}\nsnapshot: {}\n{probes}\nstore directory unchanged ({})",
        std::env::consts::OS,
        snapshot.display(),
        store_dir.display(),
    ))
}

/// Whether the stock `sqlite3` CLI is on `PATH`.
///
/// The probes need it; the project's test run does not (see the module doc), so
/// its absence skips them — the evidence then says so in place of the probe
/// results, and the run still passes.
fn sqlite3_available() -> bool {
    Command::new("sqlite3")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

/// Run one `sqlite3` probe against a live store, require it to be refused as
/// locked, and return the first line of the client's refusal text. Only ever
/// called after [`sqlite3_available`]: a spawn failure here is a hard error, not
/// a skip — running no probe is not the same as a refused probe.
///
/// Both assertions matter — a non-zero exit alone is not proof, because a lost
/// lock lets `sqlite3` reach the schema, where turso's FTS DDL (`CREATE INDEX …
/// USING fts`) is unparseable and also exits non-zero. Assert on the lock/busy
/// substring, never on a byte-exact message: the wording varies by version.
fn probe_store(store: &Path, sql: &str) -> Result<String> {
    let output = Command::new("sqlite3")
        .arg(store)
        .arg(sql)
        .output()
        .with_context(|| format!("spawning sqlite3 to probe {}", store.display()))?;
    let refusal = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    ensure!(
        !output.status.success(),
        "sqlite3 was NOT refused by {} (exit {}) — the store is unlocked: {}",
        store.display(),
        output.status,
        refusal.trim(),
    );
    let lower = refusal.to_lowercase();
    ensure!(
        lower.contains("locked") || lower.contains("busy"),
        "sqlite3 was refused by {} without a lock/busy error — the failure is not the lock: {}",
        store.display(),
        refusal.trim(),
    );
    Ok(refusal
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .to_string())
}

/// One store-directory entry as the check sees it: the name of every entry, plus
/// size and mtime for the store's MAIN files.
#[derive(Debug, PartialEq)]
struct EntryFingerprint {
    name: String,
    size: Option<u64>,
    mtime: Option<std::time::SystemTime>,
}

/// Path-based fingerprint of a store directory (see [`EntryFingerprint`]).
///
/// Metadata only — this must never open a store file, because opening and
/// closing one is exactly the regression the check exists to catch. Sidecars
/// carry no size/mtime, so the service's own `-wal` writes (its log writer
/// flushes on a 500 ms timer) cannot look like a foreign write; a foreign client
/// is caught by the name set (it creates sidecars of its own) and by the main
/// files' size+mtime, which move only when a store could actually be written.
fn dir_fingerprint(dir: &Path) -> Result<Vec<EntryFingerprint>> {
    let mut fingerprint = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let main = Path::new(&name).extension().is_some_and(|e| e == "db");
        let (size, mtime) = if main {
            (Some(metadata.len()), metadata.modified().ok())
        } else {
            (None, None)
        };
        fingerprint.push(EntryFingerprint { name, size, mtime });
    }
    fingerprint.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(fingerprint)
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsRawFd;

    use super::*;
    use crate::util::UnwrapPoison;

    /// Marker for the spawned child: it runs the check instead of the suite's
    /// tests. Set only by the driver below.
    const CHILD_ENV: &str = "MAHBOT_STORE_LOCK_CHECK_CHILD";

    /// Child mode for [`a_store_held_by_another_process_refuses_the_boot`]: boots
    /// the daemon's store bring-up against the `HOME` the driver prepared.
    const BOOT_CHILD_ENV: &str = "MAHBOT_STORE_BOOT_CHILD";

    /// A command for one of the child modes, with `HOME` pointed at the hermetic
    /// root the driver prepared.
    fn child(home: &tempfile::TempDir) -> Command {
        let mut command = Command::new(std::env::current_exe().expect("test binary path"));
        command.env("HOME", home.path());
        command
    }

    /// Run the daemon's store bring-up in a child: creates the stores on a fresh
    /// root, and must refuse a root whose store another process holds.
    fn boot_stores(home: &tempfile::TempDir) -> std::process::Output {
        child(home)
            .args(["--ignored", "--nocapture", "store_boot_child"])
            .env(BOOT_CHILD_ENV, "1")
            .output()
            .expect("spawn the store boot child")
    }

    /// The `core.db*` entries of a store directory (see [`dir_fingerprint`]).
    fn core_family(dir: &Path) -> Result<Vec<EntryFingerprint>> {
        Ok(dir_fingerprint(dir)?
            .into_iter()
            .filter(|entry| entry.name.starts_with("core.db"))
            .collect())
    }

    /// The child's stdout and stderr as one string (the harness reports a panicking
    /// test on stderr, the evidence on stdout).
    fn child_text(output: &std::process::Output) -> String {
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
    }

    /// Take the engine's own lock on `store` — fcntl `F_SETLK`, `F_WRLCK`, the
    /// whole file to its end — and return the descriptor that owns it, so the
    /// lock lives until that file is dropped. This process never boots the
    /// stores, so taking the lock here makes it the foreign process a boot must
    /// refuse.
    fn hold_store_lock(store: &Path) -> std::fs::File {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(store)
            .expect("open the store to lock");
        let mut lock: libc::flock = unsafe { std::mem::zeroed() };
        lock.l_type = libc::c_short::try_from(libc::F_WRLCK).expect("F_WRLCK fits a c_short");
        lock.l_whence = libc::c_short::try_from(libc::SEEK_SET).expect("SEEK_SET fits a c_short");
        // `l_start`/`l_len` stay 0: from the start of the file to its end, the
        // same range the engine locks.
        let locked = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETLK, &lock) };
        assert_eq!(
            locked,
            0,
            "taking the store lock failed: {}",
            std::io::Error::last_os_error()
        );
        file
    }

    /// The check as the project's own tests run it: a separate process is the
    /// whole point (a process cannot observe its own record locks), so this
    /// re-executes the test binary with the marker set and asserts on the
    /// evidence the child prints. The filter below names
    /// [`store_lock_check_child`]; if that test is ever renamed the child runs
    /// nothing and this driver fails on the missing evidence line.
    #[test]
    fn live_stores_stay_locked_for_a_foreign_client() {
        // The child inherits this process's environment, so hold the env lock
        // for the whole run: `util::test::set_env_var` callers take that same
        // lock, and a change set at the wrong instant would boot the child
        // under it.
        let _env = crate::util::test::env_lock().lock().unwrap_poison();
        let home = tempfile::TempDir::new().expect("hermetic home");
        let output = child(&home)
            .args(["--ignored", "--nocapture", "store_lock_check_child"])
            .env(CHILD_ENV, "1")
            .output()
            .expect("spawn the child store-lock check");
        let text = child_text(&output);
        assert!(
            output.status.success(),
            "store-lock check failed ({}):\n{text}",
            output.status,
        );
        assert!(
            text.contains("store-lock check: ok"),
            "the separate process must report the check's evidence:\n{text}"
        );
    }

    /// The other half of the guarantee, end to end and across two processes:
    /// while another process holds the engine's own lock on a store's main file,
    /// the daemon's bring-up must refuse — must not quarantine or replace what it
    /// cannot open, and must record the refusal on the boot-diagnostic
    /// channel. The holder takes the lock the way the service does: an fcntl
    /// `F_WRLCK` over the whole main file, held for as long as the process holds
    /// the descriptor.
    #[test]
    fn a_store_held_by_another_process_refuses_the_boot() {
        let _env = crate::util::test::env_lock().lock().unwrap_poison();
        let home = tempfile::TempDir::new().expect("hermetic home");
        let created = boot_stores(&home);
        let text = String::from_utf8_lossy(&created.stdout).into_owned();
        assert!(
            created.status.success(),
            "the stores must be created first:\n{text}"
        );

        let store_dir = home.path().join(".mahbot/db");
        // Held for the rest of the test (POSIX record locks live until the
        // descriptor is closed): this process is the foreign one, since only the
        // boot children ever open these stores.
        let _holder = hold_store_lock(&store_dir.join("core.db"));

        let before = core_family(&store_dir).expect("fingerprint the core store");
        let refused = boot_stores(&home);
        let text = child_text(&refused);
        assert!(
            !refused.status.success(),
            "a store another process holds must refuse the boot:\n{text}"
        );
        assert!(
            text.to_lowercase().contains("locked"),
            "the refusal must name the lock, not something else:\n{text}"
        );
        assert!(
            text.contains("refusing to start: a store is locked by another process"),
            "the refusal must be RECORDED on the boot-diagnostic channel (stderr), not only \
             returned to the GUI:\n{text}"
        );
        assert_eq!(
            before,
            core_family(&store_dir).expect("fingerprint the core store"),
            "a store that cannot be opened must be left exactly as it is",
        );
    }

    /// A store file that exists but is not usable refuses the boot, and does so
    /// before either store is opened: the logs store is never created, the
    /// unusable file is left byte-for-byte and timestamp-for-timestamp as it
    /// was, and the refusal lands in the storage root's durable record.
    #[test]
    fn an_unusable_store_file_refuses_the_boot() {
        let _env = crate::util::test::env_lock().lock().unwrap_poison();
        let home = tempfile::TempDir::new().expect("hermetic home");
        let root = home.path().join(".mahbot");
        let store_dir = root.join("db");
        std::fs::create_dir_all(&store_dir).expect("create the store dir");
        // A present but empty data file: not a recoverable state, and the boot
        // must refuse rather than back the journal into it or build a schema.
        std::fs::write(store_dir.join("core.db"), []).expect("write the empty store");
        let before = core_family(&store_dir).expect("fingerprint the core store");

        let refused = boot_stores(&home);
        let text = child_text(&refused);
        assert!(
            !refused.status.success(),
            "an unusable store file must refuse the boot:\n{text}"
        );
        assert!(
            text.contains("refusing to start: store 'core' is not usable"),
            "the refusal must name the store and the reason:\n{text}"
        );
        assert!(
            !store_dir.join("logs.db").exists(),
            "the gate must run before the logs store is opened — nothing may be created",
        );
        assert_eq!(
            before,
            core_family(&store_dir).expect("fingerprint the core store"),
            "the refused store must be left completely untouched",
        );
        let record = std::fs::read_to_string(root.join("error.log")).expect("error.log");
        assert!(
            record.contains("MahBot start-up refusal")
                && record.contains("store: core")
                && record.contains("the data file is empty"),
            "the refusal must be recorded durably:\n{record}"
        );
    }

    /// A store the process cannot read (permissions) refuses the boot like any
    /// other unusable file, and its refusal is recorded as environment-caused —
    /// the file is intact and the cause is outside it, so it must never be filed
    /// as damage. Skipped when the suite runs as root, where mode 000 denies
    /// nothing (the classification itself is unit-tested in `wal_guard`/`boot`).
    #[test]
    fn an_unreadable_store_file_refuses_the_boot_as_an_environment_cause() {
        use std::os::unix::fs::PermissionsExt;

        if unsafe { libc::geteuid() } == 0 {
            println!("skipped: running as root, mode 000 does not deny access");
            return;
        }
        let _env = crate::util::test::env_lock().lock().unwrap_poison();
        let home = tempfile::TempDir::new().expect("hermetic home");
        let root = home.path().join(".mahbot");
        let store_dir = root.join("db");
        std::fs::create_dir_all(&store_dir).expect("create the store dir");
        // Long enough to carry a header (512 bytes), so the defect is the read
        // and not the size: this file is a store the process is not allowed to
        // open, not a damaged one.
        let core = store_dir.join("core.db");
        std::fs::write(&core, [0x42; 512]).expect("write the store");
        std::fs::set_permissions(&core, std::fs::Permissions::from_mode(0o000))
            .expect("make the store unreadable");
        let before = core_family(&store_dir).expect("fingerprint the core store");

        let refused = boot_stores(&home);
        let text = child_text(&refused);
        assert!(
            !refused.status.success(),
            "a store the process cannot read must refuse the boot:\n{text}"
        );
        assert!(
            text.contains("refusing to start: store 'core' is not usable"),
            "the refusal must name the store and the reason:\n{text}"
        );
        assert_eq!(
            before,
            core_family(&store_dir).expect("fingerprint the core store"),
            "the refused store must be left completely untouched",
        );
        let record = std::fs::read_to_string(root.join("error.log")).expect("error.log written");
        assert!(
            record.contains("MahBot start-up refusal")
                && record.contains("store: core")
                && record.contains(crate::db::failure_record::ENVIRONMENT_CAUSE),
            "a refusal caused by the environment must say so, never read as damage:\n{record}"
        );
    }

    /// Both physical stores unusable: the refusal names every one of them, in the
    /// diagnostic output and in the durable record, and neither is touched.
    #[test]
    fn both_unusable_stores_are_named_in_the_durable_record() {
        let _env = crate::util::test::env_lock().lock().unwrap_poison();
        let home = tempfile::TempDir::new().expect("hermetic home");
        let root = home.path().join(".mahbot");
        let store_dir = root.join("db");
        std::fs::create_dir_all(&store_dir).expect("create the store dir");
        std::fs::write(store_dir.join("core.db"), []).expect("write the empty core store");
        std::fs::write(store_dir.join("logs.db"), [0x42; 128]).expect("write the bad logs store");
        let before = dir_fingerprint(&store_dir).expect("fingerprint the store dir");

        let refused = boot_stores(&home);
        let text = child_text(&refused);
        assert!(
            !refused.status.success(),
            "unusable stores must refuse the boot:\n{text}"
        );
        for store in ["core", "logs"] {
            assert!(
                text.contains(&format!("refusing to start: store '{store}' is not usable")),
                "the refusal must name the {store} store:\n{text}"
            );
        }
        let record = std::fs::read_to_string(root.join("error.log")).expect("error.log");
        assert!(
            record.contains("store: core") && record.contains("store: logs"),
            "the durable record must name both damaged stores:\n{record}"
        );
        assert_eq!(
            before,
            dir_fingerprint(&store_dir).expect("fingerprint the store dir"),
            "both refused stores must be left completely untouched",
        );
    }

    /// Boots the daemon's stores for the child's `HOME` — the setup and the
    /// refusal half of [`a_store_held_by_another_process_refuses_the_boot`].
    #[test]
    #[ignore = "driven by a_store_held_by_another_process_refuses_the_boot"]
    fn store_boot_child() {
        if std::env::var_os(BOOT_CHILD_ENV).is_none() {
            return;
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("current-thread runtime");
        let booted = runtime.block_on(async {
            crate::config::load_or_init().await?;
            crate::boot::open_stores().await.map(|_| ())
        });
        match booted {
            Ok(()) => println!("store boot: ok"),
            Err(e) => panic!("store boot FAILED: {e:#}"),
        }
    }

    /// Runs in the spawned child only; `#[ignore]` keeps it out of an ordinary
    /// test run, where the driver above is the entry point.
    #[test]
    #[ignore = "driven by the driver above, in a child process"]
    fn store_lock_check_child() {
        if std::env::var_os(CHILD_ENV).is_none() {
            return; // `--ignored` by hand: there is nothing to check
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("current-thread runtime");
        let evidence = runtime
            .block_on(verify())
            .unwrap_or_else(|e| panic!("store-lock check FAILED: {e:#}"));
        println!("{evidence}");
    }
}
