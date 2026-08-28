//! Boot pre-flight + detect classifier for single-process store corruption.
//!
//! The daemon runs in **default single-process mode** (multiprocess_wal was
//! removed), so there is no `.tshm` coordination file. The genuine corruption
//! classes are:
//!
//! - **Structural** — a bad/truncated main-DB header (quarantine + recreate).
//! - **Durable-B** — a 0-byte main DB with a non-empty WAL (healable via a
//!   PASSIVE-first checkpoint).
//!
//! A stale `.tshm` file is a leftover from a pre-removal multiprocess run; it
//! is detected and reported (`has_stale_tshm` / a boot `warn!`) but never
//! created in normal operation (single-process mode uses the standard `-shm`).
//!
//! [`inspect_store`] / [`inspect_store_at`] classify one store's file set
//! without opening the database. [`diagnose_all_stores`] is the boot
//! pre-flight that feeds each store's heal strategy in `crate::db::open_store`.

use std::path::Path;

use tracing::warn;

use crate::util::UnwrapPoison;

/// Boot pre-flight diagnoses, keyed by the store's absolute db path. Populated
/// by [`diagnose_all_stores`] (before `init_tracing`), consumed once by each
/// store's `open_store` heal path. Keyed per instance (path), not per name, so
/// test stores with the same name in different roots cannot collide.
static BOOT_DIAGNOSES: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, BootDiagnosis>>,
> = std::sync::OnceLock::new();

fn boot_diagnoses()
-> &'static std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, BootDiagnosis>> {
    BOOT_DIAGNOSES.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Record one store's pre-flight diagnosis (daemon boot only).
pub(crate) fn set_boot_diagnosis(db_path: &Path, diagnosis: BootDiagnosis) {
    boot_diagnoses()
        .lock()
        .unwrap_poison()
        .insert(db_path.to_path_buf(), diagnosis);
}

/// Take (consume) one store's boot diagnosis by its db path. `None` outside
/// the daemon boot flow (tests, CLI, non-boot store opens).
#[must_use]
pub(crate) fn take_boot_diagnosis(db_path: &Path) -> Option<BootDiagnosis> {
    boot_diagnoses().lock().unwrap_poison().remove(db_path)
}

/// True when a boot pre-flight diagnosis exists for `db_path` (not yet
/// consumed by [`crate::db::open_store`]). Lets callers that open a store
/// through the boot path know the repair flow (which runs quick_check) will
/// run — e.g. the logs store's verify step, which would otherwise duplicate
/// the boot scan.
#[must_use]
pub(crate) fn has_boot_diagnosis(db_path: &Path) -> bool {
    boot_diagnoses()
        .lock()
        .unwrap_poison()
        .contains_key(db_path)
}

/// Per-store classification captured by the boot pre-flight scan,
/// before any store is opened. The heal strategy flows from this map —
/// turso's own reopen (RebuildFromDisk → install_snapshot) would consume the
/// evidence, so the strategy must not be re-derived post-open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BootDiagnosis {
    /// No damaged state (fresh install, post-TRUNCATE, or a healthy store).
    Healthy,
    /// 0-byte main DB with a non-empty WAL / live frames — durable-B; healed
    /// by PASSIVE-first (backfill), reopen, then TRUNCATE.
    DurableB,
    /// Structural damage (truncated/zeroed main-DB header) — recreate is the
    /// only option.
    Structural,
}

impl BootDiagnosis {
    #[must_use]
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::DurableB => "durable-b",
            Self::Structural => "structural",
        }
    }

    /// Map to the single-process [`StoreClass`].
    #[must_use]
    pub(crate) fn into_store_class(self) -> StoreClass {
        match self {
            Self::Healthy => StoreClass::Healthy,
            Self::DurableB => StoreClass::DurableB,
            Self::Structural => StoreClass::Structural,
        }
    }
}

/// Single-process store classification (no `.tshm` coordination): the genuine
/// corruption classes are structural (bad/truncated main-DB header) and
/// durable-B (0-byte main DB with a non-empty WAL). `has_stale_tshm` reports a
/// leftover `.tshm` from a pre-removal multiprocess run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreClass {
    Healthy,
    DurableB,
    Structural,
}

impl StoreClass {
    /// Short stable label for logs and the `debug detect` output.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::DurableB => "durable-b",
            Self::Structural => "structural",
        }
    }
}

/// Classification of one store's file set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreArtifactStatus {
    /// Store name (matches the `--db` argument of `mahbot debug`).
    pub store: String,
    /// Single-process corruption class (main-DB header + WAL size).
    pub class: StoreClass,
    /// On-disk `-wal` size in bytes (0 when missing or empty).
    pub wal_size: u64,
    /// True when a leftover `.tshm` file exists (stale coordination debris).
    pub has_stale_tshm: bool,
}

/// Classify the file set of one store given its main database file path.
///
/// The store name is derived from the file name (`core.db` → `core`).
/// This is a pure filesystem inspection — it never opens the database, so it
/// is safe to run against live stores and is unit-testable with synthetic
/// file states.
#[must_use]
pub fn inspect_store_at(db_path: &Path) -> StoreArtifactStatus {
    let sidecars = crate::db::store_sidecars(db_path);
    let wal_size = std::fs::metadata(&sidecars.wal).map_or(0, |m| m.len());
    let has_stale_tshm = sidecars.tshm.exists();
    let class = classify_main_db(db_path, sidecars.wal.exists(), wal_size).into_store_class();
    let store = db_path.file_stem().map_or_else(
        || db_path.display().to_string(),
        |s| s.to_string_lossy().into_owned(),
    );
    StoreArtifactStatus {
        store,
        class,
        wal_size,
        has_stale_tshm,
    }
}

/// Classify the file set of one store under `root/db/`.
#[must_use]
pub fn inspect_store(root: &Path, name: &str) -> StoreArtifactStatus {
    inspect_store_at(&crate::db::store_db_path(root, name))
}

/// Main SQLite header magic (first 16 bytes of every `.db` file).
const DB_HEADER_MAGIC: &[u8; 16] = b"SQLite format 3\0";
/// Minimum main-DB size to carry a header (page 1 with a valid magic).
pub(crate) const DB_HEADER_MIN_SIZE: u64 = 100;

/// Read the 18-byte main-DB header (magic + u16 BE page-size field). `None`
/// on any I/O failure or a file shorter than the header — the caller decides
/// what `None` means (wal_guard: fail-open → healthy; debug: fail-closed).
pub(crate) fn read_db_header(db_path: &Path) -> Option<[u8; 18]> {
    use std::io::Read;
    let mut header = [0u8; 18];
    let mut file = std::fs::File::open(db_path).ok()?;
    file.read_exact(&mut header).ok()?;
    Some(header)
}

/// True when the header carries the SQLite magic and a valid page size:
/// power of two in [512, 65536]. Per the SQLite header format, 65536 is
/// encoded as raw 1 in the u16 page-size field.
#[must_use]
pub(crate) fn db_header_valid(header: &[u8; 18]) -> bool {
    if &header[..16] != DB_HEADER_MAGIC {
        return false;
    }
    let raw = u16::from_be_bytes([header[16], header[17]]);
    let page_size = if raw == 1 { 65_536 } else { u32::from(raw) };
    (512..=65_536).contains(&page_size) && page_size.is_power_of_two()
}

/// Classify the main-DB file itself: 0-byte with a live WAL
/// → durable-B; truncated/zeroed header → structural; otherwise healthy.
fn classify_main_db(db_path: &Path, wal_exists: bool, wal_size: u64) -> BootDiagnosis {
    let Ok(meta) = std::fs::metadata(db_path) else {
        return BootDiagnosis::Healthy; // no DB yet — fresh install
    };
    let size = meta.len();
    if size == 0 {
        // 0-byte DB with a non-empty WAL → durable-B (healable via PASSIVE-
        // first). A fresh install (0-byte + empty WAL) is healthy.
        return if wal_exists && wal_size > 0 {
            BootDiagnosis::DurableB
        } else {
            BootDiagnosis::Healthy
        };
    }
    // Fail-closed on truncated headers, fail-open on I/O errors: a short
    // file is structural, but an open/read failure (permission / I-O) is not
    // a quarantine trigger — leave it to the real open. The size gate stays
    // here (not inside read_db_header's None) so sizes 1–99 cannot silently
    // flip to Healthy.
    if size < DB_HEADER_MIN_SIZE {
        return BootDiagnosis::Structural;
    }
    let Some(header) = read_db_header(db_path) else {
        return BootDiagnosis::Healthy;
    };
    if db_header_valid(&header) {
        BootDiagnosis::Healthy
    } else {
        BootDiagnosis::Structural
    }
}

/// Boot pre-flight: classify every physical store (the consolidated `core.db`
/// plus the separate `logs.db`) **before any store is opened** (logs opens
/// first inside `init_tracing`; turso's own reopen would consume the evidence).
/// Runs before the process holds any lock, so path-based reads are safe; the
/// instance lock already excludes a second daemon.
///
/// The result feeds the per-store heal strategy in `crate::db::open_store`.
/// [`cleanup_stale_tshm`] is a separate boot step — it runs after this and
/// removes leftover `.tshm` coordination debris (single-process mode never
/// creates one); it never touches the `-wal`/`-shm` files.
pub fn diagnose_all_stores(root: &Path) {
    for (name, _) in crate::db::iter_checkpoint_stores() {
        let db_path = crate::db::store_db_path(root, name);
        let sidecars = crate::db::store_sidecars(&db_path);
        let wal_size = std::fs::metadata(&sidecars.wal).map_or(0, |m| m.len());
        let diagnosis = classify_main_db(&db_path, sidecars.wal.exists(), wal_size);
        set_boot_diagnosis(&db_path, diagnosis);
        if diagnosis != BootDiagnosis::Healthy {
            crate::boot::boot_diagnostic(format!(
                "boot pre-flight: store '{name}' class {} (wal_size={}) — healing will run \
                 before open",
                diagnosis.label(),
                wal_size,
            ));
        }
    }
}

/// Remove any stale `.tshm` coordination leftover from a pre-removal
/// `multiprocess_wal` run.
///
/// Single-process mode NEVER creates a `.tshm` (turso rebuilds the standard
/// `-shm` from `-wal`), so any `.tshm` present after the removal is necessarily
/// stale debris from a dead multiprocess daemon — safe to delete without a
/// liveness probe. The `-wal` file is NEVER touched: it may hold committed
/// frames from a crash, and deleting it would cause silent commit loss
/// (historically class-A). Call only when the daemon is down (boot before any
/// store opens, or `mahbot debug`'s daemon-down direct-open path).
pub fn cleanup_stale_tshm(root: &Path) {
    for (name, _) in crate::db::iter_checkpoint_stores() {
        let tshm = crate::db::store_sidecars(&crate::db::store_db_path(root, name)).tshm;
        if !tshm.exists() {
            continue;
        }
        match std::fs::remove_file(&tshm) {
            Ok(()) => warn!(
                db = %name,
                path = %tshm.display(),
                "removed stale .tshm coordination leftover from a pre-removal multiprocess run",
            ),
            Err(e) => warn!(
                db = %name,
                path = %tshm.display(),
                error = %e,
                "failed to remove stale .tshm leftover",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &std::path::Path, bytes: &[u8]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn db_header_valid_decodes_64k_page_size() {
        // SQLite encodes 65536 as raw 1 in the u16 page-size field; the boot
        // classifier must accept it (a raw 1 used to fall outside the range
        // check and misclassify a legitimate 64 KiB store as structural).
        let mut header = [0u8; 18];
        header[..16].copy_from_slice(b"SQLite format 3\0");
        header[16..18].copy_from_slice(&1u16.to_be_bytes());
        assert!(db_header_valid(&header));
        // The same field without the magic is invalid.
        header[0] = b'X';
        assert!(!db_header_valid(&header));
    }

    #[test]
    fn inspect_store_classifies_synthetic_file_sets() {
        let dir = std::env::temp_dir().join(format!("wal_guard_state_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // Healthy store: valid 64 KiB-page DB header, no WAL.
        let mut db = vec![0u8; 4096];
        db[..16].copy_from_slice(b"SQLite format 3\0");
        db[16..18].copy_from_slice(&1u16.to_be_bytes());
        write(&dir.join("db/core.db"), &db);
        let s = inspect_store(&dir, "board");
        assert_eq!(s.class, StoreClass::Healthy);
        assert!(!s.has_stale_tshm);

        // Structural: truncated main-DB header.
        write(&dir.join("db/core.db"), &[0u8; 64]);
        let s = inspect_store(&dir, "sessions");
        assert_eq!(s.class, StoreClass::Structural);

        // Durable-B: 0-byte main DB with a non-empty WAL.
        write(&dir.join("db/core.db"), &[]);
        write(&dir.join("db/core.db-wal"), &[0u8; 512]);
        let s = inspect_store(&dir, "users");
        assert_eq!(s.class, StoreClass::DurableB);

        // Fresh store: no main DB file → healthy.
        let _ = std::fs::remove_file(&dir.join("db/core.db"));
        let s = inspect_store(&dir, "config");
        assert_eq!(s.class, StoreClass::Healthy);

        // Stale .tshm debris is reported (never quarantined) while the class
        // is still computed from the main DB / WAL only.
        write(&dir.join("db/core.db-tshm"), &[0u8; 32]);
        let s = inspect_store(&dir, "chat_history");
        assert!(s.has_stale_tshm);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn inspect_store_visits_every_store() {
        let dir = std::env::temp_dir().join(format!("wal_guard_all_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // Only the PHYSICAL store files exist on disk: one consolidated domain
        // file (core.db, backing all the domain stores) + the logs file. Write
        // a valid main-DB header for both physical files so every logical name
        // resolves to a Healthy classification.
        let mut db = vec![0u8; 4096];
        db[..16].copy_from_slice(b"SQLite format 3\0");
        db[16] = 0x10; // page size 4096 (u16 BE)
        write(&dir.join("db/core.db"), &db);
        write(&dir.join("db/logs.db"), &db);
        for name in crate::db::store_names() {
            let s = inspect_store(&dir, name);
            assert_eq!(
                s.class,
                StoreClass::Healthy,
                "fixture store {name} must be healthy"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn diagnose_all_stores_classifies_structural_durable_b_and_healthy() {
        let dir = std::env::temp_dir().join(format!("wal_guard_preflight_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // (1) Structural: truncated main-DB header.
        write(&dir.join("db/core.db"), &[0u8; 64]);
        crate::db::wal_guard::diagnose_all_stores(&dir);
        assert_eq!(
            crate::db::wal_guard::take_boot_diagnosis(&crate::db::store_db_path(&dir, "board")),
            Some(BootDiagnosis::Structural)
        );

        // (2) Durable-B: 0-byte main DB with a non-empty WAL.
        write(&dir.join("db/core.db"), &[]);
        write(&dir.join("db/core.db-wal"), &[0u8; 512]);
        crate::db::wal_guard::diagnose_all_stores(&dir);
        assert_eq!(
            crate::db::wal_guard::take_boot_diagnosis(&crate::db::store_db_path(&dir, "sessions")),
            Some(BootDiagnosis::DurableB)
        );

        // (3) Healthy: 64 KiB page size (raw 1 in the header field) must not be
        // misclassified as structural (quarantine + recreate).
        let _ = std::fs::remove_file(&dir.join("db/core.db-wal"));
        let mut db = vec![0u8; 4096];
        db[..16].copy_from_slice(b"SQLite format 3\0");
        db[16..18].copy_from_slice(&1u16.to_be_bytes());
        write(&dir.join("db/core.db"), &db);
        crate::db::wal_guard::diagnose_all_stores(&dir);
        assert_eq!(
            crate::db::wal_guard::take_boot_diagnosis(&crate::db::store_db_path(
                &dir,
                "chat_history"
            )),
            Some(BootDiagnosis::Healthy)
        );

        // (4) Healthy: fresh store state (no main DB file yet).
        let _ = std::fs::remove_file(&dir.join("db/core.db"));
        crate::db::wal_guard::diagnose_all_stores(&dir);
        assert_eq!(
            crate::db::wal_guard::take_boot_diagnosis(&crate::db::store_db_path(&dir, "users")),
            Some(BootDiagnosis::Healthy)
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cleanup_stale_tshm_removes_only_tshm_not_wal() {
        let dir = std::env::temp_dir().join(format!("wal_guard_cleanup_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let db_dir = dir.join("db");
        write(&db_dir.join("core.db"), &[0u8; 4096]);
        // Simulate a stale multiprocess leftover: `.tshm` present next to the
        // main DB, and a WAL that may hold committed frames.
        write(&db_dir.join("core.db-tshm"), &[0u8; 64]);
        write(&db_dir.join("core.db-wal"), &[0xAA; 512]);
        write(&db_dir.join("logs.db-tshm"), &[0u8; 64]);

        crate::db::wal_guard::cleanup_stale_tshm(&dir);

        // The stale `.tshm` files are gone; the `-wal` (which may hold
        // committed-but-uncheckpointed frames) is NEVER touched.
        assert!(
            !db_dir.join("core.db-tshm").exists(),
            "stale .tshm should be removed"
        );
        assert!(
            !db_dir.join("logs.db-tshm").exists(),
            "stale .tshm should be removed"
        );
        assert!(
            db_dir.join("core.db-wal").exists(),
            "-wal must never be removed (class-A commit-loss footgun)"
        );
        assert_eq!(
            std::fs::metadata(db_dir.join("core.db-wal")).unwrap().len(),
            512,
            "-wal contents must be untouched"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
