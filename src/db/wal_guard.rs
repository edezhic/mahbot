//! Boot pre-flight shape classifier for the two physical stores.
//!
//! The daemon runs in **default single-process mode** (multiprocess_wal was
//! removed).
//!
//! # The usability rule
//!
//! Bring-up checks an existing store in two stages. Stage 1 is file-level and
//! runs before either store is opened: [`classify_store_shape`] stats a store's
//! data file and reads its 18-byte header. An absent file is a genuine first
//! launch (the store is created); a file that passes the shape check is
//! eligible; anything else is a [`ShapeDefect`] and the service refuses to
//! start. Stage 2 is open-level ([`crate::db::open_store`]): a present file must
//! open AND carry the product's own schema (the `schema_migrations` ledger).
//! A failure at either stage is a refusal — an existing data file is never
//! replaced, re-created, re-initialised or emptied, and damage detected only
//! while opening (e.g. `Invalid page type`) is a refusal too. Only a failure
//! refuses: a store that opens with the product's schema and passes the
//! data-preserving repairs boots even when quick_check reports a condition those
//! repairs deliberately leave report-only (see [`crate::db::open_store`]).
//!
//! A stale `.tshm` file is a leftover from a pre-removal multiprocess run; it
//! is detected and reported (`has_stale_tshm` / a boot `warn!`) but is never
//! created in normal operation.
//!
//! # The lock rule
//!
//! The engine takes a whole-file `fcntl` `F_WRLCK` record lock on a store's
//! main file and its `-wal` when it opens them, and holds it until the process
//! exits. POSIX ties record locks to the *process and inode*: closing ANY
//! descriptor this process holds for such a file drops every record lock the
//! process holds on it — the engine never re-takes it, and an in-process check
//! can never notice. So the running service must never open a store file: facts
//! about a live store come from a stat (`store_file_facts`) or from the engine
//! itself. Reading a store's header ([`classify_store_shape`] /
//! [`inspect_store_at`], via [`read_db_header`]) opens the file, so it is
//! allowed only where this process holds no lock yet: the boot bring-up (the
//! pre-flight scan and the open of each store both run before that store is
//! locked) or a separate process (`mahbot debug detect`).

use std::path::Path;

use tracing::warn;

/// A store data file's file-level shape, classified before any store is opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoreShape {
    /// No data file — a genuine first launch; the store is created.
    Absent,
    /// The data file exists and passes the file-level shape check.
    Present,
    /// The data file exists but fails the file-level shape check. The service
    /// refuses to start; the file is left untouched.
    Unusable(ShapeDefect),
}

/// Why a store data file fails the file-level shape check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShapeDefect {
    /// 0 bytes — never usable, and never back-filled from a surviving journal.
    ZeroBytes,
    /// Shorter than a valid header.
    TooShort(u64),
    /// The file could not be read (stat/read I/O failure other than NotFound).
    Unreadable,
    /// The header does not carry the SQLite magic.
    BadMagic,
    /// The header's page-size field is not a power of two in [512, 65536].
    BadPageSize,
}

impl StoreShape {
    /// Short stable label for logs and the `debug detect` output.
    #[must_use]
    pub(crate) fn label(self) -> String {
        match self {
            Self::Absent => "absent".to_string(),
            Self::Present => "present".to_string(),
            Self::Unusable(defect) => format!("unusable:{}", defect.label()),
        }
    }
}

impl ShapeDefect {
    /// Short stable label for the defect, composed into [`StoreShape::label`].
    #[must_use]
    fn label(self) -> &'static str {
        match self {
            Self::ZeroBytes => "empty",
            Self::TooShort(_) => "short",
            Self::Unreadable => "unreadable",
            Self::BadMagic => "bad-magic",
            Self::BadPageSize => "bad-page-size",
        }
    }

    /// Human one-liner, for the durable refusal record and the start-failure
    /// screen.
    #[must_use]
    pub(crate) fn reason(self) -> String {
        match self {
            Self::ZeroBytes => "the data file is empty (0 bytes)".to_string(),
            Self::TooShort(n) => {
                format!("the data file is too short for a valid header ({n} bytes)")
            }
            Self::Unreadable => "the data file could not be read".to_string(),
            Self::BadMagic => "the data file does not carry the SQLite header magic".to_string(),
            Self::BadPageSize => "the SQLite header carries an invalid page size".to_string(),
        }
    }
}

/// Classification of one store's file set for the `debug detect` output: the
/// file-level shape — from the stat facts, plus the main-DB header when the file
/// is long enough to have one — and the facts it was read from.
#[derive(Debug)]
pub(super) struct StoreArtifactStatus {
    /// File-level shape (main-DB header).
    pub class: StoreShape,
    /// On-disk `-wal` size in bytes (0 when missing or empty).
    pub wal_size: u64,
    /// True when a leftover `.tshm` file exists (stale coordination debris).
    pub has_stale_tshm: bool,
}

/// Stat-only facts about one store's file set — the fact source for the running
/// service (the periodic checkpoint cap and the persistent-failure report both
/// use this). See the module doc's lock rule for why nothing here may be opened.
#[derive(Debug)]
pub(super) struct StoreFileFacts {
    pub wal_size: u64,
    pub has_stale_tshm: bool,
}

/// Collect a store's [`StoreFileFacts`] without opening anything. The running
/// service must use this instead of [`inspect_store_at`].
#[must_use]
pub(super) fn store_file_facts(db_path: &Path) -> StoreFileFacts {
    let sidecars = crate::db::store_sidecars(db_path);
    StoreFileFacts {
        wal_size: std::fs::metadata(&sidecars.wal).map_or(0, |m| m.len()),
        has_stale_tshm: sidecars.tshm.exists(),
    }
}

/// Classify one store's file set given its main database file path. Reading the
/// main-DB header is subject to the module doc's lock rule: the boot bring-up
/// (before that store is opened) or a separate process only. Pure filesystem
/// inspection, unit-testable with synthetic file states.
#[must_use]
pub(super) fn inspect_store_at(db_path: &Path) -> StoreArtifactStatus {
    let facts = store_file_facts(db_path);
    StoreArtifactStatus {
        class: classify_store_shape(db_path),
        wal_size: facts.wal_size,
        has_stale_tshm: facts.has_stale_tshm,
    }
}

/// Classify a store data file's file-level shape: absent, present, or unusable
/// with the defect that made it so. Pure filesystem inspection (stat + an
/// 18-byte header read), unit-testable with synthetic file states.
#[must_use]
pub(crate) fn classify_store_shape(db_path: &Path) -> StoreShape {
    let meta = match std::fs::metadata(db_path) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return StoreShape::Absent,
        Err(_) => return StoreShape::Unusable(ShapeDefect::Unreadable),
    };
    let size = meta.len();
    if size == 0 {
        return StoreShape::Unusable(ShapeDefect::ZeroBytes);
    }
    if size < DB_HEADER_MIN_SIZE {
        return StoreShape::Unusable(ShapeDefect::TooShort(size));
    }
    let Some(header) = read_db_header(db_path) else {
        return StoreShape::Unusable(ShapeDefect::Unreadable);
    };
    if &header[..16] != DB_HEADER_MAGIC {
        return StoreShape::Unusable(ShapeDefect::BadMagic);
    }
    if !db_page_size_valid(&header) {
        return StoreShape::Unusable(ShapeDefect::BadPageSize);
    }
    StoreShape::Present
}

/// Main SQLite header magic (first 16 bytes of every `.db` file).
const DB_HEADER_MAGIC: &[u8; 16] = b"SQLite format 3\0";
/// Minimum main-DB size to carry a header (page 1 with a valid magic).
pub(crate) const DB_HEADER_MIN_SIZE: u64 = 100;

/// Read the 18-byte main-DB header (magic + u16 BE page-size field). `None`
/// on any I/O failure or an unreadable header — the caller decides what `None`
/// means (wal_guard: [`StoreShape::Unusable`]; debug: fail-closed).
///
/// This OPENS the store's main file — module doc's lock rule: the boot bring-up
/// (before that store is opened) or a separate process only, never a running
/// service that already holds the store's lock.
pub(crate) fn read_db_header(db_path: &Path) -> Option<[u8; 18]> {
    use std::io::Read;
    let mut header = [0u8; 18];
    let mut file = std::fs::File::open(db_path).ok()?;
    file.read_exact(&mut header).ok()?;
    Some(header)
}

/// True when the header carries a valid page size: a power of two in
/// [512, 65536]. Per the SQLite header format, 65536 is encoded as raw 1 in the
/// u16 page-size field.
#[must_use]
pub(crate) fn db_page_size_valid(header: &[u8; 18]) -> bool {
    let raw = u16::from_be_bytes([header[16], header[17]]);
    let page_size = if raw == 1 { 65_536 } else { u32::from(raw) };
    (512..=65_536).contains(&page_size) && page_size.is_power_of_two()
}

/// True when the header carries the SQLite magic and a valid page size.
#[must_use]
pub(crate) fn db_header_valid(header: &[u8; 18]) -> bool {
    &header[..16] == DB_HEADER_MAGIC && db_page_size_valid(header)
}

/// One store whose data file exists but fails the file-level shape check.
#[derive(Debug)]
pub(crate) struct UnusableStore {
    /// Physical store name (`core` or `logs`).
    pub store: &'static str,
    /// The store's data-file path.
    pub db_path: std::path::PathBuf,
    /// The defect that made the file unusable.
    pub defect: ShapeDefect,
}

/// Boot pre-flight: classify every physical store's data file (the consolidated
/// `core.db` plus the separate `logs.db`) **before either store is opened**, and
/// return **every** unusable store — empty when both are absent or shape-valid.
/// The caller turns the result into the refusal, so a refusal names all damaged
/// stores. Read-only (a stat plus one 18-byte header read per store): no engine
/// open, so the refusal leaves both stores untouched. See the module doc's lock
/// rule for when a header may be read.
///
/// [`cleanup_stale_tshm`] is a separate boot step that runs after this.
#[must_use]
pub(crate) fn scan_store_shapes(root: &Path) -> Vec<UnusableStore> {
    let mut unusable = Vec::new();
    for (name, _) in crate::db::iter_checkpoint_stores() {
        let db_path = crate::db::store_db_path(root, name);
        if let StoreShape::Unusable(defect) = classify_store_shape(&db_path) {
            unusable.push(UnusableStore {
                store: name,
                db_path,
                defect,
            });
        }
    }
    unusable
}

/// Remove any stale `.tshm` coordination leftover from a pre-removal
/// `multiprocess_wal` run.
///
/// Single-process mode NEVER creates a `.tshm` (see the module doc), so any
/// `.tshm` present after the removal is necessarily stale debris from a dead
/// multiprocess daemon — safe to delete without a liveness probe. The `-wal`
/// file is NEVER touched: it may hold committed frames from a crash, and
/// deleting it would cause silent commit loss (historically class-A). Call only
/// when the daemon is down (boot before any store opens, or `mahbot debug`'s
/// daemon-down direct-open path).
pub(crate) fn cleanup_stale_tshm(root: &Path) {
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

    /// A minimal byte image with the SQLite magic and the given page size.
    fn valid_header(page_size: u16) -> Vec<u8> {
        let mut db = vec![0u8; 4096];
        db[..16].copy_from_slice(b"SQLite format 3\0");
        db[16..18].copy_from_slice(&page_size.to_be_bytes());
        db
    }

    #[test]
    fn db_header_valid_decodes_64k_page_size() {
        // SQLite encodes 65536 as raw 1 in the u16 page-size field; the boot
        // classifier must accept it (a raw 1 used to fall outside the range
        // check and misclassify a legitimate 64 KiB store as unusable).
        let mut header = [0u8; 18];
        header[..16].copy_from_slice(b"SQLite format 3\0");
        header[16..18].copy_from_slice(&1u16.to_be_bytes());
        assert!(db_header_valid(&header));
        // The same field without the magic is invalid.
        header[0] = b'X';
        assert!(!db_header_valid(&header));
    }

    #[test]
    fn inspect_store_at_classifies_synthetic_file_sets() {
        let dir = std::env::temp_dir().join(format!("wal_guard_shape_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let db_path = dir.join("db/core.db");
        let wal = dir.join("db/core.db-wal");

        // Present: valid magic + a page size in range.
        write(&db_path, &valid_header(4096));
        let s = inspect_store_at(&db_path);
        assert_eq!(s.class, StoreShape::Present);
        assert!(!s.has_stale_tshm);

        // Unusable: truncated file — shorter than a valid header.
        write(&db_path, &[0u8; 64]);
        assert_eq!(
            inspect_store_at(&db_path).class,
            StoreShape::Unusable(ShapeDefect::TooShort(64))
        );

        // Unusable: 0 bytes, even with a non-empty WAL — never durable.
        write(&db_path, &[]);
        write(&wal, &[0u8; 512]);
        assert_eq!(
            inspect_store_at(&db_path).class,
            StoreShape::Unusable(ShapeDefect::ZeroBytes)
        );
        let _ = std::fs::remove_file(&wal);

        // Absent: no data file at all — a genuine first launch.
        let _ = std::fs::remove_file(&db_path);
        assert_eq!(inspect_store_at(&db_path).class, StoreShape::Absent);

        // Unusable: >=100 bytes without the SQLite magic.
        write(&db_path, &[0x42; 128]);
        assert_eq!(
            inspect_store_at(&db_path).class,
            StoreShape::Unusable(ShapeDefect::BadMagic)
        );

        // Unusable: valid magic with a zero page-size field.
        let mut bad_page = valid_header(4096);
        bad_page[16..18].copy_from_slice(&0u16.to_be_bytes());
        write(&db_path, &bad_page);
        assert_eq!(
            inspect_store_at(&db_path).class,
            StoreShape::Unusable(ShapeDefect::BadPageSize)
        );

        // Stale .tshm debris is reported (never removed here) while the class
        // is still computed from the main DB alone.
        write(&dir.join("db/core.db-tshm"), &[0u8; 32]);
        assert!(inspect_store_at(&db_path).has_stale_tshm);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn inspect_store_at_visits_every_store() {
        let dir = std::env::temp_dir().join(format!("wal_guard_all_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // Only the PHYSICAL store files exist on disk: one consolidated domain
        // file (core.db, backing all the domain stores) + the logs file. Write
        // a valid main-DB header for both physical files so every logical name
        // resolves to a Present classification.
        write(&dir.join("db/core.db"), &valid_header(4096));
        write(&dir.join("db/logs.db"), &valid_header(4096));
        for name in crate::db::store_names() {
            let s = inspect_store_at(&crate::db::store_db_path(&dir, name));
            assert_eq!(
                s.class,
                StoreShape::Present,
                "fixture store {name} must be present"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_store_shapes_reports_every_unusable_file() {
        let dir = std::env::temp_dir().join(format!("wal_guard_scan_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let core = dir.join("db/core.db");
        let logs = dir.join("db/logs.db");

        // Both physical stores unusable: every one is reported, so the refusal
        // names both.
        write(&core, &[]);
        write(&logs, &[0x42; 128]);
        let unusable = scan_store_shapes(&dir);
        let named: Vec<&str> = unusable.iter().map(|u| u.store).collect();
        assert_eq!(named, ["core", "logs"], "every unusable store is reported");
        assert_eq!(unusable[0].defect, ShapeDefect::ZeroBytes);
        assert!(unusable[0].db_path.ends_with("core.db"));

        // A valid core.db with logs.db absent is a clean first launch for logs.
        write(&core, &valid_header(4096));
        let _ = std::fs::remove_file(&logs);
        assert!(scan_store_shapes(&dir).is_empty());

        // Both absent: nothing to refuse.
        let _ = std::fs::remove_file(&core);
        assert!(scan_store_shapes(&dir).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An unreadable data file is unusable: `metadata` succeeds (a stat needs no
    /// read permission) while the header read fails.
    #[cfg(unix)]
    #[test]
    fn classify_store_shape_reports_an_unreadable_file() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("wal_guard_unreadable_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let db_path = dir.join("db/core.db");
        write(&db_path, &valid_header(4096));
        std::fs::set_permissions(&db_path, std::fs::Permissions::from_mode(0o000)).unwrap();
        // Root bypasses file modes, so the defect cannot be produced there.
        if std::fs::read(&db_path).is_ok() {
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }

        assert_eq!(
            classify_store_shape(&db_path),
            StoreShape::Unusable(ShapeDefect::Unreadable),
        );

        std::fs::set_permissions(&db_path, std::fs::Permissions::from_mode(0o644)).unwrap();
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
