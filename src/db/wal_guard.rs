//! Boot pre-flight shape classifier for the two physical stores. The daemon runs in **default
//! single-process mode** (multiprocess_wal was removed).
//!
//! The usability rule. Bring-up checks an existing store in two stages. Stage 1 is file-level,
//! before either store is opened: [`classify_store_shape`] stats a store's data file and reads its
//! 18-byte header — an absent file is a genuine first launch, a file that passes is eligible,
//! anything else is a [`ShapeDefect`] and the service refuses to start. Stage 2 is open-level
//! ([`crate::db::open_store`]): a present file must open AND carry the product's own schema. A
//! failure at either stage is a refusal — an existing data file is never replaced, re-created or
//! emptied, and damage detected only while opening is a refusal too. Only a failure refuses: a
//! store that opens with the product's schema and passes the data-preserving repairs boots even
//! when quick_check reports a condition those repairs deliberately leave report-only.
//!
//! The lock rule. The engine takes a whole-file `fcntl` `F_WRLCK` record lock on a store's main
//! file and its `-wal` when it opens them, and holds it until the process exits. POSIX ties record
//! locks to the process and inode: closing ANY descriptor this process holds for such a file drops
//! every record lock it holds on it — the engine never re-takes it and an in-process check cannot
//! notice. So the running service must never open a store file: facts about a live store come from
//! a stat ([`wal_size`]) or from the engine itself. Reading a header ([`classify_store_shape`] via
//! [`read_db_header`]) opens the file, so it is allowed only where this process holds no lock yet:
//! the boot bring-up. The one other per-store fact this module owns is the marker a failed runtime
//! integrity check leaves beside its store ([`note_integrity_scan_owed`]): stat-only, like the
//! `-wal` size, and read by the boot gate as a reason to scan before any reader exists.

use std::path::Path;

use tracing::warn;

/// A store data file's file-level shape, classified before any store is opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoreShape {
    /// No data file — a genuine first launch; the store is created.
    Absent,
    Present,
    /// The data file exists but fails the file-level shape check. The service refuses to start; the
    /// file is left untouched.
    Unusable(ShapeDefect),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShapeDefect {
    /// 0 bytes — never usable, and never back-filled from a surviving journal.
    ZeroBytes,
    /// Shorter than a valid header.
    TooShort(u64),
    /// The file could not be read (stat/read I/O failure other than NotFound).
    Unreadable,
    BadMagic,
    /// The header's page-size field is not a power of two in [512, 65536].
    BadPageSize,
}

impl ShapeDefect {
    /// Human one-liner, for the durable refusal record and the start-failure screen.
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

    /// True when the defect is an environment condition rather than evidence about the store's
    /// contents: a file that could not be read (permissions, a failing or full filesystem) says
    /// nothing about the data, so it must be recorded as an environment-caused refusal, never as
    /// damage.
    #[must_use]
    pub(crate) fn is_environment_caused(self) -> bool {
        matches!(self, Self::Unreadable)
    }
}

/// The `-wal` header's size in bytes (fixed by the SQLite WAL format): a `-wal` of exactly this
/// size holds no frames, so anything larger does.
pub(crate) const WAL_HEADER_BYTES: u64 = 32;

/// A stat-only file size: 0 when the file is absent, `Err` when it exists but cannot be read. The
/// shared read behind the size facts the running service compares — the `-wal` size the checkpoint
/// cap uses, and the sizes the pre-shrink gate checks page counts against.
pub(crate) fn stat_size(path: &Path) -> std::io::Result<u64> {
    match std::fs::metadata(path) {
        Ok(meta) => Ok(meta.len()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(e),
    }
}

/// The store's `-wal` size, stat-only — the fact source for the running service (the checkpoint
/// cap, the persistent-failure report) and for the boot open's integrity-scan gate. The `-wal` can
/// hold committed-but-not-checkpointed frames; the store itself may not be opened (see the lock
/// rule). A failed stat reads as 0: the conservative side for a size compared against a cap, and
/// for "no journal left behind".
#[must_use]
pub(super) fn wal_size(db_path: &Path) -> u64 {
    stat_size(&crate::db::wal_path(db_path)).unwrap_or(0)
}

/// The marker a failed runtime integrity check leaves beside its store: `<db
/// path>-needs-integrity-scan`. The periodic round must not repair under live readers, so it
/// records the failure here; the next start reads it stat-only as a reason to run the repair scan
/// before its readers exist.
fn integrity_scan_marker(db_path: &Path) -> std::path::PathBuf {
    std::path::PathBuf::from(format!("{}-needs-integrity-scan", db_path.display()))
}

/// Record that the store at `db_path` owes an integrity scan — the periodic round's durable half,
/// written with the failure report. Best-effort: a marker that cannot be written costs the next
/// start one reason to scan, reported here and never propagated (the failure record already holds
/// the finding).
pub(crate) fn note_integrity_scan_owed(db_path: &Path) {
    if let Err(e) = std::fs::write(integrity_scan_marker(db_path), b"") {
        warn!(error = %e, "could not mark the store as owing an integrity scan");
    }
}

/// True when a previous run's integrity check failed for this store and no scan has cleared the
/// marker since.
#[must_use]
pub(crate) fn integrity_scan_owed(db_path: &Path) -> bool {
    integrity_scan_marker(db_path).exists()
}

/// The scan completed every repair the store owed. Best-effort: a marker that cannot be removed
/// costs one more scan on the next start, and the periodic round re-marks a store that is still
/// failing.
pub(crate) fn clear_integrity_scan_owed(db_path: &Path) {
    if let Err(e) = std::fs::remove_file(integrity_scan_marker(db_path))
        && e.kind() != std::io::ErrorKind::NotFound
    {
        warn!(error = %e, "could not clear the store's integrity-scan marker");
    }
}

/// Classify a store data file's file-level shape: absent, present, or unusable with the defect.
/// Pure filesystem inspection (stat + an 18-byte header read), unit-testable with synthetic file
/// states.
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

/// Read the 18-byte main-DB header (magic + u16 BE page-size field). `None` on any I/O failure or
/// an unreadable header — the caller decides what `None` means. This OPENS the store's main file,
/// so it is for the boot bring-up (before that store is opened) or a separate process only, never a
/// running service that already holds the store's lock.
pub(crate) fn read_db_header(db_path: &Path) -> Option<[u8; 18]> {
    use std::io::Read;
    let mut header = [0u8; 18];
    let mut file = std::fs::File::open(db_path).ok()?;
    file.read_exact(&mut header).ok()?;
    Some(header)
}

/// True when the header carries a valid page size: a power of two in [512, 65536]. Per the SQLite
/// header format, 65536 is encoded as raw 1 in the u16 page-size field.
#[must_use]
pub(crate) fn db_page_size_valid(header: &[u8; 18]) -> bool {
    let raw = u16::from_be_bytes([header[16], header[17]]);
    let page_size = if raw == 1 { 65_536 } else { u32::from(raw) };
    (512..=65_536).contains(&page_size) && page_size.is_power_of_two()
}

#[must_use]
pub(crate) fn db_header_valid(header: &[u8; 18]) -> bool {
    &header[..16] == DB_HEADER_MAGIC && db_page_size_valid(header)
}

#[derive(Debug)]
pub(crate) struct UnusableStore {
    /// Physical store name (`core` or `logs`).
    pub store: &'static str,
    pub db_path: std::path::PathBuf,
    pub defect: ShapeDefect,
}

/// Boot pre-flight: classify every physical store's data file (the consolidated `core.db` plus the
/// separate `logs.db`) **before either store is opened**, and return **every** unusable one — empty
/// when both are absent or shape-valid. The caller turns the result into the refusal, so it names
/// all damaged stores. Read-only (a stat plus one 18-byte header read per store): no engine open,
/// so the refusal leaves both stores untouched.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &std::path::Path, bytes: &[u8]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    fn valid_header(page_size: u16) -> Vec<u8> {
        let mut db = vec![0u8; 4096];
        db[..16].copy_from_slice(b"SQLite format 3\0");
        db[16..18].copy_from_slice(&page_size.to_be_bytes());
        db
    }

    #[test]
    fn db_header_valid_decodes_64k_page_size() {
        // SQLite encodes 65536 as raw 1 in the u16 page-size field; the boot classifier must accept
        // it (a raw 1 used to fall outside the range check and misclassify a legitimate 64 KiB
        // store as unusable).
        let mut header = [0u8; 18];
        header[..16].copy_from_slice(b"SQLite format 3\0");
        header[16..18].copy_from_slice(&1u16.to_be_bytes());
        assert!(db_header_valid(&header));
        header[0] = b'X';
        assert!(!db_header_valid(&header));
    }

    #[test]
    fn classify_store_shape_classifies_synthetic_file_sets() {
        let dir = std::env::temp_dir().join(format!("wal_guard_shape_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let db_path = dir.join("db/core.db");
        let wal = dir.join("db/core.db-wal");

        // Present: valid magic + a page size in range.
        write(&db_path, &valid_header(4096));
        assert_eq!(classify_store_shape(&db_path), StoreShape::Present);

        // Unusable: truncated file — shorter than a valid header.
        write(&db_path, &[0u8; 64]);
        assert_eq!(
            classify_store_shape(&db_path),
            StoreShape::Unusable(ShapeDefect::TooShort(64))
        );

        // Unusable: 0 bytes, even with a non-empty WAL — never durable.
        write(&db_path, &[]);
        write(&wal, &[0u8; 512]);
        assert_eq!(
            classify_store_shape(&db_path),
            StoreShape::Unusable(ShapeDefect::ZeroBytes)
        );
        let _ = std::fs::remove_file(&wal);

        // Absent: no data file at all — a genuine first launch.
        let _ = std::fs::remove_file(&db_path);
        assert_eq!(classify_store_shape(&db_path), StoreShape::Absent);

        // Unusable: >=100 bytes without the SQLite magic.
        write(&db_path, &[0x42; 128]);
        assert_eq!(
            classify_store_shape(&db_path),
            StoreShape::Unusable(ShapeDefect::BadMagic)
        );

        // Unusable: valid magic with a zero page-size field.
        let mut bad_page = valid_header(4096);
        bad_page[16..18].copy_from_slice(&0u16.to_be_bytes());
        write(&db_path, &bad_page);
        assert_eq!(
            classify_store_shape(&db_path),
            StoreShape::Unusable(ShapeDefect::BadPageSize)
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn classify_store_shape_visits_every_store() {
        let dir = std::env::temp_dir().join(format!("wal_guard_all_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // Only the PHYSICAL store files exist: one consolidated domain file (core.db) plus the logs
        // file. Write a valid header for both so every logical name resolves to Present.
        write(&dir.join("db/core.db"), &valid_header(4096));
        write(&dir.join("db/logs.db"), &valid_header(4096));
        for name in crate::db::store_names() {
            assert_eq!(
                classify_store_shape(&crate::db::store_db_path(&dir, name)),
                StoreShape::Present,
                "fixture store {name} must be present"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The stat-only fact the checkpoint-failure report and the periodic WAL-size cap read: the
    /// on-disk `-wal` size, without opening the store.
    #[test]
    fn wal_size_reports_the_wal_size() {
        let dir = std::env::temp_dir().join(format!("wal_guard_facts_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let db_path = dir.join("db/core.db");
        write(&db_path, &valid_header(4096));
        write(&dir.join("db/core.db-wal"), &[0xAA; 512]);

        assert_eq!(wal_size(&db_path), 512);

        let _ = std::fs::remove_file(dir.join("db/core.db-wal"));
        assert_eq!(wal_size(&db_path), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_store_shapes_reports_every_unusable_file() {
        let dir = std::env::temp_dir().join(format!("wal_guard_scan_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let core = dir.join("db/core.db");
        let logs = dir.join("db/logs.db");

        // Both physical stores unusable: every one is reported, so the refusal names both.
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

    /// An unreadable data file is unusable: `metadata` succeeds (a stat needs no read permission)
    /// while the header read fails.
    #[cfg(unix)]
    #[test]
    fn classify_store_shape_reports_an_unreadable_file() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("wal_guard_unreadable_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let db_path = dir.join("db/core.db");
        write(&db_path, &valid_header(4096));
        std::fs::set_permissions(&db_path, std::fs::Permissions::from_mode(0o000)).unwrap();
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
}
