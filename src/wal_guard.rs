//! Detection guard for live-WAL / `.tshm` inconsistency artifacts.
//!
//! The daemon's stores run under Limbo's `multiprocess_wal` mode: each store
//! has a `-tshm` coordination file (mmap'd shared state) plus an on-disk
//! `-wal` file. A foreign standard-SQLite actor that removes/recreates the
//! `-wal`/`-shm` files under a running daemon orphans the daemon's WAL file
//! descriptor: the daemon keeps publishing WAL frames through the `-tshm`
//! header, but the on-disk `-wal` is empty (or truncated), so any reader that
//! follows the `-tshm` hits torn-frame reads.
//!
//! This module provides:
//!
//! - [`parse_tshm_header`] — parse the fixed `-tshm` coordination header
//!   (magic `TSHMWAL\0`, `max_frame`/`nbackfills`/`transaction_count` at
//!   byte offsets 56/64/72 per the `#[repr(C)]`
//!   `SharedWalCoordinationMapHeader` layout in turso_core 0.7.0).
//! - [`inspect_store`] / [`check_all_stores`] — classify each store's file
//!   set into an [`StoreArtifactStatus`] without opening the database.
//! - [`run_wal_guard_loop`] — background task that periodically inspects the
//!   live store directory and emits `tracing::warn!` diagnostics when the
//!   orphaned-WAL condition is present.
//!
//! ## Signal
//!
//! - **Orphaned WAL**: `.tshm` header advertises live frames
//!   (`max_frame > 0`) while the on-disk `-wal` is 0 bytes. This is the
//!   distinguishing signal for the live-instance artifact (the daemon writes
//!   to an unlinked inode). It is inherently racy: right after a 5-minute
//!   checkpoint `max_frame` reads 0 even for orphaned stores, so the guard
//!   only fires while the daemon actively publishes frames.

use std::path::Path;
use std::time::Duration;

use tracing::warn;

/// Magic bytes at the start of every `.tshm` coordination file
/// (`SHARED_WAL_COORDINATION_MAGIC` in turso_core 0.7.0).
pub(crate) const TSHM_MAGIC: [u8; 8] = *b"TSHMWAL\0";

/// Version of the `.tshm` coordination header (`SHARED_WAL_COORDINATION_VERSION`).
const TSHM_VERSION: u32 = 1;

/// Byte offsets of the mmap'd `SharedWalCoordinationMapHeader` (`#[repr(C)]`).
///
/// Layout (each `AtomicU*` is `repr(transparent)` over the primitive):
/// magic[8] | version u32 | reader_slot_count u32 | reader_bitmap_word_count u32 |
/// frame_index_block_capacity u32 | frame_index_block_hash_slots u32 |
/// frame_index_max_blocks u32 | frame_index_blocks u32 | frame_index_capacity u32 |
/// frame_index_len u32 | frame_index_overflowed u32 | snapshot_seq u64 |
/// **max_frame u64 @ 56** | **nbackfills u64 @ 64** | **transaction_count u64 @ 72** ...
pub(crate) const TSHM_MAX_FRAME_OFFSET: usize = 56;
pub(crate) const TSHM_NBACKFILLS_OFFSET: usize = 64;
pub(crate) const TSHM_TRANSACTION_COUNT_OFFSET: usize = 72;
/// Minimum bytes to read to cover the header fields above.
pub(crate) const TSHM_HEADER_READ_LEN: usize = 80;

/// How often the background guard re-inspects the store directory.
const WAL_GUARD_INTERVAL_SECS: u64 = 60;

/// Re-announce the persistent **orphaned-WAL** condition after this many
/// consecutive checks (avoids log spam while still surfacing a long-lived
/// condition).
const REANNOUNCE_EVERY_CHECKS: u64 = 10;

/// Parsed `.tshm` coordination header fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TshmHeader {
    /// Highest WAL frame number published by the current writer.
    pub max_frame: u64,
    /// Frames checkpointed from the head of the WAL.
    pub nbackfills: u64,
    /// Monotonic transaction counter.
    pub transaction_count: u64,
}

/// Classification of one store's file set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreArtifactStatus {
    /// Store name (matches the `--db` argument of `mahbot debug`).
    pub store: String,
    /// Parsed `.tshm` header, when a valid coordination file exists.
    pub tshm: Option<TshmHeader>,
    /// On-disk `-wal` size in bytes (0 when missing or empty).
    pub wal_size: u64,
    /// True when `.tshm` advertises live frames but the on-disk `-wal` is
    /// empty — the live-instance artifact (orphaned WAL fd).
    pub orphaned_wal: bool,
}

/// True when the `.tshm` header advertises live WAL frames (`max_frame > 0`)
/// but the on-disk `-wal` is empty — the live-instance artifact (the daemon's
/// WAL fd is orphaned, writing to an unlinked inode).
///
/// Shared by the wal-guard and the debug CLI so the central business rule
/// cannot drift between the two consumers.
#[must_use]
pub(crate) fn is_orphaned_wal(tshm: Option<TshmHeader>, wal_size: u64) -> bool {
    tshm.is_some_and(|h| h.max_frame > 0) && wal_size == 0
}

/// Parse the `.tshm` coordination header at `tshm_path`.
///
/// Returns `None` when the file is missing, too small, or fails magic/version
/// validation (i.e. not a live Limbo multiprocess-WAL coordination file).
pub(crate) fn parse_tshm_header(tshm_path: &Path) -> Option<TshmHeader> {
    use std::io::Read;
    let mut file = std::fs::File::open(tshm_path).ok()?;
    let mut bytes = [0u8; TSHM_HEADER_READ_LEN];
    file.read_exact(&mut bytes).ok()?;
    if bytes[0..8] != TSHM_MAGIC {
        return None;
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
    if version != TSHM_VERSION {
        return None;
    }
    Some(TshmHeader {
        max_frame: u64::from_le_bytes(
            bytes[TSHM_MAX_FRAME_OFFSET..TSHM_MAX_FRAME_OFFSET + 8]
                .try_into()
                .ok()?,
        ),
        nbackfills: u64::from_le_bytes(
            bytes[TSHM_NBACKFILLS_OFFSET..TSHM_NBACKFILLS_OFFSET + 8]
                .try_into()
                .ok()?,
        ),
        transaction_count: u64::from_le_bytes(
            bytes[TSHM_TRANSACTION_COUNT_OFFSET..TSHM_TRANSACTION_COUNT_OFFSET + 8]
                .try_into()
                .ok()?,
        ),
    })
}

/// Classify the file set of one store given its main database file path.
///
/// The store name is derived from the file name (`board.db` → `board`).
/// This is a pure filesystem inspection — it never opens the database, so it
/// is safe to run against live stores and is unit-testable with synthetic
/// file states.
#[must_use]
pub fn inspect_store_at(db_path: &Path) -> StoreArtifactStatus {
    let sidecars = crate::turso::store_sidecars(db_path);
    let tshm = parse_tshm_header(&sidecars.tshm);
    let wal_size = std::fs::metadata(&sidecars.wal).map_or(0, |m| m.len());
    let orphaned_wal = is_orphaned_wal(tshm, wal_size);
    let store = db_path.file_stem().map_or_else(
        || db_path.display().to_string(),
        |s| s.to_string_lossy().into_owned(),
    );
    StoreArtifactStatus {
        store,
        tshm,
        wal_size,
        orphaned_wal,
    }
}

/// Classify the file set of one store under `root/db/`.
#[must_use]
pub fn inspect_store(root: &Path, name: &str) -> StoreArtifactStatus {
    inspect_store_at(&crate::turso::store_db_path(root, name))
}

/// Inspect every checkpointable store under `root/db/`.
#[must_use]
pub fn check_all_stores(root: &Path) -> Vec<StoreArtifactStatus> {
    crate::turso::store_names()
        .into_iter()
        .map(|name| inspect_store(root, name))
        .collect()
}

/// Background loop: periodically inspect the live store directory and emit
/// warnings when the orphaned-WAL condition is present.
///
/// Warnings are emitted on condition transitions. The orphaned-WAL condition
/// is additionally re-announced every [`REANNOUNCE_EVERY_CHECKS`] checks while
/// persistent (it is the dynamic, higher-severity signal — it fluctuates with
/// daemon activity and checkpoint cycles).
pub async fn run_wal_guard_loop() {
    let root = match crate::config::default_config_dir() {
        Ok(root) => root,
        Err(e) => {
            warn!(error = %e, "wal-guard: cannot resolve storage root; guard disabled");
            return;
        }
    };

    let mut last_seen: std::collections::HashMap<String, bool> = std::collections::HashMap::new();
    let mut check_count: u64 = 0;

    loop {
        if !crate::shutdown::sleep_or_shutdown(Duration::from_secs(WAL_GUARD_INTERVAL_SECS)).await {
            break;
        }
        check_count += 1;
        for status in check_all_stores(&root) {
            let store = status.store.clone();

            if status.orphaned_wal {
                let max_frame = status.tshm.map_or(0, |h| h.max_frame);
                let announce = last_seen.get(&store).copied() != Some(true)
                    || check_count.is_multiple_of(REANNOUNCE_EVERY_CHECKS);
                if announce {
                    warn!(
                        store = %status.store,
                        max_frame,
                        wal_size = status.wal_size,
                        "wal-guard: orphaned WAL detected — .tshm advertises live frames but \
                         the on-disk -wal is empty; the daemon's WAL fd is orphaned \
                         (foreign standard-SQLite activity). Reads through .tshm will hit \
                         torn-frame errors. Query snapshot copies (see the snapshot-query \
                         procedure in the README); never delete/recreate -wal/-shm/-tshm \
                         while the daemon runs.",
                    );
                }
                last_seen.insert(store.clone(), true);
            } else {
                last_seen.insert(store.clone(), false);
            }
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

    fn tshm_bytes(max_frame: u64, nbackfills: u64, tx_count: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; TSHM_HEADER_READ_LEN];
        bytes[0..8].copy_from_slice(&TSHM_MAGIC);
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[TSHM_MAX_FRAME_OFFSET..TSHM_MAX_FRAME_OFFSET + 8]
            .copy_from_slice(&max_frame.to_le_bytes());
        bytes[TSHM_NBACKFILLS_OFFSET..TSHM_NBACKFILLS_OFFSET + 8]
            .copy_from_slice(&nbackfills.to_le_bytes());
        bytes[TSHM_TRANSACTION_COUNT_OFFSET..TSHM_TRANSACTION_COUNT_OFFSET + 8]
            .copy_from_slice(&tx_count.to_le_bytes());
        bytes
    }

    #[test]
    fn parses_valid_tshm_header() {
        let dir = std::env::temp_dir().join(format!("wal_guard_tshm_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("board.db-tshm");
        write(&path, &tshm_bytes(42, 3, 99));

        let hdr = parse_tshm_header(&path).expect("valid tshm header parses");
        assert_eq!(hdr.max_frame, 42);
        assert_eq!(hdr.nbackfills, 3);
        assert_eq!(hdr.transaction_count, 99);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_missing_bad_or_short_tshm() {
        let dir = std::env::temp_dir().join(format!("wal_guard_bad_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // Missing file.
        assert!(parse_tshm_header(&dir.join("nope.db-tshm")).is_none());
        // Wrong magic.
        let mut bytes = tshm_bytes(1, 0, 0);
        bytes[0] = b'X';
        write(&dir.join("bad.db-tshm"), &bytes);
        assert!(parse_tshm_header(&dir.join("bad.db-tshm")).is_none());
        // Too short.
        write(&dir.join("short.db-tshm"), &bytes[..16]);
        assert!(parse_tshm_header(&dir.join("short.db-tshm")).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn detects_orphaned_wal() {
        let dir = std::env::temp_dir().join(format!("wal_guard_state_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // Healthy store: tshm advertises 0 frames, wal empty.
        write(&dir.join("db/board.db-tshm"), &tshm_bytes(0, 0, 5));
        write(&dir.join("db/board.db-wal"), &[]);
        let s = inspect_store(&dir, "board");
        assert!(!s.orphaned_wal);

        // Orphaned store: tshm advertises live frames, wal empty.
        write(
            &dir.join("db/sessions.db-tshm"),
            &tshm_bytes(356, 0, 710_565),
        );
        write(&dir.join("db/sessions.db-wal"), &[]);
        let s = inspect_store(&dir, "sessions");
        assert!(s.orphaned_wal);
        assert!(s.tshm.is_some());
        assert_eq!(s.tshm.unwrap().max_frame, 356);

        // Active write on a healthy store: max_frame > 0, wal non-empty → not orphaned.
        write(&dir.join("db/users.db-tshm"), &tshm_bytes(120, 0, 3));
        write(&dir.join("db/users.db-wal"), &[0u8; 4096]);
        let s = inspect_store(&dir, "users");
        assert!(!s.orphaned_wal);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn orphaned_wal_predicate_is_exact() {
        let live = TshmHeader {
            max_frame: 1,
            nbackfills: 0,
            transaction_count: 0,
        };
        // Advertised frames + empty wal → orphaned.
        assert!(is_orphaned_wal(Some(live), 0));
        // Advertised frames + non-empty wal → healthy (active writer).
        assert!(!is_orphaned_wal(Some(live), 4096));
        // Quiet window: max_frame reads 0 right after a checkpoint → not
        // orphaned even with an empty wal.
        let quiet = TshmHeader {
            max_frame: 0,
            nbackfills: 0,
            transaction_count: 0,
        };
        assert!(!is_orphaned_wal(Some(quiet), 0));
        // No valid tshm at all → not orphaned.
        assert!(!is_orphaned_wal(None, 0));
    }

    #[test]
    fn check_all_stores_visits_every_store() {
        let dir = std::env::temp_dir().join(format!("wal_guard_all_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for name in crate::turso::store_names() {
            write(
                &dir.join(format!("db/{name}.db-tshm")),
                &tshm_bytes(0, 0, 0),
            );
        }
        let statuses = check_all_stores(&dir);
        assert_eq!(statuses.len(), crate::turso::store_names().len());
        for s in statuses {
            assert!(!s.orphaned_wal);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
