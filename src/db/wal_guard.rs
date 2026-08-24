//! Detection guard for live-WAL / `.tshm` inconsistency artifacts.
//!
//! The daemon's stores run under turso's `multiprocess_wal` mode: each store
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
//!   (116 bytes; magic `TSHMWAL\0`, `frame_index_len`/`snapshot_seq`/
//!   `max_frame`/`nbackfills`/`page_size`/`checkpoint_seq`/`salt_1`/`salt_2`
//!   per the `#[repr(C)]` `SharedWalCoordinationMapHeader` layout in
//!   turso_core 0.7.2).
//! - [`classify_wal_state`] — the single classification predicate for the
//!   coordination state of one store, shared by the checkpoint loop (blocking
//!   subset), the wal-guard loop (warn-only full predicate), the boot
//!   pre-flight, and the `debug detect` CLI verb.
//! - [`inspect_store`] — classify one store's file set into a
//!   [`StoreArtifactStatus`] without opening the database.
//! - [`run_wal_guard_loop`] — background task that periodically inspects the
//!   live store directory and emits `tracing::warn!` diagnostics when the
//!   coordination state is non-healthy.
//!
//! ## Lock-drop safety
//!
//! macOS fcntl locks are process-scoped: closing **any** fd to a file drops
//! all of the process's locks on it. The daemon must never open+close
//! `-tshm`/`-wal` files after startup, or it silently releases the byte-0
//! lifetime lock and a second process classifies its open as `Exclusive`
//! (triggering repair that wipes live reader slots). All daemon-side reads go
//! through the persistent per-store fds on [`crate::db::Connection`]; the
//! path-based helpers here are reserved for processes that hold no locks
//! (CLI, boot pre-flight, tests) and are counted by the regression counter
//! [`tshm_open_close_count`].
//!
//! ## Signal
//!
//! The classification mirrors turso's own `classify_authority_snapshot_against_wal`
//! (turso_core 0.7.2) with two deliberate deviations:
//!
//! - Case (1) (`max_frame == 0` → healthy regardless of WAL size) is looser
//!   than turso, which rebuilds for `maxf==0 && wal ∉ {0, 32}`. The looser
//!   form keeps the documented quiet window (right after a daemon checkpoint
//!   `max_frame` reads 0 even for healthy stores) non-blocking; turso's own
//!   open performs the stricter rebuild and that is its responsibility.
//! - The trusted-WAL case additionally requires the exact expected length
//!   `32 + maxf*(24+page_size)`. Mirrors turso's Trusted branch: the full
//!   predicate also validates the last frame's commit/salt/checksum against
//!   `.tshm` (a single-frame read — O(maxf) frame-content validation remains
//!   an accepted blind spot: a truncate-to-32B with continued writing and
//!   exact length + matching salts can only be caught by a full scan).
//!
//! Residual blind spots (documented, out of scope): the quiet window
//! (`wal=0, maxf=0` right after a checkpoint — indistinguishable in one
//! snapshot but safe, the next append recreates a matching header), and
//! post-restart self-consistent states (invisible to any static scan).

use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tracing::{debug, info, warn};

use crate::util::UnwrapPoison;

/// Magic bytes at the start of every `.tshm` coordination file
/// (`SHARED_WAL_COORDINATION_MAGIC` in turso_core 0.7.2).
pub(crate) const TSHM_MAGIC: [u8; 8] = *b"TSHMWAL\0";

/// Version of the `.tshm` coordination header (`SHARED_WAL_COORDINATION_VERSION`).
const TSHM_VERSION: u32 = 1;

/// Byte offsets of the mmap'd `#[repr(C)]` `SharedWalCoordinationMapHeader`
/// in turso_core 0.7.2 (storage/shared_wal_coordination.rs).
///
/// Layout (each `AtomicU*` is a `repr(C)` wrapper over the primitive):
/// magic[8] | version u32 | reader_slot_count u32 | reader_bitmap_word_count u32 |
/// frame_index_block_capacity u32 | frame_index_block_hash_slots u32 |
/// frame_index_max_blocks u32 | frame_index_blocks u32 | frame_index_capacity u32 |
/// **frame_index_len u32 @ 40** | frame_index_overflowed u32 |
/// **snapshot_seq u64 @ 48** | **max_frame u64 @ 56** | **nbackfills u64 @ 64** |
/// **transaction_count u64 @ 72** | visibility_generation u64 |
/// **checkpoint_seq u32 @ 88** | checkpoint_epoch u32 | **page_size u32 @ 96** |
/// **salt_1 u32 @ 100** | **salt_2 u32 @ 104** | checksum_1 u32 | checksum_2 u32.
pub(crate) const TSHM_FRAME_INDEX_LEN_OFFSET: usize = 40;
pub(crate) const TSHM_SNAPSHOT_SEQ_OFFSET: usize = 48;
pub(crate) const TSHM_MAX_FRAME_OFFSET: usize = 56;
pub(crate) const TSHM_NBACKFILLS_OFFSET: usize = 64;
pub(crate) const TSHM_TRANSACTION_COUNT_OFFSET: usize = 72;
pub(crate) const TSHM_CHECKPOINT_SEQ_OFFSET: usize = 88;
pub(crate) const TSHM_PAGE_SIZE_OFFSET: usize = 96;
pub(crate) const TSHM_SALT1_OFFSET: usize = 100;
pub(crate) const TSHM_SALT2_OFFSET: usize = 104;
pub(crate) const TSHM_CHECKSUM1_OFFSET: usize = 108;
pub(crate) const TSHM_CHECKSUM2_OFFSET: usize = 112;
/// Minimum bytes to read to cover every field above (checksum_2 ends at 116).
pub(crate) const TSHM_HEADER_READ_LEN: usize = 116;

/// On-disk SQLite WAL header size (turso `WAL_HEADER_SIZE`).
const WAL_HEADER_SIZE: u64 = 32;
/// On-disk SQLite WAL frame header size (turso `WAL_FRAME_HEADER_SIZE`).
const WAL_FRAME_HEADER_SIZE: u64 = 24;
/// SQLite WAL magic values (turso `WAL_MAGIC_LE` / `WAL_MAGIC_BE`).
const WAL_MAGIC_LE: u32 = 0x377f_0682;
const WAL_MAGIC_BE: u32 = 0x377f_0683;

/// How often the background guard re-inspects the store directory.
const WAL_GUARD_INTERVAL_SECS: u64 = 60;

/// Re-announce a persistent non-healthy condition after this many consecutive
/// checks (avoids log spam while still surfacing a long-lived condition).
const REANNOUNCE_EVERY_CHECKS: u64 = 10;

/// Daemon-side `-tshm`/`-wal` open+close regression counter.
///
/// Every path-based coordination read (open + close) is counted. After the
/// persistent-fd fix the daemon's own loops never open+close these files, so
/// the wal-guard's periodic reading stays flat at the value left by the boot
/// pre-flight (which runs before any lock exists and is reset right after).
/// A rising value means a daemon-side open+close path was reintroduced.
static TSHM_OPEN_CLOSE_COUNT: AtomicU64 = AtomicU64::new(0);

/// Current daemon-side `-tshm`/`-wal` open+close count.
#[must_use]
pub(crate) fn tshm_open_close_count() -> u64 {
    TSHM_OPEN_CLOSE_COUNT.load(Ordering::Relaxed)
}

/// Reset the open+close counter (called by the boot pre-flight, which runs
/// before any store lock exists and is the only legitimate pre-loop reader).
pub(crate) fn reset_tshm_open_close_count() {
    TSHM_OPEN_CLOSE_COUNT.store(0, Ordering::Relaxed);
}

/// Parsed `.tshm` coordination header fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TshmHeader {
    /// Highest WAL frame number published by the current writer.
    pub max_frame: u64,
    /// Frames checkpointed from the head of the WAL.
    pub nbackfills: u64,
    /// Monotonic transaction counter.
    pub transaction_count: u64,
    /// Length of the shared frame index (may exceed `max_frame` after a crash
    /// mid-transaction — the stale-tail signal).
    pub frame_index_len: u32,
    /// Sequence lock protecting multi-field snapshot reads (even = stable).
    pub snapshot_seq: u64,
    /// WAL generation checkpoint sequence.
    pub checkpoint_seq: u32,
    /// Database page size.
    pub page_size: u32,
    /// WAL generation salts (mirrored by the WAL header while healthy).
    pub salt_1: u32,
    pub salt_2: u32,
    /// Last WAL frame's cumulative checksum (mirrored by the last frame's
    /// checksum fields while healthy — the Trusted-branch last-frame check).
    pub checksum_1: u32,
    pub checksum_2: u32,
}

/// Validated on-disk WAL header fields (turso `WalHeader`, 32 bytes BE).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalHeaderBytes {
    pub page_size: u32,
    pub checkpoint_seq: u32,
    pub salt_1: u32,
    pub salt_2: u32,
}

/// Last WAL frame's 24-byte header (BE) — the Trusted-branch last-frame check
/// (turso `WalFrameHeader`): commit frame, salts, and the cumulative checksum
/// mirrored by `.tshm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LastFrameHeader {
    /// Commit-frame size (`db_size`); non-zero marks a commit frame.
    pub commit_frame_size: u32,
    pub salt_1: u32,
    pub salt_2: u32,
    pub checksum_1: u32,
    pub checksum_2: u32,
}

/// Classification of one store's coordination state.
///
/// Mirrors turso's `classify_authority_snapshot_against_wal` (turso_core 0.7.2)
/// with the documented deviations in the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalStateClass {
    /// `max_frame == 0` (healthy regardless of WAL size) or a WAL with the
    /// exact expected length whose header salts match `.tshm`.
    Healthy,
    /// `max_frame > 0` with a WAL smaller than the 32-byte header — the
    /// on-disk WAL was truncated/removed while live frames are advertised.
    Orphaned,
    /// `max_frame > 0` with a WAL whose header fails validation (magic /
    /// checksum / page_size) — a regrown-zeroed-header foreign WAL.
    OrphanedForeign,
    /// `max_frame > 0` with a valid WAL header whose page_size/checkpoint_seq/
    /// salts differ from `.tshm` — a foreign WAL from another generation.
    Foreign,
    /// `max_frame > 0` with a matching header but a WAL shorter than
    /// `32 + max_frame*(24+page_size)` — truncated mid-generation.
    TruncatedMidGen,
    /// `max_frame > 0` with a WAL longer than expected — warn-only.
    Oversized,
    /// `frame_index_len > max_frame` (crash mid-transaction; healed by a
    /// TRUNCATE-first checkpoint at store init).
    TornPre,
    /// `.tshm` missing, too small, or failing magic/version validation.
    Unreadable,
    /// Exact expected length + matching header salts, but the last frame's
    /// commit/salt/checksum disagrees with `.tshm` — a regrown WAL with a
    /// copied header. Full-predicate only; mirrors turso's Trusted-branch
    /// last-frame validation (`LastFrameNotCommit`/`LastFrameSaltMismatch`/
    /// `LastFrameChecksumMismatch`).
    LastFrameMismatch,
}

impl WalStateClass {
    /// True for the classes that must block checkpoints (2/3/4/6 plus the
    /// full-predicate regrown-WAL class): checkpointing them would zero-fill
    /// the DB or touch a foreign WAL. Blocking healthy stores would starve the
    /// only WAL-shrinking mechanism, so the subset is deliberately narrow; the
    /// wal-guard loop still warns on every non-healthy class.
    #[must_use]
    pub(crate) fn blocks_checkpoint(self) -> bool {
        matches!(
            self,
            Self::Orphaned
                | Self::OrphanedForeign
                | Self::Foreign
                | Self::TruncatedMidGen
                | Self::LastFrameMismatch
        )
    }

    /// Short stable label for logs and the `debug detect` output.
    #[must_use]
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Orphaned => "orphaned-wal",
            Self::OrphanedForeign => "orphaned-foreign-wal",
            Self::Foreign => "foreign-wal",
            Self::TruncatedMidGen => "truncated-wal",
            Self::Oversized => "oversized-wal",
            Self::TornPre => "torn-pre-index",
            Self::Unreadable => "unreadable-tshm",
            Self::LastFrameMismatch => "last-frame-mismatch",
        }
    }
}

/// Read sources for one store's coordination files.
///
/// The daemon passes the persistent per-store fds from
/// [`crate::db::Connection`]; processes that hold no locks (CLI, boot
/// pre-flight, tests) pass `StoreFds::none()` and the path-based fallback
/// (open+close, counted by the regression counter) is used.
#[derive(Debug, Clone, Copy, Default)]
pub struct StoreFds<'a> {
    /// Persistent read-only fd to the store's `-tshm` file.
    pub tshm: Option<&'a File>,
    /// Persistent read-only fd to the store's `-wal` file.
    pub wal: Option<&'a File>,
}

impl StoreFds<'_> {
    #[must_use]
    pub(crate) const fn none() -> Self {
        Self {
            tshm: None,
            wal: None,
        }
    }
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
    /// Validated on-disk WAL header, when readable.
    pub wal_header: Option<WalHeaderBytes>,
    /// Coordination-state classification (see [`WalStateClass`]).
    pub class: WalStateClass,
}

/// Parse a `.tshm` coordination header from raw bytes.
///
/// Returns `None` when the buffer is too small or fails magic/version
/// validation (i.e. not a live turso multiprocess-WAL coordination file).
#[must_use]
pub(crate) fn parse_tshm_header(bytes: &[u8]) -> Option<TshmHeader> {
    if bytes.len() < TSHM_HEADER_READ_LEN {
        return None;
    }
    if bytes[0..8] != TSHM_MAGIC {
        return None;
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
    if version != TSHM_VERSION {
        return None;
    }
    // Reader-slot multiplicity: turso opens the coordination map only for
    // `reader_slot_count >= 64 && % 64 == 0` (assert in
    // `create_or_open_with_mode`); a corrupted count is structural damage.
    let reader_slots = u32::from_le_bytes(bytes[12..16].try_into().ok()?);
    if reader_slots < 64 || reader_slots % 64 != 0 {
        return None;
    }
    Some(TshmHeader {
        frame_index_len: u32::from_le_bytes(
            bytes[TSHM_FRAME_INDEX_LEN_OFFSET..TSHM_FRAME_INDEX_LEN_OFFSET + 4]
                .try_into()
                .ok()?,
        ),
        snapshot_seq: u64::from_le_bytes(
            bytes[TSHM_SNAPSHOT_SEQ_OFFSET..TSHM_SNAPSHOT_SEQ_OFFSET + 8]
                .try_into()
                .ok()?,
        ),
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
        checkpoint_seq: u32::from_le_bytes(
            bytes[TSHM_CHECKPOINT_SEQ_OFFSET..TSHM_CHECKPOINT_SEQ_OFFSET + 4]
                .try_into()
                .ok()?,
        ),
        page_size: u32::from_le_bytes(
            bytes[TSHM_PAGE_SIZE_OFFSET..TSHM_PAGE_SIZE_OFFSET + 4]
                .try_into()
                .ok()?,
        ),
        salt_1: u32::from_le_bytes(
            bytes[TSHM_SALT1_OFFSET..TSHM_SALT1_OFFSET + 4]
                .try_into()
                .ok()?,
        ),
        salt_2: u32::from_le_bytes(
            bytes[TSHM_SALT2_OFFSET..TSHM_SALT2_OFFSET + 4]
                .try_into()
                .ok()?,
        ),
        checksum_1: u32::from_le_bytes(
            bytes[TSHM_CHECKSUM1_OFFSET..TSHM_CHECKSUM1_OFFSET + 4]
                .try_into()
                .ok()?,
        ),
        checksum_2: u32::from_le_bytes(
            bytes[TSHM_CHECKSUM2_OFFSET..TSHM_CHECKSUM2_OFFSET + 4]
                .try_into()
                .ok()?,
        ),
    })
}

/// Validate a 32-byte on-disk WAL header (magic, page size, checksum) and
/// return its fields. Mirrors turso's `read_validated_wal_header_from_file`.
#[must_use]
pub(crate) fn parse_wal_header(bytes: &[u8; 32]) -> Option<WalHeaderBytes> {
    let magic = u32::from_be_bytes(bytes[0..4].try_into().ok()?);
    if !matches!(magic, WAL_MAGIC_LE | WAL_MAGIC_BE) {
        return None;
    }
    let page_size = u32::from_be_bytes(bytes[8..12].try_into().ok()?);
    if !(512..=65_536).contains(&page_size) || !page_size.is_power_of_two() {
        return None;
    }
    // SQLite interprets the big-endian magic as "native endian" for the
    // checksum word ordering (turso `checksum_wal`).
    let native_endian = cfg!(target_endian = "big") == (magic & 1 != 0);
    let (calc1, calc2) = checksum_wal_prefix(&bytes[..24], native_endian);
    let stored1 = u32::from_be_bytes(bytes[24..28].try_into().ok()?);
    let stored2 = u32::from_be_bytes(bytes[28..32].try_into().ok()?);
    if calc1 != stored1 || calc2 != stored2 {
        return None;
    }
    Some(WalHeaderBytes {
        page_size,
        checkpoint_seq: u32::from_be_bytes(bytes[12..16].try_into().ok()?),
        salt_1: u32::from_be_bytes(bytes[16..20].try_into().ok()?),
        salt_2: u32::from_be_bytes(bytes[20..24].try_into().ok()?),
    })
}

/// SQLite WAL checksum over a byte prefix (must be a multiple of 8), starting
/// from a zero seed. Mirrors turso's `checksum_wal` for the header case.
fn checksum_wal_prefix(buf: &[u8], native_endian: bool) -> (u32, u32) {
    debug_assert_eq!(buf.len() % 8, 0);
    let mut s0 = 0u32;
    let mut s1 = 0u32;
    let mut i = 0;
    while i < buf.len() {
        let v0 = u32::from_ne_bytes(buf[i..i + 4].try_into().expect("8-byte aligned"));
        let v1 = u32::from_ne_bytes(buf[i + 4..i + 8].try_into().expect("8-byte aligned"));
        let (v0, v1) = if native_endian {
            (v0, v1)
        } else {
            (v0.swap_bytes(), v1.swap_bytes())
        };
        s0 = s0.wrapping_add(v0.wrapping_add(s1));
        s1 = s1.wrapping_add(v1.wrapping_add(s0));
        i += 8;
    }
    (s0, s1)
}

/// Classify the coordination state of one store.
///
/// Pure function over the parsed `.tshm` header, the on-disk WAL size, and the
/// validated WAL header — shared by the checkpoint loop, the wal-guard loop,
/// the boot pre-flight, and the `debug detect` CLI verb so the consumers
/// cannot drift. `last_frame` carries the last-frame header read by
/// [`inspect_store_at`]; `None` runs the predicate without the Trusted-branch
/// last-frame check (tests and the pre-last-frame pass — production callers
/// always supply it via [`inspect_store_at`], which re-runs the classification
/// with the last frame once the length+salt cases pass).
///
/// Deliberate extension of the checkpoint-blocking subset: `LastFrameMismatch`
/// (case 5's Trusted-branch failure) blocks checkpoints in production even
/// though the minimal blocking set {2,3,4,6} excludes it — a regrown WAL with
/// a copied header and exact length must not be checkpointed.
#[must_use]
pub(crate) fn classify_wal_state(
    tshm: Option<TshmHeader>,
    wal_size: u64,
    wal_header: Option<WalHeaderBytes>,
    last_frame: Option<LastFrameHeader>,
) -> WalStateClass {
    let Some(h) = tshm else {
        return WalStateClass::Unreadable;
    };
    // (1) max_frame == 0 → healthy regardless of WAL size (deliberately looser
    // than turso — the quiet window after a checkpoint must stay non-blocking;
    // turso's own open performs the stricter rebuild).
    if h.max_frame == 0 {
        return WalStateClass::Healthy;
    }
    // (2) max_frame > 0 with a WAL smaller than the 32-byte header.
    if wal_size < WAL_HEADER_SIZE {
        return WalStateClass::Orphaned;
    }
    // (3) unreadable WAL header (magic/checksum/page_size) → foreign zeroed
    // header regrown under the daemon.
    let Some(wal) = wal_header else {
        return WalStateClass::OrphanedForeign;
    };
    // (4) readable header whose generation fields disagree with `.tshm`.
    if wal.page_size != h.page_size
        || wal.checkpoint_seq != h.checkpoint_seq
        || wal.salt_1 != h.salt_1
        || wal.salt_2 != h.salt_2
    {
        return WalStateClass::Foreign;
    }
    let frame_size = WAL_FRAME_HEADER_SIZE + u64::from(wal.page_size);
    // Saturating: a pathologically corrupt `.tshm` max_frame (u64::MAX) must
    // not panic the arithmetic — it classifies as truncated (blocking) below.
    let expected = WAL_HEADER_SIZE.saturating_add(h.max_frame.saturating_mul(frame_size));
    // (6)/(7) length vs the max_frame-implied expectation — checked before (8)
    // so a stale tail whose frames are gone (wal < expected) blocks as
    // truncated instead of passing as non-blocking TornPre (the zero-fill
    // scenario the blocking subset exists to prevent).
    if wal_size < expected {
        return WalStateClass::TruncatedMidGen;
    }
    if wal_size > expected {
        return WalStateClass::Oversized;
    }
    // (8) stale-tail: the shared frame index outlives the committed frames.
    // Checked after the length cases so mid-write states (which are Oversized)
    // never block; a crash-persistent stale tail has wal == expected and is
    // healed by a TRUNCATE-first checkpoint at the next store open.
    if u64::from(h.frame_index_len) > h.max_frame {
        return WalStateClass::TornPre;
    }
    // (5) exact length + header-salt match. The full predicate additionally
    // mirrors turso's Trusted branch: the last frame must be a commit frame
    // whose salts and cumulative checksum match `.tshm` — a regrown WAL with a
    // copied header and exact length fails this and must not be checkpointed.
    if let Some(frame) = last_frame
        && (frame.commit_frame_size == 0
            || frame.salt_1 != h.salt_1
            || frame.salt_2 != h.salt_2
            || frame.checksum_1 != h.checksum_1
            || frame.checksum_2 != h.checksum_2)
    {
        return WalStateClass::LastFrameMismatch;
    }
    WalStateClass::Healthy
}

/// Path-based `.tshm` header read (open + close).
///
/// Only safe in processes that hold no fcntl locks on the file (CLI, boot
/// pre-flight, tests). Counted by [`tshm_open_close_count`] — the daemon's
/// loops must go through the persistent fds instead.
fn read_tshm_bytes_path(tshm_path: &Path) -> Option<[u8; TSHM_HEADER_READ_LEN]> {
    use std::io::Read;
    let mut file = std::fs::File::open(tshm_path).ok()?;
    TSHM_OPEN_CLOSE_COUNT.fetch_add(1, Ordering::Relaxed);
    let mut bytes = [0u8; TSHM_HEADER_READ_LEN];
    file.read_exact(&mut bytes).ok()?;
    Some(bytes)
}

/// Path-based WAL header read (open + close) — same lock-drop caveat as
/// [`read_tshm_bytes_path`].
fn read_wal_bytes_path(wal_path: &Path) -> Option<[u8; 32]> {
    use std::io::Read;
    let mut file = std::fs::File::open(wal_path).ok()?;
    TSHM_OPEN_CLOSE_COUNT.fetch_add(1, Ordering::Relaxed);
    let mut bytes = [0u8; 32];
    file.read_exact(&mut bytes).ok()?;
    Some(bytes)
}

/// Read the last WAL frame's 24-byte header (big-endian) at
/// `32 + (max_frame-1) * (24 + page_size)`. Mirrors turso's Trusted-branch
/// last-frame validation — a single-frame check (no O(maxf) scan), safe under
/// NoLock semantics via the persistent fd when available.
fn read_last_frame_header(
    wal_fd: Option<&File>,
    wal_path: &Path,
    tshm: &TshmHeader,
    wal: &WalHeaderBytes,
) -> Option<LastFrameHeader> {
    let frame_size = WAL_FRAME_HEADER_SIZE + u64::from(wal.page_size);
    // Saturating: a pathologically corrupt max_frame must not panic (the
    // pread then fails and the caller classifies TruncatedMidGen).
    let offset =
        WAL_HEADER_SIZE.saturating_add(tshm.max_frame.saturating_sub(1).saturating_mul(frame_size));
    let bytes: [u8; 24] = if let Some(f) = wal_fd {
        crate::db::pread_at(f, offset)?
    } else {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(wal_path).ok()?;
        TSHM_OPEN_CLOSE_COUNT.fetch_add(1, Ordering::Relaxed);
        file.seek(SeekFrom::Start(offset)).ok()?;
        let mut buf = [0u8; 24];
        file.read_exact(&mut buf).ok()?;
        buf
    };
    Some(LastFrameHeader {
        commit_frame_size: u32::from_be_bytes(bytes[4..8].try_into().ok()?),
        salt_1: u32::from_be_bytes(bytes[8..12].try_into().ok()?),
        salt_2: u32::from_be_bytes(bytes[12..16].try_into().ok()?),
        checksum_1: u32::from_be_bytes(bytes[16..20].try_into().ok()?),
        checksum_2: u32::from_be_bytes(bytes[20..24].try_into().ok()?),
    })
}

/// Read a `.tshm` header with snapshot-seq discipline: read twice, retrying
/// while the snapshot sequence is odd (a writer is mid-publish), and require
/// an even sequence to be stable across the two reads so the field group is
/// not torn.
fn read_tshm_disciplined(
    read: impl Fn() -> Option<[u8; TSHM_HEADER_READ_LEN]>,
) -> Option<TshmHeader> {
    for _ in 0..4 {
        let first = read()?;
        let h1 = parse_tshm_header(&first)?;
        if h1.snapshot_seq % 2 != 0 {
            continue;
        }
        let second = read()?;
        let h2 = parse_tshm_header(&second)?;
        if h2.snapshot_seq == h1.snapshot_seq {
            return Some(h2);
        }
    }
    None
}

/// Classify the file set of one store given its main database file path.
///
/// The store name is derived from the file name (`core.db` → `core`).
/// This is a pure filesystem inspection — it never opens the database, so it
/// is safe to run against live stores and is unit-testable with synthetic
/// file states. `fds` provides the persistent per-store fds when the caller
/// is the daemon (see [`StoreFds`]).
#[must_use]
pub fn inspect_store_at(db_path: &Path, fds: StoreFds<'_>) -> StoreArtifactStatus {
    let sidecars = crate::db::store_sidecars(db_path);
    let tshm = read_tshm_disciplined(|| match fds.tshm {
        Some(f) => crate::db::pread::<TSHM_HEADER_READ_LEN>(f),
        None => read_tshm_bytes_path(&sidecars.tshm),
    });
    let wal_size = std::fs::metadata(&sidecars.wal).map_or(0, |m| m.len());
    let wal_header = match fds.wal {
        Some(f) => crate::db::pread::<32>(f),
        None => read_wal_bytes_path(&sidecars.wal),
    };
    let wal_header = wal_header.as_ref().and_then(parse_wal_header);
    // Full predicate: when the narrow classification reaches case 5 (exact
    // length + header-salt match with live frames), apply turso's
    // Trusted-branch last-frame check before declaring the store healthy.
    let mut class = classify_wal_state(tshm, wal_size, wal_header, None);
    if class == WalStateClass::Healthy
        && let (Some(h), Some(w)) = (tshm.filter(|h| h.max_frame > 0), wal_header)
    {
        let last_frame = read_last_frame_header(fds.wal, &sidecars.wal, &h, &w);
        class = match last_frame {
            Some(frame) => classify_wal_state(tshm, wal_size, wal_header, Some(frame)),
            // The WAL shrank between the size check and the frame read — the
            // Trusted-branch validation cannot run and a truncated WAL must
            // not classify Healthy (turso maps the missing last frame to
            // Rebuild).
            None => WalStateClass::TruncatedMidGen,
        };
    }
    let store = db_path.file_stem().map_or_else(
        || db_path.display().to_string(),
        |s| s.to_string_lossy().into_owned(),
    );
    StoreArtifactStatus {
        store,
        tshm,
        wal_size,
        wal_header,
        class,
    }
}

/// Classify the file set of one store under `root/db/`.
#[must_use]
pub fn inspect_store(root: &Path, name: &str, fds: StoreFds<'_>) -> StoreArtifactStatus {
    inspect_store_at(&crate::db::store_db_path(root, name), fds)
}

// ── Boot pre-flight diagnosis ─────────────────────────────────────────────

/// Per-store classification captured by the boot pre-flight scan,
/// before any store is opened. The heal strategy flows from this map —
/// turso's own reopen (RebuildFromDisk → install_snapshot) would consume the
/// evidence, so the strategy must not be re-derived post-open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BootDiagnosis {
    /// No damaged state (fresh install, post-TRUNCATE, or a healthy store).
    Healthy,
    /// `frame_index_len > max_frame` with `max_frame > 0` — crash
    /// mid-transaction; healed by a TRUNCATE-first checkpoint.
    StaleTail,
    /// 0-byte main DB with a non-empty WAL / live frames — durable-B; healed
    /// by PASSIVE-first (backfill), reopen, then TRUNCATE.
    DurableB,
    /// A coordination class that blocks a safe reopen (orphaned/foreign/
    /// truncated WAL, regrown-header) — not healed; turso's own reopen
    /// rebuilds the `.tshm` from the WAL. The checkpoint loop blocks these
    /// only while the store is closed (the reopen consumes the evidence).
    BlockedCoordination,
    /// Structural damage (bad `.tshm` magic/version/size, truncated/zeroed
    /// main-DB header) — recreate is the only option.
    Structural,
}

impl BootDiagnosis {
    #[must_use]
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::StaleTail => "stale-tail",
            Self::DurableB => "durable-b",
            Self::BlockedCoordination => "blocked-coordination",
            Self::Structural => "structural",
        }
    }
}

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
        // first). A fresh install (0-byte + empty WAL + no .tshm) is healthy.
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
/// The result feeds the per-store heal strategy in `turso::open_store`. The
/// open+close regression counter is reset afterwards so the wal-guard's first
/// reading reflects only post-startup paths.
pub fn diagnose_all_stores(root: &Path) {
    for (name, _) in crate::db::iter_checkpoint_stores() {
        let db_path = crate::db::store_db_path(root, name);
        let sidecars = crate::db::store_sidecars(&db_path);
        let status = inspect_store_at(&db_path, StoreFds::none());
        let wal_size = status.wal_size;

        // Coordination-state first; the reopen (not this scan) performs the
        // heal/rebuild. Oversized is warn-only (post-crash uncommitted tail),
        // but the reopen is still the risky turso path, so it shares the
        // BlockedCoordination bucket — checkpoints are NOT blocked for it.
        let diagnosis = match status.class {
            WalStateClass::Healthy => {
                // Coordination healthy — inspect the main DB for durable-B /
                // structural damage.
                classify_main_db(&db_path, sidecars.wal.exists(), wal_size)
            }
            WalStateClass::TornPre => {
                // frame_index_len > max_frame (crash mid-transaction). A
                // 0-byte main DB with a non-empty WAL in this state is
                // durable-B (crash mid-first-transaction) — TRUNCATE-first
                // would destroy the frames, so route to PASSIVE-first.
                // A structurally-damaged main DB stays Structural (the label
                // must not be masked by the stale-tail fallback).
                match classify_main_db(&db_path, sidecars.wal.exists(), wal_size) {
                    BootDiagnosis::DurableB => BootDiagnosis::DurableB,
                    BootDiagnosis::Structural => BootDiagnosis::Structural,
                    _ => BootDiagnosis::StaleTail,
                }
            }
            WalStateClass::Unreadable => {
                // Invalid/missing .tshm. A missing .tshm on a store that has a
                // main DB is either a snapshot copy (fine) or structural; an
                // existing-but-invalid .tshm is structural.
                if sidecars.tshm.exists() {
                    BootDiagnosis::Structural
                } else {
                    classify_main_db(&db_path, sidecars.wal.exists(), wal_size)
                }
            }
            WalStateClass::Oversized => {
                // Warn-only: a valid WAL longer than the max_frame-implied
                // size (post-crash uncommitted tail — the bytes are present).
                // turso's reopen rebuilds over it; the panic-absorbed open is
                // the risky step, hence the shared bucket.
                BootDiagnosis::BlockedCoordination
            }
            WalStateClass::LastFrameMismatch => {
                // Regrown WAL with a copied header — the reopen rebuilds over
                // the intact main DB; committed frames are fsync-durable.
                BootDiagnosis::BlockedCoordination
            }
            _ => BootDiagnosis::BlockedCoordination,
        };
        set_boot_diagnosis(&db_path, diagnosis);
        if diagnosis != BootDiagnosis::Healthy {
            crate::boot::boot_diagnostic(format!(
                "boot pre-flight: store '{name}' class {} (max_frame={}, wal_size={}) — \
                 healing will run before open",
                diagnosis.label(),
                status.tshm.map_or(0, |h| h.max_frame),
                wal_size,
            ));
        }
    }
    reset_tshm_open_close_count();
}

/// Background loop: periodically inspect the live store directory and emit
/// warnings when the coordination state is non-healthy.
///
/// Warns on every non-healthy class (full predicate — the checkpoint loop
/// uses the narrower blocking subset). Non-healthy classes are only announced
/// after two consecutive observations (the two-observation rule for runtime
/// candidates — TornPre fires transiently while a writer is
/// mid-flush, since frame_index_len is bumped per frame but max_frame only at
/// commit), then re-announced every [`REANNOUNCE_EVERY_CHECKS`] checks while
/// persistent. Also re-checks the persistent sidecar fds against the paths
/// (an external replacement means the fds read stale inodes — announced on
/// the same throttle) and logs the daemon-side `.tshm`/`-wal` open+close
/// regression counter (see [`tshm_open_close_count`]).
pub async fn run_wal_guard_loop() {
    let root = match crate::config::default_config_dir() {
        Ok(root) => root,
        Err(e) => {
            warn!(error = %e, "wal-guard: cannot resolve storage root; guard disabled");
            return;
        }
    };

    // (class, consecutive-observation count) per store.
    let mut seen: std::collections::HashMap<String, (WalStateClass, u64)> =
        std::collections::HashMap::new();
    // (identity condition, consecutive-observation count) per store — keyed
    // per condition so a Replaced→Deleted transition restarts the count.
    let mut identity_seen: std::collections::HashMap<String, (crate::db::SidecarIdentity, u64)> =
        std::collections::HashMap::new();
    let mut check_count: u64 = 0;

    loop {
        if !crate::shutdown::sleep_or_shutdown_or_drain(Duration::from_secs(
            WAL_GUARD_INTERVAL_SECS,
        ))
        .await
        {
            break;
        }
        check_count += 1;
        for (name, conn) in crate::db::iter_checkpoint_stores() {
            let identity = conn.and_then(crate::db::Connection::check_coordination_identity);
            // Replaced: the persistent fds read stale inodes — skip the
            // inspect entirely. A Deleted sidecar is inspected instead: the
            // fd still reads the old inode while wal_size comes from the path,
            // and that coupling classifies the resulting orphaned state
            // correctly (the Replaced branch skips for exactly the
            // stale-inode reason).
            if identity == Some(crate::db::SidecarIdentity::Replaced) {
                announce_identity(
                    &mut identity_seen,
                    name,
                    crate::db::SidecarIdentity::Replaced,
                    check_count,
                );
                continue;
            }
            let fds = conn.map_or_else(StoreFds::none, crate::db::Connection::store_fds);
            let status = inspect_store(&root, name, fds);
            let store = status.store.clone();
            // The Deleted identity warn fires only when the class is healthy
            // (a deleted sidecar with no live frames is invisible to the
            // predicate) — a non-healthy class already announces on the class
            // throttle, so the loop stays at one warn per cycle (matching the
            // Replaced branch's single skip-warn).
            if identity == Some(crate::db::SidecarIdentity::Deleted)
                && status.class == WalStateClass::Healthy
            {
                announce_identity(
                    &mut identity_seen,
                    name,
                    crate::db::SidecarIdentity::Deleted,
                    check_count,
                );
            } else {
                identity_seen.remove(name);
            }
            if status.class == WalStateClass::Healthy {
                seen.remove(&store);
                continue;
            }
            let count = seen
                .get(&store)
                .filter(|(c, _)| *c == status.class)
                .map_or(1, |(_, n)| n + 1);
            seen.insert(store, (status.class, count));
            let announce =
                count == 2 || (count > 2 && check_count.is_multiple_of(REANNOUNCE_EVERY_CHECKS));
            if announce {
                warn!(
                    store = %status.store,
                    class = status.class.label(),
                    max_frame = status.tshm.map_or(0, |h| h.max_frame),
                    wal_size = status.wal_size,
                    "wal-guard: non-healthy coordination state ({}) — {}",
                    status.class.label(),
                    class_warning(status.class),
                );
            }
        }
        if check_count.is_multiple_of(REANNOUNCE_EVERY_CHECKS) {
            let open_close_count = tshm_open_close_count();
            if open_close_count == 0 {
                // Zero is the expected post-boot state — quiet at DEBUG. A
                // nonzero count is the regression signal and stays at INFO.
                debug!(
                    tshm_open_close_count = open_close_count,
                    "wal-guard: daemon-side coordination open+close count (0 expected post-boot — \
                     the daemon-side open+close regression signal)",
                );
            } else {
                info!(
                    tshm_open_close_count = open_close_count,
                    "wal-guard: daemon-side coordination open+close count (0 expected post-boot — \
                     the daemon-side open+close regression signal)",
                );
            }
        }
    }
}

/// Throttled identity announcement: 2 consecutive observations, then
/// re-announce every [`REANNOUNCE_EVERY_CHECKS`] checks. Keyed per
/// (store, condition) — a condition change restarts the count.
fn announce_identity(
    identity_seen: &mut std::collections::HashMap<String, (crate::db::SidecarIdentity, u64)>,
    name: &str,
    identity: crate::db::SidecarIdentity,
    check_count: u64,
) {
    let entry = identity_seen
        .entry(name.to_string())
        .or_insert((identity, 0));
    if entry.0 != identity {
        *entry = (identity, 0);
    }
    entry.1 += 1;
    if entry.1 == 2 || (entry.1 > 2 && check_count.is_multiple_of(REANNOUNCE_EVERY_CHECKS)) {
        match identity {
            crate::db::SidecarIdentity::Replaced => warn!(
                db = %name,
                "wal-guard: coordination files replaced by an external process \
                 (inode mismatch) — checks suspended for this store until restart",
            ),
            crate::db::SidecarIdentity::Deleted => warn!(
                db = %name,
                "wal-guard: coordination sidecar deleted by an external process \
                 (unlinked under the daemon) — the predicate will detect any \
                 resulting orphaned-WAL state",
            ),
            _ => {}
        }
    }
}

/// Per-class warning text for the wal-guard announcement — the generic
/// "foreign standard-SQLite activity" wording is only accurate for the
/// foreign/orphaned family, not for crash states or warn-only conditions.
fn class_warning(class: WalStateClass) -> &'static str {
    match class {
        WalStateClass::TornPre => {
            "frame index exceeds max_frame (crash stale-tail or in-flight write) — \
             healed by a TRUNCATE-first checkpoint at the next store open"
        }
        WalStateClass::Oversized => "WAL is larger than the max_frame-implied size (warn-only)",
        WalStateClass::Unreadable => {
            ".tshm is unreadable — coordination state cannot be classified"
        }
        WalStateClass::LastFrameMismatch => {
            "regrown WAL with a copied header (last frame fails commit/salt/checksum) — \
             checkpoints blocked"
        }
        _ => {
            "foreign standard-SQLite activity likely replaced -wal/-tshm under the daemon. \
             Reads through .tshm may hit torn-frame errors. Query snapshot copies; never \
             delete/recreate -wal/-shm/-tshm while the daemon runs."
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
        // Reader-slot multiplicity: turso requires >= 64 and % 64 == 0.
        bytes[12..16].copy_from_slice(&64u32.to_le_bytes());
        bytes[TSHM_FRAME_INDEX_LEN_OFFSET..TSHM_FRAME_INDEX_LEN_OFFSET + 4]
            .copy_from_slice(&max_frame.to_le_bytes()[..4]);
        bytes[TSHM_MAX_FRAME_OFFSET..TSHM_MAX_FRAME_OFFSET + 8]
            .copy_from_slice(&max_frame.to_le_bytes());
        bytes[TSHM_NBACKFILLS_OFFSET..TSHM_NBACKFILLS_OFFSET + 8]
            .copy_from_slice(&nbackfills.to_le_bytes());
        bytes[TSHM_TRANSACTION_COUNT_OFFSET..TSHM_TRANSACTION_COUNT_OFFSET + 8]
            .copy_from_slice(&tx_count.to_le_bytes());
        bytes
    }

    /// Build a valid WAL header (32 bytes) for a given page_size/salts.
    fn wal_header_bytes(page_size: u32, salt_1: u32, salt_2: u32) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(&WAL_MAGIC_LE.to_be_bytes());
        bytes[8..12].copy_from_slice(&page_size.to_be_bytes());
        bytes[16..20].copy_from_slice(&salt_1.to_be_bytes());
        bytes[20..24].copy_from_slice(&salt_2.to_be_bytes());
        // Mirror turso's use_native_endian expression: on this machine a LE
        // magic means the words are native-endian.
        let native_endian = cfg!(target_endian = "big") == (WAL_MAGIC_LE & 1 != 0);
        let (c1, c2) = checksum_wal_prefix(&bytes[..24], native_endian);
        bytes[24..28].copy_from_slice(&c1.to_be_bytes());
        bytes[28..32].copy_from_slice(&c2.to_be_bytes());
        bytes
    }

    #[test]
    fn parses_valid_tshm_header() {
        let bytes = tshm_bytes(42, 3, 99);
        let hdr = parse_tshm_header(&bytes).expect("valid tshm header parses");
        assert_eq!(hdr.max_frame, 42);
        assert_eq!(hdr.nbackfills, 3);
        assert_eq!(hdr.transaction_count, 99);
        assert_eq!(hdr.frame_index_len, 42); // fixture sets it to max_frame
        assert_eq!(hdr.page_size, 0); // zeroed in fixture
        assert_eq!(hdr.salt_1, 0);
        assert_eq!(hdr.salt_2, 0);
    }

    #[test]
    fn rejects_missing_bad_or_short_tshm() {
        // Wrong magic.
        let mut bytes = tshm_bytes(1, 0, 0);
        bytes[0] = b'X';
        assert!(parse_tshm_header(&bytes).is_none());
        // Too short.
        assert!(parse_tshm_header(&bytes[..16]).is_none());
    }

    #[test]
    fn wal_header_validation_rejects_bad_magic_and_checksum() {
        let good = wal_header_bytes(4096, 7, 11);
        assert!(parse_wal_header(&good).is_some());
        let mut bad_magic = good;
        bad_magic[0] = 0;
        assert!(parse_wal_header(&bad_magic).is_none());
        let mut bad_checksum = good;
        bad_checksum[24] ^= 0xff;
        assert!(parse_wal_header(&bad_checksum).is_none());
        // Invalid page size.
        let mut bad_size = good;
        bad_size[8..12].copy_from_slice(&123u32.to_be_bytes());
        assert!(parse_wal_header(&bad_size).is_none());
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

    fn tshm(max_frame: u64, frame_index_len: u32) -> TshmHeader {
        TshmHeader {
            max_frame,
            nbackfills: 0,
            transaction_count: 0,
            frame_index_len,
            snapshot_seq: 0,
            checkpoint_seq: 0,
            page_size: 4096,
            salt_1: 7,
            salt_2: 11,
            checksum_1: 0,
            checksum_2: 0,
        }
    }

    /// A last-frame header matching the fixture `tshm()` salts (checksums 0).
    fn last_frame(commit: u32, salt_1: u32, salt_2: u32) -> LastFrameHeader {
        LastFrameHeader {
            commit_frame_size: commit,
            salt_1,
            salt_2,
            checksum_1: 0,
            checksum_2: 0,
        }
    }

    #[test]
    fn classify_matches_tshm_wal_states() {
        let healthy = tshm(120, 120);
        let wal = parse_wal_header(&wal_header_bytes(4096, 7, 11)).unwrap();
        let frame_size = 4096 + 24;
        // Healthy: exact expected length + matching salts (+ last frame OK
        // in the full predicate).
        assert_eq!(
            classify_wal_state(Some(healthy), 32 + 120 * frame_size, Some(wal), None),
            WalStateClass::Healthy
        );
        assert_eq!(
            classify_wal_state(
                Some(healthy),
                32 + 120 * frame_size,
                Some(wal),
                Some(last_frame(120, 7, 11))
            ),
            WalStateClass::Healthy
        );
        // Full predicate: last frame fails commit/salt/checksum → regrown WAL.
        assert_eq!(
            classify_wal_state(
                Some(healthy),
                32 + 120 * frame_size,
                Some(wal),
                Some(last_frame(0, 7, 11))
            ),
            WalStateClass::LastFrameMismatch
        );
        assert_eq!(
            classify_wal_state(
                Some(healthy),
                32 + 120 * frame_size,
                Some(wal),
                Some(last_frame(120, 99, 11))
            ),
            WalStateClass::LastFrameMismatch
        );
        // (1) max_frame == 0 → healthy regardless of WAL size (quiet window).
        assert_eq!(
            classify_wal_state(Some(tshm(0, 0)), 0, None, None),
            WalStateClass::Healthy
        );
        assert_eq!(
            classify_wal_state(Some(tshm(0, 0)), 4096, Some(wal), None),
            WalStateClass::Healthy
        );
        // (8) stale-tail: frame index outlives committed frames.
        assert_eq!(
            classify_wal_state(Some(tshm(120, 130)), 32 + 120 * frame_size, Some(wal), None),
            WalStateClass::TornPre
        );
        // (2) max_frame > 0 with a WAL below the 32-byte header → orphaned.
        assert_eq!(
            classify_wal_state(Some(healthy), 0, None, None),
            WalStateClass::Orphaned
        );
        assert_eq!(
            classify_wal_state(Some(healthy), 16, None, None),
            WalStateClass::Orphaned
        );
        // (3) unreadable header (zeroed after truncate + regrow) → foreign.
        assert_eq!(
            classify_wal_state(Some(healthy), 4096, None, None),
            WalStateClass::OrphanedForeign
        );
        // (4) valid header with mismatched generation fields → foreign.
        let foreign = parse_wal_header(&wal_header_bytes(4096, 99, 11)).unwrap();
        assert_eq!(
            classify_wal_state(Some(healthy), 32 + 120 * frame_size, Some(foreign), None),
            WalStateClass::Foreign
        );
        // (6) truncated mid-generation.
        assert_eq!(
            classify_wal_state(Some(healthy), 32 + 50 * frame_size, Some(wal), None),
            WalStateClass::TruncatedMidGen
        );
        // (7) oversized → warn-only.
        assert_eq!(
            classify_wal_state(Some(healthy), 32 + 121 * frame_size, Some(wal), None),
            WalStateClass::Oversized
        );
        // Unreadable .tshm.
        assert_eq!(
            classify_wal_state(None, 0, None, None),
            WalStateClass::Unreadable
        );
    }

    #[test]
    fn blocking_subset_is_only_2_3_4_6_plus_last_frame() {
        assert!(WalStateClass::Orphaned.blocks_checkpoint());
        assert!(WalStateClass::OrphanedForeign.blocks_checkpoint());
        assert!(WalStateClass::Foreign.blocks_checkpoint());
        assert!(WalStateClass::TruncatedMidGen.blocks_checkpoint());
        assert!(WalStateClass::LastFrameMismatch.blocks_checkpoint());
        assert!(!WalStateClass::Healthy.blocks_checkpoint());
        assert!(!WalStateClass::Oversized.blocks_checkpoint());
        assert!(!WalStateClass::TornPre.blocks_checkpoint());
        assert!(!WalStateClass::Unreadable.blocks_checkpoint());
    }

    #[test]
    #[serial_test::serial(tshm_counter)]
    fn inspect_store_classifies_synthetic_file_sets() {
        let dir = std::env::temp_dir().join(format!("wal_guard_state_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // All 6 domain stores map onto the one consolidated file (core.db);
        // only `logs` is a separate physical file. Write sidecars for
        // core.db (reused across the three classifications below).
        // Healthy store: tshm advertises 0 frames, wal empty.
        write(&dir.join("db/core.db-tshm"), &tshm_bytes(0, 0, 5));
        write(&dir.join("db/core.db-wal"), &[]);
        let s = inspect_store(&dir, "board", StoreFds::none());
        assert_eq!(s.class, WalStateClass::Healthy);

        // Orphaned store: tshm advertises live frames, wal empty.
        write(&dir.join("db/core.db-tshm"), &tshm_bytes(356, 0, 710_565));
        write(&dir.join("db/core.db-wal"), &[]);
        let s = inspect_store(&dir, "sessions", StoreFds::none());
        assert_eq!(s.class, WalStateClass::Orphaned);
        assert!(s.tshm.is_some());
        assert_eq!(s.tshm.unwrap().max_frame, 356);

        // Missing tshm → unreadable (never blocks checkpoint).
        let _ = std::fs::remove_file(&dir.join("db/core.db-tshm"));
        write(&dir.join("db/core.db-wal"), &[0u8; 4096]);
        let s = inspect_store(&dir, "users", StoreFds::none());
        assert_eq!(s.class, WalStateClass::Unreadable);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[serial_test::serial(tshm_counter)]
    fn inspect_store_visits_every_store() {
        let dir = std::env::temp_dir().join(format!("wal_guard_all_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // Only the PHYSICAL store files exist on disk: one consolidated domain
        // file (core.db, backing all 6 domain stores) + the logs file. The
        // logical store_names() map onto these via store_db_path — write
        // healthy sidecars for both physical files so every logical name
        // resolves to a Healthy fixture.
        write(&dir.join("db/core.db-tshm"), &tshm_bytes(0, 0, 0));
        write(&dir.join("db/logs.db-tshm"), &tshm_bytes(0, 0, 0));
        // Path-based inspection (StoreFds::none) so live connections from
        // concurrently-running tests cannot leak real store state in here.
        for name in crate::db::store_names() {
            let s = inspect_store(&dir, name, StoreFds::none());
            assert_eq!(
                s.class,
                WalStateClass::Healthy,
                "fixture store {name} must be healthy"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[serial_test::serial(tshm_counter)]
    fn path_read_counts_open_close_and_fd_read_does_not() {
        let dir = std::env::temp_dir().join(format!("wal_guard_cnt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let tshm_path = dir.join("db/core.db-tshm");
        write(&tshm_path, &tshm_bytes(0, 0, 0));

        reset_tshm_open_close_count();
        let file = File::open(&tshm_path).unwrap();
        let before = tshm_open_close_count();
        let s = inspect_store_at(
            &dir.join("db/core.db"),
            StoreFds {
                tshm: Some(&file),
                wal: None,
            },
        );
        assert_eq!(s.class, WalStateClass::Healthy);
        assert_eq!(
            tshm_open_close_count(),
            before,
            "fd read must not open+close"
        );

        let s = inspect_store_at(&dir.join("db/core.db"), StoreFds::none());
        assert_eq!(s.class, WalStateClass::Healthy);
        assert_eq!(
            tshm_open_close_count(),
            before + 2,
            "path read must count an open+close pair per disciplined read (two reads)"
        );
        drop(file); // keep the temp dir clean; the fd was never closed inside the loop

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[serial_test::serial(tshm_counter)]
    fn diagnose_all_stores_classifies_stale_tail_and_structural() {
        let dir = std::env::temp_dir().join(format!("wal_guard_preflight_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // Only two PHYSICAL store files exist: the consolidated domain file
        // (core.db — every domain store name maps onto it) and the logs file.
        // Exercise each classification by re-writing core.db's sidecars and
        // re-running the pre-flight scan (diagnose_all_stores diagnoses the
        // core + logs physical pair each call).

        // (1) Stale-tail: live frames (valid matching WAL header + exact
        // expected length) with the frame index longer than max_frame.
        let mut stale = tshm_bytes(120, 0, 7);
        stale[TSHM_FRAME_INDEX_LEN_OFFSET..TSHM_FRAME_INDEX_LEN_OFFSET + 4]
            .copy_from_slice(&130u32.to_le_bytes());
        stale[TSHM_PAGE_SIZE_OFFSET..TSHM_PAGE_SIZE_OFFSET + 4]
            .copy_from_slice(&4096u32.to_le_bytes());
        write(&dir.join("db/core.db-tshm"), &stale);
        let mut wal = vec![0u8; 32 + 120 * (24 + 4096)];
        wal[..32].copy_from_slice(&wal_header_bytes(4096, 0, 0));
        write(&dir.join("db/core.db-wal"), &wal);
        crate::db::wal_guard::diagnose_all_stores(&dir);
        assert_eq!(
            crate::db::wal_guard::take_boot_diagnosis(&crate::db::store_db_path(&dir, "sessions")),
            Some(BootDiagnosis::StaleTail)
        );

        // (2) Structural: truncated main DB header (drop the stale WAL so the
        // coordination class is Healthy and the main-DB gate decides).
        let _ = std::fs::remove_file(&dir.join("db/core.db-wal"));
        write(&dir.join("db/core.db"), &[0u8; 64]);
        write(&dir.join("db/core.db-tshm"), &tshm_bytes(0, 0, 0));
        crate::db::wal_guard::diagnose_all_stores(&dir);
        assert_eq!(
            crate::db::wal_guard::take_boot_diagnosis(&crate::db::store_db_path(&dir, "board")),
            Some(BootDiagnosis::Structural)
        );

        // (3) Healthy: 64 KiB page size (raw 1 in the header field) must not be
        // misclassified as structural (quarantine + recreate).
        let mut db = vec![0u8; 4096];
        db[..16].copy_from_slice(b"SQLite format 3\0");
        db[16..18].copy_from_slice(&1u16.to_be_bytes());
        write(&dir.join("db/core.db"), &db);
        write(&dir.join("db/core.db-tshm"), &tshm_bytes(0, 0, 0));
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
        write(&dir.join("db/core.db-tshm"), &tshm_bytes(0, 0, 0));
        crate::db::wal_guard::diagnose_all_stores(&dir);
        assert_eq!(
            crate::db::wal_guard::take_boot_diagnosis(&crate::db::store_db_path(&dir, "users")),
            Some(BootDiagnosis::Healthy)
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
