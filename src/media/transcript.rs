//! The transcript cache: one recording's regions, kept under the product's own storage root.
//!
//! Never beside the file. A clip may live on a read-only mount, and a sidecar in a workspace is an
//! untracked file the owner did not ask for and the product's own cleanup sweeps away. An entry is
//! keyed by the file's canonical path and records the size and modification time it was made from,
//! so a replaced file is transcribed again and an untouched one is not. The cache is bounded
//! ([`MAX_ENTRIES`]) and pruned oldest-first, so a year of uploads cannot grow it without end.
//!
//! The sound track is transcribed whole, in five-minute pieces: each piece is the track re-encoded
//! to 16 kHz mono and taken from its own offset (the decoder seeks, it does not decode up to it),
//! and that offset is what makes the regions' times the file's own. A batch of pieces is on disk at
//! a time, so a day-long recording never has to sit in the temp root whole.
//!
//! The pass is bounded ([`TRANSCRIBE_TIMEOUT`]) and stops between pieces, so a pass that runs out of
//! time loses nothing: what it read is cached with the second it reaches, and the next look resumes
//! from there instead of repeating the work. The recogniser's words are always what is stored,
//! whatever the transcription setting says, because the sound labels come from the same pass; the
//! setting only decides what a timeline shows ([`super::read`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::ffmpeg::{self, ClipInfo};
use crate::speech::Segment;
use crate::util::UnwrapPoison;

use super::MediaError;

/// The directory under the storage root the entries live in.
const DIR: &str = "transcripts";

/// The entry format's own version: an entry from another format is a miss, never a misread. Version
/// 2 dropped the words-only flag: every entry now carries the recogniser's words.
const VERSION: u32 = 2;

/// How much sound one piece carries.
const PIECE: Duration = Duration::from_mins(5);

/// How far short of a piece's own end a pass's progress may fall and still count as having read it:
/// a decoded piece can end a fraction before the length it was cut at, and the cut, the resume and
/// the finished check all compare seconds with the same tolerance.
const PIECE_SLACK: f64 = 0.5;

/// How many pieces are held on disk at once: an hour of sound at a time, so a day-long recording
/// never has to sit in the temp root whole.
const PIECES_PER_BATCH: usize = 12;

/// How long the whole track may take to read. A pass that reaches it is cut between pieces and the
/// next look resumes from the last one it cached, so a message bubble or an agent turn is never
/// held for the whole of a long recording. The engine's own bound
/// ([`crate::speech::INFERENCE_TIMEOUT`]) still covers the one piece that is in flight when it
/// fires, which is the grace the pass is allowed on top of this.
const TRANSCRIBE_TIMEOUT: Duration = Duration::from_mins(30);

/// How many entries the cache keeps. Every clip ever read leaves one, and uploads arrive under a
/// fresh name each time, so without a bound the directory would grow for the life of the product.
const MAX_ENTRIES: usize = 256;

/// How old an unfinished write must be before it is garbage: a process killed between the write and
/// the rename leaves one behind, and nothing else ever removes it.
const PART_STALE: Duration = Duration::from_hours(1);

/// One transcription per file at a time: a second reader of the same clip waits for the first
/// instead of repeating its work — up to its own bound, never for ever.
static IN_FLIGHT: LazyLock<Mutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The whole sound track as far as it has been read.
pub(super) struct Transcribed {
    pub(super) regions: Vec<Segment>,
    /// The second of the sound track the pass reached: the track's own length when it finished,
    /// less when the bound stopped it.
    pub(super) reached: f64,
    /// Why the pass could not finish the track, when it could not — a piece it could not read, or
    /// another read holding the file. Everything read before it is still in `regions`, so a track
    /// that cannot be finished is never one that appears to hold no speech.
    pub(super) failure: Option<MediaError>,
}

/// The whole track's regions for `path`, from the cache when it still describes that file.
pub(super) async fn regions(path: &Path, info: &ClipInfo) -> Transcribed {
    let key = canonical(path).await;
    let track = info.sound_length().as_secs_f64();
    // Read once before the queue and once after it: a reader that waited may find the entry another
    // reader has just finished.
    let mut cached = load(&key, info.bytes).await;
    if let Some(done) = finished(&mut cached, track) {
        return done;
    }
    let deadline = Instant::now() + TRANSCRIBE_TIMEOUT;
    // One pass per file: a second reader waits for the first instead of repeating its work — but
    // not past a whole pass of its own, and never through a shutdown.
    let Some(_flight) = single_flight(key.clone(), deadline).await else {
        // A stop is not another read holding the file: the timeline says which it was.
        let reason = if crate::shutdown::aborting() {
            MediaError::Stopping
        } else {
            MediaError::Busy
        };
        return unwaited(cached, track, reason);
    };
    cached = load(&key, info.bytes).await;
    if let Some(done) = finished(&mut cached, track) {
        return done;
    }
    // The wait used up the pass's whole bound: there is no time left to transcribe in.
    if Instant::now() >= deadline {
        return unwaited(cached, track, MediaError::Busy);
    }
    // Resume from a pass that stopped at its bound.
    let (regions, from) = cached.map_or_else(
        || (Vec::new(), 0.0),
        |cached| (cached.regions, cached.covered),
    );
    let mut transcribed = transcribe(path, info, &key, from, regions, deadline).await;
    transcribed.reached = reached_of(transcribed.reached, track);
    transcribed
}

/// What a reader answers when it could not have the file inside its own bound: the cache as it
/// stands, with the reason it is not the whole track.
fn unwaited(cached: Option<Entry>, track: f64, reason: MediaError) -> Transcribed {
    let (regions, covered) = cached.map_or_else(
        || (Vec::new(), 0.0),
        |cached| (cached.regions, cached.covered),
    );
    Transcribed {
        regions,
        reached: reached_of(covered, track),
        failure: Some(reason),
    }
}

/// The second a read says the transcript reaches: the track's own length once the last piece is
/// read, the progress otherwise.
fn reached_of(covered: f64, track: f64) -> f64 {
    if covered >= track - PIECE_SLACK {
        track
    } else {
        covered
    }
}

/// One cache entry as a read shows it.
fn transcribed(cached: Entry, track: f64) -> Transcribed {
    Transcribed {
        regions: cached.regions,
        reached: reached_of(cached.covered, track),
        failure: None,
    }
}

/// One piece of a sound track and where it starts in the file.
struct Piece {
    start: Duration,
    length: Duration,
}

/// The pieces a track of `duration` is cut into, in order.
fn pieces(duration: Duration) -> Vec<Piece> {
    let mut pieces = Vec::new();
    let mut start = Duration::ZERO;
    while start < duration {
        let length = PIECE.min(duration.saturating_sub(start));
        pieces.push(Piece { start, length });
        start += PIECE;
    }
    pieces
}

/// The pieces a pass that has read up to `from` seconds still has to read. A piece's own start
/// decides it, never a decoded length, so a file whose audio stops early cannot leave a piece
/// looking unread.
fn remaining(pieces: Vec<Piece>, from: f64) -> Vec<Piece> {
    pieces
        .into_iter()
        .filter(|piece| piece.start.as_secs_f64() >= from - PIECE_SLACK)
        .collect()
}

/// The whole track from `from` seconds on, piece by piece: a batch's WAVs are extracted, read and
/// dropped before the next batch is extracted, so the temp root never holds a long recording whole.
/// Progress is cached at the last piece each batch got through, so a pass the deadline cuts keeps
/// everything it read. A piece that cannot be extracted, or that the recogniser cannot read, stops
/// the pass with the regions read before it: a note beside them, never a reason to drop them.
async fn transcribe(
    path: &Path,
    info: &ClipInfo,
    key: &Path,
    from: f64,
    mut regions: Vec<Segment>,
    deadline: Instant,
) -> Transcribed {
    // A piece the cache already reaches is not read again.
    let mut todo = remaining(pieces(info.sound_length()), from).into_iter();
    tracing::debug!(
        path = %path.display(),
        pieces = todo.len(),
        "Transcribing a sound track"
    );
    let mut reached = from;
    loop {
        let mut batch: Vec<(Piece, ffmpeg::Audio)> = Vec::new();
        // Never start a piece the pass has no time left for, and never let one extraction run past
        // the bound either: the bound is what the caller was promised.
        while batch.len() < PIECES_PER_BATCH && Instant::now() < deadline {
            let Some(piece) = todo.next() else { break };
            let extracted = tokio::time::timeout_at(
                deadline.into(),
                ffmpeg::extract_audio(path, Some((piece.start, piece.length))),
            )
            .await;
            match extracted {
                Ok(Ok(audio)) => batch.push((piece, audio)),
                Ok(Err(error)) => {
                    // What was already extracted is read and cached before the failure is
                    // reported, so a clip that stops decoding keeps what it got through. The
                    // extraction is what is reported: a batch that cannot be read either is not
                    // the cause.
                    let read = read_batch(&mut batch, key, info, &mut regions, deadline).await;
                    if let Some(failed) = read.failure {
                        tracing::warn!(
                            reason = %failed.message(),
                            "The pieces extracted before a decode failure could not be read"
                        );
                    }
                    return Transcribed {
                        regions,
                        reached: read.reached.unwrap_or(reached),
                        failure: Some(error.into()),
                    };
                }
                // The bound ran out mid-extraction.
                Err(_elapsed) => break,
            }
        }
        // Nothing to read: the track ends here, or the bound ran out while extracting. Either way
        // the next look starts from where this one did — a bound is not a recognition that failed.
        if batch.is_empty() || Instant::now() >= deadline {
            break;
        }
        let read = read_batch(&mut batch, key, info, &mut regions, deadline).await;
        reached = read.reached.unwrap_or(reached);
        if let Some(error) = read.failure {
            return Transcribed {
                regions,
                reached,
                failure: Some(error),
            };
        }
        // Nothing left to read, or the pass was cut without a failure — the engine's bound, a
        // stop: either way what was read is cached, and the next look starts from `reached`.
        if read.cut || read.reached.is_none() {
            break;
        }
    }
    Transcribed {
        regions,
        reached,
        failure: None,
    }
}

/// What one batch read.
struct Batch {
    /// The second the batch reaches, `None` when it read nothing.
    reached: Option<f64>,
    /// The pass was cut without a failure — the engine's bound ran out, or the product is
    /// stopping. Nothing about the file is wrong.
    cut: bool,
    /// The piece that could not be read, when one could not.
    failure: Option<MediaError>,
}

/// Read one batch of pieces in one engine pass, extend the regions, and cache them at the last
/// piece read.
async fn read_batch(
    batch: &mut Vec<(Piece, ffmpeg::Audio)>,
    key: &Path,
    info: &ClipInfo,
    regions: &mut Vec<Segment>,
    deadline: Instant,
) -> Batch {
    let files: Vec<(PathBuf, f64)> = batch
        .iter()
        .map(|(piece, audio)| (audio.path().to_path_buf(), piece.start.as_secs_f64()))
        .collect();
    let read = match crate::speech::transcribe_pieces(&files, deadline).await {
        Ok(read) => read,
        Err(error) => {
            batch.clear();
            // A bound and a stop are not failures of the recording: the timeline says how far it
            // got instead of calling it a recognition that failed.
            let cut = error
                .downcast_ref::<crate::speech::InferenceTimedOut>()
                .is_some()
                || error.downcast_ref::<crate::speech::Interrupted>().is_some();
            return Batch {
                reached: None,
                cut,
                failure: (!cut).then(|| piece_failure(&error)),
            };
        }
    };
    // The last piece the pass got through — a pass the deadline cut reads fewer than the batch
    // holds. Its own end, never the decoded end: a file whose audio stops before its container
    // says so would otherwise never be seen as read.
    let reached = read
        .pieces
        .checked_sub(1)
        .and_then(|index| batch.get(index))
        .map(|(piece, _)| (piece.start + piece.length).as_secs_f64());
    // The batch's WAVs are read: dropping the batch removes the files.
    batch.clear();
    regions.extend(read.regions);
    if let Some(reached) = reached {
        store(key, info.bytes, reached, regions).await;
    }
    // A piece the recogniser could not read stops the pass, but what it read before that piece is
    // cached above and handed back: the next look resumes from there, and the timeline shows those
    // lines with the failure as a note.
    Batch {
        reached,
        cut: false,
        failure: read.failure.as_ref().map(piece_failure),
    }
}

/// A piece that could not be read, as the timeline reports it: a piece the product's own ffmpeg
/// wrote but nothing could decode is a fact about the file, while anything else is the
/// recogniser's own failure — the two are never the same sentence.
fn piece_failure(error: &anyhow::Error) -> MediaError {
    if error.downcast_ref::<crate::speech::Unreadable>().is_some() {
        return MediaError::Undecodable;
    }
    // The engine's own summary ("the recording could not be transcribed") says what the timeline
    // says already: the cause is the part it adds.
    MediaError::SoundTrack(error.root_cause().to_string())
}

/// One file's cache entry.
#[derive(Serialize, Deserialize)]
struct Entry {
    version: u32,
    bytes: u64,
    modified_ms: u64,
    /// The second of the file the regions reach: the whole track once the last piece is read, less
    /// when a bounded pass stopped early.
    covered: f64,
    regions: Vec<Segment>,
}

/// The cached entry for `path`, when it still describes the file.
async fn load(path: &Path, bytes: u64) -> Option<Entry> {
    let entry: Entry =
        serde_json::from_slice(&tokio::fs::read(entry_path(path)?).await.ok()?).ok()?;
    if entry.version != VERSION
        || entry.bytes != bytes
        || entry.modified_ms != modified_ms(path).await?
    {
        return None;
    }
    Some(entry)
}

/// The entry as a read shows it, when it describes the file and reaches the whole track. A partial
/// entry is left where it is: it is what the pass resumes from.
fn finished(cached: &mut Option<Entry>, track: f64) -> Option<Transcribed> {
    if !cached
        .as_ref()
        .is_some_and(|cached| cached.covered >= track - PIECE_SLACK)
    {
        return None;
    }
    cached.take().map(|cached| transcribed(cached, track))
}

/// Store one file's regions. A cache that cannot be written is a warning, never the read's failure.
async fn store(path: &Path, bytes: u64, covered: f64, regions: &[Segment]) {
    // Nothing read is nothing to keep — except a stretch that was read and held no sound at all:
    // that entry is what stops a silent track from being decoded again on every look.
    if regions.is_empty() && covered <= 0.0 {
        return;
    }
    let Some(entry) = entry_path(path) else {
        return;
    };
    let Some(modified) = modified_ms(path).await else {
        return;
    };
    let Ok(json) = serde_json::to_vec(&Entry {
        version: VERSION,
        bytes,
        modified_ms: modified,
        covered,
        regions: regions.to_vec(),
    }) else {
        return;
    };
    let Some(dir) = entry.parent() else {
        return;
    };
    if let Err(error) = tokio::fs::create_dir_all(dir).await {
        tracing::warn!(%error, "The transcript cache directory could not be created");
        return;
    }
    // Written beside the entry and renamed over it, so a reader never sees half a transcript.
    let temp = entry.with_extension(format!("{}.part", std::process::id()));
    if let Err(error) = tokio::fs::write(&temp, &json).await {
        tracing::warn!(%error, "A transcript could not be cached");
        return;
    }
    if let Err(error) = tokio::fs::rename(&temp, &entry).await {
        tracing::warn!(%error, "A transcript could not be cached");
        let _ = tokio::fs::remove_file(&temp).await;
    }
    prune(dir, &entry).await;
}

/// Drop the oldest entries once the cache holds more than [`MAX_ENTRIES`], never the one just
/// written. Best effort: a directory that cannot be listed or a file that cannot be removed is a
/// warning, never a read's failure.
async fn prune(dir: &Path, fresh: &Path) {
    let Ok(mut entries) = tokio::fs::read_dir(dir).await else {
        return;
    };
    let mut written: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        let Ok(modified) = entry.metadata().await.and_then(|meta| meta.modified()) else {
            continue;
        };
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") if path != fresh => written.push((modified, path)),
            Some("part") if modified.elapsed().is_ok_and(|age| age > PART_STALE) => {
                let _ = tokio::fs::remove_file(&path).await;
            }
            _ => {}
        }
    }
    if written.len() < MAX_ENTRIES {
        return;
    }
    written.sort_unstable_by_key(|(modified, _)| *modified);
    for (_, path) in written.iter().take(written.len() + 1 - MAX_ENTRIES) {
        if let Err(error) = tokio::fs::remove_file(path).await {
            tracing::warn!(path = %path.display(), %error, "A transcript entry could not be pruned");
        }
    }
}

/// Where one file's entry lives: named by its path, so a moved file is a different entry.
fn entry_path(path: &Path) -> Option<PathBuf> {
    let digest = crate::util::hex_string(&Sha256::digest(path.as_os_str().as_encoded_bytes()));
    Some(cache_dir()?.join(format!("{digest}.json")))
}

/// The directory the entries live in: the product's own storage root, and under `cargo test` the
/// test root, so a test can never write into the owner's live tree.
#[cfg_attr(
    test,
    expect(clippy::unnecessary_wraps, reason = "the test root always resolves")
)]
fn cache_dir() -> Option<PathBuf> {
    #[cfg(test)]
    {
        Some(crate::util::test::test_root().join(DIR))
    }
    #[cfg(not(test))]
    {
        crate::config::CONFIG
            .try_storage_root()
            .or_else(|| crate::config::default_config_dir().ok())
            .map(|root| root.join(DIR))
    }
}

async fn canonical(path: &Path) -> PathBuf {
    tokio::fs::canonicalize(path)
        .await
        .unwrap_or_else(|_| path.to_path_buf())
}

/// The file's modification time in whole milliseconds, the half of the key the size cannot give.
async fn modified_ms(path: &Path) -> Option<u64> {
    let modified = tokio::fs::metadata(path).await.ok()?.modified().ok()?;
    u64::try_from(modified.duration_since(UNIX_EPOCH).ok()?.as_millis()).ok()
}

/// The one lock for `key`, held for as long as the guard lives. Idle entries are pruned on the way
/// in, so the map holds only the transcriptions in flight. `None` when the wait would outlive
/// `deadline` or a shutdown: a reader never waits on another pass for ever.
async fn single_flight(
    key: PathBuf,
    deadline: Instant,
) -> Option<tokio::sync::OwnedMutexGuard<()>> {
    let lock = {
        let mut in_flight = IN_FLIGHT.lock().unwrap_poison();
        in_flight.retain(|_, lock| Arc::strong_count(lock) > 1);
        Arc::clone(in_flight.entry(key).or_default())
    };
    let shutdown = crate::shutdown::shutdown_token();
    tokio::select! {
        guard = lock.lock_owned() => Some(guard),
        () = tokio::time::sleep_until(deadline.into()) => None,
        () = shutdown.cancelled() => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cache entry the tests hand-build: only the two fields a read looks at are meaningful.
    fn entry(covered: f64, regions: Vec<Segment>) -> Entry {
        Entry {
            version: VERSION,
            bytes: 1024,
            modified_ms: 0,
            covered,
            regions,
        }
    }

    #[test]
    fn a_track_is_cut_into_five_minute_pieces_carrying_their_offsets() {
        let cut = pieces(Duration::from_secs(700));
        let offsets: Vec<u64> = cut.iter().map(|p| p.start.as_secs()).collect();
        assert_eq!(offsets, vec![0, 300, 600]);
        assert_eq!(cut[2].length, Duration::from_secs(100));

        // An exact multiple carries no empty piece at the end.
        assert_eq!(pieces(Duration::from_secs(600)).len(), 2);
        assert!(pieces(Duration::ZERO).is_empty());
    }

    #[test]
    fn a_finished_entry_is_not_read_again_and_a_partial_one_is_kept() {
        let mut cached = Some(entry(300.0, Vec::new()));
        assert!(finished(&mut cached, 700.0).is_none(), "700 s is not read");
        assert_eq!(
            cached.as_ref().map(|cached| cached.covered),
            Some(300.0),
            "the partial entry is what the next pass resumes from"
        );

        let mut cached = Some(entry(700.0, Vec::new()));
        let done = finished(&mut cached, 700.0).expect("the track is read");
        assert!((done.reached - 700.0).abs() < f64::EPSILON);
        assert!(cached.is_none(), "a finished entry is consumed");
    }

    #[test]
    fn a_pass_resumes_from_where_it_stopped() {
        // Nothing read yet reads every piece, however short the track is.
        assert_eq!(remaining(pieces(Duration::from_secs(700)), 0.0).len(), 3);
        assert_eq!(remaining(pieces(Duration::from_millis(400)), 0.0).len(), 1);
        // A finished piece is not read again.
        let resumed = remaining(pieces(Duration::from_secs(700)), 300.0);
        assert_eq!(resumed[0].start, Duration::from_secs(300));
        assert_eq!(resumed.len(), 2);
    }

    #[tokio::test]
    async fn a_wait_that_outlives_its_own_bound_gives_up_on_the_cache() {
        // A pass holds the file: a reader that cannot have it inside its own bound does not wait
        // for ever — it answers with what the cache holds.
        let key = PathBuf::from("/transcript-cache-test/clip.mp4");
        let held = single_flight(key.clone(), Instant::now() + TRANSCRIBE_TIMEOUT)
            .await
            .expect("nothing holds this file");
        assert!(
            single_flight(key, Instant::now()).await.is_none(),
            "a pass already running is not waited out past the reader's own bound"
        );
        drop(held);

        let cached = unwaited(
            Some(entry(
                300.0,
                vec![Segment {
                    start: 0.0,
                    end: 5.0,
                    text: Some("Hi.".to_string()),
                    label: None,
                }],
            )),
            700.0,
            MediaError::Busy,
        );
        assert_eq!(cached.regions.len(), 1, "the lines already read are kept");
        assert!((cached.reached - 300.0).abs() < f64::EPSILON);
        assert_eq!(cached.failure, Some(MediaError::Busy));
    }
}
