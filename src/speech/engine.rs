//! The recogniser: one lazily-opened engine over the machine-wide model set, the transcript a
//! recording turns into, and the wake word's window listener over the same pair.
//!
//! Nothing loads at boot. The first operation that needs the engine loads it — the first voice
//! message, the composer's microphone, the Settings toggle or the wake word — either from the files
//! on disk or by fetching them first ([`crate::speech::models`]), through the shared lifecycle of
//! [`crate::speech::load`]. The two consumers sit behind separate mutexes: one transcription runs
//! at a time, and a wake-word pass never waits out a transcription (or the other way round). The
//! memory a transcription pass peaks at is given back after it
//! ([`mahbot_inference::release_process_memory`]), which is what keeps a resident dashboard from
//! holding a recognition pass's peak for the rest of the day.

use crate::util::UnwrapPoison;
use crate::util::model_state::ModelState;
use anyhow::{Context as _, Result};
use mahbot_inference::ced::{CED_MODEL, Tagger};
use mahbot_inference::listen::{Listener, PAIR};
use mahbot_inference::loader::{Pair, Role};
use mahbot_inference::transcribe::{Region, SPEECH_LABELS, Transcriber};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use crate::speech::decode;
use crate::speech::load::Lazy;
use crate::speech::models::{self, Purpose};

/// The bound on one recognition pass (10 minutes), counted from the moment the caller asks — a read
/// that queues behind another pass waits inside it, never on top of it: short clips finish in
/// seconds, long recordings in minutes, and a hung inference must not occupy a blocking thread for
/// ever. The voice pipeline, the audio-enrichment path and `mahbot audio transcribe` pass it as
/// their whole bound and answer it in their own words; a media read's pass has its own, longer
/// bound and takes this one as the grace for the piece still in flight.
pub(crate) const INFERENCE_TIMEOUT: Duration = Duration::from_mins(10);

/// What a transcript carries where the engine heard speech and read nothing. A run of
/// unread speech is one marker, and silence or non-speech audio carries none: a
/// recording that holds no speech still yields nothing at all.
pub(crate) const UNTRANSCRIBED: &str = "[untranscribed speech]";

/// The recogniser's own bound on one inference ([`INFERENCE_TIMEOUT`]) was reached.
///
/// A type rather than a message, so `mahbot audio` can answer this failure in its own
/// words — a recording that reached the bound, with nothing produced at all — without
/// matching on the engine's wording.
#[derive(Debug)]
pub(crate) struct InferenceTimedOut;

impl std::fmt::Display for InferenceTimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("speech recognition timed out")
    }
}

impl std::error::Error for InferenceTimedOut {}

/// The product stopped while a pass was queued or running.
///
/// A type rather than a message, so a caller can tell a stop from a bound and from a failure of
/// the recording — a draining product reads nothing, and nothing about the file is wrong.
#[derive(Debug)]
pub(crate) struct Interrupted;

impl std::fmt::Display for Interrupted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the product is stopping")
    }
}

impl std::error::Error for Interrupted {}

/// A piece of a recording the product's own decoder could not read.
///
/// A type rather than a message, so a caller can tell a piece that could not be decoded from one
/// the recogniser failed on: the first is a fact about the file, the second about the reading.
#[derive(Debug)]
pub(crate) struct Unreadable;

impl std::fmt::Display for Unreadable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the sound could not be decoded")
    }
}

impl std::error::Error for Unreadable {}

/// The resident-set ceiling an input is refused against — the figure the engine's
/// own guard documents as the default for a run holding a recognition pair.
const CEILING_MIB: u64 = mahbot_inference::guard::DEFAULT_MAX_RSS_MIB;

/// The product's one recogniser: the recognition pair, the tagger every recognition call depends
/// on, and the wake word's listener over the same pair — one copy of the weights, two consumers.
static ENGINE: Lazy<Recogniser> = Lazy::new(Purpose::Recognition, open_recogniser);

/// The recogniser's two consumers over one loaded pair: transcription and the wake word's window
/// listener. Each sits behind its own mutex, so a wake pass never waits out a transcription (or
/// the other way round).
struct Recogniser {
    transcriber: Arc<Mutex<Transcriber>>,
    listener: Arc<Mutex<Listener>>,
}

/// Map the pair, open the tagger and build both consumers. Runs off the async threads: it maps
/// gigabytes and compiles graphs.
fn open_recogniser() -> Result<Recogniser> {
    let pair = Arc::new(Pair::of_role(Role::Recognition)?);
    let tagger = Tagger::open(&models::root()?.join(CED_MODEL))?;
    let transcriber = Transcriber::open(Arc::clone(&pair), tagger, CEILING_MIB)?;
    Ok(Recogniser {
        transcriber: Arc::new(Mutex::new(transcriber)),
        listener: Arc::new(Mutex::new(Listener::new(pair))),
    })
}

/// One consumer handle of the loaded recogniser, or the error naming why it is not there.
fn consumer<T>(pick: impl FnOnce(&Recogniser) -> Arc<Mutex<T>>) -> Result<Arc<Mutex<T>>> {
    let engine = ENGINE.engine().context("the recogniser is not loaded")?;
    let handle = {
        let engine = engine.lock().unwrap_poison();
        pick(&engine)
    };
    Ok(handle)
}

/// The recogniser's lifecycle state, for callers resolving a user-facing status from it
/// (nothing preloads, so "not loaded" alone cannot tell a load in flight from none started).
pub(crate) fn state() -> ModelState {
    ENGINE.state()
}

pub(crate) fn is_loaded() -> bool {
    ENGINE.is_loaded()
}

pub(crate) fn is_loading() -> bool {
    ENGINE.is_loading()
}

pub(crate) fn is_failed() -> bool {
    ENGINE.is_failed()
}

/// Retry after a terminal `Failed` state: reset it and re-run the background chain.
/// Returns `true` when a retry was initiated.
pub(crate) fn retry_init() -> bool {
    ENGINE.retry_init()
}

/// Demand the engine: start the load-or-fetch chain in the background. A no-op when it is
/// already ready, already loading, or terminally failed.
pub(crate) fn spawn_background_init() {
    ENGINE.spawn_background_init();
}

/// Obtain the set once, with a notice before a fetch that really begins, and open the
/// engine — the command line's own entry point: one attempt, no retry loop, and a
/// failure is the caller's to report.
pub(crate) async fn ensure_ready_once(announce: impl FnOnce(&Path)) -> Result<()> {
    ENGINE.ensure_ready_once(announce).await
}

/// One recording into words: the whole file, decoded and then read by the engine.
pub(crate) async fn transcribe_file_async(path: &Path, timeout: Duration) -> Result<String> {
    let owned: PathBuf = path.to_owned();
    let (samples, rate) =
        tokio::task::spawn_blocking(move || decode::decode_audio_to_mono_f32(&owned))
            .await
            .context("Audio decode task panicked")??;
    transcribe_samples(samples, rate, timeout).await
}

/// One region of a recording: where it is, and what was heard there.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Segment {
    /// Seconds from the start of the recording.
    pub(crate) start: f64,
    pub(crate) end: f64,
    /// The words read here — `Some` exactly when the recogniser read words.
    pub(crate) text: Option<String>,
    /// The sound labeller's class, on a region no reading produced.
    pub(crate) label: Option<String>,
}

/// What one engine pass read: the regions of the pieces it got through, how many of the pieces it
/// was given those were — fewer when its bound cut the pass between pieces — and why it stopped
/// when a piece could not be read at all.
pub(crate) struct Read {
    pub(crate) regions: Vec<Segment>,
    pub(crate) pieces: usize,
    /// The piece the pass could not read, with everything before it kept: a caller that can cache
    /// progress decides whether the failure is its own.
    pub(crate) failure: Option<anyhow::Error>,
}

/// Pieces of one recording, each with its own start offset in seconds, read in one run before
/// `deadline`: the engine is loaded once and its peak given back once.
pub(crate) async fn transcribe_pieces(
    pieces: &[(PathBuf, f64)],
    deadline: Instant,
) -> Result<Read> {
    if pieces.is_empty() {
        return Ok(Read {
            regions: Vec::new(),
            pieces: 0,
            failure: None,
        });
    }
    let pieces = pieces
        .iter()
        .map(|(path, offset)| Piece::File {
            path: path.clone(),
            offset: *offset,
        })
        .collect();
    // A media read caches every finished piece, so the one still in flight when the pass deadline
    // fires may finish: a piece's own bound is the grace it is allowed.
    read_pieces(pieces, deadline, INFERENCE_TIMEOUT).await
}

/// Audio already in memory into words: the composer's microphone recording and the audio
/// recorded after the wake word, neither of which has a file to go through.
pub(crate) async fn transcribe_samples(
    samples: Vec<f32>,
    rate: u32,
    timeout: Duration,
) -> Result<String> {
    anyhow::ensure!(!samples.is_empty(), "the recording holds no samples");
    let pieces = vec![Piece::Memory { samples, rate }];
    // Nothing is cached here and there is no next look: the bound is the whole bound.
    let read = read_pieces(pieces, Instant::now() + timeout, Duration::ZERO).await?;
    Ok(compose(&read.regions))
}

/// One piece of a batch: what to read, and where it starts in the recording it came from.
enum Piece {
    /// A file, decoded inside the blocking pass, and its offset in seconds.
    File { path: PathBuf, offset: f64 },
    /// Audio already in memory at its own rate — one whole recording, so at offset zero.
    Memory { samples: Vec<f32>, rate: u32 },
}

/// One transcription pass at a time, held by the pass rather than by its caller: a caller that
/// reaches its own deadline returns, and the pass it started keeps the engine (and this permit)
/// until it really stops — so the next read waits for it rather than running beside it.
static PASS: LazyLock<Arc<tokio::sync::Mutex<()>>> =
    LazyLock::new(|| Arc::new(tokio::sync::Mutex::new(())));

/// Read every piece in one pass over the loaded engine: the batch is loaded once, decoded
/// and transcribed in one blocking task, and its peak given back once when the pass ends. The pass
/// stops between pieces once `deadline` has passed, so a long recording leaves its unread pieces to
/// the next look instead of holding its caller for ever.
///
/// `deadline` is the caller's own whole bound, counted from the moment it asked: the wait for the
/// permit is inside it, so a hung pass can neither outlive its own deadline nor hold a later read
/// for the life of the process. `grace` is how much longer the one piece still in flight may run
/// once the deadline has passed: a media read allows a piece's own bound because it caches every
/// finished piece and resumes from there, while a caller with nowhere to resume from allows none,
/// so its bound is its wall bound.
async fn read_pieces(pieces: Vec<Piece>, deadline: Instant, grace: Duration) -> Result<Read> {
    // The machine-wide root before anything else: a missing one is a failure naming the
    // path, never a substitute location.
    let dir = models::root()?;
    // Nothing preloads the engine: the first operation that needs it loads it here (the
    // ordinary case is the set on disk) or waits out the load a voice press started.
    anyhow::ensure!(
        ENGINE.ensure_loaded(&dir).await,
        "{}",
        models::unloaded_message(&dir, Purpose::Recognition, is_failed())
    );
    let transcriber = consumer(|recogniser| Arc::clone(&recogniser.transcriber))?;

    let shutdown = crate::shutdown::shutdown_token();
    let pass_guard = tokio::select! {
        pass = PASS.clone().lock_owned() => pass,
        () = shutdown.cancelled() => {
            return Err(anyhow::Error::new(Interrupted));
        }
        // A pass already running holds the engine; this one must not outlive its own deadline
        // waiting for it.
        () = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => {
            return Err(anyhow::Error::new(InferenceTimedOut));
        }
    };
    if Instant::now() >= deadline {
        return Err(anyhow::Error::new(InferenceTimedOut));
    }
    let pass = tokio::task::spawn_blocking(move || {
        // The guard rides with the pass, so the engine stays taken until this task really ends —
        // a caller that gave up at its own deadline cannot let a second pass start alongside it.
        let _pass = pass_guard;
        let mut engine = transcriber.lock().unwrap_poison();
        let result = read_all(&mut engine, pieces, deadline);
        // The pass is over: hand its peak back rather than holding it for the life of
        // the process. The next pass recompiles, which is the accepted price.
        mahbot_inference::release_process_memory();
        result
    });

    // The pass stops itself at the deadline; the one piece still in flight is the grace this
    // wrapper allows before a hung engine is reported instead of waited on for ever.
    let abandon = tokio::time::Instant::from_std(deadline + grace);
    let outcome = tokio::select! {
        joined = tokio::time::timeout_at(abandon, pass) => joined,
        () = shutdown.cancelled() => {
            return Err(anyhow::Error::new(Interrupted));
        }
    };
    match outcome {
        // Nothing read at all is this caller's failure; anything read before the failing piece is
        // handed back for the caller to keep.
        Ok(Ok(read)) if read.pieces == 0 && read.failure.is_some() => {
            let failure = read
                .failure
                .unwrap_or_else(|| anyhow::anyhow!("no piece was read"));
            Err(failure.context("the recording could not be transcribed"))
        }
        Ok(Ok(read)) => Ok(read),
        Ok(Err(join)) => Err(anyhow::anyhow!("Inference task panicked: {join}")),
        Err(_elapsed) => Err(anyhow::Error::new(InferenceTimedOut)),
    }
}

/// Read every piece in one run over the locked engine, each region moved into the whole
/// recording's own time base, stopping between pieces at `deadline` and at the first piece that
/// cannot be read — which is reported rather than thrown away with the pieces already read.
fn read_all(engine: &mut Transcriber, pieces: Vec<Piece>, deadline: Instant) -> Read {
    let mut read = Read {
        regions: Vec::new(),
        pieces: 0,
        failure: None,
    };
    for piece in pieces {
        if Instant::now() >= deadline {
            break;
        }
        match read_piece(engine, piece) {
            Ok(regions) => {
                read.pieces += 1;
                read.regions.extend(regions);
            }
            Err(error) => {
                read.failure = Some(error);
                break;
            }
        }
    }
    read
}

/// One piece into regions of the whole recording.
fn read_piece(engine: &mut Transcriber, piece: Piece) -> Result<Vec<Segment>> {
    let (samples, rate, offset) = match piece {
        Piece::File { path, offset } => {
            let (samples, rate) = decode::decode_audio_to_mono_f32(&path).map_err(|error| {
                // The path is an extracted temp file, not the owner's: the log keeps the chain,
                // the caller gets the typed failure.
                tracing::warn!(path = %path.display(), %error, "A transcribed piece could not be decoded");
                anyhow::Error::new(Unreadable)
            })?;
            (samples, rate, offset)
        }
        Piece::Memory { samples, rate } => (samples, rate, 0.0),
    };
    // A piece cut past the end of the sound (a track that stops before its container does) holds
    // nothing to read: it contributes no region rather than failing the whole track.
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    let heard = engine.transcribe(&samples, rate)?;
    // The regions are marked at the transcription's own rate, which is the engine's analysis rate
    // rather than the piece's: a 44.1 kHz file is resampled before it is cut.
    Ok(heard
        .regions
        .into_iter()
        .map(|region| segment(offset, heard.sample_rate, region))
        .collect())
}

/// One region of a piece as a segment of the whole recording: the marks moved by the piece's
/// own offset, so a windowed read says where in the file it was heard.
#[expect(
    clippy::cast_precision_loss,
    reason = "a sample index is exact in f64 far past any recording a machine can hold"
)]
fn segment(offset: f64, rate: u32, region: Region) -> Segment {
    let at = |samples: usize| offset + samples as f64 / f64::from(rate);
    Segment {
        start: at(region.start),
        end: at(region.end),
        text: region.text,
        label: region.label,
    }
}

/// One clip into its window embedding — enrolment and the negative pool embed a clip at a time.
pub(crate) fn embed_clip(samples: &[f32]) -> Result<Vec<f32>> {
    let listener = consumer(|recogniser| Arc::clone(&recogniser.listener))?;
    let mut listener = listener.lock().unwrap_poison();
    listener.embed(samples)
}

/// Two consecutive windows in one pass through the encoder — the wake word's batch.
pub(crate) fn embed_pair(first: &[f32], second: &[f32]) -> Result<[Vec<f32>; PAIR]> {
    let listener = consumer(|recogniser| Arc::clone(&recogniser.listener))?;
    let mut listener = listener.lock().unwrap_poison();
    listener.embed_pair(first, second)
}

/// A batch of clips embedded one at a time; a clip that fails is skipped with a warning.
pub(crate) fn embed_clips(clips: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let Ok(listener) = consumer(|recogniser| Arc::clone(&recogniser.listener)) else {
        tracing::warn!("embed_clips: the recogniser is not loaded");
        return Vec::new();
    };
    let mut listener = listener.lock().unwrap_poison();
    clips
        .iter()
        .filter_map(|clip| match listener.embed(clip) {
            Ok(embedding) => Some(embedding),
            Err(e) => {
                tracing::warn!(
                    "Failed to embed a {}-sample clip: {e} — skipping",
                    clip.len()
                );
                None
            }
        })
        .collect()
}

/// The words a batch holds, in order, with [`UNTRANSCRIBED`] wherever the engine
/// heard speech and read nothing. Silence and non-speech audio contribute nothing, so a
/// recording that holds no speech yields an empty text; a stretch of unread speech is one
/// marker however many regions it was cut into.
fn compose(segments: &[Segment]) -> String {
    let mut text = String::new();
    let mut marked = false;
    for segment in segments {
        if let Some(read) = &segment.text {
            push_words(&mut text, read);
            marked = false;
        } else if segment.label.as_deref().is_some_and(is_speech_label) {
            if !marked {
                push_words(&mut text, UNTRANSCRIBED);
            }
            marked = true;
        } else {
            marked = false;
        }
    }
    text
}

fn push_words(into: &mut String, words: &str) {
    let words = words.trim();
    if words.is_empty() {
        return;
    }
    if !into.is_empty() {
        into.push(' ');
    }
    into.push_str(words);
}

/// Whether a tagger label says the audio there is speech or singing — the engine's own
/// list, so the marker follows the same rule its readings do.
pub(crate) fn is_speech_label(label: &str) -> bool {
    let label = label.trim();
    SPEECH_LABELS
        .iter()
        .any(|class| label.eq_ignore_ascii_case(class))
}

/// The tagger's class for a stretch with nothing to hear.
const SILENCE_LABEL: &str = "Silence";

/// Whether a tagger label says the audio there is silence — the counterpart of
/// [`is_speech_label`], kept beside it so both class rules live with the engine's own vocabulary
/// and a renamed class cannot slip past one of them.
pub(crate) fn is_silence_label(label: &str) -> bool {
    label.trim().eq_ignore_ascii_case(SILENCE_LABEL)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heard(start: f64, end: f64, text: Option<&str>, label: Option<&str>) -> Segment {
        Segment {
            start,
            end,
            text: text.map(str::to_string),
            label: label.map(str::to_string),
        }
    }

    #[test]
    fn a_recording_of_silence_yields_nothing() {
        let text = compose(&[heard(0.0, 1.0, None, Some("Silence"))]);
        assert_eq!(text, "");
        let text = compose(&[heard(0.0, 1.0, None, Some("Music"))]);
        assert_eq!(text, "");
    }

    #[test]
    fn unread_speech_is_marked_once_per_stretch() {
        let text = compose(&[
            heard(0.0, 1.0, Some("hello there"), None),
            heard(1.0, 2.0, None, Some("Speech")),
            heard(2.0, 3.0, None, Some("Male speech, man speaking")),
            heard(3.0, 4.0, Some("how are you"), None),
            heard(4.0, 5.0, None, Some("Speech")),
        ]);
        assert_eq!(
            text,
            format!("hello there {UNTRANSCRIBED} how are you {UNTRANSCRIBED}")
        );
    }

    #[test]
    fn two_unread_stretches_apart_are_two_markers() {
        let text = compose(&[
            heard(0.0, 1.0, None, Some("Speech")),
            heard(1.0, 2.0, None, Some("Music")),
            heard(2.0, 3.0, None, Some("Speech")),
        ]);
        assert_eq!(text, format!("{UNTRANSCRIBED} {UNTRANSCRIBED}"));
    }

    #[test]
    fn a_recording_the_engine_could_not_read_still_says_so() {
        let text = compose(&[heard(0.0, 1.0, None, Some("Speech"))]);
        assert_eq!(text, UNTRANSCRIBED);
    }
}
