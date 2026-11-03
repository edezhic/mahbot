//! The recogniser: one lazily-opened engine over the machine-wide model set, and the transcript
//! a recording turns into.
//!
//! Nothing loads at boot. The first operation that needs the engine loads it — the first voice
//! message, the composer's microphone, or the Settings toggle — either from the files on disk
//! or by fetching them first ([`crate::speech::models`]), through the shared lifecycle of
//! [`crate::speech::load`]. One transcription runs at a time: the engine is a single `&mut`
//! handle, and the mutex is held for the whole pass. The memory a pass peaks at is given back
//! after it ([`mahbot_inference::release_process_memory`]), which is what keeps a resident
//! dashboard from holding a recognition pass's peak for the rest of the day.

use crate::util::UnwrapPoison;
use crate::util::model_state::ModelState;
use anyhow::{Context as _, Result};
use mahbot_inference::transcribe::{SPEECH_LABELS, Transcriber, Transcription};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::speech::decode;
use crate::speech::load::Lazy;
use crate::speech::models::{self, Purpose};

/// Default bound on one transcription (10 minutes): short clips finish in seconds,
/// long recordings in minutes, and a hung inference must not occupy a blocking
/// thread forever. Shared by the voice pipeline, the audio-enrichment path and
/// `mahbot audio transcribe`, which answers this bound in its own words.
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

/// The resident-set ceiling an input is refused against — the figure the engine's
/// own guard documents as the default for a run holding a recognition pair.
const CEILING_MIB: u64 = mahbot_inference::guard::DEFAULT_MAX_RSS_MIB;

/// The product's one recogniser: the pair and the audio tagger, both resolved from the
/// machine-wide root.
static ENGINE: Lazy<Transcriber> = Lazy::new(Purpose::Recognition, || {
    Transcriber::open_product(CEILING_MIB)
});

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

/// Audio already in memory into words: the composer's microphone recording and the audio
/// recorded after the wake word, neither of which has a file to go through.
pub(crate) async fn transcribe_samples(
    samples: Vec<f32>,
    rate: u32,
    timeout: Duration,
) -> Result<String> {
    anyhow::ensure!(!samples.is_empty(), "the recording holds no samples");
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
    let engine = ENGINE.engine().context("the recogniser is not loaded")?;

    let pass = tokio::task::spawn_blocking(move || {
        let mut engine = engine.lock().unwrap_poison();
        let result = engine
            .transcribe(&samples, rate)
            .map(|result| compose(&result));
        // The pass is over: hand its peak back rather than holding it for the life of
        // the process. The next pass recompiles, which is the accepted price.
        mahbot_inference::release_process_memory();
        result
    });

    let shutdown = crate::shutdown::shutdown_token();
    let outcome = tokio::select! {
        joined = tokio::time::timeout(timeout, pass) => joined,
        () = shutdown.cancelled() => {
            anyhow::bail!("Shutdown during audio transcription");
        }
    };
    match outcome {
        Ok(Ok(Ok(text))) => Ok(text),
        Ok(Ok(Err(e))) => Err(e.context("the recording could not be transcribed")),
        Ok(Err(join)) => Err(anyhow::anyhow!("Inference task panicked: {join}")),
        Err(_elapsed) => Err(anyhow::Error::new(InferenceTimedOut)),
    }
}

/// The words a result holds, in order, with [`UNTRANSCRIBED`] wherever the engine
/// heard speech and read nothing. Silence and non-speech audio contribute nothing, so a
/// recording that holds no speech yields an empty text; a stretch of unread speech is one
/// marker however many regions it was cut into.
fn compose(result: &Transcription) -> String {
    let mut text = String::new();
    let mut marked = false;
    for region in &result.regions {
        if let Some(read) = &region.text {
            push_words(&mut text, read);
            marked = false;
        } else if region.label.as_deref().is_some_and(speaks_words) {
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
fn speaks_words(label: &str) -> bool {
    let label = label.trim();
    SPEECH_LABELS
        .iter()
        .any(|class| label.eq_ignore_ascii_case(class))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mahbot_inference::transcribe::Region;

    fn region(start: usize, end: usize, text: Option<&str>, label: Option<&str>) -> Region {
        Region {
            start,
            end,
            text: text.map(str::to_string),
            label: label.map(str::to_string),
            read: text.is_some(),
        }
    }

    fn result(regions: Vec<Region>) -> Transcription {
        Transcription {
            sample_rate: 16_000,
            samples: regions.last().map_or(0, |region| region.end),
            gain: 1.0,
            regions,
        }
    }

    #[test]
    fn a_recording_of_silence_yields_nothing() {
        let text = compose(&result(vec![region(0, 16_000, None, Some("Silence"))]));
        assert_eq!(text, "");
        let text = compose(&result(vec![region(0, 16_000, None, Some("Music"))]));
        assert_eq!(text, "");
    }

    #[test]
    fn unread_speech_is_marked_once_per_stretch() {
        let text = compose(&result(vec![
            region(0, 1_000, Some("hello there"), None),
            region(1_000, 2_000, None, Some("Speech")),
            region(2_000, 3_000, None, Some("Male speech, man speaking")),
            region(3_000, 4_000, Some("how are you"), None),
            region(4_000, 5_000, None, Some("Speech")),
        ]));
        assert_eq!(
            text,
            format!("hello there {UNTRANSCRIBED} how are you {UNTRANSCRIBED}")
        );
    }

    #[test]
    fn two_unread_stretches_apart_are_two_markers() {
        let text = compose(&result(vec![
            region(0, 1_000, None, Some("Speech")),
            region(1_000, 2_000, None, Some("Music")),
            region(2_000, 3_000, None, Some("Speech")),
        ]));
        assert_eq!(text, format!("{UNTRANSCRIBED} {UNTRANSCRIBED}"));
    }

    #[test]
    fn a_recording_the_engine_could_not_read_still_says_so() {
        let text = compose(&result(vec![region(0, 16_000, None, Some("Speech"))]));
        assert_eq!(text, UNTRANSCRIBED);
        assert!(!text.trim().is_empty(), "the marker is visible text");
    }
}
