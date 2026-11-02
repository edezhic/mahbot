//! The recogniser itself: one lazily-opened engine over the machine-wide model set,
//! and the transcript a recording turns into.
//!
//! Nothing loads at boot. The first operation that needs the engine loads it — the
//! first voice message, the composer's microphone, or the Settings toggle — either
//! from the files on disk or by fetching them first ([`models`]). `Failed` is
//! terminal; the voice pipeline's bounded auto-retry and the Settings retry are the
//! only recovery knobs.
//!
//! One transcription runs at a time: the engine is a single `&mut` handle, and the
//! mutex is held for the whole pass — exactly the discipline the previous
//! recogniser had. The memory a pass peaks at is given back after it
//! ([`mahbot_inference::release_process_memory`]), which is what keeps a resident
//! dashboard from holding a recognition pass's peak for the rest of the day.

use crate::util::UnwrapPoison;
use crate::util::model_state::{AtomicModelState, ModelLoadGuard, ModelState};
use anyhow::{Context as _, Result};
use futures_util::FutureExt as _;
use mahbot_inference::transcribe::{SPEECH_LABELS, Transcriber, Transcription};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::speech::{decode, models};

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

/// Retry sleep base: 5 seconds, doubled per attempt.
const RETRY_BASE_SECS: u64 = 5;

/// Attempts one background fetch-and-open chain makes before `Failed` is terminal.
const MAX_LOAD_RETRIES: u32 = 12;

/// How often a caller that needs the recogniser checks whether an in-flight load has
/// settled ([`ensure_loaded`]).
const LOAD_WAIT_POLL: Duration = Duration::from_millis(100);

/// The longest a caller waits for a load already in flight: a set on disk reads in
/// seconds, and a fetch is not waited on at all.
const LOAD_WAIT_LIMIT: Duration = Duration::from_mins(2);

/// The opened engine — `None` until loaded.
static ENGINE: Mutex<Option<Arc<Mutex<Transcriber>>>> = Mutex::new(None);

/// Lazy-load lifecycle: Uninit → Loading → Ready / Failed.
static STATE: AtomicModelState = AtomicModelState::new(ModelState::Uninit);

/// True once a chain has moved past checking the files on disk to fetching them. A
/// caller waiting for `Loading` stops waiting when this is set: a cache read takes
/// seconds, a fetch takes minutes, and whoever just spoke gets an answer either way.
static FETCHING: AtomicBool = AtomicBool::new(false);

/// The recogniser's lifecycle state, for callers resolving a user-facing status from
/// it (nothing preloads, so "not loaded" alone cannot tell a load in flight from none
/// started).
pub(crate) fn state() -> ModelState {
    STATE.load(Ordering::Acquire)
}

pub(crate) fn is_loaded() -> bool {
    STATE.is_ready()
}

pub(crate) fn is_loading() -> bool {
    state() == ModelState::Loading
}

pub(crate) fn is_failed() -> bool {
    state() == ModelState::Failed
}

/// Retry after a terminal `Failed` state: reset it and re-run the background chain.
/// Returns `true` when a retry was initiated.
pub(crate) fn retry_init() -> bool {
    if !STATE.transition(ModelState::Failed, ModelState::Uninit) {
        return false;
    }
    spawn_background_init();
    true
}

/// Demand the engine: start the load-or-fetch chain in the background. A no-op when
/// it is already ready, already loading, or terminally failed — the caller has
/// already decided the engine is needed (a first use, a saved toggle, an explicit
/// retry), and nothing here is awaited.
pub(crate) fn spawn_background_init() {
    if !STATE.transition(ModelState::Uninit, ModelState::Loading) {
        debug!("Speech recognition: background init skipped");
        return;
    }
    tokio::spawn(async move {
        // A panic must never leave the state stuck in Loading, silently disabling
        // recognition until a restart.
        if std::panic::AssertUnwindSafe(load_retry_loop())
            .catch_unwind()
            .await
            .is_err()
        {
            warn!("Speech recognition: background init panicked — marking failed");
            STATE.store(ModelState::Failed, Ordering::Release);
        }
    });
}

/// The awaited "make sure it is loaded": a load already in flight is waited out, not
/// reported unavailable — a voice press starts the load a moment before the
/// transcription that needs its result. Only a set already on disk is opened here; a
/// fetch is spawned detached and answers `false` immediately.
pub(crate) async fn ensure_loaded(dir: &Path) -> bool {
    match STATE.load(Ordering::Acquire) {
        ModelState::Ready => return true,
        ModelState::Uninit => {}
        _ => return models::present(dir) && wait_until_settled().await,
    }
    if !STATE.transition(ModelState::Uninit, ModelState::Loading) {
        return STATE.load(Ordering::Acquire) == ModelState::Ready;
    }
    if models::present(dir) {
        match load_from_disk(dir).await {
            Ok(()) => return true,
            Err(e) => warn!("Speech recognition: {e:#}"),
        }
    }
    // What is left is a fetch, which the caller does not wait for: hand the state back
    // and let the background chain own every later transition.
    STATE.store(ModelState::Uninit, Ordering::Release);
    spawn_background_init();
    false
}

/// Wait for an in-flight load to settle, over any state source: ends as soon as the
/// answer is known — `Ready`, `Failed`/`Uninit`, a chain that has moved on to
/// fetching, or [`LOAD_WAIT_LIMIT`].
async fn wait_until_settled() -> bool {
    let deadline = tokio::time::Instant::now() + LOAD_WAIT_LIMIT;
    loop {
        match STATE.load(Ordering::Acquire) {
            ModelState::Ready => return true,
            // Failed: the load being waited on ended in failure. Uninit: an explicit
            // retry reset it, so the load this call waited on is gone.
            ModelState::Failed | ModelState::Uninit => return false,
            ModelState::Loading if FETCHING.load(Ordering::Acquire) => return false,
            ModelState::Loading => {}
        }
        if tokio::time::Instant::now() >= deadline {
            warn!("Speech recognition: the model load did not settle in time");
            return false;
        }
        tokio::time::sleep(LOAD_WAIT_POLL).await;
    }
}

/// One background chain: check-or-fetch the set, open the engine, retry with backoff
/// until the attempt budget is spent.
async fn load_retry_loop() {
    let _guard = ModelLoadGuard::new(&STATE);
    let mut delay = Duration::from_secs(RETRY_BASE_SECS);
    for attempt in 1..=MAX_LOAD_RETRIES {
        info!("Speech recognition: model load attempt {attempt}/{MAX_LOAD_RETRIES}");
        match load_or_fetch(&mut |_dir| {
            FETCHING.store(true, Ordering::Release);
        })
        .await
        {
            Ok(()) => return,
            Err(e) => warn!("Speech recognition: {e:#}"),
        }
        // A shutdown or drain ends the chain rather than sleeping through it.
        if !crate::shutdown::sleep_or_shutdown_or_drain(delay).await {
            return;
        }
        delay = (delay * 2).min(Duration::from_mins(2));
    }
    warn!("Speech recognition: the model set could not be loaded; giving up");
    STATE.store(ModelState::Failed, Ordering::Release);
}

/// Open the engine over a set already on disk — verified, never fetched: a recording is
/// waiting for its transcription, and a fetch takes minutes.
async fn load_from_disk(dir: &Path) -> Result<()> {
    models::verify_set(dir).await?;
    open_and_publish().await
}

/// Check the set, fetch what is missing or wrong, and open the engine over it. The only
/// chain that may download, so only the background one and `mahbot audio` use it.
async fn load_or_fetch(fetch_starting: &mut impl FnMut(&Path)) -> Result<()> {
    // A load that finds the set whole is a read, which a caller waits out; the flag is
    // what tells it a fetch has begun instead.
    FETCHING.store(false, Ordering::Release);
    let dir = models::root()?;
    let client = download_client()?;
    models::ensure_in(&dir, &client, fetch_starting).await?;
    open_and_publish().await
}

/// Open the engine over the verified set and publish it as ready.
async fn open_and_publish() -> Result<()> {
    let engine = tokio::task::spawn_blocking(open)
        .await
        .context("the engine's load task panicked")??;
    *ENGINE.lock().unwrap_poison() = Some(engine);
    STATE.store(ModelState::Ready, Ordering::Release);
    info!("Speech recognition: the engine is ready");
    Ok(())
}

/// Open the engine over the verified set: the recognition pair and the audio tagger,
/// both resolved from the machine-wide root.
fn open() -> Result<Arc<Mutex<Transcriber>>> {
    let transcriber = Transcriber::open_product(CEILING_MIB)?;
    Ok(Arc::new(Mutex::new(transcriber)))
}

/// Obtain the set once, with a notice before a fetch that really begins, and open the
/// engine — the command line's own entry point: one attempt, no retry loop, and a
/// failure is the caller's to report.
pub(crate) async fn ensure_ready_once(announce: impl FnOnce(&Path)) -> Result<()> {
    let mut said = Some(announce);
    let mut fetch_starting = |dir: &Path| {
        if let Some(say) = said.take() {
            say(dir);
        }
    };
    load_or_fetch(&mut fetch_starting).await
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
    anyhow::ensure!(ensure_loaded(&dir).await, "{}", unloaded_message(&dir));
    let engine = ENGINE
        .lock()
        .unwrap_poison()
        .clone()
        .context("the recogniser is not loaded")?;

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

/// Why [`ensure_loaded`] answered `false`, in the terms the caller reports: a terminal
/// load failure and a load (or fetch) still in progress are different news for someone
/// who just spoke. Both name the directory the set lives in, so the failure carries the
/// path rather than sending the owner to the log for it.
fn unloaded_message(dir: &Path) -> String {
    if is_failed() {
        format!(
            "the local speech model under {} failed to load (the loader logged why)",
            dir.display()
        )
    } else {
        format!(
            "the local speech model under {} is still loading",
            dir.display()
        )
    }
}

/// Create the HTTP client a fetch uses: no client-wide timeout, because each request
/// carries its own bound (a 1.28 GB body needs a long one), and the ring TLS provider
/// the rest of the product installs.
fn download_client() -> Result<reqwest::Client> {
    crate::util::http::install_ring_provider();
    reqwest::Client::builder()
        .user_agent(concat!(
            "mahbot/",
            env!("CARGO_PKG_VERSION"),
            " (speech model downloader)"
        ))
        .build()
        .context("Failed to create HTTP client for model download")
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
