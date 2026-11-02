//! Text-to-Speech using Supertone's Supertonic 3 model.
//!
//! The speech engine behind the `say` tool ([`crate::tools::SayTool`]): one short
//! phrase the admin's Assistant asks to have spoken, rendered and played on the
//! machine the product runs on. Synthesis runs **asynchronously** and
//! **non-blocking** — [`say_phrase`] returns as soon as the phrase is handed to the
//! engine, and playback happens in the background.
//!
//! Nothing is spoken by itself: an assistant reply, an automation or a confirmation
//! reaches the engine only through a `say` call. A phrase is then spoken when speech
//! output is enabled ([`is_config_enabled`]), the Supertonic 3 files are cached and
//! loaded, an output device could be opened, and someone is at the computer
//! ([`crate::user_activity`]) — a phrase is not spoken to an empty room. A newer
//! phrase replaces the one still being spoken ([`cancel_playback`]).
//!
//! # Pipeline (Supertonic 3)
//!
//! Four ONNX stages: a **Duration Predictor** (token durations from token IDs), a
//! **Text Encoder**, a **Vector Estimator** (8-step flow matching from noise to
//! latent vector) and a **Vocoder** (latent vectors to PCM audio).
//!
//! # State machine
//!
//! Loading runs through [`crate::util::model_state::ModelState`]: `UNINIT` ([`init_global()`]
//! must be called), `LOADING` (downloading, retry loop), `READY`, and `FAILED`
//! (terminal download or load failure).

use crate::audio::models_subdir;
use crate::config::CONFIG;
use crate::onnx::simple_eval;
use crate::util::UnwrapPoison;
use crate::util::model_state::{AtomicModelState, ModelLoadGuard, ModelState};
use anyhow::{Context as _, Result, anyhow};
use candle_core::{Device, Tensor};
use rodio::{DeviceSinkBuilder, MixerDeviceSink, Player};
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Cursor;
use std::ops::AsyncFn;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, OnceLock, RwLock};
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info, warn};

const MODEL_DIR_NAME: &str = "supertonic3";
const MODEL_REPO: &str = "Supertone/supertonic-3";
// Pinned to the "Initial Supertonic 3 release" commit (verified on HuggingFace).
// This is the commit from which the SHA256 constants below were computed.
const MODEL_REVISION: &str = "724fb5abbf5502583fb520898d45929e62f02c0b";
const HF_BASE: &str = "https://huggingface.co";
const MODEL_DOWNLOAD_TIMEOUT: Duration = Duration::from_mins(10);
const DEFAULT_TOTAL_STEPS: usize = 8;
const SPEED_FACTOR: f32 = 1.05;
const MAX_CHUNK_LENGTH: usize = 300;
/// The seed every phrase is rendered with: fixed, so the same text and voice always give
/// the same waveform — the file the command line writes and the app speaks are one audio.
const PLAYBACK_SEED: u64 = 42;
/// The pause between two spoken parts, in milliseconds: playback waits it out, and a
/// reading rendered to a file carries the same pause written into its audio.
const TTS_CHUNK_SILENCE_MS: u64 = 300;
// Timeout per-chunk receive: guards against hung synthesis (ONNX deadlock).
const SYNTHESIS_CHUNK_TIMEOUT: Duration = Duration::from_mins(5);

/// Reverb tail in milliseconds — how long after playback ends to keep [`PLAYBACK_ACTIVE`]
/// asserted, suppressing wake word detection while acoustics and output buffers drain.
const PLAYBACK_REVERB_TAIL_MS: u64 = 300;

// Expected SHA256 hex digests per model/config file: when non-empty, the hash is verified
// on every access (cache reuse and fresh download alike).

pub(crate) const DP_MODEL_SHA256: &str =
    "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db";
pub(crate) const TEXT_ENC_MODEL_SHA256: &str =
    "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff";
pub(crate) const VECTOR_EST_MODEL_SHA256: &str =
    "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c";
pub(crate) const VOCODER_MODEL_SHA256: &str =
    "085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba";
pub(crate) const TTS_JSON_SHA256: &str =
    "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09";
pub(crate) const UNICODE_INDEXER_SHA256: &str =
    "9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f";
pub(crate) const VOICE_STYLE_SHA256: &str =
    "e35604687f5d23694b8e91593a93eec0e4eca6c0b02bb8ed69139ab2ea6b0a5b";

const ONNX_DIR: &str = "onnx";
const VOICE_STYLES_DIR: &str = "voice_styles";
const DP_ONNX_NAME: &str = "duration_predictor.onnx";
const TEXT_ENC_ONNX_NAME: &str = "text_encoder.onnx";
const VECTOR_EST_ONNX_NAME: &str = "vector_estimator.onnx";
const VOCODER_ONNX_NAME: &str = "vocoder.onnx";
const TTS_JSON_NAME: &str = "tts.json";
const UNICODE_INDEXER_NAME: &str = "unicode_indexer.json";
pub(crate) const DEFAULT_VOICE_NAME: &str = "M1.json";

/// The six `onnx/` model+config files as `(filename, expected sha256)`.
const ONNX_TTS_FILES: &[(&str, &str)] = &[
    (DP_ONNX_NAME, DP_MODEL_SHA256),
    (TEXT_ENC_ONNX_NAME, TEXT_ENC_MODEL_SHA256),
    (VECTOR_EST_ONNX_NAME, VECTOR_EST_MODEL_SHA256),
    (VOCODER_ONNX_NAME, VOCODER_MODEL_SHA256),
    (TTS_JSON_NAME, TTS_JSON_SHA256),
    (UNICODE_INDEXER_NAME, UNICODE_INDEXER_SHA256),
];

/// All 10 voice styles: F1-F5 female, M1-M5 male.
pub(crate) const ALL_VOICE_STYLE_NAMES: &[&str] = &[
    "F1.json", "F2.json", "F3.json", "F4.json", "F5.json", "M1.json", "M2.json", "M3.json",
    "M4.json", "M5.json",
];

/// The speech model's own language tags: the 31 languages upstream publishes plus `na`, its
/// language-agnostic fallback. The setting's value is put in the tag as it stands.
pub(crate) const TTS_LANGUAGES: &[&str] = &[
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
    "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
];

/// Atomic model-loading state (Uninit → Loading → Ready / Failed), shared
/// [`crate::util::model_state::ModelState`] plus its [`crate::util::model_state::ModelLoadGuard`]
/// for panic safety.
static STATE: AtomicModelState = AtomicModelState::new(ModelState::Uninit);
static GLOBAL_TTS: OnceLock<RwLock<Option<Arc<TtsEngine>>>> = OnceLock::new();
static CANCEL_TX: OnceLock<broadcast::Sender<()>> = OnceLock::new();
static CANCEL_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// Reference-counted signal that TTS audio is currently playing.
///
/// Incremented when the first chunk begins playback and decremented after the last chunk
/// plus the reverb tail ([`PLAYBACK_REVERB_TAIL_MS`]). A counter rather than a boolean, so
/// overlapping calls keep the count above zero and the voice pipeline keeps suppressing wake
/// word detection instead of hearing the system's own speech.
static PLAYBACK_ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// Events emitted during TTS model download for GUI progress reporting.
#[derive(Debug, Clone)]
pub enum TtsDownloadEvent {
    FileStarted {
        name: String,
        total_bytes: u64,
    },
    FileProgress {
        name: String,
        bytes_downloaded: u64,
        total_bytes: u64,
    },
    FileCompleted {
        name: String,
    },
    Complete,
    Failed {
        error: String,
    },
}

/// Broadcast channel for TTS download progress events (GUI subscription).
pub static DOWNLOAD_EVENTS: OnceLock<broadcast::Sender<TtsDownloadEvent>> = OnceLock::new();

/// Wrapper around rodio's audio output.
///
/// rodio 0.22 splits output into a `MixerDeviceSink` (owns the cpal stream) and a cheap
/// `Arc`-cloned `Mixer`; playback and the wake-word cues ([`crate::audio::cue`]) both create
/// players from that mixer. Both members are `Send + Sync` since cpal 0.17 dropped its
/// `NotSendSyncAcrossAllPlatforms` marker, so the wrapper auto-derives both traits and a
/// future bump that re-introduces a non-`Send` type fails to compile at the `static`.
struct AudioOutputWrapper {
    _sink: MixerDeviceSink,
    mixer: rodio::mixer::Mixer,
}

static AUDIO_OUTPUT: OnceLock<AudioOutputWrapper> = OnceLock::new();
/// Serializes lazy device-opens in [`ensure_audio_output()`] so a costly `open_default_sink`
/// call is never raced by concurrent players.
static AUDIO_OUTPUT_OPEN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct TtsConfig {
    tts_version: String,
    ttl: TtlConfig,
    ae: AeConfig,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct TtlConfig {
    latent_dim: usize,
    chunk_compress_factor: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct AeConfig {
    sample_rate: u32,
    base_chunk_size: usize,
}

struct TtsEngine {
    dp_model: crate::onnx::Model,
    text_enc_model: crate::onnx::Model,
    vector_est_model: crate::onnx::Model,
    vocoder_model: crate::onnx::Model,
    unicode_indexer: Vec<i32>,
    sample_rate: u32,
    latent_dim: usize,
    chunk_compress_factor: usize,
    base_chunk_size: usize,
    device: Device,
}

/// A single style entry from the HuggingFace voice style JSON:
/// `{"data": [[[ ... ]]], "dims": [1, 8, 16], "type": "float32"}` — `data` is a 3D
/// `[batch, rows, cols]` array; the other two are ignored by serde.
#[derive(Debug, Deserialize)]
struct StyleEntry {
    data: Vec<Vec<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
struct VoiceStyleFile {
    #[serde(rename = "style_dp")]
    style_dp: StyleEntry,
    #[serde(rename = "style_ttl")]
    style_ttl: StyleEntry,
}

/// Returns `true` if TTS is enabled in config (regardless of model state), so a caller can
/// avoid an unnecessary download. TTS is opt-in: `tts_enabled` must be set to `"true"`.
#[must_use]
pub fn is_config_enabled() -> bool {
    let enabled = CONFIG.tts_enabled();
    enabled.as_deref() == Some("true")
}

#[must_use]
pub fn models_ready() -> bool {
    STATE.is_ready()
}

/// Returns `true` if the audio output device was successfully initialized, through
/// [`ensure_audio_output()`] or its silent counterpart. Models may be loaded while output is
/// unavailable (headless system, no speakers, CoreAudio failure). A device disconnected after
/// initialization is not reflected.
#[must_use]
pub fn audio_output_ready() -> bool {
    AUDIO_OUTPUT.get().is_some()
}

/// Open the OS audio output device on demand (idempotent, best-effort); returns whether
/// output is ready.
///
/// Nothing opens the device at init: the speech-enabled boot and the Settings toggle open it,
/// a phrase that finds it closed kicks an open through [`say_phrase`], and the wake-word cues
/// open it through [`ensure_audio_output_silent()`]. On failure this warns and leaves
/// [`AUDIO_OUTPUT`] unset; a later call retries.
#[must_use]
pub fn ensure_audio_output() -> bool {
    open_audio_output(true)
}

/// [`ensure_audio_output`] for the wake-word cues ([`crate::audio::cue`]), except that a
/// failed open is not warned about: a host without audio output stays silent.
#[must_use]
pub(crate) fn ensure_audio_output_silent() -> bool {
    open_audio_output(false)
}

/// Mixer of the shared output device, for playback that is not a spoken phrase. `None` until
/// the device has been opened.
pub(crate) fn playback_mixer() -> Option<rodio::mixer::Mixer> {
    AUDIO_OUTPUT.get().map(|output| output.mixer.clone())
}

/// Shared body of [`ensure_audio_output`] and [`ensure_audio_output_silent`];
/// `warn_on_failure` picks the log level.
fn open_audio_output(warn_on_failure: bool) -> bool {
    if AUDIO_OUTPUT.get().is_some() {
        return true;
    }
    // Re-check inside the guard: another thread may have won the race already.
    let _guard = AUDIO_OUTPUT_OPEN_LOCK.lock().unwrap_poison();
    if AUDIO_OUTPUT.get().is_some() {
        return true;
    }
    match DeviceSinkBuilder::open_default_sink() {
        Ok(sink) => {
            let mixer = sink.mixer().clone();
            if AUDIO_OUTPUT
                .set(AudioOutputWrapper { _sink: sink, mixer })
                .is_err()
            {
                warn!("TTS: audio output already initialized by another caller");
            }
        }
        Err(e) => {
            if warn_on_failure {
                warn!("TTS: failed to initialize audio output — playback will be disabled: {e}");
            } else {
                debug!("Audio output unavailable — playback stays silent: {e}");
            }
        }
    }
    AUDIO_OUTPUT.get().is_some()
}

/// Returns `true` if model download has permanently failed (retries exhausted or model
/// directory unresolvable).
#[must_use]
pub fn download_failed() -> bool {
    STATE.load(Ordering::Acquire) == ModelState::Failed
}

/// Retry model download after a previous failure: atomically [`ModelState::Failed`] →
/// [`ModelState::Uninit`] and [`spawn_download()`]. A no-op returning `false` in any other
/// state. The GUI-facing counterpart of [`spawn_download()`], which only starts from Uninit.
#[must_use]
pub fn retry_download() -> bool {
    if !STATE.transition(ModelState::Failed, ModelState::Uninit) {
        return false;
    }
    spawn_download();
    true
}

/// Test-only: set TTS state to a known value for deterministic testing.
#[cfg(test)]
pub(crate) fn test_set_state(state: u8) {
    STATE.store(ModelState::from_u8(state), Ordering::Release);
}

/// What became of one [`say_phrase`] call.
pub(crate) enum SayOutcome {
    Speaking,
    SpeechOff,
    NotReady(NotReady),
    Absent,
}

/// Why the engine could not speak a phrase right now.
pub(crate) enum NotReady {
    ModelsMissing,
    DownloadFailed,
    NoAudioDevice,
}

/// Speak one short phrase for the admin's Assistant, and return at once: the phrase is
/// rendered and played in the background. This is the product's whole speech output.
///
/// Whatever is missing is brought up on the side and never waited for: a model set that is
/// not loaded starts the same download ([`spawn_or_retry_download`]), and a device that is
/// not open is opened on the blocking pool, because enumerating output devices can stall.
pub(crate) fn say_phrase(text: &str) -> SayOutcome {
    if !is_config_enabled() {
        return SayOutcome::SpeechOff;
    }
    // Read before the retry below resets a failed fetch to a fresh attempt.
    let fetch_failed = download_failed();
    if !models_ready() {
        spawn_or_retry_download();
        return SayOutcome::NotReady(if fetch_failed {
            NotReady::DownloadFailed
        } else {
            NotReady::ModelsMissing
        });
    }
    if !audio_output_ready() {
        // Detached on purpose: the open outlives this answer, and a blocking task cannot be
        // cancelled.
        drop(tokio::task::spawn_blocking(ensure_audio_output));
        return SayOutcome::NotReady(NotReady::NoAudioDevice);
    }
    // Last, so the answers above keep precedence: absence only stops speech that would
    // otherwise be produced.
    if crate::user_activity::is_absent() {
        return SayOutcome::Absent;
    }
    speak(text);
    SayOutcome::Speaking
}

/// Hand `text` to the engine: spawn a background task that synthesizes the audio, plays it
/// through the shared output device, then returns.
fn speak(text: &str) {
    if text.trim().is_empty() {
        return;
    }
    // Cancel any previous playback before starting new synthesis, and subscribe to the
    // cancellation channel BEFORE the spawn so the cancel reaches the old task.
    cancel_playback();

    // Subscribed before the spawn: a `cancel_playback()` in between is not lost.
    let cancel_rx = CANCEL_TX.get().map(broadcast::Sender::subscribe);
    tokio::spawn(speak_async(text.to_string(), cancel_rx));
}

/// Cancel any currently-playing TTS audio by signalling the playback task.
///
/// If ONNX synthesis is actively running inside a blocking thread, cancellation may take up
/// to a single chunk cycle (~3–5s), as the flag is only checked between chunks.
fn cancel_playback() {
    if let Some(tx) = CANCEL_TX.get() {
        let _ = tx.send(());
    }
    if let Some(flag) = CANCEL_FLAG.get() {
        flag.store(true, Ordering::Release);
    }
}

/// Returns `true` if TTS audio is currently playing, and for
/// [`PLAYBACK_REVERB_TAIL_MS`] after the last chunk while acoustics and output buffers
/// drain. Used by the voice pipeline to suppress wake word detection.
#[must_use]
pub fn is_playback_active() -> bool {
    PLAYBACK_ACTIVE.load(Ordering::Acquire) > 0
}

/// Initialize the global TTS state.
pub fn init_global() -> Result<()> {
    GLOBAL_TTS
        .set(RwLock::new(None))
        .map_err(|_| anyhow!("GLOBAL_TTS already initialized"))?;
    // Capacity 2: we only ever need 1 receiver at a time, but broadcast needs >0.
    let (tx, _rx) = broadcast::channel(2);
    CANCEL_TX
        .set(tx)
        .map_err(|_| anyhow!("CANCEL_TX already initialized"))?;
    CANCEL_FLAG
        .set(Arc::new(AtomicBool::new(false)))
        .map_err(|_| anyhow!("CANCEL_FLAG already initialized"))?;

    // Initialize download progress broadcast channel
    let (dl_tx, _dl_rx) = broadcast::channel(64);
    DOWNLOAD_EVENTS
        .set(dl_tx)
        .map_err(|_| anyhow!("DOWNLOAD_EVENTS already initialized"))?;

    // The OS audio output device is intentionally NOT opened here: it is opened lazily by
    // [`ensure_audio_output()`] once something needs it.
    Ok(())
}

/// The ONNX model files in `dir`, in the order the set is fetched.
fn onnx_paths(dir: &Path) -> impl Iterator<Item = PathBuf> {
    ONNX_TTS_FILES
        .iter()
        .map(|(name, _)| dir.join(ONNX_DIR).join(name))
}

/// Essential TTS model file paths for a model directory, used to verify the model is fully
/// cached on disk.
fn essential_tts_paths(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<_> = onnx_paths(dir).collect();
    // Also check that at least the default voice style exists
    paths.push(dir.join(VOICE_STYLES_DIR).join(DEFAULT_VOICE_NAME));
    paths
}

/// Try to load TTS models from cache at startup; `false` means the download happens async.
pub fn try_load_cached() -> bool {
    // Fast path: models already loaded. LOADING/FAILED are accepted too — loading from disk
    // is safe regardless of the async download state, and is the recovery mechanism when the
    // download task is orphaned (STATE stuck in LOADING).
    if STATE.load(Ordering::Acquire) == ModelState::Ready {
        return true;
    }

    let Some(dir) = models_subdir(MODEL_DIR_NAME) else {
        return false;
    };

    if !essential_tts_paths(&dir).iter().all(|p| p.exists()) {
        return false;
    }

    match load_engine(&dir) {
        Ok(engine) => {
            set_engine_ready(engine);
            info!("TTS models loaded from cache");
            true
        }
        Err(e) => {
            warn!("Failed to load cached TTS models (will download async): {e}");
            false
        }
    }
}

/// Spawn the background model download retry loop.
pub fn spawn_download() {
    if STATE.transition(ModelState::Uninit, ModelState::Loading) {
        tokio::spawn(download_retry_loop());
    }
}

/// Spawn model download, retrying after a previous failure if needed: unlike
/// [`spawn_download()`] this also resets [`ModelState::Failed`] to Uninit first, which is what
/// the GUI toggle needs after a permanent failure.
pub fn spawn_or_retry_download() {
    if STATE.transition(ModelState::Uninit, ModelState::Loading) {
        tokio::spawn(download_retry_loop());
        return;
    }
    if STATE.transition(ModelState::Failed, ModelState::Uninit) {
        spawn_download();
    }
}

fn set_engine_ready(engine: TtsEngine) {
    if let Some(global) = GLOBAL_TTS.get() {
        *global.write().unwrap_poison() = Some(Arc::new(engine));
    }
    STATE.store(ModelState::Ready, Ordering::Release);
}

/// Clone the engine [`Arc`] cheaply, then drop the read lock — before
/// [`tokio::task::spawn_blocking`], so the `RwLock` is not held across a long CPU-bound operation.
fn get_engine_clone() -> Option<Arc<TtsEngine>> {
    GLOBAL_TTS.get()?.read().unwrap_poison().clone()
}

fn load_voice_style(dir: &Path, voice_name: &str) -> Result<(Tensor, Tensor)> {
    let path = dir.join(VOICE_STYLES_DIR).join(voice_name);
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read voice style: {}", path.display()))?;
    let voice: VoiceStyleFile =
        serde_json::from_str(&content).context("Failed to parse voice style JSON")?;

    let device = Device::Cpu;

    // style_dp: HuggingFace stores as 3D [batch=1, rows=8, cols=16]
    let style_dp = style_tensor(&voice.style_dp.data, "style_dp", &device)?;

    // style_ttl: HuggingFace stores as 3D [batch=1, rows=50, cols=256]
    let style_ttl = style_tensor(&voice.style_ttl.data, "style_ttl", &device)?;

    Ok((style_dp, style_ttl))
}

/// A voice style, loaded: the pair of tensors a reading is spoken with. [`ensure_ready_once`]
/// loads it — that load is how it knows the request can be served from the cache — and hands
/// it to [`render_speech_wav`], so one reading parses the style file once.
pub(crate) struct Voice {
    dp: Tensor,
    ttl: Tensor,
}

impl Voice {
    fn load(dir: &Path, voice_style: &str) -> Result<Self> {
        let (dp, ttl) = load_voice_style(dir, voice_style)?;
        Ok(Self { dp, ttl })
    }
}

/// Flatten the first batch slice of a 3D `[batch, rows, cols]` style tensor into a candle
/// `Tensor` with the batch dimension restored.
fn style_tensor(data: &[Vec<Vec<f32>>], label: &str, device: &Device) -> Result<Tensor> {
    anyhow::ensure!(!data.is_empty(), "Voice style has empty {label}");
    anyhow::ensure!(!data[0].is_empty(), "Voice style has empty {label}[0]");
    let rows = data[0].len();
    let cols = data[0][0].len();
    let flat: Vec<f32> = data[0].iter().flat_map(|v| v.iter()).copied().collect();
    anyhow::ensure!(
        flat.len() == rows * cols,
        "Flat {label} length {} doesn't match {}×{}",
        flat.len(),
        rows,
        cols,
    );
    Ok(Tensor::from_slice(&flat, (1, rows, cols), device)?)
}

fn load_engine(dir: &Path) -> Result<TtsEngine> {
    let onnx_dir = dir.join(ONNX_DIR);

    let config_content =
        std::fs::read_to_string(onnx_dir.join(TTS_JSON_NAME)).context("Failed to read tts.json")?;
    let config: TtsConfig =
        serde_json::from_str(&config_content).context("Failed to parse tts.json")?;

    let sample_rate = config.ae.sample_rate;
    let latent_dim = config.ttl.latent_dim;
    let chunk_compress_factor = config.ttl.chunk_compress_factor;
    let base_chunk_size = config.ae.base_chunk_size;

    let dp_model = crate::onnx::read_file(onnx_dir.join(DP_ONNX_NAME))?;
    let text_enc_model = crate::onnx::read_file(onnx_dir.join(TEXT_ENC_ONNX_NAME))?;
    let vector_est_model = crate::onnx::read_file(onnx_dir.join(VECTOR_EST_ONNX_NAME))?;
    let vocoder_model = crate::onnx::read_file(onnx_dir.join(VOCODER_ONNX_NAME))?;

    let indexer_content = std::fs::read_to_string(onnx_dir.join(UNICODE_INDEXER_NAME))
        .context("Failed to read unicode_indexer.json")?;
    let unicode_indexer: Vec<i32> =
        serde_json::from_str(&indexer_content).context("Failed to parse unicode_indexer.json")?;

    info!(
        "Loaded Supertonic 3 TTS models (version {}, latent {}, {}Hz)",
        config.tts_version, latent_dim, sample_rate,
    );

    Ok(TtsEngine {
        dp_model,
        text_enc_model,
        vector_est_model,
        vocoder_model,
        unicode_indexer,
        sample_rate,
        latent_dim,
        chunk_compress_factor,
        base_chunk_size,
        device: Device::Cpu,
    })
}

/// Prepare the speech engine for one rendering: the model set, loaded from the cache when it
/// is complete and fetched EXACTLY once when it is not, plus the voice that was loaded.
///
/// No download retry loop: the daemon retries while it is up, a command line has one answer
/// to give. `voice_style` names the style file the rendering will use, fetched with the rest
/// when missing. `announce_fetch` runs once, when the FIRST fetch that actually starts
/// begins. A set is complete when the ONNX files are on disk AND that ONE style file loads —
/// [`try_load_cached`] asks for the default style instead, so a caller who asked for another
/// voice does not pay for the whole set again on every run.
///
/// This prepares the engine for ONE operation and expects nothing to be up yet:
/// [`init_global`] refuses a second initialisation.
pub(crate) async fn ensure_ready_once(
    voice_style: &str,
    announce_fetch: impl FnOnce(&Path),
) -> Result<Voice> {
    let dir = models_subdir(MODEL_DIR_NAME).context("Cannot resolve the speech model directory")?;

    // The download events and the cancel channel have to exist before the engine can be
    // parked in them.
    if GLOBAL_TTS.get().is_none() {
        init_global().context("Cannot initialize the speech engine")?;
    }

    // The notice is bound to the first fetch that actually starts: a set already on disk
    // fetches nothing and says nothing.
    let mut announce_fetch = Some(announce_fetch);
    let mut fetch_starting = || {
        if let Some(say) = announce_fetch.take() {
            say(&dir);
        }
    };
    // A set on disk is loaded and verified against nothing: the reading needs the ONNX files
    // and the ONE voice asked for, so a set fetched for another voice serves this one too.
    if onnx_paths(&dir).all(|path| path.exists())
        && let Ok(voice) = Voice::load(&dir, voice_style)
    {
        match load_engine(&dir) {
            Ok(engine) => {
                set_engine_ready(engine);
                return Ok(voice);
            }
            // Present is not usable — a file left truncated by an interrupted fetch is not a
            // set — so this falls through to the verified fetch below.
            Err(e) => warn!("The cached speech models did not load, verifying them: {e}"),
        }
    }

    ensure_models_downloaded(&dir, &[voice_style], &mut fetch_starting).await?;
    let engine = load_engine(&dir).context("Cannot load the speech models")?;
    set_engine_ready(engine);
    Voice::load(&dir, voice_style)
}

/// Emit a download event through the global broadcast channel.
fn emit_download_event(event: TtsDownloadEvent) {
    if let Some(tx) = DOWNLOAD_EVENTS.get() {
        let _ = tx.send(event);
    }
}

async fn download_retry_loop() {
    run_download_retry_loop(
        &STATE,
        MODEL_DIR_NAME,
        "TTS",
        MODEL_DOWNLOAD_TIMEOUT,
        async |dir| ensure_models_downloaded(dir, ALL_VOICE_STYLE_NAMES, &mut || {}).await,
        |dir| {
            let e = load_engine(dir)?;
            set_engine_ready(e);
            emit_download_event(TtsDownloadEvent::Complete);
            info!("TTS models loaded successfully");
            Ok(())
        },
        || {
            let msg = format!("TTS download failed after {MAX_DOWNLOAD_RETRIES} retries");
            warn!("{msg}");
            emit_download_event(TtsDownloadEvent::Failed { error: msg });
            STATE.store(ModelState::Failed, Ordering::Release);
        },
    )
    .await;
}

/// File descriptor: (download URL, local path, expected SHA256 hash).
struct TtsFile {
    url: String,
    path: PathBuf,
    sha256: &'static str,
}

/// The files the speech model set is made of: the ONNX models and config, plus the voice
/// styles named in `styles`.
fn model_files(dir: &Path, styles: &[&str]) -> Vec<TtsFile> {
    let base = format!("{HF_BASE}/{MODEL_REPO}/resolve/{MODEL_REVISION}");

    let mut files: Vec<TtsFile> = ONNX_TTS_FILES
        .iter()
        .map(|(name, sha256)| TtsFile {
            url: format!("{base}/{ONNX_DIR}/{name}"),
            path: dir.join(ONNX_DIR).join(name),
            sha256,
        })
        .collect();

    files.extend(
        styles
            .iter()
            .map(|style_name| voice_style_file(dir, style_name)),
    );
    files
}

/// One voice style file out of the set the product ships.
fn voice_style_file(dir: &Path, name: &str) -> TtsFile {
    TtsFile {
        url: format!("{HF_BASE}/{MODEL_REPO}/resolve/{MODEL_REVISION}/{VOICE_STYLES_DIR}/{name}"),
        path: dir.join(VOICE_STYLES_DIR).join(name),
        // Only M1.json has a verified SHA256; the rest have empty hashes (minimum-size check only).
        sha256: if name == DEFAULT_VOICE_NAME {
            VOICE_STYLE_SHA256
        } else {
            ""
        },
    }
}

/// Fetch the model set, calling `fetch_starting` before the first file that actually has to
/// be fetched — a file already on disk is verified and skipped.
async fn ensure_models_downloaded(
    dir: &Path,
    styles: &[&str],
    fetch_starting: &mut impl FnMut(),
) -> Result<()> {
    tokio::fs::create_dir_all(dir.join(ONNX_DIR)).await?;
    tokio::fs::create_dir_all(dir.join(VOICE_STYLES_DIR)).await?;

    // Sequential downloads so per-file progress is meaningful
    for f in &model_files(dir, styles) {
        if let Err(e) = ensure_file(f, fetch_starting).await {
            let file_name = f
                .path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            emit_download_event(TtsDownloadEvent::Failed {
                error: format!("Failed to download {file_name}: {e}"),
            });
            return Err(e);
        }
    }

    Ok(())
}

/// Ensure a single file exists and is uncorrupted, or download it. `fetch_starting` is called
/// once, just before the download that actually streams.
async fn ensure_file(f: &TtsFile, fetch_starting: &mut impl FnMut()) -> Result<()> {
    let file_name = f
        .path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let mut started = false;
    let mut last_reported_bytes = 0u64;
    if ensure_downloaded(
        None,
        &f.path,
        &f.url,
        f.sha256,
        100,
        MODEL_DOWNLOAD_TIMEOUT,
        &file_name,
        |d, total_size| {
            // First call (pre-stream) carries total_bytes → FileStarted.
            if !started {
                started = true;
                fetch_starting();
                emit_download_event(TtsDownloadEvent::FileStarted {
                    name: file_name.clone(),
                    total_bytes: total_size,
                });
            }
            // Throttle: emit progress at ~1% granularity to avoid broadcast pressure.
            if total_size > 0 {
                let threshold = (total_size / 100).max(1);
                if d - last_reported_bytes >= threshold || d >= total_size {
                    last_reported_bytes = d;
                    emit_download_event(TtsDownloadEvent::FileProgress {
                        name: file_name.clone(),
                        bytes_downloaded: d,
                        total_bytes: total_size,
                    });
                }
            }
        },
    )
    .await?
    {
        // FileCompleted fires only after a real download (cached files fast-path out).
        emit_download_event(TtsDownloadEvent::FileCompleted { name: file_name });
    }
    Ok(())
}

/// Compiled regex patterns used by [`strip_markdown`].
static RE_CODE_BLOCK: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"```[\s\S]*?```").expect("RE_CODE_BLOCK"));
static RE_INLINE_CODE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"`[^`]+`").expect("RE_INLINE_CODE"));
static RE_IMAGE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"!\[[^\]]*\]\([^)]*\)").expect("RE_IMAGE"));
static RE_LINK: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"\[([^\]]*)\]\([^)]*\)").expect("RE_LINK"));
static RE_BOLD_ITALIC: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"\*{1,3}[^*]+\*{1,3}").expect("RE_BOLD_ITALIC"));
static RE_STRIKETHROUGH: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"~~[^~]+~~").expect("RE_STRIKETHROUGH"));
static RE_HEADER: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"(?m)^#{1,6}\s+").expect("RE_HEADER"));
static RE_LIST_DASH: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"(?m)^[\s]*[-*]\s+").expect("RE_LIST_DASH"));
static RE_LIST_NUM: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"(?m)^\s*\d+\.\s+").expect("RE_LIST_NUM"));
static RE_HR: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"(?m)^[-*_]{3,}\s*$").expect("RE_HR"));

/// Preprocess text: NFKD normalize, strip markdown, clean punctuation.
fn preprocess_text(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;

    // Media markers are display/plumbing artifacts, never speech — neither a path nor a
    // base64 data URI is spoken.
    let text = crate::util::MEDIA_MARKER_RE.replace_all(text, " ");
    let mut s: String = text.as_ref().nfkd().collect();
    s = strip_markdown(&s);
    s = remove_emojis(&s);
    s = normalize_symbols(&s);
    s = expand_abbreviations(&s);
    s = fix_punctuation_spacing(&s);
    s = remove_duplicate_quotes(&s);
    s = clean_whitespace(&s);

    if !has_ending_punctuation(&s) {
        s.push('.');
    }

    s
}

/// Strip markdown syntax that would be read verbatim by TTS.
fn strip_markdown(text: &str) -> String {
    let mut s = text.to_string();

    s = RE_CODE_BLOCK.replace_all(&s, " ").to_string();

    s = RE_INLINE_CODE.replace_all(&s, " ").to_string();

    s = RE_IMAGE.replace_all(&s, " ").to_string();

    s = RE_LINK.replace_all(&s, "$1").to_string();

    s = RE_BOLD_ITALIC.replace_all(&s, " ").to_string();

    s = RE_STRIKETHROUGH.replace_all(&s, " ").to_string();

    s = RE_HEADER.replace_all(&s, "").to_string();

    s = RE_LIST_DASH.replace_all(&s, "").to_string();
    s = RE_LIST_NUM.replace_all(&s, "").to_string();

    s = RE_HR.replace_all(&s, " ").to_string();

    s
}

fn remove_emojis(text: &str) -> String {
    text.chars()
        .filter(|&c| {
            let code = c as u32;
            !matches!(code,
                0x1F600..=0x1F64F | 0x1F300..=0x1F5FF |
                0x1F680..=0x1F6FF | 0x1F700..=0x1F77F |
                0x1F780..=0x1F7FF | 0x1F800..=0x1F8FF |
                0x1F900..=0x1F9FF | 0x1FA00..=0x1FA6F |
                0x1FA70..=0x1FAFF | 0x1FB00..=0x1FBFF |
                0x2600..=0x26FF | 0x2700..=0x27BF)
        })
        .collect()
}

fn normalize_symbols(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{2013}' | '\u{2011}' | '\u{2014}' => '-',
            '\u{00AF}' | '[' | ']' | '|' | '/' | '#' | '\u{2192}' | '\u{2190}' | '\u{2665}'
            | '\u{2606}' | '\u{2661}' | '\u{00A9}' | '\\' => ' ',
            '\u{201C}' | '\u{201D}' => '"',
            '\u{2018}' | '\u{2019}' | '\u{00B4}' | '`' => '\'',
            _ => c,
        })
        .collect()
}

fn expand_abbreviations(text: &str) -> String {
    text.replace('@', " at ")
        .replace("e.g.,", "for example, ")
        .replace("i.e.,", "that is, ")
}

fn fix_punctuation_spacing(text: &str) -> String {
    text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" !", "!")
        .replace(" ?", "?")
        .replace(" ;", ";")
        .replace(" :", ":")
        .replace(" '", "'")
}

fn remove_duplicate_quotes(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' || c == '\'' || c == '`' {
            while chars.peek() == Some(&c) {
                chars.next();
            }
            result.push(c);
        } else {
            result.push(c);
        }
    }
    result
}

fn clean_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_space {
                result.push(' ');
                prev_space = true;
            }
        } else {
            result.push(c);
            prev_space = false;
        }
    }
    result.trim().to_string()
}

fn has_ending_punctuation(text: &str) -> bool {
    text.chars().last().is_some_and(|c| {
        matches!(
            c,
            '.' | '!' | '?' | '\u{3002}' /* 。 */ | '\u{FF01}' /* ！ */ | '\u{FF1F}' /* ？ */
        )
    })
}

/// Encode a text string into token IDs using the engine's unicode indexer.
fn encode_text(engine: &TtsEngine, text: &str) -> (Vec<i64>, usize) {
    encode_text_with_indexer(&engine.unicode_indexer, text)
}

/// Core encoding logic that works with any unicode indexer slice; extracted for testability.
fn encode_text_with_indexer(unicode_indexer: &[i32], text: &str) -> (Vec<i64>, usize) {
    let unk_id = unicode_indexer.first().copied().unwrap_or(0);
    #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let max_idx = unicode_indexer.len() as i32 - 1;
    let ids: Vec<i64> = text
        .chars()
        .map(|c| {
            let code = c as usize;
            if code < unicode_indexer.len() {
                let tid = unicode_indexer[code];
                if tid >= 0 && tid <= max_idx {
                    return i64::from(tid);
                }
            }
            i64::from(unk_id)
        })
        .collect();
    let len = ids.len();
    (ids, len)
}

fn build_inputs(inputs: Vec<(&str, Tensor)>) -> HashMap<String, Tensor> {
    let mut map = HashMap::new();
    for (name, tensor) in inputs {
        map.insert(name.to_string(), tensor);
    }
    map
}

/// Synthesize audio for a single text chunk. Returns PCM f32 samples.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn synthesize_internal(
    engine: &TtsEngine,
    text: &str,
    style_dp: &Tensor,
    style_ttl: &Tensor,
    seed: u64,
) -> Result<Vec<f32>> {
    let dev = &engine.device;

    let (token_ids, seq_len) = encode_text(engine, text);
    if seq_len == 0 {
        anyhow::bail!("Empty token sequence");
    }

    let text_ids = Tensor::from_slice(&token_ids, (1, seq_len), dev)?;
    let text_mask = Tensor::from_slice(&vec![1.0f32; seq_len], (1, 1, seq_len), dev)?;

    let dp_out = simple_eval(
        &engine.dp_model,
        build_inputs(vec![
            ("text_ids", text_ids.clone()),
            ("style_dp", style_dp.clone()),
            ("text_mask", text_mask.clone()),
        ]),
    )
    .context("Duration predictor failed")?;
    let duration_t = extract_output(dp_out, &engine.dp_model, "dp")?;
    let durations: Vec<f32> = duration_t
        .to_vec1::<f32>()?
        .into_iter()
        .map(|d| d / SPEED_FACTOR)
        .collect();
    let total_dur: f32 = durations.iter().sum();

    let te_out = simple_eval(
        &engine.text_enc_model,
        build_inputs(vec![
            ("text_ids", text_ids),
            ("style_ttl", style_ttl.clone()),
            ("text_mask", text_mask.clone()),
        ]),
    )
    .context("Text encoder failed")?;
    let text_emb = extract_output(te_out, &engine.text_enc_model, "te")?;

    let wav_len = (total_dur * engine.sample_rate as f32).ceil() as usize;
    let chunk_size = engine.base_chunk_size * engine.chunk_compress_factor;
    let latent_len = wav_len.div_ceil(chunk_size);
    let latent_dim = engine.latent_dim * engine.chunk_compress_factor;

    let mut rng = seed;
    let noise: Vec<f32> = {
        let n = latent_dim * latent_len;
        let mut result = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            // Advance xorshift twice to get two uniform (0,1] samples.
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let u1 = (rng as f64) / (u64::MAX as f64);

            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let u2 = (rng as f64) / (u64::MAX as f64);

            // Box-Muller: two independent N(0,1) samples from two uniforms.
            let r = (-2.0 * u1.clamp(f64::MIN_POSITIVE, 1.0).ln()).sqrt();
            let theta = (2.0 * std::f64::consts::PI) * u2;
            result.push((r * theta.cos()) as f32);
            i += 1;
            if i < n {
                result.push((r * theta.sin()) as f32);
                i += 1;
            }
        }
        result
    };
    let mut xt = Tensor::from_slice(&noise, (1, latent_dim, latent_len), dev)?;

    let latent_mask = Tensor::from_slice(&vec![1.0f32; latent_len], (1, 1, latent_len), dev)?;

    let total_steps = Tensor::new(DEFAULT_TOTAL_STEPS as f32, dev)?.reshape((1,))?;
    for step in 0..DEFAULT_TOTAL_STEPS {
        let step_f = Tensor::new(step as f32, dev)?.reshape((1,))?;
        let ve_out = simple_eval(
            &engine.vector_est_model,
            build_inputs(vec![
                ("noisy_latent", xt.clone()),
                ("text_emb", text_emb.clone()),
                ("style_ttl", style_ttl.clone()),
                ("latent_mask", latent_mask.clone()),
                ("text_mask", text_mask.clone()),
                ("current_step", step_f),
                ("total_step", total_steps.clone()),
            ]),
        )
        .with_context(|| format!("Vector estimator step {step} failed"))?;
        // The Supertonic 3 vector estimator uses x₀-prediction: at each flow-matching step it
        // predicts the clean latent directly, so we replace xt rather than accumulate a
        // velocity-field integral.
        let denoised_latent = extract_output(ve_out, &engine.vector_est_model, "ve")?;
        xt = denoised_latent;
    }

    let voc_out = simple_eval(&engine.vocoder_model, build_inputs(vec![("latent", xt)]))
        .context("Vocoder failed")?;
    let wav_t = extract_output(voc_out, &engine.vocoder_model, "voc")?;

    let wav_data = if wav_t.dims().len() >= 2 {
        wav_t.squeeze(0)?.to_vec1::<f32>()?
    } else {
        wav_t.to_vec1::<f32>()?
    };

    // Clamp to [-1.0, 1.0]: the vocoder can overshoot through floating-point accumulation.
    let wav_data: Vec<f32> = wav_data.into_iter().map(|s| s.clamp(-1.0, 1.0)).collect();

    Ok(wav_data)
}

/// Synthesize with chunking for long texts, streaming each chunk through a bounded channel
/// as it becomes available. The reading is split into parts ([`reading_parts`]), each spoken
/// under the language tag `lang` (the model requires `<en>...</en>` wrapping).
///
/// # Error propagation (intentional asymmetry)
///
/// A part that fails to render is skipped with a `warn!` — one bad sentence should not abort
/// the whole response — except when it is the only part: there is then no audio at all, and
/// the error is returned. Cancellation and a dropped receiver are always fatal.
#[expect(clippy::too_many_arguments)]
fn synthesize_chunked_streaming(
    engine: &TtsEngine,
    text: &str,
    style_dp: &Tensor,
    style_ttl: &Tensor,
    lang: &str,
    cancel: Option<&AtomicBool>,
    tx: &mpsc::Sender<Vec<f32>>,
    seed: u64,
) -> Result<()> {
    // Check cancellation before starting any work
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        anyhow::bail!("Synthesis cancelled");
    }

    let parts = reading_parts(text);
    let single = parts.len() == 1;
    for part in parts {
        // Check cancellation between chunks
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            anyhow::bail!("Synthesis cancelled");
        }
        match synthesize_part(engine, &part, style_dp, style_ttl, lang, seed) {
            Ok(samples) => {
                if tx.blocking_send(samples).is_err() {
                    anyhow::bail!("Synthesis cancelled (receiver dropped)");
                }
            }
            Err(e) if !single => warn!("TTS: chunk synthesis failed, skipping chunk: {e}"),
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// The parts of one reading, in the order the product speaks them: one part when the text
/// fits [`MAX_CHUNK_LENGTH`], else the sentence-boundary split.
fn reading_parts(text: &str) -> Vec<String> {
    if text.len() <= MAX_CHUNK_LENGTH {
        vec![text.to_string()]
    } else {
        split_at_sentence_boundaries(text, MAX_CHUNK_LENGTH)
    }
}

/// Render one part of a reading under the language tag it is spoken in.
fn synthesize_part(
    engine: &TtsEngine,
    part: &str,
    style_dp: &Tensor,
    style_ttl: &Tensor,
    lang: &str,
    seed: u64,
) -> Result<Vec<f32>> {
    synthesize_internal(
        engine,
        &wrap_lang_tag(part, lang),
        style_dp,
        style_ttl,
        seed,
    )
}

/// Wrap text in a language tag for the Supertonic 3 model. The language is the caller's — the
/// speech-output setting's value (defaults to `"na"`) or the command line's `--language` — and
/// is one of [`TTS_LANGUAGES`].
fn wrap_lang_tag(text: &str, lang: &str) -> String {
    format!("<{lang}>{text}</{lang}>")
}

fn split_at_sentence_boundaries(text: &str, max_len: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for sentence in text.split_inclusive(['.', '!', '?', '\n']) {
        let s = sentence.trim();
        if s.is_empty() {
            continue;
        }
        if current.len() + s.len() > max_len && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
        }
        if s.len() > max_len {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            let mut buf = String::new();
            for word in s.split_whitespace() {
                if buf.len() + word.len() + 1 > max_len && !buf.is_empty() {
                    chunks.push(std::mem::take(&mut buf));
                }
                if !buf.is_empty() {
                    buf.push(' ');
                }
                buf.push_str(word);
            }
            if !buf.is_empty() {
                current = buf;
            }
        } else {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(s);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Render PCM float samples to an in-memory WAV file (16-bit mono PCM), header and data
/// included. Rodio plays directly from a `Cursor<Vec<u8>>`, so no ephemeral file I/O.
pub(crate) fn render_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let channels: u16 = 1;
    let bps: u16 = 16;
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bps / 8);
    let block_align = channels * (bps / 8);
    let data_size = u32::try_from(samples.len() * (bps / 8) as usize)
        .context("WAV data size exceeds u32 range")?;
    let file_size = 36 + data_size;

    let mut buf = Vec::with_capacity(44 + data_size as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&(16u32).to_le_bytes());
    buf.extend_from_slice(&(1u16).to_le_bytes()); // PCM
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bps.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    // Convert all samples to little-endian i16 bytes
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        #[expect(clippy::cast_possible_truncation)]
        let sample_i16 = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&sample_i16.to_le_bytes());
    }

    Ok(buf)
}

/// Render `text` to the spoken file's bytes: the product's own reading path — the same
/// preparation, split, per-part rendering and language tag, with the pause between parts
/// written into the audio rather than waited out — assembled into one uncompressed WAV at the
/// engine's own sample rate.
///
/// Unlike playback, a part that fails to render fails the whole reading: a file delivered with
/// a part missing would be a lie about what the text says.
pub(crate) async fn render_speech_wav(text: &str, voice: Voice, lang: &str) -> Result<Vec<u8>> {
    let processed = preprocess_text(text);

    let engine = get_engine_clone().context("the speech engine is not loaded")?;
    let sample_rate = engine.sample_rate;

    let lang = lang.to_string();
    tokio::task::spawn_blocking(move || {
        let Voice {
            dp: style_dp,
            ttl: style_ttl,
        } = voice;
        // The pause playback waits out between parts, in samples.
        let gap = usize::try_from(u64::from(sample_rate) * TTS_CHUNK_SILENCE_MS / 1000)
            .unwrap_or_default();
        let mut pcm: Vec<f32> = Vec::new();
        for part in reading_parts(&processed) {
            let samples =
                synthesize_part(&engine, &part, &style_dp, &style_ttl, &lang, PLAYBACK_SEED)?;
            if !pcm.is_empty() {
                pcm.resize(pcm.len() + gap, 0.0);
            }
            pcm.extend_from_slice(&samples);
        }
        render_wav(&pcm, sample_rate)
    })
    .await
    .context("the speech synthesis task panicked")?
}

#[expect(clippy::too_many_lines)]
async fn speak_async(text: String, mut cancel_rx: Option<broadcast::Receiver<()>>) {
    // Check cancellation before starting expensive work
    if let Some(ref mut rx) = cancel_rx
        && rx.try_recv().is_ok()
    {
        info!("TTS synthesis cancelled before start");
        return;
    }

    // Process text (lightweight, no blocking I/O)
    let processed = preprocess_text(&text);

    // The speech-output language, read once: the synthesis runs on a blocking thread that
    // must not touch the config.
    let lang = CONFIG.tts_language();

    // The engine Arc is cloned outside the blocking task so the `RwLock` is not held across
    // CPU-bound synthesis.
    let Some(engine) = get_engine_clone() else {
        return;
    };
    let sample_rate = engine.sample_rate;

    // Load the default voice style (M1) for playback.
    let Some(dir) = models_subdir(MODEL_DIR_NAME) else {
        return;
    };
    let (style_dp, style_ttl) = match load_voice_style(&dir, DEFAULT_VOICE_NAME) {
        Ok(styles) => styles,
        Err(e) => {
            warn!("TTS: failed to load default voice style: {e}");
            return;
        }
    };

    // Check output availability BEFORE the expensive synthesis: on headless systems this
    // avoids ~3-5s of work per chunk before the channel close is detected.
    let Some(audio_output) = AUDIO_OUTPUT.get() else {
        return;
    };

    // Bounded channel: capacity 4 gives natural backpressure — a synthesizer that outpaces
    // playback blocks on send after 4 queued chunks.
    let (tx, mut rx) = mpsc::channel::<Vec<f32>>(4);

    // Reset the cancellation flag for this synthesis, then clone the Arc for the
    // spawn_blocking closure.
    let cancel_flag = CANCEL_FLAG.get().map(|f| {
        f.store(false, Ordering::Release);
        Arc::clone(f)
    });

    // CPU-bound synthesis on the blocking threadpool, streaming each chunk through the channel.
    let synthesize_handle = tokio::task::spawn_blocking(move || {
        synthesize_chunked_streaming(
            &engine,
            &processed,
            &style_dp,
            &style_ttl,
            &lang,
            cancel_flag.as_deref(),
            &tx,
            PLAYBACK_SEED,
        )
    });

    // A logging task observes the synthesis result: errors and panics are logged even though
    // the blocking task is never awaited.
    tokio::spawn(async move {
        match synthesize_handle.await {
            Ok(Ok(())) => {} // Success — normal completion
            Ok(Err(e)) => warn!("TTS synthesis failed: {e}"),
            Err(e) => warn!("TTS synthesis task panicked: {e}"),
        }
    });

    // Receive chunks and play them as they arrive.
    let mut current_player: Option<Player> = None;

    // A per-chunk timeout guards against hung synthesis. On timeout the loop breaks, which
    // drops the receiver and unblocks the sender with a channel error.
    loop {
        let chunk_samples = match tokio::time::timeout(SYNTHESIS_CHUNK_TIMEOUT, rx.recv()).await {
            Ok(Some(samples)) => samples,
            Ok(None) => {
                // Channel closed cleanly — synthesis finished.
                break;
            }
            Err(_) => {
                warn!("TTS: timed out waiting for synthesis chunk (possible hang)");
                break;
            }
        };

        // Check cancellation before playing this chunk
        if let Some(ref mut rx) = cancel_rx
            && rx.try_recv().is_ok()
        {
            info!("TTS playback cancelled mid-stream");
            if let Some(ref player) = current_player {
                player.stop();
                // Wait briefly for the audio thread to flush, so no truncated burst is left
                // in the output buffer.
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            break;
        }

        // Wait for the previous chunk, then add the inter-chunk silence.
        if let Some(ref player) = current_player {
            let completed = wait_for_player(player, cancel_rx.as_mut()).await;
            if !completed {
                // Cancelled while waiting — player.stop() was already called inside
                // wait_for_player.
                tokio::time::sleep(Duration::from_millis(20)).await;
                break;
            }
            // Natural pause between speech chunks
            tokio::time::sleep(Duration::from_millis(TTS_CHUNK_SILENCE_MS)).await;
        }

        // Render WAV bytes for this chunk (in-memory, no file I/O).
        let wav_bytes = match render_wav(&chunk_samples, sample_rate) {
            Ok(bytes) => bytes,
            Err(e) => {
                warn!("TTS: failed to render WAV chunk: {e}");
                continue;
            }
        };

        // Decode the WAV bytes as an in-memory source (hound-backed, no Symphonia path).
        let cursor = Cursor::new(wav_bytes);
        let source = match rodio::Decoder::new(cursor) {
            Ok(s) => s,
            Err(e) => {
                warn!("TTS: failed to decode WAV chunk: {e}");
                continue;
            }
        };

        // A fresh player per chunk (connected to the shared mixer) so playback can be
        // cancelled per chunk. `Player::connect_new` is infallible.
        let player = Player::connect_new(&audio_output.mixer);

        if current_player.is_none() {
            // Only on the first chunk — the single fetch_sub(1) after the reverb tail
            // restores the counter to zero.
            PLAYBACK_ACTIVE.fetch_add(1, Ordering::Relaxed);
            debug!("TTS playback started — wake word detection suppressed");
        }
        player.append(source);
        current_player = Some(player);
    }

    // Wait for the last chunk to finish playing
    if let Some(ref player) = current_player {
        let _ = wait_for_player(player, cancel_rx.as_mut()).await;
    }

    // Reverb tail: keep the playback count > 0 briefly after playback ends, so room acoustics
    // and buffer drain cannot trigger the wake word. A counter, not a boolean, so a call that
    // starts during this window keeps the count >= 1.
    if current_player.is_some() {
        tokio::time::sleep(Duration::from_millis(PLAYBACK_REVERB_TAIL_MS)).await;
        PLAYBACK_ACTIVE.fetch_sub(1, Ordering::Release);
    }
}

/// Wait for a rodio player to finish playback, polling with optional cancellation support.
/// Returns `true` if it completed normally, `false` if cancelled or if the player did not
/// drain within [`PLAYER_WAIT_TIMEOUT`].
async fn wait_for_player(player: &Player, cancel_rx: Option<&mut broadcast::Receiver<()>>) -> bool {
    /// Maximum time to wait for the player to drain before giving up.
    const PLAYER_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

    let poll_loop = async {
        if let Some(rx) = cancel_rx {
            loop {
                tokio::time::sleep(Duration::from_millis(50)).await;

                if player.empty() {
                    return true;
                }

                if rx.try_recv().is_ok() {
                    info!("TTS playback cancelled");
                    player.stop();
                    return false;
                }
            }
        } else {
            // No cancellation receiver — just wait until playback finishes.
            while !player.empty() {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            true
        }
    };

    if let Ok(result) = tokio::time::timeout(PLAYER_WAIT_TIMEOUT, poll_loop).await {
        result
    } else {
        warn!("TTS player wait timed out after 30s");
        player.stop();
        false
    }
}

// ── Model set plumbing ────────────────────────────────────────────────────
//
// What the four TTS stages share: the output extractor, the per-model
// directory, the download-retry skeleton and the verify-then-download helper.

fn extract_output(
    mut outputs: HashMap<String, Tensor>,
    model: &crate::onnx::Model,
    label: &str,
) -> Result<Tensor> {
    let name = model.output_name().to_string();
    outputs
        .remove(&name)
        .ok_or_else(|| anyhow!("{label}: output '{name}' not found"))
}

/// Retry cap for the TTS model download loop.
const MAX_DOWNLOAD_RETRIES: u32 = 10;

/// Shared download-retry skeleton for the TTS model set: owns the
/// [`ModelLoadGuard`] and dir resolution, bounded loop (Ready pre-check, retry
/// cap, download-only timeout, Failed re-check, 5s→2min backoff). `on_retry_cap`
/// owns `state.store(Failed)` (module-specific tail ordering); dir-resolution
/// failure stores Failed here without a hook. The `load` closure must leave the
/// state terminal (store Ready) on success, or the guard's Loading→Failed drop
/// fires on return.
async fn run_download_retry_loop<D, L, F>(
    state: &AtomicModelState,
    dir_name: &str,
    label: &str,
    timeout: Duration,
    download: D,
    load: L,
    on_retry_cap: F,
) where
    D: AsyncFn(&Path) -> anyhow::Result<()>,
    L: Fn(&Path) -> anyhow::Result<()>,
    F: FnOnce(),
{
    // Transitions Loading→Failed on drop if the loop is cancelled or panics.
    let _guard = ModelLoadGuard::new(state);
    let Some(dir) = models_subdir(dir_name) else {
        warn!("{label}: cannot resolve model directory");
        state.store(ModelState::Failed, Ordering::Release);
        return;
    };

    let mut retry_delay = Duration::from_secs(5);
    let mut retry_count = 0u32;

    loop {
        if state.load(Ordering::Acquire) == ModelState::Ready {
            return;
        }
        retry_count += 1;
        if retry_count > MAX_DOWNLOAD_RETRIES {
            on_retry_cap();
            return;
        }

        match tokio::time::timeout(timeout, download(&dir)).await {
            Ok(Ok(())) => match load(&dir) {
                Ok(()) => return,
                Err(e) => warn!("Failed to load {label} models (will retry): {e}"),
            },
            Ok(Err(e)) => warn!("Failed to download {label} models (will retry): {e}"),
            Err(_) => warn!("{label} download timed out (will retry)"),
        }

        if state.load(Ordering::Acquire) == ModelState::Failed {
            return;
        }
        tokio::time::sleep(retry_delay).await;
        retry_delay = (retry_delay * 2).min(Duration::from_mins(2));
    }
}

/// Verify-then-download helper shared by the TTS model set.
///
/// Checks `path` before downloading (SHA256 when `sha256` is non-empty,
/// else `min_size`), re-downloading via
/// [`download_verified`](crate::util::http::download_verified) when
/// missing/corrupt/too small; returns `true` iff a download ran. `client`
/// is reused when `Some`; when `None` one is built (with `timeout`) only on
/// the download path. `timeout` maps to a per-request timeout. `label` names
/// the file in logs (must include the file name); `on_progress` runs during
/// downloads only.
///
/// Deliberately not used by `encoder.rs` (no-timeout client,
/// [`DownloadSizeCheck::None`](crate::util::http::DownloadSizeCheck::None),
/// spawn_blocking pre-check).
#[expect(clippy::cast_precision_loss, clippy::too_many_arguments)]
async fn ensure_downloaded(
    client: Option<&reqwest::Client>,
    path: &Path,
    url: &str,
    sha256: &str,
    min_size: u64,
    timeout: Duration,
    label: &str,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<bool> {
    if path.exists() {
        if sha256.is_empty() {
            let meta = tokio::fs::metadata(path).await?;
            if meta.len() >= min_size {
                return Ok(false);
            }
            warn!(
                "{label} too small ({} bytes), re-downloading: {}",
                meta.len(),
                path.display()
            );
        } else if let Err(e) = crate::util::verify_sha256(path, sha256) {
            warn!("{label} corrupt, re-downloading {}: {e}", path.display());
        } else {
            return Ok(false);
        }
        tokio::fs::remove_file(path).await?;
    }

    info!("Downloading {label}...");
    let client = match client {
        Some(c) => Cow::Borrowed(c),
        None => Cow::Owned(
            crate::util::http::build_download_client(timeout)
                .context("Failed to build HTTP client")?,
        ),
    };
    // Byte count comes from the progress closure (download_verified's
    // documented cumulative-bytes contract) — no post-success failure surface.
    let mut size = 0u64;
    crate::util::http::download_verified(
        &client,
        url,
        path,
        sha256,
        Some(timeout),
        crate::util::http::DownloadSizeCheck::Min(min_size),
        |d, total| {
            size = d;
            on_progress(d, total);
        },
    )
    .await?;
    info!("Downloaded {label} ({:.1} MB)", size as f64 / 1_048_576.0);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_ending_punctuation() {
        let cases: &[(&str, &str, bool)] = &[
            ("period", "Yes.", true),
            ("exclamation", "No!", true),
            ("question", "Maybe?", true),
            // Commas, semicolons and colons are NOT terminal punctuation for TTS.
            ("comma", "comma,", false),
            ("semicolon", "semicolon;", false),
            ("colon", "colon:", false),
            // Closing brackets, quotes, guillemets and CJK closing marks do NOT end a sentence.
            ("paren", "paren)", false),
            ("bracket", "bracket]", false),
            ("brace", "brace}", false),
            ("guillemet", "guillemet»", false),
            ("single_guillemet", "single»", false),
            ("cjk_right_corner_bracket", "cjk」", false),
            ("cjk_right_white_corner_bracket", "cjk』", false),
            ("cjk_right_black_square_bracket", "cjk】", false),
            ("cjk_right_angle_bracket", "cjk〉", false),
            ("cjk_right_double_angle", "cjk》", false),
            ("single_quote", "single'", false),
            ("double_quote", "double\"", false),
            ("no_punctuation", "Maybe", false),
            ("cjk_fullwidth_period", "CJK\u{3002}", true),
            ("cjk_fullwidth_bang", "CJK\u{FF01}", true),
            ("cjk_fullwidth_question", "CJK\u{FF1F}", true),
        ];
        for &(name, input, expected) in cases {
            assert_eq!(has_ending_punctuation(input), expected, "case: {name}");
        }
    }

    #[test]
    fn test_preprocess_text() {
        let r = preprocess_text("Hello world");
        assert_eq!(
            r, "Hello world.",
            "should normalize and add trailing period"
        );

        // Markdown stripped, emoji removed, symbols normalized, abbreviations expanded.
        // **Hi** is removed entirely — bold syntax carries no text content for TTS.
        let r2 = preprocess_text("**Hi** @user, e.g., hello 😊");
        assert_eq!(
            r2, "at user, for example, hello.",
            "should strip markdown, expand abbrevs, remove emoji, add period"
        );

        let r3 = preprocess_text("Hello, world.");
        assert_eq!(r3, "Hello, world.", "should not double-add period");
    }

    #[test]
    fn test_split_at_sentence_boundaries() {
        let cases: &[(&str, &str, usize, &[&str])] = &[
            ("single_char_sentences", "A. B. C.", 3, &["A.", "B.", "C."]),
            (
                "single_short_sentence",
                "Hello world.",
                100,
                &["Hello world."],
            ),
        ];
        for &(name, input, max_len, expected) in cases {
            let chunks = split_at_sentence_boundaries(input, max_len);
            let got: Vec<&str> = chunks.iter().map(String::as_str).collect();
            assert_eq!(got.as_slice(), expected, "case: {name}");
        }
    }

    #[test]
    fn test_render_wav_decodes_via_hound() {
        use rodio::Source;
        // The rodio 0.22 upgrade switched WAV decode from symphonia to hound (Apache-2.0
        // posture): verify it still decodes render_wav's PCM16 mono output with the right
        // rate, channel count and sample values.
        let sample_rate = 16000u32;
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let wav_bytes = render_wav(&samples, sample_rate).expect("render_wav should succeed");

        let cursor = Cursor::new(wav_bytes);
        let decoder = rodio::Decoder::new(cursor).expect("WAV should decode via hound");
        assert_eq!(
            decoder.sample_rate().get(),
            sample_rate,
            "sample rate mismatch"
        );
        assert_eq!(decoder.channels().get(), 1, "should be mono");
        let decoded_samples: Vec<f32> = decoder.collect();
        assert_eq!(
            decoded_samples.len(),
            samples.len(),
            "sample count mismatch"
        );
        for (got, expected) in decoded_samples.iter().zip(&samples) {
            assert!(
                (got - expected).abs() < 1e-3,
                "sample {got} != expected {expected}"
            );
        }
    }

    #[test]
    fn test_encode_text_with_indexer() {
        // Synthetic unicode indexer: ASCII 0-127 map to sequential IDs, UNK sits at index 0.
        // Space (32) is set to -1 so it falls through to UNK, as in a real indexer where space
        // is not a valid token.
        let mut indexer: Vec<i32> = (0..128i32).collect();
        indexer[0] = 0;
        indexer[32] = -1;

        let (ids, len) = encode_text_with_indexer(&indexer, "H w");
        assert_eq!(len, 3, "three characters encoded");
        assert_eq!(ids, vec![72, 0, 119], "space should map to UNK (ID 0)");

        let (ids2, len2) = encode_text_with_indexer(&[], "hello");
        assert_eq!(len2, 5, "five chars encoded with empty indexer");
        assert_eq!(ids2, vec![0; 5], "all map to UNK (default 0)");

        let (ids3, len3) = encode_text_with_indexer(&[42], "abc");
        assert_eq!(len3, 3);
        assert_eq!(ids3, vec![42; 3], "all chars map to UNK (first entry = 42)");

        let (ids4, len4) = encode_text_with_indexer(&[-1], "x");
        assert_eq!(len4, 1);
        assert_eq!(ids4, vec![-1i64], "UNK sentinel preserved");

        // The empty-token guard in [`synthesize_internal`] checks `seq_len == 0` after
        // [`encode_text`]: whitespace produces tokens, an empty string does not.
        let (ids, len) = encode_text_with_indexer(&vec![0i32; 128], "");
        assert_eq!(
            len, 0,
            "empty input must produce zero-length token sequence"
        );
        assert!(ids.is_empty(), "token IDs must be empty for empty input");
        let (_, len2) = encode_text_with_indexer(&vec![0i32; 128], "a");
        assert_eq!(len2, 1, "single-char input should produce one token");
        let (_, len3) = encode_text_with_indexer(&vec![0i32; 128], " ");
        assert_eq!(len3, 1, "whitespace input produces a token");
    }

    // say_phrase
    //
    // What the tool's one entry point answers while speech is off and while the model set is
    // not loaded. Neither case reaches the audio device, and neither starts a download.

    /// Run `say_phrase` under a known config and state, restoring both afterwards.
    fn say_under(config: &str, state: ModelState) -> SayOutcome {
        let snapshot = CONFIG.snapshot();
        let prev_state = STATE.load(Ordering::Acquire);
        assert!(
            CONFIG.set_string_field("tts_enabled", config),
            "tts_enabled is a known config key"
        );
        STATE.store(state, Ordering::Release);
        let outcome = say_phrase("A short phrase.");
        STATE.store(prev_state, Ordering::Release);
        CONFIG.swap(snapshot);
        outcome
    }

    #[test]
    #[serial_test::serial(tts)]
    fn say_refuses_while_speech_is_off() {
        for config in ["", "false"] {
            assert!(
                matches!(say_under(config, ModelState::Ready), SayOutcome::SpeechOff),
                "`tts_enabled = '{config}'` must be reported as speech output off"
            );
        }
    }

    /// Not-ready is an answer like the others: a set still loading is never waited for inside
    /// the call. Only the loading state is pinned here — a failed or unstarted set would kick
    /// a download, which a test must not do.
    #[test]
    #[serial_test::serial(tts)]
    fn say_reports_a_model_set_that_is_not_loaded() {
        assert!(
            matches!(
                say_under("true", ModelState::Loading),
                SayOutcome::NotReady(NotReady::ModelsMissing)
            ),
            "a loading model set must be reported as missing, not waited for"
        );
    }

    /// The counter's whole life: zero at rest, one increment per `speak_async()` call (not per
    /// chunk), one decrement after the reverb tail, and overlapping calls keeping it active
    /// until the last tail expires.
    #[test]
    #[serial_test::serial(tts)]
    fn test_playback_counter_lifecycle() {
        assert!(!is_playback_active(), "default state must be inactive");
        assert_eq!(
            PLAYBACK_ACTIVE.load(Ordering::Acquire),
            0,
            "internal counter must start at 0"
        );
        assert!(
            (200..=400).contains(&PLAYBACK_REVERB_TAIL_MS),
            "PLAYBACK_REVERB_TAIL_MS={PLAYBACK_REVERB_TAIL_MS} must be between 200 and 400",
        );

        // Single-chunk response: one increment when the chunk starts playing, one decrement
        // after the reverb tail.
        PLAYBACK_ACTIVE.fetch_add(1, Ordering::Relaxed);
        assert!(is_playback_active(), "active during playback");
        PLAYBACK_ACTIVE.fetch_sub(1, Ordering::Release);
        assert!(!is_playback_active(), "inactive after reverb tail");
        assert_eq!(PLAYBACK_ACTIVE.load(Ordering::Acquire), 0);

        // Multi-chunk: the increment happens ONCE per call, not once per chunk. Incrementing
        // per chunk leaked the counter on responses of two or more chunks.
        PLAYBACK_ACTIVE.fetch_add(1, Ordering::Relaxed);
        // Further chunks arrive with `current_sink.is_some()`: no fetch_add.
        assert_eq!(PLAYBACK_ACTIVE.load(Ordering::Acquire), 1);
        PLAYBACK_ACTIVE.fetch_sub(1, Ordering::Release);
        assert_eq!(
            PLAYBACK_ACTIVE.load(Ordering::Acquire),
            0,
            "counter must return to 0 after multi-chunk response"
        );
        assert!(!is_playback_active(), "inactive after all chunks done");

        // Overlapping calls: B starts during A's reverb tail, so A's expiry leaves it active.
        PLAYBACK_ACTIVE.fetch_add(1, Ordering::Relaxed);
        assert!(is_playback_active());
        PLAYBACK_ACTIVE.fetch_add(1, Ordering::Relaxed);
        assert!(is_playback_active());
        PLAYBACK_ACTIVE.fetch_sub(1, Ordering::Release);
        assert!(is_playback_active(), "B still playing");
        PLAYBACK_ACTIVE.fetch_sub(1, Ordering::Release);
        assert!(!is_playback_active());

        // 100 overlapping call pairs: the counter always returns to 0.
        for _ in 0..100 {
            assert_eq!(PLAYBACK_ACTIVE.load(Ordering::Acquire), 0);
            PLAYBACK_ACTIVE.fetch_add(1, Ordering::Relaxed);
            PLAYBACK_ACTIVE.fetch_add(1, Ordering::Relaxed);
            PLAYBACK_ACTIVE.fetch_sub(1, Ordering::Release);
            PLAYBACK_ACTIVE.fetch_sub(1, Ordering::Release);
            assert_eq!(
                PLAYBACK_ACTIVE.load(Ordering::Acquire),
                0,
                "counter must return to 0 after balanced add/sub pairs"
            );
        }
    }
}
