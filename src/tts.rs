//! Text-to-Speech using Supertone's Supertonic 3 model.
//!
//! # Overview
//!
//! This module provides automatic TTS synthesis for agent responses delivered
//! to the GUI dashboard. Synthesis runs **asynchronously** and **non-blocking**
//! — the message delivery completes immediately, and any audio playback
//! happens in the background.
//!
//! TTS is **not** a tool that agents call. It is background infrastructure,
//! exactly like the voice wake-word system, that automatically speaks agent
//! responses when:
//!
//! 1. The response is delivered to the GUI dashboard.
//! 2. The responding agent's role matches the user's currently-selected role.
//! 3. TTS is enabled in config.
//! 4. The Supertonic 3 model files are cached and loaded.
//!
//! # Pipeline (Supertonic 3)
//!
//! The synthesis pipeline has 4 ONNX model stages:
//!
//! 1. **Duration Predictor** – predicts token durations from token IDs.
//! 2. **Text Encoder** – encodes text features from token IDs.
//! 3. **Vector Estimator** – 8-step flow matching from noise to latent vector.
//! 4. **Vocoder** – converts latent vectors to PCM audio.
//!
//! # State machine
//!
//! The loading state uses [`AtomicU8`] with the same states as [`crate::embedder`]:
//!
//! | Value | Name     | Meaning                                      |
//! |-------|----------|----------------------------------------------|
//! | 0     | UNINIT   | TTS not loaded yet; [`init_global()`] must   |
//! |       |          | be called before the module can function.     |
//! | 1     | LOADING  | Downloading model files (retry loop).         |
//! | 2     | READY    | All models loaded and ready for synthesis.    |
//! | 3     | FAILED   | Download or load failed terminally.           |

use crate::config::CONFIG;
use crate::util::UnwrapPoison;
use anyhow::{Context, Result, anyhow};
use candle_core::{Device, Tensor};
use candle_onnx::simple_eval;
use futures_util::StreamExt;
use rodio::{OutputStream, OutputStreamHandle, Sink};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, LazyLock, OnceLock, RwLock};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::sync::{broadcast, mpsc};
use tracing::{info, warn};

// ── Constants ────────────────────────────────────────────────────────

const MODEL_DIR_NAME: &str = "supertonic3";
const MODEL_REPO: &str = "Supertone/supertonic-3";
// Pinned to the "Initial Supertonic 3 release" commit (verified on HuggingFace).
// This is the commit from which the SHA256 constants below were computed.
const MODEL_REVISION: &str = "724fb5abbf5502583fb520898d45929e62f02c0b";
const HF_BASE: &str = "https://huggingface.co";
const MODEL_DOWNLOAD_TIMEOUT: Duration = Duration::from_mins(10);
const MAX_DOWNLOAD_RETRIES: u32 = 10;
const DEFAULT_TOTAL_STEPS: usize = 8;
const SPEED_FACTOR: f32 = 1.05;
const MAX_CHUNK_LENGTH: usize = 300;
const SILENCE_DURATION: f32 = 0.3;
// Timeout per-chunk receive: guards against hung synthesis (ONNX deadlock).
const SYNTHESIS_CHUNK_TIMEOUT: Duration = Duration::from_mins(5);

// ── SHA256 integrity hashes ──────────────────────────────────────────
//
// Expected SHA256 hex digests for each model/config file.  When non-empty,
// the hash is verified on every access (both cache reuse and fresh download).
//
// To update: download a new version of the model files and compute their
// SHA256 digests, then replace the values below.

const DP_MODEL_SHA256: &str = "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db";
const TEXT_ENC_MODEL_SHA256: &str =
    "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff";
const VECTOR_EST_MODEL_SHA256: &str =
    "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c";
const VOCODER_MODEL_SHA256: &str =
    "085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba";
const TTS_JSON_SHA256: &str = "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09";
const UNICODE_INDEXER_SHA256: &str =
    "9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f";
const VOICE_STYLE_SHA256: &str = "e35604687f5d23694b8e91593a93eec0e4eca6c0b02bb8ed69139ab2ea6b0a5b";

const ONNX_DIR: &str = "onnx";
const VOICE_STYLES_DIR: &str = "voice_styles";
const DP_ONNX_NAME: &str = "duration_predictor.onnx";
const TEXT_ENC_ONNX_NAME: &str = "text_encoder.onnx";
const VECTOR_EST_ONNX_NAME: &str = "vector_estimator.onnx";
const VOCODER_ONNX_NAME: &str = "vocoder.onnx";
const TTS_JSON_NAME: &str = "tts.json";
const UNICODE_INDEXER_NAME: &str = "unicode_indexer.json";
const DEFAULT_VOICE_NAME: &str = "M1.json";

/// All 10 available voice styles from HuggingFace.
/// F1-F5 are female voices, M1-M5 are male voices.
const ALL_VOICE_STYLE_NAMES: &[&str] = &[
    "F1.json", "F2.json", "F3.json", "F4.json", "F5.json", "M1.json", "M2.json", "M3.json",
    "M4.json", "M5.json",
];

// ── State machine ─────────────────────────────────────────────────────

const STATE_UNINIT: u8 = 0;
const STATE_LOADING: u8 = 1;
const STATE_READY: u8 = 2;
const STATE_FAILED: u8 = 3;

static STATE: AtomicU8 = AtomicU8::new(STATE_UNINIT);
static GLOBAL_TTS: OnceLock<RwLock<Option<Arc<TtsEngine>>>> = OnceLock::new();
static CANCEL_TX: OnceLock<broadcast::Sender<()>> = OnceLock::new();
static CANCEL_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();

// ── Download progress events (GUI subscription) ───────────────────────

/// Events emitted during TTS model download for GUI progress reporting.
#[derive(Debug, Clone)]
pub enum TtsDownloadEvent {
    /// A file download has started.
    FileStarted { name: String, total_bytes: u64 },
    /// Download progress for a file.
    FileProgress {
        name: String,
        bytes_downloaded: u64,
        total_bytes: u64,
    },
    /// A file download has completed successfully.
    FileCompleted { name: String },
    /// All files have been downloaded and verified.
    Complete,
    /// Download failed with an error.
    Failed { error: String },
}

/// Broadcast channel for TTS download progress events (GUI subscription).
pub static DOWNLOAD_EVENTS: OnceLock<broadcast::Sender<TtsDownloadEvent>> = OnceLock::new();

/// Subscribe to TTS download progress events for the GUI subscription.
pub fn subscribe_download_events() -> broadcast::Receiver<TtsDownloadEvent> {
    DOWNLOAD_EVENTS
        .get()
        .expect("DOWNLOAD_EVENTS initialized before subscribe")
        .subscribe()
}

/// Wrapper around rodio's audio output that is `Send` on all platforms.
///
/// `rodio::OutputStream` is `!Send` on macOS because of a phantom
/// `NotSendSyncAcrossAllPlatforms` marker, but the underlying CoreAudio
/// handles are actually thread-safe. This uses `unsafe impl Send` to
/// assert thread-safety, following the same pattern as `SendMicStream`
/// in `voice.rs`.
struct AudioOutputWrapper {
    _stream: OutputStream,
    handle: OutputStreamHandle,
}

// SAFETY: `OutputStream` is conservatively `!Send` on macOS due to a
// `PhantomData<*mut ()>` marker. The underlying CoreAudio device handles
// are thread-safe for our usage: we create the stream once at startup and
// only use the `OutputStreamHandle` (which is `Send`) to create sinks from
// async tasks. The `OutputStream` itself is never moved after initialization.
// This is a well-known pattern in the cpal/rodio ecosystem.
unsafe impl Send for AudioOutputWrapper {}

// SAFETY: `OutputStream` is conservatively `!Sync` on macOS due to the same
// phantom marker as `!Send`. The underlying CoreAudio device handles are
// thread-safe for our usage — we only read the `OutputStreamHandle` (which
// is `Sync`) from multiple threads, and the `OutputStream` itself is never
// accessed after initialization. This extends the `Send` justification above.
unsafe impl Sync for AudioOutputWrapper {}

static AUDIO_OUTPUT: OnceLock<AudioOutputWrapper> = OnceLock::new();

/// Test-only counter of `speak()` calls. Incremented at the very start of
/// `speak()`, before the `is_enabled()` guard. Used by `test_init_listener_*`
/// tests to verify that the broadcast subscriber correctly dispatches to `speak()`.
#[cfg(test)]
pub(crate) static SPEAK_COUNT: AtomicU64 = AtomicU64::new(0);

// ── TTS JSON config ───────────────────────────────────────────────────

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

// ── TTS engine ───────────────────────────────────────────────────────

struct TtsEngine {
    dp_model: candle_onnx::onnx::ModelProto,
    text_enc_model: candle_onnx::onnx::ModelProto,
    vector_est_model: candle_onnx::onnx::ModelProto,
    vocoder_model: candle_onnx::onnx::ModelProto,
    unicode_indexer: Vec<i32>,
    sample_rate: u32,
    latent_dim: usize,
    chunk_compress_factor: usize,
    base_chunk_size: usize,
    device: Device,
}

/// A single style entry from the HuggingFace voice style JSON.
///
/// Actual format from HuggingFace repo:
/// ```json
/// {
///   "data": [[[ ... ]]],   // 3D array: [batch, rows, cols]
///   "dims": [1, 8, 16],    // shape description
///   "type": "float32"       // data type
/// }
/// ```
#[derive(Debug, Deserialize)]
struct StyleEntry {
    data: Vec<Vec<Vec<f32>>>,
    #[allow(dead_code)]
    dims: Vec<usize>,
    #[allow(dead_code)]
    #[serde(rename = "type")]
    data_type: String,
}

#[derive(Debug, Deserialize)]
struct VoiceStyleFile {
    #[serde(rename = "style_dp")]
    style_dp: StyleEntry,
    #[serde(rename = "style_ttl")]
    style_ttl: StyleEntry,
}

// ── Public API ───────────────────────────────────────────────────────

/// Returns `true` if TTS is enabled in config (regardless of model state).
/// Use this to avoid unnecessary download/loading when TTS is disabled.
///
/// TTS is opt-in (disabled by default) — the user must explicitly set
/// `tts_enabled` to `"true"` in config to activate it. This matches the
/// convention used by the voice assistant ([`crate::voice`]).
#[must_use]
pub fn is_config_enabled() -> bool {
    let enabled = CONFIG.tts_enabled();
    enabled.as_deref() == Some("true")
}

#[must_use]
pub fn is_enabled() -> bool {
    is_config_enabled() && STATE.load(Ordering::Acquire) == STATE_READY
}

#[must_use]
pub fn models_ready() -> bool {
    STATE.load(Ordering::Acquire) == STATE_READY
}

/// Returns `true` if model download has permanently failed
/// (retries exhausted or model directory unresolvable).
#[must_use]
pub fn download_failed() -> bool {
    STATE.load(Ordering::Acquire) == STATE_FAILED
}

/// Retry model download after a previous failure.
///
/// Atomically transitions [`STATE`] from [`STATE_FAILED`] → [`STATE_UNINIT`]
/// and calls [`spawn_download()`]. If the state is not [`STATE_FAILED`],
/// this is a no-op and returns `false`.
///
/// This is the GUI-facing counterpart of [`spawn_download()`] which only
/// transitions from [`STATE_UNINIT`] — the two functions together handle
/// the initial download and retry-after-failure paths.
#[must_use]
pub fn retry_download() -> bool {
    if STATE
        .compare_exchange(
            STATE_FAILED,
            STATE_UNINIT,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        return false;
    }
    spawn_download();
    true
}

/// Test-only: set TTS state to a known value for deterministic testing.
#[cfg(test)]
pub(crate) fn test_set_state(state: u8) {
    STATE.store(state, Ordering::Release);
}

/// Speak `text` with the default voice (M1).
///
/// Spawns a background task that synthesizes audio, plays it via rodio
/// (cross-platform audio playback), then returns. Silently ignored if TTS is
/// disabled, models aren't ready, audio output is unavailable, or text is empty.
///
/// **Note:** This does NOT trigger model initialization. Models must be loaded
/// beforehand by [`init_global()`] + [`try_load_cached()`] / [`spawn_download()`].
/// If models are not in `STATE_READY` this call is a silent no-op.
pub fn speak(text: &str) {
    #[cfg(test)]
    SPEAK_COUNT.fetch_add(1, Ordering::Release);

    if text.trim().is_empty() || !is_enabled() {
        return;
    }
    // Cancel any previous playback before starting new synthesis.
    // The subscription below happens BEFORE the spawn so that this cancel
    // signal reaches the old task before the new task subscribes.
    cancel_playback();

    // Subscribe to cancellation channel BEFORE spawning the task to avoid a
    // race where cancel_playback() between spawn and subscription is lost.
    let cancel_rx = CANCEL_TX.get().map(broadcast::Sender::subscribe);
    tokio::spawn(speak_async(text.to_string(), cancel_rx));
}

/// Cancel any currently-playing TTS audio.
///
/// Sends a cancellation signal to the playback task. If a synthesis/playback
/// cycle is in progress, the audio playback will be interrupted early.
///
/// **Note:** If ONNX synthesis is actively running inside a blocking thread,
/// cancellation may take up to a single chunk synthesis cycle (~3–5s) to
/// take effect, as the cancellation flag is only checked between chunks.
pub fn cancel_playback() {
    if let Some(tx) = CANCEL_TX.get() {
        let _ = tx.send(());
    }
    if let Some(flag) = CANCEL_FLAG.get() {
        flag.store(true, Ordering::Release);
    }
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

    // Initialize rodio audio output (best-effort: may fail on headless systems)
    match OutputStream::try_default() {
        Ok((stream, handle)) => {
            AUDIO_OUTPUT
                .set(AudioOutputWrapper {
                    _stream: stream,
                    handle,
                })
                .map_err(|_| anyhow!("AUDIO_OUTPUT already initialized"))?;
        }
        Err(e) => {
            warn!("TTS: failed to initialize audio output — playback will be disabled: {e}");
        }
    }

    Ok(())
}

/// Subscribe to [`CHAT_BROADCAST`](crate::CHAT_BROADCAST) and speak agent
/// responses aloud that match the TTS criteria.
///
/// This is the TTS trigger mechanism — it replaces what was previously an
/// ad-hoc conditional in [`broadcast_and_persist_agent_response`] with a
/// clean observer pattern: the TTS module subscribes to chat events and
/// decides for itself when to speak, rather than being invoked directly
/// from shared infrastructure.
///
/// The listener checks:
/// 1. The event is an agent message (`direction == Agent`)
/// 2. The message was delivered via the GUI dashboard (`channel == "gui"`)
/// 3. TTS is globally enabled and models are loaded
/// 4. The agent's role matches the user's currently-active GUI role
///
/// Must be called **after** [`crate::CHAT_BROADCAST`] has been initialized
/// (i.e. after [`init_message_pipeline`]).
pub fn init_listener() {
    let Some(tx) = crate::CHAT_BROADCAST.get() else {
        warn!("TTS: CHAT_BROADCAST not initialized — listener not started");
        return;
    };
    let mut rx = tx.subscribe();

    tokio::spawn(async move {
        loop {
            use crate::{ChatDirection, ChatEvent};

            match rx.recv().await {
                Ok(ChatEvent::Message {
                    direction: ChatDirection::Agent,
                    channel,
                    user_name,
                    agent_role: Some(ref role_name),
                    content,
                    ..
                }) if channel == "gui" && is_enabled() => {
                    let active_role = crate::users::resolve_active_role(&user_name).await;
                    if active_role.as_str() == role_name.as_str() {
                        speak(&content);
                    }
                }
                Ok(_) => {
                    // Not an agent GUI message — ignore
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!("TTS listener lagged by {n} messages");
                }
                Err(broadcast::error::RecvError::Closed) => {
                    info!("TTS listener: CHAT_BROADCAST closed, shutting down");
                    break;
                }
            }
        }
    });
}

/// List available voice style names (e.g., "M1.json", "F1.json", etc.).
///
/// Returns only the styles that are actually cached on disk. The full set
/// is downloaded during model initialization.
#[must_use]
pub fn list_voice_styles() -> Vec<String> {
    let Some(dir) = model_dir() else {
        return Vec::new();
    };
    let styles_dir = dir.join(VOICE_STYLES_DIR);
    let mut available = Vec::new();
    for &name in ALL_VOICE_STYLE_NAMES {
        if styles_dir.join(name).exists() {
            available.push(name.to_string());
        }
    }
    available
}

/// Synthesize `text` with the given `voice_style` and return raw mono PCM
/// samples at the specified sample rate (f32 in [-1.0, 1.0]).
///
/// This is a **pure synthesis** function — no playback, no rodio, no
/// cancellation channel. Use it for data generation, offline processing,
/// or any scenario where you need audio samples rather than immediate
/// playback.
///
/// # Arguments
///
/// * `text` — The text to synthesize. Preprocessing (markdown strip, emoji
///   removal, abbreviation expansion) is applied automatically.
/// * `voice_style` — One of the style file names (e.g. `"M1.json"`,
///   `"F1.json"`). Use [`list_voice_styles()`] to discover available styles.
/// * `seed` — Random seed for the flow-matching noise. Different seeds
///   produce different prosody/intonation while preserving same text content.
///   Use `42` for deterministic output matching the default playback behavior
///   (same as [`speak_async`]).
/// * `target_sample_rate` — Desired output sample rate in Hz. The Supertonic 3
///   model natively outputs 24 kHz; passing a different rate triggers
///   resampling. Common values: `44100` (CD quality), `16000` (voice pipeline),
///   `24000` (native, no resampling).
///
/// # Errors
///
/// Returns an error if the TTS engine is not ready, the voice style is not
/// found or cannot be parsed, or synthesis fails.
///
/// # Sample rate
///
/// The Supertonic 3 model natively outputs 24 kHz audio. The function
/// resamples to `target_sample_rate` for compatibility. For training data
/// generation targeting the voice pipeline (which expects 16 kHz), pass
/// `target_sample_rate = 16000` to avoid an intermediate 44.1 kHz step.
///
/// # Example
///
/// ```ignore
/// // Synthesize at voice pipeline rate (16 kHz)
/// let pcm = tts::synthesize("hello world", "M1.json", 42, 16000)?;
/// assert_eq!(pcm.len(), 16000 /* ≈1 second at 16kHz */);
/// ```
pub fn synthesize(
    text: &str,
    voice_style: &str,
    seed: u64,
    target_sample_rate: u32,
) -> Result<Vec<f32>> {
    let engine = get_engine_clone().context("TTS engine not ready")?;
    let dir = model_dir().context("Cannot resolve model directory")?;
    let (style_dp, style_ttl) = load_voice_style(&dir, voice_style)?;
    let processed = preprocess_text(text);
    if processed.is_empty() {
        anyhow::bail!("Empty text after preprocessing");
    }
    let native_rate = engine.sample_rate;
    let samples = synthesize_internal(&engine, &processed, &style_dp, &style_ttl, seed)?;
    // Resample from native rate to target rate
    if native_rate == target_sample_rate {
        Ok(samples)
    } else {
        Ok(crate::util::resample_audio(
            &samples,
            native_rate,
            target_sample_rate,
        ))
    }
}

/// Try to load TTS models from cache at startup.
/// Returns `true` if loaded, `false` if not (download will happen async).
pub fn try_load_cached() -> bool {
    if STATE.load(Ordering::Acquire) != STATE_UNINIT {
        return STATE.load(Ordering::Acquire) == STATE_READY;
    }

    let Some(dir) = model_dir() else {
        return false;
    };

    let mut paths: Vec<PathBuf> = vec![
        dir.join(ONNX_DIR).join(DP_ONNX_NAME),
        dir.join(ONNX_DIR).join(TEXT_ENC_ONNX_NAME),
        dir.join(ONNX_DIR).join(VECTOR_EST_ONNX_NAME),
        dir.join(ONNX_DIR).join(VOCODER_ONNX_NAME),
        dir.join(ONNX_DIR).join(TTS_JSON_NAME),
        dir.join(ONNX_DIR).join(UNICODE_INDEXER_NAME),
    ];
    // Also check that at least the default voice style exists
    paths.push(dir.join(VOICE_STYLES_DIR).join(DEFAULT_VOICE_NAME));

    let all_exist = paths.iter().all(|p| p.exists());

    if !all_exist {
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
    // CAS failure means another task already set LOADING or READY — avoid race.
    if STATE
        .compare_exchange(
            STATE_UNINIT,
            STATE_LOADING,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        return;
    }
    tokio::spawn(download_retry_loop());
}

/// Spawn model download, retrying after a previous failure if needed.
///
/// Unlike [`spawn_download()`] which only transitions from [`STATE_UNINIT`],
/// this also handles [`STATE_FAILED`] by resetting to [`STATE_UNINIT`] first,
/// making it suitable for the GUI toggle which may be activated after a
/// permanent download failure.
pub fn spawn_or_retry_download() {
    // Fast path: UNINIT → LOADING
    if STATE
        .compare_exchange(
            STATE_UNINIT,
            STATE_LOADING,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok()
    {
        tokio::spawn(download_retry_loop());
        return;
    }
    // Slow path: FAILED → UNINIT, then spawn_download handles UNINIT → LOADING
    if STATE
        .compare_exchange(
            STATE_FAILED,
            STATE_UNINIT,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok()
    {
        spawn_download();
    }
    // Otherwise already LOADING or READY — nothing to do.
}

// ── Internal helpers ─────────────────────────────────────────────────

fn model_dir() -> Option<PathBuf> {
    Some(
        CONFIG
            .try_storage_root()?
            .join("models")
            .join(MODEL_DIR_NAME),
    )
}

fn set_engine_ready(engine: TtsEngine) {
    if let Some(global) = GLOBAL_TTS.get() {
        *global.write().unwrap_poison() = Some(Arc::new(engine));
    }
    STATE.store(STATE_READY, Ordering::Release);
}

/// Clone the engine [`Arc`] cheaply, then drop the read lock.
/// Use this before [`tokio::task::spawn_blocking`] to avoid holding
/// the `RwLock` across a long CPU-bound operation.
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
    let dp_data = &voice.style_dp.data;
    anyhow::ensure!(!dp_data.is_empty(), "Voice style has empty style_dp");
    anyhow::ensure!(!dp_data[0].is_empty(), "Voice style has empty style_dp[0]");
    let dp_rows = dp_data[0].len();
    let dp_cols = dp_data[0][0].len();
    let dp_flat: Vec<f32> = dp_data[0].iter().flat_map(|v| v.iter()).copied().collect();
    anyhow::ensure!(
        dp_flat.len() == dp_rows * dp_cols,
        "Flat style_dp length {} doesn't match {}×{}",
        dp_flat.len(),
        dp_rows,
        dp_cols,
    );
    let style_dp = Tensor::from_slice(&dp_flat, (1, dp_rows, dp_cols), &device)?;

    // style_ttl: HuggingFace stores as 3D [batch=1, rows=50, cols=256]
    let ttl_data = &voice.style_ttl.data;
    anyhow::ensure!(!ttl_data.is_empty(), "Voice style has empty style_ttl");
    anyhow::ensure!(
        !ttl_data[0].is_empty(),
        "Voice style has empty style_ttl[0]"
    );
    let ttl_rows = ttl_data[0].len();
    let ttl_cols = ttl_data[0][0].len();
    let ttl_flat: Vec<f32> = ttl_data[0].iter().flat_map(|v| v.iter()).copied().collect();
    anyhow::ensure!(
        ttl_flat.len() == ttl_rows * ttl_cols,
        "Flat style_ttl length {} doesn't match {}×{}",
        ttl_flat.len(),
        ttl_rows,
        ttl_cols,
    );
    let style_ttl = Tensor::from_slice(&ttl_flat, (1, ttl_rows, ttl_cols), &device)?;

    Ok((style_dp, style_ttl))
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

    let dp_model = candle_onnx::read_file(onnx_dir.join(DP_ONNX_NAME))?;
    let text_enc_model = candle_onnx::read_file(onnx_dir.join(TEXT_ENC_ONNX_NAME))?;
    let vector_est_model = candle_onnx::read_file(onnx_dir.join(VECTOR_EST_ONNX_NAME))?;
    let vocoder_model = candle_onnx::read_file(onnx_dir.join(VOCODER_ONNX_NAME))?;

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

// ── Download retry loop ──────────────────────────────────────────────

/// Emit a download event through the global broadcast channel.
fn emit_download_event(event: TtsDownloadEvent) {
    if let Some(tx) = DOWNLOAD_EVENTS.get() {
        let _ = tx.send(event);
    }
}

struct TtsGuard;

impl Drop for TtsGuard {
    fn drop(&mut self) {
        STATE
            .compare_exchange(
                STATE_LOADING,
                STATE_FAILED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok();
    }
}

async fn download_retry_loop() {
    let _guard = TtsGuard;
    let Some(dir) = model_dir() else {
        warn!("TTS: cannot resolve model directory");
        STATE.store(STATE_FAILED, Ordering::Release);
        return;
    };

    let mut retry_delay = Duration::from_secs(5);
    let mut retry_count = 0u32;

    loop {
        if STATE.load(Ordering::Acquire) == STATE_READY {
            return;
        }
        retry_count += 1;
        if retry_count > MAX_DOWNLOAD_RETRIES {
            let msg = format!("TTS download failed after {MAX_DOWNLOAD_RETRIES} retries");
            warn!("{msg}");
            emit_download_event(TtsDownloadEvent::Failed { error: msg });
            STATE.store(STATE_FAILED, Ordering::Release);
            return;
        }

        match tokio::time::timeout(MODEL_DOWNLOAD_TIMEOUT, ensure_models_downloaded(&dir)).await {
            Ok(Ok(())) => match load_engine(&dir) {
                Ok(e) => {
                    set_engine_ready(e);
                    emit_download_event(TtsDownloadEvent::Complete);
                    info!("TTS models loaded successfully");
                    return;
                }
                Err(e) => warn!("Failed to load TTS models (will retry): {e}"),
            },
            Ok(Err(e)) => warn!("Failed to download TTS models (will retry): {e}"),
            Err(_) => warn!("TTS download timed out (will retry)"),
        }

        if STATE.load(Ordering::Acquire) == STATE_FAILED {
            return;
        }
        tokio::time::sleep(retry_delay).await;
        retry_delay = (retry_delay * 2).min(Duration::from_mins(2));
    }
}

/// File descriptor: (download URL, local path, expected SHA256 hash).
struct TtsFile {
    url: String,
    path: PathBuf,
    sha256: &'static str,
}

async fn ensure_models_downloaded(dir: &Path) -> Result<()> {
    tokio::fs::create_dir_all(dir.join(ONNX_DIR)).await?;
    tokio::fs::create_dir_all(dir.join(VOICE_STYLES_DIR)).await?;

    let base = format!("{HF_BASE}/{MODEL_REPO}/resolve/{MODEL_REVISION}");

    let mut files: Vec<TtsFile> = vec![
        TtsFile {
            url: format!("{base}/onnx/{DP_ONNX_NAME}"),
            path: dir.join(ONNX_DIR).join(DP_ONNX_NAME),
            sha256: DP_MODEL_SHA256,
        },
        TtsFile {
            url: format!("{base}/onnx/{TEXT_ENC_ONNX_NAME}"),
            path: dir.join(ONNX_DIR).join(TEXT_ENC_ONNX_NAME),
            sha256: TEXT_ENC_MODEL_SHA256,
        },
        TtsFile {
            url: format!("{base}/onnx/{VECTOR_EST_ONNX_NAME}"),
            path: dir.join(ONNX_DIR).join(VECTOR_EST_ONNX_NAME),
            sha256: VECTOR_EST_MODEL_SHA256,
        },
        TtsFile {
            url: format!("{base}/onnx/{VOCODER_ONNX_NAME}"),
            path: dir.join(ONNX_DIR).join(VOCODER_ONNX_NAME),
            sha256: VOCODER_MODEL_SHA256,
        },
        TtsFile {
            url: format!("{base}/onnx/{TTS_JSON_NAME}"),
            path: dir.join(ONNX_DIR).join(TTS_JSON_NAME),
            sha256: TTS_JSON_SHA256,
        },
        TtsFile {
            url: format!("{base}/onnx/{UNICODE_INDEXER_NAME}"),
            path: dir.join(ONNX_DIR).join(UNICODE_INDEXER_NAME),
            sha256: UNICODE_INDEXER_SHA256,
        },
    ];

    // Add all 10 voice style files. Only M1.json has a verified SHA256;
    // the rest have empty hashes (minimum-size check only) until they
    // are verified against downloads.
    for style_name in ALL_VOICE_STYLE_NAMES {
        let sha = if *style_name == DEFAULT_VOICE_NAME {
            VOICE_STYLE_SHA256
        } else {
            "" // No verified hash yet — minimum-size check only
        };
        files.push(TtsFile {
            url: format!("{base}/{VOICE_STYLES_DIR}/{style_name}"),
            path: dir.join(VOICE_STYLES_DIR).join(style_name),
            sha256: sha,
        });
    }

    // Sequential downloads so per-file progress is meaningful
    for f in &files {
        if let Err(e) = ensure_file(f).await {
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

/// Ensure a single file exists and is uncorrupted, or download it.
async fn ensure_file(f: &TtsFile) -> Result<()> {
    // Check cached file integrity
    if f.path.exists() {
        if f.sha256.is_empty() {
            // No hash configured — just check minimum size
            let meta = tokio::fs::metadata(&f.path).await?;
            if meta.len() > 100 {
                return Ok(());
            }
            // File too small, re-download
            warn!(
                "TTS file too small ({} bytes), re-downloading: {}",
                meta.len(),
                f.path.display()
            );
            tokio::fs::remove_file(&f.path).await?;
        } else {
            match verify_sha256(&f.path, f.sha256) {
                Ok(()) => return Ok(()), // file is intact
                Err(e) => {
                    warn!("TTS file corrupt, re-downloading {}: {e}", f.path.display());
                    tokio::fs::remove_file(&f.path).await?;
                }
            }
        }
    }

    // Download
    info!(
        "Downloading TTS file: {}",
        f.path.file_name().unwrap_or_default().to_string_lossy()
    );
    download_file(&f.url, &f.path, f.sha256).await
}

/// Download a single file with atomic write, SHA256 verification, and progress events.
#[allow(clippy::cast_precision_loss)]
async fn download_file(url: &str, dest: &Path, expected_hash: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(MODEL_DOWNLOAD_TIMEOUT)
        .connect_timeout(Duration::from_secs(30))
        .build()
        .context("Failed to build HTTP client")?;

    let response = client
        .get(url)
        .send()
        .await
        .context("Failed to start download")?;

    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("HTTP {status} from {url}");
    }

    let total_size = response.content_length().unwrap_or(0);
    let file_name = dest
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    // Emit FileStarted with the file name and total size (if known)
    emit_download_event(TtsDownloadEvent::FileStarted {
        name: file_name.clone(),
        total_bytes: total_size,
    });

    // Stream download to a temporary file, computing SHA256 on the fly
    let tmp = dest.with_extension("tmp");
    let mut file = tokio::fs::File::create(&tmp)
        .await
        .context("Failed to create temp file")?;

    let compute_hash = !expected_hash.is_empty();
    let mut hasher = compute_hash.then(Sha256::new);
    let mut downloaded: u64 = 0;
    let mut last_reported_bytes: u64 = 0;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Download stream error")?;
        let len = chunk.len() as u64;
        downloaded += len;
        if let Some(ref mut h) = hasher {
            h.update(&chunk);
        }
        file.write_all(&chunk)
            .await
            .context("Failed to write download chunk")?;

        // Throttle: emit progress at ~1% granularity to avoid broadcast pressure
        if total_size > 0 {
            let threshold = (total_size / 100).max(1);
            if downloaded - last_reported_bytes >= threshold || downloaded >= total_size {
                last_reported_bytes = downloaded;
                emit_download_event(TtsDownloadEvent::FileProgress {
                    name: file_name.clone(),
                    bytes_downloaded: downloaded,
                    total_bytes: total_size,
                });
            }
        }
    }

    file.flush().await?;
    drop(file);

    if downloaded < 100 {
        let _ = tokio::fs::remove_file(&tmp).await;
        anyhow::bail!("Downloaded file too small: {downloaded} bytes");
    }

    // Verify hash BEFORE renaming to final path
    if let Some(h) = hasher {
        let actual_hash = format!("{:x}", h.finalize());
        if actual_hash != expected_hash {
            let _ = tokio::fs::remove_file(&tmp).await;
            anyhow::bail!(
                "SHA256 verification failed for {}: expected {expected_hash}, got {actual_hash}",
                dest.display()
            );
        }
    }

    tokio::fs::rename(&tmp, dest)
        .await
        .with_context(|| format!("Failed to rename temp file to {}", dest.display()))?;

    let hash_str = if expected_hash.is_empty() {
        String::new()
    } else {
        format!(" (SHA256: {expected_hash})")
    };

    info!(
        "Downloaded {}{} ({:.1} MB)",
        file_name,
        hash_str,
        downloaded as f64 / 1_048_576.0
    );

    emit_download_event(TtsDownloadEvent::FileCompleted { name: file_name });
    Ok(())
}

/// Verify a file's SHA256 hash matches the expected hex string.
/// If `expected` is empty, verification is skipped (returns Ok).
fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    if expected.is_empty() {
        return Ok(()); // no hash configured — skip verification
    }

    let mut hasher = Sha256::new();
    let mut file = File::open(path)
        .with_context(|| format!("Failed to open {} for SHA256 verification", path.display()))?;
    let mut buf = vec![0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual = hex_string(&hasher.finalize());
    if actual != expected {
        anyhow::bail!(
            "SHA256 mismatch for {}: expected {expected}, got {actual}",
            path.display()
        );
    }
    Ok(())
}

fn hex_string(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut acc, b| {
            use std::fmt::Write;
            let _ = write!(acc, "{b:02x}");
            acc
        })
}

// ── Text preprocessing ───────────────────────────────────────────────

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

    let mut s: String = text.nfkd().collect();
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

    // Remove code blocks (triple backtick or indented)
    s = RE_CODE_BLOCK.replace_all(&s, " ").to_string();

    // Remove inline code
    s = RE_INLINE_CODE.replace_all(&s, " ").to_string();

    // Remove image markup: ![alt](url)
    s = RE_IMAGE.replace_all(&s, " ").to_string();

    // Replace links with their text: [text](url) → text
    s = RE_LINK.replace_all(&s, "$1").to_string();

    // Remove bold/italic markers: **text** or *text*
    s = RE_BOLD_ITALIC.replace_all(&s, " ").to_string();

    // Remove strikethrough: ~~text~~
    s = RE_STRIKETHROUGH.replace_all(&s, " ").to_string();

    // Remove headers markers: # text, ## text, etc.
    s = RE_HEADER.replace_all(&s, "").to_string();

    // Remove list markers: - text, * text, 1. text
    s = RE_LIST_DASH.replace_all(&s, "").to_string();
    s = RE_LIST_NUM.replace_all(&s, "").to_string();

    // Remove horizontal rules (multiline: ^/$ match line boundaries)
    s = RE_HR.replace_all(&s, " ").to_string();

    // Clean up excessive whitespace left by replacements
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

// ── Character encoding ───────────────────────────────────────────────

/// Encode a text string into token IDs using the engine's unicode indexer.
fn encode_text(engine: &TtsEngine, text: &str) -> (Vec<i64>, usize) {
    encode_text_with_indexer(&engine.unicode_indexer, text)
}

/// Core encoding logic that works with any unicode indexer slice.
///
/// This is extracted for testability — see `test_encode_text_with_indexer`.
fn encode_text_with_indexer(unicode_indexer: &[i32], text: &str) -> (Vec<i64>, usize) {
    let unk_id = unicode_indexer.first().copied().unwrap_or(0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
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

// ── ONNX helpers ─────────────────────────────────────────────────────

fn first_output_name(model: &candle_onnx::onnx::ModelProto) -> String {
    model
        .graph
        .as_ref()
        .and_then(|g| g.output.first())
        .map_or_else(|| "output".to_string(), |o| o.name.clone())
}

fn build_inputs(inputs: Vec<(&str, Tensor)>) -> HashMap<String, Tensor> {
    let mut map = HashMap::new();
    for (name, tensor) in inputs {
        map.insert(name.to_string(), tensor);
    }
    map
}

fn extract_output(
    mut outputs: HashMap<String, Tensor>,
    model: &candle_onnx::onnx::ModelProto,
    label: &str,
) -> Result<Tensor> {
    let name = first_output_name(model);
    outputs
        .remove(&name)
        .ok_or_else(|| anyhow!("{label}: output '{name}' not found"))
}

// ── Synthesis pipeline ───────────────────────────────────────────────

/// Synthesize audio for a single text chunk. Returns PCM f32 samples.
#[allow(
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

    // 1. Encode text
    let (token_ids, seq_len) = encode_text(engine, text);
    if seq_len == 0 {
        anyhow::bail!("Empty token sequence");
    }

    // 2. Duration predictor inputs
    let text_ids = Tensor::from_slice(&token_ids, (1, seq_len), dev)?;
    let text_mask = Tensor::from_slice(&vec![1.0f32; seq_len], (1, 1, seq_len), dev)?;
    let text_mask_flat = Tensor::from_slice(&vec![1.0f32; seq_len], (1, seq_len), dev)?;

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

    // 3. Text encoder
    let te_out = simple_eval(
        &engine.text_enc_model,
        build_inputs(vec![
            ("text_ids", text_ids),
            ("style_ttl", style_ttl.clone()),
            ("text_mask", text_mask_flat.clone()),
        ]),
    )
    .context("Text encoder failed")?;
    let text_emb = extract_output(te_out, &engine.text_enc_model, "te")?;

    // 4. Latent dimensions
    let wav_len = (total_dur * engine.sample_rate as f32).ceil() as usize;
    let chunk_size = engine.base_chunk_size * engine.chunk_compress_factor;
    let latent_len = wav_len.div_ceil(chunk_size);
    let latent_dim = engine.latent_dim * engine.chunk_compress_factor;

    // 5. Sample noise
    let mut rng = seed;
    let noise: Vec<f32> = (0..latent_dim * latent_len)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng as f32) / (u64::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    let mut xt = Tensor::from_slice(&noise, (1, latent_dim, latent_len), dev)?;

    // Latent mask
    let latent_mask = Tensor::from_slice(&vec![1.0f32; latent_len], (1, 1, latent_len), dev)?;

    // 6. Flow matching
    let total_steps = Tensor::new(DEFAULT_TOTAL_STEPS as f32, dev)?;
    for step in 0..DEFAULT_TOTAL_STEPS {
        let step_f = Tensor::new(step as f32, dev)?;
        let ve_out = simple_eval(
            &engine.vector_est_model,
            build_inputs(vec![
                ("noisy_latent", xt.clone()),
                ("text_emb", text_emb.clone()),
                ("style_ttl", style_ttl.clone()),
                ("latent_mask", latent_mask.clone()),
                ("text_mask", text_mask_flat.clone()),
                ("current_step", step_f),
                ("total_step", total_steps.clone()),
            ]),
        )
        .with_context(|| format!("Vector estimator step {step} failed"))?;
        let velocity = extract_output(ve_out, &engine.vector_est_model, "ve")?;
        let dt = 1.0 / DEFAULT_TOTAL_STEPS as f64;
        xt = xt.broadcast_add(&(velocity * dt)?)?;
    }

    // 7. Vocoder
    let voc_out = simple_eval(&engine.vocoder_model, build_inputs(vec![("latent", xt)]))
        .context("Vocoder failed")?;
    let wav_t = extract_output(voc_out, &engine.vocoder_model, "voc")?;

    // Squeeze batch dim
    let wav_data = if wav_t.dims().len() >= 2 {
        wav_t.squeeze(0)?.to_vec1::<f32>()?
    } else {
        wav_t.to_vec1::<f32>()?
    };
    Ok(wav_data)
}

/// Synthesize with chunking for long texts.
/// Language tags are applied per-chunk (model requires `<en>...</en>` wrapping).
///
/// Collects all chunk samples into a single `Vec<f32>` (legacy non-streaming API).
/// Use [`synthesize_chunked_streaming`] for streaming playback.
#[cfg_attr(not(test), allow(dead_code))]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn synthesize_chunked(
    engine: &TtsEngine,
    text: &str,
    style_dp: &Tensor,
    style_ttl: &Tensor,
    cancel: Option<&AtomicBool>,
    seed: u64,
) -> Result<Vec<f32>> {
    // Check cancellation before starting any work
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        anyhow::bail!("Synthesis cancelled");
    }

    if text.len() <= MAX_CHUNK_LENGTH {
        return synthesize_internal(engine, &wrap_lang_tag(text), style_dp, style_ttl, seed);
    }
    let silence_samples = (SILENCE_DURATION * engine.sample_rate as f32) as usize;
    let mut result = Vec::new();
    let silence = vec![0.0f32; silence_samples];

    for chunk in split_at_sentence_boundaries(text, MAX_CHUNK_LENGTH) {
        // Check cancellation between chunks
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            anyhow::bail!("Synthesis cancelled");
        }
        match synthesize_internal(engine, &wrap_lang_tag(&chunk), style_dp, style_ttl, seed) {
            Ok(samples) => {
                result.extend(samples);
                result.extend_from_slice(&silence);
            }
            Err(e) => {
                warn!("TTS: chunk synthesis failed, skipping chunk: {e}");
            }
        }
    }
    Ok(result)
}

/// Synthesize with chunking for long texts, streaming each chunk through
/// a bounded channel as it becomes available.
///
/// Language tags are applied per-chunk (model requires `<en>...</en>` wrapping).
/// Each chunk's PCM samples are sent via the provided sender. The function
/// returns `Ok(())` when all chunks have been sent, or an error on cancellation
/// or if the receiver was dropped.
///
/// # Error propagation (intentional asymmetry)
///
/// There are two code paths with different error-handling strategies:
///
/// **Short-text path** (`text.len() <= MAX_CHUNK_LENGTH`):
/// Propagates synthesis errors via `?` because there is only one chunk — a
/// failure means no audio at all, so an error is the correct response.
///
/// **Long-text path** (multiple chunks):
/// Logs individual chunk failures via `warn!` and continues to the next chunk.
/// This is intentional: a single problematic sentence should not abort the
/// entire response. The user hears partial audio (all succeeding chunks)
/// rather than silence from an aborted synthesis. Only fatal errors
/// (cancellation, receiver drop) are propagated.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn synthesize_chunked_streaming(
    engine: &TtsEngine,
    text: &str,
    style_dp: &Tensor,
    style_ttl: &Tensor,
    cancel: Option<&AtomicBool>,
    tx: &mpsc::Sender<Vec<f32>>,
    seed: u64,
) -> Result<()> {
    // Check cancellation before starting any work
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        anyhow::bail!("Synthesis cancelled");
    }

    if text.len() <= MAX_CHUNK_LENGTH {
        let samples = synthesize_internal(engine, &wrap_lang_tag(text), style_dp, style_ttl, seed)?;
        tx.blocking_send(samples)
            .map_err(|_| anyhow!("Synthesis cancelled (receiver dropped)"))?;
        return Ok(());
    }

    for chunk in split_at_sentence_boundaries(text, MAX_CHUNK_LENGTH) {
        // Check cancellation between chunks
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            anyhow::bail!("Synthesis cancelled");
        }
        match synthesize_internal(engine, &wrap_lang_tag(&chunk), style_dp, style_ttl, seed) {
            Ok(samples) => {
                if tx.blocking_send(samples).is_err() {
                    anyhow::bail!("Synthesis cancelled (receiver dropped)");
                }
            }
            Err(e) => {
                warn!("TTS: chunk synthesis failed, skipping chunk: {e}");
            }
        }
    }
    Ok(())
}

/// Wrap text in a language tag for the Supertonic 3 model.
///
/// Reads the language tag from `CONFIG.tts_language()` (defaults to `"na"` —
/// the model's language-agnostic fallback). Users can set `tts_language` in
/// config to any supported code: en, ko, ja, ar, bg, cs, da, de, el, es, et,
/// fi, fr, hi, hr, hu, id, it, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv, tr,
/// uk, vi, na.
fn wrap_lang_tag(text: &str) -> String {
    let lang = CONFIG.tts_language();
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

// ── WAV generation ───────────────────────────────────────────────────

/// Render PCM float samples to an in-memory WAV file (16-bit mono PCM).
///
/// Returns the complete WAV file bytes, including RIFF header and sample data.
/// This replaces the old `write_wav` which wrote to a temp file — rodio can
/// play directly from a `Cursor<Vec<u8>>`, eliminating ephemeral file I/O.
fn render_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
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
        #[allow(clippy::cast_possible_truncation)]
        let sample_i16 = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&sample_i16.to_le_bytes());
    }

    Ok(buf)
}

// ── Async speak ──────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
async fn speak_async(text: String, cancel_rx: Option<broadcast::Receiver<()>>) {
    let mut cancel_rx = cancel_rx;

    // Check cancellation before starting expensive work
    if let Some(ref mut rx) = cancel_rx
        && rx.try_recv().is_ok()
    {
        info!("TTS synthesis cancelled before start");
        return;
    }

    // Process text (lightweight, no blocking I/O)
    let processed = preprocess_text(&text);

    // Clone the engine Arc outside the blocking task so the RwLock is
    // not held across CPU-bound synthesis (which can take seconds).
    let Some(engine) = get_engine_clone() else {
        return;
    };
    let sample_rate = engine.sample_rate;

    // Load the default voice style (M1) for playback.
    let Some(dir) = model_dir() else {
        return;
    };
    let (style_dp, style_ttl) = match load_voice_style(&dir, DEFAULT_VOICE_NAME) {
        Ok(styles) => styles,
        Err(e) => {
            warn!("TTS: failed to load default voice style: {e}");
            return;
        }
    };

    // Check audio output availability BEFORE starting the expensive
    // CPU-bound synthesis. On headless systems this avoids wasting
    // ~3-5s of synthesis work per chunk before the channel close is
    // detected.
    let Some(audio_output) = AUDIO_OUTPUT.get() else {
        return;
    };

    // Create bounded channel for streaming chunks.
    // Capacity 4 provides natural backpressure — if synthesis outpaces
    // playback, the synthesizer blocks on send after 4 queued chunks.
    let (tx, mut rx) = mpsc::channel::<Vec<f32>>(4);

    // Reset cancellation flag for this synthesis, then clone the Arc
    // so it can be passed into the spawn_blocking closure.
    let cancel_flag = CANCEL_FLAG.get().map(|f| {
        f.store(false, Ordering::Release);
        Arc::clone(f)
    });

    // CPU-bound synthesis: run on blocking threadpool, streaming each
    // chunk through the channel as it becomes available.
    let synthesize_handle = tokio::task::spawn_blocking(move || {
        synthesize_chunked_streaming(
            &engine,
            &processed,
            &style_dp,
            &style_ttl,
            cancel_flag.as_deref(),
            &tx,
            42, // default seed for playback
        )
    });

    // Spawn a lightweight logging task that observes the synthesis result.
    // This restores error observability lost when we stopped awaiting the
    // blocking task directly — synthesis errors and panics are now logged.
    tokio::spawn(async move {
        match synthesize_handle.await {
            Ok(Ok(())) => {} // Success — normal completion
            Ok(Err(e)) => warn!("TTS synthesis failed: {e}"),
            Err(e) => warn!("TTS synthesis task panicked: {e}"),
        }
    });

    // Receive chunks and play them as they arrive.
    let mut current_sink: Option<Sink> = None;

    // Stream chunks with a per-chunk timeout to guard against hung
    // synthesis. On timeout the loop breaks, which drops the receiver
    // and causes the blocking sender to unblock with a channel error.
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
            if let Some(ref sink) = current_sink {
                sink.stop();
                // Wait briefly for the audio thread to finish flushing
                // so we don't leave a truncated burst in the output buffer.
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            break;
        }

        // Wait for previous chunk to finish playing, then add inter-chunk
        // silence between consecutive chunks.
        if let Some(ref sink) = current_sink {
            let completed = wait_for_sink(sink, cancel_rx.as_mut()).await;
            if !completed {
                // Cancelled while waiting — sink.stop() was already called
                // inside wait_for_sink. Flush output buffer and stop.
                tokio::time::sleep(Duration::from_millis(20)).await;
                break;
            }
            // Natural pause between speech chunks
            tokio::time::sleep(Duration::from_secs_f32(SILENCE_DURATION)).await;
        }

        // Render WAV bytes for this chunk (in-memory, no file I/O).
        let wav_bytes = match render_wav(&chunk_samples, sample_rate) {
            Ok(bytes) => bytes,
            Err(e) => {
                warn!("TTS: failed to render WAV chunk: {e}");
                continue;
            }
        };

        // Decode the WAV bytes as an in-memory source
        let cursor = Cursor::new(wav_bytes);
        let source = match rodio::Decoder::new(cursor) {
            Ok(s) => s,
            Err(e) => {
                warn!("TTS: failed to decode WAV chunk: {e}");
                continue;
            }
        };

        // Create a fresh sink for this chunk and begin playback immediately.
        // Each chunk gets its own sink so we can cancel per-chunk playback.
        let sink = match Sink::try_new(&audio_output.handle) {
            Ok(s) => s,
            Err(e) => {
                warn!("TTS: failed to create audio sink for chunk: {e}");
                continue;
            }
        };

        sink.append(source);
        current_sink = Some(sink);
    }

    // Wait for the last chunk to finish playing
    if let Some(ref sink) = current_sink {
        let _ = wait_for_sink(sink, cancel_rx.as_mut()).await;
    }
}

/// Wait for a rodio sink to finish playback, polling for completion
/// with optional cancellation support.
///
/// Returns `true` if playback completed normally, `false` if cancelled.
async fn wait_for_sink(sink: &Sink, cancel_rx: Option<&mut broadcast::Receiver<()>>) -> bool {
    if let Some(rx) = cancel_rx {
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;

            if sink.empty() {
                return true;
            }

            if rx.try_recv().is_ok() {
                info!("TTS playback cancelled");
                sink.stop();
                return false;
            }
        }
    } else {
        // No cancellation receiver — just wait until playback finishes.
        while !sink.empty() {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        true
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_ending_punctuation() {
        // Sentence-ending punctuation detected
        assert!(has_ending_punctuation("Yes."));
        assert!(has_ending_punctuation("No!"));
        assert!(has_ending_punctuation("Maybe?"));

        // Commas, semicolons, colons are NOT terminal punctuation for TTS
        // (they don't indicate the end of a sentence)
        assert!(!has_ending_punctuation("comma,"));
        assert!(!has_ending_punctuation("semicolon;"));
        assert!(!has_ending_punctuation("colon:"));

        // Closing brackets, quotes, guillemets, and CJK closing marks
        // are NOT terminal punctuation for TTS (they don't end a sentence)
        assert!(!has_ending_punctuation("paren)"));
        assert!(!has_ending_punctuation("bracket]"));
        assert!(!has_ending_punctuation("brace}"));
        assert!(!has_ending_punctuation("guillemet»"));
        assert!(!has_ending_punctuation("single»"));
        assert!(!has_ending_punctuation("cjk」"));
        assert!(!has_ending_punctuation("cjk』"));
        assert!(!has_ending_punctuation("cjk】"));
        assert!(!has_ending_punctuation("cjk〉"));
        assert!(!has_ending_punctuation("cjk》"));

        // Quotes are NOT terminal punctuation
        assert!(!has_ending_punctuation("single'"));
        assert!(!has_ending_punctuation("double\""));

        // No punctuation
        assert!(!has_ending_punctuation("Maybe"));

        // Fullwidth variants should also be terminal
        assert!(has_ending_punctuation("CJK\u{3002}"));
        assert!(has_ending_punctuation("CJK\u{FF01}"));
        assert!(has_ending_punctuation("CJK\u{FF1F}"));
    }

    #[test]
    fn test_preprocess_text() {
        // Basic case: trailing period added
        let r = preprocess_text("Hello world");
        assert_eq!(
            r, "Hello world.",
            "should normalize and add trailing period"
        );

        // Markdown stripped, emoji removed, symbols normalized, abbreviations expanded.
        // **Hi** is completely removed by bold stripping (correct — bold syntax carries no
        // semantic text content for TTS purposes).
        let r2 = preprocess_text("**Hi** @user, e.g., hello 😊");
        assert_eq!(
            r2, "at user, for example, hello.",
            "should strip markdown, expand abbrevs, remove emoji, add period"
        );

        // Already well-formed: no change
        let r3 = preprocess_text("Hello, world.");
        assert_eq!(r3, "Hello, world.", "should not double-add period");
    }

    #[test]
    fn test_split_at_sentence_boundaries() {
        let chunks = split_at_sentence_boundaries("A. B. C.", 3);
        assert_eq!(
            chunks.len(),
            3,
            "each single-char sentence should be its own chunk"
        );
        assert_eq!(chunks[0], "A.", "first chunk should be 'A.'");
        assert_eq!(chunks[1], "B.", "second chunk should be 'B.'");
        assert_eq!(chunks[2], "C.", "third chunk should be 'C.'");

        // Single short sentence: no splitting
        let single = split_at_sentence_boundaries("Hello world.", 100);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0], "Hello world.");
    }

    #[test]
    fn test_strip_markdown() {
        // bold removal leaves space placeholder + original space = double space
        assert_eq!(strip_markdown("**bold** text"), "  text");
        assert_eq!(strip_markdown("`inline code` here"), "  here");
        assert_eq!(strip_markdown("```\nblock\n```\nend"), " \nend");
        assert_eq!(strip_markdown("[link](url) text"), "link text");
        assert_eq!(strip_markdown("![img](url) cap"), "  cap");
        assert_eq!(strip_markdown("~~strike~~"), " ");
    }

    #[test]
    fn test_verify_sha256() {
        let tmp = std::env::temp_dir().join("test_tts_sha256.txt");
        let content = b"hello tts test content";
        std::fs::write(&tmp, content).unwrap();

        let mut hasher = Sha256::new();
        hasher.update(content);
        let correct_hash = hex_string(&hasher.finalize());

        // Matching hash passes
        assert!(verify_sha256(&tmp, &correct_hash).is_ok());

        // Non-matching hash fails
        assert!(
            verify_sha256(
                &tmp,
                "0000000000000000000000000000000000000000000000000000000000000000"
            )
            .is_err()
        );

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_render_wav() {
        let sample_rate = 44100u32;
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0, 0.0];

        let wav_bytes = render_wav(&samples, sample_rate).expect("render_wav should succeed");

        // Should be a valid WAV of 44 header + 12 data bytes
        assert_eq!(wav_bytes.len(), 56, "WAV should be 56 bytes total");

        // Verify RIFF header correctness
        assert!(
            wav_bytes.starts_with(b"RIFF"),
            "WAV should start with RIFF marker"
        );
        assert!(
            wav_bytes[8..12].starts_with(b"WAVE"),
            "WAV should contain WAVE format"
        );
        assert!(
            wav_bytes[12..16].starts_with(b"fmt "),
            "WAV should contain fmt chunk"
        );

        // Read sample rate from header (offset 24, 4 bytes LE)
        let header_sr = u32::from_le_bytes(wav_bytes[24..28].try_into().unwrap());
        assert_eq!(header_sr, sample_rate, "WAV header sample rate mismatch");

        // Read bits per sample (offset 34, 2 bytes LE)
        let bps = u16::from_le_bytes(wav_bytes[34..36].try_into().unwrap());
        assert_eq!(bps, 16, "WAV should be 16-bit PCM");

        // Read number of channels (offset 22, 2 bytes LE)
        let channels = u16::from_le_bytes(wav_bytes[22..24].try_into().unwrap());
        assert_eq!(channels, 1, "WAV should be mono");

        // Verify data chunk: expected size = 6 samples × 2 bytes = 12
        let data_size = u32::from_le_bytes(wav_bytes[40..44].try_into().unwrap());
        assert_eq!(
            data_size, 12,
            "WAV data size should be 12 bytes for 6 16-bit samples"
        );

        // Verify the last sample (samples[5] = 0.0 → i16 = 0) appears
        // at the end of the data section.
        let data_start = 44;
        let last_sample_bytes = &wav_bytes[data_start + 10..data_start + 12];
        assert_eq!(last_sample_bytes, &[0x00, 0x00], "last sample should be 0");
    }

    #[test]
    fn test_verify_sha256_empty_hash() {
        // Empty expected hash skips verification (returns Ok) — same pattern as voice.rs
        let tmp = std::env::temp_dir().join("test_tts_empty_sha256.txt");
        std::fs::write(&tmp, b"content").unwrap();
        assert!(
            verify_sha256(&tmp, "").is_ok(),
            "empty SHA256 expected hash should skip verification"
        );
        std::fs::remove_file(&tmp).ok();
    }

    // ── Tier 1: Preprocessing helpers (always-run unit tests) ─────────

    #[test]
    fn test_normalize_symbols() {
        let r = normalize_symbols("Hello—world–test\u{2011}");
        assert_eq!(r, "Hello-world-test-", "em/en dashes become hyphen");

        let r2 = normalize_symbols(
            "a\u{00AF}b[c]d|e/f#g\u{2192}h\u{2190}i\u{2665}j\u{2606}k\u{2661}l\u{00A9}m\\n",
        );
        assert_eq!(
            r2, "a b c d e f g h i j k l m n",
            "symbols like overline, brackets, pipe, slash, hash, arrows, hearts, copyright, backslash become space"
        );

        let r3 = normalize_symbols("\u{201C}hello\u{201D}");
        assert_eq!(
            r3, "\"hello\"",
            "curly double quotes become straight double quote"
        );

        let r4 = normalize_symbols("\u{2018}hello\u{2019}\u{00B4}`");
        assert_eq!(
            r4, "'hello'''",
            "curly single quotes, acute, backtick become straight single quote"
        );

        // Characters that should pass through unchanged
        let r5 = normalize_symbols("abc123.!?");
        assert_eq!(
            r5, "abc123.!?",
            "normal alphanumeric and basic punctuation pass through"
        );
    }

    #[test]
    fn test_expand_abbreviations() {
        assert_eq!(expand_abbreviations("@user"), " at user");
        // Note: "e.g.," → "for example, " (trailing space) because the original
        // text has a space after the comma — this is fine as clean_whitespace
        // later normalises it.
        assert_eq!(expand_abbreviations("e.g., hello"), "for example,  hello");
        assert_eq!(expand_abbreviations("i.e., world"), "that is,  world");
        assert_eq!(
            expand_abbreviations("multiple e.g., and i.e., here"),
            "multiple for example,  and that is,  here"
        );
        // Text without abbreviations passes through unchanged
        assert_eq!(expand_abbreviations("hello world"), "hello world");
        // Case-sensitive: only lowercase exact match
        assert_eq!(expand_abbreviations("E.G.,"), "E.G.,");
    }

    #[test]
    fn test_fix_punctuation_spacing() {
        assert_eq!(fix_punctuation_spacing("hello ,world"), "hello,world");
        assert_eq!(fix_punctuation_spacing("hello .world"), "hello.world");
        assert_eq!(fix_punctuation_spacing("hello !world"), "hello!world");
        assert_eq!(fix_punctuation_spacing("hello ?world"), "hello?world");
        assert_eq!(fix_punctuation_spacing("hello ;world"), "hello;world");
        assert_eq!(fix_punctuation_spacing("hello :world"), "hello:world");
        assert_eq!(fix_punctuation_spacing("hello 'world"), "hello'world");
        assert_eq!(
            fix_punctuation_spacing("hello , . ; : ! ? 'world"),
            "hello,.;:!?'world"
        );
        // Text without spacing issues passes through unchanged
        assert_eq!(fix_punctuation_spacing("Hello, world!"), "Hello, world!");
    }

    #[test]
    fn test_remove_duplicate_quotes() {
        assert_eq!(
            remove_duplicate_quotes(r#""hello""#),
            r#""hello""#,
            "single pair unchanged"
        );
        assert_eq!(
            remove_duplicate_quotes(r#""""hello"""""#),
            r#""hello""#,
            "double quotes deduplicated"
        );
        assert_eq!(
            remove_duplicate_quotes("''hello''"),
            "'hello'",
            "single quotes deduplicated"
        );
        assert_eq!(
            remove_duplicate_quotes("``hello``"),
            "`hello`",
            "backticks deduplicated"
        );
        assert_eq!(
            remove_duplicate_quotes(r#""'mixed"#),
            r#""'mixed"#,
            "different quote chars are not collapsed"
        );
        assert_eq!(
            remove_duplicate_quotes("no quotes"),
            "no quotes",
            "text without quotes unchanged"
        );
    }

    #[test]
    fn test_clean_whitespace() {
        assert_eq!(
            clean_whitespace("hello   world"),
            "hello world",
            "multiple spaces collapsed"
        );
        assert_eq!(
            clean_whitespace("  hello  world  "),
            "hello world",
            "leading/trailing whitespace trimmed"
        );
        assert_eq!(
            clean_whitespace("hello\tworld"),
            "hello world",
            "tabs become space"
        );
        assert_eq!(
            clean_whitespace("hello\n\nworld"),
            "hello world",
            "newlines collapsed to space"
        );
        assert_eq!(clean_whitespace("  "), "", "whitespace-only becomes empty");
        assert_eq!(
            clean_whitespace("hello world"),
            "hello world",
            "normal text unchanged"
        );
    }

    #[test]
    fn test_remove_emojis() {
        // A selection of emojis from different ranges
        let emoji_text = "Hello 😊😢🔥👍🏆🎉💯";
        assert_eq!(
            remove_emojis(emoji_text),
            "Hello ",
            "emoji characters removed"
        );

        // Emoticons range: U+1F600..=U+1F64F
        assert_eq!(remove_emojis("😀😁😂🤣😃😄😅😆"), "", "emoticons removed");

        // Symbols and pictographs range: U+1F300..=U+1F5FF
        assert_eq!(remove_emojis("🌀🌂🌁"), "", "misc symbols removed");

        // Transport range: U+1F680..=U+1F6FF
        assert_eq!(remove_emojis("🚀🚁🚂"), "", "transport symbols removed");

        // Various other emoji ranges
        assert_eq!(
            remove_emojis("🛀🛁🛂🛃🛄🛅"),
            "",
            "transport supplement removed"
        );
        assert_eq!(
            remove_emojis("🤐🤑🤒🤓🤔🤕🤖"),
            "",
            "supplemental symbols removed"
        );
        assert_eq!(remove_emojis("🥰🥱🥴🥳🥺"), "", "extended symbols removed");
        assert_eq!(remove_emojis("🦾🦿🧠🧡"), "", "symbols ext A removed");
        // Note: ZWJ sequences like 🧑‍🦰 contain U+200D (zero-width joiner) which is
        // not in the emoji ranges, so it passes through.
        assert_eq!(
            remove_emojis("🧑‍🦰"),
            "\u{200d}",
            "ZWJ character survives emoji removal"
        );

        // Misc symbols: U+2600..=U+26FF
        // Note: ☀️ contains U+FE0F (variation selector-16) which is not in the
        // emoji ranges, so it passes through. We test with bare U+2600 instead.
        assert_eq!(
            remove_emojis("\u{2600}\u{2601}\u{2602}\u{2603}"),
            "",
            "misc symbols removed"
        );

        // Dingbats: U+2700..=U+27BF
        assert_eq!(remove_emojis("✀✁✂✃✄✅"), "", "dingbats removed");

        // Non-emoji text passes through
        assert_eq!(
            remove_emojis("Hello, world!"),
            "Hello, world!",
            "plain text unchanged"
        );
    }

    /// Panic-safe guard that restores the TTS language config on drop.
    struct TtsLangGuard {
        saved: String,
    }

    impl Drop for TtsLangGuard {
        fn drop(&mut self) {
            CONFIG.set_string_field("tts_language", &self.saved);
        }
    }

    #[test]
    fn test_wrap_lang_tag() {
        let _guard = TtsLangGuard {
            saved: CONFIG.tts_language(),
        };

        // Force language to "en" for deterministic test
        let _ = CONFIG.set_string_field("tts_language", "en");
        let r = wrap_lang_tag("Hello world");
        assert_eq!(r, "<en>Hello world</en>", "text wrapped in language tag");

        // Test with language-agnostic tag
        let _ = CONFIG.set_string_field("tts_language", "na");
        let r2 = wrap_lang_tag("Test");
        assert_eq!(r2, "<na>Test</na>");
        // _guard restores original language on drop (including on panic)
    }

    // ── Tier 2: encode_text with synthetic indexer ────────────────────

    #[test]
    fn test_encode_text_with_indexer() {
        // Synthetic unicode indexer: maps ASCII chars 0-127 to sequential IDs.
        // indexer[i] = i for i in 0..128, with UNK positioned at index 0.
        // This means:
        //   - 'H' (72)  → ID 72
        //   - 'w' (119) → ID 119
        //   - '😊' (U+1F60A = 128522) → out of range → UNK (ID 0)
        //
        // We specifically set space (32) to -1 so it falls through to UNK,
        // simulating a real indexer where space is not a valid token.
        let mut indexer = vec![-1i32; 128];
        for i in 0..128 {
            indexer[i] = i as i32;
        }
        // ID 0 is the UNK token
        indexer[0] = 0;
        // Space (code 32) → UNK (not a valid token in the model)
        indexer[32] = -1;

        let (ids, len) = encode_text_with_indexer(&indexer, "H w");
        assert_eq!(len, 3, "three characters encoded");
        assert_eq!(ids, vec![72, 0, 119], "space should map to UNK (ID 0)");

        // Test with empty indexer (edge case)
        let (ids2, len2) = encode_text_with_indexer(&[], "hello");
        assert_eq!(len2, 5, "five chars encoded with empty indexer");
        assert_eq!(ids2, vec![0; 5], "all map to UNK (default 0)");

        // Test with single-entry indexer (no UNK differentiation)
        let (ids3, len3) = encode_text_with_indexer(&vec![42], "abc");
        assert_eq!(len3, 3);
        assert_eq!(ids3, vec![42; 3], "all chars map to UNK (first entry = 42)");

        // Test with negative UNK sentinel
        let (ids4, len4) = encode_text_with_indexer(&vec![-1], "x");
        assert_eq!(len4, 1);
        assert_eq!(ids4, vec![-1i64], "UNK sentinel preserved");
    }

    /// Model-free test for the empty-token-sequence guard.
    ///
    /// The guard in [`synthesize_internal`] checks `if seq_len == 0` after
    /// calling [`encode_text`]. This test validates the precondition directly
    /// using [`encode_text_with_indexer`] with a synthetic indexer, so it
    /// runs in any environment regardless of whether TTS model files are
    /// cached.
    #[test]
    fn test_encode_text_empty_input_produces_zero_len() {
        // Empty string → zero-length output (triggers the seq_len == 0 guard)
        let (ids, len) = encode_text_with_indexer(&vec![0i32; 128], "");
        assert_eq!(
            len, 0,
            "empty input must produce zero-length token sequence"
        );
        assert!(ids.is_empty(), "token IDs must be empty for empty input");

        // Non-empty string still produces tokens (guard not triggered)
        let (_, len2) = encode_text_with_indexer(&vec![0i32; 128], "a");
        assert_eq!(len2, 1, "single-char input should produce one token");

        // Whitespace-only strings are NOT empty — they produce tokens.
        // The guard only triggers for truly empty character sequences.
        let (_, len3) = encode_text_with_indexer(&vec![0i32; 128], " ");
        assert_eq!(len3, 1, "whitespace input produces a token");
    }

    // ── Tier 3: Integration tests with real models ────────────────────
    //
    // These tests require the Supertonic 3 model files to be cached in
    // `~/.mahbot/models/supertonic3/`. They are skipped if the files are
    // not found (matching the embedder test pattern).

    /// Helper to obtain a loaded [`TtsEngine`] for integration tests.
    ///
    /// Returns `None` (and skips the calling test) when:
    /// - `MAHBOT_SKIP_TTS_TESTS=1` is set
    /// - Model files are not cached on disk
    ///
    /// Caches the loaded engine via [`OnceLock`] so the ~2s load cost is
    /// paid only once per test run.
    fn test_tts_engine() -> Option<&'static TtsEngine> {
        use std::sync::OnceLock;

        // Share a single model load across all tests via OnceLock.
        static TEST_TTS_ENGINE: OnceLock<Option<TtsEngine>> = OnceLock::new();

        TEST_TTS_ENGINE
            .get_or_init(|| {
                // Skip if env var is set
                if std::env::var("MAHBOT_SKIP_TTS_TESTS").is_ok() {
                    return None;
                }

                // Collect all candidate models directories (deduplicated).
                let mut candidates = Vec::new();

                // 1. CONFIG storage root (may be a temp dir from graceful degradation test).
                if let Some(root) = crate::config::CONFIG.try_storage_root() {
                    candidates.push(root.join("models").join(MODEL_DIR_NAME));
                }

                // 2. Real home directory cache (always present in dev/CI environments).
                if let Some(home) = std::env::var("HOME").ok().filter(|h| !h.is_empty()) {
                    let real = std::path::PathBuf::from(&home)
                        .join(".mahbot")
                        .join("models")
                        .join(MODEL_DIR_NAME);
                    if !candidates.contains(&real) {
                        candidates.push(real);
                    }
                }

                // Try each candidate until we find model files.
                for dir in &candidates {
                    let onnx_dir = dir.join(ONNX_DIR);
                    // Check essential files: one ONNX model, config, indexer, and voice style
                    let essential_files = [
                        onnx_dir.join(DP_ONNX_NAME),
                        onnx_dir.join(TEXT_ENC_ONNX_NAME),
                        onnx_dir.join(VECTOR_EST_ONNX_NAME),
                        onnx_dir.join(VOCODER_ONNX_NAME),
                        onnx_dir.join(TTS_JSON_NAME),
                        onnx_dir.join(UNICODE_INDEXER_NAME),
                        dir.join(VOICE_STYLES_DIR).join(DEFAULT_VOICE_NAME),
                    ];
                    if essential_files.iter().all(|p| p.exists()) {
                        match load_engine(dir) {
                            Ok(engine) => return Some(engine),
                            Err(e) => {
                                eprintln!("WARNING: Failed to load test TTS engine: {e}");
                                return None;
                            }
                        }
                    }
                }

                // No model files found in any candidate directory.
                let last_candidate = candidates.last().map(|p| p.display().to_string());
                eprintln!(
                    "WARNING: Supertonic 3 model files not found. Looked in: {}. \
                     Set MAHBOT_SKIP_TTS_TESTS=1 to suppress this warning.",
                    last_candidate.as_deref().unwrap_or("<none>")
                );
                None
            })
            .as_ref()
    }

    /// Helper to obtain the default voice style tensors for integration tests.
    ///
    /// Returns `None` (and skips the calling test) when:
    /// - `MAHBOT_SKIP_TTS_TESTS=1` is set
    /// - The default voice style file is not cached on disk
    /// - Loading/parsing fails
    ///
    /// Caches the loaded tensors via [`OnceLock`] so the JSON parse cost is
    /// paid only once per test run.
    fn test_voice_style() -> Option<(&'static Tensor, &'static Tensor)> {
        // First check if engine is available (which implies model files exist)
        test_tts_engine()?;

        static TEST_VOICE_STYLE: OnceLock<Option<(Tensor, Tensor)>> = OnceLock::new();

        TEST_VOICE_STYLE
            .get_or_init(|| {
                let dir = model_dir()?;
                let style_path = dir.join(VOICE_STYLES_DIR).join(DEFAULT_VOICE_NAME);
                if style_path.exists() {
                    match load_voice_style(&dir, DEFAULT_VOICE_NAME) {
                        Ok(styles) => return Some(styles),
                        Err(e) => {
                            tracing::warn!("Failed to load test voice style: {e}");
                            return None;
                        }
                    }
                }
                None
            })
            .as_ref()
            .as_ref()
            .map(|(dp, ttl)| (dp, ttl))
    }

    #[test]
    fn test_synthesize_short_text_creates_valid_wav() {
        let Some(engine) = test_tts_engine() else {
            return; // Skip if no model available
        };
        let Some((style_dp, style_ttl)) = test_voice_style() else {
            return;
        };

        let samples = synthesize_internal(engine, "<en>Hello world.</en>", style_dp, style_ttl, 42)
            .expect("synthesis of short text should succeed");

        assert!(!samples.is_empty(), "synthesized audio must not be empty");
        assert!(
            samples.len() > 1000,
            "short text should produce at least 1000 samples (got {})",
            samples.len()
        );

        // Verify samples are in valid f32 range [-1.0, 1.0]
        for &s in &samples {
            assert!((-1.0..=1.0).contains(&s), "sample {s} out of valid range");
        }

        // Verify the WAV rendering works with the synthesized samples
        let wav_bytes =
            render_wav(&samples, engine.sample_rate).expect("render_wav should succeed");
        assert!(wav_bytes.starts_with(b"RIFF"), "WAV should start with RIFF");
        assert!(wav_bytes.len() > 44, "WAV should have header + data");
    }

    #[test]
    fn test_synthesize_chunked_long_text() {
        let Some(engine) = test_tts_engine() else {
            return; // Skip if no model available
        };
        let Some((style_dp, style_ttl)) = test_voice_style() else {
            return;
        };

        // Create text longer than MAX_CHUNK_LENGTH (300 chars)
        let long_text =
            "Hello world. This is a test of the chunked synthesis pipeline. ".repeat(20);
        assert!(
            long_text.len() > MAX_CHUNK_LENGTH,
            "test text must exceed chunk limit"
        );

        let samples = synthesize_chunked(engine, &long_text, style_dp, style_ttl, None, 42)
            .expect("chunked synthesis should succeed");

        assert!(
            !samples.is_empty(),
            "chunked synthesis must produce samples"
        );
        assert!(
            samples.len() > 1000,
            "chunked text should produce at least 1000 samples (got {})",
            samples.len()
        );

        // Verify samples are in valid f32 range [-1.0, 1.0]
        for &s in &samples {
            assert!((-1.0..=1.0).contains(&s), "sample {s} out of valid range");
        }
    }

    #[tokio::test]
    async fn test_synthesize_chunked_streaming_receives_chunks() {
        let Some(engine) = test_tts_engine() else {
            return; // Skip if no model available
        };
        let Some((style_dp, style_ttl)) = test_voice_style() else {
            return;
        };

        // Create text longer than MAX_CHUNK_LENGTH to trigger chunking
        let long_text =
            "Hello world. This is a test of the streaming synthesis pipeline. ".repeat(20);
        assert!(
            long_text.len() > MAX_CHUNK_LENGTH,
            "test text must exceed chunk limit"
        );

        let (tx, mut rx) = mpsc::channel::<Vec<f32>>(4);

        // Spawn the streaming synthesis on a blocking thread (as in production)
        let text_for_task = long_text.clone();
        tokio::task::spawn_blocking(move || {
            synthesize_chunked_streaming(
                engine,
                &text_for_task,
                style_dp,
                style_ttl,
                None,
                &tx,
                42,
            )
            .expect("streaming synthesis should succeed");
        })
        .await
        .expect("blocking task should not panic");

        // Collect all chunks from the channel
        let mut chunk_count = 0;
        let mut total_samples = 0usize;
        while let Some(samples) = rx.recv().await {
            assert!(!samples.is_empty(), "each chunk must produce samples");
            for &s in &samples {
                assert!((-1.0..=1.0).contains(&s), "sample {s} out of valid range");
            }
            total_samples += samples.len();
            chunk_count += 1;
        }

        assert!(
            chunk_count > 1,
            "long text should produce multiple chunks (got {chunk_count})"
        );
        assert!(
            total_samples > 1000,
            "long text should produce many samples (got {total_samples})"
        );
    }

    #[tokio::test]
    async fn test_synthesize_chunked_streaming_short_fast_path() {
        let Some(engine) = test_tts_engine() else {
            return; // Skip if no model available
        };
        let Some((style_dp, style_ttl)) = test_voice_style() else {
            return;
        };

        // Short text below MAX_CHUNK_LENGTH — should take the fast path
        // (single send, no chunk loop).
        let short_text = "Hello world.";

        let (tx, mut rx) = mpsc::channel::<Vec<f32>>(4);

        let text_for_task = short_text.to_string();
        tokio::task::spawn_blocking(move || {
            synthesize_chunked_streaming(
                engine,
                &text_for_task,
                style_dp,
                style_ttl,
                None,
                &tx,
                42,
            )
            .expect("streaming synthesis of short text should succeed");
        })
        .await
        .expect("blocking task should not panic");

        // Should receive exactly one chunk, then channel closes
        let first = rx.recv().await;
        assert!(first.is_some(), "short text must send one chunk");
        let samples = first.unwrap();
        assert!(!samples.is_empty(), "chunk must not be empty");
        assert!(
            rx.recv().await.is_none(),
            "short text must send exactly one chunk"
        );
    }

    #[tokio::test]
    async fn test_synthesize_chunked_streaming_cancelled_on_receiver_drop() {
        let Some(engine) = test_tts_engine() else {
            return; // Skip if no model available
        };
        let Some((style_dp, style_ttl)) = test_voice_style() else {
            return;
        };

        // Dropping the receiver before the sender writes causes the
        // blocking_send to fail with a channel-closed error.
        let (tx, rx) = mpsc::channel::<Vec<f32>>(4);
        let text = "Hello world. This tests receiver drop cancellation.";

        // Drop the receiver immediately — the sender will detect the
        // closed channel on its first blocking_send.
        drop(rx);

        let text_for_task = text.to_string();
        let result = tokio::task::spawn_blocking(move || {
            synthesize_chunked_streaming(engine, &text_for_task, style_dp, style_ttl, None, &tx, 42)
        })
        .await
        .expect("blocking task should not panic");

        assert!(
            result.is_err(),
            "streaming synthesis should error when receiver is dropped"
        );
        let err = format!("{:#}", result.err().unwrap());
        assert!(
            err.contains("receiver dropped"),
            "error should mention receiver drop, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_synthesize_chunked_streaming_cancelled_by_flag() {
        let Some(engine) = test_tts_engine() else {
            return; // Skip if no model available
        };
        let Some((style_dp, style_ttl)) = test_voice_style() else {
            return;
        };

        // Setting the cancel flag before synthesis starts causes an
        // immediate bail without any model work.
        let cancel_flag = Arc::new(AtomicBool::new(true));
        let (tx, _rx) = mpsc::channel::<Vec<f32>>(4);

        let text_for_task = "This should not be synthesized.".to_string();
        let result = tokio::task::spawn_blocking(move || {
            synthesize_chunked_streaming(
                engine,
                &text_for_task,
                style_dp,
                style_ttl,
                Some(&cancel_flag),
                &tx,
                42,
            )
        })
        .await
        .expect("blocking task should not panic");

        assert!(
            result.is_err(),
            "streaming synthesis should error on cancellation"
        );
        let err = format!("{:#}", result.err().unwrap());
        assert!(
            err.contains("cancelled") || err.contains("Cancelled"),
            "error should mention cancellation, got: {err}"
        );
    }

    #[test]
    fn test_synthesis_deterministic_with_seed_42() {
        let Some(engine) = test_tts_engine() else {
            return; // Skip if no model available
        };
        let Some((style_dp, style_ttl)) = test_voice_style() else {
            return;
        };

        let text = "<en>Hello world, this is a deterministic test.</en>";

        // Synthesize the same text twice and verify identical output.
        // The flow matching process is deterministic for a given input
        // (no random seed involved), so repeated calls on the same
        // hardware should produce bit-identical results.
        let samples1 = synthesize_internal(engine, text, style_dp, style_ttl, 42)
            .expect("first synthesis should succeed");
        let samples2 = synthesize_internal(engine, text, style_dp, style_ttl, 42)
            .expect("second synthesis should succeed");

        assert_eq!(
            samples1.len(),
            samples2.len(),
            "both syntheses should produce the same number of samples"
        );

        // Compare bit-exactness of the first 1000 samples using direct
        // f32 equality (which checks bit patterns).
        let check_len = samples1.len().min(1000);
        for i in 0..check_len {
            assert_eq!(
                samples1[i], samples2[i],
                "sample {i} differs: {:.10} vs {:.10}",
                samples1[i], samples2[i]
            );
        }
    }

    #[test]
    fn test_empty_text_fails_fast() {
        // synthesize_internal with empty text should bail cleanly at the
        // Rust-level guard (seq_len == 0) before reaching any ONNX model code.
        let Some(engine) = test_tts_engine() else {
            return; // Skip if no model available
        };
        let Some((style_dp, style_ttl)) = test_voice_style() else {
            return;
        };

        let result = synthesize_internal(engine, "", style_dp, style_ttl, 42);
        assert!(result.is_err(), "empty text synthesis should fail");

        // Verify the error message mentions empty input
        let err = result.err().unwrap();
        let err_msg = format!("{err:#}");
        assert!(
            err_msg.contains("Empty token sequence"),
            "error should mention 'Empty token sequence', got: {err_msg}"
        );
    }

    // ── init_listener ────────────────────────────────────────────────
    //
    // These tests verify that init_listener() correctly dispatches to
    // speak() when a matching ChatEvent::Message arrives on CHAT_BROADCAST,
    // and that the guard conditions (is_enabled) are respected.

    /// Broadcast a ChatEvent::Message with the given parameters to CHAT_BROADCAST.
    /// Panics if CHAT_BROADCAST is not initialized.
    fn broadcast_test_event(
        direction: crate::ChatDirection,
        channel: &str,
        agent_role: Option<&str>,
    ) {
        let tx = crate::CHAT_BROADCAST.get().unwrap();
        let _ = tx.send(crate::ChatEvent::Message {
            message_id: "test-tts".to_string(),
            user_name: "testuser".to_string(),
            content: "Ignore — test event.".to_string(),
            direction,
            timestamp: String::new(),
            channel: channel.to_string(),
            agent_role: agent_role.map(String::from),
            workspace: "test".to_string(),
            optimistic_id: None,
        });
    }

    #[tokio::test]
    #[serial_test::serial(tts)]
    async fn test_init_listener_dispatches_speak() {
        // Initialize test stores so resolve_active_role defaults are available
        crate::util::test::init_test_stores().await;

        // Set up CHAT_BROADCAST (idempotent — safe to call from parallel tests)
        crate::CHAT_BROADCAST.get_or_init(|| {
            let (tx, _rx) = tokio::sync::broadcast::channel(256);
            tx
        });

        // Enable TTS for the happy path
        let prev_state = STATE.load(Ordering::Acquire);
        STATE.store(STATE_READY, Ordering::Release);
        crate::config::CONFIG.set_string_field("tts_enabled", "true");

        // Reset speak counter
        SPEAK_COUNT.store(0, Ordering::Release);

        // Start the listener
        init_listener();

        // Give the listener time to subscribe before we send
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Broadcast a matching event (Agent direction, gui channel, analyst role)
        broadcast_test_event(crate::ChatDirection::Agent, "gui", Some("analyst"));

        // Wait for the listener to process (up to 500ms total)
        let mut spoke = false;
        for _ in 0..5 {
            if SPEAK_COUNT.load(Ordering::Acquire) > 0 {
                spoke = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        assert!(
            spoke,
            "speak() should have been called after matching ChatEvent::Message"
        );

        // Restore global state for other tests
        STATE.store(prev_state, Ordering::Release);
    }

    #[tokio::test]
    #[serial_test::serial(tts)]
    async fn test_init_listener_skips_when_disabled() {
        // Ensure TTS is disabled (default state: STATE_UNINIT)
        let prev_state = STATE.load(Ordering::Acquire);
        STATE.store(STATE_UNINIT, Ordering::Release);

        crate::util::test::init_test_stores().await;

        crate::CHAT_BROADCAST.get_or_init(|| {
            let (tx, _rx) = tokio::sync::broadcast::channel(256);
            tx
        });

        SPEAK_COUNT.store(0, Ordering::Release);
        init_listener();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Broadcast the same matching event
        broadcast_test_event(crate::ChatDirection::Agent, "gui", Some("analyst"));

        // Wait enough time for processing
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        assert_eq!(
            SPEAK_COUNT.load(Ordering::Acquire),
            0,
            "speak() must NOT be called when TTS is disabled"
        );

        STATE.store(prev_state, Ordering::Release);
    }
}
