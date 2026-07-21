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
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, LazyLock, OnceLock, RwLock};
use std::time::Duration;
use tokio::sync::broadcast;
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
const SYNTHESIS_TIMEOUT: Duration = Duration::from_mins(5);
const MAX_CHUNK_LENGTH: usize = 300;
const SILENCE_DURATION: f32 = 0.3;

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

// ── State machine ─────────────────────────────────────────────────────

const STATE_UNINIT: u8 = 0;
const STATE_LOADING: u8 = 1;
const STATE_READY: u8 = 2;
const STATE_FAILED: u8 = 3;

static STATE: AtomicU8 = AtomicU8::new(STATE_UNINIT);
static GLOBAL_TTS: OnceLock<RwLock<Option<Arc<TtsEngine>>>> = OnceLock::new();
static CANCEL_TX: OnceLock<broadcast::Sender<()>> = OnceLock::new();
static CANCEL_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();

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
    style_dp: Tensor,
    style_ttl: Tensor,
    sample_rate: u32,
    latent_dim: usize,
    chunk_compress_factor: usize,
    base_chunk_size: usize,
    device: Device,
}

#[derive(Debug, Deserialize)]
struct VoiceStyleFile {
    #[serde(rename = "style_dp")]
    style_dp: Vec<Vec<f32>>,
    #[serde(rename = "style_ttl")]
    style_ttl: Vec<Vec<f32>>,
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
/// Spawns a background task that synthesizes audio, plays it via the OS-native
/// audio player (afplay on macOS), then deletes the temp file.
/// Silently ignored if TTS is disabled, models aren't ready, or text is empty.
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

    // Clean up stale temp directories from previous process instances
    // that may have left orphaned WAV files (macOS does not auto-clean /tmp).
    // Run unconditionally even though config hasn't been loaded from DB yet
    // — this is a cheap directory listing with negligible cost.
    cleanup_stale_temp_dirs();

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

/// Try to load TTS models from cache at startup.
/// Returns `true` if loaded, `false` if not (download will happen async).
pub fn try_load_cached() -> bool {
    if STATE.load(Ordering::Acquire) != STATE_UNINIT {
        return STATE.load(Ordering::Acquire) == STATE_READY;
    }

    let Some(dir) = model_dir() else {
        return false;
    };

    let all_exist = [
        dir.join(ONNX_DIR).join(DP_ONNX_NAME),
        dir.join(ONNX_DIR).join(TEXT_ENC_ONNX_NAME),
        dir.join(ONNX_DIR).join(VECTOR_EST_ONNX_NAME),
        dir.join(ONNX_DIR).join(VOCODER_ONNX_NAME),
        dir.join(ONNX_DIR).join(TTS_JSON_NAME),
        dir.join(ONNX_DIR).join(UNICODE_INDEXER_NAME),
        dir.join(VOICE_STYLES_DIR).join(DEFAULT_VOICE_NAME),
    ]
    .iter()
    .all(|p| p.exists());

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
    if voice.style_dp.is_empty() || voice.style_ttl.is_empty() {
        anyhow::bail!("Voice style has empty vectors");
    }
    let device = Device::Cpu;
    let dp = Tensor::from_slice(&voice.style_dp[0], (1, voice.style_dp[0].len()), &device)?;
    let ttl = Tensor::from_slice(&voice.style_ttl[0], (1, voice.style_ttl[0].len()), &device)?;
    Ok((dp, ttl))
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

    let (style_dp, style_ttl) = load_voice_style(dir, DEFAULT_VOICE_NAME)?;

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
        style_dp,
        style_ttl,
        sample_rate,
        latent_dim,
        chunk_compress_factor,
        base_chunk_size,
        device: Device::Cpu,
    })
}

// ── Download retry loop ──────────────────────────────────────────────

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
            warn!("TTS download failed after {MAX_DOWNLOAD_RETRIES} retries");
            STATE.store(STATE_FAILED, Ordering::Release);
            return;
        }

        match tokio::time::timeout(MODEL_DOWNLOAD_TIMEOUT, ensure_models_downloaded(&dir)).await {
            Ok(Ok(())) => match load_engine(&dir) {
                Ok(e) => {
                    set_engine_ready(e);
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

    let files = [
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
        TtsFile {
            url: format!("{base}/{VOICE_STYLES_DIR}/{DEFAULT_VOICE_NAME}"),
            path: dir.join(VOICE_STYLES_DIR).join(DEFAULT_VOICE_NAME),
            sha256: VOICE_STYLE_SHA256,
        },
    ];

    // Parallel downloads for I/O-bound model files
    let download_futures: Vec<_> = files
        .into_iter()
        .map(|f| async move { ensure_file(&f).await })
        .collect();

    futures_util::future::try_join_all(download_futures).await?;
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

/// Download a single file with atomic write and SHA256 verification.
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
    let bytes = response.bytes().await.context("Failed to download file")?;

    if bytes.len() < 100 {
        anyhow::bail!("Downloaded file too small: {} bytes", bytes.len());
    }

    let tmp = dest.with_extension("tmp");
    {
        let mut f = File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.flush()?;
    }

    // Verify hash BEFORE renaming to final path
    if !expected_hash.is_empty()
        && let Err(e) = verify_sha256(&tmp, expected_hash)
    {
        let _ = std::fs::remove_file(&tmp);
        return Err(e)
            .with_context(|| format!("SHA256 verification failed for {}", dest.display()));
    }

    std::fs::rename(&tmp, dest)?;

    let hash_str = if expected_hash.is_empty() {
        String::new()
    } else {
        format!(" (SHA256: {expected_hash})")
    };

    info!(
        "Downloaded {}{} ({:.1} MB)",
        dest.file_name().unwrap_or_default().to_string_lossy(),
        hash_str,
        bytes.len() as f64 / 1_048_576.0
    );
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

fn encode_text(engine: &TtsEngine, text: &str) -> (Vec<i64>, usize) {
    let unk_id = engine.unicode_indexer.first().copied().unwrap_or(0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let max_idx = engine.unicode_indexer.len() as i32 - 1;
    let ids: Vec<i64> = text
        .chars()
        .map(|c| {
            let code = c as usize;
            if code < engine.unicode_indexer.len() {
                let tid = engine.unicode_indexer[code];
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
fn synthesize_internal(engine: &TtsEngine, text: &str) -> Result<Vec<f32>> {
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
            ("style_dp", engine.style_dp.clone()),
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
            ("style_ttl", engine.style_ttl.clone()),
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
    let mut rng = 42u64;
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
                ("style_ttl", engine.style_ttl.clone()),
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
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn synthesize_chunked(
    engine: &TtsEngine,
    text: &str,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<f32>> {
    // Check cancellation before starting any work
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        anyhow::bail!("Synthesis cancelled");
    }

    if text.len() <= MAX_CHUNK_LENGTH {
        return synthesize_internal(engine, &wrap_lang_tag(text));
    }
    let silence_samples = (SILENCE_DURATION * engine.sample_rate as f32) as usize;
    let mut result = Vec::new();
    let silence = vec![0.0f32; silence_samples];

    for chunk in split_at_sentence_boundaries(text, MAX_CHUNK_LENGTH) {
        // Check cancellation between chunks
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            anyhow::bail!("Synthesis cancelled");
        }
        if let Ok(samples) = synthesize_internal(engine, &wrap_lang_tag(&chunk)) {
            result.extend(samples);
            result.extend_from_slice(&silence);
        }
    }
    Ok(result)
}

/// Wrap text in English language tag for the Supertonic 3 model.
fn wrap_lang_tag(text: &str) -> String {
    format!("<en>{text}</en>")
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

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let channels: u16 = 1;
    let bps: u16 = 16;
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bps / 8);
    let block_align = channels * (bps / 8);
    let data_size = u32::try_from(samples.len() * (bps / 8) as usize)
        .context("WAV data size exceeds u32 range")?;
    let file_size = 36 + data_size;

    // Buffer all header and sample data, then write once
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

    let mut f = File::create(path)?;
    f.write_all(&buf)?;
    f.flush()?;
    Ok(())
}

// ── Async speak ──────────────────────────────────────────────────────

/// Temp directory for ephemeral TTS WAV files.
///
/// Uses a per-process directory name (containing the PID) so that each
/// process instance has its own isolated temp space.  Stale directories
/// from crashed processes are cleaned up at [`init_global`] time.
fn session_temp_dir() -> PathBuf {
    std::env::temp_dir().join(format!("mahbot_tts_{}", std::process::id()))
}

/// Remove stale TTS temp directories from previous process instances.
///
/// Scans the OS temp directory for `mahbot_tts_<PID>` subdirectories
/// whose PID does not match the current process.  This prevents
/// unbounded accumulation of orphaned WAV files from crashes
/// (macOS `/tmp` is not cleared on reboot).
fn cleanup_stale_temp_dirs() {
    let current_pid = std::process::id();
    let prefix = "mahbot_tts_";
    if let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) {
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_dir() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(String::from) else {
                continue;
            };
            if let Some(pid_str) = name.strip_prefix(prefix)
                && let Ok(pid) = pid_str.parse::<u32>()
                && pid != current_pid
            {
                let path = entry.path();
                if let Err(e) = std::fs::remove_dir_all(&path) {
                    warn!(
                        "TTS: failed to remove stale temp dir {}: {e}",
                        path.display()
                    );
                } else {
                    info!("TTS: cleaned up stale temp dir: {}", path.display());
                }
            }
        }
    }
}

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

    // Reset cancellation flag for this synthesis, then clone the Arc
    // so it can be passed into the spawn_blocking closure.
    let cancel_flag = CANCEL_FLAG.get().map(|f| {
        f.store(false, Ordering::Release);
        Arc::clone(f)
    });

    // CPU-bound synthesis: run on blocking threadpool with timeout.
    let synthesized = match tokio::time::timeout(
        SYNTHESIS_TIMEOUT,
        tokio::task::spawn_blocking(move || {
            synthesize_chunked(&engine, &processed, cancel_flag.as_deref())
        }),
    )
    .await
    {
        Ok(Ok(Ok(samples))) => Ok(samples),
        Ok(Ok(Err(e))) => Err(e),
        Ok(Err(join_err)) => {
            warn!("TTS synthesis task panicked: {join_err}");
            return;
        }
        Err(_) => {
            warn!("TTS synthesis timed out after {SYNTHESIS_TIMEOUT:?}");
            return;
        }
    };

    let samples = match synthesized {
        Ok(s) => s,
        Err(e) => {
            warn!("TTS synthesis failed: {e}");
            return;
        }
    };

    if samples.is_empty() {
        return;
    }

    // Check cancellation before WAV writing
    if let Some(ref mut rx) = cancel_rx
        && rx.try_recv().is_ok()
    {
        info!("TTS synthesis cancelled before WAV write");
        return;
    }

    let tmp_dir = session_temp_dir();
    if let Err(e) = std::fs::create_dir_all(&tmp_dir) {
        warn!("TTS: failed to create temp directory: {e}");
        return;
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let wav_path = tmp_dir.join(format!("tts_{ts}.wav"));

    // Write WAV file (sync I/O, small file — fine on blocking threadpool)
    let wav_write = tokio::task::spawn_blocking({
        let wav_path = wav_path.clone();
        move || write_wav(&wav_path, &samples, sample_rate)
    })
    .await;

    match wav_write {
        Ok(Ok(())) => {} // success
        Ok(Err(e)) => {
            warn!("TTS: failed to write WAV: {e}");
            // Remove the partial WAV file (disk full, permissions, etc.)
            if let Err(rm_err) = tokio::fs::remove_file(&wav_path).await {
                warn!("TTS: failed to remove partial WAV after write error: {rm_err}");
            }
            return;
        }
        Err(join_err) => {
            warn!("TTS: WAV write task panicked: {join_err}");
            // Best-effort cleanup: file may not exist if the panic occurred
            // before File::create, but remove_file on a nonexistent path is
            // harmless (returns Ok). Only real I/O errors are relevant here.
            if let Err(rm_err) = tokio::fs::remove_file(&wav_path).await
                && rm_err.kind() != std::io::ErrorKind::NotFound
            {
                warn!("TTS: failed to clean up WAV after write panic: {rm_err}");
            }
            return;
        }
    }

    // Spawn audio player and get a Child handle for cancellation support
    let mut child = match spawn_audio_player(&wav_path) {
        Ok(c) => c,
        Err(e) => {
            warn!("TTS: failed to start audio player: {e}");
            // Remove orphaned WAV file (playback never started)
            if let Err(rm_err) = tokio::fs::remove_file(&wav_path).await {
                warn!("TTS: failed to remove orphaned WAV after player error: {rm_err}");
            }
            return;
        }
    };

    // Play audio with cancellation support (kills child on cancel)
    if let Some(ref mut rx) = cancel_rx {
        tokio::select! {
            biased;
            _ = rx.recv() => {
                info!("TTS playback cancelled");
                let _ = child.kill().await;
                let _ = child.wait().await;
            }
            status = child.wait() => {
                if let Ok(status) = status
                    && !status.success()
                {
                    warn!("TTS: audio player exited with {status}");
                }
            }
        }
    } else {
        let status = child.wait().await;
        if let Ok(status) = status
            && !status.success()
        {
            warn!("TTS: audio player exited with {status}");
        }
    }

    // Cleanup temp file — log error instead of silently dropping
    if let Err(e) = tokio::fs::remove_file(&wav_path).await {
        warn!("TTS: failed to remove temp WAV file: {e}");
    }
}

/// Spawn the OS-native audio player and return a [`Child`] handle.
///
/// The caller is responsible for calling [`Child::wait()`] to reap the process,
/// and may call [`Child::kill()`] to interrupt playback early.
fn spawn_audio_player(path: &Path) -> Result<tokio::process::Child> {
    let path_s = path.to_string_lossy().to_string();

    // Platform-aware audio player selection (detected once, cached via LazyLock)
    #[cfg(target_os = "macos")]
    let player = "afplay";
    #[cfg(target_os = "linux")]
    let player = {
        static LINUX_PLAYER: LazyLock<&str> = LazyLock::new(|| {
            if std::process::Command::new("paplay")
                .arg("--version")
                .output()
                .is_ok()
            {
                "paplay"
            } else {
                "aplay"
            }
        });
        *LINUX_PLAYER
    };
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let player = "afplay"; // fallback

    let child = tokio::process::Command::new(player)
        .arg(&path_s)
        .spawn()
        .with_context(|| format!("Failed to launch audio player '{player}'"))?;

    Ok(child)
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
    fn test_write_wav() {
        let tmp = std::env::temp_dir().join("test_tts.wav");
        let sample_rate = 44100u32;
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0, 0.0];
        assert!(write_wav(&tmp, &samples, sample_rate).is_ok());
        assert!(tmp.exists());

        // Read back and verify RIFF header correctness
        let data = std::fs::read(&tmp).unwrap();
        assert!(
            data.starts_with(b"RIFF"),
            "WAV should start with RIFF marker"
        );
        assert!(
            data[8..12].starts_with(b"WAVE"),
            "WAV should contain WAVE format"
        );
        assert!(
            data[12..16].starts_with(b"fmt "),
            "WAV should contain fmt chunk"
        );

        // Read sample rate from header (offset 24, 4 bytes LE)
        let header_sr = u32::from_le_bytes(data[24..28].try_into().unwrap());
        assert_eq!(header_sr, sample_rate, "WAV header sample rate mismatch");

        // Read bits per sample (offset 34, 2 bytes LE)
        let bps = u16::from_le_bytes(data[34..36].try_into().unwrap());
        assert_eq!(bps, 16, "WAV should be 16-bit PCM");

        // Read number of channels (offset 22, 2 bytes LE)
        let channels = u16::from_le_bytes(data[22..24].try_into().unwrap());
        assert_eq!(channels, 1, "WAV should be mono");

        // Verify data chunk: expected size = 6 samples × 2 bytes = 12
        let data_size = u32::from_le_bytes(data[40..44].try_into().unwrap());
        assert_eq!(
            data_size, 12,
            "WAV data size should be 12 bytes for 6 16-bit samples"
        );

        std::fs::remove_file(&tmp).ok();
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
