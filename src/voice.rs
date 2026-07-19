//! Voice Assistant — wake word detection and voice command pipeline.
//!
//! # Architecture
//!
//! The voice assistant provides hands-free interaction via a custom wake word.
//! The pipeline stages are:
//!
//! 1. **Microphone capture** — capture mono 16 kHz audio via cpal
//! 2. **Voice activity detection** — energy-based VAD to gate processing
//! 3. **Mel spectrogram extraction** — via `melspectrogram.onnx` (candle-onnx)
//! 4. **Neural embedding** — via `embedding_model.onnx` (candle-onnx), 96-dim vectors
//! 5. **Wake word matching** — DTW with cosine distance against enrolled templates
//! 6. **Command recording** — record speech until silence or 30s cap
//! 7. **Transcription** — via existing Qwen3-ASR local transcriber
//! 8. **Routing** — transcribed text is routed to the user's active role via
//!    [`route_to_agent`] (falls back to the Manager if no active user is determined).
//!
//! The Assistant role manages this pipeline. It does NOT use an LLM agent loop.
//! Transcribed commands are routed to the user's currently active role (resolved
//! via [`route_to_agent`]) as if the user typed them.
//!
//! # Model files
//!
//! Two ONNX models are downloaded on first use:
//! - `melspectrogram.onnx` (~1.09 MB) — audio → mel spectrogram
//! - `embedding_model.onnx` (~1.33 MB) — mel spectrogram → 96-dim embedding
//!
//! Both from `littlebearlabs/openwakeword-features` (Apache 2.0).
//! Stored in `~/.mahbot/models/openwakeword/`.

use crate::config::CONFIG;
use crate::util::UnwrapPoison;
use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Target sample rate: 16 kHz mono.
pub const SAMPLE_RATE: u32 = 16_000;

/// Frame size for mel spectrogram (512 samples = 32ms at 16kHz).
const FRAME_LENGTH: usize = 512;

/// Hop length between frames (256 samples = 16ms at 16kHz).
const HOP_LENGTH: usize = 256;

/// Number of mel bands in the spectrogram.
const NUM_MEL_BANDS: usize = 32;

/// Embedding window: 76 consecutive mel frames (~1.2 seconds).
const EMBEDDING_WINDOW_FRAMES: usize = 76;

/// Embedding dimensionality.
const EMBEDDING_DIM: usize = 96;

/// Maximum command recording duration (30 seconds).
const MAX_RECORD_SECS: usize = 30;

/// Silence threshold for VAD (RMS below this = silence).
const VAD_THRESHOLD: f32 = 0.01;

/// Minimum silence duration before stopping command recording.
pub(crate) const SILENCE_DURATION: Duration = Duration::from_millis(1500);

/// Maximum number of download retries.
const MAX_DOWNLOAD_RETRIES: u32 = 10;

/// Expected SHA256 hashes for model files.
const MEL_MODEL_SHA256: &str = "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f";
const EMBED_MODEL_SHA256: &str = "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f";

/// Minimum voiced audio batch for ONNX inference (~128ms at 16kHz).
/// Processing audio in larger batches reduces ONNX calls from ~62/sec
/// to ~8/sec while maintaining real-time responsiveness.
const VOICE_BATCH_SIZE: usize = 2048;

/// Maximum number of recent embeddings to keep in the ring buffer.
/// With stride=8 (~89.5% overlap), each new embedding covers ~1.2s of audio
/// and arrives every ~128ms, keeping ~19 embeddings = ~2.4 seconds of context.
const EMBEDDING_RING_MAX: usize = 19;

/// Number of enrollment samples required.
const NUM_ENROLLMENT_SAMPLES: usize = 3;

/// Multiplier for auto-calibrated threshold (mean + MULTIPLIER * std).
const THRESHOLD_MULTIPLIER: f32 = 2.0;

// ── Model URLs and filenames ────────────────────────────────────────────

const MEL_MODEL_FILENAME: &str = "melspectrogram.onnx";
const MEL_MODEL_URL: &str =
    "https://huggingface.co/littlebearlabs/openwakeword-features/resolve/main/melspectrogram.onnx";
const MEL_MODEL_SIZE: u64 = 1_090_000;

const EMBED_MODEL_FILENAME: &str = "embedding_model.onnx";
const EMBED_MODEL_URL: &str =
    "https://huggingface.co/littlebearlabs/openwakeword-features/resolve/main/embedding_model.onnx";
const EMBED_MODEL_SIZE: u64 = 1_330_000;

/// Subdirectory under `~/.mahbot/models/` for voice models.
const MODEL_DIR_NAME: &str = "openwakeword";

/// Timeout for model download (5 minutes for ~2.4 MB total).
const MODEL_DOWNLOAD_TIMEOUT: Duration = Duration::from_mins(5);

// ═══════════════════════════════════════════════════════════════════════════
// Model loading state machine
// ═══════════════════════════════════════════════════════════════════════════

/// Model loading state with type-safe atomic access.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
enum ModelState {
    Uninit = 0,
    Loading = 1,
    Ready = 2,
    Failed = 3,
}

/// Atomic wrapper around [`ModelState`] that provides lock-free access.
struct AtomicModelState(AtomicU8);

impl AtomicModelState {
    const fn new(state: ModelState) -> Self {
        Self(AtomicU8::new(state as u8))
    }

    fn load(&self, order: Ordering) -> ModelState {
        match self.0.load(order) {
            1 => ModelState::Loading,
            2 => ModelState::Ready,
            3 => ModelState::Failed,
            _ => ModelState::Uninit,
        }
    }

    fn store(&self, state: ModelState, order: Ordering) {
        self.0.store(state as u8, order);
    }
}

static MODELS_STATE: AtomicModelState = AtomicModelState::new(ModelState::Uninit);

fn model_dir() -> Option<PathBuf> {
    let root = CONFIG.try_storage_root()?;
    Some(root.join("models").join(MODEL_DIR_NAME))
}

/// Check whether voice models are ready for inference.
pub fn models_ready() -> bool {
    MODELS_STATE.load(Ordering::Acquire) == ModelState::Ready
}

/// Check whether voice models are currently loading.
pub fn models_loading() -> bool {
    MODELS_STATE.load(Ordering::Acquire) == ModelState::Loading
}

// ═══════════════════════════════════════════════════════════════════════════
// Voice pipeline status (shared between pipeline task and GUI)
// ═══════════════════════════════════════════════════════════════════════════

/// Voice pipeline status.
#[derive(Debug, Clone)]
pub enum VoiceStatus {
    Disabled,
    LoadingModels,
    ModelError,
    Listening,
    Recording,
    Transcribing,
    MicPermissionDenied,
    MicDisconnected,
    Enrolling {
        sample: usize,
        total: usize,
    },
    /// Actively capturing speech during enrollment.
    ListeningDuringEnrollment {
        sample: usize,
        total: usize,
    },
    /// Speech detected, waiting for silence to confirm utterance end.
    WaitingForSilenceDuringEnrollment {
        sample: usize,
        total: usize,
    },
    Enrolled,
    Error(String),
}

/// A single enrollment template.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WakeWordTemplate {
    pub name: String,
    pub embeddings: Vec<Vec<f32>>,
    pub threshold: f32,
}

/// Collection of enrolled wake word templates.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct WakeWordTemplates {
    pub templates: Vec<WakeWordTemplate>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Global state
// ═══════════════════════════════════════════════════════════════════════════

static VOICE_PIPELINE: OnceLock<RwLock<VoicePipelineState>> = OnceLock::new();

/// The name of the currently active workspace, updated by the GUI when the
/// user switches workspaces. Used by [`route_to_agent`] to route transcribed
/// commands to the correct workspace.
static LAST_ACTIVE_WORKSPACE: OnceLock<RwLock<String>> = OnceLock::new();

/// The name of the currently active user, updated by the GUI when the
/// selected user changes. Used by [`route_to_agent`] to route transcribed
/// voice commands to the correct user's active role.
static LAST_ACTIVE_USER: OnceLock<RwLock<String>> = OnceLock::new();

/// Set the currently active workspace name (called from GUI on workspace switch).
pub fn set_active_workspace_name(name: &str) {
    if let Some(state) = LAST_ACTIVE_WORKSPACE.get() {
        *state.write().unwrap_poison() = name.to_string();
    }
}

/// Set the currently active user name (called from GUI on user switch).
pub fn set_active_user_name(name: &str) {
    if let Some(state) = LAST_ACTIVE_USER.get() {
        *state.write().unwrap_poison() = name.to_string();
    }
}

fn active_workspace_name() -> String {
    LAST_ACTIVE_WORKSPACE
        .get()
        .map(|s| s.read().unwrap_poison().clone())
        .unwrap_or_default()
}

fn active_user_name() -> String {
    LAST_ACTIVE_USER
        .get()
        .map(|s| s.read().unwrap_poison().clone())
        .unwrap_or_default()
}

struct VoicePipelineState {
    enabled: bool,
    status: VoiceStatus,
    templates: Arc<WakeWordTemplates>,
    enrollment_buffer: Vec<Vec<Vec<f32>>>,
    cmd_tx: Option<mpsc::UnboundedSender<VoiceCommand>>,
}

#[derive(Debug)]
pub enum VoiceCommand {
    StartListening,
    StopListening,
    StartEnrollment,
    CancelEnrollment,
    Shutdown,
}

fn voice_state() -> &'static RwLock<VoicePipelineState> {
    VOICE_PIPELINE.get().expect("VoicePipeline not initialized")
}

/// Initialize the voice pipeline state. Called during startup.
pub fn init_global() -> Result<()> {
    LAST_ACTIVE_WORKSPACE
        .set(RwLock::new(String::new()))
        .map_err(|_| anyhow!("LAST_ACTIVE_WORKSPACE already initialized"))?;
    LAST_ACTIVE_USER
        .set(RwLock::new(String::new()))
        .map_err(|_| anyhow!("LAST_ACTIVE_USER already initialized"))?;

    VOICE_PIPELINE
        .set(RwLock::new(VoicePipelineState {
            enabled: false,
            status: VoiceStatus::Disabled,
            templates: Arc::new(WakeWordTemplates::default()),
            enrollment_buffer: Vec::new(),
            cmd_tx: None,
        }))
        .map_err(|_| anyhow!("VoicePipeline already initialized"))?;

    Ok(())
}

#[must_use]
pub fn get_status() -> VoiceStatus {
    voice_state().read().unwrap_poison().status.clone()
}

#[must_use]
pub fn is_enabled() -> bool {
    voice_state().read().unwrap_poison().enabled
}

pub fn set_enabled(enabled: bool) {
    let mut state = voice_state().write().unwrap_poison();
    state.enabled = enabled;
    if !enabled {
        state.status = VoiceStatus::Disabled;
    }
}

pub fn set_status(status: VoiceStatus) {
    voice_state().write().unwrap_poison().status = status;
}

#[must_use]
pub fn get_templates() -> Arc<WakeWordTemplates> {
    voice_state().read().unwrap_poison().templates.clone()
}

pub fn set_templates(templates: Arc<WakeWordTemplates>) {
    voice_state().write().unwrap_poison().templates = templates;
}

pub fn send_command(cmd: VoiceCommand) {
    if let Some(tx) = &voice_state().read().unwrap_poison().cmd_tx {
        let _ = tx.send(cmd);
    } else {
        warn!("Voice pipeline not initialized — dropping command {cmd:?}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ONNX model loading and execution
// ═══════════════════════════════════════════════════════════════════════════

struct OnnxModels {
    mel_model: candle_onnx::onnx::ModelProto,
    embed_model: candle_onnx::onnx::ModelProto,
    device: candle_core::Device,
}

static ONNX_MODELS: OnceLock<OnnxModels> = OnceLock::new();

fn load_onnx_models(dir: &Path) -> Result<OnnxModels> {
    let mel_path = dir.join(MEL_MODEL_FILENAME);
    let embed_path = dir.join(EMBED_MODEL_FILENAME);

    if !mel_path.exists() {
        anyhow::bail!("Mel spectrogram model not found: {}", mel_path.display());
    }
    if !embed_path.exists() {
        anyhow::bail!("Embedding model not found: {}", embed_path.display());
    }

    let mel_model =
        candle_onnx::read_file(mel_path).context("Failed to load mel spectrogram ONNX model")?;
    let embed_model =
        candle_onnx::read_file(embed_path).context("Failed to load embedding ONNX model")?;

    Ok(OnnxModels {
        mel_model,
        embed_model,
        device: candle_core::Device::Cpu,
    })
}

/// Scale audio samples from float [-1, 1] range to approximate int16 range
/// using the 32768.0 multiplier from the OpenWakeWord reference.
///
/// This is what the mel model was trained with — changing to the exact int16
/// max (32767.0) would shift the numerical values and degrade model accuracy.
/// The slight offset (1.0 → 32768.0, 1 LSB above int16 max) is intentional.
fn scale_to_int16_range(samples: &[f32]) -> Vec<f32> {
    samples.iter().map(|s| s * 32768.0).collect()
}

/// Compute mel spectrogram frames from raw audio samples.
fn compute_mel_spectrogram(models: &OnnxModels, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
    use candle_core::Tensor;

    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let sample_len = samples.len();
    let scaled = scale_to_int16_range(samples);
    let input_tensor = Tensor::from_slice(&scaled, (1, sample_len), &models.device)?;

    let input_name = models
        .mel_model
        .graph
        .as_ref()
        .and_then(|g| g.input.first())
        .map_or_else(|| "input".to_string(), |i| i.name.clone());

    let mut inputs = HashMap::new();
    inputs.insert(input_name, input_tensor);

    let mut outputs = candle_onnx::simple_eval(&models.mel_model, inputs)
        .context("Mel spectrogram inference failed")?;

    let output_name = models
        .mel_model
        .graph
        .as_ref()
        .and_then(|g| g.output.first())
        .map_or_else(|| "output".to_string(), |o| o.name.clone());

    let output_tensor = outputs
        .remove(&output_name)
        .context("Mel spectrogram model produced no output")?;

    let shape = output_tensor.dims();
    debug!("Mel spectrogram output shape: {shape:?}");

    let (num_frames, num_features) = if shape.len() == 3 {
        if shape[2] as usize == NUM_MEL_BANDS {
            (shape[1] as usize, shape[2] as usize)
        } else if shape[1] as usize == NUM_MEL_BANDS {
            (shape[2] as usize, shape[1] as usize)
        } else {
            anyhow::bail!("Unexpected mel shape: {shape:?} (expected {NUM_MEL_BANDS} bands)")
        }
    } else if shape.len() == 4
        && shape[0] == 1
        && shape[1] == 1
        && shape[3] as usize == NUM_MEL_BANDS
    {
        // 4D NHWC output: (1, 1, num_frames, NUM_MEL_BANDS) — squeeze batch and channel dims.
        (shape[2] as usize, shape[3] as usize)
    } else {
        anyhow::bail!("Unexpected mel output shape: {shape:?}");
    };

    let output_data: Vec<f32> = output_tensor.flatten_all()?.to_vec1()?;

    let mut frames = Vec::with_capacity(num_frames);
    for f in 0..num_frames {
        let start = f * num_features;
        if start + num_features <= output_data.len() {
            frames.push(output_data[start..start + num_features].to_vec());
        }
    }

    Ok(frames)
}

/// Compute embedding from 76 mel frames.
fn compute_embedding(models: &OnnxModels, mel_frames: &[Vec<f32>]) -> Result<Vec<f32>> {
    use candle_core::Tensor;

    if mel_frames.len() != EMBEDDING_WINDOW_FRAMES {
        anyhow::bail!(
            "Expected {} mel frames for embedding, got {}",
            EMBEDDING_WINDOW_FRAMES,
            mel_frames.len()
        );
    }

    for (i, frame) in mel_frames.iter().enumerate() {
        if frame.len() != NUM_MEL_BANDS {
            anyhow::bail!(
                "Mel frame {i} has {} bands, expected {NUM_MEL_BANDS}",
                frame.len()
            );
        }
    }

    let flat: Vec<f32> = mel_frames.iter().flatten().copied().collect();
    // ONNX model declares 4D NHWC input (1, EMBEDDING_WINDOW_FRAMES, NUM_MEL_BANDS, 1).
    // candle_onnx::simple_eval performs strict rank validation and requires the
    // tensor rank to match the model declaration.
    let input_tensor = Tensor::from_slice(
        &flat,
        (1, EMBEDDING_WINDOW_FRAMES, NUM_MEL_BANDS, 1),
        &models.device,
    )?;

    let input_name = models
        .embed_model
        .graph
        .as_ref()
        .and_then(|g| g.input.first())
        .map_or_else(|| "input".to_string(), |i| i.name.clone());

    let mut inputs = HashMap::new();
    inputs.insert(input_name, input_tensor);

    let mut outputs = candle_onnx::simple_eval(&models.embed_model, inputs)
        .context("Embedding model inference failed")?;

    let output_name = models
        .embed_model
        .graph
        .as_ref()
        .and_then(|g| g.output.first())
        .map_or_else(|| "output".to_string(), |o| o.name.clone());

    let output_tensor = outputs
        .remove(&output_name)
        .context("Embedding model produced no output")?;

    let embedding: Vec<f32> = output_tensor.flatten_all()?.to_vec1()?;

    if embedding.len() != EMBEDDING_DIM {
        warn!(
            "Embedding model produced {} dimensions, expected {EMBEDDING_DIM}",
            embedding.len()
        );
    }

    Ok(embedding)
}

/// Extract a sequence of embeddings from raw audio by processing sliding windows.
fn extract_embeddings_from_audio(models: &OnnxModels, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
    let mel_frames = compute_mel_spectrogram(models, samples)?;

    if mel_frames.len() < EMBEDDING_WINDOW_FRAMES {
        anyhow::bail!(
            "Audio too short: got {} mel frames, need at least {}",
            mel_frames.len(),
            EMBEDDING_WINDOW_FRAMES
        );
    }

    let mut embeddings = Vec::new();
    let stride: usize = 8; // OpenWakeWord reference uses stride=8 (~89.5% overlap)

    let mut start = 0;
    while start + EMBEDDING_WINDOW_FRAMES <= mel_frames.len() {
        let window = &mel_frames[start..start + EMBEDDING_WINDOW_FRAMES];
        match compute_embedding(models, window) {
            Ok(emb) => embeddings.push(emb),
            Err(e) => debug!("Skipping embedding window: {e}"),
        }
        start += stride;
    }

    if embeddings.is_empty() {
        anyhow::bail!("No embeddings could be extracted from audio");
    }

    Ok(embeddings)
}

// ═══════════════════════════════════════════════════════════════════════════
// DTW matching with cosine distance
// ═══════════════════════════════════════════════════════════════════════════

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

fn dtw_distance(live: &[Vec<f32>], template: &[Vec<f32>]) -> f32 {
    if live.is_empty() || template.is_empty() {
        return f32::MAX;
    }

    let m = template.len();
    let mut prev = vec![f32::MAX; m];
    let mut curr = vec![f32::MAX; m];

    for (i, live_i) in live.iter().enumerate() {
        for (j, tpl_j) in template.iter().enumerate() {
            let cost = cosine_distance(live_i, tpl_j);
            if i == 0 && j == 0 {
                curr[j] = cost;
            } else if i == 0 {
                curr[j] = cost + curr[j - 1];
            } else if j == 0 {
                curr[j] = cost + prev[j];
            } else {
                curr[j] = cost + prev[j].min(prev[j - 1].min(curr[j - 1]));
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m - 1]
}

/// Match a live embedding sequence against all enrolled templates.
fn match_against_templates<'a>(
    live_sequence: &[Vec<f32>],
    templates: &'a WakeWordTemplates,
) -> Option<(f32, &'a WakeWordTemplate)> {
    let mut best: Option<(f32, &WakeWordTemplate)> = None;

    for tpl in &templates.templates {
        let dist = dtw_distance(live_sequence, &tpl.embeddings);
        if dist < tpl.threshold {
            let is_better = best.as_ref().is_none_or(|(best_dist, _)| dist < *best_dist);
            if is_better {
                best = Some((dist, tpl));
            }
        }
    }

    best
}

// ═══════════════════════════════════════════════════════════════════════════
// Audio utilities
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::cast_precision_loss)]
fn is_speech(samples: &[f32], threshold: f32) -> bool {
    if samples.is_empty() {
        return false;
    }
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    rms > threshold
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let output_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;
        if src_idx + 1 < samples.len() {
            output.push(
                (f64::from(samples[src_idx]) * (1.0 - frac)
                    + f64::from(samples[src_idx + 1]) * frac) as f32,
            );
        } else if src_idx < samples.len() {
            output.push(samples[src_idx]);
        } else {
            output.push(0.0);
        }
    }

    output
}

fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    let frames = samples.len() / ch;
    let remainder = samples.len() % ch;
    if remainder != 0 {
        warn!(
            "to_mono: discarding {remainder} sample(s) from non-aligned audio (channels={channels})",
        );
    }
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        let start = f * ch;
        let sum: f32 = samples[start..start + ch].iter().sum();
        mono.push(sum / f32::from(channels));
    }
    mono
}

#[allow(clippy::cast_possible_truncation)]
fn samples_to_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let header_size = 44;
    let data_size = samples.len() * 2;
    let total_size = header_size + data_size;
    let mut wav = Vec::with_capacity(total_size);

    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(total_size as u32 - 8).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data_size as u32).to_le_bytes());

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let int_sample = (clamped * f32::from(i16::MAX)) as i16;
        wav.extend_from_slice(&int_sample.to_le_bytes());
    }

    wav
}

// ═══════════════════════════════════════════════════════════════════════════
// Model download
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::cast_precision_loss)]
async fn download_model(
    url: &str,
    path: &Path,
    expected_size: u64,
    expected_hash: &str,
) -> Result<()> {
    let response = reqwest::Client::new()
        .get(url)
        .timeout(MODEL_DOWNLOAD_TIMEOUT)
        .send()
        .await
        .context("Failed to start model download")?;

    let bytes = response
        .bytes()
        .await
        .context("Failed to download model file")?;

    if bytes.len() < 1000 {
        anyhow::bail!("Downloaded model file is too small: {} bytes", bytes.len());
    }

    // Validate against expected size (allow 5% tolerance for minor variations)
    let size = bytes.len() as u64;
    let min_size = expected_size * 95 / 100;
    let max_size = expected_size * 105 / 100;
    if size < min_size || size > max_size {
        anyhow::bail!(
            "Downloaded model size mismatch: got {size} bytes, expected ~{expected_size} bytes",
        );
    }

    let tmp_path = path.with_extension("tmp");
    {
        let mut file = File::create(&tmp_path)?;
        file.write_all(&bytes)?;
        file.flush()?;
    }

    // Verify hash BEFORE renaming to final path. If verification fails the
    // .tmp file remains (it will be overwritten on retry) rather than leaving
    // a corrupt file at the final path that passes the exists() check.
    if !expected_hash.is_empty() {
        verify_sha256(&tmp_path, expected_hash)
            .with_context(|| format!("SHA256 verification failed for {}", path.display()))?;
    }

    std::fs::rename(&tmp_path, path)?;

    info!(
        "Downloaded {} ({:.1} MB)",
        path.display(),
        bytes.len() as f64 / 1_048_576.0
    );
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

/// Verify a file's SHA256 hash matches the expected value.
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

async fn ensure_models_downloaded() -> Result<PathBuf> {
    let dir = model_dir()
        .ok_or_else(|| anyhow!("Cannot resolve model directory (storage root not set)"))?;

    tokio::fs::create_dir_all(&dir).await?;

    let mel_path = dir.join(MEL_MODEL_FILENAME);
    if mel_path.exists() {
        verify_sha256(&mel_path, MEL_MODEL_SHA256)?;
    } else {
        info!("Downloading mel spectrogram model...");
        download_model(MEL_MODEL_URL, &mel_path, MEL_MODEL_SIZE, MEL_MODEL_SHA256).await?;
    }

    let embed_path = dir.join(EMBED_MODEL_FILENAME);
    if embed_path.exists() {
        verify_sha256(&embed_path, EMBED_MODEL_SHA256)?;
    } else {
        info!("Downloading embedding model...");
        download_model(
            EMBED_MODEL_URL,
            &embed_path,
            EMBED_MODEL_SIZE,
            EMBED_MODEL_SHA256,
        )
        .await?;
    }

    Ok(dir)
}

async fn download_retry_loop() {
    let Some(dir) = model_dir() else {
        warn!("Voice models: cannot resolve model directory");
        MODELS_STATE.store(ModelState::Failed, Ordering::Release);
        return;
    };

    let mut retry_delay = Duration::from_secs(5);
    let mut retry_count = 0u32;

    loop {
        if MODELS_STATE.load(Ordering::Acquire) >= ModelState::Ready {
            return;
        }

        retry_count += 1;
        if retry_count > MAX_DOWNLOAD_RETRIES {
            warn!("Voice model download failed after {MAX_DOWNLOAD_RETRIES} retries");
            MODELS_STATE.store(ModelState::Failed, Ordering::Release);
            set_status(VoiceStatus::ModelError);
            return;
        }

        match tokio::time::timeout(MODEL_DOWNLOAD_TIMEOUT, ensure_models_downloaded()).await {
            Ok(Ok(_)) => match load_onnx_models(&dir) {
                Ok(models) => {
                    if ONNX_MODELS.set(models).is_ok() {
                        MODELS_STATE.store(ModelState::Ready, Ordering::Release);
                        info!("Voice models loaded successfully");
                        // Clear "Loading models" status — if enabled, auto-start
                        // transitions to Listening on the next pipeline tick.
                        set_status(if is_enabled() {
                            VoiceStatus::Listening
                        } else {
                            VoiceStatus::Disabled
                        });
                        return;
                    }
                }
                Err(e) => warn!("Failed to load voice models (will retry): {e}"),
            },
            Ok(Err(e)) => warn!("Failed to download voice models (will retry): {e}"),
            Err(_) => warn!("Voice model download timed out (will retry)"),
        }

        if MODELS_STATE.load(Ordering::Acquire) >= ModelState::Failed {
            return;
        }

        tokio::time::sleep(retry_delay).await;
        retry_delay = (retry_delay * 2).min(Duration::from_mins(2));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Microphone capture
// ═══════════════════════════════════════════════════════════════════════════

/// Convert audio data to mono and resample if needed, then send to pipeline.
fn send_audio_to_pipeline(
    tx: &mpsc::UnboundedSender<Vec<f32>>,
    float_data: &[f32],
    channels: u16,
    sample_rate: u32,
) {
    let mono = to_mono(float_data, channels);
    let resampled = if sample_rate == SAMPLE_RATE {
        mono
    } else {
        resample_audio(&mono, sample_rate, SAMPLE_RATE)
    };
    let _ = tx.send(resampled);
}

fn start_microphone() -> Result<(mpsc::UnboundedReceiver<Vec<f32>>, cpal::Stream)> {
    let (tx, rx) = mpsc::unbounded_channel::<Vec<f32>>();

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("No default input device found"))?;

    let config = device
        .default_input_config()
        .context("Failed to get default input config")?;

    info!(
        "Microphone: {} ({:?}, {} Hz, {} ch)",
        device.name().unwrap_or_else(|_| "unknown".to_string()),
        config.sample_format(),
        config.sample_rate().0,
        config.channels()
    );

    // Error callback for microphone stream — must be a function pointer
    // (not a closure) so it can be used in multiple build_input_stream calls.
    #[allow(clippy::needless_pass_by_value, clippy::items_after_statements)]
    fn mic_error(err: cpal::StreamError) {
        error!("Microphone stream error: {err}");
    }

    let sample_rate = config.sample_rate().0;
    let channels = config.channels();
    let sample_tx = Arc::new(tx);

    // Helper to build audio stream for integer sample formats that need
    // conversion to f32. The F32 case is handled separately since it can
    // pass data directly without conversion.
    macro_rules! build_int_stream {
        ($device:expr, $config:expr, $sample_tx:expr, $channels:expr, $sample_rate:expr, $fmt:ty, $convert:expr) => {{
            let tx = $sample_tx.clone();
            $device.build_input_stream::<$fmt, _, _>(
                &($config).into(),
                move |data, _| {
                    let float_data: Vec<f32> = data.iter().map($convert).collect();
                    send_audio_to_pipeline(&tx, &float_data, $channels, $sample_rate);
                },
                mic_error,
                None,
            )
        }};
    }

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            // f32 samples can be passed directly — no conversion needed
            let tx = sample_tx.clone();
            device.build_input_stream::<f32, _, _>(
                &config.into(),
                move |data, _| {
                    send_audio_to_pipeline(&tx, data, channels, sample_rate);
                },
                mic_error,
                None,
            )
        }
        cpal::SampleFormat::I16 => {
            build_int_stream!(
                device,
                config,
                sample_tx,
                channels,
                sample_rate,
                i16,
                |&s| f32::from(s) / f32::from(i16::MAX)
            )
        }
        cpal::SampleFormat::U16 => {
            build_int_stream!(
                device,
                config,
                sample_tx,
                channels,
                sample_rate,
                u16,
                |&s| (f32::from(s) / f32::from(u16::MAX)) * 2.0 - 1.0
            )
        }
        _ => anyhow::bail!("Unsupported sample format: {:?}", config.sample_format()),
    }
    .context("Failed to build microphone input stream")?;

    stream.play().context("Failed to start microphone stream")?;

    info!(
        "Microphone listening started ({} Hz, {} channels)",
        sample_rate, channels
    );
    Ok((rx, stream))
}

// ═══════════════════════════════════════════════════════════════════════════
// Transcription via existing Qwen3-ASR
// ═══════════════════════════════════════════════════════════════════════════

async fn transcribe_audio(samples: &[f32]) -> Result<String> {
    let wav_bytes = samples_to_wav(samples, SAMPLE_RATE);
    let tmp_dir = std::env::temp_dir().join("mahbot_voice");
    tokio::fs::create_dir_all(&tmp_dir).await?;
    let tmp_path = tmp_dir.join(format!("cmd_{}.wav", crate::generate_id()));
    tokio::fs::write(&tmp_path, &wav_bytes).await?;

    let result = crate::providers::local_transcriber::transcribe_file_async(&tmp_path).await;

    if let Err(e) = tokio::fs::remove_file(&tmp_path).await {
        warn!("Failed to remove temp transcription file: {e}");
    }
    if let Err(e) = tokio::fs::remove_dir(&tmp_dir).await {
        // remove_dir fails with ENOTEMPTY if there are leftover files —
        // log the issue but don't fail the transcription result.
        warn!("Failed to remove temp transcription directory: {e}");
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Enrollment helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Process raw audio samples into embedding sequences (for enrollment).
pub fn process_enrollment_sample(samples: &[f32]) -> Result<Vec<Vec<f32>>> {
    let models = ONNX_MODELS
        .get()
        .ok_or_else(|| anyhow!("Voice models not loaded"))?;
    extract_embeddings_from_audio(models, samples)
}

/// Finalize enrollment: compute threshold from pairwise DTW distances.
#[allow(clippy::cast_precision_loss)]
fn finalize_enrollment(wake_word_name: &str) -> Result<WakeWordTemplate> {
    let state = voice_state().read().unwrap_poison();
    let samples = &state.enrollment_buffer;

    if samples.len() < 2 {
        anyhow::bail!("Need at least 2 enrollment samples, got {}", samples.len());
    }

    let mut distances = Vec::new();
    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            let d1 = dtw_distance(&samples[i], &samples[j]);
            let d2 = dtw_distance(&samples[j], &samples[i]);
            distances.push(d1.min(d2));
        }
    }

    if distances.is_empty() {
        anyhow::bail!("Could not compute pairwise distances");
    }

    let mean: f32 = distances.iter().sum::<f32>() / distances.len() as f32;
    let variance: f32 =
        distances.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / distances.len() as f32;
    let std_dev = variance.sqrt();
    let threshold = mean + THRESHOLD_MULTIPLIER * std_dev;

    info!("Enrollment calibration: mean={mean:.4}, std={std_dev:.4}, threshold={threshold:.4}");

    // Concatenate all embeddings from all enrollment samples into one template.
    // This gives the DTW matcher more reference data to match against.
    let mut all_embeddings = Vec::new();
    for sample_embeddings in samples {
        all_embeddings.extend(sample_embeddings.iter().cloned());
    }
    info!(
        "Enrollment: {} samples → {} embedding vectors",
        samples.len(),
        all_embeddings.len()
    );

    Ok(WakeWordTemplate {
        name: wake_word_name.to_string(),
        embeddings: all_embeddings,
        threshold,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Routing to active agent
// ═══════════════════════════════════════════════════════════════════════════

/// Route a transcribed voice command to the appropriate agent.
///
/// Resolves the active user's role and computes the deterministic agent ID,
/// then routes through the agent-ID message router.
///
/// Falls back to the Manager router if no active user can be determined.
async fn route_to_agent(text: String) {
    // Try active user first (set by GUI on user switch)
    let user_name = active_user_name();
    if !user_name.is_empty() {
        let role = crate::users::resolve_active_role(&user_name).await;
        let active_ws = active_workspace_name();
        let ws_name = if active_ws.is_empty() {
            // Fallback to user's configured workspace
            if let Ok(Some(ws)) = crate::users::get_workspace(&user_name).await {
                ws.name
            } else {
                let path = crate::users::personal_workspace_path(&user_name);
                crate::users::personal_workspace_struct(&user_name, &path).name
            }
        } else {
            active_ws
        };

        info!("Voice command -> {role} (user: {user_name}, workspace: {ws_name}): {text}",);

        let agent_id =
            crate::session::resolve_agent_id("voice", &user_name, role.as_str(), &ws_name);
        crate::message_router::route(
            &agent_id,
            crate::message_router::AgentJob {
                content: text,
                workspace_name: ws_name,
                user_name,
                channel: "voice".to_string(),
                kind: crate::message_router::JobKind::UserMessage,
                role,
                reply_target: None,
            },
        );
        return;
    }

    // Fallback to active workspace -> Manager (current behavior)
    let active = active_workspace_name();
    if !active.is_empty() {
        info!("Voice command -> Manager (active workspace: {active}): {text}");
        let agent_id = crate::session::manager_agent_id(&active);
        crate::message_router::route(
            &agent_id,
            crate::message_router::AgentJob {
                content: text,
                workspace_name: active,
                user_name: String::new(),
                channel: String::new(),
                kind: crate::message_router::JobKind::UserMessage,
                role: crate::Role::Manager,
                reply_target: None,
            },
        );
        return;
    }

    // Fallback to admin's configured workspace
    let ws = match crate::users::get_workspace("admin").await {
        Ok(Some(ws)) => ws,
        Ok(None) => {
            let path = crate::users::personal_workspace_path("admin");
            crate::users::personal_workspace_struct("admin", &path)
        }
        Err(e) => {
            warn!("Failed to get admin workspace: {e}; using personal workspace");
            let path = crate::users::personal_workspace_path("admin");
            crate::users::personal_workspace_struct("admin", &path)
        }
    };

    info!(
        "Voice command -> Manager (workspace: {}): {}",
        ws.name, text
    );
    let agent_id = crate::session::manager_agent_id(&ws.name);
    crate::message_router::route(
        &agent_id,
        crate::message_router::AgentJob {
            content: text,
            workspace_name: ws.name,
            user_name: String::new(),
            channel: String::new(),
            kind: crate::message_router::JobKind::UserMessage,
            role: crate::Role::Manager,
            reply_target: None,
        },
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Voice pipeline background task
// ═══════════════════════════════════════════════════════════════════════════

/// Safe wrapper around `Option<cpal::Stream>` to ensure `Send` on macOS.
///
/// `cpal::Stream` is conservatively marked `!Send` on macOS because of
/// `NotSendSyncAcrossAllPlatforms(PhantomData<*mut ()>)`, but the
/// underlying CoreAudio handles are actually thread-safe. We use
/// `unsafe impl Send` to assert this (a common pattern in cpal usage).
/// # Safety
///
/// `cpal::Stream` is conservatively marked `!Send` on macOS because of
/// `NotSendSyncAcrossAllPlatforms(PhantomData<*mut ()>)`, but the
/// underlying CoreAudio handles are actually thread-safe across send
/// boundaries. The CoreAudio AudioUnit and audio queue can be stopped
/// and dropped from any thread. The property listener callback
/// (`AudioObjectPropertyListener`) uses a `Box<dyn FnMut()` internally,
/// but this callback is only invoked from the CoreAudio event thread
/// while the stream is running, and the stream is always dropped from
/// the same async runtime that created it. Cross-thread moves only
/// happen when the future is passed between tokio tasks (e.g. via
/// `spawn_cancellable`), which always happens before the stream is
/// started or after it is stopped. This is a well-known pattern in the
/// cpal ecosystem — many audio applications use `unsafe impl Send`
/// for `cpal::Stream` on macOS with this justification.
#[derive(Default)]
struct SendMicStream(Option<cpal::Stream>);

impl SendMicStream {
    fn take(&mut self) -> Option<cpal::Stream> {
        self.0.take()
    }

    fn set(&mut self, stream: cpal::Stream) {
        self.0 = Some(stream);
    }
}

unsafe impl Send for SendMicStream {}

/// Runtime state for the voice pipeline main loop.
#[allow(clippy::struct_excessive_bools)]
struct PipelineCtx {
    mic_rx: Option<mpsc::UnboundedReceiver<Vec<f32>>>,
    mic_stream: SendMicStream,
    is_listening: bool,
    is_recording: bool,
    command_buffer: Vec<f32>,
    silence_since: Option<Instant>,
    enrollment_mode: bool,
    utterance_buf: Vec<f32>,
    utterance_had_speech: bool,
    utterance_silence_since: Option<Instant>,
    audio_buffer: Vec<f32>,
    mel_frame_buffer: Vec<Vec<f32>>,
    embedding_ring: Vec<Vec<f32>>,
    voice_batch: Vec<f32>,
    enrollment_pending: Option<Vec<f32>>,
    auto_start_pending: bool,
}

impl PipelineCtx {
    fn new() -> Self {
        Self {
            mic_rx: None,
            mic_stream: SendMicStream::default(),
            is_listening: false,
            is_recording: false,
            command_buffer: Vec::new(),
            silence_since: None,
            enrollment_mode: false,
            utterance_buf: Vec::new(),
            utterance_had_speech: false,
            utterance_silence_since: None,
            audio_buffer: Vec::new(),
            mel_frame_buffer: Vec::new(),
            embedding_ring: Vec::new(),
            voice_batch: Vec::new(),
            enrollment_pending: None,
            auto_start_pending: CONFIG.voice_enabled().as_deref() == Some("true"),
        }
    }

    fn handle_start_listening(&mut self) {
        // Defense-in-depth: reject if voice has been disabled between the
        // time the command was sent and the time it's processed. This
        // mirrors the guard in handle_start_enrollment.
        if !is_enabled() {
            self.auto_start_pending = false;
            warn!("Ignoring start_listening — voice assistant is disabled");
            return;
        }
        if !models_ready() {
            // Models are still loading — mark pending so check_auto_start
            // retries when they become ready (satisfies ticket req #2:
            // auto-start when models transition to Ready). This is NOT set
            // on mic failure, preventing a continuous retry loop.
            self.auto_start_pending = true;
            warn!("Voice models not ready yet");
            return;
        }
        if !self.is_listening {
            drop(self.mic_stream.take());
            match start_microphone() {
                Ok((rx, stream)) => {
                    self.mic_rx = Some(rx);
                    self.mic_stream.set(stream);
                    self.is_listening = true;
                    self.audio_buffer.clear();
                    self.mel_frame_buffer.clear();
                    self.embedding_ring.clear();
                    set_status(VoiceStatus::Listening);
                    info!("Voice pipeline: started listening");
                }
                Err(e) => {
                    warn!("Failed to start microphone: {e}");
                    set_status(VoiceStatus::MicDisconnected);
                    // auto_start_pending is NOT set here — the user must
                    // re-toggle Voice OFF/ON to retry after resolving the
                    // mic issue.
                }
            }
        }
    }

    fn handle_stop_listening(&mut self) {
        self.is_listening = false;
        self.is_recording = false;
        self.enrollment_mode = false;
        self.auto_start_pending = false;
        self.utterance_buf.clear();
        self.utterance_had_speech = false;
        self.utterance_silence_since = None;
        drop(self.mic_stream.take());
        self.mic_rx = None;
        set_status(VoiceStatus::Disabled);
        info!("Voice pipeline: stopped listening");
    }

    fn handle_start_enrollment(&mut self) {
        if !self.is_listening {
            warn!("Cannot start enrollment: microphone not running");
            set_status(VoiceStatus::Error(
                "Microphone not running — enable Voice first".to_string(),
            ));
            return;
        }
        self.enrollment_mode = true;
        self.audio_buffer.clear();
        self.utterance_buf.clear();
        self.utterance_had_speech = false;
        self.utterance_silence_since = None;
        voice_state()
            .write()
            .unwrap_poison()
            .enrollment_buffer
            .clear();
        set_status(VoiceStatus::Enrolling {
            sample: 0,
            total: NUM_ENROLLMENT_SAMPLES,
        });
        info!("Voice pipeline: enrollment started");
    }

    fn handle_cancel_enrollment(&mut self) {
        self.enrollment_mode = false;
        self.utterance_buf.clear();
        self.utterance_had_speech = false;
        self.utterance_silence_since = None;
        voice_state()
            .write()
            .unwrap_poison()
            .enrollment_buffer
            .clear();
        set_status(if self.is_listening {
            VoiceStatus::Listening
        } else {
            VoiceStatus::Disabled
        });
        info!("Voice pipeline: enrollment cancelled");
    }

    fn handle_shutdown(&mut self) {
        drop(self.mic_stream.take());
    }

    fn check_auto_start(&mut self) {
        // One-shot retry: only fires when auto_start_pending is true (set at
        // pipeline creation or by handle_start_listening when models weren't
        // ready yet). Cleared after the first attempt — no continuous retry
        // loop on mic failure.
        if models_ready() && !self.is_listening && self.auto_start_pending {
            self.auto_start_pending = false;
            send_command(VoiceCommand::StartListening);
        }
    }
}

/// Run the voice pipeline background task.
#[allow(clippy::too_many_lines)]
pub async fn run_voice_pipeline() {
    info!("Voice pipeline starting...");

    let shutdown_token = crate::shutdown::shutdown_token();
    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<VoiceCommand>();

    {
        let mut state = voice_state().write().unwrap_poison();
        state.cmd_tx = Some(cmd_tx);
    }

    // Load persisted templates from config on startup
    if let Some(json) = CONFIG.wake_word_templates()
        && let Ok(templates) = serde_json::from_str::<WakeWordTemplates>(&json)
    {
        set_templates(Arc::new(templates));
        info!(
            "Loaded {} wake word template(s) from config",
            get_templates().templates.len()
        );
    }

    // Start model download in background
    MODELS_STATE.store(ModelState::Loading, Ordering::Release);
    set_status(VoiceStatus::LoadingModels);
    tokio::spawn(download_retry_loop());

    let mut ctx = PipelineCtx::new();
    if ctx.auto_start_pending {
        set_enabled(true);
        info!("Voice assistant enabled in config — will auto-start when models are ready");
    }

    // Try auto-start immediately if models are already cached (avoids waiting
    // for the select! timeout on the first iteration).
    ctx.check_auto_start();

    loop {
        tokio::select! {
            () = shutdown_token.cancelled() => {
                info!("Voice pipeline shutting down");
                ctx.handle_shutdown();
                break;
            }

            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(VoiceCommand::StartListening) => ctx.handle_start_listening(),
                    Some(VoiceCommand::StopListening) => ctx.handle_stop_listening(),
                    Some(VoiceCommand::StartEnrollment) => ctx.handle_start_enrollment(),
                    Some(VoiceCommand::CancelEnrollment) => ctx.handle_cancel_enrollment(),
                    Some(VoiceCommand::Shutdown) | None => break,
                }
            }

            audio_chunk = async {
                if let Some(rx) = &mut ctx.mic_rx {
                    rx.recv().await
                } else {
                    std::future::pending::<Option<Vec<f32>>>().await
                }
            } => {
                let Some(samples) = audio_chunk else {
                    warn!("Microphone stream ended");
                    set_status(VoiceStatus::MicDisconnected);
                    ctx.handle_stop_listening();
                    continue;
                };

                if ctx.enrollment_mode {
                    let (sample, total) = {
                        let state = voice_state().read().unwrap_poison();
                        (state.enrollment_buffer.len(), NUM_ENROLLMENT_SAMPLES)
                    };
                    handle_enrollment_audio(&samples, &mut ctx, sample, total);
                } else if ctx.is_recording {
                    handle_recording_audio(
                        samples,
                        &mut ctx.command_buffer,
                        &mut ctx.silence_since,
                        &mut ctx.is_recording,
                    )
                    .await;
                } else {
                    handle_wake_word_detection(
                        &samples,
                        &mut ctx.audio_buffer,
                        &mut ctx.voice_batch,
                        &mut ctx.mel_frame_buffer,
                        &mut ctx.embedding_ring,
                        &mut ctx.is_recording,
                        &mut ctx.command_buffer,
                        &mut ctx.silence_since,
                    );
                }
            }

            // Periodic wake-up so check_auto_start can fire when async model
            // downloads complete after the initial select! entry.
            () = tokio::time::sleep(Duration::from_secs(1)) => {}
        }

        // Process any pending enrollment utterance (accumulated inline to avoid
        // race conditions with the command channel). ONNX inference inside
        // handle_enrollment_sample uses spawn_blocking so it doesn't block.
        if let Some(samples) = ctx.enrollment_pending.take() {
            handle_enrollment_sample(samples).await;
            // Reset enrollment_mode only on successful completion, not on
            // failure — if finalize_enrollment failed, the user can retry
            // by speaking the wake word again without re-initiating enrollment.
            if matches!(
                voice_state().read().unwrap_poison().status,
                VoiceStatus::Enrolled
            ) {
                ctx.enrollment_mode = false;
            }
        }

        // Auto-start when models become ready (async download case).
        ctx.check_auto_start();
    }

    info!("Voice pipeline exited");
}

/// Handle enrollment sample: process audio into embeddings and accumulate.
///
/// ONNX inference is CPU-bound (mel spectrogram + embedding computation).
/// It runs on a blocking thread via `spawn_blocking` to avoid starving
/// the async pipeline during enrollment.
async fn handle_enrollment_sample(samples: Vec<f32>) {
    if !models_ready() {
        warn!("Models not ready for enrollment");
        return;
    }

    // Run ONNX inference on a blocking thread to avoid blocking the async pipeline.
    let embeddings_result =
        tokio::task::spawn_blocking(move || process_enrollment_sample(&samples))
            .await
            .unwrap_or_else(|e| Err(anyhow!("Blocking task failed: {e}")));

    match embeddings_result {
        Ok(embeddings) => {
            let count = {
                let mut state = voice_state().write().unwrap_poison();
                state.enrollment_buffer.push(embeddings);
                let count = state.enrollment_buffer.len();
                // state dropped here — no lock held across await
                count
            };

            if count >= NUM_ENROLLMENT_SAMPLES {
                match finalize_enrollment("custom") {
                    Ok(template) => {
                        info!("Enrollment complete: wake word 'custom'");
                        let mut templates = get_templates();
                        let templates_mut = Arc::make_mut(&mut templates);
                        templates_mut.templates.retain(|t| t.name != "custom");
                        templates_mut.templates.push(template);
                        set_templates(templates);

                        // Persist templates to config DB
                        persist_templates().await;

                        // Clear enrollment mode
                        set_status(VoiceStatus::Enrolled);
                    }
                    Err(e) => {
                        warn!("Enrollment finalization failed: {e}");
                        set_status(VoiceStatus::Error("Enrollment failed".to_string()));
                    }
                }
            } else {
                set_status(VoiceStatus::Enrolling {
                    sample: count,
                    total: NUM_ENROLLMENT_SAMPLES,
                });
            }
        }
        Err(e) => {
            warn!("Failed to process enrollment sample: {e}");
            set_status(VoiceStatus::Error("Failed to process sample".to_string()));
        }
    }
}

/// Persist current wake word templates to the config database.
async fn persist_templates() {
    let tpl = get_templates();
    if let Ok(json) = serde_json::to_string(&tpl) {
        let store = crate::config_db::store();
        if let Err(e) = store.set_kv("wake_word_templates", &json).await {
            warn!("Failed to persist wake word templates: {e}");
        } else {
            // Update CONFIG in-memory so that the next `save_and_reload()`
            // sees `wake_word_templates` as `Some(json)` rather than `None`.
            // Without this, `save_and_reload` would delete the key on save
            // (None fields trigger delete_kv_tx), silently erasing templates.
            let _ = CONFIG.set_string_field("wake_word_templates", &json);
            info!("Wake word templates persisted to config");
        }
    }
}

/// Handle recording audio: accumulate buffer and check for silence/duration limits.
#[allow(clippy::cast_precision_loss)]
async fn handle_recording_audio(
    samples: Vec<f32>,
    command_buffer: &mut Vec<f32>,
    silence_since: &mut Option<Instant>,
    is_recording: &mut bool,
) {
    command_buffer.extend_from_slice(&samples);
    let speech = is_speech(&samples, VAD_THRESHOLD);
    if speech {
        *silence_since = None;
    } else {
        silence_since.get_or_insert_with(Instant::now);
    }

    let duration_secs = command_buffer.len() as f64 / f64::from(SAMPLE_RATE);
    let silence_timeout = silence_since.is_some_and(|t| t.elapsed() >= SILENCE_DURATION);

    if silence_timeout || duration_secs > MAX_RECORD_SECS as f64 {
        info!(
            "Recording stopped: {:.1}s, reason: {}",
            duration_secs,
            if silence_timeout {
                "silence"
            } else {
                "max duration"
            }
        );

        set_status(VoiceStatus::Transcribing);
        let cmd_buf = std::mem::take(command_buffer);

        match transcribe_audio(&cmd_buf).await {
            Ok(transcribed) => {
                info!("Transcribed: {transcribed}");
                route_to_agent(transcribed).await;
                set_status(VoiceStatus::Listening);
                *is_recording = false;
            }
            Err(e) => {
                warn!("Transcription failed: {e}");
                set_status(VoiceStatus::Error("Transcription failed".to_string()));
                tokio::time::sleep(Duration::from_secs(2)).await;
                set_status(VoiceStatus::Listening);
                *is_recording = false;
            }
        }
    }
}

/// Process accumulated voiced audio through the mel spectrogram ONNX model.
/// Batches multiple frames into a single ONNX call for efficiency.
///
/// ONNX inference (`compute_mel_spectrogram`) is CPU-bound. We wrap it in
/// `block_in_place` so the tokio runtime can run other tasks on this thread
/// during inference, consistent with the enrollment path which uses
/// `spawn_blocking` for the same purpose.
fn flush_voice_batch(voice_batch: &[f32], mel_frame_buffer: &mut Vec<Vec<f32>>) {
    if voice_batch.len() < FRAME_LENGTH {
        return; // not enough for a single frame
    }
    let Some(models) = ONNX_MODELS.get() else {
        return;
    };

    let batch = voice_batch.to_vec();
    let frames = crate::util::with_block_in_place(|| compute_mel_spectrogram(models, &batch));
    match frames {
        Ok(frames) => {
            for f in frames {
                mel_frame_buffer.push(f);
            }
            // Keep buffer bounded — older frames are discarded
            while mel_frame_buffer.len() > EMBEDDING_WINDOW_FRAMES {
                mel_frame_buffer.remove(0);
            }
        }
        Err(e) => debug!("Mel spectrogram failed: {e}"),
    }
}

/// Handle wake word detection: process audio frames through mel/embedding/DTW pipeline.
///
/// Audio arrives in small chunks (~256 samples at 16kHz). This function:
/// 1. Accumulates audio in a sliding window for VAD
/// 2. Collects voiced frames into a batch buffer
/// 3. Processes the batch through mel ONNX when enough audio is accumulated (~128ms)
/// 4. Produces embeddings and matches against enrolled wake word templates
///
/// Batching reduces ONNX inference calls from ~62/sec (per-frame) to ~8/sec.
#[allow(clippy::too_many_arguments)]
fn handle_wake_word_detection(
    samples: &[f32],
    audio_buffer: &mut Vec<f32>,
    voice_batch: &mut Vec<f32>,
    mel_frame_buffer: &mut Vec<Vec<f32>>,
    embedding_ring: &mut Vec<Vec<f32>>,
    is_recording: &mut bool,
    command_buffer: &mut Vec<f32>,
    silence_since: &mut Option<Instant>,
) {
    audio_buffer.extend_from_slice(samples);

    while audio_buffer.len() >= FRAME_LENGTH {
        // ✅ FIX: Read the frame BEFORE draining (was: drain-before-read bug)
        let frame: Vec<f32> = audio_buffer[..FRAME_LENGTH].to_vec();
        audio_buffer.drain(..HOP_LENGTH);

        // VAD gate — skip silence to avoid wasted ONNX compute
        if is_speech(&frame, VAD_THRESHOLD) {
            voice_batch.extend_from_slice(&frame);
        } else if !voice_batch.is_empty() {
            // Silence transition: flush accumulated voiced batch
            flush_voice_batch(voice_batch, mel_frame_buffer);
            voice_batch.clear();
            if try_match_wake_word_and_push_embedding(
                mel_frame_buffer,
                embedding_ring,
                is_recording,
                command_buffer,
                silence_since,
            ) {
                return;
            }
            continue;
        }

        // Process batch when enough voiced audio accumulated
        // (every ~128ms instead of every 32ms)
        if voice_batch.len() >= VOICE_BATCH_SIZE {
            flush_voice_batch(voice_batch, mel_frame_buffer);
            voice_batch.clear();
            if try_match_wake_word_and_push_embedding(
                mel_frame_buffer,
                embedding_ring,
                is_recording,
                command_buffer,
                silence_since,
            ) {
                return;
            }
        }
    }
}

/// Handle audio during enrollment mode.
///
/// Accumulates audio frames into utterances with VAD-based boundary detection.
/// When a complete utterance is detected (speech followed by silence exceeding
/// `SILENCE_DURATION`), stores the utterance in `enrollment_pending` for
/// inline processing (avoids race conditions with the command channel).
/// The entire utterance (speech + trailing silence) is captured so the ONNX
/// embedding model has enough audio data to produce a reliable template.
///
/// Updates voice status dynamically to reflect the enrollment phase:
/// - No speech yet: caller's `Enrolling` text persists
/// - Speech detected: `ListeningDuringEnrollment`
/// - Speech ended, awaiting silence: `WaitingForSilenceDuringEnrollment`
fn handle_enrollment_audio(samples: &[f32], ctx: &mut PipelineCtx, sample: usize, total: usize) {
    ctx.audio_buffer.extend_from_slice(samples);

    while ctx.audio_buffer.len() >= FRAME_LENGTH {
        let frame: Vec<f32> = ctx.audio_buffer[..FRAME_LENGTH].to_vec();
        ctx.audio_buffer.drain(..HOP_LENGTH);

        // Accumulate every frame into the utterance — both speech and silence.
        // We need the full recording (not just speech frames) to pass to
        // extract_embeddings_from_audio, which expects continuous audio.
        ctx.utterance_buf.extend_from_slice(&frame);

        if is_speech(&frame, VAD_THRESHOLD) {
            let was_waiting_for_silence = ctx.utterance_silence_since.is_some();
            if !ctx.utterance_had_speech || was_waiting_for_silence {
                // Transition from silence to speech, or speech resumed after
                // a pause before the 1.5s timeout — show "Listening…"
                set_status(VoiceStatus::ListeningDuringEnrollment { sample, total });
            }
            ctx.utterance_had_speech = true;
            ctx.utterance_silence_since = None;
        } else if ctx.utterance_had_speech {
            // After speech: track silence duration to detect utterance end.
            let sil_start = ctx.utterance_silence_since.get_or_insert_with(Instant::now);
            if sil_start.elapsed() >= SILENCE_DURATION {
                // Utterance is complete — set pending for inline processing
                // instead of sending a command through the channel (which
                // competes with continuously-arriving audio chunks).
                ctx.enrollment_pending = Some(std::mem::take(&mut ctx.utterance_buf));
                ctx.utterance_had_speech = false;
                ctx.utterance_silence_since = None;
                ctx.audio_buffer.clear();
                break;
            }
            // Set status during the first ~200ms of silence to show
            // "Keep silent to confirm…". The gate is intentionally wider
            // than a single frame (16ms) so the UI reliably transitions
            // even under scheduling jitter. The status write is idempotent.
            if sil_start.elapsed() < Duration::from_millis(200) {
                set_status(VoiceStatus::WaitingForSilenceDuringEnrollment { sample, total });
            }
        }
    }
}

/// Compute embedding from mel frames, push to ring buffer, and match against templates.
/// Returns `true` if wake word was detected (caller should clear state and return).
fn try_match_wake_word_and_push_embedding(
    mel_frame_buffer: &mut Vec<Vec<f32>>,
    embedding_ring: &mut Vec<Vec<f32>>,
    is_recording: &mut bool,
    command_buffer: &mut Vec<f32>,
    silence_since: &mut Option<Instant>,
) -> bool {
    if mel_frame_buffer.len() != EMBEDDING_WINDOW_FRAMES {
        return false;
    }
    let Some(models) = ONNX_MODELS.get() else {
        return false;
    };

    let Ok(embedding) =
        crate::util::with_block_in_place(|| compute_embedding(models, mel_frame_buffer))
    else {
        return false;
    };

    embedding_ring.push(embedding);
    while embedding_ring.len() > EMBEDDING_RING_MAX {
        embedding_ring.remove(0);
    }

    let templates = get_templates();
    if !templates.templates.is_empty()
        && !embedding_ring.is_empty()
        && let Some((dist, tpl)) = match_against_templates(embedding_ring, &templates)
    {
        info!(
            "Wake word '{}' detected! (dist={dist:.4}, threshold={})",
            tpl.name, tpl.threshold
        );
        *is_recording = true;
        command_buffer.clear();
        *silence_since = None;
        mel_frame_buffer.clear();
        embedding_ring.clear();
        set_status(VoiceStatus::Recording);
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    //! ## Test constraints
    //!
    //! - **`VOICE_PIPELINE`** must be uninitialized before
    //!   `test_voice_pipeline_commands_and_enrollment_guard` runs. Do not call
    //!   [`voice::init_global`](crate::voice::init_global) or set
    //!   `VOICE_PIPELINE` in any other test without updating this one.
    //! - **Global `CONFIG`** is read by [`PipelineCtx::new()`] to set
    //!   `auto_start_pending`. Tests implicitly depend on `CONFIG` being in its
    //!   default state (all fields `None`). If a preceding test modifies
    //!   `CONFIG`, `auto_start_pending` may be non-`false`, which this test's
    //!   assertions must still tolerate.
    use super::*;
    use std::f32::consts::PI;

    // ── cosine_distance ───────────────────────────────────────────────────

    #[test]
    fn test_cosine_distance_identical() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![1.0, 2.0, 3.0];
        let d = super::cosine_distance(&v1, &v2);
        assert!((d - 0.0).abs() < 1e-6, "expected 0, got {d}");
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];
        let d = super::cosine_distance(&v1, &v2);
        assert!((d - 1.0).abs() < 1e-6, "expected 1, got {d}");
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![-1.0, 0.0];
        let d = super::cosine_distance(&v1, &v2);
        assert!((d - 2.0).abs() < 1e-6, "expected 2, got {d}");
    }

    #[test]
    fn test_cosine_distance_zero_vector() {
        let v1 = vec![0.0, 0.0];
        let v2 = vec![1.0, 0.0];
        let d = super::cosine_distance(&v1, &v2);
        assert!(d.is_finite(), "zero vector distance should be finite");
    }

    #[test]
    fn test_cosine_distance_mismatched_lengths() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![1.0];
        let d = super::cosine_distance(&v1, &v2);
        assert!(d.is_finite(), "mismatched lengths should return finite");
    }

    // ── dtw_distance ─────────────────────────────────────────────────────

    #[test]
    fn test_dtw_distance_identical() {
        let seq = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let d = super::dtw_distance(&seq, &seq);
        assert!((d - 0.0).abs() < 1e-6, "expected 0, got {d}");
    }

    #[test]
    fn test_dtw_distance_different() {
        let s1 = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let s2 = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let d = super::dtw_distance(&s1, &s2);
        assert!(d > 0.0, "different sequences should have positive distance");
    }

    #[test]
    fn test_dtw_distance_single() {
        let s1 = vec![vec![1.0, 2.0, 3.0]];
        let s2 = vec![vec![4.0, 5.0, 6.0]];
        let d = super::dtw_distance(&s1, &s2);
        assert!(
            d > 0.0,
            "single-element sequences should give positive distance"
        );
    }

    // ── is_speech (VAD) ──────────────────────────────────────────────────

    #[test]
    fn test_is_speech_silence() {
        let silence = vec![0.0f32; 512];
        assert!(!super::is_speech(&silence, 0.01));
    }

    #[test]
    fn test_is_speech_loud() {
        let loud = vec![0.5f32; 512];
        assert!(super::is_speech(&loud, 0.01));
    }

    #[test]
    fn test_is_speech_moderate() {
        let modr = vec![0.02f32; 512];
        assert!(super::is_speech(&modr, 0.01));
    }

    #[test]
    fn test_is_speech_empty() {
        assert!(!super::is_speech(&[], 0.01));
    }

    #[test]
    fn test_is_speech_high_threshold() {
        let quiet = vec![0.005f32; 512];
        assert!(!super::is_speech(&quiet, 0.01));
        assert!(super::is_speech(&quiet, 0.001));
    }

    // ── to_mono ──────────────────────────────────────────────────────────

    #[test]
    fn test_to_mono_already_mono() {
        let input = vec![0.5, 0.3, 0.1];
        let output = super::to_mono(&input, 1);
        assert_eq!(output.len(), 3);
        approx_eq(output[0], 0.5);
        approx_eq(output[1], 0.3);
        approx_eq(output[2], 0.1);
    }

    #[test]
    fn test_to_mono_stereo() {
        let input = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
        let output = super::to_mono(&input, 2);
        assert_eq!(output.len(), 3);
        approx_eq(output[0], 2.0);
        approx_eq(output[1], 6.0);
        approx_eq(output[2], 10.0);
    }

    #[test]
    fn test_to_mono_quad() {
        let input = vec![1.0, 1.0, 1.0, 1.0, 10.0, 0.0, 0.0, 0.0];
        let output = super::to_mono(&input, 4);
        assert_eq!(output.len(), 2);
        approx_eq(output[0], 1.0);
        approx_eq(output[1], 2.5);
    }

    // ── resample_audio ───────────────────────────────────────────────────

    #[test]
    fn test_resample_audio_same_rate() {
        let input: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let output = super::resample_audio(&input, 16000, 16000);
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "expected approx equal, got {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_resample_audio_downsample() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32 * PI / 50.0).sin()).collect();
        let output = super::resample_audio(&input, 16000, 8000);
        assert!(output.len() < input.len(), "downsampled should be shorter");
        assert!(output.len() > 0, "downsampled should not be empty");
    }

    #[test]
    fn test_resample_audio_upsample() {
        let input: Vec<f32> = (0..50).map(|i| (i as f32 * PI / 25.0).sin()).collect();
        let output = super::resample_audio(&input, 8000, 16000);
        assert!(
            output.len() >= input.len() * 2 - 2,
            "upsampled should be ~2x longer, got {} vs {}*2",
            output.len(),
            input.len()
        );
    }

    #[test]
    fn test_resample_audio_empty() {
        let output = super::resample_audio(&[], 16000, 8000);
        assert!(output.is_empty());
    }

    // ── samples_to_wav ───────────────────────────────────────────────────

    #[test]
    fn test_samples_to_wav_basic() {
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.0];
        let wav = super::samples_to_wav(&samples, SAMPLE_RATE);
        assert!(wav.len() >= 44);
        assert_eq!(
            wav.len(),
            44 + samples.len() * 2,
            "WAV should have 44-byte header + 2 bytes per sample"
        );
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 1);
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 1);
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            16000
        );
    }

    #[test]
    fn test_samples_to_wav_empty() {
        let wav = super::samples_to_wav(&[], SAMPLE_RATE);
        assert_eq!(
            wav.len(),
            44,
            "empty samples should produce header-only WAV"
        );
    }

    // ── helper ───────────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "expected approx {a} == {b}");
    }

    // ── PipelineCtx tests ────────────────────────────────────────────────

    #[test]
    fn test_voice_pipeline_commands_and_enrollment_guard() {
        // Initialize global VOICE_PIPELINE once for all checks below.
        // Using assert!(is_ok()) so that parallel test execution would fail
        // fast rather than silently giving stale state.
        assert!(
            VOICE_PIPELINE
                .set(RwLock::new(VoicePipelineState {
                    enabled: false,
                    status: VoiceStatus::Disabled,
                    templates: Arc::new(WakeWordTemplates::default()),
                    enrollment_buffer: Vec::new(),
                    cmd_tx: None,
                }))
                .is_ok(),
            "VOICE_PIPELINE already initialized — tests must not share state",
        );

        // ── handle_start_enrollment guard ─────────────────────────
        let mut ctx = PipelineCtx::new();
        assert!(!ctx.is_listening, "default state should not be listening");
        assert!(
            !ctx.enrollment_mode,
            "should not be in enrollment mode initially"
        );

        ctx.handle_start_enrollment();

        assert!(
            !ctx.enrollment_mode,
            "enrollment should be rejected when mic is not running"
        );

        // ── send_command with cmd_tx = None ─────────────────────────
        // None of these should panic — they log a warning and return.
        send_command(VoiceCommand::StartListening);
        send_command(VoiceCommand::StopListening);
        send_command(VoiceCommand::StartEnrollment);
        send_command(VoiceCommand::CancelEnrollment);

        // ── Model-ready retry path (ticket req #2) ─────────────────
        // Verify that handle_start_listening sets auto_start_pending when
        // models aren't ready, and check_auto_start consumes the flag once
        // models transition to Ready.
        {
            // Enable voice so the retry path is exercised.
            voice_state().write().unwrap_poison().enabled = true;

            // Models not yet ready — handle_start_listening should set
            // auto_start_pending (one-shot retry flag).
            MODELS_STATE.store(ModelState::Loading, Ordering::Release);
            ctx.handle_start_listening();
            assert!(
                ctx.auto_start_pending,
                "auto_start_pending should be set when models are not ready"
            );

            // Models become ready — check_auto_start should consume the flag.
            MODELS_STATE.store(ModelState::Ready, Ordering::Release);
            ctx.check_auto_start();
            assert!(
                !ctx.auto_start_pending,
                "auto_start_pending should be cleared after check_auto_start"
            );

            // Second call to check_auto_start is a no-op (one-shot).
            ctx.check_auto_start();
            assert!(
                !ctx.auto_start_pending,
                "auto_start_pending must remain false after one-shot consumed"
            );

            // Clean up: reset for other checks (the actual pipeline will set
            // its own state).
            voice_state().write().unwrap_poison().enabled = false;
            MODELS_STATE.store(ModelState::Uninit, Ordering::Release);
        }

        // ── handle_enrollment_audio status transitions ───────────────
        {
            voice_state().write().unwrap_poison().enabled = true;
            ctx.is_listening = true;
            ctx.enrollment_mode = true;
            ctx.utterance_buf.clear();
            ctx.utterance_had_speech = false;
            ctx.utterance_silence_since = None;
            ctx.audio_buffer.clear();
            ctx.enrollment_pending = None;
            voice_state()
                .write()
                .unwrap_poison()
                .enrollment_buffer
                .clear();

            // Start enrollment at sample 0 of 3
            set_status(VoiceStatus::Enrolling {
                sample: 0,
                total: 3,
            });

            // 1. Silence frames: no speech detected — status stays Enrolling
            let silence = vec![0.0f32; FRAME_LENGTH];
            handle_enrollment_audio(&silence, &mut ctx, 0, 3);
            assert!(
                matches!(get_status(), VoiceStatus::Enrolling { .. }),
                "silence before speech should not change status from Enrolling"
            );

            // 2. Speech starts → ListeningDuringEnrollment
            let speech = vec![0.5f32; FRAME_LENGTH];
            handle_enrollment_audio(&speech, &mut ctx, 0, 3);
            assert!(
                matches!(get_status(), VoiceStatus::ListeningDuringEnrollment { .. }),
                "speech should trigger ListeningDuringEnrollment"
            );
            assert!(
                ctx.utterance_had_speech,
                "utterance_had_speech should be set"
            );

            // 3. Continued speech: status stays ListeningDuringEnrollment
            //    (utterance_had_speech was set by step 2, so the
            //    !utterance_had_speech gate in the function prevents re-setting)
            handle_enrollment_audio(&speech, &mut ctx, 0, 3);
            assert!(
                matches!(get_status(), VoiceStatus::ListeningDuringEnrollment { .. }),
                "continued speech should remain ListeningDuringEnrollment"
            );

            // 4. Silence after speech → WaitingForSilenceDuringEnrollment
            handle_enrollment_audio(&silence, &mut ctx, 0, 3);
            assert!(
                matches!(
                    get_status(),
                    VoiceStatus::WaitingForSilenceDuringEnrollment { .. }
                ),
                "silence after speech should trigger WaitingForSilenceDuringEnrollment"
            );

            // 5. Speech resumes before 1.5s timeout → back to ListeningDuringEnrollment
            //    This validates the was_waiting_for_silence fix for the speech-resume glitch.
            handle_enrollment_audio(&speech, &mut ctx, 0, 3);
            assert!(
                matches!(get_status(), VoiceStatus::ListeningDuringEnrollment { .. }),
                "speech resumed after pause should revert to ListeningDuringEnrollment"
            );

            // Clean up
            voice_state().write().unwrap_poison().enabled = false;
            ctx.is_listening = false;
            ctx.enrollment_mode = false;
        }
    }

    // ── Audio scaling (int16 range) ─────────────────────────────────────

    /// Verify that audio samples in [-1, 1] scale correctly to int16 range
    /// via `scale_to_int16_range`, the same function used by
    /// `compute_mel_spectrogram`. This ensures the mel model receives values
    /// in the range it was trained on.
    #[test]
    fn test_audio_scaling_to_int16_range() {
        // Corner cases: full scale, zero, and midpoint values.
        // All inputs are exact f32 powers-of-2, so multiplication by 32768.0
        // (also exact) produces exact results — f32::EPSILON (~1.19e-7) is
        // the correct tolerance here. For non-exact inputs (e.g. 0.1) use a
        // larger tolerance (e.g. 1e-3).
        let samples = vec![-1.0, 0.0, 1.0, 0.5, -0.5];
        let scaled = scale_to_int16_range(&samples);
        assert!(
            (scaled[0] + 32768.0).abs() < f32::EPSILON,
            "-1.0 should map to -32768"
        );
        assert!(
            (scaled[1] - 0.0).abs() < f32::EPSILON,
            "0.0 should map to 0"
        );
        assert!(
            (scaled[2] - 32768.0).abs() < f32::EPSILON,
            "1.0 should map to 32768"
        );
        assert!(
            (scaled[3] - 16384.0).abs() < f32::EPSILON,
            "0.5 should map to 16384"
        );
        assert!(
            (scaled[4] + 16384.0).abs() < f32::EPSILON,
            "-0.5 should map to -16384"
        );
    }
}
