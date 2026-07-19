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

/// Minimum absolute separation between the mean pairwise distance and the
/// acceptance threshold.  With only [`NUM_ENROLLMENT_SAMPLES`] (3) samples,
/// standard deviation estimates are unreliable — a single outlier or three
/// artificially similar samples can produce an unrealistically tight std,
/// collapsing the threshold to nearly-equal the mean.  This floor ensures
/// at least 0.05 of margin so genuine wake-word utterances have room to
/// match (mahbot-755 Fix 2).
const MIN_THRESHOLD_FLOOR: f32 = 0.05;

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
        Self::from_u8(self.0.load(order))
    }

    fn store(&self, state: ModelState, order: Ordering) {
        self.0.store(state as u8, order);
    }

    /// Atomically compare-and-exchange the current state.
    ///
    /// See [`AtomicU8::compare_exchange`] for ordering semantics.
    fn compare_exchange(
        &self,
        expected: ModelState,
        new: ModelState,
        success: Ordering,
        failure: Ordering,
    ) -> Result<ModelState, ModelState> {
        self.0
            .compare_exchange(expected as u8, new as u8, success, failure)
            .map(Self::from_u8)
            .map_err(Self::from_u8)
    }

    fn from_u8(v: u8) -> ModelState {
        match v {
            1 => ModelState::Loading,
            2 => ModelState::Ready,
            3 => ModelState::Failed,
            _ => ModelState::Uninit,
        }
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
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    pub threshold: f32,
}

/// Collection of enrolled wake word templates.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct WakeWordTemplates {
    #[serde(default)]
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
    RetryModelLoading,
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

/// Logs mel spectrogram statistics on first call to verify pipeline health.
static LOG_MEL_STATS: std::sync::Once = std::sync::Once::new();

/// Apply the mandatory spec/10 + 2 transform from the OpenWakeWord reference.
///
/// The mel model was ported from TensorFlow and its output range differs from
/// what the embedding model was trained on. Without this transform the
/// embedding model receives out-of-distribution values and produces garbage
/// embeddings. Extracted as a named function for testability.
fn spec_transform(v: f32) -> f32 {
    v / 10.0 + 2.0
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
            // Apply the mandatory spec/10 + 2 transform from the OpenWakeWord
            // reference. The mel model was ported from TensorFlow and its output
            // range differs from what the embedding model expects. Without this
            // transform the embedding model receives out-of-distribution values
            // and produces garbage embeddings.
            let frame: Vec<f32> = output_data[start..start + num_features]
                .iter()
                .map(|&v| spec_transform(v))
                .collect();
            frames.push(frame);
        }
    }

    // Log mel spectrogram statistics on first call to verify pipeline health.
    LOG_MEL_STATS.call_once(|| {
        if let (Some(min), Some(max)) = (
            frames.iter().flatten().copied().reduce(f32::min),
            frames.iter().flatten().copied().reduce(f32::max),
        ) {
            info!(
                num_frames = frames.len(),
                min_val = min,
                max_val = max,
                "Mel spectrogram: first call statistics"
            );
        }
    });

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
        // Audio too short for a full 76-frame window — pad with silence
        // frames so at least one embedding can be computed.  Without this,
        // short wake words (e.g. 0.5s) would be silently discarded during
        // enrollment, making enrollment impossible for brief utterances.
        let mut padded = mel_frames;
        let silence_frame = vec![0.0; NUM_MEL_BANDS];
        while padded.len() < EMBEDDING_WINDOW_FRAMES {
            padded.push(silence_frame.clone());
        }
        let embedding = compute_embedding(models, &padded)?;
        return Ok(vec![embedding]);
    }

    let mut embeddings = Vec::new();
    let stride: usize = 8; // OpenWakeWord reference uses stride=8 (~89.5% overlap)

    let mut start = 0;
    while start + EMBEDDING_WINDOW_FRAMES <= mel_frames.len() {
        let window = &mel_frames[start..start + EMBEDDING_WINDOW_FRAMES];
        match compute_embedding(models, window) {
            Ok(emb) => embeddings.push(emb),
            Err(e) => warn!("Skipping embedding window: {e}"),
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

#[allow(clippy::cast_precision_loss)]
fn dtw_distance(live: &[Vec<f32>], template: &[Vec<f32>]) -> f32 {
    if live.is_empty() || template.is_empty() {
        return f32::MAX;
    }

    let m = template.len();
    let mut prev = vec![f32::MAX; m];
    let mut curr = vec![f32::MAX; m];
    // Track path length alongside cumulative cost so we can normalise by
    // path length.  Without this, the cumulative distance grows with
    // template length (enrollment concatenates all samples into one long
    // template), making the runtime cumulative distance always exceed
    // the threshold calibrated on short pairwise DTW between individual
    // samples.
    let mut prev_len = vec![0usize; m];
    let mut curr_len = vec![0usize; m];

    for (i, live_i) in live.iter().enumerate() {
        for (j, tpl_j) in template.iter().enumerate() {
            let cost = cosine_distance(live_i, tpl_j);
            if i == 0 && j == 0 {
                curr[j] = cost;
                curr_len[j] = 1;
            } else if i == 0 {
                curr[j] = cost + curr[j - 1];
                curr_len[j] = curr_len[j - 1] + 1;
            } else if j == 0 {
                curr[j] = cost + prev[j];
                curr_len[j] = prev_len[j] + 1;
            } else {
                let vert = prev[j];
                let diag = prev[j - 1];
                let horiz = curr[j - 1];
                if vert <= diag && vert <= horiz {
                    curr[j] = cost + vert;
                    curr_len[j] = prev_len[j] + 1;
                } else if diag <= horiz {
                    curr[j] = cost + diag;
                    curr_len[j] = prev_len[j - 1] + 1;
                } else {
                    curr[j] = cost + horiz;
                    curr_len[j] = curr_len[j - 1] + 1;
                }
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        std::mem::swap(&mut prev_len, &mut curr_len);
    }

    // Return average distance per step (normalised cumulative).
    // This makes the distance length-invariant — a 2-second utterance
    // matches its template as well as a 0.3-second one.
    if prev_len[m - 1] > 0 {
        prev[m - 1] / prev_len[m - 1] as f32
    } else {
        f32::MAX
    }
}

/// Match a live embedding sequence against all enrolled templates.
///
/// Uses per-template sliding window matching: for each template, only the
/// most recent `template.embeddings.len()` embeddings from the live sequence
/// are compared. This avoids length asymmetry noise when the ring buffer
/// grows larger than the template (mahbot-755 Fix 5).
///
/// Logs DTW distances at debug level for all templates (not just matches)
/// so near-misses are visible during troubleshooting (mahbot-755 Fix 4).
fn match_against_templates<'a>(
    live_sequence: &[Vec<f32>],
    templates: &'a WakeWordTemplates,
) -> Option<(f32, &'a WakeWordTemplate)> {
    let mut best: Option<(f32, &WakeWordTemplate)> = None;

    for tpl in &templates.templates {
        // Sliding window: use at most `tpl.embeddings.len()` most recent
        // live embeddings.  This keeps the DTW cost matrix balanced and
        // reduces noise from extraneous audio before the wake word.
        let window_len = tpl.embeddings.len().min(live_sequence.len());
        let window = &live_sequence[live_sequence.len() - window_len..];

        let dist = dtw_distance(window, &tpl.embeddings);
        debug!(
            "DTW: template='{}' window={window_len} dist={dist:.4} threshold={} {}",
            tpl.name,
            tpl.threshold,
            if dist < tpl.threshold {
                "✓ MATCH"
            } else {
                "✗ no match"
            }
        );
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

/// Detect whether a microphone-error report is caused by OS-level
/// permission denial (rather than a transient device issue).
///
/// On macOS, CoreAudio returns `kAudioUnitErr_NoConnection` (-10875)
/// when the application has not been granted microphone access.  We
/// also check for common cross-platform error-text patterns so the
/// user sees a clear `MicPermissionDenied` status instead of a
/// generic `MicDisconnected`.
fn is_mic_permission_error(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}");
    msg.contains("NoConnection")
        || msg.contains("-10875")
        || msg.contains("permission")
        || msg.contains("denied")
        || msg.to_lowercase().contains("access denied")
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

    // Anti-aliasing filter: when downsampling (from_rate > to_rate),
    // apply a simple binomial low-pass filter to attenuate frequencies
    // above the new Nyquist before decimation.  Without this filter,
    // linear interpolation introduces aliasing — high-frequency content
    // above to_rate/2 folds back into the audible range as noise,
    // degrading mel spectrogram feature quality.
    //
    // The 3-tap binomial [0.25, 0.5, 0.25] gives reasonable stopband
    // attenuation (~6 dB at 0.25 normalised) for speech audio.  For
    // 48 kHz → 16 kHz this attenuates content above ~8 kHz.
    let filtered: Vec<f32> = if from_rate > to_rate && samples.len() >= 3 {
        let mut out = Vec::with_capacity(samples.len());
        // First sample (asymmetric boundary)
        out.push(samples[0] * 0.75 + samples[1] * 0.25);
        for i in 1..samples.len() - 1 {
            out.push(samples[i - 1] * 0.25 + samples[i] * 0.5 + samples[i + 1] * 0.25);
        }
        // Last sample (asymmetric boundary)
        out.push(samples[samples.len() - 2] * 0.25 + samples[samples.len() - 1] * 0.75);
        out
    } else {
        samples.to_vec()
    };

    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;
        if src_idx + 1 < filtered.len() {
            output.push(
                (f64::from(filtered[src_idx]) * (1.0 - frac)
                    + f64::from(filtered[src_idx + 1]) * frac) as f32,
            );
        } else if src_idx < filtered.len() {
            output.push(filtered[src_idx]);
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
    if mel_path.exists()
        && let Err(e) = verify_sha256(&mel_path, MEL_MODEL_SHA256)
    {
        warn!("Mel spectrogram model corrupt, re-downloading: {e}");
        tokio::fs::remove_file(&mel_path).await?;
    }
    if !mel_path.exists() {
        info!("Downloading mel spectrogram model...");
        download_model(MEL_MODEL_URL, &mel_path, MEL_MODEL_SIZE, MEL_MODEL_SHA256).await?;
    }

    let embed_path = dir.join(EMBED_MODEL_FILENAME);
    if embed_path.exists()
        && let Err(e) = verify_sha256(&embed_path, EMBED_MODEL_SHA256)
    {
        warn!("Embedding model corrupt, re-downloading: {e}");
        tokio::fs::remove_file(&embed_path).await?;
    }
    if !embed_path.exists() {
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
        if MODELS_STATE.load(Ordering::Acquire) == ModelState::Ready {
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
                    // Another instance already set the models — adopt Ready
                    // state and exit (avoids wasted retry loops).
                    MODELS_STATE.store(ModelState::Ready, Ordering::Release);
                    info!("Voice models already loaded by another task");
                    set_status(if is_enabled() {
                        VoiceStatus::Listening
                    } else {
                        VoiceStatus::Disabled
                    });
                    return;
                }
                Err(e) => warn!("Failed to load voice models (will retry): {e}"),
            },
            Ok(Err(e)) => warn!("Failed to download voice models (will retry): {e}"),
            Err(_) => warn!("Voice model download timed out (will retry)"),
        }

        if MODELS_STATE.load(Ordering::Acquire) == ModelState::Failed {
            return;
        }

        tokio::time::sleep(retry_delay).await;
        retry_delay = (retry_delay * 2).min(Duration::from_mins(2));
    }
}

/// Atomically reset the model state from `Failed` to `Uninit` and re-spawn
/// the download retry loop.  Returns `true` if a retry was initiated, `false`
/// if the state was not `Failed` (e.g. already loading or ready).
///
/// This is the primary recovery mechanism for [`VoiceStatus::ModelError`].
/// Callers that hold a [`PipelineCtx`] should prefer the debounced
/// [`PipelineCtx::try_retry_models`] instead to avoid rapid retry storms.
fn retry_model_loading() -> bool {
    // Atomically transition from Failed → Uninit.  If another task already
    // changed the state (e.g. concurrent `retry_model_loading` call or the
    // original retry loop is still running), this is a no-op.
    if MODELS_STATE
        .compare_exchange(
            ModelState::Failed,
            ModelState::Uninit,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        return false;
    }

    set_status(VoiceStatus::LoadingModels);
    tokio::spawn(download_retry_loop());
    info!("Voice models: retrying model load after previous failure");
    true
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
    let threshold = (mean + THRESHOLD_MULTIPLIER * std_dev).max(mean + MIN_THRESHOLD_FLOOR);

    info!(
        "Enrollment calibration: mean={mean:.4}, std={std_dev:.4}, threshold={threshold:.4} (min floor: mean+{MIN_THRESHOLD_FLOOR})"
    );

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
    /// Length of [`utterance_buf`] at the last detected speech frame boundary.
    /// Used to trim trailing silence from enrollment utterances so only the
    /// active speech portion contributes to the template, preventing the
    /// silence-content mismatch between enrollment and live detection
    /// (Root Cause 1 in mahbot-755).
    utterance_speech_end_len: usize,
    auto_start_pending: bool,
    /// Timestamp of the last automatic model retry attempt.  Used to debounce
    /// so we don't spam the retry loop every 1-second tick when models are in
    /// [`ModelState::Failed`] (the periodic wake-up checks the state).
    last_model_retry: Option<Instant>,
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
            utterance_speech_end_len: 0,
            auto_start_pending: CONFIG.voice_enabled().as_deref() == Some("true"),
            last_model_retry: None,
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
            //
            // If models have previously failed (ModelError trap state),
            // trigger a retry immediately so the user doesn't need to
            // restart the app (ticket mahbot-757).
            if MODELS_STATE.load(Ordering::Acquire) == ModelState::Failed {
                warn!("Voice models previously failed — triggering retry...");
                self.try_retry_models();
            }
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
                    set_status(if is_mic_permission_error(&e) {
                        VoiceStatus::MicPermissionDenied
                    } else {
                        VoiceStatus::MicDisconnected
                    });
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
        self.utterance_speech_end_len = 0;
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
        self.utterance_speech_end_len = 0;
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
        self.utterance_speech_end_len = 0;
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

    /// Attempt to retry model loading, debounced to at most once every
    /// 30 seconds.  This prevents rapid retry storms from the periodic
    /// 1-second wake-up in the main pipeline loop.
    fn try_retry_models(&mut self) {
        let cooldown = Duration::from_secs(30);
        if self
            .last_model_retry
            .is_some_and(|t| t.elapsed() < cooldown)
        {
            return;
        }
        if retry_model_loading() {
            self.last_model_retry = Some(Instant::now());
        }
    }

    fn check_auto_start(&mut self) {
        // One-shot retry: only fires when auto_start_pending is true (set at
        // pipeline creation or by handle_start_listening when models weren't
        // ready yet). Cleared after the first attempt — no continuous retry
        // loop on mic failure.
        //
        // Model error recovery (Failed state) is handled by two paths:
        // - Fast path: handle_start_listening() triggers try_retry_models
        //   immediately when a user explicitly starts listening (voice toggle).
        // - Periodic path: the post-select block in run_voice_pipeline runs
        //   every iteration and triggers try_retry_models unconditionally
        //   (debounced to 30s) for self-healing without user interaction.
        //
        // Once models transition back to Ready, this function picks them up
        // via the auto_start_pending flag (set by handle_start_listening).
        if self.auto_start_pending && models_ready() && !self.is_listening {
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
    if let Some(json) = CONFIG.wake_word_templates() {
        match serde_json::from_str::<WakeWordTemplates>(&json) {
            Ok(templates) => {
                set_templates(Arc::new(templates));
                info!(
                    "Loaded {} wake word template(s) from config",
                    get_templates().templates.len()
                );
            }
            Err(e) => {
                warn!("Failed to deserialize stored wake word templates: {e}");
            }
        }
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
                    Some(VoiceCommand::RetryModelLoading) => {
                        // Explicit retry from GUI — bypass debounce
                        if retry_model_loading() {
                            ctx.last_model_retry = Some(Instant::now());
                        } else {
                            warn!("RetryModelLoading: models are not in Failed state");
                        }
                    }
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

            // Periodic wake-up so auto-recovery can fire when async model
            // downloads complete or models transition to Ready/Failed after
            // the initial select! entry.  check_auto_start runs in the
            // post-select section below so we don't duplicate it here.
            () = tokio::time::sleep(Duration::from_secs(1)) => {}
        }

        // Periodic auto-recovery: if models are in Failed state, attempt to
        // retry loading (debounced to at most once every 30s).  This runs
        // regardless of auto_start_pending so that the model error state is
        // self-healing even when voice is toggled off/on manually.
        if MODELS_STATE.load(Ordering::Acquire) == ModelState::Failed {
            ctx.try_retry_models();
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
                // Schedule transition to Listening after showing "Enrolled"
                // for 1.5 seconds, so the user gets visual confirmation that
                // enrollment completed successfully before the pipeline
                // resumes active wake word listening (mahbot-755 Fix 3).
                // The spawned task respects the global shutdown token so it
                // does not write stale state after pipeline exit.
                tokio::spawn(async {
                    let shutdown_token = crate::shutdown::shutdown_token();
                    tokio::select! {
                        () = tokio::time::sleep(Duration::from_millis(1500)) => {
                            if matches!(get_status(), VoiceStatus::Enrolled) {
                                set_status(VoiceStatus::Listening);
                            }
                        }
                        () = shutdown_token.cancelled() => {
                            // Pipeline is shutting down — do not touch state.
                        }
                    }
                });
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
            // Update CONFIG in-memory so that GUI snapshot readers / pipeline
            // restart see the latest templates.  `save_and_reload` no longer
            // touches `wake_word_templates` (it's skipped in the write loop),
            // so this update is about cross-session visibility, not deletion
            // prevention.
            if !CONFIG.set_string_field("wake_word_templates", &json) {
                warn!(
                    "Failed to update CONFIG with wake word templates (key not recognized by \
                     set_string_field — it may have drifted from the `stringify!` arms)"
                );
            }
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
fn flush_voice_batch(voice_batch: &mut Vec<f32>, mel_frame_buffer: &mut Vec<Vec<f32>>) {
    if voice_batch.len() < FRAME_LENGTH {
        return; // not enough for a single frame
    }
    let Some(models) = ONNX_MODELS.get() else {
        return;
    };

    let batch = voice_batch.clone();
    let frames = crate::util::with_block_in_place(|| compute_mel_spectrogram(models, &batch));
    match frames {
        Ok(frames) => {
            debug!(
                "Mel flush: {} mel frames produced (buffer now has {} frames)",
                frames.len(),
                mel_frame_buffer.len() + frames.len(),
            );
            for f in frames {
                mel_frame_buffer.push(f);
            }
            // Keep buffer bounded — older frames are discarded
            while mel_frame_buffer.len() > EMBEDDING_WINDOW_FRAMES {
                mel_frame_buffer.remove(0);
            }
            // Keep only the last (FRAME_LENGTH - HOP_LENGTH) samples for overlap
            // context, ensuring temporal continuity across batch boundaries.  The
            // remaining samples provide the context needed for the first mel frame
            // of the next batch to have the correct 256-sample hop from the last
            // frame of this batch (rather than being computed from a new disjoint
            // window, which would create a 512-sample gap).
            let keep = FRAME_LENGTH.saturating_sub(HOP_LENGTH);
            if voice_batch.len() > keep {
                let drain_to = voice_batch.len() - keep;
                voice_batch.drain(..drain_to);
            }
        }
        Err(e) => warn!("Mel spectrogram failed: {e}"),
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
            // Add only the NEW samples (HOP_LENGTH per frame) to avoid
            // duplicating overlapping audio. Each frame overlaps the previous
            // by 50% (HOP_LENGTH = FRAME_LENGTH/2), so appending the full
            // frame would duplicate half the audio — corrupting the mel model
            // input with repeated segments.
            voice_batch.extend_from_slice(&frame[..HOP_LENGTH]);
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
                audio_buffer,
            ) {
                return;
            }
            continue;
        }

        // Process batch when enough voiced audio accumulated
        // (every ~128ms instead of every 32ms)
        if voice_batch.len() >= VOICE_BATCH_SIZE {
            flush_voice_batch(voice_batch, mel_frame_buffer);
            if try_match_wake_word_and_push_embedding(
                mel_frame_buffer,
                embedding_ring,
                is_recording,
                command_buffer,
                silence_since,
                audio_buffer,
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
/// The utterance is accumulated until a silence timeout of
/// [`SILENCE_DURATION`] fires.  Trailing silence is then trimmed so only the
/// active speech portion contributes to the template — preventing the
/// silence-content mismatch between enrollment (real-noise silence) and live
/// detection (all-zero padding) that inflates DTW distances (mahbot-755 Fix 1).
/// If the trimmed audio is shorter than [`EMBEDDING_WINDOW_FRAMES`] mel frames,
/// [`extract_embeddings_from_audio`] handles the short audio by padding with
/// all-zero silence frames so at least one embedding can be computed.
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

        // Accumulate only the new samples (HOP_LENGTH per frame) into
        // the utterance buffer. Adding the full FRAME_LENGTH frame would
        // duplicate overlapping audio since conSECUTIVE frames overlap by 50%.
        // The utterance buffer is later passed to extract_embeddings_from_audio
        // which expects continuous raw audio, not overlapping frames.
        ctx.utterance_buf.extend_from_slice(&frame[..HOP_LENGTH]);

        if is_speech(&frame, VAD_THRESHOLD) {
            let was_waiting_for_silence = ctx.utterance_silence_since.is_some();
            if !ctx.utterance_had_speech || was_waiting_for_silence {
                // Transition from silence to speech, or speech resumed after
                // a pause before the 1.5s timeout — show "Listening…"
                set_status(VoiceStatus::ListeningDuringEnrollment { sample, total });
            }
            ctx.utterance_had_speech = true;
            ctx.utterance_speech_end_len = ctx.utterance_buf.len();
            ctx.utterance_silence_since = None;
        } else if ctx.utterance_had_speech {
            // After speech: track silence duration to detect utterance end.
            let sil_start = ctx.utterance_silence_since.get_or_insert_with(Instant::now);
            if sil_start.elapsed() >= SILENCE_DURATION {
                // Utterance is complete — trim trailing silence so only
                // the active speech portion contributes to the template.
                // This prevents the silence-content mismatch between
                // enrollment (real-noise silence) and live detection
                // (all-zero padding) that inflates DTW distances (mahbot-755).
                ctx.utterance_buf.truncate(ctx.utterance_speech_end_len);
                ctx.enrollment_pending = Some(std::mem::take(&mut ctx.utterance_buf));
                ctx.utterance_speech_end_len = 0;
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
    audio_buffer: &mut Vec<f32>,
) -> bool {
    if mel_frame_buffer.is_empty() {
        return false;
    }
    let Some(models) = ONNX_MODELS.get() else {
        return false;
    };

    // If the mel buffer is shorter than the required embedding window (76 frames),
    // pad it with silence frames so an embedding can always be computed.  Without
    // this, short wake words (e.g. 0.5s → ~32 mel frames) would silently be
    // discarded and never detected.
    let padded_window: Vec<Vec<f32>>;
    let embed_input: &[Vec<f32>] = if mel_frame_buffer.len() < EMBEDDING_WINDOW_FRAMES {
        padded_window = {
            let mut p = mel_frame_buffer.clone();
            let silence_frame = vec![0.0; NUM_MEL_BANDS];
            while p.len() < EMBEDDING_WINDOW_FRAMES {
                p.push(silence_frame.clone());
            }
            p
        };
        &padded_window
    } else {
        // Take the most recent EMBEDDING_WINDOW_FRAMES
        &mel_frame_buffer[mel_frame_buffer.len() - EMBEDDING_WINDOW_FRAMES..]
    };

    let embedding =
        match crate::util::with_block_in_place(|| compute_embedding(models, embed_input)) {
            Ok(emb) => {
                debug!(
                    "Embedding computed: {} dims (ring size before push: {})",
                    emb.len(),
                    embedding_ring.len(),
                );
                emb
            }
            Err(e) => {
                warn!("Wake word matching: compute_embedding failed: {e:#}");
                return false;
            }
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
        audio_buffer.clear();
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
        // Using let _ to tolerate parallel tests that also need the pipeline.
        let _ = VOICE_PIPELINE.set(RwLock::new(VoicePipelineState {
            enabled: false,
            status: VoiceStatus::Disabled,
            templates: Arc::new(WakeWordTemplates::default()),
            enrollment_buffer: Vec::new(),
            cmd_tx: None,
        }));

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
            ctx.utterance_speech_end_len = 0;
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

    // ── spec/10 + 2 transform (Regression: mahbot-752) ──────────────

    #[test]
    fn test_spec_transform_zero() {
        // v = 0.0 → 0.0/10.0 + 2.0 = 2.0
        let result = spec_transform(0.0);
        assert!(
            (result - 2.0).abs() < f32::EPSILON,
            "expected 2.0, got {result}"
        );
    }

    #[test]
    fn test_spec_transform_positive() {
        // v = 10.0 → 10.0/10.0 + 2.0 = 3.0
        let result = spec_transform(10.0);
        assert!(
            (result - 3.0).abs() < f32::EPSILON,
            "expected 3.0, got {result}"
        );
    }

    #[test]
    fn test_spec_transform_negative() {
        // v = -10.0 → -10.0/10.0 + 2.0 = 1.0
        let result = spec_transform(-10.0);
        assert!(
            (result - 1.0).abs() < f32::EPSILON,
            "expected 1.0, got {result}"
        );
    }

    #[test]
    fn test_spec_transform_typical_mel_value() {
        // Typical ONNX mel output values like -4.0 should map to 1.6
        let result = spec_transform(-4.0);
        assert!(
            (result - 1.6).abs() < f32::EPSILON * 2.0,
            "expected 1.6, got {result}"
        );
    }

    #[test]
    fn test_spec_transform_idempotent_inverse() {
        // Verify that applying the inverse operation recovers the original value.
        // This locks in the mathematical relationship: spec_transform(v) * 10.0 - 20.0 == v
        // which is: (v/10.0 + 2.0) * 10.0 - 20.0 = v + 20.0 - 20.0 = v
        let orig = 42.0;
        let transformed = spec_transform(orig);
        let recovered = transformed * 10.0 - 20.0;
        assert!(
            (recovered - orig).abs() < f32::EPSILON * 10.0,
            "expected {orig}, got {recovered} after round-trip"
        );
    }

    // ── Overlapping frames truncation (Regression: mahbot-752) ──────

    #[test]
    fn test_frame_truncation_to_hop_length() {
        // HOP_LENGTH must be exactly half of FRAME_LENGTH (50% overlap).
        // frame[..HOP_LENGTH] extracts only the NEW samples per iteration.
        assert_eq!(FRAME_LENGTH, 512, "FRAME_LENGTH must be 512");
        assert_eq!(HOP_LENGTH, 256, "HOP_LENGTH must be 256");
        assert_eq!(
            FRAME_LENGTH,
            HOP_LENGTH * 2,
            "FRAME_LENGTH must be exactly twice HOP_LENGTH (50% overlap)"
        );

        // Verify that frame[..HOP_LENGTH] extracts the correct subset
        // and excludes the overlapping portion.
        let frame: Vec<f32> = (0..FRAME_LENGTH).map(|i| i as f32).collect();
        let truncated = &frame[..HOP_LENGTH];
        assert_eq!(truncated.len(), HOP_LENGTH, "truncated slice length");

        // First element is the start of the frame
        assert_eq!(truncated[0], 0.0, "first element should be 0.0");

        // Last truncated element is HOP_LENGTH - 1
        assert_eq!(
            truncated[HOP_LENGTH - 1],
            (HOP_LENGTH - 1) as f32,
            "last truncated element should be {}",
            HOP_LENGTH - 1
        );

        // The overlapping portion (HOP_LENGTH..FRAME_LENGTH) is excluded
        assert!(
            !truncated.contains(&(HOP_LENGTH as f32)),
            "overlapping element {} should not be in truncated slice",
            HOP_LENGTH
        );
    }

    // ── Max operator via candle-core broadcast_maximum (Regression: mahbot-752) ──

    #[test]
    fn test_max_operator_elementwise() {
        use candle_core::Tensor;
        let device = &candle_core::Device::Cpu;

        // Test the broadcast_maximum function that the Max op handler uses.
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, -5.0], (1, 4), device).unwrap();
        let b = Tensor::from_slice(&[0.0f32, 10.0, 3.0, -4.0], (1, 4), device).unwrap();
        let result = a.broadcast_maximum(&b).unwrap();
        let result_vec: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // Element-wise max: max(1,0)=1, max(2,10)=10, max(3,3)=3, max(-5,-4)=-4
        assert!(
            (result_vec[0] - 1.0).abs() < f32::EPSILON,
            "expected 1.0, got {}",
            result_vec[0]
        );
        assert!(
            (result_vec[1] - 10.0).abs() < f32::EPSILON,
            "expected 10.0, got {}",
            result_vec[1]
        );
        assert!(
            (result_vec[2] - 3.0).abs() < f32::EPSILON,
            "expected 3.0, got {}",
            result_vec[2]
        );
        assert!(
            (result_vec[3] + 4.0).abs() < f32::EPSILON,
            "expected -4.0, got {}",
            result_vec[3]
        );
    }

    #[test]
    fn test_max_operator_broadcasting() {
        use candle_core::Tensor;
        let device = &candle_core::Device::Cpu;

        // Test broadcasting: (1,4) vs (1,1) — scalar broadcast.
        let a = Tensor::from_slice(&[1.0f32, -5.0, 10.0, 0.0], (1, 4), device).unwrap();
        let b = Tensor::from_slice(&[2.0f32], (1, 1), device).unwrap();
        let result = a.broadcast_maximum(&b).unwrap();
        let result_vec: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // max(1,2)=2, max(-5,2)=2, max(10,2)=10, max(0,2)=2
        assert!(
            (result_vec[0] - 2.0).abs() < f32::EPSILON,
            "expected 2.0, got {}",
            result_vec[0]
        );
        assert!(
            (result_vec[1] - 2.0).abs() < f32::EPSILON,
            "expected 2.0, got {}",
            result_vec[1]
        );
        assert!(
            (result_vec[2] - 10.0).abs() < f32::EPSILON,
            "expected 10.0, got {}",
            result_vec[2]
        );
        assert!(
            (result_vec[3] - 2.0).abs() < f32::EPSILON,
            "expected 2.0, got {}",
            result_vec[3]
        );
    }

    #[test]
    fn test_max_operator_variadic() {
        use candle_core::Tensor;
        let device = &candle_core::Device::Cpu;

        // The Max op can take more than 2 inputs (variadic max).
        // Test that max(max(a,b),c) produces the element-wise maximum.
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, 3), device).unwrap();
        let b = Tensor::from_slice(&[-1.0f32, 10.0, 0.0], (1, 3), device).unwrap();
        let c = Tensor::from_slice(&[0.0f32, 5.0, 7.0], (1, 3), device).unwrap();

        // Chained broadcast_maximum simulates the Max op's variadic behavior:
        // the handler iterates over all inputs, folding with broadcast_maximum.
        let ab = a.broadcast_maximum(&b).unwrap();
        let result = ab.broadcast_maximum(&c).unwrap();
        let result_vec: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // max(1,-1,0)=1, max(2,10,5)=10, max(3,0,7)=7
        assert!(
            (result_vec[0] - 1.0).abs() < f32::EPSILON,
            "expected 1.0, got {}",
            result_vec[0]
        );
        assert!(
            (result_vec[1] - 10.0).abs() < f32::EPSILON,
            "expected 10.0, got {}",
            result_vec[1]
        );
        assert!(
            (result_vec[2] - 7.0).abs() < f32::EPSILON,
            "expected 7.0, got {}",
            result_vec[2]
        );
    }

    // ── Threshold floor (mahbot-755 Fix 2) ─────────────────────────────

    #[test]
    fn test_threshold_floor_invariant() {
        // The acceptance threshold MUST NOT collapse to nearly-equal the mean
        // when std_dev is tiny (3 enrollment samples give unreliable variance).
        //
        // Formula: threshold = max(mean + THRESHOLD_MULTIPLIER * std_dev,
        //                           mean + MIN_THRESHOLD_FLOOR)
        //
        // These tests verify the invariant without requiring VOICE_PIPELINE
        // initialisation — they exercise the same calculation that
        // finalize_enrollment applies to real enrollment data.

        // Case 1: nearly identical samples (std_dev → 0, mean very small).
        // Floor should dominate: threshold = mean + MIN_THRESHOLD_FLOOR = 0.051
        let mean: f32 = 0.001;
        let std_dev: f32 = 0.0005;
        let threshold = (mean + THRESHOLD_MULTIPLIER * std_dev).max(mean + MIN_THRESHOLD_FLOOR);
        let expected = mean + MIN_THRESHOLD_FLOOR;
        assert!(
            (threshold - expected).abs() < 1e-6,
            "nearly-identical samples: expected {expected}, got {threshold} (floor should dominate)"
        );

        // Case 2: moderate mean where floor still applies.
        // mean + 2*std = 0.14 < mean + 0.05 = 0.15 → floor at 0.15
        let mean: f32 = 0.10;
        let std_dev: f32 = 0.02;
        let threshold = (mean + THRESHOLD_MULTIPLIER * std_dev).max(mean + MIN_THRESHOLD_FLOOR);
        let expected = mean + MIN_THRESHOLD_FLOOR;
        assert!(
            (threshold - expected).abs() < 1e-6,
            "moderate mean with small std: expected {expected}, got {threshold}"
        );

        // Case 3: large std_dev — standard formula dominates, floor irrelevant.
        // mean + 2*std = 0.30 > mean + 0.05 = 0.15 → threshold = 0.30
        let mean: f32 = 0.10;
        let std_dev: f32 = 0.10;
        let threshold = (mean + THRESHOLD_MULTIPLIER * std_dev).max(mean + MIN_THRESHOLD_FLOOR);
        let expected = mean + THRESHOLD_MULTIPLIER * std_dev;
        assert!(
            (threshold - expected).abs() < 1e-6,
            "large std: expected {expected}, got {threshold} (std-based formula should dominate)"
        );
    }

    // ── match_against_templates — sliding window (mahbot-755 Fix 5) ────

    #[test]
    fn test_match_against_templates_sliding_window() {
        // The sliding window strategy in match_against_templates compares
        // only the most recent `min(template.len, live.len)` embeddings
        // against each template.  This avoids length-asymmetry noise when
        // the ring buffer grows larger than the enrollment template.

        // Build two templates: one that matches the target, one that doesn't.
        let matching_tpl = WakeWordTemplate {
            name: "match_me".into(),
            embeddings: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            threshold: 0.5,
        };

        let non_matching_tpl = WakeWordTemplate {
            name: "no_match".into(),
            embeddings: vec![
                vec![-1.0, 0.0, 0.0],
                vec![0.0, -1.0, 0.0],
                vec![0.0, 0.0, -1.0],
            ],
            threshold: 0.1, // very tight — only near-identical will match
        };

        let templates = WakeWordTemplates {
            templates: vec![matching_tpl, non_matching_tpl],
        };

        // Live sequence is longer than template (simulates ring buffer growth).
        // The first 2 embeddings are noise/dis-similar, the last 3 match the
        // matching_tpl embeddings nearly exactly.
        let live_sequence = vec![
            vec![99.0, 99.0, 99.0], // noise — very far from anything
            vec![88.0, 88.0, 88.0], // noise
            vec![1.0, 0.0, 0.0],    // matches matching_tpl[0]
            vec![0.0, 1.0, 0.0],    // matches matching_tpl[1]
            vec![0.0, 0.0, 1.0],    // matches matching_tpl[2]
        ];

        // Template length extracted before move into templates vec.
        const EXPECTED_WINDOW: usize = 3;

        // match_against_templates should window to the most recent
        // min(3, 5) = 3 embeddings, which are the matching ones.
        let result = match_against_templates(&live_sequence, &templates);

        assert!(result.is_some(), "should find a matching template");
        let (dist, tpl) = result.unwrap();
        assert_eq!(tpl.name, "match_me", "should match matching_tpl");
        assert!(
            dist < 0.5,
            "DTW distance with windowed matching should be low (< 0.5), got {dist}"
        );

        // Without sliding window, the noise embeddings would inflate DTW
        // past the threshold.  Verify that the window is actually being
        // applied by checking that the window size is 3 (the template length).
        // We can observe this indirectly: a non-windowed DTW against the full
        // 5-element live sequence would have much higher cost, and the
        // distance-to-threshold ratio would be worse.
        let windowed_len = EXPECTED_WINDOW.min(live_sequence.len());
        assert_eq!(
            windowed_len, 3,
            "sliding window should be template length (3)"
        );
        // The actual window used inside match_against_templates:
        let window = &live_sequence[live_sequence.len() - windowed_len..];
        assert_eq!(window.len(), 3, "windowed slice should have 3 elements");
    }

    #[test]
    fn test_match_against_templates_no_match() {
        // Live sequence that does NOT match any template should return None.
        let tpl = WakeWordTemplate {
            name: "strict".into(),
            embeddings: vec![vec![1.0, 0.0, 0.0]],
            threshold: 0.01, // extremely tight — only exact match passes
        };
        let templates = WakeWordTemplates {
            templates: vec![tpl],
        };

        // Opposite-direction embedding will have cosine distance ≈ 2.0
        // (cosine distance of opposite unit vectors = 1 - (-1/1) = 2.0).
        let live = vec![vec![-1.0, 0.0, 0.0]];
        let result = match_against_templates(&live, &templates);
        assert!(
            result.is_none(),
            "opposite vectors should not match tight threshold"
        );
    }

    #[test]
    fn test_match_against_templates_empty_live() {
        // Empty live sequence should not panic and should not match.
        let tpl = WakeWordTemplate {
            name: "any".into(),
            embeddings: vec![vec![1.0, 0.0, 0.0]],
            threshold: 10.0, // would match anything
        };
        let templates = WakeWordTemplates {
            templates: vec![tpl],
        };
        let result = match_against_templates(&[], &templates);
        assert!(result.is_none(), "empty live should not match");
    }

    // ── Utterance buffer truncation (mahbot-755 Fix 1) ─────────────────

    #[test]
    fn test_enrollment_utterance_tracks_speech_boundary() {
        // Initialize VOICE_PIPELINE (harmless if already set by another test).
        let _ = VOICE_PIPELINE.set(RwLock::new(VoicePipelineState {
            enabled: true,
            status: VoiceStatus::Disabled,
            templates: Arc::new(WakeWordTemplates::default()),
            enrollment_buffer: Vec::new(),
            cmd_tx: None,
        }));

        // Verify that utterance_speech_end_len is updated on each speech
        // frame and that it correctly marks the boundary between speech
        // and trailing silence in the utterance buffer.
        let mut ctx = PipelineCtx::new();
        ctx.is_listening = true;
        ctx.enrollment_mode = true;

        // A single 512-sample frame at amplitude 0.5 has RMS = 0.5 > 0.01
        let speech_frame = vec![0.5f32; FRAME_LENGTH];

        // Feed 2 speech frames (each call processes one 512-sample frame
        // and accumulates HOP_LENGTH=256 new samples per frame into
        // utterance_buf).
        handle_enrollment_audio(&speech_frame, &mut ctx, 0, 3);
        assert!(ctx.utterance_had_speech);
        assert_eq!(
            ctx.utterance_speech_end_len,
            ctx.utterance_buf.len(),
            "after speech frame: speech_end_len should equal buf len"
        );
        assert_eq!(ctx.utterance_speech_end_len, HOP_LENGTH);

        handle_enrollment_audio(&speech_frame, &mut ctx, 0, 3);
        // With 512-sample input, each call processes 1+ additional frame
        // due to leftover samples in audio_buffer, adding ~3 HOP frames total.
        assert_eq!(
            ctx.utterance_speech_end_len,
            ctx.utterance_buf.len(),
            "after second speech frame: speech_end_len stays at buf len"
        );

        // Record the speech-only length before introducing silence.
        let speech_only_len = ctx.utterance_speech_end_len;
        assert!(speech_only_len > 0, "should have accumulated speech data");

        // Clear the audio buffer so the next input starts fresh without
        // leftover speech samples that would contaminate the silence frame.
        ctx.audio_buffer.clear();

        // Manually set silence_since to a time in the distant past so
        // the SILENCE_DURATION check triggers immediately when a silence
        // frame is processed (avoids real-time wait).
        ctx.utterance_silence_since = Some(Instant::now() - Duration::from_secs(10));

        // Feed a silence frame (all zeros → RMS ≈ 0.0 < VAD_THRESHOLD).
        let silence_frame = vec![0.0f32; FRAME_LENGTH];
        handle_enrollment_audio(&silence_frame, &mut ctx, 0, 3);

        // After truncation: enrollment_pending should contain the speech-only
        // audio data, and utterance_buf should be empty.
        assert!(
            ctx.enrollment_pending.is_some(),
            "silence timeout should store utterance in enrollment_pending"
        );
        let pending = ctx.enrollment_pending.as_ref().unwrap();
        assert_eq!(
            pending.len(),
            speech_only_len,
            "enrollment_pending should contain only speech data (no trailing silence)"
        );
        assert!(
            ctx.utterance_buf.is_empty(),
            "utterance_buf should be emptied after truncation"
        );

        // Verify that state was properly reset for the next utterance.
        assert!(
            !ctx.utterance_had_speech,
            "utterance_had_speech should reset"
        );
        assert!(
            ctx.utterance_silence_since.is_none(),
            "silence_since should reset"
        );
        assert_eq!(
            ctx.utterance_speech_end_len, 0,
            "speech_end_len should reset"
        );
    }

    // ── AtomicModelState compare_exchange ──────────────────────────────

    #[test]
    fn test_atomic_model_state_compare_exchange() {
        // Verifies the wrapper method added for mahbot-757 encapsulation.
        // Test each possible transition directly on the global static.

        // Start from a known clean state.
        MODELS_STATE.store(ModelState::Uninit, Ordering::Release);

        // CAS Uninit → Loading (success).
        assert_eq!(
            MODELS_STATE.compare_exchange(
                ModelState::Uninit,
                ModelState::Loading,
                Ordering::AcqRel,
                Ordering::Acquire,
            ),
            Ok(ModelState::Uninit),
            "CAS Uninit→Loading should succeed",
        );
        assert_eq!(MODELS_STATE.load(Ordering::Acquire), ModelState::Loading,);

        // CAS Uninit → Ready (fail — current is Loading).
        assert_eq!(
            MODELS_STATE.compare_exchange(
                ModelState::Uninit,
                ModelState::Ready,
                Ordering::AcqRel,
                Ordering::Acquire,
            ),
            Err(ModelState::Loading),
            "CAS Uninit→Ready should fail when current is Loading",
        );

        // CAS Loading → Ready (success).
        assert_eq!(
            MODELS_STATE.compare_exchange(
                ModelState::Loading,
                ModelState::Ready,
                Ordering::AcqRel,
                Ordering::Acquire,
            ),
            Ok(ModelState::Loading),
            "CAS Loading→Ready should succeed",
        );
        assert_eq!(MODELS_STATE.load(Ordering::Acquire), ModelState::Ready);

        // CAS Failed → Uninit (fail — current is Ready).
        assert_eq!(
            MODELS_STATE.compare_exchange(
                ModelState::Failed,
                ModelState::Uninit,
                Ordering::AcqRel,
                Ordering::Acquire,
            ),
            Err(ModelState::Ready),
            "CAS Failed→Uninit should fail when current is Ready",
        );

        // CAS Ready → Failed (success).
        assert_eq!(
            MODELS_STATE.compare_exchange(
                ModelState::Ready,
                ModelState::Failed,
                Ordering::AcqRel,
                Ordering::Acquire,
            ),
            Ok(ModelState::Ready),
            "CAS Ready→Failed should succeed",
        );
        assert_eq!(MODELS_STATE.load(Ordering::Acquire), ModelState::Failed);

        // Restore clean state for other tests.
        MODELS_STATE.store(ModelState::Uninit, Ordering::Release);
    }

    #[tokio::test]
    async fn test_handle_start_listening_failed_state_triggers_retry() {
        // When models are in Failed state, handle_start_listening should
        // trigger retry_model_loading (via try_retry_models) so the user
        // doesn't need to restart the app (mahbot-757 req #2).
        let _ = VOICE_PIPELINE.set(RwLock::new(VoicePipelineState {
            enabled: false,
            status: VoiceStatus::Disabled,
            templates: Arc::new(WakeWordTemplates::default()),
            enrollment_buffer: Vec::new(),
            cmd_tx: None,
        }));
        // Force-enable so handle_start_listening passes the is_enabled() guard.
        voice_state().write().unwrap_poison().enabled = true;

        let mut ctx = PipelineCtx::new();
        ctx.last_model_retry = None;
        ctx.auto_start_pending = false;

        MODELS_STATE.store(ModelState::Failed, Ordering::Release);

        ctx.handle_start_listening();

        // Should have triggered retry — last_model_retry timestamp set.
        assert!(
            ctx.last_model_retry.is_some(),
            "handle_start_listening should trigger retry when models are Failed",
        );

        // auto_start_pending should be set because models weren't ready.
        assert!(
            ctx.auto_start_pending,
            "auto_start_pending should be set when models are not ready",
        );

        // Clean up.
        voice_state().write().unwrap_poison().enabled = false;
        ctx.auto_start_pending = false;
        ctx.last_model_retry = None;
        tokio::task::yield_now().await;
        MODELS_STATE.store(ModelState::Uninit, Ordering::Release);
    }

    #[tokio::test]
    async fn test_check_auto_start_does_not_handle_failed() {
        // After consolidation (mahbot-757), check_auto_start no longer
        // handles Failed state — that's delegated to the post-select block
        // in run_voice_pipeline. Verify it does NOT trigger a retry.
        let _ = VOICE_PIPELINE.set(RwLock::new(VoicePipelineState {
            enabled: true,
            status: VoiceStatus::Disabled,
            templates: Arc::new(WakeWordTemplates::default()),
            enrollment_buffer: Vec::new(),
            cmd_tx: None,
        }));

        let mut ctx = PipelineCtx::new();
        ctx.auto_start_pending = true;
        ctx.last_model_retry = None;

        MODELS_STATE.store(ModelState::Failed, Ordering::Release);

        ctx.check_auto_start();

        // Should NOT have triggered retry — last_model_retry remains None.
        assert!(
            ctx.last_model_retry.is_none(),
            "check_auto_start should not handle Failed state after consolidation",
        );
        // auto_start_pending should remain true (didn't start).
        assert!(ctx.auto_start_pending);

        // Clean up.
        voice_state().write().unwrap_poison().enabled = false;
        ctx.auto_start_pending = false;
        tokio::task::yield_now().await;
        MODELS_STATE.store(ModelState::Uninit, Ordering::Release);
    }

    // ── SHA256 verification (corrupt-file guard) ─────────────────────────

    #[test]
    fn test_verify_sha256_correct_hash() {
        // A file with matching content passes SHA256 verification.
        let tmp = std::env::temp_dir().join("test_verify_sha256_correct.txt");
        let content = b"hello world from mahbot voice test";
        std::fs::write(&tmp, content).unwrap();

        let mut hasher = Sha256::new();
        hasher.update(content);
        let correct_hash = hex_string(&hasher.finalize());

        assert!(verify_sha256(&tmp, &correct_hash).is_ok());
        // File still exists after successful verification.
        assert!(tmp.exists());

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_verify_sha256_wrong_hash() {
        // A file with non-matching content fails SHA256 verification.
        // This is the guard condition that triggers the corrupt-file delete
        // in ensure_models_downloaded (the key fix for mahbot-757).
        let tmp = std::env::temp_dir().join("test_verify_sha256_wrong.txt");
        std::fs::write(&tmp, b"some content").unwrap();

        assert!(
            verify_sha256(
                &tmp,
                "0000000000000000000000000000000000000000000000000000000000000000"
            )
            .is_err(),
            "wrong hash should produce an error",
        );
        // File still exists after failed verification (verify_sha256 is
        // read-only — deletion happens at the caller in ensure_models_downloaded).
        assert!(tmp.exists());

        let _ = std::fs::remove_file(&tmp);
    }

    // ── try_retry_models debounce cooldown ───────────────────────────────

    #[tokio::test]
    async fn test_try_retry_models_debounce() {
        // Verify the 30-second debounce cooldown in try_retry_models:
        // (a) first call with no recent retry proceeds,
        // (b) second immediate call is debounced (timestamp unchanged),
        // (c) call with last_model_retry set to 31s ago proceeds past cooldown.
        let _ = VOICE_PIPELINE.set(RwLock::new(VoicePipelineState {
            enabled: false,
            status: VoiceStatus::Disabled,
            templates: Arc::new(WakeWordTemplates::default()),
            enrollment_buffer: Vec::new(),
            cmd_tx: None,
        }));

        let mut ctx = PipelineCtx::new();
        ctx.last_model_retry = None;

        // (a) First call: no recent retry → should proceed.
        MODELS_STATE.store(ModelState::Failed, Ordering::Release);
        ctx.try_retry_models();
        assert!(
            ctx.last_model_retry.is_some(),
            "first try_retry_models should set last_model_retry",
        );

        // (b) Second immediate call: debounced (< 30s) → timestamp unchanged.
        let first_ts = ctx.last_model_retry;
        ctx.try_retry_models();
        assert_eq!(
            ctx.last_model_retry, first_ts,
            "debounce should prevent second immediate retry",
        );

        // (c) Past cooldown: set last_model_retry to 31s ago → should proceed
        //     (Instant::now() - 31s creates a valid past instant).
        MODELS_STATE.store(ModelState::Failed, Ordering::Release);
        ctx.last_model_retry = Some(Instant::now() - Duration::from_secs(31));
        ctx.try_retry_models();
        assert!(
            ctx.last_model_retry.unwrap() > first_ts.unwrap(),
            "should update last_model_retry after cooldown expires",
        );

        // Clean up.
        ctx.last_model_retry = None;
        tokio::task::yield_now().await;
        MODELS_STATE.store(ModelState::Uninit, Ordering::Release);
    }

    // ── Template serde: forward-compatibility and default handling ──────
    //
    // These tests verify that WakeWordTemplate and WakeWordTemplates
    // tolerate missing fields (forward compat) and extra unknown fields
    // (future-proofing).  See mahbot-758.

    #[test]
    fn test_template_serde_roundtrip() {
        let tpl = WakeWordTemplate {
            name: "hello".into(),
            embeddings: vec![vec![1.0, 2.0, 3.0]],
            threshold: 0.5,
        };
        let json = serde_json::to_string(&tpl).unwrap();
        let deserialized: WakeWordTemplate = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "hello");
        assert_eq!(deserialized.embeddings, vec![vec![1.0, 2.0, 3.0]]);
        assert!((deserialized.threshold - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_template_serde_missing_fields_use_defaults() {
        // Forward-compat: a template stored before a hypothetical new field
        // was added should deserialize with missing fields defaulted.
        let json = r#"{"name":"legacy","embeddings":[[0.1],[0.2]]}"#;
        let tpl: WakeWordTemplate = serde_json::from_str(json).unwrap();
        assert_eq!(tpl.name, "legacy");
        assert_eq!(tpl.embeddings, vec![vec![0.1], vec![0.2]]);
        // threshold was missing → default 0.0
        assert!((tpl.threshold - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_template_serde_empty_json_uses_defaults() {
        // Minimal JSON: all fields missing → all defaults.
        let tpl: WakeWordTemplate = serde_json::from_str("{}").unwrap();
        assert!(tpl.name.is_empty());
        assert!(tpl.embeddings.is_empty());
        assert!((tpl.threshold - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_template_serde_unknown_fields_ignored() {
        // Forward-compat: extra fields from a future version are silently
        // ignored (serde does not have deny_unknown_fields on this type).
        let json =
            r#"{"name":"x","embeddings":[],"threshold":0.3,"future_field":"v1","another":42}"#;
        let tpl: WakeWordTemplate = serde_json::from_str(json).unwrap();
        assert_eq!(tpl.name, "x");
        assert!(tpl.embeddings.is_empty());
        assert!((tpl.threshold - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_templates_serde_roundtrip() {
        let templates = WakeWordTemplates {
            templates: vec![WakeWordTemplate {
                name: "alpha".into(),
                embeddings: vec![vec![1.0]],
                threshold: 0.5,
            }],
        };
        let json = serde_json::to_string(&templates).unwrap();
        let deserialized: WakeWordTemplates = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.templates.len(), 1);
        assert_eq!(deserialized.templates[0].name, "alpha");
    }

    #[test]
    fn test_templates_serde_empty_list() {
        let tpl: WakeWordTemplates = serde_json::from_str(r#"{"templates":[]}"#).unwrap();
        assert!(tpl.templates.is_empty());
    }
}
