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

use crate::ChatDirection;
use crate::config::CONFIG;
use crate::util::UnwrapPoison;
use crate::voice_verifier::{EMBEDDING_DIM, VoiceVerifier};
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

/// Hop length between frames (256 samples at 16 kHz).  This constant controls
/// VAD frame iteration stride and silence tracking in the application code.
/// The ONNX mel spectrogram model uses its own internal stride (160 samples =
/// 10ms) — HOP_LENGTH does NOT affect mel frame spacing (mahbot-772).
const HOP_LENGTH: usize = 256;

/// Number of mel bands in the spectrogram.
const NUM_MEL_BANDS: usize = 32;

/// Embedding window: 76 consecutive mel frames (~760ms with the ONNX mel
/// model's 10ms internal stride, not 16ms — see HOP_LENGTH note above).
const EMBEDDING_WINDOW_FRAMES: usize = 76;

/// Maximum command recording duration (30 seconds).
const MAX_RECORD_SECS: usize = 30;

/// Minimum silence duration before stopping command recording.
pub(crate) const SILENCE_DURATION: Duration = Duration::from_millis(1500);

/// Silence threshold in audio samples at 16 kHz.
/// Derived from SILENCE_DURATION × SAMPLE_RATE to prevent silent drift
/// if either constant changes.
const SILENCE_THRESHOLD_SAMPLES: usize =
    (SILENCE_DURATION.as_millis() as usize * SAMPLE_RATE as usize) / 1000;

/// Silence threshold (200ms) before showing "Keep silent to confirm…" UI hint.
/// Intentionally wider than a single frame (16ms) so the UI reliably transitions
/// even under scheduling jitter.
const SILENCE_UI_GATE_SAMPLES: usize = 200 * SAMPLE_RATE as usize / 1000;

/// Duration of non-VAD audio before showing "speak louder" warning during
/// enrollment (~5s).  Derived from SAMPLE_RATE and HOP_LENGTH so the threshold
/// stays correct if frame/hop sizes are adjusted (mahbot-765).
const ENROLLMENT_NO_SPEECH_DURATION: Duration = Duration::from_secs(5);

/// Consecutive non-VAD frame threshold for enrollment no-speech warning.
/// Each frame iteration processes HOP_LENGTH new samples, so the threshold
/// is (duration × sample_rate) / hop_length to be robust against frame/hop
/// size changes.
const ENROLLMENT_NO_SPEECH_TIMEOUT_FRAMES: usize =
    (ENROLLMENT_NO_SPEECH_DURATION.as_millis() as usize * SAMPLE_RATE as usize)
        / (HOP_LENGTH * 1000);

/// Maximum number of download retries.
const MAX_DOWNLOAD_RETRIES: u32 = 10;

/// Default wake word name used by the enrollment pipeline.
/// Making this a named constant ensures that [`handle_enrollment_sample`]
/// and [`finalize_enrollment`] stay in sync if the name ever changes
/// (mahbot-771 Fix 4).
const WAKE_WORD_NAME: &str = "custom";

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
///
/// Increased from 3 to 10 (mahbot-765) to provide more representative
/// statistics for the MAD-based threshold formula
/// `threshold = min(median + K_MAD × mad, max(median × 2, 0.20))`.
/// With only 3 samples, the median absolute deviation is unstable and the
/// threshold margin can collapse below 1%, causing false negatives during
/// live detection.
const NUM_ENROLLMENT_SAMPLES: usize = 10;

/// Minimum length (in audio samples at 16kHz) for a collected ambient audio
/// chunk to be used as a negative verifier training example (mahbot-797).
///
/// Set to 0.5s of audio, which produces ~31 mel frames (padded to 76 for the
/// embedding model).  Chunks shorter than this are discarded — they would be
/// mostly padding/silence and provide negligible discriminative signal.
const MIN_NEGATIVE_AUDIO_LEN: usize = SAMPLE_RATE as usize / 2;

/// MAD multiplier for robust threshold calibration.
///
/// K_MAD = 3.0 is equivalent to ~2σ for normally-distributed data
/// (since MAD ≈ 0.6745×σ for normal distributions, 3.0×MAD ≈ 2.0×σ).
const K_MAD: f32 = 3.0;

// Minimum absolute separation between the median distance and the
// acceptance threshold (REMOVED in mahbot-770 Fix 5, reintroduced as
// MAD_THRESHOLD_FLOOR below in mahbot-775).

/// Minimum threshold floor for MAD-based calibration.
///
/// Prevents overly restrictive thresholds when enrollment samples are
/// extremely consistent (MAD ≈ 0.005 → raw threshold ≈ 0.025).  Without
/// this floor, the user's next-morning voice (slightly different speaking
/// style adding ~0.04 shift in embedding distance) would fail every frame.
///
/// The value 0.10 is chosen to:
/// - Absorb typical voice variation (morning voice, fatigue, etc.) which
///   adds ~0.02-0.05 in DTW distance.
/// - Stay well below the cap (`max(median × 2, 0.20)`) so it only affects
///   overly tight calibrations, not normal ones.
const MAD_THRESHOLD_FLOOR: f32 = 0.10;

/// Threshold for detecting clipping: samples at or above this absolute
/// value are considered clipped (near i16::MAX = 32767 in f32 [-1, 1]).
const ENROLLMENT_QUALITY_CLIPPING_THRESHOLD: f32 = 0.999;

/// Minimum acceptable utterance duration in ms for quality scoring.
pub(crate) const ENROLLMENT_QUALITY_DURATION_MIN_MS: u64 = 400;

/// Maximum acceptable utterance duration in ms for quality scoring.
/// Utterances longer than this may contain too much silence padding.
pub(crate) const ENROLLMENT_QUALITY_DURATION_MAX_MS: u64 = 2000;

/// Fraction of enrollment utterances that must pass self-test (≥8/10).
const ENROLLMENT_QUALITY_SELF_TEST_MIN_FRACTION: f32 = 0.8;

/// Enrollment prompts for multi-position guidance (mahbot-778).
/// Each entry is (prompt_text, count_of_samples_for_this_prompt).
const ENROLLMENT_PROMPTS: &[(&str, usize)] = &[
    ("Say it normally", 3),
    ("Say it a bit further from the mic", 3),
    ("Say it at a slightly different angle", 2),
    ("Say it with your normal morning voice", 2),
];

/// Minimum gap between sorted average DTW distances to consider the
/// distribution bimodal.  DTW distances are in the 0.0–2.0 range (cosine
/// distance), and intra-cluster variation is typically < 0.04 even for
/// lazy utterances.  A gap > 0.04 reliably separates two distinct
/// speaking-style clusters.
const BIMODAL_GAP_THRESHOLD: f32 = 0.04;

/// Maximum size of the raw audio ring buffer (~200ms at 16kHz = 3200
/// samples).  Used during enrollment to capture ~100ms of pre-VAD-trigger
/// and post-speech context so the template includes the onset/offset
/// phonemes that strict enrollment VAD (0.85) excludes.
const RAW_RING_MAX: usize = SAMPLE_RATE as usize / 5;

/// Context padding duration in milliseconds for VAD asymmetry mitigation
/// (mahbot-775 Fix 3).  Used to prepend ~100ms of pre-VAD-trigger context
/// and append ~100ms of post-speech context to enrollment utterances, so
/// the template includes the onset/offset phonemes that strict enrollment
/// VAD (0.85) excludes but live detection (VAD=0.5) includes.
const CONTEXT_PADDING_MS: usize = 100;

/// Context padding in audio samples at 16 kHz, derived from
/// CONTEXT_PADDING_MS to stay correct if the sample rate is adjusted.
const CONTEXT_PADDING_SAMPLES: usize = (CONTEXT_PADDING_MS * SAMPLE_RATE as usize) / 1000;

// The threshold formula is now:
// `threshold = clamp(median + K_MAD × mad, 0.10, max(median × 2, 0.20))`

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

/// Post-detection cooldown period to prevent rapid consecutive false triggers
/// (mahbot-770 Fix 2).  After a wake word detection, no further detection is
/// attempted for this duration.  Industry reference: Rhasspy Raven uses
/// `refractory_sec=2.0`, openWakeWord uses patience counters.
const WAKE_WORD_COOLDOWN: Duration = Duration::from_secs(3);

/// Sigmoid steepness for per-template soft scoring (mahbot-773).
///
/// k=10 produces a smooth transition near each template's threshold:
/// a distance 0.02 below threshold scores ~0.55, while one 0.02 above
/// scores ~0.45.  This preserves near-binary character for clear matches
/// while smoothly degrading for borderline frames, eliminating the hard
/// binary on/off that caused false rejects.
const SIGMOID_K: f32 = 10.0;

/// Minimum per-frame soft score below which the rolling window is reset
/// entirely (mahbot-773).  Prevents slow accumulation from noise frames
/// while allowing smooth degradation for borderline matches.
const NO_MATCH_RESET_THRESHOLD: f32 = 0.3;

/// Number of recent per-frame scores to keep in the rolling sum window
/// (mahbot-773).  Each frame represents ~128ms of voiced audio, so N=3
/// covers ~384ms — matching the original temporal window but using
/// accumulated weight instead of a strict consecutive binary counter.
const ROLLING_WINDOW_N: usize = 3;

/// Compile-time invariant: EMBEDDING_RING_MAX must be at least
/// ROLLING_WINDOW_N so the ring buffer can supply enough embeddings
/// for matching while the rolling window accumulates scores.
const _: () = assert!(
    EMBEDDING_RING_MAX >= ROLLING_WINDOW_N,
    "EMBEDDING_RING_MAX must be >= ROLLING_WINDOW_N to hold enough embeddings \
     for template matching"
);

/// Factor applied to `minimum_matches × ROLLING_WINDOW_N` to compute the
/// detection threshold (mahbot-773).  At 0.65, the average per-frame soft
/// score must exceed ~65% of the consensus level for detection to fire.
const MATCH_THRESHOLD_FACTOR: f32 = 0.65;

/// Detection threshold for the rolling sum of soft scores (mahbot-773).
/// Computed as: `minimum_matches × ROLLING_WINDOW_N × MATCH_THRESHOLD_FACTOR`.
/// This requires the average per-frame soft score to exceed ~65% of the
/// `minimum_matches` consensus level, providing smooth degradation for
/// borderline frames while preventing noise accumulation.
///
/// # Safety / precision
/// The `usize → f32` casts are safe because both `minimum_matches` and
/// `ROLLING_WINDOW_N` are at most 3 (a trivially small value that fits
/// exactly in f32's 23-bit mantissa).
#[expect(clippy::cast_precision_loss)]
fn match_threshold(minimum_matches: usize) -> f32 {
    (minimum_matches as f32) * (ROLLING_WINDOW_N as f32) * MATCH_THRESHOLD_FACTOR
}

/// Sigmoid function for soft scoring: maps a raw DTW distance through a
/// threshold-centred sigmoid.  Output is ~1.0 when dist ≪ threshold, ~0.5
/// at dist ≈ threshold, and ~0.0 when dist ≫ threshold.
fn sigmoid_score(dist: f32, threshold: f32) -> f32 {
    1.0 / (1.0 + (SIGMOID_K * (dist - threshold)).exp())
}

/// Process a per-frame soft score through the rolling window and determine
/// whether wake word detection should fire (mahbot-773).
///
/// Returns `true` when the rolling sum of recent scores meets or exceeds
/// `match_threshold(minimum_matches)`.  When the incoming score is below
/// [`NO_MATCH_RESET_THRESHOLD`], the window is cleared entirely to prevent
/// slow accumulation from noise.  On detection the score window is NOT
/// cleared here — the caller is responsible for full pipeline cleanup.
///
/// This function is pure with respect to global state: it only reads its
/// parameters and modifies `score_window` in place.  This makes it directly
/// testable without ONNX models or voice pipeline initialization.
fn process_wake_word_score(
    total_score: f32,
    score_window: &mut Vec<f32>,
    minimum_matches: usize,
) -> bool {
    if total_score < NO_MATCH_RESET_THRESHOLD {
        // Far from matching — reset the entire rolling window to prevent
        // slow accumulation from noise.
        if !score_window.is_empty() {
            debug!(
                "Wake word match lost: total_score={total_score:.4} < NO_MATCH_RESET_THRESHOLD \
                 (window reset, had {} scores)",
                score_window.len(),
            );
        }
        score_window.clear();
        false
    } else {
        // Good-enough frame: append score to rolling window.
        score_window.push(total_score);
        // Keep window at most ROLLING_WINDOW_N frames.
        while score_window.len() > ROLLING_WINDOW_N {
            score_window.remove(0);
        }

        let rolling_sum: f32 = score_window.iter().sum();
        let threshold = match_threshold(minimum_matches);

        debug!(
            "Wake word score: total_score={total_score:.4} rolling_sum={rolling_sum:.4}/ \
             threshold={threshold:.2} window={} (M={minimum_matches})",
            score_window.len(),
        );

        if rolling_sum >= threshold {
            info!(
                "Wake word detected! rolling_sum={rolling_sum:.4} >= {threshold:.2} \
                 (window={} scores, M={minimum_matches})",
                score_window.len(),
            );
            true
        } else {
            false
        }
    }
}

/// Sakoe-Chiba band width as a fraction of the longer sequence (mahbot-770
/// Fix 6).  Prevents pathological DTW alignments where a short noise burst
/// is stretched to match the entire template.  Industry standard is 3-5%;
/// we use 5% which is slightly more permissive.
const SAKOE_CHIBA_BAND_FRACTION: f64 = 0.05;

/// Higher VAD threshold for enrollment: only clear, close-mic speech should
/// pass during enrollment to prevent ambient noise (traffic, wind) from
/// contaminating the template (mahbot-772).  The detection VAD threshold
/// stays at 0.5 for responsiveness.
const ENROLLMENT_VAD_THRESHOLD: f32 = 0.85;

/// Minimum consecutive VAD-positive frames before setting utterance_had_speech
/// during enrollment (~48ms at 16ms/frame).  Prevents a single noise spike
/// from starting utterance accumulation (mahbot-772).
const ENROLLMENT_VAD_CONSECUTIVE_REQUIRED: usize = 3;

// ═══════════════════════════════════════════════════════════════════════════
// Neural VAD (Earshot) — replaces RMS-based `is_speech`
// ═══════════════════════════════════════════════════════════════════════════

/// Global Earshot VAD detector instance. Thread-safe behind a mutex because
/// `predict_f32` completes in ~5-6 µs, so lock contention is negligible.
/// The detector has internal state (768-sample ring buffer, pre-emphasis filter,
/// 3-frame feature context) that must be kept in sync with the audio stream.
/// Created once in [`init_global`].
static VAD_DETECTOR: OnceLock<std::sync::Mutex<earshot::Detector>> = OnceLock::new();

/// Earshot VAD threshold: scores >= this are considered speech.
/// The default (0.5) works well across environments.
const VAD_THRESHOLD: f32 = 0.5;

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
    /// Actively enrolled, waiting for the next sample.
    /// `sample` = completed samples, `total` = required, `duration_ms` = most recent
    /// utterance duration in milliseconds (0 if not yet available).
    /// `quality` = per-utterance quality score (None before the first sample).
    Enrolling {
        sample: usize,
        total: usize,
        duration_ms: u64,
        quality: Option<UtteranceQuality>,
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
    /// Number of enrollment samples used to create this template.
    /// Templates enrolled with fewer than [`NUM_ENROLLMENT_SAMPLES`] samples
    /// are invalid and require re-enrollment (mahbot-765).
    #[serde(default)]
    pub enrollment_samples: usize,
}

/// Collection of enrolled wake word templates.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WakeWordTemplates {
    #[serde(default)]
    pub templates: Vec<WakeWordTemplate>,
    /// Minimum number of templates that must match simultaneously for a
    /// detection frame to count.  Defaults to 1 for backward compatibility
    /// with single-template enrollments.  Multi-template consensus (K=3, M=2)
    /// uses minimum_matches=2 so that ≥2 templates must agree.
    #[serde(default = "default_minimum_matches")]
    pub minimum_matches: usize,
    /// Second-stage logistic regression verifier for false-trigger suppression
    /// (mahbot-777).  When [`VoiceVerifier::is_trained`] is false, all frames
    /// pass through (no-op / graceful degradation).
    #[serde(default)]
    pub verifier: crate::voice_verifier::VoiceVerifier,
}

/// Serde default: single-template legacy behavior.
fn default_minimum_matches() -> usize {
    1
}

impl Default for WakeWordTemplates {
    fn default() -> Self {
        Self {
            templates: Vec::new(),
            minimum_matches: 1,
            verifier: VoiceVerifier::default(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Enrollment quality scoring (mahbot-778)
// ═══════════════════════════════════════════════════════════════════════════

/// Quality level for a single enrollment utterance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityLevel {
    Good,       // score > 0.7
    Acceptable, // score 0.4–0.7
    Poor,       // score < 0.4
}

impl QualityLevel {
    fn from_score(score: f32) -> Self {
        if score > 0.7 {
            Self::Good
        } else if score >= 0.4 {
            Self::Acceptable
        } else {
            Self::Poor
        }
    }

    /// Returns a user-facing label for this quality level.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Good => "✅ Good sample — clear and consistent",
            Self::Acceptable => "⚠️ Acceptable — a bit quiet, try speaking closer to the mic",
            Self::Poor => "❌ Poor sample — too much noise, please re-record",
        }
    }
}

/// Per-utterance quality assessment result.
#[derive(Debug, Clone)]
pub struct UtteranceQuality {
    /// Composite quality score 0.0–1.0 (weighted combination of all factors).
    pub score: f32,
    /// Quality level derived from `score`.
    pub level: QualityLevel,
    /// Whether clipping was detected (samples at or near i16::MAX).
    pub clipping_detected: bool,
    /// Utterance duration in milliseconds.
    pub duration_ms: u64,
    /// Estimated signal-to-noise ratio in dB.
    ///
    /// When a pre-speech noise RMS is available (enrollment path via
    /// [`compute_utterance_quality`]), this is the actual SNR computed as
    /// 20*log10(speech_rms / noise_rms) — unbounded, typically 10–50 dB
    /// in quiet rooms.  When the energy-based fallback ([`estimate_snr_energy`])
    /// is used (tests or edge cases where no noise measurement exists), the
    /// value is 0–40 dB or NaN if the utterance is too short for meaningful
    /// estimation.
    pub snr_db: f32,
    /// Average DTW distance to all other enrollment utterances (lower = more
    /// consistent).  Set to 0.0 when there are fewer than 2 utterances.
    pub avg_dtw_distance: f32,
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
    /// Raw audio chunks collected during non-wake-word periods of enrollment
    /// (pre-enrollment ambient noise and inter-utterance silence).  These are
    /// processed through the ONNX embedding model at verifier training time to
    /// produce real (non-synthetic) negative examples for the verifier (mahbot-797).
    negative_audio_chunks: Vec<Vec<f32>>,
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
    VAD_DETECTOR.get_or_init(|| std::sync::Mutex::new(earshot::Detector::default()));

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
            negative_audio_chunks: Vec::new(),
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

/// Pad a sequence of mel spectrogram frames to exactly [`EMBEDDING_WINDOW_FRAMES`]
/// by appending a **tapered fade-out** toward silence instead of constant-value
/// silence frames.
///
/// # Problem (mahbot-798)
///
/// The previous implementation appended identical `spec_transform(0.0) = 2.0`
/// frames for all padding.  This produced an embedding tail that was **identical
/// regardless of acoustic content** for any audio shorter than 76 frames:
///
/// 1. **False triggers:** short audio + silence produced an embedding highly
///    similar to the enrolled template (which also had a silence-padded tail),
///    artificially lowering DTW distance.
/// 2. **Out-of-distribution input:** the embedding model was trained on real
///    mel spectrograms, not constant-valued blocks.  A block of identical 2.0
///    frames is not representative of speech or silence in natural audio.
///
/// # Fix
///
/// Instead of appending identical silence frames, **linearly taper** from the
/// last real mel frame toward the silence value (`spec_transform(0.0) = 2.0`)
/// over the `frames_needed` padding frames.  This creates a smooth transition
/// that:
///
/// - Preserves **continuity** from the real audio (no abrupt value jump).
/// - Avoids the **identical-tail** contamination (each padding frame differs
///   slightly, so the tail encodes ≈0 acoustic energy rather than a constant).
/// - Remains **in-distribution** for the embedding model (natural decay of
///   acoustic energy toward the noise floor).
///
/// If `frames` is empty (no audio at all), constant silence frames are used
/// as a fallback — there is no last frame to taper from.
///
/// If `frames` already has at least `EMBEDDING_WINDOW_FRAMES`, it is returned
/// as-is (no truncation — the caller decides the window).  This is extracted
/// as a shared helper to avoid duplicating the padding logic in both
/// [`extract_embeddings_from_audio`] and [`try_match_wake_word_and_push_embedding`].
#[allow(clippy::cast_precision_loss)]
fn pad_mel_frames_to_window(frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if frames.len() >= EMBEDDING_WINDOW_FRAMES {
        return frames.to_vec();
    }

    let frames_needed = EMBEDDING_WINDOW_FRAMES - frames.len();
    let silence_val = spec_transform(0.0);

    if frames.is_empty() {
        // No frames to taper from — fall back to constant silence padding.
        let silence_frame = vec![silence_val; NUM_MEL_BANDS];
        return vec![silence_frame; EMBEDDING_WINDOW_FRAMES];
    }

    // Tapered fade-out: linearly interpolate from the last real frame's values
    // toward the silence value over `frames_needed` padding frames.  Alpha goes
    // from ~0 (first padding frame ≈ last real frame) to ~1 (last padding frame
    // ≈ silence), providing a smooth transition.
    let last_frame = frames.last().expect("non-empty — checked above");
    let inv_count = 1.0 / (frames_needed + 1) as f32;

    let mut padded = frames.to_vec();
    padded.reserve(frames_needed);

    for i in 0..frames_needed {
        let alpha = (i + 1) as f32 * inv_count; // (i+1)/(frames_needed+1), range (0, 1)
        let frame: Vec<f32> = last_frame
            .iter()
            .map(|&v| v * (1.0 - alpha) + silence_val * alpha)
            .collect();
        padded.push(frame);
    }

    padded
}

/// Extract a sequence of embeddings from raw audio by processing sliding windows.
fn extract_embeddings_from_audio(models: &OnnxModels, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
    let mel_frames = compute_mel_spectrogram(models, samples)?;

    if mel_frames.len() < EMBEDDING_WINDOW_FRAMES {
        // Audio too short for a full 76-frame window — pad with tapered
        // fade-out frames so at least one embedding can be computed.  Without
        // this, short wake words (e.g. 0.5s) would be silently discarded during
        // enrollment, making enrollment impossible for brief utterances.
        let padded = pad_mel_frames_to_window(&mel_frames);
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

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn mean_embedding(seq: &[Vec<f32>]) -> Vec<f32> {
    assert!(
        !seq.is_empty(),
        "mean_embedding requires non-empty sequence"
    );
    let dim = seq[0].len();
    let mut mean = vec![0.0f32; dim];
    for emb in seq {
        for (i, v) in emb.iter().enumerate() {
            mean[i] += v;
        }
    }
    let n = seq.len() as f32;
    for v in &mut mean {
        *v /= n;
    }
    mean
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn dtw_distance(live: &[Vec<f32>], template: &[Vec<f32>]) -> f32 {
    if live.is_empty() || template.is_empty() {
        return f32::MAX;
    }

    let n = live.len();
    let m = template.len();

    // Sakoe-Chiba band constraint: limit warping to at most `window` steps
    // away from the diagonal.  This prevents pathological alignments where
    // a short noise burst is stretched to match a much longer template.
    // Industry standard is 3-5% of the longer sequence length; we use
    // SAKOE_CHIBA_BAND_FRACTION (5%).
    //
    // The floor of 2 (mahbot-774) ensures that moderate length differences
    // (e.g. 2 vs 4 embeddings) still allow the endpoint to be reached.
    // Using n.max(m) means the band always scales with the longer sequence
    // regardless of which argument is live vs template.
    //
    // NOTE: [`calibrate_threshold`] computes DTW in both directions and takes
    // the asymmetric minimum (d1.min(d2) in Step 2) when comparing enrollment
    // samples of different lengths.  This means the calibration path is safe
    // even though the band constrains alignment — if the band in one direction
    // is too restrictive (short sample vs long template), the reverse direction
    // may still produce a valid alignment.  A future reader modifying the
    // calibration path should preserve this bidirectional safety check.
    let window = (SAKOE_CHIBA_BAND_FRACTION * n.max(m) as f64)
        .ceil()
        .max(2.0) as usize;

    let mut prev = vec![f32::MAX; m];
    let mut curr = vec![f32::MAX; m];
    // Track path length alongside cumulative cost so we can normalise by
    // path length.  Without this, the cumulative distance grows with
    // template length — a 2-second utterance would always cost more than
    // a 0.3-second one even if they match equally well.  Normalising by
    // path length gives a length-invariant distance so that thresholds
    // calibrated on pairwise DTW between utterances of one length work
    // correctly when the live sequence is a different length.
    let mut prev_len = vec![0usize; m];
    let mut curr_len = vec![0usize; m];

    for (i, live_i) in live.iter().enumerate() {
        // Determine the column range constrained by the Sakoe-Chiba band.
        // |i - j| > window cells are skipped (inherit f32::MAX).
        let j_start = i.saturating_sub(window);
        let j_end = (i + window + 1).min(m);

        for j in j_start..j_end {
            let cost = cosine_distance(live_i, &template[j]);
            if i == 0 && j == 0 {
                curr[j] = cost;
                curr_len[j] = 1;
            } else if i == 0 {
                // First row: only horizontal moves are possible within the band
                curr[j] = if j > j_start {
                    cost + curr[j - 1]
                } else {
                    f32::MAX
                };
                curr_len[j] = if j > j_start { curr_len[j - 1] + 1 } else { 0 };
            } else if j == 0 {
                // First column: only vertical moves are possible within the band
                curr[j] = if i > 0 { cost + prev[j] } else { f32::MAX };
                curr_len[j] = if i > 0 { prev_len[j] + 1 } else { 0 };
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
        // Guard: if DTW could not reach the final column (f32::MAX), log a
        // warning and fall back to mean-embedding cosine distance so the
        // caller does not silently receive infinity (mahbot-774).
        warn!(
            "DTW failed to reach endpoint: live={n}, template={m}, window={window}. \
             Falling back to mean-embedding cosine distance."
        );
        let mean_live = mean_embedding(live);
        let mean_template = mean_embedding(template);
        cosine_distance(&mean_live, &mean_template)
    }
}

/// Compute a soft total score across all enrolled templates against the
/// current live embedding sequence (mahbot-773).
///
/// Each template contributes a sigmoid score (0.0–1.0) based on DTW distance
/// relative to its threshold, replacing the old binary match/miss.  Scores
/// are summed across all K templates to produce a continuous `total_score`.
/// The sliding window (most recent `tpl.embeddings.len()` embeddings) is
/// still applied to avoid length-asymmetry noise (mahbot-755 Fix 5).
///
/// Logs DTW distances and per-template scores at debug level.
fn score_matching_templates(live_sequence: &[Vec<f32>], templates: &WakeWordTemplates) -> f32 {
    let mut total_score = 0.0;

    for tpl in &templates.templates {
        let window_len = tpl.embeddings.len().min(live_sequence.len());
        let window = &live_sequence[live_sequence.len() - window_len..];

        let dist = dtw_distance(window, &tpl.embeddings);
        let score = sigmoid_score(dist, tpl.threshold);
        debug!(
            "DTW: template='{}' window={window_len} dist={dist:.4} threshold={} score={score:.4}",
            tpl.name, tpl.threshold,
        );
        total_score += score;
    }

    total_score
}

// ═══════════════════════════════════════════════════════════════════════════
// Audio utilities
// ═══════════════════════════════════════════════════════════════════════════

fn is_speech(samples: &[f32]) -> bool {
    let detector = VAD_DETECTOR.get_or_init(|| std::sync::Mutex::new(earshot::Detector::default()));
    let mut detector = detector.lock().unwrap_poison();
    is_speech_with_detector(samples, &mut detector, VAD_THRESHOLD)
}

/// Inner VAD check using an explicit detector reference and configurable
/// threshold.  Used by [`is_speech`] (with [`VAD_THRESHOLD`]) and by tests
/// that want to supply their own detector to avoid cross-test contamination.
///
/// Processes ALL 256-sample chunks through the detector to keep its internal
/// state (ring buffer + pre-emphasis filter) synchronized with the audio
/// stream, even when speech is detected early in the frame (mahbot-771 Fix 2).
fn is_speech_with_detector(
    samples: &[f32],
    detector: &mut earshot::Detector,
    threshold: f32,
) -> bool {
    if samples.is_empty() {
        return false;
    }

    let mut any_speech = false;

    // Process each complete 256-sample frame (Earshot requires exactly 256
    // samples per call at 16 kHz).  A typical call receives 512-sample frame
    // (FRAME_LENGTH) from the wake-word / enrollment paths, which naturally
    // splits into two 256-sample chunks.  Always process both chunks to keep
    // the detector's sliding window in sync with the actual audio stream.
    for chunk in samples.chunks_exact(256) {
        if detector.predict_f32(chunk) >= threshold {
            any_speech = true;
        }
    }

    // Trailing partial frame (<256 samples) — pad with silence to avoid
    // discarding the tail of a short burst.  Zero-padding is safe because
    // Earshot's neural model correctly rejects silence-padded frames
    // (the spectral pattern is not speech-like).
    let remainder = samples.len() % 256;
    if remainder > 0 {
        let mut padded = [0.0f32; 256];
        padded[..remainder].copy_from_slice(&samples[samples.len() - remainder..]);
        if detector.predict_f32(&padded) >= threshold {
            any_speech = true;
        }
    }

    any_speech
}

/// VAD check with a configurable threshold.  Locks the global [`VAD_DETECTOR`]
/// and delegates to [`is_speech_with_detector`].
fn is_speech_with_threshold(samples: &[f32], threshold: f32) -> bool {
    let detector = VAD_DETECTOR.get_or_init(|| std::sync::Mutex::new(earshot::Detector::default()));
    let mut detector = detector.lock().unwrap_poison();
    is_speech_with_detector(samples, &mut detector, threshold)
}

/// Reset the Earshot VAD detector's internal state (ring buffer, feature
/// context).  Used by tests and when the audio source changes to prevent
/// stale context from contaminating a new stream.
#[doc(hidden)]
pub fn reset_vad() {
    if let Some(detector) = VAD_DETECTOR.get()
        && let Ok(mut d) = detector.lock()
    {
        d.reset();
    }
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

/// Convert multi-channel audio to mono by averaging channels.
/// Kept for test use; production uses the fused [`convert_and_send_audio_to_pipeline`].
#[cfg(test)]
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

/// Convert raw audio samples to mono f32 and send to the pipeline.
///
/// Combines the format conversion and channel-averaging into a single pass,
/// avoiding the intermediate `Vec<f32>` allocation that separate convert-then-
/// to_mono steps would incur.  This reduces allocator pressure in the audio
/// input callback, which runs at audio hardware interrupt frequency.
///
/// When `T = f32` and `convert` is the identity closure `|&s| s`, this
/// function handles the F32 path identically to the integer format paths.
fn convert_and_send_audio_to_pipeline<T, F>(
    tx: &mpsc::UnboundedSender<Vec<f32>>,
    data: &[T],
    channels: u16,
    sample_rate: u32,
    convert: F,
) where
    F: Fn(&T) -> f32,
{
    // Fast path: single channel — no averaging needed, just convert and send.
    if channels == 1 {
        let mono: Vec<f32> = data.iter().map(&convert).collect();
        let resampled = if sample_rate == SAMPLE_RATE {
            mono
        } else {
            resample_audio(&mono, sample_rate, SAMPLE_RATE)
        };
        let _ = tx.send(resampled);
        return;
    }

    let ch = channels as usize;
    let frames = data.len() / ch;
    let remainder = data.len() % ch;
    if remainder != 0 {
        warn!(
            "convert_and_send: discarding {remainder} sample(s) from non-aligned audio (channels={channels})",
        );
    }
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        let start = f * ch;
        let sum: f32 = data[start..start + ch].iter().map(&convert).sum();
        mono.push(sum / f32::from(channels));
    }
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
    // conversion to f32.  Uses the combined convert+to_mono path to avoid
    // an intermediate `Vec<f32>` allocation on every callback.
    macro_rules! build_int_stream {
        ($device:expr, $config:expr, $sample_tx:expr, $channels:expr, $sample_rate:expr, $fmt:ty, $convert:expr) => {{
            let tx = $sample_tx.clone();
            $device.build_input_stream::<$fmt, _, _>(
                &($config).into(),
                move |data, _| {
                    convert_and_send_audio_to_pipeline(
                        &tx,
                        data,
                        $channels,
                        $sample_rate,
                        $convert,
                    );
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
                    // F32 can use the generic path with identity conversion,
                    // benefiting from the single-channel fast-path in
                    // convert_and_send_audio_to_pipeline.
                    convert_and_send_audio_to_pipeline(&tx, data, channels, sample_rate, |&s| s);
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

    // Pre-clean any stale files left from a prior crash so they don't
    // accumulate (ticket mahbot-760).  This is best-effort — if the
    // directory doesn't exist yet, remove_dir_all returns Ok(()).
    let _ = tokio::fs::remove_dir_all(&tmp_dir).await;

    tokio::fs::create_dir_all(&tmp_dir).await?;
    let tmp_path = tmp_dir.join(format!("cmd_{}.wav", crate::generate_id()));
    tokio::fs::write(&tmp_path, &wav_bytes).await?;

    let result = crate::providers::local_transcriber::transcribe_file_async(&tmp_path).await;

    // Remove the specific temp file.
    if let Err(e) = tokio::fs::remove_file(&tmp_path).await {
        warn!("Failed to remove temp transcription file: {e}");
    }
    // Remove the entire temp directory (including any leftover files from
    // prior crashes that weren't cleaned).  Uses remove_dir_all instead of
    // remove_dir so that ENOTEMPTY errors from orphaned files don't cause
    // unbounded accumulation (ticket mahbot-760).
    if let Err(e) = tokio::fs::remove_dir_all(&tmp_dir).await {
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

/// Compute the median of a sorted slice of f32 values.
/// The slice MUST be sorted in ascending order and MUST NOT be empty.
#[inline]
fn median_of_sorted(sorted: &[f32]) -> f32 {
    assert!(!sorted.is_empty(), "median_of_sorted called on empty slice");
    if sorted.len().is_multiple_of(2) {
        let mid = sorted.len() / 2;
        f32::midpoint(sorted[mid - 1], sorted[mid])
    } else {
        sorted[sorted.len() / 2]
    }
}

/// Compute the average DTW distance from each sample to every other sample.
///
/// Returns a vector of length `samples.len()` where the i-th entry is the
/// average (asymmetric min) DTW distance from sample i to all other samples.
/// Used by [`calibrate_threshold`] for best-template selection (GRT strategy).
#[allow(clippy::cast_precision_loss)]
fn compute_avg_dtw_distances(samples: &[Vec<Vec<f32>>]) -> Vec<f32> {
    let n = samples.len();
    assert!(
        n >= 2,
        "compute_avg_dtw_distances requires at least 2 samples, got {n}",
    );
    let mut avg_distances: Vec<f32> = Vec::with_capacity(n);
    for (i, s_i) in samples.iter().enumerate() {
        let mut sum = 0.0f32;
        for (j, s_j) in samples.iter().enumerate() {
            if i != j {
                let d1 = dtw_distance(s_i, s_j);
                let d2 = dtw_distance(s_j, s_i);
                sum += d1.min(d2);
            }
        }
        avg_distances.push(sum / (n - 1) as f32);
    }
    avg_distances
}

/// Detect whether a set of average pairwise DTW distances shows a bimodal
/// distribution (two distinct clusters) with a significant gap between them.
///
/// Returns `Some(split_index)` if bimodal, where indices < `split_index`
/// belong to the first cluster and >= `split_index` belong to the second.
/// Returns `None` if the distribution is unimodal.
///
/// Uses a simple gap-threshold approach: sort the distances, find the
/// largest consecutive gap.  If it exceeds `BIMODAL_GAP_THRESHOLD`, the
/// distribution is considered bimodal.
fn detect_bimodal_gap(sorted_distances: &[f32]) -> Option<usize> {
    if sorted_distances.len() < 4 {
        return None; // too few samples for reliable cluster detection
    }

    let mut max_gap = 0.0f32;
    let mut split_at = None;
    for i in 1..sorted_distances.len() {
        let gap = sorted_distances[i] - sorted_distances[i - 1];
        if gap > max_gap {
            max_gap = gap;
            split_at = Some(i);
        }
    }

    if max_gap > BIMODAL_GAP_THRESHOLD {
        split_at
    } else {
        None
    }
}

/// Compute the MAD (Median Absolute Deviation) of all pairwise DTW distances
/// within a cluster of samples.  Used to determine which cluster is "stricter"
/// (lower MAD = more careful/prototypical speaking style).
///
/// Returns `f32::MAX` for clusters with fewer than 2 samples (cannot compute
/// pairwise distances).
fn compute_cluster_mad(samples: &[Vec<Vec<f32>>], indices: &[usize]) -> f32 {
    if indices.len() < 2 {
        return f32::MAX;
    }
    let mut distances: Vec<f32> = Vec::new();
    for i in 0..indices.len() {
        for j in (i + 1)..indices.len() {
            let d1 = dtw_distance(&samples[indices[i]], &samples[indices[j]]);
            let d2 = dtw_distance(&samples[indices[j]], &samples[indices[i]]);
            distances.push(d1.min(d2));
        }
    }
    if distances.is_empty() {
        return f32::MAX;
    }
    distances.sort_unstable_by(|a, b| a.partial_cmp(b).expect("distances must be finite"));
    let median = median_of_sorted(&distances);
    let mut abs_devs: Vec<f32> = distances.iter().map(|d| (d - median).abs()).collect();
    abs_devs.sort_unstable_by(|a, b| {
        a.partial_cmp(b)
            .expect("absolute deviations must be finite")
    });
    median_of_sorted(&abs_devs)
}

// ═══════════════════════════════════════════════════════════════════════════
// Enrollment quality scoring (mahbot-778)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute a per-utterance quality score from the raw audio and extracted
/// embeddings, comparing against previously collected enrollment samples.
///
/// The composite score (0.0–1.0) is a weighted combination of:
/// - **DTW self-consistency** (50%): average DTW distance to other utterances,
///   lower = more consistent = higher quality.
/// - **Duration** (20%): whether the utterance is in [400ms, 2000ms].
/// - **Clipping** (15%): penalty if any sample hit i16::MAX.
/// - **SNR** (15%): estimated signal-to-noise ratio.  If `noise_rms` is
///   `Some`, uses the real pre-speech noise floor captured from the raw audio
///   ring during enrollment; otherwise falls back to an energy-based heuristic.
///
/// # Parameters
/// - `samples`: raw audio samples of the utterance.
/// - `embeddings`: extracted embeddings for this utterance.
/// - `enrollment_buffer`: all previously collected enrollment embeddings
///   (used for DTW self-consistency comparison).
/// - `noise_rms`: pre-speech ambient noise RMS captured at the moment of
///   first sustained speech detection (mahbot-782).  `None` falls back to
///   energy-based SNR estimation.
#[expect(clippy::cast_precision_loss)]
fn compute_utterance_quality(
    samples: &[f32],
    embeddings: &[Vec<f32>],
    enrollment_buffer: &[Vec<Vec<f32>>],
    noise_rms: Option<f32>,
) -> UtteranceQuality {
    let duration_ms = (samples.len() as u64 * 1000) / u64::from(SAMPLE_RATE);

    // ── Clipping detection ───────────────────────────────────────────
    let clipping_detected = samples
        .iter()
        .any(|&s| s.abs() >= ENROLLMENT_QUALITY_CLIPPING_THRESHOLD);

    // ── SNR estimation ───────────────────────────────────────────────
    // If we have a real pre-speech noise RMS captured from the raw audio
    // ring at the moment of first sustained speech, compute actual SNR as
    // 20*log10(speech_rms / noise_rms).  Otherwise fall back to energy-based
    // heuristic (estimate_snr_energy) which measures speech dynamic range
    // rather than true SNR (mahbot-782).
    let snr_db = if let Some(noise_rms) = noise_rms {
        let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
        let speech_rms = (sum_sq / samples.len() as f32).sqrt();
        if noise_rms > 1e-10 && speech_rms > noise_rms {
            20.0 * (speech_rms / noise_rms).log10()
        } else {
            0.0
        }
    } else {
        estimate_snr_energy(samples)
    };

    // ── DTW self-consistency ─────────────────────────────────────────
    let avg_dtw_distance = if enrollment_buffer.is_empty() {
        // No other utterances to compare against — neutral score.
        0.0
    } else {
        let mut sum = 0.0f32;
        let mut count = 0;
        for other in enrollment_buffer {
            let d1 = dtw_distance(embeddings, other);
            let d2 = dtw_distance(other, embeddings);
            sum += d1.min(d2);
            count += 1;
        }
        sum / count as f32
    };

    // ── Composite score ──────────────────────────────────────────────
    // DTW consistency component: DTW is in range [0, ~2] for cosine
    // distance.  Score = 1.0 - min(dtw / 1.0, 1.0).  So a DTW of 0.0
    // gives 1.0, a DTW of 1.0+ gives 0.0.
    let dtw_score = (1.0f32 - (avg_dtw_distance / 1.0).min(1.0)).max(0.0);

    // Duration score: 0.0 if too short or too long, ramping up in range.
    let duration_score = if duration_ms < ENROLLMENT_QUALITY_DURATION_MIN_MS {
        0.0
    } else if duration_ms > ENROLLMENT_QUALITY_DURATION_MAX_MS {
        0.3 // Long utterances still have some value (contain the wake word)
    } else {
        // Normalize to [0.6, 1.0] within the valid range
        0.6 + (0.4 * (duration_ms - ENROLLMENT_QUALITY_DURATION_MIN_MS) as f32
            / (ENROLLMENT_QUALITY_DURATION_MAX_MS - ENROLLMENT_QUALITY_DURATION_MIN_MS) as f32)
    };

    // Clipping penalty: 1.0 if no clipping, 0.0 if clipping detected.
    let clipping_score = if clipping_detected { 0.0 } else { 1.0 };

    // SNR score: 0.0 at 0 dB, 1.0 at 20+ dB (with smooth ramp).
    let snr_score = if snr_db.is_finite() {
        (snr_db / 20.0).clamp(0.0, 1.0)
    } else {
        // If SNR estimation fails (e.g. all samples silence), give a
        // moderate score — don't penalise the caller for our estimation
        // limitations.
        0.5
    };

    let score = dtw_score * 0.50 + duration_score * 0.20 + clipping_score * 0.15 + snr_score * 0.15;

    UtteranceQuality {
        score,
        level: QualityLevel::from_score(score),
        clipping_detected,
        duration_ms,
        snr_db,
        avg_dtw_distance,
    }
}

/// Estimate SNR using energy-based VAD (no neural model dependency).
///
/// Frames audio into 512-sample windows, computes RMS per frame,
/// classifies the top 40% RMS frames as "speech" and bottom 40% as
/// "noise" (middle 20% is ambiguous transition region).  Returns the
/// ratio in dB, clamped to [0, 40] dB to avoid extreme values from
/// synthetic/test signals.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn estimate_snr_energy(samples: &[f32]) -> f32 {
    if samples.len() < FRAME_LENGTH * 3 {
        return f32::NAN; // Too short for meaningful estimation
    }

    let mut frame_rms: Vec<f32> = Vec::new();
    for chunk in samples.chunks(FRAME_LENGTH) {
        if chunk.len() < FRAME_LENGTH / 2 {
            continue; // Skip partial trailing frames
        }
        let len = chunk.len().min(FRAME_LENGTH) as f32;
        let sum_sq: f32 = chunk.iter().map(|&s| s * s).sum();
        frame_rms.push((sum_sq / len).sqrt());
    }

    if frame_rms.len() < 3 {
        return f32::NAN;
    }

    frame_rms.sort_unstable_by(|a, b| a.partial_cmp(b).expect("RMS values must be finite"));
    let n = frame_rms.len();

    // Bottom 40% = noise floor
    let noise_len = (n as f32 * 0.4).ceil() as usize;
    let noise_rms: f32 = frame_rms[..noise_len.min(n)].iter().sum::<f32>() / noise_len as f32;

    // Top 40% = speech
    let speech_start = (n as f32 * 0.6).ceil() as usize;
    let speech_len = n.saturating_sub(speech_start);
    let speech_rms = if speech_len > 0 {
        frame_rms[speech_start..].iter().sum::<f32>() / speech_len as f32
    } else {
        return f32::NAN;
    };

    if noise_rms <= 1e-10 || speech_rms <= noise_rms {
        return 0.0; // No discernible signal
    }

    let snr = 20.0 * (speech_rms / noise_rms).log10();
    snr.clamp(0.0, 40.0)
}

/// Run a self-test of enrolled templates against the enrollment buffer.
///
/// Simulates the live detection pipeline for each enrollment utterance: feeds
/// embeddings one by one through the embedding ring, calls
/// [`score_matching_templates`] (sequence-level DTW) on each frame, and runs
/// the result through [`process_wake_word_score`] with a rolling window.
///
/// This follows the same matching algorithm used in the real detection loop
/// (DTW, embedding ring, sigmoid scoring), except it hard-codes `minimum_matches=1`
/// rather than using [`WakeWordTemplates::minimum_matches`].  During self-test the
/// utterances are matched against templates calibrated from those same utterances,
/// so DTW distances are near-zero and a single matching frame is sufficient.  Using
/// M=1 avoids the self-test becoming a threshold-sensitivity test (which would fail
/// short utterances that produce only 1–2 embedding frames) while still exercising
/// the core pipeline (DTW, ring buffer, score window accumulation, noise reset).
///
/// An utterance "triggers" if the rolling window sum exceeds [`match_threshold`]
/// at any point.
///
/// Returns `Ok(())` if the self-test passes, or `Err` with a descriptive
/// message if too many utterances fail to trigger detection.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn run_enrollment_self_test(
    enrollment_buffer: &[Vec<Vec<f32>>],
    templates: &WakeWordTemplates,
) -> Result<(), String> {
    if enrollment_buffer.is_empty() || templates.templates.is_empty() {
        return Err("Self-test skipped: no enrollment samples or no templates".to_string());
    }

    let mut passed = 0usize;

    for utterance in enrollment_buffer {
        // Fresh simulation for each utterance: no cross-utterance state.
        let mut embedding_ring: Vec<Vec<f32>> = Vec::with_capacity(EMBEDDING_RING_MAX);
        let mut score_window = Vec::new();
        let mut detected = false;

        for embedding in utterance {
            // Same ring-buffer logic as the live detection loop.
            embedding_ring.push(embedding.clone());
            while embedding_ring.len() > EMBEDDING_RING_MAX {
                embedding_ring.remove(0);
            }

            // Sequence-level DTW matching — identical to the live pipeline.
            let total_score = score_matching_templates(&embedding_ring, templates);
            // Deliberately hard-code M=1 instead of using templates.minimum_matches:
            // during self-test, utterances are matched against templates derived from
            // the same utterances, so DTW distances are near-zero and a single frame
            // suffices.  Using templates.minimum_matches (typically 2) would set the
            // threshold to 3.9, which is unreachable for short utterances that produce
            // only 1–2 embedding frames (mahbot-786).
            if process_wake_word_score(total_score, &mut score_window, 1) {
                detected = true;
                break;
            }
        }

        if detected {
            passed += 1;
        }
    }

    let required = (enrollment_buffer.len() as f32 * ENROLLMENT_QUALITY_SELF_TEST_MIN_FRACTION)
        .ceil() as usize;

    if passed < required {
        Err(format!(
            "Self-test failed: only {passed}/{} utterances triggered detection (need ≥{required}). \
             Try re-enrolling with clearer, more consistent speech.",
            enrollment_buffer.len(),
        ))
    } else {
        info!(
            "Enrollment self-test passed: {passed}/{} utterances triggered detection \
             (threshold ≥{required})",
            enrollment_buffer.len(),
        );
        Ok(())
    }
}

/// Resolve the enrollment prompt for a given sample index (0-based).
/// Returns a static string with guidance for the user's next utterance
/// (e.g. "Say it normally", "Say it further from the mic", etc.).
#[must_use]
pub fn enrollment_prompt_for_sample(sample: usize) -> &'static str {
    let mut cumulative = 0;
    for &(prompt, count) in ENROLLMENT_PROMPTS {
        cumulative += count;
        if sample < cumulative {
            return prompt;
        }
    }
    // Fallback (shouldn't happen with well-formed prompts)
    "Say the wake word clearly"
}

/// Compute the top-K most representative templates and MAD-based thresholds.
///
/// Returns a vector of up to `MAX_TEMPLATES` (K=3) tuples `(embeddings, threshold)`,
/// ranked by how representative each sample is (lowest average pairwise DTW
/// distance to other samples = most representative).
///
/// Each template gets its own individually calibrated threshold using the
/// existing MAD-based formula against the other N-1 samples (excluding itself).
///
/// # Algorithm
///
/// 1. **Rank by representativeness** (GRT strategy): compute the average DTW
///    distance from each sample to all other samples.  Sort by ascending average
///    distance — the most representative sample comes first.
/// 2. **Bimodal cluster correction** (mahbot-775): if the distribution of
///    average distances shows a clear bimodal gap (e.g. one cluster of careful
///    utterances and another of lazy/tired utterances), prefer the cluster with
///    the stricter (lower) intra-cluster MAD.  This prevents the larger lazy
///    cluster from dominating the GRT ranking.  Unimodal distributions use the
///    standard GRT ranking unchanged.
/// 3. **Per-template MAD-based threshold**: for each of the top-K samples,
///    compute DTW distances to every *other* sample.  Compute the median and
///    Median Absolute Deviation of those distances.  Set
///    `threshold = clamp(median + K_MAD × mad, 0.10, max(median × 2, 0.20))`.
/// 4. **Rank preservation**: templates are returned in order of increasing
///    average distance (most representative first), matching the old single-best
///    behavior for index 0.
///
/// The threshold floor (`MAD_THRESHOLD_FLOOR`, 0.10) was introduced in mahbot-775
/// to prevent overly restrictive thresholds from near-identical enrollment samples.
/// The cap (`max(median × 2, 0.20)`) prevents over-permissive thresholds.
///
/// A minimum threshold floor (`MAD_THRESHOLD_FLOOR`, 0.10) was reintroduced
/// in mahbot-775 to prevent overly restrictive thresholds when enrollment
/// samples are extremely consistent (e.g. quiet-room enrollment with MAD ≈
/// 0.005).  Without the floor, the raw MAD-based threshold (~0.025) would
/// reject slightly different voice states like "morning voice" that add
/// ~0.04 shift in embedding distance.  0.10 is well below the 0.20 cap,
/// so it only affects overly tight calibrations.
///
/// This function is pure (no global state) so it can be tested directly.
/// The caller is responsible for providing at least 2 samples.
///
/// NOTE: `MAX_TEMPLATES` is capped at 3 (K=3 per the multi-template consensus
/// design).  When `samples.len() < MAX_TEMPLATES`, all available samples are
/// returned.
const MAX_TEMPLATES: usize = 3;

#[allow(clippy::cast_precision_loss)]
fn calibrate_threshold(samples: &[Vec<Vec<f32>>]) -> Result<Vec<(Vec<Vec<f32>>, f32)>> {
    let n = samples.len();
    if n < 2 {
        anyhow::bail!("Need at least 2 enrollment samples, got {n}");
    }

    // ── Step 1: Rank by average pairwise DTW distance ──
    let avg_distances = compute_avg_dtw_distances(samples);

    // ── Step 1a: Bimodal cluster correction (mahbot-775) ──
    // Detect if the distribution of average distances is bimodal (two distinct
    // clusters, e.g. careful utterances vs lazy/tired utterances).  If so,
    // prefer the cluster with the stricter (lower) intra-cluster MAD so that
    // a larger lazy cluster doesn't dominate the GRT ranking.
    let mut sorted_avg: Vec<f32> = avg_distances.clone();
    sorted_avg.sort_unstable_by(|a, b| a.partial_cmp(b).expect("distances must be finite"));
    let ranked: Vec<usize> = if let Some(split) = detect_bimodal_gap(&sorted_avg) {
        // Split samples into two clusters at the gap.
        let split_threshold = f32::midpoint(sorted_avg[split - 1], sorted_avg[split]);
        let cluster_a: Vec<usize> = (0..n)
            .filter(|&i| avg_distances[i] <= split_threshold)
            .collect();
        let cluster_b: Vec<usize> = (0..n)
            .filter(|&i| avg_distances[i] > split_threshold)
            .collect();
        // Compute MAD within each cluster (lower MAD = stricter = more careful).
        let mad_a = compute_cluster_mad(samples, &cluster_a);
        let mad_b = compute_cluster_mad(samples, &cluster_b);
        // Prefer the stricter (lower MAD) cluster; if equal MAD, prefer
        // the larger cluster (more samples → more representative template).
        let preferred = if mad_a < mad_b {
            &cluster_a
        } else if mad_b < mad_a {
            &cluster_b
        } else if cluster_a.len() >= cluster_b.len() {
            &cluster_a
        } else {
            &cluster_b
        };

        info!(
            "Enrollment bimodal detection: split={split}, cluster_a_size={}, \
             cluster_b_size={}, mad_a={mad_a:.4}, mad_b={mad_b:.4}, \
             preferring cluster with MAD={:.4}",
            cluster_a.len(),
            cluster_b.len(),
            mad_a.min(mad_b),
        );
        // Rank the preferred cluster by ascending average distance.
        let mut preferred_ranked = preferred.clone();
        preferred_ranked.sort_unstable_by(|&a, &b| {
            avg_distances[a]
                .partial_cmp(&avg_distances[b])
                .expect("average distances must be finite")
        });
        preferred_ranked
    } else {
        // Unimodal distribution: use standard GRT ranking.
        let mut ranked: Vec<usize> = (0..n).collect();
        ranked.sort_unstable_by(|&a, &b| {
            avg_distances[a]
                .partial_cmp(&avg_distances[b])
                .expect("average distances must be finite")
        });
        ranked
    };

    // Take the top K (or all if < K available).
    let k = ranked.len().min(MAX_TEMPLATES);
    let top_indices = &ranked[..k];

    let mut results: Vec<(Vec<Vec<f32>>, f32)> = Vec::with_capacity(k);

    for &idx in top_indices {
        let candidate = &samples[idx];

        // ── Step 2: Compute DTW distances from this candidate to all others ──
        let mut distances: Vec<f32> = Vec::with_capacity(n - 1);
        for (j, sample) in samples.iter().enumerate() {
            if j != idx {
                let d1 = dtw_distance(candidate, sample);
                let d2 = dtw_distance(sample, candidate);
                distances.push(d1.min(d2));
            }
        }
        // distances.len() == n - 1, guaranteed >= 1 by the n >= 2 check above.

        // ── Step 3: Compute median and MAD ──
        distances.sort_unstable_by(|a, b| a.partial_cmp(b).expect("distances must be finite"));
        let median = median_of_sorted(&distances);
        let mut abs_devs: Vec<f32> = distances.iter().map(|d| (d - median).abs()).collect();
        abs_devs.sort_unstable_by(|a, b| {
            a.partial_cmp(b)
                .expect("absolute deviations must be finite")
        });
        let mad = median_of_sorted(&abs_devs);

        // ── Step 4: Apply absolute cap and floor ──
        let threshold = median + K_MAD * mad;
        let cap = (median * 2.0).max(0.20);
        let floor = MAD_THRESHOLD_FLOOR;
        let threshold = threshold.clamp(floor, cap);

        info!(
            "Enrollment calibration: candidate={idx}, median={median:.4}, mad={mad:.4}, \
             threshold={threshold:.4} (cap={cap:.4}, floor={floor:.4})"
        );

        results.push((candidate.clone(), threshold));
    }

    info!("Enrollment: {n} samples → {k} template(s): indices={top_indices:?}",);

    Ok(results)
}

/// Finalize enrollment: wraps [`calibrate_threshold`] to build up to K
/// [`WakeWordTemplate`]s from the current [`voice_state`] enrollment buffer.
///
/// Returns the list of templates and the `minimum_matches` consensus count
/// (defaults to `min(num_templates, 2)` so that ≥2 templates must agree when
/// K ≥ 2, and a single template suffices for legacy enrollments).
fn finalize_enrollment(wake_word_name: &str) -> Result<(Vec<WakeWordTemplate>, usize)> {
    let state = voice_state().read().unwrap_poison();
    let templates_data = calibrate_threshold(&state.enrollment_buffer)?;

    let count = templates_data.len();
    let templates: Vec<WakeWordTemplate> = templates_data
        .into_iter()
        .enumerate()
        .map(|(i, (embeddings, threshold))| {
            let name = if i == 0 {
                wake_word_name.to_string()
            } else {
                format!("{wake_word_name}_{}", i + 1)
            };
            WakeWordTemplate {
                name,
                embeddings,
                threshold,
                enrollment_samples: NUM_ENROLLMENT_SAMPLES,
            }
        })
        .collect();

    // Minimum matches: at least 2 for multi-template, 1 for single.
    let minimum_matches = count.min(2);

    // ── Self-test: verify ≥80% of enrollment utterances trigger detection ──
    // Re-processes each enrollment utterance through the live DTW matching
    // pipeline to catch egregious pipeline failures (dimension mismatches,
    // zero-length data, calibration bugs).  Note: since the templates are
    // calibrated FROM these same 10 utterances, the self-test cannot detect
    // generalisation problems where templates work on enrollment data but fail
    // on the user's actual voice at a different mic distance/angle.
    //
    // Known limitation: the self-test does NOT exercise the second-stage
    // logistic-regression verifier (mahbot-777), which is trained AFTER this
    // self-test runs.  A corrupted verifier would pass the self-test but fail
    // in live detection — this is a deliberate trade-off to keep the feedback
    // loop tight during the enrollment UI flow (mahbot-778).
    let ww_templates = WakeWordTemplates {
        templates: templates.clone(),
        minimum_matches,
        ..Default::default()
    };
    if let Err(msg) = run_enrollment_self_test(&state.enrollment_buffer, &ww_templates) {
        warn!("{msg}");
        return Err(anyhow!("{msg}"));
    }

    Ok((templates, minimum_matches))
}

// ═══════════════════════════════════════════════════════════════════════════
// Routing to active agent
// ═══════════════════════════════════════════════════════════════════════════

/// Broadcast a voice transcript to the GUI chat view.
///
/// Delegates to the shared [`broadcast_and_persist_user_message`] when a
/// user identity is available (broadcast + persist).  For anonymous fallback
/// paths (empty `user_name`) only the broadcast is done — inserting a
/// chat_history record with no user identity would create orphaned entries.
async fn broadcast_voice_transcript(transcript: &str, user_name: &str, workspace: &str) {
    if user_name.is_empty() {
        let message_id = crate::generate_id();
        let timestamp = crate::turso::now();
        crate::channels::broadcast_chat_event(
            &message_id,
            "",
            transcript,
            ChatDirection::User,
            "voice",
            None,
            workspace,
            None,
            &timestamp,
        );
    } else {
        crate::channels::broadcast_and_persist_user_message(
            user_name, "voice", transcript, workspace,
        )
        .await;
    }
}

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

        // Broadcast before routing so the transcript appears immediately
        // while the agent is still working.
        broadcast_voice_transcript(&text, &user_name, &ws_name).await;

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

    let active = active_workspace_name();
    if !active.is_empty() {
        info!("Voice command -> Manager (active workspace: {active}): {text}");
        broadcast_voice_transcript(&text, "", &active).await;

        let agent_id = crate::session::manager_agent_id(&active);
        crate::message_router::route(
            &agent_id,
            crate::message_router::AgentJob {
                content: text,
                workspace_name: active,
                user_name: String::new(),
                channel: "voice".to_string(),
                kind: crate::message_router::JobKind::UserMessage,
                role: crate::Role::Manager,
                reply_target: None,
            },
        );
        return;
    }

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
    broadcast_voice_transcript(&text, "", &ws.name).await;

    let agent_id = crate::session::manager_agent_id(&ws.name);
    crate::message_router::route(
        &agent_id,
        crate::message_router::AgentJob {
            content: text,
            workspace_name: ws.name,
            user_name: String::new(),
            channel: "voice".to_string(),
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
    /// Track silence duration by audio sample count rather than wall-clock
    /// time, so that system load / processing delays don't affect recording
    /// cutoff consistency (ticket mahbot-760).
    silence_sample_count: usize,
    enrollment_mode: bool,
    utterance_buf: Vec<f32>,
    utterance_had_speech: bool,
    /// Silence duration in samples for enrollment utterance detection
    /// (sample-based to avoid wall-clock drift under load).
    utterance_silence_samples: usize,
    /// Counter of consecutive non-VAD frames during enrollment.
    /// Used to detect when the user has not spoken for too long
    /// and show a "speak louder" warning (mahbot-765).
    enrollment_no_speech_frame_count: usize,
    /// Consecutive VAD-positive frame counter for enrollment sustained-speech
    /// confirmation (mahbot-772).  Accumulation only starts after this reaches
    /// [`ENROLLMENT_VAD_CONSECUTIVE_REQUIRED`] to reject single noise spikes.
    vad_positives_in_a_row: usize,
    /// VAD threshold for the current mode.  Set to [`VAD_THRESHOLD`] for
    /// detection/recording and [`ENROLLMENT_VAD_THRESHOLD`] for enrollment
    /// (mahbot-772).  Stored in the context so tests can use [`VAD_THRESHOLD`]
    /// without needing the synthetic test signal to score above 0.85.
    vad_threshold: f32,
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
    /// Timestamp of the last wake word detection (mahbot-770 Fix 2).
    /// Used to enforce a cooldown period after detection to prevent rapid
    /// consecutive false triggers.
    last_wake_word_detection: Option<Instant>,
    /// Pre-speech noise RMS captured at the moment of first sustained speech
    /// detection during enrollment.  Computed from the pre-AGC audio ring
    /// ([`pre_agc_ring`]) so AGC's asymmetric gain (4× on silence, ~1-2× on
    /// speech) does not artificially lower the SNR estimate (mahbot-785).
    /// Used for real SNR estimation in [`compute_utterance_quality`] instead
    /// of the fake SNR computed from speech dynamic range (mahbot-782).
    /// Reset to `None` after the utterance is consumed by
    /// [`handle_enrollment_sample`].
    noise_rms_estimate: Option<f32>,
    /// Rolling window of per-frame soft scores from template matching
    /// (mahbot-773).  Each element is the `total_score` (sum of sigmoid
    /// scores across all K templates) for one embedding frame (~128ms of
    /// speech).  Detection fires when the sum over this window reaches
    /// [`match_threshold`].  Cleared entirely when a frame's score drops
    /// below [`NO_MATCH_RESET_THRESHOLD`] to prevent noise accumulation.
    score_window: Vec<f32>,
    /// Rolling buffer of raw audio samples captured during enrollment.
    ///
    /// Accumulated unconditionally for all raw input audio throughout
    /// enrollment, with a rolling ~200ms capacity ([`RAW_RING_MAX`]).
    /// The ring serves two purposes for VAD asymmetry mitigation
    /// (mahbot-775 Fix 3):
    ///
    /// 1. **Pre-speech context**: When sustained speech is first confirmed,
    ///    ~100ms from this ring is prepended to the utterance to capture
    ///    the quieter onset phonemes that the strict enrollment VAD (0.85)
    ///    excludes but live detection (VAD=0.5) includes.
    ///
    /// 2. **Post-speech tail**: At the first VAD-negative frame after speech,
    ///    ~100ms from this ring is snapshotted into [`post_speech_tail`],
    ///    before the 1.5s silence timeout overwrites the ring with silence.
    ///
    /// Both uses require the ring to contain audio from *before* the
    /// respective event (sustained speech / first silence).  Accumulation
    /// is unconditional because restricting it to pre-speech would leave
    /// the ring empty of trailing phonemes for the post-speech tail capture.
    raw_audio_ring: Vec<f32>,
    /// Rolling buffer of raw audio samples captured BEFORE AGC processing.
    ///
    /// Same rolling capacity as [`raw_audio_ring`] (~200ms, [`RAW_RING_MAX`]
    /// samples).  Used exclusively for noise RMS estimation at first sustained
    /// speech detection — AGC amplifies silence (up to 4×) more than speech
    /// (~1-2×), so noise RMS from post-AGC audio produces an artificially low
    /// SNR estimate (mahbot-785).
    pre_agc_ring: Vec<f32>,
    /// Trailing audio captured at the FIRST VAD-negative frame after speech
    /// during enrollment.  Used to append ~100ms of post-speech context that
    /// the strict enrollment VAD (0.85) excludes but live detection (VAD=0.5)
    /// includes.  Captured eagerly at the first silence transition so the raw
    /// audio ring still contains the trailing speech phonemes — by the time
    /// the 1.5s silence timeout fires, the ring has been overwritten with
    /// silence (mahbot-775 Fix 3).
    post_speech_tail: Vec<f32>,
    /// Audio pre-processor for noise suppression and AGC.
    /// Applied to every incoming audio chunk before VAD / mel extraction.
    audio_preprocessor: crate::audio_preprocessor::AudioPreprocessor,
    /// Accumulates non-VAD audio frames during enrollment for use as negative
    /// training examples (mahbot-797).  Collected between utterances (pre-enrollment
    /// ambient noise, inter-utterance silence/background) and saved as chunks
    /// when sustained speech begins.
    negative_audio_buf: Vec<f32>,
}

impl PipelineCtx {
    fn new() -> Self {
        Self {
            mic_rx: None,
            mic_stream: SendMicStream::default(),
            is_listening: false,
            is_recording: false,
            command_buffer: Vec::new(),
            silence_sample_count: 0,
            enrollment_mode: false,
            utterance_buf: Vec::new(),
            utterance_had_speech: false,
            utterance_silence_samples: 0,
            enrollment_no_speech_frame_count: 0,
            vad_positives_in_a_row: 0,
            audio_buffer: Vec::new(),
            mel_frame_buffer: Vec::new(),
            embedding_ring: Vec::new(),
            voice_batch: Vec::new(),
            enrollment_pending: None,
            utterance_speech_end_len: 0,
            auto_start_pending: CONFIG.voice_enabled().as_deref() == Some("true"),
            last_model_retry: None,
            last_wake_word_detection: None,
            score_window: Vec::new(),
            noise_rms_estimate: None,
            vad_threshold: VAD_THRESHOLD,
            raw_audio_ring: Vec::new(),
            pre_agc_ring: Vec::new(),
            post_speech_tail: Vec::new(),
            audio_preprocessor: {
                use crate::audio_preprocessor::PreprocessorConfig;
                let ns = CONFIG
                    .voice_noise_suppression()
                    .as_deref()
                    .is_none_or(|v| !v.eq_ignore_ascii_case("false"));
                let agc = CONFIG
                    .voice_agc()
                    .as_deref()
                    .is_none_or(|v| !v.eq_ignore_ascii_case("false"));
                crate::audio_preprocessor::AudioPreprocessor::new(PreprocessorConfig {
                    noise_suppression: ns,
                    agc,
                })
            },
            negative_audio_buf: Vec::new(),
        }
    }

    /// Clear all audio buffers and processing state.
    ///
    /// Must be called at every state transition to prevent stale audio from
    /// contaminating the new pipeline phase (mahbot-765).  Resets:
    /// - `voice_batch`, `mel_frame_buffer`, `embedding_ring`
    /// - `audio_buffer`, `command_buffer`
    /// - `utterance_buf` and enrollment tracking fields
    /// - `enrollment_no_speech_frame_count`
    /// - `enrollment_pending` (stale utterance after cancel)
    /// - `raw_audio_ring` and `post_speech_tail` (VAD asymmetry padding)
    /// - `score_window` (sliding detection scores)
    /// - `voice_state().enrollment_buffer` (global enrollment buffer)
    fn clear_pipeline_buffers(&mut self) {
        self.voice_batch.clear();
        self.mel_frame_buffer.clear();
        self.embedding_ring.clear();
        self.audio_buffer.clear();
        self.command_buffer.clear();
        self.utterance_buf.clear();
        self.utterance_had_speech = false;
        self.utterance_silence_samples = 0;
        self.utterance_speech_end_len = 0;
        self.enrollment_no_speech_frame_count = 0;
        self.vad_positives_in_a_row = 0;
        self.enrollment_pending = None;
        self.score_window.clear();
        self.noise_rms_estimate = None;
        self.raw_audio_ring.clear();
        self.pre_agc_ring.clear();
        self.post_speech_tail.clear();
        self.vad_threshold = VAD_THRESHOLD;
        self.last_wake_word_detection = None;
        self.audio_preprocessor.clear_buffer();
        self.negative_audio_buf.clear();
        voice_state()
            .write()
            .unwrap_poison()
            .enrollment_buffer
            .clear();
        voice_state()
            .write()
            .unwrap_poison()
            .negative_audio_chunks
            .clear();
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
            self.clear_pipeline_buffers();
            // Reset VAD detector state when re-creating the mic stream to
            // prevent stale ring buffer + pre-emphasis filter state from
            // the previous stream misclassifying the first few frames
            // (mahbot-771 Fix 3).
            reset_vad();
            drop(self.mic_stream.take());
            match start_microphone() {
                Ok((rx, stream)) => {
                    self.mic_rx = Some(rx);
                    self.mic_stream.set(stream);
                    self.is_listening = true;
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
        self.clear_pipeline_buffers();
        self.is_listening = false;
        self.is_recording = false;
        self.enrollment_mode = false;
        self.auto_start_pending = false;
        // Full reset of the audio pre-processor: the mic is being torn down,
        // so the NS noise profile from this acoustic environment is no longer
        // representative.  The next start_listening may be in a different room.
        self.audio_preprocessor.reset();
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
        self.clear_pipeline_buffers();
        self.enrollment_mode = true;
        self.vad_threshold = ENROLLMENT_VAD_THRESHOLD;
        set_status(VoiceStatus::Enrolling {
            sample: 0,
            total: NUM_ENROLLMENT_SAMPLES,
            duration_ms: 0,
            quality: None,
        });
        info!("Voice pipeline: enrollment started");
    }

    fn handle_cancel_enrollment(&mut self) {
        self.clear_pipeline_buffers();
        self.enrollment_mode = false;
        self.vad_threshold = VAD_THRESHOLD;
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
            Ok(mut templates) => {
                // Filter out templates enrolled with fewer samples than
                // NUM_ENROLLMENT_SAMPLES.  Old 3-sample templates are
                // incompatible with the new 10-sample threshold formula
                // and require re-enrollment (mahbot-765).
                let before = templates.templates.len();
                templates
                    .templates
                    .retain(|t| t.enrollment_samples >= NUM_ENROLLMENT_SAMPLES);
                let filtered = before - templates.templates.len();
                if filtered > 0 {
                    warn!(
                        "Filtered out {filtered} wake word template(s) enrolled with <{NUM_ENROLLMENT_SAMPLES} samples — re-enrollment required (mahbot-765)"
                    );
                }
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

                // ── Pre-AGC ring buffer (mahbot-785) ──
                // Capture raw audio before AGC processing for noise RMS
                // estimation.  AGC amplifies silence (up to 4×) more than
                // speech (~1-2×), so noise RMS computed from post-AGC audio
                // would artificially lower the SNR estimate.  Only accumulate
                // during enrollment where noise RMS is needed — the ring would
                // be stale/irrelevant during live detection since it's reset
                // when enrollment completes.
                if ctx.enrollment_mode {
                    ctx.pre_agc_ring.extend_from_slice(&samples);
                    if ctx.pre_agc_ring.len() > RAW_RING_MAX {
                        let excess = ctx.pre_agc_ring.len() - RAW_RING_MAX;
                        ctx.pre_agc_ring.drain(..excess);
                    }
                }

                // Apply noise suppression and/or AGC pre-processing before
                // the audio reaches VAD / mel extraction / enrollment.
                let samples = ctx.audio_preprocessor.process(samples);

                if ctx.enrollment_mode {
                    let (sample, total) = {
                        let state = voice_state().read().unwrap_poison();
                        (state.enrollment_buffer.len(), NUM_ENROLLMENT_SAMPLES)
                    };
                    handle_enrollment_audio(&samples, &mut ctx, sample, total);
                } else if ctx.is_recording {
                    handle_recording_audio(samples, &mut ctx).await;
                } else {
                    handle_wake_word_detection(&samples, &mut ctx);
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
            let noise_rms = ctx.noise_rms_estimate.take();
            handle_enrollment_sample(samples, noise_rms).await;
            // Reset enrollment_mode only on successful completion, not on
            // failure — if finalize_enrollment failed, the user can retry
            // by speaking the wake word again without re-initiating enrollment.
            if matches!(
                voice_state().read().unwrap_poison().status,
                VoiceStatus::Enrolled
            ) {
                // Clear all audio buffers BEFORE resetting enrollment_mode
                // to prevent stale audio from leaking into detection mode
                // during the ~1.5s status delay (mahbot-765 Race condition fix).
                ctx.clear_pipeline_buffers();
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

/// Check that an enrollment utterance is long enough for meaningful matching.
/// Returns an error message if the sample is shorter than 400ms — this rejects
/// noise blips and coughs while accepting any real wake word utterance.
///
/// Uses wall-clock duration (not embedding count) because the Google
/// speech_embedding/1 model produces exactly 1 embedding from any 76-frame
/// window — a single embedding is the model's full 96-dim output, not
/// "incomplete".  The 400ms floor is well above the ~760ms window that a
/// typical wake word needs, so any real utterance passes (mahbot-772).
///
/// This is extracted as a separate function so it can be unit-tested without
/// requiring ONNX model inference (mahbot-770 Fix 3).
fn check_enrollment_utterance_length(
    embeddings_len: usize,
    duration_ms: u64,
) -> Result<(), String> {
    // Reject completely empty embeddings (no template data to match against),
    // regardless of duration.
    if embeddings_len == 0 {
        return Err(format!(
            "Utterance produced no embeddings ({duration_ms}ms) — speak longer"
        ));
    }
    // Duration floor: reject noise blips and coughs (mahbot-772) using the
    // same threshold as the quality scoring pipeline.  Single-embedding
    // utterances are accepted — the Google speech_embedding/1 model produces
    // exactly 1 embedding from any 76-frame window, which is its full output.
    if duration_ms < ENROLLMENT_QUALITY_DURATION_MIN_MS {
        Err(format!(
            "Utterance too short ({duration_ms}ms, {embeddings_len} embedding(s)) — speak longer"
        ))
    } else {
        Ok(())
    }
}

/// Handle enrollment sample: process audio into embeddings and accumulate.
///
/// ONNX inference is CPU-bound (mel spectrogram + embedding computation).
/// It runs on a blocking thread via `spawn_blocking` to avoid starving
/// the async pipeline during enrollment.
///
/// Implements minimum utterance length check (mahbot-772): utterances
/// shorter than 400ms are rejected to reject noise blips and coughs.
#[allow(clippy::too_many_lines)]
async fn handle_enrollment_sample(samples: Vec<f32>, noise_rms: Option<f32>) {
    if !models_ready() {
        warn!("Models not ready for enrollment");
        return;
    }

    // Compute utterance duration before moving `samples` into the closure.
    let duration_ms = (samples.len() as u64 * 1000) / u64::from(SAMPLE_RATE);

    // Clone samples for quality computation (they will be moved into
    // spawn_blocking for ONNX inference).
    let samples_for_quality = samples.clone();

    // Run ONNX inference on a blocking thread to avoid blocking the async pipeline.
    let embeddings_result =
        tokio::task::spawn_blocking(move || process_enrollment_sample(&samples))
            .await
            .unwrap_or_else(|e| Err(anyhow!("Blocking task failed: {e}")));

    match embeddings_result {
        Ok(embeddings) => {
            // ── Minimum utterance length check (mahbot-772) ──
            if let Err(msg) = check_enrollment_utterance_length(embeddings.len(), duration_ms) {
                warn!("{msg}");
                set_status(VoiceStatus::Error(msg));
                return;
            }

            let (count, quality) = {
                let mut state = voice_state().write().unwrap_poison();

                // Compute quality BEFORE pushing: compare against existing samples
                // (without the current sample in the comparison set).
                let quality = Some(compute_utterance_quality(
                    &samples_for_quality,
                    &embeddings,
                    &state.enrollment_buffer,
                    noise_rms,
                ));

                state.enrollment_buffer.push(embeddings);
                let count = state.enrollment_buffer.len();
                // state dropped here — no lock held across await
                (count, quality)
            };

            if count >= NUM_ENROLLMENT_SAMPLES {
                match finalize_enrollment(WAKE_WORD_NAME) {
                    Ok((new_templates, minimum_matches)) => {
                        info!(
                            "Enrollment complete: wake word '{WAKE_WORD_NAME}' ({} templates, minimum_matches={minimum_matches})",
                            new_templates.len(),
                        );

                        // ── Train verifier (mahbot-777, mahbot-797) ──────
                        // Uses real (non-synthetic) negative embeddings
                        // collected from pre-enrollment ambient noise and
                        // inter-utterance audio during enrollment. Falls back
                        // to synthetic Gaussian negatives only when fewer than
                        // 2 real chunks were captured (mahbot-797).
                        //
                        // Positive examples are all per-frame embeddings from
                        // the enrollment buffer (mahbot-788 Fix 3) — each
                        // utterance contributes 6-10 frames instead of a
                        // single mean-pooled vector, giving 60-100 examples.
                        let positive_embeddings = {
                            let state = voice_state().read().unwrap_poison();
                            state
                                .enrollment_buffer
                                .iter()
                                .flat_map(|sample| sample.iter().cloned())
                                .collect::<Vec<Vec<f32>>>()
                        };

                        let verifier = if positive_embeddings.is_empty() {
                            warn!(
                                "Could not train verifier: no valid positive embeddings \
                                     from {} enrollment samples",
                                positive_embeddings.len(),
                            );
                            crate::voice_verifier::VoiceVerifier::untrained()
                        } else {
                            // Collect negative audio chunks and process them
                            // through ONNX to get real negative embeddings.
                            let negative_chunks = voice_state()
                                .write()
                                .unwrap_poison()
                                .negative_audio_chunks
                                .split_off(0);
                            let n_chunks = negative_chunks.len();
                            // Require >=2 chunks to ensure embedding diversity
                            // and reduce the chance of a single noisy chunk
                            // (e.g. VAD misclassification, transient mic pop)
                            // dominating the negative class. With 2+ separate
                            // ambient/inter-utterance periods (each >=0.5s),
                            // we get a more representative sample of the
                            // deployment environment's non-wake-word audio.
                            let real_negatives = if n_chunks >= 2 && ONNX_MODELS.get().is_some() {
                                let result = tokio::task::spawn_blocking(move || {
                                    let models =
                                        ONNX_MODELS.get().expect("ONNX_MODELS checked above");
                                    let mut all_neg = Vec::new();
                                    for chunk in &negative_chunks {
                                        match extract_embeddings_from_audio(models, chunk) {
                                            Ok(embs) => all_neg.extend(embs),
                                            Err(e) => {
                                                warn!(
                                                    "Failed to extract negative embedding \
                                                     from ambient audio chunk ({} samples): \
                                                     {e}",
                                                    chunk.len(),
                                                );
                                            }
                                        }
                                    }
                                    all_neg
                                })
                                .await
                                .unwrap_or_default();
                                if result.is_empty() {
                                    None
                                } else {
                                    Some(result)
                                }
                            } else {
                                None
                            };

                            if let Some(neg_embeddings) = real_negatives {
                                let n_neg = neg_embeddings.len();
                                let v = crate::voice_verifier::VoiceVerifier::train(
                                    &positive_embeddings,
                                    &neg_embeddings,
                                    0.5,  // standard logistic regression boundary
                                    1.0,  // L2 regularization (lambda)
                                    0.01, // learning rate
                                    2000, // max iterations
                                );
                                info!(
                                    "Verifier trained from {} positive + {n_neg} real \
                                     negative embedding(s) ({})",
                                    positive_embeddings.len(),
                                    if n_neg < n_chunks {
                                        format!("{n_chunks} chunks → {n_neg} embeds")
                                    } else {
                                        format!("{n_chunks} chunks")
                                    },
                                );
                                v
                            } else {
                                let v = crate::voice_verifier::VoiceVerifier::
                                        train_with_synthetic_negatives(
                                            &positive_embeddings,
                                            0.5,
                                        );
                                info!(
                                    "Verifier trained from {} per-frame positive \
                                     embedding(s) + synthetic negatives (no real \
                                     negatives available)",
                                    positive_embeddings.len(),
                                );
                                v
                            }
                        };

                        let mut existing = get_templates();
                        let existing_mut = Arc::make_mut(&mut existing);
                        // Remove any existing templates with the same base name
                        // (e.g. "custom", "custom_2", "custom_3").
                        let base_name = WAKE_WORD_NAME.to_string();
                        existing_mut.templates.retain(|t| {
                            t.name != base_name
                                && !t.name.starts_with(&format!("{WAKE_WORD_NAME}_"))
                        });
                        existing_mut.templates.extend(new_templates);
                        existing_mut.minimum_matches = minimum_matches;
                        existing_mut.verifier = verifier;
                        set_templates(existing);

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
                    duration_ms,
                    quality,
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
///
/// Silence duration is measured in audio samples (not wall-clock time) so that
/// system load / processing delays don't affect recording cutoff consistency
/// (ticket mahbot-760).
#[allow(clippy::cast_precision_loss)]
async fn handle_recording_audio(samples: Vec<f32>, ctx: &mut PipelineCtx) {
    ctx.command_buffer.extend_from_slice(&samples);
    let speech = is_speech(&samples);
    if speech {
        ctx.silence_sample_count = 0;
    } else {
        // Accumulate silence by raw chunk size: each call receives a
        // variable-size chunk of audio samples directly from the mic.
        // This differs from the enrollment path (HOP_LENGTH per frame)
        // because recording operates on raw chunks, not frame iterations.
        ctx.silence_sample_count += samples.len();
    }

    let duration_secs = ctx.command_buffer.len() as f64 / f64::from(SAMPLE_RATE);
    let silence_timeout = ctx.silence_sample_count >= SILENCE_THRESHOLD_SAMPLES;

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
        let cmd_buf = std::mem::take(&mut ctx.command_buffer);

        match transcribe_audio(&cmd_buf).await {
            Ok(transcribed) => {
                info!("Transcribed: {transcribed}");
                route_to_agent(transcribed).await;
                set_status(VoiceStatus::Listening);
                ctx.is_recording = false;
            }
            Err(e) => {
                warn!("Transcription failed: {e}");
                set_status(VoiceStatus::Error("Transcription failed".to_string()));
                tokio::time::sleep(Duration::from_secs(2)).await;
                set_status(VoiceStatus::Listening);
                ctx.is_recording = false;
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
        Err(e) => {
            warn!("Mel spectrogram failed: {e}");
            // Clear the batch so it doesn't grow unbounded when the
            // ONNX model is consistently failing (ticket mahbot-760).
            voice_batch.clear();
        }
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
///
/// Implements cooldown (mahbot-770 Fix 2) and soft-scoring + rolling window
/// detection (mahbot-773) via the `last_wake_word_detection` and
/// `score_window` fields.
fn handle_wake_word_detection(samples: &[f32], ctx: &mut PipelineCtx) {
    // ── Cooldown check (mahbot-770 Fix 2) ──
    // If we recently detected the wake word, skip ALL processing for this
    // chunk to prevent rapid consecutive false triggers.  The audio is
    // discarded entirely (not accumulated) since the user is either in
    // command-recording mode or the cooldown is active.
    if let Some(last) = ctx.last_wake_word_detection
        && last.elapsed() < WAKE_WORD_COOLDOWN
    {
        debug!(
            "Wake word cooldown active ({}ms elapsed)",
            last.elapsed().as_millis()
        );
        // Discard all buffered audio — don't accumulate during cooldown
        // to avoid a large stale batch when the cooldown expires.
        ctx.audio_buffer.clear();
        ctx.voice_batch.clear();
        ctx.mel_frame_buffer.clear();
        ctx.embedding_ring.clear();
        ctx.score_window.clear();
        return;
    }

    ctx.audio_buffer.extend_from_slice(samples);

    // Process frames from the buffer without per-iteration O(n) drain shifts.
    // Track a consumed offset and drain everything once after the loop.
    let len = ctx.audio_buffer.len();
    let mut consumed = 0;
    while consumed + FRAME_LENGTH <= len {
        let frame = &ctx.audio_buffer[consumed..consumed + FRAME_LENGTH];

        // VAD gate — skip silence to avoid wasted ONNX compute
        if is_speech(frame) {
            // Add only the NEW samples (HOP_LENGTH per frame) to avoid
            // duplicating overlapping audio. Each frame overlaps the previous
            // by 50% (HOP_LENGTH = FRAME_LENGTH/2), so appending the full
            // frame would duplicate half the audio — corrupting the mel model
            // input with repeated segments.
            ctx.voice_batch.extend_from_slice(&frame[..HOP_LENGTH]);
        } else if !ctx.voice_batch.is_empty() {
            // Silence transition: flush accumulated voiced batch
            flush_voice_batch(&mut ctx.voice_batch, &mut ctx.mel_frame_buffer);
            ctx.voice_batch.clear();
            if try_match_wake_word_and_push_embedding(ctx) {
                return;
            }
            consumed += HOP_LENGTH;
            continue;
        }

        // Process batch when enough voiced audio accumulated
        // (every ~128ms instead of every 32ms)
        if ctx.voice_batch.len() >= VOICE_BATCH_SIZE {
            flush_voice_batch(&mut ctx.voice_batch, &mut ctx.mel_frame_buffer);
            if try_match_wake_word_and_push_embedding(ctx) {
                return;
            }
        }
        consumed += HOP_LENGTH;
    }

    // Single O(remaining) drain instead of O(remaining) per frame iteration.
    if consumed > 0 {
        ctx.audio_buffer.drain(..consumed);
    }
}

/// Handle audio during enrollment mode.
///
/// Accumulates audio frames into utterances with VAD-based boundary detection.
/// When a complete utterance is detected (speech followed by silence exceeding
/// `SILENCE_DURATION`), stores the utterance in `enrollment_pending` for
/// inline processing (avoids race conditions with the command channel).
///
/// **Verifier negative collection** (mahbot-797): Non-VAD frames captured
/// before the first detected speech (pre-enrollment ambient noise) and during
/// inter-utterance silence (audio between wake word utterances) are accumulated
/// into `ctx.negative_audio_buf`. On the first transition to sustained speech,
/// this buffer is saved as a chunk in `voice_state().negative_audio_chunks`.
/// These real (non-synthetic) negative examples are later used to train the
/// wake word verifier at enrollment finalization, replacing the old synthetic
/// Gaussian noise that caused false triggers (mahbot-797 Fix 4).
///
/// **VAD symmetry** (mahbot-765): Only VAD-positive frames are accumulated
/// into the utterance buffer, mirroring the detection pipeline
/// ([`handle_wake_word_detection`]). This eliminates the asymmetry where
/// enrollment built templates on audio that the detector never processes.
/// Utterance end is detected after [`SILENCE_THRESHOLD_SAMPLES`] consecutive
/// non-VAD-positive frames, matching the detection-side behavior.
///
/// If no speech has been detected for a prolonged period (~5s), a warning
/// status is set to prompt the user to speak louder or move closer to the mic.
///
/// Updates voice status dynamically to reflect the enrollment phase:
/// - No speech yet: caller's `Enrolling` text persists (or no-speech warning)
/// - Speech detected: `ListeningDuringEnrollment`
/// - Speech ended, awaiting silence: `WaitingForSilenceDuringEnrollment`
#[allow(clippy::too_many_lines)]
fn handle_enrollment_audio(samples: &[f32], ctx: &mut PipelineCtx, sample: usize, total: usize) {
    // ── Raw audio ring buffer (mahbot-775 Fix 3) ──
    // Accumulate ALL raw input audio into a rolling ring so we can later
    // prepend ~100ms of pre-VAD-trigger context and append ~100ms of
    // post-speech context to the utterance.  This captures the onset/offset
    // phonemes that the strict enrollment VAD (0.85) excludes but live
    // detection (VAD=0.5) includes, reducing the systematic mismatch.
    ctx.raw_audio_ring.extend_from_slice(samples);
    if ctx.raw_audio_ring.len() > RAW_RING_MAX {
        let excess = ctx.raw_audio_ring.len() - RAW_RING_MAX;
        ctx.raw_audio_ring.drain(..excess);
    }

    ctx.audio_buffer.extend_from_slice(samples);

    // Process frames with offset tracking instead of per-iteration O(n) drain.
    let len = ctx.audio_buffer.len();
    let mut consumed = 0;
    while consumed + FRAME_LENGTH <= len {
        let frame = &ctx.audio_buffer[consumed..consumed + FRAME_LENGTH];

        if is_speech_with_threshold(frame, ctx.vad_threshold) {
            // VAD-positive: accumulate only the new samples (HOP_LENGTH per
            // frame) into the utterance buffer. Adding the full FRAME_LENGTH
            // frame would duplicate overlapping audio since consecutive frames
            // overlap by 50%. The utterance buffer is later passed to
            // extract_embeddings_from_audio which expects continuous raw audio,
            // not overlapping frames.
            ctx.utterance_buf.extend_from_slice(&frame[..HOP_LENGTH]);

            ctx.vad_positives_in_a_row += 1;
            // Reset the no-speech warning counter on any VAD-positive frame.
            ctx.enrollment_no_speech_frame_count = 0;

            if ctx.vad_positives_in_a_row >= ENROLLMENT_VAD_CONSECUTIVE_REQUIRED {
                // Sustained speech confirmed (mahbot-772): update tracking.
                let was_waiting_for_silence = ctx.utterance_silence_samples > 0;

                // ── Capture noise RMS from pre-AGC ring (mahbot-785) ──
                // On the FIRST transition from silence to sustained speech,
                // capture the ambient noise RMS from the pre-AGC audio ring.
                // The pre_agc_ring stores raw mic audio before AGC gain is
                // applied — this matters because AGC amplifies silence (up to
                // 4×) disproportionately to speech (~1-2×), so post-AGC noise
                // RMS would produce an SNR estimate 6-12 dB lower than the
                // true room SNR, triggering a false low-SNR warning even in
                // quiet environments (mahbot-782 used raw_audio_ring which
                // contains post-AGC audio, causing this false-positive).
                let already_had_speech = ctx.utterance_had_speech;
                // ── Save collected ambient audio for verifier negatives (mahbot-797) ──
                // On the FIRST transition from silence to sustained speech,
                // save the accumulated non-wake-word audio (pre-enrollment
                // ambient noise or inter-utterance silence) as a potential
                // negative training example for the verifier.
                if !already_had_speech {
                    if ctx.negative_audio_buf.len() >= MIN_NEGATIVE_AUDIO_LEN {
                        voice_state()
                            .write()
                            .unwrap_poison()
                            .negative_audio_chunks
                            .push(std::mem::take(&mut ctx.negative_audio_buf));
                    } else {
                        ctx.negative_audio_buf.clear();
                    }
                }
                if !already_had_speech && ctx.noise_rms_estimate.is_none() {
                    let speech_boundary = ENROLLMENT_VAD_CONSECUTIVE_REQUIRED * HOP_LENGTH;
                    let pre_speech_end = ctx.pre_agc_ring.len().saturating_sub(speech_boundary);
                    if pre_speech_end > 0 {
                        let sum_sq: f32 = ctx.pre_agc_ring[..pre_speech_end]
                            .iter()
                            .map(|&s| s * s)
                            .sum();
                        #[expect(clippy::cast_precision_loss)]
                        let rms = (sum_sq / pre_speech_end as f32).sqrt();
                        ctx.noise_rms_estimate = Some(rms);
                    }
                }

                // ── Prepend pre-speech context from raw ring (mahbot-775 Fix 3) ──
                // On the FIRST transition from silence to sustained speech,
                // prepend ~100ms of audio from the raw input ring to capture
                // the quieter onset phonemes that VAD=0.85 excluded.
                if !already_had_speech {
                    let pad_samples = CONTEXT_PADDING_SAMPLES; // 100ms
                    let start = ctx.raw_audio_ring.len().saturating_sub(pad_samples);
                    let padding: Vec<f32> = ctx.raw_audio_ring[start..].to_vec();
                    if !padding.is_empty() {
                        let mut padded = padding;
                        padded.extend_from_slice(&ctx.utterance_buf);
                        ctx.utterance_buf = padded;
                    }
                }

                if !already_had_speech || was_waiting_for_silence {
                    // Transition from silence to speech, or speech resumed after
                    // a pause before the 1.5s timeout — show "Listening…"
                    set_status(VoiceStatus::ListeningDuringEnrollment { sample, total });
                }
                ctx.utterance_had_speech = true;
                ctx.utterance_speech_end_len = ctx.utterance_buf.len();
                ctx.utterance_silence_samples = 0;
            } else if ctx.utterance_had_speech {
                // Previously confirmed speech: a single VAD-positive frame is
                // enough to extend the utterance end and reset silence (handles
                // brief VAD gaps during continuous speech, e.g. unvoiced stops).
                ctx.utterance_speech_end_len = ctx.utterance_buf.len();
                ctx.utterance_silence_samples = 0;
            }
        } else {
            // VAD-negative: reset consecutive counter.
            ctx.vad_positives_in_a_row = 0;

            if ctx.utterance_had_speech {
                // After speech: track silence duration to detect utterance end.
                // Track by sample count (not wall-clock time) so that system load
                // / processing delays don't affect cutoff consistency (mahbot-760).
                //
                // NOTE: We accumulate HOP_LENGTH per frame iteration (not the raw
                // chunk size) because each loop iteration processes exactly
                // HOP_LENGTH new audio samples.  This differs from the recording
                // path (handle_recording_audio) which receives variable-size raw
                // chunks and accumulates chunks.len() directly — both approaches
                // correctly measure silence in audio samples at 16 kHz; they just
                // operate at different granularities (frame-level vs chunk-level).

                // ── Capture trailing speech at first silence (mahbot-775 Fix 3) ──
                // The raw audio ring still contains the quiet trailing phonemes at
                // this point (the VAD-negative frames that are just below 0.85).
                // By the time the 1.5s silence timeout fires, the ring will have
                // been fully overwritten with silence, so we snapshot the tail now.
                if ctx.utterance_silence_samples == 0 {
                    let pad_samples = CONTEXT_PADDING_SAMPLES; // 100ms
                    let start = ctx.raw_audio_ring.len().saturating_sub(pad_samples);
                    ctx.post_speech_tail = ctx.raw_audio_ring[start..].to_vec();
                }

                ctx.utterance_silence_samples += HOP_LENGTH;
                if ctx.utterance_silence_samples >= SILENCE_THRESHOLD_SAMPLES {
                    // Utterance is complete. With VAD-gated accumulation,
                    // utterance_buf already only contains speech frames, but
                    // we still truncate to utterance_speech_end_len for safety
                    // in edge cases.
                    ctx.utterance_buf.truncate(ctx.utterance_speech_end_len);
                    // ── Append post-speech tail (mahbot-775 Fix 3) ──
                    // The post_speech_tail was captured at the first silence
                    // transition while the raw ring still held the trailing
                    // speech phonemes.  Append ~100ms to match the full wake
                    // word that live detection (VAD=0.5) includes.
                    if !ctx.post_speech_tail.is_empty() {
                        ctx.utterance_buf.extend_from_slice(&ctx.post_speech_tail);
                    }
                    ctx.enrollment_pending = Some(std::mem::take(&mut ctx.utterance_buf));
                    ctx.utterance_speech_end_len = 0;
                    ctx.utterance_had_speech = false;
                    ctx.utterance_silence_samples = 0;
                    ctx.enrollment_no_speech_frame_count = 0;
                    ctx.post_speech_tail.clear();
                    // Note: noise_rms_estimate is intentionally NOT reset here.
                    // It is consumed by the main loop at line 3048 via
                    // ctx.noise_rms_estimate.take() alongside enrollment_pending.
                    // reset is handled by clear_pipeline_buffers for
                    // cancellation/completion safety.  Clearing it here would
                    // destroy the captured noise RMS before the main loop can
                    // use it for compute_utterance_quality (mahbot-782).
                    // Mark ALL samples as consumed so the post-loop drain removes
                    // everything (including the unconsumed tail).  This replaces
                    // the original `clear()` while working correctly even when
                    // `consumed > 0` — the original `clear()` + post-loop
                    // `drain(..consumed)` would panic on an empty buffer.
                    consumed = len;
                    break;
                }
                // Set status during the first 200ms of silence to show
                // "Keep silent to confirm…".
                if ctx.utterance_silence_samples < SILENCE_UI_GATE_SAMPLES {
                    set_status(VoiceStatus::WaitingForSilenceDuringEnrollment { sample, total });
                }
            } else if !ctx.utterance_had_speech {
                // Accumulate non-VAD audio for verifier negatives (mahbot-797):
                // pre-enrollment ambient noise, inter-utterance silence, or
                // any non-wake-word audio between utterances.  Each frame
                // contributes HOP_LENGTH new samples.
                ctx.negative_audio_buf
                    .extend_from_slice(&frame[..HOP_LENGTH]);

                // Pre-speech silence: increment no-speech counter.  When the
                // count reaches ENROLLMENT_NO_SPEECH_TIMEOUT_FRAMES (~5 seconds
                // of non-VAD audio), show a warning so the user knows to speak
                // louder or move closer (mahbot-765 VAD symmetry mitigation).
                ctx.enrollment_no_speech_frame_count += 1;
                // Warn after the derived frame threshold.  The constant is
                // computed from ENROLLMENT_NO_SPEECH_DURATION × SAMPLE_RATE /
                // HOP_LENGTH, so the threshold stays correct if frame/hop sizes
                // are adjusted (mahbot-765).
                if ctx.enrollment_no_speech_frame_count >= ENROLLMENT_NO_SPEECH_TIMEOUT_FRAMES {
                    set_status(VoiceStatus::Error(
                        "No speech detected — try speaking louder or move closer to microphone"
                            .to_string(),
                    ));
                    // Don't reset the counter; the status persists until VAD fires
                    // or the user re-initiates enrollment.
                }
            }
        }
        consumed += HOP_LENGTH;
    }

    // Single O(remaining) drain instead of O(remaining) per frame iteration.
    if consumed > 0 {
        ctx.audio_buffer.drain(..consumed);
    }
}

/// Compute embedding from mel frames, push to ring buffer, and match against templates.
///
/// Implements soft scoring + rolling window detection (mahbot-773): each
/// template contributes a sigmoid score (0.0–1.0) based on DTW distance;
/// scores are summed across all templates and accumulated in a rolling window.
/// Detection fires when the rolling sum exceeds [`match_threshold`].
/// The window is reset entirely when a frame's score drops below
/// [`NO_MATCH_RESET_THRESHOLD`] to prevent noise accumulation.
/// On detection, the cooldown timestamp is set and `voice_batch` is cleared.
///
/// Returns `true` if wake word was detected (caller should clear state and return).
fn try_match_wake_word_and_push_embedding(ctx: &mut PipelineCtx) -> bool {
    if ctx.mel_frame_buffer.is_empty() {
        return false;
    }
    let Some(models) = ONNX_MODELS.get() else {
        return false;
    };

    // If the mel buffer is shorter than the required embedding window (76 frames),
    // pad it with tapered fade-out frames so an embedding can always be computed.
    // Without this, short wake words (e.g. 0.5s → ~32 mel frames) would silently
    // be discarded and never detected.
    let padded_window: Vec<Vec<f32>>;
    let embed_input: &[Vec<f32>] = if ctx.mel_frame_buffer.len() < EMBEDDING_WINDOW_FRAMES {
        padded_window = pad_mel_frames_to_window(&ctx.mel_frame_buffer);
        &padded_window
    } else {
        // Take the most recent EMBEDDING_WINDOW_FRAMES
        &ctx.mel_frame_buffer[ctx.mel_frame_buffer.len() - EMBEDDING_WINDOW_FRAMES..]
    };

    let embedding =
        match crate::util::with_block_in_place(|| compute_embedding(models, embed_input)) {
            Ok(emb) => {
                debug!(
                    "Embedding computed: {} dims (ring size before push: {})",
                    emb.len(),
                    ctx.embedding_ring.len(),
                );
                emb
            }
            Err(e) => {
                warn!("Wake word matching: compute_embedding failed: {e:#}");
                return false;
            }
        };

    ctx.embedding_ring.push(embedding);
    while ctx.embedding_ring.len() > EMBEDDING_RING_MAX {
        ctx.embedding_ring.remove(0);
    }

    let templates = get_templates();
    let total_score = if !templates.templates.is_empty() && !ctx.embedding_ring.is_empty() {
        score_matching_templates(&ctx.embedding_ring, &templates)
    } else {
        0.0
    };

    // ── Soft scoring + rolling window (mahbot-773) ──
    //
    // Delegates to `process_wake_word_score()` which encapsulates the
    // no-match reset, score accumulation, window trimming, and threshold
    // check.  Extracted into a pure function for direct unit testability.
    if process_wake_word_score(
        total_score,
        &mut ctx.score_window,
        templates.minimum_matches,
    ) {
        // ── Verifier gate (mahbot-777, mahbot-788) ────────────
        // After the rolling window check passes, run the second-stage
        // logistic regression verifier to catch false positives that
        // survived DTW matching.  Near-zero CPU overhead (~1 μs per
        // frame) since it only runs on candidate frames that already
        // passed the rolling window.
        //
        // Fix 1 (mahbot-788): Check all recent ROLLING_WINDOW_N
        // embeddings and take the maximum score, instead of checking
        // only the last frame.  A weak trailing frame (end of phoneme,
        // trailing silence) should not veto detection when previous
        // frames are strong.
        //
        // Fix 2 (mahbot-797): Clear the score window on verifier rejection.
        // Without this, a single borderline pass lets accumulated scores
        // drift high enough that any speech eventually triggers detection.
        // The next frame must start from zero — DTW matching will rebuild
        // if the frame is truly a wake word.
        if templates.verifier.is_trained() {
            let start = ctx.embedding_ring.len().saturating_sub(ROLLING_WINDOW_N);
            let score = ctx.embedding_ring[start..]
                .iter()
                .map(|emb| templates.verifier.predict(emb))
                .fold(0.0f32, f32::max);
            if score < templates.verifier.threshold {
                debug!(
                    "Wake word suppressed by verifier: max_score={score:.4} < threshold={} \
                     (checked {} recent embeddings)",
                    templates.verifier.threshold,
                    ctx.embedding_ring.len() - start,
                );
                // Clear the score window so the next frame starts from zero.
                // Without this the accumulated DTW evidence eventually lets
                // any speech through (mahbot-797).
                ctx.score_window.clear();
                return false;
            }
        }

        // Wake word detected — clear pipeline state and start recording.
        ctx.is_recording = true;
        ctx.command_buffer.clear();
        ctx.silence_sample_count = 0;
        ctx.mel_frame_buffer.clear();
        ctx.embedding_ring.clear();
        ctx.audio_buffer.clear();
        ctx.voice_batch.clear();
        ctx.score_window.clear();
        ctx.last_wake_word_detection = Some(Instant::now());
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
    //!   `test_finalize_enrollment_naming` uses `get_or_init` + state reset so
    //!   it tolerates an already-initialised pipeline.
    //! - **Global `CONFIG`** is read by [`PipelineCtx::new()`] to set
    //!   `auto_start_pending`. Tests implicitly depend on `CONFIG` being in its
    //!   default state (all fields `None`). If a preceding test modifies
    //!   `CONFIG`, `auto_start_pending` may be non-`false`, which this test's
    //!   assertions must still tolerate.
    use super::*;
    use std::f32::consts::PI;

    /// Generate a speech-like 512-sample frame (32 ms at 16 kHz) that Earshot's
    /// neural VAD classifies as speech.  Uses a harmonic series (F0 + formants)
    /// to simulate vowel-like spectral structure.  All samples are in [-1, 1].
    fn speech_frame() -> Vec<f32> {
        let mut frame = Vec::with_capacity(FRAME_LENGTH);
        for i in 0..FRAME_LENGTH {
            let t = i as f32 / SAMPLE_RATE as f32;
            // Harmonic series with formant-like structure.
            // Individual amplitudes ensure the sum never exceeds 1.0.
            let sample = (2.0 * PI * 130.0 * t).sin() * 0.4      // F0 ~130 Hz (male voice)
                + (2.0 * PI * 520.0 * t).sin() * 0.25    // H2 (1st formant region)
                + (2.0 * PI * 910.0 * t).sin() * 0.15    // H3
                + (2.0 * PI * 1300.0 * t).sin() * 0.1    // H4 (2nd formant region)
                + (2.0 * PI * 2600.0 * t).sin() * 0.05; // H6 (3rd formant region)
            frame.push(sample);
        }
        // Verify within [-1, 1] (amplitudes sum to 0.95 < 1.0).
        debug_assert!(
            frame.iter().all(|&s| s.abs() <= 1.0),
            "speech_frame exceeds [-1, 1] range"
        );
        frame
    }

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

    // ── is_speech (VAD — Earshot neural VAD) ──────────────────────────────

    #[test]
    #[serial_test::serial(voice)]
    fn test_is_speech_silence() {
        super::reset_vad();
        let silence = vec![0.0f32; 512];
        assert!(!super::is_speech(&silence));
    }

    #[test]
    #[serial_test::serial(voice)]
    fn test_is_speech_loud() {
        super::reset_vad();
        // Speech-like frame with harmonic content
        let loud = speech_frame();
        assert!(super::is_speech(&loud));
    }

    #[test]
    #[serial_test::serial(voice)]
    fn test_is_speech_moderate() {
        super::reset_vad();
        // Moderate-amplitude speech-like frame (70% of full speech_frame)
        let modr: Vec<f32> = speech_frame().iter().map(|s| s * 0.7).collect();
        assert!(super::is_speech(&modr));
    }

    #[test]
    #[serial_test::serial(voice)]
    fn test_is_speech_empty() {
        assert!(!super::is_speech(&[]));
    }

    #[test]
    #[serial_test::serial(voice)]
    fn test_is_speech_tiny() {
        super::reset_vad();
        // Audio below the neural VAD threshold — too quiet for speech
        let quiet = vec![0.001f32; 512];
        assert!(!super::is_speech(&quiet));
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
    #[serial_test::serial(voice)]
    fn test_voice_pipeline_commands_and_enrollment_guard() {
        // Initialize global VOICE_PIPELINE once for all checks below.
        // Using let _ to tolerate parallel tests that also need the pipeline.
        let _ = VOICE_PIPELINE.set(RwLock::new(VoicePipelineState {
            enabled: false,
            status: VoiceStatus::Disabled,
            templates: Arc::new(WakeWordTemplates::default()),
            enrollment_buffer: Vec::new(),
            negative_audio_chunks: Vec::new(),

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

        // ── handle_start_enrollment success path ────────────────────
        {
            ctx.is_listening = true;
            ctx.handle_start_enrollment();

            assert!(
                ctx.enrollment_mode,
                "enrollment should be started when mic is running"
            );
            assert_eq!(
                ctx.vad_threshold, ENROLLMENT_VAD_THRESHOLD,
                "vad_threshold should be set to ENROLLMENT_VAD_THRESHOLD during enrollment"
            );

            // Reset for subsequent tests
            ctx.enrollment_mode = false;
            ctx.is_listening = false;
            ctx.vad_threshold = VAD_THRESHOLD;
        }

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
            reset_vad();
            voice_state().write().unwrap_poison().enabled = true;
            ctx.is_listening = true;
            ctx.enrollment_mode = true;
            ctx.utterance_buf.clear();
            ctx.utterance_had_speech = false;
            ctx.utterance_silence_samples = 0;
            ctx.utterance_speech_end_len = 0;
            ctx.audio_buffer.clear();
            ctx.enrollment_pending = None;
            voice_state()
                .write()
                .unwrap_poison()
                .enrollment_buffer
                .clear();

            // Start enrollment at sample 0 of NUM_ENROLLMENT_SAMPLES
            set_status(VoiceStatus::Enrolling {
                sample: 0,
                total: NUM_ENROLLMENT_SAMPLES,
                duration_ms: 0,
                quality: None,
            });

            // 1. Silence before speech: no speech detected — utterance_had_speech
            //    stays false, and silence_samples starts accumulating.
            let silence = vec![0.0f32; FRAME_LENGTH];
            handle_enrollment_audio(&silence, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            assert!(
                !ctx.utterance_had_speech,
                "silence before speech must not set utterance_had_speech"
            );
            // After one frame of silence (HOP_LENGTH = 256 samples),
            // utterance_silence_samples is 0 because neither
            // branch fires for pre-speech silence in frame-level
            // processing (the silence-before-speech path is a no-op).
            // With VAD-gated accumulation the else branch also doesn't
            // fire because utterance_had_speech is false.
            assert_eq!(
                ctx.utterance_silence_samples, 0,
                "silence before speech should not accumulate silence_samples"
            );

            // 2. Speech sustained confirmation requires 3 consecutive
            //    VAD-positive frames (mahbot-772).  Feed 3 speech frames
            //    with audio_buffer cleared each time to ensure each call
            //    processes exactly one clean frame.
            let speech = speech_frame();
            ctx.audio_buffer.clear();
            // First speech frame: vad_positives_in_a_row = 1
            handle_enrollment_audio(&speech, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            // Second speech frame: vad_positives_in_a_row = 2
            ctx.audio_buffer.clear();
            handle_enrollment_audio(&speech, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            // Third speech frame: sustained speech confirmed
            ctx.audio_buffer.clear();
            handle_enrollment_audio(&speech, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            assert!(
                ctx.utterance_had_speech,
                "utterance_had_speech should be set after 3 consecutive speech frames"
            );

            // 3. Continued speech: utterance_had_speech stays true,
            //    utterance_speech_end_len extends with each speech frame.
            handle_enrollment_audio(&speech, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            assert!(
                ctx.utterance_had_speech,
                "utterance_had_speech should remain true on continued speech"
            );
            assert!(
                ctx.utterance_speech_end_len == ctx.utterance_buf.len(),
                "utterance_speech_end_len should track buffer end after each speech frame"
            );

            // 4. Silence after speech → silence_samples accumulates
            //    (the full timeout at SILENCE_THRESHOLD_SAMPLES stores
            //     the utterance in enrollment_pending).
            //    Reset the VAD so the silence frame is classified with a
            //    clean slate (no pre-emphasis transient from previous speech).
            //    Clear audio_buffer first so leftover speech from step 3
            //    doesn't contaminate the first silence frame.
            ctx.audio_buffer.clear();
            reset_vad();
            ctx.utterance_silence_samples = SILENCE_THRESHOLD_SAMPLES - 1;
            let silence = vec![0.0f32; FRAME_LENGTH];
            handle_enrollment_audio(&silence, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            assert!(
                ctx.enrollment_pending.is_some(),
                "silence timeout should store utterance in enrollment_pending"
            );
            // The timeout path clears internal state, so utterance_buf
            // and utterance_had_speech are reset.
            assert!(
                ctx.utterance_buf.is_empty(),
                "utterance_buf should be emptied after timeout"
            );
            assert!(
                !ctx.utterance_had_speech,
                "utterance_had_speech should reset after timeout"
            );

            // 5. Speech after previous utterance completed → starts a
            //    new utterance (utterance_had_speech transitions to true
            //    after 3 consecutive frames, mahbot-772).
            ctx.enrollment_pending = None;
            ctx.audio_buffer.clear();
            handle_enrollment_audio(&speech, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            ctx.audio_buffer.clear();
            handle_enrollment_audio(&speech, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            ctx.audio_buffer.clear();
            handle_enrollment_audio(&speech, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            assert!(
                ctx.utterance_had_speech,
                "utterance_had_speech should become true after 3 consecutive speech frames"
            );
            assert!(
                ctx.utterance_buf.len() > 0,
                "utterance_buf should accumulate new speech after resume"
            );

            // Clean up
            voice_state().write().unwrap_poison().enabled = false;
            ctx.is_listening = false;
            ctx.enrollment_mode = false;
        }

        // ── clear_pipeline_buffers — full buffer cleanup ──────────────
        {
            // Fill all buffers with test data
            ctx.voice_batch = vec![1.0; 100];
            ctx.mel_frame_buffer = vec![vec![1.0; 10]; 5];
            ctx.embedding_ring = vec![vec![1.0; 5]; 3];
            ctx.audio_buffer = vec![1.0; 200];
            ctx.command_buffer = vec![1.0; 300];
            ctx.utterance_buf = vec![1.0; 400];
            ctx.utterance_had_speech = true;
            ctx.utterance_silence_samples = 500;
            ctx.utterance_speech_end_len = 600;
            ctx.enrollment_no_speech_frame_count = 700;
            ctx.enrollment_pending = Some(vec![1.0; 50]);
            ctx.score_window = vec![1.5, 2.0, 0.8]; // simulated scores
            ctx.raw_audio_ring = vec![1.0; 100];
            ctx.pre_agc_ring = vec![1.0; 100];
            ctx.post_speech_tail = vec![1.0; 50];
            ctx.vad_positives_in_a_row = 2;
            ctx.vad_threshold = ENROLLMENT_VAD_THRESHOLD;
            ctx.last_wake_word_detection = Some(Instant::now());
            voice_state()
                .write()
                .unwrap_poison()
                .enrollment_buffer
                .push(vec![vec![1.0; 10]; 3]);
            voice_state()
                .write()
                .unwrap_poison()
                .enrollment_buffer
                .push(vec![vec![1.0; 10]; 3]);

            ctx.clear_pipeline_buffers();

            assert!(ctx.voice_batch.is_empty(), "voice_batch should be cleared");
            assert!(
                ctx.mel_frame_buffer.is_empty(),
                "mel_frame_buffer should be cleared"
            );
            assert!(
                ctx.embedding_ring.is_empty(),
                "embedding_ring should be cleared"
            );
            assert!(
                ctx.audio_buffer.is_empty(),
                "audio_buffer should be cleared"
            );
            assert!(
                ctx.command_buffer.is_empty(),
                "command_buffer should be cleared"
            );
            assert!(
                ctx.utterance_buf.is_empty(),
                "utterance_buf should be cleared"
            );
            assert!(
                !ctx.utterance_had_speech,
                "utterance_had_speech should be reset to false"
            );
            assert_eq!(
                ctx.utterance_silence_samples, 0,
                "utterance_silence_samples should be reset to 0"
            );
            assert_eq!(
                ctx.utterance_speech_end_len, 0,
                "utterance_speech_end_len should be reset to 0"
            );
            assert_eq!(
                ctx.enrollment_no_speech_frame_count, 0,
                "enrollment_no_speech_frame_count should be reset to 0"
            );
            assert_eq!(
                ctx.vad_positives_in_a_row, 0,
                "vad_positives_in_a_row should be reset to 0"
            );
            assert!(
                ctx.enrollment_pending.is_none(),
                "enrollment_pending should be reset to None"
            );
            assert!(
                voice_state()
                    .read()
                    .unwrap_poison()
                    .enrollment_buffer
                    .is_empty(),
                "voice_state().enrollment_buffer should be cleared"
            );
            assert!(
                ctx.score_window.is_empty(),
                "score_window should be cleared"
            );
            assert!(
                ctx.raw_audio_ring.is_empty(),
                "raw_audio_ring should be cleared"
            );
            assert!(
                ctx.pre_agc_ring.is_empty(),
                "pre_agc_ring should be cleared"
            );
            assert!(
                ctx.post_speech_tail.is_empty(),
                "post_speech_tail should be cleared"
            );
            assert!(
                ctx.last_wake_word_detection.is_none(),
                "last_wake_word_detection should be reset to None"
            );
            assert_eq!(
                ctx.vad_threshold, VAD_THRESHOLD,
                "vad_threshold should be reset to VAD_THRESHOLD"
            );
        }

        // ── VAD symmetry: enrollment only accumulates VAD-positive frames ──
        //
        // Verify that handle_enrollment_audio mirrors the detection pipeline:
        // only VAD-positive frames contribute to utterance_buf (mahbot-765).
        {
            voice_state().write().unwrap_poison().enabled = true;
            ctx.is_listening = true;
            ctx.enrollment_mode = true;
            ctx.clear_pipeline_buffers();

            // Send two silence frames (non-VAD) — should NOT accumulate
            reset_vad();
            let silence = vec![0.0f32; FRAME_LENGTH];
            handle_enrollment_audio(&silence, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            handle_enrollment_audio(&silence, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            assert!(
                ctx.utterance_buf.is_empty(),
                "VAD-gated enrollment: silence frames must not accumulate in utterance_buf"
            );
            assert!(
                !ctx.utterance_had_speech,
                "VAD-gated enrollment: silence must not set utterance_had_speech"
            );

            // Send speech frames — should accumulate audio immediately but
            // only set utterance_had_speech after 3 consecutive VAD-positive
            // frames (ENROLLMENT_VAD_CONSECUTIVE_REQUIRED = 3, mahbot-772).
            // Clear audio_buffer first so leftover silence from the previous
            // step doesn't create a mixed frame (preventing deterministic
            // frame counting).
            ctx.audio_buffer.clear();
            let speech = speech_frame();
            handle_enrollment_audio(&speech, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            // Audio accumulates on every VAD-positive frame...
            assert_eq!(
                ctx.utterance_buf.len(),
                HOP_LENGTH,
                "VAD-gated enrollment: utterance_buf should contain HOP_LENGTH samples per speech frame"
            );
            // ...but utterance_had_speech requires 3 consecutive positives.
            assert!(
                !ctx.utterance_had_speech,
                "VAD-gated enrollment: first speech frame must not set utterance_had_speech alone"
            );
            assert_eq!(
                ctx.vad_positives_in_a_row, 1,
                "vad_positives_in_a_row should be 1 after one speech frame"
            );

            // Second speech frame: still not enough for sustained speech.
            ctx.audio_buffer.clear();
            handle_enrollment_audio(&speech, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            assert_eq!(
                ctx.utterance_buf.len(),
                HOP_LENGTH * 2,
                "VAD-gated enrollment: utterance_buf should contain 2× HOP_LENGTH after two speech frames"
            );
            assert!(
                !ctx.utterance_had_speech,
                "VAD-gated enrollment: second speech frame must not set utterance_had_speech alone"
            );
            assert_eq!(
                ctx.vad_positives_in_a_row, 2,
                "vad_positives_in_a_row should be 2 after two speech frames"
            );

            // Third speech frame: sustained speech confirmed.
            ctx.audio_buffer.clear();
            handle_enrollment_audio(&speech, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            assert!(
                ctx.utterance_had_speech,
                "VAD-gated enrollment: third consecutive speech frame must set utterance_had_speech"
            );
            assert!(
                ctx.utterance_buf.len() > HOP_LENGTH * 3,
                "VAD-gated enrollment: utterance_buf should be > 3× HOP_LENGTH after three speech frames \
                 (pre-speech context prepended by mahbot-775 VAD asymmetry fix), got {}",
                ctx.utterance_buf.len(),
            );

            // Silence after speech should NOT add to utterance_buf, but
            // should be tracked for utterance end detection.  The silence
            // timeout clears utterance_buf into enrollment_pending.
            // Reset VAD to prevent pre-emphasis transient from previous
            // speech frames causing the silence to be classified as speech.
            ctx.audio_buffer.clear();
            reset_vad();
            ctx.utterance_silence_samples = SILENCE_THRESHOLD_SAMPLES - 1;
            handle_enrollment_audio(&silence, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
            // Utterance_buf is cleared by the timeout (taken into
            // enrollment_pending), so it should be empty.
            assert!(
                ctx.utterance_buf.is_empty(),
                "VAD-gated enrollment: utterance_buf should be cleared after silence timeout"
            );
            // The silence timeout should fire and store enrollment_pending
            assert!(
                ctx.enrollment_pending.is_some(),
                "VAD-gated enrollment: silence timeout must store utterance in enrollment_pending"
            );

            // Clean up
            voice_state().write().unwrap_poison().enabled = false;
            ctx.is_listening = false;
            ctx.enrollment_mode = false;
        }

        // ── Template invalidation: old enrollment_samples < NUM_ENROLLMENT_SAMPLES ──
        {
            let old_tpl = WakeWordTemplate {
                name: "legacy".into(),
                embeddings: vec![vec![1.0, 2.0]],
                threshold: 0.15,
                enrollment_samples: 0, // old format — field didn't exist, defaults to 0
            };
            let new_tpl = WakeWordTemplate {
                name: "new".into(),
                embeddings: vec![vec![3.0, 4.0]],
                threshold: 0.25,
                enrollment_samples: NUM_ENROLLMENT_SAMPLES,
            };
            let mut templates = WakeWordTemplates {
                templates: vec![old_tpl, new_tpl],
                minimum_matches: 1,
                ..Default::default()
            };
            let before = templates.templates.len();
            templates
                .templates
                .retain(|t| t.enrollment_samples >= NUM_ENROLLMENT_SAMPLES);
            let filtered = before - templates.templates.len();
            assert_eq!(
                filtered, 1,
                "old template with enrollment_samples=0 should be filtered out"
            );
            assert_eq!(
                templates.templates.len(),
                1,
                "only the valid template should remain"
            );
            assert_eq!(
                templates.templates[0].name, "new",
                "the remaining template should be 'new'"
            );
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

    // ── MAD-based threshold (mahbot-769) ────────────────────────────────

    #[test]
    fn test_mad_threshold_invariants() {
        // The acceptance threshold uses three protecting stages:
        //   1. threshold = median + K_MAD × mad          (MAD-based estimate)
        //   2. threshold = threshold.min(max(median*2, 0.20))   (absolute cap)
        //   3. threshold = threshold.max(MAD_THRESHOLD_FLOOR)    (floor, 0.10)
        //
        // The floor (reintroduced in mahbot-775) prevents overly restrictive
        // thresholds from extremely consistent enrollment samples.  Without it,
        // a careful quiet-room enrollment (MAD ≈ 0.005) produces threshold ≈
        // 0.025, which rejects slightly different voice states like "morning
        // voice" that add ~0.04 shift in embedding distance.
        //
        // These tests verify each stage without requiring VOICE_PIPELINE
        // initialisation — they exercise the same calculation that
        // finalize_enrollment applies to real enrollment data.

        // ── Helper: compute threshold from a sorted list of distances ──
        let compute = |sorted: &[f32]| {
            let median = super::median_of_sorted(sorted);
            let abs_devs: Vec<f32> = sorted.iter().map(|d| (d - median).abs()).collect();
            // abs_devs is already in the same order as sorted — sort it.
            let mut ad_sorted = abs_devs;
            ad_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let mad = super::median_of_sorted(&ad_sorted);
            let cap = (median * 2.0).max(0.20);
            (median + K_MAD * mad).clamp(super::MAD_THRESHOLD_FLOOR, cap)
        };

        // Case 1: nearly identical samples (all distances tiny).
        //   MAD-based: median=0.002, mad=0.001 → 0.002 + 3.0×0.001 = 0.005
        //   cap = max(0.002×2, 0.20) = 0.20
        //   floor = 0.10
        //   → threshold = 0.005.clamp(0.10, 0.20) = 0.10 (floor applies)
        let mut d = vec![
            0.001, 0.002, 0.001, 0.003, 0.002, 0.001, 0.002, 0.001, 0.003,
        ];
        d.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold = compute(&d);
        let expected = super::MAD_THRESHOLD_FLOOR; // 0.10
        assert!(
            (threshold - expected).abs() < 1e-6,
            "nearly-identical samples: expected floor {expected}, got {threshold}",
        );

        // Case 2: moderate spread.
        //   median = 0.15, mad = 0.02
        //   MAD-based: 0.15 + 3.0×0.02 = 0.21
        //   cap = max(0.15 × 2, 0.20) = 0.30
        //   floor = 0.10
        //   → threshold = 0.21.clamp(0.10, 0.30) = 0.21 (MAD dominates)
        let mut d = vec![0.10, 0.12, 0.13, 0.14, 0.15, 0.15, 0.16, 0.18, 0.20];
        d.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold = compute(&d);
        let median = super::median_of_sorted(&d);
        let abs_devs: Vec<f32> = d.iter().map(|x| (x - median).abs()).collect();
        let mut ad = abs_devs;
        ad.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = super::median_of_sorted(&ad);
        let mad_based = median + K_MAD * mad;
        assert!(
            (threshold - mad_based).abs() < 1e-6,
            "moderate spread: expected MAD-based {mad_based}, got {threshold}",
        );

        // Case 3: extreme spread — absolute cap applies.
        //   median = 0.18, mad = 0.10
        //   MAD-based: 0.18 + 3.0 × 0.10 = 0.48
        //   cap = max(0.18 × 2, 0.20) = 0.36
        //   floor = 0.10
        //   → threshold = 0.48.clamp(0.10, 0.36) = 0.36 (clamped to cap)
        let mut d = vec![0.08, 0.10, 0.12, 0.15, 0.18, 0.50, 0.55, 0.60, 0.65];
        d.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold = compute(&d);
        let median = super::median_of_sorted(&d);
        let cap = (median * 2.0).max(0.20);
        assert!(
            (threshold - cap).abs() < 1e-6,
            "extreme spread: expected cap {cap}, got {threshold}",
        );
    }

    /// Verify that a single outlier does not significantly affect the
    /// MAD-based threshold (MAD's 50% breakdown point robustness).
    #[test]
    fn test_mad_outlier_robustness() {
        // Without outlier: 9 similar distances → threshold should be tight.
        let mut clean = vec![0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18];
        clean.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let clean_median = super::median_of_sorted(&clean);
        let clean_abs: Vec<f32> = clean.iter().map(|d| (d - clean_median).abs()).collect();
        let mut ca = clean_abs;
        ca.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let clean_mad = super::median_of_sorted(&ca);
        let clean_threshold =
            (clean_median + K_MAD * clean_mad).min((clean_median * 2.0).max(0.20));

        // With one extreme outlier: 8 similar + 1 very different.
        // MAD should remain nearly unchanged (breakdown point = 50%).
        let mut with_outlier = vec![0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 5.0];
        with_outlier.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let outlier_median = super::median_of_sorted(&with_outlier);
        let outlier_abs: Vec<f32> = with_outlier
            .iter()
            .map(|d| (d - outlier_median).abs())
            .collect();
        let mut oa = outlier_abs;
        oa.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let outlier_mad = super::median_of_sorted(&oa);
        let outlier_threshold =
            (outlier_median + K_MAD * outlier_mad).min((outlier_median * 2.0).max(0.20));

        // The median should be the same (both sets have 0.14 as the 5th element).
        assert!(
            (clean_median - outlier_median).abs() < 1e-6,
            "median shifted from {clean_median} to {outlier_median}",
        );

        // MAD should be nearly identical (outlier's absolute deviation is
        // large but falls in the upper half and doesn't affect the median
        // of absolute deviations).
        let mad_diff = (clean_mad - outlier_mad).abs();
        assert!(
            mad_diff < 1e-6,
            "MAD changed from {clean_mad} to {outlier_mad} (diff={mad_diff}) — \
             single outlier should not affect MAD",
        );

        // Threshold should therefore remain nearly unchanged.
        let thresh_diff = (clean_threshold - outlier_threshold).abs();
        assert!(
            thresh_diff < 1e-6,
            "threshold changed from {clean_threshold} to {outlier_threshold} \
             (diff={thresh_diff}) — single outlier should not affect threshold",
        );
    }

    // ── MAD threshold floor (mahbot-775 Fix 1) ──────────────────────────

    /// Test that the MAD_THRESHOLD_FLOOR (0.10) is applied when the raw
    /// MAD-based threshold would be below 0.10.
    #[test]
    fn test_mad_threshold_floor() {
        // Highly consistent enrollment: MAD ≈ 0.005, raw MAD-based threshold
        // ≈ 0.025.  Without the floor, this would be the final threshold.
        // With the floor, it should be clamped to 0.10.
        let distances = vec![0.010, 0.011, 0.012, 0.013, 0.014, 0.015];

        // Replicate the calibrate_threshold calculation.
        let mut sorted = distances.clone();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let median = super::median_of_sorted(&sorted);
        let mut abs_devs: Vec<f32> = sorted.iter().map(|d| (d - median).abs()).collect();
        abs_devs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = super::median_of_sorted(&abs_devs);
        let cap = (median * 2.0).max(0.20);
        let raw = median + K_MAD * mad;
        let threshold = raw.clamp(super::MAD_THRESHOLD_FLOOR, cap);

        // Verify the floor is above the raw MAD-based value.
        assert!(
            raw < super::MAD_THRESHOLD_FLOOR,
            "raw threshold {raw:.4} should be below floor {:.4} for consistent data",
            super::MAD_THRESHOLD_FLOOR,
        );
        // Verify the final threshold is the floor.
        assert!(
            (threshold - super::MAD_THRESHOLD_FLOOR).abs() < 1e-6,
            "threshold should be floor {:.4}, got {threshold:.4}",
            super::MAD_THRESHOLD_FLOOR,
        );

        // Test with synthetic distances that produce a very low raw threshold.
        // median = 0.01, MAD = 0.0 (all distances identical)
        // raw = 0.01, cap = max(0.02, 0.20) = 0.20
        // threshold = 0.01.clamp(0.10, 0.20) = 0.10
        let identical = vec![0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
        let mut sorted2 = identical.clone();
        sorted2.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let median2 = super::median_of_sorted(&sorted2);
        let mut abs_devs2: Vec<f32> = sorted2.iter().map(|d| (d - median2).abs()).collect();
        abs_devs2.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mad2 = super::median_of_sorted(&abs_devs2);
        let cap2 = (median2 * 2.0).max(0.20);
        let raw2 = median2 + K_MAD * mad2;
        let threshold2 = raw2.clamp(super::MAD_THRESHOLD_FLOOR, cap2);

        assert!(
            raw2 < super::MAD_THRESHOLD_FLOOR,
            "raw threshold {raw2:.4} should be below floor for identical data",
        );
        assert!(
            (threshold2 - super::MAD_THRESHOLD_FLOOR).abs() < 1e-6,
            "identical distances should produce floor threshold, got {threshold2:.4}",
        );
    }

    // ── GRT selection — bimodal cluster detection (mahbot-775 Fix 2) ──

    /// Test that when enrollment contains a bimodal distribution (careful
    /// cluster + lazy cluster), the GRT selection prefers the stricter
    /// (lower MAD) cluster rather than the larger one.
    #[test]
    fn test_grt_prefers_careful_cluster() {
        // Create 10 synthetic enrollment samples with two clusters:
        //   Careful cluster (3 samples): tight intra-cluster, low distances
        //     → embeddings near [1, 0] with small variation
        //   Lazy cluster (7 samples): tight intra-cluster but looser than
        //     careful; more samples → would dominate naive GRT ranking
        //     → embeddings near [0.5, 0.866] with moderate variation
        //
        // The careful cluster has lower intra-cluster MAD, so the bimodal
        // correction should select templates from it.
        use std::f32::consts::FRAC_PI_3;

        let careful_angles = [-0.05_f32, 0.0, 0.05];
        let lazy_angles = [
            FRAC_PI_3 - 0.10,
            FRAC_PI_3 - 0.05,
            FRAC_PI_3,
            FRAC_PI_3 + 0.05,
            FRAC_PI_3 + 0.10,
            FRAC_PI_3 - 0.08,
            FRAC_PI_3 + 0.08,
        ];

        let mut samples: Vec<Vec<Vec<f32>>> = Vec::with_capacity(10);
        for angle in &careful_angles {
            samples.push(vec![vec![angle.cos(), angle.sin()]]);
        }
        for angle in &lazy_angles {
            samples.push(vec![vec![angle.cos(), angle.sin()]]);
        }

        let results = super::calibrate_threshold(&samples)
            .expect("calibrate_threshold with 10 samples should succeed");
        assert_eq!(
            results.len(),
            3,
            "should return top 3 templates from 10 samples",
        );

        // The most representative template (index 0) should come from the
        // careful cluster (angle ≈ 0, so embedding[0][0] ≈ 1.0, [0][1] ≈ 0.0).
        let (embeddings, threshold) = &results[0];
        assert!(
            threshold.is_finite() && *threshold > 0.0,
            "threshold should be finite and positive, got {threshold}",
        );

        // The first embedding dimension should be close to 1.0 (cos(0) ≈ 1.0)
        // and the second close to 0.0, confirming it's from the careful cluster.
        let first_emb = &embeddings[0];
        assert!(
            first_emb[0] > 0.95,
            "best template should be from careful cluster (cos ≈ 1.0), got emb[0]={}",
            first_emb[0],
        );
        assert!(
            first_emb[1].abs() < 0.1,
            "best template from careful cluster should have sin ≈ 0.0, got emb[1]={}",
            first_emb[1],
        );

        // All three returned templates should be from the careful cluster
        // (since it has only 3 samples, all should be selected).
        for (i, (emb, _)) in results.iter().enumerate() {
            let e = &emb[0];
            assert!(
                e[0] > 0.95,
                "template {i} should be from careful cluster, got emb[0]={}",
                e[0],
            );
        }
    }

    /// Test GRT selection with a near-threshold bimodal gap — angular
    /// separation just barely exceeding [`BIMODAL_GAP_THRESHOLD`] (0.04).
    ///
    /// This exercises the boundary condition where the clusters are close
    /// enough that their sorted avg_distances barely produce a detectable
    /// gap, unlike the wide-separation scenario (~60°) tested above.
    #[test]
    fn test_grt_prefers_careful_cluster_near_threshold() {
        // Near-threshold clusters: careful at 0 rad, lazy at 1.2 rad (~68.8°).
        // Cosine distance between centers ≈ 0.64, giving a gap in sorted
        // avg_distances of ~0.056 — comfortably above the 0.04 threshold.
        // Wider separation than 0.5 rad was needed because the 4:6 cluster
        // ratio produces a gap of 2d/9 (see bimodal gap analysis), requiring
        // cross-cluster distance d > 0.18 for gap > 0.04.  With d ≈ 0.64,
        // gap ≈ 0.056 reliably triggers bimodal detection.  This is still
        // "near-threshold" compared to the original test (~60° separation
        // gives gap ~0.5).
        let careful_angles = [-0.02_f32, 0.0, 0.02, 0.015];
        let lazy_angles = [
            1.2_f32 - 0.08,
            1.2_f32 - 0.05,
            1.2_f32,
            1.2_f32 + 0.05,
            1.2_f32 + 0.08,
            1.2_f32 + 0.04,
        ];

        let mut samples: Vec<Vec<Vec<f32>>> =
            Vec::with_capacity(careful_angles.len() + lazy_angles.len());
        for angle in &careful_angles {
            samples.push(vec![vec![angle.cos(), angle.sin()]]);
        }
        for angle in &lazy_angles {
            samples.push(vec![vec![angle.cos(), angle.sin()]]);
        }

        let results = super::calibrate_threshold(&samples)
            .expect("calibrate_threshold with 10 samples should succeed");
        assert_eq!(
            results.len(),
            3,
            "should return top 3 templates from 10 samples",
        );

        // The most representative template should come from the careful
        // cluster (angle near 0, so cos ≈ 1.0).
        let (embeddings, threshold) = &results[0];
        assert!(
            threshold.is_finite() && *threshold > 0.0,
            "threshold should be finite and positive, got {threshold}",
        );

        let first_emb = &embeddings[0];
        assert!(
            first_emb[0] > 0.95,
            "best template should be from careful cluster (cos ≈ 1.0), got emb[0]={}",
            first_emb[0],
        );
        assert!(
            first_emb[1].abs() < 0.1,
            "best template from careful cluster should have sin ≈ 0.0, got emb[1]={}",
            first_emb[1],
        );

        // All three returned templates should be from the careful cluster.
        for (i, (emb, _)) in results.iter().enumerate() {
            let e = &emb[0];
            assert!(
                e[0] > 0.95,
                "template {i} should be from careful cluster, got emb[0]={}",
                e[0],
            );
        }
    }

    // ── score_matching_templates — sliding window (mahbot-773) ──

    #[test]
    fn test_score_matching_templates_sliding_window() {
        // The sliding window strategy in score_matching_templates compares
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
            enrollment_samples: NUM_ENROLLMENT_SAMPLES,
        };

        let non_matching_tpl = WakeWordTemplate {
            name: "no_match".into(),
            embeddings: vec![
                vec![-1.0, 0.0, 0.0],
                vec![0.0, -1.0, 0.0],
                vec![0.0, 0.0, -1.0],
            ],
            threshold: 0.1, // very tight — only near-identical will match
            enrollment_samples: NUM_ENROLLMENT_SAMPLES,
        };

        let templates = WakeWordTemplates {
            templates: vec![matching_tpl, non_matching_tpl],
            minimum_matches: 1,
            ..Default::default()
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

        // score_matching_templates should window to the most recent
        // min(3, 5) = 3 embeddings and find matching_tpl scoring near 1.0
        // while non_matching_tpl scores near 0.0.
        let total_score = score_matching_templates(&live_sequence, &templates);

        // The matching template should contribute ~1.0 (identical vectors,
        // threshold=0.5, sigmoid(0, 0.5) ≈ 0.993).
        // The non-matching template should contribute ~0.0 (opposite vectors,
        // threshold=0.1, sigmoid(2.0, 0.1) ≈ 0.0).
        assert!(
            total_score > 0.9,
            "matching template should contribute ~1.0, got {total_score}"
        );
        assert!(
            total_score < 1.1,
            "non-matching template should contribute ~0.0, got {total_score}"
        );

        // Without sliding window, the noise embeddings would inflate DTW
        // past the threshold.  Verify that the window is actually being
        // applied by checking that the window size is 3 (the template length).
        let windowed_len = EXPECTED_WINDOW.min(live_sequence.len());
        assert_eq!(
            windowed_len, 3,
            "sliding window should be template length (3)"
        );
        // The actual window used inside score_matching_templates:
        let window = &live_sequence[live_sequence.len() - windowed_len..];
        assert_eq!(window.len(), 3, "windowed slice should have 3 elements");
    }
    #[test]
    fn test_score_matching_templates_no_match() {
        // Live sequence that does NOT match any template should return near 0.
        let tpl = WakeWordTemplate {
            name: "strict".into(),
            embeddings: vec![vec![1.0, 0.0, 0.0]],
            threshold: 0.01, // extremely tight — only exact match passes
            enrollment_samples: NUM_ENROLLMENT_SAMPLES,
        };
        let templates = WakeWordTemplates {
            templates: vec![tpl],
            minimum_matches: 1,
            ..Default::default()
        };

        // Opposite-direction embedding will have cosine distance ≈ 2.0
        // (cosine distance of opposite unit vectors = 1 - (-1/1) = 2.0).
        // sigmoid(10 * (2.0 - 0.01)) ≈ e^(-19.9) ≈ 0.0
        let live = vec![vec![-1.0, 0.0, 0.0]];
        let total_score = score_matching_templates(&live, &templates);
        assert!(
            total_score < 0.01,
            "opposite vectors should score near 0, got {total_score}"
        );
    }

    #[test]
    fn test_score_matching_templates_empty_live() {
        // Empty live sequence should not panic and should return 0.0.
        let tpl = WakeWordTemplate {
            name: "any".into(),
            embeddings: vec![vec![1.0, 0.0, 0.0]],
            threshold: 10.0, // would match anything
            enrollment_samples: NUM_ENROLLMENT_SAMPLES,
        };
        let templates = WakeWordTemplates {
            templates: vec![tpl],
            minimum_matches: 1,
            ..Default::default()
        };

        // Empty live — score must be 0.0
        let total_score = score_matching_templates(&[], &templates);
        assert!(
            (total_score - 0.0).abs() < f32::EPSILON,
            "empty live should score 0.0, got {total_score}"
        );
    }

    #[test]
    fn test_score_matching_templates_multi_template() {
        // Multi-template scoring (K=3): verify that the total score
        // correctly reflects how many templates match.  Templates that
        // match well contribute ~1.0 each, non-matching contribute ~0.0.

        // Template 1: matches target pattern
        let tpl1 = WakeWordTemplate {
            name: "tpl1".into(),
            embeddings: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            threshold: 0.5,
            enrollment_samples: NUM_ENROLLMENT_SAMPLES,
        };
        // Template 2: matches target pattern (same direction)
        let tpl2 = WakeWordTemplate {
            name: "tpl2".into(),
            embeddings: vec![vec![0.9, 0.1, 0.0], vec![0.1, 0.9, 0.0]],
            threshold: 0.5,
            enrollment_samples: NUM_ENROLLMENT_SAMPLES,
        };
        // Template 3: does NOT match (opposite direction)
        let tpl3 = WakeWordTemplate {
            name: "tpl3".into(),
            embeddings: vec![vec![-1.0, 0.0, 0.0], vec![0.0, -1.0, 0.0]],
            threshold: 0.1,
            enrollment_samples: NUM_ENROLLMENT_SAMPLES,
        };

        // Live sequence matching templates 1 and 2 (not 3)
        let live = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let templates = WakeWordTemplates {
            templates: vec![tpl1, tpl2, tpl3],
            minimum_matches: 2,
            ..Default::default()
        };
        let total_score = score_matching_templates(&live, &templates);

        // tpl1 (identical vectors): ~0.993
        // tpl2 (cosine dist ≈ 0.1, threshold=0.5): ~0.982
        // tpl3 (cosine dist ≈ 2.0, threshold=0.1): ~0.0
        // Total ≈ 1.975
        assert!(
            total_score > 1.5,
            "two matching templates should score > 1.5, got {total_score}"
        );
        assert!(
            total_score < 2.5,
            "non-matching template should add ~0, got {total_score}"
        );
    }

    // ── process_wake_word_score — rolling window detection ────────────

    #[test]
    fn test_process_wake_word_score_below_reset_clears_window() {
        // When total_score < NO_MATCH_RESET_THRESHOLD, the window should be
        // cleared and the function should return false (no detection).
        let mut window = vec![0.9, 0.8]; // previously accumulated scores
        let detected = process_wake_word_score(0.1, &mut window, 1);
        assert!(!detected, "below-reset score should not detect");
        assert!(
            window.is_empty(),
            "below-reset score should clear the window"
        );
    }

    #[test]
    fn test_process_wake_word_score_below_reset_empty_window_stays_empty() {
        // When total_score < NO_MATCH_RESET_THRESHOLD and the window is
        // already empty, it should stay empty (no-op).
        let mut window: Vec<f32> = Vec::new();
        let detected = process_wake_word_score(0.1, &mut window, 1);
        assert!(!detected);
        assert!(window.is_empty(), "window should remain empty");
    }

    #[test]
    fn test_process_wake_word_score_single_frame_not_enough() {
        // A single good frame (score above NO_MATCH_RESET_THRESHOLD) with
        // minimum_matches=2 needs rolling_sum >= 2*3*0.65 = 3.9.  One frame
        // of score 1.5 is not enough.
        let mut window: Vec<f32> = Vec::new();
        let detected = process_wake_word_score(1.5, &mut window, 2);
        assert!(!detected, "single frame should not reach M=2 threshold");
        assert_eq!(window.len(), 1, "good frame should be appended to window");
        assert!((window[0] - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_process_wake_word_score_accumulation_across_frames() {
        // M=1, ROLLING_WINDOW_N=3 → threshold = 1 * 3 * 0.65 = 1.95.
        // First frame of 0.7 is not enough.  Second frame of 0.8 makes
        // rolling_sum = 1.5, still not enough.  Third frame of 0.6 makes
        // rolling_sum = 2.1 > 1.95 → detection.
        let mut window: Vec<f32> = Vec::new();

        // Frame 1: score 0.7 → rolling_sum = 0.7 < 1.95 → no detection
        let detected = process_wake_word_score(0.7, &mut window, 1);
        assert!(!detected, "frame 1 should not detect yet");
        assert_eq!(window.len(), 1);

        // Frame 2: score 0.8 → rolling_sum = 0.7 + 0.8 = 1.5 < 1.95
        let detected = process_wake_word_score(0.8, &mut window, 1);
        assert!(!detected, "frame 2 should not detect yet");
        assert_eq!(window.len(), 2);

        // Frame 3: score 0.6 → rolling_sum = 0.7 + 0.8 + 0.6 = 2.1 >= 1.95 → detect
        let detected = process_wake_word_score(0.6, &mut window, 1);
        assert!(detected, "frame 3 should trigger detection");
        assert_eq!(
            window.len(),
            3,
            "window should have 3 scores before caller clears it"
        );
    }

    #[test]
    fn test_process_wake_word_score_window_trims_to_n() {
        // With ROLLING_WINDOW_N=3, after 5 good scores the window should
        // contain only the 3 most recent.
        let mut window: Vec<f32> = Vec::new();
        for _ in 0..5 {
            process_wake_word_score(0.8, &mut window, 3);
        }
        assert_eq!(
            window.len(),
            ROLLING_WINDOW_N,
            "window should be trimmed to ROLLING_WINDOW_N"
        );
        // All entries should be 0.8
        for (i, &v) in window.iter().enumerate() {
            assert!(
                (v - 0.8).abs() < f32::EPSILON,
                "window[{i}] should be 0.8, got {v}"
            );
        }
    }

    #[test]
    fn test_process_wake_word_score_borderline_below_reset() {
        // Exactly at NO_MATCH_RESET_THRESHOLD should NOT reset (it's >= not >).
        let mut window: Vec<f32> = Vec::new();
        let detected = process_wake_word_score(NO_MATCH_RESET_THRESHOLD, &mut window, 1);
        // Single score = 0.3, threshold = 1.95 for M=1 → no detection.
        assert!(!detected, "at-threshold score should not reset");
        assert_eq!(window.len(), 1, "at-threshold score should be appended");
    }

    #[test]
    fn test_process_wake_word_score_exact_threshold_detects() {
        // Rolling sum exactly equal to match_threshold should detect.
        // M=1 → threshold = 1 * 3 * 0.65 = 1.95.
        // Three frames of 0.65 each → rolling_sum = 1.95 exactly.
        let mut window: Vec<f32> = Vec::new();
        process_wake_word_score(0.65, &mut window, 1);
        process_wake_word_score(0.65, &mut window, 1);
        let detected = process_wake_word_score(0.65, &mut window, 1);
        assert!(
            detected,
            "rolling sum exactly at match_threshold should detect"
        );
    }

    // ── calibrate_threshold (pure) and median_of_sorted ────────────────

    /// Table-driven test for [`median_of_sorted`].
    #[test]
    fn test_median_of_sorted() {
        struct Case {
            input: Vec<f32>,
            expected: f32,
        }
        let cases = [
            Case {
                input: vec![42.0],
                expected: 42.0,
            },
            Case {
                input: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                expected: 3.0,
            },
            Case {
                input: vec![1.0, 2.0, 3.0, 4.0],
                expected: 2.5,
            },
            Case {
                input: vec![10.0, 20.0, 30.0],
                expected: 20.0,
            },
            Case {
                input: vec![0.0, 100.0],
                expected: 50.0,
            },
        ];
        for (i, case) in cases.iter().enumerate() {
            let got = super::median_of_sorted(&case.input);
            assert!(
                (got - case.expected).abs() < 1e-6,
                "case {i}: median_of_sorted({:?}) = {got}, expected {}",
                case.input,
                case.expected,
            );
        }
    }

    /// Calling [`median_of_sorted`] on an empty slice should panic
    /// (precondition violation — guards against invariant drift).
    #[test]
    #[should_panic(expected = "empty slice")]
    fn test_median_of_sorted_empty_panics() {
        let empty: Vec<f32> = Vec::new();
        super::median_of_sorted(&empty);
    }

    /// Verify that [`calibrate_threshold`] picks the most representative
    /// sample as the best template.
    #[test]
    fn test_best_template_selection() {
        // Three synthetic samples with 2-dimensional embeddings so
        // cosine_distance gives predictable results.
        //
        // Sample 0: [1,0] (unit x-axis)
        // Sample 1: [0,1] (unit y-axis) — orthogonal to 0, "far" from it
        // Sample 2: [0.7, 0.7] (≈45°) — roughly equidistant between 0 and 1
        //
        // Expected: sample 2 (45°) has the lowest average distance and should
        // be selected as best template.
        //
        // dtw(0↔1) = cosine_dist([1,0], [0,1]) = 1.0  (orthogonal → distance 1)
        // dtw(0↔2) = cosine_dist([1,0], [0.7,0.7]) = 1 - 0.7/√0.98 ≈ 1 - 0.707 = 0.293
        // dtw(1↔2) = cosine_dist([0,1], [0.7,0.7]) = 1 - 0.7/√0.98 ≈ 1 - 0.707 = 0.293
        //
        // Avg distance for sample 0: (1.0 + 0.293) / 2 ≈ 0.646
        // Avg distance for sample 1: (1.0 + 0.293) / 2 ≈ 0.646
        // Avg distance for sample 2: (0.293 + 0.293) / 2 ≈ 0.293
        //
        // The embeddings are 2D in this synthetic test, so we verify
        // that the returned template is exactly sample 2's embeddings.
        let samples = vec![
            vec![vec![1.0, 0.0]],
            vec![vec![0.0, 1.0]],
            vec![vec![0.7, 0.7]],
        ];

        let results =
            super::calibrate_threshold(&samples).expect("calibrate_threshold should succeed");
        // With K=3, all three samples are returned as templates.
        assert_eq!(results.len(), 3, "should return all 3 samples as templates");

        // The most representative template (index 0) should be sample 2 (45°).
        let (embeddings, threshold) = &results[0];

        // Threshold must be finite and positive (sample 2 is representative).
        assert!(
            threshold.is_finite() && *threshold > 0.0,
            "threshold should be finite and positive, got {threshold}",
        );
        assert_eq!(
            *embeddings,
            vec![vec![0.7, 0.7]],
            "best template should be sample 2 (45°)",
        );
    }

    /// Full end-to-end test of [`calibrate_threshold`] with synthetic
    /// two-cluster embeddings.
    #[test]
    fn test_enrollment_threshold_with_known_distances() {
        // Build 9 × 96-dim unit vectors in two clusters:
        //   Cluster A (5 samples): near [1, 0, 0, ...]
        //   Cluster B (4 samples): near [-1, 0, 0, ...]
        //
        // The larger cluster (A) should supply the best template, and the
        // threshold should reflect intra-cluster distances rather than
        // inter-cluster distances (best-template selection eliminates the
        // inter-cluster noise from the statistic).
        let mut samples: Vec<Vec<Vec<f32>>> = Vec::with_capacity(9);
        for angle in [0.0_f32, 0.05, 0.10, -0.05, -0.10] {
            let mut emb = vec![0.0_f32; EMBEDDING_DIM];
            emb[0] = angle.cos();
            emb[1] = angle.sin();
            samples.push(vec![emb]);
        }
        for angle in [
            std::f32::consts::PI,
            std::f32::consts::PI + 0.08_f32,
            std::f32::consts::PI - 0.08,
            std::f32::consts::PI + 0.16,
        ] {
            let mut emb = vec![0.0_f32; EMBEDDING_DIM];
            emb[0] = angle.cos();
            emb[1] = angle.sin();
            samples.push(vec![emb]);
        }

        let results = super::calibrate_threshold(&samples)
            .expect("calibrate_threshold should succeed with 9 samples");
        assert_eq!(
            results.len(),
            3,
            "should return top 3 templates from 9 samples"
        );

        // The most representative template (index 0) should come from the
        // larger cluster (A: near [1,0,…]) rather than the smaller cluster
        // (B: near [-1,0,…]).  Since cluster-A embeddings have a positive
        // first dimension, an embedding[0][0] > 0.9 confirms the best
        // template is from cluster A.
        let (embeddings, threshold) = &results[0];
        // The threshold must be finite and positive.
        assert!(
            threshold.is_finite() && *threshold > 0.0,
            "threshold should be finite and positive, got {threshold}",
        );
        assert!(
            embeddings[0][0] > 0.9,
            "best template should be from the larger cluster (A), got [0][0]={}",
            embeddings[0][0],
        );
    }

    // ── Utterance buffer truncation (mahbot-755 Fix 1) ─────────────────

    #[test]
    #[serial_test::serial(voice)]
    fn test_enrollment_utterance_tracks_speech_boundary() {
        reset_vad();
        // Initialize VOICE_PIPELINE (harmless if already set by another test).
        let _ = VOICE_PIPELINE.set(RwLock::new(VoicePipelineState {
            enabled: true,
            status: VoiceStatus::Disabled,
            templates: Arc::new(WakeWordTemplates::default()),
            enrollment_buffer: Vec::new(),
            negative_audio_chunks: Vec::new(),

            cmd_tx: None,
        }));

        // Verify that utterance_speech_end_len is updated on each speech
        // frame and that it correctly marks the boundary between speech
        // and trailing silence in the utterance buffer.
        let mut ctx = PipelineCtx::new();
        ctx.is_listening = true;
        ctx.enrollment_mode = true;

        // A speech-like 512-sample frame with harmonic content that Earshot
        // classifies as speech.
        let speech_frame = speech_frame();

        // Feed 3 speech frames cumulative (call 1 processes 1 frame,
        // call 2 processes 2 frames due to leftover samples in audio_buffer,
        // totaling 3 frames = ENROLLMENT_VAD_CONSECUTIVE_REQUIRED).
        handle_enrollment_audio(&speech_frame, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
        // After 1 frame: audio accumulated but utterance_had_speech not yet set.
        assert!(
            !ctx.utterance_had_speech,
            "after first speech frame: utterance_had_speech requires 3 consecutive positives"
        );
        assert_eq!(
            ctx.utterance_speech_end_len, 0,
            "after first speech frame: speech_end_len should still be 0 (no sustained speech)"
        );
        assert_eq!(ctx.vad_positives_in_a_row, 1);
        assert_eq!(ctx.utterance_buf.len(), HOP_LENGTH);

        handle_enrollment_audio(&speech_frame, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
        // With 512-sample input, call 2 processes 2 frames (one from leftover
        // + one new), reaching 3 total — sustained speech confirmed.
        assert!(
            ctx.utterance_had_speech,
            "after 3 cumulative speech frames: utterance_had_speech should be set"
        );
        assert_eq!(
            ctx.utterance_speech_end_len,
            ctx.utterance_buf.len(),
            "after sustained speech: speech_end_len should equal buf len"
        );

        // Record the speech-only length before introducing silence.
        let speech_only_len = ctx.utterance_speech_end_len;
        assert!(speech_only_len > 0, "should have accumulated speech data");

        // ── Feed first silence frame to capture trailing speech (mahbot-775 Fix 3) ──
        // We must NOT pre-set utterance_silence_samples yet — the first VAD-negative
        // frame after speech is the moment the raw ring still holds the quieter
        // trailing phonemes.  The post-speech tail is captured here, before the ring
        // fills with silence during the 1.5s timeout window.
        ctx.audio_buffer.clear();
        reset_vad();
        let silence_frame = vec![0.0f32; FRAME_LENGTH];
        handle_enrollment_audio(&silence_frame, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);

        // Verify the trailing speech tail was captured.
        assert!(
            !ctx.post_speech_tail.is_empty(),
            "post_speech_tail should be captured at first silence after speech",
        );

        // Record the utterance length before the timeout fires.
        // utterance_buf is unchanged by the first silence frame (only
        // post_speech_tail was captured), so len_before_timeout == speech_only_len.
        let len_before_timeout = ctx.utterance_buf.len();

        // Manually set silence_samples just below the threshold so the next
        // VAD-negative frame (adding HOP_LENGTH) triggers the timeout immediately.
        ctx.utterance_silence_samples = SILENCE_THRESHOLD_SAMPLES.saturating_sub(HOP_LENGTH);

        // Reset the VAD detector for the second silence frame.
        reset_vad();

        // Feed a second silence frame to trigger the timeout.
        handle_enrollment_audio(&silence_frame, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);

        // After truncation: enrollment_pending should contain the speech with
        // post-speech tail appended, and utterance_buf should be empty.
        assert!(
            ctx.enrollment_pending.is_some(),
            "silence timeout should store utterance in enrollment_pending"
        );
        let pending = ctx.enrollment_pending.as_ref().unwrap();
        assert!(
            pending.len() > len_before_timeout,
            "enrollment_pending ({}) should be larger than the utterance before timeout ({}) \
             because post-speech tail was appended (mahbot-775 Fix 3)",
            pending.len(),
            len_before_timeout,
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
            ctx.utterance_silence_samples == 0,
            "silence_samples should reset"
        );
        assert_eq!(
            ctx.utterance_speech_end_len, 0,
            "speech_end_len should reset"
        );
    }

    #[test]
    #[serial_test::serial(voice)]
    fn test_enrollment_audio_break_with_consumed_frames() {
        reset_vad();
        // Regression test for the fix replacing audio_buffer.clear() with
        // consumed = len at the silence-threshold break point.
        //
        // When the break fires on a frame after the first within a single call
        // (i.e., consumed > 0), the old `clear()` + post-loop `drain(..consumed)`
        // sequence would panic because the clear() emptied the buffer before the
        // drain.  The fix marks everything as consumed (consumed = len) so the
        // post-loop drain removes all samples in one go.
        //
        // This test exercises the break on frame 2 of a 2-frame buffer — the
        // break fires after consumed has been advanced to HOP_LENGTH, so
        // consumed > 0 at the break point.
        let _ = VOICE_PIPELINE.set(RwLock::new(VoicePipelineState {
            enabled: true,
            status: VoiceStatus::Disabled,
            templates: Arc::new(WakeWordTemplates::default()),
            enrollment_buffer: Vec::new(),
            negative_audio_chunks: Vec::new(),

            cmd_tx: None,
        }));

        let mut ctx = PipelineCtx::new();
        ctx.is_listening = true;
        ctx.enrollment_mode = true;

        // Pre-fill audio_buffer with 2 frames of silence (1024 samples).
        // The call to handle_enrollment_audio below will process these frames
        // without any additional input (empty slice).
        ctx.audio_buffer = vec![0.0f32; FRAME_LENGTH * 2];

        // Simulate prior speech by setting utterance_had_speech directly.
        ctx.utterance_had_speech = true;

        // Set silence_samples so the threshold is reached on the 2nd frame:
        //
        //   Frame 1 (samples 0-511): silence
        //     → utterance_silence_samples += 256 → 23744 (< 24000, no break)
        //     → consumed = 256
        //
        //   Frame 2 (samples 256-767): silence
        //     → utterance_silence_samples += 256 → 24000 (>= 24000, BREAK)
        //     → consumed = len (= 1024)  ← the fix
        //
        //   Post-loop: drain(..1024) removes all samples without panic.
        ctx.utterance_silence_samples = SILENCE_THRESHOLD_SAMPLES - 2 * HOP_LENGTH;

        // Reset VAD immediately before processing so no other test's VAD
        // state contaminates this silence classification.
        reset_vad();

        // Process the pre-filled buffer with no additional samples.
        handle_enrollment_audio(&[], &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);

        // Verify the break path was reached and state was properly reset.
        assert!(
            ctx.enrollment_pending.is_some(),
            "silence threshold should have triggered break and stored utterance"
        );
        assert!(
            ctx.utterance_buf.is_empty(),
            "utterance_buf should be empty after mem::take in break path"
        );
        assert!(
            !ctx.utterance_had_speech,
            "utterance_had_speech should be reset after break"
        );
        assert_eq!(
            ctx.utterance_silence_samples, 0,
            "utterance_silence_samples should be reset after break"
        );
        assert_eq!(
            ctx.utterance_speech_end_len, 0,
            "utterance_speech_end_len should be reset after break"
        );
        assert!(
            ctx.audio_buffer.is_empty(),
            "audio_buffer should be fully drained by post-loop drain"
        );
    }

    // ── VAD asymmetry padding (mahbot-775 Fix 3) ────────────────────────

    /// Test that enrollment utterance includes pre-speech and post-speech
    /// context padding from the raw audio ring to reduce the VAD-threshold
    /// asymmetry between enrollment (VAD=0.85) and live detection (VAD=0.5).
    #[test]
    #[serial_test::serial(voice)]
    fn test_vad_asymmetry_padding() {
        reset_vad();
        let _ = VOICE_PIPELINE.set(RwLock::new(VoicePipelineState {
            enabled: true,
            status: VoiceStatus::Disabled,
            templates: Arc::new(WakeWordTemplates::default()),
            enrollment_buffer: Vec::new(),
            negative_audio_chunks: Vec::new(),

            cmd_tx: None,
        }));

        let mut ctx = PipelineCtx::new();
        ctx.is_listening = true;
        ctx.enrollment_mode = true;
        ctx.clear_pipeline_buffers();

        // Feed pre-speech silence to fill the raw audio ring.
        let silence_frame = vec![0.0f32; FRAME_LENGTH];
        for _ in 0..3 {
            handle_enrollment_audio(&silence_frame, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
        }
        // The raw ring should now contain silence audio.
        assert!(
            !ctx.raw_audio_ring.is_empty(),
            "raw_audio_ring should contain pre-speech silence",
        );

        // Feed speech frames to trigger sustained speech and pre-padding.
        let speech = speech_frame();
        // Feed 80ms of speech (5 frames to reach sustained speech + continue).
        for _ in 0..6 {
            ctx.audio_buffer.clear();
            handle_enrollment_audio(&speech, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);
        }
        // utterance_had_speech should now be true.
        assert!(
            ctx.utterance_had_speech,
            "utterance_had_speech should be true"
        );

        // utterance_buf should be larger than the speech-only content
        // because pre-speech context was prepended from the raw ring.
        let speech_content = HOP_LENGTH * 6; // 6 speech frames
        assert!(
            ctx.utterance_buf.len() > speech_content,
            "utterance_buf ({}) should exceed speech-only content ({}) \
             because pre-speech context was prepended",
            ctx.utterance_buf.len(),
            speech_content,
        );

        // The raw ring should still be non-empty (it accumulates all audio).
        assert!(
            !ctx.raw_audio_ring.is_empty(),
            "raw_audio_ring should continue accumulating after speech",
        );

        // ── Feed first silence frame to capture trailing speech ─────────
        // This is the FIRST VAD-negative frame after speech.  The raw audio
        // ring still contains the trailing phonemes.  Set utterance_silence_samples
        // to 0 so the first-silence detection fires (mahbot-775 Fix 3).
        reset_vad();
        ctx.utterance_silence_samples = 0;
        ctx.audio_buffer.clear();
        handle_enrollment_audio(&silence_frame, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);

        // Verify the trailing speech tail was captured from the raw ring.
        assert!(
            !ctx.post_speech_tail.is_empty(),
            "post_speech_tail should be captured from raw_audio_ring at first silence",
        );

        // Save the utterance length before the timeout fires.
        let len_before_timeout = ctx.utterance_buf.len();

        // ── Feed second silence frame to trigger the timeout ─────────────
        // Pre-set silence_samples just below threshold so one more VAD-negative
        // frame (adding HOP_LENGTH) fires the timeout.
        ctx.utterance_silence_samples = SILENCE_THRESHOLD_SAMPLES.saturating_sub(HOP_LENGTH);
        ctx.audio_buffer.clear();
        handle_enrollment_audio(&silence_frame, &mut ctx, 0, NUM_ENROLLMENT_SAMPLES);

        // enrollment_pending should contain the padded utterance.
        let pending = ctx
            .enrollment_pending
            .as_ref()
            .expect("silence timeout should store utterance in enrollment_pending");
        assert!(
            pending.len() > len_before_timeout,
            "enrollment_pending ({}) should be larger than the utterance before timeout ({}) \
             because post-speech tail was appended (mahbot-775 Fix 3)",
            pending.len(),
            len_before_timeout,
        );

        // post_speech_tail should be cleared after the timeout consumed it.
        assert!(
            ctx.post_speech_tail.is_empty(),
            "post_speech_tail should be cleared after timeout consumed it",
        );

        // Verify raw_audio_ring is cleared on pipeline reset.
        ctx.clear_pipeline_buffers();
        assert!(
            ctx.raw_audio_ring.is_empty(),
            "raw_audio_ring should be cleared after pipeline reset",
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
            negative_audio_chunks: Vec::new(),

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
            negative_audio_chunks: Vec::new(),

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
            negative_audio_chunks: Vec::new(),

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
            enrollment_samples: NUM_ENROLLMENT_SAMPLES,
        };
        let json = serde_json::to_string(&tpl).unwrap();
        let deserialized: WakeWordTemplate = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "hello");
        assert_eq!(deserialized.embeddings, vec![vec![1.0, 2.0, 3.0]]);
        assert!((deserialized.threshold - 0.5).abs() < 1e-6);
        assert_eq!(
            deserialized.enrollment_samples, NUM_ENROLLMENT_SAMPLES,
            "enrollment_samples must survive roundtrip"
        );
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
        // enrollment_samples was missing → default 0
        assert_eq!(
            tpl.enrollment_samples, 0,
            "enrollment_samples should default to 0 when missing (old format)"
        );
    }

    #[test]
    fn test_template_serde_empty_json_uses_defaults() {
        // Minimal JSON: all fields missing → all defaults.
        let tpl: WakeWordTemplate = serde_json::from_str("{}").unwrap();
        assert!(tpl.name.is_empty());
        assert!(tpl.embeddings.is_empty());
        assert!((tpl.threshold - 0.0).abs() < 1e-6);
        assert_eq!(
            tpl.enrollment_samples, 0,
            "enrollment_samples should default to 0 for empty JSON"
        );
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
        // enrollment_samples was missing → default 0
        assert_eq!(tpl.enrollment_samples, 0);
    }

    #[test]
    fn test_templates_serde_roundtrip() {
        let templates = WakeWordTemplates {
            templates: vec![WakeWordTemplate {
                name: "alpha".into(),
                embeddings: vec![vec![1.0]],
                threshold: 0.5,
                enrollment_samples: NUM_ENROLLMENT_SAMPLES,
            }],
            minimum_matches: 1,
            ..Default::default()
        };
        let json = serde_json::to_string(&templates).unwrap();
        let deserialized: WakeWordTemplates = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.templates.len(), 1);
        assert_eq!(deserialized.templates[0].name, "alpha");
        assert_eq!(
            deserialized.templates[0].enrollment_samples, NUM_ENROLLMENT_SAMPLES,
            "enrollment_samples must survive WakeWordTemplates roundtrip"
        );
        // minimum_matches survives roundtrip (1 is the default)
        assert_eq!(
            deserialized.minimum_matches, 1,
            "minimum_matches must survive WakeWordTemplates roundtrip"
        );
    }

    #[test]
    fn test_templates_serde_empty_list() {
        let tpl: WakeWordTemplates = serde_json::from_str(r#"{"templates":[]}"#).unwrap();
        assert!(tpl.templates.is_empty());
        assert_eq!(
            tpl.minimum_matches, 1,
            "minimum_matches must default to 1 for backward compat"
        );
    }

    // ── broadcast_voice_transcript ────────────────────────────────────
    //
    // These tests verify that broadcast_voice_transcript correctly emits
    // a ChatEvent::Message with direction:User on CHAT_BROADCAST.

    /// Set (or reuse) the global CHAT_BROADCAST and return a receiver
    /// with any stale messages drained.
    fn setup_chat_broadcast() -> tokio::sync::broadcast::Receiver<crate::ChatEvent> {
        // Use get_or_init to avoid a TOCTOU race: is_none() + set() can lose
        // the sender if another test sets the global between the check and
        // the set, leaving the receiver orphaned on a sender-less channel
        // (panicking with "broadcast channel closed").
        crate::CHAT_BROADCAST.get_or_init(|| {
            let (tx, _rx) = tokio::sync::broadcast::channel(256);
            tx
        });
        let mut rx = crate::CHAT_BROADCAST.get().unwrap().subscribe();
        // Drain any stale messages from previous tests sharing the same
        // global broadcast sender.
        while rx.try_recv().is_ok() {}
        rx
    }

    /// Wait for a ChatEvent::Message with the given content on the broadcast
    /// channel. Filters out events from other tests sharing the global sender.
    async fn recv_chat_event_by_content(
        rx: &mut tokio::sync::broadcast::Receiver<crate::ChatEvent>,
        expected_content: &str,
    ) -> crate::ChatEvent {
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        if let crate::ChatEvent::Message { ref content, .. } = event {
                            if content == expected_content {
                                return event;
                            }
                        }
                        continue; // event from another test — skip
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        // Messages were dropped because of a full broadcast channel
                        // (capacity 256, matching production).  Since parallel tests
                        // share the global sender, bursts can still overflow — but
                        // filtering by content makes a lagged skip harmless: the
                        // expected event may still be in the buffer, and if not,
                        // the timeout will catch it.
                        continue;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        panic!("broadcast channel closed");
                    }
                }
            }
        })
        .await
        .expect("timeout waiting for expected ChatEvent::Message")
    }

    /// Empty user_name path — broadcast-only, no chat_history persist.
    /// Verifies that the event appears on CHAT_BROADCAST with the correct
    /// direction, content, and workspace.
    #[tokio::test]
    async fn test_broadcast_voice_transcript_empty_user() {
        let mut rx = setup_chat_broadcast();

        let transcript = "test voice command";
        broadcast_voice_transcript(transcript, "", "testws").await;

        let received = recv_chat_event_by_content(&mut rx, transcript).await;

        match received {
            crate::ChatEvent::Message {
                content,
                direction,
                user_name,
                workspace,
                ..
            } => {
                assert_eq!(content, transcript, "transcript text must match");
                assert_eq!(
                    direction,
                    crate::ChatDirection::User,
                    "voice transcript must be a User-direction event"
                );
                assert_eq!(
                    user_name, "",
                    "empty user_name path must produce empty user_name"
                );
                assert_eq!(workspace, "testws", "workspace must match");
            }
            _ => panic!("Expected ChatEvent::Message, got a different variant"),
        }
    }

    /// Non-empty user_name path — full broadcast + persist.
    /// Verifies the event appears on CHAT_BROADCAST and that a matching
    /// record is written to chat_history.
    #[tokio::test]
    async fn test_broadcast_voice_transcript_known_user() {
        crate::util::test::init_test_stores().await;
        let mut rx = setup_chat_broadcast();

        let transcript = "hello from voice";
        broadcast_voice_transcript(transcript, "admin", "default").await;

        // Verify broadcast event (content-filtered against parallel-test noise).
        let received = recv_chat_event_by_content(&mut rx, transcript).await;

        match received {
            crate::ChatEvent::Message {
                content,
                direction,
                user_name,
                workspace,
                ..
            } => {
                assert_eq!(content, transcript);
                assert_eq!(direction, crate::ChatDirection::User);
                assert_eq!(user_name, "admin");
                assert_eq!(workspace, "default");
            }
            _ => panic!("Expected ChatEvent::Message, got a different variant"),
        }

        // Verify it was also persisted to chat_history.
        let store = crate::chat_history::store();
        let rows = store
            .conn
            .query("SELECT user_name, content, direction, channel FROM chat_history WHERE user_name = 'admin' ORDER BY created_at DESC LIMIT 1", ())
            .await
            .expect("query chat_history");
        assert!(!rows.is_empty(), "should have a matching chat_history row");
        let row = &rows[0];
        assert_eq!(row.get::<String>(0).unwrap(), "admin");
        assert_eq!(row.get::<String>(1).unwrap(), transcript);
        assert_eq!(row.get::<String>(2).unwrap(), "user");
        assert_eq!(row.get::<String>(3).unwrap(), "voice");
    }

    // ── Sakoe-Chiba band constraint (mahbot-770 Fix 6) ─────────────────

    /// Verify the Sakoe-Chiba band window calculation for various sequence
    /// lengths.  window = ceil(0.05 × max(n, m)), minimum 2 (mahbot-774).
    #[test]
    fn test_sakoe_chiba_window_calculation() {
        // Replicate the window formula from dtw_distance.
        let band_window = |n: usize, m: usize| -> usize {
            ((SAKOE_CHIBA_BAND_FRACTION * n.max(m) as f64)
                .ceil()
                .max(2.0)) as usize
        };

        // Equal-length small sequences: floor of 2 applies.
        assert_eq!(band_window(3, 3), 2); // ceil(0.05×3) = ceil(0.15) = 1, max(2)
        assert_eq!(band_window(1, 1), 2); // ceil(0.05×1) = ceil(0.05) = 1, max(2)
        assert_eq!(band_window(2, 2), 2); // ceil(0.05×2) = ceil(0.1) = 1, max(2)

        // Larger: 5% of 100 = 5, exceeds floor.
        assert_eq!(band_window(100, 100), 5);

        // Asymmetric: max dominates even for 1-vs-20 (ceil(1.0)=1 → floor 2).
        assert_eq!(band_window(1, 20), 2);
        assert_eq!(band_window(1, 40), 2); // ceil(0.05×40) = ceil(2.0) = 2
        assert_eq!(band_window(50, 100), 5); // max(50,100)=100 → 5% = 5
    }

    /// Identical sequences should produce near-zero DTW distance even with
    /// the Sakoe-Chiba band — the band only constrains off-diagonal warping,
    /// not on-diagonal matching.
    #[test]
    fn test_dtw_identical_sequences_with_band() {
        // 10 identical 2D unit vectors.  On-diagonal: i == j for all frames,
        // so the band (window=2 for max(10,10)*0.05=0.5→ceil→1→max(2)) is
        // not restrictive — on-diagonal always satisfies |i-j| ≤ 2.
        let vec = vec![0.7071, 0.7071]; // unit vector
        let seq = vec![vec.clone(); 10];
        let dist = dtw_distance(&seq, &seq);
        assert!(
            dist < 1e-6,
            "identical sequence should have near-zero DTW distance, got {dist}"
        );
    }

    /// The Sakoe-Chiba band prevents pathological alignments where a short
    /// live sequence warps far from the diagonal to find a matching template
    /// frame.  With window=2 (the new floor for max(10,10)*0.05=0.5→ceil→1),
    /// the band constrains |i-j| ≤ 2, so live[0] (j=0) cannot reach a match
    /// at template[3] (|0-3| = 3 > 2).  The resulting DTW path is forced
    /// through orthogonal frames, giving a positive distance.
    #[test]
    fn test_sakoe_chiba_prevents_pathological_alignment() {
        // Template: 5 frames where frames 0-2 = [1.0, 0.0] (orthogonal),
        // frames 3-4 = [0.0, 1.0] (matches live).
        //
        // Live: 3 frames [0.0, 1.0], [0.0, 1.0], [0.0, 1.0].
        //
        // Ratio = 5/3 ≈ 1.67 < 2 → DTW runs with window=2.
        // Without the band (|i-j| unlimited), live[0] could map to
        // template[3] (cost 0) for total distance 0.
        //
        // With the band (window=2): live[0] is restricted to j=0,1,2
        // (all orthogonal, cost 1.0).  The constrained optimal path is:
        //   (0,0)→(0,1)→(0,2)→(1,3)→(2,4)
        // with costs 1.0 + 1.0 + 1.0 + 0.0 + 0.0 = 3.0 over 5 steps
        // → normalised 0.6.
        let template: Vec<Vec<f32>> = {
            let mut t = Vec::new();
            t.push(vec![1.0, 0.0]); // orthogonal
            t.push(vec![1.0, 0.0]); // orthogonal
            t.push(vec![1.0, 0.0]); // orthogonal
            t.push(vec![0.0, 1.0]); // matches live
            t.push(vec![0.0, 1.0]); // matches live
            t
        };
        assert_eq!(template.len(), 5, "template must have 5 frames");

        let live = vec![vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let dist = dtw_distance(&live, &template);

        assert!(
            dist.is_finite(),
            "DTW should succeed with window=2 for n=3, m=5, got {dist}"
        );
        assert!(
            dist > 0.0,
            "band should force a positive distance by blocking the direct match"
        );
        // The expected normalised distance is 0.6 (three orthogonal steps at
        // 1.0 each distributed over 5 path steps).  We allow a small tolerance
        // for floating-point variation in the DTW path selection.
        assert!(
            (dist - 0.6).abs() < 0.01,
            "expected band-constrained distance ~0.60, got {dist}"
        );
    }

    /// When DTW cannot reach the final template column (e.g. extreme length
    /// asymmetry where |n-m| exceeds the band window), the endpoint guard
    /// falls back to mean-embedding cosine distance instead of returning
    /// f32::MAX.  This test verifies that n=1, m=20 (|1-20| = 19 > window=2)
    /// triggers the guard and produces a reasonable finite distance.
    #[test]
    fn test_dtw_endpoint_guard_fallback() {
        // Template: 20 frames, even mix of [1,0] and [0,1].
        // Live: 1 frame [0, 1].
        // Window = max(2, ceil(0.05*20)) = max(2, 1) = 2.
        // |n-m| = 19 > 2 → template[19] unreachable → guard triggers.
        let template: Vec<Vec<f32>> = {
            let mut t = Vec::new();
            for _ in 0..10 {
                t.push(vec![1.0, 0.0]); // orthogonal
            }
            for _ in 10..20 {
                t.push(vec![0.0, 1.0]); // matches live
            }
            t
        };
        assert_eq!(template.len(), 20, "template must have 20 frames");

        let live = vec![vec![0.0, 1.0]];
        let dist = dtw_distance(&live, &template);

        // Guard fallback: mean of template is [0.5, 0.5], live is [0, 1].
        // Cosine distance = 1 - 0.5/sqrt(0.5) ≈ 0.293.
        assert!(
            dist.is_finite(),
            "endpoint guard should produce finite distance, got {dist}"
        );
        assert!(
            (dist - 0.293).abs() < 0.01,
            "expected mean-cosine distance ~0.293, got {dist}"
        );
    }

    /// Asymmetric length test (mahbot-774): verify that DTW with the improved
    /// band (floor=2) can still match 2-vs-4 and 4-vs-2 embedding sequences
    /// without returning f32::MAX.  This is the exact bug scenario: multi-
    /// embedding templates from slow enrollment vs short live sequences.
    #[test]
    fn test_sakoe_chiba_band_asymmetric_short_vs_long() {
        // A simple 2D unit vector sequence where all frames are the same.
        // This ensures any DTW failure is purely from the band constraint,
        // not from embedding mismatch.
        let unit = vec![0.7071, 0.7071];

        // Case 1: live=2, template=4 (fast speech vs slow enrollment).
        let live_short = vec![unit.clone(), unit.clone()];
        let template_long = vec![unit.clone(), unit.clone(), unit.clone(), unit.clone()];
        let dist_short_live = dtw_distance(&live_short, &template_long);
        assert!(
            dist_short_live.is_finite(),
            "2-frame live vs 4-frame template should produce finite distance, got {dist_short_live}"
        );
        assert!(
            dist_short_live < 1e-6,
            "identical embedding sequences should have near-zero distance, got {dist_short_live}"
        );

        // Case 2: live=4, template=2 (slow speech vs fast enrollment).
        let live_long = vec![unit.clone(), unit.clone(), unit.clone(), unit.clone()];
        let template_short = vec![unit.clone(), unit.clone()];
        let dist_long_live = dtw_distance(&live_long, &template_short);
        assert!(
            dist_long_live.is_finite(),
            "4-frame live vs 2-frame template should produce finite distance, got {dist_long_live}"
        );
        assert!(
            dist_long_live < 1e-6,
            "identical embedding sequences should have near-zero distance, got {dist_long_live}"
        );

        // Case 3: different spectral content — verify that 2-vs-4 still
        // gives a meaningful non-zero distance (not f32::MAX).
        let x_vec = vec![1.0, 0.0];
        let y_vec = vec![0.0, 1.0];
        let live_xy = vec![x_vec.clone(), x_vec.clone()]; // [1,0], [1,0]
        let template_yx = vec![y_vec.clone(), y_vec.clone(), y_vec.clone(), y_vec.clone()]; // [0,1], [0,1], [0,1], [0,1]

        let dist_diff = dtw_distance(&live_xy, &template_yx);
        assert!(
            dist_diff.is_finite(),
            "2-frame [1,0] vs 4-frame [0,1] should produce finite distance, got {dist_diff}"
        );
        assert!(
            dist_diff > 0.0,
            "orthogonal embeddings should give positive distance"
        );
    }

    /// Endpoint guard fallback for extreme asymmetry (mahbot-774): when DTW
    /// cannot reach the final template frame because |n-m| > window, verify
    /// the guard returns mean-embedding cosine distance instead of f32::MAX.
    #[test]
    fn test_dtw_endpoint_guard_extreme_asymmetry() {
        // Ratio = 10/1 = 10 > 2 → triggers fallback.
        let unit = vec![0.7071, 0.7071];
        let short = vec![unit.clone()];
        let long = vec![unit.clone(); 10];

        // All-identical embeddings should still produce near-zero distance
        // via the mean-embedding fallback.
        let dist = dtw_distance(&short, &long);
        assert!(
            dist.is_finite(),
            "1-frame vs 10-frame should produce finite fallback distance, got {dist}"
        );
        assert!(
            dist < 1e-6,
            "identical mean embeddings should give near-zero distance, got {dist}"
        );

        // Ratio = 19/2 = 9.5 > 2 → triggers fallback with different content.
        let x_vec = vec![1.0, 0.0];
        let y_vec = vec![0.0, 1.0];
        let short_xy = vec![x_vec.clone(), x_vec.clone()]; // [1,0], [1,0]
        let long_y = vec![y_vec.clone(); 19]; // [0,1] × 19

        let dist_ortho = dtw_distance(&short_xy, &long_y);
        assert!(
            dist_ortho.is_finite(),
            "2-frame [1,0] vs 19-frame [0,1] should produce finite fallback distance, got {dist_ortho}"
        );

        // Mean of short: [1.0, 0.0]; mean of long: [0.0, 1.0].
        // Cosine distance = 1 - 0 / (1×1) = 1.0.
        assert!(
            (dist_ortho - 1.0).abs() < 0.01,
            "orthogonal unit vectors should give cosine distance of 1.0, got {dist_ortho}"
        );
    }

    /// Verify that the Sakoe-Chiba band scales correctly with the longer
    /// sequence (mahbot-774).  The window formula uses `max(n, m)`, so
    /// dtw_distance(live, template) and dtw_distance(template, live) should
    /// produce the same window size and thus comparable distances for
    /// symmetric-length inputs.
    #[test]
    fn test_sakoe_chiba_band_scales_with_longer_sequence() {
        // Use a non-trivial sequence where the order matters for DTW.
        let frame_a = vec![1.0, 0.0];
        let frame_b = vec![0.5, 0.5];
        let frame_c = vec![0.0, 1.0];
        let frame_d = vec![1.0, 1.0];

        let seq_short = vec![frame_a.clone(), frame_b.clone()];
        let seq_long = vec![
            frame_a.clone(),
            frame_b.clone(),
            frame_c.clone(),
            frame_d.clone(),
        ];

        // Both directions use the same window because n.max(m) is always 4
        // (max(2,4) = max(4,2) = 4).  The window after floor(max(2.0)) is 2:
        //   ceil(0.05 × 4).max(2.0) = ceil(0.2).max(2.0) = 1.max(2.0) = 2.
        // The distances may differ (DTW is asymmetric), but both should be
        // finite and within comparable range — neither direction should
        // produce f32::MAX from an unreachable endpoint.
        let d1 = dtw_distance(&seq_short, &seq_long);
        let d2 = dtw_distance(&seq_long, &seq_short);

        assert!(
            d1.is_finite(),
            "2→4 direction should produce finite distance, got {d1}"
        );
        assert!(
            d2.is_finite(),
            "4→2 direction should produce finite distance, got {d2}"
        );
        assert!(
            d1 > 0.0,
            "2→4 direction should give positive distance for different content"
        );
        assert!(
            d2 > 0.0,
            "4→2 direction should give positive distance for different content"
        );

        // Both distances should be of comparable magnitude (not one being
        // f32::MAX while the other is ~0).  A factor of 2 is reasonable
        // for DTW on asymmetric-length inputs with the same window.
        let (lo, hi) = if d1 < d2 { (d1, d2) } else { (d2, d1) };
        assert!(
            hi / lo < 2.0,
            "distances for 2→4 ({d1}) and 4→2 ({d2}) should be within a factor of 2"
        );
    }

    // ── Minimum utterance length check (mahbot-772) ──────────────────

    #[test]
    fn test_minimum_utterance_rejection() {
        // Utterances shorter than 400ms are rejected as noise blips/coughs.
        let short_err = check_enrollment_utterance_length(1, 300);
        assert!(
            short_err.is_err(),
            "300ms utterance should be rejected as too short"
        );
        let msg = short_err.unwrap_err();
        assert!(
            msg.contains("too short"),
            "error message should indicate 'too short': {msg}"
        );
        assert!(
            msg.contains("300ms"),
            "error message should include the duration: {msg}"
        );

        // Single-embedding utterance (600ms) should be accepted — the Google
        // speech_embedding/1 model produces exactly 1 embedding from any
        // 76-frame window; that's its full contract, not "incomplete".
        assert!(
            check_enrollment_utterance_length(1, 600).is_ok(),
            "1 embedding @ 600ms should be accepted"
        );

        // 0 embeddings with long duration: still rejected because there's
        // no data to match against (edge case guard independent of the
        // 400ms minimum).
        let zero_err = check_enrollment_utterance_length(0, 800);
        assert!(
            zero_err.is_err(),
            "0 embeddings should still be rejected as no template data available"
        );

        // Sufficiently long utterances pass.
        assert!(
            check_enrollment_utterance_length(2, 1300).is_ok(),
            "2 embeddings @ 1300ms should be accepted"
        );
        assert!(
            check_enrollment_utterance_length(5, 3000).is_ok(),
            "5 embeddings @ 3000ms should be accepted"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // Regression tests for wake word detection pipeline infrastructure
    // (cooldown, VAD gating, ring buffer, mel padding)
    // Ticket: mahbot-779
    // ═════════════════════════════════════════════════════════════════════

    // ── pad_mel_frames_to_window ───────────────────────────────────────

    #[test]
    fn test_pad_mel_frames_to_window() {
        let silence_val = super::spec_transform(0.0);
        // Tolerance: 1e-5 is well above accumulated FP round-off from
        // different computation paths between the implementation
        // (X * inv_count where inv_count = 1/(N+1)) and the test
        // (X / (N+1) with naive lerp).  Across 46 fade frames even the
        // worst-case accumulation stays below 1e-6; 1e-5 gives headroom
        // for platform differences in FMA contraction.
        let eps = 1e-5;

        // 1. Empty sequence → get EMBEDDING_WINDOW_FRAMES silence frames (fallback)
        let empty: Vec<Vec<f32>> = Vec::new();
        let padded = super::pad_mel_frames_to_window(&empty);
        assert_eq!(
            padded.len(),
            EMBEDDING_WINDOW_FRAMES,
            "empty sequence should be padded to {EMBEDDING_WINDOW_FRAMES} frames",
        );
        // All frames are constant silence (fallback for empty buffer).
        for (i, frame) in padded.iter().enumerate() {
            assert_eq!(
                frame.len(),
                NUM_MEL_BANDS,
                "each frame should have {NUM_MEL_BANDS} mel bands, got {} at frame {i}",
                frame.len(),
            );
            for (b, &val) in frame.iter().enumerate() {
                assert!(
                    (val - silence_val).abs() < eps,
                    "frame {i}, band {b}: expected {silence_val}, got {val}",
                );
            }
        }

        // 2. Partial sequence (30 frames) → tapered fade-out at end
        let mut partial = Vec::with_capacity(30);
        for i in 0..30 {
            partial.push(vec![i as f32; NUM_MEL_BANDS]);
        }
        let padded = super::pad_mel_frames_to_window(&partial);
        assert_eq!(
            padded.len(),
            EMBEDDING_WINDOW_FRAMES,
            "partial sequence should be padded to {EMBEDDING_WINDOW_FRAMES} frames",
        );
        // First 30 frames are unchanged
        for (i, frame) in padded[..30].iter().enumerate() {
            assert_eq!(frame.len(), NUM_MEL_BANDS);
            assert!(
                (frame[0] - i as f32).abs() < f32::EPSILON,
                "frame {i} should be unchanged, got frame[0]={}",
                frame[0],
            );
        }
        // Remaining frames are tapered fade-out from last value (29.0) toward silence
        let last_val = 29.0f32;
        let frames_needed = EMBEDDING_WINDOW_FRAMES - 30;
        for (j, frame) in padded[30..].iter().enumerate() {
            assert_eq!(frame.len(), NUM_MEL_BANDS);
            let alpha = (j + 1) as f32 / (frames_needed + 1) as f32;
            let expected_val = last_val * (1.0 - alpha) + silence_val * alpha;
            for &val in frame.iter() {
                assert!(
                    (val - expected_val).abs() < eps,
                    "fade frame {j}: expected {expected_val}, got {val} (alpha={alpha})",
                );
            }
        }

        // 3. Already at EMBEDDING_WINDOW_FRAMES → unchanged
        let mut exact = Vec::with_capacity(EMBEDDING_WINDOW_FRAMES);
        for i in 0..EMBEDDING_WINDOW_FRAMES {
            exact.push(vec![(i * 2) as f32; NUM_MEL_BANDS]);
        }
        let padded = super::pad_mel_frames_to_window(&exact);
        assert_eq!(
            padded.len(),
            EMBEDDING_WINDOW_FRAMES,
            "exact-length sequence should stay at {EMBEDDING_WINDOW_FRAMES}",
        );
        for (i, frame) in padded.iter().enumerate() {
            assert!(
                (frame[0] - (i * 2) as f32).abs() < f32::EPSILON,
                "frame {i} should be unchanged, got frame[0]={}",
                frame[0],
            );
        }

        // 4. Over EMBEDDING_WINDOW_FRAMES → unchanged (guard clause)
        let mut over = Vec::with_capacity(EMBEDDING_WINDOW_FRAMES + 5);
        for i in 0..EMBEDDING_WINDOW_FRAMES + 5 {
            over.push(vec![i as f32; NUM_MEL_BANDS]);
        }
        let padded = super::pad_mel_frames_to_window(&over);
        assert_eq!(
            padded.len(),
            EMBEDDING_WINDOW_FRAMES + 5,
            "over-length sequence should be unchanged",
        );
    }

    // ── handle_wake_word_detection cooldown ──────────────────────────

    #[test]
    #[serial_test::serial(voice)]
    fn test_handle_wake_word_detection_cooldown() {
        let mut ctx = PipelineCtx::new();

        // ── Cooldown ACTIVE: all buffers cleared, no processing ──
        ctx.last_wake_word_detection = Some(Instant::now());
        let last_detection_saved = ctx.last_wake_word_detection;

        // Pre-populate buffers to verify they get cleared by the cooldown path
        ctx.audio_buffer = vec![1.0f32; 512];
        ctx.voice_batch = vec![2.0f32; 256];
        ctx.mel_frame_buffer = vec![vec![3.0f32; NUM_MEL_BANDS]];
        ctx.embedding_ring = vec![vec![4.0f32; EMBEDDING_DIM]];
        ctx.score_window = vec![0.5f32; 3];

        // Feed audio — cooldown should discard it and clear everything
        let samples = vec![5.0f32; FRAME_LENGTH];
        handle_wake_word_detection(&samples, &mut ctx);

        assert!(
            ctx.audio_buffer.is_empty(),
            "audio_buffer should be cleared during cooldown",
        );
        assert!(
            ctx.voice_batch.is_empty(),
            "voice_batch should be cleared during cooldown",
        );
        assert!(
            ctx.mel_frame_buffer.is_empty(),
            "mel_frame_buffer should be cleared during cooldown",
        );
        assert!(
            ctx.embedding_ring.is_empty(),
            "embedding_ring should be cleared during cooldown",
        );
        assert!(
            ctx.score_window.is_empty(),
            "score_window should be cleared during cooldown",
        );
        assert_eq!(
            ctx.last_wake_word_detection, last_detection_saved,
            "last_wake_word_detection should remain unchanged after cooldown early-return",
        );

        // ── Cooldown EXPIRED: processing proceeds normally ──
        ctx.last_wake_word_detection =
            Some(Instant::now() - WAKE_WORD_COOLDOWN - Duration::from_secs(1));
        reset_vad();

        let speech = speech_frame();
        handle_wake_word_detection(&speech, &mut ctx);

        // With cooldown expired and speech frame: voice_batch should
        // accumulate audio (one frame adds HOP_LENGTH = 256 samples).
        assert_eq!(
            ctx.voice_batch.len(),
            HOP_LENGTH,
            "voice_batch should accumulate {} samples after cooldown expired",
            HOP_LENGTH,
        );
    }

    // ── handle_wake_word_detection VAD gating ───────────────────────

    #[test]
    #[serial_test::serial(voice)]
    fn test_handle_wake_word_detection_vad_gating() {
        let mut ctx = PipelineCtx::new();

        // 1. Silence → voice_batch stays empty, no mel flush.
        //    Inject audio_buffer directly with exactly FRAME_LENGTH silence
        //    and call with empty samples to avoid accumulation overlap.
        assert!(
            ctx.last_wake_word_detection.is_none(),
            "PipelineCtx::new() should initialize last_wake_word_detection to None",
        );
        reset_vad();
        ctx.audio_buffer = vec![0.0f32; FRAME_LENGTH + HOP_LENGTH];
        handle_wake_word_detection(&[], &mut ctx);

        assert!(
            ctx.voice_batch.is_empty(),
            "voice_batch should stay empty after silence",
        );
        assert!(
            ctx.mel_frame_buffer.is_empty(),
            "mel_frame_buffer should stay empty after silence",
        );

        // 2. Speech → voice_batch accumulates audio (HOP_LENGTH per frame).
        //    Feed 3 speech frames (5 processing iterations with 50% overlap).
        reset_vad();
        let mut speech_audio = speech_frame(); // frame 1
        speech_audio.extend_from_slice(&speech_frame()); // frame 2
        speech_audio.extend_from_slice(&speech_frame()); // frame 3
        // 3 × FRAME_LENGTH with overlap: the buffer processes
        // floor((total - FRAME_LENGTH) / HOP_LENGTH) + 1 = floor((1536-512)/256)+1 = 5 frames
        ctx.audio_buffer = speech_audio;
        handle_wake_word_detection(&[], &mut ctx);
        // Each speech frame contributes HOP_LENGTH samples to voice_batch
        // (frame[..HOP_LENGTH]), so 5 iterations → 5 × HOP_LENGTH = 1280.
        assert_eq!(
            ctx.voice_batch.len(),
            5 * HOP_LENGTH,
            "voice_batch should accumulate exactly {} samples after 5 speech frames",
            5 * HOP_LENGTH,
        );

        // 3. Speech → silence transition → flush_voice_batch called,
        //    voice_batch cleared.  Explicitly populate voice_batch to make
        //    this step self-contained (not reliant on Part 2's side effects).
        ctx.voice_batch = vec![1.0f32; HOP_LENGTH];
        reset_vad();
        ctx.audio_buffer = vec![0.0f32; FRAME_LENGTH + HOP_LENGTH];
        handle_wake_word_detection(&[], &mut ctx);

        // The first frame processed in this call (silence) hits the else-if
        // branch: voice_batch is non-empty → flush + clear.
        assert!(
            ctx.voice_batch.is_empty(),
            "voice_batch should be cleared after silence transition",
        );
        assert!(
            ctx.mel_frame_buffer.is_empty(),
            "mel_frame_buffer should be empty (no ONNX models loaded)",
        );
    }

    // ── Cooldown constant sanity (mahbot-770 Fix 2) ───────────────────
    //
    // WAKE_WORD_COOLDOWN is a tunable production constant; its exact value is
    // verified at the integration-test level (test_voice_pipeline_commands_*)
    // via the pipeline end-to-end.  A unit test that asserts a literal constant
    // would only replicate the declaration with no behavioral coverage.

    // ═════════════════════════════════════════════════════════════════════
    // Enrollment quality scoring and self-test (mahbot-778)
    // ═════════════════════════════════════════════════════════════════════

    /// Helper: create a single embedding vector from a 2D coordinate.
    fn emb_2d(x: f32, y: f32) -> Vec<f32> {
        vec![x, y]
    }

    /// Helper: create an enrollment utterance (sequence of 2D embeddings).
    fn utterance_2d(points: &[[f32; 2]]) -> Vec<Vec<f32>> {
        points.iter().map(|&[x, y]| emb_2d(x, y)).collect()
    }

    /// A consistent utterance for testing: all embeddings point along (0.7, 0.7).
    fn consistent_utterance() -> Vec<Vec<f32>> {
        utterance_2d(&[[0.7, 0.7], [0.7, 0.7], [0.7, 0.7]])
    }

    /// An outlier utterance for testing: embeddings point in opposite direction.
    fn outlier_utterance() -> Vec<Vec<f32>> {
        utterance_2d(&[[-0.7, -0.7], [-0.7, -0.7], [-0.7, -0.7]])
    }

    #[test]
    fn test_utterance_quality_score_clipping() {
        // ── Sample with clipping ──────────────────────────────────
        let mut samples_clip = vec![0.5f32; 16000]; // ~1 second of audio at 16kHz
        samples_clip[8000] = 1.0; // Hit i16::MAX equivalent
        let embeddings = vec![emb_2d(0.7, 0.7); 5];
        let quality = compute_utterance_quality(&samples_clip, &embeddings, &[], None);
        assert!(
            quality.clipping_detected,
            "clipping should be detected when a sample hits 1.0"
        );
        assert!(
            quality.score < 1.0,
            "clipping should reduce quality score, got {}",
            quality.score,
        );

        // ── Sample without clipping ───────────────────────────────
        let samples_clean = vec![0.5f32; 16000]; // No sample near 1.0
        let quality_clean = compute_utterance_quality(&samples_clean, &embeddings, &[], None);
        assert!(
            !quality_clean.clipping_detected,
            "clean sample should not have clipping detected"
        );
        assert!(
            quality_clean.score > quality.score,
            "clean sample should have higher quality score than clipped sample ({} vs {})",
            quality_clean.score,
            quality.score,
        );
    }

    #[test]
    fn test_utterance_quality_score_consistency() {
        // 5 nearly-identical utterances (all pointing along (0.7, 0.7))
        let consistent = vec![
            consistent_utterance(),
            consistent_utterance(),
            consistent_utterance(),
            consistent_utterance(),
            consistent_utterance(),
        ];

        // 1 outlier (pointing along (-0.7, -0.7) — very different)
        let outlier = outlier_utterance();

        // Samples placeholder (any non-empty value — quality uses samples for
        // duration/clipping/SNR, not for consistency).
        let samples = vec![0.5f32; 16000];

        // Compute quality for a consistent utterance (compared against the 4
        // OTHER consistent utterances — the 5th consistent utterance is the
        // one being scored, excluded from enrollment_buffer).
        let quality_consistent =
            compute_utterance_quality(&samples, &consistent[0], &consistent[1..], None);

        // Compute quality for the outlier (compared against the 5 consistent ones).
        let quality_outlier = compute_utterance_quality(&samples, &outlier, &consistent, None);

        // The outlier should have a HIGHER average DTW distance (lower consistency)
        // and thus a LOWER quality score.
        assert!(
            quality_outlier.avg_dtw_distance > quality_consistent.avg_dtw_distance,
            "outlier DTW distance ({}) should be higher than consistent DTW distance ({})",
            quality_outlier.avg_dtw_distance,
            quality_consistent.avg_dtw_distance,
        );
        assert!(
            quality_outlier.score < quality_consistent.score,
            "outlier quality score ({}) should be lower than consistent score ({})",
            quality_outlier.score,
            quality_consistent.score,
        );
    }

    #[test]
    fn test_utterance_quality_snr_with_noise_rms() {
        // ── Known-clean SNR computation ────────────────────────────
        // Signal: 0.5 amplitude sine → RMS ≈ 0.354
        // Noise:  0.05 RMS
        // Expected SNR = 20 * log10(0.354 / 0.05) ≈ 17.0 dB
        let sample_count = SAMPLE_RATE as usize; // 1 second at 16 kHz
        let signal: Vec<f32> = (0..sample_count)
            .map(|i| {
                0.5f32 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin()
            })
            .collect();
        // Reference: RMS of 0.5-amplitude sine = 0.5 / sqrt(2) ≈ 0.3536
        let speech_rms = (signal.iter().map(|&s| s * s).sum::<f32>() / signal.len() as f32).sqrt();
        assert!(
            (speech_rms - 0.3536).abs() < 0.001,
            "speech RMS should be ~0.3536, got {}",
            speech_rms,
        );

        let noise_rms = 0.05f32;
        let expected_snr_db = 20.0 * (speech_rms / noise_rms).log10();

        let embeddings = vec![emb_2d(0.7, 0.7); 5];
        let quality = compute_utterance_quality(&signal, &embeddings, &[], Some(noise_rms));

        assert!(
            quality.snr_db.is_finite(),
            "SNR should be finite when noise_rms < speech_rms, got {}",
            quality.snr_db,
        );
        assert!(
            (quality.snr_db - expected_snr_db).abs() < 0.1,
            "SNR should be ~{expected_snr_db:.1} dB, got {} dB",
            quality.snr_db,
        );

        // ── Edge case: noise_rms >= speech_rms → 0.0 dB ────────────
        let quality_loud_noise =
            compute_utterance_quality(&signal, &embeddings, &[], Some(speech_rms * 2.0));
        assert_eq!(
            quality_loud_noise.snr_db, 0.0,
            "SNR should be 0.0 when noise_rms >= speech_rms, got {}",
            quality_loud_noise.snr_db,
        );

        // ── Edge case: near-zero noise_rms → finite high SNR ───────
        let quality_no_noise = compute_utterance_quality(&signal, &embeddings, &[], Some(1e-9));
        assert!(
            quality_no_noise.snr_db.is_finite() && quality_no_noise.snr_db > 0.0,
            "SNR should be finite and positive with near-zero noise, got {}",
            quality_no_noise.snr_db,
        );
    }

    #[test]
    fn test_enrollment_self_test_passes() {
        // 10 identical utterances (all consistent)
        let utterance = utterance_2d(&[[0.7, 0.7], [0.7, 0.7], [0.7, 0.7]]);
        let enrollment_buffer: Vec<Vec<Vec<f32>>> = vec![utterance.clone(); 10];

        // Build a template from the same data (as calibrate_threshold would)
        let template = WakeWordTemplate {
            name: "test".to_string(),
            embeddings: utterance.clone(),
            threshold: 0.10, // MAD_THRESHOLD_FLOOR
            enrollment_samples: 10,
        };
        let templates = vec![template; 3]; // K=3
        let ww_templates = WakeWordTemplates {
            templates,
            minimum_matches: 1, // self-test hard-codes M=1; field is unused but keep accurate
            ..Default::default()
        };

        let result = run_enrollment_self_test(&enrollment_buffer, &ww_templates);
        assert!(
            result.is_ok(),
            "self-test with consistent utterances should pass, got: {:?}",
            result,
        );
    }

    #[test]
    fn test_enrollment_self_test_fails() {
        // Build a template from a "good" utterance
        let good_utterance = utterance_2d(&[[0.7, 0.7], [0.7, 0.7], [0.7, 0.7]]);

        // Enrollment buffer: 10 silent-like utterances (all-zero embeddings)
        // that do NOT match the template.  Each "utterance" has a single
        // zero embedding to ensure dtw_distance can run (non-empty).
        let zero_emb = emb_2d(0.0, 0.0);
        let silence_utterance = vec![zero_emb.clone(), zero_emb.clone(), zero_emb.clone()];
        let enrollment_buffer: Vec<Vec<Vec<f32>>> = vec![silence_utterance; 10];

        let template = WakeWordTemplate {
            name: "test".to_string(),
            embeddings: good_utterance,
            threshold: 0.10,
            enrollment_samples: 10,
        };
        let templates = vec![template]; // K=1
        let ww_templates = WakeWordTemplates {
            templates,
            minimum_matches: 1,
            ..Default::default()
        };

        let result = run_enrollment_self_test(&enrollment_buffer, &ww_templates);
        assert!(
            result.is_err(),
            "self-test with silent utterances against active template should fail",
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("Self-test failed"),
            "error message should indicate self-test failure: {msg}",
        );
    }

    #[test]
    #[serial_test::serial(voice)]
    fn test_finalize_enrollment_naming() {
        // Set up voice state with 10 identical enrollment samples so that
        // calibrate_threshold produces K=3 templates and self-test passes.
        let pipeline = VOICE_PIPELINE.get_or_init(|| {
            RwLock::new(VoicePipelineState {
                enabled: false,
                status: VoiceStatus::Disabled,
                templates: Arc::new(WakeWordTemplates::default()),
                enrollment_buffer: Vec::new(),
                negative_audio_chunks: Vec::new(),

                cmd_tx: None,
            })
        });
        // Always reset to clean state, even if a previous serial test already
        // initialised the pipeline (handles test-ordering fragility).
        {
            let mut state = pipeline.write().unwrap_poison();
            state.enabled = false;
            state.status = VoiceStatus::Disabled;
            state.templates = Arc::new(WakeWordTemplates::default());
            state.enrollment_buffer.clear();
            state.cmd_tx = None;
        }

        // Push 10 identical utterances into the enrollment buffer.
        let utterance = utterance_2d(&[[0.7, 0.7], [0.7, 0.7], [0.7, 0.7], [0.7, 0.7]]);
        {
            let mut state = voice_state().write().unwrap_poison();
            for _ in 0..10 {
                state.enrollment_buffer.push(utterance.clone());
            }
        }

        let result = finalize_enrollment("custom");
        assert!(
            result.is_ok(),
            "finalize_enrollment with consistent samples should succeed, got: {:?}",
            result,
        );
        let (templates, minimum_matches) = result.unwrap();

        // K=3 → 3 templates with expected names
        assert_eq!(templates.len(), 3, "should produce 3 templates for K=3",);
        assert_eq!(
            templates[0].name, "custom",
            "first template should be named 'custom'",
        );
        assert_eq!(
            templates[1].name, "custom_2",
            "second template should be named 'custom_2'",
        );
        assert_eq!(
            templates[2].name, "custom_3",
            "third template should be named 'custom_3'",
        );

        // K≥2 → minimum_matches = 2
        assert_eq!(minimum_matches, 2, "minimum_matches should be 2 for K=3",);
    }
}
