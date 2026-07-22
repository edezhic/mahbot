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
//! 5. **Wake word matching** — Conv1D/MLP classifier on a 3-embedding sliding window,
//!    followed by a logistic regression verifier (AND gate).
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
use crate::wake_word_classifier::{self, ClassifierWeights, TrainingConfig, WakeWordClassifier};
use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

// ── E2E integration test (voice-tests feature) ──────────────────────────
#[cfg(all(test, feature = "voice-tests"))]
#[path = "voice_pipeline_e2e_test.rs"]
pub(crate) mod voice_pipeline_e2e_test;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Target sample rate: 16 kHz mono.
pub const SAMPLE_RATE: u32 = 16_000;

/// Frame size for mel spectrogram (512 samples = 32ms at 16kHz).
pub(crate) const FRAME_LENGTH: usize = 512;

/// Hop length between frames (256 samples at 16 kHz).  This constant controls
/// VAD frame iteration stride and silence tracking in the application code.
/// The ONNX mel spectrogram model uses its own internal stride (160 samples =
/// 10ms) — HOP_LENGTH does NOT affect mel frame spacing (mahbot-772).
pub(crate) const HOP_LENGTH: usize = 256;

/// Number of mel bands in the spectrogram.
const NUM_MEL_BANDS: usize = 32;

/// Internal hop length of the ONNX mel spectrogram model (160 samples = 10ms
/// at 16 kHz).  This is the stride between consecutive mel frames computed
/// by the `melspectrogram.onnx` model, independent of the application-level
/// [`HOP_LENGTH`] which controls VAD frame iteration.
///
/// This constant is used to align voice batch overlap boundaries with the
/// model's internal stride so that mel frames across consecutive batches
/// have consistent temporal positions (mahbot-799).  See [`flush_voice_batch`]
/// for details.
const MEL_STRIDE: usize = 160;

/// Overlap samples retained in `voice_batch` after a mel spectrogram flush.
///
/// Set to 2 × [`MEL_STRIDE`] (320 samples = 20ms at 16kHz) so that the
/// retained overlap is a multiple of the mel model's internal stride
/// (160 samples), ensuring mel frame positions are aligned across
/// consecutive batch boundaries.
///
/// Using 2× provides two full-context crossing frames at the batch boundary.
/// See [`flush_voice_batch`] for the detailed rationale (mahbot-799).
const VOICE_BATCH_OVERLAP: usize = MEL_STRIDE * 2;

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
pub(crate) const SILENCE_THRESHOLD_SAMPLES: usize =
    (SILENCE_DURATION.as_millis() as usize * SAMPLE_RATE as usize) / 1000;

/// Silence threshold (200ms) before showing "Keep silent to confirm…" UI hint.
/// Intentionally wider than a single frame (16ms) so the UI reliably transitions
/// even under scheduling jitter.
const SILENCE_UI_GATE_SAMPLES: usize = 200 * SAMPLE_RATE as usize / 1000;

/// Capacity of the microphone audio channel feeding the voice pipeline.
///
/// 32 chunks × 512 samples / 16000 Hz ≈ 1 second of audio.  When the pipeline
/// is blocked (e.g. ONNX inference in [`handle_wake_word_detection`]),
/// [`try_send`](tokio::sync::mpsc::Sender::try_send) silently drops chunks
/// at this threshold, preventing unbounded memory growth.
///
/// # Drop policy
///
/// This is **drop-newest**: the most recent audio chunk is discarded when the
/// channel is full.  During a pipeline stall the buffered audio is slightly
/// delayed (~1 s) but temporally contiguous, so downstream processing (VAD,
/// mel extraction, wake-word classifier) operates on consistent stream
/// segments.  The wake word may be missed if it arrives entirely within the
/// dropped window, but the user will simply repeat it.
///
/// # VAD state
///
/// The [`earshot::Detector`] maintains an internal ring buffer and pre-emphasis
/// filter that stay synchronised with the audio stream *as processed* by the
/// pipeline.  Dropped chunks create a temporal gap at the stream level, but
/// the detector processes whatever it receives next — spurious VAD frames are
/// short-lived (1–2 frames) and the detector self-corrects on subsequent audio.
///
/// # Future work
///
/// The underlying latency cause is that ONNX inference (mel spectrogram,
/// embedding) runs on the async runtime via
/// [`tokio::task::block_in_place`](https://docs.rs/tokio/latest/tokio/task/fn.block_in_place.html)
/// inside [`handle_wake_word_detection`].  Moving these to
/// [`tokio::task::spawn_blocking`](https://docs.rs/tokio/latest/tokio/task/fn.spawn_blocking.html)
/// (as enrollment already does) would reduce pipeline stalls and minimise
/// the frequency of dropped chunks.  The bounded channel is a tractable
/// first step that caps memory growth without restructuring the hot path.
///
/// # See also
///
/// * [`start_microphone`] — channel creation
/// * ticket mahbot-804 — unbounded queue growth root cause
const MIC_CHANNEL_CAPACITY: usize = 32;

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

/// Maximum audio samples to accumulate in [`PipelineCtx::audio_buffer`] during
/// wake word cooldown ([`WAKE_WORD_COOLDOWN`]).  Set to 2 frames (1024 samples
/// = ~64ms) to prevent unbounded growth during a prolonged cooldown while
/// providing enough context for a smooth pipeline restart.
///
/// ## Frame-processing arithmetic
///
/// When the cooldown expires, the next mic chunk (512 samples) is appended to
/// the accumulated buffer for a total of 1024 + 512 = 1536 samples.  The frame
/// loop in [`handle_wake_word_detection`] processes
/// `floor((1536 - FRAME_LENGTH) / HOP_LENGTH) + 1 = 5` iterations per call.
/// Each iteration contributes `HOP_LENGTH` (256) samples to `voice_batch` if
/// VAD-positive, filling ~62% of [`VOICE_BATCH_SIZE`] (1280/2048) in one shot.
///
/// With 1 frame (512) accumulated: 3 iterations, ~38% of batch threshold.
/// With no accumulation: 1 iteration, ~13% — the pipeline starves.
/// Higher caps (3+ frames) offer only 2 more iterations per frame at the cost
/// of ~64ms more cooldown audio kept (diminishing returns).
const COOLDOWN_ACCUMULATION_CAP: usize = FRAME_LENGTH * 2;

/// Maximum number of recent embeddings to keep in the ring buffer.
/// With stride=8 (~89.5% overlap), each new embedding covers ~1.2s of audio
/// and arrives every ~128ms, keeping ~19 embeddings = ~2.4 seconds of context.
const EMBEDDING_RING_MAX: usize = 19;

/// Number of enrollment samples required (mahbot-765).
const NUM_ENROLLMENT_SAMPLES: usize = 10;

/// Minimum length (in audio samples at 16kHz) for a collected ambient audio
/// chunk to be used as a negative verifier training example (mahbot-797).
///
/// Set to 0.5s of audio, which produces ~31 mel frames (padded to 76 for the
/// embedding model).  Chunks shorter than this are discarded — they would be
/// mostly padding/silence and provide negligible discriminative signal.
const MIN_NEGATIVE_AUDIO_LEN: usize = SAMPLE_RATE as usize / 2;

/// Maximum number of ambient noise chunks to retain for verifier training.
/// If training repeatedly fails (ONNX not loaded, <2 chunks, or empty
/// embeddings), this cap prevents unbounded memory growth in the voice
/// pipeline state (mahbot-800).
const MAX_NEGATIVE_AUDIO_CHUNKS: usize = 100;

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

/// Maximum size of the raw audio ring buffer (~200ms at 16kHz = 3200
/// samples).  Used during enrollment to capture ~100ms of pre-VAD-trigger
/// and post-speech context so the template includes the onset/offset
/// phonemes that strict enrollment VAD (0.85) excludes.
pub(crate) const RAW_RING_MAX: usize = SAMPLE_RATE as usize / 5;

/// Context padding duration in milliseconds for VAD asymmetry mitigation
/// (mahbot-775 Fix 3).  Used to prepend ~100ms of pre-VAD-trigger context
/// and append ~100ms of post-speech context to enrollment utterances, so
/// the template includes the onset/offset phonemes that strict enrollment
/// VAD (0.85) excludes but live detection (VAD=0.5) includes.
const CONTEXT_PADDING_MS: usize = 100;

/// Context padding in audio samples at 16 kHz, derived from
/// CONTEXT_PADDING_MS to stay correct if the sample rate is adjusted.
pub(crate) const CONTEXT_PADDING_SAMPLES: usize =
    (CONTEXT_PADDING_MS * SAMPLE_RATE as usize) / 1000;

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
     for the Conv1D MLP classifier"
);

/// Factor applied to `ROLLING_WINDOW_N` to compute the detection threshold
/// (mahbot-773).  At 0.65, the average per-frame soft score must exceed ~65%
/// for detection to fire.
const MATCH_THRESHOLD_FACTOR: f32 = 0.65;

/// Detection threshold for the rolling sum of soft scores (mahbot-773).
/// Computed as: `ROLLING_WINDOW_N × MATCH_THRESHOLD_FACTOR` (which simplifies
/// to `1 × 3 × 0.65 = 1.95` since the MLP classifier produces a single score
/// per window, not multi-template consensus).
///
/// # Safety / precision
/// The `usize → f32` casts are safe because `ROLLING_WINDOW_N` is at most 3
/// (a trivially small value that fits exactly in f32's 23-bit mantissa).
#[expect(clippy::cast_precision_loss)]
fn match_threshold() -> f32 {
    (ROLLING_WINDOW_N as f32) * MATCH_THRESHOLD_FACTOR
}

/// Process a per-frame soft score through the rolling window and determine
/// whether wake word detection should fire (mahbot-773).
///
/// Returns `true` when the rolling sum of recent scores meets or exceeds
/// `match_threshold()`.  When the incoming score is below
/// [`NO_MATCH_RESET_THRESHOLD`], the window is cleared entirely to prevent
/// slow accumulation from noise.  On detection the score window is NOT
/// cleared here — the caller is responsible for full pipeline cleanup.
///
/// This function is pure with respect to global state: it only reads its
/// parameters and modifies `score_window` in place.  This makes it directly
/// testable without ONNX models or voice pipeline initialization.
fn process_wake_word_score(total_score: f32, score_window: &mut Vec<f32>) -> bool {
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
        let threshold = match_threshold();

        debug!(
            "Wake word score: total_score={total_score:.4} rolling_sum={rolling_sum:.4}/ \
             threshold={threshold:.2} window={}",
            score_window.len(),
        );

        if rolling_sum >= threshold {
            info!(
                "Wake word detected! rolling_sum={rolling_sum:.4} >= {threshold:.2} \
                 (window={} scores)",
                score_window.len(),
            );
            true
        } else {
            false
        }
    }
}

/// Process a single embedding through the wake word detection pipeline.
///
/// This is the **core detection loop** shared between the live pipeline
/// ([`try_match_wake_word_and_push_embedding`]), enrollment self-test
/// ([`run_enrollment_self_test`]), and integration tests.
///
/// It manages the ring buffer, runs the Conv1D MLP classifier forward pass,
/// applies rolling window scoring via [`process_wake_word_score`], and checks
/// the second-stage logistic regression verifier gate.  All three callers
/// exercise exactly the same code path, so changes to detection logic (ring
/// buffer sizing, MLP window, rolling sum threshold, verifier gating) are
/// automatically validated by the E2E test.
///
/// # Returns
/// - `true` — the embedding triggered wake word detection (all gates passed).
/// - `false` — continue feeding more embeddings (the ring buffer and score
///   window are updated for the next call).
///
/// # Parameters
/// - `embedding` — one 96-dim embedding vector to process.
/// - `embedding_ring` — persistent ring buffer (shared across frames in the
///   live pipeline; fresh per utterance in tests).
/// - `classifier` — trained Conv1D MLP classifier (`None` skips classification).
/// - `verifier` — trained logistic regression verifier (`None` skips the
///   second-stage gate, matching enrollment self-test behaviour).
/// - `score_window` — persistent rolling window of recent MLP confidence scores.
pub(crate) fn score_single_embedding(
    embedding: &[f32],
    embedding_ring: &mut Vec<Vec<f32>>,
    classifier: Option<&WakeWordClassifier>,
    verifier: Option<&VoiceVerifier>,
    score_window: &mut Vec<f32>,
) -> bool {
    // ── Ring buffer ───────────────────────────────────────────────────
    embedding_ring.push(embedding.to_vec());
    while embedding_ring.len() > EMBEDDING_RING_MAX {
        embedding_ring.remove(0);
    }

    // ── MLP classifier forward pass (replaces DTW template matching) ──
    // Needs 3 consecutive embeddings.  Before 3 are buffered, score is 0.
    let total_score = if let Some(classifier) = classifier {
        if embedding_ring.len() >= wake_word_classifier::WINDOW_SIZE {
            let start = embedding_ring.len() - wake_word_classifier::WINDOW_SIZE;
            let window = &embedding_ring[start..];
            classifier.forward(window)
        } else {
            0.0
        }
    } else {
        0.0
    };

    // ── Soft scoring + rolling window (mahbot-773) ────────────────────
    if process_wake_word_score(total_score, score_window) {
        // ── Verifier gate (mahbot-777, mahbot-788) ────────────────────
        // After the rolling window check passes, run the second-stage
        // logistic regression verifier to catch false positives that
        // survived the MLP classifier.
        if let Some(verifier) = verifier
            && verifier.is_trained()
        {
            let start = embedding_ring.len().saturating_sub(ROLLING_WINDOW_N);
            let max_score = embedding_ring[start..]
                .iter()
                .map(|emb| verifier.predict(emb))
                .fold(0.0f32, f32::max);
            if max_score < verifier.threshold {
                // Clear the score window so the next frame starts from zero.
                // Without this the accumulated classifier scores eventually
                // let any speech through (mahbot-797).
                score_window.clear();
                return false;
            }
        }
        return true;
    }
    false
}

// Higher VAD threshold for enrollment: only clear, close-mic speech should
/// pass during enrollment to prevent ambient noise (traffic, wind) from
/// contaminating the template (mahbot-772).  The detection VAD threshold
/// stays at 0.5 for responsiveness.
pub(crate) const ENROLLMENT_VAD_THRESHOLD: f32 = 0.85;

/// Minimum consecutive VAD-positive frames before setting utterance_had_speech
/// during enrollment (~48ms at 16ms/frame).  Prevents a single noise spike
/// from starting utterance accumulation (mahbot-772).
pub(crate) const ENROLLMENT_VAD_CONSECUTIVE_REQUIRED: usize = 3;

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
    /// Trained Conv1D classifier weights (None before enrollment).
    classifier_weights: Option<ClassifierWeights>,
    /// Cached classifier for inference (avoids per-frame clone of weights).
    /// Recreated when [`classifier_weights`] changes.
    classifier: Option<WakeWordClassifier>,
    /// Second-stage logistic regression verifier.
    verifier: VoiceVerifier,
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
            classifier_weights: None,
            classifier: None,
            verifier: VoiceVerifier::untrained(),
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
pub fn get_classifier_weights() -> Option<ClassifierWeights> {
    voice_state()
        .read()
        .unwrap_poison()
        .classifier_weights
        .clone()
}

pub fn set_classifier_weights(weights: ClassifierWeights) {
    let mut state = voice_state().write().unwrap_poison();
    state.classifier_weights = Some(weights.clone());
    state.classifier = Some(WakeWordClassifier::new(weights));
}

#[must_use]
pub fn get_verifier() -> VoiceVerifier {
    voice_state().read().unwrap_poison().verifier.clone()
}

pub fn set_verifier(verifier: VoiceVerifier) {
    voice_state().write().unwrap_poison().verifier = verifier;
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
///    artificially lowering distance to the enrollment cluster.
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
pub(crate) fn is_speech_with_detector(
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
    tx: &mpsc::Sender<Vec<f32>>,
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
            crate::util::resample_audio(&mono, sample_rate, SAMPLE_RATE)
        };
        if let Err(e) = tx.try_send(resampled) {
            debug!("Mic audio chunk dropped (1ch fast-path): {e}");
        }
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
        crate::util::resample_audio(&mono, sample_rate, SAMPLE_RATE)
    };
    if let Err(e) = tx.try_send(resampled) {
        debug!("Mic audio chunk dropped: {e}");
    }
}

fn start_microphone() -> Result<(mpsc::Receiver<Vec<f32>>, cpal::Stream)> {
    let (tx, rx) = mpsc::channel::<Vec<f32>>(MIC_CHANNEL_CAPACITY);

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

    let result = crate::providers::local_transcriber::transcribe_file_async(
        &tmp_path,
        Duration::from_secs(30),
    )
    .await;

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

// ═══════════════════════════════════════════════════════════════════════════
// Enrollment quality scoring (mahbot-778)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute a per-utterance quality score from the raw audio.
///
/// The composite score (0.0–1.0) is a weighted combination of:
/// - **Duration** (50%): whether the utterance is in [400ms, 2000ms].
/// - **Clipping** (25%): penalty if any sample hit i16::MAX.
/// - **SNR** (25%): estimated signal-to-noise ratio.  If `noise_rms` is
///   `Some`, uses the real pre-speech noise floor captured from the raw audio
///   ring during enrollment; otherwise falls back to an energy-based heuristic.
///
/// MLP classifier confidence is computed separately during enrollment
/// finalization (when the Conv1D classifier is trained on all utterances).
///
/// # Parameters
/// - `samples`: raw audio samples of the utterance.
/// - `noise_rms`: pre-speech ambient noise RMS captured at the moment of
///   first sustained speech detection (mahbot-782).  `None` falls back to
///   energy-based SNR estimation.
#[expect(clippy::cast_precision_loss)]
fn compute_utterance_quality(samples: &[f32], noise_rms: Option<f32>) -> UtteranceQuality {
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

    // ── Composite score (basic metrics only, no DTW) ──────────────────
    // During enrollment collection, quality is based on duration, clipping,
    // and SNR.  MLP confidence is computed at enrollment finalization after
    // the classifier is trained on all utterances.
    //
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

    let score = duration_score * 0.50 + clipping_score * 0.25 + snr_score * 0.25;

    UtteranceQuality {
        score,
        level: QualityLevel::from_score(score),
        clipping_detected,
        duration_ms,
        snr_db,
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

/// Run a self-test of the trained classifier against the enrollment buffer.
///
/// Simulates the live detection pipeline for each enrollment utterance: feeds
/// embeddings one by one through the embedding ring, runs the Conv1D MLP
/// classifier (`forward_pass`) on each 3-embedding window, and passes the
/// score through [`process_wake_word_score`] with a rolling window.
///
/// An utterance "triggers" if the rolling window sum exceeds the detection
/// threshold at any point.
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
    classifier: &WakeWordClassifier,
) -> Result<(), String> {
    if enrollment_buffer.is_empty() {
        return Err("Self-test skipped: no enrollment samples".to_string());
    }

    let mut passed = 0usize;

    for utterance in enrollment_buffer {
        // Fresh simulation for each utterance: no cross-utterance state.
        // Uses `score_single_embedding` (mahbot-811) which encapsulates the
        // same ring-buffer + MLP classifier + rolling window logic as the
        // live detection pipeline and the E2E integration test.
        let mut embedding_ring: Vec<Vec<f32>> = Vec::with_capacity(EMBEDDING_RING_MAX);
        let mut score_window = Vec::new();
        let mut detected = false;

        for embedding in utterance {
            if score_single_embedding(
                embedding,
                &mut embedding_ring,
                Some(classifier),
                None, // no verifier gate during enrollment self-test
                &mut score_window,
            ) {
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

/// Configuration parameters for [`segment_utterances_by_vad`].
///
/// Bundles the six module-level constants that are identical across all call
/// sites into a single struct.  This reduces the function signature from 8 to
/// 3 parameters and makes the call sites more resilient to parameter-order
/// changes.
pub(crate) struct VadSegmentationConfig {
    /// Frame size in samples (typically [`FRAME_LENGTH`] = 512).
    frame_length: usize,
    /// Frame stride in samples (typically [`HOP_LENGTH`] = 256).
    hop_length: usize,
    /// Min consecutive VAD-positive frames to confirm sustained speech
    /// (typically [`ENROLLMENT_VAD_CONSECUTIVE_REQUIRED`] = 3).
    consecutive_required: usize,
    /// Silence duration in samples before utterance ends
    /// (typically [`SILENCE_THRESHOLD_SAMPLES`] = 24 000 ≈ 1.5 s).
    silence_threshold_samples: usize,
    /// Samples of pre/post speech context to include
    /// (typically [`CONTEXT_PADDING_SAMPLES`] = 1 600 ≈ 100 ms).
    context_padding_samples: usize,
    /// Max samples in the internal raw-audio ring buffer
    /// (typically [`RAW_RING_MAX`] = 3 200 ≈ 200 ms).
    raw_ring_max: usize,
}

/// Module-level default config for [`segment_utterances_by_vad`] using the
/// standard voice-pipeline constants.
pub(crate) const DEFAULT_VAD_SEGMENTATION_CONFIG: VadSegmentationConfig = VadSegmentationConfig {
    frame_length: FRAME_LENGTH,
    hop_length: HOP_LENGTH,
    consecutive_required: ENROLLMENT_VAD_CONSECUTIVE_REQUIRED,
    silence_threshold_samples: SILENCE_THRESHOLD_SAMPLES,
    context_padding_samples: CONTEXT_PADDING_SAMPLES,
    raw_ring_max: RAW_RING_MAX,
};

/// Core VAD-gated utterance segmentation.
///
/// Processes raw audio with per-frame VAD decisions and segments it into
/// utterances using the same boundary-detection algorithm as the enrollment
/// pipeline's streaming handler ([`handle_enrollment_audio`]).  The caller
/// provides VAD decisions so this function is pure with respect to VAD state
/// management — it does not touch the global [`VAD_DETECTOR`].
///
/// # Algorithm (matches [`handle_enrollment_audio`])
///
/// 1. For each frame (stride = `config.hop_length`), check the VAD decision.
/// 2. VAD-positive frames accumulate into the current utterance.
/// 3. `config.consecutive_required` consecutive VAD-positive frames confirm
///    **sustained speech**.  On this transition the function prepends
///    pre-speech context (~100 ms) from the raw-audio ring buffer to capture
///    onset phonemes excluded by a strict VAD threshold.
/// 4. After speech, `config.silence_threshold_samples` of consecutive
///    VAD-negative audio ends the utterance.  Post-speech context (~100 ms)
///    is appended from the raw-audio ring (captured at the first silence
///    frame).
/// 5. The complete utterance is emitted and internal state resets for the
///    next utterance.
///
/// # Parameters
///
/// - `raw_audio`: Complete raw mono audio buffer (16 kHz f32 samples).
/// - `vad_decisions`: One boolean per frame (stride = `config.hop_length`).
///   Each decision is whether the frame at that position contains speech.
///   The caller is responsible for computing these (e.g. via
///   [`is_speech_with_detector`] on each frame).
/// - `config`: Segmentation parameters (see [`VadSegmentationConfig`]).
///   Use [`DEFAULT_VAD_SEGMENTATION_CONFIG`] for the standard pipeline.
///
/// # Returns
///
/// A list of utterance segments (raw audio samples, **not** VAD-subsampled),
/// in order of detection.  Each segment includes pre- and post-speech context
/// padding.  Empty if no utterances were detected.
///
/// # Panics
///
/// Panics if `vad_decisions` is empty or if the frame/hop parameters would
/// index past the end of `raw_audio`.
#[must_use]
pub(crate) fn segment_utterances_by_vad(
    raw_audio: &[f32],
    vad_decisions: &[bool],
    config: &VadSegmentationConfig,
) -> Vec<Vec<f32>> {
    let frame_length = config.frame_length;
    let hop_length = config.hop_length;
    let consecutive_required = config.consecutive_required;
    let silence_threshold_samples = config.silence_threshold_samples;
    let context_padding_samples = config.context_padding_samples;
    let raw_ring_max = config.raw_ring_max;

    // --- Validate parameters ---
    assert!(
        !vad_decisions.is_empty(),
        "segment_utterances_by_vad: vad_decisions must not be empty",
    );
    assert!(
        raw_audio.len() >= frame_length,
        "segment_utterances_by_vad: raw_audio too short \
         ({} < {frame_length})",
        raw_audio.len(),
    );

    let mut utterances: Vec<Vec<f32>> = Vec::new();
    let mut utterance_buf: Vec<f32> = Vec::new();
    let mut utterance_had_speech = false;
    let mut utterance_silence_samples: usize = 0;
    let mut utterance_speech_end_len: usize = 0;
    let mut vad_positives_in_a_row: usize = 0;
    let mut raw_audio_ring: Vec<f32> = Vec::with_capacity(raw_ring_max);
    let mut post_speech_tail: Vec<f32> = Vec::new();

    // Iterate frames at hop_length stride.
    // Each frame corresponds to one VAD decision.
    for (frame_idx, &is_speech) in vad_decisions.iter().enumerate() {
        let frame_start = frame_idx * hop_length;

        // Update raw-audio ring with the current frame's full-res samples.
        let frame_end = (frame_start + frame_length).min(raw_audio.len());
        if frame_end > frame_start {
            raw_audio_ring.extend_from_slice(&raw_audio[frame_start..frame_end]);
            if raw_audio_ring.len() > raw_ring_max {
                let excess = raw_audio_ring.len() - raw_ring_max;
                raw_audio_ring.drain(..excess);
            }
        }

        if is_speech {
            // VAD-positive: accumulate hop_length samples into utterance.
            let hop_end = (frame_start + hop_length).min(raw_audio.len());
            if hop_end > frame_start {
                utterance_buf.extend_from_slice(&raw_audio[frame_start..hop_end]);
            }

            vad_positives_in_a_row += 1;

            if vad_positives_in_a_row >= consecutive_required {
                // Sustained speech confirmed.

                if !utterance_had_speech {
                    // Prepend pre-speech context from raw-audio ring
                    // (first transition only).
                    let start = raw_audio_ring.len().saturating_sub(context_padding_samples);
                    let padding: Vec<f32> = raw_audio_ring[start..].to_vec();
                    if !padding.is_empty() {
                        let mut padded = padding;
                        padded.extend_from_slice(&utterance_buf);
                        utterance_buf = padded;
                    }
                }

                utterance_had_speech = true;
                utterance_speech_end_len = utterance_buf.len();
                utterance_silence_samples = 0;
            } else if utterance_had_speech {
                // Single VAD-positive frame after sustained speech:
                // extend utterance end and reset silence.
                utterance_speech_end_len = utterance_buf.len();
                utterance_silence_samples = 0;
            }
        } else {
            // VAD-negative: reset consecutive counter.
            vad_positives_in_a_row = 0;

            if utterance_had_speech {
                // Capture trailing speech at first silence.
                if utterance_silence_samples == 0 {
                    let start = raw_audio_ring.len().saturating_sub(context_padding_samples);
                    post_speech_tail = raw_audio_ring[start..].to_vec();
                }

                utterance_silence_samples += hop_length;

                if utterance_silence_samples >= silence_threshold_samples {
                    // Utterance is complete.
                    utterance_buf.truncate(utterance_speech_end_len);
                    if !post_speech_tail.is_empty() {
                        utterance_buf.extend_from_slice(&post_speech_tail);
                    }
                    if !utterance_buf.is_empty() {
                        utterances.push(std::mem::take(&mut utterance_buf));
                    }
                    utterance_speech_end_len = 0;
                    utterance_had_speech = false;
                    utterance_silence_samples = 0;
                    post_speech_tail.clear();
                    vad_positives_in_a_row = 0;
                }
            }
        }
    }

    utterances
}

/// Train the Conv1D wake word classifier from the enrollment buffer.
///
/// Returns the trained [`ClassifierWeights`] on success, after running the
/// self-test to verify ≥80% of enrollment utterances trigger detection.
fn finalize_enrollment(
    positive_embeddings: &[Vec<f32>],
    negative_embeddings: &[Vec<f32>],
    enrollment_buffer: &[Vec<Vec<f32>>],
) -> Result<ClassifierWeights> {
    if positive_embeddings.is_empty() {
        anyhow::bail!("No positive embeddings available for training");
    }

    // ── Train the Conv1D classifier ──
    let config = TrainingConfig::default();
    let weights =
        wake_word_classifier::train_classifier(positive_embeddings, negative_embeddings, &config)?;

    // ── Self-test: verify ≥80% of enrollment utterances trigger detection ──
    let classifier = WakeWordClassifier::new(weights.clone());
    if let Err(msg) = run_enrollment_self_test(enrollment_buffer, &classifier) {
        warn!("{msg}");
        return Err(anyhow!("{msg}"));
    }

    Ok(weights)
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

/// Resolve the workspace for a voice operation, falling back to the user's
/// configured personal workspace if no active workspace is set.
///
/// This mirrors the resolution pattern used by [`route_to_agent`] and the
/// error-broadcast path in [`handle_recording_audio`] (mahbot-812).
async fn resolve_workspace_for_voice(user_name: &str) -> String {
    let ws = active_workspace_name();
    if ws.is_empty() {
        if let Ok(Some(ws)) = crate::users::get_workspace(user_name).await {
            ws.name
        } else {
            let path = crate::users::personal_workspace_path(user_name);
            crate::users::personal_workspace_struct(user_name, &path).name
        }
    } else {
        ws
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
        let ws_name = resolve_workspace_for_voice(&user_name).await;

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
    mic_rx: Option<mpsc::Receiver<Vec<f32>>>,
    mic_stream: SendMicStream,
    is_listening: bool,
    is_recording: bool,
    command_buffer: Vec<f32>,
    /// Track silence duration by audio sample count rather than wall-clock
    /// time, so that system load / processing delays don't affect recording
    /// cutoff consistency (ticket mahbot-760).
    silence_sample_count: usize,
    enrollment_mode: bool,
    /// Accumulated VAD decisions across all frames processed this enrollment
    /// session.  Paired with [`frame_raw_audio`] for the extracted
    /// [`segment_utterances_by_vad`] function.
    frame_vad: Vec<bool>,
    /// Accumulated raw audio samples (full-resolution, NOT sub-sampled) for
    /// all frames processed this enrollment session.  Used by the extracted
    /// [`segment_utterances_by_vad`] function alongside [`frame_vad`].
    frame_raw_audio: Vec<f32>,
    /// Number of utterances already emitted by the extracted
    /// [`segment_utterances_by_vad`] function.  Reset across enrollment
    /// sessions (Cancel/Full) but preserved within a single session so the
    /// function is called on the full accumulated buffer each time.
    emitted_utterances: usize,
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
    /// Queue of completed enrollment utterances awaiting processing by the main
    /// pipeline loop.  Each element is a Vec<f32> of raw audio samples for one
    /// utterance.  The extracted [`segment_utterances_by_vad`] function detects
    /// utterance boundaries and queues completed utterances here.  The main loop
    /// pops them one at a time via [`pop_front`](VecDeque::pop_front) and
    /// processes them through [`handle_enrollment_sample`].  Using a queue
    /// (rather than a single `Option`) ensures that if multiple utterances
    /// complete within a single mic frame, all are preserved — no utterance is
    /// silently dropped (mahbot-823).
    enrollment_pending: VecDeque<Vec<f32>>,
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
    /// Rolling window of per-frame confidence scores from the Conv1D MLP
    /// classifier (mahbot-810).  Each element is the MLP confidence (0.0–1.0)
    /// for one 3-embedding window (~384ms of speech).  Detection fires when
    /// the sum over this window reaches [`match_threshold`].  Cleared entirely
    /// when a frame's score drops below [`NO_MATCH_RESET_THRESHOLD`] to
    /// prevent noise accumulation.
    score_window: Vec<f32>,
    /// Rolling buffer of raw audio samples captured BEFORE AGC processing.
    ///
    /// ~200ms capacity ([`RAW_RING_MAX`] samples).  Used exclusively for noise RMS estimation at first sustained
    /// speech detection — AGC amplifies silence (up to 4×) more than speech
    /// (~1-2×), so noise RMS from post-AGC audio produces an artificially low
    /// SNR estimate (mahbot-785).
    pre_agc_ring: Vec<f32>,
    /// Audio pre-processor for noise suppression and AGC.
    /// Applied to every incoming audio chunk before VAD / mel extraction.
    audio_preprocessor: crate::audio_preprocessor::AudioPreprocessor,
    /// Accumulates non-VAD audio frames during enrollment for use as negative
    /// training examples (mahbot-797).  Collected between utterances (pre-enrollment
    /// ambient noise, inter-utterance silence/background) and saved as chunks
    /// when sustained speech begins.
    negative_audio_buf: Vec<f32>,
    /// Timestamp until which the pipeline stays in [`VoiceStatus::Error`]
    /// before returning to [`VoiceStatus::Listening`] (mahbot-812).
    /// Set on transcription failure as a non-blocking replacement for the
    /// old 2-second sleep.
    refractory_until: Option<Instant>,
    /// Timestamp of the most recent error chat message, for rate-limiting
    /// repeated transcription failure notifications (mahbot-812).
    /// At most one error message per 10-second window.
    last_error_message_time: Option<Instant>,
}

/// Reset granularity for [`PipelineCtx::reset_pipeline_state`].
///
/// Each level maps to a category of pipeline transitions:
///
/// | Level | When to use | Behavioral summary |
/// |---|---|---|
/// | [`Full`](ResetLevel::Full) | New mic stream or acoustic environment change | Clears ALL buffers + calls `reset_vad()` + `audio_preprocessor.reset()` (new NoiseSuppressor) + resets `is_recording`, `auto_start_pending`, `vad_threshold`, `last_wake_word_detection`. Preserves global enrollment accumulators (survive mic stop/start). |
/// | [`Soft`](ResetLevel::Soft) | Same mic stream transition (enrollment↔detection, detection↔recording) | Clears audio accumulators + enrollment fields + `audio_preprocessor.clear_buffer()` (preserves NS noise profile) + preserves VAD state, `vad_threshold`, `last_wake_word_detection`, `auto_start_pending`, `is_recording`, and global enrollment accumulators |
/// | [`Cancel`](ResetLevel::Cancel) | Explicit enrollment cancellation or completion | Same buffer clearing as Soft + resets `vad_threshold` to [`VAD_THRESHOLD`] + clears `last_wake_word_detection` + clears global enrollment accumulators (`enrollment_buffer`, `negative_audio_chunks`) |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResetLevel {
    Full,
    Soft,
    Cancel,
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
            frame_vad: Vec::new(),
            frame_raw_audio: Vec::new(),
            emitted_utterances: 0,
            utterance_had_speech: false,
            utterance_silence_samples: 0,
            enrollment_no_speech_frame_count: 0,
            vad_positives_in_a_row: 0,
            audio_buffer: Vec::new(),
            mel_frame_buffer: Vec::new(),
            embedding_ring: Vec::new(),
            voice_batch: Vec::new(),
            enrollment_pending: VecDeque::new(),
            auto_start_pending: CONFIG.voice_enabled().as_deref() == Some("true"),
            last_model_retry: None,
            last_wake_word_detection: None,
            score_window: Vec::new(),
            noise_rms_estimate: None,
            vad_threshold: VAD_THRESHOLD,
            pre_agc_ring: Vec::new(),
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
            refractory_until: None,
            last_error_message_time: None,
        }
    }

    /// Parameterised pipeline state reset.
    ///
    /// | Field | Full | Soft | Cancel |
    /// |---|---|---|---|
    /// | `voice_batch`, `mel_frame_buffer`, `embedding_ring`, `audio_buffer`, `command_buffer`, `score_window`, `pre_agc_ring`, `negative_audio_buf`, `frame_vad`, `frame_raw_audio` | cleared | cleared | cleared |
    /// | `silence_sample_count` | = 0 | = 0 | = 0 |
    /// | `utterance_had_speech`, `utterance_silence_samples`, `enrollment_no_speech_frame_count`, `vad_positives_in_a_row`, `emitted_utterances`, `enrollment_pending`, `noise_rms_estimate` | cleared | cleared | cleared |
    /// | `vad_threshold` | `VAD_THRESHOLD` | preserved | `VAD_THRESHOLD` |
    /// | `last_wake_word_detection` | `None` | preserved | `None` |
    /// | `auto_start_pending` | `false` | preserved | preserved |
    /// | `is_recording` | `false` | preserved | preserved |
    /// | `audio_preprocessor` | `.reset()` | `.clear_buffer()` | `.clear_buffer()` |
    /// | VAD (`reset_vad()`) | called | NOT called | NOT called |
    /// | Global `enrollment_buffer`, `negative_audio_chunks` | preserved | preserved | cleared |
    /// | `refractory_until`, `last_error_message_time`, `last_model_retry`, `mic_rx`, `mic_stream`, `is_listening`, `enrollment_mode` | NOT touched | NOT touched | NOT touched |
    fn reset_pipeline_state(&mut self, level: ResetLevel) {
        // ── Audio accumulators (cleared by all levels) ──
        self.voice_batch.clear();
        self.mel_frame_buffer.clear();
        self.embedding_ring.clear();
        self.audio_buffer.clear();
        self.command_buffer.clear();
        self.silence_sample_count = 0;
        self.score_window.clear();
        self.pre_agc_ring.clear();
        self.negative_audio_buf.clear();

        // ── Enrollment detection/accumulator state (cleared by all levels) ──
        self.utterance_had_speech = false;
        self.utterance_silence_samples = 0;
        self.enrollment_no_speech_frame_count = 0;
        self.vad_positives_in_a_row = 0;
        self.enrollment_pending.clear();
        self.noise_rms_estimate = None;
        self.frame_vad.clear();
        self.frame_raw_audio.clear();
        self.emitted_utterances = 0;

        match level {
            ResetLevel::Full => {
                self.vad_threshold = VAD_THRESHOLD;
                self.last_wake_word_detection = None;
                self.auto_start_pending = false;
                self.is_recording = false;
                self.audio_preprocessor.reset();
                reset_vad();

                // Full does NOT clear global enrollment accumulators — those
                // survive mic stop/start cycles so mid-enrollment progress is
                // preserved across toggle-off/on (mahbot-800, mahbot-819).
                // Only ResetLevel::Cancel (explicit cancel or start-fresh)
                // clears the global enrollment buffer and negative audio chunks.
            }
            ResetLevel::Soft => {
                // Preserve VAD state, NS noise profile (clear_buffer, not reset),
                // vad_threshold, last_wake_word_detection cooldown,
                // auto_start_pending, is_recording, and global enrollment accumulators.
                self.audio_preprocessor.clear_buffer();
            }
            ResetLevel::Cancel => {
                self.vad_threshold = VAD_THRESHOLD;
                self.last_wake_word_detection = None;
                self.audio_preprocessor.clear_buffer();

                // Cancel also clears global enrollment accumulators.
                let mut state = voice_state().write().unwrap_poison();
                state.enrollment_buffer.clear();
                state.negative_audio_chunks.clear();
            }
        }
    }

    /// Transition from wake-word-detection to recording mode (mahbot-802).
    ///
    /// Performs the detection→recording handoff in the correct sequence:
    /// 1. Sets `is_recording = true` to route subsequent audio to recording
    /// 2. Clears `command_buffer` to start a fresh recording
    /// 3. Forwards the entire `audio_buffer` (processed wake-word tail +
    ///    unprocessed command-start) into `command_buffer` so ASR receives
    ///    the transition audio — ASR tolerates the extra wake-word overlap
    /// 4. Resets silence tracking for the recording phase
    /// 5. Clears intermediate detection buffers (mel, embedding, voice_batch,
    ///    score_window) to prevent stale data from contaminating the recording
    /// 6. Clears the noise-suppression frame-alignment buffer (mahbot-800 C2)
    /// 7. Records the detection timestamp for cooldown tracking
    /// 8. Does NOT reset VAD state or `vad_threshold` — the noise floor estimate
    ///    from the detection phase is deliberately carried through to recording
    ///    mode so Earshot does not need to re-establish the floor from scratch,
    ///    which would cause transient misclassification of the first few
    ///    recording frames (mahbot-802).
    ///
    /// Must be paired with `set_status(VoiceStatus::Recording)` by the caller
    /// because the status update is a side effect on global voice state, not
    /// on `PipelineCtx`.
    fn transition_to_recording(&mut self) {
        self.is_recording = true;
        // Save the wake-word tail + unprocessed command-start before the Soft
        // reset clears audio_buffer.  ASR tolerates the extra wake-word overlap
        // at the start (mahbot-802).
        let audio = std::mem::take(&mut self.audio_buffer);
        // Soft reset: clears detection buffers (mel, embedding, voice_batch,
        // score_window, pre_agc_ring,
        // negative_audio_buf, audio_preprocessor frame buffer) while
        // preserving VAD state, noise-suppressor noise profile, and
        // vad_threshold (mahbot-802).  command_buffer and silence_sample_count
        // are cleared by the reset — we re-populate command_buffer from the
        // saved audio.
        self.reset_pipeline_state(ResetLevel::Soft);
        self.command_buffer.extend_from_slice(&audio);
        self.last_wake_word_detection = Some(Instant::now());
    }

    /// Check whether the refractory period has elapsed and transition back
    /// to [`VoiceStatus::Listening`] if so (mahbot-812).
    ///
    /// Called once per main-loop iteration.  When the pipeline is in
    /// [`VoiceStatus::Error`] after a transcription failure, the refractory
    /// period (3 seconds) prevents immediate re-triggering.  Once the timer
    /// expires this method transitions back to Listening unless the pipeline
    /// is currently recording (which would mean a concurrent error path
    /// already initiated a new recording).
    fn check_refractory_period(&mut self) {
        if let Some(refractory_until) = self.refractory_until
            && Instant::now() >= refractory_until
        {
            self.refractory_until = None;
            // Only transition if we're in an Error state and not currently
            // recording — a concurrent error path could have cleared this.
            if !self.is_recording && matches!(get_status(), VoiceStatus::Error(_)) {
                set_status(VoiceStatus::Listening);
            }
        }
    }

    /// Check whether the 10-second rate-limit has elapsed since the last
    /// transcription-error chat message (mahbot-812).
    ///
    /// Returns `true` if no prior error occurred, or if at least 10 seconds
    /// have passed since the last one.  The caller broadcasts the error
    /// message on `true` and sets [`last_error_message_time`] to the current
    /// instant.
    fn should_send_error_message(&self) -> bool {
        let now = Instant::now();
        self.last_error_message_time
            .is_none_or(|t| now.duration_since(t).as_secs() >= 10)
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
            self.reset_pipeline_state(ResetLevel::Full);
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
        // Full reset: the mic stream is being torn down, so the old noise
        // profile and VAD state are no longer representative of the next
        // acoustic environment.  Full level uses audio_preprocessor.reset()
        // (new NoiseSuppressor) and reset_vad() (mahbot-800, mahbot-805).
        // Global enrollment accumulators are preserved across mic stop/start
        // so mid-enrollment progress survives toggle-off/on (mahbot-800,
        // mahbot-819).
        self.reset_pipeline_state(ResetLevel::Full);
        self.is_listening = false;
        self.enrollment_mode = false;
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

        // Resume existing enrollment progress if available (e.g., the user
        // clicked Enroll while already enrolled or mid-enrollment — the
        // global enrollment_buffer from the interrupted session is intact).
        // When starting fresh (existing_count == 0), use Cancel-level reset
        // to clear stale buffers while preserving VAD/NS continuity (same
        // mic stream, same acoustic environment) — mahbot-805.
        let existing_count = voice_state().read().unwrap_poison().enrollment_buffer.len();

        if existing_count == 0 {
            // Cancel-level reset: clears all audio buffers, enrollment
            // accumulators, and global enrollment state, but preserves
            // VAD/NS continuity (same mic stream, same acoustic environment).
            // vad_threshold will be set to ENROLLMENT_VAD_THRESHOLD below.
            self.reset_pipeline_state(ResetLevel::Cancel);
        }

        self.enrollment_mode = true;
        self.vad_threshold = ENROLLMENT_VAD_THRESHOLD;
        set_status(VoiceStatus::Enrolling {
            sample: existing_count,
            total: NUM_ENROLLMENT_SAMPLES,
            duration_ms: 0,
            quality: None,
        });
        info!(
            "Voice pipeline: enrollment started (resuming from sample {}/{NUM_ENROLLMENT_SAMPLES})",
            existing_count,
        );
    }

    fn handle_cancel_enrollment(&mut self) {
        self.reset_pipeline_state(ResetLevel::Cancel);
        self.enrollment_mode = false;
        // vad_threshold already restored to VAD_THRESHOLD by Cancel level.
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

    // Load persisted wake word classifier weights from config on startup
    if let Some(json) = CONFIG.wake_word_templates() {
        // NOTE: The config key remains "wake_word_templates" for backward
        // compatibility.  Tries new format first, falls back to old format.
        // Old format: just ClassifierWeights
        // New format: { classifier: Option<ClassifierWeights>, verifier: VoiceVerifier }

        // Try new combined model format — uses module-level PersistedModel struct

        let loaded = if let Ok(model) = serde_json::from_str::<PersistedModel>(&json) {
            if let Some(ref w) = model.classifier {
                if let Err(e) = w.validate() {
                    warn!("Stored classifier weights are invalid — re-enrollment required: {e}");
                    false
                } else {
                    set_classifier_weights(w.clone());
                    if let Some(ref v) = model.verifier {
                        set_verifier(v.clone());
                    }
                    info!("Loaded wake word model (classifier + verifier) from config");
                    true
                }
            } else {
                false
            }
        } else {
            false
        };

        // Fall back to old format (just ClassifierWeights)
        if !loaded {
            if let Ok(weights) = serde_json::from_str::<ClassifierWeights>(&json) {
                if let Err(e) = weights.validate() {
                    warn!("Stored classifier weights are invalid — re-enrollment required: {e}");
                } else {
                    set_classifier_weights(weights);
                    info!("Loaded wake word classifier weights from config (legacy format)");
                }
            } else {
                warn!(
                    "Failed to deserialize stored wake word classifier weights. \
                     Clear and re-enroll."
                );
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

        // Process any pending enrollment utterances (accumulated inline to avoid
        // race conditions with the command channel).  Using a VecDeque so all
        // completed utterances are preserved even if multiple complete within a
        // single mic frame — each is popped one per tick (mahbot-823).
        // ONNX inference inside handle_enrollment_sample uses spawn_blocking
        // so it doesn't block.
        if let Some(samples) = ctx.enrollment_pending.pop_front() {
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
                // Cancel level: clears audio buffers, enrollment accumulators,
                // restores vad_threshold to VAD_THRESHOLD, but preserves VAD/NS
                // continuity (same mic stream, same acoustic environment).
                // Does NOT call reset_vad() — the earshot noise floor estimate
                // from the enrollment phase is deliberately carried through to
                // detection mode (mahbot-805).
                ctx.reset_pipeline_state(ResetLevel::Cancel);
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

        // Transition from Error to Listening after the refractory period
        // has elapsed (mahbot-812).  This replaces the old 2-second blocking
        // sleep with a non-blocking check in the main loop.
        ctx.check_refractory_period();

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
    // Reject completely empty embeddings (no classifier input to process),
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
                let quality = Some(compute_utterance_quality(&samples_for_quality, noise_rms));

                state.enrollment_buffer.push(embeddings);
                let count = state.enrollment_buffer.len();
                // state dropped here — no lock held across await
                (count, quality)
            };

            if count >= NUM_ENROLLMENT_SAMPLES {
                // ── Collect positive and negative embeddings ──
                let enrollment_buffer = {
                    let state = voice_state().read().unwrap_poison();
                    state.enrollment_buffer.clone()
                };

                let positive_embeddings: Vec<Vec<f32>> = enrollment_buffer
                    .iter()
                    .flat_map(|sample| sample.iter().cloned())
                    .collect();

                // Extract negative embeddings from ambient audio chunks
                let (negative_embeddings, n_chunks) = {
                    let state = voice_state().read().unwrap_poison();
                    let chunks = state.negative_audio_chunks.clone();
                    let n = chunks.len();
                    (chunks, n)
                };
                let (negative_embeddings, used_real_negatives) = {
                    if n_chunks >= 2 && ONNX_MODELS.get().is_some() {
                        let result = tokio::task::spawn_blocking(move || {
                            let models = ONNX_MODELS.get().expect("ONNX_MODELS checked above");
                            let mut all_neg = Vec::new();
                            for chunk in &negative_embeddings {
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
                        let used = !result.is_empty();
                        (result, used)
                    } else {
                        (Vec::new(), false)
                    }
                };

                // Conditionally clear negative_audio_chunks
                if used_real_negatives {
                    voice_state()
                        .write()
                        .unwrap_poison()
                        .negative_audio_chunks
                        .clear();
                }

                // ── Train the Conv1D classifier (blocking, pure Rust manual backprop) ──
                let classifier_result = {
                    let pos = positive_embeddings.clone();
                    let neg = negative_embeddings.clone();
                    let buf = enrollment_buffer.clone();
                    tokio::task::spawn_blocking(move || finalize_enrollment(&pos, &neg, &buf))
                        .await
                        .unwrap_or_else(|e| Err(anyhow!("Classifier training task panicked: {e}")))
                };

                let weights = match classifier_result {
                    Ok(w) => {
                        info!(
                            "Enrollment complete: wake word '{}' (classifier trained: {} params)",
                            WAKE_WORD_NAME,
                            w.param_count(),
                        );
                        w
                    }
                    Err(e) => {
                        warn!("Enrollment finalization failed: {e}");
                        set_status(VoiceStatus::Error("Enrollment failed".to_string()));
                        return;
                    }
                };

                // ── Train verifier (mahbot-777, mahbot-797) ──────
                let verifier = if positive_embeddings.is_empty() {
                    warn!(
                        "Could not train verifier: no valid positive embeddings \
                         from {} enrollment samples",
                        positive_embeddings.len(),
                    );
                    crate::voice_verifier::VoiceVerifier::untrained()
                } else if !negative_embeddings.is_empty() {
                    let n_neg = negative_embeddings.len();
                    let v = crate::voice_verifier::VoiceVerifier::train(
                        &positive_embeddings,
                        &negative_embeddings,
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
                    let v = crate::voice_verifier::VoiceVerifier::train_with_synthetic_negatives(
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
                };

                // ── Store classifier + verifier in global state ──
                set_classifier_weights(weights.clone());
                set_verifier(verifier);

                // ── Persist to config DB ──
                persist_model_state().await;

                // Clear enrollment mode
                set_status(VoiceStatus::Enrolled);
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

/// Serialisable form of the wake word model (classifier + verifier).
/// `verifier` is optional for backward compatibility with legacy persistence.
#[derive(serde::Serialize, serde::Deserialize)]
struct PersistedModel {
    classifier: Option<ClassifierWeights>,
    verifier: Option<VoiceVerifier>,
}

/// Persist current classifier weights and verifier to the config database.
async fn persist_model_state() {
    let weights = get_classifier_weights();
    let verifier = get_verifier();
    let model = PersistedModel {
        classifier: weights,
        verifier: Some(verifier),
    };
    if let Ok(json) = serde_json::to_string(&model) {
        let store = crate::config_db::store();
        if let Err(e) = store.set_kv("wake_word_templates", &json).await {
            warn!("Failed to persist wake word model: {e}");
        } else {
            // Update CONFIG in-memory so that GUI snapshot readers / pipeline
            // restart see the latest model.  `save_and_reload` no longer
            // touches `wake_word_templates` (it's skipped in the write loop),
            // so this update is about cross-session visibility, not deletion
            // prevention.
            if !CONFIG.set_string_field("wake_word_templates", &json) {
                warn!(
                    "Failed to update CONFIG with wake word model (key not recognized by \
                     set_string_field — it may have drifted from the `stringify!` arms)"
                );
            }
            info!("Wake word model persisted to config");
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
    let speech = is_speech_with_threshold(&samples, ctx.vad_threshold);
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
                // Guard: skip routing if ASR produced an empty or
                // whitespace-only transcription (qwen-asr can return
                // Ok("") when the model outputs zero text tokens).
                if transcribed.trim().is_empty() {
                    warn!(
                        "Empty transcription — dropping ({} samples, {:.1}s)",
                        cmd_buf.len(),
                        duration_secs,
                    );
                } else {
                    info!("Transcribed: {transcribed}");
                    route_to_agent(transcribed).await;
                }

                // Cleanup: return to listening immediately on success.
                // Soft reset clears detection/recording buffers (mel, embedding,
                // voice_batch, score_window, command_buffer,
                // pre_agc_ring, negative_audio_buf) while
                // preserving VAD state, NS noise profile, vad_threshold, and the
                // wake-word cooldown timestamp to prevent immediate re-triggering
                // (mahbot-805).
                ctx.reset_pipeline_state(ResetLevel::Soft);
                ctx.is_recording = false;
                set_status(VoiceStatus::Listening);
            }
            Err(e) => {
                warn!("Transcription failed: {e}");
                set_status(VoiceStatus::Error("Transcription failed".to_string()));

                // Broadcast an error chat message (rate-limited: at most one
                // per 10 seconds) so the user sees a persistent indicator
                // instead of a flash that disappears after 2s (mahbot-812).
                if ctx.should_send_error_message() {
                    let user_name = active_user_name();
                    if !user_name.is_empty() {
                        ctx.last_error_message_time = Some(Instant::now());

                        // Resolve workspace with fallback via the shared helper
                        // (matching the route_to_agent pattern).
                        let workspace = resolve_workspace_for_voice(&user_name).await;

                        crate::channels::broadcast_and_persist_agent_response(
                            &user_name,
                            "voice",
                            "*Voice: transcription failed — try again*",
                            Some("voice".to_string()),
                            &workspace,
                        )
                        .await;
                    }
                }

                // Enforce a 3-second refractory period before returning to
                // Listening (replaces the old 2-second blocking sleep with a
                // non-blocking alternative, mahbot-812).
                ctx.refractory_until = Some(Instant::now() + Duration::from_secs(3));

                // Cleanup the recording state.
                // Soft reset clears recording/detection buffers while preserving
                // VAD/NS continuity so the noise floor estimate survives the
                // refractory period (mahbot-805).
                ctx.reset_pipeline_state(ResetLevel::Soft);
                ctx.is_recording = false;
                // Do NOT set status to Listening here — the refractory delay
                // is handled in the main loop's post-select section.
            }
        }
    }
}

/// Retain only the last [`VOICE_BATCH_OVERLAP`] samples in `voice_batch`
/// as overlap context for the next mel spectrogram batch.
///
/// This function is extracted from [`flush_voice_batch`] so the overlap
/// trimming logic can be tested in isolation (the ONNX inference inside
/// `flush_voice_batch` requires model files and cannot run in unit tests).
/// See [`test_mel_stride_overlap_alignment`] for the behavioral test.
///
/// # Caution
/// [`flush_voice_batch`] **must** call this function after each successful
/// mel flush.  The test suite validates `trim_voice_batch` in isolation but
/// cannot verify the call site — ONNX models are unavailable in unit tests,
/// so `flush_voice_batch` returns early before reaching the trim call.
/// Removing this call without a replacement would create a regression gap
/// (see [`test_mel_stride_overlap_alignment`] which documents this gap).
fn trim_voice_batch(voice_batch: &mut Vec<f32>) {
    let keep = VOICE_BATCH_OVERLAP;
    if voice_batch.len() > keep {
        let drain_to = voice_batch.len() - keep;
        voice_batch.drain(..drain_to);
    }
}

/// Process accumulated voiced audio through the mel spectrogram ONNX model.
/// Batches multiple frames into a single ONNX call for efficiency.
///
/// ONNX inference (`compute_mel_spectrogram`) is CPU-bound. We wrap it in
/// `block_in_place` so the tokio runtime can run other tasks on this thread
/// during inference, consistent with the enrollment path which uses
/// `spawn_blocking` for the same purpose.
///
/// # Overlap management
/// After a successful ONNX call, [`trim_voice_batch`] trims `voice_batch` to
/// retain only the last [`VOICE_BATCH_OVERLAP`] samples as overlap context
/// for the next batch.  This ensures mel frame positions are aligned across
/// batch boundaries (mahbot-799).  Removing this call would create a
/// regression gap — the test suite cannot verify the call site because ONNX
/// models are unavailable in unit tests (see [`trim_voice_batch`]'s
/// `# Caution` note).
fn flush_voice_batch(voice_batch: &mut Vec<f32>, mel_frame_buffer: &mut Vec<Vec<f32>>) {
    if voice_batch.len() < FRAME_LENGTH {
        return; // not enough for a single frame
    }
    let Some(models) = ONNX_MODELS.get() else {
        return;
    };

    let frames =
        crate::util::with_block_in_place(|| compute_mel_spectrogram(models, &*voice_batch));
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
            // Keep only the last VOICE_BATCH_OVERLAP (MEL_STRIDE × 2 = 320)
            // samples as overlap context across batch boundaries.  The ONNX
            // mel spectrogram model uses an internal stride of 160 samples
            // (MEL_STRIDE), so the overlap must be a multiple of 160 to align
            // mel frame positions at the batch boundary.
            //
            // Using 2 × MEL_STRIDE (320 samples = 20ms) provides two full mel
            // frames of overlap context:
            //   - Frame 0 of the new batch covers [P-320 .. P+192] — starts at
            //     a valid stride position
            //   - Frame 1 covers [P-160 .. P+352] — also valid
            // Both match exactly what continuous processing would compute.
            //
            // The previous value (FRAME_LENGTH - HOP_LENGTH = 256 samples) was
            // NOT a multiple of 160, causing ~6ms of temporal offset drift at
            // each batch boundary that accumulated ~45ms across 76 frames
            // (mahbot-799).
            trim_voice_batch(voice_batch);
        }
        Err(e) => {
            warn!("Mel spectrogram failed: {e}");
            // Clear the batch so it doesn't grow unbounded when the
            // ONNX model is consistently failing (ticket mahbot-760).
            voice_batch.clear();
        }
    }
}

/// Handle wake word detection: process audio frames through mel/embedding/Conv1D MLP classifier.
///
/// Audio arrives in small chunks (~256 samples at 16kHz). This function:
/// 1. Accumulates audio in a sliding window for VAD
/// 2. Collects voiced frames into a batch buffer
/// 3. Processes the batch through mel ONNX when enough audio is accumulated (~128ms)
/// 4. Produces embeddings and runs the Conv1D MLP classifier on 3-embedding windows
/// 5. Passes MLP confidence scores through the rolling window accumulator
///
/// Batching reduces ONNX inference calls from ~62/sec (per-frame) to ~8/sec.
///
/// Implements cooldown (mahbot-770 Fix 2) and soft-scoring + rolling window
/// detection (mahbot-773) via the `last_wake_word_detection` and
/// `score_window` fields.
fn handle_wake_word_detection(samples: &[f32], ctx: &mut PipelineCtx) {
    // ── Cooldown check (mahbot-770 Fix 2) ──
    // If we recently detected the wake word, skip ALL processing for this
    // chunk to prevent rapid consecutive false triggers.  During cooldown
    // audio accumulates into audio_buffer with a cap (mahbot-802) so that
    // when the cooldown expires the pipeline has data to process immediately;
    // intermediate detection buffers are cleared to prevent stale data from
    // the previous utterance causing false triggers (mahbot-770 Fix 2).
    if let Some(last) = ctx.last_wake_word_detection
        && last.elapsed() < WAKE_WORD_COOLDOWN
    {
        debug!(
            "Wake word cooldown active ({}ms elapsed)",
            last.elapsed().as_millis()
        );
        // Accumulate audio during cooldown so the pipeline has data
        // when cooldown expires — don't discard it entirely (mahbot-802).
        // See [`COOLDOWN_ACCUMULATION_CAP`] for the frame-processing
        // arithmetic that justifies the cap value (2 frames = 1024 samples).
        //
        // We accumulate into audio_buffer (not command_buffer) because during
        // cooldown is_recording is false, so audio is routed to
        // handle_wake_word_detection, not to handle_recording_audio.
        // command_buffer is only populated after detection transitions to
        // recording mode (mahbot-802 deviation from ticket §2).
        //
        // Invariant: audio_buffer is empty or within COOLDOWN_ACCUMULATION_CAP at
        // cooldown entry.  Caller sequencing (transition_to_recording() clears it at
        // the detection→recording handoff) ensures this invariant holds — this
        // assertion guards against future refactors that bypass that path.
        debug_assert!(
            ctx.audio_buffer.len() <= COOLDOWN_ACCUMULATION_CAP,
            "audio_buffer (len={}) exceeds COOLDOWN_ACCUMULATION_CAP ({}) at cooldown entry; \
             transition_to_recording() should have cleared it at detection→recording handoff",
            ctx.audio_buffer.len(),
            COOLDOWN_ACCUMULATION_CAP
        );
        let remaining = COOLDOWN_ACCUMULATION_CAP.saturating_sub(ctx.audio_buffer.len());
        let n = samples.len().min(remaining);
        ctx.audio_buffer.extend_from_slice(&samples[..n]);
        // Clear intermediate detection buffers to prevent stale data
        // from the previous utterance causing false detections.
        // VAD is intentionally NOT reset here: the accumulated audio_buffer
        // naturally refills Earshot's internal ring buffer when processing
        // resumes after cooldown expiry.  A manual reset_vad() would lose
        // the noise floor estimate (mahbot-802).
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
/// Accumulates VAD decisions and raw audio into `ctx.frame_vad` /
/// `ctx.frame_raw_audio`, then after the frame loop calls the extracted
/// [`segment_utterances_by_vad`] function which performs utterance-boundary
/// detection.  Newly completed utterances are queued in `enrollment_pending`
/// (a `VecDeque`) for the main loop to process one per tick (avoids race
/// conditions with the command channel).  Using a queue ensures no utterance
/// is dropped if multiple complete within a single mic frame (mahbot-823).  The frame loop maintains lightweight inline state only for
/// side-effect gating (noise RMS capture, negative audio collection, UI
/// status) — utterance construction itself is delegated to the extracted
/// function (mahbot-823).
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
/// by the extracted function into utterances, mirroring the detection pipeline
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
    ctx.audio_buffer.extend_from_slice(samples);

    // ── Accumulate full-res audio for extracted VAD gating function ──
    // Store the ORIGINAL mic samples (not sub-sampled, not framed) so
    // [`segment_utterances_by_vad`] has full 16 kHz resolution for its
    // internal raw-audio ring (correct context padding) and can access
    // frames at hop_length stride to align with `ctx.frame_vad`.
    ctx.frame_raw_audio.extend_from_slice(samples);

    // Process frames with offset tracking instead of per-iteration O(n) drain.
    let len = ctx.audio_buffer.len();
    let mut consumed = 0;
    while consumed + FRAME_LENGTH <= len {
        let frame = &ctx.audio_buffer[consumed..consumed + FRAME_LENGTH];

        // ── Accumulate VAD decision for extracted function ──
        let is_speech = is_speech_with_threshold(frame, ctx.vad_threshold);
        ctx.frame_vad.push(is_speech);

        if is_speech {
            ctx.vad_positives_in_a_row += 1;
            // Reset the no-speech warning counter on any VAD-positive frame.
            ctx.enrollment_no_speech_frame_count = 0;

            if ctx.vad_positives_in_a_row >= ENROLLMENT_VAD_CONSECUTIVE_REQUIRED {
                // Sustained speech confirmed: perform inline side-effect gating.
                // The extracted [`segment_utterances_by_vad`] function handles
                // utterance-boundary detection and construction (mahbot-823);
                // the inline code below only gates real-time side effects that
                // must happen during the frame loop: noise RMS capture (mahbot-785),
                // negative audio collection (mahbot-797), and UI status updates.

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
                        let mut state = voice_state().write().unwrap_poison();
                        if state.negative_audio_chunks.len() >= MAX_NEGATIVE_AUDIO_CHUNKS {
                            warn!(
                                "negative_audio_chunks at max ({}): discarding oldest chunk \
                                 to cap memory growth (mahbot-800)",
                                MAX_NEGATIVE_AUDIO_CHUNKS,
                            );
                            state.negative_audio_chunks.remove(0);
                        }
                        state
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

                if !already_had_speech || was_waiting_for_silence {
                    // Transition from silence to speech, or speech resumed after
                    // a pause before the 1.5s timeout — show "Listening…"
                    set_status(VoiceStatus::ListeningDuringEnrollment { sample, total });
                }
                ctx.utterance_had_speech = true;
                ctx.utterance_silence_samples = 0;
            } else if ctx.utterance_had_speech {
                // A single VAD-positive frame (below the consecutive threshold)
                // after sustained speech: just reset silence.  Handles brief
                // VAD gaps during continuous speech (e.g. unvoiced stops).
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

                ctx.utterance_silence_samples += HOP_LENGTH;

                // Inline silence threshold reset: partially duplicates the
                // extracted function's boundary logic but is needed so that
                // post-utterance silence within the same chunk accumulates
                // into negative_audio_buf rather than being discarded
                // (mahbot-797, mahbot-823).
                // Snapshot before reset so the UI status gate uses the
                // pre-reset value — after utterance completion the silence
                // samples are 0, which would spuriously trigger the "waiting
                // for silence" status for one frame (mahbot-823).
                let silence_ui_check = ctx.utterance_silence_samples;

                if ctx.utterance_silence_samples >= SILENCE_THRESHOLD_SAMPLES {
                    ctx.utterance_had_speech = false;
                    ctx.utterance_silence_samples = 0;
                    ctx.enrollment_no_speech_frame_count = 0;
                    ctx.vad_positives_in_a_row = 0;
                }

                // Set status during the first 200ms of silence to show
                // "Keep silent to confirm…".  Uses the snapshot captured
                // before the threshold reset so that utterance completion
                // does not spuriously re-trigger this status.
                if silence_ui_check < SILENCE_UI_GATE_SAMPLES {
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

    // ── Extracted VAD gating: detect completed utterances ─────────────
    // Call the same pure function that the E2E integration test exercises.
    // Uses the full accumulated audio and VAD decisions to detect utterance
    // boundaries with the same algorithm as the inline logic above.
    // `emitted_utterances` tracks how many utterances have already been
    // processed across calls — newly completed utterances are those with
    // index >= emitted_utterances.
    if !ctx.frame_vad.is_empty() {
        let utterances = segment_utterances_by_vad(
            &ctx.frame_raw_audio,
            &ctx.frame_vad,
            &DEFAULT_VAD_SEGMENTATION_CONFIG,
        );

        // Handle any newly completed utterances
        while utterances.len() > ctx.emitted_utterances {
            let new_idx = ctx.emitted_utterances;
            ctx.emitted_utterances += 1;

            // Queue the utterance in enrollment_pending (VecDeque) for the
            // main loop to process.  Using a queue ensures all utterances are
            // preserved even if multiple complete within a single mic frame.
            // The index starts at the previous emitted_utterances count, which
            // is the oldest unprocessed utterance; utterances are processed
            // in detection order.
            let utterance = utterances[new_idx].clone();
            ctx.enrollment_pending.push_back(utterance);

            // Reset inline tracking state for the next utterance.
            // Fields used for side-effect gating in the frame loop are reset
            // so the next utterance starts with a clean slate.
            ctx.utterance_had_speech = false;
            ctx.utterance_silence_samples = 0;
            ctx.vad_positives_in_a_row = 0;
            ctx.enrollment_no_speech_frame_count = 0;
            // Note: noise_rms_estimate is intentionally NOT reset here.
            // It is consumed by the main loop alongside enrollment_pending.
            // Reset is handled by reset_pipeline_state(Cancel) for
            // cancellation/completion safety (mahbot-782).
        }
    }
}

/// Compute embedding from mel frames, push to ring buffer, and match against the Conv1D/MLP classifier.
///
/// Implements the two-stage cascade (mahbot-810): the Conv1D MLP classifier
/// produces a single confidence score (0.0–1.0) from the last 3 embeddings;
/// the score is accumulated in a rolling window via [`process_wake_word_score`].
/// Detection fires when the rolling sum exceeds [`match_threshold`].
/// The window is reset entirely when a frame's score drops below
/// [`NO_MATCH_RESET_THRESHOLD`] to prevent noise accumulation.
/// If the rolling window triggers, the voice verifier (logistic regression)
/// acts as a second-stage AND gate before confirming detection.
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

    // ── Shared detection scoring (mahbot-811) ────────────────────────
    // Uses `score_single_embedding` which encapsulates the ring buffer,
    // MLP classifier forward pass, rolling window scoring, and verifier
    // gate — the same logic exercised by `run_enrollment_self_test` and
    // the E2E integration test.  Any change to detection heuristics is
    // automatically validated by the integration test.
    let state = voice_state().read().unwrap_poison();
    let classifier: Option<&WakeWordClassifier> = state.classifier.as_ref();
    let verifier = get_verifier();
    if score_single_embedding(
        &embedding,
        &mut ctx.embedding_ring,
        classifier,
        Some(&verifier),
        &mut ctx.score_window,
    ) {
        // Wake word detected — transition to recording mode.
        // See [`PipelineCtx::transition_to_recording`] for the exact
        // handoff sequence (mahbot-802).
        drop(state); // release read lock before side effects
        ctx.transition_to_recording();
        set_status(VoiceStatus::Recording);
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    // ── VAD-gated utterance segmentation tests ────────────────────────────
    // These test the pure [`segment_utterances_by_vad`] function with synthetic
    // audio and manually-computed VAD decisions.  No global voice state needed.

    /// Test config with a shorter silence threshold (10 frames ≈ 2560 samples
    /// instead of the default 94 frames ≈ 24000 samples) so tests don't need
    /// prohibitively long audio buffers.
    const TEST_VAD_CONFIG: VadSegmentationConfig = VadSegmentationConfig {
        frame_length: FRAME_LENGTH,
        hop_length: HOP_LENGTH,
        consecutive_required: ENROLLMENT_VAD_CONSECUTIVE_REQUIRED,
        silence_threshold_samples: HOP_LENGTH * 10, // 2560 samples ≈ 10 frames
        context_padding_samples: CONTEXT_PADDING_SAMPLES,
        raw_ring_max: RAW_RING_MAX,
    };

    /// Build a raw audio buffer long enough for `n_frames` frames at
    /// [`HOP_LENGTH`] stride with [`FRAME_LENGTH`] window size.
    fn audio_for_frames(n_frames: usize) -> Vec<f32> {
        let last_end = (n_frames.saturating_sub(1)) * HOP_LENGTH + FRAME_LENGTH;
        vec![0.0f32; last_end]
    }

    #[test]
    fn segment_no_speech_returns_empty() {
        // No VAD-positive frames at all → no utterances.
        let n_frames = 10;
        let audio = audio_for_frames(n_frames);
        let vad = vec![false; n_frames];
        let utterances = segment_utterances_by_vad(&audio, &vad, &TEST_VAD_CONFIG);
        assert!(utterances.is_empty(), "no speech → no utterances");
    }

    #[test]
    fn segment_single_utterance_detected() {
        // 4 speech frames (≥3 consecutive → sustained) + 10 silence frames (≥10 → threshold).
        let audio = audio_for_frames(14);
        let mut vad = vec![true; 4];
        vad.extend(vec![false; 10]);
        let utterances = segment_utterances_by_vad(&audio, &vad, &TEST_VAD_CONFIG);
        assert_eq!(
            utterances.len(),
            1,
            "sustained speech + silence → 1 utterance"
        );
        assert!(
            utterances[0].len() >= CONTEXT_PADDING_SAMPLES,
            "utterance should include pre-speech context padding"
        );
    }

    #[test]
    fn segment_multiple_utterances_separated_by_silence() {
        // 4 speech, 10 silence, 3 speech, 10 silence → 2 utterances.
        let n_frames = 4 + 10 + 3 + 10;
        let audio = audio_for_frames(n_frames);
        let mut vad = vec![true; 4];
        vad.extend(vec![false; 10]);
        vad.extend(vec![true; 3]);
        vad.extend(vec![false; 10]);
        let utterances = segment_utterances_by_vad(&audio, &vad, &TEST_VAD_CONFIG);
        assert_eq!(utterances.len(), 2, "two speech segments → two utterances");
        for (i, utt) in utterances.iter().enumerate() {
            assert!(
                utt.len() >= CONTEXT_PADDING_SAMPLES,
                "utterance {i} should include context padding"
            );
        }
    }

    #[test]
    fn segment_short_burst_not_sustained() {
        // 2 speech frames (fewer than consecutive_required = 3) → no sustained speech.
        let n_frames = 10;
        let audio = audio_for_frames(n_frames);
        let mut vad = vec![true; 2];
        vad.extend(vec![false; 8]);
        let utterances = segment_utterances_by_vad(&audio, &vad, &TEST_VAD_CONFIG);
        assert!(
            utterances.is_empty(),
            "short burst below consecutive_required → no utterance",
        );
    }

    #[test]
    fn segment_utterance_at_end_without_silence_not_emitted() {
        // 8 silence frames, then 4 speech frames at end — no trailing silence
        // to cross the threshold, so no utterance is emitted.
        let n_frames = 12;
        let audio = audio_for_frames(n_frames);
        let mut vad = vec![false; 8];
        vad.extend(vec![true; 4]);
        let utterances = segment_utterances_by_vad(&audio, &vad, &TEST_VAD_CONFIG);
        assert!(
            utterances.is_empty(),
            "speech at end without trailing silence → no utterance",
        );
    }

    // ── Refractory period state-machine tests ────────────────────────────
    // These test the Error→Listening transition logic via the canonical
    // [`PipelineCtx::check_refractory_period`] method.
    // Uses serial_test to isolate global voice-state mutations.

    #[test]
    #[serial_test::serial(voice)]
    fn refractory_transitions_from_error_to_listening() {
        let _ = init_global();

        let mut ctx = PipelineCtx::new();
        ctx.is_recording = false;
        ctx.refractory_until = Some(
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("1s in the past should not underflow"),
        );

        set_status(VoiceStatus::Error("test error".to_string()));

        ctx.check_refractory_period();

        assert!(matches!(get_status(), VoiceStatus::Listening));
        assert!(ctx.refractory_until.is_none());
    }

    #[test]
    #[serial_test::serial(voice)]
    fn refractory_does_not_transition_if_not_error() {
        let _ = init_global();

        let mut ctx = PipelineCtx::new();
        ctx.is_recording = false;
        ctx.refractory_until = Some(
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("1s in the past should not underflow"),
        );

        set_status(VoiceStatus::Disabled);

        ctx.check_refractory_period();

        // Status unchanged — not Error, so no transition.
        assert!(matches!(get_status(), VoiceStatus::Disabled));
        // Timer still cleared (the timer itself is session-level, not
        // status-dependent — always cleared when elapsed).
        assert!(ctx.refractory_until.is_none());
    }

    #[test]
    #[serial_test::serial(voice)]
    fn refractory_does_not_transition_while_recording() {
        let _ = init_global();

        let mut ctx = PipelineCtx::new();
        ctx.is_recording = true; // still recording
        ctx.refractory_until = Some(
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("1s in the past should not underflow"),
        );

        set_status(VoiceStatus::Error("test error".to_string()));

        ctx.check_refractory_period();

        // Still Error because is_recording is true.
        assert!(matches!(get_status(), VoiceStatus::Error(..)));
        assert!(ctx.refractory_until.is_none());
    }

    #[test]
    #[serial_test::serial(voice)]
    fn refractory_future_timer_does_not_transition() {
        let _ = init_global();

        let mut ctx = PipelineCtx::new();
        ctx.is_recording = false;
        ctx.refractory_until = Some(
            Instant::now()
                .checked_add(Duration::from_secs(60))
                .expect("60s in the future should not overflow"),
        );

        set_status(VoiceStatus::Error("test error".to_string()));

        ctx.check_refractory_period();

        // Timer hasn't elapsed yet — still Error and timer preserved.
        assert!(matches!(get_status(), VoiceStatus::Error(..)));
        assert!(ctx.refractory_until.is_some());
    }

    // ── Rate-limiting debounce tests ─────────────────────────────────────
    // These test the 10-second error-message rate limit via the canonical
    // [`PipelineCtx::should_send_error_message`] method.  No serial marker
    // needed — these only read from [`PipelineCtx`] fields without touching
    // global voice state.

    #[test]
    fn rate_limit_no_prior_error_allows_message() {
        let ctx = PipelineCtx::new();
        // last_error_message_time is None → should always send.
        assert!(ctx.should_send_error_message());
    }

    #[test]
    fn rate_limit_recent_error_suppresses_message() {
        let mut ctx = PipelineCtx::new();
        ctx.last_error_message_time = Some(Instant::now());
        // Just sent one → should suppress (< 10s elapsed).
        assert!(!ctx.should_send_error_message());
    }

    #[test]
    fn rate_limit_old_error_allows_message() {
        let mut ctx = PipelineCtx::new();
        ctx.last_error_message_time = Some(
            Instant::now()
                .checked_sub(Duration::from_secs(15))
                .expect("15s in the past should not underflow"),
        );
        // 15s > 10s threshold → should send.
        assert!(ctx.should_send_error_message());
    }

    #[test]
    fn rate_limit_exact_threshold_allows_message() {
        let mut ctx = PipelineCtx::new();
        ctx.last_error_message_time = Some(
            Instant::now()
                .checked_sub(Duration::from_secs(10))
                .expect("10s in the past should not underflow"),
        );
        // ≥ 10s is the threshold — exactly at the boundary should send.
        assert!(ctx.should_send_error_message());
    }

    #[test]
    fn rate_limit_just_below_threshold_suppresses_message() {
        let mut ctx = PipelineCtx::new();
        ctx.last_error_message_time = Some(
            Instant::now()
                .checked_sub(Duration::from_secs(9))
                .expect("9s in the past should not underflow"),
        );
        // 9s < 10s threshold → should suppress.
        assert!(!ctx.should_send_error_message());
    }

    // ── reset_pipeline_state level tests (mahbot-805) ─────────────────────
    // These test the three ResetLevel variants against a PipelineCtx with
    // non-default field values.  Tests that touch global voice state (Full,
    // Cancel) use #[serial_test::serial(voice)].
    //
    // The audio_preprocessor is tested indirectly: Full calls .reset() (new
    // NoiseSuppressor), Soft and Cancel call .clear_buffer() (preserves NS
    // adaptive noise profile).  These distinctions are not observable through
    // PipelineCtx's public API — they rely on the internal NoiseSuppressor
    // instance being recreated (reset) or kept (clear_buffer).

    /// Helper: build a PipelineCtx with non-default values in all mutable
    /// fields that reset_pipeline_state may touch.
    fn ctx_with_populated_buffers() -> PipelineCtx {
        let mut ctx = PipelineCtx::new();
        ctx.voice_batch = vec![0.5; 100];
        ctx.mel_frame_buffer = vec![vec![0.5; 32]; 10];
        ctx.embedding_ring = vec![vec![0.5; 96]; 3];
        ctx.audio_buffer = vec![0.5; 100];
        ctx.command_buffer = vec![0.5; 100];
        ctx.silence_sample_count = 1000;
        ctx.score_window = vec![0.5; 5];
        ctx.pre_agc_ring = vec![0.5; 100];
        ctx.negative_audio_buf = vec![0.5; 50];
        ctx.frame_vad = vec![true; 3];
        ctx.frame_raw_audio = vec![0.5; 200];
        ctx.emitted_utterances = 2;
        ctx.utterance_had_speech = true;
        ctx.utterance_silence_samples = 500;
        ctx.enrollment_no_speech_frame_count = 3;
        ctx.vad_positives_in_a_row = 5;
        ctx.enrollment_pending.push_back(vec![0.5; 50]);
        ctx.noise_rms_estimate = Some(0.1);
        ctx.vad_threshold = 0.75;
        ctx.last_wake_word_detection = Some(Instant::now() - Duration::from_secs(5));
        ctx.auto_start_pending = true;
        ctx.is_recording = true;
        ctx
    }

    /// Assert that all audio accumulators and enrollment-state fields are
    /// cleared — the common post-reset invariant shared by all level variants.
    fn assert_buffers_cleared(ctx: &PipelineCtx) {
        // Audio accumulators.
        assert!(ctx.voice_batch.is_empty());
        assert!(ctx.mel_frame_buffer.is_empty());
        assert!(ctx.embedding_ring.is_empty());
        assert!(ctx.audio_buffer.is_empty());
        assert!(ctx.command_buffer.is_empty());
        assert_eq!(ctx.silence_sample_count, 0);
        assert!(ctx.score_window.is_empty());
        assert!(ctx.pre_agc_ring.is_empty());
        assert!(ctx.negative_audio_buf.is_empty());

        // Enrollment VAD accumulation state.
        assert!(ctx.frame_vad.is_empty());
        assert!(ctx.frame_raw_audio.is_empty());
        assert_eq!(ctx.emitted_utterances, 0);

        // Enrollment detection/accumulator state.
        assert!(!ctx.utterance_had_speech);
        assert_eq!(ctx.utterance_silence_samples, 0);
        assert_eq!(ctx.enrollment_no_speech_frame_count, 0);
        assert_eq!(ctx.vad_positives_in_a_row, 0);
        assert!(ctx.enrollment_pending.is_empty());
        assert!(ctx.noise_rms_estimate.is_none());
    }

    #[test]
    #[serial_test::serial(voice)]
    fn reset_full_clears_all_buffers_and_state() {
        let _ = init_global();
        let mut ctx = ctx_with_populated_buffers();

        // Pre-populate global enrollment state.
        {
            let mut state = voice_state().write().unwrap_poison();
            state.enrollment_buffer.push(vec![vec![0.5; 96]]);
            state.negative_audio_chunks.push(vec![0.5; 100]);
        }

        ctx.reset_pipeline_state(ResetLevel::Full);

        assert_buffers_cleared(&ctx);

        // Full-specific: state flags reset.
        assert_eq!(ctx.vad_threshold, VAD_THRESHOLD);
        assert!(ctx.last_wake_word_detection.is_none());
        assert!(!ctx.auto_start_pending);
        assert!(!ctx.is_recording);

        // Global enrollment accumulators PRESERVED by Full — they survive
        // mic stop/start cycles so mid-enrollment progress is not lost on
        // toggle-off/on (mahbot-800, mahbot-819).
        let state = voice_state().read().unwrap_poison();
        assert_eq!(state.enrollment_buffer.len(), 1);
        assert_eq!(state.negative_audio_chunks.len(), 1);
    }

    #[test]
    #[serial_test::serial(voice)]
    fn reset_full_preserves_handler_managed_flags() {
        let _ = init_global();
        let mut ctx = ctx_with_populated_buffers();

        // Full should NOT touch these — they are owned by handler functions.
        ctx.is_listening = true;
        ctx.enrollment_mode = true;

        ctx.reset_pipeline_state(ResetLevel::Full);

        assert!(ctx.is_listening);
        assert!(ctx.enrollment_mode);
    }

    #[test]
    #[serial_test::serial(voice)]
    fn reset_soft_preserves_vad_threshold_cooldown_and_flags() {
        let _ = init_global();
        let mut ctx = ctx_with_populated_buffers();

        // Pre-populate global enrollment state so we can verify it's preserved.
        let saved_buffer = vec![vec![vec![0.5; 96]]]; // one utterance with one frame
        let saved_chunks = vec![vec![0.5; 100]];
        {
            let mut state = voice_state().write().unwrap_poison();
            state.enrollment_buffer = saved_buffer.clone();
            state.negative_audio_chunks = saved_chunks.clone();
        }

        let saved_threshold = ctx.vad_threshold; // 0.75
        let saved_cooldown = ctx.last_wake_word_detection;
        let saved_auto_start = ctx.auto_start_pending;
        let saved_recording = ctx.is_recording;

        ctx.reset_pipeline_state(ResetLevel::Soft);

        assert_buffers_cleared(&ctx);

        // Soft preserves these.
        assert_eq!(ctx.vad_threshold, saved_threshold);
        assert_eq!(ctx.last_wake_word_detection, saved_cooldown);
        assert_eq!(ctx.auto_start_pending, saved_auto_start);
        assert_eq!(ctx.is_recording, saved_recording);

        // Global enrollment accumulators preserved.
        let state = voice_state().read().unwrap_poison();
        assert_eq!(state.enrollment_buffer, saved_buffer);
        assert_eq!(state.negative_audio_chunks, saved_chunks);
    }

    #[test]
    #[serial_test::serial(voice)]
    fn reset_cancel_clears_enrollment_and_vad_threshold() {
        let _ = init_global();
        let mut ctx = ctx_with_populated_buffers();

        // Pre-populate global enrollment state so we can verify it's cleared.
        {
            let mut state = voice_state().write().unwrap_poison();
            state.enrollment_buffer.push(vec![vec![0.5; 96]]);
            state.negative_audio_chunks.push(vec![0.5; 100]);
        }

        let saved_auto_start = ctx.auto_start_pending;
        let saved_recording = ctx.is_recording;

        ctx.reset_pipeline_state(ResetLevel::Cancel);

        assert_buffers_cleared(&ctx);

        // Cancel clears vad_threshold and last_wake_word_detection.
        assert_eq!(ctx.vad_threshold, VAD_THRESHOLD);
        assert!(ctx.last_wake_word_detection.is_none());

        // Cancel preserves handler-managed flags (unlike Full).
        assert_eq!(ctx.auto_start_pending, saved_auto_start);
        assert_eq!(ctx.is_recording, saved_recording);

        // Global enrollment accumulators cleared (unlike Soft).
        let state = voice_state().read().unwrap_poison();
        assert!(state.enrollment_buffer.is_empty());
        assert!(state.negative_audio_chunks.is_empty());
    }

    #[test]
    #[serial_test::serial(voice)]
    fn reset_levels_preserve_session_ux_state() {
        let _ = init_global();
        // Session-level UX state (refractory_until, last_error_message_time)
        // must survive all reset levels — no level touches them.
        for level in [ResetLevel::Soft, ResetLevel::Full, ResetLevel::Cancel] {
            let mut ctx = PipelineCtx::new();
            ctx.refractory_until = Some(Instant::now());
            ctx.last_error_message_time = Some(Instant::now());

            ctx.reset_pipeline_state(level);

            assert!(
                ctx.refractory_until.is_some(),
                "refractory_until lost at {level:?}"
            );
            assert!(
                ctx.last_error_message_time.is_some(),
                "last_error_message_time lost at {level:?}"
            );
        }
    }
}
