//! Voice Assistant — wake word detection and voice command pipeline.
//!
//! # Architecture
//!
//! The voice assistant provides hands-free interaction via a custom wake word.
//! The pipeline stages are:
//!
//! 1. **Microphone capture** — capture mono 16 kHz audio via cpal
//! 2. **Voice activity detection** — Earshot neural VAD to gate processing
//! 3. **Encoder window embedding** — the trailing ≤1 s window is run through
//!    the shared Qwen3-ASR encoder's mel frontend + transformer
//!    ([`crate::audio::wake_word::encode_window`]), producing a 1024-dim
//!    L2-normalized window embedding
//! 4. **Wake word matching** — cosine similarity of the window embedding
//!    against the enrolled prototype, mapped through negative-sample
//!    calibration into a soft score, with the rolling-sum detection machinery
//!    (speaker-blind, immediate-fire).
//! 5. **Command recording** — record speech until silence or 10 min cap
//! 6. **Transcription** — via the shared Qwen3-ASR local transcriber
//! 7. **Routing** — transcribed text is routed to the user's active role via
//!    [`route_to_agent`] (falls back to the Manager if no active user is determined).
//!
//! The Assistant role manages this pipeline. It does NOT use an LLM agent loop.
//! Transcribed commands are routed to the user's currently active role (resolved
//! via [`route_to_agent`]) as if the user typed them.
//!
//! # Model sharing
//!
//! Wake word detection reuses the exact same loaded Qwen3-ASR model instance
//! as speech-to-text transcription — no separate artifact, no download
//! machinery, no model handling.  Wake word detection functions only when ASR
//! is enabled (`audio_transcription_use_local != "false"`); if transcription
//! is disabled, wake word is disabled too.
//!
//! # Scoring geometry
//!
//! Every scoring step encodes the trailing [`WINDOW_SAMPLES`] (12 160
//! samples ≈ 0.76 s) of the VAD-gated speech window through the shared
//! encoder.  The stride between scoring steps is [`SCORE_STRIDE_SAMPLES`]
//! (2560 samples ≈ 160 ms), derived from [`SCORE_STRIDE_MEL_FRAMES`] (16 mel
//! frames × 160-sample mel stride).  The raw audio ring is capped at
//! [`AUDIO_BUFFER_MAX`] (~2.0 s) so the VAD loop and the pre-wake recording
//! handoff always see a bounded trailing window.
//!
//! # Enrollment
//!
//! ~10 utterances are each embedded via [`crate::audio::wake_word::encode_window`];
//! the prototype is the L2-normalized mean of the accepted utterances
//! (no trainable head).  Negative-sample calibration (owner negatives + ambient
//! negatives) sets the cosine floor for scoring.  Persistence uses the v2
//! enrollment schema under the `wake_word_templates` config key; old v1 records
//! are rejected (re-enrollment required).

use crate::ChatDirection;
use crate::audio::wake_word::{
    self, ENROLLMENT_CONSISTENCY_MIN_FRACTION, ENROLLMENT_CONSISTENCY_MIN_SIMILARITY,
    MIN_ENROLLMENT_UTTERANCES, WAKE_WORD_EMBEDDING_DIM, WINDOW_SAMPLES, WakeWordEnrollment,
    calibrate_negatives, encode_window,
};
use crate::config::CONFIG;
use crate::turso;
use crate::util::UnwrapPoison;
use crate::util::hex_string;
use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

// ── E2E integration test (voice-tests feature) ──────────────────────────
#[cfg(feature = "voice-tests")]
#[path = "voice_pipeline_e2e_test.rs"]
pub(crate) mod voice_pipeline_e2e_test;

/// Public entry point for the voice pipeline benchmark.
///
/// Called by `benches/voice_pipeline_e2e.rs`.  Only compiled when the
/// `voice-tests` feature is enabled.
#[cfg(feature = "voice-tests")]
pub fn run_voice_pipeline_benchmark() {
    voice_pipeline_e2e_test::run_internal();
}

// Constants

/// Target sample rate: 16 kHz mono.
pub const SAMPLE_RATE: u32 = 16_000;

/// Convert a sample count to milliseconds (truncating integer division —
/// callers rely on exact threshold semantics).
#[must_use]
fn samples_to_ms(len: usize, rate: u32) -> u64 {
    (len as u64 * 1000) / u64::from(rate)
}

/// Frame size for VAD / quality frames (512 samples = 32ms at 16kHz).
pub(crate) const FRAME_LENGTH: usize = 512;

/// Hop length between frames (256 samples at 16 kHz).  This constant controls
/// VAD frame iteration stride and silence tracking in the application code.
/// The Qwen3-ASR mel frontend uses its own internal stride (160 samples =
/// 10ms) — HOP_LENGTH does NOT affect mel frame spacing.
pub(crate) const HOP_LENGTH: usize = 256;

/// Maximum command recording duration (10 minutes).
const MAX_RECORD_SECS: usize = 600;

/// Minimum silence duration before stopping command recording.
pub(crate) const SILENCE_DURATION: Duration = Duration::from_millis(1500);

/// Silence threshold in audio samples at 16 kHz.
/// Derived from SILENCE_DURATION × SAMPLE_RATE to prevent silent drift
/// if either constant changes.
pub(crate) const SILENCE_THRESHOLD_SAMPLES: usize =
    (SILENCE_DURATION.as_millis() as usize * SAMPLE_RATE as usize) / 1000;

/// Enrollment/segmentation silence threshold (~304ms = 19 hops × 256 samples).
/// Aligned to streaming detection's [`SEGMENT_TIMEOUT_HOPS`] so that utterance
/// boundaries are detected consistently between training and inference.
///
/// The longer [`SILENCE_THRESHOLD_SAMPLES`] is preserved for command recording
/// (`handle_recording_audio`) where 1.5s is appropriate for natural command
/// phrasing pauses.
pub(crate) const ENROLLMENT_SILENCE_THRESHOLD_SAMPLES: usize = SEGMENT_TIMEOUT_HOPS * HOP_LENGTH;

/// Silence threshold (200ms) before showing "Keep silent to confirm…" UI hint.
/// Intentionally wider than a single frame (16ms) so the UI reliably transitions
/// even under scheduling jitter.
const SILENCE_UI_GATE_SAMPLES: usize = 200 * SAMPLE_RATE as usize / 1000;

/// Capacity of the microphone audio channel feeding the voice pipeline.
///
/// 32 chunks × 512 samples / 16000 Hz ≈ 1 second of audio.  When the pipeline
/// is blocked (e.g. encoder inference in [`handle_wake_word_detection`]),
/// [`try_send`](tokio::sync::mpsc::Sender::try_send) silently drops chunks
/// at this threshold, preventing unbounded memory growth.
///
/// # Drop policy
///
/// This is **drop-newest**: the most recent audio chunk is discarded when the
/// channel is full.  During a pipeline stall the buffered audio is slightly
/// delayed (~1 s) but temporally contiguous, so downstream processing (VAD,
/// window encoding, wake-word scoring) operates on consistent stream
/// segments.  The wake word may be missed if it arrives entirely within the
/// dropped window, but the user will simply repeat it.
const MIC_CHANNEL_CAPACITY: usize = 32;

/// Duration of non-VAD audio before showing "speak louder" warning during
/// enrollment (~5s).  Derived from SAMPLE_RATE and HOP_LENGTH so the threshold
/// stays correct if frame/hop sizes are adjusted.
const ENROLLMENT_NO_SPEECH_DURATION: Duration = Duration::from_secs(5);

/// Consecutive non-VAD frame threshold for enrollment no-speech warning.
/// Each frame iteration processes HOP_LENGTH new samples, so the threshold
/// is (duration × sample_rate) / hop_length to be robust against frame/hop
/// size changes.
const ENROLLMENT_NO_SPEECH_TIMEOUT_FRAMES: usize =
    (ENROLLMENT_NO_SPEECH_DURATION.as_millis() as usize * SAMPLE_RATE as usize)
        / (HOP_LENGTH * 1000);

/// Default wake word phrase used when no phrase has been specified.
pub(crate) const DEFAULT_WAKE_WORD_PHRASE: &str = "mahbot";

/// Consecutive VAD-negative hops before a detection segment boundary fires
/// (~300 ms at 16 ms/hop).  The rolling score window and adaptive threshold
/// are reset at the boundary so scores cannot accumulate across separate
/// utterances.
const SEGMENT_TIMEOUT_HOPS: usize = 19;

/// Target number of user utterances for enrollment (Phase 2).
const NUM_ENROLLMENT_SAMPLES: usize = 10;

/// Minimum length of a saved ambient negative audio chunk (0.5 s).
const MIN_NEGATIVE_AUDIO_LEN: usize = SAMPLE_RATE as usize / 2;

/// Maximum number of ambient negative audio chunks retained in memory.
const MAX_NEGATIVE_AUDIO_CHUNKS: usize = 100;

/// Maximum automatic retry cycles after a terminal ASR load failure.
///
/// Each cycle is a full load-or-download chain (up to 12 download attempts ×
/// 1.88 GB in the transcriber's [`crate::audio::local_transcriber`] retry
/// loop).  The periodic pipeline self-healing path is bounded to this budget
/// so a persistently failing download cannot loop forever (retry → 12
/// download attempts → Failed → 30 s later retry → …).  Beyond the budget the
/// pipeline stops auto-retrying: the user must either press the GUI retry
/// button ([`VoiceCommand::RetryModelLoading`], which bypasses the budget) or
/// restart the app.
const MAX_AUTO_MODEL_RETRY_CYCLES: u32 = 3;

// ── Phase 3 (owner-negative) enrollment constants ────────────

/// Target VAD-positive speech seconds for Phase 3 owner-negative collection.
///
/// 15 s of real VAD-positive speech is sufficient to collect calibration data
/// and anti-prototypes while keeping the per-user negative phase short.  The
/// phase still collects owner-negative chunks for calibration and
/// anti-prototypes as designed.  The wall-clock timeout ([`PHASE3_TIMEOUT_SECS`])
/// is unchanged.
const NEGATIVES_TARGET_SECONDS: usize = 15;

/// Memory cap for owner-negative chunks (~22.5 s of 16 kHz f32 audio).
/// Auto-derived from [`NEGATIVES_TARGET_SECONDS`] (× 1.5 headroom).
const MAX_OWNER_NEGATIVE_SAMPLES: usize = SAMPLE_RATE as usize * NEGATIVES_TARGET_SECONDS * 3 / 2;

/// Wall-clock timeout for Phase 3 owner-negative collection.
const PHASE3_TIMEOUT_SECS: u64 = 120;

/// Clipping detection threshold for enrollment quality scoring.
const ENROLLMENT_QUALITY_CLIPPING_THRESHOLD: f32 = 0.999;

/// Minimum acceptable enrollment utterance duration in milliseconds.
pub(crate) const ENROLLMENT_QUALITY_DURATION_MIN_MS: u64 = 400;

/// Maximum acceptable enrollment utterance duration in milliseconds.
pub(crate) const ENROLLMENT_QUALITY_DURATION_MAX_MS: u64 = 2000;

/// Minimum fraction of enrollment utterances that must pass the enrollment
/// self-test for the model to be accepted.
const ENROLLMENT_QUALITY_SELF_TEST_MIN_FRACTION: f32 = 0.8;

/// Per-utterance enrollment prompts (guidance shown to the user as they
/// record the required samples).
const ENROLLMENT_PROMPTS: &[(&str, usize)] = &[
    ("Say it normally", 3),
    ("Say it a bit further from the mic", 3),
    ("Say it at a slightly different angle", 2),
    ("Say it with your normal morning voice", 2),
];

/// Maximum size of the raw audio ring buffer (~200ms at 16kHz = 3200
/// samples).  Used during enrollment to capture ~100ms of pre-VAD-trigger
/// and post-speech context so the template includes the onset/offset
/// phonemes that the enrollment VAD threshold excludes.
pub(crate) const RAW_RING_MAX: usize = SAMPLE_RATE as usize / 5;

// ── Wake-word scoring geometry (re-calibrated for the encoder pipeline) ──
//
// The scoring constants below were re-calibrated for the 1024-dim cosine
// soft-score space produced by [`crate::audio::wake_word::encode_window`] +
// [`Calibration::soft_score`] (raw cosine mapped through the negative-sample
// floor into [0,1]).  The old pipeline scored a sigmoid Conv1D head whose
// per-frame outputs were well above the cosine soft scores a calibrated
// enrollment produces; the thresholds were re-derived for this space.

/// Bounded detection window: the trailing 0.76 s (12 160 samples) of the
/// VAD-gated speech window — the same [`WINDOW_SAMPLES`] geometry enrollment
/// uses ([`crate::audio::wake_word::encode_window`]).
pub(crate) const WAKE_WORD_WINDOW_SAMPLES: usize = WINDOW_SAMPLES;

/// Scoring stride in samples: 16 mel frames × 160-sample mel stride = 2560
/// samples ≈ 160 ms.  The encoder forward is heavy, so the stride is widened
/// to keep the real-time budget sane while the rolling window (N=3) still
/// spans ~0.5 s of speech.
const SCORE_STRIDE_SAMPLES: usize = crate::audio::wake_word::SCORE_STRIDE_MEL_FRAMES * 160;

/// Cap for the raw audio ring (~2.0 s): one full [`WINDOW_SAMPLES`] window
/// (12 160 samples) plus up to eight scoring strides of context
/// (8 × 2560 = 20 480) so the VAD loop and pre-wake recording handoff always
/// have the most recent audio while older samples drain.
const AUDIO_BUFFER_MAX: usize = WINDOW_SAMPLES + 16 * 160 * 8;

/// Cooldown after a wake word detection — prevents rapid consecutive false
/// triggers.
const WAKE_WORD_COOLDOWN: Duration = Duration::from_secs(3);

/// Below this soft score a frame is treated as background noise: the rolling
/// window is cleared to prevent slow accumulation from noise.
/// Calibrated for the p99-floored soft-score space: the floor (0.75) maps the
/// confusable near-miss tail to ~0, so 0.35 sits above the floor-mapped noise
/// band while staying below genuine-match soft scores.
const NO_MATCH_RESET_THRESHOLD: f32 = 0.35;

// Voice pipeline metrics — always-on atomic counters

/// Total audio chunks received from the microphone channel by the pipeline
/// loop (incremented after each successful `rx.recv()`).
pub(crate) static CHUNKS_RECEIVED: AtomicU64 = AtomicU64::new(0);

/// Audio chunks dropped at the mic channel boundary (`try_send` failure).
/// These chunks were discarded because the pipeline was not consuming fast
/// enough and the bounded channel (`MIC_CHANNEL_CAPACITY` = 32 chunks) was
/// full.  This is the primary indicator of the pipeline falling behind
/// real-time audio.
pub(crate) static DROPPED_CHUNKS: AtomicU64 = AtomicU64::new(0);

/// Total window embeddings computed during wake word processing.  Each
/// embedding corresponds to one bounded ≤1 s window encoding through the
/// shared Qwen3-ASR encoder ([`crate::audio::wake_word::encode_window`]).
/// Monotonically increasing — never reset.
pub(crate) static EMBEDDINGS_COMPUTED: AtomicU64 = AtomicU64::new(0);

/// Total wall-clock time spent computing window embeddings (nanoseconds).
/// Divide by [`EMBEDDINGS_COMPUTED`] for the lifetime average per-embedding
/// latency.  Tracked via `Instant::now()` around [`encode_window`].
pub(crate) static TOTAL_EMBEDDING_TIME_NS: AtomicU64 = AtomicU64::new(0);

// ── Rolling average ring buffer for embedding latency ────────────────────
//
// The design calls for "rolling average (last N frames)" of processing
// latency.  Rather than an exponential moving average (which is an
// approximation) or a lifetime cumulative average (which becomes diagnostically
// inert as embeddings accumulate), we use a lock-free ring buffer of the most
// recent 100 window-encoding times.  N=100 covers ~16 s of audio at the
// 160 ms scoring stride — large enough to smooth noise, small enough for
// O(100) reads on the diagnostic path.
//
// Lock-free: single writer (pipeline task) stores to the ring with an atomic
// head index; readers sum O(N) entries on the diagnostic/debug path only.
// No mutex is involved on any path.

/// Number of recent window-encoding latencies tracked in the rolling average
/// ring buffer.
const EMBEDDING_LATENCY_RING_SIZE: usize = 100;

/// Lock-free ring buffer of the most recent [`EMBEDDING_LATENCY_RING_SIZE`]
/// window-encoding times (nanoseconds).  Written by the pipeline task
/// (single writer), read by diagnostic/logging code.  Never cleared — wraps
/// around on overflow.
static EMBEDDING_LATENCY_RING: [AtomicU64; EMBEDDING_LATENCY_RING_SIZE] =
    [const { AtomicU64::new(0) }; EMBEDDING_LATENCY_RING_SIZE];

/// Total number of writes to [`EMBEDDING_LATENCY_RING`].  Monotonically
/// increasing — the number of valid entries in the ring is
/// `min(writes, EMBEDDING_LATENCY_RING_SIZE)`.
static EMBEDDING_LATENCY_RING_WRITES: AtomicU64 = AtomicU64::new(0);

/// Snapshot of voice pipeline metrics for diagnostics and logging.
///
/// Returned by [`get_voice_metrics()`].  All fields are atomically-sampled
/// Relaxed reads — not guaranteed to be mutually consistent across fields in
/// the presence of concurrent increments, but good enough for diagnostic use.
///
/// The `avg_embedding_latency_ns` field is computed from the lock-free ring
/// buffer of the last [`EMBEDDING_LATENCY_RING_SIZE`] window encodings,
/// providing a true rolling average that reflects recent pipeline performance.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VoiceMetricsSnapshot {
    /// Total audio chunks received from the mic channel.
    pub chunks_received: u64,
    /// Chunks dropped at the mic channel boundary (try_send full).
    pub dropped_chunks: u64,
    /// Total window embeddings computed.
    pub embeddings_computed: u64,
    /// Cumulative window-encoding time (nanoseconds) — for lifetime
    /// average via [`Self::lifetime_avg_embedding_latency_ns`].
    pub total_embedding_time_ns: u64,
    /// Rolling average window-encoding latency in nanoseconds (last 100).
    /// Prefer this over the lifetime average for detecting recent performance
    /// changes — the lifetime average becomes diagnostically inert as the
    /// pipeline accumulates millions of embeddings.
    pub avg_embedding_latency_ns: u64,
}

impl VoiceMetricsSnapshot {
    /// Lifetime average window-encoding latency in nanoseconds (total ÷ count).
    ///
    /// Useful for overall diagnostic ("has the model gotten slower since
    /// startup?"), but becomes insensitive to recent changes after many
    /// embeddings.  Use the rolling average
    /// [`avg_embedding_latency_ns`](Self::avg_embedding_latency_ns) for
    /// detecting recent performance shifts.
    ///
    /// Returns 0 when no embeddings have been computed.
    #[must_use]
    pub fn lifetime_avg_embedding_latency_ns(&self) -> u64 {
        self.total_embedding_time_ns
            .checked_div(self.embeddings_computed)
            .unwrap_or(0)
    }

    /// Fraction of chunks dropped at the mic channel boundary (0.0 – 1.0).
    /// Returns 0.0 when no chunks have been received.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn drop_rate(&self) -> f64 {
        let total = self.chunks_received + self.dropped_chunks;
        if total == 0 {
            0.0
        } else {
            self.dropped_chunks as f64 / total as f64
        }
    }
}

/// Read a consistent snapshot of all voice pipeline metrics.
///
/// The `avg_embedding_latency_ns` field is computed by summing the lock-free
/// ring buffer entries — O(100) relaxed atomic loads, negligible for any
/// diagnostic path.
pub(crate) fn get_voice_metrics() -> VoiceMetricsSnapshot {
    // Compute the rolling average from the lock-free ring buffer.
    // Both truncation sites (u64 → usize for indexing, usize → u64 for
    // division) are safe: the ring is 100 entries, and writes won't overflow
    // usize on any target within the pipeline's lifetime.
    #[allow(clippy::cast_possible_truncation)]
    let valid = usize::try_from(EMBEDDING_LATENCY_RING_WRITES.load(Ordering::Relaxed))
        .map_or(0, |w| w.min(EMBEDDING_LATENCY_RING_SIZE));
    let avg_ns = if valid > 0 {
        #[allow(clippy::needless_range_loop)]
        let sum: u64 = EMBEDDING_LATENCY_RING[..valid]
            .iter()
            .map(|a| a.load(Ordering::Relaxed))
            .sum();
        #[allow(clippy::cast_possible_truncation)]
        {
            sum / valid as u64
        }
    } else {
        0
    };

    VoiceMetricsSnapshot {
        chunks_received: CHUNKS_RECEIVED.load(Ordering::Relaxed),
        dropped_chunks: DROPPED_CHUNKS.load(Ordering::Relaxed),
        embeddings_computed: EMBEDDINGS_COMPUTED.load(Ordering::Relaxed),
        total_embedding_time_ns: TOTAL_EMBEDDING_TIME_NS.load(Ordering::Relaxed),
        avg_embedding_latency_ns: avg_ns,
    }
}

/// A single embedding decision recorded in the activation trace.
/// Each frame represents ~160 ms of audio (one [`SCORE_STRIDE_SAMPLES`]
/// stride), so N=3 covers ~480 ms — matching the original temporal window
/// but using accumulated weight instead of a strict consecutive binary
/// counter.
const ROLLING_WINDOW_N: usize = 3;

/// Factor applied to `ROLLING_WINDOW_N` to compute the detection threshold.
/// At 0.55 (threshold 1.65), calibrated for the 1024-dim cosine soft-score
/// space with the p99 negative-sample floor (0.75).  The floor maps the
/// confusable near-miss tail to ~0, so the remaining positive margin
/// (enrollment utterances at cosine 0.94+ → soft ~0.76) sits above
/// 3 × 0.55 = 1.65.  (Was 0.65 / 1.95 for the un-floored soft scores; the
/// floor does the discrimination now, so the threshold tracks the compressed
/// positive range.)
const MATCH_THRESHOLD_FACTOR: f32 = 0.55;

/// Detection threshold for the rolling sum of soft scores.
/// Computed as: `ROLLING_WINDOW_N × MATCH_THRESHOLD_FACTOR`
/// (= `3 × 0.55 = 1.65`).  Calibrated for the cosine soft-score space with
/// the p99 negative floor.
#[expect(clippy::cast_precision_loss)]
const fn match_threshold() -> f32 {
    (ROLLING_WINDOW_N as f32) * MATCH_THRESHOLD_FACTOR
}

// ── Adaptive threshold ──────────────────────────────────────

/// Number of recent per-frame scores to track for adaptive threshold
/// statistics. At ~160 ms per frame, N=15 covers ~2.4 seconds of context.
const ADAPTIVE_WINDOW_N: usize = 15;

/// Default k multiplier for the adaptive threshold (mean + k × std).
const ADAPTIVE_K_DEFAULT: f32 = 2.5;

/// Minimum allowed adaptive k value (user-configurable range).
const ADAPTIVE_K_MIN: f32 = 1.0;

/// Maximum allowed adaptive k value (user-configurable range).
const ADAPTIVE_K_MAX: f32 = 4.0;

/// Absolute ceiling — the adaptive threshold must never exceed this value.
/// Re-calibrated for the cosine soft-score space with the p99 floor (0.75):
/// the max rolling sum is 3.0 (3 × 1.0), and the ceiling must sit above the
/// floored-positive range (~2.5 at cosine 0.96 → soft 0.84 × 3).  2.60 leaves
/// headroom for a genuine wake while capping threshold drift from noise.
const ADAPTIVE_CEILING: f32 = 2.60;

/// Safe harbor floor for the adaptive threshold, equal to [`match_threshold()`]
/// (1.65); prevents a feedback loop where false accepts push the threshold
/// lower.
const ADAPTIVE_SAFE_HARBOR: f32 = match_threshold();

/// Number of bootstrap frames to use the static threshold while the adaptive
/// window fills.
const ADAPTIVE_BOOTSTRAP_FRAMES: usize = 5;

/// Process a per-frame soft score through the rolling window and determine
/// whether wake word detection should fire.
///
/// Returns `true` when the rolling sum of recent scores meets or exceeds
/// `match_threshold()`.  When the incoming score is below
/// [`NO_MATCH_RESET_THRESHOLD`], the window is cleared entirely to prevent
/// slow accumulation from noise — unless `preserve_window_on_reset` is set.
/// On detection the score window is NOT cleared here — the caller is
/// responsible for full pipeline cleanup.
///
/// This function is pure with respect to global state: it only reads its
/// parameters and modifies `score_window` in place.  This makes it directly
/// testable without models or voice pipeline initialization.
fn process_wake_word_score(
    total_score: f32,
    score_window: &mut Vec<f32>,
    adaptive_threshold_override: Option<f32>,
    preserve_window_on_reset: bool,
) -> (bool, f32) {
    if total_score < NO_MATCH_RESET_THRESHOLD {
        // Far from matching — reset the entire rolling window to prevent
        // slow accumulation from noise.
        if !preserve_window_on_reset {
            if !score_window.is_empty() {
                debug!(
                    "Wake word match lost: total_score={total_score:.4} < NO_MATCH_RESET_THRESHOLD \
                     (window reset, had {} scores)",
                    score_window.len(),
                );
            }
            score_window.clear();
        }
        (false, 0.0)
    } else {
        // Good-enough frame: append score to rolling window.
        score_window.push(total_score);
        // Keep window at most ROLLING_WINDOW_N frames.
        while score_window.len() > ROLLING_WINDOW_N {
            score_window.remove(0);
        }

        let rolling_sum: f32 = score_window.iter().sum();
        let threshold = adaptive_threshold_override.unwrap_or_else(match_threshold);

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
            (true, rolling_sum)
        } else {
            (false, rolling_sum)
        }
    }
}

/// Process a single window embedding through the wake word detection pipeline.
///
/// This is the **core detection step** shared between the live pipeline
/// ([`handle_wake_word_detection`]) and the enrollment self-test
/// ([`run_enrollment_self_test`]).
///
/// It scores the embedding as the cosine soft score against the enrolled
/// prototype ([`WakeWordEnrollment::soft_score`]), feeds/peeks the adaptive
/// threshold, and applies rolling window scoring via
/// [`process_wake_word_score`].  Detection fires immediately when the rolling
/// sum crosses the effective threshold — the pipeline is speaker-blind with
/// no second-stage gate.
///
/// # Returns
/// - `(true, rolling_sum, total_score, effective_threshold)` — the embedding
///   triggered wake word detection.
/// - `(false, _, total_score, effective_threshold)` — continue feeding more
///   embeddings (the score window is updated for the next call).
///
/// - `rolling_sum` — the rolling window sum at the time of evaluation
///   (0.0 if the window was reset due to low score).
/// - `total_score` — the cosine soft score (0.0 when no enrollment is set).
/// - `effective_threshold` — the threshold value used for the rolling window
///   comparison this frame (adaptive threshold post-bootstrap, or static
///   [`match_threshold()`] during bootstrap / when no adaptive state is
///   configured).
///
/// # Parameters
/// - `embedding` — one 1024-dim L2-normalized window embedding.
/// - `enrollment` — the enrolled prototype+calibration (`None` disables
///   detection — total_score stays 0.0).
/// - `score_window` — persistent rolling window of recent soft scores.
/// - `adaptive_state` — optional adaptive threshold state (`None` disables
///   adaptive threshold adjustment).
/// - `adaptive_k` — multiplier for the adaptive threshold's standard-deviation
///   term (passed to [`AdaptiveThresholdState::next_threshold`]).
pub(crate) fn score_single_embedding(
    embedding: &[f32],
    enrollment: Option<&WakeWordEnrollment>,
    score_window: &mut Vec<f32>,
    mut adaptive_state: Option<&mut AdaptiveThresholdState>,
    adaptive_k: f32,
) -> (bool, f32, f32, f32) {
    // ── Cosine soft score against the enrolled prototype ───────────────
    // No enrollment → total_score = 0.0 (no detection possible).
    let total_score = enrollment.map_or(0.0, |enr| enr.soft_score(embedding));

    // Feed only background (non-wake-word) scores to the adaptive threshold
    // so it learns the noise-floor distribution without being contaminated
    // by the high scores it's trying to detect.  Scores below
    // NO_MATCH_RESET_THRESHOLD are clearly "not wake word" and represent
    // the background acoustic environment.  For wake-word-like frames we
    // call peek() which returns the current threshold without updating
    // statistics, preventing the self-defeating loop where high scores
    // inflate the threshold and block detection.
    //
    // The same feed/peek rule applies during bootstrap — the old
    // unconditional bootstrap feed inflated the adaptive threshold with
    // wake-word-like burst scores, blocking detection after the deferred
    // burst.  With the score-only rule, a high-scoring utterance
    // legitimately keeps the bootstrap alive for its whole duration: high
    // scores peek (peek() returns None during bootstrap → the static
    // match_threshold stays in effect), and only below-reset background
    // frames feed and advance the bootstrap counter.  Residual
    // contamination cannot persist across the soft reset
    // (detection→recording handoff) because no wake-word-like score ever
    // enters the statistics.
    let adaptive_override = adaptive_state.as_mut().and_then(|state| {
        if total_score < NO_MATCH_RESET_THRESHOLD {
            state.feed(total_score, adaptive_k)
        } else {
            state.peek(adaptive_k)
        }
    });

    // ── Effective threshold (adaptive post-bootstrap, static otherwise) ──
    let effective_threshold = adaptive_override.unwrap_or_else(match_threshold);

    // ── Rolling window gate ─────────────────────────────
    let (detected, rolling_sum) =
        process_wake_word_score(total_score, score_window, adaptive_override, false);

    // ── Voice debug logging ─────────────────────────────
    // When the `voice-debug` feature is enabled, log every per-frame
    // total_score along with whether it passed the reset threshold and
    // the resulting rolling sum.  The feature gate ensures zero overhead
    // when not compiled in.
    #[cfg(feature = "voice-debug")]
    {
        let passed_threshold = total_score >= NO_MATCH_RESET_THRESHOLD;
        let below_note = if passed_threshold {
            ""
        } else {
            " (below NO_MATCH_RESET_THRESHOLD — window reset)"
        };
        info!("VOICE_DEBUG: total_score={total_score:.4}{below_note} rolling_sum={rolling_sum:.4}",);
    }

    // Immediate-fire: no second-stage gate — a threshold
    // crossing fires detection on this frame (speaker-blind pipeline).
    if detected {
        (true, rolling_sum, total_score, effective_threshold)
    } else {
        (false, rolling_sum, total_score, effective_threshold)
    }
}

/// VAD threshold: scores >= this are considered speech.
/// Unified to 0.5 for both enrollment and streaming detection.
const VAD_THRESHOLD: f32 = 0.5;

/// Minimum consecutive VAD-positive frames before setting utterance_had_speech
/// during enrollment (~0ms at 16ms/frame).  Set to 1 to match streaming
/// detection behavior, which starts accumulating at the first VAD-positive
/// frame.
pub(crate) const ENROLLMENT_VAD_CONSECUTIVE_REQUIRED: usize = 1;

// Neural VAD (Earshot) — replaces RMS-based `is_speech`

/// Global Earshot VAD detector instance. Thread-safe behind a mutex because
/// `predict_f32` completes in ~5-6 µs, so lock contention is negligible.
/// The detector has internal state (768-sample ring buffer, pre-emphasis filter,
/// 3-frame feature context) that must be kept in sync with the audio stream.
/// Created once in [`init_global`].
static VAD_DETECTOR: OnceLock<std::sync::Mutex<earshot::Detector>> = OnceLock::new();

/// Mirrors [`PipelineCtx::manual_recording`] for GUI access — the flag stays
/// set through the manual transcription window, letting the Home composer
/// distinguish a mic-button ASR from a wake-word one (both share
/// [`VoiceStatus::Transcribing`]).
static MANUAL_RECORDING_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Check whether wake word detection is ready for inference.
///
/// Wake word detection shares the Qwen3-ASR transcriber model, so "models
/// ready" means the local transcriber is loaded.
#[must_use]
pub fn models_ready() -> bool {
    crate::audio::local_transcriber::is_loaded()
}

/// Whether local audio transcription is disabled by config
/// (`audio_transcription_use_local == "false"`).
///
/// The voice pipeline (wake word AND mic-button recording) shares the local
/// ASR transcriber, so with transcription disabled the model never loads and
/// the whole voice stack is unavailable — callers must surface the
/// configuration rather than a loading state that can never complete.
#[must_use]
pub fn is_transcription_disabled() -> bool {
    crate::config::CONFIG
        .audio_transcription_use_local()
        .as_deref()
        == Some("false")
}

/// Resolve the voice status once the shared ASR transcriber's state has
/// settled.  Pure helper shared by the pipeline's startup block and its
/// periodic wake-up so the two can never drift apart:
///
/// - transcriber failed → [`VoiceStatus::ModelError`] (terminal);
/// - transcriber loaded → [`VoiceStatus::Listening`] when voice is enabled,
///   [`VoiceStatus::Disabled`] (indicator hidden) otherwise;
/// - still loading → [`VoiceStatus::LoadingModels`] (transient — the
///   periodic wake-up keeps polling until a terminal state is reached).
///
/// `loaded` and `failed` are mutually exclusive states of the transcriber's
/// atomic state machine; failure is still checked first as a defensive
/// priority so a failed model can never be reported as listening.
fn resolved_model_status(
    transcriber_loaded: bool,
    transcriber_failed: bool,
    voice_enabled: bool,
) -> VoiceStatus {
    if transcriber_failed {
        VoiceStatus::ModelError
    } else if transcriber_loaded {
        if voice_enabled {
            VoiceStatus::Listening
        } else {
            VoiceStatus::Disabled
        }
    } else {
        VoiceStatus::LoadingModels
    }
}

// Voice pipeline status (shared between pipeline task and GUI)

/// Voice pipeline status.
#[derive(Debug, Clone)]
pub enum VoiceStatus {
    Disabled,
    LoadingModels,
    ModelError,
    Listening,
    Recording,
    /// Mic-button-initiated recording from the Home composer. Wake-word
    /// listening is paused while this is active.
    RecordingManual,
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
    /// Collecting owner-general speech as negative examples
    /// (Phase 3 enrollment).  `accumulated_secs` is the
    /// VAD-positive speech time collected so far, `target_secs` is the
    /// target (15 s of VAD-positive speech), `wall_clock_elapsed` is
    /// the wall-clock seconds since Phase 3 started.
    EnrollingNegatives {
        accumulated_secs: usize,
        target_secs: usize,
        wall_clock_elapsed: u64,
    },
    Enrolled,
    Error(String),
}

// Enrollment quality scoring

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

// Global state

static VOICE_PIPELINE: OnceLock<RwLock<VoicePipelineState>> = OnceLock::new();

/// The name of the currently active user, updated by the GUI when the
/// selected user changes. Used by [`route_to_agent`] to route transcribed
/// voice commands to the correct user's active role.
static LAST_ACTIVE_USER: OnceLock<RwLock<String>> = OnceLock::new();

/// Set the currently active user name (called from GUI on user switch).
pub fn set_active_user_name(name: &str) {
    if let Some(state) = LAST_ACTIVE_USER.get() {
        *state.write().unwrap_poison() = name.to_string();
    }
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
    /// Enrolled wake-word prototype + calibration (v2 schema).  `None` before
    /// a successful enrollment.
    enrollment: Option<WakeWordEnrollment>,
    /// Per-utterance 1024-dim L2-normalized window embeddings collected
    /// during enrollment (one entry per accepted utterance).
    enrollment_embeddings: Vec<Vec<f32>>,
    /// Raw audio chunks collected during non-wake-word periods of enrollment
    /// (pre-enrollment ambient noise and inter-utterance silence).  These are
    /// encoded at finalization time to produce real (non-synthetic) negative
    /// examples.
    negative_audio_chunks: Vec<Vec<f32>>,
    /// Owner-negative audio chunks collected during Phase 3 enrollment.
    /// ~15 seconds of VAD-positive general speech from the user,
    /// stored as audio chunks for embedding extraction at finalization.
    /// Preserved across Full/Soft pipeline resets (same as `negative_audio_chunks`)
    /// but cleared on Cancel (via `reset_enrollment`).
    owner_negative_chunks: Vec<Vec<f32>>,
    /// Whether the user has completed all 10 enrollment utterances (Phase 2
    /// done).  Set by `handle_enrollment_sample` when
    /// `enrolled_utterance_count >= NUM_ENROLLMENT_SAMPLES`.  The main loop
    /// reads this flag to initiate the Phase 2→3 transition
    /// (`transition_to_phase3`).  Cleared on Cancel (via `reset_enrollment`).
    utterances_collected: bool,
    /// Number of user utterances enrolled so far.  Incremented once per
    /// accepted `handle_enrollment_sample` call.  The UI counter and
    /// finalization trigger use this field.
    enrolled_utterance_count: usize,
    /// Cached wake word phrase from the last loaded / persisted enrollment.
    /// Never cleared on cancel.  Read by [`get_enrolled_phrase()`].
    model_phrase: Option<String>,
    /// Transient wake word phrase for an enrollment in progress.
    /// Set at enrollment start via [`normalize_phrase`], consumed by
    /// [`finalize_enrollment_pipeline`]. Never read by [`get_enrolled_phrase()`].
    enrolling_phrase: Option<String>,
    cmd_tx: Option<mpsc::UnboundedSender<VoiceCommand>>,
}

impl VoicePipelineState {
    /// Clear all enrollment accumulators (embeddings + utterance counter).
    ///
    /// Called by [`PipelineCtx::reset_pipeline_state`] on [`ResetLevel::Cancel`]
    /// and by tests to verify data model invariants.
    ///
    /// Clears the transient [`enrolling_phrase`] but preserves [`model_phrase`]
    /// (the cached phrase from the last loaded / persisted enrollment).
    fn reset_enrollment(&mut self) {
        self.enrollment_embeddings.clear();
        self.negative_audio_chunks.clear();
        self.owner_negative_chunks.clear();
        self.enrolled_utterance_count = 0;
        self.utterances_collected = false;
        self.enrolling_phrase = None;
    }
}

#[derive(Debug)]
pub enum VoiceCommand {
    StartListening,
    StopListening,
    /// Start wake word enrollment using the specified wake word phrase.
    /// The phrase is normalized (trimmed, lowercased, whitespace-collapsed)
    /// before being stored in the enrollment record.
    StartEnrollment(String),
    CancelEnrollment,
    /// Retry loading the shared Qwen3-ASR transcriber model after a terminal
    /// failure (the wake-word pipeline has no model machinery of its own).
    RetryModelLoading,
    /// Start a mic-button-initiated voice message recording (Home composer).
    /// Pauses wake-word listening for the duration of the recording.
    StartRecording,
    /// Stop the mic-button recording, transcribe it, and route the
    /// transcript to the user's active role agent.
    StopRecordingSend,
    /// Stop the mic-button recording and discard the captured audio.
    StopRecordingDiscard,
    Shutdown,
}

fn voice_state() -> &'static RwLock<VoicePipelineState> {
    VOICE_PIPELINE.get().expect("VoicePipeline not initialized")
}

/// Initialize the voice pipeline state. Called during startup.
pub fn init_global() -> Result<()> {
    VAD_DETECTOR.get_or_init(|| std::sync::Mutex::new(earshot::Detector::default()));

    LAST_ACTIVE_USER
        .set(RwLock::new(String::new()))
        .map_err(|_| anyhow!("LAST_ACTIVE_USER already initialized"))?;

    VOICE_PIPELINE
        .set(RwLock::new(VoicePipelineState {
            enabled: false,
            status: VoiceStatus::Disabled,
            enrollment: None,
            enrollment_embeddings: Vec::new(),
            negative_audio_chunks: Vec::new(),
            owner_negative_chunks: Vec::new(),
            enrolled_utterance_count: 0,
            utterances_collected: false,
            model_phrase: None,
            enrolling_phrase: None,
            cmd_tx: None,
        }))
        .map_err(|_| anyhow!("VoicePipeline already initialized"))?;

    Ok(())
}

#[must_use]
pub fn get_status() -> VoiceStatus {
    voice_state().read().unwrap_poison().status.clone()
}

/// Whether a mic-button (manual) recording is active — including its
/// transcription window, where the shared [`VoiceStatus::Transcribing`] is
/// ambiguous between the manual and wake-word paths.
#[must_use]
pub fn is_manual_recording() -> bool {
    MANUAL_RECORDING_ACTIVE.load(Ordering::Relaxed)
}

/// Best-effort reason a mic-button recording cannot start right now
/// (None = allowed). Mirrors the field-based guards in
/// [`PipelineCtx::handle_start_manual_recording`] for the GUI's pre-flight
/// toast; the pipeline remains the authoritative check and this mapping can
/// drift on transient transitions.  The transcription-disabled configuration
/// is permanent rather than transient, so it is checked here directly.
#[must_use]
pub fn manual_recording_blocked_reason() -> Option<&'static str> {
    // Local transcription disabled ⇒ the shared ASR model never loads, so a
    // mic-button recording can never start.  Surface the configuration
    // rather than a loading state that can never complete.
    if is_transcription_disabled() {
        return Some("Voice recording unavailable — local transcription is disabled");
    }
    match get_status() {
        VoiceStatus::Recording | VoiceStatus::Transcribing => {
            Some("A voice message is already being processed")
        }
        VoiceStatus::Enrolling { .. }
        | VoiceStatus::ListeningDuringEnrollment { .. }
        | VoiceStatus::WaitingForSilenceDuringEnrollment { .. }
        | VoiceStatus::EnrollingNegatives { .. } => Some("Voice enrollment is in progress"),
        VoiceStatus::MicPermissionDenied => {
            Some("Microphone permission denied — enable mic access to record")
        }
        VoiceStatus::MicDisconnected => Some("Microphone disconnected"),
        _ => None,
    }
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

/// The currently active wake word enrollment (prototype + calibration).
#[must_use]
pub(crate) fn get_enrollment() -> Option<WakeWordEnrollment> {
    voice_state().read().unwrap_poison().enrollment.clone()
}

/// Install a wake word enrollment into the global pipeline state.
pub(crate) fn set_enrollment(enrollment: WakeWordEnrollment) {
    let mut state = voice_state().write().unwrap_poison();
    state.enrollment = Some(enrollment);
}

pub fn send_command(cmd: VoiceCommand) {
    if let Some(tx) = &voice_state().read().unwrap_poison().cmd_tx {
        let _ = tx.send(cmd);
    } else {
        warn!("Voice pipeline not initialized — dropping command {cmd:?}");
    }
}

// ── Voice VAD (Earshot) ─────────────────────────────────────────────────

/// Inner VAD check using an explicit detector reference and configurable
/// threshold.  Used by [`is_speech_with_threshold`] (with [`VAD_THRESHOLD`])
/// and by tests that want to supply their own detector to avoid cross-test
/// contamination.
///
/// Processes ALL 256-sample chunks through the detector to keep its internal
/// state (ring buffer + pre-emphasis filter) synchronized with the audio
/// stream, even when speech is detected early in the frame.
pub(crate) fn is_speech_with_detector(
    samples: &[f32],
    detector: &mut earshot::Detector,
    threshold: f32,
) -> bool {
    if samples.is_empty() {
        return false;
    }

    let mut any_speech = false;

    // Clamp audio samples to [-1, 1] before feeding to the VAD detector.
    // TTS-generated audio and microphone capture can produce samples slightly
    // outside this range due to floating-point overflow, which triggers
    // earshot's debug_assert! in debug/test builds.
    let clamp_frame = |frame: &[f32]| -> [f32; 256] {
        let mut clamped = [0.0f32; 256];
        for (i, &s) in frame.iter().enumerate() {
            clamped[i] = s.clamp(-1.0, 1.0);
        }
        clamped
    };

    // Process each complete 256-sample frame (Earshot requires exactly 256
    // samples per call at 16 kHz).  A typical call receives 512-sample frame
    // (FRAME_LENGTH) from the wake-word / enrollment paths, which naturally
    // splits into two 256-sample chunks.  Always process both chunks to keep
    // the detector's sliding window in sync with the actual audio stream.
    for chunk in samples.chunks_exact(256) {
        if detector.predict_f32(&clamp_frame(chunk)) >= threshold {
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
        // Clamp the partial frame too (the copy_from_slice copies raw samples).
        for s in &mut padded[..remainder] {
            *s = s.clamp(-1.0, 1.0);
        }
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

// ── Voice PCM disk cache helpers ─────────────────────────────
//
// Cache bounding: two-phase eviction — age-based (stale entries
// older than voice_cache_max_age_days) followed by size-based (oldest-first
// via mtime, FIFO).  Either phase can be disabled via config (None).
//
// These helpers are consumed by the unit tests below and by the voice-tests
// e2e benchmark (`synthesize_with_pcm_cache`); nothing in the production
// detection path synthesises TTS PCM anymore, so the whole section is
// dead-code-allowed in default builds.

/// Evict stale/oversized entries from the PCM cache directory.
#[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
pub(crate) fn evict_pcm_cache(cache_dir: &Path) {
    use std::time::SystemTime;

    struct CacheEntry {
        path: PathBuf,
        size: u64,
        modified: SystemTime,
    }

    let max_size = crate::config::CONFIG.voice_cache_max_size_bytes();
    let max_age = crate::config::CONFIG.voice_cache_max_age();

    if max_size.is_none() && max_age.is_none() {
        return; // both limits disabled — nothing to evict
    }

    let dir = match std::fs::read_dir(cache_dir) {
        Ok(d) => d,
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                return; // cache directory doesn't exist yet
            }
            warn!(
                "PCM cache: failed to read directory {}: {e}",
                cache_dir.display(),
            );
            return;
        }
    };

    let mut entries: Vec<CacheEntry> = Vec::new();
    let mut total_size: u64 = 0;
    let now = SystemTime::now();

    for entry in dir.flatten() {
        let path = entry.path();

        // Skip transient .tmp files (atomic writes in progress)
        if path.extension().and_then(|e| e.to_str()) == Some("tmp") {
            continue;
        }

        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }

        let size = metadata.len();
        // Skip entries with unreadable mtime — treating them as 1970 would
        // guarantee eviction on the first age check, which is surprising.
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        total_size += size;
        entries.push(CacheEntry {
            path,
            size,
            modified,
        });
    }

    if entries.is_empty() {
        return;
    }

    let mut evicted_count: u64 = 0;
    let mut evicted_bytes: u64 = 0;

    // ── Phase 1: age-based eviction ──────────────────────────────
    if let Some(max_age) = max_age {
        entries.retain(|e| {
            let Ok(age) = now.duration_since(e.modified) else {
                return true; // clock skew — keep the entry
            };
            if age <= max_age {
                return true; // young enough — keep
            }
            // Stale entry — remove
            if std::fs::remove_file(&e.path).is_ok() {
                total_size = total_size.saturating_sub(e.size);
                evicted_bytes += e.size;
                evicted_count += 1;
            } else {
                // Best-effort: if the file can't be deleted, we still remove it
                // from our tracking list (the entry is logically gone from the
                // cache's perspective).  The file's size remains in total_size,
                // which may cause Phase 2 to over-evict slightly.  This is
                // acceptable under best-effort semantics.
                warn!(
                    "PCM cache: failed to remove stale entry {}",
                    e.path.display()
                );
            }
            false // remove from our tracking list
        });
    }

    // ── Phase 2: size-based eviction (oldest-first via mtime, FIFO) ─────
    if let Some(max_size) = max_size
        && total_size > max_size
    {
        // Sort remaining entries by mtime (oldest first) for FIFO eviction
        entries.sort_by_key(|e| e.modified);

        for e in &entries {
            if total_size <= max_size {
                break;
            }
            if std::fs::remove_file(&e.path).is_ok() {
                total_size = total_size.saturating_sub(e.size);
                evicted_bytes += e.size;
                evicted_count += 1;
            } else {
                warn!(
                    "PCM cache: failed to remove excess entry {}",
                    e.path.display()
                );
            }
        }
    }

    if evicted_count > 0 {
        #[allow(clippy::cast_precision_loss)]
        let mb = evicted_bytes as f64 / 1_048_576.0;
        info!("PCM cache: evicted {evicted_count} entries ({:.1} MB)", mb,);
    }
}

/// Deterministic cache key for one TTS-synthesised PCM utterance.
///
/// Covers: text, TTS style, seed, sample rate, and the TTS model version
/// hash.  A change to any TTS model file produces a different key and forces
/// re-synthesis on first run.
#[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
pub(crate) fn pcm_cache_key(
    text: &str,
    style: &str,
    seed: u64,
    sample_rate: u32,
    model_hash: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher.update([0u8]);
    hasher.update(style.as_bytes());
    hasher.update([0u8]);
    hasher.update(seed.to_le_bytes());
    hasher.update([0u8]);
    hasher.update(sample_rate.to_le_bytes());
    hasher.update([0u8]);
    hasher.update(model_hash.as_bytes());
    hex_string(&hasher.finalize())
}

/// Hash of all TTS model SHA256 constants for cache invalidation.
///
/// Any change to any TTS model file produces a different hash, which
/// changes the PCM cache key and triggers re-synthesis on first run.
#[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
pub(crate) fn tts_model_version_hash() -> String {
    let mut hasher = Sha256::new();
    hasher.update(crate::audio::tts::DP_MODEL_SHA256.as_bytes());
    hasher.update(crate::audio::tts::TEXT_ENC_MODEL_SHA256.as_bytes());
    hasher.update(crate::audio::tts::VECTOR_EST_MODEL_SHA256.as_bytes());
    hasher.update(crate::audio::tts::VOCODER_MODEL_SHA256.as_bytes());
    hasher.update(crate::audio::tts::TTS_JSON_SHA256.as_bytes());
    hasher.update(crate::audio::tts::UNICODE_INDEXER_SHA256.as_bytes());
    hasher.update(crate::audio::tts::VOICE_STYLE_SHA256.as_bytes());
    hex_string(&hasher.finalize())
}

/// Write PCM f32 samples to the disk cache atomically (tmp + rename).
#[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
pub(crate) fn write_pcm_cache(path: &Path, samples: &[f32]) {
    let tmp_path = path.with_extension("tmp");
    let mut data: Vec<u8> = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        data.extend_from_slice(&s.to_le_bytes());
    }
    if let Err(e) = std::fs::write(&tmp_path, &data) {
        warn!(
            "PCM cache: failed to write tmp file {}: {e}",
            tmp_path.display()
        );
        return;
    }
    if let Err(e) = std::fs::rename(&tmp_path, path) {
        warn!(
            "PCM cache: failed to rename {} -> {}: {e}",
            tmp_path.display(),
            path.display(),
        );
    }
}

/// Read PCM f32 samples from the disk cache.
///
/// Returns `None` if the cache file does not exist, has non-aligned size
/// (not a multiple of 4 bytes), or fails to read.
#[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
pub(crate) fn read_pcm_cache(path: &Path) -> Option<Vec<f32>> {
    let data = std::fs::read(path).ok()?;
    if data.is_empty() {
        warn!("PCM cache: file {} is empty — deleting", path.display(),);
        let _ = std::fs::remove_file(path);
        return None;
    }
    if data.len() % 4 != 0 {
        warn!(
            "PCM cache: file {} has non-aligned size {} — deleting",
            path.display(),
            data.len(),
        );
        let _ = std::fs::remove_file(path);
        return None;
    }
    let samples = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Some(samples)
}

/// One-shot guard for the per-miss [`evict_pcm_cache`] scan in
/// [`synthesize_with_pcm_cache`].
#[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
static PCM_EVICTION_RAN: AtomicBool = AtomicBool::new(false);

/// Synthesise a phrase via TTS, caching PCM audio to disk.
///
/// Checks the voice PCM disk cache first (keyed by text + style + seed +
/// sample rate + TTS model hash).  On cache hit, returns cached PCM
/// directly without calling TTS.  On cache miss, calls
/// [`crate::audio::tts::synthesize`], writes the result to the cache, and returns
/// the PCM.
///
/// Returns `None` if TTS synthesis fails or the cache directory cannot
/// be resolved.
#[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
pub(crate) fn synthesize_with_pcm_cache(
    text: &str,
    style: &str,
    seed: u64,
    sample_rate: u32,
    model_hash: &str,
    cache_dir: &Path,
) -> Option<Vec<f32>> {
    let key = pcm_cache_key(text, style, seed, sample_rate, model_hash);
    let cache_path = cache_dir.join(&key);

    // Allow manual cache invalidation without deleting files.
    // When set, every read is treated as a miss, forcing re-synthesis.
    let cache_bust = std::env::var("MAHBOT_TEST_CACHE_BUST").as_deref() == Ok("1");

    // Fast path: cache hit (skipped entirely when busting)
    if !cache_bust && let Some(pcm) = read_pcm_cache(&cache_path) {
        debug!("PCM cache HIT for key {key} ({text}, style={style}, seed={seed})");
        return Some(pcm);
    }

    if cache_bust {
        debug!(
            "PCM cache BYPASSED by MAHBOT_TEST_CACHE_BUST=1 ({text}, style={style}, seed={seed}) — synthesising"
        );
    } else {
        debug!("PCM cache MISS for key {key} ({text}, style={style}, seed={seed}) — synthesising");
    }

    // Cache miss — synthesise via TTS
    let Ok(pcm) = crate::audio::tts::synthesize(text, style, seed, sample_rate) else {
        return None;
    };

    // Evict stale/excess entries before writing to keep cache bounded.
    //
    // This performs a full directory scan on every cache miss.  The scan is
    // gated to run ONCE per process: the first miss still evicts, preserving
    // the safety net, while subsequent per-write scans are strictly
    // redundant.
    if !PCM_EVICTION_RAN.swap(true, Ordering::Relaxed) {
        evict_pcm_cache(cache_dir);
    }

    // Write to disk cache atomically
    write_pcm_cache(&cache_path, &pcm);
    Some(pcm)
}

// Microphone capture

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
        // try_send: drop-newest policy when the bounded channel is full
        // (see MIC_CHANNEL_CAPACITY docs).
        if tx.try_send(resampled).is_err() {
            DROPPED_CHUNKS.fetch_add(1, Ordering::Relaxed);
        }
        return;
    }

    // Multi-channel: average each frame across channels.
    let frames = data.len() / usize::from(channels);
    let mut mono: Vec<f32> = Vec::with_capacity(frames);
    for frame in data.chunks_exact(usize::from(channels)) {
        let sum: f32 = frame.iter().map(&convert).sum();
        mono.push(sum / f32::from(channels));
    }
    let resampled = if sample_rate == SAMPLE_RATE {
        mono
    } else {
        crate::util::resample_audio(&mono, sample_rate, SAMPLE_RATE)
    };
    if tx.try_send(resampled).is_err() {
        DROPPED_CHUNKS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Log a microphone stream error and update the pipeline status.
// Error callback for microphone streams — a fn item so it can be shared
// across every build_input_stream call.
#[allow(clippy::needless_pass_by_value)]
fn mic_error(err: cpal::Error) {
    error!("Microphone stream error: {err}");
    // The stream error callback runs on the audio thread; the main pipeline
    // will observe the mic stream ending and set MicDisconnected.
}

/// Build a cpal input stream for a supported sample format.
fn build_int_stream<T, F>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_tx: &Arc<mpsc::Sender<Vec<f32>>>,
    channels: u16,
    sample_rate: u32,
    convert: F,
) -> Result<cpal::Stream, cpal::Error>
where
    T: cpal::SizedSample,
    F: Fn(&T) -> f32 + Send + 'static,
{
    let tx = sample_tx.clone();
    device.build_input_stream::<T, _, _>(
        *config,
        move |data, _| {
            convert_and_send_audio_to_pipeline(&tx, data, channels, sample_rate, &convert);
        },
        mic_error,
        None,
    )
}

/// Start the microphone and return a channel of mono 16 kHz f32 samples.
fn start_microphone() -> Result<(mpsc::Receiver<Vec<f32>>, cpal::Stream)> {
    let (tx, rx) = mpsc::channel::<Vec<f32>>(MIC_CHANNEL_CAPACITY);

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("No default input device found"))?;

    let config = device
        .default_input_config()
        .context("Failed to get default input config")?;

    debug!(
        "Microphone: {} ({:?}, {} Hz, {} ch)",
        device
            .description()
            .map_or_else(|_| "unknown".to_string(), |d| d.name().to_string()),
        config.sample_format(),
        config.sample_rate(),
        config.channels()
    );

    let sample_rate = config.sample_rate();
    let channels = config.channels();
    let sample_tx = Arc::new(tx);
    let stream_config: cpal::StreamConfig = config.into();

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => build_int_stream::<f32, _>(
            &device,
            &stream_config,
            &sample_tx,
            channels,
            sample_rate,
            |&s| s,
        ),
        cpal::SampleFormat::I16 => build_int_stream::<i16, _>(
            &device,
            &stream_config,
            &sample_tx,
            channels,
            sample_rate,
            |&s| f32::from(s) / f32::from(i16::MAX),
        ),
        cpal::SampleFormat::U16 => build_int_stream::<u16, _>(
            &device,
            &stream_config,
            &sample_tx,
            channels,
            sample_rate,
            |&s| (f32::from(s) / f32::from(u16::MAX)) * 2.0 - 1.0,
        ),
        _ => anyhow::bail!("Unsupported sample format: {:?}", config.sample_format()),
    }
    .context("Failed to build microphone input stream")?;

    stream.play().context("Failed to start microphone stream")?;

    debug!(
        "Microphone listening started ({} Hz, {} channels)",
        sample_rate, channels
    );
    Ok((rx, stream))
}

// Transcription via existing Qwen3-ASR

async fn transcribe_audio(samples: &[f32]) -> Result<String> {
    let wav_bytes = crate::audio::tts::render_wav(samples, SAMPLE_RATE)?;
    let tmp_dir = std::env::temp_dir().join("mahbot_voice");

    // Pre-clean any stale files left from a prior crash so they don't
    // accumulate.  This is best-effort — if the
    // directory doesn't exist yet, remove_dir_all returns Ok(()).
    let _ = tokio::fs::remove_dir_all(&tmp_dir).await;

    tokio::fs::create_dir_all(&tmp_dir).await?;
    let tmp_path = tmp_dir.join(format!("cmd_{}.wav", crate::generate_id()));
    tokio::fs::write(&tmp_path, &wav_bytes).await?;

    // 10-minute inference timeout, shared with the enrichment path — long
    // recordings (up to the 10-minute recording cap) must not time out.
    let result = crate::audio::local_transcriber::transcribe_file_async(
        &tmp_path,
        crate::audio::local_transcriber::INFERENCE_TIMEOUT,
    )
    .await;

    // Remove the specific temp file.
    if let Err(e) = tokio::fs::remove_file(&tmp_path).await {
        warn!("Failed to remove temp transcription file: {e}");
    }
    // Remove the entire temp directory (including any leftover files from
    // prior crashes that weren't cleaned).  Uses remove_dir_all instead of
    // remove_dir so that ENOTEMPTY errors from orphaned files don't cause
    // unbounded accumulation.
    if let Err(e) = tokio::fs::remove_dir_all(&tmp_dir).await {
        warn!("Failed to remove temp transcription directory: {e}");
    }

    result
}

// Enrollment quality scoring

/// Compute a per-utterance quality score from the raw audio.
///
/// The composite score (0.0–1.0) is a weighted combination of:
/// - **Duration** (50%): whether the utterance is in [400ms, 2000ms].
/// - **Clipping** (25%): penalty if any sample hit i16::MAX.
/// - **SNR** (25%): estimated signal-to-noise ratio.  If `noise_rms` is
///   `Some`, uses the real pre-speech noise floor captured from the raw audio
///   ring during enrollment; otherwise falls back to an energy-based heuristic.
///
/// # Parameters
/// - `samples`: raw audio samples of the utterance.
/// - `noise_rms`: pre-speech ambient noise RMS captured at the moment of
///   first sustained speech detection.  `None` falls back to
///   energy-based SNR estimation.
#[expect(clippy::cast_precision_loss)]
pub(crate) fn compute_utterance_quality(
    samples: &[f32],
    noise_rms: Option<f32>,
) -> UtteranceQuality {
    let duration_ms = samples_to_ms(samples.len(), SAMPLE_RATE);

    // ── Clipping detection ───────────────────────────────────────────
    let clipping_detected = samples
        .iter()
        .any(|&s| s.abs() >= ENROLLMENT_QUALITY_CLIPPING_THRESHOLD);

    // ── SNR estimation ───────────────────────────────────────────────
    // If we have a real pre-speech noise RMS captured from the raw audio
    // ring at the moment of first sustained speech, compute actual SNR as
    // 20*log10(speech_rms / noise_rms).  Otherwise fall back to energy-based
    // heuristic (estimate_snr_energy) which measures speech dynamic range
    // rather than true SNR.
    let snr_db = if let Some(noise_rms) = noise_rms {
        // Shared RMS helper.  Empty-input NaN → 0.0 via
        // compute_rms is branch-equivalent to the old inline formula (the
        // `speech_rms > noise_rms` guard yields 0.0 in both cases).
        let speech_rms = crate::util::compute_rms(samples);
        if noise_rms > 1e-10 && speech_rms > noise_rms {
            20.0 * (speech_rms / noise_rms).log10()
        } else {
            0.0
        }
    } else {
        estimate_snr_energy(samples)
    };

    // ── Composite score (basic metrics only) ──────────────────────────
    // During enrollment collection, quality is based on duration, clipping,
    // and SNR.
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
        // Shared RMS helper.  `chunk.len().min(FRAME_LENGTH)`
        // was always `chunk.len()` here (chunks ≤ FRAME_LENGTH and the
        // half-frame skip above), so this is bit-identical.
        frame_rms.push(crate::util::compute_rms(chunk));
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

/// Run a self-test of the enrolled prototype against the enrollment buffer.
///
/// Simulates the live detection pipeline for each enrollment utterance: runs
/// each utterance's window embedding through [`score_single_embedding`] with
/// a fresh score window and no adaptive state.
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
    utterance_embeddings: &[Vec<f32>],
    enrollment: &WakeWordEnrollment,
) -> Result<(), String> {
    if utterance_embeddings.is_empty() {
        return Err("Self-test skipped: no enrollment samples".to_string());
    }

    let mut passed = 0usize;

    for embedding in utterance_embeddings {
        // Fresh simulation for each utterance: no cross-utterance state.
        //
        // Feed the utterance embedding ROLLING_WINDOW_N (3) times through the
        // rolling-sum machinery — the streaming pipeline sees the same phrase
        // across ~3 consecutive stride-gated window encodings (a ~0.5 s phrase
        // spans 3 windows at the 160 ms scoring stride).  A single embedding
        // would only ever produce a rolling sum of one soft score (max 1.0),
        // which can never reach match_threshold() = 1.65.
        let mut score_window = Vec::new();
        let mut detected_this = false;
        for _ in 0..ROLLING_WINDOW_N {
            let (detected, _, _, _) = score_single_embedding(
                embedding,
                Some(enrollment),
                &mut score_window,
                None, // no adaptive threshold during enrollment self-test
                ADAPTIVE_K_DEFAULT,
            );
            if detected {
                detected_this = true;
                break;
            }
        }
        if detected_this {
            passed += 1;
        }
    }

    let required = (utterance_embeddings.len() as f32 * ENROLLMENT_QUALITY_SELF_TEST_MIN_FRACTION)
        .ceil() as usize;

    if passed < required {
        Err(format!(
            "Self-test failed: only {passed}/{} utterances triggered detection (need ≥{required}). \
             Try re-enrolling with clearer, more consistent speech.",
            utterance_embeddings.len(),
        ))
    } else {
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

/// Segmentation parameters for [`segment_utterances_by_vad`].
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
    /// (typically [`ENROLLMENT_VAD_CONSECUTIVE_REQUIRED`] = 1 after the threshold unification).
    consecutive_required: usize,
    /// Silence duration in samples before utterance ends
    /// (typically [`ENROLLMENT_SILENCE_THRESHOLD_SAMPLES`] = 4 864 ≈ 304 ms
    /// aligned to streaming's segment timeout).
    silence_threshold_samples: usize,
    /// Samples of pre/post speech context to include
    /// (0 — enrollment now matches streaming detection
    /// by not adding context padding).
    context_padding_samples: usize,
    /// Max samples in the internal raw-audio ring buffer
    /// (typically [`RAW_RING_MAX`] = 3 200 ≈ 200 ms).
    raw_ring_max: usize,
}

/// Module-level default config for [`segment_utterances_by_vad`] using the
/// standard voice-pipeline constants.
///
/// Context padding is intentionally 0 to match the streaming detection path,
/// which does not add context padding.
///
/// Silence threshold uses [`ENROLLMENT_SILENCE_THRESHOLD_SAMPLES`] (~304ms)
/// aligned to streaming detection's [`SEGMENT_TIMEOUT_HOPS`].
pub(crate) const DEFAULT_VAD_SEGMENTATION_CONFIG: VadSegmentationConfig = VadSegmentationConfig {
    frame_length: FRAME_LENGTH,
    hop_length: HOP_LENGTH,
    consecutive_required: ENROLLMENT_VAD_CONSECUTIVE_REQUIRED,
    silence_threshold_samples: ENROLLMENT_SILENCE_THRESHOLD_SAMPLES,
    context_padding_samples: 0,
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
///    **sustained speech**.  On this transition the function may prepend
///    pre-speech context from the raw-audio ring buffer (if
///    `config.context_padding_samples > 0`).
/// 4. After speech, `config.silence_threshold_samples` of consecutive
///    VAD-negative audio ends the utterance.  Post-speech context (if
///    `config.context_padding_samples > 0`) is optionally appended from
///    the raw-audio ring (captured at the first silence frame).
/// 5. The complete utterance is emitted and internal state resets for the
///    next utterance.
///
/// # Parameters
///
/// - `raw_audio`: Complete raw mono audio buffer (16 kHz f32 samples).
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
/// in detection order.
pub(crate) fn segment_utterances_by_vad(
    raw_audio: &[f32],
    vad_decisions: &[bool],
    config: &VadSegmentationConfig,
) -> Vec<Vec<f32>> {
    let VadSegmentationConfig {
        frame_length,
        hop_length,
        consecutive_required,
        silence_threshold_samples,
        context_padding_samples,
        raw_ring_max,
    } = *config;

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

// ── Voice command routing ────────────────────────────────────────────────

async fn broadcast_voice_transcript(transcript: &str, user_name: &str, workspace: &str) {
    if user_name.is_empty() {
        // No active user (admin fallback): broadcast-only — there is no user
        // to persist under or mirror to.
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
        // Broadcast, persist, and mirror through the canonical incoming-message
        // pipeline — exactly the GUI text path — so the transcription reaches
        // Telegram with the same bindings and format. Voice is a strictly local
        // source, so the mirror cannot echo (see `mirror_gui_message_to_telegram`).
        let msg = crate::ChannelMessage {
            user_name: user_name.to_string(),
            reply_target: String::new(),
            content: transcript.to_string(),
            channel: "voice".to_string(),
            workspace: workspace.to_string(),
            optimistic_id: None,
            callback_query_id: None,
        };
        crate::channels::broadcast_and_persist_incoming_message(&msg, transcript, transcript).await;
    }
}

/// Route a transcribed voice command to the appropriate agent.
///
/// Resolves the active user's role and workspace from the user's DB record,
/// then routes through the agent-ID message router.
///
/// Falls back to the Manager router if no active user can be determined.
async fn route_to_agent(text: String) {
    // Try active user first (set by GUI on user switch)
    let user_name = active_user_name();
    if !user_name.is_empty() {
        let pool = crate::users::role_pool(&user_name).await;
        let Some(role) = crate::users::resolve_active_role_from_pool(&user_name, &pool).await
        else {
            // Empty role pool — no role is allowed to answer.
            info!("Voice command dropped (no active role) (user: {user_name}): {text}");
            return;
        };
        let ws = crate::users::resolve_workspace_for_user_name(&user_name).await;
        // Manager→Analyst fallback in personal workspaces (pool-clamped) and
        // Assistant/Artist pinning to the personal workspace, atomically.
        let (role, ws) = crate::users::effective_role_and_workspace(role, ws, &user_name, &pool);

        info!(
            "Voice command -> {role} (user: {user_name}, workspace: {})",
            ws.name
        );

        // Broadcast before routing so the transcript appears immediately
        // while the agent is still working.
        broadcast_voice_transcript(&text, &user_name, &ws.name).await;

        crate::message_router::route_user_message(
            text,
            ws.name,
            user_name,
            "voice".to_string(),
            role,
            None,
        )
        .await;
        return;
    }

    // No active user: fall back to the admin user's DB workspace (same
    // warning + personal fallback as the active-user path). Pool-gated
    // like the active-user path: an emptied admin pool drops the command,
    // and the routed role stays inside the pool — including the same
    // personal-workspace Manager→Analyst clamp.
    let ws = crate::users::resolve_workspace_for_user_name("admin").await;
    let admin_pool = crate::users::role_pool("admin").await;
    if admin_pool.is_empty() {
        info!("Voice command dropped (no active role) (user: admin): {text}");
        return;
    }
    let role = if admin_pool.contains(&crate::Role::Manager) {
        crate::Role::Manager
    } else {
        admin_pool[0]
    };
    // Assistant/Artist fall back to admin's personal workspace. The routed
    // user_name stays empty (pre-existing fallback behavior), so "admin"
    // stands in for the personal identity — an empty name would produce a
    // broken `personal:` path.
    let (role, ws) = crate::users::effective_role_and_workspace(role, ws, "admin", &admin_pool);

    info!("Voice command -> {role} (workspace: {})", ws.name);
    broadcast_voice_transcript(&text, "", &ws.name).await;

    crate::message_router::route_user_message(
        text,
        ws.name,
        String::new(),
        "voice".to_string(),
        role,
        None,
    )
    .await;
}

// Voice pipeline background task

// Adaptive threshold state

/// Tracks running mean and standard deviation of recent per-frame soft scores
/// for adaptive threshold computation.
///
/// Maintains O(1) sum/sum_sq statistics over a rolling window of the last
/// [`ADAPTIVE_WINDOW_N`] scores.  On each call to [`feed`](AdaptiveThresholdState::feed),
/// the adaptive threshold is computed as:
///
/// ```text
/// threshold = (mean_of_window + k × std_of_window) × ROLLING_WINDOW_N
/// ```
///
/// The per-frame result is scaled by [`ROLLING_WINDOW_N`] to convert from
/// per-frame score space [0,1] into rolling-sum space [0, ROLLING_WINDOW_N],
/// matching the detection comparison in [`process_wake_word_score`].  Without
/// this scaling the adaptive threshold (max ~1.0 + k × std) could never reach
/// the rolling sum range, making the feature structurally a no-op.
///
/// The result is then clamped to the safeguard range: at least
/// [`ADAPTIVE_SAFE_HARBOR`] (matching the static [`match_threshold()`]) and
/// never above [`ADAPTIVE_CEILING`].  During the
/// first [`ADAPTIVE_BOOTSTRAP_FRAMES`] frames the function returns `None`,
/// telling the caller to use the static threshold while the window fills.
#[derive(Debug, Clone)]
pub(crate) struct AdaptiveThresholdState {
    /// Rolling window of per-frame scores.
    scores: Vec<f32>,
    /// Running sum of scores in the window.
    sum: f32,
    /// Running sum of squared scores in the window.
    sum_sq: f32,
    /// Bootstrap frame counter.
    bootstrap_count: usize,
}

impl AdaptiveThresholdState {
    /// Create a new adaptive threshold tracker with empty statistics.
    pub(crate) fn new() -> Self {
        Self {
            scores: Vec::with_capacity(ADAPTIVE_WINDOW_N),
            sum: 0.0,
            sum_sq: 0.0,
            bootstrap_count: 0,
        }
    }

    /// Feed a new per-frame soft score and return the adaptive threshold.
    ///
    /// The threshold is computed from per-frame scores (range [0,1]), then
    /// scaled by [`ROLLING_WINDOW_N`] to match the rolling-sum space [0,3]
    /// where the detection comparison lives.  Returns `None` during the
    /// bootstrap period (first [`ADAPTIVE_BOOTSTRAP_FRAMES`] frames) to tell
    /// the caller to use the static threshold.  After bootstrap returns
    /// `Some(threshold)` where `threshold` is already clamped to the safeguard
    /// range ([`ADAPTIVE_SAFE_HARBOR`], [`ADAPTIVE_CEILING`]) in rolling-sum
    /// space.
    pub(crate) fn feed(&mut self, score: f32, k: f32) -> Option<f32> {
        // ── Update rolling window statistics ──
        if self.scores.len() >= ADAPTIVE_WINDOW_N {
            let oldest = self.scores.remove(0);
            self.sum -= oldest;
            self.sum_sq -= oldest * oldest;
        }
        self.scores.push(score);
        self.sum += score;
        self.sum_sq += score * score;

        // ── Bootstrap phase ──
        if self.bootstrap_count < ADAPTIVE_BOOTSTRAP_FRAMES {
            self.bootstrap_count += 1;
            return None;
        }

        // ── Compute adaptive threshold ──
        let threshold = self.compute_threshold(k);

        Some(threshold)
    }

    /// Compute the clamped adaptive threshold from current window statistics.
    ///
    /// Assumes the window is non-empty (caller must check).  Returns the
    /// clamped threshold in rolling-sum space (range [`ADAPTIVE_SAFE_HARBOR`,
    /// [`ADAPTIVE_CEILING`]]).
    ///
    /// Shared by [`feed`](Self::feed) and [`peek`](Self::peek) to avoid
    /// duplicating the computation and clamping chain.
    #[expect(clippy::cast_precision_loss)]
    fn compute_threshold(&self, k: f32) -> f32 {
        let n = self.scores.len() as f32;
        let mean = self.sum / n;
        // Population variance (divide by n, not n-1) — stable for a fixed-size window.
        let variance = (self.sum_sq / n) - (mean * mean);
        let std = variance.max(0.0).sqrt();

        // Scale from per-frame [0,1] space to rolling-sum [0,ROLLING_WINDOW_N]
        // space so the threshold is comparable to the rolling sum used in
        // process_wake_word_score.
        #[expect(clippy::cast_precision_loss)]
        let adaptive = (mean + k * std) * ROLLING_WINDOW_N as f32;

        // ── Safeguards ──
        // Lower bound is the safe harbor (static detection threshold), upper
        // bound is the ceiling.  Note: clamp() propagates NaN while the old
        // max() chain returned the harbor for NaN input — unreachable with
        // finite scores, but the semantics differ.
        adaptive.clamp(ADAPTIVE_SAFE_HARBOR, ADAPTIVE_CEILING)
    }

    /// Return the current adaptive threshold without updating statistics.
    ///
    /// Returns `None` during the bootstrap period (first
    /// [`ADAPTIVE_BOOTSTRAP_FRAMES`] frames) or when the window is empty.
    /// After bootstrap, returns `Some(threshold)` using the current window
    /// statistics and the given `k` multiplier, clamped to the same safeguard
    /// range as [`feed`](Self::feed).
    ///
    /// This is used to avoid contaminating the background statistics with
    /// wake-word-like frames.
    pub(crate) fn peek(&self, k: f32) -> Option<f32> {
        if self.bootstrap_count < ADAPTIVE_BOOTSTRAP_FRAMES {
            return None;
        }
        if self.scores.is_empty() {
            return None;
        }
        Some(self.compute_threshold(k))
    }

    /// Returns `true` while the tracker is still in the bootstrap phase
    /// (first [`ADAPTIVE_BOOTSTRAP_FRAMES`] below-reset frames since the
    /// score-only feed rule).  During bootstrap the caller
    /// feeds below-reset background scores (which populate the window and
    /// advance the bootstrap counter) and peeks wake-word-like scores —
    /// [`peek`](Self::peek) returns `None` during bootstrap, so the static
    /// [`match_threshold()`] stays in effect.
    ///
    /// Production callers no longer consult this method (the feed/peek rule
    /// is now score-only); it exists for unit tests.
    #[cfg(test)]
    pub(crate) fn is_bootstrapping(&self) -> bool {
        self.bootstrap_count < ADAPTIVE_BOOTSTRAP_FRAMES
    }

    /// Reset all statistics (called on pipeline reset / re-enrollment).
    pub(crate) fn reset(&mut self) {
        self.scores.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.bootstrap_count = 0;
    }

    /// Create a pre-warmed state that has exited the bootstrap phase,
    /// initialized with near-silence scores.  Used by the E2E
    /// benchmark so the adaptive threshold is active from the start of
    /// detection testing.  The seed value (~0.03) ensures the threshold
    /// immediately clamps to the safe harbor (1.65),
    /// matching production behavior where real audio starts from silence.
    #[cfg(any(test, feature = "voice-tests"))]
    pub(crate) fn warmed() -> Self {
        let mut state = Self::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.033, ADAPTIVE_K_DEFAULT);
        }
        state
    }

    /// The number of scores currently in the window.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.scores.len()
    }
}

// Pipeline context

/// Per-variant instrumentation collected by the wake word detection benchmark.
/// Feature-gated behind `voice-tests` — zero production overhead.
#[cfg(feature = "voice-tests")]
pub(crate) struct DetectionInstrumentation {
    /// All per-frame `[total_score, rolling_sum, threshold]` triples
    /// encountered during detection of a single variant.  The third element
    /// is the effective threshold used for the rolling window comparison
    /// (adaptive threshold post-bootstrap, or static match_threshold()
    /// during bootstrap / when no adaptive state is configured).
    pub per_frame_scores: Vec<[f32; 3]>,
    /// Count of frames where `total_score < NO_MATCH_RESET_THRESHOLD` (0.35).
    pub n_frames_below_reset: usize,
    /// Count of VAD-positive 512-sample frames during streaming detection.
    pub vad_speech_frames: usize,
    /// Peak rolling-sum score across all segments in this detection session.
    pub peak_score: f32,
    /// Per-frame adaptive threshold trajectory.
    /// Records the effective threshold value (`per_frame_scores[i][2]`) at each
    /// embedding frame so the ADAPTIVE_CEILING calibration can be data-driven.
    pub adaptive_threshold_trajectory: Vec<f32>,
    /// Count of frames where the effective threshold hit ADAPTIVE_CEILING.
    pub ceiling_limited_frames: usize,
    /// Index into [`per_frame_scores`](Self::per_frame_scores) of the first
    /// frame where the rolling sum reached the effective threshold
    /// (detection trigger).  `None` if detection never triggered on
    /// test-utterance frames.
    pub first_trigger_frame_idx: Option<usize>,
    /// Per-frame window embeddings (1024-dim, L2-normalized), one per scored
    /// window, index-aligned with [`per_frame_scores`](Self::per_frame_scores).
    ///
    /// Captured only for the benchmark's hard-negative mining pre-pass
    /// (crossing-frame embeddings of false-accept variants are injected into
    /// the anti-prototype construction).  NOT copied into per-variant
    /// reports — raw 1024-dim dumps would bloat the JSON; the only consumer
    /// is the mining pre-pass, which reads them transiently.  Kept off the
    /// per-frame hot path in production builds by the `voice-tests` gate.
    pub per_frame_embeddings: Vec<Vec<f32>>,
    /// Per-hop VAD decisions during streaming detection, in order — one entry
    /// per VAD decision (each 512-sample frame processed, feeding its new
    /// 256-sample half to the VAD).
    pub per_hop_vad: Vec<bool>,
}

#[cfg(feature = "voice-tests")]
impl DetectionInstrumentation {
    pub fn new() -> Self {
        Self {
            per_frame_scores: Vec::new(),
            n_frames_below_reset: 0,
            vad_speech_frames: 0,
            peak_score: 0.0,
            adaptive_threshold_trajectory: Vec::new(),
            ceiling_limited_frames: 0,
            first_trigger_frame_idx: None,
            per_frame_embeddings: Vec::new(),
            per_hop_vad: Vec::new(),
        }
    }
}

/// Runtime state for the voice pipeline main loop.
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct PipelineCtx {
    mic_rx: Option<mpsc::Receiver<Vec<f32>>>,
    mic_stream: Option<cpal::Stream>,
    is_listening: bool,
    is_recording: bool,
    /// Mic-button-initiated recording (Home composer). While active, wake-word
    /// detection is paused and audio accumulates into [`command_buffer`].
    manual_recording: bool,
    /// Whether wake-word listening was active before a manual recording
    /// started (or was requested while one was in progress) — restored when
    /// the manual recording ends.
    resume_listening_after_recording: bool,
    command_buffer: Vec<f32>,
    /// Track silence duration by audio sample count rather than wall-clock
    /// time, so that system load / processing delays don't affect recording
    /// cutoff consistency.
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
    /// and show a "speak louder" warning.
    enrollment_no_speech_frame_count: usize,
    /// Consecutive VAD-positive frame counter for enrollment sustained-speech
    /// confirmation.  Now, accumulation starts on
    /// the first VAD-positive frame ([`ENROLLMENT_VAD_CONSECUTIVE_REQUIRED`] = 1),
    /// matching streaming detection's first-positive behavior.
    vad_positives_in_a_row: usize,
    /// VAD threshold for the current mode.  Unified to [`VAD_THRESHOLD`] for
    /// both detection and enrollment.
    vad_threshold: f32,
    /// Separate Earshot VAD detector instance for enrollment mode.
    ///
    /// The streaming detection path uses the global [`VAD_DETECTOR`] singleton
    /// to maintain continuous noise-floor and ring-buffer state across the
    /// live microphone stream.  Enrollment mode uses its own detector instance
    /// to prevent mode-transition state contamination.
    ///
    /// Initialised to `None` outside enrollment.  Set to
    /// `Some(earshot::Detector::default())` when enrollment starts and cleared
    /// to `None` when enrollment ends or is cancelled.
    enrollment_vad: Option<earshot::Detector>,
    /// Raw audio ring for wake-word detection.  Accumulates raw mic samples
    /// (no AGC/NS) capped at [`AUDIO_BUFFER_MAX`]; the trailing ≤1 s window
    /// is encoded for scoring.
    audio_buffer: Vec<f32>,
    /// VAD-gated speech-only window for encoder scoring.
    ///
    /// Holds only the [`HOP_LENGTH`] hops that the VAD classified as speech,
    /// capped at [`WAKE_WORD_WINDOW_SAMPLES`] (draining the front).  This
    /// matches the enrollment windowing exactly — enrollment encodes
    /// VAD-segmented speech-only utterances, so scoring the same speech-only
    /// window keeps the cosine distributions aligned (raw trailing audio
    /// dilutes the embedding with context/silence and shifts detection
    /// cosines ~0.05-0.08 below enrollment, letting confusables through).
    speech_window: Vec<f32>,
    /// Number of samples at the FRONT of [`audio_buffer`] already consumed by
    /// the VAD frame loop.  The VAD loop advances this cursor instead of
    /// draining the ring, so the trailing ≤1 s window remains available for
    /// the encoder even after every frame has been VAD-checked.
    vad_cursor: usize,
    /// Samples processed since the last window encoding (scoring step).
    /// Reset to 0 after each encode; when it reaches [`SCORE_STRIDE_SAMPLES`]
    /// and speech was seen, the pipeline encodes and scores.
    last_score_sample_count: usize,
    /// Queue of completed enrollment utterances awaiting processing by the main
    /// pipeline loop.  Each element is a Vec<f32> of raw audio samples for one
    /// utterance.  The extracted [`segment_utterances_by_vad`] function detects
    /// utterance boundaries and queues completed utterances here.  The main loop
    /// pops them one at a time via [`pop_front`](VecDeque::pop_front) and
    /// processes them through [`handle_enrollment_sample`].  Using a queue
    /// (rather than a single `Option`) ensures that if multiple utterances
    /// complete within a single mic frame, all are preserved — no utterance is
    /// silently dropped.
    enrollment_pending: VecDeque<Vec<f32>>,
    auto_start_pending: bool,
    /// Timestamp of the last automatic transcriber retry attempt.  Used to
    /// debounce so we don't spam the retry path every 1-second tick when the
    /// transcriber is in [`ModelState::Failed`] (the periodic wake-up checks
    /// the state).
    last_model_retry: Option<Instant>,
    /// Timestamp of the last wake word detection.
    /// Used to enforce a cooldown period after detection to prevent rapid
    /// consecutive false triggers.
    pub(crate) last_wake_word_detection: Option<Instant>,
    /// Pre-speech noise RMS captured at the moment of first sustained speech
    /// detection during enrollment.  Computed from the raw audio ring so the
    /// SNR estimate reflects the true room noise floor.
    /// Used for real SNR estimation in [`compute_utterance_quality`].
    /// Reset to `None` after the utterance is consumed by
    /// [`handle_enrollment_sample`].
    noise_rms_estimate: Option<f32>,
    // ── Phase 3 (owner-negative) enrollment state ────────────
    /// Whether the pipeline is actively collecting owner-negative general speech
    /// during Phase 3 enrollment.  Set by `transition_to_phase3()` when the
    /// user has completed all 10 wake word utterances.
    collecting_negatives: bool,
    /// Independent audio buffer for Phase 3 VAD frame processing.  NOT shared
    /// with `audio_buffer` (used by enrollment Phase 2) to avoid interference.
    phase3_audio_buf: Vec<f32>,
    /// Silence tracking for Phase 3 chunk boundary detection.  Uses
    /// [`ENROLLMENT_SILENCE_THRESHOLD_SAMPLES`] (same constant as enrollment
    /// utterance end, aligned to streaming's segment timeout)
    /// to detect chunk boundaries.
    phase3_silence_samples: usize,
    /// Accumulated VAD-positive speech samples collected during Phase 3.
    /// Monotone 1:1 record of real audio: each VAD-positive hop contributes
    /// exactly [`HOP_LENGTH`] samples, exactly once (see [`phase3_processed`]).
    /// When `negatives_speech_samples >= SAMPLE_RATE * NEGATIVES_TARGET_SECONDS`,
    /// Phase 3 is complete and the pipeline transitions to finalization.
    negatives_speech_samples: usize,
    /// Watermark into [`phase3_audio_buf`]: number of samples at the head of
    /// the buffer already fed to the VAD.  The frame loop resumes at this
    /// index on the next mic chunk so each hop reaches the stateful VAD
    /// exactly once.  Without it, an un-drained buffer re-processes (and
    /// re-counts) every prior frame on each chunk — the quadratic Phase-3
    /// counter defect (mahbot-1782).
    phase3_processed: usize,
    /// Wall-clock start time of Phase 3 owner-negative collection.  Used for
    /// timeout — if [`PHASE3_TIMEOUT_SECS`] elapses, finalize with whatever
    /// was collected.
    phase3_start_time: Option<Instant>,
    /// Rolling window of per-frame soft scores from the cosine-prototype
    /// scorer.  Each element is the soft score (0.0–1.0) from the last
    /// scoring step.
    score_window: Vec<f32>,
    /// Consecutive VAD-negative hops since the last VAD-positive frame,
    /// persisted across calls.  When it reaches [`SEGMENT_TIMEOUT_HOPS`]
    /// (~300 ms) the per-segment detection state resets.
    segment_silence_hops: usize,
    /// Adaptive threshold tracker for the rolling score window.
    adaptive_threshold: AdaptiveThresholdState,
    /// Adaptive threshold k multiplier (from config, clamped).
    adaptive_k: f32,
    /// Accumulated non-VAD audio during enrollment (pre-speech ambient noise
    /// and inter-utterance silence), saved as negative examples at speech
    /// onset.
    negative_audio_buf: Vec<f32>,
    /// Timestamp until which the pipeline suppresses Error→Listening
    /// transitions (set after transcription failures).
    refractory_until: Option<Instant>,
    /// Timestamp of the last rate-limited transcription-error message.
    last_error_message_time: Option<Instant>,
    /// Timestamp of the last rate-limited voice notice.
    last_voice_notice_time: Option<Instant>,
    /// Remaining automatic model-retry cycles (see
    /// [`MAX_AUTO_MODEL_RETRY_CYCLES`]).  Decremented only when the periodic
    /// self-healing path actually initiates a retry; the explicit GUI retry
    /// bypasses this budget.
    auto_model_retries_left: u32,
    #[cfg(feature = "voice-tests")]
    instrumentation: DetectionInstrumentation,
}

/// Parameterised pipeline state reset level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResetLevel {
    Full,
    Soft,
    Cancel,
}

impl PipelineCtx {
    pub(crate) fn new() -> Self {
        Self {
            mic_rx: None,
            mic_stream: None,
            is_listening: false,
            is_recording: false,
            manual_recording: false,
            resume_listening_after_recording: false,
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
            speech_window: Vec::new(),
            vad_cursor: 0,
            last_score_sample_count: 0,
            enrollment_pending: VecDeque::new(),
            auto_start_pending: CONFIG.voice_enabled().as_deref() == Some("true"),
            last_model_retry: None,
            last_wake_word_detection: None,
            score_window: Vec::new(),
            noise_rms_estimate: None,
            collecting_negatives: false,
            phase3_audio_buf: Vec::new(),
            phase3_silence_samples: 0,
            negatives_speech_samples: 0,
            phase3_processed: 0,
            phase3_start_time: None,
            vad_threshold: VAD_THRESHOLD,
            enrollment_vad: None,
            negative_audio_buf: Vec::new(),
            refractory_until: None,
            last_error_message_time: None,
            last_voice_notice_time: None,
            auto_model_retries_left: MAX_AUTO_MODEL_RETRY_CYCLES,
            adaptive_threshold: AdaptiveThresholdState::new(),
            adaptive_k: {
                let k_str = crate::config::CONFIG.adaptive_k();
                k_str
                    .parse::<f32>()
                    .unwrap_or(ADAPTIVE_K_DEFAULT)
                    .clamp(ADAPTIVE_K_MIN, ADAPTIVE_K_MAX)
            },
            segment_silence_hops: 0,
            #[cfg(feature = "voice-tests")]
            instrumentation: DetectionInstrumentation::new(),
        }
    }

    /// Set the manual-recording flag on both the pipeline context and the
    /// global mirror used by the GUI ([`is_manual_recording`]).
    fn set_manual_recording(&mut self, active: bool) {
        self.manual_recording = active;
        MANUAL_RECORDING_ACTIVE.store(active, Ordering::Relaxed);
    }

    /// Run a Full reset while preserving [`auto_start_pending`] (the reset
    /// clears it; a pending wake-word auto-start must survive the
    /// recording-only-mic lifecycle).
    fn full_reset_preserving_auto_start(&mut self) {
        let auto_start = self.auto_start_pending;
        self.reset_pipeline_state(ResetLevel::Full);
        self.auto_start_pending = auto_start;
    }

    /// Parameterised pipeline state reset.
    ///
    /// | Field | Full | Soft | Cancel |
    /// |---|---|---|---|
    /// | `audio_buffer`, `command_buffer`, `score_window`, `negative_audio_buf`, `frame_vad`, `frame_raw_audio` | cleared | cleared | cleared |
    /// | `silence_sample_count`, `segment_silence_hops`, `last_score_sample_count` | = 0 | = 0 | = 0 |
    /// | `utterance_had_speech`, `utterance_silence_samples`, `enrollment_no_speech_frame_count`, `vad_positives_in_a_row`, `emitted_utterances`, `enrollment_pending`, `noise_rms_estimate` | cleared | cleared | cleared |
    /// | Phase 3 (`collecting_negatives`, `phase3_audio_buf`, `phase3_silence_samples`, `negatives_speech_samples`, `phase3_processed`, `phase3_start_time`) | cleared | cleared | cleared |
    /// | `vad_threshold` | `VAD_THRESHOLD` | preserved | `VAD_THRESHOLD` |
    /// | `enrollment_vad` | `None` | `None` | `None` |
    /// | `last_wake_word_detection` | `None` | preserved | `None` |
    /// | `auto_start_pending` | `false` | preserved | preserved |
    /// | `is_recording` | `false` | preserved | preserved |
    /// | `manual_recording` | `false` | preserved | preserved |
    /// | `resume_listening_after_recording` | preserved | preserved | preserved |
    /// | VAD (`reset_vad()`) | called | NOT called | NOT called |
    /// | Global `enrollment_embeddings`, `negative_audio_chunks` | preserved | preserved | cleared |
    /// | `refractory_until`, `last_error_message_time`, `last_voice_notice_time`, `last_model_retry`, `mic_rx`, `mic_stream`, `is_listening`, `enrollment_mode` | NOT touched | NOT touched | NOT touched |
    fn reset_pipeline_state(&mut self, level: ResetLevel) {
        // ── Audio accumulators (cleared by all levels) ──
        self.audio_buffer.clear();
        self.speech_window.clear();
        self.vad_cursor = 0;
        self.command_buffer.clear();
        self.silence_sample_count = 0;
        self.score_window.clear();
        self.negative_audio_buf.clear();
        self.segment_silence_hops = 0;
        self.last_score_sample_count = 0;

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

        // ── Phase 3 owner-negative state (cleared by all levels) ──
        self.collecting_negatives = false;
        self.phase3_audio_buf.clear();
        self.phase3_silence_samples = 0;
        self.negatives_speech_samples = 0;
        self.phase3_processed = 0;
        self.phase3_start_time = None;

        // ── Enrollment VAD detector (cleared by all levels) ──
        // Separate VAD instance prevents state contamination between
        // enrollment and streaming modes.
        self.enrollment_vad = None;

        match level {
            ResetLevel::Full => {
                self.vad_threshold = VAD_THRESHOLD;
                self.last_wake_word_detection = None;
                self.auto_start_pending = false;
                self.is_recording = false;
                self.set_manual_recording(false);
                reset_vad();
                self.adaptive_threshold.reset();

                // Full does NOT clear global enrollment accumulators — those
                // survive mic stop/start cycles so mid-enrollment progress is
                // preserved across toggle-off/on.
                // Only ResetLevel::Cancel (explicit cancel or start-fresh)
                // clears the global enrollment accumulators.
            }
            ResetLevel::Soft => {
                // Preserve VAD state, vad_threshold, last_wake_word_detection
                // cooldown, auto_start_pending, is_recording, and global
                // enrollment accumulators.
                // Clear rolling-window detection state so stale scores cannot
                // carry cross-utterance contamination across Soft pipeline
                // resets (which occur during the detection→recording handoff).
            }
            ResetLevel::Cancel => {
                self.vad_threshold = VAD_THRESHOLD;
                self.last_wake_word_detection = None;
                self.adaptive_threshold.reset();

                // Cancel also clears global enrollment accumulators.
                voice_state().write().unwrap_poison().reset_enrollment();
            }
        }
    }

    /// Reset all per-segment detection state at a VAD-driven utterance boundary.
    /// Called when [`SEGMENT_TIMEOUT_HOPS`] (~300 ms) of
    /// consecutive VAD-negative hops have been observed since the last
    /// VAD-positive frame.
    ///
    /// This prevents soft scores and rolling sums from
    /// accumulating across separate utterances separated by more than ~300 ms
    /// of silence, which was a structural source of false triggers.
    ///
    /// | Field | Reset? | Rationale |
    /// |---|---|---|
    /// | `score_window` | Yes | **Critical**: rolling scores must not accumulate across utterances — this is the primary false-trigger mechanism this function fixes |
    /// | `adaptive_threshold` | Yes | Noise floor estimate is per-segment; the 5-call bootstrap is brief and acceptable |
    /// | `segment_silence_hops` | Yes | Reset the silence counter so the next segment starts fresh |
    ///
    /// **Preserved**: `audio_buffer` (normal drain handles leftover overlap),
    /// VAD state (acoustic environment unchanged), `vad_threshold`,
    /// `last_wake_word_detection` (cooldown still active if within 3 s),
    /// `is_recording`.
    fn reset_detection_segment(&mut self) {
        // ── Clear per-segment rolling scores (PRIMARY false-trigger fix) ──
        self.score_window.clear();

        // ── Reset threshold/scores ──
        self.adaptive_threshold.reset();

        // ── Reset silence counter ──
        self.segment_silence_hops = 0;

        // ── Reset the raw ring + VAD cursor so the next utterance starts a
        // fresh segment (the trailing ≤1 s window must not span the boundary).
        self.audio_buffer.clear();
        self.speech_window.clear();
        self.vad_cursor = 0;
        self.last_score_sample_count = 0;
    }

    /// Handle the segment boundary check at the end of a detection call.
    ///
    /// Called from [`handle_wake_word_detection`] after the VAD frame loop,
    /// with the accumulated `hop_count` from the VAD-negative frame counter.
    ///
    /// If `hop_count` reaches [`SEGMENT_TIMEOUT_HOPS`], resets per-segment
    /// detection state so that soft scores and rolling sums do not accumulate
    /// across separate utterances.
    ///
    /// If `hop_count` is below the threshold, persists the counter in
    /// [`segment_silence_hops`](PipelineCtx::segment_silence_hops) so the
    /// next call to [`handle_wake_word_detection`] can continue counting.
    ///
    /// # Parameters
    ///
    /// - `hop_count`: Accumulated consecutive VAD-negative frames from the
    ///   detection call's VAD loop.
    fn handle_segment_boundary(&mut self, hop_count: usize) {
        if hop_count >= SEGMENT_TIMEOUT_HOPS {
            // Re-check the not-recording condition: a just-fired detection
            // must NOT be reset (the detection→recording handoff in
            // handle_wake_word_detection completes the transition).
            if !self.is_recording {
                self.reset_detection_segment();
            }
        } else {
            self.segment_silence_hops = hop_count;
        }
    }

    /// Check whether the refractory period has elapsed and transition back
    /// to [`VoiceStatus::Listening`] if so.
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

    /// Whether the 10-second rate limit has elapsed since `last` (true when
    /// no prior message was sent).  The transcription-error and voice-notice
    /// limiters stay independent — separate fields so a discard notice is not
    /// suppressed by a recent error message (and vice versa).
    fn should_send_rate_limited(last: Option<Instant>) -> bool {
        let now = Instant::now();
        last.is_none_or(|t| now.duration_since(t).as_secs() >= 10)
    }

    /// 10-second transcription-error rate-limit guard.
    fn should_send_error_message(&self) -> bool {
        Self::should_send_rate_limited(self.last_error_message_time)
    }

    /// 10-second voice-notice rate-limit guard.
    fn should_send_voice_notice(&self) -> bool {
        Self::should_send_rate_limited(self.last_voice_notice_time)
    }

    /// Broadcast a chat message to the active user's voice workspace.
    async fn broadcast_voice_message(&mut self, msg: &str) {
        let user_name = active_user_name();
        if user_name.is_empty() {
            return;
        }
        let role = crate::users::resolve_active_role(&user_name).await;
        let ws = crate::users::resolve_workspace_for_user_name(&user_name).await;
        // Assistant/Artist conversations live in the user's personal workspace;
        // a None role (empty pool or store failure) fails closed to the
        // resolved workspace — the notice stays visible in the current view.
        let ws = match role {
            Some(role) => crate::users::effective_workspace_for_role(role, ws, &user_name),
            None => ws,
        };
        crate::channels::broadcast_and_persist_agent_response(
            &user_name,
            "voice",
            msg,
            Some("voice".to_string()),
            &ws.name,
        )
        .await;
    }

    /// Broadcast a rate-limited transcription-failure chat message (same as
    /// the wake-word path) so the user sees a persistent indicator.
    async fn broadcast_transcription_error(&mut self) {
        if !self.should_send_error_message() {
            return;
        }
        self.last_error_message_time = Some(Instant::now());
        self.broadcast_voice_message("*Voice: transcription failed — try again*")
            .await;
    }

    /// Broadcast a rate-limited voice notice chat message (discarded
    /// recordings etc.). Shared by the wake-word and mic-button paths.
    async fn broadcast_voice_notice(&mut self, msg: &str) {
        if !self.should_send_voice_notice() {
            return;
        }
        self.last_voice_notice_time = Some(Instant::now());
        self.broadcast_voice_message(msg).await;
    }

    fn handle_start_listening(&mut self) {
        // While a mic-button recording is in progress, defer wake-word
        // listening until the recording ends (mutual exclusion). The resume
        // flag is set so `end_manual_recording` starts listening afterwards.
        if self.manual_recording {
            self.resume_listening_after_recording = true;
            info!("Voice pipeline: start_listening deferred until manual recording ends");
            return;
        }
        // Wake word shares the ASR transcriber: with transcription disabled
        // the shared model never loads, so wake word is disabled too.
        if is_transcription_disabled() {
            self.auto_start_pending = false;
            warn!(
                "Ignoring start_listening — local transcription disabled (wake word requires ASR)"
            );
            return;
        }
        // Defense-in-depth: reject if voice has been disabled between the
        // time the command was sent and the time it's processed. This
        // mirrors the guard in handle_start_enrollment.
        if !is_enabled() {
            self.auto_start_pending = false;
            warn!("Ignoring start_listening — voice assistant is disabled");
            return;
        }
        if !models_ready() {
            // The shared ASR model is still loading — mark pending so
            // check_auto_start retries when it becomes ready. This is NOT set
            // on mic failure, preventing a continuous retry loop.
            //
            // If the transcriber has previously failed (ModelError trap state),
            // trigger a retry immediately so the user doesn't need to
            // restart the app.
            if crate::audio::local_transcriber::is_failed() {
                warn!("ASR model previously failed — triggering retry...");
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
                    self.mic_stream = Some(stream);
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

    /// Stop the wake-word mic and tear down pipeline state.
    ///
    /// Returns `true` when an in-progress mic-button recording was aborted
    /// (the caller broadcasts a discard notice so the loss is not silent).
    fn handle_stop_listening(&mut self) -> bool {
        // Full reset: the mic stream is being torn down, so the old VAD
        // state is no longer representative of the next acoustic
        // environment.  Full level uses reset_vad().
        // Global enrollment accumulators are preserved across mic stop/start
        // so mid-enrollment progress survives toggle-off/on.
        let aborted_recording = self.manual_recording;
        self.reset_pipeline_state(ResetLevel::Full);
        self.is_listening = false;
        self.enrollment_mode = false;
        // Abort any mic-button recording — the mic is being torn down.
        self.set_manual_recording(false);
        self.resume_listening_after_recording = false;
        drop(self.mic_stream.take());
        self.mic_rx = None;
        set_status(VoiceStatus::Disabled);
        info!("Voice pipeline: stopped listening");
        aborted_recording
    }

    /// Returns `true` when an in-progress mic-button recording was aborted
    /// (the caller broadcasts a discard notice so the loss is not silent).
    fn handle_start_enrollment(&mut self, phrase: &str) -> bool {
        if !self.is_listening {
            warn!("Cannot start enrollment: microphone not running");
            set_status(VoiceStatus::Error(
                "Microphone not running — enable Voice first".to_string(),
            ));
            return false;
        }

        // A mic-button recording cannot coexist with enrollment: the audio
        // router gives enrollment priority, so the manual buffer would starve
        // (its auto-send cap never fires) and later resume as garbage.
        // Abort the recording and discard its buffer.
        let aborted_recording = self.manual_recording;
        if aborted_recording {
            warn!("Aborting manual recording — enrollment started");
            self.set_manual_recording(false);
            self.resume_listening_after_recording = false;
            self.command_buffer.clear();
            self.silence_sample_count = 0;
        }

        // Resume existing enrollment progress if available (e.g., the user
        // clicked Enroll while already enrolled or mid-enrollment — the
        // global enrollment_embeddings from the interrupted session is
        // intact).  When starting fresh (existing_utterances == 0), use
        // Cancel-level reset to clear stale buffers while preserving VAD
        // continuity (same mic stream, same acoustic environment).
        let existing_utterances = voice_state()
            .read()
            .unwrap_poison()
            .enrolled_utterance_count;

        if existing_utterances == 0 {
            self.reset_pipeline_state(ResetLevel::Cancel);
        } else {
            info!(
                "Resuming enrollment from utterance \
                 {existing_utterances}/{NUM_ENROLLMENT_SAMPLES}",
            );
        }

        // Store the normalized wake word phrase in the global enrollment
        // state AFTER reset, so reset_enrollment does not clear it.
        // This phrase is consumed by finalize_enrollment_pipeline() on
        // completion.
        let normalized = normalize_phrase(phrase);
        voice_state().write().unwrap_poison().enrolling_phrase = Some(normalized);

        self.enrollment_mode = true;
        // Initialize a separate VAD detector for this enrollment session
        // to prevent state contamination between enrollment and streaming
        // modes.
        self.enrollment_vad = Some(earshot::Detector::default());
        // vad_threshold is intentionally NOT changed here — it stays at
        // VAD_THRESHOLD (0.5) for both enrollment and streaming detection.
        set_status(VoiceStatus::Enrolling {
            sample: existing_utterances,
            total: NUM_ENROLLMENT_SAMPLES,
            duration_ms: 0,
            quality: None,
        });
        info!(
            "Voice pipeline: enrollment started (resuming from utterance \
             {existing_utterances}/{NUM_ENROLLMENT_SAMPLES})",
        );
        aborted_recording
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

    /// Transition the pipeline from Phase 2 (wake word utterance collection)
    /// to Phase 3 (owner-negative general speech collection).
    ///
    /// Called by the main loop when `utterances_collected` is true and
    /// `collecting_negatives` is false.  VAD threshold is already unified to
    /// [`VAD_THRESHOLD`], so no threshold switch occurs.
    /// Drains any stale enrollment pending queue, clears the negative audio
    /// buffer and silence counters, and sets the Phase 3 state flags.
    fn transition_to_phase3(&mut self) {
        // Drain any stale enrollment_pending queue (should be empty, but be safe).
        if !self.enrollment_pending.is_empty() {
            warn!(
                "transition_to_phase3: draining {} stale enrollment pending utterances",
                self.enrollment_pending.len(),
            );
            self.enrollment_pending.clear();
        }

        // Clear inter-utterance state that may carry over from Phase 2.
        self.negative_audio_buf.clear();
        self.utterance_silence_samples = 0;
        self.utterance_had_speech = false;
        self.vad_positives_in_a_row = 0;

        // Set Phase 3 state.
        self.collecting_negatives = true;
        self.phase3_start_time = Some(Instant::now());

        set_status(VoiceStatus::EnrollingNegatives {
            accumulated_secs: 0,
            target_secs: NEGATIVES_TARGET_SECONDS,
            wall_clock_elapsed: 0,
        });
        info!(
            "Voice pipeline: transitioning to Phase 3 owner-negative \
             collection (target {NEGATIVES_TARGET_SECONDS}s of VAD-positive speech)",
        );
    }

    fn handle_shutdown(&mut self) {
        self.set_manual_recording(false);
        self.resume_listening_after_recording = false;
        drop(self.mic_stream.take());
    }

    /// Start a mic-button-initiated voice message recording.
    ///
    /// Pauses wake-word detection for the duration of the recording
    /// (mutual exclusion — audio routes to [`handle_manual_recording_audio`]
    /// instead of [`handle_wake_word_detection`]). When the recording ends,
    /// wake-word listening resumes exactly as it was before it started.
    ///
    /// Works independently of the wake-word assistant: the microphone is
    /// started on demand when voice is not already listening.
    /// Returns a notice message when the on-demand microphone could not be
    /// started (the caller broadcasts it so the failure is user-visible);
    /// `None` otherwise (started, or silently rejected by a guard the GUI's
    /// pre-flight already surfaced).
    fn handle_start_manual_recording(&mut self) -> Option<&'static str> {
        if self.manual_recording {
            warn!("Manual recording already in progress");
            return None;
        }
        if self.is_recording {
            warn!("Cannot start manual recording — wake-word recording in progress");
            return None;
        }
        if self.enrollment_mode || self.collecting_negatives {
            warn!("Cannot start manual recording during enrollment");
            return None;
        }
        // With local transcription disabled the shared ASR model never loads
        // (or is unavailable after a restart) — surface the configuration,
        // not a loading state that can never complete.  Unconditional guard
        // mirroring [`manual_recording_blocked_reason`], so a still-loaded
        // in-memory model cannot contradict the disabled configuration.
        if is_transcription_disabled() {
            warn!("Cannot start manual recording — local transcription disabled");
            return Some("Voice recording unavailable — local transcription is disabled");
        }

        // Save whether wake-word listening should resume after the recording.
        self.resume_listening_after_recording = self.is_listening;

        if self.mic_rx.is_none() {
            // Voice assistant not listening — start the mic just for this
            // recording (models must be ready for transcription).
            if !models_ready() {
                warn!("Cannot start manual recording — models not ready");
                return Some("Voice models are still loading — try again in a moment");
            }
            match start_microphone() {
                Ok((rx, stream)) => {
                    self.mic_rx = Some(rx);
                    self.mic_stream = Some(stream);
                }
                Err(e) => {
                    warn!("Failed to start recording mic: {e}");
                    return Some(if is_mic_permission_error(&e) {
                        "Microphone permission denied — enable mic access to record"
                    } else {
                        "Could not start the microphone — check your input device"
                    });
                }
            }
        }

        self.set_manual_recording(true);
        self.command_buffer.clear();
        self.silence_sample_count = 0;
        set_status(VoiceStatus::RecordingManual);
        info!("Voice pipeline: manual recording started");
        None
    }

    /// Stop a mic-button recording (send or discard).
    async fn handle_stop_manual_recording(&mut self, send: bool) {
        if !self.manual_recording {
            warn!("StopRecording called but no manual recording in progress");
            return;
        }
        let cmd_buf = std::mem::take(&mut self.command_buffer);
        self.silence_sample_count = 0;
        if send && !cmd_buf.is_empty() {
            self.finalize_manual_recording(cmd_buf).await;
        } else {
            // Discarded or empty — tear down and resume listening.
            self.end_manual_recording();
            if send {
                self.broadcast_voice_notice("*Voice: no speech detected — recording discarded*")
                    .await;
            }
        }
    }

    /// Accumulate audio during a mic-button recording; auto-send at the
    /// 10-minute cap.
    #[allow(clippy::cast_precision_loss)]
    async fn handle_manual_recording_audio(&mut self, samples: &[f32]) {
        self.command_buffer.extend_from_slice(samples);
        let duration_secs = self.command_buffer.len() as f64 / f64::from(SAMPLE_RATE);
        if duration_secs > MAX_RECORD_SECS as f64 {
            debug!("Manual recording stopped: max duration ({duration_secs:.1}s)");
            let cmd_buf = std::mem::take(&mut self.command_buffer);
            self.silence_sample_count = 0;
            self.finalize_manual_recording(cmd_buf).await;
        }
    }

    /// Transcribe a completed manual-recording buffer and route it to the
    /// user's active role agent (same broadcast+routing path as wake-word
    /// commands), then restore wake-word listening state.
    async fn finalize_manual_recording(&mut self, cmd_buf: Vec<f32>) {
        set_status(VoiceStatus::Transcribing);
        match transcribe_audio(&cmd_buf).await {
            Ok(transcribed) if !transcribed.trim().is_empty() => {
                route_to_agent(transcribed).await;
            }
            Ok(_) => {
                warn!(
                    "Empty transcription — dropping manual recording ({} samples)",
                    cmd_buf.len()
                );
                self.broadcast_voice_notice("*Voice: no speech detected — recording discarded*")
                    .await;
            }
            Err(e) => {
                warn!("Manual recording transcription failed: {e}");
                self.broadcast_transcription_error().await;
            }
        }
        self.end_manual_recording();
    }

    /// Finalize a manual recording session: resume wake-word listening if it
    /// was active before (or was requested during the recording), otherwise
    /// tear down the recording-only mic.
    fn end_manual_recording(&mut self) {
        self.set_manual_recording(false);
        if self.resume_listening_after_recording {
            self.resume_listening_after_recording = false;
            if self.is_listening {
                // Wake-word mic still running — clear recording buffers while
                // preserving VAD continuity, then flip back to listening.
                self.reset_pipeline_state(ResetLevel::Soft);
                set_status(VoiceStatus::Listening);
            } else {
                // Voice was toggled ON during the manual recording. If the
                // wake-word mic cannot start right now (models still loading),
                // auto_start_pending is re-armed by handle_start_listening so
                // check_auto_start retries once models are ready. Tear down
                // the recording-only mic instead of leaving a zombie stream.
                self.handle_start_listening();
                if !self.is_listening {
                    drop(self.mic_stream.take());
                    self.mic_rx = None;
                    // Preserve a mic-failure status set by
                    // handle_start_listening so the user sees WHY the
                    // wake-word assistant did not resume.
                    if !matches!(
                        get_status(),
                        VoiceStatus::MicPermissionDenied | VoiceStatus::MicDisconnected
                    ) {
                        set_status(VoiceStatus::Disabled);
                    }
                }
            }
        } else {
            // Recording-only mic — tear it down. Preserve auto_start_pending
            // (cleared by the Full reset) so a wake-word assistant enabled in
            // config but still loading models auto-starts after the recording.
            self.full_reset_preserving_auto_start();
            self.is_listening = false;
            drop(self.mic_stream.take());
            self.mic_rx = None;
            set_status(VoiceStatus::Disabled);
        }
        debug!("Voice pipeline: manual recording ended");
    }

    /// Attempt to retry loading the shared ASR transcriber, debounced to at
    /// most once every 30 seconds.  This prevents rapid retry storms from the
    /// periodic 1-second wake-up in the main pipeline loop.
    ///
    /// Returns `true` when a retry was actually initiated.
    fn try_retry_models(&mut self) -> bool {
        let cooldown = Duration::from_secs(30);
        if self
            .last_model_retry
            .is_some_and(|t| t.elapsed() < cooldown)
        {
            return false;
        }
        if crate::audio::local_transcriber::retry_init() {
            self.last_model_retry = Some(Instant::now());
            set_status(VoiceStatus::LoadingModels);
            true
        } else {
            false
        }
    }

    /// Periodic (self-healing) retry path — bounded so a persistently failing
    /// download cannot loop forever.
    ///
    /// Unlike the user-initiated paths ([`handle_start_listening`] and the
    /// explicit [`VoiceCommand::RetryModelLoading`]), this runs without any
    /// user involvement on every pipeline iteration.  It consumes the
    /// [`MAX_AUTO_MODEL_RETRY_CYCLES`] budget and stops once exhausted —
    /// further recovery requires the GUI retry button or a restart.
    fn try_retry_models_auto(&mut self) {
        if self.auto_model_retries_left == 0 {
            return;
        }
        if self.try_retry_models() {
            self.auto_model_retries_left -= 1;
            if self.auto_model_retries_left == 0 {
                warn!(
                    "Voice: automatic ASR retry budget exhausted ({} cycles) — \
                     use the GUI retry button or restart the app",
                    MAX_AUTO_MODEL_RETRY_CYCLES,
                );
            }
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
        // Once the transcriber transitions back to Ready, this function picks
        // it up via the auto_start_pending flag (set by handle_start_listening).
        if self.auto_start_pending && models_ready() && !self.is_listening {
            self.auto_start_pending = false;
            send_command(VoiceCommand::StartListening);
        }
    }
}

/// Schedule a transition back to [`VoiceStatus::Listening`] after enrollment
/// finalization completes successfully.
///
/// Runs [`reset_pipeline_state(Cancel)`](PipelineCtx::reset_pipeline_state),
/// sets `enrollment_mode = false`, and spawns a 1.5-second delayed task that
/// transitions to [`VoiceStatus::Listening`] (respecting the global shutdown
/// token so it does not write stale state after pipeline exit).
///
/// Called from both:
/// - The Phase 2→4 direct finalization path (existing behavior)
/// - The Phase 3→4 path (new owner-negative collection path)
fn schedule_listening_transition(ctx: &mut PipelineCtx) {
    // Clear all audio buffers BEFORE resetting enrollment_mode to prevent
    // stale audio from leaking into detection mode during the ~1.5s delay.
    // Cancel level: clears audio buffers, enrollment accumulators, restores
    // vad_threshold to VAD_THRESHOLD, but preserves VAD continuity.
    // Does NOT call reset_vad() — the earshot noise floor estimate from the
    // enrollment phase is deliberately carried through to detection mode.
    ctx.reset_pipeline_state(ResetLevel::Cancel);
    ctx.enrollment_mode = false;
    // Schedule transition to Listening after showing "Enrolled" for 1.5s.
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

/// Encode raw PCM chunks through the shared Qwen3-ASR encoder into 1024-dim
/// L2-normalized window embeddings.
///
/// Each chunk is encoded once via [`encode_window`] (the trailing ≤1 s of
/// the chunk).  Runs in `spawn_blocking` — the encoder forward is CPU-bound.
/// Returns one embedding per chunk; chunks that fail to encode are skipped.
async fn extract_negative_embeddings(chunks: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    if chunks.is_empty() {
        return Vec::new();
    }
    let Some(model) = crate::audio::local_transcriber::shared_model_arc() else {
        warn!("extract_negative_embeddings: ASR model not loaded");
        return Vec::new();
    };
    tokio::task::spawn_blocking(move || {
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            match encode_window(&model, chunk) {
                Ok(emb) => embeddings.push(emb),
                Err(e) => {
                    warn!(
                        "Failed to encode negative chunk ({} samples): {e} — skipping",
                        chunk.len(),
                    );
                }
            }
        }
        embeddings
    })
    .await
    .unwrap_or_else(|e| {
        warn!("Negative embedding task panicked: {e}");
        Vec::new()
    })
}

/// Centroid-consistency gate for the enrollment utterance set.
///
/// The prototype is the L2-normalized mean of the accepted utterance
/// embeddings.  Requires ≥ [`MIN_ENROLLMENT_UTTERANCES`] (5) utterances and
/// that ≥ ceil(N × [`ENROLLMENT_CONSISTENCY_MIN_FRACTION`]) have cosine ≥
/// [`ENROLLMENT_CONSISTENCY_MIN_SIMILARITY`] (0.70) against the centroid.
///
/// Pure function so the gate is unit-testable without the ASR model.
/// Returns `Ok(prototype)` — the L2-normalized centroid used for negative
/// calibration — on pass, or `Err` with a user-facing message.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn enrollment_consistency_check(utterance_embeddings: &[Vec<f32>]) -> Result<Vec<f32>, String> {
    if utterance_embeddings.len() < MIN_ENROLLMENT_UTTERANCES {
        return Err(format!(
            "Only {} enrollment utterances collected (need ≥{MIN_ENROLLMENT_UTTERANCES}). \
             Speak clearly and close to the microphone.",
            utterance_embeddings.len(),
        ));
    }
    let dim = WAKE_WORD_EMBEDDING_DIM;
    let mut centroid = vec![0.0f32; dim];
    for emb in utterance_embeddings {
        if emb.len() != dim {
            return Err(format!(
                "Utterance embedding has dim {} (expected {dim})",
                emb.len(),
            ));
        }
        for (c, v) in centroid.iter_mut().zip(emb) {
            *c += v;
        }
    }
    let inv = 1.0 / utterance_embeddings.len() as f32;
    for c in &mut centroid {
        *c *= inv;
    }
    let norm: f32 = centroid.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-8 {
        let inv_norm = 1.0 / norm;
        for c in &mut centroid {
            *c *= inv_norm;
        }
    }

    let required =
        (utterance_embeddings.len() as f32 * ENROLLMENT_CONSISTENCY_MIN_FRACTION).ceil() as usize;
    let passed = utterance_embeddings
        .iter()
        .filter(|emb| {
            crate::vector::cosine_similarity(emb, &centroid)
                >= ENROLLMENT_CONSISTENCY_MIN_SIMILARITY
        })
        .count();
    if passed < required {
        return Err(format!(
            "Only {passed}/{} enrollment utterances match the prototype \
             (need ≥{required} at cosine ≥ {ENROLLMENT_CONSISTENCY_MIN_SIMILARITY}). \
             Try re-enrolling with clearer, more consistent speech.",
            utterance_embeddings.len(),
        ));
    }
    Ok(centroid)
}

/// Finalize enrollment: build the prototype, calibrate against negatives, run
/// the self-test, and persist the v2 enrollment record.
///
/// Called after Phase 3 owner-negative collection completes (or times out).
/// Returns `true` on success; on failure sets an error status and returns
/// `false` (the user can re-initiate enrollment).
#[expect(clippy::too_many_lines)]
async fn finalize_enrollment_pipeline() -> bool {
    if !models_ready() {
        warn!("finalize_enrollment_pipeline: models not ready");
        return false;
    }

    // ── Snapshot enrollment buffers (positives + ambient/owner negatives) ──
    let (utterance_embeddings, negative_audio_chunks, owner_negative_chunks) = {
        let state = voice_state().read().unwrap_poison();
        (
            state.enrollment_embeddings.clone(),
            state.negative_audio_chunks.clone(),
            state.owner_negative_chunks.clone(),
        )
    };
    let enrolled_phrase = voice_state()
        .read()
        .unwrap_poison()
        .enrolling_phrase
        .clone()
        .unwrap_or_else(|| DEFAULT_WAKE_WORD_PHRASE.to_string());

    // ── Consistency gate ──
    // Returns the L2-normalized prototype (centroid) used for calibration.
    let prototype = match enrollment_consistency_check(&utterance_embeddings) {
        Ok(proto) => proto,
        Err(msg) => {
            warn!("Enrollment finalization failed: {msg}");
            set_status(VoiceStatus::Error(format!("Enrollment failed: {msg}")));
            return false;
        }
    };

    // ── Build negatives (owner + ambient raw chunks) ──
    // Negative calibration sources: owner-negative Phase 3 chunks and ambient
    // non-wake-word chunks collected during enrollment.  TTS confusable/
    // unrelated phrases were part of the old Conv1D training set; the
    // prototype-cosine pipeline calibrates against real recorded negatives
    // only — simpler and sufficient for the cosine floor.
    //
    // The two encode batches are independent spawn_blocking tasks sharing the
    // read-only model Arc (per-call scratch buffers make concurrent forwards
    // safe), so they run in parallel.
    let owner_fut = async {
        if owner_negative_chunks.is_empty() {
            Vec::new()
        } else {
            extract_negative_embeddings(owner_negative_chunks).await
        }
    };
    let ambient_fut = async {
        if negative_audio_chunks.is_empty() {
            Vec::new()
        } else {
            extract_negative_embeddings(negative_audio_chunks).await
        }
    };
    let (owner_embs, ambient_embs) = tokio::join!(owner_fut, ambient_fut);
    info!(
        "Enrollment: encoded {} owner-negative + {} ambient-negative embeddings",
        owner_embs.len(),
        ambient_embs.len(),
    );
    let mut negative_embeddings: Vec<Vec<f32>> = owner_embs;
    negative_embeddings.extend(ambient_embs);

    // ── Clear chunk buffers from global state ──
    {
        let mut state = voice_state().write().unwrap_poison();
        state.negative_audio_chunks.clear();
        state.owner_negative_chunks.clear();
    }

    // ── Calibrate + build the enrollment record ──
    let calibration = calibrate_negatives(&prototype, &negative_embeddings);
    // Preserve the original creation timestamp from an existing stored record
    // (re-enrollment keeps the first creation time); trained_at is now.
    let created_at = get_enrollment()
        .filter(|e| !e.created_at.is_empty())
        .map_or_else(turso::now, |e| e.created_at.clone());
    let trained_at = turso::now();

    let Some(enrollment) = WakeWordEnrollment::build(
        enrolled_phrase.clone(),
        &utterance_embeddings,
        calibration,
        &negative_embeddings,
        created_at,
        trained_at,
    ) else {
        warn!("Enrollment finalization failed: could not build enrollment record");
        set_status(VoiceStatus::Error(
            "Enrollment failed — please re-enroll".to_string(),
        ));
        return false;
    };

    // ── Cancel guard: check before persisting ──
    if crate::shutdown::shutdown_token().is_cancelled() {
        warn!(
            "finalize_enrollment_pipeline: cancelled during finalization, \
             not persisting enrollment state"
        );
        return false;
    }

    // ── Self-test ──
    // Require ≥80% of the accepted utterances to trigger detection with a
    // fresh score window and no adaptive state.
    if let Err(e) = run_enrollment_self_test(&utterance_embeddings, &enrollment) {
        warn!("Enrollment self-test failed — model rejected: {e}.  Re-enrollment required.");
        set_status(VoiceStatus::Error(format!(
            "Enrollment validation failed: {e}.  Please try again with clearer speech."
        )));
        return false;
    }
    info!("Enrollment self-test: passed — deploying model");

    // ── Store + persist ──
    set_enrollment(enrollment.clone());
    voice_state().write().unwrap_poison().model_phrase = Some(enrolled_phrase.clone());

    if !persist_enrollment(&enrollment).await {
        warn!("Enrollment persisted to memory but failed to save to config DB");
        return false;
    }

    // ── Clear enrollment accumulators ──
    {
        let mut state = voice_state().write().unwrap_poison();
        state.enrollment_embeddings.clear();
        state.enrolled_utterance_count = 0;
        state.utterances_collected = false;
    }

    true
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

    // ── Load persisted wake word enrollment (v2 schema) ──
    if let Some(json) = CONFIG.wake_word_templates() {
        match serde_json::from_str::<WakeWordEnrollment>(&json) {
            Ok(enr)
                if enr.schema_version == wake_word::ENROLLMENT_SCHEMA_VERSION
                    && enr.embedding_dim == WAKE_WORD_EMBEDDING_DIM
                    && enr.prototype.len() == WAKE_WORD_EMBEDDING_DIM =>
            {
                let phrase = enr.phrase.clone();
                set_enrollment(enr);
                voice_state().write().unwrap_poison().model_phrase = Some(phrase.clone());
                info!("Loaded wake word enrollment (v2, phrase={phrase})");
            }
            Ok(_) | Err(_) => {
                // Legacy (v1 classifier) records and unparseable JSON are
                // rejected outright — no migration path.  The user must
                // re-enroll.
                warn!(
                    "Stored wake word enrollment is incompatible or legacy (v1). \
                     Re-enrollment required."
                );
            }
        }
    }

    // ── Model gating: wake word shares the ASR transcriber ──
    // No download machinery — the transcriber's background init owns the
    // load.  The status is resolved from the transcriber's state via
    // [`resolved_model_status`]; the periodic wake-up below re-resolves it
    // once loading finishes so the UI can never hang on LoadingModels.
    //
    // Transcription disabled ⇒ wake word disabled: the shared ASR model is
    // never loaded when `audio_transcription_use_local == "false"`, so wake
    // word cannot function.  Surface Disabled (not LoadingModels — that would
    // hang forever) and skip auto-start.
    let transcription_disabled = is_transcription_disabled();
    if transcription_disabled {
        warn!(
            "Voice assistant: local transcription disabled — wake word is disabled too \
             (shared ASR model required)"
        );
        set_status(VoiceStatus::Disabled);
    } else {
        set_status(resolved_model_status(
            crate::audio::local_transcriber::is_loaded(),
            crate::audio::local_transcriber::is_failed(),
            is_enabled(),
        ));
    }

    let mut ctx = PipelineCtx::new();
    if ctx.auto_start_pending && !transcription_disabled {
        set_enabled(true);
        info!("Voice assistant enabled in config — will auto-start when models are ready");
    }

    // Try auto-start immediately if models are already loaded (avoids waiting
    // for the select! timeout on the first iteration).
    ctx.check_auto_start();

    // Periodic metrics log via tokio::time::Interval.  Fires
    // every 60 seconds on wall-clock time regardless of audio activity.
    let mut metrics_interval = tokio::time::interval(Duration::from_mins(1));

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
                    Some(VoiceCommand::StopListening) => {
                        // Voice toggle-off mid-recording aborts the mic-button
                        // recording — broadcast a notice so the loss is visible.
                        if ctx.handle_stop_listening() {
                            ctx.broadcast_voice_notice(
                                "*Voice: recording discarded — voice assistant turned off*",
                            )
                            .await;
                        }
                    }
                    Some(VoiceCommand::StartEnrollment(phrase)) => {
                        // Enrollment aborts any in-progress mic-button
                        // recording — broadcast a notice so the loss is not
                        // silent (mirrors the toggle-off / mic-disconnect paths).
                        if ctx.handle_start_enrollment(&phrase) {
                            ctx.broadcast_voice_notice(
                                "*Voice: recording discarded — enrollment started*",
                            )
                            .await;
                        }
                    }
                    Some(VoiceCommand::CancelEnrollment) => ctx.handle_cancel_enrollment(),
                    Some(VoiceCommand::RetryModelLoading) => {
                        // Explicit retry from GUI — bypass debounce.
                        if crate::audio::local_transcriber::retry_init() {
                            ctx.last_model_retry = Some(Instant::now());
                            set_status(VoiceStatus::LoadingModels);
                        } else {
                            warn!("RetryModelLoading: transcriber is not in Failed state");
                        }
                    }
                    Some(VoiceCommand::StartRecording) => {
                        if let Some(msg) = ctx.handle_start_manual_recording() {
                            // On-demand mic failed — surface it to the user.
                            ctx.broadcast_voice_notice(msg).await;
                        }
                    }
                    Some(VoiceCommand::StopRecordingSend) => {
                        ctx.handle_stop_manual_recording(true).await;
                    }
                    Some(VoiceCommand::StopRecordingDiscard) => {
                        ctx.handle_stop_manual_recording(false).await;
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
                    // Mic loss mid-recording aborts the mic-button recording —
                    // broadcast a notice so the loss is not silent.
                    if ctx.handle_stop_listening() {
                        ctx.broadcast_voice_notice(
                            "*Voice: recording discarded — microphone disconnected*",
                        )
                        .await;
                    }
                    continue;
                };

                CHUNKS_RECEIVED.fetch_add(1, Ordering::Relaxed);

                // ── TTS playback gate ──
                // If TTS audio is actively playing through the speakers, skip
                // ALL audio processing for this chunk.  This prevents TTS echo
                // from contaminating the VAD ring buffer, window embeddings,
                // wake word scoring, enrollment, and recording.
                //
                // The gate stays active for a reverb tail period after playback
                // ends (see crate::audio::tts::PLAYBACK_REVERB_TAIL_MS).
                if crate::audio::tts::is_playback_active() {
                    continue;
                }

                // Samples flow RAW to the mode dispatch — the old AGC/NS
                // preprocessor was removed; the encoder pipeline needs no
                // preprocessing and the enrollment SNR uses the raw floor.
                if ctx.collecting_negatives {
                    handle_negative_collection_audio(&samples, &mut ctx);
                } else if ctx.enrollment_mode {
                    let (sample, total) = {
                        let state = voice_state().read().unwrap_poison();
                        (state.enrolled_utterance_count, NUM_ENROLLMENT_SAMPLES)
                    };
                    handle_enrollment_audio(&samples, &mut ctx, sample, total);
                } else if ctx.manual_recording {
                    // Mic-button recording takes priority over wake-word
                    // detection — the two are mutually exclusive.
                    ctx.handle_manual_recording_audio(&samples).await;
                } else if ctx.is_recording {
                    handle_recording_audio(samples, &mut ctx).await;
                } else {
                    handle_wake_word_detection(&samples, &mut ctx);
                }
            }

            // Periodic wake-up so auto-recovery can fire when the shared ASR
            // transcriber finishes loading or transitions to Ready/Failed
            // after the initial select! entry.  check_auto_start runs in the
            // post-select section below so we don't duplicate it here.
            () = tokio::time::sleep(Duration::from_secs(1)) => {
                // ── Transcriber state transition ──
                // Light polling.  A transcriber failure surfaces ModelError
                // from ANY status (pre-existing behavior — e.g. an
                // externally-initiated load via providers::recreate_all
                // failing while voice is off must not silently hide).
                // Otherwise, resolve a LoadingModels status to its terminal
                // state once loading finishes, using the same
                // [`resolved_model_status`] as the startup block so the
                // footer can never hang on "Loading…": Listening when voice
                // is enabled, Disabled (indicator hidden) when it is
                // disabled — regardless of which path initiated the loading
                // (pipeline start, automatic retry, or the GUI Retry button).
                if crate::audio::local_transcriber::is_failed() {
                    if !matches!(get_status(), VoiceStatus::ModelError) {
                        set_status(VoiceStatus::ModelError);
                    }
                } else if matches!(get_status(), VoiceStatus::LoadingModels) {
                    let resolved = resolved_model_status(
                        crate::audio::local_transcriber::is_loaded(),
                        crate::audio::local_transcriber::is_failed(),
                        is_enabled(),
                    );
                    if !matches!(resolved, VoiceStatus::LoadingModels) {
                        set_status(resolved);
                    }
                }
            }

            // Periodic metrics log every ~60 seconds.
            _ = metrics_interval.tick() => {
                let m = get_voice_metrics();
                let roll_avg = m.avg_embedding_latency_ns;
                let life_avg = m.lifetime_avg_embedding_latency_ns();
                debug!(
                    target: "mahbot::voice::metrics",
                    "Pipeline metrics: chunks_received={0} dropped_chunks={1} ({2:.2}%) \
                     embeddings_computed={3} rolling_avg_latency={4}ns \
                     lifetime_avg_latency={5}ns",
                    m.chunks_received,
                    m.dropped_chunks,
                    m.drop_rate() * 100.0,
                    m.embeddings_computed,
                    roll_avg,
                    life_avg,
                );
            }
        }

        // Periodic auto-recovery: if the transcriber is in Failed state,
        // attempt to retry loading (debounced to at most once every 30s and
        // bounded to MAX_AUTO_MODEL_RETRY_CYCLES cycles so a persistently
        // failing download cannot loop forever).  This runs regardless of
        // auto_start_pending so that the model error state is self-healing
        // even when voice is toggled off/on manually.
        if crate::audio::local_transcriber::is_failed() {
            ctx.try_retry_models_auto();
        }

        // ── Phase 2→3 transition ──
        // After all 10 enrollment utterances are collected, automatically
        // transition to owner-negative speech collection (Phase 3).
        let utterances_collected = voice_state().read().unwrap_poison().utterances_collected;
        if utterances_collected && !ctx.collecting_negatives {
            ctx.transition_to_phase3();
        }

        // ── Phase 3→4 transition ──
        // When the target VAD-positive speech time is reached (or wall-clock
        // timeout), finalize residual audio, build the prototype, and clean up.
        if ctx.collecting_negatives {
            let target_samples = SAMPLE_RATE as usize * NEGATIVES_TARGET_SECONDS;
            let target_met = ctx.negatives_speech_samples >= target_samples;
            let timed_out = ctx
                .phase3_start_time
                .is_some_and(|t| t.elapsed() >= Duration::from_secs(PHASE3_TIMEOUT_SECS));

            if target_met || timed_out {
                // Finalize any residual audio in phase3_audio_buf.
                if !ctx.phase3_audio_buf.is_empty() {
                    push_owner_negative_chunk(
                        std::mem::take(&mut ctx.phase3_audio_buf),
                        "residual",
                    );
                }
                // The residual take emptied the buffer: reset the state-machine
                // indices so a failed finalization (which leaves
                // collecting_negatives=true until the user retries/cancels)
                // cannot resume processing against a stale watermark into an
                // empty buffer.
                ctx.phase3_processed = 0;
                ctx.phase3_silence_samples = 0;

                // Cap is ~360k at 16kHz, well within f64 mantissa precision.
                let collected_secs = {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        (ctx.negatives_speech_samples as f64) / f64::from(SAMPLE_RATE)
                    }
                };

                info!(
                    "Phase 3 complete: {:.1}s VAD-positive speech collected \
                     (target {}s){}",
                    collected_secs,
                    NEGATIVES_TARGET_SECONDS,
                    if timed_out {
                        " (wall-clock timeout)"
                    } else {
                        ""
                    },
                );

                // Build and persist the enrollment.
                let success = finalize_enrollment_pipeline().await;

                if success {
                    set_status(VoiceStatus::Enrolled);
                    schedule_listening_transition(&mut ctx);
                }
                // On failure, the error status is already set by
                // finalize_enrollment_pipeline.  The user can retry by
                // re-initiating enrollment.
            } else {
                // Update status with current progress.
                let accumulated_secs = ctx.negatives_speech_samples / SAMPLE_RATE as usize;
                let wall_clock_elapsed = ctx.phase3_start_time.map_or(0, |t| t.elapsed().as_secs());
                set_status(VoiceStatus::EnrollingNegatives {
                    accumulated_secs,
                    target_secs: NEGATIVES_TARGET_SECONDS,
                    wall_clock_elapsed,
                });
            }
        }

        // Process any pending enrollment utterances (accumulated inline to avoid
        // race conditions with the command channel).  Using a VecDeque so all
        // completed utterances are preserved even if multiple complete within a
        // single mic frame — each is popped one per tick.
        // The encoder forward inside handle_enrollment_sample uses spawn_blocking
        // so it doesn't block.  Only processes during Phase 2 (before
        // utterances_collected), since Phase 3 handles audio independently.
        if !utterances_collected && let Some(samples) = ctx.enrollment_pending.pop_front() {
            let noise_rms = ctx.noise_rms_estimate.take();
            handle_enrollment_sample(samples, noise_rms).await;
        }

        // Transition from Error to Listening after the refractory period
        // has elapsed.  This replaces the old 2-second blocking
        // sleep with a non-blocking check in the main loop.
        ctx.check_refractory_period();

        // Auto-start when models become ready (async load case).
        ctx.check_auto_start();
    }

    info!("Voice pipeline exited");
}

/// Normalize a wake word phrase (trim, lowercase, collapse whitespace).
/// Empty input falls back to [`DEFAULT_WAKE_WORD_PHRASE`].
#[must_use]
pub(crate) fn normalize_phrase(s: &str) -> String {
    let normalized = s
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    if normalized.is_empty() {
        DEFAULT_WAKE_WORD_PHRASE.to_string()
    } else {
        normalized
    }
}

/// The currently enrolled wake word phrase (from the loaded/persisted
/// enrollment), if any.
#[must_use]
pub fn get_enrolled_phrase() -> Option<String> {
    voice_state().read().unwrap_poison().model_phrase.clone()
}

/// Persist a wake word enrollment to the config DB and update the in-memory
/// CONFIG.
///
/// The CONFIG update ensures GUI snapshot readers / pipeline restart see the
/// latest enrollment.  The per-field persist paths never write
/// `wake_word_templates` (it's structurally excluded), so this update is
/// about cross-session visibility.
///
/// Warnings are logged on failure. Returns `true` if both the DB write and the
/// CONFIG update succeeded. Callers use the return value to gate their own
/// success logging.
async fn persist_enrollment(enrollment: &WakeWordEnrollment) -> bool {
    let Ok(json) = serde_json::to_string(enrollment) else {
        warn!("Failed to serialize wake word enrollment for persistence");
        return false;
    };
    let store = crate::config_db::store();
    if let Err(e) = store.set_kv("wake_word_templates", &json).await {
        warn!("Failed to persist wake word enrollment: {e}");
        return false;
    }
    if !CONFIG.set_string_field("wake_word_templates", &json) {
        warn!(
            "Failed to update CONFIG with wake word enrollment (key not recognized by \
             set_string_field — it may have drifted from the `stringify!` arms)"
        );
        return false;
    }
    true
}

/// Handle audio during wake-word command recording (post-detection).
///
/// Accumulates audio into `command_buffer`, tracks silence by sample count,
/// and stops recording after [`SILENCE_THRESHOLD_SAMPLES`] of silence or the
/// [`MAX_RECORD_SECS`] cap, then transcribes and routes the command.
///
/// Silence duration is measured in audio samples (not wall-clock time) so that
/// system load / processing delays don't affect recording cutoff consistency.
#[allow(clippy::cast_precision_loss)]
async fn handle_recording_audio(samples: Vec<f32>, ctx: &mut PipelineCtx) {
    ctx.command_buffer.extend_from_slice(&samples);
    let speech = is_speech_with_threshold(&samples, ctx.vad_threshold);
    if speech {
        ctx.silence_sample_count = 0;
    } else {
        // Accumulate silence by raw chunk size: each call receives a
        // variable-size chunk of audio samples directly from the mic.
        ctx.silence_sample_count += samples.len();
    }

    let duration_secs = ctx.command_buffer.len() as f64 / f64::from(SAMPLE_RATE);
    let silence_timeout = ctx.silence_sample_count >= SILENCE_THRESHOLD_SAMPLES;

    if silence_timeout || duration_secs > MAX_RECORD_SECS as f64 {
        debug!(
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
                    route_to_agent(transcribed).await;
                }

                // Cleanup: return to listening immediately on success.
                // Soft reset clears detection/recording buffers (audio,
                // score_window, command_buffer, negative_audio_buf) while
                // preserving VAD state, vad_threshold, and the wake-word
                // cooldown timestamp to prevent immediate re-triggering.
                ctx.reset_pipeline_state(ResetLevel::Soft);
                ctx.is_recording = false;
                set_status(VoiceStatus::Listening);
            }
            Err(e) => {
                warn!("Transcription failed: {e}");
                set_status(VoiceStatus::Error("Transcription failed".to_string()));

                // Broadcast an error chat message (rate-limited: at most one
                // per 10 seconds) so the user sees a persistent indicator
                // instead of a flash that disappears after 2s.
                ctx.broadcast_transcription_error().await;

                // Enforce a 3-second refractory period before returning to
                // Listening (replaces the old 2-second blocking sleep with a
                // non-blocking alternative).
                ctx.refractory_until = Some(Instant::now() + Duration::from_secs(3));

                // Cleanup the recording state.
                // Soft reset clears recording/detection buffers while preserving
                // VAD continuity so the noise floor estimate survives the
                // refractory period.
                ctx.reset_pipeline_state(ResetLevel::Soft);
                ctx.is_recording = false;
                // Do NOT set status to Listening here — the refractory delay
                // is handled in the main loop's post-select section.
            }
        }
    }
}

/// Check that an enrollment utterance is long enough for meaningful matching.
/// Returns an error message if the sample is shorter than 400ms — this rejects
/// noise blips and coughs while accepting any real wake word utterance.
///
/// Uses wall-clock duration: the encoder produces exactly one window
/// embedding per utterance, so the 400ms floor (well below the ~1s window
/// that a typical wake word needs) is the meaningful quality gate.
fn check_enrollment_utterance_length(duration_ms: u64) -> Result<(), String> {
    if duration_ms < ENROLLMENT_QUALITY_DURATION_MIN_MS {
        Err(format!(
            "Utterance too short ({duration_ms}ms) — speak longer"
        ))
    } else {
        Ok(())
    }
}

/// Encode a completed enrollment utterance through the shared Qwen3-ASR
/// encoder and store the 1024-dim window embedding in the global enrollment
/// state.
///
/// The utterance is encoded ONCE (no PCM augmentation — the prototype-cosine
/// pipeline needs no variant expansion).  Rejects utterances shorter than
/// 400 ms and updates the enrollment UI status with per-utterance quality.
#[allow(clippy::cast_precision_loss)]
async fn handle_enrollment_sample(samples: Vec<f32>, noise_rms: Option<f32>) {
    if !models_ready() {
        warn!("Models not ready for enrollment");
        return;
    }

    let duration_ms = samples_to_ms(samples.len(), SAMPLE_RATE);

    // ── Minimum utterance length check ──
    if let Err(msg) = check_enrollment_utterance_length(duration_ms) {
        warn!("{msg}");
        set_status(VoiceStatus::Error(msg));
        return;
    }

    // ── Quality assessment (computed before `samples` moves into the
    // encoder task below) ──
    let quality = compute_utterance_quality(&samples, noise_rms);

    // ── Encode the utterance ONCE via the shared encoder ──
    let Some(model) = crate::audio::local_transcriber::shared_model_arc() else {
        warn!("ASR model not loaded — skipping enrollment sample");
        return;
    };
    let embedding = tokio::task::spawn_blocking(move || encode_window(&model, &samples))
        .await
        .unwrap_or_else(|e| Err(anyhow!("Enrollment encode task panicked: {e}")));
    let embedding = match embedding {
        Ok(emb) => emb,
        Err(e) => {
            warn!("Enrollment sample encoding failed: {e}");
            return;
        }
    };

    // ── Push into global enrollment state ──
    let utterance_count = {
        let mut state = voice_state().write().unwrap_poison();
        state.enrollment_embeddings.push(embedding);
        state.enrolled_utterance_count += 1;
        state.enrolled_utterance_count
        // state dropped here — no lock held across await
    };

    info!(
        "Enrolled utterance {utterance_count}/{NUM_ENROLLMENT_SAMPLES} \
         ({:.1}s, quality={:.2})",
        duration_ms as f64 / 1000.0,
        quality.score,
    );

    if utterance_count >= NUM_ENROLLMENT_SAMPLES {
        // All 10 utterances collected.  Signal that Phase 2 is complete and
        // the pipeline should transition to Phase 3 (owner-negative collection)
        // or proceed directly to finalization.
        voice_state().write().unwrap_poison().utterances_collected = true;
        // Keep the current Enrolling status until transition_to_phase3 fires.
    } else {
        set_status(VoiceStatus::Enrolling {
            sample: utterance_count,
            total: NUM_ENROLLMENT_SAMPLES,
            duration_ms,
            quality: Some(quality),
        });
    }
}

/// Handle a chunk of raw mic audio during wake-word detection.
///
/// # Pipeline (encoder-window pipeline)
///
/// 1. **Cooldown gate** — within [`WAKE_WORD_COOLDOWN`] of a detection, audio
///    is only accumulated into the raw ring (capped at [`AUDIO_BUFFER_MAX`]).
/// 2. **Raw ring accumulation** — raw mic samples are appended to
///    [`PipelineCtx::audio_buffer`], capped at [`AUDIO_BUFFER_MAX`].
/// 3. **VAD-gated frame loop** — each [`HOP_LENGTH`] hop is fed to the global
///    earshot VAD detector; per-segment silence is counted and
///    [`PipelineCtx::reset_detection_segment`] fires at
///    [`SEGMENT_TIMEOUT_HOPS`].  No mel frames are built — the raw ring is
///    the encoder input.
/// 4. **Window encoding + scoring** — when [`SCORE_STRIDE_SAMPLES`] samples
///    have been processed since the last scoring step AND speech was seen in
///    this call (or the rolling window is mid-utterance), the trailing ≤1 s
///    of the ring is encoded through the shared Qwen3-ASR encoder
///    ([`crate::audio::wake_word::encode_window`]) and scored via
///    [`score_single_embedding`].
/// 5. **Detection→recording handoff** — on detection the ring is moved into
///    [`PipelineCtx::command_buffer`] with a Soft reset so recording starts
///    with the pre-wake context.
#[allow(clippy::too_many_lines)]
pub(crate) fn handle_wake_word_detection(samples: &[f32], ctx: &mut PipelineCtx) {
    // ── Cooldown check ──
    // If we recently detected the wake word, skip ALL processing for this
    // chunk to prevent rapid consecutive false triggers.  During cooldown
    // audio accumulates into audio_buffer (capped at AUDIO_BUFFER_MAX) so
    // that when the cooldown expires the pipeline has data to process
    // immediately.
    if let Some(last) = ctx.last_wake_word_detection
        && last.elapsed() < WAKE_WORD_COOLDOWN
    {
        debug!(
            "Wake word cooldown active ({}ms elapsed)",
            last.elapsed().as_millis()
        );
        ctx.audio_buffer.extend_from_slice(samples);
        if ctx.audio_buffer.len() > AUDIO_BUFFER_MAX {
            let excess = ctx.audio_buffer.len() - AUDIO_BUFFER_MAX;
            ctx.audio_buffer.drain(..excess);
        }
        return;
    }

    // ── Accumulate into the raw audio ring (capped) ──
    ctx.audio_buffer.extend_from_slice(samples);
    if ctx.audio_buffer.len() > AUDIO_BUFFER_MAX {
        let excess = ctx.audio_buffer.len() - AUDIO_BUFFER_MAX;
        ctx.audio_buffer.drain(..excess);
        ctx.vad_cursor = ctx.vad_cursor.saturating_sub(excess);
    }

    // ── VAD-gated frame loop ──
    // Iterate HOP_LENGTH frames over the UNCONSUMED audio (from `vad_cursor`),
    // feeding each frame's NEW HOP_LENGTH samples to the global VAD detector
    // (the 512-sample frame overlaps the previous by 256 samples; feeding the
    // full frame would double-feed overlapping audio and corrupt earshot's
    // internal ring buffer).  No mel frames are built — the raw ring is the
    // encoder input, so the VAD loop only tracks speech presence and
    // per-segment silence.  The ring is NOT drained here: `vad_cursor`
    // advances instead, keeping the trailing ≤1 s window available for the
    // encoder even after every frame has been VAD-checked.
    let mut speech_seen_this_call = false;
    // Side-channel for consecutive VAD-negative hop tracking, seeded with the
    // accumulated count from previous calls so the counter is continuous.
    let mut hop_count = ctx.segment_silence_hops;

    #[cfg(feature = "voice-tests")]
    let mut per_hop_vad: Vec<bool> = Vec::new();

    while ctx.vad_cursor + FRAME_LENGTH <= ctx.audio_buffer.len() {
        let frame = &ctx.audio_buffer[ctx.vad_cursor..ctx.vad_cursor + FRAME_LENGTH];
        let is_speech = is_speech_with_threshold(&frame[..HOP_LENGTH], VAD_THRESHOLD);
        if is_speech {
            speech_seen_this_call = true;
            hop_count = 0;
            #[cfg(feature = "voice-tests")]
            {
                ctx.instrumentation.vad_speech_frames += 1;
            }
            // ── VAD-gated speech window ──
            // Append only the VAD-positive hop to the scoring window so the
            // encoder sees speech-only audio — the same distribution as
            // enrollment's VAD-segmented utterances.  Capped at
            // WAKE_WORD_WINDOW_SAMPLES (drain the front).
            ctx.speech_window.extend_from_slice(&frame[..HOP_LENGTH]);
            if ctx.speech_window.len() > WAKE_WORD_WINDOW_SAMPLES {
                let excess = ctx.speech_window.len() - WAKE_WORD_WINDOW_SAMPLES;
                ctx.speech_window.drain(..excess);
            }
        } else {
            hop_count += 1;
        }
        #[cfg(feature = "voice-tests")]
        per_hop_vad.push(is_speech);
        ctx.vad_cursor += HOP_LENGTH;
        ctx.last_score_sample_count += HOP_LENGTH;
    }

    // ── Segment boundary check ──
    // Resets the rolling score window / adaptive threshold when the
    // consecutive-silence counter reaches SEGMENT_TIMEOUT_HOPS so scores do
    // not accumulate across separate utterances.  Runs BEFORE scoring so a
    // just-fired boundary starts a fresh segment.
    ctx.handle_segment_boundary(hop_count);

    // ── Window encoding + scoring (stride-gated) ──
    // Score when SCORE_STRIDE_SAMPLES have been processed since the last
    // encode AND speech was seen in this call (or the rolling window is
    // non-empty — i.e., mid-utterance continuation after a VAD gap).
    if !ctx.is_recording
        && ctx.last_score_sample_count >= SCORE_STRIDE_SAMPLES
        && (speech_seen_this_call || !ctx.score_window.is_empty())
    {
        ctx.last_score_sample_count = 0;

        // No enrollment installed → skip the scoring step entirely.  The
        // heavy 18-layer encoder forward is gated behind BOTH the enrollment
        // and the model, so a listening session without a deployed wake word
        // does no per-stride encoding work.
        let Some(enrollment) = voice_state().read().unwrap_poison().enrollment.clone() else {
            return;
        };

        let Some(model) = crate::audio::local_transcriber::shared_model_arc() else {
            // ASR not loaded — no scoring this step.  The status is handled
            // elsewhere (LoadingModels / ModelError).
            return;
        };

        // Take the trailing ≤WAKE_WORD_WINDOW_SAMPLES of the VAD-gated speech
        // window.  Scoring speech-only audio (not the raw ring) keeps the
        // cosine distribution aligned with enrollment — raw trailing audio
        // includes context/silence that dilutes the embedding and shifts
        // positives below the calibration floor.
        let start = ctx
            .speech_window
            .len()
            .saturating_sub(WAKE_WORD_WINDOW_SAMPLES);
        let window = &ctx.speech_window[start..];

        let embed_start = Instant::now();
        let embedding = crate::util::with_block_in_place(|| encode_window(&model, window));
        match embedding {
            Ok(embedding) => {
                let elapsed = embed_start.elapsed();
                let elapsed_ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
                TOTAL_EMBEDDING_TIME_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
                EMBEDDINGS_COMPUTED.fetch_add(1, Ordering::Relaxed);
                #[allow(clippy::cast_possible_truncation)]
                let head = EMBEDDING_LATENCY_RING_WRITES.fetch_add(1, Ordering::Relaxed) as usize
                    % EMBEDDING_LATENCY_RING_SIZE;
                EMBEDDING_LATENCY_RING[head].store(elapsed_ns, Ordering::Relaxed);

                let (detected, _rolling_sum, _total_score, _effective_threshold) =
                    score_single_embedding(
                        &embedding,
                        Some(&enrollment),
                        &mut ctx.score_window,
                        Some(&mut ctx.adaptive_threshold),
                        ctx.adaptive_k,
                    );

                // The underscore-prefixed bindings are intentionally unused in
                // the default build (the instrumentation block below is
                // voice-tests-only), so the usage sites carry an expect for
                // clippy::used_underscore_binding instead of renaming them
                // (which would warn about unused variables without voice-tests).
                #[cfg(feature = "voice-tests")]
                #[expect(clippy::used_underscore_binding)]
                {
                    ctx.instrumentation.per_frame_scores.push([
                        _total_score,
                        _rolling_sum,
                        _effective_threshold,
                    ]);
                    ctx.instrumentation
                        .per_frame_embeddings
                        .push(embedding.clone());
                    if _total_score < NO_MATCH_RESET_THRESHOLD {
                        ctx.instrumentation.n_frames_below_reset += 1;
                    }
                    ctx.instrumentation
                        .adaptive_threshold_trajectory
                        .push(_effective_threshold);
                    if _effective_threshold >= ADAPTIVE_CEILING {
                        ctx.instrumentation.ceiling_limited_frames += 1;
                    }
                    if _rolling_sum > ctx.instrumentation.peak_score {
                        ctx.instrumentation.peak_score = _rolling_sum;
                    }
                    if detected {
                        ctx.instrumentation.first_trigger_frame_idx =
                            Some(ctx.instrumentation.per_frame_scores.len() - 1);
                    }
                }

                if detected {
                    // Detection fires immediately — the handoff below
                    // completes the transition to recording mode.
                    ctx.is_recording = true;
                    ctx.last_wake_word_detection = Some(Instant::now());
                    set_status(VoiceStatus::Recording);
                }
            }
            Err(e) => {
                warn!("Wake word window encoding failed: {e}");
            }
        }
    }

    #[cfg(feature = "voice-tests")]
    ctx.instrumentation.per_hop_vad.extend(per_hop_vad);

    // ── Detection→recording handoff ──
    // When detection fires, the raw ring is moved into command_buffer so
    // recording starts with the pre-wake context.  A Soft reset clears the
    // detection state (score window, adaptive threshold, audio buffers)
    // while preserving VAD continuity and the cooldown timestamp.
    if ctx.is_recording {
        let audio = std::mem::take(&mut ctx.audio_buffer);
        ctx.reset_pipeline_state(ResetLevel::Soft);
        ctx.command_buffer.extend_from_slice(&audio);
        ctx.last_score_sample_count = 0;
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
/// is dropped if multiple complete within a single mic frame.  The frame loop maintains lightweight inline state only for
/// side-effect gating (noise RMS capture, negative audio collection, UI
/// status) — utterance construction itself is delegated to the extracted
/// function.
///
/// **Negative collection**: Non-VAD frames captured
/// before the first detected speech (pre-enrollment ambient noise) and during
/// inter-utterance silence (audio between wake word utterances) are accumulated
/// into `ctx.negative_audio_buf`. On the first transition to sustained speech,
/// this buffer is saved as a chunk in `voice_state().negative_audio_chunks`.
/// These real (non-synthetic) negative examples are later encoded for
/// calibration at enrollment finalization.
///
/// **VAD symmetry**: Only VAD-positive frames are accumulated
/// by the extracted function into utterances, mirroring the detection pipeline
/// ([`handle_wake_word_detection`]). This eliminates the asymmetry where
/// enrollment built templates on audio that the detector never processes.
/// Utterance end is detected after [`ENROLLMENT_SILENCE_THRESHOLD_SAMPLES`]
/// consecutive non-VAD-positive frames (~304ms), matching the detection-side
/// segment timeout.
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
        //
        // Feed only the NEW HOP_LENGTH samples (not the full 512-sample
        // frame) to avoid double-feeding overlapping audio to earshot's
        // VAD detector.  Each frame overlaps the previous by 50%
        // (= HOP_LENGTH), so feeding the full frame duplicates 256 samples,
        // corrupting earshot's internal ring buffer, pre-emphasis filter,
        // and feature context.
        //
        // Uses the context-specific enrollment VAD detector
        // to prevent mode-transition state contamination.  Falls back to the
        // global detector if the enrollment VAD is not initialized (defensive).
        let is_speech = if let Some(ref mut det) = ctx.enrollment_vad {
            is_speech_with_detector(&frame[..HOP_LENGTH], det, ctx.vad_threshold)
        } else {
            is_speech_with_threshold(&frame[..HOP_LENGTH], ctx.vad_threshold)
        };
        ctx.frame_vad.push(is_speech);

        if is_speech {
            ctx.vad_positives_in_a_row += 1;
            // Reset the no-speech warning counter on any VAD-positive frame.
            ctx.enrollment_no_speech_frame_count = 0;

            if ctx.vad_positives_in_a_row >= ENROLLMENT_VAD_CONSECUTIVE_REQUIRED {
                // Sustained speech confirmed: perform inline side-effect gating.
                // The extracted [`segment_utterances_by_vad`] function handles
                // utterance-boundary detection and construction;
                // the inline code below only gates real-time side effects that
                // must happen during the frame loop: noise RMS capture,
                // negative audio collection, and UI status updates.

                let was_waiting_for_silence = ctx.utterance_silence_samples > 0;

                // ── Capture noise RMS from raw audio ring ──
                // On the FIRST transition from silence to sustained speech,
                // capture the ambient noise RMS from the raw audio ring (the
                // encoder pipeline applies no AGC, so the raw floor is the
                // true room noise floor).
                let already_had_speech = ctx.utterance_had_speech;
                // ── Save collected ambient audio for negatives ──
                // On the FIRST transition from silence to sustained speech,
                // save the accumulated non-wake-word audio (pre-enrollment
                // ambient noise or inter-utterance silence) as a potential
                // negative calibration example.
                if !already_had_speech {
                    if ctx.negative_audio_buf.len() >= MIN_NEGATIVE_AUDIO_LEN {
                        let mut state = voice_state().write().unwrap_poison();
                        if state.negative_audio_chunks.len() >= MAX_NEGATIVE_AUDIO_CHUNKS {
                            warn!(
                                "negative_audio_chunks at max ({}): discarding oldest chunk \
                                 to cap memory growth",
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
                    let pre_speech_end = ctx.audio_buffer.len().saturating_sub(speech_boundary);
                    if pre_speech_end > 0 {
                        // Shared RMS helper.
                        let rms = crate::util::compute_rms(&ctx.audio_buffer[..pre_speech_end]);
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
                // / processing delays don't affect cutoff consistency.
                ctx.utterance_silence_samples += HOP_LENGTH;

                // Inline silence threshold reset: partially duplicates the
                // extracted function's boundary logic but is needed so that
                // post-utterance silence within the same chunk accumulates
                // into negative_audio_buf rather than being discarded.
                // Snapshot before reset so the UI status gate uses the
                // pre-reset value — after utterance completion the silence
                // samples are 0, which would spuriously trigger the "waiting
                // for silence" status for one frame.
                let silence_ui_check = ctx.utterance_silence_samples;

                if ctx.utterance_silence_samples >= ENROLLMENT_SILENCE_THRESHOLD_SAMPLES {
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
                // Accumulate non-VAD audio for negatives:
                // pre-enrollment ambient noise, inter-utterance silence, or
                // any non-wake-word audio between utterances.  Each frame
                // contributes HOP_LENGTH new samples.
                ctx.negative_audio_buf
                    .extend_from_slice(&frame[..HOP_LENGTH]);

                // Pre-speech silence: increment no-speech counter.  When the
                // count reaches ENROLLMENT_NO_SPEECH_TIMEOUT_FRAMES (~5 seconds
                // of non-VAD audio), show a warning so the user knows to speak
                // louder or move closer (VAD symmetry mitigation).
                ctx.enrollment_no_speech_frame_count += 1;
                // Warn after the derived frame threshold.
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
            let utterance = utterances[new_idx].clone();
            ctx.enrollment_pending.push_back(utterance);

            // Reset inline tracking state for the next utterance.
            ctx.utterance_had_speech = false;
            ctx.utterance_silence_samples = 0;
            ctx.vad_positives_in_a_row = 0;
            ctx.enrollment_no_speech_frame_count = 0;
            // Note: noise_rms_estimate is intentionally NOT reset here.
            // It is consumed by the main loop alongside enrollment_pending.
        }
    }
}

// Phase 3 owner-negative audio processing

/// Append an owner-negative chunk under the [`MAX_OWNER_NEGATIVE_SAMPLES`]
/// memory cap (dropping with a warning otherwise).  `label` distinguishes the
/// final residual buffer from silence-bounded speech segments in the warning.
fn push_owner_negative_chunk(chunk: Vec<f32>, label: &str) {
    let mut state = voice_state().write().unwrap_poison();
    let total_samples: usize = state
        .owner_negative_chunks
        .iter()
        .map(std::vec::Vec::len)
        .sum();
    if total_samples + chunk.len() <= MAX_OWNER_NEGATIVE_SAMPLES {
        state.owner_negative_chunks.push(chunk);
    } else {
        warn!(
            "owner_negative_chunks at capacity ({} samples): \
             dropping {label} chunk of {} samples",
            MAX_OWNER_NEGATIVE_SAMPLES,
            chunk.len(),
        );
    }
}

/// Outcome of one [`process_phase3_frames`] pass.
///
/// Carries the Phase-3 accumulation state forward (so it survives across mic
/// chunks) plus any completed speech segments finalized during the pass.
struct Phase3Progress {
    /// Post-drain watermark: number of samples at the head of the buffer
    /// already fed to the VAD (index of the next unprocessed frame start).
    processed: usize,
    /// Silence-run accumulator (VAD-negative samples since the last
    /// VAD-positive frame or chunk boundary).
    silence_samples: usize,
    /// Monotone 1:1 VAD-positive speech counter.
    negatives_speech_samples: usize,
    /// Completed speech segments finalized this pass (full segments — never
    /// hop slivers).  The caller decides where they land (production pushes
    /// them via [`push_owner_negative_chunk`]).
    completed_chunks: Vec<Vec<f32>>,
}

/// Process new Phase-3 audio frames against caller-supplied VAD decisions.
///
/// Pure Phase-3 frame-processing state machine (mirrors the
/// [`segment_utterances_by_vad`] testability pattern): all accumulation state
/// is threaded through explicitly and the VAD decision per hop is injected,
/// so tests can drive the segmentation deterministically instead of relying
/// on the stateful earshot neural detector on synthetic audio.
///
/// # 1:1 counting guarantee (mahbot-1782 regression)
///
/// Frames are processed starting at the `processed` watermark (not at the
/// buffer head), so each hop reaches the VAD — and the speech counter —
/// exactly once across mic chunks.  The old code restarted at the buffer head
/// every call and only drained when a chunk boundary fired, so during
/// continuous speech every new chunk re-counted all previously buffered
/// frames (quadratic counter → Phase 3 completed after ~2–4 real seconds).
///
/// The counter must stay a monotone 1:1 record of real audio: after the
/// drain/watermark adjustment it neither re-counts (the original bug) nor
/// under-counts (which would silently degrade to the 120 s timeout path).
///
/// # Segment preservation
///
/// The drain is gated on the segment being closed: while no chunk boundary
/// has fired (`segment_start == 0`, an unfinalized segment open at the buffer
/// head), no drain runs, so the segment — and the unprocessed VAD-window
/// overlap — survives across mic chunks intact.  When a boundary fires, the
/// closed segment is pushed as a chunk and the drain removes everything up to
/// the boundary, so the next segment starts at the drained buffer head.  The
/// silence-run accumulator is reset at each chunk boundary so subsequent
/// chunk pushes stay aligned.
///
/// `buf` is the full accumulated audio buffer (the caller extends it with new
/// mic chunks before each call).  Returns the updated state; `buf` is drained
/// in place.
fn process_phase3_frames(
    buf: &mut Vec<f32>,
    processed: usize,
    silence_samples: usize,
    negatives_speech_samples: usize,
    mut vad: impl FnMut(&[f32]) -> bool,
) -> Phase3Progress {
    let len = buf.len();
    // Resume at the watermark: everything before it was fed to the VAD in a
    // previous call and must not be re-counted.
    let mut consumed = processed;
    // Segment start is always 0 on entry: a call that closed a segment (a
    // boundary fired) drained everything up to the boundary, so the next
    // segment always starts at the drained buffer head.
    let mut segment_start = 0;
    let mut silence_samples = silence_samples;
    let mut negatives_speech_samples = negatives_speech_samples;
    let mut completed_chunks: Vec<Vec<f32>> = Vec::new();

    while consumed + FRAME_LENGTH <= len {
        let is_speech = vad(&buf[consumed..consumed + HOP_LENGTH]);

        if is_speech {
            // 1:1 real-audio accounting: this hop contributes exactly
            // HOP_LENGTH samples of VAD-positive speech, exactly once.
            negatives_speech_samples += HOP_LENGTH;
            // Reset silence counter on any VAD-positive frame.
            silence_samples = 0;
        } else {
            silence_samples += HOP_LENGTH;
            // Check for chunk boundary: when silence exceeds ENROLLMENT_SILENCE_THRESHOLD_SAMPLES
            // (aligned to streaming's ~304ms) after sustained speech,
            // finalize the current segment as a chunk.
            if silence_samples >= ENROLLMENT_SILENCE_THRESHOLD_SAMPLES {
                let chunk_end = consumed.saturating_sub(silence_samples);
                if chunk_end > segment_start {
                    completed_chunks.push(buf[segment_start..chunk_end].to_vec());
                }
                // Advance segment_start past the silence boundary so the next
                // segment starts after this silence region.
                segment_start = consumed;
                // Reset the silence run at each boundary so subsequent chunk
                // pushes stay aligned — without the reset the run keeps
                // growing, `chunk_end` drifts below `segment_start`, and the
                // next push is delayed/misaligned.
                silence_samples = 0;
            }
        }

        consumed += HOP_LENGTH;
    }

    // Drain fully processed frames (everything before segment_start) from the
    // audio buffer, preserving the unfinalized speech segment (and the
    // unprocessed VAD-window overlap).  Everything drained is either already
    // pushed as a completed chunk or pure lead-in.  This also closes the
    // drain window: the next call enters with segment_start = 0 and a new
    // segment open at the drained buffer head.
    if segment_start > 0 {
        buf.drain(..segment_start);
        consumed -= segment_start;
    }

    Phase3Progress {
        processed: consumed,
        silence_samples,
        negatives_speech_samples,
        completed_chunks,
    }
}

/// Process incoming audio for Phase 3 owner-negative collection.
///
/// Extends [`PipelineCtx::phase3_audio_buf`] with the new mic chunk and runs
/// the pure [`process_phase3_frames`] state machine, feeding each hop to the
/// Phase-3 VAD (the stateful earshot detector when available, else the global
/// fallback) exactly once.
///
/// Uses independent buffers (`phase3_audio_buf`, `phase3_silence_samples`)
/// that are NOT shared with `negative_audio_buf` or `utterance_silence_samples`.
/// Does NOT accumulate `frame_raw_audio` or `frame_vad` (dead code — only
/// consumed by `segment_utterances_by_vad`).  Does NOT reset
/// `enrollment_no_speech_frame_count` (no-speech warnings suppressed during
/// Phase 3).
///
/// Uses [`ENROLLMENT_SILENCE_THRESHOLD_SAMPLES`] for chunk boundary detection —
/// aligned to streaming's segment timeout, same constant
/// as enrollment utterance end.
///
/// Audio chunks are finalized into `voice_state().owner_negative_chunks` (not
/// `PipelineCtx` — matches ambient negative pattern and survives mic resets),
/// via [`push_owner_negative_chunk`] which enforces the memory cap.
///
/// Updates `negatives_speech_samples` counter with the VAD-positive frames
/// detected this call (1:1 real-audio accounting — see
/// [`process_phase3_frames`]).
fn handle_negative_collection_audio(samples: &[f32], ctx: &mut PipelineCtx) {
    ctx.phase3_audio_buf.extend_from_slice(samples);
    let progress = process_phase3_frames(
        &mut ctx.phase3_audio_buf,
        ctx.phase3_processed,
        ctx.phase3_silence_samples,
        ctx.negatives_speech_samples,
        |hop| {
            if let Some(ref mut det) = ctx.enrollment_vad {
                is_speech_with_detector(hop, det, ctx.vad_threshold)
            } else {
                is_speech_with_threshold(hop, ctx.vad_threshold)
            }
        },
    );
    ctx.phase3_processed = progress.processed;
    ctx.phase3_silence_samples = progress.silence_samples;
    ctx.negatives_speech_samples = progress.negatives_speech_samples;
    for chunk in progress.completed_chunks {
        push_owner_negative_chunk(chunk, "speech");
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::set_env_var;
    use std::sync::{Mutex, MutexGuard};
    use std::time::{Duration, Instant};

    // ── VAD-gated utterance segmentation tests ────────────────────────────
    // These test the pure [`segment_utterances_by_vad`] function with synthetic
    // audio and manually-computed VAD decisions.  No global voice state needed.

    /// Test config with a shorter silence threshold (10 frames ≈ 2560 samples
    /// instead of the default 94 frames ≈ 24000 samples) so tests don't need
    /// prohibitively long audio buffers.  Context padding is 0 — matches
    /// streaming detection which does not prepend/append context padding.
    const TEST_VAD_CONFIG: VadSegmentationConfig = VadSegmentationConfig {
        frame_length: FRAME_LENGTH,
        hop_length: HOP_LENGTH,
        consecutive_required: ENROLLMENT_VAD_CONSECUTIVE_REQUIRED,
        silence_threshold_samples: HOP_LENGTH * 10, // 2560 samples ≈ 10 frames
        context_padding_samples: 0,
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
        // 4 speech frames (≥1 consecutive → sustained immediately) + 10 silence frames (≥10 → threshold).
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
            utterances[0].len() > 0,
            "utterance should contain audio samples"
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
            assert!(utt.len() > 0, "utterance {i} should contain audio samples");
        }
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

    /// The default segmentation config must use the standard pipeline
    /// constants (context padding 0, silence threshold aligned to the
    /// streaming segment timeout).
    #[test]
    fn vad_segmentation_config_defaults() {
        assert_eq!(DEFAULT_VAD_SEGMENTATION_CONFIG.frame_length, FRAME_LENGTH);
        assert_eq!(DEFAULT_VAD_SEGMENTATION_CONFIG.hop_length, HOP_LENGTH);
        assert_eq!(
            DEFAULT_VAD_SEGMENTATION_CONFIG.consecutive_required,
            ENROLLMENT_VAD_CONSECUTIVE_REQUIRED
        );
        assert_eq!(
            DEFAULT_VAD_SEGMENTATION_CONFIG.silence_threshold_samples,
            ENROLLMENT_SILENCE_THRESHOLD_SAMPLES
        );
        assert_eq!(DEFAULT_VAD_SEGMENTATION_CONFIG.context_padding_samples, 0);
        assert_eq!(DEFAULT_VAD_SEGMENTATION_CONFIG.raw_ring_max, RAW_RING_MAX);
    }

    // ── Refractory period tests ──────────────────────────────────────────

    #[test]
    #[serial_test::serial(voice)]
    fn refractory_period_transition_table() {
        let _ = init_global();

        // (case, timer elapsed?, recording?, initial status, expected status check, timer cleared?)
        let cases: [(
            &str,
            bool,
            bool,
            VoiceStatus,
            fn(&VoiceStatus) -> bool,
            bool,
        ); 4] = [
            (
                "elapsed_error_to_listening",
                true,
                false,
                VoiceStatus::Error("test error".to_string()),
                |s| matches!(s, VoiceStatus::Listening),
                true,
            ),
            (
                "elapsed_disabled_stays",
                true,
                false,
                VoiceStatus::Disabled,
                |s| matches!(s, VoiceStatus::Disabled),
                // Timer still cleared — session-level, not status-dependent.
                true,
            ),
            (
                "elapsed_recording_stays_error",
                true,
                true,
                VoiceStatus::Error("test error".to_string()),
                |s| matches!(s, VoiceStatus::Error(_)),
                true,
            ),
            (
                "future_timer_preserved",
                false,
                false,
                VoiceStatus::Error("test error".to_string()),
                |s| matches!(s, VoiceStatus::Error(_)),
                false,
            ),
        ];

        for (name, timer_elapsed, is_recording, initial, expect, timer_cleared) in cases {
            let mut ctx = PipelineCtx::new();
            ctx.is_recording = is_recording;
            ctx.refractory_until = Some(if timer_elapsed {
                Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .expect("1s in the past should not underflow")
            } else {
                Instant::now()
                    .checked_add(Duration::from_secs(60))
                    .expect("60s in the future should not overflow")
            });

            set_status(initial); // re-establish per-iteration global state

            ctx.check_refractory_period();

            assert!(
                expect(&get_status()),
                "case {name}: unexpected status after refractory check",
            );
            assert_eq!(
                ctx.refractory_until.is_none(),
                timer_cleared,
                "case {name}: refractory timer state",
            );
        }
    }

    // ── Mic-button (manual) recording state-machine tests ──────────────
    // These test the StartRecording/StopRecording interactions that do not
    // require a real microphone: reject guards and the resume/teardown
    // transitions in [`PipelineCtx::end_manual_recording`].

    #[test]
    #[serial_test::serial(voice)]
    fn manual_recording_rejects_while_wake_word_recording() {
        let _ = init_global();

        let mut ctx = PipelineCtx::new();
        ctx.is_recording = true; // wake-word recording in progress
        set_status(VoiceStatus::Recording);

        ctx.handle_start_manual_recording();

        // Rejected: no manual recording, status unchanged, no resume flag.
        assert!(!ctx.manual_recording);
        assert!(!ctx.resume_listening_after_recording);
        assert!(matches!(get_status(), VoiceStatus::Recording));
    }

    #[test]
    #[serial_test::serial(voice)]
    fn manual_recording_rejects_during_enrollment() {
        let _ = init_global();

        let mut ctx = PipelineCtx::new();
        ctx.enrollment_mode = true;
        set_status(VoiceStatus::Listening);

        ctx.handle_start_manual_recording();

        assert!(!ctx.manual_recording);
        // Status untouched — the rejection does not mutate global state.
        assert!(matches!(get_status(), VoiceStatus::Listening));
    }

    #[test]
    #[serial_test::serial(voice)]
    fn manual_recording_end_resumes_wake_word_listening() {
        let _ = init_global();

        let mut ctx = PipelineCtx::new();
        ctx.is_listening = true; // wake-word assistant active
        ctx.manual_recording = true;
        ctx.resume_listening_after_recording = true;
        set_status(VoiceStatus::RecordingManual);

        ctx.end_manual_recording();

        // Wake-word listening resumes: status Listening, mic still running.
        assert!(!ctx.manual_recording);
        assert!(!ctx.resume_listening_after_recording);
        assert!(ctx.is_listening);
        assert!(matches!(get_status(), VoiceStatus::Listening));
    }

    #[test]
    #[serial_test::serial(voice)]
    fn manual_recording_end_tears_down_recording_only_mic() {
        let _ = init_global();

        let mut ctx = PipelineCtx::new();
        ctx.manual_recording = true; // recording with a recording-only mic
        set_status(VoiceStatus::RecordingManual);

        ctx.end_manual_recording();

        // No wake-word listening before → mic torn down, voice disabled.
        assert!(!ctx.manual_recording);
        assert!(!ctx.is_listening);
        assert!(ctx.mic_rx.is_none());
        assert!(matches!(get_status(), VoiceStatus::Disabled));
    }

    #[tokio::test]
    #[serial_test::serial(voice)]
    async fn manual_recording_discard_clears_buffer_and_ends() {
        let _ = init_global();

        let mut ctx = PipelineCtx::new();
        ctx.manual_recording = true;
        ctx.command_buffer.extend_from_slice(&[0.1, 0.2, 0.3]);
        set_status(VoiceStatus::RecordingManual);

        ctx.handle_stop_manual_recording(false).await;

        assert!(!ctx.manual_recording);
        assert!(ctx.command_buffer.is_empty());
        assert!(matches!(get_status(), VoiceStatus::Disabled));
    }

    #[test]
    #[serial_test::serial(voice)]
    fn manual_recording_teardown_preserves_auto_start_pending() {
        let _ = init_global();

        let mut ctx = PipelineCtx::new();
        // Voice enabled in config but models still loading → auto-start is
        // pending. A mic-button recording must not cancel it.
        ctx.auto_start_pending = true;
        ctx.manual_recording = true; // recording-only mic
        set_status(VoiceStatus::RecordingManual);

        ctx.end_manual_recording();

        // Teardown's Full reset would clear auto_start_pending; the manual
        // recording lifecycle preserves it so check_auto_start still fires
        // once models become ready.
        assert!(!ctx.manual_recording);
        assert!(ctx.auto_start_pending);
        assert!(!ctx.is_listening);
        assert!(matches!(get_status(), VoiceStatus::Disabled));
    }

    #[test]
    #[serial_test::serial(voice)]
    fn manual_recording_aborted_when_enrollment_starts() {
        let _ = init_global();

        let mut ctx = PipelineCtx::new();
        ctx.is_listening = true; // wake-word mic running
        ctx.manual_recording = true;
        ctx.command_buffer.extend_from_slice(&[0.1, 0.2, 0.3]);
        ctx.resume_listening_after_recording = true;
        set_status(VoiceStatus::RecordingManual);

        // Enrollment owns the mic now: the manual recording is aborted and
        // its buffer discarded so it cannot resume as garbage later. The
        // return value tells the caller a recording was aborted (to surface
        // a discard notice).
        assert!(ctx.handle_start_enrollment("mahbot"));
        assert!(!ctx.manual_recording);
        assert!(!ctx.resume_listening_after_recording);
        assert!(ctx.command_buffer.is_empty());
        assert!(ctx.enrollment_mode);
    }

    // ── Model-status resolution tests ────────────────────────────────────
    // The pure [`resolved_model_status`] helper is the single source of truth
    // for the startup block AND the periodic wake-up — the wake-up bug that
    // stranded the footer on "Loading…" was an asymmetry between the two.
    // No serial marker needed: the helper reads no global state.

    #[test]
    fn resolved_model_status_failure_is_terminal() {
        // Reachable failure state (not loaded, failed) → ModelError — never
        // Disabled, never LoadingModels, regardless of the voice toggle.
        // (loaded+failed together is unreachable: the transcriber's atomic
        // state machine has no Ready → Failed transition.)
        assert!(matches!(
            resolved_model_status(false, true, true),
            VoiceStatus::ModelError
        ));
        assert!(matches!(
            resolved_model_status(false, true, false),
            VoiceStatus::ModelError
        ));
    }

    #[test]
    fn resolved_model_status_loaded_resolves_by_enabled() {
        // Loaded + enabled → Listening (unchanged behavior).
        assert!(matches!(
            resolved_model_status(true, false, true),
            VoiceStatus::Listening
        ));
        // Loaded + disabled → Disabled (the LoadingModels hang fix: the
        // footer indicator must clear once loading finishes).
        assert!(matches!(
            resolved_model_status(true, false, false),
            VoiceStatus::Disabled
        ));
    }

    #[test]
    fn resolved_model_status_still_loading_is_transient() {
        // Neither loaded nor failed → LoadingModels; the periodic wake-up
        // keeps polling until a terminal state is reached.
        assert!(matches!(
            resolved_model_status(false, false, true),
            VoiceStatus::LoadingModels
        ));
        assert!(matches!(
            resolved_model_status(false, false, false),
            VoiceStatus::LoadingModels
        ));
    }

    // ── Rate-limiting debounce tests ─────────────────────────────────────
    // These test the 10-second error-message rate limit via the canonical
    // [`PipelineCtx::should_send_error_message`] method.  No serial marker
    // needed — these only read from [`PipelineCtx`] fields without touching
    // global voice state.

    #[test]
    fn rate_limit_error_message_table() {
        // (case, seconds since last error, expected send decision)
        let cases = [
            ("no_prior_error", None, true),
            ("recent_error", Some(0), false),
            ("old_error_15s", Some(15), true),
            ("exact_threshold_10s", Some(10), true),
            ("just_below_9s", Some(9), false),
        ];

        for (name, elapsed_secs, expected) in cases {
            let mut ctx = PipelineCtx::new();
            ctx.last_error_message_time = elapsed_secs.map(|secs| {
                Instant::now()
                    .checked_sub(Duration::from_secs(secs))
                    .expect("elapsed seconds in the past should not underflow")
            });
            assert_eq!(
                ctx.should_send_error_message(),
                expected,
                "case {name}: 10s error-message rate limit",
            );
        }
    }

    #[test]
    fn voice_notice_limiter_is_independent_of_error_limiter() {
        let mut ctx = PipelineCtx::new();
        // A recent transcription error must NOT suppress a discard notice
        // (they use separate limiters so one can never starve the other).
        ctx.last_error_message_time = Some(Instant::now());
        assert!(ctx.should_send_voice_notice());

        ctx.last_voice_notice_time = Some(Instant::now());
        assert!(!ctx.should_send_voice_notice());
    }

    // ── AdaptiveThresholdState tests ──────────────────────────────────────
    // Pure unit tests for the z-score adaptive threshold tracker.  Uses
    // synthetic per-frame scores — no models or voice pipeline state.
    // Covers bootstrap phase, mean/std computation, all safeguards, and reset.

    #[test]
    fn adaptive_after_bootstrap_returns_some() {
        let mut state = AdaptiveThresholdState::new();
        for i in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            assert!(
                state.feed(0.5, ADAPTIVE_K_DEFAULT).is_none(),
                "frame {i} should return None during bootstrap",
            );
        }
        let result = state.feed(0.5, ADAPTIVE_K_DEFAULT);
        assert!(result.is_some(), "should return Some after bootstrap");
    }

    #[test]
    fn adaptive_safe_harbor_enforced() {
        // Low, constant scores produce a low adaptive value that must be
        // overridden by the safe harbor.
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.1, ADAPTIVE_K_DEFAULT);
        }
        // adaptive = (0.1 + 2.5 × 0.0) × 3 = 0.3 → well below safe harbor (1.65)
        let result = state.feed(0.1, ADAPTIVE_K_DEFAULT);
        let threshold = result.expect("should return Some after bootstrap");
        assert!(
            (threshold - ADAPTIVE_SAFE_HARBOR).abs() < 0.01,
            "with constant low score, threshold {threshold} should equal safe harbor {}",
            ADAPTIVE_SAFE_HARBOR,
        );
    }

    #[test]
    fn adaptive_ceiling_enforced() {
        // Very high-variance scores produce a high adaptive value that
        // should be capped by the ceiling.  With alternating 1.0/0.0
        // scores: mean=0.5, std≈0.5, k=2.5:
        //   adaptive = (0.5 + 2.5 × 0.5) × 3 = 5.25
        // Capped by ceiling 2.60 (re-calibrated for the cosine soft space).
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.99, ADAPTIVE_K_DEFAULT);
        }
        // Fill window with alternating 1.0/0.0 scores to create variance.
        for i in ADAPTIVE_BOOTSTRAP_FRAMES..ADAPTIVE_WINDOW_N {
            let score = if i % 2 == 0 { 1.0 } else { 0.0 };
            state.feed(score, ADAPTIVE_K_DEFAULT);
        }
        let result = state.feed(1.0, ADAPTIVE_K_DEFAULT);
        let threshold = result.expect("should return Some after bootstrap");
        assert!(
            (threshold - ADAPTIVE_CEILING).abs() < 0.01,
            "with high-variance scores, threshold {threshold} should equal ceiling {}",
            ADAPTIVE_CEILING,
        );
    }

    #[test]
    fn adaptive_reset_clears_state() {
        let mut state = AdaptiveThresholdState::new();
        // Advance past bootstrap.
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.5, ADAPTIVE_K_DEFAULT);
        }
        assert!(state.feed(0.5, ADAPTIVE_K_DEFAULT).is_some());
        assert_eq!(state.len(), ADAPTIVE_BOOTSTRAP_FRAMES + 1);

        state.reset();

        assert_eq!(state.len(), 0, "window should be empty after reset");
        assert!(
            state.feed(0.5, ADAPTIVE_K_DEFAULT).is_none(),
            "after reset, first feed should return None (re-enters bootstrap)",
        );
    }

    #[test]
    fn adaptive_warmed_clamps_to_safe_harbor() {
        // warmed() initializes with near-silence scores (~0.033).
        // The computed adaptive threshold (0.033 + 2.5 × 0.0) × 3 = 0.099
        // should be clamped to the safe harbor (1.65), matching production
        // where the threshold is fed real silence/background scores.
        let mut state = AdaptiveThresholdState::warmed();
        let threshold = state
            .feed(0.033, ADAPTIVE_K_DEFAULT)
            .expect("warmed() should exit bootstrap");
        assert!(
            (threshold - ADAPTIVE_SAFE_HARBOR).abs() < 0.01,
            "warmed() threshold {threshold} should equal safe harbor {}",
            ADAPTIVE_SAFE_HARBOR,
        );
        // Verify that all bootstrap frames were fed the near-silence score
        // and not 0.5 (which would produce threshold ~1.5 instead of 1.65).
        assert_eq!(state.len(), ADAPTIVE_BOOTSTRAP_FRAMES + 1);
    }

    #[test]
    fn adaptive_window_eviction_correctness() {
        // After filling the window and cycling scores, verify the sum/sum_sq
        // statistics produce the correct mean.
        let mut state = AdaptiveThresholdState::new();
        // Bootstrap with 0.5.
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.5, ADAPTIVE_K_DEFAULT);
        }
        // Fill remaining window slots with 0.5.
        for _ in ADAPTIVE_BOOTSTRAP_FRAMES..ADAPTIVE_WINDOW_N {
            state.feed(0.5, ADAPTIVE_K_DEFAULT);
        }
        // Window now has ADAPTIVE_WINDOW_N entries, all 0.5.
        // Feed 1.0 to trigger eviction of oldest (0.5).
        // New window: (ADAPTIVE_WINDOW_N - 1) × 0.5 + 1 × 1.0
        // mean = ((ADAPTIVE_WINDOW_N - 1) * 0.5 + 1.0) / ADAPTIVE_WINDOW_N
        let expected_mean = (ADAPTIVE_WINDOW_N as f32 - 1.0) * 0.5 / ADAPTIVE_WINDOW_N as f32
            + 1.0 / ADAPTIVE_WINDOW_N as f32;
        let result = state.feed(1.0, 0.0); // k=0 so adaptive = mean × ROLLING_WINDOW_N
        let threshold = result.expect("should return Some");
        // After eviction: mean ≈ 0.533, adaptive = 0.533 * 3 = 1.6, clamped
        // to the safe harbor 1.65.
        let expected_raw = expected_mean * ROLLING_WINDOW_N as f32;
        let clamped = expected_raw.clamp(ADAPTIVE_SAFE_HARBOR, ADAPTIVE_CEILING);
        assert!(
            (threshold - clamped).abs() < 0.001,
            "threshold {threshold} should match expected clamped value {clamped} (raw={expected_raw})",
        );
    }

    // ── AdaptiveThresholdState::peek() tests ──────────────────────────────
    // Tests for the peek() method which returns the current threshold without
    // updating statistics.  Covers bootstrap guard, empty-window check,
    // threshold correctness, and the no-mutation invariant.

    #[test]
    fn adaptive_peek_bootstrap_boundary_and_safe_harbor() {
        // peek() should return None during bootstrap, just like feed().
        // On the last bootstrap frame, feed() increments bootstrap_count
        // past the threshold, so peek() returns Some (bootstrap done).
        let mut state = AdaptiveThresholdState::new();
        // First ADAPTIVE_BOOTSTRAP_FRAMES - 1 frames: both feed and peek return None.
        for i in 0..ADAPTIVE_BOOTSTRAP_FRAMES - 1 {
            assert!(
                state.feed(0.5, ADAPTIVE_K_DEFAULT).is_none(),
                "feed frame {i} should be None during bootstrap",
            );
            assert!(
                state.peek(ADAPTIVE_K_DEFAULT).is_none(),
                "peek frame {i} should be None during bootstrap",
            );
        }
        // Last bootstrap frame: feed returns None (completes bootstrap),
        // but peek returns Some because feed already incremented bootstrap_count.
        assert!(
            state.feed(0.5, ADAPTIVE_K_DEFAULT).is_none(),
            "last feed during bootstrap should return None",
        );
        assert!(
            state.peek(ADAPTIVE_K_DEFAULT).is_some(),
            "peek should return Some after bootstrap is complete",
        );

        // After bootstrap completes, peek() returns a threshold.  Use low
        // scores (0.1) so the computed adaptive value (~0.3) stays below the
        // safe harbor (1.65), verifying that peek() produces a clamped
        // threshold rather than a raw adaptive value.
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.1, ADAPTIVE_K_DEFAULT);
        }
        let threshold = state
            .peek(ADAPTIVE_K_DEFAULT)
            .expect("peek should return Some after bootstrap");
        assert!(
            (threshold - ADAPTIVE_SAFE_HARBOR).abs() < 0.01,
            "peek threshold {threshold} should equal safe harbor {} with constant low input",
            ADAPTIVE_SAFE_HARBOR,
        );
    }

    #[test]
    fn adaptive_peek_does_not_mutate_state() {
        // Calling peek() must not change scores, sum, sum_sq, or bootstrap_count.
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.5, ADAPTIVE_K_DEFAULT);
        }
        // Feed one more to have a non-bootstrap state.
        state.feed(0.5, ADAPTIVE_K_DEFAULT);

        let before_scores = state.scores.clone();
        let before_sum = state.sum;
        let before_sum_sq = state.sum_sq;
        let before_bootstrap = state.bootstrap_count;

        // Call peek multiple times.
        for _ in 0..3 {
            let _ = state.peek(ADAPTIVE_K_DEFAULT);
        }

        assert_eq!(state.scores, before_scores, "peek must not modify scores",);
        assert!(
            (state.sum - before_sum).abs() < f32::EPSILON,
            "peek must not modify sum",
        );
        assert!(
            (state.sum_sq - before_sum_sq).abs() < f32::EPSILON,
            "peek must not modify sum_sq",
        );
        assert_eq!(
            state.bootstrap_count, before_bootstrap,
            "peek must not modify bootstrap_count",
        );
    }

    #[test]
    fn adaptive_peek_empty_after_reset() {
        // After reset, peek() must return None (empty window).
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.5, ADAPTIVE_K_DEFAULT);
        }
        assert!(state.peek(ADAPTIVE_K_DEFAULT).is_some());
        state.reset();
        assert!(state.peek(ADAPTIVE_K_DEFAULT).is_none());
    }

    #[test]
    fn adaptive_peek_threshold_in_valid_range() {
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.5, ADAPTIVE_K_DEFAULT);
        }
        let threshold = state
            .peek(ADAPTIVE_K_DEFAULT)
            .expect("Some after bootstrap");
        assert!(
            threshold >= ADAPTIVE_SAFE_HARBOR && threshold <= ADAPTIVE_CEILING,
            "peek threshold {threshold} must be within [{ADAPTIVE_SAFE_HARBOR}, {ADAPTIVE_CEILING}]",
        );
    }

    #[test]
    fn adaptive_peek_agrees_with_feed_on_same_state() {
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.5, ADAPTIVE_K_DEFAULT);
        }
        state.feed(0.7, ADAPTIVE_K_DEFAULT);
        let feed_threshold = state.feed(0.3, ADAPTIVE_K_DEFAULT);
        let peek_threshold = state.peek(ADAPTIVE_K_DEFAULT);
        assert_eq!(feed_threshold, peek_threshold);
    }

    // ── process_wake_word_score tests ───────────────────────────────────
    // Pure rolling-window scoring — no models, no pipeline state.

    #[test]
    fn score_window_resets_below_no_match_threshold() {
        let mut window = vec![0.9, 0.8, 0.7];
        let (detected, rolling) =
            process_wake_word_score(NO_MATCH_RESET_THRESHOLD - 0.01, &mut window, None, false);
        assert!(!detected);
        assert_eq!(rolling, 0.0);
        assert!(
            window.is_empty(),
            "below-reset score must clear the rolling window"
        );
    }

    #[test]
    fn score_window_accumulates_and_fires() {
        // match_threshold = 1.65 → need 3 frames summing to ≥1.65.
        let mut window = Vec::new();
        for score in [0.7, 0.7] {
            let (detected, _) = process_wake_word_score(score, &mut window, None, false);
            assert!(!detected, "2 frames of 0.7 must not fire (sum 1.4 < 1.65)");
        }
        let (detected, rolling) = process_wake_word_score(0.7, &mut window, None, false);
        assert!(detected, "3 frames of 0.7 must fire (sum 2.1 ≥ 1.65)");
        assert!(
            (rolling - 2.1).abs() < 1e-6,
            "rolling sum should be 2.1, got {rolling}"
        );
    }

    #[test]
    fn score_window_preserve_on_reset_keeps_window() {
        let mut window = vec![0.9, 0.8];
        let (detected, _) = process_wake_word_score(0.1, &mut window, None, true);
        assert!(!detected);
        assert_eq!(
            window.len(),
            2,
            "preserve_window_on_reset must not clear the window"
        );
    }

    #[test]
    fn score_window_adaptive_override_used() {
        let mut window = Vec::new();
        // A high adaptive override (3.0) must suppress detection even when
        // three 0.7 frames would otherwise fire (sum 2.1 < 3.0).
        for _ in 0..3 {
            let (detected, _) = process_wake_word_score(0.7, &mut window, Some(3.0), false);
            assert!(!detected, "adaptive override must raise the detection bar");
        }
    }

    // ── score_single_embedding (cosine-prototype) tests ─────────────────
    // The new encoder pipeline scores each window embedding as the cosine
    // soft score against the enrolled prototype.  These tests use synthetic
    // L2-normalized embeddings and a synthetic enrollment.

    /// L2-normalize a test vector.
    fn norm_embedding(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0, "test embedding must be non-zero");
        v.iter().map(|x| x / norm).collect()
    }

    /// Basis embedding: dim `d` = 1.0, everything else 0.0 (unit norm).
    fn basis_embedding(d: usize) -> Vec<f32> {
        let mut v = vec![0.0; WAKE_WORD_EMBEDDING_DIM];
        if d < WAKE_WORD_EMBEDDING_DIM {
            v[d] = 1.0;
        }
        v
    }

    /// Build a synthetic enrollment whose prototype points along basis dim 0.
    fn synthetic_enrollment(n_utterances: usize) -> WakeWordEnrollment {
        let emb = basis_embedding(0);
        WakeWordEnrollment::build(
            "mahbot".to_string(),
            &vec![emb; n_utterances],
            crate::audio::wake_word::Calibration::default(),
            &[],
            String::new(),
            String::new(),
        )
        .expect("synthetic enrollment")
    }

    #[test]
    fn score_single_no_enrollment_returns_zero_score() {
        // Without an enrollment no detection is possible: total_score = 0.0
        // and the window is reset (0.0 < NO_MATCH_RESET_THRESHOLD).
        let mut window = vec![0.9, 0.8];
        let emb = norm_embedding(&vec![1.0; WAKE_WORD_EMBEDDING_DIM]);
        let (detected, rolling, total, _) =
            score_single_embedding(&emb, None, &mut window, None, ADAPTIVE_K_DEFAULT);
        assert!(!detected);
        assert_eq!(total, 0.0);
        assert_eq!(rolling, 0.0);
        assert!(
            window.is_empty(),
            "zero score must reset the rolling window"
        );
    }

    #[test]
    fn score_single_detects_high_cosine_match() {
        // A perfect match against the prototype (cosine 1.0 → soft score 1.0
        // with the default floor) must fire detection after 3 frames.
        let enrollment = synthetic_enrollment(MIN_ENROLLMENT_UTTERANCES);
        let mut window = Vec::new();
        let mut adaptive = AdaptiveThresholdState::new();
        let emb = basis_embedding(0);
        let mut detected = false;
        for _ in 0..ROLLING_WINDOW_N {
            let (d, _, total, _) = score_single_embedding(
                &emb,
                Some(&enrollment),
                &mut window,
                Some(&mut adaptive),
                ADAPTIVE_K_DEFAULT,
            );
            assert!(
                total >= NO_MATCH_RESET_THRESHOLD,
                "a prototype match must score above the reset threshold"
            );
            detected = d;
        }
        assert!(
            detected,
            "3 consecutive high-cosine frames must fire detection"
        );
    }

    #[test]
    fn score_single_adaptive_feeds_background_only() {
        // Scores below NO_MATCH_RESET_THRESHOLD feed the adaptive statistics;
        // wake-word-like scores only peek (no statistics mutation).
        let enrollment = synthetic_enrollment(MIN_ENROLLMENT_UTTERANCES);
        let mut window = Vec::new();
        let mut adaptive = AdaptiveThresholdState::new();
        // An orthogonal embedding (basis 1 vs prototype basis 0) → cosine 0
        // → soft score 0.0 (clamped) → below reset → feeds.
        let bg = basis_embedding(1);
        let (detected, _, total, _) = score_single_embedding(
            &bg,
            Some(&enrollment),
            &mut window,
            Some(&mut adaptive),
            ADAPTIVE_K_DEFAULT,
        );
        assert!(!detected);
        assert!(total < NO_MATCH_RESET_THRESHOLD);
        assert!(window.is_empty(), "below-reset score clears the window");
        assert_eq!(
            adaptive.len(),
            1,
            "background score must feed the adaptive window"
        );
    }

    #[test]
    fn score_single_low_score_resets_window() {
        // A below-reset frame mid-utterance clears the rolling window
        // (unless preserve_window_on_reset, which the encoder pipeline never
        // sets).
        let enrollment = synthetic_enrollment(MIN_ENROLLMENT_UTTERANCES);
        let mut window = vec![0.9, 0.8];
        let bg = basis_embedding(1);
        let (detected, rolling, _, _) = score_single_embedding(
            &bg,
            Some(&enrollment),
            &mut window,
            None,
            ADAPTIVE_K_DEFAULT,
        );
        assert!(!detected);
        assert_eq!(rolling, 0.0);
        assert!(window.is_empty(), "low score must reset the rolling window");
    }

    // ── Enrollment consistency gate tests ────────────────────────────────
    // The centroid gate is a pure function so it is testable without the ASR
    // model.  Uses basis embeddings: utterances on basis 0 form a consistent
    // cluster; utterances on other bases are outliers.

    #[test]
    fn consistency_gate_fails_when_too_few_utterances() {
        let embs: Vec<Vec<f32>> = vec![basis_embedding(0); 4]; // < 5 utterances
        let err = enrollment_consistency_check(&embs).unwrap_err();
        assert!(
            err.contains("4 enrollment utterances"),
            "error should mention the utterance count: {err}"
        );
    }

    #[test]
    fn consistency_gate_fails_when_too_few_pass_threshold() {
        // 6 on basis 0 + 4 on basis 1: 6/10 pass → need ceil(10 × 0.7) = 7 → fail.
        let mut embs: Vec<Vec<f32>> = vec![basis_embedding(0); 6];
        embs.extend(vec![basis_embedding(1); 4]);
        let err = enrollment_consistency_check(&embs).unwrap_err();
        assert!(
            err.contains("6/10"),
            "error should report 6/10 passed: {err}"
        );
    }

    #[test]
    fn consistency_gate_succeeds_with_high_quality_utterances() {
        // 8 on basis 0 + 2 on basis 1: 8/10 pass → need ceil(10 × 0.7) = 7 → pass.
        let mut embs: Vec<Vec<f32>> = vec![basis_embedding(0); 8];
        embs.extend(vec![basis_embedding(1); 2]);
        assert!(
            enrollment_consistency_check(&embs).is_ok(),
            "8/10 consistent utterances must pass the gate"
        );
    }

    #[test]
    fn consistency_gate_rejects_wrong_dimension() {
        let mut embs: Vec<Vec<f32>> = vec![basis_embedding(0); MIN_ENROLLMENT_UTTERANCES];
        embs[0] = vec![0.5; 64]; // wrong dim
        let err = enrollment_consistency_check(&embs).unwrap_err();
        assert!(
            err.contains("expected 1024"),
            "error should mention the dimension mismatch: {err}"
        );
    }

    // ── PCM cache key tests ─────────────────────────────────────────────────

    #[test]
    fn test_pcm_cache_key_determinism() {
        let h = |text, style, seed| pcm_cache_key(text, style, seed, 16000, "test_hash");
        let a = h("hey mahbot", "default", 42);
        let b = h("hey mahbot", "default", 42);
        assert_eq!(a, b, "same inputs must produce same cache key");
    }

    #[test]
    fn test_pcm_cache_key_sensitivity_to_text() {
        let a = pcm_cache_key("hey mahbot", "default", 42, 16000, "hash");
        let b = pcm_cache_key("hey jarvis", "default", 42, 16000, "hash");
        assert_ne!(a, b, "different text must produce different cache keys");
    }

    #[test]
    fn test_pcm_cache_key_sensitivity_to_seed() {
        let a = pcm_cache_key("hey mahbot", "default", 41, 16000, "hash");
        let b = pcm_cache_key("hey mahbot", "default", 42, 16000, "hash");
        assert_ne!(a, b, "different seed must produce different cache keys");
    }

    #[test]
    fn test_pcm_cache_key_sensitivity_to_model_hash() {
        let a = pcm_cache_key("hey mahbot", "default", 42, 16000, "hash_a");
        let b = pcm_cache_key("hey mahbot", "default", 42, 16000, "hash_b");
        assert_ne!(
            a, b,
            "different model hash must produce different cache keys"
        );
    }

    #[test]
    fn test_tts_model_version_hash_is_non_empty() {
        let hash = tts_model_version_hash();
        assert_eq!(hash.len(), 64, "SHA-256 hex is 64 chars");
        assert!(
            hash.chars().all(|c| c.is_ascii_hexdigit()),
            "hash must be hex: {hash}"
        );
    }

    // ── Enrollment state tests ─────────────────────────────────────────────

    #[test]
    #[serial_test::serial(voice)]
    fn test_enrolled_utterance_count_tracks_utterances() {
        // enrolled_utterance_count tracks user utterances while
        // enrollment_embeddings holds exactly one 1024-dim embedding per
        // utterance (no augmentation in the encoder pipeline).
        let _ = init_global();
        let mut state = voice_state().write().unwrap_poison();
        state.reset_enrollment(); // ensure clean state from prior serial tests
        assert_eq!(state.enrollment_embeddings.len(), 0);
        assert_eq!(state.enrolled_utterance_count, 0);

        state.enrollment_embeddings.push(basis_embedding(0));
        state.enrolled_utterance_count = 1;

        assert_eq!(state.enrolled_utterance_count, 1);
        assert_eq!(state.enrollment_embeddings.len(), 1);

        // Cancel reset clears both.
        state.reset_enrollment();
        assert_eq!(state.enrolled_utterance_count, 0);
        assert!(state.enrollment_embeddings.is_empty());
    }

    // ── Voice metrics tests ────────────────────────────────────────────────

    #[test]
    fn voice_metrics_empty_snapshot_has_zero_average() {
        // Snapshot with zero embeddings should report 0ns average and 0%
        // drop rate, not NaN or a panic from division by zero.
        let snap = VoiceMetricsSnapshot {
            chunks_received: 0,
            dropped_chunks: 0,
            embeddings_computed: 0,
            total_embedding_time_ns: 0,
            avg_embedding_latency_ns: 0,
        };
        assert_eq!(snap.lifetime_avg_embedding_latency_ns(), 0);
        assert_eq!(snap.avg_embedding_latency_ns, 0);
        assert_eq!(snap.drop_rate(), 0.0);
    }

    #[test]
    fn voice_metrics_drop_rate_saturates() {
        // With no chunks received and 0 dropped, drop_rate should be 0.0,
        // not NaN or panic.
        let snap = VoiceMetricsSnapshot {
            chunks_received: 0,
            dropped_chunks: 0,
            embeddings_computed: 0,
            total_embedding_time_ns: 0,
            avg_embedding_latency_ns: 0,
        };
        assert_eq!(snap.drop_rate(), 0.0);
        // All chunks dropped → drop_rate = 1.0
        let snap = VoiceMetricsSnapshot {
            chunks_received: 0,
            dropped_chunks: 42,
            ..snap
        };
        assert!((snap.drop_rate() - 1.0).abs() < f64::EPSILON);
        // Half dropped → drop_rate = 0.5
        let snap = VoiceMetricsSnapshot {
            chunks_received: 50,
            dropped_chunks: 50,
            ..snap
        };
        assert!((snap.drop_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn voice_metrics_lifetime_average_computation() {
        let snap = VoiceMetricsSnapshot {
            chunks_received: 0,
            dropped_chunks: 0,
            embeddings_computed: 10,
            total_embedding_time_ns: 1_000_000, // 1ms total for 10 embeddings
            avg_embedding_latency_ns: 0,
        };
        assert_eq!(
            snap.lifetime_avg_embedding_latency_ns(),
            100_000,
            "1ms / 10 embeddings = 100µs per embedding"
        );
    }

    // ── samples_to_ms / quality helpers ────────────────────────────────────

    #[test]
    fn samples_to_ms_conversion() {
        assert_eq!(samples_to_ms(0, SAMPLE_RATE), 0);
        assert_eq!(samples_to_ms(16_000, SAMPLE_RATE), 1000);
        assert_eq!(samples_to_ms(400, SAMPLE_RATE), 25);
        // Truncating integer division — 1 sample = 0ms.
        assert_eq!(samples_to_ms(1, SAMPLE_RATE), 0);
    }

    #[test]
    fn compute_utterance_quality_short_utterance_scores_zero_duration() {
        // A 100ms blip is below the 400ms floor → duration score 0.  The
        // composite is still non-zero (clipping + SNR components), matching
        // the weighted formula: 0.0×0.5 + 1.0×0.25 + 0.5×0.25 = 0.25.
        let samples = vec![0.1; 1600]; // 100ms
        let q = compute_utterance_quality(&samples, None);
        assert_eq!(q.duration_ms, 100);
        assert!((q.score - 0.25).abs() < 1e-6, "score {}", q.score);
        assert_eq!(q.level, QualityLevel::Poor);
    }

    // ── normalize_phrase / wake word phrase state machine tests ──────────

    #[test]
    fn normalize_phrase_trims_lowercases_collapses() {
        assert_eq!(normalize_phrase("  HeY  MahBot  "), "hey mahbot");
        assert_eq!(normalize_phrase("OK   Computer"), "ok computer");
        assert_eq!(normalize_phrase("  hello  WORLD  "), "hello world");
        assert_eq!(normalize_phrase("singleword"), "singleword");
        assert_eq!(normalize_phrase("  already  fine  "), "already fine");
    }

    #[test]
    fn normalize_phrase_empty_returns_default() {
        assert_eq!(normalize_phrase(""), DEFAULT_WAKE_WORD_PHRASE);
        assert_eq!(normalize_phrase("   "), DEFAULT_WAKE_WORD_PHRASE);
        assert_eq!(normalize_phrase("\t\n"), DEFAULT_WAKE_WORD_PHRASE);
    }

    #[test]
    #[serial_test::serial(voice)]
    fn get_enrolled_phrase_returns_none_initially() {
        let _ = init_global();
        // Ensure clean state (another serial test may have set model_phrase)
        voice_state().write().unwrap_poison().model_phrase = None;
        assert!(get_enrolled_phrase().is_none());
    }

    #[test]
    #[serial_test::serial(voice)]
    fn get_enrolled_phrase_returns_phrase_after_set() {
        let _ = init_global();
        {
            let mut state = voice_state().write().unwrap_poison();
            state.model_phrase = Some("hey mahbot".to_string());
        }
        assert_eq!(get_enrolled_phrase(), Some("hey mahbot".to_string()));
        // Verify idempotent read
        assert_eq!(get_enrolled_phrase(), Some("hey mahbot".to_string()));
    }

    #[test]
    #[serial_test::serial(voice)]
    fn model_phrase_survives_enrollment_cancel() {
        let _ = init_global();

        // Set model_phrase (enrolled model) and enrolling_phrase (in-progress)
        {
            let mut state = voice_state().write().unwrap_poison();
            state.model_phrase = Some("hey computer".to_string());
            state.enrolling_phrase = Some("new phrase".to_string());
        }

        // Cancel enrollment — calls VoicePipelineState::reset_enrollment()
        {
            let mut state = voice_state().write().unwrap_poison();
            state.reset_enrollment();
        }

        // model_phrase must be preserved, enrolling_phrase cleared
        let state = voice_state().read().unwrap_poison();
        assert_eq!(
            state.model_phrase,
            Some("hey computer".to_string()),
            "model_phrase must survive enrollment cancel"
        );
        assert!(
            state.enrolling_phrase.is_none(),
            "enrolling_phrase must be cleared on enrollment cancel"
        );

        // get_enrolled_phrase() must still return the cached phrase
        assert_eq!(get_enrolled_phrase(), Some("hey computer".to_string()),);
    }

    // ── PipelineCtx adaptive_k clamping ───────────────────────────────────
    // PipelineCtx::new() reads adaptive_k from CONFIG and clamps it to
    // [ADAPTIVE_K_MIN, ADAPTIVE_K_MAX].  Verify this works for out-of-range
    // config values.

    #[test]
    #[serial_test::serial(config)]
    fn adaptive_k_clamped_below_min() {
        let _ = crate::config::CONFIG.set_string_field("adaptive_k", "0.1");
        let ctx = PipelineCtx::new();
        assert!(
            (ctx.adaptive_k - ADAPTIVE_K_MIN).abs() < 0.01,
            "adaptive_k {} should be clamped to min {}",
            ctx.adaptive_k,
            ADAPTIVE_K_MIN,
        );
    }

    #[test]
    #[serial_test::serial(config)]
    fn adaptive_k_clamped_above_max() {
        let _ = crate::config::CONFIG.set_string_field("adaptive_k", "9.9");
        let ctx = PipelineCtx::new();
        assert!(
            (ctx.adaptive_k - ADAPTIVE_K_MAX).abs() < 0.01,
            "adaptive_k {} should be clamped to max {}",
            ctx.adaptive_k,
            ADAPTIVE_K_MAX,
        );
    }

    #[test]
    #[serial_test::serial(config)]
    fn adaptive_k_uses_default_when_config_empty() {
        // Setting to empty string should cause parse failure, falling back
        // to ADAPTIVE_K_DEFAULT.
        let _ = crate::config::CONFIG.set_string_field("adaptive_k", "");
        let ctx = PipelineCtx::new();
        assert!(
            (ctx.adaptive_k - ADAPTIVE_K_DEFAULT).abs() < 0.01,
            "adaptive_k {} should be default {} when config is empty",
            ctx.adaptive_k,
            ADAPTIVE_K_DEFAULT,
        );
    }

    // ── reset_pipeline_state level tests ──────────────────────────────────
    // These test the three ResetLevel variants against a PipelineCtx with
    // non-default field values.  Tests that touch global voice state (Full,
    // Cancel) use #[serial_test::serial(voice)].

    /// Helper: build a PipelineCtx with non-default values in all mutable
    /// fields that reset_pipeline_state may touch.
    fn ctx_with_populated_buffers() -> PipelineCtx {
        let mut ctx = PipelineCtx::new();
        ctx.audio_buffer = vec![0.5; 100];
        ctx.command_buffer = vec![0.5; 100];
        ctx.silence_sample_count = 1000;
        ctx.score_window = vec![0.5; 5];
        ctx.last_score_sample_count = 512;
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
        ctx.collecting_negatives = true;
        ctx.phase3_audio_buf = vec![0.5; 100];
        ctx.phase3_silence_samples = 500;
        ctx.negatives_speech_samples = 1000;
        ctx.phase3_processed = 1234;
        ctx.phase3_start_time = Some(Instant::now() - Duration::from_secs(10));
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
        assert!(ctx.audio_buffer.is_empty());
        assert!(ctx.command_buffer.is_empty());
        assert_eq!(ctx.silence_sample_count, 0);
        assert!(ctx.score_window.is_empty());
        assert_eq!(ctx.last_score_sample_count, 0);
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

        // Segment boundary tracking.
        assert_eq!(
            ctx.segment_silence_hops, 0,
            "segment_silence_hops must be cleared by all reset levels"
        );

        // Phase 3 owner-negative state.
        assert!(
            !ctx.collecting_negatives,
            "collecting_negatives must be false after reset"
        );
        assert!(
            ctx.phase3_audio_buf.is_empty(),
            "phase3_audio_buf must be cleared by all reset levels"
        );
        assert_eq!(
            ctx.phase3_silence_samples, 0,
            "phase3_silence_samples must be cleared by all reset levels"
        );
        assert_eq!(
            ctx.negatives_speech_samples, 0,
            "negatives_speech_samples must be cleared by all reset levels"
        );
        assert_eq!(
            ctx.phase3_processed, 0,
            "phase3_processed must be cleared by all reset levels"
        );
        assert!(
            ctx.phase3_start_time.is_none(),
            "phase3_start_time must be None after reset"
        );
    }

    #[test]
    #[serial_test::serial(voice)]
    fn reset_full_clears_all_buffers_and_state() {
        let _ = init_global();
        let mut ctx = ctx_with_populated_buffers();

        // Pre-populate global enrollment state.
        {
            let mut state = voice_state().write().unwrap_poison();
            state.enrollment_embeddings.push(vec![0.5; 1024]);
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
        // toggle-off/on.
        let state = voice_state().read().unwrap_poison();
        assert_eq!(state.enrollment_embeddings.len(), 1);
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
        let saved_embeddings = vec![vec![0.5; 1024]];
        let saved_chunks = vec![vec![0.5; 100]];
        {
            let mut state = voice_state().write().unwrap_poison();
            state.enrollment_embeddings = saved_embeddings.clone();
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
        assert_eq!(state.enrollment_embeddings, saved_embeddings);
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
            state.enrollment_embeddings.push(vec![0.5; 1024]);
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
        assert!(state.enrollment_embeddings.is_empty());
        assert!(state.negative_audio_chunks.is_empty());
    }

    #[test]
    #[serial_test::serial(voice)]
    fn reset_levels_preserve_session_ux_state() {
        let _ = init_global();
        // Session-level UX state (refractory_until, rate limiter timestamps)
        // must survive all reset levels — no level touches them.
        for level in [ResetLevel::Soft, ResetLevel::Full, ResetLevel::Cancel] {
            let mut ctx = PipelineCtx::new();
            ctx.refractory_until = Some(Instant::now());
            ctx.last_error_message_time = Some(Instant::now());
            ctx.last_voice_notice_time = Some(Instant::now());

            ctx.reset_pipeline_state(level);

            assert!(
                ctx.refractory_until.is_some(),
                "refractory_until lost at {level:?}"
            );
            assert!(
                ctx.last_error_message_time.is_some(),
                "last_error_message_time lost at {level:?}"
            );
            assert!(
                ctx.last_voice_notice_time.is_some(),
                "last_voice_notice_time lost at {level:?}"
            );
        }
    }

    // ── Phase 3 owner-negative state machine tests ───────────────────────
    // Regression tests for mahbot-1782: the Phase-3 speech counter grew
    // quadratically (un-drained buffers re-counted every prior frame on each
    // mic chunk), tripping the (then-60 s, now 15 s) target after ~2–4 real
    // seconds.  The counting path is driven through the pure
    // [`process_phase3_frames`] with injected VAD decisions — the stateful
    // earshot neural detector is not deterministic on synthetic audio, so
    // decisions are supplied directly.

    /// Build an initial `Phase3Progress` for the state-machine tests.
    fn empty_phase3_progress() -> Phase3Progress {
        Phase3Progress {
            processed: 0,
            silence_samples: 0,
            negatives_speech_samples: 0,
            completed_chunks: Vec::new(),
        }
    }

    #[test]
    fn phase3_speech_counter_is_1_to_1_and_chunking_invariant() {
        // 60 mic chunks of 0.5 s of continuous speech (injected all-speech
        // decisions).  Each hop must be counted exactly once: the counter
        // equals the real audio duration (minus the trailing partial window,
        // which is never fed to the VAD), regardless of chunking.
        const CHUNK_SAMPLES: usize = SAMPLE_RATE as usize / 2; // 0.5 s
        const CHUNKS: usize = 60;
        let total_samples = CHUNK_SAMPLES * CHUNKS; // 480,000

        let mut buf: Vec<f32> = Vec::new();
        let mut prog = empty_phase3_progress();
        let mut seen = 0usize;
        for _ in 0..CHUNKS {
            buf.extend(std::iter::repeat_n(0.1f32, CHUNK_SAMPLES));
            prog = process_phase3_frames(
                &mut buf,
                prog.processed,
                prog.silence_samples,
                prog.negatives_speech_samples,
                |_| true, // every hop VAD-positive
            );
            seen += CHUNK_SAMPLES;
            // Invariant: the counter never exceeds the real audio duration.
            assert!(
                prog.negatives_speech_samples <= seen,
                "counter {} exceeded real audio {seen}",
                prog.negatives_speech_samples,
            );
            // 1:1 per-call monotone check: in all-speech mode every processed
            // hop is counted, so the counter must equal the watermark exactly.
            // The quadratic bug made the counter grow by the WHOLE buffer on
            // each chunk (counter ≫ processed); an under-count would leave it
            // behind (counter < processed).
            assert_eq!(
                prog.negatives_speech_samples, prog.processed,
                "counter must track processed hops 1:1 across calls \
                 (no re-count, no under-count)",
            );
        }

        // Each processed hop counted exactly once.
        let expected =
            (total_samples.saturating_sub(FRAME_LENGTH)) / HOP_LENGTH * HOP_LENGTH + HOP_LENGTH;
        assert_eq!(prog.negatives_speech_samples, expected);
        assert!(prog.negatives_speech_samples <= total_samples);

        // Chunking invariance: the same audio fed in ONE call must yield the
        // same counter.  The quadratic bug's signature was chunk-count-
        // dependent growth (each new chunk re-counted all prior frames).
        let mut one_shot = vec![0.1f32; total_samples];
        let one_shot_prog = process_phase3_frames(&mut one_shot, 0, 0, 0, |_| true);
        assert_eq!(
            one_shot_prog.negatives_speech_samples, prog.negatives_speech_samples,
            "counter must not depend on mic-chunk boundaries",
        );

        // Continuous speech produces no chunk boundaries — the unfinalized
        // segment survives intact (nothing drained, nothing pushed).
        assert!(prog.completed_chunks.is_empty());
        assert_eq!(buf.len(), total_samples);
    }

    #[test]
    fn phase3_chunks_are_full_speech_segments_across_mic_chunks() {
        // Deterministic VAD via decision-coded audio: hops of +1.0 are speech,
        // hops of -1.0 are silence; the injected predicate decodes that.  The
        // pattern (30 speech / 30 silence / 10 speech hops) is split across
        // two mic chunks so the chunk boundary AND the unfinalized segment
        // span calls.
        let decision = |hop: &[f32]| hop[0] > 0.0;
        let mut audio: Vec<f32> = Vec::new();
        for _ in 0..30 {
            audio.extend(std::iter::repeat_n(1.0f32, HOP_LENGTH));
        }
        for _ in 0..30 {
            audio.extend(std::iter::repeat_n(-1.0f32, HOP_LENGTH));
        }
        for _ in 0..10 {
            audio.extend(std::iter::repeat_n(1.0f32, HOP_LENGTH));
        }

        // Call 1: speech + the first 10 silence hops (39 full frames — the
        // silence run is below the boundary threshold, nothing finalized).
        let split = 40 * HOP_LENGTH;
        let mut buf = audio[..split].to_vec();
        let mut prog = process_phase3_frames(&mut buf, 0, 0, 0, decision);
        assert!(prog.completed_chunks.is_empty());
        assert_eq!(prog.negatives_speech_samples, 30 * HOP_LENGTH);
        assert_eq!(prog.silence_samples, 9 * HOP_LENGTH);

        // Call 2: the silence run crosses the threshold mid-call → exactly ONE
        // completed chunk holding the full first speech segment (29 of the 30
        // speech hops — the pre-existing boundary formula ends a chunk one hop
        // before the silence run starts).  A full-length segment, never a hop
        // sliver truncated by a drain.
        buf.extend_from_slice(&audio[split..]);
        prog = process_phase3_frames(
            &mut buf,
            prog.processed,
            prog.silence_samples,
            prog.negatives_speech_samples,
            decision,
        );
        assert_eq!(prog.completed_chunks.len(), 1);
        assert_eq!(prog.completed_chunks[0].len(), 29 * HOP_LENGTH);

        // Counter 1:1: hops 0..29 (30) + hops 60..68 (9) = 39 hops, once each.
        assert_eq!(prog.negatives_speech_samples, 39 * HOP_LENGTH);

        // The unfinalized second segment (speech hops 60..69) survives in the
        // buffer across the call boundary.
        assert!(buf.len() >= 10 * HOP_LENGTH);

        // Feed a few silence hops so the final speech hop is processed too —
        // and prove the continued silence after a boundary pushes nothing
        // (the silence-run reset fix: the boundary fires once per run, not on
        // every silent hop).
        for _ in 0..4 {
            buf.extend(std::iter::repeat_n(-1.0f32, HOP_LENGTH));
        }
        prog = process_phase3_frames(
            &mut buf,
            prog.processed,
            prog.silence_samples,
            prog.negatives_speech_samples,
            decision,
        );
        assert_eq!(prog.negatives_speech_samples, 40 * HOP_LENGTH);
        assert!(
            prog.completed_chunks.is_empty(),
            "continued silence must not push spurious chunks",
        );
    }

    #[test]
    fn phase3_negative_collection_audio_counts_1_to_1_through_pipeline() {
        // End-to-end regression through the production entry point.  Force
        // the stateful earshot detector to report speech on every hop by
        // setting the threshold below its documented [0,1] score range — the
        // ticket-sanctioned threshold trick.  The same audio fed in 0.5 s
        // chunks must count 1:1 (each hop fed exactly once) and must be
        // chunking-invariant through the full ctx write-back path.
        let mut ctx = PipelineCtx::new();
        ctx.enrollment_vad = Some(earshot::Detector::default());
        ctx.vad_threshold = -1.0;

        const CHUNK_SAMPLES: usize = SAMPLE_RATE as usize / 2; // 0.5 s
        let chunk = vec![0.1f32; CHUNK_SAMPLES];
        for _ in 0..10 {
            handle_negative_collection_audio(&chunk, &mut ctx);
        }
        let total = 10 * CHUNK_SAMPLES;
        let expected = (total.saturating_sub(FRAME_LENGTH)) / HOP_LENGTH * HOP_LENGTH + HOP_LENGTH;
        assert_eq!(ctx.negatives_speech_samples, expected);
        assert!(ctx.negatives_speech_samples <= total);

        // Chunking invariance through the production path: one 5 s call must
        // land on the same counter as ten 0.5 s calls.
        let mut ctx2 = PipelineCtx::new();
        ctx2.enrollment_vad = Some(earshot::Detector::default());
        ctx2.vad_threshold = -1.0;
        let big = vec![0.1f32; total];
        handle_negative_collection_audio(&big, &mut ctx2);
        assert_eq!(ctx2.negatives_speech_samples, ctx.negatives_speech_samples);

        // Continuous speech never hits a chunk boundary — nothing pushed, and
        // the unfinalized segment survives intact (no drain).
        assert_eq!(ctx.phase3_audio_buf.len(), total);
        assert!(ctx.phase3_audio_buf.iter().all(|&s| s == 0.1));
    }

    // ── handle_segment_boundary tests ───────────────────────────────────
    // Tests the extracted segment boundary check logic with the encoder
    // pipeline signature `(&mut self, hop_count: usize)`.

    #[test]
    fn handle_segment_boundary_resets_at_threshold() {
        let mut ctx = PipelineCtx::new();
        ctx.score_window = vec![0.5; 5];
        ctx.segment_silence_hops = 10;

        // Complete adaptive threshold bootstrap so the reset-to-bootstrapping
        // assertion is meaningful (not trivially true from PipelineCtx::new()).
        {
            let mut at = AdaptiveThresholdState::new();
            for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES + 1 {
                at.feed(0.5, ADAPTIVE_K_DEFAULT);
            }
            assert!(
                !at.is_bootstrapping(),
                "adaptive threshold must exit bootstrap after {} feeds",
                ADAPTIVE_BOOTSTRAP_FRAMES + 1
            );
            ctx.adaptive_threshold = at;
        }

        // hop_count at threshold → boundary fires
        ctx.handle_segment_boundary(SEGMENT_TIMEOUT_HOPS);

        // ── Per-segment state reset on ctx ──
        assert!(ctx.score_window.is_empty(), "score_window must be cleared");
        assert_eq!(
            ctx.segment_silence_hops, 0,
            "segment_silence_hops must be reset"
        );
        assert!(
            ctx.adaptive_threshold.is_bootstrapping(),
            "adaptive_threshold must be reset (re-enter bootstrap)"
        );
    }

    #[test]
    fn handle_segment_boundary_persists_below_threshold() {
        let mut ctx = PipelineCtx::new();
        ctx.score_window = vec![0.5; 5];
        ctx.segment_silence_hops = 10;

        // hop_count below threshold → state persisted
        let below_threshold = SEGMENT_TIMEOUT_HOPS - 1;
        ctx.handle_segment_boundary(below_threshold);

        // ── Per-segment state preserved on ctx ──
        assert_eq!(
            ctx.segment_silence_hops, below_threshold,
            "counter must be persisted below threshold"
        );
        assert!(
            !ctx.score_window.is_empty(),
            "score_window must survive below threshold"
        );
    }

    #[test]
    fn handle_segment_boundary_does_not_reset_when_recording() {
        // A just-fired detection (is_recording) must NOT be reset by the
        // boundary check — the detection→recording handoff completes the
        // transition and owns the state.
        let mut ctx = PipelineCtx::new();
        ctx.is_recording = true;
        ctx.score_window = vec![0.5; 5];
        ctx.segment_silence_hops = 0;

        ctx.handle_segment_boundary(SEGMENT_TIMEOUT_HOPS);

        assert!(
            !ctx.score_window.is_empty(),
            "recording mode must skip the boundary reset"
        );
        assert_eq!(
            ctx.segment_silence_hops, 0,
            "recording mode must not write back the counter"
        );
    }

    #[test]
    fn handle_segment_boundary_counting_across_calls() {
        // Simulates the VAD-gap counter accumulating across detection calls:
        // below threshold → persisted; the next call crosses the threshold
        // → boundary fires and the counter resets.
        let mut ctx = PipelineCtx::new();
        ctx.segment_silence_hops = 0;

        // Call 1: 10 silence hops (< 19) → persisted.
        ctx.handle_segment_boundary(10);
        assert_eq!(ctx.segment_silence_hops, 10);

        // Call 2: speech resets the counter to 0 (handled by the frame loop),
        // then 5 silence hops → persisted again.
        ctx.segment_silence_hops = 0;
        ctx.handle_segment_boundary(5);
        assert_eq!(ctx.segment_silence_hops, 5);

        // Call 3: crosses the threshold → boundary fires, counter resets.
        ctx.handle_segment_boundary(SEGMENT_TIMEOUT_HOPS);
        assert_eq!(ctx.segment_silence_hops, 0);
        assert!(ctx.score_window.is_empty());
    }

    // ── PCM cache eviction tests ───────────────────────────────────────────

    static EVICTION_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Write synthetic PCM data (16 KB) to `path` and optionally backdate its
    /// mtime by `age` (None = leave current).  Returns the entry size.
    fn seed_test_pcm(path: &Path, age: Option<Duration>) -> u64 {
        let samples: Vec<f32> = vec![0.0; 4096]; // 16 KB
        write_pcm_cache(path, &samples);
        if let Some(age) = age {
            let mtime = std::time::SystemTime::now() - age;
            let times = std::fs::FileTimes::new().set_modified(mtime);
            let _ = std::fs::File::open(path).and_then(|f| f.set_times(times));
        }
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    }

    /// Set config fields for the duration of a test, restoring defaults on drop.
    /// Acquires [`EVICTION_TEST_LOCK`] so that parallel tests do not race on the
    /// global CONFIG singleton.
    struct EvictionConfigGuard {
        _guard: MutexGuard<'static, ()>,
    }
    impl EvictionConfigGuard {
        fn set(size_mb: &str, age_days: &str) -> Self {
            let lock = EVICTION_TEST_LOCK.lock().unwrap();
            let _ = crate::config::CONFIG.set_string_field("voice_cache_max_size_mb", size_mb);
            let _ = crate::config::CONFIG.set_string_field("voice_cache_max_age_days", age_days);
            EvictionConfigGuard { _guard: lock }
        }
    }
    impl Drop for EvictionConfigGuard {
        fn drop(&mut self) {
            let _ = crate::config::CONFIG.set_string_field("voice_cache_max_size_mb", "");
            let _ = crate::config::CONFIG.set_string_field("voice_cache_max_age_days", "");
        }
    }

    #[test]
    fn evict_pcm_cache_nonexistent_dir_does_not_panic() {
        let tmp = tempfile::tempdir().unwrap();
        let nonexistent = tmp.path().join("does_not_exist");
        evict_pcm_cache(&nonexistent);
    }

    #[test]
    fn evict_pcm_cache_ignores_tmp_files() {
        let tmp = tempfile::tempdir().unwrap();
        // Create a .tmp file (should be ignored by eviction)
        let tmp_path = tmp.path().join("abcdef0123456789abcdef0123456789.tmp");
        let _ = std::fs::write(&tmp_path, &[0u8; 4096]);
        assert!(tmp_path.exists());

        // Also create 3 real entries (16 KB each) that are old enough to
        // trigger age-based eviction.
        let old_age = Duration::from_secs(2 * 86400);
        for i in 0..3u8 {
            let name = format!("{:064x}", i);
            seed_test_pcm(&tmp.path().join(&name), Some(old_age));
        }

        // age = 1 day triggers eviction of the old entries; .tmp file must survive
        let _guard = EvictionConfigGuard::set("0", "1");
        evict_pcm_cache(tmp.path());
        assert!(tmp_path.exists(), ".tmp files must survive eviction");
        // Verify non-tmp files were processed (old entries removed)
        for i in 0..3u8 {
            assert!(
                !tmp.path().join(format!("{:064x}", i)).exists(),
                "old entry {i} must be evicted"
            );
        }
    }

    #[test]
    fn evict_pcm_cache_age_based_removes_stale_entries() {
        let tmp = tempfile::tempdir().unwrap();

        // Create a "recent" entry
        let recent_path = tmp.path().join("a".repeat(64));
        seed_test_pcm(&recent_path, None);

        // Create an "old" entry by backdating its mtime far in the past
        let old_path = tmp.path().join("b".repeat(64));
        seed_test_pcm(&old_path, Some(Duration::from_secs(2 * 86400)));

        assert!(recent_path.exists());
        assert!(old_path.exists());

        // Evict with 1 day max age — old entry should be removed, recent kept
        let _guard = EvictionConfigGuard::set("0", "1"); // size disabled, age = 1 day
        evict_pcm_cache(tmp.path());

        assert!(
            recent_path.exists(),
            "recent entry must survive age-based eviction"
        );
        assert!(!old_path.exists(), "old entry must be evicted by age limit");
    }

    #[test]
    fn evict_pcm_cache_age_zero_disables_age_eviction() {
        let tmp = tempfile::tempdir().unwrap();

        let path = tmp.path().join("a".repeat(64));
        // Set mtime to 2 days ago
        seed_test_pcm(&path, Some(Duration::from_secs(2 * 86400)));

        // age = 0 means disabled — even old entries should survive
        let _guard = EvictionConfigGuard::set("0", "0"); // both disabled
        evict_pcm_cache(tmp.path());
        assert!(
            path.exists(),
            "entry must survive when age limit is disabled (0)"
        );
    }

    #[test]
    fn evict_pcm_cache_size_zero_disables_size_eviction() {
        let tmp = tempfile::tempdir().unwrap();

        // Create several entries totalling ~48 KB
        for i in 0..3u8 {
            let name = format!("{:064x}", i);
            seed_test_pcm(&tmp.path().join(&name), None);
        }

        // size = 0 means disabled — all entries survive regardless of total size
        let _guard = EvictionConfigGuard::set("0", "0"); // both disabled
        evict_pcm_cache(tmp.path());
        for i in 0..3u8 {
            let name = format!("{:064x}", i);
            assert!(
                tmp.path().join(&name).exists(),
                "entry {name} must survive when size limit is disabled (0)"
            );
        }
    }

    #[test]
    fn evict_pcm_cache_within_limit_keeps_all_entries() {
        let tmp = tempfile::tempdir().unwrap();

        // Create 3 entries totalling ~48 KB
        for i in 0..3u8 {
            let name = format!("{:064x}", i);
            seed_test_pcm(&tmp.path().join(&name), None);
        }

        // Default max size (100 MB) is well above 48 KB — all entries survive.
        // age = 0 disables age-based eviction so it doesn't interfere.
        let _guard = EvictionConfigGuard::set("100", "0");
        evict_pcm_cache(tmp.path());
        for i in 0..3u8 {
            let name = format!("{:064x}", i);
            assert!(
                tmp.path().join(&name).exists(),
                "entry {name} must survive when under size limit"
            );
        }
    }

    #[test]
    fn evict_pcm_cache_size_based_removes_oldest_entry_first() {
        let tmp = tempfile::tempdir().unwrap();

        // Create 68 entries (~16 KB each = ~1088 KB total, exceeding 1 MB limit)
        let count = 68;
        for i in 0..count {
            let name = format!("{:064x}", i);
            // Stagger mtime: entry 0 is oldest (67 hours ago), entry 67 is newest
            let age_hours = (count - 1 - i) as u64;
            seed_test_pcm(
                &tmp.path().join(&name),
                Some(Duration::from_secs(age_hours * 3600)),
            );
        }

        // Max size = 1 MB, age disabled
        let _guard = EvictionConfigGuard::set("1", "0");
        evict_pcm_cache(tmp.path());

        // Remaining size must be ≤ 1 MB
        let remaining: u64 = std::fs::read_dir(tmp.path())
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) != Some("tmp"))
            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
            .sum();
        assert!(
            remaining <= 1_048_576,
            "remaining size {remaining} exceeds 1 MB limit"
        );

        // The oldest entry (index 0) must be evicted
        assert!(
            !tmp.path().join(format!("{:064x}", 0)).exists(),
            "oldest entry (0) must be evicted by size limit"
        );

        // The newest entry (index 67) must survive
        assert!(
            tmp.path().join(format!("{:064x}", 67)).exists(),
            "newest entry (67) must survive size-based eviction"
        );
    }

    #[test]
    fn evict_pcm_cache_combined_age_and_size_eviction() {
        let tmp = tempfile::tempdir().unwrap();

        // Create 5 old entries (2 days old — stale)
        let old_age = Duration::from_secs(2 * 86400);
        for i in 0..5u8 {
            let name = format!("old_{:064x}", i);
            seed_test_pcm(&tmp.path().join(&name), Some(old_age));
        }

        // Create 68 recent entries (~16 KB each, totalling ~1088 KB).
        // All within the last hour so the 1-day age limit does NOT touch them.
        for i in 0..68u8 {
            let name = format!("recent_{:064x}", i);
            // Stagger mtime from 0 to 67 minutes ago (all well under 1 day)
            let age_minutes = (67 - i) as u64;
            seed_test_pcm(
                &tmp.path().join(&name),
                Some(Duration::from_secs(age_minutes * 60)),
            );
        }

        // Age = 1 day, size = 1 MB — both limits active
        let _guard = EvictionConfigGuard::set("1", "1");
        evict_pcm_cache(tmp.path());

        // All old entries must be gone (age-based eviction)
        for i in 0..5u8 {
            let name = format!("old_{:064x}", i);
            assert!(
                !tmp.path().join(&name).exists(),
                "stale entry {name} must be evicted by age limit"
            );
        }

        // The newest recent entry must survive (proves size eviction didn't
        // remove everything — combined test would pass vacuously otherwise)
        assert!(
            tmp.path().join(format!("recent_{:064x}", 67)).exists(),
            "newest recent entry must survive combined eviction"
        );

        // Remaining size (recent entries only) must be ≤ 1 MB
        let remaining: u64 = std::fs::read_dir(tmp.path())
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) != Some("tmp"))
            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
            .sum();
        assert!(
            remaining <= 1_048_576,
            "remaining size {remaining} exceeds 1 MB limit after combined eviction"
        );
    }

    // ── PCM cache read/write tests ─────────────────────────────────────────

    /// Seed the PCM cache with one 16 KB entry keyed by `(text, style, seed)`.
    fn seed_pcm_cache(cache_dir: &Path, text: &str, style: &str, seed: u64) {
        let key = pcm_cache_key(text, style, seed, SAMPLE_RATE, "hash");
        let path = cache_dir.join(&key);
        write_pcm_cache(&path, &[0.0f32; 4096]); // valid PCM, 16 KB
        assert!(path.exists());
    }

    #[test]
    fn pcm_cache_read_normal_returns_some() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("a".repeat(64));
        seed_test_pcm(&path, None);
        assert!(path.exists());

        let result = read_pcm_cache(&path);
        assert!(result.is_some(), "normal read should return cached PCM");
    }

    #[test]
    fn pcm_cache_bust_skips_cache_and_returns_none_when_tts_unavailable() {
        let tmp = tempfile::tempdir().unwrap();
        seed_pcm_cache(tmp.path(), "hello", "style", 42);

        // With bust=1: skip cache, attempt TTS (fails→None)
        let _guard = set_env_var("MAHBOT_TEST_CACHE_BUST", Some("1"));
        let result =
            synthesize_with_pcm_cache("hello", "style", 42, SAMPLE_RATE, "hash", tmp.path());
        assert!(
            result.is_none(),
            "bust=1 should bypass cache and attempt TTS (which fails without models)"
        );
        drop(_guard);
        // Without bust: cache hit → Some
        let result =
            synthesize_with_pcm_cache("hello", "style", 42, SAMPLE_RATE, "hash", tmp.path());
        assert!(
            result.is_some(),
            "without bust, cached PCM should be returned directly"
        );
    }

    #[test]
    fn pcm_cache_bust_requires_exact_value_one() {
        let tmp = tempfile::tempdir().unwrap();
        seed_pcm_cache(tmp.path(), "world", "style", 99);

        // Setting to "true" should NOT bypass the cache
        let _guard = set_env_var("MAHBOT_TEST_CACHE_BUST", Some("true"));
        let result =
            synthesize_with_pcm_cache("world", "style", 99, SAMPLE_RATE, "hash", tmp.path());
        assert!(
            result.is_some(),
            "MAHBOT_TEST_CACHE_BUST=true should NOT bypass cache"
        );
    }
}
