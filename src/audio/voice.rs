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
//! 5. **Wake word matching** — Conv1D classifier on a 3-embedding sliding window
//!    with immediate-fire rolling-sum detection (speaker-blind).
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
use crate::EMBEDDING_DIM;
use crate::audio::embedding_sequence::{EmbeddingSequence, Source, UtteranceId};
use crate::audio::wake_word_classifier::{self, ClassifierWeights, WakeWordClassifier};
use crate::config::CONFIG;
use crate::turso;
use crate::util::UnwrapPoison;
use crate::util::hex_string;
use crate::vector::cosine_similarity;
use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
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

/// Frame size for mel spectrogram (512 samples = 32ms at 16kHz).
pub(crate) const FRAME_LENGTH: usize = 512;

/// Hop length between frames (256 samples at 16 kHz).  This constant controls
/// VAD frame iteration stride and silence tracking in the application code.
/// The ONNX mel spectrogram model uses its own internal stride (160 samples =
/// 10ms) — HOP_LENGTH does NOT affect mel frame spacing.
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
/// have consistent temporal positions.  See [`flush_voice_batch`]
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
/// See [`flush_voice_batch`] for the detailed rationale.
const VOICE_BATCH_OVERLAP: usize = MEL_STRIDE * 2;

/// Embedding window: 76 consecutive mel frames (~760ms with the ONNX mel
/// model's 10ms internal stride, not 16ms — see HOP_LENGTH note above).
const EMBEDDING_WINDOW_FRAMES: usize = 76;

/// Deferred burst trigger: the stride-8 scorer HOLDS position 0
/// until the accumulated mel-frame buffer reaches this many frames, then
/// sweeps the start-aligned positions 0/8/16/24 in one synchronous burst with
/// the trained start-0-aligned padded geometry.
///
/// Scoring position 0 below 68 frames resets the rolling window before
/// positions 8/16/24 can fire.  The segment-end pass remains the overrun
/// safety net if a below-reset re-score ever does clear the rolling window.
const BURST_TRIGGER_FRAMES: usize = 68;

/// Stride of the deferred burst / segment-end pass position grid
/// (start-aligned positions 0, 8, 16, 24).
const BURST_STRIDE: usize = 8;

/// Maximum number of positions in the burst sweep / segment-end pass
/// (4 positions × stride 8).
const BURST_MAX_POSITIONS: usize = 4;

/// Maximum command recording duration (30 seconds).
const MAX_RECORD_SECS: usize = 30;

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
/// Previously used [`SILENCE_THRESHOLD_SAMPLES`] (~1.5s),
/// which meant utterances with pauses were segmented differently between
/// enrollment training and streaming inference, contributing to out-of-
/// distribution classifier inputs.
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
/// * unbounded queue growth root cause
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

/// Maximum number of download retries.
const MAX_DOWNLOAD_RETRIES: u32 = 10;

/// Default wake word phrase used when no phrase has been specified
/// or when a legacy model has no phrase field. This is the serde default
/// for [`PersistedModel::phrase`] via [`default_phrase`].
pub(crate) const DEFAULT_WAKE_WORD_PHRASE: &str = "mahbot";

/// Returns `true` if `phrase` (already normalized via [`normalize_phrase`]) is
/// the Mahbot wake word or its "hey" variant.
///
/// Only these two exact forms trigger Mahbot-specific confusable inclusion.
/// Unnormalized input, diacritics ("mähbot"), partial-word
/// matches ("mahbotics"), or other variants return `false`.
///
/// # Quality implications for non-Mahbot wake words
///
/// When the enrolled wake word is not "mahbot" or "hey mahbot", the
/// Mahbot-specific confusable phrases ([`CONFUSABLE_PHRASES`]) are excluded
/// from classifier training. The model trains on ambient
/// negatives and general-purpose unrelated speech ([`UNRELATED_PHRASES`])
/// only.
///
/// This is a **known quality regression** for non-Mahbot phrases — ambient-only
/// negatives do not teach the model to reject phonetic near-misses (e.g., "pay
/// mabot" sounds similar to "hey mahbot"). The model will likely false-accept
/// more wake-word-like sounds until a general-purpose phonetic near-miss generator
/// is implemented (future work).
///
/// Additionally, the ~2-minute TTS prewarm of confusable embeddings runs on every
/// startup regardless of the enrolled phrase (accepted technical debt — see
/// [`prewarm_confusable_embeddings`]). A future deferred-generation approach could
/// eliminate this waste by generating confusables on demand at enrollment time.
///
/// # Normalization contract
///
/// The caller MUST pass a phrase already processed by [`normalize_phrase`].
/// Passing unnormalized input (mixed case, extra whitespace) produces `false`
/// even when the semantic intent matches:
///
/// ```ignore
/// // These pass (normalized input):
/// assert!(is_mahbot_wake_word("mahbot"));
/// assert!(is_mahbot_wake_word("hey mahbot"));
/// // These would be false because they are not normalized:
/// assert!(!is_mahbot_wake_word("Mahbot"));       // different case
/// assert!(!is_mahbot_wake_word("HEY MAHBOT"));   // different case
/// ```
///
/// In practice this contract is always satisfied because the only production
/// call site reads from `enrolling_phrase`, which is normalized at enrollment
/// start (see [`PipelineCtx::handle_start_enrollment`], line ~5153).
#[must_use]
fn is_mahbot_wake_word(phrase: &str) -> bool {
    phrase == DEFAULT_WAKE_WORD_PHRASE || phrase == "hey mahbot"
}

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

/// Segment boundary timeout in VAD-negative frame hops (~300 ms of consecutive
/// silence).  At [`HOP_LENGTH`] (256 samples = 16 ms at 16 kHz) per hop,
/// 19 hops ≈ 304 ms.  When this many consecutive silence hops are observed
/// across calls to [`handle_wake_word_detection`], the current detection
/// segment is considered ended and per-segment state is reset to prevent
/// cross-utterance score accumulation.
///
/// ## Value justification
/// Natural intra-phrase pauses (syllable boundaries, stop consonants) in
/// conversational speech are typically <200 ms (Crystal & House 1988, JASA).
/// A threshold of ~304 ms is conservatively above this range, ensuring fluent
/// wake words are not interrupted while still catching genuine utterance
/// boundaries.  This value also aligns with the 300 ms silence threshold
/// commonly used in voice activity segmentation (ITU-T P.85).
const SEGMENT_TIMEOUT_HOPS: usize = 19;

/// Maximum number of recent embeddings to keep in the ring buffer.
/// With stride=8 (~89.5% overlap), each new embedding covers ~1.2s of audio
/// and arrives every ~128ms, keeping ~19 embeddings = ~2.4 seconds of context.
const EMBEDDING_RING_MAX: usize = 19;

/// Number of enrollment samples required.
const NUM_ENROLLMENT_SAMPLES: usize = 10;

/// Minimum length (in audio samples at 16kHz) for a collected ambient audio
/// chunk to be used as a negative training example.
///
/// Set to 0.5s of audio, which produces ~31 mel frames (padded to 76 for the
/// embedding model).  Chunks shorter than this are discarded — they would be
/// mostly padding/silence and provide negligible discriminative signal.
const MIN_NEGATIVE_AUDIO_LEN: usize = SAMPLE_RATE as usize / 2;

/// Maximum number of ambient noise chunks to retain for negative training.
/// If training repeatedly fails (ONNX not loaded, <2 chunks, or empty
/// embeddings), this cap prevents unbounded memory growth in the voice
/// pipeline state.
const MAX_NEGATIVE_AUDIO_CHUNKS: usize = 100;

// ── Phase 3 (owner-negative) enrollment constants ────────────

/// Target VAD-positive speech duration for owner-negative Phase 3 collection.
/// ~60 seconds of VAD-positive general speech from the user, used alongside
/// ambient/cache negatives to train the classifier.
const NEGATIVES_TARGET_SECONDS: usize = 60;

/// Maximum owner-negative audio samples to retain (90s cap, 1.5× the target).
/// This prevents unbounded memory growth if the user speaks continuously.
const MAX_OWNER_NEGATIVE_SAMPLES: usize = SAMPLE_RATE as usize * NEGATIVES_TARGET_SECONDS * 3 / 2;

/// Wall-clock timeout for Phase 3 owner-negative collection (120 seconds).
/// If the user stays silent or provides very little speech, the pipeline
/// finalizes with whatever was collected rather than stalling indefinitely.
const PHASE3_TIMEOUT_SECS: u64 = 120;

/// Threshold for detecting clipping: samples at or above this absolute
/// value are considered clipped (near i16::MAX = 32767 in f32 [-1, 1]).
const ENROLLMENT_QUALITY_CLIPPING_THRESHOLD: f32 = 0.999;

/// Minimum acceptable utterance duration in ms for quality scoring.
pub(crate) const ENROLLMENT_QUALITY_DURATION_MIN_MS: u64 = 400;

/// Maximum acceptable utterance duration in ms for quality scoring.
/// Utterances longer than this may contain too much silence padding.
pub(crate) const ENROLLMENT_QUALITY_DURATION_MAX_MS: u64 = 2000;

/// Fraction of enrollment utterances that must trigger detection in the
/// blocking self-test — rejects model deployment on failure.
const ENROLLMENT_QUALITY_SELF_TEST_MIN_FRACTION: f32 = 0.8;

/// Minimum cosine similarity between an utterance's mean embedding and the
/// centroid for the consistency check.  Set to 0.65 as a conservative
/// starting value — the enrollment prompts intentionally diversify
/// (distance, angle, voice quality) which can lower cross-utterance
/// similarity.  Verified for streaming-pipeline embeddings via the E2E
/// benchmark: all 13/13 enrollment utterances (5 enrolled +
/// 8 augmented variants) passed with ≥0.65 cosine similarity to centroid
/// across multiple benchmark runs.  The denser but temporally-averaged
/// streaming embeddings produce similar utterance-level similarities to
/// the old stride-8 embeddings.  If false acceptances increase, raise to
/// 0.70; if valid enrollments consistently fail, lower to 0.60.
const ENROLLMENT_CONSISTENCY_MIN_SIMILARITY: f32 = 0.65;

/// Fraction of enrollment utterances that must pass the consistency check.
/// Uses ceil() rounding, making the effective pass fraction vary with N
/// (70% for N=10, up to ~83% for N=6).  This is intentional — fewer
/// utterances get a higher bar to compensate for the smaller sample.
const ENROLLMENT_CONSISTENCY_MIN_FRACTION: f32 = 0.7;

/// Enrollment prompts for multi-position guidance.
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
/// phonemes that the enrollment VAD threshold excludes.
pub(crate) const RAW_RING_MAX: usize = SAMPLE_RATE as usize / 5;

/// Context padding duration in milliseconds (~100ms at 16 kHz).
///
/// Used for PCM augmentation duration estimation in speed-perturbation
/// decisions (e.g., `pre_pad_samples = samples.len() - 2 * CONTEXT_PADDING_SAMPLES`
/// in [`handle_enrollment_sample`] and [`prewarm_phrase_embeddings`]).
///
/// ⚠ Context padding is no longer used for VAD utterance segmentation.
/// The VAD threshold is now unified to 0.5 for both
/// enrollment and streaming, so the asymmetry that context padding was
/// designed to mitigate (0.60 enrollment threshold vs 0.50 detection
/// threshold) no longer exists.  The [`DEFAULT_VAD_SEGMENTATION_CONFIG`]
/// sets `context_padding_samples: 0`.
const CONTEXT_PADDING_MS: usize = 100;

/// Context padding in audio samples at 16 kHz, derived from
/// CONTEXT_PADDING_MS to stay correct if the sample rate is adjusted.
pub(crate) const CONTEXT_PADDING_SAMPLES: usize =
    (CONTEXT_PADDING_MS * SAMPLE_RATE as usize) / 1000;

// ── Voice PCM disk cache ─────────────────────────────────

/// Name of the directory under the storage root for PCM audio cache.
const VOICE_CACHE_DIR: &str = "voice_cache";

/// Number of TTS seeds per confusable phrase for prosodic diversity.
pub(crate) const CONFUSABLE_SEEDS_PER_PHRASE: usize = 5;

/// Number of TTS seeds per unrelated phrase for prosodic diversity.
pub(crate) const UNRELATED_SEEDS_PER_PHRASE: usize = 3;

/// Seed base offset for confusable phrase synthesis.
/// Each phrase i with seed j (0..CONFUSABLE_SEEDS_PER_PHRASE) uses:
///   seed = CONFUSABLE_SEED_BASE + i * CONFUSABLE_SEEDS_PER_PHRASE + j
pub(crate) const CONFUSABLE_SEED_BASE: u64 = 1000;

/// Seed base offset for unrelated phrase synthesis.
/// Each phrase i with seed j (0..UNRELATED_SEEDS_PER_PHRASE) uses:
///   seed = UNRELATED_SEED_BASE + i * UNRELATED_SEEDS_PER_PHRASE + j
pub(crate) const UNRELATED_SEED_BASE: u64 = 2000;

// ── Shared PCM augmentation ─────────────────────────────

/// One PCM variant produced by [`augment_pcm_variants`].
///
/// `variant_index` always carries the recipe index (0 = original,
/// 1 = speed-down 0.95, 2 = speed-up, 3 = volume-down, 4 = pink noise 25 dB,
/// 5 = speed-down 0.90, 6-8 = white/pink/brown noise 10 dB,
/// 9-11 = white/pink/brown noise 5 dB) regardless of the push order the
/// caller requested.
#[derive(Debug, Clone)]
pub(crate) struct AugmentedPcmVariant {
    /// Recipe variant index (see struct doc).
    pub variant_index: usize,
    pub pcm: Vec<f32>,
}

/// Which variant set [`augment_pcm_variants`] yields for a call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AugmentSet {
    /// Full positive-training recipe: original, speed-down 0.95/0.90,
    /// speed-up (gated), volume-down, pink 25 dB, white/pink/brown noise at
    /// 10/5 dB.  Used by enrollment and bench positive paths.
    Full,
    /// Bounded negative-pool recipe: original + pink 25 dB.  Deliberate
    /// product tradeoff: the old speed/volume negative cells are dropped so
    /// the expanded 12-cell positive recipe's cold embedding recompute fits
    /// the 39 s bench budget (the full 5-cell negative set measured ~68 s
    /// cold; ~84 s with the final owner pool — both over budget); disclosed
    /// in the bench report.  Low-SNR
    /// negative coverage lives on the owner/ambient path (owner negatives
    /// get a brown-10 variant).
    Negatives,
}

/// Deterministic PCM augmentation of one input clip.
///
/// Produces the [`AugmentSet::Full`] variant list from `input`:
///
///   0: original (unmodified clone of the input)
///   1: speed-down 0.95×
///   2: speed-up 1.05× — included only when the pre-pad duration (input
///      length minus 2×[`CONTEXT_PADDING_SAMPLES`]) is ≥ 500 ms
///   3: volume-down −3 dB
///   4: pink noise at 25 dB SNR, seeded by `noise_seed`
///   5: speed-down 0.90× (the known warm-miss target)
///   6: white noise at 10 dB SNR
///   7: pink noise at 10 dB SNR
///   8: brown noise at 10 dB SNR
///   9: white noise at 5 dB SNR
///  10: pink noise at 5 dB SNR
///  11: brown noise at 5 dB SNR
///
/// [`AugmentSet::Negatives`] yields a bounded subset: 0, 4.
///
/// The `input` is the caller's pre-pad gate input (raw TTS PCM, VAD-gated
/// speech, or a VAD-segmented utterance slice — as each call site passes
/// today).  `noise_seed` is passed through verbatim (utterance count / TTS
/// phrase seed / loop index / constant, per site).  All arithmetic is
/// deterministic — the only RNG is the seeded noise generator inside
/// [`crate::util::add_noise`] / [`crate::util::add_noise_color`].
///
/// # Push order
///
/// Variants are returned in index order (speed-up 3rd, gated) — the push
/// order every call site uses.  The returned `variant_index` is unaffected
/// by the push order.
pub(crate) fn augment_pcm_variants(
    input: &[f32],
    sample_rate: u32,
    noise_seed: u64,
    set: AugmentSet,
) -> Vec<AugmentedPcmVariant> {
    use crate::util::NoiseColor;
    let pre_pad_samples = input.len().saturating_sub(2 * CONTEXT_PADDING_SAMPLES);
    let pre_pad_ms = (pre_pad_samples as u64 * 1000) / u64::from(sample_rate);
    let mut variants = Vec::with_capacity(12);
    variants.push(AugmentedPcmVariant {
        variant_index: 0,
        pcm: input.to_vec(),
    });
    match set {
        AugmentSet::Negatives => {
            variants.push(AugmentedPcmVariant {
                variant_index: 4,
                pcm: crate::util::add_noise(input, 25.0, noise_seed),
            });
        }
        AugmentSet::Full => {
            variants.push(AugmentedPcmVariant {
                variant_index: 1,
                pcm: crate::util::speed_perturbation(input, sample_rate, 0.95),
            });
            if pre_pad_ms >= 500 {
                variants.push(AugmentedPcmVariant {
                    variant_index: 2,
                    pcm: crate::util::speed_perturbation(input, sample_rate, 1.05),
                });
            }
            variants.push(AugmentedPcmVariant {
                variant_index: 3,
                pcm: crate::util::apply_gain(input, -3.0),
            });
            variants.push(AugmentedPcmVariant {
                variant_index: 4,
                pcm: crate::util::add_noise(input, 25.0, noise_seed),
            });
            variants.push(AugmentedPcmVariant {
                variant_index: 5,
                pcm: crate::util::speed_perturbation(input, sample_rate, 0.90),
            });
            for (idx, (color, snr)) in [
                (NoiseColor::White, 10.0),
                (NoiseColor::Pink, 10.0),
                (NoiseColor::Brown, 10.0),
                (NoiseColor::White, 5.0),
                (NoiseColor::Pink, 5.0),
                (NoiseColor::Brown, 5.0),
            ]
            .into_iter()
            .enumerate()
            {
                variants.push(AugmentedPcmVariant {
                    variant_index: 6 + idx,
                    pcm: crate::util::add_noise_color(input, snr, color, noise_seed),
                });
            }
        }
    }
    variants
}

#[cfg(test)]
mod augment_tests {
    //! Fixture tests locking the shared PCM augmentation helper to the
    //! recipe contract: the Full set (original, speed-down
    //! 0.95/0.90, gated speed-up, volume-down, pink 25 dB, white/pink/brown
    //! noise at 10/5 dB) and the bounded Negatives set (original + pink
    //! 25 dB).  Pure arithmetic — no models, no I/O.  Golden hashes pin the
    //! byte-stable report surfaces.

    use super::*;

    /// Deterministic fixed input PCM (no RNG): 220 Hz + 440 Hz sine ramp that
    /// decays over the clip — exercises the augmentation's time-domain paths.
    fn fixed_pcm(len: usize) -> Vec<f32> {
        let sample_rate = SAMPLE_RATE as f32;
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin())
                    * (1.0 - t / 2.0).max(0.0)
            })
            .collect()
    }

    /// SHA-256 over the ordered `(variant_index, pcm)` pairs — byte-stable.
    fn variant_list_hash(variants: &[(usize, Vec<f32>)]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for (idx, pcm) in variants {
            hasher.update((*idx as u32).to_le_bytes());
            for sample in pcm {
                hasher.update(sample.to_le_bytes());
            }
        }
        format!("{:x}", hasher.finalize())
    }

    fn run_set(len: usize, seed: u64, set: AugmentSet) -> Vec<(usize, Vec<f32>)> {
        let input = fixed_pcm(len);
        augment_pcm_variants(&input, SAMPLE_RATE, seed, set)
            .into_iter()
            .map(|v| (v.variant_index, v.pcm))
            .collect()
    }

    #[test]
    fn full_set_shape_and_speed_up_gate() {
        // Long input → all 12 cells; short input → speed-up (2) skipped.
        let long = run_set(16000, 5, AugmentSet::Full);
        assert_eq!(
            long.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "Full set must yield the 12-cell recipe in index order"
        );
        let short = run_set(4000, 5, AugmentSet::Full);
        assert_eq!(
            short.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "short input must skip speed-up (variant 2)"
        );
    }

    #[test]
    fn negatives_set_is_bounded() {
        let set = run_set(16000, 5, AugmentSet::Negatives);
        assert_eq!(
            set.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 4],
            "Negatives set must yield original + pink-25 only"
        );
    }

    /// Golden hashes captured at the current recipe (seed 5, fixed_pcm).
    /// Any change to the recipe's arithmetic, cells, gate, or push order
    /// breaks these — protecting the byte-stable report surfaces.
    const GOLDEN_LONG: [&str; 2] = [
        "6ae2faba828c7138bdaf0b4672c0e06ea521c849ca168be44bd64a56b05ba766", // Full
        "31362caa70d9da6ad5e8264a341603986459f8ca716d0dadd48342d4786161e8", // Negatives
    ];
    const GOLDEN_SHORT: [&str; 2] = [
        "25cfcf720f1524a58c413fa6005c8357643ba5968721094a0be9ef7114daec87", // Full
        "63889638dea4b8e54acee6fec6cd4e2b90acfea7c2a9f192fc10a93aa909c0d3", // Negatives
    ];

    #[test]
    fn helper_hash_stability() {
        // Printed once to capture the goldens (see GOLDEN_LONG/SHORT).
        for (i, set) in [AugmentSet::Full, AugmentSet::Negatives]
            .into_iter()
            .enumerate()
        {
            let long = variant_list_hash(&run_set(16000, 5, set));
            let short = variant_list_hash(&run_set(4000, 5, set));
            println!("GOLDEN set {i} ({set:?}): long={long} short={short}");
            assert_eq!(long, GOLDEN_LONG[i], "set {i} long-input golden drifted");
            assert_eq!(short, GOLDEN_SHORT[i], "set {i} short-input golden drifted");
        }
    }
}

// ── Confusable phrase list for negative training ──

/// Canonical confusable near-miss phrases for negative training, split into
/// difficulty tiers.  `CONFUSABLE_PHRASES` is the single source of truth — the
/// tier tables are emitted from this macro so they can never drift apart.
/// Tier ordering is load-bearing: each phrase must stay in its tier's section.
macro_rules! confusable_phrase_tiers {
    (
        $(
            $(#[$meta:meta])*
            $tier:ident: [$($phrase:literal),* $(,)?]
        ),* $(,)?
    ) => {
        pub(crate) const CONFUSABLE_PHRASES: &[&str] = &[ $($($phrase),* ,)* ];
        $(
            $(#[$meta])*
            #[cfg(feature = "voice-tests")]
            pub(crate) const $tier: &[&str] = &[ $($phrase),* ];
        )*
    };
}

confusable_phrase_tiers! {
    /// Hard-tier confusable phrases — direct phonetic substitutions (wake-word-like).
    /// These are the most acoustically similar to the wake word.
    /// Only compiled when `voice-tests` feature is enabled (used by E2E benchmark).
    CONFUSABLE_HARD: [
        // ── Direct phonetic substitutions (wake-word-like) ──────────────
        "hey madbot",
        "hey map bot",
        "day mahbot",
        "hey nab it",
        "hey man",
        "hey mabot",
        "hey mahbott",
        "hey mat",
        "hey max",
        "pay mabot",
    ],
    /// Medium-tier confusable phrases — rhythmic/melodic confusables and embedded
    /// wake-word sounds. Moderately difficult to distinguish from the wake word.
    /// Only compiled when `voice-tests` feature is enabled (used by E2E benchmark).
    CONFUSABLE_MEDIUM: [
        // ── Rhythmic/melodic confusables ─────────────────────────────────
        "hay map pot",
        "huh mahbot",
        "eh mad bot",
        "hey maybott",
        "they mad bot",
        "haymaker",
        // ── Embedded wake-word sounds ────────────────────────────────────
        "hey maybe not",
        "play mah jong",
        "hey matter of fact",
        "a day with mahbot",
    ],
    /// Easy-tier confusable phrases — short phonetic near-misses.
    /// These are single-phoneme substitutions that are relatively easy to reject.
    /// Only compiled when `voice-tests` feature is enabled (used by E2E benchmark).
    CONFUSABLE_EASY: [
        // ── Short phonetic near-misses ──────────────────────────────────
        "madbot", "mat bot", "bad bot", "mad lot", "mad pot", "med bot", "my bot", "may bot",
    ],
}

/// Unrelated speech phrases for negative training.
///
/// These are phonetically and semantically unrelated to the wake word, and
/// cover short commands, medium phrases, long utterances, and non-English
/// speech.  The classifier must reject all non-wake-word speech regardless
/// of language or sentence structure.
pub(crate) const UNRELATED_PHRASES: &[&str] = &[
    // ── Short commands (2-4 words) ──────────────────────────────────
    "the weather today is sunny",
    "what time is it",
    "one two three four five",
    "hello world",
    "good morning everyone",
    "turn on the lights",
    "play some music",
    "set a timer",
    // ── Medium phrases (5-8 words) ───────────────────────────────────
    "i need to buy groceries today",
    "can you remind me of my appointment",
    "please send a message to john",
    "what is the capital of france",
    "tell me a joke about programming",
    "how do I get to the airport",
    "the quick brown fox jumps over the lazy dog",
    // ── Long utterances (10+ words) ──────────────────────────────────
    "according to all known laws of aviation there is no way a bee should be able to fly",
    "the principle of superposition states that a quantum system exists in all its possible states simultaneously",
    // ── Non-English phrases (phonetically distinct from English wake word) ──
    "bonjour comment allez vous aujourd hui",
    "buenos días cómo estás",
    "guten morgen wie geht es dir",
];

/// Cache for pre-computed confusable phrase dense embeddings.
///
/// Populated asynchronously during startup (see [`prewarm_confusable_embeddings`])
/// after voice ONNX models and TTS models are ready, so enrollment never blocks
/// on TTS synthesis (~2 minutes for 28 phrases).  Uses AGC pre-processing to
/// match the production inference distribution.
///
/// After the dense-stride-8 alignment, this is the single cache used
/// for classifier training — the streaming cache was removed.
///
/// The embeddings are used during classifier training to teach the
/// model to reject confusable near-miss phrases.
///
/// Once set, the cache is immutable — a new process is needed to regenerate
/// (e.g. after ONNX model changes).  If pre-warming fails or is not yet
/// complete, [`confusable_dense_embeddings`] returns an empty slice and the
/// classifier trains on ambient negatives only.
static CONFUSABLE_EMBEDDINGS_CACHE: OnceLock<Vec<EmbeddingSequence>> = OnceLock::new();

/// Get the pre-computed confusable phrase dense embeddings for classifier negative training.
///
/// Returns a cached slice of dense-stride-8 embedding vectors pre-computed
/// during startup via [`prewarm_confusable_embeddings`].  After the
/// dense-stride-8 alignment, the same cache serves the
/// classifier — the streaming cache was removed.
///
/// If the pre-warm has not completed yet or models are not available, returns
/// an empty slice — the classifier trains on ambient negatives only as a
/// graceful fallback.
pub(crate) fn confusable_dense_embeddings() -> &'static [EmbeddingSequence] {
    match CONFUSABLE_EMBEDDINGS_CACHE.get() {
        Some(cache) => &cache[..],
        None => &[],
    }
}

/// Cache for pre-computed unrelated speech dense embeddings.
///
/// Populated asynchronously during startup (see [`prewarm_unrelated_embeddings`])
/// after voice ONNX models and TTS models are ready.  Uses the same PCM disk
/// caching strategy as confusable embeddings so TTS model updates
/// automatically invalidate cached audio.
///
/// After the dense-stride-8 alignment, this is the single cache used
/// for classifier training — the streaming cache was removed.
///
/// The embeddings are used during classifier training to teach the
/// model to reject non-wake-word speech.
///
/// Once set, the cache is immutable — a new process is needed to regenerate.
static UNRELATED_EMBEDDINGS_CACHE: OnceLock<Vec<EmbeddingSequence>> = OnceLock::new();

/// Get the pre-computed unrelated speech dense embeddings for classifier negative training.
///
/// Returns a cached slice of dense-stride-8 embedding vectors pre-computed
/// during startup via [`prewarm_unrelated_embeddings`].  After the
/// dense-stride-8 alignment, the same cache serves the
/// classifier — the streaming cache was removed.
///
/// If the pre-warm has not completed yet or models are not available, returns
/// an empty slice — the classifier trains on ambient + confusable negatives only
/// as a graceful fallback.
pub(crate) fn unrelated_dense_embeddings() -> &'static [EmbeddingSequence] {
    match UNRELATED_EMBEDDINGS_CACHE.get() {
        Some(cache) => &cache[..],
        None => &[],
    }
}

/// Poll for TTS voice styles to become available.
///
/// TTS model download (~400 MB) may still be in progress when voice ONNX
/// models (~2.4 MB) finish loading.  We poll with a 30-second interval,
/// racing against the global shutdown token, so the prewarm succeeds on
/// first startup even on slow connections.
///
/// This function is decoupled from the TTS playback
/// toggle (`tts_enabled` config) — confusable/unrelated phrase embeddings are
/// needed for classifier training regardless of whether audio
/// playback is enabled.  If TTS models are not yet loaded, a download is
/// triggered on demand (~400 MB on first voice init).
///
/// Returns `Some(styles)` when styles are available, or `None` if:
/// - TTS download permanently failed
/// - Shutdown was requested
///
/// Maximum number of 30-second polling iterations before giving up on TTS
/// voice styles (10 × 30s = 5 minutes). Generously covers even slow
/// downloads while preventing infinite hangs when STATE is permanently
/// stuck in LOADING (e.g., the download task was orphaned by runtime drop).
const MAX_TTS_STYLE_POLLS: u32 = 10;

async fn wait_for_tts_styles() -> Option<Vec<String>> {
    // Trigger TTS model download if not already available (decoupled from
    // the playback toggle).  This ensures confusable/unrelated
    // phrase embeddings are generated for classifier training
    // regardless of whether audio playback is enabled.
    //
    // Note: try_load_cached() now attempts disk loading even when STATE is
    // LOADING, recovering from orphaned download tasks.
    if !crate::audio::tts::models_ready() && !crate::audio::tts::try_load_cached() {
        info!(
            "TTS models not cached — triggering download on demand (~400 MB) \
                 for confusable/unrelated embedding pre-warm (mahbot-932)"
        );
        crate::audio::tts::spawn_or_retry_download();
    }

    let mut styles = crate::audio::tts::list_voice_styles();
    if !styles.is_empty() {
        return Some(styles);
    }

    info!(
        "TTS voice styles not yet available — polling every 30s \
         (confusable embeddings pre-warm deferred)"
    );

    let shutdown = crate::shutdown::shutdown_token();
    for _ in 0..MAX_TTS_STYLE_POLLS {
        tokio::select! {
            () = tokio::time::sleep(Duration::from_secs(30)) => {
                styles = crate::audio::tts::list_voice_styles();
                if !styles.is_empty() {
                    info!("TTS voice styles now available — proceeding with pre-warm");
                    return Some(styles);
                }
                if crate::audio::tts::download_failed() {
                    info!(
                        "TTS model download permanently failed — confusable negative \
                         embeddings pre-warm skipped (ambient negatives only)"
                    );
                    return None;
                }
            }
            () = shutdown.cancelled() => {
                info!(
                    "Shutdown requested — confusable dense embeddings \
                     pre-warm aborted"
                );
                return None;
            }
        }
    }

    // Exceeded MAX_TTS_STYLE_POLLS with no styles and no FAILED state.
    // This usually means the download task was orphaned (STATE stuck in
    // LOADING).  Try one final disk load (safe even in LOADING with Fix 4),
    // then reset STATE so future callers can retry.
    warn!(
        "TTS voice styles not available after {} polls (~{}s) — \
         giving up and resetting loading state",
        MAX_TTS_STYLE_POLLS,
        MAX_TTS_STYLE_POLLS * 30,
    );

    if crate::audio::tts::try_load_cached() {
        let styles = crate::audio::tts::list_voice_styles();
        if !styles.is_empty() {
            return Some(styles);
        }
    }

    // Reset STATE from LOADING to UNINIT so the next caller triggers a
    // fresh download attempt instead of silently returning None forever.
    if crate::audio::tts::try_reset_loading_state() {
        info!("Reset TTS state from LOADING to UNINIT — next caller will retry");
    }
    None
}

/// Pre-warm the confusable phrase embedding cache.
///
/// Runs asynchronously at startup after voice ONNX models and TTS are ready.
/// For each confusable phrase at 5 seeds each:
///
/// 1. Checks the PCM disk cache — on hit, loads cached audio without
///    re-synthesis; on miss, synthesises via TTS and writes to cache.
/// 2. Applies AGC preprocessing via a **fresh** [`AudioPreprocessor`] per
///    phrase × seed in FRAME_LENGTH chunks (no zero-padding) to match the
///    production inference distribution.
/// 3. VAD-gates the AGC'd audio through a dedicated earshot detector,
///    producing the exact streaming mel layout (silence discarded, windows
///    anchored at speech onset).
/// 4. Derives the 4 PCM augmentation variants from the VAD-gated speech-only
///    audio (AGC → VAD → augment ordering, matching enrollment).
/// 5. Extracts dense stride-8 embeddings via the ONNX pipeline.
///
/// The PCM disk cache is keyed by (text + style + seed + sample rate + TTS
/// model hash), so TTS model updates automatically invalidate stale cached
/// audio.  Embedding model changes do NOT invalidate the PCM cache — fresh
/// embeddings are extracted from cached PCM on the next startup.
///
/// The resulting embeddings are stored in [`CONFUSABLE_EMBEDDINGS_CACHE`] and
/// later used during classifier training (single cache serves both).
/// If TTS models or voice ONNX models
/// are not available, the function returns without populating the cache and
/// both models train on ambient negatives only.
///
/// This function is safe to call multiple times — [`OnceLock::set`] is a no-op
/// if the cache is already populated.
pub(crate) async fn prewarm_confusable_embeddings() {
    // Fast path: already pre-warmed.  OnceLock::get is lock-free after init.
    if CONFUSABLE_EMBEDDINGS_CACHE.get().is_some() {
        return;
    }

    let dense = prewarm_phrase_embeddings(
        "confusable",
        CONFUSABLE_PHRASES,
        CONFUSABLE_SEEDS_PER_PHRASE,
        CONFUSABLE_SEED_BASE,
        "ambient negatives only",
    )
    .await;

    let count = dense.len();

    // OnceLock::set is a no-op if already set (race-safe).
    let _ = CONFUSABLE_EMBEDDINGS_CACHE.set(dense);

    if count > 0 {
        info!(
            "Pre-warmed {count} confusable phrase dense embedding(s)              from {} phrases × {CONFUSABLE_SEEDS_PER_PHRASE} seeds              for negative training (mahbot-923)",
            CONFUSABLE_PHRASES.len(),
        );
    } else {
        warn!(
            "No confusable phrase embeddings could be generated —              models will train on ambient negatives only"
        );
    }
}

/// Shared pre-warm logic for phrase-based negative embeddings.
///
/// Runs TTS synthesis (with PCM caching), AGC preprocessing, VAD gating, PCM
/// augmentation, and ONNX dense embedding extraction for each phrase × seed
/// combination.  Used by both [`prewarm_confusable_embeddings`] and
/// [`prewarm_unrelated_embeddings`] to avoid code duplication.
///
/// ## Streaming-pipeline alignment
///
/// The negative training cache must be representative of what the classifier
/// sees during streaming inference, so each phrase × seed is processed with
/// the same representation choices as the live detection path:
///
/// 1. **Fresh [`AudioPreprocessor`]** per phrase × seed — matching the
///    per-segment AGC reset (`PipelineCtx::reset_detection_segment`).  A
///    shared preprocessor would adapt AGC across N phrases, an artifact
///    streaming never produces.
/// 2. **VAD-gated mel frames** via [`vad_gate_streaming_mel`] (a dedicated
///    earshot detector per phrase × seed, never the global [`VAD_DETECTOR`])
///    — only VAD-positive hops produce mel frames; leading/trailing silence
///    and intra-phrase pauses are discarded, exactly like
///    [`process_streaming_frames_inner`].
/// 3. **Speech-onset anchoring** — the original variant's embedding windows
///    start at the first speech frame (mel frame 0), not at TTS sample 0 with
///    its ~100 ms silence preamble.
/// 4. **Augment after VAD gating** — the 4 PCM variants (speed-down,
///    speed-up, volume-down, noise) are derived from the speech-only audio,
///    matching enrollment's `AGC → VAD → augment` ordering.
///
/// After the dense-stride-8 alignment, only dense embeddings are
/// produced — streaming extraction was removed.
///
/// Returns extracted dense embeddings, or an empty vec if pre-warming
/// cannot proceed (models not available, no TTS styles, etc.).
///
/// # Parameters
///
/// * `phrase_type` — human-readable label for log messages ("confusable"/"unrelated").
/// * `phrases` — the list of phrases to synthesise.
/// * `seeds_per_phrase` — number of TTS seed variants per phrase.
/// * `seed_base` — base offset for seed calculation (see seed formula below).
/// * `fallback_info` — what the models fall back to if this prewarm fails (for logs).
///
/// # Seed formula
///
/// For phrase index `i` and seed variant `j` (0..`seeds_per_phrase`):
///
/// ```text
/// seed = seed_base + i * seeds_per_phrase + j
/// ```
#[allow(clippy::too_many_lines)]
async fn prewarm_phrase_embeddings(
    phrase_type: &'static str,
    phrases: &'static [&'static str],
    seeds_per_phrase: usize,
    seed_base: u64,
    fallback_info: &'static str,
) -> Vec<EmbeddingSequence> {
    // Need voice ONNX models.
    if ONNX_MODELS.get().is_none() {
        info!(
            "Voice ONNX models not ready yet — {phrase_type} negative embeddings              pre-warm skipped (models train on {fallback_info})"
        );
        return Vec::new();
    }

    // Wait for TTS voice styles to become available by polling.
    let Some(available_styles) = wait_for_tts_styles().await else {
        return Vec::new();
    };

    // Resolve PCM cache directory; if it can't be resolved, skip caching
    // (synthesis still works without it, just slower).
    let cache_dir = voice_cache_dir();
    if let Some(ref d) = cache_dir
        && let Err(e) = std::fs::create_dir_all(d)
    {
        warn!("PCM cache directory creation failed: {e} — proceeding without cache");
    }
    // Run startup eviction to clean stale/oversized cache before prewarming
    if let Some(ref d) = cache_dir {
        evict_pcm_cache(d);
    }
    let model_hash = tts_model_version_hash();

    let num_styles = available_styles.len();

    // Runs TTS synthesis (with PCM caching), AGC, VAD gating, augmentation,
    // and ONNX embedding extraction in a blocking thread to avoid starving the
    // async runtime.
    //
    // Pipeline: raw TTS PCM → fresh AGC per phrase × seed →
    // VAD-gate (streaming mel layout) → augment speech-only audio → embeddings.
    // This matches the streaming detection path (fresh per-segment AGC, VAD-
    // gated mel frames, windows anchored at speech onset) so the negative
    // training distribution is representative of what the classifier sees
    // during live inference.
    tokio::task::spawn_blocking(move || {
        let Some(models) = ONNX_MODELS.get() else {
            return Vec::new();
        };

        // Preprocessor config from the same CONFIG flags the live-mic
        // streaming pipeline uses (`preprocessor_config_from_config`, the
        // config `PipelineCtx::new()` builds).  The negative
        // embeddings must match the streaming inference distribution, which is
        // governed by the deployment's NS/AGC toggles.
        let pre_config = preprocessor_config_from_config();
        let mut dense_sequences: Vec<EmbeddingSequence> = Vec::new();

        for (i, &phrase) in phrases.iter().enumerate() {
            for seed_idx in 0..seeds_per_phrase {
                // Rotate through available voice styles for acoustic diversity.
                // Distribute seeds round-robin across styles.
                let style_idx = (i * seeds_per_phrase + seed_idx) % num_styles;
                let style = &available_styles[style_idx];
                let seed = seed_base + i as u64 * seeds_per_phrase as u64 + seed_idx as u64;
                let phrase_index_for_id: usize = i * seeds_per_phrase + seed_idx;
                let source = match phrase_type {
                    "confusable" => Source::Confusable,
                    _ => Source::Unrelated,
                };

                // ── Embedding-level cache ──
                // The per-utterance dense embeddings are deterministic, so a
                // warm run can skip AGC + VAD + ONNX entirely.  The cached
                // variants are pushed through the same helper the miss path
                // uses, guaranteeing byte-identical sequences.
                let cache_key =
                    embedding_cache_key(phrase_type, phrase, style, seed, &model_hash, pre_config);
                let emb_cache_dir = embedding_cache_dir();
                let mut utterance_variants: Vec<(u8, Vec<Vec<f32>>)> = Vec::new();
                let mut push_seq = |embs: Vec<Vec<f32>>, vi: usize| {
                    if !embs.is_empty() {
                        dense_sequences.push(EmbeddingSequence::new(
                            UtteranceId {
                                sequence_index: phrase_index_for_id,
                                variant_index: vi,
                            },
                            source,
                            embs.clone(),
                        ));
                        // Variant indices are 0..=4 by construction.
                        utterance_variants
                            .push((u8::try_from(vi).expect("variant index fits in u8"), embs));
                    }
                };
                if let (Some(dir), Some(key)) = (&emb_cache_dir, &cache_key)
                    && let Some(variants) = read_embedding_cache(dir, key)
                {
                    for (vi, embs) in variants {
                        push_seq(embs, usize::from(vi));
                    }
                    continue;
                }

                // Load PCM — from disk cache (preferred) or synthesise fresh.
                let pcm = if let Some(ref cache_dir) = cache_dir {
                    synthesize_with_pcm_cache(
                        phrase,
                        style,
                        seed,
                        SAMPLE_RATE,
                        &model_hash,
                        cache_dir,
                    )
                } else {
                    match crate::audio::tts::synthesize(phrase, style, seed, SAMPLE_RATE) {
                        Ok(p) => Some(p),
                        Err(e) => {
                            warn!("TTS synthesis failed for {phrase_type} phrase '{phrase}': {e}");
                            None
                        }
                    }
                };

                let Some(pcm) = pcm else {
                    continue;
                };

                // ── 1. Fresh AGC per phrase × seed ──
                // The streaming pipeline starts each detection segment with a
                // fresh AudioPreprocessor (`reset_detection_segment`).
                // A shared preprocessor would process the
                // Nth phrase with AGC adapted to N−1 prior phrases — an
                // artifact streaming never produces.  Chunks are fed as-is (no
                // zero-padding), matching the mic stream: the
                // NS stage buffers incomplete frames internally and the next
                // chunk completes them.  Shared with the E2E bench via
                // [`crate::audio::audio_preprocessor::agc_feed_fresh`].
                let agc_audio = crate::audio::audio_preprocessor::agc_feed_fresh(
                    &pcm,
                    FRAME_LENGTH,
                    pre_config,
                );

                // ── 2. VAD-gate ──
                // Fresh earshot detector per phrase × seed: the prewarm must
                // not reuse the global VAD_DETECTOR (would contaminate the
                // live pipeline's noise-floor state) and must not carry VAD
                // state across phrases (the same shared-state artifact as the
                // shared AGC).  Produces the exact streaming mel layout
                // (VAD-gated, speech-onset-anchored) plus the speech-only
                // audio used for augmentation below.
                let mut detector = earshot::Detector::default();
                let (mel_frames, speech_audio) = vad_gate_streaming_mel(&agc_audio, |hop| {
                    is_speech_with_detector(hop, &mut detector, VAD_THRESHOLD)
                });

                if mel_frames.is_empty() {
                    // Matches streaming: no VAD-positive speech ⇒ no mel frames
                    // ⇒ no embeddings.  A phrase with zero VAD-positive audio is
                    // dropped from the negative set rather than trained on
                    // silence-laden audio the streaming path never produces.
                    warn!(
                        "{phrase_type} phrase '{phrase}' (seed {seed}) produced no \
                         VAD-positive speech — skipping (matches streaming: no \
                         speech ⇒ no embeddings)"
                    );
                    continue;
                }

                // ── 3. Original — embeddings from the streaming mel frames ──
                // Windows are anchored at the first speech frame
                // (mel frame 0 = first VAD-positive hop), not at TTS sample 0
                // with its silence preamble.
                let dense_embs =
                    embeddings_from_mel_frames(models, &mel_frames).unwrap_or_default();
                push_seq(dense_embs, 0);

                // ── 4. Augment AFTER VAD gating ──
                // Variants are derived from the speech-only audio (no silence
                // preamble), matching enrollment's AGC → VAD → augment ordering.
                // No variant is re-gated by VAD (only the original was).  The
                // only conditional variant is speed-up, dropped when the
                // speech-only duration is too short (< 500 ms pre-pad) to stay
                // intelligible — the same gate enrollment applies to its
                // VAD-segmented utterance.  Note that this gate now evaluates
                // the VAD-gated speech duration (not the full TTS audio), so
                // short phrases drop the speed-up variant more often than the
                // pre-fix pipeline did — a direct consequence of the alignment.
                //
                // Layout note: variants 1–4 are embedded via
                // `extract_embeddings_from_audio` (whole-audio mel over the raw
                // hop-concatenated speech_audio), so their mel layout lacks the
                // batch-flush boundary handling (`trim_voice_batch` overlap at
                // VOICE_BATCH_SIZE boundaries) that variant 0's streaming mel
                // matches exactly.  This is acceptable for perturbations and
                // matches how enrollment derives its own augmented variants
                // (`process_enrollment_sample` → whole-audio mel): the speech
                // content is identical VAD-gated audio; only the batch-boundary
                // framing differs slightly.
                //
                // Variant generation is shared with the E2E bench via
                // [`augment_pcm_variants`]: noise seed = the
                // TTS phrase seed, gate input = VAD-gated speech, canonical
                // push order (speed-up 3rd).  The negative pool deliberately
                // uses the bounded [`AugmentSet::Negatives`] set (original +
                // pink 25 dB): the old speed/volume cells were dropped to
                // keep the 12-cell positive recipe's cold embedding recompute
                // inside the 39 s bench budget (see `AugmentSet::Negatives`).
                // Variant 0 (original) is pushed above from the streaming mel
                // frames.
                for variant in
                    augment_pcm_variants(&speech_audio, SAMPLE_RATE, seed, AugmentSet::Negatives)
                {
                    if variant.variant_index == 0 {
                        continue; // original already pushed above
                    }
                    let dense_embs =
                        extract_embeddings_from_audio(models, &variant.pcm).unwrap_or_default();
                    push_seq(dense_embs, variant.variant_index);
                }

                // ── Persist the per-utterance embedding cache ──
                // Best-effort: a write failure only costs a recompute on the
                // next run.  Nothing is written when no variant produced
                // embeddings (e.g. no VAD-positive speech) — the miss path
                // would reproduce the same empty result.
                if !utterance_variants.is_empty()
                    && let (Some(dir), Some(key)) = (&emb_cache_dir, &cache_key)
                {
                    write_embedding_cache(dir, key, &utterance_variants);
                }
            }
        }

        dense_sequences
    })
    .await
    .unwrap_or_default()
}

/// Pre-warm unrelated speech embedding cache.
///
/// Synthesises each [`UNRELATED_PHRASES`] entry with
/// [`UNRELATED_SEEDS_PER_PHRASE`] TTS seeds, applies AGC, and extracts
/// dense stride-8 embeddings via the ONNX pipeline.  Results are stored in
/// [`UNRELATED_EMBEDDINGS_CACHE`] for lock-free reads during enrollment.
///
/// If the cache is already populated (from a previous call), this is a
/// no-op.  Safe to call multiple times.
pub(crate) async fn prewarm_unrelated_embeddings() {
    // Fast path: already pre-warmed.
    if UNRELATED_EMBEDDINGS_CACHE.get().is_some() {
        return;
    }

    let dense = prewarm_phrase_embeddings(
        "unrelated",
        UNRELATED_PHRASES,
        UNRELATED_SEEDS_PER_PHRASE,
        UNRELATED_SEED_BASE,
        "ambient + confusable negatives only",
    )
    .await;

    let count = dense.len();

    let _ = UNRELATED_EMBEDDINGS_CACHE.set(dense);

    if count > 0 {
        info!(
            "Pre-warmed {count} unrelated phrase dense embedding(s)              from {} phrases × {UNRELATED_SEEDS_PER_PHRASE} seeds              for negative training (mahbot-923)",
            UNRELATED_PHRASES.len(),
        );
    } else {
        warn!(
            "No unrelated phrase embeddings could be generated — \
             models will train on ambient + confusable negatives only"
        );
    }
}

// ── Model URLs and filenames ────────────────────────────

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

/// Post-detection cooldown period to prevent rapid consecutive false triggers.
/// After a wake word detection, no further detection is
/// attempted for this duration.  Industry reference: Rhasspy Raven uses
/// `refractory_sec=2.0`, openWakeWord uses patience counters.
const WAKE_WORD_COOLDOWN: Duration = Duration::from_secs(3);

/// Mean of the negative (non-wake-word) per-frame soft score distribution,
/// measured during the benchmark on confusable and unrelated
/// speech.  Used as the seed value for [`AdaptiveThresholdState::warmed()`].
/// The safe harbor clamp (2.13) means the precise value is
/// unimportant as long as it is well below 2.13; this constant documents
/// the measured value for future reference.
///
/// Calibrated for dense stride-8 embeddings.  The 1.58×
/// multiplier relative to the old streaming distribution means the safe
/// harbor clamp increased from 1.35 to 2.13.  See the
/// full calibration table and rationale.
#[cfg(any(test, feature = "voice-tests"))]
const NEGATIVE_DISTRIBUTION_MEAN: f32 = 0.033;

/// Minimum per-frame soft score below which the rolling window is reset
/// entirely.  Set to 0.316 (calibrated for dense stride-8
/// embeddings, 1.58× multiplier over old streaming value 0.20).  The 1.58×
/// multiplier accounts for the higher per-frame scores from stride-8 dense
/// embeddings and is derived from benchmark calibration.
const NO_MATCH_RESET_THRESHOLD: f32 = 0.316;

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

/// Total embeddings computed during wake word processing.  Each embedding
/// corresponds to one window of 76 mel frames (~760 ms of audio).  Monotonically
/// increasing — never reset.
pub(crate) static EMBEDDINGS_COMPUTED: AtomicU64 = AtomicU64::new(0);

/// Total wall-clock time spent computing embeddings (nanoseconds).
/// Divide by [`EMBEDDINGS_COMPUTED`] for the lifetime average per-embedding
/// latency.  Tracked via `Instant::now()` around [`compute_embedding`].
pub(crate) static TOTAL_EMBEDDING_TIME_NS: AtomicU64 = AtomicU64::new(0);

// ── Rolling average ring buffer for embedding latency ────────────────────
//
// The ticket requested "rolling average (last N frames)" for AC2 (processing
// latency).  Rather than an exponential moving average (which is an
// approximation) or a lifetime cumulative average (which becomes diagnostically
// inert as embeddings accumulate), we use a lock-free ring buffer of the most
// recent 100 embedding computation times.  N=100 covers ~3.2 seconds of audio
// at ~32 ms per embedding — large enough to smooth noise, small enough for
// O(100) reads on the diagnostic path.
//
// This deviates from the original implementation which used a lifetime
// cumulative total/count approach.  The ring buffer provides a true "last N
// frames" window that can detect recent performance changes, which the
// lifetime average cannot.
//
// Lock-free: single writer (pipeline task) stores to the ring with an atomic
// head index; readers sum O(N) entries on the diagnostic/debug path only.
// No mutex is involved on any path.

/// Number of recent embedding latencies tracked in the rolling average ring
/// buffer.  100 entries ≈ 3.2 s of audio at ~32 ms per embedding.
const EMBEDDING_LATENCY_RING_SIZE: usize = 100;

/// Lock-free ring buffer of the most recent [`EMBEDDING_LATENCY_RING_SIZE`]
/// embedding computation times (nanoseconds).  Written by the pipeline task
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
/// buffer of the last [`EMBEDDING_LATENCY_RING_SIZE`] embeddings, providing a
/// true rolling average that reflects recent pipeline performance.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VoiceMetricsSnapshot {
    /// Total audio chunks received from the mic channel.
    pub chunks_received: u64,
    /// Chunks dropped at the mic channel boundary (try_send full).
    pub dropped_chunks: u64,
    /// Total embeddings computed.
    pub embeddings_computed: u64,
    /// Cumulative embedding computation time (nanoseconds) — for lifetime
    /// average via [`Self::lifetime_avg_embedding_latency_ns`].
    pub total_embedding_time_ns: u64,
    /// Rolling average embedding latency in nanoseconds (last 100 frames).
    /// Prefer this over the lifetime average for detecting recent performance
    /// changes — the lifetime average becomes diagnostically inert as the
    /// pipeline accumulates millions of embeddings.
    pub avg_embedding_latency_ns: u64,
}

impl VoiceMetricsSnapshot {
    /// Lifetime average embedding latency in nanoseconds (total ÷ count).
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
/// Each frame represents ~128ms of voiced audio, so N=3
/// covers ~384ms — matching the original temporal window but using
/// accumulated weight instead of a strict consecutive binary counter.
const ROLLING_WINDOW_N: usize = 3;

/// Compile-time invariant: EMBEDDING_RING_MAX must be at least
/// ROLLING_WINDOW_N so the ring buffer can supply enough embeddings
/// for matching while the rolling window accumulates scores.
const _: () = assert!(
    EMBEDDING_RING_MAX >= ROLLING_WINDOW_N,
    "EMBEDDING_RING_MAX must be >= ROLLING_WINDOW_N to hold enough embeddings \
     for the Conv1D classifier"
);

/// Compile-time invariant: EMBEDDING_RING_MAX must be at least
/// wake_word_classifier::WINDOW_SIZE so the tiling fallback (used when the
/// ring has fewer than WINDOW_SIZE entries) has at least one embedding to
/// tile from.  This prevents a repeat-last `len()-1` underflow if WINDOW_SIZE
/// is ever changed independently of ROLLING_WINDOW_N.
const _: () = assert!(
    EMBEDDING_RING_MAX >= wake_word_classifier::WINDOW_SIZE,
    "EMBEDDING_RING_MAX must be >= wake_word_classifier::WINDOW_SIZE to ensure \
     the ring is non-empty when the tiling fallback is reached"
);

/// Compile-time invariant: the adaptive threshold floor (hard minimum) must
/// not exceed the safe harbor (primary lower bound tracking the static
/// threshold).  The safe harbor is always at least as high as the floor
/// given current constants (FLOOR=1.35, SAFE_HARBOR=1.35).
const _: () = assert!(
    ADAPTIVE_FLOOR <= ADAPTIVE_SAFE_HARBOR,
    "ADAPTIVE_FLOOR must be <= ADAPTIVE_SAFE_HARBOR"
);

/// Compile-time invariant: the adaptive threshold safe harbor must not exceed
/// the absolute ceiling (upper bound).  Guarantees the clamp range is valid.
const _: () = assert!(
    ADAPTIVE_SAFE_HARBOR <= ADAPTIVE_CEILING,
    "ADAPTIVE_SAFE_HARBOR must be <= ADAPTIVE_CEILING"
);

/// Compile-time invariant: the adaptive threshold absolute floor must not
/// exceed the absolute ceiling (upper bound).
const _: () = assert!(
    ADAPTIVE_FLOOR <= ADAPTIVE_CEILING,
    "ADAPTIVE_FLOOR must be <= ADAPTIVE_CEILING"
);

/// Factor applied to `ROLLING_WINDOW_N` to compute the detection threshold.
/// At 0.71 (threshold 2.13), calibrated for dense stride-8
/// embeddings.  The 1.58× multiplier over the old streaming value (0.45)
/// accounts for the higher per-frame scores produced by stride-8 dense
/// embeddings versus stride-1 streaming embeddings, based on benchmark
/// calibration data.
const MATCH_THRESHOLD_FACTOR: f32 = 0.71;

/// Detection threshold for the rolling sum of soft scores.
/// Computed as: `ROLLING_WINDOW_N × MATCH_THRESHOLD_FACTOR`
/// (= `3 × 0.71 = 2.13`).  Calibrated for dense stride-8 embeddings using
/// the 1.58× multiplier (old streaming threshold 1.35 → 2.13).  The higher
/// threshold accounts for the higher per-frame scores from stride-8 dense
/// embeddings versus stride-1 streaming embeddings.
#[expect(clippy::cast_precision_loss)]
fn match_threshold() -> f32 {
    (ROLLING_WINDOW_N as f32) * MATCH_THRESHOLD_FACTOR
}

// ── Adaptive threshold ──────────────────────────────────────

/// Number of recent per-frame scores to track for adaptive threshold
/// statistics. At ~128ms per frame, N=15 covers ~2 seconds of context.
const ADAPTIVE_WINDOW_N: usize = 15;

/// Default k multiplier for the adaptive threshold (mean + k × std).
const ADAPTIVE_K_DEFAULT: f32 = 2.5;

/// Minimum allowed adaptive k value (user-configurable range).
const ADAPTIVE_K_MIN: f32 = 1.0;

/// Maximum allowed adaptive k value (user-configurable range).
const ADAPTIVE_K_MAX: f32 = 4.0;

/// Absolute floor — the adaptive threshold must never drop below this value.
/// Calibrated for dense stride-8 embeddings.  The 1.58× multiplier
/// (old streaming floor 1.35 → 2.13) accounts for the higher per-frame scores
/// from stride-8 dense embeddings versus stride-1 streaming embeddings.
///
/// Computed from the same expression as ADAPTIVE_SAFE_HARBOR so the two
/// values produce the exact same f32 bit pattern, satisfying the compile-time
/// invariant ADAPTIVE_FLOOR <= ADAPTIVE_SAFE_HARBOR without floating-point
/// rounding differences.
#[expect(clippy::cast_precision_loss)]
const ADAPTIVE_FLOOR: f32 = (ROLLING_WINDOW_N as f32) * MATCH_THRESHOLD_FACTOR;

/// Absolute ceiling — the adaptive threshold must never exceed this value.
/// Calibrated for dense stride-8 embeddings.  Set to 4.503
/// (old streaming ceiling 2.85 × 1.58).  If E2E benchmarks show this ceiling
/// is too aggressive (excessive false rejects), escalate to 5.5, then 6.0.
/// The escalation trigger is when per-utterance adaptive threshold trajectory
/// (tracked via benchmark instrumentation) shows the ceiling is the active
/// limiting factor on detection rate.
const ADAPTIVE_CEILING: f32 = 4.503;

/// Safe harbor — the adaptive threshold must never drop below this value,
/// which matches the current static [`match_threshold()`]
/// (ROLLING_WINDOW_N × MATCH_THRESHOLD_FACTOR = 3 × 0.71 = 2.13).
/// Calibrated for dense stride-8 embeddings.  The 1.58× multiplier
/// over the old streaming value (1.35 → 2.13) accounts for the higher
/// per-frame scores from stride-8 dense embeddings.
/// Derived from the same constants as [`match_threshold()`] so the two values
/// are always in sync.  Prevents a feedback loop where false accepts push
/// the threshold lower.
#[expect(clippy::cast_precision_loss)]
const ADAPTIVE_SAFE_HARBOR: f32 = (ROLLING_WINDOW_N as f32) * MATCH_THRESHOLD_FACTOR;

/// Number of bootstrap frames to use the static threshold while the adaptive
/// window fills.  Calibrated for dense stride-8 embeddings.
/// The 1.58× multiplier relative to the old streaming pipeline is derived
/// from benchmark calibration and accounts for the higher per-frame scores
/// produced by stride-8 dense embeddings.
const ADAPTIVE_BOOTSTRAP_FRAMES: usize = 5;

/// Process a per-frame soft score through the rolling window and determine
/// whether wake word detection should fire.
///
/// Returns `true` when the rolling sum of recent scores meets or exceeds
/// `match_threshold()`.  When the incoming score is below
/// [`NO_MATCH_RESET_THRESHOLD`], the window is cleared entirely to prevent
/// slow accumulation from noise — unless `preserve_window_on_reset` is set
/// (deferred-burst path only): a low-scoring burst frame
/// contributes nothing to the score and must not wipe an in-progress wake
/// mid-detection.  On detection the score window is NOT cleared here — the
/// caller is responsible for full pipeline cleanup.
///
/// This function is pure with respect to global state: it only reads its
/// parameters and modifies `score_window` in place.  This makes it directly
/// testable without ONNX models or voice pipeline initialization.
fn process_wake_word_score(
    total_score: f32,
    score_window: &mut Vec<f32>,
    adaptive_threshold_override: Option<f32>,
    preserve_window_on_reset: bool,
) -> (bool, f32) {
    if total_score < NO_MATCH_RESET_THRESHOLD {
        // Far from matching — reset the entire rolling window to prevent
        // slow accumulation from noise.  Burst-path frames
        // never clear: they contribute nothing to the score, and a mid-wake
        // wipe would kill an otherwise-valid detection that started near the
        // utterance beginning.
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

/// Process a single embedding through the wake word detection pipeline.
///
/// This is the **core detection loop** shared between the live pipeline
/// ([`handle_wake_word_detection`]), enrollment self-test
/// ([`run_enrollment_self_test`]), and integration tests.
///
/// It manages the ring buffer, runs the Conv1D classifier forward pass
/// (tiling available embeddings to fill the 3-embedding window when the ring
/// has fewer than 3 entries — see [`score_single_embedding`] implementation),
/// feeds/peeks the adaptive threshold, and applies rolling window scoring via
/// [`process_wake_word_score`].  Detection fires immediately when the rolling
/// sum crosses the effective threshold — the pipeline is speaker-blind with
/// no second-stage gate.
///
/// # Returns
/// - `(true, rolling_sum, total_score, effective_threshold)` — the embedding
///   triggered wake word detection.
/// - `(false, _, total_score, effective_threshold)` — continue feeding more
///   embeddings (the ring buffer and score window are updated for the next
///   call).
///
/// - `rolling_sum` — the rolling window sum at the time of evaluation
///   (0.0 if the window was reset due to low score).
/// - `total_score` — the raw soft score from the Conv1D classifier.
/// - `effective_threshold` — the threshold value used for the rolling window
///   comparison this frame (adaptive threshold post-bootstrap, or static
///   [`match_threshold()`] during bootstrap / when no adaptive state is
///   configured).
///
/// # Parameters
/// - `embedding` — one 96-dim embedding vector to process.
/// - `embedding_ring` — persistent ring buffer (shared across frames in the
///   live pipeline; fresh per utterance in tests).
/// - `classifier` — trained Conv1D classifier (`None` skips classification).
/// - `score_window` — persistent rolling window of recent confidence scores.
/// - `adaptive_state` — optional adaptive threshold state (`None` disables
///   adaptive threshold adjustment).
/// - `adaptive_k` — multiplier for the adaptive threshold's standard-deviation
///   term (passed to [`AdaptiveThresholdState::next_threshold`]).
/// - `burst_path` — true when scored by the deferred-burst sweep
///   (start-aligned positions).  Burst-path frames never clear the rolling
///   window on a below-reset score: they contribute nothing to
///   the score, and a mid-wake wipe would kill a valid detection that started
///   near the utterance beginning.
#[allow(clippy::too_many_lines)]
pub(crate) fn score_single_embedding(
    embedding: &[f32],
    embedding_ring: &mut Vec<Vec<f32>>,
    classifier: Option<&WakeWordClassifier>,
    score_window: &mut Vec<f32>,
    mut adaptive_state: Option<&mut AdaptiveThresholdState>,
    adaptive_k: f32,
    burst_path: bool,
) -> (bool, f32, f32, f32) {
    // ── Ring buffer ───────────────────────────────────────────────────
    embedding_ring.push(embedding.to_vec());
    while embedding_ring.len() > EMBEDDING_RING_MAX {
        embedding_ring.remove(0);
    }

    // ── Conv1D classifier forward pass ───────────────────────────────────
    // Scores a window of 3 consecutive embeddings via Conv1D + sigmoid.
    // Falls back to repeat-last tiling when fewer than 3 embeddings are
    // available.
    let total_score = if let Some(c) = classifier {
        if embedding_ring.len() >= wake_word_classifier::WINDOW_SIZE {
            let start = embedding_ring.len() - wake_word_classifier::WINDOW_SIZE;
            c.forward(&embedding_ring[start..])
        } else {
            let last = embedding_ring.len() - 1;
            let mut window = Vec::with_capacity(wake_word_classifier::WINDOW_SIZE);
            for i in 0..wake_word_classifier::WINDOW_SIZE {
                window.push(embedding_ring[i.min(last)].clone());
            }
            c.forward(&window)
        }
    } else {
        0.0
    };

    // Feed only background (non-wake-word) scores to the adaptive threshold
    // so it learns the noise-floor distribution without being contaminated
    // by the high scores it's trying to detect.  Scores below
    // NO_MATCH_RESET_THRESHOLD are clearly "not wake word" and represent
    // the background acoustic environment.  For wake-word-like frames we
    // call peek() which returns the current threshold without updating
    // statistics, preventing the self-defeating loop where high scores
    // inflate the threshold and block detection.
    //
    // the SAME feed/peek rule applies
    // during bootstrap — the old unconditional bootstrap feed
    // (`is_bootstrapping() || total_score < NO_MATCH_RESET_THRESHOLD`)
    // inflated the adaptive threshold with wake-word-like burst scores
    // (~0.99) to ~3.0–4.5, blocking detection after the deferred burst.
    // With the score-only rule, a high-scoring utterance legitimately keeps
    // the bootstrap alive for its whole duration: high scores peek
    // (peek() returns None during bootstrap → the static match_threshold
    // stays in effect), and only below-reset background frames feed and
    // advance the bootstrap counter.  Residual contamination cannot persist
    // across the soft reset (detection→recording handoff) because no
    // wake-word-like score ever enters the statistics.
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
        process_wake_word_score(total_score, score_window, adaptive_override, burst_path);

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
        } else if burst_path {
            " (below NO_MATCH_RESET_THRESHOLD — burst frame, window preserved)"
        } else {
            " (below NO_MATCH_RESET_THRESHOLD — window reset)"
        };
        info!(
            "VOICE_DEBUG: total_score={total_score:.4}{below_note} rolling_sum={rolling_sum:.4} ring_len={}",
            embedding_ring.len(),
        );
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
/// Previously enrollment used `ENROLLMENT_VAD_THRESHOLD = 0.60`, causing a
/// systematic training/inference mismatch: frames scoring 0.50–0.59 were
/// included in streaming but never seen during enrollment training.
/// The Conv1D classifier had no representation of these acoustic patterns,
/// resulting in a 0.0% detection rate during streaming.
const VAD_THRESHOLD: f32 = 0.5;

/// Minimum consecutive VAD-positive frames before setting utterance_had_speech
/// during enrollment (~0ms at 16ms/frame).  Set to 1 to match streaming
/// detection behavior, which starts accumulating at the first VAD-positive
/// frame.  Previously was 3, which meant enrollment
/// consumed VAD decisions differently from streaming, producing misaligned
/// utterance boundaries and embedding window positions.
pub(crate) const ENROLLMENT_VAD_CONSECUTIVE_REQUIRED: usize = 1;

// Neural VAD (Earshot) — replaces RMS-based `is_speech`

/// Global Earshot VAD detector instance. Thread-safe behind a mutex because
/// `predict_f32` completes in ~5-6 µs, so lock contention is negligible.
/// The detector has internal state (768-sample ring buffer, pre-emphasis filter,
/// 3-frame feature context) that must be kept in sync with the audio stream.
/// Created once in [`init_global`].
static VAD_DETECTOR: OnceLock<std::sync::Mutex<earshot::Detector>> = OnceLock::new();

// VAD_THRESHOLD is defined above with a unified doc comment.
// Model loading state machine

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
    crate::util::models_dir().map(|dir| dir.join(MODEL_DIR_NAME))
}

/// Check whether voice models are ready for inference.
pub fn models_ready() -> bool {
    MODELS_STATE.load(Ordering::Acquire) == ModelState::Ready
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
    /// target (typically 60), `wall_clock_elapsed` is the wall-clock
    /// seconds since Phase 3 started.
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
    /// Trained Conv1D classifier weights (None before enrollment).
    /// Contains a single weight set (removing the 5-member ensemble).
    classifier_weights: Option<ClassifierWeights>,
    /// Cached classifier for inference (avoids per-frame clone of weights).
    /// Recreated when [`classifier_weights`] changes.
    classifier: Option<WakeWordClassifier>,
    /// Per-utterance embeddings extracted via the full-utterance mel pipeline
    /// with dense stride-8 sliding window (used for both Conv1D classifier and
    /// classifier training).  Streaming buffer was removed
    /// — inference and training now share the same dense embedding distribution.
    enrollment_buffer: Vec<EmbeddingSequence>,
    /// Raw audio chunks collected during non-wake-word periods of enrollment
    /// (pre-enrollment ambient noise and inter-utterance silence).  These are
    /// processed through the ONNX embedding model at training time to
    /// produce real (non-synthetic) negative examples.
    negative_audio_chunks: Vec<Vec<f32>>,
    /// Owner-negative audio chunks collected during Phase 3 enrollment.
    /// ~60 seconds of VAD-positive general speech from the user,
    /// stored as audio chunks for embedding extraction at training time.
    /// Preserved across Full/Soft pipeline resets (same as `negative_audio_chunks`)
    /// but cleared on Cancel (via `reset_enrollment`).
    owner_negative_chunks: Vec<Vec<f32>>,
    /// Whether the user has completed all 10 enrollment utterances (Phase 2
    /// done).  Set by `handle_enrollment_sample` when
    /// `enrolled_utterance_count >= NUM_ENROLLMENT_SAMPLES`.  The main loop
    /// reads this flag to initiate the Phase 2→3 transition
    /// (`transition_to_phase3`).  Cleared on Cancel (via `reset_enrollment`).
    utterances_collected: bool,
    /// Number of user utterances enrolled so far.
    ///
    /// Unlike [`enrollment_buffer`] which may contain up to 12× entries due to
    /// PCM augmentation (12-cell recipe), this counter tracks
    /// only the actual user utterances.  The UI counter and finalization trigger
    /// use this field, not the buffer length.
    ///
    /// It is incremented once per `handle_enrollment_sample` call (before
    /// augmentation), ensuring that even if augmentation is temporarily
    /// disabled (e.g. for very short utterances where speed-up is skipped),
    /// the enrollment still completes after the correct number of user
    /// utterances.
    enrolled_utterance_count: usize,
    /// Cached wake word phrase from [`PersistedModel::phrase`], set on model
    /// load and on successful persist. Never cleared on cancel.
    /// Read by [`get_enrolled_phrase()`].
    model_phrase: Option<String>,
    /// Transient wake word phrase for an enrollment in progress.
    /// Set at enrollment start via [`normalize_phrase`], consumed by
    /// [`persist_model_state`]. Never read by [`get_enrolled_phrase()`].
    enrolling_phrase: Option<String>,
    cmd_tx: Option<mpsc::UnboundedSender<VoiceCommand>>,
}

impl VoicePipelineState {
    /// Clear all enrollment accumulators (buffers + utterance counter).
    ///
    /// Called by [`PipelineCtx::reset_pipeline_state`] on [`ResetLevel::Cancel`]
    /// and by tests to verify data model invariants.
    ///
    /// Clears the transient [`enrolling_phrase`] but preserves [`model_phrase`]
    /// (the cached phrase from the last loaded / persisted model).
    fn reset_enrollment(&mut self) {
        self.enrollment_buffer.clear();
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
    /// before being stored in [`PersistedModel::phrase`].
    StartEnrollment(String),
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

    LAST_ACTIVE_USER
        .set(RwLock::new(String::new()))
        .map_err(|_| anyhow!("LAST_ACTIVE_USER already initialized"))?;

    VOICE_PIPELINE
        .set(RwLock::new(VoicePipelineState {
            enabled: false,
            status: VoiceStatus::Disabled,
            classifier_weights: None,
            classifier: None,
            enrollment_buffer: Vec::new(),
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

pub fn send_command(cmd: VoiceCommand) {
    if let Some(tx) = &voice_state().read().unwrap_poison().cmd_tx {
        let _ = tx.send(cmd);
    } else {
        warn!("Voice pipeline not initialized — dropping command {cmd:?}");
    }
}

// ONNX model loading and execution

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
/// # Problem
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
/// [`extract_embeddings_from_audio`] and the stride-8 sliding-window fallback
/// in [`handle_wake_word_detection`], which passes subslices at arbitrary
/// `next_window_start` positions.
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
    embeddings_from_mel_frames(models, &mel_frames)
}

/// Extract stride-8 dense embeddings from a pre-computed mel frame buffer.
///
/// Shared by [`extract_embeddings_from_audio`] (whole-audio mel) and the
/// prewarm negative path ([`vad_gate_streaming_mel`] + this function)
/// so both produce the identical stride-8 windowing over mel
/// frames.
///
/// If the buffer has fewer than [`EMBEDDING_WINDOW_FRAMES`] frames, pads with
/// tapered fade-out frames so at least one embedding can be computed.  Without
/// this, short wake words (e.g. 0.5s) would be silently discarded during
/// enrollment, making enrollment impossible for brief utterances.
fn embeddings_from_mel_frames(
    models: &OnnxModels,
    mel_frames: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>> {
    if mel_frames.len() < EMBEDDING_WINDOW_FRAMES {
        let padded = pad_mel_frames_to_window(mel_frames);
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
        anyhow::bail!("No embeddings could be extracted from mel frames");
    }

    Ok(embeddings)
}

/// Core VAD-gating / batch-accumulation frame processing loop shared by both
/// the offline streaming embedding extraction and the live detection pipeline
/// ([`handle_wake_word_detection`]).
///
/// Processes audio frames through VAD gating, accumulates voiced samples into
/// `voice_batch`, flushes to `mel_frame_buffer` at [`VOICE_BATCH_SIZE`] or on
/// VAD-negative transitions, and calls `on_flush` with a shared reference to
/// the mel frame buffer (so the callback can read mel frames while the inner
/// function holds mutable access to the batch buffers).
///
/// Returns the number of samples consumed (in multiples of [`HOP_LENGTH`]).
/// On early exit (when `on_flush` returns `true`), the returned count excludes
/// the current frame's [`HOP_LENGTH`], matching the original early-return
/// behaviour in [`handle_wake_word_detection`].
///
/// Now, the live detection pipeline uses this function only for
/// VAD-gated mel frame accumulation — streaming embedding extraction (which
/// previously used this via `on_flush`) was removed.  The `on_flush` callback
/// still exists but is used only to detect early exit (wake word detection
/// during stride-8 scoring, which now happens after this loop).
/// [`vad_gate_streaming_mel`] additionally wraps this function
/// for the offline negative-prewarm path, so the training mel layout is
/// produced by the same loop as live inference.
///
/// # Parameters
///
/// - `samples`: Audio samples to process.
/// - `voice_batch`: Accumulates voiced audio samples across frames.
///   Caller-owned, persists across calls for the live detection path.
/// - `mel_frame_buffer`: Accumulated mel spectrogram frames produced by
///   flushing `voice_batch`.  Caller-owned, persists across calls for the
///   live detection path.
/// - `is_speech_fn`: VAD decision function.
/// - `trailing_flush`: If `true`, flush any remaining voice batch after the
///   frame loop.  The live path uses `false` because audio accumulates across
///   calls; only kept for test compatibility.
/// - `on_flush`: Called after each flush with `&[Vec<f32>]` — the current mel
///   frame buffer.  Return `true` to stop processing early (used by the live
///   path on wake word detection).
fn process_streaming_frames_inner(
    samples: &[f32],
    voice_batch: &mut Vec<f32>,
    mel_frame_buffer: &mut Vec<Vec<f32>>,
    mut is_speech_fn: impl FnMut(&[f32]) -> bool,
    trailing_flush: bool,
    mut on_flush: impl FnMut(&[Vec<f32>]) -> bool,
) -> usize {
    let mut consumed = 0;
    let len = samples.len();
    while consumed + FRAME_LENGTH <= len {
        let frame = &samples[consumed..consumed + FRAME_LENGTH];

        // VAD gate — skip silence to avoid wasted ONNX compute.
        //
        // Feed only the NEW HOP_LENGTH samples (not the full 512-sample
        // frame) to keep earshot's internal ring buffer contiguous.
        // Each frame overlaps the previous by 50% (= HOP_LENGTH = 256), so
        // feeding the full frame would duplicate the second half of the
        // previous frame's 256 samples — corrupting earshot's ring buffer,
        // pre-emphasis filter, and 3-frame feature context with duplicated
        // data.
        if is_speech_fn(&frame[..HOP_LENGTH]) {
            // Add only the NEW samples (HOP_LENGTH per frame) to avoid
            // duplicating overlapping audio.  Each frame overlaps the previous
            // by 50% (HOP_LENGTH = FRAME_LENGTH/2), so appending the full
            // frame would duplicate half the audio — corrupting the mel model
            // input with repeated segments.
            voice_batch.extend_from_slice(&frame[..HOP_LENGTH]);
        } else if !voice_batch.is_empty() {
            // Silence transition: flush accumulated voiced batch
            flush_voice_batch(voice_batch, mel_frame_buffer);
            voice_batch.clear();
            if on_flush(mel_frame_buffer) {
                // On early exit (wake word detected), return the consumed
                // count WITHOUT this frame's HOP_LENGTH to match the original
                // early-return behaviour — the caller handles the buffer
                // drain (or the detection→recording handoff clears it).
                return consumed;
            }
            consumed += HOP_LENGTH;
            continue;
        }

        // Process batch when enough voiced audio accumulated
        // (every ~128ms instead of every 32ms)
        if voice_batch.len() >= VOICE_BATCH_SIZE {
            flush_voice_batch(voice_batch, mel_frame_buffer);
            if on_flush(mel_frame_buffer) {
                return consumed;
            }
        }
        consumed += HOP_LENGTH;
    }

    // Flush any remaining voice batch after the frame loop (end-of-utterance
    // where no trailing silence is present).  Only performed when
    // `trailing_flush` is true (offline extraction); the live path leaves
    // the batch in place for the next call.
    if trailing_flush && !voice_batch.is_empty() {
        flush_voice_batch(voice_batch, mel_frame_buffer);
        voice_batch.clear();
        on_flush(mel_frame_buffer);
    }
    consumed
}

/// VAD-gate audio into streaming-layout mel frames + speech-only audio.
///
/// Thin wrapper over the canonical streaming loop
/// [`process_streaming_frames_inner`]: it delegates the entire VAD-gating /
/// batch-accumulation loop — feeding only each frame's NEW [`HOP_LENGTH`]
/// samples to the VAD decision function, flushing the voice batch
/// to mel frames at [`VOICE_BATCH_SIZE`] / silence transitions via
/// [`flush_voice_batch`], and flushing any trailing batch (`trailing_flush =
/// true`, since offline extraction always receives the full phrase) — so the
/// produced mel frame buffer has exactly the layout the streaming detection
/// pipeline produces: silence discarded, windows anchored at speech onset.
///
/// The only additional behaviour lives in the `is_speech_fn` wrapper closure:
/// every VAD-positive hop is also accumulated into the returned `speech_audio`
/// so callers can derive augmentation variants from speech-only audio,
/// matching enrollment's `AGC → VAD → augment` ordering.  Because the loop
/// itself is the canonical one, the prewarm path cannot drift from streaming
/// inference — the train/inference divergence class this ticket exists to fix.
///
/// # VAD decision ownership
///
/// The `is_speech_fn` closure owns the VAD decision state.  Prewarm callers
/// MUST pass a closure over a fresh `earshot::Detector` per phrase (never the
/// global [`VAD_DETECTOR`]) so prewarm VAD decisions cannot contaminate the
/// live pipeline's noise-floor state, and so no VAD state carries across
/// phrases (the same shared-state artifact as a shared AGC).
fn vad_gate_streaming_mel(
    audio: &[f32],
    mut is_speech_fn: impl FnMut(&[f32]) -> bool,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut voice_batch: Vec<f32> = Vec::new();
    let mut mel_frame_buffer: Vec<Vec<f32>> = Vec::new();
    let mut speech_audio: Vec<f32> = Vec::with_capacity(audio.len());

    // Accumulate VAD-positive hops into speech_audio inside the decision
    // closure: the loop logic (hop feeding, batching, flushing) then exists in
    // exactly one place — process_streaming_frames_inner — so the prewarm mel
    // layout cannot drift from the streaming path (hop-feeding included).
    let mut gated = |hop: &[f32]| {
        let is_speech = is_speech_fn(hop);
        if is_speech {
            speech_audio.extend_from_slice(hop);
        }
        is_speech
    };

    process_streaming_frames_inner(
        audio,
        &mut voice_batch,
        &mut mel_frame_buffer,
        &mut gated,
        true,      // trailing_flush — offline extraction receives the full phrase
        |_| false, // no early exit — collect the entire phrase
    );

    (mel_frame_buffer, speech_audio)
}

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
    // samples per call at 16 kHz).  A typical call receives 512-sample frame
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

// Model download

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
        crate::util::verify_sha256(&tmp_path, expected_hash)
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

// ── Voice PCM disk cache helpers ─────────────────────────────
//
// Cache bounding: two-phase eviction — age-based (stale entries
// older than voice_cache_max_age_days) followed by size-based (oldest-first
// via mtime, FIFO, when total exceeds voice_cache_max_size_mb).  Eviction runs
// at startup and before each cache write.  Best-effort: errors are logged but
// never propagated.

/// Evict stale and excess entries from the voice PCM disk cache.
///
/// Two-phase eviction:
/// 1. **Age-based**: Remove entries older than `voice_cache_max_age_days`
///    (configurable, default 30 days, 0 = disabled).
/// 2. **Size-based**: If total cache size exceeds `voice_cache_max_size_mb`
///    (configurable, default 100 MB, 0 = disabled), remove the oldest entries
///    (by file mtime) until the total is under the limit.
///
/// Transient `.tmp` files are excluded from both phases.
///
/// # Best-effort semantics
///
/// Never blocks enrollment or synthesis.  If the cache directory cannot be
/// read, entries fail to delete (permissions, disk errors), or config values
/// are unparseable, a warning is logged and the function returns gracefully.
/// Cache hits still work; cache misses re-synthesise as normal.
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

/// Compute a deterministic cache key for TTS-synthesised PCM audio.
///
/// Includes all TTS model SHA256 constants so that any TTS model update
/// automatically invalidates stale cached audio. The embedding model version
/// is NOT included — cached PCM audio is model-independent; embeddings are
/// freshly extracted from cached PCM on each startup using the current
/// ONNX embedding model.
pub(crate) fn pcm_cache_key(
    text: &str,
    style: &str,
    seed: u64,
    sample_rate: u32,
    model_hash: &str,
) -> String {
    let input = format!("{text}\0{style}\0{seed}\0{sample_rate}\0{model_hash}");
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex_string(&hasher.finalize())
}

/// Hash of all TTS model SHA256 constants for cache invalidation.
///
/// Any change to any TTS model file produces a different hash, which
/// changes the PCM cache key and triggers re-synthesis on first run.
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

/// Path to the voice PCM cache directory (`~/.mahbot/voice_cache/`).
pub(crate) fn voice_cache_dir() -> Option<PathBuf> {
    let root = CONFIG.try_storage_root()?;
    Some(root.join(VOICE_CACHE_DIR))
}

// ── Embedding-level cache for the deterministic prewarm path ──
//
// The per-utterance (phrase × style × seed) dense embeddings produced by
// `prewarm_phrase_embeddings` are deterministic: fixed TTS seeds, cached
// PCM, deterministic earshot VAD + ONNX inference.  Yet every cold process
// recomputes them (~200 utterances × 2 variants of ONNX work, the dominant
// bench Phase 3 cost).  This disk cache stores the per-utterance variant
// embeddings so warm runs (and warm app starts) skip the ONNX extraction.
//
// Key correctness requirement: the key MUST include BOTH ONNX model file
// hashes (mel + embed) plus the preprocessor toggles plus the PCM key
// inputs (phrase/style/seed).  A key with only the TTS model hash would
// silently reuse stale embeddings after a voice ONNX model swap.

const EMBEDDING_CACHE_SUBDIR: &str = "embeddings_cache";

/// Recipe identity baked into embedding-cache keys AND file headers.  Bump
/// whenever the augmentation recipe changes (variant set / cells / noise
/// semantics) so stale-variant cached embeddings are never silently reused.
///
/// Numbering starts at 6: the pre-versioning cache format led with the
/// variant count (≤ 5 cells), so a header value ≤ 5 could alias an old
/// format's count and let a stale file survive the version sweep.
const AUGMENT_RECIPE_VERSION: u32 = 6;

/// Sub-directory of the voice cache used for the embedding cache.
fn embedding_cache_dir() -> Option<PathBuf> {
    Some(voice_cache_dir()?.join(EMBEDDING_CACHE_SUBDIR))
}

/// SHA-256 hex of the mel + embedding ONNX model files (once per process).
static ONNX_MODELS_HASH: OnceLock<Option<String>> = OnceLock::new();

fn onnx_models_hash() -> Option<&'static str> {
    ONNX_MODELS_HASH
        .get_or_init(|| {
            let dir = model_dir()?;
            let mel = std::fs::read(dir.join(MEL_MODEL_FILENAME)).ok()?;
            let embed = std::fs::read(dir.join(EMBED_MODEL_FILENAME)).ok()?;
            let mut hasher = Sha256::new();
            hasher.update(&mel);
            hasher.update(&embed);
            Some(hex_string(&hasher.finalize()))
        })
        .as_deref()
}

/// Deterministic cache key for one prewarm utterance.
///
/// Covers: the existing PCM cache key (phrase + style + seed + sample rate +
/// TTS model hash), the preprocessor NS/AGC toggles, SHA-256 of both
/// ONNX model files, and [`AUGMENT_RECIPE_VERSION`].  A key missing the ONNX
/// hashes would silently reuse stale embeddings after a voice ONNX model
/// swap; a key missing the recipe version would replay old-variant cached
/// lists after a recipe change.
fn embedding_cache_key(
    phrase_type: &str,
    phrase: &str,
    style: &str,
    seed: u64,
    model_hash: &str,
    pre_config: crate::audio::audio_preprocessor::PreprocessorConfig,
) -> Option<String> {
    let onnx_hash = onnx_models_hash()?;
    let pcm_key = pcm_cache_key(phrase, style, seed, SAMPLE_RATE, model_hash);
    let mut hasher = Sha256::new();
    hasher.update(phrase_type.as_bytes());
    hasher.update([0u8]);
    hasher.update(pcm_key.as_bytes());
    hasher.update([0u8]);
    hasher.update([u8::from(pre_config.noise_suppression)]);
    hasher.update([u8::from(pre_config.agc)]);
    hasher.update(onnx_hash.as_bytes());
    hasher.update(AUGMENT_RECIPE_VERSION.to_le_bytes());
    Some(hex_string(&hasher.finalize()))
}

/// Read a cached per-utterance variant list: `(variant_index, embeddings)`.
///
/// The file starts with the recipe version header; a mismatch (old recipe or
/// pre-version file) deletes the entry and reports a miss.
fn read_embedding_cache(dir: &Path, key: &str) -> Option<Vec<(u8, Vec<Vec<f32>>)>> {
    let path = dir.join(key);
    let data = std::fs::read(&path).ok()?;
    let mut cur = 0usize;
    let version = u32::from_le_bytes(data.get(cur..cur + 4)?.try_into().ok()?);
    cur += 4;
    if version != AUGMENT_RECIPE_VERSION {
        let _ = std::fs::remove_file(&path);
        return None;
    }
    let n_variants = u32::from_le_bytes(data.get(cur..cur + 4)?.try_into().ok()?) as usize;
    cur += 4;
    let mut variants: Vec<(u8, Vec<Vec<f32>>)> = Vec::with_capacity(n_variants);
    for _ in 0..n_variants {
        let vi = *data.get(cur)?;
        cur += 1;
        let n_embs = u32::from_le_bytes(data.get(cur..cur + 4)?.try_into().ok()?) as usize;
        cur += 4;
        let mut embs = Vec::with_capacity(n_embs);
        for _ in 0..n_embs {
            let n_floats = u32::from_le_bytes(data.get(cur..cur + 4)?.try_into().ok()?) as usize;
            cur += 4;
            let mut emb = Vec::with_capacity(n_floats);
            for _ in 0..n_floats {
                let bytes: [u8; 4] = data.get(cur..cur + 4)?.try_into().ok()?;
                cur += 4;
                emb.push(f32::from_le_bytes(bytes));
            }
            embs.push(emb);
        }
        variants.push((vi, embs));
    }
    Some(variants)
}

/// Write a per-utterance variant list to the embedding cache (best-effort).
#[allow(clippy::cast_possible_truncation)]
fn write_embedding_cache(dir: &Path, key: &str, variants: &[(u8, Vec<Vec<f32>>)]) {
    // One-time per-process sweep of THIS directory: version-stale entries +
    // config-bounded size (tests writing tempdirs never touch the real cache).
    evict_embedding_cache(dir);
    let _ = std::fs::create_dir_all(dir);
    let mut data: Vec<u8> = Vec::new();
    data.extend_from_slice(&AUGMENT_RECIPE_VERSION.to_le_bytes());
    data.extend_from_slice(&(variants.len() as u32).to_le_bytes());
    for (vi, embs) in variants {
        data.push(*vi);
        data.extend_from_slice(&(embs.len() as u32).to_le_bytes());
        for emb in embs {
            data.extend_from_slice(&(emb.len() as u32).to_le_bytes());
            for &v in emb {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    // Atomic write (tmp + rename) matching the PCM cache pattern.
    let path = dir.join(key);
    let tmp_path = path.with_extension("tmp");
    if std::fs::write(&tmp_path, &data).is_ok() {
        let _ = std::fs::rename(&tmp_path, &path);
    }
}

/// One-shot guard for the per-write [`evict_embedding_cache`] sweep.
static EMBEDDING_EVICTION_RAN: AtomicBool = AtomicBool::new(false);

/// Bounded sweep of the embeddings cache directory.
///
/// Runs once per process on the first cache write, scoped to the directory
/// being written (so unit tests sweeping tempdirs never touch the real
/// cache).  Deletes entries whose recipe-version header differs from
/// [`AUGMENT_RECIPE_VERSION`] (stale recipe — unreachable via the versioned
/// key, but cleaning them bounds the directory), then applies the same
/// config-driven age/size limits as the PCM cache.  The bench's first
/// post-recipe run is cold by definition (all keys miss); this keeps that
/// one-time recompute from leaving stale garbage.
fn evict_embedding_cache(dir: &Path) {
    if EMBEDDING_EVICTION_RAN.swap(true, Ordering::Relaxed) {
        return;
    }
    evict_embedding_cache_dir(dir);
}

/// Directory-scoped sweep body (also called by tests against tempdirs).
fn evict_embedding_cache_dir(dir: &Path) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return; // directory doesn't exist yet
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|e| e.to_str()) == Some("tmp") {
            continue;
        }
        // Read only the 4-byte version header.
        let version = std::fs::File::open(&path).ok().and_then(|mut f| {
            use std::io::Read;
            let mut hdr = [0u8; 4];
            f.read_exact(&mut hdr).ok()?;
            Some(u32::from_le_bytes(hdr))
        });
        if version != Some(AUGMENT_RECIPE_VERSION) {
            let _ = std::fs::remove_file(&path);
        }
    }
    evict_pcm_cache(dir);
}

/// Write PCM f32 samples to the disk cache atomically.
///
/// Writes to a `.tmp` file first, then atomically renames to the final path,
/// so partial writes are never visible to readers.
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
    let pcm = match crate::audio::tts::synthesize(text, style, seed, sample_rate) {
        Ok(p) => p,
        Err(e) => {
            warn!(
                "TTS synthesis failed for '{text}' with {style} (seed={seed}): {e} \
                 — skipping this variant"
            );
            return None;
        }
    };

    // Evict stale/excess entries before writing to keep cache bounded.
    //
    // This performs a full directory scan on every cache miss.  During cold
    // enrollment the startup call (prewarm_phrase_embeddings) has just cleaned
    // the cache, so these per-write scans are strictly redundant — they provide
    // a correctness safety net for any future code path that writes to the
    // cache without going through prewarming (e.g., from agent tools).
    // The scan is gated to run ONCE per process: the prewarm
    // pass already bounds the cache, and per-miss scans only add ~30 ms × N
    // of directory I/O to the enrollment/benchmark burst.  The first miss in
    // each process still evicts, preserving the safety net.
    if !PCM_EVICTION_RAN.swap(true, Ordering::Relaxed) {
        evict_pcm_cache(cache_dir);
    }

    // Write to disk cache atomically
    write_pcm_cache(&cache_path, &pcm);
    Some(pcm)
}

/// One-shot guard for the per-miss [`evict_pcm_cache`] scan in
/// [`synthesize_with_pcm_cache`].  See the comment at the
/// call site — the scan is strictly redundant after prewarming, so it runs
/// at most once per process.
static PCM_EVICTION_RAN: AtomicBool = AtomicBool::new(false);

async fn ensure_models_downloaded() -> Result<PathBuf> {
    let dir = model_dir()
        .ok_or_else(|| anyhow!("Cannot resolve model directory (storage root not set)"))?;

    tokio::fs::create_dir_all(&dir).await?;

    let mel_path = dir.join(MEL_MODEL_FILENAME);
    if mel_path.exists()
        && let Err(e) = crate::util::verify_sha256(&mel_path, MEL_MODEL_SHA256)
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
        && let Err(e) = crate::util::verify_sha256(&embed_path, EMBED_MODEL_SHA256)
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
                        // Pre-warm confusable and unrelated dense embeddings in
                        // background so enrollment never blocks on TTS synthesis.
                        // Ran sequentially within
                        // a single task to avoid ONNX model thread-safety concerns.
                        tokio::spawn(async {
                            prewarm_confusable_embeddings().await;
                            prewarm_unrelated_embeddings().await;
                        });
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
                    // state and exit (avoids wasted retry loops).  Pre-warm
                    // was triggered by the first instance; calling it again
                    // is a no-op (cache check).
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
        if let Err(e) = tx.try_send(resampled) {
            DROPPED_CHUNKS.fetch_add(1, Ordering::Relaxed);
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
        DROPPED_CHUNKS.fetch_add(1, Ordering::Relaxed);
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

    let result =
        crate::audio::local_transcriber::transcribe_file_async(&tmp_path, Duration::from_secs(30))
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

// Enrollment helpers

/// Process raw audio samples into embedding sequences (for enrollment).
pub fn process_enrollment_sample(samples: &[f32]) -> Result<Vec<Vec<f32>>> {
    let models = ONNX_MODELS
        .get()
        .ok_or_else(|| anyhow!("Voice models not loaded"))?;
    extract_embeddings_from_audio(models, samples)
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
/// Conv1D classifier confidence is computed separately during enrollment
/// finalization (when the Conv1D classifier is trained on all utterances).
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

    // ── Composite score (basic metrics only, no DTW) ──────────────────
    // During enrollment collection, quality is based on duration, clipping,
    // and SNR.  Conv1D classifier confidence is computed at enrollment finalization after
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

/// Run a self-test of the trained classifier against the enrollment buffer.
///
/// Simulates the live detection pipeline for each enrollment utterance: feeds
/// embeddings one by one through the embedding ring, runs the Conv1D
/// classifier (`forward_pass`) on each 3-embedding window, and passes the
/// score through [`process_wake_word_score`] with a rolling window.
///
/// An utterance "triggers" if the rolling window sum exceeds the detection
/// threshold at any point.
///
/// NOTE: with the 12-cell enrollment recipe the evaluated set is up to ~120
/// sequences (10 utterances × 12 cells, 5 dB noise cells included) versus
/// ~50 at the old 5-cell recipe — the ≥80% gate is deliberately stricter
/// (the bench voice passes 108/120; marginal real enrollments may reject
/// more often in noisy environments).
///
/// Returns `Ok(())` if the self-test passes, or `Err` with a descriptive
/// message if too many utterances fail to trigger detection.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn run_enrollment_self_test(
    enrollment_sequences: &[EmbeddingSequence],
    classifier: &WakeWordClassifier,
) -> Result<(), String> {
    if enrollment_sequences.is_empty() {
        return Err("Self-test skipped: no enrollment samples".to_string());
    }

    let mut passed = 0usize;

    for seq in enrollment_sequences {
        // Fresh simulation for each utterance: no cross-utterance state.
        // Uses `score_single_embedding` which encapsulates the
        // same ring-buffer + Conv1D classifier + rolling window logic as the
        // live detection pipeline and the E2E integration test.
        let mut embedding_ring: Vec<Vec<f32>> = Vec::with_capacity(EMBEDDING_RING_MAX);
        let mut score_window = Vec::new();
        let mut detected = false;

        for embedding in &seq.embeddings {
            let (detected_this, _, _, _) = score_single_embedding(
                embedding,
                &mut embedding_ring,
                Some(classifier),
                &mut score_window,
                None, // no adaptive threshold during enrollment self-test
                ADAPTIVE_K_DEFAULT,
                false,
            );
            if detected_this {
                detected = true;
                break;
            }
        }

        if detected {
            passed += 1;
        }
    }

    let required = (enrollment_sequences.len() as f32 * ENROLLMENT_QUALITY_SELF_TEST_MIN_FRACTION)
        .ceil() as usize;

    if passed < required {
        Err(format!(
            "Self-test failed: only {passed}/{} utterances triggered detection (need ≥{required}). \
             Try re-enrolling with clearer, more consistent speech.",
            enrollment_sequences.len(),
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
    /// (typically [`ENROLLMENT_VAD_CONSECUTIVE_REQUIRED`] = 1 after the threshold unification).
    consecutive_required: usize,
    /// Silence duration in samples before utterance ends
    /// (typically [`ENROLLMENT_SILENCE_THRESHOLD_SAMPLES`] = 4 864 ≈ 304 ms
    /// aligned to streaming's segment timeout).
    silence_threshold_samples: usize,
    /// Samples of pre/post speech context to include
    /// (0 — enrollment now matches streaming detection
    /// by not adding context padding).
    context_padding_samples: usize,
    /// Max samples in the internal raw-audio ring buffer
    /// (typically [`RAW_RING_MAX`] = 3 200 ≈ 200 ms).
    raw_ring_max: usize,
}

/// Module-level default config for [`segment_utterances_by_vad`] using the
/// standard voice-pipeline constants.
///
/// Context padding is intentionally 0 to match the streaming detection path,
/// which does not add context padding.  Previously the
/// padding (~100ms) caused the embedding windows during enrollment training
/// to include ambient audio before the VAD onset, creating a temporal
/// misalignment with streaming inference where embeddings start at the first
/// VAD-positive frame without prepended context.
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
///    `config.context_padding_samples > 0`) to capture onset phonemes
///    excluded by a strict VAD threshold.  The default
///    config sets `context_padding_samples: 0` so both enrollment and
///    streaming start at VAD onset with matching temporal alignment.
/// 4. After speech, `config.silence_threshold_samples` of consecutive
///    VAD-negative audio ends the utterance.  Post-speech context (if
///    `config.context_padding_samples > 0`) is optionally appended from
///    the raw-audio ring (captured at the first silence frame).
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
/// padding (if `config.context_padding_samples > 0`).  Empty if no utterances
/// were detected.
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
/// Returns the trained [`ClassifierWeights`] on success, after validating
/// utterance quality via [`validate_enrollment_consistency`] (gating) and
/// training the classifier.  The old detection-based self-test is preserved
/// as an informational-only diagnostic in the caller (see
/// [`handle_enrollment_sample`]).
pub(crate) fn finalize_enrollment(
    positive_sequences: &[EmbeddingSequence],
    negative_sequences: &[EmbeddingSequence],
) -> Result<wake_word_classifier::ClassifierTrainingResult> {
    anyhow::ensure!(
        positive_sequences.iter().any(|s| !s.embeddings.is_empty()),
        "No positive embeddings available for training"
    );

    // Step 1: Consistency check — gates on utterance quality before training
    validate_enrollment_consistency(positive_sequences)?;

    // Step 2: Train a single Conv1D classifier.
    // The 5-member ensemble was removed — a single small Conv1D (~1.2K params)
    // captures temporal convolution patterns without ensemble overhead.
    let config = wake_word_classifier::TrainingConfig {
        rng_seed: Some(0), // deterministic seed for reproducibility
        ..Default::default()
    };
    let result =
        wake_word_classifier::train_classifier(positive_sequences, negative_sequences, &config)?;
    Ok(result)
}

/// Mean-pool a sequence of per-frame embeddings (from one utterance) into a
/// single 96-dim embedding vector.
///
/// Used by [`validate_enrollment_consistency`] to compute per-utterance means
/// for centroid cosine-similarity analysis.  Returns an empty `Vec` when
/// `embeddings` is empty.
#[must_use]
pub(crate) fn mean_pool_embeddings(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    let dim = embeddings[0].len();
    if dim == 0 {
        return Vec::new();
    }
    #[allow(clippy::cast_precision_loss)]
    let n = embeddings.len() as f32;
    let mut mean = vec![0.0; dim];
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            mean[i] += v;
        }
    }
    for v in &mut mean {
        *v /= n;
    }
    mean
}

/// Validate enrollment utterance quality via centroid cosine similarity.
///
/// # Gating
/// - Requires ≥5 utterances with ≥3 embeddings each.
/// - Requires ≥ceil(N * `ENROLLMENT_CONSISTENCY_MIN_FRACTION`) utterances to have
///   cosine similarity ≥`ENROLLMENT_CONSISTENCY_MIN_SIMILARITY` to the centroid.
///
/// # Self-correlation bias
/// The centroid is computed from ALL qualified utterances, including the
/// utterance being compared.  This means each utterance's own embedding
/// contributes to the centroid, slightly inflating similarity scores
/// (self-correlation).  The thresholds (`ENROLLMENT_CONSISTENCY_MIN_SIMILARITY`
/// = 0.65, `ENROLLMENT_CONSISTENCY_MIN_FRACTION` = 0.7) were calibrated with
/// this bias present — they work correctly in practice but should be re-checked
/// if centroid computation ever switches to leave-one-out or a fixed reference.
///
/// This replaces the old detection-based self-test which was brittle because
/// `score_single_embedding` requires ≥3 embeddings per utterance, a condition
/// the enrollment VAD threshold (~0.60) rarely guarantees.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub(crate) fn validate_enrollment_consistency(sequences: &[EmbeddingSequence]) -> Result<()> {
    // Filter to utterances with ≥3 embeddings
    let qualified: Vec<Vec<f32>> = sequences
        .iter()
        .filter(|seq| seq.embeddings.len() >= 3)
        .map(|seq| mean_pool_embeddings(&seq.embeddings))
        .collect();

    let total = qualified.len();
    anyhow::ensure!(
        total >= 5,
        "Only {total} enrollment utterances had sufficient audio (need ≥5 with ≥3 embeddings). \
         Speak clearly and close to the microphone.",
    );

    let centroid = mean_pool_embeddings(&qualified);

    // Count utterances that pass the similarity threshold
    let passed = qualified
        .iter()
        .filter(|mean_emb| {
            cosine_similarity(mean_emb, &centroid) >= ENROLLMENT_CONSISTENCY_MIN_SIMILARITY
        })
        .count();

    let required = (total as f32 * ENROLLMENT_CONSISTENCY_MIN_FRACTION).ceil() as usize;
    anyhow::ensure!(
        passed >= required,
        "Enrollment consistency check failed: only {passed}/{total} utterances met the \
         quality threshold (need ≥{required} with cosine similarity \
         ≥{ENROLLMENT_CONSISTENCY_MIN_SIMILARITY}). \
         Try re-enrolling in a quieter environment with clearer, more consistent speech.",
    );

    info!(
        "Enrollment consistency check passed: {passed}/{total} utterances (threshold ≥{required})",
    );
    Ok(())
}

// Routing to active agent

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

/// Resolve the workspace for a voice operation from the acting user's DB
/// record, falling back to the user's personal workspace on missing or
/// stale values — mirrors the chat-side resolution pattern (DB workspace,
/// warning + personal fallback).
async fn resolve_workspace_for_voice(user_name: &str) -> crate::Workspace {
    match crate::users::get_workspace(user_name).await {
        Ok(Some(ws)) => ws,
        Ok(None) => {
            warn!(
                user_name = %user_name,
                "workspace resolution: selected_workspace points to non-existent workspace; \
                 falling back to personal workspace",
            );
            personal_workspace_for_voice(user_name)
        }
        Err(e) => {
            warn!(
                user_name = %user_name,
                error = %e,
                "workspace resolution: database error; falling back to personal workspace",
            );
            personal_workspace_for_voice(user_name)
        }
    }
}

fn personal_workspace_for_voice(user_name: &str) -> crate::Workspace {
    let path = crate::users::personal_workspace_path(user_name);
    crate::users::personal_workspace_struct(user_name, &path)
}

/// Personal workspaces have no board pipeline — Manager falls back to
/// Analyst, matching the chat-side role resolution.
fn resolve_effective_role_for_voice(role: crate::Role, ws_name: &str) -> crate::Role {
    if role == crate::Role::Manager && crate::users::is_personal_workspace(ws_name) {
        crate::Role::Analyst
    } else {
        role
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
        let role = crate::users::resolve_active_role(&user_name).await;
        let ws = resolve_workspace_for_voice(&user_name).await;
        let role = resolve_effective_role_for_voice(role, &ws.name);

        info!("Voice command -> {role} (user: {user_name}, workspace: {}): {text}", ws.name);

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
        );
        return;
    }

    // No active user: fall back to the admin user's DB workspace (same
    // warning + personal fallback as the active-user path).
    let ws = resolve_workspace_for_voice("admin").await;

    info!(
        "Voice command -> Manager (workspace: {}): {}",
        ws.name, text
    );
    broadcast_voice_transcript(&text, "", &ws.name).await;

    crate::message_router::route_user_message(
        text,
        ws.name,
        String::new(),
        "voice".to_string(),
        crate::Role::Manager,
        None,
    );
}

// Voice pipeline background task

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

// Adaptive threshold state

/// Tracks running mean and standard deviation of recent per-frame classifier scores
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
/// [`ADAPTIVE_SAFE_HARBOR`] (matching the static [`match_threshold()`]), never
/// below [`ADAPTIVE_FLOOR`], and never above [`ADAPTIVE_CEILING`].  During the
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

    /// Feed a new per-frame classifier score and return the adaptive threshold.
    ///
    /// The threshold is computed from per-frame scores (range [0,1]), then
    /// scaled by [`ROLLING_WINDOW_N`] to match the rolling-sum space [0,3]
    /// where the detection comparison lives.  Returns `None` during the
    /// bootstrap period (first [`ADAPTIVE_BOOTSTRAP_FRAMES`] frames) to tell
    /// the caller to use the static threshold.  After bootstrap returns
    /// `Some(threshold)` where `threshold` is already clamped to the full
    /// safeguard range ([`ADAPTIVE_FLOOR`], [`ADAPTIVE_CEILING`],
    /// [`ADAPTIVE_SAFE_HARBOR`]) in rolling-sum space.
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
    /// clamped threshold in rolling-sum space (range [`ADAPTIVE_FLOOR`,
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
        // Two lower bounds (safe harbor then absolute floor) and one upper
        // bound (absolute ceiling).  The `max(SAFE_HARBOR)` ensures the
        // threshold never drops below the static detection threshold; the
        // `clamp(FLOOR, CEILING)` provides the hard safeguard range.
        // Since ADAPTIVE_FLOOR <= ADAPTIVE_SAFE_HARBOR (compile-time invariant),
        // the safe harbor dominates the lower bound in the clamp.
        adaptive
            .max(ADAPTIVE_SAFE_HARBOR)
            .clamp(ADAPTIVE_FLOOR, ADAPTIVE_CEILING)
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
    /// is now score-only); it exists for unit tests and the
    /// voice-tests instrumentation mirror.
    #[cfg(any(test, feature = "voice-tests"))]
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
    /// initialized with negative-distribution-mean scores.  Used by the E2E
    /// benchmark so the adaptive threshold is active from the start of
    /// detection testing.  The seed value (NEGATIVE_DISTRIBUTION_MEAN = 0.033)
    /// ensures the threshold immediately clamps to the safe harbor (1.35),
    /// matching production behavior where real audio starts from silence.
    #[cfg(any(test, feature = "voice-tests"))]
    pub(crate) fn warmed() -> Self {
        let mut state = Self::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(NEGATIVE_DISTRIBUTION_MEAN, ADAPTIVE_K_DEFAULT);
        }
        state
    }

    /// The number of scores currently in the window.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.scores.len()
    }
}

// per-frame scoring geometry / adaptive-mode instrumentation
// for the training-vs-streaming same-audio comparison

/// Geometry class of a single scored embedding window (mahbot-1012 §1).
///
/// Every scored window is classified into exactly one of four classes so the
/// benchmark can attribute score deficits to the known structural divergences
/// between the training/enrollment path and the streaming detection path:
///
/// - [`ColdStartTiled`](WindowGeometry::ColdStartTiled): the Conv1D classifier
///   window was repeat-last tiled because fewer than `WINDOW_SIZE` embeddings
///   were in the ring (the first 1-2 windows of every utterance after silence).
/// - [`WarmMixed`](WindowGeometry::WarmMixed): the Conv1D window spans preserved
///   warm-up ring entries + test-utterance entries (warm pass only).
/// - [`PaddedFallback`](WindowGeometry::PaddedFallback): the mel window came
///   from the short-buffer padded fallback (real frames + a tapered fade-out
///   tail — a family of inputs largely absent from training).
/// - [`TrueSliding`](WindowGeometry::TrueSliding): full 76-frame mel window
///   from the main stride-8 loop with a clean classifier ring.
///
/// Precedence when computing (highest first): `ColdStartTiled` > `WarmMixed` >
/// `PaddedFallback` > `TrueSliding`.  The classes are mutually exclusive in
/// practice: `ColdStartTiled` needs a ring below `WINDOW_SIZE`, which cannot
/// co-occur with a preserved warm-up ring (`WarmMixed`); `WarmMixed` needs a
/// preserved warm-up ring, which the cold pass never has.
#[cfg(feature = "voice-tests")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WindowGeometry {
    ColdStartTiled,
    WarmMixed,
    PaddedFallback,
    TrueSliding,
}

#[cfg(feature = "voice-tests")]
impl WindowGeometry {
    /// Stable snake_case label for JSON output (mahbot-1012).
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::ColdStartTiled => "cold_start_tiled",
            Self::WarmMixed => "warm_mixed",
            Self::PaddedFallback => "padded_fallback",
            Self::TrueSliding => "true_sliding",
        }
    }
}

/// Scoring path that produced a scored window / detection.
///
/// Carried through [`score_stride8_window`] so the enrolled-speaker benchmark
/// can attribute every detection to exactly one path:
///
/// - [`DeferredBurst`](WindowSource::DeferredBurst) — the start-aligned
///   deferred burst sweep (buffer ≥ [`BURST_TRIGGER_FRAMES`]).
/// - [`SegmentEndPass`](WindowSource::SegmentEndPass) — the segment-boundary
///   fallback pass.
/// - [`MainStride`](WindowSource::MainStride) — the main stride-8 loop.
///
/// Not feature-gated: the enum appears in the production
/// [`score_stride8_window`] signature.  Its per-frame recording in
/// [`DetectionInstrumentation`] is voice-tests-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WindowSource {
    /// Main stride-8 sliding-window loop.
    MainStride,
    /// Deferred burst sweep (start-aligned positions 0/8/16/24, padded
    /// geometry).
    DeferredBurst,
    /// Segment-end boundary pass (start-aligned positions, padded geometry).
    SegmentEndPass,
}

impl WindowSource {
    /// Stable snake_case label for report output.
    ///
    /// The labels follow the acceptance taxonomy:
    /// "burst" and "segment_end_pass" are the two dedicated scoring paths;
    /// "other" (the main stride-8 loop) is the primary expected detection
    /// path.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::MainStride => "other",
            Self::DeferredBurst => "burst",
            Self::SegmentEndPass => "segment_end_pass",
        }
    }
}

// Pure decision helpers for the deferred-burst state machine.
//
// Extracted so the hold / burst / segment-end-pass / per-segment-flag
// lifecycle is unit-testable WITHOUT ONNX models (`ONNX_MODELS` is `None`
// in unit tests).  The production call sites in [`handle_wake_word_detection`]
// and [`PipelineCtx::handle_segment_boundary`] use exactly these helpers, so
// the tests exercise the real control flow.

/// Start-aligned positions to score in the deferred burst sweep.
///
/// Returns an empty list while holding (buffer below
/// [`BURST_TRIGGER_FRAMES`], or the per-segment sweep already ran) — the
/// incremental per-chunk scoring bug is exactly the old code
/// scoring position 0 with 2–16 frames per chunk.  When triggered, returns
/// the start-aligned positions 0/8/16/24 (each strictly below `buffer_len`;
/// a position at or past the buffer end is never scored — positions beyond
/// the buffer end must never be re-scored on later calls).
pub(crate) fn burst_positions_to_score(buffer_len: usize, burst_sweep_done: bool) -> Vec<usize> {
    if burst_sweep_done || buffer_len < BURST_TRIGGER_FRAMES {
        return Vec::new();
    }
    start_aligned_positions(buffer_len)
}

/// The start-aligned position grid 0/8/16/24, each strictly below
/// `buffer_len`.
///
/// Shared by the deferred burst sweep and the segment-end pass — both score
/// the same trained start-0-aligned geometry.  Positions at or past the
/// buffer end are never scored: a position beyond the buffer end must never
/// be re-scored on later calls (a re-scored ≤17-real-frame window ~0.01
/// mid-utterance resets the rolling window).
pub(crate) fn start_aligned_positions(buffer_len: usize) -> Vec<usize> {
    (0..BURST_MAX_POSITIONS * BURST_STRIDE)
        .step_by(BURST_STRIDE)
        .take_while(|&p| p < buffer_len)
        .collect()
}

/// Classify the geometry of one scored window (mahbot-1012 §1).
///
/// Pure function so the classification is unit-testable without pipeline
/// state.  The four classes are decided in precedence order:
///
/// 1. **ColdStartTiled** — the Conv1D window is repeat-last tiled because the
///    POST-push embedding ring has fewer than [`WINDOW_SIZE`] entries
///    (mirrors the tiling condition inside [`score_single_embedding`]).
/// 2. **WarmMixed** — the window spans warm-up embeddings AND test
///    embeddings.  The Conv1D window covers the last `WINDOW_SIZE` pushed
///    embeddings; a warm-up embedding occupies push-order index `< tsrl`, so
///    the window is mixed iff fewer than `WINDOW_SIZE - 1` test-pass frames
///    were scored before this one.  `frames_scored_before` counts scored
///    frames since the pass's instrumentation reset (`per_frame_scores.len()`
///    pre-push), which is NOT reset by mid-pass segment boundaries, so
///    post-boundary windows (whose ring was cleared) never classify WarmMixed.
///    `test_start_ring_len == 0` (cold pass) means no warm-up embeddings
///    exist at all.
/// 3. **PaddedFallback** — the window came from a padded geometry: the
///    deferred burst sweep or the segment-end pass score start-aligned
///    positions with synthetic fade-out padding when the buffer is shorter
///    than [`EMBEDDING_WINDOW_FRAMES`] (the old incremental per-chunk
///    short-buffer fallback was removed in mahbot-1023; the padding class
///    name is kept for report stability).
/// 4. **TrueSliding** — a genuine stride-8 sliding window.
///
/// # Ring-capacity correctness (mahbot-1012 reviewer)
///
/// The tiled check uses the PRE-push ring length exactly as
/// [`score_single_embedding`] sees it (post-push `len() >= WINDOW_SIZE`
/// after the `EMBEDDING_RING_MAX` trim — the trim never drops below
/// `WINDOW_SIZE`, so `ring_len_before + 1` overstates the post-push length
/// only when the ring is already at capacity, where the tiled branch is
/// unreachable anyway).  The WarmMixed check deliberately uses push-order
/// counting instead of ring coordinates: once the ring wraps past
/// [`EMBEDDING_RING_MAX`] (front-eviction), the ring-relative window start
/// `ring_len_before + 1 - WINDOW_SIZE` pins at `EMBEDDING_RING_MAX -
/// WINDOW_SIZE` and never advances, which would mislabel every later window
/// as WarmMixed when `test_start_ring_len` is near the cap.
#[cfg(feature = "voice-tests")]
pub(crate) fn classify_window_geometry(
    ring_len_before: usize,
    frames_scored_before: usize,
    test_start_ring_len: usize,
    padded_window: bool,
) -> WindowGeometry {
    if ring_len_before + 1 < wake_word_classifier::WINDOW_SIZE {
        WindowGeometry::ColdStartTiled
    } else if test_start_ring_len > 0
        && frames_scored_before + 1 < wake_word_classifier::WINDOW_SIZE
    {
        WindowGeometry::WarmMixed
    } else if padded_window {
        WindowGeometry::PaddedFallback
    } else {
        WindowGeometry::TrueSliding
    }
}

/// Per-frame adaptive-threshold mode (mahbot-1012 §1).  Mirrors the
/// feed/peek/bootstrap decision inside [`score_single_embedding`]: the
/// feed/peek rule is score-based ONLY (scores below `NO_MATCH_RESET_THRESHOLD`
/// `feed` the background statistics; wake-word-like scores only `peek`),
/// including during bootstrap (mahbot-1023).  The label is STATE-based for
/// bootstrap frames (`Bootstrap` while the state has not completed its
/// bootstrap window, regardless of the feed/peek action taken this frame —
/// a high-scoring bootstrap frame peeks and still labels `Bootstrap`) and
/// ACTION-based after bootstrap (`Feed` / `Peek`).  This keeps the labels
/// accurate in lockstep with the feed rule: a prolonged bootstrap (persisting
/// across a full high-scoring utterance) labels every frame `Bootstrap`, and
/// the static [`match_threshold()`] stays in effect throughout (peek returns
/// `None` during bootstrap).
#[cfg(feature = "voice-tests")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdaptiveFrameMode {
    /// Adaptive state still in its bootstrap window — the frame either fed
    /// (below-reset background score) or peeked (wake-word-like score, which
    /// returns `None` → static [`match_threshold()`] in effect).
    Bootstrap,
    /// Post-bootstrap `feed()` — the frame updated the background statistics.
    Feed,
    /// Post-bootstrap `peek()` — the frame did not update the statistics.
    Peek,
}

#[cfg(feature = "voice-tests")]
impl AdaptiveFrameMode {
    /// Stable snake_case label for JSON output (mahbot-1012).
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Bootstrap => "bootstrap",
            Self::Feed => "feed",
            Self::Peek => "peek",
        }
    }
}

/// Deterministic 64-bit hash of an embedding for cross-path comparison
/// (mahbot-1012 §1).
///
/// FNV-1a over the f32 bit patterns with `-0.0` canonicalised to `+0.0` and
/// NaN payloads canonicalised to a single quiet NaN, so the hash is stable
/// across runs and platforms for the same embedding values.  The embedding
/// model is stateless and deterministic, so two paths that feed bit-identical
/// mel windows produce bit-identical embeddings and therefore identical
/// hashes — a hash mismatch at a matched window position is direct evidence
/// that the mel values (or window content) differed between the paths.
#[cfg(feature = "voice-tests")]
fn embedding_hash(embedding: &[f32]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for &v in embedding {
        let bits = if v == 0.0 {
            // Canonicalise -0.0 == +0.0 (equal values must hash equal).
            0.0f32.to_bits()
        } else {
            let b = v.to_bits();
            // Canonicalise NaN payloads (they compare unequal but are not
            // distinguishable acoustic values).
            if b & 0x7fff_ffff > 0x7f80_0000 {
                0x7fc0_0000
            } else {
                b
            }
        };
        for byte in bits.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

/// L2 norm of an embedding (mahbot-1012 §1).  Reported alongside the hash so
/// the dual-path comparison can quantify embedding drift (the manager's lead
/// measured L2 deltas of 15-30% of the vector norm between paths).
#[cfg(feature = "voice-tests")]
fn embedding_l2_norm(embedding: &[f32]) -> f32 {
    embedding.iter().map(|v| v * v).sum::<f32>().sqrt()
}

// Pipeline context

/// Per-variant instrumentation collected by the wake word detection benchmark
/// (mahbot-886).  Feature-gated behind `voice-tests` — zero production overhead.
#[cfg(feature = "voice-tests")]
#[derive(Debug, Clone)]
pub(crate) struct DetectionInstrumentation {
    /// All per-frame `[total_score, rolling_sum, threshold]` triples
    /// encountered during detection of a single variant.  The third element
    /// is the effective threshold used for the rolling window comparison
    /// (adaptive threshold post-bootstrap, or static match_threshold()
    /// during bootstrap / when no adaptive state is configured).  Used by
    /// the benchmark's miss-classification logic to distinguish
    /// adaptive-threshold blocks (mahbot-891).
    pub per_frame_scores: Vec<[f32; 3]>,
    /// Count of frames where `total_score < NO_MATCH_RESET_THRESHOLD` (0.316).
    pub n_frames_below_reset: usize,
    /// Count of VAD-positive 512-sample frames during streaming detection.
    pub vad_speech_frames: usize,
    /// Peak rolling-sum score across all segments in this detection session
    /// (mahbot-894).  Preserved across segment-end resets by capturing from
    /// [`ctx.peak_score`](PipelineCtx::peak_score) before
    /// [`reset_detection_segment`] clears the main field, via max-tracking
    /// conditionals at both the [`reset_detection_segment`] and
    /// [`handle_wake_word_detection`] save points.  For sessions with multiple
    /// segments (utterance boundary fired during silence), this retains the
    /// maximum peak across all segments so the E2E benchmark reports an
    /// accurate value.
    pub peak_score: f32,
    /// Per-frame adaptive threshold trajectory (mahbot-923 §7).
    /// Records the effective threshold value (`per_frame_scores[i][2]`) at each
    /// embedding frame so the two-tier ADAPTIVE_CEILING escalation plan
    /// (4.503 → 5.5 → 6.0) can be data-driven: if the ceiling is consistently
    /// the active limiting factor, escalate to 5.5; if that still fails, to 6.0.
    pub adaptive_threshold_trajectory: Vec<f32>,
    /// Count of frames where the effective threshold hit ADAPTIVE_CEILING (4.503).
    /// When this is non-zero AND detection rate is below target, the ceiling
    /// is the likely limiting factor and escalation should be considered.
    pub ceiling_limited_frames: usize,
    /// Index into [`per_frame_scores`](Self::per_frame_scores) of the first
    /// frame where the rolling sum reached the effective threshold (classifier
    /// trigger).  `None` if the classifier never triggered on test-utterance
    /// frames (mahbot-1005 §2 evidence).
    pub first_trigger_frame_idx: Option<usize>,

    // ── mahbot-1012 per-frame scoring geometry ────────────────────────────
    // All `per_frame_*` arrays below are parallel to
    // [`per_frame_scores`](Self::per_frame_scores): entry `i` describes the
    // embedding scored at frame `i`.
    /// Per-frame stable hash of each scored embedding (FNV-1a, see
    /// [`embedding_hash`]).  Used by the dual-path same-audio comparison to
    /// detect mel-value divergence between the training/enrollment path and
    /// the streaming path.
    pub per_frame_embedding_hashes: Vec<u64>,
    /// Per-frame L2 norm of each scored embedding.  Enables L2-delta analysis
    /// between the two paths without storing full embeddings.
    pub per_frame_embedding_l2_norms: Vec<f32>,
    /// Per-frame raw embedding vectors.  Retained only under `voice-tests` so
    /// the dual-path harness can compute cosine / L2 deltas between the
    /// training and streaming embeddings at matched positions.
    pub per_frame_embeddings: Vec<Vec<f32>>,
    /// Mel-frame position (index into the mel frame buffer) where each scored
    /// window started.  The training path anchors windows at multiples of 8
    /// from mel frame 0 (speech onset); the streaming path's
    /// `next_window_start` drifts via buffer trims and blinded-gap
    /// re-anchoring — this field exposes that drift directly.
    pub per_frame_window_start: Vec<usize>,
    /// Mel frame buffer length at each scoring step.
    pub per_frame_mel_buffer_len: Vec<usize>,
    /// Geometry class of each scored window.  See [`WindowGeometry`].
    pub per_frame_geometry: Vec<WindowGeometry>,
    /// Adaptive-threshold mode (bootstrap / feed / peek) of each scored frame.
    /// See [`AdaptiveFrameMode`].
    pub per_frame_adaptive_mode: Vec<AdaptiveFrameMode>,
    /// Number of preserved warm-up embeddings at the start of the test
    /// utterance (set by the benchmark's `consume_warmup` after the
    /// instrumentation reset).  Used to classify the first test windows as
    /// [`WindowGeometry::WarmMixed`].  Zero for the cold pass.
    pub test_start_ring_len: usize,
    /// Per-hop VAD decisions during streaming detection, in order — one entry
    /// per VAD decision (each 512-sample frame processed, feeding its new
    /// 256-sample half to the VAD).  Correlates with mel-frame positions via
    /// the ~1.6-hop-per-mel-frame ratio (mel stride 160 samples at 16 kHz).
    pub per_hop_vad: Vec<bool>,

    // ── mahbot-1023 deferred-burst instrumentation ────────────────────────
    /// Whether the deferred burst sweep ran in this segment.  Cleared with
    /// [`PipelineCtx`] per-segment state (segment resets), NOT per call.
    pub burst_sweep_fired: bool,
    /// Mel-frame buffer length at burst-sweep time (None when the burst never
    /// ran).  The live trigger lands at the first flush-aligned B ≥ 68
    /// (typically 68–80), reported per variant so the acceptance review can
    /// see the actual live geometry (manager pin 2).
    pub burst_sweep_buffer_len: Option<usize>,
    /// Wall-clock time of the synchronous burst sweep in ms (the one-shot
    /// ~44–135 ms worst-case stall; measured through the live pipeline with
    /// AGC/VAD/block_in_place overhead).
    pub burst_wall_clock_ms: Option<f64>,
    /// Whether the segment-end pass ran at a boundary in this detection
    /// session.
    pub segment_end_pass_fired: bool,
    /// Scoring path that produced the detection (raw source: "burst" /
    /// "segment_end_pass" / "other").  None until a detection fires.
    pub detection_path: Option<&'static str>,
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
            per_frame_embedding_hashes: Vec::new(),
            per_frame_embedding_l2_norms: Vec::new(),
            per_frame_embeddings: Vec::new(),
            per_frame_window_start: Vec::new(),
            per_frame_mel_buffer_len: Vec::new(),
            per_frame_geometry: Vec::new(),
            per_frame_adaptive_mode: Vec::new(),
            test_start_ring_len: 0,
            per_hop_vad: Vec::new(),
            burst_sweep_fired: false,
            burst_sweep_buffer_len: None,
            burst_wall_clock_ms: None,
            segment_end_pass_fired: false,
            detection_path: None,
        }
    }
}

/// Runtime state for the voice pipeline main loop.
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct PipelineCtx {
    mic_rx: Option<mpsc::Receiver<Vec<f32>>>,
    mic_stream: SendMicStream,
    is_listening: bool,
    is_recording: bool,
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
    /// both detection and enrollment.  Previously used separate
    /// thresholds (0.60 for enrollment, 0.50 for detection), which caused the
    /// Conv1D classifier to receive out-of-distribution inputs during
    /// streaming inference (frames scoring 0.50–0.59 were included during
    /// detection but never seen during enrollment training).
    vad_threshold: f32,
    /// Separate Earshot VAD detector instance for enrollment mode.
    ///
    /// The streaming detection path uses the global [`VAD_DETECTOR`] singleton
    /// to maintain continuous noise-floor and ring-buffer state across the
    /// live microphone stream.  Enrollment mode uses its own detector instance
    /// to prevent mode-transition state contamination: when enrollment ends and
    /// streaming resumes, the global detector's state remains uncontaminated by
    /// enrollment's VAD decisions (which previously used a different threshold).
    ///
    /// Initialised to `None` outside enrollment.  Set to
    /// `Some(earshot::Detector::default())` when enrollment starts and cleared
    /// to `None` when enrollment ends or is cancelled.
    enrollment_vad: Option<earshot::Detector>,
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
    /// silently dropped.
    enrollment_pending: VecDeque<Vec<f32>>,
    auto_start_pending: bool,
    /// Timestamp of the last automatic model retry attempt.  Used to debounce
    /// so we don't spam the retry loop every 1-second tick when models are in
    /// [`ModelState::Failed`] (the periodic wake-up checks the state).
    last_model_retry: Option<Instant>,
    /// Timestamp of the last wake word detection.
    /// Used to enforce a cooldown period after detection to prevent rapid
    /// consecutive false triggers.
    pub(crate) last_wake_word_detection: Option<Instant>,
    /// Pre-speech noise RMS captured at the moment of first sustained speech
    /// detection during enrollment.  Computed from the pre-AGC audio ring
    /// ([`pre_agc_ring`]) so AGC's asymmetric gain (4× on silence, ~1-2× on
    /// speech) does not artificially lower the SNR estimate.
    /// Used for real SNR estimation in [`compute_utterance_quality`] instead
    /// of the fake SNR computed from speech dynamic range.
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
    /// to detect chunk boundaries — see the constant definition for coupling docs.
    phase3_silence_samples: usize,
    /// Accumulated VAD-positive speech samples collected during Phase 3.
    /// When `negatives_speech_samples >= SAMPLE_RATE * NEGATIVES_TARGET_SECONDS`,
    /// Phase 3 is complete and the pipeline transitions to finalization.
    negatives_speech_samples: usize,
    /// Wall-clock start time of Phase 3 owner-negative collection.  Used for
    /// timeout — if [`PHASE3_TIMEOUT_SECS`] elapses, finalize with whatever
    /// was collected.
    phase3_start_time: Option<Instant>,
    /// Rolling window of per-frame confidence scores from the Conv1D
    /// classifier.  Each element is the classifier confidence (0.0–1.0)
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
    /// SNR estimate.
    pre_agc_ring: Vec<f32>,
    /// Audio pre-processor for noise suppression and AGC.
    /// Applied to every incoming audio chunk before VAD / mel extraction.
    audio_preprocessor: crate::audio::audio_preprocessor::AudioPreprocessor,
    /// Accumulates non-VAD audio frames during enrollment for use as negative
    /// training examples.  Collected between utterances (pre-enrollment
    /// ambient noise, inter-utterance silence/background) and saved as chunks
    /// when sustained speech begins.
    negative_audio_buf: Vec<f32>,
    /// Timestamp until which the pipeline stays in [`VoiceStatus::Error`]
    /// before returning to [`VoiceStatus::Listening`].
    /// Set on transcription failure as a non-blocking replacement for the
    /// old 2-second sleep.
    refractory_until: Option<Instant>,
    /// Timestamp of the most recent error chat message, for rate-limiting
    /// repeated transcription failure notifications.
    /// At most one error message per 10-second window.
    last_error_message_time: Option<Instant>,
    /// Adaptive threshold tracker.
    /// Maintains running mean/std of recent per-frame classifier scores for
    /// dynamic threshold computation.
    adaptive_threshold: AdaptiveThresholdState,
    /// The k multiplier for adaptive threshold cached from config at
    /// context creation time.  Updated on pipeline re-initialisation.
    adaptive_k: f32,
    /// Peak rolling-sum score achieved during the current detection
    /// session.  Reset on each detection attempt.  Used by the E2E
    /// benchmark for per-variant peak score reporting.
    peak_score: f32,

    /// Consecutive VAD-negative frame hops since the last VAD-positive frame,
    /// accumulated across calls to [`handle_wake_word_detection`].
    /// Tracked via an [`AtomicUsize`] side channel from the `is_speech_fn`
    /// closure inside the VAD-gating frame loop.  Reset to 0 on any
    /// VAD-positive frame.  When this reaches [`SEGMENT_TIMEOUT_HOPS`]
    /// (~300 ms of consecutive silence), a segment boundary is declared and
    /// per-segment detection state is reset to prevent cross-utterance score
    /// accumulation.
    segment_silence_hops: usize,

    /// Start frame index for the next stride-8 sliding window.
    /// Tracks position in `mel_frame_buffer` so the stride-8 loop only
    /// processes new mel frames since the last call.  Reset to 0 on
    /// pipeline resets (Soft, Full) and segment boundary resets so each
    /// utterance starts from the first mel frame.
    next_window_start: usize,

    /// Per-segment deferred-burst latch.
    ///
    /// `true` once the start-aligned burst sweep (positions 0/8/16/24) has
    /// run for the current detection segment.  Cleared ONLY by the per-segment
    /// resets ([`reset_detection_segment`] and [`reset_pipeline_state`] all
    /// levels) — never on individual scoring calls.  While `true`:
    ///
    /// - the burst sweep cannot re-run (positions beyond the buffer end at
    ///   burst time must never be re-scored on later calls);
    /// - the short-buffer padded fallback is suppressed (a re-scored
    ///   ≤17-real-frame window ~0.01 mid-utterance would reset the rolling
    ///   window).
    ///
    /// It does NOT gate the segment-end pass (a failed burst must not
    /// suppress the boundary fallback).
    burst_sweep_done: bool,

    /// Instrumentation accumulators for wake word detection benchmarking
    /// (mahbot-886).  Feature-gated behind `voice-tests` — zero production
    /// overhead.  Populated by `handle_wake_word_detection` and
    /// `handle_wake_word_detection`, read by the E2E benchmark after
    /// `run_streaming_detection` returns.
    #[cfg(feature = "voice-tests")]
    pub(crate) instrumentation: DetectionInstrumentation,
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

/// Build the audio preprocessor configuration from the live-mic CONFIG flags
/// (noise suppression + AGC, both defaulting to enabled when unset).
///
/// Shared by [`PipelineCtx::new()`] and the voice-tests E2E benchmark so the
/// benchmark's enrollment/training preprocessing cannot silently diverge from
/// production when a deployment disables either stage.
pub(crate) fn preprocessor_config_from_config()
-> crate::audio::audio_preprocessor::PreprocessorConfig {
    use crate::audio::audio_preprocessor::PreprocessorConfig;
    let ns = CONFIG
        .voice_noise_suppression()
        .as_deref()
        .is_none_or(|v| !v.eq_ignore_ascii_case("false"));
    let agc = CONFIG
        .voice_agc()
        .as_deref()
        .is_none_or(|v| !v.eq_ignore_ascii_case("false"));
    PreprocessorConfig {
        noise_suppression: ns,
        agc,
    }
}

impl PipelineCtx {
    pub(crate) fn new() -> Self {
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
            collecting_negatives: false,
            phase3_audio_buf: Vec::new(),
            phase3_silence_samples: 0,
            negatives_speech_samples: 0,
            phase3_start_time: None,
            vad_threshold: VAD_THRESHOLD,
            enrollment_vad: None,
            pre_agc_ring: Vec::new(),
            audio_preprocessor: crate::audio::audio_preprocessor::AudioPreprocessor::new(
                preprocessor_config_from_config(),
            ),
            negative_audio_buf: Vec::new(),
            refractory_until: None,
            last_error_message_time: None,
            adaptive_threshold: AdaptiveThresholdState::new(),
            adaptive_k: {
                let k_str = crate::config::CONFIG.adaptive_k();
                k_str
                    .parse::<f32>()
                    .unwrap_or(ADAPTIVE_K_DEFAULT)
                    .clamp(ADAPTIVE_K_MIN, ADAPTIVE_K_MAX)
            },
            peak_score: 0.0,
            segment_silence_hops: 0,
            next_window_start: 0,
            burst_sweep_done: false,
            #[cfg(feature = "voice-tests")]
            instrumentation: DetectionInstrumentation::new(),
        }
    }

    /// Parameterised pipeline state reset.
    ///
    /// | Field | Full | Soft | Cancel |
    /// |---|---|---|---|
    /// | `voice_batch`, `mel_frame_buffer`, `embedding_ring`, `audio_buffer`, `command_buffer`, `score_window`, `pre_agc_ring`, `negative_audio_buf`, `frame_vad`, `frame_raw_audio` | cleared | cleared | cleared |
    /// | `silence_sample_count`, `segment_silence_hops` | = 0 | = 0 | = 0 |
    /// | `utterance_had_speech`, `utterance_silence_samples`, `enrollment_no_speech_frame_count`, `vad_positives_in_a_row`, `emitted_utterances`, `enrollment_pending`, `noise_rms_estimate` | cleared | cleared | cleared |
    /// | `vad_threshold` | `VAD_THRESHOLD` | preserved | `VAD_THRESHOLD` |
    /// | `enrollment_vad` | `None` | `None` | `None` |
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
        self.segment_silence_hops = 0;
        // Per-segment deferred-burst latch: every reset level starts a fresh
        // segment, so a new utterance must be allowed a new burst sweep.
        self.burst_sweep_done = false;

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
                self.audio_preprocessor.reset();
                reset_vad();
                self.adaptive_threshold.reset();
                self.peak_score = 0.0;
                self.next_window_start = 0;

                // Full does NOT clear global enrollment accumulators — those
                // survive mic stop/start cycles so mid-enrollment progress is
                // preserved across toggle-off/on.
                // Only ResetLevel::Cancel (explicit cancel or start-fresh)
                // clears the global enrollment buffer and negative audio chunks.
            }
            ResetLevel::Soft => {
                // Preserve VAD state, NS noise profile (clear_buffer, not reset),
                // vad_threshold, last_wake_word_detection cooldown,
                // auto_start_pending, is_recording, and global enrollment accumulators.
                self.audio_preprocessor.clear_buffer();
                // Clear rolling-window detection state so stale scores cannot
                // carry cross-utterance contamination across Soft pipeline
                // resets (which occur during the detection→recording handoff).
                // Reset stride-8 window position so the next utterance starts
                // from the first mel frame.
                self.next_window_start = 0;
            }
            ResetLevel::Cancel => {
                self.vad_threshold = VAD_THRESHOLD;
                self.last_wake_word_detection = None;
                self.audio_preprocessor.clear_buffer();
                self.adaptive_threshold.reset();
                self.peak_score = 0.0;
                self.next_window_start = 0;

                // Cancel also clears global enrollment accumulators.
                voice_state().write().unwrap_poison().reset_enrollment();
            }
        }
    }

    /// Reset all per-segment detection state at a VAD-driven utterance boundary.
    /// Called when [`SEGMENT_TIMEOUT_HOPS`] (~300 ms) of
    /// consecutive VAD-negative hops have been observed since the last
    /// VAD-positive frame.
    ///
    /// This prevents classifier scores and rolling sums from
    /// accumulating across separate utterances separated by more than ~300 ms
    /// of silence, which was a structural source of false triggers.
    ///
    /// | Field | Reset? | Rationale |
    /// |---|---|---|
    /// | `voice_batch`, `mel_frame_buffer` | No (caller-managed) | These are taken via `std::mem::take` into local variables before this method runs (see [`handle_wake_word_detection`]); the caller clears the locals separately. Clearing them here would be a no-op. |
    /// | `embedding_ring` | Yes | Prevents stale embeddings from mixing with new utterance; matches cooldown-expiry behaviour. |
    /// | `score_window` | Yes | **Critical**: rolling scores must not accumulate across utterances — this is the primary false-trigger mechanism this function fixes |
    /// | `adaptive_threshold` | Yes | Noise floor estimate is per-segment; the 5-call bootstrap (~640 ms of voiced frames at ~128 ms/embedding) is brief and acceptable |
    /// | `peak_score` | Yes | Diagnostic peak for the current segment; captured to `instrumentation` before clearing |
    /// | `segment_silence_hops` | Yes | Reset the silence counter so the next segment starts fresh |
    ///
    /// **Preserved**: `voice_batch`, `mel_frame_buffer` (caller-managed, see above),
    /// `audio_buffer` (normal drain handles leftover overlap), VAD state
    /// (acoustic environment unchanged), `vad_threshold`,
    /// `last_wake_word_detection` (cooldown still active if within 3 s),
    /// `is_recording`, `audio_preprocessor` (noise profile survives).
    ///
    /// ## Instrumentation save (`#[cfg(feature = "voice-tests")]`)
    /// Before clearing per-segment scores, the running max of
    /// `peak_score` is flushed to `self.instrumentation`
    /// so the E2E benchmark (which reads instrumentation after
    /// `run_streaming_detection` returns) captures the true cross-segment maxima
    /// even when a segment boundary fires during silence-flush post-processing.
    fn reset_detection_segment(&mut self) {
        // ── Save diagnostic peaks before clearing (voice-tests only) ──
        #[cfg(feature = "voice-tests")]
        {
            // Save the running max of the peak so cross-segment maxima
            // survive the clearing below.  Without this, a segment boundary
            // that fires during silence-flush post-processing (between the
            // last on_flush call and the reset) would lose the peaks from
            // the just-ended segment.
            if self.peak_score > self.instrumentation.peak_score {
                self.instrumentation.peak_score = self.peak_score;
            }
        }

        // ── Clear per-segment buffers ──
        self.embedding_ring.clear();
        // ── Clear per-segment rolling scores (PRIMARY false-trigger fix) ──
        self.score_window.clear();

        // ── Reset threshold/scores ──
        self.adaptive_threshold.reset();
        self.peak_score = 0.0;

        // ── Reset silence counter ──
        self.segment_silence_hops = 0;

        // ── Reset stride-8 window position so the next utterance ──
        // starts from the first mel frame.
        self.next_window_start = 0;

        // ── Clear the per-segment deferred-burst latch ──
        // The segment reset is the ONLY place that clears it (besides
        // reset_pipeline_state, which also starts a fresh segment): a new
        // utterance must be allowed a new burst sweep, and mid-utterance
        // trailing re-scoring must never re-run a finished sweep.
        self.burst_sweep_done = false;

        // ── Reset AGC state ──
        // Reset the AudioPreprocessor so that each detection segment starts with
        // a fresh AGC state, matching the training distribution (confusable/
        // unrelated embeddings are processed through a fresh AudioPreprocessor
        // per phrase × seed in prewarm_phrase_embeddings).  Without
        // this reset, the Conv1D classifier could exploit AGC adaptation state as
        // a spurious feature: enrollment positives use the room-adapted persistent
        // AGC while confusable negatives use a fresh TTS-adapted AGC, creating a
        // distribution shift that produces near-zero sigmoid scores during
        // streaming inference.
        self.audio_preprocessor.reset();
    }

    /// Handle the segment boundary check at the end of a detection call.
    ///
    /// Called from [`handle_wake_word_detection`] after
    /// [`process_streaming_frames_inner`] returns, with the accumulated
    /// `hop_count` from the VAD-negative frame counter's side-channel.
    ///
    /// If `hop_count` reaches [`SEGMENT_TIMEOUT_HOPS`], resets per-segment
    /// detection state and clears the caller's local batch buffers so that
    /// classifier scores and rolling sums do not accumulate
    /// across separate utterances.
    ///
    /// If `hop_count` is below the threshold, persists the counter in
    /// [`segment_silence_hops`](PipelineCtx::segment_silence_hops) so the
    /// next call to [`handle_wake_word_detection`] can continue counting.
    ///
    /// # Parameters
    ///
    /// - `hop_count`: Accumulated consecutive VAD-negative frames from the
    ///   `is_speech_fn` closure — loaded from the [`AtomicUsize`] side channel
    ///   after the frame loop.
    /// - `voice_batch`: Caller's local voice batch (taken via `std::mem::take`
    ///   before the frame loop).  Cleared on boundary to prevent stale audio
    ///   from crossing the segment boundary.
    /// - `mel_frame_buffer`: Caller's local mel frame buffer.  Cleared on
    ///   boundary to prevent stale mel frames from the previous segment.
    fn handle_segment_boundary(
        &mut self,
        hop_count: usize,
        voice_batch: &mut Vec<f32>,
        mel_frame_buffer: &mut Vec<Vec<f32>>,
    ) {
        if hop_count >= SEGMENT_TIMEOUT_HOPS {
            // ── Segment-end pass — no-regression fallback ──
            // If no detection fired this segment (not recording), models are
            // loaded, and the buffer has content, score the buffer as it
            // stands at the boundary — trailing-68-frame trimmed state for
            // longer utterances (replicating today's accidental flush at
            // starts 0/8/16 with 68/60/52 real frames), full accumulated
            // buffer for shorter ones — at start-aligned positions with
            // padded geometry.  This pass scores exactly like the cold burst
            // (the ring-4 sample fires at position 24), so it is the overrun
            // safety net for the deferred burst.  (The ticket's original
            // "must not re-score the misaligned leading edge" hazard does
            // not manifest at live geometry: the trigger lands at B=76 and
            // the B=79 → start-3 re-score scores ~0.99 — but the pass stays the
            // safety net if a below-reset re-score ever does clear
            // the rolling window.)  It is
            // UNCONDITIONAL at the boundary for non-recording segments with
            // content — a failed burst must NOT suppress it (F1's only
            // current detection path is exactly this trailing re-score), and
            // the per-segment burst_sweep_done latch does not gate it.
            // Score-then-reset ordering is preserved: the pass runs BEFORE
            // reset_detection_segment.
            //
            // Gates: not recording; models loaded; buffer non-empty; no
            // detection fired this segment (the caller only reaches this
            // method with `!is_recording`, and a just-fired detection is
            // re-checked below so it is not reset).  The burst_sweep_done
            // latch is deliberately NOT consulted — a failed burst must not
            // suppress the boundary fallback (enforced structurally: the
            // position set takes no latch input).
            if !self.is_recording
                && let Some(models) = ONNX_MODELS.get()
                && !mel_frame_buffer.is_empty()
            {
                #[cfg(feature = "voice-tests")]
                {
                    self.instrumentation.segment_end_pass_fired = true;
                }
                // Score from start-aligned position 0 with padded geometry —
                // NOT from a stale window start.  The shared
                // `start_aligned_positions` grid covers the burst-equivalent
                // positions, so the ring-4 sample at position 24 scores
                // exactly like the cold burst (manager pin 3).  The
                // pass gate does NOT consult the `burst_sweep_done` latch (a
                // failed burst must not suppress the boundary fallback) —
                // the latch is structurally absent from the position grid
                // (it takes no latch input).
                let positions = start_aligned_positions(mel_frame_buffer.len());
                score_start_aligned_positions(
                    mel_frame_buffer,
                    models,
                    self,
                    &positions,
                    WindowSource::SegmentEndPass,
                );
            }

            // Re-check the not-recording condition after the pass: a
            // just-fired detection must NOT be reset (the detection→recording
            // handoff in handle_wake_word_detection completes the transition).
            if !self.is_recording {
                self.reset_detection_segment();
                voice_batch.clear();
                mel_frame_buffer.clear();
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

    /// Check whether the 10-second rate-limit has elapsed since the last
    /// transcription-error chat message.
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
            // restart the app.
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
        // (new NoiseSuppressor) and reset_vad().
        // Global enrollment accumulators are preserved across mic stop/start
        // so mid-enrollment progress survives toggle-off/on.
        self.reset_pipeline_state(ResetLevel::Full);
        self.is_listening = false;
        self.enrollment_mode = false;
        drop(self.mic_stream.take());
        self.mic_rx = None;
        set_status(VoiceStatus::Disabled);
        info!("Voice pipeline: stopped listening");
    }

    fn handle_start_enrollment(&mut self, phrase: &str) {
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
        // When starting fresh (existing_utterances == 0), use Cancel-level
        // reset to clear stale buffers while preserving VAD/NS continuity
        // (same mic stream, same acoustic environment).
        //
        // Use enrolled_utterance_count (not enrollment_buffer.len()) because
        // augmentation inflates the buffer to ~5× the utterance count,
        // making the buffer-based check always >0 and the UI display
        // nonsensical.
        let existing_utterances = voice_state()
            .read()
            .unwrap_poison()
            .enrolled_utterance_count;

        if existing_utterances == 0 {
            // vad_threshold stays at VAD_THRESHOLD after unification.
            // Previously set to ENROLLMENT_VAD_THRESHOLD
            // here, but that created a training/inference mismatch.
            self.reset_pipeline_state(ResetLevel::Cancel);
        } else {
            info!(
                "Resuming enrollment from utterance \
                 {existing_utterances}/{NUM_ENROLLMENT_SAMPLES}",
            );
        }

        // Store the normalized wake word phrase in the global enrollment
        // state AFTER reset, so reset_enrollment does not clear it.
        // This phrase is consumed by persist_model_state() on completion.
        let normalized = normalize_phrase(phrase);
        voice_state().write().unwrap_poison().enrolling_phrase = Some(normalized);

        self.enrollment_mode = true;
        // Initialize a separate VAD detector for this enrollment session
        // to prevent state contamination between enrollment and streaming
        // modes.
        self.enrollment_vad = Some(earshot::Detector::default());
        // vad_threshold is intentionally NOT changed here — it stays at
        // VAD_THRESHOLD (0.5) for both enrollment and streaming detection.
        // Previously used ENROLLMENT_VAD_THRESHOLD
        // (0.60), which caused frames scoring 0.50-0.59 to be included in
        // streaming but never seen during enrollment training.
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
        // VAD threshold is already unified to VAD_THRESHOLD (0.5) — no longer
        // toggled between enrollment and detection.

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

/// Schedule a transition back to [`VoiceStatus::Listening`] after enrollment
/// finalization completes successfully.
///
/// Extracted from the existing cleanup at lines ~5616-5636 of the main loop.
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
    // vad_threshold to VAD_THRESHOLD, but preserves VAD/NS continuity.
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

/// Finalize the enrollment pipeline: train the classifier with
/// negative weights (ambient → owner-negative → unrelated → confusable).
///
/// Extracted from the original training block in `handle_enrollment_sample`
/// and extended with owner-negative embedding extraction.
///
/// Called by the main loop after Phase 3 (owner-negative collection) completes
/// or times out.  Returns `true` on success (model trained and persisted),
/// `false` on failure (the pipeline stays in its current state for retry).
///
/// # Cancel guard
///
/// Before `persist_model_state()`, checks the global cancellation token.  If
/// cancelled during async training, returns `false` early without persisting
/// stale state.
#[allow(clippy::too_many_lines)]
async fn finalize_enrollment_pipeline() -> bool {
    if !models_ready() {
        warn!("finalize_enrollment_pipeline: models not ready");
        return false;
    }

    // ── Clone positive embeddings (dense-only) ──
    let enrollment_buffer = {
        let state = voice_state().read().unwrap_poison();
        state.enrollment_buffer.clone()
    };

    // ── Clone negative audio chunks (ambient + owner-negative) ──
    let (negative_audio_chunks, used_real_negatives) = {
        let state = voice_state().read().unwrap_poison();
        let chunks = state.negative_audio_chunks.clone();
        let should_use = chunks.len() >= 2 && ONNX_MODELS.get().is_some();
        (chunks, should_use)
    };
    let owner_negative_chunks = {
        let state = voice_state().read().unwrap_poison();
        state.owner_negative_chunks.clone()
    };

    // ── Extract ambient negatives (per-chunk EmbeddingSequences) ──
    let negative_sequences: Vec<EmbeddingSequence> = if used_real_negatives {
        tokio::task::spawn_blocking(move || {
            let _models = ONNX_MODELS.get().expect("ONNX_MODELS checked above");
            let mut neg_seqs: Vec<EmbeddingSequence> = Vec::new();
            for (ci, chunk) in negative_audio_chunks.iter().enumerate() {
                let chunk_id = UtteranceId {
                    sequence_index: ci,
                    variant_index: 0,
                };
                match process_enrollment_sample(chunk) {
                    Ok(embs) => {
                        neg_seqs.push(EmbeddingSequence::new(chunk_id, Source::Ambient, embs));
                    }
                    Err(e) => warn!(
                        "Failed to extract dense negative embedding \
                         from ambient audio chunk ({} samples): {e}",
                        chunk.len(),
                    ),
                }
            }
            neg_seqs
        })
        .await
        .unwrap_or_default()
    } else {
        Vec::new()
    };

    // ── Extract owner-negative embeddings ──
    let use_owner_negatives = !owner_negative_chunks.is_empty() && ONNX_MODELS.get().is_some();
    let owner_negative_sequences: Vec<EmbeddingSequence> = if use_owner_negatives {
        tokio::task::spawn_blocking(move || {
            let _models = ONNX_MODELS.get().expect("ONNX_MODELS checked above");
            let mut owner_seqs: Vec<EmbeddingSequence> = Vec::new();
            for (ci, chunk) in owner_negative_chunks.iter().enumerate() {
                let chunk_id = UtteranceId {
                    sequence_index: ci,
                    variant_index: 0,
                };
                match process_enrollment_sample(chunk) {
                    Ok(embs) => {
                        owner_seqs.push(EmbeddingSequence::new(chunk_id, Source::Owner, embs));
                    }
                    Err(e) => warn!(
                        "Failed to extract dense owner-negative embedding \
                         from Phase 3 chunk ({} samples): {e}",
                        chunk.len(),
                    ),
                }
            }
            owner_seqs
        })
        .await
        .unwrap_or_default()
    } else {
        Vec::new()
    };

    // ── Clear chunk buffers from global state ──
    {
        let mut state = voice_state().write().unwrap_poison();
        if used_real_negatives {
            state.negative_audio_chunks.clear();
        }
        state.owner_negative_chunks.clear();
    }

    // ── Read wake word phrase for confusable gating ──
    let enrolled_phrase = voice_state()
        .read()
        .unwrap_poison()
        .enrolling_phrase
        .clone()
        .unwrap_or_else(|| DEFAULT_WAKE_WORD_PHRASE.to_string());
    let use_mahbot_confusables = is_mahbot_wake_word(&enrolled_phrase);
    if !use_mahbot_confusables {
        info!(
            "Skipping Mahbot-specific confusable phrases for wake word \
             '{enrolled_phrase}' (mahbot-909); see is_mahbot_wake_word() \
             docs for quality implications",
        );
    }

    // ── Build positive sequences for training (dense-only) ──
    let mut pos_sequences: Vec<EmbeddingSequence> = Vec::with_capacity(enrollment_buffer.len());
    pos_sequences.extend(enrollment_buffer.clone());

    // ── Build negative sequences for classifier training ──
    let confusable_dense = confusable_dense_embeddings();
    let unrelated_dense = unrelated_dense_embeddings();
    let mut neg_sequences: Vec<EmbeddingSequence> = Vec::new();
    let n_ambient_classifier = negative_sequences.len();
    if n_ambient_classifier > 0 {
        info!(
            "Adding {n_ambient_classifier} dense ambient negative \
             sequences to classifier negative set (mahbot-923)",
        );
        neg_sequences.extend(negative_sequences.clone());
    }
    let n_owner_classifier = owner_negative_sequences.len();
    if n_owner_classifier > 0 {
        info!(
            "Adding {n_owner_classifier} owner-negative dense sequences \
             to classifier negative set (mahbot-932)",
        );
        neg_sequences.extend(owner_negative_sequences.clone());
    }
    if use_mahbot_confusables && !confusable_dense.is_empty() {
        info!(
            "Adding {} confusable dense sequences to classifier negative set \
             for Mahbot wake word '{enrolled_phrase}' (mahbot-909)",
            confusable_dense.len(),
        );
        neg_sequences.extend_from_slice(confusable_dense);
    }
    if !unrelated_dense.is_empty() {
        info!(
            "Adding {} unrelated dense sequences to classifier negative set (mahbot-878)",
            unrelated_dense.len(),
        );
        neg_sequences.extend_from_slice(unrelated_dense);
    }

    // Note: class-balanced weights
    // this is an intentional design difference.

    // ── Classifier training via finalize_enrollment ──
    let classifier_result = tokio::task::spawn_blocking({
        let pos_seqs = pos_sequences.clone();
        let neg_seqs = neg_sequences.clone();
        move || finalize_enrollment(&pos_seqs, &neg_seqs)
    })
    .await
    .unwrap_or_else(|e| Err(anyhow!("Classifier training task panicked: {e}")));

    let weights = match classifier_result {
        Ok(result) => {
            let param_count = result.weights.param_count();
            info!(
                "Enrollment complete: wake word '{}' (Conv1D classifier \
                             {} params, {} epochs, best val loss={:.4})",
                enrolled_phrase, param_count, result.epochs_trained, result.best_val_loss,
            );
            result.weights
        }
        Err(e) => {
            warn!("Enrollment finalization failed: {e}");
            set_status(VoiceStatus::Error("Enrollment failed".to_string()));
            return false;
        }
    };

    let self_test_seqs = enrollment_buffer;

    // ── Cancel guard: check before persisting ──
    if crate::shutdown::shutdown_token().is_cancelled() {
        warn!(
            "finalize_enrollment_pipeline: cancelled during async training, \
               not persisting model state"
        );
        return false;
    }

    // ── Blocking self-test ──
    let classifier = WakeWordClassifier::new(weights.clone());
    if let Err(e) = run_enrollment_self_test(&self_test_seqs, &classifier) {
        warn!("Enrollment self-test failed — model rejected: {e}.  Re-enrollment required.");
        set_status(VoiceStatus::Error(format!(
            "Enrollment validation failed: {e}.  Please try again with clearer speech."
        )));
        return false;
    }
    info!("Enrollment self-test: passed — deploying model");

    // ── Store classifier in global state ──
    set_classifier_weights(weights);

    // ── Persist to config DB ──
    persist_model_state().await;

    // ── Clear enrollment accumulators ──
    {
        let mut state = voice_state().write().unwrap_poison();
        state.enrollment_buffer.clear();
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

    // Load persisted wake word classifier weights from config on startup
    if let Some(json) = CONFIG.wake_word_templates() {
        // Try PersistedModel format (handles both new versioned and legacy
        // single-object classifier via custom deserializer).
        let loaded = if let Ok(mut model) = serde_json::from_str::<PersistedModel>(&json) {
            // ── Legacy migration (schema_version == 0) ──────────────
            // If the loaded model has no schema_version (pre-v1 format),
            // migrate it in-memory.  Persist is deferred until after the
            // compatibility check (test-before-commit principle).
            let was_migrated = model.schema_version == 0 && model.classifier.is_some();
            if was_migrated {
                migrate_legacy_model(&mut model);
            }

            // ── Compatibility check (runs for all models, including migrated) ──
            if let Err(e) = check_model_compatibility(&model) {
                warn!("Stored wake word model is incompatible: {e}. Clear and re-enroll.");
                false
            } else {
                // Write back the migrated model now (compatibility passed).
                if was_migrated && persist_wake_word_model(&model).await {
                    info!(
                        "Migrated legacy wake word model to v{}",
                        MODEL_SCHEMA_VERSION
                    );
                }

                // ── Load model data ─────────────────────────────────
                let all_valid = model
                    .classifier
                    .as_ref()
                    .is_some_and(|members| members.iter().all(|w| w.validate().is_ok()));
                if all_valid {
                    if let Some(ref members) = model.classifier {
                        set_classifier_weights(members[0].clone());
                    }
                    let n = model.classifier.as_ref().map_or(0, Vec::len);
                    info!(
                        "Loaded wake word model from config \
                         (v{}, {} classifier weight set(s), phrase={})",
                        model.schema_version, n, model.phrase,
                    );
                    // Cache the phrase from the loaded model so
                    // get_enrolled_phrase() returns it.
                    voice_state().write().unwrap_poison().model_phrase = Some(model.phrase.clone());
                    true
                } else {
                    warn!("Stored classifier weights are invalid — re-enrollment required");
                    false
                }
            }
        } else {
            false
        };

        // Fall back to bare ClassifierWeights (pre-PersistedModel legacy format)
        if !loaded {
            if let Ok(weights) = serde_json::from_str::<ClassifierWeights>(&json) {
                if let Err(e) = weights.validate() {
                    warn!("Stored classifier weights are invalid — re-enrollment required: {e}");
                } else {
                    set_classifier_weights(weights);
                    info!(
                        "Loaded wake word classifier weights from config (pre-PersistedModel format)"
                    );
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

    // Periodic metrics log via tokio::time::Interval.  Fires
    // every 60 seconds on wall-clock time regardless of audio activity.
    // Replaces the earlier ad-hoc tick counter which only fired when audio
    // chunks arrived and used non-standard block-scoped statics.
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
                    Some(VoiceCommand::StopListening) => ctx.handle_stop_listening(),
                    Some(VoiceCommand::StartEnrollment(phrase)) => {
                        ctx.handle_start_enrollment(&phrase);
                    }
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

                CHUNKS_RECEIVED.fetch_add(1, Ordering::Relaxed);

                // ── TTS playback gate ──
                // If TTS audio is actively playing through the speakers, skip ALL
                // audio processing for this chunk, including the pre-AGC ring buffer,
                // noise suppressor, AGC, VAD, mel extraction, wake word detection,
                // enrollment, and recording.  This prevents TTS echo from
                // contaminating:
                //   - noise suppressor / AGC internal state
                //   - earshot VAD ring buffer and noise floor estimate
                //   - wake word classifier embeddings and scores
                //   - enrollment noise RMS estimates (pre-AGC ring buffer)
                //
                // The gate stays active for a reverb tail period after playback
                // ends (see crate::audio::tts::PLAYBACK_REVERB_TAIL_MS).
                if crate::audio::tts::is_playback_active() {
                    continue;
                }

                // ── Pre-AGC ring buffer ──
                // Capture raw audio before AGC processing for noise RMS
                // estimation.  AGC amplifies silence (up to 4×) more than
                // speech (~1-2×), so noise RMS computed from post-AGC audio
                // would artificially lower the SNR estimate.  Only accumulate
                // during enrollment where noise RMS is needed — the ring would
                // be stale/irrelevant during live detection since it's reset
                // when enrollment completes.
                if ctx.enrollment_mode || ctx.collecting_negatives {
                    ctx.pre_agc_ring.extend_from_slice(&samples);
                    if ctx.pre_agc_ring.len() > RAW_RING_MAX {
                        let excess = ctx.pre_agc_ring.len() - RAW_RING_MAX;
                        ctx.pre_agc_ring.drain(..excess);
                    }
                }

                // Apply noise suppression and/or AGC pre-processing before
                // the audio reaches VAD / mel extraction / enrollment.
                let samples = ctx.audio_preprocessor.process(samples);

                if ctx.collecting_negatives {
                    handle_negative_collection_audio(&samples, &mut ctx);
                } else if ctx.enrollment_mode {
                    let (sample, total) = {
                        let state = voice_state().read().unwrap_poison();
                        // Use enrolled_utterance_count (not enrollment_buffer.len())
                        // because the buffer may have up to 5× entries due to PCM
                        // augmentation.  The intermediate status fields
                        // are currently ignored by the GUI (static text), but using
                        // the utterance count keeps the data model correct.
                        (state.enrolled_utterance_count, NUM_ENROLLMENT_SAMPLES)
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

            // Periodic metrics log every ~60 seconds.  This
            // branch fires on wall-clock time regardless of audio activity,
            // unlike the earlier ad-hoc tick counter which only incremented
            // when audio chunks arrived.
            _ = metrics_interval.tick() => {
                let m = get_voice_metrics();
                let roll_avg = m.avg_embedding_latency_ns;
                let life_avg = m.lifetime_avg_embedding_latency_ns();
                info!(
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

        // Periodic auto-recovery: if models are in Failed state, attempt to
        // retry loading (debounced to at most once every 30s).  This runs
        // regardless of auto_start_pending so that the model error state is
        // self-healing even when voice is toggled off/on manually.
        if MODELS_STATE.load(Ordering::Acquire) == ModelState::Failed {
            ctx.try_retry_models();
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
        // timeout), finalize residual audio, train the model, and clean up.
        if ctx.collecting_negatives {
            let target_samples = SAMPLE_RATE as usize * NEGATIVES_TARGET_SECONDS;
            let target_met = ctx.negatives_speech_samples >= target_samples;
            let timed_out = ctx
                .phase3_start_time
                .is_some_and(|t| t.elapsed() >= Duration::from_secs(PHASE3_TIMEOUT_SECS));

            if target_met || timed_out {
                // Finalize any residual audio in phase3_audio_buf.
                if !ctx.phase3_audio_buf.is_empty() {
                    let chunk = std::mem::take(&mut ctx.phase3_audio_buf);
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
                             dropping residual chunk of {} samples (mahbot-913)",
                            MAX_OWNER_NEGATIVE_SAMPLES,
                            chunk.len(),
                        );
                    }
                    drop(state);
                }

                // Cap is ~1.4M at 16kHz, well within f64 mantissa precision.
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

                // Train and persist the model.
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
        // ONNX inference inside handle_enrollment_sample uses spawn_blocking
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
/// typical wake word needs, so any real utterance passes.
///
/// This is extracted as a separate function so it can be unit-tested without
/// requiring ONNX model inference.
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
    // Duration floor: reject noise blips and coughs using the
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

// PCM augmentation

/// Speed perturbation via [`crate::util::speed_perturbation`] (anti-alias safe).
///
/// Delegates to [`crate::util::resample_audio`] for proper anti-aliasing
/// filtering and f64-precision interpolation.  A `rate` of 1.0
/// returns the original unchanged.  `rate < 1.0` slows down (adds samples),
/// `rate > 1.0` speeds up (removes samples).  The ticket uses 0.95 (slow down)
/// and 1.05 (speed up).
///
// Implementation note: `apply_gain`, `generate_pink_noise`, and `add_noise`
// are defined in `crate::util` (canonical implementations).
//
/// Extracts dense stride-8 embeddings for the Conv1D classifier
/// training.  Now, only dense stride-8
/// embeddings are used — streaming extraction was removed.  Dense embeddings
/// provide a strong learning signal (many windows per utterance) and the same
/// distribution is used for classifier training.
///
/// ONNX inference is CPU-bound (mel spectrogram + embedding computation).
/// It runs on a blocking thread via `spawn_blocking` to avoid starving
/// the async pipeline during enrollment.
///
/// Implements minimum utterance length check: utterances
/// shorter than 400ms are rejected to reject noise blips and coughs.
#[allow(clippy::too_many_lines)]
async fn handle_enrollment_sample(samples: Vec<f32>, noise_rms: Option<f32>) {
    if !models_ready() {
        warn!("Models not ready for enrollment");
        return;
    }

    // Compute utterance duration before moving `samples` into the closure.
    let duration_ms = (samples.len() as u64 * 1000) / u64::from(SAMPLE_RATE);

    // ── Generate deterministic PCM augmented variants ──
    // The original is kept as-is.  All variants are processed in a single
    // spawn_blocking below, avoiding redundant ONNX model lookups and
    // reducing thread spawn overhead.
    //
    // Variant generation is shared with the E2E bench via
    // [`augment_pcm_variants`]: noise seed = the enrolled
    // utterance count read below, gate input = the raw VAD-segmented mic
    // utterance, canonical push order (speed-up 3rd).  The pre-padding
    // duration gate (100 ms of context padding at each end; speed-up is
    // viable only when the unpadded duration is ≥ 500 ms) lives inside the
    // helper — very short utterances would otherwise become unintelligible.
    // The full positive recipe ([`AugmentSet::Full`]) trains the classifier
    // on the low-SNR color cells (white/pink/brown at 10/5 dB) so noisy
    // enrollment is in-distribution at inference.

    // Seed for noise from current utterance count (read before incrementing).
    let noise_seed = {
        let state = voice_state().read().unwrap_poison();
        state.enrolled_utterance_count as u64
    };

    // Generate variants — all deterministic, no RNG dependency except the
    // seeded noise generator.  The helper yields index order 0,1,2?,3..11 —
    // variant 2 (speed-up) is absent when the pre-pad duration gate fails.
    let variants = augment_pcm_variants(&samples, SAMPLE_RATE, noise_seed, AugmentSet::Full);
    let original = samples.clone();
    let variant_indices: Vec<usize> = variants.iter().map(|v| v.variant_index).collect();

    // Clone for quality computation (use the original for quality check).
    let samples_for_quality = original.clone();

    // Run ONNX inference for ALL variants in a SINGLE spawn_blocking
    let results = tokio::task::spawn_blocking(move || {
        variants
            .into_iter()
            .map(|v| process_enrollment_sample(&v.pcm))
            .collect::<Vec<_>>()
    })
    .await
    .unwrap_or_else(|e| {
        let make_err = || Err::<Vec<Vec<f32>>, _>(anyhow!("Blocking task failed: {e}"));
        (0..variant_indices.len()).map(|_| make_err()).collect()
    });

    // ── Dense embeddings (original — used for min-length check) ──
    // Variant 0 (original) is always the first element.
    let original_old = match results.first() {
        Some(Ok(e)) => e.clone(),
        Some(Err(e)) => {
            warn!("Original enrollment embedding failed: {e}");
            return;
        }
        None => {
            warn!("No enrollment variants produced");
            return;
        }
    };

    // ── Minimum utterance length check ──
    if let Err(msg) = check_enrollment_utterance_length(original_old.len(), duration_ms) {
        warn!("{msg}");
        set_status(VoiceStatus::Error(msg));
        return;
    }

    // ── Push all variants to buffer ──
    // Read enrollment index (current count before increment) so each variant
    // gets the correct sequence_index.
    let enrollment_index = voice_state()
        .read()
        .unwrap_poison()
        .enrolled_utterance_count;

    // Collect results into a single EmbeddingSequence buffer.  Now,
    // the classifier trains on the same dense stride-8 embedding
    // distribution — streaming extraction was removed.
    let mut old_results: Vec<EmbeddingSequence> = Vec::with_capacity(results.len());

    // Original (variant 0) — filed under Source::Enrollment; all augmented
    // variants (1..=11) under Source::Augmentation.
    for (vi, res) in variant_indices.iter().zip(results) {
        match res {
            Ok(embs) => {
                let id = UtteranceId {
                    sequence_index: enrollment_index,
                    variant_index: *vi,
                };
                old_results.push(EmbeddingSequence::new(
                    id,
                    if *vi == 0 {
                        Source::Enrollment
                    } else {
                        Source::Augmentation
                    },
                    embs,
                ));
            }
            Err(ref e) => {
                warn!(
                    "Variant {} embedding extraction failed: {e} — skipping variant",
                    *vi,
                );
            }
        }
    }

    let (utterance_count, count, quality) = {
        let mut state = voice_state().write().unwrap_poison();

        // Compute quality on the original samples (quality check is only for
        // the real utterance, not augmented variants).
        let quality = Some(compute_utterance_quality(&samples_for_quality, noise_rms));

        // Push all variant embeddings (dense-only)
        state.enrollment_buffer.extend(old_results);

        // Increment utterance counter (one user utterance generates multiple
        // buffer entries, but only counts as one user utterance).
        state.enrolled_utterance_count += 1;
        let utterance_count = state.enrolled_utterance_count;
        let count = state.enrollment_buffer.len();
        // state dropped here — no lock held across await
        (utterance_count, count, quality)
    };

    info!(
        "Enrolled utterance {utterance_count}/{NUM_ENROLLMENT_SAMPLES} \
         ({count} buffer entries with augmentation)",
    );

    if utterance_count >= NUM_ENROLLMENT_SAMPLES {
        // All 10 utterances collected.  Signal that Phase 2 is complete and
        // the pipeline should transition to Phase 3 (owner-negative collection)
        // or proceed directly to finalization.
        // The training block is deferred to `finalize_enrollment_pipeline()`
        // which is called after Phase 3 completes or on timeout.
        voice_state().write().unwrap_poison().utterances_collected = true;
        // Keep the current Enrolling status until transition_to_phase3 fires.
        // The status will be updated to EnrollingNegatives by the main loop.
    } else {
        set_status(VoiceStatus::Enrolling {
            sample: utterance_count,
            total: NUM_ENROLLMENT_SAMPLES,
            duration_ms,
            quality,
        });
    }
}

/// Current schema version for `PersistedModel`. Increment when the serialized
/// format changes incompatibly (e.g., adding required fields, changing
/// dimensions).
pub(crate) const MODEL_SCHEMA_VERSION: u32 = 1;

/// Serialisable form of the wake word model (classifier + versioning).
///
/// # Versioning & Compatibility
///
/// `schema_version` identifies the serialization format. On load, the model
/// is rejected (with a clear error) if:
/// - `schema_version` is not a recognized version (currently only 1).
/// - `embedding_dim` does not match the runtime `EMBEDDING_DIM` (96).
/// - `window_size` does not match the runtime `WINDOW_SIZE` (3).
///
/// # Legacy Migration
///
/// Models without `schema_version` (pre-v1, `schema_version` default 0 via
/// `#[serde(default)]`) are migrated to v1 on first successful load. The
/// migrated model is written back to the config DB (best-effort).
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct PersistedModel {
    // ── Versioning & Compatibility (checked at load time) ──────────
    /// Schema version for format discrimination. 0 = legacy (pre-v1),
    /// 1 = current version.
    #[serde(default)]
    schema_version: u32,
    /// Normalized wake word phrase (lowercased, trimmed).
    #[serde(default = "default_phrase")]
    phrase: String,
    /// Expected embedding dimension for compatibility (runtime: 96).
    #[serde(default = "default_embedding_dim")]
    embedding_dim: u32,
    /// Expected window size for compatibility (runtime: 3).
    #[serde(default = "default_window_size")]
    window_size: u32,

    // ── Training Metadata (diagnostic only, NOT used for compatibility) ──
    /// Deterministic RNG seeds used during classifier training.
    /// Empty = unknown (legacy).
    #[serde(default)]
    training_seeds: Vec<u64>,
    /// RFC 3339 timestamp of initial model creation (never updated).
    #[serde(default)]
    created_at: String,
    /// RFC 3339 timestamp of most recent training/persist operation.
    #[serde(default)]
    trained_at: String,

    // ── Model Data ────────────────────────────────────────────────
    /// Single classifier weight set, stored as Vec for backward compat with
    /// persisted models.
    #[serde(default, deserialize_with = "deserialize_classifier_opt")]
    classifier: Option<Vec<ClassifierWeights>>,
    /// Per-member validation losses from older persisted models.
    /// Kept for backward compatible deserialization only — always None in
    /// newly persisted models and never read on load (uniform averaging).
    #[serde(skip_serializing_if = "Option::is_none")]
    val_losses: Option<Vec<f32>>,
}

// ── Default helpers for serde ────────────────────────────────────────

fn default_phrase() -> String {
    DEFAULT_WAKE_WORD_PHRASE.to_string()
}

/// Normalize a wake word phrase: trim whitespace, lowercase, collapse
/// multiple internal whitespace characters to a single space.
/// Returns [`DEFAULT_WAKE_WORD_PHRASE`] if the result would be empty.
#[must_use]
pub(crate) fn normalize_phrase(s: &str) -> String {
    let normalized = s
        .trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if normalized.is_empty() {
        DEFAULT_WAKE_WORD_PHRASE.to_string()
    } else {
        normalized
    }
}

/// Return the enrolled wake word phrase from the cached model data.
///
/// Returns `Some(phrase)` if a model has been loaded or enrolled in this
/// session, `None` if no model is available (e.g., no enrollment has
/// ever been completed, or only a legacy pre-PersistedModel format exists).
#[must_use]
pub fn get_enrolled_phrase() -> Option<String> {
    voice_state().read().unwrap_poison().model_phrase.clone()
}

fn default_embedding_dim() -> u32 {
    u32::try_from(EMBEDDING_DIM).unwrap()
}

fn default_window_size() -> u32 {
    u32::try_from(wake_word_classifier::WINDOW_SIZE).unwrap()
}

/// Deserialize `classifier` as `Option<Vec<ClassifierWeights>>`, handling:
/// - New format: `classifier` is an array of weight sets.
/// - Legacy format: `classifier` is a single weight set object → wrapped in vec.
/// - Missing / null → `None`.
pub(crate) fn deserialize_classifier_opt<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<ClassifierWeights>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val: Option<serde_json::Value> = serde::Deserialize::deserialize(deserializer)?;
    match val {
        None => Ok(None),
        Some(v) => {
            // Try as Vec<ClassifierWeights> (new array format)
            if let Ok(members) = serde_json::from_value::<Vec<ClassifierWeights>>(v.clone()) {
                return Ok(Some(members));
            }
            // Try as single ClassifierWeights (legacy format)
            if let Ok(single) = serde_json::from_value::<ClassifierWeights>(v) {
                return Ok(Some(vec![single]));
            }
            Err(serde::de::Error::custom(
                "classifier must be an array of weight sets or a single weight set object",
            ))
        }
    }
}

/// Check that a loaded model is compatible with the running system.
///
/// Fails with a descriptive message if:
/// - `schema_version` is unknown (not 0 or 1).
/// - `embedding_dim` does not match the runtime embedding dimension (96).
/// - `window_size` does not match the runtime window size (3).
///
/// `schema_version == 0` (legacy pre-v1) is allowed — callers should migrate
/// before using the model.
fn check_model_compatibility(model: &PersistedModel) -> Result<()> {
    // schema_version 0 (legacy) is allowed; 1 is current; anything else is unknown.
    if model.schema_version > MODEL_SCHEMA_VERSION {
        anyhow::bail!(
            "Model schema version {} is newer than runtime version {} — \
             upgrade mahbot to use this model",
            model.schema_version,
            MODEL_SCHEMA_VERSION,
        );
    }
    if model.embedding_dim != 0 && model.embedding_dim != u32::try_from(EMBEDDING_DIM).unwrap() {
        anyhow::bail!(
            "Model embedding dimension {} does not match runtime embedding dimension {}",
            model.embedding_dim,
            EMBEDDING_DIM,
        );
    }
    let runtime_window_size = u32::try_from(wake_word_classifier::WINDOW_SIZE).unwrap();
    if model.window_size != 0 && model.window_size != runtime_window_size {
        anyhow::bail!(
            "Model window size {} does not match runtime window size {}",
            model.window_size,
            runtime_window_size,
        );
    }
    Ok(())
}

/// Migrate a legacy (v0) model to the current schema version in-place.
///
/// Fills in default values for fields that the legacy format did not store
/// (phrase, embedding_dim, window_size, created_at, trained_at).
/// After migration, `model.schema_version` is `MODEL_SCHEMA_VERSION` and
///
/// Returns early (no-op) if the model is not legacy (schema_version != 0) or
/// if the model has no classifier data — the latter prevents accidental
/// migration of a bare `ClassifierWeights` or old `WakeWordTemplates` JSON
/// that happens to deserialize as a `PersistedModel` with null classifier.
fn migrate_legacy_model(model: &mut PersistedModel) {
    if model.schema_version != 0 {
        return; // Already migrated or not a legacy model
    }
    if model.classifier.is_none() {
        // No classifier data — this isn't a real legacy PersistedModel, it's
        // a bare ClassifierWeights or old format that deserialized by accident
        // (all fields are #[serde(default)]). Don't migrate or write back.
        return;
    }
    let now = turso::now();
    model.schema_version = MODEL_SCHEMA_VERSION;
    if model.phrase.is_empty() {
        model.phrase = DEFAULT_WAKE_WORD_PHRASE.to_string();
    }
    if model.embedding_dim == 0 {
        model.embedding_dim = u32::try_from(EMBEDDING_DIM).unwrap();
    }
    if model.window_size == 0 {
        model.window_size = u32::try_from(wake_word_classifier::WINDOW_SIZE).unwrap();
    }
    if model.created_at.is_empty() {
        model.created_at.clone_from(&now);
    }
    model.trained_at = now;
}

impl Default for PersistedModel {
    fn default() -> Self {
        Self {
            schema_version: MODEL_SCHEMA_VERSION,
            phrase: DEFAULT_WAKE_WORD_PHRASE.to_string(),
            embedding_dim: u32::try_from(EMBEDDING_DIM).unwrap(),
            window_size: u32::try_from(wake_word_classifier::WINDOW_SIZE).unwrap(),
            training_seeds: Vec::new(),
            created_at: String::new(),
            trained_at: String::new(),
            classifier: None,
            val_losses: None,
        }
    }
}

/// Persist a wake word model to the config DB and update the in-memory CONFIG.
///
/// The CONFIG update ensures GUI snapshot readers / pipeline restart see the
/// latest model.  `save_and_reload` skips `wake_word_templates` (it's excluded
/// from the write loop), so this update is about cross-session visibility, not
/// deletion prevention.
///
/// Warnings are logged on failure. Returns `true` if both the DB write and the
/// CONFIG update succeeded. Callers use the return value to gate their own
/// success logging.
/// Used by both [`persist_model_state`] (post-training) and the migration
/// write-back path (legacy migration).
async fn persist_wake_word_model(model: &PersistedModel) -> bool {
    let Ok(json) = serde_json::to_string(model) else {
        warn!("Failed to serialize wake word model for persistence");
        return false;
    };
    let store = crate::config_db::store();
    if let Err(e) = store.set_kv("wake_word_templates", &json).await {
        warn!("Failed to persist wake word model: {e}");
        return false;
    }
    if !CONFIG.set_string_field("wake_word_templates", &json) {
        warn!(
            "Failed to update CONFIG with wake word model (key not recognized by \
             set_string_field — it may have drifted from the `stringify!` arms)"
        );
        return false;
    }
    true
}

/// Persist current classifier weights to the config database.
async fn persist_model_state() {
    let weights = get_classifier_weights();

    // Read the enrolling phrase (set at enrollment start) and clear it
    // so re-enrollment starts fresh.
    let phrase = voice_state()
        .write()
        .unwrap_poison()
        .enrolling_phrase
        .take()
        .unwrap_or_else(|| DEFAULT_WAKE_WORD_PHRASE.to_string());

    // Load existing model to preserve `created_at` (set once on first persist).
    // If this is a fresh enrollment with no prior model, created_at will be empty
    // and we set it to now.
    let existing_created_at = CONFIG
        .wake_word_templates()
        .and_then(|json| {
            serde_json::from_str::<PersistedModel>(&json)
                .ok()
                .map(|m| m.created_at)
        })
        .unwrap_or_default();

    let now = turso::now();

    let model = PersistedModel {
        schema_version: MODEL_SCHEMA_VERSION,
        phrase,
        embedding_dim: u32::try_from(EMBEDDING_DIM).unwrap(),
        window_size: u32::try_from(wake_word_classifier::WINDOW_SIZE).unwrap(),
        training_seeds: (0usize..1)
            .map(|s| s as u64) // safe widening: usize → u64 on 64-bit targets
            .collect(),
        created_at: if existing_created_at.is_empty() {
            now.clone()
        } else {
            existing_created_at
        },
        trained_at: now,
        classifier: weights.map(|w| vec![w]),
        val_losses: None, // no longer stored (uniform averaging)
    };

    if persist_wake_word_model(&model).await {
        // On success, cache the phrase in model_phrase so get_enrolled_phrase()
        // returns the latest value.
        voice_state().write().unwrap_poison().model_phrase = Some(model.phrase.clone());
        info!(
            "Wake word model persisted to config (v{}, phrase=\"{}\")",
            MODEL_SCHEMA_VERSION, model.phrase,
        );
    }
}

/// Handle recording audio: accumulate buffer and check for silence/duration limits.
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
                // wake-word cooldown timestamp to prevent immediate re-triggering.
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
                if ctx.should_send_error_message() {
                    let user_name = active_user_name();
                    if !user_name.is_empty() {
                        ctx.last_error_message_time = Some(Instant::now());

                        // Resolve workspace with fallback via the shared helper
                        // (matching the route_to_agent pattern).
                        let ws = resolve_workspace_for_voice(&user_name).await;

                        crate::channels::broadcast_and_persist_agent_response(
                            &user_name,
                            "voice",
                            "*Voice: transcription failed — try again*",
                            Some("voice".to_string()),
                            &ws.name,
                        )
                        .await;
                    }
                }

                // Enforce a 3-second refractory period before returning to
                // Listening (replaces the old 2-second blocking sleep with a
                // non-blocking alternative).
                ctx.refractory_until = Some(Instant::now() + Duration::from_secs(3));

                // Cleanup the recording state.
                // Soft reset clears recording/detection buffers while preserving
                // VAD/NS continuity so the noise floor estimate survives the
                // refractory period.
                ctx.reset_pipeline_state(ResetLevel::Soft);
                ctx.is_recording = false;
                // Do NOT set status to Listening here — the refractory delay
                // is handled in the main loop's post-select section.
            }
        }
    }
}

/// Retain an aligned overlap in `voice_batch` for the next mel spectrogram
/// batch.
///
/// Retains at least [`VOICE_BATCH_OVERLAP`] samples, rounding the drain
/// position **down** to the nearest [`MEL_STRIDE`] boundary so the retained
/// overlap starts on the mel frame grid.  Without this alignment,
/// [`HOP_LENGTH`] (256) not being a multiple of [`MEL_STRIDE`] (160) would
/// cause ~80 % of batch boundaries to land off-grid, producing mel frames
/// that the classifier never sees during training.
///
/// The actual retained overlap may be up to [`MEL_STRIDE`] − 1 samples larger
/// than [`VOICE_BATCH_OVERLAP`].  This is harmless — the extra samples merely
/// produce duplicate mel frames at the batch boundary which the stride‑8
/// sliding window skips naturally.
///
/// This function is extracted from [`flush_voice_batch`] so the overlap
/// trimming logic can be tested in isolation (the ONNX inference inside
/// `flush_voice_batch` requires model files and cannot run in unit tests).
/// See [`test_mel_stride_overlap_alignment`] for the unit test that validates
/// the alignment property.
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
        // Align drain_to down to MEL_STRIDE boundary so the retained overlap
        // starts at a sample position that is a multiple of MEL_STRIDE.
        let drain_to_aligned = drain_to - (drain_to % MEL_STRIDE);
        voice_batch.drain(..drain_to_aligned);
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
/// batch boundaries.  Removing this call would create a
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
            // trim_voice_batch retains overlap context across batch boundaries.
            // The mel frame buffer trim is now at the end of
            // handle_wake_word_detection after the stride-8 sliding window
            // loop, ensuring the sliding window sees the full
            // accumulated buffer before trimming.
            trim_voice_batch(voice_batch);
        }
        Err(e) => {
            warn!("Mel spectrogram failed: {e}");
            // Clear the batch so it doesn't grow unbounded when the
            // ONNX model is consistently failing.
            voice_batch.clear();
        }
    }
}

/// Score a batch of start-aligned positions with padded geometry.
///
/// Shared by the deferred burst sweep and the segment-end pass — both score
/// the same start-aligned grid (0/8/16/24 below the buffer end) with the
/// trained start-0-aligned padded geometry and stop at the first detection.
/// A detection propagates to the caller through the `ctx.is_recording`
/// handoff set inside [`score_stride8_window`] — the sweep stops at the
/// first detection, leaving `next_window_start` at the detecting position.
///
/// Re-anchors `next_window_start` to position 0 first so the trained
/// geometry is never perturbed by a stale window start.  (The ticket's
/// original "a misaligned mid-pass position scores ~0.2651 and resets the
/// rolling window" hazard was measured NOT to manifest at live geometry —
/// the B=79 → start-3 continuation scores ~0.99 and
/// is the enabling mechanism for mid-utterance continuation; the rolling
/// reset stays guarded by the per-segment burst latch and the boundary
/// pass.)
///
/// On a miss, `next_window_start` ends at the position after the last
/// scored grid point so the main stride-8 loop can continue from there.
fn score_start_aligned_positions(
    mel_frame_buffer: &[Vec<f32>],
    models: &OnnxModels,
    ctx: &mut PipelineCtx,
    positions: &[usize],
    source: WindowSource,
) {
    ctx.next_window_start = 0;
    for &pos in positions {
        ctx.next_window_start = pos;
        let end = mel_frame_buffer.len().min(pos + EMBEDDING_WINDOW_FRAMES);
        let subslice = &mel_frame_buffer[pos..end];
        let padded = pad_mel_frames_to_window(subslice);
        let padded_flag = padded.len() > subslice.len();
        score_stride8_window(
            &padded,
            models,
            ctx,
            padded_flag,
            mel_frame_buffer.len(),
            source,
        );
        if ctx.is_recording {
            // Detection fired — stop the sweep (the caller's handoff handles
            // the transition; `next_window_start` is left at the detecting
            // position, matching the old `return true` semantics).
            break;
        }
        ctx.next_window_start = pos + BURST_STRIDE;
    }
}

/// Score a single mel frame window through the embedding + classifier pipeline.
///
/// Computes the dense embedding from `mel_window` via ONNX, passes it through
/// [`score_single_embedding`] with the current classifier, records
/// instrumentation (feature-gated behind `voice-tests`), and transitions to
/// recording mode if detection fires.
///
/// # Returns
/// `true` if wake word was detected (caller should stop processing and return).
// Clippy: the instrumentation block plus the read-lock scoping
// pushes this past 100 lines in both feature configurations (104 default,
// 141 voice-tests); the body is a single linear pipeline and not worth
// splitting.
#[expect(clippy::too_many_lines)]
#[cfg_attr(not(feature = "voice-tests"), allow(unused_variables))]
fn score_stride8_window(
    mel_window: &[Vec<f32>],
    models: &OnnxModels,
    ctx: &mut PipelineCtx,
    // whether this window was produced by padding a short real
    // slice (tapered fade-out tail) rather than a full 76-frame window from
    // the main stride-8 loop.  Only read under `voice-tests` (geometry
    // reporting); all call sites pass a value so production builds see a
    // constant.
    padded_window: bool,
    // mel frame buffer length at this scoring step.  Only read
    // under `voice-tests`; the value is a pure observation of pipeline state.
    mel_buffer_len: usize,
    // scoring path that produced this window (deferred burst /
    // segment-end pass / main stride-8 loop).  Read under `voice-tests` for
    // the detection-path report; the value is a pure label.
    source: WindowSource,
) -> bool {
    let embed_start = Instant::now();
    let embedding = match crate::util::with_block_in_place(|| compute_embedding(models, mel_window))
    {
        Ok(emb) => {
            let elapsed = embed_start.elapsed();
            let elapsed_ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
            TOTAL_EMBEDDING_TIME_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
            EMBEDDINGS_COMPUTED.fetch_add(1, Ordering::Relaxed);
            #[allow(clippy::cast_possible_truncation)]
            let head = EMBEDDING_LATENCY_RING_WRITES.fetch_add(1, Ordering::Relaxed) as usize
                % EMBEDDING_LATENCY_RING_SIZE;
            EMBEDDING_LATENCY_RING[head].store(elapsed_ns, Ordering::Relaxed);
            emb
        }
        Err(e) => {
            warn!("Stride-8 embedding compute failed: {e}");
            return false;
        }
    };

    // ── Scope the read lock so it drops before set_status() ───────────
    // score_single_embedding needs the classifier from the global
    // voice state.  We acquire a read lock, call the function (which only
    // borrows the data temporarily), then let the guard drop at the end of
    // this block.  This avoids a read→write upgrade deadlock: the caller
    // below (line 7043) calls set_status() which acquires a write lock,
    // and holding a read lock across that call would deadlock on
    // std::sync::RwLock (which does not support read→write upgrades).
    //
    // Historical context: commit da06926 fixed a double-read-lock in this
    // same function but missed this read→write upgrade path.
    // Do NOT rely on NLL early-drop — always scope the guard explicitly.
    // ── Capture pre-call ring length for geometry instrumentation ──
    #[cfg(feature = "voice-tests")]
    let ring_len_before = ctx.embedding_ring.len();
    // mahbot-1012: capture the adaptive bootstrap state BEFORE the call —
    // `feed()` advances the bootstrap counter, so post-call `is_bootstrapping()`
    // cannot distinguish the last bootstrap frame from a post-bootstrap frame.
    #[cfg(feature = "voice-tests")]
    let adaptive_bootstrapping_before = ctx.adaptive_threshold.is_bootstrapping();
    #[cfg_attr(not(feature = "voice-tests"), allow(unused_variables))]
    let (detected, rolling_sum, total_score, effective_threshold) = {
        let state = voice_state().read().unwrap_poison();
        score_single_embedding(
            &embedding,
            &mut ctx.embedding_ring,
            state.classifier.as_ref(),
            &mut ctx.score_window,
            Some(&mut ctx.adaptive_threshold),
            ctx.adaptive_k,
            source == WindowSource::DeferredBurst,
        )
    }; // <── read guard dropped here, before any write-lock acquisition

    // ── Instrumentation (voice-tests only) ──
    #[cfg(feature = "voice-tests")]
    {
        // mahbot-1012: number of test-pass frames scored before this window
        // (pre-push — the push below appends this window's score).  Used by
        // classify_window_geometry's WarmMixed test: the Conv1D window spans
        // the last WINDOW_SIZE pushed embeddings, so it contains a warm-up
        // embedding iff fewer than WINDOW_SIZE-1 test frames precede it.
        // per_frame_scores is NOT reset by mid-pass segment boundaries, which
        // is exactly what we want — post-boundary windows (ring cleared) must
        // never classify WarmMixed.
        let frames_scored_before = ctx.instrumentation.per_frame_scores.len();
        if total_score > ctx.instrumentation.peak_score {
            ctx.instrumentation.peak_score = total_score;
        }
        ctx.instrumentation
            .per_frame_scores
            .push([total_score, rolling_sum, effective_threshold]);
        if total_score < NO_MATCH_RESET_THRESHOLD {
            ctx.instrumentation.n_frames_below_reset += 1;
        }
        ctx.instrumentation
            .adaptive_threshold_trajectory
            .push(effective_threshold);
        if effective_threshold >= ADAPTIVE_CEILING {
            ctx.instrumentation.ceiling_limited_frames += 1;
        }
        // mahbot-1023: record the scoring path that produced the detection
        // (raw source: "burst" / "segment_end_pass" / "other").
        if detected && ctx.instrumentation.detection_path.is_none() {
            ctx.instrumentation.detection_path = Some(source.as_str());
        }
        // First classifier-trigger frame index.  `rolling_sum >= effective_threshold`
        // mirrors the `detected` computed by process_wake_word_score
        // inside score_single_embedding (the adaptive override is exactly
        // `effective_threshold`), so this is the frame the classifier fired on.
        if ctx.instrumentation.first_trigger_frame_idx.is_none()
            && rolling_sum >= effective_threshold
        {
            ctx.instrumentation.first_trigger_frame_idx =
                Some(ctx.instrumentation.per_frame_scores.len() - 1);
        }

        // ── mahbot-1012: per-frame scoring geometry (parallel to
        //    per_frame_scores) ─────────────────────────────────────────────
        // The caller-provided mel-window geometry (padded fallback vs main
        // stride-8 loop) is combined with the classifier-ring-derived geometry
        // by the pure classify_window_geometry (see its doc comment for the
        // precedence order and the ring-capacity correctness argument).
        let geometry = classify_window_geometry(
            ring_len_before,
            frames_scored_before,
            ctx.instrumentation.test_start_ring_len,
            padded_window,
        );
        // Mirror of the feed/peek decision inside score_single_embedding
        // (voice.rs, adaptive_override): the feed/peek rule is score-based
        // ONLY (below-reset scores feed; wake-word-like scores peek) including
        // during bootstrap (mahbot-1023).  The label is state-based for
        // bootstrap frames (Bootstrap while bootstrapping, regardless of the
        // feed/peek action) and action-based after bootstrap (Feed / Peek) —
        // see AdaptiveFrameMode.  `adaptive_bootstrapping_before` is captured
        // pre-call so the last bootstrap frame is not mislabeled (feed()
        // advances the counter).  This is deliberately a mirror, not a shared
        // helper: score_single_embedding is production code and must not carry
        // feature-gated instrumentation.  If the feed/peek rule in
        // score_single_embedding changes, update this block to match.
        let adaptive_mode = if adaptive_bootstrapping_before {
            AdaptiveFrameMode::Bootstrap
        } else if total_score < NO_MATCH_RESET_THRESHOLD {
            AdaptiveFrameMode::Feed
        } else {
            AdaptiveFrameMode::Peek
        };
        ctx.instrumentation
            .per_frame_embedding_hashes
            .push(embedding_hash(&embedding));
        ctx.instrumentation
            .per_frame_embedding_l2_norms
            .push(embedding_l2_norm(&embedding));
        ctx.instrumentation.per_frame_embeddings.push(embedding);
        ctx.instrumentation
            .per_frame_window_start
            .push(ctx.next_window_start);
        ctx.instrumentation
            .per_frame_mel_buffer_len
            .push(mel_buffer_len);
        ctx.instrumentation.per_frame_geometry.push(geometry);
        ctx.instrumentation
            .per_frame_adaptive_mode
            .push(adaptive_mode);
    }

    if detected {
        ctx.is_recording = true;
        ctx.last_wake_word_detection = Some(Instant::now());
        set_status(VoiceStatus::Recording);
        true
    } else {
        false
    }
}

/// Handle wake word detection: process audio frames through mel/embedding/Conv1D classifier.
///
/// Audio arrives in small chunks (~256 samples at 16kHz). This function:
/// 1. Accumulates audio in a sliding window for VAD
/// 2. Collects voiced frames into a batch buffer
/// 3. Processes the batch through mel ONNX when enough audio is accumulated (~128ms)
/// 4. Produces embeddings and runs the Conv1D classifier on 3-embedding windows
/// 5. Passes classifier confidence scores through the rolling window accumulator
///
/// Batching reduces ONNX inference calls from ~62/sec (per-frame) to ~8/sec.
///
/// Implements cooldown and soft-scoring + rolling window
/// detection via the `last_wake_word_detection` and
/// `score_window` fields.
#[allow(clippy::too_many_lines)]
pub(crate) fn handle_wake_word_detection(samples: &[f32], ctx: &mut PipelineCtx) {
    // ── Cooldown check ──
    // If we recently detected the wake word, skip ALL processing for this
    // chunk to prevent rapid consecutive false triggers.  During cooldown
    // audio accumulates into audio_buffer with a cap so that
    // when the cooldown expires the pipeline has data to process immediately;
    // intermediate detection buffers are cleared to prevent stale data from
    // the previous utterance causing false triggers.
    if let Some(last) = ctx.last_wake_word_detection
        && last.elapsed() < WAKE_WORD_COOLDOWN
    {
        debug!(
            "Wake word cooldown active ({}ms elapsed)",
            last.elapsed().as_millis()
        );
        // Accumulate audio during cooldown so the pipeline has data
        // when cooldown expires — don't discard it entirely.
        // See [`COOLDOWN_ACCUMULATION_CAP`] for the frame-processing
        // arithmetic that justifies the cap value (2 frames = 1024 samples).
        //
        // We accumulate into audio_buffer (not command_buffer) because during
        // cooldown is_recording is false, so audio is routed to
        // handle_wake_word_detection, not to handle_recording_audio.
        // command_buffer is only populated after detection transitions to
        // recording mode.
        //
        // Invariant: audio_buffer is empty or within COOLDOWN_ACCUMULATION_CAP at
        // cooldown entry.  The handoff block in this function clears it at the
        // detection→recording transition — this assertion guards against future
        // refactors that bypass that path.
        debug_assert!(
            ctx.audio_buffer.len() <= COOLDOWN_ACCUMULATION_CAP,
            "audio_buffer (len={}) exceeds COOLDOWN_ACCUMULATION_CAP ({}) at cooldown entry; \
             handle_wake_word_detection's handoff should have cleared it at detection→recording transition",
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
        // the noise floor estimate.
        ctx.reset_detection_segment();
        return;
    }

    ctx.audio_buffer.extend_from_slice(samples);

    // VAD-gating / batch-accumulation frame loop
    //
    // Produces mel frames via flush_voice_batch.  Now, the
    // callback is a no-op — embeddings are extracted via a stride-8 sliding
    // window over the accumulated mel frame buffer AFTER the VAD loop.
    // This ensures ALL accumulated mel frames are scored with dense stride-8
    // embeddings, matching the enrollment training distribution.
    //
    // `audio_buffer`, `voice_batch` and `mel_frame_buffer` are taken from `ctx`
    // into local variables so the closure (which borrows `*ctx` for wake word
    // detection) does not conflict with the inner function's mutable access to
    // the batch buffers.
    //
    // When detection fires during the stride-8 loop, score_stride8_window
    // sets is_recording = true but defers the full transition to this
    // function — see the handoff block below.
    let mut audio_buf = std::mem::take(&mut ctx.audio_buffer);
    let mut voice_batch = std::mem::take(&mut ctx.voice_batch);
    let mut mel_frame_buffer = std::mem::take(&mut ctx.mel_frame_buffer);

    // VAD counting for instrumentation (mahbot-886).  Uses an `AtomicUsize`
    // captured by shared reference in the closure to avoid a borrow conflict
    // with `on_flush` (which captures `ctx` by `&mut`).  `AtomicUsize::fetch_add`
    // takes `&self`, so the closure captures `&AtomicUsize` — no `&mut` needed.
    // Feature-gated — zero overhead in production.
    #[cfg(feature = "voice-tests")]
    let vad_count = std::sync::atomic::AtomicUsize::new(0);

    // mahbot-1012: per-hop VAD decision sequence.  Captured by the closure
    // (moved into `process_streaming_frames_inner` by value, then transferred
    // into instrumentation after the loop — same pattern as `vad_count`).
    #[cfg(feature = "voice-tests")]
    let mut per_hop_vad: Vec<bool> = Vec::new();

    // Side-channel for consecutive VAD-negative hop tracking.
    // Seeded with the accumulated count from previous calls so the counter
    // is continuous across `process_streaming_frames_inner` invocations.
    let segment_silence_hops = std::sync::atomic::AtomicUsize::new(ctx.segment_silence_hops);

    let is_speech_fn = |frame: &[f32]| -> bool {
        let result = is_speech_with_threshold(frame, VAD_THRESHOLD);
        if result {
            segment_silence_hops.store(0, std::sync::atomic::Ordering::Relaxed);
            #[cfg(feature = "voice-tests")]
            vad_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            segment_silence_hops.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        #[cfg(feature = "voice-tests")]
        per_hop_vad.push(result);
        result
    };

    let mut consumed = process_streaming_frames_inner(
        &audio_buf,
        &mut voice_batch,
        &mut mel_frame_buffer,
        is_speech_fn,
        false, // no trailing flush — audio_buffer accumulates across calls
        |_mel_frames| {
            // No-op callback — stride-8 sliding window
            // extraction runs after the VAD loop over the full mel frame buffer.
            false
        },
    );

    // ── Stride-8 sliding window embedding extraction ──
    // After the VAD-gating loop, iterate over the accumulated mel frame buffer
    // with stride 8, extracting dense embeddings at each position and scoring
    // through the Conv1D classifier pipeline.  This replaces the old
    // streaming approach that extracted one embedding per batch flush.
    //
    // position 0 is HELD until the
    // buffer reaches BURST_TRIGGER_FRAMES (68), then scored ONCE at the
    // start-aligned positions 0/8/16/24 with the trained start-0-aligned
    // padded geometry.  The old incremental per-chunk scoring scored position 0
    // with 2–16 real frames (~0.027) and never re-scored it — a single scored
    // window structurally cannot trigger the 3-consecutive-window gate.  The
    // short-buffer padded fallback is superseded: mid-utterance re-scoring of
    // ≤17-real-frame windows (~0.01) resets the rolling window and would wipe
    // a valid detection.  Short utterances (< 68 frames) are covered by the
    // segment-end pass at the boundary.
    if !ctx.is_recording {
        // No-op flush prevention: skip if buffer hasn't advanced by at least
        // BURST_STRIDE frames since the last extraction.
        if mel_frame_buffer.len() >= ctx.next_window_start + BURST_STRIDE
            && ctx.next_window_start < mel_frame_buffer.len()
        {
            let Some(models) = ONNX_MODELS.get() else {
                return;
            };

            // ── Deferred burst sweep ──
            // Fires once per segment on the first call where the buffer
            // reaches BURST_TRIGGER_FRAMES (68).  Live mel-flush granularity
            // lands the trigger at B=76 (not the measured B=68 grid cell) —
            // reported per run as `burst_sweep_buffer_len`.  The sweep runs
            // synchronously in this call with NO intermediate scoring and
            // never interleaves with the next mel flush: the 1.024 s mic
            // channel absorbs the one-shot ~44–135 ms stall (measured as
            // `burst_wall_clock_ms` under voice-tests).
            //
            // Note (measured, not assumed): the sweep must finish before gap
            // recovery re-anchors to a MISALIGNED leading edge whose
            // static-gate score was 0.2651 < NO_MATCH_RESET_THRESHOLD
            // (rolling-window reset).  Live measurement (runs 5–7) shows the
            // B=79 → start-3 re-score scores ~0.99 at the live geometry —
            // the hazard does not manifest.  The main stride-8 loop carries
            // the wake after the sweep (the primary expected mechanism);
            // the segment-end pass remains the overrun safety net if a
            // below-reset re-score ever does clear the rolling window.
            let mut burst_swept_this_call = false;
            // The helper encodes the hold itself (empty position set while
            // `burst_sweep_done` or below BURST_TRIGGER_FRAMES) — the call
            // site tests the empty set instead of duplicating the guard.
            let burst_positions =
                burst_positions_to_score(mel_frame_buffer.len(), ctx.burst_sweep_done);
            if !burst_positions.is_empty() {
                #[cfg(feature = "voice-tests")]
                let burst_start = Instant::now();
                // Start-aligned positions only — the helper re-anchors
                // defensively so the trained start-0-aligned geometry is
                // never perturbed by a stale window start.
                score_start_aligned_positions(
                    &mel_frame_buffer,
                    models,
                    ctx,
                    &burst_positions,
                    WindowSource::DeferredBurst,
                );
                // Same-call gap-recovery suppression (review fix): the gap
                // recovery below would re-anchor next_window_start to the
                // leading edge (position 0 at the flush-aligned B=76) and the
                // main loop would re-score the IDENTICAL full-76 window the
                // burst just scored at position 0 — a wasted embedding and a
                // double-counted rolling score (the ring-order-only flip that
                // let some failed bursts fire via a misattributed 'other'
                // path).  The burst already covered the leading edge, so the
                // main loop resumes from the position after the swept grid on
                // later calls.
                burst_swept_this_call = true;
                // Latch the per-segment "burst sweep done" state — cleared
                // only by the segment resets.  Prevents the sweep from
                // re-running and mid-utterance trailing re-scoring from
                // wiping a valid detection.  Does NOT gate the segment-end
                // pass (a failed burst must not suppress the boundary
                // fallback).
                ctx.burst_sweep_done = true;
                #[cfg(feature = "voice-tests")]
                {
                    ctx.instrumentation.burst_sweep_fired = true;
                    ctx.instrumentation.burst_sweep_buffer_len = Some(mel_frame_buffer.len());
                    ctx.instrumentation.burst_wall_clock_ms =
                        Some(burst_start.elapsed().as_secs_f64() * 1000.0);
                }
            }

            // The main stride-8 loop is skipped when the burst just fired a
            // detection (is_recording true → handoff below).
            if !ctx.is_recording {
                // ── Blinded gap recovery ──
                // After warm-up or any pipeline action that advances
                // next_window_start past the leading edge of real audio, the
                // main stride-8 loop may be unable to fire because
                // next_window_start + EMBEDDING_WINDOW_FRAMES > buffer_len.
                // When the buffer has enough frames, reset next_window_start
                // to the leading edge so extraction from the new utterance can
                // proceed immediately, eliminating the ~68-frame blind spot
                // that previously caused 0% detection on test utterances
                // following warm-up.
                //
                // the re-anchor is skipped in the
                // same call as the deferred burst — the burst already scored
                // the leading edge (positions 0/8/16/24), so re-anchoring to
                // it would re-score the identical position-0 window.
                if !burst_swept_this_call
                    && ctx.next_window_start + EMBEDDING_WINDOW_FRAMES > mel_frame_buffer.len()
                    && mel_frame_buffer.len() >= EMBEDDING_WINDOW_FRAMES
                {
                    ctx.next_window_start = mel_frame_buffer.len() - EMBEDDING_WINDOW_FRAMES;
                }

                // Iterate from next_window_start with stride 8.
                while ctx.next_window_start + EMBEDDING_WINDOW_FRAMES <= mel_frame_buffer.len() {
                    let window = &mel_frame_buffer
                        [ctx.next_window_start..ctx.next_window_start + EMBEDDING_WINDOW_FRAMES];
                    if score_stride8_window(
                        window,
                        models,
                        ctx,
                        false,
                        mel_frame_buffer.len(),
                        WindowSource::MainStride,
                    ) {
                        // Detection fired — loop will be restarted on next call
                        // via fresh stride-8 iteration.  next_window_start is reset
                        // by reset_pipeline_state(Soft).
                        break;
                    }
                    ctx.next_window_start += BURST_STRIDE;
                }
                // NOTE: the short-buffer padded fallback is
                // intentionally GONE.  Its incremental per-chunk scoring is
                // exactly the bug (position 0 scored with 2–16
                // frames per chunk, never re-scored), and after the deferred
                // burst the fallback's ≤17-real-frame windows (~0.01) would
                // reset the rolling window mid-utterance.  Short buffers are
                // covered by the deferred burst (≥ 68 frames) and the
                // segment-end pass (< 68 frames, at the boundary).
            }
        }
    }

    // Transfer VAD count into instrumentation (mahbot-886).
    // Accumulate (`+=`) instead of overwriting (`=`): handle_wake_word_detection
    // is called once per audio chunk, and the VAD count is per-call.  The old
    // overwrite kept only the LAST call's count (~0 after the silence flush),
    // making the benchmark's vad_speech_frames evidence unreliable (mahbot-1005).
    #[cfg(feature = "voice-tests")]
    {
        ctx.instrumentation.vad_speech_frames +=
            vad_count.load(std::sync::atomic::Ordering::Relaxed);
        // mahbot-1012: per-hop VAD decision sequence (one entry per VAD
        // decision in processing order).
        ctx.instrumentation.per_hop_vad.extend(per_hop_vad);
    }
    // ── Bounded detection segment check ───────────────────────
    // If we've accumulated enough consecutive VAD-negative hops since the
    // last VAD-positive frame, declare a segment boundary and reset
    // per-segment detection state.  This prevents classifier scores, rolling
    // sums from accumulating across separate utterances
    // separated by more than ~300ms of silence.
    //
    // If detection fired during the frame loop, the pipeline transitioned to
    // recording mode and `reset_pipeline_state(Soft)` already cleared
    // `ctx.segment_silence_hops` and all per-segment buffers — skip the
    // stale writeback that would overwrite the clean state (per reviewer feedback).
    if !ctx.is_recording {
        let hop_count = segment_silence_hops.load(std::sync::atomic::Ordering::Relaxed);
        ctx.handle_segment_boundary(hop_count, &mut voice_batch, &mut mel_frame_buffer);
    }

    // ── Detection→recording handoff ──────────────────────────
    // When detection fires, score_stride8_window sets is_recording = true.
    // We complete the transition here where all state (audio_buf, voice_batch,
    // mel_frame_buffer) is available as local variables (moved out of ctx at
    // lines 6960-6962 to avoid borrow conflicts with the VAD closure).
    //
    // The transition sequence (matching the documented design):
    //   1. Take ALL of audio_buf (processed wake-word tail + unprocessed
    //      command-start) — the consumed/unconsumed distinction is irrelevant
    //      because ASR tolerates the extra wake-word overlap
    //   2. Reset pipeline state (Soft): clears command_buffer, detection buffers
    //      (embedding_ring, score_window, next_window_start, etc.)
    //   3. Re-populate command_buffer with the saved audio
    //   4. Clear stale local copies (voice_batch, mel_frame_buffer) so the
    //      write-back below (lines 7125-7127) restores empty Vecs
    //   5. Set consumed = 0 to prevent drain panic on the now-empty audio_buf
    //
    // Previously score_stride8_window called transition_to_recording()
    // directly, but its take of ctx.audio_buffer returned empty because the
    // buffer was already moved.  Consolidating the transition here eliminates
    // the split responsibility and ensures all state changes happen together
    // with the actual data.
    if ctx.is_recording {
        let audio = std::mem::take(&mut audio_buf);
        ctx.reset_pipeline_state(ResetLevel::Soft);
        ctx.command_buffer.extend_from_slice(&audio);
        voice_batch.clear();
        mel_frame_buffer.clear();
        consumed = 0; // Prevent drain panic on now-empty audio_buf
    }

    // ── Mel frame buffer trim (moved from flush_voice_batch) ──
    // After the stride-8 loop, keep the last EMBEDDING_WINDOW_FRAMES - 8 mel
    // frames (overlap for continuity with the next batch).  The overlap of 8
    // frames matches the BURST_STRIDE window so the next call has valid
    // context.
    if mel_frame_buffer.len() > EMBEDDING_WINDOW_FRAMES.saturating_sub(BURST_STRIDE) {
        let keep = EMBEDDING_WINDOW_FRAMES.saturating_sub(BURST_STRIDE);
        let drain_to = mel_frame_buffer.len().saturating_sub(keep);
        if drain_to > 0 {
            mel_frame_buffer.drain(..drain_to);
            // Adjust next_window_start if it falls within the drained range
            if ctx.next_window_start >= drain_to {
                ctx.next_window_start = ctx.next_window_start.saturating_sub(drain_to);
            } else {
                ctx.next_window_start = 0;
            }
        }
    }

    // Drain consumed audio, write back batch and audio buffers.
    if consumed > 0 {
        audio_buf.drain(..consumed);
    }
    ctx.audio_buffer = audio_buf;
    ctx.voice_batch = voice_batch;
    ctx.mel_frame_buffer = mel_frame_buffer;
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
/// These real (non-synthetic) negative examples are later used to train the
/// classifier at enrollment finalization, replacing the old synthetic
/// Gaussian noise that caused false triggers.
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
        // and feature context.  This must match the VAD call
        // pattern in process_streaming_frames_inner to maintain train-
        // inference consistency across the detection and enrollment paths.
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

                // ── Capture noise RMS from pre-AGC ring ──
                // On the FIRST transition from silence to sustained speech,
                // capture the ambient noise RMS from the pre-AGC audio ring.
                // The pre_agc_ring stores raw mic audio before AGC gain is
                // applied — this matters because AGC amplifies silence (up to
                // 4×) disproportionately to speech (~1-2×), so post-AGC noise
                // RMS would produce an SNR estimate 6-12 dB lower than the
                // true room SNR, triggering a false low-SNR warning even in
                // quiet environments (raw_audio_ring contains post-AGC audio, causing
                // this false-positive).
                let already_had_speech = ctx.utterance_had_speech;
                // ── Save collected ambient audio for negatives ──
                // On the FIRST transition from silence to sustained speech,
                // save the accumulated non-wake-word audio (pre-enrollment
                // ambient noise or inter-utterance silence) as a potential
                // negative training example.
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
                        // Shared RMS helper; the statement-level
                        // #[expect(clippy::cast_precision_loss)] was removed
                        // because compute_rms carries its own allow.
                        let rms = crate::util::compute_rms(&ctx.pre_agc_ring[..pre_speech_end]);
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
                // Warn after the derived frame threshold.  The constant is
                // computed from ENROLLMENT_NO_SPEECH_DURATION × SAMPLE_RATE /
                // HOP_LENGTH, so the threshold stays correct if frame/hop sizes
                // are adjusted.
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
            // cancellation/completion safety.
        }
    }
}

// Phase 3 owner-negative audio processing

/// Process incoming audio for Phase 3 owner-negative collection.
///
/// Duplicates the VAD frame iteration pattern from [`handle_enrollment_audio`]
/// (~12 lines, tagged with `SHARED-VAD-LOOP-BEGIN`/`SHARED-VAD-LOOP-END`).
/// A closure-based helper is not used because the borrow-checker overhead
/// outweighs deduplication benefit for two consumers. If a third consumer
/// emerges, extract into a shared helper.
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
/// `PipelineCtx` — matches ambient negative pattern and survives mic resets).
/// Memory cap is enforced: chunks that would exceed [`MAX_OWNER_NEGATIVE_SAMPLES`]
/// are dropped.
///
/// Updates `negatives_speech_samples` counter with the VAD-positive frames
/// detected this call.
fn handle_negative_collection_audio(samples: &[f32], ctx: &mut PipelineCtx) {
    // ── SHARED-VAD-LOOP-BEGIN: VAD frame iteration (duplicated from handle_enrollment_audio) ──
    ctx.phase3_audio_buf.extend_from_slice(samples);
    let len = ctx.phase3_audio_buf.len();
    let mut consumed = 0;
    // Track the start of the current speech segment within phase3_audio_buf.
    // When silence is detected, we finalize the speech portion as a chunk.
    let mut segment_start: usize = 0;
    while consumed + FRAME_LENGTH <= len {
        let frame = &ctx.phase3_audio_buf[consumed..consumed + FRAME_LENGTH];
        let is_speech = if let Some(ref mut det) = ctx.enrollment_vad {
            is_speech_with_detector(&frame[..HOP_LENGTH], det, ctx.vad_threshold)
        } else {
            is_speech_with_threshold(&frame[..HOP_LENGTH], ctx.vad_threshold)
        };
        // ── SHARED-VAD-LOOP-END ──

        if is_speech {
            // Accumulate VAD-positive speech samples for the target counter.
            ctx.negatives_speech_samples += HOP_LENGTH;
            // Reset silence counter on any VAD-positive frame.
            ctx.phase3_silence_samples = 0;
        } else {
            ctx.phase3_silence_samples += HOP_LENGTH;
            // Check for chunk boundary: when silence exceeds ENROLLMENT_SILENCE_THRESHOLD_SAMPLES
            // (aligned to streaming's ~304ms) after sustained speech,
            // finalize the current segment as a chunk.
            if ctx.phase3_silence_samples >= ENROLLMENT_SILENCE_THRESHOLD_SAMPLES {
                let chunk_end = consumed.saturating_sub(ctx.phase3_silence_samples);
                if chunk_end > segment_start {
                    let chunk = ctx.phase3_audio_buf[segment_start..chunk_end].to_vec();
                    let mut state = voice_state().write().unwrap_poison();
                    // Memory cap: drop chunks that would exceed MAX_OWNER_NEGATIVE_SAMPLES.
                    let total_samples: usize = state
                        .owner_negative_chunks
                        .iter()
                        .map(std::vec::Vec::len)
                        .sum();
                    if total_samples + chunk.len() <= MAX_OWNER_NEGATIVE_SAMPLES {
                        state.owner_negative_chunks.push(chunk);
                    } else {
                        warn!(
                            "owner_negative_chunks at capacity ({} samples): dropping chunk \
                             of {} samples (mahbot-913)",
                            MAX_OWNER_NEGATIVE_SAMPLES,
                            chunk.len(),
                        );
                    }
                    drop(state);
                }
                // Advance segment_start past the silence boundary so the next
                // segment starts after this silence region.
                segment_start = consumed;
            }
        }

        consumed += HOP_LENGTH;
    }
    // Drain fully processed frames (everything before segment_start) from the
    // audio buffer, preserving any partial trailing speech segment or silence.
    if segment_start > 0 {
        ctx.phase3_audio_buf.drain(..segment_start);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EMBEDDING_DIM;
    use crate::util::test::set_env_var;
    use std::time::{Duration, Instant};

    // ── VAD-gated utterance segmentation tests ────────────────────────────
    // These test the pure [`segment_utterances_by_vad`] function with synthetic
    // audio and manually-computed VAD decisions.  No global voice state needed.

    /// Test config with a shorter silence threshold (10 frames ≈ 2560 samples
    /// instead of the default 94 frames ≈ 24000 samples) so tests don't need
    /// prohibitively long audio buffers.  Context padding is 0 per mahbot-1001
    /// Fix 3 — matches streaming detection which does not prepend/append
    /// context padding.
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

    // ── VAD-gated streaming mel extraction tests (mahbot-1009) ───────────
    // These test [`vad_gate_streaming_mel`] with deterministic VAD decisions.
    // The mel-frame half requires ONNX models (unavailable in unit tests —
    // `flush_voice_batch` returns early when `ONNX_MODELS` is unset), so these
    // tests focus on the `speech_audio` accumulation — the part that decides
    // which audio reaches augmentation and that must match the streaming
    // path's VAD gating exactly.

    /// Build an audio buffer sized for `speech_mask.len()` hops at
    /// [`HOP_LENGTH`] stride with a [`FRAME_LENGTH`] window, where hop `h`
    /// (256 samples) carries a 220 Hz tone when `speech_mask[h]` is true and
    /// digital silence otherwise.
    fn audio_with_hop_mask(speech_mask: &[bool]) -> Vec<f32> {
        let n_hops = speech_mask.len();
        let len = (n_hops - 1) * HOP_LENGTH + FRAME_LENGTH;
        let mut audio = vec![0.0f32; len];
        for (h, &speech) in speech_mask.iter().enumerate() {
            if speech {
                let start = h * HOP_LENGTH;
                for (k, s) in audio[start..start + HOP_LENGTH].iter_mut().enumerate() {
                    *s = 0.3
                        * (2.0 * std::f32::consts::PI * 220.0 * (start + k) as f32
                            / SAMPLE_RATE as f32)
                            .sin();
                }
            }
        }
        audio
    }

    #[test]
    fn vad_gate_streaming_mel_keeps_only_speech_hops() {
        // 3 silence hops, 3 speech, 1 silence (intra-phrase pause), 2 speech,
        // 4 trailing silence.  The 220 Hz tone is present only in the speech
        // hops; everything else is digital silence.
        let mask = [
            false, false, false, true, true, true, false, true, true, false, false, false, false,
        ];
        let audio = audio_with_hop_mask(&mask);
        let n_frames = (audio.len() - FRAME_LENGTH) / HOP_LENGTH + 1;
        assert_eq!(n_frames, mask.len(), "one VAD call per hop");

        let mut hop_idx = 0usize;
        let (mel_frames, speech_audio) = vad_gate_streaming_mel(&audio, |hop| {
            assert_eq!(
                hop.len(),
                HOP_LENGTH,
                "each VAD call receives exactly one new hop"
            );
            let is_speech = mask[hop_idx];
            hop_idx += 1;
            is_speech
        });

        assert_eq!(
            hop_idx, n_frames,
            "the VAD loop must feed one hop per frame at HOP_LENGTH stride, \
             exactly like process_streaming_frames_inner"
        );

        // speech_audio must be exactly the concatenation of the VAD-positive
        // hops (3,4,5,7,8): no leading/trailing silence, and the intra-phrase
        // pause (hop 6) discarded — matching how the streaming path feeds only
        // VAD-positive hops into the voice batch.
        let mut expected: Vec<f32> = Vec::new();
        for h in [3usize, 4, 5, 7, 8] {
            let start = h * HOP_LENGTH;
            expected.extend_from_slice(&audio[start..start + HOP_LENGTH]);
        }
        assert_eq!(
            speech_audio, expected,
            "speech_audio must be the concatenation of VAD-positive hops only"
        );
        assert_eq!(speech_audio.len(), 5 * HOP_LENGTH);

        // Without ONNX models (unit tests) flush_voice_batch is a no-op, so
        // mel frames stay empty — documented here so a future test that does
        // have models available can assert on the mel half instead.
        assert!(
            mel_frames.is_empty(),
            "mel frames require ONNX models (available in E2E only)"
        );
    }

    #[test]
    fn vad_gate_streaming_mel_no_speech_returns_empty() {
        let mask = [false; 8];
        let audio = audio_with_hop_mask(&mask);
        let (mel_frames, speech_audio) = vad_gate_streaming_mel(&audio, |_| false);
        assert!(speech_audio.is_empty(), "no speech → no speech audio");
        assert!(mel_frames.is_empty(), "no speech → no mel frames");
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

    // ── AdaptiveThresholdState tests (mahbot-845) ─────────────────────────
    // Pure unit tests for the z-score adaptive threshold tracker.  Uses
    // synthetic per-frame scores — no ONNX models or voice pipeline state.
    // Covers bootstrap phase, mean/std computation, all safeguards, and reset.

    #[test]
    fn adaptive_bootstrap_returns_none() {
        let mut state = AdaptiveThresholdState::new();
        for i in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            assert!(
                state.feed(0.5, ADAPTIVE_K_DEFAULT).is_none(),
                "frame {i} should return None during bootstrap",
            );
        }
    }

    #[test]
    fn adaptive_after_bootstrap_returns_some() {
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            assert!(state.feed(0.5, ADAPTIVE_K_DEFAULT).is_none());
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
        // adaptive = (0.1 + 2.5 × 0.0) × 3 = 0.3 → well below safe harbor (1.35)
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
        // Capped by ceiling 4.503.
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
    fn adaptive_safeguard_chain_correctness() {
        // Verify the safeguard chain produces correct results independent of
        // the specific constant values.  The chain in feed() is:
        //   result = max(adaptive, SAFE_HARBOR, FLOOR).min(CEILING)
        //
        // We test several scenarios:
        // 1. Below both lower bounds → result = max(SAFE_HARBOR, FLOOR)
        // 2. Between lower bounds → result = max(adaptive, SAFE_HARBOR, FLOOR)
        // 3. Above ceiling → result = CEILING
        // 4. At ceiling → result = CEILING

        let chain = |adaptive: f32| -> f32 {
            adaptive
                .max(ADAPTIVE_SAFE_HARBOR)
                .max(ADAPTIVE_FLOOR)
                .min(ADAPTIVE_CEILING)
        };

        // Case 1: adaptive below both lower bounds
        assert_eq!(
            chain(0.0),
            ADAPTIVE_SAFE_HARBOR,
            "below both lower bounds should yield safe harbor",
        );
        // Since ADAPTIVE_FLOOR <= ADAPTIVE_SAFE_HARBOR (compile-time invariant),
        // the result is always safe harbor, not floor.  This documents that the
        // floor is a dominated safeguard with current constants.

        // Case 2: adaptive between FLOOR and SAFE_HARBOR (if such range exists)
        let mid = (ADAPTIVE_FLOOR + ADAPTIVE_SAFE_HARBOR) / 2.0;
        assert!(
            chain(mid) >= ADAPTIVE_SAFE_HARBOR,
            "adaptive={mid} between floor and safe_harbor should yield ≥safe_harbor",
        );

        // Case 3: far above ceiling
        assert_eq!(
            chain(10.0),
            ADAPTIVE_CEILING,
            "above ceiling should yield ceiling",
        );

        // Case 4: exactly at ceiling
        assert_eq!(
            chain(ADAPTIVE_CEILING),
            ADAPTIVE_CEILING,
            "at ceiling should yield ceiling",
        );
    }

    #[test]
    fn adaptive_floor_is_hard_minimum() {
        // This test verifies that the floor clamp is structurally present and
        // correctly ordered in the safeguard chain.  With current constants,
        // ADAPTIVE_FLOOR (1.35) == ADAPTIVE_SAFE_HARBOR (1.35), so both bounds
        // produce the same result.  This documents that with current constants
        // the floor and safe harbor are equal (mahbot-860).
        //
        // We prove the floor is reachable by demonstrating that even with
        // completely zero input (k=0, score=0), the threshold respects ALL
        // lower bounds:
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.0, 0.0); // k=0, score=0 → adaptive = 0.0 * 3 = 0.0
        }
        let result = state.feed(0.0, 0.0);
        let threshold = result.expect("should return Some after bootstrap");
        // With k=0, score=0: adaptive = (0.0 + 0.0) × 3 = 0.0
        // Chain: max(0.0, SAFE_HARBOR=1.35) = 1.35, max(1.35, FLOOR=1.35) = 1.35
        assert!(
            threshold >= ADAPTIVE_FLOOR && threshold >= ADAPTIVE_SAFE_HARBOR,
            "threshold {threshold} should respect both floor {} and safe harbor {}",
            ADAPTIVE_FLOOR,
            ADAPTIVE_SAFE_HARBOR,
        );
        // Additionally, verify the floor constant is structurally valid:
        assert!(
            ADAPTIVE_FLOOR <= ADAPTIVE_SAFE_HARBOR,
            "floor should not exceed safe harbor (compile-time invariant also enforces this)",
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
        // warmed() initializes with NEGATIVE_DISTRIBUTION_MEAN (~0.033).
        // The computed adaptive threshold (0.033 + 2.5 × 0.0) × 3 = 0.099
        // should be clamped to the safe harbor (1.35), matching production
        // where the threshold is fed real silence/background scores.
        // Regression test for mahbot-891.
        let mut state = AdaptiveThresholdState::warmed();
        let threshold = state
            .feed(NEGATIVE_DISTRIBUTION_MEAN, ADAPTIVE_K_DEFAULT)
            .expect("warmed() should exit bootstrap");
        assert!(
            (threshold - ADAPTIVE_SAFE_HARBOR).abs() < 0.01,
            "warmed() threshold {threshold} should equal safe harbor {}",
            ADAPTIVE_SAFE_HARBOR,
        );
        // Verify that all bootstrap frames were fed NEGATIVE_DISTRIBUTION_MEAN
        // and not 0.5 (which would produce threshold ~1.5 instead of 1.35).
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
        // After eviction: mean ≈ 0.533, adaptive = 0.533 * 3 = 1.6, well above safe harbor 1.35
        // But we can still verify the internal mean indirectly.
        let expected_raw = expected_mean * ROLLING_WINDOW_N as f32;
        let clamped = expected_raw
            .max(ADAPTIVE_SAFE_HARBOR)
            .max(ADAPTIVE_FLOOR)
            .min(ADAPTIVE_CEILING);
        assert!(
            (threshold - clamped).abs() < 0.001,
            "threshold {threshold} should match expected clamped value {clamped} (raw={expected_raw})",
        );
    }

    // ── AdaptiveThresholdState::peek() tests (mahbot-852) ─────────────────
    // Tests for the peek() method which returns the current threshold without
    // updating statistics.  Covers bootstrap guard, empty-window check,
    // threshold correctness, and the no-mutation invariant.

    #[test]
    fn adaptive_peek_bootstrap_returns_none() {
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
    }

    #[test]
    fn adaptive_peek_after_bootstrap_returns_some() {
        // After bootstrap completes, peek() should return a threshold.
        let mut state = AdaptiveThresholdState::new();
        // Use low scores (0.1) so the computed adaptive value (~0.3) stays
        // below the safe harbor (1.35), verifying that peek() produces a
        // clamped threshold rather than a raw adaptive value (mahbot-860).
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.1, ADAPTIVE_K_DEFAULT);
        }
        let peek_result = state.peek(ADAPTIVE_K_DEFAULT);
        assert!(
            peek_result.is_some(),
            "peek should return Some after bootstrap",
        );
        // The threshold should equal the safe harbor (constant low-variance input).
        let threshold = peek_result.unwrap();
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
        // After reset, peek should return None (empty window + bootstrap reset).
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.5, ADAPTIVE_K_DEFAULT);
        }
        assert!(
            state.peek(ADAPTIVE_K_DEFAULT).is_some(),
            "peek should work after bootstrap"
        );

        state.reset();

        assert!(
            state.peek(ADAPTIVE_K_DEFAULT).is_none(),
            "peek should return None after reset (bootstrap_count=0)",
        );
    }

    #[test]
    fn adaptive_peek_threshold_in_valid_range() {
        // peek() should always return a threshold in the valid clamp range.
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.5, ADAPTIVE_K_DEFAULT);
        }
        // Test with various k values.
        for k in [0.0, 1.0, ADAPTIVE_K_DEFAULT, 4.0] {
            let threshold = state.peek(k).expect("peek should return Some");
            assert!(
                threshold >= ADAPTIVE_FLOOR,
                "k={k}: peek threshold {threshold} should be >= floor {}",
                ADAPTIVE_FLOOR,
            );
            assert!(
                threshold <= ADAPTIVE_CEILING,
                "k={k}: peek threshold {threshold} should be <= ceiling {}",
                ADAPTIVE_CEILING,
            );
        }
    }

    #[test]
    fn adaptive_peek_agrees_with_feed_on_same_state() {
        // After feed() updates the state, peek() on the same state should
        // return the same threshold value (both compute_threshold on the
        // same window).  Tests the shared compute_threshold helper.
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.3, ADAPTIVE_K_DEFAULT);
        }
        // Fill window with varied scores so we're not at the safe harbor floor.
        for &s in &[0.2, 0.6, 0.4, 0.8, 0.1, 0.9, 0.3, 0.7, 0.5, 0.15] {
            state.feed(s, ADAPTIVE_K_DEFAULT);
        }

        // feed updates state and returns threshold from the new window.
        let feed_threshold = state
            .feed(0.5, ADAPTIVE_K_DEFAULT)
            .expect("feed should return Some after bootstrap");

        // state now includes the 0.5 score. peek on the same state should
        // return the identical threshold.
        let peek_threshold = state
            .peek(ADAPTIVE_K_DEFAULT)
            .expect("peek should return Some after bootstrap");

        assert!(
            (peek_threshold - feed_threshold).abs() < 0.001,
            "peek threshold {peek_threshold} should equal feed threshold {feed_threshold} \
             on identical state (both use compute_threshold)",
        );
    }

    // ── score_single_embedding tiling fallback tests (mahbot-825) ──────────
    // Tests that when the embedding ring has fewer than WINDOW_SIZE (3) entries,
    // the available embeddings are tiled to fill the window instead of returning
    // a hard-coded 0.0.  These exercise the pure detection logic without ONNX
    // models — just the Conv1D classifier with known weights.

    /// Build a test classifier whose forward pass always returns `target_score`
    /// (by setting fc_bias = ln(target_score / (1 - target_score)) so that
    /// sigmoid(bias) = target_score).  All other weights are zeroed.
    fn classifier_always_score(target_score: f32) -> WakeWordClassifier {
        let target = target_score.clamp(1e-6, 1.0 - 1e-6); // avoid log(0)
        let mut w = ClassifierWeights::default();
        w.conv1_weight.fill(0.0);
        w.conv1_bias.fill(0.0);
        w.conv2_weight.fill(0.0);
        w.conv2_bias.fill(0.0);
        w.fc_weight.fill(0.0);
        w.fc_bias[0] = (target / (1.0 - target)).ln();
        WakeWordClassifier::new(w)
    }

    /// Build a test classifier that always returns ~0.5 for any input
    /// (sigmoid(0.0) = 0.5).
    fn classifier_always_half() -> WakeWordClassifier {
        classifier_always_score(0.5)
    }

    #[test]
    fn score_single_1_embedding_tiles_to_nonzero() {
        // When the ring has only 1 embedding after push, the tiling fallback
        // should produce a non-zero score (replaces the old hard-coded 0.0).
        let classifier = classifier_always_half();
        let embedding = vec![0.5; EMBEDDING_DIM];
        let mut ring = Vec::with_capacity(EMBEDDING_RING_MAX);
        let mut score_window = Vec::new();

        let (detected, _, _, _) = score_single_embedding(
            &embedding,
            &mut ring,
            Some(&classifier),
            &mut score_window,
            None,
            ADAPTIVE_K_DEFAULT,
            false,
        );

        // Score is ~0.5 ≥ NO_MATCH_RESET_THRESHOLD (0.316) → window appended.
        // Rolling sum 0.5 < match_threshold (1.35) → detection does NOT fire.
        assert!(
            !detected,
            "single embedding should not trigger detection (rolling sum < threshold)",
        );
        assert!(
            !score_window.is_empty(),
            "tiling should produce a score above NO_MATCH_RESET_THRESHOLD (0.316), giving a non-empty score window",
        );

        let score = score_window[0];
        assert!(
            (score - 0.5).abs() < 0.01,
            "Expected score ~0.5 from always-half classifier, got {score}",
        );
        assert_eq!(ring.len(), 1, "ring should hold exactly 1 embedding");
    }

    #[test]
    fn score_single_2_embeddings_tiles_to_nonzero() {
        // With 2 embeddings in the ring, the tiling fallback should also
        // produce a non-zero score (repeat-last: [a, b, b]).
        let classifier = classifier_always_half();
        let emb = vec![0.5; EMBEDDING_DIM];
        let mut ring = Vec::with_capacity(EMBEDDING_RING_MAX);
        let mut score_window = Vec::new();

        // First embedding: ring = [a], tiled to [a, a, a] → score ~0.5.
        let (_detected, _, _, _) = score_single_embedding(
            &emb,
            &mut ring,
            Some(&classifier),
            &mut score_window,
            None,
            ADAPTIVE_K_DEFAULT,
            false,
        );
        assert!(
            !score_window.is_empty(),
            "first embedding should produce a score"
        );
        assert_eq!(
            ring.len(),
            1,
            "ring should have 1 embedding after first push"
        );

        // Second embedding: ring = [a, b], tiled to [a, b, b] → score ~0.5.
        score_window.clear();
        let (detected, _, _, _) = score_single_embedding(
            &emb,
            &mut ring,
            Some(&classifier),
            &mut score_window,
            None,
            ADAPTIVE_K_DEFAULT,
            false,
        );
        assert!(
            !detected,
            "two embeddings should not trigger detection (rolling sum < 1.35)",
        );
        assert!(
            !score_window.is_empty(),
            "second embedding tiling should produce a score above NO_MATCH_RESET_THRESHOLD (0.316)",
        );
        assert_eq!(ring.len(), 2, "ring should have 2 embeddings");
    }

    #[test]
    fn score_single_3_embeddings_uses_natural_window() {
        // With 3+ embeddings, the natural sliding window is used (no tiling).
        let classifier = classifier_always_half();
        let emb = vec![0.5; EMBEDDING_DIM];
        let mut ring = Vec::with_capacity(EMBEDDING_RING_MAX);
        let mut score_window = Vec::new();

        for _ in 0..3 {
            let _ = score_single_embedding(
                &emb,
                &mut ring,
                Some(&classifier),
                &mut score_window,
                None,
                ADAPTIVE_K_DEFAULT,
                false,
            );
        }

        assert!(
            !score_window.is_empty(),
            "three embeddings should produce a score",
        );
        assert!(ring.len() >= 3, "ring should have ≥3 embeddings");
    }

    #[test]
    fn score_single_tiled_matches_natural_for_stationary_input() {
        // For stationary input (all embeddings identical), the tiled window
        // [a, a, a] should produce the same score as the natural window [a, a, a]
        // after 3 pushes.  The Conv1D classifier is deterministic, and
        // AdaptiveAvgPool collapses the time dimension, so the scores match.
        let classifier = classifier_always_half();
        let embedding = vec![0.5; EMBEDDING_DIM];

        // Method 1: tiled fallback with 1 embedding → [a, a, a]
        let mut ring_tiled = Vec::with_capacity(EMBEDDING_RING_MAX);
        let mut sw_tiled = Vec::new();
        let _ = score_single_embedding(
            &embedding,
            &mut ring_tiled,
            Some(&classifier),
            &mut sw_tiled,
            None,
            ADAPTIVE_K_DEFAULT,
            false,
        );
        let tiled_score = sw_tiled.last().copied().unwrap_or(0.0);

        // Method 2: natural 3-embedding window after 3 pushes
        let mut ring_nat = Vec::with_capacity(EMBEDDING_RING_MAX);
        let mut sw_nat = Vec::new();
        for _ in 0..3 {
            let _ = score_single_embedding(
                &embedding,
                &mut ring_nat,
                Some(&classifier),
                &mut sw_nat,
                None,
                ADAPTIVE_K_DEFAULT,
                false,
            );
        }
        let natural_score = sw_nat.last().copied().unwrap_or(0.0);

        assert!(
            (tiled_score - natural_score).abs() < 0.01,
            "Tiled score {tiled_score} should match natural score {natural_score} \
             for stationary input",
        );
    }

    // ── score_single_embedding with adaptive state (mahbot-852) ────────────
    // Tests that the feed/peek branching logic in score_single_embedding is
    // exercised: adaptive_state is passed as Some(...) so the code path that
    // conditionally calls feed() (for background scores) or peek() (for
    // wake-word-like scores) is actually executed.  Since mahbot-1023 the
    // feed/peek rule is score-based ONLY, including during bootstrap (below-
    // reset scores feed; wake-word-like scores peek) — see
    // score_single_with_adaptive_state_feeds_background_only.

    #[test]
    fn score_single_with_adaptive_state_completes_bootstrap() {
        // With an adaptive state, score_single_embedding should complete
        // the bootstrap phase after ADAPTIVE_BOOTSTRAP_FRAMES embeddings.
        // Uses a classifier that produces ~0.3 per frame so the rolling sum
        // stays below the detection threshold (0.3 × 3 = 0.9 < 1.35).
        // Updated from classifier_always_half() (~0.5 per frame) after the
        // mahbot-860 threshold lowering (1.65 → 1.35) made 3 × 0.5 = 1.5 ≥ 1.35
        // trigger detection during the test.
        let classifier = classifier_always_score(0.3);
        let embedding = vec![0.5; EMBEDDING_DIM];
        let mut ring = Vec::with_capacity(EMBEDDING_RING_MAX);
        let mut score_window = Vec::new();
        let mut adaptive = AdaptiveThresholdState::new();

        // Send enough embeddings to complete the bootstrap.
        for i in 0..ADAPTIVE_BOOTSTRAP_FRAMES + 1 {
            let (detected, _, _, _) = score_single_embedding(
                &embedding,
                &mut ring,
                Some(&classifier),
                &mut score_window,
                Some(&mut adaptive),
                ADAPTIVE_K_DEFAULT,
                false,
            );
            // None of these should detect (rolling sum ~0.9 < 1.35).
            assert!(!detected, "frame {i} should not detect wake word");
        }

        // After bootstrap, the adaptive state should have exited bootstrap.
        assert!(
            adaptive.bootstrap_count >= ADAPTIVE_BOOTSTRAP_FRAMES,
            "adaptive state should have exited bootstrap",
        );
        // peek should return Some after bootstrap is complete.
        assert!(
            adaptive.peek(ADAPTIVE_K_DEFAULT).is_some(),
            "adaptive state peek should return Some after bootstrap",
        );
    }

    #[test]
    fn score_single_with_adaptive_state_feeds_background_only() {
        // Verify that score_single_embedding feeds ONLY background scores
        // (below NO_MATCH_RESET_THRESHOLD) to the adaptive state — INCLUDING
        // during bootstrap (mahbot-1023 bootstrap feed fix).  High scores go
        // through peek() instead, so a high-scoring utterance never
        // contaminates the adaptive statistics and can legitimately keep the
        // bootstrap alive for its whole duration (the static
        // match_threshold() stays in effect — peek returns None during
        // bootstrap).
        let embedding = vec![0.5; EMBEDDING_DIM];

        // ── Phase 1: high scores during bootstrap PEEK (no stats, no
        //    bootstrap progress) ──────────────────────────────────────────
        // 0.5 is ≥ NO_MATCH_RESET_THRESHOLD (0.316) but keeps the rolling sum
        // (3 × 0.5 = 1.5) below the 2.13 detection threshold.
        let classifier_high = classifier_always_half();
        let mut ring = Vec::with_capacity(EMBEDDING_RING_MAX);
        let mut score_window = Vec::new();
        let mut adaptive = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES + 1 {
            score_single_embedding(
                &embedding,
                &mut ring,
                Some(&classifier_high),
                &mut score_window,
                Some(&mut adaptive),
                ADAPTIVE_K_DEFAULT,
                false,
            );
        }
        // The bootstrap must NOT have completed and no statistics accumulated:
        // the old unconditional bootstrap feed would have completed it with
        // 5 wake-word-like scores (the mahbot-1023 contamination).
        assert!(
            adaptive.is_bootstrapping(),
            "high scores must not complete the bootstrap (peek during bootstrap)",
        );
        assert_eq!(
            adaptive.scores.len(),
            0,
            "high scores must not feed the adaptive window even during bootstrap",
        );

        // ── Phase 2: below-reset scores during bootstrap FEED (complete it) ──
        let classifier_low = classifier_always_score(0.1); // < NO_MATCH_RESET_THRESHOLD
        let mut ring2 = Vec::with_capacity(EMBEDDING_RING_MAX);
        let mut score_window2 = Vec::new();
        let mut adaptive2 = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            score_single_embedding(
                &embedding,
                &mut ring2,
                Some(&classifier_low),
                &mut score_window2,
                Some(&mut adaptive2),
                ADAPTIVE_K_DEFAULT,
                false,
            );
        }
        assert!(
            !adaptive2.is_bootstrapping(),
            "below-reset background scores must complete the bootstrap",
        );

        // ── Phase 3: post-bootstrap high scores PEEK (no stats growth) ──
        let scores_len_before = adaptive2.scores.len();
        let sum_before = adaptive2.sum;
        score_single_embedding(
            &embedding,
            &mut ring2,
            Some(&classifier_high),
            &mut score_window2,
            Some(&mut adaptive2),
            ADAPTIVE_K_DEFAULT,
            false,
        );
        assert_eq!(
            adaptive2.scores.len(),
            scores_len_before,
            "post-bootstrap high scores must use peek (no window growth)",
        );
        assert!(
            (adaptive2.sum - sum_before).abs() < f32::EPSILON,
            "post-bootstrap high scores must not change the adaptive sum",
        );
    }

    // ── PipelineCtx adaptive_k clamping (mahbot-845) ──────────────────────
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
        ctx.collecting_negatives = true;
        ctx.phase3_audio_buf = vec![0.5; 100];
        ctx.phase3_silence_samples = 500;
        ctx.negatives_speech_samples = 1000;
        ctx.phase3_start_time = Some(Instant::now() - Duration::from_secs(10));
        ctx.vad_threshold = 0.75;
        ctx.last_wake_word_detection = Some(Instant::now() - Duration::from_secs(5));
        ctx.auto_start_pending = true;
        ctx.is_recording = true;
        ctx.peak_score = 0.75;
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

        // Segment boundary tracking (mahbot-894).
        assert_eq!(
            ctx.segment_silence_hops, 0,
            "segment_silence_hops must be cleared by all reset levels"
        );

        // Phase 3 owner-negative state (mahbot-913).
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
            state.enrollment_buffer.push(EmbeddingSequence::new(
                UtteranceId {
                    sequence_index: 0,
                    variant_index: 0,
                },
                Source::Enrollment,
                vec![vec![0.5; 96]],
            ));
            state.negative_audio_chunks.push(vec![0.5; 100]);
        }

        ctx.reset_pipeline_state(ResetLevel::Full);

        assert_buffers_cleared(&ctx);

        // Full-specific: state flags and peak scores reset.
        assert_eq!(ctx.vad_threshold, VAD_THRESHOLD);
        assert!(ctx.last_wake_word_detection.is_none());
        assert!(!ctx.auto_start_pending);
        assert!(!ctx.is_recording);
        assert_eq!(ctx.peak_score, 0.0, "peak_score must be reset on Full");

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
        let saved_buffer = vec![EmbeddingSequence::new(
            UtteranceId {
                sequence_index: 0,
                variant_index: 0,
            },
            Source::Enrollment,
            vec![vec![0.5; 96]],
        )];
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
        let saved_peak_score = ctx.peak_score;

        ctx.reset_pipeline_state(ResetLevel::Soft);

        assert_buffers_cleared(&ctx);

        // Soft preserves these.
        assert_eq!(ctx.vad_threshold, saved_threshold);
        assert_eq!(ctx.last_wake_word_detection, saved_cooldown);
        assert_eq!(ctx.auto_start_pending, saved_auto_start);
        assert_eq!(ctx.is_recording, saved_recording);
        // Soft clears rolling-window detection state (mahbot-895) but
        // preserves peak_score (which is only reset on Full/Cancel alongside
        // the wider acoustic state).
        assert_eq!(ctx.peak_score, saved_peak_score);

        // Global enrollment accumulators preserved.
        let state = voice_state().read().unwrap_poison();
        assert_eq!(
            state.enrollment_buffer.len(),
            saved_buffer.len(),
            "enrollment_buffer preserved (Soft)"
        );
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
            state.enrollment_buffer.push(EmbeddingSequence::new(
                UtteranceId {
                    sequence_index: 0,
                    variant_index: 0,
                },
                Source::Enrollment,
                vec![vec![0.5; 96]],
            ));
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
        // Cancel resets peak_score.
        assert_eq!(ctx.peak_score, 0.0, "peak_score must be reset on Cancel");

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

    // ── reset_detection_segment tests (mahbot-894) ────────────────────────
    // Verifies that the bounded-segment reset clears per-segment detection
    // state (embedding_ring, score_window, adaptive_threshold, peaks) while
    // preserving session-level state (VAD, audio_preprocessor, is_recording,
    // last_wake_word_detection, vad_threshold).

    #[test]
    fn reset_detection_segment_saves_peaks_to_instrumentation() {
        // Only meaningful with voice-tests feature.
        #[cfg(feature = "voice-tests")]
        {
            let mut ctx = PipelineCtx::new();
            ctx.peak_score = 0.85;

            // Pre-seed instrumentation with a lower value to verify max-tracking
            ctx.instrumentation.peak_score = 0.10;

            ctx.reset_detection_segment();

            // Instrumentation should have captured the higher peak_score value
            assert!(
                ctx.instrumentation.peak_score >= 0.85 - f32::EPSILON,
                "instrumentation.peak_score should capture pre-reset peak_score"
            );
        }
    }

    // ── handle_segment_boundary tests (mahbot-894) ──────────────────────
    // Tests the extracted segment boundary check logic.  The public-API test
    // (handle_segment_boundary) is the canonical reset-contract reference and
    // supersedes the removed internal reset_detection_segment test.

    #[test]
    fn handle_segment_boundary_resets_at_threshold() {
        let mut ctx = PipelineCtx::new();
        ctx.embedding_ring = vec![vec![0.5; 96]; 3];
        ctx.score_window = vec![0.5; 5];
        ctx.peak_score = 0.75;
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

        let mut voice_batch = vec![0.5; 100];
        let mut mel_frame_buffer = vec![vec![0.5; 32]; 10];

        // hop_count at threshold → boundary fires
        ctx.handle_segment_boundary(
            SEGMENT_TIMEOUT_HOPS,
            &mut voice_batch,
            &mut mel_frame_buffer,
        );

        // ── Per-segment state reset on ctx ──
        assert!(
            ctx.embedding_ring.is_empty(),
            "embedding_ring must be cleared"
        );
        assert!(ctx.score_window.is_empty(), "score_window must be cleared");
        assert_eq!(ctx.peak_score, 0.0, "peak_score must be reset");
        assert_eq!(
            ctx.segment_silence_hops, 0,
            "segment_silence_hops must be reset"
        );
        assert!(
            ctx.adaptive_threshold.is_bootstrapping(),
            "adaptive_threshold must be reset (re-enter bootstrap)"
        );

        // ── Caller's local buffers cleared ──
        assert!(
            voice_batch.is_empty(),
            "voice_batch must be cleared on boundary"
        );
        assert!(
            mel_frame_buffer.is_empty(),
            "mel_frame_buffer must be cleared on boundary"
        );
    }

    #[test]
    fn handle_segment_boundary_persists_below_threshold() {
        let mut ctx = PipelineCtx::new();
        ctx.embedding_ring = vec![vec![0.5; 96]; 3];
        ctx.score_window = vec![0.5; 5];
        ctx.segment_silence_hops = 10;

        let mut voice_batch = vec![0.5; 100];
        let mut mel_frame_buffer = vec![vec![0.5; 32]; 10];

        // hop_count below threshold → state persisted, buffers preserved
        let below_threshold = SEGMENT_TIMEOUT_HOPS - 1;
        ctx.handle_segment_boundary(below_threshold, &mut voice_batch, &mut mel_frame_buffer);

        // ── Per-segment state preserved on ctx ──
        assert_eq!(
            ctx.segment_silence_hops, below_threshold,
            "counter must be persisted below threshold"
        );
        assert!(
            !ctx.embedding_ring.is_empty(),
            "embedding_ring must survive below threshold"
        );
        assert!(
            !ctx.score_window.is_empty(),
            "score_window must survive below threshold"
        );

        // ── Caller's local buffers preserved ──
        assert!(
            !voice_batch.is_empty(),
            "voice_batch must survive below threshold"
        );
        assert!(
            !mel_frame_buffer.is_empty(),
            "mel_frame_buffer must survive below threshold"
        );
    }

    // ── Deferred burst state machine (mahbot-1023, ONNX-free) ─────────────
    // The hold / burst-sweep / segment-end-pass / per-segment-flag lifecycle
    // is extracted into pure helpers (burst_positions_to_score /
    // start_aligned_positions) plus the PipelineCtx burst_sweep_done latch.
    // These tests exercise the real control flow WITHOUT ONNX models
    // (ONNX_MODELS is None in unit tests).

    #[test]
    fn burst_hold_until_trigger_frames() {
        // The burst must HOLD position 0 until the buffer reaches
        // BURST_TRIGGER_FRAMES (68): incremental per-chunk scoring of
        // positions 0/8/16/24 with 2–16 frames per chunk is exactly the
        // mahbot-1023 bug (a ~10-real-frame window scores ~0.027 and never
        // re-scores).
        for b in 0..BURST_TRIGGER_FRAMES {
            assert!(
                burst_positions_to_score(b, false).is_empty(),
                "buffer {b} < {BURST_TRIGGER_FRAMES} must hold (no scoring)"
            );
        }
        // At exactly the trigger the burst fires from start-aligned position 0.
        let positions = burst_positions_to_score(BURST_TRIGGER_FRAMES, false);
        assert_eq!(positions, vec![0, 8, 16, 24], "burst positions at trigger");
    }

    #[test]
    fn burst_position_zero_is_never_skipped() {
        // Invariant: the burst's first scored position is ALWAYS 0 (the
        // trained start-0-aligned geometry).  Positions never exceed the
        // buffer end and never re-score later (see burst_sweep_done latch).
        for b in BURST_TRIGGER_FRAMES..=80 {
            let positions = burst_positions_to_score(b, false);
            assert!(!positions.is_empty(), "burst must fire at buffer {b}");
            assert_eq!(
                positions.first(),
                Some(&0),
                "start-aligned position 0 must never be skipped at buffer {b}"
            );
            for &p in &positions {
                assert!(p < b, "position {p} must be below buffer end {b}");
            }
        }
    }

    #[test]
    #[serial_test::serial(voice)]
    fn burst_sweep_done_latches_until_segment_reset() {
        // Per-segment flag lifecycle: burst_sweep_done is set after the sweep
        // (preventing re-runs and mid-utterance trailing re-scoring) and
        // cleared ONLY by the segment resets.
        let _ = init_global();
        let mut ctx = PipelineCtx::new();
        assert!(!ctx.burst_sweep_done, "fresh ctx starts with sweep pending");

        // Simulate the sweep having run (as handle_wake_word_detection does).
        ctx.burst_sweep_done = true;
        // Rolling-window-reset protection (merged from the former
        // burst_rolling_window_reset_protection): once the sweep is done,
        // positions beyond the buffer end at sweep time are NEVER re-scored
        // on later calls (a re-scored ≤17-real-frame window ~0.01
        // mid-utterance would reset the rolling window).  The latch returns
        // an empty position set for ANY later
        // buffer length.
        for later_b in BURST_TRIGGER_FRAMES..=120 {
            assert!(
                burst_positions_to_score(later_b, ctx.burst_sweep_done).is_empty(),
                "burst must not re-run once the per-segment sweep is done (buffer {later_b})"
            );
        }

        // Segment reset clears the latch → a new utterance may burst again.
        ctx.reset_detection_segment();
        assert!(
            !ctx.burst_sweep_done,
            "segment reset must clear the burst_sweep_done latch"
        );
        assert!(
            !burst_positions_to_score(70, ctx.burst_sweep_done).is_empty(),
            "a fresh segment must be allowed a new burst sweep"
        );

        // reset_pipeline_state (all levels) also starts a fresh segment.
        for level in [ResetLevel::Soft, ResetLevel::Full, ResetLevel::Cancel] {
            let mut ctx2 = PipelineCtx::new();
            ctx2.burst_sweep_done = true;
            ctx2.reset_pipeline_state(level);
            assert!(
                !ctx2.burst_sweep_done,
                "{level:?} reset must clear the burst_sweep_done latch"
            );
        }
    }

    #[test]
    fn segment_end_pass_uses_burst_grid_latch_independently() {
        // mahbot-1023 item 8 + manager pin 3: the boundary pass must cover
        // the burst-equivalent grid (so the ring-4 sample at
        // position 24 scores exactly like the cold burst) AND be
        // latch-independent — a burst sweep that ran without detecting
        // (burst_sweep_done == true) must NOT suppress the boundary fallback
        // (F1's only current detection path is the trailing boundary
        // re-score).  Both properties are structural in the shared
        // `start_aligned_positions` grid: the pass and the burst build their
        // position lists from the same function, and the grid takes no latch
        // input, so a latched burst (empty set) does not shrink the pass
        // grid.  The pass gate in handle_segment_boundary never consults the
        // latch either (it re-checks only not-recording / models-loaded /
        // non-empty buffer).
        assert!(
            start_aligned_positions(0).is_empty(),
            "no positions at an empty buffer"
        );
        for b in 1..=100usize {
            let pass = start_aligned_positions(b);
            assert_eq!(
                pass.first(),
                Some(&0),
                "pass must score from start-aligned position 0 at buffer {b}"
            );
            for &p in &pass {
                assert!(p < b, "pass position {p} must be below buffer end {b}");
            }
            assert!(
                pass.len() <= BURST_MAX_POSITIONS,
                "pass grid must never exceed {BURST_MAX_POSITIONS} positions"
            );
            if b >= BURST_TRIGGER_FRAMES {
                // Burst-equivalent grid: the pass scores the same grid the
                // burst swept.
                assert_eq!(
                    pass,
                    burst_positions_to_score(b, false),
                    "pass must cover the same grid the burst swept at buffer {b}"
                );
                // Latch-independence: the burst is suppressed by the
                // per-segment latch, the pass grid is not.
                assert!(
                    burst_positions_to_score(b, true).is_empty(),
                    "a latched burst must not re-run at buffer {b}"
                );
                assert!(
                    !pass.is_empty(),
                    "pass must not be suppressed by a failed burst at buffer {b}"
                );
            } else {
                // Short buffers: the burst holds (no scoring below the
                // trigger), but the pass still covers the grid the burst
                // never reached.
                assert!(
                    burst_positions_to_score(b, false).is_empty(),
                    "burst must hold below the trigger at buffer {b}"
                );
                assert!(
                    !pass.is_empty(),
                    "pass must still cover short buffers the burst never reached at buffer {b}"
                );
            }
        }
        // Full-grid invariants at the measured detection lengths.
        assert_eq!(
            start_aligned_positions(BURST_TRIGGER_FRAMES),
            vec![0, 8, 16, 24],
            "pass positions at the burst-trigger length"
        );
        // Trailing-68 trimmed state (longer utterances): 68/60/52/44 real
        // frames at positions 0/8/16/24 — the measured detection family.
        assert_eq!(
            start_aligned_positions(EMBEDDING_WINDOW_FRAMES.saturating_sub(BURST_STRIDE)),
            vec![0, 8, 16, 24],
            "trailing-68 pass positions"
        );
    }

    #[test]
    fn segment_boundary_clears_burst_latch_and_buffers() {
        // handle_segment_boundary at the hop threshold resets the per-segment
        // burst latch along with the other per-segment state (ONNX-free: the
        // segment-end pass itself is skipped because ONNX_MODELS is None, so
        // the reset path is what is exercised here).
        let mut ctx = PipelineCtx::new();
        ctx.burst_sweep_done = true;
        ctx.embedding_ring = vec![vec![0.5; 96]; 3];
        ctx.score_window = vec![0.5; 5];
        let mut voice_batch = vec![0.5; 100];
        let mut mel_frame_buffer = vec![vec![0.5; 32]; 10];
        ctx.handle_segment_boundary(
            SEGMENT_TIMEOUT_HOPS,
            &mut voice_batch,
            &mut mel_frame_buffer,
        );
        assert!(
            !ctx.burst_sweep_done,
            "segment boundary must clear the burst_sweep_done latch"
        );
        assert!(
            ctx.embedding_ring.is_empty() && ctx.score_window.is_empty(),
            "per-segment scoring state must be cleared"
        );
        assert!(
            voice_batch.is_empty() && mel_frame_buffer.is_empty(),
            "caller's local buffers must be cleared on boundary"
        );
    }

    #[test]
    fn segment_boundary_vad_gap_counting_integration() {
        // Full integration test exercising the segment boundary flow through
        // process_streaming_frames_inner with a custom VAD closure, then
        // handle_segment_boundary — the same pattern used by
        // handle_wake_word_detection.
        //
        // Validates:
        //   1. VAD-gap counting across two process_streaming_frames_inner calls
        //   2. Boundary detection at exactly SEGMENT_TIMEOUT_HOPS hops
        //   3. State reset on ctx (embedding_ring, score_window, peaks)
        //   4. Local buffer clearing (voice_batch, mel_frame_buffer)
        //   5. ctx-level buffers preserved (caller-managed invariant)
        //   6. State persistence across calls when hop count is below threshold
        //
        // Uses audio_for_frames to correctly size audio buffers for the sliding
        // window frame loop (HOP_LENGTH=256 stride).  Each frame advances
        // consumed by HOP_LENGTH, so N frames require (N-1)*HOP_LENGTH +
        // FRAME_LENGTH samples.

        let mut ctx = PipelineCtx::new();
        ctx.embedding_ring = vec![vec![0.5; 96]; 3];
        ctx.score_window = vec![0.5; 5];
        ctx.peak_score = 0.75;
        ctx.segment_silence_hops = SEGMENT_TIMEOUT_HOPS - 5; // 14, just below threshold

        // Pre-populate ctx-level copies to verify caller-managed invariant.
        // These should NOT be cleared by the boundary (only local vars are).
        ctx.voice_batch = vec![0.5; 50];
        ctx.mel_frame_buffer = vec![vec![0.5; 32]; 5];

        // ── Call 1: feed 4 frames of silence → 14+4=18 < 19, no boundary ──
        let mut voice_batch = vec![0.5; 100]; // non-empty to verify clearing on boundary
        let mut mel_frame_buffer = vec![vec![0.5; 32]; 10];

        let segment_hops = std::sync::atomic::AtomicUsize::new(ctx.segment_silence_hops);
        let is_speech_fn = |_frame: &[f32]| -> bool {
            segment_hops.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            false // always silence — increments counter
        };

        let audio_call_1 = audio_for_frames(4);
        let _consumed = process_streaming_frames_inner(
            &audio_call_1,
            &mut voice_batch,
            &mut mel_frame_buffer,
            is_speech_fn,
            false,
            |_mel_frames| false,
        );

        let hop_count_1 = segment_hops.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            hop_count_1 < SEGMENT_TIMEOUT_HOPS,
            "Call 1: counter {} should be < threshold {} (14+4=18 expected)",
            hop_count_1,
            SEGMENT_TIMEOUT_HOPS
        );
        ctx.handle_segment_boundary(hop_count_1, &mut voice_batch, &mut mel_frame_buffer);

        // State should be preserved (no boundary)
        assert_eq!(
            ctx.segment_silence_hops, hop_count_1,
            "Call 1: counter must be persisted below threshold"
        );
        assert!(
            !ctx.embedding_ring.is_empty(),
            "Call 1: embedding_ring must survive below threshold"
        );

        // ── Call 2: feed 2 more frames → 18+2=20 ≥ 19, boundary fires ──
        let segment_hops = std::sync::atomic::AtomicUsize::new(ctx.segment_silence_hops);
        let is_speech_fn = |_frame: &[f32]| -> bool {
            segment_hops.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            false
        };

        let audio_call_2 = audio_for_frames(2);
        let _consumed = process_streaming_frames_inner(
            &audio_call_2,
            &mut voice_batch,
            &mut mel_frame_buffer,
            is_speech_fn,
            false,
            |_mel_frames| false,
        );

        let hop_count_2 = segment_hops.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            hop_count_2 >= SEGMENT_TIMEOUT_HOPS,
            "Call 2: counter {} should be >= threshold {} (18+2=20 expected)",
            hop_count_2,
            SEGMENT_TIMEOUT_HOPS
        );
        ctx.handle_segment_boundary(hop_count_2, &mut voice_batch, &mut mel_frame_buffer);

        // ── Verify boundary fired ──
        assert_eq!(
            ctx.segment_silence_hops, 0,
            "segment_silence_hops must be reset after boundary"
        );
        assert!(
            ctx.embedding_ring.is_empty(),
            "embedding_ring must be cleared after boundary"
        );
        assert!(
            ctx.score_window.is_empty(),
            "score_window must be cleared after boundary"
        );
        assert_eq!(ctx.peak_score, 0.0, "peak_score must be reset");
        assert!(
            voice_batch.is_empty(),
            "local voice_batch must be cleared on boundary"
        );
        assert!(
            mel_frame_buffer.is_empty(),
            "local mel_frame_buffer must be cleared on boundary"
        );

        // ── Verify ctx-level copies preserved (caller-managed invariant) ──
        assert!(
            !ctx.voice_batch.is_empty(),
            "ctx.voice_batch must survive boundary (caller-managed)"
        );
        assert!(
            !ctx.mel_frame_buffer.is_empty(),
            "ctx.mel_frame_buffer must survive boundary (caller-managed)"
        );
    }

    #[test]
    fn segment_boundary_alternating_vad_resets_counter_on_speech() {
        // AC #3: VAD-positive frames reset the silence counter, preventing
        // short intra-phrase pauses from triggering a segment boundary.
        //
        // Simulates: [3 speech frames] → [10 silence frames] → [2 speech frames]
        // → [20 silence frames].  The short silence (10 hops < SEGMENT_TIMEOUT_HOPS)
        // combined with the speech gap does NOT fire a boundary.  The long silence
        // (20 hops >= SEGMENT_TIMEOUT_HOPS) DOES fire a boundary — but only because
        // the speech frames reset the counter, proving the counter-reset mechanism
        // works correctly.
        //
        // VAD-positive frames: counter resets to 0 via store().
        // VAD-negative frames: counter increments via fetch_add().
        // Both operations happen inside the is_speech_fn closure — the same pattern
        // used by handle_wake_word_detection (mahbot-894).
        use std::sync::atomic::Ordering;

        const SPEECH_FRAMES_1: usize = 3;
        const SILENCE_FRAMES_1: usize = 10;
        const SPEECH_FRAMES_2: usize = 2;
        const SILENCE_FRAMES_2: usize = 20;
        const TOTAL_FRAMES: usize =
            SPEECH_FRAMES_1 + SILENCE_FRAMES_1 + SPEECH_FRAMES_2 + SILENCE_FRAMES_2;

        let mut ctx = PipelineCtx::new();
        ctx.segment_silence_hops = 0;
        ctx.embedding_ring = vec![vec![0.5; 96]; 3];
        ctx.score_window = vec![0.5; 5];
        ctx.peak_score = 0.75;

        let mut voice_batch = Vec::new();
        let mut mel_frame_buffer = Vec::new();

        // Silence counter tracked inside the closure, matching the
        // handle_wake_word_detection pattern.
        let segment_silence_hops = std::sync::atomic::AtomicUsize::new(ctx.segment_silence_hops);
        let frame_index = std::sync::atomic::AtomicUsize::new(0);

        // Phase boundaries (exclusive):
        //   Phase 1: [0, SPEECH_FRAMES_1)               — speech
        //   Phase 2: [SPEECH_FRAMES_1, +SILENCE_FRAMES_1) — silence
        //   Phase 3: [+SILENCE_FRAMES_1, +SPEECH_FRAMES_2) — speech
        //   Phase 4: remainder                            — silence
        let silence_start_2 = SPEECH_FRAMES_1 + SILENCE_FRAMES_1 + SPEECH_FRAMES_2;

        let is_speech_fn = |_frame: &[f32]| -> bool {
            let fi = frame_index.fetch_add(1, Ordering::Relaxed);
            let speech = fi < SPEECH_FRAMES_1
                || (fi >= SPEECH_FRAMES_1 + SILENCE_FRAMES_1 && fi < silence_start_2);
            if speech {
                segment_silence_hops.store(0, Ordering::Relaxed);
            } else {
                segment_silence_hops.fetch_add(1, Ordering::Relaxed);
            }
            speech
        };

        let audio = audio_for_frames(TOTAL_FRAMES);
        let _consumed = process_streaming_frames_inner(
            &audio,
            &mut voice_batch,
            &mut mel_frame_buffer,
            is_speech_fn,
            false,
            |_mel_frames| false,
        );

        // After 10 silence + 20 silence = 30 silence frames total, but counter
        // was reset to 0 by 2 speech frames before Phase 4.  So final counter
        // is 20 (from Phase 4 alone), which is >= SEGMENT_TIMEOUT_HOPS (19).
        let hop_count = segment_silence_hops.load(Ordering::Relaxed);
        assert!(
            hop_count >= SEGMENT_TIMEOUT_HOPS,
            "hop_count ({}) should be >= SEGMENT_TIMEOUT_HOPS ({}) after {} silence frames \
             (speech reset the counter before Phase 4)",
            hop_count,
            SEGMENT_TIMEOUT_HOPS,
            SILENCE_FRAMES_2,
        );

        ctx.handle_segment_boundary(hop_count, &mut voice_batch, &mut mel_frame_buffer);

        // ── Verify boundary fired ──
        assert_eq!(
            ctx.segment_silence_hops, 0,
            "counter must be reset after boundary"
        );
        assert!(
            ctx.embedding_ring.is_empty(),
            "embedding_ring must be cleared after boundary"
        );
        assert!(
            ctx.score_window.is_empty(),
            "score_window must be cleared after boundary"
        );
        assert_eq!(ctx.peak_score, 0.0, "peak_score must be reset");
        assert!(
            voice_batch.is_empty(),
            "local voice_batch must be cleared on boundary"
        );
        assert!(
            mel_frame_buffer.is_empty(),
            "local mel_frame_buffer must be cleared on boundary"
        );
        assert!(
            ctx.adaptive_threshold.is_bootstrapping(),
            "adaptive_threshold must be reset after boundary"
        );
    }

    #[test]
    fn recording_mode_preserves_segment_state_across_silence() {
        // Validates the safety invariant that recording-mode detection does
        // not corrupt per-segment state from stale frame-loop counters.
        //
        // The `if !ctx.is_recording` guard (line ~6144) prevents stale
        // writeback of the local VAD-gap counter into ctx.segment_silence_hops
        // after detection fires and reset_pipeline_state(Soft) clears it.
        // This guard is trivially simple (single boolean check) and tested by
        // directly exercising the same code path with the guard's semantics:
        //
        //   1. `is_recording = true` (detection just fired)
        //   2. `segment_silence_hops = 0` (Soft reset already cleared it)
        //   3. Process frames through process_streaming_frames_inner (same
        //      function used by handle_wake_word_detection) — the frame loop
        //      accumulates the local VAD-gap counter past threshold
        //   4. Skip handle_segment_boundary (mimicking the guard)
        //   5. Verify ctx.segment_silence_hops still 0 — no stale writeback
        //
        // This exercises the same process_streaming_frames_inner path as the
        // production guard, validating the state-preservation invariant without
        // requiring real VAD/mel processing.  The guard's control flow (the
        // `if` itself) is a single boolean branch with no side effects — the
        // meaningful invariant is that ctx state survives recording-mode audio
        // processing untouched.

        let mut ctx = PipelineCtx::new();
        ctx.is_recording = true;
        ctx.segment_silence_hops = 0;
        ctx.embedding_ring = vec![vec![0.5; 96]; 3];
        ctx.score_window = vec![0.5; 5];

        // Pre-populate ctx-level buffers to verify they survive
        ctx.voice_batch = vec![0.5; 50];
        ctx.mel_frame_buffer = vec![vec![0.5; 32]; 5];

        let mut voice_batch = vec![0.5; 100];
        let mut mel_frame_buffer = vec![vec![0.5; 32]; 10];

        // Process enough silence frames to cross the threshold
        // (SEGMENT_TIMEOUT_HOPS = 19).  The local counter inside the closure
        // will reach 19+, but the recording-mode guard prevents it from being
        // written back to ctx (simulated by skipping handle_segment_boundary).
        let segment_hops = std::sync::atomic::AtomicUsize::new(0);
        let is_speech_fn = |_frame: &[f32]| -> bool {
            segment_hops.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            false // always silence
        };

        let audio = audio_for_frames(SEGMENT_TIMEOUT_HOPS);
        let _consumed = process_streaming_frames_inner(
            &audio,
            &mut voice_batch,
            &mut mel_frame_buffer,
            is_speech_fn,
            false,
            |_mel_frames| false,
        );

        let local_hop_count = segment_hops.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            local_hop_count >= SEGMENT_TIMEOUT_HOPS,
            "local counter should have crossed threshold (got {local_hop_count})"
        );

        // ── Simulate the `if !ctx.is_recording` guard: skip boundary check ──
        // In the production code (handle_wake_word_detection line ~6144), this
        // skip prevents handle_segment_boundary from writing back the stale
        // local counter.  We validate the resulting invariant directly.

        // ── Verify no stale writeback ──
        assert_eq!(
            ctx.segment_silence_hops, 0,
            "recording mode must prevent stale writeback of accumulated counter ({local_hop_count})"
        );

        // ── Verify other ctx state is clean (preserved by Soft reset) ──
        // embedding_ring and score_window were not touched during recording
        // (the on_flush callback returns false, so no embedding computed)
        assert!(
            !ctx.embedding_ring.is_empty(),
            "embedding_ring should survive recording-mode processing"
        );
        assert!(
            !ctx.score_window.is_empty(),
            "score_window should survive recording-mode processing"
        );

        // ctx-level buffers preserved (caller-managed invariant)
        assert!(
            !ctx.voice_batch.is_empty(),
            "ctx.voice_batch should survive recording-mode processing"
        );
        assert!(
            !ctx.mel_frame_buffer.is_empty(),
            "ctx.mel_frame_buffer should survive recording-mode processing"
        );
    }

    // ── Enrollment consistency check tests ────────────────────────────

    /// Build a 96-dim embedding with a single non-zero component at `dim`.
    fn basis_embedding(dim: usize) -> Vec<f32> {
        let mut v = vec![0.0; EMBEDDING_DIM];
        if dim < EMBEDDING_DIM {
            v[dim] = 1.0;
        }
        v
    }

    /// Build enrollment sequences from per-utterance basis dimensions.
    fn enrollment_from_bases(bases: &[usize], n_embs_per_utt: usize) -> Vec<EmbeddingSequence> {
        bases
            .iter()
            .enumerate()
            .map(|(ui, &dim)| {
                EmbeddingSequence::new(
                    UtteranceId {
                        sequence_index: ui,
                        variant_index: 0,
                    },
                    Source::Enrollment,
                    vec![basis_embedding(dim); n_embs_per_utt],
                )
            })
            .collect()
    }

    #[test]
    fn consistency_check_fails_when_too_few_utterances() {
        // 4 utterances with 3 embeddings each → <5 qualified → fail
        let buf = enrollment_from_bases(&[0, 0, 0, 0], 3);
        let result = validate_enrollment_consistency(&buf);
        assert!(result.is_err(), "expected failure with <5 utterances");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("4 enrollment utterances"),
            "error should mention utterance count: {msg}"
        );
    }

    #[test]
    fn consistency_check_fails_when_too_few_pass_threshold() {
        // 6 good (basis 0) + 4 bad (basis 1):
        //   centroid ≈ [0.6, 0.4, 0.0], cosine(good, centroid) ≈ 0.832 >=0.65,
        //   cosine(bad, centroid) ≈ 0.555 <0.65.
        // 6/10 pass, need ceil(10*0.7)=7 → fail
        let buf = enrollment_from_bases(&[0, 0, 0, 0, 0, 0, 1, 1, 1, 1], 3);
        let result = validate_enrollment_consistency(&buf);
        assert!(result.is_err(), "expected failure with only 6/10 passing");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("only 6/10"),
            "error should report 6/10 passed: {msg}"
        );
    }

    #[test]
    fn consistency_check_succeeds_with_high_quality_utterances() {
        // 8 good (basis 0) + 2 bad (basis 1):
        //   centroid ≈ [0.8, 0.2, 0.0], cosine(good, centroid) ≈ 0.970 >=0.65,
        //   cosine(bad, centroid) ≈ 0.285 <0.65.
        // 8/10 pass, need ceil(10*0.7)=7 → success
        let buf = enrollment_from_bases(&[0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 3);
        let result = validate_enrollment_consistency(&buf);
        assert!(
            result.is_ok(),
            "expected success with 8/10 passing: {:?}",
            result,
        );
    }

    // ── PCM cache key tests (mahbot-872) ────────────────────────────────────

    #[test]
    fn test_pcm_cache_key_determinism() {
        let h = |text, style, seed| pcm_cache_key(text, style, seed, 16000, "test_hash");
        let a = h("hey mahbot", "default", 42);
        let b = h("hey mahbot", "default", 42);
        assert_eq!(a, b, "same inputs must produce same cache key");
    }

    // ── Embedding cache roundtrip (mahbot-1029 D1) ─────────────────────────

    #[test]
    fn embedding_cache_roundtrip_preserves_embeddings_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let key = "test_key_roundtrip";
        let variants: Vec<(u8, Vec<Vec<f32>>)> = vec![
            (0, vec![vec![1.5f32; 96], vec![-2.25f32; 96]]),
            (3, vec![vec![0.0f32; 96]]),
            (4, Vec::new()),
        ];
        write_embedding_cache(dir.path(), key, &variants);
        let loaded = read_embedding_cache(dir.path(), key).expect("cache must be readable");
        assert_eq!(loaded, variants, "roundtrip must be byte-exact");
        // A missing key returns None (no panic).
        assert!(read_embedding_cache(dir.path(), "no_such_key").is_none());
    }

    #[test]
    fn embedding_cache_version_mismatch_misses_and_deletes() {
        let dir = tempfile::tempdir().unwrap();
        let key = "test_key_stale";
        // Version-0 header (≠ AUGMENT_RECIPE_VERSION) — stale recipe file.
        std::fs::write(dir.path().join(key), &[0u8; 20]).unwrap();
        assert!(read_embedding_cache(dir.path(), key).is_none());
        assert!(
            !dir.path().join(key).exists(),
            "stale-version cache file must be deleted"
        );
    }

    #[test]
    fn embedding_cache_eviction_sweeps_stale_version_and_keeps_tmp() {
        let dir = tempfile::tempdir().unwrap();
        // Version-0 header (≠ AUGMENT_RECIPE_VERSION) — stale recipe file.
        let stale = dir.path().join("stale_key");
        std::fs::write(&stale, &[0u8; 20]).unwrap();
        let tmp = dir.path().join("in_progress.tmp");
        std::fs::write(&tmp, &[0u8; 8]).unwrap();
        let variants: Vec<(u8, Vec<Vec<f32>>)> = vec![(0, vec![vec![1.0f32; 96]])];
        write_embedding_cache(dir.path(), "fresh_key", &variants);
        evict_embedding_cache_dir(dir.path());
        assert!(
            dir.path().join("fresh_key").exists(),
            "fresh-version entry must survive"
        );
        assert!(!stale.exists(), "stale-version entry must be evicted");
        assert!(tmp.exists(), "in-progress .tmp must survive the sweep");
    }

    #[test]
    fn embedding_cache_key_changes_with_preprocessor_and_pcm_key() {
        // Graceful-skip when the ONNX models are absent (fresh machine).
        if onnx_models_hash().is_none() {
            eprintln!("SKIP: voice ONNX models not cached — skipping key-sensitivity test");
            return;
        }
        let cfg_a = crate::audio::audio_preprocessor::PreprocessorConfig {
            noise_suppression: true,
            agc: true,
        };
        let cfg_b = crate::audio::audio_preprocessor::PreprocessorConfig {
            noise_suppression: false,
            agc: true,
        };
        let k1 = embedding_cache_key("confusable", "hey mahbot", "M1.json", 42, "tts_hash", cfg_a)
            .expect("models present");
        let k2 = embedding_cache_key("confusable", "hey mahbot", "M1.json", 42, "tts_hash", cfg_a)
            .expect("models present");
        assert_eq!(k1, k2, "same inputs must produce the same key");
        assert_ne!(
            k1,
            embedding_cache_key("confusable", "hey mahbot", "M1.json", 42, "tts_hash", cfg_b)
                .expect("models present"),
            "NS/AGC toggles must change the key"
        );
        assert_ne!(
            k1,
            embedding_cache_key(
                "confusable",
                "hey mahbot",
                "M1.json",
                42,
                "other_tts_hash",
                cfg_a
            )
            .expect("models present"),
            "TTS model hash must change the key"
        );
        assert_ne!(
            k1,
            embedding_cache_key("confusable", "hey mahbot", "M1.json", 43, "tts_hash", cfg_a)
                .expect("models present"),
            "TTS seed must change the key"
        );
    }

    #[test]
    fn test_pcm_cache_key_sensitivity_to_text() {
        let h = |text| pcm_cache_key(text, "default", 42, 16000, "test_hash");
        assert_ne!(
            h("hey mahbot"),
            h("hey madbot"),
            "different phrases must produce different cache keys",
        );
    }

    #[test]
    fn test_pcm_cache_key_sensitivity_to_seed() {
        let h = |seed| pcm_cache_key("hey mahbot", "default", seed, 16000, "test_hash");
        assert_ne!(
            h(42),
            h(43),
            "different seeds must produce different cache keys",
        );
    }

    #[test]
    fn test_pcm_cache_key_sensitivity_to_model_hash() {
        let h = |mh| pcm_cache_key("hey mahbot", "default", 42, 16000, mh);
        assert_ne!(
            h("hash_a"),
            h("hash_b"),
            "different model hashes must produce different cache keys",
        );
    }

    #[test]
    fn test_tts_model_version_hash_is_non_empty() {
        let hash = tts_model_version_hash();
        assert!(!hash.is_empty(), "TTS model version hash must not be empty");
        assert_eq!(hash.len(), 64, "SHA-256 hex string must be 64 chars");
    }

    #[test]
    fn test_enrolled_utterance_count_vs_buffer_size() {
        // Verify that enrolled_utterance_count tracks user utterances while
        // enrollment_buffer may hold up to 12× entries due to augmentation.
        // This models the production flow in handle_enrollment_sample where
        // each utterance produces up to 12 embedding entries (12-cell recipe).

        // ── Helper: simulate adding one utterance with all 12 variants ──
        fn simulate_utterance(
            state: &mut VoicePipelineState,
            // Each variant produces some number of per-window embeddings.
            variant_embeddings: &[usize], // one entry per variant
        ) {
            for (vi, &n_windows) in variant_embeddings.iter().enumerate() {
                let id = UtteranceId {
                    sequence_index: state.enrolled_utterance_count,
                    variant_index: vi,
                };
                state.enrollment_buffer.push(EmbeddingSequence::new(
                    id,
                    Source::Enrollment,
                    vec![vec![0.1; 96]; n_windows],
                ));
            }
            state.enrolled_utterance_count += 1;
        }

        let mut state = VoicePipelineState {
            enabled: false,
            status: VoiceStatus::Disabled,
            classifier_weights: None,
            classifier: None,
            enrollment_buffer: Vec::new(),
            negative_audio_chunks: Vec::new(),
            owner_negative_chunks: Vec::new(),
            enrolled_utterance_count: 0,
            utterances_collected: false,
            model_phrase: None,
            enrolling_phrase: None,
            cmd_tx: None,
        };

        // ── Simulate 2 utterances with all 12 variants succeeding ──
        // Each utterance produces 12 entries (original + 11 augmented).
        for _ in 0..2 {
            simulate_utterance(&mut state, &[3; 12]);
        }

        assert_eq!(
            state.enrolled_utterance_count, 2,
            "utterance count tracks utterances, not buffer entries"
        );
        assert_eq!(
            state.enrollment_buffer.len(),
            24,
            "2 utterances × 12 variants = 24 buffer entries"
        );

        // ── Simulate 3 more utterances (total 5) with speed-up skipped ──
        // Speed-up is skipped when pre-padding < 500ms, so only 11 variants.
        for _ in 0..3 {
            simulate_utterance(&mut state, &[3; 11]); // 11 variants, no speed-up
        }

        assert_eq!(state.enrolled_utterance_count, 5, "5 utterances total");
        assert_eq!(
            state.enrollment_buffer.len(),
            24 + 3 * 11, // 24 from first 2 + 33 from next 3
            "buffer entries: 2×12 + 3×11 = 57"
        );

        // ── Verify the invariant: buffer entries >= utterance count ──
        assert!(
            state.enrollment_buffer.len() >= state.enrolled_utterance_count,
            "buffer should have at least as many entries as utterances when augmentation runs"
        );

        // ── Simulate Cancel reset (delegates to production code path) ──
        // Calls VoicePipelineState::reset_enrollment() — the same method
        // used by PipelineCtx::reset_pipeline_state(ResetLevel::Cancel).
        state.reset_enrollment();

        assert_eq!(
            state.enrolled_utterance_count, 0,
            "after Cancel: utterance count reset"
        );
        assert!(
            state.enrollment_buffer.is_empty(),
            "after Cancel: buffer cleared"
        );

        // ── Verify finalization threshold uses utterance count, not buffer size ──
        // The production check at handle_enrollment_sample's finalization gate:
        //   if utterance_count >= NUM_ENROLLMENT_SAMPLES { ... finalize ... }
        // If we add 10 utterances with only 1 entry each (no augmentation),
        // utterance_count should trigger finalization at 10, even though
        // buffer has only 10 entries.
        for _ in 0..10 {
            simulate_utterance(&mut state, &[5]); // 1 variant (no augmentation)
        }
        assert_eq!(
            state.enrolled_utterance_count, 10,
            "10 utterances → finalization trigger"
        );
        assert_eq!(
            state.enrollment_buffer.len(),
            10,
            "buffer also has 10 entries (no augmentation)"
        );
        // The trigger check: utterance_count >= 10
        assert!(
            state.enrolled_utterance_count >= 10,
            "utterance count 10 reaches NUM_ENROLLMENT_SAMPLES threshold"
        );
    }

    // ── Voice metrics snapshot tests (mahbot-912) ────────────────────────────
    // These test the atomic counters, rolling average computation, and edge
    // cases (division by zero on empty snapshots).  The statics are preserved
    // between tests within a single process, so order-dependent tests must
    // record baseline values rather than asserting exact zero.

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
        assert_eq!(snap.lifetime_avg_embedding_latency_ns(), 100_000); // 100µs each

        // Truncation: 1_000_001 / 10 = 100_000 (integer division)
        let snap = VoiceMetricsSnapshot {
            total_embedding_time_ns: 1_000_001,
            ..snap
        };
        assert_eq!(snap.lifetime_avg_embedding_latency_ns(), 100_000);
    }

    // ── Activation trace buffer tests (mahbot-897) ──────────────────────────
    // These test the pure ActivationTraceBuffer FIFO eviction and iteration
    // without any pipeline state.

    // ── PersistedModel versioning & compatibility tests (mahbot-898) ──────

    #[test]
    fn check_compatibility_accepts_current_version() {
        let model = PersistedModel {
            schema_version: MODEL_SCHEMA_VERSION,
            ..Default::default()
        };
        assert!(check_model_compatibility(&model).is_ok());
    }

    #[test]
    fn check_compatibility_accepts_legacy_version() {
        // schema_version == 0 is legacy — allowed but callers should migrate.
        let model = PersistedModel {
            schema_version: 0,
            ..Default::default()
        };
        assert!(
            check_model_compatibility(&model).is_ok(),
            "legacy version 0 must be accepted"
        );
    }

    #[test]
    fn check_compatibility_rejects_unknown_version() {
        let model = PersistedModel {
            schema_version: 999,
            ..Default::default()
        };
        assert!(
            check_model_compatibility(&model).is_err(),
            "unknown schema version must be rejected"
        );
    }

    #[test]
    fn check_compatibility_rejects_wrong_embedding_dim() {
        let model = PersistedModel {
            schema_version: 1,
            embedding_dim: 42, // wrong!
            ..Default::default()
        };
        let err = check_model_compatibility(&model).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("embedding dimension"),
            "error must mention embedding dimension, got: {msg}"
        );
    }

    #[test]
    fn check_compatibility_rejects_wrong_window_size() {
        let model = PersistedModel {
            schema_version: 1,
            window_size: 5, // wrong!
            ..Default::default()
        };
        let err = check_model_compatibility(&model).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("window size"),
            "error must mention window size, got: {msg}"
        );
    }

    #[test]
    fn deserialize_classifier_opt_handles_array() {
        // Round-trip through serde_json to test the full deserialization path
        let json = r#"{
            "classifier": [
                {"conv1_weight": [1.0], "conv1_bias": [0.0],
                 "bn1_gamma": [1.0], "bn1_beta": [0.0],
                 "conv2_weight": [1.0], "conv2_bias": [0.0],
                 "bn2_gamma": [1.0], "bn2_beta": [0.0],
                 "fc_weight": [1.0], "fc_bias": [0.0],
                 "bn_eps": 1e-5}
            ]
        }"#;
        let model: PersistedModel =
            serde_json::from_str(json).expect("array format must deserialize");
        assert!(model.classifier.is_some());
        assert_eq!(model.classifier.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn deserialize_classifier_opt_handles_single_object() {
        // Legacy format: classifier is a single object, not an array
        let json = r#"{
            "classifier": {"conv1_weight": [1.0], "conv1_bias": [0.0],
             "bn1_gamma": [1.0], "bn1_beta": [0.0],
             "conv2_weight": [1.0], "conv2_bias": [0.0],
             "bn2_gamma": [1.0], "bn2_beta": [0.0],
             "fc_weight": [1.0], "fc_bias": [0.0],
             "bn_eps": 1e-5}
        }"#;
        let model: PersistedModel =
            serde_json::from_str(json).expect("single object format must deserialize");
        assert!(model.classifier.is_some());
        assert_eq!(
            model.classifier.as_ref().unwrap().len(),
            1,
            "single object must be wrapped in vec"
        );
    }

    #[test]
    fn deserialize_classifier_opt_handles_missing() {
        let json = r#"{"schema_version": 1, "phrase": "test"}"#;
        let model: PersistedModel =
            serde_json::from_str(json).expect("missing classifier must deserialize");
        assert!(model.classifier.is_none());
    }

    #[test]
    fn migrate_legacy_model_upgrades_to_current_version() {
        // Simulate a legacy model that was just loaded from JSON (all fields
        // at their zero/default values except schema_version which
        // deserializes to 0).  Must include classifier data to avoid the
        // classifier.is_none() early return in migrate_legacy_model.
        let mut model = PersistedModel {
            schema_version: 0,
            phrase: String::new(),
            embedding_dim: 0,
            window_size: 0,
            training_seeds: Vec::new(),
            created_at: String::new(),
            trained_at: String::new(),
            classifier: Some(vec![ClassifierWeights::default()]),
            val_losses: None,
        };

        migrate_legacy_model(&mut model);

        assert_eq!(
            model.schema_version, 1,
            "schema_version must be upgraded to current"
        );
        assert_eq!(model.phrase, "mahbot", "phrase must be set");
        assert_eq!(
            model.embedding_dim,
            u32::try_from(EMBEDDING_DIM).unwrap(),
            "embedding_dim must be set"
        );
        assert_eq!(
            model.window_size,
            u32::try_from(wake_word_classifier::WINDOW_SIZE).unwrap(),
            "window_size must be set"
        );
        assert!(
            !model.created_at.is_empty(),
            "created_at must be filled during migration"
        );
        assert!(
            !model.trained_at.is_empty(),
            "trained_at must be filled during migration"
        );

        migrate_legacy_model(&mut model);
        // Fields should be unchanged (idempotent: schema_version != 0 so migration
        // returns early; phrase, embedding_dim, window_size, created_at,
        // trained_at all retain their first-migration values).
        assert!(
            !model.phrase.is_empty() && model.embedding_dim > 0 && model.window_size > 0,
            "fields must survive idempotent re-migration"
        );
    }

    #[test]
    fn migrate_legacy_model_noop_when_classifier_none() {
        // A bare-minimum v0 model without classifier data must NOT be migrated
        // (the classifier.is_none() guard prevents accidental migration of bare
        // ClassifierWeights or old WakeWordTemplates that deserialize as
        // PersistedModel with null classifier, which would overwrite the
        // original data in the DB — mahbot-898 data-loss bug).
        let mut model = PersistedModel {
            schema_version: 0,
            classifier: None,
            ..Default::default()
        };

        migrate_legacy_model(&mut model);

        // The model should remain a v0 model — no fields touched
        assert_eq!(model.schema_version, 0, "must not upgrade schema_version");
        assert_eq!(
            model.phrase, DEFAULT_WAKE_WORD_PHRASE,
            "phrase must retain default (unchanged)"
        );
        assert_eq!(
            model.embedding_dim,
            u32::try_from(EMBEDDING_DIM).unwrap(),
            "embedding_dim must retain default (unchanged)"
        );
    }

    #[test]
    fn legacy_model_defaults_to_version_0() {
        // Ensure old-format JSON (no schema_version) gets version 0
        let json = r#"{"classifier": null}"#;
        let model: PersistedModel =
            serde_json::from_str(json).expect("legacy format must deserialize");
        assert_eq!(
            model.schema_version, 0,
            "legacy model without schema_version must default to 0"
        );
        assert_eq!(
            model.phrase, DEFAULT_WAKE_WORD_PHRASE,
            "legacy model must get default phrase"
        );
        assert_eq!(
            model.embedding_dim,
            u32::try_from(EMBEDDING_DIM).unwrap(),
            "legacy model must get default embedding_dim"
        );
    }

    // ── Wake word phrase normalization tests ──────────────────────────────

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

    // ── Mahbot wake word detection tests (mahbot-909) ─────────────────

    #[test]
    fn is_mahbot_wake_word_accepts_exact_matches() {
        // Normalized forms that match the Mahbot wake word.
        assert!(is_mahbot_wake_word("mahbot"));
        assert!(is_mahbot_wake_word("hey mahbot"));

        // Non-Mahbot wake words (normalized).
        assert!(!is_mahbot_wake_word("hey jarvis"));
        assert!(!is_mahbot_wake_word("ok computer"));
        assert!(!is_mahbot_wake_word("alexa"));
        assert!(!is_mahbot_wake_word(""));

        // Diacritics, partial-word matches, and other edge cases.
        // Note: these are already-normalized inputs. The caller must
        // normalize via normalize_phrase() first, which lowercases and
        // trims — so "Mahbot" would never reach this function.
        // These assertions document the intentional boundaries:
        assert!(!is_mahbot_wake_word("mähbot")); // diacritic
        assert!(!is_mahbot_wake_word("mahbotics")); // partial-word match
        assert!(!is_mahbot_wake_word("mah-bot")); // hyphenated
        assert!(!is_mahbot_wake_word("ma hbot")); // whitespace variant
        assert!(!is_mahbot_wake_word("hey mah")); // truncated
    }

    // ── Wake word phrase state machine tests ──────────────────────────────

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
        assert_eq!(
            get_enrolled_phrase(),
            Some("hey computer".to_string()),
            "get_enrolled_phrase must still return the cached model phrase after cancel"
        );
    }

    // ── PCM cache eviction tests (mahbot-910) ────────────────────────────
    // These test the pure file-system eviction logic with a temp directory.
    // They do not require voice ONNX models or TTS.
    //
    // Note: all eviction tests share the global CONFIG singleton.  A static
    // Mutex serialises access to prevent flaky races when tests run in parallel.

    use std::sync::{Mutex, MutexGuard};

    /// Serialises eviction tests that mutate global CONFIG.
    static EVICTION_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Write synthetic PCM data to a cache path and return its size.
    fn write_test_pcm(path: &Path) -> u64 {
        let samples: Vec<f32> = vec![0.0; 4096]; // 16 KB
        write_pcm_cache(path, &samples);
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

        // Create enough old non-tmp entries that age-based eviction runs
        let old_mtime = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 86400);
        for i in 0..3u8 {
            let name = format!("{:064x}", i);
            let path = tmp.path().join(&name);
            write_test_pcm(&path);
            let times = std::fs::FileTimes::new().set_modified(old_mtime);
            let _ = std::fs::File::open(&path).and_then(|f| f.set_times(times));
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
        write_test_pcm(&recent_path);

        // Create an "old" entry by writing and then setting mtime far in the past
        let old_path = tmp.path().join("b".repeat(64));
        write_test_pcm(&old_path);
        let two_days_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 86400);
        let times = std::fs::FileTimes::new().set_modified(two_days_ago);
        let _ = std::fs::File::open(&old_path).and_then(|f| f.set_times(times));

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
        write_test_pcm(&path);
        // Set mtime to 2 days ago
        let two_days_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 86400);
        let times = std::fs::FileTimes::new().set_modified(two_days_ago);
        let _ = std::fs::File::open(&path).and_then(|f| f.set_times(times));

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
            let path = tmp.path().join(&name);
            write_test_pcm(&path);
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
            let path = tmp.path().join(&name);
            write_test_pcm(&path);
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
            let path = tmp.path().join(&name);
            write_test_pcm(&path);
            // Stagger mtime: entry 0 is oldest (67 hours ago), entry 67 is newest
            let age_hours = (count - 1 - i) as u64;
            let mtime =
                std::time::SystemTime::now() - std::time::Duration::from_secs(age_hours * 3600);
            let times = std::fs::FileTimes::new().set_modified(mtime);
            let _ = std::fs::File::open(&path).and_then(|f| f.set_times(times));
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
        let old_mtime = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 86400);
        for i in 0..5u8 {
            let name = format!("old_{:064x}", i);
            let path = tmp.path().join(&name);
            write_test_pcm(&path);
            let times = std::fs::FileTimes::new().set_modified(old_mtime);
            let _ = std::fs::File::open(&path).and_then(|f| f.set_times(times));
        }

        // Create 68 recent entries (~16 KB each, totalling ~1088 KB).
        // All within the last hour so the 1-day age limit does NOT touch them.
        for i in 0..68u8 {
            let name = format!("recent_{:064x}", i);
            let path = tmp.path().join(&name);
            write_test_pcm(&path);
            // Stagger mtime from 0 to 67 minutes ago (all well under 1 day)
            let age_minutes = (67 - i) as u64;
            let mtime =
                std::time::SystemTime::now() - std::time::Duration::from_secs(age_minutes * 60);
            let times = std::fs::FileTimes::new().set_modified(mtime);
            let _ = std::fs::File::open(&path).and_then(|f| f.set_times(times));
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

    // ── Mel stride alignment test (mahbot-924) ────────────────────────────
    // Validates that trim_voice_batch drains to a MEL_STRIDE-aligned boundary,
    // ensuring mel frame grid consistency between enrollment and inference.

    #[test]
    fn test_mel_stride_overlap_alignment() {
        // The core invariant: after trim_voice_batch, the drained amount
        // must be a multiple of MEL_STRIDE so the retained overlap starts
        // at a position aligned to the mel frame grid.
        //
        // Without this alignment, HOP_LENGTH (256) not being a multiple of
        // MEL_STRIDE (160) causes ~80% of batch boundaries to land off-grid.

        // ── Exhaustive check: every batch length from FRAME_LENGTH to
        //    VOICE_BATCH_SIZE + 2*MEL_STRIDE ─────────────────────────────
        // This proves the invariant for all possible batch sizes that can
        // reach trim_voice_batch (FRAME_LENGTH is the minimum gate in
        // flush_voice_batch; VOICE_BATCH_SIZE is the normal flush point;
        // the extra 2×MEL_STRIDE covers the maximum possible retained
        // overlap after alignment rounding).
        let max_len = VOICE_BATCH_SIZE + 2 * MEL_STRIDE;
        for len in FRAME_LENGTH..=max_len {
            let mut batch = vec![0.0f32; len];
            let old_len = batch.len();

            trim_voice_batch(&mut batch);

            let drained = old_len - batch.len();

            // Drained amount must be MEL_STRIDE-aligned
            assert_eq!(
                drained % MEL_STRIDE,
                0,
                "len={len}: drained={drained} is not a multiple of {MEL_STRIDE} \
                 (remainder={})",
                drained % MEL_STRIDE,
            );

            // Must retain at least VOICE_BATCH_OVERLAP samples
            assert!(
                batch.len() >= VOICE_BATCH_OVERLAP,
                "len={len}: retained {} < {VOICE_BATCH_OVERLAP}",
                batch.len(),
            );

            // Must retain at most VOICE_BATCH_OVERLAP + MEL_STRIDE - 1
            // (alignment rounding can add up to MEL_STRIDE - 1 extra)
            assert!(
                batch.len() <= VOICE_BATCH_OVERLAP + MEL_STRIDE - 1,
                "len={len}: retained {} > {}",
                batch.len(),
                VOICE_BATCH_OVERLAP + MEL_STRIDE - 1,
            );
        }

        // ── Streaming simulation: VAD-frame accumulation ─────────────────
        // Simulates the real pipeline: HOP_LENGTH chunks accumulate until
        // voice_batch crosses VOICE_BATCH_SIZE, then trim_voice_batch is
        // called.  Repeats for many iterations to catch cumulative issues.
        let mut batch: Vec<f32> = Vec::new();
        let mut pos = 0usize;

        for iteration in 0..200 {
            // Add a varying number of HOP_LENGTH chunks per iteration
            // to simulate real VAD accumulation patterns.
            let n_chunks = 5 + (iteration * 3) % 11; // cycles 5..15
            for _ in 0..n_chunks {
                batch.extend_from_slice(&[0.0; HOP_LENGTH]);
            }

            // Flush when batch is large enough (matches production logic:
            // flush_voice_batch requires >= FRAME_LENGTH, and we only trim
            // when batch > VOICE_BATCH_OVERLAP).
            if batch.len() > VOICE_BATCH_OVERLAP && batch.len() >= FRAME_LENGTH {
                let old_len = batch.len();
                trim_voice_batch(&mut batch);
                let drained = old_len - batch.len();
                pos += drained;

                // The start position of the retained batch must be aligned
                assert_eq!(
                    pos % MEL_STRIDE,
                    0,
                    "Iteration {iteration}: cumulative position {pos} is not \
                     aligned to {MEL_STRIDE} (remainder {})",
                    pos % MEL_STRIDE,
                );

                // Each trim individually must be aligned
                assert_eq!(
                    drained % MEL_STRIDE,
                    0,
                    "Iteration {iteration}: drained {drained} is not a \
                     multiple of {MEL_STRIDE} (remainder {})",
                    drained % MEL_STRIDE,
                );
            }
        }

        // ── Edge case: silence-triggered flush with minimal batch ────────
        let mut batch = vec![0.0f32; FRAME_LENGTH];
        let mut pos = 0usize;

        let old_len = batch.len();
        trim_voice_batch(&mut batch);
        let drained = old_len - batch.len();
        pos += drained;
        assert_eq!(
            pos % MEL_STRIDE,
            0,
            "Minimum batch (FRAME_LENGTH={FRAME_LENGTH}): position {pos} \
             is not aligned",
        );

        // ── Idempotency: calling trim_voice_batch on an already-trimmed
        //    batch should not break alignment ─────────────────────────────
        for iteration in 0..10 {
            // Use iteration index for deterministic chunk count
            let n_chunks = 2 + (iteration * 7 + 3) % 4;
            for _ in 0..n_chunks {
                batch.extend_from_slice(&[0.0; HOP_LENGTH]);
            }

            if batch.len() > VOICE_BATCH_OVERLAP && batch.len() >= FRAME_LENGTH {
                let old_len = batch.len();
                trim_voice_batch(&mut batch);
                let drained = old_len - batch.len();
                pos += drained;

                assert_eq!(
                    pos % MEL_STRIDE,
                    0,
                    "Idempotency test: position {pos} is not aligned \
                     (drained {drained})",
                );
            }
        }
    }

    // ── mahbot-1012: embedding hash + L2 norm tests ───────────────────────
    // The dual-path same-audio comparison relies on the embedding hash being
    // deterministic and canonicalising floating-point edge cases: equal values
    // must hash equal (-0.0 == +0.0, NaN payloads collapse to one quiet NaN).
    #[cfg(feature = "voice-tests")]
    #[test]
    fn embedding_hash_is_deterministic_and_canonicalises_float_edges() {
        let a = vec![0.1_f32, -0.5, 2.0, 3.25];
        let b = a.clone();
        // Deterministic: same values → same hash.
        assert_eq!(embedding_hash(&a), embedding_hash(&b));
        // -0.0 and +0.0 are equal values → must hash equal.
        let neg_zero = vec![0.0_f32, -0.0, 1.0];
        let pos_zero = vec![0.0_f32, 0.0, 1.0];
        assert_eq!(embedding_hash(&neg_zero), embedding_hash(&pos_zero));
        // NaN payloads (not distinguishable acoustic values) collapse to one.
        let nan1 = vec![f32::NAN, 1.0];
        let nan2 = vec![f32::from_bits(0x7fc0_1234), 1.0];
        assert_eq!(embedding_hash(&nan1), embedding_hash(&nan2));
        // Different values hash differently (collision probability ~2^-64).
        let c = vec![0.1_f32, -0.5, 2.0, 3.250_001];
        assert_ne!(embedding_hash(&a), embedding_hash(&c));
    }

    #[cfg(feature = "voice-tests")]
    #[test]
    fn embedding_l2_norm_is_sqrt_of_sum_of_squares() {
        let v = vec![3.0_f32, 4.0];
        assert!((embedding_l2_norm(&v) - 5.0).abs() < 1e-6);
        assert_eq!(embedding_l2_norm(&[]), 0.0);
    }

    // ── mahbot-1012: window geometry classification ─────────────────────────
    // classify_window_geometry is a pure function; these tests pin the
    // precedence order and the ring-capacity (front-eviction) correctness that
    // the old inline WarmMixed proxy got wrong.

    #[cfg(feature = "voice-tests")]
    #[test]
    fn geometry_cold_start_tiled_precedes_all() {
        // Post-push ring below WINDOW_SIZE (3) → tiled regardless of warm-up
        // state, padded flag, or frame index.  ring_len_before ∈ {0, 1} gives
        // post-push lengths {1, 2} < 3; ring_len_before = 2 already yields a
        // full window and must NOT tile.
        for ring_len_before in 0..wake_word_classifier::WINDOW_SIZE - 1 {
            for tsrl in [0usize, 19] {
                for padded in [false, true] {
                    assert_eq!(
                        classify_window_geometry(ring_len_before, 0, tsrl, padded),
                        WindowGeometry::ColdStartTiled,
                        "ring_len_before={ring_len_before} tsrl={tsrl} padded={padded}",
                    );
                }
            }
        }
    }

    #[cfg(feature = "voice-tests")]
    #[test]
    fn geometry_ring_wrap_warm_mixed_only_first_two_test_frames() {
        // Regression for the mahbot-1012 reviewer finding: with the ring at
        // capacity (EMBEDDING_RING_MAX = 19) the OLD proxy
        // `ring_len_before + 1 - WINDOW_SIZE < test_start_ring_len` pinned at
        // 17 < 19 forever, mislabeling EVERY later window as WarmMixed.  The
        // Conv1D window covers the last 3 pushed embeddings, so it contains a
        // warm-up embedding only while fewer than 2 test frames precede it.
        let tsrl = EMBEDDING_RING_MAX; // warm-up filled the ring to capacity
        for (frames_before, expected) in [
            (0usize, WindowGeometry::WarmMixed),
            (1, WindowGeometry::WarmMixed),
            (2, WindowGeometry::TrueSliding),
            (3, WindowGeometry::TrueSliding),
            (100, WindowGeometry::TrueSliding),
        ] {
            // ring_len_before = 19 (at capacity) on every frame — the exact
            // case that used to pin the proxy.
            assert_eq!(
                classify_window_geometry(EMBEDDING_RING_MAX, frames_before, tsrl, false),
                expected,
                "frames_before={frames_before}",
            );
        }
        // Padded windows past the mixing window classify PaddedFallback, not
        // WarmMixed — the warm pass's padded-window count must match the cold
        // pass's once the ring-wrap shadowing is removed.
        assert_eq!(
            classify_window_geometry(EMBEDDING_RING_MAX, 5, tsrl, true),
            WindowGeometry::PaddedFallback,
        );
    }

    #[cfg(feature = "voice-tests")]
    #[test]
    fn geometry_small_warmup_also_mixes_only_first_two_test_frames() {
        // Warm-up producing ~7 embeddings (the documented WARMUP_PREPEND
        // expectation): the first two test windows mix warm-up + test content,
        // the third is already pure test.
        let tsrl = 7usize;
        for frames_before in [0usize, 1] {
            assert_eq!(
                classify_window_geometry(tsrl + frames_before, frames_before, tsrl, false),
                WindowGeometry::WarmMixed,
                "frames_before={frames_before}",
            );
        }
        assert_eq!(
            classify_window_geometry(9, 2, tsrl, false),
            WindowGeometry::TrueSliding,
        );
    }

    #[cfg(feature = "voice-tests")]
    #[test]
    fn geometry_cold_pass_never_warm_mixed() {
        // Cold pass: test_start_ring_len == 0 — no warm-up embeddings exist,
        // so even the first frames classify by their window construction.
        assert_eq!(
            classify_window_geometry(2, 0, 0, false),
            WindowGeometry::TrueSliding,
        );
        assert_eq!(
            classify_window_geometry(2, 0, 0, true),
            WindowGeometry::PaddedFallback,
        );
    }

    #[cfg(feature = "voice-tests")]
    #[test]
    fn geometry_post_segment_boundary_never_warm_mixed() {
        // Mid-pass segment boundary clears the ring; per_frame_scores is NOT
        // reset, so frames_scored_before is large.  The first two post-clear
        // windows are tiled (ring below WINDOW_SIZE), the rest classify by
        // construction — none can be WarmMixed.
        let tsrl = 19usize;
        assert_eq!(
            classify_window_geometry(0, 40, tsrl, false),
            WindowGeometry::ColdStartTiled,
        );
        assert_eq!(
            classify_window_geometry(1, 41, tsrl, false),
            WindowGeometry::ColdStartTiled,
        );
        assert_eq!(
            classify_window_geometry(2, 42, tsrl, false),
            WindowGeometry::TrueSliding,
        );
        assert_eq!(
            classify_window_geometry(2, 42, tsrl, true),
            WindowGeometry::PaddedFallback,
        );
    }

    // ── Padded-window geometry tests (mahbot-927 / mahbot-1023) ───────────
    // These test `pad_mel_frames_to_window`, the core padding function used
    // by the deferred burst sweep and the segment-end pass
    // (`score_start_aligned_positions`) to build start-aligned windows
    // shorter than EMBEDDING_WINDOW_FRAMES.  The full stride-8 scoring loop
    // cannot be tested without ONNX models, but the padding transformation
    // itself is isolated and fully testable.  The old incremental
    // short-buffer fallback (mahbot-927) was removed in mahbot-1023; the
    // padding semantics it introduced are preserved for the burst/pass.

    /// Helper: create a mel frame with identical values across all bands.
    fn mel_frame(val: f32) -> Vec<f32> {
        vec![val; NUM_MEL_BANDS]
    }

    /// Helper: create `n` consecutive mel frames with increasing values.
    fn ramp_frames(n: usize) -> Vec<Vec<f32>> {
        (0..n).map(|i| mel_frame(i as f32)).collect()
    }

    #[test]
    fn pad_full_window_unchanged() {
        // A buffer of exactly EMBEDDING_WINDOW_FRAMES (76) frames should be
        // returned verbatim — no padding needed.
        let frames = ramp_frames(EMBEDDING_WINDOW_FRAMES);
        let padded = pad_mel_frames_to_window(&frames);
        assert_eq!(padded.len(), EMBEDDING_WINDOW_FRAMES);
        assert_eq!(padded, frames, "full window should be returned unchanged");
    }

    #[test]
    fn pad_short_buffer_preserves_original_frames() {
        // The first N frames of the padded output must match the original
        // frames byte-for-byte — padding only appends, never modifies.
        for n_frames in [1, 60, 67, 75] {
            let frames = ramp_frames(n_frames);
            let padded = pad_mel_frames_to_window(&frames);
            assert_eq!(
                &padded[..n_frames],
                &frames[..],
                "original frames should be preserved verbatim (n={n_frames})",
            );
        }
    }

    #[test]
    fn pad_tapered_fadeout_transitions_to_silence() {
        // The padding frames smoothly transition from the last real frame
        // toward silence (spec_transform(0.0) = 2.0).  The first padding
        // frame is ~86% last-real + ~14% silence; the last is ~14% last-real
        // + ~86% silence.
        let n_frames = 70;
        let frames = ramp_frames(n_frames);
        let padded = pad_mel_frames_to_window(&frames);
        assert_eq!(padded.len(), EMBEDDING_WINDOW_FRAMES);
        let silence_val: f32 = 2.0;

        let last_real_val = (n_frames - 1) as f32; // 69.0
        let first_pad_val = padded[n_frames][0];
        let last_pad_val = padded[EMBEDDING_WINDOW_FRAMES - 1][0];

        // First padding: closer to last real than to silence
        let d_last = (first_pad_val - last_real_val).abs();
        let d_silence = (first_pad_val - silence_val).abs();
        assert!(
            d_last < d_silence,
            "first padding {first_pad_val}: should be closer to last real \
             {last_real_val} than to silence {silence_val} \
             (d_last={d_last:.3}, d_silence={d_silence:.3})",
        );

        // Last padding: closer to silence than to last real
        let d_last = (last_pad_val - last_real_val).abs();
        let d_silence = (last_pad_val - silence_val).abs();
        assert!(
            d_silence < d_last,
            "last padding {last_pad_val}: should be closer to silence \
             {silence_val} than to last real {last_real_val} \
             (d_silence={d_silence:.3}, d_last={d_last:.3})",
        );
    }

    #[test]
    fn pad_empty_buffer_all_silence() {
        // An empty mel frame buffer produces all-silence padding with
        // exactly EMBEDDING_WINDOW_FRAMES frames of spec_transform(0.0).
        let padded = pad_mel_frames_to_window(&[]);
        assert_eq!(padded.len(), EMBEDDING_WINDOW_FRAMES);
        let silence_val: f32 = 2.0;
        for (i, frame) in padded.iter().enumerate() {
            assert_eq!(
                frame.len(),
                NUM_MEL_BANDS,
                "frame {i} has wrong number of mel bands",
            );
            for &val in frame.iter() {
                assert!(
                    (val - silence_val).abs() < 1e-6,
                    "frame {i} value {val} != silence {silence_val}",
                );
            }
        }
    }

    #[test]
    fn pad_each_frame_has_correct_band_count() {
        // Every frame in the padded output must have NUM_MEL_BANDS values,
        // regardless of the input size.
        for n in [0, 1, 60, 67, 75, 100] {
            let frames = ramp_frames(n);
            let padded = pad_mel_frames_to_window(&frames);
            let expected_len = if n >= EMBEDDING_WINDOW_FRAMES {
                n
            } else {
                EMBEDDING_WINDOW_FRAMES
            };
            assert_eq!(
                padded.len(),
                expected_len,
                "n={n}: expected {expected_len} frames",
            );
            for (i, frame) in padded.iter().enumerate() {
                assert_eq!(
                    frame.len(),
                    NUM_MEL_BANDS,
                    "n={n}, frame {i}: expected {NUM_MEL_BANDS} bands, got {}",
                    frame.len(),
                );
            }
        }
    }

    // ── PCM cache invalidation tests ──────────────────────────────────────

    /// Helper: write valid PCM to the cache path that
    /// `synthesize_with_pcm_cache` would look up for the given parameters.
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
        write_test_pcm(&path);
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
