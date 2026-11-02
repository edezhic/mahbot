//! Voice Assistant — the microphone and the voice command pipeline, and, on macOS only, wake
//! word detection.
//!
//! Every platform has the microphone (cpal) and the pipeline that turns a recording into a
//! composer message: a press starts it, the press (or the ten-minute cap) ends it, and the
//! recording goes to the recogniser in [`crate::speech`], which is where transcription lives now.
//!
//! macOS adds the wake word: Earshot VAD → a 1024-dim [`encode_window`] embedding of the trailing
//! ≤1 s of VAD-gated speech, scored by cosine against the enrolled prototype → rolling-sum
//! detection → command recording → the same recognition → the composer message. Wake word reuses
//! the Qwen3-ASR encoder ([`crate::audio::encoder`]) and needs transcription enabled
//! (`audio_transcription_use_local != "false"`); the pipeline runs unconditionally as a boot
//! background task, with no agent loop.
//!
//! Each scoring step encodes the trailing [`WINDOW_SAMPLES`] (12 160 samples ≈ 0.76 s) of the
//! VAD-gated speech window, once per [`SCORE_STRIDE_SAMPLES`] (2560 ≈ 160 ms, 16 mel frames ×
//! 160-sample stride); the raw ring is capped at [`AUDIO_BUFFER_MAX`] (~2.0 s) and dropped on a
//! fire, and a fire counts only near the start of its speech segment
//! ([`WAKE_WORD_MAX_SEGMENT_STEPS`]).  Enrollment embeds ~10 utterances into their L2-normalized
//! mean (no trainable head) and calibrates the cosine floor on owner and ambient negatives; it
//! persists under `wake_word_templates` in the v2 schema, v1 records being rejected.

#[cfg(target_os = "macos")]
use crate::audio::cue::{self, Cue};
#[cfg(target_os = "macos")]
use crate::audio::util::cosine_similarity;
#[cfg(target_os = "macos")]
use crate::audio::wake_word::{
    self, ENROLLMENT_CONSISTENCY_MIN_FRACTION, ENROLLMENT_CONSISTENCY_MIN_SIMILARITY,
    MIN_ENROLLMENT_UTTERANCES, WAKE_WORD_EMBEDDING_DIM, WINDOW_SAMPLES, WakeWordEnrollment,
    calibrate_negatives, encode_window, normalized_centroid,
};
use crate::config::CONFIG;
#[cfg(target_os = "macos")]
use crate::config::CONFIG_KEY_WAKE_WORD_TEMPLATES;
#[cfg(target_os = "macos")]
use crate::db;
use crate::runtime_events::RuntimeEvent;
use crate::util::UnwrapPoison;
use crate::util::model_state::ModelState;
use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(target_os = "macos")]
use std::collections::VecDeque;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, error, info, warn};

/// Target sample rate: 16 kHz mono.
const SAMPLE_RATE: u32 = 16_000;

/// Convert a sample count to milliseconds (truncating integer division —
/// callers rely on exact threshold semantics).
#[cfg(target_os = "macos")]
#[must_use]
fn samples_to_ms(len: usize, rate: u32) -> u64 {
    (len as u64 * 1000) / u64::from(rate)
}

/// Frame size for VAD / quality frames (512 samples = 32ms at 16kHz).
#[cfg(target_os = "macos")]
const FRAME_LENGTH: usize = 512;

/// Hop between VAD frames (256 samples at 16 kHz): the VAD iteration stride and silence tracking.
/// The Qwen3-ASR mel frontend strides its own 160 samples (10 ms) — [`HOP_LENGTH`] does not
/// affect mel frame spacing.
#[cfg(target_os = "macos")]
const HOP_LENGTH: usize = 256;

/// Maximum command recording duration (10 minutes).
const MAX_RECORD_SECS: usize = 600;

/// Minimum silence duration before stopping a wake-word command recording.
#[cfg(target_os = "macos")]
const COMMAND_SILENCE_DURATION: Duration = Duration::from_millis(1500);

/// Silence threshold in samples, derived from [`COMMAND_SILENCE_DURATION`] × sample rate so it
/// cannot silently drift if either changes.
#[cfg(target_os = "macos")]
const SILENCE_THRESHOLD_SAMPLES: usize =
    (COMMAND_SILENCE_DURATION.as_millis() as usize * SAMPLE_RATE as usize) / 1000;

/// Enrollment/segmentation silence threshold (~304ms = 19 hops × 256 samples), aligned to
/// [`SEGMENT_TIMEOUT_HOPS`] so training and inference find the same utterance boundaries.
/// [`SILENCE_THRESHOLD_SAMPLES`] stays the command-recording one, where 1.5s suits phrasing.
#[cfg(target_os = "macos")]
const ENROLLMENT_SILENCE_THRESHOLD_SAMPLES: usize = SEGMENT_TIMEOUT_HOPS * HOP_LENGTH;

/// Silence threshold (200ms) before the "Keep silent to confirm…" hint — wider than one frame
/// (16ms) so the UI transitions even under scheduling jitter.
#[cfg(target_os = "macos")]
const SILENCE_UI_GATE_SAMPLES: usize = 200 * SAMPLE_RATE as usize / 1000;

/// Capacity of the microphone channel feeding the pipeline: 32 chunks × 512 samples ≈ 1 s.  A
/// full channel drops chunks at
/// [`try_send`](tokio::sync::mpsc::Sender::try_send) while the pipeline is blocked (encoder
/// inference in [`handle_wake_word_detection`]), so memory stays bounded.
///
/// The policy is drop-newest: buffered audio lags ~1 s but stays temporally contiguous, so VAD,
/// windowing and scoring still see consistent segments.  A wake word arriving entirely inside the
/// dropped span is missed, and the user simply repeats it.
const MIC_CHANNEL_CAPACITY: usize = 32;

/// Non-VAD audio before the enrollment "speak louder" warning (~5s); see the frame threshold
/// derived from it below.
#[cfg(target_os = "macos")]
const ENROLLMENT_NO_SPEECH_DURATION: Duration = Duration::from_secs(5);

/// The same window in frames, each iteration consuming [`HOP_LENGTH`] new samples, so the
/// threshold survives frame or hop changes.
#[cfg(target_os = "macos")]
const ENROLLMENT_NO_SPEECH_TIMEOUT_FRAMES: usize =
    (ENROLLMENT_NO_SPEECH_DURATION.as_millis() as usize * SAMPLE_RATE as usize)
        / (HOP_LENGTH * 1000);

/// Consecutive VAD-negative hops before a detection segment boundary fires (~300 ms at 16 ms/hop).
/// The boundary resets the rolling score window and adaptive threshold, so scores cannot
/// accumulate across separate utterances.
#[cfg(target_os = "macos")]
const SEGMENT_TIMEOUT_HOPS: usize = 19;

#[cfg(target_os = "macos")]
const NUM_ENROLLMENT_SAMPLES: usize = 10;

#[cfg(target_os = "macos")]
const MIN_NEGATIVE_AUDIO_LEN: usize = SAMPLE_RATE as usize / 2;

/// Maximum number of ambient negative audio chunks retained in memory.
#[cfg(target_os = "macos")]
const MAX_NEGATIVE_AUDIO_CHUNKS: usize = 100;

/// Maximum automatic retry cycles after a terminal model load failure.  One cycle is a full
/// load-or-download chain (up to 12 download attempts × 1.88 GB in the encoder's retry loop),
/// and the self-healing path stays inside this budget so a persistently failing download cannot
/// loop forever (retry → 12 attempts → Failed → 30 s → retry → …).  Past it the pipeline stops
/// auto-retrying: the user presses the GUI retry button
/// ([`VoiceCommand::RetryModelLoading`], which bypasses the budget) or restarts.
const MAX_AUTO_MODEL_RETRY_CYCLES: u32 = 3;

/// Target VAD-positive speech seconds for Phase 3 owner-negative collection: enough real audio
/// for calibration data and anti-prototypes while keeping the per-user phase short.
#[cfg(target_os = "macos")]
const NEGATIVES_TARGET_SECONDS: usize = 15;

/// Memory cap for owner-negative chunks (~22.5 s of 16 kHz f32 audio).
/// Auto-derived from [`NEGATIVES_TARGET_SECONDS`] (× 1.5 headroom).
#[cfg(target_os = "macos")]
const MAX_OWNER_NEGATIVE_SAMPLES: usize = SAMPLE_RATE as usize * NEGATIVES_TARGET_SECONDS * 3 / 2;

/// Wall-clock timeout for Phase 3 owner-negative collection.
#[cfg(target_os = "macos")]
const PHASE3_TIMEOUT_SECS: u64 = 120;

/// Clipping detection threshold for enrollment quality scoring.
#[cfg(target_os = "macos")]
const ENROLLMENT_QUALITY_CLIPPING_THRESHOLD: f32 = 0.999;

/// Minimum acceptable enrollment utterance duration in milliseconds.
#[cfg(target_os = "macos")]
pub(crate) const ENROLLMENT_QUALITY_DURATION_MIN_MS: u64 = 400;

/// Maximum acceptable enrollment utterance duration in milliseconds.
#[cfg(target_os = "macos")]
pub(crate) const ENROLLMENT_QUALITY_DURATION_MAX_MS: u64 = 2000;

/// Fraction of utterances that must pass the enrollment self-test for the model to be accepted.
#[cfg(target_os = "macos")]
const ENROLLMENT_QUALITY_SELF_TEST_MIN_FRACTION: f32 = 0.8;

#[cfg(target_os = "macos")]
const ENROLLMENT_PROMPTS: &[(&str, usize)] = &[
    ("Say it normally", 3),
    ("Say it a bit further from the mic", 3),
    ("Say it at a slightly different angle", 2),
    ("Say it with your normal morning voice", 2),
];

// Scoring geometry, re-calibrated for the 1024-dim cosine soft scores of
// [`crate::audio::wake_word::encode_window`] + [`Calibration::soft_score`]: the old thresholds
// scored a sigmoid Conv1D head, whose outputs sat far above these.  The floor is derived
// parametrically from the negative mean/std ([`Calibration::soft_floor`]) and anchored at
// [`POS_MATCH_COSINE_ANCHOR`] ([`Calibration::fire_soft`]), so a hotter negative pool never
// crushes recall.

/// Detection window: the trailing 0.76 s (12 160 samples) of the VAD-gated speech window — the
/// [`WINDOW_SAMPLES`] geometry enrollment encodes.
#[cfg(target_os = "macos")]
const WAKE_WORD_WINDOW_SAMPLES: usize = WINDOW_SAMPLES;

/// Scoring stride: 16 mel frames × 160-sample mel stride = 2560 samples ≈ 160 ms.  The encoder
/// forward is heavy, so the stride is widened to keep the real-time budget sane; the rolling
/// window (N=3) still spans ~0.5 s of speech.
#[cfg(target_os = "macos")]
const SCORE_STRIDE_SAMPLES: usize =
    crate::audio::wake_word::SCORE_STRIDE_MEL_FRAMES * qwen_asr::config::HOP_LENGTH;

/// Cap for the raw audio ring (~2.0 s): one [`WINDOW_SAMPLES`] window plus up to eight scoring
/// strides (8 × 2560), so the VAD loop always has recent audio to consume while older samples
/// drain.  Only the VAD reads it — the detection→recording handoff drops it wholesale.
#[cfg(target_os = "macos")]
const AUDIO_BUFFER_MAX: usize = WINDOW_SAMPLES + 8 * SCORE_STRIDE_SAMPLES;

/// Audio discarded at the start of a wake-word recording (0.2 s).  Everything captured before the
/// fire is dropped outright, and this covers the detection lag plus the start cue ([`Cue::Start`],
/// ~70 ms, which the still-open microphone captures).  The price is the opening of the command —
/// up to ~0.5 s when detection lands late — traded for keeping the trigger phrase out of the
/// recognition input; a trailing fragment survives if the phrase is still being spoken at the fire.
/// Fixed product behaviour — there is no switch.
#[cfg(target_os = "macos")]
const POST_FIRE_DISCARD_SAMPLES: usize = 200 * SAMPLE_RATE as usize / 1000;

/// Cooldown after a detection, so a rapid consecutive false trigger cannot follow.
#[cfg(target_os = "macos")]
const WAKE_WORD_COOLDOWN: Duration = Duration::from_secs(3);

/// One stride-gated scoring step, recorded in the activation trace: ~160 ms each
/// ([`SCORE_STRIDE_SAMPLES`]), so N=3 covers ~480 ms.
#[cfg(target_os = "macos")]
const ROLLING_WINDOW_N: usize = 3;

/// Maximum position at which a fire is accepted, in stride-gated scoring steps since the first
/// step of the current speech segment.  A segment starts at the first speech after
/// [`SEGMENT_TIMEOUT_HOPS`] of silence ends the previous one, ends when silence returns; one step
/// ≈ 160 ms, so six steps cover roughly the first 1.0 s of the segment.
///
/// Six is the owner's trade: the tighter window removes substantially more false fires.  Measured
/// price against the earlier eight-step (~1.3 s) window: recognition 43/50 against 46/50 (two of
/// the three lost commands were previously-firing wake-only clips, the third
/// command-plus-other-speech), synthetic false accepts 19 of 121 against 36 of 121, and 15 merged
/// fires (19 raw events) on the hour of real audio against 18 merged (22 raw).  Fixed product
/// behaviour — there is no switch.
#[cfg(target_os = "macos")]
const WAKE_WORD_MAX_SEGMENT_STEPS: usize = 6;

/// Number of recent per-frame scores to track for adaptive threshold
/// statistics. At ~160 ms per frame, N=15 covers ~2.4 seconds of context.
#[cfg(target_os = "macos")]
const ADAPTIVE_WINDOW_N: usize = 15;

/// Default k multiplier for the adaptive threshold (mean + k × std).
#[cfg(target_os = "macos")]
const ADAPTIVE_K_DEFAULT: f32 = 2.5;

/// Ceiling the adaptive threshold must never exceed.  It sits above the anchored positive range:
/// genuine 3-frame rolling sums stay ~1.5-2.5 across the whole derived floor range [0.55, 0.80],
/// so 2.60 leaves headroom for a genuine wake while capping threshold drift from noise.
#[cfg(target_os = "macos")]
const ADAPTIVE_CEILING: f32 = 2.60;

/// Bootstrap frames fed the static threshold while the adaptive window fills.
#[cfg(target_os = "macos")]
const ADAPTIVE_BOOTSTRAP_FRAMES: usize = 5;

/// Static (non-adaptive) geometry, co-derived from the enrollment calibration: the rolling-sum
/// firing threshold (`ROLLING_WINDOW_N × Calibration::fire_soft()`) and the per-frame reset line
/// (`Calibration::reset_soft()`).  Without calibration the legacy operating point applies
/// (threshold 1.65, reset ≈0.35).
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
struct DetectionGeometry {
    static_threshold: f32,
    reset: f32,
}

#[cfg(target_os = "macos")]
impl DetectionGeometry {
    fn for_enrollment(enrollment: Option<&WakeWordEnrollment>) -> Self {
        let default_cal = crate::audio::wake_word::Calibration::default();
        let cal = enrollment.map_or(&default_cal, |e| &e.calibration);
        #[expect(clippy::cast_precision_loss)] // ROLLING_WINDOW_N ≤ 3 — exact in f32
        Self {
            static_threshold: (ROLLING_WINDOW_N as f32) * cal.fire_soft(),
            reset: cal.reset_soft(),
        }
    }
}

/// Run one per-frame soft score through the rolling window and decide whether detection fires:
/// true when the rolling sum reaches the geometry static threshold (co-derived from the enrollment
/// calibration).  A score below [`DetectionGeometry::reset`] clears the window entirely, so noise
/// cannot accumulate slowly.  The window is NOT cleared on fire — the caller owns full pipeline
/// cleanup.  Pure: only `score_window` is modified, so it is testable without models or a pipeline.
#[cfg(target_os = "macos")]
fn process_wake_word_score(
    total_score: f32,
    score_window: &mut Vec<f32>,
    geometry: DetectionGeometry,
    adaptive_threshold_override: Option<f32>,
) -> (bool, f32) {
    let reset = geometry.reset;
    if total_score < reset {
        // Far from matching — clear the whole window, so noise cannot accumulate slowly.
        if !score_window.is_empty() {
            debug!(
                "Wake word match lost: total_score={total_score:.4} < reset ({reset:.4}) \
                 (window reset, had {} scores)",
                score_window.len(),
            );
        }
        score_window.clear();
        (false, 0.0)
    } else {
        score_window.push(total_score);
        while score_window.len() > ROLLING_WINDOW_N {
            score_window.remove(0);
        }

        let rolling_sum: f32 = score_window.iter().sum();
        let threshold = adaptive_threshold_override.unwrap_or(geometry.static_threshold);

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

/// Score one window embedding: the cosine soft score against the enrolled prototype
/// ([`WakeWordEnrollment::soft_score`]), fed or peeked into the adaptive threshold, then through
/// the rolling window of [`process_wake_word_score`].  Shared by the live pipeline
/// ([`handle_wake_word_detection`]) and the enrollment self-test ([`run_enrollment_self_test`]).
/// Scoring is speaker-blind and this function adds no gate of its own — the live pipeline
/// additionally requires the fire to sit within [`WAKE_WORD_MAX_SEGMENT_STEPS`] steps of the
/// segment start.
///
/// Returns `(fired, rolling_sum, total_score, effective_threshold)`: `rolling_sum` is 0.0 when the
/// window was just reset, `total_score` is 0.0 with no enrollment (detection then impossible), and
/// `effective_threshold` is the adaptive value post-bootstrap, else the static one.  `score_window`
/// persists across calls; `enrollment == None` disables detection and `adaptive_state == None` the
/// adaptive adjustment.
#[cfg(target_os = "macos")]
fn score_single_embedding(
    embedding: &[f32],
    enrollment: Option<&WakeWordEnrollment>,
    score_window: &mut Vec<f32>,
    mut adaptive_state: Option<&mut AdaptiveThresholdState>,
    adaptive_k: f32,
) -> (bool, f32, f32, f32) {
    // Cosine soft score against the enrolled prototype; with no enrollment the score is 0.0 and no
    // detection is possible.
    let total_score = enrollment.map_or(0.0, |enr| enr.soft_score(embedding));

    // Detection geometry is co-derived from the enrollment calibration (static firing threshold and
    // per-frame reset line); with no enrollment the default legacy calibration applies.
    let geometry = DetectionGeometry::for_enrollment(enrollment);

    // Feed only background scores to the adaptive threshold, so it learns the noise floor instead
    // of being contaminated by the scores it is meant to detect. Frames below `geometry.reset` are
    // clearly not the wake word; wake-word-like frames peek() instead, which returns the threshold
    // without updating statistics — otherwise high scores inflate the threshold and block the
    // detection they should allow. The same rule governs bootstrap: peek() answers `None` there, so
    // the static threshold stays in effect and a high-scoring utterance legitimately keeps
    // bootstrap alive. No wake-word-like score ever enters the statistics, so residual
    // contamination cannot survive the soft reset.
    let adaptive_override = adaptive_state.as_mut().and_then(|state| {
        if total_score < geometry.reset {
            state.feed(total_score, adaptive_k, geometry.static_threshold)
        } else {
            state.peek(adaptive_k, geometry.static_threshold)
        }
    });

    let effective_threshold = adaptive_override.unwrap_or(geometry.static_threshold);

    let (detected, rolling_sum) =
        process_wake_word_score(total_score, score_window, geometry, adaptive_override);

    // With the `voice-debug` feature, log every per-frame total_score, whether it passed the reset
    // threshold, and the resulting rolling sum; the gate keeps it zero-overhead otherwise.
    #[cfg(feature = "voice-debug")]
    {
        let passed_threshold = total_score >= geometry.reset;
        let below_note = if passed_threshold {
            ""
        } else {
            " (below reset threshold — window reset)"
        };
        info!("VOICE_DEBUG: total_score={total_score:.4}{below_note} rolling_sum={rolling_sum:.4}",);
    }

    // Immediate fire: no gate beyond the threshold crossing — the segment-position cut is the
    // caller.
    (detected, rolling_sum, total_score, effective_threshold)
}

/// VAD speech threshold, unified for both enrollment and streaming detection.
#[cfg(target_os = "macos")]
const VAD_THRESHOLD: f32 = 0.5;

/// Consecutive VAD-positive frames before `utterance_had_speech` during enrollment: 1, matching
/// streaming detection, which starts accumulating at its first positive frame.
#[cfg(target_os = "macos")]
const ENROLLMENT_VAD_CONSECUTIVE_REQUIRED: usize = 1;

/// Global Earshot detector.  Behind a mutex because `predict_f32` takes ~5-6 µs, so contention is
/// negligible; its internal state (768-sample ring buffer, pre-emphasis filter, 3-frame feature
/// context) must stay in sync with the stream.  Created once in [`init_global`].
#[cfg(target_os = "macos")]
static VAD_DETECTOR: OnceLock<std::sync::Mutex<earshot::Detector>> = OnceLock::new();

/// Whether wake word inference can run: it encodes on the wake word's own Qwen3-ASR encoder, so
/// this is whether that model is loaded.
#[cfg(target_os = "macos")]
#[must_use]
fn models_ready() -> bool {
    crate::audio::encoder::is_loaded()
}

/// Whether local transcription is disabled by config (`audio_transcription_use_local == "false"`).
/// Nothing loads the recogniser while it is disabled, so no recording can be transcribed —
/// callers must surface the configuration, not a loading state that cannot complete.
#[must_use]
pub fn is_transcription_disabled() -> bool {
    crate::config::CONFIG
        .audio_transcription_use_local()
        .as_deref()
        == Some("false")
}

/// Resolve the voice status from the recogniser's state — pure, shared by the pipeline startup block
/// and its periodic wake-up so the two can never drift: failed → [`VoiceStatus::ModelError`]
/// (terminal); ready → [`VoiceStatus::Listening`] when voice is enabled, [`VoiceStatus::Disabled`]
/// when not; loading → [`VoiceStatus::LoadingModels`] (the wake-up keeps polling until terminal);
/// uninitialised → [`VoiceStatus::Disabled`], since the recogniser is a first-use load.
fn resolved_model_status(state: ModelState, voice_enabled: bool) -> VoiceStatus {
    match state {
        ModelState::Failed => VoiceStatus::ModelError,
        ModelState::Loading => VoiceStatus::LoadingModels,
        ModelState::Ready => {
            // Listening is the wake word's own state, and only macOS has a wake word.
            if voice_enabled && cfg!(target_os = "macos") {
                VoiceStatus::Listening
            } else {
                VoiceStatus::Disabled
            }
        }
        ModelState::Uninit => VoiceStatus::Disabled,
    }
}

/// Status resolved from the live recogniser state machine.
fn current_model_status() -> VoiceStatus {
    resolved_model_status(crate::speech::state(), is_enabled())
}

/// Demand the recogniser for a recording and report the load it starts: nothing preloads it, so a
/// demand is the first use, and without this the footer would show nothing while it loads. A no-op
/// once ready. The wake word's encoder is a loader of its own and is demanded where listening
/// starts ([`demand_encoder`]).
fn demand_transcriber() {
    crate::speech::spawn_background_init();
    if crate::speech::is_loading() {
        set_status(VoiceStatus::LoadingModels);
    }
}

/// Demand the wake word's encoder, which is what listening needs: its weights are a folder of
/// their own, so a button recording — which needs only the recogniser — must not fetch them.
#[cfg(target_os = "macos")]
fn demand_encoder() {
    crate::audio::encoder::spawn_background_init();
    if crate::audio::encoder::is_loading() {
        set_status(VoiceStatus::LoadingModels);
    }
}

/// Retry both loaders after a terminal failure: the recogniser on every platform, and the wake
/// word's encoder on macOS. `true` when either took the retry.
fn retry_model_loads() -> bool {
    let recogniser = crate::speech::retry_init();
    #[cfg(target_os = "macos")]
    let encoder = crate::audio::encoder::retry_init();
    #[cfg(not(target_os = "macos"))]
    let encoder = false;
    recogniser || encoder
}

/// Whether a loader the pipeline depends on is terminally failed: the recogniser always, and the
/// wake word's encoder while the wake word is on — an encoder nobody asked for must not put the
/// composer into an error state.
fn models_failed() -> bool {
    #[cfg(target_os = "macos")]
    if is_enabled() && crate::audio::encoder::is_failed() {
        return true;
    }
    crate::speech::is_failed()
}

#[derive(Debug, Clone, PartialEq)]
pub enum VoiceStatus {
    Disabled,
    LoadingModels,
    ModelError,
    Listening,
    /// A recording the wake word started: it ends itself on silence, and the composer ends it too.
    #[cfg(target_os = "macos")]
    RecordingWakeWord,
    /// A recording started from the composer mic button: same mechanism, no silence end, and
    /// wake-word listening paused while it runs.
    RecordingButton,
    Transcribing,
    MicPermissionDenied,
    /// The microphone went away while voice was in use.  Voice stops (no recording left hanging, no
    /// enrollment surviving) and no device is acquired again automatically: the user recovers by
    /// pressing record or toggling voice.
    MicDisconnected,
    /// Enrolled, waiting for the next sample (`sample` of `total`, `duration_ms` of the last
    /// utterance, per-utterance `quality`).
    #[cfg(target_os = "macos")]
    Enrolling {
        sample: usize,
        total: usize,
        duration_ms: u64,
        quality: Option<UtteranceQuality>,
    },
    #[cfg(target_os = "macos")]
    ListeningDuringEnrollment {
        sample: usize,
        total: usize,
    },
    #[cfg(target_os = "macos")]
    WaitingForSilenceDuringEnrollment {
        sample: usize,
        total: usize,
    },
    /// Collecting owner-general speech as negatives (Phase 3): `accumulated_secs` of VAD-positive
    /// speech towards `target_secs`, `wall_clock_elapsed` since the phase started.
    #[cfg(target_os = "macos")]
    EnrollingNegatives {
        accumulated_secs: usize,
        target_secs: usize,
        wall_clock_elapsed: u64,
    },
    #[cfg(target_os = "macos")]
    Enrolled,
    Error(String),
}

impl VoiceStatus {
    /// Whether the pipeline owns the microphone for a recording or a transcription, so the
    /// composer's mic button cannot offer a start that would be refused. (A recording the composer
    /// owns renders its own dot, not the button.)
    #[must_use]
    pub fn mic_busy(&self) -> bool {
        // The wake word's recording is a status only macOS produces.
        #[cfg(target_os = "macos")]
        if matches!(self, Self::RecordingWakeWord) {
            return true;
        }
        matches!(self, Self::RecordingButton | Self::Transcribing)
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityLevel {
    Good,       // score > 0.7
    Acceptable, // score 0.4–0.7
    Poor,       // score < 0.4
}

#[cfg(target_os = "macos")]
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

    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Good => "✅ Good sample — clear and consistent",
            Self::Acceptable => "⚠️ Acceptable — a bit quiet, try speaking closer to the mic",
            Self::Poor => "❌ Poor sample — too much noise, please re-record",
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, PartialEq)]
pub struct UtteranceQuality {
    pub score: f32,
    pub level: QualityLevel,
    pub clipping_detected: bool,
    pub duration_ms: u64,
    /// Estimated SNR in dB: 20*log10(speech_rms / noise_rms) when a pre-speech noise RMS exists
    /// (unbounded, typically 10–50 dB in quiet rooms), else 0–40 dB from the energy-based fallback,
    /// or NaN for an utterance too short to estimate.
    pub snr_db: f32,
}

static VOICE_PIPELINE: OnceLock<RwLock<VoicePipelineState>> = OnceLock::new();

struct VoicePipelineState {
    enabled: bool,
    status: VoiceStatus,
    /// Enrolled wake-word prototype + calibration (v2 schema).  `None` before
    /// a successful enrollment.
    #[cfg(target_os = "macos")]
    enrollment: Option<WakeWordEnrollment>,
    /// Per-utterance 1024-dim L2-normalized window embeddings collected
    /// during enrollment (one entry per accepted utterance).
    #[cfg(target_os = "macos")]
    enrollment_embeddings: Vec<Vec<f32>>,
    /// Ambient non-wake-word chunks (pre-enrollment noise and inter-utterance silence), encoded at
    /// finalization into real negative examples.
    #[cfg(target_os = "macos")]
    negative_audio_chunks: Vec<Vec<f32>>,
    /// Owner-negative chunks from Phase 3 (~15 s of VAD-positive general speech), embedded at
    /// finalization; preserved across Full/Soft resets, cleared on Cancel.
    #[cfg(target_os = "macos")]
    owner_negative_chunks: Vec<Vec<f32>>,
    /// Whether Phase 2 finished (all [`NUM_ENROLLMENT_SAMPLES`] utterances in): the main loop reads
    /// it to start the Phase 2→3 transition. Cleared on Cancel.
    #[cfg(target_os = "macos")]
    utterances_collected: bool,
    /// Accepted utterances so far, read by the UI counter and the finalization trigger.
    #[cfg(target_os = "macos")]
    enrolled_utterance_count: usize,
    cmd_tx: Option<mpsc::UnboundedSender<VoiceCommand>>,
}

#[cfg(target_os = "macos")]
impl VoicePipelineState {
    /// Clear the enrollment accumulators (embeddings + counter); called on [`ResetLevel::Cancel`]
    /// and by tests.
    fn reset_enrollment(&mut self) {
        self.enrollment_embeddings.clear();
        self.negative_audio_chunks.clear();
        self.owner_negative_chunks.clear();
        self.enrolled_utterance_count = 0;
        self.utterances_collected = false;
    }
}

#[derive(Debug)]
pub enum VoiceCommand {
    #[cfg(target_os = "macos")]
    StartListening,
    StopListening,
    #[cfg(target_os = "macos")]
    StartEnrollment,
    #[cfg(target_os = "macos")]
    CancelEnrollment,
    /// Retry loading the models after a terminal failure (the recogniser, and the wake word's
    /// encoder on macOS).
    RetryModelLoading,
    /// Start a recording from the composer mic button, pausing wake-word listening meanwhile.
    StartRecording,
    /// Stop the recording in progress — whichever way it started — and transcribe it; the
    /// transcript is published on [`RECORDING`] for the composer to send with its own content.
    StopRecordingSend,
    /// Stop the recording in progress and discard the captured audio.
    StopRecordingDiscard,
    /// Voice-relevant config changed (e.g. the Settings transcription toggle): wakes the loop when
    /// the periodic timer is parked because transcription is disabled, so it re-reads config and
    /// re-resolves status / auto-start.
    ConfigChanged,
}

fn voice_state() -> &'static RwLock<VoicePipelineState> {
    VOICE_PIPELINE.get().expect("VoicePipeline not initialized")
}

/// Initialize the voice pipeline state. Called during startup.
pub fn init_global() -> Result<()> {
    // The VAD is the wake word's own, on macOS.
    #[cfg(target_os = "macos")]
    VAD_DETECTOR.get_or_init(|| std::sync::Mutex::new(earshot::Detector::default()));

    VOICE_PIPELINE
        .set(RwLock::new(VoicePipelineState {
            enabled: false,
            status: VoiceStatus::Disabled,
            #[cfg(target_os = "macos")]
            enrollment: None,
            #[cfg(target_os = "macos")]
            enrollment_embeddings: Vec::new(),
            #[cfg(target_os = "macos")]
            negative_audio_chunks: Vec::new(),
            #[cfg(target_os = "macos")]
            owner_negative_chunks: Vec::new(),
            #[cfg(target_os = "macos")]
            enrolled_utterance_count: 0,
            #[cfg(target_os = "macos")]
            utterances_collected: false,
            cmd_tx: None,
        }))
        .map_err(|_| anyhow!("VoicePipeline already initialized"))?;

    Ok(())
}

#[must_use]
pub fn get_status() -> VoiceStatus {
    voice_state().read().unwrap_poison().status.clone()
}

/// How the recording in progress started: one mechanism and one message either way, differing only
/// in how it can end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecordingOrigin {
    /// Started from the composer mic button; only the composer controls end it.
    Button,
    /// Started by the wake word; it also ends itself on silence.
    #[cfg(target_os = "macos")]
    WakeWord,
}

/// Progress of a recording, delivered to the composer: whichever way it started, the composer
/// owns the message it becomes, from the start until this resolves it.
#[derive(Debug, Clone)]
pub enum RecordingEvent {
    /// A wake-word recording began: the field belongs to the message it will become. A button
    /// recording needs no event — the press opens the window itself. The composer drops a recording
    /// that starts while an earlier message is committed: the two never share one window.
    Started,
    /// Recording stopped, transcription starting: the composer takes the field content now, and the
    /// message is in flight until the transcript joins it.
    Stopped,
    /// The transcript, already carrying the spoken marker.
    Transcript(String),
    /// The recording held no speech.
    NoSpeech,
    /// The recording produced no message; the notice says why and a composer holding a committed
    /// message gives the field back.
    Failed(&'static str),
    /// Nothing was recorded (no microphone, transcription unavailable): the composer window closes
    /// and a message a press already committed goes out with the field's own content, the caller
    /// having reported the reason.
    NotStarted,
}

/// Broadcast of recording progress, warmed from `main` before the iced app runs (like the other GUI
/// senders): an uninitialized sender would end the composer subscription on its first poll.
pub(crate) static RECORDING: OnceLock<broadcast::Sender<RecordingEvent>> = OnceLock::new();

/// Initialise the recording broadcast (see [`RECORDING`]).
pub(crate) fn init_recording_tx() {
    RECORDING.get_or_init(|| broadcast::channel(16).0);
}

/// Publish recording progress; a no-op when the channel is not initialised (tests, headless).
fn publish_recording(event: RecordingEvent) {
    if let Some(tx) = RECORDING.get() {
        let _ = tx.send(event);
    }
}

/// Broadcast a composer-facing notice in the admin conversation, as the pipeline failures do.
pub(crate) async fn broadcast_notice(message: &str) {
    let user_name = crate::users::ADMIN_USER_NAME;
    let ws = crate::users::resolve_workspace_for_user_name(user_name).await;
    // Assistant conversations live in the admin's personal workspace.
    let ws = crate::users::effective_workspace_for_role(crate::Role::Assistant, ws, user_name);
    crate::channels::broadcast_and_persist_agent_response(
        user_name,
        "voice",
        message,
        "voice".to_string(),
        &ws.name,
        None,
    )
    .await;
}

/// Notice for [`VoiceStatus::MicPermissionDenied`]: the fix is on the user side.
const MIC_PERMISSION_NOTICE: &str = "Microphone permission denied — enable mic access to record";

/// Notice when a recording held no speech and the field had nothing to send either.
pub(crate) const NO_SPEECH_NOTICE: &str = "*Voice: no speech detected — recording discarded*";

/// Notice when a transcript cannot be sent because its message never went out: the words exist
/// only in that message.
pub(crate) const VOICE_UNSENT_NOTICE: &str =
    "*Voice: the voice part was not sent and cannot be restored*";

/// Notice when a recording transcription fails.
pub(crate) const TRANSCRIPTION_FAILED_NOTICE: &str =
    "*Voice: transcription failed — the voice part was not sent and cannot be restored*";

/// Window in which at most one voice notice of a class reaches the conversation. Each side keeps
/// its own memory of the last one and they share this interval, so a rapid repeat cannot become a
/// stream.
pub(crate) const NOTICE_INTERVAL: Duration = Duration::from_secs(10);

/// Best-effort reason a recording cannot start (None = allowed), mirroring the field guards in
/// [`PipelineCtx::handle_start_button_recording`] for the GUI pre-flight toast; the pipeline stays
/// authoritative and this can drift on transient transitions.  The transcription-disabled
/// configuration is permanent, so it is checked here directly.
#[must_use]
pub fn recording_blocked_reason() -> Option<&'static str> {
    // Local transcription disabled ⇒ the shared ASR model never loads ⇒ no recording can start:
    // surface the configuration, not a loading state that can never complete.
    if is_transcription_disabled() {
        return Some("Voice recording unavailable — local transcription is disabled");
    }
    match get_status() {
        VoiceStatus::Transcribing => Some("A voice message is already being processed"),
        #[cfg(target_os = "macos")]
        VoiceStatus::RecordingWakeWord => Some("A voice message is already being processed"),
        #[cfg(target_os = "macos")]
        VoiceStatus::Enrolling { .. }
        | VoiceStatus::ListeningDuringEnrollment { .. }
        | VoiceStatus::WaitingForSilenceDuringEnrollment { .. }
        | VoiceStatus::EnrollingNegatives { .. } => Some("Voice enrollment is in progress"),
        VoiceStatus::MicPermissionDenied => Some(MIC_PERMISSION_NOTICE),
        // MicDisconnected is not blocked — retrying is how the user recovers.
        _ => None,
    }
}

#[must_use]
fn is_enabled() -> bool {
    voice_state().read().unwrap_poison().enabled
}

pub fn set_enabled(enabled: bool) {
    let mut state = voice_state().write().unwrap_poison();
    state.enabled = enabled;
    if !enabled {
        set_status_and_notify(&mut state, VoiceStatus::Disabled);
    }
}

fn set_status(status: VoiceStatus) {
    set_status_and_notify(&mut voice_state().write().unwrap_poison(), status);
}

/// Write a new status, broadcasting [`RuntimeEvent::VoiceStatus`] only on a real change: enrollment
/// variants (macOS) update per sample and would otherwise emit continuously.
fn set_status_and_notify(state: &mut VoicePipelineState, status: VoiceStatus) {
    if state.status != status {
        state.status = status;
        crate::runtime_events::publish(RuntimeEvent::VoiceStatus);
    }
}

/// The currently active wake word enrollment (prototype + calibration).
#[cfg(target_os = "macos")]
#[must_use]
pub(crate) fn get_enrollment() -> Option<WakeWordEnrollment> {
    voice_state().read().unwrap_poison().enrollment.clone()
}

#[cfg(target_os = "macos")]
fn set_enrollment(enrollment: WakeWordEnrollment) {
    let mut state = voice_state().write().unwrap_poison();
    state.enrollment = Some(enrollment);
}

/// Queue a command, returning whether a pipeline is running to take it.  A no-op before the
/// pipeline initialises — the GUI controls exist earlier and a press must not panic.
pub fn send_command(cmd: VoiceCommand) -> bool {
    let Some(state) = VOICE_PIPELINE.get() else {
        warn!("Voice pipeline not initialized — dropping command {cmd:?}");
        return false;
    };
    let Some(tx) = state.read().unwrap_poison().cmd_tx.clone() else {
        warn!("Voice pipeline not running — dropping command {cmd:?}");
        return false;
    };
    tx.send(cmd).is_ok()
}

/// Test seam: stand in for a running pipeline command receiver, so [`send_command`] reports its
/// commands taken.
#[cfg(test)]
pub(crate) fn install_test_command_receiver() -> mpsc::UnboundedReceiver<VoiceCommand> {
    let _ = init_global();
    let (tx, rx) = mpsc::unbounded_channel();
    voice_state().write().unwrap_poison().cmd_tx = Some(tx);
    rx
}

/// Inner VAD check on an explicit detector with a configurable threshold —
/// [`is_speech_with_threshold`] passes the shared detector with [`VAD_THRESHOLD`],
/// [`enrollment_is_speech`] a context-local one so a mode transition cannot contaminate the
/// enrollment decision. Every 256-sample chunk goes through the detector, even once speech is
/// found, to keep its ring buffer and pre-emphasis state in sync with the stream.
#[cfg(target_os = "macos")]
fn is_speech_with_detector(
    samples: &[f32],
    detector: &mut earshot::Detector,
    threshold: f32,
) -> bool {
    if samples.is_empty() {
        return false;
    }

    let mut any_speech = false;

    // Clamp samples to [-1, 1] before the detector: TTS audio and capture can overflow slightly,
    // which trips earshot debug_assert! in debug builds.
    let clamp_frame = |frame: &[f32]| -> [f32; 256] {
        let mut clamped = [0.0f32; 256];
        for (i, &s) in frame.iter().enumerate() {
            clamped[i] = s.clamp(-1.0, 1.0);
        }
        clamped
    };

    // Process each complete 256-sample frame (Earshot requires exactly 256 at 16 kHz). Callers pass
    // one [`HOP_LENGTH`] hop; longer slices are chunked so the detector sliding window stays in
    // sync.
    for chunk in samples.as_chunks::<256>().0 {
        if detector.predict_f32(&clamp_frame(chunk)) >= threshold {
            any_speech = true;
        }
    }

    // Trailing partial frame (<256 samples): pad with silence rather than discard the tail of a
    // short burst. Earshot rejects silence-padded frames as non-speech, so padding is safe.
    let remainder = samples.len() % 256;
    if remainder > 0 {
        let mut padded = [0.0f32; 256];
        padded[..remainder].copy_from_slice(&samples[samples.len() - remainder..]);
        for s in &mut padded[..remainder] {
            *s = s.clamp(-1.0, 1.0);
        }
        if detector.predict_f32(&padded) >= threshold {
            any_speech = true;
        }
    }

    any_speech
}

/// VAD check at a configurable threshold: locks the global [`VAD_DETECTOR`].
#[cfg(target_os = "macos")]
fn is_speech_with_threshold(samples: &[f32], threshold: f32) -> bool {
    let detector = VAD_DETECTOR.get_or_init(|| std::sync::Mutex::new(earshot::Detector::default()));
    let mut detector = detector.lock().unwrap_poison();
    is_speech_with_detector(samples, &mut detector, threshold)
}

/// Enrollment/negatives VAD decision: prefer the context-local detector so a mode transition
/// cannot contaminate it; fall back to the global one when enrollment VAD is not initialised.
#[cfg(target_os = "macos")]
fn enrollment_is_speech(
    enrollment_vad: &mut Option<earshot::Detector>,
    threshold: f32,
    hop: &[f32],
) -> bool {
    match enrollment_vad {
        Some(det) => is_speech_with_detector(hop, det, threshold),
        None => is_speech_with_threshold(hop, threshold),
    }
}

/// Reset the detector internal state (ring buffer, feature context) when the audio source changes,
/// so stale context cannot contaminate a new stream.
#[cfg(target_os = "macos")]
fn reset_vad() {
    if let Some(detector) = VAD_DETECTOR.get()
        && let Ok(mut d) = detector.lock()
    {
        d.reset();
    }
}

/// Whether a microphone error is OS-level permission denial rather than a transient device issue.
/// macOS CoreAudio returns `kAudioUnitErr_NoConnection` (-10875) without microphone access;
/// common error-text patterns are checked too, so the user sees `MicPermissionDenied` rather than
/// a generic `MicDisconnected`.
fn is_mic_permission_error(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}");
    msg.contains("NoConnection")
        || msg.contains("-10875")
        || msg.contains("permission")
        || msg.contains("denied")
        || msg.to_lowercase().contains("access denied")
}

/// Status for a microphone that could not open: permission problem, or device unavailable.
fn mic_start_failure(err: &anyhow::Error) -> VoiceStatus {
    if is_mic_permission_error(err) {
        VoiceStatus::MicPermissionDenied
    } else {
        VoiceStatus::MicDisconnected
    }
}

/// Convert raw samples to mono f32 and send them to the pipeline in one pass, avoiding the
/// intermediate `Vec<f32>` that convert-then-mono would allocate — this runs at audio interrupt
/// frequency.  `T = f32` with the identity closure is the F32 path, handled identically.
fn convert_and_send_audio_to_pipeline<T, F>(
    tx: &mpsc::Sender<Vec<f32>>,
    data: &[T],
    channels: u16,
    sample_rate: u32,
    convert: F,
) where
    F: Fn(&T) -> f32,
{
    let mono: Vec<f32> = if channels == 1 {
        data.iter().map(&convert).collect()
    } else {
        data.chunks_exact(usize::from(channels))
            .map(|frame| frame.iter().map(&convert).sum::<f32>() / f32::from(channels))
            .collect()
    };
    let resampled = if sample_rate == SAMPLE_RATE {
        mono
    } else {
        super::util::resample_audio(&mono, sample_rate, SAMPLE_RATE)
    };
    // Drop-newest when the bounded channel is full (see [`MIC_CHANNEL_CAPACITY`]).
    let _ = tx.try_send(resampled);
}

/// Error callback: report the error, and when no audio can be delivered any more (device removed,
/// host gone, stream invalidated and never rebuilt) report that on the stream's loss channel.
/// Everything else keeps voice running: buffer overruns and underruns recover, a busy or rerouted
/// device recovers, and a permission problem keeps its [`VoiceStatus::MicPermissionDenied`]
/// wording.  The stream owns the channel (see [`Microphone`]), so a late report from a replaced
/// stream dies with it.
fn mic_error(err: &cpal::Error, lost_tx: &mpsc::UnboundedSender<()>) {
    error!("Microphone stream error: {err}");
    if matches!(
        err.kind(),
        cpal::ErrorKind::DeviceNotAvailable
            | cpal::ErrorKind::HostUnavailable
            | cpal::ErrorKind::StreamInvalidated
    ) {
        let _ = lost_tx.send(());
    }
}

fn build_int_stream<T, F>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_tx: &Arc<mpsc::Sender<Vec<f32>>>,
    lost_tx: &mpsc::UnboundedSender<()>,
    channels: u16,
    sample_rate: u32,
    convert: F,
) -> Result<cpal::Stream, cpal::Error>
where
    T: cpal::SizedSample,
    F: Fn(&T) -> f32 + Send + 'static,
{
    let tx = sample_tx.clone();
    let lost_tx = lost_tx.clone();
    device.build_input_stream::<T, _, _>(
        *config,
        move |data, _| {
            convert_and_send_audio_to_pipeline(&tx, data, channels, sample_rate, &convert);
        },
        move |err| mic_error(&err, &lost_tx),
        None,
    )
}

/// An open microphone: the sample stream, the `cpal::Stream` driving it, and that stream loss
/// reports — created, held and dropped together, so a loss report can only describe the microphone
/// the pipeline currently owns.
struct Microphone {
    rx: mpsc::Receiver<Vec<f32>>,
    lost_rx: mpsc::UnboundedReceiver<()>,
    #[expect(
        dead_code,
        reason = "Dropping the stream is what stops the capture; holding it is the point"
    )]
    stream: cpal::Stream,
}

impl Microphone {
    /// Await the next chunk of mono 16 kHz samples; `None` means the stream can no longer deliver
    /// audio — its error callback reported the device gone, or it ended without saying why.
    async fn next(&mut self) -> Option<Vec<f32>> {
        tokio::select! {
            chunk = self.rx.recv() => chunk,
            // `None` (the callback is gone) means the stream cannot report at all — same dead mic.
            _ = self.lost_rx.recv() => None,
        }
    }
}

/// Start the microphone; returns it with its sample and loss channels.
#[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn start_microphone() -> Result<Microphone> {
    let (tx, rx) = mpsc::channel::<Vec<f32>>(MIC_CHANNEL_CAPACITY);
    let (lost_tx, lost_rx) = mpsc::unbounded_channel::<()>();

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
            &lost_tx,
            channels,
            sample_rate,
            |&s| s,
        ),
        cpal::SampleFormat::I16 => build_int_stream::<i16, _>(
            &device,
            &stream_config,
            &sample_tx,
            &lost_tx,
            channels,
            sample_rate,
            |&s| f32::from(s) / f32::from(i16::MAX),
        ),
        cpal::SampleFormat::U16 => build_int_stream::<u16, _>(
            &device,
            &stream_config,
            &sample_tx,
            &lost_tx,
            channels,
            sample_rate,
            |&s| (f32::from(s) / f32::from(u16::MAX)) * 2.0 - 1.0,
        ),
        // The wider integer and double formats a host may default to — ALSA in particular
        // offers I32 on some devices, and a device this build cannot convert is a microphone
        // the owner cannot use.
        cpal::SampleFormat::I32 => build_int_stream::<i32, _>(
            &device,
            &stream_config,
            &sample_tx,
            &lost_tx,
            channels,
            sample_rate,
            |&s| s as f32 / 2_147_483_648.0,
        ),
        cpal::SampleFormat::I64 => build_int_stream::<i64, _>(
            &device,
            &stream_config,
            &sample_tx,
            &lost_tx,
            channels,
            sample_rate,
            |&s| s as f32 / 9_223_372_036_854_775_808.0,
        ),
        cpal::SampleFormat::F64 => build_int_stream::<f64, _>(
            &device,
            &stream_config,
            &sample_tx,
            &lost_tx,
            channels,
            sample_rate,
            |&s| s as f32,
        ),
        _ => anyhow::bail!("Unsupported sample format: {:?}", config.sample_format()),
    }
    .context("Failed to build microphone input stream")?;

    stream.play().context("Failed to start microphone stream")?;

    debug!(
        "Microphone listening started ({} Hz, {} channels)",
        sample_rate, channels
    );
    Ok(Microphone {
        rx,
        lost_rx,
        stream,
    })
}

/// One recording into words: the samples the pipeline holds, read by the recogniser as they are —
/// no temp file, no re-decode. The bound is the recogniser's own, shared with the enrichment path,
/// so a recording up to the ten-minute cap cannot time out here.
async fn transcribe_audio(samples: &[f32]) -> Result<String> {
    crate::speech::transcribe_samples(
        samples.to_vec(),
        SAMPLE_RATE,
        crate::speech::INFERENCE_TIMEOUT,
    )
    .await
}

/// Per-utterance quality score (0.0–1.0) from the raw audio, weighting duration (50%, in
/// [400ms, 2000ms]), clipping (25%, penalised on any sample at i16::MAX) and SNR (25%).  SNR uses
/// `noise_rms` — the real pre-speech floor captured from the raw ring — when present, else the
/// energy-based heuristic.
#[cfg(target_os = "macos")]
#[expect(clippy::cast_precision_loss)]
fn compute_utterance_quality(samples: &[f32], noise_rms: Option<f32>) -> UtteranceQuality {
    let duration_ms = samples_to_ms(samples.len(), SAMPLE_RATE);

    let clipping_detected = samples
        .iter()
        .any(|&s| s.abs() >= ENROLLMENT_QUALITY_CLIPPING_THRESHOLD);

    // With a real pre-speech noise RMS, the SNR is 20*log10(speech_rms / noise_rms); otherwise the
    // energy-based heuristic measures speech dynamic range rather than true SNR.
    let snr_db = if let Some(noise_rms) = noise_rms {
        let speech_rms = super::util::compute_rms(samples);
        if noise_rms > 1e-10 && speech_rms > noise_rms {
            20.0 * (speech_rms / noise_rms).log10()
        } else {
            0.0
        }
    } else {
        estimate_snr_energy(samples)
    };

    // During collection, quality is duration + clipping + SNR.  Duration: 0.0 too short, 0.3 too
    // long, ramping 0.6→1.0 in range.
    let duration_score = if duration_ms < ENROLLMENT_QUALITY_DURATION_MIN_MS {
        0.0
    } else if duration_ms > ENROLLMENT_QUALITY_DURATION_MAX_MS {
        0.3 // Long utterances still have some value (contain the wake word)
    } else {
        0.6 + (0.4 * (duration_ms - ENROLLMENT_QUALITY_DURATION_MIN_MS) as f32
            / (ENROLLMENT_QUALITY_DURATION_MAX_MS - ENROLLMENT_QUALITY_DURATION_MIN_MS) as f32)
    };

    let clipping_score = if clipping_detected { 0.0 } else { 1.0 };

    // SNR score: 0.0 at 0 dB, 1.0 at 20+ dB (with smooth ramp).
    let snr_score = if snr_db.is_finite() {
        (snr_db / 20.0).clamp(0.0, 1.0)
    } else {
        // A failed SNR estimate (e.g. all silence) gets a moderate score — do not penalise the
        // caller for an estimation limit.
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

/// Energy-based SNR, no neural model: frames the audio into 512-sample windows, takes the top 40%
/// RMS frames as speech and the bottom 40% as noise (the middle 20% is transition), and returns
/// the ratio in dB clamped to [0, 40] to avoid extremes from synthetic signals.
#[cfg(target_os = "macos")]
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
        frame_rms.push(super::util::compute_rms(chunk));
    }

    if frame_rms.len() < 3 {
        return f32::NAN;
    }

    frame_rms.sort_unstable_by(|a, b| a.partial_cmp(b).expect("RMS values must be finite"));
    let n = frame_rms.len();

    let noise_len = (n as f32 * 0.4).ceil() as usize;
    let noise_rms: f32 = frame_rms[..noise_len.min(n)].iter().sum::<f32>() / noise_len as f32;

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

/// Self-test the enrolled prototype against the enrollment buffer: run each utterance embedding
/// through [`score_single_embedding`] with a fresh score window and no adaptive state.  An
/// utterance triggers if its rolling sum crosses the detection threshold at any point.  `Ok(())`
/// when enough trigger, else `Err` with a user-facing message.
#[cfg(target_os = "macos")]
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
        // Fresh simulation per utterance, no cross-utterance state. The embedding is fed
        // ROLLING_WINDOW_N times: the streaming pipeline sees the phrase across ~3 stride-gated
        // encodings (a ~0.5 s phrase spans 3 windows at 160 ms), and one embedding could only ever
        // sum to a single soft score (max 1.0), which never reaches the static threshold.
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
            // Enrollment recognises the phrase in utterances the user just spoke.
            crate::user_activity::report_activity();
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

/// Guidance for the utterance at this 0-based index.
#[cfg(target_os = "macos")]
#[must_use]
pub fn enrollment_prompt_for_sample(sample: usize) -> &'static str {
    let mut cumulative = 0;
    for &(prompt, count) in ENROLLMENT_PROMPTS {
        cumulative += count;
        if sample < cumulative {
            return prompt;
        }
    }
    "Say the wake word clearly"
}

/// Segmentation parameters for [`segment_utterances_by_vad`]: the four module constants that are
/// identical at every call site, bundled so the signature has 3 parameters instead of 8.
#[cfg(target_os = "macos")]
struct VadSegmentationConfig {
    frame_length: usize,
    hop_length: usize,
    consecutive_required: usize,
    /// Silence before an utterance ends ([`ENROLLMENT_SILENCE_THRESHOLD_SAMPLES`] ≈ 304 ms, aligned
    /// to the streaming segment timeout).
    silence_threshold_samples: usize,
}

/// Default config for [`segment_utterances_by_vad`]: no context padding (matching streaming) and
/// the [`ENROLLMENT_SILENCE_THRESHOLD_SAMPLES`] threshold aligned to [`SEGMENT_TIMEOUT_HOPS`].
#[cfg(target_os = "macos")]
const DEFAULT_VAD_SEGMENTATION_CONFIG: VadSegmentationConfig = VadSegmentationConfig {
    frame_length: FRAME_LENGTH,
    hop_length: HOP_LENGTH,
    consecutive_required: ENROLLMENT_VAD_CONSECUTIVE_REQUIRED,
    silence_threshold_samples: ENROLLMENT_SILENCE_THRESHOLD_SAMPLES,
};

/// Segment raw audio into utterances from per-frame VAD decisions, with the boundary algorithm of
/// the enrollment streaming handler ([`handle_enrollment_audio`]). Pure with respect to VAD state —
/// it never touches [`VAD_DETECTOR`]; the caller supplies one decision per `config.hop_length`
/// frame. `config.consecutive_required` consecutive VAD-positive frames confirm sustained speech,
/// then `config.silence_threshold_samples` of VAD-negative audio ends the utterance, which is
/// emitted with internal state reset. No context padding, so segmentation matches streaming; the
/// result is raw audio segments (**not** VAD-subsampled) in detection order.
#[cfg(target_os = "macos")]
fn segment_utterances_by_vad(
    raw_audio: &[f32],
    vad_decisions: &[bool],
    config: &VadSegmentationConfig,
) -> Vec<Vec<f32>> {
    let VadSegmentationConfig {
        frame_length,
        hop_length,
        consecutive_required,
        silence_threshold_samples,
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

    for (frame_idx, &is_speech) in vad_decisions.iter().enumerate() {
        let frame_start = frame_idx * hop_length;

        if is_speech {
            let hop_end = (frame_start + hop_length).min(raw_audio.len());
            if hop_end > frame_start {
                utterance_buf.extend_from_slice(&raw_audio[frame_start..hop_end]);
            }

            vad_positives_in_a_row += 1;

            if vad_positives_in_a_row >= consecutive_required {
                utterance_had_speech = true;
                utterance_speech_end_len = utterance_buf.len();
                utterance_silence_samples = 0;
            } else if utterance_had_speech {
                utterance_speech_end_len = utterance_buf.len();
                utterance_silence_samples = 0;
            }
        } else {
            vad_positives_in_a_row = 0;

            if utterance_had_speech {
                utterance_silence_samples += hop_length;

                if utterance_silence_samples >= silence_threshold_samples {
                    utterance_buf.truncate(utterance_speech_end_len);
                    if !utterance_buf.is_empty() {
                        utterances.push(std::mem::take(&mut utterance_buf));
                    }
                    utterance_speech_end_len = 0;
                    utterance_had_speech = false;
                    utterance_silence_samples = 0;
                    vad_positives_in_a_row = 0;
                }
            }
        }
    }

    utterances
}

// Adaptive threshold state

/// Running mean/std of the recent per-frame soft scores, kept as O(1) sum/sum_sq over the last
/// [`ADAPTIVE_WINDOW_N`].  The threshold is `(mean + k × std) × ROLLING_WINDOW_N`: the scaling into
/// rolling-sum space is what matches [`process_wake_word_score`], and without it the adaptive value
/// (max ~1.0 + k × std) could never reach the rolling sum and the feature would be a no-op.  The
/// result is clamped between the caller's safe harbor (the per-enrollment static threshold) and
/// [`ADAPTIVE_CEILING`]; the first [`ADAPTIVE_BOOTSTRAP_FRAMES`] frames return `None`, telling the
/// caller to use the static threshold while the window fills.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
struct AdaptiveThresholdState {
    scores: Vec<f32>,
    sum: f32,
    sum_sq: f32,
    bootstrap_count: usize,
}

#[cfg(target_os = "macos")]
impl AdaptiveThresholdState {
    fn new() -> Self {
        Self {
            scores: Vec::with_capacity(ADAPTIVE_WINDOW_N),
            sum: 0.0,
            sum_sq: 0.0,
            bootstrap_count: 0,
        }
    }

    /// Feed a per-frame soft score (range [0,1]), scaled by [`ROLLING_WINDOW_N`] into the
    /// rolling-sum space [0,3] where detection compares. `None` during the first
    /// [`ADAPTIVE_BOOTSTRAP_FRAMES`] frames (use the static threshold); after that
    /// `Some(threshold)`, already clamped to `[harbor, ADAPTIVE_CEILING]` in rolling-sum space,
    /// where `harbor` is the caller's per-enrollment static threshold — the anti-feedback-loop
    /// property.
    fn feed(&mut self, score: f32, k: f32, harbor: f32) -> Option<f32> {
        if self.scores.len() >= ADAPTIVE_WINDOW_N {
            let oldest = self.scores.remove(0);
            self.sum -= oldest;
            self.sum_sq -= oldest * oldest;
        }
        self.scores.push(score);
        self.sum += score;
        self.sum_sq += score * score;

        if self.bootstrap_count < ADAPTIVE_BOOTSTRAP_FRAMES {
            self.bootstrap_count += 1;
            return None;
        }

        let threshold = self.compute_threshold(k, harbor);

        Some(threshold)
    }

    /// Clamped adaptive threshold from the current statistics; the window must be non-empty
    /// (checked by the caller). Shared by [`feed`](Self::feed) and [`peek`](Self::peek).
    #[expect(clippy::cast_precision_loss)]
    fn compute_threshold(&self, k: f32, harbor: f32) -> f32 {
        let n = self.scores.len() as f32;
        let mean = self.sum / n;
        // Population variance (divide by n, not n-1) — stable for a fixed-size window.
        let variance = (self.sum_sq / n) - (mean * mean);
        let std = variance.max(0.0).sqrt();

        // Scale per-frame [0,1] to rolling-sum [0,ROLLING_WINDOW_N] space, so the threshold is
        // comparable to the sum used by process_wake_word_score.
        #[expect(clippy::cast_precision_loss)]
        let adaptive = (mean + k * std) * ROLLING_WINDOW_N as f32;

        // Lower bound is the caller's safe harbor (the per-enrollment static threshold), upper the
        // ceiling. `clamp()` propagates NaN where the old `max()` chain returned the harbor:
        // unreachable with finite scores, but the semantics differ.
        adaptive.clamp(harbor, ADAPTIVE_CEILING)
    }

    /// The threshold for the current statistics without updating them: `None` during bootstrap or
    /// with an empty window, else the same clamped value [`feed`](Self::feed) gives. This is how
    /// wake-word-like frames avoid contaminating the background statistics.
    fn peek(&self, k: f32, harbor: f32) -> Option<f32> {
        if self.bootstrap_count < ADAPTIVE_BOOTSTRAP_FRAMES {
            return None;
        }
        if self.scores.is_empty() {
            return None;
        }
        Some(self.compute_threshold(k, harbor))
    }

    /// Whether the tracker is still bootstrapping (the first [`ADAPTIVE_BOOTSTRAP_FRAMES`]
    /// below-reset frames since the score-only feed rule). Production callers no longer consult it
    /// — [`peek`](Self::peek) returns `None` during bootstrap — it exists for unit tests.
    #[cfg(test)]
    fn is_bootstrapping(&self) -> bool {
        self.bootstrap_count < ADAPTIVE_BOOTSTRAP_FRAMES
    }

    fn reset(&mut self) {
        self.scores.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.bootstrap_count = 0;
    }

    /// A pre-warmed state past bootstrap, seeded with near-silence scores so the threshold is
    /// active from the start; the ~0.03 seed clamps immediately to `harbor`, as real audio from
    /// silence does.
    #[cfg(test)]
    fn warmed(harbor: f32) -> Self {
        let mut state = Self::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.033, ADAPTIVE_K_DEFAULT, harbor);
        }
        state
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.scores.len()
    }
}

enum StartAttempt {
    Started,
    /// A recording was already running and owns the composer field: nothing is reported, so its
    /// message is not closed by this press.
    AlreadyRecording,
    /// Nothing is recording. The notice, when there is one, says why.
    Refused(Option<&'static str>),
}

/// The wake word's own pipeline state — detection, scoring, enrollment and the Phase 3
/// negative collection. Only macOS has a wake word, so the whole block is gated once here
/// instead of field by field.
#[cfg(target_os = "macos")]
struct Wake {
    /// Wake-word recording: samples still to drop from the front before any reaches recognition
    /// ([`POST_FIRE_DISCARD_SAMPLES`]); armed by the detection handoff.
    post_fire_discard_samples: usize,
    /// Whether the retained part of the current wake-word recording held speech, by the same
    /// per-chunk VAD detection uses. A recording with none delivers nothing.
    recording_had_speech: bool,
    frame_vad: Vec<bool>,
    frame_raw_audio: Vec<f32>,
    /// Utterances already emitted by [`segment_utterances_by_vad`]: reset across sessions, kept
    /// within one so the function can run over the whole buffer each call.
    emitted_utterances: usize,
    utterance_had_speech: bool,
    /// Enrollment silence duration in samples (sample-based, so load cannot drift it).
    utterance_silence_samples: usize,
    /// Consecutive non-VAD frames during enrollment, for the "speak louder" warning.
    enrollment_no_speech_frame_count: usize,
    /// Consecutive VAD-positive frames for sustained-speech confirmation — accumulation starts at
    /// the first one ([`ENROLLMENT_VAD_CONSECUTIVE_REQUIRED`] = 1), like streaming detection.
    vad_positives_in_a_row: usize,
    /// VAD threshold for the current mode, unified to [`VAD_THRESHOLD`].
    vad_threshold: f32,
    /// Earshot detector for enrollment mode. The global [`VAD_DETECTOR`] keeps continuous
    /// noise-floor and ring-buffer state across the live stream, so enrollment uses its own
    /// instance to prevent mode-transition contamination: `None` outside enrollment, set to
    /// `Some(earshot::Detector::default())` from start to end or cancel.
    enrollment_vad: Option<earshot::Detector>,
    /// Raw audio ring for wake-word detection, capped at [`AUDIO_BUFFER_MAX`] and walked by the VAD
    /// loop via [`Wake::vad_cursor`]; scoring encodes the VAD-gated [`Wake::speech_window`], never
    /// this buffer.
    audio_buffer: Vec<f32>,
    /// VAD-gated speech-only window for encoder scoring: only the [`HOP_LENGTH`] hops the VAD
    /// called speech, capped at [`WAKE_WORD_WINDOW_SAMPLES`] (draining the front). Enrollment
    /// encodes VAD-segmented speech-only utterances, so scoring the same kind of window keeps the
    /// cosine distributions aligned — raw trailing audio dilutes the embedding with context/silence
    /// and shifts detection cosines ~0.05-0.08 below enrollment, letting confusables through.
    speech_window: Vec<f32>,
    /// Samples at the FRONT of [`Wake::audio_buffer`] already consumed: the loop advances this
    /// cursor instead of draining, so the ring keeps the whole trailing window even after every
    /// frame was checked.
    vad_cursor: usize,
    /// Samples since the last scoring step, reset after each encode; at [`SCORE_STRIDE_SAMPLES`]
    /// with speech seen, the pipeline encodes and scores.
    last_score_sample_count: usize,
    /// Completed enrollment utterances awaiting the main loop, which pops one per tick and runs
    /// [`handle_enrollment_sample`].  A queue rather than a single `Option`, so several utterances
    /// completing within one mic frame are all preserved.
    enrollment_pending: VecDeque<Vec<f32>>,
    /// Last detection time, for the cooldown that prevents rapid consecutive false triggers.
    last_wake_word_detection: Option<Instant>,
    /// Pre-speech noise RMS captured at the first sustained speech, from the raw audio ring so the
    /// SNR reflects the true room floor; consumed and reset to `None` by
    /// [`handle_enrollment_sample`].
    noise_rms_estimate: Option<f32>,
    /// Phase 3 audio buffer, kept apart from the Phase 2 `audio_buffer` so they cannot interfere.
    phase3_audio_buf: Vec<f32>,
    /// Phase 3 silence tracking, on [`ENROLLMENT_SILENCE_THRESHOLD_SAMPLES`] like enrollment
    /// utterance end, to detect chunk boundaries.
    phase3_silence_samples: usize,
    /// VAD-positive speech samples in Phase 3: every VAD-positive hop contributes exactly
    /// [`HOP_LENGTH`], once (see [`Wake::phase3_processed`]).  At
    /// `SAMPLE_RATE * NEGATIVES_TARGET_SECONDS` the phase completes and finalization starts.
    negatives_speech_samples: usize,
    /// Watermark into [`Wake::phase3_audio_buf`]: samples already fed to the VAD, which resumes
    /// here on the next chunk so each hop reaches the stateful detector exactly once. Without it an
    /// un-drained buffer re-processes and re-counts every prior frame per chunk — the quadratic
    /// counter defect.
    phase3_processed: usize,
    /// Wall-clock start of Phase 3, for the [`PHASE3_TIMEOUT_SECS`] finalize-anyway timeout.
    phase3_start_time: Option<Instant>,
    /// Per-frame soft scores from the cosine-prototype scorer, one per scoring step.
    score_window: Vec<f32>,
    /// Consecutive VAD-negative hops since the last positive frame; at [`SEGMENT_TIMEOUT_HOPS`]
    /// (~300 ms) the per-segment detection state resets.
    segment_silence_hops: usize,
    /// Stride-gated scoring steps since the first step of the current speech segment — every step
    /// counts, even ones re-scoring unchanged audio during a pause. Cleared where the segment ends
    /// and wherever a detection is handled, so a fire is accepted only while this is ≤
    /// [`WAKE_WORD_MAX_SEGMENT_STEPS`].
    segment_score_steps: usize,
    /// Adaptive threshold tracker for the rolling score window.
    adaptive_threshold: AdaptiveThresholdState,
    /// Adaptive threshold k multiplier (fixed at [`ADAPTIVE_K_DEFAULT`]).
    adaptive_k: f32,
    /// Non-VAD audio during enrollment (pre-speech ambient noise, inter-utterance silence), saved
    /// as a negative example at speech onset.
    negative_audio_buf: Vec<f32>,
}

#[cfg(target_os = "macos")]
impl Wake {
    fn new() -> Self {
        Self {
            post_fire_discard_samples: 0,
            recording_had_speech: false,
            frame_vad: Vec::new(),
            frame_raw_audio: Vec::new(),
            emitted_utterances: 0,
            utterance_had_speech: false,
            utterance_silence_samples: 0,
            enrollment_no_speech_frame_count: 0,
            vad_positives_in_a_row: 0,
            vad_threshold: VAD_THRESHOLD,
            enrollment_vad: None,
            audio_buffer: Vec::new(),
            speech_window: Vec::new(),
            vad_cursor: 0,
            last_score_sample_count: 0,
            enrollment_pending: VecDeque::new(),
            last_wake_word_detection: None,
            noise_rms_estimate: None,
            phase3_audio_buf: Vec::new(),
            phase3_silence_samples: 0,
            negatives_speech_samples: 0,
            phase3_processed: 0,
            phase3_start_time: None,
            score_window: Vec::new(),
            segment_silence_hops: 0,
            segment_score_steps: 0,
            adaptive_threshold: AdaptiveThresholdState::new(),
            // Fixed default — no config key backs this field.
            adaptive_k: ADAPTIVE_K_DEFAULT,
            negative_audio_buf: Vec::new(),
        }
    }
}

#[expect(clippy::struct_excessive_bools)]
struct PipelineCtx {
    /// The open microphone, if any (see [`Microphone`]).
    mic: Option<Microphone>,
    is_listening: bool,
    /// The recording in progress: the mic button and the wake word share one mechanism, differing
    /// only in how it can end.
    recording: Option<RecordingOrigin>,
    /// Whether wake-word listening resumes when the current recording ends — set when the wake-word
    /// mic was running, or the switch is on, so a recording never leaves voice stopped while the
    /// switch shows it on.
    resume_listening_after_recording: bool,
    command_buffer: Vec<f32>,
    /// Silence by audio sample count, not wall-clock, so load and processing delays cannot skew the
    /// cutoff.
    silence_sample_count: usize,
    /// The wake word's own state (macOS only).
    #[cfg(target_os = "macos")]
    wake: Wake,
    enrollment_mode: bool,
    auto_start_pending: bool,
    /// Last automatic retry time, debouncing the retry path against the 1-second tick while a
    /// loader sits in [`ModelState::Failed`].
    last_model_retry: Option<Instant>,
    /// Actively collecting owner-negative general speech (Phase 3); set by
    /// `transition_to_phase3()` once the ten wake-word utterances are done.
    collecting_negatives: bool,
    /// Timestamp until which Error→Listening is suppressed, set after transcription failures.
    refractory_until: Option<Instant>,
    /// Timestamp of the last rate-limited voice notice.
    last_voice_notice_time: Option<Instant>,
    /// Remaining automatic retry cycles ([`MAX_AUTO_MODEL_RETRY_CYCLES`]), consumed only when the
    /// self-healing path retries; the explicit GUI retry bypasses the budget.
    auto_model_retries_left: u32,
}

/// Parameterised pipeline state reset level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResetLevel {
    Full,
    /// The wake word's lighter reset: keeps the VAD continuity and the cooldown a detection
    /// handoff must not lose.
    #[cfg(target_os = "macos")]
    Soft,
    #[cfg(target_os = "macos")]
    Cancel,
}

impl PipelineCtx {
    fn new() -> Self {
        Self {
            mic: None,
            is_listening: false,
            recording: None,
            resume_listening_after_recording: false,
            command_buffer: Vec::new(),
            silence_sample_count: 0,
            enrollment_mode: false,
            auto_start_pending: CONFIG.voice_enabled().as_deref() == Some("true"),
            last_model_retry: None,
            collecting_negatives: false,
            refractory_until: None,
            last_voice_notice_time: None,
            auto_model_retries_left: MAX_AUTO_MODEL_RETRY_CYCLES,
            #[cfg(target_os = "macos")]
            wake: Wake::new(),
        }
    }

    /// Full reset while preserving [`auto_start_pending`], which the reset clears but a pending
    /// wake-word auto-start must survive the recording-only-mic lifecycle.
    fn full_reset_preserving_auto_start(&mut self) {
        let auto_start = self.auto_start_pending;
        self.reset_pipeline_state(ResetLevel::Full);
        self.auto_start_pending = auto_start;
    }

    /// Take what a wake-word recording captured and end it: the audio and whether anything was
    /// said. Every stop path goes through here so the order cannot be lost — the silence flag is
    /// read before the end clears it, the buffer is taken before the reset drops it. Soft drops the
    /// recording and detection buffers while preserving VAD continuity, `vad_threshold` and the
    /// cooldown timestamp, so the trigger cannot immediately re-fire. The caller owns the status
    /// (Listening after a normal end, Enrolling after an aborted recording).
    #[cfg(target_os = "macos")]
    fn take_wake_recording(&mut self) -> (Vec<f32>, bool) {
        let had_speech = self.wake.recording_had_speech;
        let cmd_buf = std::mem::take(&mut self.command_buffer);
        self.recording = None;
        self.end_wake_state();
        (cmd_buf, had_speech)
    }

    /// Drop a wake-word recording's buffers while keeping VAD continuity, and play the end cue its
    /// stop owes.
    #[cfg(target_os = "macos")]
    fn end_wake_state(&mut self) {
        self.reset_pipeline_state(ResetLevel::Soft);
        cue::play(Cue::End);
    }

    /// End an enrollment attempt — completed, cancelled, failed in finalization, or cut short by a
    /// lost microphone. Cancel is the only level that clears the global accumulators, and it also
    /// clears `collecting_negatives`: those are what the Phase 2→3 and 3→4 gates read. The VAD is
    /// left alone, so its noise-floor estimate carries into detection; the caller owns the status.
    #[cfg(target_os = "macos")]
    fn end_enrollment_attempt(&mut self) {
        self.reset_pipeline_state(ResetLevel::Cancel);
        self.enrollment_mode = false;
    }

    /// Parameterised reset. It only clears state and never publishes an outcome — a caller ending a
    /// recording reports that itself ([`Self::abandon_recording`]) in wording matching what
    /// happened. Audio buffers, the counters, enrollment/segment state and the Phase 3 fields clear
    /// at every level. `vad_threshold`, `enrollment_vad` and `last_wake_word_detection` are reset
    /// by Full and Cancel but preserved by Soft, as are `auto_start_pending`, `recording` and
    /// `resume_listening_after_recording`; `reset_vad()` runs on Full only, and the global
    /// enrollments are cleared by Cancel alone. `refractory_until`, the notice/retry stamps, `mic`,
    /// `is_listening` and `enrollment_mode` are never touched.
    #[cfg_attr(
        not(target_os = "macos"),
        expect(
            unused_variables,
            reason = "the level only decides how much of the wake word's state is kept"
        )
    )]
    fn reset_pipeline_state(&mut self, level: ResetLevel) {
        self.command_buffer.clear();
        self.silence_sample_count = 0;
        self.collecting_negatives = false;
        // The wake word's buffers, counters and detection state are its own half of the reset.
        #[cfg(target_os = "macos")]
        self.reset_wake_word_state(level);
    }

    /// The wake word's half of [`Self::reset_pipeline_state`]: audio rings, detection and
    /// enrollment state, and the Phase 3 fields, at the level the caller asked for.
    #[cfg(target_os = "macos")]
    fn reset_wake_word_state(&mut self, level: ResetLevel) {
        self.wake.audio_buffer.clear();
        self.wake.speech_window.clear();
        self.wake.vad_cursor = 0;
        self.wake.post_fire_discard_samples = 0;
        self.wake.recording_had_speech = false;
        self.wake.score_window.clear();
        self.wake.negative_audio_buf.clear();
        self.wake.segment_silence_hops = 0;
        self.wake.segment_score_steps = 0;
        self.wake.last_score_sample_count = 0;

        self.wake.utterance_had_speech = false;
        self.wake.utterance_silence_samples = 0;
        self.wake.enrollment_no_speech_frame_count = 0;
        self.wake.vad_positives_in_a_row = 0;
        self.wake.enrollment_pending.clear();
        self.wake.noise_rms_estimate = None;
        self.wake.frame_vad.clear();
        self.wake.frame_raw_audio.clear();
        self.wake.emitted_utterances = 0;

        self.wake.phase3_audio_buf.clear();
        self.wake.phase3_silence_samples = 0;
        self.wake.negatives_speech_samples = 0;
        self.wake.phase3_processed = 0;
        self.wake.phase3_start_time = None;

        // Enrollment VAD detector, cleared at every level so modes cannot contaminate each other.
        self.wake.enrollment_vad = None;

        match level {
            ResetLevel::Full => {
                self.wake.vad_threshold = VAD_THRESHOLD;
                self.wake.last_wake_word_detection = None;
                self.auto_start_pending = false;
                self.recording = None;
                reset_vad();
                self.wake.adaptive_threshold.reset();
            }
            ResetLevel::Soft => {
                // Soft preserves VAD state, `vad_threshold`, the cooldown, `auto_start_pending`,
                // the running recording and the global enrollments, and clears the rolling-window
                // state so stale scores cannot cross utterances (Soft runs on the
                // detection→recording handoff).
            }
            ResetLevel::Cancel => {
                self.wake.vad_threshold = VAD_THRESHOLD;
                self.wake.last_wake_word_detection = None;
                self.wake.adaptive_threshold.reset();

                voice_state().write().unwrap_poison().reset_enrollment();
            }
        }
    }

    /// Reset all per-segment detection state at a VAD-driven utterance boundary
    /// ([`SEGMENT_TIMEOUT_HOPS`] ≈ 300 ms of VAD-negative hops): the score window and adaptive
    /// threshold (both per-segment, so a brief re-bootstrap is acceptable), the silence counter and
    /// the score-step clock, which must start the next segment at position 1. This stops soft
    /// scores and rolling sums accumulating across utterances — a structural source of false
    /// triggers. The raw ring and VAD cursor are cleared with them so the trailing window cannot
    /// span the boundary; `audio_buffer` (the drain handles its overlap), VAD state,
    /// `vad_threshold`, the cooldown and `recording` are preserved.
    #[cfg(target_os = "macos")]
    fn reset_detection_segment(&mut self) {
        self.wake.score_window.clear();

        self.wake.adaptive_threshold.reset();

        self.wake.segment_silence_hops = 0;
        self.wake.segment_score_steps = 0;

        self.wake.audio_buffer.clear();
        self.wake.speech_window.clear();
        self.wake.vad_cursor = 0;
        self.wake.last_score_sample_count = 0;
    }

    /// Segment boundary check, called from [`handle_wake_word_detection`] after the VAD loop with
    /// the accumulated `hop_count` of consecutive VAD-negative frames. At [`SEGMENT_TIMEOUT_HOPS`]
    /// it resets the per-segment detection state, so soft scores and rolling sums cannot accumulate
    /// across utterances; below it the counter is persisted in
    /// [`segment_silence_hops`](Wake::segment_silence_hops) for the next call.
    #[cfg(target_os = "macos")]
    fn handle_segment_boundary(&mut self, hop_count: usize) {
        if hop_count >= SEGMENT_TIMEOUT_HOPS {
            // Re-check not-recording: a just-fired detection must NOT be reset — the handoff in
            // handle_wake_word_detection completes the transition.
            if self.recording.is_none() {
                self.reset_detection_segment();
            }
        } else {
            self.wake.segment_silence_hops = hop_count;
        }
    }

    /// Check the refractory period and return to [`VoiceStatus::Listening`] once it has elapsed;
    /// called once per main-loop iteration. After a transcription failure the 3-second period
    /// prevents immediate re-triggering. No transition while a recording runs (a concurrent error
    /// path already started one) or when the pipeline stopped listening (the microphone is gone,
    /// and the Error came from elsewhere).
    fn check_refractory_period(&mut self) {
        if let Some(refractory_until) = self.refractory_until
            && Instant::now() >= refractory_until
        {
            self.refractory_until = None;
            if self.is_listening
                && self.recording.is_none()
                && matches!(get_status(), VoiceStatus::Error(_))
            {
                set_status(VoiceStatus::Listening);
            }
        }
    }

    /// Whether [`NOTICE_INTERVAL`] has elapsed since `last` (true when nothing was sent yet).
    fn should_send_rate_limited(last: Option<Instant>) -> bool {
        last.is_none_or(|t| t.elapsed() >= NOTICE_INTERVAL)
    }

    /// Voice-notice rate-limit guard.
    fn should_send_voice_notice(&self) -> bool {
        Self::should_send_rate_limited(self.last_voice_notice_time)
    }

    /// Broadcast a rate-limited notice: the pipeline's own report for a recording that could not
    /// start.
    async fn broadcast_voice_notice(&mut self, msg: &str) {
        if !self.should_send_voice_notice() {
            return;
        }
        self.last_voice_notice_time = Some(Instant::now());
        broadcast_notice(msg).await;
    }

    #[cfg(target_os = "macos")]
    fn handle_start_listening(&mut self) {
        // While the composer's own recording runs, defer wake-word listening until it ends (mutual
        // exclusion); the resume flag makes `end_button_recording` start listening afterwards.
        if self.recording == Some(RecordingOrigin::Button) {
            self.resume_listening_after_recording = true;
            info!("Voice pipeline: start_listening deferred until the recording ends");
            return;
        }
        // Wake word shares the recogniser's model set: with transcription disabled it never loads.
        if is_transcription_disabled() {
            self.auto_start_pending = false;
            warn!(
                "Ignoring start_listening — local transcription disabled (wake word requires ASR)"
            );
            return;
        }
        // Defense in depth: voice may have been disabled between sending and processing the
        // command, as the guard in handle_start_enrollment.
        if !is_enabled() {
            self.auto_start_pending = false;
            warn!("Ignoring start_listening — voice assistant is disabled");
            return;
        }
        if !models_ready() {
            // The encoder is still loading — mark pending so check_auto_start retries when it is
            // ready; this is NOT set on mic failure, which would retry forever. A previously
            // failed load (ModelError trap) is retried immediately, so the user need not restart
            // the app. Nothing preloads the encoder, so a start is also what demands it.
            demand_encoder();
            if crate::audio::encoder::is_failed() {
                warn!("The wake word's encoder previously failed — triggering retry...");
                self.try_retry_models();
            }
            self.auto_start_pending = true;
            warn!("Voice models not ready yet");
            return;
        }
        if !self.is_listening {
            // A recording holding nothing but its own microphone cannot outlive this start: report
            // it here, where the reason is known.
            self.abandon_recording("*Voice: recording discarded — wake-word listening started*");
            self.reset_pipeline_state(ResetLevel::Full);
            self.mic = None;
            match start_microphone() {
                Ok(mic) => {
                    self.mic = Some(mic);
                    self.is_listening = true;
                    set_status(VoiceStatus::Listening);
                    info!("Voice pipeline: started listening");
                    // Open the cue device now, while nothing is recorded: a cold open on the
                    // detection path would delay the start cue past the opening the recording
                    // discards, leaking it into recognized audio.
                    cue::warm_up();
                }
                Err(e) => {
                    warn!("Failed to start microphone: {e}");
                    set_status(mic_start_failure(&e));
                    // auto_start_pending stays cleared: re-opening the device is the user's own
                    // action ([`VoiceStatus::MicDisconnected`]), never an automatic retry.
                }
            }
        }
    }

    /// Tear down the open microphone and the per-stream pipeline state. The caller owns the
    /// resulting status: [`VoiceStatus::Disabled`] on an explicit stop or a loss with the switch
    /// off, [`VoiceStatus::MicDisconnected`] when the device disappeared while voice was on. An
    /// in-progress recording is reported to the composer as lost, with `notice` saying why.
    fn teardown_microphone(&mut self, notice: &'static str) {
        // The recording is abandoned here, wake-word one included: its end cue and state clear
        // belong to the helper, which is a no-op when nothing records.
        self.abandon_recording(notice);
        // Full reset: the mic stream is torn down, so the old VAD state no longer represents the
        // next acoustic environment. Global enrollment accumulators (macOS) are deliberately
        // preserved here — toggle-off/on keeps mid-enrollment progress — so a caller that must end
        // an attempt cancels it explicitly.
        self.reset_pipeline_state(ResetLevel::Full);
        self.is_listening = false;
        self.enrollment_mode = false;
        self.resume_listening_after_recording = false;
        self.mic = None;
    }

    /// Stop the microphone — the wake word's on macOS — and tear down pipeline state.
    fn handle_stop_listening(&mut self) {
        self.teardown_microphone("*Voice: recording discarded — voice assistant turned off*");
        set_status(VoiceStatus::Disabled);
        info!("Voice pipeline: stopped listening");
    }

    /// Handle the microphone becoming unavailable (unplugged, host gone, stream invalidated),
    /// reported by the stream itself. Everything it was doing ends: an in-progress recording —
    /// wake-word one, end cue included — is discarded, and an enrollment attempt is abandoned with
    /// everything it collected dropped, so its gates cannot complete it later. Nothing is re-armed:
    /// re-opening is the user's own action ([`VoiceStatus::MicDisconnected`]).
    fn handle_microphone_lost(&mut self) {
        #[cfg(target_os = "macos")]
        self.end_enrollment_attempt();
        self.teardown_microphone("*Voice: recording discarded — microphone disconnected*");
        // Read the switch and write the status under one lock: a user turning voice off here keeps
        // the Disabled state their action wrote, never a stale one.
        {
            let mut state = voice_state().write().unwrap_poison();
            let status = if state.enabled {
                VoiceStatus::MicDisconnected
            } else {
                VoiceStatus::Disabled
            };
            set_status_and_notify(&mut state, status);
        }
        info!("Voice pipeline: microphone unavailable — voice input stopped");
    }

    /// Start an enrollment, aborting the recording it cannot coexist with.
    #[cfg(target_os = "macos")]
    fn handle_start_enrollment(&mut self) {
        if !self.is_listening {
            warn!("Cannot start enrollment: microphone not running");
            set_status(VoiceStatus::Error(
                "Microphone not running — enable Voice first".to_string(),
            ));
            return;
        }

        // No recording can coexist with enrollment: the router gives enrollment priority, so a
        // recording buffer would starve (its auto-send cap never fires) and later resume as
        // garbage. Abandoning it also ends a wake-word recording cue.
        if self.abandon_recording("*Voice: recording discarded — enrollment started*") {
            warn!("Aborting the recording — enrollment started");
            self.resume_listening_after_recording = false;
        }

        // Resume existing progress if there is any (an interrupted session keeps its global
        // embeddings). Starting fresh clears stale buffers at Cancel level, preserving VAD
        // continuity — same mic stream, same acoustic environment.
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

        self.enrollment_mode = true;
        // A separate VAD detector for this session, so modes cannot contaminate each other.
        self.wake.enrollment_vad = Some(earshot::Detector::default());
        // `vad_threshold` is intentionally not changed: it stays [`VAD_THRESHOLD`] (0.5) in both
        // modes.
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

    #[cfg(target_os = "macos")]
    fn handle_cancel_enrollment(&mut self) {
        self.end_enrollment_attempt();
        set_status(if self.is_listening {
            VoiceStatus::Listening
        } else {
            VoiceStatus::Disabled
        });
        info!("Voice pipeline: enrollment cancelled");
    }

    /// Phase 2 → Phase 3, when `utterances_collected` and not yet `collecting_negatives`.  The VAD
    /// threshold is already unified, so nothing switches; stale pending utterances are drained, the
    /// negative buffer and silence counters cleared, and the Phase 3 flags set.
    #[cfg(target_os = "macos")]
    fn transition_to_phase3(&mut self) {
        if !self.wake.enrollment_pending.is_empty() {
            warn!(
                "transition_to_phase3: draining {} stale enrollment pending utterances",
                self.wake.enrollment_pending.len(),
            );
            self.wake.enrollment_pending.clear();
        }

        self.wake.negative_audio_buf.clear();
        self.wake.utterance_silence_samples = 0;
        self.wake.utterance_had_speech = false;
        self.wake.vad_positives_in_a_row = 0;

        self.collecting_negatives = true;
        self.wake.phase3_start_time = Some(Instant::now());

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
        self.abandon_recording("*Voice: recording discarded — voice assistant stopped*");
        self.resume_listening_after_recording = false;
        self.mic = None;
    }

    /// Start a recording from the composer mic button. The composer opened its message window on
    /// the press, so a start that does not take closes it again — the field is never held by a
    /// recording that does not exist. Wake-word detection pauses for the recording (audio routes to
    /// [`handle_button_recording_audio`]) and resumes as `resume_listening_after_recording`
    /// decides. None of it needs the wake-word assistant: the microphone starts on demand and the
    /// recogniser is not a prerequisite (it loads on use), so a press is recorded even as its first
    /// load begins. Returns the notice when the on-demand mic could not start, `None` otherwise.
    fn handle_start_button_recording(&mut self) -> Option<&'static str> {
        match self.try_start_button_recording() {
            StartAttempt::Started | StartAttempt::AlreadyRecording => None,
            StartAttempt::Refused(notice) => {
                // The composer opened its message window on the press; a start that did not take
                // closes it, or the field waits for a message that never comes.
                publish_recording(RecordingEvent::NotStarted);
                notice
            }
        }
    }

    /// [`Self::handle_start_button_recording`] without the composer hand-off.
    fn try_start_button_recording(&mut self) -> StartAttempt {
        if self.recording.is_some() {
            // A running recording owns the composer field: this press leaves that window alone.
            warn!("Cannot start a recording — one is already running");
            return StartAttempt::AlreadyRecording;
        }
        if self.enrollment_mode || self.collecting_negatives {
            warn!("Cannot start a voice recording during enrollment");
            return StartAttempt::Refused(None);
        }
        // With local transcription disabled the shared ASR model never loads (or is gone after a
        // restart): surface the configuration, not a loading state that can never complete. The
        // guard is unconditional, mirroring [`recording_blocked_reason`], so a still-loaded
        // in-memory model cannot contradict it.
        if is_transcription_disabled() {
            warn!("Cannot start a voice recording — local transcription disabled");
            return StartAttempt::Refused(Some(
                "Voice recording unavailable — local transcription is disabled",
            ));
        }
        // A recogniser that failed to load is the one case a recording cannot end well: say that.
        if crate::speech::is_failed() {
            warn!("Cannot start a voice recording — the local speech model failed to load");
            set_status(VoiceStatus::ModelError);
            return StartAttempt::Refused(Some(
                "Voice unavailable — the local speech model failed to load; retry it in Settings",
            ));
        }
        // Recording needs no recogniser — transcription loads it on use and waits (see
        // `speech::transcribe_samples`) — so asking for it here is what lets the load overlap the
        // recording instead of answering the first press with a wait.
        demand_transcriber();

        if self.mic.is_none() {
            // Neither listening nor loading a model: start the mic just for this recording.
            match start_microphone() {
                Ok(mic) => self.mic = Some(mic),
                Err(e) => {
                    warn!("Failed to start recording mic: {e}");
                    // Say why the microphone is unavailable; with the mic still gone the attempt
                    // leaves that state standing rather than doing nothing.
                    let status = mic_start_failure(&e);
                    let notice = if matches!(status, VoiceStatus::MicPermissionDenied) {
                        MIC_PERMISSION_NOTICE
                    } else {
                        "Could not start the microphone — check your input device"
                    };
                    set_status(status);
                    return StartAttempt::Refused(Some(notice));
                }
            }
        }

        // Resume wake-word listening after the recording whenever the switch is on — not only when
        // the wake-word mic was already running, since the mic may have been opened just for this
        // recording
        self.resume_listening_after_recording = self.is_listening || is_enabled();
        self.recording = Some(RecordingOrigin::Button);
        self.command_buffer.clear();
        self.silence_sample_count = 0;
        set_status(VoiceStatus::RecordingButton);
        info!("Voice pipeline: recording started from the mic button");
        StartAttempt::Started
    }

    /// Stop the recording in progress — whichever way it started — and either
    /// send what it holds or discard it.
    async fn handle_stop_recording(&mut self, send: bool) {
        match self.recording {
            // Every way a recording ends publishes its outcome, so a stop with
            // nothing to stop has nothing left to answer.
            None => debug!("StopRecording with no recording in progress"),
            Some(RecordingOrigin::Button) => {
                let cmd_buf = std::mem::take(&mut self.command_buffer);
                self.silence_sample_count = 0;
                if send {
                    self.finish_button_recording(cmd_buf).await;
                } else {
                    // The user's own discard: the composer closed its window on the same press.
                    self.end_button_recording();
                }
            }
            #[cfg(target_os = "macos")]
            Some(RecordingOrigin::WakeWord) => {
                let (cmd_buf, had_speech) = self.take_wake_recording();
                if send {
                    self.finish_wake_recording(cmd_buf, had_speech).await;
                } else {
                    // The user's own discard, exactly like the silence end that found nothing to
                    // send.
                    set_status(VoiceStatus::Listening);
                }
            }
        }
    }

    /// Accumulate audio during a button recording; auto-send at the 10-minute cap.
    #[expect(clippy::cast_precision_loss)]
    async fn handle_button_recording_audio(&mut self, samples: &[f32]) {
        self.command_buffer.extend_from_slice(samples);
        let duration_secs = self.command_buffer.len() as f64 / f64::from(SAMPLE_RATE);
        if duration_secs > MAX_RECORD_SECS as f64 {
            debug!("Button recording stopped: max duration ({duration_secs:.1}s)");
            let cmd_buf = std::mem::take(&mut self.command_buffer);
            self.silence_sample_count = 0;
            self.finish_button_recording(cmd_buf).await;
        }
    }

    /// Transcribe a completed button recording, hand its outcome to the composer (which owns the
    /// message), then restore wake-word listening.
    async fn finish_button_recording(&mut self, cmd_buf: Vec<f32>) {
        // The composer takes the field content now, so the message is in flight from the moment
        // recording stops — whether the user pressed send or the cap stopped it.
        publish_recording(RecordingEvent::Stopped);
        set_status(VoiceStatus::Transcribing);
        if cmd_buf.is_empty() {
            warn!("Button recording stopped with no audio captured");
            publish_recording(RecordingEvent::NoSpeech);
        } else {
            self.publish_transcript(cmd_buf).await;
        }
        self.end_button_recording();
    }

    /// Transcribe a finished recording and publish its outcome; `false` when transcription failed
    /// (the failure is already published), the caller deciding what the status does.
    async fn publish_transcript(&mut self, cmd_buf: Vec<f32>) -> bool {
        match transcribe_audio(&cmd_buf).await {
            Ok(transcribed) if !transcribed.trim().is_empty() => {
                publish_recording(RecordingEvent::Transcript(
                    crate::channels::marked_transcript(&transcribed),
                ));
                true
            }
            Ok(_) => {
                warn!("Empty transcription — dropping ({} samples)", cmd_buf.len());
                publish_recording(RecordingEvent::NoSpeech);
                true
            }
            Err(e) => {
                warn!("Transcription failed: {e}");
                publish_recording(RecordingEvent::Failed(TRANSCRIPTION_FAILED_NOTICE));
                false
            }
        }
    }

    /// Finish a wake-word recording that ended itself (silence, the cap) or was ended from the
    /// composer: its transcript joins the field, which the composer sends as one message.
    #[cfg(target_os = "macos")]
    async fn finish_wake_recording(&mut self, cmd_buf: Vec<f32>, had_speech: bool) {
        // Nothing was said, so nothing joins the field: publishing that leaves the composer window,
        // its text and attachments exactly as they were.
        if !had_speech || cmd_buf.is_empty() {
            debug!(
                "No speech in wake-word recording — dropping {} samples",
                cmd_buf.len()
            );
            publish_recording(RecordingEvent::NoSpeech);
            set_status(VoiceStatus::Listening);
            return;
        }
        // The words join the field, which the composer takes over now; the message is in flight.
        publish_recording(RecordingEvent::Stopped);
        set_status(VoiceStatus::Transcribing);
        if self.publish_transcript(cmd_buf).await {
            set_status(VoiceStatus::Listening);
        } else {
            set_status(VoiceStatus::Error("Transcription failed".to_string()));
            // Enforce a 3-second refractory period before returning to Listening — non-blocking,
            // checked by the main loop instead of the old 2-second sleep.
            self.refractory_until = Some(Instant::now() + Duration::from_secs(3));
        }
    }

    /// End the recording in progress without producing a message — a teardown, an enrollment
    /// starting, wake-word listening starting on a recording-only mic, shutdown — and tell the
    /// composer, so a field it committed comes back instead of waiting for a message that never
    /// arrives. `notice` says why, in the words the user sees; returns whether a recording was
    /// running.
    fn abandon_recording(&mut self, notice: &'static str) -> bool {
        let Some(origin) = self.recording.take() else {
            return false;
        };
        // A wake-word recording ends like any other: its buffers go and its end cue plays. A button
        // recording owes no cue — the press that started it never got one.
        match origin {
            #[cfg(target_os = "macos")]
            RecordingOrigin::WakeWord => self.end_wake_state(),
            RecordingOrigin::Button => {
                self.command_buffer.clear();
                self.silence_sample_count = 0;
            }
        }
        publish_recording(RecordingEvent::Failed(notice));
        true
    }

    /// Finalize a button recording: resume wake-word listening when
    /// `resume_listening_after_recording` says so, else tear down the recording-only mic.
    fn end_button_recording(&mut self) {
        self.recording = None;
        // Only the wake word resumes a microphone by itself: off macOS the request is consumed and
        // the recording-only mic is torn down below, exactly as when nothing asked to resume.
        if self.resume_listening_after_recording {
            self.resume_listening_after_recording = false;
            #[cfg(target_os = "macos")]
            {
                if self.is_listening {
                    // Wake-word mic still running: clear recording buffers, keep VAD continuity,
                    // back to listening.
                    self.reset_pipeline_state(ResetLevel::Soft);
                    set_status(VoiceStatus::Listening);
                } else {
                    // Voice was toggled ON during the recording. If the wake-word mic cannot start
                    // now (models still loading), handle_start_listening re-arms auto_start_pending
                    // and check_auto_start retries; tear the recording-only mic down rather than
                    // leaving a zombie stream.
                    self.handle_start_listening();
                    if !self.is_listening {
                        self.mic = None;
                        // Keep a mic-failure status set by handle_start_listening, so the user sees
                        // WHY voice did not resume.
                        if !matches!(
                            get_status(),
                            VoiceStatus::MicPermissionDenied | VoiceStatus::MicDisconnected
                        ) {
                            set_status(VoiceStatus::Disabled);
                        }
                    }
                }
                debug!("Voice pipeline: recording ended");
                return;
            }
        }
        // Recording-only mic: tear it down, preserving auto_start_pending (cleared by the Full
        // reset) so a wake-word assistant still loading models auto-starts afterwards.
        self.full_reset_preserving_auto_start();
        self.is_listening = false;
        self.mic = None;
        set_status(VoiceStatus::Disabled);
        debug!("Voice pipeline: recording ended");
    }

    /// Retry loading the models, at most once every 30 seconds, so the periodic 1-second wake-up
    /// cannot start a retry storm. `true` when a retry was initiated.
    fn try_retry_models(&mut self) -> bool {
        let cooldown = Duration::from_secs(30);
        if self
            .last_model_retry
            .is_some_and(|t| t.elapsed() < cooldown)
        {
            return false;
        }
        if retry_model_loads() {
            self.last_model_retry = Some(Instant::now());
            set_status(VoiceStatus::LoadingModels);
            true
        } else {
            false
        }
    }

    /// Self-healing retry, bounded so a persistently failing download cannot loop forever. Unlike
    /// the user-initiated paths ([`handle_start_listening`], [`VoiceCommand::RetryModelLoading`])
    /// this runs on every pipeline iteration without user involvement; it consumes the
    /// [`MAX_AUTO_MODEL_RETRY_CYCLES`] budget and stops once exhausted, further recovery needing
    /// the GUI retry button or a restart.
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

    #[cfg(target_os = "macos")]
    fn check_auto_start(&mut self) {
        // One-shot auto-start, only while `auto_start_pending` (set at pipeline creation, or by
        // handle_start_listening when models were not ready); cleared after the first attempt, so a
        // mic failure cannot retry forever. Failed-state recovery has two paths:
        // handle_start_listening() retries immediately on an explicit start, and the post-select
        // block in run_voice_pipeline retries every iteration, debounced to 30 s. Once the
        // encoder is Ready this picks it up through the pending flag; nothing loads while
        // transcription is disabled, and past `Uninit` the demand is a cheap no-op.
        if self.auto_start_pending && !models_ready() && !is_transcription_disabled() {
            demand_encoder();
        }
        if self.auto_start_pending && models_ready() && !self.is_listening {
            self.auto_start_pending = false;
            send_command(VoiceCommand::StartListening);
        }
    }
}

/// Schedule the transition back to [`VoiceStatus::Listening`] 1.5 s after a successful
/// finalization, respecting the global shutdown token so it cannot write stale state after exit.
#[cfg(target_os = "macos")]
fn schedule_listening_transition(ctx: &mut PipelineCtx) {
    ctx.end_enrollment_attempt();
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

/// Encode raw PCM chunks through the shared encoder into 1024-dim L2-normalized window embeddings,
/// each over its trailing ≤1 s. CPU-bound, so it runs in `spawn_blocking`; failed chunks are
/// skipped.
#[cfg(target_os = "macos")]
async fn extract_negative_embeddings(chunks: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    if chunks.is_empty() {
        return Vec::new();
    }
    let Some(model) = crate::audio::encoder::shared_model_arc() else {
        warn!("extract_negative_embeddings: the encoder is not loaded");
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

/// Centroid-consistency gate for the enrollment set: the prototype is the L2-normalized mean of the
/// accepted embeddings, and it needs ≥ [`MIN_ENROLLMENT_UTTERANCES`] (5) utterances with at least
/// ceil(N × [`ENROLLMENT_CONSISTENCY_MIN_FRACTION`]) of them at cosine ≥
/// [`ENROLLMENT_CONSISTENCY_MIN_SIMILARITY`] (0.70) to the centroid.  Pure, so the gate is testable
/// without the ASR model; `Ok(prototype)` on pass, else `Err` with a user-facing message.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#[cfg(target_os = "macos")]
fn enrollment_consistency_check(utterance_embeddings: &[Vec<f32>]) -> Result<Vec<f32>, String> {
    if utterance_embeddings.len() < MIN_ENROLLMENT_UTTERANCES {
        return Err(format!(
            "Only {} enrollment utterances collected (need ≥{MIN_ENROLLMENT_UTTERANCES}). \
             Speak clearly and close to the microphone.",
            utterance_embeddings.len(),
        ));
    }
    let dim = WAKE_WORD_EMBEDDING_DIM;
    let centroid = normalized_centroid(utterance_embeddings)
        .map_err(|got| format!("Utterance embedding has dim {got} (expected {dim})"))?;

    let required =
        (utterance_embeddings.len() as f32 * ENROLLMENT_CONSISTENCY_MIN_FRACTION).ceil() as usize;
    let passed = utterance_embeddings
        .iter()
        .filter(|emb| cosine_similarity(emb, &centroid) >= ENROLLMENT_CONSISTENCY_MIN_SIMILARITY)
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

/// Finalize enrollment — build the prototype, calibrate against negatives, self-test, persist the
/// v2 record — after Phase 3 completes or times out. `true` on success; on failure the cause is
/// logged, `false` returned, and the caller ends the attempt via
/// [`PipelineCtx::end_enrollment_attempt`].
#[cfg(target_os = "macos")]
async fn finalize_enrollment_pipeline() -> bool {
    if !models_ready() {
        warn!("finalize_enrollment_pipeline: models not ready");
        return false;
    }

    let (utterance_embeddings, negative_audio_chunks, owner_negative_chunks) = {
        let state = voice_state().read().unwrap_poison();
        (
            state.enrollment_embeddings.clone(),
            state.negative_audio_chunks.clone(),
            state.owner_negative_chunks.clone(),
        )
    };

    // Consistency gate → the L2-normalized prototype (centroid) used for calibration.
    let prototype = match enrollment_consistency_check(&utterance_embeddings) {
        Ok(proto) => proto,
        Err(msg) => {
            warn!("Enrollment finalization failed: {msg}");
            set_status(VoiceStatus::Error(format!("Enrollment failed: {msg}")));
            return false;
        }
    };

    // Negative sources are the Phase 3 owner-negative chunks and the ambient non-wake-word chunks.
    // The old Conv1D training set used TTS confusable/unrelated phrases; the prototype-cosine
    // pipeline calibrates on real recordings only, which is simpler and enough for the cosine
    // floor. The two encode batches are independent `spawn_blocking` tasks sharing the read-only
    // model `Arc` (per-call scratch buffers make concurrent forwards safe), so they run in
    // parallel.
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

    {
        let mut state = voice_state().write().unwrap_poison();
        state.negative_audio_chunks.clear();
        state.owner_negative_chunks.clear();
    }

    let calibration = calibrate_negatives(&prototype, &negative_embeddings);
    // Keep the original creation timestamp (re-enrollment does not reset it); `trained_at` is now.
    let created_at = get_enrollment()
        .filter(|e| !e.created_at.is_empty())
        .map_or_else(db::now, |e| e.created_at.clone());
    let trained_at = db::now();

    let Some(enrollment) = WakeWordEnrollment::build(
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

    // Cancel guard: check before persisting.
    if crate::shutdown::shutdown_token().is_cancelled() {
        warn!(
            "finalize_enrollment_pipeline: cancelled during finalization, \
             not persisting enrollment state"
        );
        return false;
    }

    // Self-test: ≥80% of accepted utterances must trigger with a fresh window and no adaptive
    // state.
    if let Err(e) = run_enrollment_self_test(&utterance_embeddings, &enrollment) {
        warn!("Enrollment self-test failed — model rejected: {e}.  Re-enrollment required.");
        set_status(VoiceStatus::Error(format!(
            "Enrollment validation failed: {e}.  Please try again with clearer speech."
        )));
        return false;
    }
    info!("Enrollment self-test: passed — deploying model");

    set_enrollment(enrollment.clone());

    if !persist_enrollment(&enrollment).await {
        warn!("Enrollment persisted to memory but failed to save to config DB");
        // The enrollment is live in memory until exit — name the saving failure, not a wake-word
        // one.
        set_status(VoiceStatus::Error(
            "Wake word enrolled but not saved — it will be lost when the app restarts".to_string(),
        ));
        return false;
    }

    {
        let mut state = voice_state().write().unwrap_poison();
        state.enrollment_embeddings.clear();
        state.enrolled_utterance_count = 0;
        state.utterances_collected = false;
    }

    true
}

/// Run the voice pipeline background task.
#[expect(clippy::too_many_lines)]
pub async fn run_voice_pipeline() {
    info!("Voice pipeline starting...");

    let shutdown_token = crate::shutdown::shutdown_token();
    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<VoiceCommand>();

    {
        let mut state = voice_state().write().unwrap_poison();
        state.cmd_tx = Some(cmd_tx);
    }

    // Load the persisted enrollment (v2 schema).
    #[cfg(target_os = "macos")]
    if let Some(json) = CONFIG.wake_word_templates() {
        match serde_json::from_str::<WakeWordEnrollment>(&json) {
            Ok(enr)
                if enr.schema_version == wake_word::ENROLLMENT_SCHEMA_VERSION
                    && enr.embedding_dim == WAKE_WORD_EMBEDDING_DIM
                    && enr.prototype.len() == WAKE_WORD_EMBEDDING_DIM =>
            {
                set_enrollment(enr);
                info!("Loaded wake word enrollment (v2)");
            }
            Ok(_) | Err(_) => {
                // v1 classifier records and unparseable JSON are rejected outright — no migration,
                // re-enroll.
                warn!(
                    "Stored wake word enrollment is incompatible or legacy (v1). \
                     Re-enrollment required."
                );
            }
        }
    }

    // The status resolves from the recogniser's state via [`resolved_model_status`], re-resolved by
    // the periodic wake-up so the UI cannot hang on LoadingModels. Transcription disabled ⇒ the
    // wake word is disabled too (macOS), since nothing it encodes with ever loads: surface Disabled
    // (not LoadingModels, which would hang) and skip auto-start.
    let transcription_disabled = is_transcription_disabled();
    if transcription_disabled {
        set_status(VoiceStatus::Disabled);
    } else {
        set_status(current_model_status());
    }

    let mut ctx = PipelineCtx::new();
    // A wake-word auto-start is the wake word's own: it waits for the encoder, which is macOS only.
    #[cfg(target_os = "macos")]
    if ctx.auto_start_pending && !transcription_disabled {
        set_enabled(true);
        info!("Voice assistant enabled in config — will auto-start when models are ready");
    }

    // Auto-start now if models are loaded, instead of waiting for the first select! timeout.
    #[cfg(target_os = "macos")]
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
                    #[cfg(target_os = "macos")]
                    Some(VoiceCommand::StartListening) => ctx.handle_start_listening(),
                    // The recording a stop or enrollment aborts is reported to the composer, which
                    // shows the reason.
                    Some(VoiceCommand::StopListening) => ctx.handle_stop_listening(),
                    #[cfg(target_os = "macos")]
                    Some(VoiceCommand::StartEnrollment) => ctx.handle_start_enrollment(),
                    #[cfg(target_os = "macos")]
                    Some(VoiceCommand::CancelEnrollment) => ctx.handle_cancel_enrollment(),
                    Some(VoiceCommand::RetryModelLoading) => {
                        // Explicit retry from GUI — bypass debounce.
                        if retry_model_loads() {
                            ctx.last_model_retry = Some(Instant::now());
                            set_status(VoiceStatus::LoadingModels);
                        } else {
                            warn!("RetryModelLoading: no loader is in the Failed state");
                        }
                    }
                    Some(VoiceCommand::StartRecording) => {
                        if let Some(msg) = ctx.handle_start_button_recording() {
                            // On-demand mic failed — surface it to the user.
                            ctx.broadcast_voice_notice(msg).await;
                        }
                    }
                    Some(VoiceCommand::StopRecordingSend) => {
                        ctx.handle_stop_recording(true).await;
                    }
                    Some(VoiceCommand::StopRecordingDiscard) => {
                        ctx.handle_stop_recording(false).await;
                    }
                    Some(VoiceCommand::ConfigChanged) => {
                        // A runtime re-enable must re-resolve the status and re-arm auto-start
                        // here: the periodic timer is parked while transcription is disabled, so
                        // this wake replaces the 1s tick.
                        if !is_transcription_disabled()
                            && matches!(get_status(), VoiceStatus::Disabled)
                        {
                            // Auto-start was deferred at boot; re-arm so voice reaches Listening
                            // once the ASR model loads.
                            if ctx.auto_start_pending {
                                set_enabled(true);
                            }
                            // Voice enabled while transcription was disabled: the rejected
                            // StartListening cleared the pending start, so re-arm it.
                            if is_enabled() && !ctx.is_listening {
                                ctx.auto_start_pending = true;
                            }
                            set_status(current_model_status());
                            // Fire StartListening now if models are ready; the pending flag keeps
                            // it armed otherwise.
                            #[cfg(target_os = "macos")]
                            ctx.check_auto_start();
                        }
                    }
                    None => break,
                }
            }

            event = async {
                if let Some(mic) = &mut ctx.mic {
                    mic.next().await
                } else {
                    std::future::pending().await
                }
            } => {
                let Some(samples) = event else {
                    // The device is gone: the handler ends everything the microphone was doing
                    // (recording discarded, enrollment dropped) and stops claiming to listen.
                    ctx.handle_microphone_lost();
                    continue;
                };

                // TTS playback gate (macOS only): while spoken replies play, skip ALL processing
                // for the chunk, so TTS echo cannot contaminate the VAD ring buffer, the window
                // embeddings, scoring, enrollment or the recording. The gate stays active for the
                // reverb tail (see `crate::audio::tts::PLAYBACK_REVERB_TAIL_MS`).
                #[cfg(target_os = "macos")]
                if crate::audio::tts::is_playback_active() {
                    continue;
                }

                // Samples flow RAW to the mode dispatch: the old AGC/NS preprocessor is gone, the
                // encoder pipeline needs no preprocessing and enrollment SNR uses the raw floor.
                #[cfg(target_os = "macos")]
                {
                    if ctx.collecting_negatives {
                        handle_negative_collection_audio(&samples, &mut ctx);
                    } else if ctx.enrollment_mode {
                        let (sample, total) = {
                            let state = voice_state().read().unwrap_poison();
                            (state.enrolled_utterance_count, NUM_ENROLLMENT_SAMPLES)
                        };
                        handle_enrollment_audio(&samples, &mut ctx, sample, total);
                    } else if ctx.recording == Some(RecordingOrigin::Button) {
                        ctx.handle_button_recording_audio(&samples).await;
                    } else if ctx.recording.is_some() {
                        // A wake-word recording ends itself on silence.
                        handle_recording_audio(samples, &mut ctx).await;
                    } else {
                        handle_wake_word_detection(&samples, &mut ctx);
                        // A fire starts a recording: acknowledge it with the start cue.
                        if ctx.recording.is_some() {
                            cue::play(Cue::Start);
                        }
                    }
                }
                // Without a wake word there is one kind of recording: the composer's, which keeps
                // everything until the press or the cap ends it.
                #[cfg(not(target_os = "macos"))]
                if ctx.recording.is_some() {
                    ctx.handle_button_recording_audio(&samples).await;
                }
            }

            // Periodic wake-up, so auto-recovery fires when a loader finishes loading or turns
            // Ready/Failed after the initial `select!` entry (check_auto_start runs post-select,
            // not here). While transcription is disabled the timer pends instead of ticking;
            // select! rebuilds it on any loop wake (e.g. `VoiceCommand::ConfigChanged`).
            () = async {
                if is_transcription_disabled() {
                    std::future::pending::<()>().await;
                } else {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            } => {
                // Light polling. A load failure surfaces ModelError from ANY status (a load
                // started by the Settings toggle must not be hidden while voice is off). Otherwise
                // LoadingModels resolves to its terminal state through the same
                // [`resolved_model_status`] the startup block uses, so the footer can never hang on
                // "Loading…" — Listening when voice is on, Disabled when not, whatever started the
                // load.
                if models_failed() {
                    if !matches!(get_status(), VoiceStatus::ModelError) {
                        set_status(VoiceStatus::ModelError);
                    }
                } else if matches!(get_status(), VoiceStatus::LoadingModels) {
                    let resolved = current_model_status();
                    if !matches!(resolved, VoiceStatus::LoadingModels) {
                        set_status(resolved);
                    }
                }
            }

        }

        // Periodic auto-recovery: in Failed state, retry loading (at most every 30s, bounded to
        // [`MAX_AUTO_MODEL_RETRY_CYCLES`] so a failing download cannot loop forever). It runs
        // regardless of auto_start_pending, so the error state heals even when voice is toggled
        // manually.
        if models_failed() {
            ctx.try_retry_models_auto();
        }

        // The wake word's own phases (macOS only): owner-negative collection after the tenth
        // utterance, then finalization on target speech time or the wall-clock timeout.
        #[cfg(target_os = "macos")]
        run_enrollment_phases(&mut ctx).await;

        // Error → Listening once the refractory period elapses, checked non-blocking in the main
        // loop.
        ctx.check_refractory_period();

        // Auto-start when models become ready (async load case).
        #[cfg(target_os = "macos")]
        ctx.check_auto_start();
    }

    info!("Voice pipeline exited");
}

/// The wake word's own phases of the pipeline loop: the Phase 2→3 transition once the tenth
/// utterance is in, the Phase 3 collection and finalization, and one completed enrollment utterance
/// per tick through the encoder.
#[cfg(target_os = "macos")]
async fn run_enrollment_phases(ctx: &mut PipelineCtx) {
    // After all 10 utterances, transition automatically to owner-negative collection.
    let utterances_collected = voice_state().read().unwrap_poison().utterances_collected;
    if utterances_collected && !ctx.collecting_negatives {
        ctx.transition_to_phase3();
    }

    // On target speech time (or the wall-clock timeout): finalize residual audio, build, clean up.
    if ctx.collecting_negatives {
        let target_samples = SAMPLE_RATE as usize * NEGATIVES_TARGET_SECONDS;
        let target_met = ctx.wake.negatives_speech_samples >= target_samples;
        let timed_out = ctx
            .wake
            .phase3_start_time
            .is_some_and(|t| t.elapsed() >= Duration::from_secs(PHASE3_TIMEOUT_SECS));

        if target_met || timed_out {
            // Finalize any residual audio in phase3_audio_buf.
            if !ctx.wake.phase3_audio_buf.is_empty() {
                push_owner_negative_chunk(
                    std::mem::take(&mut ctx.wake.phase3_audio_buf),
                    "residual",
                );
            }
            // Cap is ~360k at 16kHz, well within f64 mantissa precision.
            let collected_secs = {
                #[expect(clippy::cast_precision_loss)]
                {
                    (ctx.wake.negatives_speech_samples as f64) / f64::from(SAMPLE_RATE)
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
                schedule_listening_transition(ctx);
            } else {
                // A failed finalization ends the attempt like a successful one — leaving it open
                // re-enters finalization on every mic chunk.
                ctx.end_enrollment_attempt();
            }
        } else {
            // Update status with current progress.
            let accumulated_secs = ctx.wake.negatives_speech_samples / SAMPLE_RATE as usize;
            let wall_clock_elapsed = ctx
                .wake
                .phase3_start_time
                .map_or(0, |t| t.elapsed().as_secs());
            set_status(VoiceStatus::EnrollingNegatives {
                accumulated_secs,
                target_secs: NEGATIVES_TARGET_SECONDS,
                wall_clock_elapsed,
            });
        }
    }

    // Process pending enrollment utterances, accumulated inline to avoid a race with the command
    // channel; a `VecDeque` keeps every utterance when several complete within one mic frame (one
    // popped per tick). The encoder forward uses `spawn_blocking`. Phase 2 only — Phase 3 handles
    // its audio independently.
    if !utterances_collected && let Some(samples) = ctx.wake.enrollment_pending.pop_front() {
        let noise_rms = ctx.wake.noise_rms_estimate.take();
        handle_enrollment_sample(samples, noise_rms).await;
    }
}

/// Persist an enrollment to the config DB and update the in-memory CONFIG, so GUI snapshot readers
/// and a pipeline restart see it — the per-field persist paths never write `wake_word_templates`.
/// Failures are logged; `true` only when both the DB write and the CONFIG update succeeded, since
/// the only caller ends the attempt on `false` and leaves the enrollment in memory for the session.
#[cfg(target_os = "macos")]
async fn persist_enrollment(enrollment: &WakeWordEnrollment) -> bool {
    let Ok(json) = serde_json::to_string(enrollment) else {
        warn!("Failed to serialize wake word enrollment for persistence");
        return false;
    };
    let store = crate::config_db::store();
    if let Err(e) = store.set_kv(CONFIG_KEY_WAKE_WORD_TEMPLATES, &json).await {
        warn!("Failed to persist wake word enrollment: {e}");
        return false;
    }
    if !CONFIG.set_string_field(CONFIG_KEY_WAKE_WORD_TEMPLATES, &json) {
        warn!(
            "Failed to update CONFIG with wake word enrollment (key not recognized by \
             set_string_field — it may have drifted from the `stringify!` arms)"
        );
        return false;
    }
    true
}

/// Audio during wake-word command recording (post-detection). The opening
/// [`POST_FIRE_DISCARD_SAMPLES`] are dropped, so recognition starts after the trigger — the
/// pre-fire audio is already gone, and only a phrase tail survives a fire that landed mid-phrase.
/// The rest accumulates into `command_buffer`, silence tracked by sample count (not wall-clock, so
/// load cannot skew the cutoff), and the recording stops after [`SILENCE_THRESHOLD_SAMPLES`] of
/// silence or the [`MAX_RECORD_SECS`] cap, then transcribes and publishes. A recording whose
/// retained audio holds no speech delivers nothing at all; a button recording never reaches this
/// gate.
#[cfg(target_os = "macos")]
#[expect(clippy::cast_precision_loss)]
async fn handle_recording_audio(samples: Vec<f32>, ctx: &mut PipelineCtx) {
    // Discard the armed opening before keeping anything — but still feed it to the VAD: the
    // detector is stateful and tracks the live stream, so skipping it would leave the trigger
    // phrase in its context and make the first retained frames read as speech, wrongly satisfying
    // the no-speech gate.
    let discarded = ctx.wake.post_fire_discard_samples.min(samples.len());
    ctx.wake.post_fire_discard_samples -= discarded;
    if discarded > 0 {
        // The verdict is intentionally unused — this only advances the detector.
        is_speech_with_threshold(&samples[..discarded], ctx.wake.vad_threshold);
    }
    let samples = &samples[discarded..];
    if samples.is_empty() {
        return;
    }

    ctx.command_buffer.extend_from_slice(samples);
    let speech = is_speech_with_threshold(samples, ctx.wake.vad_threshold);
    if speech {
        ctx.silence_sample_count = 0;
        ctx.wake.recording_had_speech = true;
    } else {
        // Accumulate silence by raw chunk size: each call gets a variable-size chunk from the mic.
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

        // The recording has ended — acknowledge that whatever follows (silence, the cap, a failed
        // transcription) — and clear its state, so the outcome below owns no reset.
        let (cmd_buf, had_speech) = ctx.take_wake_recording();
        ctx.finish_wake_recording(cmd_buf, had_speech).await;
    }
}

/// Reject an enrollment utterance shorter than 400ms: that drops noise blips and coughs while
/// accepting any real wake word. The floor is wall-clock duration, since the encoder produces
/// exactly one window embedding per utterance and 400 ms is well below the ~1s a wake word needs.
#[cfg(target_os = "macos")]
fn check_enrollment_utterance_length(duration_ms: u64) -> Result<(), String> {
    if duration_ms < ENROLLMENT_QUALITY_DURATION_MIN_MS {
        Err(format!(
            "Utterance too short ({duration_ms}ms) — speak longer"
        ))
    } else {
        Ok(())
    }
}

/// Encode a completed enrollment utterance and store its 1024-dim embedding. Encoded ONCE — the
/// prototype-cosine pipeline needs no PCM augmentation; utterances under 400 ms are rejected and
/// the enrollment status is updated with per-utterance quality.
#[cfg(target_os = "macos")]
#[expect(clippy::cast_precision_loss)]
async fn handle_enrollment_sample(samples: Vec<f32>, noise_rms: Option<f32>) {
    if !models_ready() {
        warn!("Models not ready for enrollment");
        return;
    }

    let duration_ms = samples_to_ms(samples.len(), SAMPLE_RATE);

    if let Err(msg) = check_enrollment_utterance_length(duration_ms) {
        warn!("{msg}");
        set_status(VoiceStatus::Error(msg));
        return;
    }

    // Quality is assessed before `samples` moves into the encoder task.
    let quality = compute_utterance_quality(&samples, noise_rms);

    let Some(model) = crate::audio::encoder::shared_model_arc() else {
        warn!("The encoder is not loaded — skipping enrollment sample");
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
        // All 10 collected: signal Phase 2 completion so the main loop starts owner-negative
        // collection.
        voice_state().write().unwrap_poison().utterances_collected = true;
    } else {
        set_status(VoiceStatus::Enrolling {
            sample: utterance_count,
            total: NUM_ENROLLMENT_SAMPLES,
            duration_ms,
            quality: Some(quality),
        });
    }
}

/// Append `samples` to `buf`, draining the front past `cap`; returns how many were drained so
/// callers can shift their cursors.
#[cfg(target_os = "macos")]
fn push_capped(buf: &mut Vec<f32>, samples: &[f32], cap: usize) -> usize {
    buf.extend_from_slice(samples);
    let excess = buf.len().saturating_sub(cap);
    buf.drain(..excess);
    excess
}

/// Handle a chunk of raw mic audio during wake-word detection. Within [`WAKE_WORD_COOLDOWN`] of a
/// detection, audio only accumulates into the raw ring ([`AUDIO_BUFFER_MAX`]). Each [`HOP_LENGTH`]
/// hop then goes to the global Earshot VAD, which counts per-segment silence and lets
/// [`PipelineCtx::handle_segment_boundary`] reset the segment at [`SEGMENT_TIMEOUT_HOPS`] (never
/// while recording); no mel frames are built — the raw ring is the VAD input only. When
/// [`SCORE_STRIDE_SAMPLES`] have been processed since the last step and either speech was seen in
/// this call or the rolling window is mid-utterance, the trailing ≤1 s of the VAD-gated
/// [`Wake::speech_window`] is encoded ([`crate::audio::wake_word::encode_window`]) and
/// scored ([`score_single_embedding`]); a fire counts only within the first
/// [`WAKE_WORD_MAX_SEGMENT_STEPS`] steps of the segment ([`Wake::segment_score_steps`]) and
/// a later match is refused. On detection the raw ring is dropped — all of it is pre-fire audio —
/// and the recording discards its own opening ([`POST_FIRE_DISCARD_SAMPLES`]).
#[cfg(target_os = "macos")]
fn handle_wake_word_detection(samples: &[f32], ctx: &mut PipelineCtx) {
    // Cooldown: skip ALL processing for the chunk, so a rapid consecutive false trigger cannot
    // follow. Audio still accumulates into `audio_buffer` (capped), so the pipeline has data the
    // moment the cooldown expires.
    if let Some(last) = ctx.wake.last_wake_word_detection
        && last.elapsed() < WAKE_WORD_COOLDOWN
    {
        debug!(
            "Wake word cooldown active ({}ms elapsed)",
            last.elapsed().as_millis()
        );
        push_capped(&mut ctx.wake.audio_buffer, samples, AUDIO_BUFFER_MAX);
        return;
    }

    let excess = push_capped(&mut ctx.wake.audio_buffer, samples, AUDIO_BUFFER_MAX);
    ctx.wake.vad_cursor = ctx.wake.vad_cursor.saturating_sub(excess);

    // Iterate [`HOP_LENGTH`] frames over the UNCONSUMED audio (from `vad_cursor`), feeding each
    // frame's NEW hop to the global VAD: the 512-sample frame overlaps the previous by 256, and
    // feeding the full frame would double-feed and corrupt Earshot's ring buffer. No mel frames are
    // built — this loop only tracks speech and per-segment silence, and scoring encodes the
    // VAD-gated `speech_window`, not this ring. The ring is NOT drained: `vad_cursor` marks the
    // consumed prefix and the overlap tail (up to [`FRAME_LENGTH`] - [`HOP_LENGTH`]) must survive
    // into the next call.
    let mut speech_seen_this_call = false;
    // Seeded with the count from previous calls, so the VAD-negative hop counter stays continuous.
    let mut hop_count = ctx.wake.segment_silence_hops;

    while ctx.wake.vad_cursor + FRAME_LENGTH <= ctx.wake.audio_buffer.len() {
        let frame = &ctx.wake.audio_buffer[ctx.wake.vad_cursor..ctx.wake.vad_cursor + FRAME_LENGTH];
        // The global [`VAD_DETECTOR`] keeps continuous noise-floor and ring state across the live
        // stream.
        let is_speech = is_speech_with_threshold(&frame[..HOP_LENGTH], VAD_THRESHOLD);
        if is_speech {
            speech_seen_this_call = true;
            hop_count = 0;
            // Append only the VAD-positive hop, so the encoder sees speech-only audio like
            // enrollment utterances; capped at [`WAKE_WORD_WINDOW_SAMPLES`], draining the front.
            push_capped(
                &mut ctx.wake.speech_window,
                &frame[..HOP_LENGTH],
                WAKE_WORD_WINDOW_SAMPLES,
            );
        } else {
            hop_count += 1;
        }
        ctx.wake.vad_cursor += HOP_LENGTH;
        ctx.wake.last_score_sample_count += HOP_LENGTH;
    }

    // Reset the rolling window and adaptive threshold once consecutive silence reaches
    // [`SEGMENT_TIMEOUT_HOPS`], so scores cannot accumulate across utterances. Runs BEFORE scoring,
    // so a just-fired boundary starts a fresh segment.
    ctx.handle_segment_boundary(hop_count);

    // Score when [`SCORE_STRIDE_SAMPLES`] were processed since the last encode AND speech was seen
    // in this call, or the rolling window is non-empty (mid-utterance continuation across a VAD
    // gap).
    if ctx.recording.is_none()
        && ctx.wake.last_score_sample_count >= SCORE_STRIDE_SAMPLES
        && (speech_seen_this_call || !ctx.wake.score_window.is_empty())
    {
        ctx.wake.last_score_sample_count = 0;

        // No enrollment installed → skip the step: the heavy 18-layer forward is gated behind both
        // the enrollment and the model, so listening without a deployed wake word does no
        // per-stride encoding.
        let Some(enrollment) = voice_state().read().unwrap_poison().enrollment.clone() else {
            return;
        };

        let Some(model) = crate::audio::encoder::shared_model_arc() else {
            // The encoder is not loaded — no scoring this step; the status is handled elsewhere.
            return;
        };

        // The segment clock the fire-position cut reads advances only on steps that really score: a
        // stride that bailed out above must not shorten the accept window.
        ctx.wake.segment_score_steps += 1;

        // Take the trailing ≤[`WAKE_WORD_WINDOW_SAMPLES`] of the VAD-gated speech window, not the
        // raw ring: speech-only audio keeps the cosine distribution aligned with enrollment, while
        // raw trailing audio dilutes the embedding and shifts positives below the calibration
        // floor.
        let start = ctx
            .wake
            .speech_window
            .len()
            .saturating_sub(WAKE_WORD_WINDOW_SAMPLES);
        let window = &ctx.wake.speech_window[start..];

        let embedding = crate::util::with_block_in_place(|| encode_window(&model, window));
        match embedding {
            Ok(embedding) => {
                let (detected, ..) = score_single_embedding(
                    &embedding,
                    Some(&enrollment),
                    &mut ctx.wake.score_window,
                    Some(&mut ctx.wake.adaptive_threshold),
                    ctx.wake.adaptive_k,
                );

                // A recognised phrase is a person at the computer, even when the
                // position gate below refuses the fire.
                if detected {
                    crate::user_activity::report_activity();
                }

                // A fire counts only near the start of its segment; past that position the match is
                // inert (the position only grows within a segment, so a refused match stays refused
                // until a boundary or a handled detection clears the count).
                if detected && ctx.wake.segment_score_steps <= WAKE_WORD_MAX_SEGMENT_STEPS {
                    // Fire now — the handoff below completes the transition to recording.
                    ctx.recording = Some(RecordingOrigin::WakeWord);
                    publish_recording(RecordingEvent::Started);
                    ctx.wake.last_wake_word_detection = Some(Instant::now());
                    set_status(VoiceStatus::RecordingWakeWord);
                }
            }
            Err(e) => {
                warn!("Wake word window encoding failed: {e}");
            }
        }
    }

    // Detection→recording handoff: the ring is dropped, not seeded into the recording — everything
    // it holds predates the fire, trigger phrase included, and the command must start after the
    // phrase (a trailing fragment can survive when the phrase is still being spoken at the fire). A
    // Soft reset clears the detection state (ring, speech window, score window) while preserving
    // VAD continuity, the adaptive threshold and the cooldown; the recording then discards its own
    // opening.
    if ctx.recording.is_some() {
        ctx.reset_pipeline_state(ResetLevel::Soft);
        ctx.wake.post_fire_discard_samples = POST_FIRE_DISCARD_SAMPLES;
    }
}

/// Handle audio during enrollment: accumulate VAD decisions and raw audio into `frame_vad` /
/// `frame_raw_audio`, then run [`segment_utterances_by_vad`] over the accumulated buffer and queue
/// newly completed utterances in `enrollment_pending` for the main loop (one per tick, so no
/// command-channel race and nothing dropped when several complete within one mic frame). The frame
/// loop keeps inline state only for side effects: noise RMS capture, negative audio collection and
/// UI status. Non-VAD frames before the first speech and between utterances accumulate into
/// `negative_audio_buf` and are saved as a chunk at the first sustained speech, giving real
/// (non-synthetic) negatives to calibrate on. Only VAD-positive frames become utterances, mirroring
/// the detection pipeline, and utterance end follows [`ENROLLMENT_SILENCE_THRESHOLD_SAMPLES`]
/// (~304ms), matching the detection-side segment timeout. No speech for ~5 s raises the "speak
/// louder" warning; the status otherwise tracks the phase.
#[cfg(target_os = "macos")]
#[expect(clippy::too_many_lines)] // `ctx.wake.` pushes the frame loop past the line cap
fn handle_enrollment_audio(samples: &[f32], ctx: &mut PipelineCtx, sample: usize, total: usize) {
    ctx.wake.audio_buffer.extend_from_slice(samples);

    // Store the ORIGINAL mic samples (not sub-sampled, not framed), so [`segment_utterances_by_vad`]
    // can read frames at hop stride aligned with `ctx.wake.frame_vad` at full 16 kHz.
    ctx.wake.frame_raw_audio.extend_from_slice(samples);

    let len = ctx.wake.audio_buffer.len();
    let mut consumed = 0;
    while consumed + FRAME_LENGTH <= len {
        let frame = &ctx.wake.audio_buffer[consumed..consumed + FRAME_LENGTH];

        // Feed only the NEW hop, not the full 512-sample frame, so Earshot's internal state is not
        // double-fed; prefer the context-local detector, falling back to the global one.
        let is_speech = enrollment_is_speech(
            &mut ctx.wake.enrollment_vad,
            ctx.wake.vad_threshold,
            &frame[..HOP_LENGTH],
        );
        ctx.wake.frame_vad.push(is_speech);

        if is_speech {
            ctx.wake.vad_positives_in_a_row += 1;
            // Reset the no-speech warning counter on any VAD-positive frame.
            ctx.wake.enrollment_no_speech_frame_count = 0;

            if ctx.wake.vad_positives_in_a_row >= ENROLLMENT_VAD_CONSECUTIVE_REQUIRED {
                // Sustained speech confirmed: gate the real-time side effects that must run inside
                // the frame loop — noise RMS capture, negative audio collection, UI status.
                // Boundary detection belongs to [`segment_utterances_by_vad`].

                let was_waiting_for_silence = ctx.wake.utterance_silence_samples > 0;

                // First transition from silence to sustained speech: capture the ambient noise RMS
                // from the raw ring (no AGC in this pipeline, so that is the true room floor).
                let already_had_speech = ctx.wake.utterance_had_speech;
                // Same transition: save the accumulated non-wake-word audio (pre-enrollment
                // ambience or inter-utterance silence) as a potential negative calibration example.
                if !already_had_speech {
                    if ctx.wake.negative_audio_buf.len() >= MIN_NEGATIVE_AUDIO_LEN {
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
                            .push(std::mem::take(&mut ctx.wake.negative_audio_buf));
                    } else {
                        ctx.wake.negative_audio_buf.clear();
                    }
                }
                if !already_had_speech && ctx.wake.noise_rms_estimate.is_none() {
                    let speech_boundary = ENROLLMENT_VAD_CONSECUTIVE_REQUIRED * HOP_LENGTH;
                    let pre_speech_end =
                        ctx.wake.audio_buffer.len().saturating_sub(speech_boundary);
                    if pre_speech_end > 0 {
                        // Shared RMS helper.
                        let rms =
                            super::util::compute_rms(&ctx.wake.audio_buffer[..pre_speech_end]);
                        ctx.wake.noise_rms_estimate = Some(rms);
                    }
                }

                if !already_had_speech || was_waiting_for_silence {
                    // Silence → speech, or speech resumed inside 1.5 s: show "Listening…".
                    set_status(VoiceStatus::ListeningDuringEnrollment { sample, total });
                }
                ctx.wake.utterance_had_speech = true;
                ctx.wake.utterance_silence_samples = 0;
            } else if ctx.wake.utterance_had_speech {
                // One VAD-positive frame after sustained speech: just reset silence, which rides
                // over brief VAD gaps in continuous speech (unvoiced stops).
                ctx.wake.utterance_silence_samples = 0;
            }
        } else {
            // VAD-negative: reset consecutive counter.
            ctx.wake.vad_positives_in_a_row = 0;

            if ctx.wake.utterance_had_speech {
                // After speech: track silence by sample count (not wall-clock, so load cannot skew
                // the cutoff).
                ctx.wake.utterance_silence_samples += HOP_LENGTH;

                // Inline silence reset, partly duplicating the extracted function boundary logic,
                // so post-utterance silence in the same chunk still accumulates into
                // `negative_audio_buf`. It snapshots before resetting, so the status gate reads the
                // pre-reset value — at zero it would spuriously show "waiting for silence" for one
                // frame.
                let silence_ui_check = ctx.wake.utterance_silence_samples;

                if ctx.wake.utterance_silence_samples >= ENROLLMENT_SILENCE_THRESHOLD_SAMPLES {
                    ctx.wake.utterance_had_speech = false;
                    ctx.wake.utterance_silence_samples = 0;
                    ctx.wake.enrollment_no_speech_frame_count = 0;
                    ctx.wake.vad_positives_in_a_row = 0;
                }

                // First 200ms of silence shows "Keep silent to confirm…", using the pre-reset
                // snapshot.
                if silence_ui_check < SILENCE_UI_GATE_SAMPLES {
                    set_status(VoiceStatus::WaitingForSilenceDuringEnrollment { sample, total });
                }
            } else {
                // Accumulate non-VAD audio for negatives (pre-enrollment noise, inter-utterance
                // silence), each frame contributing [`HOP_LENGTH`] new samples.
                ctx.wake
                    .negative_audio_buf
                    .extend_from_slice(&frame[..HOP_LENGTH]);

                // Pre-speech silence: count consecutive non-VAD frames and, at
                // [`ENROLLMENT_NO_SPEECH_TIMEOUT_FRAMES`] (~5 s), warn so the user speaks louder or
                // moves closer.
                ctx.wake.enrollment_no_speech_frame_count += 1;
                // Warn after the derived frame threshold.
                if ctx.wake.enrollment_no_speech_frame_count >= ENROLLMENT_NO_SPEECH_TIMEOUT_FRAMES
                {
                    set_status(VoiceStatus::Error(
                        "No speech detected — try speaking louder or move closer to microphone"
                            .to_string(),
                    ));
                    // Do not reset the counter: the status persists until VAD fires or enrollment
                    // restarts.
                }
            }
        }
        consumed += HOP_LENGTH;
    }

    // Single O(remaining) drain instead of O(remaining) per frame iteration.
    if consumed > 0 {
        ctx.wake.audio_buffer.drain(..consumed);
    }

    // Hand the full accumulated audio and VAD decisions to the same boundary algorithm as the
    // inline logic; `emitted_utterances` counts what was already processed, so newly completed ones
    // are those with index >= it.
    if !ctx.wake.frame_vad.is_empty() {
        let utterances = segment_utterances_by_vad(
            &ctx.wake.frame_raw_audio,
            &ctx.wake.frame_vad,
            &DEFAULT_VAD_SEGMENTATION_CONFIG,
        );

        while utterances.len() > ctx.wake.emitted_utterances {
            let new_idx = ctx.wake.emitted_utterances;
            ctx.wake.emitted_utterances += 1;

            // Queue it in `enrollment_pending` for the main loop, keeping every utterance when
            // several complete within one mic frame.
            let utterance = utterances[new_idx].clone();
            ctx.wake.enrollment_pending.push_back(utterance);

            ctx.wake.utterance_had_speech = false;
            ctx.wake.utterance_silence_samples = 0;
            ctx.wake.vad_positives_in_a_row = 0;
            ctx.wake.enrollment_no_speech_frame_count = 0;
            // `noise_rms_estimate` is intentionally not reset here: the main loop consumes it
            // later.
        }
    }
}

/// Append an owner-negative chunk under the [`MAX_OWNER_NEGATIVE_SAMPLES`] cap (dropping with a
/// warning); `label` distinguishes the final residual buffer from speech segments in that warning.
#[cfg(target_os = "macos")]
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

/// Outcome of one [`process_phase3_frames`] pass: the accumulation state carried across mic chunks
/// plus the segments finalized in the pass.
#[cfg(target_os = "macos")]
struct Phase3Progress {
    /// Samples at the head of the buffer already fed to the VAD (next unprocessed frame start).
    processed: usize,
    /// VAD-negative samples since the last positive frame or chunk boundary.
    silence_samples: usize,
    negatives_speech_samples: usize,
    /// Segments finalized this pass (whole segments, never hop slivers); the caller decides where
    /// they land.
    completed_chunks: Vec<Vec<f32>>,
}

/// Pure Phase-3 frame processing over caller-supplied VAD decisions, mirroring the
/// [`segment_utterances_by_vad`] pattern: all state is threaded explicitly and the decision per hop
/// is injected, so tests drive segmentation deterministically instead of relying on the stateful
/// Earshot detector over synthetic audio.
///
/// Frames start at the `processed` watermark, not the buffer head, so each hop reaches the VAD —
/// and the speech counter — exactly once across mic chunks. The old code restarted at the head
/// every call and drained only at a chunk boundary, so during continuous speech every chunk
/// re-counted the whole buffer (quadratic counter → Phase 3 completed after ~2–4 real seconds). The
/// counter must stay a monotone 1:1 record of real audio: after the drain/watermark adjustment it
/// neither re-counts nor under-counts (which would silently fall back to the 120 s timeout path).
///
/// The drain is gated on a closed segment: while no boundary has fired (`segment_start == 0`)
/// nothing is drained, so the open segment and its unprocessed VAD-window overlap survive across
/// chunks; when one fires, the segment is pushed and the drain removes everything up to it, and the
/// silence-run reset keeps later pushes aligned. `buf` is the caller-extended audio buffer, drained
/// in place.
#[cfg(target_os = "macos")]
fn process_phase3_frames(
    buf: &mut Vec<f32>,
    processed: usize,
    silence_samples: usize,
    negatives_speech_samples: usize,
    mut vad: impl FnMut(&[f32]) -> bool,
) -> Phase3Progress {
    let len = buf.len();
    // Resume at the watermark: everything before it was fed in a previous call and must not be
    // re-counted.
    let mut consumed = processed;
    // Segment start is always 0 on entry: a call that closed a segment drained everything up to the
    // boundary.
    let mut segment_start = 0;
    let mut silence_samples = silence_samples;
    let mut negatives_speech_samples = negatives_speech_samples;
    let mut completed_chunks: Vec<Vec<f32>> = Vec::new();

    while consumed + FRAME_LENGTH <= len {
        let is_speech = vad(&buf[consumed..consumed + HOP_LENGTH]);

        if is_speech {
            // 1:1 accounting: this hop contributes exactly [`HOP_LENGTH`] samples, exactly once.
            negatives_speech_samples += HOP_LENGTH;
            silence_samples = 0;
        } else {
            silence_samples += HOP_LENGTH;
            // Chunk boundary: silence past [`ENROLLMENT_SILENCE_THRESHOLD_SAMPLES`] (~304ms,
            // aligned to streaming) after sustained speech finalizes the current segment as a
            // chunk.
            if silence_samples >= ENROLLMENT_SILENCE_THRESHOLD_SAMPLES {
                let chunk_end = consumed.saturating_sub(silence_samples);
                if chunk_end > segment_start {
                    completed_chunks.push(buf[segment_start..chunk_end].to_vec());
                }
                // Advance past the silence boundary so the next segment starts after this silence.
                segment_start = consumed;
                // Reset the silence run at each boundary, or it keeps growing, `chunk_end` drifts
                // below `segment_start` and the next push is delayed and misaligned.
                silence_samples = 0;
            }
        }

        consumed += HOP_LENGTH;
    }

    // Drain everything before `segment_start`, keeping the unfinalized segment and the unprocessed
    // VAD-window overlap. What is drained was already pushed as a chunk or is pure lead-in; this
    // also closes the drain window, so the next call enters with `segment_start = 0` at the new
    // buffer head.
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

/// Process incoming audio for Phase 3: extend [`Wake::phase3_audio_buf`] with the mic chunk,
/// then run [`process_phase3_frames`], feeding each hop to the Phase-3 VAD exactly once (stateful
/// Earshot when available, else the global fallback). Its buffers are independent of
/// `negative_audio_buf` and `utterance_silence_samples`; it does not accumulate `frame_raw_audio` /
/// `frame_vad` (only `segment_utterances_by_vad` consumes those) and does not reset
/// `enrollment_no_speech_frame_count` (no no-speech warnings in Phase 3). Boundaries use
/// [`ENROLLMENT_SILENCE_THRESHOLD_SAMPLES`], and finished chunks land in
/// `voice_state().owner_negative_chunks` via [`push_owner_negative_chunk`] (which enforces the
/// memory cap) so they survive mic resets. `negatives_speech_samples` advances by this call
/// VAD-positive frames.
#[cfg(target_os = "macos")]
fn handle_negative_collection_audio(samples: &[f32], ctx: &mut PipelineCtx) {
    ctx.wake.phase3_audio_buf.extend_from_slice(samples);
    let progress = process_phase3_frames(
        &mut ctx.wake.phase3_audio_buf,
        ctx.wake.phase3_processed,
        ctx.wake.phase3_silence_samples,
        ctx.wake.negatives_speech_samples,
        |hop| enrollment_is_speech(&mut ctx.wake.enrollment_vad, ctx.wake.vad_threshold, hop),
    );
    ctx.wake.phase3_processed = progress.processed;
    ctx.wake.phase3_silence_samples = progress.silence_samples;
    ctx.wake.negatives_speech_samples = progress.negatives_speech_samples;
    for chunk in progress.completed_chunks {
        push_owner_negative_chunk(chunk, "speech");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    // Tests for the pure [`segment_utterances_by_vad`]: synthetic audio, hand-computed VAD
    // decisions.

    /// Test config with a shorter silence threshold (10 frames ≈ 2560 samples
    /// instead of the default ~304 ms ≈ 4864 samples, see
    /// [`ENROLLMENT_SILENCE_THRESHOLD_SAMPLES`]) so tests don't need
    /// prohibitively long audio buffers.
    #[cfg(target_os = "macos")]
    const TEST_VAD_CONFIG: VadSegmentationConfig = VadSegmentationConfig {
        frame_length: FRAME_LENGTH,
        hop_length: HOP_LENGTH,
        consecutive_required: ENROLLMENT_VAD_CONSECUTIVE_REQUIRED,
        silence_threshold_samples: HOP_LENGTH * 10, // 2560 samples ≈ 10 frames
    };

    /// Build a raw audio buffer long enough for `n_frames` frames at
    /// [`HOP_LENGTH`] stride with [`FRAME_LENGTH`] window size.
    #[cfg(target_os = "macos")]
    fn audio_for_frames(n_frames: usize) -> Vec<f32> {
        let last_end = (n_frames.saturating_sub(1)) * HOP_LENGTH + FRAME_LENGTH;
        vec![0.0f32; last_end]
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn segment_utterances_by_vad_table() {
        // Table-driven coverage of [`segment_utterances_by_vad`].  Each case
        // describes the per-frame VAD decision sequence as (n_frames, is_speech)
        // runs; the total frame count derives the audio buffer length.
        struct Case {
            name: &'static str,
            /// `(n_frames, is_speech)` runs describing the per-frame VAD
            /// decision sequence.  The total is the number of frames.
            vad_runs: &'static [(usize, bool)],
            expected_utterances: usize,
            /// When true, additionally assert each emitted utterance has
            /// non-empty audio samples.
            expect_non_empty: bool,
        }

        let cases = [
            Case {
                // No VAD-positive frames at all → no utterances.
                name: "no_speech_returns_empty",
                vad_runs: &[(10, false)],
                expected_utterances: 0,
                expect_non_empty: false,
            },
            Case {
                // 4 speech frames (≥1 consecutive → sustained immediately) +
                // 10 silence frames (≥10 → threshold).
                name: "single_utterance_detected",
                vad_runs: &[(4, true), (10, false)],
                expected_utterances: 1,
                expect_non_empty: true,
            },
            Case {
                // 4 speech, 10 silence, 3 speech, 10 silence → 2 utterances.
                name: "multiple_utterances_separated_by_silence",
                vad_runs: &[(4, true), (10, false), (3, true), (10, false)],
                expected_utterances: 2,
                expect_non_empty: true,
            },
            Case {
                // 8 silence frames, then 4 speech frames at end — no trailing
                // silence to cross the threshold, so no utterance is emitted.
                name: "utterance_at_end_without_silence_not_emitted",
                vad_runs: &[(8, false), (4, true)],
                expected_utterances: 0,
                expect_non_empty: false,
            },
        ];

        for case in &cases {
            let n_frames: usize = case.vad_runs.iter().map(|(count, _)| *count).sum();
            let audio = audio_for_frames(n_frames);
            let mut vad = Vec::with_capacity(n_frames);
            for &(count, is_speech) in case.vad_runs {
                vad.extend(std::iter::repeat_n(is_speech, count));
            }
            let utterances = segment_utterances_by_vad(&audio, &vad, &TEST_VAD_CONFIG);
            assert_eq!(
                utterances.len(),
                case.expected_utterances,
                "case: {name}",
                name = case.name,
            );
            if case.expect_non_empty {
                for (i, utt) in utterances.iter().enumerate() {
                    assert!(
                        !utt.is_empty(),
                        "case: {name} — utterance {i} should contain audio samples",
                        name = case.name,
                    );
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(voice)]
    fn refractory_period_transition_table() {
        let _ = init_global();

        // (case, timer elapsed?, listening?, recording?, initial status,
        //  expected status check, timer cleared?)
        #[expect(clippy::type_complexity)] // refractory transition table
        let cases: [(
            &str,
            bool,
            bool,
            bool,
            VoiceStatus,
            fn(&VoiceStatus) -> bool,
            bool,
        ); 5] = [
            (
                "elapsed_error_to_listening",
                true,
                true,
                false,
                VoiceStatus::Error("test error".to_string()),
                |s| matches!(s, VoiceStatus::Listening),
                true,
            ),
            (
                "elapsed_disabled_stays",
                true,
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
                true,
                VoiceStatus::Error("test error".to_string()),
                |s| matches!(s, VoiceStatus::Error(_)),
                true,
            ),
            (
                "future_timer_preserved",
                false,
                true,
                false,
                VoiceStatus::Error("test error".to_string()),
                |s| matches!(s, VoiceStatus::Error(_)),
                false,
            ),
            (
                // Nothing to return to: the microphone is gone, so an Error
                // status (e.g. a failed Enroll) must not become Listening.
                "elapsed_not_listening_stays_error",
                true,
                false,
                false,
                VoiceStatus::Error("test error".to_string()),
                |s| matches!(s, VoiceStatus::Error(_)),
                true,
            ),
        ];

        for (name, timer_elapsed, is_listening, is_recording, initial, expect, timer_cleared) in
            cases
        {
            let mut ctx = PipelineCtx::new();
            ctx.is_listening = is_listening;
            ctx.recording = is_recording.then_some(RecordingOrigin::Button);
            ctx.refractory_until = Some(if timer_elapsed {
                Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .expect("1s in the past should not underflow")
            } else {
                Instant::now()
                    .checked_add(Duration::from_mins(1))
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

    // The StartRecording/StopRecording interactions that need no real
    // microphone: reject guards, the resume/teardown transitions in
    // [`PipelineCtx::end_button_recording`], and the outcomes the composer sees.

    /// Both guards that close the record button: a recording already running
    /// (its window owns the composer) and an enrollment that owns the mic.
    #[test]
    #[serial_test::serial(voice)]
    fn a_start_press_never_displaces_a_running_recording_or_an_enrollment() {
        let _ = init_global();

        // A recording is in progress: refused, and the running recording is
        // left exactly as it was.
        let mut ctx = PipelineCtx::new();
        ctx.recording = Some(RecordingOrigin::Button);
        set_status(VoiceStatus::RecordingButton);

        ctx.handle_start_button_recording();

        assert_eq!(ctx.recording, Some(RecordingOrigin::Button));
        assert!(!ctx.resume_listening_after_recording);
        assert!(matches!(get_status(), VoiceStatus::RecordingButton));

        // Enrollment owns the mic: refused without touching global state.
        let mut ctx = PipelineCtx::new();
        ctx.enrollment_mode = true;
        set_status(VoiceStatus::Listening);

        ctx.handle_start_button_recording();

        assert!(ctx.recording.is_none());
        assert!(matches!(get_status(), VoiceStatus::Listening));
    }

    /// [`PipelineCtx::end_button_recording`] resolves three starting states: a
    /// wake-word assistant to return to, a recording-only mic to tear down, and
    /// a pending auto-start that teardown must not cancel.
    #[cfg(target_os = "macos")]
    #[test]
    #[serial_test::serial(voice)]
    fn the_end_press_consumes_the_resume_request_and_leaves_the_mic_as_it_was() {
        struct Case {
            name: &'static str,
            is_listening: bool,
            auto_start_pending: bool,
        }

        let _ = init_global();
        let cases = [
            Case {
                name: "resumes_wake_word_listening",
                is_listening: true,
                auto_start_pending: false,
            },
            Case {
                name: "tears_down_recording_only_mic",
                is_listening: false,
                auto_start_pending: false,
            },
            Case {
                // Voice enabled in config but models still loading → auto-start
                // is pending, and a button recording must not cancel it.
                name: "preserves_auto_start_pending",
                is_listening: false,
                auto_start_pending: true,
            },
        ];

        for case in cases {
            let mut ctx = PipelineCtx::new();
            ctx.is_listening = case.is_listening;
            ctx.auto_start_pending = case.auto_start_pending;
            ctx.recording = Some(RecordingOrigin::Button);
            // A button recording that started while listening always holds the
            // resume request; ending it consumes that request either way.
            ctx.resume_listening_after_recording = case.is_listening;
            set_status(VoiceStatus::RecordingButton);

            ctx.end_button_recording();

            assert!(
                ctx.recording.is_none(),
                "case {}: recording ended",
                case.name
            );
            assert_eq!(
                ctx.is_listening, case.is_listening,
                "case {}: listening state",
                case.name,
            );
            assert_eq!(
                ctx.auto_start_pending, case.auto_start_pending,
                "case {}: the pending auto-start survives teardown",
                case.name,
            );
            if case.is_listening {
                assert!(
                    !ctx.resume_listening_after_recording,
                    "case {}: the resume request is consumed",
                    case.name,
                );
                assert!(
                    matches!(get_status(), VoiceStatus::Listening),
                    "case {}: the wake-word mic resumes",
                    case.name,
                );
            } else {
                assert!(ctx.mic.is_none(), "case {}: mic torn down", case.name);
                assert!(
                    matches!(get_status(), VoiceStatus::Disabled),
                    "case {}: nothing is left listening",
                    case.name,
                );
            }
        }
    }

    // The device-loss scenarios themselves need real hardware; these cover the
    // state transitions the loss handler performs.

    /// A loss abandons the attempt completely — nothing may be built or stored
    /// from it later — and voice rests on the user's switch: listening when it
    /// is on, Disabled when it is off.
    #[cfg(target_os = "macos")]
    #[test]
    #[serial_test::serial(voice)]
    fn microphone_loss_abandons_the_attempt_and_rests_on_the_switch() {
        let _ = init_global();
        set_enabled(true);
        let mut ctx = PipelineCtx::new();
        ctx.is_listening = true;
        ctx.enrollment_mode = true;
        ctx.collecting_negatives = true;
        ctx.wake.negatives_speech_samples = 1234;
        ctx.auto_start_pending = true;
        {
            let mut state = voice_state().write().unwrap_poison();
            state.enrollment_embeddings.push(vec![0.5; 1024]);
            state.owner_negative_chunks.push(vec![0.5; 100]);
            state.enrolled_utterance_count = NUM_ENROLLMENT_SAMPLES;
            state.utterances_collected = true;
        }
        set_status(VoiceStatus::Enrolling {
            sample: NUM_ENROLLMENT_SAMPLES,
            total: NUM_ENROLLMENT_SAMPLES,
            duration_ms: 900,
            quality: None,
        });

        ctx.handle_microphone_lost();

        // The attempt is gone, accumulators included: a later tick cannot
        // finish it, and a fresh Enroll starts from scratch.
        assert!(!ctx.enrollment_mode);
        assert!(!ctx.collecting_negatives);
        assert_eq!(ctx.wake.negatives_speech_samples, 0);
        let state = voice_state().read().unwrap_poison();
        assert!(state.enrollment_embeddings.is_empty());
        assert!(state.owner_negative_chunks.is_empty());
        assert_eq!(state.enrolled_utterance_count, 0);
        assert!(!state.utterances_collected);
        drop(state);

        // Voice stopped with the honest status, and the user's switch is untouched.
        assert!(!ctx.is_listening);
        assert!(!ctx.auto_start_pending, "no automatic re-acquisition");
        assert!(matches!(get_status(), VoiceStatus::MicDisconnected));
        assert!(is_enabled(), "the wake-word switch is the user's setting");

        // A genuinely fresh attempt: nothing of the failed start is left that
        // could block it — the GUI pre-flight included, so the record button
        // really tries again instead of refusing.
        assert_eq!(recording_blocked_reason(), None);
        assert!(ctx.mic.is_none());
        assert!(ctx.recording.is_none());

        // With voice switched off there is nothing claiming to listen, so the
        // resting state after a loss is Disabled and the buffer is dropped.
        set_enabled(false);
        let mut ctx = PipelineCtx::new();
        ctx.is_listening = true;
        ctx.recording = Some(RecordingOrigin::Button);
        ctx.command_buffer = vec![0.5; 100];
        set_status(VoiceStatus::RecordingButton);

        ctx.handle_microphone_lost();

        assert!(ctx.recording.is_none());
        assert!(ctx.command_buffer.is_empty());
        assert!(matches!(get_status(), VoiceStatus::Disabled));
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[serial_test::serial(voice)]
    fn recording_aborted_when_enrollment_starts() {
        let _ = init_global();

        let mut ctx = PipelineCtx::new();
        ctx.is_listening = true; // wake-word mic running
        ctx.recording = Some(RecordingOrigin::Button);
        ctx.command_buffer.extend_from_slice(&[0.1, 0.2, 0.3]);
        ctx.resume_listening_after_recording = true;
        set_status(VoiceStatus::RecordingButton);

        // Enrollment owns the mic now: the recording is aborted and
        // its buffer discarded so it cannot resume as garbage later.
        ctx.handle_start_enrollment();
        assert!(ctx.recording.is_none());
        assert!(!ctx.resume_listening_after_recording);
        assert!(ctx.command_buffer.is_empty());
        assert!(ctx.enrollment_mode);
    }

    /// The composer's message is resolved by an event, never by silence: a start
    /// that does not take closes it, and a recording a teardown drops is reported
    /// lost with its reason.
    #[test]
    #[serial_test::serial(voice)]
    fn recording_endings_always_reach_the_composer() {
        let _ = init_global();
        init_recording_tx();
        let mut events = RECORDING.get().expect("warmed above").subscribe();

        // A press while a recording runs belongs to that recording: it reports
        // nothing, so the window the running recording owns is not closed.
        let mut ctx = PipelineCtx::new();
        ctx.recording = Some(RecordingOrigin::Button);
        assert!(ctx.handle_start_button_recording().is_none());
        assert!(
            events.try_recv().is_err(),
            "the running recording reports nothing"
        );
        assert_eq!(ctx.recording, Some(RecordingOrigin::Button));

        // A recording a teardown drops is reported, whichever way it started.
        ctx.handle_stop_listening();
        match events.try_recv() {
            Ok(RecordingEvent::Failed(notice)) => {
                assert!(notice.contains("turned off"), "{notice}");
            }
            other => panic!("a dropped recording must be reported, got {other:?}"),
        }
    }

    /// A wake-word recording that caught no speech resolves the composer's window
    /// without taking the field: nothing was said, so nothing joins it.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    #[serial_test::serial(voice)]
    async fn a_silent_wake_recording_leaves_the_field_alone() {
        let _ = init_global();
        init_recording_tx();
        let mut events = RECORDING.get().expect("warmed above").subscribe();

        let mut ctx = PipelineCtx::new();
        ctx.recording = Some(RecordingOrigin::WakeWord);
        set_status(VoiceStatus::RecordingWakeWord);

        ctx.handle_stop_recording(true).await;

        assert!(ctx.recording.is_none());
        assert!(matches!(events.try_recv(), Ok(RecordingEvent::NoSpeech)));
        assert!(matches!(get_status(), VoiceStatus::Listening));
    }

    /// The composer's cross, at both origins: the recording is dropped and the
    /// composer reports nothing on its own — it closed its window on the same
    /// press. A button recording takes the recording-only mic with it.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    #[serial_test::serial(voice)]
    async fn a_discarded_recording_clears_its_buffer_and_reports_nothing() {
        let _ = init_global();
        init_recording_tx();
        let mut events = RECORDING.get().expect("warmed above").subscribe();

        let mut ctx = PipelineCtx::new();
        ctx.recording = Some(RecordingOrigin::WakeWord);
        ctx.command_buffer.extend_from_slice(&[0.1, 0.2, 0.3]);
        set_status(VoiceStatus::RecordingWakeWord);

        ctx.handle_stop_recording(false).await;

        assert!(ctx.recording.is_none());
        assert!(ctx.command_buffer.is_empty());
        assert!(events.try_recv().is_err());
        assert!(matches!(get_status(), VoiceStatus::Listening));

        let mut ctx = PipelineCtx::new();
        ctx.recording = Some(RecordingOrigin::Button);
        ctx.command_buffer.extend_from_slice(&[0.1, 0.2, 0.3]);
        set_status(VoiceStatus::RecordingButton);

        ctx.handle_stop_recording(false).await;

        assert!(ctx.recording.is_none());
        assert!(ctx.command_buffer.is_empty());
        assert!(events.try_recv().is_err());
        assert!(matches!(get_status(), VoiceStatus::Disabled));
    }

    // The pure [`resolved_model_status`] helper is the single source of truth
    // for the startup block AND the periodic wake-up — the wake-up bug that
    // stranded the footer on "Loading…" was an asymmetry between the two.
    // No serial marker needed: the helper reads no global state.

    /// One rule over every state and both toggle positions: a failed load is
    /// terminal, a load in flight is waited on, an unstarted recogniser is not
    /// reported, and Ready resolves by the user's switch.
    #[test]
    fn resolved_model_status_follows_the_load_and_the_switch() {
        for enabled in [true, false] {
            assert!(
                matches!(
                    resolved_model_status(ModelState::Failed, enabled),
                    VoiceStatus::ModelError
                ),
                "a failed load is terminal, never Disabled or LoadingModels (enabled={enabled})",
            );
            assert!(
                matches!(
                    resolved_model_status(ModelState::Loading, enabled),
                    VoiceStatus::LoadingModels
                ),
                "a load in flight keeps the footer polling (enabled={enabled})",
            );
            assert!(
                matches!(
                    resolved_model_status(ModelState::Uninit, enabled),
                    VoiceStatus::Disabled
                ),
                "nothing asks for a recogniser before first use (enabled={enabled})",
            );
        }
        assert!(matches!(
            resolved_model_status(ModelState::Ready, true),
            VoiceStatus::Listening
        ));
        // The LoadingModels hang fix: the footer indicator must clear once
        // loading finishes with the switch off.
        assert!(matches!(
            resolved_model_status(ModelState::Ready, false),
            VoiceStatus::Disabled
        ));
    }

    // These test the 10-second voice-notice rate limit via the canonical
    // [`PipelineCtx::should_send_voice_notice`] method.  No serial marker
    // needed — these only read from [`PipelineCtx`] fields without touching
    // global voice state.

    #[test]
    fn rate_limit_voice_notice_table() {
        // (case, seconds since the last notice, expected send decision)
        let cases = [
            ("no_prior_notice", None, true),
            ("recent_notice", Some(0), false),
            ("old_notice_15s", Some(15), true),
            ("exact_threshold_10s", Some(10), true),
            ("just_below_9s", Some(9), false),
        ];

        for (name, elapsed_secs, expected) in cases {
            let mut ctx = PipelineCtx::new();
            ctx.last_voice_notice_time = elapsed_secs.map(|secs| {
                Instant::now()
                    .checked_sub(Duration::from_secs(secs))
                    .expect("elapsed seconds in the past should not underflow")
            });
            assert_eq!(
                ctx.should_send_voice_notice(),
                expected,
                "case {name}: 10s voice-notice rate limit",
            );
        }
    }

    // Pure unit tests for the z-score adaptive threshold tracker.  Uses
    // synthetic per-frame scores — no models or voice pipeline state.

    #[cfg(target_os = "macos")]
    #[test]
    fn adaptive_bootstrap_boundary_and_clamps() {
        let harbor = legacy_geometry().static_threshold;

        // bootstrap boundary: every feed during bootstrap returns None; the
        // frame that completes bootstrap returns Some.
        let mut state = AdaptiveThresholdState::new();
        for i in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            assert!(
                state.feed(0.5, ADAPTIVE_K_DEFAULT, harbor).is_none(),
                "bootstrap: feed frame {i} should return None",
            );
        }
        assert!(
            state.feed(0.5, ADAPTIVE_K_DEFAULT, harbor).is_some(),
            "bootstrap: feed should return Some after bootstrap",
        );

        // safe-harbor clamp: constant low scores (0.1) yield a low adaptive
        // value (~0.3) that must be overridden by the caller's static
        // detection threshold.
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.1, ADAPTIVE_K_DEFAULT, harbor);
        }
        let threshold = state
            .feed(0.1, ADAPTIVE_K_DEFAULT, harbor)
            .expect("should return Some after bootstrap");
        assert!(
            (threshold - harbor).abs() < 0.01,
            "safe-harbor clamp: threshold {threshold} should equal safe harbor {harbor}",
        );

        // warmed() initializes with near-silence scores (~0.033), and its first
        // feed exits bootstrap with the same safe-harbor clamp.
        let mut state = AdaptiveThresholdState::warmed(harbor);
        let threshold = state
            .feed(0.033, ADAPTIVE_K_DEFAULT, harbor)
            .expect("warmed() should exit bootstrap");
        assert!(
            (threshold - harbor).abs() < 0.01,
            "warmed(): threshold {threshold} should equal safe harbor {harbor}",
        );
        assert_eq!(state.len(), ADAPTIVE_BOOTSTRAP_FRAMES + 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn adaptive_ceiling_and_window_eviction() {
        let harbor = legacy_geometry().static_threshold;

        // ceiling clamp: high-variance scores cap at ADAPTIVE_CEILING. With
        // alternating 1.0/0.0 scores: mean=0.5, std≈0.5, k=2.5, so
        // adaptive = (0.5 + 2.5 × 0.5) × 3 = 5.25 → capped at 2.60.
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.99, ADAPTIVE_K_DEFAULT, harbor);
        }
        for i in ADAPTIVE_BOOTSTRAP_FRAMES..ADAPTIVE_WINDOW_N {
            let score = if i % 2 == 0 { 1.0 } else { 0.0 };
            state.feed(score, ADAPTIVE_K_DEFAULT, harbor);
        }
        let threshold = state
            .feed(1.0, ADAPTIVE_K_DEFAULT, harbor)
            .expect("should return Some after bootstrap");
        assert!(
            (threshold - ADAPTIVE_CEILING).abs() < 0.01,
            "ceiling clamp: threshold {threshold} should equal ceiling {ADAPTIVE_CEILING}",
        );

        // window eviction: after filling the window and cycling scores, the
        // sum/sum_sq statistics must produce the correct mean (k=0 so the
        // adaptive value is mean × ROLLING_WINDOW_N, clamped to [harbor, ceiling]).
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.5, ADAPTIVE_K_DEFAULT, harbor);
        }
        for _ in ADAPTIVE_BOOTSTRAP_FRAMES..ADAPTIVE_WINDOW_N {
            state.feed(0.5, ADAPTIVE_K_DEFAULT, harbor);
        }
        #[expect(clippy::cast_precision_loss)] // small test constants — exact in f32
        let window_n = ADAPTIVE_WINDOW_N as f32;
        #[expect(clippy::cast_precision_loss)] // small test constant — exact in f32
        let rolling_n = ROLLING_WINDOW_N as f32;
        let expected_mean = (window_n - 1.0) * 0.5 / window_n + 1.0 / window_n;
        let threshold = state.feed(1.0, 0.0, harbor).expect("should return Some");
        let expected_raw = expected_mean * rolling_n;
        let clamped = expected_raw.clamp(harbor, ADAPTIVE_CEILING);
        assert!(
            (threshold - clamped).abs() < 0.001,
            "window eviction: threshold {threshold} should match expected clamped value {clamped} (raw={expected_raw})",
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn adaptive_reset_and_peek() {
        let harbor = legacy_geometry().static_threshold;

        // reset() empties the window, so the next feed re-enters bootstrap and
        // peek() answers nothing.
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.5, ADAPTIVE_K_DEFAULT, harbor);
        }
        assert!(state.feed(0.5, ADAPTIVE_K_DEFAULT, harbor).is_some());
        assert_eq!(state.len(), ADAPTIVE_BOOTSTRAP_FRAMES + 1);

        state.reset();

        assert_eq!(state.len(), 0, "reset: window should be empty");
        assert!(
            state.feed(0.5, ADAPTIVE_K_DEFAULT, harbor).is_none(),
            "reset: first feed should return None (re-enters bootstrap)",
        );
        assert!(
            state.peek(ADAPTIVE_K_DEFAULT, harbor).is_none(),
            "reset: peek should return None (empty window)",
        );

        // peek() must not change scores, sum, sum_sq, or bootstrap_count.
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.5, ADAPTIVE_K_DEFAULT, harbor);
        }
        state.feed(0.5, ADAPTIVE_K_DEFAULT, harbor); // non-bootstrap state
        let before_scores = state.scores.clone();
        let before_sum = state.sum;
        let before_sum_sq = state.sum_sq;
        let before_bootstrap = state.bootstrap_count;
        for _ in 0..3 {
            let _ = state.peek(ADAPTIVE_K_DEFAULT, harbor);
        }
        assert_eq!(state.scores, before_scores, "peek must not modify scores");
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

        // peek() agrees with feed() on the same state.
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.5, ADAPTIVE_K_DEFAULT, harbor);
        }
        state.feed(0.7, ADAPTIVE_K_DEFAULT, harbor);
        let feed_threshold = state.feed(0.3, ADAPTIVE_K_DEFAULT, harbor);
        let peek_threshold = state.peek(ADAPTIVE_K_DEFAULT, harbor);
        assert_eq!(
            feed_threshold, peek_threshold,
            "peek must agree with feed on the same state",
        );

        // peek() clamps to safe harbor on constant low scores (0.1) and its
        // threshold lies within [harbor, ceiling].
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.1, ADAPTIVE_K_DEFAULT, harbor);
        }
        let threshold = state
            .peek(ADAPTIVE_K_DEFAULT, harbor)
            .expect("peek should return Some after bootstrap");
        assert!(
            (threshold - harbor).abs() < 0.01,
            "safe-harbor clamp: peek threshold {threshold} should equal safe harbor {harbor}",
        );
        let mut state = AdaptiveThresholdState::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.5, ADAPTIVE_K_DEFAULT, harbor);
        }
        let threshold = state
            .peek(ADAPTIVE_K_DEFAULT, harbor)
            .expect("Some after bootstrap");
        assert!(
            (harbor..=ADAPTIVE_CEILING).contains(&threshold),
            "valid range: peek threshold {threshold} must be within [{harbor}, {ADAPTIVE_CEILING}]",
        );
    }

    // Pure rolling-window scoring — no models, no pipeline state.

    /// Detection geometry for an enrollment with no calibration — the legacy
    /// operating point (static threshold 3 × 0.55 = 1.65, reset
    /// 0.64 × 0.55 ≈ 0.352).
    #[cfg(target_os = "macos")]
    fn legacy_geometry() -> DetectionGeometry {
        DetectionGeometry::for_enrollment(None)
    }

    /// The rolling window's three rules: a below-reset score clears it, frames
    /// accumulate to the static threshold, and an adaptive override raises the bar.
    #[cfg(target_os = "macos")]
    #[test]
    fn process_wake_word_score_clears_fires_and_honours_the_override() {
        let geometry = legacy_geometry();

        let mut window = vec![0.9, 0.8, 0.7];
        let (detected, rolling) =
            process_wake_word_score(geometry.reset - 0.01, &mut window, geometry, None);
        assert!(!detected);
        #[expect(clippy::float_cmp)] // exact reset/const comparison — bit-deterministic
        {
            assert_eq!(rolling, 0.0);
        }
        assert!(
            window.is_empty(),
            "below-reset score must clear the rolling window"
        );

        // static threshold = 3 × 0.55 = 1.65 → need 3 frames summing to ≥1.65.
        let mut window = Vec::new();
        for score in [0.7, 0.7] {
            let (detected, _) = process_wake_word_score(score, &mut window, geometry, None);
            assert!(!detected, "2 frames of 0.7 must not fire (sum 1.4 < 1.65)");
        }
        let (detected, rolling) = process_wake_word_score(0.7, &mut window, geometry, None);
        assert!(detected, "3 frames of 0.7 must fire (sum 2.1 ≥ 1.65)");
        assert!(
            (rolling - 2.1).abs() < 1e-6,
            "rolling sum should be 2.1, got {rolling}"
        );

        // A high adaptive override (3.0) must suppress detection even when
        // three 0.7 frames would otherwise fire (sum 2.1 < 3.0).
        let mut window = Vec::new();
        for _ in 0..3 {
            let (detected, _) = process_wake_word_score(0.7, &mut window, geometry, Some(3.0));
            assert!(!detected, "adaptive override must raise the detection bar");
        }
    }

    // The new encoder pipeline scores each window embedding as the cosine
    // soft score against the enrolled prototype.  These tests use synthetic
    // L2-normalized embeddings and a synthetic enrollment.

    /// Basis embedding: dim `d` = 1.0, everything else 0.0 (unit norm).
    #[cfg(target_os = "macos")]
    fn basis_embedding(d: usize) -> Vec<f32> {
        let mut v = vec![0.0; WAKE_WORD_EMBEDDING_DIM];
        if d < WAKE_WORD_EMBEDDING_DIM {
            v[d] = 1.0;
        }
        v
    }

    /// Build a synthetic enrollment whose prototype points along basis dim 0.
    #[cfg(target_os = "macos")]
    fn synthetic_enrollment(n_utterances: usize) -> WakeWordEnrollment {
        let emb = basis_embedding(0);
        WakeWordEnrollment::build(
            &vec![emb; n_utterances],
            crate::audio::wake_word::Calibration::default(),
            &[],
            String::new(),
            String::new(),
        )
        .expect("synthetic enrollment")
    }

    /// The cosine-prototype scorer's decisions: without an enrollment nothing
    /// scores, a prototype match fires after the rolling window fills, and a
    /// below-reset frame clears the window and feeds only the background statistics.
    #[cfg(target_os = "macos")]
    #[test]
    fn score_single_embedding_fires_on_a_match_and_resets_without_one() {
        // Without an enrollment no detection is possible: total_score = 0.0
        // and the window is reset (0.0 < the default reset threshold).
        let mut window = vec![0.9, 0.8];
        let emb = crate::audio::wake_word::l2_normalize(&vec![1.0; WAKE_WORD_EMBEDDING_DIM]);
        let (detected, rolling, total, _) =
            score_single_embedding(&emb, None, &mut window, None, ADAPTIVE_K_DEFAULT);
        assert!(!detected);
        #[expect(clippy::float_cmp)] // exact reset/const comparison — bit-deterministic
        {
            assert_eq!(total, 0.0);
        }
        #[expect(clippy::float_cmp)] // exact reset/const comparison — bit-deterministic
        {
            assert_eq!(rolling, 0.0);
        }
        assert!(
            window.is_empty(),
            "zero score must reset the rolling window"
        );

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
                total >= legacy_geometry().reset,
                "a prototype match must score above the reset threshold"
            );
            detected = d;
        }
        assert!(
            detected,
            "3 consecutive high-cosine frames must fire detection"
        );

        // An orthogonal embedding (basis 1 vs prototype basis 0) → cosine 0 →
        // soft score 0.0 (clamped) → below reset → clears the window and feeds
        // the adaptive statistics.
        let mut window = vec![0.9, 0.8];
        let mut adaptive = AdaptiveThresholdState::new();
        let (detected, rolling, total, _) = score_single_embedding(
            &basis_embedding(1),
            Some(&enrollment),
            &mut window,
            Some(&mut adaptive),
            ADAPTIVE_K_DEFAULT,
        );
        assert!(!detected);
        assert!(total < legacy_geometry().reset);
        #[expect(clippy::float_cmp)] // exact reset/const comparison — bit-deterministic
        {
            assert_eq!(rolling, 0.0);
        }
        assert!(window.is_empty(), "low score must reset the rolling window");
        assert_eq!(
            adaptive.len(),
            1,
            "background score must feed the adaptive window"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn static_geometry_anchors_hot_floors() {
        // A hot negative pool (mean 0.80, std 0.06) saturates the derived
        // floor at MAX_SOFT_FLOOR (0.80) and anchors the fire factor at
        // (0.89−0.80)/0.20 = 0.45 → static rolling threshold 3 × 0.45 = 1.35.
        let hot_cal = crate::audio::wake_word::Calibration {
            neg_mean: 0.80,
            neg_std: 0.06,
            neg_p99: 0.97,
            n_negatives: 200,
        };
        let enrollment = WakeWordEnrollment::build(
            &vec![basis_embedding(0); MIN_ENROLLMENT_UTTERANCES],
            hot_cal,
            &[],
            String::new(),
            String::new(),
        )
        .expect("synthetic hot enrollment");

        let geometry = DetectionGeometry::for_enrollment(Some(&enrollment));
        assert!(
            (geometry.static_threshold - 1.35).abs() < 1e-3,
            "static threshold should be ≈1.35, got {}",
            geometry.static_threshold,
        );
        assert!(
            (geometry.reset - 0.288).abs() < 1e-3,
            "reset should be ≈0.288, got {}",
            geometry.reset,
        );

        // A negative at the pooled mean (cosine 0.80) maps to soft ≈0 through
        // the enrollment's soft_score path (no anti-prototypes → pure floor mapping).
        let mut neg_at_mean = basis_embedding(0);
        neg_at_mean[1] = 0.75; // cosine ≈ 1/√(1+0.5625) = 0.80
        let neg_at_mean = crate::audio::wake_word::l2_normalize(&neg_at_mean);
        assert!(
            enrollment.soft_score(&neg_at_mean) < 0.01,
            "a negative at the pooled mean must map to soft ≈0",
        );

        // A strong frame (cosine ≈0.97 → soft ≈0.85) must NOT fire on a
        // single frame (0.85 < 1.35 — the multi-frame temporal guard survives
        // the anchor) but 3 such frames DO fire.
        let mut strong = basis_embedding(0);
        strong[1] = 0.25; // cosine ≈ 1/√(1+0.0625) ≈ 0.970
        let strong = crate::audio::wake_word::l2_normalize(&strong);
        let mut window = Vec::new();
        let (single, _, total_score, _) = score_single_embedding(
            &strong,
            Some(&enrollment),
            &mut window,
            None,
            ADAPTIVE_K_DEFAULT,
        );
        assert!(
            total_score >= geometry.reset,
            "a strong hot frame must accumulate (above reset)",
        );
        assert!(
            !single,
            "a single strong frame must not fire (rolling sum < static threshold)",
        );
        window.clear();
        let mut detected = false;
        for _ in 0..ROLLING_WINDOW_N {
            let (d, _, _, _) = score_single_embedding(
                &strong,
                Some(&enrollment),
                &mut window,
                None,
                ADAPTIVE_K_DEFAULT,
            );
            detected = d;
        }
        assert!(detected, "3 consecutive hot frames must fire detection");

        // The adaptive harbor is the enrollment's static threshold: after
        // bootstrapping on pure background, the effective threshold must
        // never drop below the static firing bar.
        window.clear();
        let mut adaptive = AdaptiveThresholdState::new();
        let mut effective_min = f32::MAX;
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES + 2 {
            let (_, _, _, effective) = score_single_embedding(
                &neg_at_mean,
                Some(&enrollment),
                &mut window,
                Some(&mut adaptive),
                ADAPTIVE_K_DEFAULT,
            );
            effective_min = effective_min.min(effective);
        }
        assert!(
            effective_min >= geometry.static_threshold - 1e-4,
            "adaptive threshold must be harbored at the static threshold \
             ({effective_min} < {})",
            geometry.static_threshold,
        );
    }

    // The centroid gate is a pure function so it is testable without the ASR
    // model.  Uses basis embeddings: utterances on basis 0 form a consistent
    // cluster; utterances on other bases are outliers.

    #[cfg(target_os = "macos")]
    #[test]
    fn consistency_gate_table() {
        // Centroid gate is pure: basis-0 embeddings form a consistent cluster,
        // basis-1 embeddings are outliers.  `bad_dim_at` corrupts one slot.
        struct Case {
            name: &'static str,
            basis0: usize,
            basis1: usize,
            bad_dim_at: Option<usize>,
            /// Expected substring of the user-facing error; `None` means the
            /// gate must succeed.
            expected_error_fragment: Option<&'static str>,
        }

        let cases = [
            Case {
                name: "fails_when_too_few_utterances",
                basis0: 4,
                basis1: 0,
                bad_dim_at: None,
                expected_error_fragment: Some("4 enrollment utterances"),
            },
            Case {
                // 6 on basis 0 + 4 on basis 1: 6/10 pass → need ceil(10 × 0.7) = 7 → fail.
                name: "fails_when_too_few_pass_threshold",
                basis0: 6,
                basis1: 4,
                bad_dim_at: None,
                expected_error_fragment: Some("6/10"),
            },
            Case {
                // 8 on basis 0 + 2 on basis 1: 8/10 pass → need ceil(10 × 0.7) = 7 → pass.
                name: "succeeds_with_high_quality_utterances",
                basis0: 8,
                basis1: 2,
                bad_dim_at: None,
                expected_error_fragment: None,
            },
            Case {
                name: "rejects_wrong_dimension",
                basis0: MIN_ENROLLMENT_UTTERANCES,
                basis1: 0,
                bad_dim_at: Some(0),
                expected_error_fragment: Some("expected 1024"),
            },
        ];

        for case in &cases {
            let mut embs: Vec<Vec<f32>> = Vec::with_capacity(case.basis0 + case.basis1);
            embs.extend(std::iter::repeat_n(basis_embedding(0), case.basis0));
            embs.extend(std::iter::repeat_n(basis_embedding(1), case.basis1));
            if let Some(idx) = case.bad_dim_at {
                embs[idx] = vec![0.5; 64];
            }
            match (
                enrollment_consistency_check(&embs),
                case.expected_error_fragment,
            ) {
                (Ok(_), None) => {}
                (Err(err), Some(fragment)) => assert!(
                    err.contains(fragment),
                    "case: {name} — error should mention {fragment:?}: {err}",
                    name = case.name,
                ),
                (Ok(_), Some(fragment)) => panic!(
                    "case: {name} — expected error containing {fragment:?}, got Ok",
                    name = case.name,
                ),
                (Err(err), None) => panic!(
                    "case: {name} — expected success, got error: {err}",
                    name = case.name,
                ),
            }
        }
    }

    #[cfg(target_os = "macos")]
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

    // These test the three ResetLevel variants against a PipelineCtx with
    // non-default field values.  The group is serialized on the voice key
    // because the tests seed/assert the process-global enrollment accumulators
    // and Full resets the shared VAD detector.

    /// Helper: build a PipelineCtx with non-default values in all mutable
    /// fields that reset_pipeline_state may touch.
    #[cfg(target_os = "macos")]
    fn ctx_with_populated_buffers() -> PipelineCtx {
        let mut ctx = PipelineCtx::new();
        ctx.wake.audio_buffer = vec![0.5; 100];
        ctx.command_buffer = vec![0.5; 100];
        ctx.silence_sample_count = 1000;
        ctx.wake.post_fire_discard_samples = 1000;
        ctx.wake.recording_had_speech = true;
        ctx.wake.score_window = vec![0.5; 5];
        ctx.wake.last_score_sample_count = 512;
        ctx.wake.negative_audio_buf = vec![0.5; 50];
        ctx.wake.frame_vad = vec![true; 3];
        ctx.wake.frame_raw_audio = vec![0.5; 200];
        ctx.wake.emitted_utterances = 2;
        ctx.wake.utterance_had_speech = true;
        ctx.wake.utterance_silence_samples = 500;
        ctx.wake.enrollment_no_speech_frame_count = 3;
        ctx.wake.vad_positives_in_a_row = 5;
        ctx.wake.enrollment_pending.push_back(vec![0.5; 50]);
        ctx.wake.noise_rms_estimate = Some(0.1);
        ctx.collecting_negatives = true;
        ctx.wake.phase3_audio_buf = vec![0.5; 100];
        ctx.wake.phase3_silence_samples = 500;
        ctx.wake.negatives_speech_samples = 1000;
        ctx.wake.phase3_processed = 1234;
        ctx.wake.phase3_start_time =
            Some(Instant::now().checked_sub(Duration::from_secs(10)).unwrap());
        ctx.wake.vad_threshold = 0.75;
        ctx.wake.last_wake_word_detection =
            Some(Instant::now().checked_sub(Duration::from_secs(5)).unwrap());
        ctx.auto_start_pending = true;
        ctx.recording = Some(RecordingOrigin::WakeWord);
        ctx.wake.segment_score_steps = 7;
        ctx
    }

    /// Assert that all audio accumulators and enrollment-state fields are
    /// cleared — the common post-reset invariant shared by all level variants.
    #[cfg(target_os = "macos")]
    fn assert_buffers_cleared(ctx: &PipelineCtx) {
        assert!(ctx.wake.audio_buffer.is_empty());
        assert!(ctx.command_buffer.is_empty());
        assert_eq!(ctx.silence_sample_count, 0);
        assert_eq!(ctx.wake.post_fire_discard_samples, 0);
        assert!(!ctx.wake.recording_had_speech);
        assert!(ctx.wake.score_window.is_empty());
        assert_eq!(ctx.wake.last_score_sample_count, 0);
        assert!(ctx.wake.negative_audio_buf.is_empty());

        assert!(ctx.wake.frame_vad.is_empty());
        assert!(ctx.wake.frame_raw_audio.is_empty());
        assert_eq!(ctx.wake.emitted_utterances, 0);

        assert!(!ctx.wake.utterance_had_speech);
        assert_eq!(ctx.wake.utterance_silence_samples, 0);
        assert_eq!(ctx.wake.enrollment_no_speech_frame_count, 0);
        assert_eq!(ctx.wake.vad_positives_in_a_row, 0);
        assert!(ctx.wake.enrollment_pending.is_empty());
        assert!(ctx.wake.noise_rms_estimate.is_none());

        assert_eq!(
            ctx.wake.segment_silence_hops, 0,
            "segment_silence_hops must be cleared by all reset levels"
        );
        assert_eq!(
            ctx.wake.segment_score_steps, 0,
            "segment_score_steps must be cleared by all reset levels"
        );

        assert!(
            !ctx.collecting_negatives,
            "collecting_negatives must be false after reset"
        );
        assert!(
            ctx.wake.phase3_audio_buf.is_empty(),
            "phase3_audio_buf must be cleared by all reset levels"
        );
        assert_eq!(
            ctx.wake.phase3_silence_samples, 0,
            "phase3_silence_samples must be cleared by all reset levels"
        );
        assert_eq!(
            ctx.wake.negatives_speech_samples, 0,
            "negatives_speech_samples must be cleared by all reset levels"
        );
        assert_eq!(
            ctx.wake.phase3_processed, 0,
            "phase3_processed must be cleared by all reset levels"
        );
        assert!(
            ctx.wake.phase3_start_time.is_none(),
            "phase3_start_time must be None after reset"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[serial_test::serial(voice)]
    fn reset_full_clears_all_buffers_and_state() {
        let _ = init_global();
        let mut ctx = ctx_with_populated_buffers();
        // Full leaves the handler-managed flags alone: they are owned by the
        // handler functions, not by the reset.
        ctx.is_listening = true;
        ctx.enrollment_mode = true;

        // Establish this test's own baseline: the accumulators are
        // process-global and the Soft-reset test leaves its own entries behind
        // (both levels preserve them by design) — clear them through the
        // module's reset before seeding the entries asserted below.
        {
            let mut state = voice_state().write().unwrap_poison();
            state.reset_enrollment();
            state.enrollment_embeddings.push(vec![0.5; 1024]);
            state.negative_audio_chunks.push(vec![0.5; 100]);
        }

        ctx.reset_pipeline_state(ResetLevel::Full);

        assert_buffers_cleared(&ctx);

        // Full-specific: state flags reset.
        #[expect(clippy::float_cmp)] // exact reset/const comparison — bit-deterministic
        {
            assert_eq!(ctx.wake.vad_threshold, VAD_THRESHOLD);
        }
        assert!(ctx.wake.last_wake_word_detection.is_none());
        assert!(!ctx.auto_start_pending);
        assert!(ctx.recording.is_none());

        // Handler-managed flags survive.
        assert!(ctx.is_listening);
        assert!(ctx.enrollment_mode);

        // Global enrollment accumulators PRESERVED by Full — they survive
        // mic stop/start cycles so mid-enrollment progress is not lost on toggle-off/on.
        let state = voice_state().read().unwrap_poison();
        assert_eq!(state.enrollment_embeddings.len(), 1);
        assert_eq!(state.negative_audio_chunks.len(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[serial_test::serial(voice)]
    fn reset_soft_and_cancel_differ_in_scope() {
        let _ = init_global();
        let mut ctx = ctx_with_populated_buffers();

        // Pre-populate global enrollment state so we can verify Soft keeps it
        // and Cancel clears it.
        let saved_embeddings = vec![vec![0.5; 1024]];
        let saved_chunks = vec![vec![0.5; 100]];
        {
            let mut state = voice_state().write().unwrap_poison();
            state.enrollment_embeddings = saved_embeddings.clone();
            state.negative_audio_chunks = saved_chunks.clone();
        }

        let saved_threshold = ctx.wake.vad_threshold; // 0.75
        let saved_cooldown = ctx.wake.last_wake_word_detection;
        let saved_auto_start = ctx.auto_start_pending;
        let saved_recording = ctx.recording;

        ctx.reset_pipeline_state(ResetLevel::Soft);

        assert_buffers_cleared(&ctx);

        // Soft preserves these.
        #[expect(clippy::float_cmp)] // exact reset/const comparison — bit-deterministic
        {
            assert_eq!(ctx.wake.vad_threshold, saved_threshold);
        }
        assert_eq!(ctx.wake.last_wake_word_detection, saved_cooldown);
        assert_eq!(ctx.auto_start_pending, saved_auto_start);
        assert_eq!(ctx.recording, saved_recording);

        // Global enrollment accumulators preserved.
        let state = voice_state().read().unwrap_poison();
        assert_eq!(state.enrollment_embeddings, saved_embeddings);
        assert_eq!(state.negative_audio_chunks, saved_chunks);
        drop(state);

        // Cancel clears vad_threshold, the detection cooldown and the global
        // accumulators, while the handler-managed flags still survive.
        let mut ctx = ctx_with_populated_buffers();
        ctx.reset_pipeline_state(ResetLevel::Cancel);

        assert_buffers_cleared(&ctx);

        #[expect(clippy::float_cmp)] // exact reset/const comparison — bit-deterministic
        {
            assert_eq!(ctx.wake.vad_threshold, VAD_THRESHOLD);
        }
        assert!(ctx.wake.last_wake_word_detection.is_none());
        assert_eq!(ctx.auto_start_pending, saved_auto_start);
        assert_eq!(ctx.recording, saved_recording);

        let state = voice_state().read().unwrap_poison();
        assert!(state.enrollment_embeddings.is_empty());
        assert!(state.negative_audio_chunks.is_empty());
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[serial_test::serial(voice)]
    fn reset_levels_preserve_session_ux_state() {
        let _ = init_global();
        // Session-level UX state (refractory_until, rate limiter timestamps)
        // must survive all reset levels — no level touches them.
        for level in [ResetLevel::Soft, ResetLevel::Full, ResetLevel::Cancel] {
            let mut ctx = PipelineCtx::new();
            ctx.refractory_until = Some(Instant::now());
            ctx.last_voice_notice_time = Some(Instant::now());

            ctx.reset_pipeline_state(level);

            assert!(
                ctx.refractory_until.is_some(),
                "refractory_until lost at {level:?}"
            );
            assert!(
                ctx.last_voice_notice_time.is_some(),
                "last_voice_notice_time lost at {level:?}"
            );
        }
    }

    // Regression tests for the Phase-3 speech counter grew
    // quadratically (un-drained buffers re-counted every prior frame on each
    // mic chunk), tripping the (then-60 s, now 15 s) target after ~2–4 real
    // seconds.  The counting path is driven through the pure
    // [`process_phase3_frames`] with injected VAD decisions — the stateful
    // earshot neural detector is not deterministic on synthetic audio, so
    // decisions are supplied directly.

    #[cfg(target_os = "macos")]
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

    #[cfg(target_os = "macos")]
    #[test]
    fn phase3_negative_collection_audio_counts_1_to_1_through_pipeline() {
        const CHUNK_SAMPLES: usize = SAMPLE_RATE as usize / 2; // 0.5 s

        // End-to-end regression through the production entry point.  Force
        // the stateful earshot detector to report speech on every hop by
        // setting the threshold below its documented [0,1] score range — the
        // ticket-sanctioned threshold trick.  The same audio fed in 0.5 s
        // chunks must count 1:1 (each hop fed exactly once) and must be
        // chunking-invariant through the full ctx write-back path.
        let mut ctx = PipelineCtx::new();
        ctx.wake.enrollment_vad = Some(earshot::Detector::default());
        ctx.wake.vad_threshold = -1.0;

        let chunk = vec![0.1f32; CHUNK_SAMPLES];
        for _ in 0..10 {
            handle_negative_collection_audio(&chunk, &mut ctx);
            // 1:1 per-call monotone check: in all-speech mode every processed
            // hop is counted, so the counter must equal the watermark exactly.
            // The quadratic bug made the counter grow by the WHOLE buffer on
            // each chunk (counter ≫ processed); an under-count would leave it
            // behind (counter < processed).
            assert_eq!(
                ctx.wake.negatives_speech_samples, ctx.wake.phase3_processed,
                "counter must track processed hops 1:1 across calls \
                 (no re-count, no under-count)",
            );
        }
        let total = 10 * CHUNK_SAMPLES;
        let expected = (total.saturating_sub(FRAME_LENGTH)) / HOP_LENGTH * HOP_LENGTH + HOP_LENGTH;
        assert_eq!(ctx.wake.negatives_speech_samples, expected);
        assert!(ctx.wake.negatives_speech_samples <= total);

        // Chunking invariance through the production path: one 5 s call must
        // land on the same counter as ten 0.5 s calls.
        let mut ctx2 = PipelineCtx::new();
        ctx2.wake.enrollment_vad = Some(earshot::Detector::default());
        ctx2.wake.vad_threshold = -1.0;
        let big = vec![0.1f32; total];
        handle_negative_collection_audio(&big, &mut ctx2);
        assert_eq!(
            ctx2.wake.negatives_speech_samples,
            ctx.wake.negatives_speech_samples
        );

        // Continuous speech never hits a chunk boundary — nothing pushed, and
        // the unfinalized segment survives intact (no drain).
        assert_eq!(ctx.wake.phase3_audio_buf.len(), total);
        #[expect(clippy::float_cmp)] // exact reset/const comparison — bit-deterministic
        {
            assert!(ctx.wake.phase3_audio_buf.iter().all(|&s| s == 0.1));
        }
    }

    // Tests the extracted segment boundary check logic with the encoder
    // pipeline signature `(&mut self, hop_count: usize)`.

    /// At the hop threshold the boundary fires and resets the per-segment
    /// state; below it the state persists, and a recording in progress skips
    /// the reset entirely.
    #[cfg(target_os = "macos")]
    #[test]
    fn handle_segment_boundary_table() {
        struct Case {
            name: &'static str,
            /// Silence hops accumulated before the call.
            initial_hops: usize,
            hop_count: usize,
            recording: bool,
            boundary_fires: bool,
            /// Silence hops the call must leave behind.
            hops_after: usize,
        }

        let cases = [
            Case {
                name: "at_threshold_fires",
                initial_hops: 10,
                hop_count: SEGMENT_TIMEOUT_HOPS,
                recording: false,
                boundary_fires: true,
                hops_after: 0,
            },
            Case {
                name: "below_threshold_persists",
                initial_hops: 10,
                hop_count: SEGMENT_TIMEOUT_HOPS - 1,
                recording: false,
                boundary_fires: false,
                hops_after: SEGMENT_TIMEOUT_HOPS - 1,
            },
            Case {
                // A just-fired detection (is_recording) must NOT be reset by
                // the boundary check — the detection→recording handoff
                // completes the transition and owns the state.
                name: "recording_skips_the_reset",
                initial_hops: 0,
                hop_count: SEGMENT_TIMEOUT_HOPS,
                recording: true,
                boundary_fires: false,
                hops_after: 0,
            },
        ];

        for case in cases {
            let mut ctx = PipelineCtx::new();
            ctx.recording = case.recording.then_some(RecordingOrigin::WakeWord);
            ctx.wake.score_window = vec![0.5; 5];
            ctx.wake.segment_silence_hops = case.initial_hops;
            ctx.wake.segment_score_steps = 7;

            // Complete adaptive threshold bootstrap so the reset-to-bootstrapping
            // assertion is meaningful (not trivially true from PipelineCtx::new()).
            {
                let mut at = AdaptiveThresholdState::new();
                for _ in 0..=ADAPTIVE_BOOTSTRAP_FRAMES {
                    at.feed(0.5, ADAPTIVE_K_DEFAULT, legacy_geometry().static_threshold);
                }
                assert!(
                    !at.is_bootstrapping(),
                    "adaptive threshold must exit bootstrap after {} feeds",
                    ADAPTIVE_BOOTSTRAP_FRAMES + 1
                );
                ctx.wake.adaptive_threshold = at;
            }

            ctx.handle_segment_boundary(case.hop_count);

            assert_eq!(
                ctx.wake.segment_silence_hops, case.hops_after,
                "case {}: silence hop counter",
                case.name,
            );
            if case.boundary_fires {
                assert!(
                    ctx.wake.score_window.is_empty(),
                    "case {}: score window",
                    case.name
                );
                assert_eq!(
                    ctx.wake.segment_score_steps, 0,
                    "case {}: segment_score_steps must be reset at the boundary",
                    case.name,
                );
                assert!(
                    ctx.wake.adaptive_threshold.is_bootstrapping(),
                    "case {}: adaptive_threshold must be reset (re-enter bootstrap)",
                    case.name,
                );
            } else {
                assert!(
                    !ctx.wake.score_window.is_empty(),
                    "case {}: score_window must survive inside a segment",
                    case.name,
                );
                assert!(
                    !ctx.wake.adaptive_threshold.is_bootstrapping(),
                    "case {}: adaptive_threshold must survive inside a segment",
                    case.name,
                );
                if !case.recording {
                    assert_eq!(
                        ctx.wake.segment_score_steps, 7,
                        "case {}: segment_score_steps must persist inside a continuing segment",
                        case.name,
                    );
                }
            }
        }
    }
}
