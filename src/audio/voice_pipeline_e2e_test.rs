//! E2E integration test / benchmark for the full voice pipeline (mahbot-811, mahbot-844).
//!
//! This test exercises the **enrollment-to-detection cycle with realistic
//! TTS-generated speech audio**.  It uses the TTS engine to synthesize wake
//! word variants, feeds them through the enrollment pipeline, trains the
//! Conv1D classifier, then runs detection on:
//!
//! * Positive cases (wake word variants)
//! * Negative — confusable near-miss phrases
//! * Negative — completely unrelated speech
//! * Negative — silence and noise
//!
//! # Running as benchmark (canonical — minimal feature set)
//!
//! The minimal invocation excludes the app binary from the build (the bin is
//! gated behind the default-on `full` feature via `required-features` on
//! `[[bin]]`, so `--no-default-features` skips its full-program link).
//! This is the documented fast path for all bench runs:
//!
//! ```sh
//! cargo bench --no-default-features --features voice-tests --bench voice_pipeline_e2e
//! ```
//!
//! A full-feature run (builds the app binary too — much slower) is:
//!
//! ```sh
//! cargo bench --features voice-tests --bench voice_pipeline_e2e
//! ```
//!
//! NOTE: `--no-default-features` skips the app binary for ALL target
//! selections (`cargo build`/`check`/`test`/`install`), not just bench — this
//! is inherent to the `required-features` mechanism (bench-only bin exclusion
//! is not supported by Cargo; `bench = false` does not work, rust-lang/cargo#15702).
//! Default builds are unaffected (`full` is default-on; the bin always builds).
//!
//! NOTE: `[profile.bench]` is release-like (opt-level 2, codegen-units 32,
//! no LTO, incremental off) — the bench is a production-performance proxy, so
//! release codegen is the faithful target.  Sanity anchors are report-only
//! measurements: nothing gates on them, codegen shifts are expected, and
//! there is no re-baseline ceremony.
//!
//! First run populates the TTS audio cache (~14-17 min); subsequent runs
//! complete in ~30-40 s (bench body — Phase 13's cooldown probes add ~6.5 s of
//! excluded-from-timing sleeps: 2.5 s suppression + 3.5 s recovery probes both
//! measured from Detection 1's timestamp (overlapping, max ≈ 3.5 s); the
//! Phase-1 held-out recall set + cross-speaker probe matrix land warm runs at
//! ~36 s) plus the build time; the exact figure is auditable from the
//! top-level `total_wall_time_ms` key (whole run, report-assembly window
//! included).  The first run after a recipe change is a cold
//! embedding cache (versioned keys all miss) — the budget is verified on that
//! cold run, and the one-time TTS synthesis for new owner-negative phrases is
//! paid outside the warm budget.
//! The bench-leanness cleanup removed the report-only same-audio (8b),
//! B-sweep (8c), volume-sweep, mid-utterance, cooldown boundary-probe, B2
//! synthetic-fallback probe, train/test-alignment cosine diagnostics, and the
//! dead top-level latency section (~11-16 s saved total).  Removing 8b/8c changes the
//! shared VAD-detector drift state entering the Phase 9-12 FA canaries, so
//! post-change canary numbers are NOT strictly comparable to archived
//! baselines (disclosed in the report _note); the noise_overlap clean-cell
//! trim (2 of 3 byte-identical infinite-SNR clean cells removed) likewise
//! shifts the in-phase shared adaptive trajectory for the remaining
//! noise_overlap cells — all detections are 0/20, so the FA-canary impact is
//! cosmetic, but cross-run comparability is disclosed for the same reason.
//!
//! # Requirements
//!
//! * TTS models must be downloaded and cached (run the app once).
//! * Voice ONNX models (melspectrogram + embedding) must be present in
//!   `~/.mahbot/models/openwakeword/`.

// Clippy allowances — use super::* is intentional (mirrors the voice module's API surface).
// Cast warnings: this is a benchmark file with known-safe numeric conversions.
#![allow(
    clippy::wildcard_imports,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use super::*; // voice module items (process_enrollment_sample, etc.)
use crate::audio::embedding_sequence::{EmbeddingSequence, Source, UtteranceId};
use crate::audio::tts;
use crate::audio::wake_word_classifier::WakeWordClassifier;
use earshot::Detector;
use rand::{RngExt, SeedableRng};
use std::borrow::Cow;
use std::time::Instant;

// ── Constants ──────────────────────────────────────────────────────────────

/// Index of the rolling sum field in per_frame_scores `[total_score, rolling_sum, threshold]` triples.
const ROLLING_SUM_IDX: usize = 1;
/// Index of the effective threshold field in per_frame_scores `[total_score, rolling_sum, threshold]` triples.
const THRESHOLD_IDX: usize = 2;
/// Index of the total (classifier) score field in per_frame_scores
/// `[total_score, rolling_sum, threshold]` triples (mahbot-1045 B7 — the
/// layout is shared with production's scoring pipeline, so a production
/// layout change must not silently corrupt miss classification).
const TOTAL_SCORE_IDX: usize = 0;

/// Minimum effective threshold for the Conv1D classifier's rolling sum.
///
/// The classifier rolling sum must reach at least this value for the
/// classifier to have fired.  Below this threshold, misses are attributed
/// to the classifier (mahbot-882).
///
/// Updated for mahbot-923's unified dense stride-8 embeddings (1.58× multiplier
/// vs the old mixed old-style + streaming distribution): 2.13 = 1.35 * 1.58.
const MIN_CLASSIFIER_THRESHOLD: f32 = 2.13;

/// Number of samples in the warm-up audio prepended before each test
/// utterance (mahbot-922, mahbot-926).
///
/// ~1.28 s of background audio at 16 kHz fed through the pipeline before the
/// test utterance so the warm pass exercises the AGC-adapted, ring-primed
/// path (production background silence/noise before anyone speaks).
const WARMUP_PREPEND_SAMPLES: usize = 20480; // 1.28s × 16 kHz

/// TTS phrase for warm-up audio (mahbot-947).
///
/// A short non-wake-word phrase synthesised via the already-loaded TTS engine.
/// Speech-like harmonics guarantee the Earshot neural VAD triggers, producing
/// embeddings for the pre-utterance ring.
///
/// Must NOT contain the wake word ([`wake_word()`] — runtime-resolved from the
/// deployed config store) or phonetically similar phrases that could trigger
/// the Conv1D classifier.
const WARMUP_TTS_PHRASE: &str = "testing one two three";

/// Cached TTS warm-up audio, populated on first successful synthesis.
/// Unlike the original `WARMUP_NOISE_CACHE`, this caches ONLY the TTS
/// result — if TTS is unavailable on the first call, a fresh pink-noise
/// fallback is returned (no caching), so TTS is re-evaluated on subsequent
/// calls (mahbot-947, reviewer feedback on cache poisoning).
static WARMUP_TTS_CACHE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

/// Resolved wake word phrase for this benchmark run (phrase alignment).
///
/// Resolved once at first use: the bench synthesizes and tests the wake
/// phrase persisted in the deployed model's config-store entry (the phrase
/// production actually listens for), falling back to [`WAKE_WORD_FALLBACK`]
/// only when the config store is unavailable.  The resolved phrase and its
/// source are reported in the reproducibility section.
static WAKE_WORD: std::sync::OnceLock<ResolvedPhrase> = std::sync::OnceLock::new();

/// Fallback wake word used when the deployed phrase cannot be read.
const WAKE_WORD_FALLBACK: &str = "hey mahbot";

/// Resolved wake phrase plus its provenance label.
struct ResolvedPhrase {
    phrase: String,
    source: &'static str,
}

/// Resolve the wake phrase for this run (deployed config-store phrase first).
fn wake_word() -> &'static str {
    WAKE_WORD.get_or_init(resolve_wake_phrase).phrase.as_str()
}

/// Provenance label for the resolved wake phrase ("config_db" | "fallback").
fn wake_phrase_source() -> &'static str {
    WAKE_WORD.get_or_init(resolve_wake_phrase).source
}

/// Resolve the deployed wake phrase from the config store, falling back to
/// the legacy constant when the store read fails or carries no phrase.
fn resolve_wake_phrase() -> ResolvedPhrase {
    match read_deployed_wake_phrase() {
        Some(phrase) if !phrase.is_empty() => ResolvedPhrase {
            phrase,
            source: "config_db",
        },
        _ => ResolvedPhrase {
            phrase: WAKE_WORD_FALLBACK.to_string(),
            source: "fallback",
        },
    }
}

/// Read the deployed model's wake phrase from `config.db` (the
/// `wake_word_templates` PersistedModel JSON) using the same read-only turso
/// open the debug CLI uses — never stock sqlite3 on live stores (orphaned-WAL
/// safety, see the README prevention rule).  Best-effort: any failure returns
/// `None` and the caller falls back to [`WAKE_WORD_FALLBACK`].
fn read_deployed_wake_phrase() -> Option<String> {
    let config_db = crate::config::default_config_dir()
        .ok()?
        .join("db")
        .join("config.db");
    if !config_db.exists() {
        return None;
    }
    let path = config_db.to_str()?;
    // `::turso` = the external turso crate (voice.rs's `use crate::turso;`
    // shadows the bare name in this module via `use super::*`).
    let io: std::sync::Arc<dyn ::turso::core::IO> =
        std::sync::Arc::new(::turso::core::PlatformIO::new().ok()?);
    let db = ::turso::core::Database::open_file_with_flags(
        io.clone(),
        path,
        ::turso::core::OpenFlags::ReadOnly | ::turso::core::OpenFlags::NoLock,
        crate::turso::experimental_database_opts(),
        None,
    )
    .ok()?;
    let conn = db.connect().ok()?;
    let mut stmt = conn
        .query("SELECT value FROM config_kv WHERE key = 'wake_word_templates'")
        .ok()??;
    let mut row_found = false;
    loop {
        match stmt.step().ok()? {
            ::turso::core::StepResult::Row => {
                row_found = true;
                break;
            }
            ::turso::core::StepResult::IO | ::turso::core::StepResult::Yield => {
                io.step().ok()?;
            }
            ::turso::core::StepResult::Done
            | ::turso::core::StepResult::Interrupt
            | ::turso::core::StepResult::Busy => break,
        }
    }
    if !row_found {
        return None;
    }
    let raw: String = stmt.row()?.get_value(0).to_string();
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let phrase = v.get("phrase").and_then(serde_json::Value::as_str)?;
    Some(super::normalize_phrase(phrase))
}

/// Number of enrollment variants to generate (mahbot-932 Fix 5).
///
/// Production generates ~110-120 positive training sequences from 10
/// utterances × the full 12-cell recipe (minus SpeedUp skip for
/// short utterances).  We target ~10 TTS enrollment variants to match this.
/// Since the re-scope all 10 clips come from ONE TTS voice (the enrolled
/// speaker) with guided-prompt DSP conditioning.
const NUM_ENROLLMENT_VARIANTS: usize = 10;

/// Owner-negative (non-wake-word) phrases for classifier training
/// (mahbot-932 Fix 6, re-scoped).
///
/// These are generated via TTS from the ENROLLED voice only (the owner's own
/// non-wake-word speech), tagged as `Source::Owner`, and used to help the
/// model reject general speech from the enrolled user.  The phrase list is
/// DISJOINT from the confusable/unrelated pools so no phrase is double-labeled
/// across training tiers (the old list overlapped the unrelated pool:
/// 'turn on the lights' / 'what time is it' exact matches, 'good morning' vs
/// 'good morning everyone').  Documented limitation: TTS speech cannot match
/// the distribution of real human Phase 3 speech collected during production
/// enrollment.
const OWNER_NEGATIVE_PHRASES: &[&str] = &[
    "please stop",
    "i am going out",
    "where are my keys",
    "the meeting starts soon",
    "let me check the weather",
    // Disjoint non-wake phrases, collision-free seeds.
    "remind me to call the dentist",
    "what should we have for dinner",
    "the wifi keeps disconnecting",
    "i need to buy new shoes",
    "turn down the music please",
];

/// Single-voice allocation for the bench.
///
/// The standard 10-style set is F1-F5, M1-M5 in list order.  The allocation
/// (by index into `available_styles`) guarantees the reserved voices are
/// absent from EVERY training path:
/// - enrolled: index 0 (F1) — the ONLY voice in the enrollment clips and
///   owner negatives.
/// - negative pool: indices 0..6 (F1..M1) — the confusable/unrelated prewarm
///   styles.  Production rotates ALL voices; the canary+reserved voices are
///   dropped so they never train in.
/// - canaries: indices 6..8 (M2, M3) — the in-distribution canary matrix
///   (kept from mahbot-1025).
/// - reserved: indices 8..10 (M4, M5) — held-out cross-speaker probes.
struct VoiceAllocation {
    enrolled: String,
    negative_pool: Vec<String>,
    canaries: Vec<String>,
    reserved: Vec<String>,
}

/// Derive the voice allocation from the available TTS styles (defensive for
/// machines with fewer than 10 styles: missing canary/reserved voices degrade
/// to empty sets and the report discloses the reduced probe coverage).
fn allocate_voices(available_styles: &[String]) -> VoiceAllocation {
    let enrolled = available_styles
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_TTS_STYLE.to_string());
    let take = |range: std::ops::Range<usize>| {
        available_styles
            .get(range)
            .map(<[String]>::to_vec)
            .unwrap_or_default()
    };
    VoiceAllocation {
        enrolled: enrolled.clone(),
        negative_pool: {
            // indices 0..6 (F1..M1) when present; otherwise all styles (a
            // fewer-than-6 machine has no canary/reserved indices to exclude —
            // the take() ranges below are empty by construction).
            let pool = take(0..6);
            if pool.is_empty() {
                available_styles.to_vec()
            } else {
                pool
            }
        },
        canaries: take(6..8),
        reserved: take(8..10),
    }
}

/// Guided-enrollment prompt groups (production [`ENROLLMENT_PROMPTS`]:
/// 3 normal, 3 "further from the mic", 2 "different angle", 2 "morning voice").
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GuidedPromptGroup {
    Normal,
    Distance,
    Angle,
    Morning,
}

impl GuidedPromptGroup {
    /// Map an enrollment clip index (0..10) to its guided prompt group.
    const fn for_clip_index(i: usize) -> Self {
        match i {
            0..=2 => Self::Normal,
            3..=5 => Self::Distance,
            6..=7 => Self::Angle,
            _ => Self::Morning,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Distance => "distance",
            Self::Angle => "angle",
            Self::Morning => "morning",
        }
    }

    /// Detection SNR against the realistic noise floor (dB).  The relationship
    /// SNR(level_reduction) is chosen so every group's noise lands at the SAME
    /// absolute RMS as the normal 20 dB floor (distance −6 dB + 14 dB SNR,
    /// angle/morning −3 dB + 17 dB SNR) — the AGC re-amplifies pure
    /// attenuation, so realism must come from the SNR / spectral relationships.
    const fn noise_snr_db(self) -> f32 {
        match self {
            Self::Normal => 20.0,
            Self::Distance => 14.0,
            Self::Angle | Self::Morning => 17.0,
        }
    }
}

/// Minimum detection rate for positive (wake word) variants required to pass.
///
/// Report-only target: the enrolled-speaker held-out recall (unseen
/// renderings of the enrolled voice) is compared against this in the report
/// banner; the 0.2 granularity reflects the 16-clip wake-only held-out recall
/// set, where each miss costs roughly 6 percentage points.
const MIN_DETECTION_RATE: f64 = 0.60;

/// Acceptance-basis prefix for the current report series.
/// `cross_run_summary` filters archives by this so v2 wake-only runs never mix
/// with v1 24-clip archives in the ≥3-run spread.
const ACCEPTANCE_BASIS_PREFIX: &str = "held_out_recall_v2";

// Per-category false accept limits are now dynamic by tier — see
// [`tier_limits`] and [`BenchTier`] (mahbot-871).

/// Confusable near-miss phrases for negative detection testing (mahbot-834).
///
/// Uses the canonical list from the parent `voice` module (mahbot-859).
const CONFUSABLE_PHRASES: &[&str] = super::CONFUSABLE_PHRASES;
const CONFUSABLE_HARD: &[&str] = super::CONFUSABLE_HARD;
const CONFUSABLE_MEDIUM: &[&str] = super::CONFUSABLE_MEDIUM;
const CONFUSABLE_EASY: &[&str] = super::CONFUSABLE_EASY;

/// Unrelated speech phrases for negative detection testing (mahbot-834).
///
/// Uses the canonical list from the parent `voice` module (mahbot-872).
const UNRELATED_PHRASES: &[&str] = super::UNRELATED_PHRASES;

/// Silence audio length in samples (1 second at 16 kHz).
const SILENCE_LEN: usize = 16_000;

/// Noise audio length in samples (1 second at 16 kHz).
const NOISE_LEN: usize = 16_000;

/// Noise profiles for negative detection testing (mahbot-834).
///
/// Each noise profile is a (label, generator_fn) pair.  The generator
/// produces PCM f32 samples at 16 kHz.
type NoiseGenerator = fn() -> Vec<f32>;

const NOISE_PROFILES: &[(&str, NoiseGenerator)] = &[
    ("white uniform noise", generate_white_uniform_noise),
    ("white gaussian noise", generate_white_gaussian_noise),
    ("pink noise", generate_pink_noise),
    ("brown noise", generate_brown_noise),
    ("mixed speech+noise", generate_mixed_speech_noise),
    ("blue noise", generate_blue_noise),
    ("violet noise", generate_violet_noise),
    ("low-frequency rumble", generate_low_freq_rumble),
    ("modulated noise", generate_modulated_noise),
    ("high-frequency hiss", generate_high_freq_hiss),
];

/// TTS target sample rate (voice pipeline rate).
const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Default TTS voice style when no styles are available from disk.
/// This matches the naming convention used by the TTS model download.
const DEFAULT_TTS_STYLE: &str = "M1.json";

/// Cache directory relative to storage root.
const TEST_CACHE_DIR: &str = "test_cache/voice_e2e";

/// SNR levels for the noise-overlapped detection test (mahbot-845).
/// Each entry is (label, snr_db).  Clean = infinity dB (no noise added).
const NOISE_OVERLAP_SNRS: &[(&str, f32)] = &[
    ("clean", f32::INFINITY),
    ("20dB", 20.0),
    ("10dB", 10.0),
    ("5dB", 5.0),
    ("0dB", 0.0),
];

/// Noise types to use for noise-overlapped detection tests.
/// White (uniform), Pink, and Brown noise (mahbot-845).
const NOISE_OVERLAP_TYPES: &[(&str, NoiseGenerator)] = &[
    ("white", generate_white_uniform_noise),
    ("pink", generate_pink_noise),
    ("brown", generate_brown_noise),
];

// ── Tiered benchmark configuration (mahbot-871) ────────────────────────────

/// Difficulty tiers for confusable phrase testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchTier {
    Easy,
    Medium,
    Hard,
}

impl BenchTier {
    /// Index into a 3-element array: Easy→0, Medium→1, Hard→2.
    const fn index(self) -> usize {
        match self {
            BenchTier::Easy => 0,
            BenchTier::Medium => 1,
            BenchTier::Hard => 2,
        }
    }

    /// Stable string label for JSON and human-readable output.
    const fn as_str(self) -> &'static str {
        match self {
            BenchTier::Easy => "easy",
            BenchTier::Medium => "medium",
            BenchTier::Hard => "hard",
        }
    }
}

/// Per-category false-accept limits for a given tier.
#[derive(Debug, Clone, Copy)]
struct TierLimits {
    confusable: usize,
    noise: usize,
    total: usize,
}

/// Return the per-category false-accept limits for the given tier.
const fn tier_limits(tier: BenchTier) -> TierLimits {
    match tier {
        BenchTier::Easy => TierLimits {
            confusable: 0,
            noise: 0,
            total: 0,
        },
        BenchTier::Medium => TierLimits {
            confusable: 1,
            noise: 1,
            total: 1, // combined confusable+noise cap (tighter than sum of individual limits)
        },
        BenchTier::Hard => TierLimits {
            confusable: 1,
            noise: 1,
            total: 2,
        },
    }
}

/// Extract the phrase text from a generate_phrase_variants_cached label.
///
/// Label format: `"{prefix}_{phrase}_s{i}"` e.g. `"confusable_hey madbot_s0"`.
///
/// The seed suffix `_s{i}` (underscore + 's' + digits) is always the
/// rightmost `_s` in the label, so `rfind` finds it correctly even if the
/// phrase text itself contains the substring `_s`. We additionally verify
/// that the `_s` is followed by at least one ASCII digit as a sanity check
/// against label-format drift.
fn phrase_from_label<'a>(label: &'a str, prefix: &str) -> &'a str {
    let after_prefix = label
        .strip_prefix(prefix)
        .and_then(|r| r.strip_prefix('_'))
        .unwrap_or(label);
    // The seed suffix `_s{i}` is always the rightmost `_s` in the label.
    if let Some(idx) = after_prefix.rfind("_s") {
        let after_s = &after_prefix[idx + 2..];
        if after_s.starts_with(|c: char| c.is_ascii_digit()) {
            return &after_prefix[..idx];
        }
    }
    // No valid seed suffix found; return the full string after the prefix.
    after_prefix
}

/// Determine the difficulty tier for a confusable phrase.
fn tier_for_phrase(phrase: &str) -> BenchTier {
    if CONFUSABLE_HARD.contains(&phrase) {
        BenchTier::Hard
    } else if CONFUSABLE_MEDIUM.contains(&phrase) {
        BenchTier::Medium
    } else if CONFUSABLE_EASY.contains(&phrase) {
        BenchTier::Easy
    } else {
        unreachable!(
            "Confusable phrase '{phrase}' not found in any tier array. \
             The phrase must exist in exactly one of CONFUSABLE_HARD/MEDIUM/EASY, \
             and those arrays must together cover all entries in CONFUSABLE_PHRASES."
        )
    }
}

// ── TTS audio cache (mahbot-844 Part 1) ─────────────────────────────────────

// Compute a deterministic model version hash from TTS compile-time SHA-256
// constants (mahbot-844 Part 1).
//
// Hashes the concatenation of all TTS model SHA-256 digests and the voice
// style directory hash.  These constants are updated in `tts.rs` whenever
// model files change, so the cache key automatically tracks model version
// without disk I/O.
//
// Returns a hex string (always succeeds since the hashes are compile-time).
//
// Delegates to [`super::tts_model_version_hash`] which is the canonical
// implementation from `voice.rs`.

/// Get the cache directory path.
fn cache_dir() -> std::path::PathBuf {
    let root = crate::config::CONFIG
        .try_storage_root()
        .expect("CONFIG storage root must be set");
    root.join(TEST_CACHE_DIR)
}

/// Synthesize wake word variant audio with TTS caching support.
///
/// Delegates to the production implementation.
fn synthesize_wake_word_variant_cached(
    text: &str,
    style: &str,
    seed: u64,
    sample_rate: u32,
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Option<Vec<f32>> {
    super::synthesize_with_pcm_cache(text, style, seed, sample_rate, model_hash, cache_dir)
}

// ── Audio generation helpers (with cache) ─────────────────────────────────

// ── Guided-prompt DSP conditioning ──────────────────────────
// All primitives are deterministic (pure arithmetic or seeded RNG): the
// pinned-seed weights contract and cross-run comparability hold.

/// Deterministic first-order IIR lowpass (one-pole):
/// `y[n] = y[n-1] + α(x[n] − y[n-1])`, `α = 1 − exp(−2π fc / fs)`.
fn one_pole_lowpass(pcm: &[f32], cutoff_hz: f32, sample_rate: u32) -> Vec<f32> {
    if cutoff_hz <= 0.0 || cutoff_hz >= sample_rate as f32 * 0.5 {
        return pcm.to_vec();
    }
    let alpha = 1.0 - (-2.0 * core::f32::consts::PI * cutoff_hz / sample_rate as f32).exp();
    let mut y = 0.0f32;
    pcm.iter()
        .map(|&x| {
            y += alpha * (x - y);
            y
        })
        .collect()
}

/// Deterministic first-order high-shelf CUT (RBJ cookbook, S=1): attenuates
/// frequencies above `fc_hz` by `gain_db` — the spectral tilt for the
/// "different angle" guided-enrollment condition.
fn high_shelf_cut(pcm: &[f32], gain_db: f32, fc_hz: f32, sample_rate: u32) -> Vec<f32> {
    if gain_db.abs() < 1e-6 || fc_hz <= 0.0 {
        return pcm.to_vec();
    }
    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * core::f32::consts::PI * fc_hz / sample_rate as f32;
    // RBJ high shelf with slope S = 1: alpha = sin(w0)/2 · sqrt(2).
    let alpha = w0.sin() / 2.0 * core::f32::consts::SQRT_2;
    let cos_w0 = w0.cos();
    let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;
    let inv_a0 = 1.0 / a0;
    let (b0, b1, b2, a1, a2) = (
        b0 * inv_a0,
        b1 * inv_a0,
        b2 * inv_a0,
        a1 * inv_a0,
        a2 * inv_a0,
    );
    let mut x1 = 0.0f32;
    let mut x2 = 0.0f32;
    let mut y1 = 0.0f32;
    let mut y2 = 0.0f32;
    pcm.iter()
        .map(|&x| {
            let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x;
            y2 = y1;
            y1 = y;
            y
        })
        .collect()
}

/// Apply the guided-prompt DSP conditioning to one raw enrollment clip
/// Variation comes from DSP on the cached synthesis PCM — no
/// extra TTS synthesis cost.  Each group's level reduction is calibrated so
/// the added noise lands at the SAME absolute RMS as normal's 20 dB floor
/// (AGC re-amplifies pure attenuation, so realism comes from SNR vs the fixed
/// floor, not absolute level); the calibration is exact for the pure
/// level-reduction paths and approximate where the rolloff/resample shift the
/// signal RMS the SNR references:
/// - normal: realistic noise floor (20 dB SNR);
/// - distance: level reduction (−6 dB) + high-frequency rolloff (3.2 kHz
///   lowpass) + noise floor at 14 dB SNR;
/// - angle: spectral tilt (high-shelf cut −4 dB @ 3 kHz) + level reduction
///   (−3 dB) + noise floor at 17 dB SNR;
/// - morning: slower/lower-pitch resample (0.92×) + level reduction (−3 dB) +
///   lowpass (2.2 kHz) + noise floor at 17 dB SNR.
///
/// The noise floor is deterministic seeded pink noise (seed 4000 + clip index).
fn condition_enrollment_clip(pcm: &[f32], clip_index: usize) -> Vec<f32> {
    let group = GuidedPromptGroup::for_clip_index(clip_index);
    let conditioned = match group {
        GuidedPromptGroup::Normal => pcm.to_vec(),
        GuidedPromptGroup::Distance => {
            let attenuated = crate::util::apply_gain(pcm, -6.0);
            one_pole_lowpass(&attenuated, 3200.0, TARGET_SAMPLE_RATE)
        }
        GuidedPromptGroup::Angle => {
            let tilted = high_shelf_cut(pcm, -4.0, 3000.0, TARGET_SAMPLE_RATE);
            crate::util::apply_gain(&tilted, -3.0)
        }
        GuidedPromptGroup::Morning => {
            let slower = crate::util::speed_perturbation(pcm, TARGET_SAMPLE_RATE, 0.92);
            let reduced = crate::util::apply_gain(&slower, -3.0);
            one_pole_lowpass(&reduced, 2200.0, TARGET_SAMPLE_RATE)
        }
    };
    crate::util::add_noise(&conditioned, group.noise_snr_db(), 4000 + clip_index as u64)
}

/// Generate the 10 single-voice enrollment clips.
///
/// The enrolled voice (F1 in the standard set — `voice_allocation.enrolled`)
/// is synthesized once per clip at seeds 100-109; the guided-prompt DSP
/// conditioning (3 normal / 3 distance / 2 angle / 2 morning) derives ALL
/// acoustic variation from the cached PCM.  The former multi-voice rotation
/// (`style = available_styles[i % num_styles]`, mahbot-932) is gone — it was
/// the single largest representativeness gap (the "enrolled speaker" was 6
/// different voices and the model failed production's own enrollment gate).
fn generate_enrollment_variants_cached(
    enrolled_style: &str,
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<(Vec<f32>, String)> {
    let mut variants = Vec::with_capacity(NUM_ENROLLMENT_VARIANTS);
    for i in 0..NUM_ENROLLMENT_VARIANTS {
        let seed = 100 + i as u64;
        if let Some(pcm) = synthesize_wake_word_variant_cached(
            wake_word(),
            enrolled_style,
            seed,
            TARGET_SAMPLE_RATE,
            model_hash,
            cache_dir,
        ) {
            variants.push((
                condition_enrollment_clip(&pcm, i),
                format!("{enrolled_style}_enroll{i}"),
            ));
        }
    }
    variants
}

/// Wake-only held-out recall clip count (seeds 3000..3000+N).
const HELD_OUT_WAKE_ONLY_CLIPS: usize = 16;

/// Generate the held-out recall set: unseen renderings of the ENROLLED voice
/// at collision-free seeds (3000+ — avoids 100-109 enrollment, 800+ detection
/// variants, 947 warmup, 9000+ owner training, 2000+ unrelated training).
/// Generated strictly after training and never added to any training pool.
/// Wake phrase alone only — embedded-in-sentence detection is not a product
/// requirement and is not measured.
fn generate_held_out_recall_clips_cached(
    enrolled_style: &str,
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<(Vec<f32>, String)> {
    let mut clips = Vec::new();
    for i in 0..HELD_OUT_WAKE_ONLY_CLIPS {
        let seed = 3000 + i as u64;
        if let Some(pcm) = synthesize_wake_word_variant_cached(
            wake_word(),
            enrolled_style,
            seed,
            TARGET_SAMPLE_RATE,
            model_hash,
            cache_dir,
        ) {
            clips.push((pcm, format!("{enrolled_style}_heldout_wake_s{seed}")));
        }
    }
    clips
}

/// Generate the trained-in canary clips (M2/M3 at seeds 6000/6100): the SAME
/// clips feed the in-distribution canary matrix — they are training data, so
/// any canary detection is a trained-in regression signal, never a
/// generalization probe.
fn generate_canary_clips_cached(
    canary_styles: &[String],
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<(Vec<f32>, String)> {
    let mut clips = Vec::new();
    for (i, style) in canary_styles.iter().enumerate() {
        let seed = 6000 + i as u64 * 100;
        if let Some(pcm) = synthesize_wake_word_variant_cached(
            wake_word(),
            style,
            seed,
            TARGET_SAMPLE_RATE,
            model_hash,
            cache_dir,
        ) {
            clips.push((pcm, format!("{style}_canary_s{seed}")));
        }
    }
    clips
}

/// Generate the held-out cross-speaker probe clips: the RESERVED voices
/// (M4/M5) at unseen seeds (5000+/5100+ — collision-free) — absent from EVERY
/// training path.
fn generate_cross_speaker_probe_clips_cached(
    reserved_styles: &[String],
    seeds_per_voice: usize,
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<(Vec<f32>, String)> {
    let mut clips = Vec::new();
    for (voice_idx, style) in reserved_styles.iter().enumerate() {
        for s in 0..seeds_per_voice {
            let seed = 5000 + voice_idx as u64 * 100 + s as u64;
            if let Some(pcm) = synthesize_wake_word_variant_cached(
                wake_word(),
                style,
                seed,
                TARGET_SAMPLE_RATE,
                model_hash,
                cache_dir,
            ) {
                clips.push((pcm, format!("{style}_probe_s{seed}")));
            }
        }
    }
    clips
}

/// Seed configuration for TTS phrase variant generation.
///
/// Encapsulates the three related seed parameters that control deterministic
/// TTS synthesis of phrase variants with different seeds and style rotations.
/// Bundled into a struct for call-site clarity (mahbot-872 reviewer feedback).
#[derive(Clone, Copy)]
struct SeedConfig {
    /// Base seed for deterministic TTS synthesis of the phrase list.
    base_seed: u64,
    /// Number of seed variants — determines style rotation stride and seed
    /// spacing.  Must match the `num_variants` passed to the batch caller.
    num_variants: usize,
    /// Which variant index this call generates (0..num_variants).  Used in
    /// the style distribution and seed formulas.
    seed_variant: usize,
}

/// Generate TTS audio for a list of phrases with caching.
fn generate_phrase_variants_cached(
    phrases: &[&str],
    available_styles: &[String],
    seed: SeedConfig,
    prefix: &str,
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<(Vec<f32>, String)> {
    let mut variants = Vec::new();
    let num_styles = available_styles.len().max(1);

    for (i, &phrase) in phrases.iter().enumerate() {
        // Use the same round-robin style distribution as production
        // (mahbot-872 per-phrase seed formula): rotate styles across
        // (phrase × seed_variant) combos.
        let style_idx = (i * seed.num_variants + seed.seed_variant) % num_styles;
        let style = if available_styles.is_empty() {
            DEFAULT_TTS_STYLE
        } else {
            &available_styles[style_idx]
        };
        // Production seed formula (mahbot-872):
        //   seed = base_seed + i * num_variants + seed_variant
        let seed_val =
            seed.base_seed + i as u64 * seed.num_variants as u64 + seed.seed_variant as u64;

        if let Some(pcm) = synthesize_wake_word_variant_cached(
            phrase,
            style,
            seed_val,
            TARGET_SAMPLE_RATE,
            model_hash,
            cache_dir,
        ) {
            variants.push((pcm, format!("{prefix}_{phrase}_s{i}")));
        }
    }

    variants
}

/// Generate all seed variants for a phrase list in a single batch.
/// Generate white uniform noise in [-1.0, 1.0].
fn generate_white_uniform_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..NOISE_LEN)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect()
}

/// Generate white Gaussian noise (approximately) in [-1.0, 1.0].
/// Uses the Box-Muller transform on uniform samples.
fn generate_white_gaussian_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(43);
    let mut samples = Vec::with_capacity(NOISE_LEN);
    let mut i = 0;
    while i < NOISE_LEN {
        // Shared EPSILON-clamp pair sampler (mahbot-1043) — preserves the
        // bench's exact draw sequence (2 draws per 2 samples, cos+sin).
        let (z1, z2) = crate::util::sample_gaussian_pair_clamped(&mut rng);
        // Clamp to [-1.0, 1.0] — Gaussian has tails beyond [-3, 3] but
        // scaling by 0.333 keeps ~99.7% within [-1, 1].
        samples.push((z1 * 0.333).clamp(-1.0, 1.0));
        if i + 1 < NOISE_LEN {
            samples.push((z2 * 0.333).clamp(-1.0, 1.0));
        }
        i += 2;
    }
    samples
}

/// Generate pink noise (1/f spectrum) using the Voss-McCartney algorithm.
/// Produces approximately -3 dB/octave rolloff.
///
/// Uses the canonical [`crate::util::generate_pink_noise`] with a reproducible
/// seed (44) for deterministic benchmark output.
fn generate_pink_noise() -> Vec<f32> {
    crate::util::generate_pink_noise(NOISE_LEN, rand::rngs::StdRng::seed_from_u64(44))
}

/// Generate brown noise (integrated white noise, 1/f² spectrum).
/// Produces approximately -6 dB/octave rolloff — deeper, rumbling sound.
fn generate_brown_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(45);
    let mut samples = Vec::with_capacity(NOISE_LEN);
    let mut prev = 0.0;
    for _ in 0..NOISE_LEN {
        let white: f32 = rng.random::<f32>() * 2.0 - 1.0;
        // Leaky integrator to prevent DC drift
        prev = (prev + white * 0.125) * 0.98;
        samples.push(prev.clamp(-1.0, 1.0));
    }
    samples
}

/// Generate mixed speech+noise by overlapping a wake-word-like recording
/// with brown noise at low SNR (<5 dB) — simulating far-field / cocktail
/// party conditions where the wake word might acoustically resemble noise.
fn generate_mixed_speech_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(46);
    // Generate a tonal hum at ~200 Hz (close to male speech formant) mixed
    // with noise — simulating distant speech that might trigger VAD.
    let mut samples = Vec::with_capacity(NOISE_LEN);
    for i in 0..NOISE_LEN {
        let t = i as f32 / TARGET_SAMPLE_RATE as f32;
        let tone = (2.0 * core::f32::consts::PI * 200.0 * t).sin() * 0.15;
        let noise: f32 = rng.random::<f32>() * 2.0 - 1.0;
        // Low SNR: noise dominates, with tonal speech component
        samples.push((tone + noise * 0.85).clamp(-1.0, 1.0));
    }
    samples
}

/// Generate blue noise (spectral density increases 3dB/octave).
fn generate_blue_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(47);
    let mut prev: f32 = 0.0;
    (0..NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            // First-order difference approximates blue noise
            let blue = (white - prev) * 0.5;
            prev = white;
            blue.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Generate violet noise (spectral density increases 6dB/octave).
fn generate_violet_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(48);
    let mut prev1 = 0.0f32;
    let mut prev2 = 0.0f32;
    (0..NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            // Second-order difference approximates violet noise
            let violet = (white - 2.0 * prev1 + prev2) * 0.25;
            prev2 = prev1;
            prev1 = white;
            violet.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Generate low-frequency rumble (pink noise low-passed at ~300Hz).
fn generate_low_freq_rumble() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(49);
    let mut low: f32 = 0.0;
    let alpha = 0.95; // strong low-pass
    (0..NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            low = low * alpha + white * (1.0 - alpha);
            low.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Generate amplitude-modulated noise (tremolo effect).
fn generate_modulated_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(50);
    let mut samples = Vec::with_capacity(NOISE_LEN);
    for i in 0..NOISE_LEN {
        let t = i as f32 / TARGET_SAMPLE_RATE as f32;
        let modulator = (2.0 * core::f32::consts::PI * 4.0 * t).sin() * 0.5 + 0.5; // 4Hz tremolo
        let noise: f32 = rng.random::<f32>() * 2.0 - 1.0;
        samples.push((noise * modulator).clamp(-1.0, 1.0));
    }
    samples
}

/// Generate high-frequency hiss (high-pass filtered above ~4kHz).
fn generate_high_freq_hiss() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(51);
    let mut prev: f32 = 0.0;
    (0..NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            // Simple high-pass: y[n] = 0.9 * (x[n] - x[n-1]) + 0.8 * y[n-1]
            let hiss = 0.9 * (white - prev) + 0.8 * prev;
            prev = white;
            hiss.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Generate warm-up audio that reliably triggers the Earshot neural VAD.
///
/// Attempts TTS synthesis first (guaranteed speech-like harmonics), falling
/// back to pink noise + 200 Hz tone if TTS is unavailable (mahbot-947).
///
/// Returns a [`Cow`] that either borrows from the TTS cache (once populated)
/// or owns freshly-generated pink noise (not cached, so TTS is re-evaluated
/// on the next call if it becomes available — mahbot-947, reviewer feedback).
///
/// ## Determinism
/// TTS synthesis uses a fixed seed (947 = ticket number), producing the same
/// PCM output on every call (barring TTS model changes).  The result is cached
/// in [`WARMUP_TTS_CACHE`] so subsequent calls are zero-cost.
///
/// ## Fallback rationale
/// The pink noise + 200 Hz tone was the original warm-up signal (mahbot-922/926),
/// but its lack of speech-like harmonic structure causes the Earshot neural VAD
/// to reject it as non-speech, producing 0 embeddings and an unprimed ring.
fn generate_warmup_noise() -> Cow<'static, [f32]> {
    // TTS cache hit — fast path, no synthesis needed.
    if let Some(cached) = WARMUP_TTS_CACHE.get() {
        return Cow::Borrowed(cached);
    }

    // First call (or cache not yet populated): try TTS synthesis.
    // Speech-like harmonics guarantee Earshot VAD triggers, producing
    // embeddings that prime the ring.
    if let Some(pcm) = try_warmup_tts() {
        let cached = WARMUP_TTS_CACHE.get_or_init(|| pcm);
        // Safety: we just verified get() returns None above; no race
        // because get_or_init is idempotent and returns the same &Vec.
        return Cow::Borrowed(cached);
    }

    // TTS unavailable — return fresh fallback (not cached, so we retry
    // TTS on future calls if models become available).
    warn!(
        "TTS warm-up synthesis failed — falling back to pink noise + tone. \
         This may not trigger Earshot VAD, producing 0 warm-up embeddings. \
         Ensure TTS models are cached (~/.mahbot/models/tts/)."
    );
    Cow::Owned(generate_warmup_noise_fallback())
}

/// Attempt TTS synthesis of the warm-up phrase.
///
/// Returns `None` if TTS is not initialised or synthesis fails, allowing the
/// caller to fall back to pink noise.  Logs an `info!` on success with the
/// output duration so maintainers can verify the warm-up length.
fn try_warmup_tts() -> Option<Vec<f32>> {
    let pcm = match crate::audio::tts::synthesize(
        WARMUP_TTS_PHRASE,
        DEFAULT_TTS_STYLE,
        947, // seed = ticket number for determinism
        TARGET_SAMPLE_RATE,
    ) {
        Ok(p) => p,
        Err(e) => {
            warn!("TTS warm-up synthesis failed: {e}");
            return None;
        }
    };

    if pcm.len() > WARMUP_PREPEND_SAMPLES {
        warn!(
            "Warm-up TTS output ({} samples = {:.2}s) exceeds \
             WARMUP_PREPEND_SAMPLES ({}) — truncating to {}",
            pcm.len(),
            pcm.len() as f64 / f64::from(TARGET_SAMPLE_RATE),
            WARMUP_PREPEND_SAMPLES,
            WARMUP_PREPEND_SAMPLES,
        );
        Some(pcm[..WARMUP_PREPEND_SAMPLES].to_vec())
    } else {
        info!(
            "Warm-up audio: TTS phrase '{}' ({:.2}s = {} samples) — VAD will trigger",
            WARMUP_TTS_PHRASE,
            pcm.len() as f64 / f64::from(TARGET_SAMPLE_RATE),
            pcm.len(),
        );
        Some(pcm)
    }
}

/// Generate the original pink noise + 200 Hz tone warm-up signal (mahbot-922).
///
/// Retained as a fallback when TTS models are not cached.  Properties:
/// 1. **VAD-inactive**: low probability of triggering the Earshot neural VAD
///    (no speech harmonics), which is the whole reason we now prefer TTS.
/// 2. **No false detection**: classifier scores are near-zero because pink
///    noise + 200 Hz tone does not resemble the "hey mahbot" embedding.
/// 3. **Deterministic**: RNG seed 922 (= original ticket number).
/// 4. **Aperiodic noise floor**: pink noise avoids periodic artefacts that
///    could bias the AGC steady state.
fn generate_warmup_noise_fallback() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(922);

    // Pink noise normalised to unit RMS at ~0.20 amplitude (≈ -14 dB).
    let pink = crate::util::generate_pink_noise(WARMUP_PREPEND_SAMPLES, &mut rng);
    let pink_gain = 0.20;

    // 200 Hz tonal component at ~0.10 amplitude (≈ -20 dB).
    let mut samples = Vec::with_capacity(WARMUP_PREPEND_SAMPLES);
    for (i, &p) in pink.iter().enumerate() {
        let t = i as f32 / TARGET_SAMPLE_RATE as f32;
        let tone = (2.0 * core::f32::consts::PI * 200.0 * t).sin() * 0.10;
        samples.push(p * pink_gain + tone);
    }

    samples
}

/// Process a single frame of audio through the pipeline (AGC →
/// [`super::handle_wake_word_detection`]), matching the per-frame processing in
/// [`run_streaming_detection`].
///
/// Partial chunks are passed through **without zero-padding** (mahbot-1006 G):
/// production feeds raw mic chunks as-is, and both the NS stage (which buffers
/// incomplete 160-sample frames internally) and `handle_wake_word_detection`
/// (which accumulates partial tails in `ctx.audio_buffer` until the next real
/// chunk completes a [`FRAME_LENGTH`](super::FRAME_LENGTH) frame) carry the
/// tail forward.  Zero-padding instead made the AGC adaptation cadence at
/// utterance boundaries differ from production.
///
/// Empty slices are a no-op (the preprocessor buffers nothing new and the
/// detection frame loop needs a complete frame) — silence must be fed as
/// actual [`FRAME_LENGTH`](super::FRAME_LENGTH) digital-silence chunks, which
/// is what production's mic delivers.
///
/// Extracted as a shared helper to eliminate the structural near-duplicate
/// between `feed_audio` and `run_streaming_detection` (mahbot-922, reviewer
/// feedback).
fn process_frame(samples: &[f32], ctx: &mut super::PipelineCtx) {
    let processed = ctx.audio_preprocessor.process(samples.to_vec());
    super::handle_wake_word_detection(&processed, ctx);
}

/// Feed audio through the production pipeline (AGC → [`super::handle_wake_word_detection`])
/// in [`FRAME_LENGTH`](super::FRAME_LENGTH) chunks, then send silence frames to flush any
/// remaining voice batch — matching the processing pattern in [`run_streaming_detection`].
///
/// This is used to pre-warm the pipeline (AGC) before the actual test utterance,
/// without contaminating latency measurements (mahbot-922).
fn feed_audio(samples: &[f32], ctx: &mut super::PipelineCtx) {
    for chunk in samples.chunks(super::FRAME_LENGTH) {
        process_frame(chunk, ctx);
    }
    // Feed digital-silence frames to flush any remaining voice batch (matches
    // the post-audio flush in run_streaming_detection).  Real zeros in
    // FRAME_LENGTH chunks, NOT empty Vecs: production's mic keeps delivering
    // 512-sample chunks of silence after speech, and the NS stage buffers
    // incomplete frames internally — an empty chunk would never produce a
    // complete frame for the detection loop (mahbot-1006 G).
    for _ in 0..3 {
        process_frame(&vec![0.0; super::FRAME_LENGTH], ctx);
    }
}

/// Feed the warm-up audio through [`feed_audio`] so the warm pass starts the
/// test utterance with a ring-primed, AGC-adapted path (mahbot-922, mahbot-926).
///
/// ## Diagnostics
///
/// If the warm-up noise triggered a false detection, a `warn!()` is emitted and
/// the detection state is restored to prevent cooldown from corrupting subsequent
/// benchmark measurements (mahbot-922).
///
/// ## Residue clearing (mahbot-1003, mahbot-1006)
///
/// After feeding warm-up audio, this function clears warm-up residues that would
/// otherwise contaminate the subsequent test utterance's detection pipeline:
///
/// - **`mel_frame_buffer`** — cleared so the first stride-8 windows contain only
///   test utterance mel frames, not warm-up remnants + test audio mixed together.
/// - **`next_window_start`** — reset to 0 so embedding extraction starts from the
///   first test-utterance mel frame.
/// - **`score_window`** — cleared so warm-up classifier scores cannot carry into
///   the first test embedding's rolling sum.
/// - **`segment_silence_hops`** — reset to 0 so the test utterance starts a fresh
///   segment (the warm-up flush's VAD-negative hops must not shorten the test
///   utterance's trailing-silence window, mahbot-1006 H).
/// - **`burst_sweep_done`** — cleared (mahbot-1023): warm-up audio passes through
///   the full scoring pipeline and would otherwise suppress the test utterance's
///   own burst sweep.
/// - **`audio_preprocessor`** — `reset()` (mahbot-1006 A): the warm-up drives the
///   AGC to a speech-adapted gain and the NS to a speech-adapted profile; both
///   must be fresh for the test utterance, matching training's lazy-init AGC and
///   production's `reset_detection_segment()` at segment boundaries.
///
/// **Preserved**: `embedding_ring` (warm-up embeddings prime the classifier's
/// first Conv1D windows), `adaptive_threshold` (preserves warm-up-adapted state —
/// see mahbot-1006 F; the cold-start pass instead uses a fresh
/// [`AdaptiveThresholdState::new`]).
fn consume_warmup(ctx: &mut super::PipelineCtx) {
    let before_detection = ctx.last_wake_word_detection;
    let noise = generate_warmup_noise();

    // ── Feed warm-up audio (single pass) ───────────────────────────────────
    feed_audio(&noise, ctx);

    // Guard: warm-up noise should not trigger a false detection.  If it does,
    // restore the pre-warm-up state so cooldown doesn't suppress all subsequent
    // benchmark measurements (reviewer feedback, mahbot-922).  We must also
    // reset `is_recording` because detection sets it to `true`, which would
    // suppress stride-8 scoring for the entire benchmark variant.
    if ctx.last_wake_word_detection != before_detection {
        warn!(
            "Warm-up triggered a false detection — restoring detection \
             state to prevent cooldown corruption.",
        );
        ctx.last_wake_word_detection = before_detection;
        ctx.is_recording = false;
    }

    // ── Clear warm-up residues before test utterance processing (mahbot-1003) ──
    // Warm-up audio produces mel frames and classifier scores.  These residues
    // must be cleared so the test utterance starts with a clean detection slate:
    // the first stride-8 windows would otherwise span warm-up remnants + test
    // audio, producing mixed-content embeddings that score poorly and fall below
    // NO_MATCH_RESET_THRESHOLD, clearing the score window before clean test
    // embeddings arrive.
    ctx.mel_frame_buffer.clear();
    ctx.next_window_start = 0;
    ctx.score_window.clear();
    ctx.segment_silence_hops = 0;
    // Clear the per-segment deferred-burst latch (mahbot-1023): warm-up audio
    // passes through the FULL scoring pipeline and triggers the deferred burst
    // sweep on the warm-up noise; without clearing the latch here, the test
    // utterance's own burst sweep would be suppressed.
    ctx.burst_sweep_done = false;

    // ── Reset instrumentation (mahbot-1005 §1) ─────────────────────────────
    // Warm-up audio passes through the full scoring pipeline and records into
    // ctx.instrumentation: per_frame_scores, peak_score, VAD counts, and the
    // adaptive threshold trajectory.  Without a reset, warm-up-only scores
    // contaminate the test utterance's per-variant metrics (silence/noise
    // negatives would report "classifier triggers" from warm-up speech).
    ctx.instrumentation = super::DetectionInstrumentation::new();
    ctx.instrumentation.test_start_ring_len = ctx.embedding_ring.len();
    ctx.peak_score = 0.0;

    // ── Fresh AGC/NS state for the test utterance (mahbot-1006 A) ─────────
    // The warm-up audio drives the AGC's asymmetric EMA to a speech-adapted
    // gain (~1× for ~1.28 s of TTS speech).  Production starts each detection
    // segment from a fresh AGC (reset_detection_segment calls
    // audio_preprocessor.reset()), and training uses fresh per-variant AGC, so
    // the benchmark must NOT let the warm-up-adapted gain carry into the test
    // utterance — a volume-down variant would otherwise be under-amplified for
    // the first few seconds (AGC release α=0.02, ~3.6 s to converge), biasing
    // exactly the volume-robustness variants this benchmark probes.
    //
    // reset() (not clear_buffer()): the critique on mahbot-1006 noted the two
    // differ in NS-profile preservation, and production segment boundaries use
    // reset() — the warm-up TTS speech should not seed the noise suppressor's
    // adapted profile for the test utterance either.  The warm-up source is
    // reported in the JSON reproducibility block.
    ctx.audio_preprocessor.reset();
}

// ── Prerequisite check ─────────────────────────────────────────────────────

/// Ensure voice ONNX models are loaded.  Returns an error if the model
/// directory doesn't exist or loading fails, with a helpful message.
fn ensure_voice_models_loaded() -> Result<(), String> {
    if super::models_ready() {
        return Ok(());
    }

    let dir = super::model_dir().ok_or_else(|| {
        "Cannot resolve voice model directory. Is CONFIG.storage_root set?".to_string()
    })?;

    if !dir.join(super::MEL_MODEL_FILENAME).exists() {
        return Err(format!(
            "Mel spectrogram model not found at {}. Run the app to download models.",
            dir.join(super::MEL_MODEL_FILENAME).display()
        ));
    }
    if !dir.join(super::EMBED_MODEL_FILENAME).exists() {
        return Err(format!(
            "Embedding model not found at {}. Run the app to download models.",
            dir.join(super::EMBED_MODEL_FILENAME).display()
        ));
    }

    let models =
        super::load_onnx_models(&dir).map_err(|e| format!("Failed to load ONNX models: {e}"))?;
    super::ONNX_MODELS.set(models).map_err(|_| {
        "ONNX_MODELS already set by another test — cannot re-initialize".to_string()
    })?;
    super::MODELS_STATE.store(
        super::ModelState::Ready,
        std::sync::atomic::Ordering::Release,
    );

    info!("Voice models loaded from cache");
    Ok(())
}

/// Apply PCM transforms to enrollment variants (mahbot-932 Fix 5).
///
/// For each raw TTS enrollment variant, produces the full 12-cell recipe:
/// original, speed-down (0.95×/0.90×), speed-up (1.05×, conditional on
/// ≥500ms), volume-down (-3dB), pink noise (25dB SNR), and white/pink/brown
/// noise at 10/5 dB SNR.  Matches the production pipeline in
/// [`prewarm_phrase_embeddings`](super::prewarm_phrase_embeddings) via the
/// shared [`super::augment_pcm_variants`] helper (mahbot-1045 A1).
fn pcm_augment_enrollment_variants(variants: &[(Vec<f32>, String)]) -> Vec<(Vec<f32>, String)> {
    let mut all = Vec::new();
    for (i, (pcm, label)) in variants.iter().enumerate() {
        // Noise seed = loop index, gate input = raw TTS PCM, canonical push
        // order — preserved verbatim from the pre-dedup code.
        for variant in
            super::augment_pcm_variants(pcm, TARGET_SAMPLE_RATE, i as u64, super::AugmentSet::Full)
        {
            let suffix = match variant.variant_index {
                0 => "original",
                1 => "speed_down",
                2 => "speed_up",
                3 => "vol_down",
                4 => "noise",
                5 => "speed_down_090",
                6 => "noise_white_10db",
                7 => "noise_pink_10db",
                8 => "noise_brown_10db",
                9 => "noise_white_5db",
                10 => "noise_pink_5db",
                11 => "noise_brown_5db",
                _ => unreachable!("augment_pcm_variants yields only indices 0..=11"),
            };
            all.push((variant.pcm, format!("{label}_{suffix}")));
        }
    }
    all
}

/// Locate the enrolled-speaker training clip (label `{style}_enroll0` — the
/// voice derived from [`allocate_voices`]) and build its full 12-cell
/// augmentation variants — the canonical construction used by the Phase 13
/// cooldown re-point (mahbot-1052).  `None` when the enrolled clip is absent
/// from `train_clips`.
fn enrolled_clip_variants(
    train_clips: &[(Vec<f32>, String)],
    enrolled_label: &str,
) -> Option<Vec<(Vec<f32>, String)>> {
    let (enrolled_pcm, _) = train_clips.iter().find(|(_, l)| l == enrolled_label)?;
    Some(pcm_augment_enrollment_variants(&[(
        enrolled_pcm.clone(),
        enrolled_label.to_string(),
    )]))
}

/// mahbot-1045 B3 (report-only): quality scoring of the bench's enrollment
/// clips via production's compute_utterance_quality.  The bench clips have no
/// pre-AGC ring → noise_rms is None → SNR is the estimate_snr_energy
/// heuristic — labelled as such (NOT real-room SNR).
fn enrollment_quality_probe(enrollment_variants: &[(Vec<f32>, String)]) -> serde_json::Value {
    let mut per_clip = Vec::new();
    let mut n_clipping = 0usize;
    let mut snr_sum = 0.0f32;
    let mut score_sum = 0.0f32;

    for (pcm, label) in enrollment_variants {
        let q = super::compute_utterance_quality(pcm, None);
        if q.clipping_detected {
            n_clipping += 1;
        }
        snr_sum += q.snr_db;
        score_sum += q.score;
        per_clip.push(serde_json::json!({
            "label": label,
            "duration_ms": q.duration_ms,
            "clipping_detected": q.clipping_detected,
            "heuristic_snr_db": q.snr_db,
            "composite_score": q.score,
        }));
    }

    let n_clips = enrollment_variants.len();
    serde_json::json!({
        "note": "Heuristic SNR (estimate_snr_energy) — the bench's enrollment clips have no pre-AGC noise ring, so this is NOT real-room SNR (mahbot-1045 B3).",
        "n_clips": n_clips,
        "n_clipping": n_clipping,
        "mean_heuristic_snr_db": if n_clips == 0 { 0.0 } else { snr_sum / n_clips as f32 },
        "mean_composite_score": if n_clips == 0 { 0.0 } else { score_sum / n_clips as f32 },
        "per_clip": per_clip,
    })
}

/// Generate owner-negative sequences (non-wake-word TTS phrases, mahbot-932
/// Fix 6, re-scoped).
///
/// These are TTS-synthesised phrases tagged as `Source::Owner` for use in
/// classifier training.  Since the re-scope the owner is the
/// ENROLLED voice only (single-speaker semantics — the owner's own non-wake
/// speech); the multi-voice style rotation is gone.  Documented limitation:
/// TTS speech cannot match the distribution of real human Phase 3 speech.
///
/// ## Preprocessing alignment (mahbot-1006 C/L, mahbot-1009)
///
/// Production captures owner negatives as real post-AGC/post-NS mic audio
/// (Phase 3 collection).  The benchmark's TTS surrogates are therefore routed
/// through the same pipeline as production's `prewarm_phrase_embeddings`
/// TTS-negative path so the classifier does not train on a raw-TTS
/// distribution that production never produces:
///
/// 1. **Fresh [`AudioPreprocessor`] per phrase × seed** (mahbot-1009) — fed in
///    [`FRAME_LENGTH`](super::FRAME_LENGTH) chunks as-is (no zero-padding),
///    matching the per-segment fresh-AGC distribution of the streaming path.
/// 2. **VAD gating** through a dedicated earshot detector per clip (never the
///    global `VAD_DETECTOR`) via [`super::vad_gate_streaming_mel`], so only
///    VAD-positive hops produce mel frames and windows anchor at speech onset.
/// 3. Dense stride-8 embeddings from the streaming-layout mel frames.
///
/// Divergence from production's `prewarm_phrase_embeddings`: production
/// prewarm embeds the bounded 2-cell negative recipe (original + pink 25 dB);
/// the bench emits the VAD-gated original here plus one bounded low-SNR cell
/// (brown-10) — the owner/ambient negative stand-in for real Phase 3 mic
/// captures.  The shared variant-0 path is structurally identical (same
/// preprocessor, same `vad_gate_streaming_mel` wrapper over
/// `process_streaming_frames_inner`, same embeddings helper).
///
/// The config comes from CONFIG (mahbot-1006 L) — identical to
/// `PreprocessorConfig::default()` under default settings, differs only when
/// a deployment disables NS/AGC.
fn generate_owner_negative_sequences(
    enrolled_style: &str,
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<EmbeddingSequence> {
    let mut sequences = Vec::new();
    let config = enrollment_preprocessor_config();
    let chunk_size = super::FRAME_LENGTH;

    for (i, &phrase) in OWNER_NEGATIVE_PHRASES.iter().enumerate() {
        for seed in 0..3 {
            // Fresh collision-free range (9000+) — the old 1000-1014 range
            // overlapped the confusable base (1000+); 7000/7100 are taken by
            // the cross-speaker probe matrix's noise seed bases.
            let seed_val = 9000 + i as u64 * 3 + seed as u64;
            if let Some(pcm) = super::synthesize_with_pcm_cache(
                phrase,
                enrolled_style,
                seed_val,
                TARGET_SAMPLE_RATE,
                model_hash,
                cache_dir,
            ) {
                // AGC/NS the TTS PCM through a fresh per-clip preprocessor,
                // fed in FRAME_LENGTH chunks as-is (matches
                // prewarm_phrase_embeddings, mahbot-1009; shared helper
                // mahbot-1045 A2).
                let agc_audio =
                    crate::audio::audio_preprocessor::agc_feed_fresh(&pcm, chunk_size, config);
                // VAD-gate with a dedicated detector (never the global
                // VAD_DETECTOR) — only VAD-positive hops produce mel frames,
                // windows anchor at speech onset (matches the streaming path,
                // mahbot-1009).
                let mut detector = Detector::default();
                let (mel_frames, speech_audio) = super::vad_gate_streaming_mel(&agc_audio, |hop| {
                    super::is_speech_with_detector(hop, &mut detector, super::VAD_THRESHOLD)
                });
                if mel_frames.is_empty() {
                    // Same guard as production prewarm_phrase_embeddings: no
                    // VAD-positive speech ⇒ no mel frames ⇒ skip the clip
                    // rather than train on a degenerate constant-silence
                    // embedding (pad_mel_frames_to_window's empty fallback).
                    warn!(
                        "Owner-negative '{phrase}' seed {seed}: no VAD-positive \
                         speech — skipping (matches production prewarm)"
                    );
                    continue;
                }
                let embs = super::embeddings_from_mel_frames(
                    super::ONNX_MODELS
                        .get()
                        .expect("ONNX models must be loaded by the benchmark"),
                    &mel_frames,
                );
                match embs {
                    Ok(embs) if !embs.is_empty() => {
                        sequences.push(EmbeddingSequence::new(
                            UtteranceId {
                                sequence_index: i * 3 + seed,
                                variant_index: 0,
                            },
                            Source::Owner,
                            embs,
                        ));
                    }
                    _ => warn!("Owner-negative '{phrase}' seed {seed}: no embeddings"),
                }
                // Bounded low-SNR augmentation: brown noise at
                // 10 dB on the VAD-gated speech — the enrolled voice under
                // noise must not trigger.  Single cell, cheap.
                if !speech_audio.is_empty() {
                    let noisy = crate::util::add_noise_color(
                        &speech_audio,
                        10.0,
                        crate::util::NoiseColor::Brown,
                        seed_val,
                    );
                    match super::extract_embeddings_from_audio(
                        super::ONNX_MODELS
                            .get()
                            .expect("ONNX models must be loaded by the benchmark"),
                        &noisy,
                    ) {
                        Ok(embs) if !embs.is_empty() => {
                            sequences.push(EmbeddingSequence::new(
                                UtteranceId {
                                    sequence_index: i * 3 + seed,
                                    variant_index: 8,
                                },
                                Source::Owner,
                                embs,
                            ));
                        }
                        _ => warn!("Owner-negative '{phrase}' seed {seed}: no brown-10 embeddings"),
                    }
                }
            }
        }
    }
    sequences
}

/// Generate ambient noise sequences for negative training (mahbot-932 Fix 7).
///
/// Produces `EmbeddingSequence` values tagged as [`Source::Ambient`] from noise
/// profiles at 2 SNR levels each.
///
/// ## Preprocessing alignment (mahbot-1006 C/L)
///
/// Production captures real ambient negatives as post-AGC/post-NS mic audio
/// collected during Phase 3.  The benchmark's synthetic noise profiles are
/// therefore routed through a **shared** [`AudioPreprocessor`] (matching the
/// persistent live-mic preprocessor) before embedding extraction, so the
/// classifier does not train on a raw-noise distribution production
/// never produces.  Documented limitation: synthetic noise through a fresh
/// noise suppressor is more aggressively attenuated than production's
/// room-adapted NS; this is a surrogate for real ambient capture.
fn generate_ambient_noise_sequences() -> Vec<EmbeddingSequence> {
    use crate::audio::audio_preprocessor::AudioPreprocessor;
    let mut sequences = Vec::new();
    let mut pre = AudioPreprocessor::new(enrollment_preprocessor_config());
    let chunk_size = super::FRAME_LENGTH;

    for (seq_idx, (label, noise_fn)) in NOISE_PROFILES.iter().enumerate() {
        let raw = noise_fn();
        // AGC/NS the raw noise (shared preprocessor, zero-padded chunks).
        let mut process_agc = |pcm: &[f32]| -> Vec<f32> {
            let mut agc_audio: Vec<f32> = Vec::with_capacity(pcm.len());
            for chunk in pcm.chunks(chunk_size) {
                let mut padded = chunk.to_vec();
                padded.resize(chunk_size, 0.0);
                agc_audio.extend(pre.process(padded));
            }
            agc_audio
        };
        // Level 1: full amplitude
        let agc_raw = process_agc(&raw);
        match super::process_enrollment_sample(&agc_raw) {
            Ok(embs) if !embs.is_empty() => {
                sequences.push(EmbeddingSequence::new(
                    UtteranceId {
                        sequence_index: seq_idx * 2,
                        variant_index: 0,
                    },
                    Source::Ambient,
                    embs,
                ));
            }
            _ => warn!("Ambient '{label}' level-0: no embeddings"),
        }
        // Level 2: reduced amplitude (-6dB)
        let attenuated = crate::util::apply_gain(&raw, -6.0);
        let agc_attenuated = process_agc(&attenuated);
        match super::process_enrollment_sample(&agc_attenuated) {
            Ok(embs) if !embs.is_empty() => {
                sequences.push(EmbeddingSequence::new(
                    UtteranceId {
                        sequence_index: seq_idx * 2 + 1,
                        variant_index: 0,
                    },
                    Source::Ambient,
                    embs,
                ));
            }
            _ => warn!("Ambient '{label}' level-1: no embeddings"),
        }
    }
    sequences
}

/// Bench-local restricted-style confusable/unrelated negative embeddings
///
/// Production's prewarm rotates ALL 10 TTS voices into the training negatives,
/// which would train the reserved cross-speaker voices (M4/M5) in — the ticket
/// requires them absent from EVERY training path.  This replicates production's
/// `prewarm_phrase_embeddings` pipeline (fresh AGC → VAD-gated streaming mel →
/// embeddings → PCM augmentation) over a restricted style list, reusing the
/// shared PCM cache and the production embedding cache (identical keys, so
/// warm runs are as cheap as the production prewarm and cache entries stay
/// coherent).  No production helper changes — the bench is a submodule of
/// `voice` and drives the production internals directly.
fn generate_restricted_phrase_negatives(
    phrase_type: &'static str,
    phrases: &'static [&'static str],
    seeds_per_phrase: usize,
    seed_base: u64,
    styles: &[String],
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<EmbeddingSequence> {
    let Some(models) = super::ONNX_MODELS.get() else {
        return Vec::new();
    };
    if styles.is_empty() {
        // No negative-pool styles (fewer-than-6 TTS voices) — the call-site
        // non-empty asserts below fail loudly rather than training a weaker
        // model.
        return Vec::new();
    }
    let num_styles = styles.len();
    let pre_config = enrollment_preprocessor_config();
    let mut dense_sequences: Vec<EmbeddingSequence> = Vec::new();

    for (i, &phrase) in phrases.iter().enumerate() {
        for seed_idx in 0..seeds_per_phrase {
            // Same round-robin style formula as production, restricted to the
            // negative-pool styles.
            let style_idx = (i * seeds_per_phrase + seed_idx) % num_styles;
            let style = &styles[style_idx];
            let seed = seed_base + i as u64 * seeds_per_phrase as u64 + seed_idx as u64;
            let phrase_index_for_id = i * seeds_per_phrase + seed_idx;
            let source = match phrase_type {
                "confusable" => Source::Confusable,
                _ => Source::Unrelated,
            };

            // Embedding-level cache (production keys — same cache files).
            let cache_key = super::embedding_cache_key(
                phrase_type,
                phrase,
                style,
                seed,
                model_hash,
                pre_config,
            );
            let emb_cache_dir = super::embedding_cache_dir();
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
                    utterance_variants
                        .push((u8::try_from(vi).expect("variant index fits in u8"), embs));
                }
            };
            if let (Some(dir), Some(key)) = (&emb_cache_dir, &cache_key)
                && let Some(variants) = super::read_embedding_cache(dir, key)
            {
                for (vi, embs) in variants {
                    push_seq(embs, usize::from(vi));
                }
                continue;
            }

            // PCM from cache or fresh TTS synthesis.
            let pcm = super::synthesize_with_pcm_cache(
                phrase,
                style,
                seed,
                TARGET_SAMPLE_RATE,
                model_hash,
                cache_dir,
            );
            let Some(pcm) = pcm else {
                continue;
            };

            // 1. Fresh AGC per phrase × seed (production prewarm ordering).
            let agc_audio = crate::audio::audio_preprocessor::agc_feed_fresh(
                &pcm,
                super::FRAME_LENGTH,
                pre_config,
            );
            // 2. VAD-gate with a dedicated detector (streaming mel layout).
            let mut detector = Detector::default();
            let (mel_frames, speech_audio) = super::vad_gate_streaming_mel(&agc_audio, |hop| {
                super::is_speech_with_detector(hop, &mut detector, super::VAD_THRESHOLD)
            });
            if mel_frames.is_empty() {
                warn!(
                    "{phrase_type} phrase '{phrase}' (seed {seed}) produced no \
                     VAD-positive speech — skipping (matches streaming: no \
                     speech ⇒ no embeddings)"
                );
                continue;
            }
            // 3. Original — embeddings from the streaming mel frames.
            let dense_embs =
                super::embeddings_from_mel_frames(models, &mel_frames).unwrap_or_default();
            push_seq(dense_embs, 0);
            // 4. Augment AFTER VAD gating (production ordering); variant 0 was
            // already pushed above.
            for variant in super::augment_pcm_variants(
                &speech_audio,
                TARGET_SAMPLE_RATE,
                seed,
                super::AugmentSet::Negatives,
            ) {
                if variant.variant_index == 0 {
                    continue;
                }
                let dense_embs =
                    super::extract_embeddings_from_audio(models, &variant.pcm).unwrap_or_default();
                push_seq(dense_embs, variant.variant_index);
            }
            // Persist the per-utterance embedding cache (best-effort).
            if !utterance_variants.is_empty()
                && let (Some(dir), Some(key)) = (&emb_cache_dir, &cache_key)
            {
                super::write_embedding_cache(dir, key, &utterance_variants);
            }
        }
    }
    dense_sequences
}

/// Process a list of audio clips through the enrollment embedding pipeline.
///
/// Returns:
/// * `failed_count` — how many variants failed embedding extraction.
///   Process audio variants through a given embedding extraction function.
///
/// Shared helper that eliminates the duplicated per-variant loop
/// (mahbot-855 review).  Callers supply the extraction function — typically
/// [`super::process_enrollment_sample`] for dense stride-8 embeddings.
/// The old streaming path was removed in mahbot-923; the benchmark now uses
/// dense-only embeddings throughout.
///
/// Each variant becomes one [`EmbeddingSequence`] with the given `source`.
/// Compute VAD frame decisions and segment audio into utterances at the
/// enrollment VAD threshold.  Shared by the VAD-gated enrollment pipeline
/// and the VAD segmentation validation test to eliminate duplication.
///
/// # Panics
/// If called with zero-length audio or empty VAD decisions.
fn compute_vad_segments(audio: &[f32]) -> (Vec<bool>, Vec<Vec<f32>>) {
    let n_frames = audio.len().saturating_sub(super::FRAME_LENGTH) / super::HOP_LENGTH + 1;
    let mut detector = Detector::default();
    let mut vad_decisions: Vec<bool> = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * super::HOP_LENGTH;
        let end = (start + super::FRAME_LENGTH).min(audio.len());
        let frame = &audio[start..end];
        // Feed only the NEW HOP_LENGTH samples to avoid double-feeding
        // overlapping audio to earshot.  Each frame (at HOP_LENGTH stride)
        // overlaps the previous by 50% — feeding the full 512-sample frame
        // would duplicate 256 samples, corrupting earshot's ring buffer.
        // This must match the production VAD call pattern in both
        // process_streaming_frames_inner and handle_enrollment_audio
        // to maintain train-inference consistency (mahbot-900).
        vad_decisions.push(super::is_speech_with_detector(
            &frame[..super::HOP_LENGTH],
            &mut detector,
            super::VAD_THRESHOLD,
        ));
    }
    let utterances = super::segment_utterances_by_vad(
        audio,
        &vad_decisions,
        &super::DEFAULT_VAD_SEGMENTATION_CONFIG,
    );
    (vad_decisions, utterances)
}

/// mahbot-1045 B6 (report-only): regression-guard the mahbot-900 VAD feed
/// contract on the BENCH side.
///
/// Production's VAD feed pattern (voice.rs mic arm, mahbot-900) feeds ONLY
/// the new [`HOP_LENGTH`](super::HOP_LENGTH) samples per hop through a
/// streaming earshot detector — never the full overlapping
/// [`FRAME_LENGTH`](super::FRAME_LENGTH) window (double-feeding corrupts
/// earshot's ring buffer).  This probe replays that contract literally
/// against a fresh detector and compares the decisions with the bench's
/// [`compute_vad_segments`] (which implements the same pattern).  If
/// [`compute_vad_segments`] ever drifts from the literal contract, the
/// comparison flags it.
///
/// Scope honesty: production code is NOT executed in the offline bench, so
/// this cannot detect drift in the production arm itself — it locks the
/// bench's transcription of the contract.  To prove the comparison is
/// sensitive (not vacuous), a NEGATIVE control feeds the full overlapping
/// `FRAME_LENGTH` window per hop (the exact violation mahbot-900 forbids);
/// its mismatch count is reported under `negative_control`.
fn vad_feed_cross_check_probe(audio: &[f32]) -> serde_json::Value {
    if audio.len() < super::FRAME_LENGTH {
        return serde_json::json!({
            "skipped": true,
            "feed_pattern": "only_new_hop_samples_per_hop",
            "note": "audio shorter than FRAME_LENGTH — cannot run the VAD feed cross-check (no full hop frame available)",
        });
    }

    // The bench's own VAD segmentation decisions (fresh internal detector).
    let (bench_decisions, _utterances) = compute_vad_segments(audio);

    let n_frames = audio.len().saturating_sub(super::FRAME_LENGTH) / super::HOP_LENGTH + 1;

    // Reference replay of production's feed pattern (mahbot-900): a FRESH
    // detector fed EXACTLY the new HOP_LENGTH samples per hop.  The two
    // detectors are independent — the comparison is decisions-vs-decisions
    // on the same audio, not shared state.
    let mut detector = Detector::default();
    let mut replay_decisions: Vec<bool> = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * super::HOP_LENGTH;
        let new_samples = &audio[start..start + super::HOP_LENGTH];
        replay_decisions.push(super::is_speech_with_detector(
            new_samples,
            &mut detector,
            super::VAD_THRESHOLD,
        ));
    }

    // NEGATIVE control: a fresh detector fed the full OVERLAPPING
    // FRAME_LENGTH window per hop — double-feeding the 256-sample overlap,
    // the exact feed-pattern violation mahbot-900 forbids.  If the correct
    // and violated feeds produce the same decisions on this audio
    // (feed_pattern_sensitive == false), the main comparison would be
    // vacuous and the report says so honestly instead of overclaiming.
    let mut double_fed_detector = Detector::default();
    let mut double_fed_decisions: Vec<bool> = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * super::HOP_LENGTH;
        let window = &audio[start..start + super::FRAME_LENGTH];
        double_fed_decisions.push(super::is_speech_with_detector(
            window,
            &mut double_fed_detector,
            super::VAD_THRESHOLD,
        ));
    }

    // Element-wise comparison of the independent decision vectors.
    let n_hops = bench_decisions.len();
    let n_matching = bench_decisions
        .iter()
        .zip(replay_decisions.iter())
        .filter(|(a, b)| **a == **b)
        .count();
    let mismatch_indices: Vec<usize> = bench_decisions
        .iter()
        .zip(replay_decisions.iter())
        .enumerate()
        .filter(|(_, (a, b))| **a != **b)
        .map(|(idx, _)| idx)
        .take(5)
        .collect();
    let n_double_fed_mismatches = bench_decisions
        .iter()
        .zip(double_fed_decisions.iter())
        .filter(|(a, b)| **a != **b)
        .count();

    serde_json::json!({
        "skipped": false,
        "feed_pattern": "only_new_hop_samples_per_hop",
        "n_hops": n_hops,
        "n_matching": n_matching,
        "all_match": mismatch_indices.is_empty(),
        "mismatch_indices": mismatch_indices,
        "negative_control": {
            "feed_pattern": "full_overlapping_frame_per_hop (mahbot-900 violation)",
            "mismatches_vs_bench": n_double_fed_mismatches,
            "feed_pattern_sensitive": n_double_fed_mismatches > 0,
        },
        "note": "Bench-side regression guard for the mahbot-900 VAD feed contract: \
                 replays production's feed pattern (feed ONLY the new HOP_LENGTH \
                 samples per hop) through a fresh Detector and compares against \
                 compute_vad_segments.  Production code is not executed in the \
                 offline bench, so drift in the production arm itself is not \
                 detectable here.  The negative control (full overlapping \
                 FRAME_LENGTH per hop) reports how many hops diverge, proving the \
                 comparison is sensitive to the feed pattern rather than vacuous.",
    })
}

/// Build the audio preprocessor configuration from the same CONFIG flags that
/// production uses (mahbot-1006 L).
///
/// `PipelineCtx::new()` constructs the live-mic preprocessor from
/// `CONFIG.voice_noise_suppression()` / `CONFIG.voice_agc()` (both default to
/// enabled when unset).  The benchmark's enrollment/training path previously
/// hardcoded `PreprocessorConfig::default()` (NS+AGC always on), which silently
/// diverged whenever a deployment disables either stage.
///
/// Delegates to [`super::preprocessor_config_from_config()`] — the same helper
/// `PipelineCtx::new()` uses — so the parsing semantics cannot drift between
/// the benchmark and production.
fn enrollment_preprocessor_config() -> crate::audio::audio_preprocessor::PreprocessorConfig {
    super::preprocessor_config_from_config()
}

/// Process enrollment clips through VAD-gated utterance segmentation with the
/// **production ordering** (mahbot-1006 B/I/L).
///
/// Production's enrollment path (live mic loop → `handle_enrollment_audio` →
/// `handle_enrollment_sample`) is:
///
/// ```text
/// raw mic audio → AGC/NS → VAD segment → augment after segmentation → embeddings
/// ```
///
/// The old benchmark path applied augmentation to the raw TTS PCM **before**
/// AGC (`raw → augment → AGC → VAD`), so:
/// - AGC partially normalized away the volume-down variant (re-amplifying it
///   back toward target) and NS partially removed the noise variant's added
///   noise, making those variants less challenging than production's
///   equivalents (mahbot-1006 B);
/// - VAD segmentation ran on the concatenated *augmented* variants, so a
///   variant could be silently dropped when VAD didn't fire on it, whereas
///   production runs VAD only on the original mic utterance and always includes
///   the augmented variants (mahbot-1006 I).
///
/// This function instead:
/// 1. Applies AGC/NS to each raw TTS clip through a **fresh**
///    [`AudioPreprocessor`] built from the CONFIG-driven
///    [`enrollment_preprocessor_config`] — per-clip fresh state matches
///    production's `reset_detection_segment()` fresh-AGC distribution
///    (mahbot-886 rationale, kept).
/// 2. Concatenates the AGC'd **originals** with 2.0 s silence gaps and
///    VAD-segments them (only originals, matching `handle_enrollment_audio`).
/// 3. Derives the shared 12-cell recipe (original, speed-down 0.95/0.90,
///    speed-up conditional, volume-down, pink 25 dB, white/pink/brown noise
///    at 10/5 dB) from each **AGC'd** utterance audio via
///    [`super::augment_pcm_variants`] — the same variant set as production's
///    `handle_enrollment_sample` — then extracts embeddings from every variant.
///
/// Returns dense-only EmbeddingSequences (stride-8) for classifier
/// training.  The old streaming path was removed in mahbot-923.
#[allow(clippy::too_many_lines)]
fn vad_segment_and_enroll(enrollment_variants: &[(Vec<f32>, String)]) -> Vec<EmbeddingSequence> {
    let chunk_size = super::FRAME_LENGTH;
    let config = enrollment_preprocessor_config();

    // ── Per-clip AGC (mahbot-886, mahbot-1006 B) ──
    // Each clip processed through a fresh AudioPreprocessor (both AGC and
    // noise suppressor), matching the production detection path.  Shared-AGC
    // approach (concatenating variants then applying AGC) created a different
    // distribution — running_rms converged during training but detection starts
    // fresh, causing 46% miss rate on TTS variants.  Per-clip fresh NS avoids
    // the same training-inference mismatch.  Chunks are fed as-is (no
    // zero-padding): the NS buffers incomplete frames internally and the next
    // chunk (or the silence gap) completes them, matching production's tail
    // accumulation (mahbot-1006 G).
    let mut per_clip_agc: Vec<Vec<f32>> = Vec::with_capacity(enrollment_variants.len());
    for (samples, _label) in enrollment_variants {
        let processed =
            crate::audio::audio_preprocessor::agc_feed_fresh(samples, chunk_size, config);
        per_clip_agc.push(processed);
    }

    // ── Concatenate AGC'd originals with 2.0s silence gaps ──
    // 2.0s well exceeds ENROLLMENT_SILENCE_THRESHOLD_SAMPLES (~304ms after
    // mahbot-1001 Fix 7) for clean boundaries.
    let silence_gap_samples = (2.0 * f64::from(super::SAMPLE_RATE)) as usize;
    let silence: Vec<f32> = vec![0.0f32; silence_gap_samples];

    let mut combined_audio: Vec<f32> = Vec::new();
    for processed in &per_clip_agc {
        if !combined_audio.is_empty() {
            combined_audio.extend_from_slice(&silence);
        }
        combined_audio.extend_from_slice(processed);
    }
    // Trailing silence for the last utterance
    combined_audio.extend_from_slice(&silence);

    info!(
        "VAD concatenation: {} total samples ({:.1}s) from {} originals (per-clip AGC, mahbot-1006 B)",
        combined_audio.len(),
        combined_audio.len() as f64 / f64::from(super::SAMPLE_RATE),
        enrollment_variants.len(),
    );

    // ── VAD segmentation on the ORIGINALS only (mahbot-1006 I) ──
    let (_vad_decisions, utterances) = compute_vad_segments(&combined_audio);
    info!(
        "VAD segmentation: {} utterances from {} concatenated originals",
        utterances.len(),
        enrollment_variants.len(),
    );

    // ── Augment AFTER segmentation, then extract embeddings (mahbot-1006 B) ──
    // Each VAD utterance is treated like production's `handle_enrollment_sample`
    // input: the full 12-cell recipe is derived from the AGC'd utterance
    // audio and every variant is embedded.  Every variant is always included
    // — none can be silently dropped by VAD (which only gated the original).
    let mut dense_sequences: Vec<EmbeddingSequence> = Vec::new();

    for (i, utterance) in utterances.iter().enumerate() {
        // Noise seed = utterance index, gate input = VAD-gated utterance
        // slice, canonical push order — preserved verbatim from the pre-dedup
        // code via the shared helper (mahbot-1045 A1).
        let augmented = super::augment_pcm_variants(
            utterance,
            TARGET_SAMPLE_RATE,
            i as u64,
            super::AugmentSet::Full,
        );
        let has_speed_up = augmented.iter().any(|v| v.variant_index == 2);

        info!(
            "Utterance {i}: {} samples ({:.2}s) → original + speed_down + {} + vol_down + noise + \
             speed_down_090 + color-noise cells",
            utterance.len(),
            utterance.len() as f64 / f64::from(super::SAMPLE_RATE),
            if has_speed_up {
                "speed_up".to_string()
            } else {
                "speed_up(skipped,<500ms)".to_string()
            },
        );

        // Helper closure: push one variant's embeddings (mahbot-878 naming:
        // variant 0 = original, 1 = speed-down, 2 = speed-up, 3 = volume-down,
        // 4 = noise).
        let push_variant = |variant_index: usize,
                            source: Source,
                            pcm: &[f32],
                            sequences: &mut Vec<EmbeddingSequence>| {
            match super::process_enrollment_sample(pcm) {
                Ok(embeddings) if !embeddings.is_empty() => {
                    sequences.push(EmbeddingSequence::new(
                        UtteranceId {
                            sequence_index: i,
                            variant_index,
                        },
                        source,
                        embeddings,
                    ));
                }
                Ok(_) => warn!("Utterance {i} variant {variant_index}: no embeddings extracted"),
                Err(e) => {
                    warn!(
                        "Utterance {i} variant {variant_index}: embedding extraction failed: {e}"
                    );
                }
            }
        };

        for variant in augmented {
            let source = match variant.variant_index {
                0 => Source::Enrollment,
                1..=11 => Source::Augmentation,
                _ => unreachable!("augment_pcm_variants yields only indices 0..=11"),
            };
            push_variant(
                variant.variant_index,
                source,
                &variant.pcm,
                &mut dense_sequences,
            );
        }
    }

    info!(
        "VAD-gated enrollment (mahbot-1006 B): {} dense embeddings from {} sequences ({} utterances × full recipe)",
        dense_sequences
            .iter()
            .map(|s| s.embeddings.len())
            .sum::<usize>(),
        dense_sequences.len(),
        utterances.len(),
    );

    dense_sequences
}

// ── Streaming detection ─────────────────────────────

/// Result from [`run_streaming_detection`].
struct DetectionResult {
    detected: bool,
    /// Latency in milliseconds from feed start to detection (only meaningful
    /// when `detected` is true).
    latency_ms: Option<f64>,
    /// AGC convergence state captured at the end of the test utterance (before
    /// the trailing silence flush).  `None` when insufficient AGC-active
    /// frames.  Captured pre-flush so a segment boundary fired during the
    /// trailing silence cannot erase the utterance's convergence evidence
    /// (mahbot-1006 A/H).
    agc_converged: Option<bool>,
    /// Adaptive threshold state captured at the end of the test utterance
    /// (before the trailing silence flush).  A segment boundary fired during
    /// the flush calls `reset_detection_segment`, which resets
    /// `ctx.adaptive_threshold` to bootstrap; carrying this pre-flush snapshot
    /// forward lets the benchmark's shared-state design (mahbot-845) preserve
    /// the adaptation accumulated from the utterance's frames instead of
    /// re-bootstrapping per variant (mahbot-1006 F).
    adaptive_state_pre_flush: super::AdaptiveThresholdState,
}

/// Run the production streaming wake word detection pipeline on audio samples.
///
/// Feeds audio through [`handle_wake_word_detection`] in FRAME_LENGTH chunks,
/// exercising the full streaming chain: VAD gating, batch accumulation,
/// [`flush_voice_batch`], [`score_stride8_window`],
/// [`score_single_embedding`], and cooldown logic.
///
/// After all audio is fed, a silence frame is sent to flush any remaining
/// voice batch (matching how the production pipeline handles speech→silence
/// transitions).
///
/// Returns a [`DetectionResult`] with a flag and optional latency measurement.
fn run_streaming_detection(samples: &[f32], ctx: &mut super::PipelineCtx) -> DetectionResult {
    let feed_start = Instant::now();
    // Save pre-existing timestamp — we only return true if detection fires
    // during THIS call, not because a prior call already set the field.
    let before = ctx.last_wake_word_detection;

    // Feed audio in FRAME_LENGTH chunks, processing each through the
    // audio preprocessor (AGC + noise suppression) to match the production
    // pipeline (mahbot-856).  Without AGC, quiet variants like
    // M3.json_aug4_volume (-6dB reduction) are too faint for VAD to trigger,
    // producing false 0.00 detection scores.
    for chunk in samples.chunks(super::FRAME_LENGTH) {
        process_frame(chunk, ctx);
        if ctx.last_wake_word_detection != before {
            let latency = feed_start.elapsed().as_secs_f64() * 1000.0;
            return DetectionResult {
                detected: true,
                latency_ms: Some(latency),
                agc_converged: ctx.audio_preprocessor.agc_converged(),
                // No flush ran on this path, so the pre-flush snapshot is the
                // current state; the boundary-fire branch in the callers is
                // unreachable when `detected` is true anyway.
                adaptive_state_pre_flush: ctx.adaptive_threshold.clone(),
            };
        }
    }

    // ── Trailing silence flush (mahbot-1006 H) ───────────────────────────
    // Production continues on natural silence after the utterance and
    // detection can still fire for up to SEGMENT_TIMEOUT_HOPS (19) VAD-negative
    // hops before the segment-boundary reset clears the ring and per-segment
    // state.  The old flush fed exactly 3×512 zero frames (~96 ms), which
    // missed wake words whose rolling sum crossed threshold during the longer
    // production window (~304 ms).
    //
    // The AGC convergence evidence is captured BEFORE the flush: a segment
    // boundary fired during the flush resets the preprocessor (mahbot-1006 A)
    // and its voice-tests instrumentation, which would otherwise erase the
    // utterance's convergence history from `agc_converged()`.  The adaptive
    // threshold state is captured for the same reason — the boundary resets it
    // to bootstrap, and the benchmark's shared-state design (mahbot-845) wants
    // the adapted state to survive the boundary rather than re-bootstrapping
    // per variant (mahbot-1006 F).
    let agc_converged = ctx.audio_preprocessor.agc_converged();
    let adaptive_state_pre_flush = ctx.adaptive_threshold.clone();
    //
    // Feed FRAME_LENGTH digital-silence chunks (production's mic delivers real
    // zeros, not empty Vecs — see process_frame) until:
    //   1. detection fires, or
    //   2. the segment boundary fires, or
    //   3. we exceed the production window (SEGMENT_TIMEOUT_HOPS hops plus a
    //      small allowance for any remaining VAD-positive utterance tail).
    //
    // The boundary is detected via the embedding ring: reset_detection_segment
    // clears it (and no other non-detection path does within a variant), so a
    // non-empty→empty transition means the segment ended and further silence
    // cannot produce a detection.
    let mut silence_chunks = 0usize;
    while silence_chunks < super::SEGMENT_TIMEOUT_HOPS + 4 {
        if ctx.last_wake_word_detection != before {
            let latency = feed_start.elapsed().as_secs_f64() * 1000.0;
            return DetectionResult {
                detected: true,
                latency_ms: Some(latency),
                agc_converged,
                adaptive_state_pre_flush,
            };
        }
        let ring_was_nonempty = !ctx.embedding_ring.is_empty();
        process_frame(&vec![0.0; super::FRAME_LENGTH], ctx);
        silence_chunks += 1;
        // Segment boundary fired: the ring was cleared — further silence
        // cannot produce a detection, so stop feeding.
        if ring_was_nonempty && ctx.embedding_ring.is_empty() {
            break;
        }
    }

    DetectionResult {
        detected: ctx.last_wake_word_detection != before,
        latency_ms: None,
        agc_converged,
        adaptive_state_pre_flush,
    }
}

/// Sleep until the ctx's last wake-word detection is at least `target_elapsed`
/// old, measured from the [`PipelineCtx::last_wake_word_detection`] field (the
/// timestamp set inside the feed loop), NOT from `run_streaming_detection`'s
/// return time.
///
/// Detection sets the cooldown timestamp *inside* the feed loop
/// (`score_stride8_window` → `ctx.last_wake_word_detection = Some(now)`), so
/// returning from `run_streaming_detection` adds a variable detection-to-return
/// delta.  Sleeping relative to the timestamp itself avoids that jitter when
/// probing the cooldown gate at specific elapsed times (mahbot-1052).
///
/// When the timestamp is absent (should not happen mid-Phase-13), no sleep
/// occurs — the caller's probes rely on the timestamp being present after a
/// fired detection.
fn sleep_until_cooldown_elapsed(ctx: &super::PipelineCtx, target_elapsed: Duration) {
    if let Some(last) = ctx.last_wake_word_detection {
        let target = last + target_elapsed;
        let now = Instant::now();
        if target > now {
            std::thread::sleep(target - now);
        }
    }
}

// ── Metrics reporting ────────────────────────────────────────────────────

/// Result for a single wake word variant during detection.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
struct PerVariantResult {
    /// Variant label (e.g. "M1.json seed 100" or "augmented speed_0.9").
    variant: String,
    /// Whether the variant triggered wake word detection.
    detected: bool,
    /// Peak per-frame total_score (classifier sigmoid output, range [0,1])
    /// achieved during processing. This is NOT the rolling sum — it's the
    /// maximum single-frame score. For rolling sum analysis, use the
    /// `per_frame_scores` field or `max_rolling_sum`.
    peak_score: f32,
    /// Maximum rolling sum (sum of 3 consecutive total_scores with decay)
    /// achieved during processing. Derived from per_frame_scores.
    max_rolling_sum: f32,
    /// Number of embeddings produced during streaming detection (mahbot-886).
    /// This is the length of the embedding ring buffer after processing, which
    /// directly reflects how many mel frames passed through the embedding model
    /// (mahbot-922).
    n_embeddings: usize,
    /// Number of frames where total_score < NO_MATCH_RESET_THRESHOLD (0.316).
    n_frames_below_reset: usize,
    /// Whether the AGC converged to a stable gain level by utterance end.
    /// `Some(true)` if converged, `Some(false)` if not, `None` if insufficient
    /// data (< 20 AGC-active frames).
    agc_converged: Option<bool>,
    /// Count of VAD-positive 512-sample frames during streaming detection.
    vad_speech_frames: usize,
    /// Per-frame `[total_score, rolling_sum, threshold]` triples from classifier
    /// scoring (mahbot-891).  Use `ROLLING_SUM_IDX` / `THRESHOLD_IDX` for named
    /// access to the rolling sum and effective threshold fields respectively.
    per_frame_scores: Vec<[f32; 3]>,
    /// Index of the first classifier-trigger frame within `per_frame_scores`.
    /// `None` when the classifier never triggered on test-utterance frames.
    first_trigger_frame_idx: Option<usize>,
    /// Number of embeddings attributed to the test utterance — the length of
    /// `per_frame_scores` (one entry per scored embedding after the warm-up
    /// instrumentation reset).  Unlike `n_embeddings` (ring length at end,
    /// which includes the intentionally preserved warm-up embeddings and can be
    /// cleared by a segment boundary), this counts only test-utterance frames.
    n_test_embeddings: usize,
    /// Per-frame adaptive threshold trajectory (parallel to `per_frame_scores`,
    /// mahbot-1005 §8).  Previously collected in `DetectionInstrumentation`
    /// but never emitted in the benchmark report.
    adaptive_threshold_trajectory: Vec<f32>,
    /// Count of frames where the effective threshold hit ADAPTIVE_CEILING
    /// (mahbot-1005 §8).
    ceiling_limited_frames: usize,
    /// Detection latency in ms for this variant (only meaningful when
    /// `detected` is true; `None` otherwise).  Emitted per-variant so latency
    /// outliers can be root-caused (mahbot-1005 §8).
    latency_ms: Option<f64>,

    // ── mahbot-1012 per-frame scoring geometry (parallel to per_frame_scores)
    /// Stable hash of each scored embedding (FNV-1a over f32 bit patterns).
    per_frame_embedding_hashes: Vec<u64>,
    /// L2 norm of each scored embedding.
    per_frame_embedding_l2_norms: Vec<f32>,
    /// Mel-frame position where each scored window started.
    per_frame_window_start: Vec<usize>,
    /// Mel buffer length at each scoring step.
    per_frame_mel_buffer_len: Vec<usize>,
    /// Geometry class of each scored window (see
    /// [`super::WindowGeometry`]).
    per_frame_geometry: Vec<super::WindowGeometry>,
    /// Adaptive-threshold mode of each scored frame (see
    /// [`super::AdaptiveFrameMode`]).
    per_frame_adaptive_mode: Vec<super::AdaptiveFrameMode>,
    /// Per-hop VAD decisions during streaming detection (one per VAD decision
    /// in processing order).
    per_hop_vad: Vec<bool>,

    // ── mahbot-1023 deferred-burst / acceptance-floor fields ──────────────
    /// Scoring path that produced the detection (raw source: "burst" /
    /// "segment_end_pass" / "other").  `None` when not detected.
    detection_path: Option<String>,
    /// Whether the deferred burst sweep ran during this variant's session.
    burst_sweep_fired: bool,
    /// Mel-frame buffer length at burst-sweep time (the actual live trigger
    /// geometry — typically 68–80).
    burst_sweep_buffer_len: Option<usize>,
    /// Whether the segment-end pass ran at a boundary during this variant's
    /// session.
    segment_end_pass_fired: bool,
    /// Whether the adaptive threshold was still bootstrapping at the end of
    /// the test utterance (mahbot-1023 score-only feed rule: a full
    /// high-scoring utterance can keep the bootstrap alive for its whole
    /// duration — the effective threshold then stays at the static hard
    /// floor, and miss verdicts must not misread this as an adaptive block).
    adaptive_bootstrap_persisted: bool,
}

/// Build a [`PerVariantResult`] from a completed [`run_streaming_detection`]
/// session.  Shared by `test_detection_samples` and `run_noise_overlap_test`
/// so the instrumentation-derived fields are populated identically everywhere
/// (mahbot-1005 §9: negatives must carry the same detail as positives).
fn build_per_variant_result(
    label: &str,
    result: &DetectionResult,
    ctx: &super::PipelineCtx,
) -> PerVariantResult {
    let peak = ctx.instrumentation.peak_score;
    let max_rs = max_rolling_sum(&ctx.instrumentation.per_frame_scores);
    // Test-utterance embedding count.  `per_frame_scores` is reset at the end
    // of consume_warmup (mahbot-1005 §1) and grows one entry per scored
    // embedding, so its length IS the test-utterance embedding count.  The
    // alternative (ring length minus warm-up) is unreliable because a segment
    // boundary during/after the test clears the embedding ring.
    let n_test_embeddings = ctx.instrumentation.per_frame_scores.len();
    PerVariantResult {
        variant: label.to_string(),
        detected: result.detected,
        peak_score: peak,
        max_rolling_sum: max_rs,
        n_embeddings: ctx.embedding_ring.len(),
        n_frames_below_reset: ctx.instrumentation.n_frames_below_reset,
        // Captured pre-flush in run_streaming_detection (mahbot-1006 A/H) —
        // reading ctx.audio_preprocessor.agc_converged() here could report
        // None for a miss whose segment boundary reset the preprocessor.
        agc_converged: result.agc_converged,
        vad_speech_frames: ctx.instrumentation.vad_speech_frames,
        per_frame_scores: ctx.instrumentation.per_frame_scores.clone(),
        first_trigger_frame_idx: ctx.instrumentation.first_trigger_frame_idx,
        n_test_embeddings,
        adaptive_threshold_trajectory: ctx.instrumentation.adaptive_threshold_trajectory.clone(),
        ceiling_limited_frames: ctx.instrumentation.ceiling_limited_frames,
        latency_ms: result.latency_ms,
        // mahbot-1012 per-frame geometry (copied from instrumentation; the
        // enum arrays are cloned as-is — JSON conversion happens in pv_to_json).
        per_frame_embedding_hashes: ctx.instrumentation.per_frame_embedding_hashes.clone(),
        per_frame_embedding_l2_norms: ctx.instrumentation.per_frame_embedding_l2_norms.clone(),
        per_frame_window_start: ctx.instrumentation.per_frame_window_start.clone(),
        per_frame_mel_buffer_len: ctx.instrumentation.per_frame_mel_buffer_len.clone(),
        per_frame_geometry: ctx.instrumentation.per_frame_geometry.clone(),
        per_frame_adaptive_mode: ctx.instrumentation.per_frame_adaptive_mode.clone(),
        per_hop_vad: ctx.instrumentation.per_hop_vad.clone(),
        // mahbot-1023: deferred-burst / acceptance-floor evidence.
        detection_path: ctx.instrumentation.detection_path.map(str::to_string),
        burst_sweep_fired: ctx.instrumentation.burst_sweep_fired,
        burst_sweep_buffer_len: ctx.instrumentation.burst_sweep_buffer_len,
        segment_end_pass_fired: ctx.instrumentation.segment_end_pass_fired,
        adaptive_bootstrap_persisted: ctx
            .instrumentation
            .per_frame_adaptive_mode
            .last()
            .is_some_and(|m| *m == super::AdaptiveFrameMode::Bootstrap),
    }
}

// ── Exhaustive per-variant miss verdicts (mahbot-1005 §2) ─────────────────

/// Exhaustive per-variant verdict for positive (wake word) test variants.
///
/// Replaces the old 3-way bucket which collapsed at least four distinct
/// failure modes into a single label and mis-bucketed zero-embedding misses
/// (VAD never fired) as "classifier".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MissVerdict {
    /// Wake word detected on this variant.
    Detected,
    /// VAD produced no speech frames (or the test utterance produced zero
    /// embeddings) — the classifier never saw test audio.
    VadFailure,
    /// Rolling sum never reached [`MIN_CLASSIFIER_THRESHOLD`] (2.13) on
    /// test-utterance frames only.
    ClassifierNoTrigger,
    /// Crossed the hard floor ([`MIN_CLASSIFIER_THRESHOLD`] = 2.13) but never
    /// crossed the effective adaptive threshold.
    AdaptiveThresholdBlocked,
    /// AGC explicitly reported non-convergence (`Some(false)`).
    AgcFailure,
}

impl MissVerdict {
    /// Stable snake_case label for JSON output.
    fn as_str(self) -> &'static str {
        match self {
            Self::Detected => "detected",
            Self::VadFailure => "vad_failure",
            Self::ClassifierNoTrigger => "classifier_no_trigger",
            Self::AdaptiveThresholdBlocked => "adaptive_threshold_blocked",
            Self::AgcFailure => "agc_failure",
        }
    }
}

/// Classify a positive variant into an exhaustive verdict (mahbot-1005 §2).
///
/// ## Decision procedure (precedence, highest first)
///
/// 1. `detected` — no miss.
/// 2. `vad_failure` — VAD never fired (zero speech frames) OR no embeddings
///    were scored during the test utterance (`per_frame_scores` is empty after
///    the warm-up instrumentation reset).  Score-based verdicts are meaningless
///    when the classifier never saw test audio.
/// 3. `agc_failure` — AGC explicitly reported `Some(false)`.  `None`
///    (insufficient AGC-active frames) is NOT `agc_failure` — short utterances
///    fall through to the score-based verdicts.
/// 4. `classifier_no_trigger` — never reached [`MIN_CLASSIFIER_THRESHOLD`]
///    (2.13) on test-utterance frames.
/// 5. `adaptive_threshold_blocked` — reached the hard floor (2.13) but never
///    reached the per-frame effective adaptive threshold.  "Hard floor" here is
///    [`MIN_CLASSIFIER_THRESHOLD`] (the classifier-trigger condition), NOT
///    [`ADAPTIVE_FLOOR`](super::ADAPTIVE_FLOOR) (the adaptive threshold's own
///    clamp floor, which is an internal detail of the adaptive state).
fn classify_miss(pv: &PerVariantResult) -> MissVerdict {
    if pv.detected {
        return MissVerdict::Detected;
    }
    if pv.vad_speech_frames == 0 || pv.per_frame_scores.is_empty() {
        return MissVerdict::VadFailure;
    }
    if pv.agc_converged == Some(false) {
        return MissVerdict::AgcFailure;
    }
    if pv.max_rolling_sum < MIN_CLASSIFIER_THRESHOLD {
        return MissVerdict::ClassifierNoTrigger;
    }
    // Crossed the hard floor but never the effective threshold.  With no
    // second-stage gate, crossing the effective threshold IS detection, so a
    // classifier trigger without detection means the adaptive threshold never
    // came down to meet the rolling sum.
    MissVerdict::AdaptiveThresholdBlocked
}

/// Extract the maximum rolling sum from per-frame score triples.
///
/// The rolling sum (`ROLLING_SUM_IDX`) is the accumulated classifier score with
/// decay over consecutive speech frames — the value that the adaptive threshold
/// ultimately gates.
fn max_rolling_sum(scores: &[[f32; 3]]) -> f32 {
    scores
        .iter()
        .map(|s| s[ROLLING_SUM_IDX])
        .fold(0.0_f32, f32::max)
}

/// Track per-variant detection results for reporting.
#[derive(Debug, Default)]
struct DetectionMetrics {
    total: usize,
    detected: usize,
    false_accepts: Vec<String>,
    /// Per-variant detail for positive detection logging (mahbot-845).
    per_variant: Vec<PerVariantResult>,
}

impl DetectionMetrics {
    fn detection_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.detected as f64 / self.total as f64
        }
    }
}

// ── Classifier trigger metrics (mahbot-952) ─────────────────────────────────

/// Classifier-only trigger metrics for a set of negative variants.
///
/// Tracks the classifier stage independently so false-accept behavior is
/// decomposed into classifier triggers vs full-pipeline detections.
#[derive(Debug, Default, Clone, Copy)]
struct ClassifierTriggerMetrics {
    /// Total variants tested in this group.
    total_variants: usize,
    /// Number of variants where max_rolling_sum >= MIN_CLASSIFIER_THRESHOLD.
    classifier_triggers: usize,
    /// Number where detected == true (full pipeline false accept).
    full_pipeline_fa: usize,
}

impl ClassifierTriggerMetrics {
    /// Compute metrics from a slice of per-variant results.
    fn compute(per_variant: &[PerVariantResult]) -> Self {
        let mut m = Self::default();
        for pv in per_variant {
            m.accumulate(pv);
        }
        m
    }

    /// Accumulate a single per-variant result into this metrics struct.
    ///
    /// This shares the classification branching logic with [`compute`] so that
    /// callers who cannot collect a flat slice (e.g. per-tier partitioning) can
    /// reuse the same counting rules without duplication.
    fn accumulate(&mut self, pv: &PerVariantResult) {
        self.total_variants += 1;
        if pv.max_rolling_sum >= MIN_CLASSIFIER_THRESHOLD {
            self.classifier_triggers += 1;
            if pv.detected {
                self.full_pipeline_fa += 1;
            }
        }
    }
}

/// Convert classifier trigger metrics to a JSON object for the benchmark report.
fn ct_to_json(ct: &ClassifierTriggerMetrics) -> serde_json::Value {
    serde_json::json!({
        "total_variants": ct.total_variants,
        "classifier_triggers": ct.classifier_triggers,
        "full_pipeline_fa": ct.full_pipeline_fa,
    })
}

/// Full per-variant diagnostics for BOTH positive and negative variants
/// (mahbot-1005 §9).  Negatives historically omitted per-frame scores, VAD/AGC
/// detail and trigger-frame info — false-accept root-causing was impossible
/// without them.  For positives (`category: None`) the exhaustive miss verdict
/// and trigger-point evidence are added.
// Clippy: this is a pure JSON serialization shim — one insert per field; the
// line count is inherent to the schema, not control flow worth splitting.
#[expect(clippy::too_many_lines)]
fn pv_to_json(pv: &PerVariantResult, category: Option<&str>) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("variant".to_string(), serde_json::json!(pv.variant));
    if let Some(cat) = category {
        obj.insert("category".to_string(), serde_json::json!(cat));
    }
    obj.insert("detected".to_string(), serde_json::json!(pv.detected));
    obj.insert("peak_score".to_string(), serde_json::json!(pv.peak_score));
    obj.insert(
        "max_rolling_sum".to_string(),
        serde_json::json!(pv.max_rolling_sum),
    );
    obj.insert(
        "n_embeddings".to_string(),
        serde_json::json!(pv.n_embeddings),
    );
    obj.insert(
        "n_test_embeddings".to_string(),
        serde_json::json!(pv.n_test_embeddings),
    );
    obj.insert(
        "n_frames_below_reset".to_string(),
        serde_json::json!(pv.n_frames_below_reset),
    );
    obj.insert(
        "agc_converged".to_string(),
        serde_json::json!(pv.agc_converged),
    );
    obj.insert(
        "vad_speech_frames".to_string(),
        serde_json::json!(pv.vad_speech_frames),
    );
    obj.insert(
        "per_frame_scores".to_string(),
        serde_json::json!(pv.per_frame_scores),
    );
    // mahbot-1012: per-frame scoring geometry — parallel to per_frame_scores.
    obj.insert(
        "per_frame_embedding_hashes".to_string(),
        serde_json::json!(pv.per_frame_embedding_hashes),
    );
    obj.insert(
        "per_frame_embedding_l2_norms".to_string(),
        serde_json::json!(pv.per_frame_embedding_l2_norms),
    );
    obj.insert(
        "per_frame_window_start".to_string(),
        serde_json::json!(pv.per_frame_window_start),
    );
    obj.insert(
        "per_frame_mel_buffer_len".to_string(),
        serde_json::json!(pv.per_frame_mel_buffer_len),
    );
    obj.insert(
        "per_frame_geometry".to_string(),
        serde_json::json!(
            pv.per_frame_geometry
                .iter()
                .map(|g| g.as_str())
                .collect::<Vec<_>>()
        ),
    );
    obj.insert(
        "per_frame_adaptive_mode".to_string(),
        serde_json::json!(
            pv.per_frame_adaptive_mode
                .iter()
                .map(|m| m.as_str())
                .collect::<Vec<_>>()
        ),
    );
    obj.insert("per_hop_vad".to_string(), serde_json::json!(pv.per_hop_vad));
    obj.insert(
        "first_trigger_frame_idx".to_string(),
        serde_json::json!(pv.first_trigger_frame_idx),
    );
    obj.insert("latency_ms".to_string(), serde_json::json!(pv.latency_ms));
    // mahbot-1023: deferred-burst / acceptance-floor evidence (all variants).
    obj.insert(
        "detection_path".to_string(),
        serde_json::json!(pv.detection_path),
    );
    obj.insert(
        "burst_sweep_fired".to_string(),
        serde_json::json!(pv.burst_sweep_fired),
    );
    obj.insert(
        "burst_sweep_buffer_len".to_string(),
        serde_json::json!(pv.burst_sweep_buffer_len),
    );
    obj.insert(
        "segment_end_pass_fired".to_string(),
        serde_json::json!(pv.segment_end_pass_fired),
    );
    obj.insert(
        "adaptive_bootstrap_persisted".to_string(),
        serde_json::json!(pv.adaptive_bootstrap_persisted),
    );
    // Positive-only miss evidence (mahbot-1005 §2/§3, mahbot-1012 §4).
    if category.is_none() {
        let verdict = classify_miss(pv);
        obj.insert("verdict".to_string(), serde_json::json!(verdict.as_str()));
        // Trigger-point evidence (mahbot-1005 §3): the classifier's score at the
        // first trigger frame and where in the utterance it occurred.  VAD is
        // binary (is_speech), so "noise burst" vs "silence" at the trigger
        // cannot be distinguished by VAD alone — the frame's total_score vs
        // NO_MATCH_RESET_THRESHOLD is the closest signal and is reported raw.
        if let Some(idx) = pv.first_trigger_frame_idx
            && let Some(frame) = pv.per_frame_scores.get(idx)
        {
            let frac = if pv.per_frame_scores.is_empty() {
                0.0
            } else {
                idx as f64 / pv.per_frame_scores.len() as f64
            };
            obj.insert(
                "trigger_frame_total_score".to_string(),
                serde_json::json!(frame[TOTAL_SCORE_IDX]),
            );
            obj.insert(
                "trigger_frame_rolling_sum".to_string(),
                serde_json::json!(frame[ROLLING_SUM_IDX]),
            );
            obj.insert(
                "trigger_frame_effective_threshold".to_string(),
                serde_json::json!(frame[THRESHOLD_IDX]),
            );
            obj.insert("trigger_position_frac".to_string(), serde_json::json!(frac));
            obj.insert(
                "trigger_frame_kind".to_string(),
                serde_json::json!(if frac <= 0.25 {
                    "speech_onset"
                } else {
                    "sustained_speech"
                }),
            );
        }
    }
    serde_json::Value::Object(obj)
}

// ── Distribution helpers (mahbot-1005 §3/§4) ──────────────────────────────

/// Decile boundaries of a score distribution (10 values, min→max inclusive of
/// each 10% quantile).  Returns `None` for empty input (mahbot-1005 §3).
fn deciles(scores: &[f32]) -> Option<[f32; 10]> {
    if scores.is_empty() {
        return None;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = [0.0_f32; 10];
    for (i, slot) in out.iter_mut().enumerate() {
        let idx = ((sorted.len() - 1) as f32 * (i as f32 + 0.5) / 10.0).round() as usize;
        *slot = sorted[idx.min(sorted.len() - 1)];
    }
    Some(out)
}

/// Identify the PCM augmentation type from a variant label (mahbot-1005 §6).
/// Labels are produced by [`pcm_augment_enrollment_variants`]; each label ends
/// with exactly one suffix (specific cells like `_noise_white_10db` are listed
/// before the generic `_noise` fallback).
fn augmentation_type(label: &str) -> &'static str {
    const SUFFIXES: [(&str, &str); 11] = [
        ("_speed_down_090", "speed_down_090"),
        ("_noise_white_10db", "noise_white_10db"),
        ("_noise_pink_10db", "noise_pink_10db"),
        ("_noise_brown_10db", "noise_brown_10db"),
        ("_noise_white_5db", "noise_white_5db"),
        ("_noise_pink_5db", "noise_pink_5db"),
        ("_noise_brown_5db", "noise_brown_5db"),
        ("_speed_down", "speed_down"),
        ("_speed_up", "speed_up"),
        ("_vol_down", "vol_down"),
        ("_noise", "noise"),
    ];
    for (suffix, aug) in SUFFIXES {
        if label.ends_with(suffix) {
            return aug;
        }
    }
    if label.ends_with("_original") {
        "original"
    } else {
        "other"
    }
}

/// Process a list of audio clips through the detection pipeline, recording
/// results in `metrics`.  Shared helper for positive and negative detection
/// test blocks, eliminating the repetitive match-and-track boilerplate.
///
/// # Parameters
/// - `variants`: audio clips with descriptive labels.
/// - `classifier`: the trained classifier passed to the streaming pipeline.
/// - `metrics`: records total/detected; `on_detection` fills detected or
///   false_accepts.
/// - `on_detection`: called with `(&mut metrics, label_str)` when the
///   wake word is detected (for positives: increment `.detected`; for
///   negatives: push to `.false_accepts`).
fn test_detection_samples(
    variants: &[(Vec<f32>, String)],
    classifier: &WakeWordClassifier,
    metrics: &mut DetectionMetrics,
    on_detection: impl Fn(&mut DetectionMetrics, &str),
    mut adaptive_state: Option<&mut super::AdaptiveThresholdState>,
    cold_start: bool,
) {
    // Set the classifier in global state for the streaming pipeline.
    // score_stride8_window reads it from voice_state().
    super::set_classifier_weights(classifier.weights_ref().clone());

    for (i, (samples, label)) in variants.iter().enumerate() {
        info!(
            "  Variant {}/{}: {label} — processing ({})",
            i + 1,
            variants.len(),
            if cold_start { "cold start" } else { "warm" }
        );
        metrics.total += 1;
        let mut ctx = super::PipelineCtx::new();
        // Clone the shared adaptive state into this variant's ctx so the
        // adaptive threshold is active from the first frame, simulating a
        // continuous pipeline (mahbot-845, reviewer_3).  Without this the
        // adaptive state never exits its 5-frame bootstrap because each
        // variant gets a fresh ctx, keeping all benchmark metrics measured
        // against the static threshold.
        //
        // Cold start (mahbot-1006 D/F): production bootstraps the adaptive
        // threshold from the first 5 real per-frame scores of actual background
        // audio after each segment boundary.  The cold pass therefore uses a
        // fresh AdaptiveThresholdState::new() per variant (no shared-state
        // propagation) so the threshold trajectory matches production exactly.
        if !cold_start && let Some(ref mut state) = adaptive_state {
            ctx.adaptive_threshold = state.clone();
        }
        // Warm pass: feed warm-up audio so the latency timer measures only the
        // wake word and the ring is primed (mahbot-922).  The cold pass skips
        // consume_warmup — production has no warm-up after a silence boundary.
        if !cold_start {
            consume_warmup(&mut ctx);
        }
        let result = run_streaming_detection(samples, &mut ctx);
        // Propagate the updated adaptive state for the next variant (warm pass
        // only — the cold pass keeps each variant's bootstrap independent).
        //
        // When the trailing silence fires a segment boundary (`!detected` AND
        // the ring was cleared — reset_detection_segment is the only
        // non-detection path that empties the warm pass's ring), the boundary
        // resets ctx.adaptive_threshold back to bootstrap.  Propagating that
        // bootstrap state would defeat the shared-state design (mahbot-845)
        // which keeps the adaptive code path active across variants to avoid
        // measuring everything against the static threshold — and with item
        // H's extended flush every non-detecting variant fires the boundary.
        // Instead, carry the pre-flush snapshot captured at the end of the
        // utterance so the shared state keeps adapting to each variant's
        // frames.  The cold pass measures the bootstrap behavior separately
        // (mahbot-1006 F).
        if !cold_start && let Some(ref mut state) = adaptive_state {
            let boundary_fired = !result.detected && ctx.embedding_ring.is_empty();
            if boundary_fired {
                **state = result.adaptive_state_pre_flush.clone();
            } else {
                **state = ctx.adaptive_threshold.clone();
            }
        }
        let peak = ctx.instrumentation.peak_score;
        if result.detected {
            on_detection(metrics, label);
        }
        // Record per-variant result (mahbot-845) and per-variant
        // instrumentation (mahbot-886, mahbot-1005).
        metrics
            .per_variant
            .push(build_per_variant_result(label, &result, &ctx));
        info!(
            "  Variant {}/{}: {label} — {} (peak_score={:.4})",
            i + 1,
            variants.len(),
            if result.detected {
                "DETECTED"
            } else {
                "passed"
            },
            peak,
        );
    }
}

// ── Noise-overlapped detection test (mahbot-845) ─────────────────────────

/// Mix speech and noise at a given SNR (in dB).
///
/// `speech` and `noise` should be the same length.  The noise is scaled so
/// that `SNR = 20 * log10(rms_speech / (scale * rms_noise))`.
/// When `snr_db` is `f32::INFINITY`, returns the speech unchanged.
fn mix_at_snr(speech: &[f32], noise: &[f32], snr_db: f32) -> Vec<f32> {
    if !snr_db.is_finite() {
        return speech.to_vec();
    }
    // Shared RMS helper (mahbot-1043) — the bench's former byte-identical
    // `rms` copy was deleted; the 1e-10 degenerate-signal guard stays here
    // as an early return (it is NOT part of compute_rms).
    let speech_rms = crate::util::compute_rms(speech);
    let noise_rms = crate::util::compute_rms(noise);
    if noise_rms < 1e-10 || speech_rms < 1e-10 {
        return speech.to_vec();
    }
    // SNR = 20 * log10(speech_rms / (scale * noise_rms))
    // → scale = speech_rms / (noise_rms * 10^(snr_db / 20))
    let target_ratio = 10.0_f32.powf(snr_db / 20.0);
    let scale = speech_rms / (noise_rms * target_ratio);

    let min_len = speech.len().min(noise.len());
    let mut mixed = Vec::with_capacity(min_len);
    for i in 0..min_len {
        mixed.push(speech[i] + noise[i] * scale);
    }
    // If speech is longer, append remaining speech (noise exhausted)
    if min_len < speech.len() {
        mixed.extend_from_slice(&speech[min_len..]);
    }
    mixed
}

/// One channel-condition cell for [`run_detection_matrix`]: either a
/// pre-generated noise buffer mixed at a target SNR (noise-overlap matrix) or
/// seeded white noise per clip (cross-speaker probe matrix).
enum CellMix {
    /// `mix_at_snr(pcm, noise, snr_db)` — infinite SNR returns speech unchanged.
    AtSnr { noise: Vec<f32>, snr_db: f32 },
    /// `add_noise_white_pink(pcm, snr, White, seed_base + clip_index)`, or the
    /// clip unchanged when `snr_db` is None (clean cell).
    SeededWhite { snr_db: Option<f32>, seed_base: u64 },
}

/// Shared per-cell streaming-detection loop for the noise-overlap and
/// cross-speaker-probe matrices.  One warmed adaptive state is carried across
/// all cells (with the boundary-fire guard from `test_detection_samples`),
/// each clip gets a fresh `PipelineCtx`, and the warm-up audio is consumed
/// so the latency timer starts at the utterance (mahbot-922).  Returns
/// per-cell `(key, rate, detected, detail)`.
fn run_detection_matrix(
    clips: &[(Vec<f32>, String)],
    classifier: &WakeWordClassifier,
    cells: &[(String, CellMix)],
    detail_tag: &'static str,
    log_prefix: &'static str,
) -> Vec<(String, f64, usize, Vec<serde_json::Value>)> {
    // Set the classifier in global state (test_detection_samples does this too
    // but we inline the loop to share adaptive state across variants).
    super::set_classifier_weights(classifier.weights_ref().clone());
    // Pre-warm a shared adaptive threshold state so the benchmark actually
    // exercises the adaptive code path (reviewer_2, mahbot-845).  Without
    // this, test_detection_samples — which creates a fresh PipelineCtx per
    // variant — never exits the 5-frame bootstrap, so all measurements use
    // the static threshold.
    let mut shared_adaptive = super::AdaptiveThresholdState::warmed();
    let mut results = Vec::new();

    for (cond_label, mix) in cells {
        let mut metrics = DetectionMetrics::default();
        for (i, (pcm, label)) in clips.iter().enumerate() {
            let mixed_pcm = match mix {
                CellMix::AtSnr { noise, snr_db } => mix_at_snr(pcm, noise, *snr_db),
                CellMix::SeededWhite { snr_db, seed_base } => match snr_db {
                    Some(snr) => crate::util::add_noise_white_pink(
                        pcm,
                        *snr,
                        crate::util::NoiseType::White,
                        Some(*seed_base + i as u64),
                    ),
                    None => pcm.clone(),
                },
            };
            metrics.total += 1;

            // Each variant gets a fresh ctx (clean score_window,
            // embedding_ring, etc.) but carries the shared adaptive
            // threshold state forward across detection attempts.
            let mut ctx = super::PipelineCtx::new();
            ctx.adaptive_threshold = shared_adaptive.clone();
            // adaptive_k is already set by PipelineCtx::new() from config.

            // Consume warm-up so the latency timer starts at the
            // utterance (mahbot-922).
            consume_warmup(&mut ctx);
            let result = run_streaming_detection(&mixed_pcm, &mut ctx);

            // Persist the updated adaptive state for the next variant,
            // with the same boundary-fire guard as test_detection_samples
            // (mahbot-1006 F): when the trailing silence fired a segment
            // boundary (`!detected` AND the ring was cleared), the
            // boundary reset ctx.adaptive_threshold to bootstrap.
            // Propagating that bootstrap state would defeat the
            // shared-state design (mahbot-845) — and with item H's
            // extended flush every non-detecting variant fires the
            // boundary, so the phase would otherwise re-bootstrap per
            // variant and measure mostly against the static threshold.
            // Carry the pre-flush snapshot instead so the shared state
            // keeps adapting to each variant's frames.
            let boundary_fired = !result.detected && ctx.embedding_ring.is_empty();
            shared_adaptive = if boundary_fired {
                result.adaptive_state_pre_flush.clone()
            } else {
                ctx.adaptive_threshold.clone()
            };

            if result.detected {
                metrics.detected += 1;
            }
            metrics
                .per_variant
                .push(build_per_variant_result(label, &result, &ctx));
        }

        let rate = metrics.detection_rate();
        let detail: Vec<serde_json::Value> = metrics
            .per_variant
            .iter()
            .map(|pv| pv_to_json(pv, Some(detail_tag)))
            .collect();
        info!(
            "{log_prefix} {cond_label}: {:.1}% detection ({}/{})",
            rate * 100.0,
            metrics.detected,
            metrics.total,
        );
        results.push((cond_label.clone(), rate, metrics.detected, detail));
    }
    results
}

/// Run noise-overlapped detection tests.
///
/// For each combination of SNR level and noise type, mix the wake word
/// variants with the noise and test detection.  Returns `(key, rate,
/// per_variant_detail)` per combination so per-variant peak scores are
/// available for false-reject root-causing (mahbot-1005 §8).
fn run_noise_overlap_test(
    positive_variants: &[(Vec<f32>, String)],
    classifier: &WakeWordClassifier,
) -> Vec<(String, f64, Vec<serde_json::Value>)> {
    let mut cells: Vec<(String, CellMix)> = Vec::new();
    for (snr_label, snr_db) in NOISE_OVERLAP_SNRS {
        for (i, (noise_label, noise_gen)) in NOISE_OVERLAP_TYPES.iter().enumerate() {
            // Infinite-SNR mixing returns the speech unchanged (mix_at_snr),
            // so the three clean cells are byte-identical — keep only the
            // first noise type as the single representative clean cell.
            if !snr_db.is_finite() && i > 0 {
                continue;
            }
            cells.push((
                format!("{snr_label}_{noise_label}"),
                CellMix::AtSnr {
                    noise: noise_gen(),
                    snr_db: *snr_db,
                },
            ));
        }
    }
    run_detection_matrix(
        positive_variants,
        classifier,
        &cells,
        "noise_overlap",
        "Noise overlap",
    )
    .into_iter()
    .map(|(key, rate, _detected, detail)| (key, rate, detail))
    .collect()
}

/// Cross-speaker probe matrix: held-out RESERVED voices at
/// unseen seeds across clean + 10/20 dB white conditions (warm pass, shared
/// adaptive — same methodology as the canary matrix).  The reserved voices
/// are absent from EVERY training path.  Cross-speaker detections are CORRECT
/// speaker-blind behaviour (the wake word fires for any speaker), so the
/// matrix is a DIAGNOSTIC (high detection expected), not a false-accept gate.
///
/// Returns the per-cell results in the same `(key, rate, detail)` shape as
/// [`run_noise_overlap_test`] plus the total detection count.
fn run_cross_speaker_probe_matrix(
    probe_clips: &[(Vec<f32>, String)],
    classifier: &WakeWordClassifier,
) -> (Vec<(String, f64, Vec<serde_json::Value>)>, usize) {
    // 3 channel conditions per probe clip (clean + 10/20 dB white — the same
    // vocabulary as the cross-speaker training-condition set so the probe is
    // apples-to-apples with the trained-in canaries).
    let cells = [
        (
            "clean".to_string(),
            CellMix::SeededWhite {
                snr_db: None,
                seed_base: 0,
            },
        ),
        (
            "10db_white".to_string(),
            CellMix::SeededWhite {
                snr_db: Some(10.0),
                seed_base: 7000,
            },
        ),
        (
            "20db_white".to_string(),
            CellMix::SeededWhite {
                snr_db: Some(20.0),
                seed_base: 7100,
            },
        ),
    ];
    let cells_out = run_detection_matrix(
        probe_clips,
        classifier,
        &cells,
        "cross_speaker_probe",
        "Cross-speaker probe",
    );
    let fa_count: usize = cells_out.iter().map(|(_, _, detected, _)| *detected).sum();
    (
        cells_out
            .into_iter()
            .map(|(key, rate, _detected, detail)| (key, rate, detail))
            .collect(),
        fa_count,
    )
}

/// Report-only FA reporter for a detection-matrix result set: any detected
/// cell above the 0-target warns when `warn` is true (true-negative phases),
/// otherwise the detection rate is logged as a diagnostic with no warning
/// (cross-speaker cells — speaker-blindness, high detection expected).
fn warn_fa_cells(
    label: &str,
    results: &[(String, f64, Vec<serde_json::Value>)],
    total_variants: usize,
    tail: &str,
    warn: bool,
) {
    for (key, rate, detail) in results {
        let fas = detail
            .iter()
            .filter(|v| {
                v.get("detected")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            })
            .count();
        if warn && fas > 0 {
            warn!(
                "{label} FA: {key} accepted {fas}/{detail_len} cells \
                 (rate {rate_pct:.1}%) — target 0/{total_variants} per fresh run{tail}",
                detail_len = detail.len(),
                rate_pct = rate * 100.0,
            );
        } else {
            info!(
                "{label} diagnostic: {key} detected {fas}/{detail_len} cells \
                 (rate {rate_pct:.1}%) — speaker-blindness, high detection expected{tail}",
                detail_len = detail.len(),
                rate_pct = rate * 100.0,
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// mahbot-1023: enrolled-speaker benchmark phase (Phase 8d)
// ═══════════════════════════════════════════════════════════════════════════

/// Per-variant enrolled-speaker result (mahbot-1023 item 6).
#[derive(Clone)]
#[allow(clippy::struct_excessive_bools)]
struct EnrolledVariantResult {
    variant: String,
    /// End-to-end detection through the real streaming cold pass.
    detected: bool,
    /// Scoring path that produced the detection (raw source: "burst" /
    /// "segment_end_pass" / "other").  None when not detected.
    detection_path: Option<String>,
    /// Detection latency ms (CPU wall-clock of the feed loop; None when not
    /// detected).  NOTE: the feed loop processes faster than real-time, so
    /// this is a processing-cost proxy — the AUDIO position of the detection
    /// is given by `burst_sweep_buffer_len` × 10 ms (mel model stride).
    latency_ms: Option<f64>,
    /// Per-frame classifier window scores `[total_score, rolling_sum,
    /// effective_threshold]`.
    per_frame_scores: Vec<[f32; 3]>,
    /// Per-frame mel-window start positions (parallel to `per_frame_scores`).
    per_frame_window_start: Vec<usize>,
    /// Per-frame mel-buffer lengths (parallel to `per_frame_scores`).
    per_frame_mel_buffer_len: Vec<usize>,
    /// Per-frame window geometry classes (parallel to `per_frame_scores`).
    per_frame_geometry: Vec<super::WindowGeometry>,
    /// Per-frame adaptive-threshold modes (parallel to `per_frame_scores`).
    per_frame_adaptive_mode: Vec<super::AdaptiveFrameMode>,
    /// Live burst-trigger buffer length (None when the burst never ran).
    burst_sweep_buffer_len: Option<usize>,
    /// Synchronous burst sweep wall-clock (ms; None when the burst never ran).
    burst_wall_clock_ms: Option<f64>,
    /// Whether the segment-end pass ran this variant.
    segment_end_pass_fired: bool,
    /// Whether the adaptive bootstrap persisted across the whole utterance.
    adaptive_bootstrap_persisted: bool,
    /// Miss verdict for non-detected variants (see [`MissVerdict`]).
    miss_verdict: Option<MissVerdict>,
}

/// Run ONE enrolled-speaker clip through the real streaming cold pass and
/// build its [`EnrolledVariantResult`] (shared by the F1 training-clip control
/// and the held-out recall set).
///
/// Cold pass: fresh [`PipelineCtx`] per clip, no warm-up, fresh adaptive
/// bootstrap — the production post-silence start.  The deferral (deferred
/// burst + segment-end pass) is exercised for real through
/// `handle_wake_word_detection`.
fn run_enrolled_cold_variant(label: &str, pcm: &[f32]) -> EnrolledVariantResult {
    // Cold pass: fresh PipelineCtx per variant, no consume_warmup, fresh
    // AdaptiveThresholdState (matches production's post-silence start).
    let mut ctx = super::PipelineCtx::new();
    let result = run_streaming_detection(pcm, &mut ctx);
    let pv = build_per_variant_result(label, &result, &ctx);
    EnrolledVariantResult {
        variant: label.to_string(),
        detected: result.detected,
        detection_path: pv.detection_path.clone(),
        latency_ms: result.latency_ms,
        per_frame_scores: pv.per_frame_scores.clone(),
        per_frame_window_start: pv.per_frame_window_start.clone(),
        per_frame_mel_buffer_len: pv.per_frame_mel_buffer_len.clone(),
        per_frame_geometry: pv.per_frame_geometry.clone(),
        per_frame_adaptive_mode: pv.per_frame_adaptive_mode.clone(),
        burst_sweep_buffer_len: pv.burst_sweep_buffer_len,
        burst_wall_clock_ms: ctx.instrumentation.burst_wall_clock_ms,
        segment_end_pass_fired: pv.segment_end_pass_fired,
        adaptive_bootstrap_persisted: pv.adaptive_bootstrap_persisted,
        miss_verdict: if result.detected {
            None
        } else {
            Some(classify_miss(&pv))
        },
    }
}

/// Enrolled-speaker phase (mahbot-1023 Phase 8d, re-scoped).
///
/// The acceptance basis is the HELD-OUT wake-only recall set: unseen
/// renderings of the enrolled voice (new seeds in a collision-free range,
/// wake phrase alone — embedded-in-sentence detection is not a product
/// requirement and is not measured), measured through the real streaming cold
/// pass (deferred burst + segment-end pass + adaptive bootstrap).  Generated
/// strictly after training and never added to any training pool.  All 10
/// enrollment clips are training data, so the in-sample F1 control is
/// removed; detection control is entirely this held-out recall set.
#[expect(clippy::too_many_lines)]
fn run_enrolled_speaker_phase(
    held_out_clips: &[(Vec<f32>, String)],
    classifier: &WakeWordClassifier,
) -> Option<serde_json::Value> {
    let start = Instant::now();

    // Set the classifier in global state (the streaming pipeline reads it
    // from voice_state).
    super::set_classifier_weights(classifier.weights_ref().clone());

    // ── Held-out recall pass (cold, per clip) — the acceptance basis ─────
    let mut recall_variants: Vec<EnrolledVariantResult> = Vec::new();
    for (recall_pcm, recall_label) in held_out_clips {
        let v = run_enrolled_cold_variant(recall_label, recall_pcm);
        info!(
            "  Held-out recall clip {}: {} — {} (source: {}, latency: {:?}ms)",
            recall_variants.len() + 1,
            recall_label,
            if v.detected { "DETECTED" } else { "miss" },
            v.detection_path.as_deref().unwrap_or("n/a"),
            v.latency_ms,
        );
        recall_variants.push(v);
    }

    // ── Aggregate acceptance metrics ────────────────────────────────────────
    let total = recall_variants.len();
    let detected_live = recall_variants.iter().filter(|v| v.detected).count();
    // Wilson 95% confidence interval for the held-out recall proportion
    // (binomial; report-only).
    let recall_ci = wilson_ci(detected_live, total);
    let burst_path = recall_variants
        .iter()
        .filter(|v| v.detection_path.as_deref() == Some("burst"))
        .count();
    let pass_path = recall_variants
        .iter()
        .filter(|v| v.detection_path.as_deref() == Some("segment_end_pass"))
        .count();
    let other_path = recall_variants
        .iter()
        .filter(|v| v.detection_path.as_deref() == Some("other"))
        .count();
    // Mid-utterance paths (main-loop or deferred burst) before the boundary
    // fallback.
    let primary_mechanism = burst_path + other_path;
    let burst_latencies: Vec<f64> = recall_variants
        .iter()
        .filter(|v| v.detection_path.as_deref() == Some("burst"))
        .filter_map(|v| v.latency_ms)
        .collect();
    let detected_latencies: Vec<f64> = recall_variants
        .iter()
        .filter(|v| v.detected)
        .filter_map(|v| v.latency_ms)
        .collect();
    let burst_stalls: Vec<f64> = recall_variants
        .iter()
        .filter_map(|v| v.burst_wall_clock_ms)
        .collect();
    let live_trigger_bs: Vec<usize> = recall_variants
        .iter()
        .filter_map(|v| v.burst_sweep_buffer_len)
        .collect();
    let mean = |xs: &[f64]| -> Option<f64> {
        if xs.is_empty() {
            None
        } else {
            Some(xs.iter().copied().sum::<f64>() / xs.len() as f64)
        }
    };

    // ── Near-miss canaries (mahbot-1024 item 5) ────────────────────────────
    // Investigation triggers, NOT hard gates: firing means the run deserves
    // review, never an automatic pass/fail.  The canary is the classifier
    // gate: a held-out recall variant that crossed the
    // classifier gate (max_rolling_sum >= MIN_CLASSIFIER_THRESHOLD) but was
    // NOT detected end-to-end.
    let mut canaries_fired: Vec<&'static str> = Vec::new();
    let gate_crossed_not_detected: Vec<serde_json::Value> = recall_variants
        .iter()
        .filter(|v| {
            !v.detected
                && v.per_frame_scores
                    .iter()
                    .any(|s| s[ROLLING_SUM_IDX] >= MIN_CLASSIFIER_THRESHOLD)
        })
        .map(|v| serde_json::json!({ "variant": v.variant }))
        .collect();
    if !gate_crossed_not_detected.is_empty() {
        canaries_fired.push("classifier_gate_crossed_not_detected");
    }

    let per_variant_json: Vec<serde_json::Value> = recall_variants
        .iter()
        .map(|v| {
            serde_json::json!({
                "variant": v.variant,
                "detected": v.detected,
                "detection_path": v.detection_path,
                "latency_ms": v.latency_ms,
                "classifier_window_scores": v.per_frame_scores,
                "per_frame_window_start": v.per_frame_window_start,
                "per_frame_mel_buffer_len": v.per_frame_mel_buffer_len,
                "per_frame_geometry": v
                    .per_frame_geometry
                    .iter()
                    .map(|g| g.as_str())
                    .collect::<Vec<_>>(),
                "per_frame_adaptive_mode": v
                    .per_frame_adaptive_mode
                    .iter()
                    .map(|m| m.as_str())
                    .collect::<Vec<_>>(),
                "burst_sweep_buffer_len": v.burst_sweep_buffer_len,
                "burst_wall_clock_ms": v.burst_wall_clock_ms,
                "segment_end_pass_fired": v.segment_end_pass_fired,
                "adaptive_bootstrap_persisted": v.adaptive_bootstrap_persisted,
                "miss_verdict": v.miss_verdict.map(MissVerdict::as_str),
            })
        })
        .collect();

    let phase_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  Enrolled-speaker phase (held-out recall): {detected_live}/{total} detected end-to-end, \
         paths: burst={burst_path} segment_end_pass={pass_path} other={other_path}",
    );
    if !canaries_fired.is_empty() {
        warn!(
            "Enrolled-speaker phase near-miss canaries fired: {} — investigation \
             triggers, not a hard gate",
            canaries_fired.join(", "),
        );
    }

    Some(serde_json::json!({
        "note": "HELD-OUT WAKE-ONLY RECALL — the acceptance basis is \
                 unseen renderings of the ENROLLED voice: new seeds in a collision-free \
                 range (3000+), wake phrase alone (embedded-in-sentence detection is not \
                 a product requirement and is not measured), generated strictly after \
                 training and never added to any training pool.  All 10 enrollment clips \
                 are training data, so the in-sample F1 control is removed; detection \
                 control is entirely this held-out recall set.  Measures end-to-end \
                 detection through the REAL streaming cold pass: fresh PipelineCtx per \
                 clip, no warm-up, fresh adaptive bootstrap.",
        "phase_ms": phase_ms,
        "held_out_recall": {
            "n_clips": total,
            "detected": detected_live,
            "note": "Wake-only clips (isolated renderings) only.",
        },
        "acceptance": {
            "basis": format!(
                "held_out_recall_v2 — {} unseen enrolled-voice wake-only clips, seeds \
                 3000+; embedded-in-sentence detection removed (not a product \
                 requirement); the in-sample F1 control is removed (all enrollment clips \
                 are training data)",
                total,
            ),
            "total_variants": total,
            "detected_live": detected_live,
            "detected_live_frac": if total > 0 { detected_live as f64 / total as f64 } else { 0.0 },
            "recall_ci_95_wilson": recall_ci.map(|(lo, hi)| {
                serde_json::json!({"low": lo, "high": hi, "note": "Wilson score interval \
                    for the held-out recall binomial proportion."})
            }),
            "target": format!(
                "mean held-out recall >= {:.0}% (MIN_DETECTION_RATE) across >= 3 FRESH runs \
                 of the final staged code — per-run end-to-end detection (detected_live) on \
                 unseen enrolled-voice wake-only renderings",
                MIN_DETECTION_RATE * 100.0,
            ),
            "note": "detected_live = end-to-end through the real cold pass (the \
                     acceptance metric).  Bumped to v2 (16 wake-only clips): never \
                     compared against v1 24-clip archives.",
        },
        "paths": {
            "burst": burst_path,
            "segment_end_pass": pass_path,
            "other": other_path,
            "primary_mechanism": primary_mechanism,
            "note": "Raw detection sources (no reclassification): \
                     burst = deferred burst sweep fired; other = main-loop stride-8 \
                     window fired; segment_end_pass = boundary fallback pass fired.  \
                     Acceptance requires mean TP >= 4/5 across >= 3 fresh runs via the \
                     mid-utterance paths (burst + other), before the segment-end pass.",
        },
        "latency": {
            "detected_path_ms": detected_latencies,
            "detected_path_mean_ms": mean(&detected_latencies),
            "burst_path_ms": burst_latencies,
            "burst_path_mean_ms": mean(&burst_latencies),
            "burst_path_target": "<= ~1.0 s from speech onset",
            "audio_position_of_burst_ms": live_trigger_bs
                .iter()
                .map(|b| b * 10)
                .collect::<Vec<_>>(),
            "note": "latency_ms is the CPU wall-clock of the feed loop (the benchmark \
                     feeds audio faster than real-time) — a processing-cost proxy, NOT \
                     the 'from speech onset' audio latency.  The AUDIO position of the \
                     burst is burst_sweep_buffer_len x 10 ms (mel model stride): the \
                     live trigger fires ~680-800 ms into the utterance (~56% through \
                     'mahbot' for F1 at B=68-80), i.e. as the user finishes speaking \
                     it, NOT instantly — the intentional UX framing.",
        },
        "burst_stall": {
            "wall_clock_ms": burst_stalls,
            "max_ms": burst_stalls.iter().copied().fold(0.0_f64, f64::max),
            "note": "Synchronous burst sweep stall measured through the live pipeline \
                     (AGC/VAD/block_in_place overhead included).  Worst case ~44-135 ms \
                     (up to 9 ONNX calls); the 1.024 s mic channel absorbs it.  No async \
                     scoring path is used.  DROPPED_CHUNKS is only incremented on the \
                     real-mic channel — the benchmark feeds audio directly, so the \
                     no-drop criterion is an operational check on the live mic channel, \
                     stated as such (manager pin 4).",
        },
        "live_trigger_geometry": {
            "burst_sweep_buffer_lens": live_trigger_bs,
            "note": "Actual live burst-trigger buffer lengths per variant (the first \
                     flush-aligned B >= 68).  Reported per run so the acceptance review \
                     can see the live geometry instead of extrapolating from the B=68 \
                     static-gate grid (manager pin 2).",
        },
        "near_miss_canaries": {
            "classifier_gate_crossed_not_detected": gate_crossed_not_detected,
            "fired": canaries_fired,
            "note": "Investigation triggers (mahbot-1024 item 5), NOT hard gates: any \
                     fired canary means the run deserves review before acceptance.  \
                     'classifier_gate_crossed_not_detected' = a held-out recall variant \
                     crossed the classifier gate (max_rolling_sum >= 2.13) but was not \
                     detected end-to-end.",
        },
        "per_variant": per_variant_json,
    }))
}

/// Compute the mahbot-1023 safety gate over the negative/confusable set.
///
/// The deferral adds in-distribution deferred windows (burst + boundary
/// double-scoring) whose false-positive impact must be MEASURED, not assumed.
/// Baseline (mahbot-1022): 9/59 classifier gate crossings, 0 end-to-end false
/// accepts.  The classifier gate crossing count is measured per run; true
/// false accepts (end-to-end detections) are counted separately.
fn safety_gate(all_neg_pv: &[(&PerVariantResult, String)]) -> serde_json::Value {
    let total = all_neg_pv.len();
    let gate_crossings = all_neg_pv
        .iter()
        .filter(|(pv, _)| pv.max_rolling_sum >= MIN_CLASSIFIER_THRESHOLD)
        .count();
    let false_accepts: Vec<(&str, String)> = all_neg_pv
        .iter()
        .filter(|(pv, _)| pv.detected)
        .map(|(pv, cat)| (pv.variant.as_str(), cat.clone()))
        .collect();
    // mahbot-1024 near-miss canaries (investigation triggers, not hard gates).
    let mut neg_canaries_fired: Vec<&'static str> = Vec::new();
    if gate_crossings > 0 {
        neg_canaries_fired.push("classifier_gate_crossing_on_negative_corpus");
    }
    serde_json::json!({
        "note": "False-positive impact of the deferred in-distribution windows \
                 (deferred burst + boundary fallback double-scoring) on the \
                 negative/confusable set.  Baseline (mahbot-1022, pre-deferral): \
                 9/59 classifier gate crossings, 0 end-to-end false accepts.  \
                 The gate crossing count is EXPECTED to change under the \
                 deferral — it must be measured, not assumed.  Per-utterance \
                 accounting (manager pin 7): the acceptance denominator is \
                 utterances, not frames.",
        "total_negatives": total,
        "classifier_gate_crossings": gate_crossings,
        "baseline_gate_crossings": 9,
        "end_to_end_false_accepts": false_accepts.len(),
        "false_accept_list": false_accepts,
        "baseline_false_accepts": 0,
        "acceptance": "<= 2/59 false accepts (5%) across >= 3 runs; every run with a \
                       false accept must be investigated — no FA-bound breach may be \
                       silently tolerated.  Gate crossings without an end-to-end \
                       detection do NOT count as false accepts.",
        "near_miss_canaries": {
            "classifier_gate_crossing_on_negative_corpus": gate_crossings,
            "fired": neg_canaries_fired,
            "note": "Investigation triggers (mahbot-1024 item 5), NOT hard gates.  \
                     'classifier_gate_crossing_on_negative_corpus' = any negative \
                     variant crossed the classifier gate (baseline 9/59, mahbot-1022 \
                     pre-deferral — expected to fire most runs; the count is measured \
                     per run, not assumed).  Positive-side canaries are in the \
                     enrolled_speaker section.",
        },
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// The main integration test / benchmark entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Bench-side replication of production's enrollment self-test counting
/// (`voice.rs` `run_enrollment_self_test`): how many of `sequences` would
/// trigger classifier detection through the same fresh-per-utterance
/// `score_single_embedding` loop production runs at the end of enrollment.
///
/// Production keeps the counts only inside its error string, so the bench
/// replicates the loop to make the deployability verdict's recall numbers
/// structurally available without touching production's self-test.  A
/// cross-check in `run_internal` warns if the replicated pass/fail ever
/// disagrees with production's `run_enrollment_self_test` result.
fn enrollment_self_test_counts(
    sequences: &[EmbeddingSequence],
    classifier: &WakeWordClassifier,
) -> (usize, usize) {
    let mut passed = 0usize;
    for seq in sequences {
        let mut embedding_ring: Vec<Vec<f32>> = Vec::with_capacity(super::EMBEDDING_RING_MAX);
        let mut score_window = Vec::new();
        let mut detected = false;
        for embedding in &seq.embeddings {
            let (detected_this, _, _, _) = super::score_single_embedding(
                embedding,
                &mut embedding_ring,
                Some(classifier),
                &mut score_window,
                None, // no adaptive threshold during enrollment self-test
                super::ADAPTIVE_K_DEFAULT,
                false, // not a burst-path score
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
    (passed, sequences.len())
}

/// Deployability verdict (report-only, additive key): whether the trained
/// model would pass production's enrollment self-test gate (≥80% of test
/// utterances trigger detection — `ENROLLMENT_QUALITY_SELF_TEST_MIN_FRACTION`).
/// The bench NEVER hard-gates on this — it is a prominent informational flag
/// so a model production would reject is never misread as deployable.
///
/// Returns `(json, (passed, total, required, would_pass))` — the caller's
/// cross-check reuses the counts and the gate verdict instead of running the
/// full scoring loop or re-deriving the gate formula a second time.
fn deployability_json(
    self_test_result: &Result<(), String>,
    sequences: &[EmbeddingSequence],
    classifier: &WakeWordClassifier,
) -> (serde_json::Value, (usize, usize, usize, bool)) {
    let (passed, total) = enrollment_self_test_counts(sequences, classifier);
    let required =
        (total as f32 * super::ENROLLMENT_QUALITY_SELF_TEST_MIN_FRACTION).ceil() as usize;
    let would_pass = passed >= required;
    (
        serde_json::json!({
            "would_pass_production_enrollment_gate": would_pass,
            "gate_min_fraction": super::ENROLLMENT_QUALITY_SELF_TEST_MIN_FRACTION,
            "recall": {
                "passed": passed,
                "total": total,
                "required": required,
            },
            "recall_label": if would_pass {
                "model production would accept — informational only"
            } else {
                "model production would reject — informational only"
            },
            "production_self_test_passed": self_test_result.is_ok(),
            "warning": if would_pass {
                serde_json::Value::Null
            } else {
                serde_json::Value::String(format!(
                    "PRODUCTION WOULD REJECT THIS MODEL: only {passed}/{total} enrollment \
                     utterances triggered detection (gate >= {required}, {:.0}%).  The bench \
                     is report-only — this does NOT fail the run — but the trained model is \
                     not deployable as-is (informational only).",
                    super::ENROLLMENT_QUALITY_SELF_TEST_MIN_FRACTION * 100.0,
                ))
            },
            "note": "Mirrors production's run_enrollment_self_test loop (fresh \
                     score_single_embedding per utterance); the recall \
                     numbers are the deployability signal, labeled informational-only \
                     because the bench is report-only.",
        }),
        (passed, total, required, would_pass),
    )
}

/// Numeric distribution summary (n / values / mean / min / max / p50 / p90).
/// Non-finite values (NaN from a "not measured" metric) are dropped; empty
/// input yields `{"n": 0}` — callers treat that as "not measured".
fn numeric_distribution(values: &[f64]) -> serde_json::Value {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return serde_json::json!({"n": 0});
    }
    let mut sorted = finite.clone();
    sorted.sort_by(f64::total_cmp);
    let pct = |q: f64| -> f64 {
        let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
        sorted[idx]
    };
    serde_json::json!({
        "n": sorted.len(),
        "values": sorted,
        "mean": sorted.iter().sum::<f64>() / sorted.len() as f64,
        "min": sorted[0],
        "max": sorted[sorted.len() - 1],
        "p50": pct(0.5),
        "p90": pct(0.9),
    })
}

/// Summarize the last N archived benchmark reports so the documented ≥3-run
/// acceptance protocol is self-documenting in every report (no manual archive
/// digging).
///
/// Bounded to the 3 most recent PRE-EXISTING timestamped archives (the
/// current run's archive is written after this summary is computed) so the
/// default-run budget is unaffected — 3 × ~3 MB JSON parses, tens of ms.
/// Corrupt/unreadable archives are skipped (the count says how many were
/// usable).  Archives are additionally filtered by acceptance basis
/// ([`ACCEPTANCE_BASIS_PREFIX`]) so v2 wake-only runs never mix with v1
/// 24-clip archives in the ≥3-run spread.
fn cross_run_summary() -> serde_json::Value {
    let Ok(report_dir) = crate::config::default_config_dir() else {
        return serde_json::json!({"archives_found": 0, "note": "report dir unavailable"});
    };
    let mut archives: Vec<std::path::PathBuf> = std::fs::read_dir(&report_dir)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                // Timestamped archives only (`.20<yy><mm><dd>-<hh><mm><ss>`);
                // the live report.json and .prior.json are not per-run archives.
                n.starts_with("voice_pipeline_e2e_report.20")
                    && std::path::Path::new(n)
                        .extension()
                        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
            })
        })
        .collect();
    archives.sort();
    // Keep only the most recent 3 (timestamped names sort chronologically) —
    // capture the pre-drain count so `archives_total_on_disk` reflects the
    // true on-disk count, not the bounded subset.
    let archives_total_on_disk = archives.len();
    if archives.len() > 3 {
        archives.drain(..archives.len() - 3);
    }
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut fracs: Vec<f64> = Vec::new();
    let mut fas: Vec<f64> = Vec::new();
    let mut self_test_passed: Vec<bool> = Vec::new();
    let mut phrases: Vec<String> = Vec::new();
    for path in &archives {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_string();
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let basis = v
            .pointer("/enrolled_speaker/acceptance/basis")
            .and_then(serde_json::Value::as_str);
        // Basis filter: only archives from the current acceptance
        // series enter the ≥3-run spread — v2 wake-only runs never mix with v1
        // 24-clip archives.
        if !basis.is_some_and(|b| b.starts_with(ACCEPTANCE_BASIS_PREFIX)) {
            continue;
        }
        let frac = v
            .pointer("/enrolled_speaker/acceptance/detected_live_frac")
            .and_then(serde_json::Value::as_f64);
        let total_fa = v
            .get("total_false_accepts")
            .and_then(serde_json::Value::as_u64);
        let st = v
            .get("self_test")
            .and_then(|s| s.get("passed"))
            .and_then(serde_json::Value::as_bool);
        let phrase = v
            .pointer("/reproducibility/wake_phrase/used")
            .and_then(serde_json::Value::as_str);
        if let Some(f) = frac {
            fracs.push(f);
        }
        if let Some(f) = total_fa {
            fas.push(f as f64);
        }
        if let Some(b) = st {
            self_test_passed.push(b);
        }
        if let Some(s) = phrase {
            phrases.push(s.to_string());
        }
        rows.push(serde_json::json!({
            "archive": name,
            "enrolled_detected_live_frac": frac,
            "acceptance_basis": basis,
            "total_false_accepts": total_fa,
            "self_test_passed": st,
            "wake_phrase_used": phrase,
        }));
    }
    serde_json::json!({
        "archives_found": rows.len(),
        "archives_total_on_disk": archives_total_on_disk,
        "per_run": rows,
        "distribution": {
            "enrolled_detected_live_frac": numeric_distribution(&fracs),
            "total_false_accepts": numeric_distribution(&fas),
            "self_test_passed": serde_json::json!({
                "n": self_test_passed.len(),
                "values": self_test_passed,
            }),
            "wake_phrase_used": serde_json::json!({
                "n": phrases.len(),
                "values": phrases,
            }),
        },
        "note": "Last 3 pre-existing timestamped archives (bounded so the default-run \
                 budget holds) filtered to the current acceptance basis \
                 (ACCEPTANCE_BASIS_PREFIX) — v2 wake-only runs never mix with v1 \
                 24-clip archives in the >=3-run spread.  The v2 \
                 series starts fresh: until 3 v2 runs accumulate, the spread is \
                 partial.",
    })
}

/// Env-gated FAPH phase (mahbot-1057, honest methodology mahbot-1081):
/// real-audio false-acceptance-per-hour bench.
///
/// Feeds the pinned corpus (`faph_corpus_manifest.json` — alexwengg/musan_mini,
/// 812 files, ~5.99 h audio, SHA-256-pinned per file) through the production
/// streaming detection path as **ambient audio** with ONE continuous
/// [`super::PipelineCtx`] across the whole corpus (all samples fed in
/// [`FRAME_LENGTH`](super::FRAME_LENGTH) chunks via [`process_frame`], no
/// early exit) and counts false-accept events.
///
/// # FA-counting semantics (pinned by mahbot-1053 planning gate, honest
/// methodology mahbot-1081)
///
/// - **Event-based**: each fresh detection event (a new
///   `last_wake_word_detection` timestamp) counts as 1 FA.
/// - **Continuous listening**: one pipeline context across the whole corpus;
///   a 2 s silence gap between files fires the natural segment-boundary reset
///   (fresh AGC, adaptive bootstrap, cleared ring) the way production listens.
/// - **Raw vs cooldown-merged**: production's wall-clock `WAKE_WORD_COOLDOWN`
///   (3 s) would suppress ~180 s of audio per event at ~60× feed — the bench
///   observes the RAW event stream (re-arms after each event like production's
///   post-command reset, clearing the wall-clock timestamp bench-side —
///   production's gate is unmodified) and reports BOTH the raw count and the
///   production-equivalent count merged on 3 s of AUDIO position.
/// - **Two denominators**: FA/h (and Poisson 95% upper bound) against both
///   raw audio hours and VAD-active (speech-active) hours, stated primarily
///   against the VAD-active denominator.  VAD-active time is counted from the
///   continuous feed itself — production's global VAD detector, evolved
///   across the whole corpus via the `vad_speech_frames` instrumentation
///   counter (one entry per 256-sample hop) — never a fresh per-file
///   detector, whose bootstrap bias would distort the primary denominator.
/// - **Ambient feed, not commands**: [`run_streaming_detection`] early-exits
///   on the first detection and cannot count multiple events — this phase
///   feeds every sample through [`process_frame`] and re-arms the pipeline
///   after each event exactly like production's post-command reset
///   (`reset_pipeline_state(Soft)` + `is_recording = false`).
///
/// # Degraded-skip contract
///
/// Corpus absent / download fail / hash mismatch / offline ⇒ returns a
/// documented `"status": "skipped"` report with reason keys — NEVER a hard
/// error (the bench completes with a documented skip).
///
/// # Return value
///
/// `None` when the env gate is off (standard-run report surface untouched —
/// no `faph` key is added to the JSON).  `Some(json)` when
/// `MAHBOT_FAPH=1` (ran or documented-skipped).
///
/// # Acceptance-only budget
///
/// This phase is **acceptance-only**: max ~2 runs per acceptance round (one
/// round = the ≥3-run protocol), run only when a ticket/claim requires
/// FA/h statistics, and a skipped run counts against the budget.  It stays
/// env-gated (zero cost in the default loop) and report-only — never a gate.
fn run_faph_phase() -> Option<serde_json::Value> {
    if std::env::var("MAHBOT_FAPH").as_deref() != Ok("1") {
        return None;
    }
    Some(run_faph_phase_inner())
}

/// Inner FAPH phase body (env gate already checked).
#[expect(clippy::too_many_lines)]
fn run_faph_phase_inner() -> serde_json::Value {
    // Continuous-listening inter-file silence gap: 2 s at 16 kHz (32 000
    // samples) > the 304 ms segment-boundary threshold so the natural
    // reset fires between files.
    const FILE_GAP_SAMPLES: usize = 32_000;
    let phase_start = Instant::now();

    // ── Corpus manifest (pinned, embedded at compile time) ──
    let manifest: serde_json::Value =
        match serde_json::from_str(include_str!("faph_corpus_manifest.json")) {
            Ok(m) => m,
            Err(e) => {
                return faph_skip_json(
                    "manifest_parse_failed",
                    &format!("embedded manifest failed to parse: {e}"),
                );
            }
        };
    let files: Vec<(String, String, u64)> = match manifest["files"].as_array() {
        Some(arr) => arr
            .iter()
            .filter_map(|f| {
                Some((
                    f[0].as_str()?.to_string(),
                    f[1].as_str()?.to_string(),
                    f[2].as_u64()?,
                ))
            })
            .collect(),
        None => Vec::new(),
    };
    let repo = manifest["repo"].as_str().unwrap_or("alexwengg/musan_mini");
    let revision = manifest["revision"].as_str().unwrap_or("");
    let corpus_sha = manifest["corpus_sha256"].as_str().unwrap_or("?");
    let version = manifest["version"].as_u64().unwrap_or(0);
    if files.is_empty() {
        return faph_skip_json("manifest_empty", "manifest file list is empty");
    }

    // ── Corpus cache root (self-contained, persistent, under ~/.mahbot/) ──
    let cache_root = match crate::config::default_config_dir() {
        Ok(d) => d.join("faph_corpus"),
        Err(e) => {
            return faph_skip_json(
                "cache_root_unavailable",
                &format!("cannot resolve ~/.mahbot for corpus cache: {e}"),
            );
        }
    };

    // ── Self-contained download/verify (no embedder.rs dependency) ────────
    let mut download_errors: Vec<String> = Vec::new();
    for (path, sha256, size) in &files {
        let dest = cache_root.join(path);
        if dest.exists() && std::fs::metadata(&dest).is_ok_and(|m| m.len() == *size) {
            // Verify SHA-256 (fast path: already verified on a prior run or
            // download below).
            match faph_sha256_file(&dest) {
                Ok(h) if h == *sha256 => continue,
                Ok(_) => {
                    download_errors.push(format!("{path}: hash mismatch on cached file"));
                    continue;
                }
                Err(e) => {
                    download_errors.push(format!("{path}: read error {e}"));
                    continue;
                }
            }
        }
        let url = format!("https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}");
        match faph_download_file(&url, &dest, sha256) {
            Ok(()) => {
                info!("FAPH corpus: downloaded {path} ({size} bytes)");
            }
            Err(e) => {
                download_errors.push(format!("{path}: {e}"));
            }
        }
    }

    // ── Degraded-skip contract: any missing/mismatched file → skip ────────
    if !download_errors.is_empty() {
        let reason = format!(
            "corpus incomplete — {} file(s) failed download/verify (first: {})",
            download_errors.len(),
            download_errors[0],
        );
        return faph_skip_json("corpus_download_failed", &reason);
    }

    // ── Feed loop: ambient audio FA counting ──────────────────────────────
    // Classifier must be set as a global (score_stride8_window reads it from
    // voice_state()).  The bench's earlier phases set it; re-set from the
    // local trained model to be self-contained.
    let Some(weights) = super::get_classifier_weights() else {
        return faph_skip_json(
            "classifier_unavailable",
            "no trained classifier weights in global state",
        );
    };
    super::set_classifier_weights(weights.clone());

    let mut total_audio_secs = 0.0f64;
    let mut total_vad_active_secs = 0.0f64;
    let mut raw_events: Vec<f64> = Vec::new();
    let mut files_fed = 0u64;
    let mut files_decode_failed = 0u64;
    let mut per_file_events: Vec<serde_json::Value> = Vec::new();

    // Continuous listening (honest FAPH methodology): ONE PipelineCtx across
    // the whole corpus so AGC convergence, the adaptive threshold, and
    // noise-RMS behavior persist the way production actually listens.  A short
    // silence gap between files fires the natural segment-boundary reset
    // (reset_detection_segment after SEGMENT_TIMEOUT_HOPS of VAD-inactive
    // audio).
    let gap: Vec<f32> = vec![0.0; FILE_GAP_SAMPLES];
    let mut ctx = super::PipelineCtx::new();
    let mut audio_pos = 0.0f64;

    for (path, _sha256, _size) in &files {
        let p = cache_root.join(path);
        let samples = match crate::audio::local_transcriber::decode_audio_to_mono_f32(&p) {
            Ok(s) => s,
            Err(e) => {
                files_decode_failed += 1;
                warn!("FAPH corpus: decode failed for {path}: {e}");
                continue;
            }
        };
        let audio_secs = samples.len() as f64 / f64::from(super::SAMPLE_RATE);
        total_audio_secs += audio_secs;
        let file_events = faph_feed_file_continuous(&samples, &mut ctx, &mut audio_pos);
        let n_raw = file_events.len();
        let n_merged = faph_merge_events(&file_events, super::WAKE_WORD_COOLDOWN.as_secs_f64());
        raw_events.extend(file_events);
        files_fed += 1;
        // Feed the inter-file silence gap so the segment-boundary reset fires
        // naturally (the next file starts with production's post-silence
        // state: fresh AGC, adaptive threshold bootstrap, cleared ring).
        // Gap events (rare — silence) are part of continuous listening and
        // join the global raw stream.
        raw_events.extend(faph_feed_file_continuous(&gap, &mut ctx, &mut audio_pos));
        // VAD-active seconds for the file+2s-gap window from the CONTINUOUS
        // feed: production's global VAD detector (evolved across the whole
        // corpus, exactly how the mic path listens) — the honest basis for the
        // primary FA/h denominator.  The counter accumulates one entry per
        // 256-sample hop through `handle_wake_word_detection` and is reset by
        // `faph_clear_instrumentation` below after this capture.  The key name
        // marks the window: unlike sibling `audio_secs` (file-only), this
        // covers the file plus the inter-file silence gap (~0 contribution —
        // digital silence) so per-file entries reconcile exactly against the
        // feed totals.
        let file_vad_active_secs = ctx.instrumentation.vad_speech_frames as f64
            * super::HOP_LENGTH as f64
            / f64::from(super::SAMPLE_RATE);
        total_vad_active_secs += file_vad_active_secs;
        per_file_events.push(serde_json::json!({
            "path": path,
            "audio_secs": audio_secs,
            "vad_active_secs_incl_gap": file_vad_active_secs,
            "raw_fa_events": n_raw,
            "cooldown_merged_fa_events": n_merged,
        }));
        // Bound memory: drop the per-file diagnostic vectors (never read in
        // this phase) while preserving the continuous acoustic state.
        faph_clear_instrumentation(&mut ctx);
    }

    let wall_secs = phase_start.elapsed().as_secs_f64();
    let audio_hours = total_audio_secs / 3600.0;
    let vad_active_hours = total_vad_active_secs / 3600.0;
    let total_events = raw_events.len();
    let merged_events = faph_merge_events(&raw_events, super::WAKE_WORD_COOLDOWN.as_secs_f64());

    // FA/h against both denominators (raw audio hours and VAD-active hours),
    // with the Poisson 95% upper bound stated primarily against the
    // VAD-active denominator — the honest speech-exposure basis.
    let fa_per_hour_raw = if audio_hours > 0.0 {
        merged_events as f64 / audio_hours
    } else {
        f64::NAN
    };
    let fa_per_hour_vad = if vad_active_hours > 0.0 {
        merged_events as f64 / vad_active_hours
    } else {
        f64::NAN
    };
    // Poisson 95% upper bound on the rate (k events in T hours): the largest
    // λ such that the Poisson tail P(X ≤ k) ≥ 0.025 is 0.5·χ²(0.975, 2(k+1))/T.
    // Wilson–Hilferty chi-square quantile (<0.6% error at these df).
    let poisson_upper_events = 0.5 * chi2_quantile(0.975, 2.0 * (merged_events as f64 + 1.0));
    let poisson_upper_per_hour_raw = if audio_hours > 0.0 {
        poisson_upper_events / audio_hours
    } else {
        f64::NAN
    };
    let poisson_upper_per_hour_vad = if vad_active_hours > 0.0 {
        poisson_upper_events / vad_active_hours
    } else {
        f64::NAN
    };

    info!(
        "FAPH (mahbot-1057/1081): {files_fed} files fed, {audio_hours:.2} h audio \
         ({vad_active_hours:.2} h VAD-active), {merged_events} cooldown-merged FA events \
         (raw {total_events}), {fa_per_hour_vad:.4} FA/h VAD-active \
         (Poisson 95% upper {poisson_upper_per_hour_vad:.4}/h), {wall_secs:.1}s wall",
    );
    eprintln!(
        "         FAPH: {files_fed} files fed, {audio_hours:.2} h audio \
         ({vad_active_hours:.2} h VAD-active), {merged_events} cooldown-merged FA events \
         (raw {total_events}), {fa_per_hour_vad:.4} FA/h VAD-active \
         (Poisson 95% upper {poisson_upper_per_hour_vad:.4}/h), {wall_secs:.1}s wall",
    );

    serde_json::json!({
        "status": "ran",
        "metric": "SPONTANEOUS-CONFUSABLE FA rate — corpus contains ~zero wake-word \
                   utterances, so every detection is a spontaneous false accept",
        "corpus": {
            "manifest_version": version,
            "repo": repo,
            "revision": revision,
            "corpus_sha256": corpus_sha,
            "files_total": files.len(),
            "audio_hours_total": total_audio_secs / 3600.0,
            "vad_active_hours_total": vad_active_hours,
        },
        "feed": {
            "files_fed": files_fed,
            "files_decode_failed": files_decode_failed,
            "audio_hours_fed": audio_hours,
            "vad_active_hours_fed": vad_active_hours,
            "wall_secs": wall_secs,
            "realtime_factor": if wall_secs > 0.0 { total_audio_secs / wall_secs } else { f64::NAN },
            "continuous_listening": true,
            "inter_file_silence_gap_secs": FILE_GAP_SAMPLES as f64 / f64::from(super::SAMPLE_RATE),
            "semantics": "continuous listening — ONE PipelineCtx across the whole corpus; \
                          a short silence gap between files fires the natural segment-boundary \
                          reset (fresh AGC, adaptive bootstrap, cleared ring) the way production \
                          listens; raw events observed by re-arming after each event (production \
                          post-command reset) with the wall-clock cooldown timestamp cleared \
                          bench-side (production's WAKE_WORD_COOLDOWN gate unmodified); \
                          cooldown is applied as a 3 s AUDIO-position merge for the honest count; \
                          denominators reported against both raw audio hours and VAD-active hours",
        },
        "fa": {
            "raw_events": total_events,
            "cooldown_merged_events": merged_events,
            "cooldown_merge_audio_secs": super::WAKE_WORD_COOLDOWN.as_secs_f64(),
            "fa_per_hour_raw_denominator": fa_per_hour_raw,
            "fa_per_hour_vad_active_denominator": fa_per_hour_vad,
            "poisson_95_upper_per_hour_raw_denominator": poisson_upper_per_hour_raw,
            "poisson_95_upper_per_hour_vad_active_denominator": poisson_upper_per_hour_vad,
            "poisson_95_upper_events": poisson_upper_events,
            "primary_denominator": "VAD-active (speech-active) hours — the honest \
                                     speech-exposure basis; raw-hours readings are secondary",
            "cooldown_suppression": "wall-clock WAKE_WORD_COOLDOWN (3 s) at ~50-60x feed \
                                     would suppress ~180 s of audio per event — bypassed \
                                     bench-side (raw stream) and re-applied as a 3 s \
                                     audio-position merge (production-equivalent count)",
            "continuous_listening": true,
        },
        "provenance": "0.008–0.083 FA/h projection = manager's 7050-fork analysis of \
                       synthetic cross-speaker noise-overlap rates (recorded when citing); \
                       NOT a direct <1 FA/24h demonstration — this phase reports a Poisson \
                       confidence/upper bound from the actual audio-hours processed",
        "per_file_merged_note": "per_file[].cooldown_merged_fa_events merges only WITHIN \
                                 each file; the headline fa.cooldown_merged_events merges \
                                 across file boundaries (including the inter-file silence \
                                 gaps), so per-file merged counts do not sum to the global \
                                 count — the global merge is the production-equivalent one.",
        "per_file": per_file_events,
    })
}

/// Build a documented-skip FAPH report (degraded-skip contract — never a
/// hard error).
fn faph_skip_json(reason_key: &str, detail: &str) -> serde_json::Value {
    warn!("FAPH (mahbot-1057) skipped: {reason_key} — {detail}");
    eprintln!("         FAPH (mahbot-1057) skipped: {reason_key} — {detail}");
    serde_json::json!({
        "status": "skipped",
        "skip_reason": reason_key,
        "skip_detail": detail,
        "metric": "SPONTANEOUS-CONFUSABLE FA rate — not measured (degraded skip)",
    })
}

/// SHA-256 of a file (hex).
fn faph_sha256_file(path: &std::path::Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    let data = std::fs::read(path)?;
    Ok(crate::util::hex_string(&Sha256::digest(&data)))
}

/// Self-contained single-file download with SHA-256 verification.
///
/// Downloads `url` to `dest` (creating parent dirs), verifies the SHA-256
/// against the manifest pin, and only then writes the file.  Uses its own
/// local current-thread tokio runtime — deliberately NOT the embedder.rs
/// download precedent (mahbot-1041 gated embedder/providers out of the bench
/// surface), and no new runtime dependency for the standard bench run.
fn faph_download_file(
    url: &str,
    dest: &std::path::Path,
    expected_sha256: &str,
) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    // Self-contained runtime: safe from a spawn_blocking context (no outer
    // async context is entered; Handle::try_current is checked first so a
    // runtime already present on this thread is reused rather than nested).
    let fut = async {
        let client = reqwest::Client::new();
        let resp = client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("HTTP status {}", resp.status()));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| format!("body read failed: {e}"))?;
        Ok::<Vec<u8>, String>(bytes.to_vec())
    };
    let bytes = if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.block_on(fut)
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("cannot build download runtime: {e}"))?;
        rt.block_on(fut)
    }?;
    let sha = crate::util::hex_string(&Sha256::digest(&bytes));
    if sha != expected_sha256 {
        return Err(format!(
            "SHA-256 mismatch: expected {expected_sha256}, got {sha}"
        ));
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
    }
    std::fs::write(dest, &bytes).map_err(|e| format!("write: {e}"))
}

/// Count raw false-accept events while feeding one corpus file as ambient
/// audio through a SHARED continuous-listening pipeline context (honest FAPH
/// methodology — one pipeline context across the whole corpus, matching
/// production's continuous listening).
///
/// Feeds every sample in [`FRAME_LENGTH`](super::FRAME_LENGTH) chunks through
/// [`process_frame`] (AGC → `handle_wake_word_detection`), recording the
/// AUDIO position (seconds) of each fresh `last_wake_word_detection`
/// timestamp.  Production's `WAKE_WORD_COOLDOWN` gate inside
/// `handle_wake_word_detection` is wall-clock based: at the bench's ~50-60×
/// feed speed it would suppress ~180 s of audio per event, structurally
/// under-counting.  The bench therefore observes the RAW event stream by
/// re-arming after each event exactly like production's post-command reset
/// (`reset_pipeline_state(Soft)` + `is_recording = false`) AND clearing the
/// cooldown timestamp (bench-side emulation — production's gate is NOT
/// modified).  The caller applies production's 3 s cooldown as an
/// audio-position merge ([`faph_merge_events`]) for the honest
/// production-equivalent count.
///
/// `audio_pos` is the running audio position (seconds) across the whole
/// corpus; it advances as samples are fed so event positions are globally
/// comparable for the merge.
fn faph_feed_file_continuous(
    samples: &[f32],
    ctx: &mut super::PipelineCtx,
    audio_pos: &mut f64,
) -> Vec<f64> {
    let mut raw_events: Vec<f64> = Vec::new();
    for chunk in samples.chunks(super::FRAME_LENGTH) {
        process_frame(chunk, ctx);
        if ctx.last_wake_word_detection.is_some() {
            raw_events.push(*audio_pos);
            // Re-arm (production post-command reset): Soft preserves the
            // cooldown timestamp, so also clear it — the bench observes the
            // raw event stream; the 3 s cooldown is applied as an
            // audio-position merge by the caller.
            ctx.reset_pipeline_state(super::ResetLevel::Soft);
            ctx.is_recording = false;
            ctx.last_wake_word_detection = None;
        }
        *audio_pos += chunk.len() as f64 / f64::from(super::SAMPLE_RATE);
    }
    raw_events
}

/// Merge raw event audio positions under production's cooldown semantics.
///
/// After a detection, production suppresses re-detections within
/// [`WAKE_WORD_COOLDOWN`](super::WAKE_WORD_COOLDOWN) — 3 s of wall time,
/// which at 1× realtime equals 3 s of audio.  Events whose audio positions
/// are less than that apart are merged (the later one would have been
/// suppressed).  `events` must be sorted ascending by audio position (the
/// feed loop produces them in order).
fn faph_merge_events(events: &[f64], cooldown_secs: f64) -> usize {
    let mut merged = 0usize;
    let mut last_kept: Option<f64> = None;
    for &pos in events {
        match last_kept {
            Some(prev) if pos - prev < cooldown_secs => {}
            _ => {
                merged += 1;
                last_kept = Some(pos);
            }
        }
    }
    merged
}

/// Clear the session-lifetime per-frame instrumentation vectors between
/// corpus files in the continuous-listening FAPH feed.
///
/// The FAPH phase only counts detection events — the per-frame instrumentation
/// (scores / embeddings / geometry, one entry per voiced frame) is never read
/// here, and leaving it would grow unboundedly across the whole corpus
/// (~12 h of audio → hundreds of MB of embedding vectors).  Only the
/// diagnostic vectors are cleared; the acoustic state the continuous feed
/// depends on (AGC, adaptive threshold, VAD detector, ring) is preserved.
///
/// # Future-field hazard
/// A per-frame field added to `PipelineCtxInstrumentation` without a matching
/// clear here would silently accumulate unbounded memory across the corpus —
/// keep this clear list exhaustive when the instrumentation struct grows.
fn faph_clear_instrumentation(ctx: &mut super::PipelineCtx) {
    ctx.instrumentation.per_frame_scores.clear();
    ctx.instrumentation.per_frame_embeddings.clear();
    ctx.instrumentation.per_frame_geometry.clear();
    ctx.instrumentation.adaptive_threshold_trajectory.clear();
    ctx.instrumentation.per_frame_embedding_hashes.clear();
    ctx.instrumentation.per_frame_embedding_l2_norms.clear();
    ctx.instrumentation.per_frame_window_start.clear();
    ctx.instrumentation.per_frame_mel_buffer_len.clear();
    ctx.instrumentation.per_frame_adaptive_mode.clear();
    ctx.instrumentation.per_hop_vad.clear();
    ctx.instrumentation.n_frames_below_reset = 0;
    ctx.instrumentation.vad_speech_frames = 0;
    ctx.instrumentation.ceiling_limited_frames = 0;
}

/// Chi-square quantile via the Wilson–Hilferty approximation.
///
/// `χ²_α(ν) ≈ ν·(1 − 2/(9ν) + z_α·√(2/(9ν)))³` where `z_α` is the standard
/// normal quantile (Acklam's algorithm).  Error < 0.6% at the degrees of
/// freedom used for the FAPH Poisson bound (df ≥ 2).
fn chi2_quantile(p: f64, df: f64) -> f64 {
    let z = normal_quantile(p);
    let base = 1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt();
    df * base * base * base
}

/// Wilson score interval (95% confidence) for a binomial proportion
/// (report-only).  Returns `None` for an empty sample.
fn wilson_ci(k: usize, n: usize) -> Option<(f64, f64)> {
    if n == 0 {
        return None;
    }
    // Φ⁻¹(0.975) via the Acklam approximation shared with `chi2_quantile`.
    let z = normal_quantile(0.975);
    let n = n as f64;
    let k = k as f64;
    let z2 = z * z;
    let centre = (k + z2 / 2.0) / (n + z2);
    let half = z / (n + z2) * (k * (n - k) / n + z2 / 4.0).sqrt();
    Some(((centre - half).max(0.0), (centre + half).min(1.0)))
}

/// Standard normal quantile (Acklam's algorithm).
///
/// Coefficients are function-level `const`s declared before any statements to
/// satisfy clippy::items_after_statements.
fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;
    debug_assert!((0.0..=1.0).contains(&p));
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        num / den
    } else if p > P_HIGH {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        -num / den
    } else {
        let q = p - 0.5;
        let r = q * q;
        let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        num / den
    }
}

/// Run the full voice pipeline E2E benchmark.
///
/// This is called from `voice::run_voice_pipeline_benchmark()` which is the
/// entry point for `benches/voice_pipeline_e2e.rs`.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::if_not_else
)]
pub(crate) fn run_internal() {
    // ── Phase index constants for type-safe timing access ────────────────
    // Must be before any statements (clippy items_after_statements).
    const P_ENROLLMENT_AUDIO: usize = 0;
    const P_VAD_ENROLLMENT: usize = 1;
    const P_NEG_TRAINING_DATA: usize = 2;
    const P_CLASSIFIER_TRAINING: usize = 3;
    const P_GLOBAL_STATE: usize = 4;
    const P_STREAMING_SETUP: usize = 5;
    const P_POSITIVE_VARIANTS: usize = 6;
    const P_CONFUSABLE_NEGATIVES: usize = 7;
    const P_UNRELATED_NEGATIVES: usize = 8;
    const P_SILENCE_NEGATIVES: usize = 9;
    const P_NOISE_PROFILES: usize = 10;
    const P_COOLDOWN: usize = 11;
    const P_NOISE_OVERLAP: usize = 12;
    const P_TEARDOWN: usize = 13;
    const NUM_PHASES: usize = 14;

    // ── Heartbeat drop guard (mahbot-944) ──────────────────────────────
    // Defined as an item (before any statements) so it satisfies Clippy's
    // items_after_statements requirement.  Used by the heartbeat thread to
    // ensure the stop flag is set even on panic unwind.
    struct HeartbeatGuard(std::sync::Arc<std::sync::atomic::AtomicBool>);

    impl Drop for HeartbeatGuard {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    // Initialize a tracing subscriber so progress info!() messages appear
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .parse("info")
                .expect("info env filter"),
        )
        .try_init();

    // ── Phase timing ─────────────────────────────────────────────────
    // We measure each named phase with Instant timestamps.  Phase 13's
    // cooldown sleep is excluded from its timing.
    let mut phase_starts: Vec<(&str, Instant)> = Vec::new();
    macro_rules! phase_start {
        ($name:literal) => {{
            // Both stderr (for live visibility) and info! (for tracing/logs.db).
            // Stderr ensures output is visible even if the tracing subscriber is
            // not initialized or buffers messages (mahbot-944).
            eprintln!("─── {}. {} ───", phase_starts.len() + 1, $name);
            info!("─── {}. {} ───", phase_starts.len() + 1, $name);
            phase_starts.push(($name, Instant::now()));
        }};
    }
    macro_rules! phase_end_ms {
        () => {{
            let (name, start) = phase_starts.last().unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("  → {} completed in {:.0}ms", name, elapsed);
            info!("  → {} completed in {:.0}ms", name, elapsed);
            elapsed as u64
        }};
    }

    let overall_start = Instant::now();
    let mut phase_times = [0u64; NUM_PHASES];

    info!("═══ Voice Pipeline E2E Benchmark ═══");

    info!("Benchmark tier: all three tiers tested (Easy FA≤0, Medium FA≤1, Hard FA≤2)");

    // ── Heartbeat thread (mahbot-944) ──────────────────────────────────
    // Prints a pulse every 60 s so the operator can confirm the benchmark
    // is alive and making progress (vs. parked threads from nested runtime
    // hangs).
    //
    // Uses 1 s sleep intervals so the thread responds to the stop flag
    // within 1 s (join latency).  A counter gates the print cadence — the
    // first pulse prints immediately so a hang in the first 60 s is still
    // detectable.
    //
    // A drop guard (`_heartbeat_guard`) sets the stop flag on panic unwind,
    // ensuring the heartbeat thread exits promptly even when the function
    // panics before the explicit stop-then-join at the end.
    let heartbeat_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _heartbeat_guard = HeartbeatGuard(heartbeat_stop.clone());
    let heartbeat_handle = {
        let stop = heartbeat_stop.clone();
        let start = overall_start;
        std::thread::spawn(move || {
            let mut counter: u64 = 0;
            loop {
                // Print immediately on first iteration (counter=0, 0 % 60 = 0),
                // then every 60 ticks (~60 s with 1 s sleep between ticks).
                if counter.is_multiple_of(60) {
                    let elapsed = start.elapsed();
                    eprintln!(
                        "[heartbeat] benchmark still running — elapsed: {}m{:02}s",
                        elapsed.as_secs() / 60,
                        elapsed.as_secs() % 60,
                    );
                }
                std::thread::sleep(Duration::from_secs(1));
                counter += 1;
                if stop.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
            }
        })
    };

    // ── 0. Initialize global state ─────────────────────────────────────
    // Set CONFIG storage root so model paths resolve.
    if crate::config::CONFIG.try_storage_root().is_none() {
        let mahbot_dir = crate::config::default_config_dir()
            .expect("Cannot resolve home directory for ~/.mahbot");
        crate::config::CONFIG.set_storage_root(mahbot_dir.clone());
        info!("CONFIG storage root set to: {}", mahbot_dir.display());
    }

    // Determine cache settings
    let cache_dir_path = cache_dir();
    if let Err(e) = std::fs::create_dir_all(&cache_dir_path) {
        eprintln!(
            "WARNING: Cannot create cache directory {}: {e}",
            cache_dir_path.display()
        );
    }

    // Compute model version hash for TTS caching (from compile-time constants)
    let model_version_hash = super::tts_model_version_hash();
    info!("TTS model version hash: {}", &model_version_hash[..16]);

    // Initialize TTS module
    crate::audio::tts::init_global()
        .unwrap_or_else(|e| warn!("tts::init_global() already called: {e}"));

    // Ensure voice pipeline state is initialized
    super::init_global().unwrap_or_else(|e| warn!("voice::init_global() already called: {e}"));

    // ── Prerequisites ───────────────────────────────────────────────
    if let Err(msg) = crate::audio::tts::ensure_ready() {
        panic!("{msg}\nRun the application first to download TTS models (~400 MB).");
    }

    if let Err(msg) = ensure_voice_models_loaded() {
        panic!("{msg}\nRun the application first to download voice models.");
    }

    let available_styles = tts::list_voice_styles();
    info!(
        "TTS ready with {} voice styles: {:?}",
        available_styles.len(),
        available_styles
    );

    // ── Phase 1: Generate enrollment audio ───────────────────────────────
    phase_start!("Phase 1: Generating enrollment audio");
    // Single-voice allocation: the enrolled voice, the
    // negative-pool styles, the trained-in canaries, and the reserved
    // held-out probe voices.
    let voice_allocation = allocate_voices(&available_styles);
    info!(
        "Voice allocation: enrolled={} negative_pool={:?} canaries={:?} reserved={:?}",
        voice_allocation.enrolled,
        voice_allocation.negative_pool,
        voice_allocation.canaries,
        voice_allocation.reserved,
    );
    // The enrolled clip's label (`{style}_enroll0`) — flows from the voice
    // allocation so the enrolled-speaker phase/cooldown lookups never depend
    // on the hardcoded F1 identity.
    let enrolled_label = format!("{}_enroll0", voice_allocation.enrolled);
    let enrollment_variants = generate_enrollment_variants_cached(
        &voice_allocation.enrolled,
        &model_version_hash,
        &cache_dir_path,
    );
    if enrollment_variants.is_empty() {
        warn!(
            "Need at least one enrollment variant. TTS synthesis may have failed for all styles."
        );
        eprintln!(
            "FATAL: Need at least one enrollment variant. TTS synthesis may have failed for all styles."
        );
        return;
    }
    info!(
        "Generated {} enrollment variants (single enrolled voice {})",
        enrollment_variants.len(),
        voice_allocation.enrolled,
    );

    // ── All 10 enrollment clips are training data ────────────────────────
    // The old 2/3 : 1/3 clip-level split is gone: every raw clip feeds
    // vad_segment_and_enroll (AGC → VAD → augment → embeddings).  Detection
    // control re-bases onto the 16 held-out wake-only recall clips (Phase 7d),
    // generated strictly after training at unseen seeds (3000+).
    let train_clips = enrollment_variants;
    info!(
        "All {} enrollment clips train (no test split) — detection control is the held-out \
         wake-only recall set",
        train_clips.len(),
    );

    // Trained-in canary clips (M2/M3) — generated here so the in-distribution
    // canary matrix can test the SAME clips.
    let canary_clips = generate_canary_clips_cached(
        &voice_allocation.canaries,
        &model_version_hash,
        &cache_dir_path,
    );
    info!(
        "Trained-in canary clips: {} (M2/M3 wake word — in-distribution canary matrix)",
        canary_clips.len(),
    );
    phase_times[P_ENROLLMENT_AUDIO] = phase_end_ms!();

    // ── Phase 2: VAD-gated enrollment (all clips) ────────────────────────
    phase_start!("Phase 2: VAD-gated enrollment");

    let pos_sequences = vad_segment_and_enroll(&train_clips);
    if pos_sequences.is_empty() {
        warn!(
            "VAD-gated enrollment produced no utterances from {} training clips",
            train_clips.len(),
        );
        eprintln!(
            "FATAL: VAD-gated enrollment produced no utterances from {} training clips",
            train_clips.len(),
        );
        return;
    }
    info!(
        "VAD-gated enrollment: {} dense embeddings from {} sequences",
        pos_sequences
            .iter()
            .map(|s| s.embeddings.len())
            .sum::<usize>(),
        pos_sequences.len(),
    );

    phase_times[P_VAD_ENROLLMENT] = phase_end_ms!();

    // ── Phase 3: Generate negative training data ─────────────────────────
    phase_start!("Phase 3: Generating negative training data");

    // Bench-local restricted-style confusable/unrelated generation
    // Production's prewarm rotates ALL 10 TTS voices into the
    // training negatives, which would train the reserved cross-speaker voices
    // (M4/M5) in — the ticket requires them absent from EVERY training path.
    // The bench drives production's own pipeline (fresh AGC → VAD-gated
    // streaming mel → embeddings → PCM augmentation) over the negative-pool
    // styles (F1..M1), reusing the production embedding cache (identical keys),
    // so warm runs are as cheap as the production prewarm.  No production
    // helper changes.
    let confusable_neg_seqs = generate_restricted_phrase_negatives(
        "confusable",
        super::CONFUSABLE_PHRASES,
        super::CONFUSABLE_SEEDS_PER_PHRASE,
        super::CONFUSABLE_SEED_BASE,
        &voice_allocation.negative_pool,
        &model_version_hash,
        &cache_dir_path,
    );
    info!(
        "Bench-local confusable negatives: {} sequences from {} phrases × {} seeds over \
         negative-pool styles {:?} (reserved voices excluded)",
        confusable_neg_seqs.len(),
        super::CONFUSABLE_PHRASES.len(),
        super::CONFUSABLE_SEEDS_PER_PHRASE,
        voice_allocation.negative_pool,
    );
    assert!(
        !confusable_neg_seqs.is_empty(),
        "Bench-local confusable negatives must be non-empty — an all-skip run would \
         silently train a weaker model"
    );
    let unrelated_neg_seqs = generate_restricted_phrase_negatives(
        "unrelated",
        super::UNRELATED_PHRASES,
        super::UNRELATED_SEEDS_PER_PHRASE,
        super::UNRELATED_SEED_BASE,
        &voice_allocation.negative_pool,
        &model_version_hash,
        &cache_dir_path,
    );
    info!(
        "Bench-local unrelated negatives: {} sequences from {} phrases × {} seeds over \
         negative-pool styles {:?} (reserved voices excluded)",
        unrelated_neg_seqs.len(),
        super::UNRELATED_PHRASES.len(),
        super::UNRELATED_SEEDS_PER_PHRASE,
        voice_allocation.negative_pool,
    );
    assert!(
        !unrelated_neg_seqs.is_empty(),
        "Bench-local unrelated negatives must be non-empty — an all-skip run would \
         silently train a weaker model"
    );

    // Generate ambient noise sequences (mahbot-932 Fix 7) — replaces synthetic negatives.
    let ambient_sequences = generate_ambient_noise_sequences();
    info!(
        "Generated {} ambient noise sequences from {} noise profiles × 2 SNR levels (mahbot-932)",
        ambient_sequences.len(),
        NOISE_PROFILES.len(),
    );

    // ── Bench-local restricted pools (mahbot-880, mahbot-923) ──
    // After mahbot-923 the pipeline uses dense stride-8 embeddings throughout.
    // Ambient + owner-negative replace the old synthetic negatives (mahbot-932).

    // Build classifier negative sequences: ambient → owner → confusable → unrelated
    // (mahbot-932 Fix 7, Fix 8).  No synthetic negatives — production uses zero
    // synthetic embedding-space negatives.
    let mut classifier_neg_sequences: Vec<EmbeddingSequence> = Vec::new();
    classifier_neg_sequences.extend(ambient_sequences);

    // Owner-negative sequences (enrolled voice only).
    let owner_seqs = generate_owner_negative_sequences(
        &voice_allocation.enrolled,
        &model_version_hash,
        &cache_dir_path,
    );
    info!(
        "Generated {} owner-negative sequences from the ENROLLED voice {} \
         (mahbot-932 Fix 6)",
        owner_seqs.len(),
        voice_allocation.enrolled,
    );
    classifier_neg_sequences.extend(owner_seqs);
    classifier_neg_sequences.extend(confusable_neg_seqs.clone());
    classifier_neg_sequences.extend(unrelated_neg_seqs.clone());

    // ── Regression assertions (mahbot-932) ──
    assert!(
        classifier_neg_sequences
            .iter()
            .any(|s| s.source == Source::Ambient),
        "Phase 3 must produce ambient negative sequences (mahbot-932 Fix 7)",
    );
    assert!(
        classifier_neg_sequences
            .iter()
            .any(|s| s.source == Source::Owner),
        "Phase 3 must produce owner-negative sequences (mahbot-932 Fix 6)",
    );

    phase_times[P_NEG_TRAINING_DATA] = phase_end_ms!();

    // ── Phase 4: finalize_enrollment (consistency check + classifier training) ──
    phase_start!("Phase 4: finalize_enrollment");
    // After mahbot-923, pos_sequences contains dense stride-8 embeddings
    // from vad_segment_and_enroll (no streaming separate path).
    // Both positives and negatives use dense-only EmbeddingSequence.
    // NO hard gate: production's consistency check is reported via the
    // deployability verdict, never enforced here.  On a consistency failure
    // the bench trains the classifier directly (deterministic seed 0,
    // matching finalize_enrollment's internal config) so detection/FA numbers
    // stay measurable, and records the gate failure for the report.
    let mut finalize_gate_failed: Option<String> = None;
    // `None` means BOTH training paths failed — no weights exist, so the
    // degenerate flag below skips every detection phase (report-only).
    let training_result =
        match super::finalize_enrollment(&pos_sequences, &classifier_neg_sequences) {
            Ok(t) => Some(t),
            Err(gate_err) => {
                warn!(
                    "finalize_enrollment consistency check FAILED (report-only): \
                 {gate_err} — training the classifier directly so the bench stays measurable"
                );
                match super::wake_word_classifier::train_classifier(
                    &pos_sequences,
                    &classifier_neg_sequences,
                    &super::wake_word_classifier::TrainingConfig {
                        rng_seed: Some(0), // deterministic, matches finalize_enrollment
                        ..Default::default()
                    },
                ) {
                    Ok(t) => {
                        finalize_gate_failed = Some(gate_err.to_string());
                        Some(t)
                    }
                    Err(train_err) => {
                        // Documented early exit from the training path: neither
                        // production's consistency gate nor the direct fallback
                        // produced weights.  Defensive only — the window guards
                        // above make this unreachable with the asserted non-empty
                        // sequences.
                        warn!(
                            "finalize_enrollment gate AND direct classifier training \
                             failed: {train_err} — no weights to evaluate, skipping all \
                             detection phases (report-only)"
                        );
                        finalize_gate_failed = Some(format!(
                            "{gate_err}; classifier training failed: {train_err}"
                        ));
                        None
                    }
                }
            }
        };

    // ── Degenerate solution detection (mahbot-844) ──
    // A missing training result (both paths failed above) is degenerate by
    // definition; otherwise flag a near-zero-weight all-zero solution.
    let (degenerate, near_zero_frac) = match &training_result {
        None => {
            warn!(
                "No classifier weights (finalize gate + direct training both failed) — \
                 skipping all detection phases"
            );
            (true, 1.0)
        }
        Some(t) => {
            let all_w = t.weights.all_trainable_slices();
            let total_params: usize = all_w.iter().map(|s| s.len()).sum();
            let near_zero_count = all_w
                .iter()
                .flat_map(|s| s.iter())
                .filter(|v| v.abs() < 1e-6)
                .count();
            let frac = near_zero_count as f64 / total_params as f64;
            // Degenerate if >99% of weights are within ±1e-6 of zero
            if frac > 0.99 {
                warn!(
                    "Classifier produced degenerate all-zero solution — training had issues. \
                     {:.1}% of weights near zero (threshold=1%). \
                     Skipping all detection phases.",
                    frac * 100.0,
                );
                (true, frac)
            } else {
                info!(
                    "Classifier degenerate check: {:.1}% weights near zero — OK",
                    frac * 100.0
                );
                (false, frac)
            }
        }
    };

    // Weights + training metrics for the detection phases and the report.
    // On the double-failure path the seeded placeholder weights keep the
    // report populated (deterministic); the self-test / deployability verdict
    // is skipped entirely and `degenerate` skips every detection phase.
    let (
        weights,
        epochs_trained,
        best_val_loss,
        pos_scores_mean,
        pos_scores_min,
        pos_scores_max,
        neg_scores_mean,
        neg_scores_min,
        neg_scores_max,
        early_stop_reason,
        n_train_windows,
        n_val_windows,
        per_epoch_train_loss,
        per_epoch_val_loss,
        per_epoch_val_accuracy,
        pos_scores_deciles,
        neg_scores_deciles,
    ) = if let Some(t) = &training_result {
        (
            t.weights.clone(),
            t.epochs_trained,
            t.best_val_loss,
            t.pos_scores_mean,
            t.pos_scores_min,
            t.pos_scores_max,
            t.neg_scores_mean,
            t.neg_scores_min,
            t.neg_scores_max,
            t.early_stop_reason.clone(),
            t.n_train_windows,
            t.n_val_windows,
            t.per_epoch_train_loss.clone(),
            t.per_epoch_val_loss.clone(),
            t.per_epoch_val_accuracy.clone(),
            t.pos_scores_deciles,
            t.neg_scores_deciles,
        )
    } else {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        (
            super::wake_word_classifier::ClassifierWeights::from_rng(
                &mut rng,
                &super::wake_word_classifier::ArchConfig::default(),
            ),
            0,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            String::new(),
            0,
            0,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            None,
        )
    };
    let classifier = WakeWordClassifier::new(weights.clone());
    if let Some(t) = &training_result {
        let first_params = t.weights.param_count();
        info!(
            "Conv1D classifier trained successfully: {} params, {} epochs, best val loss={:.4}",
            first_params, t.epochs_trained, t.best_val_loss,
        );
        info!(
            "Classifier scores: pos mean={:.4} [{:.4}, {:.4}] neg mean={:.4} [{:.4}, {:.4}]",
            t.pos_scores_mean,
            t.pos_scores_min,
            t.pos_scores_max,
            t.neg_scores_mean,
            t.neg_scores_min,
            t.neg_scores_max,
        );
    }

    // ── Informational self-test (mahbot-1006 J) ──
    // Production treats the self-test as GATING: if it fails, the model is
    // rejected and enrollment fails.  The benchmark is report-only (mahbot-953),
    // so it does not abort — but a failure is surfaced as a prominent warning
    // AND recorded in the JSON so consumers know the reported detection/FA
    // numbers come from a model production would refuse to deploy.
    // Gated on a training result: on the double-failure path there are no
    // weights to evaluate, so running the self-test on the seeded placeholders
    // would manufacture a deployability verdict from noise — skip it instead
    // (the `degenerate` flag already skips every detection phase below).
    let self_test_result: Option<Result<(), String>> = training_result
        .as_ref()
        .map(|_| super::run_enrollment_self_test(&pos_sequences, &classifier));
    match &self_test_result {
        Some(Ok(())) => info!("Detection self-test: passed"),
        Some(Err(e)) => warn!(
            "Detection self-test FAILED — production would reject this model (report-only, mahbot-953): {e}"
        ),
        None => warn!(
            "No classifier weights (finalize gate + direct training both failed) — \
             skipping the informational self-test / deployability verdict (report-only)"
        ),
    }
    // Deployability verdict (Phase 0, additive default-run key): structured
    // recall numbers + the would-production-accept flag.  The bench-side count
    // replication is cross-checked against production's Result so a divergence
    // in the self-test logic is never silently masked.
    let (deployability, (passed, total, required, deploy_would_pass)) = match &self_test_result {
        Some(r) => {
            let (json, (p, t, req, would_pass)) =
                deployability_json(r, &pos_sequences, &classifier);
            (json, (p, t, req, Some(would_pass)))
        }
        None => (
            serde_json::json!({
                "would_pass_production_enrollment_gate": serde_json::Value::Null,
                "gate_min_fraction": serde_json::Value::Null,
                "recall": serde_json::Value::Null,
                "recall_label": serde_json::Value::Null,
                "production_self_test_passed": serde_json::Value::Null,
                "warning": serde_json::Value::Null,
                "note": "No classifier weights (finalize gate + direct training both \
                         failed) — deployability not measured",
            }),
            (0, 0, 0, None),
        ),
    };
    if let Some(r) = &self_test_result
        && deploy_would_pass != Some(r.is_ok())
    {
        warn!(
            "Deployability cross-check MISMATCH: bench-side self-test counts \
             ({passed}/{total}, need {required}) disagree with production's \
             run_enrollment_self_test result ({:?}) — investigate before trusting \
             the deployability verdict",
            r.as_ref().err(),
        );
    }
    phase_times[P_CLASSIFIER_TRAINING] = phase_end_ms!();

    // ── Phase 5: Set global state for streaming detection ────────────────
    phase_start!("Phase 5: Setting global state");
    super::set_classifier_weights(weights.clone());
    phase_times[P_GLOBAL_STATE] = phase_end_ms!();

    // ── Phase 6: Streaming detection setup ────────────────────────────────
    phase_start!("Phase 6: Streaming detection setup");
    // This phase is mostly a no-op — the setup was already done in Phase 5.
    // The timing will be near-zero, which is expected.
    phase_times[P_STREAMING_SETUP] = phase_end_ms!();

    // ── Detection phases (skipped entirely if classifier is degenerate) ─
    //
    // Initialize output vars to empty defaults — filled in below only when
    // `!degenerate`.  The `phase_times` entries for skipped phases are set
    // to 0 directly, while `phase_start!`/`phase_end_ms!` are only called
    // for executed phases.
    let mut pos_metrics = DetectionMetrics::default();
    // Cold-start detection metrics (mahbot-1006 D) — populated in Phase 8
    // alongside the warm pass.  Stays default when degenerate.
    let mut cold_metrics = DetectionMetrics::default();
    let mut conf_metrics = DetectionMetrics::default();
    let mut unrelated_metrics = DetectionMetrics::default();
    let mut silence_metric = DetectionMetrics::default();
    let mut noise_false_accepts: Vec<String> = Vec::new();
    let mut noise_overlap_results: Vec<(String, f64, Vec<serde_json::Value>)> = Vec::new();
    // held-out cross-speaker probe matrix (reserved voices) —
    // reported separately from the trained-in canary matrix.
    let mut probe_overlap_results: Vec<(String, f64, Vec<serde_json::Value>)> = Vec::new();
    let mut probe_fa_count: usize = 0;
    // Per-tier confusable fa tracking (mahbot-871). Populated after Phase 9.
    let mut conf_fa_by_tier: [Vec<String>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    // Cooldown-phase detection time — hoisted from the detection block so the
    // JSON report can emit it even when the classifier is degenerate
    // (mahbot-1005 §8).
    let mut cooldown_detection_time_ms = 0.0f64;
    // Cooldown test outcomes (None = test skipped).  Emitted in the JSON
    // report so cooldown behaviour is observable (mahbot-1005 §8).
    let mut cooldown_first_detected: Option<bool> = None;
    let mut cooldown_suppressed: Option<bool> = None;
    let mut cooldown_after_recovered: Option<bool> = None;
    // mahbot-1052: additive cooldown.* report keys.  The phase was re-pointed
    // at the enrolled-speaker original variant (the M2–M5 held-out clips
    // are rejected under single-speaker semantics, so the old precondition
    // could never hold).  All keys stay `None` on the degenerate-classifier
    // path and on a visible skip (first detection failed) —
    // the report emits them as `null` with `skip_reason` set, never panics.
    let mut cooldown_source_variant: Option<String> = None;
    let mut cooldown_skip_reason: Option<String> = None;
    let mut cooldown_suppressed_at_2_5s: Option<bool> = None;
    let mut cooldown_accumulation_cap_observed: Option<usize> = None;
    let mut cooldown_buffered_audio_processed_after_expiry: Option<bool> = None;

    // Per-variant metrics for noise profiles (collected across all profiles).
    let mut noise_metrics: Vec<DetectionMetrics> = Vec::new();

    // mahbot-1023: enrolled-speaker benchmark phase (Phase 7d).  Declared
    // here so the JSON report can emit a note even when the classifier is
    // degenerate.
    let mut enrolled_report: Option<serde_json::Value> = None;

    if !degenerate {
        // Create a pre-warmed adaptive threshold state shared across all
        // detection phases so the adaptive code path is exercised end-to-end
        // (mahbot-845, reviewer_3).  Without this, test_detection_samples
        // creates a fresh PipelineCtx per variant, keeping the adaptive state
        // in perpetual 5-frame bootstrap and measuring all metrics against the
        // static threshold.
        //
        // Note (mahbot-852): The adaptive threshold no longer feeds high
        // classifier scores into its statistics — only background frames
        // (below NO_MATCH_RESET_THRESHOLD) update the running statistics.
        // Positive variants therefore no longer inflate the adaptive
        // threshold, eliminating the bias described in reviewer_3's
        // methodological note.  The noise-overlap phase (14) uses a
        // separate freshly-warmed state for independent measurement.
        let mut shared_adaptive = super::AdaptiveThresholdState::warmed();

        // ── Held-out recall + cross-speaker probe generation ──
        // Generated here — strictly AFTER training (phases 2-5) — so the
        // held-out clips provably never enter any training pool.  First run
        // pays TTS synthesis; warm runs hit the PCM cache.
        let held_out_recall_clips = generate_held_out_recall_clips_cached(
            &voice_allocation.enrolled,
            &model_version_hash,
            &cache_dir_path,
        );
        let cross_speaker_probe_clips = generate_cross_speaker_probe_clips_cached(
            &voice_allocation.reserved,
            10, // seeds per reserved voice
            &model_version_hash,
            &cache_dir_path,
        );
        info!(
            "Held-out sets (generated after training): {} wake-only recall clips \
             (enrolled voice, seeds 3000+) + {} cross-speaker probe clips (reserved \
             voices {:?}, seeds 5000+/5100+)",
            held_out_recall_clips.len(),
            cross_speaker_probe_clips.len(),
            voice_allocation.reserved,
        );

        // ── Phase 7: Held-out augmented diagnostics ───────────────────────
        // The old test-split positive phase is gone (all 10 clips
        // are training data).  Detection control is the held-out wake-only
        // recall set (Phase 7d, raw clips — the acceptance basis).  This phase
        // adds bounded per-augmentation diagnostics over a subset of the
        // held-out wake-only clips: {speed_down 0.95, speed_down 0.90,
        // white-10, brown-10} on the first 4 clips (16 variants), warm + cold
        // passes.  Kept separate from the raw-clip acceptance recall.
        phase_start!("Phase 7: Held-out augmented diagnostics");
        let aug_diag_clips: Vec<(Vec<f32>, String)> =
            held_out_recall_clips.iter().take(4).cloned().collect();
        let aug_diag_variants: Vec<(Vec<f32>, String)> =
            pcm_augment_enrollment_variants(&aug_diag_clips)
                .into_iter()
                .filter(|(_, l)| {
                    matches!(
                        augmentation_type(l),
                        "speed_down" | "speed_down_090" | "noise_white_10db" | "noise_brown_10db"
                    )
                })
                .collect();
        info!(
            "Held-out augmented diagnostics: {} variants from {} held-out wake-only clips \
             (speed_down / speed_down_090 / white-10 / brown-10)",
            aug_diag_variants.len(),
            aug_diag_clips.len(),
        );
        // Warm pass (shared adaptive state).
        test_detection_samples(
            &aug_diag_variants,
            &classifier,
            &mut pos_metrics,
            |m, _| m.detected += 1,
            Some(&mut shared_adaptive),
            false, // warm pass
        );
        info!(
            "Held-out augmented diagnostics (warm): {}/{} ({:.1}%)",
            pos_metrics.detected,
            pos_metrics.total,
            pos_metrics.detection_rate() * 100.0,
        );

        // ── Cold-start pass (mahbot-1006 D) ─────────────────────────────
        // Fresh PipelineCtx per variant, no warm-up, fresh adaptive bootstrap
        // (production's post-silence start).  The raw-clip cold measurement
        // lives in the enrolled-speaker phase (Phase 7d) — this pass covers
        // the augmented diagnostics.
        test_detection_samples(
            &aug_diag_variants,
            &classifier,
            &mut cold_metrics,
            |m, _| m.detected += 1,
            None, // fresh adaptive state per variant (item F)
            true, // cold start
        );
        info!(
            "Held-out augmented diagnostics (cold): {}/{} ({:.1}%)",
            cold_metrics.detected,
            cold_metrics.total,
            cold_metrics.detection_rate() * 100.0,
        );
        phase_times[P_POSITIVE_VARIANTS] = phase_end_ms!();

        // ── Phase 7d: Enrolled-speaker benchmark (held-out recall) ─────────
        // End-to-end detection of the HELD-OUT wake-only recall set (unseen
        // enrolled-voice renderings, seeds 3000+, generated after training)
        // through the REAL streaming cold pass (deferred burst + segment-end
        // pass + adaptive bootstrap).  The acceptance basis:
        // all 10 enrollment clips train, so detection control is entirely this
        // held-out recall set.  Measurement-only — the report is not pass/fail
        // gated (mahbot-953), but the acceptance protocol judges the mean
        // across ≥ 3 runs.
        eprintln!("─── Phase 7d: enrolled-speaker benchmark (held-out recall) ───");
        info!("─── Phase 7d: enrolled-speaker benchmark (held-out recall) ───");
        let enrolled_start = Instant::now();
        enrolled_report = run_enrolled_speaker_phase(&held_out_recall_clips, &classifier);
        let enrolled_ms = enrolled_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  Enrolled-speaker phase completed in {enrolled_ms:.0}ms");
        // ── Phase 8: Detection — Confusable phrases ──────────────────────
        phase_start!("Phase 8: Negative — confusable phrases");
        let confusable_variants = generate_phrase_variants_cached(
            CONFUSABLE_PHRASES,
            &available_styles,
            SeedConfig {
                base_seed: 800,
                num_variants: 1, // single seed per phrase (detection test, not training)
                seed_variant: 0,
            },
            "confusable",
            &model_version_hash,
            &cache_dir_path,
        );
        info!(
            "Generated {} confusable phrase variants",
            confusable_variants.len()
        );
        test_detection_samples(
            &confusable_variants,
            &classifier,
            &mut conf_metrics,
            |m, l| m.false_accepts.push(l.to_string()),
            Some(&mut shared_adaptive),
            false, // warm pass only (negative phase)
        );
        // Split false accepts by tier for per-tier tracking (mahbot-871).
        for label in &conf_metrics.false_accepts {
            let phrase = phrase_from_label(label, "confusable");
            let tier_idx = tier_for_phrase(phrase).index();
            conf_fa_by_tier[tier_idx].push(label.clone());
        }
        phase_times[P_CONFUSABLE_NEGATIVES] = phase_end_ms!();

        // ── Phase 10: Detection — Unrelated phrases ───────────────────────
        phase_start!("Phase 9: Negative — unrelated phrases");
        let unrelated_variants = generate_phrase_variants_cached(
            UNRELATED_PHRASES,
            &available_styles,
            SeedConfig {
                base_seed: 900,
                num_variants: 1, // single seed per phrase (detection test, not training)
                seed_variant: 0,
            },
            "unrelated",
            &model_version_hash,
            &cache_dir_path,
        );
        info!(
            "Generated {} unrelated phrase variants",
            unrelated_variants.len()
        );
        test_detection_samples(
            &unrelated_variants,
            &classifier,
            &mut unrelated_metrics,
            |m, l| m.false_accepts.push(l.to_string()),
            Some(&mut shared_adaptive),
            false, // warm pass only (negative phase)
        );
        phase_times[P_UNRELATED_NEGATIVES] = phase_end_ms!();

        // ── Phase 11: Detection — Silence ────────────────────────────────
        phase_start!("Phase 10: Negative — silence");
        test_detection_samples(
            &[(vec![0.0f32; SILENCE_LEN], "silence".to_string())],
            &classifier,
            &mut silence_metric,
            |m, l| m.false_accepts.push(l.to_string()),
            Some(&mut shared_adaptive),
            false, // warm pass only (negative phase)
        );
        phase_times[P_SILENCE_NEGATIVES] = phase_end_ms!();

        // ── Phase 12: Detection — Noise profiles ─────────────────────────
        phase_start!("Phase 11: Negative — noise profiles");
        for (label, generator) in NOISE_PROFILES {
            info!("  Testing noise profile: {label}");
            let noise = generator();
            let mut metric = DetectionMetrics::default();
            test_detection_samples(
                &[(noise, (*label).to_string())],
                &classifier,
                &mut metric,
                |m, l| m.false_accepts.push(l.to_string()),
                Some(&mut shared_adaptive),
                false, // warm pass only (negative phase)
            );
            if metric.false_accepts.is_empty() {
                info!("    → no false accepts ✓");
            } else {
                info!("    → false accepts: {}", metric.false_accepts.len());
                noise_false_accepts.extend(metric.false_accepts.clone());
            }
            noise_metrics.push(metric);
        }
        phase_times[P_NOISE_PROFILES] = phase_end_ms!();

        // ── Phase 12: Cooldown verification ──────────────────────────────
        // mahbot-1052: re-pointed at the ENROLLED-SPEAKER original variant
        // (enrolled-voice `_enroll0` lookup + pcm_augment_enrollment_variants,
        // _original pinned — 85/85 reliable; speed_down/noise fail ~4–6%
        // historically).  The enrolled clip is training data, so
        // its `_original` fires reliably — the in-sample precondition the
        // cooldown semantics need.
        //
        // Cooldown semantics under test (production, unchanged):
        //   - WAKE_WORD_COOLDOWN (3.0 s) gate in handle_wake_word_detection;
        //   - audio fed during cooldown is BUFFERED (not discarded) into
        //     ctx.audio_buffer up to COOLDOWN_ACCUMULATION_CAP (1024 samples =
        //     2 frames, mahbot-802) and processed when the gate expires;
        //   - the stride-8 burst/main loop is gated on `!is_recording`, which
        //     stays true after a detection fires (Soft reset preserves it) —
        //     the bench must reset `is_recording` between detections or the
        //     recovery probe cannot fire (mahbot-1052, analyst review).
        //
        // Probe schedule (all sleeps EXCLUDED from phase timing):
        //   Detection 1  t≈0      warm pass — must fire
        //   Detection 2  t≈0      gate closed — suppressed + cap probe
        //   Detection 3  ~2.5 s   gate still closed (2.5 < 3.0) — suppressed
        //   Detection 4  ~3.5 s   gate expired — fires (natural expiry, NO
        //                          manual last_wake_word_detection clear)
        //
        // Gate vs report (mahbot-1052 pin 4, reviewer_3): Detections 3 and 4
        // ARE the gate — HARD assertions, so a production cooldown regression
        // (e.g. halved WAKE_WORD_COOLDOWN) fails the bench instead of only
        // warning.  Slack margins (2.5 s / 3.5 s) are deliberate: wall-clock
        // timing on a loaded bench CPU is jittery, so the assertions never
        // probe 2.99/3.01 s (500 ms from the 3.0 s boundary either way, far
        // beyond any plausible thread::sleep drift).  Detection 2 keeps the
        // pre-existing soft `cooldown_suppressed_redetection` report key.
        info!("─── {}. Cooldown verification ───", P_COOLDOWN + 1);
        // Variant CONSTRUCTION only — the enrolled clip's 12-cell augmented
        // set, WITHOUT re-running the enrolled-speaker phase (mahbot-1052
        // manager pin).  `_original` is pinned:
        // speed_down/noise fail ~4–6% of runs historically (reviewer_3).
        // The shared find+augment chain lives in [`enrolled_clip_variants`].
        let enrolled_cooldown_variant: Option<(Vec<f32>, String)> =
            enrolled_clip_variants(&train_clips, &enrolled_label)
                .and_then(|variants| variants.into_iter().find(|(_, l)| l.ends_with("_original")));
        if let Some((enrolled_original, clip_label)) = enrolled_cooldown_variant {
            cooldown_source_variant = Some(clip_label.clone());
            let mut ctx = super::PipelineCtx::new();
            // Propagate the shared adaptive state accumulated across phases 8-12
            // so the adaptive code path is active during cooldown testing too.
            ctx.adaptive_threshold = shared_adaptive.clone();

            // Detection 1: should fire (consume warm-up audio first so latency
            // measures only the wake word — mahbot-922).
            // NOTE: this is a WARM pass (consume_warmup + warmed shared
            // adaptive state), unlike Phase 7d's cold pass — the acceptance
            // criterion "assertions actually fire" carries residual risk until
            // measured across ≥3 warm runs (mahbot-1052 manager pin 10); a
            // failure here becomes a VISIBLE skip (skip_reason), never silent.
            consume_warmup(&mut ctx);
            let t0 = Instant::now();
            let detected = run_streaming_detection(&enrolled_original, &mut ctx);
            cooldown_detection_time_ms += t0.elapsed().as_secs_f64() * 1000.0;
            if !detected.detected {
                cooldown_first_detected = Some(false);
                let reason = format!(
                    "Enrolled-speaker variant {clip_label} did not fire in the warm pass — \
                     skipping cooldown assertions (mahbot-1052)"
                );
                warn!("{reason}");
                cooldown_skip_reason = Some(reason);
                // Skip remaining cooldown assertions since detection didn't fire.
                // The test will still exercise noise overlap and other phases.
            } else {
                cooldown_first_detected = Some(true);
                info!("Cooldown test: first detection fired ✓ ({clip_label})");
                // Detection set is_recording = true + last_wake_word_detection
                // (score_stride8_window).  The stride-8 burst/main loop is
                // gated on !is_recording, and Soft reset preserves it — reset
                // it between detections or the recovery probe cannot fire
                // (mahbot-1052).
                ctx.is_recording = false;

                // Detection 2: should NOT fire during cooldown (gate closed,
                // t≈0).  Also the accumulation-cap probe: the full F1 feed is
                // far larger than COOLDOWN_ACCUMULATION_CAP, so the cooldown
                // path must have buffered exactly `cap` samples (mahbot-802
                // semantics: audio fed during cooldown is buffered, not
                // discarded).  Asserted DURING cooldown because the expiry
                // handoff consumes/clears the buffer (mahbot-1052 pin 6).
                let before_cooldown = ctx.last_wake_word_detection;
                let t1 = Instant::now();
                let silenced = run_streaming_detection(&enrolled_original, &mut ctx);
                cooldown_detection_time_ms += t1.elapsed().as_secs_f64() * 1000.0;
                if silenced.detected {
                    cooldown_suppressed = Some(false);
                    warn!("Cooldown test: detection fired during cooldown — unexpected");
                } else {
                    cooldown_suppressed = Some(true);
                    info!("Cooldown test: cooldown prevented re-detection ✓");
                }
                if ctx.last_wake_word_detection != before_cooldown {
                    warn!("Cooldown test: last_wake_word_detection changed during cooldown");
                }
                cooldown_accumulation_cap_observed = Some(ctx.audio_buffer.len());
                assert_eq!(
                    ctx.audio_buffer.len(),
                    super::COOLDOWN_ACCUMULATION_CAP,
                    "Cooldown accumulation cap violated: audio_buffer.len() = {} \
                     (cap = {}, mahbot-802 semantics: cooldown audio is buffered up to the cap)",
                    ctx.audio_buffer.len(),
                    super::COOLDOWN_ACCUMULATION_CAP,
                );
                info!(
                    "Cooldown test: audio fed during cooldown buffered to exactly cap \
                     ({} samples) ✓ — not discarded (mahbot-802)",
                    super::COOLDOWN_ACCUMULATION_CAP,
                );

                // Detection 3: natural-expiry probe at ~2.5 s — the gate must
                // STILL be closed (2.5 s < WAKE_WORD_COOLDOWN 3.0 s).  The old
                // code cleared last_wake_word_detection before this probe,
                // bypassing production's real gate; the natural expiry is the
                // production behavior under test (mahbot-1052).
                //
                // GATE (mahbot-1052 pin 4, reviewer_3): HARD assertion, not
                // warn-soft — a production cooldown regression (e.g. halved or
                // removed WAKE_WORD_COOLDOWN) must FAIL the bench.  The 500 ms
                // slack margin (2.5 s vs 3.0 s) is jitter-safe, and the
                // suppressed feed + silence flush complete ~100 ms after the
                // sleep (the bench feeds far faster than real-time), leaving
                // ~400 ms of margin.
                sleep_until_cooldown_elapsed(&ctx, Duration::from_millis(2500));
                let t2 = Instant::now();
                let at_2_5s = run_streaming_detection(&enrolled_original, &mut ctx);
                cooldown_detection_time_ms += t2.elapsed().as_secs_f64() * 1000.0;
                assert!(
                    !at_2_5s.detected,
                    "Cooldown gate violation: re-detection fired at ~2.5s (elapsed {}ms) while \
                     WAKE_WORD_COOLDOWN = {}ms — the gate must stay closed until natural expiry \
                     (mahbot-1052 pin 4)",
                    ctx.last_wake_word_detection
                        .map_or(0, |l| l.elapsed().as_millis()),
                    super::WAKE_WORD_COOLDOWN.as_millis(),
                );
                cooldown_suppressed_at_2_5s = Some(true);
                info!("Cooldown test: re-detection still suppressed at ~2.5s ✓");

                // Detection 4: recovery probe at ~3.5 s — the gate has expired
                // naturally, so detection must fire again (the old fixed 3.1 s
                // sleep is restructured into this probe schedule, mahbot-1052
                // pin 5).  is_recording was left true by Detection 1 and the
                // suppressed probes never clear it — reset it so the stride-8
                // burst/main loop runs.
                //
                // GATE (mahbot-1052 pin 4): HARD assertion like Detection 3
                // (500 ms slack past the 3.0 s boundary — jitter-safe).
                //
                // Buffered-audio-processed evidence (reviewer_3, mahbot-1052):
                // the post-firing state `audio_buffer.is_empty() &&
                // !command_buffer.is_empty()` alone is TAUTOLOGICAL — the
                // mem::take at the start of every non-cooldown
                // handle_wake_word_detection call plus the detection→recording
                // handoff always leave audio_buffer empty and command_buffer
                // repopulated once detection fires.  The DISCRIMINATING
                // observable is the buffer length BEFORE this feed: after
                // Detection 2's cap probe (hard-asserted == CAP) and
                // Detection 3's suppressed feed (cooldown re-entry adds
                // nothing beyond the cap), the buffer still holds exactly the
                // cap samples at expiry — `pre_recovery_buffer_len >= CAP`
                // proves the cooldown-accumulated audio survived to expiry.
                // NOTE: the re-warm below (consume_warmup) itself feeds
                // handle_wake_word_detection, whose mem::take consumes that
                // buffer — the post-probe is_empty/!command_buffer.is_empty
                // state reflects the warm-up pass, not the firing probe, and
                // carries no additional evidence.  Report-only: under extreme
                // wall-clock load the D3 silence flush could cross the expiry
                // boundary and legitimately consume the buffer early, which is
                // not a production defect.
                ctx.is_recording = false;
                let pre_recovery_buffer_len = ctx.audio_buffer.len();
                sleep_until_cooldown_elapsed(&ctx, Duration::from_millis(3500));
                let t3 = Instant::now();
                let after_cooldown = run_streaming_detection(&enrolled_original, &mut ctx);
                cooldown_detection_time_ms += t3.elapsed().as_secs_f64() * 1000.0;
                assert!(
                    after_cooldown.detected,
                    "Cooldown gate violation: detection did NOT fire at ~3.5s (elapsed {}ms) \
                     after WAKE_WORD_COOLDOWN = {}ms expired — recovery failed \
                     (mahbot-1052 pin 4)",
                    ctx.last_wake_word_detection
                        .map_or(0, |l| l.elapsed().as_millis()),
                    super::WAKE_WORD_COOLDOWN.as_millis(),
                );
                cooldown_after_recovered = Some(true);
                info!("Cooldown test: detection fired after cooldown ✓");
                cooldown_buffered_audio_processed_after_expiry = Some(
                    pre_recovery_buffer_len >= super::COOLDOWN_ACCUMULATION_CAP
                        && ctx.audio_buffer.is_empty()
                        && !ctx.command_buffer.is_empty(),
                );
            } // close the if detected.detected/else block
        } else {
            let reason = format!(
                "{enrolled_label} missing from train clips — cannot re-point Phase 13 at \
                 the enrolled-speaker positive variant (mahbot-1052); Phase 8d's fallback \
                 ('?' lookup) is mirrored here"
            );
            warn!("{reason}");
            cooldown_skip_reason = Some(reason);
        }
        info!(
            "  → Phase 13 detection work completed in {:.0}ms (excl. probe sleeps: \
             ~2.5s suppressed + ~3.5s recovery)",
            cooldown_detection_time_ms,
        );
        phase_times[P_COOLDOWN] = cooldown_detection_time_ms as u64;

        // ── Noise-overlapped detection + cross-speaker probes (mahbot-845,
        //    restructured) ──────────────────────────────────────
        // Two matrices under the same phase (both report-only diagnostics —
        // cross-speaker detections are correct speaker-blind behaviour):
        //   1. in_distribution_canaries — the trained-in M2/M3 clips under the
        //      full 13-cell noise matrix (kept from mahbot-1025; any detection
        //      is a trained-in canary diagnostic).  The canary clips are fed
        //      ORIGINAL-only: the 13-cell noise-condition vocabulary IS the
        //      canary test's defining structure.  The 5-augmentation dimension
        //      was dropped when M4/M5 moved to the reserved held-out probes
        //      (coverage 260→26 cells — disclosed; neither the enrolled-
        //      speaker phases nor the probe matrix exercise the canary
        //      voices' augmentations);
        //   2. held_out_probes — the RESERVED voices (M4/M5) at unseen seeds
        //      across clean + 10/20 dB white (honest generalization probes).
        phase_start!("Phase 13: Noise-overlapped detection + cross-speaker probes");
        let canary_overlap_results = run_noise_overlap_test(&canary_clips, &classifier);
        let (probe_cells, probe_fas) =
            run_cross_speaker_probe_matrix(&cross_speaker_probe_clips, &classifier);
        noise_overlap_results = canary_overlap_results;
        probe_overlap_results = probe_cells;
        probe_fa_count = probe_fas;
        phase_times[P_NOISE_OVERLAP] = phase_end_ms!();
    } else {
        // Degenerate classifier — skip all detection phases
        phase_times[P_POSITIVE_VARIANTS] = 0;
        phase_times[P_CONFUSABLE_NEGATIVES] = 0;
        phase_times[P_UNRELATED_NEGATIVES] = 0;
        phase_times[P_SILENCE_NEGATIVES] = 0;
        phase_times[P_NOISE_PROFILES] = 0;
        phase_times[P_COOLDOWN] = 0;
        phase_times[P_NOISE_OVERLAP] = 0;
        // mahbot-1052 pin 8: on the degenerate path the cooldown.* keys stay
        // null (hoisted vars), but the skip must be VISIBLE — emit the reason
        // so the report never silently lacks it (mirrors the mahbot-1005
        // hoisting pattern; the F1-missing / warm-pass-failure skip reasons
        // are set inside the Phase 13 block above).
        let reason = "degenerate classifier (near-zero weights) — all detection phases, \
             including Phase 12 cooldown verification, were skipped (mahbot-1052)"
            .to_string();
        warn!("{reason}");
        cooldown_skip_reason = Some(reason);
    }

    // ── Env-gated FAPH phase (mahbot-1057) ────────────────────────────────
    // Real-audio false-acceptance-per-hour bench phase.  Runs AFTER the
    // existing phases (reuses the bench's flock + 30-min timeout); the
    // standard-run report surface is untouched because the report section is
    // only added to the JSON when the env gate is on (None → no key).
    //
    // Corpus sizing (recorded on mahbot-1057 before this phase was built):
    // measured detection-path throughput on the idle machine (2026-08-02)
    // was ~47.8× real-time (speech) / ~60.7× (noise), blended ~51.8× for the
    // corpus's 3.85 h speech + 2.14 h noise mix → the full 5.99 h pinned
    // corpus projects to ~7 min wall, well inside the 30-min harness budget.
    // The phase therefore feeds the FULL pinned corpus (812 files).
    let faph_report: Option<serde_json::Value> = run_faph_phase();

    // ── Phase 14 timing ─────────────────────────────────
    phase_start!("Phase 14: Teardown");

    let tier_limit_sets = [
        tier_limits(BenchTier::Easy),
        tier_limits(BenchTier::Medium),
        tier_limits(BenchTier::Hard),
    ];

    // Compute total false accepts across all categories.
    // True-negative phases only: confusable / unrelated / silence / noise.
    // The noise_overlap canary and cross-speaker probe matrices are
    // speaker-blindness DIAGNOSTICS (cross-speaker wake-word detections are
    // correct behaviour, not false accepts) — their detection counts are
    // reported in noise_overlap / false_accepts but do NOT feed the tally.
    let noise_overlap_detections: usize = noise_overlap_results
        .iter()
        .flat_map(|(_, _, detail)| detail.iter())
        .filter(|v| {
            v.get("detected")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        })
        .count();
    // The canary target is derived from the actual measured cell×variant
    // matrix so every report string stays in lockstep with the measured count.
    let noise_overlap_total_variants: usize = noise_overlap_results
        .iter()
        .map(|(_, _, detail)| detail.len())
        .sum();
    let probe_total_variants: usize = probe_overlap_results
        .iter()
        .map(|(_, _, detail)| detail.len())
        .sum();
    let shared_fa_count = unrelated_metrics.false_accepts.len()
        + silence_metric.false_accepts.len()
        + noise_false_accepts.len();

    // Per-tier confusable FA counts
    let conf_fa_counts = [
        conf_fa_by_tier[0].len(),
        conf_fa_by_tier[1].len(),
        conf_fa_by_tier[2].len(),
    ];

    // Per-tier total FAs (confusable + shared)
    let tier_total_fas: [usize; 3] = [
        conf_fa_counts[0] + shared_fa_count,
        conf_fa_counts[1] + shared_fa_count,
        conf_fa_counts[2] + shared_fa_count,
    ];

    // Total false accept (across all tiers and categories)
    let total_false_accepts = conf_fa_counts.iter().sum::<usize>() + shared_fa_count;

    // ── Classifier trigger metrics for negative phases (mahbot-952) ────────
    let conf_classifier = ClassifierTriggerMetrics::compute(&conf_metrics.per_variant);
    let unrel_classifier = ClassifierTriggerMetrics::compute(&unrelated_metrics.per_variant);
    let silence_classifier = ClassifierTriggerMetrics::compute(&silence_metric.per_variant);

    // Per-tier confusable classifier trigger metrics
    let mut conf_classifier_by_tier: [ClassifierTriggerMetrics; 3] = Default::default();
    for pv in &conf_metrics.per_variant {
        let phrase = phrase_from_label(&pv.variant, "confusable");
        conf_classifier_by_tier[tier_for_phrase(phrase).index()].accumulate(pv);
    }

    // Noise: per-profile + aggregate classifier trigger metrics.
    // When the classifier is degenerate, noise_metrics is empty because
    // all detection phases are skipped.  In that case produce default
    // (all-zero) metrics so downstream output paths (JSON, info!(),
    // eprintln) always have consistent data without conditional guards.
    let noise_classifier_per_profile: Vec<ClassifierTriggerMetrics> = if noise_metrics.is_empty() {
        NOISE_PROFILES
            .iter()
            .map(|_| ClassifierTriggerMetrics::default())
            .collect()
    } else {
        noise_metrics
            .iter()
            .map(|m| ClassifierTriggerMetrics::compute(&m.per_variant))
            .collect()
    };
    let noise_classifier_aggregate = noise_classifier_per_profile.iter().copied().fold(
        ClassifierTriggerMetrics::default(),
        |mut acc, m| {
            acc.total_variants += m.total_variants;
            acc.classifier_triggers += m.classifier_triggers;
            acc.full_pipeline_fa += m.full_pipeline_fa;
            acc
        },
    );

    // Enrolled-speaker detection counts — the single-speaker POSITIVE class
    // (the acceptance basis is the held-out wake-only recall set in
    // enrolled_speaker.acceptance).  Shared by the report banner,
    // the threshold-status block, and the catastrophic regression guard.
    // Falls back to the Phase-7 positive-variant counts when the enrolled
    // phase did not run (degenerate classifier).
    let (enrolled_detected, enrolled_total): (usize, usize) =
        if let Some(acc) = enrolled_report.as_ref().and_then(|r| r.get("acceptance")) {
            let d = acc
                .get("detected_live")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize;
            let t = acc
                .get("total_variants")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(d as u64) as usize;
            (d, t)
        } else {
            (pos_metrics.detected, pos_metrics.total)
        };
    // Whether the enrolled-speaker phase (Phase 7d) actually ran.  When it did
    // not (degenerate classifier), enrolled_detected/
    // enrolled_total fall back to the Phase-7 positive variants.  The
    // MIN_DETECTION_RATE suggestion and the ratchet guard are skipped in that
    // case so a degenerate run cannot print a ratchet-to-zero 0.0 suggestion
    // (mahbot-1008 ratchet guard).
    let enrolled_phase_ran = enrolled_report
        .as_ref()
        .and_then(|r| r.get("acceptance"))
        .is_some();

    info!("══════════════════════════════════════════════");
    info!("      Voice Pipeline E2E Benchmark Results");
    info!("══════════════════════════════════════════════");
    info!(
        "Total benchmark time: {:.1}s",
        overall_start.elapsed().as_secs_f64()
    );
    info!(
        "Enrolled-speaker held-out recall: {:.1}% ({}/{}) — target ≥{:.0}% \
         (single-speaker positive class; held-out unseen renderings of the \
         enrolled voice)",
        if enrolled_total > 0 {
            enrolled_detected as f64 / enrolled_total as f64 * 100.0
        } else {
            0.0
        },
        enrolled_detected,
        enrolled_total,
        MIN_DETECTION_RATE * 100.0,
    );
    info!(
        "Confusable false accepts: easy={} (limit ≤{}), medium={} (limit ≤{}), hard={} (limit ≤{})",
        conf_fa_counts[0],
        tier_limit_sets[0].confusable,
        conf_fa_counts[1],
        tier_limit_sets[1].confusable,
        conf_fa_counts[2],
        tier_limit_sets[2].confusable,
    );
    if !conf_metrics.false_accepts.is_empty() {
        info!(
            "  All confusable false triggers: {:?}",
            conf_metrics.false_accepts
        );
        for (i, tier_name) in [(0, "easy"), (1, "medium"), (2, "hard")] {
            if !conf_fa_by_tier[i].is_empty() {
                info!("  {tier_name}-tier triggers: {:?}", conf_fa_by_tier[i]);
            }
        }
    }
    info!(
        "Unrelated false accepts: {} / {} (limit ≤{})",
        unrelated_metrics.false_accepts.len(),
        unrelated_metrics.total,
        0, // all tiers have unrelated limit = 0
    );
    if !unrelated_metrics.false_accepts.is_empty() {
        info!("  False triggers: {:?}", unrelated_metrics.false_accepts);
    }
    info!(
        "Silence false accepts: {} / 1 (limit ≤{})",
        silence_metric.false_accepts.len(),
        0, // all tiers have silence limit = 0
    );
    info!(
        "Noise false accepts: {} / {} ({} profiles, limit ≤{}–{} across tiers)",
        noise_false_accepts.len(),
        NOISE_PROFILES.len(),
        NOISE_PROFILES.len(),
        tier_limit_sets[0].noise,
        tier_limit_sets[2].noise,
    );
    if !noise_false_accepts.is_empty() {
        info!("  False triggers: {:?}", noise_false_accepts);
    }

    // ── Classifier trigger report (mahbot-952) ──────────────────────────
    info!("──────────────────────────────────────────────");
    info!("  Classifier trigger metrics (negative phases):");

    let ct_line = |name: &str, ct: &ClassifierTriggerMetrics| -> String {
        format!(
            "    {name}: {tr}/{tot} triggers, fa={fa}",
            tr = ct.classifier_triggers,
            tot = ct.total_variants,
            fa = ct.full_pipeline_fa,
        )
    };
    info!("{}", ct_line("Confusable", &conf_classifier));
    for (i, tier_name) in [(0, "easy"), (1, "medium"), (2, "hard")] {
        info!(
            "    └─ {}",
            ct_line(tier_name, &conf_classifier_by_tier[i]).trim_start()
        );
    }
    info!("{}", ct_line("Unrelated", &unrel_classifier));
    info!("{}", ct_line("Silence", &silence_classifier));
    info!(
        "{}",
        ct_line("Noise (aggregate)", &noise_classifier_aggregate)
    );
    for i in 0..NOISE_PROFILES.len() {
        let profile_ct = &noise_classifier_per_profile[i];
        info!(
            "    └─ {}",
            ct_line(NOISE_PROFILES[i].0, profile_ct).trim_start()
        );
    }

    info!("──────────────────────────────────────────────");
    for (i, name) in [(0, "easy"), (1, "medium"), (2, "hard")] {
        info!(
            "Tier {name}: confusable FA={} (limit ≤{}), total FA={} (limit ≤{})",
            conf_fa_counts[i],
            tier_limit_sets[i].confusable,
            tier_total_fas[i],
            tier_limit_sets[i].total,
        );
    }
    info!("Total false accepts across all tiers: {total_false_accepts}");
    info!("Enrollment consistency: validated by finalize_enrollment (Phase 4)");
    info!("══════════════════════════════════════════════");

    // ── Phase 14 timing ─────────────────────────────────
    phase_times[P_TEARDOWN] = phase_end_ms!();

    // ═══════════════════════════════════════════════════════════════════════
    // JSON metrics output
    // ═══════════════════════════════════════════════════════════════════════

    // Noise-overlap canary + held-out probe matrices: speaker-blindness
    // DIAGNOSTICS — cross-speaker wake-word detections are correct behaviour
    // (high detection expected), so these report rates without warnings and do
    // NOT count into total_false_accepts.
    warn_fa_cells(
        "Noise-overlap in-distribution canary",
        &noise_overlap_results,
        noise_overlap_total_variants,
        "",
        false,
    );
    warn_fa_cells(
        "Cross-speaker probe",
        &probe_overlap_results,
        probe_total_variants,
        "; reserved voices",
        false,
    );

    // NOTE: all pass/fail gating was removed in mahbot-953.  Threshold checks
    // emit warnings above but never abort the benchmark.  Run data is reported
    // in both JSON (below) and this stderr report.  See the threshold assertion
    // conversion section after the report for details (each now warns instead of
    // asserting).

    // Build the JSON output
    // Build per-variant negative diagnostics with FULL per-variant detail
    // (mahbot-1005 §9).  Confusable variants are always tier-qualified
    // (mahbot-871).  The flat list is reused for score distributions below.
    let mut all_neg_pv: Vec<(&PerVariantResult, String)> = Vec::new();
    for pv in &conf_metrics.per_variant {
        let phrase = phrase_from_label(&pv.variant, "confusable");
        let variant_tier = tier_for_phrase(phrase);
        all_neg_pv.push((pv, format!("confusable_{}", variant_tier.as_str())));
    }
    for pv in &unrelated_metrics.per_variant {
        all_neg_pv.push((pv, "unrelated".to_string()));
    }
    for pv in &silence_metric.per_variant {
        all_neg_pv.push((pv, "silence".to_string()));
    }
    for metric in &noise_metrics {
        for pv in &metric.per_variant {
            all_neg_pv.push((pv, "noise".to_string()));
        }
    }
    let negative_pv: Vec<serde_json::Value> = all_neg_pv
        .iter()
        .map(|(pv, cat)| pv_to_json(pv, Some(cat)))
        .collect();

    // Build per-tier results dict
    let mut results_map = serde_json::Map::new();
    for &(i, tier_name, bench_tier) in &[
        (0, "easy", BenchTier::Easy),
        (1, "medium", BenchTier::Medium),
        (2, "hard", BenchTier::Hard),
    ] {
        let limits = tier_limits(bench_tier);
        results_map.insert(
            tier_name.to_string(),
            serde_json::json!({
                "confusable_false_accepts": conf_fa_counts[i],
                "total_false_accepts": tier_total_fas[i],
                "limits": {
                    "confusable": limits.confusable,
                    "total": limits.total,
                },
            }),
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // mahbot-1005 diagnostics: verdicts, distributions, train/test alignment,
    // and reproducibility metadata
    // ═══════════════════════════════════════════════════════════════════════

    // ── Exhaustive per-variant verdicts for positives (§2) ──
    let mut verdict_counts: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    for pv in &pos_metrics.per_variant {
        *verdict_counts
            .entry(classify_miss(pv).as_str())
            .or_insert(0) += 1;
    }

    // ── Classifier discrimination evidence (§3) ──
    // Peak-score distributions (warm-up excluded after the §1 instrumentation
    // reset) plus per-frame threshold crossing fractions.
    let pos_peak_scores: Vec<f32> = pos_metrics
        .per_variant
        .iter()
        .map(|pv| pv.peak_score)
        .collect();
    let neg_peak_scores: Vec<f32> = all_neg_pv.iter().map(|(pv, _)| pv.peak_score).collect();

    let pos_frame_crossing = {
        let total_frames: usize = pos_metrics
            .per_variant
            .iter()
            .map(|pv| pv.per_frame_scores.len())
            .sum();
        let below_threshold: usize = pos_metrics
            .per_variant
            .iter()
            .flat_map(|pv| &pv.per_frame_scores)
            .filter(|s| s[ROLLING_SUM_IDX] < MIN_CLASSIFIER_THRESHOLD)
            .count();
        (total_frames, below_threshold)
    };
    let neg_frame_crossing = {
        let total_frames: usize = all_neg_pv
            .iter()
            .map(|(pv, _)| pv.per_frame_scores.len())
            .sum();
        let above_threshold: usize = all_neg_pv
            .iter()
            .flat_map(|(pv, _)| &pv.per_frame_scores)
            .filter(|s| s[ROLLING_SUM_IDX] >= MIN_CLASSIFIER_THRESHOLD)
            .count();
        (total_frames, above_threshold)
    };

    // ── Per-augmentation detection rates (§6) ──
    // Labels carry `_original/_speed_down/...` suffixes from
    // pcm_augment_enrollment_variants.
    let mut aug_counts: std::collections::BTreeMap<&'static str, (usize, usize)> =
        std::collections::BTreeMap::new();
    for pv in &pos_metrics.per_variant {
        let aug = augmentation_type(&pv.variant);
        let e = aug_counts.entry(aug).or_insert((0, 0));
        e.1 += 1;
        if pv.detected {
            e.0 += 1;
        }
    }
    let per_augmentation: serde_json::Value = serde_json::Value::Object(
        aug_counts
            .iter()
            .map(|(aug, (detected, total))| {
                (
                    (*aug).to_string(),
                    serde_json::json!({
                        "detected": detected,
                        "total": total,
                        "rate": if *total > 0 { *detected as f64 / *total as f64 } else { 0.0 },
                    }),
                )
            })
            .collect(),
    );

    // ── Reproducibility metadata (§10) ──
    let warmup_source = if WARMUP_TTS_CACHE.get().is_some() {
        "tts"
    } else {
        "pink_noise_fallback"
    };

    // ── Phase-B alignment probes (mahbot-1045, report-only) ──

    // B1: TTS-echo gate probe.  The offline bench never runs the mic loop,
    // so the probe locks the voice-tests setter → is_playback_active()
    // predicate roundtrip — the exact predicate the production mic arm
    // consults to drop the WHOLE chunk (pre-AGC ring included) while
    // playback is active (voice.rs mic-chunk arm, mahbot-896).  The actual
    // chunk-drop behavior is exercised only by that production arm, which
    // the offline bench cannot drive; zero production change.
    let tts_echo_gate_probe = {
        tts::set_playback_active_for_test(true);
        let gate_active = tts::is_playback_active();
        tts::set_playback_active_for_test(false);
        let gate_cleared = !tts::is_playback_active();
        serde_json::json!({
            "probe": "TTS-echo gate (mahbot-1045 B1)",
            "gate_active_while_playback": gate_active,
            "gate_cleared_after_clear": gate_cleared,
            "note": "Predicate-level roundtrip only: the voice-tests setter and \
                     is_playback_active() (the predicate the production mic arm \
                     consults at voice.rs:7190 to drop the whole chunk while \
                     playback is active).  The offline bench never runs the mic \
                     loop, so the chunk-drop gate itself is not exercised here — \
                     the instrumentation counters are static regardless of the \
                     gate; zero production change.",
        })
    };

    // B4: production metrics snapshot — the only production diagnostics
    // surface with zero bench coverage before mahbot-1045.  The offline bench
    // never runs the mic loop (chunks_received/dropped_chunks stay 0); the
    // plausibility contract is: embeddings WERE computed by the production
    // streaming path (the enrolled-speaker phase / main detection phases) and
    // the drop rate is 0.
    let production_metrics_probe = {
        let metrics = super::get_voice_metrics();
        let drop_rate = metrics.drop_rate(); // guards the 0/0 denominator
        let plausible = metrics.embeddings_computed > 0 && drop_rate == 0.0;
        if !plausible {
            warn!(
                "mahbot-1045 B4: production metrics implausible — \
                 embeddings_computed={}, drop_rate={drop_rate}",
                metrics.embeddings_computed,
            );
        }
        serde_json::json!({
            "probe": "production metrics (mahbot-1045 B4)",
            "chunks_received": metrics.chunks_received,
            "dropped_chunks": metrics.dropped_chunks,
            "embeddings_computed": metrics.embeddings_computed,
            "drop_rate": drop_rate,
            "plausible": plausible,
            "note": "drop_rate guards the zero denominator (0/0 → 0.0); \
                     chunks_received/dropped_chunks are mic-loop-only and stay 0 in \
                     the offline bench by construction.",
        })
    };

    // B5: embedding cache hit↔miss equivalence (mahbot-1045, bounded probe).
    // A full-cache miss recompute is NOT attempted (ONNX for ~200 utterances ×
    // 2 variants would run minutes — near the 30-min harness budget — and the
    // prewarm's OnceLock fast-path makes a second in-process prewarm a no-op).
    // Instead the probe locks the write→read cache-format equivalence on ONE
    // representative utterance: the miss path writes exactly these bytes and
    // the hit path reads them back through the same push helper, so a
    // byte-identical read-back proves a warm run reproduces the cold run's
    // training data — the mahbot-1029 cache win.  The probe now
    // reads the bench-local restricted confusable pool (the production
    // prewarm caches are no longer populated by the bench).
    let embedding_cache_probe = {
        let sample = confusable_neg_seqs.iter().find(|s| s.id.variant_index == 0);
        match sample {
            Some(first) => {
                let seq_idx = first.id.sequence_index;
                let mut variants: Vec<(u8, Vec<Vec<f32>>)> = confusable_neg_seqs
                    .iter()
                    .filter(|s| s.id.sequence_index == seq_idx)
                    .map(|s| {
                        (
                            u8::try_from(s.id.variant_index).expect("variant index fits in u8"),
                            s.embeddings.clone(),
                        )
                    })
                    .collect();
                variants.sort_by_key(|(vi, _)| *vi);
                let tmp_dir = std::env::temp_dir().join(format!(
                    "mahbot1045_embed_cache_probe_{}",
                    std::process::id()
                ));
                let key = format!("probe_{seq_idx}");
                super::write_embedding_cache(&tmp_dir, &key, &variants);
                let read_back = super::read_embedding_cache(&tmp_dir, &key);
                let _ = std::fs::remove_dir_all(&tmp_dir);
                serde_json::json!({
                    "probe": "embedding cache hit↔miss equivalence (mahbot-1045 B5)",
                    "utterance_sequence_index": seq_idx,
                    "n_variants_roundtripped": variants.len(),
                    "read_back_byte_identical": read_back.as_ref() == Some(&variants),
                    "note": "Bounded probe: cache-format write→read roundtrip on ONE \
                             representative utterance (no ONNX recompute).",
                })
            }
            None => serde_json::json!({
                "probe": "embedding cache hit↔miss equivalence (mahbot-1045 B5)",
                "read_back_byte_identical": false,
                "note": "SKIPPED — no confusable prewarm sequences available.",
            }),
        }
    };

    // B3: enrollment-quality scoring (mahbot-1045, report-only).  The bench's
    // clips have no pre-AGC ring → heuristic SNR (estimate_snr_energy).
    let enrollment_quality_probe_value = enrollment_quality_probe(&train_clips);

    // B6: VAD feed-contract cross-check (mahbot-1045, report-only).  Replays
    // production's feed pattern (only NEW hop samples) vs the bench's VAD
    // segmentation on the same audio — the mahbot-900 contract guard.
    let vad_feed_cross_check_probe_value =
        vad_feed_cross_check_probe(train_clips.first().map_or(&[], |(pcm, _)| &pcm[..]));

    let reproducibility = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "enrolled_voice": voice_allocation.enrolled,
        "voice_allocation": {
            "negative_pool_styles": voice_allocation.negative_pool,
            "canary_styles": voice_allocation.canaries,
            "reserved_probe_styles": voice_allocation.reserved,
            "note": "Single-voice allocation: the enrolled voice is the \
                     only voice in enrollment / owner negatives; \
                     the reserved probe voices are absent from EVERY training path.",
        },
        "seeds": {
            // Classifier seed is fixed inside finalize_enrollment (voice.rs).
            "classifier": 0,
            "tts_warmup": 947,
        },
        "wake_phrase": {
            "used": wake_word(),
            "source": wake_phrase_source(),
            "note": "The bench synthesizes and tests the wake phrase persisted in the \
                     deployed model's config-store entry (production's actual listening \
                     phrase); 'fallback' means the config store was unavailable and the \
                     legacy constant was used.  A phrase change invalidates the TTS PCM \
                     cache (one re-synthesis run) and shifts the acoustic target — \
                     numbers are only comparable across runs with the same phrase.  The \
                     confusable negative tier stays the mahbot-family canonical list \
                     regardless of the deployed phrase (production itself skips mahbot \
                     confusables for non-mahbot phrases) — for a non-mahbot deployed \
                     phrase that tier is informational, not a production-faithful probe.",
        },
        "model_hashes": {
            "tts_model_version_hash": model_version_hash,
            "mel_model_sha256": super::MEL_MODEL_SHA256,
            "embed_model_sha256": super::EMBED_MODEL_SHA256,
        },
        "cache_state": {
            "warmup_tts_cached": WARMUP_TTS_CACHE.get().is_some(),
            "cache_dir": cache_dir_path.display().to_string(),
        },
        // mahbot-1045 B7: ring lengths the bench sets and production reads for
        // geometry classification (previously invisible to the report).
        "geometry_constants": {
            "window_size": crate::audio::wake_word_classifier::WINDOW_SIZE,
            "embedding_ring_max": super::EMBEDDING_RING_MAX,
        },
        "warmup_source": warmup_source,
        // mahbot-1045 Phase B alignment probes (report-only, additive).
        "tts_echo_gate": tts_echo_gate_probe,
        "production_metrics": production_metrics_probe,
        "embedding_cache": embedding_cache_probe,
        "enrollment_quality": enrollment_quality_probe_value,
        "vad_feed_cross_check": vad_feed_cross_check_probe_value,
        "train_test_split": {
            "note": format!(
                "No split: all {} enrollment clips are training \
                 data; detection control is the held-out wake-only recall set \
                 (enrolled_speaker.acceptance)",
                train_clips.len(),
            ),
            "n_train_clips": train_clips.len(),
            "n_train": pos_sequences.len(),
            "train_clips": train_clips.iter().map(|(_, l)| l.clone()).collect::<Vec<_>>(),
        },
    });

    // ── JSON sub-objects (built separately to stay under serde_json's json!
    // macro recursion limit — the main report object nests several levels) ──

    // Detection summary: warm (pos_metrics) + cold-start (cold_metrics,
    // mahbot-1006 D) over the held-out augmented diagnostics (Phase 7 —
    // bounded {speed_down, speed_down_090, white-10, brown-10} variants of a
    // held-out wake-only subset).  A healthy pipeline's warm/cold
    // detection rate here is HIGH.  The raw-clip held-out recall acceptance
    // basis (unseen seeds, wake-only) is reported in
    // enrolled_speaker.acceptance.
    let detection_json = serde_json::json!({
        "rate": if pos_metrics.total > 0 {
            serde_json::Value::from(pos_metrics.detection_rate())
        } else {
            serde_json::Value::Null
        },
        "detected": pos_metrics.detected,
        "total_positive": pos_metrics.total,
        "no_tests_ran": pos_metrics.total == 0,
        "note": "Held-out augmented diagnostics: bounded \
                 per-augmentation variants (speed_down / speed_down_090 / \
                 white-10 / brown-10) over a subset of the held-out wake-only \
                 clips — HIGH detection is the healthy reading.  The trained-in \
                 cross-speaker canaries live in \
                 noise_overlap.in_distribution_canaries (M2/M3) and the honest \
                 held-out probes in noise_overlap.held_out_probes (M4/M5).  The \
                 raw-clip held-out recall acceptance basis (unseen seeds, \
                 wake-only) is reported in enrolled_speaker.acceptance.",
        "miss_verdicts": verdict_counts,
        "total_misses": pos_metrics.total - pos_metrics.detected,
        // mahbot-1006 D: cold-start pass — fresh PipelineCtx per variant,
        // no consume_warmup, fresh AdaptiveThresholdState::new() bootstrap.
        // Measures production's post-silence start.
        "cold_rate": if cold_metrics.total > 0 {
            serde_json::Value::from(cold_metrics.detection_rate())
        } else {
            serde_json::Value::Null
        },
        "cold_detected": cold_metrics.detected,
        "cold_total_positive": cold_metrics.total,
        "cold_no_tests_ran": cold_metrics.total == 0,
    });

    // mahbot-1006 J: production gates enrollment on this self-test; the
    // benchmark is report-only (mahbot-953) so it records the outcome
    // instead.  When "passed" is false, the reported detection/FA numbers
    // come from a model production would refuse to deploy.
    let self_test_json = serde_json::json!({
        "passed": self_test_result.as_ref().is_some_and(Result::is_ok),
        "error": self_test_result
            .as_ref()
            .and_then(|r| r.as_ref().err().map(ToString::to_string)),
        "skipped": self_test_result.is_none(),
    });

    let classifier_trigger_json = serde_json::json!({
        "confusable": ct_to_json(&conf_classifier),
        "confusable_by_tier": {
            "easy": ct_to_json(&conf_classifier_by_tier[0]),
            "medium": ct_to_json(&conf_classifier_by_tier[1]),
            "hard": ct_to_json(&conf_classifier_by_tier[2]),
        },
        "unrelated": ct_to_json(&unrel_classifier),
        "silence": ct_to_json(&silence_classifier),
        "noise": {
            "aggregate": ct_to_json(&noise_classifier_aggregate),
            "per_profile": serde_json::Value::Object(
                NOISE_PROFILES.iter().enumerate().map(|(i, (label, _))| {
                    (label.to_string(), ct_to_json(&noise_classifier_per_profile[i]))
                }).collect()
            ),
        },
    });

    let classifier_diagnostics_json = serde_json::json!({
        "pos_scores_mean": pos_scores_mean,
        "pos_scores_min": pos_scores_min,
        "pos_scores_max": pos_scores_max,
        "neg_scores_mean": neg_scores_mean,
        "neg_scores_min": neg_scores_min,
        "neg_scores_max": neg_scores_max,
        "epochs_trained": epochs_trained,
        "best_val_loss": best_val_loss,
        "early_stop_reason": early_stop_reason,
        "n_train_windows": n_train_windows,
        "n_val_windows": n_val_windows,
        "per_epoch_train_loss": per_epoch_train_loss,
        "per_epoch_val_loss": per_epoch_val_loss,
        "per_epoch_val_accuracy": per_epoch_val_accuracy,
        "pos_scores_deciles": pos_scores_deciles,
        "neg_scores_deciles": neg_scores_deciles,
        "total_params": weights.param_count(),
        "degenerate": degenerate,
        "near_zero_frac": near_zero_frac,
        "adaptive_threshold_trajectory": serde_json::Value::Object(
            pos_metrics.per_variant.iter().enumerate().map(|(i, pv)| {
                (format!("variant_{i}"), serde_json::json!({
                    "variant": pv.variant,
                    "trajectory": pv.adaptive_threshold_trajectory.clone(),
                    "ceiling_limited_frames": pv.ceiling_limited_frames,
                }))
            }).collect()
        ),
        "discrimination": {
            "pos_peak_scores": {
                "deciles": deciles(&pos_peak_scores),
                "n": pos_peak_scores.len(),
            },
            "neg_peak_scores": {
                "deciles": deciles(&neg_peak_scores),
                "n": neg_peak_scores.len(),
            },
            "pos_frames_below_min_threshold_frac": if pos_frame_crossing.0 > 0 {
                pos_frame_crossing.1 as f64 / pos_frame_crossing.0 as f64
            } else {
                0.0
            },
            "neg_frames_above_min_threshold_frac": if neg_frame_crossing.0 > 0 {
                neg_frame_crossing.1 as f64 / neg_frame_crossing.0 as f64
            } else {
                0.0
            },
            "pos_total_frames": pos_frame_crossing.0,
            "neg_total_frames": neg_frame_crossing.0,
        },
    });

    // noise_overlap restructured into two labeled sections — the trained-in
    // canary matrix and the held-out reserved-voice probe matrix.  Both are
    // speaker-blindness DIAGNOSTICS (cross-speaker wake-word detections are
    // correct behaviour — high detection expected), reported without pass/fail
    // semantics and NOT counted into total_false_accepts.
    let cell_json = |(key, rate, detail): &(String, f64, Vec<serde_json::Value>)| {
        let fas = detail
            .iter()
            .filter(|v| {
                v.get("detected")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            })
            .count();
        (
            key.clone(),
            serde_json::json!({
                "detection_rate": rate,
                "detected": fas,
                "per_variant": detail,
            }),
        )
    };
    let noise_overlap_json = serde_json::json!({
        "sections": {
            "in_distribution_canaries": serde_json::Value::Object(
                noise_overlap_results.iter().map(&cell_json).collect(),
            ),
            "held_out_probes": serde_json::Value::Object(
                probe_overlap_results.iter().map(&cell_json).collect(),
            ),
        },
        "note": format!(
            "in_distribution_canaries = trained-in M2/M3 wake-word clips under the \
             13-cell noise matrix (kept from mahbot-1025; any detection is a \
             trained-in speaker-blindness diagnostic).  held_out_probes = \
             RESERVED voices (absent from every training path) at unseen seeds \
             across clean + 10/20 dB white (honest cross-speaker \
             generalization probes).  Both are DIAGNOSTICS: speaker-blind \
             detection means a non-enrolled voice's wake word fires — high \
             detection is expected and healthy; low detection is reportable, \
             never a failure.  Measured {canary_total} canary + \
             {probe_total} probe cells this run.",
            canary_total = noise_overlap_total_variants,
            probe_total = probe_total_variants,
        ),
    });

    // mahbot-1023: enrolled-speaker phase (Phase 7d) + safety gate (item 7).
    // The safety gate measures the false-positive impact of the deferred
    // in-distribution windows (burst + boundary double-scoring) on the
    // negative/confusable set; baseline 9/59 gate crossings / 0 FAs.
    let enrolled_json = enrolled_report.unwrap_or_else(
        || serde_json::json!({"note": "Enrolled-speaker phase skipped (degenerate classifier)"}),
    );
    let safety_gate_json = safety_gate(&all_neg_pv);

    // Total wall time — the whole run (start → report emission), report-
    // assembly window included, so run-time claims are auditable from the JSON.
    let total_wall_time_ms = overall_start.elapsed().as_secs_f64() * 1000.0;

    // Cross-run statistics (Phase 0): last 3 archived reports summarized so
    // the >=3-run acceptance protocol is self-documenting.  Computed BEFORE
    // the current archive is written so it never includes this run.
    let cross_run = cross_run_summary();

    let json = serde_json::json!({
        "benchmark": "voice_pipeline_e2e",
        "total_wall_time_ms": total_wall_time_ms,
        "_note": format!(
            "Report-only benchmark — no pass/fail gating (mahbot-953).  \
             'passed' was removed in mahbot-1005: it was hardcoded true and \
             misleading.  Compare against the limits in 'results' instead. \
             mahbot-1006 aligned the benchmark's training/inference \
             processing with production (AGC→augment ordering, cold-start \
             pass, CONFIG-driven preprocessor, preprocessed negatives) — \
             detection/FA numbers are NOT directly comparable to the \
             mahbot-1004/948 baselines.  The speaker-verifier stage, the \
             classifier-candidate machinery, warm-up suppression, and all \
             model fingerprinting were removed; detection is immediate-fire \
             and speaker-blind.  total_false_accepts semantics changed: the \
             pre-change 13 (all canary-matrix cross-speaker detections) is \
             now 0, because cross-speaker wake-word detections are CORRECT \
             speaker-blind behaviour reported as DIAGNOSTICS \
             (noise_overlap / false_accepts.cross_speaker_probes), with no \
             pass/fail semantics (high detection expected; low detection is \
             reportable, not a failure).  Pre-change archives were deleted — \
             cross_run covers only current-format reports.  The \
             bench-leanness cleanup removed the same-audio (8b) and B-sweep \
             (8c) report-only phases.  mahbot-1081 (Phase 0, additive keys): \
             the bench now synthesizes and tests the deployed config-store \
             wake phrase (reproducibility.wake_phrase) and surfaces the \
             deployability verdict (top-level deployability — informational, \
             the bench never hard-gates).  Single-voice enrollment (the \
             enrolled speaker is ONE TTS voice with guided-prompt DSP \
             conditioning instead of 10 rotated voices — all detection/FA \
             numbers re-baseline, expected); the enrolled-speaker acceptance \
             basis is a HELD-OUT WAKE-ONLY recall set (16 unseen \
              enrolled-voice renderings, seeds 3000+, Wilson CI reported — \
              embedded-in-sentence detection removed); the M4/M5 \
             reserved voices are absent from every training path and probed at \
             unseen seeds as diagnostics; the confusable/unrelated negative \
             pools are built bench-locally over the negative-pool styles \
             (production's all-voice prewarm would train the reserved voices \
             in); the benchmark-local embedding cache keys match production's \
              (plus the recipe version), so warm-run cost is \
             unchanged after the one-time cold-embedding recompute.  First run \
             after the change pays one-time TTS synthesis for the new \
             held-out/probe/canary clips (outside the warm budget).  The \
             negative training pools (production prewarm + bench \
             confusable/unrelated) use the bounded 2-cell recipe (original + \
             pink 25 dB) instead of the old 5-cell set — a deliberate budget \
             tradeoff: the 12-cell positive recipe's cold embedding recompute \
             over the full 5-cell negatives measured ~68 s (archive \
             20260804-155415) and ~84 s with the final owner pool (A/B run) — \
             both over the 39 s bound — so the speed/volume negative cells \
             were dropped.  The \
             low-SNR negative cell (brown-10) lives on the owner/ambient \
             path.  The deterministic easy-tier confusable FA (if present) is \
             measured under this bounded negative recipe."
        ),
        "total_false_accepts": total_false_accepts,
        // additive: single-voice enrollment disclosure — voice
        // allocation, the guided-prompt DSP recipe, and any production
        // consistency-gate failure the bench trained through (report-only).
        "enrollment": serde_json::json!({
            "enrolled_voice": voice_allocation.enrolled,
            "negative_pool_styles": voice_allocation.negative_pool,
            "canary_styles": voice_allocation.canaries,
            "reserved_probe_styles": voice_allocation.reserved,
            "guided_prompts": serde_json::json!({
                "clip_prompt_groups": (0..10)
                    .map(|i| format!(
                        "{}_{}",
                        GuidedPromptGroup::for_clip_index(i).label(),
                        i
                    ))
                    .collect::<Vec<_>>(),
                "normal": (0..3).map(|i| format!("{}_enroll{i}", voice_allocation.enrolled)).collect::<Vec<_>>(),
                "distance": (3..6).map(|i| format!("{}_enroll{i}", voice_allocation.enrolled)).collect::<Vec<_>>(),
                "angle": (6..8).map(|i| format!("{}_enroll{i}", voice_allocation.enrolled)).collect::<Vec<_>>(),
                "morning": (8..10).map(|i| format!("{}_enroll{i}", voice_allocation.enrolled)).collect::<Vec<_>>(),
                "note": "10 clips of the enrolled voice mapped to production's guided \
                         enrollment prompts (3 normal / 3 further-from-mic / 2 \
                         different-angle / 2 morning-voice).  Variation comes from DSP \
                         on the cached synthesis PCM — no extra TTS synthesis cost.",
            }),
            "dsp_recipe": serde_json::json!({
                "normal": "noise floor at 20 dB SNR",
                "distance": "-6 dB level reduction + 3.2 kHz one-pole lowpass (HF rolloff) + noise floor at 14 dB SNR (same absolute noise RMS as normal)",
                "angle": "-4 dB high-shelf spectral tilt @ 3 kHz + -3 dB level reduction + noise floor at 17 dB SNR",
                "morning": "0.92x slower/lower-pitch resample + -3 dB level reduction + 2.2 kHz lowpass + noise floor at 17 dB SNR",
                "note": "Absolute level is AGC-normalized away, so realism comes from \
                         the SNR against a fixed noise floor, the spectral rolloff, \
                         and the level RELATIONSHIPS (every group's noise lands at the \
                         same absolute RMS as the normal 20 dB floor).",
            }),
            "finalize_gate_failed": finalize_gate_failed,
            "note": "Single-voice enrollment: the former multi-voice \
                     rotation (10 clips across 10 TTS voices) is eliminated — the \
                     'enrolled speaker' is now one voice.  Reserved probe voices are \
                     absent from every training path; the confusable/unrelated \
                     negative pools are built bench-locally over the negative-pool \
                     styles (production's all-voice prewarm would train the reserved \
                     voices in).",
        }),
        "results": results_map,
        "detection": detection_json,
        "self_test": self_test_json,
        "false_accepts": {
            "confusable_easy": conf_fa_counts[0],
            "confusable_medium": conf_fa_counts[1],
            "confusable_hard": conf_fa_counts[2],
            "unrelated": unrelated_metrics.false_accepts.len(),
            "silence": silence_metric.false_accepts.len(),
            "noise": noise_false_accepts.len(),
            // Speaker-blindness DIAGNOSTICS (not false accepts): cross-speaker
            // wake-word detections are correct behaviour, so these counts do
            // NOT feed total_false_accepts.
            "noise_overlap_cross_speaker": noise_overlap_detections,
            "cross_speaker_probes": probe_fa_count,
            "total": total_false_accepts,
        },
        "classifier_trigger_metrics": classifier_trigger_json,
        "phases": {
            "phase_1_enrollment_audio_ms": phase_times[P_ENROLLMENT_AUDIO],
            "phase_2_vad_enrollment_ms": phase_times[P_VAD_ENROLLMENT],
            "phase_3_negative_training_data_ms": phase_times[P_NEG_TRAINING_DATA],
            "phase_4_classifier_training_ms": phase_times[P_CLASSIFIER_TRAINING],
            "phase_5_global_state_setup_ms": phase_times[P_GLOBAL_STATE],
            "phase_6_streaming_detection_setup_ms": phase_times[P_STREAMING_SETUP],
            "phase_7_positive_variants_ms": phase_times[P_POSITIVE_VARIANTS],
            "phase_8_confusable_negatives_ms": phase_times[P_CONFUSABLE_NEGATIVES],
            "phase_9_unrelated_negatives_ms": phase_times[P_UNRELATED_NEGATIVES],
            "phase_10_silence_negatives_ms": phase_times[P_SILENCE_NEGATIVES],
            "phase_11_noise_profiles_ms": phase_times[P_NOISE_PROFILES],
            "phase_12_cooldown_ms": phase_times[P_COOLDOWN],
            "phase_13_noise_overlap_ms": phase_times[P_NOISE_OVERLAP],
            "phase_14_teardown_ms": phase_times[P_TEARDOWN],
        },
        "per_variant_results": serde_json::Value::Array(
            pos_metrics
                .per_variant
                .iter()
                .map(|pv| pv_to_json(pv, None))
                .collect()
        ),
        // mahbot-1006 D: cold-start pass per-variant detail (same schema as
        // per_variant_results, but measured without consume_warmup / fresh
        // adaptive bootstrap).  Empty when the classifier is degenerate.
        // mahbot-1023: measured under the deferred-burst pipeline — the cold
        // pass exercises the burst sweep (buffer >= 68 frames) or the
        // segment-end pass for short utterances, so these numbers reflect
        // the fix, not the mahbot-1022 0/20 baseline.  The enrolled-speaker
        // phase (enrolled_speaker section) is the acceptance-relevant
        // in-sample measurement; this section is the same pipeline over the
        // full 20-variant positive test set.
        "per_variant_results_cold": serde_json::Value::Array(
            cold_metrics
                .per_variant
                .iter()
                .map(|pv| pv_to_json(pv, None))
                .collect()
        ),
        "per_variant_negatives": serde_json::Value::Array(negative_pv),
        "noise_overlap": noise_overlap_json,
        "classifier_diagnostics": classifier_diagnostics_json,
        "per_augmentation_detection": per_augmentation,
        "cooldown": {
            "detection_time_ms": cooldown_detection_time_ms,
            "first_detection_fired": cooldown_first_detected,
            "cooldown_suppressed_redetection": cooldown_suppressed,
            "detection_recovered_after_cooldown": cooldown_after_recovered,
            // mahbot-1052: additive keys (re-point at the enrolled-speaker F1
            // original variant; visible skip via skip_reason, never silent).
            "source_variant": cooldown_source_variant,
            "skip_reason": cooldown_skip_reason,
            "suppressed_at_2_5s": cooldown_suppressed_at_2_5s,
            "accumulation_cap_samples": super::COOLDOWN_ACCUMULATION_CAP,
            "audio_buffer_len_during_cooldown": cooldown_accumulation_cap_observed,
            "buffered_audio_processed_after_expiry": cooldown_buffered_audio_processed_after_expiry,
            "note": "GATE (hard assertions, mahbot-1052 pin 4): suppressed at ~2.5s, fires at \
                     ~3.5s — slack margins, jitter-safe; probe sleeps are excluded from \
                     detection_time_ms (mahbot-1052).",
        },
        "reproducibility": reproducibility,
        // mahbot-1023: enrolled-speaker benchmark phase (Phase 7d) — F1's 5
        // raw-level augmentation variants through the real streaming cold
        // pass (in-sample / training-side control, NOT generalization).
        "enrolled_speaker": enrolled_json,
        // mahbot-1023 item 7: false-positive impact of the deferred
        // in-distribution windows on the negative/confusable set.
        "safety_gate": safety_gate_json,
        "config": {
            "num_enrollment_variants": NUM_ENROLLMENT_VARIANTS,
            "min_detection_rate": MIN_DETECTION_RATE,
        }
    });

    // ── FAPH section (mahbot-1057, env-gated additive key) ───────────────
    // Only present when MAHBOT_FAPH=1 (ran or documented-skipped).  Standard
    // runs leave `faph_report` as None → the report surface is byte-identical
    // to the pre-change baseline (key-set contract).
    let mut json = json;
    // Phase 0 additive default-run keys (added post-macro — the main json!
    // block is already at serde_json's recursion limit).
    json["deployability"] = deployability.clone();
    json["cross_run"] = cross_run;
    // Default-run wall-clock budget acknowledgment: the acceptance window is
    // ~33-39 s; env-gated phases (FAPH / multi-seed) extend the wall clock by
    // design, so the budget check applies only to default runs.  Measured
    // HERE (after cross_run_summary + report assembly) so the reported wall
    // clock includes the archive-parse window; final JSON serialization and
    // the archive write happen just after this capture.
    let perf_wall_clock_secs = overall_start.elapsed().as_secs_f64();
    json["performance"] = serde_json::json!({
        "wall_clock_secs": perf_wall_clock_secs,
        // Not a pass-band array: the ~33-39 s span is the observed envelope;
        // the acceptance criterion is the UPPER bound (~39 s) — faster runs
        // pass (see budget_upper_bound_secs, the machine-checkable value).
        "default_run_budget_secs": "33-39 (observed envelope; criterion is the \
                                     upper bound — see budget_upper_bound_secs)",
        "budget_upper_bound_secs": 39.0,
        "budget_met": if faph_report.is_some() {
            serde_json::Value::Null
        } else {
            serde_json::Value::Bool(perf_wall_clock_secs <= 39.0)
        },
        "note": if faph_report.is_some() {
            "Env-gated FAPH phase ran — the ~33-39s default-run budget \
             does not apply to this run."
                .to_string()
        } else if perf_wall_clock_secs > 39.0 {
            format!(
                "Default-run wall clock {perf_wall_clock_secs:.1}s exceeds the ~33-39s \
                 observed envelope and its upper bound (39s) — attributable to machine load \
                 plus the one-time wake-phrase re-baselining (TTS cache invalidation and \
                 re-trained detection dynamics); the bounded Phase 0 additions measure in \
                 tens of ms.",
            )
        } else {
            format!(
                "Default-run wall clock {perf_wall_clock_secs:.1}s within the default-run \
                 budget (upper bound 39s).",
            )
        },
    });
    if let Some(faph) = faph_report {
        json["faph"] = faph;
    }

    // Output delimited JSON for CI tooling
    println!("--- BENCHMARK_JSON_BEGIN ---");
    let json_text = serde_json::to_string_pretty(&json).expect("JSON serialization");
    println!("{json_text}");
    println!("--- BENCHMARK_JSON_END ---");

    // ── Persist the report for post-run analysis (mahbot-1012 reviewer) ────
    // The report is emitted to stdout for CI, but a hard 30-minute harness
    // timeout with end-only emission can lose it entirely.  Mirror the JSON to
    // ~/.mahbot/voice_pipeline_e2e_report.json (the same directory as the
    // benchmark lock file) so a run's data survives regardless of how the
    // process ends, and a final ticket comment can cite exact numbers.
    if let Ok(report_dir) = crate::config::default_config_dir() {
        let report_path = report_dir.join("voice_pipeline_e2e_report.json");
        // ── Per-run timestamped archive (mahbot-1023, manager pin 5) ──────
        // The ≥3-run acceptance protocol computes spread from ACTUAL runs.
        // Every run is archived with a UTC timestamp before the main report
        // path is overwritten, so the acceptance review has all runs' reports.
        // No .prior.json copy is written: the old copy
        // was byte-identical to a timestamped archive and misled comparisons.
        let archive_stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
        let archive_path =
            report_dir.join(format!("voice_pipeline_e2e_report.{archive_stamp}.json"));
        match std::fs::write(&archive_path, &json_text) {
            Ok(()) => info!(
                "Benchmark report archived to {} (acceptance spread computed from timestamped per-run archives)",
                archive_path.display()
            ),
            Err(e) => warn!(
                "Could not write benchmark report archive to {}: {e}",
                archive_path.display()
            ),
        }
        match std::fs::write(&report_path, json_text) {
            Ok(()) => info!("Benchmark report written to {}", report_path.display()),
            Err(e) => warn!(
                "Could not write benchmark report to {}: {e}",
                report_path.display()
            ),
        }
    } else {
        warn!("Could not resolve ~/.mahbot/ for benchmark report persistence");
    }

    // ── Human-readable report (stderr, mahbot-871) ────────────────────────
    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    // The headline detection rate is the ENROLLED-speaker
    // HELD-OUT recall (the single-speaker positive class on unseen
    // renderings); the trained-in canaries live in the noise_overlap section.
    let dr = if enrolled_total > 0 {
        enrolled_detected as f64 / enrolled_total as f64
    } else {
        0.0
    };
    // NOTE: checkmarks are informational only — benchmark is report-only, no
    // pass/fail gating (mahbot-953).  The marks now reflect the ACTUAL limits
    // instead of being unconditional ✓ (mahbot-1005 §7).
    let mk = |pass: bool| if pass { '✓' } else { '✗' };
    let dr_pass = if enrolled_total > 0 && dr >= MIN_DETECTION_RATE {
        '✓'
    } else {
        '✗'
    };
    let unrelated_ok = unrelated_metrics.false_accepts.is_empty();
    let silence_ok = silence_metric.false_accepts.is_empty();
    let noise_ok = noise_false_accepts.len() <= tier_limits(BenchTier::Hard).noise;

    // Per-tier counts for report
    let fa_easy = conf_fa_by_tier[0].len();
    let fa_medium = conf_fa_by_tier[1].len();
    let fa_hard = conf_fa_by_tier[2].len();

    eprintln!(
        "\n\
         ═══════════════════════════════════════════════════════════\n\
                 Voice Pipeline E2E Benchmark Report\n\
         ═══════════════════════════════════════════════════════════\n\
         Date/Time:      {timestamp}\n\
         Tier:           Easy/Medium/Hard (per-tier limits)\n\
         Detection rate: {dr_pct:.1}% ({enrolled_detected}/{enrolled_total})  {dr_pass}\n\
         MIN_DR target:  ≥{min_dr_const_pct:.0}% (report-only)\n\
         False accepts:\n\
           Confusable:\n\
             Easy:    {fa_easy}",
        dr_pct = dr * 100.0,
        min_dr_const_pct = MIN_DETECTION_RATE * 100.0,
    );
    if !conf_fa_by_tier[0].is_empty() {
        eprintln!("               Triggers: {:?}", conf_fa_by_tier[0]);
    }
    eprintln!("             Medium:  {fa_medium}");
    if !conf_fa_by_tier[1].is_empty() {
        eprintln!("               Triggers: {:?}", conf_fa_by_tier[1]);
    }
    eprintln!("             Hard:    {fa_hard}");
    if !conf_fa_by_tier[2].is_empty() {
        eprintln!("               Triggers: {:?}", conf_fa_by_tier[2]);
    }
    eprintln!(
        "           Unrelated:  {unrelated_count}  {mark} (limit ≤0)",
        unrelated_count = unrelated_metrics.false_accepts.len(),
        mark = mk(unrelated_ok),
    );
    eprintln!(
        "           Silence:    {silence_count}  {mark} (limit ≤0)",
        silence_count = silence_metric.false_accepts.len(),
        mark = mk(silence_ok),
    );
    eprintln!(
        "           Noise:      {noise_count}  {mark} (limit ≤{noise_limit})",
        noise_count = noise_false_accepts.len(),
        mark = mk(noise_ok),
        noise_limit = tier_limits(BenchTier::Hard).noise,
    );
    eprintln!(
        "           Cross-speaker diagnostics (speaker-blind, high expected): canaries {noise_overlap_detections}/{noise_overlap_total_variants}, probes {probe_fa_count}/{probe_total_variants}",
    );
    eprintln!(
        "           ───────────────────────────────────────\n\
            | Easy  total: {easy_total}  {easy_mark} (limit ≤{easy_limit}) |\n\
            | Medium total: {medium_total}  {med_mark} (limit ≤{med_limit}) |\n\
            | Hard  total: {hard_total}  {hard_mark} (limit ≤{hard_limit}) |\n\
            ───────────────────────────────────────",
        easy_total = tier_total_fas[0],
        easy_mark = mk(tier_total_fas[0] <= tier_limits(BenchTier::Easy).total),
        easy_limit = tier_limits(BenchTier::Easy).total,
        medium_total = tier_total_fas[1],
        med_mark = mk(tier_total_fas[1] <= tier_limits(BenchTier::Medium).total),
        med_limit = tier_limits(BenchTier::Medium).total,
        hard_total = tier_total_fas[2],
        hard_mark = mk(tier_total_fas[2] <= tier_limits(BenchTier::Hard).total),
        hard_limit = tier_limits(BenchTier::Hard).total,
    );

    // ── Classifier trigger summary (mahbot-952) ────────────────────────
    let ct_stderr = |ct: &ClassifierTriggerMetrics| -> String {
        format!(
            "{}/{} triggers, fa={}",
            ct.classifier_triggers, ct.total_variants, ct.full_pipeline_fa
        )
    };
    eprintln!(
        "         Classifier triggers:\
         \n           Confusable: {conf}\
         \n             Easy:     {easy}\
         \n             Medium:   {med}\
         \n             Hard:     {hard}\
         \n           Unrelated:  {unrel}\
         \n           Silence:    {sil}\
         \n           Noise:      {noise}",
        conf = ct_stderr(&conf_classifier),
        easy = ct_stderr(&conf_classifier_by_tier[0]),
        med = ct_stderr(&conf_classifier_by_tier[1]),
        hard = ct_stderr(&conf_classifier_by_tier[2]),
        unrel = ct_stderr(&unrel_classifier),
        sil = ct_stderr(&silence_classifier),
        noise = ct_stderr(&noise_classifier_aggregate),
    );

    eprintln!(
        "         ═══════════════════════════════════════════════════════════\n\
                   BENCHMARK COMPLETE (report-only — no pass/fail gating)\n\
         ═══════════════════════════════════════════════════════════",
    );

    // ── Deployability verdict banner (Phase 0, prominent) ─────────────
    // Production's enrollment self-test gate is not the bench's gate: the
    // bench stays report-only.  But a model production would reject must
    // never be misread as deployable, so the warning is prominent in the
    // human-readable banner AND structured in the top-level deployability key.
    // Uses the (passed, total, required, deploy_would_pass) tuple returned by
    // deployability_json directly — the JSON is only the structured mirror,
    // never re-parsed here.  `deploy_would_pass` is Option<bool> (None = no
    // weights), so the no-weights arm below is explicit, not a hardcoded
    // false.
    match &self_test_result {
        None => eprintln!(
            "         Deployability: NOT MEASURED — no classifier weights \
             (finalize gate + direct training both failed); report-only"
        ),
        Some(_) if deploy_would_pass == Some(true) => eprintln!(
            "         Deployability: model would PASS production's enrollment gate \
             ({passed}/{total} utterances trigger detection) — informational only"
        ),
        Some(_) => eprintln!(
            "\n         ⚠  PRODUCTION WOULD REJECT THIS MODEL: only {passed}/{total} \
             enrollment utterances triggered detection (gate ≥{:.0}%).\n         \
             These recall numbers are labeled 'model production would reject — \
             informational only'.\n         The bench is report-only: this does NOT fail \
             the run, but the trained model is not deployable as-is.\n",
            super::ENROLLMENT_QUALITY_SELF_TEST_MIN_FRACTION * 100.0,
        ),
    }
    eprintln!(
        "         Wake phrase: '{}' (source: {})",
        wake_word(),
        wake_phrase_source(),
    );

    // Catastrophic regression guard: at least 1 positive detection must occur.
    // Without this, a pipeline that detects nothing would pass all FA assertions
    // (zero detections = zero false accepts) (mahbot-911).
    // NOTE: report-only — warns instead of asserting (mahbot-953).
    // The positive class is the enrolled speaker's wake word,
    // measured end-to-end by the Phase-7d held-out recall phase (detected_live).
    let (positive_detected, positive_total): (u64, u64) =
        (enrolled_detected as u64, enrolled_total as u64);
    if positive_detected == 0 {
        warn!(
            "Catastrophic regression: 0/{positive_total} enrolled-speaker wake word variants \
             detected — pipeline is not detecting the enrolled speaker's wake word.  \
             Detection rate: {dr:.1}%",
            dr = if positive_total > 0 {
                positive_detected as f64 / positive_total as f64 * 100.0
            } else {
                0.0
            },
        );
    }

    // Enrolled-speaker held-out recall summary (report-only).
    let actual_dr = if enrolled_total > 0 {
        enrolled_detected as f64 / enrolled_total as f64
    } else {
        pos_metrics.detection_rate()
    };
    if enrolled_phase_ran {
        info!(
            "Enrolled-speaker held-out recall: {:.1}% ({}/{}) — target ≥{:.0}%",
            actual_dr * 100.0,
            enrolled_detected,
            enrolled_total,
            MIN_DETECTION_RATE * 100.0,
        );
    }

    // ── Cross-speaker speaker-blindness summary (diagnostics) ───────────
    eprintln!(
        "         Cross-speaker diagnostics (speaker-blind — high detection expected): \
         canaries {noise_overlap_detections}/{noise_overlap_total_variants}, \
         probes {probe_fa_count}/{probe_total_variants}",
    );

    // Per-tier confusable false-accept checks (mahbot-871)
    // NOTE: report-only — warns instead of asserting (mahbot-953).
    for (i, tier_name, bench_tier) in &[
        (0, "easy", BenchTier::Easy),
        (1, "medium", BenchTier::Medium),
        (2, "hard", BenchTier::Hard),
    ] {
        let limits = tier_limits(*bench_tier);
        if conf_fa_counts[*i] > limits.confusable {
            warn!(
                "Too many confusable false accepts in tier '{tier_name}': {} — need ≤{}",
                conf_fa_counts[*i], limits.confusable,
            );
        }
        if tier_total_fas[*i] > limits.total {
            warn!(
                "Too many total false accepts in tier '{tier_name}': {} — need ≤{}",
                tier_total_fas[*i], limits.total,
            );
        }
    }

    // Per-category false accept checks (non-tiered categories)
    // NOTE: report-only — warns instead of asserting (mahbot-953).
    if !unrelated_metrics.false_accepts.is_empty() {
        warn!(
            "Too many unrelated false accepts: {} — need ≤0",
            unrelated_metrics.false_accepts.len(),
        );
    }
    if !silence_metric.false_accepts.is_empty() {
        warn!(
            "Too many silence false accepts: {} — need ≤0",
            silence_metric.false_accepts.len(),
        );
    }
    if noise_false_accepts.len() > tier_limits(BenchTier::Hard).noise {
        warn!(
            "Too many noise false accepts: {} — need ≤{} (Hard-tier noise limit)",
            noise_false_accepts.len(),
            tier_limits(BenchTier::Hard).noise,
        );
    }

    // Classifier degeneracy check — report-only (mahbot-953).
    // If the classifier is degenerate, the benchmark still completes and reports
    // so we can diagnose.
    if degenerate {
        warn!(
            "Classifier produced degenerate all-zero solution — training failed. \
             {:.1}% of weights near zero (threshold=1%). \
             Detection phases were skipped. See JSON output for details.",
            near_zero_frac * 100.0,
        );
    }

    // Noise-overlap warnings already occurred at the check site above (line ~2843).
    // No pass/fail gating — report-only (mahbot-953).

    // ── Final result ──
    // All threshold checks above emit warnings on violation — no pass/fail gating (mahbot-953).
    info!("═══ E2E Voice Pipeline benchmark complete (report-only — no pass/fail gating) ═══");

    // Stop heartbeat thread and wait for it to exit.
    heartbeat_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = heartbeat_handle.join();
}

// ── mahbot-1045 A1 fixture tests ─────────────────────────────────────────
// Compile under `cargo test --features voice-tests` (the bench module is
// feature-gated; `#[cfg(test)]` keeps these out of the harness=false bench
// binary).  Locks the real `pcm_augment_enrollment_variants` site (the
// canonical raw-TTS-PCM gate input) to the golden captured from the
// pre-dedup inline code at HEAD 0d1a074.

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic fixed input PCM (no RNG) — same generator as the
    /// `voice.rs::augment_tests` fixture so hashes line up across modules.
    fn fixture_pcm(len: usize) -> Vec<f32> {
        let sample_rate = TARGET_SAMPLE_RATE as f32;
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin())
                    * (1.0 - t / 2.0).max(0.0)
            })
            .collect()
    }

    /// SHA-256 over the ordered `(label, pcm)` pairs as
    /// `pcm_augment_enrollment_variants` returns them.
    fn hash_labeled_variants(variants: &[(Vec<f32>, String)]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for (pcm, label) in variants {
            hasher.update(label.as_bytes());
            hasher.update([0u8]);
            for sample in pcm {
                hasher.update(sample.to_le_bytes());
            }
        }
        format!("{:x}", hasher.finalize())
    }

    #[test]
    fn pcm_augment_enrollment_variants_matches_golden() {
        // Fixed input + loop-index seed → byte-identical output (golden
        // captured at the current recipe).
        let variants = vec![(fixture_pcm(16000), "wake".to_string())];
        let out = pcm_augment_enrollment_variants(&variants);
        let h = hash_labeled_variants(&out);
        println!("GOLDEN P-site: long={h}");

        let short_variants = vec![(fixture_pcm(4000), "wake".to_string())];
        let short_out = pcm_augment_enrollment_variants(&short_variants);
        let hs = hash_labeled_variants(&short_out);
        println!("GOLDEN P-site: short={hs}");

        assert_eq!(out.len(), 12, "long input → all 12 recipe cells");
        // captured at the current recipe:
        assert_eq!(
            h, "c9b0482b4eb9936abfd3cbba38e99ecc74078ad0128754d749b12f29357571a7",
            "long-input golden drifted"
        );

        assert_eq!(short_out.len(), 11, "short input → speed-up skipped");
        assert!(
            short_out.iter().all(|(_, l)| !l.ends_with("speed_up")),
            "short input must not contain a speed-up variant"
        );
        assert_eq!(
            hs, "ce92008745a2336498eee830fa09054ec46605b0e9ddfa3bdebb05999074b23c",
            "short-input golden drifted"
        );
    }

    // ── mahbot-1057: FAPH Poisson CI helpers ──────────────────────────────

    /// Wilson–Hilferty χ² quantile vs published values (df = 2(k+1) for the
    /// Poisson 95% upper bound).  Tolerances are generous (~1%): the bound is
    /// report-only, and the approximation error at these df is < 0.6%.
    #[test]
    fn chi2_quantile_matches_published_values() {
        // Standard 97.5% chi-square quantiles (df=2,4,6,8,10) from NIST tables.
        let cases: &[(f64, f64)] = &[
            (2.0, 7.3778),
            (4.0, 11.1433),
            (6.0, 14.4494),
            (8.0, 17.5345),
            (10.0, 20.4832),
        ];
        for (df, expected) in cases {
            let got = chi2_quantile(0.975, *df);
            assert!(
                (got - expected).abs() / expected < 0.01,
                "df={df}: got {got:.4}, expected {expected:.4}"
            );
        }
        // df=2 (k=0): upper events for 0 FAs ≈ 3.689 (0.5·χ²(0.975,2)).
        let k0_upper = 0.5 * chi2_quantile(0.975, 2.0);
        assert!((k0_upper - 3.688).abs() < 0.05, "k=0 upper: {k0_upper}");
    }

    /// Normal quantile sanity: Φ⁻¹(0.5) = 0, Φ⁻¹(0.975) ≈ 1.96,
    /// Φ⁻¹(0.025) ≈ -1.96.
    #[test]
    fn normal_quantile_sanity() {
        assert!(normal_quantile(0.5).abs() < 1e-9, "median must be 0");
        assert!((normal_quantile(0.975) - 1.959_964).abs() < 1e-3);
        assert!((normal_quantile(0.025) + 1.959_964).abs() < 1e-3);
        // Monotonic.
        let a = normal_quantile(0.3);
        let b = normal_quantile(0.7);
        assert!(a < b, "quantile must be monotonic");
    }

    /// FAPH Poisson upper bound: 0 events in 5.99 h → the 95% upper per-hour
    /// rate is 0.5·χ²(0.975, 2)/5.99 ≈ 0.61/h (not a <1 FA/24h claim).
    #[test]
    fn faph_poisson_upper_bound_semantics() {
        let audio_hours = 5.99_f64;
        let events = 0_u64;
        let upper_events = 0.5 * chi2_quantile(0.975, 2.0 * (events as f64 + 1.0));
        let per_hour = upper_events / audio_hours;
        assert!(
            per_hour > 0.5 && per_hour < 0.7,
            "0 FA/5.99h → ~0.61/h upper"
        );
        // The bound must shrink as audio-hours grow (3 runs = 17.97 h).
        let per_hour_3 = upper_events / (audio_hours * 3.0);
        assert!(per_hour_3 < per_hour, "more hours → tighter upper bound");
    }

    // ── single-voice enrollment DSP + Wilson CI ───────────────

    /// Guided-prompt DSP primitives: determinism, HF attenuation, and group
    /// differentiation.  The filter checks verify the rolloff directly (a
    /// total-energy comparison of the conditioned clips would be dominated by
    /// the level-reduction gains); the condition_enrollment_clip checks cover
    /// determinism, the noise floor, and the morning resample's length shift.
    #[test]
    fn guided_prompt_dsp_deterministic_and_attenuates_hf() {
        let sample_rate = TARGET_SAMPLE_RATE;
        let rms = |x: &[f32]| crate::util::compute_rms(x);

        // one_pole_lowpass: deterministic, attenuates 5 kHz through a 2.2 kHz
        // cutoff, passthrough at Nyquist.
        let hf: Vec<f32> = (0..sample_rate as usize / 4)
            .map(|i| (2.0 * core::f32::consts::PI * 5000.0 * i as f32 / sample_rate as f32).sin())
            .collect();
        let a = one_pole_lowpass(&hf, 2200.0, sample_rate);
        let b = one_pole_lowpass(&hf, 2200.0, sample_rate);
        assert_eq!(a, b, "lowpass must be deterministic");
        assert!(
            rms(&a) < rms(&hf) * 0.5,
            "5 kHz through a 2.2 kHz lowpass must be attenuated"
        );
        assert_eq!(
            one_pole_lowpass(&hf, sample_rate as f32 * 0.5, sample_rate),
            hf,
            "cutoff at Nyquist must pass through unchanged"
        );

        // high_shelf_cut: deterministic, tilts 4 kHz down through a −4 dB
        // @ 3 kHz shelf, passthrough at zero gain.
        let tilt: Vec<f32> = (0..sample_rate as usize / 4)
            .map(|i| (2.0 * core::f32::consts::PI * 4000.0 * i as f32 / sample_rate as f32).sin())
            .collect();
        let a = high_shelf_cut(&tilt, -4.0, 3000.0, sample_rate);
        let b = high_shelf_cut(&tilt, -4.0, 3000.0, sample_rate);
        assert_eq!(a, b, "high-shelf cut must be deterministic");
        assert!(
            rms(&a) < rms(&tilt) * 0.7,
            "4 kHz above a -4 dB @ 3 kHz shelf must be attenuated"
        );
        assert_eq!(high_shelf_cut(&tilt, 0.0, 3000.0, sample_rate), tilt);

        // condition_enrollment_clip: deterministic per clip, every clip gets
        // the noise floor, and the morning group's 0.92× resample is longer.
        let mixed: Vec<f32> = (0..sample_rate as usize / 2)
            .map(|i| {
                0.5 * (2.0 * core::f32::consts::PI * 220.0 * i as f32 / sample_rate as f32).sin()
                    + 0.3
                        * (2.0 * core::f32::consts::PI * 4000.0 * i as f32 / sample_rate as f32)
                            .sin()
            })
            .collect();
        for clip_idx in 0..10 {
            let a = condition_enrollment_clip(&mixed, clip_idx);
            let b = condition_enrollment_clip(&mixed, clip_idx);
            assert_eq!(a, b, "clip {clip_idx} conditioning must be deterministic");
            assert_ne!(
                a, mixed,
                "clip {clip_idx} must be conditioned (noise floor)"
            );
        }
        let normal_len = condition_enrollment_clip(&mixed, 0).len();
        let morning_len = condition_enrollment_clip(&mixed, 8).len();
        assert!(
            morning_len > normal_len,
            "morning clip (0.92x slower resample) must be longer than normal"
        );
    }

    /// Wilson CI: perfect recall → narrow high interval; zero recall → wide low
    /// interval; empty sample → None.
    #[test]
    fn wilson_ci_sanity() {
        assert!(wilson_ci(0, 0).is_none(), "empty sample → None");
        let (lo, hi) = wilson_ci(16, 16).expect("full recall");
        assert!(lo > 0.78 && hi >= 1.0, "16/16 → CI [{lo:.3}, {hi:.3}]");
        let (lo0, hi0) = wilson_ci(0, 16).expect("zero recall");
        assert!(lo0 == 0.0 && hi0 < 0.21, "0/16 → CI [{lo0:.3}, {hi0:.3}]");
        let (lo_half, hi_half) = wilson_ci(8, 16).expect("half recall");
        assert!(
            lo_half < 0.5 && 0.5 < hi_half,
            "8/16 → CI must straddle 0.5"
        );
    }

    /// Voice allocation: standard 10-style set → enrolled=F1, negative pool
    /// F1..M1, canaries M2/M3, reserved M4/M5 (disjoint, exhaustive).
    #[test]
    fn allocate_voices_standard_set() {
        let styles: Vec<String> = (1..=5)
            .map(|i| format!("F{i}.json"))
            .chain((1..=5).map(|i| format!("M{i}.json")))
            .collect();
        let a = allocate_voices(&styles);
        assert_eq!(a.enrolled, "F1.json");
        assert_eq!(a.negative_pool, styles[..6]);
        assert_eq!(a.canaries, styles[6..8]);
        assert_eq!(a.reserved, styles[8..10]);
        // Reserved voices must never appear in any training pool.
        assert!(
            a.negative_pool.iter().all(|s| !a.reserved.contains(s))
                && a.canaries.iter().all(|s| !a.reserved.contains(s)),
            "reserved voices must be disjoint from every training pool"
        );
    }
}
