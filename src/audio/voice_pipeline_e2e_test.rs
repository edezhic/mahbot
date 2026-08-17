//! E2E integration test / benchmark for the full voice pipeline.
//!
//! This test exercises the **enrollment-to-detection cycle with realistic
//! TTS-generated speech audio**.  It uses the TTS engine to synthesize wake
//! word variants, feeds them through the enrollment pipeline, builds the
//! prototype enrollment (L2-normalized centroid of utterance embeddings via
//! the shared Qwen3-ASR encoder), calibrates it against negative samples,
//! then runs detection on:
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
//! release codegen is the faithful target.  Acceptance metrics are report-only
//! (measured, never gated); hard assertions exist only for production-behavior
//! contracts (e.g. the cooldown gate and accumulation cap in the cooldown
//! phase).
//!
//! First run populates the TTS audio cache (subsequent runs hit it).  The
//! encoder pipeline re-encodes raw audio through the shared Qwen3-ASR model
//! per run — there is no embedding disk cache — so the wall clock is dominated
//! by encoder forwards over the TEST surface (40-clip wake-only basis, SNR
//! envelope over it, doubled negative pools, wake-over-babble,
//! owner-negative detection, probe scale-up).  The exact wall clock is
//! auditable from the top-level `total_wall_time_ms` key (whole run,
//! report-assembly window included).
//!
//! # Requirements
//!
//! * TTS models must be downloaded and cached (run the app once).
//! * The shared Qwen3-ASR model must be present in `~/.mahbot/models/`
//!   (the same model the local transcriber loads — wake word detection
//!   reuses it; the old OpenWakeWord ONNX models are no longer used).

// Clippy allowances — use super::* is intentional (mirrors the voice module's API surface).
// Cast warnings: this is a benchmark file with known-safe numeric conversions.
#![allow(
    clippy::wildcard_imports,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use super::*; // voice module items (handle_wake_word_detection, PipelineCtx, etc.)
use crate::audio::tts;
use earshot::Detector;
use rand::{RngExt, SeedableRng};
use std::borrow::Cow;
use std::time::Instant;

// ── Constants ──────────────────────────────────────────────────────────────

/// Index of the rolling sum field in per_frame_scores `[total_score, rolling_sum, threshold]` triples.
const ROLLING_SUM_IDX: usize = 1;
/// Index of the effective threshold field in per_frame_scores `[total_score, rolling_sum, threshold]` triples.
const THRESHOLD_IDX: usize = 2;
/// Index of the total (soft) score field in per_frame_scores
/// `[total_score, rolling_sum, threshold]` triples (the
/// layout is shared with production's scoring pipeline, so a production
/// layout change must not silently corrupt miss classification).
const TOTAL_SCORE_IDX: usize = 0;

/// Minimum effective rolling-sum threshold for the detection gate.
///
/// The rolling sum must reach at least this value for the pipeline to have
/// fired.  Below this threshold, misses are attributed to the score gate.
///
/// = [`super::match_threshold()`] = ROLLING_WINDOW_N × MATCH_THRESHOLD_FACTOR
/// = 3 × 0.55 = 1.65.
const MIN_GATE_THRESHOLD: f32 = super::match_threshold();

/// Number of samples in the warm-up audio prepended before each test
/// utterance.
///
/// ~1.28 s of background audio at 16 kHz fed through the pipeline before the
/// test utterance so the warm pass exercises the ring-primed, segment-warmed
/// path (production background silence/noise before anyone speaks).
const WARMUP_PREPEND_SAMPLES: usize = 20480; // 1.28s × 16 kHz

/// TTS phrase for warm-up audio.
///
/// A short non-wake-word phrase synthesised via the already-loaded TTS engine.
/// Speech-like harmonics guarantee the Earshot neural VAD triggers, producing
/// embeddings for the pre-utterance ring.
///
/// Must NOT contain the wake word ([`wake_word()`] — runtime-resolved from the
/// deployed config store) or phonetically similar phrases that could trigger
/// detection.
const WARMUP_TTS_PHRASE: &str = "testing one two three";

/// Cached TTS warm-up audio, populated on first successful synthesis.
/// Unlike the original `WARMUP_NOISE_CACHE`, this caches ONLY the TTS
/// result — if TTS is unavailable on the first call, a fresh pink-noise
/// fallback is returned (no caching), so TTS is re-evaluated on subsequent
/// calls (avoids cache poisoning).
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

/// Number of enrollment variants to generate.
///
/// Production generates ~110-120 positive training sequences from 10
/// utterances × the full 12-cell recipe (minus SpeedUp skip for
/// short utterances).  We target ~10 TTS enrollment variants to match this.
/// Since the re-scope all 10 clips come from ONE TTS voice (the enrolled
/// speaker) with guided-prompt DSP conditioning.
const NUM_ENROLLMENT_VARIANTS: usize = 10;

/// Owner-negative (non-wake-word) phrases for negative calibration
/// (re-scoped).
///
/// These are generated via TTS from the ENROLLED voice only (the owner's own
/// non-wake-word speech) and encoded as negative samples to help the
/// prototype reject general speech from the enrolled user.  The phrase list is
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
///   (kept from an earlier bench).
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
    /// angle/morning −3 dB + 17 dB SNR) — the encoder pipeline consumes raw
    /// audio, so realism must come from the SNR / spectral relationships.
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
/// banner; the 0.2 granularity reflects the 40-clip wake-only held-out recall
/// set, where each miss costs roughly 2.5 percentage points.
const MIN_DETECTION_RATE: f64 = 0.60;

/// Acceptance-basis prefix for the current report series.
/// `cross_run_summary` filters archives by this so v3 (enlarged 40-clip basis)
/// runs never mix with v2 16-clip archives in the ≥3-run spread.
const ACCEPTANCE_BASIS_PREFIX: &str = "held_out_recall_v3";

// Per-category false accept limits are now dynamic by tier — see
// [`tier_limits`] and [`BenchTier`].

/// Confusable near-miss phrases for negative detection testing.
///
/// Bench-local copy of the canonical mahbot-family confusable list, kept for
/// negative-audio generation and per-tier FA tracking.
const CONFUSABLE_PHRASES: &[&str] = &[
    // ── Hard tier — direct phonetic substitutions (wake-word-like) ──
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
    // ── Medium tier — rhythmic/melodic confusables ──
    "hay map pot",
    "huh mahbot",
    "eh mad bot",
    "hey maybott",
    "they mad bot",
    "haymaker",
    // ── Medium tier — embedded wake-word sounds ──
    "hey maybe not",
    "play mah jong",
    "hey matter of fact",
    "a day with mahbot",
    // ── Easy tier — short phonetic near-misses ──
    "madbot",
    "mat bot",
    "bad bot",
    "mad lot",
    "mad pot",
    "med bot",
    "my bot",
    "may bot",
];
const CONFUSABLE_HARD: &[&str] = &[
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
];
const CONFUSABLE_MEDIUM: &[&str] = &[
    "hay map pot",
    "huh mahbot",
    "eh mad bot",
    "hey maybott",
    "they mad bot",
    "haymaker",
    "hey maybe not",
    "play mah jong",
    "hey matter of fact",
    "a day with mahbot",
];
const CONFUSABLE_EASY: &[&str] = &[
    "madbot", "mat bot", "bad bot", "mad lot", "mad pot", "med bot", "my bot", "may bot",
];

/// Seeds per confusable phrase and the base seed (bench-local).
const CONFUSABLE_SEEDS_PER_PHRASE: usize = 5;
const CONFUSABLE_SEED_BASE: u64 = 1000;

/// Unrelated speech phrases for negative detection testing.
///
/// Bench-local copy of the canonical unrelated-speech list.
const UNRELATED_PHRASES: &[&str] = &[
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

/// Seeds per unrelated phrase and the base seed — bench-local.
const UNRELATED_SEEDS_PER_PHRASE: usize = 3;
const UNRELATED_SEED_BASE: u64 = 2000;

/// Silence durations for the negative silence matrix: three
/// durations in samples at 16 kHz (0.5 s / 1.0 s / 2.0 s).
const SILENCE_DURATIONS: &[(&str, usize)] = &[
    ("silence_0_5s", 8_000),
    ("silence_1_0s", 16_000),
    ("silence_2_0s", 32_000),
];

/// Noise audio length in samples (1 second at 16 kHz).
const NOISE_LEN: usize = 16_000;

/// Noise profiles for negative detection testing.
///
/// Each noise profile is a (label, generator_fn) pair.  The generator
/// produces PCM f32 samples at 16 kHz.
type NoiseGenerator = fn() -> Vec<f32>;

/// Training noise profiles.  This list feeds negative calibration via
/// [`generate_ambient_noise_sequences`] — the detection-only additions
/// ([`NOISE_PROFILES_DETECTION_ONLY`]) must never be appended here.
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

/// Detection-only noise profiles: fresh generator seeds 52-55.  Chained after
/// [`NOISE_PROFILES`] at the detection call sites — never appended to the
/// training list.
const NOISE_PROFILES_DETECTION_ONLY: &[(&str, NoiseGenerator)] = &[
    ("impulse burst noise", generate_impulse_burst_noise),
    ("oscillating tone noise", generate_oscillating_tone_noise),
    ("crackle static noise", generate_crackle_static_noise),
    ("pulsed broadband noise", generate_pulsed_broadband_noise),
];

/// All noise profiles for the detection call sites: the training list plus
/// the detection-only additions (training never sees the latter).  Single
/// owner of the "detection enumerations include the detection-only profiles"
/// invariant.
fn all_noise_profiles() -> impl Iterator<Item = &'static (&'static str, NoiseGenerator)> {
    NOISE_PROFILES.iter().chain(NOISE_PROFILES_DETECTION_ONLY)
}

/// TTS target sample rate (voice pipeline rate).
const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Default TTS voice style when no styles are available from disk.
/// This matches the naming convention used by the TTS model download.
const DEFAULT_TTS_STYLE: &str = "M1.json";

/// Cache directory relative to storage root.
const TEST_CACHE_DIR: &str = "test_cache/voice_e2e";

/// SNR levels for the noise-overlapped detection test.
/// Each entry is (label, snr_db).  Clean = infinity dB (no noise added).
const NOISE_OVERLAP_SNRS: &[(&str, f32)] = &[
    ("clean", f32::INFINITY),
    ("20dB", 20.0),
    ("10dB", 10.0),
    ("5dB", 5.0),
    ("0dB", 0.0),
];

/// Noise types to use for noise-overlapped detection tests.
/// White (uniform), Pink, and Brown noise.
const NOISE_OVERLAP_TYPES: &[(&str, NoiseGenerator)] = &[
    ("white", generate_white_uniform_noise),
    ("pink", generate_pink_noise),
    ("brown", generate_brown_noise),
];

/// Overlapping-speech babble phrases: non-wake speech rendered by the bench's
/// existing TTS voices, mixed with held-out wake clips as a speech-on-speech
/// interferer.  Self-contained — no external audio assets.
const BABBLE_PHRASES: &[&str] = &[
    "let me tell you about the meeting tomorrow",
    "could you pass the remote control please",
    "the weather today is surprisingly warm",
];

/// Babble overlay seed base (9500+ — collision-free; owner-negative training
/// uses 9000-9029).
const BABBLE_SEED_BASE: u64 = 9500;

/// SNR levels for the wake-over-babble cells (dB).
const BABBLE_SNRS: &[(&str, f32)] = &[("10dB", 10.0), ("5dB", 5.0), ("0dB", 0.0)];

// ── Tiered benchmark configuration ────────────────────────────

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

/// Return the per-category false-accept comparison limits for the given tier.
///
/// Report-only: the measured per-run false-accept counts are compared against
/// these in the report banner and the `results` map.  The bench never
/// hard-gates acceptance on them; the aggregate false-accept allowance is the
/// safety-gate 5%-of-corpus rule ([`safety_gate`]).
const fn tier_limits(tier: BenchTier) -> TierLimits {
    match tier {
        BenchTier::Easy => TierLimits {
            confusable: 8,
            noise: 1,
            total: 8,
        },
        BenchTier::Medium => TierLimits {
            confusable: 6,
            noise: 1,
            total: 6,
        },
        BenchTier::Hard => TierLimits {
            confusable: 2,
            noise: 1,
            total: 3,
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

/// Confusable seed band for a variant label ("confusable" or "confusable2").
fn confusable_band(label: &str) -> &str {
    if label.starts_with("confusable2_") {
        "confusable2"
    } else {
        "confusable"
    }
}

/// Extract the phrase text from a confusable-band variant label.  The
/// detection pool spans two seed bands: band 1 labels use the "confusable"
/// prefix, band 2 uses "confusable2" (a distinct prefix, not a re-key of
/// existing clips).  Both parse to the same phrase text so [`tier_for_phrase`]
/// lookup works for either band.
fn confusable_phrase(label: &str) -> &str {
    phrase_from_label(label, confusable_band(label))
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

// ── TTS audio cache ─────────────────────────────────────

// Compute a deterministic model version hash from TTS compile-time SHA-256
// constants.
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
/// (the encoder pipeline consumes raw audio, so realism comes from SNR vs the fixed
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
/// (`style = available_styles[i % num_styles]`) is gone — it was
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

/// Wake-only held-out recall clip count (seeds 3000..3000+N).  Enlarged to
/// 40: the original 16 clips (3000-3015) plus 24 new clips in the freed
/// collision-free band (3016-3039).  The basis IS this enlarged set — there
/// is no separate 16-clip sub-pool.
const HELD_OUT_WAKE_ONLY_CLIPS: usize = 40;

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

/// Generate the wake-over-babble overlay track: non-wake phrases rendered by
/// the bench's existing TTS voices, concatenated into one multi-voice babble
/// segment.  Self-contained — no external audio assets.  The wake clips are
/// mixed against this track in-memory at the babble SNR cells; only the
/// babble clips themselves enter the PCM cache.
fn generate_babble_overlay_cached(
    available_styles: &[String],
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Option<Vec<f32>> {
    let num_styles = available_styles.len().max(1);
    let mut track: Vec<f32> = Vec::new();
    for (i, &phrase) in BABBLE_PHRASES.iter().enumerate() {
        let style = if available_styles.is_empty() {
            DEFAULT_TTS_STYLE
        } else {
            &available_styles[(i * 2) % num_styles]
        };
        if let Some(pcm) = synthesize_wake_word_variant_cached(
            phrase,
            style,
            BABBLE_SEED_BASE + i as u64,
            TARGET_SAMPLE_RATE,
            model_hash,
            cache_dir,
        ) {
            track.extend_from_slice(&pcm);
        }
    }
    (!track.is_empty()).then_some(track)
}

/// Seed configuration for TTS phrase variant generation.
///
/// Encapsulates the three related seed parameters that control deterministic
/// TTS synthesis of phrase variants with different seeds and style rotations.
/// Bundled into a struct for call-site clarity.
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
        // (per-phrase seed formula): rotate styles across
        // (phrase × seed_variant) combos.
        let style_idx = (i * seed.num_variants + seed.seed_variant) % num_styles;
        let style = if available_styles.is_empty() {
            DEFAULT_TTS_STYLE
        } else {
            &available_styles[style_idx]
        };
        // Production seed formula:
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
        // Shared EPSILON-clamp pair sampler — preserves the
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

// Detection-only noise profiles: fresh generator seeds 52-55.  These live in
// [`NOISE_PROFILES_DETECTION_ONLY`] only — appending them to
// [`NOISE_PROFILES`] would silently extend the measured negative corpus.

/// Impulse burst noise: sparse broadband clicks.
fn generate_impulse_burst_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(52);
    (0..NOISE_LEN)
        .map(|_| {
            if rng.random::<f32>() < 0.02 {
                (rng.random::<f32>() * 2.0 - 1.0).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

/// Oscillating tone noise: slow frequency sweep mixed with white noise.
fn generate_oscillating_tone_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(53);
    (0..NOISE_LEN)
        .map(|i| {
            let t = i as f32 / TARGET_SAMPLE_RATE as f32;
            let freq = 200.0 + 300.0 * (2.0 * core::f32::consts::PI * 0.5 * t).sin();
            let tone = (2.0 * core::f32::consts::PI * freq * t).sin() * 0.3;
            let noise: f32 = rng.random::<f32>() * 2.0 - 1.0;
            (tone + noise * 0.5).clamp(-1.0, 1.0)
        })
        .collect()
}

/// Crackle static noise: sparse high-amplitude transients over a quiet floor.
fn generate_crackle_static_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(54);
    let mut prev: f32 = 0.0;
    (0..NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            let crackle = if rng.random::<f32>() < 0.005 {
                white * 3.0
            } else {
                white * 0.1
            };
            let hi = 0.9 * (crackle - prev) + 0.8 * prev;
            prev = crackle;
            hi.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Pulsed broadband noise: white noise gated by a slow square envelope.
fn generate_pulsed_broadband_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(55);
    (0..NOISE_LEN)
        .map(|i| {
            let t = i as f32 / TARGET_SAMPLE_RATE as f32;
            let gate = if (t * 2.0).sin() > 0.0 { 1.0 } else { 0.0 };
            let noise: f32 = rng.random::<f32>() * 2.0 - 1.0;
            (noise * gate).clamp(-1.0, 1.0)
        })
        .collect()
}

/// Generate warm-up audio that reliably triggers the Earshot neural VAD.
///
/// Attempts TTS synthesis first (guaranteed speech-like harmonics), falling
/// back to pink noise + 200 Hz tone if TTS is unavailable.
///
/// Returns a [`Cow`] that either borrows from the TTS cache (once populated)
/// or owns freshly-generated pink noise (not cached, so TTS is re-evaluated
/// on the next call if it becomes available).
///
/// ## Determinism
/// TTS synthesis uses a fixed seed (947), producing the same
/// PCM output on every call (barring TTS model changes).  The result is cached
/// in [`WARMUP_TTS_CACHE`] so subsequent calls are zero-cost.
///
/// ## Fallback rationale
/// The pink noise + 200 Hz tone was the original warm-up signal,
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
        947, // fixed seed for determinism
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

/// Generate the original pink noise + 200 Hz tone warm-up signal.
///
/// Retained as a fallback when TTS models are not cached.  Properties:
/// 1. **VAD-inactive**: low probability of triggering the Earshot neural VAD
///    (no speech harmonics), which is the whole reason we now prefer TTS.
/// 2. **No false detection**: soft scores are near-zero because pink
///    noise + 200 Hz tone does not resemble the "hey mahbot" embedding.
/// 3. **Deterministic**: RNG seed 922 (= the original warm-up signal seed).
/// 4. **Aperiodic noise floor**: pink noise avoids periodic artefacts that
///    could bias the VAD steady state.
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

/// Process a single frame of audio through the pipeline
/// ([`super::handle_wake_word_detection`]), matching the per-frame processing
/// in [`run_streaming_detection`].
///
/// Partial chunks are passed through **without zero-padding**:
/// production feeds raw mic chunks as-is, and `handle_wake_word_detection`
/// accumulates partial tails in `ctx.audio_buffer` until the next real
/// chunk completes a [`FRAME_LENGTH`](super::FRAME_LENGTH) frame — the tail is
/// carried forward.  Zero-padding instead would shift the VAD/segment cadence
/// at utterance boundaries from production.
///
/// Empty slices are a no-op (the detection frame loop needs a complete frame) —
/// silence must be fed as actual [`FRAME_LENGTH`](super::FRAME_LENGTH)
/// digital-silence chunks, which is what production's mic delivers.
///
/// Extracted as a shared helper to eliminate the structural near-duplicate
/// between `feed_audio` and `run_streaming_detection`.
fn process_frame(samples: &[f32], ctx: &mut super::PipelineCtx) {
    super::handle_wake_word_detection(samples, ctx);
}

/// Feed audio through the production pipeline ([`super::handle_wake_word_detection`])
/// in [`FRAME_LENGTH`](super::FRAME_LENGTH) chunks, then send silence frames to flush any
/// remaining detection state — matching the processing pattern in [`run_streaming_detection`].
///
/// This is used to pre-warm the pipeline before the actual test utterance,
/// without contaminating latency measurements.
fn feed_audio(samples: &[f32], ctx: &mut super::PipelineCtx) {
    for chunk in samples.chunks(super::FRAME_LENGTH) {
        process_frame(chunk, ctx);
    }
    // Feed digital-silence frames to flush any remaining detection state (matches
    // the post-audio flush in run_streaming_detection).  Real zeros in
    // FRAME_LENGTH chunks, NOT empty Vecs: production's mic keeps delivering
    // 512-sample chunks of silence after speech — an empty chunk would never
    // produce a complete frame for the detection loop.
    for _ in 0..3 {
        process_frame(&vec![0.0; super::FRAME_LENGTH], ctx);
    }
}

/// Feed the warm-up audio through [`feed_audio`] so the warm pass starts the
/// test utterance with a ring-primed, segment-warmed path.
///
/// ## Diagnostics
///
/// If the warm-up audio triggered a false detection, a `warn!()` is emitted and
/// the detection state is restored to prevent cooldown from corrupting subsequent
/// benchmark measurements.
///
/// ## Residue clearing
///
/// After feeding warm-up audio, this function clears warm-up residues that would
/// otherwise contaminate the subsequent test utterance's detection pipeline:
///
/// - **`score_window`** — cleared so warm-up soft scores cannot carry into
///   the first test embedding's rolling sum.
/// - **`segment_silence_hops`** — reset to 0 so the test utterance starts a fresh
///   segment (the warm-up flush's VAD-negative hops must not shorten the test
///   utterance's trailing-silence window).
/// - **`last_score_sample_count`** — reset so the first test-utterance scoring
///   step waits for [`SCORE_STRIDE_SAMPLES`](super::SCORE_STRIDE_SAMPLES) of
///   test audio (the warm-up feed's stride counter must not shorten it).
///
/// **Preserved**: `audio_buffer` (the trailing raw ring — the encoder window
/// spans the most recent second of audio, so warm-up context is exactly what
/// production's ring carries), `adaptive_threshold` (preserves warm-up-adapted
/// state — see the cold-start pass, which instead uses a fresh
/// [`AdaptiveThresholdState::new`]).
fn consume_warmup(ctx: &mut super::PipelineCtx) {
    let before_detection = ctx.last_wake_word_detection;
    let noise = generate_warmup_noise();

    // ── Feed warm-up audio (single pass) ───────────────────────────────────
    feed_audio(&noise, ctx);

    // Guard: warm-up audio should not trigger a false detection.  If it does,
    // restore the pre-warm-up state so cooldown doesn't suppress all subsequent
    // benchmark measurements.  We must also
    // reset `is_recording` because detection sets it to `true`, which would
    // suppress scoring for the entire benchmark variant.
    if ctx.last_wake_word_detection != before_detection {
        warn!(
            "Warm-up triggered a false detection — restoring detection \
             state to prevent cooldown corruption.",
        );
        ctx.last_wake_word_detection = before_detection;
        ctx.is_recording = false;
    }

    // ── Clear warm-up residues before test utterance processing ──
    // Warm-up audio produces soft scores and VAD hops.  These residues
    // must be cleared so the test utterance starts with a clean detection slate:
    // warm-up scores would otherwise mix into the test utterance's rolling sum,
    // and stale segment-silence hops would shorten the test window.  The raw
    // audio ring and VAD cursor are also reset so the first test-utterance
    // encoder window is not diluted by warm-up noise.
    ctx.score_window.clear();
    ctx.segment_silence_hops = 0;
    ctx.last_score_sample_count = 0;
    ctx.audio_buffer.clear();
    ctx.speech_window.clear();
    ctx.vad_cursor = 0;

    // ── Reset instrumentation ─────────────────────────────
    // Warm-up audio passes through the full scoring pipeline and records into
    // ctx.instrumentation: per_frame_scores, peak_score, VAD counts, and the
    // adaptive threshold trajectory.  Without a reset, warm-up-only scores
    // contaminate the test utterance's per-variant metrics (silence/noise
    // negatives would report "detection triggers" from warm-up speech).
    ctx.instrumentation = super::DetectionInstrumentation::new();
}

// ── Prerequisite check ─────────────────────────────────────────────────────

/// Ensure the shared Qwen3-ASR model is loaded (the wake-word pipeline's
/// encoder — no separate OpenWakeWord ONNX artifacts exist anymore).
/// Returns an error if loading fails, with a helpful message.
fn ensure_voice_models_loaded() -> Result<(), String> {
    if super::models_ready() {
        return Ok(());
    }

    // Await the shared transcriber load.  `run_internal` runs inside a tokio
    // spawn_blocking task, so a runtime handle is usually present; fall back to
    // a fresh current-thread runtime otherwise (same pattern as
    // `faph_download_file`).
    let loaded = if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.block_on(crate::audio::local_transcriber::try_init_from_cache())
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("cannot build transcriber-init runtime: {e}"))?;
        rt.block_on(crate::audio::local_transcriber::try_init_from_cache())
    };

    if loaded {
        info!("Shared Qwen3-ASR model loaded for wake-word benchmark");
        return Ok(());
    }

    // A concurrent boot-path init may own the load — poll briefly before
    // declaring failure (the load is ~4 s with a warm page cache).
    if !crate::audio::local_transcriber::is_failed() {
        for _ in 0..75 {
            // 15 s of 200 ms polls
            if crate::audio::local_transcriber::is_loaded() {
                info!("Shared Qwen3-ASR model loaded for wake-word benchmark (concurrent init)");
                return Ok(());
            }
            if crate::audio::local_transcriber::is_failed() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }

    Err(
        "Failed to load the shared Qwen3-ASR model (missing/corrupt cache or \
         download failure). Run the application first to download models."
            .to_string(),
    )
}

/// Apply PCM transforms to enrollment variants — DETECTION-diagnostics only.
///
/// For each raw TTS variant, produces the full 12-cell recipe: original,
/// speed-down (0.95×/0.90×), speed-up (1.05×, conditional on ≥500 ms),
/// volume-down (-3 dB), pink noise (25 dB SNR), and white/pink/brown noise at
/// 10/5 dB SNR.
///
/// The encoder pipeline has no trainable head — the enrollment prototype is
/// built from the raw VAD-gated utterances only — so this recipe is a
/// detection probe: augmented held-out clips are fed through streaming
/// detection (Phase 7 diagnostics) to measure detection under
/// speed/volume/noise perturbation.
fn pcm_augment_enrollment_variants(variants: &[(Vec<f32>, String)]) -> Vec<(Vec<f32>, String)> {
    use crate::util::NoiseColor;
    let mut all = Vec::new();
    for (i, (pcm, label)) in variants.iter().enumerate() {
        // Variant indices and push order are stable across runs:
        //   pre-pad gate input = raw PCM; CONTEXT_PADDING = 1600 samples.
        let pre_pad_samples = pcm.len().saturating_sub(2 * 1600);
        let pre_pad_ms = pre_pad_samples as u64 * 1000 / u64::from(TARGET_SAMPLE_RATE);
        let noise_seed = i as u64;
        let cells: Vec<(usize, Vec<f32>)> = {
            let mut c = Vec::with_capacity(12);
            c.push((0, pcm.clone()));
            c.push((
                1,
                crate::util::speed_perturbation(pcm, TARGET_SAMPLE_RATE, 0.95),
            ));
            if pre_pad_ms >= 500 {
                c.push((
                    2,
                    crate::util::speed_perturbation(pcm, TARGET_SAMPLE_RATE, 1.05),
                ));
            }
            c.push((3, crate::util::apply_gain(pcm, -3.0)));
            c.push((4, crate::util::add_noise(pcm, 25.0, noise_seed)));
            c.push((
                5,
                crate::util::speed_perturbation(pcm, TARGET_SAMPLE_RATE, 0.90),
            ));
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
                c.push((
                    6 + idx,
                    crate::util::add_noise_color(pcm, snr, color, noise_seed),
                ));
            }
            c
        };
        for (variant_index, pcm) in cells {
            let suffix = match variant_index {
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
                _ => unreachable!("recipe yields only indices 0..=11"),
            };
            all.push((pcm, format!("{label}_{suffix}")));
        }
    }
    all
}

/// Locate the enrolled-speaker training clip (label `{style}_enroll0` — the
/// voice derived from [`allocate_voices`]) and build its full 12-cell
/// augmentation variants — the canonical construction used by the Phase 13
/// cooldown re-point.  `None` when the enrolled clip is absent
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

/// Report-only: quality scoring of the bench's enrollment
/// clips via production's compute_utterance_quality.  The bench clips have no
/// pre-speech noise ring → noise_rms is None → SNR is the estimate_snr_energy
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
        "note": "Heuristic SNR (estimate_snr_energy) — the bench's enrollment clips have no pre-speech noise ring, so this is NOT real-room SNR.",
        "n_clips": n_clips,
        "n_clipping": n_clipping,
        "mean_heuristic_snr_db": if n_clips == 0 { 0.0 } else { snr_sum / n_clips as f32 },
        "mean_composite_score": if n_clips == 0 { 0.0 } else { score_sum / n_clips as f32 },
        "per_clip": per_clip,
    })
}

/// VAD-gate raw PCM through a dedicated earshot detector (per 256-sample hop,
/// never the global `VAD_DETECTOR`), returning only the VAD-positive hops
/// concatenated — the "VAD-gated speech" the negative generators encode.
///
/// The detector's internal state carries across hops exactly like the
/// streaming path's per-segment detector, so speech onsets/offsets are judged
/// on the continuous stream.
fn vad_gate_speech(pcm: &[f32], detector: &mut Detector) -> Vec<f32> {
    let mut speech: Vec<f32> = Vec::new();
    for chunk in pcm.chunks(256) {
        if super::is_speech_with_detector(chunk, detector, super::VAD_THRESHOLD) {
            speech.extend_from_slice(chunk);
        }
    }
    speech
}

/// Generate owner-negative embeddings (non-wake-word TTS phrases, re-scoped).
///
/// These are TTS-synthesised phrases from the ENROLLED voice only
/// (single-speaker semantics — the owner's own non-wake speech); the
/// multi-voice style rotation is gone.  Documented limitation: TTS speech
/// cannot match the distribution of real human Phase 3 speech.
///
/// ## Alignment with the encoder pipeline
///
/// Production captures owner negatives as real mic audio (Phase 3 collection)
/// and encodes them through the shared Qwen3-ASR encoder
/// ([`crate::audio::wake_word::encode_window`]).  The benchmark's TTS
/// surrogates are therefore VAD-gated through a dedicated earshot detector
/// (per-clip fresh, mirroring the streaming path's per-segment detector) and
/// the VAD-positive speech is encoded once.  A bounded low-SNR cell (brown
/// noise at 10 dB on the VAD-gated speech) is emitted per phrase × seed — the
/// enrolled voice under noise must not match the prototype.
///
/// Returns one 1024-dim L2-normalized embedding per accepted cell.
fn generate_owner_negative_sequences(
    enrolled_style: &str,
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<Vec<f32>> {
    let Some(model) = crate::audio::local_transcriber::shared_model_arc() else {
        warn!("Owner negatives: shared Qwen3-ASR model not loaded");
        return Vec::new();
    };
    let mut embeddings = Vec::new();
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
                // VAD-gate with a dedicated detector (never the global
                // VAD_DETECTOR) — only VAD-positive hops are encoded.
                let mut detector = Detector::default();
                let speech_audio = vad_gate_speech(&pcm, &mut detector);
                if speech_audio.is_empty() {
                    warn!(
                        "Owner-negative '{phrase}' seed {seed}: no VAD-positive \
                         speech — skipping"
                    );
                    continue;
                }
                match crate::audio::wake_word::encode_window(&model, &speech_audio) {
                    Ok(emb) => embeddings.push(emb),
                    Err(e) => warn!("Owner-negative '{phrase}' seed {seed}: encode failed: {e}"),
                }
                // Bounded low-SNR cell: brown noise at 10 dB on the VAD-gated
                // speech — the enrolled voice under noise must not trigger.
                let noisy = crate::util::add_noise_color(
                    &speech_audio,
                    10.0,
                    crate::util::NoiseColor::Brown,
                    seed_val,
                );
                match crate::audio::wake_word::encode_window(&model, &noisy) {
                    Ok(emb) => embeddings.push(emb),
                    Err(e) => {
                        warn!("Owner-negative '{phrase}' seed {seed}: brown-10 encode failed: {e}");
                    }
                }
            }
        }
    }
    embeddings
}

/// Generate ambient noise embeddings for negative calibration.
///
/// Produces one 1024-dim L2-normalized window embedding per noise profile × 2
/// SNR levels (full amplitude and −6 dB).
///
/// ## Alignment with the encoder pipeline
///
/// Production captures real ambient negatives as raw mic audio collected
/// during Phase 3 and encodes them through the shared Qwen3-ASR encoder.  The
/// benchmark's synthetic noise profiles are encoded directly (one 1 s window
/// per profile × 2 levels) via [`crate::audio::wake_word::encode_window`].
/// Documented limitation: synthetic noise is a surrogate for real ambient
/// capture.
fn generate_ambient_noise_sequences() -> Vec<Vec<f32>> {
    let Some(model) = crate::audio::local_transcriber::shared_model_arc() else {
        warn!("Ambient negatives: shared Qwen3-ASR model not loaded");
        return Vec::new();
    };
    let mut embeddings = Vec::new();
    for (label, noise_fn) in NOISE_PROFILES {
        let raw = noise_fn();
        // Level 1: full amplitude
        match crate::audio::wake_word::encode_window(&model, &raw) {
            Ok(emb) => embeddings.push(emb),
            Err(e) => warn!("Ambient '{label}' level-0: encode failed: {e}"),
        }
        // Level 2: reduced amplitude (-6dB)
        let attenuated = crate::util::apply_gain(&raw, -6.0);
        match crate::audio::wake_word::encode_window(&model, &attenuated) {
            Ok(emb) => embeddings.push(emb),
            Err(e) => warn!("Ambient '{label}' level-1: encode failed: {e}"),
        }
    }
    embeddings
}

/// Bench-local restricted-style confusable/unrelated negative embeddings.
///
/// Production's old prewarm rotated ALL 10 TTS voices into the training
/// negatives, which would train the reserved cross-speaker voices (M4/M5)
/// in — the bench requires them absent from EVERY training path.  This
/// replicates the encoder pipeline's negative recipe over a restricted style
/// list: synthesize via the shared PCM cache, VAD-gate through a dedicated
/// earshot detector, encode the VAD-positive speech via
/// [`crate::audio::wake_word::encode_window`].
///
/// There is no embedding-level disk cache; the bench pays the encoder forwards
/// on each run — the PCM cache keeps TTS synthesis cheap.
fn generate_restricted_phrase_negatives(
    phrase_type: &'static str,
    phrases: &'static [&'static str],
    seeds_per_phrase: usize,
    seed_base: u64,
    styles: &[String],
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<Vec<f32>> {
    let Some(model) = crate::audio::local_transcriber::shared_model_arc() else {
        warn!("{phrase_type} negatives: shared Qwen3-ASR model not loaded");
        return Vec::new();
    };
    if styles.is_empty() {
        // No negative-pool styles (fewer-than-6 TTS voices) — the call-site
        // non-empty asserts below fail loudly rather than calibrating a weaker
        // enrollment.
        return Vec::new();
    }
    let num_styles = styles.len();
    let mut embeddings: Vec<Vec<f32>> = Vec::new();

    for (i, &phrase) in phrases.iter().enumerate() {
        for seed_idx in 0..seeds_per_phrase {
            // Same round-robin style formula as production, restricted to the
            // negative-pool styles.
            let style_idx = (i * seeds_per_phrase + seed_idx) % num_styles;
            let style = &styles[style_idx];
            let seed = seed_base + i as u64 * seeds_per_phrase as u64 + seed_idx as u64;

            // PCM from cache or fresh TTS synthesis.
            let Some(pcm) = super::synthesize_with_pcm_cache(
                phrase,
                style,
                seed,
                TARGET_SAMPLE_RATE,
                model_hash,
                cache_dir,
            ) else {
                continue;
            };

            // VAD-gate with a dedicated detector (never the global
            // VAD_DETECTOR) — only VAD-positive speech is encoded.
            let mut detector = Detector::default();
            let speech_audio = vad_gate_speech(&pcm, &mut detector);
            if speech_audio.is_empty() {
                warn!(
                    "{phrase_type} phrase '{phrase}' (seed {seed}) produced no \
                     VAD-positive speech — skipping (matches streaming: no \
                     speech ⇒ no embeddings)"
                );
                continue;
            }
            match crate::audio::wake_word::encode_window(&model, &speech_audio) {
                Ok(emb) => embeddings.push(emb),
                Err(e) => {
                    warn!("{phrase_type} phrase '{phrase}' (seed {seed}): encode failed: {e}");
                }
            }
        }
    }
    embeddings
}

/// Compute VAD frame decisions and segment audio into utterances at the
/// enrollment VAD threshold, using a fresh earshot detector (never the global
/// `VAD_DETECTOR`).
///
/// Each frame (stride = [`HOP_LENGTH`](super::HOP_LENGTH)) feeds only its NEW
/// [`HOP_LENGTH`](super::HOP_LENGTH) samples to the detector — the production
/// feed contract — and the decisions are passed to
/// [`super::segment_utterances_by_vad`] with the standard pipeline config.
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
        // to maintain train-inference consistency.
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

/// Report-only: regression-guard the VAD feed
/// contract on the BENCH side.
///
/// Production's VAD feed pattern (voice.rs mic arm) feeds ONLY
/// the new [`HOP_LENGTH`](super::HOP_LENGTH) samples per hop through a
/// streaming earshot detector — never the full overlapping
/// [`FRAME_LENGTH`](super::FRAME_LENGTH) window (double-feeding corrupts
/// earshot's ring buffer).  This probe replays that contract literally
/// against a fresh detector and compares the decisions with the bench's
/// [`compute_vad_segments`] (which implements the same pattern).  If
/// [`compute_vad_segments`] ever drifts from the literal contract, the
/// comparison flags it.
///
/// Scope: production code is NOT executed in the offline bench, so
/// this cannot detect drift in the production arm itself — it locks the
/// bench's transcription of the contract.  To prove the comparison is
/// sensitive (not vacuous), a NEGATIVE control feeds the full overlapping
/// `FRAME_LENGTH` window per hop (the exact violation the feed contract forbids);
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

    // Reference replay of production's feed pattern: a FRESH
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
    // the exact feed-pattern violation the feed contract forbids.  If the correct
    // and violated feeds produce the same decisions on this audio
    // (feed_pattern_sensitive == false), the main comparison would be
    // vacuous and the report states the limitation explicitly.
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
            "feed_pattern": "full_overlapping_frame_per_hop (negative control)",
            "mismatches_vs_bench": n_double_fed_mismatches,
            "feed_pattern_sensitive": n_double_fed_mismatches > 0,
        },
        "note": "Bench-side regression guard for the VAD feed contract: \
                 replays production's feed pattern (feed ONLY the new HOP_LENGTH \
                 samples per hop) through a fresh Detector and compares against \
                 compute_vad_segments.  Production code is not executed in the \
                 offline bench, so drift in the production arm itself is not \
                 detectable here.  The negative control (full overlapping \
                 FRAME_LENGTH per hop) reports how many hops diverge, proving the \
                 comparison is sensitive to the feed pattern rather than vacuous.",
    })
}

/// Mel-normalization consistency probe (report-only).
///
/// `qwen_asr::audio::mel_spectrogram` normalizes each call by the global max
/// of its input.  Enrollment embeds VAD-segmented utterances directly
/// (whole-utterance [`encode_window`]); streaming detection accumulates
/// VAD-positive hops into the speech window and encodes its trailing
/// [`WINDOW_SAMPLES`].  The acceptance criterion requires mel-normalization
/// consistency between the two paths to be VALIDATED within the benchmark.
///
/// The probe replays one real enrollment clip through BOTH paths and verifies
/// the resulting embeddings are near-identical (cosine > 0.95): both feed the
/// encoder the same VAD-gated speech, so the per-call global-max
/// normalization sees near-identical input by construction.  The residual
/// below 1.0 comes from VAD boundary asymmetries at the phrase edges (a hop
/// near the VAD threshold classified differently by the two fresh detectors),
/// NOT from mel normalization — the ~0.03 offset is already baked into the
/// pipeline calibration (the acceptance-measured recall was produced by
/// exactly this enrollment/streaming pairing).  This is the explicit check
/// the module docs refer to (report-only, never gated).
fn mel_normalization_consistency_probe(clip: &[f32]) -> serde_json::Value {
    let Some(model) = crate::audio::local_transcriber::shared_model_arc() else {
        return serde_json::json!({
            "skipped": true,
            "note": "shared Qwen3-ASR model not loaded — cannot encode",
        });
    };

    // Path A — enrollment style: VAD-segment the clip, encode the first
    // utterance whole (encode_window takes its trailing WINDOW_SAMPLES).
    // `compute_vad_segments` only terminates an utterance on
    // ENROLLMENT_SILENCE_THRESHOLD_SAMPLES (~304 ms) of trailing silence — a
    // bare wake-word clip ends mid-speech and never yields an utterance — so
    // pad with digital silence first (the same trick vad_segment_and_enroll
    // uses between concatenated clips).
    let mut padded = clip.to_vec();
    padded.extend(vec![
        0.0f32;
        super::ENROLLMENT_SILENCE_THRESHOLD_SAMPLES + 4096
    ]);
    let (_vad_decisions, utterances) = compute_vad_segments(&padded);
    let Some(utterance) = utterances.first() else {
        return serde_json::json!({
            "skipped": true,
            "note": "no VAD utterance found in the probe clip",
        });
    };
    let Ok(whole_emb) = crate::audio::wake_word::encode_window(&model, utterance) else {
        return serde_json::json!({
            "skipped": true,
            "note": "whole-utterance encode failed",
        });
    };

    // Path B — streaming style: accumulate only VAD-positive hops into a
    // speech window capped at WINDOW_SAMPLES (exactly the accumulation
    // `handle_wake_word_detection` performs), then encode the trailing window.
    let mut speech_window: Vec<f32> = Vec::new();
    let mut detector = Detector::default();
    let mut hop_start = 0usize;
    while hop_start + super::FRAME_LENGTH <= clip.len() {
        let new_samples = &clip[hop_start..hop_start + super::HOP_LENGTH];
        if super::is_speech_with_detector(new_samples, &mut detector, super::VAD_THRESHOLD) {
            speech_window.extend_from_slice(new_samples);
            let cap = crate::audio::wake_word::WINDOW_SAMPLES;
            if speech_window.len() > cap {
                let excess = speech_window.len() - cap;
                speech_window.drain(..excess);
            }
        }
        hop_start += super::HOP_LENGTH;
    }
    let Ok(stream_emb) = crate::audio::wake_word::encode_window(&model, &speech_window) else {
        return serde_json::json!({
            "skipped": true,
            "note": "streaming-style encode failed",
        });
    };

    let cosine = crate::vector::cosine_similarity(&whole_emb, &stream_emb);
    serde_json::json!({
        "skipped": false,
        "cosine": cosine,
        "consistent": cosine > 0.95,
        "whole_utterance_samples": utterance.len(),
        "speech_window_samples": speech_window.len(),
        "note": "Both paths encode the same VAD-gated speech through the same \
                 trailing WINDOW_SAMPLES window, so the mel frontend's per-call \
                 global-max normalization sees near-identical input.  The \
                 residual below 1.0 is VAD boundary asymmetry at the phrase \
                 edges (near-threshold hops), not mel normalization — the \
                 ~0.03 offset is already part of the calibrated \
                 enrollment/streaming pairing.  This is the explicit \
                 mel-normalization consistency check the acceptance criterion \
                 calls for (report-only).",
    })
}

/// Process enrollment clips through VAD-gated utterance segmentation and
/// encode each utterance through the shared Qwen3-ASR encoder.
///
/// The encoder pipeline has no trainable head and no AGC/NS preprocessing:
/// each raw TTS clip is VAD-segmented (fresh earshot detector per clip via
/// [`compute_vad_segments`]) and every utterance is embedded once via
/// [`crate::audio::wake_word::encode_window`], producing one 1024-dim
/// L2-normalized embedding per utterance.  The 12-cell PCM augmentation
/// survives only as a detection-diagnostics probe ([`pcm_augment_enrollment_variants`]).
///
/// Returns `Vec<Vec<f32>>` — one embedding per VAD utterance.
fn vad_segment_and_enroll(enrollment_variants: &[(Vec<f32>, String)]) -> Vec<Vec<f32>> {
    const SILENCE_GAP_SAMPLES: usize = 2 * 16_000; // 2.0 s at 16 kHz

    let Some(model) = crate::audio::local_transcriber::shared_model_arc() else {
        panic!("FATAL: shared Qwen3-ASR model not loaded — cannot run VAD-gated enrollment");
    };

    // Concatenate all clips with 2.0 s digital-silence gaps, then VAD-segment
    // the combined stream.  `segment_utterances_by_vad` only emits an
    // utterance once it sees `ENROLLMENT_SILENCE_THRESHOLD_SAMPLES` (~304 ms)
    // of trailing silence — a lone clip ending mid-speech never terminates.
    // The concatenation carries no preprocessing (the encoder pipeline
    // consumes raw audio) so every clip yields exactly one utterance.
    let silence: Vec<f32> = vec![0.0f32; SILENCE_GAP_SAMPLES];
    let mut combined: Vec<f32> = Vec::new();
    for (pcm, _label) in enrollment_variants {
        if !combined.is_empty() {
            combined.extend_from_slice(&silence);
        }
        combined.extend_from_slice(pcm);
    }
    combined.extend_from_slice(&silence);

    let (_vad_decisions, utterances) = compute_vad_segments(&combined);
    info!(
        "VAD concatenation: {} total samples ({:.1}s) from {} originals with 2.0s gaps",
        combined.len(),
        combined.len() as f64 / f64::from(super::SAMPLE_RATE),
        enrollment_variants.len(),
    );
    info!(
        "VAD segmentation: {} utterances from {} concatenated originals",
        utterances.len(),
        enrollment_variants.len(),
    );

    let mut utterance_embeddings: Vec<Vec<f32>> = Vec::new();
    for (i, utterance) in utterances.iter().enumerate() {
        match crate::audio::wake_word::encode_window(&model, utterance) {
            Ok(emb) => utterance_embeddings.push(emb),
            Err(e) => warn!("Enrollment utterance {i}: encode failed: {e}"),
        }
    }

    info!(
        "VAD-gated enrollment: {} utterance embeddings from {} VAD utterances across {} clips \
         (encoder pipeline — one embedding per utterance, no augmentation)",
        utterance_embeddings.len(),
        utterances.len(),
        enrollment_variants.len(),
    );

    utterance_embeddings
}

// ── Streaming detection ─────────────────────────────

/// Result from [`run_streaming_detection`].
struct DetectionResult {
    detected: bool,
    /// Latency in milliseconds from feed start to detection (only meaningful
    /// when `detected` is true).
    latency_ms: Option<f64>,
    /// Adaptive threshold state captured at the end of the test utterance
    /// (before the trailing silence flush).  A segment boundary fired during
    /// the flush calls `reset_detection_segment`, which resets
    /// `ctx.adaptive_threshold` to bootstrap; carrying this pre-flush snapshot
    /// forward lets the benchmark's shared-state design preserve
    /// the adaptation accumulated from the utterance's frames instead of
    /// re-bootstrapping per variant.
    adaptive_state_pre_flush: super::AdaptiveThresholdState,
}

/// Run the production streaming wake word detection pipeline on audio samples.
///
/// Feeds audio through [`handle_wake_word_detection`] in FRAME_LENGTH chunks,
/// exercising the full streaming chain: raw accumulation, VAD gating,
/// stride-gated window encoding, [`score_single_embedding`], and cooldown
/// logic.
///
/// After all audio is fed, silence frames are sent to flush any remaining
/// detection state (matching how the production pipeline handles
/// speech→silence transitions).
///
/// Returns a [`DetectionResult`] with a flag and optional latency measurement.
fn run_streaming_detection(samples: &[f32], ctx: &mut super::PipelineCtx) -> DetectionResult {
    let feed_start = Instant::now();
    // Save pre-existing timestamp — we only return true if detection fires
    // during THIS call, not because a prior call already set the field.
    let before = ctx.last_wake_word_detection;

    // Feed audio in FRAME_LENGTH chunks through the raw-sample streaming path
    // (no AGC/NS — the encoder pipeline consumes raw audio directly).
    for chunk in samples.chunks(super::FRAME_LENGTH) {
        process_frame(chunk, ctx);
        if ctx.last_wake_word_detection != before {
            let latency = feed_start.elapsed().as_secs_f64() * 1000.0;
            return DetectionResult {
                detected: true,
                latency_ms: Some(latency),
                // No flush ran on this path, so the pre-flush snapshot is the
                // current state; the boundary-fire branch in the callers is
                // unreachable when `detected` is true anyway.
                adaptive_state_pre_flush: ctx.adaptive_threshold.clone(),
            };
        }
    }

    // ── Trailing silence flush ───────────────────────────
    // Production continues on natural silence after the utterance and
    // detection can still fire for up to SEGMENT_TIMEOUT_HOPS (19) VAD-negative
    // hops before the segment-boundary reset clears the per-segment state.
    // The old flush fed exactly 3×512 zero frames (~96 ms), which
    // missed wake words whose rolling sum crossed threshold during the longer
    // production window (~304 ms).
    //
    // The adaptive threshold state is captured BEFORE the flush: a segment
    // boundary fired during the flush resets it to bootstrap, and the
    // benchmark's shared-state design wants the adapted state to survive the
    // boundary rather than re-bootstrapping per variant.
    let adaptive_state_pre_flush = ctx.adaptive_threshold.clone();
    //
    // Feed FRAME_LENGTH digital-silence chunks (production's mic delivers real
    // zeros, not empty Vecs — see process_frame) until:
    //   1. detection fires, or
    //   2. the segment boundary fires, or
    //   3. we exceed the production window (SEGMENT_TIMEOUT_HOPS hops plus a
    //      small allowance for any remaining VAD-positive utterance tail).
    //
    // The boundary is detected via the rolling score window:
    // reset_detection_segment clears it (and the stride-gated scorer also
    // clears it via NO_MATCH_RESET when a trailing-silence window scores
    // below the reset threshold), so an empty window means no further
    // detection is possible — further silence cannot produce a detection.
    let mut silence_chunks = 0usize;
    while silence_chunks < super::SEGMENT_TIMEOUT_HOPS + 4 {
        if ctx.last_wake_word_detection != before {
            let latency = feed_start.elapsed().as_secs_f64() * 1000.0;
            return DetectionResult {
                detected: true,
                latency_ms: Some(latency),
                adaptive_state_pre_flush,
            };
        }
        let window_was_nonempty = !ctx.score_window.is_empty();
        process_frame(&vec![0.0; super::FRAME_LENGTH], ctx);
        silence_chunks += 1;
        // Segment boundary (or NO_MATCH reset) emptied the rolling window —
        // further silence cannot produce a detection, so stop feeding.
        if window_was_nonempty && ctx.score_window.is_empty() {
            break;
        }
    }

    DetectionResult {
        detected: ctx.last_wake_word_detection != before,
        latency_ms: None,
        adaptive_state_pre_flush,
    }
}

/// Sleep until the ctx's last wake-word detection is at least `target_elapsed`
/// old, measured from the [`PipelineCtx::last_wake_word_detection`] field (the
/// timestamp set inside the feed loop), NOT from `run_streaming_detection`'s
/// return time.
///
/// Detection sets the cooldown timestamp *inside* the feed loop
/// (`handle_wake_word_detection` → `ctx.last_wake_word_detection = Some(now)`),
/// so returning from `run_streaming_detection` adds a variable
/// detection-to-return delta.  Sleeping relative to the timestamp itself
/// avoids that jitter when probing the cooldown gate at specific elapsed
/// times.
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
    /// Peak per-frame total_score (cosine soft score, range [0,1])
    /// achieved during processing. This is NOT the rolling sum — it's the
    /// maximum single-frame score. For rolling sum analysis, use the
    /// `per_frame_scores` field or `max_rolling_sum`.
    peak_score: f32,
    /// Maximum rolling sum (sum of 3 consecutive total_scores)
    /// achieved during processing. Derived from per_frame_scores.
    max_rolling_sum: f32,
    /// Number of embeddings scored during streaming detection
    /// (the length of `per_frame_scores` — one entry per scored window
    /// embedding after the warm-up instrumentation reset).
    n_embeddings: usize,
    /// Number of frames where total_score < NO_MATCH_RESET_THRESHOLD (0.35).
    n_frames_below_reset: usize,
    /// Count of VAD-positive 512-sample frames during streaming detection.
    vad_speech_frames: usize,
    /// Per-frame `[total_score, rolling_sum, threshold]` triples from
    /// scoring.  Use `ROLLING_SUM_IDX` / `THRESHOLD_IDX` for named
    /// access to the rolling sum and effective threshold fields respectively.
    per_frame_scores: Vec<[f32; 3]>,
    /// Index of the first trigger frame within `per_frame_scores`.
    /// `None` when detection never triggered on test-utterance frames.
    first_trigger_frame_idx: Option<usize>,
    /// Number of embeddings attributed to the test utterance — the length of
    /// `per_frame_scores` (one entry per scored embedding after the warm-up
    /// instrumentation reset).
    n_test_embeddings: usize,
    /// Per-frame adaptive threshold trajectory (parallel to `per_frame_scores`).
    adaptive_threshold_trajectory: Vec<f32>,
    /// Count of frames where the effective threshold hit ADAPTIVE_CEILING.
    ceiling_limited_frames: usize,
    /// Detection latency in ms for this variant (only meaningful when
    /// `detected` is true; `None` otherwise).  Emitted per-variant so latency
    /// outliers can be root-caused.
    latency_ms: Option<f64>,
    /// Per-hop VAD decisions during streaming detection (one per VAD decision
    /// in processing order).
    per_hop_vad: Vec<bool>,
}

/// Build a [`PerVariantResult`] from a completed [`run_streaming_detection`]
/// session.  Shared by `test_detection_samples` and `run_noise_overlap_test`
/// so the instrumentation-derived fields are populated identically everywhere
/// (negatives must carry the same detail as positives).
fn build_per_variant_result(
    label: &str,
    result: &DetectionResult,
    ctx: &super::PipelineCtx,
) -> PerVariantResult {
    let peak = ctx.instrumentation.peak_score;
    let max_rs = max_rolling_sum(&ctx.instrumentation.per_frame_scores);
    // Test-utterance embedding count.  `per_frame_scores` is reset at the end
    // of consume_warmup and grows one entry per scored
    // embedding, so its length IS the test-utterance embedding count.
    let n_test_embeddings = ctx.instrumentation.per_frame_scores.len();
    PerVariantResult {
        variant: label.to_string(),
        detected: result.detected,
        peak_score: peak,
        max_rolling_sum: max_rs,
        n_embeddings: n_test_embeddings,
        n_frames_below_reset: ctx.instrumentation.n_frames_below_reset,
        vad_speech_frames: ctx.instrumentation.vad_speech_frames,
        per_frame_scores: ctx.instrumentation.per_frame_scores.clone(),
        first_trigger_frame_idx: ctx.instrumentation.first_trigger_frame_idx,
        n_test_embeddings,
        adaptive_threshold_trajectory: ctx.instrumentation.adaptive_threshold_trajectory.clone(),
        ceiling_limited_frames: ctx.instrumentation.ceiling_limited_frames,
        latency_ms: result.latency_ms,
        per_hop_vad: ctx.instrumentation.per_hop_vad.clone(),
    }
}

// ── Exhaustive per-variant miss verdicts ─────────────────

/// Exhaustive per-variant verdict for positive (wake word) test variants.
///
/// Replaces the old 3-way bucket which collapsed at least four distinct
/// failure modes into a single label and mis-bucketed zero-embedding misses
/// (VAD never fired) as "score gate".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MissVerdict {
    /// Wake word detected on this variant.
    Detected,
    /// VAD produced no speech frames (or the test utterance produced zero
    /// embeddings) — the pipeline never saw test audio.
    VadFailure,
    /// Rolling sum never reached [`MIN_GATE_THRESHOLD`] (1.65) on
    /// test-utterance frames only.
    ScoreGateNoTrigger,
    /// Crossed the hard floor ([`MIN_GATE_THRESHOLD`] = 1.65) but never
    /// crossed the effective adaptive threshold.
    AdaptiveThresholdBlocked,
}

impl MissVerdict {
    /// Stable snake_case label for JSON output.
    fn as_str(self) -> &'static str {
        match self {
            Self::Detected => "detected",
            Self::VadFailure => "vad_failure",
            Self::ScoreGateNoTrigger => "score_gate_no_trigger",
            Self::AdaptiveThresholdBlocked => "adaptive_threshold_blocked",
        }
    }
}

/// Classify a positive variant into an exhaustive verdict.
///
/// ## Decision procedure (precedence, highest first)
///
/// 1. `detected` — no miss.
/// 2. `vad_failure` — VAD never fired (zero speech frames) OR no embeddings
///    were scored during the test utterance (`per_frame_scores` is empty after
///    the warm-up instrumentation reset).  Score-based verdicts are meaningless
///    when the pipeline never saw test audio.
/// 3. `score_gate_no_trigger` — never reached [`MIN_GATE_THRESHOLD`]
///    (1.65) on test-utterance frames.
/// 4. `adaptive_threshold_blocked` — reached the hard floor (1.65) but never
///    reached the per-frame effective adaptive threshold.  "Hard floor" here is
///    [`MIN_GATE_THRESHOLD`] (the trigger condition), NOT
///    [`ADAPTIVE_SAFE_HARBOR`](super::ADAPTIVE_SAFE_HARBOR) (the adaptive
///    threshold's own clamp floor, which is an internal detail of the adaptive
///    state).
fn classify_miss(pv: &PerVariantResult) -> MissVerdict {
    if pv.detected {
        return MissVerdict::Detected;
    }
    if pv.vad_speech_frames == 0 || pv.per_frame_scores.is_empty() {
        return MissVerdict::VadFailure;
    }
    if pv.max_rolling_sum < MIN_GATE_THRESHOLD {
        return MissVerdict::ScoreGateNoTrigger;
    }
    // Crossed the hard floor but never the effective threshold.  With no
    // second-stage gate, crossing the effective threshold IS detection, so a
    // trigger without detection means the adaptive threshold never
    // came down to meet the rolling sum.
    MissVerdict::AdaptiveThresholdBlocked
}

/// Extract the maximum rolling sum from per-frame score triples.
///
/// The rolling sum (`ROLLING_SUM_IDX`) is the accumulated soft score with
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
    /// Per-variant detail for positive detection logging.
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

// ── Score-gate trigger metrics ─────────────────────────────────

/// Score-gate-only trigger metrics for a set of negative variants.
///
/// Tracks the rolling-sum gate independently so false-accept behavior is
/// decomposed into gate crossings vs full-pipeline detections.
#[derive(Debug, Default, Clone, Copy)]
struct GateTriggerMetrics {
    /// Total variants tested in this group.
    total_variants: usize,
    /// Number of variants where max_rolling_sum >= MIN_GATE_THRESHOLD.
    gate_triggers: usize,
    /// Number where detected == true (full pipeline false accept).
    full_pipeline_fa: usize,
}

impl GateTriggerMetrics {
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
        if pv.max_rolling_sum >= MIN_GATE_THRESHOLD {
            self.gate_triggers += 1;
            if pv.detected {
                self.full_pipeline_fa += 1;
            }
        }
    }
}

/// Convert gate trigger metrics to a JSON object for the benchmark report.
fn gt_to_json(gt: &GateTriggerMetrics) -> serde_json::Value {
    serde_json::json!({
        "total_variants": gt.total_variants,
        "gate_triggers": gt.gate_triggers,
        "full_pipeline_fa": gt.full_pipeline_fa,
    })
}

/// Full per-variant diagnostics for BOTH positive and negative variants.
/// Negatives historically omitted per-frame scores, VAD detail and
/// trigger-frame info — false-accept root-causing was impossible
/// without them.  For positives (`category: None`) the exhaustive miss verdict
/// and trigger-point evidence are added.
fn pv_to_json(pv: &PerVariantResult, category: Option<&str>) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert(
        "pipeline".to_string(),
        serde_json::json!("qwen3-asr-encoder"),
    );
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
        "vad_speech_frames".to_string(),
        serde_json::json!(pv.vad_speech_frames),
    );
    obj.insert(
        "per_frame_scores".to_string(),
        serde_json::json!(pv.per_frame_scores),
    );
    obj.insert("per_hop_vad".to_string(), serde_json::json!(pv.per_hop_vad));
    obj.insert(
        "first_trigger_frame_idx".to_string(),
        serde_json::json!(pv.first_trigger_frame_idx),
    );
    obj.insert("latency_ms".to_string(), serde_json::json!(pv.latency_ms));
    // Positive-only miss evidence.
    if category.is_none() {
        let verdict = classify_miss(pv);
        obj.insert("verdict".to_string(), serde_json::json!(verdict.as_str()));
        // Trigger-point evidence: the soft score at the
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

// ── Distribution helpers ──────────────────────────────

/// Decile boundaries of a score distribution (10 values, min→max inclusive of
/// each 10% quantile).  Returns `None` for empty input.
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

/// Numeric summary of a score distribution: `(mean, min, max, deciles)`.
/// Returns NaN/`None` fields for empty input.
fn score_stats(scores: &[f32]) -> (f32, f32, f32, Option<[f32; 10]>) {
    if scores.is_empty() {
        return (f32::NAN, f32::NAN, f32::NAN, None);
    }
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    let min = scores.iter().copied().fold(f32::MAX, f32::min);
    let max = scores.iter().copied().fold(f32::MIN, f32::max);
    (mean, min, max, deciles(scores))
}

/// Identify the PCM augmentation type from a variant label.
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
/// - `metrics`: records total/detected; `on_detection` fills detected or
///   false_accepts.
/// - `on_detection`: called with `(&mut metrics, label_str)` when the
///   wake word is detected (for positives: increment `.detected`; for
///   negatives: push to `.false_accepts`).
fn test_detection_samples(
    variants: &[(Vec<f32>, String)],
    metrics: &mut DetectionMetrics,
    on_detection: impl Fn(&mut DetectionMetrics, &str),
    mut adaptive_state: Option<&mut super::AdaptiveThresholdState>,
    cold_start: bool,
) {
    // The enrollment is set in global state ONCE before the detection phases
    // (Phase 4/5 via super::set_enrollment) — handle_wake_word_detection reads
    // it from voice_state() per scoring step.

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
        // continuous pipeline.  Without this the
        // adaptive state never exits its 5-frame bootstrap because each
        // variant gets a fresh ctx, keeping all benchmark metrics measured
        // against the static threshold.
        //
        // Cold start: production bootstraps the adaptive
        // threshold from the first 5 real per-frame scores of actual background
        // audio after each segment boundary.  The cold pass therefore uses a
        // fresh AdaptiveThresholdState::new() per variant (no shared-state
        // propagation) so the threshold trajectory matches production exactly.
        if !cold_start && let Some(ref mut state) = adaptive_state {
            ctx.adaptive_threshold = state.clone();
        }
        // Warm pass: feed warm-up audio so the latency timer measures only the
        // wake word and the ring is primed.  The cold pass skips
        // consume_warmup — production has no warm-up after a silence boundary.
        if !cold_start {
            consume_warmup(&mut ctx);
        }
        let result = run_streaming_detection(samples, &mut ctx);
        // Propagate the updated adaptive state for the next variant (warm pass
        // only — the cold pass keeps each variant's bootstrap independent).
        //
        // When the trailing silence fires a segment boundary (`!detected` AND
        // the rolling window was cleared — reset_detection_segment is the
        // non-detection path that empties the window), the boundary
        // resets ctx.adaptive_threshold back to bootstrap.  Propagating that
        // bootstrap state would defeat the shared-state design
        // which keeps the adaptive code path active across variants to avoid
        // measuring everything against the static threshold — and with the
        // extended flush every non-detecting variant fires the boundary.
        // Instead, carry the pre-flush snapshot captured at the end of the
        // utterance so the shared state keeps adapting to each variant's
        // frames.  The cold pass measures the bootstrap behavior separately.
        if !cold_start && let Some(ref mut state) = adaptive_state {
            let boundary_fired = !result.detected && ctx.score_window.is_empty();
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
        // Record per-variant result and per-variant
        // instrumentation.
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

// ── Noise-overlapped detection test ─────────────────────────

/// Mix speech and noise at a given SNR (in dB).
///
/// `speech` and `noise` should be the same length.  The noise is scaled so
/// that `SNR = 20 * log10(rms_speech / (scale * rms_noise))`.
/// When `snr_db` is `f32::INFINITY`, returns the speech unchanged.
fn mix_at_snr(speech: &[f32], noise: &[f32], snr_db: f32) -> Vec<f32> {
    if !snr_db.is_finite() {
        return speech.to_vec();
    }
    // Shared RMS helper — the 1e-10 degenerate-signal guard lives here as an
    // early return (it is NOT part of compute_rms).
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
/// pre-generated noise buffer mixed at a target SNR (noise-overlap matrix),
/// seeded white noise per clip (cross-speaker probe matrix), or a per-clip
/// slice of a babble track (wake-over-babble).
enum CellMix {
    /// `mix_at_snr(pcm, noise, snr_db)` — infinite SNR returns speech unchanged.
    AtSnr { noise: Vec<f32>, snr_db: f32 },
    /// `add_white_noise(pcm, snr, seed_base + clip_index)`, or the
    /// clip unchanged when `snr_db` is None (clean cell).
    SeededWhite { snr_db: Option<f32>, seed_base: u64 },
    /// Clip `i` mixes against a `~pcm.len()`-long slice of the babble track at
    /// `i * (track.len() - slice_len) / clips.len()` so the full multi-voice
    /// track is sampled across the set (and SNR is computed on the actual
    /// mixed window, not the whole track).
    Babble { track: Vec<f32>, snr_db: f32 },
}

/// Shared per-cell streaming-detection loop for the noise-overlap and
/// cross-speaker-probe matrices.  One warmed adaptive state is carried across
/// all cells (with the boundary-fire guard from `test_detection_samples`),
/// each clip gets a fresh `PipelineCtx`, and the warm-up audio is consumed
/// so the latency timer starts at the utterance.  Returns
/// per-cell `(key, rate, detected, detail)`.
fn run_detection_matrix(
    clips: &[(Vec<f32>, String)],
    cells: &[(String, CellMix)],
    detail_tag: &'static str,
    log_prefix: &'static str,
) -> Vec<(String, f64, usize, Vec<serde_json::Value>)> {
    // The enrollment is global (set once before the detection phases) — the
    // inline loop shares adaptive state across variants.
    // Pre-warm a shared adaptive threshold state so the benchmark actually
    // exercises the adaptive code path.  Without
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
                    Some(snr) => {
                        crate::util::add_white_noise(pcm, *snr, Some(*seed_base + i as u64))
                    }
                    None => pcm.clone(),
                },
                CellMix::Babble { track, snr_db } => {
                    let n = pcm.len().min(track.len());
                    let span = track.len() - n;
                    let start = if span > 0 {
                        (i * span) / clips.len()
                    } else {
                        0
                    };
                    mix_at_snr(pcm, &track[start..start + n], *snr_db)
                }
            };
            metrics.total += 1;

            // Each variant gets a fresh ctx (clean score_window,
            // audio_buffer, etc.) but carries the shared adaptive
            // threshold state forward across detection attempts.
            let mut ctx = super::PipelineCtx::new();
            ctx.adaptive_threshold = shared_adaptive.clone();
            // adaptive_k is already set by PipelineCtx::new() from config.

            // Consume warm-up so the latency timer starts at the
            // utterance.
            consume_warmup(&mut ctx);
            let result = run_streaming_detection(&mixed_pcm, &mut ctx);

            // Persist the updated adaptive state for the next variant,
            // with the same boundary-fire guard as test_detection_samples:
            // when the trailing silence fired a segment
            // boundary (`!detected` AND the rolling window was cleared), the
            // boundary reset ctx.adaptive_threshold to bootstrap.
            // Propagating that bootstrap state would defeat the
            // shared-state design — and with the extended flush every
            // non-detecting variant fires the boundary, so the phase would
            // otherwise re-bootstrap per variant and measure mostly against
            // the static threshold.
            // Carry the pre-flush snapshot instead so the shared state
            // keeps adapting to each variant's frames.
            let boundary_fired = !result.detected && ctx.score_window.is_empty();
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
/// variants with the noise and test detection.  Returns the
/// `(key, rate, detected, per_variant_detail)` cell shape of
/// [`run_detection_matrix`] so callers get the per-cell detection count
/// without re-parsing the per-variant detail.
fn run_noise_overlap_test(
    positive_variants: &[(Vec<f32>, String)],
) -> Vec<(String, f64, usize, Vec<serde_json::Value>)> {
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
    run_detection_matrix(positive_variants, &cells, "noise_overlap", "Noise overlap")
}

/// Cross-speaker probe matrix: held-out RESERVED voices at
/// unseen seeds across clean + 5/10/20 dB white conditions (warm pass, shared
/// adaptive — same methodology as the canary matrix).  The reserved voices
/// are absent from EVERY training path.  Cross-speaker detections are
/// speaker-blind behaviour (the wake word fires for any speaker), so the
/// matrix is a DIAGNOSTIC, not a false-accept gate.
///
/// Returns the per-cell `(key, rate, detected, detail)` shape of
/// [`run_detection_matrix`].
fn run_cross_speaker_probe_matrix(
    probe_clips: &[(Vec<f32>, String)],
) -> Vec<(String, f64, usize, Vec<serde_json::Value>)> {
    // 4 channel conditions per probe clip (clean + 5/10/20 dB white — the same
    // vocabulary as the cross-speaker training-condition set so the probe is
    // apples-to-apples with the trained-in canaries; the 5 dB cell is an
    // additional probe condition).
    let cells = [
        (
            "clean".to_string(),
            CellMix::SeededWhite {
                snr_db: None,
                seed_base: 0,
            },
        ),
        (
            "5db_white".to_string(),
            CellMix::SeededWhite {
                snr_db: Some(5.0),
                seed_base: 7200,
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
    run_detection_matrix(
        probe_clips,
        &cells,
        "cross_speaker_probe",
        "Cross-speaker probe",
    )
}

/// Report-only FA reporter for a detection-matrix result set: any detected
/// cell above the 0-target warns when `warn` is true (true-negative phases),
/// otherwise the detection rate is logged as a diagnostic with no warning
/// (cross-speaker cells — speaker-blindness diagnostics).
fn warn_fa_cells(
    label: &str,
    results: &[(String, f64, usize, Vec<serde_json::Value>)],
    total_variants: usize,
    tail: &str,
    warn: bool,
) {
    for (key, rate, detected, detail) in results {
        if warn && *detected > 0 {
            warn!(
                "{label} FA: {key} accepted {detected}/{detail_len} cells \
                 (rate {rate_pct:.1}%) — target 0/{total_variants} per fresh run{tail}",
                detail_len = detail.len(),
                rate_pct = rate * 100.0,
            );
        } else {
            info!(
                "{label} diagnostic: {key} detected {detected}/{detail_len} cells \
                 (rate {rate_pct:.1}%) — speaker-blindness diagnostic{tail}",
                detail_len = detail.len(),
                rate_pct = rate * 100.0,
            );
        }
    }
}

/// Map one detection-matrix cell to its report JSON, using the returned
/// per-cell detection count (no per-variant re-parse).
fn cell_json(
    (key, rate, detected, detail): &(String, f64, usize, Vec<serde_json::Value>),
) -> (String, serde_json::Value) {
    (
        key.clone(),
        serde_json::json!({
            "detection_rate": rate,
            "detected": detected,
            "per_variant": detail,
        }),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Enrolled-speaker benchmark phase
// ═══════════════════════════════════════════════════════════════════════════

/// Per-variant enrolled-speaker result.
#[derive(Clone)]
#[allow(clippy::struct_excessive_bools)]
struct EnrolledVariantResult {
    variant: String,
    /// End-to-end detection through the real streaming cold pass.
    detected: bool,
    /// Detection latency ms (CPU wall-clock of the feed loop; None when not
    /// detected).  NOTE: the feed loop processes faster than real-time, so
    /// this is a processing-cost proxy, not the "from speech onset" audio
    /// latency.
    latency_ms: Option<f64>,
    /// Per-frame scores `[total_score, rolling_sum, effective_threshold]`.
    per_frame_scores: Vec<[f32; 3]>,
    /// Miss verdict for non-detected variants (see [`MissVerdict`]).
    miss_verdict: Option<MissVerdict>,
}

/// Run ONE enrolled-speaker clip through the real streaming cold pass and
/// build its [`EnrolledVariantResult`] (shared by the F1 training-clip control
/// and the held-out recall set).
///
/// Cold pass: fresh [`PipelineCtx`] per clip, no warm-up, fresh adaptive
/// bootstrap — the production post-silence start, driven for real through
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
        latency_ms: result.latency_ms,
        per_frame_scores: pv.per_frame_scores.clone(),
        miss_verdict: if result.detected {
            None
        } else {
            Some(classify_miss(&pv))
        },
    }
}

/// Enrolled-speaker phase (re-scoped).
///
/// The acceptance basis is the HELD-OUT wake-only recall set: unseen
/// renderings of the enrolled voice (new seeds in a collision-free range,
/// wake phrase alone — embedded-in-sentence detection is not a product
/// requirement and is not measured), measured through the real streaming cold
/// pass (window-encoder scoring + adaptive bootstrap).  Generated
/// strictly after training and never added to any training pool.  All 10
/// enrollment clips are training data, so the in-sample F1 control is
/// removed; detection control is entirely this held-out recall set.
#[expect(clippy::too_many_lines)]
fn run_enrolled_speaker_phase(held_out_clips: &[(Vec<f32>, String)]) -> serde_json::Value {
    let start = Instant::now();

    // The enrollment is global (set once before the detection phases) — the
    // streaming pipeline reads it from voice_state per scoring step.

    // ── Held-out recall pass (cold, per clip) — the acceptance basis ─────
    let mut recall_variants: Vec<EnrolledVariantResult> = Vec::new();
    for (recall_pcm, recall_label) in held_out_clips {
        let v = run_enrolled_cold_variant(recall_label, recall_pcm);
        info!(
            "  Held-out recall clip {}: {} — {} (latency: {:?}ms)",
            recall_variants.len() + 1,
            recall_label,
            if v.detected { "DETECTED" } else { "miss" },
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
    let window_encoder_path = detected_live;
    let primary_mechanism = window_encoder_path;
    let detected_latencies: Vec<f64> = recall_variants
        .iter()
        .filter(|v| v.detected)
        .filter_map(|v| v.latency_ms)
        .collect();
    let mean = |xs: &[f64]| -> Option<f64> {
        if xs.is_empty() {
            None
        } else {
            Some(xs.iter().copied().sum::<f64>() / xs.len() as f64)
        }
    };

    // ── Near-miss canaries ────────────────────────────
    // Investigation triggers, NOT hard gates: firing means the run deserves
    // review, never an automatic pass/fail.  The canary is the detection
    // gate: a held-out recall variant that crossed the
    // trigger threshold (max_rolling_sum >= MIN_GATE_THRESHOLD) but was
    // NOT detected end-to-end.
    let mut canaries_fired: Vec<&'static str> = Vec::new();
    let gate_crossed_not_detected: Vec<serde_json::Value> = recall_variants
        .iter()
        .filter(|v| {
            !v.detected
                && v.per_frame_scores
                    .iter()
                    .any(|s| s[ROLLING_SUM_IDX] >= MIN_GATE_THRESHOLD)
        })
        .map(|v| serde_json::json!({ "variant": v.variant }))
        .collect();
    if !gate_crossed_not_detected.is_empty() {
        canaries_fired.push("gate_crossed_not_detected");
    }

    let per_variant_json: Vec<serde_json::Value> = recall_variants
        .iter()
        .map(|v| {
            serde_json::json!({
                "variant": v.variant,
                "detected": v.detected,
                "latency_ms": v.latency_ms,
                "window_scores": v.per_frame_scores,
                "miss_verdict": v.miss_verdict.map(MissVerdict::as_str),
            })
        })
        .collect();

    let phase_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  Enrolled-speaker phase (held-out recall): {detected_live}/{total} detected end-to-end",
    );
    if !canaries_fired.is_empty() {
        warn!(
            "Enrolled-speaker phase near-miss canaries fired: {} — investigation \
             triggers, not a hard gate",
            canaries_fired.join(", "),
        );
    }

    serde_json::json!({
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
                "held_out_recall_v3 — {} unseen enrolled-voice wake-only clips, seeds \
                 3000+; embedded-in-sentence detection is not measured; the in-sample F1 \
                 control is removed (all enrollment clips are training data)",
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
                     acceptance metric).",
        },
        "paths": {
            "window_encoder": window_encoder_path,
            "primary_mechanism": primary_mechanism,
            "note": "The encoder pipeline has a SINGLE scoring path (the \
                     stride-gated trailing-window encode + rolling-sum gate); \
                     every end-to-end detection goes through it.  Acceptance \
                     compares mean held-out recall against MIN_DETECTION_RATE \
                     across >= 3 fresh runs via this single path.",
        },
        "latency": {
            "detected_path_ms": detected_latencies,
            "detected_path_mean_ms": mean(&detected_latencies),
            "note": "latency_ms is the CPU wall-clock of the feed loop (the benchmark \
                     feeds audio faster than real-time) — a processing-cost proxy, NOT \
                     the 'from speech onset' audio latency.",
        },
        "near_miss_canaries": {
            "gate_crossed_not_detected": gate_crossed_not_detected,
            "fired": canaries_fired,
            "note": "Investigation triggers, NOT hard gates: any \
                     fired canary means the run deserves review before acceptance.  \
                     'gate_crossed_not_detected' = a held-out recall variant \
                     crossed the trigger threshold (max_rolling_sum >= 1.65) but was not \
                     detected end-to-end.",
        },
        "per_variant": per_variant_json,
    })
}

/// Compute the safety gate over the negative/confusable set.
///
/// The deferral adds in-distribution deferred windows (burst + boundary
/// double-scoring) whose false-positive impact is measured here.  Gate
/// crossings (score gate hit without an end-to-end detection) and true false
/// accepts (end-to-end detections) are counted separately.
fn safety_gate(all_neg_pv: &[(&PerVariantResult, String)]) -> serde_json::Value {
    let total = all_neg_pv.len();
    let gate_crossings = all_neg_pv
        .iter()
        .filter(|(pv, _)| pv.max_rolling_sum >= MIN_GATE_THRESHOLD)
        .count();
    let false_accepts: Vec<(&str, String)> = all_neg_pv
        .iter()
        .filter(|(pv, _)| pv.detected)
        .map(|(pv, cat)| (pv.variant.as_str(), cat.clone()))
        .collect();
    // Near-miss canaries (investigation triggers, not hard gates).
    let mut neg_canaries_fired: Vec<&'static str> = Vec::new();
    if gate_crossings > 0 {
        neg_canaries_fired.push("gate_crossing_on_negative_corpus");
    }
    serde_json::json!({
        "note": "False-positive impact of the deferred in-distribution windows \
                 (deferred burst + boundary fallback double-scoring) on the \
                 negative/confusable set: end-to-end false accepts and score-gate \
                 crossings (max_rolling_sum >= MIN_GATE_THRESHOLD without an \
                 end-to-end detection) are counted per utterance — the acceptance \
                 denominator is utterances, not frames.",
        "total_negatives": total,
        "gate_crossings": gate_crossings,
        "end_to_end_false_accepts": false_accepts.len(),
        "false_accept_list": false_accepts,
        "acceptance": format!(
            "<= {allowance} false accepts (5% of the {total}-negative corpus) across \
             >= 3 fresh runs.  Gate crossings without an end-to-end detection do NOT \
             count as false accepts.",
            allowance = (total as f64 * 0.05).ceil() as usize,
            total = total,
        ),
        "near_miss_canaries": {
            "gate_crossing_on_negative_corpus": gate_crossings,
            "fired": neg_canaries_fired,
            "note": "Investigation triggers, NOT hard gates.  \
                     'gate_crossing_on_negative_corpus' = any negative \
                     variant crossed the score gate (counted per run).  \
                     Positive-side canaries are in the \
                     enrolled_speaker section.",
        },
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// The main integration test / benchmark entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Bench-side replication of production's enrollment self-test counting
/// (`voice.rs` `run_enrollment_self_test`): how many of the utterance
/// embeddings would trigger detection through the same fresh-per-utterance
/// `score_single_embedding` loop production runs at the end of enrollment.
///
/// Production keeps the counts only inside its error string, so the bench
/// replicates the loop to make the deployability verdict's recall numbers
/// structurally available without touching production's self-test.  A
/// cross-check in `run_internal` warns if the replicated pass/fail ever
/// disagrees with production's `run_enrollment_self_test` result.
fn enrollment_self_test_counts(
    utterance_embeddings: &[Vec<f32>],
    enrollment: &crate::audio::wake_word::WakeWordEnrollment,
) -> (usize, usize) {
    let mut passed = 0usize;
    for embedding in utterance_embeddings {
        // Fresh simulation for each utterance: no cross-utterance state.
        //
        // Mirror production's run_enrollment_self_test: feed the utterance
        // embedding ROLLING_WINDOW_N (3) times so the rolling-sum machinery
        // sees the same phrase across ~3 consecutive stride-gated window
        // encodings.  A single embedding can only produce a rolling sum of
        // one soft score (max 1.0) — never match_threshold() = 1.65.
        let mut score_window = Vec::new();
        let mut detected_this = false;
        for _ in 0..super::ROLLING_WINDOW_N {
            let (detected, _, _, _) = super::score_single_embedding(
                embedding,
                Some(enrollment),
                &mut score_window,
                None, // no adaptive threshold during enrollment self-test
                super::ADAPTIVE_K_DEFAULT,
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
    (passed, utterance_embeddings.len())
}

/// Deployability verdict (report-only, additive key): whether the enrollment
/// would pass production's enrollment self-test gate (≥80% of utterances
/// trigger detection — `ENROLLMENT_QUALITY_SELF_TEST_MIN_FRACTION`).
/// The bench NEVER hard-gates on this — it is a prominent informational flag
/// so an enrollment production would reject is never misread as deployable.
///
/// Returns `(json, (passed, total, required, would_pass))` — the caller's
/// cross-check reuses the counts and the gate verdict instead of running the
/// full scoring loop or re-deriving the gate formula a second time.
fn deployability_json(
    self_test_result: &Result<(), String>,
    utterance_embeddings: &[Vec<f32>],
    enrollment: &crate::audio::wake_word::WakeWordEnrollment,
) -> (serde_json::Value, (usize, usize, usize, bool)) {
    let (passed, total) = enrollment_self_test_counts(utterance_embeddings, enrollment);
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
                "enrollment production would accept — informational only"
            } else {
                "enrollment production would reject — informational only"
            },
            "production_self_test_passed": self_test_result.is_ok(),
            "warning": if would_pass {
                serde_json::Value::Null
            } else {
                serde_json::Value::String(format!(
                    "PRODUCTION WOULD REJECT THIS ENROLLMENT: only {passed}/{total} \
                     utterances triggered detection (gate >= {required}, {:.0}%).  The bench \
                     is report-only — this does NOT fail the run — but the enrollment is \
                     not deployable as-is (informational only).",
                    super::ENROLLMENT_QUALITY_SELF_TEST_MIN_FRACTION * 100.0,
                ))
            },
            "note": "Mirrors production's run_enrollment_self_test loop (fresh \
                     score_single_embedding per utterance embedding); the recall \
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
/// ([`ACCEPTANCE_BASIS_PREFIX`]) so v3 enlarged-basis runs never mix with v2
/// 16-clip archives in the ≥3-run spread.
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
        "note": "Last 3 pre-existing timestamped archives, filtered to the current \
                 acceptance basis (ACCEPTANCE_BASIS_PREFIX).  Until 3 runs with the \
                 current basis accumulate, the spread is partial.",
    })
}

/// Env-gated FAPH phase: real-audio false-acceptance-per-hour bench.
///
/// Feeds the pinned corpus (`faph_corpus_manifest.json` — alexwengg/musan_mini,
/// 812 files, ~5.99 h audio, SHA-256-pinned per file) through the production
/// streaming detection path as **ambient audio** with ONE continuous
/// [`super::PipelineCtx`] across the whole corpus (all samples fed in
/// [`FRAME_LENGTH`](super::FRAME_LENGTH) chunks via [`process_frame`], no
/// early exit) and counts false-accept events.
///
/// # FA-counting semantics
///
/// - **Event-based**: each fresh detection event (a new
///   `last_wake_word_detection` timestamp) counts as 1 FA.
/// - **Continuous listening**: one pipeline context across the whole corpus;
///   a 2 s silence gap between files fires the natural segment-boundary reset
///   (fresh detector, adaptive bootstrap, cleared ring) the way production listens.
/// - **Raw vs cooldown-merged**: production's wall-clock `WAKE_WORD_COOLDOWN`
///   (3 s) would suppress more audio than 3 s per event at the bench's
///   faster-than-real-time feed — the bench
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
    // The enrollment must be set as a global (handle_wake_word_detection reads
    // it from voice_state()).  The bench's earlier phases set it; re-set from
    // the local enrollment to be self-contained.
    let Some(enrollment) = super::get_enrollment() else {
        return faph_skip_json(
            "enrollment_unavailable",
            "no wake-word enrollment in global state",
        );
    };
    super::set_enrollment(enrollment);

    let mut total_audio_secs = 0.0f64;
    let mut total_vad_active_secs = 0.0f64;
    let mut raw_events: Vec<f64> = Vec::new();
    let mut files_fed = 0u64;
    let mut files_decode_failed = 0u64;
    let mut per_file_events: Vec<serde_json::Value> = Vec::new();
    let mut last_progress_print = std::time::Instant::now();

    // Continuous listening: ONE PipelineCtx across
    // the whole corpus so the adaptive threshold, and
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
        // Progress line: the phase feeds the whole corpus with no per-file
        // summary in the report (the aggregate is the metric); this periodic
        // line keeps the multi-hour phase observable while it runs.  Corpus
        // files vary widely in length (long speech recordings to short noise
        // clips), so the cadence is time-based, not file-count-based.
        if last_progress_print.elapsed() >= Duration::from_mins(1) {
            let wall_secs = phase_start.elapsed().as_secs_f64();
            eprintln!(
                "  FAPH progress: {files_fed}/{} files, {:.2} h audio, {:.0} s wall \
                 ({:.2}x realtime)",
                files.len(),
                total_audio_secs / 3600.0,
                wall_secs,
                if wall_secs > 0.0 {
                    total_audio_secs / wall_secs
                } else {
                    f64::NAN
                },
            );
            last_progress_print = std::time::Instant::now();
        }
        // Feed the inter-file silence gap so the segment-boundary reset fires
        // naturally (the next file starts with production's post-silence
        // state: fresh detector, adaptive threshold bootstrap, cleared ring).
        // Gap events (rare — silence) are part of continuous listening and
        // join the global raw stream.
        raw_events.extend(faph_feed_file_continuous(&gap, &mut ctx, &mut audio_pos));
        // VAD-active seconds for the file+2s-gap window from the CONTINUOUS
        // feed: production's global VAD detector (evolved across the whole
        // corpus, exactly how the mic path listens) — the basis for the
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
    // VAD-active denominator — the speech-exposure basis.
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
        "FAPH: {files_fed} files fed, {audio_hours:.2} h audio \
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
                          reset (fresh detector, adaptive bootstrap, cleared ring) the way production \
                          listens; raw events observed by re-arming after each event (production \
                          post-command reset) with the wall-clock cooldown timestamp cleared \
                          bench-side (production's WAKE_WORD_COOLDOWN gate unmodified); \
                          cooldown is applied as a 3 s AUDIO-position merge for the count; \
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
            "primary_denominator": "VAD-active (speech-active) hours — the \
                                     speech-exposure basis; raw-hours readings are secondary",
            "cooldown_suppression": "wall-clock WAKE_WORD_COOLDOWN (3 s) would suppress \
                                     more audio than 3 s per event at the bench's \
                                     faster-than-real-time feed — bypassed \
                                     bench-side (raw stream) and re-applied as a 3 s \
                                     audio-position merge (production-equivalent count)",
            "continuous_listening": true,
        },
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
    warn!("FAPH skipped: {reason_key} — {detail}");
    eprintln!("         FAPH skipped: {reason_key} — {detail}");
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
/// download precedent (gated embedder/providers out of the bench
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
        crate::util::http::install_ring_provider();
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
/// audio through a SHARED continuous-listening pipeline context (one pipeline
/// context across the whole corpus, matching production's continuous
/// listening).
///
/// Feeds every sample in [`FRAME_LENGTH`](super::FRAME_LENGTH) chunks through
/// [`handle_wake_word_detection`] directly, recording the
/// AUDIO position (seconds) of each fresh `last_wake_word_detection`
/// timestamp.  Production's `WAKE_WORD_COOLDOWN` gate inside
/// `handle_wake_word_detection` is wall-clock based: at the bench's
/// faster-than-real-time feed speed it would suppress far more audio than the
/// 3 s it represents, structurally
/// under-counting.  The bench therefore observes the RAW event stream by
/// re-arming after each event exactly like production's post-command reset
/// (`reset_pipeline_state(Soft)` + `is_recording = false`) AND clearing the
/// cooldown timestamp (bench-side emulation — production's gate is NOT
/// modified).  The caller applies production's 3 s cooldown as an
/// audio-position merge ([`faph_merge_events`]) for the
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
/// (scores / trajectory, one entry per voiced frame) is never read
/// here, and leaving it would grow unboundedly across the whole corpus
/// (~12 h of audio).  Only the
/// diagnostic vectors are cleared; the acoustic state the continuous feed
/// depends on (audio ring, adaptive threshold, VAD detector) is preserved.
///
/// # Future-field hazard
/// A per-frame field added to `DetectionInstrumentation` without a matching
/// clear here would silently accumulate unbounded memory across the corpus —
/// keep this clear list exhaustive when the instrumentation struct grows.
fn faph_clear_instrumentation(ctx: &mut super::PipelineCtx) {
    ctx.instrumentation.per_frame_scores.clear();
    ctx.instrumentation.adaptive_threshold_trajectory.clear();
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
    const P_FINALIZE_ENROLLMENT: usize = 3;
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

    // ── Heartbeat drop guard ──────────────────────────────
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
            // not initialized or buffers messages.
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

    // ── Heartbeat thread ──────────────────────────────────
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
    // vad_segment_and_enroll (VAD → encode → prototype).  Detection
    // control re-bases onto the 40 held-out wake-only recall clips (Phase 7d,
    // the enlarged set),
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

    let utterance_embeddings = vad_segment_and_enroll(&train_clips);
    if utterance_embeddings.is_empty() {
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
        "VAD-gated enrollment: {} utterance embeddings from {} clips",
        utterance_embeddings.len(),
        train_clips.len(),
    );

    phase_times[P_VAD_ENROLLMENT] = phase_end_ms!();

    // ── Phase 3: Generate negative training data ─────────────────────────
    phase_start!("Phase 3: Generating negative training data");

    // Bench-local restricted-style confusable/unrelated generation.
    // Production's old prewarm rotated ALL 10 TTS voices into the training
    // negatives, which would train the reserved cross-speaker voices
    // (M4/M5) in — the bench requires them absent from EVERY training path.
    // The bench encodes the negative-pool styles (F1..M1) through the shared
    // Qwen3-ASR encoder; the PCM cache keeps TTS synthesis cheap across runs.
    let confusable_neg_embeddings = generate_restricted_phrase_negatives(
        "confusable",
        CONFUSABLE_PHRASES,
        CONFUSABLE_SEEDS_PER_PHRASE,
        CONFUSABLE_SEED_BASE,
        &voice_allocation.negative_pool,
        &model_version_hash,
        &cache_dir_path,
    );
    info!(
        "Bench-local confusable negatives: {} embeddings from {} phrases × {} seeds over \
         negative-pool styles {:?} (reserved voices excluded)",
        confusable_neg_embeddings.len(),
        CONFUSABLE_PHRASES.len(),
        CONFUSABLE_SEEDS_PER_PHRASE,
        voice_allocation.negative_pool,
    );
    assert!(
        !confusable_neg_embeddings.is_empty(),
        "Bench-local confusable negatives must be non-empty — an all-skip run would \
         silently calibrate a weaker enrollment"
    );
    let unrelated_neg_embeddings = generate_restricted_phrase_negatives(
        "unrelated",
        UNRELATED_PHRASES,
        UNRELATED_SEEDS_PER_PHRASE,
        UNRELATED_SEED_BASE,
        &voice_allocation.negative_pool,
        &model_version_hash,
        &cache_dir_path,
    );
    info!(
        "Bench-local unrelated negatives: {} embeddings from {} phrases × {} seeds over \
         negative-pool styles {:?} (reserved voices excluded)",
        unrelated_neg_embeddings.len(),
        UNRELATED_PHRASES.len(),
        UNRELATED_SEEDS_PER_PHRASE,
        voice_allocation.negative_pool,
    );
    assert!(
        !unrelated_neg_embeddings.is_empty(),
        "Bench-local unrelated negatives must be non-empty — an all-skip run would \
         silently calibrate a weaker enrollment"
    );

    // Ambient noise embeddings — replaces synthetic negatives.
    let ambient_neg_embeddings = generate_ambient_noise_sequences();
    info!(
        "Generated {} ambient noise embeddings from {} noise profiles × 2 SNR levels",
        ambient_neg_embeddings.len(),
        NOISE_PROFILES.len(),
    );

    // Owner-negative embeddings (enrolled voice only).
    let owner_neg_embeddings = generate_owner_negative_sequences(
        &voice_allocation.enrolled,
        &model_version_hash,
        &cache_dir_path,
    );
    info!(
        "Generated {} owner-negative embeddings from the ENROLLED voice {}",
        owner_neg_embeddings.len(),
        voice_allocation.enrolled,
    );

    // ── Combine the negative pool: ambient → owner → confusable → unrelated ──
    // All 1024-dim L2-normalized window embeddings for negative calibration.
    let mut negative_embeddings: Vec<Vec<f32>> = Vec::new();
    negative_embeddings.extend(ambient_neg_embeddings);
    negative_embeddings.extend(owner_neg_embeddings);
    negative_embeddings.extend(confusable_neg_embeddings);
    negative_embeddings.extend(unrelated_neg_embeddings);
    assert!(
        !negative_embeddings.is_empty(),
        "Phase 3 must produce negative embeddings (ambient + owner + confusable + unrelated)"
    );
    info!(
        "Phase 3 negative pool: {} total 1024-dim negative embeddings for calibration",
        negative_embeddings.len(),
    );

    phase_times[P_NEG_TRAINING_DATA] = phase_end_ms!();

    // ── Phase 4: finalize_enrollment (consistency check + prototype + calibration) ──
    phase_start!("Phase 4: finalize_enrollment");
    // `utterance_embeddings` from Phase 2 (one 1024-dim embedding per VAD
    // utterance) is the enrollment basis for the prototype.
    // NO hard gate: production's consistency check is reported via the
    // deployability verdict, never enforced here.  On a consistency failure
    // there is no enrollment to evaluate — the bench records the gate failure
    // and marks the run degenerate (report-only), mirroring the old
    // double-failure path.
    let mut finalize_gate_failed: Option<String> = None;
    let prototype = match super::enrollment_consistency_check(&utterance_embeddings) {
        Ok(proto) => Some(proto),
        Err(err) => {
            warn!(
                "enrollment_consistency_check FAILED (report-only): \
                 {err} — no enrollment to evaluate, skipping all detection phases"
            );
            finalize_gate_failed = Some(err.clone());
            None
        }
    };

    // ── Build the enrollment (prototype + negative calibration) ──
    let enrollment: Option<crate::audio::wake_word::WakeWordEnrollment> = match &prototype {
        Some(proto) => {
            let calibration =
                crate::audio::wake_word::calibrate_negatives(proto, &negative_embeddings);
            let created_at = crate::turso::now();
            let trained_at = crate::turso::now();
            let phrase = super::normalize_phrase(wake_word());
            crate::audio::wake_word::WakeWordEnrollment::build(
                phrase,
                &utterance_embeddings,
                calibration,
                &negative_embeddings,
                created_at,
                trained_at,
            )
        }
        None => None,
    };

    // ── Degenerate solution detection ──
    // A missing enrollment (consistency gate failed or too few utterances) is
    // degenerate by definition — there is nothing to evaluate, so the
    // degenerate flag below skips every detection phase (report-only).
    let degenerate = match &enrollment {
        None => {
            warn!(
                "No enrollment (consistency gate failed or too few utterances) — \
                 skipping all detection phases"
            );
            true
        }
        Some(enr) => {
            info!(
                "Enrollment built: phrase='{}', {} utterances, prototype dim {}, \
                 calibration neg_mean={:.4} (p99={:.4}, n={})",
                enr.phrase,
                enr.utterance_count,
                enr.prototype.len(),
                enr.calibration.neg_mean,
                enr.calibration.neg_p99,
                enr.calibration.n_negatives,
            );
            false
        }
    };

    // ── Soft-score discrimination evidence (prototype pipeline) ──
    // With no trainable head, the "soft scores" are the enrollment soft
    // scores (cosine through calibration): positives = utterance embeddings,
    // negatives = the negative pool.  The prototype pipeline reports its own
    // calibration diagnostics instead.
    let (
        pos_scores_mean,
        pos_scores_min,
        pos_scores_max,
        neg_scores_mean,
        neg_scores_min,
        neg_scores_max,
        pos_scores_deciles,
        neg_scores_deciles,
        enrollment_diagnostics,
    ) = if let Some(enr) = &enrollment {
        let pos: Vec<f32> = utterance_embeddings
            .iter()
            .map(|e| enr.soft_score(e))
            .collect();
        let neg: Vec<f32> = negative_embeddings
            .iter()
            .map(|e| enr.soft_score(e))
            .collect();
        let pos_stats = score_stats(&pos);
        let neg_stats = score_stats(&neg);
        (
            pos_stats.0,
            pos_stats.1,
            pos_stats.2,
            neg_stats.0,
            neg_stats.1,
            neg_stats.2,
            pos_stats.3,
            neg_stats.3,
            serde_json::json!({
                "utterance_count": enr.utterance_count,
                "n_negative_embeddings": negative_embeddings.len(),
                "calibration_floor": enr.calibration.soft_floor(),
                "neg_cosine_p99": enr.calibration.neg_p99,
                "neg_cosine_mean": enr.calibration.neg_mean,
                "n_anti_prototypes": enr.negative_prototypes.len(),
                "prototype_dim": enr.prototype.len(),
            }),
        )
    } else {
        (
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            None,
            None,
            serde_json::Value::Null,
        )
    };

    // ── Informational self-test ──
    // Production treats the self-test as GATING: if it fails, the enrollment is
    // rejected and enrollment fails.  The benchmark is report-only,
    // so it does not abort — but a failure is surfaced as a prominent warning
    // AND recorded in the JSON so consumers know the reported detection/FA
    // numbers come from an enrollment production would refuse to deploy.
    // Gated on an enrollment: on the degenerate path there is no prototype to
    // evaluate, so running the self-test on nothing would manufacture a
    // deployability verdict from noise — skip it instead (the `degenerate`
    // flag already skips every detection phase below).
    let self_test_result: Option<Result<(), String>> = enrollment
        .as_ref()
        .map(|enr| super::run_enrollment_self_test(&utterance_embeddings, enr));
    match &self_test_result {
        Some(Ok(())) => info!("Enrollment self-test: passed"),
        Some(Err(e)) => warn!(
            "Enrollment self-test FAILED — production would reject this enrollment (report-only): {e}"
        ),
        None => warn!(
            "No enrollment (consistency gate failed or too few utterances) — \
             skipping the informational self-test / deployability verdict (report-only)"
        ),
    }
    // Deployability verdict (Phase 0, additive default-run key): structured
    // recall numbers + the would-production-accept flag.  The bench-side count
    // replication is cross-checked against production's Result so a divergence
    // in the self-test logic is never silently masked.
    let (deployability, (passed, total, required, deploy_would_pass)) = match &self_test_result {
        Some(r) => {
            let enr = enrollment
                .as_ref()
                .expect("self-test requires an enrollment");
            let (json, (p, t, req, would_pass)) = deployability_json(r, &utterance_embeddings, enr);
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
                "note": "No enrollment (consistency gate failed or too few \
                         utterances) — deployability not measured",
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
    phase_times[P_FINALIZE_ENROLLMENT] = phase_end_ms!();

    // ── Phase 5: Set global state for streaming detection ────────────────
    phase_start!("Phase 5: Setting global state");
    if let Some(enr) = &enrollment {
        super::set_enrollment(enr.clone());
        info!(
            "Wake word enrollment installed in global state (phrase='{}', {} utterances)",
            enr.phrase, enr.utterance_count,
        );
    } else {
        warn!("No enrollment to install — degenerate run");
    }
    phase_times[P_GLOBAL_STATE] = phase_end_ms!();

    // ── Phase 6: Streaming detection setup ────────────────────────────────
    phase_start!("Phase 6: Streaming detection setup");
    // This phase is mostly a no-op — the setup was already done in Phase 5.
    // The timing will be near-zero.
    phase_times[P_STREAMING_SETUP] = phase_end_ms!();

    // ── Detection phases (skipped entirely if the enrollment is degenerate) ─
    //
    // Initialize output vars to empty defaults — filled in below only when
    // `!degenerate`.  The `phase_times` entries for skipped phases are set
    // to 0 directly, while `phase_start!`/`phase_end_ms!` are only called
    // for executed phases.
    let mut pos_metrics = DetectionMetrics::default();
    // Cold-start detection metrics — populated in Phase 8
    // alongside the warm pass.  Stays default when degenerate.
    let mut cold_metrics = DetectionMetrics::default();
    let mut conf_metrics = DetectionMetrics::default();
    let mut unrelated_metrics = DetectionMetrics::default();
    let mut silence_metric = DetectionMetrics::default();
    let mut noise_false_accepts: Vec<String> = Vec::new();
    let mut noise_overlap_results: Vec<(String, f64, usize, Vec<serde_json::Value>)> = Vec::new();
    // held-out cross-speaker probe matrix (reserved voices) —
    // reported separately from the trained-in canary matrix.
    let mut probe_overlap_results: Vec<(String, f64, usize, Vec<serde_json::Value>)> = Vec::new();
    let mut probe_fa_count: usize = 0;
    // Per-tier confusable fa tracking. Populated after Phase 9.
    let mut conf_fa_by_tier: [Vec<String>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    // Cooldown-phase detection time — hoisted from the detection block so the
    // JSON report can emit it even when the enrollment is degenerate.
    let mut cooldown_detection_time_ms = 0.0f64;
    // Cooldown test outcomes (None = test skipped).  Emitted in the JSON
    // report so cooldown behaviour is observable.
    let mut cooldown_first_detected: Option<bool> = None;
    let mut cooldown_suppressed: Option<bool> = None;
    let mut cooldown_after_recovered: Option<bool> = None;
    // Additive cooldown.* report keys.  The phase was re-pointed
    // at the enrolled-speaker original variant (the M2–M5 held-out clips
    // are rejected under single-speaker semantics, so the old precondition
    // could never hold).  All keys stay `None` on the degenerate-enrollment
    // path and on a visible skip (first detection failed) —
    // the report emits them as `null` with `skip_reason` set, never panics.
    let mut cooldown_source_variant: Option<String> = None;
    let mut cooldown_skip_reason: Option<String> = None;
    let mut cooldown_suppressed_at_2_5s: Option<bool> = None;
    let mut cooldown_accumulation_cap_observed: Option<usize> = None;
    let mut cooldown_buffered_audio_processed_after_expiry: Option<bool> = None;

    // Per-variant metrics for noise profiles (collected across all profiles).
    let mut noise_metrics: Vec<DetectionMetrics> = Vec::new();

    // Enrolled-speaker benchmark phase (Phase 7d).  Declared
    // here so the JSON report can emit a note even when the enrollment is
    // degenerate.
    let mut enrolled_report: Option<serde_json::Value> = None;

    // Expanded TEST-surface report sections (report-only; filled inside the
    // `!degenerate` block, emitted as `null`-free keys only when present).
    let mut held_out_snr_envelope: Option<serde_json::Value> = None;
    let mut overlapping_speech: Option<serde_json::Value> = None;
    let mut owner_negative_detection: Option<serde_json::Value> = None;

    if !degenerate {
        // Create a pre-warmed adaptive threshold state shared across all
        // detection phases so the adaptive code path is exercised end-to-end.
        // Without this, test_detection_samples
        // creates a fresh PipelineCtx per variant, keeping the adaptive state
        // in perpetual 5-frame bootstrap and measuring all metrics against the
        // static threshold.
        //
        // Note: only background frames
        // (below NO_MATCH_RESET_THRESHOLD) update the running statistics.
        // Positive variants therefore do not inflate the adaptive
        // threshold.  The noise-overlap phase (14) uses a
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
            20, // seeds per reserved voice (scale-up)
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
        // All 10 enrollment clips are training data, so detection control is
        // the held-out wake-only recall set (Phase 7d, raw clips — the
        // acceptance basis).  This phase
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

        // ── Cold-start pass ─────────────────────────────
        // Fresh PipelineCtx per variant, no warm-up, fresh adaptive bootstrap
        // (production's post-silence start).  The raw-clip cold measurement
        // lives in the enrolled-speaker phase (Phase 7d) — this pass covers
        // the augmented diagnostics.
        test_detection_samples(
            &aug_diag_variants,
            &mut cold_metrics,
            |m, _| m.detected += 1,
            None, // fresh adaptive state per variant
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
        // gated, but the acceptance protocol judges the mean
        // across ≥ 3 runs.
        eprintln!("─── Phase 7d: enrolled-speaker benchmark (held-out recall) ───");
        info!("─── Phase 7d: enrolled-speaker benchmark (held-out recall) ───");
        let enrolled_start = Instant::now();
        enrolled_report = Some(run_enrolled_speaker_phase(&held_out_recall_clips));
        let enrolled_ms = enrolled_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  Enrolled-speaker phase completed in {enrolled_ms:.0}ms");
        // ── Phase 8: Detection — Confusable phrases ──────────────────────
        // The detection pool spans two seed bands: band 1 (800+) with the
        // "confusable" prefix, band 2 (810+) with the distinct "confusable2"
        // prefix (the band must not re-key existing 800+ clips).  Both bands
        // feed the same confusable false-accept tally and per-tier split.
        phase_start!("Phase 8: Negative — confusable phrases");
        let mut confusable_variants = generate_phrase_variants_cached(
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
        confusable_variants.extend(generate_phrase_variants_cached(
            CONFUSABLE_PHRASES,
            &available_styles,
            SeedConfig {
                base_seed: 810,
                num_variants: 1,
                seed_variant: 0,
            },
            "confusable2",
            &model_version_hash,
            &cache_dir_path,
        ));
        info!(
            "Generated {} confusable phrase variants (2 seed bands: 800+, 810+)",
            confusable_variants.len()
        );
        test_detection_samples(
            &confusable_variants,
            &mut conf_metrics,
            |m, l| m.false_accepts.push(l.to_string()),
            Some(&mut shared_adaptive),
            false, // warm pass only (negative phase)
        );
        // Split false accepts by tier for per-tier tracking.
        for label in &conf_metrics.false_accepts {
            let phrase = confusable_phrase(label);
            let tier_idx = tier_for_phrase(phrase).index();
            conf_fa_by_tier[tier_idx].push(label.clone());
        }
        phase_times[P_CONFUSABLE_NEGATIVES] = phase_end_ms!();

        // ── Phase 10: Detection — Unrelated phrases ───────────────────────
        // The unrelated detection pool spans two seed bands: band 1 (900+)
        // with the "unrelated" prefix, band 2 (910+) with the distinct
        // "unrelated2" prefix.
        phase_start!("Phase 9: Negative — unrelated phrases");
        let mut unrelated_variants = generate_phrase_variants_cached(
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
        unrelated_variants.extend(generate_phrase_variants_cached(
            UNRELATED_PHRASES,
            &available_styles,
            SeedConfig {
                base_seed: 910,
                num_variants: 1,
                seed_variant: 0,
            },
            "unrelated2",
            &model_version_hash,
            &cache_dir_path,
        ));
        info!(
            "Generated {} unrelated phrase variants (2 seed bands: 900+, 910+)",
            unrelated_variants.len()
        );
        test_detection_samples(
            &unrelated_variants,
            &mut unrelated_metrics,
            |m, l| m.false_accepts.push(l.to_string()),
            Some(&mut shared_adaptive),
            false, // warm pass only (negative phase)
        );
        phase_times[P_UNRELATED_NEGATIVES] = phase_end_ms!();

        // ── Phase 11: Detection — Silence ────────────────────────────────
        // Silence becomes a small matrix (three durations).
        phase_start!("Phase 10: Negative — silence");
        let silence_variants: Vec<(Vec<f32>, String)> = SILENCE_DURATIONS
            .iter()
            .map(|(label, len)| (vec![0.0f32; *len], label.to_string()))
            .collect();
        test_detection_samples(
            &silence_variants,
            &mut silence_metric,
            |m, l| m.false_accepts.push(l.to_string()),
            Some(&mut shared_adaptive),
            false, // warm pass only (negative phase)
        );
        phase_times[P_SILENCE_NEGATIVES] = phase_end_ms!();

        // ── Phase 12: Detection — Noise profiles ─────────────────────────
        // The 4 detection-only profiles join the phase but NEVER the training
        // list (NOISE_PROFILES — see NOISE_PROFILES_DETECTION_ONLY).
        phase_start!("Phase 11: Negative — noise profiles");
        for (label, generator) in all_noise_profiles() {
            info!("  Testing noise profile: {label}");
            let noise = generator();
            let mut metric = DetectionMetrics::default();
            test_detection_samples(
                &[(noise, (*label).to_string())],
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

        // ── Wake-over-babble ─────────────────────────────────────────────
        // Held-out wake clips mixed with a multi-voice TTS babble track at a
        // few SNR levels — speech-on-speech interference (the encoder
        // pipeline's behaviour under a speech interferer).  In-memory mixing; only
        // the babble clips enter the PCM cache.  Report-only diagnostic (the
        // wake word IS present).
        eprintln!("─── Wake-over-babble (report-only) ───");
        if let Some(babble_track) =
            generate_babble_overlay_cached(&available_styles, &model_version_hash, &cache_dir_path)
        {
            let babble_cells: Vec<(String, CellMix)> = BABBLE_SNRS
                .iter()
                .map(|(snr_label, snr_db)| {
                    (
                        format!("babble_{snr_label}"),
                        CellMix::Babble {
                            track: babble_track.clone(),
                            snr_db: *snr_db,
                        },
                    )
                })
                .collect();
            let babble_start = Instant::now();
            let babble_results = run_detection_matrix(
                &held_out_recall_clips,
                &babble_cells,
                "overlapping_speech",
                "Wake-over-babble",
            );
            let babble_ms = babble_start.elapsed().as_secs_f64() * 1000.0;
            let babble_total: usize = babble_results.iter().map(|(_, _, _, d)| d.len()).sum();
            let babble_detected: usize = babble_results
                .iter()
                .map(|(_, _, detected, _)| *detected)
                .sum();
            eprintln!(
                "  Wake-over-babble: {babble_detected}/{babble_total} detected ({babble_ms:.0}ms)"
            );
            overlapping_speech = Some(serde_json::json!({
                "cells": serde_json::Value::Object(
                    babble_results.iter().map(&cell_json).collect(),
                ),
                "n_cells": babble_results.len(),
                "total_variants": babble_total,
                "detected": babble_detected,
                "phase_ms": babble_ms,
                "note": "Held-out wake clips over a TTS-rendered babble track \
                         (non-wake phrases, seeds 9500+): each clip mixes against a \
                         ~1 s slice of the track, with per-clip offsets sampling the \
                         multi-voice track across the set, at 10/5/0 dB.  SNR is \
                         computed on the mixed slice.  The wake word IS present \
                         (speaker-blindness diagnostic, report-only).  In-memory \
                         mixing; the babble track is the only new cache content.",
            }));
        } else {
            warn!("Babble overlay synthesis produced no audio — wake-over-babble section skipped");
        }

        // ── Phase 12: Cooldown verification ──────────────────────────────
        // Re-pointed at the ENROLLED-SPEAKER original variant
        // (enrolled-voice `_enroll0` lookup + pcm_augment_enrollment_variants,
        // _original pinned — 85/85 reliable; speed_down/noise fail ~4–6%
        // historically).  The enrolled clip is training data, so
        // its `_original` fires reliably — the in-sample precondition the
        // cooldown semantics need.
        //
        // Cooldown semantics under test (production, unchanged):
        //   - WAKE_WORD_COOLDOWN (3.0 s) gate in handle_wake_word_detection;
        //   - audio fed during cooldown is BUFFERED (not discarded) into
        //     ctx.audio_buffer up to AUDIO_BUFFER_MAX (~2.3 s) and processed
        //     when the gate expires;
        //   - the scoring loop is gated on `!is_recording`, which
        //     stays true after a detection fires (Soft reset preserves it) —
        //     the bench must reset `is_recording` between detections or the
        //     recovery probe cannot fire (analyst review).
        //
        // Probe schedule (all sleeps EXCLUDED from phase timing):
        //   Detection 1  t≈0      warm pass — must fire
        //   Detection 2  t≈0      gate closed — suppressed + cap probe
        //   Detection 3  ~2.5 s   gate still closed (2.5 < 3.0) — suppressed
        //   Detection 4  ~3.5 s   gate expired — fires (natural expiry, NO
        //                          manual last_wake_word_detection clear)
        //
        // Gate vs report: Detections 3 and 4
        // ARE the gate — HARD assertions, so a production cooldown regression
        // (e.g. halved WAKE_WORD_COOLDOWN) fails the bench instead of only
        // warning.  Slack margins (2.5 s / 3.5 s) are deliberate: wall-clock
        // timing on a loaded bench CPU is jittery, so the assertions never
        // probe 2.99/3.01 s (500 ms from the 3.0 s boundary either way, far
        // beyond any plausible thread::sleep drift).  Detection 2 keeps the
        // pre-existing soft `cooldown_suppressed_redetection` report key.
        info!("─── {}. Cooldown verification ───", P_COOLDOWN + 1);
        // Variant CONSTRUCTION only — the enrolled clip's 12-cell augmented
        // set, WITHOUT re-running the enrolled-speaker phase.
        // `_original` is pinned: the cooldown probes run the unperturbed
        // clip.
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
            // measures only the wake word).
            // NOTE: this is a WARM pass (consume_warmup + warmed shared
            // adaptive state), unlike Phase 7d's cold pass — the acceptance
            // criterion "assertions actually fire" carries residual risk until
            // measured across ≥3 warm runs; a
            // failure here becomes a VISIBLE skip (skip_reason), never silent.
            consume_warmup(&mut ctx);
            let t0 = Instant::now();
            let detected = run_streaming_detection(&enrolled_original, &mut ctx);
            cooldown_detection_time_ms += t0.elapsed().as_secs_f64() * 1000.0;
            if !detected.detected {
                cooldown_first_detected = Some(false);
                let reason = format!(
                    "Enrolled-speaker variant {clip_label} did not fire in the warm pass — \
                     skipping cooldown assertions"
                );
                warn!("{reason}");
                cooldown_skip_reason = Some(reason);
                // Skip remaining cooldown assertions since detection didn't fire.
                // The test will still exercise noise overlap and other phases.
            } else {
                cooldown_first_detected = Some(true);
                info!("Cooldown test: first detection fired ✓ ({clip_label})");
                // Detection set is_recording = true + last_wake_word_detection
                // (handle_wake_word_detection).  The scoring loop is gated on
                // !is_recording, and Soft reset preserves it — reset
                // it between detections or the recovery probe cannot fire.
                ctx.is_recording = false;

                // Detection 2: should NOT fire during cooldown (gate closed,
                // t≈0).  Also the accumulation-cap probe: the full clip feed is
                // buffered during cooldown, so the cooldown path must have
                // buffered the fed audio (bounded by AUDIO_BUFFER_MAX, not
                // discarded).  Asserted DURING cooldown because the expiry
                // handoff consumes/clears the buffer.
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
                assert!(
                    !ctx.audio_buffer.is_empty()
                        && ctx.audio_buffer.len() <= super::AUDIO_BUFFER_MAX,
                    "Cooldown accumulation cap violated: audio_buffer.len() = {} \
                     (cap = {} — cooldown audio is buffered up to the cap, never \
                     discarded)",
                    ctx.audio_buffer.len(),
                    super::AUDIO_BUFFER_MAX,
                );
                info!(
                    "Cooldown test: audio fed during cooldown buffered ({} samples, \
                     cap {} samples) ✓ — not discarded",
                    ctx.audio_buffer.len(),
                    super::AUDIO_BUFFER_MAX,
                );

                // Detection 3: natural-expiry probe at ~2.5 s — the gate must
                // STILL be closed (2.5 s < WAKE_WORD_COOLDOWN 3.0 s).  The old
                // code cleared last_wake_word_detection before this probe,
                // bypassing production's real gate; the natural expiry is the
                // production behavior under test.
                //
                // GATE: HARD assertion, not warn-soft — a production cooldown
                // regression (e.g. halved or removed WAKE_WORD_COOLDOWN) must
                // FAIL the bench.  The 500 ms slack margin (2.5 s vs 3.0 s) is
                // jitter-safe, and the suppressed feed + silence flush complete
                // ~100 ms after the sleep (the bench feeds far faster than
                // real-time), leaving ~400 ms of margin.
                sleep_until_cooldown_elapsed(&ctx, Duration::from_millis(2500));
                let t2 = Instant::now();
                let at_2_5s = run_streaming_detection(&enrolled_original, &mut ctx);
                cooldown_detection_time_ms += t2.elapsed().as_secs_f64() * 1000.0;
                assert!(
                    !at_2_5s.detected,
                    "Cooldown gate violation: re-detection fired at ~2.5s (elapsed {}ms) while \
                     WAKE_WORD_COOLDOWN = {}ms — the gate must stay closed until natural expiry",
                    ctx.last_wake_word_detection
                        .map_or(0, |l| l.elapsed().as_millis()),
                    super::WAKE_WORD_COOLDOWN.as_millis(),
                );
                cooldown_suppressed_at_2_5s = Some(true);
                info!("Cooldown test: re-detection still suppressed at ~2.5s ✓");

                // Detection 4: recovery probe at ~3.5 s — the gate has expired
                // naturally, so detection must fire again (the old fixed 3.1 s
                // sleep is restructured into this probe schedule).
                // is_recording was left true by Detection 1 and the
                // suppressed probes never clear it — reset it so the stride-8
                // burst/main loop runs.
                //
                // GATE: HARD assertion like Detection 3
                // (500 ms slack past the 3.0 s boundary — jitter-safe).
                //
                // Buffered-audio-processed evidence:
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
                     after WAKE_WORD_COOLDOWN = {}ms expired — recovery failed",
                    ctx.last_wake_word_detection
                        .map_or(0, |l| l.elapsed().as_millis()),
                    super::WAKE_WORD_COOLDOWN.as_millis(),
                );
                cooldown_after_recovered = Some(true);
                info!("Cooldown test: detection fired after cooldown ✓");
                cooldown_buffered_audio_processed_after_expiry = Some(
                    pre_recovery_buffer_len > 0
                        && ctx.audio_buffer.is_empty()
                        && !ctx.command_buffer.is_empty(),
                );
            } // close the if detected.detected/else block
        } else {
            let reason = format!(
                "{enrolled_label} missing from train clips — cannot re-point Phase 13 at \
                 the enrolled-speaker positive variant; Phase 8d's fallback \
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

        // ── Owner-negative detection ─────────────────────────────────────
        // Runs AFTER the cooldown assertions so this report-only block cannot
        // perturb the shared adaptive state feeding Phase 12's hard gates.
        // The enrolled voice's non-wake speech is trained in (Phase 3) but was
        // never measured in detection.  This block runs the SAME owner-negative
        // clips (seeds 9000+, cache hits) through the streaming path —
        // report-only.  The clips are trained-in, so any detection is an
        // in-distribution misclassification signal (canary-style), never a
        // generalization claim; the count does NOT feed total_false_accepts.
        eprintln!("─── Owner-negative detection (report-only) ───");
        let owner_neg_start = Instant::now();
        let mut owner_neg_clips: Vec<(Vec<f32>, String)> = Vec::new();
        for (i, &phrase) in OWNER_NEGATIVE_PHRASES.iter().enumerate() {
            for s in 0..3 {
                let seed_val = 9000 + i as u64 * 3 + s as u64;
                if let Some(pcm) = synthesize_wake_word_variant_cached(
                    phrase,
                    &voice_allocation.enrolled,
                    seed_val,
                    TARGET_SAMPLE_RATE,
                    &model_version_hash,
                    &cache_dir_path,
                ) {
                    owner_neg_clips.push((pcm, format!("owner_negative_{phrase}_s{seed_val}")));
                }
            }
        }
        let mut owner_neg_metrics = DetectionMetrics::default();
        test_detection_samples(
            &owner_neg_clips,
            &mut owner_neg_metrics,
            |m, l| m.false_accepts.push(l.to_string()),
            Some(&mut shared_adaptive),
            false, // warm pass (consistent with the other negative phases)
        );
        let owner_neg_ms = owner_neg_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "  Owner-negative detection: {}/{} detected ({owner_neg_ms:.0}ms)",
            owner_neg_metrics.detected, owner_neg_metrics.total,
        );
        owner_negative_detection = Some(serde_json::json!({
            "n_clips": owner_neg_metrics.total,
            "detected": owner_neg_metrics.detected,
            "false_accepts": owner_neg_metrics.false_accepts,
            "phase_ms": owner_neg_ms,
            "per_variant": owner_neg_metrics.per_variant.iter().map(|pv| pv_to_json(pv, Some("owner_negative"))).collect::<Vec<_>>(),
            "note": "The enrolled voice's non-wake speech (OWNER_NEGATIVE_PHRASES, \
                     seeds 9000+) measured in detection for the first time.  These are \
                     the SAME clips Phase 3 trains on, so any detection is an \
                     in-distribution misclassification signal (canary-style), not a \
                     generalization claim.  Report-only; does NOT feed \
                     total_false_accepts.",
        }));

        // ── Noise-overlapped detection + cross-speaker probes
        //    (restructured) ──────────────────────────────────────
        // Two matrices under the same phase (both report-only diagnostics —
        // cross-speaker detections are correct speaker-blind behaviour):
        //   1. in_distribution_canaries — the trained-in M2/M3 clips under the
        //      full 13-cell noise matrix (kept from an earlier bench; any detection
        //      is a trained-in canary diagnostic).  The canary clips are fed
        //      ORIGINAL-only: the 13-cell noise-condition vocabulary IS the
        //      canary test's defining structure.  The 5-augmentation dimension
        //      was dropped when M4/M5 moved to the reserved held-out probes
        //      (coverage 260→26 cells — disclosed; neither the enrolled-
        //      speaker phases nor the probe matrix exercise the canary
        //      voices' augmentations);
        //   2. held_out_probes — the RESERVED voices (M4/M5) at unseen seeds
        //      across clean + 5/10/20 dB white (generalization probes).
        phase_start!("Phase 13: Noise-overlapped detection + cross-speaker probes");
        let canary_overlap_results = run_noise_overlap_test(&canary_clips);
        let probe_cells = run_cross_speaker_probe_matrix(&cross_speaker_probe_clips);
        noise_overlap_results = canary_overlap_results;
        probe_overlap_results = probe_cells;
        probe_fa_count = probe_overlap_results
            .iter()
            .map(|(_, _, detected, _)| *detected)
            .sum();
        phase_times[P_NOISE_OVERLAP] = phase_end_ms!();

        // ── Held-out SNR envelope ─────────────────────────────────────────
        // The enlarged held-out wake-only basis is tested clean only in
        // Phase 7d.  This section re-runs the FULL enlarged set through the
        // existing noise-overlap matrix (white/pink/brown × 20/10/5/0 dB +
        // clean — 13 cells) as an in-memory report section — no cache growth.
        // Exercises the encoder pipeline's behaviour under noise.
        // Report-only; warm pass with a shared adaptive state (same
        // methodology as the canary matrix).
        eprintln!("─── Held-out SNR envelope (report-only) ───");
        let snr_start = Instant::now();
        let snr_results = run_noise_overlap_test(&held_out_recall_clips);
        let snr_ms = snr_start.elapsed().as_secs_f64() * 1000.0;
        let snr_total: usize = snr_results.iter().map(|(_, _, _, d)| d.len()).sum();
        let snr_detected: usize = snr_results
            .iter()
            .map(|(_, _, detected, _)| *detected)
            .sum();
        eprintln!("  Held-out SNR envelope: {snr_detected}/{snr_total} detected ({snr_ms:.0}ms)");
        held_out_snr_envelope = Some(serde_json::json!({
            "cells": serde_json::Value::Object(
                snr_results.iter().map(&cell_json).collect(),
            ),
            "n_cells": snr_results.len(),
            "total_variants": snr_total,
            "detected": snr_detected,
            "phase_ms": snr_ms,
            "note": "The ENLARGED held-out wake-only basis (all 40 clips, seeds \
                     3000-3039) through the existing 13-cell noise matrix \
                     (white/pink/brown x 20/10/5/0 dB + clean).  In-memory mixing, \
                     no cache growth.  Report-only.",
        }));
    } else {
        // Degenerate enrollment — skip all detection phases
        phase_times[P_POSITIVE_VARIANTS] = 0;
        phase_times[P_CONFUSABLE_NEGATIVES] = 0;
        phase_times[P_UNRELATED_NEGATIVES] = 0;
        phase_times[P_SILENCE_NEGATIVES] = 0;
        phase_times[P_NOISE_PROFILES] = 0;
        phase_times[P_COOLDOWN] = 0;
        phase_times[P_NOISE_OVERLAP] = 0;
        // On the degenerate path the cooldown.* keys stay
        // null (hoisted vars), but the skip must be VISIBLE — emit the reason
        // so the report never silently lacks it (mirrors the earlier
        // hoisting pattern; the F1-missing / warm-pass-failure skip reasons
        // are set inside the Phase 13 block above).
        let reason = "degenerate enrollment (consistency gate failed or too few \
             utterances) — all detection phases, including Phase 12 cooldown \
             verification, were skipped (report-only)"
            .to_string();
        warn!("{reason}");
        cooldown_skip_reason = Some(reason);
    }

    // ── Env-gated FAPH phase ────────────────────────────────
    // Real-audio false-acceptance-per-hour bench phase.  Runs AFTER the
    // existing phases (reuses the bench's flock + harness timeout); the
    // standard-run report surface is untouched because the report section is
    // only added to the JSON when the env gate is on (None → no key).
    //
    // The phase feeds the FULL pinned corpus (812 files).
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
        .map(|(_, _, detected, _)| *detected)
        .sum();
    // The canary target is derived from the actual measured cell×variant
    // matrix so every report string stays in lockstep with the measured count.
    let noise_overlap_total_variants: usize = noise_overlap_results
        .iter()
        .map(|(_, _, _, detail)| detail.len())
        .sum();
    let probe_total_variants: usize = probe_overlap_results
        .iter()
        .map(|(_, _, _, detail)| detail.len())
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

    // ── Score-gate trigger metrics for negative phases ────────
    let conf_gate = GateTriggerMetrics::compute(&conf_metrics.per_variant);
    let unrel_gate = GateTriggerMetrics::compute(&unrelated_metrics.per_variant);
    let silence_gate = GateTriggerMetrics::compute(&silence_metric.per_variant);

    // Per-tier confusable score-gate trigger metrics (both seed bands).
    let mut conf_gate_by_tier: [GateTriggerMetrics; 3] = Default::default();
    for pv in &conf_metrics.per_variant {
        let phrase = confusable_phrase(&pv.variant);
        conf_gate_by_tier[tier_for_phrase(phrase).index()].accumulate(pv);
    }

    // Noise: per-profile + aggregate score-gate trigger metrics.
    // When the enrollment is degenerate, noise_metrics is empty because
    // all detection phases are skipped.  In that case produce default
    // (all-zero) metrics so downstream output paths (JSON, info!(),
    // eprintln) always have consistent data without conditional guards.
    let noise_gate_per_profile: Vec<GateTriggerMetrics> = if noise_metrics.is_empty() {
        all_noise_profiles()
            .map(|_| GateTriggerMetrics::default())
            .collect()
    } else {
        noise_metrics
            .iter()
            .map(|m| GateTriggerMetrics::compute(&m.per_variant))
            .collect()
    };
    let noise_gate_aggregate =
        noise_gate_per_profile
            .iter()
            .copied()
            .fold(GateTriggerMetrics::default(), |mut acc, m| {
                acc.total_variants += m.total_variants;
                acc.gate_triggers += m.gate_triggers;
                acc.full_pipeline_fa += m.full_pipeline_fa;
                acc
            });

    // Enrolled-speaker detection counts — the single-speaker POSITIVE class
    // (the acceptance basis is the held-out wake-only recall set in
    // enrolled_speaker.acceptance).  Shared by the report banner,
    // the threshold-status block, and the catastrophic regression guard.
    // Falls back to the Phase-7 positive-variant counts when the enrolled
    // phase did not run (degenerate enrollment).
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
    // not (degenerate enrollment), enrolled_detected/
    // enrolled_total fall back to the Phase-7 positive variants.  The
    // MIN_DETECTION_RATE suggestion and the ratchet guard are skipped in that
    // case so a degenerate run cannot print a ratchet-to-zero 0.0 suggestion
    // (ratchet guard).
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
        "Silence false accepts: {} / {} ({} durations, limit ≤{})",
        silence_metric.false_accepts.len(),
        silence_metric.total,
        SILENCE_DURATIONS.len(),
        0, // all tiers have silence limit = 0
    );
    let n_noise_profiles = all_noise_profiles().count();
    info!(
        "Noise false accepts: {} / {} ({} profiles, limit ≤{}–{} across tiers)",
        noise_false_accepts.len(),
        n_noise_profiles,
        n_noise_profiles,
        tier_limit_sets[0].noise,
        tier_limit_sets[2].noise,
    );
    if !noise_false_accepts.is_empty() {
        info!("  False triggers: {:?}", noise_false_accepts);
    }

    // ── Score-gate trigger report ──────────────────────────
    info!("──────────────────────────────────────────────");
    info!("  Score-gate trigger metrics (negative phases):");

    let ct_line = |name: &str, ct: &GateTriggerMetrics| -> String {
        format!(
            "    {name}: {tr}/{tot} triggers, fa={fa}",
            tr = ct.gate_triggers,
            tot = ct.total_variants,
            fa = ct.full_pipeline_fa,
        )
    };
    info!("{}", ct_line("Confusable", &conf_gate));
    for (i, tier_name) in [(0, "easy"), (1, "medium"), (2, "hard")] {
        info!(
            "    └─ {}",
            ct_line(tier_name, &conf_gate_by_tier[i]).trim_start()
        );
    }
    info!("{}", ct_line("Unrelated", &unrel_gate));
    info!("{}", ct_line("Silence", &silence_gate));
    info!("{}", ct_line("Noise (aggregate)", &noise_gate_aggregate));
    for (i, (label, _)) in all_noise_profiles().enumerate() {
        let profile_ct = &noise_gate_per_profile[i];
        info!("    └─ {}", ct_line(label, profile_ct).trim_start());
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
    // DIAGNOSTICS — cross-speaker wake-word detections are speaker-blind
    // behaviour, so these report rates without warnings and do
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

    // NOTE: acceptance/threshold checks are report-only — they emit warnings
    // above but never abort the benchmark.  Hard assertions remain only for
    // production-behavior contracts (the cooldown gate and accumulation cap in
    // the cooldown phase).  Run data is reported in both JSON (below) and this
    // stderr report.

    // Build the JSON output
    // Build per-variant negative diagnostics with FULL per-variant detail.
    // Confusable variants are always tier-qualified.
    // The flat list is reused for score distributions below.
    let mut all_neg_pv: Vec<(&PerVariantResult, String)> = Vec::new();
    for pv in &conf_metrics.per_variant {
        let phrase = confusable_phrase(&pv.variant);
        let variant_tier = tier_for_phrase(phrase);
        let band = confusable_band(&pv.variant);
        all_neg_pv.push((pv, format!("{band}_{}", variant_tier.as_str())));
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
    // Diagnostics: verdicts, distributions, train/test alignment,
    // and reproducibility metadata
    // ═══════════════════════════════════════════════════════════════════════

    // ── Exhaustive per-variant verdicts for positives ──
    let mut verdict_counts: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    for pv in &pos_metrics.per_variant {
        *verdict_counts
            .entry(classify_miss(pv).as_str())
            .or_insert(0) += 1;
    }

    // ── Score-gate discrimination evidence ──
    // Peak-score distributions (warm-up excluded after the instrumentation
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
            .filter(|s| s[ROLLING_SUM_IDX] < MIN_GATE_THRESHOLD)
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
            .filter(|s| s[ROLLING_SUM_IDX] >= MIN_GATE_THRESHOLD)
            .count();
        (total_frames, above_threshold)
    };

    // ── Per-augmentation detection rates ──
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

    // ── Reproducibility metadata ──
    let warmup_source = if WARMUP_TTS_CACHE.get().is_some() {
        "tts"
    } else {
        "pink_noise_fallback"
    };

    // ── Phase-B alignment probes (report-only) ──

    // TTS-echo gate probe.  The offline bench never runs the mic loop,
    // so the probe locks the voice-tests setter → is_playback_active()
    // predicate roundtrip — the exact predicate the production mic arm
    // consults to drop the WHOLE chunk (raw ring included) while
    // playback is active (voice.rs mic-chunk arm).  The actual
    // chunk-drop behavior is exercised only by that production arm, which
    // the offline bench cannot drive; zero production change.
    let tts_echo_gate_probe = {
        tts::set_playback_active_for_test(true);
        let gate_active = tts::is_playback_active();
        tts::set_playback_active_for_test(false);
        let gate_cleared = !tts::is_playback_active();
        serde_json::json!({
            "probe": "TTS-echo gate",
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

    // Production metrics snapshot — the only production diagnostics
    // surface with zero bench coverage.  The offline bench
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
                "Production metrics implausible — \
                 embeddings_computed={}, drop_rate={drop_rate}",
                metrics.embeddings_computed,
            );
        }
        // Real-time CPU budget probe: the scoring stride is 160 ms of audio
        // (SCORE_STRIDE_SAMPLES = 2560 samples at 16 kHz), so a per-window
        // encoder forward must stay under ~160 ms for the pipeline to keep up
        // with a live mic.  The offline bench feeds audio faster than
        // real-time, but the production encode path
        // (handle_wake_word_detection → encode_window) is executed under
        // release codegen, so the measured latency is the per-window encode
        // cost.
        let stride_budget_ns = super::SCORE_STRIDE_SAMPLES as u64 * 1_000_000_000 / 16_000;
        let rolling_latency_ns = metrics.avg_embedding_latency_ns;
        serde_json::json!({
            "probe": "production metrics",
            "chunks_received": metrics.chunks_received,
            "dropped_chunks": metrics.dropped_chunks,
            "embeddings_computed": metrics.embeddings_computed,
            "drop_rate": drop_rate,
            "plausible": plausible,
            "rolling_avg_embedding_latency_ns": rolling_latency_ns,
            "lifetime_avg_embedding_latency_ns": metrics.lifetime_avg_embedding_latency_ns(),
            "stride_budget_ns": stride_budget_ns,
            "within_stride_budget": rolling_latency_ns > 0 && rolling_latency_ns <= stride_budget_ns,
            "note": "drop_rate guards the zero denominator (0/0 → 0.0); \
                     chunks_received/dropped_chunks are mic-loop-only and stay 0 in \
                     the offline bench by construction.  The latency fields measure \
                     the production encode path (rolling avg of the last 100 window \
                     encodings) against the 160 ms scoring stride budget — a \
                     report-only real-time budget probe; the offline bench cannot \
                     measure live-mic drop behaviour.",
        })
    };

    // Embedding cache probe: there is no embedding disk cache — the encoder
    // pipeline re-encodes from raw audio on each run (the PCM cache keeps TTS
    // synthesis cheap).  The probe documents the absence.
    let embedding_cache_probe = serde_json::json!({
        "probe": "embedding cache",
        "read_back_byte_identical": serde_json::Value::Null,
        "note": "There is no embedding disk cache — the encoder pipeline \
                 re-encodes raw audio through the shared Qwen3-ASR encoder per \
                 run; only the TTS PCM cache survives.",
    });

    // Enrollment-quality scoring (report-only).  The bench's
    // clips have no pre-speech noise ring → heuristic SNR (estimate_snr_energy).
    let enrollment_quality_probe_value = enrollment_quality_probe(&train_clips);

    // VAD feed-contract cross-check (report-only).  Replays
    // production's feed pattern (only NEW hop samples) vs the bench's VAD
    // segmentation on the same audio — the VAD feed-contract guard.
    let vad_feed_cross_check_probe_value =
        vad_feed_cross_check_probe(train_clips.first().map_or(&[], |(pcm, _)| &pcm[..]));

    // Mel-normalization consistency (report-only).  Validates the acceptance
    // criterion "mel-normalization consistency between enrollment and
    // streaming is validated within the benchmark": the same real clip
    // encoded whole-utterance (enrollment) vs streaming-accumulated
    // (detection) must yield near-identical embeddings.
    let mel_norm_consistency_probe_value =
        mel_normalization_consistency_probe(train_clips.first().map_or(&[], |(pcm, _)| &pcm[..]));

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
            // No training seed — the enrollment prototype is the
            // L2-normalized mean of utterance embeddings (no trainable head).
            "tts_warmup": 947,
            "note": "No training seed — \
                     the enrollment is deterministic given the TTS PCM cache.",
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
                     phrase that tier is informational only.",
        },
        "model_hashes": {
            "tts_model_version_hash": model_version_hash,
            "pipeline": "qwen3-asr-encoder",
            "note": "The old OpenWakeWord mel/embed ONNX models are gone — the \
                     wake-word pipeline shares the local Qwen3-ASR transcriber \
                     (no separate artifact, no per-model hashes).",
        },
        "cache_state": {
            "warmup_tts_cached": WARMUP_TTS_CACHE.get().is_some(),
            "cache_dir": cache_dir_path.display().to_string(),
        },
        // Encoder-window geometry the bench drives and production reads
        // (previously invisible to the report).
        "geometry_constants": {
            "window_samples": crate::audio::wake_word::WINDOW_SAMPLES,
            "window_mel_frames": crate::audio::wake_word::WINDOW_MEL_FRAMES,
            "score_stride_mel_frames": crate::audio::wake_word::SCORE_STRIDE_MEL_FRAMES,
            "embedding_dim": crate::audio::wake_word::WAKE_WORD_EMBEDDING_DIM,
        },
        "warmup_source": warmup_source,
        // Phase-B alignment probes (report-only, additive).
        "tts_echo_gate": tts_echo_gate_probe,
        "production_metrics": production_metrics_probe,
        "embedding_cache": embedding_cache_probe,
        "enrollment_quality": enrollment_quality_probe_value,
        "vad_feed_cross_check": vad_feed_cross_check_probe_value,
        "mel_normalization_consistency": mel_norm_consistency_probe_value,
        "train_test_split": {
            "note": format!(
                "No split: all {} enrollment clips are training \
                 data; detection control is the held-out wake-only recall set \
                 (enrolled_speaker.acceptance)",
                train_clips.len(),
            ),
            "n_train_clips": train_clips.len(),
            "n_train": utterance_embeddings.len(),
            "train_clips": train_clips.iter().map(|(_, l)| l.clone()).collect::<Vec<_>>(),
        },
    });

    // ── JSON sub-objects (built separately to stay under serde_json's json!
    // macro recursion limit — the main report object nests several levels) ──

    // Detection summary: warm (pos_metrics) + cold-start (cold_metrics)
    // over the held-out augmented diagnostics (Phase 7 —
    // bounded {speed_down, speed_down_090, white-10, brown-10} variants of a
    // held-out wake-only subset).  The raw-clip held-out recall acceptance
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
                 clips.  The trained-in cross-speaker canaries live in \
                 noise_overlap.in_distribution_canaries (M2/M3) and the \
                 held-out probes in noise_overlap.held_out_probes (M4/M5).  The \
                 raw-clip held-out recall acceptance basis (unseen seeds, \
                 wake-only) is reported in enrolled_speaker.acceptance.",
        "miss_verdicts": verdict_counts,
        "total_misses": pos_metrics.total - pos_metrics.detected,
        // Cold-start pass — fresh PipelineCtx per variant,
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

    // Production gates enrollment on this self-test; the
    // benchmark is report-only so it records the outcome
    // instead.  When "passed" is false, the reported detection/FA numbers
    // come from a model production would refuse to deploy.
    let self_test_json = serde_json::json!({
        "passed": self_test_result.as_ref().is_some_and(Result::is_ok),
        "error": self_test_result
            .as_ref()
            .and_then(|r| r.as_ref().err().map(ToString::to_string)),
        "skipped": self_test_result.is_none(),
    });

    let gate_trigger_json = serde_json::json!({
        "confusable": gt_to_json(&conf_gate),
        "confusable_by_tier": {
            "easy": gt_to_json(&conf_gate_by_tier[0]),
            "medium": gt_to_json(&conf_gate_by_tier[1]),
            "hard": gt_to_json(&conf_gate_by_tier[2]),
        },
        "unrelated": gt_to_json(&unrel_gate),
        "silence": gt_to_json(&silence_gate),
        "noise": {
            "aggregate": gt_to_json(&noise_gate_aggregate),
            "per_profile": serde_json::Value::Object(
                all_noise_profiles()
                    .enumerate()
                    .map(|(i, (label, _))| {
                        (label.to_string(), gt_to_json(&noise_gate_per_profile[i]))
                    })
                    .collect()
            ),
        },
    });

    let gate_diagnostics_json = serde_json::json!({
        "pipeline": "qwen3-asr-encoder",
        // With no trainable head, the "soft scores" are the enrollment
        // soft scores (cosine through calibration): pos = utterance
        // embeddings, neg = the negative pool.  Calibration diagnostics
        // live under "enrollment_diagnostics".
        "pos_scores_mean": pos_scores_mean,
        "pos_scores_min": pos_scores_min,
        "pos_scores_max": pos_scores_max,
        "neg_scores_mean": neg_scores_mean,
        "neg_scores_min": neg_scores_min,
        "neg_scores_max": neg_scores_max,
        "pos_scores_deciles": pos_scores_deciles,
        "neg_scores_deciles": neg_scores_deciles,
        "enrollment_diagnostics": enrollment_diagnostics,
        "degenerate": degenerate,
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
    // speaker-blind behaviour), reported without pass/fail
    // semantics and NOT counted into total_false_accepts.
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
             13-cell noise matrix (any detection is a trained-in speaker-blindness \
             diagnostic).  held_out_probes = RESERVED voices (absent from every \
             training path) at unseen seeds across clean + 5/10/20 dB white.  Both \
             are DIAGNOSTICS: speaker-blind detection means a non-enrolled voice's \
             wake word fires.  Measured {canary_total} canary + \
             {probe_total} probe cells this run.",
            canary_total = noise_overlap_total_variants,
            probe_total = probe_total_variants,
        ),
    });

    // Enrolled-speaker phase (Phase 7d) + safety gate.
    // The safety gate measures the false-positive impact of the deferred
    // in-distribution windows (burst + boundary double-scoring) on the
    // negative/confusable set.
    let enrolled_json = enrolled_report.unwrap_or_else(
        || serde_json::json!({"note": "Enrolled-speaker phase skipped (degenerate enrollment)"}),
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
        "_note": "Report-only benchmark — acceptance metrics are measured and \
             compared against the documented limits, never hard-gated (hard \
             assertions exist only for production-behavior contracts in the \
             cooldown phase).  The wake-word pipeline embeds the trailing 0.76 s \
             window through the shared Qwen3-ASR encoder (1024-dim cosine soft \
             scores); enrollment is a prototype (L2-normalized centroid of \
             utterance embeddings) calibrated against negative samples; detection \
             is immediate-fire and speaker-blind.  The enrolled-speaker acceptance \
             basis is a held-out wake-only recall set (v3: 40 unseen \
             enrolled-voice renderings, seeds 3000+, Wilson CI reported; \
             embedded-in-sentence detection is not measured).  total_false_accepts \
             counts end-to-end detections over the negative corpus \
             (confusable/unrelated/silence/noise); cross-speaker wake-word \
             detections are reported as diagnostics in \
             noise_overlap / false_accepts.cross_speaker_probes and do NOT count \
             into total_false_accepts.  The confusable/unrelated negative pools \
             are built bench-locally over the negative-pool styles; the negative \
             pool is encoded directly from raw audio (ambient noise profiles × 2 \
             levels, owner-negative VAD-gated speech + brown-10, \
             confusable/unrelated VAD-gated TTS).  The TEST side: 40-clip \
             wake-only basis (seeds 3000-3039), a held-out SNR envelope over it \
             (13-cell noise matrix, in-memory), doubled confusable/unrelated \
             detection pools (new seed bands 810+/910+ with distinct label \
             prefixes), a three-duration silence matrix, 4 detection-only noise \
             profiles (fresh seeds 52-55), wake-over-babble cells (TTS-voice \
             overlays), an owner-negative detection block (trained-in clips, \
             canary-style), and a cross-speaker probe scale-up (20 seeds/voice + \
             a 5 dB cell).",
        "total_false_accepts": total_false_accepts,
        // additive: single-voice enrollment disclosure — voice
        // allocation, the guided-prompt DSP recipe, and any production
        // consistency-gate failure the bench recorded (report-only).
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
                "note": "Realism comes from the SNR against a fixed noise floor, the \
                         spectral rolloff, and the level RELATIONSHIPS (every group's \
                         noise lands at the same absolute RMS as the normal 20 dB floor).",
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
        "gate_trigger_metrics": gate_trigger_json,
        "phases": {
            "phase_1_enrollment_audio_ms": phase_times[P_ENROLLMENT_AUDIO],
            "phase_2_vad_enrollment_ms": phase_times[P_VAD_ENROLLMENT],
            "phase_3_negative_training_data_ms": phase_times[P_NEG_TRAINING_DATA],
            "phase_4_finalize_enrollment_ms": phase_times[P_FINALIZE_ENROLLMENT],
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
        // Cold-start pass per-variant detail (same schema as
        // per_variant_results, but measured without consume_warmup / fresh
        // adaptive bootstrap).  Empty when the enrollment is degenerate.
        // The cold pass exercises the burst sweep (buffer >= 68 frames) or
        // the segment-end pass for short utterances.  The enrolled-speaker
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
        "gate_diagnostics": gate_diagnostics_json,
        "per_augmentation_detection": per_augmentation,
        "cooldown": {
            "detection_time_ms": cooldown_detection_time_ms,
            "first_detection_fired": cooldown_first_detected,
            "cooldown_suppressed_redetection": cooldown_suppressed,
            "detection_recovered_after_cooldown": cooldown_after_recovered,
            // Additive keys (re-point at the enrolled-speaker F1
            // original variant; visible skip via skip_reason, never silent).
            "source_variant": cooldown_source_variant,
            "skip_reason": cooldown_skip_reason,
            "suppressed_at_2_5s": cooldown_suppressed_at_2_5s,
            "accumulation_cap_samples": super::AUDIO_BUFFER_MAX,
            "audio_buffer_len_during_cooldown": cooldown_accumulation_cap_observed,
            "buffered_audio_processed_after_expiry": cooldown_buffered_audio_processed_after_expiry,
            "note": "Hard assertions: suppressed at ~2.5s, fires at \
                     ~3.5s — slack margins, jitter-safe; probe sleeps are excluded from \
                     detection_time_ms.",
        },
        "reproducibility": reproducibility,
        // Enrolled-speaker benchmark phase (Phase 7d) — F1's 5
        // raw-level augmentation variants through the real streaming cold
        // pass (in-sample / training-side control, NOT generalization).
        "enrolled_speaker": enrolled_json,
        // False-positive impact of the deferred
        // in-distribution windows on the negative/confusable set.
        "safety_gate": safety_gate_json,
        "config": {
            "num_enrollment_variants": NUM_ENROLLMENT_VARIANTS,
            "min_detection_rate": MIN_DETECTION_RATE,
        }
    });

    // ── FAPH section (env-gated additive key) ───────────────
    // Only present when MAHBOT_FAPH=1 (ran or documented-skipped).  Standard
    // runs leave `faph_report` as None → the key is absent from the report.
    let mut json = json;
    // Phase 0 additive default-run keys (added post-macro — the main json!
    // block is already at serde_json's recursion limit).
    json["deployability"] = deployability.clone();
    json["cross_run"] = cross_run;
    // Expanded TEST-surface report sections (absent when degenerate — the keys
    // are only inserted when the corresponding phase ran).
    if let Some(v) = held_out_snr_envelope {
        json["held_out_snr_envelope"] = v;
    }
    if let Some(v) = overlapping_speech {
        json["overlapping_speech"] = v;
    }
    if let Some(v) = owner_negative_detection {
        json["owner_negative_detection"] = v;
    }
    // Wall-clock measurement: the whole run (start → report emission),
    // report-assembly window included, so run-time is auditable from the JSON.
    // No budget is asserted — the measured number is the record.
    json["performance"] = serde_json::json!({
        "wall_clock_secs": overall_start.elapsed().as_secs_f64(),
    });
    if let Some(faph) = faph_report {
        json["faph"] = faph;
    }

    // Output delimited JSON for CI tooling
    println!("--- BENCHMARK_JSON_BEGIN ---");
    let json_text = serde_json::to_string_pretty(&json).expect("JSON serialization");
    println!("{json_text}");
    println!("--- BENCHMARK_JSON_END ---");

    // ── Persist the report for post-run analysis ────
    // The report is emitted to stdout for CI, but a hard harness
    // timeout with end-only emission can lose it entirely.  Mirror the JSON to
    // ~/.mahbot/voice_pipeline_e2e_report.json (the same directory as the
    // benchmark lock file) so a run's data survives regardless of how the
    // process ends, and a final ticket comment can cite exact numbers.
    if let Ok(report_dir) = crate::config::default_config_dir() {
        let report_path = report_dir.join("voice_pipeline_e2e_report.json");
        // ── Per-run timestamped archive ──────
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

    // ── Human-readable report (stderr) ────────────────────────
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
    // pass/fail gating.  The marks now reflect the ACTUAL limits
    // instead of being unconditional ✓.
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
        "           Cross-speaker diagnostics (speaker-blind): canaries {noise_overlap_detections}/{noise_overlap_total_variants}, probes {probe_fa_count}/{probe_total_variants}",
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

    // ── Score-gate trigger summary ────────────────────────
    let ct_stderr = |ct: &GateTriggerMetrics| -> String {
        format!(
            "{}/{} triggers, fa={}",
            ct.gate_triggers, ct.total_variants, ct.full_pipeline_fa
        )
    };
    eprintln!(
        "         Score-gate crossings:\
         \n           Confusable: {conf}\
         \n             Easy:     {easy}\
         \n             Medium:   {med}\
         \n             Hard:     {hard}\
         \n           Unrelated:  {unrel}\
         \n           Silence:    {sil}\
         \n           Noise:      {noise}",
        conf = ct_stderr(&conf_gate),
        easy = ct_stderr(&conf_gate_by_tier[0]),
        med = ct_stderr(&conf_gate_by_tier[1]),
        hard = ct_stderr(&conf_gate_by_tier[2]),
        unrel = ct_stderr(&unrel_gate),
        sil = ct_stderr(&silence_gate),
        noise = ct_stderr(&noise_gate_aggregate),
    );

    eprintln!(
        "         ═══════════════════════════════════════════════════════════\n\
                   BENCHMARK COMPLETE (report-only — no pass/fail gating)\n\
         ═══════════════════════════════════════════════════════════",
    );

    // ── Deployability verdict banner (Phase 0, prominent) ─────────────
    // Production's enrollment self-test gate is not the bench's gate: the
    // bench stays report-only.  But an enrollment production would reject must
    // never be misread as deployable, so the warning is prominent in the
    // human-readable banner AND structured in the top-level deployability key.
    // Uses the (passed, total, required, deploy_would_pass) tuple returned by
    // deployability_json directly — the JSON is only the structured mirror,
    // never re-parsed here.  `deploy_would_pass` is Option<bool> (None = no
    // enrollment), so the no-enrollment arm below is explicit, not a hardcoded
    // false.
    match &self_test_result {
        None => eprintln!(
            "         Deployability: NOT MEASURED — no enrollment \
             (consistency gate failed or too few utterances); report-only"
        ),
        Some(_) if deploy_would_pass == Some(true) => eprintln!(
            "         Deployability: enrollment would PASS production's self-test gate \
             ({passed}/{total} utterances trigger detection) — informational only"
        ),
        Some(_) => eprintln!(
            "\n         ⚠  PRODUCTION WOULD REJECT THIS ENROLLMENT: only {passed}/{total} \
             utterances triggered detection (gate ≥{:.0}%).\n         \
             These recall numbers are labeled 'enrollment production would reject — \
             informational only'.\n         The bench is report-only: this does NOT fail \
             the run, but the enrollment is not deployable as-is.\n",
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
    // (zero detections = zero false accepts).
    // NOTE: report-only — warns instead of asserting.
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
        "         Cross-speaker diagnostics (speaker-blind): \
         canaries {noise_overlap_detections}/{noise_overlap_total_variants}, \
         probes {probe_fa_count}/{probe_total_variants}",
    );

    // Per-tier confusable false-accept checks
    // NOTE: report-only — warns instead of asserting.
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
    // NOTE: report-only — warns instead of asserting.
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

    // Enrollment degeneracy check — report-only.
    // If the enrollment is degenerate, the benchmark still completes and reports
    // so we can diagnose.
    if degenerate {
        warn!(
            "Enrollment degenerate (consistency gate failed or too few utterances) — \
             detection phases were skipped. See JSON output for details."
        );
    }

    // Noise-overlap warnings already occurred at the check site above (line ~2843).
    // No pass/fail gating — report-only.

    // ── Final result ──
    // All threshold checks above emit warnings on violation — no pass/fail gating.
    info!("═══ E2E Voice Pipeline benchmark complete (report-only — no pass/fail gating) ═══");

    // Stop heartbeat thread and wait for it to exit.
    heartbeat_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = heartbeat_handle.join();
}

// ── Fixture tests ─────────────────────────────────────────
// Compile under `cargo test --features voice-tests` (the bench module is
// feature-gated; `#[cfg(test)]` keeps these out of the harness=false bench
// binary).  There is no golden-hash test for `pcm_augment_enrollment_variants`
// (the encoder pipeline has no trainable head); the function survives as a
// detection-diagnostics variant generator.

#[cfg(test)]
mod tests {
    use super::*;

    // ── FAPH Poisson CI helpers ──────────────────────────────

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
    /// rate is 0.5·χ²(0.975, 2)/5.99 ≈ 0.61/h.
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
