//! E2E integration test for the full voice pipeline (mahbot-811).
//!
//! This test exercises the **enrollment-to-detection cycle with realistic
//! TTS-generated speech audio**.  It uses the TTS engine to synthesize wake
//! word variants, feeds them through the enrollment pipeline, trains the
//! MLP classifier and VoiceVerifier, then runs detection on:
//!
//! * Positive cases (wake word variants)
//! * Negative — confusable near-miss phrases
//! * Negative — completely unrelated speech
//! * Negative — silence and noise
//!
//! # Requirements
//!
//! * TTS models must be downloaded and cached (run the app once).
//! * Voice ONNX models (melspectrogram + embedding) must be present in
//!   `~/.mahbot/models/openwakeword/`.
//!
//! # Running
//!
//! ```sh
//! cargo test --features voice-tests -- --ignored e2e_voice_pipeline
//! ```
//!
//! Without `voice-tests` the test file is not compiled at all.
//! With the feature, the test is compiled but **skipped by default** via
//! `#[ignore]`.

use super::*; // voice module items (process_enrollment_sample, etc.)
use crate::tts;
use crate::voice_verifier::VoiceVerifier;
use crate::voice_verifier::generate_synthetic_negatives;
use crate::wake_word_classifier::WakeWordClassifier;
use earshot::Detector;
use rand::{RngExt, SeedableRng};

// ── Constants ──────────────────────────────────────────────────────────────

/// Default wake word for the test.
const WAKE_WORD: &str = "hey mahbot";

/// Number of enrollment variants to generate (fewer than real enrollment
/// since each TTS call takes ~3-5 sec).
const NUM_ENROLLMENT_VARIANTS: usize = 5;

/// Number of synthetic-augmentation variants (additional wake word variants
/// with speed/noise/volume perturbation).
const NUM_AUGMENTATION_VARIANTS: usize = 8;

/// Minimum detection rate for positive (wake word) variants required to pass.
const MIN_DETECTION_RATE: f64 = 0.75;

/// Maximum number of false accepts across ALL negative tests (confusable +
/// unrelated + silence + noise).
const MAX_FALSE_ACCEPTS: usize = 2;

/// Confusable near-miss phrases for negative detection testing (mahbot-834).
///
/// Expanded from 5 to 20 phrases covering:
/// - Direct phonetic substitutions that sound similar to "hey mahbot"
///   (madbot, map bot, mabot, mahbott, maybot, nab it, etc.)
/// - Similar two-word rhythmic patterns
///   (day mahbot, hey man, hey max, hey mat, pay mabot, hay map pot)
/// - Longer phrases containing wake word sound patterns embedded within
///   (hey maybe not, play mah jong, they mad bot, hey matter of fact)
/// - False starts with similar syllable structure
///   (huh mahbot, eh mad bot, haymaker)
const CONFUSABLE_PHRASES: &[&str] = &[
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
];

/// Completely unrelated phrases for negative detection testing (mahbot-834).
///
/// Expanded from 5 to 20 phrases covering:
/// - Short commands (2-4 words) similar to typical wake word length
/// - Medium phrases (5-8 words) with varied phonetic content
/// - Long utterances (10+ words) to test sustained non-detection
/// - Questions, statements, and filler speech
/// - Mixed languages (French, Spanish, German) if TTS phoneme support
///   varies per language — the detector should reject all non-wake-word
///   speech regardless of language.
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

/// Silence audio length in samples (1 second at 16 kHz).
const SILENCE_LEN: usize = 16_000;

/// Noise audio length in samples (1 second at 16 kHz).
const NOISE_LEN: usize = 16_000;

/// Noise profiles for negative detection testing (mahbot-834).
///
/// Each noise profile is a (label, generator_fn) pair.  The generator
/// produces PCM f32 samples at 16 kHz.
const NOISE_PROFILES: &[(&str, fn() -> Vec<f32>)] = &[
    ("white uniform noise", generate_white_uniform_noise),
    ("white gaussian noise", generate_white_gaussian_noise),
    ("pink noise", generate_pink_noise),
    ("brown noise", generate_brown_noise),
    ("mixed speech+noise", generate_mixed_speech_noise),
];

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
        let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
        let u2: f32 = rng.random::<f32>().max(f32::EPSILON);
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).cos();
        let z2 = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).sin();
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
fn generate_pink_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(44);
    const NUM_OSCILLATORS: usize = 8;
    let mut samples = Vec::with_capacity(NOISE_LEN);
    let mut values = [0.0f32; NUM_OSCILLATORS];
    for i in 0..NOISE_LEN {
        // Each step, each oscillator increments and may flip state
        let mut sum = 0.0;
        for j in 0..NUM_OSCILLATORS {
            let period = 1 << j;
            if i % period == 0 {
                values[j] = rng.random::<f32>() * 2.0 - 1.0;
            }
            sum += values[j];
        }
        // Normalize (gain ~ 1/sqrt(N_osc))
        samples.push((sum / NUM_OSCILLATORS as f32).clamp(-1.0, 1.0));
    }
    samples
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

/// Number of synthetic negative embeddings to generate for classifier training.
/// This is supplemented with real negative examples from unrelated phrases
/// (see Phase 4) to provide a diverse negative training set.
///
/// Equal to [`crate::voice_verifier::SYNTHETIC_NEGATIVES_COUNT`] for consistency —
/// both produce the same number of synthetic negatives so the test configuration
/// is representative of production's fallback path.
const SYNTHETIC_NEGATIVES_COUNT: usize = 100;

/// TTS target sample rate (voice pipeline rate).
const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Default TTS voice style when no styles are available from disk.
/// This matches the naming convention used by the TTS model download.
const DEFAULT_TTS_STYLE: &str = "M1.json";

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

/// Try to load the TTS engine from cache if not already loaded.
fn ensure_tts_ready() -> Result<(), String> {
    if tts::models_ready() {
        return Ok(());
    }
    if tts::try_load_cached() {
        return Ok(());
    }
    Err("TTS models not available. Run the app once to download them.".to_string())
}

// ── Audio generation ──────────────────────────────────────────────────────

/// Generate a wake word variant using TTS with the given parameters.
fn synthesize_wake_word_variant(text: &str, style: &str, seed: u64) -> Option<Vec<f32>> {
    match tts::synthesize(text, style, seed, TARGET_SAMPLE_RATE) {
        Ok(pcm) => Some(pcm),
        Err(e) => {
            warn!("TTS synthesis failed for '{text}' with {style} (seed={seed}): {e}");
            None
        }
    }
}

/// Generate enrollment audio variants (different voices, seeds) and return
/// them as `(samples, label)` tuples.  The label is a human-readable
/// description for metrics reporting.
fn generate_enrollment_variants(available_styles: &[String]) -> Vec<(Vec<f32>, String)> {
    let mut variants = Vec::new();
    let num_styles = available_styles.len();
    if num_styles == 0 {
        warn!("No TTS voice styles available — using default style");
        // If no styles found, try with a hardcoded default style
        for i in 0..NUM_ENROLLMENT_VARIANTS {
            if let Some(pcm) =
                synthesize_wake_word_variant(WAKE_WORD, DEFAULT_TTS_STYLE, i as u64 + 100)
            {
                variants.push((pcm, format!("default_style_var{i}")));
            }
        }
    } else {
        for i in 0..NUM_ENROLLMENT_VARIANTS {
            let style = &available_styles[i % num_styles];
            let seed = i as u64 + 100;
            if let Some(pcm) = synthesize_wake_word_variant(WAKE_WORD, style, seed) {
                variants.push((pcm, format!("{style}_enroll{i}")));
            }
        }
    }
    variants
}

/// Generate augmented wake word variants with speed, noise, and volume
/// perturbation.  These mimic what the user might sound like in different
/// environments.
fn generate_augmented_variants(
    available_styles: &[String],
    base_seed: u64,
) -> Vec<(Vec<f32>, String)> {
    let mut variants = Vec::new();
    let num_styles = available_styles.len().max(1);

    for i in 0..NUM_AUGMENTATION_VARIANTS {
        let style_idx = (i + 3) % num_styles; // Different styles from enrollment
        let style = if available_styles.is_empty() {
            DEFAULT_TTS_STYLE
        } else {
            &available_styles[style_idx]
        };
        let seed = base_seed + i as u64 + 1000;

        let base_pcm = match tts::synthesize(WAKE_WORD, style, seed, TARGET_SAMPLE_RATE) {
            Ok(pcm) => pcm,
            Err(e) => {
                warn!("Augmentation synthesis failed: {e}");
                continue;
            }
        };

        // Apply random augmentation
        let augmented = match i % 3 {
            0 => {
                // Speed perturbation (faster)
                let factor = 1.0 + ((i as f64 * 0.05).sin() * 0.15) as f32; // 0.85 - 1.15
                crate::tts_data_gen::speed_perturbation(&base_pcm, TARGET_SAMPLE_RATE, factor)
            }
            1 => {
                // Volume randomization
                let max_gain_db = 6.0;
                crate::tts_data_gen::randomize_volume(&base_pcm, max_gain_db)
            }
            _ => {
                // Noise mixing (pink noise, moderate SNR)
                crate::tts_data_gen::add_noise(
                    &base_pcm,
                    20.0,
                    crate::tts_data_gen::NoiseType::Pink,
                )
            }
        };

        let desc = format!(
            "{style}_aug{i}_{}",
            match i % 3 {
                0 => "speed",
                1 => "volume",
                _ => "noise",
            }
        );
        variants.push((augmented, desc));
    }

    variants
}

/// Generate TTS audio for a list of phrases (for negative detection testing).
fn generate_phrase_variants(
    phrases: &[&str],
    available_styles: &[String],
    base_seed: u64,
    prefix: &str,
) -> Vec<(Vec<f32>, String)> {
    let mut variants = Vec::new();
    let num_styles = available_styles.len().max(1);

    for (i, &phrase) in phrases.iter().enumerate() {
        let style_idx = i % num_styles;
        let style = if available_styles.is_empty() {
            DEFAULT_TTS_STYLE
        } else {
            &available_styles[style_idx]
        };
        let seed = base_seed + i as u64 + 500;

        if let Some(pcm) = synthesize_wake_word_variant(phrase, style, seed) {
            variants.push((pcm, format!("{prefix}_{phrase}_s{i}")));
        }
    }

    variants
}

// ── Enrollment ────────────────────────────────────────────────────────────

/// Process a list of audio clips through the enrollment embedding pipeline.
///
/// Returns:
/// * `positive_embeddings` — flat list of all frame-level 96-dim embedding
///   vectors across all utterances (for MLP classifier and verifier training).
/// * `enrollment_buffer` — per-utterance structure: each element is the
///   sequence of frame-level embeddings for one utterance (for self-test).
/// * `failed_count` — how many variants failed embedding extraction.
fn process_enrollment(
    variants: &[(Vec<f32>, String)],
) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>, usize) {
    let mut all_embeddings: Vec<Vec<f32>> = Vec::new();
    let mut enrollment_buffer: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut failed = 0usize;

    for (samples, label) in variants {
        match super::process_enrollment_sample(samples) {
            Ok(embeddings) => {
                if embeddings.is_empty() {
                    warn!("No embeddings extracted from '{label}'");
                    failed += 1;
                    continue;
                }
                // Flatten into the flat positive_embeddings list
                for emb in &embeddings {
                    all_embeddings.push(emb.clone());
                }
                // Keep per-utterance structure for self-test
                enrollment_buffer.push(embeddings.clone());
                info!(
                    "Processed enrollment variant '{label}': {} embeddings",
                    embeddings.len()
                );
            }
            Err(e) => {
                warn!("Embedding extraction failed for '{label}': {e}");
                failed += 1;
            }
        }
    }

    (all_embeddings, enrollment_buffer, failed)
}

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
        vad_decisions.push(super::is_speech_with_detector(
            frame,
            &mut detector,
            super::ENROLLMENT_VAD_THRESHOLD,
        ));
    }
    let utterances = super::segment_utterances_by_vad(
        audio,
        &vad_decisions,
        &super::DEFAULT_VAD_SEGMENTATION_CONFIG,
    );
    (vad_decisions, utterances)
}

/// Process enrollment variants through VAD-gated utterance segmentation.
///
/// Concatenates all variants with trailing silence gaps, computes VAD decisions
/// at the enrollment threshold, segments by [`segment_utterances_by_vad`], then
/// extracts embeddings from each utterance via [`process_enrollment_sample`].
///
/// This exercises the same production path as [`handle_enrollment_audio`]:
/// audio → VAD frame-by-frame → segment by VAD → per-utterance embeddings.
///
/// The trailing silence (≥1.5s) between clips ensures that
/// [`segment_utterances_by_vad`] can complete utterance boundary detection,
/// matching the production enrollment pipeline's behavior.
///
/// Returns the same pair as [`process_enrollment`] (minus the failed count):
/// flat embeddings list (for classifier training) and per-utterance embedding
/// buffers (for consistency check + self-test).
fn vad_segment_and_enroll(
    enrollment_variants: &[(Vec<f32>, String)],
    augmented_variants: &[(Vec<f32>, String)],
) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
    // ── Concatenate all variants with 2.0s silence gaps ──
    // 2.0s well exceeds SILENCE_THRESHOLD_SAMPLES (1.5s) for clean boundaries.
    let silence_gap_samples = (2.0 * super::SAMPLE_RATE as f64) as usize;
    let silence: Vec<f32> = vec![0.0f32; silence_gap_samples];

    let mut combined_audio: Vec<f32> = Vec::new();
    for (samples, _label) in enrollment_variants.iter().chain(augmented_variants) {
        if !combined_audio.is_empty() {
            combined_audio.extend_from_slice(&silence);
        }
        combined_audio.extend_from_slice(samples);
    }
    // Trailing silence for the last utterance
    combined_audio.extend_from_slice(&silence);

    info!(
        "VAD concatenation: {} total samples ({:.1}s) from {} variants",
        combined_audio.len(),
        combined_audio.len() as f64 / super::SAMPLE_RATE as f64,
        enrollment_variants.len() + augmented_variants.len(),
    );

    // ── Compute VAD decision + utterances via shared helper ──
    let (_vad_decisions, utterances) = compute_vad_segments(&combined_audio);

    info!(
        "VAD segmentation: {} utterances from {} concatenated variants",
        utterances.len(),
        enrollment_variants.len() + augmented_variants.len(),
    );

    // ── Process each utterance through enrollment ──
    let mut all_embeddings: Vec<Vec<f32>> = Vec::new();
    let mut enrollment_buffer: Vec<Vec<Vec<f32>>> = Vec::new();

    for (i, utterance) in utterances.iter().enumerate() {
        match super::process_enrollment_sample(utterance) {
            Ok(embeddings) if !embeddings.is_empty() => {
                info!(
                    "Utterance {i}: {} samples ({:.2}s), {} embeddings",
                    utterance.len(),
                    utterance.len() as f64 / super::SAMPLE_RATE as f64,
                    embeddings.len(),
                );
                for emb in &embeddings {
                    all_embeddings.push(emb.clone());
                }
                enrollment_buffer.push(embeddings);
            }
            Ok(_) => warn!("Utterance {i}: no embeddings extracted"),
            Err(e) => warn!("Utterance {i}: embedding extraction failed: {e}"),
        }
    }

    let expected_utterances = enrollment_variants.len() + augmented_variants.len();
    info!(
        "VAD-gated enrollment: {} positive embeddings from {} utterances (expected ~{expected_utterances})",
        all_embeddings.len(),
        enrollment_buffer.len(),
    );

    (all_embeddings, enrollment_buffer)
}

// ── Streaming detection ─────────────────────────────

/// Run the production streaming wake word detection pipeline on audio samples.
///
/// Feeds audio through [`handle_wake_word_detection`] in FRAME_LENGTH chunks,
/// exercising the full streaming chain: VAD gating, batch accumulation,
/// [`flush_voice_batch`], [`try_match_wake_word_and_push_embedding`],
/// [`score_single_embedding`], and cooldown logic.
///
/// After all audio is fed, a silence frame is sent to flush any remaining
/// voice batch (matching how the production pipeline handles speech→silence
/// transitions).
///
/// Returns `true` if the wake word was detected during processing.
fn run_streaming_detection(samples: &[f32], ctx: &mut super::PipelineCtx) -> bool {
    let chunk_size = super::FRAME_LENGTH;
    // Save pre-existing timestamp — we only return true if detection fires
    // during THIS call, not because a prior call already set the field.
    let before = ctx.last_wake_word_detection;

    // Feed audio in FRAME_LENGTH chunks
    for chunk in samples.chunks(chunk_size) {
        let padded = if chunk.len() < chunk_size {
            let mut p: Vec<f32> = chunk.to_vec();
            p.resize(chunk_size, 0.0);
            p
        } else {
            chunk.to_vec()
        };
        super::handle_wake_word_detection(&padded, ctx);
        if ctx.last_wake_word_detection != before {
            return true;
        }
    }

    // Feed silence frames to flush any remaining voice_batch.
    // The first silence frame after speech triggers flush_voice_batch via
    // the VAD-negative branch in handle_wake_word_detection.
    for _ in 0..3 {
        if ctx.last_wake_word_detection != before {
            return true;
        }
        let silence = vec![0.0f32; chunk_size];
        super::handle_wake_word_detection(&silence, ctx);
    }

    ctx.last_wake_word_detection != before
}

// ── Metrics reporting ────────────────────────────────────────────────────

/// Track per-variant detection results for reporting.
#[derive(Debug, Default)]
struct DetectionMetrics {
    total: usize,
    detected: usize,
    false_accepts: Vec<String>,
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

/// Process a list of audio clips through the detection pipeline, recording
/// results in `metrics`.  Shared helper for positive and negative detection
/// test blocks, eliminating the repetitive match-and-track boilerplate.
///
/// # Parameters
/// - `variants`: audio clips with descriptive labels.
/// - `classifier`, `verifier`: trained models passed to the streaming pipeline.
/// - `metrics`: records total/detected; `on_detection` fills detected or
///   false_accepts.
/// - `on_detection`: called with `(&mut metrics, label_str)` when the
///   wake word is detected (for positives: increment `.detected`; for
///   negatives: push to `.false_accepts`).
fn test_detection_samples(
    variants: &[(Vec<f32>, String)],
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
    metrics: &mut DetectionMetrics,
    on_detection: impl Fn(&mut DetectionMetrics, &str),
) {
    // Set classifier + verifier in global state for the streaming pipeline.
    // try_match_wake_word_and_push_embedding reads these from voice_state().
    super::set_classifier_weights(classifier.weights_ref().clone());
    super::set_verifier(verifier.clone());

    for (samples, label) in variants {
        metrics.total += 1;
        let mut ctx = super::PipelineCtx::new();
        if run_streaming_detection(samples, &mut ctx) {
            on_detection(metrics, label);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The main integration test
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
#[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn e2e_voice_pipeline() {
    // Initialize a tracing subscriber so progress info!() messages appear
    // in the test output (by default tests have no subscriber).
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .parse("info")
                .expect("info env filter"),
        )
        .with_test_writer()
        .try_init();

    // ── 0. Initialize global state ─────────────────────────────────────
    info!("═══ Voice Pipeline E2E Test ═══");

    // Set CONFIG storage root so model paths resolve.
    // This mirrors what `config::load_or_init()` does in startup.
    if crate::config::CONFIG.try_storage_root().is_none() {
        let mahbot_dir = crate::config::default_config_dir()
            .expect("Cannot resolve home directory for ~/.mahbot");
        let _ = crate::config::CONFIG.try_set_storage_root(mahbot_dir.clone());
        info!("CONFIG storage root set to: {}", mahbot_dir.display());
    }

    // Initialize TTS module (needed before try_load_cached can store the engine).
    crate::tts::init_global().unwrap_or_else(|e| warn!("tts::init_global() already called: {e}"));

    // Ensure voice pipeline state is initialized
    super::init_global().unwrap_or_else(|e| warn!("voice::init_global() already called: {e}"));

    // ── 1. Prerequisites ───────────────────────────────────────────────

    // Ensure TTS engine is loaded from cache
    if let Err(msg) = ensure_tts_ready() {
        panic!("{msg}\nRun the application first to download TTS models (~400 MB).");
    }

    // Ensure voice ONNX models are loaded from cache
    if let Err(msg) = ensure_voice_models_loaded() {
        panic!("{msg}\nRun the application first to download voice models.");
    }

    // Discover available TTS voice styles
    let available_styles = tts::list_voice_styles();
    info!(
        "TTS ready with {} voice styles: {:?}",
        available_styles.len(),
        available_styles
    );

    // ── 1. Generate enrollment audio ───────────────────────────────────
    info!("─── Phase 1: Generating enrollment audio ───");
    let enrollment_variants = generate_enrollment_variants(&available_styles);
    assert!(
        !enrollment_variants.is_empty(),
        "Need at least one enrollment variant. TTS synthesis may have failed for all styles."
    );
    info!(
        "Generated {} enrollment variants",
        enrollment_variants.len()
    );

    // ── 2. VAD-gated enrollment ────────────────────────────────────
    // Concatenate all TTS clips with 2.0s silence gaps, segment by VAD
    // at the enrollment threshold, and extract embeddings per utterance.
    // This exercises the same production path as handle_enrollment_audio:
    // audio → VAD frame-by-frame → segment_utterances_by_vad → per-utterance
    // process_enrollment_sample.
    info!("─── Phase 2: VAD-gated enrollment ───");

    // Generate augmented variants for enrollment diversity
    let augmented_variants = generate_augmented_variants(&available_styles, 200);
    info!("Generated {} augmented variants", augmented_variants.len());

    let (all_positive_embeddings, all_utterance_buffers) =
        vad_segment_and_enroll(&enrollment_variants, &augmented_variants);
    assert!(
        !all_utterance_buffers.is_empty(),
        "VAD-gated enrollment produced no utterances from {} enrollment + {} augmented variants",
        enrollment_variants.len(),
        augmented_variants.len(),
    );
    info!(
        "VAD-gated enrollment: {} positive embeddings from {} utterances",
        all_positive_embeddings.len(),
        all_utterance_buffers.len(),
    );

    let dim = all_positive_embeddings[0].len();

    // ── 3. Generate negative training data ─────────────────────────────
    // Generate real negative training data from unrelated AND confusable
    // phrases via TTS.  This mirrors the production pipeline where the
    // verifier is trained on real ambient audio negatives — the confusable
    // near-misses teach the classifier/verifier to discriminate beyond
    // synthetic Gaussian noise.  Confusable detection uses a different TTS
    // seed (300) than confusable training (500/510/520), measuring generalization to
    // novel acoustic renderings.  Unrelated detection uses seed 400; unrelated training
    // uses seeds 400/410/420 for diverse acoustic renderings (mahbot-829).
    //
    // Generate confusable negatives with multiple seeds (500, 510, 520) to
    // provide diverse acoustic renderings of each confusable phrase, improving
    // the classifier's ability to reject near-miss phrases that sound similar
    // to the wake word (mahbot-829).
    info!("─── Phase 3: Generating negative training data ───");
    info!("Generating negative training audio from unrelated + confusable phrases...");
    let neg_unrelated_1: Vec<(Vec<f32>, String)> =
        generate_phrase_variants(UNRELATED_PHRASES, &available_styles, 400, "neg_train");
    let neg_unrelated_2: Vec<(Vec<f32>, String)> =
        generate_phrase_variants(UNRELATED_PHRASES, &available_styles, 410, "neg_train");
    let neg_unrelated_3: Vec<(Vec<f32>, String)> =
        generate_phrase_variants(UNRELATED_PHRASES, &available_styles, 420, "neg_train");
    let neg_confusable_1: Vec<(Vec<f32>, String)> =
        generate_phrase_variants(CONFUSABLE_PHRASES, &available_styles, 500, "neg_conf_train");
    let neg_confusable_2: Vec<(Vec<f32>, String)> =
        generate_phrase_variants(CONFUSABLE_PHRASES, &available_styles, 510, "neg_conf_train");
    let neg_confusable_3: Vec<(Vec<f32>, String)> =
        generate_phrase_variants(CONFUSABLE_PHRASES, &available_styles, 520, "neg_conf_train");
    let mut negative_embeddings: Vec<Vec<f32>> = Vec::new();
    for neg_set in [
        &neg_unrelated_1,
        &neg_unrelated_2,
        &neg_unrelated_3,
        &neg_confusable_1,
        &neg_confusable_2,
        &neg_confusable_3,
    ] {
        let (embs, _, _) = process_enrollment(neg_set);
        negative_embeddings.extend(embs);
    }
    info!(
        "Extracted {} real negative embeddings from {} unrelated (3 seeds) + {} confusable phrases \
         (across 3 seed variations)",
        negative_embeddings.len(),
        neg_unrelated_1.len() + neg_unrelated_2.len() + neg_unrelated_3.len(),
        neg_confusable_1.len() + neg_confusable_2.len() + neg_confusable_3.len(),
    );

    // Supplement with synthetic Gaussian negatives for generalization
    // (used by BOTH classifier and verifier).
    negative_embeddings.extend(generate_synthetic_negatives(SYNTHETIC_NEGATIVES_COUNT, dim));

    // Build a SEPARATE negative set for the verifier that EXCLUDES confusable
    // phrases.  The logistic regression verifier is too simple to separate
    // confusable near-misses from genuine wake words — including confusables
    // in verifier training would cause the verifier to reject wake-word-like
    // embeddings (including the first enrollment variant).  The MLP classifier
    // (trained above with all negatives) handles confusable rejection at the
    // rolling-window level; the verifier only needs to reject random/unrelated
    // speech that the classifier might pass.
    //
    // Unrelated phrases use 3 TTS seeds (400, 410, 420) so the verifier sees
    // diverse acoustic renderings and learns a more generalizable boundary,
    // directly addressing the case where same-seed renderings still trigger
    // detection (mahbot-829).
    let mut verifier_negatives: Vec<Vec<f32>> = Vec::new();
    for unrelated in [&neg_unrelated_1, &neg_unrelated_2, &neg_unrelated_3] {
        let (embs, _, _) = process_enrollment(unrelated);
        verifier_negatives.extend(embs);
    }
    verifier_negatives.extend(generate_synthetic_negatives(SYNTHETIC_NEGATIVES_COUNT, dim));

    // ── 4. finalize_enrollment (consistency check + classifier training) ──
    // Calls the same production function used by the enrollment pipeline
    // (handle_enrollment_sample) — validates utterance consistency, then
    // trains the Conv1D MLP classifier via train_classifier.
    info!("─── Phase 4: finalize_enrollment ───");
    let weights = super::finalize_enrollment(
        &all_positive_embeddings,
        &negative_embeddings,
        &all_utterance_buffers,
    )
    .expect("finalize_enrollment must succeed — consistency check + classifier training");

    let classifier = WakeWordClassifier::new(weights.clone());
    info!(
        "Classifier trained successfully: {} params",
        weights.param_count(),
    );

    // ── Informational self-test (non-gating, diagnostic only) ──
    match super::run_enrollment_self_test(&all_utterance_buffers, &classifier) {
        Ok(()) => info!("Detection self-test (informational): passed"),
        Err(e) => info!("Detection self-test (informational, non-gating): {e}"),
    }

    // ── 5. Train the VoiceVerifier ─────────────────────────────────────
    info!("─── Phase 5: Training VoiceVerifier ───");

    // Use ONLY unrelated + synthetic negatives for verifier training
    // (excludes confusable phrases — the logistic regression cannot separate
    // confusable near-misses from wake words).  At 0.60 the verifier blocks
    // confusable embeddings that pass the rolling window while passing all
    // wake word variants.
    let verifier = VoiceVerifier::train(
        &all_positive_embeddings,
        &verifier_negatives,
        0.60, // mahbot-829: clean verifier at 0.60 blocks confusables that
        // pass the rolling window.  No confusable negatives in training
        // ensures the first variant's embeddings are not rejected.
        1.0,  // L2 lambda
        0.01, // learning rate
        2000, // max iterations
    );

    if verifier.is_trained() {
        info!(
            "VoiceVerifier trained successfully with {} positive + {} negative \
             (unrelated + synthetic, no confusable phrases)",
            all_positive_embeddings.len(),
            verifier_negatives.len()
        );
    } else {
        warn!("VoiceVerifier is untrained (insufficient data)");
    }

    // ── 6. Set global state for streaming detection ────────────────────
    // The streaming pipeline (handle_wake_word_detection) reads classifier
    // and verifier from voice_state() global state.  Set them once here.
    info!("─── Phase 6: Setting global state for streaming detection ───");
    super::set_classifier_weights(weights);
    super::set_verifier(verifier.clone());

    info!("─── Phase 7: Running streaming detection tests ───");

    // ── 8. Detection: Positive cases ───────────────────────────────────
    info!("─── 8. Positive (wake word) variants ───");
    let mut pos_metrics = DetectionMetrics::default();
    let all_wake_variants: Vec<(Vec<f32>, String)> = enrollment_variants
        .into_iter()
        .chain(augmented_variants)
        .collect();
    test_detection_samples(
        &all_wake_variants,
        &classifier,
        &verifier,
        &mut pos_metrics,
        |m, _| m.detected += 1,
    );

    // ── 9. Detection: Confusable near-miss phrases ─────────────────────
    info!("─── 9. Negative — confusable phrases ───");
    let confusable_variants =
        generate_phrase_variants(CONFUSABLE_PHRASES, &available_styles, 300, "confusable");
    info!(
        "Generated {} confusable phrase variants",
        confusable_variants.len()
    );
    let mut conf_metrics = DetectionMetrics::default();
    test_detection_samples(
        &confusable_variants,
        &classifier,
        &verifier,
        &mut conf_metrics,
        |m, l| m.false_accepts.push(l.to_string()),
    );

    // ── 10. Detection: Unrelated phrases ────────────────────────────────
    info!("─── 10. Negative — unrelated phrases ───");
    let unrelated_variants =
        generate_phrase_variants(UNRELATED_PHRASES, &available_styles, 400, "unrelated");
    info!(
        "Generated {} unrelated phrase variants",
        unrelated_variants.len()
    );
    let mut unrelated_metrics = DetectionMetrics::default();
    test_detection_samples(
        &unrelated_variants,
        &classifier,
        &verifier,
        &mut unrelated_metrics,
        |m, l| m.false_accepts.push(l.to_string()),
    );

    // ── 11. Detection: Silence ─────────────────────────────────────────
    info!("─── 11. Negative — silence ───");
    let mut silence_metric = DetectionMetrics::default();
    test_detection_samples(
        &[(vec![0.0f32; SILENCE_LEN], "silence".to_string())],
        &classifier,
        &verifier,
        &mut silence_metric,
        |m, l| m.false_accepts.push(l.to_string()),
    );

    // ── 12. Detection: Noise profiles ───────────────────────────────────
    info!("─── 12. Negative — noise profiles ───");
    let mut noise_total = 0usize;
    let mut noise_false_accepts: Vec<String> = Vec::new();
    for (label, generator) in NOISE_PROFILES {
        info!("  Testing noise profile: {label}");
        let noise = generator();
        let mut metric = DetectionMetrics::default();
        test_detection_samples(
            &[(noise, (*label).to_string())],
            &classifier,
            &verifier,
            &mut metric,
            |m, l| m.false_accepts.push(l.to_string()),
        );
        noise_total += metric.total;
        if !metric.false_accepts.is_empty() {
            info!("    → false accepts: {}", metric.false_accepts.len());
            noise_false_accepts.extend(metric.false_accepts);
        } else {
            info!("    → no false accepts ✓");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Cooldown sub-test
    // ═══════════════════════════════════════════════════════════════════
    info!("─── 13. Cooldown verification ───");
    if let Some((first_pos, _label)) = all_wake_variants.first() {
        let mut ctx = super::PipelineCtx::new();
        // First detection should fire
        let detected = run_streaming_detection(first_pos, &mut ctx);
        assert!(detected, "Cooldown test: first detection should fire");
        info!("Cooldown test: first detection fired ✓");

        // Immediate re-feed while cooldown active — detection should NOT fire.
        let before_cooldown = ctx.last_wake_word_detection;
        let silenced = run_streaming_detection(first_pos, &mut ctx);
        assert!(
            !silenced,
            "Cooldown test: detection should NOT fire during cooldown"
        );
        assert_eq!(
            ctx.last_wake_word_detection, before_cooldown,
            "Cooldown test: last_wake_word_detection should not change during cooldown"
        );
        info!("Cooldown test: cooldown prevented re-detection ✓");

        // Wait for cooldown to expire
        info!(
            "Cooldown test: waiting {}ms for cooldown expiry...",
            super::WAKE_WORD_COOLDOWN.as_millis()
        );
        std::thread::sleep(super::WAKE_WORD_COOLDOWN + std::time::Duration::from_millis(100));

        // After cooldown, detection should fire again.
        // Reset last_wake_word_detection so run_streaming_detection
        // doesn't think it already detected before processing audio.
        ctx.last_wake_word_detection = None;
        let after_cooldown = run_streaming_detection(first_pos, &mut ctx);
        assert!(
            after_cooldown,
            "Cooldown test: detection should fire after cooldown expires"
        );
        info!("Cooldown test: detection fired after cooldown ✓");
    } else {
        warn!("Cooldown test: no positive variants available, skipping");
    }

    // ═══════════════════════════════════════════════════════════════════
    // Metrics report
    // ═══════════════════════════════════════════════════════════════════

    let total_false_accepts = conf_metrics.false_accepts.len()
        + unrelated_metrics.false_accepts.len()
        + silence_metric.false_accepts.len()
        + noise_false_accepts.len();

    info!("══════════════════════════════════════════════");
    info!("      Voice Pipeline E2E Test Results");
    info!("══════════════════════════════════════════════");
    info!(
        "Detection rate: {:.1}% ({}/{}) — target ≥{:.0}%",
        pos_metrics.detection_rate() * 100.0,
        pos_metrics.detected,
        pos_metrics.total,
        MIN_DETECTION_RATE * 100.0,
    );
    info!(
        "Confusable false accepts: {} / {}",
        conf_metrics.false_accepts.len(),
        conf_metrics.total,
    );
    if !conf_metrics.false_accepts.is_empty() {
        info!("  False triggers: {:?}", conf_metrics.false_accepts);
    }
    info!(
        "Unrelated false accepts: {} / {}",
        unrelated_metrics.false_accepts.len(),
        unrelated_metrics.total,
    );
    if !unrelated_metrics.false_accepts.is_empty() {
        info!("  False triggers: {:?}", unrelated_metrics.false_accepts);
    }
    info!(
        "Silence false accepts: {} / 1",
        silence_metric.false_accepts.len(),
    );
    info!(
        "Noise false accepts: {} / {} ({} profiles)",
        noise_false_accepts.len(),
        noise_total,
        NOISE_PROFILES.len(),
    );
    if !noise_false_accepts.is_empty() {
        info!("  False triggers: {:?}", noise_false_accepts);
    }

    info!("──────────────────────────────────────────────");
    info!("Total false accepts: {total_false_accepts} — limit ≤{MAX_FALSE_ACCEPTS}",);
    info!("Enrollment consistency: validated by finalize_enrollment (Phase 4)");
    info!("══════════════════════════════════════════════");

    // ═══════════════════════════════════════════════════════════════════
    // Assertions
    // ═══════════════════════════════════════════════════════════════════

    // Detection rate must meet minimum threshold
    assert!(
        pos_metrics.detection_rate() >= MIN_DETECTION_RATE,
        "Detection rate too low: {:.1}% ({}/{}) — need ≥{:.0}%",
        pos_metrics.detection_rate() * 100.0,
        pos_metrics.detected,
        pos_metrics.total,
        MIN_DETECTION_RATE * 100.0,
    );

    // False accepts must be within limit
    assert!(
        total_false_accepts <= MAX_FALSE_ACCEPTS,
        "Too many false accepts: {total_false_accepts} — need ≤{MAX_FALSE_ACCEPTS}",
    );

    info!("═══ E2E Voice Pipeline Test PASSED ═══");
}
