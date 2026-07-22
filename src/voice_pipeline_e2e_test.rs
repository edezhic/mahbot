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
use crate::wake_word_classifier;
use crate::wake_word_classifier::{TrainingConfig, WakeWordClassifier};
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

/// Confusable near-miss phrases for negative detection testing.
const CONFUSABLE_PHRASES: &[&str] = &[
    "hey madbot",
    "hey map bot",
    "day mahbot",
    "hey nab it",
    "hey man",
];

/// Completely unrelated phrases for negative detection testing.
const UNRELATED_PHRASES: &[&str] = &[
    "the weather today is sunny",
    "what time is it",
    "one two three four five",
    "hello world",
    "good morning everyone",
];

/// Silence audio length in samples (1 second at 16 kHz).
const SILENCE_LEN: usize = 16_000;

/// Noise audio length in samples (1 second at 16 kHz).
const NOISE_LEN: usize = 16_000;

/// Number of synthetic negative embeddings to generate for classifier training.
const SYNTHETIC_NEGATIVE_COUNT: usize = 50;

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

/// Process enrollment variants through VAD gating + embedding extraction,
/// simulating the production enrollment pipeline.
///
/// For each variant: applies VAD at [`ENROLLMENT_VAD_THRESHOLD`], segments
/// utterances via [`segment_utterances_by_vad`], then extracts embeddings
/// via [`process_enrollment_sample`] from each VAD-gated utterance.
///
/// Returns the same pair as [`process_enrollment`] (minus the failed count):
/// flat embeddings list (for classifier training) and per-utterance embedding
/// buffers (for self-test).
fn process_vad_gated_enrollment(
    variants: &[(Vec<f32>, String)],
) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
    let mut all_embeddings: Vec<Vec<f32>> = Vec::new();
    let mut enrollment_buffer: Vec<Vec<Vec<f32>>> = Vec::new();

    for (samples, label) in variants {
        let (_vad_decisions, utterances) = compute_vad_segments(samples);

        if utterances.is_empty() {
            warn!("VAD gating detected no speech in variant '{label}'");
            continue;
        }

        for utterance in &utterances {
            match super::process_enrollment_sample(utterance) {
                Ok(embeddings) => {
                    if embeddings.is_empty() {
                        warn!("VAD-gated utterance '{label}' produced no embeddings");
                        continue;
                    }
                    let count = embeddings.len();
                    for emb in &embeddings {
                        all_embeddings.push(emb.clone());
                    }
                    enrollment_buffer.push(embeddings);
                    info!(
                        "VAD-gated '{label}': {count} embeddings from {} samples",
                        utterance.len(),
                    );
                }
                Err(e) => {
                    warn!("VAD-gated embedding failed for '{label}': {e}");
                    // Skip remaining utterances for this variant.
                    break;
                }
            }
        }
    }

    (all_embeddings, enrollment_buffer)
}

// ── Detection simulation ───────────────────────────

/// Simulate the live detection pipeline for a single audio clip's embeddings.
///
/// Delegates to the **production detection logic** via
/// [`super::score_single_embedding`] — the same function used by the live
/// pipeline ([`try_match_wake_word_and_push_embedding`]) and enrollment
/// self-test ([`run_enrollment_self_test`]).  Any changes to ring buffer
/// sizing, MLP window size, rolling sum threshold, or verifier gating are
/// automatically exercised by this integration test.
///
/// Returns `true` if the wake word was detected at any point in the clip.
fn simulate_detection(
    embeddings: &[Vec<f32>],
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
) -> bool {
    let mut embedding_ring: Vec<Vec<f32>> = Vec::with_capacity(super::EMBEDDING_RING_MAX);
    let mut score_window: Vec<f32> = Vec::new();
    for embedding in embeddings {
        if super::score_single_embedding(
            embedding,
            &mut embedding_ring,
            Some(classifier),
            Some(verifier),
            &mut score_window,
        ) {
            return true;
        }
    }
    false
}

// ── Metrics reporting ────────────────────────────────────────────────────

/// Track per-variant detection results for reporting.
#[derive(Debug, Default)]
struct DetectionMetrics {
    total: usize,
    detected: usize,
    false_accepts: Vec<String>,
    failures: Vec<String>, // Variants that couldn't be processed
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
/// - `classifier`, `verifier`: trained models (passed to `simulate_detection`).
/// - `metrics`: records total/failures; `on_detection` fills detected or
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
    for (samples, label) in variants {
        metrics.total += 1;
        match super::process_enrollment_sample(samples) {
            Ok(embeddings) if !embeddings.is_empty() => {
                if simulate_detection(&embeddings, classifier, verifier) {
                    on_detection(metrics, label);
                }
            }
            Ok(_) => {
                metrics.failures.push(format!("{label}: empty embeddings"));
            }
            Err(e) => {
                metrics.failures.push(format!("{label}: {e}"));
            }
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

    // ── 2. VAD-gated enrollment pipeline ───────────────────────────────
    // Process each enrollment variant through VAD gating at the enrollment
    // threshold, then extract embeddings from the gated utterances — same
    // flow as the production enrollment pipeline.  The VAD-gated data is
    // used for classifier training (Phase 4) and self-test (Phase 4b),
    // matching the production data path exactly.
    info!("─── Phase 2: VAD-gated enrollment embeddings ───");
    let (vad_positive_embeddings, vad_enrollment_buffer) =
        process_vad_gated_enrollment(&enrollment_variants);
    assert!(
        !vad_enrollment_buffer.is_empty(),
        "VAD gating produced no utterances from {variants} enrollment variants",
        variants = enrollment_variants.len(),
    );
    // Assert each VAD-gated utterance produces at least 3 embeddings.
    // With 76-frame windows at stride=8, 3 embeddings requires ~92 mel
    // frames (≈920ms of audio including context padding).  The lower
    // enrollment VAD threshold (0.60) captures ~75–85% of each utterance,
    // which with 200ms of context padding (100ms each side) produces
    // sufficient audio for ≥3 embeddings on a normal-speed wake word.
    // See `extract_embeddings_from_audio` for the stride/embedding math.
    for (i, utt_emb) in vad_enrollment_buffer.iter().enumerate() {
        assert!(
            utt_emb.len() >= 3,
            "VAD-gated utterance {i}: only {} embeddings (need ≥3). \
             Check that ENROLLMENT_VAD_THRESHOLD={threshold} captures \
             enough of each utterance, or that TTS audio is long enough.",
            utt_emb.len(),
            threshold = super::ENROLLMENT_VAD_THRESHOLD,
        );
    }
    info!(
        "VAD-gated enrollment: {} positive embeddings from {} utterances",
        vad_positive_embeddings.len(),
        vad_enrollment_buffer.len(),
    );

    // ── 2b. VAD gating: verify utterance segmentation (mahbot-823) ─────
    // Concatenate enrollment audio clips with silence gaps and process
    // through the extracted VAD gating function to verify that utterance
    // boundaries are correctly detected.  This exercises the same pure
    // function used by the production enrollment handler.
    info!("─── Phase 2b: VAD gating utterance segmentation ───");
    {
        // Concatenate enrollment clips with 1.8s of silence between them
        // (slightly longer than SILENCE_THRESHOLD_SAMPLES = 1.5s to ensure
        // clean boundary detection).
        let silence_gap_samples = (super::SILENCE_DURATION.as_millis() as usize + 300) as usize
            * super::SAMPLE_RATE as usize
            / 1000;
        let silence: Vec<f32> = vec![0.0f32; silence_gap_samples];
        let mut combined_audio: Vec<f32> = Vec::new();
        for (samples, _label) in &enrollment_variants {
            if !combined_audio.is_empty() {
                combined_audio.extend_from_slice(&silence);
            }
            combined_audio.extend_from_slice(samples);
        }

        // Use the shared helper to compute VAD segments.
        let (_vad_decisions, utterances) = compute_vad_segments(&combined_audio);

        let expected_count = enrollment_variants.len();
        info!(
            "VAD gating: {} utterances detected from {} enrollment clips",
            utterances.len(),
            expected_count,
        );

        // Assert the correct number of utterances was captured.
        // The TTS-generated enrollment clips are pre-trimmed speech
        // separated by 1.8s silence gaps, so the VAD gating should
        // detect exactly one utterance per clip.
        assert_eq!(
            utterances.len(),
            expected_count,
            "VAD gating: expected {expected_count} utterances from \
             {expected_count} enrollment clips with 1.8s silence gaps, \
             got {}",
            utterances.len(),
        );

        // Assert each utterance has adequate audio length for embedding.
        // The 400ms floor (ENROLLMENT_QUALITY_DURATION_MIN_MS) ensures VAD
        // gating captured real speech rather than noise blips.  Phase 2's
        // ≥3-embedding assertion validates production adequacy separately;
        // see Phase 2 comment for stride/embedding math.
        let min_audio_len =
            super::ENROLLMENT_QUALITY_DURATION_MIN_MS as usize * super::SAMPLE_RATE as usize / 1000;
        for (i, utt) in utterances.iter().enumerate() {
            let duration_ms = (utt.len() as u64 * 1000) / u64::from(super::SAMPLE_RATE);
            assert!(
                utt.len() >= min_audio_len,
                "Utterance {i} too short: {} samples ({duration_ms}ms) \
                 — need ≥{min_audio_len} samples ({}ms)",
                utt.len(),
                super::ENROLLMENT_QUALITY_DURATION_MIN_MS,
            );
            info!(
                "  Utterance {i}: {} samples ({duration_ms}ms) — OK",
                utt.len(),
            );
        }
    }

    // ── 3. Generate augmented wake word variants ───────────────────────
    info!("─── Phase 3: Generating augmented wake word audio ───");
    let augmented_variants = generate_augmented_variants(&available_styles, 200);
    info!("Generated {} augmented variants", augmented_variants.len());

    // Process augmented variants through VAD-gated pipeline to match
    // the production data path (mahbot-824).
    let (vad_augmented_embeddings, vad_augmented_enrollment_buffer) =
        process_vad_gated_enrollment(&augmented_variants);
    info!(
        "Extracted {} VAD-gated augmented embeddings from {} variants",
        vad_augmented_embeddings.len(),
        augmented_variants.len(),
    );

    // Merge VAD-gated enrollment + augmentation embeddings for training
    // Uses VAD-gated data from Phase 2 for the production-accurate path.
    let mut all_positive_embeddings = vad_positive_embeddings; // move
    all_positive_embeddings.extend(vad_augmented_embeddings);
    let mut all_utterance_buffers = vad_enrollment_buffer; // move
    all_utterance_buffers.extend(vad_augmented_enrollment_buffer);

    info!(
        "Combined training set: {} positive embeddings from {} utterances (VAD-gated)",
        all_positive_embeddings.len(),
        all_utterance_buffers.len(),
    );

    // ── 4. Train the MLP classifier ────────────────────────────────────
    info!("─── Phase 4: Training MLP classifier ───");

    let dim = all_positive_embeddings[0].len();

    // Generate real negative training data from unrelated phrases via TTS.
    // This mirrors the production pipeline where the verifier is trained on
    // real ambient audio negatives rather than synthetic-only (Mahbot-822).
    info!("Generating negative training audio from unrelated phrases...");
    let neg_train_variants: Vec<(Vec<f32>, String)> =
        generate_phrase_variants(UNRELATED_PHRASES, &available_styles, 400, "neg_train");
    let (mut real_neg_embeddings, _, _) = process_enrollment(&neg_train_variants);
    info!(
        "Extracted {} real negative embeddings from {} phrases",
        real_neg_embeddings.len(),
        neg_train_variants.len()
    );

    // Supplement with synthetic Gaussian negatives for generalization
    real_neg_embeddings.extend(generate_synthetic_negatives(SYNTHETIC_NEGATIVE_COUNT, dim));
    let negative_embeddings = real_neg_embeddings;

    let config = TrainingConfig::default();
    let weights = wake_word_classifier::train_classifier(
        &all_positive_embeddings,
        &negative_embeddings,
        &config,
    )
    .expect("Classifier training must succeed");

    let classifier = WakeWordClassifier::new(weights.clone());

    // ── Enrollment consistency check ──
    info!("─── Phase 4b: Enrollment consistency check ───");
    let consistency_ok = super::validate_enrollment_consistency(&all_utterance_buffers);
    assert!(
        consistency_ok.is_ok(),
        "Enrollment consistency check failed: {:?}",
        consistency_ok,
    );

    // ── Informational self-test (non-gating, diagnostic only) ──
    match super::run_enrollment_self_test(&all_utterance_buffers, &classifier) {
        Ok(()) => info!("Detection self-test (informational): passed"),
        Err(e) => info!("Detection self-test (informational, non-gating): {e}"),
    }

    // ── 5. Train the VoiceVerifier ─────────────────────────────────────
    info!("─── Phase 5: Training VoiceVerifier ───");

    // Use the same mixed (real + synthetic) negatives for verifier training.
    let verifier = VoiceVerifier::train(
        &all_positive_embeddings,
        &negative_embeddings,
        0.5,  // standard logistic regression boundary
        1.0,  // L2 lambda
        0.01, // learning rate
        2000, // max iterations
    );

    if verifier.is_trained() {
        info!(
            "VoiceVerifier trained successfully with {} positive + {} negative",
            all_positive_embeddings.len(),
            negative_embeddings.len()
        );
    } else {
        warn!("VoiceVerifier is untrained (insufficient data)");
    }

    info!("─── Phase 6: Running detection tests ───");

    // ── 7. Detection: Positive cases ───────────────────────────────────
    // NOTE: Uses `process_enrollment_sample` (stride=8), which produces
    // ~8× fewer embeddings per utterance than the live pipeline's per-frame
    // embedding extraction.  This is a known blind spot: detection that
    // depends on specific window alignments may differ in production.
    info!("─── 7. Positive (wake word) variants ───");
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

    // ── 8. Detection: Confusable near-miss phrases ─────────────────────
    info!("─── 8. Negative — confusable phrases ───");
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

    // ── 9. Detection: Unrelated phrases ────────────────────────────────
    info!("─── 9. Negative — unrelated phrases ───");
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

    // ── 10. Detection: Silence ─────────────────────────────────────────
    info!("─── 10. Negative — silence ───");
    let mut silence_metric = DetectionMetrics::default();
    test_detection_samples(
        &[(vec![0.0f32; SILENCE_LEN], "silence".to_string())],
        &classifier,
        &verifier,
        &mut silence_metric,
        |m, l| m.false_accepts.push(l.to_string()),
    );

    // ── 11. Detection: Random noise ────────────────────────────────────
    info!("─── 11. Negative — random noise ───");
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let noise: Vec<f32> = (0..NOISE_LEN)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    let mut noise_metric = DetectionMetrics::default();
    test_detection_samples(
        &[(noise, "random noise".to_string())],
        &classifier,
        &verifier,
        &mut noise_metric,
        |m, l| m.false_accepts.push(l.to_string()),
    );

    // ═══════════════════════════════════════════════════════════════════
    // Metrics report
    // ═══════════════════════════════════════════════════════════════════

    let total_false_accepts = conf_metrics.false_accepts.len()
        + unrelated_metrics.false_accepts.len()
        + silence_metric.false_accepts.len()
        + noise_metric.false_accepts.len();

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
    if !pos_metrics.failures.is_empty() {
        info!("  Processing failures: {}", pos_metrics.failures.join(", "));
    }

    info!(
        "Confusable false accepts: {} / {}",
        conf_metrics.false_accepts.len(),
        conf_metrics.total,
    );
    if !conf_metrics.false_accepts.is_empty() {
        info!("  False triggers: {:?}", conf_metrics.false_accepts);
    }
    if !conf_metrics.failures.is_empty() {
        info!("  Failures: {}", conf_metrics.failures.join(", "));
    }

    info!(
        "Unrelated false accepts: {} / {}",
        unrelated_metrics.false_accepts.len(),
        unrelated_metrics.total,
    );
    if !unrelated_metrics.false_accepts.is_empty() {
        info!("  False triggers: {:?}", unrelated_metrics.false_accepts);
    }
    if !unrelated_metrics.failures.is_empty() {
        info!("  Failures: {}", unrelated_metrics.failures.join(", "));
    }

    info!(
        "Silence false accepts: {} / 1",
        silence_metric.false_accepts.len(),
    );
    info!(
        "Noise false accepts: {} / 1",
        noise_metric.false_accepts.len(),
    );

    info!("──────────────────────────────────────────────");
    info!("Total false accepts: {total_false_accepts} — limit ≤{MAX_FALSE_ACCEPTS}",);
    info!(
        "Enrollment self-test: {}",
        match &self_test_ok {
            Ok(()) => "PASSED",
            Err(msg) => msg.as_str(),
        }
    );
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
