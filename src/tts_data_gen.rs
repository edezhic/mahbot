//! Training data generation for wake word classifier.
//!
//! This module provides utilities for generating synthetic training data
//! for the wake word classifier (mahbot-810). It uses the TTS engine
//! to synthesize wake word utterances with diverse voice styles, then
//! applies audio augmentation (noise, volume, speed) before extracting
//! 96-dim embedding vectors via the voice pipeline's mel spectrogram +
//! embedding models.
//!
//! # Reproducibility scope (mahbot-906)
//!
//! The benchmark path ([`generate_augmented_variants`] in
//! `voice_pipeline_e2e_test.rs`) uses seeded RNG via the `rng_seed:
//! Option<u64>` parameter on [`randomize_volume`] and [`add_noise`]
//! for deterministic augmentation. The production path
//! ([`augment_audio`], [`generate_training_data`]) passes `None` and
//! remains non-deterministic — this is intentional, as production batch
//! offline training data generation does not require reproducibility.
//! Only the benchmark's `randomize_volume` and `add_noise` calls were
//! modified; the 8 internal `rand::*` calls in `augment_audio` and 5
//! in `generate_training_data` are unchanged.
//!
//! # Pipeline
//!
//! 1. Synthesize audio directly at 16 kHz via [`crate::tts::synthesize()`]
//! 2. Apply augmentation: noise, volume, speed perturbation
//! 3. Run through mel spectrogram → embedding model → 96-dim vectors
//! 4. Save labeled vectors to `~/.mahbot/training/`
//!
//! # Performance
//!
//! Each synthesis takes ~3-5 seconds on CPU. 1000 samples ≈ 50-80 minutes.
//! This is a batch offline process — not real-time.

use crate::voice;
use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::path::PathBuf;
use tracing::info;

// ── Constants ────────────────────────────────────────────────────────

/// Available noise types for augmentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseType {
    /// White noise (uniform spectral density).
    White,
    /// Pink noise (1/f spectral density — more natural).
    Pink,
}

/// Confusable phrases — phonetic near-misses of the default wake word.
/// These are used to generate negative training examples.
///
/// Uses the canonical list from [`crate::voice::CONFUSABLE_PHRASES`] (mahbot-859).
const CONFUSABLE_PHRASES: &[&str] = crate::voice::CONFUSABLE_PHRASES;

/// Training data directory name under storage root.
const TRAINING_DIR: &str = "training";

// ── Audio post-processing ────────────────────────────────────────────

/// Mix white or pink noise into audio at a given SNR (dB).
///
/// When `rng_seed` is `Some(seed)`, uses a deterministic seeded RNG for
/// reproducible output. When `None`, uses an entropy-seeded RNG (current
/// non-deterministic behavior). The deterministic path is used by the
/// benchmark to ensure stable training data across re-runs (mahbot-906).
///
/// # Arguments
///
/// * `samples` — Clean audio PCM f32 in [-1.0, 1.0].
/// * `snr_db` — Desired signal-to-noise ratio in dB (typical: 10-25).
///   Lower values = more noise. Must be finite.
/// * `noise_type` — [`NoiseType::White`] or [`NoiseType::Pink`].
/// * `rng_seed` — Optional seed for deterministic RNG. `Some(seed)` produces
///   reproducible noise; `None` uses entropy-based seeding.
///
/// # Returns
///
/// Noisy audio PCM f32 in [-1.0, 1.0] (clamped).
#[must_use]
pub fn add_noise(
    samples: &[f32],
    snr_db: f32,
    noise_type: NoiseType,
    rng_seed: Option<u64>,
) -> Vec<f32> {
    if samples.is_empty() || !snr_db.is_finite() {
        return samples.to_vec();
    }

    // Create a seeded or entropy-based RNG, matching the canonical
    // Option<u64> → StdRng pattern (voice_verifier.rs, mahbot-906).
    // When None, we seed from rand::random() rather than using the
    // thread-local generator directly, isolating RNG state.
    let mut rng: StdRng = match rng_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::seed_from_u64(rand::random()),
    };

    // Generate noise using the single RNG for both white and pink branches.
    let noise: Vec<f32> = match noise_type {
        NoiseType::White => {
            // Uniform white noise in [-1.0, 1.0]
            (0..samples.len())
                .map(|_| rng.random::<f32>() * 2.0 - 1.0)
                .collect()
        }
        NoiseType::Pink => {
            // Voss-McCartney pink noise via the canonical seeded implementation
            crate::util::generate_pink_noise(samples.len(), &mut rng)
        }
    };

    // Compute RMS of signal and noise
    let signal_rms = compute_rms(samples);
    let noise_rms = compute_rms(&noise);

    if signal_rms <= 1e-10 || noise_rms <= 1e-10 {
        return samples.to_vec(); // Degenerate case — no scaling
    }

    // SNR = 20 * log10(signal_rms / noise_rms * scale)
    // scale = signal_rms / noise_rms * 10^(-SNR/20)
    let scale = (signal_rms / noise_rms) * 10.0_f32.powf(-snr_db / 20.0);

    // Mix
    let mut result = Vec::with_capacity(samples.len());
    for (&s, &n) in samples.iter().zip(noise.iter()) {
        result.push((s + n * scale).clamp(-1.0, 1.0));
    }
    result
}

/// Compute the RMS (root mean square) of audio samples.
#[allow(clippy::cast_precision_loss)]
fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Apply a random volume gain within ±`max_gain_db` dB.
///
/// When `rng_seed` is `Some(seed)`, uses a deterministic seeded RNG for
/// reproducible gain values. When `None`, uses an entropy-seeded RNG (current
/// non-deterministic behavior). The deterministic path is used by the
/// benchmark to ensure stable training data across re-runs (mahbot-906).
///
/// Each call produces a different random gain in the range
/// `[-max_gain_db, +max_gain_db]` dB, applied uniformly to all samples.
///
/// # Arguments
///
/// * `samples` — Audio PCM f32 in [-1.0, 1.0].
/// * `max_gain_db` — Maximum absolute gain in dB (e.g., 6.0 for ±6 dB).
/// * `rng_seed` — Optional seed for deterministic RNG. `Some(seed)` produces
///   reproducible gain; `None` uses entropy-based seeding.
///
/// # Returns
///
/// Gain-adjusted audio PCM f32 in [-1.0, 1.0] (clamped).
#[must_use]
pub fn randomize_volume(samples: &[f32], max_gain_db: f32, rng_seed: Option<u64>) -> Vec<f32> {
    if samples.is_empty() || max_gain_db <= 0.0 {
        return samples.to_vec();
    }

    // Create a seeded or entropy-based RNG, matching the canonical
    // Option<u64> → StdRng pattern (voice_verifier.rs, mahbot-906).
    let mut rng: StdRng = match rng_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::seed_from_u64(rand::random()),
    };

    // Random gain in [-max_gain_db, max_gain_db] dB
    let gain_db: f32 = (rng.random::<f32>() * 2.0 - 1.0) * max_gain_db;
    let linear_gain = 10.0_f32.powf(gain_db / 20.0);

    samples
        .iter()
        .map(|&s| (s * linear_gain).clamp(-1.0, 1.0))
        .collect()
}

/// Apply speed perturbation by resampling.
///
/// Changes both speed and pitch (time-domain resampling). For wake word
/// training data diversity, this is acceptable and does not require
/// pitch-preserving time-stretching.
///
/// # Arguments
///
/// * `samples` — Audio PCM f32 at `sample_rate`.
/// * `sample_rate` — Original sample rate in Hz.
/// * `factor` — Speed factor: >1.0 = faster (fewer samples), <1.0 = slower
///   (more samples). Typical range: 0.8-1.2 (±20%).
///
/// # Returns
///
/// Speed-adjusted audio at the original `sample_rate`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn speed_perturbation(samples: &[f32], sample_rate: u32, factor: f32) -> Vec<f32> {
    if samples.is_empty() || (factor - 1.0).abs() < 1e-6 {
        return samples.to_vec();
    }
    // Speed perturbation via resampling: change the effective rate
    // new_rate = sample_rate * factor
    let effective_rate = (sample_rate as f32 * factor) as u32;
    // Resample from effective_rate back to sample_rate
    // This produces the same duration as original but with shifted pitch
    crate::util::resample_audio(samples, effective_rate, sample_rate)
}

// ── Data generation ──────────────────────────────────────────────────

/// Sample rate used for all synthesized training audio (voice pipeline rate).
const TRAINING_SAMPLE_RATE: u32 = 16_000;

/// A large prime used as an LCG multiplier to mix style and iteration into
/// a unique seed for each synthesis call. 6_364_136_223_846_793_005 is a
/// well-known LCG multiplier (from Musl's `__rand48_step`) that produces
/// good avalanche behavior across 64-bit seeds.
const SEED_MIX_PRIME: u64 = 6_364_136_223_846_793_005;

/// Helper: synthesize text, apply augmentation, and extract embeddings.
///
/// Pushes up to `max_count` embedding vectors into `results`. Uses
/// `results.len()` directly for count tracking (no separate counter
/// parameter) to avoid the fragile [`Vec::len`] temporary-reference pattern.
fn synthesize_and_extract(
    text: &str,
    style: &str,
    seed: u64,
    snr_db_range: (f32, f32),
    max_count: usize,
    results: &mut Vec<Vec<f32>>,
) {
    let pcm = match crate::tts::synthesize(text, style, seed, TRAINING_SAMPLE_RATE) {
        Ok(pcm) => pcm,
        Err(e) => {
            tracing::warn!("Synthesis failed for '{text}' with {style}: {e}");
            return;
        }
    };

    let augmented = augment_audio(&pcm, TRAINING_SAMPLE_RATE, snr_db_range);

    match voice::process_enrollment_sample(&augmented) {
        Ok(embeddings) => {
            for emb in &embeddings {
                if results.len() >= max_count {
                    break;
                }
                results.push(emb.clone());
            }
        }
        Err(e) => {
            tracing::warn!("Embedding extraction failed for '{text}' with {style}: {e}");
        }
    }
}

/// Verify models are ready and return the training directory path.
fn prepare_training_env() -> Result<PathBuf> {
    if !crate::tts::models_ready() {
        anyhow::bail!("TTS engine not ready — models must be loaded first");
    }
    if !voice::models_ready() {
        anyhow::bail!("Voice pipeline models not ready — call voice::init_global() first");
    }
    let storage_root = crate::config::CONFIG
        .try_storage_root()
        .context("Cannot resolve storage root")?;
    let training_dir = storage_root.join(TRAINING_DIR);
    std::fs::create_dir_all(&training_dir)
        .with_context(|| format!("Failed to create {}", training_dir.display()))?;
    Ok(training_dir)
}

/// Generate training data for wake word classifier.
///
/// This function:
/// 1. Synthesizes `num_positive` variants of the wake word text using
///    all available TTS voice styles.
/// 2. Synthesizes `num_negative` variants of confusable phrases.
/// 3. Applies random audio augmentation to each sample.
/// 4. Runs each sample through the mel spectrogram + embedding model to
///    produce 96-dim embedding vectors.
/// 5. Saves the labeled embeddings to `~/.mahbot/training/`.
///
/// # Arguments
///
/// * `wake_word` — The wake word text to synthesize (e.g., `"hey mahbot"`).
/// * `num_positive` — Target number of positive (wake word) examples.
///   The function will oversample by cycling voice styles as needed.
/// * `num_negative` — Target number of negative (non-wake word) examples.
/// * `snr_db_range` — (min, max) SNR for noise injection, e.g. `(10.0, 25.0)`.
///
/// # Returns
///
/// A tuple `(positive_count, negative_count, output_dir)` — the actual
/// number of positive and negative embeddings saved, and the output
/// directory path.
///
/// # Integration with mahbot-810
///
/// This function is designed to be called from the classifier training
/// pipeline. The output directory (`~/.mahbot/training/`) contains:
/// - `positive_embeddings.bin` — serialized `Vec<Vec<f32>>` (rows = 96-dim vectors)
/// - `negative_embeddings.bin` — serialized `Vec<Vec<f32>>`
/// - `labels.txt` — one label per row
///
/// # Safety
///
/// This function has a safety backstop: if too many synthesis iterations
/// elapse without making progress (e.g., all `process_enrollment_sample`
/// calls return zero embeddings), the loop terminates with whatever has
/// been collected so far. This prevents infinite loops in edge cases.
///
/// # Errors
///
/// Returns an error if TTS engine or voice models are not ready, or if
/// the output directory cannot be created.
#[allow(clippy::too_many_lines)]
pub fn generate_training_data(
    wake_word: &str,
    num_positive: usize,
    num_negative: usize,
    snr_db_range: (f32, f32),
) -> Result<(usize, usize, PathBuf)> {
    let training_dir = prepare_training_env()?;
    let available_styles = crate::tts::list_voice_styles();
    if available_styles.is_empty() {
        anyhow::bail!("No voice styles available — TTS models may not be fully downloaded");
    }

    let mut positive_embeddings: Vec<Vec<f32>> = Vec::new();
    let mut negative_embeddings: Vec<Vec<f32>> = Vec::new();

    // ── Generate positive examples ────────────────────────────────────
    // Cycle through voice styles and text variations.
    // Safety backstop: if `process_enrollment_sample` returns zero embedding
    // vectors, the loop could spin forever. We cap iterations at 3× target
    // to prevent this, since at most ~20% of samples fail in practice.
    let text_variants = generate_text_variants(wake_word);
    let mut style_idx = 0;
    let max_positive_iterations = num_positive.saturating_mul(3);
    let mut positive_iterations = 0;

    while positive_embeddings.len() < num_positive && positive_iterations < max_positive_iterations
    {
        positive_iterations += 1;
        let style = &available_styles[style_idx % available_styles.len()];
        style_idx += 1;

        let text = &text_variants[rand::random_range(0..text_variants.len())];

        // Mix style index and count into seed using LCG prime multiplier
        // so different voice styles + iterations produce diverse prosody.
        let seed = (style_idx as u64)
            .wrapping_mul(SEED_MIX_PRIME)
            .wrapping_add(positive_embeddings.len() as u64);

        synthesize_and_extract(
            text,
            style,
            seed,
            snr_db_range,
            num_positive,
            &mut positive_embeddings,
        );
    }

    // ── Generate negative examples ────────────────────────────────────
    // Use confusable phrases, unrelated speech, and silence.
    style_idx = 0;
    let max_negative_iterations = num_negative.saturating_mul(3);
    let mut negative_iterations = 0;

    while negative_embeddings.len() < num_negative && negative_iterations < max_negative_iterations
    {
        negative_iterations += 1;
        let style = &available_styles[style_idx % available_styles.len()];
        style_idx += 1;

        let seed = (style_idx as u64)
            .wrapping_mul(SEED_MIX_PRIME)
            .wrapping_add(negative_embeddings.len() as u64);

        if rand::random_bool(0.1) {
            // 10% chance: generate ambient noise as a negative example.
            // Simulates room noise floor or non-speech audio. The noise level
            // varies randomly so the model sees diverse low-energy inputs.
            let noise_level: f32 = 0.01 + rand::random::<f32>() * 0.2; // -40 dBFS to -14 dBFS
            let samples: Vec<f32> = (0..TRAINING_SAMPLE_RATE as usize)
                .map(|_| (rand::random::<f32>() * 2.0 - 1.0) * noise_level)
                .collect();
            let augmented = augment_audio(&samples, TRAINING_SAMPLE_RATE, snr_db_range);
            match voice::process_enrollment_sample(&augmented) {
                Ok(embeddings) => {
                    for emb in &embeddings {
                        if negative_embeddings.len() >= num_negative {
                            break;
                        }
                        negative_embeddings.push(emb.clone());
                    }
                }
                Err(e) => {
                    tracing::warn!("Noise embedding extraction failed: {e}");
                }
            }
        } else {
            let phrase = CONFUSABLE_PHRASES[rand::random_range(0..CONFUSABLE_PHRASES.len())];
            synthesize_and_extract(
                phrase,
                style,
                seed,
                snr_db_range,
                num_negative,
                &mut negative_embeddings,
            );
        }
    }

    // ── Save to disk ──────────────────────────────────────────────────
    save_training_outputs(
        &training_dir,
        wake_word,
        &available_styles,
        snr_db_range,
        &positive_embeddings,
        &negative_embeddings,
    )?;

    info!(
        "Generated training data: {} positive, {} negative embeddings in {}",
        positive_embeddings.len(),
        negative_embeddings.len(),
        training_dir.display()
    );

    Ok((
        positive_embeddings.len(),
        negative_embeddings.len(),
        training_dir,
    ))
}

/// Write embedding files, labels, and metadata to disk.
fn save_training_outputs(
    dir: &std::path::Path,
    wake_word: &str,
    styles: &[String],
    snr_db_range: (f32, f32),
    positive_embeddings: &[Vec<f32>],
    negative_embeddings: &[Vec<f32>],
) -> Result<()> {
    save_embeddings(dir, "positive_embeddings.bin", positive_embeddings)?;
    save_embeddings(dir, "negative_embeddings.bin", negative_embeddings)?;

    // Write labels file
    let labels_path = dir.join("labels.txt");
    {
        let mut labels = String::new();
        for _ in positive_embeddings {
            labels.push_str("positive\n");
        }
        for _ in negative_embeddings {
            labels.push_str("negative\n");
        }
        std::fs::write(&labels_path, &labels)
            .with_context(|| format!("Failed to write labels to {}", labels_path.display()))?;
    }

    // Write metadata JSON
    let metadata_path = dir.join("metadata.json");
    {
        let metadata = serde_json::json!({
            "wake_word": wake_word,
            "num_positive": positive_embeddings.len(),
            "num_negative": negative_embeddings.len(),
            "voice_styles_used": styles,
            "confusable_phrases": CONFUSABLE_PHRASES,
            "snr_db_range": [snr_db_range.0, snr_db_range.1],
            "embedding_dim": 96,
        });
        let json_str = serde_json::to_string_pretty(&metadata)?;
        std::fs::write(&metadata_path, &json_str)
            .with_context(|| format!("Failed to write metadata to {}", metadata_path.display()))?;
    }

    Ok(())
}

/// Generate text variants of the wake word (capitalization, punctuation).
fn generate_text_variants(base: &str) -> Vec<String> {
    let mut variants = Vec::new();

    // Original
    variants.push(base.to_string());

    // Capitalized
    let capitalized = {
        let mut chars = base.chars();
        if let Some(c) = chars.next() {
            format!("{}{}", c.to_uppercase(), chars.as_str())
        } else {
            base.to_string()
        }
    };
    variants.push(capitalized);

    // All uppercase
    variants.push(base.to_uppercase());

    // With exclamation
    variants.push(format!("{base}!"));

    // With period
    variants.push(format!("{base}."));

    // With question mark
    variants.push(format!("{base}?"));

    // Reduplicated
    variants.push(format!("{base} {base}"));

    variants
}

/// Apply audio augmentation: noise injection, volume randomization, speed perturbation.
fn augment_audio(samples: &[f32], sample_rate: u32, snr_db_range: (f32, f32)) -> Vec<f32> {
    if samples.is_empty() {
        return samples.to_vec();
    }

    let mut audio = samples.to_vec();

    // 1. Noise injection (50% probability)
    // Note: the internal RNG calls in this function (noise type, speed
    // perturbation probability, SNR) remain unseeded because this is
    // production batch code where reproducibility is not critical. Only
    // the benchmark path (generate_augmented_variants) uses seeded RNG
    // (mahbot-906).
    if rand::random_bool(0.5) {
        let snr_db: f32 =
            snr_db_range.0 + rand::random::<f32>() * (snr_db_range.1 - snr_db_range.0);
        let noise_type = if rand::random_bool(0.5) {
            NoiseType::White
        } else {
            NoiseType::Pink
        };
        audio = add_noise(&audio, snr_db, noise_type, None);
    }

    // 2. Volume randomization (always applied with ±6 dB max)
    audio = randomize_volume(&audio, 6.0, None);

    // 3. Speed perturbation (40% probability, ±10-20%)
    if rand::random_bool(0.4) {
        let factor = if rand::random_bool(0.5) {
            // Slower: 0.8-1.0
            rand::random_range(0.8..1.0)
        } else {
            // Faster: 1.0-1.2
            rand::random_range(1.0..1.2)
        };
        audio = speed_perturbation(&audio, sample_rate, factor);
    }

    audio
}

/// Save embeddings to a binary file.
///
/// Format: 4 bytes for number of vectors (u32 LE), then for each vector:
/// 4 bytes for vector length (u32 LE), then `len * 4` bytes of f32 LE data.
fn save_embeddings(dir: &std::path::Path, filename: &str, embeddings: &[Vec<f32>]) -> Result<()> {
    let path = dir.join(filename);
    let mut data = Vec::with_capacity(embeddings.len() * (4 + 4 + 96 * 4));

    // Number of vectors (u32 LE)
    let num_vectors =
        u32::try_from(embeddings.len()).expect("number of embeddings must fit in u32");
    data.extend_from_slice(&num_vectors.to_le_bytes());

    for emb in embeddings {
        // Vector length (u32 LE)
        let emb_len = u32::try_from(emb.len()).expect("embedding length must fit in u32");
        data.extend_from_slice(&emb_len.to_le_bytes());
        // Vector data (f32 LE)
        for &val in emb {
            data.extend_from_slice(&val.to_le_bytes());
        }
    }

    std::fs::write(&path, &data)
        .with_context(|| format!("Failed to write embeddings to {}", path.display()))?;

    info!(
        "Saved {} embeddings ({} bytes) to {}",
        embeddings.len(),
        data.len(),
        path.display()
    );

    Ok(())
}

/// Load embeddings from a binary file saved by [`save_embeddings`].
///
/// Returns the number of vectors and the vectors themselves.
pub fn load_embeddings(dir: &std::path::Path, filename: &str) -> Result<Vec<Vec<f32>>> {
    let path = dir.join(filename);
    let data = std::fs::read(&path)
        .with_context(|| format!("Failed to read embeddings from {}", path.display()))?;

    if data.len() < 4 {
        anyhow::bail!("File too small: {} bytes", data.len());
    }

    let num_vectors = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let mut offset = 4;
    let mut embeddings = Vec::with_capacity(num_vectors);

    for _ in 0..num_vectors {
        if offset + 4 > data.len() {
            anyhow::bail!("Truncated file: expected vector length at offset {offset}");
        }
        let vec_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        let end = offset + vec_len * 4;
        if end > data.len() {
            anyhow::bail!("Truncated file: expected {vec_len} f32 values at offset {offset}");
        }

        let mut emb = Vec::with_capacity(vec_len);
        for i in 0..vec_len {
            let start = offset + i * 4;
            let val = f32::from_le_bytes([
                data[start],
                data[start + 1],
                data[start + 2],
                data[start + 3],
            ]);
            emb.push(val);
        }
        embeddings.push(emb);
        offset = end;
    }

    Ok(embeddings)
}

/// Get the path to the training data directory.
#[must_use]
pub fn training_dir() -> Option<PathBuf> {
    Some(crate::config::CONFIG.try_storage_root()?.join(TRAINING_DIR))
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_resample_identity() {
        let samples = vec![0.0f32, 0.5, 1.0, -0.5, -1.0];
        let result = crate::util::resample_audio(&samples, 44_100, 44_100);
        assert_eq!(result.len(), samples.len());
        for (a, b) in result.iter().zip(samples.iter()) {
            assert!((a - b).abs() < 1e-6, "values differ: {a} vs {b}");
        }
    }

    #[test]
    fn test_resample_downsample() {
        let samples: Vec<f32> = (0..1000)
            .map(|i| (i as f32 / 1000.0 * std::f32::consts::TAU).sin())
            .collect();
        let result = crate::util::resample_audio(&samples, 44_100, 16_000);
        // Expected output length: 1000 * 16000/44100 ≈ 363
        let expected_len = (1000.0_f64 * 16_000.0 / 44_100.0).ceil() as usize;
        assert_eq!(result.len(), expected_len);
        // Samples should be in range
        for &s in &result {
            assert!((-1.0..=1.0).contains(&s));
        }
    }

    #[test]
    fn test_add_white_noise() {
        // Flat signal at 0.5 — noise should produce measurable deviation
        let samples = vec![0.5f32; 1000];
        let noisy = add_noise(&samples, 10.0, NoiseType::White, None);
        assert_eq!(noisy.len(), samples.len());
        // All samples should still be in range
        for &s in &noisy {
            assert!((-1.0..=1.0).contains(&s));
        }
        // Signal should have changed (noise injection is deterministic in distribution)
        // At 10 dB SNR, signal RMS = 0.5, noise RMS ≈ 0.5/3.16 ≈ 0.158.
        // Measure noise-only RMS by subtracting the flat signal first,
        // avoiding the signal-noise cross-term that causes flakiness with
        // combined RMS (the noise sample mean can cancel the energy).
        let noise_part: Vec<f32> = noisy.iter().map(|&s| s - 0.5).collect();
        let noise_rms = compute_rms(&noise_part);
        let expected_noise_rms = 0.5 * 10.0_f32.powf(-10.0 / 20.0);
        assert!(
            (noise_rms - expected_noise_rms).abs() < 0.02,
            "White noise RMS should be ~{expected_noise_rms} at 10 dB SNR, got {noise_rms}"
        );
        // At least some samples should differ from the flat 0.5
        let changed = noisy.iter().filter(|&&s| (s - 0.5).abs() > 0.01).count();
        assert!(
            changed > 10,
            "At least some samples should be affected by noise (changed={changed})"
        );
    }

    #[test]
    fn test_add_pink_noise() {
        // Flat signal at 0.25 — pink noise should produce measurable deviation
        let samples = vec![0.25f32; 1000];
        let noisy = add_noise(&samples, 10.0, NoiseType::Pink, None);
        assert_eq!(noisy.len(), samples.len());
        for &s in &noisy {
            assert!((-1.0..=1.0).contains(&s));
        }
        // Signal should have changed (noise injection is deterministic in distribution)
        // At 10 dB SNR, signal RMS = 0.25, noise RMS ≈ 0.25/3.16 ≈ 0.079.
        // Measure noise-only RMS by subtracting the flat signal first,
        // avoiding the signal-noise cross-term that causes flakiness with
        // combined RMS (the noise sample mean can cancel the energy).
        let noise_part: Vec<f32> = noisy.iter().map(|&s| s - 0.25).collect();
        let noise_rms = compute_rms(&noise_part);
        let expected_noise_rms = 0.25 * 10.0_f32.powf(-10.0 / 20.0);
        assert!(
            (noise_rms - expected_noise_rms).abs() < 0.02,
            "Pink noise RMS should be ~{expected_noise_rms} at 10 dB SNR, got {noise_rms}"
        );
        // At least some samples should differ from the flat 0.25
        let changed = noisy.iter().filter(|&&s| (s - 0.25).abs() > 0.01).count();
        assert!(
            changed > 10,
            "At least some samples should be affected by pink noise (changed={changed})"
        );
    }

    #[test]
    fn test_randomize_volume_zero_gain() {
        let samples = vec![0.5f32; 100];
        let result = randomize_volume(&samples, 0.0, None);
        assert_eq!(result, samples);
    }

    #[test]
    fn test_randomize_volume_range() {
        let samples = vec![0.5f32; 100];
        let result = randomize_volume(&samples, 6.0, None);
        assert_eq!(result.len(), samples.len());
        for &s in &result {
            assert!((-1.0..=1.0).contains(&s));
        }
        // With ±6 dB max gain and 100 samples, at least some should differ
        // from the original 0.5 (statistically overwhelming with 100 trials)
        let changed = result.iter().filter(|&&s| (s - 0.5).abs() > 0.001).count();
        assert!(
            changed > 0,
            "Volume randomization should change at least some samples"
        );
    }

    #[test]
    fn test_speed_perturbation_identity() {
        let samples = vec![0.5f32; 100];
        let result = speed_perturbation(&samples, 16_000, 1.0);
        assert_eq!(result.len(), samples.len());
    }

    #[test]
    fn test_speed_perturbation_faster() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 / 100.0).sin()).collect();
        let result = speed_perturbation(&samples, 16_000, 1.2);
        assert!(!result.is_empty());
        // Faster speed = fewer output samples (output len < input len)
        assert!(
            result.len() < samples.len(),
            "Faster speed should produce fewer samples: {} >= {}",
            result.len(),
            samples.len()
        );
        for &s in &result {
            assert!((-1.0..=1.0).contains(&s));
        }
    }

    #[test]
    fn test_compute_rms_zero_signal() {
        let samples = vec![0.0f32; 100];
        let rms = compute_rms(&samples);
        assert!(rms < 1e-10, "Zero signal should have ~0 RMS, got {rms}");
    }

    #[test]
    fn test_compute_rms() {
        let samples = vec![1.0f32, -1.0, 1.0, -1.0];
        let rms = compute_rms(&samples);
        assert!((rms - 1.0).abs() < 1e-6, "RMS should be 1.0, got {rms}");
    }

    #[test]
    fn test_save_load_embeddings_roundtrip() {
        let dir = std::env::temp_dir().join("mahbot_tts_data_gen_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let embeddings = vec![
            vec![0.1f32, 0.2, 0.3],
            vec![0.4f32, 0.5, 0.6],
            vec![0.7f32, 0.8, 0.9],
        ];

        save_embeddings(&dir, "test_roundtrip.bin", &embeddings).unwrap();
        let loaded = load_embeddings(&dir, "test_roundtrip.bin").unwrap();

        assert_eq!(loaded.len(), embeddings.len());
        for (orig, loaded) in embeddings.iter().zip(loaded.iter()) {
            assert_eq!(orig.len(), loaded.len());
            for (a, b) in orig.iter().zip(loaded.iter()) {
                assert!((a - b).abs() < 1e-6, "values differ: {a} vs {b}");
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_generate_text_variants() {
        let variants = generate_text_variants("hey mahbot");
        // 7 variants: original, capitalized, upper, with !, with ., with ?, reduplicated
        assert_eq!(variants.len(), 7, "expected 7 text variants");
        assert!(variants.contains(&"hey mahbot".to_string()));
        assert!(variants.contains(&"Hey mahbot".to_string()));
        assert!(variants.contains(&"HEY MAHBOT".to_string()));
        assert!(variants.contains(&"hey mahbot!".to_string()));
        assert!(variants.contains(&"hey mahbot.".to_string()));
        assert!(variants.contains(&"hey mahbot?".to_string()));
        assert!(variants.contains(&"hey mahbot hey mahbot".to_string()));
    }

    #[test]
    fn test_generate_pink_noise() {
        // The former generate_pink_noise wrapper was inlined into add_noise
        // (mahbot-906). Test util::generate_pink_noise directly instead.
        let noise = crate::util::generate_pink_noise(1000, &mut StdRng::seed_from_u64(42));
        assert_eq!(noise.len(), 1000);
        // The canonical implementation normalizes to unit RMS (not
        // per-sample clamping).  This is the correct behavior for
        // audio processing — clamping is applied later in add_noise.
        // Pink noise should have non-zero RMS near 1.0 (unit normalization).
        let rms = compute_rms(&noise);
        assert!(
            rms > 0.01,
            "Pink noise RMS should be non-trivial, got {rms}"
        );
        assert!(
            (rms - 1.0).abs() < 0.1,
            "Pink noise RMS should be near 1.0 (unit normalization), got {rms}"
        );
    }

    #[test]
    fn test_randomize_volume_determinism() {
        let samples = vec![0.5f32; 100];
        let a = randomize_volume(&samples, 6.0, Some(42));
        let b = randomize_volume(&samples, 6.0, Some(42));
        assert_eq!(a, b, "same seed must produce identical output");
        let c = randomize_volume(&samples, 6.0, Some(99));
        assert_ne!(a, c, "different seed must produce different output");
    }

    #[test]
    fn test_add_noise_determinism() {
        let samples = vec![0.5f32; 100];
        let a = add_noise(&samples, 10.0, NoiseType::White, Some(42));
        let b = add_noise(&samples, 10.0, NoiseType::White, Some(42));
        assert_eq!(a, b, "same seed must produce identical white noise output");
        let c = add_noise(&samples, 10.0, NoiseType::Pink, Some(42));
        let d = add_noise(&samples, 10.0, NoiseType::Pink, Some(42));
        assert_eq!(c, d, "same seed must produce identical pink noise output");
    }
}
