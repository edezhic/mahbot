//! Audio-subsystem utilities: PCM math and resampling, deterministic
//! test-signal generation for the voice tests, embedding cosine similarity, and
//! the streaming SHA-256 verifier the audio model downloads share.
//!
//! These live here (not in `crate::util`) because the whole audio subsystem
//! is macOS-only.

use anyhow::{Context as _, Result};
#[cfg(test)]
use rand::RngExt;
#[cfg(test)]
use rand::SeedableRng;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::util::hex_string;

// ── PCM resampling ─────────────────────────────────────────────────────

/// Resample PCM audio from one sample rate to another using linear
/// interpolation with a 3-tap binomial anti-aliasing filter for downsampling.
///
/// This is the canonical implementation. All other resample call sites
/// delegate to this one, except `audio::local_transcriber`, which uses
/// `qwen_asr::audio::resample` directly.
///
/// When `from_rate > to_rate` (downsampling), a simple binomial low-pass filter
/// is applied to attenuate frequencies above the new Nyquist before decimation.
/// Without this filter, linear interpolation introduces aliasing — high-frequency
/// content above `to_rate / 2` folds back into the audible range as noise.
///
/// The 3-tap binomial `[0.25, 0.5, 0.25]` gives reasonable stopband attenuation
/// (~6 dB at 0.25 normalised) for speech audio. For 48 kHz → 16 kHz this
/// attenuates content above ~8 kHz.
///
/// # Aliasing trade-off
///
/// Linear interpolation introduces aliasing when downsampling even with the
/// pre-filter — the filter only provides ~6 dB stopband attenuation. This is
/// acceptable for speech processing and wake word training data augmentation.
/// If aliasing artifacts prove problematic, a sinc-based resampler can be
/// substituted here — all call sites except `local_transcriber` (which
/// resamples via `qwen_asr` itself) benefit automatically.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub(crate) fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let output_len = (samples.len() as f64 * ratio).ceil() as usize;

    // Anti-aliasing filter for downsampling
    let filtered: Vec<f32> = if from_rate > to_rate && samples.len() >= 3 {
        let mut out = Vec::with_capacity(samples.len());
        out.push(samples[0] * 0.75 + samples[1] * 0.25);
        for i in 1..samples.len() - 1 {
            out.push(samples[i - 1] * 0.25 + samples[i] * 0.5 + samples[i + 1] * 0.25);
        }
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

// ── Shared model-integrity verifier ────────────────────────────────────

/// Verify a file's SHA256 hash matches the expected hex string.
///
/// Shared model-integrity verifier extracted from the three near-identical
/// private streaming copies in `audio::tts`, `audio::local_transcriber`, and
/// `audio::voice` (whose empty-`expected` skip semantics and error wording had
/// drifted).  Model-download integrity is security-adjacent, so the copies are
/// consolidated here.
///
/// * If `expected` is empty, verification is skipped (returns `Ok`) — the
///   canonical "no hash configured" semantics that won the reconciliation.
/// * Uses streaming SHA256 via [`Sha256::update`] to avoid loading the entire
///   file into memory (model files can be multiple GB).
pub(crate) fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    if expected.is_empty() {
        return Ok(()); // no hash configured — skip verification
    }

    let mut hasher = Sha256::new();
    let mut file = File::open(path)
        .with_context(|| format!("Failed to open {} for SHA256 verification", path.display()))?;
    let mut buf = vec![0u8; 65536]; // 64 KB heap buffer
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

// Tests consolidated from the three former private copies (tts,
// local_transcriber, voice).  The file-not-found error path is covered with a
// NON-empty expected hash because the shared empty-hash skip returns Ok before
// opening the file (sanctioned reconciliation of the transcriber's original
// `verify_sha256(path, "").is_err()` assertion).

#[cfg(test)]
mod verify_sha256_tests {
    use super::verify_sha256;

    fn sha256_hex(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        crate::util::hex_string(&hasher.finalize())
    }

    #[test]
    fn verify_sha256_cases() {
        // Matching and mismatching hashes over a written file.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        std::fs::write(&path, b"test data").unwrap();
        let hash = sha256_hex(b"test data");
        assert!(
            verify_sha256(&path, &hash).is_ok(),
            "matching hash should pass"
        );
        assert!(
            verify_sha256(&path, &sha256_hex(b"other data")).is_err(),
            "mismatching hash should fail"
        );

        // Empty expected hash → Ok without opening the file (skip semantics).
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nonexistent.bin");
        assert!(
            verify_sha256(&missing, "").is_ok(),
            "empty hash should skip verification"
        );
        // The not-found error path requires a non-empty expected hash.
        assert!(
            verify_sha256(&missing, &sha256_hex(b"anything")).is_err(),
            "missing file with non-empty hash should fail"
        );
    }
}

// ── Audio utility functions (canonical implementations) ────────────────

/// Generate pink noise (1/f spectrum) using the Voss-McCartney algorithm.
///
/// Uses a seeded RNG for reproducibility.  The high-pass delta variant
/// naturally removes DC bias.  Output is normalized to unit RMS.
///
/// # Voss-McCartney variants
///
/// There are several common variants of Voss-McCartney pink noise:
///
/// * **High-pass delta** (this implementation): stores previous value per
///   octave, emits the difference (new - prev).  Removes DC bias naturally.
/// * **Direct-sum**: sums all octave values directly.  May accumulate DC bias.
///
/// The canonical implementation uses 16 octaves (~3 dB/octave rolloff down
/// to 0.03 Hz at 16 kHz) and the high-pass delta variant for DC-free output.
#[cfg(test)]
pub(crate) fn generate_pink_noise(len: usize, mut rng: impl rand::Rng) -> Vec<f32> {
    const NUM_OCTAVES: usize = 16;
    let mut values = [0.0f32; NUM_OCTAVES];
    let mut outputs = [0.0f32; NUM_OCTAVES];
    let mut sample_count = 0u64;
    let mut noise = Vec::with_capacity(len);

    for _ in 0..len {
        sample_count += 1;
        let mut sum = 0.0;
        for octave in 0..NUM_OCTAVES {
            // Update this octave's generator at intervals of 2^octave samples.
            if sample_count.is_multiple_of(1u64 << octave) {
                values[octave] = rng.random::<f32>() * 2.0 - 1.0;
            }
            // High-pass delta: emit difference instead of direct value.
            let new_val = values[octave];
            let delta = new_val - outputs[octave];
            outputs[octave] = new_val;
            sum += delta;
        }
        noise.push(sum);
    }

    // Normalize to unit RMS
    let rms = compute_rms(&noise).max(1e-10);
    for s in &mut noise {
        *s /= rms;
    }

    noise
}

/// Add pink noise to PCM audio at the given SNR (deterministic, seeded).
///
/// Generates unit-RMS pink noise from a seeded RNG, scales it to the SNR
/// target relative to the signal's own RMS, and clamps the mix to ±1.0.
#[cfg(test)]
pub(crate) fn add_noise(pcm: &[f32], snr_db: f32, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let signal_rms = compute_rms(pcm).max(1e-10);
    let noise = generate_pink_noise(pcm.len(), &mut rng);
    let noise_rms_target = signal_rms * 10.0_f32.powf(-snr_db / 20.0);
    let noise_rms_current = compute_rms(&noise).max(1e-10);
    let scale = noise_rms_target / noise_rms_current;
    pcm.iter()
        .zip(noise.iter())
        .map(|(&s, &n)| (s + n * scale).clamp(-1.0, 1.0))
        .collect()
}

/// Apply a fixed gain to PCM audio.
///
/// DETERMINISTIC — no RNG involved. The gain is `10^(gain_db / 20)`.
/// Negative values attenuate, positive values amplify.
#[cfg(test)]
pub(crate) fn apply_gain(pcm: &[f32], gain_db: f32) -> Vec<f32> {
    let amp = 10.0_f32.powf(gain_db / 20.0);
    pcm.iter().map(|&s| s * amp).collect()
}

/// Compute the RMS (root mean square) of audio samples.
///
/// Returns `0.0` for empty input.  Ungated so production hot
/// paths (AGC, noise generation, utterance quality) share one implementation
/// instead of hand-rolling `sum(x²)/n → sqrt`.  Callers that need a
/// divide-by-zero floor for degenerate all-zero input apply `.max(1e-10)` at
/// the call site — it is deliberately NOT part of this function.
#[expect(clippy::cast_precision_loss)]
pub(crate) fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
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
#[cfg(test)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub(crate) fn speed_perturbation(samples: &[f32], sample_rate: u32, factor: f32) -> Vec<f32> {
    if samples.is_empty() || (factor - 1.0).abs() < 1e-6 {
        return samples.to_vec();
    }
    // Speed perturbation via resampling: change the effective rate
    // new_rate = sample_rate * factor
    let effective_rate = (sample_rate as f32 * factor) as u32;
    // Resample from effective_rate back to sample_rate
    // This produces the same duration as original but with shifted pitch
    resample_audio(samples, effective_rate, sample_rate)
}

// ── Audio utility regression tests ─────────────────────────────────────
// In-place regression net for speed perturbation, gain, noise, and pink noise.
// The utilities are `cfg(test)`-gated, like the tests that exercise them.

#[cfg(test)]
mod audio_util_tests {
    use super::{add_noise, apply_gain, generate_pink_noise, speed_perturbation};
    use rand::SeedableRng;

    #[test]
    fn test_speed_perturbation_identity() {
        // rate=1.0 should return approximately the original
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..100 — exact in f32
        let pcm: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let result = speed_perturbation(&pcm, 16000, 1.0);
        assert_eq!(result.len(), pcm.len(), "identity should preserve length");
        for (a, b) in pcm.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-5, "identity should preserve values");
        }
    }

    #[test]
    fn test_speed_perturbation_rates() {
        // Slow down: rate=0.5 should produce more samples
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..100 — exact in f32
        let pcm: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let slowed = speed_perturbation(&pcm, 16000, 0.5);
        assert!(
            slowed.len() > pcm.len(),
            "rate < 1 should increase sample count"
        );
        // Speed up: rate=2.0 should produce fewer samples
        let sped_up = speed_perturbation(&pcm, 16000, 2.0);
        assert!(
            sped_up.len() < pcm.len(),
            "rate > 1 should decrease sample count"
        );
    }

    #[test]
    fn test_speed_perturbation_determinism() {
        // Same input + same rate → same output
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..100 — exact in f32
        let pcm: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let a = speed_perturbation(&pcm, 16000, 0.95);
        let b = speed_perturbation(&pcm, 16000, 0.95);
        assert_eq!(a, b, "deterministic speed perturbation");
    }

    #[test]
    fn test_apply_gain_determinism() {
        // apply_gain must be deterministic: same PCM + same gain_db → same output
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..50 — exact in f32
        let pcm: Vec<f32> = (0..50).map(|i| (i as f32 - 25.0) / 25.0).collect();
        let a = apply_gain(&pcm, -3.0);
        let b = apply_gain(&pcm, -3.0);
        assert_eq!(a, b, "apply_gain must be deterministic");
    }

    #[test]
    fn test_apply_gain_db_conversion() {
        // 0 dB → unity gain (output == input)
        let pcm: Vec<f32> = vec![0.5, -0.3, 0.1, -0.7, 0.9];
        let unity = apply_gain(&pcm, 0.0);
        for (a, b) in pcm.iter().zip(unity.iter()) {
            assert!((a - b).abs() < 1e-6, "0 dB gain should be unity");
        }
        // -6 dB → amplitude halved (10^(-6/20) ≈ 0.5)
        let attenuated = apply_gain(&pcm, -6.0);
        let expected_amp = 10.0_f32.powf(-6.0 / 20.0);
        for (orig, atten) in pcm.iter().zip(attenuated.iter()) {
            assert!(
                (atten - orig * expected_amp).abs() < 1e-6,
                "-6 dB gain should multiply by {expected_amp}"
            );
        }
        // +6 dB → amplitude doubled (10^(6/20) ≈ 2.0)
        let amplified = apply_gain(&pcm, 6.0);
        let expected_amp2 = 10.0_f32.powf(6.0 / 20.0);
        for (orig, amp) in pcm.iter().zip(amplified.iter()) {
            assert!(
                (amp - orig * expected_amp2).abs() < 1e-6,
                "+6 dB gain should multiply by {expected_amp2}"
            );
        }
    }

    #[test]
    fn test_add_noise_determinism() {
        // Same PCM + same SNR + same seed → same output
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..200 — exact in f32
        let pcm: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) / 100.0).collect();
        let a = add_noise(&pcm, 25.0, 42);
        let b = add_noise(&pcm, 25.0, 42);
        assert_eq!(a.len(), b.len(), "add_noise output length should match");
        assert_eq!(a, b, "add_noise with same seed must be deterministic");
    }

    #[test]
    fn test_add_noise_seed_variation() {
        // Different seed → different output
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..200 — exact in f32
        let pcm: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) / 100.0).collect();
        let a = add_noise(&pcm, 25.0, 42);
        let b = add_noise(&pcm, 25.0, 99);
        assert_ne!(a, b, "different seeds should produce different output");
    }

    #[test]
    fn test_add_noise_snr_approximation() {
        // At very high SNR (100 dB), the output should be very close to input
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..1000 — exact in f32
        let pcm: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 500.0).collect();
        let noisy = add_noise(&pcm, 100.0, 42);
        // Compute actual SNR of output vs input
        let signal_power: f32 = pcm.iter().map(|&s| s * s).sum();
        let noise_power: f32 = pcm
            .iter()
            .zip(noisy.iter())
            .map(|(&s, &n)| (n - s) * (n - s))
            .sum();
        let actual_snr_db = 10.0 * (signal_power / noise_power.max(1e-10)).log10();
        assert!(
            actual_snr_db > 80.0,
            "at 100 dB target, actual SNR should be high (was {actual_snr_db:.1} dB)"
        );
    }

    #[test]
    fn test_generate_pink_noise_properties() {
        // Pink noise should have positive and negative values
        let noise = generate_pink_noise(1000, rand::rngs::StdRng::seed_from_u64(42));
        assert_eq!(noise.len(), 1000);
        let has_positive = noise.iter().any(|&s| s > 0.0);
        let has_negative = noise.iter().any(|&s| s < 0.0);
        assert!(has_positive, "pink noise should have positive values");
        assert!(has_negative, "pink noise should have negative values");
        // RMS should be near 1.0 (normalized)
        let rms = super::compute_rms(&noise);
        assert!(
            (rms - 1.0).abs() < 0.1,
            "pink noise RMS should be near 1.0 (was {rms})"
        );
    }

    #[test]
    fn test_augmentation_variants_are_different() {
        // Verify that the 4 augmentation strategies produce different results
        #[expect(clippy::cast_precision_loss)] // i ∈ 0..500 — exact in f32
        let pcm: Vec<f32> = (0..500).map(|i| (i as f32 - 250.0) / 250.0).collect();
        let speed_down = speed_perturbation(&pcm, 16000, 0.95);
        let speed_up = speed_perturbation(&pcm, 16000, 1.05);
        let volume_down = apply_gain(&pcm, -3.0);
        let noise = add_noise(&pcm, 25.0, 42);

        // Each variant should differ from original AND each other
        assert_ne!(speed_down, pcm, "speed-down should differ from original");
        assert_ne!(speed_up, pcm, "speed-up should differ from original");
        assert_ne!(volume_down, pcm, "volume-down should differ from original");
        assert_ne!(noise, pcm, "noise should differ from original");

        // Verify they also differ from each other (different transforms)
        assert_ne!(
            speed_down, speed_up,
            "speed-down and speed-up should differ"
        );
        assert_ne!(
            speed_down, volume_down,
            "speed-down and volume-down should differ"
        );
        assert_ne!(speed_down, noise, "speed-down and noise should differ");
    }
}

// ── Embedding similarity ────────────────────────────────────────────────
/// Cosine similarity between two embedding vectors. Returns 0.0–1.0.
///
/// The wake-word matcher compares a captured window embedding against its
/// enrolment prototype, negatives and anti-prototypes with it, and enrolment
/// checks each utterance against the running centroid with it.
#[must_use]
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if !denom.is_finite() || denom < f32::EPSILON {
        return 0.0;
    }

    let raw = dot / denom;
    if !raw.is_finite() {
        return 0.0;
    }

    // Clamp to [0, 1] — embeddings are typically positive
    raw.clamp(0.0, 1.0)
}

#[cfg(test)]
mod cosine_similarity_tests {
    use super::cosine_similarity;

    #[test]
    fn cosine_similarity_all_cases() {
        struct Case {
            name: &'static str,
            a: &'static [f32],
            b: &'static [f32],
            expected: f32,
        }

        let cases = [
            Case {
                name: "identical vectors",
                a: &[1.0, 2.0, 3.0],
                b: &[1.0, 2.0, 3.0],
                expected: 1.0,
            },
            Case {
                name: "orthogonal vectors",
                a: &[1.0, 0.0, 0.0],
                b: &[0.0, 1.0, 0.0],
                expected: 0.0,
            },
            Case {
                name: "empty first vector",
                a: &[],
                b: &[1.0, 2.0],
                expected: 0.0,
            },
            Case {
                name: "different lengths",
                a: &[1.0, 2.0],
                b: &[1.0, 2.0, 3.0],
                expected: 0.0,
            },
        ];

        for case in &cases {
            let result = cosine_similarity(case.a, case.b);
            assert!(
                (result - case.expected).abs() < 1e-6,
                "case: {} — expected {}, got {}",
                case.name,
                case.expected,
                result,
            );
        }
    }
}
