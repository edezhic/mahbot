//! Audio-subsystem utilities: PCM resampling for the capture path, and the macOS
//! side's embedding similarity and deterministic test-signal generators.
//!
//! These live here (not in `crate::util`) because they are the audio subsystem's
//! own: the resampler serves the microphone stream on every platform, while the
//! embedding similarity, the RMS helper and the signal generators belong to the
//! wake word's enrollment and its tests, and are gated with it.

#[cfg(all(test, target_os = "macos"))]
use rand::RngExt;
#[cfg(all(test, target_os = "macos"))]
use rand::SeedableRng;

// PCM resampling

/// Resample PCM audio from one sample rate to another using linear
/// interpolation with a 3-tap binomial anti-aliasing filter for downsampling.
///
/// This is the canonical implementation: every call site delegates to it. The
/// recogniser is not one of them — it hands the engine a file's own rate and the
/// engine resamples ([`crate::speech::decode`]).
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
/// substituted here — every call site benefits automatically.
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

// Audio utility functions (canonical implementations)

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
#[cfg(all(test, target_os = "macos"))]
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
#[cfg(all(test, target_os = "macos"))]
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
#[cfg(all(test, target_os = "macos"))]
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
#[cfg(target_os = "macos")]
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
#[cfg(all(test, target_os = "macos"))]
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

// Audio utility regression tests
// In-place regression net for speed perturbation, gain, noise, and pink noise.
// The utilities are `cfg(test)`-gated, like the tests that exercise them.

#[cfg(all(test, target_os = "macos"))]
mod audio_util_tests {
    use super::{add_noise, apply_gain, generate_pink_noise, speed_perturbation};
    use rand::SeedableRng;

    /// A ramp of `len` samples spanning −1..1.
    #[expect(clippy::cast_precision_loss)] // small test lengths — exact in f32
    fn ramp(len: usize) -> Vec<f32> {
        let half = len as f32 / 2.0;
        (0..len).map(|i| (i as f32 - half) / half).collect()
    }

    /// Speed perturbation resamples: 1.0 is the input unchanged, below 1
    /// lengthens the signal and above 1 shortens it.
    #[test]
    fn speed_perturbation_changes_length_by_the_factor() {
        let pcm = ramp(100);

        let identity = speed_perturbation(&pcm, 16000, 1.0);
        assert_eq!(identity.len(), pcm.len(), "identity should preserve length");
        for (a, b) in pcm.iter().zip(identity.iter()) {
            assert!((a - b).abs() < 1e-5, "identity should preserve values");
        }

        assert!(
            speed_perturbation(&pcm, 16000, 0.5).len() > pcm.len(),
            "rate < 1 should increase sample count"
        );
        assert!(
            speed_perturbation(&pcm, 16000, 2.0).len() < pcm.len(),
            "rate > 1 should decrease sample count"
        );
    }

    /// The decibel conversion: 0 dB is unity gain, −6 dB halves the amplitude
    /// and +6 dB doubles it.
    #[test]
    fn apply_gain_scales_by_the_decibel_factor() {
        let pcm: Vec<f32> = vec![0.5, -0.3, 0.1, -0.7, 0.9];
        for gain_db in [0.0, -6.0, 6.0] {
            let expected_amp = 10.0_f32.powf(gain_db / 20.0);
            for (orig, gained) in pcm.iter().zip(apply_gain(&pcm, gain_db).iter()) {
                assert!(
                    (gained - orig * expected_amp).abs() < 1e-6,
                    "{gain_db} dB gain should multiply by {expected_amp}"
                );
            }
        }
    }

    /// Seeded pink noise is reproducible, a different seed is a different
    /// mixture, and the SNR target is honoured — at 100 dB the mix is nearly
    /// the signal.
    #[test]
    fn add_noise_tracks_the_seed_and_the_snr() {
        let pcm = ramp(200);
        let a = add_noise(&pcm, 25.0, 42);
        assert_eq!(a.len(), pcm.len(), "add_noise output length should match");
        assert_eq!(
            a,
            add_noise(&pcm, 25.0, 42),
            "same seed must be reproducible"
        );
        assert_ne!(
            a,
            add_noise(&pcm, 25.0, 99),
            "a different seed should mix differently"
        );

        let pcm = ramp(1000);
        let noisy = add_noise(&pcm, 100.0, 42);
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

// Embedding similarity
/// Cosine similarity between two embedding vectors. Returns 0.0–1.0.
///
/// The wake-word matcher compares a captured window embedding against its
/// enrolment prototype, negatives and anti-prototypes with it, and enrolment
/// checks each utterance against the running centroid with it.
#[must_use]
#[cfg(target_os = "macos")]
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

#[cfg(all(test, target_os = "macos"))]
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
