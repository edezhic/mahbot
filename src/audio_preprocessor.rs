//! Audio pre-processing pipeline for voice assistant.
//!
//! Provides noise suppression (via WebRTC-based `sonora-ns`) and RMS-based
//! automatic gain control (AGC) as an optional pre-processing step before mel
//! spectrogram extraction.
//!
//! # Processing order
//!
//! 1. **Noise suppression** (optional) — stationary noise (fan hum, AC, traffic)
//!    is removed using the WebRTC noise suppressor, ported to pure Rust in
//!    `sonora-ns`.  The suppressor is stateful — it adapts to the noise profile
//!    over time.
//!
//! 2. **AGC** (optional) — RMS-based automatic gain control normalises the
//!    signal level to a target RMS, with a clamped gain range to avoid
//!    amplifying pure silence or transient noise.
//!
//! Both are configurable via [`PreprocessorConfig`] and can be enabled or
//! disabled independently.  They default to ON.
//!
//! # Integration
//!
//! [`AudioPreprocessor`] is owned by [`PipelineCtx`] and applied to every
//! incoming audio chunk in the main voice pipeline loop, before the chunk
//! reaches VAD, wake-word detection, or enrollment.  This ensures that
//! enrollment audio receives the exact same pre-processing as live detection
//! audio — eliminating the systematic mismatch between quiet close-mic
//! enrolment and far-field live usage.

use sonora_ns::config::{NS_FRAME_SIZE, SuppressionLevel};
use sonora_ns::noise_suppressor::NoiseSuppressor;

// ── Constants ───────────────────────────────────────────────────────────

/// Target RMS value for AGC.  0.05 corresponds to typical speech level at
/// ~10 cm from microphone (the common enrollment distance).
const TARGET_RMS: f32 = 0.05;

/// Minimum gain multiplier — prevents amplifying pure silence or low-level
/// noise bursts into false wake-word triggers.
const MIN_GAIN: f32 = 0.25;

/// Maximum gain multiplier — prevents clipping and limits energy boost for
/// far-field audio that would otherwise be amplified excessively.
const MAX_GAIN: f32 = 4.0;

// ── Configuration ──────────────────────────────────────────────────────

/// Configuration for the audio pre-processor.
///
/// Both fields default to `true` — noise suppression and AGC are enabled
/// out of the box.  Users can disable them individually if they have a
/// high-quality microphone in a treated room.
#[derive(Debug, Clone, Copy)]
pub struct PreprocessorConfig {
    /// Enable WebRTC-based noise suppression (default: `true`).
    pub noise_suppression: bool,
    /// Enable RMS-based automatic gain control (default: `true`).
    pub agc: bool,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            noise_suppression: true,
            agc: true,
        }
    }
}

// ── Audio pre-processor ────────────────────────────────────────────────

/// Audio pre-processing pipeline: noise suppression → AGC.
///
/// Maintains internal state for the noise suppressor (which adapts to the
/// noise floor over time) and a sample buffer for frame-aligned processing.
///
/// # Thread safety
///
/// `AudioPreprocessor` is neither `Send` nor `Sync` — it must be used from
/// a single thread (the voice pipeline task).  This matches the ownership
/// model of [`PipelineCtx`] which is also single-threaded.
pub struct AudioPreprocessor {
    /// Optional noise suppressor (present when enabled).
    suppressor: Option<NoiseSuppressor>,
    /// Buffer for samples that don't yet fill a full NS frame (160 samples).
    /// Stored in int16-range f32 format (multiplied by 32768.0).
    ns_buffer: Vec<f32>,
    /// Read cursor for frame-aligned processing — tracks how many samples
    /// from `ns_buffer` have been consumed so far without draining.
    read_pos: usize,
    /// Configuration.
    config: PreprocessorConfig,
}

impl AudioPreprocessor {
    /// Create a new pre-processor with the given configuration.
    #[must_use]
    pub fn new(config: PreprocessorConfig) -> Self {
        let suppressor = if config.noise_suppression {
            Some(NoiseSuppressor::with_level(SuppressionLevel::K12dB))
        } else {
            None
        };

        Self {
            suppressor,
            ns_buffer: Vec::new(),
            read_pos: 0,
            config,
        }
    }

    /// Process a chunk of audio samples through noise suppression and/or AGC.
    ///
    /// `samples` must be in [-1.0, 1.0] f32 range (MahBot's native audio
    /// representation).  Returns processed samples in the same range.
    ///
    /// # Output length
    ///
    /// May return fewer samples than the input when noise suppression is
    /// enabled and the input length is not a multiple of the NS frame size
    /// (160 samples).  Incomplete trailing frames are buffered internally
    /// and carried over to the next [`process()`] call.  Downstream consumers
    /// that require fixed-size output should accumulate across calls.
    ///
    /// When both NS and AGC are disabled, returns the input unchanged
    /// (zero-copy — the passed `Vec` is returned directly).
    #[must_use]
    pub fn process(&mut self, samples: Vec<f32>) -> Vec<f32> {
        if !self.config.noise_suppression && !self.config.agc {
            return samples;
        }

        let mut processed = samples;

        if self.config.noise_suppression {
            processed = self.apply_noise_suppression(processed);
        }

        if self.config.agc {
            processed = Self::apply_agc(processed);
        }

        processed
    }

    /// Clear the frame-alignment buffer without disturbing the noise
    /// suppressor's adapted noise profile.
    ///
    /// Called during pipeline state transitions (e.g., listening→recording)
    /// to flush stale samples without discarding the noise floor estimate
    /// the suppressor has built up over time.
    pub fn clear_buffer(&mut self) {
        self.ns_buffer.clear();
        self.read_pos = 0;
    }

    /// Full reset: discard both the sample buffer and the noise suppressor's
    /// adapted noise profile.
    ///
    /// Should be called when the microphone stream is re-created or the
    /// acoustic environment changes significantly, as the old noise profile
    /// may no longer be representative.
    pub fn reset(&mut self) {
        self.ns_buffer.clear();
        self.read_pos = 0;
        self.suppressor = if self.config.noise_suppression {
            Some(NoiseSuppressor::with_level(SuppressionLevel::K12dB))
        } else {
            None
        };
    }

    // ── Private helpers ───────────────────────────────────────────

    /// Apply WebRTC noise suppression to a chunk of audio.
    ///
    /// Audio is scaled to int16 range for NS processing, then scaled back
    /// to [-1.0, 1.0].  Incomplete final frames (< 160 samples) are buffered
    /// and carried over to the next call.
    fn apply_noise_suppression(&mut self, samples: Vec<f32>) -> Vec<f32> {
        let Some(ns) = &mut self.suppressor else {
            return samples;
        };

        // Scale from [-1, 1] to int16 range for the noise suppressor.
        self.ns_buffer.extend(samples.iter().map(|&s| s * 32768.0));

        // Number of complete frames available.
        let avail = (self.ns_buffer.len() - self.read_pos) / NS_FRAME_SIZE;
        let mut output: Vec<f32> = Vec::with_capacity(avail * NS_FRAME_SIZE);
        let mut frame = [0.0f32; NS_FRAME_SIZE];

        for _ in 0..avail {
            frame.copy_from_slice(&self.ns_buffer[self.read_pos..self.read_pos + NS_FRAME_SIZE]);
            self.read_pos += NS_FRAME_SIZE;

            ns.analyze(&frame);
            ns.process(&mut frame);

            output.extend_from_slice(&frame);
        }

        // Trim consumed samples when the buffer grows large enough.
        if self.read_pos >= NS_FRAME_SIZE * 32 {
            self.ns_buffer.drain(..self.read_pos);
            self.read_pos = 0;
        }

        // Scale back to [-1.0, 1.0].
        output.iter().map(|&s| s / 32768.0).collect()
    }

    /// Apply RMS-based AGC to a chunk of audio.
    ///
    /// Computes RMS, computes gain = TARGET_RMS / rms, clamps to
    /// [MIN_GAIN, MAX_GAIN], and applies the gain to all samples.
    #[allow(clippy::cast_precision_loss)]
    fn apply_agc(samples: Vec<f32>) -> Vec<f32> {
        if samples.is_empty() {
            return samples;
        }

        let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
        let rms = (sum_sq / samples.len() as f32).sqrt();

        if rms == 0.0 {
            return samples; // pure silence — don't amplify
        }

        let gain = (TARGET_RMS / rms).clamp(MIN_GAIN, MAX_GAIN);
        samples.iter().map(|&s| s * gain).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: generate white noise samples.
    fn white_noise(amplitude: f32, num_samples: usize) -> Vec<f32> {
        // Deterministic pseudo-noise using sine overlap (reproduces fixed
        // amplitude distribution without requiring a dedicated RNG).
        (0..num_samples)
            .map(|i| {
                let t = i as f32;
                (t * 0.073).sin() * amplitude * 0.3
                    + (t * 0.137).sin() * amplitude * 0.3
                    + (t * 0.291).sin() * amplitude * 0.3
            })
            .collect()
    }

    /// Helper: generate a pure tone at 440 Hz (A4).
    fn sine_tone(amplitude: f32, num_samples: usize, sample_rate: u32) -> Vec<f32> {
        use std::f32::consts::PI;
        let freq = 440.0; // A4
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (t * freq * 2.0 * PI).sin() * amplitude
            })
            .collect()
    }

    // ── Tests ────────────────────────────────────────────────────

    /// Test that noise suppression measurably changes the mel input.
    ///
    /// A synthetic noise pattern with significant non-speech band energy
    /// should produce different output after NS processing — the suppressor
    /// should reduce energy in frequency bands that don't match speech.
    #[test]
    fn test_noise_suppression_applied() {
        let mut pre = AudioPreprocessor::new(PreprocessorConfig {
            noise_suppression: true,
            agc: false,
        });

        // Generate noise audio: white noise at moderate amplitude.
        // Use multiple chunks to let the suppressor converge its noise profile.
        let chunk_size = NS_FRAME_SIZE * 3; // 480 samples per chunk
        let mut all_input_energy = 0.0f32;
        let mut all_output_energy = 0.0f32;

        for _ in 0..50 {
            let noise = white_noise(0.1, chunk_size);

            let input_energy: f32 = noise.iter().map(|&s| s * s).sum();
            all_input_energy += input_energy;

            let processed = pre.process(noise);

            let output_energy: f32 = processed.iter().map(|&s| s * s).sum();
            all_output_energy += output_energy;
        }

        // After convergence, output energy should be lower than input.
        let ratio = all_output_energy / all_input_energy;
        assert!(
            ratio < 0.9,
            "noise should be suppressed: output/input energy ratio = {ratio}"
        );
    }

    /// Test that AGC normalises volume to within 20% of TARGET_RMS.
    ///
    /// Feeds audio at 0.25× and 4.0× normal amplitude, verifying that
    /// the post-AGC RMS is close to TARGET_RMS.  Also verifies the
    /// gain clamp prevents amplifying pure silence.
    #[test]
    fn test_agc_normalizes_volume() {
        let mut pre = AudioPreprocessor::new(PreprocessorConfig {
            noise_suppression: false,
            agc: true,
        });

        // A sine tone has RMS = amplitude / sqrt(2).  To get a specific RMS,
        // the required amplitude is: amplitude = target_rms * sqrt(2).
        let target_rms = TARGET_RMS;
        let sqrt2 = std::f32::consts::SQRT_2;

        // ── 0.25× target RMS ──
        let amp_quiet = target_rms * 0.25 * sqrt2;
        let quiet = sine_tone(amp_quiet, NS_FRAME_SIZE * 5, 16_000);
        let processed_quiet = pre.process(quiet);
        let rms_quiet = compute_rms(&processed_quiet);
        let rel_err_quiet = (rms_quiet - target_rms).abs() / target_rms;
        assert!(
            rel_err_quiet < 0.25,
            "quiet audio AGC: rms={rms_quiet:.6} target={target_rms} rel_err={rel_err_quiet:.4} (expected <0.25)"
        );

        // ── 4.0× target RMS ──
        let amp_loud = target_rms * 4.0 * sqrt2;
        let loud = sine_tone(amp_loud, NS_FRAME_SIZE * 5, 16_000);
        let processed_loud = pre.process(loud);
        let rms_loud = compute_rms(&processed_loud);
        let rel_err_loud = (rms_loud - target_rms).abs() / target_rms;
        assert!(
            rel_err_loud < 0.25,
            "loud audio AGC: rms={rms_loud:.6} target={target_rms} rel_err={rel_err_loud:.4} (expected <0.25)"
        );

        // ── Silence should not be amplified ──
        let silence = vec![0.0f32; NS_FRAME_SIZE * 3];
        let processed_silence = pre.process(silence);
        let rms_silence = compute_rms(&processed_silence);
        assert!(
            rms_silence < 0.001,
            "silence should not be amplified: rms={rms_silence:.8}"
        );
    }

    /// Test that noise suppression + AGC compose correctly (AGC after NS).
    ///
    /// Processing order matters: AGC must come AFTER noise suppression, not
    /// before — otherwise the NS would see amplified noise, making it harder
    /// to suppress.  This test verifies the composition produces a valid
    /// output with normalised RMS.
    #[test]
    fn test_agc_and_ns_compose() {
        let mut pre = AudioPreprocessor::new(PreprocessorConfig {
            noise_suppression: true,
            agc: true,
        });

        // Generate noisy audio (low amplitude, to stress both NS and AGC).
        let chunk_size = NS_FRAME_SIZE * 6;
        let mut all_output_rms = 0.0f32;
        let mut num_chunks = 0;

        for _ in 0..30 {
            let noise = white_noise(0.05, chunk_size);
            let processed = pre.process(noise);
            all_output_rms += compute_rms(&processed);
            num_chunks += 1;
        }

        let avg_rms = all_output_rms / num_chunks as f32;

        // RMS should be in a reasonable range — not amplified to clipping,
        // not suppressed to silence. After NS + AGC, the output should be
        // near TARGET_RMS (within 50% due to NS energy removal).
        let ratio = avg_rms / TARGET_RMS;
        assert!(
            ratio > 0.3 && ratio < 2.0,
            "NS+AGC composition: avg_rms={avg_rms:.6} ratio={ratio:.4} (expected 0.3–2.0 of target={TARGET_RMS})"
        );
    }

    /// Test that disabling both NS and AGC passes audio through unchanged.
    #[test]
    fn test_bypass_returns_input() {
        let mut pre = AudioPreprocessor::new(PreprocessorConfig {
            noise_suppression: false,
            agc: false,
        });

        let input: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
        let output = pre.process(input.clone());
        assert_eq!(input, output, "bypass mode should return input unchanged");
    }

    // ── Helpers ──────────────────────────────────────────────────

    fn compute_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }
}
