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

/// EMA attack coefficient for RMS level tracking.
///
/// When the current chunk RMS exceeds the running estimate (speech onset),
/// we adapt quickly (α=0.20) so that gain is reduced within ~10 chunks
/// (~320ms at 32ms/chunk), preventing clipping during sudden loud speech.
///
/// Reaches 90% of new steady-state in: ln(0.1)/ln(1-0.20) ≈ 10 updates.
///
/// This value is in the typical range for speech AGC (0.10–0.30).  At
/// 0.20, the response is fast enough to catch loud onsets before clipping
/// but slow enough that the gain does not audibly "duck" on every fricative
/// or plosive.  Higher values (≥0.30) cause audible gain pumping on
/// syllable boundaries; lower values (≤0.10) risk clipping on sudden
/// loud interjections.
const EMA_ATTACK_ALPHA: f32 = 0.20;

/// EMA release coefficient for RMS level tracking.
///
/// When the current chunk RMS is below the running estimate (speech offset
/// or transition to quieter speech), we adapt slowly (α=0.02) so that gain
/// does not "pump" (rapidly increase between syllables, amplifying background
/// noise).  The slow decay maintains a stable gain envelope across the
/// utterance.
///
/// Half-life: ln(0.5)/ln(1-0.02) ≈ 34 updates (~1.1s at 32ms/chunk).
/// Reaches 90% of decay target in: ln(0.1)/ln(1-0.02) ≈ 114 updates (~3.6s).
///
/// The value 0.02 is chosen so that the release time (~3.6s) spans an
/// entire utterance, maintaining consistent gain across syllables and
/// words.  Faster release (≥0.05) causes audible "breathing" — gain
/// increases detectably between words, amplifying background noise.
/// Slower release (≤0.01) prolongs the adaptation window unnecessarily,
/// delaying recovery after the speaker transitions from loud to quiet.
const EMA_RELEASE_ALPHA: f32 = 0.02;

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
    /// Running RMS level estimate for asymmetric EMA-based AGC.
    ///
    /// Initialised to [`TARGET_RMS`] so the first gain is 1.0× (neutral).
    /// Updated on every processed chunk using asymmetric EMA:
    /// - Fast attack (0.20) when chunk RMS exceeds running estimate
    /// - Slow release (0.02) when chunk RMS is below running estimate
    ///
    /// Reset to [`TARGET_RMS`] on [`clear_buffer()`] and [`reset()`] to
    /// prevent stale gain history (e.g. from a loud enrollment) from
    /// persisting into a different pipeline phase (e.g. live detection
    /// in a quieter environment).
    running_rms: f32,
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
            running_rms: TARGET_RMS,
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
            processed = self.apply_agc(processed);
        }

        processed
    }

    /// Clear the frame-alignment buffer without disturbing the noise
    /// suppressor's adapted noise profile.
    ///
    /// Called during pipeline state transitions (e.g., listening→recording)
    /// to flush stale samples without discarding the noise floor estimate
    /// the suppressor has built up over time.
    ///
    /// Also resets the EMA running RMS to [`TARGET_RMS`] so the AGC gain
    /// re-starts from 1.0× in the new pipeline phase.  Without this reset,
    /// stale gain history from a different acoustic environment (e.g.
    /// enrollment) could produce incorrect gain during live detection.
    ///
    /// # Convergence cost
    /// Resetting forces EMA re-adaptation from neutral (1.0× gain).
    /// Convergence time depends on the audio content of the new phase:
    /// - **Loud audio** (input RMS >> TARGET_RMS) converges in ~10 chunks
    ///   (~320ms at 32ms/chunk) via attack α=0.20.
    /// - **Quiet audio** (input RMS << TARGET_RMS) converges in ~114 chunks
    ///   (~3.6s) via release α=0.02.
    ///
    /// Pipeline state transitions at `clear_pipeline_buffers()` call sites
    /// (~11 locations) typically settle within 320ms because the new phase
    /// (listening, recording, enrollment) receives speech-level audio whose
    /// RMS is near TARGET_RMS.  The slow release case (~3.6s) only matters
    /// in the rare scenario where the new phase produces consistently quiet
    /// audio far below TARGET_RMS.
    pub fn clear_buffer(&mut self) {
        self.ns_buffer.clear();
        self.read_pos = 0;
        self.running_rms = TARGET_RMS;
    }

    /// Full reset: discard the sample buffer, the noise suppressor's adapted
    /// noise profile, and the AGC EMA state.
    ///
    /// Should be called when the microphone stream is re-created or the
    /// acoustic environment changes significantly, as the old noise profile
    /// may no longer be representative and the old gain envelope would
    /// mis-adapt to the new room acoustics.
    pub fn reset(&mut self) {
        self.ns_buffer.clear();
        self.read_pos = 0;
        self.running_rms = TARGET_RMS;
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

    /// Apply streaming EMA-based AGC to a chunk of audio.
    ///
    /// Uses asymmetric exponential moving average on the RMS level:
    ///
    /// | Condition | α   | Behavior   | 90% convergence |
    /// |-----------|-----|------------|-----------------|
    /// | Chunk RMS > running RMS (speech onset) | 0.20 | Fast attack — gain decreases quickly | ~10 chunks |
    /// | Chunk RMS ≤ running RMS (speech offset) | 0.02 | Slow release — gain increases slowly | ~114 chunks |
    ///
    /// This prevents gain pumping (loudness warble caused by AGC chasing
    /// every syllable boundary) while still reacting promptly to sudden
    /// loud speech that would otherwise clip.
    ///
    /// The gain is computed from the smoothed running RMS, not the raw
    /// chunk RMS, so natural speech amplitude variations (e.g., syllable
    /// stress) do not cause per-chunk gain oscillation.
    ///
    /// Unlike the previous stateless per-chunk AGC which computed gain
    /// independently for each mic chunk (causing chunk-to-chunk gain
    /// inconsistency — "gain pumping" — where natural speech amplitude
    /// variations across syllables produced audible loudness wobbles),
    /// this streaming version maintains a running RMS estimate that
    /// smooths out short-term fluctuations.  Both enrollment and detection
    /// already received identical stateless AGC processing (same `apply_agc`
    /// call on same-size mic chunks), so the improvement here is purely
    /// temporal smoothing rather than fixing a mode-specific asymmetry.
    #[allow(clippy::cast_precision_loss)]
    fn apply_agc(&mut self, samples: Vec<f32>) -> Vec<f32> {
        if samples.is_empty() {
            return samples;
        }

        let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
        let chunk_rms = (sum_sq / samples.len() as f32).sqrt();

        if chunk_rms == 0.0 {
            // Pure silence — freeze EMA state instead of decaying running_rms
            // toward zero.  Two failure modes are avoided:
            //
            // 1. Quiet-after-loud: If running_rms decayed during silence with
            //    release α=0.02, the gain would slowly ramp toward MAX_GAIN.
            //    Resuming speech would briefly clip before fast attack corrects
            //    it, amplifying the noise floor audibly.
            //
            // 2. Loud-after-quiet: If running_rms decayed to near zero during
            //    extended silence, the first loud chunk would receive MAX_GAIN
            //    (~4× amplification) because gain = TARGET_RMS / near-zero.
            //    Fast attack (α=0.20) would fix it within ~5 chunks, but the
            //    single over-amplified chunk could still clip.  Freezing at
            //    the pre-silence level avoids this transient entirely.
            return samples;
        }

        // Asymmetric EMA: fast attack on speech onset, slow release on offset
        let alpha = if chunk_rms > self.running_rms {
            EMA_ATTACK_ALPHA
        } else {
            EMA_RELEASE_ALPHA
        };
        self.running_rms = alpha * chunk_rms + (1.0 - alpha) * self.running_rms;

        let gain = (TARGET_RMS / self.running_rms).clamp(MIN_GAIN, MAX_GAIN);
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

    /// Test that EMA-based AGC normalises volume across multiple chunks.
    ///
    /// The asymmetric EMA adapts over time — a single chunk does not receive
    /// the full gain.  This test feeds many consecutive chunks and verifies
    /// that after convergence the output RMS approaches TARGET_RMS.
    ///
    /// Also verifies that the gain clamp prevents amplifying pure silence.
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
        let chunk_len = NS_FRAME_SIZE * 3; // 480 samples — typical mic chunk size

        // ── 0.25× target RMS (quiet speech) ──
        // Feed 200 chunks; with release α=0.02, running_rms should converge
        // to ~0.26 × TARGET_RMS, producing gain ~3.8× and output ≈ 0.95 × target.
        let amp_quiet = target_rms * 0.25 * sqrt2;
        for _ in 0..200 {
            let chunk = sine_tone(amp_quiet, chunk_len, 16_000);
            let _ = pre.process(chunk);
        }
        let final_chunk = sine_tone(amp_quiet, chunk_len * 3, 16_000);
        let processed_quiet = pre.process(final_chunk);
        let rms_quiet = compute_rms(&processed_quiet);
        let rel_err_quiet = (rms_quiet - target_rms).abs() / target_rms;
        assert!(
            rel_err_quiet < 0.15,
            "quiet AGC convergence: rms={rms_quiet:.6} target={target_rms} rel_err={rel_err_quiet:.4} (expected <0.15)"
        );

        // ── 4.0× target RMS (loud speech) ──
        // Reset preprocessor and feed 30 chunks; with attack α=0.20, running_rms
        // converges to ~4.0 × TARGET_RMS, gain clamped to 0.25 → output ≈ target.
        let mut pre = AudioPreprocessor::new(PreprocessorConfig {
            noise_suppression: false,
            agc: true,
        });
        let amp_loud = target_rms * 4.0 * sqrt2;
        for _ in 0..30 {
            let chunk = sine_tone(amp_loud, chunk_len, 16_000);
            let _ = pre.process(chunk);
        }
        let final_loud = sine_tone(amp_loud, chunk_len * 3, 16_000);
        let processed_loud = pre.process(final_loud);
        let rms_loud = compute_rms(&processed_loud);
        let rel_err_loud = (rms_loud - target_rms).abs() / target_rms;
        assert!(
            rel_err_loud < 0.15,
            "loud AGC convergence: rms={rms_loud:.6} target={target_rms} rel_err={rel_err_loud:.4} (expected <0.15)"
        );

        // ── Silence should not be amplified ──
        let silence = vec![0.0f32; chunk_len * 2];
        let processed_silence = pre.process(silence);
        let rms_silence = compute_rms(&processed_silence);
        assert!(
            rms_silence < 0.001,
            "silence should not be amplified: rms={rms_silence:.8}"
        );
    }

    /// Test that EMA AGC produces smooth gain transitions across chunk boundaries
    /// (no gain pumping).
    ///
    /// Feeds a quiet-quiet-loud-loud-quiet sequence and verifies that gain
    /// changes monotonically within each segment and the output RMS does not
    /// oscillate between extremes.
    #[test]
    fn test_agc_ema_stability() {
        let mut pre = AudioPreprocessor::new(PreprocessorConfig {
            noise_suppression: false,
            agc: true,
        });

        let target_rms = TARGET_RMS;
        let sqrt2 = std::f32::consts::SQRT_2;
        let chunk_len = NS_FRAME_SIZE * 3; // 480 samples

        // Phase 1: 15 chunks of quiet audio (0.25× target RMS).
        // With EMA release α=0.02, running_rms drops slowly, gain rises slowly.
        let amp_quiet = target_rms * 0.25 * sqrt2;
        let mut prev_gain = 1.0; // initial gain = 1.0 (running_rms = TARGET_RMS)
        for _ in 0..15 {
            let chunk = sine_tone(amp_quiet, chunk_len, 16_000);
            let processed = pre.process(chunk);
            let rms = compute_rms(&processed);
            // Gain is output_rms / input_rms. Input RMS ≈ amp_quiet / sqrt2 = 0.25 * target_rms.
            let input_rms = target_rms * 0.25;
            let gain = if input_rms > 0.0 {
                rms / input_rms
            } else {
                0.0
            };
            // Gain should increase monotonically (never decrease) during quiet
            assert!(
                gain >= prev_gain - 1e-6,
                "gain should not decrease during quiet phase: {gain:.6} < {prev_gain:.6}"
            );
            prev_gain = gain;
        }

        // Phase 2: 15 chunks of loud audio (4.0× target RMS).
        // With EMA attack α=0.20, running_rms rises quickly, gain falls quickly.
        let amp_loud = target_rms * 4.0 * sqrt2;
        for _ in 0..15 {
            let chunk = sine_tone(amp_loud, chunk_len, 16_000);
            let processed = pre.process(chunk);
            let rms = compute_rms(&processed);
            let input_rms = target_rms * 4.0;
            let gain = if input_rms > 0.0 {
                rms / input_rms
            } else {
                0.0
            };
            // Gain should decrease monotonically (never increase) during loud onsets
            assert!(
                gain <= prev_gain + 1e-6,
                "gain should not increase during loud phase: {gain:.6} > {prev_gain:.6}"
            );
            prev_gain = gain;
        }

        // After Phase 3 (10 more quiet chunks at release α=0.02), gain should
        // have risen above the Phase 2 low point (gain was ~0.25 at end of
        // Phase 2 loud segment).  The release is slow, but 10 chunks should
        // produce a measurable upward drift.
        let phase2_low = prev_gain; // gain after Phase 2 loud segment
        // Feed 40 more quiet chunks to give release time to act
        for _ in 0..40 {
            let chunk = sine_tone(amp_quiet, chunk_len, 16_000);
            let processed = pre.process(chunk);
            let rms = compute_rms(&processed);
            let input_rms = target_rms * 0.25;
            let gain = if input_rms > 0.0 {
                rms / input_rms
            } else {
                0.0
            };
            // Gain should increase monotonically (slowly, due to release α=0.02)
            assert!(
                gain >= prev_gain - 1e-6,
                "gain should not decrease during second quiet phase: {gain:.6} < {prev_gain:.6}"
            );
            prev_gain = gain;
        }

        // After 50 quiet chunks following loud speech, gain should have risen
        // measurably above the Phase 2 low point.
        assert!(
            prev_gain > phase2_low + 0.01,
            "after {} quiet chunks following loud speech, gain should rise measurably above Phase 2 low: \
             gain={prev_gain:.4} phase2_low={phase2_low:.4}",
            50,
        );
    }

    /// Test that noise suppression + AGC compose correctly (AGC after NS).
    ///
    /// Processing order matters: AGC must come AFTER noise suppression, not
    /// before — otherwise the NS would see amplified noise, making it harder
    /// to suppress.  This test verifies the composition produces a valid
    /// output with normalised RMS.
    ///
    /// Uses higher-amplitude noise (0.20) so the input RMS after NS is near
    /// TARGET_RMS, reducing the convergence time needed for the EMA-based AGC.
    #[test]
    fn test_agc_and_ns_compose() {
        let mut pre = AudioPreprocessor::new(PreprocessorConfig {
            noise_suppression: true,
            agc: true,
        });

        // Generate noisy audio at 0.20 amplitude.  White noise at this level
        // has RMS ~0.073 before NS.  After NS (K12dB suppression) the RMS
        // is reduced but typically remains near TARGET_RMS (0.05), so the
        // EMA-based AGC converges quickly.  The exact post-NS RMS depends on
        // the suppressor's internal adaptation and cannot be predicted with a
        // single number — the test assertion (0.5–1.5× of TARGET_RMS) captures
        // the acceptable range.
        let chunk_size = NS_FRAME_SIZE * 6;
        let mut all_output_rms = 0.0f32;
        let mut num_chunks = 0;

        for _ in 0..60 {
            let noise = white_noise(0.20, chunk_size);
            let processed = pre.process(noise);
            all_output_rms += compute_rms(&processed);
            num_chunks += 1;
        }

        let avg_rms = all_output_rms / num_chunks as f32;

        // RMS should be close to TARGET_RMS. After NS (12dB suppression) + EMA
        // AGC with 60 chunks at attack α=0.20, the EMA converges within ~10
        // chunks and the output stabilises near target.  The ±50% margin allows
        // for NS adaptation dynamics during the first few chunks.
        let ratio = avg_rms / TARGET_RMS;
        assert!(
            ratio > 0.5 && ratio < 1.5,
            "NS+AGC composition: avg_rms={avg_rms:.6} ratio={ratio:.4} (expected 0.5–1.5 of target={TARGET_RMS})"
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
