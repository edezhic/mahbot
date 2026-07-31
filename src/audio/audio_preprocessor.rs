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
#[cfg(feature = "voice-tests")]
use std::collections::VecDeque;

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
    /// Initialised to 0.0 (mahbot-856).  On the first non-zero chunk,
    /// [`apply_agc()`] sets this directly to the chunk RMS so the gain
    /// immediately reflects the actual audio level.  Lazy init eliminates
    /// the ~3.6s convergence delay that occurred with [`TARGET_RMS`]
    /// initialisation during quiet audio (e.g. volume-reduced variants at
    /// -6dB where RMS ≈ 0.025).  Updated via asymmetric EMA:
    /// - Fast attack (0.20) when chunk RMS exceeds running estimate
    /// - Slow release (0.02) when chunk RMS is below running estimate
    ///
    /// Reset to 0.0 on [`clear_buffer()`] and [`reset()`] so the AGC
    /// lazily initialises on the next speech frame (mahbot-856).
    /// Without this reset, stale gain history from a different acoustic
    /// environment (e.g. enrollment) could produce incorrect gain during
    /// live detection in a quieter/louder setting.
    pub(crate) running_rms: f32,

    /// History of `running_rms` values for AGC convergence detection
    /// (mahbot-886).  Bounded to the last 10 AGC-active frames.  Only
    /// compiled in `voice-tests` builds — zero production overhead.
    #[cfg(feature = "voice-tests")]
    running_rms_history: VecDeque<f32>,

    /// Count of AGC-active frames processed (frames where `apply_agc` ran
    /// on a non-silent chunk and actually updated `running_rms`).  Used by
    /// `agc_converged()` to determine whether enough audio was processed
    /// for the AGC to have reached steady state.
    #[cfg(feature = "voice-tests")]
    agc_active_frame_count: usize,
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
            // Initialise to 0.0 so the AGC lazily initialises on the first
            // non-zero frame (mahbot-856).  Initialising to TARGET_RMS caused
            // the gain to start at 1.0× for quiet audio (e.g. volume-reduced
            // variants at -6dB) and take ~114 frames (~3.6s) to converge to
            // the correct level via slow release — the utterance was already
            // over before meaningful amplification kicked in.  With lazy
            // initialisation the AGC immediately sets running_rms to the
            // actual frame level and applies the correct gain from frame 1.
            running_rms: 0.0,

            // Instrumentation fields for AGC convergence tracking (mahbot-886).
            // Feature-gated — zero runtime overhead in production builds.
            #[cfg(feature = "voice-tests")]
            running_rms_history: VecDeque::with_capacity(10),
            #[cfg(feature = "voice-tests")]
            agc_active_frame_count: 0,
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
    /// Also resets the EMA running RMS to 0.0 so the AGC lazily initialises
    /// on the next non-zero speech frame (mahbot-856).  Without this reset,
    /// stale gain history from a different acoustic environment (e.g.
    /// enrollment) could produce incorrect gain during live detection.
    ///
    /// # Convergence cost
    /// Resets the running RMS to 0.0 so the AGC lazily initialises on the
    /// next non-zero speech frame (mahbot-856).  Convergence is instantaneous
    /// — the gain on the first speech frame directly reflects TARGET_RMS /
    /// chunk_rms.  Unlike the old TARGET_RMS initialisation which required
    /// ~114 chunks (~3.6s) to converge from TARGET_RMS down to quiet audio,
    /// the lazy init approach converges in 1 frame regardless of loudness.
    pub fn clear_buffer(&mut self) {
        self.ns_buffer.clear();
        self.read_pos = 0;
        // Reset to 0.0 (lazy initialisation on next speech frame) instead of
        // TARGET_RMS, matching the constructor (mahbot-856).  Without this,
        // a pipeline reset after loud enrollment would leave running_rms at
        // TARGET_RMS while the next speech is quiet (or vice versa), causing
        // the same slow-convergence issue as the original TARGET_RMS init.
        self.running_rms = 0.0;
    }

    /// Full reset: discard the sample buffer, the noise suppressor's adapted
    /// noise profile, and the AGC EMA state.
    ///
    /// Should be called when the microphone stream is re-created or the
    /// acoustic environment changes significantly, as the old noise profile
    /// may no longer be representative and the old gain envelope would
    /// mis-adapt to the new room acoustics.
    ///
    /// Also clears the voice-tests AGC instrumentation (running RMS history and
    /// AGC-active frame count) so [`agc_converged`](Self::agc_converged)
    /// reflects only the post-reset audio — `reset()` is used at production
    /// segment boundaries and by the E2E benchmark between warm-up and test
    /// utterance (mahbot-1006 A); carrying the pre-reset history into the
    /// convergence report would mix two acoustic states.
    pub fn reset(&mut self) {
        self.ns_buffer.clear();
        self.read_pos = 0;
        self.running_rms = 0.0;
        self.suppressor = if self.config.noise_suppression {
            Some(NoiseSuppressor::with_level(SuppressionLevel::K12dB))
        } else {
            None
        };
        #[cfg(feature = "voice-tests")]
        {
            self.running_rms_history.clear();
            self.agc_active_frame_count = 0;
        }
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
    /// This avoids the "gain pumping" problem that would occur with a
    /// per-chunk AGC (where natural speech amplitude variations across
    /// syllables produce audible loudness wobbles) because the running
    /// RMS estimate smooths out short-term fluctuations.
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
            // 2. Loud-after-quiet: If running_rms was 0.0 (lazy init) and a pure
            //    silence chunk arrived before any speech, the gain calc below
            //    would divide by zero.  Early-return prevents this.
            return samples;
        }

        // Count AGC-active frames (mahbot-886).  Placed after the silence
        // guard so only non-silent chunks (where the AGC state actually
        // evolves) are counted.  Feature-gated — zero overhead in production.
        #[cfg(feature = "voice-tests")]
        {
            self.agc_active_frame_count += 1;
        }

        // Lazy initialisation: on the first non-zero chunk (pure silence
        // returns above), set running_rms directly to the chunk RMS so the
        // gain immediately reflects the actual audio level (mahbot-856).
        // Without this, a fresh AGC starting at 0.0 would need the EMA to
        // converge from zero upward — fast attack (0.20) reaches 90% in ~10
        // frames, which is acceptable but still introduces a momentary gain
        // ramp.  Direct initialisation is cleaner and avoids the ramp
        // entirely.
        if self.running_rms == 0.0 {
            self.running_rms = chunk_rms;
        } else {
            // Asymmetric EMA: fast attack on speech onset, slow release on offset
            let alpha = if chunk_rms > self.running_rms {
                EMA_ATTACK_ALPHA
            } else {
                EMA_RELEASE_ALPHA
            };
            self.running_rms = alpha * chunk_rms + (1.0 - alpha) * self.running_rms;
        }

        // Record running_rms for AGC convergence detection (mahbot-886).
        // Feature-gated — zero overhead in production.
        #[cfg(feature = "voice-tests")]
        {
            self.running_rms_history.push_back(self.running_rms);
            if self.running_rms_history.len() > 10 {
                self.running_rms_history.pop_front();
            }
        }

        let gain = (TARGET_RMS / self.running_rms).clamp(MIN_GAIN, MAX_GAIN);
        samples.iter().map(|&s| s * gain).collect()
    }

    /// Determine whether the AGC has converged to a stable gain level.
    ///
    /// Convergence is defined as the final frame's `running_rms` being within
    /// 5% of the running average of the last 10 AGC-active frames.
    ///
    /// Returns:
    /// - `Some(true)` if converged (enough AGC-active frames and stable RMS).
    /// - `Some(false)` if not converged (enough frames but RMS still moving).
    /// - `None` if insufficient data (< 20 AGC-active frames).
    ///
    /// Only available in `voice-tests` builds — zero production overhead.
    ///
    /// # Frame definition
    ///
    /// "AGC-active frame" means a call to [`apply_agc()`] on a non-silent chunk
    /// (where `chunk_rms > 0.0`).  Silence frames do not update `running_rms`
    /// and are excluded from the count.  At 32 ms per chunk, 20 AGC-active
    /// frames ≈ 640 ms of speech — the ASR EMA has proven time to reach 90%
    /// of steady state (~10 active frames for attack, ~114 for release) so
    /// 20 frames is a conservative minimum for evaluating convergence.
    /// Returns converged status: `Some(true)` if the AGC running_rms stabilized
    /// by utterance end (final frame's running_rms within 5% of the running
    /// average of the last 10 AGC-active frames), `Some(false)` if it did not,
    /// `None` if fewer than 20 AGC-active frames (insufficient data to
    /// determine convergence).
    ///
    /// An "AGC-active frame" is a non-silent chunk processed by `apply_agc`
    /// (i.e., `chunk_rms > 0.0`).  Silence frames are excluded from the count
    /// because they don't update `running_rms`.
    ///
    /// Used by the benchmark (mahbot-886) to correlate detection misses with
    /// convergence state.
    #[cfg(feature = "voice-tests")]
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn agc_converged(&self) -> Option<bool> {
        if self.agc_active_frame_count < 20 {
            // Fewer than 20 AGC-active frames = insufficient data to determine
            // convergence.  This includes very short variants or audio clips
            // where the AGC hasn't had time to reach steady state.
            return None;
        }
        if self.running_rms_history.len() < 10 {
            // Shouldn't happen if agc_active_frame_count >= 20 (history fills
            // after 10 AGC-active frames, and with 20+ frames we're guaranteed
            // a full history), but guard defensively.
            return None;
        }
        let final_rms = *self.running_rms_history.back().unwrap();
        let avg_10: f32 = self.running_rms_history.iter().copied().sum::<f32>()
            / self.running_rms_history.len() as f32;
        // avg_10 > 0.0 is guaranteed because every entry in the history comes
        // from an AGC-active frame (non-silent chunk), and running_rms is only
        // updated when chunk_rms > 0.0.
        debug_assert!(
            avg_10 > 0.0,
            "avg_10 should be > 0.0 since AGC only runs on non-silent chunks"
        );
        Some((final_rms - avg_10).abs() / avg_10 <= 0.05)
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
    /// With lazy initialisation (running_rms = 0.0, mahbot-856), the first
    /// non-zero chunk immediately sets running_rms to the chunk RMS, giving
    /// full gain from frame 1.  This test feeds 200 quiet chunks followed by
    /// 30 loud chunks (each with a fresh AGC state) and verifies that after
    /// many EMA iterations the output RMS approaches TARGET_RMS — the same
    /// steady-state that the old TARGET_RMS init produced, reached faster.
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
        // With lazy init, running_rms is set to chunk_rms on the first
        // non-zero chunk, giving gain = TARGET_RMS / 0.25*TARGET_RMS = 4.0×
        // from frame 1.  After 200 EMA iterations at the same level, the
        // running_rms stabilises at 0.25 × TARGET_RMS.  This produces the
        // same steady-state output as the old TARGET_RMS init but without
        // the ~3.6s convergence ramp (mahbot-856).
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
        // With lazy init, a fresh AGC sets running_rms to chunk_rms = 4.0 ×
        // TARGET_RMS on the first chunk, giving gain = 0.05/0.2 = 0.25×
        // (MIN_GAIN) from frame 1.  After 30 EMA iterations the running_rms
        // stays at ~4.0 × TARGET_RMS, maintaining gain at 0.25× and output
        // ≈ TARGET_RMS — converged from frame 1 rather than ramping over 30
        // chunks as with the old TARGET_RMS init (mahbot-856).
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
    /// Feeds a quiet-loud-quiet sequence and verifies that gain
    /// changes monotonically within each segment and the output RMS does not
    /// oscillate between extremes.
    ///
    /// Phase 1 (quiet) starts with lazy init (mahbot-856): running_rms = 0.0,
    /// set to chunk_rms = 0.25 × TARGET_RMS on the first non-zero frame,
    /// giving gain = 4.0× immediately.  This validates that lazy init produces
    /// a stable plateau without overshoot.  Phases 2→3 exercise the EMA release
    /// path (quiet-after-loud) which is the same as with the old TARGET_RMS init
    /// — the running_rms decays slowly from ~4.0 × TARGET_RMS toward 0.25 ×
    /// TARGET_RMS via release α=0.02.
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
        // With lazy init, running_rms is set to chunk_rms on the very first
        // non-zero chunk, so gain jumps to ~4.0× immediately.  After that the
        // running_rms stays at 0.25 × TARGET_RMS (same input level), so gain
        // stays at ~4.0× — no further increase is possible since we're already
        // at MAX_GAIN.  This phase validates that lazy init reaches a stable
        // plateau without overshoot or pumping.
        // Note: prev_gain is initialised from the first chunk's actual gain
        // (which may be slightly below 4.0 due to sine/FFT boundary effects),
        // then verified to stay stable for the remaining 14 chunks.
        let amp_quiet = target_rms * 0.25 * sqrt2;
        let first_chunk = sine_tone(amp_quiet, chunk_len, 16_000);
        let first_processed = pre.process(first_chunk);
        let first_rms = compute_rms(&first_processed);
        let input_rms = target_rms * 0.25;
        let mut prev_gain = if input_rms > 0.0 {
            first_rms / input_rms
        } else {
            0.0
        };
        for _ in 1..15 {
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
            // Gain stays at MAX_GAIN (cannot exceed 4.0, and same input level prevents decrease)
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
    /// TARGET_RMS, keeping the output within the 0.5–1.5× range despite the
    /// EMA attack dynamics.  With lazy init (mahbot-856), AGC converges in a
    /// single frame, so the test margin is more than sufficient even during
    /// the first few NS-adaptation chunks.
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

    /// Test that the silence-first guard prevents division by zero when the
    /// very first chunk is pure silence (mahbot-856 lazy init).
    ///
    /// With running_rms = 0.0 and a zero-RMS first chunk, the lazy-init
    /// branch is NOT entered (running_rms stays 0.0 after the early return
    /// for chunk_rms == 0.0).  The next non-zero chunk should then trigger
    /// lazy init and apply the correct gain immediately.
    #[test]
    fn test_lazy_init_silence_first() {
        let mut pre = AudioPreprocessor::new(PreprocessorConfig {
            noise_suppression: false,
            agc: true,
        });

        // Feed pure silence first — should not crash or divide by zero.
        let silence = vec![0.0f32; NS_FRAME_SIZE * 3]; // 480 samples
        let out = pre.process(silence);
        assert!(
            out.iter().all(|&s| s == 0.0),
            "silence should pass through unchanged"
        );

        // Now feed a speech-level chunk.  running_rms is still 0.0 (lazy init
        // was not triggered by the zero-RMS silence), so the lazy-init branch
        // fires and sets running_rms = chunk RMS, giving immediate gain.
        let target_rms = TARGET_RMS;
        let sqrt2 = std::f32::consts::SQRT_2;
        let chunk_len = NS_FRAME_SIZE * 3;
        let amp_quiet = target_rms * 0.25 * sqrt2;
        let speech = sine_tone(amp_quiet, chunk_len, 16_000);
        let processed = pre.process(speech);
        let rms = compute_rms(&processed);
        let rel_err = (rms - target_rms).abs() / target_rms;
        assert!(
            rel_err < 0.15,
            "speech after silence: rms={rms:.6} target={target_rms} rel_err={rel_err:.4} (expected <0.15)"
        );
    }

    /// Helper: exercises the lazy-init path after `cleanup` resets `running_rms` to 0.0.
    ///
    /// Phase 1: feed a quiet chunk to establish non-zero `running_rms`, then
    /// apply `cleanup` (which resets `running_rms` to 0.0 via [`clear_buffer`]
    /// or [`reset`]).  Phase 3: feed another quiet chunk — lazy init should
    /// fire and apply [`MAX_GAIN`] immediately.  With noise suppression
    /// disabled both cleanup paths are functionally equivalent (mahbot-856).
    fn run_lazy_init_after_cleanup_test(cleanup: fn(&mut AudioPreprocessor)) {
        let mut pre = AudioPreprocessor::new(PreprocessorConfig {
            noise_suppression: false,
            agc: true,
        });

        let target_rms = TARGET_RMS;
        let sqrt2 = std::f32::consts::SQRT_2;
        let chunk_len = NS_FRAME_SIZE * 3;

        // Phase 1: feed a quiet chunk to establish non-zero running_rms.
        let amp_quiet = target_rms * 0.25 * sqrt2;
        let chunk1 = sine_tone(amp_quiet, chunk_len, 16_000);
        let processed1 = pre.process(chunk1);
        let rms1 = compute_rms(&processed1);
        let rel_err1 = (rms1 - target_rms).abs() / target_rms;
        assert!(
            rel_err1 < 0.15,
            "Phase 1: rms={rms1:.6} target={target_rms} rel_err={rel_err1:.4}"
        );

        // Phase 2: apply the cleanup method.
        cleanup(&mut pre);

        // Phase 3: feed another quiet chunk.  With running_rms = 0.0, lazy
        // init fires and applies MAX_GAIN immediately.
        let chunk2 = sine_tone(amp_quiet, chunk_len, 16_000);
        let processed2 = pre.process(chunk2);
        let rms2 = compute_rms(&processed2);
        let rel_err2 = (rms2 - target_rms).abs() / target_rms;
        assert!(
            rel_err2 < 0.15,
            "Phase 3 after cleanup: rms={rms2:.6} target={target_rms} rel_err={rel_err2:.4}"
        );
    }

    /// Test that clear_buffer() resets running_rms to 0.0 so the next speech
    /// chunk triggers lazy init (mahbot-856).
    #[test]
    fn test_lazy_init_after_clear_buffer() {
        run_lazy_init_after_cleanup_test(|pre| pre.clear_buffer());
    }

    /// Test that reset() resets running_rms to 0.0 so the next speech chunk
    /// triggers lazy init (mahbot-856).  reset() also reinitialises the noise
    /// suppressor's internal state (not relevant when NS is disabled).
    #[test]
    fn test_lazy_init_after_reset() {
        run_lazy_init_after_cleanup_test(|pre| pre.reset());
    }
}
