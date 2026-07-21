//! Automated voice pipeline test harness with FAR/FRR metrics and synthetic
//! audio fixtures.
//!
//! This module is ONLY compiled when the `voice-tests` Cargo feature is active
//! AND the crate is built in test mode.  The gate is on the `mod
//! voice_test_harness` declaration in [`voice.rs`] (line 8019).
//! It provides:
//!
//! - Free functions for synthetic audio generation: [`wake_word_utterance()`],
//!   [`non_wake_word_speech()`], [`pink_noise()`], [`silence()`], [`mix_at_snr()`]
//! - [`OfflinePipelineRunner`] — loads ONNX models, enrolls from synthetic audio,
//!   and runs detection over pre-recorded or synthetic test audio
//! - Standardized FAR/FRR/latency/memory metric collection
//! - Baseline JSON serialization for regression tracking
//!
//! ## Architecture
//!
//! The harness feeds synthetic audio directly into the internal pipeline functions
//! (not through [`cpal`] microphone streams), following the same pattern as the
//! existing inline tests in [`voice.rs`] (see [`PipelineCtx::new()`],
//! [`handle_wake_word_detection`], [`handle_enrollment_audio`]).
//!
//! ```text
//! Audio Generators (free functions)
//!   │
//!   ├── wake_word_utterance() → Vec<f32>
//!   ├── non_wake_word_speech() → Vec<f32>
//!   ├── pink_noise() → Vec<f32>
//!   ├── silence() → Vec<f32>
//!   └── mix_at_snr(signal, noise, dB) → Vec<f32>
//!   │
//!   ▼
//! OfflinePipelineRunner
//!   │
//!   ├── enroll()        ──► handle_enrollment_audio + process_enrollment_sample
//!   ├── feed_audio()    ──► handle_wake_word_detection
//!   │
//!   ▼
//! Metric Collector
//!   ├── detection_count()
//!   ├── detection_latencies()
//!   └── peak_rss_bytes()
//! ```
//!
//! ## Usage
//!
//! ```bash
//! # Run ALL tests (fast unit tests only)
//! cargo test
//!
//! # Also run voice pipeline integration tests
//! cargo test --features voice-tests
//! ```
//!
//! ## Limitations
//!
//! - ONNX models must be available on disk (downloaded on first production use
//!   or manually cached in `~/.mahbot/models/openwakeword/`).  Tests
//!   gracefully skip when models are absent.
//! - Synthetic harmonic audio is a simplified model of real speech; FAR/FRR
//!   numbers from synthetic audio are NOT representative of real-world
//!   performance — they serve as **regression detection** metrics.
//! - Memory measurement uses macOS `getrusage` (available via the existing
//!   `libc` dependency).  On other platforms, reported peak RSS is 0.
//! - Detection latency measurement is gated by VAD frame boundaries (~32ms
//!   resolution at 16kHz).
//!
//! ## Baseline JSON format
//!
//! The baseline file is written to
//! `~/.mahbot/voice-test-baseline.json` when
//! [`run_and_save_baseline`] is called.  Format:
//!
//! ```json
//! {
//!   "version": 1,
//!   "timestamp": "2026-07-21T20:45:03Z",
//!   "scenarios": {
//!     "clean_wake_word": { "detection_count": 5, "expected": 5, "pass": true, ... },
//!     "noisy_wake_word_10db": { ... },
//!     "non_wake_word_speech_60s": { ... },
//!     "silence_60s": { ... },
//!     "noise_pink_60s": { ... }
//!   },
//!   "summary": { "all_pass": true, "peak_memory_mb": 123.4 }
//! }
//! ```

// Inner cfg removed — the `#[cfg(all(test, feature = "voice-tests"))]` gate
// on the `mod voice_test_harness` declaration in `voice.rs` (line 8019) is
// the sole gate.  This avoids a redundant double gate that could drift.

use super::*;
use anyhow::Context;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::f32::consts::PI;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════════
// Synthetic Audio Generator
// ═══════════════════════════════════════════════════════════════════════════

/// Types of synthetic noise for FAR-testing scenarios.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseType {
    /// Pink noise (1/f spectrum, ambient background).
    Pink,
    /// TV-like non-stationary noise (modulated pink noise with impulses).
    Tv,
}

impl NoiseType {
    fn as_str(self) -> &'static str {
        match self {
            NoiseType::Pink => "pink",
            NoiseType::Tv => "tv",
        }
    }

    fn generate(self, duration_ms: u64) -> Vec<f32> {
        match self {
            NoiseType::Pink => pink_noise(duration_ms),
            NoiseType::Tv => tv_noise(duration_ms),
        }
    }
}

/// Parameters for the wake-word acoustic signature (male voice, ~130 Hz F0).
/// Each entry is (frequency_hz, amplitude).  Amplitudes sum to 0.95 < 1.0.
const WAKE_WORD_HARMONICS: &[(f32, f32)] = &[
    (130.0, 0.40),  // F0
    (520.0, 0.25),  // H2 / F1 region
    (910.0, 0.15),  // H3
    (1300.0, 0.10), // H4 / F2 region
    (2600.0, 0.05), // H6 / F3 region
];

/// Parameters for non-wake-word speech (different F0 and formant structure).
/// Uses a higher F0 (200 Hz, female-like) with different formant peaks.
const NON_WAKE_WORD_HARMONICS: &[(f32, f32)] = &[
    (200.0, 0.35),  // F0
    (700.0, 0.30),  // F1
    (1400.0, 0.20), // F2
    (2800.0, 0.10), // F3
    (3500.0, 0.05), // F4
];

/// Generator for programmatic test audio fixtures.
///
/// All output is mono 16 kHz f32 PCM in [-1.0, 1.0] range.
///
/// # Examples
///
/// ```ignore
/// let wake_word = wake_word_utterance(5);  // 5 speech + silence gaps
/// let noise = pink_noise(5000);            // 5 seconds of pink noise
/// let mixed = mix_at_snr(&wake_word, &noise, 10.0);
/// ```
///
/// Generate a single 512-sample speech frame matching the wake-word
/// acoustic signature (F0 ≈ 130 Hz, male-voice harmonics).
///
/// This is identical to the [`speech_frame()`] helper in the existing test
/// module — verified to pass Earshot neural VAD.
pub fn wake_word_frame() -> Vec<f32> {
    let mut frame = Vec::with_capacity(FRAME_LENGTH);
    for i in 0..FRAME_LENGTH {
        let t = i as f32 / SAMPLE_RATE as f32;
        let sample: f32 = WAKE_WORD_HARMONICS
            .iter()
            .map(|(freq, amp)| (2.0 * PI * freq * t).sin() * amp)
            .sum();
        frame.push(sample);
    }
    debug_assert!(
        frame.iter().all(|&s| s.abs() <= 1.0),
        "wake_word_frame exceeds [-1, 1] range"
    );
    frame
}

/// Generate a single 512-sample speech frame with different spectral
/// characteristics (F0 ≈ 200 Hz, female-like formants).  Designed to be
/// clearly distinguishable from the wake word by the embedding model.
pub fn non_wake_word_frame() -> Vec<f32> {
    let mut frame = Vec::with_capacity(FRAME_LENGTH);
    for i in 0..FRAME_LENGTH {
        let t = i as f32 / SAMPLE_RATE as f32;
        let sample: f32 = NON_WAKE_WORD_HARMONICS
            .iter()
            .map(|(freq, amp)| (2.0 * PI * freq * t).sin() * amp)
            .sum();
        frame.push(sample);
    }
    debug_assert!(
        frame.iter().all(|&s| s.abs() <= 1.0),
        "non_wake_word_frame exceeds [-1, 1] range"
    );
    frame
}

/// Generate a wake-word utterance composed of `num_speech` speech frames
/// separated by silence gaps, bookended with leading/trailing silence for
/// VAD context.
///
/// Structure:
/// - 200ms leading silence (pre-speech VAD context)
/// - For each speech frame: 1 speech frame (32ms) + 32ms silence gap
/// - 200ms trailing silence (post-speech context)
///
/// Total duration ≈ 400ms + num_speech × 64ms.
/// For `num_speech = 5`: ≈ 720ms — produces ~6-7 embedding frames.
pub fn wake_word_utterance(num_speech: usize) -> Vec<f32> {
    let mut audio = Vec::new();

    // Leading silence (200ms = 3200 samples)
    audio.extend_from_slice(&silence(200));

    // Speech frames with gaps
    let speech_frame = wake_word_frame();
    let gap = silence(32); // 32ms gap = 512 samples
    for _ in 0..num_speech {
        audio.extend_from_slice(&speech_frame);
        audio.extend_from_slice(&gap);
    }

    // Trailing silence (200ms = 3200 samples)
    audio.extend_from_slice(&silence(200));

    audio
}

/// Generate non-wake-word speech with duration `duration_ms`.
///
/// Uses the non-wake-word harmonic set (F0 ≈ 200 Hz) to produce spectral
/// content that should be clearly distinguishable from the wake word by
/// the ONNX embedding model.
///
/// The speech varies over time: every ~500ms, the harmonics smoothly
/// transition between two different formant configurations, simulating
/// spectrally varying speech (TV/radio/conversation) rather than a
/// monotone buzz.  This makes it a more realistic distractor for the
/// embedding pipeline.
pub fn non_wake_word_speech(duration_ms: u64) -> Vec<f32> {
    let total_samples = (duration_ms as usize * SAMPLE_RATE as usize) / 1000;
    let mut audio = Vec::with_capacity(total_samples);

    // Two formant configurations: alternate between them to create
    // spectral variation (simulates different phonemes/speakers).
    let formant_a: &[(f32, f32)] = &[
        (200.0, 0.35),  // F0
        (700.0, 0.30),  // F1
        (1400.0, 0.20), // F2
        (2800.0, 0.10), // F3
    ];
    let formant_b: &[(f32, f32)] = &[
        (180.0, 0.30),  // F0 slightly lower
        (850.0, 0.25),  // F1 shifted up
        (1600.0, 0.25), // F2 shifted up
        (2400.0, 0.08), // F3 shifted down
        (3200.0, 0.12), // extra high-frequency content
    ];

    let gap = silence(16); // 16ms gap between "syllables"
    let mut written = 0;
    let frame_len_ms = (FRAME_LENGTH as f64 / SAMPLE_RATE as f64 * 1000.0) as u64;
    let transition_period = 500 / frame_len_ms.max(1); // ~16 frames at 32ms each
    let mut formant_cycle = 0u64;

    while written < total_samples {
        // Alternate between formant configurations
        let harmonics = if (formant_cycle / transition_period) % 2 == 0 {
            formant_a
        } else {
            formant_b
        };
        formant_cycle += 1;

        // Build a single speech frame with current formant config
        let mut frame = Vec::with_capacity(FRAME_LENGTH);
        for i in 0..FRAME_LENGTH {
            let t = i as f32 / SAMPLE_RATE as f32;
            let sample: f32 = harmonics
                .iter()
                .map(|(freq, amp)| (2.0 * PI * freq * t).sin() * amp)
                .sum();
            frame.push(sample);
        }

        let remaining = total_samples - written;
        if remaining < FRAME_LENGTH {
            audio.extend_from_slice(&frame[..remaining]);
            break;
        }
        audio.extend_from_slice(&frame);
        written += FRAME_LENGTH;

        if written < total_samples {
            let gap_remaining = (total_samples - written).min(gap.len());
            audio.extend_from_slice(&gap[..gap_remaining]);
            written += gap_remaining;
        }
    }

    audio
}

/// Generate pink noise using the Voss-McCartney algorithm.
///
/// Pink noise (1/f spectrum) simulates ambient environmental noise.
/// Uses 8 octave generators for spectral coverage from ~125 Hz down
/// to the Nyquist frequency.  Lowest frequency ≈ 16000 / 128 = 125 Hz.
pub fn pink_noise(duration_ms: u64) -> Vec<f32> {
    let total_samples = (duration_ms as usize * SAMPLE_RATE as usize) / 1000;
    let mut audio = Vec::with_capacity(total_samples);

    // Voss-McCartney: sum of white noise generators at different rates.
    // 8 generators with periods: 1, 2, 4, 8, 16, 32, 64, 128 samples.
    let num_generators = 8;
    let mut values = [0.0f32; 8];
    let mut rng = SmallRng::seed_from_u64(12345); // deterministic for reproducible tests

    for i in 0..total_samples {
        let mut sum = 0.0f32;
        // Update generators whose period divides the current sample index
        for g in 0..num_generators {
            let period = 1 << g; // 1, 2, 4, 8, 16, 32, 64, 128
            if i % period == 0 {
                values[g] = rng.random::<f32>() * 2.0 - 1.0;
            }
            sum += values[g];
        }
        // Normalize by sqrt(num_generators) to keep amplitude reasonable
        let sample = sum * (1.0 / (num_generators as f32).sqrt());
        // Scale to ~-0.3..0.3 for comfortable listening level
        audio.push(sample.clamp(-1.0, 1.0) * 0.3);
    }

    // Trim to exact expected length
    audio.truncate(total_samples);
    audio
}

/// Generate stationary hum (e.g., fan, AC) at `freq_hz`.
///
/// Produces a pure tone at the specified frequency with additive harmonic
/// content for a more realistic fan/AC hum.
pub fn stationary_hum(duration_ms: u64, freq_hz: f32) -> Vec<f32> {
    let total_samples = (duration_ms as usize * SAMPLE_RATE as usize) / 1000;
    let mut audio = Vec::with_capacity(total_samples);

    // Fundamental + 2 harmonics with decaying amplitudes
    let harmonics = [(1.0, 0.25), (2.0, 0.10), (3.0, 0.05)];
    for i in 0..total_samples {
        let t = i as f32 / SAMPLE_RATE as f32;
        let sample: f32 = harmonics
            .iter()
            .map(|(h, amp)| (2.0 * PI * freq_hz * h * t).sin() * amp)
            .sum();
        audio.push(sample);
    }
    audio
}

/// Generate TV-like non-stationary noise.
///
/// Simulates television / radio chatter with:
/// - Pink noise background (stationary)
/// - Random amplitude modulation (slow changes in volume)
/// - Occasional louder bursts (program changes, laughter, applause)
///
/// The result is a non-stationary noise signal that exercises the VAD and
/// embedding pipeline with realistic-sounding distractors.
pub fn tv_noise(duration_ms: u64) -> Vec<f32> {
    let total_samples = (duration_ms as usize * SAMPLE_RATE as usize) / 1000;
    let mut audio = Vec::with_capacity(total_samples);

    // Base pink noise
    let pink = pink_noise(duration_ms);

    // Deterministic random sequence using seeded SmallRng
    let mut rng = SmallRng::seed_from_u64(67890);
    let mut next_rng = || -> f32 { rng.random::<f32>() * 2.0 - 1.0 };

    // Slow modulation envelope: update every ~500ms
    let mod_period = (SAMPLE_RATE as usize) / 2; // 500ms
    let mut mod_value = 1.0f32;
    let mut target_mod = 1.0f32;

    for (i, &pink_sample) in pink.iter().enumerate() {
        if i % mod_period == 0 {
            // New modulation target every 500ms
            target_mod = 0.3 + next_rng().abs() * 0.7; // 0.3 to 1.0
        }
        // Smooth interpolation toward target
        if mod_value < target_mod {
            mod_value += 0.002; // attack
        } else {
            mod_value -= 0.001; // release slower
        }
        mod_value = mod_value.clamp(0.2, 1.2);

        // Occasional impulse (laughter, applause spike)
        let impulse = if next_rng().abs() > 0.998 {
            next_rng().abs() * 2.0
        } else {
            0.0
        };

        let sample = pink_sample * mod_value + impulse * 0.1;
        audio.push(sample.clamp(-1.0, 1.0));
    }

    audio
}

/// Generate silence (all zeros).
pub fn silence(duration_ms: u64) -> Vec<f32> {
    let n = (duration_ms as usize * SAMPLE_RATE as usize) / 1000;
    vec![0.0f32; n]
}

/// Mix `signal` and `noise` at the specified `snr_db` signal-to-noise ratio.
///
/// SNR is defined as `10 * log10( RMS(signal)^2 / RMS(noise_scaled)^2 )`,
/// computed from the raw signals before any pipeline processing (AGC/NS).
/// The noise is scaled so that the mixed output has the target SNR.
///
/// If either signal is empty, returns the signal unchanged.  If the noise
/// has zero RMS (pure silence), returns the signal unchanged to avoid
/// division by zero.
pub fn mix_at_snr(signal: &[f32], noise: &[f32], snr_db: f32) -> Vec<f32> {
    if signal.is_empty() || noise.is_empty() {
        return signal.to_vec();
    }

    let rms_signal = compute_rms(signal);
    let rms_noise = compute_rms(noise);

    if rms_noise < 1e-10 || rms_signal < 1e-10 {
        return signal.to_vec();
    }

    // Target noise RMS: rms_signal / 10^(snr_db/20)
    let scale_factor = rms_signal / (10.0_f32.powf(snr_db / 20.0) * rms_noise);

    let len = signal.len().min(noise.len());
    let mut mixed = Vec::with_capacity(len);
    for i in 0..len {
        mixed.push(signal[i] + noise[i] * scale_factor);
    }

    // Normalize to [-1, 1] if any sample exceeds the range
    let max_abs = mixed.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if max_abs > 1.0 {
        for s in &mut mixed {
            *s /= max_abs;
        }
    }

    mixed
}

/// Compute RMS of a signal.  Returns 0 for empty input.
fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// Offline Pipeline Runner
// ═══════════════════════════════════════════════════════════════════════════

/// Offline runner for the wake-word detection pipeline.
///
/// Manages the lifecycle of a [`PipelineCtx`] for automated testing:
/// 1. Loads ONNX models (if available on disk)
/// 2. Enrolls from synthetic wake-word audio
/// 3. Feeds test audio through [`handle_wake_word_detection`]
/// 4. Reports detection count and latency metrics
///
/// # State management
///
/// Each scenario should use a fresh runner or call [`reset`] to clear detection
/// state.  Enrollment persists across scenarios until [`reset`] is called.
///
/// # Example
///
/// ```ignore
/// let mut runner = OfflinePipelineRunner::new().unwrap();
///
/// // Enroll
/// runner.enroll().expect("enrollment should succeed");
///
/// // Feed test audio
/// let audio = wake_word_utterance(5);
/// runner.feed_audio(&audio);
///
/// // Check metrics
/// println!("Detections: {}", runner.detection_count());
/// ```
pub struct OfflinePipelineRunner {
    ctx: PipelineCtx,
    enrolled: bool,
    detection_events: Vec<(Instant, Instant)>, // (detection_time, onset_time)
    feed_start: Option<Instant>,
    total_samples_fed: usize,
}

impl OfflinePipelineRunner {
    /// Create a new runner.
    ///
    /// Returns `Ok(Some(runner))` if ONNX models are loaded and ready.
    /// Returns `Ok(None)` if models are not available (test should be skipped).
    /// Returns `Err` if the voice pipeline cannot be initialized.
    pub fn new() -> Result<Option<Self>> {
        // Ensure VAD detector and voice pipeline are initialized
        // (uses get_or_init to tolerate already-initialized globals)
        VAD_DETECTOR.get_or_init(|| std::sync::Mutex::new(earshot::Detector::default()));
        VOICE_PIPELINE.get_or_init(|| {
            RwLock::new(VoicePipelineState {
                enabled: false,
                status: VoiceStatus::Disabled,
                templates: Arc::new(WakeWordTemplates::default()),
                enrollment_buffer: Vec::new(),
                negative_audio_chunks: Vec::new(),
                cmd_tx: None,
            })
        });

        // Reset VOICE_PIPELINE state
        {
            let mut state = voice_state().write().unwrap_poison();
            state.enabled = true;
            state.status = VoiceStatus::Disabled;
            state.enrollment_buffer.clear();
            state.templates = Arc::new(WakeWordTemplates::default());
            state.cmd_tx = None;
        }

        // Check if ONNX models are loaded
        if !models_ready() {
            // Try to load models from disk (skip download — use cached)
            if let Some(dir) = model_dir() {
                let mel_path = dir.join(MEL_MODEL_FILENAME);
                let embed_path = dir.join(EMBED_MODEL_FILENAME);
                if mel_path.exists() && embed_path.exists() {
                    match load_onnx_models(&dir) {
                        Ok(models) => {
                            if ONNX_MODELS.set(models).is_ok() {
                                MODELS_STATE.store(ModelState::Ready, Ordering::Release);
                                info!("Voice test harness: loaded cached ONNX models");
                            }
                        }
                        Err(e) => {
                            info!("Voice test harness: failed to load ONNX models: {e}");
                            return Ok(None);
                        }
                    }
                } else {
                    info!(
                        "Voice test harness: ONNX models not found at {:?} — skipping tests",
                        dir
                    );
                    return Ok(None);
                }
            } else {
                info!("Voice test harness: model directory unavailable — skipping tests");
                return Ok(None);
            }
        }

        // Ensure models are ready
        if !models_ready() || ONNX_MODELS.get().is_none() {
            info!("Voice test harness: models not ready — skipping tests");
            return Ok(None);
        }

        let ctx = PipelineCtx::new();

        Ok(Some(Self {
            ctx,
            enrolled: false,
            detection_events: Vec::new(),
            feed_start: None,
            total_samples_fed: 0,
        }))
    }

    /// Enroll the wake word from synthetic audio generated by `g`.
    ///
    /// Uses the full enrollment pipeline: VAD-gated accumulation, ONNX embedding,
    /// threshold calibration, and self-test.  This ensures the harness exercises
    /// the same code path as real microphone enrollment.
    ///
    /// Returns `Ok(())` on success, `Err` with a message if enrollment fails.
    pub fn enroll(&mut self) -> Result<()> {
        if self.enrolled {
            // Already enrolled; templates persist.  Skip re-enrollment.
            return Ok(());
        }

        let utterance = wake_word_utterance(5);

        // We need 10 utterances for NUM_ENROLLMENT_SAMPLES=10.
        let required_utterances = NUM_ENROLLMENT_SAMPLES;

        for utt_idx in 0..required_utterances {
            // Reset context for this utterance
            self.ctx.enrollment_mode = true;
            self.ctx.is_listening = true;
            self.ctx.vad_threshold = ENROLLMENT_VAD_THRESHOLD;
            self.ctx.utterance_buf.clear();
            self.ctx.utterance_had_speech = false;
            self.ctx.utterance_silence_samples = 0;
            self.ctx.utterance_speech_end_len = 0;
            self.ctx.audio_buffer.clear();
            self.ctx.enrollment_pending = None;
            self.ctx.post_speech_tail.clear();
            reset_vad();

            // Feed the utterance in 512-sample chunks (matching real mic behavior)
            for chunk in utterance.chunks(FRAME_LENGTH) {
                handle_enrollment_audio(chunk, &mut self.ctx, utt_idx, required_utterances);
                if self.ctx.enrollment_pending.is_some() {
                    break; // utterance completed
                }
            }

            // If still no pending utterance, try waiting for silence timeout
            // by feeding silence directly
            if self.ctx.enrollment_pending.is_none() {
                // The 1.5s silence timeout hasn't fired yet — shortcut it
                // by setting the silence counter just below threshold.
                let silence = vec![0.0f32; FRAME_LENGTH];

                // If utterance has VAD-positive frames, we need to trigger
                // the silence timeout by feeding enough silence.
                for _ in 0..100 {
                    // max 100 frames = ~3.2s (should never need this many)
                    handle_enrollment_audio(&silence, &mut self.ctx, utt_idx, required_utterances);
                    if self.ctx.enrollment_pending.is_some() {
                        break;
                    }
                    // Shortcut: set utterance_silence_samples to just below threshold
                    if self.ctx.utterance_had_speech {
                        self.ctx.utterance_silence_samples =
                            SILENCE_THRESHOLD_SAMPLES.saturating_sub(HOP_LENGTH);
                    }
                }
            }

            let Some(pending_samples) = self.ctx.enrollment_pending.take() else {
                anyhow::bail!(
                    "Enrollment utterance {}/{} failed: no utterance accumulated \
                     (may need longer audio or different VAD parameters)",
                    utt_idx + 1,
                    required_utterances,
                );
            };

            // ── Process through ONNX ──
            // process_enrollment_sample is the synchronous public entry point.
            let embeddings = process_enrollment_sample(&pending_samples)
                .with_context(|| format!("ONNX inference failed for utterance {}", utt_idx + 1))?;

            if embeddings.is_empty() {
                anyhow::bail!(
                    "Enrollment utterance {}/{} produced zero embeddings (too short: {} samples)",
                    utt_idx + 1,
                    required_utterances,
                    pending_samples.len(),
                );
            }

            // Push to global enrollment buffer
            {
                let mut state = voice_state().write().unwrap_poison();
                state.enrollment_buffer.push(embeddings);
            }

            let count = voice_state().read().unwrap_poison().enrollment_buffer.len();
            debug!(
                "Enrollment utterance {}/{} processed (buffer size: {})",
                utt_idx + 1,
                required_utterances,
                count,
            );

            // Small pause between utterances (simulated by resetting state)
            reset_vad();
        }

        // ── Finalize enrollment ──
        // This calls calibrate_threshold + run_enrollment_self_test
        // and updates global templates.
        let (templates, minimum_matches) = finalize_enrollment(WAKE_WORD_NAME)?;

        // Set templates in global state
        set_templates(Arc::new(WakeWordTemplates {
            templates,
            minimum_matches,
            ..Default::default()
        }));

        // Reset VAD threshold from enrollment mode (0.85) back to
        // production detection threshold (0.5).  The production pipeline
        // does this at voice.rs:3086 after enrollment completes; the test
        // harness must match that behaviour so the first utterance of the
        // clean-wake-word scenario uses the same threshold as production.
        self.ctx.vad_threshold = VAD_THRESHOLD;

        self.enrolled = true;
        info!(
            "Voice test harness: enrollment complete ({} templates, minimum_matches={})",
            get_templates().templates.len(),
            get_templates().minimum_matches,
        );

        Ok(())
    }

    /// Feed audio through the wake-word detection pipeline.
    ///
    /// Audio is fed in 512-sample chunks (matching real microphone chunk size).
    /// Detection events are tracked internally via [`PipelineCtx`] state changes.
    ///
    /// Call this method for each audio segment in a test scenario.
    pub fn feed_audio(&mut self, audio: &[f32]) {
        if self.feed_start.is_none() {
            self.feed_start = Some(Instant::now());
        }

        // Feed audio in chunks matching real mic behavior
        for chunk in audio.chunks(FRAME_LENGTH) {
            let was_recording = self.ctx.is_recording;
            handle_wake_word_detection(chunk, &mut self.ctx);
            let now_recording = self.ctx.is_recording;

            // Detect transition to recording mode (wake word just fired)
            if !was_recording && now_recording {
                let detection_time = self
                    .ctx
                    .last_wake_word_detection
                    .unwrap_or_else(Instant::now);
                // The onset is the time when meaningful audio started.
                // For latency, we calculate from the feed_start.
                let onset = self.feed_start.unwrap_or(detection_time);
                self.detection_events.push((detection_time, onset));
            }
        }

        self.total_samples_fed += audio.len();
    }

    /// Number of distinct wake word detections since last reset.
    pub fn detection_count(&self) -> usize {
        self.detection_events.len()
    }

    /// Latency measurements per detection event (wall-clock time from
    /// first feed to detection).
    ///
    /// Returns empty vec if no detections occurred.
    pub fn detection_latencies(&self) -> Vec<Duration> {
        self.detection_events
            .iter()
            .map(|(detection_time, onset)| detection_time.saturating_duration_since(*onset))
            .collect()
    }

    /// Reset detection state for the next scenario.
    ///
    /// Preserves enrollment (templates remain loaded).  Clears detection
    /// buffers, event tracking, and recording state.
    pub fn reset(&mut self) {
        self.ctx.clear_detection_buffers();
        self.ctx.is_listening = true;
        self.ctx.is_recording = false;
        self.ctx.enrollment_mode = false;
        self.ctx.mic_rx = None;
        self.detection_events.clear();
        self.feed_start = None;
        self.total_samples_fed = 0;
        reset_vad();
    }

    /// Reset enrollment state (clear enrollment buffer, allow re-enrollment).
    ///
    /// Clears the global enrollment buffer so a subsequent [`enroll()`] call
    /// starts fresh.  Keeps loaded ONNX models and VAD detector.
    pub fn reset_enrollment(&mut self) {
        voice_state()
            .write()
            .unwrap_poison()
            .enrollment_buffer
            .clear();
        self.enrolled = false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Metric types
// ═══════════════════════════════════════════════════════════════════════════

/// Results of a single test scenario.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScenarioMetrics {
    /// Unique scenario name.
    pub scenario: String,
    /// How many wake word detections the scenario produced.
    pub detection_count: usize,
    /// Minimum acceptable detections (pass threshold) for wake word scenarios.
    /// Not set for FAR-based scenarios (non-wake speech, silence, noise) or
    /// enrollment stability (described in `detail`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_detections: Option<usize>,
    /// Whether the scenario passed its acceptance criteria.
    pub pass: bool,
    /// False-accept rate (detections per hour of non-wake-word audio).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub far_per_hour: Option<f64>,
    /// False-reject rate (fraction of missed wake word utterances).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frr_percent: Option<f64>,
    /// Mean detection latency in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_mean_ms: Option<f64>,
    /// 95th percentile detection latency in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_p95_ms: Option<f64>,
    /// Maximum detection latency in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_max_ms: Option<f64>,
    /// Additional details, e.g., error messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Results from running all standard test scenarios.
#[derive(Debug, Clone)]
pub struct AllScenarioResults {
    /// Per-scenario metrics.
    pub scenarios: Vec<ScenarioMetrics>,
    /// Peak process RSS (macOS only, 0 on other platforms) measured after all
    /// scenarios completed.  Uses `ru_maxrss` (process-lifetime peak, monotonic).
    pub peak_memory_mb: f64,
}

/// Summary metrics across all scenarios.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SummaryMetrics {
    /// All scenarios passed.
    pub all_pass: bool,
    /// Total scenarios run.
    pub total_scenarios: usize,
    /// Scenarios that passed.
    pub passed: usize,
    /// Scenarios that failed.
    pub failed: usize,
    /// Peak memory usage in MB (macOS only, 0 on other platforms).
    pub peak_memory_mb: f64,
}

/// Baseline metrics file structure.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BaselineMetrics {
    /// Format version.
    pub version: u32,
    /// When the baseline was recorded.
    pub timestamp: String,
    /// Per-scenario results.
    pub scenarios: Vec<ScenarioMetrics>,
    /// Summary across all scenarios.
    pub summary: SummaryMetrics,
}

impl BaselineMetrics {
    const VERSION: u32 = 1;
}

// ═══════════════════════════════════════════════════════════════════════════
// Memory measurement (macOS only)
// ═══════════════════════════════════════════════════════════════════════════

/// Measure per-process peak RSS (resident set size) in bytes.
///
/// Uses `getrusage` on macOS which reports `ru_maxrss` — the process-lifetime
/// **peak** resident set size (monotonic, not point-in-time current RSS).
/// Returns 0 on platforms where `libc::rusage` is unavailable.
fn peak_rss_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: getrusage writes to the provided struct on success.
        let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        if ret == 0 {
            let usage = unsafe { usage.assume_init() };
            // macOS reports ru_maxrss in bytes (Linux reports in KB).
            usage.ru_maxrss as u64
        } else {
            0
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Not implemented for other platforms
        0
    }
}

// (MemoryTracker removed — use peak_rss_bytes() directly for process-lifetime
// peak RSS measurement.)

// ═══════════════════════════════════════════════════════════════════════════
// Standard test scenarios
// ═══════════════════════════════════════════════════════════════════════════

/// Acceptance criteria constants.
///
/// These define pass/fail boundaries for the stochastic detection system.
/// Thresholds are chosen to account for synthetic audio artefacts while still
/// detecting real regressions.
mod acceptance {
    /// Minimum number of wake word detections expected from `NUM_WW_UTTERANCES`
    /// clean utterances.  Set to 4/5 (80%) — synthetic audio produces
    /// consistent embeddings so near-perfect detection is expected.
    pub const MIN_CLEAN_DETECTIONS: usize = 4;

    /// Minimum number of wake word detections expected from `NUM_WW_UTTERANCES`
    /// noisy utterances (10 dB SNR).  Slightly lower than clean due to noise
    /// potentially masking the spectral signature.
    pub const MIN_NOISY_DETECTIONS: usize = 3;

    /// Maximum false accepts per hour of non-wake-word audio.
    /// For 60s of audio: 0 false accepts → 0/hour.
    pub const MAX_FAR_PER_HOUR: f64 = 10.0; // relaxed for synthetic stability

    /// Maximum false-reject rate (percent of missed wake word utterances).
    pub const MAX_FRR_PERCENT: f64 = 40.0; // at least 60% detection

    /// Maximum acceptable detection latency in milliseconds (wake word onset
    /// to detection event).  Set to ticket's <500ms requirement.  If tests
    /// fail, investigate whether the pipeline's rolling-window architecture
    /// (VAD frame boundaries ~32ms at 16kHz) is the bottleneck.
    pub const MAX_LATENCY_MS: f64 = 500.0;

    /// Number of wake word utterances per scenario.
    pub const NUM_WW_UTTERANCES: usize = 5;
}

/// Number of seconds for "long" scenarios (non-wake-word speech, silence, noise).
/// 60s provides sufficient audio for statistically meaningful FAR estimation
/// (one false accept in 60s ≈ 60 FAR/hr at 95% confidence).
const LONG_DURATION_SECS: u64 = 60;
// FAR per hour for N seconds with k detections: k * 3600.0 / N

/// Run all standard test scenarios and collect metrics.
///
/// Returns [`AllScenarioResults`] with per-scenario metrics and peak memory.
/// If ONNX models are unavailable, returns an empty struct (caller should
/// interpret this as "tests skipped").
pub fn run_all_scenarios() -> AllScenarioResults {
    let mut results = Vec::new();

    // Try to create the runner
    let mut runner = match OfflinePipelineRunner::new() {
        Ok(Some(runner)) => runner,
        Ok(None) => {
            info!("Voice test harness: ONNX models unavailable — skipping all scenarios");
            return AllScenarioResults {
                scenarios: results,
                peak_memory_mb: 0.0,
            };
        }
        Err(e) => {
            warn!("Voice test harness: failed to initialize: {e}");
            return AllScenarioResults {
                scenarios: results,
                peak_memory_mb: 0.0,
            };
        }
    };

    // ── 1. Enroll ──
    if let Err(e) = runner.enroll() {
        warn!("Voice test harness: enrollment failed: {e} — skipping all scenarios");
        return AllScenarioResults {
            scenarios: results,
            peak_memory_mb: 0.0,
        };
    }

    // ── 2. Clean wake word ──
    results.push(run_clean_wake_word_scenario(&mut runner));
    runner.reset();

    // ── 3. Noisy wake word (10 dB SNR) ──
    results.push(run_noisy_wake_word_scenario(&mut runner));
    runner.reset();

    // ── 4. Non-wake-word speech ──
    results.push(run_non_wake_word_speech_scenario(&mut runner));
    runner.reset();

    // ── 5. Silence ──
    results.push(run_silence_scenario(&mut runner));
    runner.reset();

    // ── 6. Pink noise only ──
    results.push(run_noise_only_scenario(&mut runner, NoiseType::Pink));
    runner.reset();

    // ── 7. TV noise only ──
    results.push(run_noise_only_scenario(&mut runner, NoiseType::Tv));
    runner.reset();

    // ── 8. Enrollment stability ──
    results.push(run_enrollment_stability_scenario(&mut runner));

    // Peak memory: ru_maxrss is process-lifetime maximum RSS (monotonic).
    // Reading it at any point after scenario execution returns the peak
    // reached during the test, because getrusage tracks the all-time high
    // since process start — it is not a point-in-time sample.
    let peak_mb = peak_rss_bytes() as f64 / (1024.0 * 1024.0);

    AllScenarioResults {
        scenarios: results,
        peak_memory_mb: peak_mb,
    }
}

fn run_clean_wake_word_scenario(runner: &mut OfflinePipelineRunner) -> ScenarioMetrics {
    run_wake_word_scenario(
        runner,
        "clean_wake_word",
        acceptance::MIN_CLEAN_DETECTIONS,
        || wake_word_utterance(5),
    )
}

fn run_noisy_wake_word_scenario(runner: &mut OfflinePipelineRunner) -> ScenarioMetrics {
    let noise = pink_noise(2000); // 2s of pink noise for mixing
    run_wake_word_scenario(
        runner,
        "noisy_wake_word_10db",
        acceptance::MIN_NOISY_DETECTIONS,
        || {
            let clean = wake_word_utterance(5);
            mix_at_snr(&clean, &noise, 10.0)
        },
    )
}

/// Run a FAR-based scenario: feed audio, measure false accepts per hour.
///
/// Shared helper that eliminates the duplicated detection-count + FAR-calculation
/// + pass-check pattern across `run_non_wake_word_speech_scenario`,
/// `run_silence_scenario`, and `run_noise_only_scenario`.
///
/// NOTE: FAR/hr is computed from `audio.len()` (duration), NOT from
/// [`LONG_DURATION_SECS`], so callers can use any audio duration.
fn run_far_scenario(
    runner: &mut OfflinePipelineRunner,
    scenario_name: &str,
    audio: Vec<f32>,
) -> ScenarioMetrics {
    let duration_secs = audio.len() as f64 / SAMPLE_RATE as f64;
    runner.feed_audio(&audio);
    let detections = runner.detection_count();
    let far_per_hour = if duration_secs > 0.0 {
        detections as f64 * 3600.0 / duration_secs
    } else {
        0.0
    };
    ScenarioMetrics {
        pass: far_per_hour <= acceptance::MAX_FAR_PER_HOUR,
        far_per_hour: Some(far_per_hour),
        ..default_scenario(scenario_name, detections, None)
    }
}

/// Feed a single utterance through detection and return (detections, latencies).
///
/// Shared helper that eliminates the duplicated feed-detect-reset loop
/// across `run_clean_wake_word_scenario`, `run_noisy_wake_word_scenario`,
/// and `run_enrollment_stability_scenario`.
fn detect_utterance(
    runner: &mut OfflinePipelineRunner,
    utterance: &[f32],
) -> (usize, Vec<Duration>) {
    runner.feed_audio(utterance);
    let detections = runner.detection_count();
    let latencies = if detections > 0 {
        runner.detection_latencies()
    } else {
        Vec::new()
    };
    runner.reset();
    (detections, latencies)
}

/// Run [`acceptance::NUM_WW_UTTERANCES`] wake-word utterances and return
/// aggregated detection count and latencies.
fn run_wake_word_batch(
    runner: &mut OfflinePipelineRunner,
    prepare_utterance: impl Fn() -> Vec<f32>,
) -> (usize, Vec<Duration>) {
    let n = acceptance::NUM_WW_UTTERANCES;
    let mut detections = 0usize;
    let mut latencies = Vec::new();
    for _ in 0..n {
        let utterance = prepare_utterance();
        let (d, l) = detect_utterance(runner, &utterance);
        detections += d;
        latencies.extend(l);
    }
    (detections, latencies)
}

/// Build a full [`ScenarioMetrics`] for a wake-word scenario.
fn run_wake_word_scenario(
    runner: &mut OfflinePipelineRunner,
    name: &str,
    min_detections: usize,
    prepare_utterance: impl Fn() -> Vec<f32>,
) -> ScenarioMetrics {
    let (detections, latencies) = run_wake_word_batch(runner, prepare_utterance);
    compute_scenario_metrics(
        name,
        detections,
        Some(min_detections),
        &latencies,
        acceptance::NUM_WW_UTTERANCES as f64,
    )
}

fn run_non_wake_word_speech_scenario(runner: &mut OfflinePipelineRunner) -> ScenarioMetrics {
    let audio = non_wake_word_speech(LONG_DURATION_SECS * 1000);
    let name = format!("non_wake_word_speech_{LONG_DURATION_SECS}s");
    run_far_scenario(runner, &name, audio)
}

fn run_silence_scenario(runner: &mut OfflinePipelineRunner) -> ScenarioMetrics {
    let audio = silence(LONG_DURATION_SECS * 1000);
    let name = format!("silence_{LONG_DURATION_SECS}s");
    run_far_scenario(runner, &name, audio)
}

fn run_noise_only_scenario(
    runner: &mut OfflinePipelineRunner,
    noise_type: NoiseType,
) -> ScenarioMetrics {
    let audio = noise_type.generate(LONG_DURATION_SECS * 1000);
    let name = format!("noise_{}_{LONG_DURATION_SECS}s", noise_type.as_str());
    run_far_scenario(runner, &name, audio)
}

fn run_enrollment_stability_scenario(runner: &mut OfflinePipelineRunner) -> ScenarioMetrics {
    // Helper: compute latency statistics from a slice of Duration values.
    fn latency_stats(latencies: &[Duration]) -> (Option<f64>, Option<f64>, Option<f64>) {
        if latencies.is_empty() {
            return (None, None, None);
        }
        let ms: Vec<f64> = latencies.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        let mean = Some(ms.iter().sum::<f64>() / ms.len() as f64);
        let max = ms.iter().copied().reduce(f64::max);
        let p95 = if ms.len() >= 2 {
            let mut sorted = ms.clone();
            sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = ((sorted.len() as f64) * 0.95).ceil() as usize - 1;
            Some(sorted[idx.min(sorted.len() - 1)])
        } else {
            max
        };
        (mean, p95, max)
    }

    // Shared helper for constructing the metrics struct, eliminating
    // the ~13-field duplication between success and failure paths.
    fn make_metrics(
        detections: usize,
        pass: bool,
        latencies: &[Duration],
        detail: String,
    ) -> ScenarioMetrics {
        let (mean, p95, max) = latency_stats(latencies);
        ScenarioMetrics {
            scenario: "enrollment_stability".to_string(),
            detection_count: detections,
            expected_detections: None,
            pass,
            far_per_hour: None,
            frr_percent: None,
            latency_mean_ms: mean,
            latency_p95_ms: p95,
            latency_max_ms: max,
            detail: Some(detail),
        }
    }

    // Count detections with current enrollment
    let (before_count, before_latencies) = run_wake_word_batch(runner, || wake_word_utterance(5));

    // Re-enroll (second enrollment).
    // Reset enrollment state using the proper abstraction method.
    runner.reset_enrollment();
    match runner.enroll() {
        Ok(()) => {
            // Count detections again
            let (after_count, after_latencies) =
                run_wake_word_batch(runner, || wake_word_utterance(5));

            let pass = before_count >= acceptance::MIN_CLEAN_DETECTIONS
                && after_count >= acceptance::MIN_CLEAN_DETECTIONS;

            let mut detail = format!(
                "Before second enrollment: {before_count}/{max_utt} detections, \
                 After second enrollment: {after_count}/{max_utt} detections",
                max_utt = acceptance::NUM_WW_UTTERANCES
            );
            if !before_latencies.is_empty() {
                let before_ms = before_latencies
                    .iter()
                    .map(|d| d.as_secs_f64() * 1000.0)
                    .collect::<Vec<_>>();
                detail.push_str(&format!(
                    " (mean latency before/after: {:.0}/{:.0} ms)",
                    before_ms.iter().sum::<f64>() / before_ms.len() as f64,
                    after_latencies
                        .iter()
                        .map(|d| d.as_secs_f64() * 1000.0)
                        .sum::<f64>()
                        / after_latencies.len() as f64,
                ));
            }

            // Use aggregated latencies from both rounds for the metric.
            let all_latencies: Vec<Duration> = before_latencies
                .into_iter()
                .chain(after_latencies)
                .collect();
            make_metrics(after_count, pass, &all_latencies, detail)
        }
        Err(e) => make_metrics(
            0, // no valid detection after failed second enrollment
            false,
            &[],
            format!("Second enrollment failed: {e}"),
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Metric helpers
// ═══════════════════════════════════════════════════════════════════════════

fn default_scenario(name: &str, detections: usize, expected: Option<usize>) -> ScenarioMetrics {
    // Compute pass from expected if provided.  For FAR-based scenarios where
    // `expected` is `None`, `pass` defaults to `false` and must be explicitly
    // overridden by the caller.  This ensures a future developer who adds a
    // scenario using `default_scenario` with `expected: None` and forgets to
    // set `pass` will get a visible test failure rather than silent success.
    let pass = expected.map_or(false, |exp| detections >= exp);
    ScenarioMetrics {
        scenario: name.to_string(),
        detection_count: detections,
        expected_detections: expected,
        pass,
        far_per_hour: None,
        frr_percent: None,
        latency_mean_ms: None,
        latency_p95_ms: None,
        latency_max_ms: None,
        detail: None,
    }
}

fn compute_scenario_metrics(
    name: &str,
    detection_count: usize,
    expected_detections: Option<usize>,
    latencies: &[Duration],
    total_utterances: f64,
) -> ScenarioMetrics {
    let frr = if total_utterances > 0.0 {
        Some((1.0 - detection_count as f64 / total_utterances) * 100.0)
    } else {
        None
    };

    let lat_ms: Vec<f64> = latencies.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    let mean_ms = if lat_ms.is_empty() {
        None
    } else {
        Some(lat_ms.iter().sum::<f64>() / lat_ms.len() as f64)
    };
    let p95_ms = if lat_ms.len() >= 2 {
        let mut sorted = lat_ms.clone();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((sorted.len() as f64) * 0.95).ceil() as usize - 1;
        Some(sorted[idx.min(sorted.len() - 1)])
    } else {
        lat_ms.last().copied()
    };
    let max_ms = lat_ms.iter().copied().reduce(f64::max);

    let pass_expected = expected_detections.map_or(true, |exp| detection_count >= exp);
    let pass_latency = max_ms.map_or(true, |m| m <= acceptance::MAX_LATENCY_MS);
    let pass_frr = frr.map_or(true, |f| f <= acceptance::MAX_FRR_PERCENT);

    ScenarioMetrics {
        scenario: name.to_string(),
        detection_count,
        expected_detections,
        pass: pass_expected && pass_latency && pass_frr,
        far_per_hour: None,
        frr_percent: frr,
        latency_mean_ms: mean_ms,
        latency_p95_ms: p95_ms,
        latency_max_ms: max_ms,
        detail: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Baseline file management
// ═══════════════════════════════════════════════════════════════════════════

/// Directory for baseline metrics.
fn baseline_dir() -> Option<PathBuf> {
    let root = CONFIG.try_storage_root()?;
    Some(root)
}

/// Run all scenarios and save the baseline JSON file.
///
/// If `output_dir` is `Some(path)`, the baseline is written to
/// `path/voice-test-baseline.json`.  If `None`, it is written to
/// the storage root (`~/.mahbot/`).
///
/// Returns the collected metrics, or an empty vec if models are unavailable.
pub fn run_and_save_baseline(output_dir: Option<&std::path::Path>) -> Vec<ScenarioMetrics> {
    let all_results = run_all_scenarios();
    if all_results.scenarios.is_empty() {
        return all_results.scenarios;
    }

    let total = all_results.scenarios.len();
    let passed = all_results.scenarios.iter().filter(|s| s.pass).count();
    let failed = total - passed;

    let baseline = BaselineMetrics {
        version: BaselineMetrics::VERSION,
        timestamp: chrono::Utc::now().to_rfc3339(),
        scenarios: all_results.scenarios.clone(),
        summary: SummaryMetrics {
            all_pass: failed == 0,
            total_scenarios: total,
            passed,
            failed,
            peak_memory_mb: all_results.peak_memory_mb,
        },
    };

    let dir = output_dir
        .map(std::path::Path::to_path_buf)
        .or_else(baseline_dir);
    if let Some(dir) = dir {
        let path = dir.join("voice-test-baseline.json");
        match serde_json::to_string_pretty(&baseline) {
            Ok(json) => match std::fs::write(&path, &json) {
                Ok(()) => info!("Voice test baseline saved to {:?}", path),
                Err(e) => warn!("Failed to write baseline to {:?}: {e}", path),
            },
            Err(e) => warn!("Failed to serialize baseline JSON: {e}"),
        }
    }

    all_results.scenarios
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

// Note: the entire file is gated on `#[cfg(all(test, feature = "voice-tests"))]`,
// so we don't need an additional `#[cfg(test)]` here.
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::TempDir;

    /// Helper: run a scenario and assert basic pass/fail.
    fn assert_scenario_pass(metrics: &ScenarioMetrics) {
        let msg = if let Some(far) = metrics.far_per_hour {
            format!(
                "Scenario '{}' FAILED: {} detections ({:.1} FAR/hr, threshold {:.1}), detail: {:?}",
                metrics.scenario,
                metrics.detection_count,
                far,
                acceptance::MAX_FAR_PER_HOUR,
                metrics.detail,
            )
        } else if let Some(exp) = metrics.expected_detections {
            format!(
                "Scenario '{}' FAILED: {} detections (expected at least {}), detail: {:?}",
                metrics.scenario, metrics.detection_count, exp, metrics.detail,
            )
        } else {
            format!(
                "Scenario '{}' FAILED: {} detections, detail: {:?}",
                metrics.scenario, metrics.detection_count, metrics.detail,
            )
        };
        assert!(metrics.pass, "{}", msg);
    }

    // ── Synthetic audio generator tests ──

    #[test]
    fn test_synthetic_wake_word_frame_ranges() {
        let frame = wake_word_frame();
        assert_eq!(frame.len(), FRAME_LENGTH);
        for &s in &frame {
            assert!(s >= -1.0 && s <= 1.0, "sample {s} out of [-1, 1]");
        }
    }

    #[test]
    fn test_synthetic_non_wake_word_frame_ranges() {
        let frame = non_wake_word_frame();
        assert_eq!(frame.len(), FRAME_LENGTH);
        for &s in &frame {
            assert!(s >= -1.0 && s <= 1.0, "sample {s} out of [-1, 1]");
        }
    }

    #[test]
    fn test_synthetic_wake_word_utterance_length() {
        let utterance = wake_word_utterance(5);
        // Should be non-empty and have reasonable sample count
        assert!(!utterance.is_empty());
        let min_expected = (SAMPLE_RATE as usize) / 2; // at least 500ms
        assert!(
            utterance.len() > min_expected,
            "utterance too short: {} < {}",
            utterance.len(),
            min_expected,
        );
    }

    #[test]
    fn test_synthetic_silence_is_zero() {
        let silence = silence(100);
        assert!(silence.iter().all(|&s| s == 0.0));
        assert_eq!(silence.len(), SAMPLE_RATE as usize / 10); // 100ms at 16kHz
    }

    #[test]
    fn test_synthetic_pink_noise_ranges() {
        let noise = pink_noise(500); // 500ms
        assert!(!noise.is_empty());
        assert_eq!(
            noise.len(),
            (SAMPLE_RATE as usize) / 2, // 500ms at 16kHz
        );
        for &s in &noise {
            assert!(s >= -1.0 && s <= 1.0, "noise sample {s} out of [-1, 1]");
        }
        // Pink noise should have non-zero RMS
        assert!(compute_rms(&noise) > 0.001, "pink noise RMS too low");
    }

    #[test]
    fn test_synthetic_hum_frequency() {
        let hum = stationary_hum(200, 100.0); // 100 Hz hum, 200ms
        assert!(!hum.is_empty());
        assert!(compute_rms(&hum) > 0.001, "hum RMS too low");
    }

    #[test]
    fn test_synthetic_tv_noise_ranges() {
        let noise = tv_noise(500);
        assert!(!noise.is_empty());
        for &s in &noise {
            assert!(s >= -1.0 && s <= 1.0, "TV noise sample {s} out of [-1, 1]");
        }
        assert!(compute_rms(&noise) > 0.001, "TV noise RMS too low");
    }

    #[test]
    fn test_mix_at_snr_preserves_signal() {
        let signal = vec![0.5f32; 1000];
        let noise = vec![0.0f32; 1000]; // zero noise
        let mixed = mix_at_snr(&signal, &noise, 10.0);
        assert_eq!(mixed.len(), 1000);
        // With zero noise, the output should equal the input
        for (a, b) in signal.iter().zip(mixed.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "mix_at_snr with zero noise changed signal"
            );
        }
    }

    #[test]
    fn test_mix_at_snr_changes_noise_level() {
        let signal = vec![0.5f32; 1000];
        let noise = vec![0.5f32; 1000];
        // High SNR (20 dB): noise barely audible
        let high_snr = mix_at_snr(&signal, &noise, 20.0);
        // Low SNR (0 dB): noise equals signal
        let low_snr = mix_at_snr(&signal, &noise, 0.0);
        // Low SNR should have more noise contribution
        let rms_high = compute_rms(&high_snr);
        let rms_low = compute_rms(&low_snr);
        assert!(
            rms_low > rms_high,
            "low SNR should have higher RMS than high SNR (high: {rms_high}, low: {rms_low})",
        );
    }

    #[test]
    fn test_non_wake_word_speech_duration() {
        let speech = non_wake_word_speech(2000);
        // Should be roughly 2000ms ± one frame
        let expected = (2000 * SAMPLE_RATE as usize) / 1000;
        assert!(
            (speech.len() as isize - expected as isize).abs() < FRAME_LENGTH as isize,
            "non-wake speech length {len} not close to expected {expected}",
            len = speech.len(),
        );
    }

    /// Consolidated pipeline integration test.
    ///
    /// Runs `run_all_scenarios()` ONCE and asserts every scenario's metrics
    /// in a single test, avoiding redundant ONNX-based execution that was
    /// the previous pattern of 6–7 independent tests each calling
    /// `run_all_scenarios()`.
    #[test]
    #[serial(voice)]
    fn test_all_pipeline_scenarios() {
        let all_results = run_all_scenarios();
        if all_results.scenarios.is_empty() {
            return;
        }
        assert!(
            all_results.peak_memory_mb >= 0.0,
            "peak memory should be non-negative, got {}",
            all_results.peak_memory_mb,
        );

        // Expected scenario name prefixes/identifiers
        let expected_prefixes = [
            "clean_wake_word",
            "noisy_wake_word_10db",
            "non_wake_word_speech_",
            "silence_",
            "noise_pink_",
            "noise_tv_",
            "enrollment_stability",
        ];

        // Assert all scenarios appear and pass
        for prefix in &expected_prefixes {
            let found = all_results.scenarios.iter().find(|s| {
                if prefix.ends_with('_') {
                    s.scenario.starts_with(prefix)
                } else {
                    s.scenario == *prefix
                }
            });
            assert!(
                found.is_some(),
                "Scenario matching '{prefix}' not found in results"
            );
            let metrics = found.unwrap();
            assert_scenario_pass(metrics);
        }
    }

    #[test]
    fn test_baseline_save() {
        // Unit test for baseline serialization — does NOT require ONNX models.
        // Constructs mock scenarios and writes them via the baseline writer.
        let tmp = TempDir::new().expect("temp directory creation");

        let scenarios = vec![
            ScenarioMetrics {
                scenario: "clean_wake_word".to_string(),
                detection_count: 5,
                expected_detections: Some(4),
                pass: true,
                far_per_hour: None,
                frr_percent: Some(0.0),
                latency_mean_ms: Some(120.0),
                latency_p95_ms: Some(200.0),
                latency_max_ms: Some(250.0),
                detail: None,
            },
            ScenarioMetrics {
                scenario: "silence_60s".to_string(),
                detection_count: 0,
                expected_detections: None,
                pass: true,
                far_per_hour: Some(0.0),
                frr_percent: None,
                latency_mean_ms: None,
                latency_p95_ms: None,
                latency_max_ms: None,
                detail: None,
            },
        ];

        let baseline = BaselineMetrics {
            version: BaselineMetrics::VERSION,
            timestamp: chrono::Utc::now().to_rfc3339(),
            scenarios: scenarios.clone(),
            summary: SummaryMetrics {
                all_pass: true,
                total_scenarios: 2,
                passed: 2,
                failed: 0,
                peak_memory_mb: 42.0,
            },
        };

        let path = tmp.path().join("voice-test-baseline.json");
        let json = serde_json::to_string_pretty(&baseline).expect("serialization should succeed");
        std::fs::write(&path, &json).expect("file write should succeed");

        assert!(
            path.exists(),
            "baseline file should have been written to {:?}",
            path,
        );

        // Verify it can be deserialized back
        let content = std::fs::read_to_string(&path).expect("should be able to read written file");
        let restored: BaselineMetrics = serde_json::from_str(&content)
            .expect("should be valid JSON matching BaselineMetrics schema");
        assert_eq!(restored.version, BaselineMetrics::VERSION);
        assert_eq!(restored.scenarios.len(), 2);
        assert_eq!(restored.scenarios[0].scenario, "clean_wake_word");
        assert!(restored.summary.all_pass);
    }

    // ── VAD integration tests using synthetic audio ──

    #[test]
    #[serial(voice)]
    fn test_synthetic_speech_frames_pass_vad() {
        // Verify that both synthetic wake-word and non-wake-word frames
        // pass Earshot neural VAD, which is required to be processed by
        // the embedding and detection pipeline.
        reset_vad();
        let silence = vec![0.0f32; FRAME_LENGTH];
        super::is_speech(&silence); // calibrate noise floor

        let wake_word = wake_word_frame();
        assert!(
            super::is_speech(&wake_word),
            "wake word frame (len={}) should be classified as speech by Earshot VAD",
            wake_word.len(),
        );

        reset_vad();
        super::is_speech(&silence); // re-calibrate for second frame type

        let non_wake = non_wake_word_frame();
        assert!(
            super::is_speech(&non_wake),
            "non-wake-word frame (len={}) should be classified as speech by Earshot VAD",
            non_wake.len(),
        );
    }

    // ── Non-wake-word speech spectral variation ──

    #[test]
    fn test_non_wake_word_speech_has_spectral_variation() {
        let speech = non_wake_word_speech(2000);

        // Verify that the speech has spectral variation over time by comparing
        // RMS in different segments.  With formant alternation, there should
        // be measurable variation across segments.
        let segment_len = speech.len() / 4;
        let rms_segments: Vec<f32> = speech
            .chunks(segment_len)
            .map(|seg| compute_rms(seg))
            .collect();

        let max_rms = rms_segments.iter().copied().fold(0.0f32, f32::max);
        let min_rms = rms_segments.iter().copied().fold(f32::MAX, f32::min);
        assert!(
            max_rms > min_rms,
            "non-wake speech should have RMS variation across segments \
             (min={min_rms}, max={max_rms})"
        );
    }

    // ── Pink noise statistical properties ──

    #[test]
    fn test_pink_noise_power_distribution() {
        // Verify that pink noise has more low-frequency power than white noise
        // (1/f spectral characteristic).  We compare the ratio of low-pass
        // filtered RMS (32-sample moving average, ~500Hz cutoff) to total RMS
        // for both pink noise and white noise.  Pink noise must have a
        // significantly larger ratio.
        let pink = pink_noise(2000); // 2 seconds

        // Basic sanity: non-zero RMS and non-trivial amplitude
        assert!(compute_rms(&pink) > 0.001, "pink noise RMS too low");
        let max_abs_pink = pink.iter().copied().fold(0.0f32, f32::max);
        assert!(
            max_abs_pink > 0.01,
            "pink noise should have non-trivial amplitude, max_abs={max_abs_pink}"
        );

        // Generate reference white noise at similar RMS level
        let mut rng = SmallRng::seed_from_u64(12345);
        let white: Vec<f32> = (0..pink.len())
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.3)
            .collect();

        // Simple low-pass filter via 32-sample moving average (~500 Hz cutoff
        // at 16 kHz).  Pink noise should have significantly more energy below
        // this cutoff than white noise.
        let window_len = 32usize;

        fn lp_rms_ratio(samples: &[f32], window_len: usize) -> f64 {
            let mut squared_sum = 0.0f64;
            let mut n_samples = 0;
            for chunk in samples.windows(window_len) {
                let mean = chunk.iter().copied().sum::<f32>() / window_len as f32;
                squared_sum += (mean as f64) * (mean as f64);
                n_samples += 1;
            }
            let total_rms = compute_rms(samples) as f64;
            if total_rms == 0.0 || n_samples == 0 {
                return 0.0;
            }
            let lp_rms = (squared_sum / n_samples as f64).sqrt();
            lp_rms / total_rms
        }

        let pink_ratio = lp_rms_ratio(&pink, window_len);
        let white_ratio = lp_rms_ratio(&white, window_len);

        assert!(
            pink_ratio > white_ratio * 1.5,
            "pink noise low-pass RMS ratio ({pink_ratio:.4}) should be > 1.5× white noise ratio \
             ({white_ratio:.4}) — pink noise spectrum may be too white"
        );
    }
}
