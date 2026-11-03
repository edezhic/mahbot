//! Audio-subsystem utilities: PCM resampling for the capture path and the RMS helper the
//! enrollment quality gates read.
//!
//! These live here (not in `crate::util`) because they are the audio subsystem's own. The
//! resampler serves the microphone stream on every platform.

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
/// pre-filter — the filter only provides ~6 dB stopband attenuation. That is
/// acceptable for microphone speech.
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

/// Compute the RMS (root mean square) of audio samples.
///
/// Returns `0.0` for empty input. Callers that need a divide-by-zero floor for
/// degenerate all-zero input apply `.max(1e-10)` at the call site — it is
/// deliberately not part of this function.
#[expect(clippy::cast_precision_loss)]
pub(crate) fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}
