//! Wake-word recording cues: the two short sounds the hands-free voice path plays when a recording
//! starts and stops. They are the product's own samples — two assets embedded in the binary and
//! played through the machine's normal audio output — so they never consult an OS sound API,
//! library, theme or preference. Cue playback deliberately bypasses the spoken-phrase path of
//! [`crate::speech::synth`]: the cues play with speech output off, and must not raise the
//! speaking flag that makes the voice pipeline drop microphone audio. A host without an audio
//! output device is skipped silently.

use crate::audio::output;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};

/// The start cue: a short rising two-tone blip. Short by construction — the still-open microphone
/// captures it, so its length belongs to the recording-side budget documented at
/// [`crate::audio::voice::POST_FIRE_DISCARD_SAMPLES`].
const START_WAV: &[u8] = include_bytes!("start.wav");

/// The end cue: a short lower blip, distinct from the start cue. Short as well — the recording is
/// over, but the microphone is still open.
const END_WAV: &[u8] = include_bytes!("end.wav");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cue {
    /// A wake-word recording just started.
    Start,
    /// A wake-word recording stopped, whatever the reason.
    End,
}

/// Latched once an output-device open has failed, so a host that cannot open one is not re-probed
/// (a costly open) on every activation. [`warm_up`] clears it once per listening session, so a
/// transient failure cannot silence the cues for the whole process. The latch only suppresses the
/// probe: as soon as any path opens the device, cues play through the same mixer.
static NO_OUTPUT: AtomicBool = AtomicBool::new(false);

/// Play `cue` through the machine's normal audio output. Best-effort and non-blocking: the cue is
/// mixed into the shared output and neither held nor waited for.
pub fn play(cue: Cue) {
    let Some(mixer) = output_mixer() else {
        return;
    };
    let asset = match cue {
        Cue::Start => START_WAV,
        Cue::End => END_WAV,
    };
    let Ok(decoder) = rodio::Decoder::new(Cursor::new(asset)) else {
        return;
    };
    // `Mixer::add` is rodio's one-shot playback: the mixer takes ownership of the source, so the
    // voice pipeline never waits for the cue.
    mixer.add(decoder);
}

/// Warm the shared output device so a later [`play`] is not delayed by a cold open. Called when the
/// wake-word microphone starts: the start cue is captured by that still-open microphone and must
/// fall inside the opening the recording discards, so a device open on the detection path could
/// push it into recognized audio. Also gives each listening session one fresh probe of a device
/// that previously failed.
pub fn warm_up() {
    NO_OUTPUT.store(false, Ordering::Relaxed);
    let _ = output_mixer();
}

/// The shared output mixer, opening the device on first use. Unlike a spoken phrase this opens it
/// silently and latches a failure, so a host without audio output neither warns nor re-probes a
/// costly open.
fn output_mixer() -> Option<rodio::mixer::Mixer> {
    if !NO_OUTPUT.load(Ordering::Relaxed) && !output::ensure_audio_output_silent() {
        NO_OUTPUT.store(true, Ordering::Relaxed);
    }
    output::mixer()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both assets decode and carry audible samples: [`play`] swallows a decode error, so a
    /// corrupted or swapped cue file would otherwise silence the feature invisibly.
    #[test]
    fn embedded_cue_assets_decode_and_are_audible() {
        for (name, asset) in [("start", START_WAV), ("end", END_WAV)] {
            let decoder = rodio::Decoder::new(Cursor::new(asset))
                .unwrap_or_else(|e| panic!("{name} cue must decode: {e}"));
            let samples: Vec<f32> = decoder.collect();
            let peak = samples.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
            assert!(!samples.is_empty(), "{name} cue is empty");
            assert!(peak > 0.01, "{name} cue is silent");
        }
    }
}
