//! The machine's audio output: one shared device and mixer, and the flag that says the
//! product is speaking.
//!
//! Every sound the product makes goes through this one device: a spoken phrase
//! ([`crate::speech::synth`]) and the wake-word recording cues ([`crate::audio::cue`]) each
//! take a player from the shared mixer, so the two mix rather than fight over the output. The
//! device opens on first use — or at start when the owner has speech output on
//! ([`crate::audio::open_speech_output_if_enabled`]) — and a host without audio output stays
//! silent, with a later call retrying the open.
//!
//! The speaking flag is what the voice pipeline's echo gate reads: while a spoken phrase plays
//! — and for [`SPEAKING_TAIL`] after it — the pipeline drops microphone audio instead of
//! hearing the product itself.

use crate::util::UnwrapPoison;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use tracing::{debug, warn};

/// How long after playback ends the flag stays raised, so room acoustics and the output
/// buffers cannot reach the microphone.
pub(crate) const SPEAKING_TAIL: Duration = Duration::from_millis(300);

/// The open device and the mixer every player is created from. rodio splits the two: the sink
/// owns the host stream, and the mixer is a cheap `Arc`-cloned handle to it. Both are
/// `Send + Sync`, so a future bump that re-introduces a non-`Send` type fails at the `static`.
struct Output {
    _sink: rodio::MixerDeviceSink,
    mixer: rodio::mixer::Mixer,
}

static OUTPUT: OnceLock<Output> = OnceLock::new();
/// Serializes lazy device opens, so a costly one is never raced by a concurrent player.
static OPEN_LOCK: Mutex<()> = Mutex::new(());

/// The phrases playing right now, plus the ones still inside their tail.
static SPEAKING: AtomicUsize = AtomicUsize::new(0);

/// Raise the speaking flag for one phrase; it falls when the guard drops. A counter rather
/// than a boolean, so overlapping phrases keep the flag up until the last one is over.
pub(crate) struct Speaking(());

impl Speaking {
    pub(crate) fn start() -> Self {
        SPEAKING.fetch_add(1, Ordering::Relaxed);
        Self(())
    }
}

impl Drop for Speaking {
    fn drop(&mut self) {
        SPEAKING.fetch_sub(1, Ordering::Release);
    }
}

/// Whether the product is speaking right now, or within [`SPEAKING_TAIL`] of the last phrase.
/// The voice pipeline's echo gate reads this on every platform.
pub(crate) fn is_speaking() -> bool {
    SPEAKING.load(Ordering::Acquire) > 0
}

/// Whether the output device was opened. A device disconnected after the open is not
/// reflected; a host that has none stays `false` until a later open succeeds.
pub(crate) fn audio_output_ready() -> bool {
    OUTPUT.get().is_some()
}

/// Open the OS audio output device on demand (idempotent, best-effort); returns whether output
/// is ready.
pub(crate) fn ensure_audio_output() -> bool {
    open(true)
}

/// [`ensure_audio_output`] for the wake-word cues, except that a failed open is not warned
/// about: a host without audio output stays silent.
#[cfg_attr(
    not(target_os = "macos"),
    allow(dead_code, reason = "the only caller is the macOS wake-word cue")
)]
pub(crate) fn ensure_audio_output_silent() -> bool {
    open(false)
}

/// The shared mixer, for playback that is not a spoken phrase. `None` until the device has
/// been opened.
pub(crate) fn mixer() -> Option<rodio::mixer::Mixer> {
    OUTPUT.get().map(|output| output.mixer.clone())
}

/// Shared body of the two `ensure` calls; `warn_on_failure` picks the log level.
fn open(warn_on_failure: bool) -> bool {
    if OUTPUT.get().is_some() {
        return true;
    }
    // Re-check inside the guard: another thread may have won the race already.
    let _guard = OPEN_LOCK.lock().unwrap_poison();
    if OUTPUT.get().is_some() {
        return true;
    }
    match rodio::DeviceSinkBuilder::open_default_sink() {
        Ok(sink) => {
            let mixer = sink.mixer().clone();
            // The guard above makes this the only setter, so the set cannot lose a race.
            let _ = OUTPUT.set(Output { _sink: sink, mixer });
        }
        Err(e) => {
            if warn_on_failure {
                warn!("Failed to initialize audio output — playback will be disabled: {e}");
            } else {
                debug!("Audio output unavailable — playback stays silent: {e}");
            }
        }
    }
    OUTPUT.get().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The echo gate's whole contract: nothing is speaking at rest, one phrase raises the flag,
    /// and a second phrase keeps it raised past the first one's end.
    #[test]
    #[serial_test::serial(speech)]
    fn the_speaking_flag_holds_until_the_last_phrase_ends() {
        assert!(!is_speaking(), "nothing is speaking at rest");
        let first = Speaking::start();
        assert!(is_speaking());
        let second = Speaking::start();
        drop(first);
        assert!(is_speaking(), "a second phrase keeps the flag raised");
        drop(second);
        assert!(!is_speaking(), "the flag falls with the last phrase");
    }
}
