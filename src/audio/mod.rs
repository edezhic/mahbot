//! Audio: the microphone and the voice command pipeline, the machine's audio output, and, on
//! macOS only, wake word detection and its recording cues.
//!
//! Capture goes through the platform's own host API (`cpal`: CoreAudio, WASAPI, ALSA), so the
//! microphone and the pipeline that turns a recording into a composer message are on every
//! platform. Recognition and spoken replies are not part of this subsystem: both live in
//! [`crate::speech`] and are on every platform too.
//!
//! macOS keeps what has no other-platform support: the wake word ([`wake_word`], encoding on
//! the Qwen3-ASR encoder of [`encoder`]) and the wake-word recording cues ([`cue`]).
//!
//! Wake word detection runs entirely on the Qwen3-ASR encoder — no separate
//! embedding model, no trainable head, no AGC/NS preprocessing.

#[cfg(target_os = "macos")]
pub(crate) mod cue;
#[cfg(target_os = "macos")]
pub(crate) mod encoder;
pub(crate) mod output;
pub(crate) mod util;
pub mod voice;
#[cfg(target_os = "macos")]
pub(crate) mod wake_word;

/// The per-user `~/.mahbot/models/` subdirectory the wake word's encoder owns. Recognition and
/// spoken replies read the machine-wide root ([`crate::speech::models`]).
#[cfg(target_os = "macos")]
pub(crate) fn models_subdir(name: &str) -> Option<std::path::PathBuf> {
    crate::util::models_dir().map(|dir| dir.join(name))
}

/// Open the speech output device at start when the owner has speech output on. No model is
/// touched: both engines load on first use, and a phrase is not refused while a cold device
/// open runs.
pub fn open_speech_output_if_enabled() {
    if crate::speech::synth::is_config_enabled() {
        let _ = output::ensure_audio_output();
    }
}

/// The old synthesiser's per-user model tree, deleted once: nothing reads it now, and its
/// weights are hundreds of megabytes. A failure is said plainly — the tree is dead weight, not
/// a reason to refuse anything.
#[cfg(target_os = "macos")]
pub(crate) fn remove_retired_models() {
    let Some(dir) = models_subdir("supertonic3") else {
        return;
    };
    if !dir.exists() {
        return;
    }
    match std::fs::remove_dir_all(&dir) {
        Ok(()) => tracing::info!(dir = %dir.display(), "Removed the retired speech model tree"),
        Err(e) => {
            tracing::warn!(dir = %dir.display(), error = %e, "Could not remove the retired speech model tree");
        }
    }
}
