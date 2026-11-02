//! Audio/voice subsystem — the microphone and the voice command pipeline, and,
//! on macOS only, wake word detection, spoken replies and the recording cues.
//!
//! Capture goes through the platform's own host API (`cpal`: CoreAudio, WASAPI,
//! ALSA), so the microphone and the pipeline that turns a recording into a
//! composer message are on every platform. Recognition is not part of this
//! subsystem: it lives in [`crate::speech`] and is on every platform too.
//!
//! macOS keeps what has no other-platform support: the wake word
//! ([`wake_word`], encoding on the Qwen3-ASR encoder of [`encoder`]), the
//! spoken replies of [`tts`] with their download machinery, and the wake-word
//! recording cues ([`cue`]).
//!
//! Wake word detection runs entirely on the Qwen3-ASR encoder — no separate
//! embedding model, no trainable head, no AGC/NS preprocessing.

#[cfg(target_os = "macos")]
pub(crate) mod cue;
#[cfg(target_os = "macos")]
pub(crate) mod encoder;
#[cfg(target_os = "macos")]
pub mod tts;
pub(crate) mod util;
pub mod voice;
#[cfg(target_os = "macos")]
pub(crate) mod wake_word;

/// A per-model subdirectory under the per-user `~/.mahbot/models/` root — spoken replies
/// and the wake word's encoder each own one. Only the two macOS model sets live there:
/// recognition reads the machine-wide root ([`crate::speech`]).
#[cfg(target_os = "macos")]
pub(crate) fn models_subdir(name: &str) -> Option<std::path::PathBuf> {
    crate::util::models_dir().map(|dir| dir.join(name))
}
