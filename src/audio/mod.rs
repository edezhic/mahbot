//! Audio: the microphone, the voice command pipeline, wake-word detection and the machine's audio
//! output.
//!
//! Capture goes through the platform's own host API (`cpal`: CoreAudio, WASAPI, ALSA), so the
//! microphone and the pipeline that turns a recording into a composer message are on every
//! platform. Recognition and spoken replies are not part of this subsystem: both live in
//! [`crate::speech`].
//!
//! The wake word ([`wake_word`]) is on every platform too: it feeds VAD-gated windows to the
//! inference crate's listener and acts on the decision, and its recording cues ([`cue`]) play
//! through the shared output device.

pub(crate) mod cue;
pub(crate) mod output;
pub(crate) mod util;
pub mod voice;
pub(crate) mod wake_word;

/// Open the speech output device at start when the owner has speech output on. No model is
/// touched: both engines load on first use, and a phrase is not refused while a cold device
/// open runs.
pub fn open_speech_output_if_enabled() {
    if crate::speech::synth::is_config_enabled() {
        let _ = output::ensure_audio_output();
    }
}

/// The retired per-user model tree, deleted once: the old encoder, its weights, the cache file
/// beside them and the orphan embedding model. Nothing reads that folder now — recognition and
/// synthesis use the machine-wide root ([`crate::speech::models`]) — and its weights are
/// gigabytes. A failure is said plainly: the tree is dead weight, not a reason to refuse anything.
pub(crate) fn remove_retired_models() {
    let Some(root) = crate::config::CONFIG.try_storage_root() else {
        return;
    };
    let dir = root.join("models");
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
