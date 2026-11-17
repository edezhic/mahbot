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

/// The audio library's soname on Linux — the one the shim (`vendor/alsa-sys-dlopen`) resolves
/// every `snd_*` call through. The literal exists once per side because the shim is a separate
/// crate the published package replaces with the registry one; [`the_probe_and_the_shim_name_one_library`]
/// keeps them from drifting.
#[cfg(target_os = "linux")]
const ALSA_SONAME: &[u8] = b"libasound.so.2\0";

/// Whether the platform's audio library is loadable on this host.
///
/// On Linux the library is looked up on first use, never at start — the product must run on a
/// server that has no audio stack at all — and a host without it reports voice unavailable. Every
/// other platform's audio stack is part of the system. This is the one answer the product gives:
/// the ALSA shim's wrappers resolve the same soname and report nothing of their own, so a host
/// that passes this probe cannot have every call fall back.
#[must_use]
pub fn library_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        use std::sync::OnceLock;
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            // Nothing calls `dlclose`, so the handle stays open for the life of the process — the
            // shim's wrappers need the library loaded, and one reference is the whole cost.
            let handle = unsafe {
                libc::dlopen(
                    ALSA_SONAME.as_ptr().cast(),
                    libc::RTLD_LAZY | libc::RTLD_LOCAL,
                )
            };
            !handle.is_null()
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
}

/// The one sentence every voice surface says on a host without the audio library.
pub(crate) const LIBRARY_UNAVAILABLE: &str = concat!(
    "Voice is unavailable: the audio library (libasound2 on Linux) is not installed on this ",
    "machine."
);

/// Open the speech output device at start when the owner has speech output on. No model is
/// touched: both engines load on first use, and a phrase is not refused while a cold device
/// open runs.
pub fn open_speech_output_if_enabled() {
    if !library_available() {
        tracing::info!("{LIBRARY_UNAVAILABLE}");
        return;
    }
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

#[cfg(all(test, target_os = "linux"))]
mod tests {
    /// The probe and the shim resolve one library: a drift here would report voice available while
    /// every `snd_*` wrapper answered with a failure value.
    #[test]
    fn the_probe_and_the_shim_name_one_library() {
        let shim = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("vendor/alsa-sys-dlopen/build.rs");
        // The published package replaces the shim with the registry crate, so there is nothing to
        // compare there.
        let Ok(shim) = std::fs::read_to_string(shim) else {
            return;
        };
        let soname = std::str::from_utf8(&super::ALSA_SONAME[..super::ALSA_SONAME.len() - 1])
            .expect("the soname is text");
        assert!(
            shim.contains(soname),
            "the shim resolves a different soname than the probe ({soname})"
        );
    }
}
