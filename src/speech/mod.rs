//! Speech: the product's own engines on the inference crate, both directions.
//!
//! Recognition ([`engine`]) turns a recording into words; synthesis ([`synth`]) turns text into
//! a spoken waveform. Both run on every platform, both read the same machine-wide model folder
//! ([`models`], provisioned by the installer and never created here), and both load lazily
//! through the one lifecycle in [`load`] — nothing at boot, and only the set the operation
//! needs is fetched.
//!
//! Recognition is lazy on every platform: the surfaces are the composer's microphone, a
//! Telegram voice note or audio attachment, the audio recorded after the wake word, and
//! `mahbot audio transcribe`. Synthesis is the admin Assistant's `say` tool and `mahbot audio
//! speak`; nothing is ever spoken on its own.

pub(crate) mod decode;
pub(crate) mod engine;
mod load;
pub(crate) mod models;
pub(crate) mod synth;

pub(crate) use decode::{AUDIO_FORMATS, takes_audio};
pub(crate) use engine::{
    INFERENCE_TIMEOUT, InferenceTimedOut, UNTRANSCRIBED, ensure_ready_once, is_failed, is_loaded,
    is_loading, retry_init, spawn_background_init, state, transcribe_file_async,
    transcribe_samples,
};
