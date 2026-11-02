//! Speech recognition: the product's one recogniser, the machine-wide model set
//! it reads, and the transcript a recording turns into.
//!
//! Recognition runs on the inference crate (`mahbot_inference`). Its models live at
//! one fixed path per operating system ([`models::root`]), provisioned by the
//! installer and never created here; the product fetches the files the engine
//! needs, verifies them against pinned digests and writes them atomically, and
//! leaves everything else in that directory alone.
//!
//! Recognition is lazy on every platform: nothing loads at boot, the first
//! operation that needs the engine loads it ([`engine::spawn_background_init`]).
//! The surfaces are the composer's microphone, a Telegram voice note or audio
//! attachment, the audio recorded after the wake word, and `mahbot audio
//! transcribe`.

pub(crate) mod decode;
pub(crate) mod engine;
pub(crate) mod models;

pub(crate) use decode::{AUDIO_FORMATS, takes_audio};
pub(crate) use engine::{
    INFERENCE_TIMEOUT, InferenceTimedOut, UNTRANSCRIBED, ensure_ready_once, is_failed, is_loaded,
    is_loading, retry_init, spawn_background_init, state, transcribe_file_async,
    transcribe_samples,
};
