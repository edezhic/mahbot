//! Audio/voice subsystem — wake word detection, voice command pipeline,
//! local transcription (Qwen3-ASR), text-to-speech, and speaker verification.
//!
//! All audio-related modules are consolidated here under `crate::audio::*`.

pub(crate) mod audio_preprocessor;
pub(crate) mod embedding_sequence;
pub(crate) mod local_transcriber;
pub mod tts;
pub mod voice;
#[cfg_attr(not(test), allow(clippy::cast_possible_wrap))]
pub(crate) mod voice_verifier;
pub(crate) mod wake_word_classifier;
