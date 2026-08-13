//! Audio/voice subsystem — wake word detection, voice command pipeline,
//! local transcription (Qwen3-ASR), and text-to-speech.
//!
//! All audio-related modules are consolidated here under `crate::audio::*`.

pub(crate) mod audio_preprocessor;
pub(crate) mod embedding_sequence;
pub(crate) mod local_transcriber;
pub mod tts;
pub mod voice;
pub(crate) mod wake_word_classifier;

use anyhow::{Result, anyhow};
use candle_core::Tensor;
use std::collections::HashMap;
use std::path::PathBuf;

pub(crate) fn onnx_input_name(model: &candle_onnx::onnx::ModelProto) -> String {
    model
        .graph
        .as_ref()
        .and_then(|g| g.input.first())
        .map_or_else(|| "input".to_string(), |i| i.name.clone())
}

pub(crate) fn onnx_output_name(model: &candle_onnx::onnx::ModelProto) -> String {
    model
        .graph
        .as_ref()
        .and_then(|g| g.output.first())
        .map_or_else(|| "output".to_string(), |o| o.name.clone())
}

pub(crate) fn extract_output(
    mut outputs: HashMap<String, Tensor>,
    model: &candle_onnx::onnx::ModelProto,
    label: &str,
) -> Result<Tensor> {
    let name = onnx_output_name(model);
    outputs
        .remove(&name)
        .ok_or_else(|| anyhow!("{label}: output '{name}' not found"))
}

/// Resolve a per-model subdirectory under the shared `~/.mahbot/models/` root
/// (TTS, ASR, and wake-word models each use their own subdirectory).
pub(crate) fn models_subdir(name: &str) -> Option<PathBuf> {
    crate::util::models_dir().map(|dir| dir.join(name))
}
