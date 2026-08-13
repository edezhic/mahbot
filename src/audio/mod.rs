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
use std::ops::AsyncFn;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;
use tracing::warn;

use crate::util::model_state::{AtomicModelState, ModelLoadGuard, ModelState};

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

/// Retry cap shared by the voice and TTS model download loops.
const MAX_DOWNLOAD_RETRIES: u32 = 10;

/// Shared download-retry skeleton for the voice and TTS model sets: owns the
/// [`ModelLoadGuard`] and dir resolution, bounded loop (Ready pre-check, retry
/// cap, download-only timeout, Failed re-check, 5s→2min backoff). `on_retry_cap`
/// owns `state.store(Failed)` (module-specific tail ordering); dir-resolution
/// failure stores Failed here without a hook. The `load` closure must leave the
/// state terminal (store Ready) on success, or the guard's Loading→Failed drop
/// fires on return.
pub(crate) async fn run_download_retry_loop<D, L, F>(
    state: &AtomicModelState,
    dir_name: &str,
    label: &str,
    timeout: Duration,
    download: D,
    load: L,
    on_retry_cap: F,
) where
    D: AsyncFn(&Path) -> anyhow::Result<()>,
    L: Fn(&Path) -> anyhow::Result<()>,
    F: FnOnce(),
{
    // Transitions Loading→Failed on drop if the loop is cancelled or panics.
    let _guard = ModelLoadGuard::new(state);
    let Some(dir) = models_subdir(dir_name) else {
        warn!("{label}: cannot resolve model directory");
        state.store(ModelState::Failed, Ordering::Release);
        return;
    };

    let mut retry_delay = Duration::from_secs(5);
    let mut retry_count = 0u32;

    loop {
        if state.load(Ordering::Acquire) == ModelState::Ready {
            return;
        }
        retry_count += 1;
        if retry_count > MAX_DOWNLOAD_RETRIES {
            on_retry_cap();
            return;
        }

        match tokio::time::timeout(timeout, download(&dir)).await {
            Ok(Ok(())) => match load(&dir) {
                Ok(()) => return,
                Err(e) => warn!("Failed to load {label} models (will retry): {e}"),
            },
            Ok(Err(e)) => warn!("Failed to download {label} models (will retry): {e}"),
            Err(_) => warn!("{label} download timed out (will retry)"),
        }

        if state.load(Ordering::Acquire) == ModelState::Failed {
            return;
        }
        tokio::time::sleep(retry_delay).await;
        retry_delay = (retry_delay * 2).min(Duration::from_mins(2));
    }
}
