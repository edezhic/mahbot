//! Audio/voice subsystem — wake word detection, voice command pipeline,
//! local transcription (Qwen3-ASR), and text-to-speech.
//!
//! All audio-related modules are consolidated here under `crate::audio::*`.
//!
//! Wake word detection runs entirely on the shared Qwen3-ASR encoder
//! ([`wake_word`]) — no separate embedding model, no trainable head, no
//! AGC/NS preprocessing.

pub mod local_transcriber;
pub mod tts;
pub mod voice;
pub(crate) mod wake_word;

use anyhow::{Context as _, Result, anyhow};
use candle_core::Tensor;
use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::AsyncFn;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;
use tracing::{info, warn};

use crate::util::model_state::{AtomicModelState, ModelLoadGuard, ModelState};

pub(crate) fn onnx_output_name(model: &crate::onnx::Model) -> String {
    model.output_name().to_string()
}

pub(crate) fn extract_output(
    mut outputs: HashMap<String, Tensor>,
    model: &crate::onnx::Model,
    label: &str,
) -> Result<Tensor> {
    let name = onnx_output_name(model);
    outputs
        .remove(&name)
        .ok_or_else(|| anyhow!("{label}: output '{name}' not found"))
}

/// Resolve a per-model subdirectory under the shared `~/.mahbot/models/` root
/// (TTS and ASR each use their own subdirectory; the wake-word pipeline has
/// no model of its own — it shares the ASR model).
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

/// Verify-then-download helper shared by the TTS and voice model sets.
///
/// Checks `path` before downloading (SHA256 when `sha256` is non-empty,
/// else `min_size`), re-downloading via
/// [`download_verified`](crate::util::http::download_verified) when
/// missing/corrupt/too small; returns `true` iff a download ran. `client`
/// is reused when `Some`; when `None` one is built (with `timeout`) only on
/// the download path. `timeout` maps to a per-request timeout. `label` names
/// the file in logs (must include the file name); `on_progress` runs during
/// downloads only.
///
/// Deliberately not used by `local_transcriber.rs` (no-timeout client,
/// [`DownloadSizeCheck::None`](crate::util::http::DownloadSizeCheck::None),
/// spawn_blocking pre-check) or `embedder.rs` (existence-only pre-check,
/// [`DownloadSizeCheck::Exact`](crate::util::http::DownloadSizeCheck::Exact)).
#[expect(clippy::cast_precision_loss, clippy::too_many_arguments)]
pub(crate) async fn ensure_downloaded(
    client: Option<&reqwest::Client>,
    path: &Path,
    url: &str,
    sha256: &str,
    min_size: u64,
    timeout: Duration,
    label: &str,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<bool> {
    if path.exists() {
        if sha256.is_empty() {
            let meta = tokio::fs::metadata(path).await?;
            if meta.len() >= min_size {
                return Ok(false);
            }
            warn!(
                "{label} too small ({} bytes), re-downloading: {}",
                meta.len(),
                path.display()
            );
        } else if let Err(e) = crate::util::verify_sha256(path, sha256) {
            warn!("{label} corrupt, re-downloading {}: {e}", path.display());
        } else {
            return Ok(false);
        }
        tokio::fs::remove_file(path).await?;
    }

    info!("Downloading {label}...");
    let client = match client {
        Some(c) => Cow::Borrowed(c),
        None => Cow::Owned(
            crate::util::http::build_download_client(timeout)
                .context("Failed to build HTTP client")?,
        ),
    };
    // Byte count comes from the progress closure (download_verified's
    // documented cumulative-bytes contract) — no post-success failure surface.
    let mut size = 0u64;
    crate::util::http::download_verified(
        &client,
        url,
        path,
        sha256,
        Some(timeout),
        crate::util::http::DownloadSizeCheck::Min(min_size),
        |d, total| {
            size = d;
            on_progress(d, total);
        },
    )
    .await?;
    info!("Downloaded {label} ({:.1} MB)", size as f64 / 1_048_576.0);
    Ok(true)
}
