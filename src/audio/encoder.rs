//! The wake word's Qwen3-ASR encoder: the `qwen-asr` weights the wake-word pipeline encodes its
//! windows with ([`crate::audio::wake_word`]), loaded from a per-user folder of their own.
//!
//! Recognition no longer runs here — it is [`crate::speech`], on every platform — so this module
//! is the encoder alone, and it is retired once the wake word is ported off `qwen-asr`. It is
//! macOS-only like the wake word it serves.
//!
//! The weights (~1.88 GB BF16 safetensors) live in `~/.mahbot/models/qwen3-asr-0.6b/`, one folder
//! of the encoder's own rather than the recogniser's machine-wide model set, and are memory-mapped
//! by `qwen-asr` (zero-copy weights, minimal RSS).
//!
//! Boot never preloads it: the first demand — the voice pipeline's own ([`crate::audio::voice`]) or
//! the Settings toggle — starts the load-or-download chain in the background
//! ([`spawn_background_init`]), which the voice pipeline's gate reads back through
//! [`is_loaded`]. `Failed` is terminal; the pipeline's bounded auto-retry and the GUI retry button
//! ([`retry_init`]) are the only recovery knobs.
//!
//! Files are SHA256-verified **once, at download time** — there is no per-use re-verification of
//! the ~1.9 GB model. A silently corrupted cache is caught at load time instead:
//! `QwenCtx::load` returns `None` and init falls through to the download+verify recovery loop. The
//! load is mmap-based, so partial corruption can survive it and surface as a SIGBUS at inference
//! rather than a graceful fallback — accepted tradeoff. Missing/corrupt files trigger the retry
//! loop with exponential backoff.

use crate::audio::models_subdir;
use crate::util::UnwrapPoison;
use crate::util::model_state::{AtomicModelState, ModelLoadGuard, ModelState};
use anyhow::{Context, Result};
use futures_util::FutureExt;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{debug, info, warn};

const MODEL_REPO: &str = "Qwen/Qwen3-ASR-0.6B";

pub(crate) const MODEL_DIR_NAME: &str = "qwen3-asr-0.6b";

/// Filenames required by `qwen-asr`: `QwenCtx::load()` expects `model*.safetensors` and
/// `vocab.json` in the model directory; `merges.txt` is the BPE tokenizer's.
pub(crate) const MODEL_FILENAME: &str = "model.safetensors";
pub(crate) const VOCAB_FILENAME: &str = "vocab.json";
pub(crate) const MERGES_FILENAME: &str = "merges.txt";

/// SHA256 checksums for download integrity verification, taken from the HuggingFace repository
/// metadata. If they drift (HF re-uploads), users see a SHA256 mismatch and the model is
/// re-downloaded.
///
/// To update these constants (upstream model change): download the new files, run
/// `shasum -a 256` on each, replace the `*_SHA256` values, and update [`MODEL_FILENAME`],
/// [`VOCAB_FILENAME`], [`MERGES_FILENAME`] and the URLs in [`download_file`] if the names changed.
const MODEL_SHA256: &str = "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea";
const VOCAB_SHA256: &str = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910";
const MERGES_SHA256: &str = "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5";

/// Download timeout for the 1.88 GB model file (30 minutes); the smaller files finish far sooner
/// under this timeout because the stream is shared.
const MODEL_DOWNLOAD_TIMEOUT: Duration = Duration::from_mins(30);

/// Download timeout for config/vocabulary files (< 10 MB).
const SMALL_FILE_TIMEOUT: Duration = Duration::from_mins(1);

/// Retry sleep base: 5 seconds, doubled per attempt.
const DOWNLOAD_RETRY_BASE_SECS: u64 = 5;

/// Maximum number of download retry attempts.
const MAX_DOWNLOAD_RETRIES: u32 = 12;

/// Global singleton handle — `None` until loaded.
static GLOBAL: Mutex<Option<Encoder>> = Mutex::new(None);

/// Shared-model slot for the wake-word pipeline, populated by [`set_ready`] alongside [`GLOBAL`].
/// The wake-word path reads this slot instead of locking the context mutex, which an inference
/// holds for its full duration; cloning an `Arc` under this short mutex is O(1) and uncontended
/// (single writer on swap).
static SHARED_MODEL: Mutex<Option<Arc<qwen_asr::context::QwenModel>>> = Mutex::new(None);

/// Atomic state tracker to coordinate lazy initialization: Uninit → Loading → Ready / Failed.
/// Failed is terminal (the wake word cannot encode); the voice pipeline's bounded auto-retry and
/// the explicit GUI retry ([`crate::audio::voice::VoiceCommand::RetryModelLoading`]) are the only
/// recovery knobs.
static STATE: AtomicModelState = AtomicModelState::new(ModelState::Uninit);

/// Atomically store a ready encoder and transition state to [`ModelState::Ready`].
fn set_ready(encoder: Encoder) {
    // Populate the shared-model slot BEFORE publishing the encoder so the wake-word path can never
    // observe Ready without a model. The ctx mutex is uncontended here: the encoder has just been
    // loaded and not yet published, so nothing can be running on it.
    *SHARED_MODEL.lock().unwrap_poison() = Some(encoder.model_arc());
    *GLOBAL.lock().unwrap_poison() = Some(encoder);
    STATE.store(ModelState::Ready, Ordering::Release);
}

/// The Qwen3-ASR encoder the wake word encodes its windows with: a `qwen-asr` inference context
/// behind a shared handle.
///
/// The inner `QwenCtx` is not `Sync` (it holds a `Box<dyn Fn + Send>` token callback), so it is
/// wrapped in a `Mutex`; the `Arc` lets the global singleton's outer lock be released right after
/// cloning, so background download completion ([`set_ready`]) never contends.
pub struct Encoder {
    ctx: Arc<Mutex<qwen_asr::context::QwenCtx>>,
}

impl Encoder {
    /// Load the model from an explicit directory path. Unlike the background chain this does not
    /// resolve the directory via [`crate::audio::models_subdir`], so it does not need the
    /// CONFIG storage root to be set.
    fn try_load_from(dir: &Path) -> Option<Self> {
        let dir_str = dir.to_string_lossy().to_string();
        // The context is opened for its weights alone: the wake word encodes windows with
        // `model`, and the transcription settings a full pass would use are none of its business.
        let ctx = qwen_asr::context::QwenCtx::load(&dir_str)?;
        Some(Self {
            ctx: Arc::new(Mutex::new(ctx)),
        })
    }

    /// Clone the shared immutable [`qwen_asr::context::QwenModel`] Arc held by this context.
    ///
    /// Called once by [`set_ready`] to populate [`SHARED_MODEL`] — NOT per scoring stride (that
    /// would lock the context mutex, held for a whole pass). The encoder weights are read-only
    /// after load, so concurrent `Encoder::forward` calls with per-call scratch buffers are safe;
    /// the wake-word path passes `None` for `enc_bufs`.
    fn model_arc(&self) -> Arc<qwen_asr::context::QwenModel> {
        Arc::clone(&self.ctx.lock().unwrap_poison().model)
    }
}

struct ModelFile {
    filename: &'static str,
    url: String,
    expected_sha256: &'static str,
    timeout: Duration,
}

fn model_url(filename: &str) -> String {
    format!("https://huggingface.co/{MODEL_REPO}/resolve/main/{filename}")
}

fn model_files() -> [ModelFile; 3] {
    [
        ModelFile {
            filename: MODEL_FILENAME,
            url: model_url(MODEL_FILENAME),
            expected_sha256: MODEL_SHA256,
            timeout: MODEL_DOWNLOAD_TIMEOUT,
        },
        ModelFile {
            filename: VOCAB_FILENAME,
            url: model_url(VOCAB_FILENAME),
            expected_sha256: VOCAB_SHA256,
            timeout: SMALL_FILE_TIMEOUT,
        },
        ModelFile {
            filename: MERGES_FILENAME,
            url: model_url(MERGES_FILENAME),
            expected_sha256: MERGES_SHA256,
            timeout: SMALL_FILE_TIMEOUT,
        },
    ]
}

/// Download a file from `url` to `dest`, verifying SHA256. On mismatch the partial file is
/// removed so the next attempt re-downloads from scratch.
async fn download_file(client: &reqwest::Client, file: &ModelFile, dest: &Path) -> Result<()> {
    #[expect(clippy::cast_precision_loss)]
    fn calc_pct(downloaded: u64, total_size: u64) -> f64 {
        (downloaded as f64 / total_size as f64 * 100.0).min(100.0)
    }

    crate::util::http::download_verified(
        client,
        &file.url,
        dest,
        file.expected_sha256,
        Some(file.timeout),
        crate::util::http::DownloadSizeCheck::None,
        |downloaded, total_size| {
            if total_size > 0 {
                let pct = calc_pct(downloaded, total_size);
                debug!(
                    "Downloading {}: {:.0}% ({}/{} MB)",
                    file.filename,
                    pct,
                    downloaded / 1_048_576,
                    total_size / 1_048_576,
                );
            }
        },
    )
    .await
    .with_context(|| format!("Failed to download {}", file.filename))?;

    info!("Downloaded {} ({})", file.filename, file.expected_sha256);
    Ok(())
}

/// Create a reqwest client with sensible defaults for model downloads.
///
/// Deliberately not [`crate::util::http::build_download_client`] — no client
/// timeouts: per-request timeouts already bound each download (30 min model,
/// 1 min small files) and the custom user-agent is kept. Connect-phase
/// fail-fast is intentionally absent.
fn download_client() -> Result<reqwest::Client> {
    crate::util::http::install_ring_provider();
    reqwest::Client::builder()
        .user_agent("mahbot/0.3.0 (qwen-asr model downloader)")
        .build()
        .context("Failed to create HTTP client for model download")
}

/// Fetch the whole model set once: every file, verified as it streams, with a
/// file already on disk re-verified and re-downloaded only when it does not
/// match. No retry and no backoff — the caller decides what a failure means.
/// `fetch_starting` is called before the first file actually fetched, so a
/// caller can bind a notice to a fetch that really begins.
async fn download_once(
    client: &reqwest::Client,
    dir: &Path,
    files: &[ModelFile],
    fetch_starting: &mut impl FnMut(),
) -> Result<()> {
    for file in files {
        let dest = dir.join(file.filename);
        if dest.exists() {
            // Verify the existing file on a blocking thread (SHA256 reads the full 1.88 GB).
            let dest_clone = dest.clone();
            let expected = file.expected_sha256.to_string();
            let checksum_ok = tokio::task::spawn_blocking(move || {
                crate::util::verify_sha256(&dest_clone, &expected).is_ok()
            })
            .await
            .unwrap_or_else(|join_err| {
                warn!("Wake word encoder: SHA256 verification task panicked: {join_err}");
                false
            }); // JoinError → log and treat as checksum mismatch, will re-download

            if checksum_ok {
                continue;
            }
            warn!(
                "Wake word encoder: {} SHA256 mismatch, re-downloading",
                file.filename
            );
            tokio::fs::remove_file(&dest).await.ok();
        }

        info!("Wake word encoder: downloading {}", file.filename);
        fetch_starting();

        // download_file already verified SHA256 while streaming and removed the file on mismatch,
        // so no re-verification is needed on success.
        if let Err(e) = download_file(client, file, &dest).await {
            warn!(
                "Wake word encoder: failed to download {}: {e}",
                file.filename
            );
            tokio::fs::remove_file(&dest).await.ok();
            return Err(e);
        }
    }

    Ok(())
}

/// Background download loop with retry: downloads all model files sequentially, sleeps with
/// exponential backoff on failure and retries up to [`MAX_DOWNLOAD_RETRIES`] times, then loads the
/// model and publishes it.
async fn download_retry_loop() {
    // Loading → Failed on drop (panic/cancel safety). No-op on the terminal states the
    // success/retry-cap paths set.
    let _guard = ModelLoadGuard::new(&STATE);
    let Some(dir) = models_subdir(MODEL_DIR_NAME) else {
        warn!("Wake word encoder: cannot resolve model directory (storage root not set)");
        STATE.store(ModelState::Failed, Ordering::Release);
        return;
    };

    let client = match download_client() {
        Ok(c) => c,
        Err(e) => {
            warn!("Wake word encoder: failed to create HTTP client: {e}");
            STATE.store(ModelState::Failed, Ordering::Release);
            return;
        }
    };

    tokio::fs::create_dir_all(&dir).await.ok();

    let files = model_files();
    let mut attempt: u32 = 0;

    loop {
        attempt += 1;
        info!("Wake word encoder: model download attempt {attempt}/{MAX_DOWNLOAD_RETRIES}");

        if download_once(&client, &dir, &files, &mut || {})
            .await
            .is_ok()
        {
            // All files downloaded and verified. Load the model on a blocking thread.
            info!("Wake word encoder: all model files downloaded, loading...");
            let dir_for_load = dir.clone();
            let loaded = tokio::task::spawn_blocking(move || Encoder::try_load_from(&dir_for_load))
                .await
                .ok()
                .flatten();
            if let Some(tc) = loaded {
                info!("Wake word encoder: Qwen3-ASR model loaded successfully");
                set_ready(tc);
                return;
            }
            warn!(
                "Wake word encoder: model files present but failed to load — deleting and re-downloading"
            );
            // Delete the corrupt files so the next attempt re-downloads and re-verifies them.
            for f in &files {
                let dest = dir.join(f.filename);
                tokio::fs::remove_file(&dest).await.ok();
            }
        }

        if attempt >= MAX_DOWNLOAD_RETRIES {
            warn!("Wake word encoder: max retries ({MAX_DOWNLOAD_RETRIES}) reached, giving up");
            STATE.store(ModelState::Failed, Ordering::Release);
            return;
        }

        let sleep_secs = DOWNLOAD_RETRY_BASE_SECS * (1u64 << (attempt - 1).min(8));
        let sleep_dur = Duration::from_secs(sleep_secs.min(300));
        warn!(
            "Wake word encoder: retrying in {}s (attempt {attempt}/{MAX_DOWNLOAD_RETRIES})",
            sleep_dur.as_secs()
        );
        tokio::time::sleep(sleep_dur).await;
    }
}

/// Load the encoder from an already-resolved cache directory on a blocking thread (memory-map
/// + weight setup). `None` when any file is missing or the load fails.
async fn load_from_cache(dir: PathBuf) -> Option<Encoder> {
    tokio::task::spawn_blocking(move || Encoder::try_load_from(&dir))
        .await
        .ok()
        .flatten()
}

fn cache_present(dir: &Path) -> bool {
    dir.join(MODEL_FILENAME).exists()
        && dir.join(VOCAB_FILENAME).exists()
        && dir.join(MERGES_FILENAME).exists()
}

/// Background load-or-download chain — spawned by [`spawn_background_init`], never awaited.
///
/// STATE must already be `Loading` (the caller owns the `Uninit → Loading` transition); this
/// function owns every later one.
async fn init_background(dir: PathBuf) {
    if cache_present(&dir) {
        if let Some(tc) = load_from_cache(dir).await {
            info!("Wake word encoder: loaded from cache");
            set_ready(tc);
            return;
        }
        warn!("Wake word encoder: cached files present but failed to load — re-downloading");
    }

    info!("Wake word encoder: model not cached, downloading in background");
    download_retry_loop().await;
}

/// Try to shortcut or acquire the initialisation lock. `None` means the caller acquired the lock
/// and should proceed; `Some(true)` means already initialised; `Some(false)` means an init is
/// already in flight.
fn try_lock_init() -> Option<bool> {
    match STATE.load(Ordering::Acquire) {
        ModelState::Ready => return Some(true),
        ModelState::Uninit => {}
        _ => return Some(false),
    }

    if !STATE.transition(ModelState::Uninit, ModelState::Loading) {
        return Some(false);
    }

    None
}

/// Kick off the encoder's load-or-download chain as a background task.
///
/// The caller has already decided the encoder is needed (the voice stack's first-use demand or an
/// explicit retry); nothing here is awaited, so the ~4 s load (or a full download) runs off the
/// caller's path. Boot never calls it: starting the process is not a demand.
///
/// No-op when the encoder is already ready/loading, or when a previous attempt failed (Failed is
/// terminal — only the voice pipeline's bounded auto-retry or [`retry_init`] re-attempts).
/// Panic-safe: a panic in the spawned chain transitions STATE to [`ModelState::Failed`].
pub fn spawn_background_init() {
    if let Some(result) = try_lock_init() {
        // Ready (true) — already loaded. Loading/Failed (false) — an init is in flight or a
        // previous attempt failed terminally; nothing to start.
        debug!(result, "Wake word encoder: background init skipped");
        return;
    }

    let Some(dir) = models_subdir(MODEL_DIR_NAME) else {
        warn!("Wake word encoder: cannot resolve model directory (storage root not set)");
        STATE.store(ModelState::Failed, Ordering::Release);
        return;
    };

    tokio::spawn(async move {
        // A panic must never leave STATE stuck in Loading (silently disabling the wake word until
        // restart) — store the honest terminal state.
        let result = AssertUnwindSafe(init_background(dir)).catch_unwind().await;
        if result.is_err() {
            warn!("Wake word encoder: background init panicked — marking failed");
            STATE.store(ModelState::Failed, Ordering::Release);
        }
    });
}

pub fn is_loaded() -> bool {
    STATE.is_ready()
}

/// True while the encoder's load is in flight (`Loading`).
///
/// Nothing preloads the model, so "not loaded" alone cannot tell a load in flight from one that
/// has not started yet: this distinguishes the two for the voice pipeline's own demand report.
pub(crate) fn is_loading() -> bool {
    STATE.load(Ordering::Acquire) == ModelState::Loading
}

/// True if the encoder failed to load (terminal `Failed` state).
///
/// Used by the voice pipeline's model gating to surface
/// [`crate::audio::voice::VoiceStatus::ModelError`] when the wake word cannot encode.
pub fn is_failed() -> bool {
    STATE.load(Ordering::Acquire) == ModelState::Failed
}

/// Clone the shared Qwen3-ASR model Arc the wake-word pipeline encodes with.
///
/// Returns `None` when the encoder is not loaded (or still loading). The returned model shares the
/// exact same loaded weights the context holds — the wake-word pipeline performs no separate
/// download or model handling.
///
/// Reads the lock-free [`SHARED_MODEL`] slot populated by [`set_ready`] — it NEVER touches the
/// context mutex. The wake-word stride path calls this on every scoring step; locking the ctx
/// there would hold the encoder for the length of any pass.
pub fn shared_model_arc() -> Option<Arc<qwen_asr::context::QwenModel>> {
    SHARED_MODEL.lock().unwrap_poison().clone()
}

/// Retry initialisation after a terminal `Failed` state.
///
/// The default lifecycle treats `Failed` as terminal — no automatic
/// re-attempt.  Two explicit recovery knobs exist: the voice
/// pipeline's bounded periodic auto-retry (at most `MAX_AUTO_MODEL_RETRY_CYCLES`
/// cycles per session) and the GUI retry button
/// ([`VoiceCommand::RetryModelLoading`]), which calls this directly and
/// bypasses the auto-retry budget.  This function resets `Failed` → `Uninit`
/// and re-runs the background init chain (load-or-download).  Returns `true`
/// if a retry was initiated; `false` when the state was not `Failed` or the
/// claim was lost.
pub fn retry_init() -> bool {
    if !STATE.transition(ModelState::Failed, ModelState::Uninit) {
        return false;
    }
    spawn_background_init();
    true
}
