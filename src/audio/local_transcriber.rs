//! Local audio transcription using Qwen3-ASR via the `qwen-asr` crate.
//!
//! # Architecture
//!
//! This module is the **only** audio transcription path — the previous
//! API-based [`AudioTranscriber`] has been removed.  When the model is
//! available, inference runs fully locally with Qwen3-ASR-0.6B.  The model
//! (~1.88 GB BF16 safetensors) is downloaded on first use to
//! `~/.mahbot/models/qwen3-asr-0.6b/` and memory-mapped by the `qwen-asr`
//! crate — zero-copy weight loading with minimal RSS overhead.
//!
//! # Download strategy
//!
//! Follows the same lazy-download pattern as [`crate::embedder`]: synchronous
//! cache check on first use, then background retry loop with exponential
//! backoff if files are missing. SHA256 integrity verification is performed
//! after each download.
//!
//! # Audio format conversion
//!
//! The qwen-asr crate expects raw f32 PCM samples at 16 kHz mono. Telegram
//! delivers voice messages as OGG (Opus) and audio files as MP3 (among other
//! formats). This module decodes these formats into 16 kHz mono f32 samples
//! before passing them to qwen-asr's `transcribe_audio()`:
//!
//! * **OGG/Opus** — decoded via the `ogg` crate (OGG demuxer) and `opus-decoder`
//!   crate (pure-Rust Opus decoder, no C dependencies).
//! * **MP3** — decoded via the `minimp3` crate (Rust wrapper wrapping a C library
//!   via `minimp3-sys`; requires a C compiler at build time).
//! * **WAV** — decoded via `qwen_asr::audio::parse_wav_buffer()` (qwen-asr's
//!   built-in parser, which handles resampling from any sample rate).
//!
//! # Call site
//!
//! The [`transcribe_file_async`] function is called from
//! [`crate::channels::enrichment::transcribe_audio_marker`].  When the local
//! model is unavailable or disabled in config, the caller annotates the
//! message with just the audio-transcription icon combo — there is no API
//! fallback.

use crate::util::UnwrapPoison;
use crate::util::model_state::{AtomicModelState, ModelState};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{debug, info, warn};

// ── Model metadata ────────────────────────────────────────────────────

/// HuggingFace repo for the Qwen3-ASR 0.6B model.
const MODEL_REPO: &str = "Qwen/Qwen3-ASR-0.6B";

/// Local subdirectory under `~/.mahbot/models/` where the model is stored.
pub(crate) const MODEL_DIR_NAME: &str = "qwen3-asr-0.6b";

/// Filenames required by `qwen-asr`.
///
/// The qwen-asr crate's `QwenCtx::load()` expects `model*.safetensors` and
/// `vocab.json` in the model directory. We also download `merges.txt` for the
/// BPE tokenizer.
pub(crate) const MODEL_FILENAME: &str = "model.safetensors";
pub(crate) const VOCAB_FILENAME: &str = "vocab.json";
pub(crate) const MERGES_FILENAME: &str = "merges.txt";

/// SHA256 checksums for download integrity verification.
///
/// Obtained from the HuggingFace repository metadata and verified at download
/// time. If these drift (HF re-uploads), users will see a SHA256 mismatch error
/// and the model will be re-downloaded.
///
/// # Updating checksums
///
/// To update these constants (e.g., when the upstream model version changes):
///
/// 1. Download the new model files manually (or let the automatic download
///    complete after updating the URLs).
/// 2. Compute the SHA256 of each file:
///    ```sh
///    shasum -a 256 ~/.mahbot/models/qwen3-asr-0.6b/model.safetensors
///    shasum -a 256 ~/.mahbot/models/qwen3-asr-0.6b/vocab.json
///    shasum -a 256 ~/.mahbot/models/qwen3-asr-0.6b/merges.txt
///    ```
/// 3. Replace the corresponding `*_SHA256` constants below with the new hashes.
/// 4. If the filenames changed, also update [`MODEL_FILENAME`], [`VOCAB_FILENAME`],
///    [`MERGES_FILENAME`], and the download URLs in [`download_file`].
const MODEL_SHA256: &str = "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea";
const VOCAB_SHA256: &str = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910";
const MERGES_SHA256: &str = "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5";

/// Default inference timeout (10 minutes).
///
/// Qwen3-ASR-0.6B inference on short audio clips (< 30s) completes in
/// seconds.  Longer recordings (e.g. voice memos, meetings) may take
/// several minutes.  This timeout prevents a hung inference from
/// permanently occupying a tokio blocking thread.
///
/// Shared by the voice pipeline and the audio-enrichment path.
pub(crate) const INFERENCE_TIMEOUT: Duration = Duration::from_mins(10);

/// Download timeout for the 1.88 GB model file (30 minutes).
/// The smaller files (vocab.json, merges.txt) complete far sooner under this
/// timeout because the stream is shared.
const MODEL_DOWNLOAD_TIMEOUT: Duration = Duration::from_mins(30);

/// Download timeout for config/vocabulary files (< 10 MB).
const SMALL_FILE_TIMEOUT: Duration = Duration::from_mins(1);

/// Retry sleep base: 5 seconds, doubled per attempt.
const DOWNLOAD_RETRY_BASE_SECS: u64 = 5;

/// Maximum number of download retry attempts.
const MAX_DOWNLOAD_RETRIES: u32 = 12;

// ── State machine ─────────────────────────────────────────────────────

/// Global singleton handle — `None` until loaded.
static GLOBAL_TRANSCRIBER: Mutex<Option<QwenLocalTranscriber>> = Mutex::new(None);

/// Atomic state tracker to coordinate lazy initialization.
///
/// Model-loading lifecycle for the transcriber (Uninit → Loading → Ready /
/// Failed).  Shared wrapper extracted in
/// [`crate::util::model_state::ModelState`].  Failed is terminal — a restart
/// is required to retry (the [`transcribe_file_async`] error path leaves the
/// caller with an icon-only audio annotation).
static STATE: AtomicModelState = AtomicModelState::new(ModelState::Uninit);

/// Atomically store a ready transcriber and transition state to [`ModelState::Ready`].
fn set_transcriber_ready(tc: QwenLocalTranscriber) {
    *GLOBAL_TRANSCRIBER.lock().unwrap_poison() = Some(tc);
    STATE.store(ModelState::Ready, Ordering::Release);
}

// ── Model directory resolution ────────────────────────────────────────

/// Resolve the local model directory (`~/.mahbot/models/qwen3-asr-0.6b/`).
fn model_dir() -> Option<PathBuf> {
    crate::util::models_dir().map(|dir| dir.join(MODEL_DIR_NAME))
}

/// The Qwen3-ASR local transcriber.
///
/// Wraps a `qwen-asr` inference context behind a high-level `transcribe_file`
/// method that handles audio format decoding, resampling, and inference.
///
/// # Thread safety
///
/// The inner `QwenCtx` is not `Sync` (it contains a `Box<dyn Fn + Send>`
/// token callback), so the inner context is wrapped in a `Mutex` for interior
/// mutability. The `Arc` allows the global singleton's outer lock to be
/// released immediately after cloning the handle, preventing lock contention
/// with background download completion ([`set_transcriber_ready`]).
pub struct QwenLocalTranscriber {
    ctx: Arc<Mutex<qwen_asr::context::QwenCtx>>,
}

impl QwenLocalTranscriber {
    /// Load the model from an explicit directory path.
    ///
    /// Unlike [`try_init_from_cache`](super::try_init_from_cache), this does
    /// not resolve the directory via [`model_dir`] and therefore does not
    /// depend on the CONFIG storage root being set.
    fn try_load_from(dir: &Path) -> Option<Self> {
        let dir_str = dir.to_string_lossy().to_string();
        let mut ctx = qwen_asr::context::QwenCtx::load(&dir_str)?;
        ctx.want_language_detection = true;
        // Split audio longer than ~30 s at low-energy boundaries. Without
        // segmentation a single segment caps at 2048 decoded tokens, silently
        // truncating long recordings (voice pipeline allows up to 10 minutes).
        ctx.segment_sec = 30.0;
        Some(Self {
            ctx: Arc::new(Mutex::new(ctx)),
        })
    }

    /// Clone the inner `Arc` so callers can release the outer
    /// [`GLOBAL_TRANSCRIBER`] lock before running inference.
    fn clone_arc(&self) -> Arc<Mutex<qwen_asr::context::QwenCtx>> {
        Arc::clone(&self.ctx)
    }
}

/// Asynchronously transcribe an audio file on a blocking thread.
///
/// Decodes the audio file and runs Qwen3-ASR inference on a dedicated
/// blocking thread via [`tokio::task::spawn_blocking`], preventing the
/// CPU-heavy work from stalling the async runtime.
///
/// # Lock scoping
///
/// The outer [`GLOBAL_TRANSCRIBER`] lock is held only long enough to clone the
/// inner [`Arc<Mutex<QwenCtx>>`] handle, then released before inference begins.
/// This prevents lock contention with [`set_transcriber_ready`] — if a
/// background download completes mid-transcription, the new transcriber can be
/// swapped in without waiting for inference to finish.
///
/// # Timeout
///
/// `inference_timeout` controls the maximum wall-clock time for the ONNX
/// inference step (audio decoding is excluded from the timeout).  Both
/// callers — the voice pipeline (recordings up to 10 minutes) and the
/// enrichment path for attached audio — pass [`INFERENCE_TIMEOUT`].
///
/// A shutdown guard races against the inference so that a stalled model
/// cannot block pipeline exit.
///
/// Returns an error if the local model is unavailable; the caller annotates
/// the message with just the audio-transcription icon combo on failure.
pub async fn transcribe_file_async(path: &Path, inference_timeout: Duration) -> Result<String> {
    let owned = path.to_owned();

    // Step 1: Decode audio on a blocking thread.
    let samples = tokio::task::spawn_blocking(move || decode_audio_to_mono_f32(&owned))
        .await
        .context("Audio decode task panicked")?
        .context("Failed to decode audio file to 16 kHz mono PCM")?;

    if samples.is_empty() {
        anyhow::bail!("Audio file is empty after decoding");
    }

    // Step 2: Clone the inference handle while holding the outer lock,
    // then release the outer lock before running inference.
    // std::sync::Mutex is fine here — lock hold time is nanoseconds
    // (clone Arc + check Option), and contention is rare (only during
    // background download completion).
    let ctx_arc = {
        let guard = GLOBAL_TRANSCRIBER.lock().unwrap_poison();
        let tc = guard.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Local transcriber not available during async transcription")
        })?;
        tc.clone_arc()
    };

    // Step 3: Run inference with a timeout — the outer lock was released
    // after the clone.  The timeout prevents a hung inference from
    // permanently occupying a tokio blocking thread.  A shutdown guard
    // ensures the inference cannot block pipeline exit.
    let ctx_arc2 = ctx_arc;
    let shutdown_token = crate::shutdown::shutdown_token();
    let text = tokio::select! {
        result = tokio::time::timeout(inference_timeout, async move {
            tokio::task::spawn_blocking(move || {
                let mut ctx = ctx_arc2.lock().unwrap_poison();
                qwen_asr::transcribe::transcribe_audio(&mut ctx, &samples)
                    .ok_or_else(|| anyhow::anyhow!("Qwen3-ASR inference returned no output"))
            })
            .await
            .context("Inference task panicked")?
        }) => {
            result.context("Qwen3-ASR inference timed out")?
        }
        () = shutdown_token.cancelled() => {
            anyhow::bail!("Shutdown during audio transcription");
        }
    };

    text
}

// ── Audio decoding ────────────────────────────────────────────────────

/// Decode any supported audio file to 16 kHz mono f32 samples.
///
/// Supports WAV (directly via qwen-asr's parser for maximum compatibility),
/// OGG/Opus (Telegram voice messages), MP3, and raw audio.
///
/// `pub(crate)`: the voice-tests E2E benchmark's real-audio
/// FAPH phase reuses this decoder for the pinned corpus (WAV/OGG/MP3 only —
/// the corpus format pin), so the bench binary needs no new decoder crates.
pub(crate) fn decode_audio_to_mono_f32(path: &Path) -> Result<Vec<f32>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Read the file into memory for decoding.
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read audio file: {}", path.display()))?;

    // Fast path for WAV files — use qwen-asr's native parser.
    if ext == "wav" {
        // Attempt once; success returns immediately, failure falls through
        // to the generic decoder match below (which will fail with a clear
        // error rather than calling parse_wav_buffer a second time).
        if let Some(samples) = qwen_asr::audio::parse_wav_buffer(&data) {
            return Ok(samples);
        }
    }
    // If the WAV parser fails, fall through to the generic decoder.

    if data.len() < 8 {
        anyhow::bail!("Audio file too small: {}", path.display());
    }

    let (samples, sample_rate) = match ext.as_str() {
        "ogg" | "oga" => decode_opus_from_ogg(&data, path)?,
        "mp3" => decode_mp3(&data, path)?,
        "wav" => {
            // Fast-path above already failed — don't retry the same call.
            anyhow::bail!(
                "Failed to decode WAV file (format error): {}",
                path.display()
            );
        }
        // For unknown formats, try OGG/Opus (common Telegram format).
        _ => {
            // Check for OGG magic bytes.
            if data.len() >= 4 && &data[0..4] == b"OggS" {
                decode_opus_from_ogg(&data, path)?
            } else {
                anyhow::bail!(
                    "Unsupported audio format '.{ext}' — only WAV, OGG/Opus, and MP3 are supported"
                );
            }
        }
    };

    // Resample to 16 kHz if needed.
    if sample_rate == qwen_asr::config::SAMPLE_RATE {
        Ok(samples)
    } else {
        let resampled =
            qwen_asr::audio::resample(&samples, sample_rate, qwen_asr::config::SAMPLE_RATE);
        Ok(resampled)
    }
}

/// Decode an OGG/Opus file into mono f32 samples at the file's sample rate.
///
/// Uses the `ogg` crate for OGG demuxing and the `opus_decoder` crate for
/// pure-Rust Opus decoding (no C dependencies, no unsafe code).
fn decode_opus_from_ogg(data: &[u8], path: &Path) -> Result<(Vec<f32>, i32)> {
    use ogg::reading::PacketReader;
    use std::io::Cursor;

    let cursor = Cursor::new(data);
    let mut reader = PacketReader::new(cursor);
    let sample_rate: i32 = 16000; // We decode at 16 kHz
    let mut decoder: Option<opus_decoder::OpusDecoder> = None;
    let mut channels: usize = 1; // Will be set from OpusHead
    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match reader.read_packet() {
            Ok(Some(pkt)) => pkt,
            Ok(None) => break, // End of stream
            Err(e) => {
                warn!(path = %path.display(), error = %e, "OGG demux error, stopping");
                break;
            }
        };

        let packet_data = packet.data;

        // The first packet is the Opus identification header
        if decoder.is_none() {
            if packet_data.starts_with(b"OpusHead") {
                // OpusHead: magic(8) + version(1) + channels(1) + pre-skip(2) + input_sample_rate(4) + ...
                if packet_data.len() < 18 {
                    anyhow::bail!("Invalid Opus identification header (too short)");
                }
                channels = packet_data[9] as usize;
                match opus_decoder::OpusDecoder::new(16000u32, channels) {
                    Ok(d) => {
                        decoder = Some(d);
                    }
                    Err(e) => {
                        anyhow::bail!("Failed to create Opus decoder: {e:?}");
                    }
                }
            } else {
                anyhow::bail!("OGG file does not contain Opus data");
            }
            continue;
        }

        // Skip the OpusTags comment header packet — it's metadata, not audio
        // data, and feeding it to decode_float produces a spurious decode error.
        if packet_data.starts_with(b"OpusTags") {
            continue;
        }

        if packet_data.is_empty() {
            continue; // Empty packet
        }

        // Decode to f32. Max packet: ~120ms at 48kHz stereo = 5760 samples/channel.
        // Use generous buffer with margin for multichannel.
        let max_pcm_len = 5760 * channels.max(2);
        let mut pcm = vec![0.0f32; max_pcm_len];
        let dec = decoder.as_mut().ok_or_else(|| {
            anyhow::anyhow!("Opus decoder not initialized — missing OpusHead header")
        })?;
        match dec.decode_float(&packet_data, &mut pcm, false) {
            Ok(n_per_channel) => {
                if channels == 1 {
                    samples.extend_from_slice(&pcm[..n_per_channel]);
                } else {
                    // Stereo → mono by averaging.
                    for i in 0..n_per_channel {
                        let l = pcm[i * 2];
                        let r = pcm[i * 2 + 1];
                        samples.push((l + r) * 0.5);
                    }
                }
            }
            Err(e) => {
                warn!(path = %path.display(), error = ?e, "Opus decode error, skipping packet");
            }
        }
    }

    if samples.is_empty() {
        anyhow::bail!("No audio decoded from {}", path.display());
    }

    Ok((samples, sample_rate))
}

/// Decode an MP3 file into mono f32 samples at the file's sample rate.
///
/// Uses the `minimp3` crate — a Rust wrapper around the minimp3 C library.
/// This introduces a C compiler build dependency (`minimp3-sys` + `cc` crate).
/// If a pure-Rust MP3 decoder is needed in the future, one could replace this
/// with a subprocess call to `ffmpeg` (handling all audio formats) or a
/// pure-Rust MP3 crate like `mp3-dl`.
#[allow(clippy::cast_precision_loss)]
fn decode_mp3(data: &[u8], path: &Path) -> Result<(Vec<f32>, i32)> {
    use minimp3::Decoder as Mp3Decoder;

    let mut decoder = Mp3Decoder::new(data);
    let mut samples: Vec<f32> = Vec::new();
    let mut sample_rate: i32 = 0;

    loop {
        match decoder.next_frame() {
            Ok(frame) => {
                // frame.data is Vec<i16> in [L,R,L,R,...] interleaved or [L,L,...] for mono.
                // frame.channels is the number of channels.
                // frame.sample_rate is the sample rate in Hz.
                if sample_rate == 0 {
                    sample_rate = frame.sample_rate;
                }

                let n_ch = frame.channels;
                if n_ch == 1 {
                    // Mono: convert i16 to f32 in [-1.0, 1.0]
                    for &val in &frame.data {
                        samples.push(f32::from(val) / 32768.0);
                    }
                } else {
                    // Stereo/multi: average to mono.
                    for chunk in frame.data.chunks(n_ch) {
                        let mono: f32 = chunk.iter().map(|&v| f32::from(v)).sum::<f32>()
                            / n_ch as f32
                            / 32768.0;
                        samples.push(mono);
                    }
                }
            }
            Err(minimp3::Error::Eof) => break,
            Err(e) => {
                warn!(path = %path.display(), error = %e, "MP3 decode error, stopping");
                break;
            }
        }
    }

    if samples.is_empty() {
        anyhow::bail!("No audio decoded from {}", path.display());
    }

    Ok((samples, sample_rate))
}

// ── File download helpers ─────────────────────────────────────────────

/// Structure describing a file to download.
struct ModelFile {
    filename: &'static str,
    url: String,
    expected_sha256: &'static str,
    timeout: Duration,
}

/// Build the URL for a model file in the HuggingFace repo.
fn model_url(filename: &str) -> String {
    format!("https://huggingface.co/{MODEL_REPO}/resolve/main/{filename}")
}

/// Build the list of all required model files.
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

/// Download a file from `url` to `dest`, verifying SHA256.
///
/// On SHA256 mismatch, the partially downloaded file is removed so the next
/// attempt re-downloads from scratch.
async fn download_file(client: &reqwest::Client, file: &ModelFile, dest: &Path) -> Result<()> {
    #[allow(clippy::cast_precision_loss)]
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
    reqwest::Client::builder()
        .user_agent("mahbot/0.3.0 (qwen-asr model downloader)")
        .build()
        .context("Failed to create HTTP client for model download")
}

/// Background download loop with retry.
///
/// Downloads all model files sequentially. On failure, sleeps with exponential
/// backoff and retries up to [`MAX_DOWNLOAD_RETRIES`] times. On success,
/// loads the model and transitions to [`ModelState::Ready`]. On terminal failure,
/// transitions to [`ModelState::Failed`].
async fn download_retry_loop() {
    let Some(dir) = model_dir() else {
        warn!("Local transcriber: cannot resolve model directory (storage root not set)");
        STATE.store(ModelState::Failed, Ordering::Release);
        return;
    };

    let client = match download_client() {
        Ok(c) => c,
        Err(e) => {
            warn!("Local transcriber: failed to create HTTP client: {e}");
            STATE.store(ModelState::Failed, Ordering::Release);
            return;
        }
    };

    tokio::fs::create_dir_all(&dir).await.ok();

    let files = model_files();
    let mut attempt: u32 = 0;

    loop {
        attempt += 1;
        let mut all_ok = true;

        for file in &files {
            let dest = dir.join(file.filename);
            if dest.exists() {
                // Verify existing file on a blocking thread (SHA256 reads
                // the full 1.88 GB model file).
                let dest_clone = dest.clone();
                let expected = file.expected_sha256.to_string();
                let checksum_ok = tokio::task::spawn_blocking(move || {
                    crate::util::verify_sha256(&dest_clone, &expected).is_ok()
                })
                .await
                .unwrap_or_else(|join_err| {
                    warn!("Local transcriber: SHA256 verification task panicked: {join_err}");
                    false
                }); // JoinError → log and treat as checksum mismatch, will re-download

                if checksum_ok {
                    continue;
                }
                warn!(
                    "Local transcriber: {} SHA256 mismatch, re-downloading",
                    file.filename
                );
                // Remove the corrupted file so the download starts fresh.
                tokio::fs::remove_file(&dest).await.ok();
            }

            info!(
                "Local transcriber: downloading {} — attempt {attempt}/{MAX_DOWNLOAD_RETRIES}",
                file.filename,
            );

            match download_file(&client, file, &dest).await {
                Ok(()) => {
                    // download_file already verifies SHA256 during streaming
                    // and removes the file on mismatch, so no re-verification needed.
                }
                Err(e) => {
                    warn!(
                        "Local transcriber: failed to download {}: {e}",
                        file.filename
                    );
                    tokio::fs::remove_file(&dest).await.ok();
                    all_ok = false;
                    break;
                }
            }
        }

        if all_ok {
            // All files downloaded and verified. Load the model on a blocking
            // thread (reads model weights from disk).
            info!("Local transcriber: all model files downloaded, loading...");
            let dir_for_load = dir.clone();
            let loaded = tokio::task::spawn_blocking(move || {
                QwenLocalTranscriber::try_load_from(&dir_for_load)
            })
            .await
            .ok()
            .flatten();
            if let Some(tc) = loaded {
                info!("Local transcriber: Qwen3-ASR model loaded successfully");
                set_transcriber_ready(tc);
                return;
            }
            warn!(
                "Local transcriber: model files present but failed to load — deleting and re-downloading"
            );
            // Delete corrupted files to force a fresh download with
            // SHA256 verification, following the embedder pattern
            // (embedder.rs Phase 1 logic).
            for f in &files {
                let dest = dir.join(f.filename);
                tokio::fs::remove_file(&dest).await.ok();
            }
        }

        if attempt >= MAX_DOWNLOAD_RETRIES {
            warn!("Local transcriber: max retries ({MAX_DOWNLOAD_RETRIES}) reached, giving up");
            STATE.store(ModelState::Failed, Ordering::Release);
            return;
        }

        let sleep_secs = DOWNLOAD_RETRY_BASE_SECS * (1u64 << (attempt - 1).min(8));
        let sleep_dur = Duration::from_secs(sleep_secs.min(300));
        warn!(
            "Local transcriber: retrying in {}s (attempt {attempt}/{MAX_DOWNLOAD_RETRIES})",
            sleep_dur.as_secs()
        );
        tokio::time::sleep(sleep_dur).await;
    }
}

// ── Public API ────────────────────────────────────────────────────────

/// Shared initialization logic — checks files, verifies checksums, loads the
/// model from `dir`, and stores it into [`GLOBAL_TRANSCRIBER`].  The caller
/// ([`try_init_from_cache`]) is responsible for the STATE atomic guard.
async fn try_init_inner(dir: PathBuf) -> bool {
    let model_path = dir.join(MODEL_FILENAME);
    let vocab_path = dir.join(VOCAB_FILENAME);
    let merges_path = dir.join(MERGES_FILENAME);

    if model_path.exists() && vocab_path.exists() && merges_path.exists() {
        // Verify checksums on a blocking thread (large file I/O).
        let paths = (model_path.clone(), vocab_path.clone(), merges_path.clone());
        let checksums_ok = tokio::task::spawn_blocking(move || {
            crate::util::verify_sha256(&paths.0, MODEL_SHA256).is_ok()
                && crate::util::verify_sha256(&paths.1, VOCAB_SHA256).is_ok()
                && crate::util::verify_sha256(&paths.2, MERGES_SHA256).is_ok()
        })
        .await
        .unwrap_or_else(|join_err| {
            warn!("Local transcriber: SHA256 verification task panicked: {join_err}");
            false
        });

        if checksums_ok {
            // Load the model using the already-resolved directory
            // (moved into the blocking closure — no longer needed after this).
            let loaded =
                tokio::task::spawn_blocking(move || QwenLocalTranscriber::try_load_from(&dir))
                    .await
                    .ok()
                    .flatten();
            if let Some(tc) = loaded {
                info!("Local transcriber: loaded from cache");
                set_transcriber_ready(tc);
                return true;
            }
            warn!("Local transcriber: cached files present but failed to load");
        } else {
            warn!("Local transcriber: cached files failed checksum, will re-download");
        }
    }

    // Spawn background download.
    if tokio::runtime::Handle::try_current().is_err() {
        warn!("Local transcriber: no tokio runtime available");
        STATE.store(ModelState::Failed, Ordering::Release);
        return false;
    }

    info!("Local transcriber: model not cached, spawning background download");
    tokio::spawn(download_retry_loop());
    false
}

/// Try to shortcut or acquire the initialisation lock.
///
/// Returns `None` if the caller acquired the lock and should proceed.
/// Returns `Some(true)` if already initialised (ModelState::Ready) — caller
/// should return success.
/// Returns `Some(false)` if another thread is loading or a previous
/// attempt failed — caller should return failure.
fn try_lock_init() -> Option<bool> {
    match STATE.load(Ordering::Acquire) {
        ModelState::Ready => return Some(true),
        ModelState::Uninit => {}
        _ => return Some(false),
    }

    if STATE
        .compare_exchange(
            ModelState::Uninit,
            ModelState::Loading,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        return Some(false);
    }

    None
}

/// Initialise the local transcriber from the default cache directory
/// (resolved via [`model_dir`], which depends on the CONFIG storage root).
///
/// This is the main entry point used by the production pipeline
/// (`bootstrap` and audio enrichment).
pub async fn try_init_from_cache() -> bool {
    if let Some(result) = try_lock_init() {
        return result;
    }

    let Some(dir) = model_dir() else {
        STATE.store(ModelState::Failed, Ordering::Release);
        return false;
    };

    try_init_inner(dir).await
}

/// True if the local transcriber is loaded and ready for use.
pub fn is_loaded() -> bool {
    STATE.is_ready()
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── hex_string ─────────────────────────────────────────────────────

    #[test]
    fn test_hex_string_empty() {
        assert_eq!(crate::util::hex_string(b""), "");
    }

    #[test]
    fn test_hex_string_all_bytes() {
        let bytes: Vec<u8> = (0..=255u8).collect();
        let hex = crate::util::hex_string(&bytes);
        assert_eq!(hex.len(), 512);
        assert!(hex.starts_with("00010203"));
        assert!(hex.ends_with("fcfdfeff"));
    }

    // ── decode_audio_to_mono_f32 — synthetic WAV ────────────────────────

    /// Create a minimal valid 16-bit mono PCM WAV file in a temp directory.
    fn write_synthetic_wav(
        dir: &std::path::Path,
        filename: &str,
        sample_rate: u32,
    ) -> std::path::PathBuf {
        let path = dir.join(filename);

        // Generate a 1-second sine wave at 440 Hz.
        let num_samples = sample_rate as usize;
        let sample_rate_i32 = sample_rate as i32;
        let mut pcm_data = Vec::with_capacity(num_samples * 2);
        for i in 0..num_samples {
            let t = i as f64 / sample_rate_i32 as f64;
            let sample = (t * 440.0 * 2.0 * std::f64::consts::PI).sin();
            let val = (sample * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
            pcm_data.extend_from_slice(&val.to_le_bytes());
        }

        let data_size = pcm_data.len() as u32;
        let file_size = 36 + data_size;

        let mut wav = Vec::new();
        // RIFF header
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        // fmt chunk (16 bytes)
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&(16u32).to_le_bytes()); // chunk size
        wav.extend_from_slice(&(1u16).to_le_bytes()); // PCM format
        wav.extend_from_slice(&(1u16).to_le_bytes()); // mono
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        wav.extend_from_slice(&(2u16).to_le_bytes()); // block align
        wav.extend_from_slice(&(16u16).to_le_bytes()); // bits per sample
        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        wav.extend_from_slice(&pcm_data);

        std::fs::write(&path, &wav).unwrap();
        path
    }

    #[test]
    fn test_decode_wav_mono_16k() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_synthetic_wav(dir.path(), "test.wav", 16000);
        let result = decode_audio_to_mono_f32(&path);
        assert!(
            result.is_ok(),
            "Failed to decode 16 kHz WAV: {:?}",
            result.err()
        );
        let samples = result.unwrap();
        assert!(!samples.is_empty(), "Decoded samples should not be empty");
        // At 16 kHz, 1 second = 16000 samples
        assert_eq!(
            samples.len(),
            16000,
            "Expected 16000 samples for 1 second at 16 kHz"
        );
        // Check that samples are in valid f32 range
        for &s in &samples {
            assert!(s >= -1.0 && s <= 1.0, "Sample {s} out of range");
        }
        // Verify first few samples approximate a sine wave starting near 0
        assert!(
            (samples[0]).abs() < 0.01,
            "First sample should be near 0 for sine starting at t=0"
        );
    }

    #[test]
    fn test_decode_wav_mono_48k_resampled() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_synthetic_wav(dir.path(), "test48k.wav", 48000);
        let result = decode_audio_to_mono_f32(&path);
        assert!(
            result.is_ok(),
            "Failed to decode 48 kHz WAV: {:?}",
            result.err()
        );
        let samples = result.unwrap();
        // Resampled to 16 kHz — 1 second at 48 kHz → ~16000 samples after resampling
        assert!(!samples.is_empty(), "Decoded samples should not be empty");
        // Allow some tolerance for resampling (should be close to 16000)
        assert!(
            samples.len() >= 15500 && samples.len() <= 16500,
            "Expected ~16000 samples after resampling from 48 kHz, got {}",
            samples.len()
        );
    }

    #[test]
    fn test_decode_wav_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.wav");
        std::fs::write(&path, b"").unwrap();
        let result = decode_audio_to_mono_f32(&path);
        assert!(result.is_err(), "Empty file should fail to decode");
    }

    #[test]
    fn test_decode_wav_too_small() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.wav");
        std::fs::write(&path, b"RIFF").unwrap();
        let result = decode_audio_to_mono_f32(&path);
        assert!(result.is_err(), "Truncated WAV should fail to decode");
    }

    // ── decode_audio_to_mono_f32 — extension handling ──────────────────

    #[test]
    fn test_decode_unknown_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audio.xyz");
        std::fs::write(&path, b"not an audio file").unwrap();
        let result = decode_audio_to_mono_f32(&path);
        assert!(result.is_err(), "Unknown extension should fail to decode");
    }

    #[test]
    fn test_decode_no_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("noext");
        std::fs::write(&path, b"not an audio file").unwrap();
        let result = decode_audio_to_mono_f32(&path);
        assert!(
            result.is_err(),
            "File without extension should fail to decode"
        );
    }

    // ── OGG/Opus decode ────────────────────────────────────────────────

    /// Construct a minimal valid OGG/Opus byte stream in memory.
    fn create_test_opus_ogg() -> Vec<u8> {
        use ogg::writing::{PacketWriteEndInfo, PacketWriter};

        let mut buf = Vec::new();
        let serial = 1u32;

        // Scoped so the PacketWriter flushes before we read the buf.
        {
            let mut writer = PacketWriter::new(&mut buf);

            // Opus identification header (OpusHead)
            let mut head = Vec::new();
            head.extend_from_slice(b"OpusHead"); // magic
            head.push(1); // version
            head.push(1); // channels (mono)
            head.extend_from_slice(&0u16.to_le_bytes()); // pre-skip
            head.extend_from_slice(&48000u32.to_le_bytes()); // input sample rate
            head.extend_from_slice(&0u16.to_le_bytes()); // output gain
            head.push(0); // channel mapping family
            writer
                .write_packet(head, serial, PacketWriteEndInfo::EndPage, 0)
                .unwrap();

            // Opus comment header (OpusTags) — minimal
            let vendor = b"test";
            let mut tags = Vec::new();
            tags.extend_from_slice(b"OpusTags");
            tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
            tags.extend_from_slice(vendor);
            tags.extend_from_slice(&0u32.to_le_bytes()); // user comment list length = 0
            writer
                .write_packet(tags, serial, PacketWriteEndInfo::EndPage, 0)
                .unwrap();

            // Audio data: a zero-byte packet triggers packet-loss concealment,
            // which produces silence (no crash).
            writer
                .write_packet(b"", serial, PacketWriteEndInfo::EndStream, 0)
                .unwrap();
        }
        buf
    }

    #[test]
    fn test_decode_opus_ogg_valid_headers() {
        let data = create_test_opus_ogg();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("voice.ogg");
        std::fs::write(&path, &data).unwrap();
        // The empty audio packet triggers PLC which produces 0 samples (no
        // prior decoder state). The function should not panic and should return
        // a meaningful error about no audio being decoded.
        let result = decode_audio_to_mono_f32(&path);
        assert!(
            result.is_err(),
            "OGG/Opus with headers + silence should produce 'No audio decoded'"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("No audio decoded") || err.contains("decode"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn test_decode_opus_ogg_truncated() {
        let data = create_test_opus_ogg();
        let truncated = &data[..data.len().min(64)];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.ogg");
        std::fs::write(&path, truncated).unwrap();
        let result = decode_audio_to_mono_f32(&path);
        // Truncated OGG should produce an error from the OGG demuxer
        // (missing pages or incomplete Opus headers).
        assert!(result.is_err(), "Truncated OGG/Opus should fail to decode");
    }

    #[test]
    fn test_decode_opus_ogg_invalid_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fake.ogg");
        std::fs::write(&path, b"not an OGG file at all").unwrap();
        let result = decode_audio_to_mono_f32(&path);
        assert!(result.is_err(), "Invalid OGG data should fail to decode");
    }

    #[test]
    fn test_decode_opus_ogg_no_opus_head() {
        // Create an OGG page with no OpusHead (just invalid data in an OGG page).
        let data = b"OggS\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00invalid";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nohead.ogg");
        std::fs::write(&path, data).unwrap();
        let result = decode_audio_to_mono_f32(&path);
        assert!(
            result.is_err(),
            "OGG without OpusHead should fail to decode"
        );
    }

    // ── MP3 decode ──────────────────────────────────────────────────────

    /// Create a minimal valid MPEG2.5 Layer III frame.
    ///
    /// Frame parameters: 8 kbps, 8 kHz, mono, no CRC.
    /// Frame size = (144 * 8) / 8 = 144 bytes.
    fn create_test_mp3_frame() -> Vec<u8> {
        // Header: sync|version|layer|prot  bitrate|srate|pad|priv  mode|ext|copy|orig|emp
        //         0xFF    0xE3            0x18                   0xC0
        let mut frame = vec![0xFF, 0xE3, 0x18, 0xC0];
        frame.resize(144, 0u8);
        frame
    }

    #[test]
    fn test_decode_mp3_valid_frame() {
        let data = create_test_mp3_frame();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.mp3");
        std::fs::write(&path, &data).unwrap();
        let result = decode_audio_to_mono_f32(&path);
        // The minimp3 decoder may produce output (even from zeroed data)
        // or it may return SkippedData that our decode_mp3 treats as EOF.
        // Either way the function should not panic.
        if let Err(e) = &result {
            assert!(
                e.to_string().contains("No audio decoded") || e.to_string().contains("Unsupported"),
                "Unexpected error: {e}"
            );
        }
    }

    #[test]
    fn test_decode_mp3_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.mp3");
        std::fs::write(&path, b"").unwrap();
        let result = decode_audio_to_mono_f32(&path);
        assert!(result.is_err(), "Empty MP3 should fail to decode");
    }

    #[test]
    fn test_decode_mp3_truncated_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.mp3");
        std::fs::write(&path, b"\xFF\xFB").unwrap();
        let result = decode_audio_to_mono_f32(&path);
        assert!(
            result.is_err(),
            "Truncated MP3 header should fail to decode"
        );
    }

    #[test]
    fn test_decode_mp3_non_mp3_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fake.mp3");
        std::fs::write(&path, b"this is not an mp3 file at all").unwrap();
        let result = decode_audio_to_mono_f32(&path);
        assert!(result.is_err(), "Non-MP3 data should fail to decode");
    }
}
