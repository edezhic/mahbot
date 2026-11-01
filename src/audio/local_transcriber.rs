//! Local audio transcription using Qwen3-ASR via the `qwen-asr` crate.
//!
//! This module is the **only** audio transcription path — the API-based transcriber is gone.
//! Inference runs fully locally with Qwen3-ASR-0.6B; the model (~1.88 GB BF16 safetensors) is
//! downloaded on first use to `~/.mahbot/models/qwen3-asr-0.6b/` and memory-mapped by `qwen-asr`
//! (zero-copy weights, minimal RSS).
//!
//! Boot never preloads the recogniser: the first operation that needs it loads it — the first voice
//! message, the voice stack's own demand ([`crate::audio::voice`]), or the Settings toggle.
//! [`ensure_loaded`] is that awaited "make sure it is loaded": with the weights on disk it loads
//! them (~4 s) or waits out the cache load a voice press already started; otherwise it kicks the
//! background download retry loop and reports not-ready.
//!
//! Files are SHA256-verified **once, at download time** — there is no per-use re-verification of
//! the ~1.9 GB model (that was ~4.4 s of the old awaited boot path). A silently corrupted cache is
//! caught at load time instead: `QwenCtx::load` returns `None` and init falls through to the
//! download+verify recovery loop. The load is mmap-based, so partial corruption can survive it and
//! surface as a SIGBUS at inference rather than a graceful fallback — accepted tradeoff.
//! Missing/corrupt files trigger the retry loop with exponential backoff.
//! qwen-asr expects raw f32 PCM at 16 kHz mono, so OGG/Opus (`ogg` + `opus-decoder`, pure Rust),
//! MP3 (`minimp3`) and WAV (`qwen_asr::audio::parse_wav_buffer`, which resamples) are decoded
//! first.
//!
//! [`transcribe_file_async`] has three callers: the voice pipeline ([`crate::audio::voice`], a
//! whole recording), the enrichment path
//! ([`crate::channels::enrichment::transcribe_audio_marker`], an attached file) and
//! `mahbot audio transcribe`. With the local model unavailable or disabled, the enrichment caller
//! annotates the message with just the icon combo — there is no API fallback.

use crate::audio::models_subdir;
use crate::util::UnwrapPoison;
use crate::util::model_state::{AtomicModelState, ModelLoadGuard, ModelState};
use anyhow::{Context, Result};
use futures_util::FutureExt;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
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
/// Default inference timeout (10 minutes): short clips finish in seconds, long recordings in
/// minutes, and a hung inference must not occupy a tokio blocking thread forever. Shared by the
/// voice pipeline, the audio-enrichment path and `mahbot audio transcribe`, which answers this
/// bound in its own words.
pub(crate) const INFERENCE_TIMEOUT: Duration = Duration::from_mins(10);

/// The recogniser's own bound on one inference ([`INFERENCE_TIMEOUT`]) was reached: the recording
/// is longer than the engine will take.
///
/// A type rather than a message, so `mahbot audio` can answer this failure in its own words — a
/// recording longer than the engine takes, and never a transcript with a part of it missing —
/// without matching on the engine's wording.
#[derive(Debug)]
pub(crate) struct InferenceTimedOut;

impl std::fmt::Display for InferenceTimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Qwen3-ASR inference timed out")
    }
}

impl std::error::Error for InferenceTimedOut {}

/// Download timeout for the 1.88 GB model file (30 minutes); the smaller files finish far sooner
/// under this timeout because the stream is shared.
const MODEL_DOWNLOAD_TIMEOUT: Duration = Duration::from_mins(30);

/// Download timeout for config/vocabulary files (< 10 MB).
const SMALL_FILE_TIMEOUT: Duration = Duration::from_mins(1);

/// Retry sleep base: 5 seconds, doubled per attempt.
const DOWNLOAD_RETRY_BASE_SECS: u64 = 5;

/// Maximum number of download retry attempts.
const MAX_DOWNLOAD_RETRIES: u32 = 12;

/// How often a caller that needs the recogniser checks whether an in-flight load has settled
/// ([`ensure_loaded`]).
const LOAD_WAIT_POLL: Duration = Duration::from_millis(100);

/// The longest a caller waits for a load already in flight. Only a cache load is waited on
/// ([`ensure_loaded`]), and a cache read of the 1.88 GB of weights takes seconds even from a cold
/// disk; the bound is far below a download's own ([`MODEL_DOWNLOAD_TIMEOUT`]).
const LOAD_WAIT_LIMIT: Duration = Duration::from_mins(2);

/// Global singleton handle — `None` until loaded.
static GLOBAL_TRANSCRIBER: Mutex<Option<QwenLocalTranscriber>> = Mutex::new(None);

/// Shared-model slot for the wake-word pipeline, populated by [`set_transcriber_ready`] alongside
/// [`GLOBAL_TRANSCRIBER`]. The wake-word path reads this slot instead of locking the transcription
/// context mutex — [`transcribe_file_async`] holds the inner `QwenCtx` mutex for the FULL duration
/// of an inference (up to [`INFERENCE_TIMEOUT`]), so a per-stride lock would stall the whole voice
/// pipeline for the length of any concurrent transcription (e.g. Telegram voice enrichment).
/// Cloning an `Arc` under this short mutex is O(1) and uncontended (single writer on swap).
static SHARED_MODEL: Mutex<Option<Arc<qwen_asr::context::QwenModel>>> = Mutex::new(None);

/// Atomic state tracker to coordinate lazy initialization: Uninit → Loading → Ready / Failed.
/// Failed is terminal for transcription (the error path leaves the caller with an icon-only audio
/// annotation); the voice pipeline's bounded auto-retry and the explicit GUI retry
/// ([`VoiceCommand::RetryModelLoading`]) are the only recovery knobs.
static STATE: AtomicModelState = AtomicModelState::new(ModelState::Uninit);

/// True while the loader in [`STATE`] is downloading rather than loading a cache — set for the
/// whole run of [`download_retry_loop`], which is entered exactly when there is no cache to load.
///
/// A caller waiting for `Loading` stops waiting the moment this is set ([`wait_until_settled`]):
/// a cache read takes seconds, a download takes minutes, and whoever just spoke gets an answer
/// either way.
static DOWNLOADING: AtomicBool = AtomicBool::new(false);

/// Holds [`DOWNLOADING`] for the duration of a download, however it ends — panic or cancellation
/// included.
struct DownloadPhase;

impl DownloadPhase {
    fn enter() -> Self {
        DOWNLOADING.store(true, Ordering::Release);
        Self
    }
}

impl Drop for DownloadPhase {
    fn drop(&mut self) {
        DOWNLOADING.store(false, Ordering::Release);
    }
}

/// Atomically store a ready transcriber and transition state to [`ModelState::Ready`].
fn set_transcriber_ready(tc: QwenLocalTranscriber) {
    // Populate the shared-model slot BEFORE publishing the transcriber so the wake-word path can
    // never observe Ready without a model. The ctx mutex is uncontended here: the transcriber has
    // just been loaded and not yet published, so no inference can be running on it.
    *SHARED_MODEL.lock().unwrap_poison() = Some(tc.model_arc());
    *GLOBAL_TRANSCRIBER.lock().unwrap_poison() = Some(tc);
    STATE.store(ModelState::Ready, Ordering::Release);
}

/// The Qwen3-ASR local transcriber: a `qwen-asr` inference context behind a `transcribe_file`
/// method that handles audio decoding, resampling and inference.
///
/// The inner `QwenCtx` is not `Sync` (it holds a `Box<dyn Fn + Send>` token callback), so it is
/// wrapped in a `Mutex`; the `Arc` lets the global singleton's outer lock be released right after
/// cloning, so background download completion ([`set_transcriber_ready`]) never contends.
pub struct QwenLocalTranscriber {
    ctx: Arc<Mutex<qwen_asr::context::QwenCtx>>,
}

impl QwenLocalTranscriber {
    /// Load the model from an explicit directory path. Unlike [`ensure_loaded`] this does not
    /// resolve the directory via [`crate::audio::models_subdir`], so it does not need the CONFIG
    /// storage root to be set.
    fn try_load_from(dir: &Path) -> Option<Self> {
        let dir_str = dir.to_string_lossy().to_string();
        let mut ctx = qwen_asr::context::QwenCtx::load(&dir_str)?;
        ctx.want_language_detection = true;
        // Split audio longer than ~30 s at low-energy boundaries: without segmentation a single
        // segment caps at 2048 decoded tokens, silently truncating long recordings.
        ctx.segment_sec = 30.0;
        Some(Self {
            ctx: Arc::new(Mutex::new(ctx)),
        })
    }

    /// Clone the inner `Arc` so callers can release the outer [`GLOBAL_TRANSCRIBER`] lock before
    /// running inference.
    fn clone_arc(&self) -> Arc<Mutex<qwen_asr::context::QwenCtx>> {
        Arc::clone(&self.ctx)
    }

    /// Clone the shared immutable [`qwen_asr::context::QwenModel`] Arc held by this context.
    ///
    /// Called once by [`set_transcriber_ready`] to populate [`SHARED_MODEL`] — NOT per scoring
    /// stride (that would lock the context mutex, held for a whole inference). The encoder weights
    /// are read-only after load, so concurrent `Encoder::forward` calls with per-call scratch
    /// buffers are safe; the wake-word path passes `None` for `enc_bufs`.
    fn model_arc(&self) -> Arc<qwen_asr::context::QwenModel> {
        Arc::clone(&self.ctx.lock().unwrap_poison().model)
    }
}

/// Asynchronously transcribe an audio file on a blocking thread: decode, then Qwen3-ASR
/// inference via [`tokio::task::spawn_blocking`], so CPU-heavy work never stalls the runtime.
///
/// The outer [`GLOBAL_TRANSCRIBER`] lock is held only long enough to clone the inner
/// [`Arc<Mutex<QwenCtx>>`], then released before inference — so a background download completing
/// mid-transcription can swap the new transcriber in without waiting.
///
/// `inference_timeout` bounds the inference step only (decoding is excluded). Every caller — the
/// voice pipeline (up to 10 minutes), the enrichment path and `mahbot audio transcribe` — passes
/// [`INFERENCE_TIMEOUT`]. A shutdown guard races the inference so a stalled model cannot block
/// pipeline exit.
///
/// Errors when the local model cannot be loaded: an in-flight cache load is waited out, while a
/// download and a terminal load failure are reported distinctly, never blocked on.
pub async fn transcribe_file_async(path: &Path, inference_timeout: Duration) -> Result<String> {
    let owned = path.to_owned();

    let samples = tokio::task::spawn_blocking(move || decode_audio_to_mono_f32(&owned))
        .await
        .context("Audio decode task panicked")?
        .context("Failed to decode audio file to 16 kHz mono PCM")?;

    if samples.is_empty() {
        anyhow::bail!("Audio file is empty after decoding");
    }

    // Nothing preloads the recogniser: the first operation that needs it loads it here (the
    // ordinary case is the cached weights on disk) or waits out the load a voice press started.
    anyhow::ensure!(ensure_loaded().await, "{}", unloaded_message());

    // Clone the inference handle under the outer lock, then release it before inference.
    // std::sync::Mutex is fine: hold time is nanoseconds and contention is rare.
    let ctx_arc = {
        let guard = GLOBAL_TRANSCRIBER.lock().unwrap_poison();
        let tc = guard.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Local transcriber not available during async transcription")
        })?;
        tc.clone_arc()
    };

    // Inference with a timeout — the outer lock was released after the clone. The timeout keeps a
    // hung inference from occupying a blocking thread; the shutdown guard keeps it from blocking
    // pipeline exit.
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
            result.map_err(|_| anyhow::Error::new(InferenceTimedOut))?
        }
        () = shutdown_token.cancelled() => {
            anyhow::bail!("Shutdown during audio transcription");
        }
    };

    text
}

/// The audio names the recogniser takes — WAV, OGG/Opus and MP3 — and how each is decoded.
///
/// This is the ONE table of that rule: [`decode_audio_to_mono_f32`] dispatches through it,
/// [`takes_audio`] accepts a file by it, and a caller's refusal lists it, so a format can never be
/// taken by one and refused by another. A file under a name outside the table is decoded when its
/// bytes begin with OGG's magic — [`takes_audio`] is the rule a caller asks.
pub(crate) const AUDIO_FORMATS: &[(&str, AudioFormat)] = &[
    ("wav", AudioFormat::Wav),
    ("ogg", AudioFormat::Opus),
    ("oga", AudioFormat::Opus),
    ("mp3", AudioFormat::Mp3),
];

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AudioFormat {
    Wav,
    Opus,
    Mp3,
}

/// Decode any supported audio file to 16 kHz mono f32 samples: WAV through qwen-asr's own parser,
/// OGG/Opus (Telegram voice messages) and MP3 through their decoders.
pub(crate) fn decode_audio_to_mono_f32(path: &Path) -> Result<Vec<f32>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    // What the file's name says it is: one of the names in [`AUDIO_FORMATS`], or
    // none of them.
    let format = AUDIO_FORMATS
        .iter()
        .find(|(name, _)| *name == ext)
        .map(|(_, format)| *format);

    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read audio file: {}", path.display()))?;

    if format == Some(AudioFormat::Wav) {
        // Attempt the WAV parser once; on failure fall through to the generic decoder (which will
        // report a clear error rather than calling parse_wav_buffer a second time).
        if let Some(samples) = qwen_asr::audio::parse_wav_buffer(&data) {
            return Ok(samples);
        }
    }

    if data.len() < 8 {
        anyhow::bail!("Audio file too small: {}", path.display());
    }

    let (samples, sample_rate) = match format {
        Some(AudioFormat::Opus) => decode_opus_from_ogg(&data, path)?,
        Some(AudioFormat::Mp3) => decode_mp3(&data, path)?,
        Some(AudioFormat::Wav) => {
            anyhow::bail!(
                "Failed to decode WAV file (format error): {}",
                path.display()
            );
        }
        // For a name no format is named with, try OGG/Opus: its magic bytes are the only thing that
        // can make such a file audio.
        None => {
            if data.len() >= 4 && &data[0..4] == b"OggS" {
                decode_opus_from_ogg(&data, path)?
            } else {
                anyhow::bail!(
                    "Unsupported audio format '.{ext}' — only WAV, OGG/Opus, and MP3 are supported"
                );
            }
        }
    };

    if sample_rate == qwen_asr::config::SAMPLE_RATE {
        Ok(samples)
    } else {
        let resampled =
            qwen_asr::audio::resample(&samples, sample_rate, qwen_asr::config::SAMPLE_RATE);
        Ok(resampled)
    }
}

/// Whether the recogniser takes this file — its own rule, and only it: one of the names
/// [`AUDIO_FORMATS`] holds, or, under a name the decoder does not know, content beginning with
/// Ogg's magic. A known name is settled by the NAME.
pub(crate) fn takes_audio(path: &Path) -> std::io::Result<bool> {
    let mut file = std::fs::File::open(path)?;
    let named = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            AUDIO_FORMATS
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case(extension))
        });
    if named {
        return Ok(true);
    }
    let mut magic = [0u8; 4];
    Ok(std::io::Read::read_exact(&mut file, &mut magic).is_ok_and(|()| magic == *b"OggS"))
}

/// Decode an OGG/Opus file into mono f32 samples at the file's sample rate: `ogg` demuxing and
/// the pure-Rust `opus_decoder` (no C dependencies, no unsafe code).
fn decode_opus_from_ogg(data: &[u8], path: &Path) -> Result<(Vec<f32>, i32)> {
    use ogg::reading::PacketReader;
    use std::io::Cursor;

    let cursor = Cursor::new(data);
    let mut reader = PacketReader::new(cursor);
    let sample_rate: i32 = qwen_asr::config::SAMPLE_RATE; // We decode at 16 kHz
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

        if decoder.is_none() {
            if packet_data.starts_with(b"OpusHead") {
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

        // Skip the OpusTags comment header — metadata, not audio: feeding it to decode_float
        // produces a spurious decode error.
        if packet_data.starts_with(b"OpusTags") {
            continue;
        }

        if packet_data.is_empty() {
            continue; // Empty packet
        }

        // Decode to f32. Max packet: ~120 ms at 48 kHz stereo = 5760 samples/channel; the buffer
        // has margin for multichannel.
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
                    // Stereo → mono by averaging the midpoint: no overflow, and it rounds
                    // identically to
                    // (l + r) * 0.5 for normal samples.
                    for i in 0..n_per_channel {
                        let l = pcm[i * 2];
                        let r = pcm[i * 2 + 1];
                        samples.push(l.midpoint(r));
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

/// Decode an MP3 file into mono f32 samples at the file's sample rate through `minimp3` — a Rust
/// wrapper over the minimp3 C library, so this is a C compiler build dependency.
#[expect(clippy::cast_precision_loss)]
fn decode_mp3(data: &[u8], path: &Path) -> Result<(Vec<f32>, i32)> {
    use minimp3::Decoder as Mp3Decoder;

    let mut decoder = Mp3Decoder::new(data);
    let mut samples: Vec<f32> = Vec::new();
    let mut sample_rate: i32 = 0;

    loop {
        match decoder.next_frame() {
            Ok(frame) => {
                if sample_rate == 0 {
                    sample_rate = frame.sample_rate;
                }

                let n_ch = frame.channels;
                if n_ch == 1 {
                    for &val in &frame.data {
                        samples.push(f32::from(val) / 32768.0);
                    }
                } else {
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
                crate::audio::util::verify_sha256(&dest_clone, &expected).is_ok()
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
            tokio::fs::remove_file(&dest).await.ok();
        }

        info!("Local transcriber: downloading {}", file.filename);
        fetch_starting();

        // download_file already verified SHA256 while streaming and removed the file on mismatch,
        // so no re-verification is needed on success.
        if let Err(e) = download_file(client, file, &dest).await {
            warn!(
                "Local transcriber: failed to download {}: {e}",
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
    // A caller waiting on this load stops waiting from here on: its remaining work is a download,
    // not a cache read. Declared first so the terminal state lands before the flag clears.
    let _download = DownloadPhase::enter();
    // Loading → Failed on drop (panic/cancel safety). No-op on the terminal states the
    // success/retry-cap paths set.
    let _guard = ModelLoadGuard::new(&STATE);
    let Some(dir) = models_subdir(MODEL_DIR_NAME) else {
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
        info!("Local transcriber: model download attempt {attempt}/{MAX_DOWNLOAD_RETRIES}");

        if download_once(&client, &dir, &files, &mut || {})
            .await
            .is_ok()
        {
            // All files downloaded and verified. Load the model on a blocking thread.
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
            // Delete the corrupt files so the next attempt re-downloads and re-verifies them.
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

/// Load the transcriber from an already-resolved cache directory on a blocking thread (memory-map
/// + weight setup). `None` when any file is missing or the load fails.
async fn load_from_cache(dir: PathBuf) -> Option<QwenLocalTranscriber> {
    tokio::task::spawn_blocking(move || QwenLocalTranscriber::try_load_from(&dir))
        .await
        .ok()
        .flatten()
}

/// Prepare the recogniser for one operation: the model set, loaded from the cache when it is
/// complete and fetched EXACTLY once when it is not.
///
/// No background task and no retry loop: the daemon retries for itself, and a caller that waits
/// here waits only for a cache load — a download is reported, never waited out.
pub(crate) async fn ensure_ready_once(announce_fetch: impl FnOnce(&Path)) -> Result<()> {
    let dir = models_subdir(MODEL_DIR_NAME)
        .context("cannot resolve the model directory (the storage root is not set)")?;

    if cache_present(&dir)
        && let Some(tc) = load_from_cache(dir.clone()).await
    {
        set_transcriber_ready(tc);
        return Ok(());
    }

    // The notice is bound to the first file that is actually fetched: a set already on disk
    // fetches nothing and says nothing.
    let mut announce_fetch = Some(announce_fetch);
    let mut fetch_starting = || {
        if let Some(say) = announce_fetch.take() {
            say(&dir);
        }
    };
    tokio::fs::create_dir_all(&dir)
        .await
        .context("Failed to create the model directory")?;
    let client = download_client()?;
    download_once(&client, &dir, &model_files(), &mut fetch_starting)
        .await
        .context("Failed to fetch the recognition model")?;
    let tc = load_from_cache(dir)
        .await
        .context("the recognition model could not be loaded")?;
    set_transcriber_ready(tc);
    Ok(())
}

fn cache_present(dir: &Path) -> bool {
    dir.join(MODEL_FILENAME).exists()
        && dir.join(VOCAB_FILENAME).exists()
        && dir.join(MERGES_FILENAME).exists()
}

/// Shared cache-check + load logic used by both the awaited entry ([`ensure_loaded`]) and the
/// background load ([`init_background`]). The caller owns the STATE guard.
async fn try_init_inner(dir: PathBuf) -> bool {
    if cache_present(&dir) {
        if let Some(tc) = load_from_cache(dir).await {
            info!("Local transcriber: loaded from cache");
            set_transcriber_ready(tc);
            return true;
        }
        warn!("Local transcriber: cached files present but failed to load");
    }

    if tokio::runtime::Handle::try_current().is_err() {
        warn!("Local transcriber: no tokio runtime available");
        STATE.store(ModelState::Failed, Ordering::Release);
        return false;
    }

    info!("Local transcriber: model not cached, spawning background download");
    tokio::spawn(download_retry_loop());
    false
}

/// Background load-or-download chain — spawned by [`spawn_background_init`], never awaited.
///
/// STATE must already be `Loading` (the caller owns the `Uninit → Loading` transition); this
/// function owns every later one.
async fn init_background(dir: PathBuf) {
    if cache_present(&dir) {
        if let Some(tc) = load_from_cache(dir).await {
            info!("Local Qwen3-ASR transcriber loaded from cache");
            set_transcriber_ready(tc);
            return;
        }
        warn!("Local transcriber: cached files present but failed to load — re-downloading");
    }

    info!("Local transcriber: model not cached, downloading in background");
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

/// Initialise the local transcriber from the default cache directory
/// ([`crate::audio::models_subdir`], which depends on the CONFIG storage root).
///
/// The awaited "make sure it is loaded": a cache load already in flight is **waited out**, not
/// reported unavailable — a voice press starts the load a moment before the transcription that
/// needs its result. The wait is bounded ([`LOAD_WAIT_LIMIT`]) and succeeds only on `Ready`;
/// `Failed` is reported as the load's own failure.
///
/// A download is not waited on — it is minutes, not seconds: with no files on disk this spawns it
/// detached and returns `false`. A cache that is on disk but will not load falls through to a
/// download, and the wait ends the moment the loader moves on to it ([`DOWNLOADING`]).
///
/// With no load in flight and the files on disk this claims the load and awaits it (no SHA256
/// re-verification). Used by [`transcribe_file_async`] and the TTS e2e test.
pub async fn ensure_loaded() -> bool {
    if let Some(ready) = try_lock_init() {
        if ready {
            return true;
        }
        let cached = models_subdir(MODEL_DIR_NAME).is_some_and(|dir| cache_present(&dir));
        return cached && wait_until_settled(|| STATE.load(Ordering::Acquire)).await;
    }

    let Some(dir) = models_subdir(MODEL_DIR_NAME) else {
        STATE.store(ModelState::Failed, Ordering::Release);
        return false;
    };

    try_init_inner(dir).await
}

/// The wait itself, over any state source: ends as soon as the answer is known — `Ready`,
/// `Failed`/`Uninit`, a loader that has moved on to downloading ([`DOWNLOADING`]), or
/// [`LOAD_WAIT_LIMIT`].
async fn wait_until_settled(mut state: impl FnMut() -> ModelState) -> bool {
    let deadline = tokio::time::Instant::now() + LOAD_WAIT_LIMIT;
    loop {
        match state() {
            ModelState::Ready => return true,
            // Failed: the load being waited on ended in failure. Uninit: an explicit retry reset it,
            // so the load this call waited on is gone.
            ModelState::Failed | ModelState::Uninit => return false,
            ModelState::Loading if DOWNLOADING.load(Ordering::Acquire) => return false,
            ModelState::Loading => {}
        }
        if tokio::time::Instant::now() >= deadline {
            warn!("Local transcriber: the model load did not settle in time");
            return false;
        }
        tokio::time::sleep(LOAD_WAIT_POLL).await;
    }
}

/// Kick off the local ASR transcriber's load-or-download chain as a background task.
///
/// The caller has already decided the model is needed (the voice stack's first-use demand, a
/// config save that enables transcription, or an explicit retry); nothing here is awaited, so the
/// ~4 s load (or a full download) runs off the caller's path. Boot never calls it: starting the
/// process is not a demand.
///
/// No-op when the transcriber is already ready/loading, or when a previous attempt failed (Failed
/// is terminal — only the voice pipeline's bounded auto-retry or [`retry_init`] re-attempts).
/// Panic-safe: a panic in the spawned chain transitions STATE to [`ModelState::Failed`].
pub fn spawn_background_init() {
    if let Some(result) = try_lock_init() {
        // Ready (true) — already loaded. Loading/Failed (false) — an init is in flight or a
        // previous attempt failed terminally; nothing to start.
        debug!(result, "Local transcriber: background init skipped");
        return;
    }

    let Some(dir) = models_subdir(MODEL_DIR_NAME) else {
        warn!("Local transcriber: cannot resolve model directory (storage root not set)");
        STATE.store(ModelState::Failed, Ordering::Release);
        return;
    };

    tokio::spawn(async move {
        // A panic must never leave STATE stuck in Loading (silently disabling transcription until
        // restart) — store the honest terminal state.
        let result = AssertUnwindSafe(init_background(dir)).catch_unwind().await;
        if result.is_err() {
            warn!("Local transcriber: background init panicked — marking failed");
            STATE.store(ModelState::Failed, Ordering::Release);
        }
    });
}

pub fn is_loaded() -> bool {
    STATE.is_ready()
}

/// The transcriber's lifecycle state, for callers resolving a user-facing status from it (the voice
/// pipeline never preloads, so "not loaded" alone cannot tell a load in flight from none started).
pub(crate) fn state() -> ModelState {
    STATE.load(Ordering::Acquire)
}

/// True while the transcriber's load is in flight (`Loading`).
///
/// Nothing preloads the model, so "not loaded" alone cannot tell a load in
/// flight from one that has not started yet: this distinguishes the two for the
/// voice pipeline's own demand report and for the Settings status.
pub(crate) fn is_loading() -> bool {
    state() == ModelState::Loading
}

/// True if the local transcriber failed to load (terminal `Failed` state).
///
/// Used by the voice pipeline's model gating to surface
/// [`VoiceStatus::ModelError`] when the shared ASR model is unavailable.
pub fn is_failed() -> bool {
    STATE.load(Ordering::Acquire) == ModelState::Failed
}

/// Why [`ensure_loaded`] answered `false`, in the terms the caller reports: a
/// terminal load failure and a load (or download) still in progress are
/// different news for someone who just spoke.
fn unloaded_message() -> &'static str {
    if is_failed() {
        "the local speech model failed to load (the loader logged why)"
    } else {
        "the local speech model is still loading"
    }
}

/// Clone the shared Qwen3-ASR model Arc for encoder reuse by the wake-word pipeline.
///
/// Returns `None` when the transcriber is not loaded (or still loading).
/// The returned model shares the exact same loaded weights as transcription —
/// the wake-word pipeline performs no separate download or model handling.
///
/// Reads the lock-free [`SHARED_MODEL`] slot populated by
/// [`set_transcriber_ready`] — it NEVER touches the transcription context
/// mutex, which [`transcribe_file_async`] holds for the full duration of an
/// inference (up to [`INFERENCE_TIMEOUT`]).  The wake-word stride path calls
/// this on every scoring step; locking the ctx there would block the whole
/// voice pipeline behind any concurrent transcription.
pub fn shared_model_arc() -> Option<Arc<qwen_asr::context::QwenModel>> {
    SHARED_MODEL.lock().unwrap_poison().clone()
}

/// Retry initialisation after a terminal `Failed` state.
///
/// The default transcriber lifecycle treats `Failed` as terminal — no
/// automatic re-attempt.  Two explicit recovery knobs exist: the voice
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

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    // decode_audio_to_mono_f32 — synthetic WAV

    /// Create a minimal valid 16-bit mono PCM WAV file in a temp directory.
    fn write_synthetic_wav(
        dir: &std::path::Path,
        filename: &str,
        sample_rate: u32,
    ) -> std::path::PathBuf {
        let path = dir.join(filename);

        // 1-second 440 Hz sine; encode via the shared production WAV encoder.
        let num_samples = sample_rate as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                #[expect(clippy::cast_precision_loss)] // i < sample_rate (≤16 kHz) — exact in f32
                let t = i as f32 / sample_rate as f32;
                (t * 440.0 * 2.0 * std::f32::consts::PI).sin()
            })
            .collect();
        let wav = crate::audio::tts::render_wav(&samples, sample_rate).unwrap();

        std::fs::write(&path, wav).unwrap();
        path
    }

    /// Decoding lands on the model's rate: a 16 kHz file keeps its sample
    /// count and a 48 kHz file is resampled down to the same second.
    #[test]
    fn test_decode_wav_to_the_model_rate_table() {
        struct Case {
            name: &'static str,
            file_rate: u32,
            /// Inclusive bounds on the decoded sample count for one second.
            expected_len: (usize, usize),
            /// A resampled file starts through the anti-aliasing filter, so
            /// only the untouched rate keeps the sine's zero first sample.
            keeps_the_sine_start: bool,
        }

        let cases = [
            Case {
                name: "mono_16k",
                file_rate: 16000,
                expected_len: (16000, 16000),
                keeps_the_sine_start: true,
            },
            Case {
                // 1 second at 48 kHz → ~16000 samples after resampling.
                name: "mono_48k_resampled",
                file_rate: 48000,
                expected_len: (15_500, 16_500),
                keeps_the_sine_start: false,
            },
        ];

        for case in cases {
            let dir = tempfile::tempdir().unwrap();
            let path = write_synthetic_wav(dir.path(), "test.wav", case.file_rate);
            let samples = decode_audio_to_mono_f32(&path)
                .unwrap_or_else(|e| panic!("case {}: decode failed: {e:?}", case.name));
            assert!(!samples.is_empty(), "case {}: no samples", case.name);
            assert!(
                (case.expected_len.0..=case.expected_len.1).contains(&samples.len()),
                "case {}: expected {}..={} samples, got {}",
                case.name,
                case.expected_len.0,
                case.expected_len.1,
                samples.len(),
            );
            for &s in &samples {
                assert!(
                    (-1.0..=1.0).contains(&s),
                    "case {}: sample {s} out of range",
                    case.name,
                );
            }
            // A sine starting at t=0 begins near zero.
            if case.keeps_the_sine_start {
                assert!(
                    samples[0].abs() < 0.01,
                    "case {}: first sample should be near 0",
                    case.name,
                );
            }
        }
    }

    #[test]
    fn test_decode_wav_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.wav");
        std::fs::write(&path, b"").unwrap();
        let result = decode_audio_to_mono_f32(&path);
        assert!(result.is_err(), "Empty file should fail to decode");
    }

    // decode_audio_to_mono_f32 — extension handling

    #[test]
    fn test_decode_unknown_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audio.xyz");
        std::fs::write(&path, b"not an audio file").unwrap();
        let result = decode_audio_to_mono_f32(&path);
        assert!(result.is_err(), "Unknown extension should fail to decode");
    }

    // OGG/Opus decode

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

    // MP3 decode

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

    // First-use load wait

    /// A load in flight is waited out and reported by the state it settles on:
    /// `Ready` as loaded, `Failed` as the load's own failure. The voice press
    /// starts the load a moment before the transcription that needs it, and
    /// treating that moment as "not available" is exactly the first-utterance
    /// failure this wait exists to prevent.
    #[tokio::test]
    async fn an_in_flight_load_is_waited_out() {
        for (settled, loaded) in [(ModelState::Ready, true), (ModelState::Failed, false)] {
            let state = Arc::new(AtomicModelState::new(ModelState::Loading));
            let settling = state.clone();
            let load = tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(20)).await;
                settling.store(settled, Ordering::Release);
            });
            assert_eq!(
                wait_until_settled(|| state.load(Ordering::Acquire)).await,
                loaded,
                "a load settling on {settled:?} must be reported as loaded={loaded}"
            );
            load.await.unwrap();
        }
    }
}
