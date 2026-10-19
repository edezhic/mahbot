//! Local audio transcription using Qwen3-ASR via the `qwen-asr` crate.
//!
//! # Architecture
//!
//! This module is the **only** audio transcription path — the previous
//! API-based transcriber has been removed.  When the model is
//! available, inference runs fully locally with Qwen3-ASR-0.6B.  The model
//! (~1.88 GB BF16 safetensors) is downloaded on first use to
//! `~/.mahbot/models/qwen3-asr-0.6b/` (in the background — see below) and
//! memory-mapped by the `qwen-asr` crate — zero-copy weight loading with
//! minimal RSS overhead.
//!
//! # Boot strategy
//!
//! Boot never preloads the recogniser. The model is loaded by the first
//! operation that needs it — the first voice message, the voice stack's own
//! demand ([`crate::audio::voice`]), or the Settings toggle that turns the
//! setting on. [`ensure_loaded`] is that awaited "make sure it is loaded": with
//! the weights already on disk (the ordinary case) it loads them (~4 s), or waits
//! out the cache load a voice press already started; otherwise it kicks the
//! background download retry loop and reports that the model is not ready yet.
//!
//! # Download strategy
//!
//! Files are SHA256-verified **once, at download time** (the streaming
//! `download_verified` hasher) — there is no per-use re-verification of the
//! ~1.9 GB model (that was ~4.4 s of the old awaited boot path). A silently
//! corrupted cache is therefore caught at load time instead: `QwenCtx::load`
//! returns `None` and the init falls through to the download+verify recovery
//! loop. Note the load is mmap-based, so partial corruption can survive the
//! load and surface as a SIGBUS at inference rather than a graceful fallback —
//! accepted tradeoff. Missing/corrupt files trigger the
//! background retry loop with exponential backoff.
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
//! # Call sites
//!
//! [`transcribe_file_async`] has three callers: the voice pipeline
//! ([`crate::audio::voice`], a whole recording), the enrichment path
//! ([`crate::channels::enrichment::transcribe_audio_marker`], an attached audio
//! file) and `mahbot audio transcribe`.  When the local model is unavailable or
//! disabled in config, the enrichment caller annotates the message with just the
//! audio-transcription icon combo — there is no API fallback.

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
/// Shared by the voice pipeline, the audio-enrichment path, and
/// `mahbot audio transcribe` — which answers this bound in its own words.
pub(crate) const INFERENCE_TIMEOUT: Duration = Duration::from_mins(10);

/// The recogniser's own bound on one inference ([`INFERENCE_TIMEOUT`]) was
/// reached: the recording is longer than the engine will take.
///
/// A type rather than a message, so `mahbot audio` can answer this failure in its
/// own words — a recording longer than the engine takes, and never a transcript
/// with a part of it missing — without matching on the engine's wording.
#[derive(Debug)]
pub(crate) struct InferenceTimedOut;

impl std::fmt::Display for InferenceTimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Qwen3-ASR inference timed out")
    }
}

impl std::error::Error for InferenceTimedOut {}

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

/// How often a caller that needs the recogniser checks whether an in-flight load
/// has settled ([`ensure_loaded`]).
const LOAD_WAIT_POLL: Duration = Duration::from_millis(100);

/// The longest a caller waits for a load already in flight. Only a cache load is
/// waited on ([`ensure_loaded`]), and a cache read of the 1.88 GB of weights takes
/// seconds even from a cold disk; the bound is far below a download's own
/// ([`MODEL_DOWNLOAD_TIMEOUT`]) so that case can never be waited out.
const LOAD_WAIT_LIMIT: Duration = Duration::from_mins(2);

// ── State machine ─────────────────────────────────────────────────────

/// Global singleton handle — `None` until loaded.
static GLOBAL_TRANSCRIBER: Mutex<Option<QwenLocalTranscriber>> = Mutex::new(None);

/// Lock-free-ish shared-model slot for the wake-word pipeline.
///
/// Populated by [`set_transcriber_ready`] alongside [`GLOBAL_TRANSCRIBER`].
/// The wake-word path reads this slot instead of locking the transcription
/// context mutex — [`transcribe_file_async`] holds the inner `QwenCtx` mutex
/// for the FULL duration of an inference (up to [`INFERENCE_TIMEOUT`]), so a
/// per-stride `shared_model_arc()` that locked the ctx would stall the whole
/// voice pipeline for the length of any concurrent transcription (e.g.
/// Telegram voice-message enrichment).  Cloning an `Arc` under this short
/// mutex is O(1) and uncontended (single writer on transcriber swap).
static SHARED_MODEL: Mutex<Option<Arc<qwen_asr::context::QwenModel>>> = Mutex::new(None);

/// Atomic state tracker to coordinate lazy initialization.
///
/// Model-loading lifecycle for the transcriber (Uninit → Loading → Ready /
/// Failed).  Shared wrapper extracted in
/// [`crate::util::model_state::ModelState`].  Failed is terminal for
/// transcription (the [`transcribe_file_async`] error path leaves the caller
/// with an icon-only audio annotation); the voice pipeline's bounded
/// auto-retry and the explicit GUI retry
/// ([`VoiceCommand::RetryModelLoading`]) are the only recovery knobs.
static STATE: AtomicModelState = AtomicModelState::new(ModelState::Uninit);

/// True while the loader in [`STATE`] is downloading rather than loading a cache
/// — set for the whole run of [`download_retry_loop`], which is entered exactly
/// when there is no cache to load (or the cache on disk would not load).
///
/// A caller waiting for `Loading` stops waiting the moment this is set
/// ([`wait_until_settled`]): a cache read takes seconds, a download takes
/// minutes, and whoever just spoke gets an answer either way.
static DOWNLOADING: AtomicBool = AtomicBool::new(false);

/// Holds [`DOWNLOADING`] for the duration of a download, however it ends — panic
/// or cancellation included.
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
    // Populate the shared-model slot BEFORE publishing the transcriber so the
    // wake-word path can never observe Ready without a model.  The ctx mutex
    // is uncontended here: the transcriber has just been loaded and has not
    // been published to GLOBAL_TRANSCRIBER yet, so no inference can be
    // running on it.
    *SHARED_MODEL.lock().unwrap_poison() = Some(tc.model_arc());
    *GLOBAL_TRANSCRIBER.lock().unwrap_poison() = Some(tc);
    STATE.store(ModelState::Ready, Ordering::Release);
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
    /// Unlike [`ensure_loaded`], this does not resolve the directory via
    /// [`crate::audio::models_subdir`] and therefore does not depend on the
    /// CONFIG storage root being set.
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

    /// Clone the shared immutable [`qwen_asr::context::QwenModel`] Arc held by
    /// this transcriber's context.
    ///
    /// Called once by [`set_transcriber_ready`] to populate the lock-free
    /// [`SHARED_MODEL`] slot the wake-word pipeline reads — it is NOT called
    /// per scoring stride (that would lock the transcription context mutex,
    /// which `transcribe_file_async` holds for the full duration of an
    /// inference).  The encoder weights are read-only after load, so
    /// concurrent `Encoder::forward` calls with per-call scratch buffers are
    /// safe; the wake-word path passes `None` for `enc_bufs` (fresh buffers)
    /// rather than sharing the context-owned `EncoderBuffers`.
    fn model_arc(&self) -> Arc<qwen_asr::context::QwenModel> {
        Arc::clone(&self.ctx.lock().unwrap_poison().model)
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
/// inference step (audio decoding is excluded from the timeout).  Every caller
/// — the voice pipeline (recordings up to 10 minutes), the enrichment path for
/// attached audio, and `mahbot audio transcribe` — passes [`INFERENCE_TIMEOUT`].
///
/// A shutdown guard races against the inference so that a stalled model
/// cannot block pipeline exit.
///
/// Returns an error if the local model cannot be loaded — a cache load already
/// in flight is waited out, while a download (this call's own or another's) and
/// a terminal load failure are reported, distinctly, rather than blocked on or
/// mistaken for each other; the caller annotates the message with just the
/// audio-transcription icon combo on failure.
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

    // Nothing preloads the recogniser: the first operation that needs it loads
    // it here (the ordinary case is the cached weights already on disk) or waits
    // out the load the voice press started.
    anyhow::ensure!(ensure_loaded().await, "{}", unloaded_message());

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
            result.map_err(|_| anyhow::Error::new(InferenceTimedOut))?
        }
        () = shutdown_token.cancelled() => {
            anyhow::bail!("Shutdown during audio transcription");
        }
    };

    text
}

// ── Audio decoding ────────────────────────────────────────────────────

/// The audio names the recogniser takes — the formats are WAV, OGG/Opus and MP3
/// (see the module docs) — and how each one is decoded.
///
/// This is the ONE table of that rule: [`decode_audio_to_mono_f32`] dispatches
/// through it, [`takes_audio`] accepts a file by it, and a caller's refusal lists
/// it, so a format can never be taken by one of them and refused by another. A
/// file under a name outside the table is decoded when its bytes begin with OGG's
/// magic — [`takes_audio`] is the rule a caller asks.
pub(crate) const AUDIO_FORMATS: &[(&str, AudioFormat)] = &[
    ("wav", AudioFormat::Wav),
    ("ogg", AudioFormat::Opus),
    ("oga", AudioFormat::Opus),
    ("mp3", AudioFormat::Mp3),
];

/// The decoding one name in [`AUDIO_FORMATS`] stands for.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AudioFormat {
    /// qwen-asr's own parser for WAV.
    Wav,
    /// Pure-Rust OGG demuxing and Opus decoding.
    Opus,
    /// The `minimp3` decoder.
    Mp3,
}

/// Decode any supported audio file to 16 kHz mono f32 samples.
///
/// Supports WAV (directly via qwen-asr's parser for maximum compatibility),
/// OGG/Opus (Telegram voice messages), MP3, and raw audio.
///
/// `pub(crate)`: the voice-tests wake_word bench's real-audio
/// FAPH phase reuses this decoder for the pinned corpus (WAV/OGG/MP3 only —
/// the corpus format pin), so the bench binary needs no new decoder crates.
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

    // Read the file into memory for decoding.
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read audio file: {}", path.display()))?;

    // Fast path for WAV files — use qwen-asr's native parser.
    if format == Some(AudioFormat::Wav) {
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

    let (samples, sample_rate) = match format {
        Some(AudioFormat::Opus) => decode_opus_from_ogg(&data, path)?,
        Some(AudioFormat::Mp3) => decode_mp3(&data, path)?,
        Some(AudioFormat::Wav) => {
            // Fast-path above already failed — don't retry the same call.
            anyhow::bail!(
                "Failed to decode WAV file (format error): {}",
                path.display()
            );
        }
        // For a name no format is named with, try OGG/Opus: its magic bytes are
        // the only thing that can make such a file audio.
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

    // Resample to 16 kHz if needed.
    if sample_rate == qwen_asr::config::SAMPLE_RATE {
        Ok(samples)
    } else {
        let resampled =
            qwen_asr::audio::resample(&samples, sample_rate, qwen_asr::config::SAMPLE_RATE);
        Ok(resampled)
    }
}

/// Whether the recogniser takes this file — its own rule, and only it: one of the
/// names [`AUDIO_FORMATS`] holds, or, under a name the decoder does not know,
/// content that begins with Ogg's magic (which is how a Telegram voice note is
/// taken). A name [`decode_audio_to_mono_f32`] knows is settled by the NAME, as
/// that decoder settles it. `Err` is a file that cannot be opened at all, which a
/// caller answers as its own misuse rather than as a failure of the work. A caller
/// asks this before it fetches anything, so a typo never costs a model download.
pub(crate) fn takes_audio(path: &Path) -> std::io::Result<bool> {
    // Opening is also the readability check, and asking the rule costs one open.
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

/// Decode an OGG/Opus file into mono f32 samples at the file's sample rate.
///
/// Uses the `ogg` crate for OGG demuxing and the `opus_decoder` crate for
/// pure-Rust Opus decoding (no C dependencies, no unsafe code).
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
                    // Stereo → mono by averaging (midpoint avoids overflow and
                    // rounds identically to (l + r) * 0.5 for normal samples).
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

/// Decode an MP3 file into mono f32 samples at the file's sample rate.
///
/// Uses the `minimp3` crate — a Rust wrapper around the minimp3 C library.
/// This introduces a C compiler build dependency (`minimp3-sys` + `cc` crate).
/// If a pure-Rust MP3 decoder is needed in the future, one could replace this
/// with a subprocess call to `ffmpeg` (handling all audio formats) or a
/// pure-Rust MP3 crate like `mp3-dl`.
#[expect(clippy::cast_precision_loss)]
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
            // Verify existing file on a blocking thread (SHA256 reads
            // the full 1.88 GB model file).
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
            // Remove the corrupted file so the download starts fresh.
            tokio::fs::remove_file(&dest).await.ok();
        }

        info!("Local transcriber: downloading {}", file.filename);
        fetch_starting();

        // download_file already verifies SHA256 during streaming and removes
        // the file on mismatch, so no re-verification is needed on success.
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

/// Background download loop with retry.
///
/// Downloads all model files sequentially. On failure, sleeps with exponential
/// backoff and retries up to [`MAX_DOWNLOAD_RETRIES`] times. On success,
/// loads the model and transitions to [`ModelState::Ready`]. On terminal failure,
/// transitions to [`ModelState::Failed`].
///
/// Panic-safety: a [`ModelLoadGuard`] transitions `Loading → Failed` on drop,
/// so a panic or cancellation in this task never leaves [`STATE`] stuck in
/// [`ModelState::Loading`] (which would silently disable transcription until
/// restart). Since the first-use path drives this loop from a background task,
/// a stuck-Loading state would be both silent and permanent — Failed is the
/// honest terminal state.
async fn download_retry_loop() {
    // A caller waiting on this load stops waiting from here on: its remaining
    // work is a download, not a cache read. Declared first so the terminal state
    // lands before the flag clears.
    let _download = DownloadPhase::enter();
    // Loading → Failed on drop (panic/cancel safety). No-op on the terminal
    // states set by the success/retry-cap paths below.
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

/// Load the transcriber from an already-resolved cache directory on a blocking
/// thread (memory-map + weight setup). Returns `None` when any file is missing
/// or the load fails.
async fn load_from_cache(dir: PathBuf) -> Option<QwenLocalTranscriber> {
    tokio::task::spawn_blocking(move || QwenLocalTranscriber::try_load_from(&dir))
        .await
        .ok()
        .flatten()
}

/// Prepare the recogniser for one operation: the model set, loaded from the
/// cache when it is complete and fetched EXACTLY once when it is not.
///
/// No background task and no retry loop: the product's own daemon retries for
/// as long as it is up and never has to answer a caller, while a command line
/// has one answer to give — a failure is reported, never waited on.
/// `announce_fetch` runs once, before the FIRST file that actually has to be
/// fetched, with the model directory it goes into; a set already on disk fetches
/// nothing and says nothing.
///
/// The caller prepares the recogniser for ONE operation, so nothing here is
/// already up when this is called — the daemon never calls this function.
/// Readiness comes from the model directory alone: the set is loaded from it when
/// it is already complete, and fetched into it when it is not.
pub(crate) async fn ensure_ready_once(announce_fetch: impl FnOnce(&Path)) -> Result<()> {
    let dir = models_subdir(MODEL_DIR_NAME)
        .context("cannot resolve the model directory (the storage root is not set)")?;

    if cache_present(&dir)
        && let Some(tc) = load_from_cache(dir.clone()).await
    {
        set_transcriber_ready(tc);
        return Ok(());
    }

    // The notice is bound to the first file that is actually fetched: the set is
    // verified file by file, and a set already on disk fetches nothing and says
    // nothing.
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

/// Whether every file the recogniser needs is present in `dir`.
fn cache_present(dir: &Path) -> bool {
    dir.join(MODEL_FILENAME).exists()
        && dir.join(VOCAB_FILENAME).exists()
        && dir.join(MERGES_FILENAME).exists()
}

/// Shared cache-check + load logic used by both the awaited entry
/// ([`ensure_loaded`]) and the background load
/// ([`init_background`]). The caller is responsible for the STATE atomic
/// guard.
///
/// Decision 1: no per-boot SHA256 re-verification — files are
/// verified once at download time (`download_verified` hashes the stream; the
/// download loop re-verifies already-present files on recovery). A silently
/// corrupted cache is caught at load time instead: a `None` from
/// [`load_from_cache`] falls through to the download+verify recovery loop.
/// (Tradeoff: the mmap-based load can succeed on partial corruption —
/// accepted; see module docs.)
async fn try_init_inner(dir: PathBuf) -> bool {
    if cache_present(&dir) {
        if let Some(tc) = load_from_cache(dir).await {
            info!("Local transcriber: loaded from cache");
            set_transcriber_ready(tc);
            return true;
        }
        warn!("Local transcriber: cached files present but failed to load");
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

/// Background load-or-download chain — spawned by [`spawn_background_init`] and
/// never awaited by its caller.
///
/// STATE must already be `Loading` (the caller owns the `Uninit → Loading`
/// transition via [`try_lock_init`]). Loads from cache when all files are
/// present; otherwise runs the download+verify retry loop to completion (it is
/// already in the background — no need to detach it further). Both paths
/// transition STATE to a terminal state ([`set_transcriber_ready`] /
/// `Failed`).
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

/// Try to shortcut or acquire the initialisation lock.
///
/// Returns `None` if the caller acquired the lock and should proceed.
/// Returns `Some(true)` if already initialised (ModelState::Ready) — caller
/// should return success.
/// Returns `Some(false)` when a load is in flight (`Loading`) or a previous
/// attempt failed (`Failed`) — the caller owns telling those two apart:
/// [`ensure_loaded`] waits the load out and fails on `Failed`, while
/// [`spawn_background_init`] treats both as nothing to start.
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
/// (resolved via [`crate::audio::models_subdir`], which depends on the CONFIG
/// storage root).
///
/// The awaited "make sure it is loaded": a cache load already in flight is
/// **waited out**, not reported as unavailable — the voice press starts the load
/// a moment before the transcription that needs its result, so failing here would
/// answer the first utterance of a start with "not available" for a load that is
/// finishing. The wait is bounded ([`LOAD_WAIT_LIMIT`]) and succeeds only on
/// `Ready`; `Failed` is reported as the load's own failure.
///
/// A download is not waited on — it is minutes, not seconds. With no files on
/// disk this call either spawns the download detached and returns `false`, or,
/// when another task already has it in flight, returns `false` at once. A cache
/// that is on disk but will not load falls through to a download, and the wait
/// ends the moment the loader moves on to it ([`DOWNLOADING`]).
///
/// With no load in flight and the files on disk this call claims the load and
/// awaits it (fast with a warm page cache — no SHA256 re-verification).
///
/// Used by every first operation that needs the model:
/// [`transcribe_file_async`], the TTS e2e test, and the wake-word benchmark
/// (`voice_pipeline_e2e_test::ensure_voice_models_loaded`).
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

/// The wait itself, over any state source.
///
/// Ends as soon as the answer is known: `Ready`, `Failed`/`Uninit`, a loader that
/// has moved on to downloading ([`DOWNLOADING`]), or [`LOAD_WAIT_LIMIT`]. The
/// state machine is a bare atomic with no notification, so a short poll is the
/// whole mechanism.
async fn wait_until_settled(mut state: impl FnMut() -> ModelState) -> bool {
    let deadline = tokio::time::Instant::now() + LOAD_WAIT_LIMIT;
    loop {
        match state() {
            ModelState::Ready => return true,
            // Failed: the load being waited on ended in failure. Uninit: an
            // explicit retry reset it, so the load this call waited on is gone.
            ModelState::Failed | ModelState::Uninit => return false,
            // Still loading — unless what it is doing from here is a download.
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

/// Kick off the local ASR transcriber's load-or-download chain as a background
/// task.
///
/// The caller has already decided the model is needed (the voice stack's
/// first-use demand, a config save that enables transcription, or an explicit
/// retry); nothing here is awaited, so the ~4 s model load (or a full download
/// when the cache is missing) runs off the caller's path. Boot never calls it:
/// starting the process is not a demand.
///
/// No-op when the transcriber is already ready/loading, or when a previous
/// attempt failed (Failed is terminal for the automatic lifecycle — only the
/// voice pipeline's bounded auto-retry or the explicit GUI retry
/// ([`retry_init`]) re-attempts).
/// Panic-safe: a panic inside the spawned chain transitions STATE to
/// [`ModelState::Failed`] instead of leaving it stuck in Loading.
pub fn spawn_background_init() {
    if let Some(result) = try_lock_init() {
        // Ready (true) — already loaded. Loading/Failed (false) — an init is
        // already in flight or a previous attempt failed terminally; nothing
        // to start.
        debug!(result, "Local transcriber: background init skipped");
        return;
    }

    let Some(dir) = models_subdir(MODEL_DIR_NAME) else {
        warn!("Local transcriber: cannot resolve model directory (storage root not set)");
        STATE.store(ModelState::Failed, Ordering::Release);
        return;
    };

    tokio::spawn(async move {
        // A panic must never leave STATE stuck in Loading (silently disabling
        // transcription until restart) — store the honest terminal state.
        let result = AssertUnwindSafe(init_background(dir)).catch_unwind().await;
        if result.is_err() {
            warn!("Local transcriber: background init panicked — marking failed");
            STATE.store(ModelState::Failed, Ordering::Release);
        }
    });
}

/// True if the local transcriber is loaded and ready for use.
pub fn is_loaded() -> bool {
    STATE.is_ready()
}

/// The transcriber's lifecycle state, for callers that resolve a user-facing
/// status from it (the voice pipeline never preloads the model, so "not loaded"
/// alone cannot tell a load in flight from none having started).
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

/// Clone the shared Qwen3-ASR model Arc for encoder reuse by the wake-word
/// pipeline.
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
            assert!((-1.0..=1.0).contains(&s), "Sample {s} out of range");
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
            tags.extend_from_slice(&u32::try_from(vendor.len()).unwrap().to_le_bytes());
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

    // ── First-use load wait ─────────────────────────────────────────────

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
