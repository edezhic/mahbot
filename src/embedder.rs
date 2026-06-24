//! Semantic (vector/embedding) search for archived tickets.
//!
//! # Why this exists
//!
//! This module provides a local Candle + GGUF-based embedding model that converts
//! ticket descriptions into dense vectors. These vectors enable **semantic search** —
//! finding archived tickets by *conceptual similarity* (e.g. `"authentication bug"`
//! matching `"login issue"`) — which pure FTS keyword search alone cannot do.
//!
//! The embedding model uses **jinaai/jina-embeddings-v5-text-nano-retrieval**
//! (Q4_K_M GGUF, ~150 MB), a EuroBERT architecture (LLaMA-style encoder without
//! causal masking) loaded via the Candle framework. The model and tokenizer are
//! downloaded on first use and cached in `~/.mahbot/models/`.
//!
//! The embedder is loaded **lazily on first `embed()` call** and cached in a global
//! [`RwLock`]; embedding is computed at **ticket creation time** (to pre-compute a
//! vector for future archived search) and at **search-query time** (to vectorize
//! the query for `SearchArchivedTicketsTool`). Both paths gracefully degrade: if
//! the model files haven't been downloaded yet (or download fails), `embed()`
//! returns `None` and the caller falls back to FTS-only search. A background retry
//! loop downloads the model with exponential backoff, making the embedder available
//! without requiring a restart.
//!
//! ## Product decision
//!
//! **Do not propose removing this module or its dependencies without explicit
//! user approval.** It is a deliberate product decision, not accidental bloat
//! or dead code.

use crate::util::UnwrapPoison;
use anyhow::{Context, Result, anyhow};
use candle_core::quantized::{QMatMul, gguf_file};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Module};
use futures_util::StreamExt;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{OnceLock, RwLock};
use std::time::Duration;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

// ── Constants ────────────────────────────────────────────────────────

/// Maximum sequence length for the embedding model (jina-embeddings-v5 supports 8192).
const MAX_SEQ_LEN: usize = 8192;

/// RoPE base frequency for the model.
const ROPE_FREQ_BASE: f32 = 1_000_000.0;

/// Timeout for model file download (10 minutes for ~150 MB).
const MODEL_DOWNLOAD_TIMEOUT: Duration = Duration::from_mins(10);

/// Timeout for tokenizer file download (2 minutes for ~17 MB).
const TOKENIZER_DOWNLOAD_TIMEOUT: Duration = Duration::from_mins(2);

/// Default pad token ID for this model (tokenizer's eos_token_id = 128001).
const DEFAULT_PAD_ID: u32 = 128_001;

// ── Model URLs ───────────────────────────────────────────────────────

/// HuggingFace URL for the quantized GGUF model file.
const MODEL_URL: &str = "https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval/resolve/main/v5-nano-retrieval-Q4_K_M.gguf";

/// SHA256 checksum of the model file (verified at download time).
const MODEL_SHA256: &str = "f50822244ba0c7a348c5455b99bb8a0afd182511e8a816888c5dc65d972e51d5";

/// HuggingFace URL for the tokenizer file.
const TOKENIZER_URL: &str = "https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval/resolve/main/tokenizer.json";

/// SHA256 checksum of the tokenizer file (verified at download time).
const TOKENIZER_SHA256: &str = "98d4a1d32152d6cedf85b5e88f3b205106dca1fe72aaab34e0ac13c238421069";

// ── Global state ─────────────────────────────────────────────────────

/// Embedder state machine.
///
/// 0 = UNINIT (first call triggers load/download)
/// 1 = LOADING (background download or sync load in progress)
/// 2 = READY (embedder is available)
const STATE_UNINIT: u8 = 0;
const STATE_LOADING: u8 = 1;
const STATE_READY: u8 = 2;

/// Global embedder singleton, wrapped in an Option for graceful degradation.
static GLOBAL_EMBEDDER: OnceLock<RwLock<Option<Embedder>>> = OnceLock::new();

/// Atomic state tracker to coordinate lazy initialization.
static STATE: AtomicU8 = AtomicU8::new(STATE_UNINIT);

/// Whether a background download has been spawned.
static DOWNLOAD_SPAWNED: AtomicBool = AtomicBool::new(false);

/// Returns a reference to the global singleton [`Embedder`] RwLock.
///
/// Unlike the previous ONNX implementation, this NEVER panics. If the model
/// files are missing or download hasn't completed, the Option will be `None`
/// and [`embed()`] will gracefully return `None`.
#[must_use]
pub fn global_embedder() -> &'static RwLock<Option<Embedder>> {
    GLOBAL_EMBEDDER.get_or_init(|| RwLock::new(None))
}

/// Try to initialize the embedder (sync load from cache or spawn background download).
///
/// Called on every [`embed()`] invocation. Returns `true` if the embedder is
/// ready, `false` if it's still loading or permanently unavailable.
fn ensure_embedder() -> bool {
    // Fast path: already ready
    if STATE.load(Ordering::Acquire) == STATE_READY {
        return true;
    }

    // Already loading (or failed) — return false, embed() will return None.
    // The background retry loop will eventually set state to READY.
    if STATE.load(Ordering::Acquire) != STATE_UNINIT {
        return false;
    }

    // Try to become the initializer (atomic CAS to prevent races)
    if STATE
        .compare_exchange(
            STATE_UNINIT,
            STATE_LOADING,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        return false;
    }

    // Thread-local: try to load cached files synchronously
    let Some(models_dir) = models_dir() else {
        // CONFIG not initialized yet — can't locate model cache.
        STATE.store(STATE_UNINIT, Ordering::Release);
        return false;
    };
    let model_path = models_dir.join("v5-nano-retrieval-Q4_K_M.gguf");
    let tokenizer_path = models_dir.join("embed_tokenizer.json");

    std::fs::create_dir_all(&models_dir).ok();

    let cache_loaded = if model_path.exists() && tokenizer_path.exists() {
        match Embedder::load(&model_path, &tokenizer_path) {
            Ok(emb) => {
                *global_embedder().write().unwrap_poison() = Some(emb);
                STATE.store(STATE_READY, Ordering::Release);
                true
            }
            Err(e) => {
                warn!(reason = %e, "Failed to load cached embedding model");
                // Don't delete cached files here — let the retry loop attempt
                // to load again with backoff. The files passed SHA256 verification
                // at download time, so the failure is likely a code-level issue
                // (not corruption). Deleting on every transient error would force
                // an unnecessary ~167 MB re-download with 1-minute minimum delay.
                false
            }
        }
    } else {
        false
    };

    if cache_loaded {
        return true;
    }

    // Spawn background download (only once)
    if !DOWNLOAD_SPAWNED.swap(true, Ordering::AcqRel) {
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::spawn(download_retry_loop());
        } else {
            // No tokio runtime available (e.g., in unit tests without runtime).
            // The download will be triggered on the next call when a runtime exists.
            // Reset state to UNINIT so the next caller retries the cache check + spawn.
            // Reset both atomics: DOWNLOAD_SPAWNED first (via AcqRel swap), then STATE.
            // The swap provides an atomic full barrier: any concurrent thread calling
            // DOWNLOAD_SPAWNED.swap(true, ...) either sees the old true and skips, or
            // sees false after our store and will attempt to spawn. After the barrier,
            // STATE is set to UNINIT so the next caller re-enters ensure_embedder().
            DOWNLOAD_SPAWNED.store(false, Ordering::Release);
            STATE.store(STATE_UNINIT, Ordering::Release);
        }
    }

    false
}

// ── Public API ───────────────────────────────────────────────────────

/// Embed a single text using the global embedder singleton.
///
/// `is_query` controls whether the text is embedded as a query (prefixed with
/// `"Query: "`) or as a document (prefixed with `"Document: "`), as required
/// by the embedding model's training.
///
/// Returns `None` if:
/// - The model hasn't been downloaded yet (first call triggers background download).
/// - Download is in progress.
/// - Model loading failed (corrupted file, etc.).
/// - The embedder mutex is poisoned.
#[must_use]
pub fn embed(text: &str, is_query: bool) -> Option<Vec<f32>> {
    if !ensure_embedder() {
        return None;
    }

    let guard = global_embedder().read().unwrap_poison();
    let emb = guard.as_ref()?;
    let v = if is_query {
        emb.embed_queries(&[text]).ok()?
    } else {
        emb.embed_documents(&[text]).ok()?
    };
    v.into_iter().next()
}

// ── Background download with retry ────────────────────────────────────

/// Background retry loop that downloads model and tokenizer files.
///
/// Uses exponential backoff (1 min → 2 min → 4 min → … → 30 min max).
/// Continues indefinitely until both files are downloaded successfully.
async fn download_retry_loop() {
    let models_dir =
        models_dir().expect("CONFIG storage_root must be set before download_retry_loop runs");
    std::fs::create_dir_all(&models_dir).ok();

    let model_dest = models_dir.join("v5-nano-retrieval-Q4_K_M.gguf");
    let tokenizer_dest = models_dir.join("embed_tokenizer.json");

    // Shared HTTP client reused across retries (avoids new TLS handshake per iteration).
    let client = reqwest::Client::builder()
        .timeout(MODEL_DOWNLOAD_TIMEOUT)
        .connect_timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to build reqwest::Client for model download");

    let mut delay = Duration::from_mins(1);
    let max_delay = Duration::from_mins(30); // 30 minutes

    // Pre-check which files already exist (from a previous partial success).
    // This avoids re-downloading valid files on retry iterations.
    let mut model_has = model_dest.exists() && tokenizer_dest.exists();

    loop {
        if model_has {
            // Both files already present from a previous iteration — try to load.
            if let Ok(emb) = Embedder::load(&model_dest, &tokenizer_dest) {
                info!("Embedding model loaded successfully (from previously downloaded files)");
                *global_embedder().write().unwrap_poison() = Some(emb);
                STATE.store(STATE_READY, Ordering::Release);
                return;
            }
            // Loading failed — could be a code bug, not necessarily corrupted files.
            // Don't delete cached files; the backoff will apply and we'll retry.
            warn!("Failed to load embedding model from cached files, retrying with backoff");
        }

        // Download both files concurrently, skipping files that already exist.
        let (model_result, tokenizer_result) = tokio::join!(
            maybe_download(
                &client,
                MODEL_URL,
                &model_dest,
                MODEL_DOWNLOAD_TIMEOUT,
                Some(MODEL_SHA256)
            ),
            maybe_download(
                &client,
                TOKENIZER_URL,
                &tokenizer_dest,
                TOKENIZER_DOWNLOAD_TIMEOUT,
                Some(TOKENIZER_SHA256)
            ),
        );

        if let (Err(e_model), Err(e_tokenizer)) = (&model_result, &tokenizer_result) {
            warn!(
                model_error = %e_model,
                tokenizer_error = %e_tokenizer,
                retry_after_secs = delay.as_secs(),
                "Failed to download embedding model files, retrying"
            );
        } else if let Err(e) = &model_result {
            warn!(error = %e, retry_after_secs = delay.as_secs(), "Failed to download embedding model, retrying");
        } else if let Err(e) = &tokenizer_result {
            warn!(error = %e, retry_after_secs = delay.as_secs(), "Failed to download tokenizer, retrying");
        }

        let model_ok = model_result.is_ok();
        let tokenizer_ok = tokenizer_result.is_ok();

        if model_ok && tokenizer_ok {
            // Both downloaded successfully — try to load the embedder
            match Embedder::load(&model_dest, &tokenizer_dest) {
                Ok(emb) => {
                    info!("Embedding model loaded successfully after download");
                    *global_embedder().write().unwrap_poison() = Some(emb);
                    STATE.store(STATE_READY, Ordering::Release);
                    return;
                }
                Err(e) => {
                    warn!(reason = %e, "Failed to load model after download, retrying with backoff (files preserved)");
                    // Don't delete cached files — load failure may be a code bug,
                    // not file corruption. The backoff will apply and we'll retry.
                }
            }
        } else {
            // Partial failure: at least one download failed.
            // Don't delete successfully downloaded files — maybe_download
            // skips existing files on the next iteration, so a valid file
            // from a partial success is reused without re-downloading.
            if !model_ok {
                // Clean up .tmp file that download_file may have left on error
                let _ = std::fs::remove_file(model_dest.with_extension("tmp"));
            }
            if !tokenizer_ok {
                let _ = std::fs::remove_file(tokenizer_dest.with_extension("tmp"));
            }
        }

        // Track which files exist for the next iteration's pre-check.
        model_has = model_dest.exists() && tokenizer_dest.exists();

        // Wait with exponential backoff
        tokio::time::sleep(delay).await;
        delay = (delay * 2).min(max_delay);
    }
}

/// Download a file unless it already exists. Uses the shared HTTP client for
/// connection reuse across retries.
async fn maybe_download(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    _timeout: Duration,
    expected_sha256: Option<&str>,
) -> Result<()> {
    // Skip download if the file already exists (from a previous partial success).
    if dest.exists() {
        return Ok(());
    }
    download_file(client, url, dest, expected_sha256).await
}

/// Download a single file with atomic write and size verification.
async fn download_file(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
) -> Result<()> {
    use sha2::{Digest, Sha256};

    let response = client
        .get(url)
        .send()
        .await
        .context("Failed to send download request")?;

    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("HTTP {status} from {url}");
    }

    let total_size = response.content_length();

    // Download to temporary file, then atomically rename
    let tmp_path = dest.with_extension("tmp");
    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .context("Failed to create temp file")?;

    let mut downloaded: u64 = 0;
    let mut hasher = expected_sha256.map(|_| Sha256::new());
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Download stream error")?;
        let len = chunk.len() as u64;
        downloaded += len;
        if let Some(ref mut h) = hasher {
            h.update(&chunk);
        }
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
            .await
            .context("Failed to write download chunk")?;
    }

    // Verify file size against Content-Length header
    if let Some(expected) = total_size
        && downloaded != expected
    {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        anyhow::bail!("Download size mismatch: expected {expected} bytes, got {downloaded} bytes");
    }

    // Verify SHA256 checksum if requested (computed during download stream above)
    if let Some(expected_hex) = expected_sha256
        && let Some(hasher) = hasher
    {
        let actual_hash = format!("{:x}", hasher.finalize());
        if actual_hash != expected_hex {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            anyhow::bail!("SHA256 mismatch: expected {expected_hex}, got {actual_hash}");
        }
    }

    // Atomic rename
    tokio::fs::rename(&tmp_path, dest)
        .await
        .context("Failed to rename temp file to final path")?;

    info!(path = %dest.display(), size = downloaded, "Downloaded model file");
    Ok(())
}

// ── Model paths ──────────────────────────────────────────────────────

/// Returns the `~/.mahbot/models/` directory via CONFIG, or `None` if CONFIG
/// storage root hasn't been initialized yet.
fn models_dir() -> Option<PathBuf> {
    crate::config::CONFIG
        .try_storage_root()
        .map(|root| root.join("models"))
}

// ── GGUF metadata helpers ────────────────────────────────────────────

/// Extract a `u32` value from GGUF metadata.
fn get_meta_u32(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Result<u32> {
    metadata
        .get(key)
        .ok_or_else(|| anyhow!("Missing metadata key '{key}'"))?
        .to_u32()
        .map_err(|e| anyhow!("Failed to read metadata '{key}': {e}"))
}

/// Extract an `f32` value from GGUF metadata.
fn get_meta_f32(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Result<f32> {
    metadata
        .get(key)
        .ok_or_else(|| anyhow!("Missing metadata key '{key}'"))?
        .to_f32()
        .map_err(|e| anyhow!("Failed to read metadata '{key}': {e}"))
}

// ── EuroBERT / LLaMA-style encoder model ────────────────────────────

/// A single transformer layer (EuroBERT = LLaMA-style encoder with SwiGLU MLP).
#[derive(Debug)]
struct Layer {
    /// Attention Q projection (no bias).
    attn_q: QMatMul,
    /// Attention K projection (no bias).
    attn_k: QMatMul,
    /// Attention V projection (no bias).
    attn_v: QMatMul,
    /// Attention output projection (no bias).
    attn_o: QMatMul,
    /// Pre-attention RMSNorm weight (1D, hidden_size).
    attn_norm: Tensor,
    /// SwiGLU gate projection (no bias).
    ffn_gate: QMatMul,
    /// SwiGLU up projection (no bias).
    ffn_up: QMatMul,
    /// SwiGLU down projection (no bias).
    ffn_down: QMatMul,
    /// Pre-FFN RMSNorm weight (1D, hidden_size).
    ffn_norm: Tensor,
}

impl Layer {
    /// Forward pass through one encoder layer.
    #[allow(clippy::many_single_char_names)]
    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        n_head: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        // ── Self-attention with pre-norm ──
        let residual = x;
        let h = candle_nn::ops::rms_norm(x, &self.attn_norm, 1e-5)?;

        // Project to Q, K, V
        let q = self.attn_q.forward(&h)?;
        let k = self.attn_k.forward(&h)?;
        let v = self.attn_v.forward(&h)?;

        // Multi-head attention
        let h = Layer::apply_attention(&q, &k, &v, mask, cos, sin, n_head, head_dim)?;
        let h = self.attn_o.forward(&h)?;
        let h = (h + residual)?;

        // ── SwiGLU MLP with pre-norm ──
        let residual = &h;
        let h = candle_nn::ops::rms_norm(&h, &self.ffn_norm, 1e-5)?;

        let gate = self.ffn_gate.forward(&h)?;
        let up = self.ffn_up.forward(&h)?;
        let h = self
            .ffn_down
            .forward(&(candle_nn::ops::silu(&gate)? * up)?)?;
        let h = (h + residual)?;

        Ok(h)
    }

    /// Bidirectional multi-head attention with RoPE (no KV cache).
    #[allow(clippy::too_many_arguments)]
    fn apply_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        n_head: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = q.shape().dims3()?;

        // Reshape: [batch, seq, n_head * head_dim] -> [batch, seq, n_head, head_dim] -> [batch, n_head, seq, head_dim]
        let q = q
            .reshape((b_sz, seq_len, n_head, head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, n_head, head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, n_head, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply RoPE
        let q = Self::apply_rotary_emb(&q, cos, sin)?;
        let k = Self::apply_rotary_emb(&k, cos, sin)?;

        // Scaled dot-product attention (no causal mask — full bidirectional)
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let att = q.matmul(&k.t()?)?;
        let att = (att * scale)?;
        let mask = mask.broadcast_as(att.shape())?;
        let att = (att + mask)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v)?;

        // Reshape back: [batch, n_head, seq, head_dim] -> [batch, seq, n_embd]
        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, n_embd))?;
        Ok(y)
    }

    /// Apply rotary position embeddings.
    fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (_b_sz, _n_head, seq_len, head_dim) = x.shape().dims4()?;
        let cos = cos.narrow(0, 0, seq_len)?;
        let sin = sin.narrow(0, 0, seq_len)?;

        // Reshape to broadcast over batch and head dimensions
        let cos = cos.reshape((1, 1, seq_len, head_dim / 2))?;
        let sin = sin.reshape((1, 1, seq_len, head_dim / 2))?;

        let x_f32 = x.to_dtype(DType::F32)?;
        // Split along head_dim into two halves: [0..d/2) and [d/2..d)
        let chunks = x_f32.chunk(2, 3)?;
        let x1 = &chunks[0];
        let x2 = &chunks[1];
        let y1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let y2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        let result = Tensor::cat(&[&y1, &y2], 3)?;
        // Convert back to original dtype
        Ok(result.to_dtype(x.dtype())?)
    }
}

// ── Embedder ─────────────────────────────────────────────────────────

/// The embedding model: EuroBERT encoder + tokenizer + pooling.
pub struct Embedder {
    tokenizer: Tokenizer,
    tok_embeddings: Embedding,
    layers: Vec<Layer>,
    output_norm: Tensor,
    cos: Tensor,
    sin: Tensor,
    head_dim: usize,
    n_head: usize,
    pad_id: u32,
}

impl Embedder {
    /// Load an [`Embedder`] from cached GGUF and tokenizer files.
    ///
    /// Does NOT download — the caller is responsible for ensuring the files exist.
    /// Returns an error if files are missing, corrupted, or the model architecture
    /// is unexpected.
    #[allow(clippy::too_many_lines)]
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            anyhow!(
                "Failed to load tokenizer from {}: {e}",
                tokenizer_path.display()
            )
        })?;

        // Discover pad token ID from the tokenizer.
        // jina-embeddings-v5 uses eos_token_id = 128001 as pad token.
        let pad_id = tokenizer
            .token_to_id("<|end_of_text|>")
            .or_else(|| tokenizer.token_to_id("<|pad|>"))
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .map_or(DEFAULT_PAD_ID, |id| id);

        debug!(pad_id, "Discovered pad token ID from tokenizer");

        let device = Device::Cpu;

        // Open and read GGUF file
        let mut file = std::fs::File::open(model_path)
            .map_err(|e| anyhow!("Failed to open model file {}: {e}", model_path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow!("Failed to read GGUF file: {e}"))?;

        // Read architecture metadata
        let hidden_size = get_meta_u32(&content.metadata, "eurobert.embedding_length")? as usize;
        let n_head = get_meta_u32(&content.metadata, "eurobert.attention.head_count")? as usize;
        let head_dim = get_meta_u32(&content.metadata, "eurobert.attention.value_length")? as usize;
        let rope_freq_base =
            get_meta_f32(&content.metadata, "eurobert.rope.freq_base").unwrap_or(ROPE_FREQ_BASE);

        // Count layers by scanning tensor names
        let n_layers = content
            .tensor_infos
            .keys()
            .filter_map(|name| {
                let name = name.as_str();
                if name.starts_with("blk.") && name.ends_with(".attn_q.weight") {
                    // Extract layer index
                    name.trim_start_matches("blk.")
                        .split('.')
                        .next()?
                        .parse::<usize>()
                        .ok()
                } else {
                    None
                }
            })
            .max()
            .map(|max| max + 1)
            .context("No layer tensors found in GGUF file")?;

        info!(
            hidden_size,
            n_head, head_dim, n_layers, rope_freq_base, "Loading EuroBERT embedding model"
        );

        // ── Load token embeddings (dequantize for use with candle_nn::Embedding) ──
        let tok_embd_qt = content
            .tensor(&mut file, "token_embd.weight", &device)
            .context("Failed to load token_embd.weight")?;
        let tok_embd_f32 = tok_embd_qt
            .dequantize(&device)
            .context("Failed to dequantize token_embd.weight")?;
        let tok_embeddings = Embedding::new(tok_embd_f32, hidden_size);

        // ── Load output norm ──
        let output_norm_qt = content
            .tensor(&mut file, "output_norm.weight", &device)
            .context("Failed to load output_norm.weight")?;
        let output_norm = output_norm_qt
            .dequantize(&device)
            .context("Failed to dequantize output_norm.weight")?;

        // ── Load transformer layers ──
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let prefix = format!("blk.{i}");

            let attn_q = QMatMul::from_qtensor(
                content
                    .tensor(&mut file, &format!("{prefix}.attn_q.weight"), &device)
                    .with_context(|| format!("Failed to load {prefix}.attn_q.weight"))?,
            )
            .context("Failed to create QMatMul for attn_q")?;

            let attn_k = QMatMul::from_qtensor(
                content
                    .tensor(&mut file, &format!("{prefix}.attn_k.weight"), &device)
                    .with_context(|| format!("Failed to load {prefix}.attn_k.weight"))?,
            )
            .context("Failed to create QMatMul for attn_k")?;

            let attn_v = QMatMul::from_qtensor(
                content
                    .tensor(&mut file, &format!("{prefix}.attn_v.weight"), &device)
                    .with_context(|| format!("Failed to load {prefix}.attn_v.weight"))?,
            )
            .context("Failed to create QMatMul for attn_v")?;

            let attn_o = QMatMul::from_qtensor(
                content
                    .tensor(&mut file, &format!("{prefix}.attn_output.weight"), &device)
                    .with_context(|| format!("Failed to load {prefix}.attn_output.weight"))?,
            )
            .context("Failed to create QMatMul for attn_o")?;

            let attn_norm = content
                .tensor(&mut file, &format!("{prefix}.attn_norm.weight"), &device)
                .with_context(|| format!("Failed to load {prefix}.attn_norm.weight"))?
                .dequantize(&device)
                .context("Failed to dequantize attn_norm")?;

            let ffn_gate = QMatMul::from_qtensor(
                content
                    .tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &device)
                    .with_context(|| format!("Failed to load {prefix}.ffn_gate.weight"))?,
            )
            .context("Failed to create QMatMul for ffn_gate")?;

            let ffn_up = QMatMul::from_qtensor(
                content
                    .tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &device)
                    .with_context(|| format!("Failed to load {prefix}.ffn_up.weight"))?,
            )
            .context("Failed to create QMatMul for ffn_up")?;

            let ffn_down = QMatMul::from_qtensor(
                content
                    .tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &device)
                    .with_context(|| format!("Failed to load {prefix}.ffn_down.weight"))?,
            )
            .context("Failed to create QMatMul for ffn_down")?;

            let ffn_norm = content
                .tensor(&mut file, &format!("{prefix}.ffn_norm.weight"), &device)
                .with_context(|| format!("Failed to load {prefix}.ffn_norm.weight"))?
                .dequantize(&device)
                .context("Failed to dequantize ffn_norm")?;

            layers.push(Layer {
                attn_q,
                attn_k,
                attn_v,
                attn_o,
                attn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                ffn_norm,
            });
        }

        // ── Precompute RoPE frequencies ──
        let (cos, sin) = precompute_freqs_cis(head_dim, rope_freq_base, &device)?;

        // ── Build embedder ──
        let emb = Self {
            tokenizer,
            tok_embeddings,
            layers,
            output_norm,
            cos,
            sin,
            head_dim,
            n_head,
            pad_id,
        };

        // ── Warm-up: run a single short input to validate the model ──
        let v = emb.embed_documents(&["."])?;
        anyhow::ensure!(
            !v.is_empty() && !v[0].is_empty(),
            "Embedder warm-up produced empty output"
        );
        anyhow::ensure!(
            v[0].len() == hidden_size,
            "Embedder warm-up produced wrong dimension: expected {hidden_size}, got {}",
            v[0].len()
        );

        info!("Embedder initialized successfully (hidden_size={hidden_size}, layers={n_layers})");
        Ok(emb)
    }

    /// Embed texts as queries (prefixed with `"Query: "`).
    pub fn embed_queries(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_prefixed("Query: ", texts)
    }

    /// Embed texts as documents (prefixed with `"Document: "`).
    pub fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_prefixed("Document: ", texts)
    }

    /// Core embedding method.
    fn embed_prefixed(&self, prefix: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // ── Tokenize ──
        let encodings: Vec<_> = texts
            .iter()
            .map(|t| {
                let input = format!("{prefix}{t}");
                self.tokenizer.encode(input, true)
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("Tokenization error: {e}"))?;

        if encodings.is_empty() {
            anyhow::bail!("Empty input");
        }

        // Determine max sequence length across batch (clamped to MAX_SEQ_LEN)
        let max_len = encodings
            .iter()
            .map(|e| e.len().min(MAX_SEQ_LEN))
            .max()
            .context("Empty encoding")?;

        let batch_size = encodings.len();

        // ── Build input_ids and attention_mask ──
        let mut input_ids_vec = vec![i64::from(self.pad_id); batch_size * max_len];
        let mut attention_mask_vec = vec![0i64; batch_size * max_len];

        for (row, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let len = ids.len().min(MAX_SEQ_LEN);
            for col in 0..len {
                input_ids_vec[row * max_len + col] = i64::from(ids[col]);
                attention_mask_vec[row * max_len + col] = i64::from(mask[col]);
            }
        }

        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, max_len), &Device::Cpu)?;
        let attention_mask =
            Tensor::from_vec(attention_mask_vec, (batch_size, max_len), &Device::Cpu)?;

        // ── Forward pass through the model ──
        let embeddings = self.forward(&input_ids, &attention_mask)?;

        // ── Post-process: last-token pooling + L2 normalization ──
        let result = last_token_pool_and_l2_normalize(&embeddings, &attention_mask)?;

        Ok(result)
    }

    /// Full model forward pass.
    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (_batch_size, _seq_len) = input_ids.shape().dims2()?;

        // Create bidirectional attention mask (no causal masking since this is an encoder)
        let mask = build_attn_mask(attention_mask, &Device::Cpu)?;

        // Token embeddings
        let mut h = self.tok_embeddings.forward(input_ids)?;
        // h: [batch, seq, hidden_size]

        // Pass through all transformer layers
        for layer in &self.layers {
            h = layer.forward(&h, &mask, &self.cos, &self.sin, self.n_head, self.head_dim)?;
        }

        // Final norm
        h = candle_nn::ops::rms_norm(&h, &self.output_norm, 1e-5)?;

        Ok(h)
    }
}

// ── RoPE ─────────────────────────────────────────────────────────────

/// Precompute cosine and sine tables for rotary position embeddings.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless
)]
fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    #[allow(clippy::cast_precision_loss, clippy::cast_lossless)]
    let theta: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0_f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();

    let theta = Tensor::from_vec(theta, (head_dim / 2,), device)?;
    #[allow(clippy::cast_possible_truncation)]
    let positions = Tensor::arange(0u32, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?;

    let idx_theta = positions.matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;

    Ok((cos, sin))
}

// ── Attention mask ───────────────────────────────────────────────────

/// Build a bidirectional attention mask from a tokenizer attention mask.
///
/// The input `attention_mask` has shape `[batch, seq]` with 1 for real tokens and
/// 0 for padding. The output has shape `[batch, 1, seq, seq]` where:
/// - Entry (i,j) is 0 if both positions i and j are real tokens,
/// - Otherwise it's a large negative value (-1e10) that acts as -inf for softmax.
///
/// Uses `-1e10` instead of `f32::NEG_INFINITY` because `0 * NEG_INFINITY = NaN`
/// per IEEE 754, which would corrupt the entire attention computation.
fn build_attn_mask(attention_mask: &Tensor, device: &Device) -> Result<Tensor> {
    let (batch_size, seq_len) = attention_mask.shape().dims2()?;

    // Expand to [batch, 1, seq, seq]: mask[i, j] = mask[i] * mask[j]
    // (both tokens must be real to attend)
    let mask_f32 = attention_mask.to_dtype(DType::F32)?;
    let mask_a = mask_f32.reshape((batch_size, 1, 1, seq_len))?;
    let mask_b = mask_f32.reshape((batch_size, 1, seq_len, 1))?;
    let pairwise = mask_a.broadcast_mul(&mask_b)?;
    // pairwise now has 1 where both are real, 0 where either is padding

    // Convert to attention mask using where_cond:
    // - Where pairwise == 1 (attend): mask = 0
    // - Where pairwise == 0 (masked): mask = -1e10
    // Using -1e10 instead of NEG_INFINITY because 0 * NEG_INFINITY = NaN.
    let large_neg = Tensor::new(-1e10_f32, device)?.broadcast_as(pairwise.shape())?;
    let zero = Tensor::new(0.0_f32, device)?.broadcast_as(pairwise.shape())?;
    // Build boolean predicate: pairwise == 0
    let mask_cond = pairwise.eq(&zero)?;
    let mask = mask_cond.where_cond(&large_neg, &zero)?;

    Ok(mask)
}

// ── Pooling and normalization ────────────────────────────────────────

/// Extract embeddings via last-token pooling and L2 normalize.
///
/// Takes the embedding at the last non-padding token position for each sequence,
/// then L2-normalizes each vector.
fn last_token_pool_and_l2_normalize(
    embeddings: &Tensor,
    attention_mask: &Tensor,
) -> Result<Vec<Vec<f32>>> {
    let (batch_size, _seq_len, hidden_size) = embeddings.shape().dims3()?;

    let mut results = Vec::with_capacity(batch_size);

    // Sum attention mask along seq dimension to find last real token position
    // last_pos = sum(mask) - 1 (0-indexed)
    let mask_f32 = attention_mask.to_dtype(DType::F32)?;
    let seq_lengths: Vec<i64> = mask_f32.sum(1)?.to_vec1()?;

    for (i, &seq_len) in seq_lengths.iter().enumerate().take(batch_size) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let last_pos = (seq_len - 1).max(0) as usize;

        // Extract embedding at last position: [batch, seq, hidden] -> [hidden]
        let token_emb = embeddings.narrow(0, i, 1)?.narrow(1, last_pos, 1)?;
        let token_emb = token_emb.reshape(hidden_size)?;

        // L2 normalize
        let norm = token_emb
            .sqr()?
            .sum_all()?
            .sqrt()?
            .to_scalar::<f32>()?
            .max(1e-12);
        let normalized = token_emb.broadcast_div(&Tensor::new(norm, token_emb.device())?)?;

        let vec: Vec<f32> = normalized.to_vec1()?;
        results.push(vec);
    }

    Ok(results)
}

// ── tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::cosine_similarity;

    /// Initialize config storage root for tests using a temp directory.
    /// Returns the temp dir path (used as storage root).
    fn init_test_config() -> std::path::PathBuf {
        use std::sync::OnceLock;
        static CONFIG_INIT: OnceLock<tempfile::TempDir> = OnceLock::new();
        let tmp = CONFIG_INIT
            .get_or_init(|| tempfile::TempDir::new().expect("failed to create test temp dir"));
        let root = tmp.path().to_path_buf();
        let _ = crate::config::CONFIG.try_set_storage_root(root.clone());
        root
    }

    /// Set up a storage root pointing to `~/.mahbot` for model-dependent tests.
    /// Uses `try_set_storage_root` so it's a no-op if another test already set it.
    /// Helper to get an embedder for tests.
    ///
    /// Looks for model files in CONFIG storage root first, then falls back to
    /// `$HOME/.mahbot/models`. This ensures model-dependent tests work regardless
    /// of whether the graceful degradation test (which uses a temp dir) ran first.
    /// Returns `None` and skips test if the model files aren't available.
    fn test_embedder() -> Option<Embedder> {
        // Skip if env var is set
        if std::env::var("MAHBOT_SKIP_EMBEDDER_TESTS").is_ok() {
            return None;
        }

        // Collect all candidate models directories (deduplicated).
        let mut candidates = Vec::new();

        // 1. CONFIG storage root (may be a temp dir from graceful degradation test).
        if let Some(root) = crate::config::CONFIG.try_storage_root() {
            candidates.push(root.join("models"));
        }

        // 2. Real home directory cache (always present in dev/CI environments).
        if let Some(home) = std::env::var("HOME").ok().filter(|h| !h.is_empty()) {
            let real = std::path::PathBuf::from(&home)
                .join(".mahbot")
                .join("models");
            if !candidates.contains(&real) {
                candidates.push(real);
            }
        }

        // Try each candidate until we find model files.
        for models_dir in &candidates {
            let model_path = models_dir.join("v5-nano-retrieval-Q4_K_M.gguf");
            let tokenizer_path = models_dir.join("embed_tokenizer.json");

            if model_path.exists() && tokenizer_path.exists() {
                match Embedder::load(&model_path, &tokenizer_path) {
                    Ok(emb) => return Some(emb),
                    Err(e) => {
                        eprintln!("WARNING: Failed to load test embedder: {e}");
                        return None;
                    }
                }
            }
        }

        // No model files found in any candidate directory.
        let last_candidate = candidates.last().map(|p| p.display().to_string());
        eprintln!(
            "WARNING: Model files not found. Looked in: {}. \
             Set MAHBOT_SKIP_EMBEDDER_TESTS=1 to suppress this warning.",
            last_candidate.as_deref().unwrap_or("<none>")
        );
        None
    }

    /// Reset global embedder state for hermetic testing.
    fn reset_global_state() {
        *global_embedder().write().unwrap_poison() = None;
        STATE.store(STATE_UNINIT, Ordering::Release);
        DOWNLOAD_SPAWNED.store(false, Ordering::Release);
    }

    #[test]
    fn test_embedder_graceful_degradation() {
        // Use a temp dir as storage root — no model files there.
        let _root = init_test_config();
        reset_global_state();

        // verify: without model files, embed() returns None
        let result = embed("test", false);
        assert!(
            result.is_none(),
            "embed() should return None when model not available"
        );

        // Verify the global embedder is still empty
        let guard = global_embedder().read().unwrap_poison();
        assert!(guard.is_none(), "global embedder should remain None");
    }

    #[test]
    fn test_embedder_init() {
        let emb = match test_embedder() {
            Some(e) => e,
            None => return, // Skip if no model available
        };
        let v = emb.embed_documents(&["hello world"]).unwrap();
        assert_eq!(v.len(), 1);
        // jina-embeddings-v5 produces 768-dimensional vectors
        assert_eq!(v[0].len(), 768);
        // L2-normalized → unit vector (approximately norm 1)
        let norm: f32 = v[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "expected unit vector, got norm={norm}"
        );
    }

    #[test]
    fn test_embed_documents() {
        let emb = match test_embedder() {
            Some(e) => e,
            None => return,
        };
        let docs = &["first document", "second document about something"];
        let v = emb.embed_documents(docs).unwrap();
        assert_eq!(v.len(), 2);
        for vec in &v {
            assert_eq!(vec.len(), 768);
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "expected unit vector, got norm={norm}"
            );
        }
    }

    #[test]
    fn test_embed_queries() {
        let emb = match test_embedder() {
            Some(e) => e,
            None => return,
        };
        let v = emb.embed_queries(&["what is rust?"]).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].len(), 768);
    }

    #[test]
    fn test_similar_embeddings_are_similar() {
        let emb = match test_embedder() {
            Some(e) => e,
            None => return,
        };
        let v = emb
            .embed_documents(&[
                "rust programming language",
                "the rust programming language",
                "python programming language",
            ])
            .unwrap();
        let sim_01 = cosine_similarity(&v[0], &v[1]);
        let sim_02 = cosine_similarity(&v[0], &v[2]);
        assert!(
            sim_01 > sim_02,
            "rust/rust ({sim_01}) should be more similar than rust/python ({sim_02})"
        );
    }

    #[test]
    fn test_empty_input_fails() {
        let emb = match test_embedder() {
            Some(e) => e,
            None => return,
        };
        let result = emb.embed_documents(&[]);
        assert!(result.is_err(), "empty input should produce an error");
    }
}
