//! Semantic (vector/embedding) search for archived tickets.
//!
//! # Why this exists
//!
//! This module provides a local ONNX-based embedding model that converts ticket
//! descriptions into dense vectors.  These vectors enable **semantic search** —
//! finding archived tickets by *conceptual similarity* (e.g. `"authentication
//! bug"` matching `"login issue"`) — which pure FTS keyword search alone cannot
//! do.
//!
//! The embedding model is loaded **lazily on first `embed()` call** and cached in a
//! global [`OnceLock`]; embedding is computed at **ticket creation time** (to
//! pre-compute a vector for future archived search) and at **search-query time**
//! (to vectorize the query for `SearchArchivedTicketsTool`).  Both paths gracefully
//! degrade: if the model fails to load or inference fails, `embed()` returns `None`
//! and the caller falls back to FTS-only search.
//!
//! While the ONNX runtime and tokenizer dependencies add compilation time, the
//! feature provides real value for the Manager agent when searching historical
//! context — hybrid RRF merging (see [`crate::vector::hybrid_merge`]) combines
//! BM25 keyword hits with cosine-similarity matches, giving better recall than
//! either alone.
//!
//! ## Product decision
//!
//! **Do not propose removing this module or its dependencies without explicit
//! user approval.**  It is a deliberate product decision, not accidental bloat
//! or dead code.

use anyhow::{Context, Result, anyhow};
use ndarray::{Array2, ArrayView2, ArrayView3, Ix3};
use ort::{
    ep,
    session::{InMemorySession, Session},
    value::TensorRef,
};
use std::sync::{Mutex, OnceLock};
use tokenizers::Tokenizer;

// jinaai/jina-embeddings-v5-text-nano-retrieval q4
static TOKENIZER: &[u8] = include_bytes!("../models/embed_tokenizer.json");
static MODEL: &[u8] = include_bytes!("../models/embed.onnx");

static GLOBAL_EMBEDDER: OnceLock<Mutex<Embedder>> = OnceLock::new();

/// Returns a reference to the global singleton [`Embedder`] mutex, initializing it on
/// first access.
///
/// # Panics
///
/// Panics if the ONNX model or tokenizer cannot be loaded.
pub fn global_embedder() -> &'static Mutex<Embedder> {
    GLOBAL_EMBEDDER
        .get_or_init(|| Mutex::new(Embedder::new().expect("Failed to initialize local embedder")))
}

/// Embed a single text using the global embedder singleton.
///
/// `is_query` controls whether the text is embedded as a query (prefixed with
/// `"Query: "`) or as a document (prefixed with `"Document: "`), as required
/// by the embedding model's training.
///
/// Returns `None` if the embedder's ONNX session encounters an error (e.g.
/// model panics due to corrupted ONNX state), or if the mutex is poisoned.
#[must_use]
pub fn embed(text: &str, is_query: bool) -> Option<Vec<f32>> {
    let mut guard = global_embedder().lock().ok()?;
    let v = if is_query {
        guard.embed_queries(&[text]).ok()?
    } else {
        guard.embed_documents(&[text]).ok()?
    };
    v.into_iter().next()
}

pub struct Embedder {
    tokenizer: Tokenizer,
    session: InMemorySession<'static>,
    pad_id: i64,
}

impl Embedder {
    /// # Panics
    ///
    /// Panics if the `<|pad|>` special token is not found in the tokenizer.
    pub fn new() -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(TOKENIZER).map_err(|e| anyhow!(e))?;
        let pad_id = i64::from(tokenizer.token_to_id("<|pad|>").unwrap());
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_memory_pattern(false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_memory_directly(MODEL)?;
        let mut s = Self {
            tokenizer,
            session,
            pad_id,
        };
        // Warm-up: validate the model produces non-empty embeddings.
        let v = s.embed_documents(&["."])?;
        anyhow::ensure!(
            !v.is_empty() && !v[0].is_empty(),
            "embedder warm-up produced empty output"
        );
        Ok(s)
    }

    pub fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_prefixed("Query: ", texts)
    }

    pub fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_prefixed("Document: ", texts)
    }

    fn embed_prefixed(&mut self, prefix: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let encodings = texts
            .iter()
            .map(|t| self.tokenizer.encode(format!("{prefix}{t}"), true))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| anyhow!(e))?;

        let max_len = encodings
            .iter()
            .map(|e| e.len().min(8192))
            .max()
            .context("empty input")?;

        let mut input_ids = Array2::<i64>::from_elem((encodings.len(), max_len), self.pad_id);
        let mut attention_mask = Array2::<i64>::zeros((encodings.len(), max_len));

        for (row, e) in encodings.iter().enumerate() {
            for (col, (&id, &mask)) in e
                .get_ids()
                .iter()
                .zip(e.get_attention_mask().iter())
                .take(8192)
                .enumerate()
            {
                input_ids[(row, col)] = i64::from(id);
                attention_mask[(row, col)] = i64::from(mask);
            }
        }

        let outputs = self.session.run(ort::inputs![
            TensorRef::from_array_view(&input_ids)?,
            TensorRef::from_array_view(&attention_mask)?
        ])?;

        let hidden = outputs[0]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix3>()?;

        Ok(last_token_pool_and_l2_normalize(
            &hidden,
            &attention_mask.view(),
        ))
    }
}

fn last_token_pool_and_l2_normalize(
    hidden: &ArrayView3<'_, f32>,
    mask: &ArrayView2<'_, i64>,
) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(hidden.dim().0);

    for (h, m) in hidden.outer_iter().zip(mask.outer_iter()) {
        let last = m
            .iter()
            .map(|&x| usize::try_from(x).unwrap())
            .sum::<usize>()
            - 1;
        let mut v = h.row(last).to_vec();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);

        for x in &mut v {
            *x /= norm;
        }

        out.push(v);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::cosine_similarity;

    #[test]
    fn test_embedder_init() {
        // new() already runs a warm-up — this validates the model loads and
        // produces non-empty output.
        let mut emb = Embedder::new().expect("embedder should init");
        let v = emb.embed_documents(&["hello world"]).unwrap();
        assert_eq!(v.len(), 1);
        // jina-embeddings-v5 produces 768-dimensional vectors
        assert_eq!(v[0].len(), 768);
    }

    #[test]
    fn test_embed_documents() {
        let mut emb = Embedder::new().unwrap();
        let docs = &["first document", "second document about something"];
        let v = emb.embed_documents(docs).unwrap();
        assert_eq!(v.len(), 2);
        for vec in &v {
            assert_eq!(vec.len(), 768);
            // L2-normalized → unit vector (approximately norm 1)
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "expected unit vector, got norm={norm}"
            );
        }
    }

    #[test]
    fn test_embed_queries() {
        let mut emb = Embedder::new().unwrap();
        let v = emb.embed_queries(&["what is rust?"]).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].len(), 768);
    }

    #[test]
    fn test_similar_embeddings_are_similar() {
        let mut emb = Embedder::new().unwrap();
        let v = emb
            .embed_documents(&[
                "rust programming language",
                "the rust programming language",
                "python programming language",
            ])
            .unwrap();

        let sim_01 = cosine_similarity(&v[0], &v[1]);
        let sim_02 = cosine_similarity(&v[0], &v[2]);
        // Same-language texts should be more similar than different languages
        assert!(
            sim_01 > sim_02,
            "rust/rust ({sim_01}) should be more similar than rust/python ({sim_02})"
        );
    }

    #[test]
    fn test_empty_input_fails() {
        let mut emb = Embedder::new().unwrap();
        let result = emb.embed_documents(&[]);
        assert!(result.is_err(), "empty input should produce an error");
    }

    #[test]
    fn test_global_embedder_singleton() {
        // Pull from the global singleton — should produce valid embeddings
        let mut guard = global_embedder().lock().unwrap();
        let v = guard.embed_documents(&["singleton test"]).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].len(), 768);
    }
}
