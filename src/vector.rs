//! Vector operations — cosine similarity, hybrid merge, serialization.
//!
//! Extracted from the former memory module and now used by
//! [`crate::tools::SearchArchivedTicketsTool`] for hybrid FTS+semantic search.
//! [`crate::embedder::Embedder`] produces embeddings that are stored as
//! blobs in the tickets table and deserialized via [`bytes_to_vec`] during search.

use std::collections::HashMap;

/// Cosine similarity between two vectors. Returns 0.0–1.0.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if !denom.is_finite() || denom < f32::EPSILON {
        return 0.0;
    }

    let raw = dot / denom;
    if !raw.is_finite() {
        return 0.0;
    }

    // Clamp to [0, 1] — embeddings are typically positive
    raw.clamp(0.0, 1.0)
}

/// Serialize f32 vector to bytes (little-endian)
#[must_use]
pub fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

/// Deserialize bytes to f32 vector (little-endian)
#[must_use]
pub fn bytes_to_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// Reciprocal Rank Fusion smoothing constant.
/// Higher values reduce the influence of top-ranked results, making the
/// fusion less sensitive to score-scale differences across ranking sources.
pub(crate) const RRF_K: f32 = 60.0;

/// A scored result for hybrid merging
#[derive(Debug, Clone)]
pub struct ScoredResult {
    pub id: String,
    pub final_score: f32,
}

/// Apply RRF scoring to a ranked list and accumulate into `scores`.
///
/// Encapsulates the rank-to-reciprocal-score mapping so it can be applied
/// independently to each search source before summing across sources.
#[allow(clippy::cast_precision_loss)]
fn accumulate_rrf(scores: &mut HashMap<String, f32>, results: &[(String, f32)]) {
    let mut sorted: Vec<_> = results.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (rank, (id, _)) in sorted.iter().enumerate() {
        let rrf = 1.0 / (RRF_K + (rank + 1) as f32);
        *scores.entry(id.clone()).or_insert(0.0) += rrf;
    }
}

/// Hybrid merge: combine vector and keyword results using Reciprocal Rank Fusion (RRF).
///
/// RRF is robust to score-scale differences between cosine similarity (bounded [0, 1])
/// and BM25 scores (unbounded), making it more suitable than simple averaging.
/// Each source list is ranked independently by score, and items receive a reciprocal
/// score `1 / (K + rank)` that is summed across sources. Items appearing in both
/// sources receive additive contributions, naturally boosting their final rank.
///
/// Results are sorted by final score descending, with deterministic tiebreaking
/// by ID (lexicographic order).
#[must_use]
pub fn hybrid_merge(
    vector_results: &[(String, f32)],  // (id, cosine_similarity)
    keyword_results: &[(String, f32)], // (id, bm25_score)
) -> Vec<ScoredResult> {
    let mut rrf_scores: HashMap<String, f32> = HashMap::new();

    accumulate_rrf(&mut rrf_scores, vector_results);
    accumulate_rrf(&mut rrf_scores, keyword_results);

    // Build results sorted by final score descending
    let mut results: Vec<ScoredResult> = rrf_scores
        .into_iter()
        .map(|(id, score)| ScoredResult {
            id,
            final_score: score,
        })
        .collect();

    results.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    results
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::approx_constant,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn test_vec_bytes_roundtrip() {
        let v = vec![1.0_f32, -2.5, 0.0];
        let bytes = vec_to_bytes(&v);
        assert_eq!(bytes.len(), v.len() * 4);
        let v2 = bytes_to_vec(&bytes);
        for (a, b) in v.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_hybrid_merge_combines_results() {
        let vector = vec![("A".into(), 0.9), ("B".into(), 0.7)];
        let keyword = vec![("B".into(), 1.2), ("C".into(), 0.5)];
        let merged = hybrid_merge(&vector, &keyword);
        assert!(!merged.is_empty());
        // B appears in both → highest score
        assert_eq!(merged[0].id, "B");
    }

    #[test]
    fn test_hybrid_merge_empty_inputs() {
        let vector: Vec<(String, f32)> = vec![];
        let keyword: Vec<(String, f32)> = vec![];
        let merged = hybrid_merge(&vector, &keyword);
        assert!(merged.is_empty());
    }
}
