//! Vector operations — cosine similarity, hybrid merge, serialization.
//!
//! Used by [`crate::tools::SearchArchivedTicketsTool`] for hybrid FTS+semantic
//! search.
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
    // as_chunks::<4>() yields exactly-4-byte chunks (trailing partial bytes are
    // ignored — matching the old chunks_exact behavior), so no try_into dance.
    let (chunks, _remainder) = bytes.as_chunks::<4>();
    chunks.iter().map(|arr| f32::from_le_bytes(*arr)).collect()
}

/// Reciprocal Rank Fusion smoothing constant.
/// Higher values reduce the influence of top-ranked results, making the
/// fusion less sensitive to score-scale differences across ranking sources.
pub(crate) const RRF_K: f32 = 60.0;

/// Apply RRF scoring to a ranked list and accumulate into `scores`.
///
/// Encapsulates the rank-to-reciprocal-score mapping so it can be applied
/// independently to each search source before summing across sources.
#[expect(clippy::cast_precision_loss)]
fn accumulate_rrf(scores: &mut HashMap<String, f32>, mut results: Vec<(String, f32)>) {
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (rank, (id, _)) in results.into_iter().enumerate() {
        let rrf = 1.0 / (RRF_K + (rank + 1) as f32);
        *scores.entry(id).or_insert(0.0) += rrf;
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
/// Results are sorted by RRF score descending, with deterministic tiebreaking
/// by id (lexicographic order).
#[must_use]
pub(crate) fn hybrid_merge(
    vector_results: Vec<(String, f32)>,  // (id, cosine_similarity)
    keyword_results: Vec<(String, f32)>, // (id, bm25_score)
) -> Vec<String> {
    let mut rrf_scores: HashMap<String, f32> = HashMap::new();

    accumulate_rrf(&mut rrf_scores, vector_results);
    accumulate_rrf(&mut rrf_scores, keyword_results);

    // Collect into (id, score) pairs, sorted by score descending,
    // with deterministic tiebreaking by id (lexicographic order).
    let mut results: Vec<(String, f32)> = rrf_scores.into_iter().collect();
    results.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    results.into_iter().map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_all_cases() {
        struct Case {
            name: &'static str,
            a: &'static [f32],
            b: &'static [f32],
            expected: f32,
        }

        let cases = [
            Case {
                name: "identical vectors",
                a: &[1.0, 2.0, 3.0],
                b: &[1.0, 2.0, 3.0],
                expected: 1.0,
            },
            Case {
                name: "orthogonal vectors",
                a: &[1.0, 0.0, 0.0],
                b: &[0.0, 1.0, 0.0],
                expected: 0.0,
            },
            Case {
                name: "empty first vector",
                a: &[],
                b: &[1.0, 2.0],
                expected: 0.0,
            },
            Case {
                name: "different lengths",
                a: &[1.0, 2.0],
                b: &[1.0, 2.0, 3.0],
                expected: 0.0,
            },
        ];

        for case in &cases {
            let result = cosine_similarity(case.a, case.b);
            assert!(
                (result - case.expected).abs() < 1e-6,
                "case: {} — expected {}, got {}",
                case.name,
                case.expected,
                result,
            );
        }
    }

    #[test]
    fn vec_bytes_roundtrip() {
        let v = vec![1.0_f32, -2.5, 0.0];
        let bytes = vec_to_bytes(&v);
        assert_eq!(bytes.len(), v.len() * 4);
        let v2 = bytes_to_vec(&bytes);
        for (a, b) in v.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn hybrid_merge_all_cases() {
        struct Case {
            name: &'static str,
            vector: Vec<(String, f32)>,
            keyword: Vec<(String, f32)>,
            expected_len: usize,
            expected_first: Option<&'static str>,
        }

        let cases = [
            Case {
                // B appears in both lists, so RRF additively boosts it above
                // items that appear in only one source.
                name: "combines results from both sources — B boosted by RRF",
                vector: vec![("A".into(), 0.9), ("B".into(), 0.7)],
                keyword: vec![("B".into(), 1.2), ("C".into(), 0.5)],
                expected_len: 3,
                expected_first: Some("B"),
            },
            Case {
                name: "empty inputs yield empty output",
                vector: vec![],
                keyword: vec![],
                expected_len: 0,
                expected_first: None,
            },
        ];

        for case in cases {
            let merged = hybrid_merge(case.vector, case.keyword);
            assert_eq!(
                merged.len(),
                case.expected_len,
                "case: {} — expected len {}",
                case.name,
                case.expected_len,
            );
            if let Some(first) = case.expected_first {
                assert!(!merged.is_empty(), "case: {}", case.name);
                assert_eq!(merged[0], first, "case: {}", case.name);
            }
        }
    }
}
