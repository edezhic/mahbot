//! Client for the OpenRouter video-models catalog (`GET {base}/videos/models`).
//!
//! Mirrors the image-side catalog: fetched once (single-flight), keyed by the
//! configured provider endpoint, reused for 24h, with a short negative-cache
//! backoff on failure and fail-open semantics ([`get_catalog`] returns `None`).
//! The cache machinery lives in [`crate::tools::catalog_cache`], shared with
//! the image catalog. The video response shape differs from the image one —
//! flat per-model fields with high null variability (aleph-2 declares no
//! resolutions/durations/sizes) — so it gets its own tolerant parser. Models
//! are looked up by `id`, never `canonical_slug` (they differ for every live
//! model).

use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Parsed per-model video-generation capabilities. Every field is optional —
/// the live catalog declares none of them for some models (aleph-2).
#[derive(Debug, Clone, Default)]
pub(crate) struct VideoModelInfo {
    pub(crate) resolutions: Option<Vec<String>>,
    pub(crate) aspect_ratios: Option<Vec<String>>,
    pub(crate) durations: Option<Vec<i64>>,
    pub(crate) sizes: Option<Vec<String>>,
    pub(crate) frame_images: Option<Vec<String>>,
    pub(crate) generate_audio: Option<bool>,
    pub(crate) seed: Option<bool>,
}

/// The full parsed catalog: model id → capabilities.
#[derive(Debug, Default)]
pub(crate) struct VideoCatalog {
    models: HashMap<String, VideoModelInfo>,
}

impl VideoCatalog {
    #[must_use]
    pub(crate) fn find(&self, model: &str) -> Option<&VideoModelInfo> {
        self.models.get(model)
    }
}

/// Parse a catalog response body; per-entry tolerance in [`parse_model`],
/// envelope error contract in [`crate::tools::catalog_cache::parse_envelope`].
pub(crate) fn parse_catalog(body: &Value) -> anyhow::Result<VideoCatalog> {
    Ok(VideoCatalog {
        models: crate::tools::catalog_cache::parse_envelope(
            body,
            "Video models catalog",
            parse_model,
        )?,
    })
}

/// Parse one catalog entry; tolerates per-model nulls and unknown fields.
fn parse_model(entry: &Value) -> Option<(String, VideoModelInfo)> {
    // Lookup key is always `id`, never `canonical_slug` — they differ for
    // every live model.
    let id = entry.get("id").and_then(Value::as_str)?.to_string();
    let info = VideoModelInfo {
        resolutions: parse_str_array(entry.get("supported_resolutions")),
        aspect_ratios: parse_str_array(entry.get("supported_aspect_ratios")),
        durations: parse_i64_array(entry.get("supported_durations")),
        sizes: parse_str_array(entry.get("supported_sizes")),
        frame_images: parse_str_array(entry.get("supported_frame_images")),
        generate_audio: entry.get("generate_audio").and_then(Value::as_bool),
        seed: entry.get("seed").and_then(Value::as_bool),
    };
    Some((id, info))
}

/// Parse a string-array field; `null`/missing stays `None`.
fn parse_str_array(v: Option<&Value>) -> Option<Vec<String>> {
    v.and_then(Value::as_array).map(|arr| {
        arr.iter()
            .filter_map(Value::as_str)
            .map(String::from)
            .collect()
    })
}

/// Parse an integer-array field; `null`/missing stays `None`.
fn parse_i64_array(v: Option<&Value>) -> Option<Vec<i64>> {
    v.and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_i64).collect())
}

// ── Cached fetch (single-flight, endpoint-keyed, fail-open) ─────────

static CATALOG: crate::tools::catalog_cache::Catalog<VideoCatalog> =
    crate::tools::catalog_cache::Catalog::new(
        "/videos/models",
        "Video models catalog",
        parse_catalog,
    );

/// Return the cached catalog for the currently configured provider endpoint.
pub(crate) async fn get_catalog() -> Option<Arc<VideoCatalog>> {
    let endpoint = crate::config::CONFIG.provider_endpoint();
    get_catalog_for_endpoint(&endpoint).await
}

/// Return the cached catalog for `endpoint` if fresh, otherwise fetch it
/// (single-flight). Returns `None` (fail-open) when the catalog is
/// unavailable — retried after a short negative-cache backoff.
pub(crate) async fn get_catalog_for_endpoint(endpoint: &str) -> Option<Arc<VideoCatalog>> {
    CATALOG.get(endpoint).await
}

/// Test-only: seed the shared cache (no network).
#[cfg(test)]
pub(crate) fn seed_cache(catalog: Option<Arc<VideoCatalog>>) {
    let endpoint = crate::providers::ensure_base_url(&crate::config::CONFIG.provider_endpoint());
    CATALOG.seed(&endpoint, catalog);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_and_query_video_catalog() {
        let catalog = parse_catalog(&json!({
            "data": [
                {
                    "id": "minimax/hailuo-3",
                    "canonical_slug": "minimax/hailuo-03-20260730",
                    "supported_resolutions": ["2K"],
                    "supported_aspect_ratios": ["16:9", "9:16", "1:1"],
                    "supported_durations": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    "supported_sizes": null,
                    "supported_frame_images": ["first_frame", "last_frame"],
                    "generate_audio": true,
                    "seed": false,
                    "pricing_skus": { "duration_seconds": { "type": "second" } }
                },
                {
                    "id": "runway/aleph-2",
                    "canonical_slug": "runway/aleph-2-20260729",
                    "supported_resolutions": null,
                    "supported_durations": null,
                    "generate_audio": false,
                    "seed": true
                },
                { "canonical_slug": "no-id-model" }
            ]
        }))
        .expect("valid fixture");

        let hailuo = catalog.find("minimax/hailuo-3").expect("found by id");
        assert_eq!(hailuo.resolutions.as_deref(), Some(&["2K".to_string()][..]));
        assert_eq!(hailuo.durations.as_deref().map(<[i64]>::len), Some(11));
        assert_eq!(hailuo.generate_audio, Some(true));
        assert_eq!(hailuo.seed, Some(false));

        // Null-heavy model keeps every optional field absent.
        let aleph = catalog.find("runway/aleph-2").expect("found by id");
        assert_eq!(aleph.resolutions, None);
        assert_eq!(aleph.durations, None);
        assert_eq!(aleph.generate_audio, Some(false));

        // Entries without an `id` are skipped; canonical_slug is never the key.
        assert!(catalog.find("runway/aleph-2-20260729").is_none());
        assert!(catalog.find("no-id-model").is_none());
    }

    #[test]
    fn test_parse_catalog_tolerates_unknown_shapes() {
        let catalog = parse_catalog(&json!({
            "data": [
                { "id": "m1", "supported_durations": ["not-an-int"], "seed": "maybe" },
                { "id": "m2", "supported_resolutions": [42], "supported_aspect_ratios": null }
            ]
        }))
        .expect("tolerated");
        let m1 = catalog.find("m1").expect("m1 present");
        assert_eq!(m1.durations, Some(vec![]));
        assert_eq!(m1.seed, None);
        let m2 = catalog.find("m2").expect("m2 present");
        assert_eq!(m2.resolutions, Some(vec![]));
        assert_eq!(m2.aspect_ratios, None);
    }
}
