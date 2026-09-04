//! Consolidated image & video model-catalog cluster.
//!
//! The generic single-flight, endpoint-keyed, TTL-cached fetch machinery
//! (`Catalog`) is shared by the image and video catalog clients so their
//! cache/backoff behavior cannot drift. That generic machinery — along with
//! the shared envelope parser (`parse_envelope`), so the `{ "data": [...] }`
//! contract is single-source — now lives in [`crate::util::catalog_cache`].
//!
//! Caching: fetched once (single-flight), keyed by the configured provider
//! endpoint, and reused for `CATALOG_TTL`. A failed fetch — including a
//! timeout — is stored as a short-lived negative cache
//! (`CATALOG_RETRY_BACKOFF`) and degrades to fail-open: `Catalog::get`
//! returns `None` and callers proceed without capability data.
//!
//! The nested `image` and `video` submodules keep the twin clients'
//! namespaces separate, since each declares the same helper names
//! (`parse_catalog`, `get_catalog`, `seed_cache`, a local `CATALOG`).

/// Client for the OpenRouter image-models catalog (`GET {base}/images/models`).
///
/// The catalog declares per-model image-generation parameters (parameter
/// names, enums, ranges, output modalities). Every image-gen capability and
/// parameter decision is driven by this data — never by model-name branches.
///
/// Caching: fetched once (single-flight), keyed by the configured provider
/// endpoint, and reused for 24h. A failed fetch — including a timeout — is
/// stored as a short-lived negative cache (1-min backoff) and degrades to
/// fail-open: `get_catalog` returns `None` and callers send minimal
/// parameters instead of rejecting generation. The cache machinery lives in
/// [`crate::util::catalog_cache`], shared with the video catalog.
pub mod image {
    use crate::util::catalog_cache::{Catalog, parse_envelope};
    use serde_json::Value;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Declared constraint for one image-generation parameter.
    #[derive(Debug, Clone)]
    pub(crate) enum ParameterConstraint {
        Enum(Vec<String>),
        Range {
            max: i64,
        },
        Boolean,
        /// Unknown constraint shape — tolerated so a catalog evolution never breaks parsing.
        Unknown,
    }

    /// Whether a model declares image output in `architecture.output_modalities`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub(crate) enum ImageOutput {
        /// `image` is among the declared output modalities.
        Declared,
        /// Modalities are declared but exclude `image` (explicitly text-only).
        TextOnly,
        /// Modalities are missing or empty — treated as capable (fail-open on shape drift).
        #[default]
        Unknown,
    }

    /// Parsed per-model image-generation capabilities.
    #[derive(Debug, Clone, Default)]
    pub(crate) struct ImageModelInfo {
        /// Parameter names declared by the catalog (keys of `supported_parameters`).
        pub(crate) supported_parameters: HashMap<String, ParameterConstraint>,
        /// Declared output modalities, mapped to image-generation support.
        pub(crate) image_output: ImageOutput,
    }

    impl ImageModelInfo {
        /// Whether the catalog declares `param` for this model.
        #[must_use]
        pub(crate) fn declares(&self, param: &str) -> bool {
            self.supported_parameters.contains_key(param)
        }

        /// Whether `param` is a declared enum that includes `value`.
        #[must_use]
        pub(crate) fn enum_contains(&self, param: &str, value: &str) -> bool {
            matches!(
                self.supported_parameters.get(param),
                Some(ParameterConstraint::Enum(values)) if values.iter().any(|v| v == value)
            )
        }

        /// The declared `param` range max (e.g. `input_references` cap).
        #[must_use]
        pub(crate) fn range_max(&self, param: &str) -> Option<i64> {
            match self.supported_parameters.get(param) {
                Some(ParameterConstraint::Range { max }) => Some(*max),
                _ => None,
            }
        }
    }

    /// The full parsed catalog: model id → capabilities.
    #[derive(Debug, Default)]
    pub(crate) struct ImageCatalog {
        models: HashMap<String, ImageModelInfo>,
    }

    impl ImageCatalog {
        #[must_use]
        pub(crate) fn find(&self, model: &str) -> Option<&ImageModelInfo> {
            self.models.get(model)
        }
    }

    /// Parse a catalog response body; per-entry tolerance in [`parse_model`],
    /// envelope error contract in [`crate::util::catalog_cache::parse_envelope`].
    pub(crate) fn parse_catalog(body: &Value) -> anyhow::Result<ImageCatalog> {
        Ok(ImageCatalog {
            models: parse_envelope(body, "Image models catalog", parse_model)?,
        })
    }

    /// Parse one catalog entry; tolerates unknown model entries (skipped), missing
    /// fields, and unknown parameter shapes.
    fn parse_model(entry: &Value) -> Option<(String, ImageModelInfo)> {
        let id = entry.get("id").and_then(Value::as_str)?.to_string();
        let image_output = match entry["architecture"]["output_modalities"].as_array() {
            Some(arr) if arr.iter().any(|m| m.as_str() == Some("image")) => ImageOutput::Declared,
            Some(arr) if !arr.is_empty() => ImageOutput::TextOnly,
            _ => ImageOutput::Unknown,
        };
        let mut info = ImageModelInfo {
            image_output,
            ..ImageModelInfo::default()
        };
        if let Some(params) = entry.get("supported_parameters").and_then(Value::as_object) {
            for (name, constraint) in params {
                info.supported_parameters
                    .insert(name.clone(), parse_constraint(constraint));
            }
        }
        Some((id, info))
    }

    fn parse_constraint(v: &Value) -> ParameterConstraint {
        match v.get("type").and_then(Value::as_str) {
            Some("enum") => ParameterConstraint::Enum(
                v.get("values")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(Value::as_str)
                            .map(String::from)
                            .collect()
                    })
                    .unwrap_or_default(),
            ),
            Some("range") => ParameterConstraint::Range {
                max: v.get("max").and_then(Value::as_i64).unwrap_or(0),
            },
            Some("boolean") => ParameterConstraint::Boolean,
            _ => ParameterConstraint::Unknown,
        }
    }

    // ── Cached fetch (single-flight, endpoint-keyed, fail-open) ─────────

    static CATALOG: Catalog<ImageCatalog> =
        Catalog::new("/images/models", "Image models catalog", parse_catalog);

    /// Return the cached catalog for the default OpenRouter endpoint — media
    /// always targets OpenRouter, never a custom chat endpoint.
    pub(crate) async fn get_catalog() -> Option<Arc<ImageCatalog>> {
        let endpoint = crate::config::DEFAULT_PROVIDER_ENDPOINT;
        get_catalog_for_endpoint(endpoint).await
    }

    /// Return the cached catalog for `endpoint` if fresh, otherwise fetch it
    /// (single-flight). Returns `None` (fail-open) when the catalog is
    /// unavailable — retried after a short negative-cache backoff.
    pub(crate) async fn get_catalog_for_endpoint(endpoint: &str) -> Option<Arc<ImageCatalog>> {
        CATALOG.get(endpoint).await
    }

    /// Test-only: seed the shared cache (no network).
    #[cfg(test)]
    pub(crate) fn seed_cache(catalog: Option<Arc<ImageCatalog>>) {
        // Media always targets OpenRouter — seed the default key.
        let endpoint = crate::providers::ensure_base_url(crate::config::DEFAULT_PROVIDER_ENDPOINT);
        CATALOG.seed(&endpoint, catalog);
    }

    /// Reject a catalog-listed model that explicitly declares text-only output.
    /// (Shape drift — missing/empty modalities — is tolerated as capable.)
    fn reject_text_only(model: &str, info: &ImageModelInfo) -> anyhow::Result<()> {
        if info.image_output == ImageOutput::TextOnly {
            anyhow::bail!(
                "Model `{model}` cannot generate images: the OpenRouter image-models catalog \
                 does not list image output for it. Set an image-capable model in \
                 Settings → Image Generation and retry."
            );
        }
        Ok(())
    }

    /// Shared image-model capability resolver — the single validation entry
    /// point for all three consumers (GUI model picker, Telegram model setter,
    /// and the generation-time image-gen check).
    ///
    /// Resolution order: a hit against the cached catalog yields the capability
    /// verdict immediately (no network). A membership miss forces one atomic
    /// catalog refresh ([`Catalog::refresh_for_miss`], cooldown-bounded) and
    /// re-checks, so a model added to OpenRouter after the cached snapshot was
    /// taken is accepted instead of being rejected for up to [`CATALOG_TTL`].
    /// Outcomes:
    /// - `Ok(Some(info))` — present and not text-only; `info` feeds request building;
    /// - `Ok(None)` — catalog unavailable (fetch failed/negative backoff) → fail-open;
    /// - `Err` — confirmed absent from a freshly fetched catalog, or text-only.
    pub(crate) async fn resolve_image_model(
        endpoint: &str,
        model: &str,
    ) -> anyhow::Result<Option<ImageModelInfo>> {
        resolve_capability_with(&CATALOG, endpoint, model).await
    }

    /// Resolver core, generic over the catalog cache so tests can drive a
    /// scripted fetch on a local instance (the real `CATALOG` static is global).
    async fn resolve_capability_with(
        catalog: &Catalog<ImageCatalog>,
        endpoint: &str,
        model: &str,
    ) -> anyhow::Result<Option<ImageModelInfo>> {
        let base = crate::providers::ensure_base_url(endpoint);
        let Some(snapshot) = catalog.get(&base).await else {
            tracing::warn!(
                "Image-models catalog unavailable — skipping image-model validation (fail-open)"
            );
            return Ok(None);
        };
        if let Some(info) = snapshot.find(model) {
            return reject_text_only(model, info).map(|()| Some(info.clone()));
        }
        // Membership miss: one forced refresh + re-check. Only absence from a
        // freshly fetched catalog rejects; a failed refresh degrades to fail-open.
        let Some(snapshot) = catalog.refresh_for_miss(&base).await else {
            return Ok(None);
        };
        match snapshot.find(model) {
            Some(info) => reject_text_only(model, info).map(|()| Some(info.clone())),
            None => anyhow::bail!(
                "Model `{model}` cannot generate images: it is not in the OpenRouter \
                 image-models catalog. Set an image-capable model in Settings → \
                 Image Generation and retry."
            ),
        }
    }

    /// Write-time validation that `model` can generate images, using the catalog
    /// when available. Fail-open (a catalog outage passes the check) now lives
    /// in [`resolve_image_model`] — this validates against the currently
    /// committed endpoint.
    ///
    /// Used by the settings save path, the GUI model picker, and the Telegram
    /// model commands.
    ///
    /// # Errors
    ///
    /// - Catalog available and the model is absent from it or text-only.
    pub async fn validate_image_model(model: &str) -> anyhow::Result<()> {
        // Media always targets OpenRouter — validate against the
        // default endpoint, never a custom chat endpoint.
        let endpoint = crate::config::DEFAULT_PROVIDER_ENDPOINT;
        validate_image_model_for_endpoint(model, endpoint).await
    }

    /// Like [`validate_image_model`], but against the catalog for the explicit
    /// `endpoint` rather than the currently committed one — the settings page
    /// uses this so a model edit is checked against the endpoint it will run
    /// under (the model picker validates additions against the staged endpoint;
    /// the active-model persist against the committed one).
    pub(crate) async fn validate_image_model_for_endpoint(
        model: &str,
        endpoint: &str,
    ) -> anyhow::Result<()> {
        resolve_image_model(endpoint, model).await.map(|_| ())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::util::catalog_cache::FetchFuture;
        use serde_json::json;
        use std::collections::VecDeque;
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[test]
        fn test_parse_and_query_catalog() {
            let catalog = parse_catalog(&json!({
                "data": [
                    {
                        "id": "qwen/qwen-image-3-pro",
                        "architecture": { "output_modalities": ["image"] },
                        "supported_parameters": {
                            "resolution": { "type": "enum", "values": ["1K", "2K"] },
                            "aspect_ratio": { "type": "enum", "values": ["1:1", "9:16"] },
                            "n": { "type": "range", "min": 1, "max": 6 },
                            "input_references": { "type": "range", "min": 0, "max": 4 },
                            "seed": { "type": "boolean" }
                        }
                    },
                    {
                        "id": "text-only/model",
                        "architecture": { "output_modalities": ["text"] },
                        "supported_parameters": {}
                    }
                ]
            }))
            .expect("valid fixture");

            let image = catalog.find("qwen/qwen-image-3-pro").expect("found");
            assert_eq!(image.image_output, ImageOutput::Declared);
            assert!(image.declares("resolution"));
            assert!(!image.declares("quality"));
            assert!(image.enum_contains("aspect_ratio", "9:16"));
            assert!(!image.enum_contains("aspect_ratio", "auto"));
            assert_eq!(image.range_max("input_references"), Some(4));
            assert_eq!(image.range_max("seed"), None);

            assert_eq!(
                catalog.find("text-only/model").expect("found").image_output,
                ImageOutput::TextOnly
            );
            assert!(catalog.find("unknown/model").is_none());
        }

        #[test]
        fn test_parse_catalog_tolerates_unknown_shapes() {
            // Unknown parameter types and entries without ids must not break parsing.
            let catalog = parse_catalog(&json!({
                "data": [
                    { "id": "m1", "architecture": { "output_modalities": ["image"] },
                      "supported_parameters": { "future_param": { "type": "weird" } } },
                    { "name": "no id" }
                ]
            }))
            .expect("tolerated");
            assert!(catalog.find("m1").is_some());
        }

        // ── Resolver tests: scripted fetch on a local Catalog (hermetic) ──

        static SCRIPT: Mutex<VecDeque<anyhow::Result<serde_json::Value>>> =
            Mutex::new(VecDeque::new());
        static FETCHES: AtomicUsize = AtomicUsize::new(0);

        fn scripted_fetch(_url: &str, _label: &str) -> FetchFuture {
            FETCHES.fetch_add(1, Ordering::SeqCst);
            let next = SCRIPT.lock().expect("script lock").pop_front();
            Box::pin(
                async move { next.unwrap_or_else(|| Err(anyhow::anyhow!("script exhausted"))) },
            )
        }

        fn catalog_with_model(id: &str) -> serde_json::Value {
            json!({ "data": [ { "id": id, "architecture": { "output_modalities": ["image"] } } ] })
        }

        fn resolver_catalog() -> Catalog<ImageCatalog> {
            Catalog::new("/images/models", "Image models catalog", parse_catalog)
                .with_fetch(scripted_fetch)
        }

        fn reset_script(entries: Vec<anyhow::Result<serde_json::Value>>) {
            *SCRIPT.lock().expect("script lock") = entries.into();
            FETCHES.store(0, Ordering::SeqCst);
        }

        #[tokio::test]
        #[serial_test::serial(image_resolver)]
        async fn resolver_hit_path_performs_no_fetch() {
            let endpoint = crate::config::DEFAULT_PROVIDER_ENDPOINT;
            let base = crate::providers::ensure_base_url(endpoint);
            reset_script(vec![]);

            // Capable model present → Ok(Some), no fetch.
            let catalog = resolver_catalog();
            catalog.seed(
                &base,
                Some(Arc::new(
                    parse_catalog(&catalog_with_model("a/model")).expect("capable fixture"),
                )),
            );
            let info = resolve_capability_with(&catalog, endpoint, "a/model")
                .await
                .expect("resolved");
            assert!(info.is_some());

            // Text-only present → Err with the text-only fragment, no fetch.
            let catalog = resolver_catalog();
            catalog.seed(
                &base,
                Some(Arc::new(
                    parse_catalog(&json!({
                        "data": [ { "id": "text/model",
                                    "architecture": { "output_modalities": ["text"] } } ]
                    }))
                    .expect("text-only fixture"),
                )),
            );
            let err = resolve_capability_with(&catalog, endpoint, "text/model")
                .await
                .unwrap_err();
            assert!(err.to_string().contains("does not list image output"));

            // Shape drift (missing architecture) → Ok(Some), no fetch.
            let catalog = resolver_catalog();
            catalog.seed(
                &base,
                Some(Arc::new(
                    parse_catalog(&json!({ "data": [ { "id": "drift/model" } ] }))
                        .expect("drift fixture"),
                )),
            );
            let info = resolve_capability_with(&catalog, endpoint, "drift/model")
                .await
                .expect("drift tolerant");
            assert!(info.is_some());

            assert_eq!(FETCHES.load(Ordering::SeqCst), 0);
        }

        #[tokio::test]
        #[serial_test::serial(image_resolver)]
        async fn resolver_miss_refresh_accepts_new_model() {
            let endpoint = crate::config::DEFAULT_PROVIDER_ENDPOINT;
            let base = crate::providers::ensure_base_url(endpoint);
            let catalog = resolver_catalog();
            catalog.seed(
                &base,
                Some(Arc::new(
                    parse_catalog(&catalog_with_model("other/model")).expect("seed"),
                )),
            );
            reset_script(vec![Ok(catalog_with_model("brand/new-model"))]);

            let info = resolve_capability_with(&catalog, endpoint, "brand/new-model")
                .await
                .expect("refreshed");
            assert!(info.is_some());
            assert_eq!(FETCHES.load(Ordering::SeqCst), 1);

            let info = resolve_capability_with(&catalog, endpoint, "brand/new-model")
                .await
                .expect("now cached");
            assert!(info.is_some());
            assert_eq!(FETCHES.load(Ordering::SeqCst), 1);
        }

        #[tokio::test]
        #[serial_test::serial(image_resolver)]
        async fn resolver_fresh_absence_rejects() {
            let endpoint = crate::config::DEFAULT_PROVIDER_ENDPOINT;
            let base = crate::providers::ensure_base_url(endpoint);
            let catalog = resolver_catalog();
            catalog.seed(
                &base,
                Some(Arc::new(
                    parse_catalog(&catalog_with_model("other/model")).expect("seed"),
                )),
            );
            reset_script(vec![Ok(catalog_with_model("other/model"))]);

            let err = resolve_capability_with(&catalog, endpoint, "brand/new-model")
                .await
                .unwrap_err();
            assert!(
                err.to_string()
                    .contains("not in the OpenRouter image-models catalog")
            );
            assert_eq!(FETCHES.load(Ordering::SeqCst), 1);
        }

        #[tokio::test]
        #[serial_test::serial(image_resolver)]
        async fn resolver_cooldown_prevents_refetch_storm() {
            let endpoint = crate::config::DEFAULT_PROVIDER_ENDPOINT;
            let base = crate::providers::ensure_base_url(endpoint);
            let catalog = resolver_catalog();
            catalog.seed(
                &base,
                Some(Arc::new(
                    parse_catalog(&catalog_with_model("other/model")).expect("seed"),
                )),
            );
            reset_script(vec![Ok(catalog_with_model("other/model"))]);

            for _ in 0..2 {
                let err = resolve_capability_with(&catalog, endpoint, "brand/new-model")
                    .await
                    .unwrap_err();
                assert!(
                    err.to_string()
                        .contains("not in the OpenRouter image-models catalog")
                );
            }
            assert_eq!(FETCHES.load(Ordering::SeqCst), 1);
        }

        #[tokio::test]
        #[serial_test::serial(image_resolver)]
        async fn resolver_refresh_failure_fails_open() {
            let endpoint = crate::config::DEFAULT_PROVIDER_ENDPOINT;
            let base = crate::providers::ensure_base_url(endpoint);
            let catalog = resolver_catalog();
            catalog.seed(
                &base,
                Some(Arc::new(
                    parse_catalog(&catalog_with_model("other/model")).expect("seed"),
                )),
            );
            reset_script(vec![Err(anyhow::anyhow!("upstream down"))]);

            let info = resolve_capability_with(&catalog, endpoint, "brand/new-model")
                .await
                .expect("fail-open");
            assert!(info.is_none());
            assert_eq!(FETCHES.load(Ordering::SeqCst), 1);

            let info = resolve_capability_with(&catalog, endpoint, "brand/new-model")
                .await
                .expect("fail-open");
            assert!(info.is_none());
            assert_eq!(FETCHES.load(Ordering::SeqCst), 1);
        }
    }
}

/// Client for the OpenRouter video-models catalog (`GET {base}/videos/models`).
///
/// Mirrors the image-side catalog: fetched once (single-flight), keyed by the
/// configured provider endpoint, reused for 24h, with a short negative-cache
/// backoff on failure and fail-open semantics (`get_catalog` returns
/// `None`). The cache machinery lives in [`crate::util::catalog_cache`],
/// shared with the image catalog. The video response shape differs from the
/// image one — flat per-model fields with high null variability (aleph-2
/// declares no resolutions/durations/sizes) — so it gets its own tolerant
/// parser. Models are looked up by `id`, never `canonical_slug` (they differ
/// for every live model).
pub(crate) mod video {
    use crate::util::catalog_cache::{Catalog, parse_envelope};
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
    /// envelope error contract in [`crate::util::catalog_cache::parse_envelope`].
    pub(crate) fn parse_catalog(body: &Value) -> anyhow::Result<VideoCatalog> {
        Ok(VideoCatalog {
            models: parse_envelope(body, "Video models catalog", parse_model)?,
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

    static CATALOG: Catalog<VideoCatalog> =
        Catalog::new("/videos/models", "Video models catalog", parse_catalog);

    /// Return the cached catalog for the default OpenRouter endpoint — media
    /// always targets OpenRouter, never a custom chat endpoint.
    pub(crate) async fn get_catalog() -> Option<Arc<VideoCatalog>> {
        let endpoint = crate::config::DEFAULT_PROVIDER_ENDPOINT;
        get_catalog_for_endpoint(endpoint).await
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
        // Media always targets OpenRouter — seed the default key.
        let endpoint = crate::providers::ensure_base_url(crate::config::DEFAULT_PROVIDER_ENDPOINT);
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
}
