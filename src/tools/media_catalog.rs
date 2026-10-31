//! Image & video model-catalog clients.
//!
//! Both fetch through the single-flight, endpoint-keyed, TTL-cached machinery in
//! [`crate::util::catalog_cache`], whose module docs carry the caching and
//! fail-open contract. Media always targets OpenRouter, never a custom chat
//! endpoint, so both clients read the default endpoint.
//!
//! They live in separate submodules because each declares the same helper names
//! (`CATALOG`, `parse_catalog`, `get_catalog`, `seed_cache`).

/// Client for the OpenRouter image-models catalog (`GET {base}/images/models`).
///
/// Each entry declares its generation parameters (`supported_parameters`:
/// names, enums, ranges) and its output modalities
/// (`architecture.output_modalities`). Every image-gen capability and parameter
/// decision is driven by this data — never by model-name branches. Models are
/// keyed by `id`.
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

        /// The declared values of enum `param`; `None` when the catalog does not
        /// declare `param` or declares it with another constraint shape.
        #[must_use]
        pub(crate) fn enum_values(&self, param: &str) -> Option<&[String]> {
            match self.supported_parameters.get(param) {
                Some(ParameterConstraint::Enum(values)) => Some(values),
                _ => None,
            }
        }

        /// Whether `param` is a declared enum that includes `value`.
        #[must_use]
        pub(crate) fn enum_contains(&self, param: &str, value: &str) -> bool {
            self.enum_values(param)
                .is_some_and(|values| values.iter().any(|v| v == value))
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

    // ── Cached fetch ───────────────────────────────────────────────────

    static CATALOG: Catalog<ImageCatalog> =
        Catalog::new("/images/models", "Image models catalog", parse_catalog);

    /// The cached catalog; `None` when it is unavailable — callers fail open.
    pub(crate) async fn get_catalog() -> Option<Arc<ImageCatalog>> {
        CATALOG.get(crate::config::DEFAULT_PROVIDER_ENDPOINT).await
    }

    /// Test-only: seed the shared cache (no network).
    #[cfg(test)]
    pub(crate) fn seed_cache(catalog: Option<Arc<ImageCatalog>>) {
        let endpoint = crate::providers::ensure_base_url(crate::config::DEFAULT_PROVIDER_ENDPOINT);
        CATALOG.seed(&endpoint, catalog);
    }

    /// The remediation the account's own-model paths append to a capability verdict:
    /// it speaks to the account's pick. A walk candidate's skip line leaves it out —
    /// the model the account chose is not the one the walk passed over.
    const IMAGE_MODEL_REMEDIATION: &str =
        " Set an image-capable model in Settings → Provider → Models settings and retry.";

    /// Reject a catalog-listed model that explicitly declares text-only output.
    /// (Shape drift — missing/empty modalities — is tolerated as capable.)
    fn reject_text_only(model: &str, info: &ImageModelInfo) -> anyhow::Result<()> {
        if info.image_output == ImageOutput::TextOnly {
            anyhow::bail!(
                "Model `{model}` cannot generate images: the OpenRouter image-models catalog \
                 does not list image output for it.{IMAGE_MODEL_REMEDIATION}"
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
    /// taken is accepted instead of being rejected until that snapshot expires.
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
                 image-models catalog.{IMAGE_MODEL_REMEDIATION}"
            ),
        }
    }

    /// Write-time validation that `model` can generate images, using the catalog
    /// when available (a catalog outage passes — see [`resolve_image_model`]).
    ///
    /// Used by the Telegram model commands and the GUI model picker.
    ///
    /// # Errors
    ///
    /// - Catalog available and the model is absent from it or text-only.
    pub async fn validate_image_model(model: &str) -> anyhow::Result<()> {
        resolve_image_model(crate::config::DEFAULT_PROVIDER_ENDPOINT, model)
            .await
            .map(|_| ())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::util::catalog_cache::FetchFuture;
        use serde_json::json;
        use std::collections::VecDeque;
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

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
/// The response shape is flat per-model fields with high null variability
/// (aleph-2 declares no resolutions/durations/sizes), so it gets its own
/// tolerant parser. Models are looked up by `id`, never `canonical_slug` (they
/// differ for every live model).
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

    // ── Cached fetch ───────────────────────────────────────────────────

    static CATALOG: Catalog<VideoCatalog> =
        Catalog::new("/videos/models", "Video models catalog", parse_catalog);

    /// The cached catalog; `None` when it is unavailable — callers fail open.
    pub(crate) async fn get_catalog() -> Option<Arc<VideoCatalog>> {
        CATALOG.get(crate::config::DEFAULT_PROVIDER_ENDPOINT).await
    }

    /// Test-only: seed the shared cache (no network).
    #[cfg(test)]
    pub(crate) fn seed_cache(catalog: Option<Arc<VideoCatalog>>) {
        let endpoint = crate::providers::ensure_base_url(crate::config::DEFAULT_PROVIDER_ENDPOINT);
        CATALOG.seed(&endpoint, catalog);
    }
}
