//! Client for the OpenRouter image-models catalog (`GET {base}/images/models`).
//!
//! The catalog declares per-model image-generation parameters (parameter
//! names, enums, ranges, output modalities). Every image-gen capability and
//! parameter decision is driven by this data — never by model-name branches.
//!
//! Caching: fetched once (single-flight), keyed by the configured provider
//! endpoint, and reused for [`CATALOG_TTL`]. A failed fetch is stored as a
//! short-lived negative cache ([`CATALOG_RETRY_BACKOFF`]) and degrades to
//! fail-open: [`get_catalog`] returns `None` and callers send minimal
//! parameters instead of rejecting generation.

use crate::util::UnwrapPoison;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

/// How long a fetched catalog is considered fresh.
const CATALOG_TTL: Duration = Duration::from_hours(24);

/// Backoff before retrying a failed catalog fetch (negative cache).
const CATALOG_RETRY_BACKOFF: Duration = Duration::from_mins(1);

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

/// Parse a catalog response body. Fails on malformed envelopes but tolerates
/// unknown model entries, missing fields, and unknown parameter shapes.
///
/// # Errors
///
/// - Missing/invalid `data` array or an empty model set.
pub(crate) fn parse_catalog(body: &Value) -> anyhow::Result<ImageCatalog> {
    let data = body
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("Image models catalog response missing `data` array"))?;

    let mut models = HashMap::new();
    for entry in data {
        if let Some((id, info)) = parse_model(entry) {
            models.insert(id, info);
        }
    }
    if models.is_empty() {
        anyhow::bail!("Image models catalog contained no models");
    }
    Ok(ImageCatalog { models })
}

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

struct CatalogCache {
    endpoint: String,
    fetched_at: Instant,
    /// `None` = a failed fetch (negative cache for the backoff window).
    catalog: Option<Arc<ImageCatalog>>,
}

enum CatalogLookup {
    Fresh(Arc<ImageCatalog>),
    Backoff,
    Miss,
}

static CATALOG: OnceLock<RwLock<Option<CatalogCache>>> = OnceLock::new();
/// Serializes concurrent fetches — one network call regardless of how many
/// agents hit image_gen at once.
static CATALOG_FETCH_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

fn catalog_lock() -> &'static RwLock<Option<CatalogCache>> {
    CATALOG.get_or_init(|| RwLock::new(None))
}

async fn fetch_lock() -> tokio::sync::MutexGuard<'static, ()> {
    CATALOG_FETCH_LOCK
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

fn lookup(endpoint: &str) -> CatalogLookup {
    let guard = catalog_lock().read().unwrap_poison();
    let Some(cache) = guard.as_ref() else {
        return CatalogLookup::Miss;
    };
    if cache.endpoint != endpoint {
        return CatalogLookup::Miss;
    }
    match &cache.catalog {
        Some(catalog) if cache.fetched_at.elapsed() < CATALOG_TTL => {
            CatalogLookup::Fresh(catalog.clone())
        }
        None if cache.fetched_at.elapsed() < CATALOG_RETRY_BACKOFF => CatalogLookup::Backoff,
        _ => CatalogLookup::Miss,
    }
}

/// Return the cached catalog for the currently configured provider endpoint.
pub(crate) async fn get_catalog() -> Option<Arc<ImageCatalog>> {
    let endpoint = crate::config::CONFIG.provider_endpoint();
    get_catalog_for_endpoint(&endpoint).await
}

/// Return the cached catalog for `endpoint` if fresh, otherwise fetch it
/// (single-flight). Returns `None` (fail-open) when the catalog is
/// unavailable — retried after a short negative-cache backoff.
async fn get_catalog_for_endpoint(endpoint: &str) -> Option<Arc<ImageCatalog>> {
    let base = crate::providers::ensure_base_url(endpoint);
    match lookup(&base) {
        CatalogLookup::Fresh(catalog) => return Some(catalog),
        CatalogLookup::Backoff => return None,
        CatalogLookup::Miss => {}
    }
    let _guard = fetch_lock().await;
    match lookup(&base) {
        CatalogLookup::Fresh(catalog) => return Some(catalog),
        CatalogLookup::Backoff => return None,
        CatalogLookup::Miss => {}
    }

    let url = format!("{base}/images/models");
    match crate::util::http::get_json_from_provider(&url, "Image models catalog").await {
        Ok(body) => match parse_catalog(&body) {
            Ok(catalog) => {
                let catalog = Arc::new(catalog);
                *catalog_lock().write().unwrap_poison() = Some(CatalogCache {
                    endpoint: base,
                    fetched_at: Instant::now(),
                    catalog: Some(catalog.clone()),
                });
                Some(catalog)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to parse image-models catalog — proceeding without capability checks");
                store_failure(base);
                None
            }
        },
        Err(e) => {
            tracing::warn!(error = %e, "Failed to fetch image-models catalog — proceeding without capability checks");
            store_failure(base);
            None
        }
    }
}

fn store_failure(endpoint: String) {
    *catalog_lock().write().unwrap_poison() = Some(CatalogCache {
        endpoint,
        fetched_at: Instant::now(),
        catalog: None,
    });
}

/// Reject models that cannot generate images on the dedicated surface:
/// unknown model ids or models that explicitly declare text-only output.
/// Returns the model's catalog info for request building; models whose
/// catalog entry lacks output modalities (shape drift) are allowed through.
/// Catalog-driven — no per-model branches.
///
/// # Errors
///
/// - The model is absent from the catalog or explicitly declares no image output.
pub(crate) fn check_image_capability<'a>(
    model: &str,
    catalog: &'a ImageCatalog,
) -> anyhow::Result<&'a ImageModelInfo> {
    let Some(info) = catalog.find(model) else {
        anyhow::bail!(
            "Model `{model}` cannot generate images: it is not in the OpenRouter \
             image-models catalog. Set an image-capable model in Settings → \
             Image Generation and retry."
        );
    };
    if info.image_output == ImageOutput::TextOnly {
        anyhow::bail!(
            "Model `{model}` cannot generate images: the OpenRouter image-models catalog \
             does not list image output for it. Set an image-capable model in \
             Settings → Image Generation and retry."
        );
    }
    Ok(info)
}

/// Write-time validation that `model` can generate images, using the catalog
/// when available. Fail-open: when the catalog is unavailable the check passes
/// (with a warning) — consistent with the generation tool's fail-open
/// semantics, so a catalog outage never blocks settings writes.
///
/// Used by the settings save path, the GUI model picker, and the Telegram
/// model commands.
///
/// # Errors
///
/// - Catalog available and the model is absent from it or text-only.
pub async fn validate_image_model(model: &str) -> anyhow::Result<()> {
    let endpoint = crate::config::CONFIG.provider_endpoint();
    validate_image_model_for_endpoint(model, &endpoint).await
}

/// Like [`validate_image_model`], but against the catalog for `endpoint`
/// rather than the currently configured one — the settings save path uses
/// this when a save may switch providers.
pub(crate) async fn validate_image_model_for_endpoint(
    model: &str,
    endpoint: &str,
) -> anyhow::Result<()> {
    let Some(catalog) = get_catalog_for_endpoint(endpoint).await else {
        tracing::warn!(
            "Image-models catalog unavailable — skipping write-time model validation (fail-open)"
        );
        return Ok(());
    };
    check_image_capability(model, &catalog).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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

    #[test]
    fn test_lookup_endpoint_keyed_with_negative_backoff() {
        let endpoint = "https://openrouter.ai/api/v1";
        let other = "https://other.example/api/v1";

        // Fresh catalog → Fresh; a different provider endpoint → Miss.
        *catalog_lock().write().unwrap_poison() = Some(CatalogCache {
            endpoint: endpoint.into(),
            fetched_at: Instant::now(),
            catalog: Some(Arc::new(ImageCatalog::default())),
        });
        assert!(matches!(lookup(endpoint), CatalogLookup::Fresh(_)));
        assert!(matches!(lookup(other), CatalogLookup::Miss));

        // Failed fetch → Backoff inside the window, Miss once it expires.
        *catalog_lock().write().unwrap_poison() = Some(CatalogCache {
            endpoint: endpoint.into(),
            fetched_at: Instant::now(),
            catalog: None,
        });
        assert!(matches!(lookup(endpoint), CatalogLookup::Backoff));
        *catalog_lock().write().unwrap_poison() = Some(CatalogCache {
            endpoint: endpoint.into(),
            fetched_at: Instant::now()
                .checked_sub(CATALOG_RETRY_BACKOFF + Duration::from_secs(1))
                .expect("monotonic clock is past the backoff window"),
            catalog: None,
        });
        assert!(matches!(lookup(endpoint), CatalogLookup::Miss));

        *catalog_lock().write().unwrap_poison() = None;
    }
}
