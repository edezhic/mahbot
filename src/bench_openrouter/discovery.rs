//! OpenRouter discovery: HTTP client + catalog/endpoints/key fetches.
//!
//! The bench-openrouter subcommand fetches three live OpenRouter API
//! resources through its OWN reqwest client (never the production provider
//! stack, and never [`crate::retry`] — the bench is a standalone measurement
//! harness, not an agent path):
//!
//! - `GET /models` — the model catalog (id, pricing strings, reasoning
//!   capabilities, per-request limits).
//! - `GET /models/{author}/{slug}/endpoints` — per-provider endpoint info
//!   (status, cache pricing, uptime, provider-pinning `tag`).
//! - `GET /key` — the key's usage/limit envelope, used for the dry-run
//!   affordability preflight.
//!
//! All three raw payloads are kept verbatim in [`DiscoverySnapshot`] so Phase
//! 2 can persist them as the audit snapshot (`providers.json`). Structs are
//! permissive (unknown fields ignored, optionals where the API omits fields
//! for non-reasoning models / non-caching providers).
//!
//! Phase 1 uses `discover()` (models + endpoints + key in one pass); the
//! standalone fetch methods and several wire fields are consumed by Phase 2
//! (per-endpoint refresh, audit file writing), so they carry
//! `#![allow(dead_code)]` until then.

#![allow(dead_code)]

use anyhow::{Context, anyhow, bail};
use serde::Deserialize;
use std::time::Duration;

use crate::util::http::install_ring_provider;
use crate::util::truncate;

// ── Client ─────────────────────────────────────────────────────────

/// HTTP client for the OpenRouter discovery endpoints.
pub(crate) struct DiscoveryClient {
    client: reqwest::Client,
    key: String,
    base: String,
}

impl DiscoveryClient {
    /// Build a discovery client for the given API key.
    ///
    /// The ring TLS provider must be installed before any reqwest client is
    /// constructed (reqwest 0.13 rustls-no-provider path); every client
    /// construction in the process installs it idempotently.
    #[must_use]
    // Spec-pinned 60s request timeout (from_secs reads closer to the ticket).
    #[allow(clippy::duration_suboptimal_units)]
    pub(crate) fn new(key: String) -> Self {
        install_ring_provider();
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .build()
            .expect("failed to build OpenRouter discovery HTTP client");
        Self {
            client,
            key,
            base: "https://openrouter.ai/api/v1".to_string(),
        }
    }

    /// Build the `/models/{author}/{slug}/endpoints` path for a model id.
    ///
    /// Path segments are percent-encoded automatically by `path_segments_mut`
    /// (the author/slug split on `/`; a model id without a slash routes with
    /// an empty slug, which OpenRouter rejects on its side with a clean 404).
    fn endpoints_path(&self, model_id: &str) -> anyhow::Result<String> {
        let (author, slug) = model_id.split_once('/').unwrap_or((model_id, ""));
        let mut url = reqwest::Url::parse(&format!("{}/models", self.base))
            .with_context(|| format!("invalid OpenRouter base URL '{}'", self.base))?;
        url.path_segments_mut()
            .map_err(|()| anyhow!("cannot URL-encode model id '{model_id}'"))?
            .push(author)
            .push(slug)
            .push("endpoints");
        Ok(url.to_string())
    }

    /// GET `path` with `Authorization: Bearer <key>`.
    ///
    /// On a non-2xx response the body is read (≤500 chars) and returned as an
    /// error carrying status + body, so callers get the API's actual message
    /// (e.g. a 401 "invalid API key" or a 404 for an unknown model slug).
    async fn fetch_json(&self, path: &str) -> anyhow::Result<serde_json::Value> {
        let resp = self
            .client
            .get(format!("{}{}", self.base, path))
            .bearer_auth(&self.key)
            .send()
            .await
            .with_context(|| format!("OpenRouter request failed: GET {path}"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let detail = truncate(body.trim(), 500);
            bail!(
                "OpenRouter GET {path} failed: HTTP {status}{}",
                if detail.is_empty() {
                    String::new()
                } else {
                    format!(": {detail}")
                }
            );
        }
        resp.json::<serde_json::Value>()
            .await
            .with_context(|| format!("OpenRouter GET {path} returned invalid JSON"))
    }

    /// Fetch the full model catalog.
    pub(crate) async fn fetch_models(&self) -> anyhow::Result<Vec<ModelCatalogEntry>> {
        let v = self.fetch_json("/models").await?;
        parse_models(&v)
    }

    /// Fetch the per-provider endpoints of one model.
    pub(crate) async fn fetch_endpoints(
        &self,
        model_id: &str,
    ) -> anyhow::Result<EndpointsResponse> {
        let v = self.fetch_json(&self.endpoints_path(model_id)?).await?;
        parse_endpoints(&v, model_id)
    }

    /// Fetch the key's usage/limit envelope.
    pub(crate) async fn fetch_key(&self) -> anyhow::Result<KeyInfo> {
        let v = self.fetch_json("/key").await?;
        parse_key(&v)
    }
}

// ── Wire types (permissive Deserialize) ────────────────────────────

/// One entry of the `GET /models` catalog.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ModelCatalogEntry {
    pub id: String,
    #[serde(default)]
    pub canonical_slug: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub context_length: Option<i64>,
    #[serde(default)]
    pub links: Option<ModelLinks>,
    #[serde(default)]
    pub pricing: Option<Pricing>,
    #[serde(default)]
    pub supported_parameters: Option<Vec<String>>,
    #[serde(default)]
    pub reasoning: Option<ModelReasoning>,
    #[serde(default)]
    pub top_provider: Option<serde_json::Value>,
}

/// `links` object of a catalog entry (the endpoints detail link).
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ModelLinks {
    #[serde(default)]
    pub details: Option<String>,
}

/// Price object shared by catalog entries and endpoints.
///
/// Prices arrive as strings in USD per token (e.g. `"0.00000028"`); the
/// `input_cache_read`/`input_cache_write` fields are OPTIONAL — providers that
/// do not advertise cache pricing omit them entirely.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct Pricing {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub completion: Option<String>,
    #[serde(default)]
    pub request: Option<String>,
    #[serde(default)]
    pub input_cache_read: Option<String>,
    #[serde(default)]
    pub input_cache_write: Option<String>,
    #[serde(default)]
    pub discount: Option<f64>,
}

/// `reasoning` object of a catalog entry — OMITTED for non-reasoning models.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ModelReasoning {
    #[serde(default)]
    pub default_effort: Option<String>,
    #[serde(default)]
    pub supported_efforts: Option<Vec<String>>,
    #[serde(default)]
    pub mandatory: Option<bool>,
}

/// Wrapper of `GET /models/{author}/{slug}/endpoints`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EndpointsResponse {
    pub data: EndpointsData,
}

/// `data` of the endpoints response.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EndpointsData {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub endpoints: Vec<EndpointInfo>,
}

/// One provider endpoint serving the model.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EndpointInfo {
    /// Provider-pinning slug used in `provider.order` (e.g. `streamlake/fp8`).
    pub tag: String,
    pub name: String,
    pub provider_name: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub context_length: Option<i64>,
    #[serde(default)]
    pub max_completion_tokens: Option<i64>,
    #[serde(default)]
    pub quantization: Option<String>,
    /// STRING enum: `"0"` healthy, `"-1"`/`"-2"`/`"-3"`/`"-5"`/`"-10"`
    /// unhealthy (see [`crate::bench_openrouter::select::is_healthy_status`]
    /// for the refined bands).
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub supports_implicit_caching: Option<bool>,
    #[serde(default)]
    pub supported_parameters: Option<Vec<String>>,
    #[serde(default)]
    pub pricing: Option<Pricing>,
    #[serde(default)]
    pub latency_last_30m: Option<serde_json::Value>,
    #[serde(default)]
    pub uptime_last_1d: Option<f64>,
    #[serde(default)]
    pub uptime_last_30m: Option<f64>,
    #[serde(default)]
    pub uptime_last_5m: Option<f64>,
}

/// Key usage/limit envelope from `GET /key`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct KeyInfo {
    #[serde(default)]
    pub limit: Option<f64>,
    #[serde(default)]
    pub limit_remaining: Option<f64>,
    #[serde(default)]
    pub limit_reset: Option<String>,
    #[serde(default)]
    pub usage: Option<f64>,
    #[serde(default)]
    pub is_free_tier: Option<bool>,
    #[serde(default)]
    pub label: Option<String>,
}

// ── Parsing helpers ────────────────────────────────────────────────

/// Parse an OpenRouter price string (`"0.00000028"`) into USD per token.
///
/// Empty/absent strings and non-numeric input yield `None`; scientific
/// notation (`"1e-5"`) parses fine.
#[must_use]
pub(crate) fn parse_price(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>().ok()
}

/// Parse the `data` array of the models response.
fn parse_models(v: &serde_json::Value) -> anyhow::Result<Vec<ModelCatalogEntry>> {
    let data = v
        .get("data")
        .ok_or_else(|| anyhow!("OpenRouter models response missing 'data'"))?;
    serde_json::from_value(data.clone())
        .with_context(|| "OpenRouter models response could not be parsed")
}

/// Parse the endpoints response body for `model_id`.
fn parse_endpoints(v: &serde_json::Value, model_id: &str) -> anyhow::Result<EndpointsResponse> {
    serde_json::from_value(v.clone()).with_context(|| {
        format!("OpenRouter endpoints response for '{model_id}' could not be parsed")
    })
}

/// Parse the key response body.
fn parse_key(v: &serde_json::Value) -> anyhow::Result<KeyInfo> {
    let data = v
        .get("data")
        .ok_or_else(|| anyhow!("OpenRouter key response missing 'data'"))?;
    serde_json::from_value(data.clone())
        .with_context(|| "OpenRouter key response could not be parsed")
}

// ── Snapshot + orchestration ───────────────────────────────────────

/// Verbatim snapshot of one discovery pass — raw payloads kept for the Phase
/// 2 audit file (`providers.json`).
pub(crate) struct DiscoverySnapshot {
    /// RFC 3339 with milliseconds, e.g. `2026-08-20T00:00:00.000Z`.
    pub fetched_at: String,
    pub catalog: Vec<ModelCatalogEntry>,
    pub endpoints: EndpointsResponse,
    pub key: KeyInfo,
    pub raw_models_json: serde_json::Value,
    pub raw_endpoints_json: serde_json::Value,
    pub raw_key_json: serde_json::Value,
}

/// Run one full discovery pass: models catalog → requested-model endpoints →
/// key envelope, capturing every raw payload verbatim.
///
/// The requested model is resolved by exact `id` match in the catalog; when it
/// is absent (aliases/snapshots may still route), the endpoints lookup is
/// attempted anyway. If that 404s, the error lists up to 5 close catalog ids
/// (simple substring match) as suggestions.
pub(crate) async fn discover(
    client: &DiscoveryClient,
    model: &str,
) -> anyhow::Result<DiscoverySnapshot> {
    let fetched_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);

    let raw_models_json = client.fetch_json("/models").await?;
    let catalog = parse_models(&raw_models_json)?;

    let exact_match = catalog.iter().any(|e| e.id == model);
    let raw_endpoints_json = match client.fetch_json(&client.endpoints_path(model)?).await {
        Ok(v) => v,
        Err(e) if !exact_match => {
            return Err(anyhow!(
                "model '{model}' was not found in the OpenRouter catalog and its \
                 endpoints lookup failed: {e:#}{}",
                suggestions_text(&catalog, model)
            ));
        }
        Err(e) => return Err(e),
    };
    let endpoints = parse_endpoints(&raw_endpoints_json, model)?;

    let raw_key_json = client.fetch_json("/key").await?;
    let key = parse_key(&raw_key_json)?;

    Ok(DiscoverySnapshot {
        fetched_at,
        catalog,
        endpoints,
        key,
        raw_models_json,
        raw_endpoints_json,
        raw_key_json,
    })
}

/// "Did you mean" text for an unresolvable model id: up to 5 catalog ids that
/// share a substring with the request. Empty when nothing is close.
fn suggestions_text(catalog: &[ModelCatalogEntry], model: &str) -> String {
    let suggestions: Vec<&str> = catalog
        .iter()
        .map(|e| e.id.as_str())
        .filter(|id| !id.is_empty() && (id.contains(model) || model.contains(id)))
        .take(5)
        .collect();
    if suggestions.is_empty() {
        return String::new();
    }
    format!(
        " Did you mean: {}?",
        suggestions
            .iter()
            .map(|s| format!("'{s}'"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_price_handles_edge_cases() {
        assert_eq!(parse_price(""), None);
        assert_eq!(parse_price("   "), None);
        assert_eq!(parse_price("0"), Some(0.0));
        assert_eq!(parse_price("0.00000028"), Some(0.00000028));
        assert_eq!(parse_price("abc"), None);
        assert_eq!(parse_price("1e-5"), Some(0.00001));
    }

    #[test]
    fn parses_endpoints_response_fixture() {
        let json = serde_json::json!({
            "data": {
                "id": "deepseek/deepseek-v4-flash-0731",
                "name": "DeepSeek V4 Flash",
                "created": 1_752_000_000,
                "endpoints": [
                    {
                        "context_length": 163840,
                        "max_completion_tokens": 8192,
                        "max_prompt_tokens": null,
                        "model_id": "deepseek/deepseek-v4-flash-0731",
                        "model_name": "DeepSeek V4 Flash",
                        "name": "StreamLake",
                        "provider_name": "StreamLake",
                        "pricing": {
                            "prompt": "0.00000028",
                            "completion": "0.00000028",
                            "request": "0",
                            "input_cache_read": "0.00000009",
                            "input_cache_write": "0.00000028",
                            "discount": 0.0
                        },
                        "quantization": "fp8",
                        "status": "0",
                        "supported_parameters": ["tools", "tool_choice"],
                        "supports_implicit_caching": true,
                        "tag": "streamlake/fp8",
                        "latency_last_30m": {"p50": 1234.5},
                        "uptime_last_1d": 99.5,
                        "uptime_last_30m": null,
                        "uptime_last_5m": null,
                        "extra_unknown_field": "ignored"
                    }
                ]
            }
        });
        let parsed: EndpointsResponse = serde_json::from_value(json).expect("fixture must parse");
        assert_eq!(parsed.data.id, "deepseek/deepseek-v4-flash-0731");
        assert_eq!(parsed.data.name.as_deref(), Some("DeepSeek V4 Flash"));
        let ep = &parsed.data.endpoints[0];
        assert_eq!(ep.tag, "streamlake/fp8");
        assert_eq!(ep.name, "StreamLake");
        assert_eq!(ep.provider_name, "StreamLake");
        assert_eq!(ep.context_length, Some(163_840));
        assert_eq!(ep.max_completion_tokens, Some(8192));
        assert_eq!(ep.quantization.as_deref(), Some("fp8"));
        // status is a STRING ("0"), not a number.
        assert_eq!(ep.status.as_deref(), Some("0"));
        assert_eq!(ep.supports_implicit_caching, Some(true));
        assert_eq!(
            ep.supported_parameters.as_deref(),
            Some(&["tools".to_string(), "tool_choice".to_string()][..])
        );
        assert_eq!(ep.uptime_last_1d, Some(99.5));
        assert_eq!(ep.uptime_last_30m, None);
        // pricing strings survive verbatim.
        let pricing = ep.pricing.as_ref().expect("pricing present");
        assert_eq!(pricing.prompt.as_deref(), Some("0.00000028"));
        assert_eq!(pricing.input_cache_read.as_deref(), Some("0.00000009"));
        assert_eq!(pricing.request.as_deref(), Some("0"));
    }

    #[test]
    fn endpoints_path_builds_encoded_segments() {
        install_ring_provider();
        let client = DiscoveryClient {
            client: reqwest::Client::new(),
            key: "k".to_string(),
            base: "https://openrouter.ai/api/v1".to_string(),
        };
        assert_eq!(
            client
                .endpoints_path("deepseek/deepseek-v4-flash-0731")
                .unwrap(),
            "https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints"
        );
    }
}
