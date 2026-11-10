//! OpenRouter discovery: HTTP client + catalog/endpoints/key fetches.
//!
//! The bench-openrouter subcommand fetches three live OpenRouter resources through its OWN
//! reqwest client (never the production provider stack, never [`crate::retry`] — the bench is a
//! standalone measurement harness): `GET /models` (the catalog), `GET
//! /models/{author}/{slug}/endpoints` (per-provider endpoint info) and `GET /key` (the key's
//! usage/limit envelope, for the dry-run affordability preflight).
//!
//! All three raw payloads are kept verbatim in [`DiscoverySnapshot`] as the audit snapshot
//! (`providers.json`), so every wire shape survives even though the typed structs are trimmed
//! to what the benchmark reads. Structs are permissive.

use anyhow::{Context, anyhow, bail};
use serde::Deserialize;
use std::time::Duration;

use crate::util::http::install_ring_provider;
use crate::util::truncate;

pub(crate) struct DiscoveryClient {
    client: reqwest::Client,
    key: String,
    base: String,
}

impl DiscoveryClient {
    /// Build a discovery client for the given API key.
    ///
    /// The ring TLS provider must be installed before any reqwest client is constructed.
    #[must_use]
    // Spec-pinned 60s request timeout (from_secs states the 60s intent directly).
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

    /// Build the `/models/{author}/{slug}/endpoints` path for a model id. Segments are
    /// percent-encoded by `path_segments_mut`; an id without a slash routes with an empty slug,
    /// which OpenRouter rejects with a clean 404.
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
    /// `path` may be a bare path (resolved against the API base) or an absolute URL, used as-is —
    /// prepending the base to an absolute URL would double it. On a non-2xx response the body (≤500
    /// chars) is returned as an error carrying status + body, so callers get the API's own message.
    async fn fetch_json(&self, path: &str) -> anyhow::Result<serde_json::Value> {
        let url = if path.starts_with("http") {
            path.to_string()
        } else {
            format!("{}{}", self.base, path)
        };
        let resp = self
            .client
            .get(&url)
            .bearer_auth(&self.key)
            .send()
            .await
            .with_context(|| format!("OpenRouter request failed: GET {url}"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let detail = truncate(body.trim(), 500);
            bail!(
                "OpenRouter GET {url} failed: HTTP {status}{}",
                if detail.is_empty() {
                    String::new()
                } else {
                    format!(": {detail}")
                }
            );
        }
        resp.json::<serde_json::Value>()
            .await
            .with_context(|| format!("OpenRouter GET {url} returned invalid JSON"))
    }
}

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
    pub supported_parameters: Option<Vec<String>>,
    #[serde(default)]
    pub reasoning: Option<ModelReasoning>,
}

/// Price object shared by catalog entries and endpoints: strings in USD per token, with
/// `input_cache_read` OPTIONAL (providers that do not advertise cache pricing omit it).
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
}

/// `reasoning` object of a catalog entry — OMITTED for non-reasoning models.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ModelReasoning {
    #[serde(default)]
    pub supported_efforts: Option<Vec<String>>,
}

/// Wrapper of `GET /models/{author}/{slug}/endpoints`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EndpointsResponse {
    pub data: EndpointsData,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EndpointsData {
    #[serde(default)]
    pub endpoints: Vec<EndpointInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EndpointInfo {
    /// Provider-pinning slug used in `provider.order` (e.g. `streamlake/fp8`).
    pub tag: String,
    pub name: String,
    pub provider_name: String,
    #[serde(default)]
    pub context_length: Option<i64>,
    #[serde(default)]
    pub quantization: Option<String>,
    /// STRING enum in the OpenAPI docs but an INTEGER on the live API — accepted permissively as
    /// either shape and normalized to a string ([`deserialize_status`]).
    #[serde(default, deserialize_with = "deserialize_status")]
    pub status: Option<String>,
    #[serde(default)]
    pub supports_implicit_caching: Option<bool>,
    #[serde(default)]
    pub pricing: Option<Pricing>,
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
    pub is_free_tier: Option<bool>,
    #[serde(default)]
    pub label: Option<String>,
}

/// Parse an OpenRouter price string (`"0.00000028"`) into USD per token. Empty/absent and
/// non-numeric input yield `None`; scientific notation parses fine.
#[must_use]
pub(crate) fn parse_price(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>().ok()
}

/// Permissive deserializer for the endpoint `status`: accepts either the documented string
/// enum or the integer the live API sends, normalizes to a string, and yields `None` for any
/// other type instead of failing the whole endpoints parse.
#[allow(
    clippy::unnecessary_wraps,
    reason = "Result is the deserialize_with contract"
)]
fn deserialize_status<'de, D>(de: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(serde_json::Value::deserialize(de)
        .ok()
        .and_then(|v| match v {
            serde_json::Value::String(s) => Some(s),
            serde_json::Value::Number(n) => Some(n.to_string()),
            _ => None,
        }))
}

fn parse_models(v: &serde_json::Value) -> anyhow::Result<Vec<ModelCatalogEntry>> {
    let data = v
        .get("data")
        .ok_or_else(|| anyhow!("OpenRouter models response missing 'data'"))?;
    serde_json::from_value(data.clone())
        .with_context(|| "OpenRouter models response could not be parsed")
}

fn parse_endpoints(v: &serde_json::Value, model_id: &str) -> anyhow::Result<EndpointsResponse> {
    serde_json::from_value(v.clone()).with_context(|| {
        format!("OpenRouter endpoints response for '{model_id}' could not be parsed")
    })
}

fn parse_key(v: &serde_json::Value) -> anyhow::Result<KeyInfo> {
    let data = v
        .get("data")
        .ok_or_else(|| anyhow!("OpenRouter key response missing 'data'"))?;
    serde_json::from_value(data.clone())
        .with_context(|| "OpenRouter key response could not be parsed")
}

/// Verbatim snapshot of one discovery pass — raw payloads kept for the audit
/// snapshot (`providers.json`).
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

/// Run one full discovery pass: models catalog → requested-model endpoints → key envelope,
/// capturing every raw payload verbatim.
///
/// The model is resolved by exact `id` match; when absent (aliases/snapshots may still route)
/// the endpoints lookup is attempted anyway. If that 404s, the error lists up to 5 close
/// catalog ids as suggestions.
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

/// "Did you mean" text for an unresolvable model id: up to 5 catalog ids sharing a substring.
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

    /// The endpoints fixture: pricing strings survive verbatim (they are the cost model) and a
    /// string `status` normalizes like the integer one.
    #[test]
    fn parses_endpoints_response_fixture() {
        let json = serde_json::json!({
            "data": {
                "id": "deepseek/deepseek-v4-flash-0731",
                "name": "DeepSeek V4 Flash",
                "created": 1_752_000_000,
                "endpoints": [
                    {
                        "context_length": 163_840,
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
                            "input_cache_read": "0.00000009"
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
        let ep = &parsed.data.endpoints[0];
        assert_eq!(ep.tag, "streamlake/fp8");
        assert_eq!(ep.name, "StreamLake");
        assert_eq!(ep.provider_name, "StreamLake");
        assert_eq!(ep.context_length, Some(163_840));
        assert_eq!(ep.quantization.as_deref(), Some("fp8"));
        assert_eq!(ep.status.as_deref(), Some("0"));
        assert_eq!(ep.supports_implicit_caching, Some(true));
        let pricing = ep.pricing.as_ref().expect("pricing present");
        assert_eq!(pricing.prompt.as_deref(), Some("0.00000028"));
        assert_eq!(pricing.input_cache_read.as_deref(), Some("0.00000009"));
        assert_eq!(pricing.request.as_deref(), Some("0"));
    }

    #[test]
    fn parses_live_integer_status_shape() {
        // The live API sends `status` as an INTEGER (0, -2) despite the
        // OpenAPI docs declaring a string enum — both shapes must parse and
        // normalize to the same string form (regression for the 404/parse
        // discovery failure found in the live smoke test).
        let json = serde_json::json!({
            "data": {
                "id": "deepseek/deepseek-v4-flash-0731",
                "endpoints": [
                    {
                        "tag": "streamlake/fp8",
                        "name": "StreamLake",
                        "provider_name": "StreamLake",
                        "status": 0,
                        "quantization": "fp8",
                        "context_length": 1_048_576
                    },
                    {
                        "tag": "decart/fp4",
                        "name": "Decart",
                        "provider_name": "Decart",
                        "status": -2,
                        "quantization": "fp4",
                        "context_length": 1_048_576
                    },
                    {
                        "tag": "weird/type",
                        "name": "Weird",
                        "provider_name": "Weird",
                        "status": {"nested": true},
                        "context_length": 1_048_576
                    }
                ]
            }
        });
        let parsed: EndpointsResponse = serde_json::from_value(json).expect("fixture must parse");
        assert_eq!(parsed.data.endpoints[0].status.as_deref(), Some("0"));
        assert_eq!(parsed.data.endpoints[1].status.as_deref(), Some("-2"));
        // A present-but-wrong-typed status degrades to None, not a parse failure.
        assert_eq!(parsed.data.endpoints[2].status, None);
    }
}
