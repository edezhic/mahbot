//! Generic OpenAI-compatible provider.
//! Most LLM APIs follow the same `/v1/chat/completions` format.
//! This module provides a single implementation that works for all of them.

use crate::providers::reasoning_roundtrip;
use crate::providers::{ScopedCallError, ensure_chat_completions_url, provider_routing_json};
use crate::retry::{FailureClass, RetryFailureRecord};
use crate::util::error::{HttpError, retry_after_header};
use crate::util::json::try_repair_json;
use crate::{
    ChatMessage, ChatRequest as ProviderChatRequest, ChatResponse as ProviderChatResponse,
    ChatRole, Provider, ProviderUsage, Reasoning, ToolCall as ProviderToolCall, ToolSpec,
};
use async_trait::async_trait;
use base64::{
    Engine as _,
    engine::general_purpose::{STANDARD, URL_SAFE, URL_SAFE_NO_PAD},
};
use futures_util::StreamExt;
use reqwest::{
    Client, RequestBuilder,
    header::{HeaderMap, HeaderValue},
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// A provider that speaks the OpenAI-compatible chat completions API.
pub(crate) struct OpenAiCompatibleProvider {
    pub name: String,
    pub base_url: String,
    pub credential: Option<String>,

    /// HTTP request timeout in seconds for LLM API calls. Default: 120.
    timeout_secs: u64,
    /// Extra HTTP headers to include in all API requests.
    extra_headers: std::collections::HashMap<String, String>,
    /// Cached HTTP client with connection reuse across all API calls.
    /// Initialized lazily on first `http_client()` call.
    http_client: OnceLock<Client>,
    /// Cached HTTP client for scoped calls: NO total request
    /// timeout — per-attempt total is enforced by the scoped caller against
    /// the remaining operation budget, and idle timeouts reset while data
    /// flows. Initialized lazily on first `http_client_scoped()` call.
    http_client_scoped: OnceLock<Client>,
}

impl OpenAiCompatibleProvider {
    #[must_use]
    pub fn new(name: &str, base_url: &str, credential: Option<&str>) -> Self {
        Self {
            name: name.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            credential: credential.map(ToString::to_string),
            timeout_secs: 120,
            extra_headers: std::collections::HashMap::new(),
            http_client: OnceLock::new(),
            http_client_scoped: OnceLock::new(),
        }
    }

    /// Set extra HTTP headers to include in all API requests.
    #[must_use]
    pub fn with_extra_headers(
        mut self,
        headers: std::collections::HashMap<String, String>,
    ) -> Self {
        self.extra_headers = headers;
        self
    }

    /// Build the shared HTTP client with the given total request timeout.
    ///
    /// `Some(timeout)` yields the default client (120 s total request
    /// timeout); `None` yields the scoped client (no total timeout — the
    /// scoped call paths enforce their own per-attempt deadline and idle
    /// timeout). Connection pool, connect timeout, and extra headers are
    /// identical in both.
    fn build_client(&self, timeout: Option<Duration>) -> Client {
        crate::util::http::install_ring_provider();
        let mut builder = Client::builder().connect_timeout(Duration::from_secs(10));
        if let Some(timeout) = timeout {
            builder = builder.timeout(timeout);
        }

        if !self.extra_headers.is_empty() {
            let mut headers = HeaderMap::new();
            for (key, value) in &self.extra_headers {
                match (
                    reqwest::header::HeaderName::from_bytes(key.as_bytes()),
                    HeaderValue::from_str(value),
                ) {
                    (Ok(name), Ok(val)) => {
                        headers.insert(name, val);
                    }
                    _ => {
                        tracing::warn!(header = key, "Skipping invalid extra header name or value");
                    }
                }
            }
            builder = builder.default_headers(headers);
        }

        builder
            .build()
            .expect("Failed to build HTTP client — check TLS/network configuration")
    }

    pub(crate) fn http_client(&self) -> &Client {
        self.http_client
            .get_or_init(|| self.build_client(Some(Duration::from_secs(self.timeout_secs))))
    }

    /// Scoped HTTP client: same connection pool and headers as
    /// [`Self::http_client`] but WITHOUT the total request timeout. The scoped
    /// call paths enforce their own per-attempt deadline and idle timeout.
    pub(crate) fn http_client_scoped(&self) -> &Client {
        self.http_client_scoped
            .get_or_init(|| self.build_client(None))
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<NativeMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    /// Provider-specific fields merged at the top level of the JSON body.
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}
#[derive(Debug, Deserialize)]
struct ApiChatResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<UsageInfo>,
    /// Telemetry fields below are permissive (`opt_field`): a present-but-
    /// wrong-typed value yields NULL instead of failing the envelope parse —
    /// telemetry must never break the request path.
    #[serde(default, deserialize_with = "opt_field")]
    system_fingerprint: Option<String>,
    /// Serving upstream provider — OpenRouter's top-level response field
    /// (undocumented in the API reference but consumed by OpenRouter's own
    /// SDK; empirically present incl. cache hits, where `openrouter_metadata`
    /// is stripped). NULL when the provider omits it.
    #[serde(default, deserialize_with = "opt_field")]
    provider: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UsageInfo {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    /// OpenRouter-normalized shape: `usage.prompt_tokens_details.cached_tokens`.
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
    /// DeepSeek-native shape: `usage.prompt_cache_hit_tokens` /
    /// `usage.prompt_cache_miss_tokens` (sum equals `prompt_tokens`).
    #[serde(default)]
    prompt_cache_hit_tokens: Option<u64>,
    #[serde(default)]
    prompt_cache_miss_tokens: Option<u64>,
    /// Billed cost from `usage.cost` — the invoice amount (OpenRouter-only).
    #[serde(default, deserialize_with = "opt_field")]
    cost: Option<f64>,
    /// Raw cost breakdown (OpenRouter only); any JSON value is accepted.
    /// `opt_field`-protected: a pathological value (e.g. `1e999`, which
    /// serde_json rejects as out of range) must not fail the envelope parse.
    #[serde(default, deserialize_with = "opt_field")]
    cost_details: Option<serde_json::Value>,
}

/// Permissive deserializer for optional telemetry fields: a present value of
/// the wrong type (provider shape drift) yields `None` instead of failing the
/// whole envelope parse. Missing and `null` also yield `None`.
#[allow(
    clippy::unnecessary_wraps,
    reason = "Result is the deserialize_with contract"
)]
fn opt_field<'de, D, T>(de: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    Ok(serde_json::Value::deserialize(de)
        .ok()
        .and_then(|v| serde_json::from_value(v).ok()))
}

#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

/// Remove `<think>...</think>` blocks from model output.
/// Some reasoning models (e.g. `MiniMax`) embed their chain-of-thought inline
/// in the `content` field rather than a separate `reasoning_content` field.
/// The resulting `<think>` tags must be stripped before returning to the user.

#[derive(Debug, Deserialize, Serialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    /// Reasoning/thinking models (e.g. Qwen3, GLM-4) may return their output
    /// in `reasoning_content` instead of `content`. Preserved on the response
    /// for replay/display — never promoted into visible text (see
    /// [`effective_content_optional`](Self::effective_content_optional)).
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_details: Option<serde_json::Value>,
    #[serde(default)]
    tool_calls: Option<Vec<ApiToolCall>>,
}

impl ResponseMessage {
    /// Extract the model's visible text content, stripping `<think>...</think>`
    /// blocks that some models (e.g. `MiniMax`) embed inline in `content`.
    ///
    /// There is NO reasoning fallback here: a reasoning-only response (empty
    /// content, no tool calls) stays empty so the agent loop can classify the
    /// class early and recover via bounded continuation (see
    /// [`crate::agent::Agent::recover_reasoning_only_stop`]) instead of
    /// surfacing chain-of-thought as the visible answer. Reasoning fields are
    /// preserved separately on the response for replay and display.
    fn effective_content_optional(&self) -> Option<String> {
        self.content
            .as_ref()
            .filter(|c| !c.is_empty())
            .and_then(|c| crate::providers::reasoning::strip_think_tags(c))
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct ApiToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    function: Option<ApiToolCallFunction>,

    // Compatibility: Some providers (e.g., older GLM) may use 'name' directly
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,

    // Compatibility: DeepSeek sometimes wraps arguments differently
    #[serde(
        rename = "parameters",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    parameters: Option<serde_json::Value>,
}

/// Resolve tool-call name from `function.name` or top-level `name`.
/// Returns the first non-empty name, or `None` if both are absent/empty.
#[must_use]
pub(crate) fn resolve_tool_call_name(
    function_name: Option<&str>,
    direct_name: Option<&str>,
) -> Option<String> {
    function_name
        .filter(|n| !n.is_empty())
        .or_else(|| direct_name.filter(|n| !n.is_empty()))
        .map(String::from)
}

/// Resolve tool-call arguments from `function.arguments`, top-level `arguments`,
/// or the `parameters` field (DeepSeek compatibility where arguments arrive as an object).
/// Returns the first non-empty arguments string, or `None` if all are absent/empty.
#[must_use]
pub(crate) fn resolve_tool_call_arguments(
    function_arguments: Option<&str>,
    direct_arguments: Option<&str>,
    parameters: Option<&serde_json::Value>,
) -> Option<String> {
    if let Some(args) = function_arguments.filter(|a| !a.is_empty()) {
        return Some(args.to_string());
    }
    if let Some(args) = direct_arguments.filter(|a| !a.is_empty()) {
        return Some(args.to_string());
    }
    // Compatibility: Some providers return parameters as object instead of string
    parameters.and_then(|params| serde_json::to_string(params).ok())
}

impl ApiToolCall {
    /// Extract function name with fallback logic for various provider formats
    fn function_name(&self) -> Option<String> {
        resolve_tool_call_name(
            self.function.as_ref().and_then(|f| f.name.as_deref()),
            self.name.as_deref(),
        )
    }

    /// Extract arguments with fallback logic and type conversion
    fn function_arguments(&self) -> Option<String> {
        resolve_tool_call_arguments(
            self.function.as_ref().and_then(|f| f.arguments.as_deref()),
            self.arguments.as_deref(),
            self.parameters.as_ref(),
        )
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct ApiToolCallFunction {
    #[serde(default)]
    pub(crate) name: Option<String>,
    #[serde(default)]
    pub(crate) arguments: Option<String>,
}

#[derive(Debug, Serialize)]
struct NativeMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ApiToolCall>>,
    /// Raw reasoning content from thinking models; pass-through for providers
    /// that require it in assistant tool-call history messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_details: Option<serde_json::Value>,
}

impl NativeMessage {
    #[cfg(test)]
    fn user(content: &str) -> Self {
        NativeMessage {
            role: "user".into(),
            content: Some(MessageContent::Text(content.into())),
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
            reasoning: None,
            reasoning_details: None,
        }
    }
}

// ── Message content types for API serialization ──

/// Image subtypes DeepSeek (and the OpenAI image_url shape) will actually
/// decode. Placeholders like `data:image/...;base64,...` must not match.
const NATIVE_IMAGE_SUBTYPES: &[&str] = &["jpeg", "jpg", "png", "gif", "webp"];

/// True when `path` is a payload the chat-completions `image_url` field accepts:
/// an http(s) URL, or a `data:image/{jpeg|png|gif|webp};base64,…` URI whose
/// payload is valid base64 **and** decodes to a real jpeg/png/gif/webp raster.
/// Prose / path / ellipsis / fake-base64 markers stay in the surrounding text.
#[must_use]
fn is_native_image_url(path: &str) -> bool {
    if crate::util::is_http_url(path) {
        return true;
    }
    let Some(bytes) = decode_image_data_uri_payload(path) else {
        return false;
    };
    is_supported_raster(&bytes)
}

/// Pull the base64 payload out of a `data:image/<subtype>[;params];base64,…`
/// URI and decode it. `None` for a non-image MIME, empty payload, or bytes
/// that are not valid standard / URL-safe base64.
#[must_use]
fn decode_image_data_uri_payload(path: &str) -> Option<Vec<u8>> {
    let rest = path.strip_prefix("data:image/")?;
    let (head, payload) = rest.rsplit_once(";base64,")?;
    let subtype = head.split(';').next().unwrap_or("").trim();
    if !NATIVE_IMAGE_SUBTYPES
        .iter()
        .any(|allowed| subtype.eq_ignore_ascii_case(allowed))
    {
        return None;
    }
    decode_base64_payload(payload)
}

/// Decode standard or URL-safe base64, stripping ASCII whitespace. Rejects
/// alphabet lookalikes that are not actually decodable (`...`, truncated pad).
#[must_use]
fn decode_base64_payload(s: &str) -> Option<Vec<u8>> {
    let compact: Cow<'_, [u8]> = if s.as_bytes().iter().any(u8::is_ascii_whitespace) {
        Cow::Owned(
            s.bytes()
                .filter(|b| !b.is_ascii_whitespace())
                .collect::<Vec<u8>>(),
        )
    } else {
        Cow::Borrowed(s.as_bytes())
    };
    if compact.is_empty() {
        return None;
    }
    STANDARD
        .decode(compact.as_ref())
        .ok()
        .or_else(|| URL_SAFE.decode(compact.as_ref()).ok())
        .or_else(|| URL_SAFE_NO_PAD.decode(compact.as_ref()).ok())
}

/// True when `bytes` sniff as jpeg/png/gif/webp **and** decode as a raster
/// (catches valid-base64 garbage like `abcd` and truncated files whose magic
/// still looks like PNG).
#[must_use]
fn is_supported_raster(bytes: &[u8]) -> bool {
    let Ok(format) = image::guess_format(bytes) else {
        return false;
    };
    if !matches!(
        format,
        image::ImageFormat::Jpeg
            | image::ImageFormat::Png
            | image::ImageFormat::Gif
            | image::ImageFormat::WebP
    ) {
        return false;
    }
    image::load_from_memory(bytes).is_ok()
}

/// Parse `[IMAGE:…]` markers from content, returning cleaned text and extracted
/// native image payloads.
///
/// Uses the shared [`MEDIA_MARKER_RE`](crate::util::MEDIA_MARKER_RE) to find markers.
/// Only data-URI payloads that decode to a jpeg/png/gif/webp raster, plus
/// http(s) IMAGE payloads, are extracted (and stripped from the cleaned text)
/// — those become [`MessagePart::ImageUrl`]. Prose/path IMAGE markers, empty
/// `[IMAGE:]`, fake/truncated data URIs, and non‑IMAGE markers (e.g.
/// `[AUDIO:…]`) are left untouched in the cleaned text.
#[must_use]
pub(crate) fn parse_image_markers(content: &str) -> (String, Vec<String>) {
    let mut refs: Vec<String> = Vec::new();

    let cleaned = crate::util::MEDIA_MARKER_RE
        .replace_all(content, |caps: &regex::Captures| {
            let (kind, path) = crate::util::parse_media_marker(caps);
            let path = path.trim();

            if kind == "IMAGE" && is_native_image_url(path) {
                refs.push(path.to_string());
                // Native IMAGE payloads are stripped — they become image parts.
                String::new()
            } else {
                // AUDIO/VIDEO markers and non-native IMAGE markers stay verbatim.
                caps.get_match().as_str().to_string()
            }
        })
        .to_string();

    (cleaned.trim().to_string(), refs)
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum MessageContent {
    Text(String),
    Parts(Vec<MessagePart>),
    Null,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum MessagePart {
    Text { text: String },
    ImageUrl { image_url: ImageUrlPart },
}

#[derive(Debug, Serialize)]
pub(crate) struct ImageUrlPart {
    pub url: String,
}

/// Convert a role+content pair into the appropriate [`MessageContent`] variant.
///
/// For [`ChatRole::User`] content, native image payloads (data URIs that
/// decode to a real jpeg/png/gif/webp raster, and http(s) URLs inside
/// `[IMAGE:…]`) are parsed into [`MessagePart::ImageUrl`] entries alongside
/// the cleaned text — every role now emits native image parts. Prose/path
/// markers and fake/truncated data URIs stay in the text. Everything else is
/// returned as [`MessageContent::Text`].
///
/// The old estimator-side mirror of this marker handling
/// (`crate::session::estimate_tokens`) was removed with the
/// token-estimation heuristic — this conversion is now the only place
/// marker parsing lives.
pub(crate) fn to_message_content(role: ChatRole, content: &str) -> MessageContent {
    if role != ChatRole::User {
        return MessageContent::Text(content.to_string());
    }

    // Fast path: avoid regex work when there are no IMAGE markers at all.
    // All valid markers begin with "[IMAGE:" so a simple substring check is safe.
    if !content.contains("[IMAGE:") {
        return MessageContent::Text(content.to_string());
    }

    let (cleaned_text, image_refs) = parse_image_markers(content);
    if image_refs.is_empty() {
        return MessageContent::Text(content.to_string());
    }

    let mut parts = Vec::with_capacity(image_refs.len() + 1);
    let trimmed_text = cleaned_text.trim();
    if !trimmed_text.is_empty() {
        parts.push(MessagePart::Text {
            text: trimmed_text.to_string(),
        });
    }

    for image_ref in image_refs {
        parts.push(MessagePart::ImageUrl {
            image_url: ImageUrlPart { url: image_ref },
        });
    }

    MessageContent::Parts(parts)
}

impl OpenAiCompatibleProvider {
    fn convert_tool_specs(tools: Option<&[ToolSpec]>) -> Option<Vec<serde_json::Value>> {
        let items = tools?;
        let converted: Vec<_> = items
            .iter()
            .map(|tool| {
                let params = tool.parameters.clone();
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": params,
                    }
                })
            })
            .collect();
        if converted.is_empty() {
            None
        } else {
            Some(converted)
        }
    }

    fn convert_messages_for_native(messages: &[ChatMessage]) -> Vec<NativeMessage> {
        messages
            .iter()
            .map(|message| {
                // Shared fields extracted from a `DecodedNativeHistoryMessage` used
                // to build provider-native message types.
                // Tool calls are returned as `Vec<ToolCall>` so each provider can convert them
                // to its own tool-call type.
                let decoded = crate::session::decode_native_history_message(message);
                let Some((role, content, tool_call_id, tool_calls, reasoning)) =
                    decoded.map(|msg| match msg {
                        crate::session::DecodedNativeHistoryMessage::Assistant {
                            content,
                            tool_calls,
                            reasoning,
                        } => (
                            ChatRole::Assistant.to_string(),
                            content,
                            None, // tool_call_id
                            tool_calls,
                            reasoning,
                        ),
                        crate::session::DecodedNativeHistoryMessage::ToolResult {
                            tool_call_id,
                            content,
                        } => (
                            ChatRole::Tool.to_string(),
                            Some(content),
                            Some(tool_call_id),
                            None, // tool_calls
                            None, // reasoning
                        ),
                    })
                else {
                    return NativeMessage {
                        role: message.role.to_string(),
                        content: Some(to_message_content(message.role, &message.content)),
                        tool_call_id: None,
                        tool_calls: None,
                        reasoning: None,
                        reasoning_content: None,
                        reasoning_details: None,
                    };
                };
                let has_tool_calls = tool_calls.as_ref().is_some_and(|c| !c.is_empty());
                let (r_reasoning, r_content, r_details) =
                    reasoning_roundtrip::native_reasoning_triple_for_replay(
                        reasoning.as_ref(),
                        has_tool_calls,
                    );
                let tool_calls = tool_calls.map(|tc| {
                    tc.into_iter()
                        .map(|tc| ApiToolCall {
                            id: Some(tc.id),
                            kind: Some("function".to_string()),
                            function: Some(ApiToolCallFunction {
                                name: Some(tc.name),
                                arguments: Some(
                                    serde_json::to_string(&tc.arguments)
                                        .unwrap_or_else(|_| "{}".into()),
                                ),
                            }),
                            name: None,
                            arguments: None,
                            parameters: None,
                        })
                        .collect()
                });
                let has_reasoning =
                    r_content.is_some() || r_reasoning.is_some() || r_details.is_some();
                let content = match (&content, has_reasoning, has_tool_calls) {
                    (Some(s), _, _) => Some(MessageContent::Text(s.clone())),
                    (None, true, true) => Some(MessageContent::Null),
                    (None, true, false) => Some(MessageContent::Text(String::new())),
                    (None, false, _) => None,
                };
                NativeMessage {
                    role,
                    content,
                    tool_call_id,
                    tool_calls,
                    reasoning: r_reasoning,
                    reasoning_content: r_content,
                    reasoning_details: r_details,
                }
            })
            .collect()
    }
}

/// Parse tool-call arguments JSON with repair fallback and fallback to empty object on parse failure.
#[must_use]
fn parse_tool_call_arguments(name: &str, arguments: &str) -> serde_json::Value {
    serde_json::from_str(arguments).unwrap_or_else(|parse_err| {
        if let Some(value) = try_repair_json::<serde_json::Value>(arguments) {
            tracing::debug!(
                function = %name,
                original_error = %parse_err,
                "Repaired malformed JSON in tool-call arguments"
            );
            return value;
        }
        tracing::debug!(
            function = %name,
            arguments = %arguments,
            error = %parse_err,
            "Invalid JSON in tool-call arguments, using empty object"
        );
        serde_json::json!({})
    })
}

/// Shared helper to build a [`ProviderToolCall`] from parsed API tool-call data.
///
/// Delegates argument parsing to [`parse_tool_call_arguments`], which handles
/// JSON parsing with repair fallback, and generates a fallback ID when none is
/// provided.
#[must_use]
fn make_provider_tool_call(id: Option<String>, name: String, arguments: &str) -> ProviderToolCall {
    let arguments = parse_tool_call_arguments(&name, arguments);
    ProviderToolCall {
        id: id.unwrap_or_else(crate::generate_id),
        name,
        arguments,
    }
}

/// Normalize provider-reported cache tokens into `(cached, miss)`.
///
/// OpenRouter reports only the hit side (`prompt_tokens_details.cached_tokens`;
/// the miss side is computed as `prompt_tokens − cached`), while DeepSeek
/// reports both sides natively (`prompt_cache_hit_tokens` /
/// `prompt_cache_miss_tokens`). Native miss wins; otherwise the computed
/// miss is used when both operands are known.
#[must_use]
fn normalize_cache_tokens(
    cached_tokens: Option<u64>,
    hit_tokens: Option<u64>,
    miss_tokens: Option<u64>,
    prompt_tokens: Option<u64>,
) -> (Option<u64>, Option<u64>) {
    let cached = cached_tokens.or(hit_tokens);
    let miss = miss_tokens.or_else(|| match (cached, prompt_tokens) {
        (Some(c), Some(p)) => p.checked_sub(c),
        _ => None,
    });
    (cached, miss)
}

impl OpenAiCompatibleProvider {
    /// `finish_reason`, `upstream_provider`, `system_fingerprint` are three
    /// consecutive `Option<String>`s — keep call sites in envelope order.
    fn parse_native_response(
        message: ResponseMessage,
        usage: Option<ProviderUsage>,
        finish_reason: Option<String>,
        upstream_provider: Option<String>,
        system_fingerprint: Option<String>,
    ) -> ProviderChatResponse {
        let text = message.effective_content_optional();
        let reasoning = Reasoning::from_optional_parts(
            message.reasoning,
            message.reasoning_content,
            message.reasoning_details,
        );
        let tool_calls = message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .filter_map(|tc| {
                let name = tc.function_name()?;
                let arguments = tc.function_arguments().unwrap_or_default();
                Some(make_provider_tool_call(tc.id, name, &arguments))
            })
            .collect::<Vec<_>>();

        ProviderChatResponse {
            text,
            tool_calls,
            usage,
            reasoning,
            finish_reason,
            upstream_provider,
            system_fingerprint,
        }
    }

    /// Shared success tail of [`Provider::chat`] and [`Provider::chat_scoped`]:
    /// usage mapping (incl. provider-reported cache tokens), first-choice
    /// extraction, native-response parsing, and the tool-turn debug log.
    /// `no_response` builds the error when no choice exists.
    fn finalize_response<E>(
        &self,
        model: &str,
        native_response: ApiChatResponse,
        no_response: impl FnOnce() -> E,
    ) -> Result<ProviderChatResponse, E> {
        let usage = native_response.usage.map(|u| {
            let (cached, miss) = normalize_cache_tokens(
                u.prompt_tokens_details
                    .as_ref()
                    .and_then(|d| d.cached_tokens),
                u.prompt_cache_hit_tokens,
                u.prompt_cache_miss_tokens,
                u.prompt_tokens,
            );
            ProviderUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cached_input_tokens: cached,
                cache_miss_tokens: miss,
                cost: u.cost,
                cost_details: u.cost_details,
            }
        });
        let upstream_provider = native_response.provider.clone();
        let choice = native_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(no_response)?;
        let finish_reason = choice.finish_reason;
        let message = choice.message;

        let result = Self::parse_native_response(
            message,
            usage,
            finish_reason,
            upstream_provider,
            native_response.system_fingerprint,
        );

        if !result.tool_calls.is_empty() && result.reasoning.is_none() {
            tracing::debug!(
                provider = %self.name,
                model,
                "tool turn: parsed response has no reasoning fields",
            );
        }

        Ok(result)
    }

    /// Build the HTTP request for [`Provider::chat`] / [`Provider::chat_scoped`].
    /// This function itself is synchronous; the caller sends the request asynchronously.
    ///
    /// The request BYTES are identical regardless of which client is used —
    /// the scoped client only differs in timeout semantics, never in payload.
    fn build_http_request_with_client(
        &self,
        client: &Client,
        request: &ProviderChatRequest,
    ) -> RequestBuilder {
        let native = Self::convert_messages_for_native(&request.messages);
        let tool_specs = Self::convert_tool_specs(request.tools.as_deref());

        let mut extra = serde_json::Map::new();

        // OpenRouter provider preferences — OpenRouter-only (req 7):
        // the block has no meaning outside OpenRouter, so it is never sent to
        // custom endpoints. Per-request `data_collection: allow` overrides the
        // account-level strict privacy default so data-collecting paid endpoints
        // remain reachable; optional routing fields are merged into the same
        // object when configured.
        if crate::config::is_default_endpoint(&self.base_url) {
            let mut provider = request
                .provider_order
                .as_deref()
                .and_then(provider_routing_json)
                .and_then(|v| v.as_object().cloned())
                .unwrap_or_default();
            provider.insert("data_collection".to_string(), serde_json::json!("allow"));
            extra.insert("provider".to_string(), serde_json::Value::Object(provider));
        }

        // Reasoning effort — model-family-aware for custom endpoints.
        // The default OpenRouter endpoint stays byte-identical: the effort
        // passes through unchanged (OpenRouter normalizes the value per
        // model). Custom endpoints get the family's
        // native field/value vocabulary (e.g. Ollama 400s on `xhigh`), so
        // reasoning works by default on self-hosted servers too.
        match reasoning_fields_for_request(
            &self.base_url,
            &request.model,
            request.reasoning_effort.as_deref(),
        ) {
            ReasoningFields::Effort(value) => {
                extra.insert("reasoning_effort".to_string(), serde_json::json!(value));
            }
            ReasoningFields::ThinkingEnabled => {
                extra.insert(
                    "thinking".to_string(),
                    serde_json::json!({ "type": "enabled" }),
                );
            }
            ReasoningFields::Omit => {}
        }

        let payload = ChatCompletionRequest {
            model: request.model.clone(),
            messages: native,
            max_tokens: request.max_tokens,
            tool_choice: tool_specs.as_ref().map(|_| "auto".to_string()),
            tools: tool_specs,
            extra,
        };

        let url = ensure_chat_completions_url(&self.base_url);
        let builder = client.post(url).json(&payload);
        self.attach_auth_header(builder)
    }

    /// Attach the `Authorization: Bearer` header if a credential is configured.
    /// Returns the builder (with or without the header added) for chaining.
    fn attach_auth_header(&self, mut builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref credential) = self.credential {
            builder = builder.header("Authorization", format!("Bearer {credential}"));
        }
        builder
    }
}

// ── Model-family reasoning translation for custom endpoints ──

/// Reasoning-field outcome for a chat-completions request after
/// model-family translation.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ReasoningFields {
    /// Send `reasoning_effort: <value>`.
    Effort(String),
    /// Send `thinking: {"type": "enabled"}` and NO `reasoning_effort` (MiMo).
    ThinkingEnabled,
    /// Send neither `reasoning_effort` nor `thinking` (MiniMax, or no effort).
    Omit,
}

/// Pure, deterministic translation of a role's reasoning effort into the
/// request fields a chat-completions endpoint accepts. Computed at
/// request-build time so every retry attempt stays byte-identical.
///
/// - No effort (`None`/empty) → no reasoning fields, regardless of endpoint.
/// - Default OpenRouter endpoint → the effort passes through unchanged
///   (byte-identical behavior; OpenRouter normalizes the
///   value per model).
/// - Custom endpoint → [`translate_for_custom_endpoint`] applies the
///   model family's native field/value vocabulary.
#[must_use]
fn reasoning_fields_for_request(
    endpoint: &str,
    model: &str,
    effort: Option<&str>,
) -> ReasoningFields {
    let Some(effort) = effort.filter(|e| !e.is_empty()) else {
        return ReasoningFields::Omit;
    };
    if crate::config::is_default_endpoint(endpoint) {
        return ReasoningFields::Effort(effort.to_string());
    }
    translate_for_custom_endpoint(model, effort)
}

/// Per-model-family reasoning vocabulary for custom (non-OpenRouter)
/// endpoints: OpenRouter normalizes `reasoning_effort` per model, but each
/// self-hosted family has its own accepted values/field shapes (Ollama 400s
/// on `xhigh`; MiMo and MiniMax have no effort parameter at all).
///
/// Detection is a bare case-insensitive substring match, first-match-wins in
/// [`ReasoningFamily`] declaration order. NO version parsing: older siblings
/// that contain a family name (minimax-m2.x, gemini-3.1, deepseek-v3.x,
/// kimi-k2.x, glm-4.x, grok-4.5...) receive the latest-family vocabulary.
/// "No support for old versions" means no guarantee and no version-specific
/// code — NOT routing them to the fallback — so an older sibling may reject
/// the emitted value (accepted, documented limitation).
#[must_use]
fn translate_for_custom_endpoint(model: &str, effort: &str) -> ReasoningFields {
    let family = detect_family(model);
    match family {
        // MiMo: no reasoning_effort parameter — enable thinking via the
        // `thinking` object instead. The family's field shape dominates for
        // ANY present effort value (incl. theoretical medium/low/minimal/none):
        // there is no effort field to pass through on this family.
        ReasoningFamily::Mimo => ReasoningFields::ThinkingEnabled,
        // MiniMax M3: thinking is on by default on the OpenAI-compatible
        // endpoint — omit reasoning_effort AND the thinking object. The
        // "thinking:{type:adaptive} if verified" clause is deliberately NOT
        // implemented: there is no live endpoint to verify it, so omit-both
        // is the only behavior. Field shape dominates for any effort value.
        ReasoningFamily::MiniMax => ReasoningFields::Omit,
        _ => {
            let value = match (family, effort) {
                // hy3 (Tencent): xhigh→high, high→low — the mapped value must
                // be sent so reasoning is actually enabled/raised. Note:
                // Tencent silently raises low→high whenever tools are present
                // (i.e. on all mahbot agent calls), so the one-step-below-max
                // intent is defeated on the hot path — harmless, still below
                // max.
                // gemini 3.7 Flash: xhigh→high, high→medium.
                (ReasoningFamily::Hy3 | ReasoningFamily::Gemini, "xhigh") => "high",
                (ReasoningFamily::Hy3, "high") => "low",
                (ReasoningFamily::Gemini, "high") => "medium",
                // deepseek v4 / kimi k3 / glm 5.x: max is the top level.
                // The fallback (any unknown family, incl. GPT models) shares
                // this shape. Accepted residual risk: `max` 400s on
                // OpenAI-native GPT-5.x and strict {low,medium,high}-only
                // servers (vLLM/LM Studio) — Ollama accepts it, so the common
                // self-hosted case is covered.
                (
                    ReasoningFamily::Deepseek
                    | ReasoningFamily::Kimi
                    | ReasoningFamily::Glm
                    | ReasoningFamily::Fallback,
                    "xhigh",
                ) => "max",
                // muse-spark / grok already accept xhigh; every unmapped
                // effort value (medium/low/minimal/none) passes through
                // unchanged for the effort-bearing families.
                _ => effort,
            };
            ReasoningFields::Effort(value.to_string())
        }
    }
}

/// Reasoning-effort vocabulary families. Declaration order is
/// the family-detection order: a case-insensitive substring match on the
/// model name, first-match-wins (deepseek, kimi, glm, hy3, muse-spark, grok,
/// gemini, mimo, minimax), then the fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningFamily {
    Deepseek,
    Kimi,
    Glm,
    Hy3,
    MuseSpark,
    Grok,
    Gemini,
    Mimo,
    MiniMax,
    Fallback,
}

#[must_use]
fn detect_family(model: &str) -> ReasoningFamily {
    let m = model.to_ascii_lowercase();
    if m.contains("deepseek") {
        ReasoningFamily::Deepseek
    } else if m.contains("kimi") {
        ReasoningFamily::Kimi
    } else if m.contains("glm") {
        ReasoningFamily::Glm
    } else if m.contains("hy3") {
        ReasoningFamily::Hy3
    } else if m.contains("muse-spark") {
        ReasoningFamily::MuseSpark
    } else if m.contains("grok") {
        ReasoningFamily::Grok
    } else if m.contains("gemini") {
        ReasoningFamily::Gemini
    } else if m.contains("mimo") {
        ReasoningFamily::Mimo
    } else if m.contains("minimax") {
        ReasoningFamily::MiniMax
    } else {
        ReasoningFamily::Fallback
    }
}

/// Outcome of an idle-timeout body read.
enum BodyReadOutcome {
    /// Body fully read.
    Complete(Vec<u8>),
    /// Read failed partway (transport / idle timeout / shutdown).
    /// Carries the partial bytes plus a message for the error trail.
    Failed {
        partial: Vec<u8>,
        message: String,
        class: FailureClass,
    },
}

/// Read a response body chunk-by-chunk with an idle timeout that resets while
/// data flows, also bounding the whole read by `deadline`.
///
/// `idle_timeout` guards against a stalled connection mid-body — the
/// truncation signature this hardening targets. The read is shutdown-abortable
/// via [`crate::shutdown::race_shutdown`].
///
/// Every chunk wait is bounded by `min(idle_timeout, remaining_budget)`, so a
/// stalled chunk can never overshoot the operation deadline by more than the
/// scheduling latency between the timeout firing and the next check — the
/// wall-clock cap holds precisely for the body-read phase, not just the send
/// phase. When the tighter bound was the wall budget, the failure classifies
/// as [`FailureClass::WallClockExceeded`] rather than a truncation idle
/// timeout.
async fn read_body_idle(
    response: reqwest::Response,
    idle_timeout: Duration,
    deadline: Instant,
) -> BodyReadOutcome {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return BodyReadOutcome::Failed {
                partial: body,
                message: "response body read exceeded remaining operation budget".to_string(),
                class: FailureClass::WallClockExceeded,
            };
        }
        let wait_bound = idle_timeout.min(remaining);
        let next_chunk = crate::shutdown::race_shutdown(stream.next());
        let chunk = match tokio::time::timeout(wait_bound, next_chunk).await {
            Err(_) => {
                if Instant::now() >= deadline {
                    // The tighter bound was the wall budget — classify as such,
                    // not as an idle timeout.
                    return BodyReadOutcome::Failed {
                        partial: body,
                        message: "response body read exceeded remaining operation budget"
                            .to_string(),
                        class: FailureClass::WallClockExceeded,
                    };
                }
                return BodyReadOutcome::Failed {
                    partial: body,
                    message: format!(
                        "response body read idle timeout after {idle_timeout:?} \
                         with no data flowing"
                    ),
                    class: FailureClass::TruncatedEnvelope,
                };
            }
            Ok(Err(_)) => {
                return BodyReadOutcome::Failed {
                    partial: body,
                    message: "shutdown during response body read".to_string(),
                    class: FailureClass::Shutdown,
                };
            }
            Ok(Ok(Some(chunk))) => chunk,
            Ok(Ok(None)) => break,
        };
        match chunk {
            Ok(bytes) => body.extend_from_slice(&bytes),
            Err(e) => {
                return BodyReadOutcome::Failed {
                    partial: body,
                    message: format!("{e}"),
                    class: FailureClass::TruncatedEnvelope,
                };
            }
        }
    }
    BodyReadOutcome::Complete(body)
}

/// Best-effort extraction of `finish_reason` from a possibly-truncated JSON
/// body.
///
/// A full `ApiChatResponse` parse is attempted first; if the envelope is cut
/// mid-body we fall back to lenient JSON value parsing so telemetry survives
/// where possible.
fn envelope_telemetry(body: &str) -> Option<String> {
    if let Ok(native) = serde_json::from_str::<ApiChatResponse>(body) {
        return native.choices.first().and_then(|c| c.finish_reason.clone());
    }
    // Lenient fallback for truncated envelopes.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
        return value
            .pointer("/choices/0/finish_reason")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
    }
    None
}

#[async_trait]
impl Provider for OpenAiCompatibleProvider {
    async fn warmup(&self) -> anyhow::Result<()> {
        // Hit the chat completions URL with a GET to establish the connection pool.
        // The server will likely return 405 Method Not Allowed, which is fine -
        // the goal is TLS handshake and HTTP/2 negotiation.
        let url = ensure_chat_completions_url(&self.base_url);
        let builder = self.http_client().get(&url);
        let _ = self.attach_auth_header(builder).send().await?;
        Ok(())
    }

    /// Scoped single-attempt chat: one HTTP request, no
    /// provider-internal retries, idle-timeout body reads, per-attempt total
    /// bounded by the remaining operation deadline.
    async fn chat_scoped(
        &self,
        request: ProviderChatRequest,
        idle_timeout: Duration,
        deadline: Instant,
    ) -> Result<ProviderChatResponse, ScopedCallError> {
        let req_builder = self.build_http_request_with_client(self.http_client_scoped(), &request);
        let model = request.model;

        // ── Send — bounded by the idle timeout (TTFB) AND the remaining budget ──
        // A pre-header server stall must not consume the whole operation
        // budget on attempt 1 (burst-shaped recovery needs later attempts), so
        // the header wait is capped by `idle_timeout`; the remaining wall
        // budget remains the outer bound.
        let remaining = deadline.saturating_duration_since(Instant::now());
        let send_timeout = idle_timeout.min(remaining);
        let send_fut = crate::shutdown::race_shutdown(req_builder.send());
        let response = match tokio::time::timeout(send_timeout, send_fut).await {
            Err(_) => {
                let budget_expired = remaining <= idle_timeout;
                let err = if budget_expired {
                    anyhow::anyhow!("{} request exceeded remaining operation budget", self.name)
                } else {
                    anyhow::anyhow!(
                        "{} request timed out waiting for response headers \
                         (idle timeout {idle_timeout:?})",
                        self.name
                    )
                };
                let class = if budget_expired {
                    FailureClass::WallClockExceeded
                } else {
                    FailureClass::Transport
                };
                return Err(scoped_simple_error(err, class));
            }
            Ok(Err(_)) => {
                let err = anyhow::anyhow!("shutdown during request");
                return Err(scoped_simple_error(err, FailureClass::Shutdown));
            }
            Ok(Ok(Err(e))) => {
                let err = anyhow::Error::from(e).context(format!("{} transport error", self.name));
                return Err(scoped_simple_error(err, FailureClass::Transport));
            }
            Ok(Ok(Ok(resp))) => resp,
        };

        // ── Response metadata (telemetry) ──
        let content_length = response.content_length();

        if !response.status().is_success() {
            return Err(scoped_http_error(self, response, idle_timeout, deadline).await);
        }

        // ── Read body with idle timeout ──
        let (body_bytes, read_failure) =
            match read_body_idle(response, idle_timeout, deadline).await {
                BodyReadOutcome::Complete(bytes) => (bytes, None),
                BodyReadOutcome::Failed {
                    partial,
                    message,
                    class,
                } => (partial, Some((message, class))),
            };
        let body_str = String::from_utf8_lossy(&body_bytes).into_owned();
        let actual_len = body_bytes.len();

        if let Some((read_msg, class)) = read_failure {
            let err = anyhow::anyhow!("{} error reading response body: {read_msg}", self.name);
            return Err(scoped_metadata_error(err, class, &body_str, None));
        }

        let native_response: ApiChatResponse = match serde_json::from_str(&body_str) {
            Ok(native) => native,
            Err(e) => {
                // Content-length vs actual comparison: equal ⇒ the server
                // finalized a short body; shorter ⇒ framing anomaly (truncated
                // envelope — the same defect class as body-read errors).
                let truncated = content_length.is_some_and(|cl| cl != actual_len as u64);
                let class = if truncated {
                    FailureClass::TruncatedEnvelope
                } else {
                    FailureClass::Parse
                };
                let err = anyhow::anyhow!(
                    "{} chat completions parse error: {e}; body ({}): {:.500}",
                    self.name,
                    body_str.len(),
                    body_str
                );
                return Err(scoped_metadata_error(err, class, &body_str, None));
            }
        };

        self.finalize_response(&model, native_response, || {
            scoped_simple_error(
                anyhow::anyhow!("No response from {}", self.name),
                FailureClass::NoResponse,
            )
        })
    }
}

/// Build a scoped-call error with no envelope telemetry (send-phase failures).
fn scoped_simple_error(err: anyhow::Error, class: FailureClass) -> ScopedCallError {
    let record = RetryFailureRecord::new_simple(class, &err, None);
    ScopedCallError::new(err, record, class)
}

/// Build a scoped-call error with envelope telemetry (finish_reason).
fn scoped_metadata_error(
    err: anyhow::Error,
    class: FailureClass,
    body: &str,
    retry_after_ms: Option<u64>,
) -> ScopedCallError {
    let record =
        RetryFailureRecord::with_metadata(class, &err, envelope_telemetry(body), retry_after_ms);
    ScopedCallError::new(err, record, class)
}

/// Build a [`ScopedCallError`] from an HTTP error response (non-2xx).
///
/// Reads the error body with idle-timeout semantics so a stalled error-body
/// read cannot hang past the operation deadline, and classifies via the
/// provider error classifier.
async fn scoped_http_error(
    provider: &OpenAiCompatibleProvider,
    response: reqwest::Response,
    idle_timeout: Duration,
    deadline: Instant,
) -> ScopedCallError {
    let status = response.status().as_u16();
    let retry_after_ms = retry_after_header(response.headers());
    let (body_bytes, message) = match read_body_idle(response, idle_timeout, deadline).await {
        BodyReadOutcome::Complete(bytes) => (bytes, None),
        BodyReadOutcome::Failed {
            partial, message, ..
        } => (partial, Some(message)),
    };
    let body = String::from_utf8_lossy(&body_bytes).into_owned();

    let http_err = HttpError {
        status,
        body: body.clone(),
        context: provider.name.clone(),
    };
    let inner = anyhow::Error::from(http_err);
    let body_read_failed = message.is_some();
    let inner = match message {
        None => inner,
        Some(read_msg) => inner.context(format!(
            "{} error reading response body: {read_msg}",
            provider.name
        )),
    };

    let class = crate::providers::failure_class(
        crate::providers::reliable::classify_err(&inner),
        body_read_failed,
    );
    scoped_metadata_error(inner, class, &body, retry_after_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::test_request;

    /// 1×1 red PNG as a data URI — used wherever tests need a payload that
    /// survives [`is_native_image_url`]'s decode + raster sniff.
    fn tiny_png_data_uri() -> String {
        use std::io::Cursor;
        let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 0, 255]));
        let mut buf = Vec::new();
        img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
            .expect("test PNG must encode");
        format!("data:image/png;base64,{}", STANDARD.encode(&buf))
    }

    #[tokio::test]
    async fn chat_without_key_attempts_request() {
        let p = OpenAiCompatibleProvider::new("Local", "http://127.0.0.1:1", None);
        let result = p
            .chat(test_request(vec![ChatMessage::user("hello")], None))
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            !err_msg.contains("API key not set"),
            "should not get credential error, got: {err_msg}"
        );
    }

    /// Provider routing is OpenRouter-only (req 7): asserted on
    /// the serialized request body at the builder choke point — the block is
    /// sent for the default endpoint and suppressed for a custom endpoint,
    /// while `reasoning_effort` is sent unchanged to the default endpoint and
    /// translated per model family on custom endpoints.
    #[test]
    fn provider_routing_block_suppressed_for_custom_endpoint() {
        let mut request = test_request(vec![ChatMessage::user("hello")], None);
        request.provider_order = Some("DeepSeek".to_string());
        request.reasoning_effort = Some("xhigh".to_string());

        let body = |p: &OpenAiCompatibleProvider| {
            let req = p
                .build_http_request_with_client(p.http_client_scoped(), &request)
                .build()
                .expect("request builds");
            String::from_utf8(
                req.body()
                    .expect("full body")
                    .as_bytes()
                    .expect("bytes")
                    .to_vec(),
            )
            .expect("utf8 body")
        };

        // Default endpoint → routing block present, reasoning_effort present.
        let or_provider = OpenAiCompatibleProvider::new(
            "OpenRouter",
            crate::config::DEFAULT_PROVIDER_ENDPOINT,
            Some("sk-or"),
        );
        let or_body = body(&or_provider);
        assert!(
            or_body.contains("\"provider\""),
            "default endpoint must send the provider block: {or_body}"
        );
        assert!(
            or_body.contains("DeepSeek"),
            "routing order must be inside the block: {or_body}"
        );
        assert!(
            or_body.contains("\"data_collection\":\"allow\""),
            "default endpoint must override account privacy with data_collection allow: {or_body}"
        );
        assert!(
            or_body.contains("xhigh"),
            "reasoning_effort must be sent to the default endpoint: {or_body}"
        );

        // Default endpoint without routing still sends data_collection allow.
        let mut no_routing = test_request(vec![ChatMessage::user("hello")], None);
        no_routing.provider_order = None;
        let no_routing_body = {
            let req = or_provider
                .build_http_request_with_client(or_provider.http_client_scoped(), &no_routing)
                .build()
                .expect("request builds");
            String::from_utf8(
                req.body()
                    .expect("full body")
                    .as_bytes()
                    .expect("bytes")
                    .to_vec(),
            )
            .expect("utf8 body")
        };
        assert!(
            no_routing_body.contains("\"data_collection\":\"allow\""),
            "default endpoint must send data_collection allow even without routing: {no_routing_body}"
        );
        assert!(
            !no_routing_body.contains("\"order\""),
            "no routing order when provider_order is unset: {no_routing_body}"
        );

        // Custom endpoint → routing block suppressed; reasoning_effort is
        // translated for the custom endpoint's model family.
        // The 'test' model matches no family → fallback → xhigh→max.
        let custom_provider =
            OpenAiCompatibleProvider::new("Custom endpoint", "http://localhost:8080/v1", None);
        let custom_body = body(&custom_provider);
        assert!(
            !custom_body.contains("\"provider\""),
            "custom endpoint must not receive the OpenRouter routing block: {custom_body}"
        );
        assert!(
            custom_body.contains("\"reasoning_effort\":\"max\""),
            "custom endpoints must receive the translated reasoning effort (fallback 'test' model: xhigh→max): {custom_body}"
        );
        assert!(
            !custom_body.contains("xhigh"),
            "custom endpoints must not receive the untranslated effort: {custom_body}"
        );
    }

    /// Full model-family × effort translation table for custom endpoints
    /// every family, the fallback, unmapped-effort passthrough,
    /// field-shape dominance for MiMo/MiniMax, family over-capture of older
    /// siblings, case-insensitivity, and first-match-wins ordering. Also
    /// asserts requirement 2: the OpenRouter path stays byte-identical
    /// ("xhigh"/"high" unchanged) for every case.
    #[test]
    fn reasoning_translation_table_for_custom_endpoints() {
        use ReasoningFields::{Effort, Omit, ThinkingEnabled};

        // (label, model, effort, expected fields on a custom endpoint)
        let cases: &[(&str, &str, &str, ReasoningFields)] = &[
            // deepseek (v4): xhigh→max, high→high.
            (
                "deepseek v4 xhigh",
                "deepseek/deepseek-v4-pro-0813",
                "xhigh",
                Effort("max".into()),
            ),
            (
                "deepseek v4 high",
                "deepseek/deepseek-v4-pro-0813",
                "high",
                Effort("high".into()),
            ),
            // Older deepseek v3 still matches the family (documented
            // over-capture — no version-specific handling).
            (
                "deepseek v3 (older sibling, family match)",
                "deepseek/deepseek-v3",
                "xhigh",
                Effort("max".into()),
            ),
            // kimi (k3): xhigh→max, high→high.
            (
                "kimi k3 xhigh",
                "moonshotai/kimi-k3",
                "xhigh",
                Effort("max".into()),
            ),
            (
                "kimi k3 high",
                "moonshotai/kimi-k3",
                "high",
                Effort("high".into()),
            ),
            // glm (5.x): xhigh→max, high→high.
            (
                "glm 5.3 xhigh",
                "zai-org/glm-5.3",
                "xhigh",
                Effort("max".into()),
            ),
            (
                "glm 5.3 high",
                "zai-org/glm-5.3",
                "high",
                Effort("high".into()),
            ),
            // hy3 (Tencent): xhigh→high, high→low.
            (
                "hy3 xhigh",
                "tencent/hunyuan-hy3",
                "xhigh",
                Effort("high".into()),
            ),
            (
                "hy3 high",
                "tencent/hunyuan-hy3",
                "high",
                Effort("low".into()),
            ),
            // muse-spark: xhigh→xhigh, high→high.
            (
                "muse-spark xhigh",
                "meta/muse-spark-1.2",
                "xhigh",
                Effort("xhigh".into()),
            ),
            (
                "muse-spark high",
                "meta/muse-spark-1.2",
                "high",
                Effort("high".into()),
            ),
            // grok: xhigh→xhigh, high→high.
            (
                "grok xhigh",
                "x-ai/grok-4.6",
                "xhigh",
                Effort("xhigh".into()),
            ),
            ("grok high", "x-ai/grok-4.6", "high", Effort("high".into())),
            // gemini 3.7 Flash: xhigh→high, high→medium.
            (
                "gemini xhigh",
                "google/gemini-3.7-flash",
                "xhigh",
                Effort("high".into()),
            ),
            (
                "gemini high",
                "google/gemini-3.7-flash",
                "high",
                Effort("medium".into()),
            ),
            // MiMo: field-shape dominates — thinking:{type:enabled}, no
            // effort — for ANY present effort value (incl. theoretical
            // medium/low/minimal/none).
            ("mimo xhigh", "xiaomi/mimo", "xhigh", ThinkingEnabled),
            (
                "mimo medium (field-shape dominates)",
                "xiaomi/mimo-7b",
                "medium",
                ThinkingEnabled,
            ),
            // MiniMax: omit everything for ANY present effort value (M3
            // thinking is on by default on the OpenAI-compatible endpoint).
            ("minimax m3 xhigh", "minimax/minimax-m3", "xhigh", Omit),
            (
                "minimax m3 low (field-shape dominates)",
                "minimax/minimax-m3",
                "low",
                Omit,
            ),
            // Older minimax m2.x still matches the family (documented
            // over-capture).
            (
                "minimax m2.1 (older sibling, family match)",
                "minimax/minimax-m2.1",
                "high",
                Omit,
            ),
            // Fallback (unknown families, incl. GPT models): xhigh→max,
            // high→high; unmapped values pass through unchanged.
            (
                "fallback gpt xhigh",
                "openai/gpt-5.6",
                "xhigh",
                Effort("max".into()),
            ),
            (
                "fallback high",
                "openai/gpt-4.1",
                "high",
                Effort("high".into()),
            ),
            (
                "fallback medium passes through",
                "openai/gpt-5.6",
                "medium",
                Effort("medium".into()),
            ),
            (
                "fallback none passes through",
                "openai/gpt-5.6",
                "none",
                Effort("none".into()),
            ),
            // Unmapped effort passes through for effort-bearing families too.
            (
                "deepseek medium passes through",
                "deepseek/deepseek-v4",
                "medium",
                Effort("medium".into()),
            ),
            (
                "gemini minimal passes through",
                "google/gemini-3.7-flash",
                "minimal",
                Effort("minimal".into()),
            ),
            // Case-insensitive family detection.
            (
                "DeepSeek case-insensitive",
                "DEEPSEEK/deepseek-v4",
                "xhigh",
                Effort("max".into()),
            ),
            // First-match-wins ordering: 'deepseek-hy3' matches deepseek
            // first (family declaration order).
            (
                "deepseek-hy3 first-match deepseek",
                "deepseek-hy3",
                "xhigh",
                Effort("max".into()),
            ),
        ];

        let custom_endpoint = "http://localhost:8080/v1";
        for (label, model, effort, expected) in cases {
            let actual = reasoning_fields_for_request(custom_endpoint, model, Some(effort));
            assert_eq!(&actual, expected, "{label}: model={model} effort={effort}");
        }

        // No effort (None or empty) → no reasoning fields on either endpoint.
        assert_eq!(
            reasoning_fields_for_request(custom_endpoint, "deepseek/deepseek-v4", None),
            Omit
        );
        assert_eq!(
            reasoning_fields_for_request(custom_endpoint, "deepseek/deepseek-v4", Some("")),
            Omit
        );
        assert_eq!(
            reasoning_fields_for_request(
                crate::config::DEFAULT_PROVIDER_ENDPOINT,
                "deepseek/deepseek-v4",
                None
            ),
            Omit
        );

        // Requirement 2 — the OpenRouter path stays byte-identical: every
        // family/effort passes through unchanged (no translation).
        let or_endpoint = crate::config::DEFAULT_PROVIDER_ENDPOINT;
        for (label, model, effort, _) in cases {
            assert_eq!(
                reasoning_fields_for_request(or_endpoint, model, Some(effort)),
                Effort(effort.to_string()),
                "OpenRouter must pass effort through unchanged: {label} model={model}"
            );
        }
    }

    #[test]
    #[expect(clippy::type_complexity)]
    fn resolve_tool_call_name_cases() {
        let cases: &[(&str, Option<&str>, Option<&str>, Option<&str>)] = &[
            (
                "function wins",
                Some("func_name"),
                Some("direct_name"),
                Some("func_name"),
            ),
            (
                "direct fallback",
                None,
                Some("direct_name"),
                Some("direct_name"),
            ),
            ("both none", None, None, None),
            (
                "empty function name",
                Some(""),
                Some("direct_name"),
                Some("direct_name"),
            ),
            (
                "empty direct name",
                Some("func_name"),
                Some(""),
                Some("func_name"),
            ),
            ("both empty", Some(""), Some(""), None),
        ];
        for (name, fn_name, direct_name, expected) in cases {
            assert_eq!(
                resolve_tool_call_name(*fn_name, *direct_name),
                expected.map(String::from),
                "{name}",
            );
        }
    }

    #[test]
    #[expect(clippy::type_complexity)]
    fn resolve_tool_call_arguments_cases() {
        let cases: &[(&str, Option<&str>, Option<&str>, Option<&str>, Option<&str>)] = &[
            (
                "function wins",
                Some(r#"{"key":"func_val"}"#),
                Some(r#"{"key":"direct_val"}"#),
                None,
                Some(r#"{"key":"func_val"}"#),
            ),
            (
                "direct fallback",
                None,
                Some(r#"{"key":"val"}"#),
                None,
                Some(r#"{"key":"val"}"#),
            ),
            ("both none", None, None, None, None),
            (
                "empty function args",
                Some(""),
                Some(r#"{"key":"val"}"#),
                None,
                Some(r#"{"key":"val"}"#),
            ),
            (
                "empty direct args",
                Some(r#"{"key":"val"}"#),
                Some(""),
                None,
                Some(r#"{"key":"val"}"#),
            ),
            ("both empty", Some(""), Some(""), None, None),
            (
                "parameters fallback",
                None,
                None,
                Some(r#"{"command":"pwd"}"#),
                Some(r#"{"command":"pwd"}"#),
            ),
            (
                "string fields empty with parameters",
                Some(""),
                Some(""),
                Some(r#"{"command":"ls"}"#),
                Some(r#"{"command":"ls"}"#),
            ),
            (
                "function wins over parameters",
                Some(r#"{"key":"val"}"#),
                None,
                Some(r#"{"query":"test"}"#),
                Some(r#"{"key":"val"}"#),
            ),
        ];
        for (name, fn_args, direct_args, params_json, expected) in cases {
            let params = params_json.map(|j| serde_json::from_str(j).expect(name));
            assert_eq!(
                resolve_tool_call_arguments(*fn_args, *direct_args, params.as_ref()),
                expected.map(String::from),
                "{name}",
            );
        }
    }

    // ----------------------------------------------------------
    // URL endpoint tests
    // ----------------------------------------------------------

    #[test]
    fn parse_native_response_preserves_tool_call_id() {
        let message = ResponseMessage {
            content: None,
            tool_calls: Some(vec![ApiToolCall {
                id: Some("call_123".to_string()),
                kind: Some("function".to_string()),
                function: Some(ApiToolCallFunction {
                    name: Some("shell".to_string()),
                    arguments: Some(r#"{"command":"pwd"}"#.to_string()),
                }),
                name: None,
                arguments: None,
                parameters: None,
            }]),
            reasoning_content: None,
            reasoning: None,
            reasoning_details: None,
        };

        let parsed =
            OpenAiCompatibleProvider::parse_native_response(message, None, None, None, None);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].id, "call_123");
        assert_eq!(parsed.tool_calls[0].name, "shell");
    }

    #[test]
    fn convert_messages_for_native_maps_tool_result_payload() {
        let input = vec![ChatMessage::tool_result("call_abc", "done")];

        let converted = OpenAiCompatibleProvider::convert_messages_for_native(&input);
        assert_eq!(converted[0].tool_call_id.as_deref(), Some("call_abc"));
        assert!(matches!(
            converted[0].content.as_ref(),
            Some(MessageContent::Text(value)) if value == "done"
        ));
    }

    #[test]
    fn convert_messages_for_native_converts_user_image_markers_to_image_parts() {
        let uri = tiny_png_data_uri();
        let input = vec![ChatMessage::user(format!(
            "System primer [IMAGE:{uri}] user turn"
        ))];

        let converted = OpenAiCompatibleProvider::convert_messages_for_native(&input);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
        assert!(matches!(
            converted[0].content.as_ref(),
            Some(MessageContent::Parts(parts))
                if parts.iter().any(|p| matches!(
                    p,
                    MessagePart::ImageUrl { image_url }
                        if image_url.url == uri
                ))
        ));
    }

    #[test]
    fn effective_content_optional_never_promotes_reasoning() {
        // Empty content with reasoning → stays empty (the reasoning-only-stop
        // class is recovered by the agent loop, never promoted to visible text).
        let json = r#"{"choices":[{"message":{"content":"","reasoning_content":"Thinking output here"}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0]
                .message
                .effective_content_optional()
                .unwrap_or_default(),
            ""
        );
        // Null content, reasoning present → stays empty
        let json =
            r#"{"choices":[{"message":{"content":null,"reasoning_content":"Fallback text"}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0]
                .message
                .effective_content_optional()
                .unwrap_or_default(),
            ""
        );
        // Reasoning present but no content field at all → stays empty
        let json = r#"{"choices":[{"message":{"reasoning_content":"Only thinking"}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0]
                .message
                .effective_content_optional()
                .unwrap_or_default(),
            ""
        );
        // Normal content, reasoning present → uses content (ignores reasoning)
        let json = r#"{"choices":[{"message":{"content":"Normal response","reasoning_content":"Should be ignored"}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0]
                .message
                .effective_content_optional()
                .unwrap_or_default(),
            "Normal response"
        );
        // Content only think tags → empty (think tags are reasoning, not content)
        let json = r#"{"choices":[{"message":{"content":"<think>secret</think>","reasoning_content":"Fallback text"}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0]
                .message
                .effective_content_optional()
                .unwrap_or_default(),
            ""
        );
        // Think tags plus visible text → visible text only
        let json =
            r#"{"choices":[{"message":{"content":"<think>secret</think>\nVisible answer"}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0]
                .message
                .effective_content_optional()
                .unwrap_or_default(),
            "Visible answer"
        );
        // Both absent → empty
        let json = r#"{"choices":[{"message":{}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0]
                .message
                .effective_content_optional()
                .unwrap_or_default(),
            ""
        );
        // Normal model without reasoning_content
        let json = r#"{"choices":[{"message":{"content":"Hello from Venice!"}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.choices[0].message.reasoning_content.is_none());
        assert_eq!(
            resp.choices[0]
                .message
                .effective_content_optional()
                .unwrap_or_default(),
            "Hello from Venice!"
        );
    }

    #[tokio::test]
    async fn warmup_without_key_attempts_connection() {
        let provider = OpenAiCompatibleProvider::new("test", "http://127.0.0.1:1", None);
        let result = provider.warmup().await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            !err_msg.contains("API key not set"),
            "should not get credential error, got: {err_msg}"
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Native tool calling tests
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn parse_image_markers_extracts_multiple_markers() {
        let uri = tiny_png_data_uri();
        let input = format!("Check this [IMAGE:{uri}] and this [IMAGE:https://example.com/b.jpg]");
        let (cleaned, refs) = parse_image_markers(&input);

        assert_eq!(cleaned, "Check this  and this");
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0], uri);
        assert_eq!(refs[1], "https://example.com/b.jpg");
    }

    #[test]
    fn parse_image_markers_keeps_invalid_empty_marker() {
        let input = "hello [IMAGE:] world";
        let (cleaned, refs) = parse_image_markers(input);

        assert_eq!(cleaned, "hello [IMAGE:] world");
        assert!(refs.is_empty());
    }

    /// Stripping native `[IMAGE:]` payloads from history messages leaves only
    /// the text portion — those markers never reach the provider as text; the
    /// marker scan (and native image-part conversion) consumes them first.
    #[test]
    fn parse_image_markers_strips_markers_leaving_caption() {
        let uri = tiny_png_data_uri();
        let input = format!("[IMAGE:{uri}]\n\nDescribe this screenshot");
        let (cleaned, refs) = parse_image_markers(&input);
        assert_eq!(cleaned, "Describe this screenshot");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0], uri);
    }

    /// An image-only message (no caption) should produce an empty string after
    /// marker stripping, so callers can drop it from history.
    #[test]
    fn parse_image_markers_image_only_message_becomes_empty() {
        let uri = tiny_png_data_uri();
        let input = format!("[IMAGE:{uri}]");
        let (cleaned, refs) = parse_image_markers(&input);
        assert!(
            cleaned.is_empty(),
            "expected empty string, got: {cleaned:?}"
        );
        assert_eq!(refs.len(), 1);
    }

    /// Non‑IMAGE markers (AUDIO, VIDEO) and non-native IMAGE markers (file
    /// paths, prose placeholders) are preserved verbatim in the cleaned
    /// output. This test covers the mixed case to prevent regression of the
    /// preservation behaviour.
    #[test]
    fn parse_image_markers_preserves_audio_and_video_markers() {
        let input =
            "[AUDIO:/tmp/sound.mp3] Listen to this [VIDEO:/tmp/clip.mp4] and [IMAGE:/tmp/img.png]";
        let (cleaned, refs) = parse_image_markers(input);

        assert_eq!(
            cleaned,
            "[AUDIO:/tmp/sound.mp3] Listen to this [VIDEO:/tmp/clip.mp4] and [IMAGE:/tmp/img.png]"
        );
        assert!(refs.is_empty());
    }

    /// Prose placeholders written as `[IMAGE:...]` / `[IMAGE:path]` / ellipsis
    /// data-URIs must stay in the text — they are not image payloads and must
    /// not become `image_url` parts (DeepSeek rejects them as unsupported).
    #[test]
    fn parse_image_markers_keeps_prose_placeholders() {
        let input = concat!(
            "tool, command, [IMAGE:...] marker, or [IMAGE:path] syntax, ",
            "or [IMAGE:data:image/...;base64,...], ",
            "or [IMAGE:data:image/jpeg;base64,...], ",
            "or [IMAGE:data:image/png;base64,abcd]"
        );
        let (cleaned, refs) = parse_image_markers(input);
        assert_eq!(cleaned, input);
        assert!(refs.is_empty());
    }

    /// A truncated PNG (valid magic + IHDR, no IDAT) must stay in the text —
    /// DeepSeek rejects it as an unsupported/invalid image.
    #[test]
    fn parse_image_markers_rejects_truncated_raster() {
        let uri = tiny_png_data_uri();
        let b64 = uri
            .strip_prefix("data:image/png;base64,")
            .expect("tiny png uri");
        let bytes = STANDARD.decode(b64).expect("tiny png base64");
        let truncated = &bytes[..bytes.len().min(33)];
        let input = format!(
            "[IMAGE:data:image/png;base64,{}]",
            STANDARD.encode(truncated)
        );
        let (cleaned, refs) = parse_image_markers(&input);
        assert_eq!(cleaned, input);
        assert!(refs.is_empty());
    }

    #[test]
    fn to_message_content_converts_image_markers_to_openai_parts() {
        let uri = tiny_png_data_uri();
        let content = format!("Describe this\n\n[IMAGE:{uri}]");
        let value = serde_json::to_value(to_message_content(ChatRole::User, &content)).unwrap();
        let parts = value
            .as_array()
            .expect("multimodal content should be an array");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "Describe this");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], uri);
    }

    #[test]
    fn to_message_content_keeps_prose_image_markers_as_text() {
        let content = "List any existing mechanism (tool, command, [IMAGE:...] marker, data URI)";
        let value = serde_json::to_value(to_message_content(ChatRole::User, content)).unwrap();
        assert_eq!(value, serde_json::json!(content));
    }

    #[test]
    fn to_message_content_mixes_prose_marker_with_native_payload() {
        let uri = tiny_png_data_uri();
        let content = format!("see [IMAGE:...] and [IMAGE:{uri}]");
        let value = serde_json::to_value(to_message_content(ChatRole::User, &content)).unwrap();
        let parts = value.as_array().expect("mixed content should be an array");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "see [IMAGE:...] and");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], uri);
    }

    #[test]
    fn to_message_content_keeps_plain_text_for_non_user_roles() {
        let value = serde_json::to_value(to_message_content(
            ChatRole::System,
            "You are a helpful assistant.",
        ))
        .unwrap();
        assert_eq!(value, serde_json::json!("You are a helpful assistant."));
    }

    #[test]
    fn request_serializes_with_tools() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        })];

        let req = ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![NativeMessage::user("What is the weather?")],
            max_tokens: Some(32000),
            tools: Some(tools),
            tool_choice: Some("auto".to_string()),
            extra: serde_json::Map::new(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"tools\""));
        assert!(json.contains("get_weather"));
        assert!(json.contains("\"tool_choice\":\"auto\""));
    }

    #[test]
    fn response_with_tool_calls_deserializes() {
        let json = r#"{
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"London\"}"
                        }
                    }]
                }
            }]
        }"#;

        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        let msg = &resp.choices[0].message;
        assert!(msg.content.is_none());
        let tool_calls = msg.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
        assert_eq!(
            tool_calls[0]
                .function
                .as_ref()
                .unwrap()
                .arguments
                .as_deref(),
            Some("{\"location\":\"London\"}")
        );
    }

    #[test]
    fn response_with_multiple_tool_calls() {
        let json = r#"{
            "choices": [{
                "message": {
                    "content": "I'll check both.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\":\"London\"}"
                            }
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "get_time",
                                "arguments": "{\"timezone\":\"UTC\"}"
                            }
                        }
                    ]
                }
            }]
        }"#;

        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        let msg = &resp.choices[0].message;
        assert_eq!(msg.content.as_deref(), Some("I'll check both."));
        let tool_calls = msg.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
        assert_eq!(
            tool_calls[1].function.as_ref().unwrap().name.as_deref(),
            Some("get_time")
        );
    }

    #[test]
    fn response_with_no_tool_calls_has_empty_vec() {
        let json = r#"{"choices":[{"message":{"content":"Just text, no tools."}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        let msg = &resp.choices[0].message;
        assert_eq!(msg.content.as_deref(), Some("Just text, no tools."));
        assert!(msg.tool_calls.is_none());
    }
    #[test]
    fn api_response_parses_usage() {
        let json = r#"{
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 150, "completion_tokens": 60}
        }"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(150));
        assert_eq!(usage.completion_tokens, Some(60));
    }

    #[test]
    fn cache_tokens_normalized_for_both_provider_shapes() {
        // OpenRouter: only cached_tokens reported — miss is computed.
        let (cached, miss) = normalize_cache_tokens(Some(90), None, None, Some(150));
        assert_eq!(cached, Some(90));
        assert_eq!(miss, Some(60));
        // DeepSeek: both sides native — no computation needed.
        let (cached, miss) = normalize_cache_tokens(None, Some(90), Some(60), Some(150));
        assert_eq!(cached, Some(90));
        assert_eq!(miss, Some(60));
        // Unknown prompt total — miss stays unknown.
        let (cached, miss) = normalize_cache_tokens(Some(90), None, None, None);
        assert_eq!(cached, Some(90));
        assert_eq!(miss, None);
    }

    #[test]
    fn api_response_parses_cached_tokens() {
        // OpenRouter-shaped usage with prompt_tokens_details.cached_tokens.
        let json = r#"{
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 60,
                "prompt_tokens_details": {"cached_tokens": 90}
            }
        }"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, Some(90));
    }

    #[test]
    fn api_response_parses_without_usage() {
        let json = r#"{"choices": [{"message": {"content": "Hello"}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.usage.is_none());
    }

    #[test]
    fn api_response_parses_cost_fingerprint_and_upstream_provider() {
        // OpenRouter envelope with billed cost, system_fingerprint and the
        // top-level serving provider.
        let json = r#"{
            "choices": [{"message": {"content": "Hello"}}],
            "provider": "DeepSeek",
            "system_fingerprint": "fp_44709d6fcb",
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 60,
                "cost": 0.0012,
                "cost_details": {
                    "upstream_inference_prompt_cost": 0.0008,
                    "upstream_inference_completions_cost": 0.0004
                }
            }
        }"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.system_fingerprint.as_deref(), Some("fp_44709d6fcb"));
        assert_eq!(resp.provider.as_deref(), Some("DeepSeek"));
        let usage = resp.usage.unwrap();
        assert!((usage.cost.expect("cost") - 0.0012).abs() < 1e-12);
        // Parse→re-serialize normalizes key order (BTreeMap) — the stored
        // TEXT is reference-only; SQL slicing uses cost REAL.
        assert_eq!(
            usage.cost_details.as_ref().map(serde_json::Value::to_string),
            Some(
                r#"{"upstream_inference_completions_cost":0.0004,"upstream_inference_prompt_cost":0.0008}"#
                    .to_string()
            )
        );
        assert_eq!(
            usage
                .cost_details
                .expect("cost_details")
                .pointer("/upstream_inference_prompt_cost")
                .and_then(serde_json::Value::as_f64),
            Some(0.0008)
        );
    }

    #[test]
    fn upstream_provider_present_on_cache_hit() {
        // Cache hits strip openrouter_metadata, but the top-level `provider`
        // field survives — the eviction/backend-switch rows this telemetry
        // exists to attribute must not be NULL.
        let json = r#"{
            "choices": [{"message": {"content": "cached answer"}}],
            "provider": "Friendli",
            "system_fingerprint": "fp_44709d6fcb",
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 8},
                "cost": 0.0001
            }
        }"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.provider.as_deref(), Some("Friendli"));
        // Missing/absent provider stays None (NULL) — non-OpenRouter endpoints.
        let resp: ApiChatResponse =
            serde_json::from_str(r#"{"choices":[{"message":{"content":"x"}}]}"#).unwrap();
        assert!(resp.provider.is_none());
    }

    #[test]
    fn wrong_typed_telemetry_fields_do_not_break_envelope_parse() {
        // Provider shape drift (system_fingerprint as number, cost as string,
        // provider as number) must yield NULL — not a parse failure
        // that would retry an otherwise-successful response.
        let json = r#"{
            "choices": [{"message": {"content": "Hello"}}],
            "provider": 42,
            "system_fingerprint": 42,
            "usage": {"cost": "not-a-number", "cost_details": [1, 2]}
        }"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.system_fingerprint.is_none());
        assert!(resp.provider.is_none());
        let usage = resp.usage.unwrap();
        assert!(usage.cost.is_none());
        assert!(usage.cost_details.is_some(), "Value accepts any JSON");
        // Missing fields stay None too.
        let resp: ApiChatResponse =
            serde_json::from_str(r#"{"choices":[{"message":{"content":"x"}}]}"#).unwrap();
        assert!(resp.system_fingerprint.is_none());
        assert!(resp.provider.is_none());
        assert!(resp.usage.is_none());
    }

    #[test]
    fn pathological_cost_details_number_does_not_break_envelope_parse() {
        // serde_json rejects `1e999` as an out-of-range number — without the
        // `opt_field` gate on cost_details this would fail the whole envelope
        // parse and retry on byte-identical bytes.
        let json = r#"{
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"cost_details": {"upstream_inference_prompt_cost": 1e999}}
        }"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        let usage = resp.usage.unwrap();
        assert!(usage.cost_details.is_none());
        assert!(usage.cost.is_none());
    }

    // ─────────────────────────────────────────────────────────────────────
    // reasoning_content pass-through tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn parse_native_response_captures_reasoning_content() {
        let message = ResponseMessage {
            content: Some("answer".to_string()),
            reasoning_content: Some("thinking step".to_string()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: Some(vec![ApiToolCall {
                id: Some("call_1".to_string()),
                kind: Some("function".to_string()),
                function: Some(ApiToolCallFunction {
                    name: Some("shell".to_string()),
                    arguments: Some(r#"{"cmd":"ls"}"#.to_string()),
                }),
                name: None,
                arguments: None,
                parameters: None,
            }]),
        };

        let parsed =
            OpenAiCompatibleProvider::parse_native_response(message, None, None, None, None);
        let rc = parsed
            .reasoning
            .as_ref()
            .and_then(|r| r.reasoning_content.clone());
        assert_eq!(rc.as_deref(), Some("thinking step"));
        assert_eq!(parsed.text.as_deref(), Some("answer"));
        assert_eq!(parsed.tool_calls.len(), 1);
    }

    #[test]
    fn parse_native_response_none_reasoning_content_for_normal_model() {
        let message = ResponseMessage {
            content: Some("hello".to_string()),
            reasoning_content: None,
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
        };

        let parsed =
            OpenAiCompatibleProvider::parse_native_response(message, None, None, None, None);
        assert!(parsed.reasoning.is_none());
        assert_eq!(parsed.text.as_deref(), Some("hello"));
    }

    #[test]
    fn convert_messages_for_native_round_trips_reasoning_content() {
        // Simulate stored assistant history JSON that includes reasoning_content
        let history_json = serde_json::json!({
            "content": "I will check",
            "tool_calls": [{
                "id": "tc_1",
                "name": "shell",
                "arguments": "{\"cmd\":\"ls\"}"
            }],
            "reasoning_content": "Let me think about this..."
        });

        let messages = vec![ChatMessage::assistant(history_json.to_string())];
        let native = OpenAiCompatibleProvider::convert_messages_for_native(&messages);
        assert_eq!(native.len(), 1);
        assert_eq!(native[0].role, "assistant");
        assert_eq!(
            native[0].reasoning_content.as_deref(),
            Some("Let me think about this...")
        );
        assert!(native[0].tool_calls.is_some());
    }

    #[test]
    fn convert_messages_for_native_no_reasoning_content_when_absent() {
        // Normal model history without reasoning_content key
        let history_json = serde_json::json!({
            "content": "I will check",
            "tool_calls": [{
                "id": "tc_1",
                "name": "shell",
                "arguments": "{\"cmd\":\"ls\"}"
            }]
        });

        let messages = vec![ChatMessage::assistant(history_json.to_string())];
        let native = OpenAiCompatibleProvider::convert_messages_for_native(&messages);
        assert_eq!(native.len(), 1);
        assert!(native[0].reasoning_content.is_none());
    }

    #[test]
    fn convert_messages_for_native_synthesizes_reasoning_content_from_details_for_tool_calls() {
        let details = serde_json::json!([
            {"type": "reasoning.text", "text": "from details", "format": "x", "index": 0}
        ]);
        let history_json = serde_json::json!({
            "content": "I will check",
            "tool_calls": [{
                "id": "tc_1",
                "name": "shell",
                "arguments": "{\"cmd\":\"ls\"}"
            }],
            "reasoning_details": details.clone(),
        });

        let messages = vec![ChatMessage::assistant(history_json.to_string())];
        let native = OpenAiCompatibleProvider::convert_messages_for_native(&messages);
        assert_eq!(native.len(), 1);
        assert_eq!(native[0].reasoning_content.as_deref(), Some("from details"));
        assert_eq!(native[0].reasoning_details.as_ref(), Some(&details));
    }
}
