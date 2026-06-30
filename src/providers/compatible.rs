//! Generic OpenAI-compatible provider.
//! Most LLM APIs follow the same `/v1/chat/completions` format.
//! This module provides a single implementation that works for all of them.

use crate::providers::compatible_streaming::drain_sse_into_channel;
use crate::providers::reasoning_roundtrip;
use crate::providers::{ensure_chat_completions_url, provider_routing_json};
use crate::util::error::HttpError;
use crate::{
    ChatMessage, ChatRequest as ProviderChatRequest, ChatResponse as ProviderChatResponse,
    MessageRole, Provider, ProviderUsage, Reasoning, StreamError, StreamEvent, StreamResult,
    ToolCall as ProviderToolCall, ToolSpec,
};
use async_trait::async_trait;
use futures_util::{StreamExt, stream};
use regex::Regex;
use reqwest::{
    Client, RequestBuilder,
    header::{HeaderMap, HeaderValue},
};
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use std::sync::OnceLock;

/// A provider that speaks the OpenAI-compatible chat completions API.
/// Used by: Venice, Vercel AI Gateway, Cloudflare AI Gateway, Moonshot,
/// Synthetic, `OpenCode` Zen, `OpenCode` Go, `Z.AI`, `GLM`, `MiniMax`, Bedrock, Qianfan, Groq, Mistral, `xAI`, etc.
pub struct OpenAiCompatibleProvider {
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

    pub(crate) fn http_client(&self) -> &Client {
        self.http_client.get_or_init(|| {
            let mut builder = Client::builder()
                .timeout(std::time::Duration::from_secs(self.timeout_secs))
                .connect_timeout(std::time::Duration::from_secs(10));

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
                            tracing::warn!(
                                header = key,
                                "Skipping invalid extra header name or value"
                            );
                        }
                    }
                }
                builder = builder.default_headers(headers);
            }

            builder.build().unwrap_or_else(|error| {
                tracing::warn!("Failed to build custom client: {error}");
                Client::new()
            })
        })
    }

    fn requires_tool_stream(&self) -> bool {
        let host_requires_tool_stream = reqwest::Url::parse(&self.base_url)
            .ok()
            .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
            .is_some_and(|host| host == "api.z.ai" || host.ends_with(".z.ai"));

        host_requires_tool_stream || matches!(self.name.as_str(), "zai" | "z.ai")
    }

    fn tool_stream_for_tools(&self, has_tools: bool) -> Option<bool> {
        if has_tools && self.requires_tool_stream() {
            Some(true)
        } else {
            None
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<NativeMessage>,
    temperature: f64,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_stream: Option<bool>,
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
}

#[derive(Debug, Deserialize)]
struct UsageInfo {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

/// Remove `<think>...</think>` blocks from model output.
/// Some reasoning models (e.g. `MiniMax`) embed their chain-of-thought inline
/// in the `content` field rather than a separate `reasoning_content` field.
/// The resulting `<think>` tags must be stripped before returning to the user.
pub(crate) fn strip_think_tags(s: &str) -> String {
    static THINK_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?s)<think>.*?</think>|<think>.*$").expect("think tag regex must compile")
    });
    THINK_RE.replace_all(s, "").trim().to_string()
}

#[derive(Debug, Deserialize, Serialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    /// Reasoning/thinking models (e.g. Qwen3, GLM-4) may return their output
    /// in `reasoning_content` instead of `content`. Used as automatic fallback.
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
    /// Extract text content, falling back to `reasoning_content` when `content`
    /// is missing or empty. Reasoning/thinking models (Qwen3, GLM-4, etc.)
    /// often return their output solely in `reasoning_content`.
    /// Strips `<think>...</think>` blocks that some models (e.g. `MiniMax`) embed
    /// inline in `content` instead of using a separate field.
    fn effective_content_optional(&self) -> Option<String> {
        if let Some(content) = self.content.as_ref().filter(|c| !c.is_empty()) {
            let stripped = strip_think_tags(content);
            if !stripped.is_empty() {
                return Some(stripped);
            }
        }

        crate::util::merged_reasoning_string(
            self.reasoning_content
                .as_ref()
                .map(|c| c.trim().to_string())
                .filter(|c| !c.is_empty()),
            self.reasoning
                .as_ref()
                .map(|c| c.trim().to_string())
                .filter(|c| !c.is_empty()),
        )
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

impl ApiToolCall {
    /// Extract function name with fallback logic for various provider formats
    fn function_name(&self) -> Option<String> {
        // Standard OpenAI format: tool_calls[].function.name
        if let Some(ref func) = self.function
            && let Some(ref name) = func.name
        {
            return Some(name.clone());
        }
        // Fallback: direct name field
        self.name.clone()
    }

    /// Extract arguments with fallback logic and type conversion
    fn function_arguments(&self) -> Option<String> {
        // Standard OpenAI format: tool_calls[].function.arguments (string)
        if let Some(ref func) = self.function
            && let Some(ref args) = func.arguments
        {
            return Some(args.clone());
        }
        // Fallback: direct arguments field
        if let Some(ref args) = self.arguments {
            return Some(args.clone());
        }
        // Compatibility: Some providers return parameters as object instead of string
        if let Some(ref params) = self.parameters {
            return serde_json::to_string(params).ok();
        }
        None
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

/// Parse `[IMAGE:path]` markers from content, returning cleaned text and extracted paths.
///
/// Uses the shared [`MEDIA_MARKER_RE`](crate::util::MEDIA_MARKER_RE) to find markers.
/// Non‑IMAGE markers (e.g. `[AUDIO:…]`) are left untouched in the cleaned text.
/// Empty `[IMAGE:]` markers are preserved verbatim.
#[must_use]
fn parse_image_markers(content: &str) -> (String, Vec<String>) {
    let mut refs: Vec<String> = Vec::new();
    let mut cleaned = String::with_capacity(content.len());
    let mut last_end = 0;

    for caps in crate::util::MEDIA_MARKER_RE.captures_iter(content) {
        let m = caps.get(0).expect("MEDIA_MARKER_RE: expected full match");

        // Emit text before this match.
        cleaned.push_str(&content[last_end..m.start()]);

        if caps
            .name("kind")
            .expect("MEDIA_MARKER_RE: expected 'kind' group")
            .as_str()
            == "IMAGE"
        {
            let path = caps
                .name("path")
                .expect("MEDIA_MARKER_RE: expected 'path' group")
                .as_str()
                .trim();
            refs.push(path.to_string());
            // IMAGE markers are stripped — don't emit anything.
        } else {
            // AUDIO/VIDEO markers are preserved verbatim.
            cleaned.push_str(m.as_str());
        }

        last_end = m.end();
    }

    // Emit remaining text after the last match.
    cleaned.push_str(&content[last_end..]);

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
/// When `allow_user_image_parts` is true and the role is [`MessageRole::User`], image markers
/// (e.g. `[IMAGE:data:image/png;base64,...]`) are parsed into [`MessagePart::ImageUrl`]
/// entries alongside the cleaned text. Otherwise the raw content is returned as
/// [`MessageContent::Text`].
pub(crate) fn to_message_content(
    role: MessageRole,
    content: &str,
    allow_user_image_parts: bool,
) -> MessageContent {
    if role != MessageRole::User || !allow_user_image_parts {
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
    /// Check if a tool name is valid for OpenAI-compatible APIs.
    /// Must match `^[a-zA-Z0-9_-]{1,64}$`.
    fn is_valid_tool_name(name: &str) -> bool {
        !name.is_empty()
            && name.len() <= 64
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    }

    fn convert_tool_specs(tools: Option<&[ToolSpec]>) -> Option<Vec<serde_json::Value>> {
        let items = tools?;
        let converted: Vec<_> = items
            .iter()
            .filter(|t| Self::is_valid_tool_name(&t.name))
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

    fn convert_messages_for_native(
        messages: &[ChatMessage],
        allow_user_image_parts: bool,
    ) -> Vec<NativeMessage> {
        messages
            .iter()
            .map(|message| {
                let decoded = crate::session::decode_native_history_message(message);
                let Some(parts) =
                    decoded.map(crate::session::DecodedNativeHistoryMessage::into_parts)
                else {
                    return NativeMessage {
                        role: message.role.to_string(),
                        content: Some(to_message_content(
                            message.role,
                            &message.content,
                            allow_user_image_parts,
                        )),
                        tool_call_id: None,
                        tool_calls: None,
                        reasoning: None,
                        reasoning_content: None,
                        reasoning_details: None,
                    };
                };
                let has_tool_calls = parts.tool_calls.as_ref().is_some_and(|c| !c.is_empty());
                let (r_plain, r_content, r_details) =
                    reasoning_roundtrip::native_reasoning_triple_for_replay(
                        parts.reasoning.as_ref(),
                        has_tool_calls,
                    );
                let tool_calls = parts.tool_calls.map(|tc| {
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
                let has_reasoning = r_content.is_some() || r_plain.is_some() || r_details.is_some();
                let content = match (&parts.content, has_reasoning, has_tool_calls) {
                    (Some(s), _, _) => Some(MessageContent::Text(s.clone())),
                    (None, true, true) => Some(MessageContent::Null),
                    (None, true, false) => Some(MessageContent::Text(String::new())),
                    (None, false, _) => None,
                };
                NativeMessage {
                    role: parts.role,
                    content,
                    tool_call_id: parts.tool_call_id,
                    tool_calls,
                    reasoning: r_plain,
                    reasoning_content: r_content,
                    reasoning_details: r_details,
                }
            })
            .collect()
    }
}

/// Parse tool-call arguments JSON with repair fallback and fallback to empty object on parse failure.
#[must_use]
pub(crate) fn parse_tool_call_arguments(name: &str, arguments: &str) -> serde_json::Value {
    serde_json::from_str(arguments).unwrap_or_else(|parse_err| {
        // Attempt JSON repair before falling back to empty object
        if let Ok(repaired) = jsonrepair_rs::jsonrepair(arguments)
            && let Ok(value) = serde_json::from_str(&repaired)
        {
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

impl OpenAiCompatibleProvider {
    fn parse_native_response(message: ResponseMessage) -> ProviderChatResponse {
        let text = message.effective_content_optional();
        let reasoning = Reasoning::from_optional_parts(
            message.reasoning.clone(),
            message.reasoning_content.clone(),
            message.reasoning_details.clone(),
        );
        let tool_calls = message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .filter_map(|tc| {
                let name = tc.function_name()?;
                let arguments = tc.function_arguments().unwrap_or("{}".to_string());
                let parsed_arguments = parse_tool_call_arguments(&name, &arguments);
                Some(ProviderToolCall {
                    id: tc.id.unwrap_or_else(crate::generate_id),
                    name,
                    arguments: parsed_arguments,
                })
            })
            .collect::<Vec<_>>();

        ProviderChatResponse {
            text,
            tool_calls,
            usage: None,
            reasoning,
        }
    }

    /// Build the HTTP request for both synchronous [`Provider::chat`] and streaming [`Provider::stream_chat`] calls.
    fn build_chat_request_raw(
        &self,
        request: &ProviderChatRequest,
        stream: bool,
    ) -> RequestBuilder {
        let native =
            Self::convert_messages_for_native(&request.messages, request.allow_image_parts);
        let tool_specs = Self::convert_tool_specs(request.tools.as_deref());

        let has_tools = tool_specs.as_ref().is_some_and(|t| !t.is_empty());
        let mut extra = serde_json::Map::new();

        // Provider routing — per-request values only; no global fallback.
        // If provider_order is present and non-empty, build the routing block.
        if let Some(order) = &request.provider_order
            && let Some(routing) =
                provider_routing_json(order, request.provider_allow_fallbacks.unwrap_or(false))
        {
            extra.insert("provider".to_string(), routing);
        }

        // Reasoning effort
        if let Some(effort) = request
            .reasoning_effort
            .as_deref()
            .filter(|e| !e.is_empty())
        {
            extra.insert("reasoning_effort".to_string(), serde_json::json!(effort));
        }

        let payload = ChatCompletionRequest {
            model: request.model.clone(),
            messages: native,
            temperature: f64::from(request.temperature),
            max_tokens: 32000,
            stream: Some(stream),
            tool_stream: self.tool_stream_for_tools(has_tools),
            tool_choice: tool_specs.as_ref().map(|_| "auto".to_string()),
            tools: tool_specs,
            extra,
        };

        let url = ensure_chat_completions_url(&self.base_url);
        let builder = self.http_client().post(url).json(&payload);
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

#[async_trait]
impl Provider for OpenAiCompatibleProvider {
    async fn chat(&self, request: ProviderChatRequest) -> anyhow::Result<ProviderChatResponse> {
        let model = request.model.clone();
        let req_builder = self.build_chat_request_raw(&request, false);

        let response = crate::shutdown::race_shutdown(req_builder.send())
            .await
            .map_err(|_| anyhow::anyhow!("shutdown during request"))?
            .map_err(|e| {
                anyhow::Error::from(e).context(format!("{} transport error", self.name))
            })?;

        if !response.status().is_success() {
            let http_err = HttpError::from_response(response, &self.name).await;
            return Err(anyhow::Error::from(http_err));
        }

        let body = crate::shutdown::race_shutdown(response.text())
            .await
            .map_err(|_| anyhow::anyhow!("shutdown during response body read"))?
            .map_err(|e| {
                anyhow::Error::from(e).context(format!("{} error reading response body", self.name))
            })?;

        let body_value: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
            anyhow::anyhow!(
                "{} chat completions JSON parse error: {e}; body={}",
                self.name,
                body
            )
        })?;

        let native_response: ApiChatResponse = serde_json::from_value(body_value)
            .map_err(|e| anyhow::anyhow!("{} chat completions response shape: {e}", self.name))?;

        let usage = native_response.usage.map(|u| ProviderUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cached_input_tokens: None,
        });
        let message = native_response
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message)
            .ok_or_else(|| anyhow::anyhow!("No response from {}", self.name))?;

        let mut result = Self::parse_native_response(message);
        result.usage = usage;

        if !result.tool_calls.is_empty() && result.reasoning.is_none() {
            tracing::debug!(
                provider = %self.name,
                model,
                "tool turn: parsed response has no reasoning fields",
            );
        }

        Ok(result)
    }

    fn stream_chat(
        &self,
        request: ProviderChatRequest,
    ) -> stream::BoxStream<'static, StreamResult<StreamEvent>> {
        let req_builder = self.build_chat_request_raw(&request, true);

        let req_builder = req_builder.header("Accept", "text/event-stream");

        let (tx, rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(100);

        let provider_name = self.name.clone();

        tokio::spawn(async move {
            let response = match crate::shutdown::race_shutdown(req_builder.send()).await {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    let _ = tx.send(Err(StreamError::Http(e.to_string()))).await;
                    return;
                }
                Err(_) => return,
            };

            if !response.status().is_success() {
                let http_err = HttpError::from_response(response, &provider_name).await;
                let _ = tx
                    .send(Err(StreamError::Provider(http_err.to_string())))
                    .await;
                return;
            }

            drain_sse_into_channel(response, &tx).await;
        });

        stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|event| (event, rx))
        })
        .boxed()
    }

    async fn warmup(&self) -> anyhow::Result<()> {
        // Hit the chat completions URL with a GET to establish the connection pool.
        // The server will likely return 405 Method Not Allowed, which is fine -
        // the goal is TLS handshake and HTTP/2 negotiation.
        let url = ensure_chat_completions_url(&self.base_url);
        let builder = self.http_client().get(&url);
        let _ = self.attach_auth_header(builder).send().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChatRequest;

    fn make_provider(name: &str, url: &str, key: Option<&str>) -> OpenAiCompatibleProvider {
        OpenAiCompatibleProvider::new(name, url, key)
    }

    #[tokio::test]
    async fn chat_without_key_attempts_request() {
        let p = make_provider("Local", "http://127.0.0.1:1", None);
        let result = p
            .chat(ChatRequest {
                messages: vec![ChatMessage::user("hello")],
                tools: None,
                model: "test".to_string(),
                allow_image_parts: false,
                temperature: 0.1,
                reasoning_effort: None,
                provider_order: None,
                provider_allow_fallbacks: None,
            })
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            !err_msg.contains("API key not set"),
            "should not get credential error, got: {err_msg}"
        );
    }

    #[test]
    fn tool_call_function_resolution() {
        // Name falls back to top-level name
        let call: ApiToolCall = serde_json::from_value(serde_json::json!({
            "name": "recall",
            "arguments": "{\"query\":\"latest roadmap\"}"
        }))
        .unwrap();
        assert_eq!(call.function_name().as_deref(), Some("recall"));

        // Arguments fall back to parameters object
        let call: ApiToolCall = serde_json::from_value(serde_json::json!({
            "name": "shell",
            "parameters": {"command": "pwd"}
        }))
        .unwrap();
        assert_eq!(
            call.function_arguments().as_deref(),
            Some("{\"command\":\"pwd\"}")
        );

        // Nested function field preferred over top-level
        let call: ApiToolCall = serde_json::from_value(serde_json::json!({
            "name": "ignored_name",
            "arguments": "{\"query\":\"ignored\"}",
            "function": {
                "name": "recall",
                "arguments": "{\"query\":\"preferred\"}"
            }
        }))
        .unwrap();
        assert_eq!(call.function_name().as_deref(), Some("recall"));
        assert_eq!(
            call.function_arguments().as_deref(),
            Some("{\"query\":\"preferred\"}")
        );
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

        let parsed = OpenAiCompatibleProvider::parse_native_response(message);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].id, "call_123");
        assert_eq!(parsed.tool_calls[0].name, "shell");
    }

    #[test]
    fn convert_messages_for_native_maps_tool_result_payload() {
        let input = vec![ChatMessage::tool(
            r#"{"tool_call_id":"call_abc","content":"done"}"#,
        )];

        let converted = OpenAiCompatibleProvider::convert_messages_for_native(&input, true);
        assert_eq!(converted[0].tool_call_id.as_deref(), Some("call_abc"));
        assert!(matches!(
            converted[0].content.as_ref(),
            Some(MessageContent::Text(value)) if value == "done"
        ));
    }

    #[test]
    fn convert_messages_for_native_keeps_user_image_markers_as_text_when_disabled() {
        let input = vec![ChatMessage::user(
            "System primer [IMAGE:data:image/png;base64,abcd] user turn",
        )];

        let converted = OpenAiCompatibleProvider::convert_messages_for_native(&input, false);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
        assert!(matches!(
            converted[0].content.as_ref(),
            Some(MessageContent::Text(value))
                if value == "System primer [IMAGE:data:image/png;base64,abcd] user turn"
        ));
    }

    #[test]
    fn strip_think_tags_tests() {
        assert_eq!(strip_think_tags("visible<think>hidden"), "visible");
        assert_eq!(
            strip_think_tags("Answer A <think>hidden 1</think> and B <think>hidden 2</think> done"),
            "Answer A  and B  done"
        );
        assert_eq!(strip_think_tags("Visible<think>hidden tail"), "Visible");
        // Multi-line think block (requires (?s) dotall flag)
        assert_eq!(
            strip_think_tags("Hello<think>\nmulti\nline\n</think>world"),
            "Helloworld"
        );
        // Multiple multi-line blocks
        assert_eq!(
            strip_think_tags("<think>\nblock 1\n</think>A<think>\nblock 2\n</think>B"),
            "AB"
        );
        // Empty think block
        assert_eq!(
            strip_think_tags("before<think></think>after"),
            "beforeafter"
        );
    }

    #[test]
    fn reasoning_content_fallback() {
        // Empty content, reasoning present → uses reasoning
        let json = r#"{"choices":[{"message":{"content":"","reasoning_content":"Thinking output here"}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0]
                .message
                .effective_content_optional()
                .unwrap_or_default(),
            "Thinking output here"
        );
        // Null content, reasoning present → uses reasoning
        let json =
            r#"{"choices":[{"message":{"content":null,"reasoning_content":"Fallback text"}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0]
                .message
                .effective_content_optional()
                .unwrap_or_default(),
            "Fallback text"
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
        // Content only think tags → uses reasoning
        let json = r#"{"choices":[{"message":{"content":"<think>secret</think>","reasoning_content":"Fallback text"}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0]
                .message
                .effective_content_optional()
                .unwrap_or_default(),
            "Fallback text"
        );
        assert_eq!(
            resp.choices[0]
                .message
                .effective_content_optional()
                .as_deref(),
            Some("Fallback text")
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
        let provider = make_provider("test", "http://127.0.0.1:1", None);
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
        let input = "Check this [IMAGE:/tmp/a.png] and this [IMAGE:https://example.com/b.jpg]";
        let (cleaned, refs) = parse_image_markers(input);

        assert_eq!(cleaned, "Check this  and this");
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0], "/tmp/a.png");
        assert_eq!(refs[1], "https://example.com/b.jpg");
    }

    #[test]
    fn parse_image_markers_keeps_invalid_empty_marker() {
        let input = "hello [IMAGE:] world";
        let (cleaned, refs) = parse_image_markers(input);

        assert_eq!(cleaned, "hello [IMAGE:] world");
        assert!(refs.is_empty());
    }

    /// Stripping `[IMAGE:]` markers from history messages leaves only the text
    /// portion, which is the behaviour needed for non-vision providers (#3674).
    #[test]
    fn parse_image_markers_strips_markers_leaving_caption() {
        let input = "[IMAGE:/tmp/photo.jpg]\n\nDescribe this screenshot";
        let (cleaned, refs) = parse_image_markers(input);
        assert_eq!(cleaned, "Describe this screenshot");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0], "/tmp/photo.jpg");
    }

    /// An image-only message (no caption) should produce an empty string after
    /// marker stripping, so callers can drop it from history.
    #[test]
    fn parse_image_markers_image_only_message_becomes_empty() {
        let input = "[IMAGE:/tmp/photo.jpg]";
        let (cleaned, refs) = parse_image_markers(input);
        assert!(
            cleaned.is_empty(),
            "expected empty string, got: {cleaned:?}"
        );
        assert_eq!(refs.len(), 1);
    }

    /// Non‑IMAGE markers (AUDIO, VIDEO) are preserved verbatim in the cleaned
    /// output while IMAGE markers are stripped. This test covers the mixed case
    /// to prevent regression of the preservation behaviour.
    #[test]
    fn parse_image_markers_preserves_audio_and_video_markers() {
        let input =
            "[AUDIO:/tmp/sound.mp3] Listen to this [VIDEO:/tmp/clip.mp4] and [IMAGE:/tmp/img.png]";
        let (cleaned, refs) = parse_image_markers(input);

        assert_eq!(
            cleaned,
            "[AUDIO:/tmp/sound.mp3] Listen to this [VIDEO:/tmp/clip.mp4] and"
        );
        assert_eq!(refs, vec!["/tmp/img.png"]);
    }

    #[test]
    fn to_message_content_converts_image_markers_to_openai_parts() {
        let content = "Describe this\n\n[IMAGE:data:image/png;base64,abcd]";
        let value =
            serde_json::to_value(to_message_content(MessageRole::User, content, true)).unwrap();
        let parts = value
            .as_array()
            .expect("multimodal content should be an array");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "Describe this");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,abcd");
    }

    #[test]
    fn to_message_content_keeps_markers_as_text_when_user_image_parts_disabled() {
        let content = "Policy [IMAGE:data:image/png;base64,abcd]";
        let value =
            serde_json::to_value(to_message_content(MessageRole::User, content, false)).unwrap();
        assert_eq!(value, serde_json::json!(content));
    }

    #[test]
    fn to_message_content_keeps_plain_text_for_non_user_roles() {
        let value = serde_json::to_value(to_message_content(
            MessageRole::System,
            "You are a helpful assistant.",
            true,
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
            temperature: 0.7,
            max_tokens: 32000,
            stream: Some(false),
            tool_stream: None,
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
    fn zai_tool_requests_enable_tool_stream() {
        let provider = make_provider("zai", "https://api.z.ai/api/paas/v4", None);
        let req = ChatCompletionRequest {
            model: "glm-5".to_string(),
            messages: vec![NativeMessage::user("List /tmp")],
            temperature: 0.7,
            max_tokens: 32000,
            stream: Some(false),
            tool_stream: provider.tool_stream_for_tools(true),
            tools: Some(vec![serde_json::json!({
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"}
                        }
                    }
                }
            })]),
            tool_choice: Some("auto".to_string()),
            extra: serde_json::Map::new(),
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"tool_stream\":true"));
    }

    #[test]
    fn non_zai_provider_omits_tool_stream_regardless_of_streaming() {
        let provider = make_provider("custom", "https://proxy.example.com/v1", None);
        // tool_stream_for_tools should return None for non-Z.AI providers
        assert_eq!(provider.tool_stream_for_tools(true), None);
        assert_eq!(provider.tool_stream_for_tools(false), None);
    }

    #[test]
    fn z_ai_host_enables_tool_stream_for_custom_profiles() {
        let provider = make_provider("custom", "https://api.z.ai/api/coding/paas/v4", None);
        assert_eq!(provider.tool_stream_for_tools(true), Some(true));
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
    fn api_response_parses_without_usage() {
        let json = r#"{"choices": [{"message": {"content": "Hello"}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.usage.is_none());
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

        let parsed = OpenAiCompatibleProvider::parse_native_response(message);
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

        let parsed = OpenAiCompatibleProvider::parse_native_response(message);
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
        let native = OpenAiCompatibleProvider::convert_messages_for_native(&messages, true);
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
        let native = OpenAiCompatibleProvider::convert_messages_for_native(&messages, true);
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
        let native = OpenAiCompatibleProvider::convert_messages_for_native(&messages, true);
        assert_eq!(native.len(), 1);
        assert_eq!(native[0].reasoning_content.as_deref(), Some("from details"));
        assert_eq!(native[0].reasoning_details.as_ref(), Some(&details));
    }
}
