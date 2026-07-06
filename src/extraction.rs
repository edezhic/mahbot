//! Structured extraction from conversation history.
//!
//! Provides LLM-powered extraction of structured data (JSON) from agent
//! conversation history, with retry logic.

use serde::de::DeserializeOwned;

use crate::providers::chat;
use crate::util::parse_fenced_json;
use crate::{ChatMessage, ChatRequest, ToolSpec};

// ── Extraction config ─────────────────────────────────────────────────

/// LLM parameters for structured extraction.
///
/// Groups the parameters that should be byte-identical to the original agent
/// call so the provider can reuse the cached KV-cache prefix.
///
/// KV-cache preservation: callers must pass the same `model`, `temperature`,
/// `reasoning_effort`, `tool_specs`, `max_tokens`, `provider_order`, and
/// `provider_allow_fallbacks` that the agent's work loop uses so the provider
/// can reuse the cached prefix.
pub(crate) struct ExtractionConfig<'a> {
    /// The model identifier (used for both the LLM call and provider routing).
    pub model: &'a str,
    /// Tool specifications for function calling.
    pub tool_specs: &'a [ToolSpec],
    /// Temperature for the LLM call.
    pub temperature: f32,
    /// Reasoning effort (e.g. `"low"`, `"high"`).  `None` disables reasoning.
    pub reasoning_effort: Option<String>,
    /// Maximum retry attempts before bailing.
    pub max_attempts: usize,
    /// Maximum tokens for the LLM response (mirrors [`crate::ChatRequest::max_tokens`]).
    pub max_tokens: Option<u32>,
    /// Provider routing order (mirrors [`crate::ChatRequest::provider_order`]).
    pub provider_order: Option<String>,
    /// Allow provider fallbacks (mirrors [`crate::ChatRequest::provider_allow_fallbacks`]).
    pub provider_allow_fallbacks: Option<bool>,
}

// ── Retry extraction ──────────────────────────────────────────────────

/// Retry a structured JSON extraction from conversation history.
///
/// Pushes `extraction_prompt` into the history, then loops up to
/// [`config.max_attempts`](ExtractionConfig::max_attempts) calling the LLM.
/// On each iteration:
/// - Tool calls → treat as failure, push `retry_prompt`, retry
/// - Non-parseable text → push raw assistant text + `retry_prompt`, retry
/// - Valid JSON matching `T` → return immediately
///
/// Pass `extraction_prompt = ""` if the prompt is already embedded in `history`.
///
/// KV-cache preservation: the [`config`](ExtractionConfig) fields (`model`,
/// `temperature`, `reasoning_effort`, `tool_specs`, `max_tokens`, `provider_order`,
/// `provider_allow_fallbacks`) must be byte-identical to the original agent call
/// so the provider can reuse the cached prefix.
pub(crate) async fn retry_extract_structured<T: DeserializeOwned>(
    history: &[ChatMessage],
    extraction_prompt: &str,
    retry_prompt: &str,
    config: ExtractionConfig<'_>,
) -> anyhow::Result<T> {
    let mut extraction_history = history.to_vec();

    // Only push the extraction prompt if non-empty — caller may have embedded it
    if !extraction_prompt.is_empty() {
        extraction_history.push(ChatMessage::user(extraction_prompt));
    }

    let mut last_raw = String::new();

    for _attempt in 1..=config.max_attempts {
        let response = chat(ChatRequest {
            messages: extraction_history.clone(),
            tools: Some(config.tool_specs.to_vec()),
            model: config.model.to_string(),
            allow_image_parts: false, // extractions never need image parts
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            reasoning_effort: config.reasoning_effort.clone(),
            provider_order: config.provider_order.clone(),
            provider_allow_fallbacks: config.provider_allow_fallbacks,
        })
        .await?;

        last_raw = response.text_or_empty().to_string();

        // Try to parse as T (handles markdown fencing internally) — only if no tool calls
        if response.tool_calls.is_empty()
            && let Ok(result) = parse_fenced_json::<T>(&last_raw)
        {
            return Ok(result);
        }

        // Tool calls or parse failure — push raw assistant text + retry prompt, continue
        extraction_history.push(ChatMessage::assistant(last_raw.clone()));
        extraction_history.push(ChatMessage::user(retry_prompt));
    }

    let snippet: String = last_raw.chars().take(300).collect();
    anyhow::bail!(
        "Failed to extract structured response after {max_attempts} attempts. Last raw: {snippet}",
        max_attempts = config.max_attempts,
    )
}
