//! Structured extraction from conversation history.
//!
//! Provides LLM-powered extraction of structured data (JSON) from agent
//! conversation history, with retry logic.

use serde::de::DeserializeOwned;

use crate::prompt::load_prompt;
use crate::providers::chat;
use crate::util::json::parse_fenced_json;
use crate::{ChatMessage, ChatRequest};

// ── Retry extraction ──────────────────────────────────────────────────

/// Retry a structured JSON extraction from conversation history.
///
/// Pushes `extraction_prompt` into the history, then loops up to
/// `max_attempts` calling the LLM.
/// On each iteration:
/// - Tool calls → treat as failure, push `retry_prompt`, retry
/// - Non-parseable text → push raw assistant text + `retry_prompt`, retry
/// - Valid JSON matching `T` → return immediately
///
/// Pass `extraction_prompt = ""` if the prompt is already embedded in `history`.
///
/// KV-cache preservation: the `params` fields (`model`, `temperature`,
/// `reasoning_effort`, `tools`, `max_tokens`, `provider_order`,
/// `provider_allow_fallbacks`) must be byte-identical to the original agent call
/// so the provider can reuse the cached prefix.
pub(crate) async fn retry_extract_structured<T: DeserializeOwned>(
    history: &[ChatMessage],
    extraction_prompt: &str,
    params: &ChatRequest,
    max_attempts: usize,
) -> anyhow::Result<T> {
    let mut extraction_history = history.to_vec();

    // Only push the extraction prompt if non-empty — caller may have embedded it
    if !extraction_prompt.is_empty() {
        extraction_history.push(ChatMessage::user(extraction_prompt));
    }

    let retry_prompt = load_prompt("extraction/retry.md");
    let mut last_raw = String::new();

    for _attempt in 1..=max_attempts {
        let response = chat(ChatRequest {
            messages: extraction_history.clone(),
            allow_image_parts: false, // extractions never need image parts
            ..params.clone()
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
        extraction_history.push(ChatMessage::user(retry_prompt.as_str()));
    }

    let snippet: String = last_raw.chars().take(300).collect();
    anyhow::bail!(
        "Failed to extract structured response after {max_attempts} attempts. Last raw: {snippet}",
    )
}
