//! Structured extraction from conversation history.
//!
//! Provides LLM-powered extraction of structured data (JSON) from agent
//! conversation history, with retry logic and callback-button parsing.
//! The provider dependency is natural here — extraction *is* an LLM call.

use serde::de::DeserializeOwned;

use crate::config::CONFIG;
use crate::providers::chat;
use crate::util::parse_fenced_json;
use crate::{ChatMessage, ChatRequest, ToolSpec};

// ── Extraction config ─────────────────────────────────────────────────

/// LLM parameters for structured extraction.
///
/// Groups the parameters that should be byte-identical to the original agent
/// call so the provider can reuse the cached KV-cache prefix.  Provider
/// routing is derived from [`model`](Self::model) by
/// [`CONFIG.model_routing`] internally.
///
/// KV-cache preservation: callers must pass the same `temperature`,
/// `reasoning_effort`, `model`, and `tool_specs` that the agent's work loop
/// uses so the provider can reuse the cached prefix.
pub(crate) struct ExtractionConfig<'a> {
    /// The model identifier (used for both the LLM call and provider routing).
    pub model: &'a str,
    /// Tool specifications for function calling.  Pass `None` to omit tools.
    pub tool_specs: Option<&'a [ToolSpec]>,
    /// Temperature for the LLM call.
    pub temperature: f32,
    /// Reasoning effort (e.g. `"low"`, `"high"`).  `None` disables reasoning.
    pub reasoning_effort: Option<String>,
    /// Maximum retry attempts before bailing.
    pub max_attempts: usize,
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
/// `temperature`, `reasoning_effort`, `tool_specs`) must be byte-identical to
/// the original agent call so the provider can reuse the cached prefix.
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

    let routing = CONFIG.model_routing(config.model);
    for _attempt in 1..=config.max_attempts {
        let response = chat(ChatRequest {
            messages: extraction_history.clone(),
            tools: config.tool_specs.map(<[ToolSpec]>::to_vec),
            model: config.model.to_string(),
            allow_image_parts: false, // extractions never need image parts
            temperature: config.temperature,
            reasoning_effort: config.reasoning_effort.clone(),
            provider_order: routing.provider_order.clone(),
            provider_allow_fallbacks: routing.allow_fallbacks,
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

// ── Option extraction types ───────────────────────────────────────────

/// Callback data prefix for dynamic option buttons.
pub(crate) const CALLBACK_PREFIX: &str = "__opt__";

/// Decode callback data from inline keyboard interactions.
///
/// Returns `(ticket_id, label)` on success (`ticket_id` is `None` when the
/// callback data was generated without one).  Returns `None` when `content`
/// does not carry the `CALLBACK_PREFIX`.
///
/// # Format contract
///
/// The callback data uses `|` as a delimiter between the optional ticket-id
/// and the label.  The join and split therefore assume that `ticket_id` must
/// not contain `|`; the label may contain `|`.
///
/// **Examples:**
/// - `__opt__ticket-id|Label` → `(Some("ticket-id"), "Label")`
/// - `__opt__|Label` → `(None, "Label")`
/// - `__opt__BareLabel` → `(None, "BareLabel")`
#[must_use]
pub fn decode_callback(content: &str) -> Option<(Option<String>, String)> {
    let rest = content.strip_prefix(CALLBACK_PREFIX)?;
    Some(match rest.split_once('|') {
        Some((tid, lbl)) if !tid.is_empty() => (Some(tid.to_string()), lbl.to_string()),
        Some((_, lbl)) => (None, lbl.to_string()),
        None => (None, rest.to_string()),
    })
}

// ── Action prefixes (__act__) ───────────────────────────────────────

/// Callback data prefix for action callbacks (e.g., model selection, clear session).
pub(crate) const ACTION_PREFIX: &str = "__act__";

/// Decode action callback data.
///
/// Returns `(action, payload)` on success, `None` when `content` does not
/// carry the `ACTION_PREFIX`.
///
/// # Format
///
/// `__act__<action>|<payload>` where `<action>` is the action name and
/// `<payload>` is the action-specific data (may be empty).
///
/// **Examples:**
/// - `__act__set_image_model|google/gemini-3.1-flash-image-preview`
///   → `("set_image_model", "google/gemini-3.1-flash-image-preview")`
/// - `__act__clear_session|` → `("clear_session", "")`
/// - `__act__clear_session` → `("clear_session", "")`
#[must_use]
pub fn decode_action(content: &str) -> Option<(String, String)> {
    let rest = content.strip_prefix(ACTION_PREFIX)?;
    match rest.split_once('|') {
        Some((action, payload)) => Some((action.to_string(), payload.to_string())),
        None => Some((rest.to_string(), String::new())),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod callback_tests {
    use super::{decode_action, decode_callback};

    // ── decode_callback ───────────────────────────────────────────────────

    #[test]
    fn test_decode_callback() {
        struct Case {
            name: &'static str,
            input: &'static str,
            expected: Option<(Option<&'static str>, &'static str)>,
        }

        let cases = [
            Case {
                name: "with ticket id",
                input: "__opt__mahbot-123|Option A",
                expected: Some((Some("mahbot-123"), "Option A")),
            },
            Case {
                name: "empty ticket id",
                input: "__opt__|Label",
                expected: Some((None, "Label")),
            },
            Case {
                name: "no delimiter",
                input: "__opt__BareLabel",
                expected: Some((None, "BareLabel")),
            },
            Case {
                name: "rejects non prefix",
                input: "random_text",
                expected: None,
            },
            Case {
                name: "rejects empty",
                input: "",
                expected: None,
            },
            Case {
                name: "label with extra pipes",
                input: "__opt__ticket|A|B|C",
                expected: Some((Some("ticket"), "A|B|C")),
            },
            // Labels containing '|' test a deliberate `split_once` behavior:
            // `split_once('|')` splits on the *first* pipe only, so the label
            // captures everything after it.  Neither ticket_id nor label should
            // contain `|` in practice (per the format contract in the doc comment).
            Case {
                name: "only prefix and pipe",
                input: "__opt__|",
                expected: Some((None, "")),
            },
        ];

        for case in &cases {
            let result = decode_callback(case.input);
            let expected = case
                .expected
                .map(|(tid, lbl)| (tid.map(String::from), lbl.to_string()));
            assert_eq!(result, expected, "case: {}", case.name);
        }
    }

    // ── decode_action ─────────────────────────────────────────────────────

    #[test]
    fn test_decode_action() {
        struct Case {
            name: &'static str,
            input: &'static str,
            expected: Option<(&'static str, &'static str)>,
        }

        let cases = [
            Case {
                name: "with payload",
                input: "__act__set_image_model|google/gemini-3.1-flash-image-preview",
                expected: Some(("set_image_model", "google/gemini-3.1-flash-image-preview")),
            },
            Case {
                name: "empty payload pipe",
                input: "__act__clear_session|",
                expected: Some(("clear_session", "")),
            },
            Case {
                name: "no pipe",
                input: "__act__clear_session",
                expected: Some(("clear_session", "")),
            },
            Case {
                name: "rejects non prefix",
                input: "random_text",
                expected: None,
            },
            Case {
                name: "rejects empty",
                input: "",
                expected: None,
            },
        ];

        for case in &cases {
            let result = decode_action(case.input);
            let expected = case
                .expected
                .map(|(action, payload)| (action.to_string(), payload.to_string()));
            assert_eq!(result, expected, "case: {}", case.name);
        }
    }
}
