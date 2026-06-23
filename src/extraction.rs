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

    for _attempt in 1..=config.max_attempts {
        let routing = CONFIG.model_routing(config.model);
        let response = chat(ChatRequest {
            messages: extraction_history.clone(),
            tools: config.tool_specs.map(<[ToolSpec]>::to_vec),
            model: config.model.to_string(),
            allow_image_parts: false, // extractions never need image parts
            temperature: config.temperature,
            reasoning_effort: config.reasoning_effort.clone(),
            provider_order: routing.provider_order,
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

/// Check whether `content` begins with `CALLBACK_PREFIX`.
///
/// Fast prefix-only check — useful as an early filter before calling
/// [`decode_callback`].
#[must_use]
pub fn is_callback(content: &str) -> bool {
    content.starts_with(CALLBACK_PREFIX)
}

/// Decode callback data from inline keyboard interactions.
///
/// Returns `(ticket_id, label)` on success (`ticket_id` is `None` when the
/// callback data was generated without one).  Returns `None` when `content`
/// does not carry the `CALLBACK_PREFIX`.
///
/// # Format contract
///
/// The callback data uses `|` as a delimiter between the optional ticket-id
/// and the label.  Both join and split therefore assume that neither
/// `ticket_id` nor `label` may contain a `|` character.
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

/// Check whether `content` begins with `ACTION_PREFIX`.
///
/// Fast prefix-only check — useful as an early filter before calling
/// [`decode_action`].
#[must_use]
pub fn is_action(content: &str) -> bool {
    content.starts_with(ACTION_PREFIX)
}

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
    use super::{decode_action, decode_callback, is_action, is_callback};

    // ── is_callback ───────────────────────────────────────────────────────

    #[test]
    fn is_callback_matches_prefix() {
        assert!(is_callback("__opt__ticket123|Label"));
        assert!(is_callback("__opt__|Label"));
        assert!(is_callback("__opt__bare"));
    }

    #[test]
    fn is_callback_rejects_non_prefix() {
        assert!(!is_callback("not_opt_something"));
        assert!(!is_callback(""));
        assert!(!is_callback("__op__ticket|Label"));
    }

    // ── decode_callback ───────────────────────────────────────────────────

    #[test]
    fn decode_callback_with_ticket_id() {
        let (ticket_id, label) = decode_callback("__opt__mahbot-123|Option A").unwrap();
        assert_eq!(ticket_id.as_deref(), Some("mahbot-123"));
        assert_eq!(label, "Option A");
    }

    #[test]
    fn decode_callback_empty_ticket_id() {
        // "__opt__|Label" — empty ticket_id before the delimiter
        let (ticket_id, label) = decode_callback("__opt__|Label").unwrap();
        assert_eq!(ticket_id.as_deref(), None);
        assert_eq!(label, "Label");
    }

    #[test]
    fn decode_callback_no_delimiter() {
        // No '|' at all — everything after prefix is the label
        let (ticket_id, label) = decode_callback("__opt__BareLabel").unwrap();
        assert_eq!(ticket_id.as_deref(), None);
        assert_eq!(label, "BareLabel");
    }

    #[test]
    fn decode_callback_rejects_non_prefix() {
        assert!(decode_callback("random_text").is_none());
        assert!(decode_callback("").is_none());
    }

    #[test]
    fn decode_callback_label_with_extra_pipes() {
        // Labels containing '|' are a known fragility — the format contract
        // assumes neither ticket_id nor label may contain '|'.
        // split_once('|') splits on the *first* pipe, so the label captures
        // everything after it.
        let (ticket_id, label) = decode_callback("__opt__ticket|A|B|C").unwrap();
        assert_eq!(ticket_id.as_deref(), Some("ticket"));
        assert_eq!(label, "A|B|C");
    }

    #[test]
    fn decode_callback_only_prefix_and_pipe() {
        // "__opt__|" — empty ticket_id, empty label
        let (ticket_id, label) = decode_callback("__opt__|").unwrap();
        assert_eq!(ticket_id.as_deref(), None);
        assert_eq!(label, "");
    }

    // ── is_action ───────────────────────────────────────────────────────

    #[test]
    fn is_action_matches_prefix() {
        assert!(is_action("__act__set_image_model|model-name"));
        assert!(is_action("__act__clear_session|"));
        assert!(is_action("__act__clear_session"));
    }

    #[test]
    fn is_action_rejects_non_prefix() {
        assert!(!is_action("not_act_something"));
        assert!(!is_action(""));
        assert!(!is_action("__ac__something"));
    }

    // ── decode_action ───────────────────────────────────────────────────

    #[test]
    fn decode_action_with_payload() {
        let (action, payload) =
            decode_action("__act__set_image_model|google/gemini-3.1-flash-image-preview").unwrap();
        assert_eq!(action, "set_image_model");
        assert_eq!(payload, "google/gemini-3.1-flash-image-preview");
    }

    #[test]
    fn decode_action_empty_payload_with_pipe() {
        let (action, payload) = decode_action("__act__clear_session|").unwrap();
        assert_eq!(action, "clear_session");
        assert_eq!(payload, "");
    }

    #[test]
    fn decode_action_no_pipe() {
        let (action, payload) = decode_action("__act__clear_session").unwrap();
        assert_eq!(action, "clear_session");
        assert_eq!(payload, "");
    }

    #[test]
    fn decode_action_rejects_non_prefix() {
        assert!(decode_action("random_text").is_none());
        assert!(decode_action("").is_none());
    }
}
