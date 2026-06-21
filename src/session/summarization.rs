//! Conversation summarization — constants and helpers only.
//!
//! The summarization LLM call itself has moved to `crate::Agent::summarize`
//! so that all parameters (model, temperature, reasoning_effort, tools,
//! provider routing) are byte-identical to the agent's work loop.
//! This module retains the constants and helpers used by `Session::apply_summary`.

use crate::ChatMessage;

pub const SUMMARIZATION_THRESHOLD: usize = 400_000;

/// Stored session rows and second `history` entry after compaction use this prefix so channel
/// orchestration can re-inject the summary on later turns (baseline `system` rows stay excluded).
pub const PREVIOUS_CONVERSATION_SUMMARY_PREFIX: &str = "Previous conversation summary:\n\n";

/// Rough token count for history (~4 chars/token + 4 tokens per-message overhead)
#[must_use]
pub fn estimate_tokens(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|m| m.content.len().div_ceil(4) + 4)
        .sum()
}
