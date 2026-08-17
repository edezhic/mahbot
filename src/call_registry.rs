//! Live registry of in-flight non-agent (single-shot utility) LLM calls.
//!
//! Standalone calls — consolidation, research-orchestration passes,
//! joint-verdict synthesis — register here on start and remove themselves on
//! completion via a RAII guard. Calls originating inside an agent run (the
//! agent loop, verdict extraction, summarization) never register; agents are
//! tracked separately in [`crate::registry::AGENT_REGISTRY`]. The tracking is
//! purely observational — it carries no cancellation semantics and never
//! affects call behavior, retries, or results.

use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use crate::registry::ParentKey;
use crate::util::UnwrapPoison;
use chrono::{DateTime, Utc};
use serde::Serialize;

/// Monotonically increasing entry id — lets each guard remove exactly its own
/// entry on drop, regardless of drop order.
static NEXT_ENTRY_ID: AtomicU64 = AtomicU64::new(1);

/// Public snapshot of one in-flight call — serializable, no internals exposed.
#[derive(Clone, Debug, Serialize)]
pub struct NonAgentCallHandle {
    /// Call kind — the same purpose string the call's `ChatRequestMeta` uses
    /// (e.g. `"consolidate"`, `"synthesis"`, `"gap_extract"`,
    /// `"research_wrap_up"`, `"media_transcription"`).
    pub kind: &'static str,
    pub workspace: String,
    pub started_at: DateTime<Utc>,
    /// Grouping key for the Running Agents view — the DIRECT PARENT
    /// INVOCATION this call belongs to (ticket / analyze round / research run).
    /// `None` means the call is genuinely unattributable (workspace-scoped
    /// section with a visually distinct marker).
    pub parent_key: Option<ParentKey>,
    /// Human-readable label for the DIRECT PARENT INVOCATION this call
    /// belongs to (ticket title / analyze question / research question) —
    /// shown on the group header of the Running Agents view. Purely
    /// presentational; never affects call behavior.
    pub parent_label: Option<String>,
    /// True when this entry is a whole-operation lifetime guard (the research
    /// orchestrator holds one guard for the entire run) — such entries render
    /// inside their research group as a run-lifetime indicator, not as a
    /// transient LLM-call card.
    pub run_lifetime: bool,
}

#[derive(Default)]
pub struct NonAgentCallRegistry {
    inner: Mutex<HashMap<u64, NonAgentCallHandle>>,
}

impl NonAgentCallRegistry {
    /// Register an in-flight non-agent LLM call and return the RAII guard
    /// that removes it on drop — cleanup is guaranteed on completion and on
    /// failure (including early returns and task cancellation).
    ///
    /// `parent` attaches the call to its DIRECT PARENT INVOCATION for the
    /// Running Agents grouping (ticket / analyze round / research run); `None`
    /// makes the call render in the workspace-scoped unattributable section.
    /// `parent_label` is the human-readable label of that parent invocation
    /// (ticket title / analyze question / research question) — purely
    /// presentational. `run_lifetime` marks whole-operation lifetime guards
    /// (research orchestrator) that render as a run-lifetime indicator, not a
    /// transient call card.
    pub fn register(
        &'static self,
        kind: &'static str,
        workspace: &str,
        parent: Option<ParentKey>,
        run_lifetime: bool,
        parent_label: Option<String>,
    ) -> NonAgentCallGuard {
        let id = NEXT_ENTRY_ID.fetch_add(1, Ordering::Relaxed);
        self.inner.lock().unwrap_poison().insert(
            id,
            NonAgentCallHandle {
                kind,
                workspace: workspace.to_string(),
                started_at: Utc::now(),
                parent_key: parent,
                parent_label,
                run_lifetime,
            },
        );
        NonAgentCallGuard { id, registry: self }
    }

    /// Snapshot of all in-flight non-agent LLM calls (serializable).
    #[must_use]
    pub fn list(&self) -> Vec<NonAgentCallHandle> {
        self.inner
            .lock()
            .unwrap_poison()
            .values()
            .cloned()
            .collect()
    }
}

/// RAII guard: removes its registry entry on drop.
pub struct NonAgentCallGuard {
    id: u64,
    registry: &'static NonAgentCallRegistry,
}

impl Drop for NonAgentCallGuard {
    fn drop(&mut self) {
        self.registry.inner.lock().unwrap_poison().remove(&self.id);
    }
}

/// Global static registry.
pub static NON_AGENT_CALLS: LazyLock<NonAgentCallRegistry> =
    LazyLock::new(NonAgentCallRegistry::default);

/// Static human-readable label for a non-agent call kind — the single source
/// of truth for every place a raw kind string would otherwise surface
/// (Running Agents call rows, the footer zap tooltip).
///
/// The mapping is deliberately EXHAUSTIVE over every kind the codebase
/// registers today; unknown/future kinds fall back to a generic label so a
/// raw snake_case name can never leak onto the page. The fallback is
/// `"Other LLM work"` — the same wording used for the unattributed
/// workspace-scoped section, so a future kind reads as generic pipeline work.
#[must_use]
pub fn call_kind_label(kind: &str) -> &'static str {
    match kind {
        "consolidate" => "Analyze consolidation",
        "synthesis" => "Ticket synthesis",
        "synthesize" => "Research synthesis",
        "decompose_merge" => "Research plan merge",
        "gap_extract" => "Research gap extraction",
        "abstain_check" => "Research answerability check",
        "claim_annotate" => "Research claim annotation",
        "confirm_links" => "Research link confirmation",
        "research_wrap_up" => "Research wrap-up",
        "research_orchestrator" => "Research orchestrator",
        "media_transcription" => "Media transcription",
        _ => "Other LLM work",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_carries_parent_key_and_run_lifetime() {
        let kind: &'static str = "consolidate";
        let guard = NON_AGENT_CALLS.register(
            kind,
            "ws1",
            Some(ParentKey::AnalyzeRound("round_1".to_string())),
            false,
            Some("Why is CI flaky?".to_string()),
        );
        // The run-lifetime flag is a distinct registration mode (whole-run
        // orchestrator guards); both must round-trip through the same path.
        let orchestrator: &'static str = "research_orchestrator";
        let orchestrator_guard = NON_AGENT_CALLS.register(
            orchestrator,
            "ws1",
            Some(ParentKey::Research("job_9".to_string())),
            true,
            None,
        );
        let handles = NON_AGENT_CALLS.list();
        let h = handles
            .iter()
            .find(|h| h.kind == kind)
            .expect("call registered");
        assert_eq!(h.workspace, "ws1");
        assert_eq!(
            h.parent_key,
            Some(ParentKey::AnalyzeRound("round_1".to_string()))
        );
        assert_eq!(
            h.parent_label.as_deref(),
            Some("Why is CI flaky?"),
            "parent label round-trips"
        );
        assert!(!h.run_lifetime);
        let o = handles
            .iter()
            .find(|h| h.kind == orchestrator)
            .expect("orchestrator registered");
        assert!(o.run_lifetime, "run-lifetime flag must round-trip");
        assert_eq!(o.parent_key, Some(ParentKey::Research("job_9".to_string())));
        drop(guard);
        drop(orchestrator_guard);
        assert!(
            !NON_AGENT_CALLS
                .list()
                .iter()
                .any(|h| h.kind == kind || h.kind == orchestrator),
            "guard drops remove their own entries"
        );
    }

    #[test]
    fn call_kind_labels_cover_every_known_kind_without_raw_names() {
        // Every kind the codebase registers must map to a human-readable
        // label, and unknown/future kinds must fall back to a generic label —
        // a raw snake_case name can never leak onto the page.
        for kind in [
            "consolidate",
            "synthesis",
            "synthesize",
            "decompose_merge",
            "gap_extract",
            "abstain_check",
            "claim_annotate",
            "confirm_links",
            "research_wrap_up",
            "research_orchestrator",
            "media_transcription",
        ] {
            let label = call_kind_label(kind);
            assert_ne!(
                label, kind,
                "raw kind '{kind}' must never be used as its own label"
            );
            assert!(
                !label.contains('_'),
                "label must be human-readable: {label}"
            );
        }
        assert_eq!(
            call_kind_label("future_unknown_kind"),
            "Other LLM work",
            "unknown kinds fall back to a generic label"
        );
    }
}
