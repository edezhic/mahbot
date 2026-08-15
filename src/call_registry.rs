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
    /// (e.g. `"consolidate"`, `"synthesis"`, `"gap_extract"`).
    pub kind: &'static str,
    pub workspace: String,
    pub started_at: DateTime<Utc>,
    /// Grouping key for the Running Agents view — the DIRECT PARENT
    /// INVOCATION this call belongs to (ticket / ask round / research run).
    /// `None` means the call is genuinely unattributable (workspace-scoped
    /// section with a visually distinct marker).
    pub parent_key: Option<ParentKey>,
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
    /// Running Agents grouping (ticket / ask round / research run); `None`
    /// makes the call render in the workspace-scoped unattributable section.
    /// `run_lifetime` marks whole-operation lifetime guards (research
    /// orchestrator) that render as a run-lifetime indicator, not a transient
    /// call card.
    pub fn register(
        &'static self,
        kind: &'static str,
        workspace: &str,
        parent: Option<ParentKey>,
        run_lifetime: bool,
    ) -> NonAgentCallGuard {
        let id = NEXT_ENTRY_ID.fetch_add(1, Ordering::Relaxed);
        self.inner.lock().unwrap_poison().insert(
            id,
            NonAgentCallHandle {
                kind,
                workspace: workspace.to_string(),
                started_at: Utc::now(),
                parent_key: parent,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_carries_parent_key_and_run_lifetime() {
        let kind: &'static str = "consolidate";
        let guard = NON_AGENT_CALLS.register(
            kind,
            "ws1",
            Some(ParentKey::AskRound("round_1".to_string())),
            false,
        );
        // The run-lifetime flag is a distinct registration mode (whole-run
        // orchestrator guards); both must round-trip through the same path.
        let orchestrator: &'static str = "research_orchestrator";
        let orchestrator_guard = NON_AGENT_CALLS.register(
            orchestrator,
            "ws1",
            Some(ParentKey::Research("job_9".to_string())),
            true,
        );
        let handles = NON_AGENT_CALLS.list();
        let h = handles
            .iter()
            .find(|h| h.kind == kind)
            .expect("call registered");
        assert_eq!(h.workspace, "ws1");
        assert_eq!(
            h.parent_key,
            Some(ParentKey::AskRound("round_1".to_string()))
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
}
