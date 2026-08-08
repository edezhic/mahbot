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
}

#[derive(Default)]
pub struct NonAgentCallRegistry {
    inner: Mutex<HashMap<u64, NonAgentCallHandle>>,
}

impl NonAgentCallRegistry {
    /// Register an in-flight non-agent LLM call and return the RAII guard
    /// that removes it on drop — cleanup is guaranteed on completion and on
    /// failure (including early returns and task cancellation).
    pub fn register(&'static self, kind: &'static str, workspace: &str) -> NonAgentCallGuard {
        let id = NEXT_ENTRY_ID.fetch_add(1, Ordering::Relaxed);
        self.inner.lock().unwrap_poison().insert(
            id,
            NonAgentCallHandle {
                kind,
                workspace: workspace.to_string(),
                started_at: Utc::now(),
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
