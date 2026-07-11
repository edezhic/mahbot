//! Global registry of running agents with cancellation support.

use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;

use crate::util::UnwrapPoison;
use chrono::{DateTime, Utc};
use serde::Serialize;
use tokio_util::sync::CancellationToken;

/// Monotonically increasing generation counter for registry entries.
/// Used by [`deregister`](AgentRegistry::deregister) to detect stale entries
/// — when a new agent is registered with the same `run_id` (e.g. the Manager
/// interrupt-and-resume pattern), the old entry's generation will not match
/// the new entry, so `deregister` will not incorrectly remove the replacement.
static NEXT_ENTRY_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Public handle returned by `list()` — serializable, no cancel_token exposed.
#[derive(Clone, Debug, Serialize)]
pub struct AgentHandle {
    pub run_id: String,
    pub role: String,
    pub ticket_id: Option<String>,
    /// Filesystem path of the workspace (not the name) — this is used for
    /// agent display/location and is intentionally distinct from the
    /// workspace_name identifier used in the board database.
    pub workspace_path: String,
    pub started_at: DateTime<Utc>,
    pub label: String,
}

struct AgentEntry {
    generation: u64,
    handle: AgentHandle,
    cancel_token: CancellationToken,
}

#[derive(Default)]
pub struct AgentRegistry {
    inner: Mutex<HashMap<String, AgentEntry>>,
}

impl AgentRegistry {
    /// Register an agent entry and return the generation counter.
    ///
    /// Used by [`crate::Agent::new`] where deregistration is handled by [`crate::Agent::drop`]
    /// instead of a guard.
    pub fn register(
        &self,
        run_id: String,
        role: String,
        ticket_id: Option<String>,
        ws: &crate::Workspace,
        label: String,
        cancel_token: CancellationToken,
    ) -> u64 {
        let generation = NEXT_ENTRY_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let handle = AgentHandle {
            run_id: run_id.clone(),
            role,
            ticket_id,
            workspace_path: ws.path.clone(),
            started_at: Utc::now(),
            label,
        };
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(old) = map.remove(&run_id) {
            old.cancel_token.cancel();
        }
        map.insert(
            run_id,
            AgentEntry {
                generation,
                handle,
                cancel_token,
            },
        );
        generation
    }

    /// Cancel a specific agent by run_id. Removes it from the registry.
    ///
    /// Prefer [`cancel_by_ticket_id`](AgentRegistry::cancel_by_ticket_id) or
    /// [`cancel_by_role_and_workspace_path`](AgentRegistry::cancel_by_role_and_workspace_path)
    /// for external callers — this method bypasses the generation-based safety check
    /// that guards against stale `run_id` references.
    fn cancel(&self, run_id: &str) {
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(entry) = map.remove(run_id) {
            entry.cancel_token.cancel();
        }
    }

    /// Cancel all agents matching a predicate.
    ///
    /// The lock is dropped **before** calling [`cancel`](AgentRegistry::cancel) on each matched
    /// run ID to avoid deadlock — `cancel` acquires the same lock internally.
    ///
    /// # Lock-ordering invariant
    ///
    /// The predicate is evaluated while the lock is held, then the lock is released
    /// and cancellation proceeds without it. This is the only safe ordering — any
    /// future `cancel_by_*` method MUST follow this pattern or risk deadlock.
    fn cancel_matching<F>(&self, predicate: F)
    where
        F: Fn(&AgentEntry) -> bool,
    {
        let to_cancel: Vec<String> = {
            let map = self.inner.lock().unwrap_poison();
            map.iter()
                .filter(|(_, entry)| predicate(entry))
                .map(|(id, _)| id.clone())
                .collect()
        };
        for run_id in to_cancel {
            self.cancel(&run_id);
        }
    }

    /// Cancel all agents running for a specific `ticket_id`.
    /// Used on ticket phase transitions — stops any agent currently working on it.
    pub fn cancel_by_ticket_id(&self, ticket_id: &str) {
        self.cancel_matching(|entry| entry.handle.ticket_id.as_deref() == Some(ticket_id));
    }

    /// Cancel all agents running for a specific role within a specific workspace path.
    /// Used when maintenance is disabled for a workspace — stops the in-flight maintainer agent.
    pub fn cancel_by_role_and_workspace_path(&self, role: &str, ws_path: &str) {
        self.cancel_matching(|entry| {
            entry.handle.role == role && entry.handle.workspace_path == ws_path
        });
    }

    /// Snapshot of all currently running agents (serializable).
    #[must_use]
    pub fn list(&self) -> Vec<AgentHandle> {
        self.inner
            .lock()
            .unwrap_poison()
            .values()
            .map(|e| e.handle.clone())
            .collect()
    }

    /// Cancel all running agents. Used during daemon shutdown.
    pub fn shutdown_all(&self) {
        let entries: Vec<(String, CancellationToken)> = self
            .inner
            .lock()
            .unwrap_poison()
            .drain()
            .map(|(id, entry)| (id, entry.cancel_token))
            .collect();
        for (_id, token) in entries {
            token.cancel();
        }
    }

    /// Remove a registry entry only if its generation still matches.
    /// Used by [`crate::Agent::drop`] to safely deregister without stale-removal risk.
    pub fn deregister(&self, run_id: &str, generation: u64) {
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(entry) = map.get(run_id)
            && entry.generation == generation
        {
            map.remove(run_id);
        }
    }
}

/// Global static registry.
pub static AGENT_REGISTRY: LazyLock<AgentRegistry> = LazyLock::new(AgentRegistry::default);
