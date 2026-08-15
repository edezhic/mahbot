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
/// — when a new agent is registered with the same `agent_id` (e.g. the Manager
/// interrupt-and-resume pattern), the old entry's generation will not match
/// the new entry, so `deregister` will not incorrectly remove the replacement.
static NEXT_ENTRY_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Monotonically increasing tool-instance id for live tool instrumentation.
/// Each tool execution inside an agent gets a unique id so parallel tools can
/// coexist in [`AgentEntry::current_tools`] and each guard removes exactly its
/// own entry.
static NEXT_TOOL_ID: AtomicU64 = AtomicU64::new(1);

/// Grouping key for the Running Agents live view: the DIRECT PARENT
/// INVOCATION a running work item belongs to.
///
/// - [`Ticket`](ParentKey::Ticket) — all agents working on the same ticket
///   plus LLM calls belonging to the ticket's own work (e.g. joint-verdict
///   synthesis of a review round).
/// - [`AskRound`](ParentKey::AskRound) — the parallel analysts of one ask
///   round plus its consolidation LLM call (members share the round key).
/// - [`Research`](ParentKey::Research) — all sub-agents and orchestrator LLM
///   calls of one research run (members share the durable research job id).
///
/// `None` means the work item is a workspace singleton (manager / maintainer /
/// discovery / direct chat) or — for non-agent calls — genuinely
/// unattributable orchestrator work (workspace-scoped section).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum ParentKey {
    Ticket(String),
    AskRound(String),
    Research(String),
}

/// One tool currently executing inside a running agent — the live
/// instrumentation shown on the Running Agents page. `args` is
/// credential-scrubbed and truncated to a bounded length at registration.
#[derive(Clone, Debug, Serialize)]
pub struct RunningTool {
    pub name: String,
    pub args: String,
}

/// Public handle returned by `list()` — serializable, no cancel_token exposed.
#[derive(Clone, Debug, Serialize)]
pub struct AgentHandle {
    pub agent_id: String,
    pub role: String,
    pub ticket_id: Option<String>,
    /// Filesystem path of the workspace (not the name) — this is used for
    /// agent display/location and is intentionally distinct from the
    /// workspace_name identifier used in the board database.
    pub workspace_path: String,
    /// Workspace NAME — the identifier used by the board database and the
    /// dashboard's workspace map. Displayed on Running Agents cards/group
    /// headers and used for paused-state lookups.
    pub workspace_name: String,
    /// Grouping key for the Running Agents view — see [`ParentKey`].
    pub parent_key: Option<ParentKey>,
    pub started_at: DateTime<Utc>,
    pub label: String,
    /// Snapshot of the tools currently executing inside this agent, taken at
    /// `list()` time. Empty when the agent is between tool executions (in an
    /// LLM call, between rounds) — absence is honest.
    pub current_tools: Vec<RunningTool>,
}

struct AgentEntry {
    generation: u64,
    handle: AgentHandle,
    cancel_token: CancellationToken,
    /// Live tool instrumentation for this agent (mutable across the agent's
    /// lifetime; snapshot into [`AgentHandle::current_tools`] by `list()`).
    /// Every access is already serialized by the outer
    /// [`AgentRegistry::inner`] lock (tool_started/tool_finished/list all
    /// hold it), so the Vec needs no further synchronization. Each entry
    /// carries the unique tool instance id for exact removal.
    current_tools: Vec<RunningToolEntry>,
}

/// Internal live-tool entry: the public tool fields plus the unique instance
/// id used for exact removal by [`RunningToolGuard`].
#[derive(Clone, Debug)]
struct RunningToolEntry {
    id: u64,
    tool: RunningTool,
}

impl RunningToolEntry {
    fn to_handle(&self) -> RunningTool {
        self.tool.clone()
    }
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
    #[allow(clippy::too_many_arguments)] // one positional arg per handle field; callers use literals
    pub fn register(
        &self,
        agent_id: String,
        role: String,
        ticket_id: Option<String>,
        ws: &crate::Workspace,
        label: String,
        cancel_token: CancellationToken,
        parent_key: Option<ParentKey>,
    ) -> u64 {
        let generation = NEXT_ENTRY_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let handle = AgentHandle {
            agent_id: agent_id.clone(),
            role,
            ticket_id,
            workspace_path: ws.path.clone(),
            workspace_name: ws.name.clone(),
            parent_key,
            started_at: Utc::now(),
            label,
            current_tools: Vec::new(),
        };
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(old) = map.remove(&agent_id) {
            old.cancel_token.cancel();
        }
        map.insert(
            agent_id,
            AgentEntry {
                generation,
                handle,
                cancel_token,
                current_tools: Vec::new(),
            },
        );
        generation
    }

    /// Register a tool as currently executing inside a running agent.
    ///
    /// The returned guard removes the entry on drop (RAII — guaranteed
    /// cleanup on completion AND on failure, including early returns and task
    /// cancellation). Each tool execution gets a unique instance id, so
    /// parallel read-only tools coexist in the entry's `current_tools` and
    /// each guard removes exactly its own entry.
    ///
    /// Generation-safety: the entry is only mutated when `generation` matches
    /// the current entry — a stale guard from a finished/restarted agent can
    /// never mutate a different (replacement) agent's live card.
    pub fn tool_started(
        &self,
        agent_id: &str,
        generation: u64,
        name: &str,
        args: &serde_json::Value,
    ) -> RunningToolGuard {
        let tool_id = NEXT_TOOL_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let args_str = serde_json::to_string(args).unwrap_or_default();
        let args_scrubbed = crate::util::scrub_credentials(&args_str);
        let args = crate::util::truncate_bytes(&args_scrubbed, MAX_LIVE_ARG_LENGTH).to_string();
        {
            let mut map = self.inner.lock().unwrap_poison();
            if let Some(entry) = map.get_mut(agent_id)
                && entry.generation == generation
            {
                entry.current_tools.push(RunningToolEntry {
                    id: tool_id,
                    tool: RunningTool {
                        name: name.to_string(),
                        args,
                    },
                });
            }
        }
        RunningToolGuard {
            agent_id: agent_id.to_string(),
            generation,
            tool_id,
        }
    }

    /// Remove a specific tool instance from the entry's live tools. No-op when
    /// the agent is gone or the generation no longer matches (stale guard).
    fn tool_finished(&self, agent_id: &str, generation: u64, tool_id: u64) {
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(entry) = map.get_mut(agent_id)
            && entry.generation == generation
        {
            entry.current_tools.retain(|t| t.id != tool_id);
        }
    }

    /// Cancel a specific agent by agent_id. Removes it from the registry.
    ///
    /// Prefer [`cancel_by_ticket_id`](AgentRegistry::cancel_by_ticket_id) or
    /// [`cancel_by_role_and_workspace_path`](AgentRegistry::cancel_by_role_and_workspace_path)
    /// for external callers — this method bypasses the generation-based safety check
    /// that guards against stale `agent_id` references.
    fn cancel(&self, agent_id: &str) {
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(entry) = map.remove(agent_id) {
            entry.cancel_token.cancel();
        }
    }

    /// Cancel all agents matching a predicate.
    ///
    /// The lock is dropped **before** calling [`cancel`](AgentRegistry::cancel) on each matched
    /// agent ID to avoid deadlock — `cancel` acquires the same lock internally.
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
        for agent_id in to_cancel {
            self.cancel(&agent_id);
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
            .map(|e| {
                let mut handle = e.handle.clone();
                handle.current_tools = e
                    .current_tools
                    .iter()
                    .map(RunningToolEntry::to_handle)
                    .collect();
                handle
            })
            .collect()
    }

    /// Check whether an agent with the given `agent_id` is registered.
    #[must_use]
    pub fn contains(&self, agent_id: &str) -> bool {
        self.inner.lock().unwrap_poison().contains_key(agent_id)
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
    pub fn deregister(&self, agent_id: &str, generation: u64) {
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(entry) = map.get(agent_id)
            && entry.generation == generation
        {
            map.remove(agent_id);
        }
    }
}

/// Maximum length of the serialized arguments stored in the live tool
/// instrumentation (Running Agents page). Longer arguments are truncated at a
/// UTF-8-safe boundary. Credentials are scrubbed BEFORE truncation so the
/// visible window never leaks a secret that truncation would otherwise hide.
const MAX_LIVE_ARG_LENGTH: usize = 200;

/// RAII guard: removes its tool instance from the agent's live tools on drop.
///
/// Creation is generation-gated (the entry is only mutated when the agent's
/// current generation matches); removal is generation-gated the same way — a
/// stale guard from a finished/restarted agent can never remove a tool from
/// the replacement agent's card.
pub struct RunningToolGuard {
    agent_id: String,
    generation: u64,
    tool_id: u64,
}

impl Drop for RunningToolGuard {
    fn drop(&mut self) {
        AGENT_REGISTRY.tool_finished(&self.agent_id, self.generation, self.tool_id);
    }
}

/// Global static registry.
pub static AGENT_REGISTRY: LazyLock<AgentRegistry> = LazyLock::new(AgentRegistry::default);

#[cfg(test)]
mod tests {
    use super::*;

    /// Register a bare agent in the GLOBAL registry and return its generation
    /// counter. Each test uses a unique agent_id (via [`crate::generate_suffix`]),
    /// so concurrent tests never collide on shared state.
    fn register_test_agent(agent_id: &str, ws_path: &str) -> u64 {
        let ws = crate::Workspace {
            name: agent_id.to_string(),
            path: ws_path.to_string(),
            ..Default::default()
        };
        AGENT_REGISTRY.register(
            agent_id.to_string(),
            "analyst".to_string(),
            None,
            &ws,
            "test".to_string(),
            CancellationToken::new(),
            None,
        )
    }

    #[test]
    fn tool_started_shows_in_list_and_guard_removes_it() {
        let agent_id = format!("tool_guard_{}", crate::generate_suffix());
        let generation = register_test_agent(&agent_id, "/tmp/ws");
        {
            let guard = AGENT_REGISTRY.tool_started(
                &agent_id,
                generation,
                "read",
                &serde_json::json!({"path": "/tmp/ws/file.rs"}),
            );
            let handles = AGENT_REGISTRY.list();
            let h = handles
                .iter()
                .find(|h| h.agent_id == agent_id)
                .expect("agent registered");
            assert_eq!(h.current_tools.len(), 1);
            assert_eq!(h.current_tools[0].name, "read");
            assert!(h.current_tools[0].args.contains("file.rs"));
            drop(guard);
        }
        let handles = AGENT_REGISTRY.list();
        let h = handles
            .iter()
            .find(|h| h.agent_id == agent_id)
            .expect("agent still registered");
        assert!(
            h.current_tools.is_empty(),
            "guard drop must remove the tool entry"
        );
        AGENT_REGISTRY.deregister(&agent_id, generation);
    }

    #[test]
    fn parallel_tools_coexist_and_remove_individually() {
        let agent_id = format!("tool_par_{}", crate::generate_suffix());
        let generation = register_test_agent(&agent_id, "/tmp/ws");
        let guard_a = AGENT_REGISTRY.tool_started(
            &agent_id,
            generation,
            "search",
            &serde_json::json!({"query": "alpha"}),
        );
        let guard_b = AGENT_REGISTRY.tool_started(
            &agent_id,
            generation,
            "read",
            &serde_json::json!({"path": "/tmp/ws/b.rs"}),
        );
        let handles = AGENT_REGISTRY.list();
        let h = handles
            .iter()
            .find(|h| h.agent_id == agent_id)
            .expect("agent registered");
        assert_eq!(
            h.current_tools.len(),
            2,
            "parallel tools must both be visible (honest representation)"
        );
        drop(guard_a);
        let handles = AGENT_REGISTRY.list();
        let h = handles
            .iter()
            .find(|h| h.agent_id == agent_id)
            .expect("agent registered");
        assert_eq!(h.current_tools.len(), 1);
        assert_eq!(h.current_tools[0].name, "read");
        drop(guard_b);
        AGENT_REGISTRY.deregister(&agent_id, generation);
    }

    #[test]
    fn stale_guard_cannot_mutate_replacement_agent() {
        let agent_id = format!("tool_stale_{}", crate::generate_suffix());
        let generation = register_test_agent(&agent_id, "/tmp/ws");
        // A tool starts, then the agent is deregistered (finished/restarted).
        let stale_guard = AGENT_REGISTRY.tool_started(
            &agent_id,
            generation,
            "shell",
            &serde_json::json!({"command": "ls"}),
        );
        AGENT_REGISTRY.deregister(&agent_id, generation);
        // Replacement registers with a NEW generation.
        let new_gen = register_test_agent(&agent_id, "/tmp/ws");
        // Stale guard drops — must NOT remove the replacement's tool.
        drop(stale_guard);
        let fresh_guard = AGENT_REGISTRY.tool_started(
            &agent_id,
            new_gen,
            "read",
            &serde_json::json!({"path": "/tmp/ws/new.rs"}),
        );
        let handles = AGENT_REGISTRY.list();
        let h = handles
            .iter()
            .find(|h| h.agent_id == agent_id)
            .expect("replacement registered");
        assert_eq!(
            h.current_tools.len(),
            1,
            "stale guard must not mutate the replacement's card"
        );
        assert_eq!(h.current_tools[0].name, "read");
        drop(fresh_guard);
        AGENT_REGISTRY.deregister(&agent_id, new_gen);
    }

    #[test]
    fn tool_args_are_credential_scrubbed_and_bounded() {
        let agent_id = format!("tool_scrub_{}", crate::generate_suffix());
        let generation = register_test_agent(&agent_id, "/tmp/ws");
        let secret = format!("token-{}", crate::generate_suffix());
        let long_args = serde_json::json!({
            "command": format!("echo {}", "a".repeat(1000)),
            "api_key": secret,
        });
        let guard = AGENT_REGISTRY.tool_started(&agent_id, generation, "shell", &long_args);
        let handles = AGENT_REGISTRY.list();
        let h = handles
            .iter()
            .find(|h| h.agent_id == agent_id)
            .expect("agent registered");
        assert_eq!(h.current_tools.len(), 1);
        assert!(
            !h.current_tools[0].args.contains(&secret),
            "credentials must never leak into the live view"
        );
        assert!(
            h.current_tools[0].args.len() <= MAX_LIVE_ARG_LENGTH + 16,
            "args must be bounded: {}",
            h.current_tools[0].args.len()
        );
        drop(guard);
        AGENT_REGISTRY.deregister(&agent_id, generation);
    }

    #[test]
    fn parent_key_and_workspace_name_round_trip() {
        let agent_id = format!("parent_key_{}", crate::generate_suffix());
        let ws = crate::Workspace {
            name: "ws_parent".to_string(),
            path: "/tmp/ws_parent".to_string(),
            ..Default::default()
        };
        let generation = AGENT_REGISTRY.register(
            agent_id.clone(),
            "engineer".to_string(),
            Some("T-42".to_string()),
            &ws,
            "label".to_string(),
            CancellationToken::new(),
            Some(ParentKey::Ticket("T-42".to_string())),
        );
        let handles = AGENT_REGISTRY.list();
        let h = handles
            .iter()
            .find(|h| h.agent_id == agent_id)
            .expect("agent registered");
        assert_eq!(h.workspace_name, "ws_parent");
        assert_eq!(h.parent_key, Some(ParentKey::Ticket("T-42".to_string())));
        AGENT_REGISTRY.deregister(&agent_id, generation);
    }
}
