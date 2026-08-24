//! Global registry of running agents with cancellation support.
//!
//! Also tracks in-flight non-agent (single-shot utility) LLM calls —
//! consolidation, research-orchestration passes, joint-verdict synthesis.
//! These register here on start and remove themselves on completion via a RAII
//! guard. Calls originating inside an agent run (the agent loop, verdict
//! extraction, summarization) never register; agents are tracked separately in
//! [`AGENT_REGISTRY`]. The call tracking is purely observational — it carries
//! no cancellation semantics and never affects call behavior, retries, or
//! results.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

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
/// - [`AnalyzeRound`](ParentKey::AnalyzeRound) — the parallel analysts of one analyze
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
    AnalyzeRound(String),
    Research(String),
}

/// One tool currently executing inside a running agent — the live
/// instrumentation shown on the Running Agents page.
///
/// `args` holds the tool's arguments as STRUCTURED key-value pairs with FULL,
/// untruncated values, deliberately NOT credential-scrubbed: the Running
/// Agents view shows exactly what the agent passed to the tool, including any
/// secrets (the user wants complete visibility). This is a live-view-only
/// divergence — durable logs (tool-call stats) and failure feedback remain
/// scrubbed.
///
/// Pairs are derived from the JSON argument object at registration (keys in
/// the object's iteration order): string values are taken verbatim, nested
/// objects/arrays/scalars are compact-JSON-stringified. Non-object arguments
/// (bare strings/arrays/null) collapse to a single `("args", …)` pair. The
/// row view truncates values at display time; the hover tooltip shows
/// everything untruncated.
#[derive(Clone, Debug, Serialize)]
pub struct RunningTool {
    pub name: String,
    pub args: Vec<(String, String)>,
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
    /// Human-readable label for the DIRECT PARENT INVOCATION this agent
    /// belongs to (ticket title / analyze question / research question) —
    /// shown on the group header of the Running Agents view. `None` for
    /// workspace singletons. Purely presentational; never affects pipeline
    /// behavior.
    pub parent_label: Option<String>,
    pub started_at: DateTime<Utc>,
    pub label: String,
    /// Snapshot of the tools currently executing inside this agent, taken at
    /// `list()` time. Empty when the agent is between tool executions (in an
    /// LLM call, between rounds) — absence is honest.
    pub current_tools: Vec<RunningTool>,
    /// The most recent tool that COMPLETED inside this agent (kept until a
    /// newer tool replaces it) — lets the Running Agents page keep showing a
    /// finished fast tool instead of flashing it and vanishing. Recorded on
    /// every completion (success or failure); purely observational.
    pub last_tool: Option<RunningTool>,
    /// Live activity indicator for non-tool LLM phases inside this agent
    /// (e.g. "extracting", "summarizing", "transcribing"), taken at `list()`
    /// time. `None` between phases — the agent's card is the single tracker
    /// for these calls (no separate registry rows are ever created for them).
    pub activity: Option<String>,
    /// Real provider-reported session length (input + output tokens of the
    /// agent's last successful LLM call), mirrored here observationally from
    /// the durable per-session value. `None` until the first successful
    /// usage-bearing call of the agent's life (new / pre-migration sessions
    /// — approved no-backfill). Purely presentational: the Running Agents
    /// page reads the registry, never the database.
    pub session_tokens: Option<u64>,
}

/// Agent identity + registry generation propagated to tool execution via a
/// task-local, so tool-internal LLM calls (e.g. media transcription in video
/// tool results) can attribute themselves to the owning agent's card and
/// telemetry. `None` outside an agent run (inbound enrichment, tests).
#[derive(Clone, Debug)]
pub(crate) struct AgentTracking {
    pub agent_id: String,
    pub generation: u64,
    pub role: String,
    pub workspace: String,
}

struct AgentEntry {
    generation: u64,
    handle: AgentHandle,
    cancel_token: CancellationToken,
    /// Whether a GENUINE user/operator stop was requested for this agent.
    /// Set only by [`AgentRegistry::cancel_by_ticket_id_user`]; code-driven
    /// internal cancellations (re-dispatch, register replacement, phase
    /// transition/supersede) fire only [`Self::cancel_token`].
    user_stop: Arc<AtomicBool>,
    /// Live tool instrumentation for this agent (mutable across the agent's
    /// lifetime; snapshot into [`AgentHandle::current_tools`] by `list()`).
    /// Every access is already serialized by the outer
    /// [`AgentRegistry::inner`] lock (tool_started/tool_finished/list all
    /// hold it), so the Vec needs no further synchronization. Each entry
    /// carries the unique tool instance id for exact removal.
    current_tools: Vec<RunningToolEntry>,
    /// The most recent COMPLETED tool — kept after `tool_finished` removes
    /// the entry from `current_tools` so the Running Agents page can keep
    /// showing it until a newer tool replaces it (fast tools no longer flash
    /// and vanish). Snapshot into [`AgentHandle::last_tool`] by `list()`.
    last_tool: Option<RunningTool>,
    /// Live activity label (e.g. "extracting", "transcribing") for the
    /// non-tool LLM phase currently running inside this agent; `None` between
    /// phases. Single slot: an agent runs sequentially, so at most one
    /// activity can be live at a time. Snapshot into
    /// [`AgentHandle::activity`] by `list()`.
    activity: Option<String>,
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

/// Flatten a tool's JSON arguments into structured key-value display pairs.
///
/// Object arguments become one pair per key (keys in the object's iteration
/// order); anything else (bare strings, arrays, scalars, null) collapses to a
/// single `("args", …)` pair. Values are full and unscrubbed — see
/// [`RunningTool`].
fn tool_arg_pairs(args: &serde_json::Value) -> Vec<(String, String)> {
    match args {
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(k, v)| (k.clone(), json_value_to_display_string(v)))
            .collect(),
        other => vec![("args".to_string(), json_value_to_display_string(other))],
    }
}

/// Render one JSON argument value as a display string: string values verbatim,
/// everything else (nested objects/arrays/scalars) as compact JSON.
fn json_value_to_display_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
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
    ///
    /// `parent_label` is the human-readable label of the agent's DIRECT PARENT
    /// INVOCATION (ticket title / analyze question / research question) for the
    /// Running Agents group header — purely presentational.
    #[expect(clippy::too_many_arguments)] // one positional arg per handle field; callers use literals
    pub fn register(
        &self,
        agent_id: String,
        role: String,
        ticket_id: Option<String>,
        ws: &crate::Workspace,
        label: String,
        cancel_token: CancellationToken,
        parent_key: Option<ParentKey>,
        parent_label: Option<String>,
        user_stop: Arc<AtomicBool>,
    ) -> u64 {
        let generation = NEXT_ENTRY_GENERATION.fetch_add(1, Ordering::Relaxed);
        let handle = AgentHandle {
            agent_id: agent_id.clone(),
            role,
            ticket_id,
            workspace_path: ws.path.clone(),
            workspace_name: ws.name.clone(),
            parent_key,
            parent_label,
            started_at: Utc::now(),
            label,
            current_tools: Vec::new(),
            last_tool: None,
            activity: None,
            session_tokens: None,
        };
        let mut map = self.inner.lock().unwrap_poison();
        // Replacing an old agent cancels the old token but is an INTERNAL
        // (non-user) cancellation — the new agent is a re-dispatch/replacement,
        // not a genuine user stop, so `old.user_stop` is deliberately never set.
        if let Some(old) = map.remove(&agent_id) {
            old.cancel_token.cancel();
        }
        map.insert(
            agent_id,
            AgentEntry {
                generation,
                handle,
                cancel_token,
                user_stop,
                current_tools: Vec::new(),
                last_tool: None,
                activity: None,
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
        let tool_id = NEXT_TOOL_ID.fetch_add(1, Ordering::Relaxed);
        let args = tool_arg_pairs(args);
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

    /// Remove a specific tool instance from the entry's live tools and record
    /// it as the entry's last COMPLETED tool (kept until a newer tool replaces
    /// it — the Running Agents page shows running tools if any, else the last
    /// completed one). No-op when the agent is gone or the generation no longer
    /// matches (stale guard).
    fn tool_finished(&self, agent_id: &str, generation: u64, tool_id: u64) {
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(entry) = map.get_mut(agent_id)
            && entry.generation == generation
            && let Some(pos) = entry.current_tools.iter().position(|t| t.id == tool_id)
        {
            let removed = entry.current_tools.remove(pos);
            entry.last_tool = Some(removed.tool);
        }
    }

    /// Register a live activity label on a running agent's card (e.g.
    /// "extracting" during a verdict extraction, "transcribing" during a
    /// media-transcription LLM call inside a tool). Purely observational — no
    /// cancellation semantics.
    ///
    /// The returned guard clears the label on drop (RAII — guaranteed cleanup
    /// on completion AND on failure, including early returns and task
    /// cancellation). Generation-safety: the entry is only mutated when
    /// `generation` matches the current entry — a stale guard from a
    /// finished/restarted agent can never mutate a replacement agent's card.
    ///
    /// Single-slot semantics: an agent runs sequentially, so at most one
    /// activity is live at a time; a second registration overwrites the label
    /// and both guards clear on drop (the slot is `Option`-reset, not
    /// reference-counted).
    pub fn activity_started(&self, agent_id: &str, generation: u64, label: &str) -> ActivityGuard {
        {
            let mut map = self.inner.lock().unwrap_poison();
            if let Some(entry) = map.get_mut(agent_id)
                && entry.generation == generation
            {
                entry.activity = Some(label.to_string());
            }
        }
        ActivityGuard {
            agent_id: agent_id.to_string(),
            generation,
        }
    }

    /// Clear the entry's live activity label. No-op when the agent is gone or
    /// the generation no longer matches (stale guard).
    fn activity_finished(&self, agent_id: &str, generation: u64) {
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(entry) = map.get_mut(agent_id)
            && entry.generation == generation
        {
            entry.activity = None;
        }
    }

    /// Update the observational session-length metric on a live agent card.
    ///
    /// The value is the REAL provider-reported length (input + output tokens)
    /// of the agent's last successful LLM call, persisted durably per session
    /// by the caller and mirrored here for display only. Generation-safety: a
    /// stale writer from a finished/restarted agent can never mutate a
    /// replacement agent's card.
    pub fn set_session_tokens(&self, agent_id: &str, generation: u64, token_length: u64) {
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(entry) = map.get_mut(agent_id)
            && entry.generation == generation
        {
            entry.handle.session_tokens = Some(token_length);
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

    /// Cancel all agents running for a `ticket_id` as a GENUINE user/operator
    /// stop: sets the user-stop flag on each matched agent AND fires its
    /// cancel token. Distinct from [`Self::cancel_by_ticket_id`], which is the
    /// code-driven/internal cancellation used by re-dispatch, phase
    /// transitions, supersede, and claim. Follows the same lock-ordering
    /// invariant as [`cancel_matching`] (set the flag under the lock, cancel
    /// outside it).
    pub fn cancel_by_ticket_id_user(&self, ticket_id: &str) {
        let to_cancel: Vec<String> = {
            let map = self.inner.lock().unwrap_poison();
            let mut ids = Vec::new();
            for (id, entry) in map.iter() {
                if entry.handle.ticket_id.as_deref() == Some(ticket_id) {
                    entry.user_stop.store(true, Ordering::SeqCst);
                    ids.push(id.clone());
                }
            }
            ids
        };
        for agent_id in to_cancel {
            self.cancel(&agent_id);
        }
    }

    /// Cancel all agents running for a specific role within a specific workspace path.
    /// Used when maintenance is disabled for a workspace — stops the in-flight maintainer agent.
    pub fn cancel_by_role_and_workspace_path(&self, role: &str, ws_path: &str) {
        self.cancel_matching(|entry| {
            entry.handle.role == role && entry.handle.workspace_path == ws_path
        });
    }

    /// Cancel every agent belonging to a direct parent invocation (ticket /
    /// analyze round / research run) as a group. Used by the research
    /// manual-cancel path to stop the whole run (analysts, verifier, coder,
    /// and the cleanup agent — they share the run's [`ParentKey::Research`]).
    ///
    /// Follows the lock-ordering invariant documented on
    /// [`cancel_matching`](AgentRegistry::cancel_matching): the predicate runs
    /// under the lock, cancellation outside it.
    pub fn cancel_by_parent_key(&self, parent: &ParentKey) {
        let parent = parent.clone();
        self.cancel_matching(move |entry| entry.handle.parent_key.as_ref() == Some(&parent));
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
                handle.last_tool.clone_from(&e.last_tool);
                handle.activity.clone_from(&e.activity);
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

/// RAII guard: clears the agent's live activity label on drop.
///
/// Creation is generation-gated; removal is generation-gated the same way — a
/// stale guard from a finished/restarted agent can never clear the
/// replacement agent's card.
pub struct ActivityGuard {
    agent_id: String,
    generation: u64,
}

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        AGENT_REGISTRY.activity_finished(&self.agent_id, self.generation);
    }
}

/// Global static registry.
pub static AGENT_REGISTRY: LazyLock<AgentRegistry> = LazyLock::new(AgentRegistry::default);

/// Monotonically increasing entry id — lets each guard remove exactly its own
/// entry on drop, regardless of drop order.
static NEXT_ENTRY_ID: AtomicU64 = AtomicU64::new(1);

/// Public snapshot of one in-flight call — serializable, no internals exposed.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct NonAgentCallHandle {
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
pub(crate) struct NonAgentCallRegistry {
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

    /// Remove every in-flight call belonging to a direct parent invocation
    /// (ticket / analyze round / research run). Purely observational — the
    /// RAII guards' drops become no-ops. Used by the research manual-cancel
    /// path so the run's group disappears from Running Agents immediately
    /// instead of lingering until the orchestrator task returns (which can be
    /// minutes mid-LLM-call).
    pub fn remove_by_parent_key(&self, parent: &ParentKey) {
        let parent = parent.clone();
        self.inner
            .lock()
            .unwrap_poison()
            .retain(|_, h| h.parent_key.as_ref() != Some(&parent));
    }
}

/// RAII guard: removes its registry entry on drop.
pub(crate) struct NonAgentCallGuard {
    id: u64,
    registry: &'static NonAgentCallRegistry,
}

impl Drop for NonAgentCallGuard {
    fn drop(&mut self) {
        self.registry.inner.lock().unwrap_poison().remove(&self.id);
    }
}

/// Global static registry.
pub(crate) static NON_AGENT_CALLS: LazyLock<NonAgentCallRegistry> =
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
pub(crate) fn call_kind_label(kind: &str) -> &'static str {
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
            None,
            std::sync::Arc::new(AtomicBool::new(false)),
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
            assert!(
                h.current_tools[0]
                    .args
                    .iter()
                    .any(|(k, v)| k == "path" && v == "/tmp/ws/file.rs")
            );
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
    fn activity_shows_in_list_and_guard_clears_it() {
        let agent_id = format!("activity_guard_{}", crate::generate_suffix());
        let generation = register_test_agent(&agent_id, "/tmp/ws");
        {
            let guard = AGENT_REGISTRY.activity_started(&agent_id, generation, "extracting");
            let handles = AGENT_REGISTRY.list();
            let h = handles
                .iter()
                .find(|h| h.agent_id == agent_id)
                .expect("agent registered");
            assert_eq!(
                h.activity.as_deref(),
                Some("extracting"),
                "activity must be visible on the live card"
            );
            drop(guard);
        }
        let handles = AGENT_REGISTRY.list();
        let h = handles
            .iter()
            .find(|h| h.agent_id == agent_id)
            .expect("agent still registered");
        assert!(
            h.activity.is_none(),
            "guard drop must clear the activity label"
        );
        AGENT_REGISTRY.deregister(&agent_id, generation);
    }

    #[test]
    fn stale_activity_guard_cannot_clear_replacement_agent() {
        let agent_id = format!("activity_stale_{}", crate::generate_suffix());
        let generation = register_test_agent(&agent_id, "/tmp/ws");
        // An activity starts, then the agent is deregistered (finished/restarted).
        let stale_guard = AGENT_REGISTRY.activity_started(&agent_id, generation, "extracting");
        AGENT_REGISTRY.deregister(&agent_id, generation);
        // Replacement registers with a NEW generation and starts its own activity.
        let new_gen = register_test_agent(&agent_id, "/tmp/ws");
        let fresh_guard = AGENT_REGISTRY.activity_started(&agent_id, new_gen, "transcribing");
        // Stale guard drops — must NOT clear the replacement's activity.
        drop(stale_guard);
        let handles = AGENT_REGISTRY.list();
        let h = handles
            .iter()
            .find(|h| h.agent_id == agent_id)
            .expect("replacement agent registered");
        assert_eq!(
            h.activity.as_deref(),
            Some("transcribing"),
            "stale guard must not clear the replacement's activity"
        );
        drop(fresh_guard);
        AGENT_REGISTRY.deregister(&agent_id, new_gen);
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
    fn tool_args_are_full_and_structured() {
        let agent_id = format!("tool_full_{}", crate::generate_suffix());
        let generation = register_test_agent(&agent_id, "/tmp/ws");
        let secret = format!("token-{}", crate::generate_suffix());
        let long_command = format!("echo {}", "a".repeat(1000));
        let args = serde_json::json!({
            "command": long_command,
            "api_key": secret,
        });
        let guard = AGENT_REGISTRY.tool_started(&agent_id, generation, "shell", &args);
        let handles = AGENT_REGISTRY.list();
        let h = handles
            .iter()
            .find(|h| h.agent_id == agent_id)
            .expect("agent registered");
        assert_eq!(h.current_tools.len(), 1);
        let pairs = &h.current_tools[0].args;
        assert!(
            pairs.iter().any(|(k, v)| k == "api_key" && v == &secret),
            "live view shows full unscrubbed values (deliberate divergence from durable logs)"
        );
        assert!(
            pairs
                .iter()
                .any(|(k, v)| k == "command" && v == &long_command),
            "values are not truncated at registration"
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
            Some("Fix login bug".to_string()),
            std::sync::Arc::new(AtomicBool::new(false)),
        );
        let handles = AGENT_REGISTRY.list();
        let h = handles
            .iter()
            .find(|h| h.agent_id == agent_id)
            .expect("agent registered");
        assert_eq!(h.workspace_name, "ws_parent");
        assert_eq!(h.parent_key, Some(ParentKey::Ticket("T-42".to_string())));
        assert_eq!(
            h.parent_label.as_deref(),
            Some("Fix login bug"),
            "parent label round-trips"
        );
        AGENT_REGISTRY.deregister(&agent_id, generation);
    }

    #[test]
    fn completed_tool_is_kept_as_last_tool_until_replaced() {
        let agent_id = format!("last_tool_{}", crate::generate_suffix());
        let generation = register_test_agent(&agent_id, "/tmp/ws");
        // First tool completes → becomes the last tool.
        {
            let guard = AGENT_REGISTRY.tool_started(
                &agent_id,
                generation,
                "read",
                &serde_json::json!({"path": "/tmp/ws/a.rs"}),
            );
            drop(guard);
        }
        let handles = AGENT_REGISTRY.list();
        let h = handles
            .iter()
            .find(|h| h.agent_id == agent_id)
            .expect("agent registered");
        assert!(h.current_tools.is_empty(), "no running tools left");
        let last = h
            .last_tool
            .as_ref()
            .expect("completed tool kept as last_tool");
        assert_eq!(last.name, "read");
        assert!(
            last.args
                .iter()
                .any(|(k, v)| k == "path" && v == "/tmp/ws/a.rs")
        );
        // A second completion replaces the first.
        {
            let guard = AGENT_REGISTRY.tool_started(
                &agent_id,
                generation,
                "edit",
                &serde_json::json!({"path": "/tmp/ws/b.rs"}),
            );
            drop(guard);
        }
        let handles = AGENT_REGISTRY.list();
        let h = handles
            .iter()
            .find(|h| h.agent_id == agent_id)
            .expect("agent registered");
        assert_eq!(
            h.last_tool.as_ref().map(|t| t.name.as_str()),
            Some("edit"),
            "newer completed tool replaces the old one"
        );
        AGENT_REGISTRY.deregister(&agent_id, generation);
    }

    #[test]
    fn cancel_by_parent_key_cancels_matching_agents_only() {
        // Two agents of runA + one agent of runB — a group cancel of runA must
        // fire exactly the runA tokens and leave runB untouched.
        let a1 = format!("cancel_parent_a1_{}", crate::generate_suffix());
        let a2 = format!("cancel_parent_a2_{}", crate::generate_suffix());
        let b1 = format!("cancel_parent_b1_{}", crate::generate_suffix());
        let ws = crate::Workspace {
            name: "ws_cancel_parent".to_string(),
            path: "/tmp/ws_cancel_parent".to_string(),
            ..Default::default()
        };
        let register_with_parent = |agent_id: &str, parent: ParentKey| {
            AGENT_REGISTRY.register(
                agent_id.to_string(),
                "analyst".to_string(),
                None,
                &ws,
                "test".to_string(),
                CancellationToken::new(),
                Some(parent),
                None,
                std::sync::Arc::new(AtomicBool::new(false)),
            )
        };
        let gen_a1 = register_with_parent(&a1, ParentKey::Research("runA".to_string()));
        let gen_a2 = register_with_parent(&a2, ParentKey::Research("runA".to_string()));
        let gen_b = register_with_parent(&b1, ParentKey::Research("runB".to_string()));

        AGENT_REGISTRY.cancel_by_parent_key(&ParentKey::Research("runA".to_string()));

        // runA agents are gone from the registry (cancel removes the entry);
        // runB survives.
        assert!(
            !AGENT_REGISTRY.contains(&a1) && !AGENT_REGISTRY.contains(&a2),
            "runA agents cancelled (entry removed)"
        );
        assert!(
            AGENT_REGISTRY.contains(&b1),
            "runB agent untouched by the runA group cancel"
        );
        let b = AGENT_REGISTRY
            .list()
            .into_iter()
            .find(|h| h.agent_id == b1)
            .expect("runB agent still registered");
        assert_eq!(
            b.parent_key,
            Some(ParentKey::Research("runB".to_string())),
            "runB agent keeps its parent key"
        );
        AGENT_REGISTRY.deregister(&b1, gen_b);
        let _ = (gen_a1, gen_a2);
    }

    #[test]
    fn register_carries_parent_key_and_run_lifetime() {
        // Unique test-only kinds: `NON_AGENT_CALLS` is a process-global shared
        // by parallel tests (analyze consolidation and research orchestration
        // register the literal "consolidate"/"research_orchestrator" kinds),
        // so a non-unique kind would make this test flakily observe another
        // test's live registration and fail its post-drop emptiness assertion.
        let kind: &'static str = "registry_test_consolidate_kind";
        let guard = NON_AGENT_CALLS.register(
            kind,
            "ws1",
            Some(ParentKey::AnalyzeRound("round_1".to_string())),
            false,
            Some("Why is CI flaky?".to_string()),
        );
        // The run-lifetime flag is a distinct registration mode (whole-run
        // orchestrator guards); both must round-trip through the same path.
        let orchestrator: &'static str = "registry_test_orchestrator_kind";
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

    #[test]
    fn remove_by_parent_key_removes_matching_calls_only() {
        // Two calls of runA + one call of runB — removing runA must drop its
        // calls from the live list and leave runB's call visible.
        let run_a1 = NON_AGENT_CALLS.register(
            "consolidate",
            "ws1",
            Some(ParentKey::Research("runA".to_string())),
            false,
            None,
        );
        let run_a2 = NON_AGENT_CALLS.register(
            "synthesis",
            "ws1",
            Some(ParentKey::Research("runA".to_string())),
            false,
            None,
        );
        let run_b = NON_AGENT_CALLS.register(
            "synthesize",
            "ws1",
            Some(ParentKey::Research("runB".to_string())),
            false,
            None,
        );

        NON_AGENT_CALLS.remove_by_parent_key(&ParentKey::Research("runA".to_string()));

        let handles = NON_AGENT_CALLS.list();
        assert!(
            !handles
                .iter()
                .any(|h| h.parent_key == Some(ParentKey::Research("runA".to_string()))),
            "runA calls removed from the live list"
        );
        assert!(
            handles
                .iter()
                .any(|h| h.parent_key == Some(ParentKey::Research("runB".to_string()))),
            "runB call still visible"
        );
        // The guards' drops are now no-ops (their entries are already gone) —
        // dropping them must not panic and must not disturb anything else.
        drop(run_a1);
        drop(run_a2);
        drop(run_b);
    }
}
