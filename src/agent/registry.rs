//! Global registry of running agents with cancellation support. Also tracks in-flight non-agent
//! (single-shot utility) LLM calls — consolidation, research-orchestration passes, joint-verdict
//! synthesis — which register here on start and remove themselves through a RAII guard. Calls
//! originating inside an agent run never register; agents are tracked separately in
//! [`AGENT_REGISTRY`], and in-agent utility calls surface on the agent's card as an activity label
//! ([`AgentRegistry::activity_started`]). The call tracking is purely observational — no
//! cancellation semantics, no effect on behavior, retries or results. Every mutation that changes
//! the live view publishes
//! [`RuntimeEvent::Registries`](crate::runtime_events::RuntimeEvent::Registries) so the GUI
//! re-renders.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::runtime_events::{RuntimeEvent, publish};
use crate::util::UnwrapPoison;
use chrono::{DateTime, Utc};
use serde::Serialize;
use tokio_util::sync::CancellationToken;

/// Monotonically increasing generation counter for registry entries, so
/// [`deregister`](AgentRegistry::deregister) can detect stale entries: a new agent registered
/// under the same `agent_id` (the Manager interrupt-and-resume pattern) leaves the old
/// entry's generation unmatched, and the old entry cannot remove the replacement.
static NEXT_ENTRY_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Grouping key for the Running Agents live view: the DIRECT PARENT INVOCATION a running work
/// item belongs to.
///
/// - [`Ticket`](ParentKey::Ticket) — the agents on one ticket plus LLM calls belonging to its
///   own work (the joint-verdict synthesis of a verification round).
/// - [`AnalyzeRound`](ParentKey::AnalyzeRound) — the parallel analysts of one round plus its
///   consolidation call (members share the round key).
/// - [`Research`](ParentKey::Research) — sub-agents and orchestrator calls of one research
///   run (members share the durable job id).
/// - [`Launch`](ParentKey::Launch) — one branch launch, keyed by the launch's agent id (one
///   launch is one conversation).
///
/// `None` means a workspace singleton (manager / maintainer / discovery / direct chat) or, for
/// non-agent calls, genuinely unattributable orchestrator work.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum ParentKey {
    Ticket(String),
    AnalyzeRound(String),
    Research(String),
    Launch(String),
}

/// One tool call in the GUI's session-ledger projection (`session_view`) — the payload of a
/// ledger tool block, consumed by the session views and the Running Agents cards.
///
/// `args` holds the arguments as STRUCTURED key-value pairs with FULL, untruncated values:
/// the Running Agents view shows exactly what the agent passed, secrets included (the user
/// wants complete visibility). Pairs come from the JSON argument object at decode time:
/// strings verbatim, nested values compact-JSON-stringified, non-object arguments collapsed
/// to a single `("args", …)` pair.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RunningTool {
    pub name: String,
    pub args: Vec<(String, String)>,
}

impl RunningTool {
    /// Build a `RunningTool` from a decoded native-history tool call: full argument pairs from
    /// the call's JSON arguments.
    pub(crate) fn from_tool_call(call: &crate::ToolCall) -> Self {
        Self {
            name: call.name.clone(),
            args: tool_arg_pairs(&call.arguments),
        }
    }
}

/// Public handle returned by `list()` — serializable, no cancel_token exposed.
#[derive(Clone, Debug, Serialize)]
pub struct AgentHandle {
    pub agent_id: String,
    pub role: String,
    pub ticket_id: Option<String>,
    /// Filesystem path of the workspace (not the name) — distinct from `workspace_name`, and
    /// not used for display: the only reader is
    /// [`cancel_by_role_and_workspace_path`](AgentRegistry::cancel_by_role_and_workspace_path).
    pub workspace_path: String,
    /// Workspace NAME — the identifier the board database and the dashboard's workspace map
    /// use. Displayed on cards and group headers, and used for paused-state lookups.
    pub workspace_name: String,
    /// Grouping key for the Running Agents view — see [`ParentKey`].
    pub parent_key: Option<ParentKey>,
    /// Human-readable label for the DIRECT PARENT INVOCATION this agent belongs to (ticket
    /// title / analyze question / research question), shown on the group header. `None` for
    /// workspace singletons. Purely presentational.
    pub parent_label: Option<String>,
    pub started_at: DateTime<Utc>,
    /// Registry generation for this run — the Running Agents page keys expanded state by
    /// (agent_id, generation) so a recycled agent_id never inherits a stale expansion.
    pub generation: u64,
    /// Live activity indicator for non-tool LLM phases inside this agent (e.g. "extracting",
    /// "transcribing"), taken at `list()` time. `None` between phases.
    pub activity: Option<String>,
    /// Whether a workspace-pause freeze was signalled for this run, taken at `list()` time. A
    /// frozen agent stops at its next LLM round boundary, so it is not working: the Sessions
    /// page keeps its row in place instead of pinning it to the top.
    pub paused: bool,
}

/// The caller's identity for the warning row a tool makes about itself. The agent id is
/// [`crate::agent::CURRENT_TOOL_AGENT_ID`]'s and is read from there, never carried twice. `None`
/// outside an agent run.
#[derive(Clone, Debug)]
pub(crate) struct AgentTracking {
    pub role: String,
    /// The ticket the run works on — empty when it has none. The log row's ticket scope for a
    /// record that never travels through the agent's span.
    pub ticket: String,
}

struct AgentEntry {
    generation: u64,
    handle: AgentHandle,
    cancel_token: CancellationToken,
    /// The turn's urgent-message interrupt (see [`AgentRegistry::interrupt`]). Fired, never
    /// removed: it dies with the entry.
    interrupt_token: CancellationToken,
    /// Whether a workspace-pause (strict freeze) cancellation was requested for this agent — a
    /// FREEZE distinct from a code-driven cancellation.
    pause_stop: Arc<AtomicBool>,
    /// Live activity label for the non-tool LLM phase currently running inside this agent;
    /// `None` between phases. Single slot: an agent runs sequentially. Snapshot into
    /// [`AgentHandle::activity`] by `list()`.
    activity: Option<String>,
}

/// Flatten a tool's JSON arguments into structured key-value display pairs: object arguments
/// become one pair per key (in iteration order), anything else collapses to a single
/// `("args", …)` pair. Values are full — see [`RunningTool`].
fn tool_arg_pairs(args: &serde_json::Value) -> Vec<(String, String)> {
    match args {
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(k, v)| (k.clone(), json_value_to_display_string(v)))
            .collect(),
        other => vec![("args".to_string(), json_value_to_display_string(other))],
    }
}

/// Render one JSON argument value as a display string: strings verbatim, everything else as
/// compact JSON.
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
    /// Used by [`crate::Agent::new`], where deregistration is handled by [`crate::Agent::drop`]
    /// instead of a guard. `parent_label` is the parent invocation's human-readable label, and
    /// `interrupt_token` the turn's urgent-message interrupt, stored so the router can fire it
    /// by agent id without knowing the run.
    #[expect(clippy::too_many_arguments)] // one positional arg per handle field; callers use literals
    pub fn register(
        &self,
        agent_id: String,
        role: String,
        ticket_id: Option<String>,
        ws: &crate::Workspace,
        cancel_token: CancellationToken,
        interrupt_token: CancellationToken,
        parent_key: Option<ParentKey>,
        parent_label: Option<String>,
        pause_stop: Arc<AtomicBool>,
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
            generation,
            activity: None,
            paused: false,
        };
        let mut map = self.inner.lock().unwrap_poison();
        // Replacing an old agent cancels the old token — the new agent is a replacement.
        if let Some(old) = map.remove(&agent_id) {
            old.cancel_token.cancel();
        }
        map.insert(
            agent_id,
            AgentEntry {
                generation,
                handle,
                cancel_token,
                interrupt_token,
                pause_stop,
                activity: None,
            },
        );
        publish(RuntimeEvent::Registries);
        generation
    }

    /// Cut the turn an urgent user message interrupted: fire the interrupt of the agent
    /// running under `agent_id`, if any. Not a cancellation — the token is the turn's own, so
    /// the run settles its in-flight tool calls and finalizes its session. A concluded turn, or
    /// no run at all, has nothing to cut.
    pub fn interrupt(&self, agent_id: &str) {
        let map = self.inner.lock().unwrap_poison();
        if let Some(entry) = map.get(agent_id) {
            entry.interrupt_token.cancel();
        }
    }

    /// Register a live activity label on a running agent's card (e.g. "extracting",
    /// "transcribing"). Purely observational — no cancellation semantics.
    ///
    /// The returned guard clears the label on drop (guaranteed on completion, failure and task
    /// cancellation). Generation-gated both ways: a stale guard can never mutate a replacement
    /// agent's card. Single slot: a second registration overwrites the label, and both guards
    /// clear on drop.
    pub(crate) fn activity_started(
        &self,
        agent_id: &str,
        generation: u64,
        label: &str,
    ) -> ActivityGuard {
        {
            let mut map = self.inner.lock().unwrap_poison();
            if let Some(entry) = map.get_mut(agent_id)
                && entry.generation == generation
            {
                entry.activity = Some(label.to_string());
                publish(RuntimeEvent::Registries);
            }
        }
        ActivityGuard {
            agent_id: agent_id.to_string(),
            generation,
        }
    }

    /// Clear the entry's live activity label. No-op when the agent is gone or the generation
    /// no longer matches.
    fn activity_finished(&self, agent_id: &str, generation: u64) {
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(entry) = map.get_mut(agent_id)
            && entry.generation == generation
        {
            entry.activity = None;
            publish(RuntimeEvent::Registries);
        }
    }

    /// Cancel a specific agent by agent_id, removing it from the registry. Prefer
    /// [`cancel_by_ticket_id`](AgentRegistry::cancel_by_ticket_id) or
    /// [`cancel_by_role_and_workspace_path`](AgentRegistry::cancel_by_role_and_workspace_path):
    /// those match on a stable predicate, while this bypasses the generation safety check.
    fn cancel(&self, agent_id: &str) {
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(entry) = map.remove(agent_id) {
            entry.cancel_token.cancel();
            publish(RuntimeEvent::Registries);
        }
    }

    /// Cancel all agents matching a predicate.
    ///
    /// # Lock-ordering invariant
    ///
    /// The predicate is evaluated while the lock is held, then the lock is released and
    /// cancellation proceeds without it — `cancel` acquires the same lock. Any future
    /// `cancel_by_*` method MUST follow this pattern or risk deadlock.
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

    /// Cancel all agents running for a `ticket_id` — used on ticket phase transitions.
    pub fn cancel_by_ticket_id(&self, ticket_id: &str) {
        self.cancel_matching(|entry| entry.handle.ticket_id.as_deref() == Some(ticket_id));
    }

    /// Cancel all agents running for a `workspace_name` as a workspace-pause (strict freeze):
    /// sets the pause-stop flag on each matched agent WITHOUT firing its cancel token, so its
    /// current round finishes and it stops cooperatively at the next LLM round boundary. The
    /// flag lets the failure classifier treat a frozen run as a FREEZE (leave the ticket for
    /// unpause), not a failure or a cancellation.
    ///
    /// Scoped to TICKET-parented pipeline agents — managers, maintainers and on-demand
    /// analyze/research sub-agents (`ticket_id = None`) keep running, so a pause only halts the
    /// ticket pipeline.
    pub fn cancel_by_workspace_pause(&self, workspace_name: &str) {
        let mut changed = false;
        {
            let map = self.inner.lock().unwrap_poison();
            for entry in map.values() {
                if entry.handle.workspace_name == workspace_name && entry.handle.ticket_id.is_some()
                {
                    changed |= !entry.pause_stop.swap(true, Ordering::SeqCst);
                }
            }
        }
        // No cancel token fired: the pause is cooperative — each agent stops at the next LLM
        // round boundary. The publish is what lets a live view show the freeze (the registry's
        // membership did not change).
        if changed {
            publish(RuntimeEvent::Registries);
        }
    }

    /// Clear the cooperative pause-stop flag for every agent in a workspace, so an agent that
    /// was mid-round when the pause fired does not bail "frozen" after the resume.
    pub fn clear_workspace_pause(&self, workspace_name: &str) {
        let mut changed = false;
        {
            let map = self.inner.lock().unwrap_poison();
            for entry in map.values() {
                if entry.handle.workspace_name == workspace_name {
                    changed |= entry.pause_stop.swap(false, Ordering::SeqCst);
                }
            }
        }
        if changed {
            publish(RuntimeEvent::Registries);
        }
    }

    /// Cancel all agents running for a specific role within a specific workspace path — used
    /// when maintenance is disabled for a workspace.
    pub fn cancel_by_role_and_workspace_path(&self, role: &str, ws_path: &str) {
        self.cancel_matching(|entry| {
            entry.handle.role == role && entry.handle.workspace_path == ws_path
        });
    }

    /// Cancel every agent belonging to a direct parent invocation (ticket / analyze round /
    /// research run) as a group — the GUI's research stop and the analyze tool's cancel action.
    ///
    /// Follows the lock-ordering invariant documented on
    /// [`cancel_matching`](AgentRegistry::cancel_matching).
    pub fn cancel_by_parent_key(&self, parent: &ParentKey) {
        let parent = parent.clone();
        self.cancel_matching(move |entry| entry.handle.parent_key.as_ref() == Some(&parent));
    }

    #[must_use]
    pub fn list(&self) -> Vec<AgentHandle> {
        self.inner
            .lock()
            .unwrap_poison()
            .values()
            .map(|e| {
                let mut handle = e.handle.clone();
                handle.activity.clone_from(&e.activity);
                handle.paused = e.pause_stop.load(Ordering::SeqCst);
                handle
            })
            .collect()
    }

    /// The worker or custom-role agent working in `folder` right now, other than `caller` —
    /// the roles that edit a folder's tree. Path equality is component-based, so a stored
    /// trailing separator is not a different folder.
    #[must_use]
    pub fn worker_in_folder(&self, folder: &Path, caller: Option<&str>) -> Option<AgentHandle> {
        self.list().into_iter().find(|agent| {
            let role = agent.role.as_str();
            (role == crate::Role::Worker.as_str() || role == crate::Role::Custom.as_str())
                && Some(agent.agent_id.as_str()) != caller
                && Path::new(&agent.workspace_path) == folder
        })
    }

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

    /// Remove a registry entry only if its generation still matches — [`crate::Agent::drop`]
    /// deregisters through this.
    pub fn deregister(&self, agent_id: &str, generation: u64) {
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(entry) = map.get(agent_id)
            && entry.generation == generation
        {
            map.remove(agent_id);
            publish(RuntimeEvent::Registries);
        }
    }
}

/// RAII guard: clears the agent's live activity label on drop. Creation and removal are both
/// generation-gated, so a stale guard can never clear the replacement agent's card.
pub(crate) struct ActivityGuard {
    agent_id: String,
    generation: u64,
}

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        AGENT_REGISTRY.activity_finished(&self.agent_id, self.generation);
    }
}

pub static AGENT_REGISTRY: LazyLock<AgentRegistry> = LazyLock::new(AgentRegistry::default);

/// Monotonically increasing entry id — each guard removes exactly its own entry on drop,
/// regardless of drop order.
static NEXT_ENTRY_ID: AtomicU64 = AtomicU64::new(1);

/// Public snapshot of one in-flight call — serializable, no internals exposed.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct NonAgentCallHandle {
    /// Call kind — the same purpose string the call's `ChatRequestMeta` uses (e.g.
    /// `"consolidate"`, `"synthesis"`, `"cli_command"`).
    pub kind: &'static str,
    pub workspace: String,
    pub started_at: DateTime<Utc>,
    /// Grouping key for the Running Agents view. `None` means the call is genuinely
    /// unattributable (workspace-scoped section with a distinct marker).
    pub parent_key: Option<ParentKey>,
    /// Human-readable label for the DIRECT PARENT INVOCATION this call belongs to, shown on
    /// the group header. Purely presentational.
    pub parent_label: Option<String>,
    /// True when this entry is a whole-operation lifetime guard (the research orchestrator
    /// holds one for the entire run): it renders inside its group as a run-lifetime indicator,
    /// not as a transient LLM-call card.
    pub run_lifetime: bool,
}

#[derive(Default)]
pub(crate) struct NonAgentCallRegistry {
    inner: Mutex<HashMap<u64, NonAgentCallHandle>>,
}

impl NonAgentCallRegistry {
    /// Register an in-flight non-agent LLM call and return the RAII guard that removes it on
    /// drop — cleanup is guaranteed on completion, failure and task cancellation.
    ///
    /// `parent` attaches the call to its DIRECT PARENT INVOCATION for the grouping (`None`
    /// renders in the unattributable section), `parent_label` is that invocation's label, and
    /// `run_lifetime` marks whole-operation guards (research orchestrator).
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
        publish(RuntimeEvent::Registries);
        NonAgentCallGuard { id, registry: self }
    }

    #[must_use]
    pub fn list(&self) -> Vec<NonAgentCallHandle> {
        self.inner
            .lock()
            .unwrap_poison()
            .values()
            .cloned()
            .collect()
    }

    /// Remove every in-flight call belonging to a direct parent invocation. Purely
    /// observational — the guards' drops become no-ops. Used by the manual-cancel paths so the
    /// group disappears immediately instead of lingering until its task returns.
    pub fn remove_by_parent_key(&self, parent: &ParentKey) {
        let parent = parent.clone();
        let mut map = self.inner.lock().unwrap_poison();
        let before = map.len();
        map.retain(|_, h| h.parent_key.as_ref() != Some(&parent));
        if map.len() != before {
            publish(RuntimeEvent::Registries);
        }
    }

    /// Remove a single in-flight call by id — the guard drop's sole removal point, so a guard
    /// whose entry already went with [`Self::remove_by_parent_key`] is a no-op.
    fn remove_one(&self, id: u64) {
        let removed = self.inner.lock().unwrap_poison().remove(&id).is_some();
        if removed {
            publish(RuntimeEvent::Registries);
        }
    }
}

#[must_use]
pub(crate) struct NonAgentCallGuard {
    id: u64,
    registry: &'static NonAgentCallRegistry,
}

impl Drop for NonAgentCallGuard {
    fn drop(&mut self) {
        self.registry.remove_one(self.id);
    }
}

pub(crate) static NON_AGENT_CALLS: LazyLock<NonAgentCallRegistry> =
    LazyLock::new(NonAgentCallRegistry::default);

/// Static human-readable label for a non-agent call kind — the single source of truth for
/// every place a raw kind string would otherwise surface.
///
/// The mapping is deliberately EXHAUSTIVE over every kind registered today; unknown kinds fall
/// back to `"Other LLM work"`, the same wording as the unattributed section, so a raw
/// snake_case name can never leak onto the page.
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
        // The provider transcription path is gone, but the rows it left are not: they keep their
        // label on the page.
        "media_transcription" => "Media transcription",
        "cli_command" => "Command line",
        _ => "Other LLM work",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Register a bare agent in the GLOBAL registry and return its generation counter. Each
    /// test uses a unique agent_id (via [`crate::generate_suffix`]).
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
            CancellationToken::new(),
            CancellationToken::new(),
            None,
            None,
            std::sync::Arc::new(AtomicBool::new(false)),
        )
    }

    #[test]
    fn stale_activity_guard_cannot_clear_replacement_agent() {
        let agent_id = format!("activity_stale_{}", crate::generate_suffix());
        let generation = register_test_agent(&agent_id, "/tmp/ws");
        let stale_guard = AGENT_REGISTRY.activity_started(&agent_id, generation, "extracting");
        AGENT_REGISTRY.deregister(&agent_id, generation);
        let new_gen = register_test_agent(&agent_id, "/tmp/ws");
        let fresh_guard = AGENT_REGISTRY.activity_started(&agent_id, new_gen, "transcribing");
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
    fn tool_args_are_full_and_structured() {
        let secret = format!("token-{}", crate::generate_suffix());
        let long_command = format!("echo {}", "a".repeat(1000));
        let args = serde_json::json!({
            "command": long_command,
            "api_key": secret,
        });
        let call = crate::ToolCall {
            id: "1".to_string(),
            name: "shell".to_string(),
            arguments: args,
        };
        let tool = RunningTool::from_tool_call(&call);
        assert!(
            tool.args
                .iter()
                .any(|(k, v)| k == "api_key" && v == &secret),
            "live view shows full values"
        );
        assert!(
            tool.args
                .iter()
                .any(|(k, v)| k == "command" && v == &long_command),
            "values are not truncated at flattening"
        );
    }

    #[test]
    fn cancel_by_parent_key_cancels_matching_agents_only() {
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
                CancellationToken::new(),
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
    fn interrupt_fires_the_turns_own_token_and_nothing_else() {
        let agent_id = format!("interrupt_token_{}", crate::generate_suffix());
        let ws = crate::Workspace {
            name: agent_id.clone(),
            path: "/tmp/ws_interrupt".to_string(),
            ..Default::default()
        };
        let cancel_token = CancellationToken::new();
        let interrupt_token = CancellationToken::new();
        let generation = AGENT_REGISTRY.register(
            agent_id.clone(),
            "assistant".to_string(),
            None,
            &ws,
            cancel_token.clone(),
            interrupt_token.clone(),
            None,
            None,
            std::sync::Arc::new(AtomicBool::new(false)),
        );

        AGENT_REGISTRY.interrupt(&agent_id);
        assert!(interrupt_token.is_cancelled(), "the running turn is cut");
        assert!(
            !cancel_token.is_cancelled(),
            "a cut is not a cancellation of the agent"
        );

        // A concluded or never-registered turn has nothing to cut, and saying so is not an error.
        AGENT_REGISTRY.deregister(&agent_id, generation);
        AGENT_REGISTRY.interrupt(&agent_id);
    }

    #[test]
    fn remove_by_parent_key_removes_matching_calls_only() {
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
        // The guards' drops are now no-ops — they must not panic and must not disturb
        // anything else.
        drop(run_a1);
        drop(run_a2);
        drop(run_b);
    }
}
