//! Workspace storage — persisted workspace metadata and contexts.
//!
//! Also handles workspace analysis: spawning a Discovery agent to explore a new
//! workspace and produce role-specific context summaries.

use crate::Role;
use crate::Workspace;
use crate::WorkspaceStatus;
use crate::agent::role::DIAGNOSTICS_ROLE;
use crate::agent::run_default_agent;
use crate::config_db::ConfigStore;
use crate::db::{self, Value};
use crate::outcome::Outcome;
use crate::util::UnwrapPoison;
use crate::workspace_profile::{ProfileRead, ProfileState};
use anyhow::{Context, Result};
use chrono::{DateTime, Timelike, Utc};
use futures_util::future::join_all;
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use strum::IntoEnumIterator;
use tracing::warn;

crate::define_store! {
    pub static WORKSPACES: WorkspaceStore,
    expect = "workspace::WORKSPACES not initialized — call init_all_stores() first",
}

pub async fn get_by_name(name: &str) -> Result<Option<Workspace>> {
    store().get_by_name(name).await
}

// `discovery_generation` and `diagnostics_generation` are deliberately excluded from
// this list: they are read through [`WorkspaceStore::get_generation`]'s single-column
// SELECT (plus `claim_pending_for_discovery`'s `UPDATE .. RETURNING`).
crate::columns! {
    WORKSPACE_COLUMNS [WS] {
        NAME                  => "name",
        PATH                  => "path",
        STATUS                => "status",
        MAINTENANCE_ENABLED    => "maintenance",
        PAUSED                => "paused",
        MAINTAINER_LAST_RUN_AT  => "maintainer_last_run_at",
        DIAGNOSTICS           => "diagnostics",
        NOTES                  => "notes",
        LAST_ANALYZED_COMMIT   => "last_analyzed_commit",
        PROFILE_CONFIRMED      => "profile_confirmed",
        PROFILE                => "profile",
    }
}

crate::columns! {
    EDITOR_TAB_COLUMNS [ET] {
        FILE_PATH    => "file_path",
        TAB_ORDER    => "tab_order",
        IS_ACTIVE    => "is_active",
        IS_DIRTY     => "is_dirty",
        DIRTY_CONTENT => "dirty_content",
    }
}

/// Generation counter metadata: DB column name and log label.
#[derive(Clone, Copy)]
struct GenerationColumn {
    name: &'static str,
    log_label: &'static str,
}

impl GenerationColumn {
    /// Bumped by [`WorkspaceStore::rediscover`].
    const DISCOVERY: Self = Self {
        name: "discovery_generation",
        log_label: "Discovery",
    };
    /// Bumped by [`WorkspaceStore::rediscover_diagnostics`] and [`WorkspaceStore::set_diagnostics`].
    const DIAGNOSTICS: Self = Self {
        name: "diagnostics_generation",
        log_label: "Diagnostics",
    };
}

/// Check the generation counter for `column`: return `true` if the calling
/// task is still the latest (OK to proceed), `false` if a newer run bumped
/// it while this task worked (stale — do not proceed).
async fn check_generation(
    storage: &WorkspaceStore,
    workspace_name: &str,
    generation: i64,
    column: GenerationColumn,
    label: &str,
) -> bool {
    let current_gen = storage
        .get_generation(workspace_name, column)
        .await
        .unwrap_or(generation + 1);
    if current_gen != generation {
        tracing::warn!(
            workspace_name,
            captured_gen = generation,
            current_gen,
            label = %label,
            "{} generation mismatch — skipping stale write",
            column.log_label
        );
        return false;
    }
    true
}

/// Run workspace discovery for a single role. `discovery_generation` is the counter
/// captured at spawn time: before writing, the current generation is re-read and a
/// mismatch means a newer [`WorkspaceStore::rediscover`] made this result stale, so the
/// write is skipped silently.
async fn run_workspace_discovery(
    ws: &Workspace,
    role: Role,
    discovery_generation: i64,
) -> Result<(), DiscoveryRunError> {
    run_discovery_task(
        ws,
        role.discovery_prompt(),
        discovery_generation,
        Some(role.as_str()),
    )
    .await
}

/// Run general (non-role) workspace discovery — a project overview used by
/// non-agent LLM calls. Stored as the NULL-role context row.
async fn run_general_discovery(
    ws: &Workspace,
    discovery_generation: i64,
) -> Result<(), DiscoveryRunError> {
    run_discovery_task(
        ws,
        crate::prompt::load_prompt("discovery/general.md"),
        discovery_generation,
        None,
    )
    .await
}

/// Shared discovery execution: run the Discovery agent on `prompt`, guard
/// against stale writes, then persist the result. `role` is `None` for the
/// general (non-role) context, `Some(name)` for a role-keyed context; the
/// role string also serves as the task label (`"general"` when `None`).
async fn run_discovery_task(
    ws: &Workspace,
    prompt: String,
    discovery_generation: i64,
    role: Option<&str>,
) -> Result<(), DiscoveryRunError> {
    let label = role.unwrap_or("general");
    let storage = WORKSPACES
        .get()
        .context("WORKSPACES not initialized")?
        .clone();

    tracing::info!(workspace_name = ws.name, role = %label, "Starting workspace discovery");

    let agent_id = crate::address::AgentAddress::discovery(&ws.name, label).id();
    let (agent, response) = run_default_agent(
        &agent_id,
        Role::Discovery,
        ws,
        &prompt,
        false,
        None,
        crate::work_unit::WorkUnit::Turn,
    )
    .await;
    let Some(response) = response else {
        return Err(discovery_no_response_error(&agent, "Discovery"));
    };

    let content = response.trim().to_string();
    if content.is_empty() {
        return Err(DiscoveryRunError::fatal(anyhow::anyhow!(
            "Empty response for '{label}'"
        )));
    }

    // Guard against stale writes: if another rediscover has been triggered
    // while this discovery ran, skip the context write.
    if !check_generation(
        &storage,
        &ws.name,
        discovery_generation,
        GenerationColumn::DISCOVERY,
        "context",
    )
    .await
    {
        return Ok(());
    }

    let result = match role {
        Some(r) => storage.set_context(&ws.name, r, &content).await,
        None => storage.set_general_context(&ws.name, &content).await,
    };
    if let Err(e) = result {
        tracing::error!(workspace_name = ws.name, role = %label, error = %e, "Failed to store context");
        return Err(DiscoveryRunError::fatal(e));
    }

    tracing::info!(workspace_name = ws.name, role = %label, "Workspace discovery for {label} completed");
    Ok(())
}

/// Run diagnostics discovery: a Discovery agent scans build files for the format, lint,
/// type-check, build and unit-test commands, and its scoped extraction yields the
/// structured result. `diagnostics_generation` guards against stale writes — a newer
/// [`WorkspaceStore::rediscover_diagnostics`] or [`WorkspaceStore::set_diagnostics`]
/// skips the write — and on failure the existing data is left untouched.
async fn run_workspace_diagnostics(
    ws: &Workspace,
    diagnostics_generation: i64,
) -> Result<(), DiscoveryRunError> {
    let storage = WORKSPACES
        .get()
        .context("WORKSPACES not initialized")?
        .clone();

    tracing::info!(workspace_name = ws.name, "Starting diagnostics discovery");

    let agent_id = crate::address::AgentAddress::discovery(&ws.name, DIAGNOSTICS_ROLE).id();

    // Load the diagnostics discovery prompt directly (not a role-specific discovery prompt).
    let prompt = crate::prompt::load_prompt("discovery/diagnostics.md");

    let (mut agent, response) = run_default_agent(
        &agent_id,
        Role::Discovery,
        ws,
        &prompt,
        false,
        None,
        crate::work_unit::WorkUnit::Turn,
    )
    .await;
    if response.is_none() {
        return Err(discovery_no_response_error(&agent, "Diagnostics discovery"));
    }

    // Keep the Agent alive after run_default_agent() for extract_verdict —
    // it needs agent.session.history() and agent.tool_specs.
    let extraction_prompt = crate::prompt::load_prompt("extraction/diagnostics.md");

    // Prefix-cache preservation: `agent.extract_verdict` uses the agent's own
    // parameters (model, reasoning_effort, tools, pinned provider) with no
    // response_format override — the extraction request shares the Discovery
    // agent-loop prefix, only the extraction prompt is new. Retry exhaustion
    // (RetryExhausted) surfaces the scoped loop's failure trail.
    let cmds: crate::DiagnosticsCommands = agent
        .extract_verdict(&extraction_prompt, None, None)
        .await
        .map_err(|e| DiscoveryRunError::classified(&agent, anyhow::Error::from(e)))?;

    // Guard against stale writes.
    if !check_generation(
        &storage,
        &ws.name,
        diagnostics_generation,
        GenerationColumn::DIAGNOSTICS,
        DIAGNOSTICS_ROLE,
    )
    .await
    {
        return Ok(());
    }

    storage.set_diagnostics(&ws.name, &cmds).await?;

    tracing::info!(
        workspace_name = ws.name,
        format = ?cmds.format,
        lint = ?cmds.lint,
        build = ?cmds.build,
        unit_test = ?cmds.unit_test,
        "Diagnostics discovery completed"
    );
    Ok(())
}

/// Apply the final status after a discovery run completes, once all role discoveries and
/// diagnostics have finished. Separate so tests can check the status invariants without running
/// real agents.
///
/// `Done` sets `ready`, and the poll step holds every claim while the workspace is `pending` or
/// `failed`, so queued work is picked up on the next cycle. `Failed` sets `failed` (terminal,
/// manual Re-analyze as before); `Deferred` sets `pending` and arms the pickup cooldown. None of
/// them touches `paused`: that column means the owner (or a stage death) stopped the workspace, so
/// a freeze that landed while discovery ran is left in force, and an unpause is the owner's own
/// action.
/// Panics never reach here — they abort the spawned task and boot reclassification recovers the
/// row (see [`spawn_panic_guarded`]). Every write is skipped when a newer
/// [`WorkspaceStore::rediscover`] bumped the generation while discovery was in flight.
async fn finalize_discovery(
    storage: &WorkspaceStore,
    ws_name: &str,
    ws_path: &str,
    discovery_generation: i64,
    outcome: Outcome,
    errors: &[String],
) {
    // Final guard: if a newer rediscover was triggered while this task ran,
    // all three write sites (contexts, diagnostics, status) have already been
    // individually guarded.  This check catches the status write.
    if !check_generation(
        storage,
        ws_name,
        discovery_generation,
        GenerationColumn::DISCOVERY,
        "final status",
    )
    .await
    {
        return;
    }

    match outcome {
        Outcome::Done => {
            // Success — any prior provider-failure cooldown is obsolete.
            clear_pending_pickup_cooldown(ws_name);

            // Capture the current git HEAD commit hash for nightly re-analysis detection.
            // If the git command fails (not a git repo, no commits, or other error),
            // store NULL — this is not an error for the discovery itself.
            let commit_hash = crate::git::commands::run_git_head(std::path::Path::new(ws_path))
                .await
                .ok();

            if let Err(e) = storage
                .exec_update_with_updated_at(
                    "status = ?, last_analyzed_commit = ?",
                    vec![
                        Value::from(WorkspaceStatus::Ready.to_string()),
                        Value::from(commit_hash.as_deref()),
                    ],
                    ws_name,
                )
                .await
            {
                tracing::warn!(
                    workspace = ws_name,
                    error = %e,
                    "Failed to update workspace status after discovery",
                );
            }

            tracing::info!(
                workspace_name = ws_name,
                "Workspace analysis complete — all roles ready"
            );
        }
        Outcome::Failed => {
            let msg = errors.join("; ");
            // Terminal — the workspace leaves the pending-pickup cycle; drop any
            // cooldown armed by an earlier provider-class failure.
            clear_pending_pickup_cooldown(ws_name);
            if let Err(e) = storage.set_status(ws_name, &WorkspaceStatus::Failed).await {
                // A failed write strands the workspace in `analyzing` until boot
                // recovery — log so it is not silent.
                tracing::warn!(
                    workspace = ws_name,
                    error = %e,
                    "Failed to mark workspace Failed after fatal discovery failure"
                );
            }
            tracing::warn!(workspace_name = ws_name, error = %msg, "Workspace analysis failed");
        }
        Outcome::Deferred => {
            // Every failure was provider-class: return to pending and let the pickup
            // step retry after the cooldown. The cooldown is armed BEFORE the status
            // write: while the row is still `analyzing` nothing can claim it, so no poll
            // round sees `pending` with no cooldown armed and spawns a duplicate run.
            record_pending_pickup_cooldown(ws_name);
            if let Err(e) = storage.set_status(ws_name, &WorkspaceStatus::Pending).await {
                // A failed write strands the workspace in `analyzing` with the
                // cooldown armed until boot recovery — log so it is not silent.
                tracing::warn!(
                    workspace = ws_name,
                    error = %e,
                    "Failed to return workspace to Pending after provider-class failure"
                );
            }
            tracing::warn!(
                workspace_name = ws_name,
                error = %errors.join("; "),
                "Workspace analysis stalled on provider failure — pending pickup retry after cooldown"
            );
        }
        // Discovery only ever produces the three values [`fold_discovery_result`] writes.
        Outcome::Running | Outcome::Empty | Outcome::Stopped | Outcome::Superseded => {}
    }
}

/// Two-bucket classification of a discovery failure. `Transient` is provider-class (auth
/// 401/403, quota, transport/5xx, shutdown/cancel): the workspace returns to Pending and
/// the pickup step retries it after an in-memory cooldown. `Fatal` is genuine (internal
/// runtime errors, parse failures): the workspace goes Failed, terminal. Panics never
/// reach this classifier — they strand the row in `analyzing` until boot reclassifies it
/// to `pending`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum DiscoveryFailureKind {
    Transient,
    Fatal,
}

/// A discovery failure carrying its two-bucket classification.
///
/// The classification rides the error so the parallel discovery tasks in
/// [`spawn_workspace_discovery`] can aggregate mixed outcomes with a
/// "any fatal → Failed" precedence without string matching.
#[derive(Debug)]
struct DiscoveryRunError {
    kind: DiscoveryFailureKind,
    error: anyhow::Error,
}

impl DiscoveryRunError {
    fn transient(error: anyhow::Error) -> Self {
        Self {
            kind: DiscoveryFailureKind::Transient,
            error,
        }
    }

    fn fatal(error: anyhow::Error) -> Self {
        Self {
            kind: DiscoveryFailureKind::Fatal,
            error,
        }
    }

    /// Wrap an error with the two-bucket classification for a failed
    /// discovery agent run (the shared wrap used by every agent-failure
    /// site in the discovery path).
    fn classified(agent: &crate::Agent, error: anyhow::Error) -> Self {
        match classify_discovery_failure(agent, Some(&error)) {
            DiscoveryFailureKind::Transient => Self::transient(error),
            DiscoveryFailureKind::Fatal => Self::fatal(error),
        }
    }
}

impl std::fmt::Display for DiscoveryRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)
    }
}

/// Default `?` conversion: an unclassified plain error (DB failure, missing global) is a
/// genuine runtime failure. Agent-run failures must go through
/// [`DiscoveryRunError::classified`] instead — the only path yielding `transient`.
impl From<anyhow::Error> for DiscoveryRunError {
    fn from(error: anyhow::Error) -> Self {
        Self::fatal(error)
    }
}

/// Classify a failed discovery run into the two-bucket scheme, mirroring
/// [`crate::agent::failure_classification`]'s ordering: drain/shutdown/cancellation are
/// transient (the run was cut short, not a genuine failure — a cancelled discovery
/// returns to Pending rather than the historical terminal Failed). A typed
/// [`crate::retry::RetryExhausted`] class decides provider-vs-genuine; everything else is
/// fatal. Panics never reach here (see [`spawn_panic_guarded`]).
///
/// `InvalidRequest`/`ContentRejection`/`Guardrail` are transient because the first covers auth
/// 401/403, invalid-model and tool-schema 400s — all config-fixable, so the workspace must wait in
/// Pending for the user rather than go terminal, and the escalating cooldown bounds the retry rate
/// — and the other two are refusals a repaired or shortened request can clear. `Parse`/validation
/// is fatal: the provider answered, so the defect is ours. `error` is the raw error when the call
/// site has one; the typed class `run_agent` captured takes precedence.
fn classify_discovery_failure(
    agent: &crate::Agent,
    error: Option<&anyhow::Error>,
) -> DiscoveryFailureKind {
    if agent.cut_short() {
        return DiscoveryFailureKind::Transient;
    }
    let class = agent
        .failure_class
        .or_else(|| error.and_then(crate::agent::failure_class_from_error));
    let Some(class) = class else {
        return DiscoveryFailureKind::Fatal;
    };
    match class {
        crate::retry::FailureClass::Transport
        | crate::retry::FailureClass::Unavailable
        | crate::retry::FailureClass::TruncatedEnvelope
        | crate::retry::FailureClass::NoResponse
        | crate::retry::FailureClass::TruncatedOutput
        | crate::retry::FailureClass::InvalidRequest
        | crate::retry::FailureClass::ContentRejection
        | crate::retry::FailureClass::Guardrail
        | crate::retry::FailureClass::Deadline
        | crate::retry::FailureClass::Shutdown => DiscoveryFailureKind::Transient,
        crate::retry::FailureClass::Parse
        | crate::retry::FailureClass::OutOfRangeScore
        | crate::retry::FailureClass::SchemaMismatch
        | crate::retry::FailureClass::Membership
        | crate::retry::FailureClass::Completeness
        | crate::retry::FailureClass::ContradictionAgents
        | crate::retry::FailureClass::ValidationOther => DiscoveryFailureKind::Fatal,
    }
}

/// Fold one discovery sub-run result into the aggregate [`Outcome`] shared by the role, general and
/// diagnostics sites: any error degrades [`Outcome::Done`], any fatal forces [`Outcome::Failed`],
/// and the error string is recorded (with `prefix`) for the finalize message. Only those three
/// values are reachable here.
fn fold_discovery_result(
    outcome: &mut Outcome,
    errors: &mut Vec<String>,
    result: Result<(), DiscoveryRunError>,
    prefix: &str,
) {
    if let Err(e) = result {
        errors.push(format!("{prefix}{e}"));
        if e.kind == DiscoveryFailureKind::Fatal {
            *outcome = Outcome::Failed;
        } else if *outcome == Outcome::Done {
            *outcome = Outcome::Deferred;
        }
    }
}

/// The classified error for an agent run that produced no response — shared by
/// [`run_discovery_task`] and [`run_workspace_diagnostics`], which differ only in the task label.
fn discovery_no_response_error(agent: &crate::Agent, task_label: &str) -> DiscoveryRunError {
    DiscoveryRunError::classified(
        agent,
        anyhow::anyhow!(
            "{task_label} agent returned no response: {}",
            agent.failure_reason("unknown error")
        ),
    )
}

/// Base cooldown for the first provider-class discovery failure (15 min).
const PENDING_PICKUP_COOLDOWN_BASE_MINS: u64 = 15;
/// Ceiling for the escalating pickup cooldown (4 h).
const PENDING_PICKUP_COOLDOWN_MAX_MINS: u64 = 240;

/// Escalating cooldown for the `attempts`-th consecutive provider-class failure: 15 min,
/// 30 min, 1 h, 2 h, then 4 h forever. Memory-only by design: after a restart a pending
/// workspace may be retried once immediately, and escalation keeps a persistently-down
/// provider from burning the full LLM retry budget every 15 minutes forever.
fn pending_pickup_cooldown_duration(attempts: u32) -> Duration {
    let exponent = attempts.saturating_sub(1).min(4);
    let minutes = (PENDING_PICKUP_COOLDOWN_BASE_MINS * 2u64.pow(exponent))
        .min(PENDING_PICKUP_COOLDOWN_MAX_MINS);
    Duration::from_mins(minutes)
}

#[derive(Debug, Clone, Copy)]
struct PickupCooldown {
    /// Instant before which the pickup step must not claim the workspace.
    deadline: Instant,
    /// Consecutive provider-class discovery failures since the last success or manual reset —
    /// drives the escalating cooldown duration.
    attempts: u32,
}

/// In-memory cooldown map: workspace name → state. Written by [`finalize_discovery`] on
/// provider-class failure; cleared on successful discovery, manual rediscover, delete and re-add.
static PENDING_PICKUP_COOLDOWNS: OnceLock<std::sync::Mutex<HashMap<String, PickupCooldown>>> =
    OnceLock::new();

fn pending_pickup_cooldowns() -> &'static std::sync::Mutex<HashMap<String, PickupCooldown>> {
    PENDING_PICKUP_COOLDOWNS.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Arm the escalating cooldown for a provider-class failure, so the pending pickup for
/// `ws_name` waits before claiming it again (see [`pending_pickup_cooldown_duration`]).
/// Success, manual rediscover, delete and re-add reset it. A transient finalize landing
/// after a concurrent delete's clear leaves a stale entry — harmless (the map is bounded
/// by workspace count) and cleared again on re-add.
fn record_pending_pickup_cooldown(ws_name: &str) {
    let mut map = pending_pickup_cooldowns().lock().unwrap_poison();
    let attempts = map.get(ws_name).map_or(0, |c| c.attempts) + 1;
    let deadline = Instant::now() + pending_pickup_cooldown_duration(attempts);
    map.insert(ws_name.to_string(), PickupCooldown { deadline, attempts });
}

/// Arm the cooldown with an explicit deadline (test hook; production callers use
/// [`record_pending_pickup_cooldown`]).
#[cfg(test)]
fn record_pending_pickup_cooldown_until(ws_name: &str, deadline: Instant) {
    let mut map = pending_pickup_cooldowns().lock().unwrap_poison();
    map.insert(
        ws_name.to_string(),
        PickupCooldown {
            deadline,
            attempts: 1,
        },
    );
}

/// Clear the cooldown for a workspace — also resets the escalation counter.
fn clear_pending_pickup_cooldown(ws_name: &str) {
    pending_pickup_cooldowns()
        .lock()
        .unwrap_poison()
        .remove(ws_name);
}

/// Whether the pickup step must wait out a cooldown before claiming `ws_name`. Expired
/// entries are deliberately retained (the map is bounded by workspace count): the
/// `attempts` escalation counter must survive expiry, or pruning here would reset the
/// escalation on every pickup cycle.
pub(crate) fn pending_pickup_cooldown_active(ws_name: &str) -> bool {
    let map = pending_pickup_cooldowns().lock().unwrap_poison();
    map.get(ws_name)
        .is_some_and(|c| Instant::now() < c.deadline)
}

/// Spawn `future` in a panic-guarded sub-task and await it, logging a workspace-level
/// error if it panics or is cancelled. Unlike the ticket-dispatch panic recovery, this
/// only logs: a non-prompt panic leaves the workspace in `analyzing`, which
/// `reclassify_analyzing_to_pending` recovers at the next startup.
async fn spawn_panic_guarded(
    ws_name: &str,
    task: &str,
    future: impl std::future::Future<Output = ()> + Send + 'static,
) {
    let inner = tokio::spawn(future);
    if let Err(e) = inner.await {
        let kind = if e.is_panic() { "panic" } else { "cancelled" };
        tracing::error!(
            workspace_name = %ws_name,
            kind = kind,
            error = %e,
            "{task} task failed",
        );
    }
}

/// Spawn a background task running per-role and general context discovery, and
/// diagnostics discovery unless `discover_diagnostics` is `false` (a re-analysis, or a
/// pickup that found existing diagnostics) — user-managed diagnostics survive.
/// `discovery_generation` is captured at spawn time and guards against stale writes.
pub fn spawn_workspace_discovery(
    ws: &Workspace,
    discovery_generation: i64,
    discover_diagnostics: bool,
) {
    let ws = ws.clone();
    tokio::spawn(async move {
        let ws_name = ws.name.clone();
        let inner_ws_name = ws_name.clone();
        let inner_ws_path = ws.path.clone();
        let inner = async move {
            // Build role discovery futures (always needed); the general (non-role) project
            // overview discovery runs alongside them.
            let role_futures: Vec<_> = Role::iter()
                .filter(|r| crate::agent::role::role_info(r).has_discovery)
                .map(|role| {
                    let ws = ws.clone();
                    async move { run_workspace_discovery(&ws, role, discovery_generation).await }
                })
                .collect();

            let (role_results, general_result, diagnostics_result) = if discover_diagnostics {
                // Read the diagnostics generation for the guard. Separate from
                // discovery_generation: both counters are independent (diagnostics is bumped by
                // set_diagnostics/rediscover_diagnostics, discovery by rediscover).
                let diag_gen = match WORKSPACES.get() {
                    Some(s) => s
                        .get_generation(&inner_ws_name, GenerationColumn::DIAGNOSTICS)
                        .await
                        .unwrap_or(0),
                    None => 0,
                };
                tokio::join!(
                    join_all(role_futures),
                    run_general_discovery(&ws, discovery_generation),
                    run_workspace_diagnostics(&ws, diag_gen),
                )
            } else {
                let (roles, general) = tokio::join!(
                    join_all(role_futures),
                    run_general_discovery(&ws, discovery_generation),
                );
                (roles, general, Ok(()))
            };

            let mut outcome = Outcome::Done;
            let mut errors: Vec<String> = Vec::new();

            for result in role_results {
                fold_discovery_result(&mut outcome, &mut errors, result, "");
            }

            // General/diagnostics failures fail the run; the failure kind decides whether the
            // workspace returns to Pending (provider-class) or goes Failed (genuine).
            fold_discovery_result(&mut outcome, &mut errors, general_result, "");
            fold_discovery_result(
                &mut outcome,
                &mut errors,
                diagnostics_result,
                "Diagnostics discovery failed: ",
            );

            let Some(storage) = WORKSPACES.get() else {
                tracing::error!("WORKSPACES not initialized during final status update");
                return;
            };

            finalize_discovery(
                storage,
                &inner_ws_name,
                &inner_ws_path,
                discovery_generation,
                outcome,
                &errors,
            )
            .await;
        };

        // Box the discovery payload: it embeds role/general/diagnostics futures and exceeds
        // the 16 KB large_futures threshold as an inline async-block.
        spawn_panic_guarded(&ws_name, "spawn_workspace_discovery", Box::pin(inner)).await;
    });
}

/// Spawn a background task running diagnostics discovery only — no per-role context
/// discovery. Used by the GUI's "Re-discover diagnostics"; `diagnostics_generation`
/// guards against stale writes via [`check_generation`].
fn spawn_diagnostics_discovery(ws: &Workspace, diagnostics_generation: i64) {
    let ws = ws.clone();
    tokio::spawn(async move {
        let ws_name = ws.name.clone();

        let inner = async move {
            if let Err(e) = run_workspace_diagnostics(&ws, diagnostics_generation).await {
                tracing::error!(
                    workspace_name = %ws.name,
                    error = %e,
                    "Diagnostics rediscovery failed",
                );
            }
        };

        spawn_panic_guarded(&ws_name, "spawn_diagnostics_discovery", inner).await;
    });
}

/// The longest workspace name [`validate_name`] admits.
pub(crate) const MAX_NAME_LEN: usize = 40;

/// Validate a workspace name: ASCII letters and underscores only, starting with a letter,
/// at most [`MAX_NAME_LEN`] characters.
fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("Workspace name must not be empty");
    }
    if name.len() > MAX_NAME_LEN {
        anyhow::bail!("Workspace name must not exceed {MAX_NAME_LEN} characters");
    }
    if !name.chars().all(|c| c.is_ascii_alphabetic() || c == '_') {
        anyhow::bail!("Workspace name must contain only ASCII letters (a-z, A-Z) and underscores");
    }
    if !name.starts_with(|c: char| c.is_ascii_alphabetic()) {
        anyhow::bail!("Workspace name must start with a letter");
    }
    Ok(())
}

/// Append the platform's separator to a managed workspace's root unless it
/// already ends with one, so the stored root reads the way it always has. A
/// drive or volume root already carries one, and that separator is what keeps it
/// absolute rather than drive-relative (`C:\`, not `C:`).
fn ensure_trailing_separator(mut path: String) -> String {
    if !path.ends_with(std::path::MAIN_SEPARATOR) {
        path.push(std::path::MAIN_SEPARATOR);
    }
    path
}

/// Canonicalize a user-provided path for workspace storage: expand `~`, resolve it
/// ([`std::fs::canonicalize`]), drop the platform's verbatim prefix
/// ([`crate::util::strip_verbatim_prefix`]) and append the trailing separator a managed
/// root is shown with. The error message is meant for the user.
fn canonicalize_workspace_path(raw: &str) -> Result<String, String> {
    let expanded = crate::util::expand_tilde(raw);

    let canonical = crate::util::with_block_in_place(|| {
        std::fs::canonicalize(&expanded).map_err(|e| {
            if expanded.exists() {
                format!("Cannot access path '{}': {e}", expanded.display())
            } else {
                format!("Path does not exist: {}", expanded.display())
            }
        })
    })?;

    if !canonical.is_dir() {
        return Err(format!("Path is not a directory: {}", canonical.display()));
    }

    Ok(ensure_trailing_separator(
        crate::util::strip_verbatim_prefix(&canonical)
            .to_string_lossy()
            .to_string(),
    ))
}

/// Synthesize a workspace from an absolute path to an existing directory, so a
/// custom-role launch can name *any* folder as its workspace. `None` unless `name` is an
/// absolute path that exists and is a directory. The name is the canonical path itself,
/// NOT the `dir-…` short key a delegation round uses: it is the durable key a launch is
/// recorded under, and [`crate::users::resolve_workspace`] accepts an absolute path, so
/// the address must be the folder itself for the launch to resolve after a restart. A
/// managed workspace with that name (or that folder — `WorkspaceStore::get_by_path`) is
/// resolved by the caller first and still wins.
#[must_use]
pub(crate) fn synthesize_from_path(name: &str) -> Option<Workspace> {
    let path = Path::new(name);
    if !path.is_absolute() {
        return None;
    }
    let (exists, is_dir) = crate::util::with_block_in_place(|| (path.exists(), path.is_dir()));
    if !exists || !is_dir {
        return None;
    }
    let canonical = crate::stored_workspace_path(path);
    let address = canonical.to_string_lossy().to_string();
    Some(Workspace::ephemeral_run(&address, &canonical))
}

fn workspace_from_row(row: &db::Row) -> anyhow::Result<Workspace> {
    Ok(Workspace {
        name: row.get(COL_WS_NAME)?,
        path: row.get(COL_WS_PATH)?,
        status: row
            .get::<String>(COL_WS_STATUS)?
            .parse::<WorkspaceStatus>()?,
        maintenance_enabled: row.get::<bool>(COL_WS_MAINTENANCE_ENABLED)?,
        paused: row.get::<bool>(COL_WS_PAUSED)?,
        maintainer_last_run_at: row.get::<Option<String>>(COL_WS_MAINTAINER_LAST_RUN_AT)?,
        diagnostics: row.get::<Option<String>>(COL_WS_DIAGNOSTICS)?,
        notes: row.get::<String>(COL_WS_NOTES)?,
        last_analyzed_commit: row.get::<Option<String>>(COL_WS_LAST_ANALYZED_COMMIT)?,
        profile_confirmed: row.get::<bool>(COL_WS_PROFILE_CONFIRMED)?,
        profile: row.get::<Option<String>>(COL_WS_PROFILE)?,
    })
}

/// The workspace's status as the owner is shown it in one listing line: an unconfirmed profile, a
/// confirmed profile this build cannot read, or another writer holding the folder. A workspace
/// whose own stage or maintainer holds the folder is working, not waiting, so that hold is not
/// reported — a second writer beside it still is. Reading never holds anything, so a pass over the
/// folder is never a wait. The invalid-profile report is a report, not a wait: it never stops the
/// pipeline, which is why the name says status rather than why.
#[must_use]
pub(crate) fn status_reason(ws: &Workspace, read: &ProfileRead) -> Option<String> {
    wait_reason(ws, read, false)
}

/// Whether the workspace is paused as it stands now — the live read every writer takes at the
/// moment it starts, so a pause landing between a decision and the work stops it. `false` for a
/// name with no managed row: a personal workspace cannot be paused. The read is retried once; a
/// pause that still cannot be read is an `Err`, which is neither a pause — nothing would ever lift
/// that — nor permission to write.
pub(crate) async fn paused_now(name: &str) -> anyhow::Result<bool> {
    let Some(store) = WORKSPACES.get() else {
        return Ok(false);
    };
    let mut attempt = 0;
    loop {
        attempt += 1;
        match store
            .conn
            .query_optional(
                "SELECT paused FROM workspaces WHERE name = ?1",
                db::params![name],
                |row| row.get::<i64>(0),
            )
            .await
        {
            Ok(paused) => return Ok(paused == Some(1)),
            Err(e) if attempt == 1 => {
                tracing::debug!(
                    workspace = name,
                    error = %e,
                    "Retrying the workspace's pause read",
                );
            }
            Err(e) => {
                return Err(e).with_context(|| format!("reading the pause of workspace '{name}'"));
            }
        }
    }
}

/// The pipeline's own gate: the same wait conditions, with the folder's hold counted even when the
/// holder is this workspace's own work — which is exactly when no further work may start — and
/// with an invalid profile reported nowhere, because it never stops the workspace.
#[must_use]
pub(crate) fn pipeline_gate(ws: &Workspace, read: &ProfileRead) -> Option<String> {
    wait_reason(ws, read, true)
}

/// Why the pipeline would not claim this workspace now, in report order: the profile first, then
/// the folder's hold. `gate` is the pipeline's own view, and it differs in exactly two arms — the
/// workspace's own stage and its maintainer count as writers there, because that is when no
/// further work may start, where the owner-facing listing calls that work IS the workspace working;
/// and an invalid profile is reported only to the owner, because it never stops the workspace.
fn wait_reason(ws: &Workspace, read: &ProfileRead, gate: bool) -> Option<String> {
    match read.state {
        // One short phrase, because a listing line carries it — the fuller status, including how to
        // confirm, is what `workspace profile` prints.
        ProfileState::Unconfirmed => {
            return Some("waiting for its profile to be confirmed".to_string());
        }
        // A report, not a wait: the workspace runs on the default, which the owner has to be told —
        // the parse error stays in `show` and `workspace profile`, a listing line only says what
        // state it is, and nothing is logged from a read.
        ProfileState::Invalid(_) if !gate => {
            return Some("its confirmed profile is invalid — running on the default".to_string());
        }
        ProfileState::Invalid(_) | ProfileState::Confirmed => {}
    }
    let agents = crate::agent::registry::AGENT_REGISTRY.list();
    let own_work = if gate {
        Vec::new()
    } else {
        own_work_groups(&agents, ws)
    };
    let excluded: Vec<&str> = own_work.iter().map(String::as_str).collect();
    let holder =
        crate::agent::registry::AgentRegistry::holder_among(agents, ws.as_path(), &excluded)?;
    Some(format!(
        "waiting for {} to finish",
        holder_name(&holder.role)
    ))
}

/// The workspace's own work holds its folder under its own group keys — excluded by GROUP, not by
/// holder: the workspace's own work is a group, whichever agent of it happens to hold the folder —
/// so the remaining holder is any writer outside those groups, not merely the first one that is
/// not.
fn own_work_groups(agents: &[crate::agent::registry::AgentHandle], ws: &Workspace) -> Vec<String> {
    agents
        .iter()
        .filter(|agent| Path::new(&agent.workspace_path) == ws.as_path())
        .filter(|agent| is_own_work(agent))
        .map(crate::agent::registry::AgentHandle::group)
        .collect()
}

/// Whether a run belongs to the workspace's own work: a ticket's phase round, or the workspace's
/// own maintainer, which registers as a workspace singleton ([`crate::work_unit::WorkUnit::Turn`],
/// no parent). A parentless run of any other role is another writer that happens to carry no parent
/// key — a router-dispatched worker round is the common case — so it is a holder and must be named.
/// A launch and a chain are separate writers too. The caller has already restricted the holder to
/// this workspace.
fn is_own_work(agent: &crate::agent::registry::AgentHandle) -> bool {
    match agent.parent_key.as_ref() {
        None => matches!(agent.role.parse::<Role>(), Ok(Role::Maintainer)),
        Some(crate::agent::registry::ParentKey::Ticket(_)) => true,
        Some(_) => false,
    }
}

/// What a paused workspace does to a writer: one clause, so the pipeline's gate, a launch and an
/// agent's own run — before it starts and when the freeze stops it — say the same thing. Reading
/// (a discovery pass, a separately launched analysis) is outside it: it runs in a paused workspace.
/// The caller names the workspace, so this must not name it again.
pub(crate) const PAUSED_REFUSAL: &str = "is paused — no writer starts here until it is resumed";

/// How a folder's holder reads in a sentence, by the role it runs as. The one mapping both voices
/// use — a launch refusal and a workspace's wait name the same writer the same way.
#[must_use]
pub(crate) fn holder_name(role: &str) -> String {
    match role.parse::<Role>() {
        Ok(Role::Worker) => "a worker".to_string(),
        Ok(Role::Custom) => "a custom role".to_string(),
        Ok(Role::Qa) => "the functional tester".to_string(),
        // Only a pipeline stage's analyst holds the folder — a separately launched analysis reads —
        // so the holder is named by the work, as the `send` reference names it.
        Ok(Role::Analyst) => "the pipeline's analysis stage".to_string(),
        Ok(Role::Reviewer) => "a code reviewer".to_string(),
        Ok(Role::Discovery) => "discovery".to_string(),
        Ok(Role::Maintainer) => "the maintainer".to_string(),
        Ok(Role::Engineer) => "the engineer".to_string(),
        Ok(Role::Sanitation) => "the sanitation step".to_string(),
        Ok(Role::Assistant) => "the assistant".to_string(),
        Ok(Role::Manager) => "the manager".to_string(),
        // A role string comes from `Role::to_string()`, so this is unreachable — but a holder is
        // never misnamed as nothing.
        Err(_) => format!("the {role}"),
    }
}

/// Max length of workspace notes in chars (char-level truncation — never
/// byte-slice, which would panic on multi-byte characters at the boundary).
pub(crate) const MAX_WORKSPACE_NOTES_CHARS: usize = 4000;

/// Single source of truth for the notes char-cap — plain char-boundary take
/// (no ellipsis, unlike `util::truncate`), shared by DB write, prompt build,
/// and GUI editor.
pub(crate) fn truncate_workspace_notes(s: &str) -> String {
    s.chars().take(MAX_WORKSPACE_NOTES_CHARS).collect()
}

impl WorkspaceStore {
    /// UPDATE `workspaces` setting `set_clause` plus `updated_at = now` for one named
    /// row. Returns the affected-row count, so a caller can refuse `0` (no such
    /// workspace) instead of reporting a write that never happened — see
    /// [`Self::set_paused`].
    async fn exec_update_with_updated_at(
        &self,
        set_clause: &str,
        set_params: Vec<db::Value>,
        name: &str,
    ) -> Result<u64> {
        let sql = format!("UPDATE workspaces SET {set_clause}, updated_at = ? WHERE name = ?");
        let mut params = set_params;
        params.push(Value::from(db::now()));
        params.push(Value::from(name));
        Ok(self.conn.execute(&sql, params).await?)
    }

    /// Insert a new workspace with status `pending`; the discovery spawn is deferred to the
    /// management poll's pickup step, so a first-run "workspace added before the key" no longer
    /// flips to Failed seconds later. The new row is not paused: its `pending` status — not a pause
    /// — is what keeps the ticket pipeline out until the first discovery runs, so a freeze set
    /// while it does stays in force.
    pub async fn add(&self, name: &str, path: &str) -> Result<Workspace> {
        validate_name(name)?;

        // Canonicalize and validate the path so bad paths never enter the system.
        let path = canonicalize_workspace_path(path).map_err(|e| anyhow::anyhow!("{e}"))?;
        // A folder or a name already managed is refused with the workspace that holds it: the
        // engine's own UNIQUE error names a column, not something a person can act on. What is
        // wrong and nothing about how to fix it: the same text reaches a tool caller, a terminal
        // caller and a desktop toast, and only one of those has an action to name.
        let managed = self.list().await?;
        if let Some(existing) = managed.iter().find(|ws| ws.path == path) {
            anyhow::bail!(
                "the folder {path} is already managed as workspace '{}'",
                existing.name
            );
        }
        if managed.iter().any(|ws| ws.name == name) {
            anyhow::bail!("a workspace named '{name}' already exists");
        }
        let now = db::now();
        let pending = WorkspaceStatus::Pending.to_string();
        let ws = self
            .conn
            .execute_one(
                &format!(
                    "INSERT INTO workspaces (name, path, status, created_at, updated_at, paused) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6) RETURNING {WORKSPACE_COLUMNS}"
                ),
                db::params![name, path, pending, now.clone(), now.clone(), 0],
                workspace_from_row,
            )
            .await?;
        // No discovery spawn here: the pickup step claims the pending
        // workspace once the provider is configured. discovery_generation
        // defaults to 0 in the schema — the pickup reads the live value.
        // Clear any stale cooldown from a previously-deleted same-name row.
        clear_pending_pickup_cooldown(name);
        // The folder becomes a repository so the pipeline can commit its work. An existing
        // repository is left alone, a failed init never fails the add, and the init runs after
        // the INSERT (a rejected add must not touch the folder) but before the index below, so
        // the engine is built with the repository already in place.
        crate::git::commands::init_repo_if_absent(std::path::Path::new(&ws.path)).await;
        if let Err(e) =
            crate::workspace_index::get_or_init_engine(name, std::path::Path::new(&ws.path))
        {
            tracing::warn!(workspace_name = name, error = %e, "Failed to init search engine on workspace add");
        }
        Ok(ws)
    }

    /// List all workspaces ordered by name.
    pub async fn list(&self) -> Result<Vec<Workspace>> {
        self.conn
            .query_map_strict_cached(
                &format!("SELECT {WORKSPACE_COLUMNS} FROM workspaces ORDER BY name"),
                db::params![],
                workspace_from_row,
            )
            .await
    }

    pub async fn get_by_name(&self, name: &str) -> Result<Option<Workspace>> {
        self.conn
            .query_optional_cached(
                &format!("SELECT {WORKSPACE_COLUMNS} FROM workspaces WHERE name = ?1"),
                db::params![name],
                workspace_from_row,
            )
            .await
    }

    /// The managed workspace whose folder is `path` — a caller may name a workspace by
    /// path instead of name ([`crate::users::resolve_workspace`]). Rows store the
    /// canonical path with a trailing separator, so the candidate is normalized the same way.
    pub async fn get_by_path(&self, path: &str) -> Result<Option<Workspace>> {
        let canonical = ensure_trailing_separator(
            crate::stored_workspace_path(std::path::Path::new(path))
                .to_string_lossy()
                .to_string(),
        );
        self.conn
            .query_optional_cached(
                &format!("SELECT {WORKSPACE_COLUMNS} FROM workspaces WHERE path = ?1"),
                db::params![canonical],
                workspace_from_row,
            )
            .await
    }

    /// Delete a workspace by name. Context rows are cascaded automatically. The
    /// admin's `users.selected_workspace` is resolved against the surviving
    /// projects: onto [`crate::users::fallback_workspace_name`]'s pick — the one
    /// start-up makes too — or onto the personal workspace (NULL) when none
    /// survives; an empty or dangling value is repaired the same way, since those
    /// are not steady states either (start-up repairs both too). Guest selections
    /// naming the deleted workspace are cleared: a guest always resolves to their
    /// personal workspace.
    ///
    /// The workspace's research runs are cancelled and swept first (releasing run
    /// folders + archives), and the remaining job rows and undelivered
    /// workspace-stop notices are deleted IN the workspace-row transaction so they
    /// commit atomically. The associated search engine is removed from the in-memory
    /// registry.
    ///
    /// Returns the admin's active project after the deletion (`None` = the
    /// personal workspace) — read back through the same transaction, so the
    /// caller reports where the desktop stands rather than a derivation of it.
    pub async fn delete(&self, name: &str) -> Result<Option<String>> {
        // Fail-closed: boot SKIPS jobs whose workspace is unresolvable (their
        // only deletion path is this abandon), so a failed abandon followed by
        // a deleted workspace row would orphan the jobs permanently. The
        // deletion aborts instead and the caller surfaces the error.
        let swept = crate::jobs::abandon_workspace_research_runs(name).await?;
        let tx = self.conn.begin_tx().await?;
        tx.execute("DELETE FROM workspaces WHERE name = ?1", db::params![name])
            .await?;
        // Every selection naming the deleted workspace is cleared here; the admin's is
        // resolved onto the fallback below.
        tx.execute(
            "UPDATE users SET selected_workspace = NULL WHERE selected_workspace = ?1",
            db::params![name],
        )
        .await?;
        // The projects left — read through the transaction, so the deletion above is visible.
        let surviving = tx
            .query_map_strict("SELECT name FROM workspaces", db::params![], |row| {
                row.get::<String>(0)
            })
            .await?;
        let next = crate::users::fallback_workspace_name(surviving.iter().map(String::as_str));
        tx.execute(
            "UPDATE users SET selected_workspace = ?1 WHERE name = ?2 AND \
             (selected_workspace IS NULL \
              OR selected_workspace NOT IN (SELECT name FROM workspaces))",
            db::params![next, crate::users::ADMIN_USER_NAME],
        )
        .await?;
        let active_now = tx
            .query_map_strict(
                "SELECT selected_workspace FROM users WHERE name = ?1",
                db::params![crate::users::ADMIN_USER_NAME],
                |row| row.get::<Option<String>>(0),
            )
            .await?
            .into_iter()
            .next()
            .flatten();
        // Remaining job rows go with the workspace row in ONE atomic commit; CASCADE removes
        // their rosters.
        let remaining = tx
            .execute(
                "DELETE FROM jobs WHERE workspace_name = ?1",
                db::params![name],
            )
            .await?;
        // Undelivered workspace-stop notices go with the workspace: a notice naming a
        // workspace that no longer exists can never be delivered.
        tx.execute(
            "DELETE FROM pause_notices WHERE workspace_name = ?1",
            db::params![name],
        )
        .await?;
        tx.commit().await?;
        if swept > 0 || remaining > 0 {
            tracing::info!(
                workspace_name = name,
                swept_runs = swept,
                remaining_jobs = remaining,
                "abandoned workspace jobs"
            );
        }
        crate::workspace_index::remove_engine(name);
        clear_pending_pickup_cooldown(name);
        Ok(active_now)
    }

    pub async fn set_status(&self, name: &str, status: &WorkspaceStatus) -> Result<()> {
        self.exec_update_with_updated_at("status = ?", vec![Value::from(status.to_string())], name)
            .await
            .map(|_| ())
    }

    /// Atomically claim a pending workspace for its first discovery: `pending` →
    /// `analyzing`, returning the row's live `discovery_generation` captured in the same
    /// statement so a concurrent `rediscover` cannot race the read. The claim does not pause, and a
    /// pause does not stop it: discovery reads the folder, so it is outside the freeze. `Ok(None)`
    /// when the row was no longer pending — a concurrent pickup, GUI Re-analyze or delete won the
    /// race, and the caller must not spawn.
    pub(crate) async fn claim_pending_for_discovery(&self, name: &str) -> Result<Option<i64>> {
        self.conn
            .execute_optional(
                "UPDATE workspaces SET status = ?, updated_at = ? \
                 WHERE name = ? AND status = ? RETURNING discovery_generation",
                db::params![
                    WorkspaceStatus::Analyzing.to_string(),
                    db::now(),
                    name,
                    WorkspaceStatus::Pending.to_string()
                ],
                |row| row.get(0),
            )
            .await
    }

    /// Boot recovery: discovery leaves no job rows, so a workspace still in `analyzing` at
    /// startup means a crashed mid-discovery run. Reclassify it to `pending` so the pickup step
    /// retries it. Returns the number reclassified.
    pub async fn reclassify_analyzing_to_pending(&self) -> Result<u64> {
        let affected = self
            .conn
            .execute(
                "UPDATE workspaces SET status = ?, updated_at = ? WHERE status = ?",
                db::params![
                    WorkspaceStatus::Pending.to_string(),
                    db::now(),
                    WorkspaceStatus::Analyzing.to_string()
                ],
            )
            .await?;
        if affected > 0 {
            tracing::info!(
                count = affected,
                "Boot recovery: reclassified stranded analyzing workspaces to pending"
            );
        }
        Ok(affected)
    }

    /// Set or clear the maintenance toggle for a workspace. A name with no row is an error
    /// rather than a silent no-op (see [`Self::set_paused`]).
    pub async fn set_maintenance_enabled(&self, name: &str, enabled: bool) -> Result<()> {
        let val: i64 = i64::from(enabled);
        // Enabling clears last_run_at so the maintainer runs on the very next poll cycle; a kept
        // interrupted session resumes regardless (resume bypasses the debounce gate).
        let set_clause = if enabled {
            "maintenance = ?, maintainer_last_run_at = NULL"
        } else {
            "maintenance = ?"
        };
        let affected = self
            .exec_update_with_updated_at(set_clause, vec![Value::from(val)], name)
            .await?;
        anyhow::ensure!(
            affected > 0,
            "No managed workspace exists under the name '{name}'."
        );
        if enabled {
            tracing::info!(workspace = name, "Maintainer enabled");
        } else {
            // Cancel any running maintainer agent for this workspace so it doesn't keep creating
            // tickets after maintenance was disabled.
            if let Some(ws) = self.get_by_name(name).await? {
                crate::agent::registry::AGENT_REGISTRY
                    .cancel_by_role_and_workspace_path(Role::Maintainer.as_str(), &ws.path);
            }
            tracing::info!(workspace = name, "Maintainer disabled");
        }
        Ok(())
    }

    /// Record that the owner accepted this workspace's profile: what is confirmed is the
    /// workspace's settings as they stand, not a snapshot of the values (an unpinned field keeps
    /// following the default).
    pub async fn confirm_profile(&self, name: &str) -> Result<u64> {
        let affected = self
            .exec_update_with_updated_at("profile_confirmed = 1", Vec::new(), name)
            .await?;
        anyhow::ensure!(
            affected > 0,
            "No managed workspace exists under the name '{name}'."
        );
        Ok(affected)
    }

    /// Store the fields the owner customised; `None` returns every field to the default.
    pub async fn set_profile(&self, name: &str, profile: Option<&str>) -> Result<u64> {
        let affected = self
            .exec_update_with_updated_at("profile = ?", vec![Value::from(profile)], name)
            .await?;
        anyhow::ensure!(
            affected > 0,
            "No managed workspace exists under the name '{name}'."
        );
        Ok(affected)
    }

    /// Set or clear the pipeline pause toggle for a workspace. A name with no row is an error
    /// rather than a silent no-op: every caller reports the toggle as applied, so a pause that
    /// wrote nothing would be reported as one.
    ///
    /// Pausing is [`Self::pause_with_record`] with nothing to record, so there is one pause
    /// write; resuming is the plain update below, followed by the two re-drives a resume owes: the
    /// turns the freeze kept and the rounds it stopped.
    pub async fn set_paused(&self, name: &str, paused: bool) -> Result<()> {
        if paused {
            let stopped = self.pause_with_record(name, async |_, _| Ok(())).await?;
            anyhow::ensure!(
                stopped.is_some(),
                "No managed workspace exists under the name '{name}'."
            );
            return Ok(());
        }
        let affected = self
            .exec_update_with_updated_at("paused = ?", vec![Value::from(0)], name)
            .await?;
        anyhow::ensure!(
            affected > 0,
            "No managed workspace exists under the name '{name}'."
        );
        // Clear the cooperative pause-stop flags left on in-flight agents so a resumed
        // workspace does not freeze them at their next LLM round boundary.
        crate::agent::registry::AGENT_REGISTRY.clear_workspace_pause(name);
        // The turns a pause kept are envelopes, not rounds: deliver them here, so resuming is all
        // the owner has to do to be answered — no restart.
        match crate::jobs::replay_pending_for_workspace(name).await {
            Ok(0) => {}
            Ok(delivered) => tracing::info!(
                workspace = name,
                delivered,
                "Delivered the turns a pause kept",
            ),
            Err(e) => tracing::warn!(
                workspace = name,
                error = %e,
                "Failed to deliver the turns a pause kept",
            ),
        }
        // The freeze stopped rounds mid-flight; resuming re-drives exactly those, so a resumed
        // workspace picks its work back up instead of waiting for the next restart.
        match crate::jobs::resumable_for_workspace(name).await {
            Ok(jobs) => crate::pipeline::resume_jobs(jobs).await,
            Err(e) => tracing::warn!(
                workspace = name,
                error = %e,
                "Failed to re-drive the rounds a resume should pick up",
            ),
        }
        tracing::info!(workspace = name, "Workspace pipeline resumed");
        Ok(())
    }

    /// The one pause write: pause the workspace and run `record` in the SAME transaction, so a
    /// pause is never committed without the write that explains it — a stage death's notice.
    /// `record` is handed whether this call performed the pause. Returns that same flag, or `None`
    /// when no such workspace exists (nothing was written).
    ///
    /// A workspace already paused by something else is not paused again, but the cooperative
    /// freeze is signalled for it too: its in-flight agents must stop at their next round
    /// boundary either way.
    pub(crate) async fn pause_with_record(
        &self,
        name: &str,
        record: impl AsyncFnOnce(&db::TxGuard<'_>, bool) -> Result<()>,
    ) -> Result<Option<bool>> {
        let tx = self.conn.begin_tx().await?;
        let paused = tx
            .query_map_strict(
                "SELECT paused FROM workspaces WHERE name = ?1",
                db::params![name.to_string()],
                |row| row.get::<i64>(0),
            )
            .await?
            .into_iter()
            .next();
        let Some(paused) = paused else {
            tx.rollback().await?;
            return Ok(None);
        };
        let stopped = paused == 0;
        if stopped {
            tx.execute(
                "UPDATE workspaces SET paused = 1, updated_at = ?1 WHERE name = ?2",
                db::params![db::now(), name.to_string()],
            )
            .await?;
        }
        record(&tx, stopped).await?;
        tx.commit().await?;
        crate::agent::registry::AGENT_REGISTRY.cancel_by_workspace_pause(name);
        if stopped {
            tracing::info!(workspace = name, "Workspace pipeline paused");
        }
        Ok(Some(stopped))
    }

    /// Pause every managed workspace — the `mahbot paused` startup mode. Each row goes
    /// through [`Self::set_paused`], so the pause is the ordinary one; rows already
    /// paused are left untouched. Returns the number of managed workspaces, all of them
    /// now paused. Personal workspaces have no row here.
    pub async fn pause_all(&self) -> Result<usize> {
        let workspaces = self.list().await?;
        for ws in &workspaces {
            if !ws.paused {
                self.set_paused(&ws.name, true).await?;
            }
        }
        Ok(workspaces.len())
    }

    /// Record a completed maintainer run by writing `maintainer_last_run_at`
    /// (plus `updated_at`). The debounce interval itself is the fixed
    /// [`Workspace::MAINTAINER_DEBOUNCE_MINS`] — the per-workspace adaptive
    /// value was retired.
    pub async fn set_maintainer_last_run_at(&self, name: &str, last_run_at: &str) -> Result<()> {
        self.exec_update_with_updated_at(
            "maintainer_last_run_at = ?",
            vec![Value::from(last_run_at)],
            name,
        )
        .await
        .map(|_| ())
    }

    /// Store the per-workspace maintainer recommendations blob (replace-only) —
    /// a JSON object {"recommendations": [...], "generated_at": RFC3339} produced
    /// by the post-run maintainer extraction. `None` clears it.
    pub(crate) async fn set_maintainer_recommendations(
        &self,
        name: &str,
        json: Option<&str>,
    ) -> Result<()> {
        self.exec_update_with_updated_at(
            "maintainer_recommendations = ?",
            vec![Value::from(json)],
            name,
        )
        .await
        .map(|_| ())
    }

    /// Read the raw maintainer recommendations JSON blob (None = absent/NULL).
    pub(crate) async fn get_maintainer_recommendations(
        &self,
        name: &str,
    ) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_optional(
                "SELECT maintainer_recommendations FROM workspaces WHERE name = ?1",
                db::params![name],
                |row| row.get::<Option<String>>(0),
            )
            .await?
            .flatten())
    }

    /// Store discovered diagnostics commands for a workspace.
    ///
    /// Also bumps `diagnostics_generation` so any in-flight diagnostics
    /// discovery task will see a generation mismatch and skip its stale write
    /// (see [`check_generation`]).
    pub(crate) async fn set_diagnostics(
        &self,
        name: &str,
        commands: &crate::DiagnosticsCommands,
    ) -> Result<()> {
        let json = serde_json::to_string(commands)?;
        self.exec_update_with_updated_at(
            "diagnostics = ?, diagnostics_generation = diagnostics_generation + 1",
            vec![Value::from(json)],
            name,
        )
        .await
        .map(|_| ())
    }

    pub(crate) async fn get_diagnostics(
        &self,
        name: &str,
    ) -> Result<Option<crate::DiagnosticsCommands>> {
        // query_optional + flatten preserves the NULL-vs-missing conflation:
        // a NULL diagnostics column and a missing row both yield None.
        let json: Option<String> = self
            .conn
            .query_optional(
                "SELECT diagnostics FROM workspaces WHERE name = ?1",
                db::params![name],
                |row| row.get::<Option<String>>(0),
            )
            .await?
            .flatten();
        Ok(json.map(|json| serde_json::from_str(&json)).transpose()?)
    }

    /// Set freeform user-curated context notes for a workspace.
    ///
    /// Truncates to `MAX_WORKSPACE_NOTES_CHARS` characters as defense-in-depth
    /// against prompt bloat. Notes are appended to every agent's system prompt.
    pub async fn set_notes(&self, name: &str, notes: &str) -> Result<()> {
        let notes = truncate_workspace_notes(notes);
        self.exec_update_with_updated_at("notes = ?", vec![Value::from(notes)], name)
            .await
            .map(|_| ())
    }

    /// Clear all workspace context rows (role-keyed and the general NULL-role
    /// row) for a workspace.
    /// Called by [`Self::rediscover`] before spawning a new discovery task so that
    /// stale context entries from a previous discovery don't persist.
    async fn clear_contexts(&self, name: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM workspace_contexts WHERE workspace_name = ?1",
                db::params![name],
            )
            .await?;
        Ok(())
    }

    /// Read the current generation counter for a workspace's `column`.
    ///
    /// Used by discovery tasks to check whether the counter was bumped while
    /// they ran; if it no longer matches, their writes are stale.
    async fn get_generation(&self, name: &str, column: GenerationColumn) -> Result<i64> {
        self.conn
            .query_row(
                &format!("SELECT {} FROM workspaces WHERE name = ?1", column.name),
                db::params![name],
                |row| row.get(0),
            )
            .await
            .map_err(Into::into)
    }

    /// Trigger re-analysis: status back to `analyzing`, stale per-role contexts cleared,
    /// and analysis spawned with a fresh generation counter. Unlike
    /// [`Self::rediscover_diagnostics`] this does **not** clear diagnostics — they are
    /// preserved untouched, and diagnostics discovery runs only when none exist yet (a
    /// never-analyzed `Pending` workspace re-analyzed manually would otherwise reach
    /// Ready with none). Manual Re-analyze bypasses the pending-pickup cooldown by
    /// design, so any in-memory cooldown is cleared here. The pause column is left alone, and
    /// nothing refuses a re-analysis: reading is outside both the freeze and the folder's hold, so
    /// a workspace the owner stopped, or one another writer is working in, is still re-analysed.
    pub async fn rediscover(&self, name: &str) -> Result<()> {
        let ws = self
            .get_by_name(name)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Workspace {name} not found"))?;

        // Atomically increment the discovery generation counter so any
        // still-running discovery task from a previous rediscover will
        // see a generation mismatch and skip its writes.
        // NOTE: diagnostics is deliberately NOT cleared — user-managed
        // diagnostics survive re-analysis.
        self.exec_update_with_updated_at(
            "discovery_generation = discovery_generation + 1, status = ?",
            vec![Value::from(WorkspaceStatus::Analyzing.to_string())],
            name,
        )
        .await?;

        clear_pending_pickup_cooldown(name);

        // Clear stale per-role context entries so that old discovery tasks
        // that beat the generation check cannot leave partial data behind.
        self.clear_contexts(name).await?;
        let generation = self
            .get_generation(name, GenerationColumn::DISCOVERY)
            .await?;
        // Skip diagnostics discovery when diagnostics already exist so
        // user-managed diagnostics survive re-analysis; discover them on a
        // never-analyzed workspace (the same rule the pending pickup uses).
        let discover_diagnostics = ws.diagnostics.is_none();
        spawn_workspace_discovery(&ws, generation, discover_diagnostics);

        Ok(())
    }

    /// Re-discover diagnostics commands for an existing workspace (without
    /// re-running per-role context discovery).
    ///
    /// Bumps the diagnostics generation (invalidating any in-flight diagnostics
    /// discovery tasks), clears the current diagnostics, and spawns a lightweight
    /// diagnostics-only discovery task.
    ///
    /// Unlike [`Self::rediscover`], this does **not** touch workspace status,
    /// paused state, per-role contexts, or the discovery generation counter. Nothing refuses it
    /// either: it is a read of the folder, so a freeze and another writer leave it alone.
    pub async fn rediscover_diagnostics(&self, name: &str) -> Result<()> {
        let ws = self
            .get_by_name(name)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Workspace {name} not found"))?;

        // Bump diagnostics_generation and clear diagnostics so the fresh
        // discovery starts from scratch. The generation bump also cancels
        // any in-flight diagnostics discovery tasks. Does NOT touch
        // discovery_generation — role discovery is unaffected.
        self.exec_update_with_updated_at(
            "diagnostics_generation = diagnostics_generation + 1, diagnostics = NULL",
            vec![],
            name,
        )
        .await?;

        let generation = self
            .get_generation(name, GenerationColumn::DIAGNOSTICS)
            .await?;
        spawn_diagnostics_discovery(&ws, generation);

        Ok(())
    }

    pub async fn get_context(&self, name: &str, role: &str) -> Result<Option<String>> {
        self.conn
            .query_optional(
                "SELECT content FROM workspace_contexts WHERE workspace_name = ?1 AND role = ?2",
                db::params![name, role],
                |row| row.get::<String>(0),
            )
            .await
    }

    async fn set_context(&self, name: &str, role: &str, content: &str) -> Result<()> {
        let now = db::now();
        self.conn
            .upsert_row(
                "UPDATE workspace_contexts SET content = ?1, created_at = ?2 \
                 WHERE workspace_name = ?3 AND role = ?4",
                || db::params![content, now.as_str(), name, role],
                "INSERT INTO workspace_contexts (workspace_name, role, content, created_at) VALUES (?1, ?2, ?3, ?4) \
                 ON CONFLICT(workspace_name, role) DO NOTHING",
                db::params![name, role, content, now.as_str()],
            )
            .await?;
        Ok(())
    }

    /// Get the general (non-role) workspace context, or `None` when discovery
    /// has not produced one yet.
    pub async fn get_general_context(&self, name: &str) -> Result<Option<String>> {
        self.conn
            .query_optional(
                "SELECT content FROM workspace_contexts WHERE workspace_name = ?1 AND role IS NULL",
                db::params![name],
                |row| row.get::<String>(0),
            )
            .await
    }

    /// Upsert the general (non-role) workspace context. The partial unique
    /// index guarantees at most one NULL-role row per workspace.
    async fn set_general_context(&self, name: &str, content: &str) -> Result<()> {
        let now = db::now();
        self.conn
            .upsert_row(
                "UPDATE workspace_contexts SET content = ?1, created_at = ?2 \
                 WHERE workspace_name = ?3 AND role IS NULL",
                || db::params![content, now.as_str(), name],
                "INSERT INTO workspace_contexts (workspace_name, role, content, created_at) VALUES (?1, NULL, ?2, ?3) \
                 ON CONFLICT(workspace_name) WHERE role IS NULL DO NOTHING",
                db::params![name, content, now.as_str()],
            )
            .await?;
        Ok(())
    }

    /// Save the current set of open editor tabs for a workspace. Replaces all existing records for
    /// this workspace.
    pub async fn save_editor_tabs(&self, name: &str, tabs: &[EditorTabRecord]) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        tx.execute(
            "DELETE FROM editor_tabs WHERE workspace_name = ?1",
            db::params![name],
        )
        .await?;
        for tab in tabs {
            tx.execute(
                "INSERT INTO editor_tabs (workspace_name, file_path, tab_order, is_active, is_dirty, dirty_content) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                db::params![
                    name,
                    tab.file_path.clone(),
                    i64::try_from(tab.tab_order).unwrap_or(i64::MAX),
                    i64::from(tab.is_active),
                    i64::from(tab.is_dirty),
                    tab.dirty_content.clone(),
                ],
            )
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn load_editor_tabs(&self, name: &str) -> Result<Vec<EditorTabRecord>> {
        let rows = self.conn
            .query_map(
                &format!("SELECT {EDITOR_TAB_COLUMNS} FROM editor_tabs WHERE workspace_name = ?1 ORDER BY tab_order"),
                db::params![name],
                |row| -> std::result::Result<EditorTabRecord, String> {
                    Ok(EditorTabRecord {
                        file_path: row
                            .get::<String>(COL_ET_FILE_PATH)
                            .map_err(|e| format!("failed to read file_path: {e}"))?,
                        tab_order: usize::try_from(
                            row.get::<i64>(COL_ET_TAB_ORDER)
                                .map_err(|e| format!("failed to read tab_order: {e}"))?,
                        )
                        .unwrap_or(0),
                        is_active: row
                            .get::<bool>(COL_ET_IS_ACTIVE)
                            .map_err(|e| format!("failed to read is_active: {e}"))?,
                        is_dirty: row
                            .get::<bool>(COL_ET_IS_DIRTY)
                            .map_err(|e| format!("failed to read is_dirty: {e}"))?,
                        dirty_content: row
                            .get::<Option<String>>(COL_ET_DIRTY_CONTENT)
                            .map_err(|e| format!("failed to read dirty_content: {e}"))?,
                    })
                },
            )
            .await?;
        let mut tabs = Vec::new();
        for row in rows {
            let tab = row.map_err(|e| anyhow::anyhow!("Failed to parse editor tab row: {e}"))?;
            if tab.file_path.is_empty() || tab.file_path.trim().is_empty() {
                // Defense-in-depth: the file_path column is NOT NULL in the
                // schema and DB errors now propagate before reaching this check,
                // but an empty string could still appear via corruption or other
                // code paths constructing EditorTabRecord. Skip rather than
                // resolve to workspace root.
                warn!(
                    workspace = %name,
                    tab_order = tab.tab_order,
                    "Skipping editor tab with empty file_path — would resolve to workspace root"
                );
                continue;
            }
            tabs.push(tab);
        }
        Ok(tabs)
    }
}

/// A single editor tab record for persistence.
#[derive(Debug, Clone)]
pub struct EditorTabRecord {
    pub file_path: String,
    pub tab_order: usize,
    pub is_active: bool,
    pub is_dirty: bool,
    /// Unsaved buffer text when `is_dirty` is true.
    pub dirty_content: Option<String>,
}

/// `config_kv` key holding the RFC 3339 UTC timestamp of the last nightly rediscovery
/// pass start. Deliberately not a `workspaces` column: rows are deleted during
/// rediscovery, so the timestamp must outlive workspace churn. It sits in the preserved
/// shared namespace, which reload does not purge.
pub(crate) const NIGHTLY_DISCOVERY_LAST_PASS_KV_KEY: &str = "nightly_discovery_last_pass_at";

/// Returns `true` when the given local hour falls within the nightly
/// re-analysis window (2:00–3:00 AM, inclusive of 2, exclusive of 3).
fn is_nightly_check_hour(local_hour: u32) -> bool {
    (2..3).contains(&local_hour)
}

/// Whether the rolling 7-day frequency gate allows a nightly pass: no pass recorded yet
/// (`None`), an unparseable timestamp (fail-open — the pass-start write self-heals the
/// stored state), or at least 7 × 24 h of wall-clock time elapsed since the recorded
/// pass start (DST-safe, no fixed weekday). A well-formed timestamp less than 7 days old
/// blocks the pass, including a future one (clock skew), so regeneration happens at most
/// once per 7 days.
fn nightly_gate_allows(last_pass_at: Option<&str>, now: DateTime<Utc>) -> bool {
    match last_pass_at {
        None => true,
        Some(raw) => match db::parse_utc_timestamp(raw) {
            Ok(last) => now.signed_duration_since(last) >= chrono::Duration::days(7),
            Err(e) => {
                warn!(
                    nightly_discovery_last_pass_at = %raw,
                    error = %e,
                    "Failed to parse nightly discovery last-pass timestamp, letting through"
                );
                true
            }
        },
    }
}

/// Evaluate the 7-day gate and, when the pass is allowed, record the pass start in
/// `config_kv` BEFORE any workspace processing: a pass that starts counts even if it
/// finds nothing or is interrupted. A read failure (missing store, read error,
/// unparseable value) fails open and the write self-heals the state; a pass-start write
/// failure fails closed, so the at-most-once-per-7-days invariant holds.
async fn nightly_gate_should_run(config_store: Option<&ConfigStore>) -> bool {
    // Fail-open on read.
    let last_pass_at = if let Some(store) = config_store {
        match store.get_kv(NIGHTLY_DISCOVERY_LAST_PASS_KV_KEY).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Nightly check: failed to read last-pass timestamp — running pass ungated"
                );
                None
            }
        }
    } else {
        tracing::warn!("Nightly check: CONFIG_STORE not initialized — running pass ungated");
        None
    };

    if !nightly_gate_allows(last_pass_at.as_deref(), Utc::now()) {
        tracing::debug!("Nightly check: last pass is less than 7 days old — skipping this night");
        return false;
    }

    // Record the pass start. Fail-closed: skip the pass when the timestamp
    // cannot be recorded.
    if let Some(store) = config_store {
        let started_at = db::now();
        if let Err(e) = store
            .set_kv(NIGHTLY_DISCOVERY_LAST_PASS_KV_KEY, &started_at)
            .await
        {
            tracing::warn!(
                error = %e,
                "Nightly check: failed to record pass start — skipping this pass"
            );
            return false;
        }
    }

    true
}

/// Returns `true` when rediscovery should be triggered because the
/// workspace has new git commits relative to its stored analysis state.
fn has_new_commits(last_analyzed_commit: Option<&str>, current_hash: &str) -> bool {
    match last_analyzed_commit {
        Some(stored) => stored != current_hash,
        None => true,
    }
}

/// Run the nightly re-analysis loop: wake every 30 minutes and, inside the 2:00–3:00 AM
/// local window, run one discovery pass — each Ready workspace whose HEAD differs from
/// the stored analysis commit is rediscovered. The pass is gated to at most one per 7
/// days (see [`nightly_gate_should_run`]); missed windows do not accumulate.
///
/// Workspaces are processed sequentially, so each discovery completes before the next is
/// checked. Edge cases: an empty pass still counts (the timestamp is written first, so the
/// next pass waits for the next eligible night — manual Re-analyze stays available); a
/// non-git workspace or one whose path is gone is skipped with a warning, and since
/// `last_analyzed_commit` stays NULL there is no re-discovery loop; a batch that starts
/// before 3:00 is finished, as the window is checked once per wake cycle.
pub async fn run_nightly_check_loop() {
    let interval = Duration::from_mins(30);
    let shutdown = crate::shutdown::shutdown_token();

    loop {
        if !crate::shutdown::sleep_or_shutdown_or_drain(interval).await {
            break;
        }

        if !is_nightly_check_hour(chrono::Local::now().hour()) {
            continue;
        }

        // Rolling frequency gate: at most one pass per 7 days, measured from
        // pass start. Fail-open on read (missing store / read error /
        // unparseable value), fail-closed on the pass-start write — see
        // [`nightly_gate_should_run`].
        if !nightly_gate_should_run(crate::config_db::CONFIG_STORE.get()).await {
            continue;
        }

        let store = if let Some(s) = WORKSPACES.get() {
            s.clone()
        } else {
            tracing::warn!("Nightly check: WORKSPACES not initialized");
            continue;
        };

        let workspaces = match store.list().await {
            Ok(list) => list,
            Err(e) => {
                tracing::warn!(error = %e, "Nightly check: failed to list workspaces");
                continue;
            }
        };

        // Process Ready workspaces sequentially to avoid LLM resource spikes.
        for ws in &workspaces {
            if shutdown.is_cancelled() {
                break;
            }

            // A paused workspace is re-analysed like any other: a discovery pass reads the folder,
            // so the pause neither refuses it nor is lifted by it.
            if ws.status != WorkspaceStatus::Ready {
                continue;
            }

            let repo_path = std::path::Path::new(&ws.path);

            let current_hash = match crate::git::commands::run_git_head(repo_path).await {
                Ok(hash) => hash,
                Err(e) => {
                    // Non-git workspace, no commits, or git not available — skip.
                    tracing::debug!(
                        workspace = %ws.name,
                        error = %e,
                        "Nightly check: git rev-parse HEAD failed — skipping workspace",
                    );
                    continue;
                }
            };

            let should_rediscover =
                has_new_commits(ws.last_analyzed_commit.as_deref(), &current_hash);

            if !should_rediscover {
                tracing::debug!(
                    workspace = %ws.name,
                    "Nightly check: no new commits — skipping",
                );
                continue;
            }

            tracing::info!(
                workspace = %ws.name,
                "Nightly check: new commits detected — triggering rediscover",
            );

            if let Err(e) = store.rediscover(&ws.name).await {
                tracing::warn!(
                    workspace = %ws.name,
                    error = %e,
                    "Nightly check: rediscover failed",
                );
                continue;
            }

            await_discovery(&store, &ws.name, &shutdown).await;
        }
    }
}

/// Wait for a discovery pass to finish before the nightly loop moves on: status polled every 10
/// seconds, with a generous 4-hour deadline.
async fn await_discovery(
    store: &WorkspaceStore,
    name: &str,
    shutdown: &tokio_util::sync::CancellationToken,
) {
    let deadline = std::time::Instant::now() + Duration::from_hours(4);
    loop {
        if shutdown.is_cancelled() {
            return;
        }
        if std::time::Instant::now() >= deadline {
            tracing::warn!(
                workspace = %name,
                "Nightly check: discovery timed out — proceeding to next workspace",
            );
            return;
        }
        tokio::time::sleep(Duration::from_secs(10)).await;
        match store.get_by_name(name).await {
            Ok(Some(current)) if current.status != WorkspaceStatus::Analyzing => return,
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(
                    workspace = %name,
                    error = %e,
                    "Nightly check: failed to poll workspace status",
                );
                return;
            }
        }
    }
}

/// Create a minimal [`Workspace`] from a path for testing.
/// The name is derived from the path's file name.
#[cfg(test)]
#[must_use]
pub fn test_ws(path: impl AsRef<std::path::Path>) -> Workspace {
    Workspace::from_path(path.as_ref())
}

/// Create a minimal [`Workspace`] with an explicit path and name.
#[cfg(test)]
#[must_use]
pub fn test_ws_named(path: &str, name: &str) -> Workspace {
    Workspace {
        name: name.to_string(),
        path: path.to_string(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Open a temporary workspace store for testing.
    /// Returns (store, temp_dir). The temp_dir is kept alive for the lifetime
    /// of the store (~ the test function).
    async fn test_store() -> (WorkspaceStore, TempDir) {
        crate::open_test_store!(WorkspaceStore, "workspace")
    }

    /// Helper: insert a workspace row directly with full control over fields,
    /// bypassing `add()` (which has side-effects like initializing search
    /// engine globals).
    async fn insert_direct(
        store: &WorkspaceStore,
        name: &str,
        path: &str,
        paused: bool,
        maintenance_enabled: bool,
        discovery_generation: i64,
        diagnostics_generation: i64,
    ) -> Workspace {
        let now = crate::db::now();
        let paused_int: i64 = i64::from(paused);
        let maint_int: i64 = i64::from(maintenance_enabled);
        store
            .conn
            .execute(
                "INSERT INTO workspaces (name, path, created_at, updated_at, paused, maintenance, discovery_generation, diagnostics_generation, profile_confirmed) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1)",
                crate::db::params![name, path, now.clone(), now.clone(), paused_int, maint_int, discovery_generation, diagnostics_generation],
            )
            .await
            .expect("insert workspace");
        Workspace {
            name: name.to_string(),
            path: path.to_string(),
            status: WorkspaceStatus::Pending,
            maintenance_enabled,
            paused,
            maintainer_last_run_at: None,
            diagnostics: None,
            notes: String::new(),
            last_analyzed_commit: None,
            profile_confirmed: true,
            profile: None,
        }
    }

    #[tokio::test]
    async fn schema_default_is_paused() {
        // Insert WITHOUT specifying paused, relying on the schema DEFAULT.
        let (store, _tmp) = test_store().await;
        let now = crate::db::now();
        store
            .conn
            .execute(
                "INSERT INTO workspaces (name, path, created_at, updated_at) \
                 VALUES (?1, ?2, ?3, ?4)",
                crate::db::params!["schema_test", "/tmp/schema_test", now.clone(), now.clone()],
            )
            .await
            .expect("insert workspace");

        let ws = store
            .get_by_name("schema_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(
            ws.paused,
            "Schema DEFAULT should produce paused = true for new rows"
        );
    }

    #[tokio::test]
    async fn set_paused_toggles_pause_state() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "toggle_test", "/tmp/toggle_test", true, false, 0, 0).await;

        store.set_paused("toggle_test", false).await.unwrap();
        let fetched = store
            .get_by_name("toggle_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(
            !fetched.paused,
            "Should be unpaused after set_paused(false)"
        );

        store.set_paused("toggle_test", true).await.unwrap();
        let fetched = store
            .get_by_name("toggle_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(fetched.paused, "Should be paused after set_paused(true)");

        // A name with no row is refused: the toggle is reported as applied by
        // every caller, so it must not silently do nothing.
        let err = store
            .set_paused("toggle_test_absent", true)
            .await
            .expect_err("an unknown workspace must be refused");
        assert!(
            err.to_string().contains("No managed workspace exists"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn pause_all_pauses_every_managed_workspace() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "all_ready", "/tmp/all_ready", false, false, 0, 0).await;
        insert_direct(&store, "all_paused", "/tmp/all_paused", true, false, 0, 0).await;

        assert_eq!(store.pause_all().await.unwrap(), 2);
        for name in ["all_ready", "all_paused"] {
            let ws = store.get_by_name(name).await.unwrap().expect("exists");
            assert!(ws.paused, "{name} must be paused");
        }
    }

    #[tokio::test]
    async fn set_maintenance_toggles_maintenance_state() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "maint_test", "/tmp/maint_test", true, false, 0, 0).await;

        store
            .set_maintenance_enabled("maint_test", true)
            .await
            .unwrap();
        let fetched = store
            .get_by_name("maint_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(
            fetched.maintenance_enabled,
            "Should have maintenance enabled after set_maintenance_enabled(true)"
        );

        store
            .set_maintenance_enabled("maint_test", false)
            .await
            .unwrap();
        let fetched = store
            .get_by_name("maint_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(
            !fetched.maintenance_enabled,
            "Should have maintenance disabled after set_maintenance_enabled(false)"
        );

        // A name with no row is refused rather than reported as a write that
        // happened — the footer toggle must not claim success for nothing.
        assert!(
            store
                .set_maintenance_enabled("no_such_ws", true)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn set_notes_roundtrip() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "notes_test", "/tmp/notes_test", true, false, 0, 0).await;

        // Initial state should be empty string (NOT NULL DEFAULT '')
        let ws = store
            .get_by_name("notes_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(ws.notes.is_empty(), "New workspace should have empty notes");

        let test_notes = "These are important context notes for agents.";
        store
            .set_notes("notes_test", test_notes)
            .await
            .expect("set_notes");
        let ws = store
            .get_by_name("notes_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert_eq!(ws.notes, test_notes, "Notes should round-trip correctly");

        let updated_notes = "Updated notes with more context.";
        store
            .set_notes("notes_test", updated_notes)
            .await
            .expect("set_notes");
        let ws = store
            .get_by_name("notes_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert_eq!(
            ws.notes, updated_notes,
            "Notes update should round-trip correctly"
        );

        let long_notes = "x".repeat(MAX_WORKSPACE_NOTES_CHARS + 1000);
        store
            .set_notes("notes_test", &long_notes)
            .await
            .expect("set_notes");
        let ws = store
            .get_by_name("notes_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert_eq!(
            ws.notes.chars().count(),
            MAX_WORKSPACE_NOTES_CHARS,
            "Notes should be truncated to {MAX_WORKSPACE_NOTES_CHARS} chars"
        );
        assert_eq!(
            ws.notes,
            "x".repeat(MAX_WORKSPACE_NOTES_CHARS),
            "Notes content should match truncated"
        );

        let multi_byte = "é".repeat(MAX_WORKSPACE_NOTES_CHARS + 1000);
        store
            .set_notes("notes_test", &multi_byte)
            .await
            .expect("set_notes");
        let ws = store
            .get_by_name("notes_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert_eq!(
            ws.notes.chars().count(),
            MAX_WORKSPACE_NOTES_CHARS,
            "Notes should be truncated to {MAX_WORKSPACE_NOTES_CHARS} chars (multi-byte)"
        );
        assert_eq!(
            ws.notes,
            "é".repeat(MAX_WORKSPACE_NOTES_CHARS),
            "Notes content should match truncated (multi-byte, no broken chars)"
        );
    }

    #[tokio::test]
    /// A finished discovery sets the status only: a pause that landed while the analysis ran is
    /// the owner's (or a stage death's) and stays in force, so a freeze survives a re-analysis.
    async fn a_finished_discovery_leaves_a_pause_in_force() {
        for (suffix, generation) in [("gen0", 0), ("gen1", 1)] {
            let (store, _tmp) = test_store().await;
            insert_direct(
                &store,
                suffix,
                &format!("/tmp/{suffix}"),
                true,
                false,
                generation,
                generation,
            )
            .await;
            finalize_discovery(
                &store,
                suffix,
                &format!("/tmp/{suffix}"),
                generation,
                Outcome::Done,
                &[],
            )
            .await;

            let ws = store
                .get_by_name(suffix)
                .await
                .expect("fetch")
                .expect("exists");
            assert!(
                ws.paused,
                "a pause set before the round finished must survive it (gen {generation})"
            );
            assert_eq!(
                ws.status,
                WorkspaceStatus::Ready,
                "Status should be 'ready'"
            );
        }
    }

    #[tokio::test]
    async fn finalize_discovery_failure_keeps_paused() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "fail_gen0", "/tmp/fail_gen0", true, false, 0, 0).await;

        // Act: discovery failed with a genuine (non-provider) failure
        // (outcome = Fatal) — must be terminal Failed as before.
        let errors = vec!["Empty response for 'general'".to_string()];
        finalize_discovery(
            &store,
            "fail_gen0",
            "/tmp/fail_gen0",
            0,
            Outcome::Failed,
            &errors,
        )
        .await;

        let ws = store
            .get_by_name("fail_gen0")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(ws.paused, "Should remain paused after discovery failure");
        assert_eq!(
            ws.status,
            WorkspaceStatus::Failed,
            "Status should be 'failed'"
        );
    }

    #[tokio::test]
    async fn finalize_discovery_provider_failure_returns_to_pending() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "prov_gen0", "/tmp/prov_gen0", true, false, 0, 0).await;

        // Act: discovery failed with provider-class failures only
        // (outcome = Transient) — must return to Pending and arm the cooldown.
        let errors = vec![
            "Diagnostics discovery failed: exhausted retry budget (last: transport): connection reset"
                .to_string(),
        ];
        finalize_discovery(
            &store,
            "prov_gen0",
            "/tmp/prov_gen0",
            0,
            Outcome::Deferred,
            &errors,
        )
        .await;

        let ws = store
            .get_by_name("prov_gen0")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(
            ws.paused,
            "a pause is left in force by every outcome, this one included"
        );
        assert_eq!(
            ws.status,
            WorkspaceStatus::Pending,
            "Provider-class discovery failure should return the workspace to pending"
        );
        assert!(
            pending_pickup_cooldown_active("prov_gen0"),
            "Provider-class failure must arm the in-memory pickup cooldown"
        );
        // A successful finalize clears the cooldown.
        finalize_discovery(&store, "prov_gen0", "/tmp/prov_gen0", 0, Outcome::Done, &[]).await;
        assert!(
            !pending_pickup_cooldown_active("prov_gen0"),
            "Successful discovery must clear the pickup cooldown"
        );
    }

    #[tokio::test]
    async fn finalize_discovery_stale_generation_skips_writes() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "stale", "/tmp/stale", true, false, 0, 0).await;

        // Bump the generation behind the scenes (simulates a concurrent
        // rediscover() call).
        store
            .exec_update_with_updated_at("discovery_generation = 1", vec![], "stale")
            .await
            .expect("bump generation");

        finalize_discovery(&store, "stale", "/tmp/stale", 0, Outcome::Done, &[]).await;

        let ws = store
            .get_by_name("stale")
            .await
            .expect("fetch")
            .expect("exists");
        // The writes should have been skipped because the generation
        // no longer matches.
        assert!(
            ws.paused,
            "Should stay paused — writes skipped by generation guard"
        );
        assert_eq!(
            ws.status,
            WorkspaceStatus::Pending,
            "Status should remain unchanged — writes skipped"
        );
    }

    /// A re-analysis changes the status only: the pause column is the owner's, so a workspace that
    /// was running stays unpaused and one the owner stopped stays stopped.
    #[tokio::test]
    async fn a_reanalysis_leaves_the_pause_state_alone() {
        let (store, _tmp) = test_store().await;
        insert_direct(
            &store,
            "rediscover_test",
            "/tmp/rediscover_test",
            false,
            false,
            0,
            0,
        )
        .await;
        store
            .set_status("rediscover_test", &WorkspaceStatus::Ready)
            .await
            .unwrap();

        let ws = store
            .get_by_name("rediscover_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(!ws.paused, "Precondition: workspace should start unpaused");
        assert_eq!(
            ws.status,
            WorkspaceStatus::Ready,
            "Precondition: status should be 'ready'"
        );

        store
            .rediscover("rediscover_test")
            .await
            .expect("rediscover");

        let ws = store
            .get_by_name("rediscover_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(
            !ws.paused,
            "rediscover() must not pause a workspace that was running"
        );
        assert_eq!(
            ws.status,
            WorkspaceStatus::Analyzing,
            "rediscover() still takes the workspace into 'analyzing'"
        );
    }

    #[tokio::test]
    #[serial_test::serial(config_persist)] // swaps the process-global CONFIG
    async fn add_persists_and_returns_the_workspace_row() {
        let (store, _tmp) = test_store().await;
        let dir = TempDir::new().expect("temp dir for workspace path");

        // add() requires: search engine globals initialized + CONFIG storage
        // root set. Initialize the minimum globals (init_global is idempotent).
        crate::workspace_index::init_global();
        // Point the storage root at the shared test root. A per-test TempDir
        // would leave the global storage root pointing at a deleted directory
        // once the TempDir drops — the shared root lives for the whole
        // process and is removed at process exit.
        let _ = crate::config::CONFIG.try_set_storage_root(crate::util::test::test_root().clone());
        crate::config::CONFIG.swap(crate::config::ConfigData::STRUCT_FIELDS_DEFAULT);

        let ws = store
            .add("add_test", dir.path().to_str().unwrap())
            .await
            .expect("add workspace");

        assert!(
            !ws.paused,
            "add() must return an unpaused Workspace: its status, not a pause, keeps the \
             pipeline out until discovery completes"
        );
        assert_eq!(
            ws.status,
            WorkspaceStatus::Pending,
            "add() must return a Workspace with status = pending — \
             discovery is deferred to the pickup step until the provider is configured"
        );
        // Pins schema defaults (add() reads them back via RETURNING).
        assert!(
            !ws.maintenance_enabled,
            "add() must return a Workspace with maintenance_enabled = false"
        );

        // Judged through the OS, not against the constructor that produced it. Only the
        // no-verbatim check needs a Windows host to fail: a verbatim root still resolves,
        // separator and all.
        assert_eq!(
            std::fs::canonicalize(&ws.path).expect("the platform must resolve the stored root"),
            std::fs::canonicalize(dir.path()).expect("canonicalize the managed dir"),
            "the stored root must resolve to the managed directory"
        );
        assert!(
            !ws.path.contains(r"\\?\"),
            "no consumer may receive the platform's verbatim form: {:?}",
            ws.path
        );
        assert!(
            ws.path.ends_with(std::path::MAIN_SEPARATOR),
            "a managed root keeps the trailing separator it has always been \
             shown with: {:?}",
            ws.path
        );

        let fetched = store
            .get_by_name("add_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(
            !fetched.paused,
            "Persisted workspace must not be born paused"
        );
        assert_eq!(
            fetched.status,
            WorkspaceStatus::Pending,
            "Persisted workspace must have status = pending"
        );
    }

    /// A folder or a name that is already managed is refused with the workspace that holds it,
    /// not with the engine's own UNIQUE error — the admin has to know which one it collides with.
    #[tokio::test]
    #[serial_test::serial(config_persist)] // swaps the process-global CONFIG
    async fn add_refuses_a_folder_or_name_that_is_already_managed() {
        let (store, _tmp) = test_store().await;
        let first = TempDir::new().expect("temp dir for the managed folder");
        let second = TempDir::new().expect("temp dir for the second folder");
        crate::workspace_index::init_global();
        let _ = crate::config::CONFIG.try_set_storage_root(crate::util::test::test_root().clone());
        crate::config::CONFIG.swap(crate::config::ConfigData::STRUCT_FIELDS_DEFAULT);

        store
            .add("dup_test", first.path().to_str().unwrap())
            .await
            .expect("the first add lands");

        let same_folder = store
            .add("dup_other", first.path().to_str().unwrap())
            .await
            .expect_err("the folder is already managed");
        assert!(
            same_folder
                .to_string()
                .contains("already managed as workspace 'dup_test'"),
            "got: {same_folder}"
        );
        let same_name = store
            .add("dup_test", second.path().to_str().unwrap())
            .await
            .expect_err("the name is already taken");
        assert!(
            same_name
                .to_string()
                .contains("a workspace named 'dup_test' already exists"),
            "got: {same_name}"
        );
        assert_eq!(
            store.list().await.expect("list").len(),
            1,
            "a refused add leaves no row behind"
        );
    }

    /// Adding a folder makes it a repository with nothing committed in it, and adding a folder
    /// that already is a repository leaves that repository exactly as it was.
    #[tokio::test]
    #[serial_test::serial(config_persist)] // swaps the process-global CONFIG
    async fn add_initialises_a_repository_only_when_the_folder_has_none() {
        let (store, _tmp) = test_store().await;
        crate::workspace_index::init_global();
        let _ = crate::config::CONFIG.try_set_storage_root(crate::util::test::test_root().clone());
        crate::config::CONFIG.swap(crate::config::ConfigData::STRUCT_FIELDS_DEFAULT);

        // A folder without a repository: `git init` writes `.git` and commits nothing.
        let fresh = TempDir::new().expect("temp dir for workspace path");
        store
            .add("add_git_fresh", fresh.path().to_str().unwrap())
            .await
            .expect("add workspace");
        let entries: Vec<_> = std::fs::read_dir(fresh.path())
            .expect("read the added folder")
            .map(|entry| entry.expect("entry").file_name())
            .collect();
        assert_eq!(
            entries,
            [std::ffi::OsString::from(".git")],
            "`git init` must be all the add creates in the folder"
        );
        assert!(
            crate::git::commands::run_git_command(fresh.path(), &["rev-parse", "HEAD"])
                .await
                .is_err(),
            "the initialised repository must have no commit"
        );

        // A folder that already is a repository: untouched — same HEAD, nothing re-initialised.
        let (_repo, repo_path) = crate::util::test::init_temp_repo();
        let head_before = crate::git::commands::run_git_command(&repo_path, &["rev-parse", "HEAD"])
            .await
            .expect("read HEAD before the add");
        store
            .add("add_git_existing", repo_path.to_str().unwrap())
            .await
            .expect("add existing repository");
        let head_after = crate::git::commands::run_git_command(&repo_path, &["rev-parse", "HEAD"])
            .await
            .expect("read HEAD after the add");
        assert_eq!(
            head_before, head_after,
            "an existing repository must be left exactly as it was"
        );
    }

    #[tokio::test]
    async fn claim_pending_for_discovery_is_atomic_and_returns_live_generation() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "pickup", "/tmp/pickup", false, false, 3, 0).await;

        let generation = store
            .claim_pending_for_discovery("pickup")
            .await
            .expect("claim")
            .expect("pending row should be claimable");
        assert_eq!(generation, 3, "must return the live discovery_generation");

        let ws = store
            .get_by_name("pickup")
            .await
            .expect("fetch")
            .expect("exists");
        assert_eq!(
            ws.status,
            WorkspaceStatus::Analyzing,
            "claim must transition pending → analyzing"
        );
        assert!(
            !ws.paused,
            "the claim does not pause: the analyzing status is what keeps the pipeline out"
        );

        // A second claim on the same row is a no-op (atomicity guard).
        let second = store
            .claim_pending_for_discovery("pickup")
            .await
            .expect("claim");
        assert!(
            second.is_none(),
            "a non-pending row must not be claimable twice"
        );
    }

    #[tokio::test]
    async fn reclassify_analyzing_to_pending_recovers_stranded_workspaces() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "stranded1", "/tmp/stranded1", true, false, 0, 0).await;
        insert_direct(&store, "stranded2", "/tmp/stranded2", true, false, 0, 0).await;
        insert_direct(&store, "fine", "/tmp/fine", false, false, 0, 0).await;
        store
            .set_status("stranded1", &WorkspaceStatus::Analyzing)
            .await
            .unwrap();
        store
            .set_status("stranded2", &WorkspaceStatus::Analyzing)
            .await
            .unwrap();
        store
            .set_status("fine", &WorkspaceStatus::Ready)
            .await
            .unwrap();

        let affected = store
            .reclassify_analyzing_to_pending()
            .await
            .expect("reclassify");
        assert_eq!(affected, 2, "only analyzing workspaces are reclassified");

        for name in ["stranded1", "stranded2"] {
            let ws = store
                .get_by_name(name)
                .await
                .expect("fetch")
                .expect("exists");
            assert_eq!(
                ws.status,
                WorkspaceStatus::Pending,
                "stranded analyzing workspace must become pending at boot"
            );
        }
        let fine = store
            .get_by_name("fine")
            .await
            .expect("fetch")
            .expect("exists");
        assert_eq!(
            fine.status,
            WorkspaceStatus::Ready,
            "non-analyzing workspaces are untouched"
        );
    }

    /// Build an agent with a preset typed failure class (simulating a failed
    /// run_agent) and classify it.
    fn classify_with(class: Option<crate::retry::FailureClass>) -> DiscoveryFailureKind {
        let ws = test_ws("/tmp/classify_ws");
        let mut agent = crate::Agent::new(
            "classify-test".into(),
            crate::Role::Discovery,
            &ws,
            crate::work_unit::WorkUnit::Turn,
            String::new(),
            String::new(),
            false,
            None,
        );
        agent.failure_class = class;
        classify_discovery_failure(&agent, None)
    }

    #[test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    fn discovery_failure_taxonomy_maps_provider_and_genuine_classes() {
        // classify_discovery_failure consults the process-global drain flag;
        // serialize against the drain-flag writers so a concurrent drain-flag
        // test cannot flip `Fatal` assertions to `Transient` mid-loop (see
        // `failure_classification_recognizes_drain` for the convention).
        // Provider-class → Transient (workspace returns to Pending).
        for class in [
            crate::retry::FailureClass::Transport,
            crate::retry::FailureClass::Unavailable,
            crate::retry::FailureClass::TruncatedEnvelope,
            crate::retry::FailureClass::NoResponse,
            crate::retry::FailureClass::TruncatedOutput,
            crate::retry::FailureClass::InvalidRequest, // auth / quota / invalid model
            crate::retry::FailureClass::ContentRejection, // repaired or shortened, it may pass
            crate::retry::FailureClass::Guardrail,
            crate::retry::FailureClass::Shutdown,
        ] {
            assert_eq!(
                classify_with(Some(class)),
                DiscoveryFailureKind::Transient,
                "{class:?} must be provider-class (Transient)"
            );
        }

        // Genuine failures → Fatal (workspace goes Failed).
        for class in [
            crate::retry::FailureClass::Parse,
            crate::retry::FailureClass::OutOfRangeScore,
            crate::retry::FailureClass::SchemaMismatch,
            crate::retry::FailureClass::Membership,
            crate::retry::FailureClass::Completeness,
            crate::retry::FailureClass::ContradictionAgents,
            crate::retry::FailureClass::ValidationOther,
        ] {
            assert_eq!(
                classify_with(Some(class)),
                DiscoveryFailureKind::Fatal,
                "{class:?} must be a genuine failure (Fatal)"
            );
        }

        // No typed class (runtime errors, panics) → Fatal.
        assert_eq!(
            classify_with(None),
            DiscoveryFailureKind::Fatal,
            "unclassified runtime failures must be Fatal"
        );
    }

    /// A workspace pause freezes a discovery in flight. That is a cooperative stop, so the
    /// workspace returns to Pending for the unpause to re-drive rather than going terminally Failed.
    #[test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    fn a_frozen_discovery_is_a_cooperative_stop() {
        let ws = test_ws("/tmp/frozen_discovery_ws");
        let agent = crate::Agent::new(
            "frozen-discovery-test".into(),
            crate::Role::Discovery,
            &ws,
            crate::work_unit::WorkUnit::Turn,
            String::new(),
            String::new(),
            false,
            None,
        );
        agent.mark_paused_frozen();
        assert_eq!(
            classify_discovery_failure(&agent, None),
            DiscoveryFailureKind::Transient,
            "a freeze returns the workspace to Pending rather than Failed"
        );
    }

    /// The pickup cooldown's whole lifecycle on its in-memory map: armed, it
    /// gates the pickup; an expired deadline does not; `clear` disarms it; the
    /// duration doubles per consecutive failure up to the 4 h cap, and the
    /// escalation counter increments with each record and is reset by `clear`.
    #[test]
    fn pending_pickup_cooldown_arms_escalates_expires_and_resets() {
        clear_pending_pickup_cooldown("cooldown_ws");
        assert!(
            !pending_pickup_cooldown_active("cooldown_ws"),
            "no cooldown by default"
        );

        record_pending_pickup_cooldown_until(
            "cooldown_ws",
            Instant::now() + Duration::from_mins(1),
        );
        assert!(
            pending_pickup_cooldown_active("cooldown_ws"),
            "armed cooldown must gate the pickup"
        );

        clear_pending_pickup_cooldown("cooldown_ws");
        assert!(
            !pending_pickup_cooldown_active("cooldown_ws"),
            "clear must disarm the cooldown"
        );

        // Expiry: a past deadline no longer gates the pickup.
        let past = Instant::now()
            .checked_sub(Duration::from_mins(1))
            .expect("instant subtraction cannot underflow in a test");
        record_pending_pickup_cooldown_until("cooldown_ws", past);
        assert!(
            !pending_pickup_cooldown_active("cooldown_ws"),
            "an expired cooldown must not gate the pickup"
        );
        clear_pending_pickup_cooldown("cooldown_ws");

        // The cooldown duration doubles per consecutive failure, capped at 4h.
        assert_eq!(pending_pickup_cooldown_duration(1), Duration::from_mins(15));
        assert_eq!(pending_pickup_cooldown_duration(2), Duration::from_mins(30));
        assert_eq!(pending_pickup_cooldown_duration(3), Duration::from_hours(1));
        assert_eq!(pending_pickup_cooldown_duration(4), Duration::from_hours(2));
        assert_eq!(
            pending_pickup_cooldown_duration(5),
            Duration::from_hours(4),
            "escalation caps at 4 h"
        );
        assert_eq!(
            pending_pickup_cooldown_duration(99),
            Duration::from_hours(4),
            "escalation never exceeds the 4 h cap"
        );

        // Consecutive production records increment the stored attempt count
        // (the escalation driver); clear resets it.
        clear_pending_pickup_cooldown("escalate_ws");
        record_pending_pickup_cooldown("escalate_ws");
        record_pending_pickup_cooldown("escalate_ws");
        let map = pending_pickup_cooldowns().lock().unwrap_poison();
        let entry = map
            .get("escalate_ws")
            .expect("cooldown entry after two failures");
        assert_eq!(entry.attempts, 2, "two failures → attempt count 2");
        let remaining = entry.deadline.saturating_duration_since(Instant::now());
        assert!(
            remaining >= Duration::from_mins(25),
            "second failure must arm a ~30 min cooldown (got {remaining:?})"
        );
        drop(map);
        clear_pending_pickup_cooldown("escalate_ws");
        assert!(
            !pending_pickup_cooldown_active("escalate_ws"),
            "clear must reset the cooldown"
        );
    }

    #[tokio::test]
    async fn rediscover_diagnostics_clears_and_bumps() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "redia_test", "/tmp/redia_test", false, false, 0, 0).await;

        let cmds = crate::DiagnosticsCommands {
            build: Some("cargo build".into()),
            ..Default::default()
        };
        store
            .set_diagnostics("redia_test", &cmds)
            .await
            .expect("set_diagnostics");
        assert!(
            store
                .get_diagnostics("redia_test")
                .await
                .expect("get_diagnostics")
                .is_some()
        );

        let diag_gen_before = store
            .get_generation("redia_test", GenerationColumn::DIAGNOSTICS)
            .await
            .expect("get_generation");
        assert_eq!(diag_gen_before, 1, "Should be 1 after set_diagnostics");

        store
            .rediscover_diagnostics("redia_test")
            .await
            .expect("rediscover_diagnostics");

        assert!(
            store
                .get_diagnostics("redia_test")
                .await
                .expect("get_diagnostics")
                .is_none(),
            "rediscover_diagnostics should clear diagnostics"
        );

        let diag_gen_after = store
            .get_generation("redia_test", GenerationColumn::DIAGNOSTICS)
            .await
            .expect("get_generation");
        assert_eq!(
            diag_gen_after, 2,
            "rediscover_diagnostics should bump diagnostics_generation to 2"
        );

        let discovery_gen_val = store
            .get_generation("redia_test", GenerationColumn::DISCOVERY)
            .await
            .expect("get_generation");
        assert_eq!(
            discovery_gen_val, 0,
            "rediscover_diagnostics should NOT affect discovery_generation"
        );
    }

    #[tokio::test]
    async fn general_context_roundtrip_single_row_per_workspace() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "gctx", "/tmp/gctx", true, false, 0, 0).await;

        assert_eq!(store.get_general_context("gctx").await.unwrap(), None);

        store
            .set_general_context("gctx", "overview v1")
            .await
            .unwrap();
        store
            .set_general_context("gctx", "overview v2")
            .await
            .unwrap();
        assert_eq!(
            store.get_general_context("gctx").await.unwrap().as_deref(),
            Some("overview v2")
        );

        // Partial unique index: a second NULL-role row must be rejected.
        let err = store
            .conn
            .execute(
                "INSERT INTO workspace_contexts (workspace_name, role, content, created_at) \
                 VALUES (?1, NULL, 'dup', ?2)",
                crate::db::params!["gctx", crate::db::now()],
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("UNIQUE"), "got: {err}");

        // Role-keyed rows stay distinct from the NULL-role row.
        store.set_context("gctx", "Manager", "mgr").await.unwrap();
        assert_eq!(
            store.get_general_context("gctx").await.unwrap().as_deref(),
            Some("overview v2")
        );
        assert_eq!(
            store
                .get_context("gctx", "Manager")
                .await
                .unwrap()
                .as_deref(),
            Some("mgr")
        );

        // A second workspace has its own NULL-role row.
        insert_direct(&store, "gctx2", "/tmp/gctx2", true, false, 0, 0).await;
        store
            .set_general_context("gctx2", "other overview")
            .await
            .unwrap();
        assert_eq!(
            store.get_general_context("gctx2").await.unwrap().as_deref(),
            Some("other overview")
        );
    }

    #[test]
    fn nightly_gate_allows_table() {
        // Hold `now` constant across all rows so elapsed-duration
        // computations stay deterministic.
        let now = Utc::now();
        let exact_7d = (now - chrono::Duration::days(7)).to_rfc3339();
        let one_sec_short =
            (now - chrono::Duration::days(7) + chrono::Duration::seconds(1)).to_rfc3339();
        let just_ran = now.to_rfc3339();
        let eight_days = (now - chrono::Duration::days(8)).to_rfc3339();
        let future = (now + chrono::Duration::hours(1)).to_rfc3339();

        let cases: &[(Option<&str>, bool, &str)] = &[
            (
                None,
                true,
                "No recorded pass (first night ever) must be allowed",
            ),
            (
                Some(exact_7d.as_str()),
                true,
                "Exactly 7 days elapsed must be allowed (>= 7 days)",
            ),
            (
                Some(one_sec_short.as_str()),
                false,
                "6d23h59m59s elapsed must be blocked",
            ),
            (
                Some(just_ran.as_str()),
                false,
                "A just-recorded pass must block",
            ),
            (
                Some(eight_days.as_str()),
                true,
                "8 days elapsed must be allowed",
            ),
            (
                Some(future.as_str()),
                false,
                "A future timestamp must block the pass",
            ),
            (
                Some("not-a-timestamp"),
                true,
                "An unparseable timestamp must let the pass through",
            ),
        ];
        for (last, expected, msg) in cases {
            assert_eq!(nightly_gate_allows(*last, now), *expected, "{msg}");
        }
    }

    #[tokio::test]
    async fn nightly_gate_records_pass_start_and_blocks() {
        let (store, _tmp) = crate::open_test_store!(crate::config_db::ConfigStore, "config");

        assert!(
            nightly_gate_should_run(Some(&store)).await,
            "First pass (no stored timestamp) must run",
        );
        assert!(
            store
                .get_kv(NIGHTLY_DISCOVERY_LAST_PASS_KV_KEY)
                .await
                .unwrap()
                .is_some(),
            "Pass start must be recorded in config_kv",
        );

        // Immediately after: the recorded pass start blocks the next pass.
        assert!(
            !nightly_gate_should_run(Some(&store)).await,
            "A second pass within the same 7-day window must be blocked",
        );

        // Corrupt stored value: fail-open, then the pass-start write
        // self-heals with a fresh parseable timestamp.
        store
            .set_kv(NIGHTLY_DISCOVERY_LAST_PASS_KV_KEY, "garbage")
            .await
            .unwrap();
        assert!(
            nightly_gate_should_run(Some(&store)).await,
            "An unparseable stored value must let the pass through",
        );
        let healed = store
            .get_kv(NIGHTLY_DISCOVERY_LAST_PASS_KV_KEY)
            .await
            .unwrap()
            .expect("pass-start write must store a value");
        assert!(
            crate::db::parse_utc_timestamp(&healed).is_ok(),
            "pass-start write must self-heal the stored value",
        );
    }

    #[tokio::test]
    async fn delete_resolves_the_admin_selection_and_clears_guests() {
        // delete() abandons the workspace's jobs through the jobs layer, which reads the
        // shared session store — initialize all stores first. That abandon runs against
        // the GLOBAL store and is a no-op here: this test exercises the selection cleanup
        // only (see jobs::workspace_delete_abandons_all_jobs).
        crate::util::test::init_test_stores().await;
        let (store, _tmp) = test_store().await;
        // delete() removes the workspace's search engine from the global
        // registry (init_global is idempotent).
        crate::workspace_index::init_global();
        for name in ["doomed_ws", "survivor_a", "survivor_b", "survivor_c"] {
            insert_direct(&store, name, &format!("/tmp/{name}"), false, false, 0, 0).await;
        }

        // Seed users via raw SQL: two guests reference the workspace being
        // deleted, one references an unrelated workspace, one has no selection,
        // and the admin's desktop is on the workspace being deleted.
        for (name, ws) in [
            ("alice", Some("doomed_ws")),
            ("bob", Some("doomed_ws")),
            ("carol", Some("other_ws")),
            ("dave", None),
            (crate::users::ADMIN_USER_NAME, Some("doomed_ws")),
        ] {
            store
                .conn
                .execute(
                    "INSERT INTO users (name, selected_workspace) VALUES (?1, ?2)",
                    crate::db::params![name, ws],
                )
                .await
                .expect("seed user");
        }

        // The admin follows the desktop's fallback pick — the first project in
        // the footer switcher's order — not the personal folder.
        assert_eq!(
            store.delete("doomed_ws").await.expect("delete workspace"),
            Some("survivor_a".to_string())
        );

        assert!(
            store.get_by_name("doomed_ws").await.unwrap().is_none(),
            "the workspace row must be deleted"
        );

        // Guest selections referencing it are cleared (NULL = personal
        // workspace): they keep their own behaviour.
        for name in ["alice", "bob"] {
            let selected = selected_workspace(&store, name).await;
            assert_eq!(selected, None, "{name}'s selection must be cleared");
        }
        assert_eq!(
            selected_workspace(&store, crate::users::ADMIN_USER_NAME)
                .await
                .as_deref(),
            Some("survivor_a")
        );

        // Other users are untouched: an unrelated selection is kept and a
        // user with no selection stays NULL.
        assert_eq!(
            selected_workspace(&store, "carol").await.as_deref(),
            Some("other_ws")
        );
        assert_eq!(selected_workspace(&store, "dave").await, None);

        // Deleting a project the desktop is not on moves nothing.
        assert_eq!(
            store.delete("survivor_c").await.expect("delete workspace"),
            Some("survivor_a".to_string())
        );
        assert_eq!(
            selected_workspace(&store, crate::users::ADMIN_USER_NAME)
                .await
                .as_deref(),
            Some("survivor_a")
        );

        // An empty admin selection is not a steady state: the deletion repairs it
        // onto the fallback rather than leaving the desktop outside every project.
        store
            .conn
            .execute(
                "UPDATE users SET selected_workspace = NULL WHERE name = ?1",
                crate::db::params![crate::users::ADMIN_USER_NAME],
            )
            .await
            .expect("clear the admin selection");
        assert_eq!(
            store.delete("survivor_b").await.expect("delete workspace"),
            Some("survivor_a".to_string())
        );

        // The last project leaves the admin on the personal folder.
        assert_eq!(
            store.delete("survivor_a").await.expect("delete workspace"),
            None
        );
        assert_eq!(
            selected_workspace(&store, crate::users::ADMIN_USER_NAME).await,
            None
        );
    }

    /// Read a user's `selected_workspace` column (NULL ↔ `None`).
    async fn selected_workspace(store: &WorkspaceStore, name: &str) -> Option<String> {
        store
            .conn
            .query_optional(
                "SELECT selected_workspace FROM users WHERE name = ?1",
                crate::db::params![name],
                |row| row.get::<Option<String>>(0),
            )
            .await
            .expect("query user workspace")
            .flatten()
    }

    /// A confirmed profile this build cannot read is a fact the owner is told — and never a reason
    /// to stop the workspace, which runs on the default.
    #[test]
    fn an_invalid_confirmed_profile_is_reported_to_the_owner_but_never_gates_the_pipeline() {
        let ws = Workspace {
            profile: Some("not json".to_string()),
            profile_confirmed: true,
            ..test_ws_named("/tmp/ws_invalid_profile", "ws_invalid_profile")
        };
        let reported = status_reason(&ws, &crate::workspace_profile::read(&ws))
            .expect("the owner is told the profile is invalid");
        assert!(
            reported.contains("confirmed profile is invalid") && reported.contains("default"),
            "got: {reported}"
        );
        assert!(
            pipeline_gate(&ws, &crate::workspace_profile::read(&ws)).is_none(),
            "an invalid profile never stops the workspace"
        );
    }

    /// A writer beside the workspace's own stage work is still another writer: the own work is what
    /// the workspace is doing, and the listing must not read idle while the gate is blocking.
    #[tokio::test]
    async fn a_second_writer_beside_the_workspaces_own_stage_work_is_still_reported() {
        use crate::agent::registry::{AGENT_REGISTRY, ParentKey};
        let ws = Workspace {
            profile_confirmed: true,
            ..test_ws_named("/tmp/ws_second_writer", "ws_second_writer")
        };
        let own = format!("own_{}", crate::generate_suffix());
        let own_generation = AGENT_REGISTRY.register(
            own.clone(),
            Role::Engineer.as_str().to_string(),
            None,
            &ws,
            std::sync::Arc::new(crate::stop::Stop::default()),
            Some(ParentKey::Ticket("second_writer_ticket".to_string())),
            None,
        );
        let other = format!("other_{}", crate::generate_suffix());
        let other_generation = AGENT_REGISTRY.register(
            other.clone(),
            Role::Worker.as_str().to_string(),
            None,
            &ws,
            std::sync::Arc::new(crate::stop::Stop::default()),
            Some(ParentKey::Launch(other.clone())),
            None,
        );

        let reported = status_reason(&ws, &crate::workspace_profile::read(&ws))
            .expect("the other writer is reported");
        assert!(
            reported.contains("waiting for a worker to finish"),
            "the holder reads as a person, not as its role token: {reported}"
        );
        assert!(
            pipeline_gate(&ws, &crate::workspace_profile::read(&ws)).is_some(),
            "the gate blocks either way"
        );

        AGENT_REGISTRY.deregister(&other, other_generation);
        AGENT_REGISTRY.deregister(&own, own_generation);
    }

    /// A parentless writer that is not the workspace's own background pass is still another
    /// writer: a router-dispatched worker round carries no parent key, and the listing must name it
    /// rather than report the workspace idle while the pipeline's gate blocks on it.
    #[tokio::test]
    async fn a_parentless_writer_beside_the_workspaces_own_work_is_reported() {
        use crate::agent::registry::AGENT_REGISTRY;
        let ws = Workspace {
            profile_confirmed: true,
            ..test_ws_named("/tmp/ws_parentless_writer", "ws_parentless_writer")
        };
        let round = format!("worker_{}", crate::generate_suffix());
        let generation = AGENT_REGISTRY.register(
            round.clone(),
            Role::Worker.as_str().to_string(),
            None,
            &ws,
            std::sync::Arc::new(crate::stop::Stop::default()),
            None,
            None,
        );

        let reported = status_reason(&ws, &crate::workspace_profile::read(&ws))
            .expect("a parentless worker round is a writer, not the workspace working");
        assert!(
            reported.contains("waiting for a worker to finish"),
            "the holder reads as a person, not as its role token: {reported}"
        );

        AGENT_REGISTRY.deregister(&round, generation);
    }

    /// Reading takes no hold and is not a writer, in either voice: a discovery pass, a separately
    /// launched analysis and a research run leave the folder free — neither the owner-facing
    /// listing nor the pipeline's gate sees a writer beside them — while a ticket's phase round and
    /// a research run's cleanup step hold it. The line is the work, not the role: the pipeline's
    /// analysis stage runs an analyst under a ticket and does hold.
    #[test]
    fn a_reading_pass_holds_nothing_for_either_voice() {
        use crate::agent::registry::{AGENT_REGISTRY, ParentKey};
        let ws = Workspace {
            profile_confirmed: true,
            ..test_ws_named("/tmp/ws_reading_holds_nothing", "ws_reading_holds_nothing")
        };
        let read = crate::workspace_profile::read(&ws);
        let mut registered = Vec::new();
        for (role, parent) in [
            (Role::Discovery, None),
            (
                Role::Analyst,
                Some(ParentKey::AnalyzeRound("reading_round".to_string())),
            ),
            (
                Role::Worker,
                Some(ParentKey::Research("reading_run".to_string())),
            ),
        ] {
            let agent_id = format!("{}_{}", role.as_str(), crate::generate_suffix());
            let generation = AGENT_REGISTRY.register(
                agent_id.clone(),
                role.as_str().to_string(),
                None,
                &ws,
                std::sync::Arc::new(crate::stop::Stop::default()),
                parent,
                None,
            );
            registered.push((agent_id, generation));
            assert!(
                AGENT_REGISTRY.folder_holder(ws.as_path(), &[]).is_none(),
                "{role} is reading: it holds nothing",
            );
            assert!(
                status_reason(&ws, &read).is_none(),
                "{role} is not another writer for the listing",
            );
            assert!(
                pipeline_gate(&ws, &read).is_none(),
                "and the gate starts work beside it",
            );
        }
        for (agent_id, generation) in registered {
            AGENT_REGISTRY.deregister(&agent_id, generation);
        }

        // A research run's cleanup deletes the run's folder: the one writer under that parent.
        let cleanup = format!("cleanup_{}", crate::generate_suffix());
        let cleanup_generation = AGENT_REGISTRY.register(
            cleanup.clone(),
            Role::Sanitation.as_str().to_string(),
            None,
            &ws,
            std::sync::Arc::new(crate::stop::Stop::default()),
            Some(ParentKey::Research("reading_run".to_string())),
            None,
        );
        assert_eq!(
            AGENT_REGISTRY
                .folder_holder(ws.as_path(), &[])
                .map(|holder| holder.agent_id),
            Some(cleanup.clone()),
            "a research run's cleanup writes the folder",
        );
        AGENT_REGISTRY.deregister(&cleanup, cleanup_generation);

        let stage = format!("ticket_{}", crate::generate_suffix());
        let stage_key = ParentKey::Ticket("reading_holds_ticket".to_string());
        let stage_generation = AGENT_REGISTRY.register(
            stage.clone(),
            Role::Analyst.as_str().to_string(),
            None,
            &ws,
            std::sync::Arc::new(crate::stop::Stop::default()),
            Some(stage_key.clone()),
            None,
        );
        let holder = AGENT_REGISTRY
            .folder_holder(ws.as_path(), &[])
            .expect("a ticket stage writes the folder");
        assert_eq!(holder.agent_id, stage, "the stage is the writer");
        assert!(
            !own_work_groups(&AGENT_REGISTRY.list(), &ws).is_empty(),
            "and the listing still calls the workspace's own stage its own work",
        );

        AGENT_REGISTRY.deregister(&stage, stage_generation);
    }

    /// The one holder naming both voices use: the launch roles by their own names, a pipeline role
    /// as itself.
    #[test]
    fn a_holder_is_named_the_same_way_in_a_wait_and_in_a_refusal() {
        assert_eq!(holder_name(Role::Worker.as_str()), "a worker");
        assert_eq!(holder_name(Role::Custom.as_str()), "a custom role");
        assert_eq!(holder_name(Role::Engineer.as_str()), "the engineer");
    }
}
