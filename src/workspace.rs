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
use crate::session::discovery_agent_id;
use crate::util::UnwrapPoison;
use anyhow::{Context, Result};
use chrono::{DateTime, Timelike, Utc};
use futures_util::future::join_all;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use strum::IntoEnumIterator;
use tracing::warn;

crate::define_store! {
    /// Global workspace store.
    pub static WORKSPACES: WorkspaceStore,
    expect = "workspace::WORKSPACES not initialized — call init_all_stores() first",
}

/// Look up a workspace by its name.
pub async fn get_by_name(name: &str) -> Result<Option<Workspace>> {
    store().get_by_name(name).await
}

// Column definitions for workspace SELECT queries.
// Note: `discovery_generation` and `diagnostics_generation` are intentionally
// excluded from this column list: both are read only via their own single-column
// SELECT in [`WorkspaceStore::get_generation`] and are never part of a workspace struct query.
crate::columns! {
    WORKSPACE_COLUMNS [WS] {
        NAME                  => "name",
        PATH                  => "path",
        STATUS                => "status",
        MAINTENANCE_ENABLED    => "maintenance",
        PAUSED                => "paused",
        MAINTAINER_DEBOUNCE_MINS => "maintainer_debounce_mins",
        MAINTAINER_LAST_RUN_AT  => "maintainer_last_run_at",
        DIAGNOSTICS           => "diagnostics",
        NOTES                  => "notes",
        LAST_ANALYZED_COMMIT   => "last_analyzed_commit",
    }
}

// ── Editor tab column constants ───────────────────────────────────────

crate::columns! {
    EDITOR_TAB_COLUMNS [ET] {
        FILE_PATH    => "file_path",
        TAB_ORDER    => "tab_order",
        IS_ACTIVE    => "is_active",
        IS_DIRTY     => "is_dirty",
        DIRTY_CONTENT => "dirty_content",
    }
}

// ── Workspace state list column constants ────────────────────────────

crate::columns! {
    WS_STATE_COLUMNS [WSST] {
        NAME         => "name",
        PAUSED       => "paused",
        MAINTENANCE_ENABLED => "maintenance",
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

/// Run workspace discovery for a single role, returning the result.
///
/// `discovery_generation` is the generation counter captured at spawn time.
/// Before writing the context, we re-read the current generation from the DB;
/// if it no longer matches, a newer [`WorkspaceStore::rediscover`] call has been made and this
/// task's result is stale — the write is skipped silently.
///
/// Returns `Ok(())` on success, or a classified [`DiscoveryRunError`].
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

    // Create a Discovery agent pointed at the workspace
    let agent_id = discovery_agent_id(&ws.name, label);
    let (agent, response) =
        run_default_agent(&agent_id, Role::Discovery, ws, &prompt, None, None, None).await;
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

/// Run diagnostics discovery — scan the workspace for dev tooling commands.
///
/// Runs a Discovery agent (using `Role::Discovery`'s tools: shell, read, search)
/// to scan build files and identify commands for format, lint, type-check, build,
/// and unit-test categories. Extracts structured output via the agent's scoped
/// extraction ([`crate::agent::Agent::extract_verdict`] with no validation).
///
/// `diagnostics_generation` guards against stale writes — if a newer
/// [`WorkspaceStore::rediscover_diagnostics`] or [`WorkspaceStore::set_diagnostics`]
/// was triggered while diagnostics were being computed, the write is skipped.
///
/// On failure, existing diagnostics data is left untouched.
async fn run_workspace_diagnostics(
    ws: &Workspace,
    diagnostics_generation: i64,
) -> Result<(), DiscoveryRunError> {
    let storage = WORKSPACES
        .get()
        .context("WORKSPACES not initialized")?
        .clone();

    tracing::info!(workspace_name = ws.name, "Starting diagnostics discovery");

    let agent_id = discovery_agent_id(&ws.name, DIAGNOSTICS_ROLE);

    // Load the diagnostics discovery prompt directly (not a role-specific discovery prompt).
    let prompt = crate::prompt::load_prompt("discovery/diagnostics.md");

    let (agent, response) =
        run_default_agent(&agent_id, Role::Discovery, ws, &prompt, None, None, None).await;
    let Some(_response) = response else {
        return Err(discovery_no_response_error(&agent, "Diagnostics discovery"));
    };

    // Keep the Agent alive after run_default_agent() for extract_verdict —
    // it needs agent.session.history() and agent.tool_specs.
    let extraction_prompt = crate::prompt::load_prompt("extraction/diagnostics.md");

    // Prefix-cache preservation: `agent.extract_verdict` uses the agent's own
    // parameters (model, reasoning_effort, tools, provider routing) with no
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

// ── Discovery completion finalizer ────────────────────────────────

/// Apply the final status and pause state after a discovery run completes.
///
/// This is called from [`spawn_workspace_discovery`] after all role discoveries
/// and diagnostics have finished.  Extracted as a separate function so unit tests
/// can verify the paused-behavior invariants without running real agents.
///
/// ## Invariants
///
/// - [`DiscoveryOutcome::AllOk`]: sets status to `ready` and unpauses the
///   workspace. The discovery flow itself set the analysis pause
///   (`add`/`rediscover`/pickup write `paused = 1` alongside
///   `status = Analyzing`), and rediscovery is the documented unpause path —
///   so a successful discovery clears it. While that pause is set, the claim
///   pipeline's gate also holds automatic Backlog→Analysis and
///   Queued→InDevelopment claims (see `run_claim_pipeline` in
///   the management module), so backlog and Queued tickets wait out the
///   discovery and are picked up on the next poll cycle after this unpause.
/// - [`DiscoveryOutcome::Fatal`] (at least one non-provider failure — runtime
///   errors, parse failures): sets status to `failed` (terminal, manual
///   Re-analyze as before) and leaves `paused` untouched. Panics never reach
///   this branch — they abort the spawned task and are recovered by boot
///   reclassification to `pending` (see [`spawn_panic_guarded`]).
/// - [`DiscoveryOutcome::Transient`] (every failure is provider-class —
///   auth/quota/transport/5xx/shutdown): sets status to `pending` and arms
///   the in-memory pickup cooldown, so the management poll's pickup step
///   retries the workspace once the provider is available again (no retry
///   storm). `paused` stays 1 (the analysis pause).
/// - Before any write, checks the generation guard: if a newer [`WorkspaceStore::rediscover`]
///   bumped the generation while discovery was in flight, the writes are skipped.
///
/// ## Known limitation
///
/// A pause that lands mid-flight (failure auto-pause or manual toggle after the
/// user unpaused an analyzing workspace) is indistinguishable from the
/// discovery's own analysis pause — the paused column carries no owner or
/// timestamp. The nightly loop skips paused workspaces, which closes the
/// scheduled path; the residual window requires a manual unpause mid-analysis.
async fn finalize_discovery(
    storage: &WorkspaceStore,
    ws_name: &str,
    ws_path: &str,
    discovery_generation: i64,
    outcome: DiscoveryOutcome,
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
        DiscoveryOutcome::AllOk => {
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
                    "status = ?, paused = 0, last_analyzed_commit = ?",
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

            tracing::info!(workspace = ws_name, "Workspace pipeline resumed");
            tracing::info!(
                workspace_name = ws_name,
                "Workspace analysis complete — all roles ready"
            );
        }
        DiscoveryOutcome::Fatal => {
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
        DiscoveryOutcome::Transient => {
            // Every failure was provider-class: return to pending. The pickup step
            // retries after the in-memory cooldown — the workspace waits for the
            // provider (or a key fix) without burning retry budgets repeatedly.
            //
            // Arm the cooldown BEFORE the status write: while the row is still
            // `analyzing` the pickup step cannot claim it, so there is no window
            // where a poll round sees `pending` with no cooldown armed and spawns
            // a duplicate discovery run (arming after the await would leave a
            // TOCTOU gap that defeats the cooldown for this cycle).
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
    }
}

// ── Discovery failure classification ──────────────────────────────

/// Two-bucket classification of a discovery failure.
///
/// * [`DiscoveryFailureKind::Transient`] — provider-class (auth 401/403,
///   quota/insufficient balance, transport/unreachable/5xx, shutdown/cancel):
///   the workspace returns to Pending and the pickup step retries it after an
///   in-memory cooldown.
/// * [`DiscoveryFailureKind::Fatal`] — genuine failure (internal runtime
///   errors, parse failures): the workspace goes Failed, terminal (manual
///   Re-analyze as before). Panics never reach this classifier — they abort
///   the spawned discovery task and strand the workspace in `analyzing` until
///   the boot reclassification reclasses it to `pending` (a retryable
///   treatment, not terminal Failed).
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

/// Default conversion for `?` propagation: an unclassified plain error (DB
/// failure, missing global, ...) is a genuine runtime failure — fatal.
/// Agent-run failures must NOT use this conversion: they go through
/// [`DiscoveryRunError::classified`], which buckets the typed
/// [`crate::retry::FailureClass`] — that is the only path that yields
/// [`DiscoveryRunError::transient`].
impl From<anyhow::Error> for DiscoveryRunError {
    fn from(error: anyhow::Error) -> Self {
        Self::fatal(error)
    }
}

/// Classify a failed discovery agent run into the two-bucket scheme.
///
/// Mirrors [`crate::agent::failure_classification`] ordering: drain/shutdown/
/// cancellation are treated as transient (the run was cut short, not a genuine
/// failure — the workspace retries after restart/cooldown; a cancelled
/// discovery deliberately returns to Pending instead of the historical
/// terminal Failed, since cancellation is not a workspace defect). A typed
/// [`crate::retry::RetryExhausted`] class decides provider-vs-genuine for
/// retry-loop terminations; everything else (untyped runtime errors) is
/// fatal. Panics never reach the classifier — they abort the spawned task
/// and strand the workspace in `analyzing` until boot reclassifies it to
/// `pending` (see [`spawn_panic_guarded`]).
///
/// ## Deliberate bucket decisions
///
/// * **`NonRetryable` → Transient** — the class covers auth 401/403 (the
///   ticket's primary "bad or missing key" case) as well as invalid-model and
///   tool-schema 400s. All of these are *config*-fixable — the workspace must
///   wait in Pending for the user to correct the settings rather than go
///   terminal Failed. The escalating cooldown bounds the retry rate, so a
///   permanently-wrong config cannot burn unbounded tokens.
/// * **`WallClockExceeded` → Transient** — the 12-minute retry budget being
///   exhausted is almost always the symptom of sustained provider
///   unavailability, which the cooldown is designed to ride out.
/// * **`Parse`/validation classes → Fatal** — the provider answered, so the
///   failure is a genuine pipeline/format defect; retrying later would not
///   help.
///
/// `error` is the raw error when one is available at the call site (e.g. a
/// failed `extract_verdict`); the typed class captured by `run_agent` on the
/// agent takes precedence (it is the original error of the agent loop).
fn classify_discovery_failure(
    agent: &crate::Agent,
    error: Option<&anyhow::Error>,
) -> DiscoveryFailureKind {
    if crate::shutdown::is_draining()
        || crate::shutdown::shutdown_token().is_cancelled()
        || agent.is_cancelled()
    {
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
        | crate::retry::FailureClass::TruncatedEnvelope
        | crate::retry::FailureClass::NoResponse
        | crate::retry::FailureClass::TruncatedOutput
        | crate::retry::FailureClass::NonRetryable
        | crate::retry::FailureClass::WallClockExceeded
        | crate::retry::FailureClass::Shutdown => DiscoveryFailureKind::Transient,
        crate::retry::FailureClass::Parse
        | crate::retry::FailureClass::OutOfRangeScore
        | crate::retry::FailureClass::Membership
        | crate::retry::FailureClass::Completeness
        | crate::retry::FailureClass::ContradictionAgents
        | crate::retry::FailureClass::ValidationOther => DiscoveryFailureKind::Fatal,
    }
}

/// Aggregate outcome of a discovery run — encodes the success/fatal/transient
/// trichotomy structurally (a success implies no fatal failure), replacing a
/// pair of bools that could express invalid combinations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiscoveryOutcome {
    /// Every sub-run succeeded.
    AllOk,
    /// At least one fatal (non-provider) failure — terminal `Failed`.
    Fatal,
    /// Only provider-class (transient) failures — return to `Pending`.
    Transient,
}

/// Fold one discovery sub-run result into the aggregate outcome shared by the
/// role/general/diagnostics sites in [`spawn_workspace_discovery`]: degrades
/// [`DiscoveryOutcome::AllOk`] on any error, forces
/// [`DiscoveryOutcome::Fatal`] on fatal errors ("any fatal → Failed"
/// precedence), and records the error string (with `prefix`) for the finalize
/// message. Error strings are scrubbed here so agent failure chains (which
/// may embed provider error bodies) never reach the finalize logs
/// unscrubbed.
fn fold_discovery_result(
    outcome: &mut DiscoveryOutcome,
    errors: &mut Vec<String>,
    result: Result<(), DiscoveryRunError>,
    prefix: &str,
) {
    if let Err(e) = result {
        errors.push(crate::util::scrub_credentials(&format!("{prefix}{e}")));
        if e.kind == DiscoveryFailureKind::Fatal {
            *outcome = DiscoveryOutcome::Fatal;
        } else if *outcome == DiscoveryOutcome::AllOk {
            *outcome = DiscoveryOutcome::Transient;
        }
    }
}

/// Build the classified error for an agent run that produced no response.
/// Shared by [`run_discovery_task`] and [`run_workspace_diagnostics`] — the
/// only difference between the two sites is the task label in the message.
fn discovery_no_response_error(agent: &crate::Agent, task_label: &str) -> DiscoveryRunError {
    DiscoveryRunError::classified(
        agent,
        anyhow::anyhow!(
            "{task_label} agent returned no response: {}",
            agent.failure_reason("unknown error")
        ),
    )
}

// ── Pending-pickup cooldown (in-memory) ───────────────────────────

/// Base cooldown for the first provider-class discovery failure (15 min).
const PENDING_PICKUP_COOLDOWN_BASE_MINS: u64 = 15;
/// Ceiling for the escalating pickup cooldown (4 h).
const PENDING_PICKUP_COOLDOWN_MAX_MINS: u64 = 240;

/// Escalating cooldown for the `attempts`-th consecutive provider-class
/// discovery failure: 15 min, 30 min, 1 h, 2 h, then 4 h forever.
///
/// Memory-only by design: no DB writes, no new columns; after a
/// restart a pending workspace may be retried once immediately (acceptable).
/// Escalation bounds the long-run retry duty cycle of a persistently-down
/// provider — the full 12-minute retry budget no longer burns every 15
/// minutes forever, which would violate the ticket's "no repeated 12-minute
/// LLM-burn loops" criterion.
fn pending_pickup_cooldown_duration(attempts: u32) -> Duration {
    let exponent = attempts.saturating_sub(1).min(4);
    let minutes = (PENDING_PICKUP_COOLDOWN_BASE_MINS * 2u64.pow(exponent))
        .min(PENDING_PICKUP_COOLDOWN_MAX_MINS);
    Duration::from_mins(minutes)
}

/// Per-workspace cooldown state.
#[derive(Debug, Clone, Copy)]
struct PickupCooldown {
    /// Instant before which the pickup step must not claim the workspace.
    deadline: Instant,
    /// Consecutive provider-class discovery failures since the last success
    /// or manual reset — drives the escalating cooldown duration.
    attempts: u32,
}

/// In-memory cooldown map: workspace name → cooldown state. Written by
/// [`finalize_discovery`] on provider-class failure; cleared on successful
/// discovery, manual rediscover, workspace delete, and re-add.
static PENDING_PICKUP_COOLDOWNS: OnceLock<std::sync::Mutex<HashMap<String, PickupCooldown>>> =
    OnceLock::new();

fn pending_pickup_cooldowns() -> &'static std::sync::Mutex<HashMap<String, PickupCooldown>> {
    PENDING_PICKUP_COOLDOWNS.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Arm the (escalating) cooldown for a provider-class discovery failure: the
/// pending pickup for `ws_name` waits before claiming it again. Each
/// consecutive failure lengthens the wait (see
/// [`pending_pickup_cooldown_duration`]); success, manual rediscover, delete,
/// and re-add reset it via [`clear_pending_pickup_cooldown`].
///
/// In the rare race where a transient finalize lands after a concurrent
/// delete's clear, the re-armed entry is stale — harmless (the map is bounded
/// by workspace count) and cleared again on re-add.
pub(crate) fn record_pending_pickup_cooldown(ws_name: &str) {
    let mut map = pending_pickup_cooldowns().lock().unwrap_poison();
    let attempts = map.get(ws_name).map_or(0, |c| c.attempts) + 1;
    let deadline = Instant::now() + pending_pickup_cooldown_duration(attempts);
    map.insert(ws_name.to_string(), PickupCooldown { deadline, attempts });
}

/// Arm the cooldown with an explicit deadline (test hook; production callers
/// use [`record_pending_pickup_cooldown`]).
#[cfg(test)]
pub(crate) fn record_pending_pickup_cooldown_until(ws_name: &str, deadline: Instant) {
    let mut map = pending_pickup_cooldowns().lock().unwrap_poison();
    map.insert(
        ws_name.to_string(),
        PickupCooldown {
            deadline,
            attempts: 1,
        },
    );
}

/// Clear the cooldown for a workspace (successful discovery, manual
/// rediscover, delete, re-add) — also resets the escalation counter.
pub(crate) fn clear_pending_pickup_cooldown(ws_name: &str) {
    pending_pickup_cooldowns()
        .lock()
        .unwrap_poison()
        .remove(ws_name);
}

/// Whether the pickup step must wait out a cooldown for `ws_name` before
/// claiming its pending workspace.
///
/// Expired entries are deliberately retained (the map is bounded by workspace
/// count): the `attempts` escalation counter must survive expiry so the next
/// provider-class failure arms a *longer* cooldown rather than restarting at
/// the 15-minute base — pruning here would reset the escalation on every
/// pickup cycle. Entries are removed by [`clear_pending_pickup_cooldown`] on
/// success, manual rediscover, delete, and re-add.
pub(crate) fn pending_pickup_cooldown_active(ws_name: &str) -> bool {
    let map = pending_pickup_cooldowns().lock().unwrap_poison();
    map.get(ws_name)
        .is_some_and(|c| Instant::now() < c.deadline)
}

/// Spawn `future` in a panic-guarded sub-task and await it, logging a
/// workspace-level error if the task panics or is cancelled.
///
/// NOTE: Unlike the ticket-dispatch panic recovery (which transitions the
/// ticket to Failed), this guard only logs and does NOT transition the
/// workspace to "failed". A non-prompt panic leaves the workspace in
/// "analyzing" — visible in logs; the boot reclassification
/// (`reclassify_analyzing_to_pending`) recovers it at the next startup.
async fn spawn_panic_guarded(
    ws_name: &str,
    task: &str,
    future: impl std::future::Future<Output = ()> + Send + 'static,
) {
    let inner = tokio::spawn(future);
    match inner.await {
        Ok(()) => {}
        Err(e) => {
            let kind = if e.is_panic() { "panic" } else { "cancelled" };
            tracing::error!(
                workspace_name = %ws_name,
                kind = kind,
                error = %e,
                "{task} task failed",
            );
        }
    }
}

/// Spawn a background task that runs workspace discovery (per-role and
/// general non-role context generation) and optionally diagnostics discovery.
///
/// `discovery_generation` is the generation counter captured at spawn time.
/// The discovery functions use it to guard against stale writes.
///
/// When `discover_diagnostics` is `false` (e.g. during a re-analysis via
/// [`WorkspaceStore::rediscover`], or a pickup that found existing
/// diagnostics), diagnostics discovery is skipped so that user-managed
/// diagnostics survive re-analysis.
pub fn spawn_workspace_discovery(
    ws: &Workspace,
    discovery_generation: i64,
    discover_diagnostics: bool,
) {
    let ws = ws.clone();
    tokio::spawn(async move {
        let ws_name = ws.name.clone();
        let ws_path = ws.path.clone();

        let ws_name_for_finalize = ws_name.clone();
        let ws_name_for_inner = ws_name.clone();
        let ws_path_for_finalize = ws_path.clone();
        let inner = async move {
            // Build role discovery futures (always needed). The general
            // (non-role) project overview discovery runs alongside them.
            let role_futures: Vec<_> = Role::iter()
                .filter(|r| crate::agent::role::role_info(r).has_discovery)
                .map(|role| {
                    let ws = ws.clone();
                    async move { run_workspace_discovery(&ws, role, discovery_generation).await }
                })
                .collect();

            // Run role discovery + general discovery always, optionally with diagnostics.
            let (role_results, general_result, diagnostics_result) = if discover_diagnostics {
                // Read diagnostics generation from DB for the generation guard.
                // Separate from discovery_generation: both counters are independent
                // (diagnostics is bumped by set_diagnostics/rediscover_diagnostics,
                // discovery is bumped by rediscover).
                let diag_gen = match WORKSPACES.get() {
                    Some(s) => s
                        .get_generation(&ws_name_for_inner, GenerationColumn::DIAGNOSTICS)
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

            let mut outcome = DiscoveryOutcome::AllOk;
            let mut errors: Vec<String> = Vec::new();

            for result in role_results {
                fold_discovery_result(&mut outcome, &mut errors, result, "");
            }

            // General/diagnostics failures fail the run; the failure kind
            // decides whether the workspace returns to Pending (provider-class)
            // or goes Failed (genuine). Transient diagnostics failures are
            // retried via the pickup step, not terminal.
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
                &ws_name_for_finalize,
                &ws_path_for_finalize,
                discovery_generation,
                outcome,
                &errors,
            )
            .await;
        };

        // Box the discovery payload: it embeds role/general/diagnostics
        // futures (incl. ChatResponse-sized structs) and exceeds the
        // 16 KB large_futures threshold as an inline async-block.
        spawn_panic_guarded(&ws_name, "spawn_workspace_discovery", Box::pin(inner)).await;
    });
}

/// Spawn a background task that runs diagnostics discovery only.
///
/// Unlike [`spawn_workspace_discovery`], this does **not** run per-role
/// context discovery — it only re-discovers diagnostics commands.
/// Used by [`WorkspaceStore::rediscover_diagnostics`] for the "Re-discover
/// diagnostics" button in the GUI.
///
/// `diagnostics_generation` is the generation counter captured at spawn time.
/// [`run_workspace_diagnostics`] uses it to guard against stale writes via
/// [`check_generation`].
pub fn spawn_diagnostics_discovery(ws: &Workspace, diagnostics_generation: i64) {
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

/// Validate a workspace name against the naming rules.
///
/// Rules:
/// - ASCII letters (a-z, A-Z) and underscores only
/// - Must start with a letter — no leading underscore
/// - At least one letter — not underscores-only
/// - Maximum 40 characters
fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("Workspace name must not be empty");
    }
    if name.len() > 40 {
        anyhow::bail!("Workspace name must not exceed 40 characters");
    }
    if !name.chars().all(|c| c.is_ascii_alphabetic() || c == '_') {
        anyhow::bail!("Workspace name must contain only ASCII letters (a-z, A-Z) and underscores");
    }
    if !name.starts_with(|c: char| c.is_ascii_alphabetic()) {
        anyhow::bail!("Workspace name must start with a letter");
    }
    if !name.chars().any(|c| c.is_ascii_alphabetic()) {
        anyhow::bail!("Workspace name must contain at least one letter");
    }
    Ok(())
}

/// Ensure a directory path string ends with a single `/`.
fn ensure_trailing_slash(path: &str) -> String {
    let trimmed = path.trim_end_matches('/');
    format!("{trimmed}/")
}

/// Canonicalize a user-provided path for workspace storage.
///
/// Expands `~` to the user's home directory, then uses
/// [`std::fs::canonicalize`] to resolve relative segments and symlinks.
/// Returns a clear error message on failure so callers can surface it
/// to the user (e.g. "Path does not exist" or "Not a directory").
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

    Ok(canonical.to_string_lossy().to_string())
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
        maintainer_debounce_mins: row.get::<i64>(COL_WS_MAINTAINER_DEBOUNCE_MINS)?,
        maintainer_last_run_at: row.get::<Option<String>>(COL_WS_MAINTAINER_LAST_RUN_AT)?,
        diagnostics: row.get::<Option<String>>(COL_WS_DIAGNOSTICS)?,
        notes: row.get::<String>(COL_WS_NOTES)?,
        last_analyzed_commit: row.get::<Option<String>>(COL_WS_LAST_ANALYZED_COMMIT)?,
        ephemeral: false,
    })
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
    /// Run an UPDATE on `workspaces` that sets `set_clause` plus
    /// `updated_at = now` for a single named row — mirrors the ticket-update
    /// helper in `board.rs` to keep placeholder numbering uniform.
    async fn exec_update_with_updated_at(
        &self,
        set_clause: &str,
        set_params: Vec<db::Value>,
        name: &str,
    ) -> Result<()> {
        let sql = format!("UPDATE workspaces SET {set_clause}, updated_at = ? WHERE name = ?");
        let mut params = set_params;
        params.push(Value::from(db::now()));
        params.push(Value::from(name));
        self.conn.execute(&sql, params).await?;
        Ok(())
    }

    /// Insert a new workspace and register it for analysis.
    ///
    /// The workspace is written with status `pending` (paused=1) and the
    /// discovery spawn is deferred to the management poll's pickup step:
    /// discovery starts automatically once the LLM provider is configured, so
    /// a first-run "workspace added before the key" no longer flips to Failed
    /// seconds later. Manual Re-analyze and the nightly rediscovery path are
    /// unaffected.
    pub async fn add(&self, name: &str, path: &str) -> Result<Workspace> {
        // Validate the workspace name.
        validate_name(name)?;

        // Canonicalize and validate the path so bad paths never enter the system.
        let canonical = canonicalize_workspace_path(path).map_err(|e| anyhow::anyhow!("{e}"))?;
        let path = ensure_trailing_slash(&canonical);
        let now = db::now();
        let pending = WorkspaceStatus::Pending.to_string();
        let ws = self
            .conn
            .query_row(
                &format!(
                    "INSERT INTO workspaces (name, path, status, created_at, updated_at, paused) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6) RETURNING {WORKSPACE_COLUMNS}"
                ),
                db::params![name, path, pending, now.clone(), now.clone(), 1],
                workspace_from_row,
            )
            .await?;
        // No discovery spawn here: the pickup step claims the pending
        // workspace once the provider is configured. discovery_generation
        // defaults to 0 in the schema — the pickup reads the live value.
        // Clear any stale cooldown from a previously-deleted same-name row.
        clear_pending_pickup_cooldown(name);
        // Eagerly initialize the shared search engine for this workspace.
        if let Err(e) =
            crate::search_engine::get_or_init_engine(name, std::path::Path::new(&ws.path), false)
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

    /// Lightweight fetch of only name, paused, and maintenance_enabled columns.
    /// Used by the GUI sidebar's periodic state refresh — avoids fetching
    /// all workspace columns when only toggle state is needed.
    pub async fn list_states(&self) -> Result<Vec<(String, bool, bool)>> {
        let rows = self
            .conn
            .query(
                &format!("SELECT {WS_STATE_COLUMNS} FROM workspaces ORDER BY name"),
                db::params![],
            )
            .await?;
        let mut states = Vec::with_capacity(rows.len());
        for row in &rows {
            let name: String = row.get(COL_WSST_NAME)?;
            let paused: bool = row.get(COL_WSST_PAUSED)?;
            let maintenance_enabled: bool = row.get(COL_WSST_MAINTENANCE_ENABLED)?;
            states.push((name, paused, maintenance_enabled));
        }
        Ok(states)
    }

    /// Look up a workspace by name.
    pub async fn get_by_name(&self, name: &str) -> Result<Option<Workspace>> {
        self.conn
            .query_optional_cached(
                &format!("SELECT {WORKSPACE_COLUMNS} FROM workspaces WHERE name = ?1"),
                db::params![name],
                workspace_from_row,
            )
            .await
    }

    /// Delete a workspace by name. Context rows are cascaded automatically.
    /// The associated search engine is also removed from the in-memory registry.
    pub async fn delete(&self, name: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM workspaces WHERE name = ?1", db::params![name])
            .await?;
        crate::search_engine::remove_engine(name);
        clear_pending_pickup_cooldown(name);
        Ok(())
    }

    /// Update the status of a workspace.
    pub async fn set_status(&self, name: &str, status: &WorkspaceStatus) -> Result<()> {
        self.exec_update_with_updated_at("status = ?", vec![Value::from(status.to_string())], name)
            .await
    }

    /// Atomically claim a pending workspace for its first discovery:
    /// transitions status `pending` → `analyzing` (with the analysis pause)
    /// and returns the row's live `discovery_generation`, captured in the same
    /// statement so a concurrent `rediscover` generation bump cannot race the
    /// read.
    ///
    /// Returns `Ok(None)` when the row was no longer pending — a concurrent
    /// pickup, GUI Re-analyze, or delete won the race; the caller must not
    /// spawn.
    pub(crate) async fn claim_pending_for_discovery(&self, name: &str) -> Result<Option<i64>> {
        self.conn
            .query_optional(
                "UPDATE workspaces SET status = ?, paused = 1, updated_at = ? \
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

    /// Boot recovery: discovery leaves no job rows, so a workspace still in
    /// `analyzing` at startup means a crashed/panicked mid-discovery run.
    /// Reclassify it to `pending` so the management poll's pickup step retries
    /// it (or waits for the provider). Returns the number of reclassified
    /// workspaces.
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

    /// Set or clear the maintenance toggle for a workspace.
    pub async fn set_maintenance_enabled(&self, name: &str, enabled: bool) -> Result<()> {
        let val: i64 = i64::from(enabled);
        if enabled {
            // Reset debounce state so the maintainer runs on the very next
            // 1-minute poll cycle (last_run_at = NULL bypasses the debounce
            // gate), regardless of how long the previous interval was.
            self.exec_update_with_updated_at(
                "maintenance = ?, maintainer_debounce_mins = 5, maintainer_last_run_at = NULL",
                vec![Value::from(val)],
                name,
            )
            .await?;
        } else {
            self.exec_update_with_updated_at("maintenance = ?", vec![Value::from(val)], name)
                .await?;
            // Cancel any running maintainer agent for this workspace so it
            // doesn't continue creating tickets after maintenance was disabled.
            if let Some(ws) = self.get_by_name(name).await? {
                crate::agent::registry::AGENT_REGISTRY
                    .cancel_by_role_and_workspace_path(Role::Maintainer.as_str(), &ws.path);
            }
        }
        if enabled {
            tracing::info!(workspace = name, "Maintainer enabled");
        } else {
            tracing::info!(workspace = name, "Maintainer disabled");
        }
        Ok(())
    }

    /// Set or clear the pipeline pause toggle for a workspace.
    pub async fn set_paused(&self, name: &str, paused: bool) -> Result<()> {
        let val: i64 = i64::from(paused);
        self.exec_update_with_updated_at("paused = ?", vec![Value::from(val)], name)
            .await?;
        if paused {
            // Keyed to the user/operator/failure pause — the one place that
            // sets `paused` via this method. The discovery "analysis-pause"
            // writes `paused` via DIRECT SQL in `add`/`claim_pending_for_discovery`/
            // `rediscover`, NOT via `set_paused`, so it does NOT trigger this
            // cancel and must not cancel the in-flight discovery agents.
            //
            // The store→registry call mirrors `set_maintainer` below (which
            // cancels Maintainer agents): the store is the single choke point
            // for every pause entry point (GUI, Telegram, failure), so the
            // cancel is guaranteed rather than left to each caller.
            crate::agent::registry::AGENT_REGISTRY.cancel_by_workspace_pause(name);
            tracing::info!(workspace = name, "Workspace pipeline paused");
        } else {
            // Clear the cooperative pause-stop flags left on any in-flight
            // agents so a resumed workspace does not freeze them at their next
            // LLM round boundary.
            crate::agent::registry::AGENT_REGISTRY.clear_workspace_pause(name);
            tracing::info!(workspace = name, "Workspace pipeline resumed");
        }
        Ok(())
    }

    /// Update the maintenance debounce state atomically.
    ///
    /// Sets both `maintainer_debounce_mins` and `maintainer_last_run_at` in one
    /// UPDATE along with `updated_at`.
    pub async fn set_maintenance_debounce(
        &self,
        name: &str,
        debounce_mins: i64,
        last_run_at: &str,
    ) -> Result<()> {
        self.exec_update_with_updated_at(
            "maintainer_debounce_mins = ?, maintainer_last_run_at = ?",
            vec![Value::from(debounce_mins), Value::from(last_run_at)],
            name,
        )
        .await
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
    }

    /// Retrieve discovered diagnostics commands for a workspace.
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
        match json {
            Some(json) => Ok(Some(serde_json::from_str(&json)?)),
            None => Ok(None),
        }
    }

    /// Set freeform user-curated context notes for a workspace.
    ///
    /// Truncates to `MAX_WORKSPACE_NOTES_CHARS` characters as defense-in-depth
    /// against prompt bloat. Notes are appended to every agent's system prompt.
    pub async fn set_notes(&self, name: &str, notes: &str) -> Result<()> {
        let notes = truncate_workspace_notes(notes);
        self.exec_update_with_updated_at("notes = ?", vec![Value::from(notes)], name)
            .await
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

    /// Trigger re-analysis of an existing workspace.
    /// Resets status to "analyzing", clears stale per-role contexts, and
    /// spawns analysis with a fresh generation counter.
    ///
    /// Unlike [`Self::rediscover_diagnostics`], this does **not** clear
    /// diagnostics — user-managed diagnostics survive re-analysis. Diagnostics
    /// discovery runs only when none exist yet (a never-analyzed `Pending`
    /// workspace re-analyzed manually would otherwise reach Ready with no
    /// diagnostics ever discovered, since `add()` no longer spawns discovery);
    /// when diagnostics are present they are preserved untouched.
    ///
    /// Manual Re-analyze bypasses the pending-pickup cooldown by design: it
    /// spawns discovery immediately, so any in-memory cooldown is cleared
    /// here (a later provider-class failure re-arms it).
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
            "discovery_generation = discovery_generation + 1, status = ?, paused = 1",
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
    /// paused state, per-role contexts, or the discovery generation counter.
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

    /// Get a single context entry by workspace name and role.
    pub async fn get_context(&self, name: &str, role: &str) -> Result<Option<String>> {
        self.conn
            .query_optional(
                "SELECT content FROM workspace_contexts WHERE workspace_name = ?1 AND role = ?2",
                db::params![name, role],
                |row| row.get::<String>(0),
            )
            .await
    }

    /// Upsert a single context entry for a workspace and role.
    pub async fn set_context(&self, name: &str, role: &str, content: &str) -> Result<()> {
        let now = db::now();
        self.conn
            .execute(
                "INSERT INTO workspace_contexts (workspace_name, role, content, created_at) VALUES (?1, ?2, ?3, ?4) \
                 ON CONFLICT(workspace_name, role) DO UPDATE SET content = excluded.content, created_at = excluded.created_at",
                db::params![name, role, content, now],
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
    pub async fn set_general_context(&self, name: &str, content: &str) -> Result<()> {
        let now = db::now();
        self.conn
            .execute(
                "INSERT INTO workspace_contexts (workspace_name, role, content, created_at) VALUES (?1, NULL, ?2, ?3) \
                 ON CONFLICT(workspace_name) WHERE role IS NULL DO UPDATE SET content = excluded.content, created_at = excluded.created_at",
                db::params![name, content, now],
            )
            .await?;
        Ok(())
    }

    // ── Editor tab persistence ─────────────────────────────────

    /// Save the current set of open editor tabs for a workspace.
    /// Replaces all existing records for this workspace.
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

    /// Load the saved open editor tabs for a workspace.
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

/// List all workspaces (for display).
pub async fn get_workspaces() -> anyhow::Result<Vec<Workspace>> {
    let store = WORKSPACES
        .get()
        .ok_or_else(|| anyhow::anyhow!("Workspace store not initialized"))?;
    store.list().await
}

/// `config_kv` key holding the RFC 3339 UTC timestamp of the last nightly
/// rediscovery pass start.
///
/// Deliberately stored in `config_kv` (in the consolidated `core.db`) rather than in the
/// workspaces table: the schema has no migration path (new columns are
/// invisible on existing live databases) and workspace rows are deleted
/// during rediscovery, so the timestamp must live in a table that outlives
/// workspace churn. Unknown `config_kv` keys are purged on reload, except the
/// preserved shared namespaces (this key and telegram_role_pin:*) which are left
/// untouched.
pub(crate) const NIGHTLY_DISCOVERY_LAST_PASS_KV_KEY: &str = "nightly_discovery_last_pass_at";

/// Returns `true` when the given local hour falls within the nightly
/// re-analysis window (2:00–3:00 AM, inclusive of 2, exclusive of 3).
fn is_nightly_check_hour(local_hour: u32) -> bool {
    (2..3).contains(&local_hour)
}

/// Returns `true` when the rolling 7-day frequency gate allows a nightly pass.
///
/// A pass is allowed when no pass has been recorded yet (`None` — first night
/// ever), when the stored timestamp is unparseable (fail-open, mirroring the
/// maintainer-debounce precedent; the pass-start write then records a fresh
/// timestamp and self-heals the stored state), or when at least 7 × 24 h
/// (wall-clock duration, DST-safe, no fixed weekday pinning) have elapsed
/// since the recorded pass start.
///
/// A well-formed timestamp less than 7 days old blocks the pass — including
/// a future timestamp (clock skew → negative elapsed), which stays blocked
/// until 7 days after that timestamp — so per-workspace regeneration happens
/// at most once per 7 days.
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

/// Evaluate the rolling 7-day frequency gate and, when the pass is allowed,
/// record the pass start in `config_kv` BEFORE any workspace processing: a
/// pass that starts counts as the last pass even if it finds zero eligible
/// workspaces or is interrupted mid-way.
///
/// Returns `true` when the nightly pass should run. Failure policy:
///
/// * **Read** (missing store, read error, unparseable stored value):
///   fail-open — the pass runs and the pass-start write records a fresh
///   timestamp, self-healing the state (maintainer-debounce precedent).
/// * **Pass-start write failure**: fail-closed — the pass is skipped so the
///   at-most-once-per-7-days invariant holds (running without recording would
///   allow a second pass within the same 7-day window on a later night).
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

/// Run the nightly re-analysis loop.
///
/// Wakes every 30 minutes. During the 2:00–3:00 AM local time window, runs a
/// single discovery pass: each Ready workspace is checked for new git commits
/// and rediscovered when its HEAD differs from the stored analysis commit.
/// The pass is gated by a rolling 7-day frequency limit — a pass runs only if
/// the previous pass started at least 7 days ago (timestamp recorded in
/// `config_kv` at pass start), so per-workspace regeneration happens at most
/// once per 7 days. Missed windows do not accumulate: the next eligible night
/// simply runs the pass (no fixed weekday pinning).
///
/// Workspaces are processed sequentially — each discovery must complete
/// before the next workspace is checked, avoiding resource spikes.
///
/// ## Edge-case handling
///
/// * **First run / no stored timestamp**: the gate lets the first pass
///   through; the pass-start timestamp is then recorded in `config_kv`.
/// * **Empty pass (all workspaces paused / no new commits)**: still counts as
///   a pass — the timestamp is written before any workspace is inspected, so
///   an unpause right after an empty pass waits for the next eligible night
///   (manual Reanalyze stays available and ungated).
/// * **Temp-dir cleaner**: each allowed pass also dispatches the periodic
///   temp-dir cleaner (`crate::temp`) fire-and-forget from inside
///   the gated block, so it inherits the at-most-once-per-7-days cadence
///   and is fully isolated from workspace processing.
/// * **Pass-start timestamp write failure**: fail-closed — the pass is
///   skipped so the at-most-once-per-7-days invariant holds; the next
///   eligible night retries.
/// * **Unparseable timestamp / config store read failure**: fail-open — the
///   pass runs and the pass-start write self-heals the stored value
///   (maintainer-debounce precedent).
/// * **Non-git workspaces / no commits**: `git rev-parse HEAD` fails,
///   these workspaces are skipped with a warning. No infinite re-discovery
///   loop because `last_analyzed_commit` stays NULL and git keeps failing.
/// * **Mid-window processing**: if the machine wakes at 2:55 and processing
///   extends past 3:00, all workspaces in the current batch are still
///   processed (the time window is checked once per 30-minute wake cycle).
///   No new processing starts in subsequent wake cycles outside the window.
/// * **Workspace path gone / git missing**: logged as a warning, the
///   workspace is skipped.
pub async fn run_nightly_check_loop() {
    let interval = Duration::from_mins(30);
    let shutdown = crate::shutdown::shutdown_token();

    loop {
        if !crate::shutdown::sleep_or_shutdown_or_drain(interval).await {
            break;
        }

        // Only proceed during the 2:00–3:00 AM local time window.
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

        // Periodic temp-dir cleaner (fire-and-forget, Sanitation role):
        // dispatched INSIDE the gated block so it inherits the pass cadence
        // (at most one cleaner per 7 days — the gate's pass-start timestamp
        // also dedups the two 30-min wakes in this window). Fully isolated
        // from the discovery pass: a dispatch failure only logs and the pass
        // proceeds (and vice versa — a pass failure never blocks the next
        // week's cleaner).
        if let Err(e) = crate::temp::dispatch_temp_cleanup().await {
            tracing::warn!(error = %e, "Nightly check: temp-dir cleaner dispatch failed — discovery pass continues");
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

            if ws.status != WorkspaceStatus::Ready || ws.paused {
                // Paused workspaces are skipped: a pause (including an automatic
                // technical-failure pause) is only lifted via the normal unpause
                // path — nightly rediscovery must not silently resume it.
                continue;
            }

            let repo_path = std::path::Path::new(&ws.path);

            // Run git rev-parse HEAD to get the current commit.
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

            // Compare with the stored hash.
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

            // Wait for discovery to complete before checking the next workspace.
            // Poll status every 10 seconds with a generous timeout (4 hours).
            let deadline = std::time::Instant::now() + Duration::from_hours(4);
            loop {
                if shutdown.is_cancelled() {
                    break;
                }
                if std::time::Instant::now() >= deadline {
                    tracing::warn!(
                        workspace = %ws.name,
                        "Nightly check: discovery timed out — proceeding to next workspace",
                    );
                    break;
                }

                tokio::time::sleep(Duration::from_secs(10)).await;

                match store.get_by_name(&ws.name).await {
                    Ok(Some(current)) if current.status != WorkspaceStatus::Analyzing => {
                        break;
                    }
                    Ok(_) => {} // Still analyzing — keep polling.
                    Err(e) => {
                        tracing::warn!(
                            workspace = %ws.name,
                            error = %e,
                            "Nightly check: failed to poll workspace status",
                        );
                        break;
                    }
                }
            }
        }
    }
}

// Test helpers

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

// Tests

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
                "INSERT INTO workspaces (name, path, created_at, updated_at, paused, maintenance, discovery_generation, diagnostics_generation) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
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
            maintainer_debounce_mins: 5,
            maintainer_last_run_at: None,
            diagnostics: None,
            notes: String::new(),
            last_analyzed_commit: None,
            ephemeral: false,
        }
    }

    // ── Schema / struct consistency ─────────────────────────────

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

        // Unpause
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

        // Re-pause
        store.set_paused("toggle_test", true).await.unwrap();
        let fetched = store
            .get_by_name("toggle_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(fetched.paused, "Should be paused after set_paused(true)");
    }

    #[tokio::test]
    async fn set_maintenance_toggles_maintenance_state() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "maint_test", "/tmp/maint_test", true, false, 0, 0).await;

        // Enable maintenance
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

        // Disable maintenance
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

        // Set notes and verify round-trip
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

        // Verify that updating notes works
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

        // Verify truncation at the cap
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

        // Verify UTF-8 safe truncation (multi-byte characters)
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
    async fn list_states_returns_name_paused_maintenance() {
        let (store, _tmp) = test_store().await;

        // Insert two workspaces with different toggle states.
        insert_direct(&store, "alice", "/tmp/alice", true, false, 0, 0).await;
        store.set_maintenance_enabled("alice", false).await.unwrap();

        insert_direct(&store, "bob", "/tmp/bob", false, false, 0, 0).await;
        store.set_maintenance_enabled("bob", true).await.unwrap();

        let states = store.list_states().await.expect("list_states");
        assert_eq!(states.len(), 2, "Should return both workspaces");

        // Build a map for assertion.
        let mut map: std::collections::HashMap<&str, (bool, bool)> =
            std::collections::HashMap::new();
        for (name, paused, maintenance_enabled) in &states {
            map.insert(name.as_str(), (*paused, *maintenance_enabled));
        }

        assert_eq!(
            map.get("alice").copied(),
            Some((true, false)),
            "Alice: paused=true, maintenance_enabled=false"
        );
        assert_eq!(
            map.get("bob").copied(),
            Some((false, true)),
            "Bob: paused=false, maintenance_enabled=true"
        );
    }

    // ── finalize_discovery — auto-unpause invariants ─────────────

    #[tokio::test]
    async fn finalize_discovery_success_auto_unpauses() {
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
                DiscoveryOutcome::AllOk,
                &[],
            )
            .await;

            let ws = store
                .get_by_name(suffix)
                .await
                .expect("fetch")
                .expect("exists");
            assert!(
                !ws.paused,
                "Should auto-unpause after discovery OK (gen {generation})"
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
            DiscoveryOutcome::Fatal,
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
            DiscoveryOutcome::Transient,
            &errors,
        )
        .await;

        let ws = store
            .get_by_name("prov_gen0")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(ws.paused, "Should remain paused (analysis pause)");
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
        finalize_discovery(
            &store,
            "prov_gen0",
            "/tmp/prov_gen0",
            0,
            DiscoveryOutcome::AllOk,
            &[],
        )
        .await;
        assert!(
            !pending_pickup_cooldown_active("prov_gen0"),
            "Successful discovery must clear the pickup cooldown"
        );
    }

    #[tokio::test]
    async fn finalize_discovery_stale_generation_skips_writes() {
        let (store, _tmp) = test_store().await;
        // Start paused with generation 0.
        insert_direct(&store, "stale", "/tmp/stale", true, false, 0, 0).await;

        // Bump the generation behind the scenes (simulates a concurrent
        // rediscover() call).
        store
            .exec_update_with_updated_at("discovery_generation = 1", vec![], "stale")
            .await
            .expect("bump generation");

        // Act: try to finalize with the stale generation 0.
        finalize_discovery(
            &store,
            "stale",
            "/tmp/stale",
            0,
            DiscoveryOutcome::AllOk,
            &[],
        )
        .await;

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

    #[tokio::test]
    async fn rediscover_sets_paused() {
        let (store, _tmp) = test_store().await;
        // Start with paused = false and status = ready (simulating a fully
        // discovered workspace).
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

        // Act: rediscover.
        store
            .rediscover("rediscover_test")
            .await
            .expect("rediscover");

        // Assert: paused is set immediately by the UPDATE.
        let ws = store
            .get_by_name("rediscover_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(
            ws.paused,
            "rediscover() must set paused = true when transitioning to 'analyzing'"
        );
    }

    // ── Integration: add() returns paused: true ──────────────────

    #[tokio::test]
    #[serial_test::serial(config_persist)] // swaps the process-global CONFIG
    async fn add_returns_paused_true() {
        let (store, _tmp) = test_store().await;
        let dir = TempDir::new().expect("temp dir for workspace path");

        // add() requires: search engine globals initialized + CONFIG storage
        // root set. Initialize the minimum globals (init_global is idempotent).
        crate::search_engine::init_global();
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
            ws.paused,
            "add() must return a Workspace with paused = true"
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
        assert_eq!(
            ws.maintainer_debounce_mins, 5,
            "add() must return a Workspace with maintainer_debounce_mins = 5"
        );

        // Also verify via get_by_name.
        let fetched = store
            .get_by_name("add_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(
            fetched.paused,
            "Persisted workspace must have paused = true"
        );
        assert_eq!(
            fetched.status,
            WorkspaceStatus::Pending,
            "Persisted workspace must have status = pending"
        );
    }

    // ── Pending-pickup: atomic claim + boot reclassification ──────

    #[tokio::test]
    async fn claim_pending_for_discovery_is_atomic_and_returns_live_generation() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "pickup", "/tmp/pickup", true, false, 3, 0).await;

        // Claim succeeds and returns the live generation counter.
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
        assert!(ws.paused, "claim must set the analysis pause");

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

    // ── Discovery failure classification taxonomy ─────────────────

    /// Build an agent with a preset typed failure class (simulating a failed
    /// run_agent) and classify it.
    fn classify_with(class: Option<crate::retry::FailureClass>) -> DiscoveryFailureKind {
        let ws = test_ws("/tmp/classify_ws");
        let mut agent = crate::Agent::new(
            "classify-test".into(),
            crate::Role::Discovery,
            &ws,
            None,
            String::new(),
            String::new(),
            false,
            None,
            None,
        );
        agent.failure_class = class;
        classify_discovery_failure(&agent, None)
    }

    #[test]
    fn discovery_failure_taxonomy_maps_provider_and_genuine_classes() {
        // Provider-class → Transient (workspace returns to Pending).
        for class in [
            crate::retry::FailureClass::Transport,
            crate::retry::FailureClass::TruncatedEnvelope,
            crate::retry::FailureClass::NoResponse,
            crate::retry::FailureClass::TruncatedOutput,
            crate::retry::FailureClass::NonRetryable, // auth / quota / invalid model
            crate::retry::FailureClass::WallClockExceeded,
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

    #[tokio::test]
    async fn pending_pickup_cooldown_arms_clears_and_expires() {
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
    }

    #[test]
    fn pending_pickup_cooldown_escalates_and_resets() {
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
    async fn editor_tabs_round_trip_dirty_content() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "ws1", "/tmp/ws1", false, false, 0, 0).await;

        let tabs = vec![EditorTabRecord {
            file_path: "notes.md".to_string(),
            tab_order: 0,
            is_active: true,
            is_dirty: true,
            dirty_content: Some("draft text".to_string()),
        }];
        store
            .save_editor_tabs("ws1", &tabs)
            .await
            .expect("save tabs");

        let loaded = store.load_editor_tabs("ws1").await.expect("load tabs");
        assert_eq!(loaded.len(), 1);
        assert!(loaded[0].is_active);
        assert!(loaded[0].is_dirty);
        assert_eq!(loaded[0].dirty_content.as_deref(), Some("draft text"));
    }

    // ── Diagnostics API tests ─────────────────────────────────────

    #[tokio::test]
    async fn set_diagnostics_roundtrip() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "diag_test", "/tmp/diag_test", false, false, 0, 0).await;

        let cmds = crate::DiagnosticsCommands {
            format: Some("cargo fmt".into()),
            format_check: Some("cargo fmt -- --check".into()),
            lint: Some("cargo clippy -- -D warnings".into()),
            ..Default::default()
        };
        store
            .set_diagnostics("diag_test", &cmds)
            .await
            .expect("set_diagnostics");

        let loaded = store
            .get_diagnostics("diag_test")
            .await
            .expect("get_diagnostics")
            .expect("should have diagnostics");
        assert_eq!(loaded.format.as_deref(), Some("cargo fmt"));
        assert_eq!(loaded.format_check.as_deref(), Some("cargo fmt -- --check"));
        assert_eq!(loaded.lint.as_deref(), Some("cargo clippy -- -D warnings"));
        assert!(loaded.lint_fix.is_none());
        assert!(loaded.type_check.is_none());
        assert!(loaded.build.is_none());
        assert!(loaded.unit_test.is_none());
    }

    #[tokio::test]
    async fn get_generation_reads_diagnostics_column() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "gen_test", "/tmp/gen_test", false, false, 5, 3).await;

        let diag_gen_val = store
            .get_generation("gen_test", GenerationColumn::DIAGNOSTICS)
            .await
            .expect("get_generation");
        assert_eq!(
            diag_gen_val, 3,
            "Should return the stored diagnostics_generation"
        );
    }

    #[tokio::test]
    async fn set_diagnostics_bumps_diagnostics_generation() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "bump_test", "/tmp/bump_test", false, false, 0, 0).await;

        let cmds = crate::DiagnosticsCommands::default();
        store
            .set_diagnostics("bump_test", &cmds)
            .await
            .expect("set_diagnostics");

        let diag_gen_val = store
            .get_generation("bump_test", GenerationColumn::DIAGNOSTICS)
            .await
            .expect("get_generation");
        assert_eq!(
            diag_gen_val, 1,
            "set_diagnostics should bump diagnostics_generation to 1"
        );
    }

    #[tokio::test]
    async fn rediscover_diagnostics_clears_and_bumps() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "redia_test", "/tmp/redia_test", false, false, 0, 0).await;

        // Set some diagnostics first.
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

        // Verify generation before rediscover.
        let diag_gen_before = store
            .get_generation("redia_test", GenerationColumn::DIAGNOSTICS)
            .await
            .expect("get_generation");
        assert_eq!(diag_gen_before, 1, "Should be 1 after set_diagnostics");

        // Act: rediscover diagnostics (doesn't spawn real agent, just bumps and clears).
        store
            .rediscover_diagnostics("redia_test")
            .await
            .expect("rediscover_diagnostics");

        // Diagnostics should now be None.
        assert!(
            store
                .get_diagnostics("redia_test")
                .await
                .expect("get_diagnostics")
                .is_none(),
            "rediscover_diagnostics should clear diagnostics"
        );

        // Generation should have been bumped.
        let diag_gen_after = store
            .get_generation("redia_test", GenerationColumn::DIAGNOSTICS)
            .await
            .expect("get_generation");
        assert_eq!(
            diag_gen_after, 2,
            "rediscover_diagnostics should bump diagnostics_generation to 2"
        );

        // discovery_generation should NOT have been touched.
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
    async fn diagnostics_generation_guard_skips_stale_writes() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "diag_stale", "/tmp/diag_stale", true, false, 0, 0).await;

        // Set some initial diagnostics (this bumps gen to 1).
        let cmds = crate::DiagnosticsCommands {
            format: Some("cargo fmt".into()),
            ..Default::default()
        };
        store
            .set_diagnostics("diag_stale", &cmds)
            .await
            .expect("set_diagnostics");

        // Bump the diagnostics_generation behind the scenes (simulates a concurrent
        // rediscover_diagnostics() or set_diagnostics() call).
        store
            .conn
            .execute(
                "UPDATE workspaces SET diagnostics_generation = 99 WHERE name = ?1",
                crate::db::params!["diag_stale"],
            )
            .await
            .expect("bump diagnostics_generation");

        // Capture the stale generation (1) and verify the guard catches it.
        let stale_gen_val = 1;
        let is_ok = check_generation(
            &store,
            "diag_stale",
            stale_gen_val,
            GenerationColumn::DIAGNOSTICS,
            "test",
        )
        .await;
        assert!(!is_ok, "check_generation should reject stale generation");

        // Fresh generation should pass.
        let fresh_gen_val = 99;
        let is_ok = check_generation(
            &store,
            "diag_stale",
            fresh_gen_val,
            GenerationColumn::DIAGNOSTICS,
            "test",
        )
        .await;
        assert!(is_ok, "check_generation should accept fresh generation");
    }

    // ── General (non-role) context tests ──────────────────────────

    #[tokio::test]
    async fn general_context_roundtrip_single_row_per_workspace() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "gctx", "/tmp/gctx", true, false, 0, 0).await;

        // No stored general context before discovery writes one.
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

    // ── is_nightly_check_hour — time-window boundary tests ────────

    #[test]
    fn nightly_check_hour_window_boundaries() {
        let cases: &[(u32, bool, &str)] = &[
            (1, false, "1:00 AM is before the window"),
            (2, true, "2:00 AM is the start of the window"),
            (3, false, "3:00 AM is excluded from the window"),
            (4, false, "4:00 AM is after the window"),
            (0, false, "Midnight is outside the window"),
            (12, false, "Noon is outside the window"),
            (23, false, "11 PM is outside the window"),
        ];
        for (hour, expected, msg) in cases {
            assert_eq!(is_nightly_check_hour(*hour), *expected, "{msg}");
        }
    }

    // ── nightly_gate_allows — rolling 7-day frequency gate tests ──

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

    // ── nightly_gate_should_run — end-to-end config_kv behaviour ──

    #[tokio::test]
    async fn nightly_gate_records_pass_start_and_blocks() {
        let (store, _tmp) = crate::open_test_store!(crate::config_db::ConfigStore, "config");

        // First pass: no stored timestamp → allowed, pass start recorded.
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

    // ── has_new_commits — commit comparison tests ────────────────

    #[test]
    fn new_commits_null_stored_triggers_rediscovery() {
        assert!(
            has_new_commits(None, "abc123"),
            "NULL last_analyzed_commit should trigger rediscovery",
        );
    }

    #[test]
    fn new_commits_matching_hash_skips() {
        assert!(
            !has_new_commits(Some("abc123"), "abc123"),
            "Same hash should not trigger rediscovery",
        );
    }

    #[test]
    fn new_commits_different_hash_triggers() {
        assert!(
            has_new_commits(Some("abc123"), "def456"),
            "Different hash should trigger rediscovery",
        );
    }

    #[test]
    fn new_commits_empty_current_hash_triggers() {
        // Edge case: current_hash is empty (shouldn't happen from git
        // rev-parse HEAD, but the function handles it gracefully).
        assert!(
            has_new_commits(Some("abc123"), ""),
            "Empty current hash should trigger rediscovery",
        );
    }
}
