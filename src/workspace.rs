//! Workspace storage — persisted workspace metadata and contexts.
//!
//! Also handles workspace analysis: spawning a Discovery agent to explore a new
//! workspace and produce role-specific context summaries.

use crate::Role;
use crate::Workspace;
use crate::WorkspaceStatus;
use crate::agent::run_agent;
use crate::session::discovery_session_key;
use crate::turso::{self};
use anyhow::{Context, Result};
use futures_util::future::join_all;
use strum::IntoEnumIterator;
use tracing::warn;

crate::define_store! {
    /// Global workspace store.
    pub(crate) static WORKSPACES: WorkspaceStore,
    db_name = "workspaces",
    schema = SCHEMA,
    post_open = after_open,
    expect = "workspace::WORKSPACES not initialized — call workspace::init_global() in main.rs",
}

/// Look up a workspace by its name.
pub async fn get_by_name(name: &str) -> Result<Option<Workspace>> {
    store().get_by_name(name).await
}

const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS workspaces (
    name       TEXT PRIMARY KEY,
    path       TEXT NOT NULL UNIQUE,
    status     TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    maintenance INTEGER NOT NULL DEFAULT 0,
    paused      INTEGER NOT NULL DEFAULT 1,
    maintainer_debounce_mins INTEGER NOT NULL DEFAULT 5,
    maintainer_last_run_at TEXT,
    diagnostics TEXT,
    diagnostics_updated_at TEXT,
    diagnostics_generation INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    discovery_generation INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS workspace_contexts (
    workspace_name TEXT NOT NULL REFERENCES workspaces(name) ON DELETE CASCADE,
    role           TEXT NOT NULL,
    content        TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    UNIQUE(workspace_name, role)
);
CREATE TABLE IF NOT EXISTS editor_tabs (
    workspace_name TEXT NOT NULL REFERENCES workspaces(name) ON DELETE CASCADE,
    file_path      TEXT NOT NULL,
    tab_order      INTEGER NOT NULL DEFAULT 0,
    is_active      INTEGER NOT NULL DEFAULT 0,
    is_dirty       INTEGER NOT NULL DEFAULT 0,
    dirty_content  TEXT,
    PRIMARY KEY (workspace_name, file_path)
);";

// Column definitions for workspace SELECT queries.
// Note: `discovery_generation` is intentionally excluded from this column list:
// it is read only via its own single-column SELECT in
// [`WorkspaceStore::get_discovery_generation`] and is never part of a workspace struct query.
crate::columns! {
    WORKSPACE_COLUMNS [WS] {
        NAME                  => "name",
        PATH                  => "path",
        STATUS                => "status",
        CREATED_AT            => "created_at",
        UPDATED_AT            => "updated_at",
        MAINTENANCE_ENABLED    => "maintenance",
        PAUSED                => "paused",
        MAINTAINER_DEBOUNCE_MINS => "maintainer_debounce_mins",
        MAINTAINER_LAST_RUN_AT  => "maintainer_last_run_at",
        DIAGNOSTICS           => "diagnostics",
        DIAGNOSTICS_UPDATED_AT => "diagnostics_updated_at",
        NOTES                  => "notes",
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

/// Check the discovery generation counter: return `true` if the calling task
/// is still the latest (OK to proceed), `false` if a newer [`WorkspaceStore::rediscover`] has
/// been triggered (stale — do not proceed).
async fn check_discovery_generation(
    storage: &WorkspaceStore,
    workspace_name: &str,
    discovery_generation: i64,
    label: &str,
) -> bool {
    let current_gen = storage
        .get_discovery_generation(workspace_name)
        .await
        .unwrap_or(discovery_generation + 1);
    if current_gen != discovery_generation {
        tracing::warn!(
            workspace_name,
            captured_gen = discovery_generation,
            current_gen,
            label = %label,
            "Discovery generation mismatch — skipping stale write"
        );
        return false;
    }
    true
}

/// Check the diagnostics generation counter: return `true` if the calling task
/// is still the latest (OK to proceed), `false` if a newer
/// [`WorkspaceStore::rediscover_diagnostics`] or [`WorkspaceStore::set_diagnostics`]
/// has been triggered (stale — do not proceed).
async fn check_diagnostics_generation(
    storage: &WorkspaceStore,
    workspace_name: &str,
    diagnostics_generation: i64,
    label: &str,
) -> bool {
    let current_gen = storage
        .get_diagnostics_generation(workspace_name)
        .await
        .unwrap_or(diagnostics_generation + 1);
    if current_gen != diagnostics_generation {
        tracing::warn!(
            workspace_name,
            captured_gen = diagnostics_generation,
            current_gen,
            label = %label,
            "Diagnostics generation mismatch — skipping stale write"
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
/// Returns `Ok(())` on success, or an error describing what went wrong.
async fn run_workspace_discovery(
    ws: &Workspace,
    role: Role,
    discovery_generation: i64,
) -> Result<()> {
    let storage = WORKSPACES
        .get()
        .context("WORKSPACES not initialized")?
        .clone();

    tracing::info!(workspace_name = ws.name, role = %role, "Starting workspace discovery");

    // Create a Discovery agent pointed at the workspace
    let agent_id = discovery_session_key(&ws.name, role.as_str());
    let prompt = role.discovery_prompt();
    let (_agent, response) = run_agent(agent_id, Role::Discovery, ws, None, &prompt).await;
    let response =
        response.context("Discovery agent returned no response (cancelled or failed)")?;

    let content = response.trim().to_string();
    if content.is_empty() {
        anyhow::bail!("Empty response for role '{role}'");
    }

    // Guard against stale writes: if another rediscover has been triggered
    // while this discovery ran, skip the context write.
    if !check_discovery_generation(&storage, &ws.name, discovery_generation, "context").await {
        return Ok(());
    }

    if let Err(e) = storage.set_context(&ws.name, role.as_str(), &content).await {
        tracing::error!(workspace_name = ws.name, role = %role, error = %e, "Failed to store context");
        return Err(e);
    }

    tracing::info!(workspace_name = ws.name, role = %role, "Workspace discovery for {role} completed");
    Ok(())
}

/// Run diagnostics discovery — scan the workspace for dev tooling commands.
///
/// Runs a Discovery agent (using `Role::Discovery`'s tools: shell, read, search)
/// to scan build files and identify commands for format, lint, type-check, build,
/// and unit-test categories. Extracts structured output via [`crate::extraction::retry_extract_structured`].
///
/// `diagnostics_generation` guards against stale writes — if a newer
/// [`WorkspaceStore::rediscover_diagnostics`] or [`WorkspaceStore::set_diagnostics`]
/// was triggered while diagnostics were being computed, the write is skipped.
///
/// On failure, existing diagnostics data is left untouched.
async fn run_workspace_diagnostics(ws: &Workspace, diagnostics_generation: i64) -> Result<()> {
    let storage = WORKSPACES
        .get()
        .context("WORKSPACES not initialized")?
        .clone();

    tracing::info!(workspace_name = ws.name, "Starting diagnostics discovery");

    let agent_id = discovery_session_key(&ws.name, "diagnostics");

    // Load the diagnostics discovery prompt directly (not a role-specific discovery prompt).
    let prompt = crate::prompt::load_prompt("discovery/diagnostics.md");

    let (agent, response) = run_agent(agent_id, Role::Discovery, ws, None, &prompt).await;
    response.context("Diagnostics discovery agent returned no response (cancelled or failed)")?;

    // Keep the Agent alive after run_agent() for retry_extract_structured —
    // it needs agent.session.history() and agent.tool_specs.
    let extraction_prompt = crate::prompt::load_prompt("extraction/diagnostics.md");

    // KV-cache preservation: `agent.extract_structured` uses the agent's own
    // parameters (model, temperature, reasoning_effort, tools, provider routing)
    // so the extraction call is byte-identical to the original Discovery agent
    // call — the provider can reuse the cached prefix.
    let cmds: crate::DiagnosticsCommands = agent.extract_structured(&extraction_prompt, 3).await?;

    // Guard against stale writes — uses diagnostics_generation column.
    if !check_diagnostics_generation(&storage, &ws.name, diagnostics_generation, "diagnostics")
        .await
    {
        return Ok(());
    }

    let now = turso::now();
    storage.set_diagnostics(&ws.name, &cmds, &now).await?;

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
/// - If `all_ok`: sets status to `ready` and always unpauses the workspace.
///   A successful discovery — whether initial or rediscovery — brings the
///   workspace back to life.
/// - If **not** `all_ok`: sets status to `failed` and leaves `paused` untouched.
/// - Before any write, checks the generation guard: if a newer [`WorkspaceStore::rediscover`]
///   bumped the generation while discovery was in flight, the writes are skipped.
async fn finalize_discovery(
    storage: &WorkspaceStore,
    ws_name: &str,
    discovery_generation: i64,
    all_ok: bool,
    errors: &[String],
) {
    // Final guard: if a newer rediscover was triggered while this task ran,
    // all three write sites (contexts, diagnostics, status) have already been
    // individually guarded.  This check catches the status write.
    if !check_discovery_generation(storage, ws_name, discovery_generation, "final status").await {
        return;
    }

    if all_ok {
        // Single atomic UPDATE for both status and paused columns, following
        // the same pattern used by WorkspaceStore::rediscover.
        let _ = storage
            .conn
            .execute(
                "UPDATE workspaces SET status = 'ready', paused = 0, updated_at = ? WHERE name = ?",
                turso::params![turso::now(), ws_name],
            )
            .await;
        tracing::info!(workspace = ws_name, "Workspace pipeline resumed");
        tracing::info!(
            workspace_name = ws_name,
            "Workspace analysis complete — all roles ready"
        );
    } else {
        let msg = errors.join("; ");
        let _ = storage.set_status(ws_name, &WorkspaceStatus::Failed).await;
        tracing::warn!(workspace_name = ws_name, error = %msg, "Workspace analysis failed");
    }
}

/// Spawn a background task that runs workspace discovery (per-role context
/// generation) and optionally diagnostics discovery.
///
/// `discovery_generation` is the generation counter captured at spawn time.
/// Both discovery functions use it to guard against stale writes.
///
/// When `discover_diagnostics` is `false` (e.g. during a re-analysis via
/// [`WorkspaceStore::rediscover`]), diagnostics discovery is skipped so that
/// user-managed diagnostics survive re-analysis.
pub fn spawn_workspace_discovery(
    ws: &Workspace,
    discovery_generation: i64,
    discover_diagnostics: bool,
) {
    let ws = ws.clone();
    tokio::spawn(async move {
        let ws_name = ws.name.clone();

        // Run the discovery body in a sub-task so that panics are captured
        // via the JoinHandle instead of being silently swallowed.
        //
        // NOTE: Unlike the ticket-dispatch panic recovery (which transitions
        // the ticket to Failed), this guard only logs and does NOT transition
        // the workspace to "failed". Non-prompt panics will leave the workspace
        // in "analyzing" — visible in logs but not recovered.
        let ws_name_for_finalize = ws_name.clone();
        let ws_name_for_inner = ws_name.clone();
        let inner = tokio::spawn(async move {
            // Build role discovery futures (always needed).
            let role_futures: Vec<_> = Role::iter()
                .filter(|r| crate::role::role_info(r).has_discovery)
                .map(|role| {
                    let ws = ws.clone();
                    async move { run_workspace_discovery(&ws, role, discovery_generation).await }
                })
                .collect();

            // Run role discovery always, optionally with diagnostics.
            let (role_results, diagnostics_result) = if discover_diagnostics {
                // Read diagnostics generation from DB for the generation guard.
                // Separate from discovery_generation: both counters are independent
                // (diagnostics is bumped by set_diagnostics/rediscover_diagnostics,
                // discovery is bumped by rediscover).
                let diag_gen = match WORKSPACES.get() {
                    Some(s) => s
                        .get_diagnostics_generation(&ws_name_for_inner)
                        .await
                        .unwrap_or(0),
                    None => 0,
                };
                tokio::join!(
                    join_all(role_futures),
                    run_workspace_diagnostics(&ws, diag_gen),
                )
            } else {
                let roles = join_all(role_futures).await;
                (roles, Ok(()))
            };

            let mut all_ok = true;
            let mut errors: Vec<String> = Vec::new();

            for result in role_results {
                match result {
                    Ok(()) => {}
                    Err(e) => {
                        all_ok = false;
                        errors.push(e.to_string());
                    }
                }
            }

            // Diagnostics failure is fatal.
            if let Err(e) = diagnostics_result {
                all_ok = false;
                errors.push(format!("Diagnostics discovery failed: {e}"));
            }

            let Some(storage) = WORKSPACES.get() else {
                tracing::error!("WORKSPACES not initialized during final status update");
                return;
            };

            finalize_discovery(
                storage,
                &ws_name_for_finalize,
                discovery_generation,
                all_ok,
                &errors,
            )
            .await;
        });

        match inner.await {
            Ok(()) => {}
            Err(e) => {
                let kind = if e.is_panic() { "panic" } else { "cancelled" };
                tracing::error!(
                    workspace_name = %ws_name,
                    kind = kind,
                    error = %e,
                    "spawn_workspace_discovery task failed",
                );
            }
        }
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
/// [`check_diagnostics_generation`].
pub fn spawn_diagnostics_discovery(ws: &Workspace, diagnostics_generation: i64) {
    let ws = ws.clone();
    tokio::spawn(async move {
        let ws_name = ws.name.clone();

        // Use the same panic-catching pattern as spawn_workspace_discovery.
        let inner = tokio::spawn(async move {
            if let Err(e) = run_workspace_diagnostics(&ws, diagnostics_generation).await {
                tracing::error!(
                    workspace_name = %ws.name,
                    error = %e,
                    "Diagnostics rediscovery failed",
                );
            }
        });

        match inner.await {
            Ok(()) => {}
            Err(e) => {
                let kind = if e.is_panic() { "panic" } else { "cancelled" };
                tracing::error!(
                    workspace_name = %ws_name,
                    kind = kind,
                    error = %e,
                    "spawn_diagnostics_discovery task failed",
                );
            }
        }
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

fn workspace_from_row(row: &turso::Row) -> anyhow::Result<Workspace> {
    Ok(Workspace {
        name: row.get(COL_WS_NAME)?,
        path: row.get(COL_WS_PATH)?,
        status: row
            .get::<String>(COL_WS_STATUS)?
            .parse::<WorkspaceStatus>()?,
        created_at: row.get(COL_WS_CREATED_AT)?,
        updated_at: row.get(COL_WS_UPDATED_AT)?,
        maintenance_enabled: row.get::<bool>(COL_WS_MAINTENANCE_ENABLED)?,
        paused: row.get::<bool>(COL_WS_PAUSED)?,
        maintainer_debounce_mins: row.get::<i64>(COL_WS_MAINTAINER_DEBOUNCE_MINS)?,
        maintainer_last_run_at: row.get::<Option<String>>(COL_WS_MAINTAINER_LAST_RUN_AT)?,
        diagnostics: row.get::<Option<String>>(COL_WS_DIAGNOSTICS)?,
        diagnostics_updated_at: row.get::<Option<String>>(COL_WS_DIAGNOSTICS_UPDATED_AT)?,
        notes: row.get::<String>(COL_WS_NOTES)?,
    })
}

impl WorkspaceStore {
    /// Run a query that returns zero-or-one workspace row, mapping the result to
    /// `Ok(Some(ws))` / `Ok(None)` / `Err`.
    async fn query_one(
        &self,
        where_clause: &str,
        params: impl turso::IntoParams + Send + 'static,
    ) -> Result<Option<Workspace>> {
        let sql = format!("SELECT {WORKSPACE_COLUMNS} FROM workspaces WHERE {where_clause}");
        self.conn
            .query_optional(&sql, params, workspace_from_row)
            .await
    }

    /// Insert a new workspace and kick off analysis.
    pub async fn add(&self, name: &str, path: &str) -> Result<Workspace> {
        // Validate the workspace name.
        validate_name(name)?;

        // Canonicalize and validate the path so bad paths never enter the system.
        let canonical = canonicalize_workspace_path(path).map_err(|e| anyhow::anyhow!("{e}"))?;
        let path = ensure_trailing_slash(&canonical);
        let now = turso::now();
        self.conn
            .execute(
                "INSERT INTO workspaces (name, path, created_at, updated_at, paused) VALUES (?1, ?2, ?3, ?4, ?5)",
                turso::params![name, path.clone(), now.clone(), now.clone(), 1],
            )
            .await?;
        let ws = Workspace {
            name: name.to_string(),
            path: path.clone(),
            status: WorkspaceStatus::Pending,
            created_at: now.clone(),
            updated_at: now.clone(),
            maintenance_enabled: false,
            paused: true,
            maintainer_debounce_mins: 5,
            maintainer_last_run_at: None,
            diagnostics: None,
            diagnostics_updated_at: None,
            notes: String::new(),
        };
        let _ = self.set_status(name, &WorkspaceStatus::Analyzing).await;
        // New workspace: discovery_generation defaults to 0 in the schema.
        // Generation 0 means "the first discovery" — if rediscover() bumps
        // the generation before this task finishes, the task's context/
        // diagnostics/status writes will be skipped by the generation guard.
        spawn_workspace_discovery(&ws, 0, true);
        // Eagerly initialize the shared search engine for this workspace.
        if let Err(e) =
            crate::search_engine::get_or_init_engine(name, std::path::Path::new(&ws.path))
        {
            tracing::warn!(workspace_name = name, error = %e, "Failed to init search engine on workspace add");
        }
        Ok(ws)
    }

    /// List all workspaces ordered by name.
    pub async fn list(&self) -> Result<Vec<Workspace>> {
        let rows = self
            .conn
            .query_map(
                &format!("SELECT {WORKSPACE_COLUMNS} FROM workspaces ORDER BY name"),
                turso::params![],
                workspace_from_row,
            )
            .await?;
        Ok(rows
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Lightweight fetch of only name, paused, and maintenance_enabled columns.
    /// Used by the GUI sidebar's periodic state refresh — avoids fetching
    /// all workspace columns when only toggle state is needed.
    pub async fn list_states(&self) -> Result<Vec<(String, bool, bool)>> {
        let rows = self
            .conn
            .query(
                &format!("SELECT {WS_STATE_COLUMNS} FROM workspaces ORDER BY name"),
                turso::params![],
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
        self.query_one("name = ?1", turso::params![name]).await
    }

    /// Delete a workspace by name. Context rows are cascaded automatically.
    /// The associated search engine is also removed from the in-memory registry.
    pub async fn delete(&self, name: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM workspaces WHERE name = ?1",
                turso::params![name],
            )
            .await?;
        crate::search_engine::remove_engine(name);
        Ok(())
    }

    /// Update the status of a workspace.
    pub async fn set_status(&self, name: &str, status: &WorkspaceStatus) -> Result<()> {
        let now = turso::now();
        self.conn
            .execute(
                "UPDATE workspaces SET status = ?1, updated_at = ?2 WHERE name = ?3",
                turso::params![status.to_string(), now.clone(), name],
            )
            .await?;
        Ok(())
    }

    /// Set or clear the maintenance toggle for a workspace.
    pub async fn set_maintenance_enabled(&self, name: &str, enabled: bool) -> Result<()> {
        let now = turso::now();
        let val: i64 = i64::from(enabled);
        if enabled {
            // Reset debounce state so the maintainer runs on the very next
            // 1-minute poll cycle (last_run_at = NULL bypasses the debounce
            // gate), regardless of how long the previous interval was.
            self.conn
                .execute(
                    "UPDATE workspaces SET maintenance = ?1, maintainer_debounce_mins = 5, maintainer_last_run_at = NULL, updated_at = ?2 WHERE name = ?3",
                    turso::params![val, now, name],
                )
                .await?;
        } else {
            self.conn
                .execute(
                    "UPDATE workspaces SET maintenance = ?1, updated_at = ?2 WHERE name = ?3",
                    turso::params![val, now, name],
                )
                .await?;
            // Cancel any running maintainer agent for this workspace so it
            // doesn't continue creating tickets after maintenance was disabled.
            if let Some(ws) = self.get_by_name(name).await? {
                crate::registry::AGENT_REGISTRY
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
        let now = turso::now();
        let val: i64 = i64::from(paused);
        self.conn
            .execute(
                "UPDATE workspaces SET paused = ?1, updated_at = ?2 WHERE name = ?3",
                turso::params![val, now, name],
            )
            .await?;
        if paused {
            tracing::info!(workspace = name, "Workspace pipeline paused");
        } else {
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
        let now = turso::now();
        self.conn
            .execute(
                "UPDATE workspaces SET maintainer_debounce_mins = ?1, maintainer_last_run_at = ?2, updated_at = ?3 WHERE name = ?4",
                turso::params![debounce_mins, last_run_at, now.clone(), name],
            )
            .await?;
        Ok(())
    }

    /// Store discovered diagnostics commands for a workspace.
    ///
    /// Also bumps `diagnostics_generation` so any in-flight diagnostics
    /// discovery task will see a generation mismatch and skip its stale write
    /// (see [`check_diagnostics_generation`]).
    pub async fn set_diagnostics(
        &self,
        name: &str,
        commands: &crate::DiagnosticsCommands,
        timestamp: &str,
    ) -> Result<()> {
        let json = serde_json::to_string(commands)?;
        self.conn
            .execute(
                "UPDATE workspaces SET diagnostics = ?1, diagnostics_updated_at = ?2, diagnostics_generation = diagnostics_generation + 1, updated_at = ?3 WHERE name = ?4",
                turso::params![json, timestamp, turso::now(), name],
            )
            .await?;
        Ok(())
    }

    /// Retrieve discovered diagnostics commands for a workspace.
    pub async fn get_diagnostics(&self, name: &str) -> Result<Option<crate::DiagnosticsCommands>> {
        match self
            .conn
            .query_row(
                "SELECT diagnostics FROM workspaces WHERE name = ?1",
                turso::params![name],
                |row| row.get::<Option<String>>(0),
            )
            .await
        {
            Ok(Some(json)) => {
                let cmds: crate::DiagnosticsCommands = serde_json::from_str(&json)?;
                Ok(Some(cmds))
            }
            Ok(None) | Err(::turso::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Set freeform user-curated context notes for a workspace.
    ///
    /// Truncates to 4000 characters (safe UTF-8) as defense-in-depth against
    /// prompt bloat. Notes are appended to every agent's system prompt.
    pub async fn set_notes(&self, name: &str, notes: &str) -> Result<()> {
        // Safe UTF-8 char-level truncation — must not use byte slicing
        // which would panic on multi-byte characters at the boundary.
        let notes: String = notes.chars().take(4000).collect();
        self.conn
            .execute(
                "UPDATE workspaces SET notes = ?1, updated_at = ?2 WHERE name = ?3",
                turso::params![notes, turso::now(), name],
            )
            .await?;
        Ok(())
    }

    /// Clear all per-role workspace contexts for a workspace.
    /// Called by [`Self::rediscover`] before spawning a new discovery task so that
    /// stale context entries from a previous discovery don't persist.
    async fn clear_contexts(&self, name: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM workspace_contexts WHERE workspace_name = ?1",
                turso::params![name],
            )
            .await?;
        Ok(())
    }

    /// Read the current `discovery_generation` for a workspace.
    ///
    /// Used by discovery tasks to check whether a newer rediscover has been
    /// triggered — if the generation no longer matches, the task's writes
    /// are stale and must be skipped.
    async fn get_discovery_generation(&self, name: &str) -> Result<i64> {
        self.conn
            .query_row(
                "SELECT discovery_generation FROM workspaces WHERE name = ?1",
                turso::params![name],
                |row| row.get(0),
            )
            .await
            .map_err(Into::into)
    }

    /// Read the current `diagnostics_generation` for a workspace.
    ///
    /// Used by diagnostics discovery tasks to check whether a newer
    /// [`WorkspaceStore::rediscover_diagnostics`] or [`WorkspaceStore::set_diagnostics`]
    /// has been triggered — if the generation no longer matches, the task's
    /// writes are stale and must be skipped.
    async fn get_diagnostics_generation(&self, name: &str) -> Result<i64> {
        self.conn
            .query_row(
                "SELECT diagnostics_generation FROM workspaces WHERE name = ?1",
                turso::params![name],
                |row| row.get(0),
            )
            .await
            .map_err(Into::into)
    }

    /// Trigger re-analysis of an existing workspace.
    /// Resets status to "analyzing", clears stale per-role contexts, and
    /// spawns analysis with a fresh generation counter.
    ///
    /// Unlike [`Self::rediscover_diagnostics`], this does **not** clear or
    /// re-discover diagnostics — user-managed diagnostics survive re-analysis.
    pub async fn rediscover(&self, name: &str) -> Result<()> {
        let ws = self
            .get_by_name(name)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Workspace {name} not found"))?;

        let now = turso::now();

        // Atomically increment the discovery generation counter so any
        // still-running discovery task from a previous rediscover will
        // see a generation mismatch and skip its writes.
        // NOTE: diagnostics is deliberately NOT cleared — user-managed
        // diagnostics survive re-analysis.
        self.conn
            .execute(
                "UPDATE workspaces SET discovery_generation = discovery_generation + 1, status = 'analyzing', paused = 1, updated_at = ?1 WHERE name = ?2",
                turso::params![now, name],
            )
            .await?;

        // Clear stale per-role context entries so that old discovery tasks
        // that beat the generation check cannot leave partial data behind.
        self.clear_contexts(name).await?;

        let generation = self.get_discovery_generation(name).await?;
        // Skip diagnostics discovery — see Part 1 of mahbot-726.
        spawn_workspace_discovery(&ws, generation, false);

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
    /// paused state, per-role contexts, or [`Self::discovery_generation`].
    pub async fn rediscover_diagnostics(&self, name: &str) -> Result<()> {
        let ws = self
            .get_by_name(name)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Workspace {name} not found"))?;

        let now = turso::now();

        // Bump diagnostics_generation and clear diagnostics so the fresh
        // discovery starts from scratch. The generation bump also cancels
        // any in-flight diagnostics discovery tasks. Does NOT touch
        // discovery_generation — role discovery is unaffected.
        self.conn
            .execute(
                "UPDATE workspaces SET diagnostics_generation = diagnostics_generation + 1, diagnostics = NULL, diagnostics_updated_at = NULL, updated_at = ?1 WHERE name = ?2",
                turso::params![now, name],
            )
            .await?;

        let generation = self.get_diagnostics_generation(name).await?;
        spawn_diagnostics_discovery(&ws, generation);

        Ok(())
    }

    /// Get a single context entry by workspace name and role.
    pub async fn get_context(&self, name: &str, role: &str) -> Result<Option<String>> {
        self.conn
            .query_optional(
                "SELECT content FROM workspace_contexts WHERE workspace_name = ?1 AND role = ?2",
                turso::params![name, role],
                |row| row.get::<String>(0),
            )
            .await
    }

    /// Upsert a single context entry for a workspace and role.
    pub async fn set_context(&self, name: &str, role: &str, content: &str) -> Result<()> {
        let now = turso::now();
        self.conn
            .execute(
                "INSERT INTO workspace_contexts (workspace_name, role, content, created_at) VALUES (?1, ?2, ?3, ?4) \
                 ON CONFLICT(workspace_name, role) DO UPDATE SET content = excluded.content, created_at = excluded.created_at",
                turso::params![name, role, content, now],
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
            turso::params![name],
        )
        .await?;
        for tab in tabs {
            tx.execute(
                "INSERT INTO editor_tabs (workspace_name, file_path, tab_order, is_active, is_dirty, dirty_content) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                turso::params![
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
                turso::params![name],
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

impl WorkspaceStore {
    /// Post-open setup: run schema migrations for the workspaces table.
    async fn after_open(&self) -> anyhow::Result<()> {
        run_workspace_migrations(&self.conn).await
    }
}

/// Run schema migrations for the `workspaces` table.
///
/// Currently adds the `notes` column if missing (existing databases).
/// Uses a `PRAGMA table_info` existence check for idempotency rather than
/// `PRAGMA user_version` versioning, because there is only a single
/// migration and no future migrations are planned for this column.
async fn run_workspace_migrations(conn: &turso::Connection) -> anyhow::Result<()> {
    let table_info = conn
        .query("PRAGMA table_info('workspaces')", ())
        .await
        .context("Failed to read PRAGMA table_info for workspaces table")?;

    let has_notes = table_info
        .iter()
        .any(|row| row.get::<String>(1).ok().as_deref() == Some("notes"));

    if !has_notes {
        tracing::info!("Schema migration: adding workspaces.notes column");
        conn.execute(
            "ALTER TABLE workspaces ADD COLUMN notes TEXT NOT NULL DEFAULT ''",
            (),
        )
        .await
        .context("Schema migration failed: unable to add workspaces.notes")?;
        conn.checkpoint().await.context(
            "Schema migration failed: unable to checkpoint after adding workspaces.notes",
        )?;
        tracing::info!("Schema migration complete: added workspaces.notes column");
    }

    // Migration: add diagnostics_generation column.
    // This column is used by the generation-guard mechanism to protect user-edited
    // diagnostics from being overwritten by stale discovery tasks.
    let has_diag_gen = table_info
        .iter()
        .any(|row| row.get::<String>(1).ok().as_deref() == Some("diagnostics_generation"));

    if !has_diag_gen {
        tracing::info!("Schema migration: adding workspaces.diagnostics_generation column");
        conn.execute(
            "ALTER TABLE workspaces ADD COLUMN diagnostics_generation INTEGER NOT NULL DEFAULT 0",
            (),
        )
        .await
        .context("Schema migration failed: unable to add workspaces.diagnostics_generation")?;
        conn.checkpoint().await.context(
            "Schema migration failed: unable to checkpoint after adding workspaces.diagnostics_generation",
        )?;
        tracing::info!("Schema migration complete: added workspaces.diagnostics_generation column");
    }
    Ok(())
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

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        let now = crate::turso::now();
        let paused_int: i64 = i64::from(paused);
        let maint_int: i64 = i64::from(maintenance_enabled);
        store
            .conn
            .execute(
                "INSERT INTO workspaces (name, path, created_at, updated_at, paused, maintenance, discovery_generation, diagnostics_generation) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                crate::turso::params![name, path, now.clone(), now.clone(), paused_int, maint_int, discovery_generation, diagnostics_generation],
            )
            .await
            .expect("insert workspace");
        Workspace {
            name: name.to_string(),
            path: path.to_string(),
            status: WorkspaceStatus::Pending,
            created_at: now.clone(),
            updated_at: now.clone(),
            maintenance_enabled,
            paused,
            maintainer_debounce_mins: 5,
            maintainer_last_run_at: None,
            diagnostics: None,
            diagnostics_updated_at: None,
            notes: String::new(),
        }
    }

    // ── Schema / struct consistency ─────────────────────────────

    #[tokio::test]
    async fn schema_default_is_paused() {
        // Insert WITHOUT specifying paused, relying on the schema DEFAULT.
        let (store, _tmp) = test_store().await;
        let now = crate::turso::now();
        store
            .conn
            .execute(
                "INSERT INTO workspaces (name, path, created_at, updated_at) \
                 VALUES (?1, ?2, ?3, ?4)",
                crate::turso::params!["schema_test", "/tmp/schema_test", now.clone(), now.clone()],
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

        // Verify 4000 char truncation
        let long_notes = "x".repeat(5000);
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
            4000,
            "Notes should be truncated to 4000 chars"
        );
        assert_eq!(
            ws.notes,
            "x".repeat(4000),
            "Notes content should match truncated"
        );

        // Verify UTF-8 safe truncation (multi-byte characters)
        let multi_byte = "é".repeat(5000);
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
            4000,
            "Notes should be truncated to 4000 chars (multi-byte)"
        );
        assert_eq!(
            ws.notes,
            "é".repeat(4000),
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
            finalize_discovery(&store, suffix, generation, true, &[]).await;

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

        // Act: discovery failed (all_ok = false).
        let errors = vec!["Diagnostics discovery failed: timeout".to_string()];
        finalize_discovery(&store, "fail_gen0", 0, false, &errors).await;

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
    async fn finalize_discovery_stale_generation_skips_writes() {
        let (store, _tmp) = test_store().await;
        // Start paused with generation 0.
        insert_direct(&store, "stale", "/tmp/stale", true, false, 0, 0).await;

        // Bump the generation behind the scenes (simulates a concurrent
        // rediscover() call).
        let now = crate::turso::now();
        store
            .conn
            .execute(
                "UPDATE workspaces SET discovery_generation = 1, updated_at = ?1 WHERE name = ?2",
                crate::turso::params![now, "stale"],
            )
            .await
            .expect("bump generation");

        // Act: try to finalize with the stale generation 0.
        finalize_discovery(&store, "stale", 0, true, &[]).await;

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
    async fn add_returns_paused_true() {
        let (store, _tmp) = test_store().await;
        let dir = TempDir::new().expect("temp dir for workspace path");

        // add() requires: search engine globals initialized + CONFIG storage
        // root set.  Initialize the minimum globals.
        if !crate::search_engine::registry_initialized() {
            crate::search_engine::init_global();
        }
        // Set a throwaway storage root if not already set (the OnceLock
        // panics on double-set, so we only set if not already set).
        let tmp_root = TempDir::new().expect("storage root temp dir");
        let _ = crate::config::CONFIG.try_set_storage_root(tmp_root.path().to_path_buf());
        crate::config::CONFIG.swap(crate::config::ConfigData::STRUCT_FIELDS_DEFAULT);

        let ws = store
            .add("add_test", dir.path().to_str().unwrap())
            .await
            .expect("add workspace");

        assert!(
            ws.paused,
            "add() must return a Workspace with paused = true"
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
        let now = crate::turso::now();
        store
            .set_diagnostics("diag_test", &cmds, &now)
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
    async fn get_diagnostics_generation_default() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "gen_test", "/tmp/gen_test", false, false, 5, 3).await;

        let diag_gen_val = store
            .get_diagnostics_generation("gen_test")
            .await
            .expect("get_diagnostics_generation");
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
        let now = crate::turso::now();
        store
            .set_diagnostics("bump_test", &cmds, &now)
            .await
            .expect("set_diagnostics");

        let diag_gen_val = store
            .get_diagnostics_generation("bump_test")
            .await
            .expect("get_diagnostics_generation");
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
        let now = crate::turso::now();
        store
            .set_diagnostics("redia_test", &cmds, &now)
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
            .get_diagnostics_generation("redia_test")
            .await
            .expect("get_diagnostics_generation");
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
            .get_diagnostics_generation("redia_test")
            .await
            .expect("get_diagnostics_generation");
        assert_eq!(
            diag_gen_after, 2,
            "rediscover_diagnostics should bump diagnostics_generation to 2"
        );

        // discovery_generation should NOT have been touched.
        let discovery_gen_val = store
            .get_discovery_generation("redia_test")
            .await
            .expect("get_discovery_generation");
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
        let now = crate::turso::now();
        store
            .set_diagnostics("diag_stale", &cmds, &now)
            .await
            .expect("set_diagnostics");

        // Bump the diagnostics_generation behind the scenes (simulates a concurrent
        // rediscover_diagnostics() or set_diagnostics() call).
        store
            .conn
            .execute(
                "UPDATE workspaces SET diagnostics_generation = 99 WHERE name = ?1",
                crate::turso::params!["diag_stale"],
            )
            .await
            .expect("bump diagnostics_generation");

        // Capture the stale generation (1) and verify the guard catches it.
        let stale_gen_val = 1;
        let is_ok = check_diagnostics_generation(&store, "diag_stale", stale_gen_val, "test").await;
        assert!(
            !is_ok,
            "check_diagnostics_generation should reject stale generation"
        );

        // Fresh generation should pass.
        let fresh_gen_val = 99;
        let is_ok = check_diagnostics_generation(&store, "diag_stale", fresh_gen_val, "test").await;
        assert!(
            is_ok,
            "check_diagnostics_generation should accept fresh generation"
        );
    }
}
