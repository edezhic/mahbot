//! Workspace storage — persisted workspace metadata and contexts.
//!
//! Also handles workspace analysis: spawning a Discovery agent to explore a new
//! workspace and produce role-specific context summaries.

use crate::Role;
use crate::Workspace;
use crate::agent::run_agent;
use crate::global_store;
use crate::session::discovery_session_key;
use crate::turso::{self, Connection};
use anyhow::{Context, Result};
use futures_util::future::join_all;
use std::path::Path;
use tracing::warn;

global_store! {
    /// Global workspace store.
    pub static WORKSPACES: WorkspaceStorage,
    constructor = WorkspaceStorage::open,
    expect = "workspace::WORKSPACES not initialized — call workspace::init_global() in main.rs",
}

/// Look up a workspace by its filesystem path.
pub async fn get_by_path(path: &str) -> Result<Option<Workspace>> {
    store().get_by_path(path).await
}

/// Look up a workspace by its name.
pub async fn get_by_name(name: &str) -> Result<Option<Workspace>> {
    store().get_by_name(name).await
}

/// Turso-backed workspace storage.
#[derive(Clone, Debug)]
pub struct WorkspaceStorage {
    pub(crate) conn: Connection,
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

/// Column list for workspace SELECT queries.
///
/// The column order here must match the positional indices defined in
/// [`COL_WS_NAME`] through [`COL_WS_DIAGNOSTICS_UPDATED_AT`], which are used
/// in [`workspace_from_row`].
///
/// `discovery_generation` is intentionally excluded from this column list: it
/// is read only via its own single-column SELECT in
/// [`WorkspaceStorage::get_discovery_generation`] and is never part of a workspace struct query.
const WORKSPACE_COLUMNS: &str = "name, path, status, created_at, updated_at, \
     maintenance, paused, maintainer_debounce_mins, maintainer_last_run_at, \
     diagnostics, diagnostics_updated_at";

/// Column-index constants for [`WORKSPACE_COLUMNS`].
///
/// These replace hardcoded positional indices in [`workspace_from_row`].
/// With named constants, the compiler catches references to undefined column
/// constants — for instance, removing a constant but forgetting to update a
/// `row.get()` call produces a compile error rather than a silent field
/// mapping bug.
const COL_WS_NAME: usize = 0;
const COL_WS_PATH: usize = 1;
const COL_WS_STATUS: usize = 2;
const COL_WS_CREATED_AT: usize = 3;
const COL_WS_UPDATED_AT: usize = 4;
const COL_WS_MAINTENANCE: usize = 5;
const COL_WS_PAUSED: usize = 6;
const COL_WS_MAINTAINER_DEBOUNCE_MINS: usize = 7;
const COL_WS_MAINTAINER_LAST_RUN_AT: usize = 8;
const COL_WS_DIAGNOSTICS: usize = 9;
const COL_WS_DIAGNOSTICS_UPDATED_AT: usize = 10;

/// Check the discovery generation counter: return `true` if the calling task
/// is still the latest (OK to proceed), `false` if a newer [`WorkspaceStorage::rediscover`] has
/// been triggered (stale — do not proceed).
async fn check_discovery_generation(
    storage: &WorkspaceStorage,
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

/// Run workspace discovery for a single role, returning the result.
///
/// `discovery_generation` is the generation counter captured at spawn time.
/// Before writing the context, we re-read the current generation from the DB;
/// if it no longer matches, a newer [`WorkspaceStorage::rediscover`] call has been made and this
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
/// `discovery_generation` guards against stale writes — if a newer [`WorkspaceStorage::rediscover`]
/// was triggered while diagnostics were being computed, the write is skipped.
///
/// On failure, existing diagnostics data is left untouched.
async fn run_workspace_diagnostics(ws: &Workspace, discovery_generation: i64) -> Result<()> {
    let storage = WORKSPACES
        .get()
        .context("WORKSPACES not initialized")?
        .clone();

    tracing::info!(workspace_name = ws.name, "Starting diagnostics discovery");

    let agent_id = discovery_session_key(&ws.name, "diagnostics");

    // Load the diagnostics discovery prompt directly (not a role-specific discovery prompt).
    let prompt = crate::prompt::load_prompt("discovery/diagnostics.md");

    let (agent, response) = run_agent(agent_id, Role::Discovery, ws, None, &prompt).await;
    let _response = response
        .context("Diagnostics discovery agent returned no response (cancelled or failed)")?;

    // Keep the Agent alive after run_agent() for retry_extract_structured —
    // it needs agent.session.history() and agent.tool_specs.
    let extraction_prompt = crate::prompt::load_prompt("extraction/diagnostics.md");
    let retry_prompt = crate::prompt::load_prompt("extraction/retry.md");

    // KV-cache preservation: `agent.extract_structured` uses the agent's own
    // parameters (model, temperature, reasoning_effort, tools, provider routing)
    // so the extraction call is byte-identical to the original Discovery agent
    // call — the provider can reuse the cached prefix.
    let cmds: crate::DiagnosticsCommands = agent
        .extract_structured(&extraction_prompt, &retry_prompt, 3)
        .await?;

    // Guard against stale writes — see run_workspace_discovery.
    if !check_discovery_generation(&storage, &ws.name, discovery_generation, "diagnostics").await {
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

// Spawn analysis for all analysis-eligible roles and set final status.
// Runs diagnostics discovery concurrently with role discovery via `tokio::join!`.
// Manager decision: diagnostics failure is FATAL — workspace status becomes "failed".
//
// `discovery_generation` is the generation counter captured by the caller
// (either `add` or `rediscover`). Before writing final status, the task
// re-reads the generation from the DB; if it no longer matches, a newer
// rediscover call has been made and this task's results are stale — all
// writes are skipped.
// ── Discovery completion finalizer ────────────────────────────────

/// Apply the final status and pause state after a discovery run completes.
///
/// This is called from [`spawn_workspace_discovery`] after all role discoveries
/// and diagnostics have finished.  Extracted as a separate function so unit tests
/// can verify the paused-behavior invariants without running real agents.
///
/// ## Invariants
///
/// - If `all_ok`: sets status to `ready`. When `discovery_generation == 0` (the
///   initial discovery), also unpauses the workspace.  Rediscovery (generation > 0)
///   does **not** auto-unpause — if a user explicitly paused the workspace and
///   triggered rediscovery, their choice is preserved.
/// - If **not** `all_ok`: sets status to `failed` and leaves `paused` untouched.
/// - Before any write, checks the generation guard: if a newer [`WorkspaceStorage::rediscover`]
///   bumped the generation while discovery was in flight, the writes are skipped.
async fn finalize_discovery(
    storage: &WorkspaceStorage,
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
        let _ = storage.set_status(ws_name, "ready").await;
        // Auto-unpause only on the initial discovery (generation 0).
        // Rediscovery should NOT auto-unpause — if a user explicitly paused the
        // workspace and then triggered rediscovery, the workspace stays paused.
        if discovery_generation == 0 {
            let _ = storage.set_paused(ws_name, false).await;
        }
        tracing::info!(
            workspace_name = ws_name,
            "Workspace analysis complete — all roles ready"
        );
    } else {
        let msg = errors.join("; ");
        let _ = storage.set_status(ws_name, "failed").await;
        tracing::warn!(workspace_name = ws_name, error = %msg, "Workspace analysis failed");
    }
}

pub fn spawn_workspace_discovery(ws: &Workspace, discovery_generation: i64) {
    let ws = ws.clone();
    tokio::spawn(async move {
        let ws_name = ws.name.clone();

        // Run role discovery and diagnostics discovery concurrently.
        let (role_results, diagnostics_result) = tokio::join!(
            join_all(
                <crate::Role as strum::IntoEnumIterator>::iter()
                    .filter(|r| crate::role::role_info(r).has_discovery)
                    .map(|role| {
                        let ws = ws.clone();
                        async move {
                            run_workspace_discovery(&ws, role, discovery_generation).await
                        }
                    }),
            ),
            run_workspace_diagnostics(&ws, discovery_generation),
        );

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

        finalize_discovery(storage, &ws_name, discovery_generation, all_ok, &errors).await;
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

/// Normalize a workspace path: ensure it ends with a single `/`.
fn normalize_path(path: &str) -> String {
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
    let expanded = crate::config::expand_tilde(raw);

    let canonical = std::fs::canonicalize(&expanded).map_err(|e| {
        if expanded.exists() {
            format!("Cannot access path '{}': {e}", expanded.display())
        } else {
            format!("Path does not exist: {}", expanded.display())
        }
    })?;

    if !canonical.is_dir() {
        return Err(format!("Path is not a directory: {}", canonical.display()));
    }

    Ok(canonical.to_string_lossy().to_string())
}

fn workspace_from_row(row: &turso::Row) -> Result<Workspace, ::turso::Error> {
    Ok(Workspace {
        name: row.get(COL_WS_NAME)?,
        path: row.get(COL_WS_PATH)?,
        status: row.get(COL_WS_STATUS)?,
        created_at: row.get(COL_WS_CREATED_AT)?,
        updated_at: row.get(COL_WS_UPDATED_AT)?,
        maintenance: row.get::<bool>(COL_WS_MAINTENANCE)?,
        paused: row.get::<bool>(COL_WS_PAUSED)?,
        maintainer_debounce_mins: row.get::<i64>(COL_WS_MAINTAINER_DEBOUNCE_MINS)?,
        maintainer_last_run_at: row.get::<Option<String>>(COL_WS_MAINTAINER_LAST_RUN_AT)?,
        diagnostics: row.get::<Option<String>>(COL_WS_DIAGNOSTICS)?,
        diagnostics_updated_at: row.get::<Option<String>>(COL_WS_DIAGNOSTICS_UPDATED_AT)?,
    })
}

impl WorkspaceStorage {
    /// Open (or create) the workspaces database at `root/db/workspaces.db`.
    ///
    /// The `dirty_content` column was added to the `editor_tabs` table in an
    /// earlier version; it is now part of the `CREATE TABLE` schema and no
    /// migration is needed.
    pub async fn open(root: &Path) -> Result<Self> {
        let db_path = root.join("db/workspaces.db");
        let conn = turso::open_with_schema(&db_path, SCHEMA).await?;

        Ok(Self { conn })
    }

    /// Run a query that returns zero-or-one workspace row, mapping the result to
    /// `Ok(Some(ws))` / `Ok(None)` / `Err`.
    async fn query_one(
        &self,
        where_clause: &str,
        params: impl turso::IntoParams + Send + 'static,
    ) -> Result<Option<Workspace>> {
        let sql = format!("SELECT {WORKSPACE_COLUMNS} FROM workspaces WHERE {where_clause}");
        match self.conn.query_row(&sql, params, workspace_from_row).await {
            Ok(ws) => Ok(Some(ws)),
            Err(::turso::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Insert a new workspace and kick off analysis.
    pub async fn add(&self, name: &str, path: &str) -> Result<Workspace> {
        // Validate the workspace name.
        validate_name(name)?;

        // Canonicalize and validate the path so bad paths never enter the system.
        let canonical = canonicalize_workspace_path(path).map_err(|e| anyhow::anyhow!("{e}"))?;
        let path = normalize_path(&canonical);
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
            status: "pending".to_string(),
            created_at: now.clone(),
            updated_at: now.clone(),
            maintenance: false,
            paused: true,
            maintainer_debounce_mins: 5,
            maintainer_last_run_at: None,
            diagnostics: None,
            diagnostics_updated_at: None,
        };
        let _ = self.set_status(name, "analyzing").await;
        // New workspace: discovery_generation defaults to 0 in the schema.
        // Generation 0 means "the first discovery" — if rediscover() bumps
        // the generation before this task finishes, the task's context/
        // diagnostics/status writes will be skipped by the generation guard.
        spawn_workspace_discovery(&ws, 0);
        // Eagerly initialize the shared search engine for this workspace.
        if let Err(e) = crate::search_engine::get_or_init_engine(&ws) {
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
        let mut workspaces = Vec::new();
        for row in rows {
            workspaces.push(row?);
        }
        Ok(workspaces)
    }

    /// Lightweight fetch of only name, paused, and maintenance columns.
    /// Used by the GUI sidebar's periodic state refresh — avoids fetching
    /// all workspace columns when only toggle state is needed.
    pub async fn list_states(&self) -> Result<Vec<(String, bool, bool)>> {
        let rows = self
            .conn
            .query(
                "SELECT name, paused, maintenance FROM workspaces ORDER BY name",
                turso::params![],
            )
            .await?;
        let mut states = Vec::with_capacity(rows.len());
        for row in &rows {
            let name: String = row.get(0)?;
            let paused: bool = row.get(1)?;
            let maintenance: bool = row.get(2)?;
            states.push((name, paused, maintenance));
        }
        Ok(states)
    }

    /// Look up a workspace by name.
    pub async fn get_by_name(&self, name: &str) -> Result<Option<Workspace>> {
        self.query_one("name = ?1", turso::params![name]).await
    }

    /// Look up a workspace by its filesystem path.
    pub async fn get_by_path(&self, path: &str) -> Result<Option<Workspace>> {
        self.query_one("path = ?1", turso::params![normalize_path(path)])
            .await
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
    pub async fn set_status(&self, name: &str, status: &str) -> Result<()> {
        let now = turso::now();
        self.conn
            .execute(
                "UPDATE workspaces SET status = ?1, updated_at = ?2 WHERE name = ?3",
                turso::params![status, now.clone(), name],
            )
            .await?;
        Ok(())
    }

    /// Set or clear the maintenance toggle for a workspace.
    pub async fn set_maintenance(&self, name: &str, enabled: bool) -> Result<()> {
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
    pub async fn set_diagnostics(
        &self,
        name: &str,
        commands: &crate::DiagnosticsCommands,
        timestamp: &str,
    ) -> Result<()> {
        let json = serde_json::to_string(commands)?;
        self.conn
            .execute(
                "UPDATE workspaces SET diagnostics = ?1, diagnostics_updated_at = ?2, updated_at = ?3 WHERE name = ?4",
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

    /// Trigger re-analysis of an existing workspace.
    /// Resets status to "analyzing", clears diagnostics columns, clears stale
    /// per-role contexts, and spawns analysis with a fresh generation counter.
    pub async fn rediscover(&self, name: &str) -> Result<()> {
        let ws = self
            .get_by_name(name)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Workspace {name} not found"))?;

        let now = turso::now();

        // Atomically increment the discovery generation counter so any
        // still-running discovery task from a previous rediscover will
        // see a generation mismatch and skip its writes.
        self.conn
            .execute(
                "UPDATE workspaces SET discovery_generation = discovery_generation + 1, status = 'analyzing', paused = 1, diagnostics = NULL, diagnostics_updated_at = NULL, updated_at = ?1 WHERE name = ?2",
                turso::params![now, name],
            )
            .await?;

        // Clear stale per-role context entries so that old discovery tasks
        // that beat the generation check cannot leave partial data behind.
        self.clear_contexts(name).await?;

        let generation = self.get_discovery_generation(name).await?;
        spawn_workspace_discovery(&ws, generation);

        Ok(())
    }

    /// Get a single context entry by workspace name and role.
    pub async fn get_context(&self, workspace_name: &str, role: &str) -> Result<Option<String>> {
        match self
            .conn
            .query_row(
                "SELECT content FROM workspace_contexts WHERE workspace_name = ?1 AND role = ?2",
                turso::params![workspace_name, role],
                |row| row.get::<String>(0),
            )
            .await
        {
            Ok(content) => Ok(Some(content)),
            Err(::turso::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Upsert a single context entry for a workspace and role.
    pub async fn set_context(&self, workspace_name: &str, role: &str, content: &str) -> Result<()> {
        let now = turso::now();
        self.conn
            .execute(
                "INSERT INTO workspace_contexts (workspace_name, role, content, created_at) VALUES (?1, ?2, ?3, ?4) \
                 ON CONFLICT(workspace_name, role) DO UPDATE SET content = excluded.content, created_at = excluded.created_at",
                turso::params![workspace_name, role, content, now],
            )
            .await?;
        Ok(())
    }

    // ── Editor tab persistence ─────────────────────────────────

    /// Save the current set of open editor tabs for a workspace.
    /// Replaces all existing records for this workspace.
    pub async fn save_editor_tabs(
        &self,
        workspace_name: &str,
        tabs: &[EditorTabRecord],
    ) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        tx.execute(
            "DELETE FROM editor_tabs WHERE workspace_name = ?1",
            turso::params![workspace_name],
        )
        .await?;
        for tab in tabs {
            tx.execute(
                "INSERT INTO editor_tabs (workspace_name, file_path, tab_order, is_active, is_dirty, dirty_content) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                turso::params![
                    workspace_name,
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
    pub async fn load_editor_tabs(&self, workspace_name: &str) -> Result<Vec<EditorTabRecord>> {
        let rows = self.conn
            .query_map(
                "SELECT file_path, tab_order, is_active, is_dirty, dirty_content FROM editor_tabs WHERE workspace_name = ?1 ORDER BY tab_order",
                turso::params![workspace_name],
                |row| -> std::result::Result<EditorTabRecord, String> {
                    Ok(EditorTabRecord {
                        file_path: row.get::<String>(0).unwrap_or_default(),
                        tab_order: usize::try_from(row.get::<i64>(1).unwrap_or(0)).unwrap_or(0),
                        is_active: row.get::<i64>(2).unwrap_or(0) != 0,
                        is_dirty: row.get::<i64>(3).unwrap_or(0) != 0,
                        dirty_content: row.get::<Option<String>>(4).unwrap_or(None),
                    })
                },
            )
            .await?;
        let mut tabs = Vec::new();
        for row in rows {
            let tab = row.map_err(|e| anyhow::anyhow!("Failed to parse editor tab row: {e}"))?;
            if tab.file_path.is_empty() || tab.file_path.trim().is_empty() {
                warn!(
                    workspace = %workspace_name,
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
        maintainer_debounce_mins: 5,
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

    /// Verify that the number of columns in [`WORKSPACE_COLUMNS`] matches the highest
    /// column-index constant + 1. If this test fails, a column was added or removed
    /// from the string list without updating the corresponding `COL_WS_*` constants,
    /// or vice versa — a silent data corruption hazard.
    #[test]
    fn workspace_columns_count_matches_column_constants() {
        let count = WORKSPACE_COLUMNS.split(',').count();
        assert_eq!(
            COL_WS_DIAGNOSTICS_UPDATED_AT + 1,
            count,
            "WORKSPACE_COLUMNS has {count} entries but COL_WS_DIAGNOSTICS_UPDATED_AT ({}) + 1 = {}",
            COL_WS_DIAGNOSTICS_UPDATED_AT,
            COL_WS_DIAGNOSTICS_UPDATED_AT + 1,
        );
    }

    /// Open a temporary workspace store for testing.
    /// Returns (store, temp_dir). The temp_dir is kept alive for the lifetime
    /// of the store (~ the test function).
    async fn test_store() -> (WorkspaceStorage, TempDir) {
        let tmp = TempDir::new().expect("temp dir");
        let store = WorkspaceStorage::open(tmp.path())
            .await
            .expect("open workspace store");
        (store, tmp)
    }

    /// Helper: insert a workspace row directly with full control over fields,
    /// bypassing `add()` (which has side-effects like initializing search
    /// engine globals).
    async fn insert_direct(
        store: &WorkspaceStorage,
        name: &str,
        path: &str,
        paused: bool,
        maintenance: bool,
        discovery_generation: i64,
    ) -> Workspace {
        let now = crate::turso::now();
        let paused_int: i64 = i64::from(paused);
        let maint_int: i64 = i64::from(maintenance);
        store
            .conn
            .execute(
                "INSERT INTO workspaces (name, path, created_at, updated_at, paused, maintenance, discovery_generation) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                crate::turso::params![name, path, now.clone(), now.clone(), paused_int, maint_int, discovery_generation],
            )
            .await
            .expect("insert workspace");
        Workspace {
            name: name.to_string(),
            path: path.to_string(),
            status: "pending".to_string(),
            created_at: now.clone(),
            updated_at: now.clone(),
            maintenance,
            paused,
            maintainer_debounce_mins: 5,
            maintainer_last_run_at: None,
            diagnostics: None,
            diagnostics_updated_at: None,
        }
    }

    // ── Schema / struct consistency ─────────────────────────────

    #[tokio::test]
    async fn new_workspace_starts_paused() {
        let (store, _tmp) = test_store().await;
        let ws = insert_direct(&store, "test_ws", "/tmp/test_ws", true, false, 0).await;
        assert!(ws.paused, "In-memory workspace should have paused = true");

        // Round-trip through the DB.
        let fetched = store
            .get_by_name("test_ws")
            .await
            .expect("fetch workspace")
            .expect("workspace exists");
        assert!(
            fetched.paused,
            "Persisted workspace should have paused = true"
        );
    }

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
        insert_direct(&store, "toggle_test", "/tmp/toggle_test", true, false, 0).await;

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
        insert_direct(&store, "maint_test", "/tmp/maint_test", true, false, 0).await;

        // Enable maintenance
        store.set_maintenance("maint_test", true).await.unwrap();
        let fetched = store
            .get_by_name("maint_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(
            fetched.maintenance,
            "Should have maintenance enabled after set_maintenance(true)"
        );

        // Disable maintenance
        store.set_maintenance("maint_test", false).await.unwrap();
        let fetched = store
            .get_by_name("maint_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(
            !fetched.maintenance,
            "Should have maintenance disabled after set_maintenance(false)"
        );
    }

    #[tokio::test]
    async fn list_states_returns_name_paused_maintenance() {
        let (store, _tmp) = test_store().await;

        // Insert two workspaces with different toggle states.
        insert_direct(&store, "alice", "/tmp/alice", true, false, 0).await;
        store.set_maintenance("alice", false).await.unwrap();

        insert_direct(&store, "bob", "/tmp/bob", false, false, 0).await;
        store.set_maintenance("bob", true).await.unwrap();

        let states = store.list_states().await.expect("list_states");
        assert_eq!(states.len(), 2, "Should return both workspaces");

        // Build a map for assertion.
        let mut map: std::collections::HashMap<&str, (bool, bool)> =
            std::collections::HashMap::new();
        for (name, paused, maintenance) in &states {
            map.insert(name.as_str(), (*paused, *maintenance));
        }

        assert_eq!(
            map.get("alice").copied(),
            Some((true, false)),
            "Alice: paused=true, maintenance=false"
        );
        assert_eq!(
            map.get("bob").copied(),
            Some((false, true)),
            "Bob: paused=false, maintenance=true"
        );
    }

    // ── finalize_discovery — auto-unpause invariants ─────────────

    #[tokio::test]
    async fn finalize_discovery_success_gen0_auto_unpauses() {
        let (store, _tmp) = test_store().await;
        // Start paused with discovery_generation = 0.
        insert_direct(&store, "gen0", "/tmp/gen0", true, false, 0).await;

        // Act: simulate initial discovery completion (all_ok = true, gen 0).
        finalize_discovery(&store, "gen0", 0, true, &[]).await;

        let ws = store
            .get_by_name("gen0")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(!ws.paused, "Should auto-unpause after initial discovery OK");
        assert_eq!(ws.status, "ready", "Status should be 'ready'");
    }

    #[tokio::test]
    async fn finalize_discovery_success_gen1_no_auto_unpause() {
        let (store, _tmp) = test_store().await;
        // Start paused with discovery_generation = 1 (rediscovery case).
        insert_direct(&store, "gen1", "/tmp/gen1", true, false, 1).await;

        // Act: simulate rediscovery completing successfully (generation 1).
        finalize_discovery(&store, "gen1", 1, true, &[]).await;

        let ws = store
            .get_by_name("gen1")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(
            ws.paused,
            "Should NOT auto-unpause on rediscovery (gen > 0)"
        );
        assert_eq!(ws.status, "ready", "Status should be 'ready'");
    }

    #[tokio::test]
    async fn finalize_discovery_failure_keeps_paused() {
        let (store, _tmp) = test_store().await;
        insert_direct(&store, "fail_gen0", "/tmp/fail_gen0", true, false, 0).await;

        // Act: discovery failed (all_ok = false).
        let errors = vec!["Diagnostics discovery failed: timeout".to_string()];
        finalize_discovery(&store, "fail_gen0", 0, false, &errors).await;

        let ws = store
            .get_by_name("fail_gen0")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(ws.paused, "Should remain paused after discovery failure");
        assert_eq!(ws.status, "failed", "Status should be 'failed'");
    }

    #[tokio::test]
    async fn finalize_discovery_stale_generation_skips_writes() {
        let (store, _tmp) = test_store().await;
        // Start paused with generation 0.
        insert_direct(&store, "stale", "/tmp/stale", true, false, 0).await;

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
            ws.status, "pending",
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
        )
        .await;
        store.set_status("rediscover_test", "ready").await.unwrap();

        let ws = store
            .get_by_name("rediscover_test")
            .await
            .expect("fetch")
            .expect("exists");
        assert!(!ws.paused, "Precondition: workspace should start unpaused");
        assert_eq!(ws.status, "ready", "Precondition: status should be 'ready'");

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
        crate::config::CONFIG.swap(crate::config::ConfigData::default());

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
        insert_direct(&store, "ws1", "/tmp/ws1", false, false, 0).await;

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
        assert!(loaded[0].is_dirty);
        assert_eq!(loaded[0].dirty_content.as_deref(), Some("draft text"));
    }
}
