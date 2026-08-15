//! Per-user identity, permissions, workspace and role preferences, and channel bindings.
//!
//! Two tables in `users.db`:
//! - `users` — canonical user identity: `name`, `permissions`, `selected_workspace`, `selected_role`.
//! - `user_channels` — channel bindings: maps a channel+identifier (e.g. Telegram @username)
//!   to a user. The `reply_target` is stored here (per-channel routing address).
//!
//! User identity is independent of any external channel. Changing a Telegram
//! `@username` does not affect the user's identity. Users are created via the
//! GUI dashboard, and channels are bound explicitly.
//!
//! ## Personal workspaces
//!
//! When `selected_workspace` is NULL, the user has a personal workspace at
//! `~/.mahbot/userspaces/<name>/`. It is NOT registered in `workspaces.db` —
//! computed on the fly. Personal workspaces have no board pipeline, no
//! maintainer, no diagnostics discovery.

use crate::Role;
use crate::Workspace;
use crate::WorkspaceStatus;
use crate::git_commands::run_git_output;
use crate::turso::{self, TxGuard};
use anyhow::{Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use strum::IntoEnumIterator;
use tracing::warn;

crate::define_store! {
    /// Global user store.
    pub(crate) static USER_STORE: UserStore,
    db_name = "users",
    schema = SCHEMA,
    post_open = ensure_admin_user,
    expect = "USER_STORE not initialized — call init_global() first",
}

const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS users (
    name                TEXT PRIMARY KEY,
    permissions         TEXT,
    selected_workspace  TEXT,
    selected_role       TEXT,
    role_pool_initialized INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS user_channels (
    user_name   TEXT NOT NULL REFERENCES users(name),
    channel     TEXT NOT NULL,
    identifier  TEXT NOT NULL,
    reply_target TEXT,
    UNIQUE(channel, identifier)
);
CREATE TABLE IF NOT EXISTS user_roles (
    user_name   TEXT NOT NULL REFERENCES users(name),
    role        TEXT NOT NULL,
    PRIMARY KEY (user_name, role)
);";

// ── Column index constants ──────────────────────────────────

// users table (4-column SELECT: name, permissions, selected_workspace, selected_role)
crate::columns! {
    USERS_COLUMNS [USERS] {
        NAME                => "name",
        PERMISSIONS         => "permissions",
        SELECTED_WORKSPACE  => "selected_workspace",
        SELECTED_ROLE       => "selected_role",
    }
}

// user_channels table (3-column SELECT: channel, identifier, reply_target)
crate::columns! {
    USER_CHANNEL_COLUMNS [UC] {
        CHANNEL      => "channel",
        IDENTIFIER   => "identifier",
        REPLY_TARGET => "reply_target",
    }
}

impl UserStore {
    /// Auto-create the admin user if this is a fresh database.
    async fn ensure_admin_user(&self) -> Result<()> {
        let rows = self
            .conn
            .query("SELECT 1 FROM users WHERE name = 'admin'", turso::params![])
            .await?;
        if rows.is_empty() {
            // Analyst first so add_user seeds the pre-pool default active
            // role in the same transaction — no two-step reset with a
            // Manager-default crash window on fresh installs.
            let mut roles = vec![Role::Analyst];
            roles.extend(Role::iter().filter(|r| *r != Role::Analyst));
            self.add_user("admin", Some("full"), &roles).await?;
        }
        Ok(())
    }

    // ── User CRUD ─────────────────────────────────────────────

    /// Create a new user with the given role pool. The active role is set to
    /// the first pool role. Also creates their personal workspace directory
    /// under `~/.mahbot/userspaces/<name>/` with `git init` (non-fatal on
    /// failure). Idempotent — re-adding an existing user preserves their
    /// stored preferences and adds the given roles to their pool. The
    /// `role_pool_initialized` marker is set unconditionally (schema parity
    /// with live databases).
    pub async fn add_user(
        &self,
        name: &str,
        permissions: Option<&str>,
        roles: &[Role],
    ) -> Result<()> {
        let inserted = self
            .conn
            .execute(
                "INSERT OR IGNORE INTO users (name, permissions, role_pool_initialized) \
                 VALUES (?1, ?2, 1)",
                turso::params![name, permissions],
            )
            .await?;
        let tx = self.conn.begin_tx().await?;
        for role in roles {
            tx.execute(
                "INSERT OR IGNORE INTO user_roles (user_name, role) VALUES (?1, ?2)",
                turso::params![name, role.as_str()],
            )
            .await?;
        }
        if inserted > 0
            && let Some(first) = roles.first()
        {
            tx.execute(
                "UPDATE users SET selected_role = ?1 WHERE name = ?2",
                turso::params![first.as_str(), name],
            )
            .await?;
        }
        // Mark the pool as initialized even when the INSERT OR IGNORE
        // no-op'd for an existing user.
        tx.execute(
            "UPDATE users SET role_pool_initialized = 1 WHERE name = ?1",
            turso::params![name],
        )
        .await?;
        tx.commit().await?;

        // Create personal workspace directory.
        init_personal_workspace_dir(name).await;

        Ok(())
    }

    /// Replace a user's role pool. The active role stays in the pool when
    /// possible; a removed selection falls back to the first remaining role,
    /// and an emptied pool clears the selection (no routing until reassigned).
    pub async fn set_user_roles(&self, name: &str, roles: &[Role]) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        tx.execute(
            "DELETE FROM user_roles WHERE user_name = ?1",
            turso::params![name],
        )
        .await?;
        for role in roles {
            tx.execute(
                "INSERT INTO user_roles (user_name, role) VALUES (?1, ?2)",
                turso::params![name, role.as_str()],
            )
            .await?;
        }
        tx.execute(
            "UPDATE users SET role_pool_initialized = 1 WHERE name = ?1",
            turso::params![name],
        )
        .await?;
        let selected: Option<String> = tx
            .query_row(
                "SELECT selected_role FROM users WHERE name = ?1",
                turso::params![name],
                |row| row.get::<Option<String>>(0),
            )
            .await
            .context("Failed to read selected_role while updating role pool")?;
        match selected {
            Some(cur) if roles.iter().any(|r| r.as_str() == cur) => {}
            Some(_) => {
                // Current selection was removed — fall back to the first
                // remaining pool role (or clear when the pool is empty).
                let fallback = roles.first().map(|r| r.as_str().to_string());
                tx.execute(
                    "UPDATE users SET selected_role = ?1 WHERE name = ?2",
                    turso::params![fallback, name],
                )
                .await?;
            }
            None => {}
        }
        tx.commit().await?;
        Ok(())
    }

    /// List the roles in a user's pool, in canonical [`Role`] iteration order.
    pub async fn get_user_roles(&self, user_name: &str) -> Result<Vec<Role>> {
        let rows = self
            .conn
            .query_map_strict(
                "SELECT role FROM user_roles WHERE user_name = ?1",
                turso::params![user_name],
                |row| row.get::<String>(0),
            )
            .await?;
        let parsed: Vec<Role> = rows.iter().filter_map(|s| s.parse::<Role>().ok()).collect();
        Ok(Role::iter().filter(|r| parsed.contains(r)).collect())
    }

    /// Delete a user and all their child rows (channel bindings, role pool).
    /// The role-pool rows must be removed explicitly — `user_roles` has a
    /// NO-ACTION FK to `users`, so the parent DELETE would fail under
    /// `PRAGMA foreign_keys = ON` otherwise.
    pub async fn delete_user(&self, name: &str) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        tx.execute(
            "DELETE FROM user_roles WHERE user_name = ?1",
            turso::params![name],
        )
        .await?;
        tx.execute(
            "DELETE FROM user_channels WHERE user_name = ?1",
            turso::params![name],
        )
        .await?;
        tx.execute("DELETE FROM users WHERE name = ?1", turso::params![name])
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Fetch a single nullable column from the user's row, if the user exists.
    ///
    /// A NULL column and a missing row both yield `None` (NULL-vs-missing
    /// conflation preserved from the pre-helper implementation).
    async fn user_column(&self, column: &str, user_name: &str) -> Result<Option<String>> {
        self.conn
            .query_optional(
                &format!("SELECT {column} FROM users WHERE name = ?1"),
                turso::params![user_name],
                |row| row.get::<Option<String>>(0),
            )
            .await
            .map(Option::flatten)
    }

    /// Get the selected workspace name for a user, if any.
    pub async fn get_selected_workspace_name(&self, user_name: &str) -> Result<Option<String>> {
        self.user_column("selected_workspace", user_name).await
    }

    /// Get the active role for a user, if any.
    pub async fn get_active_role(&self, user_name: &str) -> Result<Option<String>> {
        self.user_column("selected_role", user_name).await
    }

    /// Get the permissions value for a user (NULL = restricted, "full" = admin).
    pub async fn get_permissions(&self, user_name: &str) -> Result<Option<String>> {
        self.user_column("permissions", user_name).await
    }

    /// Find the first user whose channel binding for `channel` has a
    /// `reply_target` matching `target` (exact, or `target:thread`). Used for
    /// reverse lookup of the recipient of an outbound message — first match
    /// wins when multiple users share a chat (group chats).
    pub async fn resolve_user_by_reply_target(
        &self,
        channel: &str,
        target: &str,
    ) -> Result<Option<String>> {
        let rows = self
            .conn
            .query(
                "SELECT user_name, reply_target FROM user_channels WHERE channel = ?1",
                turso::params![channel],
            )
            .await?;
        let thread_prefix = format!("{target}:");
        for row in rows {
            let user_name: String = row.get(0)?;
            let reply_target: Option<String> = row.get(1)?;
            if let Some(t) = reply_target
                && (t == target || t.starts_with(&thread_prefix))
            {
                return Ok(Some(user_name));
            }
        }
        Ok(None)
    }

    // ── Channel bindings ──────────────────────────────────────

    /// Bind a channel to a user. `channel` is e.g. `"telegram"`, `identifier`
    /// is the channel-specific identifier (Telegram @username without the @ prefix).
    /// Uses INSERT OR REPLACE — binding a username already assigned to another
    /// user silently reassigns it.
    pub async fn bind_channel(
        &self,
        user_name: &str,
        channel: &str,
        identifier: &str,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO user_channels (user_name, channel, identifier) \
                 VALUES (?1, ?2, ?3)",
                turso::params![user_name, channel, identifier],
            )
            .await?;
        Ok(())
    }

    /// Unbind a channel from a user.
    pub async fn unbind_channel(
        &self,
        user_name: &str,
        channel: &str,
        identifier: &str,
    ) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM user_channels WHERE user_name = ?1 AND channel = ?2 AND identifier = ?3",
                turso::params![user_name, channel, identifier],
            )
            .await?;
        Ok(())
    }

    /// Update the reply_target for a channel binding (called on every incoming message).
    pub async fn update_channel_contact(
        &self,
        channel: &str,
        identifier: &str,
        reply_target: &str,
    ) -> Result<()> {
        self.conn
            .execute(
                "UPDATE user_channels SET reply_target = ?1 \
                 WHERE channel = ?2 AND identifier = ?3",
                turso::params![reply_target, channel, identifier],
            )
            .await?;
        Ok(())
    }

    /// Resolve a channel+identifier pair to a user name. Returns `None` if
    /// no binding exists (user not authorized on this channel).
    pub async fn resolve_user_by_channel(
        &self,
        channel: &str,
        identifier: &str,
    ) -> Result<Option<String>> {
        self.conn
            .query_optional(
                "SELECT user_name FROM user_channels WHERE channel = ?1 AND identifier = ?2",
                turso::params![channel, identifier],
                |row| row.get::<String>(0),
            )
            .await
    }

    /// Get all channel bindings for a user.
    pub async fn get_user_channels(&self, user_name: &str) -> Result<Vec<ChannelBinding>> {
        self.conn
            .query_map_strict(
                &format!("SELECT {USER_CHANNEL_COLUMNS} FROM user_channels WHERE user_name = ?1"),
                turso::params![user_name],
                |row| {
                    Ok::<_, ::turso::Error>(ChannelBinding {
                        channel: row.get::<String>(COL_UC_CHANNEL)?,
                        identifier: row.get::<String>(COL_UC_IDENTIFIER)?,
                        reply_target: row.get::<Option<String>>(COL_UC_REPLY_TARGET)?,
                    })
                },
            )
            .await
    }

    /// Convert a `users` table row into a [`UserRecord`], loading channel bindings.
    async fn user_record_from_row(&self, row: &turso::Row) -> Result<UserRecord> {
        let name: String = row.get(COL_USERS_NAME)?;
        let roles = self.get_user_roles(&name).await.unwrap_or_default();
        Ok(UserRecord {
            name: name.clone(),
            permissions: row.get::<Option<String>>(COL_USERS_PERMISSIONS)?,
            selected_workspace: row.get::<Option<String>>(COL_USERS_SELECTED_WORKSPACE)?,
            selected_role: row.get::<Option<String>>(COL_USERS_SELECTED_ROLE)?,
            roles: roles.iter().map(|r| r.as_str().to_string()).collect(),
            channels: self.get_user_channels(&name).await.unwrap_or_default(),
        })
    }

    // ── Lookup / listing ──────────────────────────────────────

    /// Shared listing body: run `suffix` (everything after `FROM users`) and
    /// collect full [`UserRecord`]s with channel bindings.
    async fn list_users_where(
        &self,
        suffix: &str,
        params: impl turso::IntoParams + Send + 'static,
    ) -> Result<Vec<UserRecord>> {
        let sql = format!("SELECT {USERS_COLUMNS} FROM users {suffix}");
        let rows = self.conn.query(&sql, params).await?;
        let mut users = Vec::with_capacity(rows.len());
        for row in rows {
            users.push(self.user_record_from_row(&row).await?);
        }
        Ok(users)
    }

    /// Find all users whose `selected_workspace` matches the given name
    /// (shared workspaces only — personal workspace users with NULL are excluded).
    pub async fn find_by_workspace(&self, workspace_name: &str) -> Result<Vec<UserRecord>> {
        self.list_users_where(
            "WHERE selected_workspace = ?1",
            turso::params![workspace_name],
        )
        .await
    }

    /// Find a single user by exact name, returning their full record with channel bindings.
    /// Returns `None` if no such user exists.
    pub async fn find_by_name(&self, user_name: &str) -> Result<Option<UserRecord>> {
        let rows = self
            .conn
            .query(
                &format!("SELECT {USERS_COLUMNS} FROM users WHERE name = ?1"),
                turso::params![user_name],
            )
            .await?;
        match rows.into_iter().next() {
            Some(row) => Ok(Some(self.user_record_from_row(&row).await?)),
            None => Ok(None),
        }
    }

    /// List all users.
    pub async fn list_users(&self) -> Result<Vec<UserRecord>> {
        self.list_users_where("", turso::params![]).await
    }

    /// Find the user with admin (full) permissions, if any.
    pub async fn find_admin(&self) -> Result<Option<UserRecord>> {
        self.list_users_where("WHERE permissions = ?1", turso::params!["full"])
            .await
            .map(|users| users.into_iter().next())
    }

    /// Atomically update user preferences (role, workspace, permissions) in a single
    /// transaction. Use [`FieldUpdate::Unchanged`] to leave a column as-is or
    /// [`FieldUpdate::Clear`] to explicitly clear it to NULL.
    pub async fn update_user(
        &self,
        name: &str,
        role_name: FieldUpdate<'_>,
        workspace_name: FieldUpdate<'_>,
        permissions: FieldUpdate<'_>,
    ) -> Result<()> {
        let tx = self.conn.begin_tx().await?;

        upsert_user_column(&tx, name, "selected_role", role_name).await?;
        upsert_user_column(&tx, name, "selected_workspace", workspace_name).await?;
        upsert_user_column(&tx, name, "permissions", permissions).await?;

        tx.commit().await?;
        Ok(())
    }
}

/// Represents an optional update to a user column.
///
/// Used by [`UserStore::update_user`] to express whether a column should be
/// left alone, set to NULL, or updated to a specific value — replacing the
/// confusing `Option<Option<&str>>` tri-state with a self-documenting enum.
#[derive(Debug, Clone, Copy)]
pub enum FieldUpdate<'a> {
    /// Leave the column unchanged (no SQL update).
    Unchanged,
    /// Set the column to NULL.
    Clear,
    /// Set the column to the given value.
    Set(&'a str),
}

/// Upsert a single user column within an existing transaction.
///
/// The `field` parameter MUST be a compile-time string literal to prevent SQL injection.
async fn upsert_user_column(
    tx: &TxGuard<'_>,
    name: &str,
    field: &str,
    value: FieldUpdate<'_>,
) -> Result<()> {
    let val: Option<&str> = match value {
        FieldUpdate::Unchanged => return Ok(()),
        FieldUpdate::Clear => None,
        FieldUpdate::Set(v) => Some(v),
    };
    let sql = format!(
        "INSERT INTO users (name, {field}) VALUES (?1, ?2) \
         ON CONFLICT(name) DO UPDATE SET {field} = excluded.{field}"
    );
    tx.execute(&sql, turso::params![name, val]).await?;
    Ok(())
}

// ── UserRecord ────────────────────────────────────────────────

/// A full user row, returned by [`UserStore::list_users`].
#[derive(Debug, Clone, Serialize)]
pub struct UserRecord {
    /// The canonical user name.
    pub name: String,
    /// Permissions: NULL (restricted) or "full" (admin).
    pub permissions: Option<String>,
    /// Selected shared workspace name, NULL = personal workspace.
    pub selected_workspace: Option<String>,
    /// Selected active role, NULL = pool-dependent default (Analyst when in
    /// the pool, else the first pool role). Empty pool → no routing.
    pub selected_role: Option<String>,
    /// The role pool — role names the user is allowed to use.
    pub roles: Vec<String>,
    /// Channel bindings for this user (Telegram, etc.).
    pub channels: Vec<ChannelBinding>,
}

impl UserRecord {
    /// Whether this user has admin (full) permissions.
    #[must_use]
    pub fn is_admin(&self) -> bool {
        is_admin_permissions(self.permissions.as_deref())
    }
}

/// Whether a permissions value grants admin rights (`"full"`).
#[must_use]
pub fn is_admin_permissions(permissions: Option<&str>) -> bool {
    permissions == Some("full")
}

/// A single channel binding for a user.
#[derive(Debug, Clone, Serialize)]
pub struct ChannelBinding {
    /// The channel type (e.g. "telegram").
    pub channel: String,
    /// The channel-specific identifier (e.g. Telegram @username).
    pub identifier: String,
    /// Routing address for replies on this channel (e.g. Telegram chat_id:thread_id).
    pub reply_target: Option<String>,
}

// ── Personal workspace path helper ────────────────────────────

/// Return the filesystem path for a user's personal workspace:
/// `~/.mahbot/userspaces/<name>/`.
///
/// This path is computed on the fly — personal workspaces are NOT registered
/// in `workspaces.db`.
///
/// When CONFIG is not initialized (e.g. in tests), falls back to the default
/// config directory path.
#[must_use]
pub fn personal_workspace_path(user_name: &str) -> PathBuf {
    let storage_root = crate::config::default_config_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("mahbot_userspaces"));
    storage_root.join("userspaces").join(user_name)
}

/// Creates the personal workspace directory for a user and runs `git init`
/// inside it. Both failures are non-fatal — they are logged as warnings but
/// the caller continues normally.
async fn init_personal_workspace_dir(name: &str) {
    let path = personal_workspace_path(name);
    if let Err(e) = tokio::fs::create_dir_all(&path).await {
        warn!(
            path = %path.display(),
            error = %e,
            "Failed to create personal workspace directory"
        );
    }
    // Try git init; non-fatal on failure.
    match run_git_output(&path, &["init", "-q"]).await {
        Ok(o) if o.status.success() => {}
        Ok(_) => warn!(
            path = %path.display(),
            "git init failed for personal workspace (git may not be installed)"
        ),
        Err(e) => warn!(
            path = %path.display(),
            error = %e,
            "git init failed for personal workspace"
        ),
    }
}

// ── Free functions ──────────────────────────────────────────────

/// Get the raw `selected_workspace` column value for a user.
/// Returns `None` if the user has no stored preference (NULL) or if the
/// user doesn't exist.  Unlike [`get_workspace`], this does NOT synthesize
/// a personal workspace fallback — the caller decides how to interpret NULL.
pub async fn get_raw_selected_workspace(user_name: &str) -> Result<Option<String>> {
    store().get_selected_workspace_name(user_name).await
}

/// Get the current active workspace for a user.
///
/// If `selected_workspace` is set, looks up from `workspaces.db`.
/// If NULL, constructs a personal workspace from the user's name
/// (path: `~/.mahbot/userspaces/<user_name>/`).
pub async fn get_workspace(user_name: &str) -> Result<Option<Workspace>> {
    let s = store();
    let selected = s.get_selected_workspace_name(user_name).await?;
    if let Some(ws_name) = selected {
        // Shared workspace: look up from workspaces.db
        crate::workspace::get_by_name(&ws_name).await
    } else {
        // Personal workspace: construct from userspace path
        let path = personal_workspace_path(user_name);
        Ok(Some(personal_workspace_struct(user_name, &path)))
    }
}

/// Build a `Workspace` struct for a personal workspace.
/// Has no diagnostics, no maintenance, no discovery — minimal defaults.
#[must_use]
pub fn personal_workspace_struct(user_name: &str, path: &Path) -> Workspace {
    let mut ws = Workspace::from_path(path);
    ws.name = format!("personal:{user_name}");
    ws.status = WorkspaceStatus::Ready;
    ws.maintainer_debounce_mins = Workspace::MAX_MAINTAINER_DEBOUNCE_MINS;
    ws
}

/// Resolve the workspace for a user, falling back to a personal workspace
/// if `get_workspace` fails or returns `None`.
pub async fn resolve_workspace_for_user_name(user_name: &str) -> Workspace {
    match get_workspace(user_name).await {
        Ok(Some(ws)) => ws,
        Ok(None) => {
            warn!(
                user_name = %user_name,
                "workspace resolution: selected_workspace points to non-existent workspace; \
                 falling back to personal workspace",
            );
            let path = personal_workspace_path(user_name);
            personal_workspace_struct(user_name, &path)
        }
        Err(e) => {
            warn!(
                user_name = %user_name,
                error = %e,
                "workspace resolution: database error; falling back to personal workspace",
            );
            let path = personal_workspace_path(user_name);
            personal_workspace_struct(user_name, &path)
        }
    }
}

/// Fail-closed pool read for routing: returns `(pool, read_failed)` so the
/// caller can distinguish a genuinely empty pool from a transient store
/// error (and avoid a misleading 'no active role' user notice on the
/// latter). The warning is logged here — a single warn site shared with
/// [`role_pool`].
pub async fn role_pool_status(user_name: &str) -> (Vec<Role>, bool) {
    match store().get_user_roles(user_name).await {
        Ok(pool) => (pool, false),
        Err(e) => {
            tracing::warn!(error = %e, user_name, "Failed to read role pool");
            (Vec::new(), true)
        }
    }
}

/// The role pool for a user — the roles they are allowed to use.
/// Empty when the user has no roles assigned, or when the store read fails
/// (fail closed with a warning — an operator log distinguishes a transient
/// DB error from a genuinely empty pool).
pub async fn role_pool(user_name: &str) -> Vec<Role> {
    role_pool_status(user_name).await.0
}

/// Persist a user's active role. Callers must ensure `role` is in the
/// user's pool (see [`role_pool`]).
pub async fn switch_active_role(user_name: &str, role: Role) -> Result<()> {
    store()
        .update_user(
            user_name,
            FieldUpdate::Set(role.as_str()),
            FieldUpdate::Unchanged,
            FieldUpdate::Unchanged,
        )
        .await
}

/// Resolve the active role for a user from their role pool.
///
/// Returns `None` when the user has no roles assigned (empty pool) — the
/// caller must not route messages to any agent.
///
/// A stored selection is honoured when it is still in the pool; a selection
/// outside the pool (e.g. after a pool edit) falls back to the first pool
/// role. Without a stored selection, Analyst is used when available (the
/// pre-pool default), otherwise the first pool role.
pub async fn resolve_active_role(user_name: &str) -> Option<Role> {
    let pool = role_pool(user_name).await;
    resolve_active_role_from_pool(user_name, &pool).await
}

/// Resolve the active role from an already-fetched pool — avoids a second
/// `user_roles` read when the caller needs the pool anyway (e.g. the
/// Telegram command menu). Fails closed on a `selected_role` read error
/// (warn + no routing), matching the pool-read failure policy.
pub async fn resolve_active_role_from_pool(user_name: &str, pool: &[Role]) -> Option<Role> {
    if pool.is_empty() {
        return None;
    }
    let selected = match store().get_active_role(user_name).await {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, user_name, "Failed to read selected role");
            return None;
        }
    };
    match selected {
        Some(name) => name
            .parse::<Role>()
            .ok()
            .filter(|r| pool.contains(r))
            .or_else(|| pool.first().copied()),
        None if pool.contains(&Role::Analyst) => Some(Role::Analyst),
        None => pool.first().copied(),
    }
}

/// The role that answers for a user in a workspace: personal workspaces do
/// not support the Manager agent (no board pipeline), so Manager falls back
/// to Analyst. The fallback stays inside the user's pool — when Analyst is
/// not in the pool (e.g. a Manager-only pool), the Manager selection is kept
/// (the active-role invariant — the routed role is always one of the pool
/// roles — takes precedence over the pipeline fallback). Product note: this
/// is a deliberate shift from the pre-pool rule (Manager→Analyst
/// unconditionally on personal workspaces); an admin can restore the old
/// fallback for a user by adding Analyst to their pool. Canonical home for
/// the chat and voice routing paths.
#[must_use]
fn resolve_effective_role(role: Role, ws_name: &str, pool: &[Role]) -> Role {
    if role == Role::Manager && is_personal_workspace(ws_name) {
        if pool.contains(&Role::Analyst) {
            Role::Analyst
        } else {
            role
        }
    } else {
        role
    }
}

/// Whether an agent role is pinned to the user's personal workspace:
/// Assistant and Artist always work there regardless of the selected
/// workspace. An empty `user_name` disables pinning — there is no personal
/// identity to pin to, so callers with an unresolvable user must pass the
/// real user explicitly (the voice admin fallback passes "admin").
#[must_use]
fn pins_to_personal(role: Role, ws_name: &str, user_name: &str) -> bool {
    !user_name.is_empty()
        && (role == Role::Assistant || role == Role::Artist)
        && !is_personal_workspace(ws_name)
}

/// Resolve the [`Workspace`] an agent role actually operates in: Assistant
/// and Artist always work in the user's personal workspace regardless of the
/// selected workspace, giving path-dependent callers (enrichment uploads,
/// generated-media writes) the personal workspace's filesystem path.
/// An empty `user_name` disables pinning (no personal identity to pin to),
/// so callers must pass a resolvable user (the voice admin fallback passes
/// "admin").
/// Accepted user decision (no migration): media written before pinning to a
/// project workspace's `uploads/`/`generated/` stays there and is no longer
/// reachable by Artist tools (e.g. video_edit path confinement).
#[must_use]
pub(crate) fn effective_workspace_for_role(
    role: Role,
    ws: Workspace,
    user_name: &str,
) -> Workspace {
    if pins_to_personal(role, &ws.name, user_name) {
        let path = personal_workspace_path(user_name);
        personal_workspace_struct(user_name, &path)
    } else {
        ws
    }
}

/// Resolve the effective (role, workspace) pair atomically: apply
/// `resolve_effective_role` (Manager→Analyst in personal workspaces) then
/// pin Assistant/Artist to the user's personal workspace via
/// `effective_workspace_for_role`. The transformations act on disjoint role
/// sets, but a single call keeps session identity and the pinned workspace
/// consistent at every routing entry point.
#[must_use]
pub fn effective_role_and_workspace(
    role: Role,
    ws: Workspace,
    user_name: &str,
    pool: &[Role],
) -> (Role, Workspace) {
    let role = resolve_effective_role(role, &ws.name, pool);
    let ws = effective_workspace_for_role(role, ws, user_name);
    (role, ws)
}

/// Resolve the (role, workspace) a user's messages route to and their
/// session lives in — the same resolution as routing, so ClearChat and
/// Telegram /clear always clear the actual recipient: the DB-selected
/// workspace, the pool-clamped active role (Analyst fallback for an
/// empty pool), the personal-workspace Manager→Analyst remap, and
/// Assistant/Artist pinning.
pub async fn resolve_session_target(user_name: &str) -> (Role, Workspace) {
    let (ws, pool) = tokio::join!(
        resolve_workspace_for_user_name(user_name),
        role_pool(user_name),
    );
    let role = resolve_active_role_from_pool(user_name, &pool)
        .await
        .unwrap_or(Role::Analyst);
    effective_role_and_workspace(role, ws, user_name, &pool)
}

/// Resolve a channel+identifier pair to the canonical user name.
/// Returns `None` if no binding exists (user not authorized on this channel).
pub async fn resolve_user_by_channel(channel: &str, identifier: &str) -> Option<String> {
    let store = USER_STORE.get()?;
    store
        .resolve_user_by_channel(channel, identifier)
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, ?channel, ?identifier, "Failed to resolve user by channel");
            None
        })
}

/// Resolve the canonical user name whose channel binding's `reply_target`
/// matches the given outbound recipient (exact or `target:thread`).
/// First match wins for group chats shared by multiple users.
pub async fn resolve_user_by_reply_target(channel: &str, target: &str) -> Option<String> {
    let store = USER_STORE.get()?;
    store
        .resolve_user_by_reply_target(channel, target)
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, ?channel, ?target, "Failed to resolve user by reply target");
            None
        })
}

/// Whether the named user has admin (full) permissions. Users without a row
/// or with NULL permissions are not admins.
pub async fn is_admin(user_name: &str) -> bool {
    match USER_STORE.get() {
        Some(store) => match store.get_permissions(user_name).await {
            Ok(perms) => is_admin_permissions(perms.as_deref()),
            Err(e) => {
                tracing::warn!(error = %e, user_name, "Failed to read permissions");
                false
            }
        },
        None => false,
    }
}

/// Update reply_target for a channel binding (called on every incoming message).
pub async fn update_channel_contact(
    channel: &str,
    identifier: &str,
    reply_target: &str,
) -> Result<()> {
    store()
        .update_channel_contact(channel, identifier, reply_target)
        .await
}

/// The user name for a `personal:` workspace name (`personal:{user}`), or
/// `None` when the name is not a personal workspace.
#[must_use]
pub fn personal_user_name(workspace_name: &str) -> Option<&str> {
    workspace_name.strip_prefix("personal:")
}

/// Check whether a workspace name refers to a personal workspace
/// (prefix `personal:`).
#[must_use]
pub fn is_personal_workspace(workspace_name: &str) -> bool {
    personal_user_name(workspace_name).is_some()
}

#[cfg(test)]
pub(crate) mod test_util {
    use super::*;

    /// Initialize a test user store with known users and channel bindings.
    /// Safe to call multiple times — delegates to [`init_test_stores`] to
    /// ensure all global stores are initialized, then supplements
    /// USER_STORE with telegram-specific users and channel bindings.
    pub(crate) async fn init_test_store() {
        // Ensure all global stores are initialized (idempotent OnceCell).
        crate::util::test::init_test_stores().await;

        // Supplement USER_STORE with telegram-specific test users and
        // bindings.  Both `add_user` (INSERT OR IGNORE) and `bind_channel`
        // (INSERT OR REPLACE) are idempotent.
        if let Some(store) = USER_STORE.get() {
            let all_roles = Role::iter().collect::<Vec<_>>();
            store
                .add_user("alice", Some("full"), &all_roles)
                .await
                .expect("failed to add alice to test USER_STORE");
            store
                .add_user("bob", None, &all_roles)
                .await
                .expect("failed to add bob to test USER_STORE");
            store
                .bind_channel("alice", "telegram", "alice")
                .await
                .expect("failed to bind alice telegram");
            store
                .bind_channel("bob", "telegram", "bob")
                .await
                .expect("failed to bind bob telegram");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn role_pool_lifecycle() {
        crate::util::test::init_test_stores().await;
        let store = store();

        // add_user seeds the active role from the first pool role.
        store
            .add_user("pool_user", None, &[Role::Analyst, Role::Coder])
            .await
            .unwrap();
        assert_eq!(resolve_active_role("pool_user").await, Some(Role::Analyst));

        // Removing the active role from the pool falls back to the first
        // remaining role.
        store
            .set_user_roles("pool_user", &[Role::Coder])
            .await
            .unwrap();
        assert_eq!(resolve_active_role("pool_user").await, Some(Role::Coder));

        // A selection outside the pool (defensive) falls back to the first
        // pool role instead of routing to an unallowed role.
        store
            .update_user(
                "pool_user",
                FieldUpdate::Set("engineer"),
                FieldUpdate::Unchanged,
                FieldUpdate::Unchanged,
            )
            .await
            .unwrap();
        assert_eq!(resolve_active_role("pool_user").await, Some(Role::Coder));

        // Emptying the pool stops routing.
        store.set_user_roles("pool_user", &[]).await.unwrap();
        assert_eq!(resolve_active_role("pool_user").await, None);

        // Re-adding a role restores routing; an unset selection keeps the
        // pre-pool Analyst default when it is in the pool.
        store
            .set_user_roles("pool_user", &[Role::Analyst])
            .await
            .unwrap();
        assert_eq!(resolve_active_role("pool_user").await, Some(Role::Analyst));
    }

    #[tokio::test]
    async fn delete_user_removes_role_pool_rows() {
        crate::util::test::init_test_stores().await;
        let store = store();

        // add_user grants the role rows; deleting them must not trip the
        // NO-ACTION FK on user_roles.user_name.
        store
            .add_user("doomed", None, &[Role::Analyst, Role::Coder])
            .await
            .unwrap();
        store.delete_user("doomed").await.unwrap();
        assert!(
            store
                .conn
                .query(
                    "SELECT 1 FROM user_roles WHERE user_name = 'doomed'",
                    crate::turso::params![],
                )
                .await
                .unwrap()
                .is_empty(),
            "user_roles rows must be deleted with the user"
        );
        assert_eq!(
            store.get_user_roles("doomed").await.unwrap(),
            Vec::<Role>::new()
        );
    }

    #[test]
    fn pinning_helpers_pin_assistant_artist_to_personal() {
        // Assistant/Artist + non-personal workspace + non-empty user pin.
        assert!(pins_to_personal(Role::Assistant, "ws1", "alice"));
        assert!(pins_to_personal(Role::Artist, "ws1", "alice"));
        // Already personal, other roles, and empty user_name never pin.
        assert!(!pins_to_personal(
            Role::Assistant,
            "personal:alice",
            "alice"
        ));
        assert!(!pins_to_personal(Role::Manager, "ws1", "alice"));
        assert!(!pins_to_personal(Role::Assistant, "ws1", ""));

        let project = Workspace {
            name: "ws1".to_string(),
            ..Default::default()
        };
        let personal = effective_workspace_for_role(Role::Assistant, project.clone(), "alice");
        assert_eq!(personal.name, "personal:alice");
        assert!(
            personal.path.ends_with("userspaces/alice"),
            "the personal workspace must use the userspaces path, got: {}",
            personal.path
        );
        // Manager keeps the project workspace; already-personal passes through.
        let kept = effective_workspace_for_role(Role::Manager, project.clone(), "alice");
        assert_eq!(kept.name, "ws1");
        let already = effective_workspace_for_role(Role::Artist, personal.clone(), "alice");
        assert_eq!(already.name, "personal:alice");

        // Atomic composition: Manager→Analyst remap in personal workspaces and
        // Assistant/Artist pinning resolve in one call.
        let (role, ws) = effective_role_and_workspace(
            Role::Manager,
            Workspace {
                name: "personal:alice".to_string(),
                ..Default::default()
            },
            "alice",
            &[Role::Analyst],
        );
        assert_eq!(role, Role::Analyst);
        assert_eq!(ws.name, "personal:alice");
    }

    #[tokio::test]
    async fn resolve_session_target_matches_routing() {
        crate::util::test::init_test_stores().await;
        let user = "home_clear_target";
        let store = store();
        store
            .add_user(
                user,
                Some("full"),
                &[Role::Manager, Role::Assistant, Role::Analyst],
            )
            .await
            .unwrap();
        crate::util::test::create_test_workspace(
            "/tmp/home_clear_target_ws",
            "ws_home_clear_target",
        )
        .await;

        // Manager active in a project DB workspace → Manager@project
        // (the routed recipient), even when the GUI picker is on the personal
        // workspace.
        store
            .update_user(
                user,
                FieldUpdate::Set("manager"),
                FieldUpdate::Set("ws_home_clear_target"),
                FieldUpdate::Unchanged,
            )
            .await
            .unwrap();
        let (role, ws) = resolve_session_target(user).await;
        assert_eq!(role, Role::Manager);
        assert_eq!(ws.name, "ws_home_clear_target");

        // Assistant active → pinned to the personal workspace regardless of
        // the DB workspace.
        store
            .update_user(
                user,
                FieldUpdate::Set("assistant"),
                FieldUpdate::Unchanged,
                FieldUpdate::Unchanged,
            )
            .await
            .unwrap();
        let (role, ws) = resolve_session_target(user).await;
        assert_eq!(role, Role::Assistant);
        assert_eq!(ws.name, "personal:home_clear_target");

        // Manager active with no DB workspace → Manager clamps to
        // Analyst@personal (the personal-workspace invariant).
        store
            .update_user(
                user,
                FieldUpdate::Set("manager"),
                FieldUpdate::Clear,
                FieldUpdate::Unchanged,
            )
            .await
            .unwrap();
        let (role, ws) = resolve_session_target(user).await;
        assert_eq!(role, Role::Analyst);
        assert_eq!(ws.name, "personal:home_clear_target");
    }
}
