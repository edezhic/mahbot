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
use crate::global_store;
use crate::turso::{self, Connection, TxGuard};
use anyhow::Result;
use serde::Serialize;
use std::path::{Path, PathBuf};
use tracing::warn;

global_store! {
    /// Global user store.
    pub static USER_STORE: UserStorage,
    constructor = UserStorage::open,
}

/// Turso-backed user preferences storage.
#[derive(Clone, Debug)]
pub struct UserStorage {
    pub(crate) conn: Connection,
}

const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS users (
    name                TEXT PRIMARY KEY,
    permissions         TEXT,
    selected_workspace  TEXT,
    selected_role       TEXT
);
CREATE TABLE IF NOT EXISTS user_channels (
    user_name   TEXT NOT NULL REFERENCES users(name),
    channel     TEXT NOT NULL,
    identifier  TEXT NOT NULL,
    reply_target TEXT,
    UNIQUE(channel, identifier)
);";

impl UserStorage {
    /// Open (or create) the users database at `root/db/users.db`.
    /// On fresh databases, auto-creates the `admin` user with full permissions.
    pub async fn open(root: &Path) -> Result<Self> {
        let db_path = root.join("db/users.db");
        let conn = turso::open_with_schema(&db_path, SCHEMA).await?;
        let this = Self { conn };
        this.ensure_admin_user().await?;
        Ok(this)
    }

    /// Auto-create the admin user if this is a fresh database.
    async fn ensure_admin_user(&self) -> Result<()> {
        let rows = self
            .conn
            .query("SELECT 1 FROM users WHERE name = 'admin'", turso::params![])
            .await?;
        if rows.is_empty() {
            self.conn
                .execute(
                    "INSERT INTO users (name, permissions) VALUES ('admin', 'full')",
                    turso::params![],
                )
                .await?;
            // Also create the admin's personal workspace directory.
            init_personal_workspace_dir("admin").await;
        }
        Ok(())
    }

    // ── User CRUD ─────────────────────────────────────────────

    /// Create a new user. Also creates their personal workspace directory
    /// under `~/.mahbot/userspaces/<name>/` with `git init` (non-fatal on failure).
    pub async fn add_user(&self, name: &str, permissions: Option<&str>) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR IGNORE INTO users (name, permissions) VALUES (?1, ?2)",
                turso::params![name, permissions],
            )
            .await?;

        // Create personal workspace directory.
        init_personal_workspace_dir(name).await;

        Ok(())
    }

    /// Delete a user and all their channel bindings (cascading).
    pub async fn delete_user(&self, name: &str) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
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

    /// Get permissions for a user.
    pub async fn get_permissions(&self, name: &str) -> Result<Option<String>> {
        let rows = self
            .conn
            .query(
                "SELECT permissions FROM users WHERE name = ?1",
                turso::params![name],
            )
            .await?;
        match rows.into_iter().next() {
            Some(row) => Ok(row.get::<Option<String>>(0)?),
            None => Ok(None),
        }
    }

    // ── Workspace ─────────────────────────────────────────────

    /// Set the selected workspace for a user. Pass `None` to clear it
    /// (switching back to personal workspace).
    pub async fn set_workspace(&self, user_name: &str, workspace_name: Option<&str>) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO users (name, selected_workspace) VALUES (?1, ?2) \
                 ON CONFLICT(name) DO UPDATE SET selected_workspace = excluded.selected_workspace",
                turso::params![user_name, workspace_name],
            )
            .await?;
        Ok(())
    }

    /// Get the selected workspace name for a user, if any.
    pub async fn get_selected_workspace_name(&self, user_name: &str) -> Result<Option<String>> {
        let rows = self
            .conn
            .query(
                "SELECT selected_workspace FROM users WHERE name = ?1",
                turso::params![user_name],
            )
            .await?;
        match rows.into_iter().next() {
            Some(row) => Ok(row.get::<Option<String>>(0)?),
            None => Ok(None),
        }
    }

    // ── Role ──────────────────────────────────────────────────

    /// Set the active role for a user.
    pub async fn set_active_role(&self, user_name: &str, role_name: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO users (name, selected_role) VALUES (?1, ?2) \
                 ON CONFLICT(name) DO UPDATE SET selected_role = excluded.selected_role",
                turso::params![user_name, role_name],
            )
            .await?;
        Ok(())
    }

    /// Get the active role for a user, if any.
    pub async fn get_active_role(&self, user_name: &str) -> Result<Option<String>> {
        let rows = self
            .conn
            .query(
                "SELECT selected_role FROM users WHERE name = ?1",
                turso::params![user_name],
            )
            .await?;
        match rows.into_iter().next() {
            Some(row) => Ok(row.get::<Option<String>>(0)?),
            None => Ok(None),
        }
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
        let rows = self
            .conn
            .query(
                "SELECT user_name FROM user_channels WHERE channel = ?1 AND identifier = ?2",
                turso::params![channel, identifier],
            )
            .await?;
        match rows.into_iter().next() {
            Some(row) => Ok(Some(row.get::<String>(0)?)),
            None => Ok(None),
        }
    }

    /// Get all channel bindings for a user.
    pub async fn get_user_channels(&self, user_name: &str) -> Result<Vec<ChannelBinding>> {
        let rows = self
            .conn
            .query(
                "SELECT channel, identifier, reply_target FROM user_channels WHERE user_name = ?1",
                turso::params![user_name],
            )
            .await?;
        let mut bindings = Vec::new();
        for row in rows {
            bindings.push(ChannelBinding {
                channel: row.get::<String>(0)?,
                identifier: row.get::<String>(1)?,
                reply_target: row.get::<Option<String>>(2)?,
            });
        }
        Ok(bindings)
    }

    // ── Lookup / listing ──────────────────────────────────────

    /// Find all users whose `selected_workspace` matches the given name
    /// (shared workspaces only — personal workspace users with NULL are excluded).
    pub async fn find_by_workspace(&self, workspace_name: &str) -> Result<Vec<UserRecord>> {
        let rows = self
            .conn
            .query(
                "SELECT name, permissions, selected_workspace, selected_role \
                 FROM users WHERE selected_workspace = ?1",
                turso::params![workspace_name],
            )
            .await?;
        let mut users = Vec::new();
        for row in rows {
            let name: String = row.get(0)?;
            users.push(UserRecord {
                name: name.clone(),
                permissions: row.get::<Option<String>>(1)?,
                selected_workspace: row.get::<Option<String>>(2)?,
                selected_role: row.get::<Option<String>>(3)?,
                channels: self.get_user_channels(&name).await.unwrap_or_default(),
            });
        }
        Ok(users)
    }

    /// List all users.
    pub async fn list_users(&self) -> Result<Vec<UserRecord>> {
        let rows = self
            .conn
            .query(
                "SELECT name, permissions, selected_workspace, selected_role FROM users",
                turso::params![],
            )
            .await?;
        let mut users = Vec::new();
        for row in rows {
            let name: String = row.get(0)?;
            users.push(UserRecord {
                name: name.clone(),
                permissions: row.get::<Option<String>>(1)?,
                selected_workspace: row.get::<Option<String>>(2)?,
                selected_role: row.get::<Option<String>>(3)?,
                channels: self.get_user_channels(&name).await.unwrap_or_default(),
            });
        }
        Ok(users)
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

        upsert_user_field(&tx, name, "selected_role", role_name).await?;
        upsert_user_field(&tx, name, "selected_workspace", workspace_name).await?;
        upsert_user_field(&tx, name, "permissions", permissions).await?;

        tx.commit().await?;
        Ok(())
    }
}

/// Represents an optional update to a user column.
///
/// Used by [`UserStorage::update_user`] to express whether a column should be
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
async fn upsert_user_field(
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

/// A full user row, returned by [`UserStorage::list_users`].
#[derive(Debug, Clone, Serialize)]
pub struct UserRecord {
    /// The canonical user name.
    pub name: String,
    /// Permissions: NULL (restricted) or "full" (admin).
    pub permissions: Option<String>,
    /// Selected shared workspace name, NULL = personal workspace.
    pub selected_workspace: Option<String>,
    /// Selected active role, NULL = default (analyst).
    pub selected_role: Option<String>,
    /// Channel bindings for this user (Telegram, etc.).
    pub channels: Vec<ChannelBinding>,
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
        .unwrap_or_else(|_| std::env::temp_dir().join("mahbot_test_userspaces"));
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
    match tokio::process::Command::new("git")
        .arg("init")
        .arg("-q")
        .current_dir(&path)
        .output()
        .await
    {
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

/// Switch a user's active workspace to a shared workspace (from workspaces.db).
/// Returns the `Workspace` on success.
pub async fn set_workspace(user_name: &str, name: &str) -> Result<Workspace> {
    let ws = match crate::workspace::get_by_name(name).await {
        Ok(Some(ws)) => ws,
        Ok(None) => anyhow::bail!("Workspace '{name}' not found"),
        Err(e) => anyhow::bail!("Database error looking up workspace '{name}': {e}"),
    };
    let user_store = store();
    user_store.set_workspace(user_name, Some(&ws.name)).await?;
    Ok(ws)
}

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
    ws.status = "ready".to_string();
    ws.maintainer_debounce_mins = 240;
    let now = turso::now();
    ws.created_at.clone_from(&now);
    ws.updated_at = now;
    ws
}

/// Set the active role for a user.
pub async fn set_active_role(user_name: &str, role_name: &str) -> Result<()> {
    store().set_active_role(user_name, role_name).await
}

/// Get the active role for a user, if any.
pub async fn get_active_role(user_name: &str) -> Result<Option<String>> {
    store().get_active_role(user_name).await
}

/// Resolve the active role for a user. Defaults to Analyst when unset.
pub async fn resolve_active_role(user_name: &str) -> Role {
    match get_active_role(user_name).await {
        Ok(Some(name)) => name.parse::<Role>().unwrap_or(Role::Analyst),
        _ => Role::Analyst,
    }
}

/// Check if a user has "full" permissions.
pub async fn is_full_user(user_name: &str) -> bool {
    let Some(store) = USER_STORE.get() else {
        return false;
    };
    matches!(
        store.get_permissions(user_name).await,
        Ok(Some(p)) if p == "full"
    )
}

/// Resolve a channel+identifier pair to the canonical user name.
/// Returns `None` if no binding exists (user not authorized on this channel).
pub async fn resolve_user_by_channel(channel: &str, identifier: &str) -> Option<String> {
    let store = USER_STORE.get()?;
    store
        .resolve_user_by_channel(channel, identifier)
        .await
        .unwrap_or(None)
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

/// Check whether a workspace name refers to a personal workspace
/// (prefix `personal:`).
#[must_use]
pub fn is_personal_workspace(workspace_name: &str) -> bool {
    workspace_name.starts_with("personal:")
}

#[cfg(test)]
pub(crate) mod test_util {
    use super::*;

    /// Initialize a test user store with known users and channel bindings.
    /// Safe to call multiple times — only the first call sets the global store.
    pub(crate) async fn init_test_store() {
        if USER_STORE.get().is_some() {
            return;
        }
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir for test user store");
        let store = UserStorage::open(dir.path())
            .await
            .expect("Failed to open test user store");
        store
            .add_user("alice", Some("full"))
            .await
            .expect("Failed to add alice");
        store
            .add_user("bob", None)
            .await
            .expect("Failed to add bob");
        // Bind alice to a Telegram @username
        store
            .bind_channel("alice", "telegram", "alice")
            .await
            .expect("Failed to bind alice telegram");
        // Bind bob to a Telegram @username
        store
            .bind_channel("bob", "telegram", "bob")
            .await
            .expect("Failed to bind bob telegram");
        // Leak the TempDir so the directory stays alive for the duration of tests.
        let _ = Box::leak(Box::new(dir));
        let _ = USER_STORE.set(store);
    }
}
