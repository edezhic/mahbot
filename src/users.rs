//! Per-user identity, permissions, workspace and role preferences, and channel bindings.
//!
//! Two tables in the consolidated domain database (`core.db`):
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
//! `~/.mahbot/userspaces/<name>/`. It is NOT registered in the `workspaces` table —
//! computed on the fly. Personal workspaces have no board pipeline, no
//! maintainer, no diagnostics discovery.

use crate::Role;
use crate::Workspace;
use crate::WorkspaceStatus;
use crate::db::{self, TxGuard};
use crate::git::commands::run_git_output;
use anyhow::Result;
use serde::Serialize;
use std::path::PathBuf;
use tracing::warn;

/// Sentinel that `extract_sender_user_name` (src/channels/telegram.rs)
/// substitutes for Telegram senders without an @username. Never a valid
/// binding identifier: a binding with this identifier would authorize every
/// username-less sender as its owner.
pub(crate) const TELEGRAM_UNKNOWN_SENTINEL: &str = "unknown";

/// Normalize a Telegram handle for binding: trim surrounding whitespace and
/// strip one leading `@`, then trim again. Rejects an empty handle and the
/// reserved `TELEGRAM_UNKNOWN_SENTINEL` (matched case-sensitively — a real
/// user legitimately named @Unknown stays bindable).
fn normalize_telegram_handle(handle: &str) -> anyhow::Result<String> {
    let handle = handle.trim();
    let handle = handle.strip_prefix('@').unwrap_or(handle).trim();
    if handle.is_empty() {
        anyhow::bail!("Telegram handle is empty");
    }
    if handle == TELEGRAM_UNKNOWN_SENTINEL {
        anyhow::bail!("'unknown' is a reserved Telegram handle and cannot be bound");
    }
    Ok(handle.to_string())
}

crate::define_store! {
    /// Global user store.
    pub static USER_STORE: UserStore,
    post_open = ensure_admin_user,
    expect = "USER_STORE not initialized — call init_all_stores() first",
}

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
    ///
    /// Runs idempotently from both [`crate::db::init_all_stores`] (production,
    /// on the shared consolidated connection) and each isolated user store open.
    pub(crate) async fn ensure_admin_user(&self) -> Result<()> {
        if !self.user_exists("admin").await? {
            // Fresh admin: full permissions + selected_role=Support (the first
            // onboarding-pool role).
            self.add_user("admin", Some("full"), Role::Support).await?;
        }
        Ok(())
    }

    // ── User CRUD ─────────────────────────────────────────────

    /// Create a new user with the given active role. The user's role pool is
    /// permission-derived (no `user_roles` rows) — `default_role` is the
    /// persisted active role, only applied on a fresh insert. Also creates
    /// their personal workspace directory under
    /// `~/.mahbot/userspaces/<name>/` with `git init` (non-fatal on failure).
    /// Idempotent — re-adding an existing user preserves their stored
    /// preferences.
    pub async fn add_user(
        &self,
        name: &str,
        permissions: Option<&str>,
        default_role: Role,
    ) -> Result<()> {
        let inserted = self
            .conn
            .execute(
                "INSERT OR IGNORE INTO users (name, permissions) \
                 VALUES (?1, ?2)",
                db::params![name, permissions],
            )
            .await?;
        let tx = self.conn.begin_tx().await?;
        if inserted > 0 {
            tx.execute(
                "UPDATE users SET selected_role = ?1 WHERE name = ?2",
                db::params![default_role.as_str(), name],
            )
            .await?;
        }
        tx.commit().await?;

        ensure_personal_workspace(name).await;

        Ok(())
    }

    /// Delete a user and all their child rows (channel bindings). The
    /// permission-derived role pool carries no per-user rows, so there is
    /// nothing else to remove.
    pub async fn delete_user(&self, name: &str) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        tx.execute(
            "DELETE FROM user_channels WHERE user_name = ?1",
            db::params![name],
        )
        .await?;
        tx.execute("DELETE FROM users WHERE name = ?1", db::params![name])
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Fetch a single nullable column from the user's row, if the user exists.
    ///
    /// A NULL column and a missing row both yield `None`.
    async fn user_column(&self, column: &str, user_name: &str) -> Result<Option<String>> {
        self.conn
            .query_optional(
                &format!("SELECT {column} FROM users WHERE name = ?1"),
                db::params![user_name],
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

    /// Whether a user row with this name exists.
    pub async fn user_exists(&self, name: &str) -> Result<bool> {
        let rows = self
            .conn
            .query("SELECT 1 FROM users WHERE name = ?1", db::params![name])
            .await?;
        Ok(!rows.is_empty())
    }

    // ── Channel bindings ──────────────────────────────────────

    /// Low-level upsert binding a `(channel, identifier)` pair to a user.
    /// `channel` is e.g. `"telegram"`, `identifier` is the channel-specific
    /// identifier (Telegram @username without the @ prefix). Uses
    /// INSERT OR REPLACE — a `(channel, identifier)` pair already bound to
    /// another user is silently reassigned. User-facing Telegram bind paths
    /// must call [`UserStore::validate_telegram_bind`] first (reserved-sentinel
    /// + anti-steal guards).
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
                db::params![user_name, channel, identifier],
            )
            .await?;
        Ok(())
    }

    /// Validate a Telegram handle on a user-facing bind path and return the
    /// normalized identifier (trimmed, leading `@` stripped) ready for
    /// [`UserStore::bind_channel`]. Rejects the reserved "unknown" sentinel and
    /// fails closed on a handle already bound to a DIFFERENT user (anti-steal:
    /// `bind_channel` is INSERT OR REPLACE and would silently reassign it).
    /// Rebinding the same user stays allowed.
    pub async fn validate_telegram_bind(&self, user_name: &str, handle: &str) -> Result<String> {
        let handle = normalize_telegram_handle(handle)?;
        if let Some(existing) = self.resolve_user_by_channel("telegram", &handle).await?
            && existing != user_name
        {
            anyhow::bail!("@{handle} is already bound to user '{existing}'");
        }
        Ok(handle)
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
                db::params![user_name, channel, identifier],
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
                db::params![reply_target, channel, identifier],
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
                db::params![channel, identifier],
                |row| row.get::<String>(0),
            )
            .await
    }

    /// Get all channel bindings for a user.
    pub async fn get_user_channels(&self, user_name: &str) -> Result<Vec<ChannelBinding>> {
        self.conn
            .query_map_strict(
                &format!("SELECT {USER_CHANNEL_COLUMNS} FROM user_channels WHERE user_name = ?1"),
                db::params![user_name],
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
    async fn user_record_from_row(&self, row: &db::Row) -> Result<UserRecord> {
        let name: String = row.get(COL_USERS_NAME)?;
        let permissions = row.get::<Option<String>>(COL_USERS_PERMISSIONS)?;
        let roles = role_pool_for_permissions(permissions.as_deref());
        Ok(UserRecord {
            name: name.clone(),
            permissions: permissions.clone(),
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
        params: impl db::IntoParams + Send + 'static,
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
        self.list_users_where("WHERE selected_workspace = ?1", db::params![workspace_name])
            .await
    }

    /// Find a single user by exact name, returning their full record with channel bindings.
    /// Returns `None` if no such user exists.
    pub async fn find_by_name(&self, user_name: &str) -> Result<Option<UserRecord>> {
        self.list_users_where("WHERE name = ?1", db::params![user_name])
            .await
            .map(|users| users.into_iter().next())
    }

    /// List all users.
    pub async fn list_users(&self) -> Result<Vec<UserRecord>> {
        self.list_users_where("", db::params![]).await
    }

    /// Find the user with admin (full) permissions, if any.
    pub async fn find_admin(&self) -> Result<Option<UserRecord>> {
        self.list_users_where("WHERE permissions = ?1", db::params!["full"])
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

    /// Set the user's active image-generation model (their Telegram picker
    /// choice). An empty/whitespace value resolves to the default at read time.
    pub async fn set_image_gen_model(&self, name: &str, model: &str) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        upsert_user_column(&tx, name, "image_gen_model", FieldUpdate::Set(model)).await?;
        tx.commit().await?;
        Ok(())
    }

    /// Set the user's active video model (their Telegram picker choice).
    /// An empty/whitespace value resolves to the default at read time.
    pub async fn set_video_model(&self, name: &str, model: &str) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        upsert_user_column(&tx, name, "video_model", FieldUpdate::Set(model)).await?;
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
    tx.execute(&sql, db::params![name, val]).await?;
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
    /// Selected active role, NULL = pool-dependent default (the first pool
    /// role). Empty pool → no routing.
    pub selected_role: Option<String>,
    /// The role pool — the roles the user is allowed to use, derived from
    /// their permissions (the permission-derived pool). No longer read from
    /// a `user_roles` table.
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
fn is_admin_permissions(permissions: Option<&str>) -> bool {
    permissions == Some("full")
}

/// The role pool for a permissions value, derived at read time (no `user_roles`
/// table). Full-permissions (admin) users get the onboarding pool; all other
/// users are limited to the personal assistant roles.
#[must_use]
fn role_pool_for_permissions(permissions: Option<&str>) -> Vec<Role> {
    if is_admin_permissions(permissions) {
        vec![Role::Support, Role::Assistant, Role::Manager, Role::Artist]
    } else {
        vec![Role::Assistant, Role::Artist]
    }
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

/// The userspaces root: `<storage_root>/userspaces`.
///
/// This is the single resolution point for where user workspaces live — shared
/// by [`personal_workspace_path`] and `research_cleanup::sweep_media` so both
/// always agree.
///
/// Resolves under the CONFIG storage root when it is set (production:
/// `~/.mahbot`, set at startup; tests: the shared test root, set by
/// `crate::util::test::init_test_stores`). Otherwise it falls back to
/// [`fallback_storage_root`] — production uses the default config directory,
/// tests the shared test root — so a test reaching this fallback (e.g.
/// `add_user` without `init_test_stores`) still stays inside test-owned
/// storage, never the real user config directory.
#[must_use]
pub(crate) fn userspaces_root() -> PathBuf {
    let storage_root = crate::config::CONFIG
        .try_storage_root()
        .unwrap_or_else(fallback_storage_root);
    storage_root.join("userspaces")
}

/// Storage root used when CONFIG has none set yet.
///
/// Production: the default config directory (`~/.mahbot`). Tests: the shared
/// test root — a test that reaches this fallback must never write into the
/// real user config directory. Production always has the storage root set at
/// startup, so the divergence is unobservable in the shipped binary.
#[cfg(test)]
fn fallback_storage_root() -> PathBuf {
    crate::util::test::test_root().clone()
}

#[cfg(not(test))]
fn fallback_storage_root() -> PathBuf {
    crate::config::default_config_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("mahbot_userspaces"))
}

/// Return the filesystem path for a user's personal workspace:
/// `<storage_root>/userspaces/<name>/`.
///
/// This path is computed on the fly — personal workspaces are NOT registered
/// in the `workspaces` table.
#[must_use]
pub fn personal_workspace_path(user_name: &str) -> PathBuf {
    userspaces_root().join(user_name)
}

/// The canonical GUI-wide workspace name for a user's personal workspace:
/// `personal:{user_name}`. This is the single search-engine key shared by the
/// agent side and the dashboard.
#[must_use]
pub fn personal_workspace_name(user_name: &str) -> String {
    format!("personal:{user_name}")
}

/// Ensure the personal workspace directory for a user exists and is
/// git-initialized. Creates the directory if it's missing and runs `git init`
/// only when no repo is present yet (idempotent otherwise). Both failures are
/// non-fatal — they are logged as warnings but the caller continues normally.
pub(crate) async fn ensure_personal_workspace(name: &str) {
    let path = personal_workspace_path(name);
    if let Err(e) = tokio::fs::create_dir_all(&path).await {
        warn!(
            path = %path.display(),
            error = %e,
            "Failed to create personal workspace directory"
        );
    }
    // Try git init only when there is no repo yet; non-fatal on failure.
    if path.join(".git").exists() {
        return;
    }
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

/// Active image-gen model for `user_name`: the user's explicit Telegram
/// picker choice, or the hardcoded default when unset/unresolvable.
pub async fn resolve_image_gen_model(user_name: &str) -> String {
    resolve_user_model_column(user_name, "image_gen_model")
        .await
        .unwrap_or_else(|| crate::config::DEFAULT_IMAGE_GEN_MODEL.to_string())
}

/// Active video model for `user_name`: the user's explicit Telegram
/// picker choice, or the hardcoded default when unset/unresolvable.
pub async fn resolve_video_model(user_name: &str) -> String {
    resolve_user_model_column(user_name, "video_model")
        .await
        .unwrap_or_else(|| crate::config::DEFAULT_VIDEO_MODEL.to_string())
}

/// Read a user's model column via a single-column SELECT (this is a
/// per-tool-call/per-Artist-turn hot path, so no full `UserRecord` load).
/// `None` — user missing, column unset/empty, or a DB error (logged; fail-open
/// to the default, matching the generation tools' semantics).
async fn resolve_user_model_column(user_name: &str, column: &str) -> Option<String> {
    match store().user_column(column, user_name).await {
        Ok(value) => value
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty()),
        Err(e) => {
            tracing::warn!(user_name, column, error = %e, "user model lookup failed; using default");
            None
        }
    }
}

/// Get the current active workspace for a user.
///
/// If `selected_workspace` is set, looks up from the `workspaces` table.
/// If NULL, constructs a personal workspace from the user's name
/// (path: `~/.mahbot/userspaces/<user_name>/`).
async fn get_workspace(user_name: &str) -> Result<Option<Workspace>> {
    let s = store();
    let selected = s.get_selected_workspace_name(user_name).await?;
    if let Some(ws_name) = selected {
        crate::workspace::get_by_name(&ws_name).await
    } else {
        Ok(Some(personal_workspace_struct(user_name)))
    }
}

/// Resolve a workspace by name, synthesizing a personal workspace
/// (`personal:{user}`) when the name is not in the `workspaces` table.
///
/// Shared by the message router and the boot resume path so both treat
/// synthetic personal workspaces identically.
pub async fn resolve_workspace(workspace_name: &str) -> Result<Option<Workspace>> {
    if let Some(ws) = crate::workspace::get_by_name(workspace_name).await? {
        // A shared workspace row whose name looks like a personal workspace can
        // only be a legacy row (validate_name rejects ':'), so `personal:{user}`
        // shadows the personal-workspace key. Log it so the operator can clean
        // it up — the personal key wins the registry elsewhere.
        if is_personal_workspace(&ws.name) {
            warn!(
                workspace_name = %ws.name,
                "Shared workspace name shadows the personal workspace key 'personal:{{user}}'"
            );
        }
        Ok(Some(ws))
    } else if is_personal_workspace(workspace_name) {
        let user_name = personal_user_name(workspace_name)
            .expect("invariant: is_personal_workspace checked the prefix");
        Ok(Some(personal_workspace_struct(user_name)))
    } else {
        Ok(None)
    }
}

/// Build a `Workspace` struct for a personal workspace.
/// Has no diagnostics, no maintenance, no discovery — minimal defaults.
#[must_use]
pub(crate) fn personal_workspace_struct(user_name: &str) -> Workspace {
    let mut ws = Workspace::from_path(&personal_workspace_path(user_name));
    ws.name = personal_workspace_name(user_name);
    ws.status = WorkspaceStatus::Ready;
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
            personal_workspace_struct(user_name)
        }
        Err(e) => {
            warn!(
                user_name = %user_name,
                error = %e,
                "workspace resolution: database error; falling back to personal workspace",
            );
            personal_workspace_struct(user_name)
        }
    }
}

/// Fail-closed pool read for routing: returns `(pool, read_failed)` so the
/// caller can distinguish a genuinely empty pool from a transient store
/// error (and avoid a misleading 'no active role' user notice on the
/// latter). The warning is logged here — a single warn site shared with
/// [`role_pool`].
pub async fn role_pool_status(user_name: &str) -> (Vec<Role>, bool) {
    match store().get_permissions(user_name).await {
        Ok(perms) => (role_pool_for_permissions(perms.as_deref()), false),
        Err(e) => {
            tracing::warn!(error = %e, user_name, "Failed to read role pool");
            (Vec::new(), true)
        }
    }
}

/// The role pool for a user — the roles they are allowed to use, derived from
/// their permissions at read time. Empty when the store read fails (fail
/// closed with a warning), which an operator log distinguishes from a
/// legitimate admin-permission classification.
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
/// outside the pool falls back to the first pool role. Without a stored
/// selection, the first pool role is used (Support for a full-permissions
/// admin, Assistant otherwise).
pub async fn resolve_active_role(user_name: &str) -> Option<Role> {
    let pool = role_pool(user_name).await;
    resolve_active_role_from_pool(user_name, &pool).await
}

/// Resolve the active role from an already-fetched pool — avoids a second
/// pool read when the caller needs the pool anyway (e.g. the
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
        None => pool.first().copied(),
    }
}

/// The role that answers for a user in a workspace: personal workspaces do
/// not support the Manager agent (no board pipeline), so Manager falls back
/// to Assistant. The fallback stays inside the user's pool — a Manager-only
/// pool (not possible under the permission-derived pools) would keep the
/// Manager selection. Canonical home for the chat and voice routing paths.
#[must_use]
fn resolve_effective_role(role: Role, ws_name: &str, pool: &[Role]) -> Role {
    if role == Role::Manager && is_personal_workspace(ws_name) {
        if pool.contains(&Role::Assistant) {
            Role::Assistant
        } else {
            role
        }
    } else {
        role
    }
}

/// Whether an agent role is pinned to the user's personal workspace:
/// Assistant, Artist, and Support always work there regardless of the
/// selected workspace. An empty `user_name` disables pinning — there is no
/// personal identity to pin to, so callers with an unresolvable user must
/// pass the real user explicitly (the voice admin fallback passes "admin").
#[must_use]
fn pins_to_personal(role: Role, ws_name: &str, user_name: &str) -> bool {
    !user_name.is_empty()
        && (role == Role::Assistant || role == Role::Artist || role == Role::Support)
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
        personal_workspace_struct(user_name)
    } else {
        ws
    }
}

/// Resolve the effective (role, workspace) pair atomically: apply
/// `resolve_effective_role` (Manager→Assistant in personal workspaces) then
/// pin Assistant/Artist/Support to the user's personal workspace via
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
/// workspace, the pool-clamped active role (Assistant fallback for an
/// empty pool), the personal-workspace Manager→Assistant remap, and
/// Assistant/Artist/Support pinning.
pub async fn resolve_session_target(user_name: &str) -> (Role, Workspace) {
    let (ws, pool) = tokio::join!(
        resolve_workspace_for_user_name(user_name),
        role_pool(user_name),
    );
    let role = resolve_active_role_from_pool(user_name, &pool)
        .await
        .unwrap_or(Role::Assistant);
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
        .conn
        .query(
            "SELECT user_name, reply_target FROM user_channels WHERE channel = ?1",
            db::params![channel],
        )
        .await
        .and_then(|rows| {
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
        })
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
            store
                .add_user("alice", Some("full"), Role::Support)
                .await
                .expect("failed to add alice to test USER_STORE");
            store
                .add_user("bob", None, Role::Assistant)
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

    #[test]
    fn role_pool_for_permissions_is_permission_derived() {
        // Full-permissions (admin) → the onboarding pool; everyone else → the
        // personal assistant roles. The default (first) role drives routing.
        assert_eq!(
            role_pool_for_permissions(Some("full")),
            vec![Role::Support, Role::Assistant, Role::Manager, Role::Artist]
        );
        assert_eq!(
            role_pool_for_permissions(None),
            vec![Role::Assistant, Role::Artist]
        );
    }

    #[tokio::test]
    async fn role_pool_lifecycle() {
        crate::util::test::init_test_stores().await;
        let store = store();

        // add_user persists the default active role; the pool is
        // permission-derived (no user_roles rows).
        store
            .add_user("pool_user", None, Role::Assistant)
            .await
            .unwrap();
        // A non-full user's permission-derived pool is always
        // [Assistant, Artist]; the persisted default role resolves as active.
        assert_eq!(
            role_pool("pool_user").await,
            vec![Role::Assistant, Role::Artist]
        );
        assert_eq!(
            resolve_active_role("pool_user").await,
            Some(Role::Assistant)
        );

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
        assert_eq!(
            resolve_active_role("pool_user").await,
            Some(Role::Assistant)
        );
    }

    #[tokio::test]
    async fn delete_user_removes_channel_rows() {
        crate::util::test::init_test_stores().await;
        let store = store();

        // add_user creates the user; deleting it removes the channel bindings
        // and the user row. There is no `user_roles` table anymore (the pool is
        // permission-derived), so nothing else needs a cascade.
        store
            .add_user("doomed", None, Role::Assistant)
            .await
            .unwrap();
        store.delete_user("doomed").await.unwrap();
        assert!(
            store
                .conn
                .query(
                    "SELECT 1 FROM user_channels WHERE user_name = 'doomed'",
                    crate::db::params![],
                )
                .await
                .unwrap()
                .is_empty(),
            "user_channels rows must be deleted with the user"
        );
        assert!(
            store.find_by_name("doomed").await.unwrap().is_none(),
            "the user row must be deleted"
        );
    }

    #[test]
    fn normalize_telegram_handle_rules() {
        assert_eq!(normalize_telegram_handle("alice").unwrap(), "alice");
        assert_eq!(normalize_telegram_handle("  alice  ").unwrap(), "alice");
        assert_eq!(normalize_telegram_handle("@alice").unwrap(), "alice");
        // A leading @ is stripped once, then the remainder is trimmed.
        assert_eq!(normalize_telegram_handle(" @ alice ").unwrap(), "alice");
        // Reserved sentinel is rejected (case-sensitively — "Unknown" stays bindable).
        assert!(normalize_telegram_handle("unknown").is_err());
        assert!(normalize_telegram_handle("   ").is_err());
        assert!(normalize_telegram_handle("@").is_err());
        assert_eq!(normalize_telegram_handle("Unknown").unwrap(), "Unknown");
    }

    #[tokio::test]
    async fn validate_telegram_bind_guards() {
        crate::util::test::init_test_stores().await;
        let store = store();
        store
            .add_user("bind_guard_owner", None, Role::Assistant)
            .await
            .unwrap();
        store
            .bind_channel("bind_guard_owner", "telegram", "guard_handle")
            .await
            .unwrap();

        // Rebinding the same owner is allowed.
        assert_eq!(
            store
                .validate_telegram_bind("bind_guard_owner", "@guard_handle")
                .await
                .unwrap(),
            "guard_handle"
        );

        // Anti-steal: a different user cannot take over the handle.
        let err = store
            .validate_telegram_bind("bind_guard_other", "guard_handle")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("bind_guard_owner"),
            "error must name the current owner: {err}"
        );

        // Reserved sentinel is rejected, fail-closed.
        let err = store
            .validate_telegram_bind("bind_guard_owner", "unknown")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("reserved"),
            "error must mention 'reserved': {err}"
        );
    }

    #[test]
    fn pinning_helpers_pin_assistant_artist_support_to_personal() {
        // Assistant/Artist/Support + non-personal workspace + non-empty user pin.
        assert!(pins_to_personal(Role::Assistant, "ws1", "alice"));
        assert!(pins_to_personal(Role::Artist, "ws1", "alice"));
        assert!(pins_to_personal(Role::Support, "ws1", "alice"));
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

        // Atomic composition: Manager→Assistant remap in personal workspaces and
        // Assistant/Artist/Support pinning resolve in one call. A pool that
        // contains Assistant remaps Manager; without Assistant the Manager
        // selection is kept (the active-role invariant).
        let (role, ws) = effective_role_and_workspace(
            Role::Manager,
            Workspace {
                name: "personal:alice".to_string(),
                ..Default::default()
            },
            "alice",
            &[Role::Assistant],
        );
        assert_eq!(role, Role::Assistant);
        assert_eq!(ws.name, "personal:alice");
        let (role, _ws) = effective_role_and_workspace(
            Role::Manager,
            Workspace {
                name: "personal:alice".to_string(),
                ..Default::default()
            },
            "alice",
            &[Role::Analyst],
        );
        assert_eq!(role, Role::Manager);
    }

    #[tokio::test]
    async fn resolve_session_target_matches_routing() {
        crate::util::test::init_test_stores().await;
        let user = "home_clear_target";
        let store = store();
        store
            .add_user(user, Some("full"), Role::Manager)
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

        // Manager active with no DB workspace → Manager remaps to
        // Assistant@personal (the personal-workspace invariant; the pool
        // contains Assistant).
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
        assert_eq!(role, Role::Assistant);
        assert_eq!(ws.name, "personal:home_clear_target");
    }
}
