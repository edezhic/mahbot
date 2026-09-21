//! Per-account identity, workspace preference, and channel bindings.
//!
//! Two tables in the consolidated domain database (`core.db`):
//! - `users` — canonical account identity: `name`, `selected_workspace`,
//!   `granted_tools` (the per-account custom-tool grants).
//! - `user_channels` — channel bindings: maps a channel+identifier to an
//!   account. The `reply_target` is stored here (per-channel routing address).
//!   A Telegram identifier is either a nickname or a person's numeric id — an
//!   account holds at most one Telegram binding, so the two are alternatives.
//!
//! Account identity is independent of any external channel. Changing a Telegram
//! `@username` does not affect the account's identity. Accounts are created via
//! the GUI dashboard, and channels are bound explicitly.
//!
//! ## Accounts
//!
//! The admin is the one account named [`ADMIN_USER_NAME`] ([`is_admin_name`]);
//! every other account is a guest. Every account routes to the single
//! Assistant, so no account stores an agent-role selection.
//!
//! ## Personal workspaces
//!
//! When `selected_workspace` is NULL, the account has a personal workspace at
//! `~/.mahbot/userspaces/<name>/`. It is NOT registered in the `workspaces` table —
//! computed on the fly. Personal workspaces have no board pipeline, no
//! maintainer, no diagnostics discovery.

use crate::Role;
use crate::Workspace;
use crate::WorkspaceStatus;
use crate::db::{self, TxGuard};
use crate::git::commands::run_git_output;
use anyhow::Result;
use std::path::PathBuf;
use tracing::warn;

/// Sentinel that `extract_sender_user_name` (src/channels/telegram.rs)
/// substitutes for Telegram senders without an @username. It is no nickname
/// anyone can hold: binding it is refused, and it never matches a sender — a
/// sender without a nickname is matched by their numeric id, or not at all.
pub(crate) const TELEGRAM_UNKNOWN_SENTINEL: &str = "unknown";

/// Telegram's own non-person identities, refused both as bindings and as
/// senders: the anonymous group administrator (1087968824), a message
/// attributed to a channel (136817688), and a channel post auto-forwarded into
/// a linked discussion group (777000) — the same number as Telegram's own
/// service account.
const TELEGRAM_SERVICE_IDS: [&str; 3] = ["1087968824", "136817688", "777000"];

/// The nicknames those identities carry (777000 has none). Telegram usernames
/// are case-insensitive, so the comparison is too.
const TELEGRAM_SERVICE_NICKNAMES: [&str; 2] = ["groupanonymousbot", "channel_bot"];

/// The app's own admin account — the single owner identity the desktop GUI
/// always acts as. Seeded idempotently by [`UserStore::ensure_admin_user`].
pub(crate) const ADMIN_USER_NAME: &str = "admin";

/// How a Telegram binding identifier is written: as a number or as a nickname.
enum TelegramKey {
    /// Canonical form: digits, no leading `+`, no leading zeros.
    Number(String),
    /// Byte-for-byte as entered or stored.
    Nickname(String),
}

/// The digits of a number-shaped value: after trimming, a single optional
/// leading `+`, nothing but digits.
fn telegram_digits(value: &str) -> Option<&str> {
    let digits = value.trim();
    let digits = digits.strip_prefix('+').unwrap_or(digits);
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())).then_some(digits)
}

/// Classify a Telegram binding identifier by its shape.
///
/// A Telegram nickname can never be a bare number, so the shape decides: a
/// number-shaped value is a numeric id, canonicalized here so the same number
/// written differently is one and the same value. Anything else keeps its
/// nickname meaning. `None` only for a value that is number-shaped yet nothing
/// but zeros: no person's number.
fn telegram_key(value: &str) -> Option<TelegramKey> {
    let Some(digits) = telegram_digits(value) else {
        return Some(TelegramKey::Nickname(value.to_string()));
    };
    let canonical = digits.trim_start_matches('0');
    (!canonical.is_empty()).then(|| TelegramKey::Number(canonical.to_string()))
}

/// The canonical form of a numeric id — digits with the leading zeros removed.
/// `None` when the value is not number-shaped or is nothing but zeros.
fn canonical_telegram_number(value: &str) -> Option<String> {
    match telegram_key(value) {
        Some(TelegramKey::Number(number)) => Some(number),
        _ => None,
    }
}

/// Normalize a value entered as a Telegram binding and refuse what can never
/// belong to a person: the reserved `unknown` sentinel, Telegram's own service
/// identities under either spelling, and a value that is nothing but zeros.
/// A single leading `@` is stripped as it always has been; what remains decides
/// the kind, and a number is kept as the number itself.
fn normalize_telegram_binding(value: &str) -> anyhow::Result<String> {
    let trimmed = value.trim();
    let trimmed = trimmed.strip_prefix('@').unwrap_or(trimmed).trim();
    match telegram_key(trimmed) {
        Some(TelegramKey::Number(id)) => {
            if is_telegram_service_id(&id) {
                anyhow::bail!("'{id}' is a Telegram service identity and cannot be bound");
            }
            Ok(id)
        }
        Some(TelegramKey::Nickname(nickname)) => {
            if nickname.is_empty() {
                anyhow::bail!("Telegram binding is empty");
            }
            if nickname == TELEGRAM_UNKNOWN_SENTINEL {
                anyhow::bail!("'unknown' is a reserved Telegram nickname and cannot be bound");
            }
            if is_telegram_service_nickname(&nickname) {
                anyhow::bail!("'@{nickname}' is a Telegram service identity and cannot be bound");
            }
            Ok(nickname)
        }
        None => anyhow::bail!("a Telegram id of nothing but zeros is not a person's number"),
    }
}

/// Whether a canonical numeric id is one of Telegram's own service identities.
pub(crate) fn is_telegram_service_id(id: &str) -> bool {
    TELEGRAM_SERVICE_IDS.contains(&id)
}

/// Whether a nickname is the one a Telegram service identity carries.
fn is_telegram_service_nickname(nickname: &str) -> bool {
    let lowercase = nickname.to_ascii_lowercase();
    TELEGRAM_SERVICE_NICKNAMES.contains(&lowercase.as_str())
}

/// Whether a stored Telegram identifier is written as a number — digits alone,
/// however spelled — rather than as a nickname. A stored value that is
/// number-shaped but nothing but zeros is no person's number and binds nobody;
/// it is still written as a number, so it is shown as an id and never as `@0`.
fn telegram_identifier_is_number(identifier: &str) -> bool {
    telegram_digits(identifier).is_some()
}

/// Name a Telegram binding in a message a person reads: a nickname keeps the
/// `@` it is entered with, a number is labelled as an id — never in nickname
/// form.
pub(crate) fn describe_telegram_binding(identifier: &str) -> String {
    if telegram_identifier_is_number(identifier) {
        format!("id {}", identifier.trim())
    } else {
        format!("@{identifier}")
    }
}

/// The Settings → Users row's label for an account's Telegram binding: the
/// nickname exactly as it has always been shown, a number labelled as an id so
/// it can never read as a nickname.
pub(crate) fn settings_binding_label(identifier: &str) -> String {
    if telegram_identifier_is_number(identifier) {
        describe_telegram_binding(identifier)
    } else {
        identifier.to_string()
    }
}

crate::define_store! {
    /// Global account store.
    pub static USER_STORE: UserStore,
    post_open = ensure_admin_user,
    expect = "USER_STORE not initialized — call init_all_stores() first",
}

// ── Column index constants ──────────────────────────────────

// users table (3-column SELECT: name, selected_workspace, granted_tools)
crate::columns! {
    USERS_COLUMNS [USERS] {
        NAME                => "name",
        SELECTED_WORKSPACE  => "selected_workspace",
        GRANTED_TOOLS       => "granted_tools",
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
    /// Auto-create the admin account if this is a fresh database.
    ///
    /// Runs idempotently from both [`crate::db::init_all_stores`] (production,
    /// on the shared consolidated connection) and each isolated user store open.
    /// This is the ONLY path that may create the reserved admin name; every
    /// user-facing path refuses it ([`validate_new_user_name`]).
    pub(crate) async fn ensure_admin_user(&self) -> Result<()> {
        if !self.user_exists(ADMIN_USER_NAME).await? {
            self.add_user(ADMIN_USER_NAME).await?;
        }
        Ok(())
    }

    // ── Account CRUD ─────────────────────────────────────────────

    /// Create an account: the seeding path for the admin, and the path
    /// user-facing callers use for guests — those must refuse the reserved
    /// admin name first ([`validate_new_user_name`]). Also creates the personal
    /// workspace directory under `~/.mahbot/userspaces/<name>/` with `git init`
    /// (non-fatal on failure). Idempotent — re-adding an existing account
    /// preserves its stored preferences.
    pub async fn add_user(&self, name: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR IGNORE INTO users (name) VALUES (?1)",
                db::params![name],
            )
            .await?;
        ensure_personal_workspace(name).await;
        Ok(())
    }

    /// Delete an account and all their child rows (channel bindings). Channel
    /// bindings are an account's only child rows, so nothing else needs a
    /// cascade.
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

    /// Fetch a single nullable column from the account's row, if it exists.
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

    /// Get the selected workspace name for an account, if any.
    async fn get_selected_workspace_name(&self, user_name: &str) -> Result<Option<String>> {
        self.user_column("selected_workspace", user_name).await
    }

    /// Whether an account row with this name exists.
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
    /// identifier (a Telegram nickname or canonical numeric id). Uses
    /// INSERT OR REPLACE — a `(channel, identifier)` pair already bound to
    /// another user is silently reassigned. User-facing Telegram bind paths go
    /// through [`UserStore::bind_telegram`], which validates first (reserved
    /// sentinel + service identities + at-most-one + anti-steal guards).
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

    /// Validate a Telegram binding value on a user-facing path and return the
    /// normalized identifier ready for [`UserStore::bind_channel`].
    ///
    /// Refuses, in order: a value that can never belong to a person
    /// ([`normalize_telegram_binding`]); an account that already holds a
    /// Telegram binding (an account holds at most one, so switching ways is
    /// remove-then-attach — re-attaching even the identical value is refused
    /// like any other attach); and a value already bound to another account
    /// (anti-steal: `bind_channel` is INSERT OR REPLACE and would silently
    /// reassign it). The at-most-one rule is Telegram-only — other channels
    /// keep storing what they store today.
    pub async fn validate_telegram_bind(&self, user_name: &str, value: &str) -> Result<String> {
        let identifier = normalize_telegram_binding(value)?;
        if let Some(existing) = self.telegram_binding(user_name).await? {
            anyhow::bail!(
                "'{user_name}' already has a Telegram binding ({}) — remove it on the \
                 Settings → Users page first, then attach the new one",
                describe_telegram_binding(&existing)
            );
        }
        // The at-most-one check above proved this account holds no Telegram row
        // at all, so an owner found here is always someone else.
        if let Some(owner) = self.telegram_binding_owner(&identifier).await? {
            anyhow::bail!(
                "{} is already bound to user '{owner}'",
                describe_telegram_binding(&identifier)
            );
        }
        Ok(identifier)
    }

    /// Attach a Telegram binding to an account: validate, then record it.
    /// Returns the normalized identifier (the number itself for a numeric id).
    pub async fn bind_telegram(&self, user_name: &str, value: &str) -> Result<String> {
        let identifier = self.validate_telegram_bind(user_name, value).await?;
        self.attach_telegram_binding(user_name, &identifier).await?;
        Ok(identifier)
    }

    /// [`UserStore::bind_telegram`] for the admin, whose own bind has always
    /// also recorded a nickname as its starting address — kept for
    /// compatibility with how that address has always been stored, not because
    /// a nickname is sendable (it is not a Telegram chat id, so it only starts
    /// working once the admin writes to the bot). A number is a real address on
    /// every path; no other path records a nickname address.
    pub async fn bind_telegram_for_admin(&self, value: &str) -> Result<String> {
        let identifier = self.bind_telegram(ADMIN_USER_NAME, value).await?;
        if !telegram_identifier_is_number(&identifier) {
            self.update_channel_contact("telegram", &identifier, &identifier)
                .await?;
        }
        Ok(identifier)
    }

    /// Record an already-validated Telegram `identifier` as `user_name`'s
    /// binding, and — when it is a number — that number as the binding's own
    /// delivery address. A number is a valid private-chat address, so it is
    /// known from the start (Telegram still forbids a bot to open the
    /// conversation until that person writes); a nickname has no address until
    /// its owner writes, so nothing is recorded for it here.
    pub async fn attach_telegram_binding(&self, user_name: &str, identifier: &str) -> Result<()> {
        self.bind_channel(user_name, "telegram", identifier).await?;
        if telegram_identifier_is_number(identifier) {
            self.update_channel_contact("telegram", identifier, identifier)
                .await?;
        }
        Ok(())
    }

    /// The account's Telegram binding identifier, if it holds one. An account
    /// holds at most one; a legacy account with several reports the first.
    pub(crate) async fn telegram_binding(&self, user_name: &str) -> Result<Option<String>> {
        Ok(self
            .get_user_channels(user_name)
            .await?
            .into_iter()
            .find(|c| c.channel == "telegram")
            .map(|c| c.identifier))
    }

    /// The account that already holds `identifier` — under any spelling of the
    /// same number — if any. Uses the same matching as the sender path, so a
    /// value that differs only in spelling is recognised as the same value.
    async fn telegram_binding_owner(&self, identifier: &str) -> Result<Option<String>> {
        let matched = match telegram_key(identifier) {
            Some(TelegramKey::Number(number)) => {
                self.resolve_telegram_user(Some(&number), None).await?
            }
            _ => self.resolve_telegram_user(None, Some(identifier)).await?,
        };
        Ok(matched.map(|matched| matched.user_name))
    }

    /// Resolve a Telegram sender to the account that owns them, by the sender's
    /// OWN identity — never by the conversation the event arrived in: the
    /// numeric id first (a number belongs to that one person and survives a
    /// rename or a dropped nickname), then the nickname, which decides only
    /// when no number matches. `nickname` is `None` for a sender Telegram
    /// reports without one. Returns the matched binding's stored identifier
    /// alongside the account, so the caller refreshes the address of the
    /// binding that actually matched.
    ///
    /// A stored value is matched by the shape it was always meant to have, so
    /// an existing binding spelled `00123` is the number 123; one that equals a
    /// service identity authorizes nobody.
    pub async fn resolve_telegram_user(
        &self,
        numeric_id: Option<&str>,
        nickname: Option<&str>,
    ) -> Result<Option<TelegramMatch>> {
        let numeric_id = numeric_id.and_then(canonical_telegram_number);
        let rows = self
            .conn
            .query(
                "SELECT user_name, identifier FROM user_channels WHERE channel = 'telegram'",
                db::params![],
            )
            .await?;
        let mut nickname_match = None;
        for row in &rows {
            let user_name: String = row.get(0)?;
            let identifier: String = row.get(1)?;
            match telegram_key(&identifier) {
                // A stored service identity authorizes nobody.
                Some(TelegramKey::Number(number)) if !is_telegram_service_id(&number) => {
                    if numeric_id.as_deref() == Some(number.as_str()) {
                        return Ok(Some(TelegramMatch {
                            user_name,
                            identifier,
                        }));
                    }
                }
                // The reserved sentinel is never a match: a binding with it
                // authorizes nobody.
                Some(TelegramKey::Nickname(nick))
                    if nick != TELEGRAM_UNKNOWN_SENTINEL
                        && nickname_match.is_none()
                        && nickname == Some(nick.as_str()) =>
                {
                    nickname_match = Some(TelegramMatch {
                        user_name,
                        identifier,
                    });
                }
                _ => {}
            }
        }
        Ok(nickname_match)
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

    /// Convert a `users` table row into a [`UserRecordEntry`], loading channel
    /// bindings.
    async fn user_entry_from_row(&self, row: &db::Row) -> Result<UserRecordEntry> {
        let name: String = row.get(COL_USERS_NAME)?;
        // A failed channel read must not render as "no binding": carry the
        // failure beside the record so the GUI can surface it instead of
        // offering to bind.
        let (channels, channels_error) = match self.get_user_channels(&name).await {
            Ok(channels) => (channels, None),
            Err(e) => (Vec::new(), Some(e.to_string())),
        };
        Ok(UserRecordEntry {
            record: UserRecord {
                name,
                selected_workspace: row.get::<Option<String>>(COL_USERS_SELECTED_WORKSPACE)?,
                granted_tools: parse_grants(row.get::<Option<String>>(COL_USERS_GRANTED_TOOLS)?),
                channels,
            },
            channels_error,
        })
    }

    // ── Lookup / listing ──────────────────────────────────────

    /// Shared listing body: run `suffix` (everything after `FROM users`) and
    /// collect one [`UserRecordEntry`] per row with channel bindings.
    async fn list_users_where(
        &self,
        suffix: &str,
        params: impl db::IntoParams + Send + 'static,
    ) -> Result<Vec<UserRecordEntry>> {
        let sql = format!("SELECT {USERS_COLUMNS} FROM users {suffix}");
        let rows = self.conn.query(&sql, params).await?;
        let mut users = Vec::with_capacity(rows.len());
        for row in rows {
            users.push(self.user_entry_from_row(&row).await?);
        }
        Ok(users)
    }

    /// Find a single account by exact name, returning their full record with
    /// channel bindings. Returns `None` if no such account exists.
    pub async fn find_by_name(&self, user_name: &str) -> Result<Option<UserRecord>> {
        self.list_users_where("WHERE name = ?1", db::params![user_name])
            .await
            .map(|users| users.into_iter().next().map(|entry| entry.record))
    }

    /// List every account with the outcome of each account's channel-binding read.
    pub async fn list_users(&self) -> Result<Vec<UserRecordEntry>> {
        self.list_users_where("", db::params![]).await
    }

    /// Set the account's selected shared workspace, beside
    /// [`Self::set_image_gen_model`] / [`Self::set_video_model`]. `None` clears
    /// the column — the account's personal workspace (computed on the fly);
    /// `Some(name)` selects that shared workspace.
    pub(crate) async fn set_selected_workspace(
        &self,
        name: &str,
        workspace_name: Option<&str>,
    ) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        upsert_user_column(&tx, name, "selected_workspace", workspace_name).await?;
        tx.commit().await?;
        Ok(())
    }

    /// Set the user's active image-generation model (their Telegram picker
    /// choice). An empty/whitespace value resolves to the default at read time.
    pub async fn set_image_gen_model(&self, name: &str, model: &str) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        upsert_user_column(&tx, name, "image_gen_model", Some(model)).await?;
        tx.commit().await?;
        Ok(())
    }

    /// Set the user's active video model (their Telegram picker choice).
    /// An empty/whitespace value resolves to the default at read time.
    pub async fn set_video_model(&self, name: &str, model: &str) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        upsert_user_column(&tx, name, "video_model", Some(model)).await?;
        tx.commit().await?;
        Ok(())
    }

    // ── Custom tool grants ────────────────────────────────────

    /// The custom tools granted to `user_name`, byte-sorted; empty when the
    /// user has no grants or no row.
    pub(crate) async fn get_grants(&self, user_name: &str) -> Result<Vec<String>> {
        Ok(parse_grants(
            self.user_column("granted_tools", user_name).await?,
        ))
    }

    /// Grant the custom tool `tool` to `user_name`. Granting is idempotent, and
    /// a grant never inserts a users row (a ghost user would appear in the GUI),
    /// so the caller reports the missing user itself.
    pub(crate) async fn add_grant(&self, user_name: &str, tool: &str) -> Result<GrantChange> {
        self.update_grants(user_name, |grants| {
            if grants.iter().any(|g| g == tool) {
                return false;
            }
            grants.push(tool.to_string());
            true
        })
        .await
    }

    /// Revoke the custom tool `tool` from `user_name`. Idempotent: an absent
    /// grant is a silent no-op. The result tells the caller which of the three
    /// cases it hit, so it can report a missing user instead of a false
    /// confirmation — a revoke never inserts a row either.
    pub(crate) async fn remove_grant(&self, user_name: &str, tool: &str) -> Result<GrantChange> {
        self.update_grants(user_name, |grants| {
            let before = grants.len();
            grants.retain(|g| g != tool);
            grants.len() != before
        })
        .await
    }

    /// Every user with at least one granted custom tool, ordered by user name.
    /// The empty-list filter runs in Rust rather than SQL: `[]` and an
    /// unparseable value are both empty sets, and the users table is tiny.
    pub(crate) async fn list_grants(&self) -> Result<Vec<(String, Vec<String>)>> {
        let rows = self
            .conn
            .query_map_strict(
                "SELECT name, granted_tools FROM users ORDER BY name",
                db::params![],
                |row| -> Result<(String, Vec<String>)> {
                    Ok((
                        row.get::<String>(0)?,
                        parse_grants(row.get::<Option<String>>(1)?),
                    ))
                },
            )
            .await?;
        Ok(rows
            .into_iter()
            .filter(|(_, grants)| !grants.is_empty())
            .collect())
    }

    /// Read-modify-write the `users.granted_tools` set for `user_name` in one
    /// transaction. `mutate` edits the parsed, sorted set; returning `false`
    /// skips the write (the caller's idempotent no-op). A missing row is left
    /// untouched, never inserted — the caller reports that from the returned
    /// [`GrantChange`] instead.
    async fn update_grants(
        &self,
        user_name: &str,
        mutate: impl FnOnce(&mut Vec<String>) -> bool,
    ) -> Result<GrantChange> {
        let tx = self.conn.begin_tx().await?;
        let rows = tx
            .query(
                "SELECT granted_tools FROM users WHERE name = ?1",
                db::params![user_name],
            )
            .await?;
        let Some(row) = rows.first() else {
            tx.rollback().await?;
            return Ok(GrantChange::NoUser);
        };
        let mut grants = parse_grants(row.get::<Option<String>>(0)?);
        if !mutate(&mut grants) {
            tx.rollback().await?;
            return Ok(GrantChange::Unchanged);
        }
        // The closures only add or remove one name, so a push is the one thing
        // that can break the parsed set's sorted order.
        grants.sort();
        tx.execute(
            "UPDATE users SET granted_tools = ?1 WHERE name = ?2",
            db::params![serde_json::to_string(&grants)?, user_name],
        )
        .await?;
        tx.commit().await?;
        Ok(GrantChange::Changed)
    }
}

/// What a grants read-modify-write did.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum GrantChange {
    /// No such `users` row — nothing was written, and none is inserted.
    NoUser,
    /// The row exists and already held the requested state — nothing written.
    Unchanged,
    /// The row exists and its grant set changed.
    Changed,
}

/// Parse the `users.granted_tools` JSON array column. A NULL, empty or
/// unparseable value yields an empty list (fail-closed: unknown grants are
/// no grants).
fn parse_grants(raw: Option<String>) -> Vec<String> {
    let mut grants = raw
        .and_then(|raw| serde_json::from_str::<Vec<String>>(&raw).ok())
        .unwrap_or_default();
    grants.sort();
    grants.dedup();
    grants
}

/// Upsert a single user column within an existing transaction.
///
/// The `field` parameter MUST be a compile-time string literal to prevent SQL injection.
async fn upsert_user_column(
    tx: &TxGuard<'_>,
    name: &str,
    field: &str,
    value: Option<&str>,
) -> Result<()> {
    tx.upsert_row(
        &format!("UPDATE users SET {field} = ?1 WHERE name = ?2"),
        || db::params![value, name],
        &format!(
            "INSERT INTO users (name, {field}) VALUES (?1, ?2) \
             ON CONFLICT(name) DO NOTHING"
        ),
        db::params![name, value],
    )
    .await?;
    Ok(())
}

// ── UserRecord ────────────────────────────────────────────────

/// A full account row, e.g. returned by [`UserStore::find_by_name`].
#[derive(Debug, Clone)]
pub struct UserRecord {
    /// The canonical account name. When it is [`ADMIN_USER_NAME`], this is the
    /// admin account.
    pub name: String,
    /// Selected shared workspace name, NULL = personal workspace.
    pub selected_workspace: Option<String>,
    /// The custom tools granted to this account (the `users.granted_tools` JSON
    /// array column), byte-sorted; empty when none. A grant recorded for the
    /// admin is carried and listed like any other; only the account-management
    /// card omits it, since the whole catalogue is implicitly available to the
    /// admin.
    pub granted_tools: Vec<String>,
    /// Channel bindings for this account (Telegram, etc.).
    pub channels: Vec<ChannelBinding>,
}

/// One `users` row together with the outcome of its channel-binding read. The
/// domain [`UserRecord`] is channel-empty both when the account has no bindings
/// and when the read failed, so a surface that renders bindings needs the
/// outcome to tell the two apart; it travels beside the record, not inside it.
#[derive(Debug, Clone)]
pub struct UserRecordEntry {
    pub record: UserRecord,
    /// `Some` when the channel-binding read failed; `record.channels` is then
    /// empty.
    pub channels_error: Option<String>,
}

/// The one admin test: an account is the admin exactly when its name is
/// [`ADMIN_USER_NAME`] — no stored token, flag or kind ever marks one. Callers
/// in a sync context (GUI state) apply it to a name they already hold; a bare,
/// untrusted name goes through [`is_admin`], which backs the same test with the
/// account's row.
#[must_use]
pub(crate) fn is_admin_name(name: &str) -> bool {
    name == ADMIN_USER_NAME
}

/// Refuse the reserved admin name on a user-facing account-creation path: only
/// [`UserStore::ensure_admin_user`] may create the admin account.
pub(crate) fn validate_new_user_name(name: &str) -> Result<()> {
    if is_admin_name(name) {
        anyhow::bail!(
            "'{ADMIN_USER_NAME}' is the admin account — only guest accounts can be created"
        );
    }
    Ok(())
}

/// A single channel binding for an account.
#[derive(Debug, Clone)]
pub struct ChannelBinding {
    /// The channel type (e.g. "telegram").
    pub channel: String,
    /// The channel-specific identifier (a Telegram nickname, or a numeric id).
    pub identifier: String,
    /// Routing address for replies on this channel (e.g. Telegram chat_id:thread_id).
    pub reply_target: Option<String>,
}

/// The account a Telegram sender belongs to, with the binding that matched it.
#[derive(Debug, Clone)]
pub struct TelegramMatch {
    /// The account's canonical name.
    pub user_name: String,
    /// The stored identifier of the binding that matched — the binding whose
    /// delivery address follows this sender's conversation.
    pub identifier: String,
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

/// Whether `user_name` may be used to resolve a personal-workspace path:
/// non-empty, no path separators, not a dot entry. Everything that resolves
/// `userspaces/<user>` for listing or writing guards here — a synthetic or
/// path-bearing name must never address the userspaces root or another
/// user's directory.
#[must_use]
pub(crate) fn is_valid_personal_user_name(user_name: &str) -> bool {
    !user_name.trim().is_empty()
        && !user_name.contains(['/', '\\'])
        && !matches!(user_name, "." | "..")
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
///
/// User-facing resolution must go through [`resolve_selected_workspace_name`]
/// (which applies the admin-only membership clamp); the remaining callers of
/// this raw read are admin-gated Telegram admin-command paths.
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
/// per-tool-call/per-Assistant-turn hot path, so no full `UserRecord` load).
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

/// Resolve the account's admin-aware selected workspace name — THE single
/// admin-aware resolution primitive, shared by message routing and the GUI
/// (boot restore, reverse-sync, merge partner).
///
/// Workspace membership is admin-only: the admin gets their stored selection
/// (a legacy stored personal value normalizes to the canonical
/// `personal:{user}` name); a guest ALWAYS yields their `personal:{user}`
/// name, never a shared workspace (a shared selection is clamped, with a
/// warning — defense in depth against rogue re-attachment). A NULL
/// `selected_workspace` and a missing account row both yield the canonical
/// `personal:{user}` name; `None` is reserved for a failed read (warned) — the
/// caller then applies its own personal default.
pub(crate) async fn resolve_selected_workspace_name(user_name: &str) -> Option<String> {
    let stored = match store().get_selected_workspace_name(user_name).await {
        Ok(stored) => stored,
        Err(e) => {
            warn!(user_name = %user_name, error = %e, "Failed to read selected workspace");
            return None;
        }
    };
    if is_admin_name(user_name) {
        return Some(match stored {
            Some(ws) if !is_personal_workspace(&ws) => ws,
            _ => personal_workspace_name(user_name),
        });
    }
    if let Some(ws) = stored
        && !is_personal_workspace(&ws)
    {
        warn!(
            user_name = %user_name,
            workspace = %ws,
            "guest has a shared selected_workspace — clamping to their personal workspace"
        );
    }
    Some(personal_workspace_name(user_name))
}

/// Get the current active workspace for an account, admin-aware.
///
/// Resolves through [`resolve_selected_workspace_name`]: the admin gets their
/// stored workspace (shared or personal), a guest always resolves to their
/// personal workspace. `None` (a read failure) yields the personal workspace.
async fn get_workspace(user_name: &str) -> Result<Option<Workspace>> {
    match resolve_selected_workspace_name(user_name).await {
        Some(ws_name) => resolve_workspace(&ws_name).await,
        None => Ok(Some(personal_workspace_struct(user_name))),
    }
}

/// Resolve a workspace by name, synthesizing a personal workspace
/// (`personal:{user}`) when the name is not in the `workspaces` table.
///
/// Personal workspaces are NOT stored in the table — they live at
/// `~/.mahbot/userspaces/<user>/` and are constructed on the fly as ephemeral
/// [`Workspace`] structs.
///
/// Returns `Ok(Some(ws))` when the workspace is found or constructed,
/// `Ok(None)` when the workspace genuinely does not exist (and is not a
/// personal workspace), `Err` on database errors.
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

/// The user-facing role pinned to the account's personal workspace.
#[must_use]
fn is_pinned_role(role: Role) -> bool {
    matches!(role, Role::Assistant)
}

/// Execution-time invariant enforcer: a pinned role (Assistant) must never
/// run outside the envelope user's OWN personal workspace, regardless of what
/// a producer stored — a non-personal workspace is re-pinned to
/// `personal:{user}`, and a personal workspace belonging to a different user
/// is re-pinned to the envelope user's own. Returns `None` when the routing
/// must be refused outright: a pinned role with an empty `user_name` has no
/// personal identity to pin to, and running it unpinned is never acceptable,
/// so callers reject the job loudly instead. Non-pinned roles pass through
/// unchanged.
#[must_use]
pub(crate) fn enforce_personal_pinning(
    role: Role,
    workspace_name: &str,
    user_name: &str,
) -> Option<String> {
    if !is_pinned_role(role) {
        return Some(workspace_name.to_string());
    }
    if user_name.is_empty() {
        return None;
    }
    Some(personal_workspace_name(user_name))
}

/// Resolve the [`Workspace`] an agent role actually operates in: the Assistant
/// always works in the user's personal workspace regardless of the selected
/// workspace, giving path-dependent callers (enrichment uploads,
/// generated-media writes) the personal workspace's filesystem path. Other
/// roles pass through unchanged.
/// An empty `user_name` disables pinning (no personal identity to pin to),
/// so callers must pass a resolvable user (the voice path passes
/// [`ADMIN_USER_NAME`]).
/// Accepted user decision (no migration): media written before pinning to a
/// project workspace's `uploads/`/`generated/` stays there and is no longer
/// reachable by Assistant tools (e.g. video_edit path confinement).
#[must_use]
pub fn effective_workspace_for_role(role: Role, ws: Workspace, user_name: &str) -> Workspace {
    if !is_pinned_role(role) {
        return ws;
    }
    // Pass-through asymmetry with `enforce_personal_pinning`: an already-personal
    // name stays as-is, even when it names another user, whereas the enforcer
    // re-pins that case to the envelope user's own workspace.
    if is_personal_workspace(&ws.name) {
        return ws;
    }
    if user_name.is_empty() {
        tracing::error!(
            role = %role.as_str(),
            workspace = %ws.name,
            "Personal-workspace pin bypassed: pinned role with empty user_name — caller must pass a resolvable user"
        );
        return ws;
    }
    personal_workspace_struct(user_name)
}

/// Resolve the (role, workspace) an account's messages route to and their
/// session lives in — the same resolution as routing, so ClearChat and
/// Telegram /clear always clear the actual recipient: the account's selected
/// workspace with Assistant pinning applied, and the Assistant role, every
/// account's only role, which the session-key builders consume alongside it.
pub async fn resolve_session_target(user_name: &str) -> (Role, Workspace) {
    let ws = resolve_workspace_for_user_name(user_name).await;
    (
        Role::Assistant,
        effective_workspace_for_role(Role::Assistant, ws, user_name),
    )
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

/// The admin test backed by an account store: `user_name` is the admin exactly
/// when it is the admin's name ([`is_admin_name`]) AND `store` holds its row.
/// `None` (no store) and a failed row read both answer false, so the answer
/// never defaults to admin rights.
async fn is_admin_in(store: Option<&UserStore>, user_name: &str) -> bool {
    if !is_admin_name(user_name) {
        return false;
    }
    let Some(store) = store else {
        return false;
    };
    store.user_exists(user_name).await.unwrap_or_else(|e| {
        warn!(error = %e, user_name, "Failed to read the admin account");
        false
    })
}

/// Whether `user_name` is the admin account — [`is_admin_in`] bound to the
/// process-global account store.
pub async fn is_admin(user_name: &str) -> bool {
    is_admin_in(USER_STORE.get(), user_name).await
}

/// The custom tools granted to `user_name`, byte-sorted. A read that fails
/// yields no grants, which fails every access decision closed.
pub(crate) async fn granted_tools(user_name: &str) -> Vec<String> {
    match USER_STORE.get() {
        Some(store) => match store.get_grants(user_name).await {
            Ok(grants) => grants,
            Err(e) => {
                tracing::warn!(error = %e, user_name, "Failed to read granted custom tools");
                Vec::new()
            }
        },
        None => Vec::new(),
    }
}

/// Render a custom-tool grant list for display: `a, b`, or `none` when empty.
#[must_use]
pub(crate) fn format_grants(grants: &[String]) -> String {
    if grants.is_empty() {
        "none".to_string()
    } else {
        grants.join(", ")
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

    /// Initialize a test account store with known accounts and channel bindings.
    /// Safe to call multiple times — delegates to [`init_test_stores`] to
    /// ensure all global stores are initialized, then supplements
    /// USER_STORE with telegram-specific accounts and channel bindings.
    pub(crate) async fn init_test_store() {
        // Ensure all global stores are initialized (idempotent OnceCell).
        crate::util::test::init_test_stores().await;

        // Supplement USER_STORE with telegram-specific test accounts and
        // bindings.  Both `add_user` (INSERT OR IGNORE) and `bind_channel`
        // (INSERT OR REPLACE) are idempotent.
        if let Some(store) = USER_STORE.get() {
            store
                .add_user("alice")
                .await
                .expect("failed to add alice to test USER_STORE");
            store
                .add_user("bob")
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

    /// Run `body`, restoring the shared seeded admin row's `selected_workspace`
    /// to the value it had on entry — including when `body` panics, so a failed
    /// assertion cannot leak a test-only workspace into the sibling tests
    /// serialized on `gui_admin_workspace`, which read this shared row.
    ///
    /// Unwinding by hand rather than with a `Drop` guard: the restore awaits the
    /// store, which no destructor can do.
    pub(crate) async fn with_admin_workspace_restored<F>(body: F)
    where
        F: std::future::Future<Output = ()>,
    {
        use futures_util::FutureExt as _;

        let store = store();
        let previous = get_raw_selected_workspace(ADMIN_USER_NAME)
            .await
            .expect("read admin selected_workspace");
        let outcome = std::panic::AssertUnwindSafe(body).catch_unwind().await;
        store
            .set_selected_workspace(ADMIN_USER_NAME, previous.as_deref())
            .await
            .expect("restore admin selected_workspace");
        if let Err(panic) = outcome {
            std::panic::resume_unwind(panic);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The admin test is the account's name, backed by the account's own row:
    /// a guest name, a name with no row, and a missing store are never the
    /// admin.
    #[tokio::test]
    async fn admin_test_is_the_account_name_backed_by_its_row() {
        let (store, _dir) = crate::open_test_store!(UserStore, "user");
        assert!(
            is_admin_in(Some(&store), ADMIN_USER_NAME).await,
            "the seeded admin account is the admin"
        );

        store.add_user("guest").await.unwrap();
        assert!(
            !is_admin_in(Some(&store), "guest").await,
            "a guest account is not the admin"
        );

        store.delete_user(ADMIN_USER_NAME).await.unwrap();
        assert!(
            !is_admin_in(Some(&store), ADMIN_USER_NAME).await,
            "the admin's name without an account row is not the admin"
        );
        assert!(
            !is_admin_in(None, ADMIN_USER_NAME).await,
            "a missing account store is not the admin"
        );
    }

    #[tokio::test]
    async fn delete_user_removes_channel_rows() {
        crate::util::test::init_test_stores().await;
        let store = store();

        // add_user creates the account; deleting it removes the channel
        // bindings and the account row — the only child rows an account has.
        store.add_user("doomed").await.unwrap();
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

    /// Custom-tool grants round-trip through the `users.granted_tools` JSON
    /// column: byte-sorted, deduped, idempotent, and never a reason to insert a
    /// user row.
    #[tokio::test]
    async fn grant_storage_round_trip() {
        crate::util::test::init_test_stores().await;
        let store = store();
        let user = "grant_round_trip";
        store.add_user(user).await.unwrap();
        assert!(
            store.get_grants(user).await.unwrap().is_empty(),
            "a fresh user has no grants"
        );

        // Out-of-order grants land byte-sorted in the row's JSON column.
        store.add_grant(user, "zeta").await.unwrap();
        store.add_grant(user, "alpha").await.unwrap();
        let stored: Option<String> = store
            .conn
            .query_row(
                "SELECT granted_tools FROM users WHERE name = ?1",
                db::params![user],
                |row| row.get(0),
            )
            .await
            .unwrap();
        assert_eq!(stored.as_deref(), Some(r#"["alpha","zeta"]"#));

        // Granting twice is a no-op; the parsed record carries the grants.
        assert_eq!(
            store.add_grant(user, "alpha").await.unwrap(),
            GrantChange::Unchanged,
            "re-granting a tool the user already holds changes nothing"
        );
        assert_eq!(
            store.get_grants(user).await.unwrap(),
            vec!["alpha".to_string(), "zeta".to_string()]
        );
        assert_eq!(
            store
                .find_by_name(user)
                .await
                .unwrap()
                .unwrap()
                .granted_tools,
            vec!["alpha".to_string(), "zeta".to_string()]
        );

        // Revoking an absent grant — or revoking from a user without a row —
        // is a silent no-op.
        assert_eq!(
            store.remove_grant(user, "absent").await.unwrap(),
            GrantChange::Unchanged,
            "revoking a grant the user does not hold changes nothing"
        );
        assert_eq!(
            store
                .remove_grant("grant_no_such_user", "alpha")
                .await
                .unwrap(),
            GrantChange::NoUser,
            "revoking from a missing user reports that no row exists"
        );
        assert_eq!(
            store.get_grants(user).await.unwrap(),
            vec!["alpha".to_string(), "zeta".to_string()]
        );

        // Granting to a user without a row reports the missing row instead of
        // creating one.
        assert_eq!(
            store.add_grant("grant_ghost_user", "alpha").await.unwrap(),
            GrantChange::NoUser,
            "granting to an unknown user must report that no row exists"
        );
        assert!(
            store
                .find_by_name("grant_ghost_user")
                .await
                .unwrap()
                .is_none(),
            "a failed grant must not leave a ghost user row"
        );

        // list_grants lists every user with grants, ordered by user name — the
        // store is process-wide, so this asserts this user's own entry rather
        // than the list being exactly it.
        let listed = store.list_grants().await.unwrap();
        assert!(
            listed.contains(&(
                user.to_string(),
                vec!["alpha".to_string(), "zeta".to_string()]
            )),
            "got: {listed:?}"
        );
        assert!(
            listed.iter().all(|(_, grants)| !grants.is_empty()),
            "a user with no grants must not be listed: {listed:?}"
        );
    }

    #[test]
    fn normalize_telegram_binding_rules() {
        // Nicknames: trim, one leading '@' stripped, then trimmed again.
        assert_eq!(normalize_telegram_binding("alice").unwrap(), "alice");
        assert_eq!(normalize_telegram_binding("  alice  ").unwrap(), "alice");
        assert_eq!(normalize_telegram_binding("@alice").unwrap(), "alice");
        assert_eq!(normalize_telegram_binding(" @ alice ").unwrap(), "alice");
        // Reserved sentinel is rejected (case-sensitively — "Unknown" stays bindable).
        assert!(normalize_telegram_binding("unknown").is_err());
        assert!(normalize_telegram_binding("   ").is_err());
        assert!(normalize_telegram_binding("@").is_err());
        assert_eq!(normalize_telegram_binding("Unknown").unwrap(), "Unknown");
        // A nickname can never be a bare number, so digits are the number
        // itself — however they are spelled.
        assert_eq!(
            normalize_telegram_binding("123456789").unwrap(),
            "123456789"
        );
        assert_eq!(normalize_telegram_binding("  +00123 ").unwrap(), "123");
        assert_eq!(normalize_telegram_binding("@123").unwrap(), "123");
        assert_eq!(normalize_telegram_binding("alice123").unwrap(), "alice123");
        // Nothing but zeros is no person's number.
        assert!(normalize_telegram_binding("0").is_err());
        assert!(normalize_telegram_binding("000").is_err());
        assert!(normalize_telegram_binding("+000").is_err());
        // Telegram's own service identities, under either spelling.
        assert!(normalize_telegram_binding("777000").is_err());
        assert!(normalize_telegram_binding("1087968824").is_err());
        assert!(normalize_telegram_binding("136817688").is_err());
        assert!(normalize_telegram_binding("@GroupAnonymousBot").is_err());
        assert!(normalize_telegram_binding("channel_bot").is_err());
    }

    #[tokio::test]
    async fn validate_telegram_bind_guards() {
        crate::util::test::init_test_stores().await;
        let store = store();
        store.add_user("bind_guard_owner").await.unwrap();
        store
            .bind_channel("bind_guard_owner", "telegram", "guard_handle")
            .await
            .unwrap();

        // Anti-steal: a different user cannot take over the handle.
        let err = store
            .validate_telegram_bind("bind_guard_other", "guard_handle")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("bind_guard_owner"),
            "error must name the current owner: {err}"
        );

        // At-most-one: even re-attaching the identical value is refused, and
        // the refusal names the binding to remove first.
        let err = store
            .validate_telegram_bind("bind_guard_owner", "@guard_handle")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("@guard_handle") && err.to_string().contains("remove"),
            "error must name the existing binding and how to free the account: {err}"
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

    /// A binding by number is kept as the number itself, carries its own
    /// delivery address from the moment it is attached, and matches a sender by
    /// that number.
    #[tokio::test]
    async fn numeric_telegram_bindings_match_by_sender_identity() {
        let (store, _dir) = crate::open_test_store!(UserStore, "numeric_telegram_bind");
        for name in ["num_owner", "nick_owner", "num_taker"] {
            store.add_user(name).await.unwrap();
        }

        store.bind_telegram("nick_owner", "frank").await.unwrap();
        assert_eq!(
            store.bind_telegram("num_owner", " +00123 ").await.unwrap(),
            "123",
            "the number itself is what is kept"
        );
        assert_eq!(
            store.get_user_channels("num_owner").await.unwrap()[0]
                .reply_target
                .as_deref(),
            Some("123"),
            "a number is its own deliverable address"
        );
        // The same number written differently is the same value: another
        // account cannot take it over by spelling it differently.
        let err = store
            .validate_telegram_bind("num_taker", " 00123 ")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("num_owner"),
            "a differently spelled number is the same binding: {err}"
        );

        // A sender with no nickname is matched by their number alone. (The
        // number outranking a bound nickname is pinned end-to-end in the
        // `channels::telegram` tests.)
        let matched = store
            .resolve_telegram_user(Some("123"), None)
            .await
            .unwrap()
            .expect("the number binds");
        assert_eq!(matched.user_name, "num_owner");
        assert_eq!(matched.identifier, "123");

        // The nickname decides where no number matches.
        let matched = store
            .resolve_telegram_user(Some("124"), Some("frank"))
            .await
            .unwrap()
            .expect("a nickname-only sender matches their nickname");
        assert_eq!(matched.user_name, "nick_owner");

        // An unbound identity matches nobody.
        assert!(
            store
                .resolve_telegram_user(Some("124"), Some("frank_ish"))
                .await
                .unwrap()
                .is_none()
        );

        // A stored spelling of the same number is the same binding.
        store
            .unbind_channel("num_owner", "telegram", "123")
            .await
            .unwrap();
        store
            .bind_channel("nick_owner", "telegram", "00123 ")
            .await
            .unwrap();
        let matched = store
            .resolve_telegram_user(Some("123"), None)
            .await
            .unwrap()
            .expect("a stored non-canonical spelling is matched as the number");
        assert_eq!(matched.user_name, "nick_owner");

        // A stored service identity authorizes nobody.
        store
            .bind_channel("nick_owner", "telegram", "777000")
            .await
            .unwrap();
        assert!(
            store
                .resolve_telegram_user(Some("777000"), None)
                .await
                .unwrap()
                .is_none()
        );

        // A legacy row holding the reserved sentinel stays inert: it is never a
        // nickname match, so a sender reporting that nickname cannot authorize
        // through it.
        store
            .bind_channel("nick_owner", "telegram", "unknown")
            .await
            .unwrap();
        assert!(
            store
                .resolve_telegram_user(None, Some("unknown"))
                .await
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn effective_workspace_pins_assistant_to_personal() {
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
        // Non-pinned roles keep the project workspace; already-personal passes through.
        let kept = effective_workspace_for_role(Role::Engineer, project.clone(), "alice");
        assert_eq!(kept.name, "ws1");
        let already = effective_workspace_for_role(Role::Assistant, personal.clone(), "alice");
        assert_eq!(already.name, "personal:alice");
        // An empty user_name disables pinning — the workspace passes through.
        let unpinned = effective_workspace_for_role(Role::Assistant, project, "");
        assert_eq!(unpinned.name, "ws1");
    }

    #[test]
    fn enforce_personal_pinning_repins_pinned_roles_and_refuses_empty_user() {
        // Pinned roles in a non-personal workspace re-pin to the user's personal.
        assert_eq!(
            enforce_personal_pinning(Role::Assistant, "proj-ws", "alice"),
            Some("personal:alice".to_string())
        );
        // The user's own personal workspace resolves to itself.
        assert_eq!(
            enforce_personal_pinning(Role::Assistant, "personal:alice", "alice"),
            Some("personal:alice".to_string())
        );
        // Another user's personal workspace is re-pinned to the envelope
        // user's own personal workspace (pinned roles are always own-personal).
        assert_eq!(
            enforce_personal_pinning(Role::Assistant, "personal:bob", "alice"),
            Some("personal:alice".to_string())
        );
        // Non-pinned roles pass through unchanged even with an empty user.
        assert_eq!(
            enforce_personal_pinning(Role::Engineer, "proj-ws", ""),
            Some("proj-ws".to_string())
        );
        // Pinned role with empty user has no personal identity — refuse.
        assert_eq!(
            enforce_personal_pinning(Role::Assistant, "proj-ws", ""),
            None
        );
    }

    #[tokio::test]
    async fn resolve_session_target_matches_routing() {
        crate::util::test::init_test_stores().await;
        let user = "home_clear_target";
        let store = store();
        store.add_user(user).await.unwrap();
        crate::util::test::create_test_workspace(
            "/tmp/home_clear_target_ws",
            "ws_home_clear_target",
        )
        .await;

        // Every account routes to the single Assistant, which is pinned to the
        // personal workspace — even when the stored workspace is a project.
        store
            .set_selected_workspace(user, Some("ws_home_clear_target"))
            .await
            .unwrap();
        let (role, ws) = resolve_session_target(user).await;
        assert_eq!(role, Role::Assistant);
        assert_eq!(ws.name, "personal:home_clear_target");
    }

    #[tokio::test]
    #[serial_test::serial(gui_admin_workspace)] // writes the shared seeded admin row
    async fn resolve_selected_workspace_name_is_admin_aware() {
        crate::util::test::init_test_stores().await;
        let store = store();

        crate::users::test_util::with_admin_workspace_restored(async {
            // The admin + a shared selection → keeps the shared workspace.
            store
                .set_selected_workspace(ADMIN_USER_NAME, Some("ws_shared"))
                .await
                .unwrap();
            assert_eq!(
                resolve_selected_workspace_name(ADMIN_USER_NAME).await,
                Some("ws_shared".to_string())
            );
            // The admin + a stored personal value → normalized to the canonical name.
            store
                .set_selected_workspace(ADMIN_USER_NAME, Some("personal:admin"))
                .await
                .unwrap();
            assert_eq!(
                resolve_selected_workspace_name(ADMIN_USER_NAME).await,
                Some("personal:admin".to_string())
            );
            // The admin + NULL selection → personal.
            store
                .set_selected_workspace(ADMIN_USER_NAME, None)
                .await
                .unwrap();
            assert_eq!(
                resolve_selected_workspace_name(ADMIN_USER_NAME).await,
                Some("personal:admin".to_string())
            );
        })
        .await;

        // A guest + a shared selection → clamped to personal.
        store
            .set_selected_workspace("u_shared", Some("ws_shared"))
            .await
            .unwrap();
        assert_eq!(
            resolve_selected_workspace_name("u_shared").await,
            Some("personal:u_shared".to_string())
        );
        // A guest + NULL selection → personal.
        store.set_selected_workspace("u_null", None).await.unwrap();
        assert_eq!(
            resolve_selected_workspace_name("u_null").await,
            Some("personal:u_null".to_string())
        );
        // An account with no row resolves to the same personal default.
        assert_eq!(
            resolve_selected_workspace_name("no_such_account").await,
            Some("personal:no_such_account".to_string())
        );
    }

    #[tokio::test]
    #[serial_test::serial(gui_admin_workspace)] // writes the shared seeded admin row
    async fn resolve_workspace_for_user_name_is_admin_aware() {
        crate::util::test::init_test_stores().await;
        let store = store();
        crate::util::test::create_test_workspace("/tmp/resolve_admin_aware_ws", "ws_admin_aware")
            .await;

        // A guest with a shared selection that exists in the table is still
        // clamped to the personal workspace (never a shared one).
        store
            .set_selected_workspace("u_shared", Some("ws_admin_aware"))
            .await
            .unwrap();
        let ws = resolve_workspace_for_user_name("u_shared").await;
        assert_eq!(ws.name, "personal:u_shared");

        // The admin with a shared selection that exists keeps the shared workspace.
        crate::users::test_util::with_admin_workspace_restored(async {
            store
                .set_selected_workspace(ADMIN_USER_NAME, Some("ws_admin_aware"))
                .await
                .unwrap();
            let ws = resolve_workspace_for_user_name(ADMIN_USER_NAME).await;
            assert_eq!(ws.name, "ws_admin_aware");
        })
        .await;
    }
}
