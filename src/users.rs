//! Per-account identity, workspace preference, and channel bindings, in two tables of the
//! consolidated domain database (`core.db`): `users` (canonical identity — `name`,
//! `selected_workspace`, `granted_tools`) and `user_channels` (bindings mapping a
//! channel+identifier to an account, with that channel's `reply_target` routing address).
//! A Telegram identifier is either a nickname or a numeric id; an account holds at most
//! one Telegram binding, so the two are alternatives.
//!
//! Account identity is independent of any external channel: changing a Telegram
//! `@username` changes nothing. Accounts come from Settings → Users or the assistant's
//! `mahbot_config` tool, and channels are bound explicitly.
//!
//! The admin is the one account named [`ADMIN_USER_NAME`] ([`is_admin_name`]); every other
//! account is a guest. Every account routes to the single Assistant, so no account stores
//! a role selection.
//!
//! A name is the account's identity AND its personal-workspace folder, so a name about to
//! become a new account is refused when an existing account's or the admin's name is the
//! same once case and Unicode composition are ignored ([`NAME_COLLISION_RULE`]). Only a
//! genuinely new name is judged; a stored account keeps its name.
//!
//! A NULL `selected_workspace` means a personal workspace at `~/.mahbot/userspaces/<name>/`
//! — computed on the fly, no `workspaces` row, no board pipeline, maintainer or
//! diagnostics discovery.

use crate::Role;
use crate::Workspace;
use crate::WorkspaceStatus;
use crate::db::{self, TxGuard};
use crate::git::commands::run_git_output;
use crate::util::UnwrapPoison;
use anyhow::{Context as _, Result};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};
use tracing::{error, warn};

/// Sentinel that `extract_sender_user_name` substitutes for senders without an @username.
/// Nobody can hold it: binding it is refused and it never matches a sender (such a sender
/// is matched by numeric id, or not at all).
pub(crate) const TELEGRAM_UNKNOWN_SENTINEL: &str = "unknown";

/// Telegram's own non-person identities, refused as bindings and as senders: the anonymous
/// group administrator (1087968824), a message attributed to a channel (136817688), and a
/// channel post auto-forwarded into a linked discussion group (777000) — the same number as
/// Telegram's own service account.
const TELEGRAM_SERVICE_IDS: [&str; 3] = ["1087968824", "136817688", "777000"];

/// The nicknames those identities carry (777000 has none), compared case-insensitively.
const TELEGRAM_SERVICE_NICKNAMES: [&str; 2] = ["groupanonymousbot", "channel_bot"];

/// The single owner identity the desktop GUI always acts as, seeded idempotently by
/// [`UserStore::ensure_admin_user`].
pub(crate) const ADMIN_USER_NAME: &str = "admin";

/// How a Telegram binding identifier is written.
enum TelegramKey {
    /// Canonical form: digits, no leading `+`, no leading zeros.
    Number(String),
    /// Byte-for-byte as entered or stored.
    Nickname(String),
}

/// After trimming, a single optional leading `+`, nothing but digits.
fn telegram_digits(value: &str) -> Option<&str> {
    let digits = value.trim();
    let digits = digits.strip_prefix('+').unwrap_or(digits);
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())).then_some(digits)
}

/// A nickname is never a bare number, so the shape decides: a number-shaped value is a
/// numeric id, canonicalised here so the same number written differently is one value.
/// Anything else keeps its nickname meaning. `None` only for a value that is number-shaped
/// yet nothing but zeros: no person's number.
fn telegram_key(value: &str) -> Option<TelegramKey> {
    let Some(digits) = telegram_digits(value) else {
        return Some(TelegramKey::Nickname(value.to_string()));
    };
    let canonical = digits.trim_start_matches('0');
    (!canonical.is_empty()).then(|| TelegramKey::Number(canonical.to_string()))
}

/// `None` when the value is not number-shaped or is nothing but zeros.
fn canonical_telegram_number(value: &str) -> Option<String> {
    match telegram_key(value) {
        Some(TelegramKey::Number(number)) => Some(number),
        _ => None,
    }
}

/// Refuses what can never belong to a person: the reserved `unknown` sentinel, Telegram's
/// own service identities under either spelling, and a value of nothing but zeros. A single
/// leading `@` is stripped as it always has been; what remains decides the kind.
fn normalize_telegram_binding(value: &str) -> anyhow::Result<String> {
    let trimmed = value.trim();
    let trimmed = trimmed.strip_prefix('@').unwrap_or(trimmed).trim();
    match telegram_key(trimmed) {
        Some(TelegramKey::Number(id)) => {
            if is_telegram_service_id(&id) {
                anyhow::bail!(
                    "'{id}' is a Telegram service identity, not a person — pass the account's \
                     own numeric id or @username instead"
                );
            }
            Ok(id)
        }
        Some(TelegramKey::Nickname(nickname)) => {
            if nickname.is_empty() {
                anyhow::bail!(
                    "Telegram binding is empty — pass the account's numeric id or @username"
                );
            }
            if nickname == TELEGRAM_UNKNOWN_SENTINEL {
                anyhow::bail!(
                    "'unknown' is a reserved Telegram nickname, not an account — pass the \
                     account's own numeric id or @username instead"
                );
            }
            if is_telegram_service_nickname(&nickname) {
                anyhow::bail!(
                    "'@{nickname}' is a Telegram service identity, not a person — pass the \
                     account's own numeric id or @username instead"
                );
            }
            Ok(nickname)
        }
        None => anyhow::bail!(
            "a Telegram id of nothing but zeros is not a person's number — pass the account's \
             numeric id or @username"
        ),
    }
}

pub(crate) fn is_telegram_service_id(id: &str) -> bool {
    TELEGRAM_SERVICE_IDS.contains(&id)
}

fn is_telegram_service_nickname(nickname: &str) -> bool {
    let lowercase = nickname.to_ascii_lowercase();
    TELEGRAM_SERVICE_NICKNAMES.contains(&lowercase.as_str())
}

/// Digits alone, however spelled. A stored value that is number-shaped but nothing but
/// zeros binds nobody; it is still written as a number, so it shows as an id, never `@0`.
fn telegram_identifier_is_number(identifier: &str) -> bool {
    telegram_digits(identifier).is_some()
}

/// A nickname keeps the `@` it is entered with; a number is labelled as an id.
pub(crate) fn describe_telegram_binding(identifier: &str) -> String {
    if telegram_identifier_is_number(identifier) {
        format!("id {}", identifier.trim())
    } else {
        format!("@{identifier}")
    }
}

/// The nickname exactly as it has always been shown, a number labelled as an id so it can
/// never read as a nickname.
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

/// The row is in place when [`UserStore::add_user`] returns `Ok`; its personal workspace folder is
/// a second thing that call tries to create, and the name rule or the platform can refuse it.
/// Callers report both halves, so a partial creation is never answered as a whole success.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddUserOutcome {
    /// The account and its personal workspace folder are both in place.
    Complete,
    /// In place, but its folder could not be created — also reported out of band.
    WithoutPersonalWorkspace,
}

impl UserStore {
    /// Auto-create the admin account if this is a fresh database.
    ///
    /// Runs idempotently from [`crate::db::init_all_stores`] and each isolated user store
    /// open, and is the ONLY path that may create the reserved admin name; every user-facing
    /// path refuses it ([`refuse_admin_name`]).
    pub(crate) async fn ensure_admin_user(&self) -> Result<()> {
        if !self.user_exists(ADMIN_USER_NAME).await? {
            self.add_user(ADMIN_USER_NAME).await?;
        }
        Ok(())
    }

    /// The seeding path for the admin and for guest callers, who must refuse the reserved
    /// admin name first ([`refuse_admin_name`]) and pass the collision rule
    /// ([`UserStore::refuse_colliding_name`]). Also creates the personal workspace directory
    /// under `~/.mahbot/userspaces/<name>/` with `git init`; a failure there is reported out
    /// of band and named by the returned [`AddUserOutcome`], never by an `Err` — the account
    /// exists whatever happened. Idempotent: re-adding preserves stored preferences.
    pub async fn add_user(&self, name: &str) -> Result<AddUserOutcome> {
        // Only a name that becomes a new account's folder must satisfy the folder rule: a
        // stored account is never re-judged, and its missing folder is created by
        // [`ensure_personal_workspace`], which applies the rule when it does.
        if !self.user_exists(name).await? {
            validate_folder_name(name)?;
            // The admin's own name is exempt: bootstrapping the owner account must never
            // fail, and every user-facing path refuses that name first.
            if !is_admin_name(name) {
                self.refuse_colliding_name(name).await?;
            }
        }
        self.conn
            .execute(
                "INSERT OR IGNORE INTO users (name) VALUES (?1)",
                db::params![name],
            )
            .await?;
        // A failed folder creation (the rule refusing the name, or the platform the
        // directory) is reported out of band, never silently skipped — the account exists
        // either way, and the caller is told which half it got.
        if let Err(e) = ensure_personal_workspace(name).await {
            self.report_personal_workspace_failure(name, &e).await;
            return Ok(AddUserOutcome::WithoutPersonalWorkspace);
        }
        Ok(AddUserOutcome::Complete)
    }

    /// Refuse a name about to become a NEW account when, with case and Unicode composition
    /// ignored ([`collides_with_existing`]), it is an existing account's or the admin's name:
    /// macOS resolves such spellings to one folder, and Windows (case-insensitive by default)
    /// the case difference alone, so the new account would read and write the other's files.
    ///
    /// The comparison covers every stored name plus the reserved admin name, whose row
    /// [`UserStore::ensure_admin_user`] may not have seeded yet. The rule is the same on
    /// every platform — it is about which account is meant, not the volume this process runs on.
    async fn refuse_colliding_name(&self, name: &str) -> Result<()> {
        let names = self.user_names().await?;
        // A stored name byte-identical to this one is the idempotent re-add of
        // an account that already exists — the one thing this rule never judges
        // — so it is no collision however the two reads interleave.
        let judged = names
            .iter()
            .map(String::as_str)
            .filter(|stored| *stored != name);
        if collides_with_existing(name, judged) {
            anyhow::bail!("{}", name_taken_message(name));
        }
        Ok(())
    }

    /// Report a personal workspace whose folder could not be created: a
    /// failure-level record, plus a message on the account's own Telegram chats
    /// (or, when the account has none, the admin's), so the owner is told
    /// rather than left with a workspace that silently never exists. The
    /// account itself survives — this is a complaint, not a refusal to serve.
    ///
    /// The search tool retries its account's folder on every call, so each of
    /// the two reports is claimed once per process; the owner's message is
    /// claimed only where a send is actually attempted, so a failure with nowhere to go still
    /// reaches the owner once there is a transport.
    pub(crate) async fn report_personal_workspace_failure(
        &self,
        name: &str,
        error: &anyhow::Error,
    ) {
        if claim_report(name, error, "log") {
            error!(
                user_name = %name,
                path = %personal_workspace_path(name).display(),
                error = %error,
                "Personal workspace folder could not be created"
            );
        }
        // A report with nowhere to go must not spend the owner's one message: this runs
        // inside USER_STORE's own bring-up, where `channel_registry()` panics rather than
        // reporting no channels, and the channel may not be registered even then.
        if crate::CHANNEL_REGISTRY
            .get()
            .and_then(|registry| registry.get("telegram"))
            .is_none()
        {
            return;
        }
        // Through `self`, never the global store: this can run inside USER_STORE's own
        // bring-up, where `store()` would panic.
        let mut targets =
            telegram_reply_targets(self.get_user_channels(name).await.unwrap_or_default());
        if targets.is_empty() && name != ADMIN_USER_NAME {
            targets = telegram_reply_targets(
                self.get_user_channels(ADMIN_USER_NAME)
                    .await
                    .unwrap_or_default(),
            );
        }
        // Nothing to send through, so the message is not claimed — a later attempt (once the
        // account has a binding) still reaches the owner.
        if targets.is_empty() || !claim_report(name, error, "owner") {
            return;
        }
        let message = owner_message(name, error);
        for target in targets {
            crate::channels::telegram::send_reply(&target, &message).await;
        }
    }

    /// Bindings are an account's only child rows, so nothing else needs a cascade.
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

    async fn get_selected_workspace_name(&self, user_name: &str) -> Result<Option<String>> {
        self.user_column("selected_workspace", user_name).await
    }
    pub async fn user_exists(&self, name: &str) -> Result<bool> {
        let rows = self
            .conn
            .query("SELECT 1 FROM users WHERE name = ?1", db::params![name])
            .await?;
        Ok(!rows.is_empty())
    }

    /// Every stored account name, byte-exact as stored — what a name about to
    /// become a new account is compared against.
    async fn user_names(&self) -> Result<Vec<String>> {
        let rows = self
            .conn
            .query("SELECT name FROM users", db::params![])
            .await?;
        let mut names = Vec::with_capacity(rows.len());
        for row in rows {
            names.push(row.get::<String>(0)?);
        }
        Ok(names)
    }

    /// `(channel, identifier)` with, say, channel `"telegram"` and a nickname or canonical
    /// numeric id. INSERT OR REPLACE, so a pair already bound to another user is silently
    /// reassigned; user-facing Telegram paths go through [`UserStore::bind_telegram`], which
    /// validates first (sentinel, service identities, at-most-one, anti-steal).
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

    /// Ready for [`UserStore::bind_channel`]. Refuses, in order: a value that can never
    /// belong to a person ([`normalize_telegram_binding`]); an account that already holds a
    /// Telegram binding (at most one — switching ways is remove-then-attach, so re-attaching
    /// even the same value is refused); and a value bound to another account (anti-steal:
    /// `bind_channel` would silently reassign it). The at-most-one rule is Telegram-only.
    pub async fn validate_telegram_bind(&self, user_name: &str, value: &str) -> Result<String> {
        let identifier = normalize_telegram_binding(value)?;
        if let Some(existing) = self.telegram_binding(user_name).await? {
            anyhow::bail!(
                "'{user_name}' already has a Telegram binding ({}) — remove it on the \
                 Settings → Users page first, then attach the new one",
                describe_telegram_binding(&existing)
            );
        }
        // The at-most-one check above proved this account holds no Telegram row, so an owner
        // found here is always someone else.
        if let Some(owner) = self.telegram_binding_owner(&identifier).await? {
            anyhow::bail!(
                "{} is already bound to user '{owner}' — remove that binding on the \
                 Settings → Users page first, then attach it to this account",
                describe_telegram_binding(&identifier)
            );
        }
        Ok(identifier)
    }

    /// Validate, then record. Returns the normalized identifier (the number itself for a
    /// numeric id).
    pub async fn bind_telegram(&self, user_name: &str, value: &str) -> Result<String> {
        let identifier = self.validate_telegram_bind(user_name, value).await?;
        self.attach_telegram_binding(user_name, &identifier).await?;
        Ok(identifier)
    }

    /// For the admin alone, whose bind has always also recorded a nickname as its starting
    /// address — kept for compatibility with how that address has always been stored, not
    /// because a nickname is sendable (it only starts working once the admin writes to the
    /// bot). A number is a real address on every path.
    pub async fn bind_telegram_for_admin(&self, value: &str) -> Result<String> {
        let identifier = self.bind_telegram(ADMIN_USER_NAME, value).await?;
        if !telegram_identifier_is_number(&identifier) {
            self.update_channel_contact("telegram", &identifier, &identifier)
                .await?;
        }
        Ok(identifier)
    }

    /// When `identifier` is a number, that number is also the binding's delivery address: a
    /// valid private-chat address from the start (Telegram still forbids a bot to open the
    /// conversation until that person writes). A nickname has no address yet, so nothing is
    /// recorded for it.
    pub async fn attach_telegram_binding(&self, user_name: &str, identifier: &str) -> Result<()> {
        self.bind_channel(user_name, "telegram", identifier).await?;
        if telegram_identifier_is_number(identifier) {
            self.update_channel_contact("telegram", identifier, identifier)
                .await?;
        }
        Ok(())
    }

    /// An account holds at most one; a legacy account with several reports the first.
    pub(crate) async fn telegram_binding(&self, user_name: &str) -> Result<Option<String>> {
        Ok(self
            .get_user_channels(user_name)
            .await?
            .into_iter()
            .find(|c| c.channel == "telegram")
            .map(|c| c.identifier))
    }

    /// Under any spelling of the same number, matched as on the sender path.
    async fn telegram_binding_owner(&self, identifier: &str) -> Result<Option<String>> {
        let matched = match telegram_key(identifier) {
            Some(TelegramKey::Number(number)) => {
                self.resolve_telegram_user(Some(&number), None).await?
            }
            _ => self.resolve_telegram_user(None, Some(identifier)).await?,
        };
        Ok(matched.map(|matched| matched.user_name))
    }

    /// By the sender's OWN identity, never by the conversation the event arrived in: the
    /// numeric id first (it belongs to that one person and survives a rename or a dropped
    /// nickname), then the nickname, which decides only when no number matches. `nickname` is
    /// `None` for a sender Telegram reports without one. The matched binding's stored
    /// identifier comes back alongside the account, so the caller refreshes the address that
    /// actually matched.
    ///
    /// A stored value is matched by the shape it was meant to have (`00123` is the number
    /// 123); one equal to a service identity authorizes nobody.
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

    async fn user_entry_from_row(&self, row: &db::Row) -> Result<UserRecordEntry> {
        let name: String = row.get(COL_USERS_NAME)?;
        // A failed channel read must not render as "no binding": carry the failure beside the
        // record so the GUI can surface it instead of offering to bind.
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

    /// Run `suffix` (everything after `FROM users`) and collect one [`UserRecordEntry`] per
    /// row with channel bindings.
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

    /// `None` if no such account exists.
    pub async fn find_by_name(&self, user_name: &str) -> Result<Option<UserRecord>> {
        self.list_users_where("WHERE name = ?1", db::params![user_name])
            .await
            .map(|users| users.into_iter().next().map(|entry| entry.record))
    }

    pub async fn list_users(&self) -> Result<Vec<UserRecordEntry>> {
        self.list_users_where("", db::params![]).await
    }

    /// `None` clears the column (the account's personal workspace, computed on the fly);
    /// `Some(name)` selects that managed workspace.
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

    /// An empty/whitespace value resolves to the default at read time.
    pub async fn set_image_gen_model(&self, name: &str, model: &str) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        upsert_user_column(&tx, name, "image_gen_model", Some(model)).await?;
        tx.commit().await?;
        Ok(())
    }

    /// An empty/whitespace value resolves to the default at read time.
    pub async fn set_video_model(&self, name: &str, model: &str) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        upsert_user_column(&tx, name, "video_model", Some(model)).await?;
        tx.commit().await?;
        Ok(())
    }

    /// The local day (`YYYY-MM-DD`) of the last nightly memory inventory ask, if any.
    pub(crate) async fn memory_inventory_day(&self, name: &str) -> Result<Option<String>> {
        self.user_column("memory_inventory_day", name).await
    }

    /// Record the day the memory inventory asked this account. Written before
    /// the message is handed over, so a restart cannot ask twice that day.
    pub(crate) async fn set_memory_inventory_day(&self, name: &str, day: &str) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        upsert_user_column(&tx, name, "memory_inventory_day", Some(day)).await?;
        tx.commit().await?;
        Ok(())
    }

    /// Byte-sorted; empty when the user has no grants or no row.
    pub(crate) async fn get_grants(&self, user_name: &str) -> Result<Vec<String>> {
        Ok(parse_grants(
            self.user_column("granted_tools", user_name).await?,
        ))
    }

    /// Idempotent, and a grant never inserts a users row (a ghost user would appear in the
    /// GUI), so the caller reports the missing user itself.
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

    /// Idempotent: an absent grant is a silent no-op. The result tells the caller which of the
    /// three cases it hit, so it can report a missing user instead of a false confirmation — a
    /// revoke never inserts a row either.
    pub(crate) async fn remove_grant(&self, user_name: &str, tool: &str) -> Result<GrantChange> {
        self.update_grants(user_name, |grants| {
            let before = grants.len();
            grants.retain(|g| g != tool);
            grants.len() != before
        })
        .await
    }

    /// Ordered by user name. The empty-list filter runs in Rust rather than SQL: `[]` and an
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

    /// In one transaction. `mutate` edits the parsed, sorted set; returning `false` skips the
    /// write (the caller's idempotent no-op). A missing row is left untouched, never inserted —
    /// the caller reports that from the returned [`GrantChange`].
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
        // The closures only add or remove one name, so a push is the one thing that can break
        // the parsed set's sorted order.
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

/// The cause, and the single remedy: an account whose name the rule refuses keeps its name —
/// there is no rename — so it has to be removed and added again.
fn owner_message(name: &str, error: &anyhow::Error) -> String {
    let cause = if let Some(problem) = crate::util::folder_name::folder_name_problem(name) {
        format!(
            "the name cannot be a folder name on every platform the service runs on ({problem}), so \
             this account cannot keep a workspace under it — remove the account and add it again \
             under a name the service accepts"
        )
    } else {
        format!(
            "the storage location {} cannot be written ({error}) — check permissions and free \
             space, then try again",
            personal_workspace_path(name).display()
        )
    };
    format!(
        "⚠️ The personal workspace folder for the account '{name}' could not be created, so nothing \
         that account stores in its workspace is kept: {cause}."
    )
}

/// The part that did not happen, and nothing more — no cause and no remedy, because another
/// writer can create the folder later.
#[must_use]
pub(crate) fn missing_personal_workspace_note(name: &str) -> String {
    format!(
        "The personal workspace folder for '{name}' could not be created, so that account has \
         nowhere to keep its workspace files."
    )
}

/// Claimed once per process and separately per medium, so a report with no way to go out does
/// not spend the other's; the key is per medium, account and cause. `"log"` and `"owner"` are
/// the two media.
static REPORTED_WORKSPACE_FAILURES: LazyLock<Mutex<HashSet<String>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Whether this report has not gone out yet, claiming it for `medium` if so. Called only where
/// the report is actually made, so a claim is never spent on an attempt with nowhere to go.
fn claim_report(name: &str, error: &anyhow::Error, medium: &str) -> bool {
    REPORTED_WORKSPACE_FAILURES
        .lock()
        .unwrap_poison()
        .insert(format!("{medium}\u{0}{name}\u{0}{error}"))
}

/// A failed read yields no targets — never a partial send list.
fn telegram_reply_targets(bindings: Vec<ChannelBinding>) -> Vec<String> {
    bindings
        .into_iter()
        .filter(|b| b.channel == "telegram")
        .filter_map(|b| b.reply_target)
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum GrantChange {
    /// Nothing was written, and none is inserted.
    NoUser,
    /// Already held the requested state — nothing written.
    Unchanged,
    /// Its grant set changed.
    Changed,
}

/// A NULL, empty or unparseable value yields an empty list (fail-closed: unknown grants are no
/// grants).
fn parse_grants(raw: Option<String>) -> Vec<String> {
    let mut grants = raw
        .and_then(|raw| serde_json::from_str::<Vec<String>>(&raw).ok())
        .unwrap_or_default();
    grants.sort();
    grants.dedup();
    grants
}

/// `field` MUST be a compile-time string literal to prevent SQL injection.
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

#[derive(Debug, Clone)]
pub struct UserRecord {
    /// When it is [`ADMIN_USER_NAME`], this is the admin account.
    pub name: String,
    /// Selected managed workspace name, NULL = personal workspace.
    pub selected_workspace: Option<String>,
    /// The `users.granted_tools` JSON array, byte-sorted; empty when none. A grant recorded for
    /// the admin is carried and listed like any other; only the account-management card omits it,
    /// the whole catalogue being implicitly available to the admin.
    pub granted_tools: Vec<String>,
    /// Channel bindings for this account (Telegram, etc.).
    pub channels: Vec<ChannelBinding>,
}

/// [`UserRecord`] is channel-empty both when the account has no bindings and when the read
/// failed, so a surface that renders bindings needs the outcome to tell the two apart; it
/// travels beside the record, not inside it.
#[derive(Debug, Clone)]
pub struct UserRecordEntry {
    pub record: UserRecord,
    /// `Some` when the read failed; `record.channels` is then empty.
    pub channels_error: Option<String>,
}

/// An account is the admin exactly when its name is [`ADMIN_USER_NAME`] — no stored token, flag
/// or kind ever marks one. Sync callers (GUI state) apply it to a name they hold; a bare,
/// untrusted name goes through [`is_admin`], which backs the same test with the row.
#[must_use]
pub(crate) fn is_admin_name(name: &str) -> bool {
    name == ADMIN_USER_NAME
}

/// Only [`UserStore::ensure_admin_user`] may create the admin account, and the name IS the admin
/// marker, so binding a channel to it would address the admin. [`UserStore::add_user`] exempts
/// the same name from the collision rule for that one seeding path.
pub(crate) fn refuse_admin_name(name: &str) -> Result<()> {
    if is_admin_name(name) {
        anyhow::bail!(
            "'{ADMIN_USER_NAME}' is the admin account — only guest accounts can be created"
        );
    }
    Ok(())
}

/// The other half of the account-name rule, in the words the account-adding tool's schema
/// answers with: a new account's name is refused when another account or the admin already holds
/// it once case and Unicode composition are ignored.
pub(crate) const NAME_COLLISION_RULE: &str = "A new account's name is also refused when another \
     account or the admin already has it, ignoring case and Unicode composition — on macOS those \
     spellings are one folder, on Windows a case difference alone already is.";

/// The exact duplicate and the case/composition collision alike, so a surface reports both the
/// same way.
pub(crate) fn name_taken_message(name: &str) -> String {
    format!("A user named '{name}' already exists")
}

/// The one place the collision rule lives, judged by [`UserStore::add_user`] and the desktop's
/// add-user dialog. `existing` is the caller's reading of the stored names; the reserved admin name
/// counts whether or not its row is in that reading. A byte-identical name collides here too — the
/// store's exemption of the idempotent re-add is its own policy, applied at its call site.
pub(crate) fn collides_with_existing<'a>(
    name: &str,
    existing: impl IntoIterator<Item = &'a str>,
) -> bool {
    let key = crate::util::caseless_name_key(name);
    std::iter::once(ADMIN_USER_NAME)
        .chain(existing)
        .any(|existing| crate::util::caseless_name_key(existing) == key)
}

/// The name is the only unbounded part of the taken-name line, so this keeps it one line.
const MAX_NAME_IN_REFUSAL_CHARS: usize = 24;

/// Its first [`MAX_NAME_IN_REFUSAL_CHARS`] characters, then an ellipsis if there were more.
fn one_line_name(name: &str) -> String {
    match name.char_indices().nth(MAX_NAME_IN_REFUSAL_CHARS) {
        Some((end, _)) => format!("{}…", &name[..end]),
        None => name.to_string(),
    }
}

/// The refusals the creation path applies, in order — the reserved admin name and the folder
/// rule ([`validate_new_user_name`]), then the collision ([`collides_with_existing`]) — so a
/// name the dialog judges unusable never reaches the store. `None` exactly when the name is one
/// to create a new account with.
pub(crate) fn new_account_name_problem<'a>(
    name: &str,
    existing: impl IntoIterator<Item = &'a str>,
) -> Option<String> {
    if is_admin_name(name) {
        return Some(
            "This name cannot be used because it is the admin account's name.".to_string(),
        );
    }
    if let Some(problem) = crate::util::folder_name::folder_name_problem(name) {
        return Some(format!("This name cannot be used because {problem}."));
    }
    if collides_with_existing(name, existing) {
        return Some(name_taken_message(&one_line_name(name)));
    }
    None
}

/// The name is the folder's name verbatim, so it must be creatable on every platform the
/// product runs on (see [`crate::util::folder_name`]).
fn validate_folder_name(name: &str) -> Result<()> {
    let Some(problem) = crate::util::folder_name::folder_name_problem(name) else {
        return Ok(());
    };
    anyhow::bail!(
        "'{name}' cannot be a user name because {problem}. The account name is used verbatim \
         as the name of its personal workspace folder, so it must be creatable on every \
         platform MahBot runs on — pick another name."
    );
}

/// The reserved admin name ([`refuse_admin_name`]) and the folder rule
/// ([`validate_folder_name`]). The third — collision with an existing name — needs the store
/// and lives in [`UserStore::add_user`], the funnel every creation path goes through.
pub(crate) fn validate_new_user_name(name: &str) -> Result<()> {
    refuse_admin_name(name)?;
    validate_folder_name(name)
}

#[derive(Debug, Clone)]
pub struct ChannelBinding {
    /// The channel type (e.g. "telegram").
    pub channel: String,
    /// The channel-specific identifier (a Telegram nickname, or a numeric id).
    pub identifier: String,
    /// Routing address for replies on this channel (e.g. Telegram chat_id:thread_id).
    pub reply_target: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TelegramMatch {
    pub user_name: String,
    /// The stored identifier of the binding that matched — whose delivery address follows this
    /// sender's conversation.
    pub identifier: String,
}

/// The userspaces root, `<storage_root>/userspaces`: the single resolution point, shared by
/// [`personal_workspace_path`] and `research_cleanup::sweep_media` so both always agree.
///
/// Resolves under the CONFIG storage root when set, else [`fallback_storage_root`] — so a test
/// reaching the fallback (`add_user` without `init_test_stores`) still stays inside test-owned
/// storage, never the real user config directory.
#[must_use]
pub(crate) fn userspaces_root() -> PathBuf {
    let storage_root = crate::config::CONFIG
        .try_storage_root()
        .unwrap_or_else(fallback_storage_root);
    storage_root.join("userspaces")
}

/// Production: the default config directory. Tests: the shared test root, so a test reaching this
/// fallback never writes into the real config directory; production always sets the root at
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

/// Non-empty, no path separators, not a dot entry. Applied on both sides: the read-side callers
/// (the `<workspace-structure>` picture, the compaction dump, the change report) before touching
/// an existing folder, and [`ensure_personal_workspace`] before creating one, so a stored name
/// can never address the userspaces root or another account's directory. A new account's name is
/// refused outright by [`validate_folder_name`], the stricter cross-platform rule.
#[must_use]
pub(crate) fn is_valid_personal_user_name(user_name: &str) -> bool {
    !user_name.trim().is_empty()
        && !user_name.contains(['/', '\\'])
        && !matches!(user_name, "." | "..")
}

/// `<storage_root>/userspaces/<name>/`, computed on the fly — personal workspaces have no
/// `workspaces` row.
#[must_use]
pub fn personal_workspace_path(user_name: &str) -> PathBuf {
    userspaces_root().join(user_name)
}

/// Personal workspaces are not `workspaces` rows, so a row carrying this prefix is a legacy
/// leftover rather than a managed workspace.
const PERSONAL_WORKSPACE_PREFIX: &str = "personal:";

/// `personal:{user_name}`: the single search-engine key shared by agent side and dashboard.
#[must_use]
pub fn personal_workspace_name(user_name: &str) -> String {
    format!("{PERSONAL_WORKSPACE_PREFIX}{user_name}")
}

/// Creates the directory if missing and runs `git init` only when no repo is present yet
/// (idempotent otherwise).
///
/// `Err` when the name cannot be a folder name on every platform, or the platform refuses to
/// create the directory. Nothing is ever created outside `userspaces/`; the caller reports the
/// failure. A `git init` failure stays a warning — it leaves a usable directory behind.
pub(crate) async fn ensure_personal_workspace(name: &str) -> Result<()> {
    // Containment: a name resolving outside its own folder never reaches the filesystem.
    if !is_valid_personal_user_name(name) {
        anyhow::bail!("'{name}' is not a usable personal-workspace name");
    }
    let path = personal_workspace_path(name);
    if !path.is_dir() {
        // The cross-platform rule, applied where a folder is about to be created: a stored name
        // only THIS platform would accept keeps the folder it already has. Writers inside an
        // account's folder bring it into being with their own `create_dir_all`.
        validate_folder_name(name)?;
        tokio::fs::create_dir_all(&path)
            .await
            .with_context(|| format!("create the folder {}", path.display()))?;
    }
    // `git init` only when there is no repo yet; non-fatal on failure.
    if path.join(".git").exists() {
        return Ok(());
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
    Ok(())
}

/// Active image-gen model for `user_name`: the user's explicit Telegram picker choice, or the
/// hardcoded default when unset/unresolvable.
pub async fn resolve_image_gen_model(user_name: &str) -> String {
    resolve_user_model_column(user_name, "image_gen_model")
        .await
        .unwrap_or_else(|| crate::config::DEFAULT_IMAGE_GEN_MODEL.to_string())
}

/// Active video model for `user_name`: the user's explicit Telegram picker choice, or the hardcoded
/// default when unset/unresolvable.
pub async fn resolve_video_model(user_name: &str) -> String {
    resolve_user_model_column(user_name, "video_model")
        .await
        .unwrap_or_else(|| crate::config::DEFAULT_VIDEO_MODEL.to_string())
}

/// A single-column SELECT, because this is a per-tool-call hot path (no full `UserRecord` load).
/// `None` — user missing, column unset, or a DB error (logged; fail-open to the default).
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

/// THE single admin-aware resolution primitive, shared by message routing and the GUI (boot
/// restore, reverse-sync).
///
/// Workspace membership is admin-only: the admin gets their stored selection (a legacy personal
/// value normalizes to `personal:{user}`); a guest ALWAYS gets their `personal:{user}` name,
/// never a managed workspace (a managed selection is clamped with a warning — defence in depth
/// against rogue re-attachment). A NULL `selected_workspace` and a missing row both yield
/// `personal:{user}`; `None` is reserved for a failed read (warned), and the caller then applies
/// its own answer without writing one back.
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
            "guest has a managed selected_workspace — clamping to their personal workspace"
        );
    }
    Some(personal_workspace_name(user_name))
}

/// The first managed name in the footer picker's order (`workspaces.name`, ascending). Start-up
/// and a deletion share this one rule, so a restart never changes the active project. A personal
/// workspace is never a candidate.
#[must_use]
pub(crate) fn fallback_workspace_name<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Option<&'a str> {
    names
        .into_iter()
        .filter(|name| !is_personal_workspace(name))
        .min()
}

/// Every `workspaces` row except a personal space, in the footer picker's order
/// (`workspaces.name`, already ordered by the store). A personal workspace is never included:
/// those are not `workspaces` rows, and a legacy row named `personal:{user}` is filtered out.
///
/// Two consumers: the Assistant's `workspace` tool (the whole set for `list`/`board`/`pause`/
/// `resume`, and the set a ticket id's workspace prefix must belong to), and the path tool's
/// admin write policy, which refuses a write inside any managed workspace.
pub async fn managed_workspaces() -> Result<Vec<Workspace>> {
    let store = crate::workspace::WORKSPACES
        .get()
        .ok_or_else(|| anyhow::anyhow!("the workspace store is not open"))?;
    Ok(store
        .list()
        .await?
        .into_iter()
        .filter(|ws| !is_personal_workspace(&ws.name))
        .collect())
}

/// Resolves through [`resolve_selected_workspace_name`]: the admin gets their stored workspace
/// (managed or personal), a guest always their personal one. `None` (a read failure) yields the
/// personal workspace.
async fn get_workspace(user_name: &str) -> Result<Option<Workspace>> {
    match resolve_selected_workspace_name(user_name).await {
        Some(ws_name) => resolve_workspace(&ws_name).await,
        None => Ok(Some(personal_workspace_struct(user_name))),
    }
}

/// Synthesizing a personal workspace (`personal:{user}`) when the name is not in the
/// `workspaces` table: personal workspaces are NOT stored there — they live at
/// `~/.mahbot/userspaces/<user>/` as ephemeral [`Workspace`] structs.
///
/// An absolute path to an existing directory is also a legal name
/// (`crate::workspace::synthesize_from_path`): a custom-role launch may name any folder. A path
/// that is a *managed* workspace's own folder resolves to that workspace, so the launch runs
/// with its notes, pause state and search engine instead of a second, path-named area over the
/// same tree. Any other path is addressed by the canonical path itself; a managed workspace with
/// that name (checked first) still wins.
///
/// `Ok(None)` when the workspace genuinely does not exist (and is not a personal or directory
/// workspace), `Err` on database errors. Shared by the message router and the boot resume path
/// so both treat synthetic personal workspaces identically.
pub async fn resolve_workspace(workspace_name: &str) -> Result<Option<Workspace>> {
    if let Some(ws) = crate::workspace::get_by_name(workspace_name).await? {
        // A managed row whose name looks personal can only be a legacy row (`validate_name`
        // rejects ':'), so `personal:{user}` shadows the key; log it so the operator can clean
        // it up.
        if is_personal_workspace(&ws.name) {
            warn!(
                workspace_name = %ws.name,
                "Managed workspace name shadows the personal workspace key 'personal:{{user}}'"
            );
        }
        Ok(Some(ws))
    } else if is_personal_workspace(workspace_name) {
        let user_name = personal_user_name(workspace_name)
            .expect("invariant: is_personal_workspace checked the prefix");
        Ok(Some(personal_workspace_struct(user_name)))
    } else if let Some(ws) = crate::workspace::synthesize_from_path(workspace_name) {
        match crate::workspace::store().get_by_path(&ws.path).await? {
            Some(managed) => Ok(Some(managed)),
            None => Ok(Some(ws)),
        }
    } else {
        Ok(None)
    }
}

/// Has no diagnostics, no maintenance, no discovery — minimal defaults.
#[must_use]
pub(crate) fn personal_workspace_struct(user_name: &str) -> Workspace {
    let mut ws = Workspace::from_path(&personal_workspace_path(user_name));
    ws.name = personal_workspace_name(user_name);
    ws.status = WorkspaceStatus::Ready;
    ws
}

/// Falling back to a personal workspace if `get_workspace` fails or returns `None`.
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

#[must_use]
fn is_pinned_role(role: Role) -> bool {
    matches!(role, Role::Assistant)
}

/// A pinned role (Assistant) must never run outside the envelope user's OWN personal
/// workspace, whatever a producer stored: a non-personal workspace is re-pinned to
/// `personal:{user}`, and another user's personal workspace to the envelope user's own. `None`
/// means the routing must be refused outright — a pinned role with an empty `user_name` has no
/// identity to pin to, and running it unpinned is never acceptable. Non-pinned roles pass through.
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

/// The Assistant always works in the user's personal workspace regardless of the selected one,
/// giving path-dependent callers (enrichment uploads, generated-media writes) its filesystem
/// path; other roles pass through. An empty `user_name` disables pinning, so callers must pass a
/// resolvable user (the voice path passes [`ADMIN_USER_NAME`]).
///
/// Accepted (no migration): media written before pinning to a managed workspace's
/// `uploads/`/`generated/` stays there and is no longer reachable by Assistant tools.
#[must_use]
pub fn effective_workspace_for_role(role: Role, ws: Workspace, user_name: &str) -> Workspace {
    if !is_pinned_role(role) {
        return ws;
    }
    // Asymmetric with `enforce_personal_pinning`: an already-personal name stays as-is, even
    // when it names another user, where the enforcer re-pins it.
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

/// The same resolution as routing, so ClearChat and Telegram /clear always clear the actual
/// recipient: the account's selected workspace with Assistant pinning applied, and the Assistant
/// role (every account's only role), which the session-key builders consume alongside it.
pub async fn resolve_session_target(user_name: &str) -> (Role, Workspace) {
    let ws = resolve_workspace_for_user_name(user_name).await;
    (
        Role::Assistant,
        effective_workspace_for_role(Role::Assistant, ws, user_name),
    )
}

/// Matched exactly, or as `target:thread`. First match wins for group chats shared by several
/// users.
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

/// Exactly when the name is the admin's ([`is_admin_name`]) AND `store` holds its row. `None`
/// (no store) and a failed read both answer false, so the answer never defaults to admin rights.
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

/// [`is_admin_in`] bound to the process-global account store.
pub async fn is_admin(user_name: &str) -> bool {
    is_admin_in(USER_STORE.get(), user_name).await
}

/// A failed read yields no grants, which fails every access decision closed.
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

/// Called on every incoming message.
pub async fn update_channel_contact(
    channel: &str,
    identifier: &str,
    reply_target: &str,
) -> Result<()> {
    store()
        .update_channel_contact(channel, identifier, reply_target)
        .await
}

/// `personal:{user}`, or `None` when the name is not a personal workspace.
#[must_use]
pub fn personal_user_name(workspace_name: &str) -> Option<&str> {
    workspace_name.strip_prefix(PERSONAL_WORKSPACE_PREFIX)
}

/// A personal workspace's name carries the `personal:` prefix.
#[must_use]
pub fn is_personal_workspace(workspace_name: &str) -> bool {
    personal_user_name(workspace_name).is_some()
}

#[cfg(test)]
pub(crate) mod test_util {
    use super::*;

    /// Safe to call multiple times: [`init_test_stores`] initializes every global store, and
    /// this supplements USER_STORE with telegram-specific accounts and bindings.
    pub(crate) async fn init_test_store() {
        // Initialize every global store (idempotent OnceCell).
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
        let previous = store
            .get_selected_workspace_name(ADMIN_USER_NAME)
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

    /// Run `body` with the given `(path, name)` fixtures seeded into the shared
    /// workspace store, deleting each row afterwards — including when `body`
    /// panics, so a failed assertion cannot leak a test-only workspace into the
    /// sibling tests that share this process-global store. The seeding is inside
    /// the guard too, so a fixture that is already there cannot leave the rows
    /// seeded before it behind.
    ///
    /// Unwinding by hand, like [`with_admin_workspace_restored`]: the cleanup
    /// awaits the store, which no destructor can do.
    pub(crate) async fn with_test_workspaces<F>(workspaces: &[(&str, &str)], body: F)
    where
        F: std::future::Future<Output = ()>,
    {
        use futures_util::FutureExt as _;

        let outcome = std::panic::AssertUnwindSafe(async {
            for (path, name) in workspaces {
                crate::util::test::create_test_workspace(path, name).await;
            }
            body.await;
        })
        .catch_unwind()
        .await;
        for (_, name) in workspaces {
            crate::workspace::store()
                .delete(name)
                .await
                .expect("delete test workspace");
        }
        if let Err(panic) = outcome {
            std::panic::resume_unwind(panic);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The admin test is the account's name, backed by the account's own row:
    /// a guest name, a name with no row, and a missing store are never the admin.
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

    /// A name the platform cannot turn into a folder is refused before its row
    /// exists, and never relocates the personal workspace: the userspaces root
    /// gains nothing outside itself.
    #[tokio::test]
    async fn add_user_refuses_a_name_that_cannot_be_a_folder() {
        crate::util::test::init_test_stores().await;
        let store = store();
        for name in ["../evil", "a/b", "/tmp/evil", "nul", "trailing."] {
            let err = store.add_user(name).await.unwrap_err();
            assert!(
                err.to_string().contains(name),
                "the refusal must name the offending name: {err}"
            );
            assert!(
                !store.user_exists(name).await.unwrap(),
                "a refused name must not leave an account row"
            );
        }
        assert!(
            !userspaces_root().join("../evil").exists(),
            "a parent reference must not escape the userspaces root"
        );
    }

    /// A new name that is an existing account's or the admin's name once case
    /// and Unicode composition are ignored is refused the way an exact
    /// duplicate is, and leaves nothing behind: no row, no folder.
    #[tokio::test]
    async fn add_user_refuses_a_name_an_existing_one_matches_ignoring_case_and_composition() {
        let (store, _dir) = crate::open_test_store!(UserStore, "user");
        // Names no other test holds: the userspaces root is shared by every test
        // in this process, and a folder another test created first would be the
        // one this test's assertions read back.
        let composed = "Caf\u{e9}Collide";
        let decomposed = "Cafe\u{301}Collide";
        let ligature = "Liga\u{fb03}Collide";
        store.add_user("AliceCollide").await.unwrap();
        store.add_user(composed).await.unwrap();
        store.add_user("stra\u{df}eCollide").await.unwrap();
        store.add_user(ligature).await.unwrap();

        // Every difference the comparison must ignore: case, Unicode
        // composition, and the full case folding this volume itself applies —
        // `ß` and `ss` are one letter, and so are a three-character fold's
        // `ﬃ` and `ffi`; plus the admin's own name.
        for colliding in [
            "alicecollide",
            "ALICECOLLIDE",
            decomposed,
            "Admin",
            "ADMIN",
            "STRASSECOLLIDE",
            "strassecollide",
            "LigaffiCollide",
        ] {
            let err = store.add_user(colliding).await.unwrap_err().to_string();
            assert_eq!(
                err,
                name_taken_message(colliding),
                "a colliding name must be refused like an exact duplicate"
            );
            assert!(
                !store.user_exists(colliding).await.unwrap(),
                "a refused name must not leave an account row"
            );
        }

        // Nothing half-created: no row (asserted above), and no folder of its
        // own. On this case-insensitive volume a folder created for a refused
        // spelling would BE the accepted account's folder, so what is checked
        // here is the entry names — the accepted spellings kept verbatim, and no
        // refused spelling an entry, which is what a case-sensitive host shows.
        let folders = userspace_folder_names();
        for accepted in ["AliceCollide", composed, "stra\u{df}eCollide", ligature] {
            assert!(
                folders.iter().any(|folder| folder == accepted),
                "'{accepted}' must keep the folder it was created with: {folders:?}"
            );
        }
        for refused in [
            "ALICECOLLIDE",
            decomposed,
            "strassecollide",
            "LigaffiCollide",
        ] {
            assert!(
                !folders.iter().any(|folder| folder == refused),
                "'{refused}' must not be a folder: {folders:?}"
            );
        }

        // Only the equivalences the volume itself applies are ignored: a name
        // matching nothing under them is still accepted, verbatim, and gets its
        // folder like any other.
        store.add_user("AliceCollide2").await.unwrap();
        assert!(
            userspace_folder_names()
                .iter()
                .any(|folder| folder == "AliceCollide2"),
            "an accepted name keeps its account and its folder"
        );
    }

    /// The owner's own name is never refused and a stored account is never
    /// re-judged: the reserved name is refused for a case variant that is not
    /// the owner even with no row of its own to compare against, bootstrapping
    /// the owner succeeds beside a guest holding such a variant, and completing
    /// that stored guest is accepted — the rule judges only names it is about to create.
    #[tokio::test]
    async fn add_user_never_refuses_the_admin_or_a_stored_name() {
        let (store, _dir) = crate::open_test_store!(UserStore, "user");
        // The store's own bring-up seeded the owner; the rule is judged for it
        // exactly when its row is absent.
        store.delete_user(ADMIN_USER_NAME).await.unwrap();

        // A case variant of the reserved name is not the owner, and is refused
        // by the reserved name itself — which is in the comparison set whether or
        // not its row exists.
        let err = store.add_user("Admin").await.unwrap_err().to_string();
        assert_eq!(
            err,
            name_taken_message("Admin"),
            "the reserved owner name must collide even with no row of its own"
        );

        // A pre-existing installation can hold a guest whose name is a case
        // variant of the owner's, inserted directly because `add_user` now
        // refuses to create it — which is the point.
        store
            .conn
            .execute(
                "INSERT INTO users (name) VALUES (?1)",
                crate::db::params!["ADMIN"],
            )
            .await
            .unwrap();

        // Bootstrap beside it: the reserved name is exempt, so the owner row is
        // created rather than refused, and the start-up is never blocked.
        store.ensure_admin_user().await.unwrap();
        assert!(
            store.user_exists(ADMIN_USER_NAME).await.unwrap(),
            "bootstrapping the owner account must never be blocked by the rule"
        );

        store.add_user("ADMIN").await.unwrap();
        assert!(
            store.user_exists("ADMIN").await.unwrap(),
            "a stored name is completed, not re-judged"
        );
    }

    /// The personal-folder names present in the userspaces root, spelled as the
    /// filesystem stores them.
    fn userspace_folder_names() -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(userspaces_root()) else {
            return Vec::new();
        };
        let mut names: Vec<String> = entries
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// A name that already identifies an account is never re-judged: an account
    /// stored by an older installation keeps the name it has, so this completes
    /// it — and the folder the rule refuses is still not created for it.
    #[tokio::test]
    async fn add_user_keeps_a_stored_name_the_folder_rule_would_refuse() {
        crate::util::test::init_test_stores().await;
        let store = store();
        // The name holds a separator-free illegal character, so the rule refuses
        // it while every platform this test runs on can still create a folder
        // called that — which is what makes the folder assertion below a real one.
        let legacy = "legacy:name";
        store
            .conn
            .execute(
                "INSERT INTO users (name) VALUES (?1)",
                crate::db::params![legacy],
            )
            .await
            .unwrap();

        let outcome = store.add_user(legacy).await.unwrap();
        assert_eq!(
            outcome,
            AddUserOutcome::WithoutPersonalWorkspace,
            "the folder the rule refuses must be reported as not created"
        );
        assert!(
            !personal_workspace_path(legacy).exists(),
            "completing a stored account must not create the folder the rule refuses"
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

    /// A pinned role always resolves to the account's own personal workspace —
    /// whose folder is that account's folder under the userspaces root — and
    /// re-pins from a project or another user's personal workspace; non-pinned
    /// roles pass through, and a pinned role with no user has no personal
    /// identity to pin to.
    #[test]
    fn enforce_personal_pinning_repins_pinned_roles_and_refuses_empty_user() {
        // The resolution itself: the personal workspace keeps the account's
        // name and folder, an already-personal one passes through, and an empty
        // user disables pinning.
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
        assert_eq!(
            effective_workspace_for_role(Role::Engineer, project.clone(), "alice").name,
            "ws1"
        );
        assert_eq!(
            effective_workspace_for_role(Role::Assistant, personal, "alice").name,
            "personal:alice"
        );
        assert_eq!(
            effective_workspace_for_role(Role::Assistant, project, "").name,
            "ws1"
        );

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
            // The admin + a managed selection → keeps the managed workspace.
            store
                .set_selected_workspace(ADMIN_USER_NAME, Some("ws_managed"))
                .await
                .unwrap();
            assert_eq!(
                resolve_selected_workspace_name(ADMIN_USER_NAME).await,
                Some("ws_managed".to_string())
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

        // A guest + a managed selection → clamped to personal.
        store
            .set_selected_workspace("u_managed", Some("ws_managed"))
            .await
            .unwrap();
        assert_eq!(
            resolve_selected_workspace_name("u_managed").await,
            Some("personal:u_managed".to_string())
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

        // A guest with a managed selection that exists in the table is still
        // clamped to the personal workspace (never a managed one).
        store
            .set_selected_workspace("u_managed", Some("ws_admin_aware"))
            .await
            .unwrap();
        let ws = resolve_workspace_for_user_name("u_managed").await;
        assert_eq!(ws.name, "personal:u_managed");

        // The admin with a managed selection that exists keeps the managed workspace.
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
