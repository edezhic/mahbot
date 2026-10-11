//! Admin-defined custom roles: the record (name, authorial prompt, tool subset,
//! behaviour without an identifier) plus its validation. The record carries no
//! owner column because there is exactly one owner — the admin — and two
//! surfaces reach these functions: the admin's configuration tool (a role the
//! admin granted `mahbot_config` runs at his strength and can call them) and
//! the local Settings page. The tool set is the boundary, by design.

use anyhow::Result;

use crate::db;

crate::define_store! {
    pub(crate) static CUSTOM_ROLES: CustomRoleStore,
    expect = "CUSTOM_ROLES not initialized — call init_all_stores() first",
}

// ── Column index constants ────────────────────────────────────────────────

crate::columns! {
    CUSTOM_ROLE_COLUMNS [CR] {
        NAME     => "name",
        PROMPT   => "prompt",
        TOOLS    => "tools",
        BEHAVIOR => "behavior",
    }
}

/// Upper bound on a custom role's name, in characters.
pub(crate) const MAX_ROLE_NAME_CHARS: usize = 32;

/// Upper bound on a custom role branch identifier, in characters.
pub(crate) const MAX_ROLE_IDENTIFIER_CHARS: usize = 32;

/// What a custom role does when it is launched without a branch identifier:
/// keep one continuable conversation ([`Self::Continuous`]) or run once and
/// report back ([`Self::OneOff`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CustomRoleBehavior {
    Continuous,
    OneOff,
}

impl CustomRoleBehavior {
    /// The stored column value.
    #[must_use]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Continuous => "continuous",
            Self::OneOff => "one_off",
        }
    }

    /// Parse a stored column value.
    pub(crate) fn parse(s: &str) -> anyhow::Result<Self> {
        match s {
            "continuous" => Ok(Self::Continuous),
            "one_off" => Ok(Self::OneOff),
            other => anyhow::bail!("unknown custom role behavior '{other}'"),
        }
    }
}

/// The label the settings picker shows — never the stored value
/// ([`CustomRoleBehavior::as_str`] is what the column holds).
impl std::fmt::Display for CustomRoleBehavior {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Continuous => "Keep one branch",
            Self::OneOff => "One-off run",
        })
    }
}

/// One admin-authored custom role.
#[derive(Debug, Clone)]
pub(crate) struct CustomRole {
    pub name: String,
    /// The administrator's authorial text — the whole definition of the role.
    pub prompt: String,
    /// The chosen subset of the admin Assistant's tool names.
    pub tools: Vec<String>,
    pub behavior: CustomRoleBehavior,
}

fn custom_role_from_row(row: &db::Row) -> Result<CustomRole> {
    Ok(CustomRole {
        name: row.get(COL_CR_NAME)?,
        prompt: row.get(COL_CR_PROMPT)?,
        tools: decode_tools(&row.get::<String>(COL_CR_TOOLS)?),
        behavior: CustomRoleBehavior::parse(&row.get::<String>(COL_CR_BEHAVIOR)?)?,
    })
}

/// Encode the tool subset for the `tools` column — only this module writes it.
fn encode_tools(tools: &[String]) -> String {
    serde_json::to_string(tools).expect("a tool name list is always serializable")
}

/// Decode the `tools` column. A value that does not decode reads as the empty
/// subset rather than failing the whole listing — only this module writes the
/// column, so an undecodable value is not a state the product produces.
fn decode_tools(raw: &str) -> Vec<String> {
    serde_json::from_str(raw).unwrap_or_default()
}

/// Every custom role, ordered by name.
pub(crate) async fn list() -> Result<Vec<CustomRole>> {
    let sql = format!("SELECT {CUSTOM_ROLE_COLUMNS} FROM custom_roles ORDER BY name ASC");
    let rows = store().conn.query(&sql, db::params![]).await?;
    rows.iter().map(custom_role_from_row).collect()
}

/// One custom role by its exact name.
pub(crate) async fn get(name: &str) -> Result<Option<CustomRole>> {
    let sql = format!("SELECT {CUSTOM_ROLE_COLUMNS} FROM custom_roles WHERE name = ?1");
    let rows = store().conn.query(&sql, db::params![name]).await?;
    rows.first().map(custom_role_from_row).transpose()
}

/// Create a custom role. The name must be free (an existing role is never
/// silently overwritten) and every field is validated first.
pub(crate) async fn create(
    name: &str,
    prompt: &str,
    tools: &[String],
    behavior: CustomRoleBehavior,
) -> Result<()> {
    validate_name(name)?;
    validate_tools_against(tools, &[])?;
    anyhow::ensure!(
        !prompt.trim().is_empty(),
        "a custom role needs a prompt — the administrator's text is the whole definition"
    );
    anyhow::ensure!(
        get(name).await?.is_none(),
        "a custom role named '{name}' already exists"
    );
    store()
        .conn
        .execute(
            "INSERT INTO custom_roles (name, prompt, tools, behavior, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            db::params![
                name,
                prompt,
                encode_tools(tools),
                behavior.as_str(),
                db::now(),
            ],
        )
        .await?;
    Ok(())
}

/// Update the provided fields of an existing custom role; the omitted ones keep
/// their stored value. The record is read, patched and written back whole.
pub(crate) async fn update(
    name: &str,
    prompt: Option<&str>,
    tools: Option<&[String]>,
    behavior: Option<CustomRoleBehavior>,
) -> Result<()> {
    let existing = get(name)
        .await?
        .ok_or_else(|| anyhow::anyhow!("no custom role named '{name}'"))?;
    let prompt = prompt.unwrap_or(&existing.prompt);
    anyhow::ensure!(
        !prompt.trim().is_empty(),
        "a custom role needs a prompt — the administrator's text is the whole definition"
    );
    let tools = tools.unwrap_or(&existing.tools);
    validate_tools_against(tools, &existing.tools)?;
    let behavior = behavior.unwrap_or(existing.behavior);
    store()
        .conn
        .execute(
            "UPDATE custom_roles SET prompt = ?2, tools = ?3, behavior = ?4 WHERE name = ?1",
            db::params![name, prompt, encode_tools(tools), behavior.as_str()],
        )
        .await?;
    Ok(())
}

/// Delete a custom role: confirm it exists, drop its record, remove the alarms
/// addressed to it, stop every branch of the role (so a running one is cancelled
/// and never resumed), then tell the owner through his Assistant.
///
/// The record goes FIRST: everything after it is best-effort, so a failure
/// leaves the role fully deleted and reported rather than half-removed — and a
/// record whose alarm removal failed still gets them removed when they fire
/// ([`crate::alarms::fire_custom_role_alarm`] drops an alarm whose target role
/// is gone).
pub(crate) async fn delete(name: &str) -> Result<()> {
    // Check existence FIRST: a nonexistent role must never mutate its alarms.
    anyhow::ensure!(get(name).await?.is_some(), "no custom role named '{name}'");
    let deleted = store()
        .conn
        .execute(
            "DELETE FROM custom_roles WHERE name = ?1",
            db::params![name],
        )
        .await?;
    // Sanity check on the row count (the existence check above already
    // guarantees it): never claim success for a delete of nothing.
    anyhow::ensure!(deleted > 0, "no custom role named '{name}'");
    // Past the record: nothing below aborts, so the owner is always told.
    let removed = match crate::alarms::delete_alarms_for_role(name).await {
        Ok(removed) => removed,
        Err(e) => {
            tracing::warn!(
                role = %name,
                error = %e,
                "Failed to remove a deleted custom role's alarms — they are dropped when they fire"
            );
            0
        }
    };
    // "Continued branches of a deleted role are never resumed — they no longer
    // have a prompt or a tool set": stop each one now, so a running launch is
    // cancelled and its deferred envelopes cannot re-drive it after a restart.
    // A failed stop must not skip the owner notice below, so log it and
    // continue; likewise a failed branch listing is best-effort (the record is
    // already gone) and must not abort the delete.
    let branches = match crate::session::store().custom_role_sessions(name).await {
        Ok(branches) => branches,
        Err(e) => {
            tracing::warn!(
                role = %name,
                error = %e,
                "Failed to list a deleted custom role's branches — continuing without stops"
            );
            Vec::new()
        }
    };
    for agent_id in &branches {
        if let Err(e) = crate::launch::stop(agent_id).await {
            tracing::warn!(
                role = %name,
                agent_id = %agent_id,
                error = %e,
                "Failed to stop a branch of the deleted custom role — continuing"
            );
        }
    }
    let content = crate::prompt::substitute(
        &crate::prompt::load_prompt("custom_role_deleted.md"),
        &[("{{role}}", name)],
    );
    if let Err(e) = crate::agent::message_router::deliver_assistant_notice(
        &crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME),
        crate::users::ADMIN_USER_NAME,
        content,
    )
    .await
    {
        tracing::warn!(
            role = %name,
            error = %e,
            "Failed to deliver the custom role deletion notice — routing best-effort"
        );
    }
    tracing::debug!(role = %name, removed, branches = branches.len(), "custom role deleted");
    Ok(())
}

/// Validate a custom role's name: non-empty, within
/// [`MAX_ROLE_NAME_CHARS`], every char ASCII alphanumeric / `_` / `-`, the
/// first char an ASCII letter, and not colliding with a built-in agent
/// namespace (see [`crate::session::custom_role_name_is_reserved`]).
pub(crate) fn validate_name(name: &str) -> Result<()> {
    validate_segment(name, MAX_ROLE_NAME_CHARS, "a custom role name")?;
    anyhow::ensure!(
        name.starts_with(|c: char| c.is_ascii_alphabetic()),
        "a custom role name must start with a letter"
    );
    anyhow::ensure!(
        !crate::session::custom_role_name_is_reserved(name),
        "a custom role name must not collide with a built-in agent namespace"
    );
    Ok(())
}

/// Validate a custom role branch identifier: the same charset and length rule
/// as a name, but it may start with any allowed char and carries no reserved
/// namespace restriction (an identifier is never the leading segment of an
/// agent address).
pub(crate) fn validate_identifier(identifier: &str) -> Result<()> {
    validate_segment(identifier, MAX_ROLE_IDENTIFIER_CHARS, "an identifier")
        .map_err(|e| anyhow::anyhow!("invalid custom role branch: {e}"))
}

/// Validate a coder branch name: the same charset and length rule as a custom
/// role identifier, with a wording of its own.
pub(crate) fn validate_coder_branch(branch: &str) -> Result<()> {
    validate_segment(branch, MAX_ROLE_IDENTIFIER_CHARS, "a branch")
        .map_err(|e| anyhow::anyhow!("invalid coder branch: {e}"))
}

/// Shared charset/length check for a name or an identifier.
fn validate_segment(value: &str, max_chars: usize, what: &str) -> Result<()> {
    anyhow::ensure!(!value.is_empty(), "{what} must not be empty");
    anyhow::ensure!(
        value.chars().count() <= max_chars,
        "{what} must be at most {max_chars} characters"
    );
    for c in value.chars() {
        anyhow::ensure!(
            c.is_ascii_alphanumeric() || c == '_' || c == '-',
            "{what} may only contain ASCII letters, digits, '_' and '-'"
        );
    }
    Ok(())
}

/// Validate a tool subset a caller wrote, against the catalog of what a custom
/// role may be given ([`crate::Role::custom_role_tool_catalog`]). Each entry is
/// `"tool"` (the tool with its narrowest kind) or `"tool:kind1,kind2"` (see
/// [`crate::tools::grants`]); an empty subset is allowed, an unknown tool name,
/// an unknown kind and a tool named twice are not.
///
/// Names the record ALREADY holds are exempt from the catalog check: a tool the
/// running installation cannot serve right now (e.g. `computer` on macOS) stays
/// on the record — the launch simply omits it — so editing another field must
/// never drop it.
pub(crate) fn validate_tools_against(tools: &[String], held: &[String]) -> Result<()> {
    let catalog = crate::Role::custom_role_tool_catalog();
    // parse() validates the grammar: an unknown kind, a kind on a kindless tool,
    // a kind named twice on one tool, and a tool named twice.
    let grants = crate::tools::grants::parse(tools)?;
    let held: Vec<&str> = held
        .iter()
        .map(|entry| crate::tools::grants::tool_name(entry))
        .collect();
    for grant in &grants {
        anyhow::ensure!(
            held.contains(&grant.tool.as_str()) || catalog.contains(&grant.tool.as_str()),
            "unknown tool '{}' for a custom role",
            grant.tool
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stored grant grammar: a tool name, or `tool:kind1,kind2`. A bare name
    /// is the tool's narrowest kind; an unknown kind, a kind on a tool that has
    /// no kinds, an unknown tool and a tool named twice are all refused.
    #[test]
    fn validate_tools_against_reads_the_kind_grammar() {
        let entries =
            |items: &[&str]| -> Vec<String> { items.iter().map(|s| (*s).to_string()).collect() };
        assert!(
            validate_tools_against(&entries(&["analyze:quick,deep", "send:coder"]), &[]).is_ok()
        );
        assert!(validate_tools_against(&entries(&["analyze", "send", "read"]), &[]).is_ok());
        assert!(
            validate_tools_against(&entries(&["analyze:sideways"]), &[]).is_err(),
            "an unknown kind is refused"
        );
        assert!(
            validate_tools_against(&entries(&["read:quick"]), &[]).is_err(),
            "a kind on a tool that has no kinds is refused"
        );
        assert!(
            validate_tools_against(&entries(&["not-a-tool"]), &[]).is_err(),
            "an unknown tool is refused"
        );
        assert!(
            validate_tools_against(&entries(&["read", "read"]), &[]).is_err(),
            "a tool named twice is refused"
        );
        assert!(
            validate_tools_against(&entries(&["analyze:quick", "analyze:deep"]), &[]).is_err(),
            "the same tool split across two entries is refused too"
        );
        assert!(
            validate_tools_against(&entries(&["gone"]), &entries(&["gone"])).is_ok(),
            "a name the record already holds stays valid"
        );
    }

    #[test]
    fn validate_name_accepts_a_good_name_and_rejects_the_rest() {
        assert!(validate_name("briefer").is_ok());
        assert!(validate_name("a_b-1").is_ok());

        // The first char must be a letter.
        assert!(validate_name("1abc").is_err());
        // The record's length bound.
        assert!(validate_name(&"a".repeat(MAX_ROLE_NAME_CHARS + 1)).is_err());
        // ':' is the branch separator and never enters a name.
        assert!(validate_name("a:b").is_err());
        // A name that merely starts with a namespace word shadows nothing.
        assert!(validate_name("rolex").is_ok());
        assert!(validate_name("roleplay").is_ok());
        // Built-in agent namespaces.
        for reserved in ["ticket", "maintainer", "role"] {
            assert!(
                validate_name(reserved).is_err(),
                "'{reserved}' must collide with a built-in namespace"
            );
        }
    }

    #[test]
    fn validate_identifier_shares_the_charset_but_allows_any_first_char() {
        assert!(validate_identifier("b").is_ok());
        assert!(validate_identifier("1-ab").is_ok());
        assert!(validate_identifier("").is_err());
        assert!(validate_identifier("a:b").is_err());
        assert!(validate_identifier(&"a".repeat(MAX_ROLE_IDENTIFIER_CHARS + 1)).is_err());
    }

    #[test]
    fn branch_addresses_are_injective() {
        let default = crate::session::custom_role_agent_id("a_b", None);
        let named = crate::session::custom_role_agent_id("a", Some("b"));
        assert_eq!(default, "role_a_b");
        assert_eq!(named, "role_a:b");
        assert_ne!(
            default, named,
            "('a_b', None) must never alias ('a', Some('b'))"
        );
    }

    /// Stopping a launch is what the recovery paths read: the branch carries the
    /// persisted mark and its deferred envelopes are gone, so boot replay and
    /// the dead-session poller both leave it alone.
    #[tokio::test]
    async fn stop_marks_the_branch_and_drops_its_deferred_envelopes() {
        crate::util::test::init_test_stores().await;
        let branch = crate::session::custom_role_agent_id("stopper", Some("night"));
        crate::util::test::seed_session_row(
            &crate::session::store().conn,
            &branch,
            crate::Role::Custom.as_str(),
            "work on it",
        )
        .await;
        let job = crate::agent::message_router::AgentJob {
            content: "a late result".to_string(),
            workspace_name: crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME),
            user_name: crate::users::ADMIN_USER_NAME.to_string(),
            channel: "gui".to_string(),
            kind: crate::agent::message_router::MessageKind::AnalyzeToolResult,
            role: crate::Role::Custom,
            reply_target: None,
            pending_job_id: None,
            originating_workspace: None,
            custom_role: Some(crate::Launch {
                role: crate::Role::Custom,
                name: "stopper".to_string(),
                identifier: Some("night".to_string()),
                prompt: "prompt".to_string(),
                tools: Vec::new(),
                agent_id: branch.clone(),
                reply_to: None,
            }),
            reply_to_custom: None,
        };
        crate::agent::message_router::persist_pending(&job, "stop_marker_pending".to_string())
            .await
            .unwrap();

        crate::launch::stop(&branch).await.unwrap();

        assert!(crate::session::store().get_launch_stopped(&branch).await);
        let left = crate::jobs::list_pending_jobs(&crate::session::store().conn)
            .await
            .unwrap();
        assert!(
            !left.iter().any(|row| row.id == "stop_marker_pending"),
            "a stopped launch's deferred envelopes are dropped"
        );
    }
}
