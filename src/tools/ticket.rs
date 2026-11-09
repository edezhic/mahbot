//! Ticket management tools for the Manager role: create, update, list, inspect.

use crate::Role;
use crate::Workspace;
use crate::pipeline::board::store as board_store;
use crate::pipeline::board::{TicketLabel, TicketParams, TicketPhase};
use crate::tools::Tool;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::fmt::Write;

pub(crate) const GET_TICKET_DESC_MAX: usize = 500;
pub(crate) const GET_TICKET_COMMENT_MAX: usize = 200;
pub(crate) const GET_TICKET_LAST_N_FULL: usize = 3;

/// Accepts a bare number (resolved to `{ws}-{number}`) or the fully prefixed form
/// for the bound workspace; a prefix from any other workspace is rejected.
fn resolve_ticket_id(ws_name: &str, raw: &str) -> Result<String> {
    let (prefix, id) = parse_ticket_id(raw)?;
    match prefix {
        None => Ok(format!("{ws_name}-{id}")),
        Some(p) if p == ws_name => Ok(id.to_string()),
        Some(_) => anyhow::bail!(
            "Ticket '{id}' belongs to a different workspace — ticket tools only manage \
             tickets from workspace '{ws_name}'"
        ),
    }
}

/// Like [`resolve_ticket_id`], but a well-formed prefixed ID from any workspace is
/// taken as written — the archive search spans the installation, and a hit is
/// useless if the ticket cannot be opened wherever it was filed.
fn resolve_any_ticket_id(ws_name: &str, raw: &str) -> Result<String> {
    let (prefix, id) = parse_ticket_id(raw)?;
    match prefix {
        None => Ok(format!("{ws_name}-{id}")),
        Some(_) => Ok(id.to_string()),
    }
}

/// The prefix is `None` for a bare number — resolved against the bound workspace —
/// and `Some(prefix)` for a well-formed `{prefix}-{seq}`, where `id` is the trimmed
/// raw, i.e. the full prefixed ticket ID. The prefix shape mirrors
/// `workspace::validate_name` ([a-zA-Z_]+ starting with a letter).
///
/// Shared with the admin's `workspace` tool, which accepts only the full
/// `<workspace>-<number>` spelling.
pub(crate) fn parse_ticket_id(raw: &str) -> Result<(Option<&str>, &str)> {
    let id = raw.trim();
    anyhow::ensure!(!id.is_empty(), "Ticket ID must not be empty");
    let seq_ok = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if seq_ok(id) {
        return Ok((None, id));
    }
    if let Some((prefix, seq)) = id.rsplit_once('-') {
        let prefix_ok = prefix.starts_with(|c: char| c.is_ascii_alphabetic())
            && prefix.chars().all(|c| c.is_ascii_alphabetic() || c == '_');
        if prefix_ok && seq_ok(seq) {
            return Ok((Some(prefix), id));
        }
    }
    anyhow::bail!("Invalid ticket ID '{id}' — expected a bare number or '<workspace>-<number>'")
}

/// The `reporter` is fixed at construction from the using role — invisible to the
/// agent, no parameter to pass. Bound to one workspace at construction time.
pub struct CreateTicketTool {
    reporter: String,
    ws_name: String,
}

impl CreateTicketTool {
    pub fn new(reporter: impl Into<String>, ws: &Workspace) -> Self {
        Self {
            reporter: reporter.into(),
            ws_name: ws.name.clone(),
        }
    }

    /// Only the Manager states `priority` and `label`; every other creator gets
    /// the fixed defaults.
    fn is_manager(&self) -> bool {
        self.reporter == crate::Role::Manager.as_str()
    }

    fn build_params(
        &self,
        title: &str,
        description: &str,
        prerequisites: &[String],
        priority: i64,
        label: TicketLabel,
    ) -> TicketParams {
        TicketParams {
            title: title.to_string(),
            description: description.to_string(),
            workspace_name: self.ws_name.clone(),
            phase: TicketPhase::Backlog,
            prerequisites: prerequisites.to_vec(),
            reporter: self.reporter.clone(),
            priority,
            label,
        }
    }
}

#[async_trait]
impl Tool for CreateTicketTool {
    fn name(&self) -> &'static str {
        "create_ticket"
    }

    /// The shared instructions plus the Manager-only fragment, so a holder
    /// without those arguments never learns they exist.
    fn description(&self) -> String {
        let mut parts = vec![crate::prompt::load_prompt("tool/create_ticket.md")];
        if self.is_manager() {
            parts.push(crate::prompt::load_prompt("tool/create_ticket_manager.md"));
        }
        parts.join("\n\n")
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let mut props = serde_json::Map::new();
        props.insert(
            "title".into(),
            json!({
                "type": "string",
                "description": "Short title/summary of the task"
            }),
        );
        props.insert(
            "description".into(),
            json!({
                "type": "string",
                "description": "Detailed description of the task"
            }),
        );
        props.insert(
            "prerequisites".into(),
            json!({
                "type": "array",
                "description": "Optional list of ticket IDs that must be completed (done, archived, or cancelled) before this ticket can be claimed",
                "items": {
                    "type": "string"
                }
            }),
        );
        props.insert(
            "supersede".into(),
            json!({
                "type": "string",
                "description": "Optional ticket ID to supersede — atomically cancels the old ticket, creates this one as its replacement, and rewires any dependents to point to the new ID. A finished (done) ticket cannot be superseded."
            }),
        );

        // The two management arguments belong to the Manager alone: the
        // Maintainer's priority is hardcoded and every ticket it files is `maint`.
        let mut required = vec!["title", "description"];
        if self.is_manager() {
            props.insert(
                "priority".into(),
                json!({
                    "type": "integer",
                    "description": "Priority level: 0 = highest urgency, 1 (default), 2, 3, ... Higher numbers = lower priority"
                }),
            );
            props.insert(
                "label".into(),
                json!({
                    "type": "string",
                    "enum": ["feat", "fix", "maint"],
                    "description": "Classification label: `feat` adds a capability, `fix` repairs a problem, `maint` is everything else"
                }),
            );
            required.push("label");
        }

        super::tool_params_schema(&serde_json::Value::Object(props), &required)
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let title = super::get_str(&args, "title")?;
        let description = super::get_str(&args, "description")?;

        let prerequisites = super::get_str_array(&args, "prerequisites")?
            .iter()
            .map(|s| resolve_ticket_id(&self.ws_name, s))
            .collect::<Result<Vec<_>>>()?;

        let supersede_id = super::get_opt_str_strict(&args, "supersede")?
            .map(|s| resolve_ticket_id(&self.ws_name, s))
            .transpose()?;

        // A value outside the three is refused; every other creator files a
        // maintenance ticket. Set once at creation — never written again; a
        // superseding ticket states its own.
        let label = if self.is_manager() {
            super::get_str(&args, "label")?.parse::<TicketLabel>()?
        } else {
            TicketLabel::Maint
        };

        // Manager may set it explicitly; otherwise inherit from the superseded
        // ticket, else the role default.
        let explicit_priority = if self.is_manager() {
            super::get_opt_i64(&args, "priority")?
        } else {
            None
        };

        let store = board_store();

        let prereq_note = if prerequisites.is_empty() {
            String::new()
        } else {
            format!(" with prerequisites: {}", prerequisites.join(", "))
        };

        let priority: i64 = match (&supersede_id, explicit_priority) {
            (Some(supersede_id), None) => {
                // No explicit priority — inherit from the superseded ticket.
                // Priority is immutable after ticket creation, so reading it
                // outside the supersede transaction has no TOCTOU race.
                match store.get_ticket_priority(supersede_id).await? {
                    Some(p) => p,
                    None => anyhow::bail!(
                        "Superseded ticket {supersede_id} not found when reading priority",
                    ),
                }
            }
            (_, Some(p)) => p,
            // Maintainer gets 3; the manager and every other reporter default to 1.
            (None, None) => {
                if self.reporter == "maintainer" {
                    3
                } else {
                    1
                }
            }
        };

        let params = self.build_params(title, description, &prerequisites, priority, label);
        if let Some(supersede_id) = supersede_id {
            guard_not_pipeline_occupied(store, &supersede_id).await?;

            let id = store
                .supersede_and_create(&supersede_id, &params, &self.reporter)
                .await?;
            Ok(format!(
                "Superseded {supersede_id} → created ticket {id}: {title}{prereq_note}"
            ))
        } else {
            let id = store.create_ticket(&params).await?;
            Ok(format!("Created ticket {id}: {title}{prereq_note}"))
        }
    }
}

pub struct UpdateTicketTool {
    ws_name: String,
}

impl UpdateTicketTool {
    pub fn new(ws: &Workspace) -> Self {
        Self {
            ws_name: ws.name.clone(),
        }
    }
}

#[async_trait]
impl Tool for UpdateTicketTool {
    fn name(&self) -> &'static str {
        "update_ticket"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "ticket_id": {
                    "type": "string",
                    "description": "The ticket id"
                },
                "phase": {
                    "type": "string",
                    "description": "New phase for the ticket. Valid manual transitions: backlog (return to queue), planning (paused state awaiting further decision whether to proceed with the ticket or cancel it), queued (send to engineer), cancelled (abandon), failed (mark unsuccessful), done (mark complete; irreversible). Do NOT manually set other pipeline-managed phases (analysis, in_development, verification, in_sanitation) — the board poller handles these automatically and manual transitions will interfere with running agents. A finished (done) ticket's phase is immutable: it cannot be moved to any other phase."
                }
            }),
            &["ticket_id", "phase"],
        )
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let ticket_id = resolve_ticket_id(&self.ws_name, super::get_str(&args, "ticket_id")?)?;
        let new_phase = super::get_str(&args, "phase")?;

        let parsed_phase = new_phase.parse::<TicketPhase>()?;

        let store = board_store();

        guard_not_pipeline_occupied(store, &ticket_id).await?;

        store
            .transition_to(&ticket_id, None, parsed_phase, Role::Manager.as_str())
            .await?;

        Ok(format!("Ticket {ticket_id} phase updated to '{new_phase}'"))
    }
}

pub struct ListTicketsTool {
    ws_name: String,
}

impl ListTicketsTool {
    pub fn new(ws: &Workspace) -> Self {
        Self {
            ws_name: ws.name.clone(),
        }
    }
}

#[async_trait]
impl Tool for ListTicketsTool {
    fn name(&self) -> &'static str {
        "list_tickets"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "phase": {
                    "type": "string",
                    "description": "Optional phase filter (e.g. 'queued', 'in_development', 'done', 'cancelled'). When omitted, 'done' and 'cancelled' tickets are excluded — use an explicit phase filter to include them. Use 'search_archived_tickets' to find archived tickets."
                }
            }),
            &[],
        )
    }

    fn side_effects(&self) -> bool {
        false // read-only board query
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let raw_phase = super::get_opt_str(&args, "phase");

        // "archived" is no longer a phase — redirect to search_archived_tickets.
        if let Some(s) = raw_phase
            && s.eq_ignore_ascii_case("archived")
        {
            anyhow::bail!(
                "Archived tickets are not accessible via 'list_tickets'. \
                     Use 'search_archived_tickets' to search the archive."
            );
        }

        let phase_filter = match raw_phase {
            Some(s) => {
                let parsed = s.parse::<TicketPhase>()?;
                Some(parsed)
            }
            None => None,
        };

        let store = board_store();

        let mut tickets = store
            .list_all_tickets(Some(&self.ws_name), phase_filter)
            .await?;

        // Without a filter, unblocking phases are excluded; an explicit filter is
        // respected as-is.
        if phase_filter.is_none() {
            tickets.retain(|t| !t.phase.is_unblocking());
        }

        if tickets.is_empty() {
            return Ok("No tickets found.".to_string());
        }

        let mut output = String::from("Tickets:\n");
        for t in &tickets {
            let _ = writeln!(output, "{}", t.short_display());
        }

        Ok(output)
    }
}

/// Unlike the write tools, not confined to its bound workspace: a fully prefixed
/// ID from any workspace is accepted (the archive search spans the installation,
/// so its hits must open wherever they were filed), while a bare number still
/// resolves against the bound workspace.
pub struct GetTicketTool {
    reporter: String,
    ws_name: String,
}

impl GetTicketTool {
    pub fn new(reporter: impl Into<String>, ws: &Workspace) -> Self {
        Self {
            reporter: reporter.into(),
            ws_name: ws.name.clone(),
        }
    }
}

#[async_trait]
impl Tool for GetTicketTool {
    fn name(&self) -> &'static str {
        "get_ticket"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "ticket_id": {
                    "type": "string",
                    "description": "The ticket id"
                },
                "full": {
                    "type": "boolean",
                    "description": "Return the complete un-truncated ticket. Default false."
                }
            }),
            &["ticket_id"],
        )
    }

    fn side_effects(&self) -> bool {
        false // read-only board query
    }

    fn preserve_full_output(&self) -> bool {
        // Ticket content can exceed the ~5 KB tool-output trigger — the LLM needs
        // it in full.
        true
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let ticket_id = resolve_any_ticket_id(&self.ws_name, super::get_str(&args, "ticket_id")?)?;
        let full = super::get_bool(&args, "full", false)?;

        let store = board_store();

        let Some(ticket) = store.get_ticket(&ticket_id).await? else {
            anyhow::bail!("Ticket {ticket_id} not found");
        };

        if full || ticket.reporter != self.reporter {
            Ok(ticket.detailed_display())
        } else {
            Ok(ticket.detailed_display_limited(
                GET_TICKET_DESC_MAX,
                GET_TICKET_COMMENT_MAX,
                GET_TICKET_LAST_N_FULL,
            ))
        }
    }
}

pub struct AddCommentTool {
    ws_name: String,
}

impl AddCommentTool {
    pub fn new(ws: &Workspace) -> Self {
        Self {
            ws_name: ws.name.clone(),
        }
    }
}

#[async_trait]
impl Tool for AddCommentTool {
    fn name(&self) -> &'static str {
        "add_comment"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "ticket_id": {
                    "type": "string",
                    "description": "The ID of the ticket to comment on"
                },
                "content": {
                    "type": "string",
                    "description": "The comment content"
                }
            }),
            &["ticket_id", "content"],
        )
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let ticket_id = resolve_ticket_id(&self.ws_name, super::get_str(&args, "ticket_id")?)?;
        let content = super::get_str(&args, "content")?;

        let store = board_store();

        store
            .add_comment(&ticket_id, Role::Manager.as_str(), content)
            .await?;

        Ok(format!("Comment added to ticket {ticket_id}"))
    }
}

/// Refuses a ticket in a pipeline-occupied phase. Used by create-ticket (when
/// superseding) and update-ticket to keep in-flight tickets unmodified during
/// automated processing. `add_comment` deliberately bypasses it — comments are
/// allowed mid-pipeline and reach running agents as soft deferred messages.
async fn guard_not_pipeline_occupied(
    store: &crate::pipeline::board::BoardStore,
    ticket_id: &str,
) -> Result<()> {
    if let Some(current_phase) = store.get_ticket_phase(ticket_id).await?
        && current_phase.is_pipeline_occupied()
    {
        anyhow::bail!(
            "Ticket {ticket_id} is currently in phase '{current_phase}' — \
                 in-flight tickets cannot be modified through automated tools. \
                 Use the GUI to cancel or manage this ticket manually.",
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::TicketBuilder;
    use crate::util::test::expect_ticket;
    use crate::util::test::make_ticket;
    use crate::workspace::test_ws;
    use serde_json::json;

    /// Required of the Manager, refused outside the three values, and never offered
    /// to a creator that does not state it — whose tickets are `maint`.
    #[tokio::test]
    async fn create_ticket_labels() {
        crate::util::test::init_test_stores().await;
        let store = board_store();
        let ws = test_ws("/tmp/test_ws_labels");
        let args = |title: &str| json!({ "title": title, "description": "desc" });
        let created_id = |answer: &str| {
            answer
                .strip_prefix("Created ticket ")
                .and_then(|rest| rest.split(':').next())
                .expect("the answer names the created ticket")
                .to_string()
        };

        let manager = CreateTicketTool::new("manager", &ws);
        assert!(
            manager.parameters_schema()["required"]
                .as_array()
                .is_some_and(|required| required.iter().any(|name| name.as_str() == Some("label"))),
            "the manager must be required to state a label"
        );
        let err = manager
            .execute(&ws, args("No label"))
            .await
            .expect_err("a manager's ticket without a label is refused")
            .to_string();
        assert!(err.contains("missing required argument \"label\""), "{err}");

        let mut bad = args("Bad label");
        bad["label"] = json!("docs");
        let err = manager
            .execute(&ws, bad)
            .await
            .expect_err("a label outside the three is refused")
            .to_string();
        assert!(err.contains("feat, fix, maint"), "{err}");

        let mut fix = args("Fix it");
        fix["label"] = json!("fix");
        let answer = manager
            .execute(&ws, fix)
            .await
            .expect("the ticket is created");
        assert_eq!(
            expect_ticket(store, &created_id(&answer)).await.label,
            TicketLabel::Fix
        );

        // A holder that never states the label does not learn it exists, and files `maint`.
        let maintainer = CreateTicketTool::new("maintainer", &ws);
        assert!(
            maintainer.parameters_schema()["properties"]
                .get("label")
                .is_none()
        );
        assert!(!maintainer.description().contains("label"));
        let answer = maintainer
            .execute(&ws, args("Tidy up"))
            .await
            .expect("the ticket is created");
        assert_eq!(
            expect_ticket(store, &created_id(&answer)).await.label,
            TicketLabel::Maint
        );
    }

    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // shared global board — a concurrent boot reset would clobber the Verification fixture
    async fn test_update_ticket_tool() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/ws");
        let id = make_ticket(
            store,
            &ws,
            "Test",
            crate::pipeline::board::TicketPhase::Backlog,
        )
        .await;

        let tool = UpdateTicketTool::new(&ws);
        let args = json!({
            "ticket_id": id,
            "phase": "verification"
        });
        tool.execute(&ws, args).await.expect("execute");
        let ticket = store
            .get_ticket(&id)
            .await
            .expect("get_ticket")
            .expect("ticket exists");
        assert_eq!(ticket.phase, TicketPhase::Verification);
    }

    #[tokio::test]
    async fn test_list_tickets_tool() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/tmp_ws");

        let _id_a = make_ticket(
            store,
            &ws,
            "A",
            crate::pipeline::board::TicketPhase::Backlog,
        )
        .await;
        let id_b = make_ticket(
            store,
            &ws,
            "B",
            crate::pipeline::board::TicketPhase::Backlog,
        )
        .await;
        let id_c = make_ticket(
            store,
            &ws,
            "C",
            crate::pipeline::board::TicketPhase::Backlog,
        )
        .await;
        store
            .transition_to(&id_c, Some(TicketPhase::Backlog), TicketPhase::Done, "test")
            .await
            .expect("transition to done");
        store
            .transition_to(
                &id_b,
                Some(TicketPhase::Backlog),
                TicketPhase::Cancelled,
                "test",
            )
            .await
            .expect("transition to cancelled");

        let tool = ListTicketsTool::new(&ws);

        let args = json!({});
        let result = tool.execute(&ws, args).await.expect("execute");
        assert!(
            result.contains('A'),
            "Default listing should include active ticket A"
        );
        assert!(
            !result.contains('B'),
            "Default listing should exclude cancelled ticket B"
        );
        assert!(
            !result.contains('C'),
            "Default listing should exclude done ticket C"
        );

        let args = json!({"phase": "done"});
        let result = tool.execute(&ws, args).await.expect("execute");
        assert!(
            result.contains('C'),
            "Explicit 'done' filter should include done ticket C"
        );
        assert!(
            !result.contains('A'),
            "Explicit 'done' filter should exclude active ticket A"
        );

        let args = json!({"phase": "cancelled"});
        let result = tool.execute(&ws, args).await.expect("execute");
        assert!(
            result.contains('B'),
            "Explicit 'cancelled' filter should include cancelled ticket B"
        );

        let args = json!({"phase": "backlog"});
        let result = tool.execute(&ws, args).await.expect("execute");
        assert!(
            result.contains('A'),
            "Explicit 'backlog' filter should include ticket A"
        );
    }

    /// The phase the ticket is in, and the comment it must carry afterwards.
    struct AddCommentCase {
        name: &'static str,
        phase: TicketPhase,
        content: &'static str,
    }

    #[tokio::test]
    async fn add_comment_cases() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/ws");
        for case in [
            AddCommentCase {
                name: "a comment on a backlog ticket is stored",
                phase: TicketPhase::Backlog,
                content: "This is a test comment",
            },
            // Commenting on a pipeline-occupied ticket is not a mutation.
            AddCommentCase {
                name: "a comment on a pipeline-occupied ticket is allowed",
                phase: TicketPhase::InDevelopment,
                content: "This should be allowed now",
            },
        ] {
            let id = TicketBuilder::new(store, &ws)
                .title("CommentTest")
                .desc("add a comment")
                .phase(case.phase)
                .create()
                .await
                .unwrap_or_else(|err| panic!("[{}] create: {err}", case.name));

            let result = AddCommentTool::new(&ws)
                .execute(&ws, json!({"ticket_id": id, "content": case.content}))
                .await
                .unwrap_or_else(|err| panic!("[{}] a comment is allowed here: {err}", case.name));
            assert!(
                result.contains(&id),
                "[{}] the answer names the ticket: {result}",
                case.name
            );

            let ticket = expect_ticket(store, &id).await;
            let stored = ticket.comments.last().expect("at least one comment");
            assert_eq!(stored.role, Role::Manager.as_str(), "[{}] role", case.name);
            assert_eq!(stored.content, case.content, "[{}] content", case.name);
        }
    }

    /// One tool that must refuse an unknown phase.
    enum InvalidPhaseTool {
        Update,
        List,
    }

    struct InvalidPhaseCase {
        name: &'static str,
        tool: InvalidPhaseTool,
    }

    #[tokio::test]
    async fn invalid_phase_cases() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/ws");
        let id = make_ticket(
            store,
            &ws,
            "Test",
            crate::pipeline::board::TicketPhase::Backlog,
        )
        .await;

        for case in [
            InvalidPhaseCase {
                name: "an update refuses an unknown phase",
                tool: InvalidPhaseTool::Update,
            },
            InvalidPhaseCase {
                name: "a listing refuses an unknown phase",
                tool: InvalidPhaseTool::List,
            },
        ] {
            let result = match case.tool {
                InvalidPhaseTool::Update => {
                    UpdateTicketTool::new(&ws)
                        .execute(
                            &ws,
                            json!({"ticket_id": id.clone(), "phase": "invalid_phase"}),
                        )
                        .await
                }
                InvalidPhaseTool::List => {
                    ListTicketsTool::new(&ws)
                        .execute(&ws, json!({"phase": "bogus_phase"}))
                        .await
                }
            };
            assert!(result.is_err(), "[{}] invalid phase should fail", case.name);
        }
    }

    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // shared global board — a concurrent boot reset would clobber the InDevelopment fixture
    async fn test_guard_not_pipeline_occupied_cases() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/ws");

        let occupied_id = TicketBuilder::new(store, &ws)
            .title("OccupiedGuard")
            .desc("in development")
            .phase(TicketPhase::InDevelopment)
            .create()
            .await
            .expect("create");

        let unoccupied_id = TicketBuilder::new(store, &ws)
            .title("UnoccupiedGuard")
            .desc("backlog")
            .phase(TicketPhase::Backlog)
            .create()
            .await
            .expect("create");

        let result = guard_not_pipeline_occupied(store, &occupied_id).await;
        assert!(result.is_err(), "Occupied-phase ticket should be rejected");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("in-flight"),
            "Error should mention in-flight restriction"
        );

        let result = guard_not_pipeline_occupied(store, &unoccupied_id).await;
        assert!(result.is_ok(), "Unoccupied ticket should pass the guard");

        // Nothing to block on a non-existent ticket.
        let result = guard_not_pipeline_occupied(store, "nonexistent_id").await;
        assert!(
            result.is_ok(),
            "Non-existent ticket should silently pass the guard"
        );
    }

    /// A ticket in a pipeline-occupied phase (InDevelopment), for the wiring tests
    /// that verify tools call `guard_not_pipeline_occupied`.
    async fn create_occupied_ticket() -> String {
        crate::util::test::init_test_stores().await;
        let store = board_store();
        TicketBuilder::new(store, &test_ws("/ws"))
            .title("BlockMe")
            .desc("in flight")
            .phase(TicketPhase::InDevelopment)
            .create()
            .await
            .expect("create")
    }

    /// One mutation that the pipeline-occupied guard must refuse.
    enum BlockedMutation {
        Update,
        SupersedeCreate,
    }

    struct BlockedMutationCase {
        name: &'static str,
        mutation: BlockedMutation,
    }

    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // shared global board — a concurrent boot reset would clobber the InDevelopment fixture
    async fn pipeline_occupied_mutation_cases() {
        for case in [
            BlockedMutationCase {
                name: "an update is refused",
                mutation: BlockedMutation::Update,
            },
            BlockedMutationCase {
                name: "a supersede is refused",
                mutation: BlockedMutation::SupersedeCreate,
            },
        ] {
            let id = create_occupied_ticket().await;
            let ws = test_ws("/ws");
            let result = match case.mutation {
                BlockedMutation::Update => {
                    UpdateTicketTool::new(&ws)
                        .execute(&ws, json!({"ticket_id": id, "phase": "backlog"}))
                        .await
                }
                BlockedMutation::SupersedeCreate => {
                    CreateTicketTool::new("test", &ws)
                        .execute(
                            &ws,
                            json!({
                                "title": "Replacement",
                                "description": "trying to supersede",
                                "supersede": id,
                            }),
                        )
                        .await
                }
            };
            assert!(
                result.is_err(),
                "[{}] a pipeline-occupied ticket cannot be modified",
                case.name
            );
        }
    }

    /// The explicit priority (if any) and the one the replacement must end up with.
    struct SupersedePriorityCase {
        name: &'static str,
        explicit: Option<i64>,
        expected: i64,
    }

    #[tokio::test]
    async fn supersede_priority_cases() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/tmp/test_ws_supersede_priority");

        for case in [
            SupersedePriorityCase {
                name: "an omitted priority is inherited from the superseded ticket",
                explicit: None,
                expected: 0,
            },
            SupersedePriorityCase {
                name: "an explicit priority overrides inheritance",
                explicit: Some(5),
                expected: 5,
            },
        ] {
            // 0 = highest urgency.
            let old_id = TicketBuilder::new(store, &ws)
                .title("Old ticket")
                .priority(0)
                .create()
                .await
                .expect("create old ticket");

            let mut args = json!({
                "title": "Replacement",
                "description": "superseding the old ticket",
                "supersede": &old_id,
                // The replacement inherits neither priority nor label.
                "label": "fix",
            });
            if let Some(priority) = case.explicit {
                args["priority"] = json!(priority);
            }
            let result = CreateTicketTool::new("manager", &ws)
                .execute(&ws, args)
                .await
                .expect("supersede should succeed");
            assert!(
                result.contains("Superseded"),
                "[{}] expected supersede output: {result}",
                case.name
            );

            let old_ticket = expect_ticket(store, &old_id).await;
            let new_id = old_ticket
                .superseded_by
                .as_deref()
                .expect("old ticket should have superseded_by");
            let new_ticket = expect_ticket(store, new_id).await;
            assert_eq!(
                new_ticket.priority, case.expected,
                "[{}] replacement priority",
                case.name
            );
            assert_eq!(
                new_ticket.label,
                TicketLabel::Fix,
                "[{}] the replacement states its own label",
                case.name
            );
            assert_eq!(old_ticket.label, TicketLabel::Feat);
        }
    }

    #[test]
    fn test_resolve_ticket_id() {
        assert_eq!(resolve_ticket_id("ws", "123").unwrap(), "ws-123");
        assert_eq!(resolve_ticket_id("ws", "  7 ").unwrap(), "ws-7");
        assert_eq!(resolve_ticket_id("ws", "ws-123").unwrap(), "ws-123");
        assert_eq!(resolve_ticket_id("my_ws", "my_ws-42").unwrap(), "my_ws-42");
        let err = format!("{}", resolve_ticket_id("ws", "other-123").unwrap_err());
        assert!(
            err.contains("different workspace"),
            "foreign prefix should be rejected: {err}"
        );
        for bad in [
            "", "abc", "-5", "ws-abc", "123-456", "ws-", "ws-12-34", "___-123",
        ] {
            assert!(
                resolve_ticket_id("ws", bad).is_err(),
                "malformed ID '{bad}' should be rejected"
            );
        }

        // Binds a bare number to the bound workspace but takes any well-formed
        // prefix verbatim; malformed IDs are the shared parser's business.
        assert_eq!(resolve_any_ticket_id("ws", "123").unwrap(), "ws-123");
        assert_eq!(resolve_any_ticket_id("ws", "  7 ").unwrap(), "ws-7");
        assert_eq!(
            resolve_any_ticket_id("ws", "other-123").unwrap(),
            "other-123"
        );
        assert_eq!(resolve_any_ticket_id("ws", "my_ws-42").unwrap(), "my_ws-42");
    }

    #[tokio::test]
    async fn test_get_ticket_id_resolution() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/ws");
        let id = TicketBuilder::new(store, &ws)
            .title("Bound")
            .desc("binding test")
            .create()
            .await
            .expect("create");

        let ticket = crate::util::test::expect_ticket(store, &id).await;
        let expected = ticket.detailed_display();

        let seq = id.strip_prefix("ws-").expect("prefixed id");
        let result = GetTicketTool::new("manager", &ws)
            .execute(&ws, json!({"ticket_id": seq, "full": true}))
            .await
            .expect("bare number should resolve");
        assert_eq!(result, expected);

        // Accepted even from another workspace — the archive search spans the
        // installation, so its hits must open here.
        let foreign = test_ws("/other");
        let result = GetTicketTool::new("manager", &foreign)
            .execute(&foreign, json!({"ticket_id": id, "full": true}))
            .await
            .expect("a prefixed ID from any workspace is accepted");
        assert_eq!(result, expected);
    }
}
