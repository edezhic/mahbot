//! Ticket management tools for the Manager role.
//!
//! These tools allow the Manager to create tickets on the board,
//! update their phase, list them, and inspect individual tickets.

use crate::Role;
use crate::Workspace;
use crate::pipeline::board::store as board_store;
use crate::pipeline::board::{TicketParams, TicketPhase};
use crate::tools::Tool;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::fmt::Write;

const GET_TICKET_DESC_MAX: usize = 500;
const GET_TICKET_COMMENT_MAX: usize = 200;
const GET_TICKET_LAST_N_FULL: usize = 3;

// ── Ticket ID resolution ─────────────────────────────────────────

/// Resolve a raw ticket ID argument against the tool's bound workspace.
///
/// Accepts either a bare number (resolved to `{ws}-{number}`) or the fully
/// prefixed form for the bound workspace. Prefixed IDs from any other
/// workspace are rejected with a clear error.
fn resolve_ticket_id(ws_name: &str, raw: &str) -> Result<String> {
    let id = raw.trim();
    anyhow::ensure!(!id.is_empty(), "Ticket ID must not be empty");
    let seq_ok = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    // Bare number → bound workspace ticket.
    if seq_ok(id) {
        return Ok(format!("{ws_name}-{id}"));
    }
    // Prefixed form `{workspace}-{seq}` — only the bound workspace is accepted.
    // The prefix shape mirrors workspace::validate_name ([a-zA-Z_]+ starting with
    // a letter); keep the two grammars in sync.
    if let Some((prefix, seq)) = id.rsplit_once('-') {
        let prefix_ok = prefix.starts_with(|c: char| c.is_ascii_alphabetic())
            && prefix.chars().all(|c| c.is_ascii_alphabetic() || c == '_');
        if prefix_ok && seq_ok(seq) {
            if prefix == ws_name {
                return Ok(id.to_string());
            }
            anyhow::bail!(
                "Ticket '{id}' belongs to a different workspace — ticket tools only manage \
                 tickets from workspace '{ws_name}'"
            );
        }
    }
    anyhow::bail!("Invalid ticket ID '{id}' — expected a bare number or '{ws_name}-<number>'");
}

// ── CreateTicketTool ─────────────────────────────────────────────

/// Tool for creating tickets. The `reporter` field is set at construction
/// time based on which role is using the tool — invisible to the agent,
/// no parameter to pass. Bound to a single workspace at construction time.
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

    /// Build a [`TicketParams`] from the shared fields used by both creation branches.
    fn build_params(
        &self,
        title: &str,
        description: &str,
        prerequisites: &[String],
        embedding_bytes: Option<Vec<u8>>,
        priority: i64,
    ) -> TicketParams {
        TicketParams {
            title: title.to_string(),
            description: description.to_string(),
            workspace_name: self.ws_name.clone(),
            phase: TicketPhase::Backlog,
            prerequisites: prerequisites.to_vec(),
            reporter: self.reporter.clone(),
            embedding: embedding_bytes,
            priority,
        }
    }
}

#[async_trait]
impl Tool for CreateTicketTool {
    fn name(&self) -> &'static str {
        "create_ticket"
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
                "description": "Optional ticket ID to supersede — atomically cancels the old ticket, creates this one as its replacement, and rewires any dependents to point to the new ID"
            }),
        );

        // Only Manager sees the priority parameter — Maintainer always uses
        // a hardcoded value and must not have it in the schema.
        if self.reporter == "manager" {
            props.insert(
                "priority".into(),
                json!({
                    "type": "integer",
                    "description": "Priority level: 0 = highest urgency, 1 (default), 2, 3, ... Higher numbers = lower priority"
                }),
            );
        }

        super::tool_params_schema(&serde_json::Value::Object(props), &["title", "description"])
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let title = super::get_str(&args, "title")?;
        let description = super::get_str(&args, "description")?;

        let prerequisites: Vec<String> = match args.get("prerequisites") {
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .map(|v| {
                    let s = v.as_str().ok_or_else(|| {
                        anyhow::anyhow!("prerequisites must contain ticket ID strings, got {v}")
                    })?;
                    resolve_ticket_id(&self.ws_name, s)
                })
                .collect::<Result<_>>()?,
            Some(_) => {
                anyhow::bail!("prerequisites must be an array of ticket ID strings");
            }
            None => Vec::new(),
        };

        let supersede_id: Option<String> = match args.get("supersede") {
            Some(serde_json::Value::String(s)) => Some(resolve_ticket_id(&self.ws_name, s)?),
            Some(_) => anyhow::bail!("supersede must be a ticket ID string"),
            None => None,
        };

        // Priority: Manager can set explicitly; otherwise inherit from the
        // superseded ticket (if superseding) or use the role default.
        let explicit_priority: Option<i64> = if self.reporter == "manager" {
            args.get("priority").and_then(serde_json::Value::as_i64)
        } else {
            None
        };

        let store = board_store();
        let embedding_bytes: Option<Vec<u8>> =
            crate::embedder::embed_document(title).map(|v| crate::vector::vec_to_bytes(&v));

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
            (None, None) => {
                if self.reporter == "manager" {
                    1
                } else if self.reporter == "maintainer" {
                    3
                } else {
                    1
                }
            }
        };

        let params = self.build_params(
            title,
            description,
            &prerequisites,
            embedding_bytes,
            priority,
        );
        if let Some(supersede_id) = supersede_id {
            guard_not_pipeline_occupied(store, &supersede_id).await?;

            let id = store.supersede_and_create(&supersede_id, &params).await?;
            Ok(format!(
                "Superseded {supersede_id} → created ticket {id}: {title}{prereq_note}"
            ))
        } else {
            let id = store.create_ticket(&params).await?;
            Ok(format!("Created ticket {id}: {title}{prereq_note}"))
        }
    }
}

// ── UpdateTicketTool ─────────────────────────────────────────────

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
                    "description": "New phase for the ticket. Valid manual transitions: backlog (return to queue), planning (paused state awaiting further decision whether to proceed with the ticket or cancel it), queued (send to engineer), cancelled (abandon), failed (mark unsuccessful), done (mark complete). Do NOT manually set other pipeline-managed phases (analysis, in_development, in_diagnostics, in_review, in_qa, in_sanitation) — the board poller handles these automatically and manual transitions will interfere with running agents."
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

        // Guard: refuse to update a ticket that is in a pipeline-occupied phase.
        guard_not_pipeline_occupied(store, &ticket_id).await?;

        store.transition_to(&ticket_id, None, parsed_phase).await?;

        Ok(format!("Ticket {ticket_id} phase updated to '{new_phase}'"))
    }
}

// ── ListTicketsTool ─────────────────────────────────────────────

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

        // "archived" is no longer a phase — redirect to search_archived_tickets
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

        // Default: exclude unblocking phases. When an explicit phase filter
        // is provided, respect it as-is.
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

// ── GetTicketTool ───────────────────────────────────────────────

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
        // Ticket content (descriptions, comments) can exceed the 5 KB
        // budget — the LLM needs it in full.
        true
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let ticket_id = resolve_ticket_id(&self.ws_name, super::get_str(&args, "ticket_id")?)?;
        let full = super::get_bool(&args, "full", false);

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

// ── AddCommentTool ──────────────────────────────────────────────

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

/// Guard: refuse to proceed if the ticket is in a pipeline-occupied phase.
/// The create-ticket tool (when superseding an existing ticket) and the
/// update-ticket tool use this to prevent modifications to in-flight tickets
/// during automated pipeline processing. `add_comment` deliberately bypasses
/// it — comments are allowed mid-pipeline and are delivered to running agents
/// as soft deferred messages.
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

    fn comment(role: &str, content: &str, ts: &str) -> crate::pipeline::board::TicketComment {
        crate::pipeline::board::TicketComment {
            role: role.to_string(),
            content: content.to_string(),
            created_at: ts.to_string(),
        }
    }

    #[tokio::test]
    async fn test_create_ticket_tool() {
        crate::util::test::init_test_stores().await;

        let ws = test_ws("/tmp/test_ws");
        let tool = CreateTicketTool::new("test", &ws);
        let args = json!({
            "title": "Test ticket",
            "description": "A test description",
        });
        let result = tool.execute(&ws, args).await.expect("execute");
        assert!(
            result.contains("Test ticket"),
            "Output should contain title"
        );
    }

    #[tokio::test]
    async fn test_create_ticket_rejects_non_string_prerequisite() {
        crate::util::test::init_test_stores().await;

        let ws = test_ws("/tmp/test_ws");
        let tool = CreateTicketTool::new("test", &ws);
        let args = json!({
            "title": "Test",
            "description": "desc",
            "prerequisites": [123],
        });
        let err = tool.execute(&ws, args).await.unwrap_err();
        assert!(
            format!("{err}").contains("ticket ID strings"),
            "non-string prerequisite should be rejected: {err}"
        );
    }

    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // shared global board — a concurrent boot reset would clobber the InQa fixture
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
            "phase": "in_qa"
        });
        tool.execute(&ws, args).await.expect("execute");
        let ticket = store
            .get_ticket(&id)
            .await
            .expect("get_ticket")
            .expect("ticket exists");
        assert_eq!(ticket.phase, TicketPhase::InQa);
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
        // Transition C to 'done' and B to 'cancelled'
        store
            .transition_to(&id_c, Some(TicketPhase::Backlog), TicketPhase::Done)
            .await
            .expect("transition to done");
        store
            .transition_to(&id_b, Some(TicketPhase::Backlog), TicketPhase::Cancelled)
            .await
            .expect("transition to cancelled");

        let tool = ListTicketsTool::new(&ws);

        // Default (no filter): excludes done and cancelled — only A should appear
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

        // Explicit filter for 'done': includes done tickets
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

        // Explicit filter for 'cancelled': includes cancelled tickets
        let args = json!({"phase": "cancelled"});
        let result = tool.execute(&ws, args).await.expect("execute");
        assert!(
            result.contains('B'),
            "Explicit 'cancelled' filter should include cancelled ticket B"
        );

        // Filter for an active phase that ticket A is actually in
        let args = json!({"phase": "backlog"});
        let result = tool.execute(&ws, args).await.expect("execute");
        assert!(
            result.contains('A'),
            "Explicit 'backlog' filter should include ticket A"
        );
    }

    #[tokio::test]
    async fn test_list_tickets_invalid_phase() {
        crate::util::test::init_test_stores().await;

        let ws = crate::workspace::test_ws("/tmp");
        let tool = ListTicketsTool::new(&ws);
        let args = json!({"phase": "bogus_phase"});
        let result = tool.execute(&ws, args).await;
        assert!(result.is_err(), "Invalid phase should fail");
    }

    #[tokio::test]
    async fn test_get_ticket_tool() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/ws");
        let id = TicketBuilder::new(store, &ws)
            .title("GetTest")
            .desc("get me")
            .create()
            .await
            .expect("create");

        // Fetch the ticket to compare against detailed_display()
        let ticket = crate::util::test::expect_ticket(store, &id).await;
        let expected = ticket.detailed_display();

        let tool = GetTicketTool::new("manager", &ws);
        let args = json!({"ticket_id": id, "full": true});
        let result = tool.execute(&ws, args).await.expect("execute");
        assert_eq!(
            result, expected,
            "GetTicketTool output must match Ticket::detailed_display()"
        );
    }

    #[tokio::test]
    async fn test_add_comment_tool() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/ws");
        let id = TicketBuilder::new(store, &ws)
            .title("CommentTest")
            .desc("add a comment")
            .create()
            .await
            .expect("create");

        let tool = AddCommentTool::new(&ws);
        let args = json!({
            "ticket_id": id,
            "content": "This is a test comment",
        });
        let result = tool.execute(&ws, args).await.expect("execute");
        assert!(result.contains(&id), "Output should contain ticket id");

        // Verify comment was stored
        let ticket = crate::util::test::expect_ticket(store, &id).await;
        let comment = ticket.comments.last().expect("at least one comment");
        assert_eq!(comment.role, Role::Manager.as_str());
        assert_eq!(comment.content, "This is a test comment");
    }

    #[tokio::test]
    async fn test_invalid_phase() {
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
            "phase": "invalid_phase"
        });
        let result = tool.execute(&ws, args).await;
        assert!(result.is_err(), "Invalid phase should fail");
    }

    // ── Pipeline-occupied guard tests ────────────────────────────

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

        // Non-existent ticket silently passes (nothing to block)
        let result = guard_not_pipeline_occupied(store, "nonexistent_id").await;
        assert!(
            result.is_ok(),
            "Non-existent ticket should silently pass the guard"
        );
    }

    /// Create a ticket in a pipeline-occupied phase (InDevelopment).
    /// Used by wiring tests below to verify tools call `guard_not_pipeline_occupied`.
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

    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // shared global board — a concurrent boot reset would clobber the InDevelopment fixture
    async fn test_update_ticket_blocked_by_pipeline() {
        let id = create_occupied_ticket().await;
        let ws = test_ws("/ws");
        let result = UpdateTicketTool::new(&ws)
            .execute(&ws, json!({"ticket_id": id, "phase": "backlog"}))
            .await;
        assert!(
            result.is_err(),
            "Pipeline-occupied ticket should reject update"
        );
    }

    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // shared global board — a concurrent boot reset would clobber the InDevelopment fixture
    async fn test_create_ticket_supersede_blocked_by_pipeline() {
        let id = create_occupied_ticket().await;
        let ws = test_ws("/ws");
        let result = CreateTicketTool::new("test", &ws)
            .execute(
                &ws,
                json!({"title": "Replacement", "description": "trying to supersede", "supersede": id}),
            )
            .await;
        assert!(
            result.is_err(),
            "Supersede should be rejected for pipeline-occupied tickets"
        );
    }

    #[tokio::test]
    async fn test_supersede_inherits_priority() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/tmp/test_ws_supersede_priority");

        // Create a ticket with a non-default priority (0 = highest urgency).
        let old_id = TicketBuilder::new(store, &ws)
            .title("Urgent")
            .priority(0)
            .create()
            .await
            .expect("create old ticket");

        // Supersede without specifying priority — should inherit priority 0.
        let tool = CreateTicketTool::new("manager", &ws);
        let args = json!({
            "title": "Replacement",
            "description": "superseding the urgent ticket",
            "supersede": old_id,
        });
        let result = tool
            .execute(&ws, args)
            .await
            .expect("supersede should succeed");
        assert!(result.contains("Superseded"), "Expected supersede output");

        // Find the new ticket ID via the old ticket's superseded_by link.
        let old_ticket = expect_ticket(store, &old_id).await;
        let new_id = old_ticket
            .superseded_by
            .as_deref()
            .expect("old ticket should have superseded_by");

        // Verify the new ticket inherited priority 0 from the old ticket.
        let new_ticket = expect_ticket(store, new_id).await;
        assert_eq!(
            new_ticket.priority, 0,
            "Should inherit priority from superseded ticket"
        );
    }

    #[tokio::test]
    async fn test_supersede_explicit_priority_overrides_inheritance() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/tmp/test_ws_supersede_explicit_priority");

        // Create a ticket with a non-default priority.
        let old_id = TicketBuilder::new(store, &ws)
            .title("Old ticket")
            .priority(0)
            .create()
            .await
            .expect("create old ticket");

        // Supersede with explicit priority — should use the explicit value,
        // not inherit the old ticket's priority.
        let tool = CreateTicketTool::new("manager", &ws);
        let args = json!({
            "title": "Replacement",
            "description": "superseding with explicit priority",
            "supersede": old_id,
            "priority": 5,
        });
        let result = tool
            .execute(&ws, args)
            .await
            .expect("supersede should succeed");
        assert!(result.contains("Superseded"), "Expected supersede output");

        let old_ticket = expect_ticket(store, &old_id).await;
        let new_id = old_ticket
            .superseded_by
            .as_deref()
            .expect("old ticket should have superseded_by");

        let new_ticket = expect_ticket(store, new_id).await;
        assert_eq!(
            new_ticket.priority, 5,
            "Explicit priority should override inheritance from old ticket"
        );
    }

    #[tokio::test]
    async fn test_add_comment_allowed_on_pipeline_occupied() {
        let id = create_occupied_ticket().await;
        let ws = test_ws("/ws");
        let result = AddCommentTool::new(&ws)
            .execute(
                &ws,
                json!({"ticket_id": id, "content": "This should be allowed now"}),
            )
            .await;
        assert!(
            result.is_ok(),
            "Comments should be allowed on pipeline-occupied tickets"
        );
    }

    #[test]
    fn test_resolve_ticket_id() {
        // Bare numbers resolve against the bound workspace.
        assert_eq!(resolve_ticket_id("ws", "123").unwrap(), "ws-123");
        assert_eq!(resolve_ticket_id("ws", "  7 ").unwrap(), "ws-7");
        // Fully prefixed form for the bound workspace is accepted.
        assert_eq!(resolve_ticket_id("ws", "ws-123").unwrap(), "ws-123");
        assert_eq!(resolve_ticket_id("my_ws", "my_ws-42").unwrap(), "my_ws-42");
        // Foreign workspace prefixes are rejected.
        let err = format!("{}", resolve_ticket_id("ws", "other-123").unwrap_err());
        assert!(
            err.contains("different workspace"),
            "foreign prefix should be rejected: {err}"
        );
        // Malformed IDs are rejected.
        for bad in [
            "", "abc", "-5", "ws-abc", "123-456", "ws-", "ws-12-34", "___-123",
        ] {
            assert!(
                resolve_ticket_id("ws", bad).is_err(),
                "malformed ID '{bad}' should be rejected"
            );
        }
    }

    #[tokio::test]
    async fn test_get_ticket_workspace_binding() {
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

        // Bare numeric ID resolves against the bound workspace.
        let seq = id.strip_prefix("ws-").expect("prefixed id");
        let result = GetTicketTool::new("manager", &ws)
            .execute(&ws, json!({"ticket_id": seq, "full": true}))
            .await
            .expect("bare number should resolve");
        assert_eq!(result, expected);

        // A tool bound to another workspace rejects the foreign ticket.
        let foreign = test_ws("/other");
        let err = GetTicketTool::new("manager", &foreign)
            .execute(&foreign, json!({"ticket_id": id}))
            .await
            .unwrap_err();
        assert!(
            format!("{err}").contains("different workspace"),
            "foreign ticket should be rejected: {err}"
        );
    }

    #[tokio::test]
    async fn test_detailed_display_limited_truncates_description_and_non_tail_comments() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/ws");
        let long_desc = "x".repeat(600);
        let id = TicketBuilder::new(store, &ws)
            .title("Limited")
            .desc(&long_desc)
            .create()
            .await
            .expect("create");

        let mut ticket = crate::util::test::expect_ticket(store, &id).await;
        ticket.comments = vec![
            comment("engineer", &"a".repeat(300), "2026-01-01T00:00:00Z"),
            comment("qa", &"b".repeat(250), "2026-01-02T00:00:00Z"),
            comment("reviewer", &"c".repeat(150), "2026-01-03T00:00:00Z"),
            comment("manager", &"d".repeat(100), "2026-01-04T00:00:00Z"),
            comment("assistant", &"e".repeat(80), "2026-01-05T00:00:00Z"),
        ];

        let out = ticket.detailed_display_limited(500, 200, 3);

        // (a) description capped at 500 chars + ellipsis
        assert!(out.contains(&format!("Description: {}…", "x".repeat(500))));
        assert!(
            !out.contains(&long_desc),
            "full description must not appear"
        );
        // (d) header metadata never truncated
        assert!(out.contains(&format!("Ticket: {}", ticket.id)));
        assert!(out.contains("Title: Limited"));
        assert!(out.contains("Reporter: test"));
        assert!(out.contains("Priority: P1"));
        // (b) comments older than the last three capped at 200 chars + ellipsis
        assert!(out.contains(&format!(
            "[engineer] (2026-01-01T00:00:00): {}…",
            "a".repeat(200)
        )));
        assert!(out.contains(&format!("[qa] (2026-01-02T00:00:00): {}…", "b".repeat(200))));
        // (c) last three comments shown in full
        assert!(out.contains(&format!(
            "[reviewer] (2026-01-03T00:00:00): {}",
            "c".repeat(150)
        )));
        assert!(out.contains(&format!(
            "[manager] (2026-01-04T00:00:00): {}",
            "d".repeat(100)
        )));
        assert!(out.contains(&format!(
            "[assistant] (2026-01-05T00:00:00): {}",
            "e".repeat(80)
        )));
    }

    #[tokio::test]
    async fn test_detailed_display_limited_three_or_fewer_comments_all_full() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/ws");
        let id = TicketBuilder::new(store, &ws)
            .title("FewComments")
            .desc("short desc")
            .create()
            .await
            .expect("create");

        let mut ticket = crate::util::test::expect_ticket(store, &id).await;
        ticket.comments = vec![
            comment("engineer", &"a".repeat(300), "2026-01-01T00:00:00Z"),
            comment("qa", &"b".repeat(250), "2026-01-02T00:00:00Z"),
            comment("reviewer", &"c".repeat(150), "2026-01-03T00:00:00Z"),
        ];

        let out = ticket.detailed_display_limited(500, 200, 3);
        assert!(out.contains(&"a".repeat(300)));
        assert!(out.contains(&"b".repeat(250)));
        assert!(out.contains(&"c".repeat(150)));
        assert!(
            !out.contains('…'),
            "with ≤3 comments none should be truncated"
        );
    }

    #[tokio::test]
    async fn test_get_ticket_default_truncated_for_same_agent() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/ws");
        let long_desc = "y".repeat(600);
        let id = TicketBuilder::new(store, &ws)
            .title("SameAgent")
            .reporter("manager")
            .desc(&long_desc)
            .create()
            .await
            .expect("create");

        let tool = GetTicketTool::new("manager", &ws);
        let result = tool
            .execute(&ws, json!({"ticket_id": id}))
            .await
            .expect("execute");
        assert!(
            result.contains(&format!("Description: {}…", "y".repeat(500))),
            "same-agent default view must truncate the description"
        );
        assert!(!result.contains(&long_desc));
    }

    #[tokio::test]
    async fn test_get_ticket_full_output_for_different_creator() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/ws");
        let long_desc = "z".repeat(600);
        let id = TicketBuilder::new(store, &ws)
            .title("OtherAgent")
            .reporter("maintainer")
            .desc(&long_desc)
            .create()
            .await
            .expect("create");

        let ticket = crate::util::test::expect_ticket(store, &id).await;
        let expected = ticket.detailed_display();

        let tool = GetTicketTool::new("manager", &ws);
        let result = tool
            .execute(&ws, json!({"ticket_id": id}))
            .await
            .expect("execute");
        assert_eq!(
            result, expected,
            "a different creator must produce the full un-truncated ticket"
        );
        assert!(result.contains(&long_desc));
    }

    #[tokio::test]
    async fn test_get_ticket_full_flag_returns_complete() {
        crate::util::test::init_test_stores().await;

        let store = board_store();
        let ws = test_ws("/ws");
        let long_desc = "w".repeat(600);
        let id = TicketBuilder::new(store, &ws)
            .title("FullFlag")
            .reporter("manager")
            .desc(&long_desc)
            .create()
            .await
            .expect("create");

        let ticket = crate::util::test::expect_ticket(store, &id).await;
        let expected = ticket.detailed_display();

        let tool = GetTicketTool::new("manager", &ws);
        let result = tool
            .execute(&ws, json!({"ticket_id": id, "full": true}))
            .await
            .expect("execute");
        assert_eq!(
            result, expected,
            "full=true must return the complete ticket"
        );
        assert!(result.contains(&long_desc));
    }
}
