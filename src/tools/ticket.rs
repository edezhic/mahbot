//! Ticket management tools for the Manager role.
//!
//! These tools allow the Manager to create tickets on the board,
//! update their status, list them, and inspect individual tickets.

use crate::Role;
use crate::Workspace;
use crate::board::{TicketParams, TicketPhase};
use crate::tools::Tool;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::fmt::Write;

// ── CreateTicketTool ─────────────────────────────────────────────

/// Tool for creating tickets. The `reporter` field is set at construction
/// time based on which role is using the tool — invisible to the agent,
/// no parameter to pass.
pub struct CreateTicketTool {
    reporter: String,
}

impl CreateTicketTool {
    pub fn new(reporter: impl Into<String>) -> Self {
        Self {
            reporter: reporter.into(),
        }
    }

    /// Build a [`TicketParams`] from the shared fields used by both creation branches.
    fn build_params(
        &self,
        ws: &Workspace,
        title: &str,
        description: &str,
        prerequisites: &[String],
        embedding_bytes: Option<&Vec<u8>>,
    ) -> TicketParams {
        TicketParams {
            title: title.to_string(),
            description: description.to_string(),
            workspace_name: ws.name.clone(),
            phase: TicketPhase::Backlog,
            prerequisites: prerequisites.to_vec(),
            reporter: self.reporter.clone(),
            embedding: embedding_bytes.cloned(),
        }
    }
}

#[async_trait]
impl Tool for CreateTicketTool {
    fn name(&self) -> &'static str {
        "create_ticket"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "title": {
                    "type": "string",
                    "description": "Short title/summary of the task"
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the task"
                },
                "prerequisites": {
                    "type": "array",
                    "description": "Optional list of ticket IDs that must be completed (done, archived, or cancelled) before this ticket can be claimed",
                    "items": {
                        "type": "string"
                    }
                },
                "supersede": {
                    "type": "string",
                    "description": "Optional ticket ID to supersede — atomically cancels the old ticket, creates this one as its replacement, and rewires any dependents to point to the new ID"
                },
            }),
            &["title", "description"],
        )
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let title = super::get_str(&args, "title")?;
        let description = super::get_str(&args, "description")?;

        let prerequisites: Vec<String> = match args.get("prerequisites") {
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            Some(_) => {
                anyhow::bail!("prerequisites must be an array of ticket ID strings");
            }
            None => Vec::new(),
        };

        let supersede_id: Option<String> = match args.get("supersede") {
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(_) => anyhow::bail!("supersede must be a ticket ID string"),
            None => None,
        };

        let store = crate::board::store();
        let embedding_bytes: Option<Vec<u8>> =
            crate::embedder::embed(title, false).map(|v| crate::vector::vec_to_bytes(&v));

        let prereq_note = if prerequisites.is_empty() {
            String::new()
        } else {
            format!(" with prerequisites: {}", prerequisites.join(", "))
        };

        let params = self.build_params(
            ws,
            title,
            description,
            &prerequisites,
            embedding_bytes.as_ref(),
        );
        if let Some(supersede_id) = supersede_id {
            guard_not_pipeline_blocking(store, &supersede_id).await?;

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

pub struct UpdateTicketTool;

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
                "status": {
                    "type": "string",
                    "description": "New status for the ticket. Valid manual transitions: backlog (return to queue), ready_for_development (send to engineer), cancelled (abandon), failed (mark unsuccessful), done (mark complete), qa_passed (advance failed ticket past failed — only valid from 'failed' phase for Manager triage of minor issues). Do NOT manually set other pipeline-managed phases (analysis, planning, in_development, in_diagnostics, diagnostics_done, in_review, reviewed, in_qa) — the board poller handles these automatically and manual transitions will interfere with running agents."
                }
            }),
            &["ticket_id", "status"],
        )
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let ticket_id = super::get_str(&args, "ticket_id")?;
        let new_status = super::get_str(&args, "status")?;

        let parsed_status = new_status.parse::<TicketPhase>()?;

        let store = crate::board::store();

        // Guard: refuse to update a ticket that is in a pipeline-blocking phase.
        guard_not_pipeline_blocking(store, ticket_id).await?;

        store
            .transition_to(ticket_id, None, parsed_status, None)
            .await?;

        Ok(format!(
            "Ticket {ticket_id} status updated to '{new_status}'"
        ))
    }
}

// ── ListTicketsTool ─────────────────────────────────────────────

pub struct ListTicketsTool;

#[async_trait]
impl Tool for ListTicketsTool {
    fn name(&self) -> &'static str {
        "list_tickets"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "status": {
                    "type": "string",
                    "description": "Optional status filter (e.g. 'ready_for_development', 'in_development', 'done', 'cancelled'). When omitted, 'done' and 'cancelled' tickets are excluded — use an explicit status filter to include them. Use 'search_archived_tickets' to find archived tickets."
                }
            }),
            &[],
        )
    }

    fn side_effects(&self, _args: &serde_json::Value) -> bool {
        false // read-only board query
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let raw_status = super::get_opt_str(&args, "status");

        // "archived" is no longer a status — redirect to search_archived_tickets
        if let Some(s) = raw_status
            && s.eq_ignore_ascii_case("archived")
        {
            anyhow::bail!(
                "Archived tickets are not accessible via 'list_tickets'. \
                     Use 'search_archived_tickets' to search the archive."
            );
        }

        let status_filter = match raw_status {
            Some(s) => {
                let parsed = s.parse::<TicketPhase>()?;
                Some(parsed)
            }
            None => None,
        };

        let store = crate::board::store();

        let mut tickets = store
            .list_all_tickets(Some(&ws.name), status_filter)
            .await?;

        // Default: exclude unblocking phases. When an explicit status filter
        // is provided, respect it as-is.
        if status_filter.is_none() {
            tickets.retain(|t| !t.status.is_unblocking());
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

pub struct GetTicketTool;

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
                }
            }),
            &["ticket_id"],
        )
    }

    fn side_effects(&self, _args: &serde_json::Value) -> bool {
        false // read-only board query
    }

    /// Override format_output to preserve full ticket content.
    /// The default truncation at 5K chars would lose long comments/descriptions
    /// that LLMs need to read in their entirety.
    fn format_output(&self, output: &str) -> String {
        output.to_string()
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let ticket_id = super::get_str(&args, "ticket_id")?;

        let store = crate::board::store();

        match store.get_ticket(ticket_id).await? {
            Some(ticket) => Ok(ticket.detailed_display()),
            None => anyhow::bail!("Ticket {ticket_id} not found"),
        }
    }
}

// ── AddCommentTool ──────────────────────────────────────────────

pub struct AddCommentTool;

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
        let ticket_id = super::get_str(&args, "ticket_id")?;
        let content = super::get_str(&args, "content")?;

        let store = crate::board::store();

        // Guard: refuse to comment on a ticket that is in a pipeline-blocking phase.
        guard_not_pipeline_blocking(store, ticket_id).await?;

        store
            .add_comment(ticket_id, Role::Manager.as_str(), content)
            .await?;

        Ok(format!("Comment added to ticket {ticket_id}"))
    }
}

/// Guard: refuse to proceed if the ticket is in a pipeline-blocking phase.
/// All three user-facing ticket tools use this to prevent modifications
/// to in-flight tickets during automated pipeline processing.
async fn guard_not_pipeline_blocking(
    store: &crate::board::BoardStore,
    ticket_id: &str,
) -> Result<()> {
    if let Some(current_phase) = store.get_ticket_phase(ticket_id).await?
        && current_phase.is_pipeline_blocking()
    {
        anyhow::bail!(
            "Ticket {ticket_id} is currently in phase '{status}' — \
                 in-flight tickets cannot be modified through automated tools. \
                 Use the GUI to cancel or manage this ticket manually.",
            status = current_phase.as_ref(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::TicketBuilder;
    use crate::util::test::make_ticket;
    use crate::workspace::test_ws;
    use serde_json::json;

    #[tokio::test]
    async fn test_create_ticket_tool() {
        crate::util::test::init_test_stores().await;

        let tool = CreateTicketTool::new("test");
        let ws = test_ws("/tmp/test_ws");
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
    async fn test_update_ticket_tool() {
        crate::util::test::init_test_stores().await;

        let store = crate::board::store();
        let id = make_ticket(
            store,
            test_ws("/ws"),
            "Test",
            crate::board::TicketPhase::Backlog,
        )
        .await;

        let tool = UpdateTicketTool;
        let ws = test_ws("/tmp");
        let args = json!({
            "ticket_id": id,
            "status": "in_qa"
        });
        tool.execute(&ws, args).await.expect("execute");
        let ticket = store
            .get_ticket(&id)
            .await
            .expect("get_ticket")
            .expect("ticket exists");
        assert_eq!(ticket.status, TicketPhase::InQa);
    }

    #[tokio::test]
    async fn test_list_tickets_tool() {
        crate::util::test::init_test_stores().await;

        let store = crate::board::store();
        let ws = test_ws("/tmp_ws");

        let _id_a = make_ticket(store, ws.clone(), "A", crate::board::TicketPhase::Backlog).await;
        let id_b = make_ticket(store, ws.clone(), "B", crate::board::TicketPhase::Backlog).await;
        let id_c = make_ticket(store, ws.clone(), "C", crate::board::TicketPhase::Backlog).await;
        // Transition C to 'done' and B to 'cancelled'
        store
            .transition_to(&id_c, Some(TicketPhase::Backlog), TicketPhase::Done, None)
            .await
            .expect("transition to done");
        store
            .transition_to(
                &id_b,
                Some(TicketPhase::Backlog),
                TicketPhase::Cancelled,
                None,
            )
            .await
            .expect("transition to cancelled");

        let tool = ListTicketsTool;

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
        let args = json!({"status": "done"});
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
        let args = json!({"status": "cancelled"});
        let result = tool.execute(&ws, args).await.expect("execute");
        assert!(
            result.contains('B'),
            "Explicit 'cancelled' filter should include cancelled ticket B"
        );

        // Filter for an active phase that ticket A is actually in
        let args = json!({"status": "backlog"});
        let result = tool.execute(&ws, args).await.expect("execute");
        assert!(
            result.contains('A'),
            "Explicit 'backlog' filter should include ticket A"
        );
    }

    #[tokio::test]
    async fn test_list_tickets_invalid_status() {
        crate::util::test::init_test_stores().await;

        let ws = crate::workspace::test_ws("/tmp");
        let tool = ListTicketsTool;
        let args = json!({"status": "bogus_status"});
        let result = tool.execute(&ws, args).await;
        assert!(result.is_err(), "Invalid status should fail");
    }

    #[tokio::test]
    async fn test_get_ticket_tool() {
        crate::util::test::init_test_stores().await;

        let store = crate::board::store();
        let id = TicketBuilder::new(store, test_ws("/ws"))
            .title("GetTest")
            .desc("get me")
            .create()
            .await
            .expect("create");

        // Fetch the ticket to compare against detailed_display()
        let ticket = crate::util::test::expect_ticket(store, &id).await;
        let expected = ticket.detailed_display();

        let tool = GetTicketTool;
        let ws = test_ws("/tmp");
        let args = json!({"ticket_id": id});
        let result = tool.execute(&ws, args).await.expect("execute");
        assert_eq!(
            result, expected,
            "GetTicketTool output must match Ticket::detailed_display()"
        );
    }

    #[tokio::test]
    async fn test_add_comment_tool() {
        crate::util::test::init_test_stores().await;

        let store = crate::board::store();
        let id = TicketBuilder::new(store, test_ws("/ws"))
            .title("CommentTest")
            .desc("add a comment")
            .create()
            .await
            .expect("create");

        let tool = AddCommentTool;
        let ws = test_ws("/tmp");
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
    async fn test_invalid_status() {
        crate::util::test::init_test_stores().await;

        let store = crate::board::store();
        let id = make_ticket(
            store,
            test_ws("/ws"),
            "Test",
            crate::board::TicketPhase::Backlog,
        )
        .await;

        let tool = UpdateTicketTool;
        let ws = test_ws("/tmp");
        let args = json!({
            "ticket_id": id,
            "status": "invalid_status"
        });
        let result = tool.execute(&ws, args).await;
        assert!(result.is_err(), "Invalid status should fail");
    }

    // ── Pipeline-blocking guard tests ────────────────────────────

    #[tokio::test]
    async fn test_guard_not_pipeline_blocking_cases() {
        crate::util::test::init_test_stores().await;

        let store = crate::board::store();
        let ws = test_ws("/ws");

        let blocking_id = TicketBuilder::new(store, ws.clone())
            .title("BlockingGuard")
            .desc("in development")
            .phase(TicketPhase::InDevelopment)
            .create()
            .await
            .expect("create");

        let nonblocking_id = TicketBuilder::new(store, ws)
            .title("NonBlockingGuard")
            .desc("backlog")
            .phase(TicketPhase::Backlog)
            .create()
            .await
            .expect("create");

        let result = guard_not_pipeline_blocking(store, &blocking_id).await;
        assert!(result.is_err(), "Blocking-phase ticket should be rejected");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("in-flight"),
            "Error should mention in-flight restriction"
        );

        let result = guard_not_pipeline_blocking(store, &nonblocking_id).await;
        assert!(result.is_ok(), "Non-blocking ticket should pass the guard");

        // Non-existent ticket silently passes (nothing to block)
        let result = guard_not_pipeline_blocking(store, "nonexistent_id").await;
        assert!(
            result.is_ok(),
            "Non-existent ticket should silently pass the guard"
        );
    }

    /// Create a ticket in a pipeline-blocking phase (InDevelopment).
    /// Used by wiring tests below to verify tools call `guard_not_pipeline_blocking`.
    async fn create_blocking_ticket() -> String {
        crate::util::test::init_test_stores().await;
        let store = crate::board::store();
        TicketBuilder::new(store, test_ws("/ws"))
            .title("BlockMe")
            .desc("in flight")
            .phase(TicketPhase::InDevelopment)
            .create()
            .await
            .expect("create")
    }

    #[tokio::test]
    async fn test_update_ticket_blocked_by_pipeline() {
        let id = create_blocking_ticket().await;
        let result = UpdateTicketTool
            .execute(
                &test_ws("/tmp"),
                json!({"ticket_id": id, "status": "backlog"}),
            )
            .await;
        assert!(
            result.is_err(),
            "Pipeline-blocking ticket should reject update"
        );
    }

    #[tokio::test]
    async fn test_create_ticket_supersede_blocked_by_pipeline() {
        let id = create_blocking_ticket().await;
        let result = CreateTicketTool::new("test")
            .execute(
                &test_ws("/ws"),
                json!({"title": "Replacement", "description": "trying to supersede", "supersede": id}),
            )
            .await;
        assert!(
            result.is_err(),
            "Supersede should be rejected for pipeline-blocking tickets"
        );
    }

    #[tokio::test]
    async fn test_add_comment_blocked_by_pipeline() {
        let id = create_blocking_ticket().await;
        let result = AddCommentTool
            .execute(
                &test_ws("/tmp"),
                json!({"ticket_id": id, "content": "This should be blocked"}),
            )
            .await;
        assert!(
            result.is_err(),
            "Pipeline-blocking ticket should reject comments"
        );
    }

    #[tokio::test]
    async fn test_update_ticket_allowed_when_not_blocked() {
        crate::util::test::init_test_stores().await;

        let store = crate::board::store();
        let id = TicketBuilder::new(store, test_ws("/ws"))
            .title("NotBlocked")
            .desc("backlog ticket")
            .phase(TicketPhase::Backlog)
            .create()
            .await
            .expect("create");

        let tool = UpdateTicketTool;
        let ws = test_ws("/tmp");
        let args = json!({
            "ticket_id": id,
            "status": "done"
        });
        let result = tool.execute(&ws, args).await;
        assert!(
            result.is_ok(),
            "Non-pipeline ticket should update freely: {:?}",
            result.err()
        );
    }
}
