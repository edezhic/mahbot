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

pub(crate) const GET_TICKET_DESC_MAX: usize = 500;
pub(crate) const GET_TICKET_COMMENT_MAX: usize = 200;
pub(crate) const GET_TICKET_LAST_N_FULL: usize = 3;

// ── Ticket ID resolution ─────────────────────────────────────────

/// Resolve a raw ticket ID argument against the tool's bound workspace.
///
/// Accepts either a bare number (resolved to `{ws}-{number}`) or the fully
/// prefixed form for the bound workspace. Prefixed IDs from any other
/// workspace are rejected with a clear error.
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

/// Like [`resolve_ticket_id`], but a well-formed prefixed ID from any
/// workspace is taken as written. Used by the read-only [`GetTicketTool`]:
/// the archive search spans the installation, and a hit is useless if the
/// ticket cannot be opened wherever it was filed.
fn resolve_any_ticket_id(ws_name: &str, raw: &str) -> Result<String> {
    let (prefix, id) = parse_ticket_id(raw)?;
    match prefix {
        None => Ok(format!("{ws_name}-{id}")),
        Some(_) => Ok(id.to_string()),
    }
}

/// Split `raw` into `(workspace prefix, id)`: the prefix is `None` for a bare
/// number — which the caller resolves against its bound workspace — and
/// `Some(prefix)` for a well-formed `{prefix}-{seq}` argument, in which case
/// `id` is `raw` trimmed, i.e. the full prefixed ticket ID. The prefix shape
/// mirrors `workspace::validate_name` ([a-zA-Z_]+ starting with a letter).
///
/// Shared with the admin's `workspace` tool, which accepts only the full
/// `<workspace>-<number>` spelling and refuses every other one.
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
        priority: i64,
    ) -> TicketParams {
        TicketParams {
            title: title.to_string(),
            description: description.to_string(),
            workspace_name: self.ws_name.clone(),
            phase: TicketPhase::Backlog,
            prerequisites: prerequisites.to_vec(),
            reporter: self.reporter.clone(),
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
                "description": "Optional ticket ID to supersede — atomically cancels the old ticket, creates this one as its replacement, and rewires any dependents to point to the new ID. A finished (done) ticket cannot be superseded."
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

        let prerequisites = super::get_str_array(&args, "prerequisites")?
            .iter()
            .map(|s| resolve_ticket_id(&self.ws_name, s))
            .collect::<Result<Vec<_>>>()?;

        let supersede_id = super::get_opt_str_strict(&args, "supersede")?
            .map(|s| resolve_ticket_id(&self.ws_name, s))
            .transpose()?;

        // Priority: Manager can set explicitly; otherwise inherit from the
        // superseded ticket (if superseding) or use the role default.
        let explicit_priority = if self.reporter == "manager" {
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
            // No explicit priority: maintainer gets 3, while the manager
            // (and every other reporter) intentionally defaults to 1.
            (None, None) => {
                if self.reporter == "maintainer" {
                    3
                } else {
                    1
                }
            }
        };

        let params = self.build_params(title, description, &prerequisites, priority);
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

        // Guard: refuse to update a ticket that is in a pipeline-occupied phase.
        guard_not_pipeline_occupied(store, &ticket_id).await?;

        store
            .transition_to(&ticket_id, None, parsed_phase, Role::Manager.as_str())
            .await?;

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

/// Read one ticket by ID. Unlike the write tools it is not confined to its
/// bound workspace: a fully prefixed ID from any workspace is accepted, while a
/// bare number still resolves against the bound workspace.
///
/// The archive search spans the installation, so `get_ticket` must be able to
/// open its hits wherever they were filed.
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
        // Ticket content (descriptions, comments) can exceed the ~5 KB
        // tool-output trigger — the LLM needs it in full.
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

    /// What a create-ticket call must produce.
    enum CreateArgsExpect {
        Created {
            starts_with: &'static str,
            contains: &'static [&'static str],
            not_contains: &'static [&'static str],
        },
        /// Substrings the usage error must carry.
        Usage(&'static [&'static str]),
    }

    /// One create-ticket argument scenario.
    struct CreateArgsCase {
        name: &'static str,
        role: &'static str,
        args: serde_json::Value,
        expected: CreateArgsExpect,
    }

    #[tokio::test]
    async fn create_ticket_argument_cases() {
        crate::util::test::init_test_stores().await;

        let ws = test_ws("/tmp/test_ws");
        for case in [
            CreateArgsCase {
                name: "a non-string prerequisite is a usage error",
                role: "test",
                args: json!({
                    "title": "Test",
                    "description": "desc",
                    "prerequisites": [123],
                }),
                expected: CreateArgsExpect::Usage(&[
                    "usage: ",
                    r#"argument "prerequisites" must be an array of strings"#,
                ]),
            },
            // The created ticket answers with its own name, and an argument given as
            // null is absent rather than spelled out.
            CreateArgsCase {
                name: "null optional arguments are absent",
                role: "manager",
                args: json!({
                    "title": "Null optionals",
                    "description": "desc",
                    "prerequisites": null,
                    "supersede": null,
                    "priority": null,
                }),
                expected: CreateArgsExpect::Created {
                    starts_with: "Created ticket ",
                    contains: &["Null optionals"],
                    not_contains: &["with prerequisites"],
                },
            },
            CreateArgsCase {
                name: "a non-integer priority is a usage error",
                role: "manager",
                args: json!({
                    "title": "Bad priority",
                    "description": "desc",
                    "priority": "high",
                }),
                expected: CreateArgsExpect::Usage(&[
                    "usage: ",
                    r#"argument "priority" must be an integer"#,
                ]),
            },
        ] {
            let tool = CreateTicketTool::new(case.role, &ws);
            match (case.expected, tool.execute(&ws, case.args).await) {
                (CreateArgsExpect::Usage(needles), Err(err)) => {
                    let msg = err.to_string();
                    for needle in needles {
                        assert!(
                            msg.contains(needle),
                            "[{}] expected {needle:?}: {msg}",
                            case.name
                        );
                    }
                }
                (
                    CreateArgsExpect::Created {
                        starts_with,
                        contains,
                        not_contains,
                    },
                    Ok(result),
                ) => {
                    assert!(
                        result.starts_with(starts_with),
                        "[{}] got: {result}",
                        case.name
                    );
                    for needle in contains {
                        assert!(
                            result.contains(needle),
                            "[{}] expected {needle:?}: {result}",
                            case.name
                        );
                    }
                    for needle in not_contains {
                        assert!(
                            !result.contains(needle),
                            "[{}] expected no {needle:?}: {result}",
                            case.name
                        );
                    }
                }
                (_, other) => panic!("[{}] unexpected outcome: {other:?}", case.name),
            }
        }
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
        // Transition C to 'done' and B to 'cancelled'
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

    /// One add-comment call: the phase the ticket is in, and the comment the
    /// ticket must carry afterwards.
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
            // A pipeline-occupied ticket may not be mutated, but commenting on it
            // is not a mutation.
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

    /// One supersede call: the explicit priority (if any) and the priority the
    /// replacement must end up with.
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
            // A ticket with a non-default priority (0 = highest urgency).
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

            // The new ticket hangs off the old ticket's superseded_by link.
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
        }
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

        // `resolve_any_ticket_id` binds a bare number to the bound workspace
        // but takes any well-formed prefix verbatim; malformed IDs are the
        // shared parser's business, already pinned above.
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

        // Bare numeric ID resolves against the bound workspace.
        let seq = id.strip_prefix("ws-").expect("prefixed id");
        let result = GetTicketTool::new("manager", &ws)
            .execute(&ws, json!({"ticket_id": seq, "full": true}))
            .await
            .expect("bare number should resolve");
        assert_eq!(result, expected);

        // A fully prefixed ID is accepted even from another workspace — the
        // archive search spans the installation, so its hits must open here.
        let foreign = test_ws("/other");
        let result = GetTicketTool::new("manager", &foreign)
            .execute(&foreign, json!({"ticket_id": id, "full": true}))
            .await
            .expect("a prefixed ID from any workspace is accepted");
        assert_eq!(result, expected);
    }
}
