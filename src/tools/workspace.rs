//! The admin's own view into the managed workspaces: list them, add one
//! (which activates it), freeze/unfreeze their pipelines, read a board and a
//! single ticket, and leave a clarification note on a ticket.
//!
//! It manages no board. Adding a managed workspace is the only write here that
//! can change the desktop's active workspace; the desktop's own project list is
//! the other surface that selects one.
use anyhow::anyhow;
use async_trait::async_trait;
use serde_json::json;
use std::fmt::Write as _;

use crate::pipeline::board::{TicketPhase, store as board_store};
use crate::tools::ToolAnswer;
use crate::tools::ticket::{GET_TICKET_COMMENT_MAX, GET_TICKET_DESC_MAX, GET_TICKET_LAST_N_FULL};
use crate::workspace::WorkspaceStore;
use crate::{Role, Tool, Workspace, WorkspaceStatus};

/// The tool's own byte budget, needed because the framework's ~5 KB tool-output
/// truncation is off for it (see [`WorkspaceTool::preserve_full_output`]): the
/// `board` listing and the shortened `ticket` view bound themselves, and a byte
/// cap is required alongside `BOARD_LINE_LIMIT` because a title, a description
/// and a comment are all unbounded. `ticket` with `full: true` is deliberately
/// exempt.
const OUTPUT_BYTE_LIMIT: usize = 24_000;
const BOARD_LINE_LIMIT: usize = 100;

/// What an action that covers the whole managed set answers when there is
/// none — one wording, so `list` and a fleet-wide pause say the same thing.
const NO_WORKSPACE: &str = "No managed workspaces exist.";

/// The `workspace` tool: the admin's project view — list / add / board / ticket
/// / comment / pause / resume.
pub(crate) struct WorkspaceTool;

#[async_trait]
impl Tool for WorkspaceTool {
    fn name(&self) -> &'static str {
        "workspace"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "action": {
                    "type": "string",
                    "enum": ["list", "add", "pause", "resume", "board", "ticket", "comment"],
                    "description": "Which action to perform; the descriptions of `workspace`, `name`, `path`, `phase`, `ticket_id`, `full` and `content` say which action each field belongs to."
                },
                "workspace": {
                    "type": "string",
                    "description": "(list / board / pause / resume) The managed workspace to act on, spelled exactly as the `<managed-workspaces>` block or `list` spells it — matching is exact, never approximate. Omit it to cover every managed workspace, which for `pause` / `resume` is a fleet-wide freeze, so name the workspace whenever one was named to you."
                },
                "name": {
                    "type": "string",
                    "description": "(add) The new workspace's short unique name, used in ticket ids and the GUI."
                },
                "path": {
                    "type": "string",
                    "description": "(add) Absolute path to the software project directory to add."
                },
                "phase": {
                    "type": "string",
                    "description": "(board) Narrow the listing to one ticket phase (e.g. 'queued', 'in_development', 'done', 'cancelled'). Omit it to list every non-archived ticket."
                },
                "ticket_id": {
                    "type": "string",
                    "description": "(ticket / comment) The ticket's full id — `<workspace>-<number>`, which already carries its workspace. A bare number is refused."
                },
                "full": {
                    "type": "boolean",
                    "description": "(ticket) Return the ticket whole and unclipped. Default false."
                },
                "content": {
                    "type": "string",
                    "description": "(comment) The clarification note to leave on the ticket."
                }
            }),
            &["action"],
        )
    }

    fn preserve_full_output(&self) -> bool {
        // A `ticket` answer can carry hundreds of KB of comments, so a fixed
        // ~5 KB tool-output truncation is wrong for the whole tool; the read
        // paths bound themselves instead (`OUTPUT_BYTE_LIMIT`).
        true
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        Ok(self.answer(args).await?.text)
    }

    async fn execute_answer(
        &self,
        _ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<ToolAnswer> {
        self.answer(args).await
    }
}

impl WorkspaceTool {
    /// The one action dispatch both faces of the tool go through: `execute`
    /// renders it to its text (so an agent's view is unchanged) and
    /// [`Tool::execute_answer`] hands it over with the partiality flag a
    /// command-line caller reads. There is no second code path to drift from
    /// this one.
    async fn answer(&self, args: serde_json::Value) -> anyhow::Result<ToolAnswer> {
        match super::get_str(&args, "action")? {
            "list" => Ok(ToolAnswer::done(
                list_workspaces(super::get_opt_str_strict(&args, "workspace")?).await?,
            )),
            "add" => add_workspace(&args).await,
            "pause" => pause_or_resume(write_address(&args, "pause")?, true).await,
            "resume" => pause_or_resume(write_address(&args, "resume")?, false).await,
            "board" => Ok(ToolAnswer::done(
                list_board(
                    super::get_opt_str_strict(&args, "workspace")?,
                    super::get_opt_str_strict(&args, "phase")?,
                )
                .await?,
            )),
            "ticket" => Ok(ToolAnswer::done(
                show_ticket(
                    super::get_str(&args, "ticket_id")?,
                    super::get_bool(&args, "full", false)?,
                )
                .await?,
            )),
            "comment" => Ok(ToolAnswer::done(
                add_note(
                    super::get_str(&args, "ticket_id")?,
                    super::get_str(&args, "content")?,
                )
                .await?,
            )),
            other => Err(anyhow!(
                "unknown action '{other}' — expected one of: list, add, pause, resume, board, ticket, comment"
            )),
        }
    }
}

// ── Shared resolution ────────────────────────────────────────────

/// The workspace a `pause` / `resume` was pointed at: the `workspace` argument,
/// or `None` for every managed workspace when none is named.
///
/// The one action pair that can act on a single workspace or on all of them must
/// never widen because the address was spelled some other way — a folder path, a
/// ticket id, `add`'s `name`, or a key no schema declares — so with no `workspace`
/// argument such a call may name nothing else at all. An extra field is a usage
/// error phrased around that rule, never around the key that carried it, so the
/// message cannot name a key the caller did not write. The read actions (`list`,
/// `board`) take the same address but ignore a stray field, as the other tools
/// do; only a write that would cover the fleet has to say so.
///
/// A JSON `null` counts as absent, here as everywhere in the product (providers
/// fill omitted arguments with nulls), so a call whose address arrived nulled is
/// the unnamed form — which the tool description says plainly.
fn write_address<'a>(args: &'a serde_json::Value, action: &str) -> anyhow::Result<Option<&'a str>> {
    if let Some(address) = super::get_opt_str_strict(args, "workspace")? {
        return Ok(Some(address));
    }
    if args.as_object().is_some_and(|obj| {
        obj.iter()
            .any(|(key, value)| key.as_str() != "action" && !value.is_null())
    }) {
        return Err(anyhow!(
            "usage: with no \"workspace\" this `{action}` covers every managed workspace, so \
             the call may name nothing else — hint: pass \"workspace\" to act on one workspace, or \
             drop the extra field to really cover every managed workspace"
        ));
    }
    Ok(None)
}

/// Resolve `name` to a managed workspace — never a personal space, including
/// a legacy `personal:*` row. Matching is exact, and a miss names the live set
/// because the admin names workspaces from a session-start context block that
/// may predate an add or a delete.
async fn managed_workspace(name: &str) -> anyhow::Result<Workspace> {
    let name = name.trim();
    let workspaces = crate::users::managed_workspaces().await?;
    if let Some(ws) = workspaces.iter().find(|ws| ws.name == name) {
        return Ok(ws.clone());
    }
    let managed = if workspaces.is_empty() {
        "none".to_string()
    } else {
        workspaces
            .iter()
            .map(|ws| ws.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    Err(if crate::users::is_personal_workspace(name) {
        anyhow!(
            "'{name}' is a personal space, not a managed workspace — personal spaces have \
             no pipeline and no board; managed workspaces: {managed}"
        )
    } else {
        anyhow!(
            "No managed workspace exists under the name '{name}' — managed workspaces: {managed}"
        )
    })
}

/// Resolve a full `<workspace>-<number>` ticket id, refusing every other
/// spelling with the expected form, and an id whose workspace prefix is not
/// a managed workspace.
async fn managed_ticket_id(raw: &str) -> anyhow::Result<String> {
    // Only the full spelling addresses a ticket here: the project manager's
    // own ticket tools resolve a bare number against the workspace they are
    // bound to, and this tool is bound to none.
    let Ok((Some(prefix), full)) = crate::tools::ticket::parse_ticket_id(raw) else {
        return Err(anyhow!(
            "Ticket id '{}' is not a full '<workspace>-<number>' id — `ticket` and `comment` \
             are addressed by a ticket's full id, which already carries its workspace.",
            raw.trim()
        ));
    };
    // An id naming a workspace that is not managed is refused as an unknown
    // workspace, with the managed set named.
    managed_workspace(prefix).await?;
    Ok(full.to_string())
}

// ── list ─────────────────────────────────────────────────────────

/// The managed workspaces, one line each, with the admin's active one marked.
///
/// The active workspace resolves exactly as the desktop does and counts only
/// when it is one of the managed names: a personal value, or a name no longer
/// a managed workspace, means nothing is active — while the resolver's `None`
/// is a failed read, which is reported as such rather than as nothing being
/// active. A narrowed listing still uses the full managed set to decide all
/// this.
async fn list_workspaces(only: Option<&str>) -> anyhow::Result<String> {
    let workspaces = crate::users::managed_workspaces().await?;
    if workspaces.is_empty() {
        return Ok(NO_WORKSPACE.to_string());
    }
    let resolved =
        crate::users::resolve_selected_workspace_name(crate::users::ADMIN_USER_NAME).await;
    let active = resolved
        .as_deref()
        .filter(|name| workspaces.iter().any(|ws| &ws.name == name));
    let listed = match only {
        Some(name) => vec![managed_workspace(name).await?],
        None => workspaces,
    };
    let mut listing = String::from("Managed workspaces:\n");
    for ws in &listed {
        let paused = if ws.paused { ", paused" } else { "" };
        let maintenance = if ws.maintenance_enabled {
            ", maintenance on"
        } else {
            ""
        };
        let active_marker = if active == Some(ws.name.as_str()) {
            " [active]"
        } else {
            ""
        };
        let _ = writeln!(
            listing,
            "- {} ({}{paused}{maintenance}): {}{active_marker}",
            ws.name, ws.status, ws.path
        );
    }
    if resolved.is_none() {
        let _ = writeln!(listing, "The active workspace could not be read.");
    } else if active.is_none() {
        let _ = writeln!(listing, "No workspace is active.");
    }
    Ok(listing.trim_end().to_string())
}

// ── add ──────────────────────────────────────────────────────────

/// Add a managed workspace and make it the admin's active one — adding and
/// activating are one rule on every path (the desktop's own add form does the
/// same), so a newly added project is the one the desktop works with.
///
/// PARTIAL exactly when the add succeeded but the activation write did not: the
/// workspace is added either way, and the answer already says which half
/// happened.
async fn add_workspace(args: &serde_json::Value) -> anyhow::Result<ToolAnswer> {
    let name = super::get_str(args, "name")?;
    let path = super::get_str(args, "path")?;
    crate::workspace::store().add(name, path).await?;
    // Adding activates: the row exists by now, so a failed selection write is
    // reported as the partial outcome it is rather than as an add that did not
    // happen.
    match crate::users::store()
        .set_selected_workspace(crate::users::ADMIN_USER_NAME, Some(name))
        .await
    {
        Ok(()) => Ok(ToolAnswer::done(format!(
            "Added managed workspace '{name}' at {path} and made it your active workspace. \
             The workspace is being picked up — if the model provider is configured its first \
             discovery pass will start shortly."
        ))),
        Err(e) => Ok(ToolAnswer::partial(format!(
            "Added managed workspace '{name}' at {path}, but making it your active workspace \
             failed: {e} — activate it from the desktop's own workspace list."
        ))),
    }
}

// ── pause / resume ───────────────────────────────────────────────

/// What a pause/resume request does to one workspace — the rule both forms
/// below follow, so the named and the fleet-wide answer can never disagree about
/// which rows are written.
enum PipelineChange {
    /// Not `ready`: only a `ready` row is dispatchable (the poll gates every
    /// claim and phase dispatch on the status), so a non-ready row's `paused`
    /// flag is the discovery's own analysis pause rather than a freeze of running
    /// work, and it is never written.
    NotReady,
    /// Already in the requested state: reported as such, never written again.
    Already,
    /// The request changes the state and needs the write.
    Write,
}

fn pipeline_change(ws: &Workspace, paused: bool) -> PipelineChange {
    if ws.status != WorkspaceStatus::Ready {
        PipelineChange::NotReady
    } else if ws.paused == paused {
        PipelineChange::Already
    } else {
        PipelineChange::Write
    }
}

/// Pause or resume the named workspace's pipeline, or every managed
/// workspace at once when no name was given.
async fn pause_or_resume(name: Option<&str>, paused: bool) -> anyhow::Result<ToolAnswer> {
    match name {
        Some(name) => Ok(ToolAnswer::done(
            set_one_pipeline_paused(name, paused).await?,
        )),
        None => set_all_pipelines_paused(paused).await,
    }
}

/// Pause or resume one workspace's pipeline.
async fn set_one_pipeline_paused(name: &str, paused: bool) -> anyhow::Result<String> {
    let ws = managed_workspace(name).await?;
    match pipeline_change(&ws, paused) {
        PipelineChange::NotReady => Err(anyhow!(
            "Workspace '{}' is {} — only a ready workspace has a pipeline to pause or resume.",
            ws.name,
            ws.status
        )),
        PipelineChange::Already => Ok(if paused {
            format!("The pipeline of workspace '{}' is already paused.", ws.name)
        } else {
            format!(
                "The pipeline of workspace '{}' is not paused — nothing to resume.",
                ws.name
            )
        }),
        PipelineChange::Write => {
            crate::workspace::store()
                .set_paused(&ws.name, paused)
                .await?;
            let verb = if paused { "paused" } else { "resumed" };
            Ok(format!("Pipeline {verb} for workspace '{}'.", ws.name))
        }
    }
}

/// Pause or resume every workspace of `workspaces` where the write is honest —
/// see [`pipeline_change`] — one line per workspace, or [`NO_WORKSPACE`] for an
/// empty set.
///
/// Takes the store the writes go to and the set they cover, so the fleet-wide
/// form is not hardwired to the process-global store: production passes that one
/// with the managed workspaces, the test its own store with rows of its own,
/// and a fleet-wide write can then never reach a row another test created.
///
/// Never [`WorkspaceStore::pause_all`](crate::workspace::WorkspaceStore::pause_all):
/// that would also pause not-ready rows and the discovery's own analysis pause.
///
/// PARTIAL exactly when the requested state was not reached for a workspace: a
/// write that failed (the rows written before it are real, so the fleet-wide form
/// did not do the whole of what was asked) or a workspace skipped as not ready
/// (its pipeline was not paused/resumed, and the answer above says so). A
/// workspace already in the requested state leaves the operation whole — the
/// state the caller asked for holds.
///
/// The not-ready skip counts in BOTH directions on purpose, `resume` included,
/// where a pending workspace is not paused anyway: the named form refuses a
/// not-ready workspace the same way in both directions ("only a ready workspace
/// has a pipeline to pause or resume" — its flag is the discovery's own analysis
/// pause, not a freeze of running work), and one rule for both forms is what
/// keeps them from disagreeing about the same workspace.
async fn set_pipelines_paused(
    store: &WorkspaceStore,
    workspaces: &[Workspace],
    paused: bool,
) -> anyhow::Result<ToolAnswer> {
    if workspaces.is_empty() {
        return Ok(ToolAnswer::done(NO_WORKSPACE));
    }
    let header = if paused {
        "Paused every managed workspace where possible:"
    } else {
        "Resumed every managed workspace where possible:"
    };
    let mut out = format!("{header}\n");
    let mut incomplete = false;
    for ws in workspaces {
        let line = match pipeline_change(ws, paused) {
            // The caller asked for every managed workspace; this one was left
            // untouched, so the call is not the whole of what was asked.
            PipelineChange::NotReady => {
                incomplete = true;
                format!("- {}: skipped — not ready ({})", ws.name, ws.status)
            }
            PipelineChange::Already if paused => format!("- {}: already paused", ws.name),
            PipelineChange::Already => format!("- {}: not paused", ws.name),
            PipelineChange::Write => match store.set_paused(&ws.name, paused).await {
                Ok(()) if paused => format!("- {}: paused", ws.name),
                Ok(()) => format!("- {}: resumed", ws.name),
                // Reported like every other outcome: the rows written before a
                // failure are real, so the report must cover each workspace.
                Err(e) => {
                    incomplete = true;
                    format!("- {}: failed — {e}", ws.name)
                }
            },
        };
        let _ = writeln!(out, "{line}");
    }
    let report = out.trim_end().to_string();
    Ok(if incomplete {
        ToolAnswer::partial(report)
    } else {
        ToolAnswer::done(report)
    })
}

/// Every managed workspace at once: the global store and the set it must
/// cover.
async fn set_all_pipelines_paused(paused: bool) -> anyhow::Result<ToolAnswer> {
    let workspaces = crate::users::managed_workspaces().await?;
    set_pipelines_paused(crate::workspace::store(), &workspaces, paused).await
}

// ── board ────────────────────────────────────────────────────────

/// List the live tickets of the managed workspaces, optionally narrowed to
/// one workspace and/or one phase.
async fn list_board(only: Option<&str>, phase: Option<&str>) -> anyhow::Result<String> {
    let workspaces = crate::users::managed_workspaces().await?;
    let phase_filter = match phase {
        Some(s) if s.eq_ignore_ascii_case("archived") => {
            anyhow::bail!(
                "Archived tickets are never listed — archiving is a one-way visibility switch that hides a ticket from every listing, and a single archived ticket stays readable with `ticket` by its id."
            );
        }
        // `TicketPhase::from_str`'s error already lists the valid phases.
        Some(s) => Some(s.parse::<TicketPhase>()?),
        None => None,
    };
    let only = match only {
        Some(name) => Some(managed_workspace(name).await?.name),
        None => None,
    };
    let mut tickets = board_store().list_all_tickets(None, phase_filter).await?;
    // A ticket of a workspace that no longer exists is never listed.
    tickets.retain(|t| {
        workspaces.iter().any(|ws| ws.name == t.workspace_name)
            && only.as_ref().is_none_or(|name| name == &t.workspace_name)
    });
    if tickets.is_empty() {
        return Ok("No tickets found.".to_string());
    }
    let total = tickets.len();
    let mut lines = Vec::new();
    let mut bytes = 0;
    for t in &tickets {
        let mut line = t.short_display();
        if !lines.is_empty()
            && (lines.len() == BOARD_LINE_LIMIT || bytes + line.len() > OUTPUT_BYTE_LIMIT)
        {
            break;
        }
        // The first line is always shown — the notice must never read "the first
        // 0 of N tickets" — so a line larger than the whole budget is cut rather
        // than escaping the tool's own limit.
        line = crate::util::cut_bytes(&line, OUTPUT_BYTE_LIMIT);
        bytes += line.len() + 1;
        lines.push(line);
    }
    let shown = lines.len();
    let mut out = String::from("Tickets:\n");
    for line in &lines {
        let _ = writeln!(out, "{line}");
    }
    if shown < total {
        let _ = write!(
            out,
            "\n(partial listing: showing the first {shown} of {total} tickets — \
             narrow it with `phase`, name a single `workspace`, or read one ticket with `ticket`.)"
        );
    }
    Ok(out)
}

// ── ticket / comment ─────────────────────────────────────────────

/// Read one ticket by its full id. The default is the shortened form for every
/// ticket, whoever filed it.
async fn show_ticket(ticket_id: &str, full: bool) -> anyhow::Result<String> {
    let id = managed_ticket_id(ticket_id).await?;
    let Some(ticket) = board_store().get_ticket(&id).await? else {
        return Err(anyhow!("Ticket {id} not found."));
    };
    if full {
        return Ok(ticket.detailed_display());
    }
    let shortened = ticket.detailed_display_limited(
        GET_TICKET_DESC_MAX,
        GET_TICKET_COMMENT_MAX,
        GET_TICKET_LAST_N_FULL,
    );
    // The newest comments come back in full, so a ticket carrying a huge one is
    // still bounded here — `full: true` is the unbounded read.
    if shortened.len() <= OUTPUT_BYTE_LIMIT {
        return Ok(shortened);
    }
    Ok(format!(
        "{}…\n(shortened view cut short — pass `full: true` for the ticket whole and unclipped.)",
        crate::util::truncate_bytes(&shortened, OUTPUT_BYTE_LIMIT)
    ))
}

/// Leave a clarification note on a ticket. Refused when the ticket does not
/// exist, so a note is never stored for a missing ticket.
async fn add_note(ticket_id: &str, content: &str) -> anyhow::Result<String> {
    anyhow::ensure!(
        !content.trim().is_empty(),
        "usage: \"content\" is empty — hint: pass the note's text to leave on the ticket"
    );
    let id = managed_ticket_id(ticket_id).await?;
    if board_store().get_ticket_phase(&id).await?.is_none() {
        return Err(anyhow!("Ticket {id} not found — no note was left."));
    }
    board_store()
        .add_comment(&id, Role::Assistant.as_str(), content)
        .await?;
    Ok(format!("Note added to ticket {id} as 'assistant'."))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_args(action: &str, name: Option<&str>) -> serde_json::Value {
        match name {
            Some(name) => json!({ "action": action, "workspace": name }),
            None => json!({ "action": action }),
        }
    }

    /// The refusals that keep the tool from reporting a change it did not make.
    #[tokio::test]
    async fn refusals_are_plain_and_named() {
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/ws_refusals");
        let tool = WorkspaceTool;

        let err = tool
            .execute(&ws, tool_args("pause", Some("ws_absent")))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("No managed workspace exists"), "got: {err}");

        let err = tool
            .execute(&ws, tool_args("resume", Some("personal:admin")))
            .await
            .unwrap_err()
            .to_string();
        // A personal space is refused as such and still names the live set.
        assert!(
            err.contains("personal space") && err.contains("managed workspaces:"),
            "got: {err}"
        );

        // A freshly added workspace is `pending` — it has no pipeline yet,
        // so nothing may be frozen or resumed.
        crate::util::test::create_test_workspace("/tmp/ws_pending", "ws_pending").await;
        let err = tool
            .execute(&ws, tool_args("pause", Some("ws_pending")))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("pending") && err.contains("ready"),
            "got: {err}"
        );
        assert!(
            !crate::workspace::get_by_name("ws_pending")
                .await
                .expect("read")
                .expect("exists")
                .paused,
            "the refusal must not pause the workspace"
        );
    }

    #[tokio::test]
    async fn pause_and_resume_a_ready_workspace() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/ws_ready");
        let tool = WorkspaceTool;
        crate::util::test::create_test_workspace("/tmp/ws_ready", "ws_ready").await;
        crate::workspace::store()
            .set_status("ws_ready", &WorkspaceStatus::Ready)
            .await
            .expect("mark ready");

        let reply = tool
            .execute(&ws, tool_args("pause", Some("ws_ready")))
            .await
            .expect("pause");
        assert!(
            reply.contains("ws_ready") && reply.contains("paused"),
            "got: {reply}"
        );
        assert!(
            crate::workspace::get_by_name("ws_ready")
                .await
                .expect("read")
                .expect("exists")
                .paused
        );

        let listing = tool
            .execute(&ws, tool_args("list", None))
            .await
            .expect("list");
        assert!(
            listing.contains("- ws_ready (ready, paused):"),
            "got: {listing}"
        );

        // Re-pausing reports the state it is already in, never a second freeze.
        let reply = tool
            .execute(&ws, tool_args("pause", Some("ws_ready")))
            .await
            .expect("re-pause");
        assert!(reply.contains("already paused"), "got: {reply}");

        let reply = tool
            .execute(&ws, tool_args("resume", Some("ws_ready")))
            .await
            .expect("resume");
        assert!(reply.contains("resumed"), "got: {reply}");
        assert!(
            !crate::workspace::get_by_name("ws_ready")
                .await
                .expect("read")
                .expect("exists")
                .paused
        );
        assert!(
            !tool
                .execute(&ws, tool_args("list", None))
                .await
                .expect("list")
                .contains("- ws_ready (ready, paused)"),
            "a resumed workspace must not carry the pause marker"
        );
    }

    /// Every managed workspace at once: the ready one is written, the
    /// not-ready one is skipped and reported as skipped.
    ///
    /// Driven against the test's own store, so the fleet-wide write can only ever
    /// reach rows this test created.
    #[tokio::test]
    async fn pause_every_managed_workspace_skips_not_ready() {
        let (store, _dir) = crate::open_test_store!(WorkspaceStore, "workspace");
        let root = crate::util::test::test_root();
        let ready = root.join("ws_fleet");
        let pending = root.join("ws_fleet_pending");
        for dir in [&ready, &pending] {
            std::fs::create_dir_all(dir).expect("create workspace dir");
        }
        let ready = ready.to_str().expect("utf-8 path");
        store.add("ws_fleet", ready).await.expect("add ws_fleet");
        store
            .set_status("ws_fleet", &WorkspaceStatus::Ready)
            .await
            .expect("mark ready");
        // A fresh row is created paused (the pickup step lifts it), so the fleet
        // pause below starts from a running pipeline.
        store.set_paused("ws_fleet", false).await.expect("unpause");
        // Another not-ready workspace must be skipped rather than written — left
        // unpaused, so a write would show.
        store
            .add("ws_fleet_pending", pending.to_str().expect("utf-8 path"))
            .await
            .expect("add ws_fleet_pending");
        store
            .set_paused("ws_fleet_pending", false)
            .await
            .expect("unpause ws_fleet_pending");
        let workspaces = store.list().await.expect("list");

        let reply = set_pipelines_paused(&store, &workspaces, true)
            .await
            .expect("pause all");
        assert!(
            reply.text.contains("- ws_fleet: paused"),
            "got: {}",
            reply.text
        );
        assert!(
            reply
                .text
                .contains("- ws_fleet_pending: skipped — not ready (pending)"),
            "got: {}",
            reply.text
        );
        assert!(
            reply.partial,
            "a workspace the call left untouched makes the fleet call partial: {}",
            reply.text
        );
        assert!(
            store
                .get_by_name("ws_fleet")
                .await
                .expect("read")
                .expect("exists")
                .paused
        );
        assert!(
            !store
                .get_by_name("ws_fleet_pending")
                .await
                .expect("read")
                .expect("exists")
                .paused,
            "a not-ready workspace must not be paused"
        );

        // A second fleet call, resuming: the ready workspace is written again and
        // the not-ready one is still skipped.
        let workspaces = store.list().await.expect("list");
        let reply = set_pipelines_paused(&store, &workspaces, false)
            .await
            .expect("resume all");
        assert!(
            reply.text.contains("- ws_fleet: resumed"),
            "got: {}",
            reply.text
        );
        assert!(
            reply.partial,
            "the not-ready workspace is left untouched by the resume too: {}",
            reply.text
        );

        // Nothing managed: the same plain wording `list` uses.
        let reply = set_pipelines_paused(&store, &[], true)
            .await
            .expect("empty set");
        assert_eq!(reply.text, NO_WORKSPACE);
    }

    /// The unnamed form of `pause` / `resume` is the fleet-wide write, so it may
    /// only be reached by a call that names nothing else: any other field —
    /// whether it names a place (`name` and `path` belong to `add`, a ticket id
    /// to the ticket actions), nothing this tool knows, or the ticket id the
    /// product's own argument repair makes of a mistaken `id` — is a usage error.
    /// The read actions take the same address and ignore such a field, as every
    /// other tool does.
    #[tokio::test]
    async fn a_misspelled_address_never_escalates_to_the_fleet() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let tool = WorkspaceTool;
        crate::util::test::create_test_workspace("/tmp/ws_address", "ws_address").await;
        crate::workspace::store()
            .set_status("ws_address", &WorkspaceStatus::Ready)
            .await
            .expect("mark ready");
        let ws = crate::workspace::test_ws("/tmp/ws_address");

        for call in [
            json!({ "action": "pause", "name": "ws_address" }),
            json!({ "action": "resume", "path": "/tmp/ws_address" }),
            json!({ "action": "resume", "ticket_id": "ws_address-1" }),
            json!({ "action": "pause", "workspace_name": "ws_address" }),
            crate::tools::normalize_tool_call(
                "workspace",
                json!({ "action": "pause", "id": "ws_address" }),
            )
            .1,
        ] {
            let err = tool
                .execute(&ws, call.clone())
                .await
                .unwrap_err()
                .to_string();
            assert!(err.contains("usage:"), "for {call}: {err}");
        }

        // A wrong-typed address is a usage error too, and it names the field the
        // caller did write.
        for value in [
            json!(3),
            json!(["ws_address"]),
            json!({ "name": "ws_address" }),
        ] {
            let err = tool
                .execute(&ws, json!({ "action": "resume", "workspace": value }))
                .await
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("usage:") && err.contains("workspace"),
                "got: {err}"
            );
        }
        assert!(
            !crate::workspace::get_by_name("ws_address")
                .await
                .expect("read")
                .expect("exists")
                .paused,
            "a refused address must not pause anything"
        );

        // A stray field on a read action is ignored rather than refused: the
        // answer is the listing the call gives without it, neither widened nor
        // dropped. (Narrowed to this one workspace, so a test running beside this
        // one cannot change what is listed.)
        for args in [
            json!({ "action": "list", "workspace": "ws_address" }),
            json!({ "action": "list", "workspace": "ws_address", "folder": "/tmp/ws_address" }),
        ] {
            let listing = tool.execute(&ws, args.clone()).await.expect("list");
            assert!(
                listing.starts_with("Managed workspaces:\n- ws_address ("),
                "for {args}: {listing}"
            );
        }

        // The named form still reaches exactly the workspace it names.
        let reply = tool
            .execute(&ws, json!({ "action": "pause", "workspace": "ws_address" }))
            .await
            .expect("pause");
        assert!(
            reply.contains("Pipeline paused for workspace 'ws_address'"),
            "got: {reply}"
        );
    }

    /// A bare number is never a valid `ticket` / `comment` address.
    #[tokio::test]
    async fn ticket_actions_refuse_a_bare_number() {
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/ws_bare");
        let tool = WorkspaceTool;

        let err = tool
            .execute(&ws, json!({ "action": "ticket", "ticket_id": "42" }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("<workspace>-<number>"), "got: {err}");

        let err = tool
            .execute(
                &ws,
                json!({ "action": "comment", "ticket_id": "42", "content": "hi" }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("<workspace>-<number>"), "got: {err}");
    }

    /// `ticket` refuses an id whose workspace is not managed and an id with
    /// no such ticket.
    #[tokio::test]
    async fn ticket_refuses_a_not_managed_workspace_and_missing_ticket() {
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/ws_ticket");
        let tool = WorkspaceTool;

        let err = tool
            .execute(
                &ws,
                json!({ "action": "ticket", "ticket_id": "ws_ghost-1" }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("No managed workspace exists"), "got: {err}");

        crate::util::test::create_test_workspace("/tmp/ws_ticket", "ws_ticket").await;
        let err = tool
            .execute(
                &ws,
                json!({ "action": "ticket", "ticket_id": "ws_ticket-9999" }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("not found"), "got: {err}");
    }

    /// `comment` on a ticket that does not exist leaves no note.
    #[tokio::test]
    async fn comment_refuses_a_missing_ticket() {
        crate::util::test::init_test_stores().await;
        let tool = WorkspaceTool;
        crate::util::test::create_test_workspace("/tmp/ws_note", "ws_note").await;
        let ws = crate::workspace::test_ws("/tmp/ws_note");

        let err = tool
            .execute(
                &ws,
                json!({ "action": "comment", "ticket_id": "ws_note-9999", "content": "   " }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("usage:"), "got: {err}");

        let err = tool
            .execute(
                &ws,
                json!({ "action": "comment", "ticket_id": "ws_note-9999", "content": "hi" }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("not found") && err.contains("no note"),
            "got: {err}"
        );
        assert!(
            board_store()
                .get_comments("ws_note-9999")
                .await
                .expect("read")
                .is_empty(),
            "a missing ticket must gain no note"
        );
    }

    /// `board` refuses the archived phase, lists finished tickets of managed
    /// workspaces and never a ticket of one that is not managed.
    #[tokio::test]
    async fn board_refuses_archived_and_skips_workspaces_that_are_not_managed() {
        crate::util::test::init_test_stores().await;
        let tool = WorkspaceTool;
        let ws = crate::util::test::create_test_workspace("/tmp/ws_board", "ws_board").await;
        let done_id = crate::util::test::make_ticket(
            board_store(),
            &ws,
            "workspace board done",
            TicketPhase::Done,
        )
        .await;
        // A ticket of a workspace that is not managed.
        let ghost = crate::workspace::test_ws_named("/tmp/ws_ghost", "ws_ghost");
        let ghost_id =
            crate::util::test::make_ticket(board_store(), &ghost, "ghost done", TicketPhase::Done)
                .await;

        let err = tool
            .execute(&ws, json!({ "action": "board", "phase": "archived" }))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Archived tickets are never listed"),
            "got: {err}"
        );

        let listing = tool
            .execute(&ws, json!({ "action": "board", "phase": "done" }))
            .await
            .expect("board");
        assert!(
            listing.contains(&done_id),
            "a done ticket of a managed workspace must be listed: {listing}"
        );
        assert!(
            !listing.contains(&ghost_id),
            "a ticket of a workspace that is not managed must never be listed: {listing}"
        );
        // The default listing keeps finished tickets — deliberately unlike the
        // manager's own listing, which hides them.
        let listing = tool
            .execute(&ws, json!({ "action": "board" }))
            .await
            .expect("board");
        assert!(
            listing.contains(&done_id),
            "a done ticket must be listed without a phase filter: {listing}"
        );
    }

    /// `list` marks the admin's selected workspace and says so when none is.
    #[tokio::test]
    #[serial_test::serial(gui_admin_workspace)] // writes the shared seeded admin row
    async fn list_marks_the_active_workspace_or_says_none() {
        crate::util::test::init_test_stores().await;
        let tool = WorkspaceTool;
        let ws = crate::workspace::test_ws("/tmp/ws_active");
        crate::users::test_util::with_test_workspaces(
            &[("/tmp/ws_active", "ws_active")],
            crate::users::test_util::with_admin_workspace_restored(async {
                crate::users::store()
                    .set_selected_workspace(crate::users::ADMIN_USER_NAME, Some("ws_active"))
                    .await
                    .expect("select");
                let listing = tool
                    .execute(&ws, tool_args("list", None))
                    .await
                    .expect("list");
                assert!(listing.contains("- ws_active ("), "got: {listing}");
                assert!(listing.contains("[active]"), "got: {listing}");
                assert!(
                    !listing.contains("No workspace is active."),
                    "got: {listing}"
                );

                // A personal selection is not a managed workspace, so
                // nothing is active.
                crate::users::store()
                    .set_selected_workspace(crate::users::ADMIN_USER_NAME, Some("personal:admin"))
                    .await
                    .expect("select personal");
                let listing = tool
                    .execute(&ws, tool_args("list", None))
                    .await
                    .expect("list");
                assert!(
                    listing.contains("No workspace is active."),
                    "got: {listing}"
                );
                assert!(!listing.contains("[active]"), "got: {listing}");
            }),
        )
        .await;
    }
}
