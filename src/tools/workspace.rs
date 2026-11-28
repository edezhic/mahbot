//! The admin's own view into the managed workspaces: list them, add one (which activates
//! it), inspect one, delete one, freeze/unfreeze their pipelines, switch maintenance on or
//! off, restart discovery, write their rules and their diagnostics commands, read and change
//! their profile, make one the active project, read a board and a single ticket, and leave a
//! clarification note on a ticket.
//!
//! The detail of that surface is not here: every action and every field is documented in the
//! `reference` action ([`crate::settings::reference`], the same text `mahbot config reference`
//! prints), so the standing description stays small and every field's description stays short.
//!
//! It manages no board. Adding a managed workspace activates it; the desktop itself moves
//! the active project when the active one is deleted, and onto a project at start-up
//! whenever the stored one no longer exists.
use anyhow::anyhow;
use async_trait::async_trait;
use serde_json::json;
use std::fmt::Write as _;

use crate::pipeline::board::{TicketPhase, store as board_store};
use crate::tools::ToolAnswer;
use crate::tools::ticket::{GET_TICKET_COMMENT_MAX, GET_TICKET_DESC_MAX, GET_TICKET_LAST_N_FULL};
use crate::workspace::WorkspaceStore;
use crate::workspace_profile;
use crate::{Role, Tool, Workspace, WorkspaceStatus};

/// The tool's own byte budget, needed because the framework's ~5 KB tool-output truncation is
/// off for it (see [`WorkspaceTool::preserve_full_output`]): the `board` listing and the
/// shortened `ticket` view bound themselves, and a byte cap is required alongside
/// `BOARD_LINE_LIMIT` because a title, a description and a comment are all unbounded.
/// `ticket` with `full: true` is deliberately exempt.
const OUTPUT_BYTE_LIMIT: usize = 24_000;
const BOARD_LINE_LIMIT: usize = 100;

/// What an action that covers the whole managed set answers when there is none — one
/// wording, so `list` and a fleet-wide pause say the same thing.
const NO_WORKSPACE: &str = "No managed workspaces exist.";

/// The `workspace` tool: the admin's project view and the settings of each project —
/// `reference`, `list`, `show`, `add`, `delete`, `pause`, `resume`, `maintenance`,
/// `rediscover`, `rediscover_diagnostics`, `set_diagnostics`, `set_rules`, `use`, `profile`,
/// `confirm_profile`, `set_profile`, `reset_profile`, `board`, `ticket`, `comment`.
pub(crate) struct WorkspaceTool;

/// This tool's schema, built once: [`Tool::parameters_schema`] hands out a copy, and the
/// unknown-field guard reads the same list without rebuilding it on every call.
fn schema() -> &'static serde_json::Value {
    static SCHEMA: std::sync::LazyLock<serde_json::Value> = std::sync::LazyLock::new(build_schema);
    &SCHEMA
}

/// Every action the tool serves, in the order the struct doc lists them.
const ACTIONS: [&str; 20] = [
    "reference",
    "list",
    "show",
    "add",
    "delete",
    "pause",
    "resume",
    "maintenance",
    "rediscover",
    "rediscover_diagnostics",
    "set_diagnostics",
    "set_rules",
    "use",
    "profile",
    "confirm_profile",
    "set_profile",
    "reset_profile",
    "board",
    "ticket",
    "comment",
];

/// The profile fields' descriptions, shared by the `workspace` and `config` schemas: one text for
/// the one setting, whichever tool the assistant reaches it through.
pub(crate) const STAGE_FIELD: &str = "The stage whose participants change: analysis, planning, \
                                      in_development, verification or in_sanitation — passed \
                                      with its `participants`.";
pub(crate) const PARTICIPANTS_FIELD: &str = "The stage's participants: analyst, manager, \
                                             engineer, reviewer, tester or sanitation. An empty \
                                             list returns the stage to the default.";
pub(crate) const PASS_THRESHOLD_FIELD: &str = "Verification's passing score, 0-10.";
pub(crate) const REQUIRE_ALL_CHECKS_FIELD: &str = "Whether every project command must pass.";

fn build_schema() -> serde_json::Value {
    let fields = json!({
        "action": {
            "type": "string",
            "enum": ACTIONS,
            "description": "The operation; its `reference` action lists every one."
        },
        "workspace": {
            "type": "string",
            "description": "The managed workspace, spelled as `list` spells it."
        },
        "name": {
            "type": "string",
            "description": "(add) Short unique name."
        },
        "path": {
            "type": "string",
            "description": "(add) The project directory."
        },
        "phase": {
            "type": "string",
            "description": "(board) One ticket phase."
        },
        "ticket_id": {
            "type": "string",
            "description": "(ticket/comment) Full id `<workspace>-<number>`."
        },
        "full": {
            "type": "boolean",
            "description": "(ticket) Whole and unclipped. Default false."
        },
        "content": {
            "type": "string",
            "description": "(comment) The note to leave."
        },
        "rules": {
            "type": "string",
            "description": "(set_rules) Project rules; empty clears."
        },
        "maintenance": {
            "type": "boolean",
            "description": "(maintenance) Background maintainer on or off."
        },
        "format": {
            "type": "string",
            "description": "(set_diagnostics) Format command."
        },
        "format_check": {
            "type": "string",
            "description": "(set_diagnostics) Format-check command."
        },
        "lint": {
            "type": "string",
            "description": "(set_diagnostics) Lint command."
        },
        "lint_fix": {
            "type": "string",
            "description": "(set_diagnostics) Lint-fix command."
        },
        "type_check": {
            "type": "string",
            "description": "(set_diagnostics) Type-check command."
        },
        "build": {
            "type": "string",
            "description": "(set_diagnostics) Build command."
        },
        "unit_test": {
            "type": "string",
            "description": "(set_diagnostics) Unit-test command."
        },
        "stage": {
            "type": "string",
            "description": STAGE_FIELD
        },
        "participants": {
            "type": "array",
            "items": { "type": "string" },
            "description": PARTICIPANTS_FIELD
        },
        "pass_threshold": {
            "type": "integer",
            "description": PASS_THRESHOLD_FIELD
        },
        "require_all_checks": {
            "type": "boolean",
            "description": REQUIRE_ALL_CHECKS_FIELD
        }
    });
    super::tool_params_schema(&fields, &["action"])
}

#[async_trait]
impl Tool for WorkspaceTool {
    fn name(&self) -> &'static str {
        "workspace"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        schema().clone()
    }

    fn preserve_full_output(&self) -> bool {
        // A `ticket` answer can carry hundreds of KB of comments, so the fixed ~5 KB
        // truncation is wrong for the whole tool; the read paths bound themselves.
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
    /// The one action dispatch both faces of the tool go through: `execute` renders it to its
    /// text, and [`Tool::execute_answer`] hands it over with the partiality flag a command-line
    /// caller reads. There is no second code path to drift from this one.
    async fn answer(&self, args: serde_json::Value) -> anyhow::Result<ToolAnswer> {
        // Before anything is read: a field the schema does not declare is a usage error naming
        // the reference, never silently ignored — a misspelled `workspace` would otherwise act
        // on the whole fleet.
        super::refuse_unknown_fields(schema(), &args, super::REFERENCE_ACTION_HINT)?;
        match super::get_str(&args, "action")? {
            "reference" => Ok(ToolAnswer::done(crate::settings::reference())),
            "list" => Ok(ToolAnswer::done(
                list_workspaces(super::get_opt_str_strict(&args, "workspace")?).await?,
            )),
            "show" => Ok(ToolAnswer::done(
                show_workspace(super::get_str(&args, "workspace")?).await?,
            )),
            "add" => add_workspace(&args).await,
            "delete" => Ok(ToolAnswer::done(
                delete_workspace(super::get_str(&args, "workspace")?).await?,
            )),
            "pause" => pause_or_resume(write_address(&args, "pause")?, true).await,
            "resume" => pause_or_resume(write_address(&args, "resume")?, false).await,
            "maintenance" => Ok(ToolAnswer::done(set_maintenance(&args).await?)),
            "rediscover" => Ok(ToolAnswer::done(
                rediscover(super::get_str(&args, "workspace")?).await?,
            )),
            "rediscover_diagnostics" => Ok(ToolAnswer::done(
                rediscover_diagnostics(super::get_str(&args, "workspace")?).await?,
            )),
            "set_diagnostics" => Ok(ToolAnswer::done(set_diagnostics(&args).await?)),
            "set_rules" => Ok(ToolAnswer::done(set_rules(&args).await?)),
            "profile" => Ok(ToolAnswer::done(
                show_profile(super::get_str(&args, "workspace")?).await?,
            )),
            "confirm_profile" => Ok(ToolAnswer::done(
                confirm_profile(super::get_str(&args, "workspace")?).await?,
            )),
            "set_profile" => Ok(ToolAnswer::done(set_profile(&args).await?)),
            "reset_profile" => Ok(ToolAnswer::done(
                reset_profile(super::get_str(&args, "workspace")?).await?,
            )),
            "use" => Ok(ToolAnswer::done(
                use_workspace(super::get_str(&args, "workspace")?).await?,
            )),
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
                "unknown action '{other}' — call the `reference` action for every action and field"
            )),
        }
    }
}

/// The workspace a `pause` / `resume` was pointed at: the `workspace` argument, or `None`
/// for every managed workspace when none is named.
///
/// The one action pair that can act on a single workspace or on all of them must never widen
/// because the address was spelled some other way — a folder path, a ticket id, or `add`'s
/// `name` — so with no `workspace` argument such a call may name nothing else at all. An
/// extra declared field is a usage error phrased around that rule, never around the key that
/// carried it; a key no schema declares is refused earlier, for every action
/// ([`super::refuse_unknown_fields`]). The read actions take the same address but ignore a
/// stray declared field; only a write that would cover the fleet has to say so.
///
/// A JSON `null` counts as absent, as everywhere in the product (providers fill omitted
/// arguments with nulls).
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

/// Resolve `name` to a managed workspace — never a personal space, including a legacy
/// `personal:*` row. Matching is exact, and a miss names the live set because the admin names
/// workspaces from a session-start context block that may predate an add or a delete.
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

/// Resolve a full `<workspace>-<number>` ticket id, refusing every other spelling with the
/// expected form, and an id whose workspace prefix is not a managed workspace.
async fn managed_ticket_id(raw: &str) -> anyhow::Result<String> {
    // Only the full spelling addresses a ticket here: the manager's own ticket tools resolve a
    // bare number against the workspace they are bound to, and this tool is bound to none.
    let Ok((Some(prefix), full)) = crate::tools::ticket::parse_ticket_id(raw) else {
        return Err(anyhow!(
            "Ticket id '{}' is not a full '<workspace>-<number>' id — `ticket` and `comment` \
             are addressed by a ticket's full id, which already carries its workspace.",
            raw.trim()
        ));
    };
    // An id naming a workspace that is not managed is refused as an unknown workspace, with
    // the managed set named.
    managed_workspace(prefix).await?;
    Ok(full.to_string())
}

/// The managed workspaces, one line each, with the admin's active one marked.
///
/// The active workspace is the stored selection and counts only when it is one of the managed
/// names: a personal value, or a name no longer managed, means nothing is active — while the
/// resolver's `None` is a failed read, reported as such. The desktop repairs such a selection
/// at start-up and after a deletion, so the two surfaces agree in the steady state.
async fn list_workspaces(only: Option<&str>) -> anyhow::Result<String> {
    let workspaces = crate::users::managed_workspaces().await?;
    // A named workspace is answered about itself: with none managed at all, that is the not-found
    // refusal naming the name, not a report on the empty fleet.
    if workspaces.is_empty() && only.is_none() {
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
        let waiting = crate::workspace::status_reason(ws, &workspace_profile::read(ws))
            .map(|reason| format!(", {reason}"))
            .unwrap_or_default();
        let _ = writeln!(
            listing,
            "- {} ({}{paused}{maintenance}{waiting}): {}{active_marker}",
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

/// Add a managed workspace and make it the admin's active one — adding and activating are one
/// rule on every path (the desktop's own add form does the same).
///
/// PARTIAL exactly when the add succeeded but the activation write did not.
async fn add_workspace(args: &serde_json::Value) -> anyhow::Result<ToolAnswer> {
    let name = super::get_str(args, "name")?;
    let path = super::get_str(args, "path")?;
    // The stored path, not the caller's spelling: the store canonicalises it, and an answer that
    // echoed `./proj` back would read as a different folder from the one `list` shows.
    let added = crate::workspace::store().add(name, path).await?;
    let path = added.path;
    // Adding activates: the row exists by now, so a failed selection write is reported as the
    // partial outcome it is rather than as an add that did not happen.
    match crate::users::store()
        .set_selected_workspace(crate::users::ADMIN_USER_NAME, Some(name))
        .await
    {
        Ok(()) => Ok(ToolAnswer::done(format!(
            "Added managed workspace '{name}' at {path} and made it your active workspace. \
             The workspace is being picked up — if the model provider is configured its first \
             discovery pass will start shortly. Its pipeline does not run until you confirm its \
             profile, which is the profile as shipped unless you edit it first: confirm it with \
             the `workspace` tool's `confirm_profile` action (or the `config` tool's)."
        ))),
        Err(e) => Ok(ToolAnswer::partial(format!(
            "Added managed workspace '{name}' at {path}, but making it your active workspace \
             failed: {e} — make it active again with the `use` action."
        ))),
    }
}

/// One workspace in full: what `list` shows, plus the rules and the diagnostics commands, so
/// the admin can see what an agent there is told and what verification will run.
///
/// The rules are the admin's own text, printed verbatim under an indented block (the store caps
/// them at [`MAX_WORKSPACE_NOTES_CHARS`](crate::workspace::MAX_WORKSPACE_NOTES_CHARS)), and the
/// commands come out label by label, an absent one spelled "not discovered".
async fn show_workspace(name: &str) -> anyhow::Result<String> {
    let ws = managed_workspace(name).await?;
    // The read's two negatives are different facts: for the admin a successful read always yields
    // `Some` (the personal workspace at worst), so `None` is a failed read and is reported as
    // unknown — never as "not active".
    let active = crate::users::resolve_selected_workspace_name(crate::users::ADMIN_USER_NAME).await;
    let commands = crate::workspace::store().get_diagnostics(&ws.name).await?;
    let read = workspace_profile::read(&ws);
    let mut out = format!("Workspace '{}':\n", ws.name);
    let _ = writeln!(out, "- path: {}", ws.path);
    let _ = writeln!(out, "- status: {}", ws.status);
    // One line, not two: a workspace that claims nothing is not "running" beside a wait — the
    // pipeline line carries the reason it is not, or says it starts nothing yet.
    let waiting = crate::workspace::status_reason(&ws, &read);
    let _ = writeln!(
        out,
        "- pipeline: {}",
        if ws.paused {
            "paused".to_string()
        } else if let Some(waiting) = waiting {
            waiting
        } else if matches!(
            ws.status,
            WorkspaceStatus::Pending | WorkspaceStatus::Failed
        ) {
            format!("nothing to claim while the workspace is {}", ws.status)
        } else {
            // Ready, or re-analysing: a pass over the folder is reading, so it never holds the
            // pipeline out.
            "running".to_string()
        }
    );
    // The parse error stays out of the listing line and is spelled here, where the owner asked.
    if let workspace_profile::ProfileState::Invalid(error) = &read.state {
        let _ = writeln!(out, "  (what cannot be read: {error})");
    }
    let _ = writeln!(
        out,
        "- maintenance: {}",
        if ws.maintenance_enabled { "on" } else { "off" }
    );
    let _ = writeln!(
        out,
        "- active project: {}",
        match active.as_deref() {
            Some(active) if active == ws.name => "yes",
            Some(_) => "no",
            None => "unknown (the selection could not be read)",
        }
    );
    if ws.notes.trim().is_empty() {
        let _ = writeln!(out, "- rules: none");
    } else {
        let _ = writeln!(out, "- rules:");
        for line in ws.notes.lines() {
            let _ = writeln!(out, "  {line}");
        }
    }
    match commands.as_ref().filter(|commands| !commands.is_empty()) {
        Some(commands) => {
            let _ = writeln!(out, "- diagnostics:");
            write_commands(&mut out, commands, "  ");
        }
        None => {
            let _ = writeln!(out, "- diagnostics: not discovered");
        }
    }
    Ok(out.trim_end().to_string())
}

/// The workspace profile as the owner's assistant reads it: what is in force, and the project's
/// checks — the check list keeps its own column and its own writers.
pub(crate) async fn show_profile(name: &str) -> anyhow::Result<String> {
    use crate::workspace_profile::ProfileState;
    let ws = managed_workspace(name).await?;
    let read = workspace_profile::read(&ws);
    let state = &read.state;
    let mut out = format!("Workspace '{}' profile:\n", ws.name);
    match state {
        ProfileState::Unconfirmed => {
            let _ = writeln!(
                out,
                "- status: waiting for confirmation — the pipeline does not run until the \
                 profile is confirmed; the `confirm_profile` action (here, or on the `config` \
                 tool) accepts it"
            );
        }
        ProfileState::Invalid(error) => {
            let _ = writeln!(
                out,
                "- status: the confirmed profile is invalid ({error}) — the workspace runs on \
                 the default profile until it is fixed"
            );
        }
        ProfileState::Confirmed => {
            let _ = writeln!(out, "- status: confirmed");
        }
    }
    // For an unconfirmed workspace the stages below are what confirmation would put in force, so
    // the header says so rather than reading as the roster the pipeline runs on.
    if matches!(state, ProfileState::Unconfirmed) {
        let _ = writeln!(out, "stages (what confirmation puts in force):");
    } else {
        let _ = writeln!(out, "stages:");
    }
    out.push_str(&workspace_profile::render(&read.profile));
    let _ = writeln!(out);
    let commands = crate::workspace::store().get_diagnostics(&ws.name).await?;
    match commands.as_ref().filter(|commands| !commands.is_empty()) {
        Some(commands) => {
            let _ = writeln!(out, "checks:");
            write_commands(&mut out, commands, "  ");
        }
        None => {
            let _ = writeln!(out, "checks: not discovered");
        }
    }
    Ok(out.trim_end().to_string())
}

/// Accept the workspace's profile. Confirmation records that the owner saw and accepted the
/// workspace's settings; it copies no value, so an unpinned field keeps following the default.
pub(crate) async fn confirm_profile(name: &str) -> anyhow::Result<String> {
    use crate::workspace_profile::Profile;
    let ws = managed_workspace(name).await?;
    let unreadable = ws
        .profile
        .as_deref()
        .and_then(|profile| Profile::resolve(Some(profile)).err());
    // Confirming is the one moment the owner is told, and the workspace then runs on the default
    // until the profile is repaired. The owner action returns the same information, so this is one
    // warning beside it, not a record the poll round has to remember.
    if let Some(error) = &unreadable {
        tracing::warn!(
            workspace_name = ws.name,
            error = %error,
            "The confirmed workspace profile cannot be used — running on the default profile \
             until it is fixed",
        );
    }
    crate::workspace::store()
        .confirm_profile(&ws.name)
        .await
        .map(|_| ())?;
    if unreadable.is_some() {
        return Ok(format!(
            "Workspace '{}' profile confirmed, but the stored profile cannot be read — the \
             workspace runs on the default profile until it is fixed; hint: `reset_profile` \
             returns it to the default, which repairs it.",
            ws.name
        ));
    }
    Ok(format!(
        "Workspace '{}' profile confirmed — the pipeline runs from the next stage round.",
        ws.name
    ))
}

/// Change part of the workspace's profile. Every named field is pinned; a stage with an
/// explicitly empty participant list follows the default again.
///
/// A stage and its participants are one edit: `participants` without a `stage` names nobody's
/// roster and a `stage` without `participants` would read as "this stage has nobody" and wipe it,
/// so either alone is a usage error. An invalid result is refused and nothing is stored.
pub(crate) async fn set_profile(args: &serde_json::Value) -> anyhow::Result<String> {
    use crate::workspace_profile::StoredProfile;
    let ws = managed_workspace(super::get_str(args, "workspace")?).await?;
    let mut stored = match ws.profile.as_deref() {
        Some(json) => StoredProfile::parse(json).map_err(|e| {
            anyhow!(
                "{e} — hint: `reset_profile` returns the profile to the default, which repairs \
                 a stored profile this build cannot read"
            )
        })?,
        None => StoredProfile::default(),
    };
    // The two fields are read as a PAIR: an absent key is not an empty roster.
    let stage = super::get_opt_str_strict(args, "stage")?;
    let participants = args
        .get("participants")
        .filter(|value| !value.is_null())
        .map(|_| super::get_str_array(args, "participants"))
        .transpose()?;
    match (stage, participants) {
        (None, Some(_)) => {
            return Err(anyhow!(
                "usage: \"participants\" is a stage's roster, so the call must name the \
                 \"stage\" too — hint: pass both, or drop \"participants\""
            ));
        }
        (Some(_), None) => {
            return Err(anyhow!(
                "usage: a stage needs its participants — hint: pass \"participants\" with the \
                 stage's participants, or an empty list to return it to the default"
            ));
        }
        (Some(stage), Some(participants)) => {
            let stage = crate::workspace_profile::stage_named(stage).ok_or_else(|| {
                anyhow!(
                    "usage: '{}' is not a stage — pass analysis, planning, in_development, \
                     verification or in_sanitation",
                    stage.trim()
                )
            })?;
            if participants.is_empty() {
                stored.stages.remove(stage.as_ref());
            } else {
                stored
                    .stages
                    .insert(stage.as_ref().to_string(), participants);
            }
        }
        (None, None) => {}
    }
    if let Some(threshold) = super::get_opt_u64(args, "pass_threshold")? {
        stored.pass_threshold = Some(u8::try_from(threshold).map_err(|_| {
            anyhow!("usage: the pass threshold must be a whole number between 0 and 10")
        })?);
    }
    if args
        .get("require_all_checks")
        .is_some_and(|value| !value.is_null())
    {
        stored.require_all_checks = Some(super::get_bool(args, "require_all_checks", true)?);
    }
    stored
        .validate()
        .map_err(|e| anyhow!("usage: {e} — the profile was not changed"))?;
    let json = (!stored.is_empty()).then(|| stored.to_json());
    crate::workspace::store()
        .set_profile(&ws.name, json.as_deref())
        .await?;
    // An unconfirmed workspace's pipeline is not waiting for the next round — it is not running at
    // all, and this edit is not what lifts that.
    Ok(if ws.profile_confirmed {
        format!(
            "Workspace '{}' profile updated — it takes effect from the next stage round; a round \
             already running finishes on the settings it started with.",
            ws.name
        )
    } else {
        format!(
            "Workspace '{}' profile updated — its pipeline still waits for the profile to be \
             confirmed, and runs from the next stage round once it is.",
            ws.name
        )
    })
}

/// Return every profile field to the shipped default, keeping the confirmation.
pub(crate) async fn reset_profile(name: &str) -> anyhow::Result<String> {
    let ws = managed_workspace(name).await?;
    crate::workspace::store()
        .set_profile(&ws.name, None)
        .await?;
    Ok(if ws.profile_confirmed {
        format!("Workspace '{}' profile reset to the default.", ws.name)
    } else {
        format!(
            "Workspace '{}' profile reset to the default — its pipeline still waits for the \
             profile to be confirmed.",
            ws.name
        )
    })
}

/// Delete a managed workspace: its row, its jobs and its research runs go; the project on disk
/// is untouched, which the answer names so the admin knows the folder can be added back.
///
/// The active project afterwards is the store's own answer — the deletion resolves the admin's
/// selection inside its transaction — so the reply reports where the product stands rather than
/// a derivation of it.
async fn delete_workspace(name: &str) -> anyhow::Result<String> {
    let ws = managed_workspace(name).await?;
    let active = match crate::workspace::store().delete(&ws.name).await? {
        Some(next) => format!("Your active project is now '{next}'."),
        None => "You are back on your personal workspace.".to_string(),
    };
    Ok(format!(
        "Deleted managed workspace '{name}'. Its jobs and its research runs are gone with it. \
         {folder_note} The tickets filed against it stay in the board but are no longer listed. \
         {active}",
        name = ws.name,
        folder_note = folder_note(&ws.path)
    ))
}

/// What a deleted workspace left on disk, as far as the folder itself can prove: a folder removed
/// by hand is not reported as still there, and one that is no repository has none to name.
fn folder_note(path: &str) -> String {
    let folder = std::path::Path::new(path);
    if !folder.exists() {
        format!("The project folder at {path} is gone.")
    } else if crate::git::commands::is_git_repo(folder) {
        format!("The project folder at {path} (git repository included) stays.")
    } else {
        format!("The project folder at {path} stays.")
    }
}

/// Turn the background maintainer on or off for one workspace.
///
/// Both effects are the store's own write: enabling clears the debounce so the maintainer runs
/// on the next poll, and disabling cancels a maintainer agent already running there — so the
/// answer reports the state, never a second implementation of either.
async fn set_maintenance(args: &serde_json::Value) -> anyhow::Result<String> {
    // The argument is judged before a store is read, so a bad call touches nothing.
    let Some(on) = super::get_opt_bool(args, "maintenance")? else {
        return Err(anyhow!(
            "usage: missing required argument \"maintenance\" (expected a boolean) — hint: pass \
             true to run the background maintainer or false to stop it"
        ));
    };
    let ws = managed_workspace(super::get_str(args, "workspace")?).await?;
    crate::workspace::store()
        .set_maintenance_enabled(&ws.name, on)
        .await?;
    Ok(if on {
        format!(
            "Maintenance is now on for workspace '{}'. The debounce is cleared, so the maintainer \
             runs on the next poll.",
            ws.name
        )
    } else {
        format!(
            "Maintenance is now off for workspace '{}'. Any maintainer agent running there was \
             cancelled.",
            ws.name
        )
    })
}

/// Re-analyse a workspace: the store sets it back to `analyzing`, clears the per-role contexts
/// and spawns a fresh discovery. Nothing refuses it — reading is outside the freeze and the
/// folder's hold — and the answer says the pipeline keeps running beside the pass and that the
/// owner's pause is untouched.
async fn rediscover(name: &str) -> anyhow::Result<String> {
    let ws = managed_workspace(name).await?;
    crate::workspace::store().rediscover(&ws.name).await?;
    Ok(format!(
        "Discovery restarted for workspace '{}': its per-role contexts were cleared. The \
         pipeline keeps running beside the pass; its pause state is untouched.",
        ws.name
    ))
}

/// Re-detect only the project's commands: the store leaves the status, the pause and the
/// per-role contexts alone, which is the whole difference from [`rediscover`].
async fn rediscover_diagnostics(name: &str) -> anyhow::Result<String> {
    let ws = managed_workspace(name).await?;
    crate::workspace::store()
        .rediscover_diagnostics(&ws.name)
        .await?;
    Ok(format!(
        "Diagnostics discovery restarted for workspace '{}': its stored commands were cleared and \
         are being re-detected — its status, its pause and its per-role contexts are untouched.",
        ws.name
    ))
}

/// Apply one given command field over the stored one: absent keeps it, an empty string clears
/// it, a value sets it.
fn apply_command(slot: &mut Option<String>, given: Option<&str>) {
    if let Some(value) = given {
        *slot = crate::util::none_if_empty(value);
    }
}

/// One line per diagnostics command, an absent one spelled "not discovered", indented under
/// whatever header the caller printed.
fn write_commands(out: &mut String, commands: &crate::DiagnosticsCommands, indent: &str) {
    for (label, command) in commands.commands() {
        let _ = writeln!(
            out,
            "{indent}- {label}: {}",
            command.unwrap_or("not discovered")
        );
    }
}

/// Write the project's diagnostics commands by hand.
///
/// The stored blob is read first and the given fields applied over it, so a partial call can
/// never blank the commands it did not name. A call naming no command field at all is refused
/// before the store: it would re-write the identical blob and bump `diagnostics_generation`,
/// cancelling an in-flight discovery for nothing. The store bumps that generation on a real write,
/// which is what invalidates a discovery already in flight; the answer says so, because a
/// hand-written command would otherwise seem to vanish when the stale discovery landed.
async fn set_diagnostics(args: &serde_json::Value) -> anyhow::Result<String> {
    let format = super::get_opt_str_strict(args, "format")?;
    let format_check = super::get_opt_str_strict(args, "format_check")?;
    let lint = super::get_opt_str_strict(args, "lint")?;
    let lint_fix = super::get_opt_str_strict(args, "lint_fix")?;
    let type_check = super::get_opt_str_strict(args, "type_check")?;
    let build = super::get_opt_str_strict(args, "build")?;
    let unit_test = super::get_opt_str_strict(args, "unit_test")?;
    anyhow::ensure!(
        [
            format,
            format_check,
            lint,
            lint_fix,
            type_check,
            build,
            unit_test
        ]
        .iter()
        .any(Option::is_some),
        "usage: name at least one command field — format, format_check, lint, lint_fix, \
         type_check, build, unit_test (an empty string clears one)"
    );

    let ws = managed_workspace(super::get_str(args, "workspace")?).await?;
    let mut commands = crate::workspace::store()
        .get_diagnostics(&ws.name)
        .await?
        .unwrap_or_default();
    apply_command(&mut commands.format, format);
    apply_command(&mut commands.format_check, format_check);
    apply_command(&mut commands.lint, lint);
    apply_command(&mut commands.lint_fix, lint_fix);
    apply_command(&mut commands.type_check, type_check);
    apply_command(&mut commands.build, build);
    apply_command(&mut commands.unit_test, unit_test);
    crate::workspace::store()
        .set_diagnostics(&ws.name, &commands)
        .await?;
    let mut out = format!("Diagnostics for workspace '{}':\n", ws.name);
    write_commands(&mut out, &commands, "");
    let _ = write!(
        out,
        "\nWriting invalidated any diagnostics discovery already in flight, so its stale result \
         cannot overwrite these."
    );
    Ok(out)
}

/// Write the project rules.
///
/// The store caps them at [`MAX_WORKSPACE_NOTES_CHARS`](crate::workspace::MAX_WORKSPACE_NOTES_CHARS)
/// characters, so a longer input is reported as truncated rather than silently shortened.
async fn set_rules(args: &serde_json::Value) -> anyhow::Result<String> {
    let rules = super::get_str(args, "rules")?;
    let ws = managed_workspace(super::get_str(args, "workspace")?).await?;
    crate::workspace::store().set_notes(&ws.name, rules).await?;
    // The stored length is the store's own truncation, read back through the same function it
    // writes with, so the answer and the row cannot disagree.
    let stored = crate::workspace::truncate_workspace_notes(rules);
    let mut out = if stored.is_empty() {
        format!("The rules of workspace '{}' are now empty.", ws.name)
    } else {
        format!(
            "The rules of workspace '{}' are now {} characters long.",
            ws.name,
            stored.chars().count()
        )
    };
    if rules.chars().count() > stored.chars().count() {
        let _ = write!(
            out,
            " They were truncated to {} characters.",
            crate::workspace::MAX_WORKSPACE_NOTES_CHARS
        );
    }
    Ok(out)
}

/// Make one managed workspace the admin's active project — the same stored selection the
/// desktop's own switcher writes, so it takes effect without a restart: the dashboard reads the
/// selection back on a users change.
async fn use_workspace(name: &str) -> anyhow::Result<String> {
    let ws = managed_workspace(name).await?;
    crate::users::store()
        .set_selected_workspace(crate::users::ADMIN_USER_NAME, Some(&ws.name))
        .await?;
    Ok(format!(
        "'{}' is now your active project at {}. It takes effect without a restart — the \
         dashboard reads the stored selection back on a users change.",
        ws.name, ws.path
    ))
}

/// What a pause/resume request does to one workspace — the rule both forms follow, so the
/// named and the fleet-wide answer can never disagree about which rows are written. A pause is
/// written for any status: every workspace has work a freeze holds, and one whose first discovery
/// has not run is held out of the pipeline by the `pending` status until it has.
enum PipelineChange {
    /// Already in the requested state: reported as such, never written again.
    Already,
    Write,
}

fn pipeline_change(ws: &Workspace, paused: bool) -> PipelineChange {
    if ws.paused == paused {
        PipelineChange::Already
    } else {
        PipelineChange::Write
    }
}

/// Pause or resume the named workspace's pipeline, or every managed workspace at once when no
/// name was given.
async fn pause_or_resume(name: Option<&str>, paused: bool) -> anyhow::Result<ToolAnswer> {
    match name {
        Some(name) => Ok(ToolAnswer::done(
            set_one_pipeline_paused(name, paused).await?,
        )),
        None => set_all_pipelines_paused(paused).await,
    }
}

async fn set_one_pipeline_paused(name: &str, paused: bool) -> anyhow::Result<String> {
    let ws = managed_workspace(name).await?;
    match pipeline_change(&ws, paused) {
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

/// Pause or resume every workspace of `workspaces` where the write is honest — see
/// [`pipeline_change`] — one line per workspace, or [`NO_WORKSPACE`] for an empty set.
///
/// Takes the store the writes go to and the set they cover, so the fleet-wide form is not
/// hardwired to the process-global store: a fleet-wide write can then never reach a row a test
/// created.
///
/// PARTIAL exactly when the requested state was not reached for a workspace: a write that failed.
/// A workspace already in the requested state leaves the operation whole.
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
            PipelineChange::Already if paused => format!("- {}: already paused", ws.name),
            PipelineChange::Already => format!("- {}: not paused", ws.name),
            PipelineChange::Write => match store.set_paused(&ws.name, paused).await {
                Ok(()) if paused => format!("- {}: paused", ws.name),
                Ok(()) => format!("- {}: resumed", ws.name),
                // Reported like every other outcome: the rows written before a failure are real, so
                // the report must cover each workspace.
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

/// Every managed workspace at once: the global store and the set it must cover.
async fn set_all_pipelines_paused(paused: bool) -> anyhow::Result<ToolAnswer> {
    let workspaces = crate::users::managed_workspaces().await?;
    set_pipelines_paused(crate::workspace::store(), &workspaces, paused).await
}

/// List the live tickets of the managed workspaces, optionally narrowed to one workspace
/// and/or one phase.
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
        // The first line is always shown — the notice must never read "the first 0 of N
        // tickets" — so a line larger than the whole budget is cut.
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

/// Read one ticket by its full id. The default is the shortened form for every ticket,
/// whoever filed it.
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
    // The newest comments come back in full, so a ticket carrying a huge one is still bounded
    // here — `full: true` is the unbounded read.
    if shortened.len() <= OUTPUT_BYTE_LIMIT {
        return Ok(shortened);
    }
    Ok(format!(
        "{}…\n(shortened view cut short — pass `full: true` for the ticket whole and unclipped.)",
        crate::util::truncate_bytes(&shortened, OUTPUT_BYTE_LIMIT)
    ))
}

/// Leave a clarification note on a ticket. Refused when the ticket does not exist, so a note
/// is never stored for a missing ticket.
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

    /// Every managed workspace at once: each row is written, whatever its status — a pause is the
    /// pipeline's switch, and a workspace whose first discovery has not run is held out of the
    /// pipeline by its `pending` status until it has. Driven against the test's own store, so the
    /// fleet-wide write can only reach rows this test created.
    #[tokio::test]
    async fn pause_every_managed_workspace_writes_every_row() {
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
        store
            .add("ws_fleet_pending", pending.to_str().expect("utf-8 path"))
            .await
            .expect("add ws_fleet_pending");
        let workspaces = store.list().await.expect("list");

        let reply = set_pipelines_paused(&store, &workspaces, true)
            .await
            .expect("pause all");
        assert!(
            reply.text.contains("- ws_fleet: paused")
                && reply.text.contains("- ws_fleet_pending: paused"),
            "every row is written, whatever its status: {}",
            reply.text
        );
        assert!(!reply.partial, "nothing was left undone: {}", reply.text);
        for name in ["ws_fleet", "ws_fleet_pending"] {
            assert!(
                store
                    .get_by_name(name)
                    .await
                    .expect("read")
                    .expect("exists")
                    .paused,
                "{name} is paused"
            );
        }

        // The fleet resume lifts every row back.
        let workspaces = store.list().await.expect("list");
        let reply = set_pipelines_paused(&store, &workspaces, false)
            .await
            .expect("resume all");
        assert!(
            reply.text.contains("- ws_fleet: resumed")
                && reply.text.contains("- ws_fleet_pending: resumed"),
            "got: {}",
            reply.text
        );
        assert!(
            !reply.partial,
            "a resume is never left undone: {}",
            reply.text
        );
        for name in ["ws_fleet", "ws_fleet_pending"] {
            assert!(
                !store
                    .get_by_name(name)
                    .await
                    .expect("read")
                    .expect("exists")
                    .paused,
                "{name} is resumed"
            );
        }

        // Nothing managed: the same plain wording `list` uses.
        let reply = set_pipelines_paused(&store, &[], true)
            .await
            .expect("empty set");
        assert_eq!(reply.text, NO_WORKSPACE);
    }

    /// The unnamed form of `pause` / `resume` is the fleet-wide write, so it may only be
    /// reached by a call that names nothing else: any other *declared* field — whether it names
    /// a place (`name` and `path` belong to `add`, a ticket id to the ticket actions) or the
    /// ticket id the product's argument repair makes of a mistaken `id` — is a usage error. A key
    /// no schema declares is refused earlier, for every action. The read actions take the same
    /// address and ignore a declared field they do not use.
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

        // A key no schema declares never reaches the address rule at all: the one unknown-field
        // check every action goes through refuses it first, naming the reference.
        for call in [
            json!({ "action": "pause", "workspace_name": "ws_address" }),
            json!({ "action": "list", "workspace": "ws_address", "folder": "/tmp/ws_address" }),
        ] {
            let err = tool
                .execute(&ws, call.clone())
                .await
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("unknown field") && err.contains("reference"),
                "for {call}: {err}"
            );
        }

        // A wrong-typed address is a usage error too, and it names the field the caller wrote.
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

        // A stray *declared* field on a read action is ignored rather than refused: the answer is
        // the listing the call gives without it, neither widened nor dropped. (Narrowed to this
        // one workspace, so a test running beside this one cannot change what is listed.)
        for args in [
            json!({ "action": "list", "workspace": "ws_address" }),
            json!({ "action": "list", "workspace": "ws_address", "phase": "done" }),
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

    /// `ticket` refuses an id whose workspace is not managed and an id with no such ticket.
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

    /// `board` refuses the archived phase, lists finished tickets of managed workspaces and
    /// never a ticket of one that is not managed.
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
        // The default listing keeps finished tickets — deliberately unlike the manager's own
        // listing, which hides them.
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

                // A personal selection is not a managed workspace, so nothing is active.
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

    /// A named listing is answered about that workspace: it narrows to the one named, and a name
    /// that is not managed is the not-found refusal naming it rather than a report on the fleet.
    #[tokio::test]
    async fn a_named_listing_answers_about_that_workspace() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        crate::users::test_util::with_test_workspaces(
            &[("/tmp/ws_named", "ws_named"), ("/tmp/ws_other", "ws_other")],
            async {
                let listing = list_workspaces(Some("ws_named"))
                    .await
                    .expect("a named listing");
                assert!(listing.contains("- ws_named ("), "got: {listing}");
                assert!(
                    !listing.contains("- ws_other ("),
                    "a named listing narrows to the one named: {listing}"
                );

                let missing = list_workspaces(Some("ws_absent"))
                    .await
                    .expect_err("an unmanaged name")
                    .to_string();
                assert!(missing.contains("ws_absent"), "got: {missing}");
                assert!(
                    !missing.contains("No managed workspaces exist."),
                    "a named listing answers about the name, not the fleet: {missing}"
                );
            },
        )
        .await;
    }

    /// `reference` hands over the settings reference verbatim.
    #[tokio::test]
    async fn reference_is_the_settings_reference_verbatim() {
        let ws = crate::workspace::test_ws("/tmp/ws_reference");
        let reply = WorkspaceTool
            .execute(&ws, json!({ "action": "reference" }))
            .await
            .expect("reference");
        assert_eq!(reply, crate::settings::reference());
    }

    /// `show` is the one read that carries a workspace's own settings: its address and state,
    /// the rules an agent there is told, and the commands verification will run.
    #[tokio::test]
    async fn show_reports_a_workspaces_settings() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let tool = WorkspaceTool;
        let ws = crate::workspace::test_ws("/tmp/ws_show");
        crate::util::test::create_test_workspace("/tmp/ws_show", "ws_show").await;
        crate::workspace::store()
            .set_maintenance_enabled("ws_show", true)
            .await
            .expect("maintenance on");
        crate::workspace::store()
            .set_notes("ws_show", "be nice to the fixtures")
            .await
            .expect("rules");
        crate::workspace::store()
            .set_diagnostics(
                "ws_show",
                &crate::DiagnosticsCommands {
                    format: Some("cargo fmt".to_string()),
                    ..Default::default()
                },
            )
            .await
            .expect("diagnostics");

        let reply = tool
            .execute(&ws, tool_args("show", Some("ws_show")))
            .await
            .expect("show");
        assert!(reply.starts_with("Workspace 'ws_show':"), "got: {reply}");
        for expected in [
            "- path: /tmp/ws_show",
            "- status: pending",
            "- pipeline: nothing to claim while the workspace is pending",
            "- maintenance: on",
            "- active project: no",
            "- rules:\n  be nice to the fixtures",
            "  - format: cargo fmt",
            "  - build: not discovered",
        ] {
            assert!(reply.contains(expected), "missing {expected:?} in: {reply}");
        }

        // No rules and no commands yet: both are spelled out rather than left blank.
        crate::util::test::create_test_workspace("/tmp/ws_show_bare", "ws_show_bare").await;
        let reply = tool
            .execute(&ws, tool_args("show", Some("ws_show_bare")))
            .await
            .expect("show bare");
        assert!(reply.contains("- rules: none"), "got: {reply}");
        assert!(
            reply.contains("- diagnostics: not discovered"),
            "got: {reply}"
        );
        assert!(reply.contains("- maintenance: off"), "got: {reply}");
    }

    /// The store's 4000-character cap is reported, never applied silently.
    #[tokio::test]
    async fn set_rules_reports_the_stores_truncation() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let tool = WorkspaceTool;
        let ws = crate::workspace::test_ws("/tmp/ws_rules");
        crate::util::test::create_test_workspace("/tmp/ws_rules", "ws_rules").await;
        let cap = crate::workspace::MAX_WORKSPACE_NOTES_CHARS;

        let short = tool
            .execute(
                &ws,
                json!({ "action": "set_rules", "workspace": "ws_rules", "rules": "be nice" }),
            )
            .await
            .expect("set rules");
        assert!(
            short.contains("now 7 characters long") && !short.contains("truncated"),
            "got: {short}"
        );

        let long = "r".repeat(cap + 10);
        let reply = tool
            .execute(
                &ws,
                json!({ "action": "set_rules", "workspace": "ws_rules", "rules": long }),
            )
            .await
            .expect("set long rules");
        assert!(
            reply.contains(&format!("now {cap} characters long"))
                && reply.contains(&format!("truncated to {cap} characters")),
            "got: {reply}"
        );
        let stored = crate::workspace::get_by_name("ws_rules")
            .await
            .expect("read")
            .expect("exists");
        assert_eq!(stored.notes.chars().count(), cap);
    }

    /// An absent command keeps what the row holds and an empty string clears it, so a partial
    /// `set_diagnostics` can never blank the commands it did not name.
    #[tokio::test]
    async fn set_diagnostics_keeps_an_omitted_field_and_clears_an_empty_one() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let tool = WorkspaceTool;
        let ws = crate::workspace::test_ws("/tmp/ws_cmds");
        crate::util::test::create_test_workspace("/tmp/ws_cmds", "ws_cmds").await;

        let reply = tool
            .execute(
                &ws,
                json!({
                    "action": "set_diagnostics",
                    "workspace": "ws_cmds",
                    "format": "cargo fmt",
                    "unit_test": "cargo test",
                }),
            )
            .await
            .expect("set diagnostics");
        assert!(
            reply.contains("- format: cargo fmt")
                && reply.contains("- unit-test: cargo test")
                && reply.contains("- build: not discovered"),
            "got: {reply}"
        );
        assert!(reply.contains("invalidated"), "got: {reply}");

        // `format` cleared by the empty string, `unit_test` kept by its absence.
        let reply = tool
            .execute(
                &ws,
                json!({ "action": "set_diagnostics", "workspace": "ws_cmds", "format": "" }),
            )
            .await
            .expect("clear format");
        assert!(reply.contains("- format: not discovered"), "got: {reply}");
        let stored = crate::workspace::store()
            .get_diagnostics("ws_cmds")
            .await
            .expect("read")
            .expect("diagnostics");
        assert_eq!(stored.format, None);
        assert_eq!(stored.unit_test.as_deref(), Some("cargo test"));
    }

    /// `maintenance` judges its switch before it reads a workspace, and reports the state the
    /// store itself wrote.
    #[tokio::test]
    async fn maintenance_needs_its_switch_before_a_workspace() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/ws_maint");
        crate::util::test::create_test_workspace("/tmp/ws_maint", "ws_maint").await;

        let err = WorkspaceTool
            .execute(
                &ws,
                json!({ "action": "maintenance", "workspace": "ws_maint_ghost" }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("missing required argument \"maintenance\""),
            "a bad call must not reach the workspace lookup: {err}"
        );

        let reply = WorkspaceTool
            .execute(
                &ws,
                json!({ "action": "maintenance", "workspace": "ws_maint", "maintenance": true }),
            )
            .await
            .expect("maintenance on");
        assert!(
            reply.contains("Maintenance is now on for workspace 'ws_maint'")
                && reply.contains("next poll"),
            "got: {reply}"
        );
        assert!(
            crate::workspace::get_by_name("ws_maint")
                .await
                .expect("read")
                .expect("exists")
                .maintenance_enabled
        );
    }

    /// A `set_diagnostics` naming no command field is refused before the store: re-writing the
    /// identical blob would bump the generation and cancel an in-flight discovery for nothing.
    /// The refusal comes before the workspace lookup, so nothing is read at all.
    #[tokio::test]
    async fn set_diagnostics_refuses_a_call_that_names_no_command() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/ws_cmds_none");
        crate::util::test::create_test_workspace("/tmp/ws_cmds_none", "ws_cmds_none").await;

        for address in ["ws_cmds_none", "ws_cmds_ghost"] {
            let err = WorkspaceTool
                .execute(
                    &ws,
                    json!({ "action": "set_diagnostics", "workspace": address }),
                )
                .await
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("at least one command field") && err.contains("unit_test"),
                "got: {err}"
            );
        }
        assert!(
            crate::workspace::store()
                .get_diagnostics("ws_cmds_none")
                .await
                .expect("read")
                .is_none(),
            "a refused call must not write anything"
        );
    }

    /// What a deleted workspace left on disk is read from the folder itself: a folder removed by
    /// hand is not reported as still there, and only a repository names one.
    #[test]
    fn the_delete_answer_names_a_folder_only_as_it_really_is() {
        let folder = tempfile::TempDir::new().expect("temp dir");
        let path = folder.path().to_str().expect("utf-8 path");
        assert_eq!(
            folder_note(path),
            format!("The project folder at {path} stays.")
        );
        std::fs::create_dir(folder.path().join(".git")).expect("make a repository");
        assert_eq!(
            folder_note(path),
            format!("The project folder at {path} (git repository included) stays.")
        );
        std::fs::remove_dir_all(folder.path()).expect("remove the folder by hand");
        assert_eq!(
            folder_note(path),
            format!("The project folder at {path} is gone.")
        );
    }

    /// `delete` removes the row and names what it left behind, so the admin knows the folder can
    /// be added back and the tickets are still there.
    #[tokio::test]
    #[serial_test::serial(gui_admin_workspace)] // writes the shared seeded admin row
    async fn delete_names_what_remains_on_disk() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        crate::users::test_util::with_admin_workspace_restored(async {
            let tool = WorkspaceTool;
            let ws = crate::workspace::test_ws("/tmp/ws_del_gone");
            let gone =
                crate::util::test::create_test_workspace("/tmp/ws_del_gone", "ws_del_gone").await;
            crate::util::test::create_test_workspace("/tmp/ws_del_keep", "ws_del_keep").await;
            let ticket = crate::util::test::make_ticket(
                board_store(),
                &gone,
                "delete leaves me",
                TicketPhase::Queued,
            )
            .await;
            crate::users::store()
                .set_selected_workspace(crate::users::ADMIN_USER_NAME, Some("ws_del_gone"))
                .await
                .expect("select");

            let reply = tool
                .execute(&ws, tool_args("delete", Some("ws_del_gone")))
                .await
                .expect("delete");
            assert!(
                reply.contains("Deleted managed workspace 'ws_del_gone'"),
                "got: {reply}"
            );
            assert!(reply.contains("/tmp/ws_del_gone"), "got: {reply}");
            // The answer names a repository only when the folder really has one — a deleted
            // project that is no repository has none to keep.
            assert_eq!(
                reply.contains("git repository"),
                crate::git::commands::is_git_repo(std::path::Path::new("/tmp/ws_del_gone")),
                "the repository half must match what is on disk: {reply}"
            );
            assert!(reply.contains("tickets"), "got: {reply}");
            assert!(
                reply.contains("Your active project is now '"),
                "another managed workspace survives, so the selection must move to it: {reply}"
            );
            assert!(
                crate::workspace::get_by_name("ws_del_gone")
                    .await
                    .expect("read")
                    .is_none(),
                "the row is gone"
            );
            assert!(
                board_store()
                    .get_ticket(&ticket)
                    .await
                    .expect("read")
                    .is_some(),
                "deleting a workspace must not delete its tickets"
            );
            assert_ne!(
                crate::users::resolve_selected_workspace_name(crate::users::ADMIN_USER_NAME)
                    .await
                    .as_deref(),
                Some("ws_del_gone"),
                "the active project must not stay on the deleted workspace"
            );

            // A second delete is refused with the live set named.
            let err = tool
                .execute(&ws, tool_args("delete", Some("ws_del_gone")))
                .await
                .unwrap_err()
                .to_string();
            assert!(err.contains("No managed workspace exists"), "got: {err}");
        })
        .await;
    }

    /// `use` writes the admin's stored selection — the same row the desktop's switcher writes —
    /// so it takes effect without a restart.
    #[tokio::test]
    #[serial_test::serial(gui_admin_workspace)] // writes the shared seeded admin row
    async fn use_makes_a_workspace_the_active_project() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        crate::users::test_util::with_test_workspaces(
            &[("/tmp/ws_use", "ws_use")],
            crate::users::test_util::with_admin_workspace_restored(async {
                let ws = crate::workspace::test_ws("/tmp/ws_use");
                let reply = WorkspaceTool
                    .execute(&ws, tool_args("use", Some("ws_use")))
                    .await
                    .expect("use");
                assert!(
                    reply.contains("'ws_use' is now your active project")
                        && reply.contains("without a restart"),
                    "got: {reply}"
                );
                assert_eq!(
                    crate::users::resolve_selected_workspace_name(crate::users::ADMIN_USER_NAME)
                        .await
                        .as_deref(),
                    Some("ws_use")
                );

                // A name that is not managed is refused, and the selection is left alone.
                let err = WorkspaceTool
                    .execute(&ws, tool_args("use", Some("ws_ghost")))
                    .await
                    .unwrap_err()
                    .to_string();
                assert!(err.contains("No managed workspace exists"), "got: {err}");
                assert_eq!(
                    crate::users::resolve_selected_workspace_name(crate::users::ADMIN_USER_NAME)
                        .await
                        .as_deref(),
                    Some("ws_use")
                );
            }),
        )
        .await;
    }

    /// The two re-analysis actions report what they started and what they touched: a full
    /// rediscovery pauses the pipeline, while a diagnostics one leaves the status, the pause and the
    /// per-role contexts alone and clears the stored commands.
    #[tokio::test]
    #[serial_test::serial(gui_admin_workspace)]
    async fn the_reanalysis_actions_report_what_they_started() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        crate::users::test_util::with_test_workspaces(
            &[("/tmp/ws_reanalyse", "ws_reanalyse")],
            async {
                let ws = crate::workspace::test_ws("/tmp/ws_reanalyse");
                let commands = crate::DiagnosticsCommands {
                    format: Some("cargo fmt".to_string()),
                    ..crate::DiagnosticsCommands::default()
                };
                crate::workspace::store()
                    .set_diagnostics("ws_reanalyse", &commands)
                    .await
                    .expect("seed the commands");

                let reply = WorkspaceTool
                    .execute(
                        &ws,
                        tool_args("rediscover_diagnostics", Some("ws_reanalyse")),
                    )
                    .await
                    .expect("rediscover_diagnostics");
                assert!(
                    reply.contains("stored commands were cleared")
                        && reply
                            .contains("status, its pause and its per-role contexts are untouched"),
                    "got: {reply}"
                );
                assert!(
                    crate::workspace::store()
                        .get_diagnostics("ws_reanalyse")
                        .await
                        .expect("read")
                        .is_none(),
                    "the stored commands are cleared while they are re-detected"
                );

                // Reading is outside the freeze: a re-analysis of a paused workspace runs, and the
                // pause it does not touch stays in force.
                crate::workspace::store()
                    .set_paused("ws_reanalyse", true)
                    .await
                    .expect("freeze the workspace");
                WorkspaceTool
                    .execute(&ws, tool_args("rediscover", Some("ws_reanalyse")))
                    .await
                    .expect(
                        "a paused workspace is still re-analysed: reading is outside the freeze",
                    );
                WorkspaceTool
                    .execute(
                        &ws,
                        tool_args("rediscover_diagnostics", Some("ws_reanalyse")),
                    )
                    .await
                    .expect("diagnostics discovery is reading too");
                let row = crate::workspace::store()
                    .get_by_name("ws_reanalyse")
                    .await
                    .expect("read")
                    .expect("the workspace is still there");
                assert!(
                    row.paused,
                    "a re-analysis leaves the owner's pause in force"
                );
            },
        )
        .await;
    }

    /// The schema's action list and the dispatch are one surface: the enum is what the model is
    /// offered, so an entry without a dispatch arm would be a dead end. The match has one string
    /// arm per action, and this scan is what keeps the two from drifting.
    #[test]
    fn every_action_the_schema_offers_is_dispatched() {
        let source = include_str!("workspace.rs");
        for action in ACTIONS {
            assert!(
                source.contains(&format!("\"{action}\" =>")),
                "'{action}' is offered by the schema but has no dispatch arm"
            );
        }
    }

    /// The four profile actions ride the tool's own dispatch: a missing arm answers "unknown
    /// action" instead of reaching the implementation.
    #[tokio::test]
    async fn the_profile_actions_are_reachable_through_the_tool_dispatch() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/ws_profile_dispatch");
        for action in ["profile", "confirm_profile", "set_profile", "reset_profile"] {
            let error = WorkspaceTool
                .execute(&ws, tool_args(action, Some("no_such_workspace")))
                .await
                .expect_err("an unknown workspace is refused");
            assert!(
                !error.to_string().contains("unknown action"),
                "'{action}' must reach the profile implementation, got: {error}"
            );
        }
    }

    /// An action and a field this tool does not know both teach where the whole surface is: the
    /// `reference` action, which is one text shared with `mahbot config reference`.
    #[tokio::test]
    async fn an_unknown_action_and_field_name_the_reference() {
        let ws = crate::workspace::test_ws("/tmp/ws_unknown");
        let err = WorkspaceTool
            .execute(&ws, json!({ "action": "explode" }))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("unknown action 'explode'") && err.contains("`reference` action"),
            "got: {err}"
        );

        let err = WorkspaceTool
            .execute(
                &ws,
                json!({ "action": "list", "folder": "/tmp/ws_unknown" }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("unknown field 'folder'") && err.contains("reference"),
            "got: {err}"
        );
    }

    /// The profile is a workspace setting the owner's assistant reads, confirms and edits with no
    /// window open.
    #[tokio::test]
    async fn profile_actions_read_confirm_edit_and_reset() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/ws_profile");
        let tool = WorkspaceTool;
        crate::util::test::create_test_workspace("/tmp/ws_profile", "ws_profile").await;
        crate::workspace::store()
            .set_status("ws_profile", &WorkspaceStatus::Ready)
            .await
            .expect("mark ready");
        crate::util::test::unconfirm_test_workspace("ws_profile").await;

        let listing = tool
            .execute(&ws, tool_args("list", None))
            .await
            .expect("list");
        assert!(
            listing.contains("waiting for its profile to be confirmed"),
            "the listing says the workspace waits: {listing}"
        );

        let read = tool
            .execute(&ws, tool_args("profile", Some("ws_profile")))
            .await
            .expect("profile");
        assert!(
            read.contains("waiting for confirmation")
                && read.contains("`confirm_profile`")
                && read.contains("stages (what confirmation puts in force):")
                && read.contains("- analysis: analyst"),
            "the profile says how to confirm it: {read}"
        );

        let reply = tool
            .execute(&ws, tool_args("confirm_profile", Some("ws_profile")))
            .await
            .expect("confirm");
        assert!(reply.contains("confirmed"), "got: {reply}");

        let reply = tool
            .execute(
                &ws,
                json!({
                    "action": "set_profile",
                    "workspace": "ws_profile",
                    "stage": "verification",
                    "participants": ["tester"],
                    "pass_threshold": 7,
                    "require_all_checks": false,
                }),
            )
            .await
            .expect("set profile");
        assert!(reply.contains("next stage round"), "got: {reply}");
        let read = tool
            .execute(&ws, tool_args("profile", Some("ws_profile")))
            .await
            .expect("profile");
        assert!(
            read.contains("- verification: tester")
                && read.contains("pass threshold: 7/10")
                && read.contains("every check must pass: no"),
            "got: {read}"
        );

        let reply = tool
            .execute(&ws, tool_args("reset_profile", Some("ws_profile")))
            .await
            .expect("reset");
        assert!(reply.contains("reset to the default"), "got: {reply}");
        assert!(
            !reply.contains("still waits"),
            "a confirmed workspace runs on the reset profile: {reply}"
        );
        let read = tool
            .execute(&ws, tool_args("profile", Some("ws_profile")))
            .await
            .expect("profile");
        assert!(
            read.contains("- verification: reviewer, tester")
                && read.contains("pass threshold: 9/10"),
            "got: {read}"
        );
    }

    /// An edit on an unconfirmed workspace promises no next round: the pipeline is not running at
    /// all, and neither the edit nor the reset is what lifts that.
    #[tokio::test]
    async fn an_edit_on_an_unconfirmed_workspace_says_its_pipeline_still_waits() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/ws_unconfirmed_edit");
        crate::util::test::create_test_workspace("/tmp/ws_unconfirmed_edit", "ws_unconfirmed_edit")
            .await;
        crate::util::test::unconfirm_test_workspace("ws_unconfirmed_edit").await;

        let reply = WorkspaceTool
            .execute(
                &ws,
                json!({
                    "action": "set_profile",
                    "workspace": "ws_unconfirmed_edit",
                    "pass_threshold": 8,
                }),
            )
            .await
            .expect("set profile");
        assert!(
            reply.contains("still waits for the profile to be confirmed")
                && !reply.contains("takes effect from the next stage round"),
            "got: {reply}"
        );

        let reply = WorkspaceTool
            .execute(&ws, tool_args("reset_profile", Some("ws_unconfirmed_edit")))
            .await
            .expect("reset");
        assert!(
            reply.contains("reset to the default")
                && reply.contains("still waits for the profile to be confirmed"),
            "got: {reply}"
        );
    }

    /// A stage and its roster are one edit: either alone is refused, the same participant twice is
    /// not a roster, and a refused edit stores nothing.
    #[tokio::test]
    async fn a_profile_edit_names_a_stage_with_its_roster() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/ws_profile_edits");
        let tool = WorkspaceTool;
        crate::util::test::create_test_workspace("/tmp/ws_profile_edits", "ws_profile_edits").await;
        tool.execute(&ws, tool_args("confirm_profile", Some("ws_profile_edits")))
            .await
            .expect("confirm");
        tool.execute(
            &ws,
            json!({
                "action": "set_profile",
                "workspace": "ws_profile_edits",
                "stage": "verification",
                "participants": ["tester"],
            }),
        )
        .await
        .expect("set the roster");

        // A roster without its stage would silently change nothing.
        let err = tool
            .execute(
                &ws,
                json!({
                    "action": "set_profile",
                    "workspace": "ws_profile_edits",
                    "participants": ["tester"],
                }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("usage:")
                && err.contains("\"participants\"")
                && err.contains("\"stage\""),
            "got: {err}"
        );

        // A stage without its participants would read as an empty roster and wipe it.
        let err = tool
            .execute(
                &ws,
                json!({
                    "action": "set_profile",
                    "workspace": "ws_profile_edits",
                    "stage": "verification",
                }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("a stage needs its participants"), "got: {err}");

        // The same participant twice is not a roster.
        let err = tool
            .execute(
                &ws,
                json!({
                    "action": "set_profile",
                    "workspace": "ws_profile_edits",
                    "stage": "verification",
                    "participants": ["tester", "tester"],
                }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("twice"), "got: {err}");

        // A participant outside its stage is refused too.
        let err = tool
            .execute(
                &ws,
                json!({
                    "action": "set_profile",
                    "workspace": "ws_profile_edits",
                    "stage": "analysis",
                    "participants": ["reviewer"],
                }),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not do the analysis"), "got: {err}");

        // Every refusal above left the stored roster alone.
        let read = tool
            .execute(&ws, tool_args("profile", Some("ws_profile_edits")))
            .await
            .expect("profile");
        assert!(
            read.contains("- verification: tester") && read.contains("- analysis: analyst"),
            "a refused edit changes nothing: {read}"
        );
    }

    /// An explicitly empty participant list is how a stage returns to the default — the one edit
    /// that is not refused for naming nobody.
    #[tokio::test]
    async fn an_empty_roster_returns_a_stage_to_the_default() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/ws_profile_default_roster");
        let tool = WorkspaceTool;
        crate::util::test::create_test_workspace(
            "/tmp/ws_profile_default_roster",
            "ws_profile_default_roster",
        )
        .await;
        tool.execute(
            &ws,
            tool_args("confirm_profile", Some("ws_profile_default_roster")),
        )
        .await
        .expect("confirm");
        tool.execute(
            &ws,
            json!({
                "action": "set_profile",
                "workspace": "ws_profile_default_roster",
                "stage": "verification",
                "participants": ["tester"],
            }),
        )
        .await
        .expect("set the roster");

        tool.execute(
            &ws,
            json!({
                "action": "set_profile",
                "workspace": "ws_profile_default_roster",
                "stage": "verification",
                "participants": [],
            }),
        )
        .await
        .expect("clear the roster");
        let read = tool
            .execute(&ws, tool_args("profile", Some("ws_profile_default_roster")))
            .await
            .expect("profile");
        assert!(
            read.contains("- verification: reviewer, tester"),
            "an empty list returns the stage to the default: {read}"
        );
    }

    /// The `config` tool is the owner's other way into the profile: the same four actions, the same
    /// answers, one implementation.
    #[tokio::test]
    async fn the_config_tool_reads_confirms_edits_and_resets_the_profile() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/ws_config_profile");
        let tool = crate::tools::mahbot_config::MahbotConfigTool;
        crate::util::test::create_test_workspace("/tmp/ws_config_profile", "ws_config_profile")
            .await;
        crate::util::test::unconfirm_test_workspace("ws_config_profile").await;

        let read = tool
            .execute(
                &ws,
                json!({ "action": "profile", "workspace": "ws_config_profile" }),
            )
            .await
            .expect("profile");
        assert!(read.contains("waiting for confirmation"), "got: {read}");

        let reply = tool
            .execute(
                &ws,
                json!({ "action": "confirm_profile", "workspace": "ws_config_profile" }),
            )
            .await
            .expect("confirm");
        assert!(reply.contains("confirmed"), "got: {reply}");

        let reply = tool
            .execute(
                &ws,
                json!({
                    "action": "set_profile",
                    "workspace": "ws_config_profile",
                    "pass_threshold": 6,
                }),
            )
            .await
            .expect("set");
        assert!(reply.contains("next stage round"), "got: {reply}");

        let reply = tool
            .execute(
                &ws,
                json!({ "action": "reset_profile", "workspace": "ws_config_profile" }),
            )
            .await
            .expect("reset");
        assert!(reply.contains("reset to the default"), "got: {reply}");
    }

    /// Adding a workspace says the one thing the owner must do next: its pipeline does not run until
    /// the profile is confirmed, and confirming accepts the profile as shipped unless it is edited.
    #[tokio::test]
    async fn the_add_answer_says_the_pipeline_waits_for_the_profile() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/ws_added_answer");
        let folder = crate::util::test::test_root().join("ws_added_answer_folder");
        std::fs::create_dir_all(&folder).expect("create the project folder");

        let reply = WorkspaceTool
            .execute(
                &ws,
                json!({
                    "action": "add",
                    "name": "ws_added_answer",
                    "path": folder.to_str().expect("utf-8 path"),
                }),
            )
            .await
            .expect("add");
        assert!(
            reply.contains("does not run until you confirm its profile")
                && reply.contains("`confirm_profile`")
                && reply.contains("as shipped unless you edit it"),
            "got: {reply}"
        );
    }
}
