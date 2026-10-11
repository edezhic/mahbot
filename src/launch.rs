//! Launching and stopping a continuable branch: a custom role's, or a coder
//! branch's.
//!
//! Both kinds are normal durable envelopes ([`crate::Launch`]) persisted BEFORE
//! they are routed, so they survive a restart, and stopped through one path.
//! A custom role runs the administrator's authored role; a coder branch runs a
//! normal [`crate::Role::Coder`] agent on a continuation of its own conversation.

use anyhow::Result;

/// Build the durable launch envelope for a branch — the shape every launch path
/// (a tool call, a user command, an alarm) shares.
fn launch_job(
    content: String,
    workspace_name: String,
    launch: crate::Launch,
) -> crate::agent::message_router::AgentJob {
    crate::agent::message_router::AgentJob {
        content,
        workspace_name,
        user_name: crate::users::ADMIN_USER_NAME.to_string(),
        channel: "gui".to_string(),
        kind: crate::agent::message_router::MessageKind::UserMessage,
        role: launch.role,
        reply_target: None,
        pending_job_id: None,
        originating_workspace: None,
        custom_role: Some(launch),
        reply_to_custom: None,
    }
}

/// Launch `name`'s custom role with `message` as the launch message.
///
/// `branch` selects a continuable named branch; without one the record's
/// behaviour decides (one common branch, or a one-off run). `place` names a
/// registered workspace, a `personal:{user}` workspace or any folder on the
/// machine; `None` runs in the owner's personal folder, exactly as the owner's
/// Assistant does. The launch is persisted as a durable envelope BEFORE it is
/// routed, so it survives a restart. Returns a one-line confirmation for the caller.
pub(crate) async fn custom_role(
    name: &str,
    message: &str,
    branch: Option<&str>,
    place: Option<&str>,
) -> Result<String> {
    anyhow::ensure!(
        !message.trim().is_empty(),
        "usage: the launch message must not be empty"
    );
    let role = crate::custom_role::get(name)
        .await?
        .ok_or_else(|| anyhow::anyhow!("no custom role named '{name}'"))?;

    // An explicit empty/whitespace identifier counts as absent.
    let identifier = branch.map(str::trim).filter(|id| !id.is_empty());
    if let Some(id) = identifier {
        crate::custom_role::validate_identifier(id)?;
    }

    // Resolve the workspace address: `None` (or empty/whitespace) runs in the
    // owner's personal folder; a named value must resolve to a registered
    // workspace, a `personal:{user}` key or an absolute path.
    let workspace_name = match place.map(str::trim).filter(|w| !w.is_empty()) {
        None => crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME),
        Some(raw) => resolve_launch_workspace(raw).await?,
    };

    // Resolve the branch address: an explicit identifier resumes/creates that
    // named branch; otherwise the record's behaviour picks between the one
    // common branch and a fresh one-off run.
    let agent_id = match identifier {
        Some(id) => crate::session::custom_role_agent_id(name, Some(id)),
        None => match role.behavior {
            crate::custom_role::CustomRoleBehavior::Continuous => {
                crate::session::custom_role_agent_id(name, None)
            }
            crate::custom_role::CustomRoleBehavior::OneOff => {
                crate::session::custom_role_run_agent_id(name)
            }
        },
    };

    let launch = crate::Launch {
        role: crate::Role::Custom,
        name: name.to_string(),
        identifier: identifier.map(str::to_string),
        prompt: role.prompt,
        tools: role.tools,
        agent_id: agent_id.clone(),
        reply_to: None,
    };
    let job = launch_job(message.to_string(), workspace_name.clone(), launch);

    // Durable delivery: the envelope is persisted BEFORE it is routed, so a
    // restart between the two replays it at boot.
    crate::agent::message_router::stamp_and_route(job, &agent_id, "custom role launch").await;

    Ok(format!(
        "custom role '{name}' launched on branch {agent_id} in {workspace_name}"
    ))
}

/// Launch (or continue) a coder branch — a normal [`crate::Role::Coder`] agent on
/// the persistent conversation `coder_{branch}`.
///
/// `place` is the folder the branch works in, already resolved by the same rule
/// as a single coder run (so a registered project, anything inside one, the
/// filesystem root and the product's data directory were refused before this is
/// called); `None` runs in `caller_ws` — the caller's own workspace, where a
/// coder runs today when none is given. A later use pointing at another folder
/// runs it there; the branch name identifies the conversation and its history,
/// so the same name resumes the same conversation. Returns a one-line
/// acknowledgement with no job number — a coder branch's report arrives when its
/// turn finishes, not as a durable round.
///
/// `asker` is the launch of the custom role that asked, when one did: it makes
/// the report return into the asking role's branch rather than the admin's
/// Assistant, addressed by the caller's own workspace. A coder branch has
/// nothing to configure and no place in Settings.
pub(crate) async fn coder_branch(
    caller_ws: &crate::Workspace,
    branch: &str,
    message: &str,
    place: Option<&crate::Workspace>,
    asker: Option<crate::Launch>,
) -> Result<String> {
    anyhow::ensure!(
        !message.trim().is_empty(),
        "usage: the message must not be empty"
    );
    let branch = branch.trim();
    crate::custom_role::validate_coder_branch(branch)?;

    // The branch's folder ADDRESS: the caller's own workspace name when no
    // folder was given (where a coder runs today when none is), otherwise the
    // canonical folder path — the same durable address a custom role's launch
    // stores for a folder, so the next turn and a recovery resolve the branch
    // back into the same place.
    let workspace_name = place.map_or_else(|| caller_ws.name.clone(), |ws| ws.path.clone());

    let agent_id = crate::session::coder_branch_agent_id(branch);
    // The report returns to the conversation that asked: this branch's own
    // address when a custom role asked (the launch carries the asker), the
    // admin's Assistant otherwise. The routed job's destination is read from the
    // launch's own field, so the durable launch and the routing decision cannot
    // drift.
    let reply_to = asker.map(|launch| crate::agent::message_router::Asker {
        launch,
        workspace: caller_ws.name.clone(),
    });
    let launch = crate::Launch {
        role: crate::Role::Coder,
        name: String::new(),
        identifier: Some(branch.to_string()),
        prompt: String::new(),
        tools: Vec::new(),
        agent_id: agent_id.clone(),
        reply_to: reply_to.map(Box::new),
    };

    let job = launch_job(message.to_string(), workspace_name.clone(), launch);

    // Durable delivery: the envelope is persisted BEFORE it is routed (and
    // BEFORE any session write), so a restart between the two replays it at boot.
    crate::agent::message_router::stamp_and_route(job, &agent_id, "coder branch launch").await;

    // The acknowledgement names the resolved folder path, exactly as the
    // single-run coder's does (same trimmed spelling) — the branch's durable
    // address stays in `workspace_name`.
    let folder = place.map_or_else(|| caller_ws.path.clone(), |ws| ws.path.clone());
    let folder = folder.trim_end_matches(std::path::MAIN_SEPARATOR);
    Ok(format!(
        "Coder branch '{branch}' is on it in {folder}; its report will arrive in this conversation."
    ))
}

/// How a branch is named in the lists: a custom role's name (plus its branch
/// identifier when it was named), or a coder branch's own label.
#[must_use]
pub(crate) fn display_label(launch: &crate::Launch) -> String {
    let identifier = launch.identifier.as_deref();
    match launch.role {
        crate::Role::Coder => format!("Coder branch \"{}\"", identifier.unwrap_or_default()),
        _ => match identifier {
            Some(id) => format!("Custom role \"{}\" — branch \"{id}\"", launch.name),
            None => format!("Custom role \"{}\"", launch.name),
        },
    }
}

/// Stop one branch launch: cancel the running agent, mark the branch as
/// deliberately stopped (so automatic recovery never restarts it), drop the
/// launch's deferred envelopes (so boot replay does not replay them) and release
/// the engine of the folder it worked in (see
/// [`crate::search_engine::release_path_named_engine`]).
///
/// The persisted mark — not the in-memory cancel signal — is what survives a
/// restart; a later explicit launch of the same branch clears it
/// (`Agent::work`).
///
/// A stop cancels the launch's own agent only: a durable round it had already
/// dispatched (analyze/send) keeps running, and its result arrives later as a new
/// turn of the branch.
pub(crate) async fn stop(agent_id: &str) -> Result<()> {
    crate::agent::registry::AGENT_REGISTRY.cancel_by_parent_key(
        &crate::agent::registry::ParentKey::Launch(agent_id.to_string()),
    );
    crate::session::store()
        .set_launch_stopped(agent_id, true)
        .await?;
    crate::jobs::delete_pending_jobs_for(agent_id).await?;
    // The branch is going away, so release the engine of the folder it last
    // worked in: nothing else keeps it alive (see
    // `crate::search_engine::release_path_named_engine`). Best-effort — a failed
    // read just leaves it registered.
    if let Some(workspace) = crate::session::store()
        .session_workspace_name(agent_id)
        .await
    {
        crate::search_engine::release_path_named_engine(&workspace);
    }
    Ok(())
}

/// Resolve a launch's `place` argument into the workspace ADDRESS stored with the
/// launch — the resolved workspace's own name: a registered workspace keeps its
/// name (whether the launch named it by name or by its folder), `personal:{user}`
/// its key, and any other folder its canonical absolute path
/// ([`crate::workspace::synthesize_from_path`] names a path workspace by the path
/// itself, so all three are durable keys [`crate::users::resolve_workspace`] can
/// resolve back).
pub(crate) async fn resolve_launch_workspace(raw: &str) -> Result<String> {
    let ws = crate::users::resolve_workspace(raw).await?.ok_or_else(|| {
        anyhow::anyhow!(
            "usage: the place must be a registered workspace name or an absolute path \
             to an existing directory"
        )
    })?;
    Ok(ws.name)
}
