//! Launching and stopping a continuable branch: a custom role's, or a worker branch's.
//!
//! Both kinds are normal durable envelopes ([`crate::Launch`]) persisted BEFORE
//! they are routed, so they survive a restart, and stopped through one path.
//! A custom role runs the administrator's authored role; a worker branch runs a
//! normal [`crate::Role::Worker`] agent on a continuation of its own conversation.
//!
//! The durable record types of a `send` chain live here, beside the launch that carries them
//! ([`crate::Launch::chain`]), so the persistence and session layers can read a chain without
//! depending on a tool module; the chain's behaviour is [`crate::tools::chain`].
//!
//! [`Step`] names its addressee in `send`'s vocabulary ([`crate::tools::send::Addressee`]), so this
//! module and that tool reference each other by type. The dependency is deliberate and one-way in
//! behaviour: `send` is the only producer of a chain, and nothing here calls into the tool.
//!
//! The rule that two agents must not work in one folder lives here too ([`BusyFolder`]), because
//! every launch is the place that has to apply it.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::tools::send::Addressee;

/// The longest chain: the outer call plus nine nested `for`.
pub(crate) const MAX_STEPS: usize = 10;

/// One step of a chain: the call's own shape minus `message`.
///
/// A field this build does not know makes the whole record unreadable
/// ([`crate::launch::Stored::Unreadable`]) rather than silently dropped: a step that carried a
/// schema must never come back as an ordinary delegation returning prose.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub addressee: Addressee,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub place: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The JSON schema this step's answer must come back as, when the call asked for one
    /// ([`crate::extract`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extract: Option<crate::extract::Extract>,
}

impl Step {
    /// The step a chain-carrying durable round is: `send`'s worker, the only round kind a chain
    /// hands its answer on from.
    #[must_use]
    pub(crate) fn worker() -> Self {
        Self {
            addressee: Addressee::Worker,
            place: None,
            branch: None,
            name: None,
            extract: None,
        }
    }

    /// The role this step runs as, which is the model its answer's transformation runs on.
    ///
    /// Every address names its own role, the three a chain refuses included, so a value this build
    /// cannot run as a step still names the role its transformation would use rather than falling
    /// back to a sentinel.
    #[must_use]
    pub(crate) fn role(&self) -> crate::Role {
        match self.addressee {
            Addressee::Worker | Addressee::WorkerBranch => crate::Role::Worker,
            Addressee::Role => crate::Role::Custom,
            Addressee::Manager => crate::Role::Manager,
            Addressee::Guest | Addressee::Assistant => crate::Role::Assistant,
        }
    }

    /// What this step is, for the step after it and for the failure it may cause.
    #[must_use]
    pub(crate) fn label(&self) -> String {
        match self.addressee {
            Addressee::Worker => crate::Role::Worker.display_label().to_string(),
            Addressee::WorkerBranch => {
                display_label(crate::Role::Worker, "", self.branch.as_deref())
            }
            Addressee::Role => display_label(
                crate::Role::Custom,
                self.name.as_deref().unwrap_or_default(),
                self.branch.as_deref(),
            ),
            // Parsing refuses every other address, so this arm only serves the type.
            other => other.as_str().to_string(),
        }
    }
}

/// A chain in flight: the step running now, the steps after it, and where it answers.
///
/// It rides on the record of the step running now (see [`crate::tools::chain`]), never on a
/// store of its own.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Chain {
    /// One id for the whole chain — its Running Agents group and its stop.
    pub id: String,
    /// The task exactly as first sent: every step receives it unchanged.
    pub task: String,
    /// The user the chain answers to (the owner) and the channel it started on.
    pub user: String,
    pub channel: String,
    /// The workspace a step that names no `place` runs in — the one the same delegation from the
    /// owner's Assistant runs in today.
    pub workspace: String,
    /// The address of the step running now, and its 1-based position.
    pub current: Step,
    pub step: usize,
    /// The steps after it, in order. Empty means the current step is the last.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub next: Vec<Step>,
}

impl Chain {
    /// How many steps the chain has: the one running now and the ones after it.
    #[must_use]
    pub(crate) fn total(&self) -> usize {
        self.step + self.next.len()
    }
}

/// A chain value read back from a store: `jobs.chain` or `session_metadata.chain`.
///
/// A chain step must never come back as an ordinary delegation — its answer would reach the caller
/// mid-chain — so a record that can no longer be read is its own state rather than "no chain".
#[derive(Clone, Debug)]
pub enum Stored {
    /// An ordinary round or branch.
    Ordinary,
    /// A chain step.
    Step(Box<Chain>),
    /// A chain step whose record this build cannot read. Nothing can be handed on, so the step is
    /// abandoned and its caller told the chain stopped. The chain id is kept when the value still
    /// names one, so the ending can retire the chain's other conversations.
    Unreadable(Option<String>),
}

impl Stored {
    /// Read a stored value. An unreadable one is logged here, once, where it is recognised.
    #[must_use]
    pub(crate) fn parse(raw: Option<String>) -> Self {
        match raw {
            None => Self::Ordinary,
            Some(raw) => match serde_json::from_str(&raw) {
                Ok(chain) => Self::Step(Box::new(chain)),
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "a stored chain is unreadable — its step must be abandoned, not delivered",
                    );
                    Self::Unreadable(named_chain_id(&raw))
                }
            },
        }
    }

    /// The chain this round is a step of, when it is one.
    #[must_use]
    pub(crate) fn step(&self) -> Option<&Chain> {
        match self {
            Self::Step(chain) => Some(chain),
            Self::Ordinary | Self::Unreadable(_) => None,
        }
    }

    /// Whether the record is a chain step that can no longer be read.
    #[must_use]
    pub(crate) fn is_unreadable(&self) -> bool {
        matches!(self, Self::Unreadable(_))
    }

    /// The chain an unreadable record still names, when it names one: enough to end that chain
    /// properly, though not to run any step of it.
    #[must_use]
    pub(crate) fn unreadable_chain(&self) -> Option<&str> {
        match self {
            Self::Unreadable(id) => id.as_deref(),
            Self::Ordinary | Self::Step(_) => None,
        }
    }
}

/// The chain id a value this build cannot read as a chain still names. The field is read on its own
/// because an ending must still retire the chain's conversations; a value that names nothing is
/// unidentifiable, and only the one failure is left.
fn named_chain_id(raw: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(raw)
        .ok()?
        .get("id")?
        .as_str()
        .map(str::to_string)
}

/// The durable record id of one step. Deterministic, so a hand-off replayed after it committed
/// finds the next step's record instead of starting it twice.
#[must_use]
pub(crate) fn record_id(chain_id: &str, step: usize) -> String {
    format!("chain_{chain_id}_{step}")
}

/// The chain id and step a record id names, when it names one — all a corrupt record still says.
#[must_use]
pub(crate) fn record_parts(record: &str) -> Option<(&str, usize)> {
    let rest = record.strip_prefix("chain_")?;
    let (chain_id, step) = rest.rsplit_once('_')?;
    Some((chain_id, step.parse().ok()?))
}

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
        urgent: false,
        extract: None,
    }
}

/// How a launch treats a folder another agent is already working in.
#[derive(Clone, Copy)]
pub(crate) enum BusyFolder<'a> {
    /// An ordinary delegation refuses it: two agents in one folder overwrite each other's edits.
    Refuse,
    /// A chain step. Its own chain's earlier steps and the conversation it targets are not another
    /// agent — it waits its turn there — while any other agent in the folder still fails it.
    Chain(&'a Chain),
    /// A launch nobody asked for and nobody can refuse, an automation's: run it as before.
    Ignore,
}

impl BusyFolder<'_> {
    /// The chain a step of which this launch is, when it is one.
    fn chain(&self) -> Option<&Chain> {
        match self {
            Self::Chain(chain) => Some(chain),
            Self::Refuse | Self::Ignore => None,
        }
    }

    /// Refuse `folder` under this rule. `target` is the conversation the launch is aimed at, when
    /// it has one: a chain step waits its turn in it.
    pub(crate) fn check(&self, folder: &crate::Workspace, target: Option<&str>) -> Result<()> {
        // The caller is not ANOTHER agent: its own run is dispatching, and a custom role handing a
        // worker its own folder is the normal delegation. The command line has no session.
        let caller = crate::agent::tool_agent_id().ok();
        let (ignore, chain) = match self {
            Self::Refuse => (caller.as_deref(), None),
            Self::Chain(chain) => (target, Some(chain.id.as_str())),
            Self::Ignore => return Ok(()),
        };
        refuse_busy_folder_excluding(folder, ignore, chain)
    }
}

/// Refuse a folder another agent is already working in: two agents in one folder overwrite each
/// other's edits, so parallel agents that can collide must not be started in one folder.
pub(crate) fn refuse_busy_folder(folder: &crate::Workspace) -> Result<()> {
    BusyFolder::Refuse.check(folder, None)
}

fn refuse_busy_folder_excluding(
    folder: &crate::Workspace,
    ignore: Option<&str>,
    chain: Option<&str>,
) -> Result<()> {
    let Some(running) = crate::agent::registry::AGENT_REGISTRY.worker_in_folder_excluding_chain(
        folder.as_path(),
        ignore,
        chain,
    ) else {
        return Ok(());
    };
    let working = if running.role == crate::Role::Worker.as_str() {
        "a worker"
    } else {
        "a custom role"
    };
    // The folder as the acknowledgements spell it — the resolved path, one way.
    let folder = folder.path.trim_end_matches(std::path::MAIN_SEPARATOR);
    Err(anyhow::anyhow!(
        "usage: {working} is already working in {folder}; the message was not sent, because two \
         agents in one folder overwrite each other's edits and parallel agents that can collide \
         must not be started in one folder — hint: split the work across different folders, or wait \
         for the running one to finish, then send again."
    ))
}

/// Launch `name`'s custom role with `message` as the launch message.
///
/// `branch` selects a continuable named branch; without one the record's
/// behaviour decides (one common branch, or a one-off run). `workspace` is the
/// folder the launch runs in, already resolved by [`launch_workspace`]. The
/// launch is persisted as a durable envelope BEFORE it is routed, so it survives
/// a restart. Returns a one-line confirmation for the caller.
///
/// `busy` is the folder rule — checked here because every custom-role launch passes
/// through this one place, so no launch path can forget it.
///
/// `answered_for` decides what a failed durable write means — see
/// [`crate::agent::message_router::hand_over`]. An agent's producer keeps the
/// best-effort route on a failed write; the command line's gets the failure back
/// and the launch does NOT happen.
#[expect(clippy::too_many_arguments)] // one positional argument per fact
pub(crate) async fn custom_role(
    name: &str,
    message: &str,
    branch: Option<&str>,
    workspace: &crate::Workspace,
    asker: Option<crate::agent::message_router::Asker>,
    busy: BusyFolder<'_>,
    extract: Option<&crate::extract::Extract>,
    answered_for: crate::agent::message_router::AnsweredFor,
) -> Result<String> {
    anyhow::ensure!(
        !message.trim().is_empty(),
        "usage: the launch message must not be empty"
    );
    let role = crate::custom_role::get(name)
        .await?
        .ok_or_else(|| anyhow::anyhow!("no custom role named '{name}'"))?;

    let identifier = branch.map(str::trim).filter(|id| !id.is_empty());
    if let Some(id) = identifier {
        crate::custom_role::validate_identifier(id)?;
    }

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

    // A chain step waits its turn in the conversation it names — its own steps and that
    // conversation are not another agent — while any other agent in the folder still fails the
    // chain; an ordinary delegation refuses the folder outright.
    busy.check(workspace, Some(&agent_id))?;

    // The answer (and the failure notice) returns to the agent that launched this one, or to the
    // owner's Assistant when nothing launched it — one carrier, the launch itself.
    let launch = crate::Launch {
        role: crate::Role::Custom,
        name: name.to_string(),
        identifier: identifier.map(str::to_string),
        prompt: role.prompt,
        tools: role.tools,
        agent_id: agent_id.clone(),
        reply_to: asker.map(Box::new),
        chain: busy.chain().cloned().map(Box::new),
        extract: extract.cloned(),
    };
    hand_over_launch(launch, message, &workspace.name, answered_for).await?;

    Ok(format!(
        "custom role '{name}' launched on branch {agent_id} in {}",
        workspace.name
    ))
}

/// Launch (or continue) a worker branch — a normal [`crate::Role::Worker`] agent on
/// the persistent conversation `worker_{branch}`.
///
/// `place` is the folder the branch works in, already resolved by the same rule
/// as a single worker run (so a managed workspace, anything inside one, the
/// filesystem root and the product's data directory were refused before this is
/// called); `None` runs in `caller_ws` — the caller's own workspace, where a
/// worker runs today when none is given. A later use pointing at another folder
/// runs it there; the branch name identifies the conversation and its history,
/// so the same name resumes the same conversation. Returns a one-line
/// acknowledgement with no job number — a worker branch's report arrives when its
/// turn finishes, not as a durable round.
///
/// `asker` is the launch of the custom role that asked, when one did: it makes
/// the report return into the asking role's branch rather than the admin's
/// Assistant, addressed by the caller's own workspace. A worker branch has
/// nothing to configure and no place in Settings.
///
/// `busy` is the folder rule — checked here because every worker-branch launch passes
/// through this one place.
///
/// `answered_for` decides what a failed durable write means — see
/// [`crate::agent::message_router::hand_over`]. An agent's producer keeps the
/// best-effort route on a failed write; the command line's gets the failure back
/// and the launch does NOT happen.
#[expect(clippy::too_many_arguments)] // one positional argument per fact
pub(crate) async fn worker_branch(
    caller_ws: &crate::Workspace,
    branch: &str,
    message: &str,
    place: Option<&crate::Workspace>,
    asker: Option<crate::Launch>,
    busy: BusyFolder<'_>,
    extract: Option<&crate::extract::Extract>,
    answered_for: crate::agent::message_router::AnsweredFor,
) -> Result<String> {
    anyhow::ensure!(
        !message.trim().is_empty(),
        "usage: the message must not be empty"
    );
    let branch = branch.trim();
    crate::custom_role::validate_worker_branch(branch)?;

    // The branch's folder ADDRESS: the caller's own workspace name when no
    // folder was given (where a worker runs today when none is), otherwise the
    // canonical folder path — the same durable address a custom role's launch
    // stores for a folder, so the next turn and a recovery resolve the branch
    // back into the same place.
    let workspace_name = place.map_or_else(|| caller_ws.name.clone(), |ws| ws.path.clone());
    let folder = place.unwrap_or(caller_ws);
    let agent_id = crate::session::worker_branch_agent_id(branch);
    busy.check(folder, Some(&agent_id))?;

    // The report returns to the conversation that asked: this branch's own
    // address when a custom role asked (the launch carries the asker), the
    // admin's Assistant otherwise. The routed job's destination is read from the
    // launch's own field, so the durable launch and the routing decision cannot drift.
    let reply_to = asker.map(|launch| crate::agent::message_router::Asker {
        launch,
        workspace: caller_ws.name.clone(),
    });
    let launch = crate::Launch {
        role: crate::Role::Worker,
        name: String::new(),
        identifier: Some(branch.to_string()),
        prompt: String::new(),
        tools: Vec::new(),
        agent_id: agent_id.clone(),
        reply_to: reply_to.map(Box::new),
        chain: busy.chain().cloned().map(Box::new),
        extract: extract.cloned(),
    };

    hand_over_launch(launch, message, &workspace_name, answered_for).await?;

    let folder = folder.path.trim_end_matches(std::path::MAIN_SEPARATOR);
    Ok(format!(
        "Worker branch '{branch}' is on it in {folder}; its report arrives as a <worker-report> message."
    ))
}

/// Hand a built launch over as one durable envelope: persisted BEFORE it is routed, so it survives
/// a restart, and addressed by the launch's own agent id. The chain steps a launch carries ride on
/// it, so a hand-off replayed after it committed finds the same durable record.
pub(crate) async fn hand_over_launch(
    launch: crate::Launch,
    message: &str,
    workspace_name: &str,
    answered_for: crate::agent::message_router::AnsweredFor,
) -> Result<()> {
    let target = launch.agent_id.clone();
    let job = launch_job(message.to_string(), workspace_name.to_string(), launch);
    crate::agent::message_router::hand_over(job, &target, "launch", answered_for).await
}

/// How a launch names itself — in Running Agents, and as the producer a chain step is told about.
#[must_use]
pub(crate) fn display_label(role: crate::Role, name: &str, identifier: Option<&str>) -> String {
    match role {
        crate::Role::Worker => format!("Worker branch \"{}\"", identifier.unwrap_or_default()),
        _ => match identifier {
            Some(id) => format!("Custom role \"{name}\" — branch \"{id}\""),
            None => format!("Custom role \"{name}\""),
        },
    }
}

/// Stop one branch launch: cancel the running agent, mark the branch as
/// deliberately stopped (so automatic recovery never restarts it), drop the
/// launch's deferred envelopes (so boot replay does not replay them) and release
/// the engine of the folder it worked in (see
/// [`crate::workspace_index::release_path_named_engine`]).
///
/// The persisted mark — not the in-memory cancel signal — is what survives a
/// restart; a later explicit launch of the same branch clears it (`Agent::work`).
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
    // `crate::workspace_index::release_path_named_engine`). Best-effort — a failed
    // read just leaves it registered.
    if let Some(workspace) = crate::session::store()
        .session_workspace_name(agent_id)
        .await
    {
        crate::workspace_index::release_path_named_engine(&workspace);
    }
    Ok(())
}

/// Resolve a launch's `place` into the workspace the launch runs in — the
/// workspace ADDRESS stored with the launch is its `name`: a managed
/// workspace keeps its name (whether the launch named it by name or by its
/// folder), `personal:{user}` its key, and any other folder its canonical
/// absolute path ([`crate::workspace::synthesize_from_path`] names a path
/// workspace by the path itself, so all three are durable keys
/// [`crate::users::resolve_workspace`] can resolve back). An absent, empty or
/// whitespace `place` runs in the owner's personal folder, exactly as the owner's Assistant does.
pub(crate) async fn launch_workspace(place: Option<&str>) -> Result<crate::Workspace> {
    match place.map(str::trim).filter(|w| !w.is_empty()) {
        None => Ok(crate::users::personal_workspace_struct(
            crate::users::ADMIN_USER_NAME,
        )),
        Some(raw) => crate::users::resolve_workspace(raw).await?.ok_or_else(|| {
            anyhow::anyhow!(
                "usage: the place must be a managed workspace name or an absolute path \
                 to an existing directory"
            )
        }),
    }
}
