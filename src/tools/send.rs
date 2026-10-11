//! SendTool — hand one message to exactly one addressee.
//!
//! The addressees are the coder (a single coder run), a coder branch (a
//! continuable coder conversation), a project workspace's Manager, and an
//! administrator-defined custom role. A holder with one addressee does not
//! declare the choice argument; a holder with several does, and a value outside
//! its set is refused.
//!
//! The Engineer's instance addresses a coder only and runs it synchronously —
//! the coder's report returns inline. The admin Assistant and a custom role
//! dispatch asynchronously: the coder reports later as a `<coder-report>`
//! message, a coder branch's turn delivers its answer into the conversation that
//! asked, the Manager answers as an `<manager-message>`, and a custom role's
//! answer arrives as a `<custom-role-message>`. Argument mechanics and refusals
//! live in [`super::delegation`] and [`crate::launch`].

use crate::prompt::load_prompt;
use crate::tools::Tool;
use crate::tools::analyze::DispatchMode;
use crate::{Role, Workspace};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

/// One of `send`'s optional arguments — the shared `message` is not one. The
/// spelling is the interface's own, and the lead opens that argument's
/// description.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Argument {
    /// The folder the addressee works in (which project a `manager` call
    /// messages).
    Place,
    /// The branch a coder branch or a custom role continues.
    Branch,
    /// A custom role's name.
    Name,
}

impl Argument {
    /// Every optional argument, in the order their descriptions are declared.
    const ALL: [Self; 3] = [Self::Place, Self::Branch, Self::Name];

    /// The argument's spelling in the interface.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Place => "place",
            Self::Branch => "branch",
            Self::Name => "name",
        }
    }

    /// The lead of this argument's description.
    const fn lead(self) -> &'static str {
        match self {
            Self::Place => "The folder this call names",
            Self::Branch => "The branch to continue or create",
            Self::Name => "The custom role's name",
        }
    }
}

/// What one addressee takes besides the shared `message`: which argument, what
/// it holds for that addressee (phrased to read after the argument's lead), and
/// whether the addressee cannot do without it.
///
/// This table is the single owner of the per-addressee requirements: the choice
/// declaration, every argument description, and the arguments a one-addressee
/// holder has no alternative to are all read from it.
struct Requirement {
    arg: Argument,
    note: &'static str,
    required: bool,
}

/// Exactly one addressee of a `send` call. The order of [`Self::ALL`] is the
/// interface order — the narrowest kind is first.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Addressee {
    /// A single coder run: implement a scoped task, report back.
    Coder,
    /// A continuable coder conversation: continue (or create) a named branch.
    CoderBranch,
    /// The Manager of one registered project workspace.
    Manager,
    /// An administrator-defined custom role.
    Role,
}

impl Addressee {
    /// The argument naming the choice.
    pub const ARG: &'static str = "addressee";

    /// Every addressee, in interface order (narrowest first).
    pub const ALL: [Self; 4] = [Self::Coder, Self::CoderBranch, Self::Manager, Self::Role];

    /// The wire spellings of [`Self::ALL`], in the same order.
    pub const STRS: &'static [&'static str] = &[
        Self::Coder.as_str(),
        Self::CoderBranch.as_str(),
        Self::Manager.as_str(),
        Self::Role.as_str(),
    ];

    /// The wire spelling of this addressee.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Coder => "coder",
            Self::CoderBranch => "coder_branch",
            Self::Manager => "manager",
            Self::Role => "role",
        }
    }

    /// Parse a wire spelling.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.as_str() == value)
    }

    /// What this addressee does, in the choice declaration's own words.
    const fn does(self) -> &'static str {
        match self {
            Self::Coder => "runs one scoped implementation and reports back",
            Self::CoderBranch => "continues a named coder branch",
            Self::Manager => "messages a project's Manager",
            Self::Role => "starts a custom role",
        }
    }

    /// The arguments this addressee takes besides the shared `message`.
    fn requirements(self) -> &'static [Requirement] {
        /// The coder's folder rule, shared by both coder kinds so their `place`
        /// text stays one entry.
        const OUTSIDE_PROJECTS: &str = "any directory outside the registered projects";
        match self {
            Self::Coder => &[Requirement {
                arg: Argument::Place,
                note: OUTSIDE_PROJECTS,
                required: false,
            }],
            Self::CoderBranch => &[
                Requirement {
                    arg: Argument::Place,
                    note: OUTSIDE_PROJECTS,
                    required: false,
                },
                Requirement {
                    arg: Argument::Branch,
                    note: "the name of the branch to continue, created on its first use (ASCII \
                           letters, digits, `_` and `-`, at most 32 chars)",
                    required: true,
                },
            ],
            Self::Manager => &[Requirement {
                arg: Argument::Place,
                note: "the project to message, by its registered name or its own folder",
                required: true,
            }],
            Self::Role => &[
                Requirement {
                    arg: Argument::Place,
                    note: "a registered project, a folder path, or `personal:<user>`",
                    required: false,
                },
                Requirement {
                    arg: Argument::Branch,
                    note: "a named branch of the role to continue, whose history is restored",
                    required: false,
                },
                Requirement {
                    arg: Argument::Name,
                    note: "the name of an existing custom role",
                    required: true,
                },
            ],
        }
    }

    /// The choice declaration's clause: what this addressee does, and what it
    /// cannot do without.
    fn clause(self) -> String {
        let required: Vec<String> = self
            .requirements()
            .iter()
            .filter(|requirement| requirement.required)
            .map(|requirement| format!("`{}`", requirement.arg.as_str()))
            .collect();
        if required.is_empty() {
            self.does().to_string()
        } else {
            format!("{} ({} required)", self.does(), required.join(", "))
        }
    }
}

pub struct SendTool {
    /// Controls how a coder run is dispatched (see [`DispatchMode`]).
    dispatch_mode: DispatchMode,
    /// The role of the calling agent. Used to route async results to the
    /// correct agent channel.
    pub caller_role: Role,
    /// The addressees this holder may use, in interface order. A single
    /// addressee is never declared as a choice argument.
    kinds: Vec<Addressee>,
    /// Whether the optional `place` argument is accepted. Every addressee takes
    /// a folder, so every async holder declares it; the Engineer's coder-only
    /// instance does not.
    accepts_place: bool,
}

impl SendTool {
    /// The Engineer's instance: a coder only, synchronously, no `place` — the
    /// coder's report returns inline in the call's result.
    #[must_use]
    pub fn engineer() -> Self {
        Self {
            dispatch_mode: DispatchMode::Sync,
            caller_role: Role::Engineer,
            kinds: vec![Addressee::Coder],
            accepts_place: false,
        }
    }

    /// The admin Assistant's variant: every addressee, asynchronously, with the
    /// optional `place`.
    #[must_use]
    pub fn admin(caller_role: Role) -> Self {
        Self {
            dispatch_mode: DispatchMode::Async,
            caller_role,
            kinds: Addressee::ALL.to_vec(),
            accepts_place: true,
        }
    }

    /// A custom role's variant: exactly the granted addressees, asynchronously,
    /// with the optional `place`. Never empty — a grant entry that named no kind
    /// takes the narrowest addressee (`coder`).
    #[must_use]
    pub fn granted(kinds: &[Addressee]) -> Self {
        let kinds = if kinds.is_empty() {
            vec![Addressee::Coder]
        } else {
            kinds.to_vec()
        };
        Self {
            dispatch_mode: DispatchMode::Async,
            caller_role: Role::Custom,
            kinds,
            accepts_place: true,
        }
    }

    /// The wire spellings of this holder's addressees.
    fn allowed(&self) -> Vec<&'static str> {
        self.kinds.iter().map(|kind| kind.as_str()).collect()
    }

    /// Whether this holder declares the choice argument: it has more than one
    /// addressee to choose between. A one-value choice is never declared.
    fn declares_choice(&self) -> bool {
        self.kinds.len() > 1
    }

    /// Whether this holder's interface declares `arg` — the one question the
    /// rendered description, the schema and the reading path all ask.
    fn declares(&self, arg: Argument) -> bool {
        (arg != Argument::Place || self.accepts_place)
            && self
                .kinds
                .iter()
                .any(|addressee| addressee.requirements().iter().any(|req| req.arg == arg))
    }

    /// The optional argument this call names: `None` when it names none **or when
    /// this holder does not declare it** — an argument a holder is never offered
    /// is inert, never a refusal. A value of the wrong type is a usage error for
    /// every addressee, exactly as the declared string type says.
    fn optional<'a>(&self, args: &'a serde_json::Value, arg: Argument) -> Result<Option<&'a str>> {
        if !self.declares(arg) {
            return Ok(None);
        }
        match args.get(arg.as_str()) {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::String(value)) => Ok(Some(value)),
            Some(other) => Err(super::wrong_type(arg.as_str(), "a string", other)),
        }
    }

    /// The optional `place` this call names: `None` when it names none, and
    /// `None` for a holder that declares no `place` at all — the Engineer's
    /// coder-only instance runs where the caller is.
    async fn optional_place(&self, args: &serde_json::Value) -> Result<Option<Workspace>> {
        if self.declares(Argument::Place) {
            super::delegation::resolve(super::delegation::DelegationKind::Coder, args).await
        } else {
            Ok(None)
        }
    }

    /// One note under its label(s) as the model reads it: labelled only where
    /// this holder declares the choice, the note alone where it does not — a
    /// one-addressee holder must never read a value it cannot pass. The single
    /// place a `send` note gets its label.
    fn labelled(&self, labels: &[&str], note: &str) -> String {
        if self.declares_choice() {
            let names = labels
                .iter()
                .map(|label| format!("`{label}`"))
                .collect::<Vec<_>>()
                .join("/");
            format!("{names}: {note}")
        } else {
            note.to_string()
        }
    }

    /// The description of one optional argument: its lead, what it holds for
    /// every addressee this holder has that takes it, and whether it can be left
    /// out. `None` when this holder does not declare the argument at all.
    fn describe(&self, arg: Argument) -> Option<String> {
        if !self.declares(arg) {
            return None;
        }
        let takers: Vec<(Addressee, &'static Requirement)> = Addressee::ALL
            .into_iter()
            .filter(|addressee| self.kinds.contains(addressee))
            .filter_map(|addressee| {
                addressee
                    .requirements()
                    .iter()
                    .find(|requirement| requirement.arg == arg)
                    .map(|requirement| (addressee, requirement))
            })
            .collect();
        let declared = self.declares_choice();
        // Addressees that share a note read as one entry under their joint label.
        let mut grouped: Vec<(Vec<&'static str>, &'static Requirement)> = Vec::new();
        for (addressee, requirement) in &takers {
            let same = grouped.iter_mut().find(|(_, held)| {
                held.note == requirement.note && held.required == requirement.required
            });
            match same {
                Some((labels, _)) => labels.push(addressee.as_str()),
                None => grouped.push((vec![addressee.as_str()], requirement)),
            }
        }
        let notes: Vec<String> = grouped
            .iter()
            .map(|(labels, requirement)| self.labelled(labels, requirement.note))
            .collect();
        let needed_by: Vec<String> = takers
            .iter()
            .filter(|(_, requirement)| requirement.required)
            .map(|(addressee, _)| format!("`{}`", addressee.as_str()))
            .collect();
        // Whether the argument can be left out follows from the choices this
        // holder has: with the choice declared, only the choices that cannot do
        // without it are named; without it, its one addressee's own answer is
        // unconditional.
        let tail = match needed_by.len() {
            0 => " Omit for the default.".to_string(),
            _ if !declared => " Required.".to_string(),
            needed if needed < takers.len() => {
                format!(
                    " Omit for the default; required for {}.",
                    needed_by.join(", ")
                )
            }
            _ => format!(" Required for {}.", needed_by.join(", ")),
        };
        Some(format!("{} — {}.{tail}", arg.lead(), notes.join("; ")))
    }
}

#[async_trait]
impl Tool for SendTool {
    fn name(&self) -> &'static str {
        "send"
    }

    /// Compose the description from shared fragments: the base text, the choice
    /// note when a choice is declared, one note per held addressee, and the
    /// async/sync note.
    fn description(&self) -> String {
        let mut parts = vec![load_prompt("tool/send.md")];
        if self.declares_choice() {
            parts.push(load_prompt("tool/send_addressee.md"));
        }
        for addressee in Addressee::ALL {
            if !self.kinds.contains(&addressee) {
                continue;
            }
            // The coder note is the one that differs by dispatch: the sync
            // holder returns the coder's report inline, the async holders get it
            // later as a `<coder-report>`.
            let note = match addressee {
                Addressee::Coder if !self.dispatch_mode.is_async() => {
                    load_prompt("tool/send_coder_sync.md")
                }
                other => load_prompt(&format!("tool/send_{}.md", other.as_str())),
            };
            parts.push(self.labelled(&[addressee.as_str()], &note));
        }
        // Only an async holder returns before the work is done; the sync one's
        // own note already says its report returns in the call's result.
        if self.dispatch_mode.is_async() {
            parts.push(load_prompt("tool/send_async.md"));
        }
        parts.join("\n\n")
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let mut properties = json!({
            "message": {
                "type": "string",
                "description": "The work to hand over; make it self-contained — the addressee cannot see your conversation."
            }
        });
        let mut required = vec!["message"];
        // The choice names only the addressees this holder has: a holder is never
        // told about a choice it cannot make.
        if self.declares_choice() {
            let clauses: Vec<String> = self
                .kinds
                .iter()
                .map(|addressee| format!("`{}` {}", addressee.as_str(), addressee.clause()))
                .collect();
            properties[Addressee::ARG] = json!({
                "type": "string",
                "enum": self.allowed(),
                "description": format!("Who receives it. {}", clauses.join("; ")),
            });
            required.push(Addressee::ARG);
        }
        // Every optional argument the holder's addressees take, described from
        // the one per-addressee table — `place` included only where the holder
        // declares it: the Engineer's coder-only instance runs where the caller is.
        for arg in Argument::ALL {
            if let Some(description) = self.describe(arg) {
                properties[arg.as_str()] = json!({
                    "type": "string",
                    "description": description
                });
            }
        }
        // A one-addressee holder declares no choice, so whatever its addressee
        // cannot do without becomes unconditional.
        if let [only] = self.kinds.as_slice() {
            required.extend(
                only.requirements()
                    .iter()
                    .filter(|requirement| requirement.required)
                    .map(|requirement| requirement.arg.as_str()),
            );
        }
        super::tool_params_schema(&properties, &required)
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let message = super::get_dispatch_message(&args)?;
        let addressee = super::chosen_kind(
            &args,
            Addressee::ARG,
            &self.kinds,
            Addressee::as_str,
            Addressee::parse,
        )?;

        match addressee {
            Addressee::Coder => {
                if !self.dispatch_mode.is_async() {
                    // The Engineer's sync coder run: the report returns inline.
                    return super::SyncDurableCore::Coder
                        .run_sync_dispatch(ws, message, self.caller_role)
                        .await;
                }
                // A coder round writes outside the registered projects: only a
                // durable background round can honour the optional `place`.
                let place = self.optional_place(&args).await?;
                let job_id = crate::generate_id();
                super::SyncDurableCore::Coder.spawn_dispatch(
                    ws,
                    place.as_ref(),
                    message,
                    self.caller_role,
                    job_id,
                );
                let folder = place
                    .as_ref()
                    .map_or_else(|| ws.path.clone(), |p| p.path.clone());
                Ok(format!(
                    "Work handed to a coder in {}. Its report will arrive in this conversation as \
                     a <coder-report> message.",
                    folder.trim_end_matches(std::path::MAIN_SEPARATOR)
                ))
            }
            Addressee::CoderBranch => {
                let branch = super::get_str(&args, Argument::Branch.as_str())?;
                // A branch is pointed at a folder by the same rule as a single
                // coder run — the branch may not be a way around the coder's
                // refusals.
                let place = self.optional_place(&args).await?;
                // Read the asker before the launch: the report returns to the
                // conversation that asked.
                let asker = crate::agent::tool_custom_launch();
                crate::launch::coder_branch(ws, branch, message, place.as_ref(), asker).await
            }
            Addressee::Manager => {
                let (_, user_name) = crate::agent::tool_identity()?;
                // Read before any spawn: routes the Manager's answer back into
                // this branch when the caller is a custom role.
                let asker = crate::agent::tool_custom_launch();
                // A project's Manager cannot do without the project it messages,
                // so `place` is a required argument of a holder that can reach
                // this arm — and such a holder always declares it (pinned by the
                // holder sweep). `get_str` still refuses a call that omits it.
                let place = super::get_str(&args, Argument::Place.as_str())?;
                let project = super::delegation::resolve_project(place).await?;
                // Envelope wrapping happens only on the routed Manager-bound job.
                let envelope = crate::prompt::substitute(
                    &crate::prompt::load_prompt("assistant_message.md"),
                    &[
                        (
                            "{{user_name}}",
                            super::envelope_attr(user_name.as_str()).as_str(),
                        ),
                        ("{{message}}", message),
                    ],
                );
                crate::agent::message_router::route_agent_message_to_manager(
                    envelope,
                    project.name.clone(),
                    user_name,
                    asker.map(|launch| crate::agent::message_router::Asker {
                        launch,
                        workspace: ws.name.clone(),
                    }),
                )
                .await;
                Ok(format!(
                    "Message delivered to the manager of workspace '{}'.",
                    project.name
                ))
            }
            Addressee::Role => {
                let name = super::get_str(&args, Argument::Name.as_str())?;
                // Read through the declaring filter like every other optional
                // argument. `place` here is NOT resolved the way a coder's is:
                // `launch::custom_role` accepts the personal-workspace spelling
                // too, which the coder's refusals do not.
                crate::launch::custom_role(
                    name,
                    message,
                    self.optional(&args, Argument::Branch)?,
                    self.optional(&args, Argument::Place)?,
                )
                .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Run a dispatch tool the way the agent loop does — outside any spawn, with
    /// the caller identity in the task-locals the tools read.
    async fn dispatch(
        tool: &dyn Tool,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        crate::agent::CURRENT_TOOL_USER_NAME
            .scope("admin".to_string(), async {
                crate::agent::CURRENT_TOOL_CHANNEL
                    .scope("gui".to_string(), async {
                        crate::agent::CURRENT_TOOL_AGENT_ID
                            .scope(Some("agent_harness".to_string()), async {
                                tool.execute(ws, args).await
                            })
                            .await
                    })
                    .await
            })
            .await
    }

    /// A registered spy consumer for one conversation. The jobs a test asserts
    /// on may arrive in any order, so a job that does not match the search is
    /// kept for the next one instead of being dropped — an early report meant for
    /// a later assertion is never lost. Dropping it unregisters the conversation,
    /// so a test leaves no consumer behind for the (shared) router.
    struct Spy {
        agent_id: String,
        rx: tokio::sync::mpsc::UnboundedReceiver<crate::agent::message_router::AgentJob>,
        unclaimed: Vec<crate::agent::message_router::AgentJob>,
    }

    impl Drop for Spy {
        fn drop(&mut self) {
            crate::agent::message_router::unregister_agent(&self.agent_id);
        }
    }

    impl Spy {
        /// Register this test as the conversation's consumer.
        fn register(agent_id: &str) -> Self {
            Self {
                agent_id: agent_id.to_string(),
                rx: crate::agent::message_router::register_agent(agent_id),
                unclaimed: Vec::new(),
            }
        }

        /// The next job for this conversation, in arrival order.
        async fn next_job(&mut self) -> crate::agent::message_router::AgentJob {
            if self.unclaimed.is_empty() {
                self.receive().await
            } else {
                self.unclaimed.remove(0)
            }
        }

        /// The next job carrying `needle` — matched by what it carries (an
        /// analysis by its job id, a coder report by its `<coder-report>`
        /// envelope), never by arrival position, so a slower round or a repeated
        /// delivery cannot cross-wire the assertions.
        async fn report(&mut self, needle: &str) -> crate::agent::message_router::AgentJob {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
            loop {
                if let Some(at) = self
                    .unclaimed
                    .iter()
                    .position(|job| job.content.contains(needle))
                {
                    return self.unclaimed.remove(at);
                }
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                let job = match tokio::time::timeout(remaining, self.rx.recv()).await {
                    Ok(Some(job)) => job,
                    Ok(None) => panic!("the consumer must still be registered"),
                    Err(elapsed) => {
                        panic!("no routed job carrying {needle:?} arrived within 20s: {elapsed}")
                    }
                };
                self.unclaimed.push(job);
            }
        }

        /// Receive one job, panicking when nothing arrives or the consumer is gone.
        async fn receive(&mut self) -> crate::agent::message_router::AgentJob {
            tokio::time::timeout(std::time::Duration::from_secs(20), self.rx.recv())
                .await
                .expect("a job must be routed within 20s")
                .expect("the consumer must still be registered")
        }
    }

    /// The kind of a durable round's job row, once the dispatch task has written
    /// it — the observable proof that the round was really created.
    async fn round_kind(job_id: &str) -> String {
        let conn = &crate::session::store().conn;
        deadline_poll(|| async {
            conn.query(
                "SELECT kind FROM jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap()
            .first()
            .map(|row| row.get::<String>(0).unwrap())
        })
        .await
        .unwrap_or_else(|| panic!("no job row for {job_id} — the round was never created"))
    }

    /// The ids of the durable rounds of `kind` that exist right now.
    async fn job_ids_of_kind(kind: &str) -> Vec<String> {
        crate::session::store()
            .conn
            .query(
                "SELECT id FROM jobs WHERE kind = ?1",
                crate::db::params![kind],
            )
            .await
            .unwrap()
            .into_iter()
            .map(|row| row.get::<String>(0).unwrap())
            .collect()
    }

    /// Poll `probe` until it yields a value or a few seconds pass — the durable
    /// writes happen in spawned dispatch tasks, so they are not synchronous.
    async fn deadline_poll<T, F, Fut>(probe: F) -> Option<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Option<T>>,
    {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if let Some(value) = probe().await {
                return Some(value);
            }
            if std::time::Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }

    /// Count the deferred envelopes addressed to one agent.
    async fn pending_for(agent_id: &str) -> i64 {
        crate::session::store()
            .conn
            .query_row(
                "SELECT COUNT(*) FROM pending_jobs WHERE target_agent_id = ?1",
                crate::db::params![agent_id],
                |row| row.get::<i64>(0),
            )
            .await
            .unwrap()
    }

    /// The two dispatch tools really dispatch: every kind a holder has creates
    /// the round or reaches the addressee it names, a coder branch is created,
    /// continued in another folder and stopped, and a holder that lacks a kind is
    /// refused it.
    #[tokio::test]
    #[serial_test::serial(provider, channel_registry, drain)]
    async fn every_dispatch_kind_is_dispatched() {
        use std::sync::Arc;

        crate::util::test::init_management_test_stores().await;
        let _policy =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let _provider = crate::util::test::install_fake_provider(Arc::new(
            crate::util::test::FakeProvider::new().ok("the work is done"),
        ));
        // The response path of a branch that really runs reaches the channel
        // registry, exactly as the running service's bootstrap leaves it.
        crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);

        let project_dir = tempfile::tempdir().unwrap();
        let project_path =
            crate::util::strip_verbatim_prefix(&std::fs::canonicalize(project_dir.path()).unwrap())
                .to_str()
                .unwrap()
                .to_string();
        let project = crate::util::test::create_test_workspace(&project_path, "ws_dispatch").await;
        let home = tempfile::tempdir().unwrap();
        // The dispatcher's own workspace: the admin Assistant runs in the admin's
        // personal one, which is why its reports address the assistant's session.
        let caller = crate::workspace::test_ws_named(
            home.path().to_str().unwrap(),
            &crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME),
        );
        let dir = tempfile::tempdir().unwrap();
        let folder =
            crate::util::strip_verbatim_prefix(&std::fs::canonicalize(dir.path()).unwrap())
                .to_str()
                .unwrap()
                .to_string();

        // The conversations a report can land in, each a registered spy consumer.
        let assistant = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut assistant_spy = Spy::register(&assistant);
        let manager_agent = crate::session::manager_agent_id(&project.name);
        let mut manager_spy = Spy::register(&manager_agent);
        // The deferred envelopes that already exist: the cleanup at the end
        // removes only the rows this harness creates, because the shared global
        // store is written by parallel tests too (see `src/session/mod.rs`).
        let pending_before: std::collections::HashSet<String> =
            crate::jobs::list_pending_jobs(&crate::session::store().conn)
                .await
                .expect("list pending")
                .into_iter()
                .map(|row| row.id)
                .collect();

        let analyze = crate::tools::AnalyzeTool::admin(Role::Assistant);
        let send = SendTool::admin(Role::Assistant);

        // ── analyze, both depths ──────────────────────────────────────────
        let ack = dispatch(
            &analyze,
            &caller,
            json!({ "message": "what is here", "depth": "quick" }),
        )
        .await
        .expect("a quick analysis dispatches");
        let quick_job_id = ack
            .split("job ")
            .nth(1)
            .and_then(|rest| rest.split(')').next())
            .expect("an analysis answer names its job")
            .to_string();
        assert_eq!(round_kind(&quick_job_id).await, "analyze");
        // Each report is picked by ITS job id, never by the shared envelope tag
        // alone — the at-least-once bookkeeping can replay an envelope.
        let report = assistant_spy.report(&quick_job_id).await;
        assert!(
            report.content.contains("<analyze-tool-result>")
                && report.content.contains("Job: ")
                && report.content.contains(&quick_job_id),
            "an analysis report keeps its job number: {}",
            report.content
        );

        let ack = dispatch(
            &analyze,
            &caller,
            json!({ "message": "why is this so", "depth": "deep" }),
        )
        .await
        .expect("a deep analysis dispatches");
        let deep_job_id = ack
            .split("job ")
            .nth(1)
            .and_then(|rest| rest.split(')').next())
            .expect("a deep analysis answer names its job")
            .to_string();
        assert_eq!(round_kind(&deep_job_id).await, "research");
        // The deep investigation is long-running by design: its own completion is
        // never required, so its report is not awaited here. The round row — a
        // `research` job carrying the job id the deep answer names — is the proof
        // that the deep kind really dispatched. Both analysis depths report under
        // the same envelope with their own job number, which is what keeps one
        // analysis apart from another (and from the coder reports below).
        //
        // The round this test started is stopped the way the Running Agents view
        // stops one, so no long-running investigation outlives the test.
        crate::research_cancel::cancel_research_run(&deep_job_id).await;

        // ── send coder (a single run) ─────────────────────────────────────
        let coder_jobs_before = job_ids_of_kind("implement").await;
        let ack = dispatch(
            &send,
            &caller,
            json!({ "message": "write it", "addressee": "coder", "place": &folder }),
        )
        .await
        .expect("a coder dispatch succeeds");
        assert!(
            !ack.contains("job "),
            "a coder dispatch names no job number: {ack}"
        );
        // The round is proven by ITS OWN row — the new `implement` job — never by
        // "some row of that kind exists" in the shared store.
        let job_id = deadline_poll(|| async {
            job_ids_of_kind("implement")
                .await
                .into_iter()
                .find(|id| !coder_jobs_before.contains(id))
        })
        .await
        .expect("a coder dispatch creates its own implement round");
        assert_eq!(round_kind(&job_id).await, "implement");
        let report = assistant_spy.report("<coder-report").await;
        assert!(
            report.content.contains("<coder-report") && !report.content.contains("Job:"),
            "a coder report carries no job number: {}",
            report.content
        );
        assert!(
            report.content.contains(&folder),
            "a coder report names the folder it worked in: {}",
            report.content
        );

        // ── send manager, by the project's name and by its own folder ─────
        dispatch(
            &send,
            &caller,
            json!({ "message": "hello", "addressee": "manager", "place": &project.name }),
        )
        .await
        .expect("the manager is reachable by the project's name");
        let by_name = manager_spy.report("<assistant-message").await;
        assert!(
            by_name.content.contains("<assistant-message"),
            "{}",
            by_name.content
        );
        dispatch(
            &send,
            &caller,
            json!({ "message": "hello again", "addressee": "manager", "place": &project.path }),
        )
        .await
        .expect("the project's own folder resolves to the same Manager");
        let by_folder = manager_spy.report("<assistant-message").await;
        assert_eq!(by_folder.workspace_name, by_name.workspace_name);

        // ── send role ─────────────────────────────────────────────────────
        crate::custom_role::create(
            "dispatchable",
            "you do the work",
            &[],
            crate::custom_role::CustomRoleBehavior::Continuous,
        )
        .await
        .unwrap();
        let role_branch = crate::session::custom_role_agent_id("dispatchable", None);
        let mut role_spy = Spy::register(&role_branch);
        dispatch(
            &send,
            &caller,
            json!({ "message": "take this", "addressee": "role", "name": "dispatchable" }),
        )
        .await
        .expect("a custom role is startable");
        let launch = role_spy.report("take this").await;
        assert_eq!(
            launch.custom_role.as_ref().map(|c| c.role),
            Some(Role::Custom)
        );

        // One rule for `place` across every addressee: a value that is not a
        // string is refused, never silently read as absent — which would run the
        // work in the default folder instead of telling the caller.
        for (addressee, extra) in [
            ("coder", json!({})),
            ("coder_branch", json!({ "branch": "typed" })),
            ("manager", json!({})),
            ("role", json!({ "name": "dispatchable" })),
        ] {
            let mut call = json!({ "message": "x", "addressee": addressee });
            call.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            call["place"] = json!(123);
            let err = dispatch(&send, &caller, call)
                .await
                .expect_err("a non-string `place` is refused for every addressee");
            assert!(
                err.to_string().contains("must be a string"),
                "{addressee}: {err}"
            );
        }

        // ── send coder_branch: created, continued, moved, stopped ─────────
        let branch = "harness";
        let branch_agent = crate::session::coder_branch_agent_id(branch);
        let mut branch_spy = Spy::register(&branch_agent);
        let ack = dispatch(
            &send,
            &caller,
            json!({
                "message": "first",
                "addressee": "coder_branch",
                "branch": branch,
                "place": &folder,
            }),
        )
        .await
        .expect("a coder branch is creatable");
        assert!(!ack.contains("job "), "a coder branch names no job: {ack}");
        let first = branch_spy.next_job().await;
        assert_eq!(
            first.custom_role.as_ref().map(|c| c.role),
            Some(Role::Coder)
        );
        assert_eq!(
            first
                .custom_role
                .as_ref()
                .and_then(|c| c.identifier.as_deref()),
            Some(branch)
        );
        assert_ne!(
            first.workspace_name, caller.name,
            "the branch runs in the folder it was given"
        );

        // A second use of the same branch continues the same conversation, in the
        // folder THAT use names.
        let other = tempfile::tempdir().unwrap();
        let other_folder =
            crate::util::strip_verbatim_prefix(&std::fs::canonicalize(other.path()).unwrap())
                .to_str()
                .unwrap()
                .to_string();
        dispatch(
            &send,
            &caller,
            json!({
                "message": "second",
                "addressee": "coder_branch",
                "branch": branch,
                "place": &other_folder,
            }),
        )
        .await
        .expect("a coder branch is continuable");
        let second = branch_spy.next_job().await;
        assert_eq!(
            second.custom_role.as_ref().map(|c| c.agent_id.clone()),
            Some(branch_agent.clone()),
            "the same branch name is the same conversation"
        );
        assert_ne!(
            second.workspace_name, first.workspace_name,
            "the folder is taken from the use that named it"
        );

        // A use that names no folder runs where a coder runs today.
        dispatch(
            &send,
            &caller,
            json!({ "message": "third", "addressee": "coder_branch", "branch": branch }),
        )
        .await
        .expect("a coder branch runs without a folder");
        let third = branch_spy.next_job().await;
        assert_eq!(third.workspace_name, caller.name);

        // Stopping the branch marks it and drops the envelopes it never consumed.
        assert!(pending_for(&branch_agent).await > 0);
        crate::launch::stop(&branch_agent).await.unwrap();
        assert!(
            crate::session::store()
                .get_launch_stopped(&branch_agent)
                .await
        );
        assert_eq!(pending_for(&branch_agent).await, 0);

        // A branch that runs delivers its report into the asking conversation and
        // shows up as one session entry.
        let reporter = crate::session::coder_branch_agent_id("reporter");
        dispatch(
            &send,
            &caller,
            json!({
                "message": "report back",
                "addressee": "coder_branch",
                "branch": "reporter",
                "place": &folder,
            }),
        )
        .await
        .expect("a coder branch dispatches");
        let report = assistant_spy.report("branch=\"reporter\"").await;
        assert!(
            report.content.contains("<coder-report")
                && report.content.contains("branch=\"reporter\""),
            "a coder branch's report names its branch: {}",
            report.content
        );
        let branch_row = deadline_poll(|| async {
            crate::session::store()
                .list_sessions_with_metadata()
                .await
                .into_iter()
                .find(|s| s.agent_id == reporter)
        })
        .await
        .expect("a coder branch is one session-list entry");
        assert_eq!(branch_row.role.as_deref(), Some("coder"));
        assert_eq!(branch_row.custom_identifier.as_deref(), Some("reporter"));

        // Deleting the branch exactly as the Sessions page does — abandon its
        // unfinished jobs, then drop the session. The shared global store is
        // written by parallel tests too (see `src/session/mod.rs`), so the
        // claim is scoped to the agents this harness itself worked with: of
        // them only the two coder branches ever ran, so only they hold a
        // session — the reporter's deletion must take its own and nothing else
        // of the harness's.
        let owned = [
            assistant.clone(),
            crate::session::manager_agent_id(&project.name),
            role_branch.clone(),
            branch_agent.clone(),
            reporter.clone(),
        ];
        let before = deadline_poll(|| async {
            let listed: Vec<String> = crate::session::store()
                .list_sessions_with_metadata()
                .await
                .into_iter()
                .map(|s| s.agent_id)
                .collect();
            let owned_listed: Vec<String> = owned
                .iter()
                .filter(|id| listed.contains(*id))
                .cloned()
                .collect();
            (owned_listed.contains(&branch_agent) && owned_listed.contains(&reporter))
                .then_some(owned_listed)
        })
        .await
        .expect("the harness's own coder branches are listed before the delete");

        crate::jobs::abandon_session_jobs(&reporter).await.unwrap();
        crate::session::store().delete(&reporter).await.unwrap();
        let after = deadline_poll(|| async {
            let listed = crate::session::store().list_sessions_with_metadata().await;
            (!listed.iter().any(|s| s.agent_id == reporter)).then_some(listed)
        })
        .await
        .expect("the deleted branch must leave the session list");
        let disturbed: Vec<&String> = before
            .iter()
            .filter(|id| **id != reporter && !after.iter().any(|s| s.agent_id == **id))
            .collect();
        assert!(
            disturbed.is_empty(),
            "deleting one branch must not disturb the harness's other sessions: {disturbed:?}"
        );

        // ── a restricted holder is refused the kinds it does not hold ─────
        let deep_only =
            crate::tools::AnalyzeTool::granted(&[crate::tools::analyze::AnalyzeDepth::Deep]);
        // An absent choice is this holder's one kind; naming the other is refused.
        let ack = dispatch(&deep_only, &caller, json!({ "message": "deep one" }))
            .await
            .expect("a deep-only holder runs its one kind when no choice is named");
        assert!(ack.contains("Deep analysis dispatched"), "{ack}");
        let err = dispatch(
            &deep_only,
            &caller,
            json!({ "message": "quick one", "depth": "quick" }),
        )
        .await
        .expect_err("a deep-only holder has no quick kind");
        assert!(
            err.to_string().contains("`depth` must be one of deep"),
            "{err}"
        );
        let quick_only =
            crate::tools::AnalyzeTool::granted(&[crate::tools::analyze::AnalyzeDepth::Quick]);
        let err = dispatch(
            &quick_only,
            &caller,
            json!({ "message": "deep one", "depth": "deep" }),
        )
        .await
        .expect_err("a quick-only holder has no deep kind");
        assert!(
            err.to_string().contains("`depth` must be one of quick"),
            "{err}"
        );
        // A one-kind holder declares no choice at all.
        assert!(
            quick_only.parameters_schema()["properties"]
                .get("depth")
                .is_none()
        );
        assert!(
            crate::tools::AnalyzeTool::admin(Role::Assistant).parameters_schema()["properties"]["depth"]
                .get("enum")
                .is_some()
        );

        let coder_only = SendTool::granted(&[Addressee::Coder]);
        for addressee in ["coder_branch", "manager", "role"] {
            let err = dispatch(
                &coder_only,
                &caller,
                json!({ "message": "x", "addressee": addressee }),
            )
            .await
            .expect_err("a coder-only holder has no other addressee");
            assert!(
                err.to_string().contains("`addressee` must be one of coder"),
                "{err}"
            );
        }

        // Leave the shared global store as the harness found it: every dispatch
        // above persisted its durable envelope for the conversation it reached
        // (that IS the at-least-once guarantee), and no consumer outlives this
        // test to confirm them. Only the rows this harness created are removed —
        // scoped by the snapshot taken before the first dispatch, because the
        // shared store is written by parallel tests too, and one of them could
        // be holding an envelope for the very session ids used here.
        for row in crate::jobs::list_pending_jobs(&crate::session::store().conn)
            .await
            .expect("list pending")
        {
            if !pending_before.contains(&row.id) {
                crate::jobs::delete_pending_job(&crate::session::store().conn, &row.id)
                    .await
                    .expect("delete the harness's own envelope");
            }
        }
    }

    /// The addressees a holder's text may name: the choice declaration names
    /// every held addressee, an argument's text names exactly the held addressees
    /// that take it, and a holder that declares no choice reads no label at all.
    fn assert_labels_follow_the_choice(kinds: &[Addressee]) {
        let declared = kinds.len() > 1;
        let tool = SendTool::granted(kinds);
        let description = tool.description();
        for kind in Addressee::ALL {
            let labelled = format!("`{}`", kind.as_str());
            let named = declared && kinds.contains(&kind);
            assert_eq!(
                description.contains(&labelled),
                named,
                "{kinds:?} reads {labelled}: {description}"
            );
        }
        let schema = tool.parameters_schema();
        for (name, property) in schema["properties"].as_object().unwrap() {
            let text = property["description"].as_str().unwrap_or_default();
            let named: Vec<Addressee> = match (declared, name == Addressee::ARG) {
                // No choice is declared, so no label is read at all.
                (false, _) => Vec::new(),
                (true, true) => kinds.to_vec(),
                (true, false) => Argument::ALL
                    .iter()
                    .find(|arg| arg.as_str() == name)
                    .map(|arg| {
                        kinds
                            .iter()
                            .filter(|kind| kind.requirements().iter().any(|req| req.arg == *arg))
                            .copied()
                            .collect()
                    })
                    .unwrap_or_default(),
            };
            for kind in Addressee::ALL {
                let labelled = format!("`{}`", kind.as_str());
                assert_eq!(
                    text.contains(&labelled),
                    named.contains(&kind),
                    "{kinds:?} reads {labelled} in `{name}`: {text}"
                );
            }
        }
    }

    /// The arguments a holder declares: one where a held addressee takes it (and
    /// only then), with nothing required unconditionally while a choice is
    /// declared — a requirement belongs to the choice that needs it.
    fn assert_arguments_follow_the_addressees(kinds: &[Addressee]) {
        let declared = kinds.len() > 1;
        let schema = SendTool::granted(kinds).parameters_schema();
        let properties = schema["properties"].as_object().unwrap();
        let mut expected: Vec<&str> = vec!["message"];
        if declared {
            expected.push(Addressee::ARG);
        }
        for arg in Argument::ALL {
            let taken = kinds
                .iter()
                .any(|kind| kind.requirements().iter().any(|req| req.arg == arg));
            assert_eq!(
                properties.contains_key(arg.as_str()),
                taken,
                "{kinds:?} and `{}`",
                arg.as_str()
            );
            if taken {
                expected.push(arg.as_str());
            }
        }
        let mut present: Vec<&str> = properties.keys().map(String::as_str).collect();
        present.sort_unstable();
        expected.sort_unstable();
        assert_eq!(present, expected, "{kinds:?}");

        let required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        if declared {
            assert_eq!(required, ["message", Addressee::ARG], "{kinds:?}");
            return;
        }
        // Without a choice, what its one addressee cannot do without is
        // unconditional.
        let mut expected_required = vec!["message"];
        expected_required.extend(
            kinds[0]
                .requirements()
                .iter()
                .filter(|req| req.required)
                .map(|req| req.arg.as_str()),
        );
        assert_eq!(required, expected_required, "{kinds:?}");
    }

    /// How an argument's text states whether it can be left out: named for the
    /// choice that needs it, and never as the unconditional requirement of a
    /// holder that has a choice to satisfy instead.
    fn assert_requirements_are_named_per_choice(kinds: &[Addressee]) {
        let declared = kinds.len() > 1;
        let schema = SendTool::granted(kinds).parameters_schema();
        let properties = schema["properties"].as_object().unwrap();
        for arg in Argument::ALL {
            let Some(text) = properties
                .get(arg.as_str())
                .and_then(|property| property["description"].as_str())
            else {
                continue;
            };
            let needed: Vec<Addressee> = kinds
                .iter()
                .filter(|kind| {
                    kind.requirements()
                        .iter()
                        .any(|req| req.arg == arg && req.required)
                })
                .copied()
                .collect();
            if declared {
                assert!(
                    !text.ends_with(" Required."),
                    "{kinds:?}: `{}` is not unconditional: {text}",
                    arg.as_str()
                );
                for kind in needed {
                    let named = format!("required for `{}`", kind.as_str());
                    assert!(text.to_lowercase().contains(&named), "{kinds:?}: {text}");
                }
            } else {
                assert_eq!(
                    text.ends_with(" Required."),
                    !needed.is_empty(),
                    "{kinds:?}: {text}"
                );
            }
        }
    }

    /// What an arm may read directly with `get_str`: its addressee's required
    /// arguments must be declared for every holder that can reach that arm —
    /// `get_str` cannot filter, so an undeclared one there would be read even
    /// though the interface never offers it. Swept over every holder.
    fn assert_required_arguments_are_declared(kinds: &[Addressee]) {
        let tool = SendTool::granted(kinds);
        for kind in kinds {
            for requirement in kind.requirements() {
                assert!(
                    !requirement.required || tool.declares(requirement.arg),
                    "{kinds:?}: `{}` requires `{}`, which the holder must declare",
                    kind.as_str(),
                    requirement.arg.as_str()
                );
            }
        }
    }

    /// A holder reads only the choices and the requirements it has. Swept over
    /// EVERY holder and every argument (all non-empty subsets of
    /// [`Addressee::ALL`]), so the class cannot come back one instance at a time:
    /// a new addressee or argument joins the sweep by existing. The Engineer's
    /// sync coder — the one holder that declares no `place` — sits outside the
    /// subsets and is pinned by its own tests.
    #[test]
    fn a_holder_reads_only_the_choices_and_requirements_it_has() {
        let all = Addressee::ALL;
        for mask in 1u32..(1 << all.len()) {
            let kinds: Vec<Addressee> = all
                .iter()
                .enumerate()
                .filter(|(bit, _)| mask & (1 << bit) != 0)
                .map(|(_, kind)| *kind)
                .collect();
            assert_labels_follow_the_choice(&kinds);
            assert_arguments_follow_the_addressees(&kinds);
            assert_requirements_are_named_per_choice(&kinds);
            assert_required_arguments_are_declared(&kinds);
        }

        // The prose itself, once: a choice holder reads each held addressee under
        // its own label, and the two coder kinds share one note.
        let admin = SendTool::admin(Role::Assistant);
        assert!(admin.description().contains("`coder`: One coder sub-agent"));
        assert!(
            admin
                .description()
                .contains("`manager`: Deliver an internal message")
        );
        let place = admin.parameters_schema()["properties"]["place"]["description"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(place.contains("`coder`/`coder_branch`:"), "{place}");

        // The Engineer's own instance: a sync coder with no folder at all.
        let engineer = SendTool::engineer();
        assert!(engineer.description().contains("One coder sub-agent"));
        let schema = engineer.parameters_schema();
        assert_eq!(schema["required"], json!(["message"]));
        assert!(schema["properties"].get("addressee").is_none());
        assert!(schema["properties"].get("place").is_none());
    }

    /// The Engineer's sync `send` through the tool wrapper, end to end: the
    /// coder really runs, and its report comes back inline in the call's own
    /// result — never as the async `<coder-report>` envelope, and with no job
    /// number (nothing was left behind for a later delivery).
    #[tokio::test]
    #[serial_test::serial(provider, channel_registry, drain)] // serializes the process-global fake provider + channel registry + shutdown drain flag
    async fn engineer_send_returns_the_coder_report_inline() {
        use std::sync::Arc;

        crate::util::test::init_management_test_stores().await;
        let _policy =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let _provider = crate::util::test::install_fake_provider(Arc::new(
            crate::util::test::FakeProvider::new().ok("the coder's report"),
        ));

        let ws = crate::workspace::test_ws("/tmp/test_ws_send_engineer_sync");
        let result = dispatch(
            &SendTool::engineer(),
            &ws,
            json!({ "message": "do the thing" }),
        )
        .await
        .expect("the engineer's sync coder dispatch succeeds");

        assert!(
            result.contains("the coder's report"),
            "the coder's report returns inline: {result}"
        );
        assert!(
            !result.contains("<coder-report"),
            "the sync path returns the report itself, never the async envelope: {result}"
        );
        assert!(
            !result.contains("job "),
            "a sync coder dispatch names no job number: {result}"
        );

        // The engineer declares no `place`: a call naming one reads it as absent
        // (inert, never a refusal) and still runs in the caller's workspace.
        dispatch(
            &SendTool::engineer(),
            &ws,
            json!({ "message": "again", "place": "/no/such/folder" }),
        )
        .await
        .expect("an undeclared argument is inert for the holder that does not declare it");
    }

    /// A choice value the holder does not offer is refused, naming its own set —
    /// never silently narrowed.
    #[tokio::test]
    async fn unheld_addressee_values_are_refused() {
        let ws = crate::workspace::test_ws("/tmp/test_ws_send_refuse");
        let engineer = SendTool::engineer();
        let err = engineer
            .execute(&ws, json!({ "message": "do it", "addressee": "manager" }))
            .await
            .expect_err("the engineer addresses a coder only");
        let err = err.to_string();
        assert!(err.contains("`addressee` must be one of coder"), "{err}");

        let admin = SendTool::admin(Role::Assistant);
        let err = admin
            .execute(&ws, json!({ "message": "do it", "addressee": "bogus" }))
            .await
            .expect_err("an unknown addressee is refused");
        let err = err.to_string();
        assert!(err.contains("coder, coder_branch, manager, role"), "{err}");
    }

    /// Regression: the manager send stays fully internal — the
    /// `<assistant-message>` envelope routes to the workspace Manager
    /// (live route + durable pending_jobs copy), while NOTHING is mirrored
    /// into the workspace users' chat_history or their channel bindings.
    /// A workspace member is attached with a live spy-channel binding —
    /// exactly the setup the removed mirroring delivered to.
    #[tokio::test]
    #[serial_test::serial(channel_registry, drain)] // serializes the process-global channel registry + shutdown drain flag
    async fn manager_send_stays_internal() {
        use crate::util::UnwrapPoison;
        use std::sync::Arc;

        crate::util::test::init_management_test_stores().await;

        let ws =
            crate::util::test::create_test_workspace("/tmp/mahbot/ws_mirror", "ws_mirror").await;

        // A workspace member bound to a spy channel — the removed mirroring
        // would have persisted + transport-delivered for exactly this user.
        let store = crate::users::store();
        store.add_user("mirror_user").await.unwrap();
        store
            .set_selected_workspace("mirror_user", Some(&ws.name))
            .await
            .unwrap();
        store
            .bind_channel("mirror_user", "spy", "mirror_user")
            .await
            .unwrap();

        let (spy, sent) = crate::util::test::SpyChannel::new("spy");
        let registry = crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);
        registry.register(Arc::new(spy) as Arc<dyn crate::Channel>);

        // Register the Manager consumer so the routed envelope lands here
        // instead of spawning a live consumer loop (the spy unregisters itself
        // on drop). The deferred rows the dispatch persists are cleaned up below.
        let target = crate::session::manager_agent_id(&ws.name);
        let mut spy = Spy::register(&target);
        let pending_before: std::collections::HashSet<String> =
            crate::jobs::list_pending_jobs(&crate::session::store().conn)
                .await
                .unwrap()
                .into_iter()
                .map(|row| row.id)
                .collect();

        let before = chat_history_rows(&ws.name).await;

        let res = crate::agent::CURRENT_TOOL_USER_NAME
            .scope("mirror_user".to_string(), async {
                crate::agent::CURRENT_TOOL_AGENT_ID
                    .scope(Some("agent_mirror".to_string()), async {
                        SendTool::admin(Role::Assistant)
                            .execute(
                                &ws,
                                json!({
                                    "addressee": "manager",
                                    "place": &ws.name,
                                    "message": "delegated work",
                                }),
                            )
                            .await
                    })
                    .await
            })
            .await;
        assert!(res.is_ok(), "tool must succeed: {:#}", res.unwrap_err());

        // The envelope reached the Manager consumer.
        let job = tokio::time::timeout(std::time::Duration::from_secs(30), spy.next_job())
            .await
            .expect("the manager envelope must be routed within 30s");
        assert_eq!(
            job.kind,
            crate::agent::message_router::MessageKind::AgentMessage
        );
        assert_eq!(job.workspace_name, ws.name);
        assert!(job.content.contains("delegated work"));

        // A durable pending_jobs copy exists for boot resume.
        let pending = crate::jobs::list_pending_jobs(&crate::session::store().conn)
            .await
            .unwrap();
        assert!(
            pending
                .iter()
                .any(|p| p.target_agent_id == target && p.envelope.contains("delegated work")),
            "durable envelope copy expected in pending_jobs"
        );

        // Nothing mirrored into chat history or channel bindings.
        assert_eq!(chat_history_rows(&ws.name).await, before);
        assert!(
            sent.lock().unwrap_poison().is_empty(),
            "no channel delivery expected"
        );

        // Leave the shared store as the test found it: only the rows this test
        // created (a parallel test may be holding an envelope for this session).
        for row in pending {
            if !pending_before.contains(&row.id) {
                crate::jobs::delete_pending_job(&crate::session::store().conn, &row.id)
                    .await
                    .expect("delete this test's own envelope");
            }
        }
    }

    async fn chat_history_rows(workspace: &str) -> i64 {
        crate::session::store()
            .conn
            .query_row(
                "SELECT COUNT(*) FROM chat_history WHERE workspace = ?1",
                crate::db::params![workspace],
                |row| row.get::<i64>(0),
            )
            .await
            .unwrap()
    }
}
