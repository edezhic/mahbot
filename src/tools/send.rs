//! SendTool — hand one message to exactly one addressee.
//!
//! The addressees are the worker (a single worker run), a worker branch (a
//! continuable worker conversation), a project workspace's Manager, and an
//! administrator-defined custom role. Every holder declares the `addressee`
//! choice and every call states it — including a holder that can address only
//! one agent — and a value outside its set is refused.
//!
//! The one caller with no agent session behind it is the command line
//! (`mahbot send`): there the owner's own assistant is an address too — what an
//! unnamed address means — and every handover is durable BEFORE that caller is
//! answered, since it is gone the moment it returns.
//!
//! The Engineer's instance addresses a worker only and runs it synchronously —
//! the worker's report returns inline. The admin Assistant and a custom role
//! dispatch asynchronously: the worker reports later as a `<worker-report>`
//! message, a worker branch's turn delivers its answer into the conversation that
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
    /// The branch a worker branch or a custom role continues.
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

/// Exactly one addressee of a `send` call. [`Self::ALL`] is the dispatch tool's
/// own set — the order is the interface order, the narrowest kind first; the
/// command line's vocabulary leads with the owner's own assistant and lives in
/// [`Self::COMMAND_LINE`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Addressee {
    /// The owner's own assistant — the command line's default address, and one
    /// an agent is never offered (it is not in [`Self::ALL`]).
    Assistant,
    /// A single worker run: make one scoped change, report back.
    Worker,
    /// A continuable worker conversation: continue (or create) a named branch.
    WorkerBranch,
    /// The Manager of one registered project workspace.
    Manager,
    /// An administrator-defined custom role.
    Role,
}

/// What the command line shows for its `--addressee` flag, word for word. It
/// lives beside the vocabulary it describes rather than in the command line's own
/// registry, so the two cannot drift apart unnoticed: the spellings and the
/// REQUIRED argument of every address are checked against the table below by
/// `the_address_help_is_the_vocabulary_it_describes`.
pub(crate) const ADDRESSEE_FLAG_HELP: &str = "Who receives it: 'assistant' (your own \
     assistant — the same as naming none), 'worker' (one scoped worker run in the folder \
     `--place` names, your own workspace when it names none), 'worker_branch' (a \
     continuable worker conversation, needs `--branch`), 'manager' (the Manager of the \
     project `--place` names — what a script aiming at a project should use), or 'role' \
     (a custom role, needs `--name`). An address takes only its own flags: a flag the \
     address does not take is refused rather than ignored — `--branch` with 'worker', or \
     `--place` with 'assistant', fails the call";

impl Addressee {
    /// The argument naming the choice.
    pub const ARG: &'static str = "addressee";

    /// Every addressee the dispatch tool itself offers, in interface order
    /// (narrowest first). The vocabulary a GRANT is drawn from.
    pub const ALL: [Self; 4] = [Self::Worker, Self::WorkerBranch, Self::Manager, Self::Role];

    /// Every addressee the COMMAND LINE may name: the tool's own set plus the
    /// owner's own assistant, which leads it. The holder decides which of them
    /// it offers.
    pub const COMMAND_LINE: [Self; 5] = [
        Self::Assistant,
        Self::Worker,
        Self::WorkerBranch,
        Self::Manager,
        Self::Role,
    ];

    /// The wire spellings of [`Self::ALL`], in the same order.
    pub const STRS: &'static [&'static str] = &[
        Self::Worker.as_str(),
        Self::WorkerBranch.as_str(),
        Self::Manager.as_str(),
        Self::Role.as_str(),
    ];

    /// The wire spelling of this addressee.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Assistant => "assistant",
            Self::Worker => "worker",
            Self::WorkerBranch => "worker_branch",
            Self::Manager => "manager",
            Self::Role => "role",
        }
    }

    /// Parse a wire spelling — the whole vocabulary the command line may name;
    /// the holder decides which of them it offers. This is the address vocabulary
    /// of a CALL (the command line's); the vocabulary a GRANT is drawn from is the
    /// narrower [`Self::STRS`], and a stored grant entry is validated against that
    /// before it ever reaches here (see [`crate::tools::grants`]), so a custom role
    /// can never be granted the owner's own assistant.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Self::COMMAND_LINE.into_iter().find(|a| a.as_str() == value)
    }

    /// What this addressee does, in the choice declaration's own words.
    const fn does(self) -> &'static str {
        match self {
            Self::Assistant => "hands the message to your own assistant",
            Self::Worker => "runs one scoped change and reports back",
            Self::WorkerBranch => "continues a named worker branch",
            Self::Manager => "messages a project's Manager",
            Self::Role => "starts a custom role",
        }
    }

    /// The arguments this addressee takes besides the shared `message`.
    fn requirements(self) -> &'static [Requirement] {
        /// The worker's folder rule, shared by both worker kinds so their `place`
        /// text stays one entry.
        const OUTSIDE_PROJECTS: &str = "any directory outside the registered projects";
        match self {
            // The message is all it takes.
            Self::Assistant => &[],
            Self::Worker => &[Requirement {
                arg: Argument::Place,
                note: OUTSIDE_PROJECTS,
                required: false,
            }],
            Self::WorkerBranch => &[
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

/// One note under its label(s) as the model reads it. The single place a `send`
/// note gets its label.
fn labelled(labels: &[&str], note: &str) -> String {
    let names = labels
        .iter()
        .map(|label| format!("`{label}`"))
        .collect::<Vec<_>>()
        .join("/");
    format!("{names}: {note}")
}

pub struct SendTool {
    /// Controls how a worker run is dispatched (see [`DispatchMode`]).
    dispatch_mode: DispatchMode,
    /// The role of the calling agent. Used to route async results to the
    /// correct agent channel.
    pub caller_role: Role,
    /// The addressees this holder may use, in interface order. A single
    /// addressee is declared and required like any other.
    kinds: Vec<Addressee>,
    /// Whether the optional `place` argument is accepted. Every addressee takes
    /// a folder, so every async holder declares it; the Engineer's worker-only
    /// instance does not.
    accepts_place: bool,
}

impl SendTool {
    /// The Engineer's instance: a worker only, synchronously, no `place` — the
    /// worker's report returns inline in the call's result.
    #[must_use]
    pub fn engineer() -> Self {
        Self {
            dispatch_mode: DispatchMode::Sync,
            caller_role: Role::Engineer,
            kinds: vec![Addressee::Worker],
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
    /// takes the narrowest addressee (`worker`).
    #[must_use]
    pub fn granted(kinds: &[Addressee]) -> Self {
        let kinds = if kinds.is_empty() {
            vec![Addressee::Worker]
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

    /// Whether this call is the COMMAND LINE's — the owner acting through
    /// `mahbot send`, with no agent session behind it (see [`crate::cli`]). The
    /// command line marks its own call with
    /// [`CURRENT_TOOL_COMMAND_LINE`](crate::agent::CURRENT_TOOL_COMMAND_LINE),
    /// which only [`crate::cli::instance`] sets — nothing a call carries
    /// identifies the executor, because a message the command line hands over
    /// runs the addressee's own turn with the command line's channel in place,
    /// and that turn is an agent's own call like any other. This is the one
    /// surface where an unnamed address means the owner's own assistant and
    /// where that assistant may be named; an agent's call must state an
    /// address, so an agent session is never a command line.
    fn command_line() -> bool {
        crate::agent::tool_command_line()
    }

    /// The addressees THIS CALL may name: the holder's own inside an agent
    /// session, the command line's own vocabulary outside one. A holder's set is
    /// what its tool offers an agent; the command line takes the whole vocabulary,
    /// which adds the owner's own assistant — the one address an agent's holder
    /// never has. The CLI's own holder is the admin Assistant's `send`, whose set
    /// is already the tool's, so no holder the instance serves a call with can
    /// narrow this surface.
    fn addressed(&self) -> Vec<Addressee> {
        if Self::command_line() {
            Addressee::COMMAND_LINE.to_vec()
        } else {
            self.kinds.clone()
        }
    }

    /// The addressee the call must name: an absent (or explicit null) value is
    /// refused in the choice's own words instead of taking the narrowest kind, so
    /// no caller — not even a holder that can address only one agent — is ever
    /// handed a silent pick. `chosen_kind`'s absent-means-narrowest stays the rule
    /// where the choice is an optional NARROWING: `analyze`'s depth, and the
    /// tolerant arm an in-flight call frame reconnects through.
    fn required_addressee(args: &serde_json::Value, held: &[Addressee]) -> Result<Addressee> {
        if matches!(
            args.get(Addressee::ARG),
            None | Some(serde_json::Value::Null)
        ) {
            return Err(anyhow::anyhow!(
                "usage: `{}` is required — it must be one of {}",
                Addressee::ARG,
                held.iter()
                    .map(|addressee| addressee.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        super::chosen_kind(
            args,
            Addressee::ARG,
            held,
            Addressee::as_str,
            Addressee::parse,
        )
    }

    /// The addressee this call names. An agent's call MUST name one (the
    /// mandatory address); the command line's own rule is that an unnamed
    /// address means the owner's assistant — naming it is the same as naming
    /// none — and no other address is ever chosen silently.
    fn addressee(&self, args: &serde_json::Value) -> Result<Addressee> {
        // The command line's own rule first: its unnamed address means the owner's
        // own assistant, and naming it is the same as naming none.
        if Self::command_line()
            && matches!(
                args.get(Addressee::ARG),
                None | Some(serde_json::Value::Null)
            )
        {
            return Ok(Addressee::Assistant);
        }
        Self::required_addressee(args, &self.addressed())
    }

    /// Refuse an optional argument the chosen address does not take. An agent's
    /// call is built from its holder's own interface, so an argument it cannot
    /// use is inert by construction; a script names the address and the flags
    /// apart, and a combination that cannot mean anything must be refused rather
    /// than dispatched with the value dropped.
    fn refuse_unused(args: &serde_json::Value, addressee: Addressee) -> Result<()> {
        for arg in Argument::ALL {
            let given = args.get(arg.as_str()).is_some_and(|value| !value.is_null());
            let taken = addressee
                .requirements()
                .iter()
                .any(|requirement| requirement.arg == arg);
            if given && !taken {
                return Err(anyhow::anyhow!(
                    "usage: `{}` is not an argument of the `{}` address",
                    arg.as_str(),
                    addressee.as_str()
                ));
            }
        }
        Ok(())
    }

    /// Who answers for the handovers this call makes: the command line's caller
    /// is answered and gone the moment it returns, so its handovers are REFUSED
    /// when they could not be made durable; an agent's own turn is durable
    /// already, so its handovers are routed best-effort instead of dropped.
    fn answered_for() -> crate::agent::message_router::AnsweredFor {
        if Self::command_line() {
            crate::agent::message_router::AnsweredFor::CommandLine
        } else {
            crate::agent::message_router::AnsweredFor::Agent
        }
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
    /// worker-only instance runs where the caller is.
    async fn optional_place(&self, args: &serde_json::Value) -> Result<Option<Workspace>> {
        if self.declares(Argument::Place) {
            super::delegation::resolve(super::delegation::DelegationKind::Worker, args).await
        } else {
            Ok(None)
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
            .map(|(labels, requirement)| labelled(labels, requirement.note))
            .collect();
        let needed_by: Vec<String> = takers
            .iter()
            .filter(|(_, requirement)| requirement.required)
            .map(|(addressee, _)| format!("`{}`", addressee.as_str()))
            .collect();
        // Whether the argument can be left out follows from the choices this
        // holder has: only the choices that cannot do without it are named.
        let tail = match needed_by.len() {
            0 => " Omit for the default.".to_string(),
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
    /// note, one note per held addressee, and the async/sync note.
    fn description(&self) -> String {
        let mut parts = vec![
            load_prompt("tool/send.md"),
            load_prompt("tool/send_addressee.md"),
        ];
        for addressee in Addressee::ALL {
            if !self.kinds.contains(&addressee) {
                continue;
            }
            // The worker note is the one that differs by dispatch: the sync
            // holder returns the worker's report inline, the async holders get it
            // later as a `<worker-report>`.
            let note = match addressee {
                Addressee::Worker if !self.dispatch_mode.is_async() => {
                    load_prompt("tool/send_worker_sync.md")
                }
                other => load_prompt(&format!("tool/send_{}.md", other.as_str())),
            };
            parts.push(labelled(&[addressee.as_str()], &note));
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
        // Every optional argument the holder's addressees take, described from
        // the one per-addressee table — `place` included only where the holder
        // declares it: the Engineer's worker-only instance runs where the caller is.
        for arg in Argument::ALL {
            if let Some(description) = self.describe(arg) {
                properties[arg.as_str()] = json!({
                    "type": "string",
                    "description": description
                });
            }
        }
        // A one-addressee holder declares the choice like any other, so the single
        // addressee's own required arguments are stated as required because there
        // is no other addressee to choose between.
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

    #[expect(clippy::too_many_lines)]
    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let message = super::get_dispatch_message(&args)?;
        let addressee = self.addressee(&args)?;
        // The command line's own strictness: its caller names the address and
        // the flags apart, so a combination that cannot mean anything is refused
        // here rather than dispatched with the value silently dropped (see
        // [`Self::refuse_unused`]).
        if Self::command_line() {
            Self::refuse_unused(&args, addressee)?;
        }

        match addressee {
            Addressee::Assistant => {
                // Only the command line reaches this arm: the owner's own
                // assistant is not one of the addressees an agent's holder
                // declares (see [`Addressee::ALL`]), so an agent that names it
                // is refused before any dispatch.
                crate::agent::message_router::deliver_command_line_message(message.to_string())
                    .await?;
                Ok(
                    "Message handed to your assistant, which answers on the channels it \
                    normally uses."
                        .to_string(),
                )
            }
            Addressee::Worker => {
                if !self.dispatch_mode.is_async() {
                    // The Engineer's sync worker run: the report returns inline.
                    return super::SyncDurableCore::Worker
                        .run_sync_dispatch(ws, message, self.caller_role)
                        .await;
                }
                // A worker round writes outside the registered projects: only a
                // durable background round can honour the optional `place`.
                let place = self.optional_place(&args).await?;
                let job_id = crate::generate_id();
                if Self::command_line() {
                    // Answered and gone the moment this returns: the round's
                    // durable rows are written BEFORE the acknowledgement, and
                    // the round then runs as a resume of its own rows.
                    crate::tools::implement::hand_over_worker_round(
                        ws,
                        place.as_ref(),
                        message,
                        self.caller_role,
                        &job_id,
                    )
                    .await?;
                } else {
                    super::SyncDurableCore::Worker.spawn_dispatch(
                        ws,
                        place.as_ref(),
                        message,
                        self.caller_role,
                        job_id,
                    );
                }
                let folder = place
                    .as_ref()
                    .map_or_else(|| ws.path.clone(), |p| p.path.clone());
                Ok(format!(
                    "Work handed to a worker in {}; its report arrives as a <worker-report> message.",
                    folder.trim_end_matches(std::path::MAIN_SEPARATOR)
                ))
            }
            Addressee::WorkerBranch => {
                let branch = super::get_str(&args, Argument::Branch.as_str())?;
                // A branch is pointed at a folder by the same rule as a single
                // worker run — the branch may not be a way around the worker's
                // refusals.
                let place = self.optional_place(&args).await?;
                // Read the asker before the launch: the report returns to the
                // conversation that asked.
                let asker = crate::agent::tool_custom_launch();
                crate::launch::worker_branch(
                    ws,
                    branch,
                    message,
                    place.as_ref(),
                    asker,
                    Self::answered_for(),
                )
                .await
            }
            Addressee::Manager => {
                // The calling session's user is who the message is attributed
                // to. A project's Manager is reachable with no agent session
                // behind the call — the command line acts as the machine's
                // owner — so the user is all this arm needs (`tool_identity`
                // also demands a session id, which the arm never uses).
                let user_name = crate::agent::tool_user_name();
                anyhow::ensure!(
                    !user_name.is_empty(),
                    "No user identity available — a message to a project's Manager must run in a \
                     user session or on the command line."
                );
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
                    Self::answered_for(),
                )
                .await?;
                Ok(format!(
                    "Message delivered to the manager of workspace '{}'.",
                    project.name
                ))
            }
            Addressee::Role => {
                let name = super::get_str(&args, Argument::Name.as_str())?;
                // Read through the declaring filter like every other optional
                // argument. `place` here is NOT resolved the way a worker's is:
                // `launch::custom_role` accepts the personal-workspace spelling
                // too, which the worker's refusals do not.
                crate::launch::custom_role(
                    name,
                    message,
                    self.optional(&args, Argument::Branch)?,
                    self.optional(&args, Argument::Place)?,
                    Self::answered_for(),
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

    /// Run a dispatch tool the way a turn does: `channel` is the transport the
    /// message arrived over and `agent_id` the session driving the turn — `None`
    /// for no session. A turn is an agent's call whatever carried it, so the
    /// command line's own marker is NOT set here. Scoped outside any spawn, with
    /// the caller identity in the task-locals the tools read.
    async fn dispatch_over(
        channel: &str,
        agent_id: Option<&str>,
        tool: &dyn Tool,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        crate::agent::CURRENT_TOOL_USER_NAME
            .scope("admin".to_string(), async {
                crate::agent::CURRENT_TOOL_CHANNEL
                    .scope(channel.to_string(), async {
                        match agent_id {
                            Some(agent_id) => {
                                crate::agent::CURRENT_TOOL_AGENT_ID
                                    .scope(Some(agent_id.to_string()), async {
                                        tool.execute(ws, args).await
                                    })
                                    .await
                            }
                            None => tool.execute(ws, args).await,
                        }
                    })
                    .await
            })
            .await
    }

    /// The transport an agent's own turn arrived over in these tests — the channel
    /// the rounds such a turn dispatches carry.
    const TURN_CHANNEL: &str = "gui";

    /// A dispatch from an agent's own turn.
    async fn dispatch(
        tool: &dyn Tool,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        dispatch_over(TURN_CHANNEL, Some("agent_harness"), tool, ws, args).await
    }

    /// Run a dispatch tool the way the COMMAND LINE does: the marker only
    /// [`crate::cli::instance`] sets, with the owner's own identity and the command
    /// line's channel, and NO agent session behind it.
    async fn dispatch_command_line(
        tool: &dyn Tool,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        dispatch_command_line_as(crate::users::ADMIN_USER_NAME, tool, ws, args).await
    }

    /// The same call under the identity the caller carries — `""` for a command
    /// line that names no owner at all.
    async fn dispatch_command_line_as(
        user: &str,
        tool: &dyn Tool,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        crate::agent::CURRENT_TOOL_COMMAND_LINE
            .scope(true, async {
                crate::agent::CURRENT_TOOL_USER_NAME
                    .scope(user.to_string(), async {
                        crate::agent::CURRENT_TOOL_CHANNEL
                            .scope(crate::cli::CHANNEL.to_string(), async {
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
        /// analysis by its job id, a worker report by its `<worker-report>`
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

    /// The ids of the durable rounds of `kind` that exist right now, narrowed to
    /// the rounds ONE PRODUCER's call wrote (`channel` is the caller's own — see
    /// [`crate::jobs`]). The store is shared with parallel tests, so a test that
    /// must know its own rows narrows by provenance instead of taking the whole
    /// set.
    async fn job_ids_of_kind(kind: &str, channel: &str) -> Vec<String> {
        crate::session::store()
            .conn
            .query(
                "SELECT id FROM jobs WHERE kind = ?1 AND channel = ?2",
                crate::db::params![kind, channel],
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
    /// the round or reaches the addressee it names, a worker branch is created,
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
        // analysis apart from another (and from the worker reports below).
        //
        // The round this test started is stopped the way the Running Agents view
        // stops one, so no long-running investigation outlives the test.
        crate::research_cancel::cancel_research_run(&deep_job_id).await;

        // ── send worker (a single run) ─────────────────────────────────────
        let worker_jobs_before = job_ids_of_kind("implement", TURN_CHANNEL).await;
        let ack = dispatch(
            &send,
            &caller,
            json!({ "message": "write it", "addressee": "worker", "place": &folder }),
        )
        .await
        .expect("a worker dispatch succeeds");
        assert!(
            !ack.contains("job "),
            "a worker dispatch names no job number: {ack}"
        );
        // The round is proven by ITS OWN row — the new `implement` job carrying
        // this turn's channel — never by "some row of that kind exists" in the
        // shared store.
        let job_id = deadline_poll(|| async {
            job_ids_of_kind("implement", TURN_CHANNEL)
                .await
                .into_iter()
                .find(|id| !worker_jobs_before.contains(id))
        })
        .await
        .expect("a worker dispatch creates its own implement round");
        assert_eq!(round_kind(&job_id).await, "implement");
        let report = assistant_spy.report("<worker-report").await;
        assert!(
            report.content.contains("<worker-report") && !report.content.contains("Job:"),
            "a worker report carries no job number: {}",
            report.content
        );
        assert!(
            report.content.contains(&folder),
            "a worker report names the folder it worked in: {}",
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
            ("worker", json!({})),
            ("worker_branch", json!({ "branch": "typed" })),
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

        // ── send worker_branch: created, continued, moved, stopped ─────────
        let branch = "harness";
        let branch_agent = crate::session::worker_branch_agent_id(branch);
        let mut branch_spy = Spy::register(&branch_agent);
        let ack = dispatch(
            &send,
            &caller,
            json!({
                "message": "first",
                "addressee": "worker_branch",
                "branch": branch,
                "place": &folder,
            }),
        )
        .await
        .expect("a worker branch is creatable");
        assert!(!ack.contains("job "), "a worker branch names no job: {ack}");
        let first = branch_spy.next_job().await;
        assert_eq!(
            first.custom_role.as_ref().map(|c| c.role),
            Some(Role::Worker)
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
                "addressee": "worker_branch",
                "branch": branch,
                "place": &other_folder,
            }),
        )
        .await
        .expect("a worker branch is continuable");
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

        // A use that names no folder runs where a worker runs today.
        dispatch(
            &send,
            &caller,
            json!({ "message": "third", "addressee": "worker_branch", "branch": branch }),
        )
        .await
        .expect("a worker branch runs without a folder");
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
        let reporter = crate::session::worker_branch_agent_id("reporter");
        dispatch(
            &send,
            &caller,
            json!({
                "message": "report back",
                "addressee": "worker_branch",
                "branch": "reporter",
                "place": &folder,
            }),
        )
        .await
        .expect("a worker branch dispatches");
        let report = assistant_spy.report("branch=\"reporter\"").await;
        assert!(
            report.content.contains("<worker-report")
                && report.content.contains("branch=\"reporter\""),
            "a worker branch's report names its branch: {}",
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
        .expect("a worker branch is one session-list entry");
        assert_eq!(branch_row.role.as_deref(), Some("worker"));
        assert_eq!(branch_row.custom_identifier.as_deref(), Some("reporter"));

        // Deleting the branch exactly as the Sessions page does — abandon its
        // unfinished jobs, then drop the session. The shared global store is
        // written by parallel tests too (see `src/session/mod.rs`), so the
        // claim is scoped to the agents this harness itself worked with: of
        // them only the two worker branches ever ran, so only they hold a
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
        .expect("the harness's own worker branches are listed before the delete");

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

        let worker_only = SendTool::granted(&[Addressee::Worker]);
        for addressee in ["worker_branch", "manager", "role"] {
            let err = dispatch(
                &worker_only,
                &caller,
                json!({ "message": "x", "addressee": addressee }),
            )
            .await
            .expect_err("a worker-only holder has no other addressee");
            assert!(
                err.to_string()
                    .contains("`addressee` must be one of worker"),
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

    /// A provider whose every call parks until the test opens the gate, so a
    /// round's durable rows can be observed while the round cannot have finished.
    struct GatedProvider {
        open: tokio::sync::watch::Receiver<bool>,
    }

    #[async_trait::async_trait]
    impl crate::Provider for GatedProvider {
        async fn chat_scoped(
            &self,
            _request: crate::ChatRequest,
        ) -> Result<crate::ChatResponse, crate::providers::ScopedCallError> {
            let mut open = self.open.clone();
            while !*open.borrow_and_update() {
                if open.changed().await.is_err() {
                    break;
                }
            }
            Ok(crate::ChatResponse {
                text: Some("the work is done".to_string()),
                ..crate::ChatResponse::default()
            })
        }

        async fn warmup(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    /// A provider that parks every call until the test opens the gate and then
    /// panics: a round that dies in flight, at a moment the test chooses.
    struct DyingProvider {
        open: tokio::sync::watch::Receiver<bool>,
    }

    #[async_trait::async_trait]
    impl crate::Provider for DyingProvider {
        async fn chat_scoped(
            &self,
            _request: crate::ChatRequest,
        ) -> Result<crate::ChatResponse, crate::providers::ScopedCallError> {
            let mut open = self.open.clone();
            while !*open.borrow_and_update() {
                if open.changed().await.is_err() {
                    break;
                }
            }
            panic!("the provider blew up")
        }

        async fn warmup(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    /// The ids of the `implement` rounds a COMMAND-LINE call wrote: the rounds
    /// whose `channel` is the one a command-line caller carries
    /// ([`crate::cli::CHANNEL`]). An agent's own turn carries that same channel
    /// when it was triggered by a command-line message, so this narrows to a
    /// command-line CALL only while no such turn runs — which holds for every
    /// test here, whose conversations spies consume.
    async fn command_line_round_ids() -> Vec<String> {
        job_ids_of_kind("implement", crate::cli::CHANNEL).await
    }

    /// How many deferred envelopes waiting for `agent_id` carry `content` — the
    /// handover whose durability is being pinned, told apart from a parallel
    /// test's row for the same conversation by the message itself.
    async fn pending_for_content(agent_id: &str, content: &str) -> usize {
        crate::session::store()
            .conn
            .query(
                "SELECT envelope FROM pending_jobs WHERE target_agent_id = ?1",
                crate::db::params![agent_id],
            )
            .await
            .unwrap()
            .into_iter()
            .filter(|row| row.get::<String>(0).unwrap().contains(content))
            .count()
    }

    /// The messages the command-line handovers of
    /// `the_command_lines_handovers_are_durable_before_the_acknowledgement` carry.
    /// Each dispatch and the assertion on what that handover must have left
    /// behind read the SAME definition, so the two cannot drift apart.
    const UNNAMED_ASSISTANT_TEXT: &str = "hello";
    const NAMED_ASSISTANT_TEXT: &str = "hello again";
    const MANAGER_TEXT: &str = "from the command line";
    const BRANCH_TEXT: &str = "carry on";
    const ROLE_TEXT: &str = "take this";

    /// The ids of the owner's own conversation rows holding any of `contents` —
    /// what a command-line handover to the assistant persists through
    /// [`crate::channels::broadcast_and_persist_incoming_message`].
    async fn owner_message_ids(contents: &[&str]) -> Vec<String> {
        let workspace = crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME);
        let entries = crate::channels::chat_history::store()
            .load_conversation(crate::users::ADMIN_USER_NAME, &workspace)
            .await
            .expect("the owner's own conversation loads");
        entries
            .iter()
            .filter(|entry| {
                entry.direction == crate::ChatDirection::User
                    && contents.contains(&entry.content.as_str())
            })
            .map(|entry| entry.message_id.clone())
            .collect()
    }

    /// The command line is answered and GONE the moment a call returns, so every
    /// handover it makes must be durable BEFORE the acknowledgement — and the
    /// address it names must describe where a restart picks the work up. Each
    /// durability claim below is checked IMMEDIATELY after `execute` returns: no
    /// `deadline_poll` is allowed for the pin, because its whole point is that
    /// the durable record exists when (not some time after) the caller is
    /// answered. Every provider call the `send` round makes parks until this test
    /// opens the gate, so "immediately" cannot race the round's own progress.
    #[tokio::test]
    #[serial_test::serial(provider, channel_registry, drain)]
    async fn the_command_lines_handovers_are_durable_before_the_acknowledgement() {
        use std::sync::Arc;

        crate::util::test::init_management_test_stores().await;
        let _policy =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // Every provider call the round makes parks until the gate opens below, so
        // the round CANNOT have finished when the durability pin reads the store.
        let (gate_tx, gate_rx) = tokio::sync::watch::channel(false);
        let _provider =
            crate::util::test::install_fake_provider(Arc::new(GatedProvider { open: gate_rx }));
        crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);

        let project_dir = tempfile::tempdir().unwrap();
        let project_path =
            crate::util::strip_verbatim_prefix(&std::fs::canonicalize(project_dir.path()).unwrap())
                .to_str()
                .unwrap()
                .to_string();
        let project =
            crate::util::test::create_test_workspace(&project_path, "ws_cli_durable").await;
        let home = tempfile::tempdir().unwrap();
        // The command line's own workspace: the admin's personal one, where a
        // handover that names no folder runs.
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

        crate::custom_role::create(
            "cli_role",
            "you do the work",
            &[],
            crate::custom_role::CustomRoleBehavior::Continuous,
        )
        .await
        .unwrap();

        // Every conversation the command line's handovers reach is a registered
        // spy consumer, so nothing is left unconsumed — and the worker round the
        // resume path really runs reports back into the assistant's session.
        let assistant = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut assistant_spy = Spy::register(&assistant);
        let manager_agent = crate::session::manager_agent_id(&project.name);
        let mut manager_spy = Spy::register(&manager_agent);
        let _branch_spy = Spy::register(&crate::session::worker_branch_agent_id("cli_branch"));
        let role_agent = crate::session::custom_role_agent_id("cli_role", None);
        let _role_spy = Spy::register(&role_agent);

        // The deferred envelopes that already exist: cleanup removes only the
        // rows this harness creates, because the shared global store is written
        // by parallel tests too (see `src/session/mod.rs`).
        let pending_before: std::collections::HashSet<String> =
            crate::jobs::list_pending_jobs(&crate::session::store().conn)
                .await
                .expect("list pending")
                .into_iter()
                .map(|row| row.id)
                .collect();
        // The conversation rows the assistant handovers below persist, likewise
        // snapshotted: the store is shared, so only the rows that appear with
        // these texts are this harness's to reclaim.
        let history_before: Vec<String> =
            owner_message_ids(&[UNNAMED_ASSISTANT_TEXT, NAMED_ASSISTANT_TEXT]).await;

        let send = SendTool::admin(Role::Assistant);

        // ── the owner's own assistant, unnamed and named ──────────────────
        // Each pin is scoped to the handover it is about — this conversation AND
        // this message's own text — never to a count over the shared store, which
        // a parallel test's row for the same conversation could satisfy. The count
        // is read before and after rather than asserted to be exactly one: a
        // leftover row from an earlier run is not this handover's business.
        let before = pending_for_content(&assistant, UNNAMED_ASSISTANT_TEXT).await;
        let ack =
            dispatch_command_line(&send, &caller, json!({ "message": UNNAMED_ASSISTANT_TEXT }))
                .await
                .expect("an unnamed command-line address means the owner's assistant");
        assert!(
            ack.contains("assistant") && !ack.contains("<worker-report"),
            "the acknowledgement names the assistant: {ack}"
        );
        assert!(
            pending_for_content(&assistant, UNNAMED_ASSISTANT_TEXT).await > before,
            "the unnamed handover is durable when its caller is answered"
        );

        let before = pending_for_content(&assistant, NAMED_ASSISTANT_TEXT).await;
        let ack = dispatch_command_line(
            &send,
            &caller,
            json!({ "message": NAMED_ASSISTANT_TEXT, "addressee": "assistant" }),
        )
        .await
        .expect("naming the assistant is the same as naming none");
        assert!(
            ack.contains("assistant") && !ack.contains("<worker-report"),
            "{ack}"
        );
        assert!(
            pending_for_content(&assistant, NAMED_ASSISTANT_TEXT).await > before,
            "the named handover is durable when its caller is answered"
        );

        // ── the worker: the round's own row exists before the ack ──────────
        let worker_jobs_before = command_line_round_ids().await;
        let ack = dispatch_command_line(
            &send,
            &caller,
            json!({ "message": "go", "addressee": "worker", "place": &folder }),
        )
        .await
        .expect("a command-line worker dispatch succeeds");
        assert!(
            ack.contains("<worker-report"),
            "a worker dispatch names its report: {ack}"
        );
        let worker_jobs_after = command_line_round_ids().await;
        assert!(
            worker_jobs_after
                .iter()
                .any(|id| !worker_jobs_before.contains(id)),
            "the round's own row must exist the moment the command line is answered \
             (the durability pin, no poll): before={worker_jobs_before:?} after={worker_jobs_after:?}"
        );
        // The round really runs through the RESUME path: its report arrives in
        // the assistant's conversation and names the folder it worked in, which
        // proves the row asserted above is THIS round's. Opening the gate is what
        // lets it run at all.
        gate_tx
            .send(true)
            .expect("the gated provider holds the gate");
        let report = assistant_spy.report("<worker-report").await;
        assert!(
            report.content.contains(&folder),
            "the worker report names the folder the round ran in: {}",
            report.content
        );

        // ── a project's Manager ───────────────────────────────────────────
        let before = pending_for_content(&manager_agent, MANAGER_TEXT).await;
        let ack = dispatch_command_line(
            &send,
            &caller,
            json!({
                "message": MANAGER_TEXT,
                "addressee": "manager",
                "place": &project.name,
            }),
        )
        .await
        .expect("the command line reaches a project's Manager");
        assert!(
            ack.contains(&project.name),
            "the acknowledgement names the workspace: {ack}"
        );
        assert!(
            pending_for_content(&manager_agent, MANAGER_TEXT).await > before,
            "the Manager handover is durable when its caller is answered"
        );
        let delivered = manager_spy.report("<assistant-message").await;
        assert!(
            delivered.content.contains("<assistant-message"),
            "{}",
            delivered.content
        );

        // ── a worker branch ────────────────────────────────────────────────
        let branch_agent = crate::session::worker_branch_agent_id("cli_branch");
        let before = pending_for_content(&branch_agent, BRANCH_TEXT).await;
        dispatch_command_line(
            &send,
            &caller,
            json!({
                "message": BRANCH_TEXT,
                "addressee": "worker_branch",
                "branch": "cli_branch",
            }),
        )
        .await
        .expect("the command line reaches a worker branch");
        assert!(
            pending_for_content(&branch_agent, BRANCH_TEXT).await > before,
            "the worker branch launch is durable when its caller is answered"
        );

        // ── a custom role ─────────────────────────────────────────────────
        let before = pending_for_content(&role_agent, ROLE_TEXT).await;
        dispatch_command_line(
            &send,
            &caller,
            json!({ "message": ROLE_TEXT, "addressee": "role", "name": "cli_role" }),
        )
        .await
        .expect("the command line reaches a custom role");
        assert!(
            pending_for_content(&role_agent, ROLE_TEXT).await > before,
            "the custom-role launch is durable when its caller is answered"
        );

        // Leave the shared global store as the harness found it: the deferred
        // envelopes every handover above persisted are deleted, scoped by the
        // snapshot taken before the first dispatch. The worker round's own job row
        // needs no cleanup: the resume path terminalizes it as the report is
        // delivered, which the assertion above already waited for.
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
        // The two handovers to the assistant are the ones that reach a
        // conversation, and each left its copy of the owner's own message behind.
        // A leftover row is not inert: it is a message from the owner no turn was
        // ever produced for. Asserted before it is reclaimed — a reclaim that finds
        // nothing reclaims nothing.
        let dirty: Vec<String> = owner_message_ids(&[UNNAMED_ASSISTANT_TEXT, NAMED_ASSISTANT_TEXT])
            .await
            .into_iter()
            .filter(|id| !history_before.contains(id))
            .collect();
        assert_eq!(
            dirty.len(),
            2,
            "each assistant handover persists the owner's own message"
        );
        for id in dirty {
            crate::channels::chat_history::store()
                .conn
                .execute(
                    "DELETE FROM chat_history WHERE message_id = ?1",
                    crate::db::params![id],
                )
                .await
                .expect("reclaim the conversation row");
        }
    }

    /// A round that dies in flight reports its failure where its report would have
    /// gone. The command line is answered and gone the moment a handover returns,
    /// so a failure nobody is told about would leave work the caller was told had
    /// been handed over with nothing to notice it by until the next start — and a
    /// row left launched would have every boot retry a round that dies the same
    /// way.
    #[tokio::test]
    #[serial_test::serial(provider, channel_registry, drain)]
    async fn a_round_that_dies_in_flight_reports_its_failure() {
        use std::sync::Arc;

        crate::util::test::init_management_test_stores().await;
        let _policy =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // The round parks at its first call until the gate opens below: the row the
        // ack promises is therefore still there to be read, and is provably THIS
        // round's.
        let (gate_tx, gate_rx) = tokio::sync::watch::channel(false);
        let _provider =
            crate::util::test::install_fake_provider(Arc::new(DyingProvider { open: gate_rx }));
        crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);

        let home = tempfile::tempdir().unwrap();
        let caller = crate::workspace::test_ws_named(
            home.path().to_str().unwrap(),
            &crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME),
        );
        let assistant = crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME);
        let mut assistant_spy = Spy::register(&assistant);
        let rounds_before = command_line_round_ids().await;

        let ack = dispatch_command_line(
            &SendTool::admin(Role::Assistant),
            &caller,
            json!({ "message": "go", "addressee": "worker" }),
        )
        .await
        .expect("the handover itself succeeds — the round it starts is what dies");
        assert!(ack.contains("<worker-report"), "{ack}");
        let mine: Vec<String> = command_line_round_ids()
            .await
            .into_iter()
            .filter(|id| !rounds_before.contains(id))
            .collect();
        assert_eq!(
            mine.len(),
            1,
            "the ack means the round's own row is down: {mine:?}"
        );

        gate_tx
            .send(true)
            .expect("the dying provider holds the gate");
        let report = assistant_spy.report("<worker-report").await;
        assert!(
            report.content.contains("panicked"),
            "the failure must reach the conversation the report would have: {}",
            report.content
        );

        // Terminalized by that report, never left launched for the next boot to
        // retry it.
        let left = command_line_round_ids().await;
        assert!(
            !left.iter().any(|id| mine.contains(id)),
            "the panicked round's row must not stay launched for the next boot: {left:?}"
        );
        // Reclaim the deferred envelope it delivered — by this round's own id,
        // because the store is shared with parallel tests whose rows are none of
        // this harness's business.
        for id in &mine {
            crate::jobs::delete_pending_job(&crate::session::store().conn, id)
                .await
                .expect("delete the harness's own envelope");
        }
    }

    /// The addressees a holder's text may name: the choice declaration names
    /// every held addressee, and an argument's text names exactly the held
    /// addressees that take it.
    fn assert_labels_follow_the_choice(kinds: &[Addressee]) {
        let tool = SendTool::granted(kinds);
        let description = tool.description();
        for kind in Addressee::ALL {
            let labelled = format!("`{}`", kind.as_str());
            let named = kinds.contains(&kind);
            assert_eq!(
                description.contains(&labelled),
                named,
                "{kinds:?} reads {labelled}: {description}"
            );
        }
        let schema = tool.parameters_schema();
        for (name, property) in schema["properties"].as_object().unwrap() {
            let text = property["description"].as_str().unwrap_or_default();
            let named: Vec<Addressee> = if name == Addressee::ARG {
                kinds.to_vec()
            } else {
                Argument::ALL
                    .iter()
                    .find(|arg| arg.as_str() == name)
                    .map(|arg| {
                        kinds
                            .iter()
                            .filter(|kind| kind.requirements().iter().any(|req| req.arg == *arg))
                            .copied()
                            .collect()
                    })
                    .unwrap_or_default()
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

    /// The arguments a holder declares: `message` and the choice always, and one
    /// more where a held addressee takes it (and only then) — a requirement
    /// belongs to the choice that needs it.
    fn assert_arguments_follow_the_addressees(kinds: &[Addressee]) {
        let schema = SendTool::granted(kinds).parameters_schema();
        let properties = schema["properties"].as_object().unwrap();
        let mut expected: Vec<&str> = vec!["message", Addressee::ARG];
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
        let mut expected_required = vec!["message", Addressee::ARG];
        // With one addressee there is no other choice to make, so its own
        // required arguments stand.
        if let [only] = kinds {
            expected_required.extend(
                only.requirements()
                    .iter()
                    .filter(|req| req.required)
                    .map(|req| req.arg.as_str()),
            );
        }
        assert_eq!(required, expected_required, "{kinds:?}");
    }

    /// How an argument's text states whether it can be left out: named for the
    /// choice that needs it, never as an unconditional requirement.
    fn assert_requirements_are_named_per_choice(kinds: &[Addressee]) {
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
            assert!(
                !text.ends_with(" Required."),
                "{kinds:?}: `{}` is not unconditional: {text}",
                arg.as_str()
            );
            for kind in needed {
                let named = format!("required for `{}`", kind.as_str());
                assert!(text.to_lowercase().contains(&named), "{kinds:?}: {text}");
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
    /// sync worker — the one holder that declares no `place` — sits outside the
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
        // its own label, and the two worker kinds share one note.
        let admin = SendTool::admin(Role::Assistant);
        assert!(
            admin
                .description()
                .contains("`worker`: One worker sub-agent")
        );
        assert!(
            admin
                .description()
                .contains("`manager`: Deliver an internal message")
        );
        let place = admin.parameters_schema()["properties"]["place"]["description"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(place.contains("`worker`/`worker_branch`:"), "{place}");

        // The Engineer's own instance: a sync worker that declares the choice
        // like any other holder, and no folder at all.
        let engineer = SendTool::engineer();
        assert!(engineer.description().contains("One worker sub-agent"));
        assert!(
            engineer
                .description()
                .contains(load_prompt("tool/send_addressee.md").as_str()),
            "the addressee-required note is read by every holder: {}",
            engineer.description()
        );
        let schema = engineer.parameters_schema();
        assert_eq!(schema["required"], json!(["message", "addressee"]));
        assert_eq!(schema["properties"]["addressee"]["enum"], json!(["worker"]));
        assert!(schema["properties"].get("place").is_none());
    }

    /// The Engineer's sync `send` through the tool wrapper, end to end: the
    /// worker really runs, and its report comes back inline in the call's own
    /// result — never as the async `<worker-report>` envelope, and with no job
    /// number (nothing was left behind for a later delivery).
    #[tokio::test]
    #[serial_test::serial(provider, channel_registry, drain)] // serializes the process-global fake provider + channel registry + shutdown drain flag
    async fn engineer_send_returns_the_worker_report_inline() {
        use std::sync::Arc;

        crate::util::test::init_management_test_stores().await;
        let _policy =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let _provider = crate::util::test::install_fake_provider(Arc::new(
            crate::util::test::FakeProvider::new().ok("the worker's report"),
        ));

        let ws = crate::workspace::test_ws("/tmp/test_ws_send_engineer_sync");
        let result = dispatch(
            &SendTool::engineer(),
            &ws,
            json!({ "message": "do the thing", "addressee": "worker" }),
        )
        .await
        .expect("the engineer's sync worker dispatch succeeds");

        assert!(
            result.contains("the worker's report"),
            "the worker's report returns inline: {result}"
        );
        assert!(
            !result.contains("<worker-report"),
            "the sync path returns the report itself, never the async envelope: {result}"
        );
        assert!(
            !result.contains("job "),
            "a sync worker dispatch names no job number: {result}"
        );

        // The engineer declares no `place`: a call naming one reads it as absent
        // (inert, never a refusal) and still runs in the caller's workspace.
        dispatch(
            &SendTool::engineer(),
            &ws,
            json!({ "message": "again", "addressee": "worker", "place": "/no/such/folder" }),
        )
        .await
        .expect("an undeclared argument is inert for the holder that does not declare it");

        // The address is never assumed, not even for the one-addressee holder.
        let err = dispatch(&SendTool::engineer(), &ws, json!({ "message": "again" }))
            .await
            .expect_err("a call that names no addressee is refused");
        let err = err.to_string();
        assert!(
            err.contains("`addressee`") && err.to_lowercase().contains("required"),
            "{err}"
        );
    }

    /// The three spellings of the address vocabulary are one list written three
    /// ways: the grant vocabulary ([`Addressee::STRS`]) is [`Addressee::ALL`]'s
    /// spellings, the command line's ([`Addressee::COMMAND_LINE`]) is that list led
    /// by the owner's own assistant, and `parse` reads exactly the command line's.
    /// A drift between them would mis-scope a grant (the assistant is not
    /// grantable) or leave an accepted address out of the declarations.
    #[test]
    fn the_address_vocabularies_are_one_list() {
        let all: Vec<&str> = Addressee::ALL.iter().map(|a| a.as_str()).collect();
        assert_eq!(all, Addressee::STRS.to_vec(), "the grant vocabulary");
        let mut command_line: Vec<&str> = vec![Addressee::Assistant.as_str()];
        command_line.extend(all.iter().copied());
        assert_eq!(
            command_line,
            Addressee::COMMAND_LINE
                .iter()
                .map(|a| a.as_str())
                .collect::<Vec<_>>(),
            "the command line's vocabulary"
        );
        for addressee in Addressee::COMMAND_LINE {
            assert_eq!(
                Addressee::parse(addressee.as_str()),
                Some(addressee),
                "'{}' must parse back to itself",
                addressee.as_str()
            );
        }
        assert_eq!(Addressee::parse("bogus"), None);
    }

    /// The help the command line shows for `--addressee` is the vocabulary's own
    /// words, and it must stay that way: it names every address the command line
    /// takes AND every argument each address REQUIRES, so a new address — or a new
    /// required argument on an existing one — cannot ship with the help silent
    /// about it. (The command line's registry ties flags to the tool's ARGUMENTS,
    /// never to the vocabulary those arguments accept.)
    #[test]
    fn the_address_help_is_the_vocabulary_it_describes() {
        let help = ADDRESSEE_FLAG_HELP;
        for addressee in Addressee::COMMAND_LINE {
            let spelling = format!("'{}'", addressee.as_str());
            assert!(
                help.contains(&spelling),
                "the help must name the {spelling} address: {help}"
            );
            for requirement in addressee
                .requirements()
                .iter()
                .filter(|requirement| requirement.required)
            {
                let flag = format!("--{}", requirement.arg.as_str().replace('_', "-"));
                assert!(
                    help.contains(&flag),
                    "the help must say the {spelling} address needs {flag}: {help}"
                );
            }
        }
    }

    /// The command line's own strictness: a script names the address and the flags
    /// apart, so an argument the named address does not take is refused rather
    /// than dispatched with the value silently dropped. An agent's call is built
    /// from its holder's own interface — where the combination cannot arise — and
    /// an argument it cannot use stays inert there, as it always is.
    #[tokio::test]
    async fn a_command_line_flag_the_address_does_not_take_is_refused() {
        let ws = crate::workspace::test_ws("/tmp/test_ws_send_unused");
        let send = SendTool::admin(Role::Assistant);
        for (addressee, flag) in [
            ("assistant", "place"),
            ("assistant", "branch"),
            ("worker", "branch"),
            ("worker", "name"),
            ("manager", "name"),
        ] {
            let mut args = json!({ "message": "go", "addressee": addressee });
            args[flag] = json!(if flag == "place" { "/tmp" } else { "x" });
            let err = dispatch_command_line(&send, &ws, args)
                .await
                .expect_err("an argument the address does not take is refused")
                .to_string();
            assert!(
                err.contains(flag) && err.contains(addressee),
                "the refusal names the argument and the address: {err}"
            );
        }
    }

    /// A choice value the holder does not offer is refused, naming its own set —
    /// never silently narrowed. The calls run in an agent session: on the command
    /// line the vocabulary is the command line's own (see [`Addressee::COMMAND_LINE`]).
    #[tokio::test]
    async fn unheld_addressee_values_are_refused() {
        let ws = crate::workspace::test_ws("/tmp/test_ws_send_refuse");
        let engineer = SendTool::engineer();
        let err = dispatch(
            &engineer,
            &ws,
            json!({ "message": "do it", "addressee": "manager" }),
        )
        .await
        .expect_err("the engineer addresses a worker only");
        let err = err.to_string();
        assert!(err.contains("`addressee` must be one of worker"), "{err}");

        let admin = SendTool::admin(Role::Assistant);
        let err = dispatch(
            &admin,
            &ws,
            json!({ "message": "do it", "addressee": "bogus" }),
        )
        .await
        .expect_err("an unknown addressee is refused");
        let err = err.to_string();
        assert!(
            err.contains("worker, worker_branch, manager, role"),
            "{err}"
        );

        let err = dispatch(
            &admin,
            &ws,
            json!({ "message": "do it", "addressee": "assistant" }),
        )
        .await
        .expect_err("an agent session never names the owner's own assistant");
        let err = err.to_string();
        assert!(
            err.contains("worker, worker_branch, manager, role"),
            "{err}"
        );
    }

    /// What decides is the marker the executor sets: a turn that ARRIVED over the
    /// command line's channel — the assistant answering a `mahbot send`, whose
    /// envelope carries the tag — is an agent's own call, so it must state its
    /// address and can never name the owner's own assistant. The tag is a
    /// transport, not a caller.
    #[tokio::test]
    async fn a_command_line_delivered_turn_is_still_an_agents_own() {
        let ws = crate::workspace::test_ws("/tmp/test_ws_send_cli_turn");
        let admin = SendTool::admin(Role::Assistant);
        let err = dispatch_over(
            crate::cli::CHANNEL,
            Some("agent_harness"),
            &admin,
            &ws,
            json!({ "message": "do it" }),
        )
        .await
        .expect_err("an agent's turn states its address whatever channel it arrived over");
        let err = err.to_string();
        assert!(
            err.contains("`addressee`") && err.to_lowercase().contains("required"),
            "{err}"
        );

        let err = dispatch_over(
            crate::cli::CHANNEL,
            Some("agent_harness"),
            &admin,
            &ws,
            json!({ "message": "do it", "addressee": "assistant" }),
        )
        .await
        .expect_err("an agent's turn never names the owner's own assistant");
        let err = err.to_string();
        assert!(
            err.contains("worker, worker_branch, manager, role"),
            "{err}"
        );
    }

    /// The address is never assumed: an agent's call that names no addressee is
    /// refused by every holder, even one that can address only a single agent.
    /// The refusal happens before any dispatch, so no store or provider is needed.
    /// (The command line's own rule differs: there an unnamed address means the
    /// owner's assistant.)
    #[tokio::test]
    async fn a_missing_addressee_is_refused_for_every_holder() {
        let ws = crate::workspace::test_ws("/tmp/test_ws_send_required");
        for holder in [
            SendTool::admin(Role::Assistant),
            SendTool::engineer(),
            SendTool::granted(&[Addressee::Worker]),
            SendTool::granted(&[Addressee::Manager]),
        ] {
            let err = dispatch(&holder, &ws, json!({ "message": "do it" }))
                .await
                .expect_err("a call that names no addressee is refused");
            let err = err.to_string();
            assert!(
                err.contains("`addressee`") && err.to_lowercase().contains("required"),
                "{err}"
            );
        }
    }

    /// The command line's handover carries the owner it acts for: a call that
    /// names none is refused rather than attributed to nobody and addressed to a
    /// `personal:` workspace that does not exist. (The command line itself always
    /// scopes the owner, so this pins the contract, as the Manager arm's own check
    /// does.)
    #[tokio::test]
    async fn a_command_line_handover_without_an_owner_is_refused() {
        let ws = crate::workspace::test_ws("/tmp/test_ws_send_ownerless");
        let err = dispatch_command_line_as(
            "",
            &SendTool::admin(Role::Assistant),
            &ws,
            json!({ "message": "no owner" }),
        )
        .await
        .expect_err("a handover that names no owner is refused");
        assert!(err.to_string().contains("No user identity"), "{err}");
    }

    /// The resume tolerance the required address depends on: an in-flight `send`
    /// frame that predates the choice still reconnects to its unfinished sync
    /// worker round instead of being refused or restarted. A frame naming another
    /// addressee stays unresolvable and is re-executed, as today.
    #[test]
    fn an_absent_address_still_resumes_the_sync_worker() {
        use crate::tools::SyncDurableCore;
        assert!(matches!(
            SyncDurableCore::from_call("send", &json!({ "message": "m" })),
            Some(SyncDurableCore::Worker)
        ));
        assert!(matches!(
            SyncDurableCore::from_call("send", &json!({ "message": "m", "addressee": "worker" })),
            Some(SyncDurableCore::Worker)
        ));
        assert!(
            SyncDurableCore::from_call("send", &json!({ "message": "m", "addressee": "manager" }))
                .is_none()
        );
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
