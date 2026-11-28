//! SendTool — hand one message to exactly one addressee.
//!
//! The addressees are the worker, a worker branch, a managed workspace's Manager, a custom
//! role, and — for the owner's Assistant alone — a guest account's own assistant, a one-way
//! handover nothing answers. Every holder declares the `addressee` choice and every call states
//! it; a value outside the holder's set is refused.
//!
//! The command line (`mahbot send`) is the one caller with no agent session behind it: there the
//! owner's own assistant is an address too (what an unnamed address means), and every handover is
//! durable BEFORE that caller is answered.
//!
//! The Engineer's instance addresses a worker only and runs it synchronously. The admin Assistant
//! and a custom role dispatch asynchronously: a `<worker-report>`, a worker branch's answer into
//! the asking conversation, an `<manager-message>`, a `<custom-role-message>`.
//!
//! The owner's Assistant (and its command-line twin) may also chain the call with `for`: the
//! addressee's answer then goes straight on to the next addressee instead of coming back, so one
//! call runs a whole sequence and only the final answer arrives ([`crate::tools::chain`]). The
//! same holders may ask for the answer as JSON with `extract`, which replaces it
//! ([`crate::extract`]). No other holder declares either argument.
//!
//! Every call that would start a worker or a custom role first refuses a folder any other writer
//! already holds: every role but the Manager counts, the admin Assistant included, and the refusal
//! names which (see [`crate::launch::FolderHeld`]).

use crate::prompt::load_prompt;
use crate::tools::Tool;
use crate::tools::analyze::DispatchMode;
use crate::{Role, Workspace};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// One of `send`'s optional arguments — the shared `message` is not one.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Argument {
    /// The folder the addressee works in (which project a `manager` call messages).
    Place,
    /// The branch a worker branch or a custom role continues.
    Branch,
    /// The custom role a `role` call starts, or the account a `guest` call hands to.
    Name,
}

impl Argument {
    /// Every optional argument, in the order their descriptions are declared.
    pub(crate) const ALL: [Self; 3] = [Self::Place, Self::Branch, Self::Name];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Place => "place",
            Self::Branch => "branch",
            Self::Name => "name",
        }
    }
}

/// `send`'s chain argument — where this step's answer goes next. Only the owner's Assistant's
/// `send`, and its command-line twin, declare it (see [`crate::tools::chain`]).
pub(crate) const FOR_ARG: &str = "for";

/// What one addressee takes besides the shared `message`: which argument, the note describing
/// what it holds for that addressee, and whether the addressee cannot do without it.
///
/// This table is the single owner of the per-addressee requirements: the choice
/// declaration, every argument description and the one-addressee holder's required
/// arguments are all read from it.
struct Requirement {
    arg: Argument,
    note: &'static str,
    required: bool,
}

/// Exactly one addressee of a call. [`Self::OFFERED`] is the dispatch tool's own set,
/// [`Self::COMMAND_LINE`] what the command line may name, [`Self::STRS`] what a custom
/// role's grant may name.
/// [`Self::STRS`] is the narrower set a custom role's grant may name.
///
/// The wire spelling is [`Self::as_str`]/[`Self::parse`] alone: serde goes through them, so a
/// stored chain or launch can never spell an address differently from a call.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Addressee {
    /// The owner's own assistant — the command line's default address, and one
    /// an agent is never offered.
    Assistant,
    /// A single worker run: make one scoped change, report back.
    Worker,
    /// A continuable worker conversation: continue (or create) a named branch.
    WorkerBranch,
    Manager,
    Role,
    /// A guest account's own assistant, named by that account: the message runs as the guest
    /// and nothing comes back.
    Guest,
}

/// What the command line shows for its `--addressee` flag. It lives beside the vocabulary
/// it describes so the two cannot drift: `the_address_help_is_the_vocabulary_it_describes`
/// checks every spelling and REQUIRED argument against the table below, and
/// `the_address_help_says_where_a_chain_step_works` checks the folder rule.
pub(crate) const ADDRESSEE_FLAG_HELP: &str = "Who receives it: 'assistant' (your own \
     assistant — the same as naming none), 'worker' (one scoped worker run in the folder \
     `--place` names), 'worker_branch' (a continuable worker conversation, needs \
     `--branch`), 'manager' (the Manager of the project `--place` names — what a script \
     aiming at a project should use), 'role' (a custom role, needs `--name`), or 'guest' \
     (another account's own assistant, one-way — nothing comes back, needs `--name` with \
     that account's name). A step that names no `--place` works in the folder the nearest \
     earlier step named; with none named anywhere before it, it keeps its default — your \
     own workspace. An address takes only its own flags: a flag the address does not take \
     is refused rather than ignored — `--branch` with 'worker', or `--place` with \
     'assistant', fails the call";

impl Addressee {
    pub const ARG: &'static str = "addressee";

    /// Every addressee an agent's holder can have, narrowest first — the admin Assistant's
    /// set, since `guest` is no other holder's. A GRANT draws from [`Self::STRS`].
    pub const OFFERED: [Self; 5] = [
        Self::Worker,
        Self::WorkerBranch,
        Self::Manager,
        Self::Role,
        Self::Guest,
    ];

    /// Every addressee the COMMAND LINE may name: the tool's own set plus the owner's own
    /// assistant, which leads it.
    pub const COMMAND_LINE: [Self; 6] = [
        Self::Assistant,
        Self::Worker,
        Self::WorkerBranch,
        Self::Manager,
        Self::Role,
        Self::Guest,
    ];

    /// The grantable addressees, in interface order — what
    /// [`crate::tools::grants::kinds_of`] hands a custom role: never the owner's own assistant
    /// nor `guest`.
    pub const STRS: &'static [&'static str] = &[
        Self::Worker.as_str(),
        Self::WorkerBranch.as_str(),
        Self::Manager.as_str(),
        Self::Role.as_str(),
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Assistant => "assistant",
            Self::Worker => "worker",
            Self::WorkerBranch => "worker_branch",
            Self::Manager => "manager",
            Self::Role => "role",
            Self::Guest => "guest",
        }
    }

    /// Parse a wire spelling — the whole vocabulary of a CALL. A stored grant entry never
    /// reaches here unvalidated (see [`crate::tools::grants`]).
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Self::COMMAND_LINE.into_iter().find(|a| a.as_str() == value)
    }

    /// What this addressee does, in the choice declaration's own words.
    const fn does(self) -> &'static str {
        match self {
            Self::Assistant => "hands the message to your own assistant",
            Self::Worker => "one scoped run",
            Self::WorkerBranch => "a continuable branch",
            Self::Manager => "a workspace's Manager",
            Self::Role => "a custom role",
            Self::Guest => "a guest's assistant",
        }
    }

    /// Whether this addressee takes `arg` besides the shared `message`.
    #[must_use]
    pub(crate) fn takes(self, arg: Argument) -> bool {
        self.requirements()
            .iter()
            .any(|requirement| requirement.arg == arg)
    }

    /// The arguments this addressee cannot do without.
    #[must_use]
    pub(crate) fn requires(self) -> Vec<Argument> {
        self.requirements()
            .iter()
            .filter(|requirement| requirement.required)
            .map(|requirement| requirement.arg)
            .collect()
    }

    /// The arguments this addressee takes besides the shared `message`.
    fn requirements(self) -> &'static [Requirement] {
        /// The worker's folder rule, shared by both worker kinds so their `place` text is one
        /// entry.
        const OUTSIDE_MANAGED_WORKSPACES: &str = "a folder outside managed workspaces";
        match self {
            // The message is all it takes.
            Self::Assistant => &[],
            Self::Worker => &[Requirement {
                arg: Argument::Place,
                note: OUTSIDE_MANAGED_WORKSPACES,
                required: false,
            }],
            Self::WorkerBranch => &[
                Requirement {
                    arg: Argument::Place,
                    note: OUTSIDE_MANAGED_WORKSPACES,
                    required: false,
                },
                Requirement {
                    arg: Argument::Branch,
                    note: "the branch to continue",
                    required: true,
                },
            ],
            Self::Manager => &[Requirement {
                arg: Argument::Place,
                note: "the managed workspace",
                required: true,
            }],
            Self::Role => &[
                Requirement {
                    arg: Argument::Place,
                    note: "a workspace, path or `personal:<user>`",
                    required: false,
                },
                Requirement {
                    arg: Argument::Branch,
                    note: "a branch of the role",
                    required: false,
                },
                Requirement {
                    arg: Argument::Name,
                    note: "a custom role's name",
                    required: true,
                },
            ],
            Self::Guest => &[Requirement {
                arg: Argument::Name,
                note: "an account name",
                required: true,
            }],
        }
    }

    /// Whether this address returns an answer of its own, which a schema can shape. `guest` and
    /// `assistant` never do, so a schema combined with either is refused whole.
    fn returns_an_answer(self) -> bool {
        matches!(
            self,
            Self::Worker | Self::WorkerBranch | Self::Manager | Self::Role
        )
    }

    /// The choice declaration's clause: what this addressee does and what it cannot do without.
    fn clause(self) -> String {
        let required: Vec<String> = self
            .requires()
            .into_iter()
            .map(|arg| format!("`{}`", arg.as_str()))
            .collect();
        if required.is_empty() {
            self.does().to_string()
        } else {
            format!("{} ({} required)", self.does(), required.join(", "))
        }
    }
}

impl From<Addressee> for String {
    fn from(addressee: Addressee) -> Self {
        addressee.as_str().to_string()
    }
}

impl TryFrom<String> for Addressee {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        Self::parse(&value).ok_or_else(|| format!("`{value}` is not an address"))
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
    /// The role of the calling agent, used to route async results to the correct agent channel.
    pub caller_role: Role,
    /// The addressees this holder may use, in interface order. A single
    /// addressee is declared and required like any other.
    kinds: Vec<Addressee>,
    /// Whether the optional `place` is accepted. Every addressee takes a folder, so every async
    /// holder declares it; the Engineer's worker-only instance does not.
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
            kinds: Addressee::OFFERED.to_vec(),
            accepts_place: true,
        }
    }

    /// A custom role's variant: exactly the granted addressees, asynchronously, with the
    /// optional `place`. Never empty — a grant entry naming no kind takes the narrowest
    /// addressee (`worker`), so such a holder never has the `guest` handover.
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

    fn allowed(&self) -> Vec<&'static str> {
        self.kinds.iter().map(|kind| kind.as_str()).collect()
    }

    /// Whether this call is the COMMAND LINE's — the owner acting through `mahbot send`, with
    /// no agent session behind it (marked by [`crate::agent::CURRENT_TOOL_COMMAND_LINE`], which
    /// only [`crate::cli::instance`] sets). Nothing a call carries identifies the executor,
    /// because a message the command line hands over runs the addressee's own turn with the
    /// command line's channel in place. This is the one surface where an unnamed address means
    /// the owner's own assistant and where that assistant may be named.
    fn command_line() -> bool {
        crate::agent::tool_command_line()
    }

    /// Whether this holder's `send` is the owner's Assistant's own surface — the only one that
    /// declares `for` and `extract` — or its command-line twin, which is the same toolset. No
    /// other holder has either argument.
    fn owner_surface(&self) -> bool {
        self.caller_role.is_owner_assistant()
    }

    /// An argument that needs an answer of its own must name where that answer goes: the command
    /// line's unnamed address is the owner's own assistant, which `consequence` says cannot take it.
    fn require_named_addressee(
        args: &serde_json::Value,
        arg: &str,
        consequence: &str,
    ) -> Result<()> {
        if Self::command_line()
            && matches!(
                args.get(Addressee::ARG),
                None | Some(serde_json::Value::Null)
            )
        {
            anyhow::bail!(
                "usage: `{arg}` needs an explicit `{}` — the command line's unnamed address is your \
                 own assistant, which {consequence}",
                Addressee::ARG
            );
        }
        Ok(())
    }

    /// The addressees THIS CALL may name: the holder's own inside an agent session, the
    /// command line's own vocabulary outside one — which adds the owner's own assistant, the
    /// one address an agent's holder never has.
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
    /// mandatory address); the command line's own rule is that an unnamed address means the
    /// owner's assistant — naming it is the same as naming none.
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

    /// Refuse an optional argument the chosen address does not take. An agent's call is built
    /// from its holder's interface, so an unused argument is inert by construction; a script
    /// names address and flags apart, and a combination that cannot mean anything is refused
    /// rather than dispatched with the value dropped.
    fn refuse_unused(args: &serde_json::Value, addressee: Addressee) -> Result<()> {
        for arg in Argument::ALL {
            let given = args.get(arg.as_str()).is_some_and(|value| !value.is_null());
            if given && !addressee.takes(arg) {
                return Err(anyhow::anyhow!(
                    "usage: `{}` is not an argument of the `{}` address",
                    arg.as_str(),
                    addressee.as_str()
                ));
            }
        }
        Ok(())
    }

    /// Who answers for the handovers this call makes: the command line's caller is answered and
    /// gone the moment it returns, so its handovers are REFUSED when they could not be made
    /// durable; an agent's turn is durable already, so its handovers are routed best-effort.
    fn answered_for() -> crate::agent::message_router::AnsweredFor {
        if Self::command_line() {
            crate::agent::message_router::AnsweredFor::Caller
        } else {
            crate::agent::message_router::AnsweredFor::Nobody
        }
    }

    /// Whether this holder's interface declares `arg` — the one question the
    /// rendered description, the schema and the reading path all ask.
    fn declares(&self, arg: Argument) -> bool {
        (arg != Argument::Place || self.accepts_place)
            && self.kinds.iter().any(|addressee| addressee.takes(arg))
    }

    /// The optional argument this call names: `None` when it names none **or when the holder
    /// does not declare it** — an argument a holder is never offered is inert, never a refusal.
    fn optional<'a>(&self, args: &'a serde_json::Value, arg: Argument) -> Result<Option<&'a str>> {
        if !self.declares(arg) {
            return Ok(None);
        }
        super::get_opt_str_strict(args, arg.as_str())
    }

    /// The optional `place` this call names: `None` when it names none, and for a holder that
    /// declares no `place` (the Engineer's worker-only instance runs where the caller is).
    async fn optional_place(&self, args: &serde_json::Value) -> Result<Option<Workspace>> {
        if self.declares(Argument::Place) {
            super::delegation::resolve(super::delegation::DelegationKind::Worker, args).await
        } else {
            Ok(None)
        }
    }

    /// The schema this call asks its answer to match, when it asks for one. The argument belongs
    /// to the owner's Assistant and its command-line twin alone; an address that returns no answer
    /// of its own refuses it whole rather than silently ignoring it.
    fn extract(
        &self,
        args: &serde_json::Value,
        addressee: Addressee,
    ) -> Result<Option<crate::extract::Extract>> {
        let Some(raw) = args
            .get(crate::extract::ARG)
            .filter(|value| !value.is_null())
        else {
            return Ok(None);
        };
        if !self.owner_surface() {
            anyhow::bail!(
                "usage: `{}` is not an argument of this tool",
                crate::extract::ARG
            );
        }
        Self::require_named_addressee(args, crate::extract::ARG, "returns no answer of its own")?;
        if !addressee.returns_an_answer() {
            anyhow::bail!(
                "usage: `{}` is not an argument of the `{}` address",
                crate::extract::ARG,
                addressee.as_str()
            );
        }
        Ok(Some(crate::extract::Extract::from_arg(raw)?))
    }

    /// The chain this call starts, when it carries `for`. The whole value is read and every step
    /// checked before anything runs, so a chain that cannot finish is refused whole.
    async fn chain(
        &self,
        args: &serde_json::Value,
        ws: &Workspace,
        addressee: Addressee,
        task: &str,
        extract: Option<&crate::extract::Extract>,
    ) -> Result<Option<crate::launch::Chain>> {
        let Some(raw) = args.get(FOR_ARG).filter(|value| !value.is_null()) else {
            return Ok(None);
        };
        // `for` belongs to the owner's Assistant and its command-line twin alone. The other
        // holders refuse it as an argument they do not have, without saying what it is.
        if !self.owner_surface() {
            anyhow::bail!("usage: `{FOR_ARG}` is not an argument of this tool");
        }
        Self::require_named_addressee(args, FOR_ARG, "can never be a chain step")?;
        let next = super::chain::parse(raw)?;
        let current = crate::launch::Step {
            addressee,
            place: self.optional(args, Argument::Place)?.map(str::to_string),
            branch: self.optional(args, Argument::Branch)?.map(str::to_string),
            name: self.optional(args, Argument::Name)?.map(str::to_string),
            extract: extract.cloned(),
        };
        // Everything knowable now is checked now, in step order: an unknown role, an invalid
        // branch, an unusable place or a folder a step would inherit and cannot work in refuses the
        // whole call, so no step runs.
        super::chain::precheck(&current, &next).await?;
        Ok(Some(crate::launch::Chain {
            id: super::chain::id_of(task, &current, &next),
            task: task.to_string(),
            user: crate::agent::tool_user_name(),
            channel: crate::agent::tool_channel(),
            workspace: ws.name.clone(),
            // Step one has no earlier step to inherit a folder from: its own `place` is its folder,
            // and the hand-off carries it to the steps after it.
            inherited: None,
            current,
            step: 1,
            next,
        }))
    }

    /// The description of one optional argument: what it holds for every addressee that takes
    /// it. Whether it can be left out is the choice declaration's own `(… required)` marker, and
    /// what each addressee means by it is the reference's.
    fn describe(&self, arg: Argument) -> Option<String> {
        if !self.declares(arg) {
            return None;
        }
        let takers: Vec<(Addressee, &'static Requirement)> = Addressee::OFFERED
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
            let same = grouped
                .iter_mut()
                .find(|(_, held)| held.note == requirement.note);
            match same {
                Some((labels, _)) => labels.push(addressee.as_str()),
                None => grouped.push((vec![addressee.as_str()], requirement)),
            }
        }
        let mut parts: Vec<String> = grouped
            .iter()
            .map(|(labels, requirement)| labelled(labels, requirement.note))
            .collect();
        // A folder named on a step is the folder the chain's later steps work in: only the holder of
        // `for` can name one that moves a chain, so only its `place` says so.
        if arg == Argument::Place && self.owner_surface() {
            parts.push(load_prompt("tool/send_place_chain.md").trim().to_string());
        }
        Some(parts.join("; "))
    }
}

#[async_trait]
impl Tool for SendTool {
    fn name(&self) -> &'static str {
        "send"
    }

    /// Compose the description from shared fragments: the base text, the choice note, one
    /// note per held addressee, the async/sync note, and the on-demand pointer last.
    fn description(&self) -> String {
        let mut parts = vec![
            load_prompt("tool/send.md"),
            load_prompt("tool/send_addressee.md"),
        ];
        for addressee in Addressee::OFFERED {
            if !self.kinds.contains(&addressee) {
                continue;
            }
            // The worker note is the one that differs by dispatch: the sync holder returns the
            // report inline, the async holders get it later as a `<worker-report>`.
            let note = match addressee {
                Addressee::Worker if !self.dispatch_mode.is_async() => {
                    load_prompt("tool/send_worker_sync.md")
                }
                other => load_prompt(&format!("tool/send_{}.md", other.as_str())),
            };
            parts.push(labelled(&[addressee.as_str()], &note));
        }
        // Only an async holder returns before the work is done; the sync one's note already says
        // its report returns in the call's result.
        if self.dispatch_mode.is_async() {
            parts.push(load_prompt("tool/send_async.md"));
        }
        // `for` is the owner's Assistant's own argument: one clause here, its shape and its step
        // rule in the schema description and the on-demand reference only a holder of it reaches,
        // and nothing on any other holder's description.
        if self.owner_surface() {
            parts.push(labelled(&[FOR_ARG], load_prompt("tool/send_for.md").trim()));
        }
        // `extract` likewise: one clause here, the subset and the outcome in the on-demand
        // reference only a holder of the argument can reach.
        if self.owner_surface() {
            parts.push(labelled(
                &[crate::extract::ARG],
                load_prompt("tool/send_extract.md").trim(),
            ));
        }
        // One pointer, last, whatever the holder's addressees and mode assembled above.
        crate::tools::tool_reference::standing(self.name(), &parts.join("\n\n"))
    }
    fn parameters_schema(&self) -> serde_json::Value {
        let mut properties = json!({
            "message": {
                "type": "string",
                "description": "The work to hand over — self-contained: the addressee cannot see this conversation."
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
        // Every optional argument the holder's addressees take, described from the one
        // per-addressee table — `place` included only where the holder declares it.
        for arg in Argument::ALL {
            if let Some(description) = self.describe(arg) {
                properties[arg.as_str()] = json!({
                    "type": "string",
                    "description": description
                });
            }
        }
        // A one-addressee holder declares the choice like any other, so the single addressee's
        // required arguments are stated as required: there is no other to choose between.
        if let [only] = self.kinds.as_slice() {
            required.extend(only.requires().into_iter().map(Argument::as_str));
        }
        // The chain argument is an object of the call's own shape; a recursive schema is not
        // reliably produced by every model, so the shape lives in its description.
        if self.owner_surface() {
            // The command line carries the value as JSON text, so both forms are declared.
            properties[FOR_ARG] = json!({
                "type": ["object", "string"],
                "description": load_prompt("tool/send_for_schema.md").trim(),
            });
        }
        // Likewise `extract`: an object or its JSON text.
        if self.owner_surface() {
            properties[crate::extract::ARG] = json!({
                "type": ["object", "string"],
                "description": load_prompt("tool/send_extract_schema.md").trim(),
            });
        }
        super::tool_params_schema(&properties, &required)
    }

    #[expect(clippy::too_many_lines)] // one call's whole dispatch: its address, its chain, its path
    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let message = super::get_dispatch_message(&args)?;
        let addressee = self.addressee(&args)?;
        // The command line's own strictness: it names address and flags apart, so a combination
        // that cannot mean anything is refused here (see [`Self::refuse_unused`]).
        if Self::command_line() {
            Self::refuse_unused(&args, addressee)?;
        }
        // The schema, when the call asks for one: read and validated whole before anything runs.
        let extract = self.extract(&args, addressee)?;
        // A chain replaces the per-addressee acknowledgement: one acknowledgement for the whole
        // chain, and only the final answer comes back.
        if let Some(chain) = self
            .chain(&args, ws, addressee, message, extract.as_ref())
            .await?
        {
            return super::chain::start(chain, ws, Self::answered_for()).await;
        }

        match addressee {
            Addressee::Assistant => {
                // Only the command line reaches this arm: the owner's own assistant is not among
                // the addressees an agent's holder declares, so an agent naming it is refused
                // before dispatch.
                let address =
                    crate::address::AgentAddress::assistant(crate::agent::tool_user_name());
                let id = crate::agent::message_router::deliver_command_line_message(
                    message.to_string(),
                    &address.id(),
                )
                .await?;
                Ok(crate::jobs::acknowledged(
                    "Message handed to your assistant, which answers on the channels it \
                     normally uses."
                        .to_string(),
                    id.as_deref(),
                ))
            }
            Addressee::Worker => {
                if !self.dispatch_mode.is_async() {
                    // The Engineer's sync worker run: the report returns inline.
                    crate::launch::refuse_busy_folder(ws).await?;
                    return super::SyncDurableCore::Worker
                        .run_sync_dispatch(ws, message, self.caller_role)
                        .await;
                }
                // A worker round writes outside the managed workspaces: only a durable background
                // round can honour the optional `place`.
                let place = self.optional_place(&args).await?;
                crate::launch::refuse_busy_folder(place.as_ref().unwrap_or(ws)).await?;
                let job_id = crate::generate_id();
                let id = if Self::command_line() {
                    // Answered and gone the moment this returns: the durable rows are written
                    // BEFORE the acknowledgement, and the round then runs as a resume of its own
                    // rows.
                    crate::tools::implement::hand_over_worker_round(
                        ws,
                        place.as_ref(),
                        message,
                        self.caller_role,
                        &job_id,
                        extract.as_ref(),
                    )
                    .await?
                } else {
                    // An agent's own dispatch: its turn is durable already, and its report comes
                    // back later, so it names no id.
                    super::SyncDurableCore::Worker.spawn_dispatch(
                        ws,
                        place.as_ref(),
                        message,
                        self.caller_role,
                        job_id,
                        extract.clone(),
                    );
                    None
                };
                let folder = place
                    .as_ref()
                    .map_or_else(|| ws.path.clone(), |p| p.path.clone());
                Ok(crate::jobs::acknowledged(
                    format!(
                        "Work handed to a worker in {}; its report arrives as a <worker-report> \
                         message.",
                        folder.trim_end_matches(std::path::MAIN_SEPARATOR)
                    ),
                    id.as_deref(),
                ))
            }
            Addressee::WorkerBranch => {
                let branch = super::get_str(&args, Argument::Branch.as_str())?;
                // A branch is pointed at a folder by the same rule as a single
                // worker run — the branch may not be a way around the worker's refusals.
                let place = self.optional_place(&args).await?;
                // Read the asker before the launch: the report returns to the
                // conversation that asked.
                let asker = crate::agent::tool_delegation_launch();
                crate::launch::worker_branch(
                    ws,
                    branch,
                    message,
                    place.as_ref(),
                    asker,
                    crate::launch::BusyFolder::Refuse,
                    extract.as_ref(),
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
                let asker = crate::agent::tool_delegation_launch();
                // A project's Manager cannot do without the project it messages, so `place` is a
                // required argument of a holder that can reach this arm; `get_str` still refuses a
                // call omitting it.
                let place = super::get_str(&args, Argument::Place.as_str())?;
                let project = super::delegation::resolve_managed_workspace(place).await?;
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
                let address = crate::address::AgentAddress::manager(&project.name);
                let id = crate::agent::message_router::route_agent_message_to_manager(
                    envelope,
                    project.name.clone(),
                    user_name,
                    &address.id(),
                    asker.map(|launch| crate::agent::message_router::Asker {
                        launch,
                        workspace: ws.name.clone(),
                    }),
                    extract.clone(),
                    Self::answered_for(),
                )
                .await?;
                Ok(crate::jobs::acknowledged(
                    format!(
                        "Message delivered to the manager of workspace '{}'.",
                        project.name
                    ),
                    id.as_deref(),
                ))
            }
            Addressee::Role => {
                let name = super::get_str(&args, Argument::Name.as_str())?;
                // Read through the declaring filter like every other optional argument. `place`
                // here is NOT resolved as a worker's is: `launch::launch_workspace` accepts the
                // personal-workspace spelling too.
                let workspace =
                    crate::launch::launch_workspace(self.optional(&args, Argument::Place)?).await?;
                // The answer (and the failure notice) returns to the agent that launched this
                // role, or to the owner's Assistant when nothing did.
                let asker = crate::agent::tool_delegation_launch().map(|launch| {
                    crate::agent::message_router::Asker {
                        launch,
                        workspace: ws.name.clone(),
                    }
                });
                crate::launch::custom_role(
                    name,
                    message,
                    self.optional(&args, Argument::Branch)?,
                    &workspace,
                    asker,
                    crate::launch::BusyFolder::Refuse,
                    extract.as_ref(),
                    Self::answered_for(),
                )
                .await
            }
            Addressee::Guest => {
                let name = super::get_str(&args, Argument::Name.as_str())?;
                // The words of what happened here, not the account-creation rule
                // (`users::refuse_admin_name`).
                anyhow::ensure!(
                    !crate::users::is_admin_name(name),
                    "'{name}' is your own account — a `guest` call would be your assistant \
                     messaging itself; name a guest account instead"
                );
                // Byte-exact matching: no case or composition folding anywhere on
                // this path, and the refusal repeats the name as it was written.
                anyhow::ensure!(
                    crate::users::store().user_exists(name).await?,
                    "no account is named '{name}' — a `guest` call takes an existing account's \
                     name, exactly as it is written"
                );
                let content = crate::prompt::substitute(
                    &crate::prompt::load_prompt("guest_message.md"),
                    &[("{{message}}", message)],
                );
                let address = crate::address::AgentAddress::assistant(name);
                crate::agent::message_router::route_guest_message(
                    content,
                    name,
                    &address.id(),
                    Self::answered_for(),
                )
                .await?;
                Ok(format!(
                    "Message handed to the assistant of '{name}', one-way: it runs as that \
                     guest and nothing comes back."
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Run a dispatch tool the way a turn does: `channel` is the transport the message arrived
    /// over and `agent_id` the session driving the turn — `None` for no session. A turn is an
    /// agent's call whatever carried it, so the command line's marker is NOT set here.
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

    async fn dispatch(
        tool: &dyn Tool,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        dispatch_over(TURN_CHANNEL, Some("agent_harness"), tool, ws, args).await
    }

    /// Run a dispatch tool the way the COMMAND LINE does: the marker only
    /// [`crate::cli::instance`] sets, with the owner's own identity and no agent session.
    async fn dispatch_command_line(
        tool: &dyn Tool,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        dispatch_command_line_as(crate::users::ADMIN_USER_NAME, tool, ws, args).await
    }

    /// The same call under the identity the caller carries — `""` when no owner is named.
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

    /// A registered spy consumer for one conversation. Jobs may arrive in any order, so one
    /// that does not match the search is kept for the next instead of dropped. Dropping it
    /// unregisters the conversation, so a test leaves no consumer behind for the shared router.
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

        /// The next job carrying `needle` — matched by what it carries, never by arrival position,
        /// so a slower round or a repeated delivery cannot cross-wire the assertions.
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

        /// Assert nothing was routed to this conversation — a handover is queued
        /// before the call returns, so anything sent would already be here.
        fn assert_nothing_routed(&mut self) {
            assert!(self.unclaimed.is_empty(), "a job was routed here");
            assert!(self.rx.try_recv().is_err(), "a job was routed here");
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

    /// The ids of the durable rounds of `kind` that exist right now, narrowed to the rounds ONE
    /// PRODUCER's call wrote: the store is shared with parallel tests, so a test that must know
    /// its own rows narrows by provenance.
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

    /// Poll `probe` until it yields a value or a few seconds pass — the durable writes happen
    /// in spawned dispatch tasks.
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

    /// The two dispatch tools really dispatch: every round-creating kind a holder has creates
    /// the round or reaches the addressee it names. (The `guest` handover starts no round —
    /// `the_admin_hands_one_message_to_a_guest_assistant` covers it.)
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
        let assistant = crate::address::AgentAddress::assistant(crate::users::ADMIN_USER_NAME).id();
        let mut assistant_spy = Spy::register(&assistant);
        let manager_agent = crate::address::AgentAddress::manager(&project.name).id();
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

        let ack = dispatch(
            &analyze,
            &caller,
            json!({ "message": "what is here", "depth": "quick" }),
        )
        .await
        .expect("a quick analysis dispatches");
        let quick_job_id = ack_job_id(&ack).expect("an analysis answer names its job");
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
        let deep_job_id = ack_job_id(&ack).expect("a deep analysis answer names its job");
        assert_eq!(round_kind(&deep_job_id).await, "research");
        // The deep investigation is long-running: its report is not awaited here — the round
        // row (a `research` job carrying the job id the deep answer names) is the proof that it
        // dispatched. The round is stopped the way the Running Agents view stops one.
        crate::research_cancel::cancel_research_run(&deep_job_id).await;

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
        // The round is proven by ITS OWN row — the new `implement` job carrying this turn's
        // channel — never by "some row of that kind exists" in the shared store.
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

        crate::custom_role::create(
            "dispatchable",
            "you do the work",
            &[],
            crate::custom_role::CustomRoleBehavior::Continuous,
        )
        .await
        .unwrap();
        let role_branch = crate::address::AgentAddress::custom_role("dispatchable", None).id();
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

        // One rule for `place` across every addressee: a value that is not a string is refused,
        // never silently read as absent — which would run the work in the default folder.
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

        let branch = "harness";
        let branch_agent = crate::address::AgentAddress::worker_branch(branch.to_owned()).id();
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
        let reporter = crate::address::AgentAddress::worker_branch("reporter".to_owned()).id();
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

        // Deleting the branch exactly as the Sessions page does — abandon its unfinished jobs, then
        // drop the session. The store is written by parallel tests too, so the claim is scoped to
        // the agents this harness worked with: only the two worker branches ever ran.
        let owned = [
            assistant.clone(),
            crate::address::AgentAddress::manager(&project.name).id(),
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
        for addressee in ["worker_branch", "manager", "role", "guest"] {
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

        // Leave the shared store as the harness found it: every dispatch above persisted its
        // durable envelope, and no consumer outlives this test to confirm them. Only the rows this
        // harness created are removed — scoped by the snapshot taken before the first dispatch.
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

    /// One folder holds one worker: a `send` that would start a second one there is refused
    /// before anything is dispatched, and goes through again the moment the running one is gone.
    #[tokio::test]
    #[serial_test::serial(provider, channel_registry, drain)]
    async fn a_busy_folder_is_refused_until_the_running_worker_finishes() {
        use std::sync::Arc;

        crate::util::test::init_management_test_stores().await;
        let _policy =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let _provider = crate::util::test::install_fake_provider(Arc::new(
            crate::util::test::FakeProvider::new().ok("the work is done"),
        ));
        crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);

        let home = tempfile::tempdir().unwrap();
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

        let send = SendTool::admin(Role::Assistant);

        // A worker working in that folder, registered the way a running one is.
        let running = format!("busy_folder_{}", crate::generate_suffix());
        let generation = crate::agent::registry::AGENT_REGISTRY.register(
            running.clone(),
            Role::Worker.as_str().to_string(),
            None,
            &crate::Workspace::ephemeral_run(&running, dir.path()),
            std::sync::Arc::new(crate::stop::Stop::default()),
            None,
            None,
        );

        let jobs_before = job_ids_of_kind("implement", TURN_CHANNEL).await;
        // The folder by its RAW temp path (on macOS behind the `/var` symlink): the guard compares
        // what the product resolves a folder to, never the spelling a call uses.
        let err = dispatch(
            &send,
            &caller,
            json!({
                "message": "write it",
                "addressee": "worker",
                "place": dir.path().to_str().unwrap(),
            }),
        )
        .await
        .expect_err("a worker is already working in that folder");
        assert!(
            err.downcast_ref::<crate::launch::FolderHeld>().is_some(),
            "the refusal is typed, so a caller with no one to refuse (an automation's fire) \
             waits instead of failing the work"
        );
        let err = err.to_string();
        for expected in [
            "usage: a worker is already working in",
            folder.as_str(),
            "the message was not sent",
            "parallel agents that can collide must not be started in one folder",
            "split the work across different folders",
            "wait for the running one to finish",
        ] {
            assert!(
                err.contains(expected),
                "the refusal must say {expected:?}: {err}"
            );
        }
        assert!(
            !job_ids_of_kind("implement", TURN_CHANNEL)
                .await
                .iter()
                .any(|id| !jobs_before.contains(id)),
            "a refused call dispatches nothing: {err}"
        );
        // It is the folder that is busy, not the tool: another folder is fine.
        let elsewhere = tempfile::tempdir().unwrap();
        assert!(
            crate::launch::refuse_busy_folder(&crate::Workspace::ephemeral_run(
                "elsewhere",
                elsewhere.path()
            ))
            .await
            .is_ok(),
            "a folder nothing runs in is never refused"
        );

        // The run finishes — no release, no timeout: the entry is simply gone.
        crate::agent::registry::AGENT_REGISTRY.deregister(&running, generation);

        // What is left in the folder is the CALLER's own run, which is never the other agent: a
        // custom role handing a worker its own folder is the normal delegation.
        let own = format!("busy_folder_own_{}", crate::generate_suffix());
        let own_generation = crate::agent::registry::AGENT_REGISTRY.register(
            own.clone(),
            Role::Custom.as_str().to_string(),
            None,
            &crate::Workspace::ephemeral_run(&own, dir.path()),
            std::sync::Arc::new(crate::stop::Stop::default()),
            None,
            None,
        );
        crate::agent::CURRENT_TOOL_AGENT_ID
            .scope(Some(own.clone()), async {
                crate::launch::refuse_busy_folder(&crate::Workspace::ephemeral_run(
                    "own",
                    dir.path(),
                ))
                .await
            })
            .await
            .expect("the caller's own run never blocks its own call");
        crate::agent::registry::AGENT_REGISTRY.deregister(&own, own_generation);

        // The very same call to that folder now goes through.
        let mut spy = Spy::register(
            &crate::address::AgentAddress::assistant(crate::users::ADMIN_USER_NAME).id(),
        );
        dispatch(
            &send,
            &caller,
            json!({ "message": "write it", "addressee": "worker", "place": &folder }),
        )
        .await
        .expect("an idle folder takes a worker");
        let report = spy.report("<worker-report").await;
        assert!(
            report.content.contains(&folder),
            "the worker really ran in that folder: {}",
            report.content
        );
    }

    /// A provider whose every call parks until the test opens the gate, so a round's durable
    /// rows can be observed while the round cannot have finished.
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

    /// A provider that parks every call until the test opens the gate and then panics: a
    /// round that dies in flight, at a moment the test chooses.
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

    /// The ids of the `implement` rounds a COMMAND-LINE call wrote: the rounds whose `channel`
    /// is [`crate::cli::CHANNEL`]. An agent's own turn carries that channel when a command-line
    /// message triggered it, so this narrows to a CALL only while no such turn runs.
    async fn command_line_round_ids() -> Vec<String> {
        job_ids_of_kind("implement", crate::cli::CHANNEL).await
    }

    /// The command line's own worker rounds that are still `launched` — the rows a boot would
    /// resume. A settled round keeps its row as the record its id names, so a terminal row is not
    /// one of these.
    async fn launched_command_line_round_ids() -> Vec<String> {
        crate::session::store()
            .conn
            .query(
                "SELECT id FROM jobs WHERE kind = 'implement' AND channel = ?1 \
                 AND status = 'launched'",
                crate::db::params![crate::cli::CHANNEL],
            )
            .await
            .unwrap()
            .into_iter()
            .map(|row| row.get::<String>(0).unwrap())
            .collect()
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

    /// The ids of the owner's own conversation rows holding any of `contents` — what a
    /// command-line handover to the assistant persists.
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

    /// The id an acknowledgement prints — `(job <id>)` — or `None` when it names none.
    fn ack_job_id(ack: &str) -> Option<String> {
        ack.split("(job ")
            .nth(1)
            .and_then(|rest| rest.split(')').next())
            .map(str::to_string)
    }

    /// The record `ack` names is `launched`, and names the chain's own length when the handover
    /// started one.
    async fn assert_running_record(ack: &str, chain_steps: Option<usize>) {
        let id =
            ack_job_id(ack).unwrap_or_else(|| panic!("the acknowledgement names its id: {ack}"));
        let record = crate::session::store()
            .conn
            .query_optional(
                "SELECT status, chain_steps FROM jobs WHERE id = ?1",
                crate::db::params![id.as_str()],
                |row| Ok::<_, anyhow::Error>((row.get::<String>(0)?, row.get::<Option<i64>>(1)?)),
            )
            .await
            .expect("read the record row")
            .unwrap_or_else(|| panic!("a record stands under {id}"));
        assert_eq!(record.0, crate::outcome::Outcome::Running.as_str());
        assert_eq!(
            record.1.and_then(|steps| usize::try_from(steps).ok()),
            chain_steps
        );
    }

    /// The command line is answered and GONE the moment a call returns, so every handover must
    /// be durable BEFORE the acknowledgement. Each durability claim below is checked IMMEDIATELY
    /// after `execute` returns — no `deadline_poll` for the pin. Every provider call the round
    /// makes parks until this test opens the gate, so "immediately" cannot race its progress.
    /// `handover_records` serialises it against the boot sweep, which settles every `launched`
    /// record.
    #[tokio::test]
    #[serial_test::serial(provider, channel_registry, drain, handover_records)]
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
        let assistant = crate::address::AgentAddress::assistant(crate::users::ADMIN_USER_NAME).id();
        let mut assistant_spy = Spy::register(&assistant);
        let manager_agent = crate::address::AgentAddress::manager(&project.name).id();
        let mut manager_spy = Spy::register(&manager_agent);
        let _branch_spy = Spy::register(
            &crate::address::AgentAddress::worker_branch("cli_branch".to_owned()).id(),
        );
        let role_agent = crate::address::AgentAddress::custom_role("cli_role", None).id();
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

        // Each pin is scoped to the handover it is about — this conversation AND this message's own
        // text — never to a count over the shared store. The count is read before and after: a
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
        let round_id = ack_job_id(&ack).expect("the acknowledgement names the round's id");
        let worker_jobs_after = command_line_round_ids().await;
        assert!(
            worker_jobs_after
                .iter()
                .any(|id| !worker_jobs_before.contains(id)),
            "the round's own row must exist the moment the command line is answered \
             (the durability pin, no poll): before={worker_jobs_before:?} after={worker_jobs_after:?}"
        );
        // The round really runs through the RESUME path: its report arrives in the assistant's
        // conversation and names the folder it worked in. Opening the gate is what lets it run.
        gate_tx
            .send(true)
            .expect("the gated provider holds the gate");
        let report = assistant_spy.report("<worker-report").await;
        assert!(
            report.content.contains(&folder),
            "the worker report names the folder the round ran in: {}",
            report.content
        );
        // The round's row is its work, not a receipt: it is settled when the work ends, so the next
        // boot cannot run the round a second time.
        let (status, _) = crate::util::test::record_row(&round_id)
            .await
            .expect("the finished round's row stays as the record of its handover");
        assert_eq!(
            status,
            crate::outcome::Outcome::Done.as_str(),
            "the finished round leaves no launched row behind"
        );

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

        let branch_agent =
            crate::address::AgentAddress::worker_branch("cli_branch".to_owned()).id();
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

        // Leave the shared store as the harness found it: the deferred envelopes are deleted,
        // scoped by the snapshot taken before the first dispatch. The worker round's own job row
        // needs no cleanup — the resume path terminalizes it as the report is delivered.
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
        // The two handovers to the assistant each left its copy of the owner's own message behind:
        // a message from the owner no turn was produced for. Asserted before it is reclaimed.
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

    /// Every command-line acknowledgement names the record its id answers, and the printed id
    /// resolves to a record still `running` — for a chain, one naming the chain's own length. An
    /// agent's own `send` and the one-way `guest` address open no record, so they print no id.
    #[tokio::test]
    #[serial_test::serial(provider, channel_registry, drain, handover_records)]
    async fn every_command_line_acknowledgement_names_its_record() {
        use std::sync::Arc;

        crate::util::test::init_management_test_stores().await;
        let _policy =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        let _provider = crate::util::test::install_fake_provider(Arc::new(
            crate::util::test::FakeProvider::new().ok("the work is done"),
        ));
        crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);

        let project_dir = tempfile::tempdir().unwrap();
        let project_path =
            crate::util::strip_verbatim_prefix(&std::fs::canonicalize(project_dir.path()).unwrap())
                .to_str()
                .unwrap()
                .to_string();
        let project = crate::util::test::create_test_workspace(&project_path, "ws_cli_ack").await;
        let home = tempfile::tempdir().unwrap();
        // The command line's own workspace: the admin's personal one, where a handover that names
        // no folder runs.
        let caller = crate::workspace::test_ws_named(
            home.path().to_str().unwrap(),
            &crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME),
        );

        // Every conversation a handover reaches is a registered consumer, so no envelope is left to
        // spawn a turn of its own.
        let assistant = crate::address::AgentAddress::assistant(crate::users::ADMIN_USER_NAME).id();
        let mut assistant_spy = Spy::register(&assistant);
        let manager_agent = crate::address::AgentAddress::manager(&project.name).id();
        let mut manager_spy = Spy::register(&manager_agent);
        let branch_agent =
            crate::address::AgentAddress::worker_branch("cli_ack_branch".to_owned()).id();
        let mut branch_spy = Spy::register(&branch_agent);
        let agent_branch_agent =
            crate::address::AgentAddress::worker_branch("cli_ack_agent_branch".to_owned()).id();
        let mut agent_branch_spy = Spy::register(&agent_branch_agent);
        crate::custom_role::create(
            "cli_ack_role",
            "you do the work",
            &[],
            crate::custom_role::CustomRoleBehavior::Continuous,
        )
        .await
        .unwrap();
        let role_agent = crate::address::AgentAddress::custom_role("cli_ack_role", None).id();
        let mut role_spy = Spy::register(&role_agent);
        let mut chain_spy = Spy::register(
            &crate::address::AgentAddress::worker_branch("cli_ack_chain_one".to_owned()).id(),
        );
        crate::users::store()
            .add_user("cli_ack_guest")
            .await
            .unwrap();
        let mut guest_spy =
            Spy::register(&crate::address::AgentAddress::assistant("cli_ack_guest").id());

        // The deferred envelopes that already exist: cleanup removes only the rows this harness
        // creates, because the shared global store is written by parallel tests too.
        let pending_before: std::collections::HashSet<String> =
            crate::jobs::list_pending_jobs(&crate::session::store().conn)
                .await
                .expect("list pending")
                .into_iter()
                .map(|row| row.id)
                .collect();

        let send = SendTool::admin(Role::Assistant);

        let ack = dispatch_command_line(
            &send,
            &caller,
            json!({ "message": "ack from the shell to the assistant" }),
        )
        .await
        .expect("an unnamed command-line address means the owner's assistant");
        assert_running_record(&ack, None).await;
        let routed = assistant_spy.next_job().await;
        assert_eq!(
            routed.pending_job_id.as_deref(),
            ack_job_id(&ack).as_deref(),
            "the envelope carries the very id the acknowledgement printed"
        );

        let ack = dispatch_command_line(
            &send,
            &caller,
            json!({
                "message": "ack from the shell to the manager",
                "addressee": "manager",
                "place": &project.name,
            }),
        )
        .await
        .expect("the command line reaches a project's Manager");
        assert_running_record(&ack, None).await;
        manager_spy.next_job().await;

        let ack = dispatch_command_line(
            &send,
            &caller,
            json!({
                "message": "ack from the shell to the branch",
                "addressee": "worker_branch",
                "branch": "cli_ack_branch",
            }),
        )
        .await
        .expect("the command line reaches a worker branch");
        assert_running_record(&ack, None).await;
        branch_spy.next_job().await;

        let ack = dispatch_command_line(
            &send,
            &caller,
            json!({
                "message": "ack from the shell to the role",
                "addressee": "role",
                "name": "cli_ack_role",
            }),
        )
        .await
        .expect("the command line reaches a custom role");
        assert_running_record(&ack, None).await;
        role_spy.next_job().await;

        // A chain's record is the chain's own id, and its length is the record's own field.
        let ack = dispatch_command_line(
            &send,
            &caller,
            json!({
                "message": "ack from the shell to the chain",
                "addressee": "worker_branch",
                "branch": "cli_ack_chain_one",
                "for": { "addressee": "worker_branch", "branch": "cli_ack_chain_two" },
            }),
        )
        .await
        .expect("the command line starts a chain");
        assert_running_record(&ack, Some(2)).await;
        chain_spy.next_job().await;

        // An agent's own turn opens no record, so its acknowledgement names none — and the id its
        // durable envelope carries names no dispatch row.
        let ack = dispatch(
            &send,
            &caller,
            json!({
                "message": "ack from an agent's own turn",
                "addressee": "worker_branch",
                "branch": "cli_ack_agent_branch",
            }),
        )
        .await
        .expect("an agent's own send still reaches a worker branch");
        assert!(
            ack_job_id(&ack).is_none(),
            "an agent's own send names no id: {ack}"
        );
        let routed = agent_branch_spy.next_job().await;
        let pending = routed.pending_job_id.expect("the handover is durable");
        assert!(
            crate::util::test::record_row(&pending).await.is_none(),
            "an agent's own handover opens no record"
        );

        // The one-way `guest` address answers nothing, so it names no id either.
        let ack = dispatch_command_line(
            &send,
            &caller,
            json!({
                "message": "ack from the shell to the guest",
                "addressee": "guest",
                "name": "cli_ack_guest",
            }),
        )
        .await
        .expect("the owner's assistant reaches a guest's");
        assert!(
            ack_job_id(&ack).is_none(),
            "a one-way handover names no id: {ack}"
        );
        let routed = guest_spy.next_job().await;
        let pending = routed.pending_job_id.expect("the handover is durable");
        assert!(
            crate::util::test::record_row(&pending).await.is_none(),
            "a one-way handover opens no record"
        );

        // Leave the shared store as the harness found it: the deferred envelopes are deleted, scoped
        // by the snapshot taken before the first dispatch, and so is the owner's own message the
        // assistant handover persisted.
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
        crate::channels::chat_history::store()
            .conn
            .execute(
                "DELETE FROM chat_history WHERE content = ?1",
                crate::db::params!["ack from the shell to the assistant"],
            )
            .await
            .expect("reclaim the owner's own message");
    }

    /// A round that dies in flight reports its failure where its report would have gone: the
    /// command line is answered and gone, so an unreported failure would leave work the caller was
    /// told had been handed over with nothing to notice it by until the next start — and a row
    /// left launched would have every boot retry a round that dies the same way.
    #[tokio::test]
    #[serial_test::serial(provider, channel_registry, drain)]
    async fn a_round_that_dies_in_flight_reports_its_failure() {
        use std::sync::Arc;

        crate::util::test::init_management_test_stores().await;
        let _policy =
            crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
        // The round parks at its first call until the gate opens below: the row the
        // ack promises is therefore still there to be read, and is provably THIS round's.
        let (gate_tx, gate_rx) = tokio::sync::watch::channel(false);
        let _provider =
            crate::util::test::install_fake_provider(Arc::new(DyingProvider { open: gate_rx }));
        crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);

        let home = tempfile::tempdir().unwrap();
        let caller = crate::workspace::test_ws_named(
            home.path().to_str().unwrap(),
            &crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME),
        );
        let assistant = crate::address::AgentAddress::assistant(crate::users::ADMIN_USER_NAME).id();
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
        let left = launched_command_line_round_ids().await;
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

    /// The three vocabularies are one list written three ways: [`Addressee::STRS`]
    /// is [`Addressee::OFFERED`] without `guest`, and [`Addressee::COMMAND_LINE`]
    /// is that one led by the owner's own assistant. A drift either mis-scopes a
    /// grant or drops an accepted address from the declarations.
    #[test]
    fn the_address_vocabularies_are_one_list() {
        let offered: Vec<&str> = Addressee::OFFERED.iter().map(|a| a.as_str()).collect();
        let grantable: Vec<&str> = offered
            .iter()
            .copied()
            .filter(|spelling| *spelling != Addressee::Guest.as_str())
            .collect();
        assert_eq!(grantable, Addressee::STRS.to_vec(), "the grant vocabulary");
        assert!(
            !Addressee::STRS.contains(&Addressee::Assistant.as_str()),
            "the owner's own assistant is never grantable"
        );
        let mut command_line: Vec<&str> = vec![Addressee::Assistant.as_str()];
        command_line.extend(offered.iter().copied());
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

    /// The command line's `--addressee` help is the vocabulary's own words: it names every
    /// address the command line takes and every argument each REQUIRES, so a new address (or a new
    /// required argument) cannot ship with the help silent about it.
    #[test]
    fn the_address_help_is_the_vocabulary_it_describes() {
        let help = ADDRESSEE_FLAG_HELP;
        for addressee in Addressee::COMMAND_LINE {
            let spelling = format!("'{}'", addressee.as_str());
            assert!(
                help.contains(&spelling),
                "the help must name the {spelling} address: {help}"
            );
            for arg in addressee.requires() {
                let flag = format!("--{}", arg.as_str().replace('_', "-"));
                assert!(
                    help.contains(&flag),
                    "the help must say the {spelling} address needs {flag}: {help}"
                );
            }
        }
    }

    /// The command line's `--addressee` help says where a chain step that names no folder works:
    /// inside a chain that is the nearest earlier step's folder, never the caller's own.
    #[test]
    fn the_address_help_says_where_a_chain_step_works() {
        let help = ADDRESSEE_FLAG_HELP;
        assert!(
            help.contains("works in the folder the nearest earlier step named"),
            "the help must say where a chain step that names no folder works: {help}"
        );
        assert!(
            !help.contains("your own workspace when it names none"),
            "and must not promise the caller's folder inside a chain: {help}"
        );
    }

    /// How a refused address call is made.
    enum RefusalCall {
        /// An agent session's call: the holder and its arguments.
        Agent(SendTool, serde_json::Value),
        /// A command-line handover that names no owner.
        Unscoped(SendTool, serde_json::Value),
        /// Command-line calls naming an argument the address does not take.
        UnusedFlag(SendTool, &'static [(&'static str, &'static str)]),
    }

    /// One refused call and what its refusal must say.
    struct RefusalCase {
        name: &'static str,
        call: RefusalCall,
        contains: &'static [&'static str],
        /// Substrings the lowercased refusal must carry.
        lowercase_contains: &'static [&'static str],
    }

    #[expect(clippy::too_many_lines)] // one table over every refused address
    #[tokio::test]
    async fn address_refusal_cases() {
        let ws = crate::workspace::test_ws("/tmp/test_ws_send_refuse");
        let mut cases = vec![RefusalCase {
            name: "an argument the address does not take is refused",
            call: RefusalCall::UnusedFlag(
                SendTool::admin(Role::Assistant),
                &[
                    ("assistant", "place"),
                    ("assistant", "branch"),
                    ("worker", "branch"),
                    ("worker", "name"),
                    ("manager", "name"),
                    ("guest", "place"),
                    ("guest", "branch"),
                ],
            ),
            contains: &[],
            lowercase_contains: &[],
        }];
        // A choice value the holder does not offer is refused, naming its own
        // set — never silently narrowed.
        cases.extend([
            RefusalCase {
                name: "the engineer addresses a worker only",
                call: RefusalCall::Agent(
                    SendTool::engineer(),
                    json!({ "message": "do it", "addressee": "manager" }),
                ),
                contains: &["`addressee` must be one of worker"],
                lowercase_contains: &[],
            },
            RefusalCase {
                name: "an unknown addressee names the offered set",
                call: RefusalCall::Agent(
                    SendTool::admin(Role::Assistant),
                    json!({ "message": "do it", "addressee": "bogus" }),
                ),
                contains: &["worker, worker_branch, manager, role, guest"],
                lowercase_contains: &[],
            },
            RefusalCase {
                name: "an agent session never names the owner's own assistant",
                call: RefusalCall::Agent(
                    SendTool::admin(Role::Assistant),
                    json!({ "message": "do it", "addressee": "assistant" }),
                ),
                contains: &["worker, worker_branch, manager, role, guest"],
                lowercase_contains: &[],
            },
            RefusalCase {
                name: "a custom role never acquires the guest handover",
                call: RefusalCall::Agent(
                    SendTool::granted(&[Addressee::Role]),
                    json!({ "message": "do it", "addressee": "guest", "name": "anyone" }),
                ),
                contains: &["`addressee` must be one of role"],
                lowercase_contains: &[],
            },
        ]);
        // The address is never assumed: an agent's call that names none is
        // refused by every holder, even one that can address only one agent.
        for (name, holder) in [
            (
                "the admin assistant states its address",
                SendTool::admin(Role::Assistant),
            ),
            ("the engineer states its address", SendTool::engineer()),
            (
                "a worker-granted role states its address",
                SendTool::granted(&[Addressee::Worker]),
            ),
            (
                "a manager-granted role states its address",
                SendTool::granted(&[Addressee::Manager]),
            ),
        ] {
            cases.push(RefusalCase {
                name,
                call: RefusalCall::Agent(holder, json!({ "message": "do it" })),
                contains: &["`addressee`"],
                lowercase_contains: &["required"],
            });
        }
        // The command line's handover carries the owner it acts for.
        cases.push(RefusalCase {
            name: "a handover that names no owner is refused",
            call: RefusalCall::Unscoped(
                SendTool::admin(Role::Assistant),
                json!({ "message": "no owner" }),
            ),
            contains: &["No user identity"],
            lowercase_contains: &[],
        });
        for case in cases {
            let expected = match case.call {
                RefusalCall::Agent(tool, args) => dispatch(&tool, &ws, args).await,
                RefusalCall::Unscoped(tool, args) => {
                    dispatch_command_line_as("", &tool, &ws, args).await
                }
                RefusalCall::UnusedFlag(tool, combinations) => {
                    for (addressee, flag) in combinations {
                        let mut args = json!({ "message": "go", "addressee": addressee });
                        args[flag] = json!(if *flag == "place" { "/tmp" } else { "x" });
                        let err = dispatch_command_line(&tool, &ws, args)
                            .await
                            .expect_err("an argument the address does not take is refused")
                            .to_string();
                        assert!(
                            err.contains(flag) && err.contains(addressee),
                            "[{}] the refusal names the argument and the address: {err}",
                            case.name
                        );
                    }
                    continue;
                }
            };
            let err = match expected {
                Err(err) => err.to_string(),
                Ok(ok) => panic!("[{}] expected a refusal, got: {ok}", case.name),
            };
            for needle in case.contains {
                assert!(
                    err.contains(needle),
                    "[{}] expected {needle:?}: {err}",
                    case.name
                );
            }
            let lowercase = err.to_lowercase();
            for needle in case.lowercase_contains {
                assert!(
                    lowercase.contains(needle),
                    "[{}] expected {needle:?}: {err}",
                    case.name
                );
            }
        }
    }

    /// Regression: the manager send stays fully internal — the `<assistant-message>` envelope
    /// routes to the workspace Manager (live route + durable pending_jobs copy), while NOTHING is
    /// mirrored into the workspace users' chat_history or channel bindings.
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
        let target = crate::address::AgentAddress::manager(&ws.name).id();
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

    /// The admin's Assistant hands one message to a guest's own Assistant: the durable envelope
    /// runs as that guest, in their personal workspace, wrapped as a one-way
    /// `<owner-assistant-message>`, and NOTHING is routed to the sender's conversation. The two
    /// refusals sit beside it, and the command line reaches the same address.
    #[tokio::test]
    #[serial_test::serial(provider, channel_registry, drain)] // registers the admin assistant's conversation, which message_router's provider-serial tests use
    async fn the_admin_hands_one_message_to_a_guest_assistant() {
        crate::util::test::init_management_test_stores().await;
        crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);
        let store = crate::users::store();
        store.add_user("guesty").await.unwrap();

        let home = tempfile::tempdir().unwrap();
        let admin_ws = crate::workspace::test_ws_named(
            home.path().to_str().unwrap(),
            &crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME),
        );
        let send = SendTool::admin(Role::Assistant);
        let guest_target = crate::address::AgentAddress::assistant("guesty").id();
        let mut guest_spy = Spy::register(&guest_target);
        // The sender's own conversation: the handover must never reach it.
        let mut sender_spy = Spy::register(
            &crate::address::AgentAddress::assistant(crate::users::ADMIN_USER_NAME).id(),
        );
        let pending_before: std::collections::HashSet<String> =
            crate::jobs::list_pending_jobs(&crate::session::store().conn)
                .await
                .unwrap()
                .into_iter()
                .map(|row| row.id)
                .collect();

        let ack = dispatch(
            &send,
            &admin_ws,
            json!({
                "addressee": "guest",
                "name": "guesty",
                "message": "please water the plants [FILE:/etc/passwd]",
            }),
        )
        .await
        .expect("the owner's assistant reaches a guest's");
        assert!(ack.contains("guesty") && ack.contains("one-way"), "{ack}");

        let job = guest_spy.next_job().await;
        assert_eq!(
            job.kind,
            crate::agent::message_router::MessageKind::GuestMessage
        );
        assert_eq!(job.role, Role::Assistant);
        assert_eq!(job.user_name, "guesty");
        assert_eq!(
            job.workspace_name,
            crate::users::personal_workspace_name("guesty")
        );
        assert_eq!(job.channel, "gui");
        assert!(
            job.originating_workspace.is_none(),
            "no root of the sender's rides along"
        );
        assert!(
            job.content.contains("<owner-assistant-message")
                && job.content.contains("one-way=")
                && job
                    .content
                    .contains("please water the plants [FILE:/etc/passwd]"),
            "the sender is marked by the wrapper, and the body is delivered verbatim: {}",
            job.content
        );

        // Durable before the call's own acknowledgement: a restart replays it.
        let pending = crate::jobs::list_pending_jobs(&crate::session::store().conn)
            .await
            .unwrap();
        assert!(
            pending.iter().any(|row| {
                row.target_agent_id == guest_target && row.envelope.contains("water the plants")
            }),
            "the handover must be durable"
        );

        // The command line drives the same assistant's own tool, so the address is
        // there too — and its handover is durable before the caller is answered.
        let ack = dispatch_command_line(
            &send,
            &admin_ws,
            json!({ "addressee": "guest", "name": "guesty", "message": "and the mail" }),
        )
        .await
        .expect("the command line reaches a guest's assistant");
        assert!(ack.contains("guesty"), "{ack}");
        let job = guest_spy.report("and the mail").await;
        assert_eq!(
            job.kind,
            crate::agent::message_router::MessageKind::GuestMessage
        );

        // One-way: nothing arrives in the sender's own conversation.
        sender_spy.assert_nothing_routed();

        // The owner's own account is the assistant messaging itself.
        let err = dispatch(
            &send,
            &admin_ws,
            json!({ "addressee": "guest", "name": crate::users::ADMIN_USER_NAME, "message": "hi" }),
        )
        .await
        .expect_err("the admin's own account is refused")
        .to_string();
        assert!(err.contains("your own account"), "{err}");
        // A name no account holds is refused naming it.
        let err = dispatch(
            &send,
            &admin_ws,
            json!({ "addressee": "guest", "name": "nobody_here", "message": "hi" }),
        )
        .await
        .expect_err("an unknown account is refused")
        .to_string();
        assert!(err.contains("nobody_here"), "{err}");
        // Leave the shared store as the test found it: only the rows this test
        // created (a parallel test may be holding an envelope for this session).
        let after = crate::jobs::list_pending_jobs(&crate::session::store().conn)
            .await
            .unwrap();
        for row in after {
            if !pending_before.contains(&row.id) {
                crate::jobs::delete_pending_job(&crate::session::store().conn, &row.id)
                    .await
                    .expect("delete this test's own envelope");
            }
        }
    }

    /// `for` and `extract` are the owner's Assistant's own arguments: its schema and description
    /// carry both, no other holder's does.
    #[test]
    fn only_the_owner_assistants_send_declares_for_and_extract() {
        let admin = SendTool::admin(Role::Assistant);
        for arg in [FOR_ARG, crate::extract::ARG] {
            assert!(
                admin.parameters_schema()["properties"].get(arg).is_some(),
                "the admin's schema offers `{arg}`"
            );
            assert!(
                admin.description().contains(&format!("`{arg}`")),
                "the admin's description mentions `{arg}`: {}",
                admin.description()
            );
        }
        for other in [
            SendTool::engineer(),
            SendTool::granted(&[Addressee::Worker]),
        ] {
            for arg in [FOR_ARG, crate::extract::ARG] {
                assert!(
                    other.parameters_schema()["properties"].get(arg).is_none(),
                    "no other holder's schema offers `{arg}`"
                );
                assert!(
                    !other.description().contains(&format!("`{arg}`")),
                    "no other holder's description mentions `{arg}`: {}",
                    other.description()
                );
            }
        }
        // Where a chain step that names no folder works is the same holder's rule: only a `send`
        // that takes `for` says it.
        let place = |tool: &SendTool| {
            tool.parameters_schema()["properties"]["place"]["description"]
                .as_str()
                .unwrap_or_default()
                .to_string()
        };
        assert!(
            place(&admin).contains("the chain's later steps work in"),
            "the admin's `place` says a folder named there moves the chain: {}",
            place(&admin)
        );
        for other in [
            SendTool::engineer(),
            SendTool::granted(&[Addressee::Worker]),
        ] {
            assert!(
                !place(&other).contains("chain"),
                "a holder without `for` is told nothing of a chain: {}",
                place(&other)
            );
        }
    }

    /// An address spells the same in a stored chain or launch as in a call: one wire spelling,
    /// [`Addressee::as_str`], because a stored step's answer depends on it.
    #[test]
    fn an_address_spells_the_same_in_a_store_as_in_a_call() {
        for addressee in Addressee::COMMAND_LINE {
            let stored = serde_json::to_value(addressee).expect("an address serializes");
            assert_eq!(
                stored,
                serde_json::json!(addressee.as_str()),
                "the stored spelling is the call's"
            );
            assert_eq!(
                serde_json::from_value::<Addressee>(stored).expect("and reads back"),
                addressee
            );
        }
        assert!(
            serde_json::from_value::<Addressee>(serde_json::json!("nobody")).is_err(),
            "an address outside the vocabulary is refused"
        );
    }

    /// A chain belongs to the owner's Assistant and its command-line twin: the other holders
    /// refuse `for` as an argument they do not have (without saying what it is), and no chain can
    /// start with an address that is not a step.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn a_chain_belongs_to_the_owner_assistant_alone() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("send_chain_gate")
                .to_string_lossy(),
            "send_chain_gate_ws",
        )
        .await;
        let next = json!({"addressee": "worker"});

        // The Engineer's worker-only instance has no such argument.
        let err = dispatch(
            &SendTool::engineer(),
            &ws,
            json!({"message": "x", "addressee": "worker", "for": next}),
        )
        .await
        .expect_err("the Engineer refuses `for`")
        .to_string();
        assert!(err.contains("is not an argument"), "{err}");
        assert!(
            !err.contains("chain") && !err.contains("step"),
            "the refusal does not describe the argument: {err}"
        );
        // A custom role holding a grant likewise.
        let err = dispatch(
            &SendTool::granted(&[Addressee::Worker]),
            &ws,
            json!({"message": "x", "addressee": "worker", "for": next}),
        )
        .await
        .expect_err("a granted role refuses `for`")
        .to_string();
        assert!(err.contains("is not an argument"), "{err}");

        // No chain starts with an address that cannot be a step.
        for (addressee, extra) in [
            ("guest", json!({"name": "someone"})),
            ("manager", json!({"place": &ws.name})),
        ] {
            let mut args = json!({"message": "x", "addressee": addressee, "for": next});
            for (key, value) in extra.as_object().expect("object") {
                args[key] = value.clone();
            }
            let err = dispatch(&SendTool::admin(Role::Assistant), &ws, args)
                .await
                .expect_err("a chain cannot start there")
                .to_string();
            assert!(err.contains("cannot"), "{addressee}: {err}");
        }

        // The command line may name the owner's own assistant — but never as a chain's step.
        let err = dispatch_command_line(
            &SendTool::admin(Role::Assistant),
            &ws,
            json!({"message": "x", "addressee": "assistant", "for": next}),
        )
        .await
        .expect_err("the assistant cannot start a chain")
        .to_string();
        assert!(err.contains("cannot"), "{err}");

        // The command line's unnamed address is the owner's own assistant, which can never be a
        // step, so a chain there must say where step one goes.
        let err = dispatch_command_line(
            &SendTool::admin(Role::Assistant),
            &ws,
            json!({"message": "x", "for": next}),
        )
        .await
        .expect_err("an unnamed address cannot start a chain")
        .to_string();
        assert!(err.contains("explicit"), "{err}");

        // A step that names no such role is refused whole, before anything runs.
        let err = dispatch(
            &SendTool::admin(Role::Assistant),
            &ws,
            json!({
                "message": "x",
                "addressee": "worker",
                "for": {"addressee": "role", "name": "no_such_role_anywhere"},
            }),
        )
        .await
        .expect_err("an unknown role is refused whole")
        .to_string();
        assert!(err.contains("no_such_role_anywhere"), "{err}");

        // `extract` belongs to the same holders, and its refusal says no more than that: it never
        // describes what the argument does.
        let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        for tool in [
            SendTool::engineer(),
            SendTool::granted(&[Addressee::Worker]),
        ] {
            let err = dispatch(
                &tool,
                &ws,
                json!({"message": "x", "addressee": "worker", "extract": schema}),
            )
            .await
            .expect_err("only the owner's Assistant takes a schema")
            .to_string();
            assert!(err.contains("is not an argument"), "{err}");
            for undescribed in ["schema", "JSON", "answer", "chain"] {
                assert!(
                    !err.contains(undescribed),
                    "the refusal does not describe the argument: {err}"
                );
            }
        }
        // The addresses that return no answer of their own refuse a schema whole, whatever the
        // holder — on the agent path and on the command line's.
        let guest =
            json!({"message": "x", "addressee": "guest", "name": "someone", "extract": schema});
        let err = dispatch(&SendTool::admin(Role::Assistant), &ws, guest.clone())
            .await
            .expect_err("a guest returns no answer to shape")
            .to_string();
        assert!(
            err.contains("`extract`") && err.contains("`guest`"),
            "{err}"
        );
        let err = dispatch_command_line(&SendTool::admin(Role::Assistant), &ws, guest)
            .await
            .expect_err("a guest returns no answer to shape")
            .to_string();
        assert!(
            err.contains("`extract`") && err.contains("`guest`"),
            "{err}"
        );
        let err = dispatch_command_line(
            &SendTool::admin(Role::Assistant),
            &ws,
            json!({"message": "x", "addressee": "assistant", "extract": schema}),
        )
        .await
        .expect_err("the owner's own assistant returns no answer to shape")
        .to_string();
        assert!(
            err.contains("`extract`") && err.contains("`assistant`"),
            "{err}"
        );
        // The command line's unnamed address is that same assistant, so a schema there must say
        // where the answer goes.
        let err = dispatch_command_line(
            &SendTool::admin(Role::Assistant),
            &ws,
            json!({"message": "x", "extract": schema}),
        )
        .await
        .expect_err("an unnamed address returns no answer to shape")
        .to_string();
        assert!(
            err.contains("explicit") && err.contains("`extract`"),
            "{err}"
        );
    }

    /// A schema is an argument of every address that returns an answer of its own — a worker run, a
    /// branch, a manager and a custom role — and of no other.
    #[test]
    fn a_schema_is_read_for_every_address_that_returns_an_answer() {
        let tool = SendTool::admin(Role::Assistant);
        let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let expected = crate::extract::Extract::from_arg(&schema).expect("the schema is accepted");
        for addressee in [
            Addressee::Worker,
            Addressee::WorkerBranch,
            Addressee::Manager,
            Addressee::Role,
        ] {
            assert!(
                addressee.returns_an_answer(),
                "{} takes a schema",
                addressee.as_str()
            );
            let args = json!({"message": "x", "extract": schema});
            assert_eq!(
                tool.extract(&args, addressee)
                    .expect("this address takes a schema"),
                Some(expected.clone()),
                "the schema is read for {}",
                addressee.as_str()
            );
        }
        for addressee in [Addressee::Guest, Addressee::Assistant] {
            let args = json!({"message": "x", "extract": schema});
            let err = tool
                .extract(&args, addressee)
                .expect_err("this address returns no answer to shape")
                .to_string();
            assert!(
                err.contains("`extract`") && err.contains(&format!("`{}`", addressee.as_str())),
                "{err}"
            );
        }
        // No schema, no transformation: an ordinary call reads `None`.
        assert_eq!(
            tool.extract(&json!({"message": "x"}), Addressee::Worker)
                .expect("an absent schema is not a refusal"),
            None
        );
    }

    /// The schema arrives as the object an agent passes or as the JSON text the command line
    /// passes, and both are the same schema.
    #[test]
    fn a_schema_is_the_same_object_or_json_text() {
        let tool = SendTool::admin(Role::Assistant);
        let object = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let from_object = tool
            .extract(&json!({"extract": object}), Addressee::Worker)
            .expect("an object is accepted");
        let from_text = tool
            .extract(&json!({"extract": object.to_string()}), Addressee::Worker)
            .expect("JSON text is accepted");
        assert_eq!(from_object, from_text);
        assert!(from_object.is_some(), "and both name a schema");
    }

    /// A schema is refused by an address that returns no answer of its own, even for the holder that
    /// has the argument — and before the address does any work of its own (the account is never
    /// looked up).
    #[tokio::test]
    async fn a_guest_address_refuses_a_schema_even_for_the_holder_that_has_it() {
        let ws = crate::workspace::test_ws("send_schema_guest");
        let err = dispatch(
            &SendTool::admin(Role::Assistant),
            &ws,
            json!({
                "message": "x",
                "addressee": "guest",
                "name": "no_such_account_anywhere",
                "extract": {"type": "object", "properties": {"a": {"type": "string"}}},
            }),
        )
        .await
        .expect_err("a guest returns no answer to shape")
        .to_string();
        assert!(
            err.contains("`extract`") && err.contains("`guest`"),
            "{err}"
        );
        assert!(
            !err.contains("no_such_account_anywhere"),
            "the schema is refused before the account is even looked up: {err}"
        );
    }

    /// A schema that cannot be enforced is refused whole, before anything runs: no round is spawned
    /// and no envelope is queued.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn a_malformed_schema_is_refused_before_anything_runs() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("send_schema_refused")
                .to_string_lossy(),
            "send_schema_refused_ws",
        )
        .await;
        let conn = &crate::session::store().conn;
        let pending = || async {
            let mut ids: Vec<String> = crate::jobs::list_pending_jobs(conn)
                .await
                .expect("list pending")
                .into_iter()
                .map(|row| row.id)
                .collect();
            ids.sort();
            ids
        };
        let before = pending().await;
        let oversized = format!(
            "{{\"type\":\"string\",\"description\":\"{}\"}}",
            "x".repeat(crate::extract::MAX_TEXT_BYTES)
        );
        for bad in [
            // A keyword this product cannot enforce is refused rather than silently ignored.
            json!({"type": "object", "allOf": []}),
            json!(""),
            json!("not json at all"),
            json!(oversized),
        ] {
            let err = dispatch(
                &SendTool::admin(Role::Assistant),
                &ws,
                json!({"message": "x", "addressee": "worker", "extract": bad}),
            )
            .await
            .expect_err("a schema that cannot be enforced is refused whole")
            .to_string();
            assert!(err.contains("extract"), "{bad}: {err}");
        }
        assert_eq!(before, pending().await, "nothing was dispatched or queued");
    }

    /// A fresh chain records no inherited folder, whatever its first step names: the call is step
    /// one, its own `place` is its folder, and the hand-off carries it to the steps after it.
    #[tokio::test]
    async fn a_fresh_chain_records_no_inherited_folder() {
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("send_chain_folder")
                .to_string_lossy(),
            "send_chain_folder_ws",
        )
        .await;
        let scratch = tempfile::tempdir().unwrap();
        let folder = crate::stored_workspace_path(scratch.path())
            .to_string_lossy()
            .into_owned();
        let send = SendTool::admin(Role::Assistant);

        // The call that starts a chain is its first step, and it names a folder like any other.
        let chain = send
            .chain(
                &json!({"place": &folder, "for": {"addressee": "worker"}}),
                &ws,
                Addressee::Worker,
                "draft it",
                None,
            )
            .await
            .expect("the chain is usable")
            .expect("a chain");
        assert_eq!(
            chain.current.place.as_deref(),
            Some(folder.as_str()),
            "step one's own `place` is its folder"
        );
        assert_eq!(
            chain.inherited, None,
            "nothing earlier has named a folder for step one to inherit"
        );
        assert_eq!(
            chain.workspace, ws.name,
            "the chain's own workspace is the caller's and never moves"
        );

        // With no step naming a folder, nothing changes.
        let bare = send
            .chain(
                &json!({"for": {"addressee": "worker"}}),
                &ws,
                Addressee::Worker,
                "draft it",
                None,
            )
            .await
            .expect("the chain is usable")
            .expect("a chain");
        assert_eq!(bare.inherited, None);
    }

    /// A chain whose step would inherit a folder it cannot work in at all is refused whole, naming
    /// the step and the folder, before anything is dispatched or queued.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn a_step_that_would_inherit_a_folder_it_cannot_work_in_refuses_the_call() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("send_chain_refused")
                .to_string_lossy(),
            "send_chain_refused_ws",
        )
        .await;
        let managed_dir = crate::util::test::test_root().join("send_chain_refused_project");
        std::fs::create_dir_all(&managed_dir).unwrap();
        let managed = crate::util::test::create_test_workspace(
            &crate::stored_workspace_path(&managed_dir).to_string_lossy(),
            "send_chain_refused_project",
        )
        .await;
        crate::custom_role::create(
            "folder_role",
            "you write",
            &[],
            crate::custom_role::CustomRoleBehavior::Continuous,
        )
        .await
        .unwrap();
        let conn = &crate::session::store().conn;
        let pending = || async {
            let mut ids: Vec<String> = crate::jobs::list_pending_jobs(conn)
                .await
                .expect("list pending")
                .into_iter()
                .map(|row| row.id)
                .collect();
            ids.sort();
            ids
        };
        let before = pending().await;

        // A role may be pointed at a managed workspace; the worker step after it cannot work there.
        let err = dispatch(
            &SendTool::admin(Role::Assistant),
            &ws,
            json!({
                "message": "write it, then polish it",
                "addressee": "role",
                "name": "folder_role",
                "place": &managed.name,
                "for": {"addressee": "worker"},
            }),
        )
        .await
        .expect_err("a worker cannot inherit a managed workspace")
        .to_string();
        assert!(err.starts_with("usage:"), "{err}");
        assert!(err.contains("step 2 of 2"), "the step is named: {err}");
        assert!(
            err.contains(&managed.name),
            "the folder it would inherit is named: {err}"
        );
        assert!(err.contains("hint:"), "{err}");
        assert_eq!(before, pending().await, "nothing was dispatched or queued");
    }

    /// A chain starts with ONE acknowledgement for the whole chain, and step one's durable
    /// envelope already carries the chain, the original task and where in the chain it is.
    #[tokio::test]
    #[serial_test::serial(drain)]
    async fn a_chain_starts_with_one_acknowledgement() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("send_chain_start")
                .to_string_lossy(),
            "send_chain_start_ws",
        )
        .await;
        let conn = &crate::session::store().conn;
        let before: Vec<String> = crate::jobs::list_pending_jobs(conn)
            .await
            .expect("list pending")
            .into_iter()
            .map(|row| row.id)
            .collect();

        let ack = dispatch(
            &SendTool::admin(Role::Assistant),
            &ws,
            json!({
                "message": "write it, then polish it",
                "addressee": "worker_branch",
                "branch": "drafting",
                "for": {"addressee": "worker_branch", "branch": "polishing"},
            }),
        )
        .await
        .expect("a chain starts");
        assert!(ack.contains("Chain of 2 steps"), "{ack}");
        assert!(ack.contains("only the final answer"), "{ack}");
        assert!(
            !ack.contains("its report arrives"),
            "the chain's acknowledgement replaces the per-addressee one: {ack}"
        );

        let branch = crate::address::AgentAddress::worker_branch("drafting".to_owned()).id();
        let row = crate::jobs::list_pending_jobs(conn)
            .await
            .expect("list pending")
            .into_iter()
            .find(|row| !before.contains(&row.id) && row.target_agent_id == branch)
            .expect("step one's durable envelope");
        let job: crate::agent::message_router::AgentJob =
            serde_json::from_str(&row.envelope).expect("a decodable envelope");
        assert!(
            job.content.contains("write it, then polish it"),
            "{}",
            job.content
        );
        assert!(job.content.contains("chain-step"), "{}", job.content);
        assert!(job.content.contains("step=\"1\""), "{}", job.content);
        assert!(job.content.contains("of=\"2\""), "{}", job.content);
        assert!(
            job.content.contains("continues to"),
            "step one is told its answer continues: {}",
            job.content
        );
        let chain = job
            .custom_role
            .as_ref()
            .and_then(|launch| launch.chain.as_deref())
            .expect("step one carries the chain");
        assert_eq!(chain.total(), 2);
        assert_eq!(chain.next.len(), 1);
        assert_eq!(chain.task, "write it, then polish it");
        assert_eq!(
            row.id,
            crate::launch::record_id(&chain.id, 1),
            "step one's record id is the chain's own, so a replay finds it"
        );

        crate::jobs::delete_chain_rows(&chain.id)
            .await
            .expect("clean up this test's chain");
    }
}
