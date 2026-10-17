//! Command-line frontends onto the RUNNING product.
//!
//! A command line here is not a second implementation of anything: `mahbot
//! config …`, `mahbot workspace …` and `mahbot send …` carry the operation into
//! the instance that is already up, where it is executed by the very tool an
//! agent's assistant calls — the same validation, and for the setting families
//! the same immediate effect on the live product (a `send` hands its message to
//! an agent, one-way, and the answer arrives in the owner's own assistant
//! conversation). `mahbot document …`, `mahbot image …`, `mahbot video …` and
//! `mahbot vector …` do the same for the product's producing tools — the
//! document tool, the picture tool, the two video tools and the vector
//! rasterizer — so a script reaches a document or a PDF, a picture, a clip and a
//! rendered vector with the very operations the assistant has, the same values,
//! limits and mutual exclusions, and the account's own model choice. The
//! frontend never opens or writes a store itself, so there is no second writer
//! and no second copy of any rule.
//!
//! What the producing families add is a destination: where the produced files
//! go is the COMMAND LINE's own operand, carried beside the operation's own
//! values and never an argument of the tool (see [`FlagKind::Destination`]) —
//! the one flag outside the rule that every flag is an argument the tool reads.
//! The agents keep writing where they always wrote and are told nothing new; the
//! command line acts as the machine's owner, so the files land where that owner
//! asks, and it refuses a destination the product's own cleanup would later
//! empty ([`resolve_output_dir`]).
//!
//! The paths a producing family takes are the CALLER's, and they are resolved
//! where the caller is rather than where the product runs: a relative one in the
//! directory this command was run in, `~` against the home directory, and a
//! reference file that is not there or cannot be read refused as the caller's own
//! misuse, before anything is sent ([`caller_path`] — a picture a structured
//! value carries included, see [`resolve_embedded_paths`]).
//!
//! This file holds the instance's own readiness signal — whether operations may
//! be served at all, and the waiter the instance side blocks on — plus the
//! execution entry point in [`instance`], and the frontend's own command line:
//! the family registry that ties every sub-command to the tool it runs (an
//! operation per action, the tool's own name where it takes no action, or the
//! family word itself where the family IS the operation), the argv parsing under
//! it, the help, and the [`Exit`] classes a script reads. The
//! flag grammar itself is [`grammar`], which `mahbot audio` reads too — the
//! browser-automation command and the OpenRouter benchmark read their own flags
//! and word their own refusals.

pub(crate) mod grammar;
pub(crate) mod instance;

use std::fmt::Write as _;
use std::sync::OnceLock;

use crate::db::ipc::{CallAttempt, CallOutcome, CallRequest};
use crate::util::lock::InstanceLockState;

/// The product's top-level `--help` text: every public entry point, the
/// command-line families among them.
///
/// It lives beside the family registry rather than in the binary so that the
/// check that every entry point is listed runs in the library's own test lane
/// (`every_entry_point_is_listed_in_the_top_level_usage`) — one reachable only
/// by someone who already knew its word would not be an entry point.
pub const TOP_LEVEL_USAGE: &str = "\
mahbot — autonomous agentic engineering system with a GUI dashboard daemon

Usage:
  mahbot                 Launch the GUI dashboard daemon
  mahbot paused          Launch the daemon with every registered workspace
                         paused (no ticket is claimed or advanced)
  mahbot audio <operation> [flags]
                         Speech on the command line (macOS): a recording into
                         text, text into a spoken WAV file
  mahbot chrome <args>   Browser automation CLI over the shared chrome core
  mahbot debug           Read-only SQL query tool against the live stores
  mahbot bench-openrouter <args>
                         Standalone OpenRouter provider benchmark
  mahbot config <operation> [flags]
                         Set the RUNNING product up (Telegram bot, users, tool
                         grants, web search, custom roles)
  mahbot workspace <operation> [flags]
                         Work with the RUNNING product's workspaces and tickets
  mahbot send [flags]    Hand one message to an agent and finish: one-way — the
                         answer arrives in your assistant's conversation
  mahbot document <operation> [flags]
                         Produce a document or a PDF with the RUNNING product's
                         document tool
  mahbot image gen [flags]
                         Produce a picture with the RUNNING product's picture tool
  mahbot video <gen|edit> [flags]
                         Produce or edit a clip with the RUNNING product's video
                         tools
  mahbot vector rasterize [flags]
                         Render a vector file into an ordinary picture with the
                         RUNNING product's rasterizer

Options:
  -h, --help             Print this help and exit
  -V, --version          Print the version and exit";

/// The transport tag the command line's own operations are attributed to. The
/// one-way `send` presents a message to the owner's assistant over it — an
/// incoming message from the owner like any other, mirroring to his Telegram as
/// one of his own and reaching the history the way a message he typed into the
/// dashboard (`gui`) or spoke (`voice`) does. The tag is the product's own
/// record: it tells a shell-sent message apart from those, and never reaches the
/// message.
pub(crate) const CHANNEL: &str = "cli";

/// Whether this instance can serve a command-line operation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ServiceState {
    /// The instance is still bringing itself up.
    Starting,
    /// The instance finished starting: it is the whole product, and an
    /// operation may be served.
    Ready,
    /// The instance did not finish starting up: it can never serve one.
    Unavailable,
}

/// The instance's own service state, as a `watch` so a waiter is woken the
/// moment it changes. A `OnceLock` because there is exactly one instance per
/// process and this sender never drops, so a subscriber always has a live
/// counterpart.
static SERVICE: OnceLock<tokio::sync::watch::Sender<ServiceState>> = OnceLock::new();

fn service_sender() -> &'static tokio::sync::watch::Sender<ServiceState> {
    SERVICE.get_or_init(|| tokio::sync::watch::channel(ServiceState::Starting).0)
}

/// The instance has finished bringing itself up: operations can be served.
pub fn mark_ready() {
    service_sender().send_replace(ServiceState::Ready);
}

/// The instance did not finish starting up, so it can never serve operations.
pub fn mark_unavailable() {
    service_sender().send_replace(ServiceState::Unavailable);
}

/// How an operation's wait for the product ended.
pub(crate) enum ServiceGate {
    /// The whole product is up — the operation may run.
    Ready,
    /// The boot failed: operations can never be served here.
    Unavailable,
    /// The instance began shutting down — before it finished starting, or after:
    /// nothing ran, so the caller may retry.
    NotAccepted,
    /// The instance neither finished starting nor reported a failed boot within
    /// [`SERVICE_READY_TIMEOUT`] — a wedged boot. Nothing ran, so the caller is
    /// told so instead of being held forever.
    StillStarting,
}

/// How long an operation waits for the instance to finish starting before it is
/// refused. The wait for the product to come up has to be bounded — a wedged
/// boot would otherwise hold the caller with no answer and no way to tell it is
/// waiting — while the OPERATION itself, once the instance has taken it on, is
/// never given up on.
const SERVICE_READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Wait until this instance can serve a command-line operation.
///
/// An operation must not be served before [`mark_ready`]: until that point the
/// product is not the WHOLE product — a `mahbot_config` action or a workspace
/// registration is a side effect that routes into the message pipeline, and
/// landing it in a half-built instance would act on something that is not yet
/// there. Waiting is the honest answer.
///
/// The caller is never simply dropped while it waits: the wait resolves on
/// `Ready`, on `Unavailable` (the boot failed, so a frontend must not hang
/// forever on an instance that can never serve it), on
/// [`StillStarting`](ServiceGate::StillStarting) after [`SERVICE_READY_TIMEOUT`],
/// or when the shutdown token fires — before `Ready` as well as while starting,
/// since an exit already under way must not begin new work — where
/// [`NotAccepted`] carries nothing run, so the caller may retry safely.
///
/// [`NotAccepted`]: ServiceGate::NotAccepted
pub(crate) async fn wait_for_service() -> ServiceGate {
    let mut rx = service_sender().subscribe();
    let shutdown = crate::shutdown::shutdown_token();
    // One bound for the whole wait, not one per wake-up.
    let bound = tokio::time::sleep(SERVICE_READY_TIMEOUT);
    tokio::pin!(bound);
    loop {
        // The product has begun its exit: nothing new is taken on, whatever the
        // state says — an operation must not start under an exit already under
        // way (nothing ran, so the caller may retry).
        if shutdown.is_cancelled() {
            return ServiceGate::NotAccepted;
        }
        match *rx.borrow_and_update() {
            ServiceState::Ready => return ServiceGate::Ready,
            ServiceState::Unavailable => return ServiceGate::Unavailable,
            ServiceState::Starting => {}
        }
        tokio::select! {
            changed = rx.changed() => {
                if changed.is_err() {
                    // The sender is a process-lifetime static; a dropped one
                    // means the process is tearing down, where nothing can be
                    // served.
                    return ServiceGate::NotAccepted;
                }
            }
            () = shutdown.cancelled() => {
                return ServiceGate::NotAccepted;
            }
            () = &mut bound => {
                return ServiceGate::StillStarting;
            }
        }
    }
}

// ── The exit classes ──────────────────────────────────────────────

/// The exit-code classes a command line here separates — the same numbering the
/// browser-automation command and `mahbot audio` use: success, a failed
/// operation, the environment, and the command's own misuse.
///
/// What separates [`Usage`](Exit::Usage) from [`Failed`](Exit::Failed) is who
/// judges. Misuse covers what a frontend can settle without the target at all: a
/// word no family or operation has, a flag the operation does not take, a flag
/// with no value after it, a token that is not a flag at all, an argument that is
/// not valid text — and, for a frontend that opens the files itself, what the
/// filesystem says about the paths it was handed: an input that is missing, is not
/// a regular file, or cannot be read; a destination directory that is not there; a
/// destination that is already taken. Everything the TARGET of the command judges
/// is a failed operation instead: a value the tool does not accept, an argument
/// the tool finds missing, a refusal the tool returns, an operation it carried out
/// only in part.
///
/// A frontend that only CARRIES the operation to a running product — this one,
/// over IPC — holds no copy of the target's own rules (see the module doc), so
/// it must never guess at one and call it misuse: a wrong class would make a
/// script skip a real refusal or keep retrying a malformed invocation. A frontend
/// that RUNS the target in its own process — `mahbot audio` — may judge the
/// closed vocabulary that target publishes — voice and language names, the audio
/// formats it takes, the one container it writes, text that says nothing — as the
/// caller's own misuse, because there the rule is data the frontend reads rather
/// than a guess. What the classes mean for a given family is said by that
/// family's own help.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Exit {
    /// The operation was carried out.
    Ok = 0,
    /// The operation was not carried out, was carried out only in part, or its
    /// outcome could not be confirmed after the call went out (it may have been
    /// carried out, which is never the environment's class).
    Failed = 1,
    /// Nothing ran: the environment — the instance, the engine, its models —
    /// could not take the operation on. The environment, not the operation.
    Environment = 2,
    /// The caller's own mistake: an invocation that cannot mean what it asks, and
    /// — for a frontend that opens the files itself — the paths it was handed, as
    /// the filesystem itself judges them (see the class's own doc for the line).
    Usage = 3,
}

impl Exit {
    /// The process exit code for this class.
    pub(crate) const fn code(self) -> i32 {
        self as i32
    }
}

// ── The family / operation / flag registry ────────────────────────

/// One command-line family: the frontend onto a capability of the product — one
/// of its tools, or the pair of tools that share a capability.
pub(crate) struct Family {
    /// The command-line word (`mahbot <word> …`).
    pub(crate) word: &'static str,
    /// The product tools this family fronts, in the order its help lists them:
    /// one for most families, and both video tools where a capability is split
    /// across two.
    pub(crate) tools: &'static [&'static str],
    /// The protocol value this family's operation frames carry: the generation of
    /// the command line's surface it needs. A family the channel has already
    /// served carries the value
    /// [`CALL_STORE_BASE`](crate::db::ipc::CALL_STORE_BASE) and keeps working
    /// against a product one build behind; a family this build adds carries the
    /// current one, so a product that does not know it refuses the call as the
    /// environment's class instead of answering it as a failed operation (see
    /// [`crate::db::ipc`]).
    pub(crate) protocol: &'static str,
    /// One line for `mahbot -h`, and the family's own help opening on it.
    pub(crate) purpose: &'static str,
    /// What follows the family's word: an operation to run, or nothing at all.
    pub(crate) surface: Surface,
}

/// What follows a family's word on the command line.
pub(crate) enum Surface {
    /// `mahbot <word> <operation> [flags]`: one operation per command-line
    /// sub-command, each running one of the family's tools — under the tool's own
    /// `action` where it has one, and as the tool itself where it has none.
    Operations(&'static [Operation]),
    /// The family IS its operation: `mahbot <word> [flags]`, no operation word
    /// at all. The single tool the family fronts takes no `action`, and the args
    /// it is handed carry no `action` key.
    Standalone(&'static [Flag]),
}

/// One command-line operation: the tool it runs and, where the tool has one, the
/// action it runs it under.
pub(crate) struct Operation {
    /// The tool this operation runs — one of the family's own.
    pub(crate) tool: &'static str,
    /// The tool's own `action` value, or `None` where the tool takes no action
    /// (the picture tool and both video tools): the operation then IS that tool,
    /// and the args it is handed carry no `action` key.
    pub(crate) action: Option<&'static str>,
    /// One line for the family's help and the operation's own help.
    pub(crate) purpose: &'static str,
    /// The flags this operation takes, in the order its help lists them.
    pub(crate) flags: &'static [Flag],
}

/// The sub-command word of one operation, derived rather than written down so a
/// sub-command can never be spelled a way its tool or action is not: the tool's
/// own action with `_` written `-` (`fill_template` → `fill-template`), or — for
/// a tool that takes no action — the tool's own name with the family word it
/// carries dropped (`image_gen` in `image` → `gen`).
fn operation_word(family: &Family, operation: &Operation) -> String {
    match operation.action {
        Some(action) => action.replace('_', "-"),
        None => operation
            .tool
            .strip_prefix(family.word)
            .unwrap_or(operation.tool)
            .trim_start_matches('_')
            .replace('_', "-"),
    }
}

/// Whether this family's operations produce FILES — derived from the flags the
/// registry already carries rather than stated a second time: a producing
/// family's every operation carries the destination flag
/// ([`FlagKind::Destination`]) and no other family carries one anywhere, which
/// is what the registry test holds.
#[must_use]
fn produces_files(family: &Family) -> bool {
    match family.surface {
        Surface::Operations(operations) => operations
            .iter()
            .flat_map(|operation| operation.flags)
            .any(|flag| matches!(flag.kind, FlagKind::Destination)),
        Surface::Standalone(flags) => flags
            .iter()
            .any(|flag| matches!(flag.kind, FlagKind::Destination)),
    }
}

/// The words a family answers to, as a usage line states them: the operations'
/// own sub-command words where the family's tools take no `action` at all (the
/// picture tool's one, the two video tools'), so the line names the words it
/// really answers to; `<operation>` where an `action` picks one of many; and
/// `[flags]` alone where the family IS its operation. Derived from the registry,
/// so no help can advertise a word the command line no longer answers to
/// (`every_entry_point_is_listed_in_the_top_level_usage`).
#[must_use]
fn family_syntax(family: &Family) -> String {
    match family.surface {
        Surface::Standalone(_) => format!("mahbot {} [flags]", family.word),
        Surface::Operations(operations) => {
            let words: Vec<String> = operations
                .iter()
                .map(|operation| operation_word(family, operation))
                .collect();
            match (
                operations
                    .iter()
                    .all(|operation| operation.action.is_none()),
                words.as_slice(),
            ) {
                (true, [word]) => format!("mahbot {} {word} [flags]", family.word),
                (true, words) => format!("mahbot {} <{}> [flags]", family.word, words.join("|")),
                (false, _) => format!("mahbot {} <operation> [flags]", family.word),
            }
        }
    }
}

/// One flag of one operation: the tool argument it fills.
pub(crate) struct Flag {
    /// The tool's own argument name; the command-line spelling is derived from
    /// it (`ticket_id` → `--ticket-id`), so a flag can never be spelled a way
    /// the tool does not read. The one exception is the destination flag
    /// ([`FlagKind::Destination`]), whose name is the command line's own.
    pub(crate) arg: &'static str,
    /// How a value reaches the argument.
    pub(crate) kind: FlagKind,
    /// A value the caller hands over as a secret: never echoed, recorded or
    /// answered back (the instance removes it from everything it reports).
    pub(crate) secret: bool,
    /// One line for the operation's help.
    pub(crate) purpose: &'static str,
}

/// How a flag carries its value — the shape of the tool argument it fills.
///
/// Two spellings of a value are this command line's own, read the same way by
/// every kind that carries one: a value written `./@…` is the literal text that
/// begins with `@` (the `./` is dropped and nothing is read), and — for the kinds
/// that say so ([`Self::has_file_form`]) — a value written `@<file>` IS what that
/// file holds. The escape exists because a text value may legitimately begin with
/// `@` (a prompt naming someone), which is also the only spelling a file name
/// cannot be confused with.
///
/// A kind whose value names a path also RESOLVES it (see [`caller_path`]): a
/// `Path` its one value, a `PathList` every entry, a `Structured` every string the
/// fields it carries name.
pub(crate) enum FlagKind {
    /// One value per occurrence: `--name bob`. The value is read as text, or —
    /// like the list and structure kinds below — taken from a FILE by writing
    /// `@<file>`, so a text too long for one command line travels whole.
    Value,
    /// One value per occurrence that NAMES A FILE on this machine
    /// (`--path report.pdf`): a string argument like [`Value`](Self::Value),
    /// whose value the command line resolves before it is sent — a relative one
    /// in the directory this command was run in, and `~` against the home
    /// directory — so the operation opens the file the caller meant. The product
    /// has its own working directory, and a relative path sent on as written
    /// would name a different file there. Reference material must also be there
    /// and readable: a path that is not is the caller's own misuse, while a
    /// public URL — which only the product can fetch — is left to the product.
    /// No `@<file>` form: the value already names the file.
    Path,
    /// One NUMBER per occurrence, for a tool argument that is an integer or a
    /// real (`--duration 8`). A value that is no JSON number is sent on verbatim
    /// as the text it is, so the tool's own refusal — not this frontend's — is
    /// what the caller gets: whether a value is one the tool accepts is the
    /// tool's to judge.
    Number,
    /// Repeatable, one array entry per occurrence, for an array-of-texts
    /// argument (`--tools analyze:quick,deep --tools send`) — an entry is taken
    /// VERBATIM, so an entry that itself contains a comma is unambiguous. An
    /// occurrence with an EMPTY value is the EMPTY list — `--tools=` — which
    /// tools read differently from an absent key (`update_custom_role` clears the
    /// role's tools with it while the absent key leaves them as they are). An
    /// occurrence whose value starts with `@` names a FILE holding the entries as
    /// a JSON array of strings, so a list too long for one command line travels
    /// whole; the `./@…` escape gives an entry that begins with `@` (see the
    /// enum's own doc).
    List,
    /// [`List`](Self::List)'s file-naming form: the same array of texts, with
    /// every entry (and the file named after `@`) resolved the way
    /// [`Path`](Self::Path) resolves its one value.
    PathList,
    /// One STRUCTURED value per occurrence, for an argument that is an object or
    /// an array of objects: the value is the JSON it is written as
    /// (`--values '{"name": "Ada"}'`), or `@<file>` naming a file that holds it.
    /// A value that is no JSON at all is sent on verbatim as the text it is, so
    /// the tool judges it; a FILE that cannot be read, or that does not hold JSON,
    /// is the caller's own misuse — the file form is this command line's own.
    ///
    /// The field names it carries are the ones a REFERENCE FILE stands under
    /// inside the structure — the `image` of a block, of an edit that places one
    /// — and every string there is resolved like a [`Path`](Self::Path) value:
    /// the product resolves them against its own directories, so a relative path
    /// sent on as written would name a file the caller never meant.
    Structured(&'static [&'static str]),
    /// A switch, for a tool argument that is a boolean: the bare `--full` is
    /// `true`, `--full=true` is the same, and `--full=false` is the `false` a bare
    /// switch cannot carry. Both values must be spellable because a tool reads
    /// them differently — an argument left out is the tool's own default, which is
    /// not the same as an explicit `false` where the default is the provider's (a
    /// video's `--generate-audio`).
    Present,
    /// The destination of the files the operation produces — the command line's
    /// OWN operand, and the single flag no tool reads. It names the directory
    /// produced files go into, and it is the exception the whole registry turns
    /// on: the tools' arguments and the agents' instructions know nothing of it
    /// (see the module docs). No `@<file>` form: it names a directory itself.
    Destination,
}

impl FlagKind {
    /// Whether this kind reads its value from the file its `@<file>` form names.
    ///
    /// The kinds whose value can be long or structured — a text, a list, a
    /// structure — have it, so a caller is never capped by the command line's own
    /// length. [`Path`](Self::Path) and [`Destination`](Self::Destination) do
    /// not: their value names the file (or directory) itself, so a file holding
    /// it adds nothing. [`Number`](Self::Number) and [`Present`](Self::Present)
    /// do not either: neither carries text.
    #[must_use]
    const fn has_file_form(&self) -> bool {
        matches!(
            self,
            Self::Value | Self::List | Self::PathList | Self::Structured(_)
        )
    }
}

/// The command lines this file fronts: the admin Assistant's own toolset — the
/// tools each family's `tools` names. Nothing else the owner holds is reachable
/// through the channel (the instance serves a tool by name only when a family
/// below fronts it). Where the words are named elsewhere they are derived from
/// here — [`family_words`] for the whole set, [`producing_families`] for the
/// families that make files — so no surface keeps a list of its own to go stale.
pub(crate) const FAMILIES: &[Family] = &[
    Family {
        word: "config",
        tools: &["mahbot_config"],
        protocol: crate::db::ipc::CALL_STORE_BASE,
        purpose: "Set the running product up — its Telegram bot, its users and \
                  their tool grants, web search, custom roles",
        surface: Surface::Operations(&[
            Operation {
                tool: "mahbot_config",
                action: Some("setup_telegram_bot"),
                purpose: "Save the Telegram bot token from BotFather",
                flags: &[Flag {
                    arg: "token",
                    kind: FlagKind::Value,
                    secret: true,
                    purpose: "The bot token BotFather gave you",
                }],
            },
            Operation {
                tool: "mahbot_config",
                action: Some("bind_telegram"),
                purpose: "Bind your own Telegram account, so the bot's messages reach you",
                flags: &[Flag {
                    arg: "handle",
                    kind: FlagKind::Value,
                    secret: false,
                    purpose: "Your Telegram username or your numeric id — write './@name' for the \
                              '@' spelling, which otherwise names a file",
                }],
            },
            Operation {
                tool: "mahbot_config",
                action: Some("add_user"),
                purpose: "Create a guest account and bind its Telegram account",
                flags: &[
                    Flag {
                        arg: "name",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "The new user's display name, which is also their personal \
                                  workspace folder",
                    },
                    Flag {
                        arg: "telegram",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "Their Telegram username or their numeric id — write './@name' \
                                  for the '@' spelling, which otherwise names a file",
                    },
                ],
            },
            Operation {
                tool: "mahbot_config",
                action: Some("setup_web_search"),
                purpose: "Choose the web-search provider and store its API key",
                flags: &[
                    Flag {
                        arg: "provider",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "'firecrawl' or 'exa'",
                    },
                    Flag {
                        arg: "key",
                        kind: FlagKind::Value,
                        secret: true,
                        purpose: "The API key for the chosen provider",
                    },
                ],
            },
            Operation {
                tool: "mahbot_config",
                action: Some("grant_tool"),
                purpose: "Give a user one of your custom tools",
                flags: &[USER_FLAG, CUSTOM_TOOL_FLAG],
            },
            Operation {
                tool: "mahbot_config",
                action: Some("revoke_tool"),
                purpose: "Take one of your custom tools back from a user",
                flags: &[USER_FLAG, CUSTOM_TOOL_FLAG],
            },
            Operation {
                tool: "mahbot_config",
                action: Some("list_grants"),
                purpose: "List the custom tools granted to a user, or to every user",
                flags: &[Flag {
                    arg: "user",
                    kind: FlagKind::Value,
                    secret: false,
                    purpose: "The user's name; omit it to list every user's grants",
                }],
            },
            Operation {
                tool: "mahbot_config",
                action: Some("create_custom_role"),
                purpose: "Create a custom role from a prompt",
                flags: &[
                    Flag {
                        arg: "name",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "The role's name — ASCII letters, digits, '_' and '-' only, and \
                                  not a built-in agent namespace",
                    },
                    Flag {
                        arg: "prompt",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "The whole definition of the role: everything it is told besides \
                                  the folder it works in",
                    },
                    Flag {
                        arg: "tools",
                        kind: FlagKind::List,
                        secret: false,
                        purpose: "A tool name, optionally `name:kind1,kind2` for one with kinds; \
                                  repeat once per tool (omit for none)",
                    },
                    Flag {
                        arg: "behavior",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "'continuous' (one shared branch of the role) or 'one_off' (runs \
                                  once and reports back)",
                    },
                ],
            },
            Operation {
                tool: "mahbot_config",
                action: Some("update_custom_role"),
                purpose: "Change a custom role's prompt, tools or behaviour",
                flags: &[
                    Flag {
                        arg: "name",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "The custom role to change",
                    },
                    Flag {
                        arg: "prompt",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "The role's new definition (omit to leave it as it is)",
                    },
                    Flag {
                        arg: "tools",
                        kind: FlagKind::List,
                        secret: false,
                        purpose: "The role's new tool subset; repeat once per tool (omit to leave \
                                  it as it is)",
                    },
                    Flag {
                        arg: "behavior",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "The role's new behaviour: 'continuous' or 'one_off' (omit to \
                                  leave it as it is)",
                    },
                ],
            },
            Operation {
                tool: "mahbot_config",
                action: Some("list_custom_roles"),
                purpose: "List the custom roles",
                flags: &[],
            },
            Operation {
                tool: "mahbot_config",
                action: Some("delete_custom_role"),
                purpose: "Delete a custom role",
                flags: &[Flag {
                    arg: "name",
                    kind: FlagKind::Value,
                    secret: false,
                    purpose: "The custom role to delete",
                }],
            },
        ]),
    },
    Family {
        word: "workspace",
        tools: &["workspace"],
        protocol: crate::db::ipc::CALL_STORE_BASE,
        purpose: "Work with the running product's workspaces, their tickets and their board",
        surface: Surface::Operations(&[
            Operation {
                tool: "workspace",
                action: Some("list"),
                purpose: "List the registered workspaces",
                flags: &[Flag {
                    arg: "workspace",
                    kind: FlagKind::Value,
                    secret: false,
                    purpose: "One registered workspace; omit it to list every one",
                }],
            },
            Operation {
                tool: "workspace",
                action: Some("add"),
                purpose: "Register a software project directory as a workspace",
                flags: &[
                    Flag {
                        arg: "name",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "The new workspace's short unique name, used in ticket ids and \
                                  the GUI",
                    },
                    Flag {
                        arg: "path",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "Absolute path to the project directory to register",
                    },
                ],
            },
            // `pause` (and `resume`, which says the same) can act on ONE
            // workspace or on the whole fleet, and the two forms are told apart
            // by ABSENCE: with no `--workspace` the args object the tool is
            // handed carries no `workspace` key at all, and that absence IS the
            // fleet-wide form. Naming a workspace is then the only thing the
            // call may carry — the tool refuses a stray field precisely so a
            // fleet-wide write can never be widened by accident, so the
            // frontend must never invent one and must never send a `null`
            // (which the tool reads as the unnamed form).
            Operation {
                tool: "workspace",
                action: Some("pause"),
                purpose: "Freeze one registered workspace, or every one when none is named",
                flags: &[Flag {
                    arg: "workspace",
                    kind: FlagKind::Value,
                    secret: false,
                    purpose: "The registered workspace to freeze; omit it to freeze EVERY \
                              registered workspace (the fleet-wide pause — a workspace that is \
                              not ready is skipped and left untouched, which makes that result \
                              a partial one)",
                }],
            },
            // See `pause`: the same fleet-wide rule, the same absent key.
            Operation {
                tool: "workspace",
                action: Some("resume"),
                purpose: "Resume one registered workspace, or every one when none is named",
                flags: &[Flag {
                    arg: "workspace",
                    kind: FlagKind::Value,
                    secret: false,
                    purpose: "The registered workspace to resume; omit it to resume EVERY \
                              registered workspace (the fleet-wide resume — a workspace that is \
                              not ready is skipped and left untouched, which makes that result \
                              a partial one)",
                }],
            },
            Operation {
                tool: "workspace",
                action: Some("board"),
                purpose: "List the tickets of one workspace, or of every one, by phase",
                flags: &[
                    Flag {
                        arg: "workspace",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "One registered workspace; omit it to cover every one",
                    },
                    Flag {
                        arg: "phase",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "Narrow the listing to one ticket phase (e.g. 'queued', \
                                  'in_development', 'done'); omit it for every non-archived ticket",
                    },
                ],
            },
            Operation {
                tool: "workspace",
                action: Some("ticket"),
                purpose: "Show one ticket",
                flags: &[
                    Flag {
                        arg: "ticket_id",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "The ticket's full id — '<workspace>-<number>', which already \
                                  carries its workspace (a bare number is refused)",
                    },
                    Flag {
                        arg: "full",
                        kind: FlagKind::Present,
                        secret: false,
                        purpose: "The ticket whole and unclipped — omitted, the tool's own \
                                  default is the clipped form",
                    },
                ],
            },
            Operation {
                tool: "workspace",
                action: Some("comment"),
                purpose: "Leave a clarification note on a ticket",
                flags: &[
                    Flag {
                        arg: "ticket_id",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "The ticket's full id — '<workspace>-<number>'",
                    },
                    Flag {
                        arg: "content",
                        kind: FlagKind::Value,
                        secret: false,
                        purpose: "The clarification note to leave on the ticket",
                    },
                ],
            },
        ]),
    },
    Family {
        word: "send",
        tools: &["send"],
        // The family this build adds: a product built before it answers the call
        // as a failed operation, so it carries the current value and is refused
        // as the environment's class instead.
        protocol: crate::db::ipc::CALL_STORE,
        purpose: "Hand one message to an agent and finish: one-way — no reply is waited for or \
                  printed, and the answer arrives in your assistant's conversation",
        surface: Surface::Standalone(&[
            Flag {
                arg: "message",
                kind: FlagKind::Value,
                secret: false,
                purpose: "The work to hand over; make it self-contained — the addressee cannot \
                          see your conversation (required)",
            },
            // The address vocabulary's own words: it lives beside the vocabulary
            // (`tools::send`), which checks it names every address and every
            // argument an address requires.
            Flag {
                arg: "addressee",
                kind: FlagKind::Value,
                secret: false,
                purpose: crate::tools::send::ADDRESSEE_FLAG_HELP,
            },
            Flag {
                arg: "place",
                kind: FlagKind::Value,
                secret: false,
                purpose: "The folder the addressee works in: for 'manager', the project to \
                          message, by its registered name or its own folder; for 'role', a \
                          registered project, a folder path, or 'personal:<user>'; a coder or \
                          a coder branch works in a directory outside the registered projects \
                          (one inside is refused), and in your own workspace when it names none",
            },
            Flag {
                arg: "branch",
                kind: FlagKind::Value,
                secret: false,
                purpose: "The branch to continue or create — a coder branch's, or a custom \
                          role's own named branch (ASCII letters, digits, '_' and '-', at most \
                          32 chars)",
            },
            Flag {
                arg: "name",
                kind: FlagKind::Value,
                secret: false,
                purpose: "The custom role to start (it must exist)",
            },
        ]),
    },
    // ── The producing families: documents, pictures, video and vectors ─────
    //
    // These families front the tools every Assistant holds regardless of its
    // account (the media models they use resolve per-user), so the command line
    // reaches exactly what the owner's own assistant can already produce — the
    // same operations, the same values, the same limits and the same mutual
    // exclusions, all judged by the tool itself. What the command line adds is
    // the DESTINATION, its own operand ([`FlagKind::Destination`]).
    Family {
        word: "document",
        tools: &["document"],
        // The families this build adds: a product built before them serves no
        // such tool, so the call comes back as the environment's class — nothing
        // ran — rather than as a failed operation.
        protocol: crate::db::ipc::CALL_STORE,
        purpose: "Produce a document or a PDF with the running product's own document tool — \
                  write, fill a template, edit an Office file, and every PDF operation",
        surface: Surface::Operations(&[
            Operation {
                tool: "document",
                action: Some("create"),
                purpose: "Write a new file: docx/pptx/pdf blocks, or xlsx sheets",
                flags: &[
                    DOCUMENT_FORMAT_FLAG,
                    DOCUMENT_CONTENT_FLAG,
                    DOCUMENT_SHEETS_FLAG,
                    DOCUMENT_METADATA_FLAG,
                    DOCUMENT_FILE_NAME_FLAG,
                    OUTPUT_DIR_FLAG,
                ],
            },
            Operation {
                tool: "document",
                action: Some("fill_template"),
                purpose: "Fill a copy of your own .docx/.pptx/.xlsx sample's placeholders",
                flags: &[
                    DOCUMENT_PATH_FLAG,
                    DOCUMENT_VALUES_FLAG,
                    DOCUMENT_FILE_NAME_FLAG,
                    OUTPUT_DIR_FLAG,
                ],
            },
            Operation {
                tool: "document",
                action: Some("docx_edit"),
                purpose: "Edit a .docx/.docm",
                flags: &[
                    DOCUMENT_PATH_FLAG,
                    IMAGE_EDITS_FLAG,
                    DOCUMENT_FILE_NAME_FLAG,
                    OUTPUT_DIR_FLAG,
                ],
            },
            Operation {
                tool: "document",
                action: Some("xlsx_edit"),
                purpose: "Edit an .xlsx/.xlsm",
                flags: &[
                    DOCUMENT_PATH_FLAG,
                    XLSX_EDITS_FLAG,
                    DOCUMENT_FILE_NAME_FLAG,
                    OUTPUT_DIR_FLAG,
                ],
            },
            Operation {
                tool: "document",
                action: Some("pptx_edit"),
                purpose: "Edit a .pptx/.pptm",
                flags: &[
                    DOCUMENT_PATH_FLAG,
                    IMAGE_EDITS_FLAG,
                    DOCUMENT_FILE_NAME_FLAG,
                    OUTPUT_DIR_FLAG,
                ],
            },
            Operation {
                tool: "document",
                action: Some("pdf_merge"),
                purpose: "Merge PDFs into one new file",
                flags: &[
                    DOCUMENT_PATHS_FLAG,
                    DOCUMENT_METADATA_FLAG,
                    DOCUMENT_FILE_NAME_FLAG,
                    OUTPUT_DIR_FLAG,
                ],
            },
            Operation {
                tool: "document",
                action: Some("pdf_split"),
                purpose: "Split a PDF, one produced file per page entry",
                flags: &[
                    DOCUMENT_PATH_FLAG,
                    DOCUMENT_PAGES_FLAG,
                    DOCUMENT_METADATA_FLAG,
                    DOCUMENT_FILE_NAME_FLAG,
                    OUTPUT_DIR_FLAG,
                ],
            },
            Operation {
                tool: "document",
                action: Some("pdf_rotate"),
                purpose: "Add a rotation to a PDF's pages",
                flags: &[
                    DOCUMENT_PATH_FLAG,
                    DOCUMENT_DEGREES_FLAG,
                    DOCUMENT_PAGES_FLAG,
                    DOCUMENT_METADATA_FLAG,
                    DOCUMENT_FILE_NAME_FLAG,
                    OUTPUT_DIR_FLAG,
                ],
            },
            Operation {
                tool: "document",
                action: Some("pdf_text"),
                purpose: "Draw text on a PDF's pages, as page content rather than an annotation",
                flags: &[
                    DOCUMENT_PATH_FLAG,
                    DOCUMENT_TEXT_FLAG,
                    DOCUMENT_PAGES_FLAG,
                    DOCUMENT_X_FLAG,
                    DOCUMENT_Y_FLAG,
                    DOCUMENT_SIZE_FLAG,
                    DOCUMENT_COLOR_FLAG,
                    DOCUMENT_STAMP_FLAG,
                    DOCUMENT_ROTATE_FLAG,
                    DOCUMENT_METADATA_FLAG,
                    DOCUMENT_FILE_NAME_FLAG,
                    OUTPUT_DIR_FLAG,
                ],
            },
            Operation {
                tool: "document",
                action: Some("pdf_image"),
                purpose: "Place a PNG or JPEG on a PDF's pages, as page content rather than an \
                          annotation",
                flags: &[
                    DOCUMENT_PATH_FLAG,
                    DOCUMENT_IMAGE_FLAG,
                    DOCUMENT_PAGES_FLAG,
                    DOCUMENT_X_FLAG,
                    DOCUMENT_Y_FLAG,
                    DOCUMENT_WIDTH_FLAG,
                    DOCUMENT_HEIGHT_FLAG,
                    DOCUMENT_METADATA_FLAG,
                    DOCUMENT_FILE_NAME_FLAG,
                    OUTPUT_DIR_FLAG,
                ],
            },
            Operation {
                tool: "document",
                action: Some("pdf_form_fill"),
                purpose: "Fill a PDF's form fields into a copy",
                flags: &[
                    DOCUMENT_PATH_FLAG,
                    DOCUMENT_VALUES_FLAG,
                    DOCUMENT_FLATTEN_FLAG,
                    DOCUMENT_METADATA_FLAG,
                    DOCUMENT_FILE_NAME_FLAG,
                    OUTPUT_DIR_FLAG,
                ],
            },
        ]),
    },
    Family {
        word: "image",
        tools: &["image_gen"],
        protocol: crate::db::ipc::CALL_STORE,
        purpose: "Produce a picture with the running product's own picture tool",
        surface: Surface::Operations(&[Operation {
            tool: "image_gen",
            action: None,
            purpose: "Produce a picture from a description, or from reference pictures",
            flags: &[
                IMAGE_PROMPT_FLAG,
                IMAGE_IMAGES_FLAG,
                IMAGE_ASPECT_RATIO_FLAG,
                IMAGE_SIZE_FLAG,
                OUTPUT_DIR_FLAG,
            ],
        }]),
    },
    Family {
        word: "video",
        tools: &["video_gen", "video_edit"],
        protocol: crate::db::ipc::CALL_STORE,
        purpose: "Produce or edit a clip with the running product's own video tools",
        surface: Surface::Operations(&[
            Operation {
                tool: "video_gen",
                action: None,
                purpose: "Produce a clip from a description, or from a reference picture",
                flags: &[
                    VIDEO_PROMPT_FLAG,
                    VIDEO_IMAGES_FLAG,
                    VIDEO_DURATION_FLAG,
                    VIDEO_RESOLUTION_FLAG,
                    VIDEO_ASPECT_RATIO_FLAG,
                    VIDEO_SIZE_FLAG,
                    VIDEO_GENERATE_AUDIO_FLAG,
                    VIDEO_SEED_FLAG,
                    OUTPUT_DIR_FLAG,
                ],
            },
            Operation {
                tool: "video_edit",
                action: None,
                purpose: "Edit a clip: restyle it from a description, or give its first and \
                          last frames",
                flags: &[
                    EDIT_VIDEO_URL_FLAG,
                    EDIT_INSTRUCTION_FLAG,
                    EDIT_DURATION_FLAG,
                    EDIT_IMAGES_FLAG,
                    EDIT_FIRST_FRAME_FLAG,
                    EDIT_LAST_FRAME_FLAG,
                    OUTPUT_DIR_FLAG,
                ],
            },
        ]),
    },
    Family {
        word: "vector",
        tools: &["vector_rasterize"],
        protocol: crate::db::ipc::CALL_STORE,
        purpose: "Turn a vector file into an ordinary picture with the running product's own \
                  rasterizer",
        surface: Surface::Operations(&[Operation {
            tool: "vector_rasterize",
            action: None,
            purpose: "Render an SVG file into a PNG picture",
            flags: &[VECTOR_PATH_FLAG, OUTPUT_DIR_FLAG],
        }]),
    },
];

/// The rasterizer's `path` argument: the vector file to render.
const VECTOR_PATH_FLAG: Flag = Flag {
    arg: "path",
    kind: FlagKind::Path,
    secret: false,
    purpose: "The vector file to render (an SVG today) — resolved where this command runs, and \
              read with the owner's own authority",
};

/// The destination flag every producing family's operation carries: the command
/// line's OWN operand, the one flag no tool reads (see
/// [`FlagKind::Destination`]).
const OUTPUT_DIR_FLAG: Flag = Flag {
    arg: "output_dir",
    kind: FlagKind::Destination,
    secret: false,
    purpose: "The directory the produced files go into; it must already exist, and it is the \
              directory this command is run in when it is left out",
};

/// The document tool's shared `file_name` argument: the base name of what it
/// produces, which every one of its operations takes — so every one of them
/// carries it.
const DOCUMENT_FILE_NAME_FLAG: Flag = Flag {
    arg: "file_name",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The base name of the produced file, without its extension (which follows the \
              format); a free name is used, so nothing already there is overwritten",
};

/// The document tool's `format` argument — `create` only.
const DOCUMENT_FORMAT_FLAG: Flag = Flag {
    arg: "format",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The format to write: docx, pptx, xlsx or pdf",
};

/// The document tool's `content` argument — `create`'s blocks, for every format
/// but xlsx.
const DOCUMENT_CONTENT_FLAG: Flag = Flag {
    arg: "content",
    kind: FlagKind::Structured(&["image"]),
    secret: false,
    purpose: "The blocks to lay down, as a JSON array — one object per block, with the \
              'text'/'table'/'image'/'...' kind it names beside its own fields",
};

/// The document tool's `sheets` argument — `create`'s worksheets.
const DOCUMENT_SHEETS_FLAG: Flag = Flag {
    arg: "sheets",
    kind: FlagKind::Structured(&[]),
    secret: false,
    purpose: "The worksheets to write, as a JSON array — one object per sheet with its 'name', \
              its 'rows' and its own fields",
};

/// The document tool's `metadata` argument — every operation that writes a PDF.
const DOCUMENT_METADATA_FLAG: Flag = Flag {
    arg: "metadata",
    kind: FlagKind::Structured(&[]),
    secret: false,
    purpose: "A PDF's own description, as a JSON object keyed {{pdf_metadata_fields}}; a named \
              field is set (a date is {{pdf_date_format}}), an empty string clears one, a field \
              you do not name is left as the file has it, and an existing modified date advances \
              to the write",
};

/// The document tool's `path` argument: the one input file most of its
/// operations act on.
const DOCUMENT_PATH_FLAG: Flag = Flag {
    arg: "path",
    kind: FlagKind::Path,
    secret: false,
    purpose: "The input file to work on",
};

/// The document tool's `paths` argument — `pdf_merge`'s inputs, in merge order.
const DOCUMENT_PATHS_FLAG: Flag = Flag {
    arg: "paths",
    kind: FlagKind::PathList,
    secret: false,
    purpose: "One input PDF, in the order it must be merged; repeat once per file, or name a \
              file holding them as a JSON array with '@<file>'",
};

/// The document tool's `pages` argument: which pages an operation acts on.
const DOCUMENT_PAGES_FLAG: Flag = Flag {
    arg: "pages",
    kind: FlagKind::List,
    secret: false,
    purpose: "One page entry: \"all\", a page, a range, or a comma list like \"1-3,5\" (\"all\" \
              stands alone); repeat once per entry, or name a file holding them as a JSON array \
              with '@<file>'; left out, every page",
};

/// The document tool's `edits` argument for `xlsx_edit`, which places no
/// picture.
const XLSX_EDITS_FLAG: Flag = Flag {
    arg: "edits",
    kind: FlagKind::Structured(&[]),
    secret: false,
    purpose: EDITS_PURPOSE,
};

/// The same argument for `docx_edit` and `pptx_edit`, whose `add_image` names a
/// reference file — so their list resolves that `image` where the command runs.
const IMAGE_EDITS_FLAG: Flag = Flag {
    arg: "edits",
    kind: FlagKind::Structured(&["image"]),
    secret: false,
    purpose: EDITS_PURPOSE,
};

/// What the `edits` argument is, as every operation that takes it states it.
const EDITS_PURPOSE: &str = "The edits to apply, as a JSON array — one object per edit, naming the \
                              operation it is ('format_text', 'set_cell', 'add_slide', …) beside \
                              its own fields";

/// The document tool's `values` argument: the placeholders a template fill or a
/// form fill writes.
const DOCUMENT_VALUES_FLAG: Flag = Flag {
    arg: "values",
    kind: FlagKind::Structured(&[]),
    secret: false,
    purpose: "The values to write, as a JSON object keyed by placeholder name (or form field \
              name)",
};

/// The document tool's `degrees` argument — `pdf_rotate`.
const DOCUMENT_DEGREES_FLAG: Flag = Flag {
    arg: "degrees",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The rotation to add, in degrees (a multiple of the tool's own step)",
};

/// The document tool's `text` argument — `pdf_text`.
const DOCUMENT_TEXT_FLAG: Flag = Flag {
    arg: "text",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The text to draw",
};

/// The document tool's `image` argument — `pdf_image`.
const DOCUMENT_IMAGE_FLAG: Flag = Flag {
    arg: "image",
    kind: FlagKind::Path,
    secret: false,
    purpose: "The PNG or JPEG to place",
};

/// The document tool's `x` argument — the `pdf_text`/`pdf_image` position.
const DOCUMENT_X_FLAG: Flag = Flag {
    arg: "x",
    kind: FlagKind::Number,
    secret: false,
    purpose: "Across the page, in points",
};

/// The document tool's `y` argument — the `pdf_text`/`pdf_image` position.
const DOCUMENT_Y_FLAG: Flag = Flag {
    arg: "y",
    kind: FlagKind::Number,
    secret: false,
    purpose: "Down the page, in points",
};

/// The document tool's `size` argument — `pdf_text`'s font size.
const DOCUMENT_SIZE_FLAG: Flag = Flag {
    arg: "size",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The font size, in points",
};

/// The document tool's `color` argument — `pdf_text`'s colour.
const DOCUMENT_COLOR_FLAG: Flag = Flag {
    arg: "color",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The text colour, as hex digits",
};

/// The document tool's `stamp` argument — `pdf_text`.
const DOCUMENT_STAMP_FLAG: Flag = Flag {
    arg: "stamp",
    kind: FlagKind::Present,
    secret: false,
    purpose: "Draw a box around the text",
};

/// The document tool's `rotate` argument — `pdf_text`.
const DOCUMENT_ROTATE_FLAG: Flag = Flag {
    arg: "rotate",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The text's own rotation, in degrees",
};

/// The document tool's `width` argument — `pdf_image`'s placed width.
const DOCUMENT_WIDTH_FLAG: Flag = Flag {
    arg: "width",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The placed width, in points",
};

/// The document tool's `height` argument — `pdf_image`'s placed height.
const DOCUMENT_HEIGHT_FLAG: Flag = Flag {
    arg: "height",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The placed height, in points",
};

/// The document tool's `flatten` argument — `pdf_form_fill`.
const DOCUMENT_FLATTEN_FLAG: Flag = Flag {
    arg: "flatten",
    kind: FlagKind::Present,
    secret: false,
    purpose: "Bake the fields into the page content and remove them",
};

/// The picture tool's `prompt` argument.
const IMAGE_PROMPT_FLAG: Flag = Flag {
    arg: "prompt",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The text description of the picture to produce (required)",
};

/// The picture tool's `images` argument: the reference pictures a
/// picture-from-pictures request is given.
const IMAGE_IMAGES_FLAG: Flag = Flag {
    arg: "images",
    kind: FlagKind::PathList,
    secret: false,
    purpose: "One reference picture, by path; repeat once per picture, or name a file holding \
              them as a JSON array with '@<file>' — given any, the request is \
              picture-to-picture rather than text-to-picture",
};

/// The picture tool's `aspect_ratio` argument.
const IMAGE_ASPECT_RATIO_FLAG: Flag = Flag {
    arg: "aspect_ratio",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The aspect ratio, like '16:9', '1:1' or '4:3'",
};

/// The picture tool's `size` argument.
const IMAGE_SIZE_FLAG: Flag = Flag {
    arg: "size",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The picture's size — a tier like '1K'/'2K' or explicit pixels like '1024x1024'; \
              a model that cannot express the value is not sent the request",
};

/// The generating video tool's `prompt` argument.
const VIDEO_PROMPT_FLAG: Flag = Flag {
    arg: "prompt",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The text description of the clip to produce (required)",
};

/// The generating video tool's `images` argument: its one reference picture.
const VIDEO_IMAGES_FLAG: Flag = Flag {
    arg: "images",
    kind: FlagKind::PathList,
    secret: false,
    purpose: "The path of ONE reference picture to start the clip from — at most one is \
              accepted, and given it the request is picture-to-clip rather than \
              text-to-clip",
};

/// The video tools' `duration` argument.
const VIDEO_DURATION_FLAG: Flag = Flag {
    arg: "duration",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The clip's length in seconds — the values the account's own model takes are the \
              ones it takes",
};

/// The generating video tool's `resolution` argument.
const VIDEO_RESOLUTION_FLAG: Flag = Flag {
    arg: "resolution",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The resolution to ask the model for, as that model states it",
};

/// The video tools' `aspect_ratio` argument.
const VIDEO_ASPECT_RATIO_FLAG: Flag = Flag {
    arg: "aspect_ratio",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The aspect ratio to ask the model for, as that model states it",
};

/// The generating video tool's `size` argument.
const VIDEO_SIZE_FLAG: Flag = Flag {
    arg: "size",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The exact size, as WxH (like '1280x720')",
};

/// The generating video tool's `generate_audio` argument.
const VIDEO_GENERATE_AUDIO_FLAG: Flag = Flag {
    arg: "generate_audio",
    kind: FlagKind::Present,
    secret: false,
    purpose: "Ask for an audio track, where the model offers one — omitted, the model's own \
              default applies and `=false` asks for a silent clip",
};

/// The generating video tool's `seed` argument.
const VIDEO_SEED_FLAG: Flag = Flag {
    arg: "seed",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The seed for a repeatable generation",
};

/// The editing video tool's `video_url` argument: the clip to edit, and the
/// restyling form's own input.
const EDIT_VIDEO_URL_FLAG: Flag = Flag {
    arg: "video_url",
    kind: FlagKind::Path,
    secret: false,
    purpose: "The clip to edit: a local path or a public https URL — required for a restyle, \
              and mutually exclusive with the first and last frames",
};

/// The editing video tool's `instruction` argument.
const EDIT_INSTRUCTION_FLAG: Flag = Flag {
    arg: "instruction",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The edit to apply, described in words (required)",
};

/// The editing video tool's `duration` argument.
const EDIT_DURATION_FLAG: Flag = Flag {
    arg: "duration",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The produced clip's length in seconds — the values the account's own model takes \
              are the ones it takes",
};

/// The editing video tool's `images` argument: the reference pictures that guide
/// a restyle.
const EDIT_IMAGES_FLAG: Flag = Flag {
    arg: "images",
    kind: FlagKind::PathList,
    secret: false,
    purpose: "One reference picture guiding the restyle, by local path or public https URL; \
              repeat once per picture, or name a file holding them as a JSON array with \
              '@<file>' — needs a clip, and excludes the first and last frames",
};

/// The editing video tool's `first_frame` argument — the frame-anchor form's
/// first anchor.
const EDIT_FIRST_FRAME_FLAG: Flag = Flag {
    arg: "first_frame",
    kind: FlagKind::Path,
    secret: false,
    purpose: "The picture to use as the exact first frame, by local path or public https URL — \
              the frame-anchor form, which excludes a clip and reference pictures",
};

/// The editing video tool's `last_frame` argument — the frame-anchor form's
/// second anchor.
const EDIT_LAST_FRAME_FLAG: Flag = Flag {
    arg: "last_frame",
    kind: FlagKind::Path,
    secret: false,
    purpose: "The picture to use as the exact last frame, by local path or public https URL — \
              the frame-anchor form, which excludes a clip and reference pictures",
};

/// The `user` argument of the grant actions, shared by `grant-tool` and
/// `revoke-tool` so the two cannot describe it differently.
const USER_FLAG: Flag = Flag {
    arg: "user",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The user's name",
};

/// The `tool` argument of the grant actions — see [`USER_FLAG`].
const CUSTOM_TOOL_FLAG: Flag = Flag {
    arg: "tool",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The custom tool's name — a script in the 'shared' folder of your personal workspace",
};

/// One of the command lines this module fronts, resolved from its word by
/// [`family`]. Opaque on purpose: a caller either has one — the word names a
/// family — or it does not, so the gate and the dispatch cannot disagree and
/// the registry is looked up once.
pub struct Cli(&'static Family);

/// Resolve a command-line word to its family: `Some` when `mahbot <word>` is
/// one of these command lines, `None` when the caller must keep dispatching.
#[must_use]
pub fn family(word: &str) -> Option<Cli> {
    FAMILIES.iter().find(|family| family.word == word).map(Cli)
}

/// Whether this command line fronts `tool` by name.
///
/// The instance serves an operation only for a tool a family fronts, so the
/// surface reachable through the channel is exactly the surface these
/// frontends publish (and exactly what the registry tests tie to the tools'
/// own action sets) — never the owner's whole toolset.
pub(crate) fn fronts_tool(tool: &str) -> bool {
    family_tools().any(|fronted| fronted == tool)
}

/// Every tool the families front, in registry order.
fn family_tools() -> impl Iterator<Item = &'static str> {
    FAMILIES
        .iter()
        .flat_map(|family| family.tools.iter().copied())
}

/// The family words, in the order the top-level help lists them.
pub(crate) fn family_words() -> impl Iterator<Item = &'static str> {
    FAMILIES.iter().map(|family| family.word)
}

/// The family words that make files ([`produces_files`]), as the read-only
/// surfaces name them: backticked, joined by a slash (`document`/`image`/`video`/
/// `vector`). Written here alone, so a family added to the registry is named — and
/// refused — by every one of them without an edit, and none can go stale.
pub(crate) fn producing_families() -> String {
    FAMILIES
        .iter()
        .filter(|family| produces_files(family))
        .map(|family| format!("`{}`", family.word))
        .collect::<Vec<_>>()
        .join("/")
}

// ── Parsing ───────────────────────────────────────────────────────

/// What an operation's flag tail turned out to be.
enum Tail {
    /// The caller asked for the operation's own help — a `-h`/`--help` where a
    /// flag could stand.
    Help,
    /// The call to send.
    Call(Parsed),
}

/// One parsed command line: what is sent with the call.
struct Parsed {
    /// The tool's args object: an operation with an `action` carries it, one
    /// whose tool takes none has no `action` key at all, plus exactly the flags
    /// that were given. An absent flag leaves its key ABSENT, never null — the
    /// tools tell absent from null, and `pause`/`resume` read the absence of
    /// `workspace` as the fleet-wide form.
    args: serde_json::Value,
    /// The values of the `secret` flags that were given, in the order they were
    /// given, as the values that actually travelled: a secret handed over as
    /// `@<file>` is the FILE's content, which is what must never be echoed,
    /// recorded or reported back. Nothing else is ever sent.
    secrets: Vec<String>,
    /// The directory the produced files must go into, exactly as the caller
    /// spelled it — `None` when the destination flag was left out. The
    /// DESTINATION is not one of the tool's arguments (see
    /// [`FlagKind::Destination`]): it travels beside them, and it is resolved
    /// into a directory before anything is sent ([`resolve_output_dir`]).
    output_dir: Option<String>,
}

/// The command-line spelling of a flag's argument: the tool's own argument name
/// with `_` written `-` (`ticket_id` → `--ticket-id`).
fn flag_spelling(arg: &str) -> String {
    arg.replace('_', "-")
}

/// Parse one operation's flag tail — the tokens after the operation word.
///
/// The grammar itself lives in [`grammar`]; what is this frontend's own is the
/// handling under it. A `List` flag accumulates one array entry per occurrence,
/// an empty value emptying it (see [`FlagKind::List`]). Everything a reason says
/// is a flag's NAME and never a flag's VALUE: a secret the caller handed over
/// must not travel back out in an error text, so a flag is refused by its
/// `--spelling` alone (`--full=<value>` names the flag, never the value), and a
/// token that is no flag at all is not echoed either — a secret mistyped as a
/// positional is still a secret. A mistake this frontend can judge by itself — a
/// token that is not a flag, a flag this operation does not take, a missing
/// value, a flag given twice, a value in the FILE form the caller got wrong — is
/// returned as its reason, for [`Exit::Usage`]. A flag the operation DOES take
/// with a value the tool will not accept is not a mistake this command can
/// judge, so it is sent on: the product refuses it, and that is a failed
/// operation — which is why a `Number` or a `Structured` value that is no number
/// and no JSON travels as the text it is rather than being refused here.
///
/// `action` is the tool's own action for the call, and `None` where the tool
/// takes none at all (a standalone family's own word, or a producing tool's own
/// operation).
fn parse_flags(
    action: Option<&'static str>,
    operation_word: &str,
    flags: &'static [Flag],
    tokens: &[String],
) -> Result<Tail, String> {
    let mut args = serde_json::Map::new();
    if let Some(action) = action {
        args.insert(
            "action".to_string(),
            serde_json::Value::String(action.to_string()),
        );
    }
    let mut secrets = Vec::new();
    let mut output_dir: Option<String> = None;
    // The flags already given, so a `Value`/`Present` repeat is refused instead
    // of silently keeping the last one, and the `List` entries, which only
    // become the array at the end.
    let mut given: Vec<&'static str> = Vec::new();
    let mut lists: Vec<(&'static str, Vec<serde_json::Value>)> = Vec::new();
    let spellings: Vec<String> = flags.iter().map(|flag| flag_spelling(flag.arg)).collect();

    let mut rest = tokens.iter();
    while let Some(token) = rest.next() {
        let (name, inline) = match grammar::read_token(token)? {
            grammar::Token::Help => return Ok(Tail::Help),
            grammar::Token::Flag { name, inline } => (name, inline),
        };
        let Some(index) = spellings.iter().position(|spelling| spelling == name) else {
            return Err(grammar::unknown_flag(name, operation_word));
        };
        let flag = &flags[index];

        if !matches!(flag.kind, FlagKind::List | FlagKind::PathList) {
            if given.contains(&flag.arg) {
                return Err(grammar::repeated_flag(name));
            }
            given.push(flag.arg);
        }

        match flag.kind {
            FlagKind::Present => {
                let value = match inline {
                    Some(text) => text.parse::<bool>().map_err(|_| bool_switch_misuse(name))?,
                    None => true,
                };
                args.insert(flag.arg.to_string(), serde_json::Value::Bool(value));
            }
            FlagKind::Value
            | FlagKind::Path
            | FlagKind::Number
            | FlagKind::List
            | FlagKind::PathList
            | FlagKind::Structured(_)
            | FlagKind::Destination => {
                let value = match inline {
                    Some(value) => value.to_string(),
                    None => rest
                        .next()
                        .cloned()
                        .ok_or_else(|| grammar::flag_value_missing(name))?,
                };
                take_value(flag, &value, &mut args, &mut lists, &mut output_dir)?;
                if flag.secret {
                    // A secret is what the flag actually handed over, never the
                    // spelling that named it: the file form makes the FILE's
                    // content the value (see `Parsed::secrets`).
                    secrets.extend(held_texts(flag, &args, &lists));
                }
            }
        }
    }

    for (arg, entries) in lists {
        args.insert(arg.to_string(), serde_json::Value::Array(entries));
    }
    Ok(Tail::Call(Parsed {
        args: serde_json::Value::Object(args),
        secrets,
        output_dir,
    }))
}

/// Put one value-carrying flag's value where it belongs: into the args object
/// for every kind a tool reads, and aside for the destination, which no tool
/// reads (see [`FlagKind::Destination`]).
///
/// The `@` file form and the `./@…` escape are read off the value once, here: the
/// escape is the caller's way to write a value that itself begins with `@` — the
/// one spelling a file name cannot be confused with — and it is honoured by every
/// kind, while reading a FILE is the kinds that have that form
/// ([`FlagKind::has_file_form`]). A value whose flag names a path itself
/// (`--path`, `--output-dir`) is never read from a file: it names the file.
fn take_value(
    flag: &Flag,
    value: &str,
    args: &mut serde_json::Map<String, serde_json::Value>,
    lists: &mut Vec<(&'static str, Vec<serde_json::Value>)>,
    output_dir: &mut Option<String>,
) -> Result<(), String> {
    let arg = flag.arg.to_string();
    let escaped = value.strip_prefix("./@").map(|rest| format!("@{rest}"));
    let value: &str = escaped.as_deref().unwrap_or(value);
    let file: Option<&str> = if escaped.is_none() && flag.kind.has_file_form() {
        value.strip_prefix('@')
    } else {
        None
    };
    match flag.kind {
        FlagKind::Destination => *output_dir = Some(value.to_string()),
        FlagKind::Value => {
            let held = match file {
                Some(file) => text_from_file(flag, file)?,
                None => value.to_string(),
            };
            args.insert(arg, serde_json::Value::String(held));
        }
        FlagKind::Path => {
            args.insert(arg, serde_json::Value::String(caller_path(flag, value)?));
        }
        FlagKind::Number => {
            args.insert(arg, number_or_text(value));
        }
        FlagKind::Structured(fields) => {
            let mut structured = match file {
                Some(file) => json_from_file(flag, file)?,
                None => serde_json::from_str(value)
                    .unwrap_or_else(|_| serde_json::Value::String(value.to_string())),
            };
            resolve_embedded_paths(flag, fields, &mut structured)?;
            args.insert(arg, structured);
        }
        FlagKind::List | FlagKind::PathList => {
            // One array entry per occurrence, verbatim: the entry is never split
            // on its commas, so `name:kind1,kind2` stays one tool specification
            // and a `List` of a comma-carrying value needs no escaping. An
            // occurrence whose value is `@<file>` adds the entries that file
            // holds, so a list too long for one command line travels whole. An
            // EMPTY value empties the list as of that point — see
            // [`FlagKind::List`]. A path list resolves every entry the way a
            // single path is resolved.
            let named = matches!(flag.kind, FlagKind::PathList);
            let entries = if let Some(index) = lists.iter().position(|(held, _)| *held == flag.arg)
            {
                &mut lists[index].1
            } else {
                lists.push((flag.arg, Vec::new()));
                let index = lists.len() - 1;
                &mut lists[index].1
            };
            // A path list resolves every entry the way a single path is resolved.
            let mut entry = |value: &str| -> Result<(), String> {
                let held = if named {
                    caller_path(flag, value)?
                } else {
                    value.to_string()
                };
                entries.push(serde_json::Value::String(held));
                Ok(())
            };
            match file {
                Some(file) => {
                    for held in list_entries_from_file(flag, file)? {
                        entry(held.as_str().expect("a list entry is a string"))?;
                    }
                }
                None if value.is_empty() => entries.clear(),
                None => entry(value)?,
            }
        }
        FlagKind::Present => unreachable!("a switch carries no value"),
    }
    Ok(())
}

/// The texts a flag's own argument now holds, as the values a secret flag really
/// handed over — for the file forms, what the file held rather than the spelling
/// that named it.
fn held_texts(
    flag: &Flag,
    args: &serde_json::Map<String, serde_json::Value>,
    lists: &[(&'static str, Vec<serde_json::Value>)],
) -> Vec<String> {
    if let Some((_, entries)) = lists.iter().find(|(held, _)| *held == flag.arg) {
        return entries
            .iter()
            .filter_map(|entry| entry.as_str().map(str::to_string))
            .collect();
    }
    args.get(flag.arg)
        .and_then(|value| value.as_str().map(str::to_string))
        .into_iter()
        .collect()
}

/// The JSON a `Number` flag carries: the value read as the number it is, and the
/// text it is otherwise — whether the tool takes the value is the tool's to
/// judge, never this frontend's.
fn number_or_text(value: &str) -> serde_json::Value {
    serde_json::from_str::<serde_json::Value>(value)
        .ok()
        .filter(serde_json::Value::is_number)
        .unwrap_or_else(|| serde_json::Value::String(value.to_string()))
}

/// A switch given a value that is no boolean: the form is spelled out, so the
/// caller is told what to write and not only what was wrong.
fn bool_switch_misuse(name: &str) -> String {
    format!("flag '--{name}' is a switch: write it bare for true, or '--{name}=false' for false")
}

/// A path a flag names, as the caller meant it and checked.
///
/// `~` is expanded against the home directory and a relative path is resolved in
/// the directory this command was run in — the product has a working directory of
/// its own, and it is not this one, so a relative path sent on as written would
/// name a different file there, while a shell driving this command means the file
/// it can see. The named reference material must then be THERE and readable: that
/// is the caller's own to get right, so a path that is missing or unreadable is
/// misuse, judged before anything is sent. It must also be a REGULAR file: the
/// product reads what it is given with no timeout of its own, so a pipe with no
/// writer would hold the daemon — and the operation's flight guard — forever, and
/// a device would read without end. Only metadata is looked at here, so the
/// refusal itself cannot block, and a URL — which only the product can fetch — is
/// left alone.
fn caller_path(flag: &Flag, value: &str) -> Result<String, String> {
    if value.is_empty() || crate::util::is_http_url(value) {
        return Ok(value.to_string());
    }
    let path = {
        let expanded = crate::util::expand_tilde(value);
        if expanded.is_absolute() {
            expanded.to_string_lossy().into_owned()
        } else {
            // No working directory to resolve against: the value travels as the
            // caller wrote it, and the product says why it cannot open it.
            std::env::current_dir().map_or_else(
                |_| value.to_string(),
                |here| here.join(expanded).to_string_lossy().into_owned(),
            )
        }
    };
    let spelling = flag_spelling(flag.arg);
    let unreadable = |spelling: &str| {
        format!("flag '--{spelling}' names a reference file that is not there or cannot be read")
    };
    let Ok(meta) = std::fs::metadata(&path) else {
        return Err(unreadable(&spelling));
    };
    if !meta.is_file() {
        return Err(format!(
            "flag '--{spelling}' names something that is not a regular file; a reference file must \
             be one"
        ));
    }
    if std::fs::File::open(&path).is_err() {
        return Err(unreadable(&spelling));
    }
    Ok(path)
}

/// Resolve every reference file a structured value names under one of `fields` —
/// a document's image block, an edit that places a picture.
///
/// The product resolves those against its OWN directories (a workspace's
/// `generated/`, a personal workspace's `uploads/`), so a relative path sent on
/// as written would read a file the caller never named. Each string under such a
/// field is resolved exactly like a `--path` value ([`caller_path`]) — in the
/// directory this command runs in, `~` against the home directory, a URL left
/// alone — and so is every string of an array under one; a member of another
/// shape is left to the tool, which judges it the same way wherever it came from.
fn resolve_embedded_paths(
    flag: &Flag,
    fields: &'static [&'static str],
    value: &mut serde_json::Value,
) -> Result<(), String> {
    match value {
        serde_json::Value::Object(members) => {
            for (key, member) in members.iter_mut() {
                if fields.contains(&key.as_str()) {
                    resolve_embedded_path(flag, member)?;
                } else {
                    resolve_embedded_paths(flag, fields, member)?;
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                resolve_embedded_paths(flag, fields, item)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// One member a structured value names a reference file under: the string it is,
/// or every string of the array it is.
fn resolve_embedded_path(flag: &Flag, member: &mut serde_json::Value) -> Result<(), String> {
    match member {
        serde_json::Value::String(text) => *text = caller_path(flag, text)?,
        serde_json::Value::Array(items) => {
            for item in items {
                if let serde_json::Value::String(text) = item {
                    *text = caller_path(flag, text)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Upper bound on the file a flag's `@<file>` form may read: a value travels as
/// one command-line argument, so a file far past any command line's length is a
/// mistaken path rather than a value to send — and a bound is what keeps a
/// mistyped endless file from becoming an out-of-memory.
const MAX_VALUE_FILE_BYTES: u64 = 8 * 1024 * 1024;

/// The text of the file a flag's `@<file>` form names — the ONE read every value
/// form shares, so no two of them can guard differently.
///
/// The file must be a REGULAR file of a bounded size. The parsing path has no
/// timeout of its own, so a pipe (`--prompt @<fifo>`) would hold the command
/// forever and an endless device (`--prompt @/dev/zero`) would read until memory
/// ran out. Every one of those is the caller's own misuse, judged before anything
/// is sent — and the reason names the FLAG and never the path, like every other
/// reason.
fn value_file_text(flag: &Flag, file: &str, what: &str) -> Result<String, String> {
    let spelling = flag_spelling(flag.arg);
    let unusable = |why: &str| {
        format!(
            "flag '--{spelling}' names a file that {why}; '@<file>' must name a file holding {what}"
        )
    };
    let path = crate::util::expand_tilde(file);
    let Ok(meta) = std::fs::metadata(&path) else {
        return Err(unusable("could not be read"));
    };
    if !meta.is_file() {
        return Err(unusable("is not a regular file"));
    }
    if meta.len() > MAX_VALUE_FILE_BYTES {
        return Err(unusable(&format!(
            "is over {} MB",
            MAX_VALUE_FILE_BYTES / (1024 * 1024)
        )));
    }
    std::fs::read_to_string(&path).map_err(|_| unusable("could not be read"))
}

/// The text the file a value flag's `@<file>` form names holds, without one
/// trailing line break — a file's last newline is the file's, not the value's.
///
/// The file form is this command line's own, so a file it cannot use is the
/// caller's misuse and is judged here: the reason names the FLAG and never the
/// path (the same rule every other reason follows).
fn text_from_file(flag: &Flag, file: &str) -> Result<String, String> {
    let text = value_file_text(flag, file, "the value")?;
    Ok(text
        .strip_suffix('\n')
        .map_or(text.as_str(), |rest| {
            rest.strip_suffix('\r').unwrap_or(rest)
        })
        .to_string())
}

/// Read the file a `Structured` flag's `@<file>` form names and take the JSON it
/// holds.
///
/// The file form is this command line's own, so a file it cannot use is the
/// caller's misuse and is judged here: the reason names the FLAG and never the
/// path (the same rule every other reason follows).
fn json_from_file(flag: &Flag, file: &str) -> Result<serde_json::Value, String> {
    let spelling = flag_spelling(flag.arg);
    let text = value_file_text(flag, file, "the value")?;
    serde_json::from_str(&text).map_err(|_| {
        format!(
            "flag '--{spelling}' names a file that does not hold the value; '@<file>' must \
             name a file holding it as JSON"
        )
    })
}

/// Read the file a `List` flag's `@<file>` form names and take the entries it
/// holds — a JSON array of strings, and nothing else.
fn list_entries_from_file(flag: &Flag, file: &str) -> Result<Vec<serde_json::Value>, String> {
    let spelling = flag_spelling(flag.arg);
    let text = value_file_text(flag, file, "the entries as a JSON array")?;
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(serde_json::Value::Array(entries))
            if entries.iter().all(serde_json::Value::is_string) =>
        {
            Ok(entries)
        }
        _ => Err(format!(
            "flag '--{spelling}' names a file that does not hold the entries; '@<file>' must name \
             a file holding them as a JSON array of strings"
        )),
    }
}

// ── Help ──────────────────────────────────────────────────────────

/// The exit classes and the one sentence that makes them the contract — shared
/// by every help (a family's and each operation's) so no two can word it
/// differently.
const EXIT_CLASS_HELP: &str = "\nExit codes:\n  0  the operation was carried out\n  1  the \
     operation failed — the product refused it, carried it out only in part, or the exchange was \
     severed after the call went out (an operation that may be under way is never the \
     environment's class)\n  2  the environment — nothing ran: the product is not up, its \
     channel was not reachable, or it could not take the operation on\n  3  this command was \
     misused\nA script reads the exit code, not the text: a refusal still prints the product's \
     own answer on stdout, so only the code says whether the operation happened.\n";

/// What one of these command lines IS — shared by every help for the same
/// reason as the exit classes: a script author entering through one operation's
/// own help must learn the same contract as one reading the family's.
const RUNNING_CONTRACT_HELP: &str = "\nA frontend onto the RUNNING product, acting with the \
     owner's authority and with no agent session behind it: the operation is carried into the \
     instance that is already up and the owner's own tool runs it there — the same validation, \
     and the same immediate effect (a value it writes is in use at once, with no restart). The \
     product must be running: there is no way to run these operations against its stores \
     directly.\n";

/// Where the two streams go — shared by every help, like the two blocks above.
const OUTPUT_HELP: &str = "\nOutput:\n  stdout  the product's answer, word for word\n  stderr  \
     this command's own diagnostics\n";

/// How a value may be given from a file — shared by every help, because the form
/// belongs to this command line rather than to any one family's flags (see
/// [`FlagKind::has_file_form`]).
const VALUE_FILE_HELP: &str = "\nValues:\n  A text value, a list or a structure may also be given \
     from a FILE by writing `@<file>`: a text value is the file's whole text (without one trailing \
     line break), while a list and a structure must hold JSON there. Write `./@…` for a value that \
     itself begins with `@` — the `./` is dropped and no file is read. A flag whose value names a \
     file or a directory itself (`--path`, `--output-dir`) takes no file form.\n";

/// What a producing family's help says beyond the tool's own flags: where
/// produced files go, and how the paths the caller hands over are read.
///
/// Shared by every help of such a family (the family's and each operation's), so
/// a script author reads the same thing wherever it enters. The destination's row
/// is rendered from the registry ([`OUTPUT_DIR_FLAG`]), so its spelling and its
/// description are stated once. The "Paths" paragraph belongs here for the same
/// reason: a path this frontend resolves is its own doing, not a tool's.
fn producing_help() -> String {
    let mut out = String::from("\nWhere produced files go:\n");
    out.push_str(&flag_rows(std::iter::once(&OUTPUT_DIR_FLAG)));
    out.push_str(
        "\nThe names are this command's own: a free one is chosen so nothing already there is \
         overwritten, and an operation producing several files puts all of them there. The \
         product's own data directory and its temporary areas are refused, because the product's \
         own cleanup would take a file left there.\n\nPaths:\n  Every path this command takes as a \
         flag — the destination, a reference file, and every path a value carries, a structured \
         value's own included — is resolved where this command runs: a relative one in the \
         directory it was run in, and `~` against your home directory; a reference file must be \
         there and readable.\n",
    );
    out
}

/// `mahbot <word> -h`: what this family's command line is and what a caller
/// reads back from it.
fn family_help(family: &Family) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "mahbot {} — {}", family.word, family.purpose);
    out.push_str(RUNNING_CONTRACT_HELP);
    match family.surface {
        Surface::Operations(operations) => {
            let listed = family
                .tools
                .iter()
                .map(|tool| format!("`{tool}`"))
                .collect::<Vec<_>>()
                .join(" and ");
            let (noun, them, verb) = if family.tools.len() == 1 {
                ("tool", "that tool", "offers")
            } else {
                ("tools", "those tools", "offer")
            };
            let _ = write!(
                out,
                "\nEvery operation below is run by the same {listed} {noun} the owner's assistant \
                 calls, so this command line offers exactly the operations {them} {verb}.\n"
            );
            let _ = write!(out, "\nUsage: {}\n", family_syntax(family));
            out.push_str("\nOperations:\n");
            let width = operations
                .iter()
                .map(|operation| operation_word(family, operation).len())
                .max()
                .unwrap_or_default();
            for operation in operations {
                let _ = writeln!(
                    out,
                    "  {:<width$}  {}",
                    operation_word(family, operation),
                    operation.purpose
                );
            }
            if produces_files(family) {
                out.push_str(&producing_help());
            }
            out.push_str(if produces_files(family) {
                "\nFlags:\n  The one flag every operation takes is the destination above; every \
                 other flag belongs to the operation whose tool argument it names.\n\n  -h, --help  \
                 Print this help and exit\n"
            } else {
                "\nFlags:\n  None are common to every operation: a flag belongs to the operation \
                 whose tool argument it names.\n\n  -h, --help  Print this help and exit\n"
            });
        }
        // The family word IS the operation, so there is no operation list below
        // and nothing to say about how one of them is run.
        Surface::Standalone(flags) => {
            let _ = write!(out, "\nUsage: {}\n", family_syntax(family));
            out.push_str("\nFlags:\n");
            out.push_str(&flag_rows(flags.iter()));
            out.push_str("\n  -h, --help  Print this help and exit\n");
        }
    }
    out.push_str(EXIT_CLASS_HELP);
    out.push_str(OUTPUT_HELP);
    out.push_str(VALUE_FILE_HELP);
    out
}

/// A flag's purpose with the shared vocabulary filled in: a description field set
/// or a date grammar the rules state appears in the help as those rules state it,
/// so neither can drift from the schema and the reading.
fn flag_purpose(flag: &Flag) -> String {
    flag.purpose
        .replace(
            "{{pdf_metadata_fields}}",
            &crate::docgen::pdf_metadata_fields_spelled(),
        )
        .replace("{{pdf_date_format}}", crate::docgen::pdf_date_format())
}

/// The flag table of a help: one width-aligned row per flag, a secret's value
/// never echoed here because it is never echoed anywhere.
fn flag_rows<'a>(flags: impl IntoIterator<Item = &'a Flag>) -> String {
    let rows: Vec<(String, String)> = flags
        .into_iter()
        .map(|flag| {
            let mut spelling = format!("--{}", flag_spelling(flag.arg));
            match flag.kind {
                FlagKind::Value => spelling.push_str(" <value>"),
                FlagKind::Path => spelling.push_str(" <path>"),
                FlagKind::Number => spelling.push_str(" <number>"),
                FlagKind::List => spelling.push_str(" <value> (repeatable; empty clears the list)"),
                FlagKind::PathList => {
                    spelling.push_str(" <path> (repeatable; empty clears the list)");
                }
                FlagKind::Structured(_) => spelling.push_str(" <json>"),
                FlagKind::Present => spelling.push_str("[=<true|false>]"),
                FlagKind::Destination => spelling.push_str(" <dir>"),
            }
            let description = if flag.secret {
                format!(
                    "{} — the value is never echoed or recorded",
                    flag_purpose(flag)
                )
            } else {
                flag_purpose(flag)
            };
            (spelling, description)
        })
        .collect();
    let width = rows
        .iter()
        .map(|(spelling, _)| spelling.len())
        .max()
        .unwrap_or_default();
    let mut out = String::new();
    for (spelling, description) in rows {
        let _ = writeln!(out, "  {spelling:<width$}  {description}");
    }
    out
}

/// `mahbot <word> <operation> -h`: what one operation does and the flags it
/// takes — plus the same contract the family help states, because a script
/// author may enter here first.
fn operation_help(family: &Family, operation: &Operation) -> String {
    let word = operation_word(family, operation);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "mahbot {} {} — {}",
        family.word, word, operation.purpose
    );
    let _ = write!(out, "\nUsage: mahbot {} {} [flags]\n", family.word, word);
    if produces_files(family) {
        out.push_str(&producing_help());
    }
    out.push_str("\nFlags:\n");
    // The destination is stated once, in the block above: it is this command
    // line's own operand rather than one of the tool's arguments, so it takes no
    // row of the flag table.
    let flags: Vec<&Flag> = operation
        .flags
        .iter()
        .filter(|flag| !matches!(flag.kind, FlagKind::Destination))
        .collect();
    if flags.is_empty() {
        out.push_str("  none\n");
    }
    out.push_str(&flag_rows(flags));
    out.push_str(RUNNING_CONTRACT_HELP);
    out.push_str(EXIT_CLASS_HELP);
    out.push_str(OUTPUT_HELP);
    out.push_str(VALUE_FILE_HELP);
    out
}

// ── The run ───────────────────────────────────────────────────────

/// Run one family's command line, returning the process exit code.
///
/// The product's own answer goes to STDOUT exactly as the product worded it
/// (never shortened, never reformatted); everything this command says on its
/// own goes to STDERR. Help goes to stdout. The exit code is the thing a script
/// reads — see [`Exit`].
///
/// A producing family's destination is resolved HERE, before the environment is
/// probed, because it is a mistake this command judges by itself — a destination
/// that is missing, unusable or the product's own to sweep is the caller's own
/// misuse, and nothing is sent.
pub async fn run(cli: Cli, args: &[String]) -> i32 {
    let family = cli.0;
    let (tool, parsed) = match family.surface {
        Surface::Operations(operations) => {
            let Some(named) = args.first().map(String::as_str) else {
                return misuse(family, "no operation was given");
            };
            if grammar::is_help(named) {
                crate::util::print_stdout(&family_help(family));
                return Exit::Ok.code();
            }
            let Some(operation) = operations
                .iter()
                .find(|operation| operation_word(family, operation) == named)
            else {
                return misuse(family, &format!("unknown operation '{named}'"));
            };
            match parse_flags(operation.action, named, operation.flags, &args[1..]) {
                Ok(Tail::Call(parsed)) => (operation.tool, parsed),
                // A `-h`/`--help` where a flag could stand is this operation's
                // help — the same rule the browser-automation command follows,
                // and it turns a would-be usage error into help rather than the
                // other way round. A `-h` where a VALUE stands is that value
                // instead (see `parse_flags`): a script must never read success
                // for an operation that did not run.
                Ok(Tail::Help) => {
                    crate::util::print_stdout(&operation_help(family, operation));
                    return Exit::Ok.code();
                }
                Err(reason) => return misuse(family, &reason),
            }
        }
        // The family word IS the operation, and the one tool the family fronts
        // takes no `action`: the first token is already a flag. A bare `mahbot
        // send` therefore sends nothing and the product refuses it as the missing
        // message — a failed operation, exactly as a missing required flag is for
        // every other family (this frontend never judges what the tool requires).
        Surface::Standalone(flags) => {
            let tool = *family
                .tools
                .first()
                .expect("a standalone family fronts exactly one tool");
            match parse_flags(None, family.word, flags, args) {
                Ok(Tail::Call(parsed)) => (tool, parsed),
                Ok(Tail::Help) => {
                    crate::util::print_stdout(&family_help(family));
                    return Exit::Ok.code();
                }
                Err(reason) => return misuse(family, &reason),
            }
        }
    };

    // The destination, on the same footing as the words: a producing family's
    // operation writes where the caller says, and a place this command cannot
    // write into is misuse, judged before the environment is probed.
    let output_dir = if produces_files(family) {
        match resolve_output_dir(parsed.output_dir.as_deref()) {
            Ok(dir) => Some(dir),
            Err(reason) => return misuse(family, &reason),
        }
    } else {
        None
    };

    // These commands act ON the running product and have no other way to run,
    // so the storage root and the instance lock are settled before anything is
    // sent.
    let root = match running_root(family).await {
        Ok(root) => root,
        Err(code) => return code,
    };

    let call = CallRequest {
        tool: tool.to_string(),
        args: parsed.args,
        secrets: parsed.secrets,
        output_dir,
    };
    let decision = verdict(
        family.word,
        crate::db::ipc::ipc_call_with_wait(&root, &call, family.protocol).await,
    );
    if let Some(answer) = decision.answer {
        crate::util::print_stdout(&answer);
    }
    if let Some(diagnostic) = decision.diagnostic {
        crate::util::print_stderr(&diagnostic);
    }
    decision.code.code()
}

/// The running product's storage root, or the class to exit with when there is
/// none to send to.
///
/// These commands act ON the instance and have no other way to run, so the lock
/// is settled before anything is sent. The probe is the same fail-closed one
/// `mahbot debug` routes on: an `Unknown` lock is never read as "no instance is
/// running". A FREE lock on its own is not "nothing is running" either — a
/// product that is restarting or updating itself releases the lock for the
/// seconds the replacement needs to take it — so the probe is ridden out over
/// the same short window the channel itself uses (seconds, not milliseconds: see
/// `MAHBOT_IPC_BOUND_TIMEOUT_SECS`), and only a window that stays free is
/// reported as nothing running.
async fn running_root(family: &Family) -> Result<std::path::PathBuf, i32> {
    let root = match crate::config::default_config_dir() {
        Ok(root) => root,
        Err(e) => {
            crate::util::print_stderr(&format!(
                "mahbot {}: cannot resolve the storage root: {e:#}",
                family.word
            ));
            return Err(Exit::Environment.code());
        }
    };
    let deadline = std::time::Instant::now() + crate::db::ipc::ipc_bound_timeout();
    let mut attempt = 0usize;
    loop {
        match crate::util::lock::instance_lock_state_settled(&root) {
            InstanceLockState::Held => return Ok(root),
            InstanceLockState::Free if std::time::Instant::now() >= deadline => {
                crate::util::print_stderr(&format!(
                    "mahbot {}: no instance is running against {} — these commands act on the \
                     running product: start mahbot and try again",
                    family.word,
                    root.display()
                ));
                return Err(Exit::Environment.code());
            }
            InstanceLockState::Unknown(e) if std::time::Instant::now() >= deadline => {
                crate::util::print_stderr(&format!(
                    "mahbot {}: cannot tell whether a mahbot instance holds {} ({e}) — an \
                     operation is only ever sent to a running product, so none was sent",
                    family.word,
                    root.display()
                ));
                return Err(Exit::Environment.code());
            }
            // Between boots, or a probe that could not settle: wait and look again.
            InstanceLockState::Free | InstanceLockState::Unknown(_) => {}
        }
        tokio::time::sleep(crate::db::ipc::retry_backoff(attempt)).await;
        attempt += 1;
    }
}

/// Report a misuse on STDERR and hand back its class: the reason, then the one
/// line that says how to see the right form. Nothing was sent, and no operation
/// judged anything.
fn misuse(family: &Family, reason: &str) -> i32 {
    crate::util::print_stderr(&format!("mahbot {}: {reason}", family.word));
    crate::util::print_stderr(&format!("run 'mahbot {} -h' for usage.", family.word));
    Exit::Usage.code()
}

// ── The destination ───────────────────────────────────────────────

/// Resolve a producing family's destination — the `--output-dir` value, or the
/// directory this command was run in when it was left out — into the absolute
/// directory the produced files go into, or the reason it cannot be used.
///
/// The destination is the caller's own operand, so it is spelled as the caller
/// sees it and RESOLVED where this command runs, in the directory it was run in
/// — never against the product's own working directory. It must already exist:
/// this command creates no directory for the caller. And it must not be a place
/// the product keeps for itself, because a produced file left there would not
/// stay ([`cleanup_governed`]). Every failure here — a destination that is
/// missing, is not a directory or cannot be written into — is the caller's own
/// misuse, judged before anything is sent.
///
/// The one place a governed destination can slip through is a daemon whose own
/// temp environment differs from this process's: it may scan a root this process
/// cannot name ([`crate::temp::cleanup_scan_roots_unpinned`]).
fn resolve_output_dir(named: Option<&str>) -> Result<String, String> {
    let (raw, what) = match named {
        Some(value) => (
            crate::util::expand_tilde(value),
            format!("the destination '{value}'"),
        ),
        None => (
            std::env::current_dir().map_err(|e| {
                format!(
                    "the directory this command was run in cannot be used as the destination: {e}"
                )
            })?,
            "the directory this command was run in".to_string(),
        ),
    };
    let resolved =
        std::fs::canonicalize(&raw).map_err(|e| format!("{what} cannot be used: {e}"))?;
    if !resolved.is_dir() {
        return Err(format!(
            "{what} is not a directory — the destination is the directory the produced files go \
             into"
        ));
    }
    if let Some(place) = cleanup_governed(&resolved) {
        return Err(format!(
            "{what} is {place} — a produced file must go to a directory of your own"
        ));
    }
    // A destination the caller cannot write into is the caller's own to fix:
    // judged here rather than discovered as a failed operation, so the exit code
    // says "your invocation" and nothing is sent. `access(2)` answers it on unix
    // without writing anything into the caller's directory; off unix there is no
    // such query, and the windowed build that platform serves is outside this
    // contract either way, so the product's own refusal is the answer there.
    #[cfg(unix)]
    if !crate::util::dir_is_writable(&resolved) {
        return Err(format!("{what} cannot be written into"));
    }
    // The plain spelling rather than the platform's verbatim form, so the value
    // that travels to the instance is the one the caller wrote.
    Ok(crate::util::strip_verbatim_prefix(&resolved)
        .to_string_lossy()
        .into_owned())
}

/// The place the product keeps for itself that `dir` lies in, or `None` when the
/// directory is the caller's own to use.
///
/// The roots come from [`crate::temp`] — the module that owns which roots the
/// product's own cleaner may act in, so the command line and the cleaner cannot
/// disagree about a place the cleaner acts in — plus the product's own data
/// directory, whose live stores, personal workspaces (`generated`/`uploads`,
/// which its media sweep empties) and document-kit scratch are all its own. The
/// test is on the RESOLVED path, so a spelling that only looks innocent — a link
/// into one of them — is refused all the same: both sides of every comparison go
/// through [`crate::util::is_within`], which drops the platform's verbatim
/// spelling. A machine's temp area is refused as a whole, whether or not the
/// cleaner happens to sweep it: a produced file belongs in a directory of the
/// caller's own.
fn cleanup_governed(dir: &std::path::Path) -> Option<&'static str> {
    const DATA_DIR: &str = "the product's own data directory, where it keeps its own files";
    const TEMP: &str = "the product's own temporary area, which its cleanup keeps empty";
    if let Ok(root) = crate::config::default_config_dir() {
        let root = std::fs::canonicalize(&root).unwrap_or(root);
        if crate::util::is_within(dir, &root) {
            return Some(DATA_DIR);
        }
    }
    if crate::temp::cleanup_scan_roots_unpinned()
        .iter()
        .any(|root| crate::util::is_within(dir, root))
    {
        return Some(TEMP);
    }
    None
}

/// What an argument the process cannot read as text is told — the same sentence
/// wherever it is met, and it never names the argument: the bytes may be a value
/// the caller meant as a secret.
const NON_TEXT_ARGUMENT: &str = "an argument that is not valid text was given — no entry point \
                                 and no value can be read from it, and it is not printed here; \
                                 check what you passed";

/// Refuse an argument the runner cannot read as text, returning this command
/// line's misuse class.
///
/// The binary needs this before it can tell which entry point it was invoked as:
/// an `argv[1]` that is not text names none of them, and an argument that cannot
/// be read must be answered as the caller's own mistake rather than by the panic
/// the text-only reader raises — a panic would leave the four classes a script
/// reads and print the very bytes the caller passed.
#[must_use]
pub fn refuse_non_text_argument() -> i32 {
    crate::util::print_stderr(&format!("mahbot: {NON_TEXT_ARGUMENT}"));
    Exit::Usage.code()
}

/// What one attempt's result means for the caller.
struct Verdict {
    /// The product's own answer, printed to stdout verbatim — present whenever
    /// the product answered the operation itself, a refusal included.
    answer: Option<String>,
    /// This command's own diagnostic, printed to stderr.
    diagnostic: Option<String>,
    /// The exit class the caller reads.
    code: Exit,
}

/// Classify one attempt at an operation. A pure function of the attempt, so the
/// decision can be read (and tested) with no socket anywhere near it.
///
/// An answer the product gave goes to stdout whatever its outcome: its wording
/// is what the caller is owed, and the exit code — not the text — says whether
/// the operation happened. An operation carried out only in part gets one short
/// diagnostic line as well, because the product's answer names the part that
/// did not succeed in the product's own words and the caller must not be left
/// to read them for a verdict. An operation the product never took on (or
/// cannot take at all) has no answer for the caller: the product's own wording
/// — or this command's, when it could not reach the product — goes to stderr,
/// and the class is the environment's. An exchange severed after the frame went
/// out is the one failure that is NOT the environment's: the operation may have
/// been carried out, so its class is a failed operation, and the diagnostic
/// says plainly that the outcome is unknown and that it was deliberately not
/// sent again.
fn verdict(word: &str, attempt: CallAttempt) -> Verdict {
    match attempt {
        CallAttempt::Answered(reply) => match reply.outcome {
            CallOutcome::Done => Verdict {
                answer: Some(reply.answer),
                diagnostic: None,
                code: Exit::Ok,
            },
            CallOutcome::Partial => Verdict {
                answer: Some(reply.answer),
                diagnostic: Some(format!(
                    "mahbot {word}: the product reports the operation was carried out only in \
                     part — its answer above names the part that did not succeed"
                )),
                code: Exit::Failed,
            },
            CallOutcome::Refused => Verdict {
                answer: Some(reply.answer),
                diagnostic: None,
                code: Exit::Failed,
            },
            // A product that did not take the operation on. `Unavailable` reaches
            // the caller as the instance answered it — an unrestricted socket, a
            // boot that failed, or one that has not finished, whose own sentence
            // says whether a later attempt may help — while `NotAccepted` is
            // folded into `NotServed` by the wire client before the caller sees
            // it. Either way the product answered nothing about the operation
            // itself, so the instance's own sentence is this command's diagnostic
            // here, prefixed like the rest of them.
            CallOutcome::NotAccepted | CallOutcome::Unavailable => Verdict {
                answer: None,
                diagnostic: Some(format!("mahbot {word}: {}", reply.answer)),
                code: Exit::Environment,
            },
        },
        // The operation never reached the product (or the product never took it
        // on): nothing ran, and this command's own wording says what to do next.
        CallAttempt::NotServed(reason) => Verdict {
            answer: None,
            diagnostic: Some(format!("mahbot {word}: {reason}")),
            code: Exit::Environment,
        },
        CallAttempt::TooOld => Verdict {
            answer: None,
            diagnostic: Some(format!(
                "mahbot {word}: the running product is older than this command line and cannot \
                 serve its operations — restart it from this build and try again"
            )),
            code: Exit::Environment,
        },
        // Nothing readable came back after the frame went out — an exchange that
        // was severed, or a reply this build cannot read. The operation may have
        // been carried out, so the class is a failed operation (never the
        // environment's). The conclusion is stated ONCE, and the cause sits in
        // the parentheses: the carried error says what happened, never what the
        // class is or whether it was sent again.
        CallAttempt::Severed(e) => Verdict {
            answer: None,
            diagnostic: Some(format!(
                "mahbot {word}: the outcome of the operation could not be confirmed, so it may \
                 have been carried out, and it was deliberately NOT sent again ({e:#})"
            )),
            code: Exit::Failed,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chrome::contract::OutKind;
    use crate::db::ipc::CallReply;

    /// Every entry point is discoverable from `mahbot -h`: the families from the
    /// registry, and the command lines that belong to none of them — `paused`,
    /// the speech command, the browser CLI, the debugger and the benchmark. One
    /// left out of the top-level usage would be reachable only by someone who
    /// already knew its word; `mahbot audio` above all, which has to be
    /// discoverable on a platform where it can only refuse.
    ///
    /// The families are checked in the derived form ([`family_syntax`]), so a
    /// tool or operation rename cannot leave the top-level help stale; the
    /// non-family entries have no registry to derive from and are checked by
    /// their own word.
    ///
    /// The bare launch is not in the list: it has no word after `mahbot`, so the
    /// shape this loop looks for cannot match it.
    ///
    /// Here — beside the registry — so the library's own test lane holds the tie,
    /// not just the whole-workspace run.
    #[test]
    fn every_entry_point_is_listed_in_the_top_level_usage() {
        for word in ["paused", "audio", "chrome", "debug", "bench-openrouter"] {
            let entry = format!("  mahbot {word} ");
            assert!(
                TOP_LEVEL_USAGE.lines().any(|line| line.starts_with(&entry)),
                "'mahbot {word}' has no entry in the top-level usage"
            );
        }
        for family in FAMILIES {
            let entry = format!("  {}", family_syntax(family));
            assert!(
                TOP_LEVEL_USAGE.lines().any(|line| line.starts_with(&entry)),
                "'{entry}' has no entry in the top-level usage"
            );
        }
    }

    /// The admin Assistant's own toolset — the toolset the command line fronts.
    fn admin_toolset() -> Vec<Box<dyn crate::Tool>> {
        crate::Role::Assistant.tools(
            &crate::workspace::test_ws("test"),
            true,
            std::sync::Arc::new(crate::tools::chrome::ChromeRunSessions::default()),
        )
    }

    /// The parameters schema of one tool.
    fn tool_schema(tool: &str) -> serde_json::Value {
        let tools = admin_toolset();
        crate::tools::find_tool(&tools, tool)
            .unwrap_or_else(|| panic!("the admin Assistant must hold '{tool}'"))
            .parameters_schema()
    }

    /// The action words a tool's schema publishes: its `action` enum where it has
    /// one, and the keys of its `oneOf` entries where each action declares its
    /// own arguments.
    fn tool_actions(schema: &serde_json::Value) -> Vec<&str> {
        let action = &schema["properties"]["action"];
        if let Some(actions) = action["enum"].as_array() {
            return actions
                .iter()
                .map(|action| action.as_str().expect("action words are strings"))
                .collect();
        }
        action["oneOf"]
            .as_array()
            .unwrap_or_else(|| panic!("the tool states its actions as an enum or a oneOf"))
            .iter()
            .flat_map(|entry| {
                entry["properties"]
                    .as_object()
                    .expect("an action entry names the action")
                    .keys()
                    .map(String::as_str)
            })
            .collect()
    }

    /// Whether a tool's actions declare their OWN arguments (`oneOf`) rather
    /// than sharing one argument set beside the action (`enum`). Only the former
    /// can be swept per operation; for the latter the tool's own schema states
    /// the arguments once and every action takes them all.
    fn declares_per_action_arguments(tool: &str) -> bool {
        tool_schema(tool)["properties"]["action"]
            .get("enum")
            .is_none()
    }

    /// The arguments one command line can fill: the tool's whole top-level
    /// argument set, or — for a tool whose actions declare their own arguments —
    /// the action's entry plus the arguments that stand beside the action.
    fn argument_schema(operation: &Operation) -> serde_json::Map<String, serde_json::Value> {
        let schema = tool_schema(operation.tool);
        let top = schema["properties"]
            .as_object()
            .expect("the tool's properties")
            .clone();
        let Some(action) = operation.action else {
            return top;
        };
        // An enum-shaped action states only WHICH action runs: its arguments
        // stand beside it at the tool's top level.
        if !declares_per_action_arguments(operation.tool) {
            return top.into_iter().filter(|(key, _)| key != "action").collect();
        }
        let entries = schema["properties"]["action"]["oneOf"]
            .as_array()
            .expect("the tool's action entries");
        let entry = entries
            .iter()
            .find_map(|entry| entry["properties"].get(action))
            .unwrap_or_else(|| {
                panic!(
                    "'{}' has no schema entry for action '{action}'",
                    operation.tool
                )
            });
        let declared = entry["properties"]
            .as_object()
            .unwrap_or_else(|| panic!("'{action}' must declare its properties"));
        // A root argument an action opts into belongs to that action: keeping the
        // root copy would demand a form for it on an operation whose action
        // refuses it. Only the root arguments no action declares stand beside
        // every action.
        let per_action: Vec<&str> = entries
            .iter()
            .flat_map(|entry| entry["properties"].as_object().into_iter())
            .flat_map(|actions| actions.values())
            .flat_map(|action| action["properties"].as_object().into_iter())
            .flat_map(|properties| properties.keys().map(String::as_str))
            .collect();
        let mut args: serde_json::Map<String, serde_json::Value> = top
            .into_iter()
            .filter(|(key, _)| key != "action" && !per_action.contains(&key.as_str()))
            .collect();
        args.extend(declared.clone());
        args
    }

    /// The words that appear more than once in `words` — a duplicated operation
    /// or flag is a registry bug an equality comparison would otherwise hide.
    fn duplicates<'a>(words: impl IntoIterator<Item = &'a str>) -> Vec<&'a str> {
        let mut seen: Vec<&'a str> = Vec::new();
        let mut repeated: Vec<&'a str> = Vec::new();
        for word in words {
            if seen.contains(&word) {
                if !repeated.contains(&word) {
                    repeated.push(word);
                }
            } else {
                seen.push(word);
            }
        }
        repeated
    }

    /// The registry a test parses against: `mahbot <family> <operation>`.
    fn operation(family_word: &str, operation_word: &str) -> &'static Operation {
        let family = family_of(family_word);
        match family.surface {
            Surface::Operations(operations) => operations
                .iter()
                .find(|entry| super::operation_word(family, entry) == operation_word)
                .unwrap_or_else(|| {
                    panic!("no 'mahbot {family_word} {operation_word}' command line")
                }),
            Surface::Standalone(_) => panic!("'mahbot {family_word}' has no operation word"),
        }
    }

    /// The family a word names, or a panic naming the word.
    fn family_of(family_word: &str) -> &'static Family {
        family(family_word).map_or_else(
            || panic!("'mahbot {family_word}' is no family"),
            |cli| cli.0,
        )
    }

    /// Parse an operation's flag tail, as one list of tokens.
    fn parse(family_word: &str, operation_word: &str, tokens: &[&str]) -> Result<Tail, String> {
        let operation = operation(family_word, operation_word);
        parse_flags(
            operation.action,
            operation_word,
            operation.flags,
            &tokens.iter().map(ToString::to_string).collect::<Vec<_>>(),
        )
    }

    /// Parse a standalone family's flag tail: it has no operation word, so the
    /// whole tail is flags.
    fn parse_standalone(family_word: &str, tokens: &[&str]) -> Result<Tail, String> {
        let cli =
            family(family_word).unwrap_or_else(|| panic!("'mahbot {family_word}' is no family"));
        let Surface::Standalone(flags) = cli.0.surface else {
            panic!("'mahbot {family_word}' is not a standalone family");
        };
        parse_flags(
            None,
            cli.0.word,
            flags,
            &tokens.iter().map(ToString::to_string).collect::<Vec<_>>(),
        )
    }

    fn parsed(family: &str, operation_word: &str, tokens: &[&str]) -> Parsed {
        match parse(family, operation_word, tokens) {
            Ok(Tail::Call(parsed)) => parsed,
            Ok(Tail::Help) => panic!("'{family} {operation_word}' must not ask for help"),
            Err(e) => panic!("'{family} {operation_word}' must parse: {e}"),
        }
    }

    /// A standalone parse that must yield a call.
    fn parsed_standalone(family_word: &str, tokens: &[&str]) -> Parsed {
        match parse_standalone(family_word, tokens) {
            Ok(Tail::Call(parsed)) => parsed,
            Ok(Tail::Help) => panic!("'mahbot {family_word}' must not ask for help"),
            Err(e) => panic!("'mahbot {family_word}' must parse: {e}"),
        }
    }

    /// The command line and the tools must agree on what exists, in both
    /// directions: an operation the tool cannot run, an action the command line
    /// cannot reach, a tool no operation runs, and a duplicate on either side all
    /// fail here. A tool that takes no `action` has none in the registry either —
    /// its operation IS the tool. A standalone family has no operation word at
    /// all: the family word IS its operation.
    #[test]
    fn operations_match_the_tools_own_actions() {
        for family in FAMILIES {
            assert!(
                !family.tools.is_empty(),
                "the '{}' family fronts no tool at all",
                family.word
            );
            let Surface::Operations(operations) = family.surface else {
                assert_eq!(
                    family.tools.len(),
                    1,
                    "the standalone '{}' family is its own operation, so it fronts one tool",
                    family.word
                );
                assert!(
                    tool_schema(family.tools[0])["properties"]
                        .get("action")
                        .is_none(),
                    "'{}' publishes no operation, but its tool takes an `action`",
                    family.tools[0]
                );
                continue;
            };
            for operation in operations {
                assert!(
                    family.tools.contains(&operation.tool),
                    "'mahbot {} {}' runs '{}', which the family does not front",
                    family.word,
                    operation_word(family, operation),
                    operation.tool
                );
            }
            // The words a family answers to are DERIVED (an action kebab-cased, a
            // tool's own name with the family word dropped), so two operations of
            // one family could derive the same word — and the first match would
            // shadow the second in the lookup `run` does, silently. A family that
            // spans two tools is where that is reachable at all.
            let words: Vec<String> = operations
                .iter()
                .map(|operation| operation_word(family, operation))
                .collect();
            assert!(
                words.iter().all(|word| !word.is_empty()),
                "'mahbot {}' derives an empty operation word",
                family.word
            );
            assert!(
                duplicates(words.iter().map(String::as_str)).is_empty(),
                "'mahbot {}' derives the same operation word twice, so one would shadow the other",
                family.word
            );
            for tool in family.tools {
                let ours: Vec<Option<&str>> = operations
                    .iter()
                    .filter(|operation| operation.tool == *tool)
                    .map(|operation| operation.action)
                    .collect();
                assert!(
                    !ours.is_empty(),
                    "'{tool}' is fronted by 'mahbot {}' but no operation runs it",
                    family.word
                );
                let schema = tool_schema(tool);
                let taken: Vec<&str> = ours.iter().flatten().copied().collect();
                if taken.is_empty() {
                    assert!(
                        schema["properties"].get("action").is_none(),
                        "'{tool}' takes an `action`, so its operations must name one"
                    );
                    continue;
                }
                assert_eq!(
                    taken.len(),
                    ours.len(),
                    "one of '{tool}'s command lines names an action where another does not"
                );
                let mut theirs: Vec<&str> = tool_actions(&schema);
                assert!(
                    duplicates(taken.iter().copied()).is_empty(),
                    "the '{}' command line repeats an action of '{tool}'",
                    family.word
                );
                assert!(
                    duplicates(theirs.iter().copied()).is_empty(),
                    "'{tool}' repeats an action"
                );
                let mut ours = taken;
                ours.sort_unstable();
                theirs.sort_unstable();
                assert_eq!(
                    ours, theirs,
                    "'mahbot {}' must expose exactly the actions of its tool '{tool}'",
                    family.word
                );
            }
        }
    }

    /// A tool that takes no action is named by its operations from its OWN name
    /// — the family word dropped and the rest kebab-cased — so a sub-command
    /// cannot be spelled a way its tool is not.
    #[test]
    fn action_less_operations_are_named_from_their_tools() {
        for family in FAMILIES {
            let Surface::Operations(operations) = family.surface else {
                continue;
            };
            for operation in operations.iter().filter(|o| o.action.is_none()) {
                let word = operation_word(family, operation);
                let held = word.replace('-', "_");
                assert!(
                    !held.is_empty(),
                    "'mahbot {} {word}' derives no word",
                    family.word
                );
                assert_eq!(
                    operation.tool,
                    format!("{}_{held}", family.word),
                    "'mahbot {} {word}' must be its tool '{}' with the family word dropped",
                    family.word,
                    operation.tool
                );
            }
        }
    }

    /// One command line of a family, as the flag tie is swept: what it is called,
    /// the tool it runs, the arguments it can fill, and the flags it carries.
    struct CommandLine {
        called: String,
        tool: &'static str,
        arguments: serde_json::Map<String, serde_json::Value>,
        flags: &'static [Flag],
    }

    /// The JSON types one flag kind can fill.
    fn fillable_types(kind: &FlagKind) -> &'static [&'static str] {
        match kind {
            // One string, so it can only fill a string argument — and a path is
            // a string the command line resolves.
            FlagKind::Value | FlagKind::Path => &["string"],
            // The number it is.
            FlagKind::Number => &["integer", "number"],
            // Entries, as an array — each one a path for a path list.
            FlagKind::List | FlagKind::PathList => &["array"],
            // The JSON it is — and, as the text it falls back to, a string.
            FlagKind::Structured(_) => &["array", "object", "string"],
            FlagKind::Present => &["boolean"],
            // The destination is the command line's own; it fills no argument.
            FlagKind::Destination => &[],
        }
    }

    /// A flag kind's own name, for the message a failed tie prints.
    fn flag_kind_name(kind: &FlagKind) -> &'static str {
        match kind {
            FlagKind::Value => "Value",
            FlagKind::Path => "Path",
            FlagKind::Number => "Number",
            FlagKind::List => "List",
            FlagKind::PathList => "PathList",
            FlagKind::Structured(_) => "Structured",
            FlagKind::Present => "Present",
            FlagKind::Destination => "Destination",
        }
    }

    /// The JSON types an argument's schema entry states.
    fn argument_types(property: &serde_json::Value) -> Vec<&str> {
        match &property["type"] {
            serde_json::Value::String(kind) => vec![kind.as_str()],
            serde_json::Value::Array(kinds) => kinds
                .iter()
                .map(|kind| kind.as_str().expect("a type word is a string"))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Every property name a schema subtree declares, at any depth — the fields a
    /// structured value can carry, so a declared path-bearing field can be held
    /// against the tool's own schema rather than against a second list.
    fn property_names(schema: &serde_json::Value) -> Vec<String> {
        let mut names = Vec::new();
        match schema {
            serde_json::Value::Object(members) => {
                if let Some(serde_json::Value::Object(properties)) = members.get("properties") {
                    names.extend(properties.keys().cloned());
                }
                for member in members.values() {
                    names.extend(property_names(member));
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    names.extend(property_names(item));
                }
            }
            _ => {}
        }
        names
    }
    /// The property names a schema subtree declares at any depth whose own schema
    /// IS one of the tool's picture arguments: a field built that way names a FILE,
    /// so the resolution the command line does is told which fields it owes — the
    /// other direction of the field tie. The picture's schema differs by target
    /// (a `create` block may target a PDF), so the whole family is read.
    fn file_naming_fields(
        schema: &serde_json::Value,
        pictures: &[serde_json::Value],
    ) -> Vec<String> {
        let mut names = Vec::new();
        match schema {
            serde_json::Value::Object(members) => {
                if let Some(serde_json::Value::Object(properties)) = members.get("properties") {
                    for (name, property) in properties {
                        if pictures.contains(property) {
                            names.push(name.clone());
                        }
                    }
                }
                for member in members.values() {
                    names.extend(file_naming_fields(member, pictures));
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    names.extend(file_naming_fields(item, pictures));
                }
            }
            _ => {}
        }
        names
    }

    /// The command lines of one family, as the flag and argument ties sweep them:
    /// what each is called, the tool it runs, the arguments it can fill, and the
    /// flags it carries.
    fn command_lines(family: &'static Family) -> Vec<CommandLine> {
        match family.surface {
            Surface::Operations(operations) => operations
                .iter()
                .map(|operation| CommandLine {
                    called: format!("{} {}", family.word, operation_word(family, operation)),
                    tool: operation.tool,
                    arguments: argument_schema(operation),
                    flags: operation.flags,
                })
                .collect(),
            Surface::Standalone(flags) => vec![CommandLine {
                called: family.word.to_string(),
                tool: family.tools[0],
                arguments: tool_schema(family.tools[0])["properties"]
                    .as_object()
                    .expect("the tool's properties")
                    .clone(),
                flags,
            }],
        }
    }

    /// Every flag must be an argument the tool actually reads, and it must be
    /// able to fill that argument's own type — swept per OPERATION, so a flag one
    /// operation of a family carries is judged against that operation's arguments
    /// rather than against a sibling's. The ONE exception is the destination
    /// ([`FlagKind::Destination`]): the command line's own operand, which is why
    /// it is the one flag that is not looked up among the tool's arguments — and
    /// it is carried by EVERY operation of a producing family and by nothing
    /// else.
    #[test]
    fn flags_stay_tied_to_the_tools_own_arguments() {
        let mut file_fields = 0usize;
        for family in FAMILIES {
            let lines = command_lines(family);
            for line in &lines {
                assert!(
                    duplicates(line.flags.iter().map(|flag| flag.arg)).is_empty(),
                    "'mahbot {}' repeats a flag",
                    line.called
                );
                for flag in line.flags {
                    if matches!(flag.kind, FlagKind::Destination) {
                        continue;
                    }
                    let property = line.arguments.get(flag.arg).unwrap_or_else(|| {
                        panic!(
                            "'{}' is not an argument of 'mahbot {}'",
                            flag.arg, line.called
                        )
                    });
                    let kinds = argument_types(property);
                    assert!(
                        fillable_types(&flag.kind)
                            .iter()
                            .any(|fill| kinds.contains(fill)),
                        "the {} flag '{}' cannot fill a {kinds:?} argument of 'mahbot {}'",
                        flag_kind_name(&flag.kind),
                        flag.arg,
                        line.called,
                    );
                    // A structured value's path-bearing fields are swept the same
                    // way, in BOTH directions: the command line resolves the
                    // fields it names, so a field the tool does not have there
                    // would be a resolution this frontend invented — and a field
                    // the tool builds as one that NAMES A FILE, which the schema
                    // itself says (the tool's own `image` argument), must be
                    // declared here, so a field added to the tool later cannot
                    // silently stop being resolved.
                    if let FlagKind::Structured(fields) = &flag.kind {
                        let declared = property_names(property);
                        for field in *fields {
                            assert!(
                                declared.iter().any(|name| name == field),
                                "'{field}' is no field of the '{}' argument of 'mahbot {}'",
                                flag.arg,
                                line.called,
                            );
                        }
                        let naming_a_file =
                            file_naming_fields(property, &crate::tools::document::image_args());
                        file_fields += naming_a_file.len();
                        for field in naming_a_file {
                            assert!(
                                fields.contains(&field.as_str()),
                                "'{field}' of the '{}' argument of 'mahbot {}' names a file, so \
                                 the command line must resolve it — declare it on the flag",
                                flag.arg,
                                line.called,
                            );
                        }
                    }
                }
            }
            // A producing family is one whose flag list carries the destination
            // at all; what this holds is that EVERY operation of it carries
            // exactly one, so no operation of a producing family is left without
            // a destination to run under.
            if produces_files(family) {
                for line in &lines {
                    let carried = line
                        .flags
                        .iter()
                        .filter(|flag| matches!(flag.kind, FlagKind::Destination))
                        .count();
                    assert_eq!(
                        carried, 1,
                        "'mahbot {}' carries {carried} destination flags where a producing \
                         family's every operation carries one",
                        line.called
                    );
                }
            }
        }
        // The reverse direction has to have found every one of them, or it would
        // hold by being vacuous: the document tool's picture stands in a `create`
        // block, in a docx placing edit and in a pptx placing edit — three fields
        // the command line owes a resolution.
        assert!(
            file_fields >= 3,
            "the picture fields naming a file were not all found ({file_fields}) — the tie would \
             be vacuous"
        );
    }

    /// Every argument a tool reads — bar the action — must have a command-line
    /// form, so a new argument cannot silently arrive with no way to set it.
    ///
    /// Swept per OPERATION where a tool's actions declare their OWN arguments
    /// ([`declares_per_action_arguments`]): there a flag one operation omits is a
    /// real hole that a union over the tool's lines would hide. Where the actions
    /// share one argument set beside the action, the union is the right rule —
    /// the tool's own schema states those arguments once, beside the action, and
    /// every action takes them all.
    #[test]
    fn every_tool_argument_has_a_command_line_form() {
        for family in FAMILIES {
            let lines = command_lines(family);
            for tool in family.tools {
                let strict = declares_per_action_arguments(tool);
                let mut reachable: Vec<&str> = Vec::new();
                let mut declared: Vec<&str> = Vec::new();
                for line in lines.iter().filter(|line| line.tool == *tool) {
                    let line_reachable: Vec<&str> = line
                        .flags
                        .iter()
                        .filter(|flag| !matches!(flag.kind, FlagKind::Destination))
                        .map(|flag| flag.arg)
                        .collect();
                    let line_declared: Vec<&str> = line
                        .arguments
                        .keys()
                        .map(String::as_str)
                        .filter(|key| *key != "action")
                        .collect();
                    if strict {
                        for argument in &line_declared {
                            assert!(
                                line_reachable.contains(argument),
                                "'{argument}' of '{tool}' has no command-line form on 'mahbot {}'",
                                line.called
                            );
                        }
                    }
                    reachable.extend(line_reachable);
                    declared.extend(line_declared);
                }
                if !strict {
                    for argument in &declared {
                        assert!(
                            reachable.contains(argument),
                            "'{argument}' of '{tool}' has no command-line form on 'mahbot {}'",
                            family.word
                        );
                    }
                }
            }
        }
    }

    /// Every family declares a protocol value the listener serves. Both sides are
    /// plain strings in different modules, so nothing else can see a family sent
    /// under a value the IPC layer does not recognise — every call of it would
    /// come back as an unknown store (see [`crate::db::ipc`]).
    #[test]
    fn every_family_carries_a_protocol_value_the_listener_serves() {
        for cli in FAMILIES {
            assert!(
                crate::db::ipc::CALL_STORES.contains(&cli.protocol),
                "'mahbot {}' carries '{}', which the listener does not serve",
                cli.word,
                cli.protocol
            );
        }
        // Only a family THIS build adds carries the current value: every family
        // the channel has already served keeps the value those builds know, so a
        // product one build behind still serves it rather than being told to
        // restart. The list is the checkpoint — a new family means deciding,
        // deliberately, which of the two it is.
        let current: Vec<&str> = FAMILIES
            .iter()
            .filter(|cli| cli.protocol == crate::db::ipc::CALL_STORE)
            .map(|cli| cli.word)
            .collect();
        assert_eq!(
            current,
            vec!["send", "document", "image", "video", "vector"],
            "the families this build adds"
        );
    }

    /// The instance serves a tool by NAME only when a family publishes it, so
    /// the owner's other tools are not reachable through the channel however
    /// privileged the socket is. The three named below are the checkpoint: a
    /// family that ever publishes one of them changes this list deliberately.
    #[test]
    fn tools_no_family_publishes_are_not_served_by_name() {
        for tool in ["shell", "edit", "analyze"] {
            assert!(
                !fronts_tool(tool),
                "'{tool}' is published by no family, so the channel must not serve it by name"
            );
        }
    }

    /// Both helps carry the whole contract: that the product must be running,
    /// that these commands act with the owner's authority, and the exit classes a
    /// script reads — a caller entering through one operation's own help must
    /// learn no less than one reading the family's. A standalone family has no
    /// operation help to enter through, so its family help must carry its usage
    /// line and every flag row itself.
    #[test]
    fn every_help_states_the_running_product_and_the_exit_classes() {
        for family in FAMILIES {
            let helps = match family.surface {
                Surface::Operations(operations) => {
                    vec![family_help(family), operation_help(family, &operations[0])]
                }
                Surface::Standalone(flags) => {
                    let help = family_help(family);
                    let usage = format!("Usage: mahbot {} [flags]", family.word);
                    assert!(help.contains(&usage), "'{usage}' is missing from:\n{help}");
                    for flag in flags {
                        let spelling = format!("--{}", flag_spelling(flag.arg));
                        assert!(
                            help.contains(&spelling),
                            "'{spelling}' has no row in:\n{help}"
                        );
                    }
                    vec![help]
                }
            };
            for help in &helps {
                assert!(
                    !help.contains("{{"),
                    "a help placeholder was not expanded:\n{help}"
                );
                for required in [
                    "must be running",
                    "owner's authority",
                    "Exit codes:",
                    "0  the operation was carried out",
                    "3  this command was misused",
                    "Output:",
                ] {
                    assert!(
                        help.contains(required),
                        "'{required}' is missing from:\n{help}"
                    );
                }
            }
            // A family that produces files states the destination in BOTH of its
            // helps — its row is rendered from the registry, so the spelling and
            // the description cannot drift from the flag the operations carry.
            if produces_files(family) {
                for help in &helps {
                    assert!(
                        help.contains("--output-dir <dir>"),
                        "'mahbot {}' produces files without stating the destination in a help",
                        family.word
                    );
                }
            }
        }
    }

    /// Both spellings of a flag reach the tool, and what was given is what is
    /// sent — the args object is `{"action": …}` plus the flags of the call.
    #[test]
    fn parsing_reads_both_flag_forms() {
        let p = parsed("workspace", "add", &["--name", "proj", "--path=/tmp/proj"]);
        assert_eq!(
            p.args,
            serde_json::json!({"action": "add", "name": "proj", "path": "/tmp/proj"})
        );
        assert!(p.secrets.is_empty());
    }

    /// A `List` flag accumulates one entry per occurrence, in order, and each
    /// entry is taken verbatim — its commas are its own.
    #[test]
    fn parsing_a_list_flag_collects_verbatim_entries_in_order() {
        let p = parsed(
            "config",
            "create-custom-role",
            &[
                "--name",
                "triager",
                "--prompt",
                "the whole definition",
                "--tools",
                "analyze:quick,deep",
                "--tools",
                "send",
                "--behavior",
                "one_off",
            ],
        );
        assert_eq!(
            p.args["tools"],
            serde_json::json!(["analyze:quick,deep", "send"])
        );
        assert_eq!(p.args["behavior"], "one_off");
        assert_eq!(p.args["prompt"], "the whole definition");
    }

    /// An empty value on a list flag is the EMPTY list — which
    /// `update_custom_role` reads as "clear the role's tools" while an absent
    /// flag leaves them as they are. Occurrences after it add to it.
    #[test]
    fn parsing_an_empty_list_value_is_the_empty_list() {
        let cleared = parsed(
            "config",
            "update-custom-role",
            &["--name", "triager", "--tools="],
        );
        assert_eq!(cleared.args["tools"], serde_json::json!([]));

        let after = parsed(
            "config",
            "update-custom-role",
            &["--name", "triager", "--tools=", "--tools", "send"],
        );
        assert_eq!(after.args["tools"], serde_json::json!(["send"]));

        // The absence that leaves the argument alone is still an absence.
        let untouched = parsed("config", "update-custom-role", &["--name", "triager"]);
        assert!(untouched.args.get("tools").is_none(), "{}", untouched.args);
    }

    /// A switch becomes `true`; an absent flag leaves NO key — which is the
    /// whole difference `pause`/`resume` turn on, where the absent `workspace`
    /// key is the fleet-wide form the tool documents. The other value a tool
    /// argument of this kind takes is written `=<false>`, because an absent
    /// argument is the TOOL's default and not the same thing: a video's audio
    /// track is the provider's own unless the caller asks for silence.
    #[test]
    fn parsing_a_switch_is_true_and_an_absent_flag_leaves_no_key() {
        let full = parsed("workspace", "ticket", &["--ticket-id", "proj-7", "--full"]);
        assert_eq!(
            full.args,
            serde_json::json!({"action": "ticket", "ticket_id": "proj-7", "full": true})
        );

        let explicit = parsed(
            "workspace",
            "ticket",
            &["--ticket-id", "proj-7", "--full=true"],
        );
        assert_eq!(explicit.args["full"], serde_json::json!(true));

        let silenced = parsed(
            "video",
            "gen",
            &["--prompt", "a cat", "--generate-audio=false"],
        );
        assert_eq!(
            silenced.args["generate_audio"],
            serde_json::json!(false),
            "a script must be able to ask for the silent clip the tool accepts"
        );
        let loud = parsed("video", "gen", &["--prompt", "a cat", "--generate-audio"]);
        assert_eq!(loud.args["generate_audio"], serde_json::json!(true));
        let silent = parsed("video", "gen", &["--prompt", "a cat"]);
        assert!(
            silent.args.get("generate_audio").is_none(),
            "an absent switch leaves the provider's own default: {}",
            silent.args
        );

        // A switch given a value that is no boolean is misuse, and the reason
        // spells the form rather than echoing the value.
        let reason = parse(
            "video",
            "gen",
            &["--prompt", "a cat", "--generate-audio=maybe"],
        )
        .err()
        .expect("a switch given a non-boolean is misuse");
        assert!(reason.contains("--generate-audio"), "{reason}");
        assert!(
            reason.contains("=false"),
            "the form is spelled out: {reason}"
        );
        assert!(
            !reason.contains("maybe"),
            "a reason must never carry a value: {reason}"
        );

        let clipped = parsed("workspace", "ticket", &["--ticket-id", "proj-7"]);
        assert!(
            clipped.args.get("full").is_none(),
            "an absent switch must leave no key: {}",
            clipped.args
        );

        let fleet = parsed("workspace", "pause", &[]);
        assert_eq!(fleet.args, serde_json::json!({"action": "pause"}));
        assert!(
            fleet.args.get("workspace").is_none(),
            "a fleet-wide pause must send no `workspace` key at all: {}",
            fleet.args
        );
    }

    /// The values of the secret flags — and only those — travel as secrets.
    #[test]
    fn parsing_hands_secret_flag_values_over_as_secrets() {
        let p = parsed("config", "setup-telegram-bot", &["--token", "NNN:AAA"]);
        assert_eq!(p.args["token"], "NNN:AAA");
        assert_eq!(p.secrets, vec!["NNN:AAA".to_string()]);

        let p = parsed(
            "config",
            "setup-web-search",
            &["--provider", "exa", "--key=SEKRIT"],
        );
        assert_eq!(p.secrets, vec!["SEKRIT".to_string()]);
    }

    /// Help is asked for by a token where a FLAG could stand, and only there:
    /// where a VALUE stands the very next token IS the value, a leading dash and
    /// all. A script must never read success for an operation that did not run,
    /// so `--token -h` performs the operation and the product judges the value.
    #[test]
    fn parsing_reads_a_dash_led_value_as_a_value_and_help_only_where_a_flag_stands() {
        let p = parsed("config", "setup-telegram-bot", &["--token", "-h"]);
        assert_eq!(p.args["token"], "-h");
        assert_eq!(
            p.secrets,
            vec!["-h".to_string()],
            "a dash-led secret is a secret like any other"
        );

        let p = parsed("workspace", "board", &["--phase", "--help"]);
        assert_eq!(p.args["phase"], "--help");

        assert!(
            matches!(parse("workspace", "list", &["-h"]), Ok(Tail::Help)),
            "a bare -h where a flag could stand asks for the help"
        );
        assert!(
            matches!(
                parse("workspace", "board", &["--workspace", "w", "--help"]),
                Ok(Tail::Help)
            ),
            "so does one after a flag that already has its value"
        );
    }

    /// Anything this frontend can judge by itself is misuse, and its reason says
    /// what is wrong with WHICH flag — while never carrying a flag's value back
    /// out.
    #[test]
    fn parsing_refuses_misuse_without_echoing_a_value() {
        /// One misuse case: the tokens, the flag its reason must name, and the
        /// values that must not come back in that reason.
        struct Case {
            family: &'static str,
            operation: &'static str,
            tokens: &'static [&'static str],
            names: &'static str,
            values: &'static [&'static str],
        }
        let cases = [
            Case {
                family: "workspace",
                operation: "list",
                tokens: &["--nope", "VAL1"],
                names: "--nope",
                values: &["VAL1"],
            },
            Case {
                family: "workspace",
                operation: "list",
                tokens: &["--workspace"],
                names: "--workspace",
                values: &[],
            },
            Case {
                family: "workspace",
                operation: "list",
                tokens: &["VAL3"],
                names: "unexpected argument",
                values: &["VAL3"],
            },
            Case {
                family: "workspace",
                operation: "ticket",
                tokens: &["--ticket-id", "proj-7", "--full=VAL4"],
                names: "--full",
                values: &["VAL4"],
            },
            Case {
                family: "config",
                operation: "grant-tool",
                tokens: &["--user", "VAL5", "--user", "VAL6"],
                names: "--user",
                values: &["VAL5", "VAL6"],
            },
        ];
        for Case {
            family,
            operation: operation_word,
            tokens,
            names,
            values,
        } in cases
        {
            let reason = parse(family, operation_word, tokens)
                .err()
                .unwrap_or_else(|| panic!("'{family} {operation_word}' {tokens:?} must be misuse"));
            assert!(
                reason.contains(names),
                "'{family} {operation_word}' must name '{names}', got: {reason}"
            );
            for value in values {
                assert!(
                    !reason.contains(value),
                    "a usage error must never carry '{value}': {reason}"
                );
            }
        }

        // A secret that was given stays out of the next mistake's reason.
        let reason = parse(
            "config",
            "setup-telegram-bot",
            &["--token", "SEKRIT", "--nope"],
        )
        .err()
        .expect("a stray flag is misuse");
        assert!(
            !reason.contains("SEKRIT"),
            "a usage error must never carry a value: {reason}"
        );
        // Neither does a secret mistyped as a positional — the token the caller
        // wrote there is not echoed at all.
        let reason = parse("config", "setup-telegram-bot", &["SEKRIT"])
            .err()
            .expect("a stray token is misuse");
        assert!(
            !reason.contains("SEKRIT"),
            "a usage error must never carry a token: {reason}"
        );
    }

    /// A number flag carries the number it is, and a value that is no number
    /// travels as the text it is — whether the tool takes such a value is the
    /// tool's own judgement, never this frontend's.
    #[test]
    fn parsing_a_number_flag_carries_the_number_and_sends_anything_else_on() {
        let p = parsed(
            "document",
            "pdf-rotate",
            &["--path", "Cargo.toml", "--degrees", "90"],
        );
        assert_eq!(p.args["degrees"], serde_json::json!(90));
        assert!(
            p.args["degrees"].is_number(),
            "a number must travel as one: {}",
            p.args
        );

        let p = parsed("video", "gen", &["--prompt", "a cat", "--seed", "7"]);
        assert_eq!(p.args["seed"], serde_json::json!(7));

        // A value the tool will not accept is not this command's to judge.
        let p = parsed(
            "document",
            "pdf-rotate",
            &["--path", "Cargo.toml", "--degrees", "sideways"],
        );
        assert_eq!(p.args["degrees"], serde_json::json!("sideways"));
    }

    /// A structured flag carries the JSON it is written as, and a value that is
    /// no JSON travels as the text it is — for the tool to refuse.
    #[test]
    fn parsing_a_structured_flag_carries_the_json() {
        let p = parsed(
            "document",
            "create",
            &[
                "--format",
                "pdf",
                "--content",
                r#"[{"type": "paragraph", "text": "hi"}]"#,
            ],
        );
        assert_eq!(p.args["content"][0]["type"], "paragraph");

        let p = parsed(
            "document",
            "fill-template",
            &["--path", "Cargo.toml", "--values", r#"{"name": "Ada"}"#],
        );
        assert_eq!(p.args["values"]["name"], "Ada");

        let p = parsed(
            "document",
            "create",
            &["--format", "pdf", "--content", "not json"],
        );
        assert_eq!(p.args["content"], serde_json::json!("not json"));
    }

    /// A value too long for one command line may be given by naming a file that
    /// holds it: a `List` flag's `@<file>` adds the entries the file holds, and a
    /// structured flag's `@<file>` takes the JSON the file holds. The file form
    /// is this command line's own, so a file it cannot use is misuse — and the
    /// reason names the FLAG, never the path.
    #[test]
    fn parsing_reads_a_value_from_a_file_the_caller_names() {
        let dir = crate::util::test::test_root().join("cli-file-values");
        std::fs::create_dir_all(&dir).expect("a place for the value files");

        let list = dir.join("entries.json");
        std::fs::write(&list, r#"["Cargo.toml", "Cargo.lock"]"#).expect("write the entries");
        let from_file = format!("@{}", list.display());
        let p = parsed(
            "document",
            "pdf-merge",
            &["--paths", &from_file, "--paths", "README.md"],
        );
        let here = std::env::current_dir().expect("a working directory");
        let resolved = |name: &str| here.join(name).to_string_lossy().into_owned();
        assert_eq!(
            p.args["paths"],
            serde_json::json!([
                // The entries a file holds are paths like any other, so each is
                // resolved where this command runs.
                resolved("Cargo.toml"),
                resolved("Cargo.lock"),
                resolved("README.md"),
            ])
        );

        let value = dir.join("values.json");
        std::fs::write(&value, r#"{"name": "Ada"}"#).expect("write the value");
        let from_file = format!("@{}", value.display());
        let p = parsed(
            "document",
            "fill-template",
            &["--path", "Cargo.toml", "--values", &from_file],
        );
        assert_eq!(p.args["values"]["name"], "Ada");

        // A file that does not hold what the flag takes is misuse, and the
        // reason never echoes the path.
        let wrong = dir.join("wrong.json");
        std::fs::write(&wrong, r#"{"a": 1}"#).expect("write the wrong shape");
        let from_file = format!("@{}", wrong.display());
        let reason = parse("document", "pdf-merge", &["--paths", &from_file])
            .err()
            .expect("a file that does not hold the entries is misuse");
        assert!(reason.contains("--paths"), "{reason}");
        assert!(
            !reason.contains(wrong.to_string_lossy().as_ref()),
            "a reason must never carry a path value: {reason}"
        );

        let missing = format!("@{}", dir.join("nothing-here.json").display());
        let reason = parse("document", "pdf-merge", &["--paths", &missing])
            .err()
            .expect("a file that cannot be read is misuse");
        assert!(reason.contains("--paths"), "{reason}");
    }

    /// A plain text value has the same file form, so a text too long for one
    /// command line travels whole: `--prompt @<file>` is the file's text (without
    /// one trailing line break), and the `./@…` escape gives a value that begins
    /// with `@` — the one spelling a file name cannot be confused with.
    #[test]
    fn parsing_reads_a_text_value_from_a_file_the_caller_names() {
        let dir = crate::util::test::test_root().join("cli-text-values");
        std::fs::create_dir_all(&dir).expect("a place for the value files");

        let brief = dir.join("brief.txt");
        std::fs::write(&brief, "a cat asleep on a keyboard\n").expect("write the brief");
        let from_file = format!("@{}", brief.display());
        let p = parsed("image", "gen", &["--prompt", &from_file]);
        assert_eq!(p.args["prompt"], "a cat asleep on a keyboard");

        // A value that begins with `@` is written `./@…`: the `./` is dropped and
        // no file is read, whether or not a file of that name exists.
        std::fs::write(dir.join("alice"), "not the prompt").expect("write a decoy file");
        let decoy = format!("./@{}", dir.join("alice").display());
        let p = parsed("image", "gen", &["--prompt", &decoy]);
        assert_eq!(
            p.args["prompt"],
            format!("@{}", dir.join("alice").display())
        );

        // A file that is not there is misuse, and the reason names the flag and
        // never the path.
        let missing = format!("@{}", dir.join("nothing-here.txt").display());
        let reason = parse("image", "gen", &["--prompt", &missing])
            .err()
            .expect("a file that cannot be read is misuse");
        assert!(reason.contains("--prompt"), "{reason}");
        assert!(
            !reason.contains("nothing-here"),
            "a reason must never carry a path value: {reason}"
        );

        // A secret given from a file is the FILE's content: that is what must not
        // reach a record, an echo or the answer — never the spelling that named it.
        let key = dir.join("provider-key.txt");
        std::fs::write(&key, "sk-secret-from-a-file\n").expect("write the key");
        let from_file = format!("@{}", key.display());
        let p = parsed(
            "config",
            "setup-web-search",
            &["--provider", "exa", "--key", &from_file],
        );
        assert_eq!(p.args["key"], "sk-secret-from-a-file");
        assert_eq!(p.secrets, vec!["sk-secret-from-a-file".to_string()]);

        // A flag whose value names a path itself takes no file form: `@` is just
        // the first character of the name it was given.
        let named = dir.join("@report.pdf");
        std::fs::write(&named, "not a pdf").expect("write a file whose name begins with @");
        let p = parsed(
            "document",
            "pdf-rotate",
            &["--path", &named.to_string_lossy(), "--degrees", "90"],
        );
        assert_eq!(p.args["path"], named.to_string_lossy().as_ref());
        let p = parsed(
            "document",
            "create",
            &["--format", "pdf", "--output-dir", "@mine"],
        );
        assert_eq!(p.output_dir.as_deref(), Some("@mine"));
    }

    /// The file a value is read from must be a REGULAR file of a bounded size:
    /// the parsing path has no timeout, so a pipe would hold the command forever
    /// and an endless device would read until memory ran out, while a file far
    /// past any command line is a mistaken path rather than a value.
    #[test]
    fn the_file_a_value_is_read_from_must_be_a_regular_file() {
        let dir = crate::util::test::test_root().join("cli-value-file-guards");
        std::fs::create_dir_all(&dir).expect("a place for the guard fixtures");

        // A directory holds no value to read.
        let reason = parse(
            "image",
            "gen",
            &["--prompt", &format!("@{}", dir.display())],
        )
        .err()
        .expect("a directory is misuse");
        assert!(reason.contains("--prompt"), "{reason}");
        assert!(reason.contains("regular file"), "{reason}");

        // A file past the bound is refused: a value travels as one argument.
        let huge = dir.join("huge.txt");
        std::fs::File::create(&huge)
            .and_then(|file| file.set_len(MAX_VALUE_FILE_BYTES + 1))
            .expect("a sparse file past the bound");
        let reason = parse(
            "image",
            "gen",
            &["--prompt", &format!("@{}", huge.display())],
        )
        .err()
        .expect("a value file past the bound is misuse");
        assert!(reason.contains("over 8 MB"), "{reason}");

        #[cfg(unix)]
        {
            // A pipe would block the read until something writes it.
            let fifo = dir.join("value.fifo");
            let _ = std::fs::remove_file(&fifo);
            let path = std::ffi::CString::new(fifo.to_string_lossy().as_bytes())
                .expect("a path without a NUL");
            // SAFETY: `mkfifo` takes a NUL-terminated path and a mode.
            if unsafe { libc::mkfifo(path.as_ptr(), 0o600) } == 0 {
                let reason = parse(
                    "image",
                    "gen",
                    &["--prompt", &format!("@{}", fifo.display())],
                )
                .err()
                .expect("a pipe is misuse");
                assert!(reason.contains("regular file"), "{reason}");
            }
            // An endless device would read until memory runs out.
            let reason = parse("image", "gen", &["--prompt", "@/dev/zero"])
                .err()
                .expect("a device is misuse");
            assert!(reason.contains("regular file"), "{reason}");
        }
    }

    /// The destination travels BESIDE the tool's arguments and never among them,
    /// and a producing operation sends it exactly as the caller spelled it.
    #[test]
    fn parsing_hands_the_destination_beside_the_tools_arguments() {
        let p = parsed(
            "document",
            "pdf-rotate",
            &[
                "--path",
                "Cargo.toml",
                "--degrees",
                "90",
                "--output-dir",
                "/somewhere/of/my/own",
            ],
        );
        assert_eq!(p.output_dir.as_deref(), Some("/somewhere/of/my/own"));
        assert!(
            p.args.get("output_dir").is_none(),
            "the destination is not an argument of the tool: {}",
            p.args
        );
        // The tool's own arguments are untouched beside it.
        assert_eq!(p.args["action"], "pdf_rotate");
        assert_eq!(p.args["degrees"], serde_json::json!(90));

        // Left out, it stays unset: the frontend resolves the default itself.
        let p = parsed("document", "create", &["--format", "pdf"]);
        assert_eq!(p.output_dir, None);
    }

    /// A flag that names a FILE carries the path the CALLER meant: `~` expanded
    /// against the home directory and a relative path resolved in the directory
    /// this command was run in — the product has a working directory of its own,
    /// and a relative path sent on as written would name a different file there.
    /// A value that names no file, and an absolute path or a public URL, are left
    /// exactly as they were written.
    #[test]
    fn parsing_resolves_a_path_where_the_command_runs() {
        let here = std::env::current_dir().expect("a working directory");
        let relative = |name: &str| here.join(name).to_string_lossy().into_owned();
        let absolute = relative("Cargo.toml");

        let p = parsed(
            "document",
            "pdf-rotate",
            &["--path", "Cargo.toml", "--degrees", "90"],
        );
        assert_eq!(p.args["path"], serde_json::json!(absolute));

        let p = parsed(
            "document",
            "pdf-merge",
            &["--paths", "Cargo.toml", "--paths", &absolute],
        );
        assert_eq!(p.args["paths"][0], absolute);
        assert_eq!(
            p.args["paths"][1], absolute,
            "an absolute path stays as it is"
        );

        // A flag whose value may be a URL leaves a URL alone.
        let p = parsed(
            "video",
            "edit",
            &[
                "--instruction",
                "restyle it",
                "--video-url",
                "https://example.test/clip.mp4",
                "--images",
                "Cargo.toml",
            ],
        );
        assert_eq!(p.args["video_url"], "https://example.test/clip.mp4");
        assert_eq!(p.args["images"][0], relative("Cargo.toml"));

        // `~` is the home directory, exactly as a shell would read it — asserted
        // on the destination, the one path operand that may be a directory: a
        // bare `~` is a directory, and a reference file must be a regular one.
        if let Ok(home) = std::env::var("HOME") {
            let resolved = resolve_output_dir(Some("~")).expect("the home directory is usable");
            assert_eq!(
                resolved.trim_end_matches('/'),
                crate::util::strip_verbatim_prefix(
                    &std::fs::canonicalize(&home).expect("the home directory resolves")
                )
                .to_string_lossy()
                .trim_end_matches('/'),
                "a bare ~ is the home directory"
            );
        }

        // A name is not a path, and a plain value is not one either.
        let p = parsed(
            "document",
            "create",
            &["--format", "docx", "--file-name", "report"],
        );
        assert_eq!(p.args["file_name"], "report");
        let p = parsed("image", "gen", &["--prompt", "a cat by a window"]);
        assert_eq!(p.args["prompt"], "a cat by a window");
    }

    /// A path a STRUCTURED value carries resolves exactly like a `--path` value:
    /// the product resolves an embedded image against its own directories, so a
    /// relative one sent on as written would read a file the caller never named —
    /// and a field that names no file is the caller's own text.
    #[test]
    fn parsing_resolves_a_path_inside_a_structured_value() {
        let here = std::env::current_dir().expect("a working directory");
        let joined = here.join("Cargo.toml").to_string_lossy().into_owned();

        // The image a create's own block embeds, and the one an edit places.
        let p = parsed(
            "document",
            "create",
            &[
                "--format",
                "docx",
                "--content",
                r#"[{"type": "paragraph", "text": "hi"}, {"type": "image", "image": "Cargo.toml"}]"#,
            ],
        );
        assert_eq!(p.args["content"][0].get("image"), None);
        assert_eq!(p.args["content"][1]["image"], serde_json::json!(joined));

        let p = parsed(
            "document",
            "pptx-edit",
            &[
                "--path",
                "Cargo.toml",
                "--edits",
                r#"[{"op": "add_image", "slide": 1, "image": "Cargo.toml"}]"#,
            ],
        );
        assert_eq!(p.args["edits"][0]["image"], serde_json::json!(joined));

        let p = parsed(
            "document",
            "docx-edit",
            &[
                "--path",
                "Cargo.toml",
                "--edits",
                r#"[{"op": "add_image", "after": "x", "image": "Cargo.toml"}]"#,
            ],
        );
        assert_eq!(p.args["edits"][0]["image"], serde_json::json!(joined));

        // A URL is left to the product, which is the one that can fetch it.
        let p = parsed(
            "document",
            "create",
            &[
                "--format",
                "docx",
                "--content",
                r#"[{"type": "image", "image": "https://example.test/picture.png"}]"#,
            ],
        );
        assert_eq!(
            p.args["content"][0]["image"],
            "https://example.test/picture.png"
        );

        // A field no file stands under is a value like any other, however it
        // reads: a sheet cell that looks like a path is the caller's text.
        let p = parsed(
            "document",
            "create",
            &[
                "--format",
                "xlsx",
                "--sheets",
                r#"[{"name": "S", "rows": [["generated/logo.png"]]}]"#,
            ],
        );
        assert_eq!(p.args["sheets"][0]["rows"][0][0], "generated/logo.png");

        // A reference file inside a value that is not there is the caller's own
        // misuse: the reason names the flag and never the path.
        let reason = parse(
            "document",
            "create",
            &[
                "--format",
                "docx",
                "--content",
                r#"[{"type": "image", "image": "no-such-picture-here.png"}]"#,
            ],
        )
        .err()
        .expect("a missing reference file inside a value is misuse");
        assert!(reason.contains("--content"), "{reason}");
        assert!(!reason.contains("no-such-picture-here"), "{reason}");
    }

    /// A reference path that is not there, or cannot be read, is the caller's own
    /// misuse — judged before anything is sent, and the reason names the flag and
    /// never the path.
    #[test]
    fn parsing_refuses_a_reference_path_that_is_not_there() {
        let reason = parse(
            "document",
            "pdf-rotate",
            &["--path", "./no-such-file-here.pdf"],
        )
        .err()
        .expect("a missing reference file is misuse");
        assert!(reason.contains("--path"), "{reason}");
        assert!(
            !reason.contains("no-such-file-here"),
            "a reason must never carry a path value: {reason}"
        );

        let reason = parse(
            "document",
            "pdf-merge",
            &[
                "--paths",
                "Cargo.toml",
                "--paths",
                "./no-such-file-here.pdf",
            ],
        )
        .err()
        .expect("one missing entry of a path list is misuse");
        assert!(reason.contains("--paths"), "{reason}");

        // A URL is not a path at all: only the product can fetch one, and its
        // answer says what happened.
        let p = parsed(
            "video",
            "edit",
            &[
                "--instruction",
                "restyle",
                "--video-url",
                "https://example.test/x.mp4",
            ],
        );
        assert_eq!(p.args["video_url"], "https://example.test/x.mp4");

        // Something that exists but is no regular file is misuse as well: the
        // product would read it with no timeout of its own, so a pipe with no
        // writer would hold the daemon — and the operation's flight guard —
        // forever. Only metadata is looked at, so the refusal cannot block.
        let dir = crate::util::test::test_root().join("cli-reference-not-a-file");
        std::fs::create_dir_all(&dir).expect("a directory that is no reference file");
        let reason = parse(
            "document",
            "pdf-rotate",
            &["--path", &dir.to_string_lossy()],
        )
        .err()
        .expect("a directory is not a reference file");
        assert!(reason.contains("--path"), "{reason}");
        assert!(reason.contains("regular file"), "{reason}");
        assert!(
            !reason.contains("cli-reference-not-a-file"),
            "a reason must never carry a path value: {reason}"
        );

        #[cfg(unix)]
        {
            let fifo = dir.join("clip.mp4");
            let _ = std::fs::remove_file(&fifo);
            let path = std::ffi::CString::new(fifo.to_string_lossy().as_bytes())
                .expect("a path without a NUL");
            // SAFETY: `mkfifo` takes a NUL-terminated path and a mode.
            if unsafe { libc::mkfifo(path.as_ptr(), 0o600) } == 0 {
                let reason = parse("video", "edit", &["--video-url", &fifo.to_string_lossy()])
                    .err()
                    .expect("a pipe is not a reference file");
                assert!(reason.contains("regular file"), "{reason}");
                // The structured fields a value carries are held to it too.
                let content = format!(
                    r#"[{{"type": "image", "image": "{}"}}]"#,
                    fifo.to_string_lossy()
                );
                let reason = parse(
                    "document",
                    "create",
                    &["--format", "docx", "--content", &content],
                )
                .err()
                .expect("a pipe inside a value is no reference file either");
                assert!(reason.contains("--content"), "{reason}");
                assert!(reason.contains("regular file"), "{reason}");
            }
        }
    }

    /// The destination is resolved on the machine this command runs on, in the
    /// directory it was run in: a directory of the caller's own is taken as it
    /// resolves, a missing one or one that is not a directory is misuse, and one
    /// the product keeps for itself is refused however it is spelled.
    #[test]
    fn the_destination_is_resolved_and_the_products_own_places_are_refused() {
        // The default destination is the directory this command was run in. A
        // tree checked out under a governed root is refused like any other place
        // — the rule is about the PLACE, not about who asks — so the default and
        // the same directory named explicitly must agree either way.
        let cwd = std::env::current_dir().expect("a working directory");
        let cwd_string = cwd.to_string_lossy().into_owned();
        let named = resolve_output_dir(Some(&cwd_string)).ok();
        assert_eq!(resolve_output_dir(None).ok(), named);
        if let Some(resolved) = named {
            assert_eq!(
                resolved,
                crate::util::strip_verbatim_prefix(
                    &std::fs::canonicalize(&cwd).expect("the working directory canonicalizes")
                )
                .to_string_lossy()
                .into_owned()
            );
        }

        // A destination that is not there is misuse, never created.
        let absent = crate::util::test::test_root().join("no-such-destination");
        let reason = resolve_output_dir(Some(&absent.to_string_lossy()))
            .expect_err("a missing destination is misuse");
        assert!(reason.contains("cannot be used"), "{reason}");
        assert!(!absent.exists(), "the command must create no directory");

        // A destination that is a file is misuse: it is not a directory.
        let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let reason = resolve_output_dir(Some(&file.to_string_lossy()))
            .expect_err("a file is not a destination");
        assert!(reason.contains("is not a directory"), "{reason}");

        // The product's own temporary area, however it is spelled — a link into
        // it is refused all the same. The check is on the RESOLVED path, so a
        // machine where the area is not there has nothing to resolve and nothing
        // to refuse.
        let temp = std::env::temp_dir();
        if temp.is_dir() {
            let reason = resolve_output_dir(Some(&temp.to_string_lossy()))
                .expect_err("the product's temporary area is refused");
            assert!(reason.contains("temporary area"), "{reason}");
            #[cfg(unix)]
            {
                let link = crate::util::test::test_root().join("cli-destination-link");
                let _ = std::fs::remove_file(&link);
                if std::os::unix::fs::symlink(&temp, &link).is_ok() {
                    let reason = resolve_output_dir(Some(&link.to_string_lossy()))
                        .expect_err("a path resolving into it is refused too");
                    assert!(reason.contains("temporary area"), "{reason}");
                }
            }
        }

        // The product's own data directory, and everything under it. The rule
        // itself is asserted without the filesystem — a machine where the
        // directory does not exist yet is one where nothing can be resolved
        // inside it, not one where it is a usable destination.
        let data = crate::config::default_config_dir().expect("the storage root resolves");
        let inside = data.join("somewhere");
        assert!(
            cleanup_governed(&inside).is_some(),
            "the product's own data directory is refused"
        );
        if data.is_dir() {
            let reason = resolve_output_dir(Some(&data.to_string_lossy()))
                .expect_err("the product's data directory is refused");
            assert!(reason.contains("data directory"), "{reason}");
        }
    }

    /// A family whose tool takes no `action` at all: the family word IS the
    /// operation, so what it sends carries no `action` key and an absent flag
    /// leaves no key.
    #[test]
    fn a_family_with_no_actions_is_its_own_operation() {
        let sent = parsed_standalone(
            "send",
            &[
                "--message",
                "do the thing",
                "--addressee",
                "manager",
                "--place",
                "proj",
            ],
        );
        assert_eq!(
            sent.args,
            serde_json::json!({
                "message": "do the thing",
                "addressee": "manager",
                "place": "proj",
            })
        );
        assert!(sent.secrets.is_empty(), "no flag of 'send' is a secret");
        assert!(
            sent.args.get("branch").is_none(),
            "an absent flag leaves no key: {}",
            sent.args
        );

        // The required flags are the tool's own judgement, not this frontend's:
        // a bare family word sends an args object with no key at all.
        assert_eq!(parsed_standalone("send", &[]).args, serde_json::json!({}));
    }

    /// The exit classes are the browser-automation command's own numbering,
    /// class for class — a script that reads exit codes must not have to learn a
    /// second table. Asserted against that table rather than restated, so the
    /// two cannot drift.
    #[test]
    fn exit_classes_match_the_browser_automation_command() {
        /// The class each of the other command's kinds belongs to, matched
        /// exhaustively so a new kind has to be placed here.
        fn class_of(kind: OutKind) -> Exit {
            match kind {
                OutKind::Ok | OutKind::Empty => Exit::Ok,
                OutKind::Timeout
                | OutKind::Network
                | OutKind::Redesign
                | OutKind::NotFound
                | OutKind::Error => Exit::Failed,
                OutKind::Environment => Exit::Environment,
                OutKind::Usage => Exit::Usage,
            }
        }

        for kind in [
            OutKind::Ok,
            OutKind::Empty,
            OutKind::Timeout,
            OutKind::Network,
            OutKind::Redesign,
            OutKind::NotFound,
            OutKind::Error,
            OutKind::Environment,
            OutKind::Usage,
        ] {
            assert_eq!(
                kind.exit_code(),
                class_of(kind).code(),
                "OutKind::{} must keep its class",
                kind.as_str()
            );
        }
    }

    /// What one attempt becomes: the stream each part goes to, and the class a
    /// script reads — decided with no socket in sight.
    #[test]
    fn attempt_decisions_map_to_the_documented_streams_and_exit_codes() {
        let done = verdict(
            "config",
            CallAttempt::Answered(CallReply::done("token saved")),
        );
        assert_eq!(done.answer.as_deref(), Some("token saved"));
        assert_eq!(done.diagnostic, None);
        assert_eq!(done.code, Exit::Ok);

        let refused = verdict(
            "config",
            CallAttempt::Answered(CallReply::refused("no such user 'bob'")),
        );
        assert_eq!(refused.answer.as_deref(), Some("no such user 'bob'"));
        assert_eq!(refused.diagnostic, None, "a refusal needs no note of ours");
        assert_eq!(refused.code, Exit::Failed);

        let partial = verdict(
            "workspace",
            CallAttempt::Answered(CallReply::partial("froze 2 of 3")),
        );
        assert_eq!(partial.answer.as_deref(), Some("froze 2 of 3"));
        assert!(
            partial.diagnostic.is_some(),
            "a partly carried-out operation needs its own note"
        );
        assert_eq!(partial.code, Exit::Failed);

        // The two the wire client should never hand over, classified as a
        // product that did not take the operation on: the instance's sentence
        // reaches stderr as this command's diagnostic, prefixed like the rest.
        for reply in [
            CallReply::not_accepted("still starting"),
            CallReply::unavailable("cannot serve operations"),
        ] {
            let wording = reply.answer.clone();
            let decision = verdict("workspace", CallAttempt::Answered(reply));
            assert_eq!(decision.answer, None);
            assert_eq!(
                decision.diagnostic.as_deref(),
                Some(format!("mahbot workspace: {wording}").as_str())
            );
            assert_eq!(decision.code, Exit::Environment);
        }

        let not_served = verdict(
            "workspace",
            CallAttempt::NotServed("no endpoint".to_string()),
        );
        assert_eq!(not_served.answer, None);
        assert_eq!(
            not_served.diagnostic.as_deref(),
            Some("mahbot workspace: no endpoint"),
            "a diagnostic of ours names this command"
        );
        assert_eq!(not_served.code, Exit::Environment);

        let too_old = verdict("workspace", CallAttempt::TooOld);
        assert_eq!(too_old.answer, None);
        assert!(
            too_old
                .diagnostic
                .as_deref()
                .is_some_and(|line| line.contains("older") && line.contains("restart")),
            "a too-old product must say what to do: {:?}",
            too_old.diagnostic
        );
        assert_eq!(too_old.code, Exit::Environment);

        let severed = verdict(
            "workspace",
            CallAttempt::Severed(anyhow::anyhow!("the answer could not be read")),
        );
        assert_eq!(severed.answer, None);
        let line = severed.diagnostic.as_deref().expect("a diagnostic");
        assert!(
            line.contains("could not be confirmed") && line.contains("NOT sent again"),
            "an unconfirmed outcome must say plainly what is unknown: {line}"
        );
        assert_eq!(
            line.matches("sent again").count(),
            1,
            "the conclusion is stated once, with the cause beside it: {line}"
        );
        assert_eq!(severed.code, Exit::Failed);
    }

    /// Misuse is exit 3, and it is decided before the environment is touched: an
    /// instance at some other storage root — or none at all — must not change
    /// the answer, and this test has neither.
    #[tokio::test]
    async fn misuse_is_exit_3_before_the_environment_is_probed() {
        let invocations: [&[&str]; 12] = [
            &["config"],
            &["config", "no-such-operation"],
            &["workspace", "list", "--nope", "x"],
            &["workspace", "list", "--workspace"],
            &["workspace", "list", "proj"],
            &["send", "--nope"],
            &["document"],
            &["document", "no-such-operation"],
            &["image", "gen", "--nope"],
            &["video", "gen", "--prompt"],
            // The producing families' destination is judged here too: a place
            // that is not there is misuse, and nothing is sent.
            &[
                "image",
                "gen",
                "--prompt",
                "x",
                "--output-dir",
                "/no/such/place",
            ],
            // The destination is the command line's own: a family that produces
            // nothing does not take it.
            &["workspace", "list", "--output-dir", "/tmp"],
        ];
        for invocation in invocations {
            let args = invocation
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            let cli =
                family(&args[0]).unwrap_or_else(|| panic!("'{}' must be a family word", args[0]));
            let code = run(cli, &args[1..]).await;
            assert_eq!(code, Exit::Usage.code(), "{invocation:?}");
        }
    }
}
