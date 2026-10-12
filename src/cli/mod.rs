//! Command-line frontends onto the RUNNING product.
//!
//! A command line here is not a second implementation of anything: `mahbot
//! config …` and `mahbot workspace …` carry the operation into the instance
//! that is already up, where it is executed by the very tool an agent's
//! assistant calls — the same validation and the same immediate effect on the
//! live product. The frontend never opens or writes a store itself, so there is
//! no second writer and no second copy of any rule.
//!
//! This file holds the instance's own readiness signal — whether operations may
//! be served at all, and the waiter the instance side blocks on — plus the
//! execution entry point in [`instance`], and the frontend's own command line:
//! the family/operation/flag registry that ties every sub-command to one tool
//! action, the argv parsing under it, the help, and the [`Exit`] classes a
//! script reads. The flag grammar itself is [`grammar`], which `mahbot audio`
//! reads too — the browser-automation command and the OpenRouter benchmark read
//! their own flags and word their own refusals.

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

Options:
  -h, --help             Print this help and exit
  -V, --version          Print the version and exit";

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

/// One command-line family: the frontend onto ONE of the product's tools.
pub(crate) struct Family {
    /// The command-line word (`mahbot <word> <operation>`).
    pub(crate) word: &'static str,
    /// The product tool every operation of this family runs.
    pub(crate) tool: &'static str,
    /// One line for `mahbot -h`, and the family's own help opening on it.
    pub(crate) purpose: &'static str,
    /// The operations, one per action of `tool`.
    pub(crate) operations: &'static [Operation],
}

/// One command-line operation: the tool action it runs, under a sub-command
/// word derived from it.
pub(crate) struct Operation {
    /// The tool's own `action` value.
    pub(crate) action: &'static str,
    /// One line for the family's help and the operation's own help.
    pub(crate) purpose: &'static str,
    /// The flags this operation takes, in the order its help lists them.
    pub(crate) flags: &'static [Flag],
}

impl Operation {
    /// The sub-command word: the tool's own action, kebab-cased. Derived rather
    /// than written down, so a sub-command can never be spelled a way its action
    /// is not.
    fn word(&self) -> String {
        self.action.replace('_', "-")
    }
}

/// One flag of one operation: the tool argument it fills.
pub(crate) struct Flag {
    /// The tool's own argument name; the command-line spelling is derived from
    /// it (`ticket_id` → `--ticket-id`), so a flag can never be spelled a way
    /// the tool does not read.
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
pub(crate) enum FlagKind {
    /// One value per occurrence: `--name bob`.
    Value,
    /// Repeatable, one array entry per occurrence (`--tools analyze:quick,deep
    /// --tools send`) — an entry is taken VERBATIM, so an entry that itself
    /// contains a comma is unambiguous. An occurrence with an EMPTY value is the
    /// EMPTY list — `--tools=` — which tools read differently from an absent key
    /// (`update_custom_role` clears the role's tools with it while the absent key
    /// leaves them as they are).
    List,
    /// A switch, present or absent (`--full`), for a tool argument that is a
    /// boolean.
    Present,
}

/// The command lines this file fronts: the admin Assistant's own `mahbot_config`
/// and `workspace` tools — what its toolset holds and nothing else.
pub(crate) const FAMILIES: &[Family] = &[
    Family {
        word: "config",
        tool: "mahbot_config",
        purpose: "Set the running product up — its Telegram bot, its users and \
                  their tool grants, web search, custom roles",
        operations: &[
            Operation {
                action: "setup_telegram_bot",
                purpose: "Save the Telegram bot token from BotFather",
                flags: &[Flag {
                    arg: "token",
                    kind: FlagKind::Value,
                    secret: true,
                    purpose: "The bot token BotFather gave you",
                }],
            },
            Operation {
                action: "bind_telegram",
                purpose: "Bind your own Telegram account, so the bot's messages reach you",
                flags: &[Flag {
                    arg: "handle",
                    kind: FlagKind::Value,
                    secret: false,
                    purpose: "Your Telegram @username (with or without the @), or your numeric id",
                }],
            },
            Operation {
                action: "add_user",
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
                        purpose: "Their Telegram @username (with or without the @), or their \
                                  numeric id",
                    },
                ],
            },
            Operation {
                action: "setup_web_search",
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
                action: "grant_tool",
                purpose: "Give a user one of your custom tools",
                flags: &[USER_FLAG, CUSTOM_TOOL_FLAG],
            },
            Operation {
                action: "revoke_tool",
                purpose: "Take one of your custom tools back from a user",
                flags: &[USER_FLAG, CUSTOM_TOOL_FLAG],
            },
            Operation {
                action: "list_grants",
                purpose: "List the custom tools granted to a user, or to every user",
                flags: &[Flag {
                    arg: "user",
                    kind: FlagKind::Value,
                    secret: false,
                    purpose: "The user's name; omit it to list every user's grants",
                }],
            },
            Operation {
                action: "create_custom_role",
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
                action: "update_custom_role",
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
                action: "list_custom_roles",
                purpose: "List the custom roles",
                flags: &[],
            },
            Operation {
                action: "delete_custom_role",
                purpose: "Delete a custom role",
                flags: &[Flag {
                    arg: "name",
                    kind: FlagKind::Value,
                    secret: false,
                    purpose: "The custom role to delete",
                }],
            },
        ],
    },
    Family {
        word: "workspace",
        tool: "workspace",
        purpose: "Work with the running product's workspaces, their tickets and their board",
        operations: &[
            Operation {
                action: "list",
                purpose: "List the registered workspaces",
                flags: &[Flag {
                    arg: "workspace",
                    kind: FlagKind::Value,
                    secret: false,
                    purpose: "One registered workspace; omit it to list every one",
                }],
            },
            Operation {
                action: "add",
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
                action: "pause",
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
                action: "resume",
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
                action: "board",
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
                action: "ticket",
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
                        purpose: "Send 'full: true' — the ticket whole and unclipped (the tool's \
                                  own default is the clipped form)",
                    },
                ],
            },
            Operation {
                action: "comment",
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
        ],
    },
];

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
    FAMILIES.iter().any(|family| family.tool == tool)
}

/// The family words, in the order the top-level help lists them.
pub(crate) fn family_words() -> impl Iterator<Item = &'static str> {
    FAMILIES.iter().map(|family| family.word)
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
    /// The tool's args object: `{"action": …}` plus exactly the flags that were
    /// given. An absent flag leaves its key ABSENT, never null — the tools tell
    /// absent from null, and `pause`/`resume` read the absence of `workspace`
    /// as the fleet-wide form.
    args: serde_json::Value,
    /// The values of the `secret` flags that were given, in the order they were
    /// given. Nothing else is ever sent.
    secrets: Vec<String>,
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
/// value, a flag given twice — is returned as its reason, for [`Exit::Usage`]. A
/// flag the operation DOES take with a value the tool will not accept is not a
/// mistake this command can judge, so it is sent on: the product refuses it, and
/// that is a failed operation.
fn parse_flags(operation: &Operation, tokens: &[String]) -> Result<Tail, String> {
    let mut args = serde_json::Map::new();
    args.insert(
        "action".to_string(),
        serde_json::Value::String(operation.action.to_string()),
    );
    let mut secrets = Vec::new();
    // The flags already given, so a `Value`/`Present` repeat is refused instead
    // of silently keeping the last one, and the `List` entries, which only
    // become the array at the end.
    let mut given: Vec<&'static str> = Vec::new();
    let mut lists: Vec<(&'static str, Vec<serde_json::Value>)> = Vec::new();
    let spellings: Vec<String> = operation
        .flags
        .iter()
        .map(|flag| flag_spelling(flag.arg))
        .collect();

    let mut rest = tokens.iter();
    while let Some(token) = rest.next() {
        let (name, inline) = match grammar::read_token(token)? {
            grammar::Token::Help => return Ok(Tail::Help),
            grammar::Token::Flag { name, inline } => (name, inline),
        };
        let Some(index) = spellings.iter().position(|spelling| spelling == name) else {
            return Err(grammar::unknown_flag(name, &operation.word()));
        };
        let flag = &operation.flags[index];

        if !matches!(flag.kind, FlagKind::List) {
            if given.contains(&flag.arg) {
                return Err(grammar::repeated_flag(name));
            }
            given.push(flag.arg);
        }

        match flag.kind {
            FlagKind::Present => {
                if inline.is_some() {
                    return Err(format!("flag '--{name}' is a switch and takes no value"));
                }
                args.insert(flag.arg.to_string(), serde_json::Value::Bool(true));
            }
            FlagKind::Value | FlagKind::List => {
                let value = match inline {
                    Some(value) => value.to_string(),
                    None => rest
                        .next()
                        .cloned()
                        .ok_or_else(|| grammar::flag_value_missing(name))?,
                };
                if flag.secret {
                    secrets.push(value.clone());
                }
                if matches!(flag.kind, FlagKind::Value) {
                    args.insert(flag.arg.to_string(), serde_json::Value::String(value));
                } else {
                    // One array entry per occurrence, verbatim: the entry is
                    // never split on its commas, so `name:kind1,kind2` stays one
                    // tool specification and a future `List` of a comma-carrying
                    // value needs no escaping. An EMPTY value empties the list as
                    // of that point — see [`FlagKind::List`].
                    let entries =
                        if let Some(index) = lists.iter().position(|(arg, _)| *arg == flag.arg) {
                            &mut lists[index].1
                        } else {
                            lists.push((flag.arg, Vec::new()));
                            let index = lists.len() - 1;
                            &mut lists[index].1
                        };
                    if value.is_empty() {
                        entries.clear();
                    } else {
                        entries.push(serde_json::Value::String(value));
                    }
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
    }))
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

/// `mahbot <word> -h`: what this family's command line is and what a caller
/// reads back from it.
fn family_help(family: &Family) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "mahbot {} — {}", family.word, family.purpose);
    out.push_str(RUNNING_CONTRACT_HELP);
    let _ = write!(
        out,
        "\nEvery operation below is run by the same `{}` tool the owner's assistant calls, so this \
         command line offers exactly the operations that tool offers.\n",
        family.tool
    );
    let _ = write!(out, "\nUsage: mahbot {} <operation> [flags]\n", family.word);
    out.push_str("\nOperations:\n");
    let width = family
        .operations
        .iter()
        .map(|operation| operation.word().len())
        .max()
        .unwrap_or_default();
    for operation in family.operations {
        let _ = writeln!(out, "  {:<width$}  {}", operation.word(), operation.purpose);
    }
    out.push_str(
        "\nFlags:\n  None are common to every operation: a flag belongs to the operation whose \
         tool argument it names.\n\n  -h, --help  Print this help and exit\n",
    );
    out.push_str(EXIT_CLASS_HELP);
    out.push_str(OUTPUT_HELP);
    out
}

/// `mahbot <word> <operation> -h`: what one operation does and the flags it
/// takes — plus the same contract the family help states, because a script
/// author may enter here first.
fn operation_help(family: &Family, operation: &Operation) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "mahbot {} {} — {}",
        family.word,
        operation.word(),
        operation.purpose
    );
    let _ = write!(
        out,
        "\nUsage: mahbot {} {} [flags]\n",
        family.word,
        operation.word()
    );
    out.push_str("\nFlags:\n");
    if operation.flags.is_empty() {
        out.push_str("  none\n");
    }
    let rows: Vec<(String, String)> = operation
        .flags
        .iter()
        .map(|flag| {
            let mut spelling = format!("--{}", flag_spelling(flag.arg));
            match flag.kind {
                FlagKind::Value => spelling.push_str(" <value>"),
                FlagKind::List => spelling.push_str(" <value> (repeatable; empty clears the list)"),
                FlagKind::Present => {}
            }
            let description = if flag.secret {
                format!("{} — the value is never echoed or recorded", flag.purpose)
            } else {
                flag.purpose.to_string()
            };
            (spelling, description)
        })
        .collect();
    let width = rows
        .iter()
        .map(|(spelling, _)| spelling.len())
        .max()
        .unwrap_or_default();
    for (spelling, description) in rows {
        let _ = writeln!(out, "  {spelling:<width$}  {description}");
    }
    out.push_str(RUNNING_CONTRACT_HELP);
    out.push_str(EXIT_CLASS_HELP);
    out.push_str(OUTPUT_HELP);
    out
}

// ── The run ───────────────────────────────────────────────────────

/// Run one family's command line, returning the process exit code.
///
/// The product's own answer goes to STDOUT exactly as the product worded it
/// (never shortened, never reformatted); everything this command says on its
/// own goes to STDERR. Help goes to stdout. The exit code is the thing a script
/// reads — see [`Exit`].
pub async fn run(cli: Cli, args: &[String]) -> i32 {
    let family = cli.0;

    let Some(operation_word) = args.first().map(String::as_str) else {
        return misuse(family, "no operation was given");
    };
    if grammar::is_help(operation_word) {
        crate::util::print_stdout(&family_help(family));
        return Exit::Ok.code();
    }
    let Some(operation) = family
        .operations
        .iter()
        .find(|operation| operation.word() == operation_word)
    else {
        return misuse(family, &format!("unknown operation '{operation_word}'"));
    };
    let parsed = match parse_flags(operation, &args[1..]) {
        Ok(Tail::Call(parsed)) => parsed,
        // A `-h`/`--help` where a flag could stand is this operation's help — the
        // same rule the browser-automation command follows, and it turns a
        // would-be usage error into help rather than the other way round. A
        // `-h` where a VALUE stands is that value instead (see `parse_flags`):
        // a script must never read success for an operation that did not run.
        Ok(Tail::Help) => {
            crate::util::print_stdout(&operation_help(family, operation));
            return Exit::Ok.code();
        }
        Err(reason) => return misuse(family, &reason),
    };

    // These commands act ON the running product and have no other way to run,
    // so the storage root and the instance lock are settled BEFORE anything is
    // sent. The probe is the same fail-closed one `mahbot debug` routes on: an
    // `Unknown` lock is never read as "no instance is running".
    let root = match crate::config::default_config_dir() {
        Ok(root) => root,
        Err(e) => {
            crate::util::print_stderr(&format!(
                "mahbot {}: cannot resolve the storage root: {e:#}",
                family.word
            ));
            return Exit::Environment.code();
        }
    };
    // A free lock on its own is NOT "nothing is running": a product that is
    // restarting or updating itself releases the lock for the seconds the
    // replacement needs to take it, so the probe is ridden out over the same
    // short window the channel itself uses (seconds, not milliseconds — see
    // `MAHBOT_IPC_BOUND_TIMEOUT_SECS`) and only a window that stays free is
    // reported as nothing running. An `Unknown` probe is ridden out the same
    // way and only then refused, never read as "not running".
    let deadline = std::time::Instant::now() + crate::db::ipc::ipc_bound_timeout();
    let mut attempt = 0usize;
    loop {
        match crate::util::lock::instance_lock_state_settled(&root) {
            InstanceLockState::Held => break,
            InstanceLockState::Free if std::time::Instant::now() >= deadline => {
                crate::util::print_stderr(&format!(
                    "mahbot {}: no instance is running against {} — these commands act on the \
                     running product: start mahbot and try again",
                    family.word,
                    root.display()
                ));
                return Exit::Environment.code();
            }
            InstanceLockState::Unknown(e) if std::time::Instant::now() >= deadline => {
                crate::util::print_stderr(&format!(
                    "mahbot {}: cannot tell whether a mahbot instance holds {} ({e}) — an \
                     operation is only ever sent to a running product, so none was sent",
                    family.word,
                    root.display()
                ));
                return Exit::Environment.code();
            }
            // Between boots, or a probe that could not settle: wait and look again.
            InstanceLockState::Free | InstanceLockState::Unknown(_) => {}
        }
        tokio::time::sleep(crate::db::ipc::retry_backoff(attempt)).await;
        attempt += 1;
    }

    let call = CallRequest {
        tool: family.tool.to_string(),
        args: parsed.args,
        secrets: parsed.secrets,
    };
    let decision = verdict(
        family.word,
        crate::db::ipc::ipc_call_with_wait(&root, &call).await,
    );
    if let Some(answer) = decision.answer {
        crate::util::print_stdout(&answer);
    }
    if let Some(diagnostic) = decision.diagnostic {
        crate::util::print_stderr(&diagnostic);
    }
    decision.code.code()
}

/// Report a misuse on STDERR and hand back its class: the reason, then the one
/// line that says how to see the right form. Nothing was sent, and no operation
/// judged anything.
fn misuse(family: &Family, reason: &str) -> i32 {
    crate::util::print_stderr(&format!("mahbot {}: {reason}", family.word));
    crate::util::print_stderr(&format!("run 'mahbot {} -h' for usage.", family.word));
    Exit::Usage.code()
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
                "mahbot {word}: the running product was built before these commands existed, so \
                 it cannot serve them — restart it from this build and try again"
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
    /// The bare launch is not in the list: it has no word after `mahbot`, so the
    /// shape this loop looks for cannot match it.
    ///
    /// Here — beside the registry — so the library's own test lane holds the tie,
    /// not just the whole-workspace run.
    #[test]
    fn every_entry_point_is_listed_in_the_top_level_usage() {
        for word in family_words().chain(["paused", "audio", "chrome", "debug", "bench-openrouter"])
        {
            let entry = format!("  mahbot {word} ");
            assert!(
                TOP_LEVEL_USAGE.lines().any(|line| line.starts_with(&entry)),
                "'mahbot {word}' has no entry in the top-level usage"
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

    /// The parameters schema of the tool one family fronts.
    fn tool_schema(family: &Family) -> serde_json::Value {
        let tools = admin_toolset();
        crate::tools::find_tool(&tools, family.tool)
            .unwrap_or_else(|| panic!("the admin Assistant must hold '{}'", family.tool))
            .parameters_schema()
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
        family(family_word)
            .and_then(|cli| {
                cli.0
                    .operations
                    .iter()
                    .find(|entry| entry.word() == operation_word)
            })
            .unwrap_or_else(|| panic!("no 'mahbot {family_word} {operation_word}' command line"))
    }

    /// Parse an operation's flag tail, as one list of tokens.
    fn parse(family: &str, operation_word: &str, tokens: &[&str]) -> Result<Tail, String> {
        parse_flags(
            operation(family, operation_word),
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

    /// The command line and its tool must agree on what exists, in both
    /// directions: an operation the tool cannot run, an action the command line
    /// cannot reach, and a duplicate on either side all fail here.
    #[test]
    fn operations_match_the_tools_own_actions() {
        for family in FAMILIES {
            let schema = tool_schema(family);
            let mut theirs: Vec<&str> = schema["properties"]["action"]["enum"]
                .as_array()
                .expect("the tool's action enum")
                .iter()
                .map(|action| action.as_str().expect("action words are strings"))
                .collect();

            assert!(
                duplicates(family.operations.iter().map(|operation| operation.action)).is_empty(),
                "the '{}' command line repeats an action",
                family.word
            );
            assert!(
                duplicates(theirs.iter().copied()).is_empty(),
                "the '{}' tool repeats an action",
                family.tool
            );

            let mut ours: Vec<&str> = family
                .operations
                .iter()
                .map(|operation| operation.action)
                .collect();
            ours.sort_unstable();
            theirs.sort_unstable();
            assert_eq!(
                ours, theirs,
                "'mahbot {}' must expose exactly the actions of its tool '{}'",
                family.word, family.tool
            );
        }
    }

    /// Every flag must be an argument the tool actually reads, and every
    /// argument the tool reads — bar the action — must have a command-line form,
    /// so a new argument cannot silently arrive with no way to set it.
    #[test]
    fn flags_stay_tied_to_the_tools_own_arguments() {
        for family in FAMILIES {
            let schema = tool_schema(family);
            let properties = schema["properties"]
                .as_object()
                .expect("the tool's properties");
            let mut reachable: Vec<&str> = Vec::new();
            for operation in family.operations {
                assert!(
                    duplicates(operation.flags.iter().map(|flag| flag.arg)).is_empty(),
                    "'mahbot {} {}' repeats a flag",
                    family.word,
                    operation.word()
                );
                for flag in operation.flags {
                    let property = properties.get(flag.arg).unwrap_or_else(|| {
                        panic!("'{}' is not an argument of '{}'", flag.arg, family.tool)
                    });
                    let kind = property["type"].as_str().unwrap_or_default();
                    match flag.kind {
                        // The command line carries every non-list flag as one
                        // string, so a `Value` flag can only fill a string
                        // argument — anything else could never be set.
                        FlagKind::Value => assert_eq!(
                            kind, "string",
                            "'{}' of '{}' is not a string argument",
                            flag.arg, family.tool
                        ),
                        FlagKind::List => assert_eq!(
                            kind, "array",
                            "'{}' of '{}' is not the array argument",
                            flag.arg, family.tool
                        ),
                        FlagKind::Present => assert_eq!(
                            kind, "boolean",
                            "'{}' of '{}' is not the boolean argument",
                            flag.arg, family.tool
                        ),
                    }
                    reachable.push(flag.arg);
                }
            }
            for argument in properties.keys().filter(|key| key.as_str() != "action") {
                assert!(
                    reachable.contains(&argument.as_str()),
                    "'{argument}' of '{}' has no command-line form",
                    family.tool
                );
            }
        }
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
    /// learn no less than one reading the family's.
    #[test]
    fn every_help_states_the_running_product_and_the_exit_classes() {
        for family in FAMILIES {
            let helps = [
                family_help(family),
                operation_help(family, &family.operations[0]),
            ];
            for help in helps {
                for required in [
                    "must be running",
                    "owner's authority",
                    "Exit codes:",
                    "0  the operation was carried out",
                    "3  this command was misused",
                ] {
                    assert!(
                        help.contains(required),
                        "'{required}' is missing from:\n{help}"
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
    /// key is the fleet-wide form the tool documents.
    #[test]
    fn parsing_a_switch_is_true_and_an_absent_flag_leaves_no_key() {
        let full = parsed("workspace", "ticket", &["--ticket-id", "proj-7", "--full"]);
        assert_eq!(
            full.args,
            serde_json::json!({"action": "ticket", "ticket_id": "proj-7", "full": true})
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
                .is_some_and(|line| line.contains("built before") && line.contains("restart")),
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
        let invocations: [&[&str]; 5] = [
            &["config"],
            &["config", "no-such-operation"],
            &["workspace", "list", "--nope", "x"],
            &["workspace", "list", "--workspace"],
            &["workspace", "list", "proj"],
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
