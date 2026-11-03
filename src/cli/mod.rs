//! Command-line frontends onto the RUNNING product.
//!
//! A family here is no second implementation: `mahbot config …`,
//! `mahbot workspace …`, `mahbot send …` and the producing families (`document`,
//! `image`, `video`, `rasterize`, plus the document family's own `read`) carry
//! the operation into the live instance, where the very tool an agent calls runs
//! it — the same validation and limits, and the same immediate effect on the
//! product (a `send` hands its message to an agent one-way). No store is opened
//! or written here, so there is no second writer and no second copy of any rule.
//!
//! A producing family adds a destination — the COMMAND LINE's own operand, never
//! a tool argument ([`FlagKind::Destination`]), refused where the product's own
//! cleanup would later empty it ([`resolve_output_dir`]) — and takes the
//! CALLER's paths: relative to this command's directory, `~` against home, a
//! reference file that is missing or unreadable refused as misuse before
//! anything is sent ([`caller_path`], [`resolve_embedded_paths`]).
//!
//! Also here: the readiness signal and its waiter, the execution entry point
//! ([`instance`]), the family registry, the command table ([`entry_point`]), argv
//! parsing, help and [`Exit`]. The flag grammar is [`grammar`], which `mahbot audio`
//! reads too; the browser-automation command and the OpenRouter benchmark read
//! their own flags and word their own refusals.

pub(crate) mod grammar;
pub(crate) mod instance;

use std::fmt::Write as _;
use std::sync::OnceLock;

use crate::db::ipc::{CallAttempt, CallOutcome, CallRequest};
use crate::util::lock::InstanceLockState;

/// The product's top-level `--help` text: every public entry point, the command-line
/// families among them. It lives beside the registry so the tie that every entry
/// point is listed runs in the library's test lane
/// (`every_entry_point_resolves_and_is_listed_in_the_top_level_usage`).
pub const TOP_LEVEL_USAGE: &str = "\
mahbot — autonomous agentic engineering system with a GUI dashboard daemon

Usage:
  mahbot                 Launch the GUI dashboard daemon
  mahbot paused          Launch the daemon with every managed workspace
                         paused (no ticket is claimed or advanced)
  mahbot audio <operation> [flags]
                         Speech on the command line: a recording into text, text
                         into a spoken WAV file
  mahbot chrome <args>   Browser automation CLI over the shared chrome core
  mahbot debug           Read-only SQL query tool against the live stores
  mahbot bench-openrouter <args>
                         Standalone OpenRouter provider benchmark
  mahbot config <operation> [flags]
                         Set the RUNNING product up (Telegram bot, users, tool
                         grants, web search, custom roles)
  mahbot workspace <operation> [flags]
                         Work with the RUNNING product's workspaces and tickets
  mahbot send [flags]    Hand one message to an agent and finish — the answer
                         arrives later in your assistant's conversation, or
                         nothing does when the addressee is one-way
  mahbot document <operation> [flags]
                         Read a document with the RUNNING product's reader, or
                         produce one with its document tool
  mahbot image gen [flags]
                         Produce a picture with the RUNNING product's picture tool
  mahbot video <gen|edit> [flags]
                         Produce or edit a clip with the RUNNING product's video
                         tools
  mahbot rasterize [flags]
                         Turn a file into an ordinary picture with the RUNNING
                         product's rasterizer

Options:
  -h, --help             Print this help and exit
  -V, --version          Print the version and exit";

/// The transport tag the command line's own operations are attributed to: the
/// one-way `send` reaches the history as a message the owner typed (`gui`) or
/// spoke (`voice`) does, but recorded as shell-sent. It never reaches the
/// message itself.
pub(crate) const CHANNEL: &str = "cli";

/// Whether this instance can serve a command-line operation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ServiceState {
    /// The instance is still bringing itself up.
    Starting,
    /// It finished starting: the whole product, so an operation may be served.
    Ready,
    /// The instance did not finish starting up: it can never serve one.
    Unavailable,
}

/// The instance's own service state, as a `watch` so a waiter wakes the moment it
/// changes. A `OnceLock` because one process is one instance and this sender
/// never drops, so a subscriber always has a live counterpart.
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
    /// The instance began shutting down: nothing ran, so the caller may retry.
    NotAccepted,
    /// Neither finished nor failed within [`SERVICE_READY_TIMEOUT`] — a wedged
    /// boot. Nothing ran, so the caller is told instead of being held forever.
    StillStarting,
}

/// How long an operation waits for the instance to finish starting before it is
/// refused: a wedged boot must not hold the caller with no answer. Once the instance
/// has taken the operation on, it is never given up on.
const SERVICE_READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Wait until this instance can serve a command-line operation.
///
/// Nothing is served before [`mark_ready`]: until then the product is not the
/// WHOLE product, and a `mahbot_config` action or a workspace registration would
/// land in a half-built instance. The caller is never simply dropped: the wait
/// resolves on `Ready`, on `Unavailable`, on
/// [`StillStarting`](ServiceGate::StillStarting) after [`SERVICE_READY_TIMEOUT`], or
/// when the shutdown token fires — where
/// [`NotAccepted`](ServiceGate::NotAccepted) carries nothing run, so the caller
/// may retry safely.
pub(crate) async fn wait_for_service() -> ServiceGate {
    let mut rx = service_sender().subscribe();
    let shutdown = crate::shutdown::shutdown_token();
    // One bound for the whole wait, not one per wake-up.
    let bound = tokio::time::sleep(SERVICE_READY_TIMEOUT);
    tokio::pin!(bound);
    loop {
        // The product has begun its exit: nothing new is taken on, whatever the
        // state says (nothing ran, so the caller may retry).
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
                    // A process-lifetime static that dropped means the process is
                    // tearing down, where nothing can be served.
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

/// The exit-code classes a command line here separates — the same numbering the
/// browser-automation command and `mahbot audio` use.
///
/// Who judges separates misuse from failure. Misuse is what a frontend settles
/// without the target: a word or flag nobody has, a flag with no value, a token
/// that is no flag, an argument that is not text — and, for a frontend that opens
/// files itself, the filesystem's verdict on the paths it was handed. Everything
/// the TARGET judges is a failed operation instead. A frontend that only CARRIES
/// the operation (this one) holds no copy of the target's rules, so it never
/// guesses a class; `mahbot audio`, which RUNS its target, may judge that target's
/// published vocabulary as misuse. Each family's help says what its classes mean.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Exit {
    /// The operation was carried out.
    Ok = 0,
    /// Not carried out, carried out only in part, or an outcome unknowable after
    /// the call went out (it may have happened — never the environment's class).
    Failed = 1,
    /// Nothing ran: the environment — the instance, the engine, its models —
    /// could not take the operation on. The environment, not the operation.
    Environment = 2,
    /// The caller's own mistake: an invocation that cannot mean what it asks, and —
    /// for a frontend that opens the files itself — the paths it was handed.
    Usage = 3,
}

impl Exit {
    /// The process exit code for this class.
    pub(crate) const fn code(self) -> i32 {
        self as i32
    }
}

/// One command-line family: the frontend onto one capability of the product — one
/// of its tools, or the pair of tools that share a capability.
pub(crate) struct Family {
    /// The command-line word (`mahbot <word> …`).
    pub(crate) word: &'static str,
    /// The product tools this family fronts, in the order its help lists them —
    /// both video tools where a capability is split across two.
    pub(crate) tools: &'static [&'static str],
    /// The protocol value this family's operation frames carry: the generation of
    /// the surface it needs. A family the channel has already served carries
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
    /// `mahbot <word> <operation> [flags]`: one operation per sub-command, each
    /// running one of the family's tools — or, in one case, the product's own
    /// reader ([`DOCUMENT_READ`]).
    Operations(&'static [Operation]),
    /// The family IS its operation: `mahbot <word> [flags]`, no operation word at
    /// all, and the args carry no `action` key.
    Standalone(&'static [Flag]),
}

/// One command-line operation: the tool it runs and, where the tool has one, the
/// action it runs it under.
pub(crate) struct Operation {
    /// The tool this operation runs — one of the family's own.
    pub(crate) tool: &'static str,
    /// The tool's own `action` value, or `None` where the tool takes no action
    /// (the picture tool and both video tools): the operation then IS that tool.
    /// [`DOCUMENT_READ`] is the one action no tool holds.
    pub(crate) action: Option<&'static str>,
    /// One line for the family's help and the operation's own help.
    pub(crate) purpose: &'static str,
    /// The flags this operation takes, in the order its help lists them.
    pub(crate) flags: &'static [Flag],
}

/// The sub-command word of one operation, derived rather than written down so a
/// sub-command can never be spelled a way its tool or action is not: the action
/// with `_` written `-` (`fill_template` → `fill-template`), or the tool's own
/// name with the family word dropped (`image_gen` in `image` → `gen`).
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

/// The argument carrying the picture tool's generations: its flags are one
/// generation's own settings, so this command builds the single generation it
/// asks for itself — every other tool's flags ARE its arguments.
const GENERATIONS_ARG: &str = "generations";

/// Fold the picture tool's flat flags into the one generation this command asks for,
/// with the prompt left beside it and a flag left out left out of the generation
/// too, where the tool reads it as unset.
fn fold_one_generation(args: &mut serde_json::Value) {
    /// The settings one picture's generation carries — the tool schema's own
    /// entries, held to them by the tie in the tests.
    const GENERATION_SETTINGS: &[&str] = &["images", "aspect_ratio", "size"];
    let Some(map) = args.as_object_mut() else {
        return;
    };
    let mut generation = serde_json::Map::new();
    for setting in GENERATION_SETTINGS {
        if let Some(value) = map.remove(*setting) {
            generation.insert((*setting).to_string(), value);
        }
    }
    map.insert(
        GENERATIONS_ARG.to_string(),
        serde_json::Value::Array(vec![generation.into()]),
    );
}

/// Whether this family's operations produce FILES — derived from the flags the
/// registry already carries rather than stated a second time: every operation of a
/// producing family carries the destination flag ([`FlagKind::Destination`]) and no
/// other family carries one anywhere, which is what the registry test holds.
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
/// own sub-command words where the family's tools take no `action`, `<operation>`
/// where an `action` picks one, `[flags]` alone where the family IS its
/// operation — derived from the registry, so no help can advertise a stale word
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
    /// The tool's own argument name; the command-line spelling is derived from it
    /// (`ticket_id` → `--ticket-id`), so a flag can never be spelled a way the tool
    /// does not read. The exception is the destination flag
    /// ([`FlagKind::Destination`]), whose name is the command line's own.
    pub(crate) arg: &'static str,
    /// How a value reaches the argument.
    pub(crate) kind: FlagKind,
    /// A value the caller hands over as a secret: never echoed, recorded or
    /// answered back.
    pub(crate) secret: bool,
    /// One line for the operation's help.
    pub(crate) purpose: &'static str,
}

/// How a flag carries its value — the shape of the tool argument it fills.
///
/// Two spellings are this command line's own, read the same way by every kind
/// that carries one: `./@…` is the literal text beginning with `@` (the escape
/// exists because a text value may legitimately begin with it), and — for the
/// kinds that say so ([`Self::has_file_form`]) — `@<file>` IS what that file
/// holds. A kind whose value names a path also RESOLVES it (see [`caller_path`]):
/// a `Path` its one value, a `PathList` every entry, a `Structured` every string
/// the fields it carries name.
pub(crate) enum FlagKind {
    /// One value per occurrence: `--name bob`, read as text or — like the kinds
    /// below — taken from a FILE by writing `@<file>`.
    Value,
    /// One value per occurrence that NAMES A FILE on this machine
    /// (`--path report.pdf`), resolved before it is sent: a relative one in the
    /// directory this command was run in, and `~` against the home directory. The
    /// product has a working directory of its own, and a relative path sent on
    /// would name a different file there. It must also be there and readable —
    /// that is the caller's misuse, while a URL is left to the product.
    /// No `@<file>` form: the value already names the file.
    Path,
    /// One NUMBER per occurrence (`--duration 8`). A value that is no JSON number
    /// travels as the text it is, so the tool's own refusal — not this
    /// frontend's — is what the caller gets: whether a value is acceptable is the
    /// tool's to judge.
    Number,
    /// Repeatable, one array entry per occurrence — taken VERBATIM, so an entry
    /// that itself holds a comma is unambiguous. An EMPTY value is the EMPTY list
    /// (`--tools=`), which tools read differently from an absent key
    /// (`update_custom_role` clears the role's tools with it). An occurrence
    /// starting with `@` names a FILE holding the entries as a JSON array of
    /// strings; the `./@…` escape gives an entry that begins with `@`.
    List,
    /// [`List`](Self::List)'s file-naming form: the same array of texts, with
    /// every entry (and the file named after `@`) resolved the way
    /// [`Path`](Self::Path) resolves its one value.
    PathList,
    /// One STRUCTURED value per occurrence, for an object or an array of objects:
    /// the JSON it is written as, or `@<file>` naming a file that holds it. A
    /// value that is no JSON is sent on verbatim for the tool to judge; a file
    /// that cannot be read, or holds no JSON, is this command line's own misuse.
    ///
    /// The field names it carries are the ones a REFERENCE FILE stands under — the
    /// `image` of a block, of an edit that places one — and every string there is
    /// resolved like a [`Path`](Self::Path) value, since the product would resolve
    /// them against its own directories.
    Structured(&'static [&'static str]),
    /// A switch, for a boolean argument: the bare `--full` and `--full=true` are
    /// `true`, `--full=false` the `false` a bare switch cannot carry. Both must be
    /// spellable — an argument left out is the tool's own default, which is not
    /// `false` where the default is the provider's (a video's `--generate-audio`).
    Present,
    /// The destination of the files the operation produces — the command line's
    /// OWN operand, and the single flag no tool reads: the exception the whole
    /// registry turns on (see the module docs). No `@<file>` form: it names a
    /// directory itself.
    Destination,
}

impl FlagKind {
    /// Whether this kind reads its value from the file its `@<file>` form names.
    ///
    /// The kinds whose value can be long or structured — a text, a list, a
    /// structure — have it, so a caller is never capped by the command line's own
    /// length. [`Path`](Self::Path) and [`Destination`](Self::Destination) do not:
    /// their value names the file (or directory) itself. [`Number`](Self::Number)
    /// and [`Present`](Self::Present) do not either: neither carries text.
    #[must_use]
    const fn has_file_form(&self) -> bool {
        matches!(
            self,
            Self::Value | Self::List | Self::PathList | Self::Structured(_)
        )
    }
}

/// The document family's `read` operation word — the `action` value the instance
/// reads as "the product's own reader"
/// ([`crate::tools::read_document::read_for_command_line`]). Deliberately no action
/// of the document tool: agents read documents through the `read` tool they have,
/// so the tool gains no action and its prompt nothing new. The registry test ties
/// every tool-backed operation to its tool's actions and pins this one separately.
pub(crate) const DOCUMENT_READ: &str = "read";

/// The command lines this file fronts: the admin Assistant's own toolset — the
/// tools each family's `tools` names, and nothing else the owner holds is reachable
/// through the channel. Words are derived from here ([`family_words`],
/// [`producing_families`]), so no surface keeps a list of its own to go stale.
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
                        purpose: "'deepseek' or 'exa'",
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
                purpose: "List the managed workspaces",
                flags: &[Flag {
                    arg: "workspace",
                    kind: FlagKind::Value,
                    secret: false,
                    purpose: "One managed workspace; omit it to list every one",
                }],
            },
            Operation {
                tool: "workspace",
                action: Some("add"),
                purpose: "Add a software project directory as a managed workspace",
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
                        purpose: "Absolute path to the project directory to add",
                    },
                ],
            },
            // `pause` (and `resume`, which says the same) act on ONE workspace or
            // the whole fleet, told apart by ABSENCE: with no `--workspace` the
            // args carry no `workspace` key at all, and that absence IS the
            // fleet-wide form. Naming one is then the only field allowed — the tool
            // refuses a stray field so a fleet-wide write can never be widened by
            // accident, so never invent one and never send a `null`.
            Operation {
                tool: "workspace",
                action: Some("pause"),
                purpose: "Freeze one managed workspace, or every one when none is named",
                flags: &[Flag {
                    arg: "workspace",
                    kind: FlagKind::Value,
                    secret: false,
                    purpose: "The managed workspace to freeze; omit it to freeze EVERY \
                              managed workspace (the fleet-wide pause — a workspace that is \
                              not ready is skipped and left untouched, which makes that result \
                              a partial one)",
                }],
            },
            // See `pause`: the same fleet-wide rule, the same absent key.
            Operation {
                tool: "workspace",
                action: Some("resume"),
                purpose: "Resume one managed workspace, or every one when none is named",
                flags: &[Flag {
                    arg: "workspace",
                    kind: FlagKind::Value,
                    secret: false,
                    purpose: "The managed workspace to resume; omit it to resume EVERY \
                              managed workspace (the fleet-wide resume — a workspace that is \
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
                        purpose: "One managed workspace; omit it to cover every one",
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
        // This build's own family: a product built before it answers the call as a
        // failed operation, so it carries the current value and is refused as the
        // environment's class instead.
        protocol: crate::db::ipc::CALL_STORE,
        purpose: "Hand one message to an agent and finish — an answer arrives later in your \
                  assistant's conversation, and nothing does for the one-way 'guest' address",
        surface: Surface::Standalone(&[
            Flag {
                arg: "message",
                kind: FlagKind::Value,
                secret: false,
                purpose: "The work to hand over; make it self-contained — the addressee cannot \
                          see your conversation (required)",
            },
            // The address vocabulary's own words, which `tools::send` checks name
            // every address and every argument an address requires.
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
                purpose: "The folder the addressee works in: for 'manager', the managed \
                          workspace to message, by its managed name or its own folder; for \
                          'role', a managed workspace, a folder path, or 'personal:<user>'; a \
                          worker or a worker branch works in a directory outside the managed \
                          workspaces (one inside is refused), and in your own workspace when it \
                          names none",
            },
            Flag {
                arg: "branch",
                kind: FlagKind::Value,
                secret: false,
                purpose: "The branch to continue or create — a worker branch's, or a custom \
                          role's own named branch (ASCII letters, digits, '_' and '-', at most \
                          32 chars)",
            },
            Flag {
                arg: "name",
                kind: FlagKind::Value,
                secret: false,
                purpose: "The name the chosen address takes: for 'role', the custom role to \
                          start (it must exist); for 'guest', the account name of the guest whose \
                          own assistant receives the message",
            },
        ]),
    },
    // These families front the tools every Assistant holds regardless of account (the
    // media models resolve per-user), so the command line reaches what the owner's own
    // assistant can already do. What it adds is the DESTINATION ([`FlagKind::Destination`]).
    Family {
        word: "document",
        tools: &["document"],
        // This build's own families: a product built before them serves no such
        // tool, so the call comes back as the environment's class — nothing ran —
        // rather than as a failed operation.
        protocol: crate::db::ipc::CALL_STORE,
        purpose: "Read a document with the product's own reader, or produce one with its document \
                  tool — write, fill a template, edit an Office file, and every PDF operation",
        surface: Surface::Operations(&[
            Operation {
                tool: "document",
                action: Some(DOCUMENT_READ),
                purpose: "Read a document with the running product's own reader: its full text and \
                          every image it produces, in a folder of their own",
                flags: &[DOCUMENT_PATH_FLAG, DOCUMENT_FILE_NAME_FLAG, OUTPUT_DIR_FLAG],
            },
            Operation {
                tool: "document",
                action: Some("create"),
                purpose: "Write a new file: docx/pptx/pdf blocks, or xlsx sheets",
                flags: &[
                    DOCUMENT_FORMAT_FLAG,
                    DOCUMENT_CONTENT_FLAG,
                    DOCUMENT_SHEETS_FLAG,
                    DOCUMENT_SLIDE_SIZE_FLAG,
                    DOCUMENT_HEADER_FLAG,
                    DOCUMENT_FOOTER_FLAG,
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
                    IMAGE_EDITS_FLAG,
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
                action: Some("pdf_crop"),
                purpose: "Crop a PDF page's visible area",
                flags: &[
                    DOCUMENT_PATH_FLAG,
                    DOCUMENT_AREA_FLAG,
                    DOCUMENT_PAGES_FLAG,
                    DOCUMENT_METADATA_FLAG,
                    DOCUMENT_FILE_NAME_FLAG,
                    OUTPUT_DIR_FLAG,
                ],
            },
            Operation {
                tool: "document",
                action: Some("pdf_resize"),
                purpose: "Resize a PDF page, scaling its content to fit",
                flags: &[
                    DOCUMENT_PATH_FLAG,
                    DOCUMENT_WIDTH_FLAG,
                    DOCUMENT_HEIGHT_FLAG,
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
                    DOCUMENT_OPACITY_FLAG,
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
        word: "rasterize",
        tools: &["rasterize"],
        protocol: crate::db::ipc::CALL_STORE,
        purpose: "Turn a file into an ordinary picture with the running product's own rasterizer",
        surface: Surface::Standalone(&[RASTER_PATH_FLAG, RASTER_PAGE_FLAG, OUTPUT_DIR_FLAG]),
    },
];

const RASTER_PATH_FLAG: Flag = Flag {
    arg: "path",
    kind: FlagKind::Path,
    secret: false,
    purpose: "The file to render — an SVG, one page of a PDF, or an HTML file inside your \
              personal workspace — or an http(s) address; resolved where this command runs, and \
              read with the owner's own authority",
};

const RASTER_PAGE_FLAG: Flag = Flag {
    arg: "page",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The 1-based page to render, for a PDF of several pages",
};

/// The destination every producing operation carries: the command line's OWN
/// operand, the one flag no tool reads (see [`FlagKind::Destination`]).
const OUTPUT_DIR_FLAG: Flag = Flag {
    arg: "output_dir",
    kind: FlagKind::Destination,
    secret: false,
    purpose: "The directory the operation writes into — the produced file(s), or the one folder \
              they are all in; it must already exist, and it is the directory this command is run \
              in when it is left out",
};

/// The base name of what it produces — every document operation carries it.
const DOCUMENT_FILE_NAME_FLAG: Flag = Flag {
    arg: "file_name",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The base name of what is produced, without an extension where one is appended; a \
              free name is used, so nothing already there is overwritten",
};

const DOCUMENT_FORMAT_FLAG: Flag = Flag {
    arg: "format",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The format to write: docx, pptx, xlsx or pdf",
};

/// `create`'s slide size (a `pdf_text`'s `size` is a font size, its own flag).
const DOCUMENT_SLIDE_SIZE_FLAG: Flag = Flag {
    arg: "size",
    kind: FlagKind::Value,
    secret: false,
    purpose: "A pptx slide size: widescreen, standard or portrait; unnamed is the tool's own default deck",
};

/// `create`'s blocks, for every format but xlsx.
const DOCUMENT_CONTENT_FLAG: Flag = Flag {
    arg: "content",
    kind: FlagKind::Structured(&["image"]),
    secret: false,
    purpose: "The blocks to lay down, as a JSON array — one object per block, with the \
              'text'/'table'/'image'/'...' kind it names beside its own fields",
};

const DOCUMENT_HEADER_FLAG: Flag = Flag {
    arg: "header",
    kind: FlagKind::Value,
    secret: false,
    purpose: "A created .docx's running header, one paragraph per line; {page} is the page \
              number and {pages} the document's page count",
};

const DOCUMENT_FOOTER_FLAG: Flag = Flag {
    arg: "footer",
    kind: FlagKind::Value,
    secret: false,
    purpose: "A created .docx's running footer, one paragraph per line; {page} is the page \
              number and {pages} the document's page count",
};

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

/// The one input most operations act on — and the file the reader reads.
const DOCUMENT_PATH_FLAG: Flag = Flag {
    arg: "path",
    kind: FlagKind::Path,
    secret: false,
    purpose: "The input file to work on",
};

const DOCUMENT_PATHS_FLAG: Flag = Flag {
    arg: "paths",
    kind: FlagKind::PathList,
    secret: false,
    purpose: "One input PDF, in the order it must be merged; repeat once per file, or name a \
              file holding them as a JSON array with '@<file>'",
};

const DOCUMENT_PAGES_FLAG: Flag = Flag {
    arg: "pages",
    kind: FlagKind::List,
    secret: false,
    purpose: "One page entry: \"all\", a page, a range, or a comma list like \"1-3,5\" (\"all\" \
              stands alone); repeat once per entry, or name a file holding them as a JSON array \
              with '@<file>'; left out, every page",
};

/// The document tool's `edits` argument, whose `add_image` names a reference
/// file — so the list resolves that `image` where the command runs.
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

const DOCUMENT_VALUES_FLAG: Flag = Flag {
    arg: "values",
    kind: FlagKind::Structured(&[]),
    secret: false,
    purpose: "The values to write, as a JSON object keyed by placeholder name (or form field \
              name)",
};

const DOCUMENT_DEGREES_FLAG: Flag = Flag {
    arg: "degrees",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The rotation to add, in degrees (a multiple of the tool's own step)",
};

const DOCUMENT_AREA_FLAG: Flag = Flag {
    arg: "area",
    kind: FlagKind::Structured(&[]),
    secret: false,
    purpose: "The area to keep, as a JSON object of x, y, width and height — fractions of the \
              page's visible area, from the displayed top-left",
};

const DOCUMENT_TEXT_FLAG: Flag = Flag {
    arg: "text",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The text to draw",
};

const DOCUMENT_IMAGE_FLAG: Flag = Flag {
    arg: "image",
    kind: FlagKind::Path,
    secret: false,
    purpose: "The PNG or JPEG to place",
};

const DOCUMENT_X_FLAG: Flag = Flag {
    arg: "x",
    kind: FlagKind::Number,
    secret: false,
    purpose: "Across the page, in points",
};

const DOCUMENT_Y_FLAG: Flag = Flag {
    arg: "y",
    kind: FlagKind::Number,
    secret: false,
    purpose: "Down the page, in points",
};

const DOCUMENT_SIZE_FLAG: Flag = Flag {
    arg: "size",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The font size, in points",
};

const DOCUMENT_COLOR_FLAG: Flag = Flag {
    arg: "color",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The text colour, as hex digits",
};

const DOCUMENT_STAMP_FLAG: Flag = Flag {
    arg: "stamp",
    kind: FlagKind::Present,
    secret: false,
    purpose: "Draw a box around the text",
};

const DOCUMENT_OPACITY_FLAG: Flag = Flag {
    arg: "opacity",
    kind: FlagKind::Number,
    secret: false,
    purpose: "Opacity from just above zero to one; unnamed is fully opaque",
};

const DOCUMENT_ROTATE_FLAG: Flag = Flag {
    arg: "rotate",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The text's own rotation, in degrees",
};

/// `pdf_image`'s placed width, or `pdf_resize`'s new page width.
const DOCUMENT_WIDTH_FLAG: Flag = Flag {
    arg: "width",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The width, in points: a picture's placed width, or the page width a resize makes",
};

/// `pdf_image`'s placed height, or `pdf_resize`'s new page height.
const DOCUMENT_HEIGHT_FLAG: Flag = Flag {
    arg: "height",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The height, in points: a picture's placed height, or the page height a resize makes",
};

const DOCUMENT_FLATTEN_FLAG: Flag = Flag {
    arg: "flatten",
    kind: FlagKind::Present,
    secret: false,
    purpose: "Bake the fields into the page content and remove them",
};

const IMAGE_PROMPT_FLAG: Flag = Flag {
    arg: "prompt",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The text description of the picture to produce (required)",
};

/// The one generation's reference pictures: given any, it is picture-to-picture.
const IMAGE_IMAGES_FLAG: Flag = Flag {
    arg: "images",
    kind: FlagKind::PathList,
    secret: false,
    purpose: "One reference picture, by path; repeat once per picture, or name a file holding \
              them as a JSON array with '@<file>' — given any, the request is \
              picture-to-picture rather than text-to-picture",
};

const IMAGE_ASPECT_RATIO_FLAG: Flag = Flag {
    arg: "aspect_ratio",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The aspect ratio, like '16:9', '1:1' or '4:3'",
};

const IMAGE_SIZE_FLAG: Flag = Flag {
    arg: "size",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The picture's size — a tier like '1K'/'2K' or explicit pixels like '1024x1024'; \
              a model that cannot express the value is not sent the request",
};

const VIDEO_PROMPT_FLAG: Flag = Flag {
    arg: "prompt",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The text description of the clip to produce (required)",
};

const VIDEO_IMAGES_FLAG: Flag = Flag {
    arg: "images",
    kind: FlagKind::PathList,
    secret: false,
    purpose: "The path of ONE reference picture to start the clip from — at most one is \
              accepted, and given it the request is picture-to-clip rather than \
              text-to-clip",
};

const VIDEO_DURATION_FLAG: Flag = Flag {
    arg: "duration",
    kind: FlagKind::Number,
    secret: false,
    purpose: "The clip's length in seconds — the values the account's own model takes are the \
              ones it takes",
};

const VIDEO_RESOLUTION_FLAG: Flag = Flag {
    arg: "resolution",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The resolution to ask the model for, as that model states it",
};

const VIDEO_ASPECT_RATIO_FLAG: Flag = Flag {
    arg: "aspect_ratio",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The aspect ratio to ask the model for, as that model states it",
};

const VIDEO_SIZE_FLAG: Flag = Flag {
    arg: "size",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The exact size, as WxH (like '1280x720')",
};

const VIDEO_GENERATE_AUDIO_FLAG: Flag = Flag {
    arg: "generate_audio",
    kind: FlagKind::Present,
    secret: false,
    purpose: "Ask for an audio track, where the model offers one — omitted, the model's own \
              default applies and `=false` asks for a silent clip",
};

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

const EDIT_INSTRUCTION_FLAG: Flag = Flag {
    arg: "instruction",
    kind: FlagKind::Value,
    secret: false,
    purpose: "The edit to apply, described in words (required)",
};

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

/// The frame-anchor form's first anchor.
const EDIT_FIRST_FRAME_FLAG: Flag = Flag {
    arg: "first_frame",
    kind: FlagKind::Path,
    secret: false,
    purpose: "The picture to use as the exact first frame, by local path or public https URL — \
              the frame-anchor form, which excludes a clip and reference pictures",
};

/// The frame-anchor form's second anchor.
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
/// [`entry_point`]. Opaque on purpose: the gate and the dispatch cannot disagree.
pub struct Cli(&'static Family);

/// Resolve a command-line word to its family: the registry lookup under
/// [`entry_point`].
#[must_use]
fn family(word: &str) -> Option<Cli> {
    FAMILIES.iter().find(|family| family.word == word).map(Cli)
}

/// The word the login registration passes, so a start the system made at login can
/// tell itself apart from a hand launch. Hidden like `__grep-engine`.
pub(crate) const LOGIN_START_ARG: &str = "__login-start";

/// One entry point a launch's first word can name: the binary's whole command table,
/// the families beside the fixed words. The ladder dispatches on this answer alone,
/// so a word with no variant here is refused as the caller's own misuse.
pub enum EntryPoint {
    /// `mahbot` with no first word: the dashboard daemon.
    Daemon,
    /// `mahbot paused`: the daemon, with every managed workspace paused before
    /// the poll loop starts.
    Paused,
    /// `mahbot __login-start`: the daemon as the login registration starts it — not a
    /// second launch, coming up with the tray only.
    LoginStart,
    /// `-h`/`--help`.
    Help,
    /// `-V`/`--version`.
    Version,
    /// `mahbot debug` — the read-only query tool ([`crate::db::debug`]).
    Debug,
    /// `mahbot __grep-engine` — the shell's own grep serving.
    GrepEngine,
    /// `mahbot __env-dump` — the shell-environment reader's child dumper.
    EnvDump,
    /// `mahbot bench-openrouter` — the standalone provider benchmark.
    BenchOpenRouter,
    /// `mahbot audio` — the speech command.
    Audio,
    /// `mahbot chrome` — the browser-automation command.
    Chrome,
    /// One of the command-line families ([`FAMILIES`]).
    Family(Cli),
}

/// Resolve a launch's first word — `None` when it had none — to the entry point it
/// names, or `None` when this build has no such word. The words are compiled in, so
/// an instance takes no part in the answer.
#[must_use]
pub fn entry_point(word: Option<&str>) -> Option<EntryPoint> {
    Some(match word {
        None => EntryPoint::Daemon,
        Some("paused") => EntryPoint::Paused,
        Some(LOGIN_START_ARG) => EntryPoint::LoginStart,
        Some("-h" | "--help") => EntryPoint::Help,
        Some("-V" | "--version") => EntryPoint::Version,
        Some("debug") => EntryPoint::Debug,
        Some("__grep-engine") => EntryPoint::GrepEngine,
        Some("__env-dump") => EntryPoint::EnvDump,
        Some("bench-openrouter") => EntryPoint::BenchOpenRouter,
        Some("audio") => EntryPoint::Audio,
        Some("chrome") => EntryPoint::Chrome,
        Some(word) => EntryPoint::Family(family(word)?),
    })
}

/// Whether this command line fronts `tool` by name. The instance serves an
/// operation only for a tool a family fronts, so the surface reachable through
/// the channel is exactly these frontends — never the owner's whole toolset.
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

/// The family words that make files ([`produces_files`]), as the read-only surfaces
/// name them: backticked, joined by a slash. Derived here alone, so a family added
/// to the registry is named — and refused — by every one of them without an edit.
pub(crate) fn producing_families() -> String {
    FAMILIES
        .iter()
        .filter(|family| produces_files(family))
        .map(|family| format!("`{}`", family.word))
        .collect::<Vec<_>>()
        .join("/")
}

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
    /// The tool's args object: the `action` where the tool takes one, one whose
    /// tool takes none has no `action` key at all, plus exactly the flags given.
    /// An absent flag leaves its key ABSENT, never null — the tools tell the two
    /// apart, and `pause`/`resume` read the absent `workspace` as the fleet-wide form.
    args: serde_json::Value,
    /// The values the `secret` flags really handed over, in order, as they
    /// travelled: a `@<file>` secret is the FILE's content. Nothing else is sent.
    secrets: Vec<String>,
    /// The directory the produced files must go into, exactly as the caller
    /// spelled it — `None` when the flag was left out. The DESTINATION is not a
    /// tool argument (see [`FlagKind::Destination`]): it travels beside them, and
    /// is resolved before anything is sent ([`resolve_output_dir`]).
    output_dir: Option<String>,
}

/// A flag's command-line spelling: the argument name with `_` written `-`.
fn flag_spelling(arg: &str) -> String {
    arg.replace('_', "-")
}

/// Parse one operation's flag tail — the tokens after the operation word.
///
/// The grammar is [`grammar`]; what is this frontend's own is the handling under
/// it, with a `List` accumulating an entry per occurrence. Every reason names a
/// flag and never a VALUE — a mistyped secret must not travel back out — and a
/// token that is no flag is not echoed either. What this frontend judges by itself
/// is [`Exit::Usage`]; a value the tool refuses travels as written and is a failed
/// operation, which is why a `Number` or a `Structured` value that is neither
/// travels as the text it is.
///
/// `action` is the tool's own action, and `None` where the tool takes none.
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
    // The flags already given, so a repeat is refused rather than silently kept,
    // and the `List` entries, which become the array at the end.
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
                    // A secret is what the flag handed over, never the spelling that
                    // named it: the file form makes the FILE's content the value.
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

/// Put one value-carrying flag's value where it belongs: into the args object for
/// every kind a tool reads, and aside for the destination, which no tool reads.
///
/// The `@` file form and the `./@…` escape are read off the value once, here, for
/// every kind; reading a FILE is only the kinds that have that form
/// ([`FlagKind::has_file_form`]). A flag whose value names a path itself is never
/// read from a file: it names the file.
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
            // One array entry per occurrence, verbatim: never split on its commas,
            // so `name:kind1,kind2` stays one tool specification. An EMPTY value
            // empties the list as of that point ([`FlagKind::List`]); a path list
            // resolves every entry the way a single path is resolved.
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

/// The texts a flag's argument now holds, as the values a secret flag really handed
/// over — for the file forms, what the file held rather than the spelling.
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

/// The JSON a `Number` flag carries: the number it is, else the text — whether the
/// tool takes it is the tool's to judge.
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
/// `~` expands against the home directory and a relative path resolves in the
/// directory this command was run in — the product's working directory is another
/// one, so a path sent on as written would name a different file. The reference
/// material must be THERE, readable and a REGULAR file: the product reads it with
/// no timeout, so a pipe with no writer or an endless device would hold the daemon
/// forever. Missing or unreadable is misuse, judged from metadata alone before
/// anything is sent, so the refusal cannot block; a URL is left to the product.
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

/// Resolve every reference file a structured value names under one of `fields` (a
/// document's image block, an edit that places a picture), which the product would
/// otherwise resolve against its OWN directories (a workspace's `generated/`, a
/// personal workspace's `uploads/`). Each string there, and each string of an array
/// there, is resolved like a `--path` value ([`caller_path`]); a member of another
/// shape is left to the tool.
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

/// Upper bound on the file a flag's `@<file>` form may read: a value travels as one
/// argument, so a file far past any command line is a mistaken path rather than a
/// value — and the bound keeps an endless file from becoming an out-of-memory.
const MAX_VALUE_FILE_BYTES: u64 = 8 * 1024 * 1024;

/// The text of the file a flag's `@<file>` form names — the ONE read every value
/// form shares, so no two of them can guard differently.
///
/// The file must be a REGULAR file of a bounded size: the parse has no timeout, so
/// a pipe (`--prompt @<fifo>`) would hold the command forever and an endless device
/// (`--prompt @/dev/zero`) would read until memory ran out. That is the caller's
/// own misuse, and the reason names the FLAG and never the path.
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
/// trailing line break — a file's last newline is the file's, not the value's. A
/// file it cannot use is misuse, and the reason names the FLAG, never the path.
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
/// holds; a file it cannot use is misuse, the reason naming the FLAG, not the path.
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

/// The exit classes and the one sentence that makes them the contract — shared by
/// every help, so no two can word it differently.
const EXIT_CLASS_HELP: &str = "\nExit codes:\n  0  the operation was carried out\n  1  the \
     operation failed — the product refused it, carried it out only in part, or the exchange was \
     severed after the call went out (an operation that may be under way is never the \
     environment's class)\n  2  the environment — nothing ran: the product is not up, its \
     channel was not reachable, or it could not take the operation on\n  3  this command was \
     misused\nA script reads the exit code, not the text: a refusal still prints the product's \
     own answer on stdout, so only the code says whether the operation happened.\n";

/// What one of these command lines IS — shared by every help, so a caller entering
/// through an operation's help learns the same contract as one reading the family's.
const RUNNING_CONTRACT_HELP: &str = "\nA frontend onto the RUNNING product, acting with the \
     owner's authority and with no agent session behind it: the operation is carried into the \
     instance that is already up and the product itself runs it there — the same validation, and \
     the same immediate effect (a value it writes is in use at once, with no restart). The product \
     must be running: there is no way to run these operations against its stores directly.\n";

/// Where the two streams go — shared by every help, like the contract blocks.
const OUTPUT_HELP: &str = "\nOutput:\n  stdout  the product's answer, word for word\n  stderr  \
     this command's own diagnostics\n";

/// How a value may be given from a file — shared by every help, because the form
/// belongs to this command line rather than to any family's flags.
const VALUE_FILE_HELP: &str = "\nValues:\n  A text value, a list or a structure may also be given \
     from a FILE by writing `@<file>`: a text value is the file's whole text (without one trailing \
     line break), while a list and a structure must hold JSON there. Write `./@…` for a value that \
     itself begins with `@` — the `./` is dropped and no file is read. A flag whose value names a \
     file or a directory itself (`--path`, `--output-dir`) takes no file form.\n";

/// What a producing family's help adds to the tool's own flags: where produced
/// files go, and how the caller's paths are read — shared by every help of such a
/// family. The destination row is rendered from the registry
/// ([`OUTPUT_DIR_FLAG`]), so its spelling and description are stated once, and
/// the "Paths" paragraph belongs here because the resolution is this frontend's.
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

/// `mahbot <word> -h`: what this family's command line is and what a caller reads.
fn family_help(family: &Family) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "mahbot {} — {}", family.word, family.purpose);
    out.push_str(RUNNING_CONTRACT_HELP);
    match family.surface {
        Surface::Operations(operations) => {
            let _ = writeln!(
                out,
                "\nEvery operation below is carried out inside the product, by the same code the \
                 owner's assistant runs — so this command line brings no behaviour of its own."
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
                 other flag belongs to the operation whose argument it names.\n\n  -h, --help  \
                 Print this help and exit\n"
            } else {
                "\nFlags:\n  None are common to every operation: a flag belongs to the operation \
                 whose argument it names.\n\n  -h, --help  Print this help and exit\n"
            });
        }
        // The family word IS the operation: no operation list below.
        Surface::Standalone(flags) => {
            let _ = write!(out, "\nUsage: {}\n", family_syntax(family));
            if produces_files(family) {
                out.push_str(&producing_help());
            }
            out.push_str("\nFlags:\n");
            out.push_str(&flag_table(flags));
            out.push_str("\n  -h, --help  Print this help and exit\n");
        }
    }
    out.push_str(EXIT_CLASS_HELP);
    out.push_str(OUTPUT_HELP);
    out.push_str(VALUE_FILE_HELP);
    out
}

/// A flag's purpose with the shared vocabulary filled in, from the rules themselves,
/// so the help cannot drift from the schema and the reading.
fn flag_purpose(flag: &Flag) -> String {
    flag.purpose
        .replace(
            "{{pdf_metadata_fields}}",
            &crate::docgen::pdf_metadata_fields_spelled(),
        )
        .replace("{{pdf_date_format}}", crate::docgen::pdf_date_format())
}

/// The flag table of a help: one width-aligned row per flag, a secret's value never
/// echoed here because it is never echoed anywhere.
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

/// A help's flag table: the destination is stated separately, being the command
/// line's own operand, and a table with nothing else in it says so.
fn flag_table(flags: &[Flag]) -> String {
    let flags: Vec<&Flag> = flags
        .iter()
        .filter(|flag| !matches!(flag.kind, FlagKind::Destination))
        .collect();
    if flags.is_empty() {
        return "  none\n".to_string();
    }
    flag_rows(flags)
}

/// `mahbot <word> <operation> -h`: what one operation does and the flags it takes —
/// plus the same contract the family help states, since a caller may enter here.
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
    out.push_str(&flag_table(operation.flags));
    out.push_str(RUNNING_CONTRACT_HELP);
    out.push_str(EXIT_CLASS_HELP);
    out.push_str(OUTPUT_HELP);
    out.push_str(VALUE_FILE_HELP);
    out
}

/// Run one family's command line, returning the process exit code.
///
/// The product's answer goes to STDOUT exactly as it worded it; everything this
/// command says on its own goes to STDERR, help to stdout, and the exit code is
/// what a script reads ([`Exit`]). A producing family's destination is resolved
/// HERE, before the environment is probed: it is misuse this command judges by
/// itself, so nothing is sent.
pub async fn run(cli: Cli, args: &[String]) -> i32 {
    let family = cli.0;
    let (tool, mut parsed) = match family.surface {
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
                // A `-h` where a flag could stand is this operation's help — the rule
                // the browser-automation command follows too, turning a would-be usage
                // error into help. Where a VALUE stands it is that value instead (see
                // `parse_flags`): a script must never read success for a run that did not happen.
                Ok(Tail::Help) => {
                    crate::util::print_stdout(&operation_help(family, operation));
                    return Exit::Ok.code();
                }
                Err(reason) => return misuse(family, &reason),
            }
        }
        // The family word IS the operation and the tool takes no `action`, so the
        // first token is already a flag: a bare `mahbot send` sends nothing and the
        // product refuses it as the missing message — a failed operation, since this
        // frontend never judges what the tool requires.
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

    // The picture tool's flags are one generation's settings, and this asks for one
    // picture: it builds that generation itself.
    if tool == "image_gen" {
        fold_one_generation(&mut parsed.args);
    }

    // The destination, like the words: a producing family writes where the caller
    // says, and a place it cannot write into is misuse, judged before the probe.
    let output_dir = if produces_files(family) {
        match resolve_output_dir(parsed.output_dir.as_deref()) {
            Ok(dir) => Some(dir),
            Err(reason) => return misuse(family, &reason),
        }
    } else {
        None
    };

    // These commands act ON the running product and cannot run otherwise, so the
    // storage root and the instance lock are settled before anything is sent.
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
/// The probe is the same fail-closed one `mahbot debug` routes on: an `Unknown`
/// lock is never read as "no instance is running". A FREE lock is not "nothing is
/// running" either — a product restarting or updating itself releases it for the
/// seconds the replacement needs — so it is ridden out over the window the channel
/// itself uses (`MAHBOT_IPC_BOUND_TIMEOUT_SECS`), and only a window that stays
/// free is reported as nothing running.
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

/// Report a misuse on STDERR and hand back its class: the reason, then the line
/// that says how to see the right form. Nothing was sent and nothing judged.
fn misuse(family: &Family, reason: &str) -> i32 {
    crate::util::print_stderr(&format!("mahbot {}: {reason}", family.word));
    crate::util::print_stderr(&format!("run 'mahbot {} -h' for usage.", family.word));
    Exit::Usage.code()
}

/// Resolve a producing command's destination — the `--output-dir` value, or the
/// directory this command was run in — into the absolute directory the produced
/// files go into, or the reason it cannot be used. Shared with the `mahbot chrome` screenshot.
///
/// It is resolved where this command runs, never against the product's working
/// directory; it must already exist (no directory is created for the caller) and
/// must not be a place the product keeps for itself, where a produced file would
/// not stay ([`cleanup_governed`]). Every failure is the caller's misuse, judged
/// before anything is sent. The one gap: a daemon whose temp environment differs
/// from this process's may scan a root this process cannot name
/// ([`crate::temp::cleanup_scan_roots_unpinned`]).
pub(crate) fn resolve_output_dir(named: Option<&str>) -> Result<String, String> {
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
    // A destination the caller cannot write into is the caller's own to fix, judged
    // here so the exit code says "your invocation" and nothing is sent. `access(2)`
    // answers it on unix without writing into the caller's directory; off unix the
    // product's own refusal is the answer.
    #[cfg(unix)]
    if !crate::util::dir_is_writable(&resolved) {
        return Err(format!("{what} cannot be written into"));
    }
    // The plain spelling, so the value that travels is the one the caller wrote.
    Ok(crate::util::strip_verbatim_prefix(&resolved)
        .to_string_lossy()
        .into_owned())
}

/// The place the product keeps for itself that `dir` lies in, or `None` when the
/// directory is the caller's own to use.
///
/// The roots come from [`crate::temp`] — the module that owns where the product's
/// cleaner may act, so the two cannot disagree — plus its data directory, whose
/// stores, personal-workspace `generated`/`uploads` (emptied by its media sweep)
/// and document-kit scratch are its own. The test is on the RESOLVED path, so a
/// link into one of them is refused all the same ([`crate::util::is_within`]), and
/// a machine's temp area is refused as a whole.
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
/// everywhere, never naming it: the bytes may be a value the caller meant as a secret.
const NON_TEXT_ARGUMENT: &str = "an argument that is not valid text was given — no entry point \
                                 and no value can be read from it, and it is not printed here; \
                                 check what you passed";

/// Refuse an argument the runner cannot read as text, returning this command line's
/// misuse class — needed before the entry point can even be told, since a non-text
/// `argv[1]` names none of them, and the panic the text-only reader raises would
/// break the classes a script reads and print the caller's own bytes.
#[must_use]
pub fn refuse_non_text_argument() -> i32 {
    crate::util::print_stderr(&format!("mahbot: {NON_TEXT_ARGUMENT}"));
    Exit::Usage.code()
}

/// What a first word no entry point has is told. It is never echoed
/// ([`grammar::read_token`]): a first word is as likely a stray secret as a typo.
const UNKNOWN_ENTRY_POINT: &str = "the first word names no entry point of this build — nothing \
                                   was started; run 'mahbot -h' for the commands this build has";

/// Refuse a first word this build has no entry point for, returning this command
/// line's misuse class. It must not fall through to the daemon, where it would start
/// one or report the instance lock instead of the caller's own mistake.
#[must_use]
pub fn refuse_unknown_entry_point() -> i32 {
    crate::util::print_stderr(&format!("mahbot: {UNKNOWN_ENTRY_POINT}"));
    Exit::Usage.code()
}

/// What one attempt's result means for the caller.
struct Verdict {
    /// The product's own answer, printed to stdout verbatim — present whenever it answered
    /// the operation itself, a refusal included.
    answer: Option<String>,
    /// This command's own diagnostic, printed to stderr.
    diagnostic: Option<String>,
    /// The exit class the caller reads.
    code: Exit,
}

/// Classify one attempt at an operation. A pure function of the attempt, so the
/// decision can be read — and tested — with no socket near it.
///
/// An answer the product gave goes to stdout whatever its outcome: the exit code,
/// not the text, says whether the operation happened, and a partial run adds one
/// short diagnostic, so the caller need not read the product's wording for a
/// verdict. An operation never taken on has no answer: the product's wording (or
/// this command's) goes to stderr and the class is the environment's. An exchange
/// severed after the frame went out is NOT the environment's — it may have
/// happened — so it is a failed operation, and the diagnostic states once that the
/// outcome is unknown and that it was deliberately not sent again.
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
            // A product that did not take the operation on: `Unavailable` reaches the
            // caller as the instance answered it (an unrestricted socket, a failed or
            // unfinished boot, whose sentence says whether a retry may help), and
            // `NotAccepted` is folded into `NotServed` by the wire client. Nothing was
            // answered about the operation itself, so the instance's sentence becomes
            // this command's diagnostic, prefixed like the rest.
            CallOutcome::NotAccepted | CallOutcome::Unavailable => Verdict {
                answer: None,
                diagnostic: Some(format!("mahbot {word}: {}", reply.answer)),
                code: Exit::Environment,
            },
        },
        // Never reached the product: nothing ran, and this command's wording says what
        // to do next.
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
        // Nothing readable came back after the frame went out: the operation may have
        // been carried out, so this is a failed operation, never the environment's —
        // stated ONCE, with the cause in parentheses and no claim about the class.
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

    /// Every entry point this build has is discoverable from `mahbot -h` — the
    /// families in the derived form ([`family_syntax`]), the rest by their own word —
    /// so none is reachable only by someone who already knew it, `mahbot audio`
    /// above all on a platform where it can only refuse. Resolution is pinned word
    /// by word, so a misrouting is caught here rather than by a run, and both
    /// directions are held: a word with no entry point resolves to nothing and never
    /// falls back to the daemon, which is the one wordless entry.
    #[test]
    fn every_entry_point_resolves_and_is_listed_in_the_top_level_usage() {
        assert!(matches!(entry_point(Some("-h")), Some(EntryPoint::Help)));
        assert!(matches!(
            entry_point(Some("--help")),
            Some(EntryPoint::Help)
        ));
        assert!(matches!(entry_point(Some("-V")), Some(EntryPoint::Version)));
        assert!(matches!(
            entry_point(Some("--version")),
            Some(EntryPoint::Version)
        ));
        assert!(matches!(
            entry_point(Some("paused")),
            Some(EntryPoint::Paused)
        ));
        assert!(matches!(
            entry_point(Some("debug")),
            Some(EntryPoint::Debug)
        ));
        assert!(matches!(
            entry_point(Some("__grep-engine")),
            Some(EntryPoint::GrepEngine)
        ));
        assert!(matches!(
            entry_point(Some("__env-dump")),
            Some(EntryPoint::EnvDump)
        ));
        assert!(matches!(
            entry_point(Some(LOGIN_START_ARG)),
            Some(EntryPoint::LoginStart)
        ));
        assert!(matches!(
            entry_point(Some("bench-openrouter")),
            Some(EntryPoint::BenchOpenRouter)
        ));
        assert!(matches!(
            entry_point(Some("audio")),
            Some(EntryPoint::Audio)
        ));
        assert!(matches!(
            entry_point(Some("chrome")),
            Some(EntryPoint::Chrome)
        ));
        for word in ["paused", "audio", "chrome", "debug", "bench-openrouter"] {
            let entry = format!("  mahbot {word} ");
            assert!(
                TOP_LEVEL_USAGE.lines().any(|line| line.starts_with(&entry)),
                "'mahbot {word}' has no entry in the top-level usage"
            );
        }
        for family in FAMILIES {
            assert!(
                matches!(entry_point(Some(family.word)), Some(EntryPoint::Family(_))),
                "'{}' must resolve to its family",
                family.word
            );
            let entry = format!("  {}", family_syntax(family));
            assert!(
                TOP_LEVEL_USAGE.lines().any(|line| line.starts_with(&entry)),
                "'{entry}' has no entry in the top-level usage"
            );
        }
        assert!(matches!(entry_point(None), Some(EntryPoint::Daemon)));
        assert!(entry_point(Some("frobnicate")).is_none());
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

    /// The action words a tool's schema publishes: its `action` enum, or its `oneOf` keys.
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

    /// Whether a tool's actions declare their OWN arguments (`oneOf`) rather than
    /// sharing one set beside the action (`enum`): only the former sweeps per
    /// operation, the latter stating its arguments once for every action.
    fn declares_per_action_arguments(tool: &str) -> bool {
        tool_schema(tool)["properties"]["action"]
            .get("enum")
            .is_none()
    }

    /// The arguments one command line can fill: the tool's whole top-level set, or —
    /// where actions declare their own — the action's entry plus those beside it.
    fn argument_schema(operation: &Operation) -> serde_json::Map<String, serde_json::Value> {
        // The reader is no tool action: its arguments are the two inputs the reading takes.
        if operation.action == Some(DOCUMENT_READ) {
            return ["path", "file_name"]
                .into_iter()
                .map(|name| (name.to_string(), serde_json::json!({ "type": "string" })))
                .collect();
        }
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
        // A root argument an action opts into belongs to that action: keeping the root
        // copy would demand a form for it where the action refuses it.
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

    /// The words appearing more than once in `words` — a registry bug equality hides.
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

    /// Parse a standalone family's flag tail: no operation word, so it is all flags.
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

    /// Every operation names a tool its family fronts, and the words are derived and
    /// unique: two operations could derive the same word (an action kebab-cased, a
    /// tool's own name with the family word dropped) and the first would shadow the
    /// second in `run`'s lookup. A standalone family has no operation word at all.
    #[test]
    fn every_family_fronts_its_tools_under_unique_words() {
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
        }
    }

    /// The command line and the tools must agree on what exists, in both directions:
    /// an operation the tool cannot run, an action the command line cannot reach, a
    /// duplicate on either side. A tool taking no `action` has none in the registry
    /// either; the one operation no tool action backs is pinned separately
    /// ([`the_reader_is_the_one_operation_no_tool_backs`]).
    #[test]
    fn operations_match_the_tools_own_actions() {
        for family in FAMILIES {
            let Surface::Operations(operations) = family.surface else {
                continue;
            };
            for tool in family.tools {
                let ours: Vec<Option<&str>> = operations
                    .iter()
                    // The reader is no tool action: it is pinned in its own test.
                    .filter(|operation| {
                        !(operation.tool == "document" && operation.action == Some(DOCUMENT_READ))
                    })
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

    /// The reader is the one operation no tool action backs: exactly one operation
    /// names it, the document family's, and the tool publishes no such action.
    #[test]
    fn the_reader_is_the_one_operation_no_tool_backs() {
        let mut readers = Vec::new();
        for family in FAMILIES {
            let Surface::Operations(operations) = family.surface else {
                continue;
            };
            for operation in operations {
                if operation.action == Some(DOCUMENT_READ) {
                    readers.push((family, operation));
                }
            }
        }
        assert_eq!(
            readers.len(),
            1,
            "exactly one operation must name the reader's action"
        );
        let (family, operation) = readers[0];
        assert_eq!(
            family.word, "document",
            "the reader belongs to the document family"
        );
        assert_eq!(
            operation.tool, "document",
            "the reader is served by the document tool"
        );
        assert!(
            !tool_actions(&tool_schema("document")).contains(&DOCUMENT_READ),
            "the reader must not be an action of the document tool"
        );
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

    /// One command line of a family, as the tie sweeps it: name, tool, arguments, flags.
    struct CommandLine {
        called: String,
        tool: &'static str,
        arguments: serde_json::Map<String, serde_json::Value>,
        flags: &'static [Flag],
    }

    impl CommandLine {
        /// The schema a flag fills: one of the tool's own, or a generation setting.
        fn property(&self, arg: &str) -> Option<&serde_json::Value> {
            self.arguments.get(arg).or_else(|| {
                self.arguments
                    .get(self.generation()?)?
                    .get("items")?
                    .get("properties")?
                    .get(arg)
            })
        }

        /// The argument holding the picture command's one generation, if this is it.
        fn generation(&self) -> Option<&'static str> {
            (self.tool == "image_gen").then_some(super::GENERATIONS_ARG)
        }

        /// The one generation's declared settings, when the command line builds one.
        fn generation_settings(&self) -> Option<&serde_json::Map<String, serde_json::Value>> {
            self.arguments
                .get(self.generation()?)?
                .get("items")?
                .get("properties")?
                .as_object()
        }
    }

    /// The JSON types one flag kind can fill.
    fn fillable_types(kind: &FlagKind) -> &'static [&'static str] {
        match kind {
            // One string, so it fills a string argument — a path being one it resolves.
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

    /// Every property name a schema subtree declares, at any depth: the fields a
    /// structured value can carry, held against the tool's schema, not a second list.
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
    /// The property names a schema subtree declares whose own schema IS one of the
    /// tool's picture arguments: such a field names a FILE, so the field tie's other
    /// direction learns which fields the resolution owes. The picture's schema
    /// differs by target, so the whole family is read.
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

    /// The command lines of one family, as the ties sweep them: name, tool, arguments, flags.
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
            Surface::Standalone(flags) => {
                let tool = family.tools[0];
                vec![CommandLine {
                    called: family.word.to_string(),
                    tool,
                    arguments: tool_schema(tool)["properties"]
                        .as_object()
                        .expect("the tool's properties")
                        .clone(),
                    flags,
                }]
            }
        }
    }

    /// Every flag must be an argument the tool actually reads, and able to fill that
    /// argument's type — swept per OPERATION, so a flag is judged against its own
    /// operation. The ONE exception is the destination ([`FlagKind::Destination`]),
    /// the command line's own operand: the one flag not looked up among the tool's
    /// arguments, carried by every operation of a producing family and nothing else.
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
                    let property = line.property(flag.arg).unwrap_or_else(|| {
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
                    // A structured value's path-bearing fields are swept in BOTH
                    // directions: a field the tool does not have would be a resolution
                    // this frontend invented, and a field the tool builds as one that
                    // NAMES A FILE (its own `image` argument) must be declared here, so
                    // a field added later cannot silently stop being resolved.
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
                // The picture tool's flags fill the one generation its command builds.
                if let Some(declared) = line.generation_settings() {
                    let mut declared: Vec<&str> = declared.keys().map(String::as_str).collect();
                    let mut folded: Vec<&str> = line
                        .flags
                        .iter()
                        .filter(|flag| !matches!(flag.kind, FlagKind::Destination))
                        .map(|flag| flag.arg)
                        .filter(|arg| !line.arguments.contains_key(*arg))
                        .collect();
                    declared.sort_unstable();
                    folded.sort_unstable();
                    assert_eq!(
                        declared, folded,
                        "'mahbot {}' folds its per-generation flags into one generation",
                        line.called
                    );
                }
            }
            // Every operation of a producing family carries the destination, exactly one.
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
        // The reverse direction has to have found every one of them, or it would hold
        // vacuously: the document tool's picture stands in a `create` block and in three
        // placing edits — four fields the command line owes a resolution.
        assert!(
            file_fields >= 4,
            "the picture fields naming a file were not all found ({file_fields}) — the tie would \
             be vacuous"
        );
    }

    /// Every argument a tool reads — bar the action — must have a command-line form, so
    /// a new argument cannot arrive with no way to set it. Swept per OPERATION where
    /// actions declare their OWN arguments ([`declares_per_action_arguments`]): a flag
    /// one operation omits is a real hole a union would hide, while where the actions
    /// share one set the union is the right rule.
    #[test]
    fn every_tool_argument_has_a_command_line_form() {
        for family in FAMILIES {
            let lines = command_lines(family);
            for tool in family.tools {
                let strict = declares_per_action_arguments(tool);
                let mut reachable: Vec<&str> = Vec::new();
                let mut declared: Vec<&str> = Vec::new();
                for line in lines.iter().filter(|line| line.tool == *tool) {
                    // Its generation argument is this frontend's OWN shape, never a flag.
                    let line_reachable: Vec<&str> = line
                        .flags
                        .iter()
                        .filter(|flag| !matches!(flag.kind, FlagKind::Destination))
                        .map(|flag| flag.arg)
                        .chain(line.generation())
                        .collect();
                    let line_declared: Vec<&str> = line
                        .arguments
                        .keys()
                        .map(String::as_str)
                        .filter(|key| *key != "action")
                        .chain(line.generation())
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

    /// Every family declares a protocol value the listener serves — plain strings in two
    /// modules, so nothing else can see a family sent under an unknown value, whose every
    /// call would come back as an unknown store (see [`crate::db::ipc`]).
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
        // Only a family THIS build adds carries the current value: one the channel has
        // already served keeps the older value, so a product one build behind still
        // serves it. The list is the checkpoint for that decision.
        let current: Vec<&str> = FAMILIES
            .iter()
            .filter(|cli| cli.protocol == crate::db::ipc::CALL_STORE)
            .map(|cli| cli.word)
            .collect();
        assert_eq!(
            current,
            vec!["send", "document", "image", "video", "rasterize"],
            "the families this build adds"
        );
    }

    /// The instance serves a tool by NAME only when a family publishes it, so the
    /// owner's other tools are unreachable through the channel however privileged the
    /// socket is. The tools named below are the checkpoint for that.
    #[test]
    fn tools_no_family_publishes_are_not_served_by_name() {
        for tool in ["shell", "edit", "analyze", "say"] {
            assert!(
                !fronts_tool(tool),
                "'{tool}' is published by no family, so the channel must not serve it by name"
            );
        }
    }

    /// Both helps carry the whole contract — the product must be running, these
    /// commands act with the owner's authority, and the exit classes — so a caller
    /// entering through an operation's help learns no less. A standalone family has
    /// no operation help, so its family help carries the usage line and every flag row.
    #[test]
    fn every_help_states_the_running_product_and_the_exit_classes() {
        for family in FAMILIES {
            let helps = match family.surface {
                Surface::Operations(operations) => std::iter::once(family_help(family))
                    .chain(
                        operations
                            .iter()
                            .map(|operation| operation_help(family, operation)),
                    )
                    .collect(),
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
            // A producing family states the destination in BOTH helps, from the registry.
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

    /// Every flag kind the grammar admits, typed as the tool's schema declares it and
    /// carried verbatim. An ABSENT flag leaves no key; a value the tool will refuse
    /// travels as written, except a non-boolean switch, which is this command's misuse.
    #[test]
    #[expect(clippy::too_many_lines)] // one section per flag kind, in flag order
    fn parsing_carries_each_value_in_the_type_its_flag_declares() {
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

        // An empty value on a list flag is the EMPTY list; occurrences after it add to it,
        // and an absent flag is still an absence.
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
        let untouched = parsed("config", "update-custom-role", &["--name", "triager"]);
        assert!(untouched.args.get("tools").is_none(), "{}", untouched.args);

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

        // A non-boolean switch value is misuse, and the reason spells the form.
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
        let p = parsed(
            "document",
            "pdf-rotate",
            &["--path", "Cargo.toml", "--degrees", "sideways"],
        );
        assert_eq!(p.args["degrees"], serde_json::json!("sideways"));

        // A structured value travels as its JSON, and one that is no JSON as text.
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

    /// The values of the SECRET flags — and only those — travel as secrets, kept out
    /// of every record and echo. Which token is a value is decided by where it stands:
    /// after a flag the next token IS the value, a leading dash and all (a script must
    /// never read success for an operation that did not run), while where a FLAG could
    /// stand `-h` asks for the help.
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

        // A dash-led value is a value, and a dash-led secret a secret like any other.
        let p = parsed("config", "setup-telegram-bot", &["--token", "-h"]);
        assert_eq!(p.args["token"], "-h");
        assert_eq!(
            p.secrets,
            vec!["-h".to_string()],
            "a dash-led secret is a secret like any other"
        );
        let p = parsed("workspace", "board", &["--phase", "--help"]);
        assert_eq!(p.args["phase"], "--help");

        // Where a flag could stand, the help word asks for the help.
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

    /// Anything this frontend can judge by itself is misuse, its reason naming the flag.
    #[test]
    fn parsing_refuses_misuse_without_echoing_a_value() {
        /// One misuse case: the tokens, the flag its reason names, and the values never echoed.
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
        // Neither is a secret mistyped as a positional: the token is not echoed.
        let reason = parse("config", "setup-telegram-bot", &["SEKRIT"])
            .err()
            .expect("a stray token is misuse");
        assert!(
            !reason.contains("SEKRIT"),
            "a usage error must never carry a token: {reason}"
        );
    }

    /// A value too long for one command line may be given by naming a file: a `List`'s
    /// `@<file>` adds the entries it holds, a structured one's takes its JSON, a text
    /// one's is its text without one trailing line break. `./@…` escapes a value that
    /// begins with `@`, and a flag whose value names a path itself has no file form.
    /// A file it cannot use is misuse, the reason naming the FLAG, never the path; a
    /// secret given from a file is the FILE's content, which must reach no record or echo.
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
                // The entries a file holds are paths, each resolved where this command runs.
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

        // A file holding the wrong thing is misuse, the reason never echoing the path.
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

        let text = crate::util::test::test_root().join("cli-text-values");
        std::fs::create_dir_all(&text).expect("a place for the value files");

        let brief = text.join("brief.txt");
        std::fs::write(&brief, "a cat asleep on a keyboard\n").expect("write the brief");
        let from_file = format!("@{}", brief.display());
        let p = parsed("image", "gen", &["--prompt", &from_file]);
        assert_eq!(p.args["prompt"], "a cat asleep on a keyboard");

        // A value beginning with `@` is written `./@…`: no file is read, existing or not.
        std::fs::write(text.join("alice"), "not the prompt").expect("write a decoy file");
        let decoy = format!("./@{}", text.join("alice").display());
        let p = parsed("image", "gen", &["--prompt", &decoy]);
        assert_eq!(
            p.args["prompt"],
            format!("@{}", text.join("alice").display())
        );

        // A file that is not there is misuse, the reason naming the flag, never the path.
        let missing = format!("@{}", text.join("nothing-here.txt").display());
        let reason = parse("image", "gen", &["--prompt", &missing])
            .err()
            .expect("a file that cannot be read is misuse");
        assert!(reason.contains("--prompt"), "{reason}");
        assert!(
            !reason.contains("nothing-here"),
            "a reason must never carry a path value: {reason}"
        );

        // A secret given from a file is the FILE's content: that must reach no record,
        // echo or answer — never the spelling that named it.
        let key = text.join("provider-key.txt");
        std::fs::write(&key, "sk-secret-from-a-file\n").expect("write the key");
        let from_file = format!("@{}", key.display());
        let p = parsed(
            "config",
            "setup-web-search",
            &["--provider", "exa", "--key", &from_file],
        );
        assert_eq!(p.args["key"], "sk-secret-from-a-file");
        assert_eq!(p.secrets, vec!["sk-secret-from-a-file".to_string()]);

        // A flag whose value names a path itself takes no file form.
        let named = text.join("@report.pdf");
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

        // A missing file, last of all, is misuse naming the flag.
        let missing = format!("@{}", dir.join("nothing-here.json").display());
        let reason = parse("document", "pdf-merge", &["--paths", &missing])
            .err()
            .expect("a file that cannot be read is misuse");
        assert!(reason.contains("--paths"), "{reason}");
    }

    /// A flag that names a FILE carries the path the CALLER meant: `~` expanded against
    /// the home directory, a relative path resolved in this command's directory — the
    /// product's own working directory would name a different file. A value that names
    /// no file, an absolute path and a public URL are left as written.
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

        // `~` is the home directory: a bare `~` is one, a reference file must be a file.
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

    /// The picture family's flat flags become the one generation it asks for: the
    /// settings go inside the generation, the prompt beside the list, and a flag left
    /// out is left out of the generation rather than sent flat for the tool to ignore.
    #[test]
    fn the_picture_family_folds_its_flags_into_one_generation() {
        let mut args = serde_json::json!({
            "prompt": "retouch the defects",
            "images": ["reference.png"],
            "size": "1K",
        });
        super::fold_one_generation(&mut args);
        assert_eq!(args["prompt"], "retouch the defects");
        assert!(
            args.get("images").is_none() && args.get("size").is_none(),
            "no setting stays a flat argument the tool would ignore: {args}"
        );
        let generations = args[crate::cli::GENERATIONS_ARG]
            .as_array()
            .expect("the batch form");
        assert_eq!(generations.len(), 1, "one picture is a batch of one");
        assert_eq!(generations[0]["images"][0], "reference.png");
        assert_eq!(generations[0]["size"], "1K");
        assert!(
            generations[0].get("aspect_ratio").is_none(),
            "a flag left out is unset in the generation: {}",
            generations[0]
        );

        // A call carrying no setting is still one generation: all optional, none invented.
        let mut bare = serde_json::json!({ "prompt": "a cat" });
        super::fold_one_generation(&mut bare);
        assert_eq!(bare["generations"][0], serde_json::json!({}), "{bare}");
    }

    /// A path a STRUCTURED value carries resolves exactly like a `--path` value, since
    /// the product resolves an embedded image against its own directories — and a field
    /// that names no file is the caller's own text.
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

        // A field no file stands under is a plain value: a path-like cell is caller text.
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

        // A reference file inside a value that is missing is misuse, naming the flag.
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

    /// A file the caller names — a reference path, a path inside a structured value, a file
    /// a value is read from — must be a REGULAR file it can read: a pipe would hold the
    /// command forever (neither the parse nor the product has a timeout), an endless device
    /// would exhaust memory, and a value file far past any command line is a mistaken path.
    /// All of it is judged before anything is sent, naming the FLAG and never the path.
    #[test]
    #[expect(clippy::too_many_lines)] // one section per way a named file can be unusable
    fn the_file_the_caller_names_must_be_a_regular_file_it_can_read() {
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

        // A URL is not a path at all.
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

        // Something that exists but is no regular file is misuse: only metadata is read.
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

        // The same rule for a file a VALUE is read from, with a bound on its size since a
        // value travels as one argument.
        let values = crate::util::test::test_root().join("cli-value-file-guards");
        std::fs::create_dir_all(&values).expect("a place for the guard fixtures");
        let reason = parse(
            "image",
            "gen",
            &["--prompt", &format!("@{}", values.display())],
        )
        .err()
        .expect("a directory is misuse");
        assert!(reason.contains("--prompt"), "{reason}");
        assert!(reason.contains("regular file"), "{reason}");

        let huge = values.join("huge.txt");
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

                // A pipe a VALUE would be read from is refused the same way.
                let value_fifo = values.join("value.fifo");
                let _ = std::fs::remove_file(&value_fifo);
                let path = std::ffi::CString::new(value_fifo.to_string_lossy().as_bytes())
                    .expect("a path without a NUL");
                // SAFETY: `mkfifo` takes a NUL-terminated path and a mode.
                if unsafe { libc::mkfifo(path.as_ptr(), 0o600) } == 0 {
                    let reason = parse(
                        "image",
                        "gen",
                        &["--prompt", &format!("@{}", value_fifo.display())],
                    )
                    .err()
                    .expect("a pipe is misuse");
                    assert!(reason.contains("regular file"), "{reason}");
                }
            }
            // An endless device would read until memory runs out.
            let reason = parse("image", "gen", &["--prompt", "@/dev/zero"])
                .err()
                .expect("a device is misuse");
            assert!(reason.contains("regular file"), "{reason}");
        }
    }

    /// The destination is resolved on this machine, in the directory this command was run
    /// in: a missing one, or one that is no directory, is misuse, and one the product
    /// keeps for itself is refused however it is spelled.
    #[test]
    fn the_destination_is_resolved_and_the_products_own_places_are_refused() {
        // The default destination is the directory this command was run in. A tree under a
        // governed root is refused like any other place — the rule is about the PLACE — so
        // the default and the same directory named explicitly agree.
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

        // The product's own temporary area, however it is spelled — a link into it is
        // refused too. The check is on the RESOLVED path, so a machine without the area
        // has nothing to refuse.
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

        // The product's own data directory, and everything under it. The rule is asserted
        // without the filesystem: a machine without the directory has nothing resolvable
        // inside it, not a usable destination.
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

    /// A family whose tool takes no `action`: the family word IS the operation.
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

        // The required flags are the tool's judgement: a bare family word sends no key.
        assert_eq!(parsed_standalone("send", &[]).args, serde_json::json!({}));
    }

    /// The exit classes are the browser-automation command's own numbering, class for
    /// class, and asserted against that table rather than restated, so the two cannot
    /// drift into a second table for a script to learn.
    #[test]
    fn exit_classes_match_the_browser_automation_command() {
        /// The class each of the other command's kinds belongs to, matched exhaustively.
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

    /// What one attempt becomes: where each part goes and the class a script reads.
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

        // The two the wire client should never hand over: not taken on, sentence to stderr.
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

    /// Misuse is exit 3, decided before the environment is touched: no instance here.
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
            // The destination is judged here too: a place that is not there is misuse.
            &[
                "image",
                "gen",
                "--prompt",
                "x",
                "--output-dir",
                "/no/such/place",
            ],
            // A family that produces nothing does not take the destination.
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
