//! Library crate for mahbot.
//!
//! The binary (`main.rs`) uses the same module tree. This crate root
//! provides the public API for both the dashboard and background agent dispatch.

#![warn(clippy::pedantic)]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown
)]

pub mod agent;
pub mod audio;
pub mod bench_openrouter;
pub(crate) mod boot;
pub mod channels;
pub mod config;
pub mod config_db;
pub(crate) mod consensus;
pub mod db;
pub(crate) mod embedder;
pub mod git;
pub mod gui;
pub mod jobs;
pub mod logs;
pub(crate) mod onnx;
pub mod pipeline;
pub(crate) mod prompt;
pub mod providers;
pub(crate) mod research_cancel;
pub mod research_cleanup;
pub(crate) mod retry;
pub mod search_engine;
pub mod self_update;
pub mod session;
pub mod shutdown;
pub(crate) mod stats;
pub mod temp;
pub mod tools;
pub mod users;
pub mod util;
pub(crate) mod vector;
pub mod workspace;

/// Hidden grep-engine subcommand entry (dispatched from `main()` before the
/// instance lock; the shell tool's read-only grep interception rewrites
/// served invocations to this subcommand of the current binary).
#[cfg(unix)]
pub use tools::shell::grep_engine::run_engine as run_grep_engine;

/// Test/subprocess-harness rewrite entry with an explicit home (fixture `~`
/// operands); production single-file gate. Only compiled for the e2e harness.
#[cfg(all(unix, feature = "grep-engine-e2e"))]
#[doc(hidden)]
pub use tools::shell::grep_engine::try_serve_command_for_test as grep_engine_rewrite_for_test;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};
use strum::IntoEnumIterator;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::session::Session;
use crate::util::UnwrapPoison;

// Diagnostics commands — discovered dev tooling

/// Discovered diagnostic commands for a workspace.
///
/// All fields are optional — `None` means no such tooling was detected.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct DiagnosticsCommands {
    /// Auto-formatter command (e.g., `cargo fmt`)
    pub format: Option<String>,
    /// Format check without changes (e.g., `cargo fmt -- --check`)
    pub format_check: Option<String>,
    /// Linter, idempotent (e.g., `cargo clippy -- -D warnings`)
    pub lint: Option<String>,
    /// Auto-fix lint issues (e.g., `cargo clippy --fix --allow-dirty`)
    pub lint_fix: Option<String>,
    /// Type checking without full compilation (e.g., `cargo check`)
    pub type_check: Option<String>,
    /// Full build command (e.g., `cargo build`)
    pub build: Option<String>,
    /// Run unit tests, fast (e.g., `cargo test`)
    pub unit_test: Option<String>,
}

impl DiagnosticsCommands {
    /// Number of command categories (must match the array length in [`Self::commands`]).
    pub const COMMAND_COUNT: usize = 7;

    /// Label of the `unit-test` slot. Shared with the diagnostics runner so
    /// its extended-timeout wiring can't silently drift from the label list.
    pub(crate) const UNIT_TEST_LABEL: &str = "unit-test";

    /// Static labels for the 7 command categories, matching the order in [`Self::commands`].
    pub const COMMAND_LABELS: [&str; Self::COMMAND_COUNT] = [
        "format",
        "format-check",
        "lint-fix",
        "lint",
        "type-check",
        "build",
        Self::UNIT_TEST_LABEL,
    ];

    /// Ordered iterator of (label, command) pairs. `None` entries are undiscovered — skipped.
    #[must_use]
    pub fn commands(&self) -> [(&'static str, Option<&str>); Self::COMMAND_COUNT] {
        [
            ("format", self.format.as_deref()),
            ("format-check", self.format_check.as_deref()),
            ("lint-fix", self.lint_fix.as_deref()),
            ("lint", self.lint.as_deref()),
            ("type-check", self.type_check.as_deref()),
            ("build", self.build.as_deref()),
            (Self::UNIT_TEST_LABEL, self.unit_test.as_deref()),
        ]
    }

    /// Build `DiagnosticsCommands` from an array of edit buffers (same ordering
    /// as [`Self::commands`]). Empty strings become `None` — skip during execution.
    #[must_use]
    pub fn from_buffers(buffers: &[String; Self::COMMAND_COUNT]) -> Self {
        Self {
            format: crate::util::none_if_empty(&buffers[0]),
            format_check: crate::util::none_if_empty(&buffers[1]),
            lint_fix: crate::util::none_if_empty(&buffers[2]),
            lint: crate::util::none_if_empty(&buffers[3]),
            type_check: crate::util::none_if_empty(&buffers[4]),
            build: crate::util::none_if_empty(&buffers[5]),
            unit_test: crate::util::none_if_empty(&buffers[6]),
        }
    }

    /// True if no diagnostics commands were discovered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands().iter().all(|(_, cmd)| cmd.is_none())
    }
}

// Core type: Workspace

/// Lifecycle status of a workspace.
///
/// Mirrors the `status` TEXT column in the workspaces database table.
/// Conversion to/from strings happens at the DB boundary only.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Default,
    strum::Display,
    strum::AsRefStr,
    strum::EnumString,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum WorkspaceStatus {
    /// Workspace registered but discovery has not started.
    #[default]
    Pending,
    /// Discovery agent is running.
    Analyzing,
    /// Discovery completed successfully — workspace is operational.
    Ready,
    /// Discovery failed — workspace cannot be used.
    Failed,
}

/// A persisted workspace entry.
#[derive(Debug, Clone, Serialize)]
pub struct Workspace {
    pub name: String,
    pub path: String,
    pub status: WorkspaceStatus,
    /// Whether the maintainer agent is enabled for this workspace.
    ///
    /// `true`  — the maintainer loop processes this workspace on each cycle.
    /// `false` — the maintainer skips this workspace entirely (default).
    ///
    /// Unlike [`Self::paused`] (which blocks automatic analysis and development
    /// claims but allows all other pipeline phases to run normally),
    /// `maintenance_enabled` specifically controls only the maintainer loop.
    /// A paused workspace can still be maintained if
    /// `maintenance_enabled` is `true`, and vice versa.
    ///
    /// Persisted in the `workspaces` table with a `DEFAULT 0` schema default.
    /// Toggled via the settings panel in the GUI.
    pub maintenance_enabled: bool,
    /// Whether automatic claim dispatch is paused for this workspace (blocks
    /// backlog → analysis and ready_for_development → in_development). Later
    /// pipeline phases (review, QA, diagnostics, sanitation, maintainer) run
    /// normally, and tickets already past analysis keep progressing.
    ///
    /// Automatically set to `true` on technical/agent failures (dispatch panic,
    /// agent run failure, all verifiers failing, user cancelling an in-flight
    /// run) so queued development tickets aren't claimed and don't cascade.
    /// Lifted via the normal unpause path (GUI toggle or rediscovery; the
    /// nightly loop skips paused workspaces). A discovery already in flight
    /// when the pause lands can still clear it on completion — residual race,
    /// see `finalize_discovery`.
    pub paused: bool,
    /// Minutes until the next maintainer run.
    /// Reset to 1 when a run creates tickets; doubled on empty runs
    /// (clamped to [5, 240] before doubling, hard‑capped at 240).
    /// Sequence after ticket creation: 1 → (if next empty) 10 → 20 → … → 240.
    pub maintainer_debounce_mins: i64,
    /// RFC 3339 timestamp of the last completed maintainer run.
    /// `None` means the workspace has never been maintained.
    pub maintainer_last_run_at: Option<String>,
    /// JSON blob of discovered dev commands (format, lint, build, test, etc.).
    /// `None` before the first diagnostics discovery run.
    pub diagnostics: Option<String>,
    /// Freeform user-curated context notes appended to every agent's
    /// system prompt. Empty by default. Max 4000 characters.
    /// Persisted in the `workspaces.notes` column.
    /// Survives `rediscover()` — never touched by automated analysis.
    pub notes: String,
    /// The git HEAD commit hash captured after the last successful discovery.
    /// `None` if the workspace is not a git repository or has no commits.
    /// Used by the nightly re-analysis check to detect new commits.
    pub last_analyzed_commit: Option<String>,
    /// Ephemeral per-run workspace (research run roots): never registered in
    /// the `workspaces` table, created on the fly for a single run's lifetime.
    /// Ephemeral workspaces get local handling in shared tool paths — e.g.
    /// the search tool downgrades an empty index to a warning instead of an
    /// error (a fresh per-run folder has no files yet).
    pub ephemeral: bool,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            name: String::default(),
            path: String::default(),
            status: WorkspaceStatus::Pending,
            maintenance_enabled: bool::default(),
            paused: bool::default(),
            maintainer_debounce_mins: 5,
            maintainer_last_run_at: Option::default(),
            diagnostics: Option::default(),
            notes: String::default(),
            last_analyzed_commit: Option::default(),
            ephemeral: bool::default(),
        }
    }
}

impl Workspace {
    /// Maximum allowed maintainer debounce in minutes (4 hours).
    ///
    /// Used as both the input-clamp upper bound before doubling and the
    /// absolute output cap after doubling in `advance_debounce`; also
    /// used as the cap in `should_skip_maintainer_debounce` and the
    /// GUI display. If these two semantics ever need to diverge, introduce
    /// a separate constant for the input-clamp bound.
    pub const MAX_MAINTAINER_DEBOUNCE_MINS: i64 = 240;

    /// Return the workspace path as a `&Path`.
    #[must_use]
    pub fn as_path(&self) -> &Path {
        Path::new(&self.path)
    }

    /// Create a minimal Workspace from a filesystem path.
    ///
    /// Used for test helpers, personal workspaces, and as a fallback when a
    /// workspace is not found in the database.
    ///
    /// The name is derived from the last path component (directory name).
    /// The stored path is canonicalized so that `is_path_safe_for_workspace`
    /// (which uses lexical `starts_with`) produces correct results even
    /// when the workspace base is behind a symlink (e.g. `/tmp` → `/private/tmp`
    /// on macOS).
    #[must_use]
    pub fn from_path(path: &Path) -> Self {
        // Canonicalize to match production workspace creation (see
        // canonicalize_workspace_path). Falls back to the raw path if the
        // directory does not exist yet (rare in tests, but harmless).
        let stored = crate::util::with_block_in_place(|| {
            std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
        });
        Self {
            name: last_path_component(&stored),
            path: stored.to_string_lossy().to_string(),
            ..Default::default()
        }
    }

    /// Return a human-readable display name derived from the workspace path.
    ///
    /// Uses the last path component (directory name).
    #[must_use]
    pub fn display_name(&self) -> String {
        last_path_component(self.as_path())
    }

    /// Create an ephemeral per-run workspace (research run roots). `name`
    /// becomes the search-engine registry key (e.g. the research `job_id`);
    /// the workspace is never registered in the `workspaces` table and its lifetime
    /// is bounded by the run it serves.
    #[must_use]
    pub fn ephemeral_run(name: &str, path: &Path) -> Self {
        let stored = crate::util::with_block_in_place(|| {
            std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
        });
        Self {
            name: name.to_string(),
            path: stored.to_string_lossy().to_string(),
            ephemeral: true,
            ..Default::default()
        }
    }
}

/// Extract the last component of a path as a string.
fn last_path_component(path: &Path) -> String {
    path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// URL-safe NanoID alphabet (A-Z, a-z, 0-9, -, _).
const NANOID_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-";

/// Generate a NanoID of the given length using the URL-safe alphabet.
#[must_use]
fn generate_nanoid(length: usize) -> String {
    (0..length)
        .map(|_| {
            let idx = (rand::random::<u8>() % 64) as usize;
            NANOID_ALPHABET[idx] as char
        })
        .collect()
}

/// Generate a unique identifier (10-char NanoID, ~60 bits entropy).
#[must_use]
pub fn generate_id() -> String {
    generate_nanoid(10)
}

/// Generate a short unique suffix (6-char NanoID, ~36 bits entropy).
/// Used to disambiguate retry cycles in parallel agent dispatches.
#[must_use]
pub(crate) fn generate_suffix() -> String {
    generate_nanoid(6)
}

// ── Command predicates ─────────────────────────────────

/// Represents a recognized Telegram bot text command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotCommand {
    /// `/start` — show the welcome message.
    Start,
    /// `/clear` — reset the user's current session.
    Clear,
    /// `/image_models` — show image model selection keyboard (Artist).
    ImageModels,
    /// `/video_models` — show video model selection keyboard (Artist).
    VideoModels,
    /// `/board` — list the active workspace's tickets (admin).
    Board,
    /// `/archive` — archive done & cancelled tickets (admin).
    Archive,
    /// `/pause` — pause the active workspace's pipeline (admin).
    Pause,
    /// `/unpause` — resume the active workspace's pipeline (admin).
    Unpause,
    /// `/maintenance on|off` — toggle the active workspace's maintainer (admin).
    Maintenance,
    /// `/maintenance_on` — enable the workspace maintainer (admin, menu form).
    MaintenanceOn,
    /// `/maintenance_off` — disable the workspace maintainer (admin, menu form).
    MaintenanceOff,
    /// `/role_name` — switch the user's active role (pool-gated).
    SwitchRole(Role),
}

/// Parse a Telegram bot text command from message content.
///
/// Returns `Some(BotCommand)` if the content is a recognised command,
/// case-insensitive. Returns `None` for all other content, including other
/// `/`-prefixed text which is routed to the agent pipeline.
#[must_use]
pub fn parse_bot_command(content: &str) -> Option<BotCommand> {
    let content = content.trim();
    let cmd = content.split_once(' ').map_or(content, |(cmd, _)| cmd);
    match cmd.to_ascii_lowercase().as_str() {
        "/start" => Some(BotCommand::Start),
        "/clear" => Some(BotCommand::Clear),
        "/image_models" => Some(BotCommand::ImageModels),
        "/video_models" => Some(BotCommand::VideoModels),
        "/board" => Some(BotCommand::Board),
        "/archive" => Some(BotCommand::Archive),
        "/pause" => Some(BotCommand::Pause),
        "/unpause" => Some(BotCommand::Unpause),
        "/maintenance" => Some(BotCommand::Maintenance),
        "/maintenance_on" => Some(BotCommand::MaintenanceOn),
        "/maintenance_off" => Some(BotCommand::MaintenanceOff),
        other => {
            let name = other.strip_prefix('/')?;
            Role::iter()
                .find(|r| r.as_str() == name)
                .map(BotCommand::SwitchRole)
        }
    }
}

// ── Channel trait + types ───────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ChannelMessage {
    /// The user's canonical name — stable identifier resolved from channel
    /// binding at auth time. Never derived from Telegram `@username` directly.
    pub user_name: String,
    pub reply_target: String,
    pub content: String,
    /// The channel this message originated from (e.g. `"telegram"`, `"gui"`).
    /// Set by each channel's `listen()` loop when constructing the message.
    pub channel: String,
    /// The workspace this message targets.
    pub workspace: String,
    /// Optimistic message ID set by the GUI sender for deduplication in the
    /// ChatEvent handler — `None` for non-GUI channels (Telegram, callbacks).
    pub optimistic_id: Option<String>,
    /// Callback query ID from Telegram inline keyboard interactions.
    /// Only set for callback queries (`__act__` prefix),
    /// used to acknowledge and dismiss the Telegram loading spinner.
    pub callback_query_id: Option<String>,
}

/// An outbound message to deliver on a channel.
///
/// ## Telegram-specific legacy
///
/// The `reply_markup` field carries Telegram inline_keyboard JSON. Non-Telegram
/// channels ignore it. Inline keyboard construction happens in `main.rs`
/// (e.g. `build_models_keyboard`) — other channels receive empty or harmless
/// payloads. The self-update path stores `reply_target` (via
/// `ChannelMessage::reply_target`) into `recipient` for admin Telegram
/// notifications during the update process — see the channel's
/// `resolve_recipient()` method which bridges the two fields.
#[derive(Debug, Clone)]
pub struct SendMessage {
    pub content: String,
    pub recipient: String,
    /// Optional inline keyboard markup to attach to the message.
    pub reply_markup: Option<serde_json::Value>,
}

#[async_trait]
pub trait Channel: Send + Sync {
    async fn send(&self, message: &SendMessage) -> anyhow::Result<()>;
    async fn listen(&self, tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()>;
    fn name(&self) -> &'static str;

    /// Return a reference to `self` as `&dyn Any` for downcasting.
    ///
    /// Used by the Telegram channel hot-reload path (`restart_telegram_listener`)
    /// to atomically swap the running channel without losing Telegram update
    /// continuity.
    fn as_any(&self) -> &dyn std::any::Any;

    async fn start_typing(&self, _recipient: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Resolve a recipient address for a known user on this channel.
    ///
    /// Given the user's canonical `user_name` and their stored `reply_target`,
    /// returns `Some(address)` if the user is reachable on this channel,
    /// or `None` if they are not.
    ///
    /// The default implementation returns `Some(reply_target.to_string())`,
    /// which is correct for the Telegram channel (reply_target is the chat_id).
    /// GUI channels override to return `Some(user_name.to_string())` since users
    /// are addressed by user name on that channel.
    fn resolve_recipient(&self, _user_name: &str, reply_target: &str) -> Option<String> {
        Some(reply_target.to_string())
    }
}

// ── Chat event broadcast (GUI live display) ─────────────────────

/// Direction of a chat message: from the user or from an agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatDirection {
    User,
    Agent,
    /// System-generated divider marker inserted when a user clears their
    /// session. Rendered as a horizontal rule with a label in the GUI.
    Divider,
}

/// A chat event broadcast to GUI subscribers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatEvent {
    /// A complete chat message (user or agent).
    Message {
        /// NanoID for deduplication.
        message_id: String,
        /// The user's canonical MahBot user name.
        user_name: String,
        /// Full message content.
        content: String,
        /// Whether this is from a user or an agent.
        direction: ChatDirection,
        /// ISO-8601 timestamp.
        timestamp: String,
        /// The channel this message was delivered on (e.g. "gui", "telegram", "voice").
        channel: String,
        /// The agent's role (e.g. "manager", "engineer"), if from an agent.
        agent_role: Option<String>,
        /// The workspace this message belongs to.
        workspace: String,
        /// Optional ID the GUI sender generated (via `ChannelMessage::optimistic_id`) so
        /// the Home page can replace its optimistic message with the confirmed one.
        optimistic_id: Option<String>,
    },
    /// Typing indicator event.
    Typing {
        /// The user's canonical MahBot user name.
        user_name: String,
        /// Whether typing started (true) or stopped (false).
        is_typing: bool,
        /// The workspace this typing indicator belongs to.
        workspace: String,
    },
}

/// Global broadcast channel for live chat events in the GUI.
/// Set during `init_message_pipeline`. Capacity 256 to prevent
/// `Lagged` errors in burst scenarios.
pub static CHAT_BROADCAST: OnceLock<broadcast::Sender<ChatEvent>> = OnceLock::new();

/// Global sender for the pipeline message channel.
/// Set during `init_message_pipeline`, used by `GuiChannel::listen()`
/// to forward GUI-originated messages into the shared pipeline.
pub static MESSAGE_TX: OnceLock<tokio::sync::mpsc::Sender<ChannelMessage>> = OnceLock::new();

/// Global sender for GUI-to-channel messages.
/// The Iced UI pushes `ChannelMessage` values here; `GuiChannel::listen()`
/// reads them from the paired receiver and forwards them into the pipeline.
pub static GUI_MESSAGE_TX: OnceLock<tokio::sync::mpsc::UnboundedSender<ChannelMessage>> =
    OnceLock::new();

// ── Channel registry ────────────────────────────────────────────

/// Registry of all active channels, keyed by [`Channel::name()`].
///
/// Replaces the old single-channel `OnceCell` with a multi-channel
/// `RwLock<HashMap>`. Channels register themselves during startup;
/// lookups are used to route outbound replies, Manager responses,
/// and typing indicators.
#[derive(Default)]
pub struct ChannelRegistry {
    channels: RwLock<HashMap<String, Arc<dyn Channel>>>,
}

impl ChannelRegistry {
    /// Register a channel under its name. Duplicate names are silently skipped
    /// with a warning logged.
    pub fn register(&self, channel: Arc<dyn Channel>) {
        let name = channel.name().to_string();
        let mut map = self.channels.write().unwrap_poison();
        if let std::collections::hash_map::Entry::Vacant(entry) = map.entry(name.clone()) {
            entry.insert(channel);
        } else {
            tracing::warn!(channel = %name, "Channel registry: duplicate name — skipping register");
        }
    }

    /// Look up a channel by name. Returns `None` if not found.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Channel>> {
        self.channels.read().unwrap_poison().get(name).cloned()
    }

    /// Replace a channel with a new instance, returning the previous one if any.
    /// Unlike [`Self::register`], this **always** inserts — it replaces an existing
    /// channel with the same name rather than silently skipping it.
    ///
    /// Used for hot-reloading channels (e.g. Telegram bot token change) without
    /// a full application restart.
    pub fn replace(&self, channel: Arc<dyn Channel>) -> Option<Arc<dyn Channel>> {
        let name = channel.name().to_string();
        self.channels.write().unwrap_poison().insert(name, channel)
    }

    /// Remove a channel by name, returning the removed channel if it existed.
    ///
    /// Used to tear down a channel that is no longer configured (e.g. Telegram
    /// token cleared in Settings).
    pub fn unregister(&self, name: &str) -> Option<Arc<dyn Channel>> {
        self.channels.write().unwrap_poison().remove(name)
    }

    /// Return a snapshot of all registered channels (name → Arc clone).
    pub fn list(&self) -> Vec<(String, Arc<dyn Channel>)> {
        self.channels
            .read()
            .unwrap_poison()
            .iter()
            .map(|(k, v)| (k.clone(), Arc::clone(v)))
            .collect()
    }
}

/// Global channel registry — channels register during startup.
pub static CHANNEL_REGISTRY: OnceLock<ChannelRegistry> = OnceLock::new();

/// Shorthand to get the global channel registry.
/// Panics if not initialized — call during startup after channels are set up.
#[must_use]
pub fn channel_registry() -> &'static ChannelRegistry {
    CHANNEL_REGISTRY
        .get()
        .expect("CHANNEL_REGISTRY not initialized — must be set during bootstrap")
}

// ── Role ────────────────────────────────────────────────────────

/// Typed role identifier for agents.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    strum::Display,
    strum::EnumIter,
    strum::IntoStaticStr,
)]
#[strum(serialize_all = "lowercase")]
pub enum Role {
    Manager,
    Engineer,
    Analyst,
    Coder,
    Qa,
    Reviewer,
    Discovery,
    Artist,
    Maintainer,
    Sanitation,
    Assistant,
}

// ── Agent ───────────────────────────────────────────────────────

/// A single tool call record, stored per-call in the logs database.
///
/// Each tool invocation produces one record with its full serialized
/// arguments, execution duration, and success/failure outcome.
/// Records are accumulated in-memory in the agent and flushed to
/// the logs store on session finalization.
#[derive(Debug, Clone)]
pub(crate) struct ToolCallRecord {
    /// The tool's name.
    pub tool_name: String,
    /// Serialized arguments as JSON string (credentials scrubbed).
    pub arguments: String,
    /// Execution duration in milliseconds.
    pub duration_ms: i64,
    /// Whether the tool call succeeded.
    pub success: bool,
    /// Error message on failure (`None` on success).
    pub error_message: Option<String>,
}

/// Running context for an agent turn.
pub struct Agent {
    /// Session persistence key; also serves as the registry agent ID.
    /// Direct-chat agents use `{user}_{ws}_{role}` as the key (stable across messages,
    /// channel-agnostic per user+workspace+role).
    /// Sub-agents and ticket handlers use a fresh NanoID per invocation.
    #[expect(clippy::struct_field_names)]
    pub(crate) agent_id: String,
    /// The agent's role (Manager, Engineer, Analyst, etc.).
    role: Role,
    /// Agent owns its session — all session methods take the key from `self.agent_id`.
    pub(crate) session: Session,
    /// Execution workspace for this agent.
    workspace: Arc<crate::Workspace>,
    /// Agent-owned tool set.
    tools: Vec<Box<dyn crate::Tool>>,
    /// Cached tool specs — computed once from `tools` at construction time.
    pub(crate) tool_specs: Vec<ToolSpec>,
    /// Cancellation token for cooperative mid-loop cancellation (e.g. /stop).
    cancel_token: CancellationToken,
    /// Whether a GENUINE user/operator stop was requested for this run.
    /// Set only by user-facing stop paths (e.g. GUI ticket cancel via
    /// `cancel_by_ticket_id_user`); code-driven internal cancellations
    /// (re-dispatch, register replacement, phase transition) set the
    /// generic `cancel_token` but NEVER this flag. Used to distinguish
    /// a real user stop from an internal cancellation in the pipeline
    /// failure classifier.
    user_stop: Arc<std::sync::atomic::AtomicBool>,
    /// Board ticket this agent is currently working on (set for board-dispatched agents).
    ticket: Option<crate::pipeline::board::Ticket>,
    /// Generation counter from the agent registry — used in [`Drop`] for
    /// safe deregistration. 0 means not registered (e.g. test agents).
    generation: u64,
    /// Per-call tool stats accumulated during this agent's work loop.
    /// Flushed to the logs store on [`Agent::finalize_session`]; silently
    /// lost if the agent is dropped without finalization.
    tool_stats: std::sync::Mutex<Vec<crate::ToolCallRecord>>,
    /// The user who triggered this agent run — used by tools (e.g. AnalyzeTool)
    /// to route async sub-agent results back to the correct user.
    pub(crate) user_name: String,
    /// The channel origin (gui, telegram, voice) of the triggering message.
    pub(crate) channel: String,
    /// DIRECT PARENT INVOCATION grouping key for the Running Agents view
    /// (ticket / analyze round / research run). `None` for workspace singletons.
    /// Propagated to sub-agents spawned by tools via
    /// [`CURRENT_TOOL_PARENT_KEY`](crate::agent::CURRENT_TOOL_PARENT_KEY).
    pub(crate) parent_key: Option<crate::agent::registry::ParentKey>,
    /// Human-readable label of the DIRECT PARENT INVOCATION (ticket title /
    /// analyze question / research question) — shown on the Running Agents
    /// group header. Purely presentational. Propagated to sub-agents spawned
    /// by tools via
    /// [`CURRENT_TOOL_PARENT_LABEL`](crate::agent::CURRENT_TOOL_PARENT_LABEL).
    pub(crate) parent_label: Option<String>,
    /// Optional receiver for mid-work messages (e.g., ticket comments).
    /// When set, the `llm_loop` drains this channel before each LLM call
    /// and injects received messages into the session history.
    pub(crate) incoming_rx:
        Option<tokio::sync::mpsc::UnboundedReceiver<crate::agent::message_router::AgentJob>>,
    /// Round-fixed timestamp for the first user message (one value per round
    /// for byte-identical task messages across parallel members).
    pub(crate) round_ts: Option<String>,
    /// Leader-stagger signal: fired after the agent's first LLM call
    /// completes (success or failure, full retry budget) or when the leader
    /// bails at the phase gate before any LLM call; `None` for non-leader
    /// members.
    pub(crate) first_call_notify: Option<std::sync::Arc<tokio::sync::Notify>>,
    /// Raw failure detail from the last run — set by [`crate::agent::run_agent`]
    /// on error, left `None` on success or cancellation (callers classify
    /// cancellation via the cancel tokens).
    pub(crate) failure: Option<String>,
    /// Typed provider failure classification from the last run — set by
    /// [`crate::agent::run_agent`] alongside [`Self::failure`] when the error
    /// chain carries a [`crate::retry::RetryExhausted`]; `None` for runtime
    /// errors, cancellations, and successes. Consumed by workspace discovery
    /// to distinguish provider-class failures (workspace returns to Pending)
    /// from genuine failures (workspace goes Failed).
    pub(crate) failure_class: Option<crate::retry::FailureClass>,
    /// Agent-scoped registry of background shell sessions (Full shell roles
    /// only). Live sessions are force-killed on agent teardown ([`Drop`]) —
    /// see [`crate::tools::shell::BackgroundSessions`]. Reachable from the
    /// per-call tool context via
    /// [`crate::agent::CURRENT_TOOL_BACKGROUND_SESSIONS`].
    background_sessions: std::sync::Arc<crate::tools::shell::BackgroundSessions>,
}

// ── Verdict type ─────────────────────────────────────────────────

/// Result of a single review or QA verification pass.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Verdict {
    /// Quality score from 0 (worst) to 10 (best).
    #[serde(deserialize_with = "de_verdict_score")]
    pub score: u8,
    /// List of specific issues detected in the response.
    #[serde(rename = "issues")]
    pub issues_detected: Vec<String>,
}

/// Deserialize a verdict score, self-healing common model emissions:
/// - [0,10] values pass through; fractional values floor (8.5 → 8).
/// - [11,100] divides by 10 then floors (85 → 8, 11 → 1, 100 → 10).
/// - Integers 101–255 pass through so downstream validation keeps
///   classifying them as out-of-range (failure-class stability); the
///   float spellings of that band (101.0, 255.0) fail closed.
/// - Everything else (non-numeric, negative, the (10,11) gap, fractions
///   outside the bands) fails closed — range checks precede any cast.
#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn de_verdict_score<'de, D>(deserializer: D) -> Result<u8, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let value = serde_json::Value::deserialize(deserializer)?;
    let Some(n) = value.as_number() else {
        return Err(D::Error::custom("verdict score must be a number"));
    };
    let f = n
        .as_f64()
        .ok_or_else(|| D::Error::custom("verdict score out of accepted ranges"))?;
    if (0.0..=10.0).contains(&f) {
        Ok(f.floor() as u8)
    } else if (11.0..=100.0).contains(&f) {
        Ok((f / 10.0).floor() as u8)
    } else if (101.0..=255.0).contains(&f) && n.is_i64() {
        // Out-of-range integers pass through so downstream validation
        // keeps their out-of-range classification (failure-class stability).
        // Floats in this band (e.g. 255.0) fail closed.
        Ok(f as u8)
    } else {
        Err(D::Error::custom("verdict score out of accepted ranges"))
    }
}

#[cfg(test)]
mod verdict_score_tests {
    use super::*;

    #[test]
    fn verdict_score_accepts_native_and_percent_bands() {
        struct Case {
            json: &'static str,
            expected: Option<u8>,
        }
        let cases = [
            // Native band [0,10]: integers unchanged, fractions floor.
            Case {
                json: r#"{"score": 8, "issues": []}"#,
                expected: Some(8),
            },
            Case {
                json: r#"{"score": 10, "issues": []}"#,
                expected: Some(10),
            },
            Case {
                json: r#"{"score": 8.5, "issues": []}"#,
                expected: Some(8),
            },
            Case {
                json: r#"{"score": 10.0, "issues": []}"#,
                expected: Some(10),
            },
            // Percent band [11,100]: divide by 10, then floor.
            Case {
                json: r#"{"score": 85, "issues": []}"#,
                expected: Some(8),
            },
            Case {
                json: r#"{"score": 11, "issues": []}"#,
                expected: Some(1),
            },
            Case {
                json: r#"{"score": 100, "issues": []}"#,
                expected: Some(10),
            },
            Case {
                json: r#"{"score": 85.5, "issues": []}"#,
                expected: Some(8),
            },
            // Integers 101–255 pass through for downstream out-of-range
            // classification (failure-class stability).
            Case {
                json: r#"{"score": 101, "issues": []}"#,
                expected: Some(101),
            },
            Case {
                json: r#"{"score": 255, "issues": []}"#,
                expected: Some(255),
            },
            // ...but the float spellings of that band fail closed, pinning
            // the lexical-form distinction (255 accepted vs 255.0 rejected).
            Case {
                json: r#"{"score": 101.0, "issues": []}"#,
                expected: None,
            },
            Case {
                json: r#"{"score": 255.0, "issues": []}"#,
                expected: None,
            },
            // (10,11) gap and everything outside the bands fail closed.
            Case {
                json: r#"{"score": 10.5, "issues": []}"#,
                expected: None,
            },
            Case {
                json: r#"{"score": 10.9, "issues": []}"#,
                expected: None,
            },
            Case {
                json: r#"{"score": -1, "issues": []}"#,
                expected: None,
            },
            Case {
                json: r#"{"score": -1.0, "issues": []}"#,
                expected: None,
            },
            Case {
                json: r#"{"score": 100.5, "issues": []}"#,
                expected: None,
            },
            Case {
                json: r#"{"score": 300, "issues": []}"#,
                expected: None,
            },
            Case {
                json: r#"{"score": "high", "issues": []}"#,
                expected: None,
            },
        ];
        for case in &cases {
            let parsed = serde_json::from_str::<Verdict>(case.json);
            match case.expected {
                Some(score) => assert_eq!(
                    parsed.expect("must deserialize").score,
                    score,
                    "case: {}",
                    case.json
                ),
                None => assert!(parsed.is_err(), "must fail closed: {}", case.json),
            }
        }
    }
}

/// Result of a sanitation agent's file inspection.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct SanitationVerdict {
    /// Whether the ticket passes sanitation (`true` = clean, `false` = garbage detected).
    pub pass: bool,
    /// List of garbage/unwanted files found (empty when `pass` is `true`).
    #[serde(default)]
    pub garbage_files: Vec<String>,
    /// Rationale for the decision.
    pub rationale: String,
}

/// Concise structured summary of an engineer's completed work, extracted from
/// the engineer's session for the ticket comment.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct EngineerSummary {
    /// Concise bullet points of what was implemented / fixed / executed.
    #[serde(default)]
    pub items: Vec<String>,
    /// One short paragraph summarizing the completed work (optional — legacy
    /// extractions and models emitting null leave it absent).
    pub summary: Option<String>,
}

// ── Tool trait + types ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[async_trait]
pub(crate) trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    /// Human-readable description of this tool.
    ///
    /// The default implementation loads the description from
    /// `src/prompt/tool/{name}.md` in the embedded prompt assets.
    fn description(&self) -> String {
        crate::prompt::load_prompt(&format!("tool/{}.md", self.name()))
    }
    fn parameters_schema(&self) -> serde_json::Value;
    async fn execute(
        &self,
        ws: &crate::Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String>;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name().to_string(),
            description: self.description(),
            parameters: self.parameters_schema(),
        }
    }

    /// Whether this tool's output should be scrubbed for credentials.
    ///
    /// The call chain is (see `scrub_tool_output` in `tools/mod.rs`):
    ///
    /// ```text
    /// agent::execute_tool
    ///   └─ scrub_tool_output(tool, args, output)
    ///        └─ tool.should_scrub_output(args)
    ///             └─ (if true) scrub_credentials(output)
    ///             └─ (if false) output as-is
    /// ```
    ///
    /// There are three distinct policy patterns across the codebase:
    ///
    /// 1. **Scrub-all** (trait default: `true`) — web_search, browser, edit,
    ///    analyze, ticket, media-gen tools, and most others. The raw output may
    ///    contain credentials, so it is always scrubbed before the LLM sees it.
    ///
    /// 2. **Skip scrubbing entirely** (`false`) — shell and search tools.
    ///    - The shell tool's internal `apply_profile_pipeline` already scrubs
    ///      stdout and stderr once at pipeline entry; returning `false` avoids
    ///      double-scrubbing. Implementers of tools that scrub internally **must**
    ///      return `false` from this method to prevent redundant scrubbing.
    ///    - The search tool returns source code content where credential patterns
    ///      are harmless and should be shown accurately to the model.
    ///
    /// 3. **Context-sensitive** — read tool. Scrubs only when the file path
    ///    matches `is_sensitive_file_path` (config, credential, or key files).
    ///    Non-sensitive files (regular source code) are returned as-is.
    ///
    /// **If your tool performs internal credential scrubbing**, override this
    /// method to return `false` so the agent-level pass does not double-scrub.
    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        true
    }

    /// Whether this tool has side effects that would conflict with parallel
    /// execution of other tools.
    ///
    /// Returns `true` by default (conservative — assume mutating). Override to
    /// `false` for read-only tools (read, search, web_search, etc.) so they can
    /// be grouped for parallel execution within a single LLM turn.
    fn side_effects(&self) -> bool {
        true
    }

    /// Whether the tool should be advertised to the model right now.
    ///
    /// Returns `true` by default. Tools backed by an external daemon/service
    /// override this to hide themselves while that service is down, so the
    /// model doesn't burn calls against a dead backend (e.g. browser when the
    /// chrome-use daemon is unreachable).
    ///
    /// The result is read once when an agent is constructed and kept for the
    /// agent's lifetime: an agent created during an outage permanently lacks
    /// the tool even after recovery, and a long-lived agent keeps advertising
    /// it after the backend dies (its runtime fail-fast then handles that).
    /// This is deliberate — agents are short-lived per run, and re-checking
    /// per turn would add latency and complexity.
    fn is_advertised(&self) -> bool {
        true
    }

    /// Return the media marker prefix for this tool, if it generates media files.
    ///
    /// Returns the prefix string used to wrap a generated file path in the agent's
    /// response content, e.g. `"[IMAGE:"` or `"[VIDEO:"`. The full marker
    /// `"{prefix}path]"` is constructed by appending the path and a closing `]`.
    ///
    /// Tools that produce media (images, videos, audio) should override this to
    /// return their marker prefix. Non-media tools should leave the default `None`.
    /// The marker format is `[TYPE:path]` where `TYPE` is the uppercase media type.
    fn media_marker(&self) -> Option<&'static str> {
        None
    }

    /// Format the media output marker `[TYPE:path]` for a generated file.
    /// Single source of the marker shape parsed by [`crate::util::MEDIA_MARKER_RE`].
    fn format_media_result(&self, output_path: &Path) -> String {
        let marker_prefix = self
            .media_marker()
            .expect("media tool must define a media marker");
        format!("{marker_prefix}{}]", output_path.to_string_lossy())
    }

    /// Whether tool output must bypass the default 5 KB truncation
    /// (e.g. ticket details, search results, media markers the LLM
    /// needs in full). Default: false — output is truncated.
    fn preserve_full_output(&self) -> bool {
        false
    }

    /// Format tool output for LLM consumption.
    ///
    /// Called when tool results are embedded into the conversation history
    /// (both in native tool-call mode and degraded text mode). The default
    /// implementation uses head+tail truncation via [`crate::util::truncate_tool_output`]
    /// (hardcoded 5 KB limit); override to produce a smarter summary (e.g. trim
    /// repetitive CLI output or extract key facts from search results).
    fn format_output(&self, output: &str) -> String {
        if self.preserve_full_output() {
            output.to_string()
        } else {
            crate::util::truncate_tool_output(output)
        }
    }

    /// Build an optional per-call image payload to be injected as a synthetic
    /// User-role message after this tool's result block, so a vision-capable
    /// model can see an on-disk image as a native image part.
    ///
    /// Default `None` — most tools produce no image. The read tool overrides
    /// this to re-encode a raster (PNG/JPEG/WebP) file into a bounded JPEG
    /// data-URI. It is the single decoder for native images — `execute` only
    /// cheap-sniffs magic bytes, so the decode happens here, once, guarded by
    /// a robust file-magic check rather than the annotation wording. Safety:
    /// the path must obey the same workspace-boundary validation as
    /// [`Tool::execute`] — an arbitrary `[IMAGE:/path]` reference embedded in
    /// model text is never resolved here.
    async fn image_payload(
        &self,
        _ws: &crate::Workspace,
        _args: &serde_json::Value,
    ) -> Option<crate::tools::ImagePayload> {
        None
    }
}

// ── Chat role enum ─────────────────────────────────────────────

/// Typed role for a [`ChatMessage`].
///
/// Replaces the previous `String`-based role to prevent typos and
/// make the chat-role API self-documenting.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, strum::Display, strum::EnumString,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "lowercase")]
pub(crate) enum ChatRole {
    /// System prompt message.
    System,
    /// User (human) message.
    User,
    /// Assistant (LLM) message.
    Assistant,
    /// Tool result message.
    Tool,
}

// ── Provider trait + types ──────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn tool_result(tool_call_id: &str, content: &str) -> Self {
        let payload = ToolResultPayload {
            tool_call_id: tool_call_id.to_string(),
            content: content.to_string(),
        };
        Self {
            role: ChatRole::Tool,
            content: serde_json::to_string(&payload)
                .expect("ToolResultPayload is always serializable"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Shared payload struct for tool-result messages.
/// Ensures the writer (`ChatMessage::tool_result`) and reader
/// (`decode_native_history_message`) stay in sync on field names,
/// preventing silent field-name drift.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ToolResultPayload {
    pub tool_call_id: String,
    pub content: String,
}

#[expect(clippy::struct_field_names)]
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Reasoning {
    pub reasoning: Option<String>,
    pub reasoning_content: Option<String>,
    pub reasoning_details: Option<serde_json::Value>,
}

impl Reasoning {
    /// Returns `None` when all fields are empty/`None`.
    #[must_use]
    pub fn from_optional_parts(
        reasoning: Option<String>,
        reasoning_content: Option<String>,
        reasoning_details: Option<serde_json::Value>,
    ) -> Option<Self> {
        let details = reasoning_details.filter(|v| !v.is_null());
        let this = Self {
            reasoning,
            reasoning_content,
            reasoning_details: details,
        };
        (!this.is_empty()).then_some(this)
    }

    #[must_use]
    const fn is_empty(&self) -> bool {
        self.reasoning.is_none()
            && self.reasoning_content.is_none()
            && self.reasoning_details.is_none()
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ProviderUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    /// Prompt tokens not served from cache (prompt_tokens − cached when the
    /// provider reports only the hit side).
    pub cache_miss_tokens: Option<u64>,
    /// Billed cost from `usage.cost` — the invoice amount (OpenRouter-only).
    pub cost: Option<f64>,
    /// Raw provider `usage.cost_details` breakdown (reference only).
    pub cost_details: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ChatResponse {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<ProviderUsage>,
    pub reasoning: Option<Reasoning>,
    /// `choices[0].finish_reason` from the provider envelope (stop/length/tool_calls/…).
    pub finish_reason: Option<String>,
    /// Serving upstream provider — top-level OpenRouter response field
    /// (empirical, undocumented in the API reference); NULL when omitted.
    pub upstream_provider: Option<String>,
    /// Provider `system_fingerprint` (eviction/backend-switch attribution).
    pub system_fingerprint: Option<String>,
}

impl ChatResponse {
    #[must_use]
    pub fn text_or_empty(&self) -> &str {
        self.text.as_deref().unwrap_or("")
    }
}

/// Default max tokens for LLM calls (32K output generation limit — NOT the context window
/// size — this is the *generation limit* sent as `max_tokens` to the provider).
/// Used as the fallback when callers don't explicitly set `max_tokens`.
pub(crate) const DEFAULT_MAX_TOKENS: u32 = 32_000;

// ── Voice pipeline shared constants ─────────────────────────────────────────
/// Operation metadata for per-request LLM stats logging (`llm_requests` table
/// in logs.db). Attached by call sites with agent/ticket context; requests
/// without metadata (test doubles, ad-hoc calls) are not logged.
#[derive(Debug, Clone)]
pub(crate) struct ChatRequestMeta {
    /// Purpose tag: "agent" (agent loop incl. direct chat / discovery /
    /// sub-agents / maintainer), "extraction", "summarize", "consolidate",
    /// "research_wrap_up" (deadline wrap-up extraction), "media_transcription"
    /// (vision-model transcription of inbound videos / video tool results);
    /// "agent-continuation" / "summarize-continuation" tag the operation-level
    /// reasoning-only-stop recovery issued by
    /// [`crate::agent::Agent::recover_reasoning_only_stop`] — one row per
    /// continuation operation (success on resolution / failure on exhaustion;
    /// `retry_attempts` carries the attempt count).
    pub purpose: &'static str,
    pub agent_id: String,
    pub role: String,
    pub workspace: String,
    pub ticket_id: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<ToolSpec>>,
    pub model: String,
    /// Maximum tokens for the model to generate.
    /// When `Some(n)`, sent as `max_tokens` in the API request body.
    /// When `None`, the parameter is omitted (provider defaults apply).
    pub max_tokens: Option<u32>,
    /// Per-role reasoning effort (xhigh, high, medium, low, minimal).
    /// When set, sent as `reasoning_effort` in the API request body.
    /// When `None`, the parameter is omitted (model defaults apply).
    pub reasoning_effort: Option<String>,
    /// Provider routing: provider order list (comma-separated slugs).
    /// When `Some` and non-empty, sent as the provider routing block.
    /// When `None`, no provider routing is sent (OpenRouter defaults apply).
    pub provider_order: Option<String>,
    /// Optional operation metadata for per-request LLM stats logging.
    pub meta: Option<ChatRequestMeta>,
}

/// Validator callback for scoped structured extraction: rejects
/// a parsed value (e.g. verdict score ∉ [0,10]). A rejection is treated as a
/// parse failure inside the extraction retry loop (re-prompted, fail-closed).
/// Must be `Send + Sync` because extraction runs inside `join_all` futures.
pub(crate) type ExtractionValidator<T> = dyn Fn(&T) -> Result<(), String> + Send + Sync;

#[async_trait]
pub(crate) trait Provider: Send + Sync {
    /// Send a chat request using the model specified in the request.
    ///
    /// The default implementation delegates to [`Self::chat_scoped`] with the
    /// standard scoped timeouts (used by test doubles and any provider that
    /// only implements the scoped path).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Default kept for test doubles; production dispatches chat_scoped"
        )
    )]
    async fn chat(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
        let deadline = std::time::Instant::now() + crate::retry::DEFAULT_OPERATION_TIMEOUT;
        self.chat_scoped(request, crate::retry::DEFAULT_IDLE_TIMEOUT, deadline)
            .await
            .map_err(|e| e.inner)
    }

    /// Single-attempt chat for the outer retry paths.
    ///
    /// Used by the outer retry loops in [`crate::retry`]. Contract:
    ///
    /// - **No provider-internal retries** — this call makes exactly one HTTP
    ///   request; the outer loop is the single retry authority.
    /// - **Idle-timeout semantics** — `idle_timeout` resets while data flows
    ///   (a stalled connection mid-body is the truncation signature this
    ///   hardening targets).
    /// - **Per-attempt total = remaining wall budget** — implementations
    ///   should not outlive `deadline`.
    /// - On failure the returned [`ScopedCallError`] carries per-attempt
    ///   diagnostics (classification, error chain, finish_reason) for the
    ///   retry trail.
    ///
    /// Real providers must implement this method (or override [`Self::chat`]).
    async fn chat_scoped(
        &self,
        request: ChatRequest,
        idle_timeout: std::time::Duration,
        deadline: std::time::Instant,
    ) -> Result<ChatResponse, crate::providers::ScopedCallError>;

    async fn warmup(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// A skill is a user-defined or community-built capability.
/// Skills live in `<workspace>/skills/<name>/` (also
/// `<workspace>/.claude/skills/<name>/` and
/// `<workspace>/.agents/skills/<name>/`) and are defined via `SKILL.md`.
/// They provide instructions and context for the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Skill {
    pub name: String,
    pub description: String,
    pub location: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_commands_returns_all_7_in_order() {
        let diag = DiagnosticsCommands {
            format: Some("cargo fmt".into()),
            format_check: Some("cargo fmt -- --check".into()),
            lint: Some("cargo clippy -- -D warnings".into()),
            lint_fix: Some("cargo clippy --fix --allow-dirty".into()),
            type_check: Some("cargo check".into()),
            build: Some("cargo build".into()),
            unit_test: Some("cargo test".into()),
        };

        let cmds = diag.commands();
        assert_eq!(cmds.len(), 7);
        assert_eq!(cmds[0].0, "format");
        assert_eq!(cmds[1].0, "format-check");
        assert_eq!(cmds[2].0, "lint-fix");
        assert_eq!(cmds[3].0, "lint");
        assert_eq!(cmds[4].0, "type-check");
        assert_eq!(cmds[5].0, "build");
        assert_eq!(cmds[6].0, "unit-test");

        assert_eq!(cmds[0].1, Some("cargo fmt"));
        assert_eq!(cmds[6].1, Some("cargo test"));
    }

    #[test]
    fn diagnostics_is_empty() {
        let empty = DiagnosticsCommands::default();
        assert!(empty.is_empty());

        let partial = DiagnosticsCommands {
            format: Some("cargo fmt".into()),
            ..Default::default()
        };
        assert!(!partial.is_empty());
    }

    #[test]
    fn from_buffers_empty_strings_become_none() {
        let buffers = [const { String::new() }; DiagnosticsCommands::COMMAND_COUNT];
        let cmds = DiagnosticsCommands::from_buffers(&buffers);
        assert!(cmds.is_empty());
        assert!(cmds.format.is_none());
        assert!(cmds.unit_test.is_none());
    }

    #[test]
    fn commands_and_from_buffers_are_consistent() {
        // Verify that commands() and from_buffers() agree on field order.
        let original = DiagnosticsCommands {
            format: Some("fmt".into()),
            format_check: Some("fmt-check".into()),
            lint_fix: Some("lint-fix".into()),
            lint: Some("lint".into()),
            type_check: Some("type-check".into()),
            build: Some("build".into()),
            unit_test: Some("test".into()),
        };
        let cmds = original.commands();
        let buffers: [String; DiagnosticsCommands::COMMAND_COUNT] =
            std::array::from_fn(|i| cmds[i].1.unwrap_or("").to_string());
        let restored = DiagnosticsCommands::from_buffers(&buffers);
        assert_eq!(restored.format, original.format);
        assert_eq!(restored.format_check, original.format_check);
        assert_eq!(restored.lint_fix, original.lint_fix);
        assert_eq!(restored.lint, original.lint);
        assert_eq!(restored.type_check, original.type_check);
        assert_eq!(restored.build, original.build);
        assert_eq!(restored.unit_test, original.unit_test);
    }

    #[test]
    fn parse_bot_command_coverage() {
        use BotCommand::*;
        let cases: Vec<(&str, Option<BotCommand>)> = vec![
            // Positive: exact match
            ("/start", Some(Start)),
            // Positive: case-insensitive
            ("/STart", Some(Start)),
            ("/Start", Some(Start)),
            ("/START", Some(Start)),
            // Positive: with args/trailing space
            ("/start foo", Some(Start)),
            ("/start   ", Some(Start)),
            // Positive: leading/trailing whitespace
            ("  /start", Some(Start)),
            ("  /start  ", Some(Start)),
            ("  /start foo  ", Some(Start)),
            // /clear
            ("/clear", Some(Clear)),
            ("/CLEAR", Some(Clear)),
            ("/clear session", Some(Clear)),
            ("  /clear  ", Some(Clear)),
            // /image_models
            ("/image_models", Some(ImageModels)),
            ("/IMAGE_MODELS", Some(ImageModels)),
            ("/image_models foo", Some(ImageModels)),
            ("  /image_models  ", Some(ImageModels)),
            // /video_models
            ("/video_models", Some(VideoModels)),
            ("/Video_Models", Some(VideoModels)),
            ("/video_models foo", Some(VideoModels)),
            // admin commands
            ("/board", Some(Board)),
            ("/BOARD", Some(Board)),
            ("/board foo", Some(Board)),
            ("/archive", Some(Archive)),
            ("/archive foo", Some(Archive)),
            ("/pause", Some(Pause)),
            ("/pause foo", Some(Pause)),
            ("/unpause", Some(Unpause)),
            ("/unpause foo", Some(Unpause)),
            ("/maintenance", Some(Maintenance)),
            ("/maintenance on", Some(Maintenance)),
            ("/maintenance off", Some(Maintenance)),
            // Menu-form maintenance commands (distinct from the typed form)
            ("/maintenance_on", Some(MaintenanceOn)),
            ("/MAINTENANCE_ON", Some(MaintenanceOn)),
            ("/maintenance_on ", Some(MaintenanceOn)),
            ("/maintenance_off", Some(MaintenanceOff)),
            ("/Maintenance_Off", Some(MaintenanceOff)),
            ("/maintenance_off foo", Some(MaintenanceOff)),
            // Role-switch commands: each pool role is a direct command
            ("/manager", Some(SwitchRole(Role::Manager))),
            ("/engineer", Some(SwitchRole(Role::Engineer))),
            ("/analyst", Some(SwitchRole(Role::Analyst))),
            ("/coder", Some(SwitchRole(Role::Coder))),
            ("/qa", Some(SwitchRole(Role::Qa))),
            ("/reviewer", Some(SwitchRole(Role::Reviewer))),
            ("/discovery", Some(SwitchRole(Role::Discovery))),
            ("/artist", Some(SwitchRole(Role::Artist))),
            ("/maintainer", Some(SwitchRole(Role::Maintainer))),
            ("/sanitation", Some(SwitchRole(Role::Sanitation))),
            ("/assistant", Some(SwitchRole(Role::Assistant))),
            ("/ARTIST", Some(SwitchRole(Role::Artist))),
            ("/engineer foo", Some(SwitchRole(Role::Engineer))),
            ("  /coder  ", Some(SwitchRole(Role::Coder))),
            // Negative: partial /-prefix matches
            ("/", None),
            ("/s", None),
            ("/stard", None),
            ("/started", None),
            ("/cleared", None),
            ("/model", None),
            ("/models", None), // removed — falls through to the agent pipeline
            ("/image", None),
            ("/video", None),
            ("/boardx", None),
            ("/maintenance_onn", None),
            ("/maintenance_o", None),
            ("/engineerr", None),
            ("/managr", None),
            ("/artiste", None),
            ("/ reset", None),
            // Negative: missing slash or empty
            ("start", None),
            ("clear", None),
            ("models", None),
            ("", None),
            ("  ", None),
            ("not a command", None),
        ];
        for (input, expected) in cases {
            assert_eq!(parse_bot_command(input), expected, "input: {input:?}");
        }
    }
}
