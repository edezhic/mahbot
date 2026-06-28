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
pub mod board;
pub mod channels;
pub mod chat_history;
pub mod config;
pub mod config_db;
pub mod debug;
pub mod diff_parse;
pub mod embedder;
pub mod extraction;
pub mod gui;
pub mod logs;
pub mod maintainer;
pub mod management;
pub mod manager_queue;
pub mod prompt;
pub mod providers;
pub mod registry;
pub mod role;
pub mod search_engine;
pub mod self_update;
pub mod session;
pub mod shutdown;
pub mod skills;
pub mod stats;
pub mod ticket_buffer;
pub mod tools;
pub mod turso;
pub mod users;
pub mod util;
pub mod vector;
pub mod workspace;

use async_trait::async_trait;
use futures_util::stream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::session::Session;
use crate::util::UnwrapPoison;

// ---------------------------------------------------------------------------
// Diagnostics commands — discovered dev tooling
// ---------------------------------------------------------------------------

/// Discovered diagnostic commands for a workspace.
///
/// All fields are optional — `None` means no such tooling was detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticsCommands {
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
    /// Ordered iterator of (label, command) pairs. `None` entries are undiscovered — skipped.
    #[must_use]
    pub fn commands(&self) -> [(&'static str, Option<&str>); 7] {
        [
            ("format", self.format.as_deref()),
            ("format-check", self.format_check.as_deref()),
            ("lint-fix", self.lint_fix.as_deref()),
            ("lint", self.lint.as_deref()),
            ("type-check", self.type_check.as_deref()),
            ("build", self.build.as_deref()),
            ("unit-test", self.unit_test.as_deref()),
        ]
    }

    /// True if no diagnostics commands were discovered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands().iter().all(|(_, cmd)| cmd.is_none())
    }
}

// ---------------------------------------------------------------------------
// Core type: Workspace
// ---------------------------------------------------------------------------

/// A persisted workspace entry.
#[derive(Debug, Clone, Serialize)]
pub struct Workspace {
    pub name: String,
    pub path: String,
    pub status: String,
    pub created_at: String,
    pub updated_at: String,
    pub maintenance: bool,
    /// Whether development dispatch is paused for this workspace (blocks
    /// ready_for_development → in_development). All other pipeline phases
    /// (analysis, review, QA, diagnostics, sanitation, maintainer) run normally.
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
    /// RFC 3339 timestamp of the last completed diagnostics discovery run.
    /// `None` before the first diagnostics discovery run.
    pub diagnostics_updated_at: Option<String>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            name: String::default(),
            path: String::default(),
            status: String::default(),
            created_at: String::default(),
            updated_at: String::default(),
            maintenance: bool::default(),
            paused: bool::default(),
            maintainer_debounce_mins: 5,
            maintainer_last_run_at: Option::default(),
            diagnostics: Option::default(),
            diagnostics_updated_at: Option::default(),
        }
    }
}

impl Workspace {
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
        let stored = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        Self {
            name: dir_name(&stored),
            path: stored.to_string_lossy().to_string(),
            maintainer_debounce_mins: 5,
            ..Default::default()
        }
    }

    /// Return a human-readable display name derived from the workspace path.
    ///
    /// Uses the last path component (directory name).
    #[must_use]
    pub fn display_name(&self) -> String {
        dir_name(self.as_path())
    }
}

/// Extract the last directory component of a path as a string.
fn dir_name(path: &Path) -> String {
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
pub fn generate_suffix() -> String {
    generate_nanoid(6)
}

// ── Command predicates ─────────────────────────────────

/// Check whether `content` is a `/start` command.
///
/// Command names are case-insensitive. `/start` is the only recognized
/// dispatch-level command. All other `/`-prefixed text is treated as a
/// normal message (routed to the agent).
#[must_use]
pub fn is_start_command(content: &str) -> bool {
    content
        .trim()
        .strip_prefix('/')
        .map(|cmd| cmd.split_once(' ').map_or(cmd, |(c, _)| c))
        .is_some_and(|cmd| cmd.eq_ignore_ascii_case("start"))
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
    pub source_channel: String,
    /// The workspace this message targets.
    pub workspace: String,
    /// Optimistic message ID set by the GUI sender for deduplication in the
    /// ChatEvent handler — `None` for non-GUI channels (Telegram, callbacks).
    pub message_id: Option<String>,
    /// Callback query ID from Telegram inline keyboard interactions.
    /// Only set for callback queries (`__opt__` or `__act__` prefixes),
    /// used to acknowledge and dismiss the Telegram loading spinner.
    pub callback_query_id: Option<String>,
}

/// An outbound message to deliver on a channel.
///
/// ## Telegram-specific legacy
///
/// The `reply_markup` field carries Telegram inline_keyboard JSON. Non-Telegram
/// channels ignore it. Inline keyboard construction happens in `main.rs`
/// (e.g. `build_start_keyboard`) — other channels receive empty or harmless
/// payloads. The self-update path stores `reply_target` for admin Telegram
/// notifications during the update process.
#[derive(Debug, Clone)]
pub struct SendMessage {
    pub content: String,
    pub recipient: String,
    /// Optional inline keyboard markup to attach to the message.
    pub reply_markup: Option<serde_json::Value>,
    /// The agent's role that produced this message, if sent by an agent.
    /// Used by chat history and GUI live display to show which role responded.
    pub agent_role: Option<String>,
    /// The workspace this message belongs to.
    pub workspace: String,
}

#[async_trait]
pub trait Channel: Send + Sync {
    async fn send(&self, message: &SendMessage) -> anyhow::Result<()>;
    async fn listen(&self, tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()>;
    fn name(&self) -> &str;

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
pub enum ChatDirection {
    User,
    Agent,
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
        /// The agent's role (e.g. "manager", "engineer"), if from an agent.
        agent_role: Option<String>,
        /// The workspace this message belongs to.
        workspace: String,
        /// Optional ID the GUI sender generated (via `ChannelMessage::message_id`) so
        /// the Home page can replace its optimistic message with the confirmed one.
        optimistic_id: Option<String>,
        /// Optional inline keyboard markup (Telegram `inline_keyboard` JSON).
        /// `#[serde(default)]` handles missing fields for backward compatibility
        /// during hot-reload or mixed-version scenarios.
        #[serde(default)]
        reply_markup: Option<serde_json::Value>,
    },
    /// Typing indicator event.
    Typing {
        /// The user's canonical MahBot user name.
        user_name: String,
        /// Whether typing started (true) or stopped (false).
        is_typing: bool,
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
    Debug, Clone, Copy, PartialEq, Eq, Hash, strum::Display, strum::EnumIter, strum::IntoStaticStr,
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
}

// ── Agent ───────────────────────────────────────────────────────

/// Accumulated usage stats for a single tool, collected in-memory during
/// an agent's work loop and flushed to `stats.db` on session finalization.
#[derive(Debug, Clone, Default)]
pub struct ToolUsage {
    pub call_count: u64,
    pub errors: Vec<String>,
}

/// Running context for an agent turn.
pub struct Agent {
    /// Session persistence key; also serves as the registry run ID.
    /// Direct-chat agents use `{channel}_{sender}_{role}_{ws_name}` as the key (stable across messages).
    /// Sub-agents and ticket handlers use a fresh NanoID per invocation.
    pub(crate) id: String,
    /// The agent's role (Manager, Engineer, Analyst, etc.).
    role: Role,
    /// Agent owns its session — all session methods take the key from `self.id`.
    pub(crate) session: Session,
    /// Execution workspace for this agent.
    workspace: Arc<crate::Workspace>,
    /// Agent-owned tool set.
    tools: Vec<Box<dyn crate::Tool>>,
    /// Cached tool specs — computed once from `tools` at construction time.
    pub(crate) tool_specs: Vec<ToolSpec>,
    /// Cancellation token for cooperative mid-loop cancellation (e.g. /stop).
    cancel_token: CancellationToken,
    /// Board ticket this agent is currently working on (set for board-dispatched agents).
    ticket: Option<crate::board::Ticket>,
    /// Generation counter from the agent registry — used in [`Drop`] for
    /// safe deregistration. 0 means not registered (e.g. test agents).
    generation: u64,
    /// Per-tool usage stats accumulated during this agent's work loop.
    /// Flushed to `stats.db` on [`Agent::finalize_session`]; silently
    /// lost if the agent is dropped without finalization.
    tool_stats: std::sync::Mutex<std::collections::HashMap<String, ToolUsage>>,
}

// ── Verdict type ─────────────────────────────────────────────────

/// Result of a single review or QA verification pass.
#[derive(Deserialize)]
pub struct Verdict {
    /// Quality score from 0 (worst) to 10 (best).
    pub score: u8,
    /// Optional constructive critique from the verifier.
    pub critique: Option<String>,
    /// List of specific issues detected in the response.
    #[serde(rename = "issues")]
    pub issues_detected: Vec<String>,
}

/// Result of a sanitation agent's file inspection.
#[derive(Deserialize)]
pub struct SanitationVerdict {
    /// Whether the ticket passes sanitation (`true` = clean, `false` = garbage detected).
    pub pass: bool,
    /// List of garbage/unwanted files found (empty when `pass` is `true`).
    #[serde(default)]
    pub garbage_files: Vec<String>,
    /// Rationale for the decision.
    pub rationale: String,
}

// ── Tool trait + types ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[async_trait]
pub trait Tool: Send + Sync {
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
    /// The default implementation returns `true` (scrub everything). Override for
    /// tools that produce output where credential patterns are harmless or expected
    /// (e.g. source code reads), so the model sees accurate content.
    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        true
    }

    /// Whether this tool has side effects that would conflict with parallel
    /// execution of other tools.
    ///
    /// Returns `true` by default (conservative — assume mutating). Override to
    /// `false` for read-only tools (read, search, web_search, etc.) so they can
    /// be grouped for parallel execution within a single LLM turn.
    ///
    /// Takes `args` so tools can inspect parameters (e.g. ShellTool checking
    /// its mode), but most tools ignore it and return a constant.
    fn side_effects(&self, _args: &serde_json::Value) -> bool {
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

    /// Format tool output for LLM consumption.
    ///
    /// Called when tool results are embedded into the conversation history
    /// (both in native tool-call mode and degraded text mode). The default
    /// implementation uses head+tail truncation via [`crate::util::format_tool_output`]
    /// (hardcoded 5K limit); override to produce a smarter summary (e.g. trim
    /// repetitive CLI output or extract key facts from search results).
    fn format_output(&self, output: &str) -> String {
        crate::util::format_tool_output(output)
    }

    /// Format a tool channel notification.
    ///
    /// Called twice per tool execution:
    /// - `Before` — before the tool runs (`outcome` is `None`).
    /// - `After`   — after the tool completes (`outcome` is `Some`).
    ///
    /// Return `None` to suppress the notification for this phase.
    /// Return `Some(msg)` to send that exact message.
    fn debug_output(
        &self,
        phase: ToolOutputPhase,
        args: &serde_json::Value,
        outcome: Option<&crate::tools::ToolExecutionOutcome>,
    ) -> Option<String> {
        match phase {
            ToolOutputPhase::Before => {
                let args_preview = crate::util::summarize_args(args);
                Some(format!("🔧 `{}`({})", self.name(), args_preview))
            }
            ToolOutputPhase::After => {
                let outcome = outcome?;
                let status = if outcome.success { "✅" } else { "❌" };
                let name = self.name();
                let preview: String = outcome.output.chars().take(600).collect();
                let preview = preview.trim();
                if preview.is_empty() {
                    Some(format!("{status} `{name}`"))
                } else {
                    Some(format!("{status} `{name}` → {preview}"))
                }
            }
        }
    }
}

/// Phase of tool output notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolOutputPhase {
    /// Notification sent before tool execution.
    Before,
    /// Notification sent after tool execution completes.
    After,
}

// ── Message role enum ──────────────────────────────────────────

/// Typed role for a [`ChatMessage`].
///
/// Replaces the previous `String`-based role to prevent typos and
/// make the message-role API self-documenting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    /// System prompt message.
    System,
    /// User (human) message.
    User,
    /// Assistant (LLM) message.
    Assistant,
    /// Tool result message.
    Tool,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::Tool => write!(f, "tool"),
        }
    }
}

impl std::str::FromStr for MessageRole {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "system" => Ok(Self::System),
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            "tool" => Ok(Self::Tool),
            other => Err(format!("Unknown message role: {other}")),
        }
    }
}

// ── Provider trait + types ──────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
}

impl ChatMessage {
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }

    fn tool(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
        }
    }

    /// Construct a tool-role message with a structured `tool_call_id` + `content` payload.
    /// Used to attach a tool result to a specific assistant tool-call.
    #[must_use]
    pub fn tool_result(tool_call_id: &str, content: &str) -> Self {
        let payload = serde_json::json!({
            "tool_call_id": tool_call_id,
            "content": content,
        });
        Self::tool(payload.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reasoning {
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
pub struct ProviderUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct ChatResponse {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<ProviderUsage>,
    pub reasoning: Option<Reasoning>,
}

impl ChatResponse {
    #[must_use]
    pub fn text_or_empty(&self) -> &str {
        self.text.as_deref().unwrap_or("")
    }
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<ToolSpec>>,
    pub model: String,
    pub allow_image_parts: bool,
    pub temperature: f32,
    /// Per-role reasoning effort (xhigh, high, medium, low, minimal).
    /// When set, sent as `reasoning_effort` in the API request body.
    /// When `None`, the parameter is omitted (model defaults apply).
    pub reasoning_effort: Option<String>,
    /// Provider routing: provider order list (comma-separated slugs).
    /// When `Some` and non-empty, sent as the provider routing block.
    /// When `None`, no provider routing is sent (OpenRouter defaults apply).
    pub provider_order: Option<String>,
    /// Provider routing: allow fallbacks when provider_order is set.
    /// When `None` (and provider_order is set), defaults to `false`.
    pub provider_allow_fallbacks: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub delta: String,
    pub reasoning: Option<Reasoning>,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Text delta from the assistant.
    TextDelta(StreamChunk),
    /// Structured tool call emitted during streaming.
    ToolCall(ToolCall),
    /// Stream has completed.
    Final,
}

#[async_trait]
pub trait Provider: Send + Sync {
    /// Send a chat request using the model specified in the request.
    async fn chat(&self, request: ChatRequest) -> anyhow::Result<ChatResponse>;

    async fn warmup(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> stream::BoxStream<'static, StreamResult<StreamEvent>>;
}

pub type StreamResult<T> = std::result::Result<T, StreamError>;

/// Structured error for streaming responses.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("HTTP error: {0}")]
    Http(String),

    #[error("JSON parse error: {0}")]
    Json(serde_json::Error),

    #[error("Provider error: {0}")]
    Provider(String),
}

/// A skill is a user-defined or community-built capability.
/// Skills live in `<workspace>/skills/<name>/` (also
/// `<workspace>/.claude/skills/<name>/` and
/// `<workspace>/.agents/skills/<name>/`) and are defined via `SKILL.md`.
/// They provide instructions and context for the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
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
        let empty = DiagnosticsCommands {
            format: None,
            format_check: None,
            lint: None,
            lint_fix: None,
            type_check: None,
            build: None,
            unit_test: None,
        };
        assert!(empty.is_empty());

        let partial = DiagnosticsCommands {
            format: Some("cargo fmt".into()),
            format_check: None,
            lint: None,
            lint_fix: None,
            type_check: None,
            build: None,
            unit_test: None,
        };
        assert!(!partial.is_empty());
    }
}
