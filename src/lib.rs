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

// One agent's work loop, tool rounds, role metadata and running-agent registry.
pub mod agent;
// Automations that outlive the process, and who is told when one fires.
pub mod automations;
// Admin-defined custom roles: the record, its validation, its tool subset.
pub(crate) mod custom_role;
// The durable rows behind async work, boot recovery and the shutdown drain.
pub mod jobs;
// Branch lifecycle for custom roles and worker branches.
pub(crate) mod launch;

// Where parallel participants' answers are merged into one verdict.
pub(crate) mod consensus;
// Tickets from backlog to done: claim, dispatch, bounce handling, pause.
pub mod pipeline;

// Git wrappers and diff parsing, for the worktree checks, the dashboard and nightly runs.
pub(crate) mod git;
// Conversation persistence: the in-memory session, its store, the transcript snapshots.
pub mod session;
// The account row: a preferred workspace and channel recipients.
pub mod users;
// Workspace records, discovery and analysis, and the editor's open tabs.
pub mod workspace;
// The per-workspace file index shared by search, read, edit and the Git page.
pub mod workspace_index;

// Every agent-callable tool.
pub mod tools;

// The embedded document kit, its `rules.json` vocabulary and the PDF font.
pub(crate) mod docgen;
// The one converter every document read passes through: OOXML, legacy and PDF.
pub(crate) mod document;
// The CFB reader for pre-OOXML Word, Excel and PowerPoint files.
pub(crate) mod legacy;
// The ZIP-package reader for `.docx`, `.xlsx` and `.pptx` parts.
pub(crate) mod ooxml;
// What a PDF says outside its page text: metadata, annotations, AcroForm fields.
pub(crate) mod pdf_marks;
// The printed shape of a converted document: block headers, unit marks, stories.
pub(crate) mod reader_output;

// The product's own ffmpeg and ffprobe, never the owner's copy, for probing and frames.
pub mod ffmpeg;
// What a video or a recording says: frames as pictures of four, a timestamped transcript, sound
// labels, and the one timeline a model reads them as.
pub(crate) mod media;
// HyperFrames, the HTML renderer: installed by the managed bun, run with its own browser.
pub mod hyperframes;

// A standalone benchmark for OpenRouter models, run outside the product.
pub mod bench_openrouter;
// Model providers: the OpenAI-compatible client, reasoning replay, error classes.
pub mod providers;
// The single retry authority for LLM calls: attempt schedules and backoff.
pub(crate) mod retry;

// The message plane: intake, enrichment, reply references, chat history, transports.
pub mod channels;

// The `config_kv` and model-routing rows in the domain store.
pub mod config_db;
// All persistence: the Turso layer, schema catalog, change capture and the debug surface.
pub mod db;
// How long each by-product class lives, and the paced sweep that removes it.
pub mod log_retention;
// The logs store: tracing installation, the batched writer and the GUI's live log feed.
pub mod logs;
// Two row families — one per tool call, one per LLM request.
pub(crate) mod stats;
// Fire-and-forget rows about the product's own work, drained by their own task.
pub(crate) mod telemetry;

// The product's artwork, and every surface outside the window that needs it.
pub mod app_icon;
// Spawning chrome-use and keeping its envelope, forms and action registry in one place.
pub(crate) mod chrome;
// Every page and widget of the dashboard, plus the iced application entry.
pub mod gui;
// What a fresh install is walked through before it is usable.
pub(crate) mod onboarding;
// A broadcast that tells the dashboard an in-memory registry changed.
pub(crate) mod runtime_events;
// The macOS menu bar / Windows notification area icon, its click and quit menu.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub mod tray;
// The ways into the window from outside it, and the product's one in-app quit.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub mod window_requests;

// The operation channel a `mahbot` command line carries into the live instance.
pub mod cli;
// Loading and substitution for the embedded prompt assets.
pub(crate) mod prompt;

// The microphone-to-meaning stack: capture, the voice command pipeline, the wake word, and the
// machine's audio output.
pub mod audio;
// The product's own speech engines on every platform: recognition, spoken replies, and the
// machine-wide model set both read.
pub(crate) mod speech;
// The `mahbot audio` command line: a recording into text, text into a spoken file.
pub(crate) mod audio_cli;

// Handing the terminal back, with what it would have printed going to `daemon.log`.
pub mod detach;
// How the process comes up: one ordering for every store open.
pub mod boot;
// Defaults, persisted overrides, model routing and the process-global handle.
pub mod config;
// The low-disk notice, raised when free space on the temp volume falls below 5%.
pub mod disk_notice;
// The nightly memory inventory: an assistant whose memory no longer fits is asked to trim.
pub mod memory_inventory;
// Asking the OS not to throttle a daemon whose window is never visible.
pub mod power;
// Start at login: the product's own registration with the operating system.
pub mod autostart;
// Downloading, verifying and swapping in a new version, under the instance lock.
pub mod self_update;
// The owner's own shell environment, re-read in the background.
pub mod shell_env;
// Shutdown and drain: the token, the signal sources and the two-request protocol.
pub mod shutdown;
// Ownership of the daemon's private temp directory, from creation to cleanup.
pub mod temp;
// Who is at the computer right now: the machine-wide active/absent state and its
// input reader.
pub(crate) mod user_activity;

// The cancel action behind the `analyze` tool, confined to the caller's own runs.
pub(crate) mod analyze_cancel;
// Stopping a research run on request, and settling runs cancelled earlier.
pub(crate) mod research_cancel;
// The single release point for a research run's folders, archive and command dump.
pub mod research_cleanup;
// Stop flags for running deep-research and analyze rounds.
pub(crate) mod round_cancel;

// Shared helpers: HTTP and TLS, locking, paths, managed binaries, media markers, fixtures.
pub mod util;

/// Hidden grep-engine subcommand entry (dispatched from `main()` before the
/// instance lock; the shell tool's transparent grep interception rewrites
/// served invocations to this subcommand of the current binary).
pub use tools::shell::grep_engine::run_engine as run_grep_engine;

/// `mahbot audio` subcommand entry (dispatched from `main()` before the instance
/// lock — the speech engine's file-into-text and text-into-file operations,
/// neither of which needs the running product).
pub use audio_cli::run as run_audio_cli;

/// `mahbot chrome` subcommand entry (dispatched from `main()` before the
/// instance lock — browser automation that runs alongside the daemon).
pub use chrome::cli::run_cli as run_chrome_cli;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::session::Session;
use crate::util::UnwrapPoison;

/// Discovered diagnostic commands for a workspace; `None` means the tooling was
/// not detected.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct DiagnosticsCommands {
    pub format: Option<String>,
    pub format_check: Option<String>,
    pub lint: Option<String>,
    pub lint_fix: Option<String>,
    pub type_check: Option<String>,
    pub build: Option<String>,
    pub unit_test: Option<String>,
}

impl DiagnosticsCommands {
    /// Must match the array length in [`Self::commands`].
    pub const COMMAND_COUNT: usize = 7;

    /// Spelled once so the label list and [`Self::commands`] cannot drift.
    pub(crate) const UNIT_TEST_LABEL: &str = "unit-test";

    pub const COMMAND_LABELS: [&str; Self::COMMAND_COUNT] = [
        "format",
        "format-check",
        "lint-fix",
        "lint",
        "type-check",
        "build",
        Self::UNIT_TEST_LABEL,
    ];

    /// `None` entries are undiscovered — skipped.
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

    /// Same ordering as [`Self::commands`]. Empty strings become `None`.
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

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands().iter().all(|(_, cmd)| cmd.is_none())
    }
}

/// Lifecycle status of a workspace, mirroring the `status` TEXT column in the
/// workspaces table; conversion to/from strings happens at the DB boundary only.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Default, strum::Display, strum::EnumString,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum WorkspaceStatus {
    /// Managed workspace added but discovery has not started.
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
    /// Gates only the maintainer loop, unlike [`Self::paused`] (which freezes the
    /// whole pipeline): a paused workspace can still be maintained, and disabling
    /// maintenance cancels a running maintainer outright where pausing is
    /// cooperative. Persisted with a `DEFAULT 0` schema default.
    pub maintenance_enabled: bool,
    /// A strict freeze: no pipeline stage advances — including the dispatcher
    /// finalizers — and its agents freeze at their next LLM round boundary until
    /// manual resume (no cancel token is fired, so a round in flight finishes its
    /// turn first). Set automatically on technical failures, and by a
    /// `mahbot paused` launch, so queued tickets aren't claimed and don't cascade;
    /// lifted by the normal unpause path. A discovery in flight when the pause
    /// lands clears it on completion without checking who set it — see
    /// `finalize_discovery`.
    pub paused: bool,
    /// RFC 3339 timestamp of the last completed maintainer run; `None` means never.
    pub maintainer_last_run_at: Option<String>,
    /// JSON blob of discovered dev commands; `None` before the first discovery run.
    pub diagnostics: Option<String>,
    /// Freeform user-curated context notes appended to every agent's system prompt,
    /// max 4000 characters. Survives `rediscover()` — never touched by analysis.
    pub notes: String,
    /// The git HEAD commit captured after the last successful discovery, read by the
    /// nightly re-analysis check; `None` if not a git repository or no commits.
    pub last_analyzed_commit: Option<String>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            name: String::default(),
            path: String::default(),
            status: WorkspaceStatus::Pending,
            maintenance_enabled: bool::default(),
            paused: bool::default(),
            maintainer_last_run_at: Option::default(),
            diagnostics: Option::default(),
            notes: String::default(),
            last_analyzed_commit: Option::default(),
        }
    }
}

impl Workspace {
    /// The only gate for fresh maintainer runs, in minutes.
    pub const MAINTAINER_DEBOUNCE_MINS: i64 = 5;

    #[must_use]
    pub fn as_path(&self) -> &Path {
        Path::new(&self.path)
    }

    /// A minimal Workspace from a filesystem path, named after its last component.
    /// Used by test helpers, personal workspaces and as the not-found fallback.
    #[must_use]
    pub fn from_path(path: &Path) -> Self {
        let stored = stored_workspace_path(path);
        Self {
            name: last_path_component(&stored),
            path: stored.to_string_lossy().to_string(),
            ..Default::default()
        }
    }

    /// The workspace path's last component, for display.
    #[must_use]
    pub fn display_name(&self) -> String {
        last_path_component(self.as_path())
    }

    /// A per-run workspace (research run roots, the temp cleaner, targeted
    /// delegation rounds). `name` becomes the search-engine registry key (e.g. the
    /// research `job_id`); it has no `workspaces` row and lives as long as its run.
    #[must_use]
    pub fn ephemeral_run(name: &str, path: &Path) -> Self {
        Self {
            name: name.to_string(),
            path: stored_workspace_path(path).to_string_lossy().to_string(),
            ..Default::default()
        }
    }
}

/// The path a [`Workspace`] built from a filesystem path stores: canonicalized, so
/// containment stays correct even when the base is behind a symlink (`/tmp` →
/// `/private/tmp` on macOS), with the platform's verbatim prefix dropped (see
/// [`crate::util::strip_verbatim_prefix`]). Falls back to the raw path when the
/// directory does not exist yet.
pub(crate) fn stored_workspace_path(path: &Path) -> PathBuf {
    crate::util::with_block_in_place(|| {
        std::fs::canonicalize(path).map_or_else(
            |_| path.to_path_buf(),
            |canonical| crate::util::strip_verbatim_prefix(&canonical),
        )
    })
}

fn last_path_component(path: &Path) -> String {
    path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

const NANOID_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-";

#[must_use]
fn generate_nanoid(length: usize) -> String {
    (0..length)
        .map(|_| {
            let idx = (rand::random::<u8>() % 64) as usize;
            NANOID_ALPHABET[idx] as char
        })
        .collect()
}

/// A 10-char NanoID, ~60 bits entropy.
#[must_use]
pub fn generate_id() -> String {
    generate_nanoid(10)
}

/// A 6-char NanoID (~36 bits), disambiguating retry cycles in parallel dispatches.
#[must_use]
pub(crate) fn generate_suffix() -> String {
    generate_nanoid(6)
}

/// Represents a recognized Telegram bot text command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotCommand {
    /// `/start` — show the welcome message.
    Start,
    /// `/clear` — reset the user's current session.
    Clear,
    /// `/image_models` — show image model selection keyboard.
    ImageModels,
    /// `/video_models` — show video model selection keyboard.
    VideoModels,
    /// `/board` — list every workspace's tickets (admin).
    Board,
    /// `/archive` — archive done & cancelled tickets from every workspace (admin).
    Archive,
    /// `/update` — self-update and restart the daemon (admin, only offered when the
    /// shared availability cache confirms an update).
    Update,
    /// `/agents` — replies that there is nothing to switch; no longer in any command
    /// menu, kept as a graceful reply.
    Agents,
}

/// Parse a Telegram bot text command, case-insensitively. `None` for anything else,
/// including other `/`-prefixed text, which is routed to the agent pipeline.
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
        "/update" => Some(BotCommand::Update),
        "/agents" => Some(BotCommand::Agents),
        _ => None,
    }
}

/// `Default` yields an **undeliverable** placeholder (all string fields empty): it
/// exists only as a `..Default::default()` spread for local-source construction
/// sites, keeping Telegram-only fields out of their literals. Never dispatch
/// `ChannelMessage::default()` — an empty `channel`/`user_name`/`content` is not
/// routable; the one shape that legitimately carries an empty `content` is a
/// collected burst, whose text lives in its [`parts`](Self::parts). A new
/// non-`Option` field is silently defaulted at every spread site.
#[derive(Debug, Clone, Default)]
pub struct ChannelMessage {
    /// The user's canonical name, resolved from channel binding at auth time —
    /// never derived from Telegram `@username`.
    pub user_name: String,
    pub reply_target: String,
    pub content: String,
    /// What the owner's own bubble shows, when it differs from [`content`](Self::content): a
    /// video's timeline reaches the model with its pictures as ordinary `[IMAGE:]` markers, and
    /// the bubble shows the same timeline without the pictures. `None` means the two are the same.
    pub display_content: Option<String>,
    /// The channel this message originated from (e.g. `"telegram"`, `"gui"`).
    pub channel: String,
    pub workspace: String,
    /// Set by the GUI sender for deduplication in the ChatEvent handler; `None` for
    /// non-GUI channels.
    pub optimistic_id: Option<String>,
    /// Only set for Telegram callback queries (`__act__` prefix), to dismiss the
    /// loading spinner.
    pub callback_query_id: Option<String>,
    /// The message this one replies to, for the quote header.
    pub reply_reference: Option<crate::channels::ReplyReference>,
    /// Channel-side chat id of the originating message, so it can be addressed in
    /// place (e.g. editing a picker keyboard after a callback).
    pub chat_id: Option<String>,
    /// Channel-side message id of the originating message (see [`Self::chat_id`]).
    pub message_id: Option<i64>,
    /// The staging directories this message owns inbound attachments for: one for a
    /// single message, several for a merged album. The producer names them with
    /// `util::telegram_staging_dir_name`, so this is the set the message may read,
    /// copy from and delete.
    pub attachment_dirs: Vec<String>,
    /// The messages a collected burst is composed of, in send order, when the
    /// Telegram channel merged several into one request (see
    /// `channels::telegram_group`). Empty for a message delivered on its own. One
    /// level deep — a part never carries parts.
    pub parts: Vec<ChannelMessage>,
    /// Telegram's own `message.date` (Unix seconds), never this product's clock.
    pub update_date: Option<i64>,
    /// Decided from Telegram's own fields (`forward_origin`, or the older
    /// `forward_*`), never from rendered text: the burst collector gives a group its
    /// longer window on this flag, and the intake's `[Forwarded from …]` prefix is
    /// written from the same decision.
    pub forwarded: bool,
    /// The Telegram intake's record entries this message stands for, one per incoming
    /// update it carries: the intake settles them once the message reaches the
    /// pipeline, so one it never handed on stays held for a later run. Empty for
    /// every other channel. One level deep, like `parts`.
    pub update_ids: Vec<i64>,
    /// `content` leads with a voice recording's transcript. Only the dispatch log
    /// line differs, naming such a message without its content — spoken words stay
    /// out of the logs.
    pub dictated: bool,
    /// The user's text opened with the urgent mark, which the surface consumed before
    /// composing `content` (see `channels::take_urgent`): the message cuts the
    /// Assistant's running turn instead of waiting behind it. Only a surface a user
    /// types into sets it; anything the product composes or routes leaves it `false`.
    pub urgent: bool,
}

/// An outbound message to deliver on a channel.
///
/// `reply_markup` is Telegram inline_keyboard JSON, built in `main.rs`
/// (`build_models_keyboard`); other channels ignore it. The self-update path stores
/// `reply_target` into `recipient` for admin notifications — see the channel's
/// `resolve_recipient()`.
#[derive(Debug, Clone)]
pub struct SendMessage {
    pub content: String,
    pub recipient: String,
    pub reply_markup: Option<serde_json::Value>,
    /// Workspace roots anchoring outbound media markers: the authoring agent's
    /// workspace plus, for relayed content, the one it originated in. A relative
    /// `[IMAGE:]`/`[AUDIO:]`/`[VIDEO:]`/`[FILE:]` target names a file in the first
    /// root that has it, and `[FILE:]` delivery is confined to these roots. Empty =
    /// no workspace: no relative marker is deliverable and no file is authorized.
    pub file_roots: Vec<std::path::PathBuf>,
}

/// One media marker a transport could not attach, reported so the conversation that wrote it can
/// be told. `marker` is the marker as the reply wrote it — bounded, and without an address's
/// credentials — or just `[KIND]` for an inline payload, which names no path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefusedMarker {
    pub marker: String,
    /// Why it could not be delivered, in the same words the display layer uses.
    pub reason: String,
}

/// What one transport send did besides delivering: the markers it refused.
#[derive(Debug, Clone, Default)]
pub struct SendOutcome {
    pub refused: Vec<RefusedMarker>,
}

/// A send that failed, carrying what it did before the failure: the markers it refused, and
/// whether the message's text reached the owner. A transport that fails on one attachment has still
/// delivered the text, and the note about a refused marker must say which happened — otherwise the
/// assistant resends only the file and the reply's text is lost.
#[derive(Debug)]
pub struct SendFailure {
    pub error: anyhow::Error,
    /// Whether the message body reached the owner before the failure.
    pub delivered: bool,
    /// The markers refused before the failure, in the order they were found.
    pub refused: Vec<RefusedMarker>,
}

impl std::fmt::Display for SendFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::error::Error for SendFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error.as_ref())
    }
}

#[async_trait]
pub trait Channel: Send + Sync {
    /// The error carries the refusals collected before the failure, so no caller can lose them.
    async fn send(&self, message: &SendMessage) -> Result<SendOutcome, SendFailure>;
    async fn listen(&self, tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()>;
    fn name(&self) -> &'static str;

    /// As `&dyn Any`, so the Telegram hot-reload path can swap the running channel
    /// without losing update continuity.
    fn as_any(&self) -> &dyn std::any::Any;

    async fn start_typing(&self, _recipient: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// The address a known user is reachable at on this channel, or `None` if they
    /// are not. The default is `reply_target` (the Telegram chat_id); GUI channels
    /// override it with the user name.
    fn resolve_recipient(&self, _user_name: &str, reply_target: &str) -> Option<String> {
        Some(reply_target.to_string())
    }
}

// Chat event broadcast (GUI live display)

/// Direction of a chat message: from the user or from an agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatDirection {
    User,
    Agent,
    /// Inserted when a user clears their session; rendered as a labelled rule.
    Divider,
}

/// A chat event broadcast to GUI subscribers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatEvent {
    /// A complete chat message (user or agent).
    Message {
        message_id: String,
        user_name: String,
        content: String,
        direction: ChatDirection,
        /// ISO-8601.
        timestamp: String,
        channel: String,
        /// The agent's role (e.g. "manager"), if from an agent.
        agent_role: Option<String>,
        workspace: String,
        /// So the Home page can replace its optimistic message with the confirmed one.
        optimistic_id: Option<String>,
        /// Live GUI display only, deliberately NOT persisted to chat_history (e.g. the
        /// scripted onboarding exchange).
        transient: bool,
        #[serde(default)]
        reply_reference: Option<crate::channels::ReplyReference>,
    },
    /// Typing indicator event.
    Typing {
        user_name: String,
        is_typing: bool,
        workspace: String,
    },
}

/// Global broadcast for live chat events in the GUI, set by `init_message_pipeline`.
/// Capacity 256 keeps `Lagged` errors out of burst scenarios.
pub static CHAT_BROADCAST: OnceLock<broadcast::Sender<ChatEvent>> = OnceLock::new();

/// Global sender for the pipeline message channel, set by `init_message_pipeline`;
/// `GuiChannel::listen()` forwards GUI-originated messages through it.
pub static MESSAGE_TX: OnceLock<tokio::sync::mpsc::Sender<ChannelMessage>> = OnceLock::new();

/// Global sender for GUI-to-channel messages: the Iced UI pushes `ChannelMessage`
/// values here and `GuiChannel::listen()` drains them into the pipeline.
pub static GUI_MESSAGE_TX: OnceLock<tokio::sync::mpsc::UnboundedSender<ChannelMessage>> =
    OnceLock::new();

/// Registry of all active channels, keyed by [`Channel::name()`], routing outbound
/// replies, agent responses and typing indicators.
#[derive(Default)]
pub struct ChannelRegistry {
    channels: RwLock<HashMap<String, Arc<dyn Channel>>>,
    /// What the replaces under each name have moved past: [`ReplacedChannels`].
    replaced: RwLock<HashMap<String, Arc<ReplacedChannels>>>,
}

/// The channels a name's replaces have moved past and no replacement has confirmed
/// stopped — what a replacement waits out before it takes over. Held per name by the
/// registry, not by the channels: every replacement landing while an earlier one
/// waits must wait for the same frames (see `channels::telegram`'s `stop_listeners`).
///
/// An entry stays alive until [`Self::retire`] drops it, and the registry holds one
/// per name for the process's life: a second user of [`ChannelRegistry::replaced`]
/// that records without retiring would pin its channels alive forever. The channel a
/// hand-over is about to start is deliberately left in the list, so an entry may
/// outlive the hand-over that first met it.
#[derive(Default)]
pub(crate) struct ReplacedChannels {
    channels: Mutex<Vec<Arc<dyn Channel>>>,
}

impl ReplacedChannels {
    /// The channels moved past so far, oldest first.
    #[must_use]
    pub(crate) fn snapshot(&self) -> Vec<Arc<dyn Channel>> {
        self.channels.lock().unwrap_poison().clone()
    }

    /// Note a channel the registry has moved past. Recording one twice is a no-op: a
    /// save and a token teardown can both move past it, and it must not be waited out twice.
    pub(crate) fn record(&self, channel: Arc<dyn Channel>) {
        let mut held = self.channels.lock().unwrap_poison();
        if !held.iter().any(|held| Arc::ptr_eq(held, &channel)) {
            held.push(channel);
        }
    }

    /// Drop a channel, once a replacement has confirmed it stopped.
    pub(crate) fn retire(&self, channel: &Arc<dyn Channel>) {
        self.channels
            .lock()
            .unwrap_poison()
            .retain(|held| !Arc::ptr_eq(held, channel));
    }
}

impl ChannelRegistry {
    /// Register a channel under its name; a duplicate name is skipped with a warning.
    pub fn register(&self, channel: Arc<dyn Channel>) {
        let name = channel.name().to_string();
        let mut map = self.channels.write().unwrap_poison();
        if let std::collections::hash_map::Entry::Vacant(entry) = map.entry(name.clone()) {
            entry.insert(channel);
        } else {
            tracing::warn!(channel = %name, "Channel registry: duplicate name — skipping register");
        }
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Channel>> {
        self.channels.read().unwrap_poison().get(name).cloned()
    }

    /// Replace a channel, returning the previous one if any and recording it as a
    /// channel the new instance has to wait out (the name's `ReplacedChannels`, via
    /// `Self::replaced`). Unlike [`Self::register`], this **always** inserts. Used to
    /// hot-reload a channel (e.g. a Telegram bot token change) without a restart.
    pub fn replace(&self, channel: Arc<dyn Channel>) -> Option<Arc<dyn Channel>> {
        let name = channel.name().to_string();
        let previous = self
            .channels
            .write()
            .unwrap_poison()
            .insert(name.clone(), channel);
        if let Some(previous) = &previous {
            self.replaced(&name).record(Arc::clone(previous));
        }
        previous
    }

    /// What the replaces under `name` have moved past. The same handle every time, so a
    /// replacement landing while an earlier one waits reads — and records into — the
    /// same list.
    #[must_use]
    pub(crate) fn replaced(&self, name: &str) -> Arc<ReplacedChannels> {
        if let Some(held) = self.replaced.read().unwrap_poison().get(name) {
            return Arc::clone(held);
        }
        let mut map = self.replaced.write().unwrap_poison();
        Arc::clone(
            map.entry(name.to_string())
                .or_insert_with(|| Arc::new(ReplacedChannels::default())),
        )
    }

    /// Remove a channel only if it is still `expected` — the instance the caller
    /// stopped — and report whether it was. A teardown that runs after the registry has
    /// moved on (a new token saved while the old listener winds down) must leave the
    /// live channel in place, or every outbound send loses its channel and no later
    /// restart reaches the listener actually running.
    pub fn unregister_if(&self, expected: &Arc<dyn Channel>) -> bool {
        let name = expected.name().to_string();
        let mut map = self.channels.write().unwrap_poison();
        if map
            .get(&name)
            .is_some_and(|current| Arc::ptr_eq(current, expected))
        {
            map.remove(&name);
            true
        } else {
            false
        }
    }

    /// A snapshot of all registered channels (name → Arc clone).
    pub fn list(&self) -> Vec<(String, Arc<dyn Channel>)> {
        self.channels
            .read()
            .unwrap_poison()
            .iter()
            .map(|(k, v)| (k.clone(), Arc::clone(v)))
            .collect()
    }
}

pub static CHANNEL_REGISTRY: OnceLock<ChannelRegistry> = OnceLock::new();

/// The global channel registry. Panics if not initialized — call after bootstrap.
#[must_use]
pub fn channel_registry() -> &'static ChannelRegistry {
    CHANNEL_REGISTRY
        .get()
        .expect("CHANNEL_REGISTRY not initialized — must be set during bootstrap")
}

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
    Worker,
    Qa,
    Reviewer,
    Discovery,
    Maintainer,
    Sanitation,
    Assistant,
    Custom,
}

/// A launch: one continuable branch of a custom role, or a worker branch. For a custom
/// role it carries the authorial definition as it stood when the launch was created — a
/// later edit of the record takes effect on the NEXT launch — for a worker branch the
/// definition fields are empty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Launch {
    /// `custom` for an administrator-defined role, `worker` for a worker branch.
    /// Envelopes stored before the field existed can only be custom-role launches.
    #[serde(default = "Launch::custom_role_default")]
    pub role: Role,
    /// The custom role's name; empty for a worker branch.
    pub name: String,
    /// The branch this launch continues; `None` for an unnamed branch (the record's
    /// default branch or a one-off run).
    #[serde(default)]
    pub identifier: Option<String>,
    /// The authorial prompt text, taken at launch; empty for a worker branch.
    pub prompt: String,
    /// The chosen subset of the admin Assistant's tool names, taken at launch; empty
    /// for a worker branch.
    pub tools: Vec<String>,
    /// The resolved session/branch address (`role_{name}`, `role_{name}:{id}`,
    /// `rolerun_{name}:{suffix}` or `worker_{branch}`).
    pub agent_id: String,
    /// The conversation that asked for this launch's work, so the report returns into
    /// that branch instead of the admin's Assistant. PER-LAUNCH: each use names its
    /// asker, so a later use with a different asker overrides it; `None` for a launch
    /// the owner or an automation started. Boxed because the asker itself holds a
    /// [`crate::Launch`], and `#[serde(default)]` keeps older envelopes decodable.
    #[serde(default)]
    pub reply_to: Option<Box<crate::agent::message_router::Asker>>,
}

impl Launch {
    /// The role of an envelope stored before the `role` field existed: only a
    /// custom-role launch could have written one.
    fn custom_role_default() -> Role {
        Role::Custom
    }
}

impl Role {
    /// The role a dispatched round belongs to: a round carrying a launch runs as the
    /// launch's own role whatever the tool's construction role was — the launch is the
    /// address authority, and the job row and result envelope must say so.
    #[must_use]
    pub(crate) fn for_launch(self, launch: Option<&Launch>) -> Self {
        launch.map_or(self, |launch| launch.role)
    }
}

/// One tool invocation's record, handed to the telemetry queue on session finalization.
#[derive(Debug, Clone)]
pub(crate) struct ToolCallRecord {
    pub tool_name: String,
    /// Serialized arguments as JSON (credentials scrubbed).
    pub arguments: String,
    /// Milliseconds.
    pub duration_ms: i64,
    pub success: bool,
    /// `None` on success.
    pub error_message: Option<String>,
}

/// The per-turn state the consumer loop and the post-delivery summarization gate read:
/// [`Agent::work`] fills it, then the per-session consumer loop reads it once the turn is
/// over (see [`crate::agent::Agent::maybe_summarize_after_delivery`]).
#[derive(Default)]
pub(crate) struct TurnState {
    /// The turn settled previously dangling tool results — the trail a resume is
    /// recovered from, which a compaction's retention window would drop.
    pub(crate) settled_tool_results: bool,
    /// An urgent message cut the turn: its session entry is the phrase alone, and the
    /// consumer loop delivers nothing — no chat row, no channel message.
    pub(crate) interrupted: bool,
}

/// How a failed agent run was classified when it failed — the snapshot kept on
/// [`Agent::run_failure`] so later process-global stops cannot reclassify it.
#[derive(Clone, Copy)]
pub(crate) enum RunFailure {
    /// A genuine technical failure (the pipeline reset/discard path).
    Technical,
    /// A cooperative stop (drain, shutdown, pause-freeze, deliberate cancel).
    CooperativeStop,
}

/// Running context for an agent turn.
pub struct Agent {
    /// Session persistence key, also the registry agent ID: direct-chat agents use
    /// `{user}_{ws_name}_{role}` (or the deduped `{user}_personal:{role}`), stable
    /// per user+workspace+role; sub-agents and ticket handlers use a fresh NanoID.
    #[expect(clippy::struct_field_names)]
    pub(crate) agent_id: String,
    role: Role,
    /// The agent owns its session — every session method takes the key from
    /// `self.agent_id`.
    pub(crate) session: Session,
    workspace: Arc<crate::Workspace>,
    tools: Vec<Box<dyn crate::Tool>>,
    /// Chrome sessions this run opened, closed through the browser's own extension.
    chrome_sessions: std::sync::Arc<crate::tools::chrome::ChromeRunSessions>,
    /// Computed once from `tools` at construction time.
    pub(crate) tool_specs: Vec<ToolSpec>,
    /// Cooperative mid-loop cancellation (a deliberate stop — GUI cancel, re-dispatch,
    /// phase transition, or shutdown).
    cancel_token: CancellationToken,
    /// A workspace-pause (strict freeze) cancellation was requested for this run, set
    /// by `cancel_by_workspace_pause` — a FREEZE distinct from a code-driven
    /// cancellation; the pipeline failure classifier leaves the ticket in its source
    /// phase for unpause.
    pause_stop: Arc<std::sync::atomic::AtomicBool>,
    /// IMMUTABLE bail-time snapshot of whether this run stopped at a cooperative
    /// workspace-pause freeze — set once at the LLM-round boundary, never cleared.
    /// Unlike the live `pause_stop` flag, it survives the window where the agent is
    /// still registered and the workspace is resumed between the bail and the pipeline
    /// finalizer reading it, so a frozen run is never misrouted to the hard-failure path.
    paused_frozen: std::sync::atomic::AtomicBool,
    /// The board ticket this agent is working on (board-dispatched agents only).
    ticket: Option<crate::pipeline::board::Ticket>,
    /// Generation counter from the agent registry, for safe deregistration in [`Drop`];
    /// 0 means not registered (e.g. test agents).
    generation: u64,
    /// Per-call tool stats, enqueued for the log store on
    /// [`Agent::finalize_session`] — silently lost if the agent is dropped first.
    tool_stats: std::sync::Mutex<Vec<crate::ToolCallRecord>>,
    /// The user who triggered this run — tools route async sub-agent results back
    /// through it.
    pub(crate) user_name: String,
    /// Gates the Assistant's widened toolset and role prompt.
    pub(crate) is_admin: bool,
    /// The channel origin (gui, telegram, voice, cli) of the triggering message.
    pub(crate) channel: String,
    /// DIRECT PARENT INVOCATION grouping key for the Running Agents view (ticket /
    /// analyze round / research run); `None` for workspace singletons. Propagated to
    /// sub-agents via [`CURRENT_TOOL_PARENT_KEY`](crate::agent::CURRENT_TOOL_PARENT_KEY).
    pub(crate) parent_key: Option<crate::agent::registry::ParentKey>,
    /// The DIRECT PARENT INVOCATION's label (ticket title / analyze or research
    /// question), purely presentational for the group header. Propagated to sub-agents
    /// via [`CURRENT_TOOL_PARENT_LABEL`](crate::agent::CURRENT_TOOL_PARENT_LABEL).
    pub(crate) parent_label: Option<String>,
    /// Mid-work messages (e.g. ticket comments): `llm_loop` drains this channel before
    /// each LLM call and injects what it finds into the session history.
    pub(crate) incoming_rx:
        Option<tokio::sync::mpsc::UnboundedReceiver<crate::agent::message_router::AgentJob>>,
    /// Round-fixed timestamp for the first user message, so parallel members send
    /// byte-identical task messages.
    pub(crate) round_ts: Option<String>,
    /// Leader-stagger signal, fired after the agent's first LLM call completes (success
    /// or failure, full retry budget) or when the leader bails at the phase gate before
    /// any LLM call; `None` for non-leader members.
    pub(crate) first_call_notify: Option<std::sync::Arc<tokio::sync::Notify>>,
    /// Raw failure detail from the last run, set by [`crate::agent::run_agent`] on
    /// error; `None` on success or cancellation (callers classify cancellation via the
    /// cancel tokens).
    pub(crate) failure: Option<String>,
    /// Typed provider failure classification, set alongside [`Self::failure`] when the
    /// error chain carries a [`crate::retry::RetryExhausted`]; `None` for runtime
    /// errors, cancellations and successes. Workspace discovery reads it to tell
    /// provider-class failures (workspace returns to Pending) from genuine ones (Failed).
    pub(crate) failure_class: Option<crate::retry::FailureClass>,
    /// The classification captured at failure time (not re-derived), so a later
    /// process-global stop cannot reclassify the run; `None` on success.
    pub(crate) run_failure: Option<RunFailure>,
    /// Whether the turn ended via a successful `sleep` call.
    pub(crate) sleep_ended: bool,
    /// Whether this turn's user message was recorded. The turn's personal-workspace
    /// change report rides that message, so a turn that recorded none must not advance
    /// the folder's baseline — its changes stay pending for the next recorded message.
    /// Read by [`crate::agent::run_agent`], set in [`crate::Agent::work`].
    pub(crate) turn_recorded: bool,
    /// Per-turn state the consumer loop and the post-delivery gate read — see [`TurnState`].
    pub(crate) turn_state: TurnState,
    /// Background shell sessions (Full shell roles only), force-killed on teardown
    /// ([`Drop`]) — see [`crate::tools::shell::BackgroundSessions`]. Reachable from the
    /// tool-batch context via [`crate::agent::CURRENT_TOOL_BACKGROUND_SESSIONS`].
    background_sessions: std::sync::Arc<crate::tools::shell::BackgroundSessions>,
    /// Fired by [`crate::agent::registry::AgentRegistry::interrupt`] on an urgent user
    /// message: the turn is cut where it stands — a pending model request or tool call
    /// is dropped — and ends with [`TurnState::interrupted`] set instead of an answer. Never a
    /// cancellation: [`Self::cancel_token`] is untouched, so the turn still settles its
    /// tool results and finalizes its session. One token per turn, so an urgent message
    /// interrupts at most the turn it cuts.
    pub(crate) interrupt_token: CancellationToken,
}

/// Result of a single verification pass (code reviewer or functional tester).
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Verdict {
    /// 0 (worst) to 10 (best).
    #[serde(deserialize_with = "de_verdict_score")]
    pub score: u8,
    #[serde(rename = "issues")]
    pub issues_detected: Vec<String>,
}

/// Qualitative grade attached to one analysis finding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum IssueGrade {
    Minor,
    Major,
    Blocker,
}

impl IssueGrade {
    /// Lowercase label used in the rendered comment.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Minor => "minor",
            Self::Major => "major",
            Self::Blocker => "blocker",
        }
    }
}

/// One graded issue from an analysis analyst (score-less analysis verdict).
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct AnalysisIssue {
    pub text: String,
    pub grade: IssueGrade,
}

/// Unlike the shared `Verdict` (the verification stage's score-based shape), analysis
/// drops the score and grades each issue minor/major/blocker.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct AnalysisVerdict {
    #[serde(rename = "issues")]
    pub issues_detected: Vec<AnalysisIssue>,
}

/// Whether an aggregated blocker hinders the main implementation path or is a real but
/// non-blocking risk/edge case. Any human label is emitted only at render time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BlockerKind {
    MainPathBlocker,
    RiskEdgeCase,
}

/// Severity of an aggregated blocker if left unaddressed. `Ord` uses declaration order
/// so the two-verifier merge can take the max.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum BlockerSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// One blocker's substance outcome from an escalation analyst.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct BlockerVerificationItem {
    /// 0-based index in the injected list.
    pub index: usize,
    pub kind: BlockerKind,
    pub severity: BlockerSeverity,
    /// What the blocker affects / how it constrains the ticket (required).
    pub impact: String,
    /// Evidence-backed justification (required).
    pub reasoning: String,
}

/// Result of one escalation analyst verifying the aggregated blocker list.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct BlockerVerificationVerdict {
    /// One item per injected blocker (coverage validated).
    pub verdicts: Vec<BlockerVerificationItem>,
}

/// Deserialize a verdict score, self-healing common model emissions:
/// - [0,10] passes through, fractional values floor (8.5 → 8).
/// - [11,100] divides by 10 then floors (85 → 8, 11 → 1, 100 → 10).
/// - Integers 101–255 pass through so downstream validation keeps classifying them as
///   out-of-range (failure-class stability); the float spellings fail closed.
/// - Everything else fails closed — range checks precede any cast.
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
        // Out-of-range integers pass through for failure-class stability; floats in
        // this band (e.g. 255.0) fail closed.
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
            Case {
                json: r#"{"score": 101, "issues": []}"#,
                expected: Some(101),
            },
            Case {
                json: r#"{"score": 255, "issues": []}"#,
                expected: Some(255),
            },
            // The float spellings of that band fail closed — the lexical form decides.
            Case {
                json: r#"{"score": 101.0, "issues": []}"#,
                expected: None,
            },
            Case {
                json: r#"{"score": 255.0, "issues": []}"#,
                expected: None,
            },
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
    /// `true` = clean, `false` = garbage detected.
    pub pass: bool,
    /// Empty when `pass` is `true`.
    #[serde(default)]
    pub garbage_files: Vec<String>,
    pub rationale: String,
}

/// Concise structured summary of an engineer's completed work, extracted from the
/// engineer's session for the ticket comment.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct EngineerSummary {
    /// What was implemented / fixed / executed.
    #[serde(default)]
    pub items: Vec<String>,
    /// Measurements or figures the ticket explicitly asked for.
    #[serde(default)]
    pub reported: Vec<String>,
    /// One short paragraph summarizing the completed work (optional — legacy
    /// extractions and models emitting null leave it absent).
    pub summary: Option<String>,
}

// Tool trait + types

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

    /// The tool's answer to one operation, with the flag a frontend that must
    /// treat a partly carried-out operation as a failure reads. The default is
    /// the tool's plain `execute` answer and never partial; a tool whose
    /// operation can be carried out in part overrides it.
    async fn execute_answer(
        &self,
        ws: &crate::Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<crate::tools::ToolAnswer> {
        Ok(crate::tools::ToolAnswer {
            text: self.execute(ws, args).await?,
            partial: false,
        })
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name().to_string(),
            description: self.description(),
            parameters: self.parameters_schema(),
        }
    }

    /// Whether this tool's output should be scrubbed for credentials. The default is
    /// `true` (scrub-all, via `scrub_tool_output` in `tools/mod.rs`).
    ///
    /// **A tool that scrubs internally or returns stored text as it is must return
    /// `false`**, so the agent-level pass neither double-scrubs nor rewrites the answer:
    /// shell (`apply_profile_pipeline` already scrubs stdout and stderr once at pipeline
    /// entry), search (source code, where credential patterns are harmless and should be
    /// shown accurately), the debug query (a stored cell is the answer), and read (which
    /// scrubs in `read_resolved` by the *resolved* path — an args-only decision cannot
    /// see symlink targets or typo-recovered files). `web_search` scrubs its own picture
    /// text on that call only.
    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        true
    }

    /// Whether this tool has side effects that would conflict with parallel execution.
    /// `true` by default (conservative — assume mutating). Return `false` only when a
    /// sibling running at the same time can neither change what this call observes nor
    /// target what it writes: inspection tools qualify, even when they write artifacts of
    /// their own while reading (a shell spill, converted-document files), because each
    /// goes to a path unique to the call. `false` groups the tool for parallel execution
    /// within a single LLM turn.
    fn side_effects(&self) -> bool {
        true
    }

    /// Whether a SUCCESSFUL execution ends the run gracefully after the current tool
    /// round commits (no further LLM call). Failed outcomes never end the turn. This is
    /// the `sleep` tool's trigger.
    fn ends_turn_on_success(&self) -> bool {
        false
    }

    /// Whether the tool should be advertised to the model right now; tools backed by an
    /// external service override this to hide themselves while it is down, so the model
    /// doesn't burn calls against a dead backend.
    ///
    /// Read once when an agent is constructed and kept for its lifetime: an agent
    /// created during an outage permanently lacks the tool, and a long-lived agent keeps
    /// advertising it after the backend dies (its runtime fail-fast then handles that).
    /// Deliberate — agents are short-lived per run, and re-checking per turn would add latency.
    fn is_advertised(&self) -> bool {
        true
    }

    /// The prefix wrapping a generated file path in the agent's response content, e.g.
    /// `"[IMAGE:"`; the full marker `"{prefix}path]"` is built by `format_media_result`.
    /// Non-media tools leave the default `None`.
    fn media_marker(&self) -> Option<&'static str> {
        None
    }

    /// Whether every successful answer carries a media marker: `false` for a tool that
    /// also answers with text alone (the renderer returns documents and lint reports as
    /// well as videos), so a marker-less answer is not a fault to report.
    fn media_marker_always(&self) -> bool {
        true
    }

    /// Format the media output marker `[TYPE:path]` for a generated file. Single source
    /// of the marker shape parsed by [`crate::util::MEDIA_MARKER_RE`].
    fn format_media_result(&self, output_path: &Path) -> String {
        let marker_prefix = self
            .media_marker()
            .expect("media tool must define a media marker");
        format!("{marker_prefix}{}]", output_path.to_string_lossy())
    }

    /// Whether tool output must bypass the default ~5 KB truncation (e.g. ticket
    /// details, search results, media markers the LLM needs in full).
    fn preserve_full_output(&self) -> bool {
        false
    }

    /// Format tool results for embedding into the conversation history (native tool-call
    /// mode and degraded text mode alike). The default is head+tail truncation via
    /// [`crate::util::truncate_tool_output`] (a ~5 KB trigger, with the omission marker
    /// additional to it); override to summarize more smartly. A tool whose every answer must be
    /// kept whole says so with [`Tool::preserve_full_output`]; one call whose answer must not be
    /// cut — a media timeline — says so on its own
    /// [`crate::tools::ToolOutput::preserve_text`].
    fn format_output(&self, output: &str) -> String {
        if self.preserve_full_output() {
            output.to_string()
        } else {
            crate::util::truncate_tool_output(output)
        }
    }

    /// The image payload injected as a synthetic User-role message after this tool's
    /// result block, so a vision-capable model sees an on-disk image natively.
    ///
    /// Overridden by read (a raster read) and chrome (screenshots), each re-encoding an
    /// on-disk PNG/JPEG/WebP into a bounded JPEG data-URI; for read this is the only
    /// decode of a native image file, guarded by a file-magic check rather than by the
    /// annotation wording. A converted document's rasters are decoded as it produces
    /// them, and the computer tool captures through [`Tool::execute_with_payloads`]
    /// instead because its attachment can fail and the result must say so. Safety: the
    /// path must obey the same workspace-boundary validation as [`Tool::execute`] — an
    /// arbitrary `[IMAGE:/path]` reference embedded in model text is never resolved here.
    async fn image_payload(
        &self,
        _ws: &crate::Workspace,
        _args: &serde_json::Value,
    ) -> Option<crate::tools::ImagePayload> {
        None
    }

    /// Execute this tool and return its text with any images it produced — the agent
    /// loop's single entry point for a tool call. The default runs [`Tool::execute`] and
    /// pairs the text with [`Tool::image_payload`] (see
    /// [`crate::tools::with_image_payload`]); a tool whose images are a by-product of its
    /// own execution overrides it, so one execution yields both.
    async fn execute_with_payloads(
        &self,
        ws: &crate::Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<crate::tools::ToolOutput> {
        let text = self.execute(ws, args.clone()).await?;
        Ok(crate::tools::with_image_payload(self, text, ws, &args).await)
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, strum::Display, strum::EnumString,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "lowercase")]
pub(crate) enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

/// The payload of a tool-result message, shared so the writer
/// (`ChatMessage::tool_result`) and reader (`decode_native_history_message`) cannot
/// drift on field names.
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
    /// Prompt tokens not served from cache (prompt_tokens − cached when the provider
    /// reports only the hit side).
    pub cache_miss_tokens: Option<u64>,
    /// Billed cost from `usage.cost` — the invoice amount (OpenRouter-only).
    pub cost: Option<f64>,
    /// Raw `usage.cost_details` breakdown (reference only).
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
    /// Top-level OpenRouter response field, undocumented in the API reference.
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

/// The fallback `max_tokens` when a caller sets none: the *generation* limit (32K
/// output tokens), not the context window.
pub(crate) const DEFAULT_MAX_TOKENS: u32 = 32_000;

/// Operation metadata for per-request LLM stats logging (the `llm_requests` table and
/// the `llm_failures` per-attempt trail in logs.db). Requests without metadata (test
/// doubles, ad-hoc calls) are not logged.
#[derive(Debug, Clone)]
pub(crate) struct ChatRequestMeta {
    /// Purpose tag: "agent" (agent loop incl. direct chat / discovery / sub-agents /
    /// maintainer), "extraction", "summarize", "consolidate", "research_wrap_up";
    /// "agent-continuation" / "summarize-continuation" tag the
    /// operation-level reasoning-only-stop recovery issued by
    /// [`crate::agent::Agent::recover_reasoning_only_stop`] — one row per continuation
    /// operation, with `retry_attempts` carrying the attempt count.
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
    /// Sent as `max_tokens`; omitted when `None` (provider defaults apply).
    pub max_tokens: Option<u32>,
    /// Sent as `reasoning_effort` (xhigh, high, medium, low, minimal); omitted when
    /// `None` (model defaults apply).
    pub reasoning_effort: Option<String>,
    /// Comma-separated provider slugs, sent as the provider routing block when non-empty;
    /// omitted when `None` (OpenRouter defaults apply).
    pub provider_order: Option<String>,
    /// Operation metadata for per-request LLM stats logging.
    pub meta: Option<ChatRequestMeta>,
}

/// Validator for scoped structured extraction: rejecting a parsed value (e.g. a verdict
/// score outside [0,10]) is treated as a parse failure inside the extraction retry loop
/// (re-prompted, fail-closed). `Send + Sync` because extraction runs inside `join_all` futures.
pub(crate) type ExtractionValidator<T> = dyn Fn(&T) -> Result<(), String> + Send + Sync;

#[async_trait]
pub(crate) trait Provider: Send + Sync {
    /// Single-attempt chat for the outer retry loops in [`crate::retry`]. Contract:
    ///
    /// - **No provider-internal retries** — exactly one HTTP request; the outer loop is
    ///   the single retry authority.
    /// - **Idle-timeout semantics** — the 600 s idle timeout
    ///   ([`crate::retry::DEFAULT_IDLE_TIMEOUT`]) bounds the TTFB wait and each
    ///   body-read chunk wait, resetting while data flows; a stalled connection
    ///   mid-body is the truncation signature this hardening targets. No total request
    ///   timeout applies.
    /// - On failure the returned [`ScopedCallError`] carries per-attempt diagnostics
    ///   (classification, error chain, finish_reason) for the retry trail.
    async fn chat_scoped(
        &self,
        request: ChatRequest,
    ) -> Result<ChatResponse, crate::providers::ScopedCallError>;

    async fn warmup(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// A user-defined or community-built capability, defined by a `SKILL.md` in
/// `<workspace>/skills/<name>/` (also `.claude/skills/<name>/` and
/// `.agents/skills/<name>/`), providing instructions and context for the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Skill {
    pub name: String,
    pub description: String,
    pub location: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A field-order mismatch would silently swap one project command for another in
    /// the diagnostics editor.
    #[test]
    fn commands_and_from_buffers_are_consistent() {
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
            ("/start", Some(Start)),
            ("/STart", Some(Start)),
            ("/Start", Some(Start)),
            ("/START", Some(Start)),
            ("/start foo", Some(Start)),
            ("/start   ", Some(Start)),
            ("  /start", Some(Start)),
            ("  /start  ", Some(Start)),
            ("  /start foo  ", Some(Start)),
            ("/clear", Some(Clear)),
            ("/CLEAR", Some(Clear)),
            ("/clear session", Some(Clear)),
            ("  /clear  ", Some(Clear)),
            ("/image_models", Some(ImageModels)),
            ("/IMAGE_MODELS", Some(ImageModels)),
            ("/image_models foo", Some(ImageModels)),
            ("  /image_models  ", Some(ImageModels)),
            ("/video_models", Some(VideoModels)),
            ("/Video_Models", Some(VideoModels)),
            ("/video_models foo", Some(VideoModels)),
            ("/board", Some(Board)),
            ("/BOARD", Some(Board)),
            ("/board foo", Some(Board)),
            ("/archive", Some(Archive)),
            ("/archive foo", Some(Archive)),
            ("/update", Some(Update)),
            ("/UPDATE", Some(Update)),
            ("/update foo", Some(Update)),
            ("  /update  ", Some(Update)),
            ("/agents", Some(Agents)),
            ("/AGENTS", Some(Agents)),
            ("/agents foo", Some(Agents)),
            ("  /agents  ", Some(Agents)),
            // Partial /-prefix matches.
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
            // The workspace switcher, pipeline freeze and maintenance toggle are no longer
            // commands in any spelling — they route as ordinary chat.
            ("/workspace", None),
            ("/WORKSPACE", None),
            ("/workspace foo", None),
            ("/pause", None),
            ("/PAUSE", None),
            ("/pause foo", None),
            ("/unpause", None),
            ("/Unpause foo", None),
            ("/maintenance", None),
            ("/maintenance on", None),
            ("/maintenance off", None),
            ("/maintenance_on", None),
            ("/MAINTENANCE_ON", None),
            ("/maintenance_off", None),
            ("/Maintenance_Off foo", None),
            ("/maintenance_onn", None),
            ("/maintenance_o", None),
            ("/updatex", None),
            ("/updatee", None),
            ("/engineerr", None),
            ("/managr", None),
            ("/artiste", None),
            // Former per-role switch commands, removed.
            ("/manager", None),
            ("/engineer", None),
            ("/artist", None),
            ("/assistant", None),
            ("/ reset", None),
            // Missing slash or empty.
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
