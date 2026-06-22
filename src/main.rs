#![warn(clippy::pedantic)]

use anyhow::Result;
use chrono::{Duration as ChronoDuration, Utc};
use futures_util::FutureExt;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::spawn;
use tracing::{debug, error, info, warn};

use std::future::Future;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use mahbot::channels::{
    send_channel_reply, send_channel_reply_with_buttons, spawn_scoped_typing_task, stop_typing,
    write_incoming_to_broadcast,
};
use mahbot::config::CONFIG;
use mahbot::extraction::{decode_callback, is_callback};
use mahbot::gui::{BOOT_LOG_STORE, Dashboard, JETBRAINS_MONO, Message as DashboardMessage};
use mahbot::manager_queue;
use mahbot::session::{Session, direct_session_key, manager_session_key};
use mahbot::util::UnwrapPoison;
use mahbot::{Agent, Channel, ChannelMessage, Command, Role, Workspace};

/// JetBrainsMono-Regular.ttf embedded for Iced dashboard default font.
const JETBRAINS_MONO_FONT_BYTES: &[u8] = include_bytes!("gui/JetBrainsMono-Regular.ttf");

/// JetBrainsMono-Bold.ttf embedded for header text in the Iced dashboard.
const JETBRAINS_MONO_BOLD_FONT_BYTES: &[u8] = include_bytes!("gui/JetBrainsMono-Bold.ttf");

/// Build a list of inline keyboard buttons for Telegram. Each entry is a
/// `(label, callback_data)` tuple.
fn build_inline_buttons(
    items: impl IntoIterator<Item = (String, String)>,
) -> Vec<serde_json::Value> {
    items
        .into_iter()
        .map(|(label, callback_data)| {
            serde_json::json!({
                "text": label,
                "callback_data": callback_data,
            })
        })
        .collect()
}

/// Resolve the workspace for a user, falling back to a personal workspace
/// if `get_workspace` fails or returns `None`.
async fn resolve_workspace_for_user(msg: &ChannelMessage) -> Workspace {
    if let Ok(Some(ws)) = mahbot::users::get_workspace(&msg.user_name).await {
        ws
    } else {
        // Fallback: construct a bare workspace from the user_name.
        // This should not happen in practice since get_workspace always
        // returns a personal workspace when no shared workspace is selected.
        let path = mahbot::users::personal_workspace_path(&msg.user_name);
        mahbot::users::personal_workspace_struct(&msg.user_name, &path)
    }
}

/// Enrich a message with multimodal transcription and link summarization.
///
/// Multimodal enrichment dispatches by role: full transcription if the role
/// requires it (currently only Artist), otherwise non-multimodal processing
/// (text extraction from photos/audio). Link enrichment always runs,
/// prepending URL summaries to the message content.
async fn enrich_message_for_role(msg: &mut ChannelMessage, role: Role, ws: &Workspace) {
    let strategy = if role.requires_multimodal() {
        mahbot::channels::EnrichmentStrategy::Multimodal {
            workspace_path: Some(ws.as_path().to_path_buf()),
        }
    } else {
        mahbot::channels::EnrichmentStrategy::NonMultimodal
    };
    mahbot::channels::enrich_message(msg, &strategy).await;

    // Link enrichment always runs, regardless of modality.
    let enriched = mahbot::channels::enrich_links(&msg.content).await;
    if enriched != msg.content {
        tracing::info!(
            channel = %msg.source_channel,
            user_name = %msg.user_name,
            "Link enricher: prepended URL summaries to message"
        );
        msg.content = enriched;
    }
}

/// Handle a dynamic option callback (prefixed `__opt__`).
///
/// Parses the callback data, constructs an injected user message
/// (e.g. "mahbot-625 - A"), and routes it to the Manager session,
/// bypassing the user's currently active role.
async fn handle_option_callback(mut msg: ChannelMessage) {
    let Some((ticket_id, label)) = decode_callback(&msg.content) else {
        return;
    };

    // Construct the injected user message
    msg.content = match &ticket_id {
        Some(ticket_id_val) => format!("{ticket_id_val} - {label}"),
        None => label,
    };

    let ws = resolve_workspace_for_user(&msg).await;

    // Route directly to Manager session, bypassing resolve_active_role.
    // Enrichment is skipped — synthetic callback text has no media markers or URLs.
    manager_queue::manager_queue().enqueue(manager_queue::ManagerJob {
        content: msg.content.clone(),
        workspace_name: ws.name.clone(),
        kind: manager_queue::JobKind::UserMessage,
    });
}

/// Build a session key for a given user, role, and source channel.
/// Non-Manager sessions include the channel scope: `{channel}_{user_name}_{role}_{ws}`.
/// Manager sessions stay channel-agnostic: `manager_{ws_name}`.
async fn build_session_key(user_name: &str, role: &Role, source_channel: &str) -> String {
    let ws_name = match mahbot::users::get_workspace(user_name).await {
        Ok(Some(ws)) => ws.name,
        _ => "unknown".to_string(),
    };
    if *role == Role::Manager {
        manager_session_key(&ws_name)
    } else {
        direct_session_key(source_channel, user_name, role.as_str(), &ws_name)
    }
}

/// Run [`bootstrap_mahbot`] and convert panics into `Err` so the dashboard shows
/// a boot error instead of hanging on "Starting…" forever.
async fn bootstrap_mahbot_safe() -> Result<(), String> {
    match AssertUnwindSafe(bootstrap_mahbot()).catch_unwind().await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e.to_string()),
        Err(payload) => Err(format_startup_panic(&*payload)),
    }
}

/// Extract a human-readable message from a panic payload (e.g. from
/// [`catch_unwind`](futures_util::FutureExt::catch_unwind)).
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        msg.to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Format a startup panic payload into an error string for the boot log.
fn format_startup_panic(payload: &(dyn std::any::Any + Send)) -> String {
    format!("Startup panicked: {}", panic_message(payload))
}

/// Async startup for `MahBot` — runs on Iced's Tokio runtime via a boot [`Task`].
async fn bootstrap_mahbot() -> Result<()> {
    mahbot::config::load_or_init().await?;

    let (log_store, log_broadcast) =
        mahbot::logs::init_tracing(&CONFIG.global_storage_root()).await?;

    let _ = mahbot::gui::LOG_BROADCAST.set(log_broadcast);

    mahbot::search_engine::init_global(); // sync — no I/O
    mahbot::ticket_buffer::init_global(); // sync — no I/O
    mahbot::manager_queue::init_global()?;

    tokio::try_join!(
        mahbot::session::init_global(),
        mahbot::workspace::init_global(),
        mahbot::users::init_global(),
        mahbot::board::init_global(),
        mahbot::stats::init_global(),
        mahbot::chat_history::init_global(),
    )?;

    // Config DB must be initialized and loaded before providers,
    // so that API keys and model settings take effect.
    mahbot::config_db::init_global().await?;
    mahbot::config::reload_from_db().await?;
    mahbot::providers::init_global().await?;

    // Clear stale editor dirty flags from previous sessions.
    // Dirty flags are meaningful only within a single session.
    if let Err(e) = mahbot::workspace::store()
        .clear_all_editor_dirty_flags()
        .await
    {
        warn!(error = %e, "Failed to clear editor dirty flags on startup");
    }

    spawn_background_tasks(log_store.clone());

    info!("MahBot initialized — dashboard ready");

    BOOT_LOG_STORE
        .set(log_store.as_ref().clone())
        .map_err(|_| anyhow::anyhow!("BOOT_LOG_STORE already set"))?;

    let admin_target = mahbot::self_update::resolve_admin_telegram_target().await;
    mahbot::self_update::notify_admin("✅ MahBot is back online.", admin_target.as_ref()).await;

    Ok(())
}

/// Global `JoinSet` tracking all background task handles for clean shutdown.
static BACKGROUND_TASKS: std::sync::Mutex<Option<JoinSet<()>>> = std::sync::Mutex::new(None);

/// Spawn a cancellable background task that runs `fut` until the global
/// shutdown token is cancelled. The future must return `()` — use
/// [`race_shutdown`](mahbot::shutdown::race_shutdown) if you need to
/// capture a return value.
///
/// Unlike a bare [`JoinSet::spawn`], this function catches panics inside
/// `fut` and logs them via [`tracing::error!`] so that background tasks
/// don't die silently. The `name` parameter identifies the task in the
/// log message.
fn spawn_cancellable<F>(
    tasks: &mut JoinSet<()>,
    shutdown_token: &CancellationToken,
    name: &'static str,
    fut: F,
) where
    F: Future<Output = ()> + Send + 'static,
{
    let cancel = shutdown_token.clone();
    tasks.spawn(async move {
        tokio::select! {
            result = AssertUnwindSafe(fut).catch_unwind() => {
                if let Err(payload) = result {
                    error!(
                        "Background task panicked [{name}]: {}",
                        panic_message(&*payload),
                    );
                }
            }
            () = cancel.cancelled() => {},
        }
    });
}

fn spawn_background_tasks(log_store: Arc<mahbot::logs::LogStore>) {
    let mut tasks = JoinSet::<()>::new();
    let shutdown_token = mahbot::shutdown::shutdown_token();

    tasks.spawn(cleanup_loop_task("Session cleanup", |cutoff| async move {
        mahbot::session::cleanup_old_transient_sessions(&cutoff).await
    }));

    tasks.spawn(cleanup_loop_task("Log cleanup", {
        let store = log_store;
        move |cutoff| {
            let store = store.clone();
            async move { store.delete_older_than("INFO", &cutoff).await }
        }
    }));

    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "maintainer",
        mahbot::maintainer::run_maintainer_loop(),
    );

    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "archive-cancelled",
        mahbot::board::run_archive_cancelled_loop(),
    );

    // Eagerly initialize search engines for all existing workspaces.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "search-engine-init",
        mahbot::search_engine::init_all_engines(),
    );

    let rx = init_message_pipeline(&mut tasks, &shutdown_token);

    // `handle_messages` runs unconditionally. When no channels are registered,
    // tx is never cloned into a listener, rx is dropped, and the handler exits
    // gracefully (rx.recv() returns `None` immediately).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "message-handler",
        handle_messages(rx),
    );

    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "management",
        mahbot::management::run_management(),
    );

    // Listen for SIGTERM/SIGINT and trigger shutdown — cancels the global
    // token, which the dashboard subscription picks up to close the window.
    tasks.spawn(async move {
        if mahbot::shutdown::wait_for_shutdown_signal().await.is_ok() {
            info!("Received OS signal, triggering shutdown");
            mahbot::shutdown::shutdown();
        }
    });

    // Store handles so shutdown_after_dashboard can await completion.
    {
        let mut guard = BACKGROUND_TASKS.lock().unwrap_poison();
        let _ = guard.insert(tasks);
    }
}

/// Initialize the message pipeline: creates the shared mpsc channel,
/// broadcast channel, channel registry, and spawns Telegram + GUI
/// channel listeners. Returns the receiver half for [`handle_messages`].
fn init_message_pipeline(
    tasks: &mut JoinSet<()>,
    cancel: &CancellationToken,
) -> tokio::sync::mpsc::Receiver<ChannelMessage> {
    // Create the shared message channel before any channel listeners are
    // spawned. All channels push into the same tx; rx is consumed by the
    // single `handle_messages` consumer. `ChannelMessage.source_channel`
    // disambiguates origins.
    let (tx, rx) = tokio::sync::mpsc::channel::<ChannelMessage>(100);

    // Store pipeline tx globally so GuiChannel can forward messages,
    // and keep a local clone for channel listener registration below.
    mahbot::MESSAGE_TX
        .set(tx.clone())
        .expect("MESSAGE_TX already set — should be first init");

    // Clone tx for GuiChannel before it's consumed by the Telegram listener.
    let gui_pipeline_tx = tx.clone();

    // Create the chat broadcast channel (capacity 256 for burst tolerance).
    let (chat_tx, _chat_rx) = tokio::sync::broadcast::channel::<mahbot::ChatEvent>(256);
    mahbot::CHAT_BROADCAST
        .set(chat_tx)
        .expect("CHAT_BROADCAST already set — should be first init");

    // Initialize the channel registry (empty — channels register below).
    let _ = mahbot::CHANNEL_REGISTRY.set(mahbot::ChannelRegistry::default());

    // Only create and start the Telegram channel if a bot token is configured.
    if let Some(token) = CONFIG.telegram_bot_token() {
        use mahbot::channels::telegram::TelegramChannel;
        let channel: Arc<dyn Channel> = Arc::new(TelegramChannel::new(token));
        mahbot::channel_registry().register(Arc::clone(&channel));
        spawn_cancellable(tasks, cancel, "telegram-listener", {
            let channel = Arc::clone(&channel);
            async move {
                let _ = channel.listen(tx).await;
            }
        });
    } else {
        info!("No Telegram bot token configured — running in dashboard-only mode");
    }

    // Always register the GUI channel — even in dashboard-only mode it provides
    // the bridge between the Iced UI and the message pipeline.
    {
        use mahbot::channels::gui::GuiChannel;
        let (gui_channel, gui_tx) = GuiChannel::new();
        mahbot::GUI_MESSAGE_TX
            .set(gui_tx)
            .expect("GUI_MESSAGE_TX already set — should be first init");
        let gui_channel: Arc<dyn Channel> = Arc::new(gui_channel);
        mahbot::channel_registry().register(Arc::clone(&gui_channel));
        spawn_cancellable(tasks, cancel, "gui-listener", {
            let channel = Arc::clone(&gui_channel);
            async move {
                let _ = channel.listen(gui_pipeline_tx).await;
            }
        });
    }

    rx
}

async fn shutdown_after_dashboard() {
    info!("Dashboard window closed — shutting down");
    mahbot::shutdown::shutdown();
    mahbot::registry::AGENT_REGISTRY.shutdown_all();
    mahbot::tools::browser::close_all_browser_sessions().await;

    // Take the JoinSet out of the lock before awaiting (drop guard).
    let maybe_tasks = {
        let mut guard = BACKGROUND_TASKS.lock().unwrap_poison();
        guard.take()
    };

    if let Some(mut tasks) = maybe_tasks {
        while let Some(result) = tasks.join_next().await {
            match result {
                Ok(()) => {}
                Err(e) if e.is_cancelled() => {
                    debug!("background task cancelled during shutdown");
                }
                Err(e) => {
                    warn!("background task panicked: {e}");
                }
            }
        }
    }
}

fn main() -> Result<()> {
    mahbot::shutdown::install_fatal_signal_handlers();

    // Debug subcommand: run SQL query directly, skip all GUI/daemon setup.
    // Must be checked before lock acquisition so the debug tool can query
    // databases while the daemon is running. No tracing init, no lock, no Iced.
    if std::env::args().nth(1).as_deref() == Some("debug") {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        match rt.block_on(mahbot::debug::run_debug()) {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                eprintln!("Error: {e:#}");
                std::process::exit(1);
            }
        }
    }

    // Detect self-update availability before any async work.
    let update_available = mahbot::self_update::is_update_available();

    // Resolve storage root before config init, so we can acquire the lock.
    let storage_root = mahbot::config::default_config_dir()?;

    // Acquire the instance lock before Iced runtime starts.
    // Stored in a global so the update flow can release/re-acquire it.
    mahbot::self_update::acquire_lock(&storage_root)?;

    // Read persisted window state (sync, before Iced runtime starts).
    let window_state = mahbot::gui::read_window_state();

    iced::application(
        move || {
            (
                Dashboard::loading(update_available),
                iced::Task::perform(bootstrap_mahbot_safe(), DashboardMessage::Boot),
            )
        },
        Dashboard::update,
        Dashboard::view,
    )
    .title(Dashboard::title)
    .font(iced_fonts::LUCIDE_FONT_BYTES)
    .font(JETBRAINS_MONO_FONT_BYTES)
    .font(JETBRAINS_MONO_BOLD_FONT_BYTES)
    .default_font(JETBRAINS_MONO)
    .subscription(Dashboard::subscription)
    .theme(Dashboard::theme)
    .window(iced::window::Settings {
        size: iced::Size::new(window_state.width, window_state.height),
        position: window_state.position(),
        min_size: Some(iced::Size::new(800.0, 500.0)),
        ..iced::window::Settings::default()
    })
    .exit_on_close_request(false)
    .run()
    .map_err(|e| anyhow::anyhow!("Iced application error: {e}"))?;

    // Iced dropped its runtime; use a short-lived one for async teardown.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("shutdown runtime: {e}"))?;
    rt.block_on(shutdown_after_dashboard());

    Ok(())
}

/// Background cleanup loop adapter — runs every 10 minutes until cancelled.
async fn cleanup_loop_task<F, Fut>(label: &'static str, cleanup: F)
where
    F: Fn(String) -> Fut + Send + 'static,
    Fut: Future<Output = anyhow::Result<u64>> + Send,
{
    loop {
        if !mahbot::shutdown::sleep_or_shutdown(Duration::from_mins(10)).await {
            break;
        }
        let cutoff = (Utc::now() - ChronoDuration::hours(8)).to_rfc3339();
        match cleanup(cutoff).await {
            Ok(n) if n > 0 => info!(deleted = n, "{label}: deleted old entries"),
            Ok(_) => tracing::debug!("{label}: nothing to delete"),
            Err(e) => warn!(error = %e, "{label} failed"),
        }
    }
}

async fn handle_messages(mut rx: tokio::sync::mpsc::Receiver<ChannelMessage>) {
    let shutdown_token = mahbot::shutdown::shutdown_token();

    loop {
        let mut msg = tokio::select! {
            () = shutdown_token.cancelled() => break,
            msg = rx.recv() => match msg {
                Some(msg) => msg,
                None => break,
            },
        };

        // Handle dynamic option callbacks — route directly to Manager
        // session, bypassing the user's currently active role.
        if is_callback(&msg.content) {
            spawn(handle_option_callback(msg));
            continue;
        }

        if handle_dispatch_command(&mut msg).await {
            continue;
        }

        spawn(process_channel_message(msg));
    }
}

/// Error reply helper — formats `Error: {e}` and sends it.
async fn reply_error(msg: &ChannelMessage, e: impl std::fmt::Display) {
    send_channel_reply(format!("Error: {e}"), msg).await;
}

/// Handle a dispatch-level command. Returns `true` if the message was handled
/// (loop should `continue`), `false` if it should be processed by the agent.
async fn handle_dispatch_command(msg: &mut ChannelMessage) -> bool {
    let cmd = parse(&msg.content);
    let Some(cmd) = cmd else {
        return false;
    };

    let is_full = mahbot::users::is_full_user(&msg.user_name).await;

    // Permission gate: certain admin commands require full-permission users.
    if !is_full && cmd.needs_full_user() {
        send_channel_reply(
            "This command is only available to full-permission users.".to_owned(),
            msg,
        )
        .await;
        return true;
    }

    // Resolve workspace for downstream broadcasts (non-dispatch messages
    // use the same helper via process_channel_message).
    let ws = resolve_workspace_for_user(msg).await;
    msg.workspace = ws.name;

    match cmd {
        Command::NewSession => handle_new_session(msg).await,
        Command::Update => handle_update(msg).await,
        Command::Workspaces => handle_workspaces(msg).await,
        Command::Workspace(arg) => handle_workspace(msg, arg).await,
        Command::Agent(arg) => handle_agent(msg, arg).await,
        Command::Agents => handle_agents(msg).await,
    }
    true
}

async fn handle_new_session(msg: &ChannelMessage) {
    let role = mahbot::users::resolve_active_role(&msg.user_name).await;
    let session_key = build_session_key(&msg.user_name, &role, &msg.source_channel).await;
    let reply = Session::reset(&session_key).await;
    send_channel_reply(reply, msg).await;
}

async fn handle_update(msg: &ChannelMessage) {
    if !mahbot::self_update::is_update_available() {
        send_channel_reply(
            "Self-update is not available on this installation.".to_owned(),
            msg,
        )
        .await;
        return;
    }

    // Ack to requester before update starts — exit(0) prevents any later reply.
    send_channel_reply("Update started, I'll be back shortly…".to_owned(), msg).await;

    // Fire and forget — the update process handles its own lifecycle.
    // Clone what we need from msg before the reference goes out of scope.
    let source_channel = msg.source_channel.clone();
    let reply_target = msg.reply_target.clone();
    let workspace = msg.workspace.clone();
    tokio::spawn(async move {
        if let Err(e) = mahbot::self_update::execute_update().await {
            let err_msg = format!("Self-update failed: {e}");

            // Try to notify the original requester.
            if let Some(channel) = mahbot::channel_registry().get(&source_channel) {
                let reply = mahbot::SendMessage {
                    content: err_msg,
                    recipient: reply_target,
                    reply_markup: None,
                    agent_role: None,
                    workspace,
                };
                let _ = channel.send(&reply).await;
            }
        }
    });
}

async fn handle_workspaces(msg: &ChannelMessage) {
    match mahbot::workspace::get_workspaces().await {
        Ok(workspaces) if workspaces.is_empty() => {
            send_channel_reply(
                "No workspaces configured. Add one via the dashboard.".to_owned(),
                msg,
            )
            .await;
        }
        Ok(workspaces) => {
            let buttons = build_inline_buttons(workspaces.iter().map(|ws| {
                let label = match ws.status.as_str() {
                    "ready" => format!("{} ✅", ws.name),
                    "analyzing" => format!("{} 🔄", ws.name),
                    "failed" => format!("{} ❌", ws.name),
                    "pending" => format!("{} ⏳", ws.name),
                    _ => ws.name.clone(),
                };
                (label, format!("/workspace {}", ws.name))
            }));
            send_channel_reply_with_buttons(
                "📂 **Workspaces:**".to_owned(),
                msg,
                Some(buttons),
                None,
            )
            .await;
        }
        Err(e) => {
            send_channel_reply(format!("Failed to list workspaces: {e}"), msg).await;
        }
    }
}

async fn handle_workspace(msg: &ChannelMessage, arg: Option<String>) {
    match arg {
        Some(name) => match mahbot::users::set_workspace(&msg.user_name, &name).await {
            Ok(ref ws) => {
                send_channel_reply(
                    format!("✅ Active workspace: **{}** (`{}`)", ws.name, ws.path),
                    msg,
                )
                .await;
            }
            Err(e) => reply_error(msg, e).await,
        },
        None => match mahbot::users::get_workspace(&msg.user_name).await {
            Ok(Some(ref ws)) => {
                send_channel_reply(
                    format!(
                        "Current workspace: **{}** (`{}`) — status: {}",
                        ws.name, ws.path, ws.status
                    ),
                    msg,
                )
                .await;
            }
            Ok(None) => {
                send_channel_reply(
                    "No workspace selected. Use /workspaces to pick one.".to_owned(),
                    msg,
                )
                .await;
            }
            Err(e) => reply_error(msg, e).await,
        },
    }
}

async fn handle_agent(msg: &ChannelMessage, arg: Option<String>) {
    match arg {
        Some(role_name) => {
            if !Role::selectable_roles().contains(&role_name.as_str()) {
                send_channel_reply(
                    format!(
                        "❌ Invalid role `{role_name}`. Must be one of: {}",
                        Role::selectable_roles().join(", ")
                    ),
                    msg,
                )
                .await;
                return;
            }
            match mahbot::users::set_active_role(&msg.user_name, &role_name).await {
                Ok(()) => {
                    send_channel_reply(format!("✅ Role changed to: **{role_name}**"), msg).await;
                }
                Err(e) => reply_error(msg, e).await,
            }
        }
        None => match mahbot::users::get_active_role(&msg.user_name).await {
            Ok(Some(ref name)) => {
                send_channel_reply(format!("Current role: **{name}**"), msg).await;
            }
            Ok(None) => {
                send_channel_reply(
                    "No role selected. Defaulting to **analyst**.".to_owned(),
                    msg,
                )
                .await;
            }
            Err(e) => reply_error(msg, e).await,
        },
    }
}

async fn handle_agents(msg: &ChannelMessage) {
    let buttons = build_inline_buttons(
        Role::selectable_roles()
            .iter()
            .map(|name| (name.to_string(), format!("/agent {name}"))),
    );
    send_channel_reply_with_buttons(
        "🤖 **Available roles:**".to_owned(),
        msg,
        Some(buttons),
        None,
    )
    .await;
}

async fn process_channel_message(mut msg: ChannelMessage) {
    tracing::info!(
        "💬 [{}] from {}: {}",
        msg.source_channel,
        msg.user_name,
        mahbot::util::truncate(&msg.content, 80)
    );

    let ws = resolve_workspace_for_user(&msg).await;

    // Populate workspace on the message so downstream broadcasts and
    // chat_history writes carry the correct workspace (non-Manager agent
    // responses go through send_channel_reply_with_buttons, which reads
    // msg.workspace for broadcast filtering).
    msg.workspace = ws.name.clone();

    // Broadcast incoming user message to GUI dashboard and persist to
    // chat_history. Workspace resolution must happen first so the
    // workspace field is correct.
    write_incoming_to_broadcast(&msg).await;

    let role = mahbot::users::resolve_active_role(&msg.user_name).await;

    // Personal workspaces do not support the Manager agent — no board
    // pipeline, no maintainer. If the role is Manager and we're in a
    // personal workspace, fall back to Analyst.
    let effective_role = if role == Role::Manager && mahbot::users::is_personal_workspace(&ws.name)
    {
        Role::Analyst
    } else {
        role
    };

    // Enrichment: multimodal + link summarization — applied after
    // effective_role resolution so the correct role's settings are used.
    enrich_message_for_role(&mut msg, effective_role, &ws).await;

    // Manager messages route through the serialized queue; non-Manager roles
    // use the traditional inline agent dispatch path.
    if effective_role == Role::Manager {
        manager_queue::manager_queue().enqueue(manager_queue::ManagerJob {
            content: msg.content.clone(),
            workspace_name: ws.name.clone(),
            kind: manager_queue::JobKind::UserMessage,
        });
        return;
    }

    // ── Non-Manager inline agent dispatch ─────────────────────────
    // Agent creation, work execution, and reply delivery all happen
    // inline in the calling task.

    let session_key = direct_session_key(
        &msg.source_channel,
        &msg.user_name,
        effective_role.as_str(),
        &ws.name,
    );
    let mut agent = Agent::new(session_key, effective_role, &ws, None);
    let cancel = agent.cancel_token();
    let typing_handle = spawn_scoped_typing_task(
        msg.reply_target.clone(),
        msg.source_channel.clone(),
        cancel.clone(),
    );

    let agent_result = tokio::select! {
        () = cancel.cancelled() => return,
        result = agent.work(&msg.content) => result,
    };

    let response = match agent_result {
        Ok(response) => response,
        Err(e) => {
            tracing::error!("❌ Agent error: {e}");
            format!("⚠️ `{e}`")
        }
    };

    if agent.is_cancelled() {
        stop_typing(typing_handle).await;
        return;
    }

    send_channel_reply_with_buttons(response, &msg, None, Some(effective_role.to_string())).await;

    cancel.cancel();
    stop_typing(typing_handle).await;
}

/// Parse a channel message's content and classify it.
///
/// Command names are case-insensitive. Arguments preserve their original case
/// (e.g. workspace names) except for `/agent`, where the argument is lowercased
/// because role names are lowercase.
#[must_use]
pub fn parse(content: &str) -> Option<Command> {
    let cmd_line = content.trim().strip_prefix('/')?;
    let (cmd, arg) = cmd_line
        .split_once(' ')
        .map_or((cmd_line, ""), |(c, a)| (c, a.trim()));
    let arg = (!arg.is_empty()).then(|| arg.to_string());

    match cmd.to_ascii_lowercase().as_str() {
        "new" | "start" => Some(Command::NewSession),
        "update" => Some(Command::Update),
        "workspaces" => Some(Command::Workspaces),
        "workspace" => Some(Command::Workspace(arg)),
        "agents" => Some(Command::Agents),
        "agent" => Some(Command::Agent(arg.map(|a| a.to_ascii_lowercase()))),
        _ => None,
    }
}
