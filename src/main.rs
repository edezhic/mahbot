#![warn(clippy::pedantic)]

use anyhow::Result;
use chrono::{Duration as ChronoDuration, Utc};
use futures_util::FutureExt;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::spawn;
use tracing::{debug, error, info, warn};

use std::borrow::Cow;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use mahbot::channels::telegram::{decode_action, decode_callback};
use mahbot::channels::{
    send_channel_reply, spawn_scoped_typing_task, stop_typing, write_incoming_to_broadcast,
};
use mahbot::config::CONFIG;
use mahbot::gui::{BOOT_LOG_STORE, Dashboard, JETBRAINS_MONO, Message as DashboardMessage};
use mahbot::manager_queue;
use mahbot::parse_bot_command;
use mahbot::session::{Session, direct_session_key, session_key};
use mahbot::util::UnwrapPoison;
use mahbot::{Agent, BotCommand, Channel, ChannelMessage, Role, Workspace};
/// JetBrainsMono-Regular.ttf embedded for Iced dashboard default font.
const JETBRAINS_MONO_FONT_BYTES: &[u8] = include_bytes!("gui/JetBrainsMono-Regular.ttf");

/// JetBrainsMono-Bold.ttf embedded for header text in the Iced dashboard.
const JETBRAINS_MONO_BOLD_FONT_BYTES: &[u8] = include_bytes!("gui/JetBrainsMono-Bold.ttf");

/// Resolve the workspace for a user, falling back to a personal workspace
/// if `get_workspace` fails or returns `None`.
async fn resolve_workspace_for_user(msg: &ChannelMessage) -> Workspace {
    match mahbot::users::get_workspace(&msg.user_name).await {
        Ok(Some(ws)) => return ws,
        Ok(None) => {
            tracing::warn!(
                user_name = %msg.user_name,
                "workspace resolution: selected_workspace points to non-existent workspace; \
                 falling back to personal workspace",
            );
        }
        Err(e) => {
            tracing::warn!(
                user_name = %msg.user_name,
                error = %e,
                "workspace resolution: database error; falling back to personal workspace",
            );
        }
    }
    personal_workspace(msg)
}

fn personal_workspace(msg: &ChannelMessage) -> Workspace {
    let path = mahbot::users::personal_workspace_path(&msg.user_name);
    mahbot::users::personal_workspace_struct(&msg.user_name, &path)
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
    if let Cow::Owned(s) = enriched {
        tracing::info!(
            channel = %msg.channel,
            user_name = %msg.user_name,
            "Link enricher: prepended URL summaries to message"
        );
        msg.content = s;
    }
}

/// Handle a dynamic option callback (prefixed `__opt__`).
///
/// Parses the callback data, constructs an injected user message
/// (e.g. "mahbot-123 - A"), and routes it to the Manager session,
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
    manager_queue::manager_queue().enqueue_user_message(msg.content, ws.name);
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

/// Format a startup panic payload into an error string for the boot log.
fn format_startup_panic(payload: &(dyn std::any::Any + Send)) -> String {
    format!("Startup panicked: {}", mahbot::util::panic_message(payload))
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

    mahbot::turso::init_all_stores().await?;

    // Config DB must be loaded before providers, so that API keys
    // and model settings take effect.
    mahbot::config::reload_from_db().await?;
    mahbot::providers::init_global().await?;

    BOOT_LOG_STORE
        .set(log_store.as_ref().clone())
        .map_err(|_| anyhow::anyhow!("BOOT_LOG_STORE already set"))?;

    spawn_background_tasks(log_store.clone());

    info!("MahBot initialized — dashboard ready");

    let admin_target = mahbot::self_update::resolve_admin_telegram_target().await;
    tokio::spawn(async move {
        mahbot::self_update::notify_admin("✅ MahBot is back online.", admin_target.as_deref())
            .await;
    });

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
                        mahbot::util::panic_message(&*payload),
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

    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "session-cleanup",
        run_cleanup_loop("Session cleanup", |cutoff| async move {
            mahbot::session::cleanup_old_transient_sessions(&cutoff).await
        }),
    );

    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "log-cleanup",
        run_cleanup_loop("Log cleanup", {
            let store = log_store;
            move |cutoff| {
                let store = store.clone();
                async move { store.delete_older_than("INFO", &cutoff).await }
            }
        }),
    );

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

    // `run_message_dispatch_loop` runs unconditionally. When no channels are registered,
    // tx is never cloned into a listener, rx is dropped, and the handler exits
    // gracefully (rx.recv() returns `None` immediately).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "message-handler",
        run_message_dispatch_loop(rx),
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
        let result = AssertUnwindSafe(mahbot::shutdown::wait_for_shutdown_signal())
            .catch_unwind()
            .await;
        match result {
            Ok(Ok(())) => {
                info!("Received OS signal, triggering shutdown");
                mahbot::shutdown::shutdown();
            }
            Ok(Err(e)) => {
                error!("Signal handler failed to set up: {e}");
            }
            Err(payload) => {
                error!(
                    "Signal handler panicked: {}",
                    mahbot::util::panic_message(&*payload),
                );
            }
        }
    });

    // Periodic WAL checkpoint as defense-in-depth against data loss from
    // crashes or missed shutdown-path checkpoints.
    spawn_cancellable(&mut tasks, &shutdown_token, "auto-checkpoint", async {
        loop {
            if !mahbot::shutdown::sleep_or_shutdown(Duration::from_mins(5)).await {
                break;
            }
            mahbot::checkpoint::checkpoint_all_databases().await;
            mahbot::checkpoint::verify_all_databases().await;
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
/// channel listeners. Returns the receiver half for [`run_message_dispatch_loop`].
fn init_message_pipeline(
    tasks: &mut JoinSet<()>,
    cancel: &CancellationToken,
) -> tokio::sync::mpsc::Receiver<ChannelMessage> {
    // Create the shared message channel before any channel listeners are
    // spawned. All channels push into the same tx; rx is consumed by the
    // single `run_message_dispatch_loop` consumer. `ChannelMessage.channel`
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
async fn run_cleanup_loop<F, Fut>(label: &'static str, cleanup: F)
where
    F: Fn(String) -> Fut + Send + 'static,
    Fut: Future<Output = Result<u64>> + Send,
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

async fn run_message_dispatch_loop(mut rx: tokio::sync::mpsc::Receiver<ChannelMessage>) {
    let shutdown_token = mahbot::shutdown::shutdown_token();

    loop {
        let msg = tokio::select! {
            () = shutdown_token.cancelled() => break,
            msg = rx.recv() => match msg {
                Some(msg) => msg,
                None => break,
            },
        };

        // Handle dynamic option callbacks — route directly to Manager
        // session, bypassing the user's currently active role.
        if decode_callback(&msg.content).is_some() {
            spawn(handle_option_callback(msg));
            continue;
        }

        // Handle action callbacks (__act__ prefix) — route to inline handler
        // that updates config / clears session without involving the Manager agent.
        if decode_action(&msg.content).is_some() {
            handle_action_callback(msg).await;
            continue;
        }

        if handle_bot_command(&msg).await {
            continue;
        }

        spawn(process_channel_message(msg));
    }
}

/// Handle Telegram bot text commands (`/start`, `/clear`, `/models`).
/// Returns `true` if the message was handled (loop should `continue`),
/// `false` if it should be processed by the agent pipeline.
///
/// Only Telegram gets command handling; GUI and other channels route
/// these messages as normal text (returns false to fall through to
/// `process_channel_message`).
async fn handle_bot_command(msg: &ChannelMessage) -> bool {
    let Some(cmd) = parse_bot_command(&msg.content) else {
        return false;
    };

    if msg.channel != "telegram" {
        return false;
    }

    match cmd {
        BotCommand::Start => handle_start_command(msg).await,
        BotCommand::Clear => handle_clear_command(msg).await,
        BotCommand::Models => handle_models_command(msg).await,
    }
    true
}

/// Handle `/start` command for Telegram — sends a minimal welcome message
/// listing available commands (no inline keyboard).
async fn handle_start_command(msg: &ChannelMessage) {
    let reply = mahbot::SendMessage {
        content: "\u{1F916} Welcome to MahBot!\n\n\
                  Available commands:\n\
                  /start — Show this message\n\
                  /clear — Reset your session\n\
                  /models — Select generation models"
            .to_string(),
        recipient: msg.reply_target.clone(),
        reply_markup: None,
    };
    if let Some(channel) = mahbot::channel_registry().get(&msg.channel) {
        let _ = channel.send(&reply).await;
    }
}

/// Handle `/clear` command for Telegram — deletes the current session.
async fn handle_clear_command(msg: &ChannelMessage) {
    let (ws, role) = tokio::join!(
        resolve_workspace_for_user(msg),
        mahbot::users::resolve_active_role(&msg.user_name),
    );
    let sk = session_key(&msg.channel, &msg.user_name, role.as_str(), &ws.name);
    let reply = Session::delete(&sk).await;
    send_channel_reply(reply, msg, None).await;
}

/// Handle `/models` command for Telegram — shows model selection keyboard
/// (image and video models for Artist, clear session button for other roles).
async fn handle_models_command(msg: &ChannelMessage) {
    let reply_markup = build_models_keyboard(msg).await;
    let reply = mahbot::SendMessage {
        content: "Select a model:".to_string(),
        recipient: msg.reply_target.clone(),
        reply_markup: Some(reply_markup),
    };
    // Send directly through the channel so the inline_keyboard structure
    // (rows of buttons) is preserved exactly — send_channel_reply does not
    // support inline keyboards, so this path bypasses it for multi-row
    // replies like the /models menu.
    if let Some(channel) = mahbot::channel_registry().get(&msg.channel) {
        let _ = channel.send(&reply).await;
    }
}

/// Build inline keyboard for model selection based on the user's current role.
///
/// Returns the full Telegram `inline_keyboard` JSON array, where each element
/// is a row (list of buttons in that row). Each button gets its own row.
///
/// For Artist: shows image model selection, video model selection, and clear session.
/// For other roles: shows only clear session.
async fn build_models_keyboard(msg: &ChannelMessage) -> serde_json::Value {
    let role = mahbot::users::resolve_active_role(&msg.user_name).await;

    let clear_session_btn = serde_json::json!([{
        "text": "Clear session",
        "callback_data": "__act__clear_session|",
    }]);

    let rows = if role == Role::Artist {
        let mut rows: Vec<serde_json::Value> = Vec::new();

        // Image model buttons — each on its own row
        build_model_button_rows(
            &mut rows,
            &CONFIG.image_gen_models(),
            &CONFIG.image_gen_model(),
            "__act__set_image_model",
        );

        // Video model buttons — each on its own row
        build_model_button_rows(
            &mut rows,
            &CONFIG.video_gen_models(),
            &CONFIG.video_gen_model(),
            "__act__set_video_model",
        );

        rows.push(clear_session_btn);
        rows
    } else {
        vec![clear_session_btn]
    };

    serde_json::json!({ "inline_keyboard": rows })
}

/// Push one row per model to `rows`, marking the active model with ✓.
fn build_model_button_rows(
    rows: &mut Vec<serde_json::Value>,
    models: &[String],
    active_model: &str,
    action_prefix: &str,
) {
    for model in models {
        let label = if model == active_model {
            format!("\u{2713} {model}")
        } else {
            model.clone()
        };
        rows.push(serde_json::json!([{
            "text": label,
            "callback_data": format!("{action_prefix}|{model}"),
        }]));
    }
}

/// Handle an action callback (`__act__` prefix).
///
/// Actions are processed inline without involving the Manager agent queue.
async fn handle_action_callback(msg: ChannelMessage) {
    let Some((action, payload)) = decode_action(&msg.content) else {
        tracing::warn!("Malformed __act__ callback data: {}", &msg.content);
        return;
    };

    match action.as_str() {
        "set_image_model" => {
            handle_set_model_action(&msg, &payload, "image_gen_model", "Image").await;
        }
        "set_video_model" => {
            handle_set_model_action(&msg, &payload, "video_gen_model", "Video").await;
        }
        "clear_session" => {
            // Acknowledge callback silently first (dismiss spinner)
            answer_telegram_callback(&msg, None).await;

            // Resolve workspace and role in parallel, then reset session
            let (ws, role) = tokio::join!(
                resolve_workspace_for_user(&msg),
                mahbot::users::resolve_active_role(&msg.user_name),
            );
            let sk = session_key(&msg.channel, &msg.user_name, role.as_str(), &ws.name);
            let reply = Session::delete(&sk).await;
            send_channel_reply(reply, &msg, None).await;
        }
        _ => {
            // Always acknowledge callback queries to dismiss the Telegram
            // loading spinner, even for unknown actions.
            answer_telegram_callback(&msg, None).await;
            tracing::warn!(action = %action, "Unknown __act__ action — ignoring");
        }
    }
}

/// Common handler for setting a model config field via callback action.
///
/// Validates payload, writes to `config_kv` table, updates the in-memory
/// config, and acknowledges the callback with a toast.
async fn handle_set_model_action(
    msg: &ChannelMessage,
    payload: &str,
    config_key: &str,
    display_name: &str,
) {
    if payload.is_empty() {
        tracing::warn!(config_key, "{config_key} action with empty payload");
        answer_telegram_callback(msg, Some("No model specified.".to_string())).await;
        return;
    }
    // Direct-write to config_kv table (bypasses save_and_reload which
    // triggers provider warmup — unnecessary for a model name change).
    let store = mahbot::config_db::store();
    if let Err(e) = store.set_kv(config_key, payload).await {
        tracing::error!(config_key, error = %e, "Failed to save {config_key}");
        answer_telegram_callback(msg, Some(format!("Failed to save model: {e}"))).await;
        return;
    }
    // Lightweight in-memory update — no DB read, no provider warmup
    let _ = CONFIG.set_string_field(config_key, payload);

    answer_telegram_callback(
        msg,
        Some(format!("{display_name} generation model set to: {payload}")),
    )
    .await;
}

/// Acknowledge a Telegram callback query with an optional toast message.
/// If the message doesn't have a `callback_query_id` (non-Telegram channel),
/// this is a no-op.
async fn answer_telegram_callback(msg: &ChannelMessage, toast: Option<String>) {
    let Some(cq_id) = &msg.callback_query_id else {
        return;
    };
    if let Some(channel) = mahbot::channel_registry().get("telegram")
        && let Some(tc) = channel
            .as_any()
            .downcast_ref::<mahbot::channels::telegram::TelegramChannel>()
    {
        tc.answer_callback_query(cq_id, toast.as_deref()).await;
    }
}

async fn process_channel_message(mut msg: ChannelMessage) {
    tracing::info!(
        "💬 [{}] from {}: {}",
        msg.channel,
        msg.user_name,
        mahbot::util::truncate(&msg.content, 80)
    );

    let (ws, role) = tokio::join!(
        resolve_workspace_for_user(&msg),
        mahbot::users::resolve_active_role(&msg.user_name),
    );

    // Populate workspace on the message so downstream broadcasts and
    // chat_history writes carry the correct workspace (non-Manager agent
    // responses go through send_channel_reply, which reads
    // msg.workspace for broadcast filtering).
    msg.workspace = ws.name.clone();

    // Broadcast incoming user message to GUI dashboard + persist to
    // chat_history, and mirror GUI messages to the user's Telegram
    // chats as blockquotes. Workspace resolution must happen first so
    // the workspace field is correct. Both run before enrichment so
    // the original user-typed text is preserved.
    tokio::join!(
        write_incoming_to_broadcast(&msg),
        mahbot::channels::mirror_gui_message_to_telegram(&msg),
    );

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
        manager_queue::manager_queue().enqueue_user_message(msg.content, ws.name);
        return;
    }

    // ── Non-Manager inline agent dispatch ─────────────────────────
    // Agent creation, work execution, and reply delivery all happen
    // inline in the calling task.

    let session_key = direct_session_key(
        &msg.channel,
        &msg.user_name,
        effective_role.as_str(),
        &ws.name,
    );
    let mut agent = Agent::new(session_key, effective_role, &ws, None);
    let cancel = agent.cancel_token();
    let typing_stop = CancellationToken::new();
    let typing_handle = spawn_scoped_typing_task(
        msg.reply_target.clone(),
        msg.channel.clone(),
        typing_stop.clone(),
    );

    // Core work block — cancel/error paths `break 'work` to skip
    // reply delivery, then cleanup runs unconditionally below.
    'work: {
        let agent_result = tokio::select! {
            () = cancel.cancelled() => { break 'work; },
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
            break 'work;
        }

        send_channel_reply(response, &msg, Some(effective_role.to_string())).await;
    }

    // Single cleanup — always runs, even on cancellation
    typing_stop.cancel();
    stop_typing(typing_handle).await;
}
