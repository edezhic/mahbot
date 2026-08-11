#![warn(clippy::pedantic)]

use anyhow::Result;
use chrono::{Duration as ChronoDuration, Utc};
use futures_util::FutureExt;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::task::spawn;
use tracing::{debug, error, info, warn};

use std::borrow::Cow;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use mahbot::channels::broadcast_and_persist_incoming_message;
use mahbot::channels::telegram::{decode_action, decode_callback, user_command_entries};
use mahbot::config::CONFIG;
use mahbot::gui::{BOOT_LOG_STORE, Dashboard, JETBRAINS_MONO, Message as DashboardMessage};
use mahbot::message_router;
use mahbot::parse_bot_command;
use mahbot::session::clear_session;
use mahbot::util::UnwrapPoison;
use mahbot::{BotCommand, Channel, ChannelMessage, Role, Workspace};
/// JetBrainsMono-Regular.ttf embedded for Iced dashboard default font.
const JETBRAINS_MONO_FONT_BYTES: &[u8] = include_bytes!("gui/JetBrainsMono-Regular.ttf");

/// JetBrainsMono-Bold.ttf embedded for header text in the Iced dashboard.
const JETBRAINS_MONO_BOLD_FONT_BYTES: &[u8] = include_bytes!("gui/JetBrainsMono-Bold.ttf");

/// INFO-log retention window (hours): the log-cleanup loop deletes INFO
/// entries older than this. Independent of the session-purge cutoff.
const LOG_RETENTION_HOURS: i64 = 8;

/// Handle a dynamic option callback (prefixed `__opt__`).
///
/// Constructs an injected user message (e.g. "mahbot-123 - A") from the
/// already-decoded callback data, and routes it to the Manager session,
/// bypassing the user's currently active role.
async fn handle_option_callback(mut msg: ChannelMessage, decoded: (Option<String>, String)) {
    let (ticket_id, label) = decoded;

    // Construct the injected user message
    msg.content = match &ticket_id {
        Some(ticket_id_val) => format!("{ticket_id_val} - {label}"),
        None => label,
    };

    let ws = mahbot::users::resolve_workspace_for_user_name(&msg.user_name).await;

    // Route directly to Manager session, bypassing resolve_active_role.
    // Enrichment is skipped — synthetic callback text has no media markers or URLs.
    message_router::route_user_message(
        msg.content,
        ws.name,
        msg.user_name,
        msg.channel,
        Role::Manager,
        None,
    )
    .await;
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
    mahbot::message_router::init_global()?;
    mahbot::audio::voice::init_global()?;
    mahbot::audio::tts::init_global()?;

    mahbot::turso::init_all_stores().await?;

    // Config DB must be loaded before providers, so that API keys
    // and model settings take effect.
    mahbot::config::reload_from_db().await?;
    mahbot::providers::init_global().await?;

    // Try to load TTS models from cache; if not available, spawn background download.
    // Only run when TTS is enabled in config to avoid unnecessary ~400 MB download.
    if mahbot::audio::tts::is_config_enabled() && !mahbot::audio::tts::try_load_cached() {
        mahbot::audio::tts::spawn_download();
    }

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

#[allow(clippy::too_many_lines)]
fn spawn_background_tasks(log_store: Arc<mahbot::logs::LogStore>) {
    let mut tasks = JoinSet::<()>::new();
    let shutdown_token = mahbot::shutdown::shutdown_token();

    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "session-cleanup",
        run_cleanup_loop(
            "Session cleanup",
            mahbot::jobs::PURGE_CUTOFF_HOURS,
            |cutoff| async move {
                // Stale-job purge FIRST (cross-DB orchestrator: board rollback +
                // sessions DELETE) — protected sessions only become eligible for
                // the TTL guard after the purge cascade removes their job rows.
                let purged = mahbot::jobs::purge_stale_jobs(&cutoff).await?;
                let cleaned = mahbot::session::cleanup_old_transient_sessions(&cutoff).await?;
                Ok(purged + cleaned)
            },
        ),
    );

    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "log-cleanup",
        run_cleanup_loop("Log cleanup", LOG_RETENTION_HOURS, {
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

    // Dead-session recovery: detects user-agent sessions that failed silently
    // (user sent a message, no agent responded, no agent is running) and
    // automatically re-triggers the agent.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "dead-session-recovery",
        mahbot::session::dead_session::run_dead_session_recovery_loop(),
    );

    // Nightly workspace re-analysis: checks for new git commits and
    // triggers rediscover during the 3-4 AM local time window.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "nightly-check",
        mahbot::workspace::run_nightly_check_loop(),
    );

    // Live-WAL/tshm artifact guard: periodically inspects the store file set
    // and warns when the daemon's WAL fd is orphaned (live frames advertised
    // by .tshm but an empty on-disk -wal). Detection only — never touches the
    // files.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "wal-guard",
        mahbot::wal_guard::run_wal_guard_loop(),
    );

    // Eagerly initialize search engines for all existing workspaces.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "search-engine-init",
        mahbot::search_engine::init_all_engines(),
    );

    // Browser daemon health watchdog: classifies chrome-use health from the
    // daemon-free status and auto-restarts with bounded backoff when it is
    // down; wedges surface on real browser calls (fail-fast) and wake this
    // watchdog (browser relay daemon only — never the mahbot service itself).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "browser-daemon",
        mahbot::tools::browser_daemon::run_watchdog(),
    );

    // Voice assistant pipeline — runs in background, manages wake word
    // detection, command recording, transcription, and routing.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "voice-pipeline",
        mahbot::audio::voice::run_voice_pipeline(),
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

    // Listen for SIGTERM/SIGINT and drive the two-signal drain protocol.
    // First signal begins the drain (wait_for_shutdown_signal calls
    // drain_begin and keeps listening); the task returns only on a SECOND
    // signal, which force-cancels the drain. Clean drain completion is
    // driven by the drain-watch task below (fires the token when no
    // in-flight agents or orchestrator calls remain).
    tasks.spawn(async move {
        let result = AssertUnwindSafe(mahbot::shutdown::wait_for_shutdown_signal())
            .catch_unwind()
            .await;
        match result {
            Ok(Ok(())) => {
                info!("Second signal received — force-cancelling drain");
                mahbot::shutdown::force_cancel();
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

    // Drain-watch: while the drain flag is set, poll the agent registry AND
    // the non-agent call registry. Drain-cut ticket_stage/ask rounds
    // intentionally leave their jobs status='launched' for boot resume
    // (decision 20), so a jobs-table count cannot reach zero in the common
    // drain — and even research jobs, which DO terminalize mid-drain via the
    // partial-report path, tell us nothing about cut rounds — so the
    // registries are the authoritative in-flight signal (orchestrator-only
    // LLM calls — ask consolidation, research synthesis — are tracked in
    // NON_AGENT_CALLS; the research orchestrator holds a whole-run guard).
    // Clean exit when both empty; force-cancel stragglers at the 10-minute
    // cap (in-flight ops with >10 min remaining budget are guaranteed-aborted
    // and boot-resume via status=running).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "drain-watch",
        mahbot::jobs::run_drain_watch(),
    );

    // Periodic WAL checkpoint as hygiene: compact committed frames and bound
    // WAL growth/reopen cost (committed data is fsync-durable at COMMIT
    // regardless of checkpoints). Non-truncating (PASSIVE) below the WAL-size
    // cap — TRUNCATE resets the shared WAL frame index, which is the
    // two-writer corruption vector under live connections, so the periodic
    // loop avoids it (see checkpoint::periodic_checkpoint_all_databases).
    spawn_cancellable(&mut tasks, &shutdown_token, "auto-checkpoint", async {
        loop {
            if !mahbot::shutdown::sleep_or_shutdown_or_drain(Duration::from_mins(5)).await {
                break;
            }
            // The self-update path checkpoints all stores itself and cancels
            // the shutdown token right after; skip a periodic round that would
            // land inside the update handoff window. A round already in flight
            // when the update starts can still finish — bounded to one
            // iteration, since the next sleep breaks on the cancelled token.
            if mahbot::self_update::update_is_finalizing() {
                break;
            }
            mahbot::checkpoint::periodic_checkpoint_all_databases().await;
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

    // Spawn the TTS listener which subscribes to CHAT_BROADCAST and triggers
    // audio playback for matching agent responses.
    mahbot::audio::tts::init_listener();

    // Initialize the channel registry (empty — channels register below).
    let _ = mahbot::CHANNEL_REGISTRY.set(mahbot::ChannelRegistry::default());

    // Only create and start the Telegram channel if a bot token is configured.
    if let Some(token) = CONFIG.telegram_bot_token() {
        use mahbot::channels::telegram::TelegramChannel;
        let channel: std::sync::Arc<TelegramChannel> =
            std::sync::Arc::new(TelegramChannel::new(token));
        // Register bot commands with Telegram API (fire-and-forget, non-blocking).
        tokio::spawn({
            let tc = std::sync::Arc::clone(&channel);
            async move {
                tc.set_my_commands().await;
            }
        });
        let channel: Arc<dyn Channel> = channel;
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

    // Always register the voice channel so the message routing system can
    // resolve the "voice" channel name when delivering agent responses.
    // There is no listener — the voice pipeline runs its own mic-capture
    // loop managed by `run_voice_pipeline`.
    mahbot::channels::voice::register_global();

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

    // Single-writer checkpoint: the iced runtime is gone, so no background
    // writer is live. Relocated here from save_and_exit (which ran it while
    // background writers were still active — contradicting single-writer).
    mahbot::checkpoint::checkpoint_all_databases().await;
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

    // Hidden grep-engine subcommand: served by the shell tool's read-only
    // grep interception. Also dispatched before lock acquisition — the daemon
    // holds the flock, so a normally-dispatched second process would fail.
    #[cfg(unix)]
    if std::env::args().nth(1).as_deref() == Some("__grep-engine") {
        let code = mahbot::run_grep_engine(&std::env::args().skip(2).collect::<Vec<_>>());
        std::process::exit(code);
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

/// Background cleanup loop adapter — runs every 10 minutes until cancelled
/// or the graceful drain begins (stale purge must not race the drain).
/// `cutoff_hours` is the retention window for this specific cleanup policy.
async fn run_cleanup_loop<F, Fut>(label: &'static str, cutoff_hours: i64, cleanup: F)
where
    F: Fn(String) -> Fut + Send + 'static,
    Fut: Future<Output = Result<u64>> + Send,
{
    loop {
        if !mahbot::shutdown::sleep_or_shutdown_or_drain(Duration::from_mins(10)).await {
            break;
        }
        let cutoff = (Utc::now() - ChronoDuration::hours(cutoff_hours)).to_rfc3339();
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
        if let Some(decoded) = decode_callback(&msg.content) {
            spawn(handle_option_callback(msg, decoded));
            continue;
        }

        // Handle action callbacks (__act__ prefix) — route to a handler that
        // updates config / clears session without involving the Manager agent.
        // Spawned (like __opt__ callbacks) so a slow catalog validation in
        // set_image_model never stalls the shared dispatch loop.
        if let Some(decoded) = decode_action(&msg.content) {
            spawn(handle_action_callback(msg, decoded));
            continue;
        }

        if handle_bot_command(&msg).await {
            continue;
        }

        spawn(process_channel_message(msg));
    }
}

/// Handle Telegram bot text commands. Returns `true` if the message was
/// handled (loop should `continue`), `false` if it should be processed by
/// the agent pipeline.
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
        BotCommand::Clear => handle_clear_session(msg).await,
        // Artist-gated commands: denial when Artist is not in the user's pool.
        BotCommand::ImageModels | BotCommand::VideoModels => {
            if mahbot::users::role_pool(&msg.user_name)
                .await
                .contains(&Role::Artist)
            {
                handle_models_command(msg, cmd == BotCommand::ImageModels).await;
            } else {
                send_telegram_reply(
                    msg,
                    "This command is only available to Artist users.".to_string(),
                )
                .await;
            }
        }
        // Role-switch commands: pool-gated.
        BotCommand::SwitchRole(role) => handle_role_switch(msg, role).await,
        // Admin-gated commands: denial for non-admin users.
        BotCommand::Board
        | BotCommand::Archive
        | BotCommand::Pause
        | BotCommand::Unpause
        | BotCommand::Maintenance
        | BotCommand::MaintenanceOn
        | BotCommand::MaintenanceOff => {
            if mahbot::users::is_admin(&msg.user_name).await {
                handle_admin_command(msg, cmd).await;
            } else {
                send_telegram_reply(
                    msg,
                    "This command is only available to admin users.".to_string(),
                )
                .await;
            }
        }
    }
    true
}

/// Send a plain-text reply directly on the Telegram channel (no router
/// broadcast/persist — used for command responses, not agent replies).
async fn send_telegram_reply(msg: &ChannelMessage, content: String) {
    let _ = mahbot::channels::telegram::send_direct(&msg.reply_target, content, None).await;
}

/// Handle a role-switch command (`/role_name`) — pool-gated, persists the
/// new active role and confirms after a successful switch.
async fn handle_role_switch(msg: &ChannelMessage, role: Role) {
    if !mahbot::users::role_pool(&msg.user_name)
        .await
        .contains(&role)
    {
        send_telegram_reply(
            msg,
            format!(
                "Role '{}' is not in your allowed roles — ask an admin to add it.",
                role.as_str()
            ),
        )
        .await;
        return;
    }
    match mahbot::users::switch_active_role(&msg.user_name, role).await {
        Ok(()) => {
            let text = format!("Active role switched to {}.", role.display_label());
            // Telegram-only pin flow; detached so an unreachable API can't
            // stall the dispatch loop — fire-and-forget, so a shutdown may
            // drop the notification (the switch is already persisted).
            let Some(channel) = mahbot::channel_registry().get("telegram") else {
                return;
            };
            let reply_target = msg.reply_target.clone();
            tokio::spawn(async move {
                let tc = channel
                    .as_any()
                    .downcast_ref::<mahbot::channels::telegram::TelegramChannel>()
                    .expect("registered telegram channel");
                tc.send_role_switch_notification(&reply_target, &text).await;
            });
        }
        Err(e) => send_telegram_reply(msg, format!("Failed to switch role: {e}")).await,
    }
}

/// Handle `/start` command for Telegram — sends a per-user welcome message
/// listing the commands available to the current role/admin state (no inline
/// keyboard).
async fn handle_start_command(msg: &ChannelMessage) {
    let mut lines = vec![
        "\u{1F916} Welcome to MahBot!\n\nAvailable commands:".to_string(),
        "/start — Show this message".to_string(),
    ];
    for (cmd, desc) in user_command_entries(&msg.user_name).await {
        lines.push(format!("/{cmd} — {desc}"));
    }
    send_telegram_reply(msg, lines.join("\n")).await;
}

/// Handle session clearing for `/clear` and the "Clear session" inline button —
/// deletes the current session and confirms via the canonical delivery path.
async fn handle_clear_session(msg: &ChannelMessage) {
    // Clear the session the user actually talks to: the same (role, workspace)
    // resolution as routing — DB-selected workspace, pool-clamped active role
    // with Analyst fallback, personal-workspace Manager→Analyst remap, and
    // Assistant/Artist pinning.
    let (effective_role, ws) = mahbot::users::resolve_session_target(&msg.user_name).await;
    let reply = clear_session(
        &msg.channel,
        &msg.user_name,
        effective_role.as_str(),
        &ws.name,
    )
    .await;
    deliver_clear_reply(&reply, msg, &ws, effective_role).await;
}

/// Deliver a session-clear confirmation via the router's raw `reply_target`
/// path (broadcast + persist + transport). The caller passes the already
/// effective role (pool-clamped, personal-workspace Manager→Analyst remap
/// applied) so the confirmation bubble matches agent responses.
async fn deliver_clear_reply(
    reply: &str,
    msg: &ChannelMessage,
    ws: &Workspace,
    effective_role: Role,
) {
    message_router::deliver_unregistered_user_response(
        reply,
        &message_router::AgentJob {
            content: reply.to_string(),
            workspace_name: ws.name.clone(),
            user_name: msg.user_name.clone(),
            channel: msg.channel.clone(),
            kind: message_router::JobKind::UserMessage,
            role: effective_role,
            reply_target: Some(msg.reply_target.clone()),
            pending_job_id: None,
        },
        &effective_role,
    )
    .await;
}

/// Handle `/image_models` / `/video_models` commands for Telegram — shows
/// the image or video model selection keyboard (Artist role).
async fn handle_models_command(msg: &ChannelMessage, is_image: bool) {
    let reply_markup = build_models_keyboard(is_image);
    let content = if is_image {
        "Select an image model:".to_string()
    } else {
        "Select a video model:".to_string()
    };
    // Send directly through the channel so the inline_keyboard structure
    // (rows of buttons) is preserved exactly — the router delivery path
    // has no inline-keyboard support, so this bypasses it for multi-row
    // replies like the model menus.
    let _ = mahbot::channels::telegram::send_direct(&msg.reply_target, content, Some(reply_markup))
        .await;
}

/// Build inline keyboard for image or video model selection.
///
/// Returns the full Telegram `inline_keyboard` JSON array, where each element
/// is a row (list of buttons in that row). Each button gets its own row,
/// followed by a clear-session button.
fn build_models_keyboard(is_image: bool) -> serde_json::Value {
    let mut rows: Vec<serde_json::Value> = Vec::new();

    // Model buttons — each on its own row
    let (mut models, active, action_prefix) = if is_image {
        (
            CONFIG.image_gen_models(),
            CONFIG.image_gen_model(),
            "__act__set_image_model",
        )
    } else {
        (
            CONFIG.video_models(),
            CONFIG.video_model(),
            "__act__set_video_model",
        )
    };
    // Merge the active model into the rendered list when the list omits it,
    // so the ✓ indicator unambiguously shows the active model.
    if !models.iter().any(|m| m == &active) {
        models.push(active.clone());
    }
    build_model_button_rows(&mut rows, &models, &active, action_prefix);

    rows.push(serde_json::json!([{
        "text": "Clear session",
        "callback_data": "__act__clear_session|",
    }]));

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

// ── Admin commands (board / archive / pause / unpause / maintenance) ─────

/// Resolve the user's shared active workspace for admin commands. Returns
/// `None` when the user has no shared workspace selected (personal or
/// undefined) — mirroring the GUI's "no active workspace" guard.
async fn resolve_admin_workspace(msg: &ChannelMessage) -> Result<Option<String>, String> {
    let selected = mahbot::users::get_raw_selected_workspace(&msg.user_name)
        .await
        .map_err(|e| format!("Failed to read workspace selection: {e}"))?;
    match selected {
        Some(name) if !name.trim().is_empty() => {
            let ws = mahbot::workspace::get_by_name(&name)
                .await
                .map_err(|e| format!("Failed to look up workspace: {e}"))?;
            match ws {
                Some(_) => Ok(Some(name)),
                None => Err(format!("Active workspace '{name}' no longer exists.")),
            }
        }
        _ => Ok(None),
    }
}

/// Handle admin-gated commands (`/board`, `/archive`, `/pause`, `/unpause`,
/// `/maintenance`). All reuse the same store methods the GUI calls so the
/// two surfaces can never diverge.
async fn handle_admin_command(msg: &ChannelMessage, cmd: mahbot::BotCommand) {
    // `/maintenance` validates its on|off argument before anything else —
    // a missing/invalid arg gets a usage response regardless of workspace
    // state. Lowercased first: command recognition is case-insensitive.
    let maintenance_arg = if cmd == BotCommand::Maintenance {
        let arg = msg.content.trim().to_ascii_lowercase();
        match arg.strip_prefix("/maintenance").map_or("", str::trim) {
            "on" => Some(true),
            "off" => Some(false),
            _ => {
                send_telegram_reply(msg, "Usage: /maintenance on|off".to_string()).await;
                return;
            }
        }
    } else {
        None
    };

    let ws_name = match resolve_admin_workspace(msg).await {
        Ok(Some(name)) => name,
        Ok(None) => {
            send_telegram_reply(
                msg,
                "No active workspace — select a shared workspace in Settings → Users.".to_string(),
            )
            .await;
            return;
        }
        Err(e) => {
            send_telegram_reply(msg, e).await;
            return;
        }
    };

    match (cmd, maintenance_arg) {
        (BotCommand::Board, _) => handle_board_listing(msg, &ws_name).await,
        (BotCommand::Archive, _) => {
            let count = mahbot::board::store()
                .archive_all_done_and_cancelled(Some(&ws_name))
                .await;
            match count {
                Ok(n) => send_telegram_reply(msg, format!("Archived {n} tickets.")).await,
                Err(e) => send_telegram_reply(msg, format!("Failed to archive tickets: {e}")).await,
            }
        }
        (BotCommand::Pause, _) => toggle_workspace_state(msg, &ws_name, true, false).await,
        (BotCommand::Unpause, _) => toggle_workspace_state(msg, &ws_name, false, false).await,
        (BotCommand::Maintenance, Some(enable)) => {
            toggle_workspace_state(msg, &ws_name, enable, true).await;
        }
        (BotCommand::MaintenanceOn, _) => toggle_workspace_state(msg, &ws_name, true, true).await,
        (BotCommand::MaintenanceOff, _) => toggle_workspace_state(msg, &ws_name, false, true).await,
        // Impossible: invalid /maintenance args returned early above, and the
        // non-admin commands never reach this handler.
        _ => unreachable!(),
    }
}

/// Apply a pause or maintenance toggle via the workspace store (the same
/// method the GUI toggle uses) and confirm the requested state.
async fn toggle_workspace_state(
    msg: &ChannelMessage,
    ws_name: &str,
    enable: bool,
    is_maintenance: bool,
) {
    let store = mahbot::workspace::store();
    let result = if is_maintenance {
        store.set_maintenance_enabled(ws_name, enable).await
    } else {
        store.set_paused(ws_name, enable).await
    };
    if let Err(e) = result {
        send_telegram_reply(msg, format!("Failed to update workspace '{ws_name}': {e}")).await;
        return;
    }
    let verb = match (is_maintenance, enable) {
        (true, true) => "Maintenance enabled",
        (true, false) => "Maintenance disabled",
        (false, true) => "Workspace pipeline paused",
        (false, false) => "Workspace pipeline resumed",
    };
    send_telegram_reply(msg, format!("{verb} for '{ws_name}'.")).await;
}

/// Handle `/board` — list the active workspace's non-archived tickets in the
/// exact order the GUI board column shows them (shared ordering helper).
async fn handle_board_listing(msg: &ChannelMessage, ws_name: &str) {
    let tickets = match mahbot::board::store()
        .list_all_tickets(Some(ws_name), None)
        .await
    {
        Ok(t) => t,
        Err(e) => {
            send_telegram_reply(msg, format!("Failed to load board: {e}")).await;
            return;
        }
    };
    let ordered = mahbot::board::BoardStore::board_display_order(&tickets);
    if ordered.is_empty() {
        send_telegram_reply(msg, "No tickets.".to_string()).await;
        return;
    }
    // `•` bullet instead of `*` — the listing is rendered through Telegram's
    // markdown→HTML conversion, where a leading `*` would pair with a `*` in
    // a ticket title and swallow the status/id into an italic span. State is
    // bold, the ticket ID is monospace; each line converts independently, so
    // markdown-special characters in a title cannot corrupt other lines.
    let listing = ordered
        .iter()
        .map(|t| mahbot::channels::telegram::format_board_line(&t.phase, &t.id, &t.title))
        .collect::<Vec<_>>()
        .join("\n");
    send_telegram_reply(msg, listing).await;
}

/// Handle an action callback (`__act__` prefix).
///
/// Actions are processed inline without involving the Manager agent queue.
async fn handle_action_callback(msg: ChannelMessage, decoded: (String, String)) {
    let (action, payload) = decoded;

    match action.as_str() {
        "set_image_model" => {
            handle_set_model_action(&msg, &payload, "image_gen_model", "Image generation", true)
                .await;
        }
        "set_video_model" => {
            handle_set_model_action(&msg, &payload, "video_model", "Video", false).await;
        }
        "clear_session" => {
            // Acknowledge callback silently first (dismiss spinner)
            answer_telegram_callback(&msg, None).await;
            handle_clear_session(&msg).await;
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
///
/// The write spans a DB write plus an in-memory update as separate steps, and
/// handlers are spawned (the dispatch loop must stay non-blocking), so a lock
/// preserves the previous inline ordering — interleaved writes would leave
/// `config_kv` and `CONFIG` divergent. Acquired before validation so rapid
/// taps apply in tap order.
static MODEL_WRITE_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

async fn handle_set_model_action(
    msg: &ChannelMessage,
    payload: &str,
    config_key: &str,
    display_name: &str,
    validate_image: bool,
) {
    let _guard = MODEL_WRITE_LOCK
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await;
    if payload.is_empty() {
        tracing::warn!(config_key, "{config_key} action with empty payload");
        answer_telegram_callback(msg, Some("No model specified.".to_string())).await;
        return;
    }
    // Write-time validation: reject image models that cannot generate images
    // (fail-open when the catalog is unavailable — matching the generation
    // tool's semantics).
    if validate_image
        && let Err(e) = mahbot::tools::image_catalog::validate_image_model(payload).await
    {
        answer_telegram_callback(msg, Some(format!("Invalid image model: {e}"))).await;
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

    answer_telegram_callback(msg, Some(format!("{display_name} model set to: {payload}"))).await;
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

    let (ws, (pool, pool_read_failed)) = tokio::join!(
        mahbot::users::resolve_workspace_for_user_name(&msg.user_name),
        mahbot::users::role_pool_status(&msg.user_name),
    );
    let role = mahbot::users::resolve_active_role_from_pool(&msg.user_name, &pool).await;

    // Personal workspaces map Manager→Analyst (pool-clamped), and
    // Assistant/Artist always work in the user's personal workspace regardless
    // of the selected workspace — both resolve atomically, before enrichment
    // and before `msg.workspace` is set so uploads, broadcast, persist and
    // chat_history stay consistent with the routed workspace.
    let (effective_role, ws) = match role {
        Some(role) => {
            let (effective_role, ws) =
                mahbot::users::effective_role_and_workspace(role, ws, &msg.user_name, &pool);
            (Some(effective_role), ws)
        }
        None => (None, ws),
    };

    // Populate workspace on the message so downstream broadcasts and
    // chat_history writes carry the correct (effective) workspace.
    msg.workspace = ws.name.clone();

    // Save original content before enrichment so we persist the raw
    // user-typed text to chat_history (avoids storing large data URIs from
    // multimodal image processing in the Artist role).
    let original_content = msg.content.clone();

    // ── Media-marker enrichment (audio transcription, image processing) ──
    // Runs BEFORE broadcast so the GUI receives transcription text instead
    // of raw `[AUDIO:path]` markers.  Link enrichment runs separately AFTER
    // broadcast to avoid showing AI-generated URL summaries in the user's
    // own message bubble.
    let strategy = if effective_role.is_some_and(|r| r.requires_multimodal()) {
        mahbot::channels::EnrichmentStrategy::Multimodal {
            workspace_path: Some(ws.as_path().to_path_buf()),
        }
    } else {
        mahbot::channels::EnrichmentStrategy::NonMultimodal
    };
    mahbot::channels::enrich_message(&mut msg, &strategy).await;

    // ── Broadcast, persist, and mirror ─────────────────────────────────
    // Broadcast enriched content to GUI (audio transcription visible, data
    // URI images renderable).  Persist original content to chat_history (no
    // data URI bloat), except audio-only messages which persist the enriched
    // transcription (icon + text) so the temp file path never reaches chat
    // history.  Mirror uses the same persist_content (media markers stripped by
    // the mirror function) — for audio-only messages that is the enriched
    // transcription (icon + text).
    let persist_content = if mahbot::channels::has_only_audio_markers(&original_content) {
        &msg.content
    } else {
        &original_content
    };
    broadcast_and_persist_incoming_message(&msg, &msg.content, persist_content).await;

    // ── Link enrichment (URL summaries for agent context) ─────────────
    // Runs after broadcast so AI-generated summaries don't appear in the
    // user's own message bubble.
    let enriched = mahbot::channels::enrich_links(&msg.content).await;
    if let Cow::Owned(s) = enriched {
        tracing::info!(
            channel = %msg.channel,
            user_name = %msg.user_name,
            "Link enricher: prepended URL summaries to message"
        );
        msg.content = s;
    }

    // ── Route through the agent-ID message router ─────────────────
    // An empty role pool means no role is allowed — the message was still
    // broadcast/persisted above, but no agent answers. The user notice
    // fires only for a genuinely empty pool; a transient store read
    // failure (pool or selected role) drops silently — operator warns
    // are already logged, and the notice would be misleading.
    let Some(effective_role) = effective_role else {
        if msg.channel == "telegram" && !pool_read_failed && pool.is_empty() {
            send_telegram_reply(
                &msg,
                "You have no active role assigned — ask an admin to assign roles \
                 in Settings → Users."
                    .to_string(),
            )
            .await;
        }
        return;
    };

    // Every message resolves to a deterministic agent ID and routes
    // through the per-agent consumer loop.  Different agent IDs get
    // different consumer loops = true parallelism.
    message_router::route_user_message(
        msg.content,
        ws.name,
        msg.user_name,
        msg.channel,
        effective_role,
        Some(msg.reply_target),
    )
    .await;
}
