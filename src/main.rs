#![warn(clippy::pedantic)]
// Built for the GUI subsystem so a Windows launch never puts a console window on
// the screen: the platform gives such a process no console, and it does not block
// the shell that started it. A launch with no console loses this file's own prints
// — accepted, never fatal: they go through `mahbot::util::print_stdout` /
// `print_stderr`, where the policy lives. Conditional so the binary's own test
// harness stays a console program and `cargo test --bins` output stays visible.
#![cfg_attr(not(test), windows_subsystem = "windows")]

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

use mahbot::agent::message_router;
use mahbot::channels::broadcast_and_persist_incoming_message;
use mahbot::channels::telegram::{ControlInput, control_input, user_command_entries};
use mahbot::config::CONFIG;
use mahbot::gui::{BOOT_LOG_STORE, Dashboard, JETBRAINS_MONO, Message as DashboardMessage};
use mahbot::session::clear_session;
use mahbot::util::UnwrapPoison;
use mahbot::{BotCommand, Channel, ChannelMessage, Role, Workspace};
/// JetBrainsMono-Regular.ttf embedded for Iced dashboard default font.
const JETBRAINS_MONO_FONT_BYTES: &[u8] = include_bytes!("gui/JetBrainsMono-Regular.ttf");

/// JetBrainsMono-Bold.ttf embedded for header text in the Iced dashboard.
const JETBRAINS_MONO_BOLD_FONT_BYTES: &[u8] = include_bytes!("gui/JetBrainsMono-Bold.ttf");

/// JetBrainsMono-Italic.ttf embedded for italic narration text in the Iced dashboard.
const JETBRAINS_MONO_ITALIC_FONT_BYTES: &[u8] = include_bytes!("gui/JetBrainsMono-Italic.ttf");

/// Top-level `--help` text. Lists only the public subcommands (hidden
/// internals like `__grep-engine` are excluded). Printed through
/// `util::print_stdout`, which adds the terminating newline.
const TOP_LEVEL_USAGE: &str = "\
mahbot — autonomous agentic engineering system with a GUI dashboard daemon

Usage:
  mahbot                 Launch the GUI dashboard daemon
  mahbot chrome <args>   Browser automation CLI over the shared chrome core
  mahbot debug           Read-only SQL query tool against the live stores
  mahbot bench-openrouter <args>
                         Standalone OpenRouter provider benchmark

Options:
  -h, --help             Print this help and exit
  -V, --version          Print the version and exit";

/// INFO-log retention window (hours): the log-cleanup loop deletes INFO
/// entries older than this. Independent of the session-purge cutoff.
const LOG_RETENTION_HOURS: i64 = 8;

/// Run [`bootstrap_mahbot`] and convert panics into `Err` so the dashboard shows
/// a boot error instead of hanging on "Starting…" forever. A panic is recorded
/// as a start-up failure like any other; the `Ok(Err(_))` arm is not recorded
/// again, since the failing step already recorded it.
async fn bootstrap_mahbot_safe() -> Result<(), String> {
    match AssertUnwindSafe(bootstrap_mahbot()).catch_unwind().await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(format!("{e:#}")),
        Err(payload) => {
            let message = mahbot::util::panic_message(&*payload);
            // Recorded like any other start-up failure; the error it returns is
            // what the start-failure screen shows.
            let boot_error = mahbot::boot::record_startup_failure(
                "bootstrap",
                mahbot::boot::startup_panic_error(&message),
            );
            Err(format!("{boot_error:#}"))
        }
    }
}

/// Async startup for `MahBot` — runs on Iced's Tokio runtime via a boot [`Task`].
async fn bootstrap_mahbot() -> Result<()> {
    mahbot::config::load_or_init()
        .await
        .map_err(|e| mahbot::boot::record_startup_failure("config::load_or_init", e))?;

    // Bring the stores up in their one supported order — the pre-flight
    // classification, the logs store, the process-global inits and the
    // consolidated domain store. Shared with the store-lock check so the two
    // cannot drift.
    let log_store = mahbot::boot::open_stores().await?;

    // Start the CDC-driven chronicle subscriber (materializes ticket_chronicle
    // transitions from ticket change events). Must run after the stores are up.
    mahbot::pipeline::chronicle::start_subscriber();

    // Config DB must be loaded before providers, so that API keys
    // and model settings take effect.
    mahbot::config::reload_from_db()
        .await
        .map_err(|e| mahbot::boot::record_startup_failure("config::reload_from_db", e))?;

    // macOS-only local Qwen3-ASR transcriber: start the load-or-download chain
    // as a background task. Config is authoritative here (honors a user-set
    // audio_transcription_use_local=false) and this runs BEFORE the provider
    // init below, so the ~4s model load overlaps with the rest of boot. Never
    // awaited — the app and background services start regardless.
    #[cfg(target_os = "macos")]
    mahbot::audio::local_transcriber::spawn_background_init_if_enabled();
    mahbot::providers::init_global()
        .map_err(|e| mahbot::boot::record_startup_failure("providers::init_global", e))?;

    // macOS-only TTS model load: try the cache first, otherwise spawn the
    // background download. Only runs when TTS is enabled in config, to avoid an
    // unnecessary ~400 MB download.
    #[cfg(target_os = "macos")]
    if mahbot::audio::tts::is_config_enabled() {
        // Open the OS audio output device for a TTS-enabled boot (matches the
        // pre-change behavior where init_global opened it unconditionally).
        let _ = mahbot::audio::tts::ensure_audio_output();
        if !mahbot::audio::tts::try_load_cached() {
            mahbot::audio::tts::spawn_download();
        }
    }

    BOOT_LOG_STORE
        .set(log_store.as_ref().clone())
        .map_err(|_| {
            mahbot::boot::record_startup_failure(
                "BOOT_LOG_STORE::set",
                anyhow::anyhow!("BOOT_LOG_STORE already set"),
            )
        })?;

    spawn_background_tasks(log_store.clone());

    info!(
        version = mahbot::self_update::VERSION,
        "MahBot initialized — dashboard ready"
    );

    let admin_target = mahbot::self_update::resolve_admin_telegram_target().await;
    tokio::spawn(async move {
        mahbot::self_update::notify_admin(
            &mahbot::self_update::back_online_message(),
            admin_target.as_deref(),
        )
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

#[expect(clippy::too_many_lines)]
fn spawn_background_tasks(log_store: Arc<mahbot::logs::LogStore>) {
    let mut tasks = JoinSet::<()>::new();
    let shutdown_token = mahbot::shutdown::shutdown_token();

    // The owner's own shell environment, read as early as the product can: the
    // first read starts here, and the loop re-reads it every 10 minutes so an
    // agent's commands run in the environment the owner's own shell exports
    // (see `mahbot::shell_env`). Commands use the reduced fallback until this
    // succeeds, and never wait for it.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "shell-env-read",
        mahbot::shell_env::run_reader_loop(),
    );

    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "session-cleanup",
        run_cleanup_loop(
            "Session cleanup",
            mahbot::jobs::PURGE_CUTOFF_HOURS,
            |cutoff| async move {
                // Stale-job purge FIRST (single-connection orchestrator: board rollback +
                // sessions DELETE) — protected sessions only become eligible for
                // the TTL guard after the purge cascade removes their job rows.
                let purged = mahbot::jobs::purge_stale_jobs(&cutoff).await?;
                let cleaned = mahbot::session::cleanup_old_transient_sessions(&cutoff).await?;
                // Assistant generated/uploads keep-detection — after the purge
                // so the jobs-state is final; assistant sessions are never
                // transient, so keep-evidence is live regardless of ordering.
                // (Research run folders are NOT swept here — `release_run_folder`
                // is the single release point, invoked per-job by completion,
                // the cancel sweep, and boot resume; crash leftovers are the OS
                // temp sweep's or the periodic temp cleaner's job.)
                let media = mahbot::research_cleanup::sweep_media().await?;
                Ok(purged + cleaned + media)
            },
        ),
    );

    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "log-cleanup",
        run_cleanup_loop("Log cleanup", LOG_RETENTION_HOURS, {
            let store = log_store.clone();
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
        mahbot::agent::maintainer::run_maintainer_loop(),
    );

    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "archive-cancelled",
        mahbot::pipeline::board::run_archive_cancelled_loop(),
    );

    // Alarm/reminder sweep: fires due reminders back into the Assistant's
    // personal session. The first tick acts as the boot-time overdue scan.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "alarm-sweep",
        mahbot::alarms::run_alarm_sweep_loop(),
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
    // triggers rediscover during the 2-3 AM local time window, gated to
    // at most one pass per 7 days (rolling, recorded at pass start).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "nightly-check",
        mahbot::workspace::run_nightly_check_loop(),
    );

    // Periodic temp cleaner (Sanitation): reclaims the product's agents'
    // abandoned temp artifacts. Its own loop, decoupled from workspace work,
    // with a free-space-adaptive cadence (see `mahbot::temp`).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "temp-cleanup",
        mahbot::temp::run_temp_cleanup_loop(),
    );

    // Debug IPC query endpoint: `mahbot debug` connects to this local socket
    // (UDS on Unix, named pipe on Windows) to run read-only SQL against the
    // live stores while an instance holds the lock. Binds after the stores
    // (DOMAIN_CONN + LOG_STORE) are up.
    let ipc_root = mahbot::config::CONFIG.global_storage_root();
    spawn_cancellable(&mut tasks, &shutdown_token, "debug-ipc", async move {
        mahbot::db::ipc::run_ipc_listener(&ipc_root, log_store.clone()).await;
    });

    // Eagerly initialize search engines for all existing workspaces.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "search-engine-init",
        mahbot::search_engine::init_all_engines(),
    );

    // Warm the computer tool's capture-channel probe once at boot so the cached
    // availability result is ready before the first agent runs; the probe would
    // otherwise run lazily on the first capture request. Not on macOS, where the
    // computer tool does not exist.
    #[cfg(not(target_os = "macos"))]
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "computer-capabilities",
        mahbot::tools::computer::warm_capture_probe(),
    );

    // Periodic refresh of the shared update-availability cache, which both the
    // GUI update button and the Telegram `/update` menu read, so the two surfaces
    // cannot diverge. Ticks immediately, then every 10 minutes, and every tick is
    // also what starts the unattended update of a downloaded copy when the check
    // finds a strictly newer release (see `mahbot::self_update`).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "update-availability",
        mahbot::self_update::run_update_availability_refresh(),
    );

    // A documented test-only hook drives a whole update with no window at all —
    // see `mahbot::self_update`; it returns at once unless `MAHBOT_UPDATE_TO_VERSION`
    // named the version to move to.
    //
    // Spawned detached rather than through `spawn_cancellable`: an update owns the
    // drain hand-off itself — it begins the drain and *waits* for the shutdown
    // token before checkpointing, releasing the lock and starting the replacement —
    // so a task that gives way to that same token would be dropped at the one
    // moment it must keep going. The Telegram-triggered update is spawned the same
    // way, for the same reason.
    tokio::spawn(mahbot::self_update::run_env_named_update());

    // Chrome daemon health watchdog: classifies chrome-use health from the
    // daemon-free status and auto-restarts with bounded backoff when it is
    // down; wedges surface on real chrome calls (fail-fast) and wake this
    // watchdog (chrome relay daemon only — never the mahbot service itself).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "chrome-daemon",
        mahbot::tools::chrome_daemon::run_watchdog(),
    );

    // Ended-run chrome session releases: the sessions the runs that ended handed
    // over (`mahbot::tools::chrome_release` owns what a release does and trusts).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "chrome-run-releases",
        mahbot::tools::chrome_release::run_session_release_queue(),
    );

    // The product's own tools — the chrome-use browser helper and the bun
    // runtime. Both follow the shared start-time policy in
    // `managed_bin::install_on_start`: a missing copy is installed straight away,
    // an existing one is refreshed once the boot has settled, with no version
    // comparison, and a non-fatal failure is retried on the next start.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "chrome-use-manager",
        mahbot::tools::chrome_daemon::run_chrome_use_management(),
    );

    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "bun-manager",
        mahbot::tools::bun::run_bun_management(),
    );

    // macOS-only voice-assistant pipeline: runs in the background, managing
    // wake-word detection, command recording, transcription, and routing.
    #[cfg(target_os = "macos")]
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "voice-pipeline",
        mahbot::audio::voice::run_voice_pipeline(),
    );

    let rx = init_message_pipeline(&mut tasks, &shutdown_token);

    // `run_message_dispatch_loop` runs unconditionally and exits only via the
    // shutdown token: `MESSAGE_TX` (a process-lifetime `OnceLock` global set in
    // `init_message_pipeline`) and the always-registered GUI listener hold
    // sender clones, so `rx.recv()` never returns `None` (the `None => break`
    // arm is unreachable in practice).
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
        mahbot::pipeline::run_management(),
    );

    // Listen for stop requests and drive the two-request drain protocol: signals
    // on macOS/Linux, the console control handler on Windows (the sources are
    // installed by `shutdown::install_stop_request_sources`). The first request begins the drain
    // (wait_for_shutdown_signal calls drain_begin and keeps listening); the task
    // returns only on a request that abandons the drain — a SECOND signal, or a
    // force-cancel-class console request — which force-cancels it. Clean drain
    // completion is driven by the drain-watch task below (fires the token when no
    // in-flight agents or orchestrator calls remain).
    tasks.spawn(async move {
        let result = AssertUnwindSafe(mahbot::shutdown::wait_for_shutdown_signal())
            .catch_unwind()
            .await;
        match result {
            Ok(Ok(cause)) => {
                // Names the request that ended the drain, where the platform gives it a name
                // (a Windows console request); the fallback is the literal this line always
                // carried, for the second Unix signal.
                info!(
                    "{} — force-cancelling drain",
                    cause.unwrap_or("Second signal received")
                );
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
    // the non-agent call registry. Drain-cut phase jobs intentionally leave
    // their jobs status='launched' for the puller to re-drive at boot, so a
    // jobs-table count cannot reach zero in the common drain — and even
    // research jobs, which DO terminalize mid-drain via the partial-report
    // path, tell us nothing about cut rounds — so the registries are the
    // authoritative in-flight signal (orchestrator-only LLM calls — analysis
    // consolidation, research synthesis — are tracked in NON_AGENT_CALLS; the
    // research orchestrator holds a whole-run guard).
    // Clean exit when both empty; force-cancel stragglers at the 10-minute
    // cap (in-flight ops with >10 min remaining budget are guaranteed-aborted
    // and boot-resume via status='launched').
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
    // live-writer corruption vector under live connections, so the periodic
    // loop avoids it (see checkpoint::periodic_checkpoint_and_verify).
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
            // One hygiene round: WAL checkpoint + integrity verification.
            mahbot::db::checkpoint::periodic_checkpoint_and_verify().await;
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

    // macOS-only TTS listener: subscribes to CHAT_BROADCAST and triggers audio
    // playback for matching agent responses.
    #[cfg(target_os = "macos")]
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
        use mahbot::channels::GuiChannel;
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

    // Register the voice channel so the message routing system can resolve the
    // "voice" channel name when delivering agent responses. There is no
    // listener — the voice pipeline runs its own mic-capture loop managed by
    // `run_voice_pipeline` — and the whole voice pipeline is macOS-only.
    #[cfg(target_os = "macos")]
    mahbot::channels::register_global();

    rx
}

async fn shutdown_after_dashboard() {
    // The stop trace: names the last stop request the platform recorded, and keeps "the
    // window closed" wording when nothing was recorded. This line is dropped outright and
    // reaches no one — `shutdown::EXIT_TRIGGER` documents why, and what that leaves of a
    // stop's reason.
    info!(
        "{} — shutting down",
        mahbot::shutdown::exit_trigger().unwrap_or("Dashboard window closed")
    );
    // No shutdown() here — the token is already cancelled whenever this function
    // runs: the ordinary exit path (`save_and_exit`) fires it before leaving the
    // iced runtime, and a drain path arrives with it fired already.
    //
    // The exits that run none of this are named once, in `shutdown`'s module docs; a new exit
    // path that drops the runtime without firing the token belongs on that list.
    mahbot::agent::registry::AGENT_REGISTRY.shutdown_all();

    let release = mahbot::tools::chrome_release::flush_and_close_all_chrome_sessions();
    match mahbot::shutdown::urgent_release_budget() {
        // Only a stop the platform gave a kill deadline to (a Windows console close or
        // session end) bounds the stage before the checkpoint; a cut is best-effort and
        // the exit path carries on to it.
        Some(budget) => {
            if tokio::time::timeout(budget, release).await.is_err() {
                warn!("urgent shutdown: browser session release cut at its {budget:?} budget");
            }
        }
        None => release.await,
    }

    // Take the JoinSet out of the lock before awaiting (drop guard).
    let maybe_tasks = {
        let mut guard = BACKGROUND_TASKS.lock().unwrap_poison();
        guard.take()
    };

    // No budget here: these tasks live on the iced runtime, which is dropped before
    // this path runs, so joining them is a sweep of already-cancelled work.
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
    mahbot::db::checkpoint::checkpoint_all_databases().await;
}

fn main() -> Result<()> {
    mahbot::shutdown::install_fatal_signal_handlers();
    mahbot::shutdown::install_panic_hook();

    // Debug subcommand: run SQL query directly, skip all GUI/instance setup.
    // Must be checked before lock acquisition so the debug tool can query
    // databases while an instance is running. No tracing init, no Iced.
    if std::env::args().nth(1).as_deref() == Some("debug") {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        match rt.block_on(mahbot::db::debug::run_debug()) {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                mahbot::util::print_stderr(&format!("Error: {e:#}"));
                std::process::exit(1);
            }
        }
    }

    // Hidden grep-engine subcommand: served by the shell tool's transparent
    // grep interception. Also dispatched before lock acquisition — an instance
    // holds the lock, so a normally-dispatched second process would fail.
    if std::env::args().nth(1).as_deref() == Some("__grep-engine") {
        let code = mahbot::run_grep_engine(&std::env::args().skip(2).collect::<Vec<_>>());
        std::process::exit(code);
    }

    // Hidden environment-dumper subcommand: the shell-env reader spawns this
    // binary as a child of the owner's own shell, so it inherits exactly the
    // environment that shell exported and prints it to stdout. Dispatched here,
    // before the instance lock and before `temp::init_temp_root()`, and it must
    // touch neither — nor tracing, nor the stores: an instance holds the lock,
    // and the dumper's only job is to write its environment out.
    if std::env::args().nth(1).as_deref() == Some("__env-dump") {
        std::process::exit(mahbot::shell_env::dump_environment(
            &std::env::args().skip(2).collect::<Vec<_>>(),
        ));
    }

    // bench-openrouter subcommand: standalone OpenRouter provider benchmark.
    // Dispatched before lock acquisition — it must work while another instance
    // holds the lock. It never opens a live store: the config-store lookup goes
    // through that instance's debug channel while the location is held, is
    // skipped with a printed notice when the location may be held but the channel
    // cannot be reached, and reads the store directly only when nothing holds it.
    if std::env::args().nth(1).as_deref() == Some("bench-openrouter") {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let code = rt.block_on(mahbot::bench_openrouter::run_cli());
        std::process::exit(code);
    }

    // `mahbot chrome` subcommand: browser automation CLI over the
    // shared chrome core. Dispatched before lock acquisition + temp-root init
    // so it can run alongside the instance (it uses its own session namespace);
    // chrome-use resolves to the product's own copy at its standard install
    // directory, which needs no config.
    if std::env::args().nth(1).as_deref() == Some("chrome") {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let args = std::env::args().skip(2).collect::<Vec<_>>();
        let code = rt.block_on(mahbot::run_chrome_cli(&args));
        std::process::exit(code);
    }

    // Top-level help/version flags: print and exit before temp-root init and
    // lock acquisition, so they work while an instance holds the lock (and when
    // it doesn't — without this they would boot the GUI).
    // Only exact argv[1] tokens are matched; subcommand-level flags
    // (`mahbot chrome -h` etc.) keep routing to their own CLI handlers.
    match std::env::args().nth(1).as_deref() {
        Some("-h" | "--help") => {
            mahbot::util::print_stdout(TOP_LEVEL_USAGE);
            return Ok(());
        }
        Some("-V" | "--version") => {
            mahbot::util::print_stdout(concat!("mahbot ", env!("CARGO_PKG_VERSION")));
            return Ok(());
        }
        _ => {}
    }

    // Subscribe to the platform's stop requests before boot and before the interface
    // exists; `shutdown` names the three sources.
    mahbot::shutdown::install_stop_request_sources();

    // Resolve the storage root before the temp root: it is a pure environment read
    // (`mahbot::config::default_config_dir`), resolved before config init so the
    // instance lock below can be acquired before Iced starts, and the two pre-boot
    // steps below are recorded against it. The process-global root is not set yet,
    // and on a launch with no console a stderr-only note is lost — the durable
    // record in `<root>/error.log` is what keeps them recoverable.
    let storage_root = mahbot::config::default_config_dir()
        .map_err(|e| mahbot::boot::record_startup_failure("config::default_config_dir", e))?;

    // Consolidate ALL instance temp files under one private root
    // (`/tmp/mahbot`, mode 0700; `<user temp>\mahbot` on Windows) and pin the
    // platform's temp variables to it — BEFORE any temp use (config, logs,
    // stores, shell children). The debug, __grep-engine, __env-dump,
    // bench-openrouter, chrome and help/version dispatches above must NOT create
    // the root (they exit before this point).
    mahbot::temp::init_temp_root().map_err(|e| {
        mahbot::boot::record_launch_failure(&storage_root, "temp::init_temp_root", e)
    })?;

    // Acquire the instance lock before Iced runtime starts.
    // Stored in a global so the update flow can release/re-acquire it. The
    // refusal is also filed in the durable failure record under the root resolved
    // above: this is a pre-boot step, so the process-global root is not set
    // yet. Its text is passed through unchanged — the shell tool's stale-binary
    // detection matches on it. On a product launch on Windows that the platform gave
    // no console and no standard streams (the double-click case) the text reaches
    // nowhere and the launch is silent: no window, no text, exit code 1 — the filed
    // block is what a second launch leaves behind, and the runner's own-image step
    // reads that same text off the pipe it wired.
    mahbot::self_update::acquire_lock(&storage_root).map_err(|e| {
        mahbot::boot::record_launch_failure(&storage_root, "self_update::acquire_lock", e)
    })?;

    // Read persisted window state (sync, before Iced runtime starts).
    let window_state = mahbot::gui::read_window_state();

    // Initialise the git file-change broadcast before the iced application
    // runs (matching LOG_BROADCAST), so the file-change subscription always
    // has a source — an uninitialized one would end the subscription stream.
    mahbot::gui::init_git_file_change_tx();
    // Initialise the git-commit broadcast so the pipeline-commit subscription
    // always has a source (same convention as the file-change broadcast).
    mahbot::gui::init_git_commit_tx();
    // Warm the CDC ticket sender before the app runs so the board change
    // subscription also has a source; otherwise it ends on the first frame and
    // the board freezes at the initial snapshot (Iced never re-spawns it).
    mahbot::gui::init_board_change_tx();
    // Warm the CDC senders for the workspaces/users/user_channels subscriptions
    // before the app runs (same immediately-ending-stream rationale as the
    // board sender above).
    mahbot::gui::init_workspace_tx();
    mahbot::gui::init_users_tx();
    mahbot::gui::init_user_channels_tx();
    // The runtime-change broadcast (agent registry / transcript / voice status)
    // must have a source before the iced app runs, same convention.
    mahbot::gui::init_runtime_event_tx();

    // The icon on the surfaces the window itself does not carry.
    mahbot::app_icon::install_desktop_integration();

    iced::application(
        move || {
            (
                Dashboard::loading(),
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
    .font(JETBRAINS_MONO_ITALIC_FONT_BYTES)
    .default_font(JETBRAINS_MONO)
    .subscription(Dashboard::subscription)
    .theme(Dashboard::theme)
    .window(dashboard_window_settings(&window_state))
    .exit_on_close_request(false)
    .run()
    // Returning here with an error leaves without the teardown below — one of the exits
    // `shutdown`'s module docs name — and without the shutdown token fired: an iced error means
    // the interface never ran, so there would be nothing to drain.
    .map_err(|e| anyhow::anyhow!("Iced application error: {e}"))?;

    // Iced dropped its runtime; use a short-lived one for async teardown.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("shutdown runtime: {e}"))?;
    rt.block_on(shutdown_after_dashboard());

    Ok(())
}

/// The dashboard window's own settings: the geometry this account was last closed
/// with, and the icon and announced name the platform's identity surfaces need.
fn dashboard_window_settings(state: &mahbot::gui::WindowState) -> iced::window::Settings {
    iced::window::Settings {
        size: iced::Size::new(state.width, state.height),
        position: state.position(),
        min_size: Some(iced::Size::new(800.0, 500.0)),
        // The window's own icon where the platform has one (the title bar on
        // Windows and X11); macOS and Wayland carry no window icon and ignore it.
        icon: mahbot::app_icon::window_icon(),
        // Linux: the name the window announces. A desktop environment matches it
        // against the launcher entry's file name to know which entry — and so which
        // icon — belongs to this window; with nothing announced no entry is ever
        // associated with it.
        #[cfg(target_os = "linux")]
        platform_specific: iced::window::settings::PlatformSpecific {
            application_id: mahbot::app_icon::LINUX_APP_ID.to_owned(),
            ..Default::default()
        },
        ..iced::window::Settings::default()
    }
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
            Ok(n) if n > 0 => tracing::debug!(deleted = n, "{label}: deleted old entries"),
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

        // This classification decides what the control plane does itself; the
        // burst collector's gate (`is_control_message`) is it plus the button
        // press, so nothing handled here is ever collected into a burst.
        match control_input(&msg) {
            // An action payload is handled without the agent. Spawned so a slow
            // catalog validation in set_image_model never stalls this loop.
            Some(ControlInput::Action(decoded)) => {
                spawn(handle_action_callback(msg, decoded));
            }
            Some(ControlInput::Command(cmd)) => {
                handle_bot_command(&msg, cmd).await;
            }
            // An ordinary message — or a button press whose payload is neither an
            // action nor a command — goes to the agent, as it always has.
            None => {
                spawn(process_channel_message(msg));
            }
        }
    }
}

/// Handle a bot command on Telegram, already classified and parsed by
/// [`control_input`].
async fn handle_bot_command(msg: &ChannelMessage, cmd: BotCommand) {
    match cmd {
        BotCommand::Start => handle_start_command(msg).await,
        BotCommand::Clear => handle_clear_session(msg).await,
        // Media model selection is available to every user (models are stored
        // per-user, and every user has the Assistant role).
        BotCommand::ImageModels | BotCommand::VideoModels => {
            handle_models_command(msg, cmd == BotCommand::ImageModels).await;
        }
        // `/agents` replies that there is nothing to switch (the Assistant is
        // every account's only role).
        BotCommand::Agents => handle_agents_command(msg).await,
        // Global admin command: `/update` has its own dispatch path (it is
        // workspace-independent and must not go through `handle_admin_command`,
        // which requires a selected shared workspace). The handler applies its
        // own admin gate + availability pre-check.
        BotCommand::Update => mahbot::self_update::handle_update_command(msg).await,
        // Admin-gated commands: denial for guests. `/workspace` shares the gate
        // but not the body: `handle_admin_command` requires an already-active
        // shared workspace — the very choice `/workspace` exists to make.
        BotCommand::Board
        | BotCommand::Archive
        | BotCommand::Pause
        | BotCommand::Unpause
        | BotCommand::Maintenance
        | BotCommand::MaintenanceOn
        | BotCommand::MaintenanceOff
        | BotCommand::Workspace => {
            if mahbot::users::is_admin(&msg.user_name).await {
                if cmd == BotCommand::Workspace {
                    handle_workspace_command(msg).await;
                } else {
                    handle_admin_command(msg, cmd).await;
                }
            } else {
                send_telegram_reply(msg, mahbot::self_update::ADMIN_ONLY_CMD_MSG.to_string()).await;
            }
        }
    }
}

/// `base` with `tail` appended as a second sentence, or `base` alone when there is
/// no tail — the one place a reply's tail is joined, so the spacing lives here.
fn with_tail(base: String, tail: Option<&str>) -> String {
    match tail {
        Some(tail) => format!("{base} {tail}"),
        None => base,
    }
}

/// Point an admin at the switcher — for text a command cannot honour, or a
/// command with no active workspace to act on. `None` when there is no choice to
/// make: callers name `/workspace` only where it can act, and with fewer than two
/// shared workspaces it refuses itself. One sentence for both replies, so it
/// stands on its own wherever it is appended.
async fn switcher_pointer() -> Option<&'static str> {
    mahbot::users::workspace_switcher_available()
        .await
        .then_some("Use /workspace to choose the active workspace.")
}

/// Send a plain-text reply directly on the Telegram channel (no router
/// broadcast/persist — used for command responses, not agent replies).
async fn send_telegram_reply(msg: &ChannelMessage, content: String) {
    mahbot::channels::telegram::send_reply(&msg.reply_target, &content).await;
}

/// Handle `/agents` — reply that there is nothing to switch: the Assistant is
/// every account's only role.
async fn handle_agents_command(msg: &ChannelMessage) {
    send_telegram_reply(
        msg,
        "You have only the Assistant role — there is nothing to switch.".to_string(),
    )
    .await;
}

/// Handle `/workspace` — show the shared workspaces as a tappable list; the
/// tapped button makes that workspace the admin's active one (the same choice
/// the desktop footer picker writes). With fewer than two shared workspaces
/// there is nothing to switch, so the command refuses instead of falling
/// through to the Assistant as ordinary chat.
async fn handle_workspace_command(msg: &ChannelMessage) {
    let workspaces = match mahbot::users::registered_workspaces().await {
        Ok(workspaces) => workspaces,
        Err(e) => {
            send_telegram_reply(msg, format!("Failed to load workspaces: {e}")).await;
            return;
        }
    };
    // What the command answers when it cannot act, or `None` while it can.
    // Written once: it is both the bare command's reply and the tail of a
    // stray-text refusal, which has to explain the same thing.
    let unavailable = if mahbot::users::switcher_exists(workspaces.len()) {
        None
    } else if workspaces.is_empty() {
        Some("No shared workspace is registered — there is nothing to switch.")
    } else {
        Some("Only one shared workspace is registered — there is nothing to switch.")
    };
    if let Some(refusal) = stray_text_refusal(&msg.content) {
        // The tail answers the command either way: why it cannot act, or the list
        // a bare `/workspace` posts when it can.
        let tail = unavailable.unwrap_or("Send it on its own to get the list.");
        send_telegram_reply(msg, with_tail(refusal, Some(tail))).await;
        return;
    }
    if let Some(unavailable) = unavailable {
        send_telegram_reply(msg, unavailable.to_string()).await;
        return;
    }
    // The active workspace: unset leaves every entry unmarked.
    let active = match mahbot::users::get_raw_selected_workspace(&msg.user_name).await {
        Ok(name) => name,
        Err(e) => {
            send_telegram_reply(msg, format!("Failed to read the active workspace: {e}")).await;
            return;
        }
    };
    let keyboard =
        mahbot::channels::telegram::workspace_picker_keyboard(&workspaces, active.as_deref());
    // Send directly through the channel so the inline_keyboard structure (rows
    // of buttons) is preserved exactly — the router delivery path has no
    // inline-keyboard support, same as the model pickers.
    if let Err(e) = mahbot::channels::telegram::send_direct(
        &msg.reply_target,
        "Select the active workspace:".to_string(),
        Some(keyboard),
    )
    .await
    {
        warn!(error = %e, "Failed to send the workspace picker");
    }
}

/// Handle `/start` command for Telegram — sends a per-user welcome message
/// listing the commands available to the current admin state (no inline
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
    // resolution as routing — DB-selected workspace, the single Assistant role,
    // and Assistant pinning.
    let (effective_role, ws) = mahbot::users::resolve_session_target(&msg.user_name).await;
    let reply = match clear_session(&msg.user_name, effective_role.as_str(), &ws.name).await {
        Ok(reply) => reply,
        Err(e) => {
            tracing::warn!(user = %msg.user_name, error = %e, "/clear failed — session kept");
            format!("Failed to clear the session: {e:#}")
        }
    };
    deliver_clear_reply(&reply, msg, &ws, effective_role).await;
}

/// Deliver a session-clear confirmation via the router's raw `reply_target`
/// path (broadcast + persist + transport). The caller passes the already
/// effective role (Assistant pinning applied) so the confirmation bubble
/// matches agent responses.
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
            kind: message_router::MessageKind::UserMessage,
            role: effective_role,
            reply_target: Some(msg.reply_target.clone()),
            pending_job_id: None,
            originating_workspace: None,
        },
        &effective_role,
        &[],
    )
    .await;
}

/// Handle `/image_models` / `/video_models` commands for Telegram — shows
/// the image or video model selection keyboard.
async fn handle_models_command(msg: &ChannelMessage, is_image: bool) {
    let reply_markup = build_models_keyboard(is_image, &msg.user_name).await;
    let content = if is_image {
        "Select an image model:".to_string()
    } else {
        "Select a video model:".to_string()
    };
    // Send directly through the channel so the inline_keyboard structure
    // (rows of buttons) is preserved exactly — the router delivery path
    // has no inline-keyboard support, so this bypasses it for multi-row
    // replies like the model menus.
    if let Err(e) =
        mahbot::channels::telegram::send_direct(&msg.reply_target, content, Some(reply_markup))
            .await
    {
        warn!(error = %e, "Failed to send the model picker");
    }
}

/// Build inline keyboard for image or video model selection for `user_name`.
///
/// Returns the full Telegram `inline_keyboard` JSON array, where each element
/// is a row (list of buttons in that row). Each button gets its own row,
/// followed by a clear-session button.
async fn build_models_keyboard(is_image: bool, user_name: &str) -> serde_json::Value {
    let mut rows: Vec<serde_json::Value> = Vec::new();

    // Model buttons — each on its own row
    let (mut models, active, action_prefix) = if is_image {
        (
            CONFIG.image_gen_models(),
            mahbot::users::resolve_image_gen_model(user_name).await,
            "__act__set_image_model",
        )
    } else {
        (
            CONFIG.video_models(),
            mahbot::users::resolve_video_model(user_name).await,
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
/// undefined) — mirroring the GUI's "no active workspace" guard — and the
/// workspace row otherwise, so replies can name it the way the desktop does.
async fn resolve_admin_workspace(msg: &ChannelMessage) -> Result<Option<Workspace>, String> {
    let selected = mahbot::users::get_raw_selected_workspace(&msg.user_name)
        .await
        .map_err(|e| format!("Failed to read workspace selection: {e}"))?;
    match selected {
        Some(name) if !name.trim().is_empty() => {
            let ws = mahbot::workspace::get_by_name(&name)
                .await
                .map_err(|e| format!("Failed to look up workspace: {e}"))?;
            match ws {
                Some(ws) => Ok(Some(ws)),
                None => Err(format!("Active workspace '{name}' no longer exists.")),
            }
        }
        _ => Ok(None),
    }
}

/// Handle the admin-gated commands that act on the active workspace (`/board`,
/// `/archive`, `/pause`, `/unpause`, `/maintenance`). All reuse the same store
/// methods the GUI calls, so the two surfaces can never diverge.
async fn handle_admin_command(msg: &ChannelMessage, cmd: mahbot::BotCommand) {
    // The pause/resume pair reports trailing text rather than dropping it:
    // `/pause <name>` used to act on the active workspace while reading as if it
    // targeted the named one. The other admin commands are unaffected.
    if matches!(cmd, BotCommand::Pause | BotCommand::Unpause)
        && let Some(refusal) = stray_text_refusal(&msg.content)
    {
        // Only a refusal needs the pointer, and it costs a workspace read.
        let tail = switcher_pointer().await;
        send_telegram_reply(msg, with_tail(refusal, tail)).await;
        return;
    }

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

    let ws = match resolve_admin_workspace(msg).await {
        Ok(Some(ws)) => ws,
        Ok(None) => {
            let text = with_tail(
                "No active workspace is selected.".to_string(),
                switcher_pointer().await,
            );
            send_telegram_reply(msg, text).await;
            return;
        }
        Err(e) => {
            send_telegram_reply(msg, e).await;
            return;
        }
    };

    match (cmd, maintenance_arg) {
        (BotCommand::Board, _) => handle_board_listing(msg, &ws).await,
        (BotCommand::Archive, _) => {
            let count = mahbot::pipeline::board::store()
                .archive_all_done_and_cancelled(Some(&ws.name))
                .await;
            match count {
                Ok(n) => {
                    send_telegram_reply(
                        msg,
                        format!("Archived {n} tickets in {}.", ws.display_name()),
                    )
                    .await;
                }
                Err(e) => {
                    send_telegram_reply(
                        msg,
                        format!("Failed to archive tickets in {}: {e}", ws.display_name()),
                    )
                    .await;
                }
            }
        }
        (BotCommand::Pause, _) => toggle_workspace_state(msg, &ws, true, false).await,
        (BotCommand::Unpause, _) => toggle_workspace_state(msg, &ws, false, false).await,
        (BotCommand::Maintenance, Some(enable)) => {
            toggle_workspace_state(msg, &ws, enable, true).await;
        }
        (BotCommand::MaintenanceOn, _) => toggle_workspace_state(msg, &ws, true, true).await,
        (BotCommand::MaintenanceOff, _) => toggle_workspace_state(msg, &ws, false, true).await,
        // Impossible: an invalid `/maintenance` argument returned early above,
        // and `/workspace` — the only other admin-gated command — never reaches
        // this handler (see the dispatch arm).
        _ => unreachable!(),
    }
}

/// The refusal for text typed after `/pause`, `/unpause` or `/workspace` — the
/// three commands that report trailing text rather than dropping it (`/pause
/// <name>` once acted on the active workspace while reading as if it targeted
/// the named one) — or `None` when there is none. So the caller can use it as
/// the guard, and join its own tail with [`with_tail`]. The command word is
/// echoed as typed so the admin can locate it in the chat.
fn stray_text_refusal(content: &str) -> Option<String> {
    let mut words = content.split_whitespace();
    let cmd_word = words.next().unwrap_or_default();
    words.next()?;
    Some(format!("`{cmd_word}` takes no other text."))
}

/// Apply a pause or maintenance toggle via the workspace store (the same
/// method the GUI toggle uses) and confirm the requested state.
async fn toggle_workspace_state(
    msg: &ChannelMessage,
    ws: &Workspace,
    enable: bool,
    is_maintenance: bool,
) {
    let store = mahbot::workspace::store();
    let result = if is_maintenance {
        store.set_maintenance_enabled(&ws.name, enable).await
    } else {
        store.set_paused(&ws.name, enable).await
    };
    if let Err(e) = result {
        send_telegram_reply(
            msg,
            format!("Failed to update workspace '{}': {e}", ws.display_name()),
        )
        .await;
        return;
    }
    let verb = match (is_maintenance, enable) {
        (true, true) => "Maintenance enabled",
        (true, false) => "Maintenance disabled",
        (false, true) => "Workspace pipeline paused",
        (false, false) => "Workspace pipeline resumed",
    };
    send_telegram_reply(msg, format!("{verb} for '{}'.", ws.display_name())).await;
}

/// Handle `/board` — list the active workspace's non-archived tickets in the
/// exact order the GUI board column shows them (shared ordering helper).
///
/// Every project-reporting reply names the workspace it refers to: the admin
/// can switch workspaces from the same chat, so a bare ticket list would leave
/// the subject ambiguous.
async fn handle_board_listing(msg: &ChannelMessage, ws: &Workspace) {
    let tickets = match mahbot::pipeline::board::store()
        .list_all_tickets(Some(&ws.name), None)
        .await
    {
        Ok(t) => t,
        Err(e) => {
            send_telegram_reply(
                msg,
                format!("Failed to load the board for {}: {e}", ws.display_name()),
            )
            .await;
            return;
        }
    };
    let ordered = mahbot::pipeline::board::BoardStore::board_display_order(&tickets);
    if ordered.is_empty() {
        send_telegram_reply(msg, format!("{} — no tickets", ws.display_name())).await;
        return;
    }
    // The line opens with a per-phase emoji rather than a `•`/`*` bullet: a
    // leading `*` would pair with a `*` in a ticket title and swallow the id
    // and title into an italic span. The ticket ID is monospace; each line
    // converts independently, so markdown-special characters in a title cannot
    // corrupt other lines.
    let listing = ordered
        .iter()
        .map(|t| mahbot::channels::telegram::format_board_line(&t.phase, &t.id, &t.title))
        .collect::<Vec<_>>()
        .join("\n");
    let header = format!("{} — {} tickets", ws.display_name(), ordered.len());
    send_telegram_reply(msg, format!("{header}\n{listing}")).await;
}

/// Handle an action callback (`__act__` prefix).
///
/// Actions are processed inline without involving the Manager agent queue.
async fn handle_action_callback(msg: ChannelMessage, decoded: (String, String)) {
    let (action, payload) = decoded;

    match action.as_str() {
        "set_image_model" => {
            handle_set_model_action(&msg, &payload, "Image generation", true).await;
        }
        "set_video_model" => {
            handle_set_model_action(&msg, &payload, "Video", false).await;
        }
        "clear_session" => {
            // Acknowledge callback silently first (dismiss spinner)
            answer_telegram_callback(&msg, None).await;
            handle_clear_session(&msg).await;
        }
        "set_workspace" => handle_set_workspace_action(&msg, &payload).await,
        _ => {
            // Acknowledge callback queries to dismiss the Telegram loading
            // spinner and surface a toast for unknown actions. This catches
            // stale inline keyboards (e.g. a removed role-picker button) whose
            // tap must not fail silently.
            answer_telegram_callback(
                &msg,
                Some("This action is no longer available.".to_string()),
            )
            .await;
            tracing::warn!(action = %action, "Unknown __act__ action — ignoring");
        }
    }
}

/// Serializes picker callbacks — the per-user model pickers and the
/// active-workspace picker — so rapid taps of one picker apply in tap order and
/// each picker's in-place ✓ refresh lands on the row that actually won. One
/// shared lock: both are the same pattern, and a single lock cannot drift. It is
/// held across the HTTP that follows a tap (the model write's request, the
/// workspace confirmation reply and the markup refresh), the accepted tradeoff
/// for deterministic ordering — human-paced taps and the HTTP client timeout
/// caps any stall. The tap's acknowledgement and posting a picker are not under
/// it: neither is part of the order the taps must apply in.
static PICKER_WRITE_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

/// Common handler for setting a per-user model via callback action.
///
/// Validates the payload (image models against the catalog, fail-open),
/// writes the user's `image_gen_model`/`video_model` column, refreshes the
/// pressed picker's ✓ in place on success, and acknowledges the callback
/// with a toast.
async fn handle_set_model_action(
    msg: &ChannelMessage,
    payload: &str,
    display_name: &str,
    validate_image: bool,
) {
    // Serialized with every other picker tap (see `PICKER_WRITE_LOCK`).
    let toast = {
        let _guard = PICKER_WRITE_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        match set_user_model(&msg.user_name, payload, display_name, validate_image).await {
            Ok(toast) => {
                refresh_pressed_models_keyboard(msg, validate_image).await;
                Some(toast)
            }
            Err(toast) => Some(toast),
        }
    };
    answer_telegram_callback(msg, toast).await;
}

/// Switch the presser's active workspace from a picker tap.
///
/// The payload is the registered workspace name; the write goes through the
/// canonical `set_active_workspace` guard (admin-only, non-personal, must
/// exist), so a stale keyboard cannot point the active choice at nothing. The
/// tap path pre-judges nothing: whether the switch is allowed — admin-only is
/// absolute, and the payload is public knowledge, so a guest holding a forwarded
/// copy or hand-crafting the callback data in a group must change nothing — is
/// decided once, by the write itself, and its refusal is reported in the write's
/// own words rather than as the command gate's generic notice.
async fn handle_set_workspace_action(msg: &ChannelMessage, payload: &str) {
    // Serialized with every other picker tap (see `PICKER_WRITE_LOCK`): the
    // write, the ✓ refresh and the reply all land in tap order, the reply's HTTP
    // call included. The acknowledgement below is outside it — it has no order to
    // keep and must not wait behind other taps.
    {
        let _guard = PICKER_WRITE_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let reply = match mahbot::users::set_active_workspace(&msg.user_name, payload).await {
            Ok(ws) => {
                refresh_pressed_workspaces_keyboard(msg).await;
                // A reply, not just a toast: the rest of the chat (which of
                // /pause and /unpause the menu offers, what /board reports)
                // follows the switch, and the menu refresh rides the outbound
                // send.
                format!("Active workspace: {}", ws.display_name())
            }
            // Plainly, in the chat: the toast is transient and only the presser
            // sees it in a group.
            Err(e) => e.to_string(),
        };
        send_telegram_reply(msg, reply).await;
    }
    // The reply above is the confirmation; the tap only dismisses its spinner
    // rather than repeating the same sentence as a toast.
    answer_telegram_callback(msg, None).await;
}

/// Best-effort in-place refresh of the pressed picker keyboard — `keyboard`
/// replacing the one the callback's own message carries. Any failure (a callback
/// without source-message identity, a deleted/48h-expired message, an identical
/// keyboard, a transport error) is cosmetic: the pick itself is already reported,
/// so the failure is logged at debug and skipped.
async fn refresh_pressed_picker_keyboard(
    msg: &ChannelMessage,
    keyboard: &serde_json::Value,
    what: &str,
) {
    let (Some(chat_id), Some(message_id)) = (&msg.chat_id, msg.message_id) else {
        return;
    };
    if let Some(channel) = mahbot::channel_registry().get("telegram")
        && let Some(tc) = channel
            .as_any()
            .downcast_ref::<mahbot::channels::telegram::TelegramChannel>()
        && let Err(e) = tc.edit_reply_markup(chat_id, message_id, keyboard).await
    {
        tracing::debug!(?e, what, "picker keyboard refresh skipped");
    }
}

/// Move the pressed model picker's ✓ to the just-saved model.
async fn refresh_pressed_models_keyboard(msg: &ChannelMessage, is_image: bool) {
    let keyboard = build_models_keyboard(is_image, &msg.user_name).await;
    refresh_pressed_picker_keyboard(msg, &keyboard, "model").await;
}

/// Move the pressed workspace picker's ✓ to the just-selected workspace.
async fn refresh_pressed_workspaces_keyboard(msg: &ChannelMessage) {
    // The switch itself is already reported to the admin, so a failed read is a
    // plain skip.
    let (Ok(workspaces), Ok(active)) = (
        mahbot::users::registered_workspaces().await,
        mahbot::users::get_raw_selected_workspace(&msg.user_name).await,
    ) else {
        return;
    };
    let keyboard =
        mahbot::channels::telegram::workspace_picker_keyboard(&workspaces, active.as_deref());
    refresh_pressed_picker_keyboard(msg, &keyboard, "workspace").await;
}

/// Validate and write one per-user model pick, returning the callback toast.
/// Validation order: payload presence, image-model catalog check (fail-open
/// when the catalog is unavailable — matching the generation tool's
/// semantics), user existence (a stale keyboard tapped by an unknown sender
/// must not mint a bare users row), then the direct column write (no config
/// warmup).
async fn set_user_model(
    user_name: &str,
    payload: &str,
    display_name: &str,
    validate_image: bool,
) -> Result<String, String> {
    if payload.is_empty() {
        tracing::warn!(display_name, "{display_name} action with empty payload");
        return Err("No model specified.".to_string());
    }
    if validate_image
        && let Err(e) = mahbot::tools::media_catalog::image::validate_image_model(payload).await
    {
        return Err(format!("Invalid image model: {e}"));
    }
    let store = mahbot::users::store();
    if !store.user_exists(user_name).await.unwrap_or(false) {
        return Err("User is not registered.".to_string());
    }
    let result = if validate_image {
        store.set_image_gen_model(user_name, payload).await
    } else {
        store.set_video_model(user_name, payload).await
    };
    match result {
        Ok(()) => Ok(format!("{display_name} model set to: {payload}")),
        Err(e) => {
            tracing::error!(user_name, error = %e, "Failed to save per-user model");
            Err(format!("Failed to save model: {e}"))
        }
    }
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

async fn process_channel_message(msg: ChannelMessage) {
    // A collected burst arrives as one message whose `parts` are the individual
    // messages in send order (see `channels::telegram_group`); a message that
    // arrived on its own is the one-part case of the same flow, so both go
    // through one path. Either way the parts are never empty — a lone message is
    // wrapped into one, a collected burst always holds at least two.
    let user_name = msg.user_name.clone();
    let channel = msg.channel.clone();
    let reply_target = msg.reply_target.clone();
    let mut parts = if msg.parts.is_empty() {
        vec![msg]
    } else {
        msg.parts
    };

    let ws = mahbot::users::resolve_workspace_for_user_name(&user_name).await;

    // Every account routes to the single Assistant role, and the Assistant
    // always works in the user's personal workspace regardless of the selected
    // workspace — resolved before enrichment and before the burst's workspace is
    // set so uploads, broadcast, persist and chat_history stay consistent with
    // the routed workspace.
    let ws = mahbot::users::effective_workspace_for_role(mahbot::Role::Assistant, ws, &user_name);

    // Populate workspace on every part so downstream broadcasts and
    // chat_history writes carry the correct (effective) workspace.
    for part in &mut parts {
        part.workspace.clone_from(&ws.name);
    }

    // The raw user-typed text of every part, so chat_history stores what the
    // user wrote rather than the enriched form (which can carry large data URIs
    // from image processing) — captured together with the log line, before
    // enrichment rewrites the content.
    let mut originals = Vec::with_capacity(parts.len());
    for part in &parts {
        tracing::info!(
            "💬 [{}] from {}: {}",
            part.channel,
            part.user_name,
            mahbot::util::truncate(&part.content, 80)
        );
        originals.push(part.content.clone());
    }

    // ── Media-marker enrichment (audio transcription, image processing) ──
    // Runs BEFORE broadcast so the GUI receives the enriched form instead of
    // raw `[AUDIO:path]` markers (on macOS the transcription text, elsewhere the
    // honest "not supported" note). Media-marker enrichment turns images into
    // native data-URI parts carrying the original bytes (re-encoded to a
    // bounded JPEG only when they would exceed the encoded-payload cap) and
    // videos into a workspace copy + transcription.
    // The parts are independent and this is the expensive step — transcription,
    // image decoding — so the whole burst is enriched at once, exactly as the
    // parts were when each was its own task.
    // Every routed turn is the Assistant's, so a workspace path is always
    // attached: workspace copies and the video transcription it triggers
    // are always seen.
    futures_util::future::join_all(
        parts
            .iter_mut()
            .map(|part| mahbot::channels::enrich_message(part, Some(ws.as_path()))),
    )
    .await;

    // ── Broadcast, persist, link enrichment, reply marker ───────────────
    // Each part is handled on its own, in send order, so the user's own
    // messages keep their exact appearance while the assistant receives the
    // whole burst as one request below. The part bubbles appear once the
    // slowest part of the burst is enriched, which is what keeps them in the
    // order they were sent instead of interleaved with a late transcription.
    for (part, original_content) in parts.iter_mut().zip(&originals) {
        // Broadcast/mirror. `persist_content` decides what reaches chat
        // history: the raw original text, or the data-URI-stripped enriched
        // content when the markers name inbound attachments whose temp paths
        // must not be persisted.
        let content_for_history =
            mahbot::channels::persist_content(original_content, &part.content);
        broadcast_and_persist_incoming_message(part, &part.content, &content_for_history).await;

        // ── Link enrichment (URL summaries for agent context) ─────────────
        // Runs after broadcast so AI-generated summaries don't appear in the
        // user's own message bubble.
        let enriched = mahbot::channels::enrich_links(&part.content).await;
        if let Cow::Owned(s) = enriched {
            tracing::info!(
                channel = %part.channel,
                user_name = %part.user_name,
                "Link enricher: prepended URL summaries to message"
            );
            part.content = s;
        }

        // ── Reply marker ────────────────────────────────────────────────
        // Prepend the reply marker AFTER link enrichment (so URL summaries never
        // see it) and BEFORE routing. Broadcast + persist ran above on the
        // marker-free content, so the marker never reaches chat_history.
        if let Some(reply) = part.reply_reference.clone() {
            part.content = mahbot::channels::apply_reply_marker(&part.content, &reply);
        }
    }

    let content = mahbot::channels::compose_group_content(parts);

    // ── Route through the agent-ID message router ─────────────────
    // Every account routes to the single Assistant, so there is no role to
    // resolve — and no role-store read that could fail closed — before routing.
    // Every message resolves to a deterministic agent ID and routes through the
    // per-agent consumer loop: different agent IDs get different consumer
    // loops = true parallelism. The whole burst is routed as one request.
    message_router::route_user_message(
        content,
        ws.name,
        user_name,
        channel,
        mahbot::Role::Assistant,
        Some(reply_target),
    )
    .await;
}
