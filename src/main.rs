//! The product's entry point: the verb ladder dispatched before the instance lock,
//! the service's composition, and the iced window that is a layer on top of it.
//!
//! The service is the product; the window is a layer that exists only where a graphical system
//! does. [`mahbot::service::detect_window_mode`] decides that once, before any display is touched: a
//! host with no display boots the same service on its own Tokio runtime and never creates a
//! window, while a desktop runs iced's event loop and the same boot as its first task. A window
//! that cannot be created is recorded and the service carries on without it.
//!
//! The module tree is the library's; this file owns only the wiring the library
//! cannot do for itself — bootstrap, background-task spawn, the message pipeline,
//! the shutdown drain — and the Telegram command surface. A Unix launch started
//! from a terminal is detached from it before anything else: the launcher starts the service in
//! a session of its own and holds the terminal only until the service reports that it is up (or
//! says it is still starting), and what the service prints lands in `<storage root>/daemon.log`
//! (`mahbot::detach`). Built for the GUI subsystem so a Windows launch never puts a console
//! window on the screen — such a process gets no console, so its own prints are lost (accepted,
//! never fatal); they go through `mahbot::util::print_stdout` / `print_stderr`, where the policy
//! lives. Conditional so the bin's test harness stays a console program.
#![warn(clippy::pedantic)]
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
use mahbot::cli::EntryPoint;
use mahbot::gui::{Dashboard, JETBRAINS_MONO, Message as DashboardMessage};
use mahbot::logs::LogStore;
use mahbot::power::Report;
use mahbot::session::clear_session;
use mahbot::util::UnwrapPoison;
use mahbot::{BotCommand, Channel, ChannelMessage, Role, Workspace};
const JETBRAINS_MONO_FONT_BYTES: &[u8] = include_bytes!("gui/JetBrainsMono-Regular.ttf");

const JETBRAINS_MONO_BOLD_FONT_BYTES: &[u8] = include_bytes!("gui/JetBrainsMono-Bold.ttf");

const JETBRAINS_MONO_ITALIC_FONT_BYTES: &[u8] = include_bytes!("gui/JetBrainsMono-Italic.ttf");

/// Set once the service's boot has begun, so a window failure after that point is a crash to
/// report rather than a reason to boot the service a second time on a fresh runtime.
static BOOT_STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Run [`bootstrap_mahbot`], converting panics into `Err` so the dashboard shows a boot error
/// instead of hanging on "Starting…" (the `Ok(Err(_))` arm is not recorded again — the failing
/// step already did). The log store travels with the result: the logs page is built from it.
async fn bootstrap_mahbot_safe(
    start_paused: bool,
    slowdown: Option<Report>,
    window: mahbot::service::WindowMode,
) -> Result<LogStore, String> {
    match AssertUnwindSafe(bootstrap_mahbot(start_paused, slowdown, window))
        .catch_unwind()
        .await
    {
        Ok(Ok(store)) => Ok(store),
        Ok(Err(e)) => {
            // A failed boot can never serve command-line operations, and a frontend must
            // not hang forever waiting for one. The launcher waiting on this launch's word is
            // told the reason here — a boot that failed must never report success.
            mahbot::cli::mark_unavailable();
            let message = format!("{e:#}");
            mahbot::detach::report_failure(&message);
            Err(message)
        }
        Err(payload) => {
            let message = mahbot::util::panic_message(&*payload);
            // Recorded like any other start-up failure; the error is what the start-failure
            // screen shows.
            let boot_error = mahbot::boot::record_startup_failure(
                "bootstrap",
                mahbot::boot::startup_panic_error(&message),
            );
            mahbot::cli::mark_unavailable();
            let message = format!("{boot_error:#}");
            mahbot::detach::report_failure(&message);
            Err(message)
        }
    }
}

/// Async startup, run on whichever runtime the service got — iced's when a window is created, the
/// service's own when it is not.
///
/// `start_paused` is the `mahbot paused` mode: every managed workspace is paused before the poll
/// loop that claims work starts. `slowdown` is what the platform answered when the daemon path
/// asked for the slowdown exemption. `window` is only what the boot line says: the service is the
/// same with or without a window.
async fn bootstrap_mahbot(
    start_paused: bool,
    slowdown: Option<Report>,
    window: mahbot::service::WindowMode,
) -> Result<LogStore> {
    BOOT_STARTED.store(true, std::sync::atomic::Ordering::SeqCst);
    // The stop listener is registered before anything else the boot awaits, on whichever runtime
    // the service got: a stop during boot is a drain from its first instruction, not a default
    // signal kill. It is not part of the background set — it ends with the token, and a process
    // on its way out must not wait for a second signal to join it.
    spawn_stop_listener();
    mahbot::config::load_or_init()
        .await
        .map_err(|e| mahbot::boot::record_startup_failure("config::load_or_init", e))?;

    // The one supported store order — pre-flight classification, logs store, process-global
    // inits, consolidated domain store. Shared with the store-lock check so the two cannot drift.
    let log_store = mahbot::boot::open_stores().await?;

    // Once, in plain words, now that the log store can hold it: a host with no graphical system
    // is the service's ordinary home, not an error and not a degraded state.
    if let mahbot::service::WindowMode::Headless(reason) = window {
        info!(reason, "MahBot runs as a service without a window");
        mahbot::util::print_stdout(&format!(
            "MahBot runs as a service without a window: {reason}"
        ));
    }

    // An INFO status line rather than a problem: written once per start, with nothing
    // de-duplicating it. A platform that made no request says nothing.
    if let Some(report) = slowdown {
        info!(
            request = report.request,
            result = %report.reply,
            "Platform slowdown exemption requested for the life of the service"
        );
    }

    // Materializes ticket_chronicle transitions from ticket change events; needs the
    // stores up.
    mahbot::pipeline::chronicle::start_subscriber();

    // Before providers, so API keys and model settings take effect.
    mahbot::config::reload_from_db()
        .await
        .map_err(|e| mahbot::boot::record_startup_failure("config::reload_from_db", e))?;

    // Kept true at every start: an install registers the system's startup entry, an update
    // re-points it at the file that now runs, and the owner's decision is never overridden
    // (`mahbot::autostart`).
    mahbot::autostart::reconcile();

    // Recognition and spoken replies are deliberately NOT loaded here: each engine loads on
    // first use on every platform (`mahbot::speech::engine`, `mahbot::speech::synth`), so an
    // install that never records or speaks loads nothing at start. The model files that are
    // present are hashed once per start by the `speech-integrity` background task
    // (`spawn_background_tasks`), off the request path, so no load ever pays for the digest.
    mahbot::providers::init_global()
        .map_err(|e| mahbot::boot::record_startup_failure("providers::init_global", e))?;

    // The speech output device is opened at start when the owner has it on, so the first phrase
    // is not refused while a cold device open runs. No model is loaded here.
    mahbot::audio::open_speech_output_if_enabled();

    // `mahbot paused`: every managed workspace is paused here, while the stores are up and
    // the poll loop is not yet spawned, so its first round already finds them paused. This is
    // the ordinary workspace pause — same persisted flag, same `set_paused` path as the
    // dashboard toggle — so it survives restarts, and beyond the ticket pipeline standing
    // still nothing else stops. A pause that cannot be written fails the launch: the daemon
    // must never come up unpaused on a swallowed error.
    if start_paused {
        let workspaces = mahbot::workspace::store()
            .pause_all()
            .await
            .map_err(|e| mahbot::boot::record_startup_failure("workspace::pause_all", e))?;
        info!(
            workspaces,
            "Started with the paused argument: all managed workspaces are on pause"
        );
    }

    // A record a previous process left `running` is either replayed by the recovery the background
    // set starts — its durable rows still stand — or failed here, before that replay, so nothing
    // reads as running forever.
    mahbot::send_outcome::fail_interrupted().await;

    spawn_background_tasks(log_store.clone());

    // The instance is whole from here, so command-line operations may be served — they are
    // side effects that may route into the message pipeline.
    mahbot::cli::mark_ready();

    info!(
        version = mahbot::self_update::VERSION,
        "MahBot initialized — service ready"
    );

    // "Running" is reported only now: the launcher's word means the service is up, not that the
    // instance lock was taken. A boot that failed never reaches this line, and the launcher says
    // so with the reason instead.
    mahbot::detach::report_running();

    // Detached from startup, as is the wait for a transport: a self-hosted start opens with
    // the local Bot API server not yet published, and the notice leaves once it is.
    tokio::spawn(mahbot::self_update::announce_startup(start_paused));

    // Reaches the admin assistant through the same queue a message from the owner enters;
    // everything the notice settles is recorded inside, after the hand-over.
    tokio::spawn(mahbot::self_update::announce_update_to_assistant());

    Ok(log_store.as_ref().clone())
}

/// Global `JoinSet` tracking all background task handles for clean shutdown.
static BACKGROUND_TASKS: std::sync::Mutex<Option<JoinSet<()>>> = std::sync::Mutex::new(None);

/// Run `fut` guarded: a panic is caught and logged via [`tracing::error!`] — background work never
/// dies silently — and the future ends with the global shutdown token. `name` identifies the task.
async fn guarded<F>(shutdown_token: CancellationToken, name: &'static str, fut: F)
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    tokio::select! {
        result = AssertUnwindSafe(fut).catch_unwind() => {
            if let Err(payload) = result {
                error!(
                    "Background task panicked [{name}]: {}",
                    mahbot::util::panic_message(&*payload),
                );
            }
        }
        () = shutdown_token.cancelled() => {},
    }
}

/// Run `fut` until the global shutdown token is cancelled, inside the joined set. The future must
/// return `()` — use [`race_shutdown`](mahbot::shutdown::race_shutdown) to capture a value.
fn spawn_cancellable<F>(
    tasks: &mut JoinSet<()>,
    shutdown_token: &CancellationToken,
    name: &'static str,
    fut: F,
) where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    tasks.spawn(guarded(shutdown_token.clone(), name, fut));
}

/// The stop listener: the first request begins the drain and a second force-cancels it. Spawned
/// by the boot before its first await, so a stop during boot is a drain too, and ended by the
/// token — the service is on its way out then, and nothing may wait for a second signal. Not part
/// of the joined set: a stop is what ends it.
fn spawn_stop_listener() {
    tokio::spawn(guarded(
        mahbot::shutdown::shutdown_token(),
        "stop-listener",
        async {
            match mahbot::shutdown::wait_for_shutdown_signal().await {
                Ok(cause) => {
                    // Names the request that ended the drain where the platform gives it a name;
                    // the fallback is the literal this line always carried, for the second Unix signal.
                    info!(
                        "{} — force-cancelling drain",
                        cause.unwrap_or("Second signal received")
                    );
                    mahbot::shutdown::force_cancel();
                }
                Err(e) => error!("Signal handler failed to set up: {e}"),
            }
        },
    ));
}

#[expect(clippy::too_many_lines)]
fn spawn_background_tasks(log_store: Arc<mahbot::logs::LogStore>) {
    let mut tasks = JoinSet::<()>::new();
    let shutdown_token = mahbot::shutdown::shutdown_token();

    // The owner's own shell environment, read as early as possible and re-read every 10
    // minutes, so an agent's commands run in the environment the owner's shell exports
    // (`mahbot::shell_env`). Commands use the reduced fallback until it succeeds, never wait.
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
                // Stale-job purge FIRST: protected sessions become eligible for the session
                // TTL only after the purge cascade removes their job rows.
                let purged = mahbot::jobs::purge_stale_jobs(&cutoff).await?;
                let cleaned = mahbot::session::cleanup_old_transient_sessions().await?;
                // Assistant generated/uploads keep-detection, after the purge so the jobs state
                // is final; a file younger than a day is never a candidate. Research run folders
                // are NOT swept here — `release_run_folder` is the single release point; crash
                // leftovers belong to the OS temp sweep or the periodic temp cleaner.
                let media = mahbot::research_cleanup::sweep_media().await?;
                Ok(purged + cleaned + media)
            },
        ),
    );

    // Expired telemetry records are removed in bounded, paced batches so the store never
    // stalls the work flowing through it (`mahbot::log_retention`).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "log-cleanup",
        mahbot::log_retention::run_retention_loop(log_store.clone()),
    );

    // A handover's record is kept a day after it finishes, then deleted in paced batches
    // (`mahbot::send_outcome`).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "send-outcome-cleanup",
        run_cleanup_loop(
            "Send outcome cleanup",
            mahbot::send_outcome::RETENTION_HOURS,
            mahbot::send_outcome::purge_expired,
        ),
    );

    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "telegram-bot-api",
        mahbot::channels::run_telegram_server_supervisor(),
    );

    // The chat window's conversation cap, and the repair of what a store already holds past
    // it (`mahbot::channels::chat_history`).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "chat-history-cap",
        mahbot::channels::run_cap_loop(),
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

    // Fires due automations back into the Assistant's personal session; the first tick is the
    // boot-time overdue scan.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "automation-sweep",
        mahbot::automations::run_automation_sweep_loop(),
    );

    // Detects user-agent sessions that failed silently (message sent, no agent responded, none
    // running) and re-triggers the agent.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "dead-session-recovery",
        mahbot::session::dead_session::run_dead_session_recovery_loop(),
    );

    // Triggers rediscover during the 2-3 AM local window, gated to at most one rolling pass
    // per 7 days (recorded at pass start).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "nightly-check",
        mahbot::workspace::run_nightly_check_loop(),
    );
    // Measures every account's memory file during the 3-4 AM window and asks the assistants
    // whose file no longer fits their context to clean up, once a night each.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "memory-inventory",
        mahbot::memory_inventory::run_memory_inventory_loop(),
    );

    // Reclaims the agents' abandoned temp artifacts, on its own loop decoupled from workspace
    // work, with a free-space-adaptive cadence (see `mahbot::temp`).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "temp-cleanup",
        mahbot::temp::run_temp_cleanup_loop(),
    );

    // Tells the admin assistant when the temp volume's free space falls below 5%, on a
    // widening reminder ladder. Deletes nothing (`mahbot::disk_notice`).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "low-disk-notice",
        mahbot::disk_notice::run_low_disk_notice_loop(),
    );

    // `mahbot debug` connects to this local socket (UDS on Unix, named pipe on Windows) to run
    // read-only SQL while an instance holds the lock. Binds after the stores are up.
    let ipc_root = mahbot::config::CONFIG.global_storage_root();
    spawn_cancellable(&mut tasks, &shutdown_token, "debug-ipc", async move {
        mahbot::db::ipc::run_ipc_listener(&ipc_root, log_store.clone()).await;
    });

    // Eagerly initialize the per-workspace file indexes for all existing workspaces.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "workspace-index-init",
        mahbot::workspace_index::init_all_engines(),
    );

    // Warms the capture-channel probe once at boot so the cached availability result is ready
    // before the first agent runs.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "computer-capabilities",
        mahbot::tools::computer::warm_capture_probe(),
    );

    // Ticks immediately then every 10 minutes, so the GUI update button and the Telegram
    // `/update` menu cannot diverge. A tick that finds a strictly newer release also starts the
    // unattended update of a downloaded copy (`mahbot::self_update`).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "update-availability",
        mahbot::self_update::run_update_availability_refresh(),
    );

    // A documented test-only hook drives a whole update with no window; it returns at once
    // unless `MAHBOT_UPDATE_TO_VERSION` named a version.
    //
    // Spawned detached rather than through `spawn_cancellable`: an update owns the drain hand-off
    // itself — it begins the drain and *waits* for the shutdown token before checkpointing,
    // releasing the lock and starting the replacement — so a task yielding to that token would be
    // dropped at the one moment it must keep going. The Telegram-triggered update is spawned the
    // same way.
    tokio::spawn(mahbot::self_update::run_env_named_update());

    // Classifies chrome-use health from the daemon-free status and restarts with bounded
    // backoff when it is down; wedges surface on real chrome calls (fail-fast) and wake this
    // watchdog — the chrome relay daemon only, never the mahbot service.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "chrome-daemon",
        mahbot::tools::chrome_daemon::run_watchdog(),
    );

    // Ended-run chrome session releases: the sessions the runs that ended handed over
    // (`mahbot::tools::chrome_release` owns what a release does and trusts).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "chrome-run-releases",
        mahbot::tools::chrome_release::run_session_release_queue(),
    );

    // Shared start-time policy (`managed_bin::install_on_start`): an absent or non-executable
    // copy is installed straight away, one in place is compared with the newest release once the
    // boot has settled and downloaded only when the versions differ. A failure is retried on the
    // next start.
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

    // Nothing about the build is compiled in: the newest build the release stream offers is
    // looked up and installed, and one published later is picked up on a following start. A
    // failure is recorded, retried on the next start, and never removes a working copy.
    spawn_cancellable(&mut tasks, &shutdown_token, "ffmpeg-install", async {
        mahbot::ffmpeg::install_on_start().await;
    });

    // The vendor's newest release, installed on first use and refreshed at start; a failure is
    // recorded and retried, never removing a working copy.
    spawn_cancellable(&mut tasks, &shutdown_token, "photocraft-install", async {
        mahbot::photocraft::install_on_start().await;
    });

    // Installed from the npm registry on first use and brought to the newest release at start.
    // A failure is recorded and retried, never removing a working copy.
    spawn_cancellable(&mut tasks, &shutdown_token, "hyperframes-refresh", async {
        mahbot::hyperframes::refresh_on_start().await;
    });

    // Hashes the speech model files that are already present, once per start and off the request
    // path: a load only stats them, so the first use of an engine never pays a multi-gigabyte
    // digest. Nothing is fetched here — a missing file keeps its on-demand download.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "speech-integrity",
        mahbot::verify_speech_models_once(),
    );

    // Manages command recording, transcription, routing and wake-word detection.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "voice-pipeline",
        mahbot::audio::voice::run_voice_pipeline(),
    );

    // Drops a speech engine that has had no use for five minutes, and hands the released weights
    // back once the whole process is cold. The wake word's listening tier is kept while the wake
    // word or an enrolment needs it, apart from the instant a clear re-requests it; everything
    // else goes.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "speech-idle-release",
        mahbot::run_speech_idle_release(mahbot::audio::voice::listening_tier),
    );

    let rx = init_message_pipeline(&mut tasks, &shutdown_token);

    // Exits only via the shutdown token: `MESSAGE_TX` (set in `init_message_pipeline`) and the
    // always-registered GUI listener hold sender clones, so `rx.recv()` never returns `None`.
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

    // The workspace-stop notices a stage's technical death records: rendered per attempt with
    // the workspace's pause state read then, and retried until a Manager turn completes with
    // them. Runs alongside the pipeline because the pause it reports is exactly what stops the
    // pipeline in that workspace.
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "workspace-stop-notices",
        mahbot::pipeline::notice::run_delivery_loop(),
    );

    // The two-request drain protocol's listener is spawned by the boot itself
    // (`spawn_stop_listener`), before the first store is opened, so a stop during boot is a drain
    // too.

    // While the drain flag is set, poll the agent registry AND the non-agent call registry. A
    // jobs-table count cannot do it: drain-cut phase jobs intentionally stay status='launched'
    // for the boot puller, and research jobs that DO terminalize mid-drain say nothing about cut
    // rounds. Clean exit when both empty; force-cancel stragglers at the 10-minute cap (ops with
    // >10 min of budget left are guaranteed-aborted and boot-resumed).
    spawn_cancellable(
        &mut tasks,
        &shutdown_token,
        "drain-watch",
        mahbot::jobs::run_drain_watch(),
    );

    // Periodic WAL checkpoint as hygiene: compacts committed frames and bounds WAL growth and
    // reopen cost. Non-truncating (PASSIVE) below the WAL-size cap — TRUNCATE resets the shared
    // WAL frame index, the live-writer corruption vector under live connections, so the loop
    // keeps it rare and the engine refuses the reset while another connection holds the store
    // (answering blocked, not failing) — see `checkpoint::periodic_checkpoint_and_verify`. The
    // round runs every ten minutes on the store's maintenance connection (`db::Maintenance`), so
    // ordinary reads and writes do not wait for the writer behind it; a store that cannot give
    // the round one runs it on the writing connection.
    spawn_cancellable(&mut tasks, &shutdown_token, "auto-checkpoint", async {
        loop {
            if !mahbot::shutdown::sleep_or_shutdown_or_drain(Duration::from_mins(10)).await {
                break;
            }
            // The self-update path checkpoints every store itself and cancels the shutdown
            // token right after; a round already in flight when an update starts can still
            // finish — bounded to one iteration, since the next sleep breaks on the token.
            if mahbot::self_update::update_is_finalizing() {
                break;
            }
            // One hygiene round: WAL checkpoint + integrity verification.
            mahbot::db::checkpoint::periodic_checkpoint_and_verify().await;
        }
    });

    // Kept so shutdown_service can await completion.
    {
        let mut guard = BACKGROUND_TASKS.lock().unwrap_poison();
        let _ = guard.insert(tasks);
    }
}

/// Creates the shared mpsc and broadcast channels and the channel registry, and spawns the
/// Telegram + GUI listeners. Returns the receiver half for [`run_message_dispatch_loop`].
fn init_message_pipeline(
    tasks: &mut JoinSet<()>,
    cancel: &CancellationToken,
) -> tokio::sync::mpsc::Receiver<ChannelMessage> {
    // Before any listener is spawned: all channels push into the same tx, one
    // `run_message_dispatch_loop` consumes rx, and `ChannelMessage.channel` disambiguates origins.
    let (tx, rx) = tokio::sync::mpsc::channel::<ChannelMessage>(100);

    // Global so GuiChannel can forward messages; a local clone registers the listeners below.
    mahbot::MESSAGE_TX
        .set(tx.clone())
        .expect("MESSAGE_TX already set — should be first init");

    let gui_pipeline_tx = tx.clone();

    // Create the chat broadcast channel (capacity 256 for burst tolerance).
    let (chat_tx, _chat_rx) = tokio::sync::broadcast::channel::<mahbot::ChatEvent>(256);
    mahbot::CHAT_BROADCAST
        .set(chat_tx)
        .expect("CHAT_BROADCAST already set — should be first init");

    // Nothing subscribes to CHAT_BROADCAST to speak — the only speech is the admin Assistant's
    // `say` tool, which calls the engine itself. The registry starts empty; channels register
    // below.
    let _ = mahbot::CHANNEL_REGISTRY.set(mahbot::ChannelRegistry::default());

    // A task of its own: which Bot API it talks to (cloud or the product's local server) is the
    // mode the owner chose, and a switch the last shutdown interrupted is finished before any
    // listener runs.
    spawn_cancellable(
        tasks,
        cancel,
        "telegram-channel",
        mahbot::channels::telegram::run_configured_channel(tx.clone()),
    );

    // Always — even in dashboard-only mode it is the bridge between the Iced UI and the pipeline.
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

    // So routing can resolve the "voice" channel name when delivering agent responses. There
    // is no listener: the voice pipeline runs its own mic loop (`run_voice_pipeline`).
    mahbot::channels::register_global();

    rx
}

/// Tear the service down after its stop: the interface may or may not have taken part, and the
/// work here is the same either way.
async fn shutdown_service() {
    // Names the last stop request the platform recorded, keeping the "window close" wording when
    // nothing was. The line is written before the store is closed, and a headless run's log ends
    // with it because that run's own runtime still owns the log writer (`logs::flush_log_writer`);
    // the windowed path's writer went with iced's runtime.
    info!(
        "{} — shutting down",
        mahbot::shutdown::exit_trigger().unwrap_or("window close")
    );
    // No shutdown() here — the token is already cancelled whenever this runs: the ordinary
    // exit path (`save_and_exit`) fires it before leaving the iced runtime, and a drain path
    // arrives with it fired. The exits that run none of this are named in `shutdown`'s module
    // docs; a new exit path that drops the runtime without firing the token belongs there.
    mahbot::agent::registry::AGENT_REGISTRY.shutdown_all();

    let release = mahbot::tools::chrome_release::flush_ended_run_chrome_releases();
    match mahbot::shutdown::urgent_release_budget() {
        // Only a stop the platform gave a kill deadline to (a Windows console close or session
        // end) bounds the stage before the checkpoint; a cut is best-effort.
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

    // These tasks live on the service's runtime; in the windowed path that runtime was iced's
    // and is already gone, so joining them is a sweep of already-cancelled work.
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

    // Single-writer checkpoint: the iced runtime is gone, so no background writer is live.
    // Relocated from save_and_exit, which ran it while writers were still active.
    mahbot::db::checkpoint::checkpoint_all_databases().await;

    // The local Bot API server is a child of this process: it goes away with it, so the next
    // start finds nothing to adopt and cannot end up with two pollers on one bot token.
    mahbot::channels::stop_telegram_server().await;

    // The drain's own lines are the last of a run, and the writer batches on a timer: hand them to
    // the store before the runtime that owns the writer goes with it. Only a headless run still
    // has that writer here — the windowed path's belonged to iced's runtime, already dropped.
    mahbot::logs::flush_log_writer().await;
}

/// A short-lived current-thread runtime for a command-line entry point: it runs one operation
/// and exits.
fn command_line_runtime() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?)
}

/// The words after the command word (`argv[0]` is the binary, `argv[1]` the resolved word).
fn command_line_args() -> Vec<String> {
    std::env::args().skip(2).collect()
}

// Deliberately one function: every dispatch is visible in one place. Every entry point the
// process can be launched as is dispatched here.
#[expect(clippy::too_many_lines)]
fn main() -> Result<()> {
    // The hand-off marker belongs to this launch alone: read and cleared before any thread exists,
    // so a shell this instance spawns cannot hand it down to a `mahbot` started from it — that one
    // is an ordinary launch and must be refused at once, not made to wait out the hand-off.
    mahbot::self_update::take_handoff_marker();
    mahbot::shutdown::install_fatal_signal_handlers();
    mahbot::shutdown::install_panic_hook();

    // An argument that cannot be read as text can name no entry point and fill no value, and
    // must not be echoed — the caller may have passed a secret with a mistyped byte. Refused
    // before the ladder reads anything: the text-only reader would panic, both losing the four
    // exit classes a script reads and printing the very bytes passed.
    if std::env::args_os()
        .skip(1)
        .any(|arg| arg.to_str().is_none())
    {
        std::process::exit(mahbot::cli::refuse_non_text_argument());
    }

    // The command table (`mahbot::cli`) holds every first word this build has, hidden verbs
    // included. A word it has none for is the caller's mistake, refused here — before the
    // storage root, the temp root and the instance lock.
    let Some(entry) = mahbot::cli::entry_point(std::env::args().nth(1).as_deref()) else {
        std::process::exit(mahbot::cli::refuse_unknown_entry_point());
    };

    // Every command-line entry point runs its own frontend and exits before the daemon's
    // pre-boot steps: each must work while an instance holds the lock. What is left for the
    // daemon is a bare launch, `paused` and `__login-start`.
    let (start_paused, login_start) = match entry {
        // Read-only SQL with no GUI or instance setup — must work WHILE an instance holds the
        // lock, hence before lock acquisition. No tracing init, no Iced.
        EntryPoint::Debug => {
            let rt = command_line_runtime()?;
            match rt.block_on(mahbot::db::debug::run_debug()) {
                Ok(()) => std::process::exit(0),
                Err(e) => {
                    mahbot::util::print_stderr(&format!("Error: {e:#}"));
                    std::process::exit(1);
                }
            }
        }

        // Served by the shell tool's transparent grep interception, so also before lock
        // acquisition.
        EntryPoint::GrepEngine => {
            std::process::exit(mahbot::run_grep_engine(&command_line_args()));
        }

        // The shell-env reader spawns this as a child of the owner's own shell, so it inherits
        // exactly that environment and prints it. It must touch neither the instance lock nor
        // `temp::init_temp_root()` — nor tracing, nor the stores.
        EntryPoint::EnvDump => {
            std::process::exit(mahbot::shell_env::dump_environment(&command_line_args()));
        }

        // Standalone: must work while another instance holds the lock. It never opens a live
        // store — the config-store lookup goes through that instance's debug channel, is skipped
        // with a printed notice when the location may be held but the channel cannot be reached,
        // and reads the store directly only when nothing holds it.
        EntryPoint::BenchOpenRouter => {
            let rt = command_line_runtime()?;
            std::process::exit(rt.block_on(mahbot::bench_openrouter::run_cli()));
        }

        // A recording into text, text into a spoken file. It touches the speech models — the one
        // machine-wide set both engines read — and the product's own copy of its media toolchain
        // where a recording's format needs it, and never a store or the settings, so it works
        // while the product is stopped and alongside it.
        EntryPoint::Audio => {
            let rt = command_line_runtime()?;
            std::process::exit(rt.block_on(mahbot::run_audio_cli(&command_line_args())));
        }

        // Over the shared chrome core, alongside the instance (its own session namespace).
        // chrome-use resolves to the product's own copy at its standard install directory.
        EntryPoint::Chrome => {
            let rt = command_line_runtime()?;
            std::process::exit(rt.block_on(mahbot::run_chrome_cli(&command_line_args())));
        }

        // `mahbot <family> …`, every family `mahbot::cli::FAMILIES` publishes: frontends onto
        // the product's own tools. The operations run IN the instance, and this process never
        // opens one of its stores.
        EntryPoint::Family(cli) => {
            let rt = command_line_runtime()?;
            std::process::exit(rt.block_on(mahbot::cli::run(cli, &command_line_args())));
        }

        // Print and exit before the temp root and the lock, so they work while an instance holds
        // it — and when it doesn't, instead of booting the GUI. Only exact first tokens match; a
        // subcommand-level flag reaches its own frontend above.
        EntryPoint::Help => {
            mahbot::util::print_stdout(mahbot::cli::TOP_LEVEL_USAGE);
            return Ok(());
        }
        EntryPoint::Version => {
            mahbot::util::print_stdout(concat!("mahbot ", env!("CARGO_PKG_VERSION")));
            return Ok(());
        }

        // A startup mode, not a command line: the launch continues into the normal GUI startup
        // with the flag that pauses every managed workspace before any background task starts.
        EntryPoint::Paused => (true, false),
        // A login start is an ordinary daemon launch, differing only in the window rule and in
        // the second-launch rule below.
        EntryPoint::LoginStart => (false, true),
        // A bare `mahbot`: the daemon.
        EntryPoint::Daemon => (false, false),
    };

    // From the first instructions of the daemon's start-up — as early as the launch can tell
    // it is one — a hangup or a suspending signal must never end the service; this closes the
    // window before the signal task exists.
    mahbot::shutdown::install_ignored_signals();

    // Resolved before anything else: it is a pure environment read
    // (`mahbot::config::default_config_dir`), the detach below names the log file in it, the
    // instance lock is acquired under it, and pre-boot steps are recorded against it. The
    // process-global root is not set yet, and a note that only ever reached a terminal is lost
    // with it — the durable record in `<root>/error.log` keeps such a step recoverable.
    let storage_root = mahbot::config::default_config_dir()
        .map_err(|e| mahbot::boot::record_startup_failure("config::default_config_dir", e))?;

    // Started from a terminal, the service detaches here: the prompt comes back, it runs in a
    // session of its own with none of the shell's streams, and everything it prints lands in
    // `<root>/daemon.log`. A launch the terminal did not start — a pipe, a supervisor, the
    // login registration, the update hand-off — carries on in the foreground.
    mahbot::detach::detach_from_terminal(&storage_root)?;

    // Asked for before anything else the daemon does, so it holds for the whole life of the
    // service. Every command-line verb above is dispatched before this point on purpose, and a
    // restart or self-update comes back through here rather than carrying anything over. What
    // came back travels into the boot task, which reports it once the logs store can hold it.
    let slowdown = mahbot::power::exempt_from_slowdown();

    // The service is primary; the window is a layer that exists only where a graphical system
    // does. The decision is read once here — before any window state is touched, and before the
    // stop sources ask whether there is a window at all — so a server never reaches for a display
    // it does not have.
    let window_mode = mahbot::service::detect_window_mode();

    // Subscribe to the platform's stop requests before boot and before the interface exists;
    // `shutdown` names the three sources.
    mahbot::shutdown::install_stop_request_sources(window_mode);

    // ALL instance temp files under one private root (`/tmp/mahbot`, mode 0700; `<user
    // temp>\mahbot` on Windows), with the platform's temp variables pinned to it — BEFORE any
    // temp use. The dispatches above and a refused unknown first word must NOT create the root,
    // exiting before this point.
    mahbot::temp::init_temp_root().map_err(|e| {
        let e = mahbot::boot::record_launch_failure(&storage_root, "temp::init_temp_root", e);
        mahbot::detach::report_failure(&format!("{e:#}"));
        e
    })?;

    // Acquired before the Iced runtime starts and stored in a global so the update flow can
    // release and re-acquire it.
    //
    // A second launch while one is running starts nothing and raises nothing: the window is the
    // user's own to bring up, from the tray (`window_requests`), so a keepalive automation's start
    // branch, an agent's restart and a script's "open the app" all leave it where it is. What the
    // launch answers stays what it answered on this platform before (the branches below). An
    // update hand-off is not a second launch at all, and its refusal is filed in the durable
    // failure record under the root resolved above with its text unchanged — the shell tool
    // recognises a stale self-update binary by it (`self_update::second_instance_refusal`). A
    // Windows launch with no console stays silent; a Unix one from a terminal says it there.
    if let Err(refusal) = mahbot::self_update::acquire_lock(&storage_root) {
        // A login start is not a second launch either: the system asked and one is up, so this
        // exits quietly — no failure filed, exit 0. A location that could not be taken for any
        // other reason is filed like a real failure.
        if login_start && mahbot::self_update::is_instance_refusal(&refusal) {
            mahbot::detach::report_already_running();
            return Ok(());
        }
        // Where the window can be hidden, the running copy is asked whether it is taking launches:
        // answered, this launch reports "already running" and exits 0; a boot that has not bound
        // its endpoint yet or a drain under way is refused below, as it always was.
        #[cfg(any(target_os = "macos", target_os = "windows"))]
        if !mahbot::self_update::is_handoff()
            && mahbot::self_update::is_instance_refusal(&refusal)
            && mahbot::db::ipc::launch_taken(&storage_root)
        {
            mahbot::detach::report_already_running();
            return Ok(());
        }
        let refusal = mahbot::boot::record_launch_failure(
            &storage_root,
            "self_update::acquire_lock",
            refusal,
        );
        // This refusal is the launch's whole answer, said on the waiting terminal; the filed
        // block stays the durable half.
        mahbot::detach::report_failure(&format!("{refusal:#}"));
        return Err(refusal);
    }

    // The window is the layer on top of the service this launch decided on.
    if window_mode.is_headless() {
        return run_service_headless(start_paused, slowdown, window_mode);
    }

    // Read persisted window state (sync, before Iced runtime starts).
    let window_state = mahbot::gui::read_window_state();
    // The work area of the screen a fresh window opens on, which caps a fresh size: read before
    // iced's event loop exists (see `mahbot::gui::screen`).
    let work_area = mahbot::gui::screen::work_area();

    // Before the iced application runs (matching LOG_BROADCAST), so the file-change subscription
    // always has a source — an uninitialized one would end the stream.
    mahbot::gui::init_git_file_change_tx();
    // Same convention: the pipeline-commit subscription always has a source.
    mahbot::gui::init_git_commit_tx();
    // Same, for the board change subscription: otherwise it ends on the first frame and the
    // board freezes at the initial snapshot (Iced never re-spawns it).
    mahbot::gui::init_board_change_tx();
    // Same immediately-ending-stream rationale, for workspaces/users/user_channels.
    mahbot::gui::init_workspace_tx();
    mahbot::gui::init_users_tx();
    mahbot::gui::init_user_channels_tx();
    // The runtime-change broadcast (agent registry / transcript / voice status), same convention.
    mahbot::gui::init_runtime_event_tx();
    // The voice-recording broadcast the Home composer subscribes to, so a transcript has
    // somewhere to land.
    mahbot::gui::init_recording_tx();

    // The icon on the surfaces the window itself does not carry.
    mahbot::app_icon::install_desktop_integration();

    // The tray icon — the one way back to a window a close hid, and the product's only in-app
    // quit. The channel first, so a gesture arriving before the dashboard subscribes waits for it.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    mahbot::window_requests::init();

    // The click that brings the window up and the menu that quits. macOS creates the status item
    // on its main thread once iced's loop runs; Windows on a thread of its own. Never fatal.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    mahbot::tray::install();

    // A restart the updater made for a hidden window comes back hidden: the window is created
    // invisible and stays so until the tray brings it up — or the boot rescue shows it, for a
    // failed boot or no tray icon (`gui`). A login start is hidden the same way. An ordinary cold
    // start always shows it.
    let start_hidden = cfg!(any(target_os = "macos", target_os = "windows"))
        && (mahbot::self_update::starts_hidden() || login_start);
    mahbot::gui::app_window::set_hidden(start_hidden);

    // The exit path persists the last reported size and position, and a fresh window's start-up
    // width is settled by the board's first read (see `StartupWidth`).
    let window_settings = dashboard_window_settings(&window_state, work_area, start_hidden);
    // Window creation is a platform call that panics rather than returning on a broken display
    // (iced_winit's event loop and window creation both do), so the decision to keep serving is
    // made here: a window that cannot be created is recorded and the service runs without it.
    // `contain_panics` is the house seam: the panic is caught and kept off stderr, which would
    // otherwise report a fatal-looking crash for a service that goes on serving.
    let run = mahbot::shutdown::contain_panics(|| {
        iced::application(
            move || {
                (
                    Dashboard::loading(&window_state, work_area),
                    iced::Task::perform(
                        bootstrap_mahbot_safe(
                            start_paused,
                            slowdown,
                            mahbot::service::WindowMode::Window,
                        ),
                        DashboardMessage::Boot,
                    ),
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
        .window(window_settings)
        .exit_on_close_request(false)
        .run()
    });

    // The interface is gone, so its tray icon goes with it: a Windows notification-area icon whose
    // owner never removed it outlives the process as a stale one. The self-update tail reaches the
    // removal from its own thread too.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    mahbot::tray::remove();

    // The interface's own failure, whatever ended it: an error iced returned, or a panic it
    // raised — window creation and the event loop panic rather than return on a broken display,
    // and so does a widget. Named apart from the window's creation, which is only one cause.
    let failure = match run {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(format!("the interface could not start: {e}")),
        Err(payload) => Some(format!(
            "the interface stopped on a panic: {}",
            mahbot::util::panic_message(&*payload)
        )),
    };
    if let Some(failure) = failure {
        return window_failed(&failure, &storage_root, start_paused, slowdown);
    }

    // The interface ended with the service already drained. Iced dropped its runtime; a
    // short-lived one runs the async teardown.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("shutdown runtime: {e}"))?;
    rt.block_on(shutdown_service());

    Ok(())
}

/// The runtime the service runs on when no window owns one. Multi-threaded like iced's, because
/// the service's tasks — agent rounds, store writes, transports — are the same either way.
fn service_runtime() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?)
}

/// Run the whole service with no window: the same boot, the same background work and the same
/// drain, on its own runtime. A failed boot is reported to the launcher and returned as the
/// process's own failure, so a start that did not come up never reports success.
fn run_service_headless(
    start_paused: bool,
    slowdown: Option<Report>,
    window: mahbot::service::WindowMode,
) -> Result<()> {
    service_runtime()?.block_on(async move {
        if let Err(message) = bootstrap_mahbot_safe(start_paused, slowdown, window).await {
            // The launcher was already told the reason by the boot's own failure path; the
            // process still ends with it, so a start that did not come up never reports success.
            return Err(anyhow::anyhow!(message));
        }
        // A signal begins the drain; the drain-watch task cancels the token once no in-flight
        // agent or orchestrator call remains (or force-cancels at its cap).
        mahbot::shutdown::shutdown_token().cancelled().await;
        shutdown_service().await;
        Ok(())
    })
}

/// An interface that failed: record why, then keep the service running without it — unless the
/// boot had already begun on the window's runtime, in which case that runtime is gone with the
/// window and the process cannot come back as the same service.
///
/// Filed against the root resolved before the boot, not through `record_startup_failure`: the
/// process-global root is unset until the boot has begun, and this failure can come before it.
fn window_failed(
    failure: &str,
    storage_root: &std::path::Path,
    start_paused: bool,
    slowdown: Option<Report>,
) -> Result<()> {
    let recorded =
        mahbot::boot::record_launch_failure(storage_root, "gui::run", anyhow::anyhow!("{failure}"));
    if BOOT_STARTED.load(std::sync::atomic::Ordering::SeqCst) {
        error!("{recorded:#} — the service had already begun booting, so the process ends");
        return Err(recorded);
    }
    warn!("{recorded:#} — the service keeps running without a window");
    run_service_headless(
        start_paused,
        slowdown,
        mahbot::service::WindowMode::Headless("the interface could not start"),
    )
}

/// The opening geometry ([`mahbot::gui::WindowState::opening_size`], capped to the screen's
/// `work_area`), whether the window opens hidden, the platform's frame, and the icon and announced
/// name its identity surfaces need.
fn dashboard_window_settings(
    state: &mahbot::gui::WindowState,
    work_area: Option<iced::Size>,
    start_hidden: bool,
) -> iced::window::Settings {
    iced::window::Settings {
        size: state.opening_size(work_area),
        position: state.position(),
        // A hidden start is a window that is never shown. macOS shows its own window once the
        // title strip is hidden (`gui::app_window`), so iced leaves it hidden there.
        visible: !start_hidden && cfg!(not(target_os = "macos")),
        min_size: Some(iced::Size::new(
            mahbot::gui::MIN_WINDOW_WIDTH,
            mahbot::gui::MIN_WINDOW_HEIGHT,
        )),
        // Windows keeps no frame and asks the system to round the corners of what is left — one
        // request a system that does not know it ignores, so Windows 10 keeps square corners.
        // macOS keeps the system's titled window with its whole title strip hidden
        // (`gui::app_window`). Linux keeps the frame it draws.
        decorations: cfg!(not(target_os = "windows")),
        // macOS: the strip draws nothing and the content fills the window. The full-size content
        // view also keeps the frame equal to the content rect, so the saved size and position
        // open exactly as saved.
        #[cfg(target_os = "macos")]
        platform_specific: iced::window::settings::PlatformSpecific {
            title_hidden: true,
            titlebar_transparent: true,
            fullsize_content_view: true,
        },
        // Windows: the whole request — the system's own corners.
        #[cfg(target_os = "windows")]
        platform_specific: iced::window::settings::PlatformSpecific {
            corner_preference: iced::window::settings::platform::CornerPreference::Round,
            ..Default::default()
        },
        // The window's own icon where the platform has one (on Windows the small one, the big one
        // attached separately); macOS and Wayland carry none and ignore it.
        icon: mahbot::app_icon::window_icon(),
        // Linux: the name the window announces. A desktop environment matches it against the
        // launcher entry's file name to pick the entry — and so the icon; with nothing announced
        // no entry is associated.
        #[cfg(target_os = "linux")]
        platform_specific: iced::window::settings::PlatformSpecific {
            application_id: mahbot::app_icon::LINUX_APP_ID.to_owned(),
            ..Default::default()
        },
        ..iced::window::Settings::default()
    }
}

/// Background cleanup loop adapter — every 10 minutes until cancelled or the graceful drain
/// begins (stale purge must not race it). `cutoff_hours` is the window of the one step that
/// takes the derived cutoff.
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

        // This classification decides what the control plane does itself; the burst collector's
        // gate is it plus the button press, so nothing handled here is collected into a burst.
        match control_input(&msg) {
            // An action payload is handled without the agent. Spawned so a slow catalog validation
            // in set_image_model never stalls this loop.
            Some(ControlInput::Action(decoded)) => {
                spawn(handle_action_callback(msg, decoded));
            }
            Some(ControlInput::Command(cmd)) => {
                handle_bot_command(&msg, cmd).await;
            }
            // An ordinary message — or a button press that is neither action nor command — goes
            // to the agent, as it always has.
            None => {
                spawn(process_channel_message(msg));
            }
        }
    }
}

/// Handle a bot command on Telegram, already classified and parsed by [`control_input`].
async fn handle_bot_command(msg: &ChannelMessage, cmd: BotCommand) {
    match cmd {
        BotCommand::Start => handle_start_command(msg).await,
        BotCommand::Clear => handle_clear_session(msg).await,
        // Media model selection is available to every user (models are stored per-user, and every
        // user has the Assistant role).
        BotCommand::ImageModels | BotCommand::VideoModels => {
            handle_models_command(msg, cmd == BotCommand::ImageModels).await;
        }
        // `/agents` replies that there is nothing to switch (the Assistant is every account's only
        // role).
        BotCommand::Agents => handle_agents_command(msg).await,
        // Has its own dispatch path because it is workspace-independent and owns its own admin
        // gate and availability pre-check.
        BotCommand::Update => mahbot::self_update::handle_update_command(msg).await,
        // Denial for guests. `/board` and `/archive` span every workspace and must not depend on
        // an active one.
        BotCommand::Board | BotCommand::Archive => {
            if mahbot::users::is_admin(&msg.user_name).await {
                if cmd == BotCommand::Board {
                    handle_board_listing(msg).await;
                } else {
                    handle_archive_command(msg).await;
                }
            } else {
                send_telegram_reply(msg, mahbot::self_update::ADMIN_ONLY_CMD_MSG.to_string()).await;
            }
        }
    }
}

/// Plain-text reply straight on the Telegram channel — no router broadcast/persist, for
/// command responses rather than agent replies.
async fn send_telegram_reply(msg: &ChannelMessage, content: String) {
    mahbot::channels::telegram::send_reply(&msg.reply_target, &content).await;
}

/// Handle `/agents` — reply that there is nothing to switch: the Assistant is every account's only
/// role.
async fn handle_agents_command(msg: &ChannelMessage) {
    send_telegram_reply(
        msg,
        "You have only the Assistant role — there is nothing to switch.".to_string(),
    )
    .await;
}

/// Handle `/start` command for Telegram — sends a per-user welcome message listing the commands
/// available to the current admin state (no inline keyboard).
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

/// Handle the `/clear` command — deletes the current session and confirms via the canonical
/// delivery path.
async fn handle_clear_session(msg: &ChannelMessage) {
    // The session the user actually talks to: the same (role, workspace) resolution as routing —
    // DB-selected workspace, the single Assistant role, Assistant pinning.
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

/// Deliver via the router's raw `reply_target` path (broadcast + persist + transport). The
/// caller passes the already effective role (Assistant pinning applied) so the bubble matches
/// agent responses.
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
            pause_notice_id: None,
            originating_workspace: None,
            custom_role: None,
            reply_to_custom: None,
            urgent: false,
            extract: None,
        },
        &effective_role,
        &[],
    )
    .await;
}

/// Show the image or video model selection keyboard.
async fn handle_models_command(msg: &ChannelMessage, is_image: bool) {
    let reply_markup =
        mahbot::channels::telegram::build_models_keyboard(is_image, &msg.user_name).await;
    let content = if is_image {
        "Select an image model:".to_string()
    } else {
        "Select a video model:".to_string()
    };
    // Sent directly so the `inline_keyboard` structure (rows of buttons) survives exactly — the
    // router's delivery path has no inline-keyboard support.
    if let Err(e) =
        mahbot::channels::telegram::send_direct(&msg.reply_target, content, Some(reply_markup))
            .await
    {
        warn!(error = %e, "Failed to send the model picker");
    }
}

/// List every workspace's non-archived tickets in the GUI board column's order, formatted by
/// [`board_listing_text`](mahbot::channels::telegram::board_listing_text). The listing spans
/// every workspace and never the account's selected one; a long one goes out across several
/// messages through the channel's own chunking.
async fn handle_board_listing(msg: &ChannelMessage) {
    let tickets = match mahbot::pipeline::board::store()
        .list_all_tickets(None, None)
        .await
    {
        Ok(t) => t,
        Err(e) => {
            send_telegram_reply(msg, format!("Failed to load the board: {e}")).await;
            return;
        }
    };
    let ordered = mahbot::pipeline::board::BoardStore::board_display_order(&tickets);
    send_telegram_reply(
        msg,
        mahbot::channels::telegram::board_listing_text(&ordered),
    )
    .await;
}

/// Archive every done and cancelled ticket in the installation, from every workspace.
/// Workspace-independent like `/board`: the admin's active workspace is neither read nor required.
async fn handle_archive_command(msg: &ChannelMessage) {
    // `/archive foo` used to be read as a workspace-scoped archive with a stray argument. The
    // command has no target at all, so the refusal says so.
    let mut words = msg.content.split_whitespace();
    let cmd_word = words.next().unwrap_or_default();
    if words.next().is_some() {
        send_telegram_reply(
            msg,
            format!(
                "`{cmd_word}` takes no other text. \
                 It has no target — it archives done and cancelled tickets across all workspaces."
            ),
        )
        .await;
        return;
    }
    match mahbot::pipeline::board::store()
        .archive_all_done_and_cancelled()
        .await
    {
        Ok(n) => {
            send_telegram_reply(
                msg,
                format!(
                    "Archived {n} ticket{} across all workspaces.",
                    if n == 1 { "" } else { "s" }
                ),
            )
            .await;
        }
        Err(e) => send_telegram_reply(msg, format!("Failed to archive tickets: {e}")).await,
    }
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
        _ => {
            // Dismisses the loading spinner and surfaces a toast for unknown actions, so a stale
            // inline keyboard (e.g. a removed role-picker button) never fails silently.
            answer_telegram_callback(
                &msg,
                Some("This action is no longer available.".to_string()),
            )
            .await;
            tracing::warn!(action = %action, "Unknown __act__ action — ignoring");
        }
    }
}

/// Serializes the per-user model-picker callbacks so rapid taps apply in tap order and each
/// picker's in-place ✓ refresh lands on the row that won. Held across the HTTP that follows a tap
/// (the model write and the markup refresh) — the accepted tradeoff for deterministic ordering,
/// since human-paced taps and the client timeout cap any stall. The tap's acknowledgement and
/// posting a picker are not under it: neither is part of the order.
static PICKER_WRITE_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

/// Validates the payload (image models against the catalog, fail-open), writes the user's
/// `image_gen_model`/`video_model` column, refreshes the pressed picker's ✓ in place, and
/// acknowledges with a toast.
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

/// Rebuild the keyboard and replace the one the callback's own message carries. Any failure (no
/// source-message identity, a deleted message, an identical keyboard, a transport error) is
/// cosmetic — the pick is applied — so it is logged at debug and skipped.
async fn refresh_pressed_models_keyboard(msg: &ChannelMessage, is_image: bool) {
    let (Some(chat_id), Some(message_id)) = (&msg.chat_id, msg.message_id) else {
        return;
    };
    let keyboard =
        mahbot::channels::telegram::build_models_keyboard(is_image, &msg.user_name).await;
    if let Some(channel) = mahbot::channel_registry().get("telegram")
        && let Some(tc) = channel
            .as_any()
            .downcast_ref::<mahbot::channels::telegram::TelegramChannel>()
        && let Err(e) = tc.edit_reply_markup(chat_id, message_id, &keyboard).await
    {
        tracing::debug!(?e, "model picker keyboard refresh skipped");
    }
}

/// Validation order: payload presence, image-model catalog check (fail-open when the catalog is
/// unavailable, matching the generation tool), then user existence (a stale keyboard tapped by an
/// unknown sender must not mint a bare users row), then the direct column write (no config warmup).
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

/// With no `callback_query_id` (a non-Telegram channel) this is a no-op.
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
    // A collected burst is one message whose `parts` are the individual messages in send order
    // (`channels::telegram_group`); a lone message is the one-part case of the same flow. Either
    // way parts are never empty.
    let user_name = msg.user_name.clone();
    let channel = msg.channel.clone();
    let reply_target = msg.reply_target.clone();
    // The urgency the surface decided when it took the mark off the typed text; the envelope
    // carries it for a lone message, an album and a burst alike (`channels::take_urgent`).
    let urgent = msg.urgent;
    let mut parts = if msg.parts.is_empty() {
        vec![msg]
    } else {
        msg.parts
    };

    let ws = mahbot::users::resolve_workspace_for_user_name(&user_name).await;

    // Every account routes to the single Assistant role, which always works in the user's
    // personal workspace regardless of the selected one — resolved before enrichment and before
    // the burst's workspace is set, so uploads, broadcast, persist and chat_history agree.
    let ws = mahbot::users::effective_workspace_for_role(mahbot::Role::Assistant, ws, &user_name);

    // On every part, so downstream broadcasts and chat_history writes carry the effective one.
    for part in &mut parts {
        part.workspace.clone_from(&ws.name);
    }

    // An urgent message cuts the assistant's running turn NOW — on staging, before anything about
    // the message is understood, so the cut never waits for attachment extraction. The turn it
    // creates answers it with everything already waiting. The guard is held until that hand-over:
    // dropping it ends the wait the announcement started.
    let announcement = if urgent && !mahbot::shutdown::aborting() {
        mahbot::agent::message_router::announce_urgent(
            &mahbot::agent::message_router::user_message_agent_id(
                &user_name,
                mahbot::Role::Assistant,
                &ws.name,
            ),
        )
    } else {
        None
    };

    // The raw typed text of every part, so chat_history stores what the user wrote rather than the
    // enriched form (which can carry large data URIs), captured with the log line before
    // enrichment rewrites the content.
    let mut originals = Vec::with_capacity(parts.len());
    for part in &parts {
        // A dictated message is named without its content: the transcript is not logged.
        if part.dictated {
            tracing::info!("💬 [{}] from {}: (dictated)", part.channel, part.user_name);
        } else {
            tracing::info!(
                "💬 [{}] from {}: {}",
                part.channel,
                part.user_name,
                mahbot::util::truncate(&part.content, 80)
            );
        }
        originals.push(part.content.clone());
    }

    // Before broadcast, so the GUI receives the enriched form instead of raw `[AUDIO:path]`
    // markers. Images become native data-URI parts carrying the original bytes (re-encoded to a
    // bounded JPEG only over the encoded-payload cap); videos become a workspace copy plus the
    // clip's timeline; a recording becomes a workspace copy plus its transcript. The parts are
    // independent and this is the expensive step, so the whole burst is enriched at once.
    futures_util::future::join_all(
        parts
            .iter_mut()
            .map(|part| mahbot::channels::enrich_message(part, Some(ws.as_path()))),
    )
    .await;

    // Each part in send order, so the user's own messages keep their exact appearance while the
    // assistant receives the whole burst as one request below. The bubbles appear once the slowest
    // part is enriched, which keeps them in send order instead of interleaved with a late
    // transcription.
    // Every part belongs to the one incoming message, so they share its page allowance.
    let mut links = mahbot::channels::LinkBudget::per_message();
    for (part, original_content) in parts.iter_mut().zip(&originals) {
        // Broadcast/mirror. The owner's bubble shows the display form — a clip's timeline without
        // the frames the model reads as pictures. `persist_content` decides what reaches chat
        // history: the raw original text, or the data-URI-stripped content when the markers name
        // inbound attachments whose temp paths must not be persisted.
        let display = part.display_content.as_deref().unwrap_or(&part.content);
        let content_for_history = mahbot::channels::persist_content(original_content, display);
        broadcast_and_persist_incoming_message(part, display, &content_for_history).await;

        // After broadcast, so AI summaries don't appear in the user's own bubble.
        let enriched = mahbot::channels::enrich_links(&part.content, &mut links).await;
        if let Cow::Owned(s) = enriched {
            tracing::info!(
                channel = %part.channel,
                user_name = %part.user_name,
                "Link enricher: prepended URL summaries to message"
            );
            part.content = s;
        }

        // Prepend AFTER link enrichment (so URL summaries never see it) and BEFORE routing.
        // Broadcast + persist ran above on marker-free content.
        if let Some(reply) = part.reply_reference.clone() {
            part.content = mahbot::channels::apply_reply_marker(&part.content, &reply);
        }
    }

    let content = mahbot::channels::compose_group_content(parts);

    // Every account routes to the single Assistant, so there is no role to resolve. Every message
    // resolves to a deterministic agent ID and routes through the per-agent consumer loop —
    // different IDs, different loops, true parallelism. The whole burst is one request.
    message_router::route_user_message(
        content,
        ws.name,
        user_name,
        channel,
        mahbot::Role::Assistant,
        Some(reply_target),
        urgent,
    )
    .await;
    // The message is queued: the turn cut for it can stop waiting.
    drop(announcement);
}
