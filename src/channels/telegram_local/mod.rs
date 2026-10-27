//! The self-hosted Telegram Bot API server — the optional mode in which the bot
//! is served by a server this product installs, supervises and updates itself
//! instead of Telegram's cloud API.
//!
//! Why it exists: the cloud caps a bot at 20 MB per download and 50 MB per
//! upload, so a video, archive or document above those arrives refused. A local
//! server has no download cap and accepts uploads up to 2000 MB, which is the
//! whole point of the switch.
//!
//! ## The durable decision
//!
//! `config_kv.telegram_local_server` is the decision (`"local"`/`"cloud"`) and
//! `config_kv.telegram_local_migration` is the journal of a switch that has not
//! finished. Both are written before the first step runs, so a kill, a crash or
//! a reboot resumes exactly where the switch stopped ([`migrate`]) — no state
//! exists in which the product has forgotten that a switch was in progress.
//!
//! ## One server, always
//!
//! [`server`] owns the process: started with the service, adopted or stopped at
//! boot when a previous process survived, probed every minute, restarted with
//! backoff, and stopped on the way out. The record proves itself by its process
//! start time before anything is done with its pid — an unprovable record is
//! dropped and its pid left alone; a port is only ever used for a process proven
//! that way, and a fresh one is chosen for each start. Two pollers on one bot
//! token split its updates, so a surviving process is never joined by a second
//! one.
//!
//! ## What the transport uses
//!
//! [`serving_mode`] follows the side the bot is physically on: the journal records
//! where the switch started and the two irreversible facts (leaving the cloud,
//! leaving the local server), and nothing else can move the transport. A request
//! for the direction already under way keeps that origin; the opposite one begins a
//! journey from the side the bot is on now.
//! [`local_endpoint`] is the only endpoint a listener may use in [`Mode::Local`],
//! and it is published only once the bot has actually arrived there. There is
//! deliberately no fallback to the cloud: once the bot has left it, a cloud call
//! would log it back in and split its updates.
//!
//! ## What the interface is told
//!
//! [`Health`] reports installing, starting and failures, and nothing else: a
//! local server that is installed and answering has no state word, because "the
//! process answers" says nothing about Telegram and a bot nobody writes to is not
//! a fault. A failure carries its reason. A switch waiting for the owner keeps its
//! own cause in the journal, so it survives a restart and is not overwritten by a
//! later probe.

mod install;
mod migrate;
mod server;

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::config::CONFIG;
use crate::util::UnwrapPoison;

/// Which Bot API the bot is served by.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    Cloud,
    Local,
}

impl Mode {
    /// The mode a stored value names, strictly: anything else names none. A
    /// journal target that cannot be read must not be guessed at.
    pub(super) fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            value if value.eq_ignore_ascii_case("local") => Some(Mode::Local),
            value if value.eq_ignore_ascii_case("cloud") => Some(Mode::Cloud),
            _ => None,
        }
    }

    /// The configured mode, which falls back to the cloud: an absent or unreadable
    /// setting means the owner never asked for anything else.
    pub(super) fn from_value(value: Option<&str>) -> Self {
        value.and_then(Self::parse).unwrap_or(Mode::Cloud)
    }

    /// The stored spelling of a mode; the counterpart of [`Mode::parse`].
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Mode::Cloud => "cloud",
            Mode::Local => "local",
        }
    }
}

/// The mode the owner asked for, from the durable config row.
fn configured_mode() -> Mode {
    Mode::from_value(CONFIG.telegram_local_server().as_deref())
}

/// The mode the transport serves right now.
pub(crate) fn serving_mode() -> Mode {
    serving_from(configured_mode(), migrate::current().as_ref())
}

/// The mode the transport serves, derived from the durable decision and the
/// pending switch — never from the mode a switch is heading to.
///
/// A switch is a journey between two sides, and the bot is physically on one of
/// them at every moment: on the side it started from until the journal records
/// that it left, on the other one after. That is why the journal keeps the
/// starting side: a switch back starts from the local server with neither fact
/// recorded, which the target alone cannot tell from a switch that has not yet
/// left the cloud. Nothing else can move the transport, so a killed switch can
/// never route it to a server the bot is not on.
fn serving_from(configured: Mode, switch: Option<&migrate::Journal>) -> Mode {
    let Some(switch) = switch else {
        return configured;
    };
    let Some(from) = switch.origin_mode() else {
        return configured;
    };
    let left = match from {
        Mode::Cloud => switch.cloud_logged_out,
        Mode::Local => switch.local_left,
    };
    if left {
        match from {
            Mode::Cloud => Mode::Local,
            Mode::Local => Mode::Cloud,
        }
    } else {
        from
    }
}

/// What the interface shows about the feature.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Health {
    /// Cloud mode: nothing local is involved.
    Cloud,
    /// Installing or starting the server, before it answers.
    Starting,
    /// The process answers: nothing to report.
    Running,
    /// The local server is not serving the bot; [`Status::last_error`] names the
    /// cause once a step has named it.
    Down,
}

/// The feature's state, read by the Settings page at render time.
///
/// Two kinds of field live here. `target`, `serving`, `migrating`, `blocked` and
/// `credentials_set` are derived by [`status`] from the journal and the config on
/// every read, so a stored value of theirs means nothing; the rest is what the
/// supervisor last observed and is stored between reads.
#[derive(Clone, Debug)]
pub(crate) struct Status {
    /// The mode the owner asked for.
    pub(crate) target: Mode,
    /// The mode the transport uses right now.
    pub(crate) serving: Mode,
    pub(crate) health: Health,
    /// The last failure, in words the owner can act on.
    pub(crate) last_error: Option<String>,
    pub(crate) port: Option<u16>,
    /// A switch is in progress.
    pub(crate) migrating: bool,
    /// Why the switch is waiting for the owner, when it is.
    pub(crate) blocked: Option<String>,
    /// The credentials are on file.
    pub(crate) credentials_set: bool,
}

impl Status {
    fn idle() -> Self {
        Self {
            target: Mode::Cloud,
            serving: Mode::Cloud,
            health: Health::Cloud,
            last_error: None,
            port: None,
            migrating: false,
            blocked: None,
            credentials_set: false,
        }
    }
}

static STATUS: OnceLock<std::sync::Mutex<Status>> = OnceLock::new();

fn status_cell() -> &'static std::sync::Mutex<Status> {
    STATUS.get_or_init(|| std::sync::Mutex::new(Status::idle()))
}

/// The current state.
///
/// The stored fields (health, port, last error) come from the running
/// supervisor; the switch's facts are derived here, once per call, from the one
/// journal read. A pending switch stays `migrating` while it is blocked too: it is
/// waiting for the owner, not finished. A journal whose only remaining work is the
/// chat message ([`migrate::Journal::finishing`]) is not migrating — nothing about
/// the bot is in flight.
pub(crate) fn status() -> Status {
    let switch = migrate::current();
    let mut snapshot = status_cell().lock().unwrap_poison().clone();
    snapshot.credentials_set = credentials_set();
    snapshot.migrating = switch.as_ref().is_some_and(|switch| !switch.finishing());
    snapshot.target = switch
        .as_ref()
        .and_then(migrate::Journal::target_mode)
        .unwrap_or_else(configured_mode);
    snapshot.serving = serving_from(configured_mode(), switch.as_ref());
    snapshot.blocked = switch.as_ref().and_then(|switch| switch.blocked.clone());
    if let Some(reason) = snapshot.blocked.clone() {
        // The journal's own cause outranks any later probe: a switch waiting for
        // the owner is not an ordinary outage, and it stays readable after a
        // restart and under a liveness tick.
        snapshot.last_error = Some(reason);
    }
    if snapshot.serving == Mode::Cloud {
        // The bot is on the cloud: whatever a local server last reported is not
        // what carries it, and reporting the local side's health would describe a
        // transport the owner is not using.
        snapshot.health = Health::Cloud;
        snapshot.port = None;
    } else if snapshot.health == Health::Cloud {
        // The idle value: a local mode whose server has not been observed yet is
        // starting, never the cloud's.
        snapshot.health = Health::Starting;
    }
    snapshot
}

/// Replace the state and let the interface redraw.
fn set_status(update: impl FnOnce(&mut Status)) {
    {
        let mut guard = status_cell().lock().unwrap_poison();
        update(&mut guard);
    }
    crate::runtime_events::publish(crate::runtime_events::RuntimeEvent::TelegramLocal);
}

/// The server's Bot API base, when it is serving.
///
/// `None` in local mode means no endpoint exists right now, and a listener must
/// not run at all — pointing it at the cloud would undo the switch.
pub(crate) fn local_endpoint() -> Option<String> {
    let port = status_cell().lock().unwrap_poison().port?;
    Some(format!("http://127.0.0.1:{port}"))
}

/// The server's own directory tree: the only place a local-mode hand-over may
/// come from.
pub(crate) fn incoming_files_dir() -> Option<std::path::PathBuf> {
    server::download_root()
}

/// The `api_id`/`api_hash` pair as configured, normalised the way the server is
/// launched with it — the one place that normalises, so the value that was
/// validated is the value that is used.
fn credentials() -> (String, String) {
    let api_id = CONFIG.telegram_api_id().unwrap_or_default();
    let api_hash = CONFIG.telegram_api_hash().unwrap_or_default();
    (
        api_id.trim().to_string(),
        api_hash.trim().to_ascii_lowercase(),
    )
}

fn credentials_set() -> bool {
    let (api_id, api_hash) = credentials();
    validate_credentials(&api_id, &api_hash).is_ok()
}

/// The per-field shape problems of a pasted pair, each under the field it is
/// about. Shape only — whether Telegram accepts the pair is decided at the first
/// real login, and is reported as the cause Telegram names.
pub(crate) fn credential_problems(
    api_id: &str,
    api_hash: &str,
) -> (Option<String>, Option<String>) {
    (parse_api_id(api_id).err(), parse_api_hash(api_hash).err())
}

/// Both fields of a well-formed pair. Shape only — whether Telegram accepts the
/// pair is decided at the first real login, and is reported as the cause Telegram
/// names. The values a switch launches are [`credentials`]'s.
pub(crate) fn validate_credentials(api_id: &str, api_hash: &str) -> Result<(), String> {
    parse_api_id(api_id)?;
    parse_api_hash(api_hash)
}

/// `api_id` as the server takes it, or what is wrong with it. Shape only: an empty
/// field, a pasted sentence, a number too large for the protocol.
fn parse_api_id(api_id: &str) -> Result<(), String> {
    let api_id = api_id.trim();
    if api_id.is_empty() {
        return Err("api_id is missing — copy the number my.telegram.org shows".to_string());
    }
    if !api_id.chars().all(|c| c.is_ascii_digit()) {
        return Err(
            "api_id must be a number — my.telegram.org shows it as digits only".to_string(),
        );
    }
    match api_id.parse::<u32>() {
        Ok(0) => Err("api_id must be a number greater than zero".to_string()),
        Ok(_) => Ok(()),
        Err(_) => Err("api_id is not a number — copy it again from my.telegram.org".to_string()),
    }
}

/// `api_hash`'s shape, or what is wrong with it. Case is not checked: a hex value
/// means the same in either case, and [`credentials`] is what normalises it.
fn parse_api_hash(api_hash: &str) -> Result<(), String> {
    let api_hash = api_hash.trim();
    if api_hash.is_empty() {
        return Err(
            "api_hash is missing — copy the 32-character value from my.telegram.org".to_string(),
        );
    }
    if api_hash.len() != 32 || !api_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "api_hash must be 32 hexadecimal characters — the value pasted has {}",
            api_hash.len()
        ));
    }
    Ok(())
}

/// What the boot of the message pipeline should do about the listener.
pub(crate) enum Boot {
    /// Start the channel against the base the current mode names.
    Start,
    /// A resumed switch already started it.
    Running,
    /// The local mode has no server yet; the supervisor starts one and then the
    /// listener. Nothing may talk to the cloud in the meantime.
    Waiting,
}

/// How often the process is checked: a tokenless request, cheap enough to run
/// every minute.
const LIVENESS_INTERVAL: Duration = Duration::from_mins(1);

/// How often the local server is asked whether it takes the bot: `getMe` is the
/// one question a refusal or a rejected pair answers.
const REACHABILITY_INTERVAL: Duration = Duration::from_mins(5);

/// How often the release stream is asked for a newer server.
const UPDATE_INTERVAL: Duration = Duration::from_hours(1);

/// How often the server's own disk use is trimmed.
const HOUSEKEEPING_INTERVAL: Duration = Duration::from_hours(1);

/// Restart delays after a failed start. A start that keeps failing settles at one
/// attempt an hour instead of spinning; a server that dies between ticks is
/// restarted on the liveness tick, which is once a minute at worst.
const RESTART_BACKOFF: [Duration; 4] = [
    Duration::from_secs(5),
    Duration::from_secs(30),
    Duration::from_mins(2),
    Duration::from_mins(10),
];
const SETTLED_RETRY: Duration = Duration::from_hours(1);

/// How often a switch waiting for the owner is reconsidered: often enough that a
/// cause which is gone (an outage that ended, a pair that was replaced elsewhere)
/// clears by itself, rarely enough that a cause which is not gone costs nothing.
const BLOCKED_RETRY: Duration = Duration::from_hours(1);

/// Whether the owner has already been told about the current failure episode.
static FAILURE_NOTIFIED: AtomicBool = AtomicBool::new(false);

/// A build that is installed and waiting for the supervisor to restart the server
/// onto it. The download itself runs beside the loop ([`start_server_update`]).
static UPDATE_INSTALLED: AtomicBool = AtomicBool::new(false);

/// One update check at a time: a download outlasting the interval must not be
/// started beside itself.
static UPDATE_RUNNING: AtomicBool = AtomicBool::new(false);

/// Prepare the server for this start: adopt or stop what a previous run left,
/// finish a switch that did not, and start a server the current mode needs.
///
/// Called before the Telegram channel is built, so a bot whose mode is local is
/// only ever started against a server that exists.
pub(crate) async fn prepare_for_boot() -> Boot {
    server::adopt_or_reap().await;
    if let Some(instance) = server::read_record()
        && server::alive(&instance).await
    {
        publish_endpoint(instance.port);
    }

    if migrate::pending() {
        migrate::resume().await;
        if crate::channel_registry().get("telegram").is_some() {
            return Boot::Running;
        }
    }

    if configured_mode() == Mode::Local
        && !migrate::pending()
        && let Err(reason) = ensure_local().await
    {
        report_failure(&reason);
    }
    // One question, the same one the channel itself answers when it is built from
    // the config: the cloud always has a base, the local mode only once a server
    // has published its endpoint. The supervisor is what closes the gap.
    if serving_mode() == Mode::Local && local_endpoint().is_none() {
        Boot::Waiting
    } else {
        Boot::Start
    }
}

/// Start (or confirm) the local server for the configured credentials.
async fn ensure_local() -> Result<(), String> {
    if crate::config::CONFIG.telegram_bot_token().is_none() {
        return Err("Telegram has no bot token configured, so there is no bot to serve".into());
    }
    let (api_id, api_hash) = credentials();
    validate_credentials(&api_id, &api_hash)?;
    set_status(|status| status.health = Health::Starting);
    let fingerprint = server::credentials_fingerprint(&api_id, &api_hash);
    let instance = server::ensure(&fingerprint).await?;
    publish_endpoint(instance.port);
    Ok(())
}

/// Record the endpoint the transport may use: the server answers, so the state
/// is healthy and the current failure episode, if any, is over.
pub(super) fn publish_endpoint(port: u16) {
    set_status(|status| {
        status.port = Some(port);
        status.health = Health::Running;
        status.last_error = None;
    });
    FAILURE_NOTIFIED.store(false, Ordering::SeqCst);
}

/// Forget the endpoint: the server it named is gone, and nothing may be built
/// against it. In local mode that means no listener at all until one exists.
pub(super) fn clear_endpoint() {
    set_status(|status| status.port = None);
}

/// Record a failure and tell the owner once per episode.
fn report_failure(reason: &str) {
    set_status(|status| {
        status.health = Health::Down;
        status.last_error = Some(reason.to_string());
    });
    tracing::warn!("telegram-bot-api: {reason}");
    if !FAILURE_NOTIFIED.swap(true, Ordering::SeqCst) {
        let text = format!(
            "Telegram is unreachable: {reason}. MahBot keeps trying. \
             If it does not come back, switch to the cloud Bot API in Settings → Integrations."
        );
        tokio::spawn(async move { send_owner_notice(&text).await });
    }
}

/// The Bot API's own words for a failed call, without the transport's wrapper.
/// The wrapper is a credential-shaped `key: value` pair, which every log and
/// tool scrubber masks into nonsense ("Unau*[REDACTED]"); the caller wants the
/// cause, not that.
fn api_cause(reason: &str) -> &str {
    reason
        .split_once(crate::channels::telegram::REJECTED_TOKEN_PREFIX)
        .map_or(reason, |(_, cause)| cause)
}

/// Send one chat notice to the owner's own Telegram chat, when there is one.
///
/// Best-effort: `false` means no owner chat could be resolved or the send failed,
/// and only the caller knows whether that is worth persisting.
async fn send_owner_notice(text: &str) -> bool {
    let Some(target) = crate::self_update::resolve_admin_telegram_target().await else {
        tracing::info!("telegram-bot-api: no owner chat to notify");
        return false;
    };
    match crate::channels::telegram::send_direct(&target, text.to_string(), None).await {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(error = %e, "telegram-bot-api: the owner could not be notified");
            false
        }
    }
}

/// Whether a local login was refused because Telegram rejected the pasted pair —
/// the one outcome that only re-entering it can fix.
///
/// The server answers in about a second with `Unauthorized: invalid
/// api-id/api-hash`; retrying with the same values cannot succeed, and the owner
/// has to correct them. The shape problems the switch blocks on before it starts
/// name the pair the other way round (`api_id`, `api_hash`), and those are the
/// same repair: nothing else can change either field.
pub(crate) fn credentials_rejected(reason: &str) -> bool {
    // Only the pair's own words: a wrong bot token also answers `Unauthorized`,
    // and the remedy for that is the token field, not this dialog.
    let reason = reason.to_ascii_lowercase();
    reason.contains("api-id")
        || reason.contains("api-hash")
        || reason.contains("api_id")
        || reason.contains("api_hash")
}

/// Ask for a switch to `target`: the decision is recorded durably before
/// anything runs, and the switch itself is resumed by the supervisor if it does
/// not finish here.
pub(crate) async fn request_switch(target: Mode) -> anyhow::Result<()> {
    if target == Mode::Local {
        if CONFIG.telegram_bot_token().is_none() {
            anyhow::bail!("Telegram is not configured yet: add a bot token first");
        }
        let (api_id, api_hash) = credentials();
        validate_credentials(&api_id, &api_hash).map_err(anyhow::Error::msg)?;
    }
    // Liveness is the supervisor's business: asking for the mode that is already
    // in force must not run the switch again — a second cutover would log the bot
    // out of the cloud a second time while it is being served locally. A finished
    // switch whose chat message is still pending is settled too: its own notice is
    // on its way.
    let settled = configured_mode() == target && (!migrate::pending() || migrate::finishing());
    if settled {
        return Ok(());
    }
    migrate::start(target).await?;
    Ok(())
}

/// The supervisor: keeps the server alive, updated and clean, and finishes any
/// switch a crash interrupted.
///
/// Its whole cost when everything is fine is one tokenless request a minute, and
/// a `getMe` every five minutes to catch a server that answers but refuses the
/// bot.
pub(crate) async fn run_supervisor() {
    let mut backoff = 0usize;
    // `None` means nothing is scheduled: the steady state wakes on the liveness
    // minute alone, and only a restart that failed wakes the loop earlier.
    let mut retry_at: Option<tokio::time::Instant> = None;
    let mut reachability_at = tokio::time::Instant::now();
    let mut update_at = tokio::time::Instant::now();
    let mut housekeeping_at = tokio::time::Instant::now();
    let mut blocked_at = tokio::time::Instant::now();

    loop {
        // A scheduled restart waits its own delay; everything else is the minute.
        // Without this the ladder would be quantised to whole minutes.
        let wait = if configured_mode() == Mode::Local && !migrate::pending() {
            retry_at.map_or(LIVENESS_INTERVAL, |at| {
                at.saturating_duration_since(tokio::time::Instant::now())
                    .min(LIVENESS_INTERVAL)
            })
        } else {
            LIVENESS_INTERVAL
        };
        if !crate::shutdown::sleep_or_shutdown(wait).await {
            break;
        }
        // A switch in flight owns the server: it starts, restarts and removes the
        // process on its own, so the supervisor stays out of its way until it is
        // done. A switch *waiting for the owner* is different — whatever serves the
        // bot meanwhile still has to be kept alive, or the state that needs the
        // owner's help would also take the bot down with it. It is reconsidered
        // slowly: a cause that is gone clears by itself, and one that is not costs
        // one attempt an hour.
        if migrate::pending() {
            if !migrate::is_blocked() {
                migrate::resume().await;
            } else if tokio::time::Instant::now() >= blocked_at {
                blocked_at = tokio::time::Instant::now() + BLOCKED_RETRY;
                migrate::resume().await;
            }
        }
        // Read after the resume, never before: a switch that finished in this tick
        // must not look like one still running. A journal carrying only the pending
        // chat message of a finished switch is not one either.
        let switching = migrate::pending() && !migrate::is_blocked() && !migrate::finishing();
        // A rejected pair cannot log the bot in, so nothing this tick could start
        // would succeed: the reconnect waits for the values to change, which is
        // either the owner's dialog or the switch's own retry.
        let waiting_on_pair = migrate::waiting_on_rejected_pair();

        // The guards come first: consuming the flag before them would drop an
        // installed build whenever a switch is in flight, and the next hourly check
        // would download the same artifact again.
        if serving_mode() == Mode::Local
            && !switching
            && UPDATE_INSTALLED.swap(false, Ordering::SeqCst)
        {
            apply_server_update().await;
        }

        if serving_mode() == Mode::Local {
            if !switching && !waiting_on_pair {
                let mut up = match server::read_record() {
                    Some(instance) => server::alive(&instance).await,
                    None => false,
                };
                if up {
                    backoff = 0;
                    retry_at = None;
                } else if retry_at.is_none_or(|at| tokio::time::Instant::now() >= at) {
                    match ensure_local().await {
                        Ok(()) => {
                            backoff = 0;
                            retry_at = None;
                            up = true;
                        }
                        Err(reason) => {
                            report_failure(&reason);
                            let delay = RESTART_BACKOFF
                                .get(backoff)
                                .copied()
                                .unwrap_or(SETTLED_RETRY);
                            backoff = backoff.saturating_add(1);
                            retry_at = Some(tokio::time::Instant::now() + delay);
                        }
                    }
                }
                // A restarted process gets a fresh port, so the listener is rebuilt
                // whenever the endpoint it was built for is not the one serving — that,
                // and a boot that raced the server, is why a live server and a live
                // listener are stated separately.
                if up
                    && local_endpoint().is_some()
                    && crate::channels::telegram::listener_base() != local_endpoint()
                {
                    start_listener_after_recovery();
                }
            }
        } else if !switching && server::read_record().is_some() {
            // The cloud owns no local process; one left behind by a revert is
            // stopped here rather than nobody's business.
            server::stop().await;
        }

        if tokio::time::Instant::now() >= reachability_at {
            reachability_at = tokio::time::Instant::now() + REACHABILITY_INTERVAL;
            check_reachability().await;
        }

        if serving_mode() == Mode::Local
            && !switching
            && !waiting_on_pair
            && tokio::time::Instant::now() >= update_at
        {
            update_at = tokio::time::Instant::now() + UPDATE_INTERVAL;
            start_server_update();
        }

        if tokio::time::Instant::now() >= housekeeping_at {
            housekeeping_at = tokio::time::Instant::now() + HOUSEKEEPING_INTERVAL;
            server::sweep_state_files().await;
        }
    }
}

/// Start the listener against the base the current mode names, from a task: the
/// supervisor must not wait for a listen loop that never returns.
fn start_listener_after_recovery() {
    tokio::spawn(async move {
        if let Err(e) = crate::channels::telegram::restart_telegram_listener(
            CONFIG.telegram_bot_token().as_deref(),
        ) {
            report_failure(&format!("the bot could not be reconnected: {e}"));
        }
    });
}

/// Probe the local server and report a failure the settings line can name.
///
/// Two facts hide behind "the server works": the process answering on its port,
/// and the server accepting the bot against it. The first is probed tokenlessly,
/// the second with `getMe`. Either failure is an error with its reason; both
/// answering is the healthy state, which has nothing to say — a bot nobody writes
/// to looks exactly like a busy one from here.
async fn check_reachability() {
    // The probe is about the local server, so it runs only while that is what serves
    // — and not while a switch is in flight, which owns the diagnosis: it knows
    // which side the bot is on, and a probe of the wrong side would overwrite its
    // cause. A switch waiting for the owner does not own it: the local server is
    // what carries the bot then, and a failure of it is exactly what the owner must
    // see.
    if serving_mode() != Mode::Local
        || (migrate::pending() && !migrate::is_blocked() && !migrate::finishing())
    {
        return;
    }
    let Some(port) = status().port else {
        // Nothing serves locally, and a switch waiting for the owner is why: the
        // status must not keep claiming a start that is not running.
        if migrate::is_blocked() {
            set_status(|status| status.health = Health::Down);
        }
        return;
    };
    if !server::probe(port).await {
        set_status(|status| {
            status.health = Health::Down;
            status.last_error = Some("the local Bot API server stopped answering".into());
        });
        return;
    }
    // Whether the server knows the token is a fact it can answer, and a refusal is
    // a failure like any other: the reason the owner needs is the server's own.
    if let (Some(base), Some(token)) = (local_endpoint(), CONFIG.telegram_bot_token())
        && let Err(e) = crate::channels::telegram::TelegramChannel::probe_bot(&base, &token).await
    {
        let failure = e.to_string();
        let reason = api_cause(&failure);
        set_status(|status| {
            status.health = Health::Down;
            status.last_error = Some(format!("the local server refused the bot: {reason}"));
        });
        return;
    }
    set_status(|status| {
        status.health = Health::Running;
        status.last_error = None;
    });
    FAILURE_NOTIFIED.store(false, Ordering::SeqCst);
}

/// Check the release stream and install a newer build, beside the loop.
///
/// Off the loop on purpose: a tens-of-megabytes download on a slow link would
/// otherwise stop the liveness probe, the restart of a crashed server and the
/// reachability check for its whole duration. The installed build is applied by
/// [`apply_server_update`] on the next tick.
fn start_server_update() {
    if UPDATE_RUNNING.swap(true, Ordering::SeqCst) {
        return;
    }
    tokio::spawn(async {
        install_newer_build().await;
        UPDATE_RUNNING.store(false, Ordering::SeqCst);
    });
}

/// Install a newer build when the release stream offers one, one-way.
async fn install_newer_build() {
    let Some(current) =
        server::read_record().and_then(|record| record.version.as_deref().and_then(parse_version))
    else {
        // Without a version to compare there is nothing to decide: downloading the
        // newest build on every tick would be the alternative, and re-installing the
        // same server repeatedly is not an update.
        tracing::debug!("telegram-bot-api: the running server reports no comparable version");
        return;
    };
    match install::latest().await {
        Ok(release) if current < release.version => {
            let version = release.version.clone();
            tracing::info!(%version, "telegram-bot-api: updating the server");
            match install::install(release).await {
                Ok(_installed) => UPDATE_INSTALLED.store(true, Ordering::SeqCst),
                Err(reason) => tracing::warn!("telegram-bot-api: update failed: {reason}"),
            }
        }
        // One-way by construction: an older artifact is never installed over a
        // newer one, so a server that cannot read state written by a newer build is
        // never the one chosen.
        Ok(_) => {}
        // A release stream that cannot be read leaves the current build in place:
        // say so rather than let a server that never updates look up to date.
        Err(reason) => {
            tracing::warn!("telegram-bot-api: the release stream could not be read: {reason}");
        }
    }
}

/// Restart the server onto the build [`start_server_update`] installed.
///
/// The state directory is untouched, so the bot's session survives the swap: the
/// restarted process picks it up exactly where it was.
async fn apply_server_update() {
    server::stop().await;
    match ensure_local().await {
        Ok(()) => {
            // The new process listens on a fresh port, so the listener is rebuilt now
            // rather than at the next tick: the bot is only away for the restart.
            start_listener_after_recovery();
        }
        Err(reason) => report_failure(&reason),
    }
}

/// A release version out of the text the artifact reports (`Bot API 10.3`).
fn parse_version(reported: &str) -> Option<semver::Version> {
    let numeric = reported
        .split_whitespace()
        .find(|word| word.chars().next().is_some_and(|c| c.is_ascii_digit()))?;
    crate::util::managed_bin::parse_short_version(numeric)
}

/// Stop the server on the way out.
pub(crate) async fn stop_for_exit() {
    if configured_mode() == Mode::Local || server::read_record().is_some() {
        server::stop().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_malformed_pair_is_named_field_by_field() {
        // Both empty: each field says what it is missing, and neither complaint
        // stands in for the other.
        let (id, hash) = credential_problems("", "");
        assert!(
            id.as_deref()
                .is_some_and(|problem| problem.contains("api_id is missing"))
        );
        assert!(
            hash.as_deref()
                .is_some_and(|problem| problem.contains("api_hash is missing"))
        );

        let (id, hash) = credential_problems("12ab", &"a".repeat(32));
        assert!(
            id.as_deref()
                .is_some_and(|problem| problem.contains("must be a number"))
        );
        assert!(hash.is_none());

        let (id, hash) = credential_problems("0", &"a".repeat(32));
        assert!(
            id.as_deref()
                .is_some_and(|problem| problem.contains("greater than zero"))
        );
        assert!(hash.is_none());

        let (id, hash) = credential_problems("1234567", "abc");
        assert!(id.is_none());
        assert!(
            hash.as_deref()
                .is_some_and(|problem| problem.contains("32 hexadecimal"))
        );

        // The api_id's complaint is the one reported first, whichever field the
        // other problem is in.
        assert!(
            validate_credentials("", "")
                .unwrap_err()
                .contains("api_id is missing")
        );
        assert!(
            validate_credentials("12ab", "abc")
                .unwrap_err()
                .contains("must be a number")
        );
        assert!(
            validate_credentials("1234567", "abc")
                .unwrap_err()
                .contains("32 hexadecimal")
        );
        validate_credentials(" 1234567 ", "ABCDEF0123456789ABCDEF0123456789")
            .expect("a well-formed pair is accepted, in either case");
    }

    #[test]
    fn the_serving_mode_follows_the_side_the_bot_is_on() {
        // A switch requested from one side to the other, with the leave recorded or
        // not — the two facts a resumed switch reads.
        let switch =
            |from: &str, target: &str, cloud_logged_out: bool, local_left: bool| migrate::Journal {
                from: from.into(),
                target: target.into(),
                cloud_logged_out,
                local_left,
                ..Default::default()
            };

        // No switch: the configured mode is what serves.
        assert_eq!(serving_from(Mode::Cloud, None), Mode::Cloud);
        assert_eq!(serving_from(Mode::Local, None), Mode::Local);

        // Into the local server: the cloud keeps serving until the bot has left it.
        assert_eq!(
            serving_from(Mode::Local, Some(&switch("cloud", "local", false, false))),
            Mode::Cloud
        );
        assert_eq!(
            serving_from(Mode::Local, Some(&switch("cloud", "local", true, false))),
            Mode::Local
        );

        // Back to the cloud: the local server keeps serving until the bot has left
        // it. The starting side is what says so — the target alone cannot tell this
        // apart from a switch that has not left the cloud yet.
        assert_eq!(
            serving_from(Mode::Cloud, Some(&switch("local", "cloud", true, false))),
            Mode::Local
        );
        assert_eq!(
            serving_from(Mode::Cloud, Some(&switch("local", "cloud", true, true))),
            Mode::Cloud
        );

        // A row nothing wrote names no side: the configured mode stands, rather than
        // being read as a switch back to the cloud.
        let unreadable = migrate::Journal {
            target: "cloud".into(),
            ..Default::default()
        };
        assert_eq!(serving_from(Mode::Cloud, Some(&unreadable)), Mode::Cloud);
        assert_eq!(serving_from(Mode::Local, Some(&unreadable)), Mode::Local);
    }

    #[test]
    fn a_rejected_pair_is_recognised_by_what_the_server_says() {
        assert!(credentials_rejected(
            "Telegram rejected the bot token — Unauthorized: invalid api-id/api-hash"
        ));
        // The shape problems the switch blocks on before the server starts name
        // the pair the other way round; the same dialog is the only repair.
        assert!(credentials_rejected(
            "api_hash must be 32 hexadecimal characters — the value pasted has 12"
        ));
        // A wrong bot token is a different remedy and must not open this dialog.
        assert!(!credentials_rejected(
            "Telegram rejected the bot token — Unauthorized"
        ));
        assert!(!credentials_rejected("connection refused"));
    }

    #[test]
    fn the_cause_is_read_without_the_transports_wrapper() {
        assert_eq!(
            api_cause("Telegram rejected the bot token — Unauthorized"),
            "Unauthorized"
        );
        assert_eq!(
            api_cause("the Bot API could not be reached"),
            "the Bot API could not be reached"
        );
    }

    #[test]
    fn a_reported_version_reads_with_or_without_a_patch() {
        assert_eq!(
            parse_version("Bot API 10.3"),
            Some(semver::Version::new(10, 3, 0))
        );
        assert_eq!(
            parse_version("Bot API 10.3.1"),
            Some(semver::Version::new(10, 3, 1))
        );
        assert_eq!(parse_version("Telegram Bot API server"), None);
    }
}
