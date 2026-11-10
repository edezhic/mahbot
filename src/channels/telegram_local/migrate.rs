//! The switch between Telegram's cloud Bot API and the locally served one, as a
//! journal that survives a crash.
//!
//! A switch is a sequence of steps with exactly one irreversible point per direction:
//! leaving the cloud (`logOut`) on the way in, and leaving the local server on the way
//! back. The journal records the facts that cannot be re-derived — the side the switch
//! started from, whether the cloud `logOut` was issued and ran, and whether the bot has
//! left the local server — so a resumed switch neither re-enters the cloud nor routes
//! the transport to a server the bot is not logged into.

use serde::{Deserialize, Serialize};

use super::{Health, Mode, api_cause, credentials, set_status};

/// How a step that failed may be retried.
enum Step {
    /// Waiting for the owner: retrying cannot help.
    Blocked(String),
    /// Worth another attempt.
    Retry(String),
    /// The listener is still finishing the batch it held: the bot is served the whole time,
    /// so this is not a failure and must not spend a retry.
    Draining(String),
}

/// The switch a crash interrupted, as the config row keeps it.
#[derive(Default, Serialize, Deserialize)]
// One flag per durable fact of the switch: folding them into one state machine would lose
// the independent progress a resumed switch reads.
#[expect(clippy::struct_excessive_bools)]
pub(super) struct Journal {
    /// The side the bot was on when the switch was requested. Recorded rather than inferred: a
    /// switch back starts on the local server with neither leave recorded.
    #[serde(default)]
    pub(super) from: String,
    pub(super) target: String,
    /// The cloud `logOut` request was issued. Its answer may have been lost with the process,
    /// which is what tells a resumed attempt that `Unauthorized` means the bot has left.
    #[serde(default)]
    pub(super) cloud_logout_started: bool,
    /// The bot has left the cloud. Never repeat the cloud `logOut`: a stray cloud call
    /// re-logs the bot in there and splits its updates.
    #[serde(default)]
    pub(super) cloud_logged_out: bool,
    #[serde(default)]
    pub(super) local_left: bool,
    /// The intake record was cleared for this switch. Recorded so a failed purge is retried
    /// instead of leaving rows whose `update_id`s would swallow the next server's events.
    #[serde(default)]
    pub(super) intake_cleared: bool,
    /// Why the switch is waiting for the owner.
    #[serde(default)]
    pub(super) blocked: Option<String>,
    /// The `api_id`/`api_hash` fingerprint a rejection was about: a retry with the same pair
    /// cannot succeed, so it is not attempted.
    #[serde(default)]
    pub(super) blocked_credentials: Option<String>,
    #[serde(default)]
    pub(super) notice_sent: bool,
    /// The completion message, kept until it reaches the owner: a message that could not be
    /// delivered is retried from here — clearing the journal first would lose it.
    #[serde(default)]
    pub(super) notice: Option<String>,
    /// The completion message needs no further attempt (it arrived, or its attempts ran out).
    /// Recorded before the journal is cleared, so a failing clear cannot resend it.
    #[serde(default)]
    pub(super) notice_settled: bool,
    /// Delivery attempts for [`Self::notice`]: an unreachable owner chat must not retry forever.
    #[serde(default)]
    pub(super) notice_attempts: u32,
    /// Retryable attempts since the request.
    #[serde(default)]
    pub(super) attempts: u32,
}

impl Journal {
    /// The side this switch started from; `None` for a row nothing wrote.
    pub(super) fn origin_mode(&self) -> Option<Mode> {
        Mode::parse(&self.from)
    }

    /// The mode this switch is heading to. Strict on purpose: an unreadable target must not be
    /// read as "back to the cloud", the one direction that would route the transport to a server
    /// the bot has left.
    pub(super) fn target_mode(&self) -> Option<Mode> {
        Mode::parse(&self.target)
    }

    /// Whether the block may be reconsidered: a cause that is gone clears by itself, while a
    /// rejected pair that was not replaced cannot be retried into success.
    pub(super) fn block_may_clear(&self) -> bool {
        let Some(reason) = self.blocked.as_deref() else {
            return true;
        };
        if !super::credentials_rejected(reason) {
            return true;
        }
        self.blocked_credentials.as_deref() != Some(current_pair_fingerprint().as_str())
    }

    /// Whether the intake record has to be cleared: once the bot has actually left the side this
    /// journey started on. Clearing it while the bot is still there would process an unconfirmed
    /// update twice, and both servers number updates independently.
    pub(super) fn intake_must_be_purged(&self) -> bool {
        if self.intake_cleared {
            return false;
        }
        match self.origin_mode() {
            Some(Mode::Local) => self.local_left,
            _ => self.cloud_logged_out,
        }
    }

    /// Whether every step is done and only the chat message remains: nothing about the bot is in
    /// flight any more, but the journal carries the message across a restart.
    pub(super) fn finishing(&self) -> bool {
        self.notice.is_some() || self.notice_settled
    }
}

/// The journal, when a switch has not finished.
pub(super) fn current() -> Option<Journal> {
    serde_json::from_str(crate::config::CONFIG.telegram_local_migration()?.trim()).ok()
}

pub(super) fn pending() -> bool {
    current().is_some()
}

/// Whether the switch is waiting for the owner: a blocked switch is not retried on a tick.
pub(super) fn is_blocked() -> bool {
    current().is_some_and(|journal| journal.blocked.is_some())
}

/// Whether every step of the switch is done and only its chat message is pending.
pub(super) fn finishing() -> bool {
    current().is_some_and(|journal| journal.finishing())
}

/// The fingerprint of the pair configured right now: corrected values are retried, unchanged
/// ones are not.
fn current_pair_fingerprint() -> String {
    let (api_id, api_hash) = super::credentials();
    super::server::credentials_fingerprint(&api_id, &api_hash)
}

/// Whether the switch is waiting for a pair Telegram already refused: the one state in which
/// starting a listener or a server cannot succeed.
pub(super) fn waiting_on_rejected_pair() -> bool {
    current().is_some_and(|journal| journal.blocked.is_some() && !journal.block_may_clear())
}

async fn save(journal: &Journal) -> anyhow::Result<()> {
    let body = serde_json::to_string(journal)?;
    crate::config::persist_settled_string_field(
        crate::config::CONFIG_KEY_TELEGRAM_LOCAL_MIGRATION,
        &body,
    )
    .await?;
    Ok(())
}

async fn clear() -> anyhow::Result<()> {
    crate::config::persist_settled_string_field(
        crate::config::CONFIG_KEY_TELEGRAM_LOCAL_MIGRATION,
        "",
    )
    .await?;
    Ok(())
}

/// The single switch runs at a time: two would interleave their `logOut`s.
fn switch_lock() -> &'static tokio::sync::Mutex<()> {
    static SWITCH: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    SWITCH.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Record the decision and begin the switch.
///
/// The write is serialized with a running switch: the journal and the mode are
/// one decision, and a second request landing between them would leave the two
/// disagreeing about which way the bot is going.
pub(super) async fn start(target: Mode) -> anyhow::Result<()> {
    let _guard = switch_lock().lock().await;
    let journal = requested(current(), super::serving_mode(), target);
    // The journal first: it is the decision, and a crash between the two writes
    // leaves a switch with nothing recorded as its goal otherwise.
    save(&journal).await?;
    // The mode row is the same decision but a separate write, and it is re-asserted
    // when the switch completes ([`finish_notice`]): a failure here must not report
    // the switch as not started — the supervisor is already resuming it — so the
    // mismatch is logged and repaired instead of passed to the owner.
    if let Err(e) = crate::config::persist_settled_string_field(
        crate::config::CONFIG_KEY_TELEGRAM_LOCAL_SERVER,
        target.as_str(),
    )
    .await
    {
        tracing::warn!(
            error = %e,
            "Telegram: the switch mode could not be recorded; it will be written when the switch completes"
        );
    }
    set_status(|status| {
        status.health = Health::Starting;
        status.last_error = None;
    });
    tracing::info!(target = target.as_str(), "Telegram: switching Bot API");
    tokio::spawn(resume());
    Ok(())
}

/// The journal a request for `target` starts from.
///
/// A retry of the direction already under way keeps the journey's origin and every
/// step it recorded — re-stamping them would reset progress and could flip which
/// side the transport serves. A *reversal* is a new journey: it starts where the
/// bot actually is now, and its intake record starts unpurged because the side it
/// is going back to has accumulated its own ids since.
fn requested(existing: Option<Journal>, serving: Mode, target: Mode) -> Journal {
    let mut journal = existing.unwrap_or_default();
    let retry = journal.origin_mode().is_some() && journal.target_mode() == Some(target);
    if !retry {
        journal.from = serving.as_str().to_string();
        // Every step fact describes the side this journey started from, so none of
        // it survives a reversal: a cloud `logOut` the other direction issued would
        // let this one cut over before it left, and a leave it recorded would flip
        // which side serves. A reversal's intake record starts unpurged too — the
        // side it is going back to has accumulated its own ids since.
        journal.cloud_logout_started = false;
        journal.cloud_logged_out = false;
        journal.local_left = false;
        journal.intake_cleared = false;
    }
    journal.target = target.as_str().to_string();
    journal.blocked = None;
    journal.blocked_credentials = None;
    journal.attempts = 0;
    journal.notice_sent = false;
    // A message a previous journey left undelivered belongs to that journey: this
    // one produces its own when it finishes.
    journal.notice = None;
    journal.notice_settled = false;
    journal.notice_attempts = 0;
    journal
}

/// How many retryable attempts a switch gets before it is treated as blocked and
/// the owner is told once. Long enough to outlast the cloud's ~10-minute refusal
/// to take a bot back after a `logOut`, short enough that a mute bot is explained.
const SWITCH_ATTEMPT_LIMIT: u32 = 15;

/// Continue the pending switch, from whatever step it reached.
pub(super) async fn resume() {
    let _guard = switch_lock().lock().await;
    let Some(mut journal) = current() else {
        return;
    };
    // A switch whose steps are done and whose message has not gone out yet: only
    // the message remains, and it is retried here rather than lost with the journal.
    if journal.finishing() {
        finish_notice(&mut journal).await;
        return;
    }
    if journal.blocked.is_some() {
        // A block is reconsidered only when its cause can have gone: a retry with
        // the pair Telegram just refused cannot succeed, so it is not attempted.
        if !journal.block_may_clear() {
            return;
        }
        journal.blocked = None;
        journal.blocked_credentials = None;
        journal.attempts = 0;
        if let Err(e) = save(&journal).await {
            tracing::warn!(error = %e, "Telegram: the switch journal could not be saved");
        }
    }
    let Some(target) = journal.target_mode() else {
        tracing::warn!("Telegram: the switch journal names no mode; leaving it alone");
        return;
    };
    let result = match target {
        Mode::Local => to_local(&mut journal).await,
        Mode::Cloud => to_cloud(&mut journal).await,
    };
    match result {
        Ok(()) => {
            journal.blocked = None;
            finish_notice(&mut journal).await;
        }
        Err(Step::Blocked(reason)) => block(&mut journal, &reason).await,
        Err(Step::Draining(reason)) => {
            // The bot is served throughout; only the switch waits. Spending the
            // retry budget here would block a switch requested while a large file
            // arrives, in the feature built for large files.
            tracing::info!("Telegram: the switch waits for the listener: {reason}");
        }
        Err(Step::Retry(reason)) => {
            journal.attempts = journal.attempts.saturating_add(1);
            if let Err(e) = save(&journal).await {
                tracing::warn!(error = %e, "Telegram: the switch journal could not be saved");
            }
            set_status(|status| status.last_error = Some(reason.clone()));
            if journal.attempts < SWITCH_ATTEMPT_LIMIT {
                tracing::warn!("Telegram: the switch will be retried: {reason}");
            } else {
                // Retrying forever in silence leaves a mute bot saying "working…"
                // with nothing in the chat: past the limit this is the owner's
                // problem to see, once.
                block(&mut journal, &reason).await;
            }
        }
    }
}

/// Record that the switch waits for the owner, and say so once.
///
/// The cause is persisted rather than only shown, so it survives a restart and is
/// not overwritten by a later probe; a rejected pair is remembered with the values
/// it was about, so it is not retried until they change.
async fn block(journal: &mut Journal, reason: &str) {
    journal.blocked = Some(reason.to_string());
    journal.blocked_credentials =
        super::credentials_rejected(reason).then(current_pair_fingerprint);
    if let Err(e) = save(journal).await {
        tracing::warn!(error = %e, "Telegram: the switch journal could not be saved");
    }
    // The cause is all this adds to the health of the side that still serves the
    // bot — that is that side's own business — and reporting it as down would
    // misdescribe a transport the owner is using right now.
    set_status(|status| status.last_error = Some(reason.to_string()));
    tracing::warn!("Telegram: the switch is waiting for the owner: {reason}");
    // Told once per episode: the flag is persisted before the send and cleared
    // again when the send reports it did not reach the owner, so a crash in
    // between can lose the message but cannot repeat it.
    if journal.notice_sent {
        return;
    }
    journal.notice_sent = true;
    if let Err(e) = save(journal).await {
        tracing::warn!(error = %e, "Telegram: the notice flag could not be saved");
    }
    if super::send_owner_notice(&blocked_notice(reason)).await {
        return;
    }
    journal.notice_sent = false;
    if let Err(e) = save(journal).await {
        tracing::warn!(error = %e, "Telegram: the notice flag could not be cleared");
    }
}

/// What the owner is told when the switch cannot go on without him.
/// The sentence about reachability follows what serves the bot: the side the journal records,
/// and whether a listener polls for it. A switch blocked before the cutover leaves the bot on
/// the cloud, and calling it unreachable would be wrong.
fn blocked_notice(reason: &str) -> String {
    let listening = crate::channel_registry().get("telegram").is_some();
    let reachability = match (super::serving_mode(), listening) {
        (Mode::Cloud, true) => {
            "The bot is still served by Telegram's cloud Bot API and keeps working."
        }
        (Mode::Cloud, false) => {
            "Nothing is polling for the bot right now; the switch keeps trying to bring it back."
        }
        (Mode::Local, true) => {
            "The bot is served by the local Bot API server only: if that server cannot reach \
             Telegram, the bot is not reachable until that is fixed."
        }
        // The same arm covers a server that is not running and one that runs but could not be
        // logged in: either way nothing polls for the bot.
        (Mode::Local, false) => {
            "The bot cannot be reached right now: nothing is polling for it until the cause \
             above is fixed."
        }
    };
    format!(
        "⚠️ Telegram could not finish switching its Bot API: {reason}\n\
         {reachability} It has not gone back to the cloud on its own. Settings → Integrations \
         shows the state: correcting the entry there finishes the switch, and switching back to \
         the cloud clears the local server."
    )
}

/// Whether a failed cloud `logOut` may be read as "the bot has already left".
///
/// Only a resumed attempt can: its earlier request may have gone through before the process
/// died. The first attempt stays strict — there `Unauthorized` tells a bot that never left
/// apart from a token that never worked.
fn cloud_logout_done(resumed: bool, cause: &str) -> bool {
    resumed
        && api_cause(cause)
            .to_ascii_lowercase()
            .contains("unauthorized")
}

/// Put the cloud listener back after a failed step, while the bot is still on the cloud.
///
/// Nothing else rebuilds a listener for that side while the switch is pending, so a step that
/// fails after the listener was taken away would otherwise leave the bot deaf until a retry.
fn restore_cloud_listener() {
    // A listener that is already polling the cloud is what this asks for.
    if crate::channel_registry().get("telegram").is_some() {
        return;
    }
    let token = crate::config::CONFIG.telegram_bot_token();
    if let Err(e) = crate::channels::telegram::restart_telegram_listener(token.as_deref()) {
        tracing::warn!(error = %e, "Telegram: the cloud listener could not be restored");
    }
}

/// Take the bot out of the cloud, or leave the cloud serving it.
///
/// Every step runs before the cutover, so any failure leaves the bot where it was, and the
/// listener is put back before the retry — except once the cloud has accepted the `logOut`.
async fn leave_cloud(journal: &mut Journal) -> Result<(), Step> {
    let result = release_cloud(journal).await;
    // A failed write *after* the `logOut` succeeded is not a failure to leave: the in-memory
    // journal records the leave, and restoring the listener would send a token-bearing request to
    // the cloud for a bot that has just left — the stray call that splits its updates.
    if matches!(result, Err(Step::Retry(_))) && !journal.cloud_logged_out {
        restore_cloud_listener();
    }
    result
}

/// The steps of [`leave_cloud`], with the bot still on the cloud when they fail.
async fn release_cloud(journal: &mut Journal) -> Result<(), Step> {
    // The cloud listener goes first: it is what holds the bot's cloud slot.
    crate::channels::telegram::stop_telegram_listener()
        .await
        .map_err(|e| Step::Draining(e.to_string()))?;
    let Some(token) = crate::config::CONFIG.telegram_bot_token() else {
        return Err(Step::Blocked("Telegram has no bot token configured".into()));
    };
    // The request is recorded before it is issued: a process killed between the answer and the
    // journal write would otherwise re-issue a `logOut` the cloud already carried out.
    let resumed = journal.cloud_logout_started;
    journal.cloud_logout_started = true;
    save(journal)
        .await
        .map_err(|e| Step::Retry(e.to_string()))?;
    match crate::channels::telegram::TelegramChannel::log_out(
        crate::channels::telegram::CLOUD_API_BASE,
        &token,
    )
    .await
    {
        Ok(()) => {}
        Err(e) if cloud_logout_done(resumed, &e.to_string()) => {}
        Err(e) => {
            return Err(Step::Retry(format!(
                "the bot could not leave the cloud: {e}"
            )));
        }
    }
    journal.cloud_logged_out = true;
    save(journal)
        .await
        .map_err(|e| Step::Retry(e.to_string()))?;
    Ok(())
}

async fn to_local(journal: &mut Journal) -> Result<(), Step> {
    if crate::config::CONFIG.telegram_bot_token().is_none() {
        return Err(Step::Blocked(
            "Telegram has no bot token configured, so there is no bot to serve".into(),
        ));
    }
    let (api_id, api_hash) = credentials();
    super::validate_credentials(&api_id, &api_hash).map_err(Step::Blocked)?;
    let fingerprint = super::server::credentials_fingerprint(&api_id, &api_hash);
    // Installed, started and answering while the cloud keeps serving: this touches nothing the
    // bot is using.
    let instance = super::server::ensure(&fingerprint)
        .await
        .map_err(Step::Retry)?;

    if !journal.cloud_logged_out {
        leave_cloud(journal).await?;
    }

    // Published only now: the endpoint is what a listener may talk to, and a local listener must
    // never exist before the bot has left the cloud (a published one would also survive a failed
    // switch).
    super::publish_endpoint(instance.port);

    // The two servers number updates independently, so nothing recorded against one may stand in
    // for the other's events. Cleared after the listener drained, so everything it could hand on
    // already was.
    if journal.intake_must_be_purged() {
        crate::channels::telegram_intake::purge_all()
            .await
            .map_err(|e| Step::Retry(format!("the intake record could not be cleared: {e}")))?;
        journal.intake_cleared = true;
        save(journal)
            .await
            .map_err(|e| Step::Retry(e.to_string()))?;
    }

    set_status(|status| status.health = Health::Starting);

    if let Err(e) = crate::channels::telegram::start_telegram_listener().await {
        let reason = e.to_string();
        if super::credentials_rejected(&reason) {
            // Nothing local is serving the bot until the pair is corrected, so the cause is named
            // now rather than after the server is stopped — the server's own words.
            let cause = api_cause(&reason).to_string();
            set_status(|status| {
                status.health = Health::Down;
                status.last_error = Some(cause.clone());
            });
            // The server caches a failed login for half an hour, so it goes away with the wrong
            // pair, and its endpoint goes with it so no tick builds a listener against a dead port.
            super::server::stop().await;
            return Err(Step::Blocked(cause));
        }
        return Err(Step::Retry(format!(
            "the local server did not accept the bot: {reason}"
        )));
    }

    // The listener is up and the server accepts the bot: the healthy state, which
    // the settings line does not report at all.
    set_status(|status| {
        status.health = Health::Running;
        status.last_error = None;
    });
    // Kept in the journal until it reaches the owner: this is the last step, so a message that
    // failed here would otherwise be lost.
    journal.notice = Some(format!(
        "✅ MahBot is available again — now through its own local Telegram Bot API server \
         ({}). Large files no longer meet Telegram's cloud limits: up to 2000 MB can be sent, \
         and incoming files are no longer refused at 20 MB.",
        instance.version.as_deref().unwrap_or("Bot API server")
    ));
    save(journal)
        .await
        .map_err(|e| Step::Retry(e.to_string()))?;
    Ok(())
}

/// Switch the bot back onto the cloud, removing everything local.
///
/// The local server's `logOut` is best-effort: a server that cannot reach Telegram or holds
/// the wrong token must not keep the owner from the one mode that works.
async fn to_cloud(journal: &mut Journal) -> Result<(), Step> {
    crate::channels::telegram::stop_telegram_listener()
        .await
        .map_err(|e| Step::Draining(e.to_string()))?;

    if !journal.local_left {
        // Only a bot the local server actually served can be left there: a switch that failed
        // before the cutover published no endpoint. The side the journey started from is that fact.
        if journal.origin_mode() == Some(Mode::Local)
            && let (Some(base), Some(token)) = (
                super::local_endpoint(),
                crate::config::CONFIG.telegram_bot_token(),
            )
            && let Err(e) = crate::channels::telegram::TelegramChannel::log_out(&base, &token).await
        {
            tracing::warn!(error = %e, "Telegram: the local server did not release the bot");
        }
        // Recorded before the state goes away: once gone the bot cannot be asked to leave again,
        // and this flag is what lets the transport serve the cloud on a later attempt.
        journal.local_left = true;
        save(journal)
            .await
            .map_err(|e| Step::Retry(e.to_string()))?;
    }

    // Stops the process, removes its state and clears the endpoint it published.
    super::server::remove_state().await;
    if journal.intake_must_be_purged() {
        // Both servers number updates independently: a row the local server left held must not
        // swallow the cloud's next event as a re-offer, and only once the bot has actually left it
        // — a revert that never cut over leaves the cloud record as the only thing deduping
        // re-offers.
        crate::channels::telegram_intake::purge_all()
            .await
            .map_err(|e| Step::Retry(format!("the intake record could not be cleared: {e}")))?;
        journal.intake_cleared = true;
        save(journal)
            .await
            .map_err(|e| Step::Retry(e.to_string()))?;
    }

    if crate::config::CONFIG.telegram_bot_token().is_none() {
        // No bot to start: taking everything local away is the whole switch.
        return Ok(());
    }

    set_status(|status| status.health = Health::Starting);
    // The cloud may refuse the bot for several minutes after it left: the supervisor repeats the
    // attempt, and the journal keeps the switch pending meanwhile.
    crate::channels::telegram::start_telegram_listener()
        .await
        .map_err(|e| {
            Step::Retry(format!(
                "the cloud Bot API has not taken the bot back yet: {e}"
            ))
        })?;

    set_status(|status| {
        status.health = Health::Cloud;
        status.last_error = None;
    });
    journal.notice = Some(
        "✅ MahBot is back on Telegram's cloud Bot API. The local server and its data have been \
         removed, so files are limited again by the cloud (20 MB to the bot, 50 MB from it)."
            .to_string(),
    );
    save(journal)
        .await
        .map_err(|e| Step::Retry(e.to_string()))?;
    Ok(())
}

/// How often the completion message is retried: a journal kept for an undeliverable message
/// would keep the interface reporting a switch forever.
const NOTICE_ATTEMPTS: u32 = 5;

/// Send the completion message a finished switch left pending, then clear the journal.
///
/// Retrying keeps the one message the owner is promised from being lost with the journal. A
/// settled message is recorded before the journal goes, so a failing clear cannot resend it.
async fn finish_notice(journal: &mut Journal) {
    if let Some(text) = journal.notice.clone() {
        if !super::send_owner_notice(&text).await {
            journal.notice_attempts = journal.notice_attempts.saturating_add(1);
            if journal.notice_attempts < NOTICE_ATTEMPTS {
                if let Err(e) = save(journal).await {
                    tracing::warn!(error = %e, "Telegram: the switch journal could not be saved");
                }
                return;
            }
            tracing::warn!("Telegram: the switch notice could not be delivered; dropping it");
        }
        journal.notice = None;
        journal.notice_settled = true;
        if let Err(e) = save(journal).await {
            tracing::warn!(error = %e, "Telegram: the switch journal could not be saved");
        }
    }
    // The mode row is written with the journal but not in the same statement: a failed second
    // write would clear the journal with the product still naming the other side, and the
    // supervisor would stop the server the bot is on. Asserted before the journal goes.
    if let Some(target) = journal.target_mode()
        && super::configured_mode() != target
    {
        match crate::config::persist_settled_string_field(
            crate::config::CONFIG_KEY_TELEGRAM_LOCAL_SERVER,
            target.as_str(),
        )
        .await
        {
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, "Telegram: the switch mode could not be recorded");
                return;
            }
        }
    }
    if let Err(e) = clear().await {
        tracing::warn!(error = %e, "Telegram: the switch journal could not be cleared");
    }
    // The journal is what the interface reads the switch from, so the redraw follows its removal.
    crate::runtime_events::publish(crate::runtime_events::RuntimeEvent::TelegramLocal);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The switch's state machine, driven from the journal alone: forward, blocked, retried, and
    /// back — the owner's only recovery path.
    #[test]
    fn the_switch_walks_forward_blocks_retries_and_comes_back() {
        // Forward: cloud → local. Until the cloud is left, the cloud is what serves.
        let mut journal = requested(None, Mode::Cloud, Mode::Local);
        assert_eq!(journal.origin_mode(), Some(Mode::Cloud));
        assert_eq!(serving(&journal), Mode::Cloud);
        assert!(!journal.intake_must_be_purged());

        // The cloud logOut went through: the bot is on the local server from then on, and only now
        // may the intake record of the cloud's update ids go.
        journal.cloud_logged_out = true;
        assert_eq!(serving(&journal), Mode::Local);
        assert!(journal.intake_must_be_purged());
        journal.intake_cleared = true;
        assert!(!journal.intake_must_be_purged());

        // Telegram refused the pair: the switch waits, a retry with the same values is not
        // attempted, and the journal keeps the local side serving.
        journal.blocked = Some("Unauthorized: invalid api-id/api-hash".into());
        journal.blocked_credentials = Some(current_pair_fingerprint());
        assert!(!journal.block_may_clear());
        assert_eq!(serving(&journal), Mode::Local);

        // A request for the direction already under way neither re-stamps the origin nor loses the
        // facts.
        let retried = requested(Some(journal), Mode::Local, Mode::Local);
        assert_eq!(retried.origin_mode(), Some(Mode::Cloud));
        assert!(retried.cloud_logged_out);
        assert!(retried.blocked.is_none());
        assert_eq!(serving(&retried), Mode::Local);

        // Back: local → cloud. A reversal is a new journey from the side the bot is on now, with
        // its intake record unpurged; the local server keeps serving until the bot has left it.
        let mut back = requested(Some(retried), Mode::Local, Mode::Cloud);
        assert_eq!(back.origin_mode(), Some(Mode::Local));
        assert!(
            !back.cloud_logout_started && !back.cloud_logged_out && !back.local_left,
            "the other direction's steps are not this journey's progress"
        );
        assert!(
            !back.intake_must_be_purged(),
            "the record of the side being left is purged when the bot leaves it, not before"
        );
        assert_eq!(serving(&back), Mode::Local);
        back.local_left = true;
        assert_eq!(serving(&back), Mode::Cloud);
        assert!(
            back.intake_must_be_purged(),
            "the local server's update ids must not stand in for the cloud's"
        );
        back.intake_cleared = true;
        assert!(!back.intake_must_be_purged());

        // A retried revert keeps its own origin and does not flip the serving side.
        let retried_back = requested(Some(back), Mode::Cloud, Mode::Cloud);
        assert_eq!(retried_back.origin_mode(), Some(Mode::Local));
        assert_eq!(serving(&retried_back), Mode::Cloud);

        // A finished switch whose message is still pending is a settled mode: a new request starts
        // its own message.
        let mut finished = retried_back;
        finished.notice = Some("the local server serves the bot now".into());
        let next = requested(Some(finished), Mode::Cloud, Mode::Local);
        assert!(
            next.notice.is_none(),
            "a new journey is not owed the previous journey's message"
        );

        // A delivered message whose journal clear failed is still a settled switch; a new request
        // starts afresh.
        let mut delivered = requested(None, Mode::Local, Mode::Cloud);
        delivered.notice_settled = true;
        assert!(delivered.notice.is_none());
        assert!(
            !requested(Some(delivered), Mode::Local, Mode::Cloud).notice_settled,
            "a new journey has its own message to deliver"
        );

        // A brand-new journey after the switch completed starts where the bot is.
        let forward_again = requested(None, Mode::Cloud, Mode::Local);
        assert_eq!(forward_again.origin_mode(), Some(Mode::Cloud));
        assert_eq!(serving(&forward_again), Mode::Cloud);
    }

    /// A revert of a *settled* local mode — the ordinary way back, whose journal starts empty: the
    /// local release and the purge are both owed.
    #[test]
    fn a_revert_from_a_settled_local_mode_releases_and_purges_the_local_side() {
        let mut journal = requested(None, Mode::Local, Mode::Cloud);
        assert_eq!(journal.origin_mode(), Some(Mode::Local));
        assert!(!journal.cloud_logged_out);
        assert_eq!(serving(&journal), Mode::Local);
        // Nothing is purged while the bot is still on the local server.
        assert!(!journal.intake_must_be_purged());

        journal.local_left = true;
        assert_eq!(serving(&journal), Mode::Cloud);
        assert!(journal.intake_must_be_purged());
    }

    /// The serving side for a journal, with the configured mode set to its target.
    fn serving(journal: &Journal) -> Mode {
        super::super::serving_from(
            journal.target_mode().expect("a journal names a target"),
            Some(journal),
        )
    }

    #[test]
    fn only_a_resumed_attempt_reads_unauthorized_as_already_out() {
        // The first attempt must not mistake a token Telegram never accepted for a bot that has
        // left.
        assert!(!cloud_logout_done(
            false,
            "Telegram rejected the bot token — Unauthorized"
        ));
        // A resumed attempt follows a request whose answer was lost: `Unauthorized` is the state
        // the step asks for.
        assert!(cloud_logout_done(
            true,
            "Telegram rejected the bot token — Unauthorized"
        ));
        // Anything else is still a failure, resumed or not.
        assert!(!cloud_logout_done(true, "the Bot API could not be reached"));
    }
}
