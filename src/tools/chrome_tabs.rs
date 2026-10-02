//! Closing the browser tabs one chrome session created — straight from the browser's own
//! extension, never from chrome-use's endpoint-bound record.
//!
//! The names this module is handed are the physical session names the product mints
//! (`agent-tab-<12 hex>-<tab>`), but it assumes nothing about their shape: whatever names a
//! caller recorded are the names it looks for, matched exactly against the live tab-group
//! titles in the browser.
//!
//! ## Why the extension ledger, not chrome-use's `.created-targets.json`
//!
//! chrome-use keeps a per-session record of the tabs it created in
//! `~/.chrome-use/<session>.created-targets.json`, bound to the relay endpoint it was written
//! under: re-mint that endpoint — a Chrome restart, a re-pairing, an extension re-install —
//! and the record points at a browser that no longer exists, so a `session stop` routed
//! through it closes nothing. The extension's OWN ledger (`ownedTabs`) is independent of any
//! endpoint, so it is the durable record of which ids the extension will remove, and the only
//! one.
//!
//! ## What makes the removal safe
//!
//! `tabs.remove` is the extension's allow-listed door, and the extension refuses the WHOLE
//! call unless every id it is handed is one it holds (agent-created or adopted tabs only).
//! Two things back a close: the ledger decides which ids may be asked for at all, and the
//! exact match on the recorded title decides what is asked for ([`group_ids_titled`]). The
//! call's own exit status is never proof: closure is verified from the browser's own answers,
//! read exactly as [`plan_removal`] read them before the call.
//!
//! ## The session these reads run in
//!
//! An `extension call` goes through a chrome-use session, and a call that names none is given
//! one derived from the working directory — a name no product rule owns, whose daemon opens an
//! `about:blank` group titled that name in the owner's real browser. Every call here therefore
//! names a session, and the caller's choice ([`close_sessions`]'s `session`) is what keeps the
//! product from opening anything of its own: it is a name the work itself already has — the run
//! whose tabs are being closed — and a session that still exists brings no new page with it. A
//! name whose daemon chrome-use had already recycled is started by the call like any other
//! session, which opens the page it then works in inside that session's own group; the caller
//! borrows only names a record holds, so that record's own close removes the group again — with
//! the one exception the Honest limits below name (a namespace still protected when that close
//! runs).
//!
//! [`OWN_SESSION`] is what is left for a pass with no name to borrow, and a pass that used it
//! lets it go ([`release_own_session`]) through the same connection-independent route a run's
//! tabs are closed by, so a re-minted relay endpoint cannot strand it; a let-go that route
//! cannot confirm is queued for the caller's let-go step, which asks the session's own graceful
//! `session stop` — the route that closes the tabs a session created even when the extension's
//! ledger no longer reaches them. What is left running is in the family the product's session
//! sweeps act on.
//!
//! ## Honest limits
//!
//! - A browser side with no ownership door at all — no extension installed in the driving
//!   profile, one installed but disabled, or an ab-connect older than the one that has it —
//!   has no `extension state` / `extension call` route; the only one left is chrome-use's own
//!   `session stop` by name, which this module falls back to when it can establish the gap.
//!   That route's only verdict is the helper's own report that the session is stopped — on a
//!   host with no ledger there is no second opinion to take (see [`stop_by_name`]) — while every
//!   failure it reports stays a retry. What is lost there is the sweep's side of the work:
//!   nothing can enumerate the browser's groups, so a leftover no record names is out of reach;
//!   that host-level gap, not any one run, is what the caller reports once
//!   ([`ownership_door_missing`]). A relay that is merely unreachable (Chrome closed, extension
//!   restarting) is NOT that gap: the door is there, and those names stay retries.
//! - A group renamed in Chrome is invisible to this route — the match is exact on the
//!   recorded name. A rename that KEEPS `crate::tools::chrome::AGENT_TAB_PREFIX` is still
//!   reclaimed by the release module's sweep (it enumerates live titles, not recorded names);
//!   a name that leaves the prefix is left alone rather than guessed at, because nothing ties
//!   those tabs to a session (or to this product) any more.
//! - Ownership is per tab and lives only in the installed extension: one updated or
//!   re-installed since the tabs were created no longer holds them, so those tabs are
//!   reported as unclosable rather than guessed at.
//! - The ledger is a list of ids and nothing else, so this route cannot tell a tab the session
//!   CREATED from one the relay merely ADOPTED — chrome-use marks a re-owned leftover adopted,
//!   and the extension holds "agent-created or adopted tabs only", while `parse_owned_tabs`
//!   gets no ownership kind to filter on. What keeps the close off the owner's own tabs is
//!   therefore its scope, not a kind: the exact title of a group this product minted, and only
//!   ids the extension already holds. A tab the agent took over by hand through chrome-use's own
//!   `tab adopt` (which nothing in this product uses) would be closed with its session's, and a
//!   tab the owner drags into an agent group the relay later adopts with them; both are accepted
//!   residuals, named here rather than filtered on a kind that does not travel.
//! - A close of an ended run's tabs that the browser answered and left standing is a `Retry`
//!   with a [`RetryLeg::LeftOpen`] leg, never a verdict from one pass (see that leg and
//!   [`remove_answered`]); every other failure proves nothing and stays [`RetryLeg::Silent`].
//!   The caller counts these answers per ended run and concludes that run's names as unclosable
//!   once the browser has answered that way its policy's number of times — reported with the run
//!   and dropped from its record, and still not abandoned: the reclaim sweep examines such a
//!   group again at its own revisit cadence, so a browser that recovered (an extension restarted
//!   with a fresh ledger, a `tabs.remove` that now goes through) still gets it closed.
//! - A mis-shaped element in any of the three envelope reads (`tabGroups.query` group without
//!   an integer `id` or string `title`, `tabs.query` element without an integer `groupId`, a
//!   non-integer `ownedTabs` entry) makes that whole answer unreadable ([`parse_live_groups`] /
//!   [`parse_tabs_by_group`] / [`parse_owned_tabs`]) — one fact about the host, not a per-name
//!   silence, so the caller files one de-duplicated host-level notice for it
//!   ([`saw_unreadable_answer`], latched by the reads this route decides from). On a host that has
//!   the ownership door, every name that read would have settled stays a retry with no conclusion
//!   — the safe direction (a silently dropped element would read as a group that is gone); a
//!   door-less host instead takes [`fallback_outcomes`]' `session stop` for a failed plan read,
//!   which can still settle a name that way. An unreadable answer to the removal or to a
//!   `session stop` is deliberately not latched: neither settles a name by itself, so a name the
//!   pass cannot settle from them stays a retry with a `debug!` line, since a later pass asks
//!   again.
//! - A borrowed read session that is WEDGED — its daemon registered but not answering — leaves
//!   every read of that pass a failure that settles nothing about the tabs, so
//!   the pass settles nothing and stops nothing, its names stay retries, and the next pass
//!   borrows the same name again. What clears it is chrome-use's own replacement of an
//!   unreachable worker on the next browser command through that session ("the next browser
//!   command stops the unreachable worker and starts a clean replacement for the same session",
//!   its own documented behaviour), not this product: this route never stops a session a run may
//!   own, and a stop here would be a second route to the same end.
//! - The one page any read here can add is chrome-use's own: a call that starts a session whose
//!   daemon had been recycled opens the page it then works in. In a borrowed session it lands in
//!   that session's group, which the record holding the name closes — EXCEPT while that
//!   namespace is protected (its run holds it again, or a resumable job claims it), when the
//!   page waits inside that run's group for a later pass, and except when the extension's ledger
//!   no longer holds it, when that close reports it as an unclosable leftover instead of
//!   dropping it. A pass that stops a name only to read again ([`confirm_unclosable`]) mints one
//!   such page in the session it restarted: that name settles gone and the pass's let-go step
//!   stops the session right after, or it is concluded unclosable — the record is then dropped
//!   with the leftover reported and the page it left is a live `agent-tab-*` group no run or
//!   record claims, which the reclaim sweep closes at its next ask while the ledger holds it. A
//!   pass with no name to borrow at all — the sweep armed for a leftover no record names, while
//!   no queued or parked record carries one either — reads in [`OWN_SESSION`] and shows its
//!   scratch group, let go by the same pass: rare, driven by a real leftover rather than a
//!   timer.

use super::chrome_daemon::{ExtensionState, NoVerdict, extension_state_from, run_cli_json_at};
use crate::tools::chrome::AGENT_TAB_PREFIX;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::debug;

/// Bound on one browser-read call.
const TAB_READ_TIMEOUT: Duration = Duration::from_secs(15);

/// The share of a pass one phase of three browser reads is owed, each allowed
/// [`TAB_READ_TIMEOUT`]. Two phases of a pass have this shape: the reads that plan a close
/// (`tabGroups.query` in [`read_live_groups`], the extension's ledger and `tabs.query` in
/// [`read_ownership`]) and [`confirm_unclosable`]'s own deciding reads (`extension state`,
/// `tabs.query` and the live-group re-read). [`confirm_unclosable`] gives its stops only what
/// the pass has left beyond its own share, so one stop cannot starve reads that decide for
/// every name at once — and on a pass that has already spent the planning share plus the
/// removal and verification reads, no whole stop is left, so none is spawned: the names that
/// stop would have covered stay retries, concluded by no read of that pass, and the reads
/// settle only the names whose stops the pass did hold. The fallback a failed plan read takes
/// ([`fallback_outcomes`]: one `extension status` read plus one `session stop` per name) is
/// bounded by the pass deadline, not by this reserve; a read that overruns the pass anyway
/// leaves every name a retry, never a verdict.
pub(crate) const READ_PHASE_RESERVE: Duration = TAB_READ_TIMEOUT.saturating_mul(3);

/// The oldest ab-connect whose `extension state` / `extension call` door this
/// module needs (upstream's own capability gate).
const MIN_EXTENSION_VERSION: semver::Version = semver::Version::new(0, 5, 25);

/// Why one name is not settled yet. Only the caller knows which of the two it can
/// act on: a silent leg establishes nothing, a left-open leg is the browser's own
/// answer about this run's close.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetryLeg {
    /// The browser or the helper never answered. Nothing was established, so the
    /// name is simply tried again.
    Silent,
    /// The browser answered this run's close and the tabs were left standing: the removal
    /// refused as not the extension's own to make ([`remove_answered`]), or answered success
    /// while a ledger-owned tab still stood in the group ([`verify`]'s reading). Everything
    /// else a failed call can carry says nothing about the tabs and stays [`Self::Silent`]. The
    /// caller retries, and says so durably — an answered-and-left-open name nobody can act on
    /// must not stay a line that ages out.
    LeftOpen,
}

/// Outcome for one session name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TabOutcome {
    /// No tab of that session is left in the browser (the browser's own answer).
    Gone,
    /// Established: no route this cleanup has closes the tabs this session left — the record
    /// stops retrying it, while the reclaim sweep keeps examining such a group.
    Unclosable,
    /// The browser side did not settle it; the caller tries again later.
    Retry(RetryLeg),
}

/// What the browser side reports about the ownership door, out of `extension status`:
/// the version of the extension the relay is answering for right now, and — when nothing
/// is connected — the version installed in the browser's own driving profile.
///
/// The envelope's `installed` flag is NEITHER of those: it is the native-messaging host
/// manifest, which this product writes on every start, so it says nothing about whether
/// the extension is there (a host with no Chrome profile answers `installed: true`). The
/// fields below are the ones that answer the question — the same field
/// `crate::tools::chrome_daemon`'s own extension-state reading keys on.
#[derive(Default, Clone)]
struct ExtensionStatus {
    /// `liveExtensionVersion`: the extension the relay answered for, when it answered.
    live: Option<semver::Version>,
    /// `chromeExtension.version`: the one installed in the driving profile, when one is.
    installed: Option<semver::Version>,
    /// Whether the envelope described the installed extension as ABSENT at all — an
    /// explicit `chromeExtension: null`, which is the CLI's own absence signal. A missing
    /// key is shape-uncertain (an old CLI) and an object without a version string
    /// establishes nothing either, so neither means "no extension": both fail open through
    /// [`Self::door_missing`].
    absent: bool,
    /// Whether the extension is installed but DISABLED (a non-empty `disableReasons`):
    /// there is nothing to answer an `extension call`, whatever version it reports, so
    /// this decides [`Self::door_missing`] on its own.
    disabled: bool,
}

impl ExtensionStatus {
    /// Whether this host provably has no ownership door: an extension installed but
    /// disabled, one older than the gate — the live one if the relay has one, otherwise
    /// the installed one — or an installed extension the envelope described as absent. A
    /// relay that is merely unreachable (Chrome closed, extension restarting) still
    /// reports a usable extension, so it is NOT that gap: the door is there, and those
    /// names stay retries.
    fn door_missing(&self) -> bool {
        if self.disabled {
            return true;
        }
        let too_old = |version: &semver::Version| *version < MIN_EXTENSION_VERSION;
        match (&self.live, &self.installed) {
            (Some(live), _) => too_old(live),
            (None, Some(installed)) => too_old(installed),
            (None, None) => self.absent,
        }
    }
}

/// Whether `name`'s session belongs to a run that owns it again RIGHT NOW. Asked
/// immediately before each act a close performs — the removal, and each `session stop` —
/// because every read that came before takes seconds: a run that re-registered its
/// namespace in that window owns its sessions, and none of them may be touched.
pub(crate) type OwnedAgain<'a> = &'a (dyn Fn(&str) -> bool + Send + Sync);

/// The chrome-use session this module's own browser reads run in — see the module doc.
/// It must stay a name [`crate::tools::chrome_daemon::is_mahbot_session_name`] sweeps:
/// that is what closes the group when a pass dies before [`release_own_session`] reaches
/// it.
pub(crate) const OWN_SESSION: &str = "mahbot-chrome-ephemeral-tabs";

/// Every call this module has made through [`OWN_SESSION`], as a running count, and the
/// count a let-go has already covered. A pass owes the teardown exactly when a call
/// happened since the last one — which also serves a pass overlapping it (the driver and
/// the shutdown flush): there is one session name, so whatever scratch group exists is the
/// one the last teardown closed, and whichever pass tears down last leaves nothing behind.
/// Both let-gos advance it, because either one ends the group those calls opened:
/// [`release_own_session`] once the browser confirmed its ledger route closed the group, and
/// [`forget_settled_sessions`] once a graceful `session stop` settled the session. The second
/// matters as much as the first — with the marker left behind there, the next pass starts that
/// session again and mints a fresh blank group for no new reason, which is the one page this
/// whole choice of session exists to keep out of the owner's browser.
static DOOR_CALLS: AtomicU64 = AtomicU64::new(0);
static DOOR_RELEASED: AtomicU64 = AtomicU64::new(0);
/// Calls in flight right now. A let-go must never stop the session under one that is still
/// running, so it defers to the next pass instead — [`release_own_session`] before it starts
/// one, and [`forget_settled_sessions`] before its stop.
static DOOR_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
/// The calls the browser side answered with a successful envelope, as a running count: a
/// scratch group exists only if one of them did. The session's daemon opens that group when it
/// starts, and every call here runs with chrome-use's own relay self-heal suppressed, so a call
/// that never got an answer never opened one either ([`own_scratch_group_possible`]).
static DOOR_ANSWERED: AtomicU64 = AtomicU64::new(0);
/// Whether a browser answer this process received carried no decision this route can take
/// ([`NoVerdict::Unreadable`]): a field parser that met a mis-shaped element, bytes with no
/// envelope at all, or an envelope with neither a success verdict nor a reason for failing. Such
/// an answer settles no name, and the fact is about the host rather than about one name, so the
/// caller files the one host-level notice for it ([`saw_unreadable_answer`]). Latched for the
/// process: the fact is about the browser side, not about one pass.
static UNREADABLE_ANSWER: AtomicBool = AtomicBool::new(false);

/// Holds one [`DOOR_IN_FLIGHT`] count for as long as one call runs: dropped however that call
/// ends — a completion, a panic, or the whole pass future being dropped mid-call (shutdown,
/// a self-update landing while a pass of up to a pass budget is in flight) — so the count can
/// never be left standing and block every later let-go.
struct DoorCallInFlight;

impl Drop for DoorCallInFlight {
    fn drop(&mut self) {
        DOOR_IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// One browser call through `session` — the session the work itself already has wherever one
/// exists (a run whose tabs are being closed), so a pass with a name to borrow does not mint a
/// session of ours. The counts above are [`OWN_SESSION`]'s alone: they are what its let-go owes,
/// and a borrowed session owes none (it is a run's own, stopped by the let-go step like any
/// settled session). A session that is gone is started by this call like any other — chrome-use
/// opens the page its daemon then works in — which is why the caller borrows a name a record
/// holds: that record's own close removes the page again, unless that namespace is still
/// protected when the close runs — the one case the module's Honest limits name (which name a
/// pass borrows is `chrome_release`'s own `read_session`). A `session stop` acts on a session
/// that already exists and calls [`run_cli_json_at`] directly.
async fn door_call(
    cli: &Path,
    args: &[&str],
    session: &str,
    timeout: Duration,
) -> Result<Value, NoVerdict> {
    let ours = session == OWN_SESSION;
    // The guard must be built only for a call of ours: one built to be discarded would still run
    // its `Drop` and underflow the count, so this cannot be a `then_some`.
    let _in_flight = if ours {
        DOOR_CALLS.fetch_add(1, Ordering::SeqCst);
        DOOR_IN_FLIGHT.fetch_add(1, Ordering::SeqCst);
        Some(DoorCallInFlight)
    } else {
        None
    };
    let result = run_cli_json_at(cli, args, Some(session), timeout).await;
    if ours && result.is_ok() {
        DOOR_ANSWERED.fetch_add(1, Ordering::SeqCst);
    }
    result
}

/// Whether a scratch group could exist at all: true once the browser side answered one of this
/// module's calls (see [`DOOR_ANSWERED`]). On a host where it never did — the browser is closed,
/// the relay is gone — an unconfirmed let-go says nothing about the owner's tab strip, so the
/// caller surfaces nothing for it.
pub(crate) fn own_scratch_group_possible() -> bool {
    DOOR_ANSWERED.load(Ordering::SeqCst) > 0
}

/// Whether a browser answer ever carried no decision this route can take (see
/// [`UNREADABLE_ANSWER`]). Such an answer settles no name, so the caller reports that as a fact
/// about the host rather than letting it hide behind a `debug!`.
pub(crate) fn saw_unreadable_answer() -> bool {
    UNREADABLE_ANSWER.load(Ordering::SeqCst)
}

/// Test-only: forget an unreadable answer one test made, so it cannot make another test's pass
/// report the host row.
#[cfg(test)]
pub(crate) fn reset_unreadable_answer() {
    UNREADABLE_ANSWER.store(false, Ordering::SeqCst);
}

/// Test-only: hold one [`DOOR_IN_FLIGHT`] count for as long as the returned guard lives, so a
/// test can reach the deferral a call of ours in flight causes without running a real call.
/// The calls-and-releases marker is deliberately untouched: this is not a call, and moving it
/// would make a later pass owe a teardown nothing asked for.
#[cfg(test)]
pub(crate) fn hold_door_call_in_flight() -> impl Drop {
    DOOR_IN_FLIGHT.fetch_add(1, Ordering::SeqCst);
    DoorCallInFlight
}

/// Let go of the session this module's own browser reads ran in: the tab group its first
/// command opened in the owner's real browser — an `about:blank` group titled [`OWN_SESSION`],
/// visible in his tab strip for the seconds a pass reads — is closed through the extension's own
/// door; when that door confirmed the group gone, the session's chrome-use record is dropped
/// with the `--force` stop. When it did not, the caller queues the name and the graceful stop —
/// the one route that closes the tabs a session created — is issued by the caller's let-go step
/// ([`crate::tools::chrome_release`]'s, the single place that stop is asked for) or by the
/// queue's own retry on its armed deadline, so a visible group never waits for the next start's
/// sweep. See the module doc for why this is our one scratch group.
///
/// `true` when nothing of ours is known to be left: nothing was owed, no call has been made
/// since the last let-go, a call was in flight and this pass deferred to the next, or the
/// browser confirmed the scratch group gone — where a `--force` stop that did not answer leaves
/// only chrome-use's own per-session record behind, which is bookkeeping and never a tab, and
/// nothing here retries it. `false` means exactly one thing: this pass made the reads and the
/// browser did not confirm the scratch group gone — including when a read itself failed, so a
/// group may or may not be there — which is the one thing a caller surfaces, queues and retries.
/// A `false` also leaves this pass's reads outstanding, so whichever pass runs next tries the
/// let-go again.
///
/// Nothing happens when nothing is owed or a call is in flight; that check is the
/// check-then-act every read here is (passes CAN overlap — the driver and the shutdown
/// flush). One outcome is the scratch group left for a later pass or the boot sweep: the pass
/// that saw the in-flight call defers. The other is a pass that reads "nothing in flight"
/// just before an overlapping pass starts a read and stops the shared session under that
/// call, leaving it unanswered — those names stay retries (the safe direction) while this
/// pass still closes the scratch group. Best-effort throughout: never a run's leftover.
pub(crate) async fn release_own_session(cli: &Path, deadline: Instant) -> bool {
    if DOOR_IN_FLIGHT.load(Ordering::SeqCst) != 0
        || DOOR_CALLS.load(Ordering::SeqCst) == DOOR_RELEASED.load(Ordering::SeqCst)
    {
        return true;
    }
    let gone = close_own_scratch_group(cli, deadline).await;
    if gone {
        // The marker is taken BEFORE the stop below: what it covers is the reads that saw the
        // group gone, and the stop takes the session's daemon down — a call an overlapping pass
        // starts in between would start a fresh daemon and mint a new scratch group there, so
        // covering it would make that pass skip the very group it left standing. `fetch_max`, so
        // a let-go another pass ran meanwhile is not pulled back to this one's count. The window
        // between those reads and this load is accepted and not chased: a call started in it is
        // covered, and that pass's own let-go skips — at most one blank group waits for that
        // pass's next call or the boot session sweep, and no run's tab is involved either way.
        DOOR_RELEASED.fetch_max(DOOR_CALLS.load(Ordering::SeqCst), Ordering::SeqCst);
        // The ledger route took the group's tabs, so only chrome-use's record is left, and
        // `--force` is the form that clears it without asking the session's browser again
        // (live-verified: a graceful stop on a session whose tab is already gone fails with
        // "ownership was preserved" and keeps the record, while `--force` drops it). An
        // answered or unanswered one, the group is gone: a record this leaves behind is the
        // helper's own bookkeeping, and nothing here retries it.
        if let Some(timeout) = spawn_budget(deadline, crate::chrome::SESSION_STOP_TIMEOUT)
            && let Err(verdict) = run_cli_json_at(
                cli,
                &["session", "stop", "--force"],
                Some(OWN_SESSION),
                timeout,
            )
            .await
        {
            debug!(
                reason = %verdict.text(),
                "the browser-read session's record could not be dropped"
            );
        }
    }
    // An unconfirmed group is NOT stopped here: the graceful stop is the caller's let-go step,
    // which runs in this same pass and asks it once (see the doc above), while stopping it here
    // as well would put two `session stop` children on one stuck session per pass.
    gone
}

/// Close the scratch group [`OWN_SESSION`]'s daemon opened, through the extension's own
/// door, and say whether the browser confirms it is gone — `false` for every leg that
/// answered nothing, and for a group holding a tab the extension does not hold as its own,
/// which this route may not remove.
///
/// This is the single-name core of the same pipeline [`close_sessions`] owns through
/// [`plan_removal`]/[`remove_and_verify`]: the same ledger decides which ids may be removed
/// at all, and the removal call is never its own proof — the browser's own re-read
/// ([`groups_gone`]) is. Two things are deliberately different, and must stay so: there is
/// no door-less fallback here (a host with no ownership door simply cannot close this
/// group), and a group holding a tab the extension does not own is not confirmed here — the
/// caller queues the session's own graceful `session stop`, chrome-use's documented cleanup
/// path for the tabs a session created, which is not limited to the ids the extension's
/// ledger still holds (see [`crate::tools::chrome_release::let_go_own_session`]).
async fn close_own_scratch_group(cli: &Path, deadline: Instant) -> bool {
    // The one pass that reads in [`OWN_SESSION`] by construction: the group IS that session's,
    // so there is nothing else to borrow.
    let door = Door {
        cli,
        session: OWN_SESSION,
        deadline,
    };
    let Ok(groups) = read_live_groups(door).await else {
        return false;
    };
    let ids = group_ids_titled(&groups, OWN_SESSION);
    if ids.is_empty() {
        return true; // never opened, or already gone
    }
    let Ok((owned, tabs)) = read_ownership(door).await else {
        return false;
    };
    let targets = owned_group_tabs(&ids, &tabs, &owned);
    if targets.is_empty() {
        return false;
    }
    let _ = remove_tabs(door, &targets).await;
    // Success of the call is not proof: the browser's own answer is.
    groups_gone(door, &ids).await
}

/// Close the tabs of `names`, one outcome per name, in the order given.
/// `deadline` bounds the whole sequence: a call whose remaining budget is spent
/// is not spawned (the affected names come back as `Retry`). `session` is the chrome-use session
/// every browser read runs in — the caller passes a session the work itself already has wherever
/// one exists (see [`crate::tools::chrome_release`]), because a name that still exists opens no
/// new page; [`OWN_SESSION`] is what is left for a caller with nothing to borrow.
pub(crate) async fn close_sessions(
    cli: &Path,
    names: &[String],
    session: &str,
    deadline: Instant,
    owned_again: OwnedAgain<'_>,
) -> Vec<(String, TabOutcome)> {
    if names.is_empty() {
        return Vec::new();
    }
    let door = Door {
        cli,
        session,
        deadline,
    };
    let mut outcomes: Vec<Option<TabOutcome>> = vec![None; names.len()];

    // The LIVE enumeration, and the only one: `extension state`'s `groups` map is the
    // extension's own in-memory name→id map — empty after a service-worker restart, and
    // never the browser's live group list — so it must never be enumerated.
    let groups = match read_live_groups(door).await {
        Ok(groups) => groups,
        Err(reason) => {
            let all_names: Vec<usize> = (0..names.len()).collect();
            fallback_outcomes(door, names, &all_names, &reason, owned_again, &mut outcomes).await;
            return zip_outcomes(names, outcomes);
        }
    };

    let matched: Vec<Vec<i64>> = names
        .iter()
        .map(|name| group_ids_titled(&groups, name))
        .collect();
    for (i, group_ids) in matched.iter().enumerate() {
        if group_ids.is_empty() {
            // A name with no live group left nothing to close (a group with no tabs no
            // longer exists), so the common case — every recorded name already gone —
            // settles here, on the one read, and never reaches a remove call.
            outcomes[i] = Some(TabOutcome::Gone);
        }
    }
    if matched.iter().all(Vec::is_empty) {
        return zip_outcomes(names, outcomes);
    }

    let (owned, tabs) = match read_ownership(door).await {
        Ok(ownership) => ownership,
        Err(reason) => {
            let affected: Vec<usize> = matched
                .iter()
                .enumerate()
                .filter_map(|(i, ids)| (!ids.is_empty()).then_some(i))
                .collect();
            fallback_outcomes(door, names, &affected, &reason, owned_again, &mut outcomes).await;
            return zip_outcomes(names, outcomes);
        }
    };

    // Asked here, at the moment of the acts: the reads above took seconds, and a run that
    // re-registered its namespace inside that window owns its sessions again.
    let taken_back: Vec<bool> = names.iter().map(|name| owned_again(name)).collect();
    let plan = plan_removal(&matched, &tabs, &owned, &taken_back);
    for (i, settled) in plan.settled.iter().enumerate().filter(|(_, s)| s.is_some()) {
        outcomes[i].clone_from(settled);
    }

    if !plan.verifying.is_empty() {
        remove_and_verify(door, &matched, &plan, &owned, &mut outcomes).await;
    }

    // No name is called unclosable on one read of state the extension keeps elsewhere:
    // each one gets the session's own route and a second read of the browser first.
    let unclosable: Vec<usize> = outcomes
        .iter()
        .enumerate()
        .filter(|(_, outcome)| matches!(outcome, Some(TabOutcome::Unclosable)))
        .map(|(i, _)| i)
        .collect();
    if !unclosable.is_empty() {
        confirm_unclosable(
            door,
            names,
            &matched,
            &unclosable,
            owned_again,
            &mut outcomes,
        )
        .await;
    }
    zip_outcomes(names, outcomes)
}

/// Titles of every live tab group the browser holds whose name starts with
/// `crate::tools::chrome::AGENT_TAB_PREFIX` ("agent-tab-"). `Err` carries the
/// reason the browser could not be asked.
pub(crate) async fn agent_group_names(
    cli: &Path,
    session: &str,
    deadline: Instant,
) -> Result<Vec<String>, NoVerdict> {
    let door = Door {
        cli,
        session,
        deadline,
    };
    let groups = read_live_groups(door).await?;
    Ok(groups
        .into_iter()
        .map(|group| group.title)
        .filter(|title| title.starts_with(AGENT_TAB_PREFIX))
        .collect())
}

/// Let go of a session chrome-use's own record still claims: one `chrome-use` child each takes
/// the session's daemon down, and the verb follows what the name is still waiting for.
///
/// A settled run's session is bookkeeping by the time it reaches this queue — its tabs went
/// through the extension's ledger — and `session stop --force` is what clears that record: it
/// is `session stop` with the reconnect-and-close step skipped, so the record is dropped
/// instead of kept for a browser that is gone (its answer counts whatever it could not reach in
/// `forgottenTabs`). A GRACEFUL stop on such a record fails with "ownership was preserved" and
/// keeps it (live-verified), which would leave this queue retrying a record forever.
///
/// This module's own scratch session ([`OWN_SESSION`]) is here for the opposite reason: it is
/// queued when [`release_own_session`] could not confirm its group gone, so the group may still
/// be standing in the owner's tab strip, and only the graceful stop closes the tabs a session
/// created — the reconnect-and-close route, which reaches them even when the extension's ledger
/// no longer holds them (live-verified: the group a read minted is gone after it, its record
/// with it). `--force` here would drop the record and leave that group where it is.
///
/// These are one sequential `chrome-use` child each, so `deadline` can cut the sequence
/// short. Each stop is attempted with whatever the pass has left ([`spawn_budget`]) — the
/// whole [`crate::chrome::SESSION_STOP_TIMEOUT`] when it holds that much, a best-effort
/// shorter one otherwise — and what that attempt came to decides the class: a stop the pass
/// could give its whole bound and the helper did not settle is a failure; one it cut short
/// asked the helper but could not hear a real answer, so it is not an answer to charge. What
/// the sequence did with each name — the stop ran and failed, was cut short or never asked,
/// belongs to a run that owns it again, or was not asked at all because a call of ours held
/// the session — comes back in [`LetGoOutcome`] for the caller to act on.
///
/// [`OWN_SESSION`] is asked for around the two rules its own reads give it: it is never
/// stopped while a call of ours is in flight ([`DOOR_IN_FLIGHT`] — the same deferral
/// [`release_own_session`] makes, reported as [`LetGoOutcome::busy`] because nothing is owed
/// until that call is done and the pass holding it has run its own let-go step), and a stop
/// that lands is what lets the reads those calls covered stop being owed ([`DOOR_RELEASED`]),
/// so the next pass does not start the session again for reads this stop already ended.
pub(crate) async fn forget_settled_sessions(
    cli: &Path,
    names: &[String],
    deadline: Instant,
    owned_again: OwnedAgain<'_>,
) -> LetGoOutcome {
    let mut outcome = LetGoOutcome::default();
    for (index, name) in names.iter().enumerate() {
        if owned_again(name) {
            outcome.owned.push(name.clone());
            continue;
        }
        // The verb each name is still waiting for: a settled run's session has only its record
        // left, the scratch session may still have its group (see the doc above).
        let ours = name == OWN_SESSION;
        let stop: &[&str] = if ours {
            &["session", "stop"]
        } else {
            &["session", "stop", "--force"]
        };
        if ours && DOOR_IN_FLIGHT.load(Ordering::SeqCst) != 0 {
            // A call of ours is running in this very session: a stop now would end it under that
            // call. The name is deferred, not failed — nothing was asked — and told apart from a
            // deferral because nothing is owed until that call, and the let-go step behind it,
            // are done.
            outcome.busy.push(name.clone());
            continue;
        }
        // The calls this stop is about to cover, read before it runs: a call an overlapping pass
        // starts meanwhile is that pass's to let go, not this one's.
        let calls_before_stop = ours.then(|| DOOR_CALLS.load(Ordering::SeqCst));
        let Some(timeout) = spawn_budget(deadline, crate::chrome::SESSION_STOP_TIMEOUT) else {
            outcome.deferred.extend(names[index..].iter().cloned());
            return outcome;
        };
        // Whether this stop got its whole bound — the same rule [`stop_budget`] applies, read
        // off the budget above, because the stop is worth attempting either way (a helper whose
        // daemon is up answers well inside the bound — live-verified at seconds) while only a
        // whole one's failure may be charged.
        let whole = timeout == crate::chrome::SESSION_STOP_TIMEOUT;
        match run_cli_json_at(cli, stop, Some(name), timeout).await {
            // The helper settled the session: its record is gone, and for the scratch session
            // that stop is also what removed its group, so the reads these calls covered are no
            // longer owed ([`DOOR_RELEASED`]). `fetch_max`, so an overlapping pass that has
            // already covered more is not pulled back.
            Ok(_) => {
                if let Some(calls) = calls_before_stop {
                    DOOR_RELEASED.fetch_max(calls, Ordering::SeqCst);
                }
            }
            Err(verdict) => {
                if whole {
                    // The stop had its whole bound and the helper still did not settle the
                    // session, so its record is still there: the name comes back for the caller
                    // to queue.
                    debug!(
                        name = %name,
                        reason = %verdict.text(),
                        "a settled chrome session could not be let go"
                    );
                    outcome.tried.push(name.clone());
                } else {
                    // A cut short stop is best-effort: the name is kept, but not charged for an
                    // answer the pass never had the time to hear (see [`LetGoOutcome::deferred`]).
                    debug!(
                        name = %name,
                        reason = %verdict.text(),
                        "a settled chrome session's stop was cut to the pass's remaining budget"
                    );
                    outcome.deferred.push(name.clone());
                }
            }
        }
    }
    outcome
}

/// What one [`forget_settled_sessions`] sequence came to, one class per name it did not let
/// go of — the four a caller acts on differently.
#[derive(Default)]
pub(crate) struct LetGoOutcome {
    /// Names a `session stop` RAN for — with its whole bound — and did not settle. Only these
    /// are charged a failed attempt: the stop is the helper not answering, not a budget the
    /// pass never spent.
    pub(crate) tried: Vec<String>,
    /// Names whose session a run owns again right now: stopping one is what would end that
    /// run's own session, so they are not ours to retry — the run's own end hands over its own
    /// release.
    pub(crate) owned: Vec<String>,
    /// Names the pass could not give a stop its whole bound: either its budget was spent and no
    /// stop was spawned at all, or the one it spawned was cut short and its answer is therefore no
    /// answer. Nothing either way may be charged — a name the pass never heard a real verdict for
    /// is deferred, not failed.
    pub(crate) deferred: Vec<String>,
    /// [`OWN_SESSION`] while a call of ours is running in it, which a stop must never land
    /// under: nothing was asked and nothing is charged, like [`Self::deferred`] — and told apart
    /// from it because nothing is owed until that call is done and the pass holding it has spent
    /// its own let-go step, so a caller arms the next attempt at one whole stop rather than at the
    /// base gap.
    pub(crate) busy: Vec<String>,
}

// ── Browser answers ──────────────────────────────────────────────────────────

/// One pass's browser side: the chrome-use binary, the session every read and removal runs in
/// ([`close_sessions`] owns the choice), and the deadline the whole pass shares. `Copy`, so a
/// decide-and-act step hands it on without ceremony — and module-private: the entry points
/// spell their own `cli`/`session`/`deadline` out, so a caller outside never builds one.
#[derive(Debug, Clone, Copy)]
struct Door<'a> {
    cli: &'a Path,
    session: &'a str,
    deadline: Instant,
}

/// One group the browser holds right now, from `tabGroups.query`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LiveGroup {
    id: i64,
    title: String,
}

/// The bound for one spawn: the attempt's remaining budget, capped at `cap`, or
/// `None` when the budget is already spent (nothing is spawned then).
fn spawn_budget(deadline: Instant, cap: Duration) -> Option<Duration> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    (!remaining.is_zero()).then(|| remaining.min(cap))
}

/// The bound for a `session stop` a pass means to run to completion: its whole
/// [`crate::chrome::SESSION_STOP_TIMEOUT`] while `deadline` still holds that much, `None`
/// otherwise — so a stop is never spawned with a truncated bound, since one cut off at the
/// start could not have finished and would be read as a route tried when it was not. Used by
/// the two passes whose stops are meant to run whole ([`confirm_unclosable`]'s extra route per
/// name — best-effort in itself, since a stop that proved nothing about the tabs is not even a
/// route tried, while the names its stops leave uncovered stay retries — and [`stop_by_name`]'s
/// only route on a door-less host).
/// [`forget_settled_sessions`] spends [`spawn_budget`] instead — it attempts a stop with
/// whatever the pass has left — and reads that stop's answer by this function's rule: only one
/// that got the whole bound is charged a failed attempt, and a cut-short one is deferred
/// rather than counted. [`release_own_session`] is the other [`spawn_budget`] stop, whose
/// answer decides nothing at all.
fn stop_budget(deadline: Instant) -> Option<Duration> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    (remaining >= crate::chrome::SESSION_STOP_TIMEOUT)
        .then_some(crate::chrome::SESSION_STOP_TIMEOUT)
}

/// Whether a failed call must be read as saying nothing about the tabs rather than as a verdict on
/// them: it never got past the browser side's own structure — the transport, the daemon, or a
/// wrapper that refused before the extension was reached — it was cut off at our own bound
/// ([`NoVerdict::TimedOut`]), or it came back with no decision in it at all (see
/// [`note_unreadable`]). Such a call neither answered for the tabs nor showed its own route
/// failing to close them, so a name it covered stays queued rather than concluded.
fn proved_nothing_about_the_tabs(reason: &NoVerdict) -> bool {
    match reason {
        NoVerdict::TimedOut | NoVerdict::SpawnFailure | NoVerdict::Unreadable => true,
        NoVerdict::Reported(message) => crate::chrome::contract::unreached_browser_error(message),
    }
}

/// Note a [`NoVerdict::Unreadable`] as it passes: the answer carried no decision this route can
/// take — a field a parser here refuses to guess at, no envelope at all, or an envelope with
/// neither a verdict nor a reason — which the caller owes the owner as a fact about the host
/// ([`saw_unreadable_answer`]). Every other leg — an unreachable browser, a refusal — is a
/// transient this route retries without a report.
fn note_unreadable(reason: NoVerdict) -> NoVerdict {
    if matches!(reason, NoVerdict::Unreadable) {
        UNREADABLE_ANSWER.store(true, Ordering::SeqCst);
    }
    reason
}

/// A name this pass could not settle. The cause is logged here at `debug!` — a per-attempt
/// reason is not durable, and the caller owes the Issues view one row per left-open NAME, not
/// one per ladder step — and only the leg travels, because that is all the caller acts on.
fn retry(leg: RetryLeg, cause: &str) -> TabOutcome {
    debug!(
        cause,
        "chrome tab close not settled — retrying on a later pass"
    );
    TabOutcome::Retry(leg)
}

/// Leave every name in `indexes` a retry with the same `cause` — the shape a browser read
/// that did not happen leaves behind, since one such read decides for every name at once.
fn retry_all(
    outcomes: &mut [Option<TabOutcome>],
    indexes: impl IntoIterator<Item = usize>,
    cause: &str,
) {
    for index in indexes {
        outcomes[index] = Some(retry(RetryLeg::Silent, cause));
    }
}

/// Ask chrome-use for the browser's live tab groups.
async fn read_live_groups(door: Door<'_>) -> Result<Vec<LiveGroup>, NoVerdict> {
    let timeout = spawn_budget(door.deadline, TAB_READ_TIMEOUT).ok_or(NoVerdict::TimedOut)?;
    let envelope = door_call(
        door.cli,
        &["extension", "call", "tabGroups.query", "{}"],
        door.session,
        timeout,
    )
    .await
    .map_err(note_unreadable)?;
    parse_live_groups(&envelope).map_err(note_unreadable)
}

/// Whether none of `group_ids` is in the browser's live group list any more — the
/// browser's own answer to "did the removal take", never the call's exit status.
async fn groups_gone(door: Door<'_>, group_ids: &[i64]) -> bool {
    read_live_groups(door)
        .await
        .is_ok_and(|live| live.iter().all(|group| !group_ids.contains(&group.id)))
}

/// Ask chrome-use for the extension's own tab ledger and the browser's tab list.
async fn read_ownership(
    door: Door<'_>,
) -> Result<(HashSet<i64>, HashMap<i64, Vec<i64>>), NoVerdict> {
    let owned = read_owned_tabs(door).await?;
    let tabs = read_tabs_by_group(door).await?;
    Ok((owned, tabs))
}

/// Ask chrome-use for `extension state` and keep its `ownedTabs` ledger.
async fn read_owned_tabs(door: Door<'_>) -> Result<HashSet<i64>, NoVerdict> {
    let timeout = spawn_budget(door.deadline, TAB_READ_TIMEOUT).ok_or(NoVerdict::TimedOut)?;
    let envelope = door_call(door.cli, &["extension", "state"], door.session, timeout)
        .await
        .map_err(note_unreadable)?;
    parse_owned_tabs(&envelope).map_err(note_unreadable)
}

/// Ask chrome-use for `tabs.query` and index the answer by owning group.
async fn read_tabs_by_group(door: Door<'_>) -> Result<HashMap<i64, Vec<i64>>, NoVerdict> {
    let timeout = spawn_budget(door.deadline, TAB_READ_TIMEOUT).ok_or(NoVerdict::TimedOut)?;
    let envelope = door_call(
        door.cli,
        &["extension", "call", "tabs.query", "{}"],
        door.session,
        timeout,
    )
    .await
    .map_err(note_unreadable)?;
    parse_tabs_by_group(&envelope).map_err(note_unreadable)
}

/// Ask chrome-use what the browser side says about itself, from `extension status`.
/// `None` for every leg that answered nothing readable — the caller then treats the
/// door as "cannot establish".
async fn read_extension_status(door: Door<'_>) -> Option<ExtensionStatus> {
    let timeout = spawn_budget(door.deadline, TAB_READ_TIMEOUT)?;
    let envelope = door_call(door.cli, &["extension", "status"], door.session, timeout)
        .await
        .map_err(note_unreadable)
        .ok()?;
    Some(parse_extension_status(&envelope))
}

/// Whether the browser side provably has no ownership door at all
/// ([`ExtensionStatus::door_missing`], see the module doc's Honest limits). `false` when the
/// door is there, and also when nothing could be established about it.
pub(crate) async fn ownership_door_missing(cli: &Path, session: &str, deadline: Instant) -> bool {
    read_extension_status(Door {
        cli,
        session,
        deadline,
    })
    .await
    .is_some_and(|status| status.door_missing())
}

/// Remove `tab_ids` in one call. The ids are the call's own single argument — a nested
/// array, because the extension spreads the JSON as `chrome.tabs.remove`'s positional
/// arguments, so `[[1,2]]` is one argument (the id array) while `[1,2]` would be a tab id
/// and a callback.
async fn remove_tabs(door: Door<'_>, tab_ids: &[i64]) -> Result<(), NoVerdict> {
    let timeout = spawn_budget(door.deadline, TAB_READ_TIMEOUT).ok_or(NoVerdict::TimedOut)?;
    let payload = serde_json::to_string(&[tab_ids]).expect("an array of tab ids always serializes");
    door_call(
        door.cli,
        &["extension", "call", "tabs.remove", &payload],
        door.session,
        timeout,
    )
    .await
    .map(|_| ())
}

/// The extension's own words when it refuses a `tabs.remove`: relayed verbatim through the
/// helper's error field, as `call: tabs.remove refused — tab N is not owned by this relay
/// (agent-created or adopted tabs only)`. This is the only fragment that counts as the
/// browser answering about the tabs; a rephrase upstream degrades the leg to
/// [`RetryLeg::Silent`], never to a false left-open answer.
const REFUSAL_FRAGMENT: &str = "not owned by this relay";

/// Whether a failed `tabs.remove` was the browser side answering about the tabs, or a leg that
/// proved nothing about them: a relay that is not connected, a wedged daemon, a capability or CLI
/// rejection, a call cut off at our own bound and an unreadable answer all settle nothing, so none
/// of them may be read as the browser answering for the tabs — the caller concludes a left-open
/// name only once that answer has repeated. Strictly positive: only the extension's own ownership
/// sentence in the error counts.
fn remove_answered(reason: &NoVerdict) -> bool {
    let NoVerdict::Reported(message) = reason else {
        return false;
    };
    message.contains(REFUSAL_FRAGMENT)
}

/// Hand [`RemovalPlan::tab_ids`] to the browser's own removal door, then settle every
/// [`RemovalPlan::verifying`] name from the fresh read that follows ([`verify`]): the call's
/// own success is never proof of closure, only the browser's answers are.
async fn remove_and_verify(
    door: Door<'_>,
    matched: &[Vec<i64>],
    plan: &RemovalPlan,
    owned: &HashSet<i64>,
    outcomes: &mut [Option<TabOutcome>],
) {
    let remove_leg = match remove_tabs(door, &plan.tab_ids).await {
        // The call ran and the tab is still in the ledger: the browser answered for a tab
        // it holds and left it standing.
        Ok(()) => RetryLeg::LeftOpen,
        Err(reason) => {
            // A failed call closes nothing, but it is not a verdict either — the browser's
            // own answers still decide, read exactly as the plan read them. A ledger that no
            // longer holds these tabs is a verdict even here (the door and the ledger disagree
            // because the extension was replaced since); tabs it still lists leave the call a
            // retry — and only when the browser side actually answered about them, since only
            // the extension's own refusal sentence counts ([`remove_answered`]).
            debug!(
                reason = %reason.text(),
                "chrome tabs.remove did not go through — reading the browser again"
            );
            if remove_answered(&reason) {
                RetryLeg::LeftOpen
            } else {
                RetryLeg::Silent
            }
        }
    };
    verify(door, matched, &plan.verifying, owned, remove_leg, outcomes).await;
}

/// The fallback for a browser answer this module could not get: a host with no ownership door
/// takes chrome-use's own `session stop` by name for the names in `indices`; any other outcome
/// — a door that is there, or one that could not be established — leaves them as `Retry`,
/// because the failure was transient. The door is probed once per close
/// ([`ownership_door_missing`]); both call sites in [`close_sessions`] return as soon as this
/// answers.
async fn fallback_outcomes(
    door: Door<'_>,
    names: &[String],
    indices: &[usize],
    reason: &NoVerdict,
    owned_again: OwnedAgain<'_>,
    outcomes: &mut [Option<TabOutcome>],
) {
    if ownership_door_missing(door.cli, door.session, door.deadline).await {
        stop_by_name(door, names, indices, owned_again, outcomes).await;
        return;
    }
    retry_all(outcomes, indices.iter().copied(), reason.text());
}

/// The extension cannot answer ownership queries, so close each session the only way left:
/// chrome-use's own `session stop` by name. A stop the helper reports as done settles the name —
/// it reports success only after every tab it CREATED is confirmed closed, and on a host with no
/// door there is no ledger for a second opinion — while every failure stays a retry, because it
/// proves nothing about the tabs. A per-run verdict from here would state something this host
/// cannot establish, so what the helper's word cannot cover is the host-level residue the caller
/// reports once (see the module doc's Honest limits).
///
/// The names are asked in [`rotated_order`] — see there why the rotation matters.
async fn stop_by_name(
    door: Door<'_>,
    names: &[String],
    indices: &[usize],
    owned_again: OwnedAgain<'_>,
    outcomes: &mut [Option<TabOutcome>],
) {
    for slot in rotated_order(indices.len()) {
        let index = indices[slot];
        let name = &names[index];
        // The status read above took seconds, and stopping a session is what ends the run
        // driving it if one is back.
        if owned_again(name) {
            outcomes[index] = Some(retry(RetryLeg::Silent, "the run owns its session again"));
            continue;
        }
        let Some(timeout) = stop_budget(door.deadline) else {
            outcomes[index] = Some(retry(
                RetryLeg::Silent,
                "the attempt budget no longer holds a whole stop",
            ));
            continue;
        };
        outcomes[index] = Some(
            match run_cli_json_at(door.cli, &["session", "stop"], Some(name), timeout).await {
                Ok(_) => TabOutcome::Gone,
                Err(verdict) => retry(RetryLeg::Silent, verdict.text()),
            },
        );
    }
}

/// Verify a removal from the browser's own answers: re-read the live groups, and
/// for a name whose groups survive, the tabs still in them. Each name is then read
/// by [`settled_by_tabs`] — the same reading the removal was planned with, so a
/// name cannot mean one thing before the call and another after it.
///
/// `remove_leg` is the browser's answer about the call that did not close every tab; see
/// [`RetryLeg::LeftOpen`] for the two shapes that produce it.
async fn verify(
    door: Door<'_>,
    matched: &[Vec<i64>],
    verifying: &[usize],
    owned: &HashSet<i64>,
    remove_leg: RetryLeg,
    outcomes: &mut [Option<TabOutcome>],
) {
    let live = match read_live_groups(door).await {
        Ok(groups) => groups,
        Err(reason) => {
            retry_all(outcomes, verifying.iter().copied(), reason.text());
            return;
        }
    };
    let live_ids: HashSet<i64> = live.into_iter().map(|group| group.id).collect();

    let mut surviving: Vec<(usize, Vec<i64>)> = Vec::new();
    for &i in verifying {
        let group_ids: Vec<i64> = matched[i]
            .iter()
            .copied()
            .filter(|id| live_ids.contains(id))
            .collect();
        if group_ids.is_empty() {
            outcomes[i] = Some(TabOutcome::Gone);
        } else {
            surviving.push((i, group_ids));
        }
    }
    if surviving.is_empty() {
        return;
    }

    let tabs = match read_tabs_by_group(door).await {
        Ok(tabs) => tabs,
        Err(reason) => {
            retry_all(outcomes, surviving.iter().map(|(i, _)| *i), reason.text());
            return;
        }
    };
    for (i, group_ids) in surviving {
        // A group that still holds a tab the extension owns was not closed; one
        // holding only tabs it does not own is a leftover no route here can close.
        // A group whose tabs all left between the two reads is simply gone.
        outcomes[i] = Some(match settled_by_tabs(&tabs_of(&group_ids, &tabs), owned) {
            Some(settled) => settled,
            None => retry(remove_leg, "the browser did not remove the session's tabs"),
        });
    }
}

/// Where the next rotated stop order starts — the order one pass asks its `session stop`s in,
/// shared by the confirmation ([`confirm_unclosable`]) and the door-less fallback
/// ([`stop_by_name`]). Process-wide and monotonic, deliberately: nothing in it is per name,
/// since a name that needs a stop has had its turn due.
static STOP_ROTATION: AtomicUsize = AtomicUsize::new(0);

/// The order one pass asks its stops in — the same order one further along on every call.
/// A pass's stop slice fits about one stop, so a stable order would spend it on the same
/// leading names for as long as they need convincing while the tail never got its turn; on
/// a door-less host ([`stop_by_name`]), where the stop is the only close route there is,
/// that tail is exactly what would be left open.
fn rotated_order(len: usize) -> impl Iterator<Item = usize> {
    let rotation = if len == 0 {
        0
    } else {
        STOP_ROTATION.fetch_add(1, Ordering::Relaxed) % len
    };
    (0..len).map(move |offset| (offset + rotation) % len)
}

/// The last word before a name is called a leftover no route can close.
///
/// The extension's ledger holds none of the tabs in that name's group, which is the whole
/// reason no `tabs.remove` may be asked for them. Two things can be true instead of "no
/// route can close these": chrome-use's own per-session record (endpoint-bound, but it may
/// still match) can close the tabs it created, and the ledger read can simply have failed —
/// the extension's own storage read is not reported to us, and the empty ledger it then
/// hands back reads exactly like one that lost its tabs. So the other route is asked once
/// and the browser is read again: only an answer that REPEATS — still no tab the extension
/// owns in a group still standing — is the verdict. A read that fails, a stop the pass
/// could not give its whole bound, or a tab the second read finds in the ledger leaves the
/// name a retry for the next pass to plan again.
async fn confirm_unclosable(
    door: Door<'_>,
    names: &[String],
    matched: &[Vec<i64>],
    indexes: &[usize],
    owned_again: OwnedAgain<'_>,
    outcomes: &mut [Option<TabOutcome>],
) {
    // The stops are one best-effort extra route each, and their answers are not read: the
    // reads below are what decides every name, except the ones a stop the pass could not hold
    // in full left queued (see the loop). So the stops get only what is left beyond the reads'
    // own phase reserve ([`READ_PHASE_RESERVE`]).
    let stops_until = door
        .deadline
        .checked_sub(READ_PHASE_RESERVE)
        .unwrap_or(door.deadline);
    // The names are asked in the rotated order ([`rotated_order`]).
    let order: Vec<usize> = rotated_order(indexes.len())
        .map(|offset| indexes[offset])
        .collect();
    // The names the reads below must NOT conclude: one whose run owns the session again,
    // and one whose stop proved nothing about the tabs. Both stay queued — a leftover is
    // never called unclosable while a route that may close it has not been tried, and the
    // next pass asks the next name's stop.
    let mut keep_queued: HashSet<usize> = HashSet::new();
    for (position, &i) in order.iter().enumerate() {
        // The reads that decided this took seconds; stopping a session is what ends the
        // run driving it if one is back, and the session's own route is asked only while
        // the run that would own it is still gone.
        if owned_again(&names[i]) {
            outcomes[i] = Some(retry(RetryLeg::Silent, "the run owns its session again"));
            keep_queued.insert(i);
            continue;
        }
        let Some(timeout) = stop_budget(stops_until) else {
            for &j in &order[position..] {
                outcomes[j] = Some(retry(
                    RetryLeg::Silent,
                    "the pass's stop slice no longer holds a whole stop",
                ));
                keep_queued.insert(j);
            }
            break;
        };
        // Whatever this call answers is not read as the verdict: a host WITH the door has
        // the browser as its authority, so this is one more route tried, not a new one — and
        // a failure that proved nothing about the tabs is not even a route tried.
        match run_cli_json_at(door.cli, &["session", "stop"], Some(&names[i]), timeout).await {
            Err(verdict) if proved_nothing_about_the_tabs(&verdict) => {
                outcomes[i] = Some(retry(
                    RetryLeg::Silent,
                    "the session's own stop proved nothing about the tabs",
                ));
                keep_queued.insert(i);
            }
            Err(verdict) => debug!(
                name = %names[i],
                reason = %verdict.text(),
                "the session's own stop route did not close its tabs"
            ),
            Ok(_) => {}
        }
    }
    // Nothing left to read for: every name here is already answered as a retry — a run owning
    // its session again, or a stop this pass could not hold in full — and the concluding loop
    // below skips exactly those names, so the three reads would decide nothing.
    if keep_queued.len() == indexes.len() {
        return;
    }
    let (owned, tabs) = match read_ownership(door).await {
        Ok(ownership) => ownership,
        Err(reason) => {
            retry_all(outcomes, indexes.iter().copied(), reason.text());
            return;
        }
    };
    let live = match read_live_groups(door).await {
        Ok(live) => live,
        Err(reason) => {
            retry_all(outcomes, indexes.iter().copied(), reason.text());
            return;
        }
    };
    let live_ids: HashSet<i64> = live.into_iter().map(|group| group.id).collect();
    for &i in indexes {
        // A name a run took back while this ran, or one whose stop proved nothing about the
        // tabs, is NOT read from the browser: it was already answered as a retry above,
        // and classifying it here would write a verdict about tabs a live run owns, or one
        // no route was actually tried for.
        if keep_queued.contains(&i) {
            continue;
        }
        let group_ids: Vec<i64> = matched[i]
            .iter()
            .copied()
            .filter(|id| live_ids.contains(id))
            .collect();
        outcomes[i] = Some(match settled_by_tabs(&tabs_of(&group_ids, &tabs), &owned) {
            Some(settled) => settled,
            None => retry(
                RetryLeg::Silent,
                "the browser still holds a tab the extension owns",
            ),
        });
    }
}

// ── Decisions ────────────────────────────────────────────────────────────────

/// The browser answered what it holds; this is what the pass decided from it.
struct RemovalPlan {
    /// One slot per requested name: an outcome where the answer already settles
    /// it, `None` where only the remove call and its verification can.
    settled: Vec<Option<TabOutcome>>,
    /// The ids to hand to `tabs.remove`, in first-seen order, each once.
    tab_ids: Vec<i64>,
    /// The names awaiting the remove call's verification, in name order.
    verifying: Vec<usize>,
}

/// The ids of the live groups whose title is exactly `title` — the match every close in
/// this module is scoped by: a group renamed in Chrome simply matches nothing.
fn group_ids_titled(groups: &[LiveGroup], title: &str) -> Vec<i64> {
    groups
        .iter()
        .filter(|group| group.title == title)
        .map(|group| group.id)
        .collect()
}

/// What the browser's own answer ALREADY settles about one name's tabs: `None`
/// while a tab the extension owns is still there, because only the remove call and
/// a fresh read can say what becomes of it. Shared by [`plan_removal`] and
/// [`verify`] so the same browser answer can never be read two ways.
fn settled_by_tabs(tabs: &[i64], owned: &HashSet<i64>) -> Option<TabOutcome> {
    if tabs.is_empty() {
        return Some(TabOutcome::Gone);
    }
    if tabs.iter().all(|tab| !owned.contains(tab)) {
        return Some(TabOutcome::Unclosable);
    }
    None
}

/// Decide, per name, what the browser's own answer allows right now. `taken_back` is the
/// caller's liveness re-check, one flag per name ([`OwnedAgain`]): a name a run owns again
/// is left alone whatever the browser answered — except one whose title the browser holds no
/// live group of, which is [`TabOutcome::Gone`] whoever owns it, there being nothing left to
/// leave alone.
fn plan_removal(
    matched: &[Vec<i64>],
    tabs_by_group: &HashMap<i64, Vec<i64>>,
    owned: &HashSet<i64>,
    taken_back: &[bool],
) -> RemovalPlan {
    let mut settled: Vec<Option<TabOutcome>> = vec![None; matched.len()];
    let mut tab_ids: Vec<i64> = Vec::new();
    let mut seen: HashSet<i64> = HashSet::new();
    let mut verifying: Vec<usize> = Vec::new();
    for (i, group_ids) in matched.iter().enumerate() {
        if group_ids.is_empty() {
            // No live group of that title: nothing to close, whoever owns the name — and the
            // liveness re-check below must not turn that into a retry, which would keep a record
            // for a rung longer over a group the browser has just said is not there.
            settled[i] = Some(TabOutcome::Gone);
            continue;
        }
        if taken_back[i] {
            settled[i] = Some(retry(RetryLeg::Silent, "the run owns its sessions again"));
            continue;
        }
        let tabs = tabs_of(group_ids, tabs_by_group);
        if let Some(outcome) = settled_by_tabs(&tabs, owned) {
            settled[i] = Some(outcome);
            continue;
        }
        for tab in owned_group_tabs(group_ids, tabs_by_group, owned) {
            if seen.insert(tab) {
                tab_ids.push(tab);
            }
        }
        verifying.push(i);
    }
    RemovalPlan {
        settled,
        tab_ids,
        verifying,
    }
}

/// Every tab the browser places in any of `group_ids`.
fn tabs_of(group_ids: &[i64], tabs_by_group: &HashMap<i64, Vec<i64>>) -> Vec<i64> {
    let mut tabs: Vec<i64> = group_ids
        .iter()
        .filter_map(|id| tabs_by_group.get(id))
        .flatten()
        .copied()
        .collect();
    tabs.sort_unstable();
    tabs.dedup();
    tabs
}

/// The tabs of `group_ids` the extension's own ledger holds — the only ids `tabs.remove`
/// may be asked for, since the extension refuses the whole call otherwise.
fn owned_group_tabs(
    group_ids: &[i64],
    tabs_by_group: &HashMap<i64, Vec<i64>>,
    owned: &HashSet<i64>,
) -> Vec<i64> {
    tabs_of(group_ids, tabs_by_group)
        .into_iter()
        .filter(|tab| owned.contains(tab))
        .collect()
}

/// One `(name, outcome)` per requested name, in the order given. Every name is
/// decided by the time this runs; a gap would be a bug in this module, so it comes back
/// as a retry — a record kept alive and retried costs a pass, a panic here would kill
/// the release task for the whole boot.
fn zip_outcomes(names: &[String], outcomes: Vec<Option<TabOutcome>>) -> Vec<(String, TabOutcome)> {
    names
        .iter()
        .cloned()
        .zip(outcomes)
        .map(|(name, outcome)| {
            (
                name,
                outcome.unwrap_or_else(|| {
                    retry(
                        RetryLeg::Silent,
                        "this pass left a requested name undecided",
                    )
                }),
            )
        })
        .collect()
}

// ── Envelope parsing ─────────────────────────────────────────────────────────

/// The live group list out of a `tabGroups.query` envelope. `Err` when the
/// browser's answer carried no readable list at all — a plain refusal the caller
/// turns into `Retry`, never into "nothing exists" — and also when a listed group
/// carries no integer `id` or no string `title`: a group silently dropped from
/// the answer would read as a group that is gone, the one verdict this route must
/// never invent.
fn parse_live_groups(envelope: &Value) -> Result<Vec<LiveGroup>, NoVerdict> {
    let result = envelope
        .get("data")
        .and_then(|data| data.get("result"))
        .and_then(Value::as_array)
        .ok_or(NoVerdict::Unreadable)?;
    result
        .iter()
        .map(|group| {
            Ok(LiveGroup {
                id: group
                    .get("id")
                    .and_then(Value::as_i64)
                    .ok_or(NoVerdict::Unreadable)?,
                title: group
                    .get("title")
                    .and_then(Value::as_str)
                    .ok_or(NoVerdict::Unreadable)?
                    .to_string(),
            })
        })
        .collect()
}

/// The extension's `ownedTabs` ledger out of an `extension state` envelope. Only
/// the ledger is read: `data.groups` in the same answer is the extension's own
/// in-memory name→id map — empty after a service-worker restart, and never the
/// browser's live group list ([`read_live_groups`] is that) — so it must never be
/// enumerated. An element that is not an integer makes the whole answer unreadable
/// rather than dropping that tab: the ledger is what allows a close to be asked for
/// at all, and a swallowed entry would read as a tab the extension does not hold.
fn parse_owned_tabs(envelope: &Value) -> Result<HashSet<i64>, NoVerdict> {
    let owned = envelope
        .get("data")
        .and_then(|data| data.get("ownedTabs"))
        .and_then(Value::as_array)
        .ok_or(NoVerdict::Unreadable)?;
    owned
        .iter()
        .map(|tab| tab.as_i64().ok_or(NoVerdict::Unreadable))
        .collect()
}

/// The tabs out of a `tabs.query` envelope, indexed by the group each belongs to.
/// `groupId` is `-1` for an ungrouped tab, which belongs to no session's group and
/// is simply skipped. A tab carrying no integer `id` or `groupId` makes the whole
/// answer unreadable: a tab dropped here would leave its group looking empty,
/// which reads as a group whose tabs are all gone.
fn parse_tabs_by_group(envelope: &Value) -> Result<HashMap<i64, Vec<i64>>, NoVerdict> {
    let result = envelope
        .get("data")
        .and_then(|data| data.get("result"))
        .and_then(Value::as_array)
        .ok_or(NoVerdict::Unreadable)?;
    let mut by_group: HashMap<i64, Vec<i64>> = HashMap::new();
    for tab in result {
        let id = tab
            .get("id")
            .and_then(Value::as_i64)
            .ok_or(NoVerdict::Unreadable)?;
        let group = tab
            .get("groupId")
            .and_then(Value::as_i64)
            .ok_or(NoVerdict::Unreadable)?;
        if group >= 0 {
            by_group.entry(group).or_default().push(id);
        }
    }
    Ok(by_group)
}

/// What the browser side reports about the ownership door, out of an `extension status`
/// envelope: the live extension's version and the one installed in the driving profile
/// (see [`ExtensionStatus`]), plus whether the extension is absent at all — the CLI's own
/// explicit-null signal, read by the same parser `chrome_daemon` uses
/// ([`extension_state_from`]), so the absence rule has one home — and whether it is
/// installed but disabled, which that same parser reports.
fn parse_extension_status(envelope: &Value) -> ExtensionStatus {
    let version = |value: Option<&Value>| {
        value
            .and_then(Value::as_str)
            .and_then(|raw| semver::Version::parse(raw).ok())
    };
    let data = envelope.get("data");
    let state = extension_state_from(envelope);
    ExtensionStatus {
        live: version(data.and_then(|data| data.get("liveExtensionVersion"))),
        installed: version(
            data.and_then(|data| data.get("chromeExtension"))
                .and_then(|extension| extension.get("version")),
        ),
        absent: state == ExtensionState::Absent,
        disabled: state == ExtensionState::Disabled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The door session must stay in the family the product's own session sweeps close:
    /// that sweep — not this module — is what closes the group when a pass dies before
    /// [`release_own_session`] reaches it.
    #[test]
    fn the_door_session_is_a_mahbot_session_name() {
        assert!(crate::tools::chrome_daemon::is_mahbot_session_name(
            OWN_SESSION
        ));
    }

    /// The stop order rotates one name further along on every call, so a pass's single-stop
    /// slice does not spend itself on the same leading name while the tail never gets a turn;
    /// a lone name is always itself.
    #[test]
    #[serial_test::serial(chrome_release)]
    fn rotated_order_moves_the_start_by_one_each_call() {
        let len = 5;
        let first: Vec<usize> = rotated_order(len).collect();
        let second: Vec<usize> = rotated_order(len).collect();
        assert_eq!(
            second,
            (0..len)
                .map(|offset| (offset + first[0] + 1) % len)
                .collect::<Vec<_>>(),
            "one step further along than the call before it"
        );
        assert_eq!(rotated_order(1).collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn names_match_group_titles_exactly() {
        let groups = vec![
            LiveGroup {
                id: 1,
                title: "agent-tab-aaaa-bbbb-a".to_string(),
            },
            LiveGroup {
                id: 2,
                title: "agent-tab-aaaa-bbbb-a".to_string(),
            },
            LiveGroup {
                id: 3,
                title: "not-a-session".to_string(),
            },
        ];
        let names = [
            "agent-tab-aaaa-bbbb-a".to_string(),
            "agent-tab-aaaa-bbbb-a-renamed".to_string(),
        ];
        assert_eq!(group_ids_titled(&groups, &names[0]), vec![1, 2]);
        assert!(group_ids_titled(&groups, &names[1]).is_empty());
    }

    #[test]
    fn plan_splits_gone_unclosable_and_closeable() {
        let tabs_by_group = HashMap::from([(10, vec![100, 101]), (11, vec![200]), (12, vec![300])]);
        let owned: HashSet<i64> = [100, 200].into_iter().collect();
        // name 0: no group                -> Gone
        // name 1: group holds an owned tab -> closeable
        // name 2: group holds only tabs the extension no longer holds -> Unclosable
        let matched = vec![vec![], vec![10], vec![12]];
        let plan = plan_removal(&matched, &tabs_by_group, &owned, &[false; 3]);
        assert_eq!(plan.settled[0], Some(TabOutcome::Gone));
        assert_eq!(plan.settled[1], None);
        assert_eq!(plan.settled[2], Some(TabOutcome::Unclosable));
        assert_eq!(plan.tab_ids, vec![100]);
        assert_eq!(plan.verifying, vec![1]);
    }

    /// A name whose run registered again while the pass was reading the browser is left
    /// alone for as long as the browser still holds a group of its title: the sessions are
    /// the run's. A title the browser holds no group of is the exception — there is nothing
    /// to leave alone, and the read that said so decides it.
    #[test]
    fn a_name_taken_back_mid_pass_leaves_a_live_group_alone() {
        let tabs_by_group = HashMap::from([(10, vec![100])]);
        let owned: HashSet<i64> = [100].into_iter().collect();
        let plan = plan_removal(&[vec![10]], &tabs_by_group, &owned, &[true]);
        assert_eq!(
            plan.settled[0],
            Some(TabOutcome::Retry(RetryLeg::Silent)),
            "the caller keeps its record and closes nothing"
        );
        assert!(plan.tab_ids.is_empty());
        assert!(plan.verifying.is_empty());

        // A name the browser holds no group of is gone whoever owns it: the re-check above is
        // about tabs to leave alone, and there are none — keeping the record for a rung longer
        // would be a pass over a decision the same read already made.
        let absent = plan_removal(&[vec![]], &tabs_by_group, &owned, &[true]);
        assert_eq!(absent.settled[0], Some(TabOutcome::Gone));
    }

    #[test]
    fn a_group_with_no_tabs_is_gone() {
        let tabs_by_group = HashMap::new();
        let owned: HashSet<i64> = [100].into_iter().collect();
        let plan = plan_removal(&[vec![10]], &tabs_by_group, &owned, &[false]);
        assert_eq!(plan.settled[0], Some(TabOutcome::Gone));
        assert!(plan.tab_ids.is_empty());
        assert!(plan.verifying.is_empty());
    }

    /// Only the extension's own refusal sentence is the browser answering about the tabs. A
    /// structured error that says nothing about them — a relay, daemon or auto-launch
    /// wrapper, a capability or CLI rejection, a timeout text — proves nothing, and neither
    /// does an answer with no readable envelope: all of them stay a silent retry, so a
    /// transient or structural failure is never counted as the browser answering.
    #[test]
    fn only_the_extensions_own_refusal_counts_as_an_answered_close() {
        let refusal = NoVerdict::Reported(
            "call: tabs.remove refused — tab 7 is not owned by this relay (agent-created or \
             adopted tabs only)"
                .to_string(),
        );
        assert!(remove_answered(&refusal));
        assert!(!proved_nothing_about_the_tabs(&refusal));
        for never in [
            NoVerdict::Reported("relay isn't connected".to_string()),
            NoVerdict::Reported("Browser not launched".to_string()),
            NoVerdict::Reported(
                "could not drive your Chrome through the ab-connect extension".to_string(),
            ),
            NoVerdict::Reported(
                "extension call requires ab-connect 0.5.25 or newer (the extension did not \
                 announce the `call` capability)"
                    .to_string(),
            ),
            NoVerdict::Reported("CDP command timed out after 30s: ABExt.call".to_string()),
            NoVerdict::TimedOut,
            NoVerdict::SpawnFailure,
            NoVerdict::Unreadable,
        ] {
            assert!(
                !remove_answered(&never),
                "{never:?} must not read as a refusal"
            );
            assert!(
                proved_nothing_about_the_tabs(&never),
                "{never:?} must prove nothing about the tabs"
            );
        }
    }

    #[test]
    fn live_groups_parse_from_the_result_list() {
        let envelope = json!({
            "success": true,
            "data": {
                "policy": "call-v1",
                "result": [
                    {"id": 1_026_066_125, "title": "not-a-session", "windowId": 304_840_673},
                    {"id": 5, "title": "agent-tab-aaaa-bbbb-a"}
                ]
            }
        });
        assert_eq!(
            parse_live_groups(&envelope).expect("groups"),
            vec![
                LiveGroup {
                    id: 1_026_066_125,
                    title: "not-a-session".to_string(),
                },
                LiveGroup {
                    id: 5,
                    title: "agent-tab-aaaa-bbbb-a".to_string(),
                },
            ]
        );
        assert!(parse_live_groups(&json!({"success": true, "data": {}})).is_err());
        // A listed group the answer does not describe completely makes the whole answer
        // unreadable: dropping it would read as a group that is gone.
        assert!(
            parse_live_groups(&json!({
                "data": {"result": [{"id": "5", "title": "agent-tab-aaaa-bbbb-a"}]}
            }))
            .is_err()
        );
        assert!(
            parse_live_groups(&json!({
                "data": {"result": [{"id": 5, "title": 7}]}
            }))
            .is_err()
        );
        assert!(parse_live_groups(&json!({"data": {"result": [{"id": 5}]}})).is_err());
    }

    #[test]
    fn owned_tabs_come_from_the_ledger_not_the_stale_group_map() {
        let envelope = json!({
            "success": true,
            "data": {
                "connected": true,
                "ownedTabs": [304_864_598, 304_865_350],
                "groups": [{"id": 1, "name": "stale"}, {"id": 2, "name": "stale"}]
            }
        });
        assert_eq!(
            parse_owned_tabs(&envelope).expect("ledger"),
            [304_864_598, 304_865_350].into_iter().collect()
        );
        // An entry the answer cannot read as a tab id makes the ledger unreadable rather
        // than smaller: a swallowed tab would read as one the extension does not hold.
        assert!(parse_owned_tabs(&json!({"data": {"ownedTabs": [1, "2"]}})).is_err());
    }

    #[test]
    fn tabs_are_indexed_by_group_and_ungrouped_tabs_are_skipped() {
        let envelope = json!({
            "success": true,
            "data": {
                "policy": "call-v1",
                "result": [
                    {"id": 1, "groupId": 10, "windowId": 3},
                    {"id": 2, "groupId": -1, "windowId": 3},
                    {"id": 3, "groupId": 10, "windowId": 4},
                    {"id": 4, "groupId": 11, "windowId": 4}
                ]
            }
        });
        let by_group = parse_tabs_by_group(&envelope).expect("tabs");
        assert_eq!(by_group.get(&10), Some(&vec![1, 3]));
        assert_eq!(by_group.get(&11), Some(&vec![4]));
        assert!(!by_group.contains_key(&-1));
        // A tab the answer does not describe completely makes the whole answer unreadable:
        // skipping it would leave its group looking empty, which reads as gone.
        assert!(
            parse_tabs_by_group(&json!({
                "data": {"result": [{"id": 1, "groupId": "10"}]}
            }))
            .is_err()
        );
        assert!(
            parse_tabs_by_group(&json!({
                "data": {"result": [{"id": 1}]}
            }))
            .is_err()
        );
    }

    #[test]
    fn extension_status_reads_the_door() {
        // The field shapes are the real ones: `installed` is the host manifest this
        // product writes on every start and answers nothing about the extension.
        let old_live = json!({
            "success": true,
            "data": {
                "installed": true,
                "liveExtensionVersion": "0.5.24",
                "chromeExtension": {"version": "0.5.24"}
            }
        });
        assert!(
            parse_extension_status(&old_live).door_missing(),
            "an extension older than the gate has no door"
        );
        // No extension in the driving profile: the live version AND chromeExtension are
        // null, while `installed` stays true — the host manifest is still there.
        let absent = json!({
            "success": true,
            "data": {
                "installed": true,
                "liveExtensionVersion": null,
                "chromeExtension": null
            }
        });
        assert!(
            parse_extension_status(&absent).door_missing(),
            "an absent extension is the same gap, told with chromeExtension"
        );
        // Installed but DISABLED: nothing answers an `extension call`, so this is the same
        // no-door host whatever version the extension reports.
        let disabled = json!({
            "success": true,
            "data": {
                "installed": true,
                "chromeExtension": {"version": "0.5.26", "disableReasons": ["user"]}
            }
        });
        assert!(
            parse_extension_status(&disabled).door_missing(),
            "a disabled extension's door is missing even when its version passes the gate"
        );
        // Installed but not connected right now (Chrome closed, relay restarting): the
        // door is there, so these names stay retries.
        let unconnected = json!({
            "success": true,
            "data": {
                "installed": true,
                "liveExtensionVersion": null,
                "chromeExtension": {"version": "0.5.26"}
            }
        });
        assert!(!parse_extension_status(&unconnected).door_missing());
        // An old one that is merely not connected is still no door.
        let old_unconnected = json!({
            "data": {"installed": true, "chromeExtension": {"version": "0.5.24"}}
        });
        assert!(parse_extension_status(&old_unconnected).door_missing());
        // The live version decides while there is one.
        let live_new = json!({
            "data": {"installed": true, "liveExtensionVersion": "0.5.26"}
        });
        assert!(!parse_extension_status(&live_new).door_missing());
        // An envelope that does not describe the extension at all (an old CLI), or one that
        // describes it without a version, proves nothing and never claims the fixable gap.
        assert!(!parse_extension_status(&json!({"data": {}})).door_missing());
        assert!(!parse_extension_status(&json!({"data": {"chromeExtension": {}}})).door_missing());
    }
}
