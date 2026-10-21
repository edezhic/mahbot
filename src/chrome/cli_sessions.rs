//! The command-line surface's own session record: which sessions `mahbot chrome`
//! created, when each was last used, and which are marked as kept.
//!
//! chrome-use keeps no lifetime of its own for a named session, and its `session list`
//! reports only the sessions whose daemon is still alive (the product pins that daemon's
//! idle exit at five minutes). This record is what the surface needs for both: a small
//! JSON file at the storage root, written by the calls that use a session and read by
//! every sweep.
//!
//! ## The lifetime
//!
//! A session this CLI creates — named or throwaway — lives [`LIFETIME_MINUTES`] minutes
//! after its last use, and a later call with the same name restarts that clock. When
//! the lifetime runs out the session is closed with `session stop`, which closes the
//! tabs it CREATED and leaves the ones it adopted. Three triggers enforce it: every
//! CLI invocation sweeps first (so it works where no service process runs), the chrome
//! watchdog sweeps at its upkeep tick, and chrome-use is handed the same number as its
//! own group-cleanup backstop ([`backstop_ms`], `AGENT_BROWSER_GROUP_CLEANUP_MS`; the
//! helper defers that sweep while the project's other browser work is live — a known
//! limitation, not worked around).
//!
//! A session is never closed while a call is using it, and that is the call's own
//! statement rather than an arithmetic argument: a call adds a mark naming its process
//! when it starts ([`note_use`]) and removes it when it ends ([`note_end`]), and no sweep
//! closes a session that carries a live mark. A mark is live only while the
//! process that set it is still there and its own deadline has not passed, so a call that
//! crashed or was killed outright cannot pin a session: its mark is stale, and the
//! ordinary lifetime closes the session from its last use.
//!
//! ## The keep mark
//!
//! `--keep` marks a session as having no lifetime. Only the call that CREATES the
//! session can set it: a session that already exists without it cannot gain it later,
//! because its lifetime was already the contract. The mark is sticky (later calls
//! without the flag do not clear it), every sweep skips it, and only an explicit `mahbot
//! chrome session stop` ends it. Its existence test has two halves: this record (a session
//! recorded without the mark refuses it), and chrome-use's own list (a session running
//! with no record here — created before this record existed, or after it was lost — is
//! refused the mark too). chrome-use's list sees live daemons only, so a session whose
//! daemon already exited counts as freshly created: there is no daemon left for `0` to
//! reach, and whether chrome-use's group sweep ever touches its leftovers is outside this
//! product's reach.
//!
//! ## One lock, held across a close
//!
//! Concurrent CLI processes and the service share the record, so every read and write
//! goes through one whole-file lock ([`Ledger`]), held off the async worker wherever there
//! is one — another process can hold it for the length of its own close, and in the
//! service that wait must not stall a worker. The expiry sweep keeps the lock across the
//! `session stop` it runs, so a call that arrives for that session waits out the close
//! instead of interleaving with it. A caller for any other session waits as well, but that
//! is the one long hold (one bounded stop per pass), after which it records its use
//! against whatever the close left. A close the helper does not confirm is retried on the
//! next pass up to [`CLOSE_ATTEMPT_LIMIT`] — then the record is dropped rather than ask a
//! browser side that stays down, on every tick, forever. A close that never reached the
//! helper (no binary to run) asked nothing and is not counted: that record waits for the
//! helper to return, since there is no browser side to close anything meanwhile.
//!
//! Fail-open: no storage root, or an unreadable/unparsable file, is an empty record —
//! the CLI still works, it just cannot time anything out. That is the opposite of
//! `util::lock`'s probes, deliberately: those gate opening the live stores and must
//! never read "the probe could not tell" as "free", while this file only decides when a
//! session's tabs are closed, so a broken record must not take the CLI down with it.
//! It also takes its whole-file lock blocking (std's [`File::lock`]), which
//! [`crate::util::lock`] does not offer — its probe is non-blocking by design.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::tools::chrome_daemon::SessionRecovery;

/// How long a session this CLI created lives after its last use.
pub(crate) const LIFETIME_MINUTES: u64 = 10;

/// The same lifetime in the record's unit, and the number chrome-use is handed as its
/// cleanup backstop.
const LIFETIME_MS: u64 = LIFETIME_MINUTES * 60_000;

/// How long a call's in-flight mark is honoured whatever the process table says: the
/// backstop for a pid the system has handed to another process. A call that outlives it
/// (its own `--timeout` is the caller's business, not this record's) loses the mark, and
/// the ordinary lifetime applies again — the pre-mark behaviour, not a regression.
const IN_FLIGHT_TTL_MS: u64 = 60 * 60 * 1000;

/// Consecutive closes the helper left unconfirmed before the record is dropped. The
/// sweep runs at the service's upkeep tick and on every CLI call, so a browser side that
/// is really gone settles within a few passes instead of being asked forever.
const CLOSE_ATTEMPT_LIMIT: u32 = 5;

const SESSIONS_FILE_NAME: &str = "chrome-cli-sessions.json";
const LOCK_FILE_NAME: &str = "chrome-cli-sessions.lock";

#[derive(Debug, Serialize, Deserialize)]
struct SessionRecord {
    last_used_ms: u64,
    #[serde(default)]
    kept: bool,
    /// Consecutive closes the helper did not confirm; every use resets it.
    #[serde(default)]
    unconfirmed_closes: u32,
    /// The calls using this session right now: what keeps every sweep away from it. One
    /// mark per process, so two concurrent calls on one name cannot clear each other's.
    #[serde(default)]
    in_flight: Vec<InFlight>,
}

/// A call in progress, as its record carries it: the process running it, and the moment
/// after which the mark is stale whatever the process table says.
#[derive(Debug, Serialize, Deserialize)]
struct InFlight {
    pid: u32,
    until_ms: u64,
}

impl InFlight {
    fn new(now: u64) -> Self {
        Self {
            pid: std::process::id(),
            until_ms: now.saturating_add(IN_FLIGHT_TTL_MS),
        }
    }

    /// Whether the call that set this mark can still be running: its process is there and
    /// its own deadline has not passed. Either answer "no" makes the mark stale, so a
    /// crashed or killed call never pins a session.
    fn is_live(&self, now: u64) -> bool {
        now < self.until_ms && crate::util::process_alive(self.pid)
    }
}

type Records = BTreeMap<String, SessionRecord>;

/// One row of `mahbot chrome session list`.
pub(crate) struct SessionRow {
    /// The name a person reads for this session ([`display_name`]).
    pub(crate) name: String,
    pub(crate) kept: bool,
    pub(crate) idle_seconds: u64,
}

/// What one call did to the record.
pub(crate) struct UseOutcome {
    /// This call created the session's record.
    pub(crate) created: bool,
    /// This call created the session and set the keep mark on it.
    pub(crate) marked: bool,
    /// The session carries the mark after this call.
    pub(crate) kept: bool,
}

struct Ledger {
    root: PathBuf,
    records: Records,
    /// Held until this value drops; the O.S. releases the lock when the file closes.
    _lock: File,
}

impl Ledger {
    /// Take the record under its whole-file lock. The lock is deliberately held across a
    /// close another process is running (see the module doc), so this is one of the few
    /// blocking waits longer than the sub-millisecond scope [`crate::util::with_block_in_place`]
    /// is written for: the helper moves the wait off the async worker, so a service tick
    /// waits here without stalling the runtime.
    fn open() -> Option<Self> {
        crate::util::with_block_in_place(|| {
            let root = storage_root()?;
            let lock = lock_root(&root)?;
            let records = read_records(&root);
            Some(Self {
                root,
                records,
                _lock: lock,
            })
        })
    }

    fn persist(&self) {
        let json = match serde_json::to_string(&self.records) {
            Ok(json) => json,
            Err(error) => {
                warn!(error = %error, "chrome CLI session record not written: serialization failed");
                return;
            }
        };
        let path = self.root.join(SESSIONS_FILE_NAME);
        if let Err(error) =
            crate::util::with_block_in_place(|| crate::util::write_json_record(&path, &json))
        {
            warn!(error = %error, "chrome CLI session record not written");
        }
    }
}

fn storage_root() -> Option<PathBuf> {
    crate::config::CONFIG
        .try_storage_root()
        .or_else(|| crate::config::default_config_dir().ok())
}

fn lock_root(root: &Path) -> Option<File> {
    if let Err(error) = std::fs::create_dir_all(root) {
        warn!(error = %error, "chrome CLI session record unavailable: storage root not creatable");
        return None;
    }
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join(LOCK_FILE_NAME))
    {
        Ok(file) => file,
        Err(error) => {
            warn!(error = %error, "chrome CLI session record unavailable: lock file not openable");
            return None;
        }
    };
    match file.lock() {
        Ok(()) => Some(file),
        Err(error) => {
            warn!(error = %error, "chrome CLI session record unavailable: lock not taken");
            None
        }
    }
}

fn read_records(root: &Path) -> Records {
    let path = root.join(SESSIONS_FILE_NAME);
    let json = match std::fs::read_to_string(&path) {
        Ok(json) => json,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Records::new(),
        Err(error) => {
            warn!(error = %error, path = %path.display(), "chrome CLI session record unreadable — starting empty");
            return Records::new();
        }
    };
    match serde_json::from_str(&json) {
        Ok(records) => records,
        Err(error) => {
            warn!(error = %error, path = %path.display(), "chrome CLI session record unparsable — starting empty");
            Records::new()
        }
    }
}

/// The `AGENT_BROWSER_GROUP_CLEANUP_MS` value for a session: the lifetime, or `0` — the
/// value chrome-use reads as "no idle cleanup" — for a kept one.
pub(crate) fn backstop_ms(kept: bool) -> u64 {
    if kept { 0 } else { LIFETIME_MS }
}

/// The name a person reads for a session: the recorded key without the CLI's own namespace
/// prefix, which belongs to the record rather than to the name — a throwaway session's
/// generated name is shown the same way.
pub(crate) fn display_name(full: &str) -> &str {
    full.strip_prefix(crate::chrome::CLI_SESSION_PREFIX)
        .unwrap_or(full)
}

/// The one refusal line for a keep request on a session that already exists without the
/// mark (recorded, or live in chrome-use with no record).
pub(crate) fn mark_refusal(name: &str) -> String {
    let name = display_name(name);
    format!(
        "session {name} already exists without the keep mark — stop it with \
         `mahbot chrome session stop {name}` and start it again"
    )
}

/// The record's use transition: the lifetime restarts from `now`, and the next close gets a
/// fresh attempt.
fn record_use(record: &mut SessionRecord, now: u64) {
    record.last_used_ms = now;
    record.unconfirmed_closes = 0;
}

fn apply_use(
    records: &mut Records,
    name: &str,
    keep: bool,
    now: u64,
) -> Result<UseOutcome, String> {
    if let Some(record) = records.get_mut(name) {
        if keep && !record.kept {
            return Err(mark_refusal(name));
        }
        record_use(record, now);
        return Ok(UseOutcome {
            created: false,
            marked: false,
            kept: record.kept,
        });
    }
    records.insert(
        name.to_string(),
        SessionRecord {
            last_used_ms: now,
            kept: keep,
            unconfirmed_closes: 0,
            in_flight: Vec::new(),
        },
    );
    Ok(UseOutcome {
        created: true,
        marked: keep,
        kept: keep,
    })
}

fn due_for_close(records: &Records, now: u64) -> Option<String> {
    records
        .iter()
        .filter(|(_, record)| !record.kept)
        // A call using the session says so; the mark is the only reason needed, and a mark
        // whose call is gone is no reason at all.
        .filter(|(_, record)| !record.in_flight.iter().any(|mark| mark.is_live(now)))
        .filter(|(_, record)| now.saturating_sub(record.last_used_ms) >= LIFETIME_MS)
        .min_by_key(|(_, record)| record.last_used_ms)
        .map(|(name, _)| name.clone())
}

/// Refuse a keep request the record already rules out, without changing anything.
/// `Ok(true)` when the record already holds this session (the call is not the one that
/// creates it); only meaningful for a keep request.
pub(crate) fn check_keep(name: &str, keep: bool) -> Result<bool, String> {
    if !keep {
        return Ok(false);
    }
    let Some(ledger) = Ledger::open() else {
        return Ok(false);
    };
    match ledger.records.get(name) {
        Some(record) if !record.kept => Err(mark_refusal(name)),
        other => Ok(other.is_some()),
    }
}

/// Record a call's use of one session and apply its keep request: the lifetime's clock
/// restarts, this call adds its own in-flight mark ([`note_end`] removes it), and `marked`
/// says this call created the session with the mark.
pub(crate) fn note_use(name: &str, keep: bool) -> Result<UseOutcome, String> {
    let Some(mut ledger) = Ledger::open() else {
        return Ok(UseOutcome {
            created: false,
            marked: false,
            kept: keep,
        });
    };
    let now = crate::util::unix_millis();
    let outcome = apply_use(&mut ledger.records, name, keep, now)?;
    if let Some(record) = ledger.records.get_mut(name) {
        let pid = std::process::id();
        record.in_flight.retain(|mark| mark.pid != pid);
        record.in_flight.push(InFlight::new(now));
    }
    ledger.persist();
    Ok(outcome)
}

/// This call is done with the session: its own mark goes, and the lifetime's clock restarts
/// from here — a session is idle from the end of the call that used it, not from its start.
pub(crate) fn note_end(name: &str) {
    let Some(mut ledger) = Ledger::open() else {
        return;
    };
    let pid = std::process::id();
    let Some(record) = ledger.records.get_mut(name) else {
        return;
    };
    let marks = record.in_flight.len();
    record.in_flight.retain(|mark| mark.pid != pid);
    if record.in_flight.len() == marks {
        return;
    }
    record.last_used_ms = crate::util::unix_millis();
    ledger.persist();
}

/// `session status` counts as a later call with the same name: restart a recorded
/// session's lifetime and, from that one look, say whether it carries the mark. `None`
/// when the record does not hold the session — a diagnostic does not record one.
pub(crate) fn touch(name: &str) -> Option<bool> {
    let mut ledger = Ledger::open()?;
    let kept = {
        let record = ledger.records.get_mut(name)?;
        record_use(record, crate::util::unix_millis());
        record.kept
    };
    ledger.persist();
    Some(kept)
}

/// Every recorded session name (the start-up sweep leaves these alone).
pub(crate) fn recorded_names() -> BTreeSet<String> {
    Ledger::open().map_or_else(BTreeSet::new, |ledger| ledger.records.into_keys().collect())
}

/// Forget a session that was explicitly stopped or closed — the whole record, so it also
/// takes any in-flight mark a concurrent call put on it (a narrow loss of protection that
/// needs an overlap, an unready browser and the same name).
pub(crate) fn forget(name: &str) {
    let Some(mut ledger) = Ledger::open() else {
        return;
    };
    if ledger.records.remove(name).is_some() {
        ledger.persist();
    }
}

/// The recorded sessions, by name — the name a person reads ([`display_name`]), never the
/// record's own key.
pub(crate) fn list() -> Vec<SessionRow> {
    let now = crate::util::unix_millis();
    Ledger::open().map_or_else(Vec::new, |ledger| {
        ledger
            .records
            .into_iter()
            .map(|(name, record)| SessionRow {
                name: display_name(&name).to_string(),
                kept: record.kept,
                idle_seconds: now.saturating_sub(record.last_used_ms) / 1000,
            })
            .collect()
    })
}

/// Close the one session whose lifetime ran out, if any: the oldest non-kept record
/// at least [`LIFETIME_MS`] old. A close the helper does not confirm keeps the record for
/// the next pass, and drops it after [`CLOSE_ATTEMPT_LIMIT`] of them; a close that never
/// reached the helper was not asked at all, so the record simply waits for it to return.
pub(crate) async fn sweep_expired() {
    let Some(mut ledger) = Ledger::open() else {
        return;
    };
    let now = crate::util::unix_millis();
    // A mark left by a call that is gone (crashed, killed outright, or past its own
    // deadline) is no evidence of anything: dropping it keeps a dead pid out of the record.
    let mut changed = prune_stale_marks(&mut ledger.records, now);
    let Some(name) = due_for_close(&ledger.records, now) else {
        if changed {
            ledger.persist();
        }
        return;
    };
    match crate::tools::chrome_daemon::stop_session_by_name(&name).await {
        SessionRecovery::Stopped => {
            ledger.records.remove(&name);
            changed = true;
            info!(session = %name, "closed an idle chrome session");
        }
        SessionRecovery::NotStarted => {
            debug!(session = %name, "idle chrome session not closed: chrome-use could not be started");
        }
        SessionRecovery::Unanswered => {
            let given_up = ledger.records.get_mut(&name).is_some_and(|record| {
                record.unconfirmed_closes = record.unconfirmed_closes.saturating_add(1);
                record.unconfirmed_closes >= CLOSE_ATTEMPT_LIMIT
            });
            changed = true;
            if given_up {
                ledger.records.remove(&name);
                warn!(
                    session = %name,
                    "idle chrome session close not confirmed — dropping its record"
                );
            } else {
                debug!(
                    session = %name,
                    "idle chrome session close not confirmed — retried on the next sweep"
                );
            }
        }
    }
    if changed {
        ledger.persist();
    }
}

/// Drop every mark whose call is no longer there. `due_for_close` already ignores them, so
/// this changes no outcome — it only keeps a dead pid from riding the record until the
/// session is closed. `true` when something went.
fn prune_stale_marks(records: &mut Records, now: u64) -> bool {
    let mut pruned = false;
    for record in records.values_mut() {
        let marks = record.in_flight.len();
        record.in_flight.retain(|mark| mark.is_live(now));
        pruned |= record.in_flight.len() != marks;
    }
    pruned
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(last_used_ms: u64, kept: bool) -> SessionRecord {
        SessionRecord {
            last_used_ms,
            kept,
            unconfirmed_closes: 0,
            in_flight: Vec::new(),
        }
    }

    #[test]
    fn a_session_is_due_only_after_its_lifetime_and_never_when_kept() {
        let mut records = Records::new();
        records.insert("mahbot-chrome-young".to_string(), record(1_000, false));
        records.insert("mahbot-chrome-old".to_string(), record(0, false));
        records.insert("mahbot-chrome-kept".to_string(), record(0, true));
        assert_eq!(
            due_for_close(&records, LIFETIME_MS).as_deref(),
            Some("mahbot-chrome-old")
        );
        assert_eq!(due_for_close(&records, LIFETIME_MS - 1), None);
    }

    #[test]
    fn a_marked_session_is_never_due_while_its_call_runs() {
        let mut records = Records::new();
        let mut busy = record(0, false);
        // This process is the one running "the call", so the mark is live.
        busy.in_flight = vec![InFlight::new(0)];
        records.insert("mahbot-chrome-busy".to_string(), busy);
        assert_eq!(due_for_close(&records, LIFETIME_MS), None);

        // A mark past its own deadline is stale whatever the process table says.
        let mut expired = record(0, false);
        expired.in_flight = vec![InFlight {
            pid: std::process::id(),
            until_ms: 1,
        }];
        records.insert("mahbot-chrome-expired".to_string(), expired);
        assert_eq!(
            due_for_close(&records, LIFETIME_MS).as_deref(),
            Some("mahbot-chrome-expired")
        );
    }

    #[test]
    fn the_mark_is_set_only_by_the_call_that_creates_the_session() {
        let mut records = Records::new();
        let created = apply_use(&mut records, "mahbot-chrome-a", true, 10).unwrap();
        assert!(created.created && created.marked && created.kept);
        let later = apply_use(&mut records, "mahbot-chrome-a", false, 20).unwrap();
        assert!(!later.marked && later.kept);
        assert_eq!(records["mahbot-chrome-a"].last_used_ms, 20);

        let plain = apply_use(&mut records, "mahbot-chrome-b", false, 30).unwrap();
        assert!(!plain.marked && !plain.kept);
        assert!(apply_use(&mut records, "mahbot-chrome-b", true, 40).is_err());
        assert!(apply_use(&mut records, "mahbot-chrome-a", true, 50).is_ok());
    }
}
