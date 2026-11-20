//! The session-scoped Chrome profile pin.
//!
//! A chrome-use session is bound to ONE Chrome profile, and the helper picks it by its own rule
//! when the caller names none. With several profiles on a machine that rule silently moves a run
//! between browsers, so this module lets the caller NAME a profile and keeps every later call of
//! that session on it.
//!
//! ## The pin
//!
//! [`prepare`] records the resolved profile (display name + directory) in a JSON record at the
//! storage root (`chrome-session-profiles.json`), keyed by the PHYSICAL session name, and every
//! chrome-use call for that session carries the directory as its `--browser`. A recycled daemon, a
//! later `mahbot chrome` invocation and a product restart therefore resolve the same pin. It is
//! dropped when the session ends ([`forget`]) and is never a user setting. A pin that cannot be
//! recorded is an environment refusal: a call dispatched without it would land in the helper's own
//! profile choice.
//!
//! ## The three refusals
//!
//! A name matching nothing or several is a bad argument ([`OutKind::Usage`]) listing the profiles
//! as display name and directory only — never e-mail, id or connected state — which is also how
//! names are discovered. A profile that cannot be brought up is an environment condition
//! ([`OutKind::Environment`]). A tab already working in another profile is a conflict
//! ([`OutKind::Conflict`]): decided before dispatch from the helper's own report and re-checked
//! after the step, so a helper that quietly bound elsewhere can never pass as success.
//!
//! ## Resolution and bring-up
//!
//! Names resolve the way chrome-use resolves `--browser` — exact display name, directory, e-mail
//! or id, then a unique prefix of the display name — and the resolved DIRECTORY is what is passed
//! on. A named profile that is not running is brought up as a pre-step with its own bound
//! ([`BRING_UP_BUDGET`]); only the caller's own call may let the extension open an unfocused window
//! (`AGENT_BROWSER_OPEN_PROFILE_WINDOW`), never an internal one. A missing or disabled extension is
//! refused before either path. Focus is taken only when the admin's Assistant deliberately raises a
//! tab for the owner (the `show_tab` tool); chrome-use's own profile-window bring-up briefly
//! foregrounding Chrome is the helper's behaviour, not the product's call.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chrome::contract::{OutKind, Reader};
use crate::chrome::lock_json::LockedJson;
use crate::chrome::spawn::CliRun;

/// The oldest chrome-use that carries BOTH the `--browser` profile pin and the unfocused-window
/// bring-up. `--open-profile-window` / `AGENT_BROWSER_OPEN_PROFILE_WINDOW` landed in v1.5.287: a
/// profile that is connected but has no window otherwise fails `profile_not_open`, so a lower floor
/// would let the bring-up silently not happen. Below it a profile call fails loudly instead of
/// silently driving the helper's default profile.
const MIN_HELPER_VERSION: (u64, u64, u64) = (1, 5, 287);

/// Bound on the daemon-free `browsers --json` reads (listing, the session's live binding, the
/// post-step serving report). It is a product probe, not agent work.
const LIST_BOUND: Duration = Duration::from_secs(10);

/// The bring-up pre-step's own bound. It is NOT taken out of the action's declared budget: the
/// action keeps its normal clock once the profile is up, and the whole call stays bounded by this
/// plus that clock.
const BRING_UP_BUDGET: Duration = Duration::from_secs(45);

/// The relay wait declared to `chrome-use connect`, inside [`BRING_UP_BUDGET`] so the helper gives
/// up before the product's kill.
const BRING_UP_WAIT_SECS: u64 = 40;

const PINS_FILE_NAME: &str = "chrome-session-profiles.json";
const LOCK_FILE_NAME: &str = "chrome-session-profiles.lock";

/// One Chrome profile as the helper reports it, reduced to what this module needs. The e-mail and
/// id are read only for NAME matching; they are never printed.
#[expect(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct Row {
    name: String,
    dir: String,
    email: Option<String>,
    id: Option<String>,
    has_extension: bool,
    extension_disabled: bool,
    connected: bool,
    /// Whether this is the profile the session is bound to right now.
    session: bool,
}

/// A resolved Chrome profile: the name a person reads and the directory the helper selects by.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Profile {
    pub(crate) display: String,
    pub(crate) dir: String,
}

/// How a refusal names a profile: its display name and directory, never its e-mail or id. A
/// profile whose display name is its directory is named once.
fn describe_pair(name: &str, dir: &str) -> String {
    if name == dir {
        format!("\"{dir}\"")
    } else {
        format!("\"{name}\" ({dir})")
    }
}

impl Profile {
    fn from_row(row: &Row) -> Self {
        Self {
            display: row.name.clone(),
            dir: row.dir.clone(),
        }
    }

    /// This profile as a refusal names it.
    fn describe(&self) -> String {
        describe_pair(&self.display, &self.dir)
    }
}

/// A profile refusal: the distinct condition it is, plus the text the caller reads. The text is
/// always written in this product's terms — chrome-use's own advice (`connect --browser …`, `start a
/// new session with --session …`) is never relayed. The fields are private so every refusal is one
/// of the three constructors below.
#[derive(Debug)]
pub(crate) struct Refusal {
    kind: OutKind,
    message: String,
    /// The profile a CONFLICT is actually in, for the envelope's structured fields. `None` for the
    /// other two conditions.
    profile: Option<Profile>,
}

impl Refusal {
    fn usage(message: String) -> Self {
        Self {
            kind: OutKind::Usage,
            message,
            profile: None,
        }
    }

    fn environment(message: String) -> Self {
        Self {
            kind: OutKind::Environment,
            message,
            profile: None,
        }
    }

    fn conflict(message: String, actual: Profile) -> Self {
        Self {
            kind: OutKind::Conflict,
            message,
            profile: Some(actual),
        }
    }

    pub(crate) fn kind(&self) -> OutKind {
        self.kind
    }

    pub(crate) fn message(&self) -> &str {
        &self.message
    }

    pub(crate) fn profile(&self) -> Option<&Profile> {
        self.profile.as_ref()
    }
}

/// How long a pin whose session the CLI record does not know is still trusted. Every call that uses
/// a session refreshes its pin, so a pin idle this long belongs to a session that died with its
/// process; without the bound one abandoned pin pair would disable the session-less launch for the
/// life of the product.
const PIN_MAX_IDLE_MS: u64 = 2 * 60 * 60 * 1000;

/// What a session-less launch path may do with the pins that exist: nothing when there are none,
/// the one profile when every pinned session wants the same one, and nothing at all when they
/// disagree — a named session brings its own browser up, so a global launch must not guess.
pub(crate) enum LaunchTarget {
    None,
    Single(Profile),
    Mixed,
}

/// The distinct profiles the live sessions are pinned to, for the session-less launch path.
/// `owned` is the caller's liveness rule for a pin whose session the CLI record does not know (the
/// watchdog's live agent runs); the launch path supplies it, so this module needs no knowledge of
/// how a run names its sessions.
#[must_use]
pub(crate) fn launch_target(owned: impl Fn(&str) -> bool) -> LaunchTarget {
    // The CLI record supplies ownership for the pins it knows. Its lock is released before the pin
    // lock is taken, so the two are never nested.
    let recorded = crate::chrome::cli_sessions::recorded_names();
    let Some(mut store) = LockedJson::<Pins>::open_existing(PINS_FILE_NAME, LOCK_FILE_NAME) else {
        return LaunchTarget::None;
    };
    // A record that cannot be read says nothing about which profiles the live sessions use, so the
    // session-less path launches nothing rather than guessing.
    if store.broken {
        return LaunchTarget::None;
    }
    let (target, pruned) = launch_target_of(&mut store.value, |session| {
        recorded.contains(session) || owned(session)
    });
    if pruned {
        store.persist();
    }
    target
}

/// The launch decision over the pins that exist, pruning only the ones `owned` does not claim and
/// whose last use is older than [`PIN_MAX_IDLE_MS`]. Pure, so the Single/Mixed branch — the one a
/// stale pin pair once broke — is testable without a store. `true` in the second slot means the
/// pins changed and the store must be persisted.
fn launch_target_of(pins: &mut Pins, owned: impl Fn(&str) -> bool) -> (LaunchTarget, bool) {
    let now = crate::util::unix_millis();
    let before = pins.len();
    pins.retain(|session, pin| owned(session) || now.saturating_sub(pin.at_ms) < PIN_MAX_IDLE_MS);
    let pruned = pins.len() != before;
    let mut by_dir: BTreeMap<String, Profile> = BTreeMap::new();
    for pin in pins.values() {
        by_dir
            .entry(pin.profile.dir.clone())
            .or_insert_with(|| pin.profile.clone());
    }
    let mut it = by_dir.into_values();
    let target = match (it.next(), it.next()) {
        (None, _) => LaunchTarget::None,
        (Some(profile), None) => LaunchTarget::Single(profile),
        (Some(_), Some(_)) => LaunchTarget::Mixed,
    };
    (target, pruned)
}

/// One agent-run session per distinct pinned profile, plus `fallback` for the unpinned side: the
/// sessions a profile-blind sweep must run one pass in so every profile's tabs are reached.
/// `is_door` is the caller's rule for a session the sweep may borrow — a command-line session is not
/// the sweep's to start.
#[must_use]
pub(crate) fn sweep_sessions(fallback: &str, is_door: impl Fn(&str) -> bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut fallback_dir = None;
    if let Ok(Some(pins)) = read_pins() {
        for (session, pin) in &pins {
            if is_door(session) && seen.insert(pin.profile.dir.clone()) {
                out.push(session.clone());
            }
        }
        fallback_dir = pins.get(fallback).map(|pin| pin.profile.dir.clone());
    }
    if fallback_dir.is_none_or(|dir| !seen.contains(&dir)) {
        out.push(fallback.to_string());
    }
    out
}

/// Resolve and validate the profile this call must use, bringing it up when it is not running.
/// `requested` is the name the caller gave THIS call; with none, the session's `stored` pin is used
/// (a session that has never been given one stays profile-blind and returns `Ok(None)`). `subject`
/// is the name the caller's frontend shows for the session/tab, and `reader` words a conflict for
/// that reader (a tool `tab` or a command-line `session`).
///
/// Order matters: the name is resolved and refused, a conflict is decided, and the extension state
/// is checked — all before the profile is brought up, so a refused call opens nothing and neither
/// creates nor evicts a tab. The name is resolved before the helper's version floor and before a
/// damaged pin record, so discovery by refusal works on every helper.
pub(crate) async fn prepare(
    cli: &Path,
    session: &str,
    subject: &str,
    requested: Option<&str>,
    stored: PinState,
    reader: Reader,
) -> Result<Option<Profile>, Refusal> {
    let pinned = stored.profile().cloned();
    // A session that was never given a profile — by this call or an earlier one — runs profile-blind,
    // exactly as before profiles existed: no listing, no bring-up, nothing to refuse. A record that
    // exists but cannot be read is not a profile the session was given, so it does not refuse this
    // call either; only a call that named one has a binding to lose.
    if requested.is_none() && pinned.is_none() {
        return Ok(None);
    }

    let rows = list(cli, Some(session), LIST_BOUND)
        .await
        .map_err(Refusal::environment)?;
    let target = match (requested, pinned.clone()) {
        (Some(name), _) => resolve(name, &rows)?,
        (None, Some(pinned)) => pinned,
        // The blind case returned above.
        (None, None) => return Ok(None),
    };
    // The name resolved, but the record that would show the session is not pinned elsewhere cannot
    // be read: refused rather than left to the helper's own profile choice.
    if matches!(stored, PinState::Unreadable) {
        return Err(unreadable_pins());
    }

    // A helper too old to honour the parameter would silently drive its default profile.
    let min = semver::Version::new(
        MIN_HELPER_VERSION.0,
        MIN_HELPER_VERSION.1,
        MIN_HELPER_VERSION.2,
    );
    match crate::tools::chrome_daemon::cli_version().await {
        Some(version) if version >= min => {}
        Some(version) => {
            return Err(Refusal::environment(format!(
                "chrome-use {version} is too old to select a Chrome profile; that needs {min} or \
                 newer. The product updates its own copy on start — retry after it does."
            )));
        }
        None => {
            return Err(Refusal::environment(
                "could not read chrome-use's version, so a Chrome profile cannot be selected"
                    .to_string(),
            ));
        }
    }

    // The session's live binding is the helper's own report; with no live binding the recorded pin
    // stands for it. Either way a different profile is a conflict, never a silent switch — decided
    // before the target profile's own state is even considered.
    let bound = rows
        .iter()
        .find(|row| row.session)
        .map(Profile::from_row)
        .or_else(|| pinned.clone());
    if let Some(bound) = bound
        && bound.dir != target.dir
    {
        let message = conflict(subject, &bound, &target, reader);
        return Err(Refusal::conflict(message, bound));
    }

    let Some(row) = rows.iter().find(|row| row.dir == target.dir) else {
        return Err(Refusal::environment(format!(
            "profile {} no longer exists on this machine. Available profiles: {}.",
            target.describe(),
            listing(&rows.iter().collect::<Vec<_>>())
        )));
    };
    if !row.has_extension {
        return Err(Refusal::environment(format!(
            "profile {} has no chrome-use extension installed, so it cannot be driven. Install the \
             extension in that profile first.",
            target.describe()
        )));
    }
    if row.extension_disabled {
        return Err(Refusal::environment(format!(
            "profile {} has the chrome-use extension disabled. Enable it in that profile first.",
            target.describe()
        )));
    }

    if !row.connected {
        bring_up(cli, &target).await.map_err(|reason| {
            Refusal::environment(format!(
                "profile {} could not be brought up: {reason}.",
                target.describe()
            ))
        })?;
        // `connect` returns once the relay answered; the list is the helper's own confirmation
        // that the profile is really reachable before the action is dispatched.
        let confirmed = list(cli, Some(session), LIST_BOUND)
            .await
            .map_err(Refusal::environment)?;
        if !confirmed
            .iter()
            .any(|row| row.dir == target.dir && row.connected)
        {
            return Err(Refusal::environment(format!(
                "profile {} did not come up: chrome-use reported no relay for it.",
                target.describe()
            )));
        }
    }

    if !pin(session, &target) {
        return Err(Refusal::environment(format!(
            "profile {} could not be recorded for this session, so the call would risk landing in \
             another profile. Retry.",
            target.describe()
        )));
    }
    Ok(Some(target))
}

/// The profile that actually served the session, from the helper's own report. `None` when the
/// helper cannot answer or the session is not bound (yet) — the caller then claims nothing. `bound`
/// is the caller's own: one sharing a deadline across a whole action must not let this read spend
/// the helper's whole list bound.
pub(crate) async fn of_session(cli: &Path, session: &str, bound: Duration) -> Option<Profile> {
    let rows = list(cli, Some(session), bound).await.ok()?;
    rows.iter().find(|row| row.session).map(Profile::from_row)
}

/// The refusal for a session or tab that is working in another profile than the one this call
/// asked for.
fn conflict(subject: &str, actual: &Profile, named: &Profile, reader: Reader) -> String {
    crate::chrome::contract::profile_conflict_message(
        reader,
        Some(subject),
        &actual.describe(),
        Some(&named.describe()),
    )
}

/// After a step: the profile the helper reports for `session`, checked against `named`. `Ok(Some)`
/// is the profile that served the call (to surface), `Ok(None)` when the helper cannot report, and
/// `Err` the conflict refusal when it served another profile. One check for both frontends, so a
/// helper that quietly bound elsewhere can never pass as success.
pub(crate) async fn served(
    cli: &Path,
    session: &str,
    subject: &str,
    named: &Profile,
    reader: Reader,
) -> Result<Option<Profile>, Refusal> {
    let Some(actual) = of_session(cli, session, LIST_BOUND).await else {
        return Ok(None);
    };
    if actual.dir == named.dir {
        return Ok(Some(actual));
    }
    let message = conflict(subject, &actual, named, reader);
    Err(Refusal::conflict(message, actual))
}

/// Whether a chrome-use verb works IN the session's browser, and so must carry its profile pin.
/// The session-less diagnostics (`session`, `browsers`, `status`, `daemon`, `connect`, `doctor`,
/// install/update verbs) are answered without resolving the session's browser; passing them a
/// profile adds a resolution that could refuse a stop for no gain. `extension` calls DO carry it:
/// they list, group and close the session's tabs, and without the pin they would read whichever
/// profile the helper picked.
fn verb_carries_profile(args: &[&str]) -> bool {
    !matches!(
        args.first().copied(),
        Some(
            "session"
                | "browsers"
                | "status"
                | "daemon"
                | "connect"
                | "doctor"
                | "install"
                | "update"
        )
    )
}

/// The `--browser` directory a session-scoped spawn must carry, if any: the session's pin for a verb
/// that works in its browser. One recipe for every spawn funnel, so a call can never re-bind the
/// session to another profile behind the caller's back. A record that cannot be read yields none:
/// the caller-facing call already refused in [`prepare`], and a maintenance call closing tabs in the
/// helper's own profile is the pre-existing behaviour, not a new one.
#[must_use]
pub(crate) fn browser_arg(args: &[&str], session: Option<&str>) -> Option<String> {
    let session = session.filter(|_| verb_carries_profile(args))?;
    let pins = read_pins().ok().flatten()?;
    pins.get(session).map(|pin| pin.profile.dir.clone())
}

/// One recorded pin: the profile a session was given and when it was last confirmed. The timestamp
/// is what lets the session-less launch target drop a pin whose session died with its process.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Pin {
    profile: Profile,
    #[serde(default)]
    at_ms: u64,
}

type Pins = BTreeMap<String, Pin>;

/// The recorded pins as they are on disk, read without the lock. `Err(())` when a record exists but
/// cannot be read: a caller whose correctness depends on it refuses, a maintenance caller reads it
/// as none.
fn read_pins() -> Result<Option<Pins>, ()> {
    LockedJson::<Pins>::read_existing(PINS_FILE_NAME)
}

/// What one read of the pin store says about a session. The record itself is a file that can be
/// damaged, and the two frontends must agree on what that means, so the read says which of the three
/// it was instead of collapsing "damaged" into "no pin".
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PinState {
    /// The session was never given a profile.
    None,
    /// The session is pinned to this profile.
    Pinned(Profile),
    /// A record exists but cannot be read, so the session's binding is unknown.
    Unreadable,
}

impl PinState {
    pub(crate) fn profile(&self) -> Option<&Profile> {
        match self {
            Self::Pinned(profile) => Some(profile),
            Self::None | Self::Unreadable => None,
        }
    }
}

/// The recorded pin of `session`, read once by the caller and handed to [`prepare`]. A damaged
/// record says the session's binding is unknown rather than that there is none — [`prepare`] refuses
/// it for a call that named a profile, and stays profile-blind for one that did not.
pub(crate) fn pinned(session: &str) -> PinState {
    match read_pins() {
        Ok(pins) => pins
            .and_then(|pins| pins.get(session).map(|pin| pin.profile.clone()))
            .map_or(PinState::None, PinState::Pinned),
        Err(()) => PinState::Unreadable,
    }
}

/// Every recorded pin, read in one store pass — the snapshot a caller that groups many names by
/// profile needs, instead of one read per name.
#[must_use]
pub(crate) fn pins() -> BTreeMap<String, Profile> {
    read_pins()
        .ok()
        .flatten()
        .map_or_else(BTreeMap::new, |pins| {
            pins.into_iter()
                .map(|(session, pin)| (session, pin.profile))
                .collect()
        })
}

/// Record `profile` as the profile of `session`, refreshing the moment it was last confirmed.
/// `false` when the record could not be written — the caller must refuse rather than dispatch a
/// call that would carry no profile.
fn pin(session: &str, profile: &Profile) -> bool {
    let Some(mut store) = LockedJson::<Pins>::open(PINS_FILE_NAME, LOCK_FILE_NAME) else {
        return false;
    };
    // Writing over a record that could not be read would drop every other session's pin.
    if store.broken {
        return false;
    }
    let at_ms = crate::util::unix_millis();
    store.value.insert(
        session.to_string(),
        Pin {
            profile: profile.clone(),
            at_ms,
        },
    );
    store.persist()
}

/// Drop the pin of a session that ended.
pub(crate) fn forget(session: &str) {
    forget_all(std::iter::once(session));
}

/// Drop the pins of several sessions that ended, in one store pass.
pub(crate) fn forget_all<'a>(sessions: impl IntoIterator<Item = &'a str>) {
    let Some(mut store) = LockedJson::<Pins>::open_existing(PINS_FILE_NAME, LOCK_FILE_NAME) else {
        return;
    };
    // A record that cannot be read is left for the owner rather than emptied.
    if store.broken {
        return;
    }
    let mut changed = false;
    for session in sessions {
        changed |= store.value.remove(session).is_some();
    }
    if changed {
        store.persist();
    }
}

/// The refusal for a pin record that exists but cannot be read: the session's profile is unknown,
/// and a call dispatched without it would land in the helper's own profile choice.
fn unreadable_pins() -> Refusal {
    Refusal::environment(format!(
        "the record of which Chrome profile each session is pinned to ({PINS_FILE_NAME}) cannot be \
         read. Repair or remove it in the product's storage root, then retry."
    ))
}

/// Read every Chrome profile the helper knows, optionally asking which one `session` is bound to,
/// each helper call bounded by `bound`.
async fn list(cli: &Path, session: Option<&str>, bound: Duration) -> Result<Vec<Row>, String> {
    let value = crate::tools::chrome_daemon::run_cli_json_at(cli, &["browsers"], session, bound)
        .await
        .map_err(|verdict| {
            // The helper's own error text can name profile accounts; a product-terms reason is
            // all a refusal needs, and the other legs' texts are factual.
            match verdict {
                crate::tools::chrome_daemon::NoVerdict::Reported(_) => {
                    "could not list Chrome profiles: the helper refused the read".to_string()
                }
                other => format!("could not list Chrome profiles: {}", other.text()),
            }
        })?;
    parse_rows(&value).ok_or_else(|| "chrome-use returned no profile list".to_string())
}

fn parse_rows(value: &Value) -> Option<Vec<Row>> {
    let list = crate::tools::chrome_daemon::real_browser_list(value)?;
    Some(list.iter().filter_map(Row::from_value).collect())
}

impl Row {
    fn from_value(value: &Value) -> Option<Self> {
        let dir = value.get("dir").and_then(Value::as_str)?.trim().to_string();
        if dir.is_empty() {
            return None;
        }
        let text = |key: &str| {
            value
                .get(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        let name = text("name").unwrap_or_else(|| dir.clone());
        Some(Self {
            name,
            dir,
            email: text("email"),
            id: text("id"),
            // Only the helper's own word counts: a row that omits the key leaves the extension
            // UNKNOWN, and unknown ends in the plain no-extension refusal rather than in a guess
            // that could open the extension store.
            has_extension: value.get("hasExtension").and_then(Value::as_bool) == Some(true),
            extension_disabled: value
                .get("extensionDisabled")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            connected: value
                .get("connected")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            session: value
                .get("session")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }
}

/// Resolve `name` the way the helper resolves `--browser`: exact display name, directory, e-mail or
/// id, then a unique prefix of the display name. A name matching nothing or several is a usage
/// refusal that lists the profiles (display name and directory only).
fn resolve(name: &str, rows: &[Row]) -> Result<Profile, Refusal> {
    let want = name.trim();
    if want.is_empty() {
        return Err(unknown(name, rows));
    }
    let lower = want.to_lowercase();
    let matches = |field: fn(&Row) -> Option<&str>| -> Vec<&Row> {
        rows.iter()
            .filter(|row| field(row).is_some_and(|value| value.to_lowercase() == lower))
            .collect()
    };
    for field in [
        (|row: &Row| Some(row.name.as_str())) as fn(&Row) -> Option<&str>,
        |row: &Row| Some(row.dir.as_str()),
        |row: &Row| row.email.as_deref(),
        |row: &Row| row.id.as_deref(),
    ] {
        match matches(field).as_slice() {
            [] => {}
            [one] => return Ok(Profile::from_row(one)),
            several => return Err(ambiguous(name, several)),
        }
    }
    let prefixed: Vec<&Row> = rows
        .iter()
        .filter(|row| row.name.to_lowercase().starts_with(&lower))
        .collect();
    match prefixed.as_slice() {
        [] => Err(unknown(name, rows)),
        [one] => Ok(Profile::from_row(one)),
        several => Err(ambiguous(name, several)),
    }
}

fn unknown(name: &str, rows: &[Row]) -> Refusal {
    let all: Vec<&Row> = rows.iter().collect();
    Refusal::usage(format!(
        "invalid profile \"{name}\": no Chrome profile matches. Available profiles: {}.",
        listing(&all)
    ))
}

fn ambiguous(name: &str, matches: &[&Row]) -> Refusal {
    Refusal::usage(format!(
        "invalid profile \"{name}\": it matches several Chrome profiles: {}. Name the exact \
         display name or directory.",
        listing(matches)
    ))
}

/// The profile list as a refusal prints it: display name and directory only, never an e-mail
/// address, id or connected state.
fn listing(rows: &[&Row]) -> String {
    if rows.is_empty() {
        return "none".to_string();
    }
    rows.iter()
        .map(|row| describe_pair(&row.name, &row.dir))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Bring an unconnected profile up: Chrome starts in it when it is not running, and a running Chrome
/// opens an unfocused window in it. The extension's absence or disablement was refused before this,
/// so the helper never opens the Web Store or the extensions page.
async fn bring_up(cli: &Path, profile: &Profile) -> Result<(), String> {
    let wait = BRING_UP_WAIT_SECS.to_string();
    let args = [
        "connect",
        "--browser",
        profile.dir.as_str(),
        "--wait",
        wait.as_str(),
    ];
    match crate::tools::chrome_daemon::run_cli_bounded_at(cli, &args, None, BRING_UP_BUDGET).await {
        CliRun::Output(out) if out.status.success() => Ok(()),
        CliRun::Output(out) => Err(format!("chrome-use connect exited with {}", out.status)),
        CliRun::SpawnFailure => Err("chrome-use could not be started".to_string()),
        CliRun::TimedOut => Err(format!(
            "it did not come up within {}s",
            BRING_UP_BUDGET.as_secs()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rows() -> Vec<Row> {
        parse_rows(&json!({"data": {"browsers": [
            {"name": "Work", "dir": "Default", "email": "work@example.com", "id": "id-work",
             "hasExtension": true, "connected": true, "session": false},
            {"name": "Your Chrome", "dir": "Profile 8", "email": "home@example.com",
             "id": "id-home", "hasExtension": true, "connected": true, "session": false},
            {"name": "Work Extra", "dir": "Profile 2", "email": null, "id": null,
             "hasExtension": false, "extensionDisabled": false, "connected": false, "session": false},
        ]}}))
        .expect("rows parse")
    }

    #[test]
    fn a_name_resolves_by_display_dir_email_id_and_unique_prefix() {
        for (name, dir) in [
            ("Work", "Default"),
            ("Default", "Default"),
            ("work@example.com", "Default"),
            ("id-home", "Profile 8"),
            ("Your", "Profile 8"),
        ] {
            assert_eq!(resolve(name, &rows()).unwrap().dir, dir, "name {name}");
        }
    }

    #[test]
    fn an_exact_name_wins_over_a_prefix() {
        // "Work" is both an exact display name and the prefix of "Work Extra".
        assert_eq!(resolve("Work", &rows()).unwrap().dir, "Default");
    }

    #[test]
    fn an_unknown_name_lists_every_profile_without_addresses() {
        let refusal = resolve("nope", &rows()).unwrap_err();
        assert_eq!(refusal.kind(), OutKind::Usage);
        assert!(
            refusal.message().contains("\"Work\" (Default)"),
            "{}",
            refusal.message()
        );
        assert!(
            refusal.message().contains("\"Your Chrome\" (Profile 8)"),
            "{}",
            refusal.message()
        );
        assert!(
            !refusal.message().contains("work@example.com"),
            "{}",
            refusal.message()
        );
        assert!(
            !refusal.message().contains("id-work"),
            "{}",
            refusal.message()
        );
    }

    #[test]
    fn an_ambiguous_prefix_lists_only_the_matches_without_addresses() {
        let refusal = resolve("Wor", &rows()).unwrap_err();
        assert_eq!(refusal.kind(), OutKind::Usage);
        assert!(
            refusal.message().contains("\"Work\" (Default)"),
            "{}",
            refusal.message()
        );
        assert!(
            refusal.message().contains("\"Work Extra\" (Profile 2)"),
            "{}",
            refusal.message()
        );
        assert!(
            !refusal.message().contains("Your Chrome"),
            "{}",
            refusal.message()
        );
        assert!(
            !refusal.message().contains("home@example.com"),
            "{}",
            refusal.message()
        );
    }

    #[test]
    fn page_and_group_verbs_carry_the_profile() {
        for verb in [
            "open",
            "click",
            "tab",
            "close",
            "get",
            "eval",
            "screenshot",
            // Listing, grouping and closing a session's tabs must read its own profile.
            "extension",
        ] {
            assert!(verb_carries_profile(&[verb]), "{verb} works in the browser");
        }
        for verb in [
            "session", "browsers", "status", "daemon", "connect", "doctor",
        ] {
            assert!(
                !verb_carries_profile(&[verb]),
                "{verb} is a session-less diagnostic"
            );
        }
    }

    #[test]
    fn browser_arg_follows_the_session_pin_and_the_verb() {
        // No pin: nothing to carry, whatever the verb.
        assert_eq!(browser_arg(&["open"], Some("agent-tab-no-pin")), None);
        let profile = Profile {
            display: "Your Chrome".to_string(),
            dir: "Profile 8".to_string(),
        };
        pin("agent-tab-browser-arg", &profile);
        assert_eq!(
            browser_arg(&["open"], Some("agent-tab-browser-arg")).as_deref(),
            Some("Profile 8")
        );
        assert_eq!(
            browser_arg(&["status"], Some("agent-tab-browser-arg")),
            None
        );
        forget("agent-tab-browser-arg");
    }

    #[test]
    fn a_conflict_names_the_actual_profile_and_never_an_address() {
        let actual = Profile {
            display: "Your Chrome".to_string(),
            dir: "Profile 8".to_string(),
        };
        let named = Profile {
            display: "Work".to_string(),
            dir: "Default".to_string(),
        };
        let message = conflict("work", &actual, &named, Reader::Cli);
        assert!(message.contains("\"Your Chrome\" (Profile 8)"), "{message}");
        assert!(message.contains("\"Work\" (Default)"), "{message}");
        assert!(message.contains("new session"), "{message}");
        assert!(!message.contains('@'), "{message}");
        let tool = conflict("work", &actual, &named, Reader::Tool);
        assert!(tool.contains("new tab"), "{tool}");
        assert!(!tool.contains("session"), "{tool}");
    }

    fn pin_at(pins: &mut Pins, session: &str, dir: &str, at_ms: u64) {
        pins.insert(
            session.to_string(),
            Pin {
                profile: Profile {
                    display: dir.to_string(),
                    dir: dir.to_string(),
                },
                at_ms,
            },
        );
    }

    #[test]
    fn the_launch_decision_is_one_profile_or_none_when_they_disagree() {
        let mut pins = Pins::new();
        assert!(matches!(
            launch_target_of(&mut pins, |_| true),
            (LaunchTarget::None, false)
        ));
        let now = crate::util::unix_millis();
        pin_at(&mut pins, "a", "Default", now);
        pin_at(&mut pins, "b", "Default", now);
        let (target, pruned) = launch_target_of(&mut pins, |_| true);
        assert!(!pruned);
        assert!(
            matches!(target, LaunchTarget::Single(profile) if profile.dir == "Default"),
            "two sessions on one profile launch that profile"
        );
        pin_at(&mut pins, "c", "Profile 8", now);
        assert!(matches!(
            launch_target_of(&mut pins, |_| true),
            (LaunchTarget::Mixed, false)
        ));
    }

    #[test]
    fn an_abandoned_pin_is_pruned_and_an_owned_one_is_kept() {
        let mut pins = Pins::new();
        // Both are older than the idle bound; only the owned one survives, so the stale pair that
        // once disabled the session-less launch becomes a single-profile launch.
        pin_at(&mut pins, "gone", "Profile 8", 0);
        pin_at(&mut pins, "live", "Default", 0);
        let (target, pruned) = launch_target_of(&mut pins, |session| session == "live");
        assert!(pruned);
        assert!(!pins.contains_key("gone"));
        assert!(pins.contains_key("live"));
        assert!(
            matches!(target, LaunchTarget::Single(profile) if profile.dir == "Default"),
            "the abandoned pin no longer forces Mixed"
        );
    }

    #[test]
    fn the_store_round_trips_and_forgets_a_session() {
        let name = "agent-tab-test-profile-store";
        let profile = Profile {
            display: "Your Chrome".to_string(),
            dir: "Profile 8".to_string(),
        };
        pin(name, &profile);
        assert_eq!(pinned(name), PinState::Pinned(profile));
        forget(name);
        assert_eq!(pinned(name), PinState::None);
    }
}
