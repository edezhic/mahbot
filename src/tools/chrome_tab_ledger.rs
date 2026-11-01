//! Which browser tabs one agent run has open, in what order the run last addressed them, and which
//! of them the tab cap closed.
//!
//! The `chrome` tool keeps at most [`MAX_AGENT_TABS`] tabs per run, one logical `tab` name each,
//! and the cap is enforced from this record rather than from the browser, because the browser is
//! not always reachable: an unreached call must not create a tab, and a tab the cap closed must
//! stay refused until a call that really reached the browser mints the page under that name again.
//!
//! The module is the record alone — pure state plus one durable JSON file, no browser call. A
//! caller feeds it what it learned (`reached`, the close's own outcome) and reads back what to do
//! ([`note_addressed`] picks a victim, [`take_eviction`] re-checks that pick before the close,
//! [`settle_eviction`] records what the close came to); `crate::tools::chrome_release` drives the
//! boot prune and the run-release deletion.
//!
//! One entry per tab namespace (`agent-tab-<12 hex>-`, minted by
//! `chrome::run_session_namespace`; the key is opaque here). Each carries the addressed tabs with
//! the recency sequence of the call that last addressed them — so the least-recently
//! ADDRESSED tab, not the oldest created, is the one the cap closes — and whether the cap
//! closed it.
//!
//! Only a call the browser answered for moves the record: an appearance for an unknown name, an
//! address (recency) for an open one, a revival for a closed one. A call that left no page moves
//! nothing, however the caller reports it.
//!
//! The record is written after every mutation (compact JSON, tmp+rename), so a run resumed after a
//! self-update or restart finds its tabs. Failure is fail-open throughout: no storage root means
//! in-memory only, and an unreadable or unparsable file counts as empty rather than fatal.

use crate::util::UnwrapPoison;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use tracing::warn;

pub(crate) const MAX_AGENT_TABS: usize = 5;

/// At the storage root. A file rather than a DB row because it must survive a restart and be
/// written incrementally; it is small and written whole on every mutation.
const TAB_LEDGER_FILE_NAME: &str = "chrome-agent-tabs.json";

#[derive(Clone, Default, Serialize, Deserialize)]
struct NamespaceRecord {
    /// Recency is the order the agent addressed tabs.
    #[serde(default)]
    seq: u64,
    #[serde(default)]
    tabs: Vec<TabEntry>,
}

impl NamespaceRecord {
    /// A larger value per addressed call, so the smallest open value is the least recently
    /// addressed tab.
    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct TabEntry {
    name: String,
    /// Of the call that last addressed this tab.
    #[serde(default)]
    seq: u64,
    /// Addressing a closed tab is an error until a reached call revives it.
    #[serde(default)]
    closed: bool,
    /// The cap picked it but the browser has not confirmed it gone. Never serialized — after a
    /// restart an unconfirmed eviction is only a tab that may still be open.
    #[serde(skip)]
    evicting: bool,
}

/// Ordered by namespace so the file's bytes are stable.
type Namespaces = BTreeMap<String, NamespaceRecord>;

#[derive(Default)]
struct LedgerState {
    /// The path the record was loaded from (`None` = in-memory only). A newly resolved store
    /// — the configured path changing, or a test seam installing its own dir — forces a
    /// reload, so one store's tabs are never carried under another's.
    loaded_path: Option<PathBuf>,
    namespaces: Namespaces,
}

impl LedgerState {
    fn ensure_loaded(&mut self) {
        let path = store_path();
        if self.loaded_path == path {
            return;
        }
        self.namespaces = path.as_deref().map_or_else(Namespaces::new, read_ledger);
        self.loaded_path = path;
    }

    /// Fail-open: an in-memory-only store or a failed write leaves whatever was there, and a
    /// serialization failure never publishes an empty payload over a live file.
    fn persist(&self) {
        let Some(path) = store_path() else {
            return;
        };
        let json = match serde_json::to_string(&self.namespaces) {
            Ok(json) => json,
            Err(error) => {
                warn!(error = %error, "chrome tab ledger not written: serialization failed");
                return;
            }
        };
        if let Err(error) = crate::util::write_json_record(&path, &json) {
            warn!(error = %error, path = %path.display(), "chrome tab ledger not written");
        }
    }
}

/// A missing, unreadable or unparsable file counts as empty (fail-open).
fn read_ledger(path: &Path) -> Namespaces {
    let json = match std::fs::read_to_string(path) {
        Ok(json) => json,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Namespaces::new(),
        Err(error) => {
            warn!(error = %error, path = %path.display(), "chrome tab ledger unreadable — starting empty");
            return Namespaces::new();
        }
    };
    match serde_json::from_str(&json) {
        Ok(namespaces) => namespaces,
        Err(error) => {
            warn!(error = %error, path = %path.display(), "chrome tab ledger unparsable — starting empty");
            Namespaces::new()
        }
    }
}

/// Loaded lazily on first access, reloaded whenever the configured store moves.
static LEDGER: OnceLock<Mutex<LedgerState>> = OnceLock::new();

fn ledger_state() -> &'static Mutex<LedgerState> {
    LEDGER.get_or_init(|| Mutex::new(LedgerState::default()))
}

/// The configured storage root, or whatever a test asked for.
fn store_path() -> Option<PathBuf> {
    #[cfg(test)]
    if let Some(store) = test_store() {
        return Some(store);
    }
    crate::config::CONFIG
        .try_storage_root()
        .map(|root| root.join(TAB_LEDGER_FILE_NAME))
}

/// The name, plus the recency sequence the tab had when picked — the pick's identity, which lets
/// [`take_eviction`] and [`settle_eviction`] tell a stale pick (addressed again since) from theirs.
#[derive(Clone, Debug)]
pub(crate) struct Eviction {
    pub(crate) name: String,
    pub(crate) seq: u64,
}

/// `reached` says the browser really answered for the page (the caller decides it): only then does
/// the call count — as an appearance, an address or a revival. Returns the tab to close when this
/// call put the record over the cap; it stays counted open until `settle_eviction` says otherwise.
#[must_use]
pub(crate) fn note_addressed(namespace: &str, tab: &str, reached: bool) -> Option<Eviction> {
    let mut state = ledger_state().lock().unwrap_poison();
    state.ensure_loaded();
    let (victim, changed) = record_addressed(&mut state.namespaces, namespace, tab, reached);
    if changed {
        state.persist();
    }
    victim
}

/// The victim the cap picked (if any) and whether the record changed, i.e. must be persisted.
fn record_addressed(
    namespaces: &mut Namespaces,
    namespace: &str,
    tab: &str,
    reached: bool,
) -> (Option<Eviction>, bool) {
    // A call the browser never answered for left no page: no namespace, no tab, no recency, no
    // revival — it can neither mint a phantom tab nor keep a real one off the cap by looking
    // recent.
    if !reached {
        return (None, false);
    }
    let record = namespaces.entry(namespace.to_string()).or_default();
    if let Some(index) = record.tabs.iter().position(|entry| entry.name == tab) {
        // Present: the browser answered for this tab, so the call is an address — it takes the
        // newest recency, which is what keeps the tab the run works with off the cap's list — and
        // a revival for a tab the cap closed (an `open` minting the page under this name again).
        let revived = record.tabs[index].closed;
        record.tabs[index].closed = false;
        record.tabs[index].seq = record.next_seq();
        // An address to an open tab leaves the count where it was, so only a revival can have
        // crossed the cap.
        return if revived {
            (choose_eviction(record, tab), true)
        } else {
            (None, true)
        };
    }
    // Absent: the browser answered for this name, so its page is a tab of the run's.
    let seq = record.next_seq();
    record.tabs.push(TabEntry {
        name: tab.to_string(),
        seq,
        ..TabEntry::default()
    });
    (choose_eviction(record, tab), true)
}

/// After an appearance or revival the cap may be crossed: pick the least-recently-addressed
/// open tab — not the one just addressed, not one already eviction-in-flight — mark it in
/// flight and return it. `None` within the cap, or when every other tab is being evicted.
fn choose_eviction(record: &mut NamespaceRecord, current: &str) -> Option<Eviction> {
    let open = record.tabs.iter().filter(|entry| !entry.closed).count();
    if open <= MAX_AGENT_TABS {
        return None;
    }
    let victim = record
        .tabs
        .iter_mut()
        .filter(|entry| !entry.closed && !entry.evicting && entry.name != current)
        .min_by_key(|entry| entry.seq)?;
    victim.evicting = true;
    Some(Eviction {
        name: victim.name.clone(),
        seq: victim.seq,
    })
}

/// Addressing a closed tab is an error until an `open` with an address mints the page again.
#[must_use]
pub(crate) fn is_closed(namespace: &str, tab: &str) -> bool {
    let mut state = ledger_state().lock().unwrap_poison();
    state.ensure_loaded();
    state
        .namespaces
        .get(namespace)
        .and_then(|record| record.tabs.iter().find(|entry| entry.name == tab))
        .is_some_and(|entry| entry.closed)
}

/// `false` when the run addressed the tab since it was picked (its recency moved on): the eviction
/// is dropped rather than run under the call that just came back to it, leaving the record over the
/// cap for the next appearance to pick again.
pub(crate) fn take_eviction(namespace: &str, eviction: &Eviction) -> bool {
    let mut state = ledger_state().lock().unwrap_poison();
    state.ensure_loaded();
    let Some(entry) = state.namespaces.get_mut(namespace).and_then(|record| {
        record
            .tabs
            .iter_mut()
            .find(|entry| entry.name == eviction.name)
    }) else {
        return false;
    };
    if entry.seq == eviction.seq && !entry.closed {
        return true;
    }
    entry.evicting = false;
    false
}

/// `gone` (the browser confirmed the tab is gone) marks it closed, unless the run addressed it
/// since the pick — the close raced a call, and a page minted since must not be counted away.
/// `!gone` only clears the in-flight mark, so the next over-cap appearance picks it again.
pub(crate) fn settle_eviction(namespace: &str, eviction: &Eviction, gone: bool) {
    let mut state = ledger_state().lock().unwrap_poison();
    state.ensure_loaded();
    let closed_now = {
        let Some(entry) = state.namespaces.get_mut(namespace).and_then(|record| {
            record
                .tabs
                .iter_mut()
                .find(|entry| entry.name == eviction.name)
        }) else {
            return;
        };
        let closed_now = gone && entry.seq == eviction.seq && !entry.closed;
        // The in-flight mark is never serialized, so clearing it needs no write.
        entry.evicting = false;
        if closed_now {
            entry.closed = true;
        }
        closed_now
    };
    if closed_now {
        state.persist();
    }
}

/// Its tabs are gone: the run-end release concluded.
pub(crate) fn forget_namespace(namespace: &str) {
    let mut state = ledger_state().lock().unwrap_poison();
    state.ensure_loaded();
    if state.namespaces.remove(namespace).is_some() {
        state.persist();
    }
}

/// Once at boot: a namespace no live run, no held release record and no resumable job claims is
/// past its tabs.
pub(crate) fn prune(protected: impl Fn(&str) -> bool) {
    let mut state = ledger_state().lock().unwrap_poison();
    state.ensure_loaded();
    let before = state.namespaces.len();
    state.namespaces.retain(|namespace, _| protected(namespace));
    if state.namespaces.len() != before {
        state.persist();
    }
}

#[cfg(test)]
static TEST_STORE: Mutex<Option<PathBuf>> = Mutex::new(None);

/// `None` means it resolves from the configured root. A test's store is always a file; the
/// in-memory-only case is the production one, where no root resolves.
#[cfg(test)]
fn test_store() -> Option<PathBuf> {
    TEST_STORE.lock().unwrap_poison().clone()
}

/// Returns the previous seam value so an RAII guard can restore it on drop, panics included.
#[cfg(test)]
fn swap_store(store: PathBuf) -> Option<PathBuf> {
    TEST_STORE.lock().unwrap_poison().replace(store)
}

#[cfg(test)]
fn restore_store(previous: Option<PathBuf>) {
    *TEST_STORE.lock().unwrap_poison() = previous;
}

/// Throw the in-memory record away without writing, so the next access reloads from the file —
/// how a test simulates a restart.
#[cfg(test)]
fn reset_state() {
    *ledger_state().lock().unwrap_poison() = LedgerState::default();
}

/// A private record store for a test: restores the previous store and resets the in-memory record
/// on drop, panics included, so no test touches the shared root's record or leaks state. A test
/// holding this must carry `#[serial_test::serial(chrome_tabs)]`.
#[cfg(test)]
pub(crate) struct TestLedgerGuard {
    path: PathBuf,
    previous: Option<PathBuf>,
    /// Holds the record file for the guard's lifetime.
    _dir: tempfile::TempDir,
}

#[cfg(test)]
impl TestLedgerGuard {
    pub(crate) fn install() -> Self {
        let dir = tempfile::tempdir().expect("chrome tab ledger test dir");
        let path = dir.path().join(TAB_LEDGER_FILE_NAME);
        let previous = swap_store(path.clone());
        reset_state();
        Self {
            path,
            previous,
            _dir: dir,
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// The view a restart reads back.
    #[cfg(unix)]
    pub(crate) fn holds(&self, namespace: &str) -> bool {
        let Ok(json) = std::fs::read_to_string(&self.path) else {
            return false;
        };
        serde_json::from_str::<serde_json::Value>(&json)
            .is_ok_and(|record| record.get(namespace).is_some())
    }
}

#[cfg(test)]
impl Drop for TestLedgerGuard {
    fn drop(&mut self) {
        // Reset before restoring the store, so nothing from the temp store can be written to the
        // real one by a later access.
        reset_state();
        restore_store(self.previous.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn namespaces() -> Namespaces {
        let mut state = ledger_state().lock().unwrap_poison();
        state.ensure_loaded();
        state.namespaces.clone()
    }

    /// Fill a fresh namespace to the cap with `a..e`; no cap is crossed on the way.
    fn fill_to_cap(namespace: &str) {
        for tab in ["a", "b", "c", "d", "e"] {
            assert!(
                note_addressed(namespace, tab, true).is_none(),
                "no cap crossed while filling to {MAX_AGENT_TABS}"
            );
        }
    }

    #[test]
    #[serial_test::serial(chrome_tabs)]
    fn confirmed_evictions_keep_the_cap() {
        let _guard = TestLedgerGuard::install();
        let ns = "agent-tab-aaaaaaaaaaaa-";
        for i in 0..12 {
            let tab = format!("tab-{i}");
            if let Some(victim) = note_addressed(ns, &tab, true) {
                assert!(
                    !is_closed(ns, &victim.name),
                    "the victim stays open until settled"
                );
                settle_eviction(ns, &victim, true);
                assert!(
                    is_closed(ns, &victim.name),
                    "a confirmed eviction closes the tab"
                );
            }
        }
        let open = (0..12)
            .filter(|i| !is_closed(ns, &format!("tab-{i}")))
            .count();
        assert_eq!(
            open, MAX_AGENT_TABS,
            "the cap left exactly {MAX_AGENT_TABS} open"
        );
    }

    #[test]
    #[serial_test::serial(chrome_tabs)]
    fn failed_eviction_stays_open_and_is_picked_again() {
        let _guard = TestLedgerGuard::install();
        let ns = "agent-tab-cccccccccccc-";
        fill_to_cap(ns);
        let victim = note_addressed(ns, "f", true).expect("the cap was crossed");
        assert_eq!(victim.name, "a");
        settle_eviction(ns, &victim, false);
        assert!(
            !is_closed(ns, &victim.name),
            "a failed eviction leaves the tab open"
        );
        assert_eq!(
            note_addressed(ns, "g", true).map(|eviction| eviction.name),
            Some("a".to_string())
        );
    }

    /// A pick the run has since addressed again is stale: the tab stays open.
    #[test]
    #[serial_test::serial(chrome_tabs)]
    fn a_settle_for_a_pick_the_run_addressed_again_is_ignored() {
        let _guard = TestLedgerGuard::install();
        let ns = "agent-tab-999999999999-";
        fill_to_cap(ns);
        let victim = note_addressed(ns, "f", true).expect("the cap was crossed");
        assert_eq!(victim.name, "a");
        assert!(note_addressed(ns, "a", true).is_none());
        settle_eviction(ns, &victim, true);
        assert!(
            !is_closed(ns, &victim.name),
            "the close lost the race: the tab the run went back to stays open"
        );
        assert_eq!(
            note_addressed(ns, "g", true).map(|eviction| eviction.name),
            Some("b".to_string())
        );
    }

    /// The close is dropped and the pick left to the next appearance.
    #[test]
    #[serial_test::serial(chrome_tabs)]
    fn take_eviction_drops_a_pick_the_run_addressed_again() {
        let _guard = TestLedgerGuard::install();
        let ns = "agent-tab-888888888888-";
        fill_to_cap(ns);
        let victim = note_addressed(ns, "f", true).expect("the cap was crossed");
        assert!(take_eviction(ns, &victim), "the pick is still owed");
        assert!(note_addressed(ns, "a", true).is_none());
        assert!(
            !take_eviction(ns, &victim),
            "the run went back to the tab: the close is dropped"
        );
        // The dropped pick must not leave the tab marked eviction-in-flight, or it could never be
        // picked again.
        assert_eq!(
            note_addressed(ns, "g", true).map(|eviction| eviction.name),
            Some("b".to_string())
        );
    }

    #[test]
    #[serial_test::serial(chrome_tabs)]
    fn a_closed_name_reopens_only_through_a_reached_call() {
        let _guard = TestLedgerGuard::install();
        let ns = "agent-tab-dddddddddddd-";
        fill_to_cap(ns);
        let victim = note_addressed(ns, "f", true).expect("the cap was crossed");
        assert_eq!(victim.name, "a");
        settle_eviction(ns, &victim, true);
        assert!(is_closed(ns, "a"));

        assert!(note_addressed(ns, "a", false).is_none());
        assert!(is_closed(ns, "a"));

        // The revival crosses the cap, so a victim comes back; settle it to keep the fixture tidy.
        let revictim = note_addressed(ns, "a", true);
        assert!(!is_closed(ns, "a"), "a reached call brings the name back");
        if let Some(name) = revictim {
            settle_eviction(ns, &name, true);
        }
    }

    #[test]
    #[serial_test::serial(chrome_tabs)]
    fn an_unreached_call_on_an_unknown_name_records_nothing() {
        let guard = TestLedgerGuard::install();
        let ns = "agent-tab-eeeeeeeeeeee-";
        assert!(note_addressed(ns, "a", false).is_none());
        assert!(!is_closed(ns, "a"));
        assert!(
            !guard.path().exists(),
            "an unreached call must not create a phantom tab or touch the file"
        );
        assert!(namespaces().is_empty(), "no namespace entry was created");

        assert!(note_addressed(ns, "a", true).is_none());
        assert!(
            namespaces().contains_key(ns),
            "the reached call recorded the namespace"
        );
    }

    /// The surviving recency order still decides the victim.
    #[test]
    #[serial_test::serial(chrome_tabs)]
    fn the_record_survives_a_restart_with_its_order_intact() {
        let _guard = TestLedgerGuard::install();
        let ns = "agent-tab-ffffffffffff-";
        fill_to_cap(ns);
        let victim = note_addressed(ns, "f", true).expect("the cap was crossed");
        assert_eq!(victim.name, "a");
        settle_eviction(ns, &victim, true);

        reset_state();
        assert!(is_closed(ns, "a"), "the closed tab survived the restart");
        assert!(note_addressed(ns, "f", true).is_none());
        assert_eq!(
            note_addressed(ns, "g", true).map(|eviction| eviction.name),
            Some("b".to_string())
        );
    }

    #[test]
    #[serial_test::serial(chrome_tabs)]
    fn prune_and_forget_drop_records() {
        let _guard = TestLedgerGuard::install();
        let keep = "agent-tab-222222222222-";
        for ns in ["agent-tab-111111111111-", keep, "agent-tab-333333333333-"] {
            assert!(note_addressed(ns, "a", true).is_none());
        }
        assert_eq!(namespaces().len(), 3);

        prune(|ns| ns == keep);
        assert_eq!(namespaces().len(), 1);
        assert!(namespaces().contains_key(keep));
        reset_state();
        assert_eq!(namespaces().len(), 1);

        forget_namespace(keep);
        assert!(namespaces().is_empty());
        reset_state();
        assert!(namespaces().is_empty(), "the drop was persisted");
    }
}
