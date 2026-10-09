//! Which browser tabs one agent run has open, in what order the run last addressed them, and
//! which of them the tab cap closed.
//!
//! The agent-facing `chrome` tool keeps at most [`MAX_AGENT_TABS`] tabs open per run. A tab is
//! one logical `tab` name the agent uses, and the cap is enforced from this record rather than
//! from the browser, because the browser is not always reachable: an unreached call must not
//! create a tab, and a tab the cap closed must stay refused until a call that really reached the
//! browser mints the page under that name again.
//!
//! This module is the record alone — pure state plus one durable JSON file. It makes no browser
//! call: a caller feeds it what it learned (`reached`, the close's own outcome) and reads back
//! what to do ([`note_addressed`] picks a victim, [`take_eviction`] re-checks that pick before the
//! close, [`settle_eviction`] records what the close came to), and the release path drives its
//! boot prune and its run-release deletion ([`crate::tools::chrome_release`]).
//!
//! ## What the record holds
//!
//! One entry per tab namespace (`agent-tab-<12 hex>-`, minted by
//! [`crate::tools::chrome::run_session_namespace`]; this module treats the key as opaque). Each
//! namespace carries the tabs the run addressed, each with the recency sequence of the call that
//! last addressed it — so the least-recently-ADDRESSED tab, not the oldest created, is the one
//! the cap closes — and whether the cap closed it.
//!
//! Only a call the browser answered for moves the record: it is an appearance for a name the
//! record does not hold, an address (recency) for one it holds open, and a revival for one the cap
//! closed. A call that left no page — never reached the browser, or found none of the session's to
//! drive — moves nothing, however the caller reports it.
//!
//! ## Durable, incremental, self-dropping
//!
//! The record is written after every mutation (compact JSON, tmp+rename) so a run resumed after
//! a self-update or a service restart finds the tabs its earlier segment left open; it is dropped
//! with the tabs at run end ([`forget_namespace`]), and once at boot every namespace no live run
//! claims is dropped ([`prune`]) because its tabs are gone.
//!
//! Failure is fail-open throughout: no storage root means in-memory only (still fully
//! functional), and an unreadable or unparsable file is treated as empty rather than fatal.

use crate::util::UnwrapPoison;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use tracing::warn;

/// How many tabs one agent run may keep open.
pub(crate) const MAX_AGENT_TABS: usize = 5;

/// Durable record of the tabs one run addresses, at the storage root. A file rather than a DB row
/// because the record must survive a service restart and be written incrementally as the agent
/// works; it is small and written whole on every mutation.
const TAB_LEDGER_FILE_NAME: &str = "chrome-agent-tabs.json";

/// The tabs one namespace addressed, plus the recency sequence the next call will hand out.
#[derive(Clone, Default, Serialize, Deserialize)]
struct NamespaceRecord {
    /// Next recency sequence to hand out; recency is the order the agent addressed tabs.
    #[serde(default)]
    seq: u64,
    #[serde(default)]
    tabs: Vec<TabEntry>,
}

impl NamespaceRecord {
    /// The next recency sequence — a new, larger value per addressed call, so the smallest open
    /// value is the least recently addressed tab.
    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }
}

/// One logical tab name and what the run has done with it.
#[derive(Clone, Default, Serialize, Deserialize)]
struct TabEntry {
    /// The logical tab name the agent uses, matched exactly.
    name: String,
    /// Recency sequence of the call that last addressed this tab.
    #[serde(default)]
    seq: u64,
    /// Set when the cap closed this tab: addressing it is an error until a reached call revives
    /// it.
    #[serde(default)]
    closed: bool,
    /// The cap picked this tab but the browser has not confirmed it gone yet. Never serialized —
    /// an unconfirmed eviction after a restart is only a tab that may still be open.
    #[serde(skip)]
    evicting: bool,
}

/// The whole record: one entry per namespace, ordered by namespace so the file's bytes are stable.
type Namespaces = BTreeMap<String, NamespaceRecord>;

/// In-memory mirror of the record file, plus the store path it was loaded from.
#[derive(Default)]
struct LedgerState {
    /// The store path the in-memory record was loaded from (`None` = in-memory only). Whenever the
    /// store the record resolves to differs from this — the configured path changing, or a test
    /// seam installing its own dir — the record is reloaded from the new one, so one store's tabs
    /// are never carried under another's.
    loaded_path: Option<PathBuf>,
    namespaces: Namespaces,
}

impl LedgerState {
    /// Reload the record when the configured store differs from the one already loaded.
    fn ensure_loaded(&mut self) {
        let path = store_path();
        if self.loaded_path == path {
            return;
        }
        self.namespaces = path.as_deref().map_or_else(Namespaces::new, read_ledger);
        self.loaded_path = path;
    }

    /// Write the whole record out (compact JSON, atomic tmp+rename). Fail-open: an in-memory-only
    /// store or a failed write leaves whatever was there, and a serialization failure never
    /// publishes an empty payload over a live file.
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

/// Read the record file, treating a missing, unreadable or unparsable file as empty (fail-open,
/// same as the release record file).
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

/// The process-global record, loaded lazily on first access and reloaded whenever the configured
/// store moves.
static LEDGER: OnceLock<Mutex<LedgerState>> = OnceLock::new();

fn ledger_state() -> &'static Mutex<LedgerState> {
    LEDGER.get_or_init(|| Mutex::new(LedgerState::default()))
}

/// The store the record lives in: the configured storage root, or whatever a test asked for.
fn store_path() -> Option<PathBuf> {
    #[cfg(test)]
    if let Some(store) = test_store() {
        return Some(store);
    }
    crate::config::CONFIG
        .try_storage_root()
        .map(|root| root.join(TAB_LEDGER_FILE_NAME))
}

/// One tab the cap picked for closing: the logical name, plus the recency sequence the tab had
/// when it was picked. The sequence is the pick's identity — [`take_eviction`] and
/// [`settle_eviction`] tell a pick the run has since addressed again (stale) from the one they were
/// made for.
#[derive(Clone, Debug)]
pub(crate) struct Eviction {
    pub(crate) name: String,
    pub(crate) seq: u64,
}

/// Record one addressed call. `reached` says the browser really answered for the page (the caller
/// decides it): only then does the call count at all — as an appearance for a name the record does
/// not hold, an address for one it holds, or a revival for one the cap closed. Returns the tab to
/// close when this call put the record over the cap; the tab stays counted open until
/// `settle_eviction` says otherwise.
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

/// The mutation behind [`note_addressed`], returning the victim the cap picked (if any) and
/// whether the record changed — i.e. whether it must be persisted.
fn record_addressed(
    namespaces: &mut Namespaces,
    namespace: &str,
    tab: &str,
    reached: bool,
) -> (Option<Eviction>, bool) {
    // A call the browser never answered for left no page, so it counts for nothing at all: no
    // namespace, no tab, no recency, no revival. The record describes what really happened in the
    // browser — so a call that never got there can neither mint a phantom tab nor keep a real one
    // off the cap's list by looking like recent use.
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

/// After an appearance or a revival the cap may be crossed: pick the least-recently-addressed open
/// tab — excluding the one just addressed and any tab already eviction-in-flight — mark it
/// eviction-in-flight and return it. `None` when the record is within the cap, or when every
/// other open tab is already being evicted (that eviction is the work in progress).
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

/// Whether this tab was closed by the cap: addressing it is an error until an `open` with an
/// address mints a page under the name again (see [`note_addressed`]).
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

/// Whether the pick is still owed, asked right before the close runs: `false` when the run has
/// addressed the tab since it was picked (its recency moved on) — the run wants that tab, so the
/// eviction is dropped rather than run under the call that just came back to it. An eviction
/// dropped here leaves the record over the cap, and the next appearance picks again.
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

/// The outcome of an eviction started by [`note_addressed`]: `gone` (the browser confirmed the tab
/// is gone) marks the tab closed, unless the run has addressed the tab since the pick — then the
/// close raced a call the run made, and a page the run has since minted under the name must not be
/// counted away. `!gone` only clears the in-flight mark, so the tab stays counted open and the next
/// appearance over the cap picks it again.
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
        // The in-flight mark is never serialized, so clearing it alone needs no write.
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

/// Drop the whole record of one namespace (its tabs are gone — the run-end release concluded).
pub(crate) fn forget_namespace(namespace: &str) {
    let mut state = ledger_state().lock().unwrap_poison();
    state.ensure_loaded();
    if state.namespaces.remove(namespace).is_some() {
        state.persist();
    }
}

/// Drop every record `protected` does not claim (called once at boot: a namespace no live run, no
/// held release record and no resumable job claims is past its tabs).
pub(crate) fn prune(protected: impl Fn(&str) -> bool) {
    let mut state = ledger_state().lock().unwrap_poison();
    state.ensure_loaded();
    let before = state.namespaces.len();
    state.namespaces.retain(|namespace, _| protected(namespace));
    if state.namespaces.len() != before {
        state.persist();
    }
}

// ── Test seam ────────────────────────────────────────────────────────────────

#[cfg(test)]
static TEST_STORE: Mutex<Option<PathBuf>> = Mutex::new(None);

/// The store a test installed, if any — `None` means no test holds the record and it resolves from
/// the configured root. A test's store is always a file (the in-memory-only case is the production
/// one, where no root resolves at all).
#[cfg(test)]
fn test_store() -> Option<PathBuf> {
    TEST_STORE.lock().unwrap_poison().clone()
}

/// Install a store, returning the previous seam value so an RAII guard can restore it on drop —
/// including during a panic.
#[cfg(test)]
fn swap_store(store: PathBuf) -> Option<PathBuf> {
    TEST_STORE.lock().unwrap_poison().replace(store)
}

/// Restore a previously swapped-out seam value.
#[cfg(test)]
fn restore_store(previous: Option<PathBuf>) {
    *TEST_STORE.lock().unwrap_poison() = previous;
}

/// Throw the in-memory record away without writing, so the next access reloads from the file —
/// how a test simulates a service restart.
#[cfg(test)]
fn reset_state() {
    *ledger_state().lock().unwrap_poison() = LedgerState::default();
}

/// A private record store for a test, restoring the previous store and resetting the in-memory
/// record on drop — including during a panic — so no test reads or writes the shared root's record
/// and no test leaks state into another. A test holding this must carry
/// `#[serial_test::serial(chrome_tabs)]`.
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

    /// The record file this guard's store writes to.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Whether the durable record still holds `namespace` — the view a restart reads back.
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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole in-memory record, reloading it from the configured store first.
    fn namespaces() -> Namespaces {
        let mut state = ledger_state().lock().unwrap_poison();
        state.ensure_loaded();
        state.namespaces.clone()
    }

    /// Fill a fresh namespace to the cap with `a..e`, oldest created first, asserting no cap was
    /// crossed on the way.
    fn fill_to_cap(namespace: &str) {
        for tab in ["a", "b", "c", "d", "e"] {
            assert!(
                note_addressed(namespace, tab, true).is_none(),
                "no cap crossed while filling to {MAX_AGENT_TABS}"
            );
        }
    }

    /// Ten-plus appearances with a confirmed eviction each time leave exactly [`MAX_AGENT_TABS`]
    /// open.
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

    /// The victim is the least-recently-ADDRESSED tab, not the oldest created.
    #[test]
    #[serial_test::serial(chrome_tabs)]
    fn victim_is_the_least_recently_addressed_tab() {
        let _guard = TestLedgerGuard::install();
        let ns = "agent-tab-bbbbbbbbbbbb-";
        fill_to_cap(ns);
        // Re-address the oldest CREATED tab (`a`) so it is now the most recently addressed.
        assert!(
            note_addressed(ns, "a", true).is_none(),
            "a re-address does not cross the cap"
        );
        // One more appearance crosses the cap: the victim must be `b`, the second-oldest created.
        assert_eq!(
            note_addressed(ns, "f", true).map(|eviction| eviction.name),
            Some("b".to_string())
        );
    }

    /// A failed eviction leaves the tab counted open, and the very next over-cap appearance picks
    /// that same tab again.
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
        // Still the least-recently-addressed open tab, so the next over-cap call picks it again.
        assert_eq!(
            note_addressed(ns, "g", true).map(|eviction| eviction.name),
            Some("a".to_string())
        );
    }

    /// A settled eviction whose pick the run has since addressed again is stale: the tab stays
    /// open, because a page the run has just minted under that name must not be counted away.
    #[test]
    #[serial_test::serial(chrome_tabs)]
    fn a_settle_for_a_pick_the_run_addressed_again_is_ignored() {
        let _guard = TestLedgerGuard::install();
        let ns = "agent-tab-999999999999-";
        fill_to_cap(ns);
        let victim = note_addressed(ns, "f", true).expect("the cap was crossed");
        assert_eq!(victim.name, "a");
        // The run comes back to the victim before the close concludes.
        assert!(note_addressed(ns, "a", true).is_none());
        settle_eviction(ns, &victim, true);
        assert!(
            !is_closed(ns, &victim.name),
            "the close lost the race: the tab the run went back to stays open"
        );
        // And the dropped pick is asked for again by the next appearance over the cap.
        assert_eq!(
            note_addressed(ns, "g", true).map(|eviction| eviction.name),
            Some("b".to_string())
        );
    }

    /// An eviction the run has addressed again before the close started is not owed: the close is
    /// dropped and the pick is left to the next appearance.
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
        // picked again: the next over-cap appearance picks the next least-recently-addressed tab.
        assert_eq!(
            note_addressed(ns, "g", true).map(|eviction| eviction.name),
            Some("b".to_string())
        );
    }

    /// A closed name stays closed for an unreached call, is refused, and comes back only through a
    /// reached call.
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

        // An unreached call cannot revive it: it is still gone.
        assert!(note_addressed(ns, "a", false).is_none());
        assert!(is_closed(ns, "a"));

        // A reached call revives it (the `open` minting the page under the name again). The
        // revival crosses the cap, so a victim is returned; settle it to keep the fixture tidy.
        let revictim = note_addressed(ns, "a", true);
        assert!(!is_closed(ns, "a"), "a reached call brings the name back");
        if let Some(name) = revictim {
            settle_eviction(ns, &name, true);
        }
    }

    /// An unreached call on a name the record does not hold records nothing at all.
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

        // A reached call on the same name is then a plain appearance.
        assert!(note_addressed(ns, "a", true).is_none());
        assert!(
            namespaces().contains_key(ns),
            "the reached call recorded the namespace"
        );
    }

    /// The record survives a simulated restart, and the surviving recency order still decides the
    /// victim.
    #[test]
    #[serial_test::serial(chrome_tabs)]
    fn the_record_survives_a_restart_with_its_order_intact() {
        let _guard = TestLedgerGuard::install();
        let ns = "agent-tab-ffffffffffff-";
        fill_to_cap(ns);
        let victim = note_addressed(ns, "f", true).expect("the cap was crossed");
        assert_eq!(victim.name, "a");
        settle_eviction(ns, &victim, true);

        // Same store path: throw the in-memory record away and reload from the file.
        reset_state();
        assert!(is_closed(ns, "a"), "the closed tab survived the restart");
        // Address `f` once more (it stays open, no cap change), then add `g`: with the surviving
        // order the least-recently-addressed open tab is `b`.
        assert!(note_addressed(ns, "f", true).is_none());
        assert_eq!(
            note_addressed(ns, "g", true).map(|eviction| eviction.name),
            Some("b".to_string())
        );
    }

    /// `prune` keeps only the namespaces the predicate claims, and `forget_namespace` drops one
    /// record.
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
        // The prune is persisted: a restart still sees only the kept namespace.
        reset_state();
        assert_eq!(namespaces().len(), 1);

        forget_namespace(keep);
        assert!(namespaces().is_empty());
        reset_state();
        assert!(namespaces().is_empty(), "the drop was persisted");
    }
}
