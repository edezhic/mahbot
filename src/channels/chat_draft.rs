//! Persisted composer draft + pending reply-to state, keyed by (user,
//! resolved send-target workspace).
//!
//! The in-memory map is mutated synchronously at capture time so it is always
//! current; only the file write is debounced. Every write serializes the whole
//! map under the one mutex, so a stale debounce can never clobber newer data
//! and the exit flush can't be overwritten by an in-flight write.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};

use serde::{Deserialize, Serialize};

use crate::channels::ReplyReference;
use crate::util::{UnwrapPoison, unix_millis};

/// Hard bound on the number of (`user`, `workspace`) draft entries persisted —
/// the map is pruned to the most recently saved on every write.
const MAX_PERSISTED_ENTRIES: usize = 64;
/// File name inside `~/.mahbot/` holding the draft map.
const DRAFT_FILE_NAME: &str = "chat-draft.json";

/// A single persisted composer draft for one (`user`, workspace) context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct DraftEntry {
    #[serde(default)]
    pub(crate) text: String,
    #[serde(default)]
    pub(crate) reply: Option<ReplyReference>,
    /// Unix millis of the last [`DraftStore::set`]; used only for pruning.
    #[serde(default)]
    pub(crate) saved_at: u64,
}

impl DraftEntry {
    /// Whether the draft holds nothing to restore — an empty/whitespace text
    /// with no pending reply.
    #[must_use]
    fn is_empty(&self) -> bool {
        self.text.trim().is_empty() && self.reply.is_none()
    }
}

/// Outer key = user; inner key = resolved send-target workspace. Nesting
/// avoids any key-collision/escaping issues between the two names.
type DraftFile = BTreeMap<String, BTreeMap<String, DraftEntry>>;

/// The shared composer-draft store.
///
/// `path: None` marks the store disabled (HOME unresolvable) — all ops no-op.
pub(crate) struct DraftStore {
    path: Option<PathBuf>,
    entries: Mutex<DraftFile>,
}

impl DraftStore {
    /// Resolve the on-disk draft path from `$HOME` (None when unset/empty).
    #[must_use]
    fn file_path() -> Option<PathBuf> {
        let home = std::env::var("HOME").ok()?;
        if home.is_empty() {
            return None;
        }
        Some(PathBuf::from(home).join(".mahbot").join(DRAFT_FILE_NAME))
    }

    /// Load the draft file (missing/corrupt → empty map, fail-open).
    fn load(path: Option<PathBuf>) -> Self {
        let entries = path
            .as_deref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|json| serde_json::from_str(&json).ok())
            .unwrap_or_default();
        Self {
            path,
            entries: Mutex::new(entries),
        }
    }

    /// The process-global store, parsed once at init.
    #[must_use]
    pub(crate) fn global() -> &'static Arc<DraftStore> {
        static GLOBAL: LazyLock<Arc<DraftStore>> =
            LazyLock::new(|| Arc::new(DraftStore::load(DraftStore::file_path())));
        &GLOBAL
    }

    /// Test/injection constructor; `Some(path)` reads+parses that file.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn at(path: Option<PathBuf>) -> Arc<DraftStore> {
        Arc::new(DraftStore::load(path))
    }

    /// Capture a draft snapshot into the in-memory map. An empty draft removes
    /// the (`user`, `workspace`) entry. Does not touch the file.
    pub(crate) fn set(
        &self,
        user: &str,
        workspace: &str,
        text: String,
        reply: Option<ReplyReference>,
    ) {
        if self.path.is_none() {
            return;
        }
        let entry = DraftEntry {
            text,
            reply,
            saved_at: unix_millis(),
        };
        if entry.is_empty() {
            self.remove(user, workspace);
            return;
        }
        self.entries
            .lock()
            .unwrap_poison()
            .entry(user.to_string())
            .or_default()
            .insert(workspace.to_string(), entry);
    }

    /// Drop the (`user`, `workspace`) entry. Does not touch the file.
    pub(crate) fn remove(&self, user: &str, workspace: &str) {
        if self.path.is_none() {
            return;
        }
        let mut map = self.entries.lock().unwrap_poison();
        if let Some(inner) = map.get_mut(user) {
            inner.remove(workspace);
            if inner.is_empty() {
                map.remove(user);
            }
        }
    }

    /// Read the current snapshot for a (`user`, `workspace`) context.
    #[must_use]
    pub(crate) fn get(&self, user: &str, workspace: &str) -> Option<DraftEntry> {
        self.path.as_ref()?;
        let map = self.entries.lock().unwrap_poison();
        map.get(user)?.get(workspace).cloned()
    }

    /// Write the whole in-memory map to disk (compact JSON, atomic tmp+rename).
    /// While holding the lock, prunes the map to the newest
    /// [`MAX_PERSISTED_ENTRIES`] by `saved_at`. Fail-open: errors are ignored.
    pub(crate) fn persist(&self) {
        let Some(path) = &self.path else {
            return;
        };
        let mut map = self.entries.lock().unwrap_poison();
        prune(&mut map);
        let json = serde_json::to_string(&*map).unwrap_or_default();
        let tmp = path.with_extension("json.tmp");
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&tmp, json);
        let _ = std::fs::rename(&tmp, path);
    }

    /// Spawn the sync write on the blocking pool. Holding the std mutex
    /// across the blocking write serializes concurrent writes and orders
    /// them against the synchronous exit flush. A no-op without a runtime
    /// (detached test contexts) — the in-memory map stays authoritative.
    pub(crate) fn persist_async(self: Arc<Self>) {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn_blocking(move || self.persist());
        }
    }
}

/// Drop entries beyond [`MAX_PERSISTED_ENTRIES`], keeping the newest by
/// `saved_at`.
fn prune(map: &mut DraftFile) {
    let total: usize = map.values().map(BTreeMap::len).sum();
    if total <= MAX_PERSISTED_ENTRIES {
        return;
    }
    let mut all: Vec<(String, String, u64)> = map
        .iter()
        .flat_map(|(user, ws)| {
            ws.iter()
                .map(move |(workspace, entry)| (user.clone(), workspace.clone(), entry.saved_at))
        })
        .collect();
    all.sort_by_key(|(_, _, saved_at)| *saved_at);
    let keep_from = all.len().saturating_sub(MAX_PERSISTED_ENTRIES);
    let keep: std::collections::HashSet<(String, String)> = all[keep_from..]
        .iter()
        .map(|(user, workspace, _)| (user.clone(), workspace.clone()))
        .collect();
    map.retain(|user, ws| {
        ws.retain(|workspace, _| keep.contains(&(user.clone(), workspace.clone())));
        !ws.is_empty()
    });
}

/// Flush the global in-memory map synchronously (exit paths).
pub(crate) fn flush_global() {
    DraftStore::global().persist();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> (tempfile::TempDir, Arc<DraftStore>) {
        let dir = tempfile::tempdir().unwrap();
        let store = DraftStore::at(Some(dir.path().join(DRAFT_FILE_NAME)));
        (dir, store)
    }

    #[test]
    fn round_trip_set_get_persist_reload() {
        let (dir, store) = temp_store();
        let path = dir.path().join(DRAFT_FILE_NAME);
        store.set("alice", "ws1", "hello".to_string(), None);
        assert_eq!(
            store.get("alice", "ws1").map(|e| e.text).as_deref(),
            Some("hello")
        );
        store.persist();
        let reloaded = DraftStore::at(Some(path));
        let entry = reloaded.get("alice", "ws1").expect("entry persisted");
        assert_eq!(entry.text, "hello");
        assert_eq!(entry.reply, None);
    }

    #[test]
    fn corrupt_file_fails_open_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DRAFT_FILE_NAME);
        std::fs::write(&path, "not valid json{").unwrap();
        let store = DraftStore::at(Some(path));
        assert_eq!(store.get("alice", "ws1"), None);
    }

    #[test]
    fn empty_draft_removes_the_entry() {
        let (_dir, store) = temp_store();
        store.set("alice", "ws1", "hello".to_string(), None);
        store.set("alice", "ws1", "   ".to_string(), None);
        assert_eq!(store.get("alice", "ws1"), None);
    }

    #[test]
    fn pruning_keeps_the_newest_entries() {
        let (dir, store) = temp_store();
        let path = dir.path().join(DRAFT_FILE_NAME);
        {
            let mut map = store.entries.lock().unwrap_poison();
            for i in 0..(MAX_PERSISTED_ENTRIES + 10) {
                map.entry("user".to_string()).or_default().insert(
                    format!("ws{i}"),
                    DraftEntry {
                        text: format!("text{i}"),
                        reply: None,
                        saved_at: i as u64,
                    },
                );
            }
        }
        store.persist();
        let reloaded = DraftStore::at(Some(path));
        let total: usize = reloaded
            .entries
            .lock()
            .unwrap_poison()
            .values()
            .map(BTreeMap::len)
            .sum();
        assert_eq!(total, MAX_PERSISTED_ENTRIES);
        // The newest (highest saved_at) entries survive; the oldest are pruned.
        for i in 10..(MAX_PERSISTED_ENTRIES + 10) {
            assert!(
                reloaded.get("user", &format!("ws{i}")).is_some(),
                "ws{i} must survive"
            );
        }
        for i in 0..10 {
            assert!(
                reloaded.get("user", &format!("ws{i}")).is_none(),
                "ws{i} must be pruned"
            );
        }
    }

    #[test]
    fn persist_leaves_no_tmp_file_behind() {
        let (dir, store) = temp_store();
        let path = dir.path().join(DRAFT_FILE_NAME);
        store.set("alice", "ws1", "hello".to_string(), None);
        store.persist();
        assert!(path.exists());
        assert!(!path.with_extension("json.tmp").exists());
    }
}
