//! Persisted composer draft + pending reply-to state for the chat message box.
//!
//! There is exactly one draft: the chat has a single composer, and what is
//! typed in it is not tied to the workspace picked in the footer. The in-memory
//! draft is mutated synchronously at capture time so it is always current; only
//! the file write is debounced. Every write serializes the whole draft under the
//! one mutex, so a stale debounce can never clobber newer data and the exit
//! flush can't be overwritten by an in-flight write.

use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};

use serde::{Deserialize, Serialize};

use crate::channels::ReplyReference;
use crate::util::UnwrapPoison;

/// File name inside the storage root holding the draft.
const DRAFT_FILE_NAME: &str = "chat-draft.json";

/// The persisted composer draft: the message text and the pending reply-to.
///
/// Loads fail open: a file that does not yield a draft — missing, corrupt, or
/// left in the previous per-(user, workspace) shape — is discarded by
/// [`DraftStore::load`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ChatDraft {
    #[serde(default)]
    pub(crate) text: String,
    #[serde(default)]
    pub(crate) reply: Option<ReplyReference>,
}

impl ChatDraft {
    /// Whether the draft holds nothing to restore — an empty/whitespace text
    /// with no pending reply.
    #[must_use]
    fn is_empty(&self) -> bool {
        self.text.trim().is_empty() && self.reply.is_none()
    }
}

/// The shared composer-draft store.
///
/// `path: None` marks the store disabled (no resolvable storage root) — all ops no-op.
pub(crate) struct DraftStore {
    path: Option<PathBuf>,
    draft: Mutex<ChatDraft>,
}

impl DraftStore {
    /// Resolve the on-disk draft path from the storage root (`None` when it cannot be
    /// resolved), the same root every other daemon file lives in.
    #[must_use]
    fn file_path() -> Option<PathBuf> {
        crate::config::default_config_dir()
            .ok()
            .map(|dir| dir.join(DRAFT_FILE_NAME))
    }

    /// Load the draft file (missing/corrupt → no draft, fail-open). A file that
    /// yields no draft is removed, so the file exists exactly while a draft does
    /// and a file left by the previous per-(user, workspace) shape cannot linger.
    fn load(path: Option<PathBuf>) -> Self {
        let draft: ChatDraft = path
            .as_deref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|json| serde_json::from_str(&json).ok())
            .unwrap_or_default();
        if draft.is_empty()
            && let Some(p) = &path
        {
            let _ = std::fs::remove_file(p);
        }
        Self {
            path,
            draft: Mutex::new(draft),
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

    /// Capture the composer snapshot. Does not touch the file.
    pub(crate) fn set(&self, text: String, reply: Option<ReplyReference>) {
        if self.path.is_none() {
            return;
        }
        *self.draft.lock().unwrap_poison() = ChatDraft { text, reply };
    }

    /// Drop the stored draft. Does not touch the file.
    pub(crate) fn remove(&self) {
        if self.path.is_none() {
            return;
        }
        *self.draft.lock().unwrap_poison() = ChatDraft::default();
    }

    /// Read the current draft; empty when none is stored.
    #[must_use]
    pub(crate) fn get(&self) -> ChatDraft {
        if self.path.is_none() {
            return ChatDraft::default();
        }
        self.draft.lock().unwrap_poison().clone()
    }

    /// Write the draft to disk (compact JSON, atomic tmp+rename). A draft with
    /// nothing in it removes the file, so the file exists exactly while a draft
    /// does. Fail-open: errors are ignored.
    pub(crate) fn persist(&self) {
        let Some(path) = &self.path else {
            return;
        };
        // Held across the write: concurrent persists must not interleave on the
        // shared tmp path (see the module docs).
        let draft = self.draft.lock().unwrap_poison();
        if draft.is_empty() {
            let _ = std::fs::remove_file(path);
            return;
        }
        if let Ok(json) = serde_json::to_string(&*draft) {
            let _ = crate::util::write_json_record(path, &json);
        }
    }

    /// Spawn the sync write on the blocking pool. Holding the std mutex
    /// across the blocking write serializes concurrent writes and orders
    /// them against the synchronous exit flush. A no-op without a runtime
    /// (detached test contexts) — the in-memory draft stays authoritative.
    pub(crate) fn persist_async(self: Arc<Self>) {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn_blocking(move || self.persist());
        }
    }
}

/// Flush the global in-memory draft synchronously (exit paths).
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

    fn reply(author: &str) -> ReplyReference {
        ReplyReference {
            author: author.to_string(),
            snippet: "snippet".to_string(),
        }
    }

    #[test]
    fn round_trip_set_get_persist_reload() {
        let (dir, store) = temp_store();
        let path = dir.path().join(DRAFT_FILE_NAME);
        store.set("hello".to_string(), Some(reply("Assistant")));
        assert_eq!(store.get().text, "hello");
        assert_eq!(store.get().reply, Some(reply("Assistant")));
        store.persist();
        let reloaded = DraftStore::at(Some(path));
        assert_eq!(reloaded.get().text, "hello");
        assert_eq!(reloaded.get().reply, Some(reply("Assistant")));
    }

    #[test]
    fn corrupt_file_fails_open_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DRAFT_FILE_NAME);
        std::fs::write(&path, "not valid json{").unwrap();
        let store = DraftStore::at(Some(path));
        assert!(store.get().is_empty());
    }

    #[test]
    fn legacy_per_workspace_file_loads_as_no_draft_and_is_removed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DRAFT_FILE_NAME);
        std::fs::write(&path, r#"{"admin":{"ws1":{"text":"old","reply":null}}}"#).unwrap();
        let store = DraftStore::at(Some(path.clone()));
        assert!(store.get().is_empty());
        assert!(
            !path.exists(),
            "the old per-workspace entry must not linger"
        );
    }

    #[test]
    fn empty_draft_clears_the_stored_one() {
        let (_dir, store) = temp_store();
        store.set("hello".to_string(), None);
        store.set("   ".to_string(), None);
        assert!(store.get().is_empty());
    }

    #[test]
    fn a_reply_without_text_is_a_draft() {
        let (dir, store) = temp_store();
        let path = dir.path().join(DRAFT_FILE_NAME);
        store.set(String::new(), Some(reply("Assistant")));
        store.persist();
        assert!(path.exists());
        assert_eq!(
            DraftStore::at(Some(path)).get().reply,
            Some(reply("Assistant"))
        );
    }

    #[test]
    fn persist_leaves_no_tmp_file_behind() {
        let (dir, store) = temp_store();
        let path = dir.path().join(DRAFT_FILE_NAME);
        store.set("hello".to_string(), None);
        store.persist();
        assert!(path.exists());
        assert!(!path.with_extension("json.tmp").exists());
    }

    #[test]
    fn an_empty_draft_removes_the_file() {
        let (dir, store) = temp_store();
        let path = dir.path().join(DRAFT_FILE_NAME);
        store.set("hello".to_string(), None);
        store.persist();
        store.remove();
        store.persist();
        assert!(!path.exists());
    }
}
