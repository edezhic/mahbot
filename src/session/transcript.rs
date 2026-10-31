//! Live read-only transcript snapshots for running agents.
//!
//! A running agent's in-memory `Session` conversation is not yet fully
//! persisted (the unpersisted tail). This module exposes the live transcript
//! as a lock-free, read-only, point-in-time snapshot that in-process readers —
//! the Running Agents GUI and the Sessions page's live-session view — load on
//! top of the durable DB rows.
//!
//! ## Mechanism
//!
//! Each running agent owns an [`ArcSwap<TranscriptSnapshot>`] holder. The
//! `Session` is the SOLE writer: on every mutation of history or token count it
//! copy-on-writes a fresh snapshot and `.store()`s it. Readers call
//! `.load_full()` to get an owned `Arc<TranscriptSnapshot>` (a refcount bump,
//! no data copy) they can hold across an `.await`.
//!
//! The holder is indexed globally by `agent_id` in [`TRANSCRIPT_REGISTRY`],
//! generation-guarded so an `agent_id` reused by a replacement run never
//! surfaces or mutates a stale run's snapshot.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;

use arc_swap::ArcSwap;

use crate::ChatMessage;
use crate::util::UnwrapPoison;

/// A coherent read-only point-in-time view of a running agent's live session
/// transcript.
///
/// `history` is the full in-memory conversation (system prompt → current
/// unpersisted tail); `token_count` is the LIVE session length as currently
/// recorded by the agent — not a finalized-only figure.
#[derive(Debug, Default)]
pub struct TranscriptSnapshot {
    pub history: Vec<ChatMessage>,
    pub token_count: Option<u64>,
}

/// One registered agent's snapshot holder.
struct TranscriptEntry {
    generation: u64,
    holder: Arc<ArcSwap<TranscriptSnapshot>>,
}

/// Global keyed accessor for running-agent transcript snapshots.
///
/// Indexed by `agent_id`; each entry is generation-guarded so a replacement
/// agent with the same `agent_id` gets a FRESH holder and a stale
/// deregistration can never remove the replacement's entry.
#[derive(Default)]
pub(crate) struct TranscriptRegistry {
    inner: Mutex<HashMap<String, TranscriptEntry>>,
}

impl TranscriptRegistry {
    /// Register a fresh snapshot holder for `agent_id` at the given registry
    /// generation. Returns the holder the agent's `Session` publishes into.
    ///
    /// Registering over an existing `agent_id` (a replacement run) replaces the
    /// old holder — a reader holding an old `Arc<TranscriptSnapshot>` keeps
    /// that immutable snapshot, and the replacement writes into a fresh holder.
    pub fn register(&self, agent_id: String, generation: u64) -> Arc<ArcSwap<TranscriptSnapshot>> {
        let holder = Arc::new(ArcSwap::from_pointee(TranscriptSnapshot::default()));
        let mut map = self.inner.lock().unwrap_poison();
        map.insert(
            agent_id,
            TranscriptEntry {
                generation,
                holder: holder.clone(),
            },
        );
        holder
    }

    /// Remove the entry only if its generation still matches — a stale
    /// deregistration from a finished/restarted agent can never remove a
    /// replacement run's snapshot holder.
    pub fn deregister(&self, agent_id: &str, generation: u64) {
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(entry) = map.get(agent_id)
            && entry.generation == generation
        {
            map.remove(agent_id);
        }
    }

    /// The current live snapshot for `agent_id`, if that agent has registered
    /// one. Returns an owned `Arc` (refcount bump, no data copy) the caller
    /// can hold across an `.await`.
    #[must_use]
    pub fn snapshot(&self, agent_id: &str) -> Option<Arc<TranscriptSnapshot>> {
        let holder = {
            self.inner
                .lock()
                .unwrap_poison()
                .get(agent_id)
                .map(|e| e.holder.clone())
        }?;
        Some(holder.load_full())
    }
}

/// Global static registry.
pub(crate) static TRANSCRIPT_REGISTRY: LazyLock<TranscriptRegistry> =
    LazyLock::new(TranscriptRegistry::default);

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    fn next_gen() -> u64 {
        static GEN: AtomicU64 = AtomicU64::new(1);
        GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// The generation is the identity of a run: a deregister carrying any other
    /// generation touches nothing, neither the entry it re-registered over nor
    /// the fresh holder a replacement run registered, and only the matching
    /// generation removes it.
    #[test]
    fn only_the_matching_generation_removes_an_entry() {
        let agent_id = format!("snap_dereg_{}", crate::generate_suffix());
        let generation = next_gen();
        let holder_a = TRANSCRIPT_REGISTRY.register(agent_id.clone(), generation);

        // Stale (wrong-generation) deregister is a no-op.
        TRANSCRIPT_REGISTRY.deregister(&agent_id, generation + 1);
        assert!(
            TRANSCRIPT_REGISTRY.snapshot(&agent_id).is_some(),
            "stale deregister must not remove the entry"
        );

        // Replacement run registers a fresh holder with a new generation; the
        // old run's deregister (now stale) must NOT remove it.
        let gen_b = next_gen();
        let holder_b = TRANSCRIPT_REGISTRY.register(agent_id.clone(), gen_b);
        TRANSCRIPT_REGISTRY.deregister(&agent_id, generation);
        assert!(
            TRANSCRIPT_REGISTRY.snapshot(&agent_id).is_some(),
            "replacement entry survives the stale deregister"
        );

        // Matching deregister removes it.
        TRANSCRIPT_REGISTRY.deregister(&agent_id, gen_b);
        assert!(
            TRANSCRIPT_REGISTRY.snapshot(&agent_id).is_none(),
            "matching deregister removes the entry"
        );
        let _ = (holder_a, holder_b);
    }
}
