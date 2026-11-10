//! Live read-only transcript snapshots for running agents: the in-memory `Session`
//! conversation is not yet fully persisted, so in-process readers (the Running Agents GUI
//! and the Sessions page's live view) load this snapshot on top of the durable rows.
//!
//! Each running agent owns an [`ArcSwap<TranscriptSnapshot>`] holder; the `Session` is the
//! SOLE writer and copy-on-writes a fresh snapshot on every history or token-count mutation.
//! Readers `.load_full()` (a refcount bump, no data copy) and hold it across an `.await`. The
//! holder is indexed by `agent_id` in [`TRANSCRIPT_REGISTRY`], generation-guarded so a reused
//! `agent_id` never surfaces or mutates a stale run's snapshot.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;

use arc_swap::ArcSwap;

use crate::ChatMessage;
use crate::util::UnwrapPoison;

/// A coherent read-only point-in-time view of a running agent's live transcript: `history`
/// is the full in-memory conversation (system prompt → unpersisted tail), and `token_count`
/// is the LIVE session length, not a finalized-only figure.
#[derive(Debug, Default)]
pub struct TranscriptSnapshot {
    pub history: Vec<ChatMessage>,
    pub token_count: Option<u64>,
}

struct TranscriptEntry {
    generation: u64,
    holder: Arc<ArcSwap<TranscriptSnapshot>>,
}

/// Global keyed accessor for running-agent snapshots, indexed by `agent_id`; each entry is
/// generation-guarded so a replacement agent with the same id gets a FRESH holder and a stale
/// deregistration can never remove the replacement's entry.
#[derive(Default)]
pub(crate) struct TranscriptRegistry {
    inner: Mutex<HashMap<String, TranscriptEntry>>,
}

impl TranscriptRegistry {
    /// Register a fresh holder for `agent_id` at the given generation and return it.
    /// Registering over an existing id replaces the old holder — a reader holding the old
    /// `Arc<TranscriptSnapshot>` keeps that immutable snapshot.
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

    /// Remove the entry only if its generation still matches, so a stale deregistration
    /// cannot remove a replacement run's holder.
    pub fn deregister(&self, agent_id: &str, generation: u64) {
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(entry) = map.get(agent_id)
            && entry.generation == generation
        {
            map.remove(agent_id);
        }
    }

    /// The current live snapshot for `agent_id`, if registered — an owned `Arc` the caller
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

    /// The generation is the identity of a run: a deregister carrying any other generation
    /// touches nothing, and only the matching one removes the entry.
    #[test]
    fn only_the_matching_generation_removes_an_entry() {
        let agent_id = format!("snap_dereg_{}", crate::generate_suffix());
        let generation = next_gen();
        let holder_a = TRANSCRIPT_REGISTRY.register(agent_id.clone(), generation);

        TRANSCRIPT_REGISTRY.deregister(&agent_id, generation + 1);
        assert!(
            TRANSCRIPT_REGISTRY.snapshot(&agent_id).is_some(),
            "stale deregister must not remove the entry"
        );

        // A replacement run registers a fresh holder with a new generation; the old run's
        // deregister (now stale) must NOT remove it.
        let gen_b = next_gen();
        let holder_b = TRANSCRIPT_REGISTRY.register(agent_id.clone(), gen_b);
        TRANSCRIPT_REGISTRY.deregister(&agent_id, generation);
        assert!(
            TRANSCRIPT_REGISTRY.snapshot(&agent_id).is_some(),
            "replacement entry survives the stale deregister"
        );

        TRANSCRIPT_REGISTRY.deregister(&agent_id, gen_b);
        assert!(
            TRANSCRIPT_REGISTRY.snapshot(&agent_id).is_none(),
            "matching deregister removes the entry"
        );
        let _ = (holder_a, holder_b);
    }
}
