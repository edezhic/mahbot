//! In-process per-ticket dispatch-in-flight latch.
//!
//! The pipeline's only re-dispatch guard is [`crate::jobs::job_has_launched_agents`],
//! a non-atomic DB presence check. During a stage handoff — after the prior
//! stage's `launched` roster rows are checkpointed/cleared and before the next
//! stage's rows are written — that guard reads `false` while the ticket is still
//! in the pipeline-occupied source phase, so a concurrent 1s poll tick can
//! dispatch the same stage a second time.
//!
//! This latch is the in-process single-winner claim that closes that window: a
//! stage dispatch claims it synchronously before any `await`/`spawn`, holds it
//! for the whole dispatch, and releases it when the stage is fully finalized.
//! A redundant dispatch (poll re-dispatch, handoff, boot resume) fails the claim
//! and bails *before* cancelling a healthy agent or registering a roster row.
//!
//! It is deliberately in-memory (not a `tickets` column) so a crash clears it and
//! it stays outside `complete_ticket_job`/`clear_implementation_roster` wipe
//! semantics. The [`LatchGuard`] generation counter makes a stale release (a
//! superseded dispatch's guard) a no-op if the latch was already re-claimed by
//! the next stage — so a handoff never has its new claim cleared by the prior
//! stage's drop.

use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::Mutex;

use crate::util::UnwrapPoison;

/// A per-ticket dispatch-in-flight latch.
///
/// [`try_claim`](Self::try_claim) atomically marks a ticket as "a pipeline stage
/// dispatch is in flight" and returns a [`LatchGuard`]; a second claim for the
/// same ticket returns `None`. The guard releases the latch on drop (and on
/// explicit [`LatchGuard::release`]). Generations make a stale release a no-op so
/// a handoff that re-claims for the next stage is never cleared by the prior
/// stage's guard.
#[derive(Debug, Default)]
pub(crate) struct DispatchLatch {
    inner: Mutex<HashMap<String, LatchState>>,
}

#[derive(Debug, Default, Clone, Copy)]
struct LatchState {
    claimed: bool,
    generation: u64,
}

impl DispatchLatch {
    /// Try to claim the latch for `ticket_id`. Returns `Some(guard)` on success,
    /// `None` if a stage dispatch is already in flight for the ticket.
    #[must_use]
    pub(crate) fn try_claim(&self, ticket_id: &str) -> Option<LatchGuard<'_>> {
        let mut map = self.inner.lock().unwrap_poison();
        let state = map.entry(ticket_id.to_string()).or_default();
        if state.claimed {
            return None;
        }
        state.generation += 1;
        state.claimed = true;
        Some(LatchGuard {
            latch: self,
            ticket_id: ticket_id.to_string(),
            generation: state.generation,
            active: true,
        })
    }

    /// Release the claim for `ticket_id` iff it still belongs to `generation`.
    fn release(&self, ticket_id: &str, generation: u64) {
        let mut map = self.inner.lock().unwrap_poison();
        if let Some(state) = map.get_mut(ticket_id)
            && state.claimed
            && state.generation == generation
        {
            state.claimed = false;
        }
    }
}

/// The claim guard: holds the latch until dropped (or explicitly released).
///
/// Dropping the guard releases the latch, so a panic or early return never
/// strands a ticket. [`release`](Self::release) is the explicit handoff point at
/// which a finalized stage frees the latch for the next stage's claim; it is
/// idempotent with the drop.
#[must_use]
pub(crate) struct LatchGuard<'a> {
    latch: &'a DispatchLatch,
    ticket_id: String,
    generation: u64,
    active: bool,
}

impl LatchGuard<'_> {
    /// Explicitly release the latch (e.g. at a stage handoff before the next
    /// stage is dispatched). Idempotent with [`Drop`].
    pub(crate) fn release(mut self) {
        if self.active {
            self.latch.release(&self.ticket_id, self.generation);
            self.active = false;
        }
    }
}

impl Drop for LatchGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            self.latch.release(&self.ticket_id, self.generation);
        }
    }
}

/// The process-global latch. Starts empty at boot (in-memory), so a crash clears
/// any in-flight markers and the boot-resume path never sees a stale latch.
pub(crate) static LATCH: LazyLock<DispatchLatch> = LazyLock::new(DispatchLatch::default);

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn one_winner_among_concurrent_claims() {
        let latch = Arc::new(DispatchLatch::default());
        let ticket = "t1".to_string();

        // A held claim (simulating an in-flight stage dispatch) makes every
        // concurrent attempt a loser — exactly one winner. The winner's guard is
        // deliberately kept alive while all the other threads attempt.
        let winner = latch.try_claim(&ticket).expect("first claim wins");
        let mut handles = Vec::new();
        for _ in 0..8 {
            let latch = Arc::clone(&latch);
            let ticket = ticket.clone();
            handles.push(std::thread::spawn(move || {
                latch.try_claim(&ticket).is_some()
            }));
        }
        let losers = handles
            .into_iter()
            .map(|h| h.join().expect("claim thread"))
            .filter(|won| *won)
            .count();
        assert_eq!(
            losers, 0,
            "a held claim must block every concurrent attempt"
        );

        // Release (drop) the winner and confirm a fresh dispatch can claim.
        drop(winner);
        assert!(
            latch.try_claim(&ticket).is_some(),
            "released latch must be reclaimable"
        );
    }

    #[test]
    fn stale_release_does_not_clear_new_claim() {
        let latch = DispatchLatch::default();
        // First dispatch claims and explicitly releases at a handoff.
        let first = latch.try_claim("t1").expect("first claim");
        first.release();
        // The next stage claims a fresh generation.
        let second = latch.try_claim("t1").expect("second claim after handoff");
        // A stale release from a superseded (old generation) guard must not clear
        // the new claim.
        latch.release("t1", 1);
        assert!(
            latch.try_claim("t1").is_none(),
            "stale release must not clear the new claim"
        );
        drop(second);
        assert!(
            latch.try_claim("t1").is_some(),
            "latch must be free after the live guard drops"
        );
    }
}
