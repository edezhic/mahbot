//! Bounded retention of the output channels a process that outlived its command still
//! holds: the chrome-use CLI's helper is deliberately left running, and the shell tool's
//! leftover survives only when the stop it makes at detection did not take.
//!
//! Closing such a read end would fault the leftover on its next write (SIGPIPE on unix, a
//! broken pipe on Windows) and kill the very process the caller was told about, so the
//! channel is kept open. Holding one costs a reader task and a descriptor, so the set is
//! capped: past [`MAX_RETAINED_CHANNELS`] the OLDEST is released (its reader aborted), the
//! newest is kept — it is what the caller was just told about. The "left running"
//! guarantee therefore holds only while the channel is RETAINED; the cap's bound on
//! descriptors is real, the leftover's survival past it is not promised.
//!
//! A reader is retained only when the call that started it ENDS, so the cap can never abort
//! the reader of a call still in flight.

use std::collections::VecDeque;
use std::sync::{LazyLock, Mutex};

use crate::util::UnwrapPoison;

pub(crate) const MAX_RETAINED_CHANNELS: usize = 16;

/// The retained reader tasks, oldest first.
static RETAINED: LazyLock<Mutex<VecDeque<tokio::task::JoinHandle<()>>>> =
    LazyLock::new(|| Mutex::new(VecDeque::new()));

/// Keep a reader task draining after the call that started it returned. Finished readers
/// are dropped first (they hold nothing), then the oldest past [`MAX_RETAINED_CHANNELS`].
pub(crate) fn retain_channel(task: tokio::task::JoinHandle<()>) {
    // A finished reader holds nothing; pushing it would let it displace — and
    // abort — an oldest still-live channel.
    if task.is_finished() {
        return;
    }
    let mut retained = RETAINED.lock().unwrap_poison();
    retained.retain(|task| !task.is_finished());
    retained.push_back(task);
    while retained.len() > MAX_RETAINED_CHANNELS {
        if let Some(oldest) = retained.pop_front() {
            oldest.abort();
        }
    }
}
