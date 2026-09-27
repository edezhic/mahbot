//! Bounded retention of the output channels a process that outlived its command
//! still holds.
//!
//! A command that exits while a process it started still holds the command's
//! output channel is reported as FINISHED with that leftover named and left
//! RUNNING (it was started on purpose) — so the product keeps the read end open:
//! closing it would fault the leftover on its next write (SIGPIPE on unix, a
//! broken pipe on Windows) and kill the very process the report says is still
//! running.
//!
//! A reader is retained only when the call that started it ENDS, so the set holds
//! only channels that outlived their call — the cap can never abort the reader of
//! a call still in flight. Holding a channel costs a reader task and a file
//! descriptor, so the number held at once is bounded: past
//! [`MAX_RETAINED_CHANNELS`] the OLDEST is released (its reader aborted), which
//! closes that read end — a leftover whose next write hits the closed pipe can
//! then be terminated by the platform (SIGPIPE on unix, a broken pipe on Windows).
//! The newest is the one kept: it is what the caller was just told about.
//!
//! So the "left running" guarantee is what the product provides while it RETAINS
//! the channel: releasing it (past the cap, or when the product stops draining)
//! ends that. The cap's bound on descriptors is real; the leftover's survival is
//! not promised past it.
//!
//! The two readers that can outlive their command both retain through here: the
//! shell's capture readers and the chrome-use CLI's output pipes.

use std::collections::VecDeque;
use std::sync::{LazyLock, Mutex};

use crate::util::UnwrapPoison;

/// How many leftover output channels the product keeps open at once.
pub(crate) const MAX_RETAINED_CHANNELS: usize = 16;

/// The retained reader tasks, oldest first.
static RETAINED: LazyLock<Mutex<VecDeque<tokio::task::JoinHandle<()>>>> =
    LazyLock::new(|| Mutex::new(VecDeque::new()));

/// Keep a reader task draining after the call that started it has returned.
/// Readers whose channel already reached EOF are dropped first (they hold
/// nothing), then the oldest channel past [`MAX_RETAINED_CHANNELS`] is released.
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
