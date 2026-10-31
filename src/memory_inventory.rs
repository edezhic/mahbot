//! The nightly memory inventory.
//!
//! Every account's Assistant keeps its own [`MEMORY_FILE`] beside the rest of
//! its personal folder, and a context render keeps only the first
//! [`MAX_WORKSPACE_FILE_CHARS`] of it. Once a night, inside the 03:00–03:59
//! local window, every account whose file is longer than that is sent one
//! ordinary message asking it to inventory and clean up what it keeps — its
//! memory, and the notes and working material it accumulated itself. Nothing
//! here edits a file: the work belongs to the Assistant, with the tools it
//! already has.
//!
//! At most one message per account per local day, recorded in
//! `users.memory_inventory_day` BEFORE the message is handed over, so a restart
//! cannot ask twice. A night lost to a shutdown, a sleeping machine or a crash
//! in that gap is not caught up on, and a file still over the limit is asked
//! about again on a later night.
//!
//! The window gates the tick, not the work a tick already started: a pass that
//! began before 04:00 still asks every account it found over the limit, the same
//! rule the product's other nightly work follows.
//!
//! [`MEMORY_FILE`]: crate::prompt::MEMORY_FILE
//! [`MAX_WORKSPACE_FILE_CHARS`]: crate::prompt::MAX_WORKSPACE_FILE_CHARS

use std::time::Duration;

use chrono::Timelike;
use tracing::{debug, info, warn};

/// The local hour the inventory window opens (03:00–03:59).
const INVENTORY_HOUR: u32 = 3;

/// Wake interval — the same half-hour tick the product's other nightly work
/// uses. The window is an hour wide, so a running daemon gets one or two ticks
/// inside it and a tick outside it does nothing.
const TICK: Duration = Duration::from_mins(30);

/// How long the hand-over is given to appear in the registry before the pass
/// stops waiting for it.
const TURN_START_WAIT: Duration = Duration::from_secs(30);

/// How long one assistant's turn is given once it is running, before the next
/// account is asked anyway.
const TURN_WAIT: Duration = Duration::from_mins(10);

/// How often [`wait_for_turn`] looks at the registry.
const TURN_POLL: Duration = Duration::from_secs(5);

/// The product-authored message asking for an inventory.
const NOTICE_ASSET: &str = "memory_inventory_notice.md";

/// Run the nightly memory inventory loop.
pub async fn run_memory_inventory_loop() {
    loop {
        if !crate::shutdown::sleep_or_shutdown_or_drain(TICK).await {
            break;
        }
        if chrono::Local::now().hour() != INVENTORY_HOUR || !crate::config::assistant_can_run() {
            continue;
        }
        inventory_pass().await;
    }
}

/// Measure every account's memory file and ask the assistants whose file no
/// longer fits into their context.
async fn inventory_pass() {
    let Some(store) = crate::users::USER_STORE.get() else {
        warn!("Memory inventory: USER_STORE not initialized — skipping this night");
        return;
    };
    let accounts = match store.list_users().await {
        Ok(accounts) => accounts,
        Err(e) => {
            warn!(error = %e, "Memory inventory: failed to list accounts");
            return;
        }
    };

    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    for entry in accounts {
        let user = entry.record.name;
        let asked_today = match store.memory_inventory_day(&user).await {
            Ok(day) => day.as_deref() == Some(today.as_str()),
            Err(e) => {
                // Without the record the at-most-once promise cannot be kept.
                warn!(user = %user, error = %e, "Memory inventory: failed to read the last asked day");
                continue;
            }
        };
        if asked_today {
            continue;
        }

        let Some(memory) = read_memory(&user).await else {
            continue;
        };
        // The render trims a file section before measuring it, so the trigger
        // measures the same text the context does.
        let memory = memory.trim();
        if !crate::prompt::exceeds_workspace_file_limit(memory) {
            continue;
        }

        // Record the day BEFORE the hand-over: a crash in between loses one
        // night, while the opposite order could ask the same assistant twice.
        if let Err(e) = store.set_memory_inventory_day(&user, &today).await {
            warn!(user = %user, error = %e, "Memory inventory: failed to record the day — not asking");
            continue;
        }

        let agent_id = crate::session::assistant_agent_id(&user);
        info!(
            user = %user,
            chars = memory.chars().count(),
            "Memory inventory: asking the assistant to clean up"
        );
        crate::agent::message_router::post_assistant_notice(
            &agent_id,
            &user,
            crate::prompt::load_prompt(NOTICE_ASSET),
        );
        if !wait_for_turn(&agent_id).await {
            return;
        }
    }
}

/// The account's memory file text, or `None` when it has none or the file is
/// not readable text — nothing to measure and nothing to ask about.
async fn read_memory(user: &str) -> Option<String> {
    let path = crate::users::personal_workspace_path(user).join(crate::prompt::MEMORY_FILE);
    tokio::fs::read_to_string(path).await.ok()
}

/// Wait until the assistant's queued turn is over, so the next account is asked
/// after it rather than beside it. `false` means the product is shutting down or
/// draining and the pass must stop.
///
/// The registry entry is created when the turn's `Agent` is built and dropped
/// when the turn ends, including the summary the session writes after delivery.
/// Two consecutive empty polls are what counts as idle — one can land in the
/// gap between two turns of the same consumer — and a notice that starts no turn
/// at all is given [`TURN_START_WAIT`] before the pass stops waiting for it.
async fn wait_for_turn(agent_id: &str) -> bool {
    let started = tokio::time::Instant::now();
    let mut running = false;
    let mut idle_polls = 0;
    loop {
        if !crate::shutdown::sleep_or_shutdown_or_drain(TURN_POLL).await {
            return false;
        }
        if crate::agent::registry::AGENT_REGISTRY.contains(agent_id) {
            running = true;
            idle_polls = 0;
        } else {
            idle_polls += 1;
            if idle_polls >= 2 && (running || started.elapsed() >= TURN_START_WAIT) {
                return true;
            }
        }
        if started.elapsed() >= TURN_WAIT {
            debug!(
                agent_id,
                "Memory inventory: turn is still running — moving on"
            );
            return true;
        }
    }
}
