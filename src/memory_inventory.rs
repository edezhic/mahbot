//! Nightly memory inventory: once a night, per account, ask the admin Assistant to trim a
//! personal memory file that has outgrown the workspace file limit.
//!
//! At most one asking per account per local day (the day is recorded BEFORE the ask, so a crash
//! cannot ask twice), and accounts are asked one at a time — each queued turn is awaited before the
//! next account, so the passes never run side by side.

use std::time::Duration;

use chrono::Timelike;
use tracing::{debug, info, warn};

const INVENTORY_HOUR: u32 = 3;

/// Wake interval — the same half-hour tick the product's other nightly work
/// uses. The window is an hour wide, so a running daemon gets one or two ticks
/// inside it and a tick outside it does nothing.
const TICK: Duration = Duration::from_mins(30);

/// How long a turn may take to appear before an idle assistant counts as finished.
const TURN_START_WAIT: Duration = Duration::from_secs(30);

/// Hard bound on one queued turn: past it the pass moves to the next account rather than wait.
const TURN_WAIT: Duration = Duration::from_mins(10);

const TURN_POLL: Duration = Duration::from_secs(5);

/// Prompt asset holding the notice posted to the assistant.
const NOTICE_ASSET: &str = "memory_inventory_notice.md";

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
