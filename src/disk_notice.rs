//! The built-in low-disk notice.
//!
//! When free space on the volume holding the daemon's temp root falls below 5%
//! of that volume's capacity, the admin assistant is told to start cleaning
//! there, and reminded on a widening ladder (`15 min → 1 h → 12 h → 24 h → 24 h
//! …`) until the space comes back. Nothing is cleaned here: the notice, and the
//! decision that follows it, belong to the assistant it addresses. The periodic
//! temp cleaner ([`crate::temp`]) keeps its own cadence and is not coupled to
//! this one.
//!
//! The check reads the same volume the temp root is pinned on, with
//! [`crate::util::disk::free_and_capacity`], so a tick is one filesystem read
//! plus one config read and never a model turn — a turn is spent only when a
//! notice is actually handed over.
//!
//! The ladder is persisted in `config_kv` ([`LADDER_KV_KEY`]) so a restart or an
//! automatic update neither resets the cadence nor causes an early repeat, and
//! mirrored in memory so a failed config write cannot turn the cadence into a
//! repeat on every tick. The record is settled only AFTER the notice has been
//! handed over, so a crash in between repeats a notice — the lesser fault than
//! silence.

use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use tracing::{debug, info, warn};

use crate::util::UnwrapPoison;

/// How often the cadence loop wakes. A tick is one filesystem read plus one
/// config read, so the wake is frequent enough for the first rung to land
/// within a minute of its 15 minutes.
const WAKE: Duration = Duration::from_mins(1);

/// Free space below this percentage of the volume's capacity is low.
const LOW_FREE_PCT: u64 = 5;

/// Prompt asset carrying the fixed notice text.
const NOTICE_PROMPT_KEY: &str = "disk_space_notice.md";

/// `config_kv` key holding the escalation ladder. Owned by this subsystem —
/// preserved by the config orphan purge.
pub(crate) const LADDER_KV_KEY: &str = "disk_notice_ladder";

/// One escalation's progress: how many notices this crossing has delivered and
/// when the last one left. An absent record means no crossing is in progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Ladder {
    /// Deliveries made since free space last came back above the threshold.
    /// `0` is the disarmed record a recovery leaves behind.
    deliveries: u32,
    /// Wall clock of the last delivery — the origin every interval counts from.
    delivered_at: DateTime<Utc>,
}

impl Ladder {
    fn encode(self) -> String {
        format!("{}@{}", self.deliveries, self.delivered_at.to_rfc3339())
    }

    fn parse(raw: &str) -> Option<Self> {
        let (deliveries, at) = raw.split_once('@')?;
        Some(Self {
            deliveries: deliveries.parse().ok()?,
            delivered_at: crate::db::parse_utc_timestamp(at).ok()?,
        })
    }

    /// The wait after the `deliveries`-th notice: 15 minutes, then one hour,
    /// then 12 hours, then 24 hours from the fourth delivery on. A disarmed
    /// record is due at once.
    fn next_due_in(self) -> ChronoDuration {
        match self.deliveries {
            0 => ChronoDuration::zero(),
            1 => ChronoDuration::minutes(15),
            2 => ChronoDuration::hours(1),
            3 => ChronoDuration::hours(12),
            _ => ChronoDuration::hours(24),
        }
    }

    /// Wall-clock, like the temp cleaner's own cadence: a record ahead of the
    /// clock waits for it rather than repeating a known delivery.
    fn is_due(self, now: DateTime<Utc>) -> bool {
        now.signed_duration_since(self.delivered_at) >= self.next_due_in()
    }

    /// One rung further, measured from `now` — a machine that was down across
    /// several rungs delivers one notice and advances one step, never a burst.
    fn advanced(self, now: DateTime<Utc>) -> Self {
        Self {
            deliveries: self.deliveries.saturating_add(1),
            delivered_at: now,
        }
    }

    /// Free space is back at or above the threshold: the next crossing starts
    /// again at the first rung.
    fn recovered(now: DateTime<Utc>) -> Self {
        Self {
            deliveries: 0,
            delivered_at: now,
        }
    }

    fn delivered(now: DateTime<Utc>) -> Self {
        Self {
            deliveries: 1,
            delivered_at: now,
        }
    }
}

/// The last state this process settled. The persisted record is authoritative
/// across restarts; this mirror keeps a failed config write from re-delivering
/// the notice on every tick. The two are merged by recency.
static MIRROR: Mutex<Option<Ladder>> = Mutex::new(None);

/// The low-disk notice's cadence loop, spawned once from the background-task
/// set. Panic posture matches the sibling periodic loops: the task-level guard
/// in `spawn_cancellable` logs and ends the loop, and the next boot resumes the
/// ladder from its persisted record.
pub async fn run_low_disk_notice_loop() {
    // The owner is told through the channels the assistant replies on, so a boot
    // waits for the transport the way the product's other boot notices do
    // (`self_update::announce_startup`): a self-hosted start opens before its
    // server is published, and a notice handed over then could only reach the
    // GUI. A wait that ends with the process is a notice nobody is left to read.
    if !crate::channels::telegram::wait_for_transport().await {
        return;
    }
    loop {
        tick().await;
        if !crate::shutdown::sleep_or_shutdown_or_drain(WAKE).await {
            break;
        }
    }
}

/// One tick's decision: deliver the notice, settle a recovery, or leave
/// everything as it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Idle,
    /// Free space is back at or above the threshold: re-arm the ladder.
    Rearm,
    /// Deliver one notice and settle this ladder.
    Notice(Ladder),
}

/// What a tick does with one reading: an unreadable one (a zero capacity is
/// not a reading of anything) freezes the escalation, a healthy one re-arms it,
/// and a low one delivers at most one notice, only while the ladder is due and
/// the installation can run an assistant turn.
fn step(
    free: u64,
    capacity: u64,
    ladder: Option<Ladder>,
    can_run: bool,
    now: DateTime<Utc>,
) -> Step {
    if capacity == 0 {
        return Step::Idle;
    }
    if !below_threshold(free, capacity) {
        return if ladder.is_some_and(|ladder| ladder.deliveries > 0) {
            Step::Rearm
        } else {
            Step::Idle
        };
    }
    if !can_run || !ladder.is_none_or(|ladder| ladder.is_due(now)) {
        return Step::Idle;
    }
    Step::Notice(ladder.map_or_else(|| Ladder::delivered(now), |ladder| ladder.advanced(now)))
}

/// One cadence tick: read the volume and act on one decision.
async fn tick() {
    let Some(store) = crate::config_db::CONFIG_STORE.get() else {
        return;
    };
    // The daemon pins its temp environment to the private root at startup
    // (`crate::temp::init_temp_root`), so this is the product's own temp
    // directory on every platform — the same lookup the temp cleaner measures.
    let temp_root = std::env::temp_dir();
    let Some((free, capacity)) =
        crate::util::with_block_in_place(|| crate::util::disk::free_and_capacity(&temp_root))
    else {
        return;
    };
    let persisted = match store.get_kv(LADDER_KV_KEY).await {
        Ok(Some(raw)) => {
            let ladder = Ladder::parse(&raw);
            if ladder.is_none() {
                // A row this build cannot read (an older or a newer encoding) is
                // no escalation: drop it, so it is not re-read and re-logged on
                // every tick.
                debug!(value = %raw, "Unparseable low-disk notice state — dropped");
                if let Err(e) = store.delete_kv(LADDER_KV_KEY).await {
                    debug!(error = %e, "Failed to drop the unparseable low-disk notice state");
                }
            }
            ladder
        }
        Ok(None) => None,
        Err(e) => {
            warn!(error = %e, "Failed to read the low-disk notice state — no notice is produced");
            return;
        }
    };
    let ladder = [persisted, *MIRROR.lock().unwrap_poison()]
        .into_iter()
        .flatten()
        .max_by_key(|ladder| ladder.delivered_at);
    let now = Utc::now();

    match step(free, capacity, ladder, assistant_can_run(), now) {
        Step::Idle => {}
        Step::Rearm => {
            settle(store, Ladder::recovered(now)).await;
            debug!(
                "Free space is back above the low-disk threshold — the notice ladder is re-armed"
            );
        }
        Step::Notice(ladder) => {
            deliver(free, capacity, &temp_root);
            settle(store, ladder).await;
        }
    }
}

/// Whether free space is below the notice's share of the volume's capacity.
fn below_threshold(free: u64, capacity: u64) -> bool {
    free < capacity.saturating_mul(LOW_FREE_PCT) / 100
}

/// Whether the installation can run an assistant turn at all. A notice that
/// lands before onboarding finishes or with no provider configured is a
/// guaranteed failing turn, so the check stays silent and the ladder does not
/// advance until the product can answer.
fn assistant_can_run() -> bool {
    crate::config::provider_configured()
        && crate::config::CONFIG.onboarding_stage() == crate::config::OnboardingState::Finished
}

/// Hand the notice to the admin assistant's ordinary queue: its reply reaches
/// the owner through the usual channels.
fn deliver(free: u64, capacity: u64, temp_root: &Path) {
    crate::agent::message_router::post_assistant_notice(
        &crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME),
        crate::users::ADMIN_USER_NAME,
        render_notice(free, capacity, temp_root),
    );
    info!(
        free,
        capacity, "Low disk space notice delivered to the admin assistant"
    );
}

/// Make `ladder` the current state: the in-memory mirror first (a failed write
/// must not repeat the notice on the next tick), then the durable record.
async fn settle(store: &crate::config_db::ConfigStore, ladder: Ladder) {
    *MIRROR.lock().unwrap_poison() = Some(ladder);
    if let Err(e) = store.set_kv(LADDER_KV_KEY, &ladder.encode()).await {
        warn!(error = %e, "Failed to record the low-disk notice state");
    }
}

/// The fixed notice, with the volume's figures and the measured temp root
/// substituted.
fn render_notice(free: u64, capacity: u64, temp_root: &Path) -> String {
    let free_text = crate::tools::listing::human_readable_size(free);
    let capacity_text = crate::tools::listing::human_readable_size(capacity);
    let temp_root_text = temp_root.display().to_string();
    crate::prompt::substitute(
        &crate::prompt::load_prompt(NOTICE_PROMPT_KEY),
        &[
            ("{{free}}", &free_text),
            ("{{capacity}}", &capacity_text),
            ("{{temp_root}}", &temp_root_text),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(minutes: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(0, 0).expect("epoch") + ChronoDuration::minutes(minutes)
    }

    fn notice(step: Step) -> Ladder {
        match step {
            Step::Notice(ladder) => ladder,
            other => panic!("expected a notice, got {other:?}"),
        }
    }

    #[test]
    fn one_low_reading_delivers_one_notice_per_rung() {
        let (free, capacity) = (40, 1000);
        let mut ladder = None;
        let mut delivered_at = 0;
        // The first rung has no wait: a low reading with no state delivers at
        // once, and every later rung waits its interval from the notice before.
        for (made, wait_minutes) in [
            (1, 0),
            (2, 15),
            (3, 60),
            (4, 12 * 60),
            (5, 24 * 60),
            (6, 24 * 60),
        ] {
            if made > 1 {
                assert_eq!(
                    step(
                        free,
                        capacity,
                        ladder,
                        true,
                        at(delivered_at + wait_minutes - 1)
                    ),
                    Step::Idle,
                    "rung {made} waited too little"
                );
            }
            let delivered = notice(step(
                free,
                capacity,
                ladder,
                true,
                at(delivered_at + wait_minutes),
            ));
            assert_eq!(delivered.deliveries, made, "each notice advances one rung");
            ladder = Some(delivered);
            delivered_at += wait_minutes;
        }
        assert_eq!(
            step(free, capacity, ladder, false, at(delivered_at + 24 * 60)),
            Step::Idle,
            "an installation that cannot run a turn stays silent"
        );
    }

    #[test]
    fn recovery_rearms_but_an_unreadable_reading_freezes() {
        let armed = Some(Ladder::delivered(at(0)));
        assert_eq!(step(50, 1000, armed, true, at(15)), Step::Rearm);
        assert_eq!(
            step(50, 1000, Some(Ladder::recovered(at(1))), true, at(2)),
            Step::Idle,
            "an already re-armed ladder is left alone"
        );
        assert_eq!(
            step(0, 0, armed, true, at(15)),
            Step::Idle,
            "a zero capacity is no reading: the escalation neither advances nor resets"
        );
    }

    #[test]
    fn the_ladder_round_trips_through_its_persisted_form() {
        let ladder = Ladder {
            deliveries: 3,
            delivered_at: at(4321),
        };
        assert_eq!(Ladder::parse(&ladder.encode()), Some(ladder));
        assert_eq!(Ladder::parse("garbage"), None);
        assert_eq!(Ladder::parse("3@not-a-time"), None);
    }

    #[test]
    fn the_notice_is_the_fixed_text_with_the_figures_filled_in() {
        let temp_root = Path::new("/tmp/mahbot");
        assert_eq!(
            render_notice(5 * 1024 * 1024 * 1024, 100 * 1024 * 1024 * 1024, temp_root),
            "<disk-space-notice>\n\
             Free space is below 5% on the volume holding mahbot's temp directory: 5.4G free of 107.4G.\n\
             Garbage left by agents piles up in /tmp/mahbot — start cleaning there.\n\
             Cleanup is the administrative assistant's job. Be careful not to touch files that running agents are using right now.\n\
             If cleaning brings free space back above 5%, say nothing. If it does not, tell the owner that space is running out and suggest what else could be cleaned.\n\
             </disk-space-notice>\n",
        );
    }
}
