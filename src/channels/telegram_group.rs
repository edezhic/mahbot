//! Burst collector for the Telegram inbound path.
//!
//! Telegram gives bots no "still typing" signal, so a short quiet window per
//! account and chat is the only available mechanism to fuse one burst into one
//! request; the collector hands the parts to the pipeline in send order.
//!
//! One quiet window is not enough on its own, because a forwarded attachment is
//! handed to the collector only once its fetch has finished while the comment
//! that came with it opens a group whose much shorter window keeps running. The
//! collector therefore holds a group while a message of the same sender in the
//! same chat is still being prepared ([`Preparations`]) — whatever that message
//! is, an attachment as much as a forwarded one — and takes the hold from that
//! preparation: the mark is cleared before the part is handed over, and the
//! group's window starts at that clear rather than at its last part — the moment
//! in between is what the window covers — so the part it produced lands inside it.
//! The ceiling yields to a held group for the same reason: a ceiling falling at
//! the end of a hold would cut the group before that part had been taken in. The
//! wait is the preparation's own — bounded by the transfer, which abandons a
//! fetch that overruns it — and the caps still bound the group's size: they release
//! a group whatever is being prepared for it, a burst that large being beyond what
//! one turn is for. Only that pair is affected; a group with nothing of its
//! sender's in flight keeps the window it has today.
//!
//! Every release — a lone message's included — is recorded durably (see
//! [`crate::channels::telegram_intake::BurstRecord`]).

use crate::ChannelMessage;
use crate::channels::telegram::{ListenerFrame, is_control_message};
use crate::channels::telegram_intake::{BurstRecord, Intake};
use crate::util::UnwrapPoison;
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;
use tokio::time::Instant;

/// Quiet window after a lone plain part — imperceptible against an assistant
/// turn, so a single message still feels immediate.
pub(super) const QUIET_WINDOW: Duration = Duration::from_secs(1);
/// Quiet window once the group already holds a second part or forwarded content:
/// the user is evidently still sending, so the longer wait is worth it.
const EXTENDED_WINDOW: Duration = Duration::from_secs(3);
/// Hard ceiling from the group's first part, so a long stream still moves.
const CEILING: Duration = Duration::from_secs(5);
/// Release the group as soon as it holds this many parts...
const MAX_PARTS: usize = 6;
/// ...or this many characters of raw part text, which keeps a burst that is
/// still being typed from growing without bound. Enrichment (captions,
/// transcriptions, native image parts) is added after this check, so this bounds
/// the user's own text; a part's attachment payload is bounded by the inbound
/// limits of one message, exactly as it is without grouping.
const MAX_CHARS: usize = 4000;

/// Boundary between the parts of a burst in the one request the assistant
/// receives. The parts are already in send order; the boundary is explicit
/// because a blank line alone is indistinguishable from the user's own
/// paragraph break.
const PART_SEPARATOR: &str = "\n\n--- next message ---\n\n";

/// How far the listener may run ahead of the pipeline: a stalled pipeline still
/// backpressures the poll loop through the collector, while one poll's worth of
/// slack absorbs a batch.
pub(crate) const QUEUE_CAPACITY: usize = 100;

/// The one attachment this listener is preparing right now, by sender and chat.
///
/// The mark is taken by `try_parse_attachment_message` — once the message is known
/// to be an attachment and its sender and chat are resolved — and dropped when that
/// call returns, so it covers the fetch and every refusal path in between, while a
/// message that is not an attachment also takes none. That pair's group is not
/// released while the mark is on it; nothing else reads it (see the module doc).
#[derive(Debug)]
pub(crate) struct Preparations {
    /// The pair being prepared, if any. The listener walks a batch in order and
    /// prepares one message at a time, so there is never more than one.
    preparing: std::sync::Mutex<Option<(String, String)>>,
    /// Signalled on every mark and clear. A `watch` rather than a `Notify` because
    /// what a waiter acts on is the state, not the notification: one that missed
    /// a change still reads the newest state on its next pass.
    changed: tokio::sync::watch::Sender<()>,
}

impl Preparations {
    pub(crate) fn new() -> Self {
        Self {
            preparing: std::sync::Mutex::new(None),
            changed: tokio::sync::watch::channel(()).0,
        }
    }

    /// A receiver that resolves on every mark and clear.
    pub(crate) fn watch(&self) -> tokio::sync::watch::Receiver<()> {
        self.changed.subscribe()
    }

    /// Mark `user`'s messages in `target` as being prepared until the returned
    /// guard is dropped.
    pub(crate) fn begin(&self, user: &str, target: &str) -> Preparation<'_> {
        let key = (user.to_string(), target.to_string());
        let replaced = self.preparing.lock().unwrap_poison().replace(key.clone());
        debug_assert!(
            replaced.is_none(),
            "two preparations at once: the listener prepares one message at a time"
        );
        self.bump();
        Preparation {
            preparations: self,
            key,
        }
    }

    /// Whether anything of `user` in `target` is being prepared right now.
    fn busy(&self, user: &str, target: &str) -> bool {
        self.preparing
            .lock()
            .unwrap_poison()
            .as_ref()
            .is_some_and(|(u, t)| u.as_str() == user && t.as_str() == target)
    }

    /// Signal the change, so a waiter re-reads the state.
    fn bump(&self) {
        self.changed.send_modify(|()| {});
    }
}

/// One preparation in flight; dropping it clears the mark.
pub(crate) struct Preparation<'a> {
    preparations: &'a Preparations,
    key: (String, String),
}

impl Drop for Preparation<'_> {
    fn drop(&mut self) {
        let mut preparing = self.preparations.preparing.lock().unwrap_poison();
        // The mark is this guard's to clear — unless a later one has taken the
        // pair over, which the mark being a single pair cannot normally happen.
        if preparing.as_ref() == Some(&self.key) {
            *preparing = None;
        }
        drop(preparing);
        self.preparations.bump();
    }
}

/// One collected part: the message, and the moment the collector took it in —
/// the wall clock the release's record keeps, read here rather than at the
/// release so the record names the part's own moment.
struct Part {
    message: ChannelMessage,
    taken_at: SystemTime,
}

impl Part {
    fn new(message: ChannelMessage) -> Self {
        Self {
            message,
            taken_at: SystemTime::now(),
        }
    }
}

/// One burst: the parts of one account in one chat/thread, in send order.
/// Messages the same account sends in two chats (or two forum topics) never land
/// in one group.
struct OpenGroup {
    parts: Vec<Part>,
    chars: usize,
    first_at: Instant,
    last_at: Instant,
    /// A preparation of this group's sender in this chat was in flight while the
    /// group was open, so the group is not due however long its own window runs.
    held: bool,
    /// When the last preparation for this group's pair ended: the quiet window
    /// runs from there, not from the last part, so a part that took seconds to
    /// fetch does not shorten the wait for the companion message that follows.
    held_until: Option<Instant>,
}

impl OpenGroup {
    /// Start a group from its first part.
    fn new(part: ChannelMessage, now: Instant) -> Self {
        let chars = part.content.chars().count();
        Self {
            parts: vec![Part::new(part)],
            chars,
            first_at: now,
            last_at: now,
            held: false,
            held_until: None,
        }
    }

    /// `(user_name, reply_target)`. Every part of a group carries the same
    /// addressing — it is the key — so the group stores no copy of it.
    fn key(&self) -> (&str, &str) {
        let first = &self.parts[0].message;
        (first.user_name.as_str(), first.reply_target.as_str())
    }

    fn matches(&self, part: &ChannelMessage) -> bool {
        self.key() == (part.user_name.as_str(), part.reply_target.as_str())
    }

    /// Fold one more part in and report the cap it reached, if any.
    fn add(&mut self, part: ChannelMessage, now: Instant) -> Option<Release> {
        self.chars += part.content.chars().count();
        self.last_at = now;
        self.parts.push(Part::new(part));
        if self.parts.len() >= MAX_PARTS {
            Some(Release::PartCap)
        } else if self.chars > MAX_CHARS {
            Some(Release::CharCap)
        } else {
            None
        }
    }

    /// The user is evidently still sending — the group already holds a second
    /// part or forwarded content anywhere in it — so the extended window applies.
    fn extended(&self) -> bool {
        self.parts.len() > 1 || self.parts.iter().any(|part| part.message.forwarded)
    }

    /// When the group is due and which rule decides it: the window from the last
    /// part, or from the end of the last preparation for this pair once there was
    /// one, and — for a group that was never held — the ceiling from the first part
    /// (see the module doc for why a held group waits out its window instead).
    fn due(&self) -> (Instant, Release) {
        let baseline = self
            .held_until
            .map_or(self.last_at, |ended| ended.max(self.last_at));
        let extended = self.extended();
        let windowed = baseline
            + if extended {
                EXTENDED_WINDOW
            } else {
                QUIET_WINDOW
            };
        let ceiling = self.first_at + CEILING;
        if self.held_until.is_some() || windowed <= ceiling {
            (
                windowed,
                if extended {
                    Release::Extended
                } else {
                    Release::Quiet
                },
            )
        } else {
            (ceiling, Release::Ceiling)
        }
    }
}

/// What released a group: the `telegram_bursts.released_by` value, and the
/// answer to "which window applied" for the split a complaint is about.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Release {
    /// The short window a lone part that is not forwarded gets.
    Quiet,
    /// The longer window once the group holds a second part or forwarded content.
    Extended,
    /// The ceiling from the group's first part.
    Ceiling,
    /// The group reached [`MAX_PARTS`].
    PartCap,
    /// The group reached [`MAX_CHARS`].
    CharCap,
    /// Control input in that chat released the group ahead of itself.
    Control,
    /// The listener is going away.
    Stop,
}

impl Release {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Quiet => "quiet_window",
            Self::Extended => "extended_window",
            Self::Ceiling => "ceiling",
            Self::PartCap => "part_cap",
            Self::CharCap => "char_cap",
            Self::Control => "control",
            Self::Stop => "stop",
        }
    }
}

/// One released group: its parts, and what released it.
struct Released {
    parts: Vec<Part>,
    release: Release,
}

impl Released {
    /// The one message the burst becomes: the parts in send order (each keeping
    /// its own content, markers, attachments and reply reference until the
    /// pipeline expands them) plus the addressing the pipeline routes with,
    /// taken from the newest part — every part carries the same one, so any of
    /// them speaks for the group. `content` belongs to the parts. A group of one
    /// is that message unchanged, moved rather than copied.
    fn into_message(self) -> ChannelMessage {
        let parts: Vec<ChannelMessage> = self.parts.into_iter().map(|part| part.message).collect();
        match <[ChannelMessage; 1]>::try_from(parts) {
            Ok([only]) => only,
            Err(parts) => group_message(parts),
        }
    }

    /// The durable record of this release (see [`Intake::record_burst`]).
    fn record(&self, released_at: SystemTime) -> BurstRecord {
        // A released group always holds at least one part (see `OpenGroup`).
        let newest = self
            .parts
            .last()
            .expect("a released group holds at least one part");
        let part_times: Vec<serde_json::Value> = self
            .parts
            .iter()
            .map(|part| {
                serde_json::json!({
                    "sent_at": part
                        .message
                        .update_date
                        .and_then(|secs| chrono::DateTime::from_timestamp(secs, 0))
                        .map(|at| at.to_rfc3339()),
                    "taken_at": chrono::DateTime::<chrono::Utc>::from(part.taken_at)
                        .to_rfc3339(),
                })
            })
            .collect();
        BurstRecord {
            user_name: newest.message.user_name.clone(),
            reply_target: newest.message.reply_target.clone(),
            parts: self.parts.len(),
            released_by: self.release.as_str(),
            released_at: chrono::DateTime::<chrono::Utc>::from(released_at).to_rfc3339(),
            part_times: serde_json::Value::Array(part_times).to_string(),
        }
    }
}

/// Collects one burst per account+chat. Every method takes the current instant
/// so the timing policy is testable without sleeping.
#[derive(Default)]
struct Collector {
    open: Vec<OpenGroup>,
}

impl Collector {
    /// Add one part of an account's burst. Returns the group when it must be
    /// released now (it reached the part or character cap).
    fn push(&mut self, part: ChannelMessage, now: Instant) -> Option<Released> {
        let Some(at) = self.open.iter().position(|group| group.matches(&part)) else {
            // A fresh group holds this one part, so neither cap is reached yet.
            self.open.push(OpenGroup::new(part, now));
            return None;
        };
        let release = self.open[at].add(part, now)?;
        Some(self.take_at(at, release))
    }

    /// Note the preparations in flight onto the open groups: a group whose pair
    /// is being prepared is held, and one whose preparation has just ended takes
    /// its window from that moment.
    fn observe(&mut self, preparations: &Preparations, now: Instant) {
        for group in &mut self.open {
            let (user, target) = group.key();
            if preparations.busy(user, target) {
                group.held = true;
            } else if group.held {
                group.held = false;
                group.held_until = Some(now);
            }
        }
    }

    /// The earliest group deadline, `None` when every open group is held.
    fn deadline(&self) -> Option<Instant> {
        self.open
            .iter()
            .filter(|group| !group.held)
            .map(|group| group.due().0)
            .min()
    }

    /// Every group whose window has closed. A held group is never due.
    fn release_due(&mut self, now: Instant) -> Vec<Released> {
        self.take_where(|group| {
            if group.held {
                return None;
            }
            let (due_at, release) = group.due();
            (due_at <= now).then_some(release)
        })
    }

    /// Every open group of one account in one chat/thread — the group a control
    /// message arriving in that chat is handed over ahead of, so a state-clearing
    /// command does not land in front of it. Order is preserved only as enqueue
    /// order: the group is put on the pipeline before the control message is.
    fn release_chat(&mut self, user_name: &str, reply_target: &str) -> Vec<Released> {
        self.take_where(|group| {
            (group.key() == (user_name, reply_target)).then_some(Release::Control)
        })
    }

    /// Every open group — the listener is going away.
    fn release_all(&mut self) -> Vec<Released> {
        self.take_where(|_| Some(Release::Stop))
    }

    /// Take out every group `pred` releases, in the order the groups were opened.
    fn take_where(&mut self, pred: impl Fn(&OpenGroup) -> Option<Release>) -> Vec<Released> {
        let mut ready = Vec::new();
        let mut at = 0;
        while at < self.open.len() {
            if let Some(release) = pred(&self.open[at]) {
                ready.push(self.take_at(at, release));
            } else {
                at += 1;
            }
        }
        ready
    }

    /// Take the group at `at`.
    fn take_at(&mut self, at: usize, release: Release) -> Released {
        Released {
            parts: self.open.remove(at).parts,
            release,
        }
    }
}

/// The single message a burst becomes: the parts in send order (each keeping its
/// own text, markers, attachments and reply reference until the pipeline expands
/// them) plus the addressing the pipeline routes with, taken from the newest
/// part — every part carries the same one, so any of them speaks for the group.
/// `content` belongs to the parts. The record entries the parts stand for are
/// moved onto the envelope, which is what the hand-over settles: a part does not
/// keep entries the envelope already carries.
fn group_message(mut parts: Vec<ChannelMessage>) -> ChannelMessage {
    // A released group always holds at least one part (see `OpenGroup`).
    let newest = parts
        .last()
        .expect("a released group holds at least one part");
    let user_name = newest.user_name.clone();
    let reply_target = newest.reply_target.clone();
    let channel = newest.channel.clone();
    let update_ids = parts
        .iter_mut()
        .flat_map(|part| std::mem::take(&mut part.update_ids))
        .collect();
    ChannelMessage {
        user_name,
        reply_target,
        channel,
        parts,
        update_ids,
        ..Default::default()
    }
}

/// The one request the assistant receives for a collected burst: the parts in
/// send order with an explicit boundary between them. A lone message is its own
/// content, moved out rather than copied.
#[must_use]
pub fn compose_group_content(parts: Vec<ChannelMessage>) -> String {
    match <[ChannelMessage; 1]>::try_from(parts) {
        Ok([only]) => only.content,
        Err(parts) => parts
            .iter()
            .map(|part| part.content.as_str())
            .collect::<Vec<_>>()
            .join(PART_SEPARATOR),
    }
}

/// Spawn the collector that turns the listener's stream of inbound messages into
/// one message per burst. Its input sender lives in the listener, so dropping it
/// — the listener exiting, which is what a bot-token hot reload does — releases
/// an open group immediately instead of waiting out its window.
///
/// The collector owns the listener's frame for its whole body: a listener whose
/// future is dropped or that panics still leaves the frame open until its
/// collector has handed on (and settled) the burst it held.
pub(crate) fn spawn_collector(
    input: mpsc::Receiver<ChannelMessage>,
    output: mpsc::Sender<ChannelMessage>,
    intake: Intake,
    preparations: std::sync::Arc<Preparations>,
    frame: ListenerFrame,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(run_collector(input, output, intake, preparations, frame))
}

async fn run_collector(
    mut input: mpsc::Receiver<ChannelMessage>,
    output: mpsc::Sender<ChannelMessage>,
    intake: Intake,
    preparations: std::sync::Arc<Preparations>,
    _frame: ListenerFrame,
) {
    let mut collector = Collector::default();
    let mut holds_changed = preparations.watch();
    loop {
        // Everything the listener has already handed over belongs to the group it
        // was sent to: a part that arrives in the same instant a window closes
        // must not be split off from it.
        loop {
            match input.try_recv() {
                Ok(msg) => {
                    if !absorb(&mut collector, &output, &intake, msg).await {
                        return;
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    hand_over_all(&output, &intake, collector.release_all()).await;
                    return;
                }
            }
        }
        let now = Instant::now();
        collector.observe(&preparations, now);
        if !hand_over_all(&output, &intake, collector.release_due(now)).await {
            return;
        }
        tokio::select! {
            () = wait_until(collector.deadline()) => {}
            // The end of a preparation is what starts a held group's window;
            // without this the collector would sleep past it.
            Ok(()) = holds_changed.changed() => {}
            msg = input.recv() => {
                let Some(msg) = msg else {
                    hand_over_all(&output, &intake, collector.release_all()).await;
                    return;
                };
                if !absorb(&mut collector, &output, &intake, msg).await {
                    return;
                }
            }
        }
    }
}

/// Take one part from the listener: control input is never collected — the group
/// open in that chat goes on the pipeline ahead of it — and joins no burst, so
/// it is handed over without a record; everything else joins the burst of its
/// sender in that chat.
async fn absorb(
    collector: &mut Collector,
    output: &mpsc::Sender<ChannelMessage>,
    intake: &Intake,
    msg: ChannelMessage,
) -> bool {
    if is_control_message(&msg) {
        if !hand_over_all(
            output,
            intake,
            collector.release_chat(&msg.user_name, &msg.reply_target),
        )
        .await
        {
            return false;
        }
        // Control input is never collected: there is no group to record.
        return send_to_pipeline(output, intake, msg).await;
    }
    if let Some(released) = collector.push(msg, Instant::now())
        && !hand_over(output, intake, released).await
    {
        return false;
    }
    true
}

/// Sleep until `deadline`, or forever when nothing is held.
async fn wait_until(deadline: Option<Instant>) {
    match deadline {
        Some(at) => tokio::time::sleep_until(at).await,
        None => std::future::pending().await,
    }
}

/// Hand every released group to the pipeline, in order. Returns `false` when the
/// pipeline is gone.
async fn hand_over_all(
    output: &mpsc::Sender<ChannelMessage>,
    intake: &Intake,
    groups: Vec<Released>,
) -> bool {
    for group in groups {
        if !hand_over(output, intake, group).await {
            return false;
        }
    }
    true
}

/// Hand one released group to the pipeline: its parts as one message (a group of
/// one is that message unchanged, moved rather than copied), and — once the
/// message is on the queue — the settlement of the record entries it stands for
/// and the durable record of the release. Returns `false` when the pipeline is
/// gone.
async fn hand_over(
    output: &mpsc::Sender<ChannelMessage>,
    intake: &Intake,
    released: Released,
) -> bool {
    let record = released.record(SystemTime::now());
    if released.parts.len() > 1 {
        tracing::info!(
            user_name = %record.user_name,
            parts = record.parts,
            released_by = record.released_by,
            "Telegram burst: releasing collected messages as one request"
        );
    }
    if !send_to_pipeline(output, intake, released.into_message()).await {
        return false;
    }
    intake.record_burst(&record).await;
    true
}

/// Put one message on the pipeline and settle the record entries it carries.
/// Returns `false` when the pipeline is gone — logged, because a released group
/// that cannot be handed over is the one message the collector itself gives up;
/// a group that was not handed over is not recorded either (its events stay held
/// in the intake's record and come back).
async fn send_to_pipeline(
    output: &mpsc::Sender<ChannelMessage>,
    intake: &Intake,
    mut msg: ChannelMessage,
) -> bool {
    let settled = std::mem::take(&mut msg.update_ids);
    if output.send(msg).await.is_ok() {
        intake.settle(&settled).await;
        return true;
    }
    tracing::warn!(
        "Telegram burst: the message pipeline is gone; the released group stays held in the intake's record"
    );
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal burst part: only the fields the collector keys and merges on.
    fn part(user: &str, target: &str, content: &str) -> ChannelMessage {
        ChannelMessage {
            user_name: user.to_string(),
            reply_target: target.to_string(),
            channel: "telegram".to_string(),
            content: content.to_string(),
            ..Default::default()
        }
    }

    /// A frame of its own for the collector to own: the check has no listener, so
    /// nothing waits for it to end.
    fn test_frame() -> ListenerFrame {
        ListenerFrame::enter(
            &std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            &std::sync::Arc::new(tokio::sync::Notify::new()),
        )
    }

    /// No message is being prepared in these checks, so the collector holds no
    /// group on a preparation's account.
    fn no_preparations() -> std::sync::Arc<Preparations> {
        std::sync::Arc::new(Preparations::new())
    }

    #[test]
    fn lone_plain_part_releases_after_quiet_window() {
        let mut collector = Collector::default();
        let t0 = Instant::now();
        assert!(collector.push(part("alice", "chat", "hello"), t0).is_none());
        assert_eq!(collector.deadline(), Some(t0 + Duration::from_secs(1)));

        assert!(
            collector
                .release_due(t0 + Duration::from_millis(999))
                .is_empty()
        );
        let released = collector.release_due(t0 + Duration::from_secs(1));
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].parts.len(), 1);
        assert_eq!(released[0].parts[0].message.content, "hello");
        assert_eq!(released[0].release, Release::Quiet);
    }

    #[test]
    fn second_part_extends_the_window() {
        let mut collector = Collector::default();
        let t0 = Instant::now();
        collector.push(part("alice", "chat", "one"), t0);
        collector.push(
            part("alice", "chat", "two"),
            t0 + Duration::from_millis(500),
        );
        assert_eq!(
            collector.deadline(),
            Some(t0 + Duration::from_millis(500) + EXTENDED_WINDOW)
        );

        // Forwarded content alone already counts as "still sending".
        let mut forwarded = Collector::default();
        let mut forwarded_part = part("alice", "chat", "hi");
        forwarded_part.forwarded = true;
        forwarded.push(forwarded_part, t0);
        assert_eq!(forwarded.deadline(), Some(t0 + EXTENDED_WINDOW));
    }

    #[test]
    fn deadline_never_exceeds_the_ceiling() {
        let mut collector = Collector::default();
        let t0 = Instant::now();
        collector.push(part("alice", "chat", "one"), t0);
        collector.push(part("alice", "chat", "two"), t0 + Duration::from_secs(2));
        collector.push(part("alice", "chat", "three"), t0 + Duration::from_secs(4));
        assert_eq!(collector.deadline(), Some(t0 + CEILING));
    }

    #[test]
    fn part_cap_releases_the_group() {
        let mut collector = Collector::default();
        let t0 = Instant::now();
        for _ in 0..MAX_PARTS - 1 {
            assert!(collector.push(part("alice", "chat", "x"), t0).is_none());
        }
        let released = collector
            .push(part("alice", "chat", "x"), t0)
            .expect("reaching the part cap must release");
        assert_eq!(released.parts.len(), MAX_PARTS);
        assert_eq!(released.release, Release::PartCap);
    }

    #[test]
    fn character_cap_releases_the_group() {
        let mut collector = Collector::default();
        let t0 = Instant::now();
        let half = "x".repeat(MAX_CHARS / 2 + 1);
        assert!(collector.push(part("alice", "chat", &half), t0).is_none());
        let released = collector
            .push(part("alice", "chat", &half), t0)
            .expect("exceeding the character cap must release");
        assert_eq!(released.parts.len(), 2);
        assert_eq!(released.release, Release::CharCap);

        // A group that stays within the cap keeps its window.
        let half = "x".repeat(MAX_CHARS / 2);
        let mut at_cap = Collector::default();
        at_cap.push(part("alice", "chat", &half), t0);
        assert!(at_cap.push(part("alice", "chat", &half), t0).is_none());
        assert!(at_cap.deadline().is_some());
    }

    #[test]
    fn groups_are_keyed_by_user_and_chat() {
        let t0 = Instant::now();
        let mut collector = Collector::default();
        collector.push(part("alice", "chat-a", "a1"), t0);
        collector.push(part("bob", "chat-a", "b1"), t0);
        collector.push(part("alice", "chat-b", "a2"), t0);
        collector.push(part("alice", "chat-a", "a3"), t0);

        // A control message's chat releases that conversation's group only: the
        // other chats' half-collected bursts keep waiting for their own window.
        let released = collector.release_chat("alice", "chat-a");
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].parts.len(), 2);
        assert_eq!(released[0].parts[0].message.content, "a1");
        assert_eq!(released[0].parts[1].message.content, "a3");
        assert_eq!(released[0].release, Release::Control);

        // Draining leaves nothing held and nothing pending.
        assert_eq!(collector.release_all().len(), 2);
        assert!(collector.release_all().is_empty());
        assert_eq!(collector.deadline(), None);
    }

    #[test]
    fn compose_group_content_joins_parts_in_order() {
        let parts = vec![part("alice", "chat", "one"), part("alice", "chat", "two")];
        assert_eq!(
            compose_group_content(parts),
            format!("one{PART_SEPARATOR}two")
        );

        let single = vec![part("alice", "chat", "solo")];
        assert_eq!(compose_group_content(single), "solo");
    }

    /// Await the next message the collector releases. Fails rather than hanging
    /// when it releases nothing; both callers below are deterministic and must
    /// not need the guard.
    async fn released(rx: &mut mpsc::Receiver<ChannelMessage>) -> ChannelMessage {
        tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("the collector released nothing")
            .expect("the collector stopped early")
    }

    /// An intake of its own, recording into a store no other test touches, plus
    /// that store and its backing directory — the directory must stay bound, or
    /// the store writes to an unlinked file.
    async fn test_intake() -> (
        Intake,
        crate::channels::telegram_intake::TelegramIntakeStore,
        tempfile::TempDir,
    ) {
        let (store, dir) = crate::open_test_store!(
            crate::channels::telegram_intake::TelegramIntakeStore,
            "telegram_intake"
        );
        let intake = Intake::new(store.clone(), std::sync::Arc::from("run-test"), "42");
        (intake, store, dir)
    }

    #[tokio::test]
    async fn collector_hands_a_command_over_after_the_open_group() {
        let (input, input_rx) = mpsc::channel(QUEUE_CAPACITY);
        let (output, mut output_rx) = mpsc::channel(8);
        let (intake, _store, _dir) = test_intake().await;
        spawn_collector(input_rx, output, intake, no_preparations(), test_frame());

        input.send(part("alice", "chat", "one")).await.unwrap();
        input.send(part("alice", "chat", "two")).await.unwrap();
        input.send(part("alice", "chat", "/pause")).await.unwrap();

        let group = released(&mut output_rx).await;
        assert_eq!(group.parts.len(), 2);
        assert!(group.content.is_empty());
        let command = released(&mut output_rx).await;
        assert_eq!(command.content, "/pause");
    }

    #[tokio::test]
    async fn collector_releases_the_intact_group_when_the_listener_goes_away() {
        let (intake, store, _dir) = test_intake().await;
        let (input, input_rx) = mpsc::channel(QUEUE_CAPACITY);
        let (output, mut output_rx) = mpsc::channel(8);
        spawn_collector(
            input_rx,
            output,
            intake.clone(),
            no_preparations(),
            test_frame(),
        );

        let mut first = part("alice", "chat", "[Forwarded from @bob] one");
        first.forwarded = true;
        first.attachment_dirs = vec!["dir-one".to_string()];
        first.update_ids = vec![11];
        let mut second = part("alice", "chat", "two");
        second.attachment_dirs = vec!["dir-two".to_string()];
        second.update_ids = vec![12];
        for id in [11, 12] {
            intake
                .hold(&[&serde_json::json!({ "update_id": id })])
                .await
                .expect("hold");
        }
        input.send(first).await.unwrap();
        input.send(second).await.unwrap();
        drop(input);

        // The burst is released whole: the parts in send order, each with its own
        // content, markers and attachments, under the addressing the pipeline
        // routes with.
        let group = released(&mut output_rx).await;
        assert_eq!(group.parts.len(), 2);
        assert_eq!(group.parts[0].content, "[Forwarded from @bob] one");
        assert!(
            group.parts[0].forwarded,
            "the forwarded fact must ride through the collector, not just its label"
        );
        assert_eq!(group.parts[0].attachment_dirs, vec!["dir-one".to_string()]);
        assert_eq!(group.parts[1].content, "two");
        assert_eq!(group.parts[1].attachment_dirs, vec!["dir-two".to_string()]);
        assert!(group.content.is_empty());
        assert_eq!(group.user_name, "alice");
        assert_eq!(group.reply_target, "chat");
        assert_eq!(group.channel, "telegram");
        // The record entries the parts stand for move onto the envelope, and the
        // hand-over takes the envelope's off there to settle them: neither the
        // parts nor the message that reaches the pipeline keep them. That the
        // events really were settled is what the take-over at the end of this
        // check pins.
        assert!(group.parts[0].update_ids.is_empty());
        assert!(group.parts[1].update_ids.is_empty());
        assert!(group.update_ids.is_empty());
        assert_eq!(
            compose_group_content(group.parts.clone()),
            format!("[Forwarded from @bob] one{PART_SEPARATOR}two")
        );
        // The collector leaves with the listener — which is also the point by
        // which the hand-over has settled the events: the released burst is
        // resolved in the record, so a later run has nothing to take up.
        assert!(output_rx.recv().await.is_none());
        assert!(
            Intake::new(store.clone(), std::sync::Arc::from("later-run"), "42")
                .take_over()
                .await
                .expect("take over")
                .is_empty(),
            "the hand-over must settle the events so no later run takes them up"
        );
    }

    /// A group of a sender still being prepared is held however long its own
    /// window runs, and the window then starts at the end of the preparation, so
    /// the part the fetch produces lands inside it.
    #[test]
    fn a_group_is_held_while_its_sender_is_still_being_prepared() {
        let preparations = no_preparations();
        let mut collector = Collector::default();
        let t0 = Instant::now();
        collector.push(part("alice", "chat", "one"), t0);

        let guard = preparations.begin("alice", "chat");
        collector.observe(&preparations, t0);
        assert!(
            collector.deadline().is_none(),
            "a group whose pair is being prepared must have no deadline"
        );
        assert!(
            collector
                .release_due(t0 + Duration::from_secs(30))
                .is_empty(),
            "however long its own window has run, a held group must not be released"
        );

        // The fetch ends: the part that took seconds lands in the same group, and
        // the group's window runs from the end of the preparation.
        drop(guard);
        collector.observe(&preparations, t0 + Duration::from_secs(2));
        collector.push(part("alice", "chat", "two"), t0 + Duration::from_secs(2));
        assert_eq!(
            collector.deadline(),
            Some(t0 + Duration::from_secs(2) + EXTENDED_WINDOW)
        );
        let released = collector.release_due(t0 + Duration::from_secs(2) + EXTENDED_WINDOW);
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].parts.len(), 2);
        assert_eq!(released[0].release, Release::Extended);
    }

    /// A preparation is scoped to its own pair: another sender's group in the
    /// same chat keeps its own window and is released on it, while the prepared
    /// sender's own group waits.
    #[test]
    fn a_preparation_holds_only_the_pair_it_belongs_to() {
        let preparations = no_preparations();
        let mut collector = Collector::default();
        let t0 = Instant::now();
        collector.push(part("alice", "chat-a", "a1"), t0);
        collector.push(part("bob", "chat-a", "b1"), t0);
        let _guard = preparations.begin("bob", "chat-a");
        collector.observe(&preparations, t0);

        // Alice's group is not bob's: it keeps its own window.
        assert_eq!(collector.deadline(), Some(t0 + QUIET_WINDOW));
        let released = collector.release_due(t0 + QUIET_WINDOW);
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].parts[0].message.user_name, "alice");

        // Bob's own group waits for his fetch, however long alice's window ran.
        assert!(collector.deadline().is_none());
        assert!(
            collector
                .release_due(t0 + Duration::from_secs(30))
                .is_empty()
        );
    }

    /// The ceiling does not cut a group that was held: a preparation that ends
    /// past the ceiling releases the group after its own window from that end.
    #[test]
    fn the_ceiling_waits_with_a_held_group() {
        let preparations = no_preparations();
        let mut collector = Collector::default();
        let t0 = Instant::now();
        collector.push(part("alice", "chat", "one"), t0);
        let guard = preparations.begin("alice", "chat");
        collector.observe(&preparations, t0);
        drop(guard);

        let ended = t0 + CEILING + Duration::from_secs(1);
        collector.observe(&preparations, ended);
        assert_eq!(collector.deadline(), Some(ended + QUIET_WINDOW));
    }

    /// A clear is not a release: the mark is dropped before the part it produced is
    /// handed over, and the group's window starts at the clear, so a collector pass
    /// landing in the moment between takes the group as waiting rather than
    /// releasing it without that part.
    #[test]
    fn a_cleared_preparation_does_not_release_the_group_it_held() {
        let preparations = no_preparations();
        let mut collector = Collector::default();
        let t0 = Instant::now();
        collector.push(part("alice", "chat", "one"), t0);
        let guard = preparations.begin("alice", "chat");
        collector.observe(&preparations, t0);

        // The preparation ends; its part is still on its way to the collector.
        drop(guard);
        let cleared = t0 + Duration::from_secs(2);
        collector.observe(&preparations, cleared);
        assert!(
            collector.release_due(cleared).is_empty(),
            "the group must not be released before the part the preparation produced arrives"
        );
        assert_eq!(collector.deadline(), Some(cleared + QUIET_WINDOW));

        // The part arrives a moment later: it lands in the group it belongs to.
        collector.push(part("alice", "chat", "two"), cleared);
        let released = collector.release_due(cleared + EXTENDED_WINDOW);
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].parts.len(), 2);
        assert_eq!(released[0].release, Release::Extended);
    }

    /// The durable record of a release: who and where from the newest part, how
    /// many parts, which rule released it, and per part the platform's own time
    /// (null when there is none) and the moment the collector took it in.
    #[test]
    fn the_record_names_each_part_and_what_released_it() {
        let mut collector = Collector::default();
        let t0 = Instant::now();
        let mut first = part("alice", "chat", "one");
        first.update_date = Some(1_700_000_000);
        let mut second = part("alice", "chat", "two");
        second.update_date = None;
        let pushed_at = SystemTime::now();
        collector.push(first, t0);
        collector.push(second, t0 + Duration::from_secs(1));

        // Two parts: the extended window from the last part, not the quiet one.
        let at = t0 + Duration::from_secs(1) + EXTENDED_WINDOW;
        let released = collector.release_due(at);
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].release, Release::Extended);

        let released_at = SystemTime::now();
        let record = released[0].record(released_at);
        assert_eq!(record.parts, 2);
        assert_eq!(record.released_by, "extended_window");
        assert_eq!(record.user_name, "alice");
        assert_eq!(record.reply_target, "chat");

        let part_times: Vec<serde_json::Value> =
            serde_json::from_str(&record.part_times).expect("part_times is JSON");
        assert_eq!(part_times.len(), 2, "one entry per part");
        assert_eq!(
            part_times[0]["sent_at"].as_str(),
            Some(
                chrono::DateTime::from_timestamp(1_700_000_000, 0)
                    .expect("a valid timestamp")
                    .to_rfc3339()
                    .as_str()
            )
        );
        assert!(part_times[1]["sent_at"].is_null());
        // Each part is recorded at its own moment: between the moment the parts
        // were taken in and the release, and no earlier than its predecessor's.
        let taken_at = |at: usize| {
            chrono::DateTime::parse_from_rfc3339(
                part_times[at]["taken_at"]
                    .as_str()
                    .expect("taken_at is set"),
            )
            .expect("taken_at is RFC 3339")
        };
        let (pushed_at, released_at) = (
            chrono::DateTime::<chrono::Utc>::from(pushed_at),
            chrono::DateTime::<chrono::Utc>::from(released_at),
        );
        for at in 0..part_times.len() {
            assert!(
                taken_at(at) >= pushed_at && taken_at(at) <= released_at,
                "part {at} was not recorded at its own moment"
            );
        }
        assert!(
            taken_at(0) <= taken_at(1),
            "the parts must be recorded in send order"
        );
    }

    /// Every rule a group can be released by names itself in the record: the column
    /// is what a person asking why a burst split reads, and a release the record can
    /// only call by the wrong name is worse than no record.
    #[test]
    fn every_release_rule_is_named_in_the_record() {
        let rules = [
            (Release::Quiet, "quiet_window"),
            (Release::Extended, "extended_window"),
            (Release::Ceiling, "ceiling"),
            (Release::PartCap, "part_cap"),
            (Release::CharCap, "char_cap"),
            (Release::Control, "control"),
            (Release::Stop, "stop"),
        ];
        for (rule, named) in rules {
            let released = Released {
                parts: vec![Part::new(part("alice", "chat", "one"))],
                release: rule,
            };
            assert_eq!(
                released.record(SystemTime::now()).released_by,
                named,
                "{rule:?} must be recorded as {named}"
            );
        }
    }
}
