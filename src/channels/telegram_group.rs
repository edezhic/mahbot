//! Burst collector for the Telegram inbound path.
//!
//! Telegram gives bots no "still typing" signal, so a short quiet window per
//! account and chat is the only available mechanism to fuse one burst into one
//! request; the collector hands the parts to the pipeline in send order.

use crate::ChannelMessage;
use crate::channels::telegram::{FORWARD_ATTRIBUTION_PREFIX, is_control_message};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;

/// Quiet window after a lone plain part — imperceptible against an assistant
/// turn, so a single message still feels immediate.
const QUIET_WINDOW: Duration = Duration::from_secs(1);
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

/// One burst: the parts of one account in one chat/thread, in send order.
/// Messages the same account sends in two chats (or two forum topics) never land
/// in one group.
struct OpenGroup {
    parts: Vec<ChannelMessage>,
    chars: usize,
    first_at: Instant,
    last_at: Instant,
}

impl OpenGroup {
    /// Start a group from its first part.
    fn new(part: ChannelMessage, now: Instant) -> Self {
        let chars = part.content.chars().count();
        Self {
            parts: vec![part],
            chars,
            first_at: now,
            last_at: now,
        }
    }

    /// `(user_name, reply_target)`. Every part of a group carries the same
    /// addressing — it is the key — so the group stores no copy of it.
    fn key(&self) -> (&str, &str) {
        let first = &self.parts[0];
        (first.user_name.as_str(), first.reply_target.as_str())
    }

    fn matches(&self, part: &ChannelMessage) -> bool {
        self.key() == (part.user_name.as_str(), part.reply_target.as_str())
    }

    /// Fold one more part in and report whether the group must be released now:
    /// it reached the part or character cap.
    fn add(&mut self, part: ChannelMessage, now: Instant) -> bool {
        self.chars += part.content.chars().count();
        self.last_at = now;
        self.parts.push(part);
        self.parts.len() >= MAX_PARTS || self.chars > MAX_CHARS
    }

    /// The user is evidently still sending — the group already holds a second
    /// part or forwarded content — so the extended window applies.
    fn extended(&self) -> bool {
        self.parts.len() > 1
            || self.parts[0]
                .content
                .starts_with(FORWARD_ATTRIBUTION_PREFIX)
    }

    /// When this group is released: its quiet window from the last part, capped
    /// by the ceiling from the first part.
    fn deadline(&self) -> Instant {
        let window = if self.extended() {
            EXTENDED_WINDOW
        } else {
            QUIET_WINDOW
        };
        (self.last_at + window).min(self.first_at + CEILING)
    }
}

/// Collects one burst per account+chat. Every method takes the current instant
/// so the timing policy is testable without sleeping.
#[derive(Default)]
struct Collector {
    open: Vec<OpenGroup>,
}

impl Collector {
    /// Add one part of an account's burst. Returns the parts of the group when
    /// it must be released now (it reached the part or character cap).
    fn push(&mut self, part: ChannelMessage, now: Instant) -> Option<Vec<ChannelMessage>> {
        let Some(at) = self.open.iter().position(|group| group.matches(&part)) else {
            // A fresh group holds this one part, so neither cap is reached yet.
            self.open.push(OpenGroup::new(part, now));
            return None;
        };
        if !self.open[at].add(part, now) {
            return None;
        }
        Some(self.open.remove(at).parts)
    }

    /// The earliest group deadline, `None` when nothing is held.
    fn deadline(&self) -> Option<Instant> {
        self.open.iter().map(OpenGroup::deadline).min()
    }

    /// Every group whose window has closed.
    fn release_due(&mut self, now: Instant) -> Vec<Vec<ChannelMessage>> {
        self.take_where(|group| group.deadline() <= now)
    }

    /// Every open group of one account in one chat/thread — the group a control
    /// message arriving in that chat is handed over ahead of, so a state-clearing
    /// command does not land in front of it. Order is preserved only as enqueue
    /// order: the group is put on the pipeline before the control message is.
    fn release_chat(&mut self, user_name: &str, reply_target: &str) -> Vec<Vec<ChannelMessage>> {
        self.take_where(|group| group.key() == (user_name, reply_target))
    }

    /// Every open group — the listener is going away.
    fn release_all(&mut self) -> Vec<Vec<ChannelMessage>> {
        self.take_where(|_| true)
    }

    /// Take out every group matching `pred`, in the order the groups were opened.
    fn take_where(&mut self, pred: impl Fn(&OpenGroup) -> bool) -> Vec<Vec<ChannelMessage>> {
        let mut ready = Vec::new();
        let mut at = 0;
        while at < self.open.len() {
            if pred(&self.open[at]) {
                ready.push(self.open.remove(at).parts);
            } else {
                at += 1;
            }
        }
        ready
    }
}

/// The single message a burst becomes: the parts in send order (each keeping its
/// own text, markers, attachments and reply reference until the pipeline expands
/// them) plus the addressing the pipeline routes with, taken from the newest
/// part — every part carries the same one, so any of them speaks for the group.
/// `content` belongs to the parts.
fn group_message(parts: Vec<ChannelMessage>) -> ChannelMessage {
    // A released group always holds at least one part (see `OpenGroup`).
    let newest = parts
        .last()
        .expect("a released group holds at least one part");
    let user_name = newest.user_name.clone();
    let reply_target = newest.reply_target.clone();
    let channel = newest.channel.clone();
    ChannelMessage {
        user_name,
        reply_target,
        channel,
        parts,
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
pub(crate) fn spawn_collector(
    input: mpsc::Receiver<ChannelMessage>,
    output: mpsc::Sender<ChannelMessage>,
) {
    tokio::spawn(run_collector(input, output));
}

async fn run_collector(
    mut input: mpsc::Receiver<ChannelMessage>,
    output: mpsc::Sender<ChannelMessage>,
) {
    let mut collector = Collector::default();
    loop {
        tokio::select! {
            () = wait_until(collector.deadline()) => {
                if !hand_over_all(&output, collector.release_due(Instant::now())).await { return; }
            }
            msg = input.recv() => {
                let Some(msg) = msg else {
                    hand_over_all(&output, collector.release_all()).await;
                    return;
                };
                if is_control_message(&msg) {
                    // Control input is never collected; the group open in that
                    // chat goes on the pipeline ahead of it (`release_chat`).
                    if !hand_over_all(&output, collector.release_chat(&msg.user_name, &msg.reply_target)).await { return; }
                    if output.send(msg).await.is_err() { return; }
                    continue;
                }
                if let Some(group) = collector.push(msg, Instant::now())
                    && !hand_over(&output, group).await { return; }
            }
        }
    }
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
    groups: Vec<Vec<ChannelMessage>>,
) -> bool {
    for group in groups {
        if !hand_over(output, group).await {
            return false;
        }
    }
    true
}

/// Hand one released group to the pipeline: a group of one is that message
/// unchanged, a longer burst becomes one message carrying its parts. Returns
/// `false` when the pipeline is gone — logged, because a released group that
/// cannot be handed over is the one message the collector itself gives up.
async fn hand_over(output: &mpsc::Sender<ChannelMessage>, group: Vec<ChannelMessage>) -> bool {
    let msg = match <[ChannelMessage; 1]>::try_from(group) {
        Ok([only]) => only,
        Err(parts) => {
            let merged = group_message(parts);
            tracing::info!(
                user_name = %merged.user_name,
                parts = merged.parts.len(),
                "Telegram burst: releasing collected messages as one request"
            );
            merged
        }
    };
    if output.send(msg).await.is_ok() {
        return true;
    }
    tracing::warn!("Telegram burst: the message pipeline is gone; dropped a released group");
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
        assert_eq!(released[0].len(), 1);
        assert_eq!(released[0][0].content, "hello");
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
        forwarded.push(part("alice", "chat", "[Forwarded from @bob] hi"), t0);
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
        assert_eq!(released.len(), MAX_PARTS);
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
        assert_eq!(released.len(), 2);

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
        assert_eq!(released[0].len(), 2);
        assert_eq!(released[0][0].content, "a1");
        assert_eq!(released[0][1].content, "a3");

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

    #[tokio::test]
    async fn collector_hands_a_command_over_after_the_open_group() {
        let (input, input_rx) = mpsc::channel(QUEUE_CAPACITY);
        let (output, mut output_rx) = mpsc::channel(8);
        spawn_collector(input_rx, output);

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
        let (input, input_rx) = mpsc::channel(QUEUE_CAPACITY);
        let (output, mut output_rx) = mpsc::channel(8);
        spawn_collector(input_rx, output);

        let mut first = part("alice", "chat", "[Forwarded from @bob] one");
        first.attachment_dirs = vec!["dir-one".to_string()];
        let mut second = part("alice", "chat", "two");
        second.attachment_dirs = vec!["dir-two".to_string()];
        input.send(first).await.unwrap();
        input.send(second).await.unwrap();
        drop(input);

        // The burst is released whole: the parts in send order, each with its own
        // content, markers and attachments, under the addressing the pipeline
        // routes with.
        let group = released(&mut output_rx).await;
        assert_eq!(group.parts.len(), 2);
        assert_eq!(group.parts[0].content, "[Forwarded from @bob] one");
        assert_eq!(group.parts[0].attachment_dirs, vec!["dir-one".to_string()]);
        assert_eq!(group.parts[1].content, "two");
        assert_eq!(group.parts[1].attachment_dirs, vec!["dir-two".to_string()]);
        assert!(group.content.is_empty());
        assert_eq!(group.user_name, "alice");
        assert_eq!(group.reply_target, "chat");
        assert_eq!(group.channel, "telegram");
        assert_eq!(
            compose_group_content(group.parts.clone()),
            format!("[Forwarded from @bob] one{PART_SEPARATOR}two")
        );
        // The collector leaves with the listener.
        assert!(output_rx.recv().await.is_none());
    }
}
