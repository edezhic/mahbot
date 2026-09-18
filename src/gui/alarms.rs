//! Alarms dashboard page — every active alarm of every user.
//!
//! The page reads the active alarms and renders them as cards, grouped by
//! owner, with a live countdown. Its only action is deleting one alarm from its
//! card's context menu: the card leaves the page through the next refresh.
//! Nothing here creates, edits or fires an alarm. The page re-reads the list
//! once a second while it is open; the countdown is rendered from the wall
//! clock on every re-render, so a tick that starts no read still moves the
//! digits.

use std::collections::BTreeMap;

use chrono::{DateTime, Local, Utc};
use iced::widget::{Column, column, container, row, text};
use iced::{Alignment, Element, Length, Task};
use iced_fonts::lucide;

use crate::alarms::Alarm;

use super::ToastMessage;
use super::common::PolledList;
use super::menus::{ContextMenu, MenuItem};
use super::{theme, widgets};

/// The notice for an alarm that was already gone when the delete ran — a
/// one-shot that fired, or one removed elsewhere: no failure, but no success.
const ALREADY_GONE: &str = "Alarm was already gone";

#[derive(Debug, Clone)]
pub(crate) enum AlarmsMessage {
    /// One refresh interval elapsed.
    Tick,
    /// A read finished. `Err` keeps the last loaded list on screen.
    Refreshed(Result<Vec<Alarm>, String>),
    /// Delete the alarm the context menu was opened on, identified by its own
    /// row: another account's alarm — or a gone account's — is deleted the same
    /// way.
    Delete { id: String, session_id: String },
    /// Outcome of an action, shown by the dashboard as a brief notice.
    Toast(ToastMessage),
}

pub(crate) struct AlarmsState {
    list: PolledList<Alarm>,
}

impl AlarmsState {
    pub(crate) const fn new() -> Self {
        Self {
            list: PolledList::new(),
        }
    }

    /// Read the active alarms now, unless a read is in flight. Called on
    /// navigation, so opening the page always starts a fresh read.
    pub(crate) fn refresh(&mut self) -> Task<AlarmsMessage> {
        if !self.list.begin() {
            return Task::none();
        }
        Task::perform(load(), AlarmsMessage::Refreshed)
    }

    pub(crate) fn update(&mut self, message: AlarmsMessage) -> Task<AlarmsMessage> {
        match message {
            AlarmsMessage::Tick => self.refresh(),
            AlarmsMessage::Refreshed(result) => {
                self.list.settle(result);
                Task::none()
            }
            // The card is left to the page's own refresh: it is gone from the
            // store, so the next read drops it, and a failed delete leaves it
            // in place.
            AlarmsMessage::Delete { id, session_id } => Task::perform(
                async move {
                    crate::alarms::remove_alarm(&session_id, &id)
                        .await
                        .map_err(|e| e.to_string())
                },
                delete_outcome,
            ),
            // The dashboard shows the notice and never routes it here.
            AlarmsMessage::Toast(_) => Task::none(),
        }
    }

    pub(crate) fn view(&self) -> Element<'_, AlarmsMessage> {
        let mut content = column![];

        // Error display — inset to align with the vscroll-wrapped list below.
        content = widgets::push_error_banner_inset(content, self.list.error());

        if !self.list.loaded() {
            content = content.push(widgets::scroll_h_inset(widgets::loading_text()));
        } else if self.list.entries().is_empty() {
            // A failed read must not masquerade as a legitimately empty list:
            // the error banner above is the only thing rendered then.
            if self.list.error().is_none() {
                content = content.push(widgets::empty_state_placeholder(
                    lucide::bell::<iced::Theme, iced::Renderer>(),
                    "No active alarms",
                    theme::TEXT_MUTED,
                ));
            }
        } else {
            let now = Local::now();
            let mut sections = Column::new().spacing(theme::SPACE_16);
            for (owner, alarms) in group_by_owner(self.list.entries()) {
                // Cards SPACE_4 apart — tighter than the SPACE_6 below a
                // heading and the SPACE_16 between owners, so each owner's
                // cards read as one group.
                let mut cards = Column::new().spacing(theme::SPACE_4);
                for alarm in alarms {
                    cards = cards.push(render_alarm(alarm, now));
                }
                let section = Column::new()
                    .spacing(theme::SPACE_6)
                    .push(widgets::section_heading(owner))
                    .push(cards);
                sections = sections.push(section);
            }
            content = content.push(widgets::vscroll(sections));
        }

        widgets::page(content)
    }
}

/// Read every active alarm of every owner.
async fn load() -> Result<Vec<Alarm>, String> {
    crate::alarms::list_all_active_alarms()
        .await
        .map_err(|e| e.to_string())
}

/// The alarms grouped by owner, name-ordered, the members in the order the
/// store returned them (next fire time within the owner). Grouping is by the
/// name recorded on the alarm itself, so an alarm whose account is gone is
/// still listed under it.
fn group_by_owner(alarms: &[Alarm]) -> Vec<(&str, Vec<&Alarm>)> {
    let mut groups: BTreeMap<&str, Vec<&Alarm>> = BTreeMap::new();
    for alarm in alarms {
        groups
            .entry(alarm.user_name.as_str())
            .or_default()
            .push(alarm);
    }
    groups.into_iter().collect()
}

/// One alarm as its own card — the log row's surface, padding and width — with
/// a right-click menu whose single item deletes it.
fn render_alarm(alarm: &Alarm, now: DateTime<Local>) -> Element<'_, AlarmsMessage> {
    let mut entry = Column::new().spacing(theme::SPACE_2).push(
        row![
            text(&alarm.text)
                .size(theme::TEXT_14)
                .color(theme::TEXT_PRIMARY)
                .width(Length::Fill),
            text(format_next_fire(&alarm.next_fire_at, now))
                .size(theme::TEXT_12)
                .font(super::JETBRAINS_MONO)
                .color(theme::TEXT_SECONDARY),
        ]
        .spacing(theme::SPACE_12)
        .align_y(Alignment::Start),
    );
    if let Some(trigger) = &alarm.trigger {
        // The arguments are shown exactly as stored: they are the same values
        // that reach the assistant's prompt and the firing notice.
        entry = entry.push(
            text(trigger.render())
                .size(theme::TEXT_11)
                .font(super::JETBRAINS_MONO)
                .color(theme::TEXT_MUTED),
        );
    }

    // Keyed by the alarm's row: a refresh that reshuffles the list closes this
    // menu instead of redirecting its Delete at another alarm.
    ContextMenu::new(
        container(entry)
            .padding(theme::PAD_6)
            .width(Length::Fill)
            .style(theme::surface_card_style),
        vec![MenuItem::with_icon(
            lucide::advanced_text::trash,
            "Delete".into(),
            AlarmsMessage::Delete {
                id: alarm.id.clone(),
                session_id: alarm.session_id.clone(),
            },
        )],
    )
    .key(&alarm.id)
    .into()
}

/// The notice a delete attempt is reported with.
fn delete_outcome(result: Result<Option<Alarm>, String>) -> AlarmsMessage {
    match result {
        Ok(Some(_)) => AlarmsMessage::Toast(ToastMessage::Deleted),
        Ok(None) => AlarmsMessage::Toast(ToastMessage::Warning(ALREADY_GONE.to_string())),
        Err(e) => AlarmsMessage::Toast(ToastMessage::Error(e)),
    }
}

/// The next-fire label of an alarm against the local clock: `now` once it is
/// due, a countdown `Xh Ym Zs` while it is later today (always all three units,
/// so the digits shrink in place), and the absolute local time `at HH:MM DD.MM`
/// on any other day. A stored value that is not a timestamp is shown as stored.
fn format_next_fire(next_fire_at: &str, now: DateTime<Local>) -> String {
    let Ok(fire) = crate::db::parse_utc_timestamp(next_fire_at) else {
        return next_fire_at.to_string();
    };
    let now_utc = now.with_timezone(&Utc);
    if fire <= now_utc {
        return "now".to_string();
    }
    let local = fire.with_timezone(&Local);
    if local.date_naive() != now.date_naive() {
        return format!("at {}", local.format("%H:%M %d.%m"));
    }
    let seconds = (fire - now_utc).num_seconds();
    format!(
        "{}h {}m {}s",
        seconds / 3600,
        (seconds % 3600) / 60,
        seconds % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone as _;

    /// The local wall clock the countdown is rendered against.
    fn local_now() -> DateTime<Local> {
        Local
            .with_ymd_and_hms(2026, 9, 17, 12, 0, 0)
            .single()
            .expect("a valid local time")
    }

    /// The stored form of a local instant (the column is RFC3339 UTC).
    fn stored(at: DateTime<Local>) -> String {
        at.with_timezone(&Utc).to_rfc3339()
    }

    #[test]
    fn format_next_fire_covers_due_countdown_other_day_and_junk() {
        let now = local_now();
        // Overdue (and exactly due) read as due.
        assert_eq!(
            format_next_fire(&stored(now - chrono::Duration::minutes(1)), now),
            "now"
        );
        assert_eq!(format_next_fire(&stored(now), now), "now");
        // Later the same day: always all three units.
        assert_eq!(
            format_next_fire(&stored(now + chrono::Duration::seconds(30)), now),
            "0h 0m 30s"
        );
        assert_eq!(
            format_next_fire(
                &stored(now + chrono::Duration::seconds(2 * 3600 + 5 * 60 + 3)),
                now
            ),
            "2h 5m 3s"
        );
        // Another day: the absolute local time.
        assert_eq!(
            format_next_fire(&stored(now + chrono::Duration::days(1)), now),
            "at 12:00 18.09"
        );
        // A value that is not a timestamp is shown as stored.
        assert_eq!(format_next_fire("not a timestamp", now), "not a timestamp");
    }

    fn alarm(owner: &str, text: &str) -> Alarm {
        Alarm {
            id: String::new(),
            session_id: String::new(),
            user_name: owner.to_string(),
            kind: "one-shot".to_string(),
            text: text.to_string(),
            trigger: None,
            interval_seconds: None,
            next_fire_at: String::new(),
        }
    }

    #[test]
    fn group_by_owner_orders_by_owner_and_keeps_members() {
        // Owner order is the grouping's own, not the store's: rows of one owner
        // are merged even when they are not adjacent.
        let alarms = [
            alarm("bob", "two"),
            alarm("alice", "one"),
            alarm("alice", "three"),
        ];
        let groups = group_by_owner(&alarms);
        let owners: Vec<&str> = groups.iter().map(|(owner, _)| *owner).collect();
        assert_eq!(owners, ["alice", "bob"]);
        let texts: Vec<Vec<&str>> = groups
            .iter()
            .map(|(_, group)| group.iter().map(|a| a.text.as_str()).collect())
            .collect();
        assert_eq!(texts, [vec!["one", "three"], vec!["two"]]);
    }
}
