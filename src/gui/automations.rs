//! Automations dashboard page — every active automation of every user.
//!
//! The page reads the active automations and renders them as cards, grouped by
//! owner, with a live countdown. Its only action is deleting one automation from its
//! card's context menu: the card leaves the page through the next refresh.
//! Nothing here creates, edits or fires an automation. The page re-reads the list
//! once a second while it is open; the countdown is rendered from the wall
//! clock on every re-render, so a tick that starts no read still moves the
//! digits.

use std::collections::BTreeMap;

use chrono::{DateTime, Local, Utc};
use iced::widget::{Column, column, container, row, text};
use iced::{Alignment, Element, Length, Task};
use iced_fonts::lucide;

use crate::automations::{Automation, StoredTrigger};

use super::ToastMessage;
use super::common::PolledList;
use super::menus::{ContextMenu, MenuItem};
use super::{theme, widgets};

/// The notice for an automation that was already gone when the delete ran — a
/// one-shot that fired, or one removed elsewhere: no failure, but no success.
const ALREADY_GONE: &str = "Automation was already gone";

#[derive(Debug, Clone)]
pub(crate) enum AutomationsMessage {
    /// One refresh interval elapsed.
    Tick,
    /// A read finished. `Err` keeps the last loaded list on screen.
    Refreshed(Result<Vec<Automation>, String>),
    /// Delete the automation the context menu was opened on, identified by its own
    /// row: another account's automation — or a gone account's — is deleted the same
    /// way.
    Delete { id: String, session_id: String },
    /// Outcome of an action, shown by the dashboard as a brief notice.
    Toast(ToastMessage),
}

pub(crate) struct AutomationsState {
    list: PolledList<Automation>,
}

impl AutomationsState {
    pub(crate) const fn new() -> Self {
        Self {
            list: PolledList::new(),
        }
    }

    /// Read the active automations now, unless a read is in flight. Called on
    /// navigation, so opening the page always starts a fresh read.
    pub(crate) fn refresh(&mut self) -> Task<AutomationsMessage> {
        if !self.list.begin() {
            return Task::none();
        }
        Task::perform(load(), AutomationsMessage::Refreshed)
    }

    pub(crate) fn update(&mut self, message: AutomationsMessage) -> Task<AutomationsMessage> {
        match message {
            AutomationsMessage::Tick => self.refresh(),
            AutomationsMessage::Refreshed(result) => {
                self.list.settle(result);
                Task::none()
            }
            // The card is left to the page's own refresh: it is gone from the
            // store, so the next read drops it, and a failed delete leaves it
            // in place.
            AutomationsMessage::Delete { id, session_id } => Task::perform(
                async move {
                    crate::automations::remove_automation(&session_id, &id)
                        .await
                        .map_err(|e| e.to_string())
                },
                delete_outcome,
            ),
            // The dashboard shows the notice and never routes it here.
            AutomationsMessage::Toast(_) => Task::none(),
        }
    }

    pub(crate) fn view(&self) -> Element<'_, AutomationsMessage> {
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
                    lucide::workflow::<iced::Theme, iced::Renderer>(),
                    "No active automations",
                    theme::TEXT_MUTED,
                ));
            }
        } else {
            let now = Local::now();
            let mut sections = Column::new().spacing(theme::SPACE_16);
            for (owner, automations) in group_by_owner(self.list.entries()) {
                let mut cards = Column::new().spacing(theme::SPACE_8);
                for automation in automations {
                    cards = cards.push(render_automation(automation, now));
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

/// Read every active automation of every owner.
async fn load() -> Result<Vec<Automation>, String> {
    crate::automations::list_all_active_automations()
        .await
        .map_err(|e| e.to_string())
}

/// The automations grouped by owner, name-ordered, the members in the order the
/// store returned them (next fire time within the owner). Grouping is by the
/// name recorded on the automation itself, so an automation whose account is gone is
/// still listed under it.
fn group_by_owner(automations: &[Automation]) -> Vec<(&str, Vec<&Automation>)> {
    let mut groups: BTreeMap<&str, Vec<&Automation>> = BTreeMap::new();
    for automation in automations {
        groups
            .entry(automation.user_name.as_str())
            .or_default()
            .push(automation);
    }
    groups.into_iter().collect()
}

/// The check's arguments as `key: value` pairs in the order they are stored,
/// `None` when it takes none. A string value is shown bare — the line is a label
/// for a human, not machine output — and any other value keeps its compact JSON.
fn render_args(trigger: &StoredTrigger) -> Option<String> {
    let StoredTrigger::Tool(trigger) = trigger else {
        return None;
    };
    (!trigger.args.is_empty()).then(|| {
        trigger
            .args
            .iter()
            .map(|(key, value)| match value {
                // A string value is its own text; the JSON form would quote it.
                serde_json::Value::String(raw) => format!("{key}: {raw}"),
                value => format!("{key}: {value}"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    })
}

/// One automation as its own card — the log row's surface, padding and width — with
/// a right-click menu whose single item deletes it.
///
/// An automation that runs a check headlines the check: the tool's name in the type
/// the automation's own text uses, its arguments kept beside it small and secondary,
/// and the text demoted under that line. A plain automation keeps its text as the
/// headline. The countdown sits beside whichever line leads.
fn render_automation(
    automation: &Automation,
    now: DateTime<Local>,
) -> Element<'_, AutomationsMessage> {
    let mut headline = row![];
    let mut demoted_text = None;
    match &automation.trigger {
        None => {
            headline = headline.push(
                text(&automation.text)
                    .size(theme::TEXT_14)
                    .color(theme::TEXT_PRIMARY)
                    .width(Length::Fill),
            );
        }
        Some(trigger) => {
            let mut check = row![
                text(trigger.name())
                    .size(theme::TEXT_14)
                    .color(theme::TEXT_PRIMARY)
            ]
            .spacing(theme::SPACE_6)
            .align_y(Alignment::Center);
            // Folded at any glyph so one long token cannot run under the countdown.
            if let Some(args) = render_args(trigger) {
                check = check.push(
                    text(args)
                        .size(theme::TEXT_11)
                        .font(super::JETBRAINS_MONO)
                        .color(theme::TEXT_SECONDARY)
                        .wrapping(iced::widget::text::Wrapping::WordOrGlyph),
                );
            }
            headline = headline.push(check.width(Length::Fill));
            // An empty automation text draws nothing, so the row ends after the
            // check line rather than showing a blank rung.
            if !automation.text.trim().is_empty() {
                demoted_text = Some(
                    text(&automation.text)
                        .size(theme::TEXT_13)
                        .color(theme::TEXT_SECONDARY)
                        .width(Length::Fill),
                );
            }
        }
    }
    let mut entry = Column::new().spacing(theme::SPACE_2).push(
        headline
            .push(
                text(format_next_fire(&automation.next_fire_at, now))
                    .size(theme::TEXT_12)
                    .font(super::JETBRAINS_MONO)
                    .color(theme::TEXT_PRIMARY),
            )
            .spacing(theme::SPACE_12)
            .align_y(Alignment::Start),
    );
    if let Some(demoted_text) = demoted_text {
        entry = entry.push(demoted_text);
    }
    if let Some(target) = automation.role_target() {
        // The role the automation is addressed to, worded exactly as the assistant
        // sees it — an automation that wakes the Assistant has no such line.
        entry = entry.push(
            text(target)
                .size(theme::TEXT_11)
                .font(super::JETBRAINS_MONO)
                .color(theme::TEXT_MUTED),
        );
    }

    // Keyed by the automation's row: a refresh that reshuffles the list closes this
    // menu instead of redirecting its Delete at another automation.
    ContextMenu::new(
        container(entry)
            .padding(theme::PAD_6)
            .width(Length::Fill)
            .style(theme::surface_card_style),
        vec![MenuItem::with_icon(
            lucide::advanced_text::trash,
            "Delete".into(),
            AutomationsMessage::Delete {
                id: automation.id.clone(),
                session_id: automation.session_id.clone(),
            },
        )],
    )
    .key(&automation.id)
    .into()
}

/// The notice a delete attempt is reported with.
fn delete_outcome(result: Result<Option<Automation>, String>) -> AutomationsMessage {
    match result {
        Ok(Some(_)) => AutomationsMessage::Toast(ToastMessage::Deleted),
        Ok(None) => AutomationsMessage::Toast(ToastMessage::Warning(ALREADY_GONE.to_string())),
        Err(e) => AutomationsMessage::Toast(ToastMessage::Error(e)),
    }
}

/// The next-fire label of an automation against the local clock: `now` once it is
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
    use chrono::{Duration, TimeZone as _};

    fn local_now() -> DateTime<Local> {
        Local
            .with_ymd_and_hms(2026, 9, 17, 12, 0, 0)
            .single()
            .expect("a valid local time")
    }

    fn stored(at: DateTime<Local>) -> String {
        at.with_timezone(&Utc).to_rfc3339()
    }

    #[test]
    fn format_next_fire_covers_due_countdown_other_day_and_junk() {
        let now = local_now();
        let at = |offset: Duration| stored(now + offset);
        let hms = Duration::seconds(2 * 3600 + 5 * 60 + 3);
        let cases: &[(&str, Duration, &str)] = &[
            ("overdue", Duration::minutes(-1), "now"),
            ("exactly due", Duration::zero(), "now"),
            ("seconds later", Duration::seconds(30), "0h 0m 30s"),
            ("hours minutes seconds", hms, "2h 5m 3s"),
            ("another day", Duration::days(1), "at 12:00 18.09"),
        ];
        for &(name, offset, expected) in cases {
            assert_eq!(format_next_fire(&at(offset), now), expected, "case {name}");
        }
        assert_eq!(format_next_fire("not a timestamp", now), "not a timestamp");
    }

    fn automation(owner: &str, text: &str) -> Automation {
        Automation {
            id: String::new(),
            session_id: String::new(),
            user_name: owner.to_string(),
            kind: "one-shot".to_string(),
            text: text.to_string(),
            trigger: None,
            role: None,
            identifier: None,
            workspace: None,
            interval_seconds: None,
            next_fire_at: String::new(),
            unfinished_attempts: 0,
        }
    }

    fn trigger(args: serde_json::Value) -> StoredTrigger {
        let serde_json::Value::Object(args) = args else {
            unreachable!("test arguments are an object")
        };
        StoredTrigger::Tool(crate::automations::Trigger {
            tool: "wait".to_string(),
            args,
        })
    }

    #[test]
    fn render_args_cases() {
        let cases: &[(&str, StoredTrigger, Option<&str>)] = &[
            ("unreadable trigger", StoredTrigger::Unreadable, None),
            ("no arguments", trigger(serde_json::json!({})), None),
            (
                "every value kind",
                trigger(serde_json::json!({
                    "action": "night",
                    "dry_run": true,
                    "probe_seconds": 120,
                    "window": {"from": "09:00"},
                })),
                Some(
                    "action: night, dry_run: true, probe_seconds: 120, window: {\"from\":\"09:00\"}",
                ),
            ),
        ];
        for (name, stored, expected) in cases {
            assert_eq!(
                render_args(stored),
                expected.map(str::to_string),
                "case {name}: rendered arguments"
            );
        }
    }

    #[test]
    fn group_by_owner_orders_by_owner_and_keeps_members() {
        let automations = [
            automation("bob", "two"),
            automation("alice", "one"),
            automation("alice", "three"),
        ];
        let groups = group_by_owner(&automations);
        let owners: Vec<&str> = groups.iter().map(|(owner, _)| *owner).collect();
        assert_eq!(owners, ["alice", "bob"]);
        let texts: Vec<Vec<&str>> = groups
            .iter()
            .map(|(_, group)| group.iter().map(|a| a.text.as_str()).collect())
            .collect();
        assert_eq!(texts, [vec!["one", "three"], vec!["two"]]);
    }
}
