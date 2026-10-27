//! What a browser step leaves behind in Chrome's tab groups: the agent's group is named and
//! collapsed, and the command line's own tabs are kept out of it.
//!
//! chrome-use opens every tab it creates in the browser's tab group for that session's project
//! (`✦ <project>`, one group shared by the sessions of the project), and adding a tab to a group
//! expands it. Each frontend leaves its groups collapsed afterwards:
//!
//! - [`settle`], after an agent's step: the groups the session's own tabs sit in are collapsed, and
//!   the group chrome-use placed them in is named [`AGENT_GROUP_TITLE`] through chrome-use's own
//!   `session name` — the one naming path, because it moves the project record the next agent tab is
//!   placed by along with the live group ([`nameable`] and [`name_group`] own the gates).
//! - [`leave_command_line_tabs`], after a command-line step that created a tab: that tab is taken
//!   out of chrome-use's group before the result is out, so the owner can work in it beside their
//!   own automation, and the groups this session's tabs sit in are collapsed.
//! - [`collapse_own_groups`], after any other command-line step in a named session: the groups its
//!   tabs sit in are collapsed, as this command always has after its envelope.
//!
//! The command line's tails report nothing: whatever a leg did not manage is silent, because the
//! condition has no effect on the next step (a tab left in the group is closed with the group's
//! sessions later). Only [`settle`] answers a [`Note`], and its caller is the agent's own tool.
//!
//! The verbs covered are the ones whose opened tab can be established: `open`, whose reply names no
//! tab, by the caller reading the session's tabs before the step ([`created_tabs`]), and `click`,
//! whose reply names it. A tab another verb opens (`eval`'s `window.open`, a `press` on a targeted
//! link) is not named by its reply, so it is not moved.

use super::chrome_daemon::{NoVerdict, run_cli_json_at};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::{Duration, Instant};

/// The label the agent's tab group is named with, the mark included: the group is named by
/// chrome-use's own `session name`, which may append a suffix, so a title that starts with this
/// string reads the name.
const AGENT_GROUP_TITLE: &str = "✦ agents";

/// The mark chrome-use puts in front of a project group's title: what tells a group it minted from
/// one the owner made.
const PROJECT_MARK: &str = "✦ ";

/// The budget one stage of a tail's chrome-use calls shares: cosmetic work riding a step's return
/// costs a bounded wait, never the step, and no stage spends another stage's budget.
const TAIL_BUDGET: Duration = Duration::from_secs(5);

/// Bound on one call within a stage.
const CALL_TIMEOUT: Duration = Duration::from_secs(2);

/// A tail's trouble as the Issues view keys it: the failing leg's own reason — a distinct cause is a
/// distinct row — and the line the owner reads. Only [`settle`] answers one, and its caller is the
/// agent's tool, which files it in its own process.
pub(crate) struct Note {
    pub(crate) reason: &'static str,
    pub(crate) detail: String,
}

/// The legs a tail's note names, one reason each.
const TABS_UNREAD_REASON: &str = "chrome-tabs:tabs-unread";
const GROUP_NOT_COLLAPSED_REASON: &str = "chrome-tabs:group-not-collapsed";
const GROUP_NOT_NAMED_REASON: &str = "chrome-tabs:agent-group-not-named";

/// What a command-line step hands its tail about the tab it created.
pub(crate) enum Opened<'a> {
    /// `open` names no tab in its reply: the tabs the step created are the session's created tabs
    /// that were not there when the step started, which the caller read first ([`created_tabs`]).
    Baseline(&'a BTreeSet<i64>),
    /// `click` names the tab the page opened, by the session's own tab id.
    Tab(&'a str),
}

/// Leave the agent's tab group named and collapsed after one step. Answers the first leg that did
/// not work; the caller files it as the product's once-only issue, and a step's own result never
/// carries it.
pub(crate) async fn settle(cli: &Path, session: &str) -> Option<Note> {
    let tabs = match own_rows(cli, session).await {
        Ok(tabs) => tabs,
        Err(reason) => return answered(cli, session, Some(unread(&reason))).await,
    };
    // The collapse comes first and on its own budget: it is what the owner sees, so a naming stage
    // that is slow or refused can never cost it. The naming decision reads the placement group's
    // state from this pass instead of asking the browser the same question twice.
    let (mut trouble, states) = collapse_each(cli, session, groups_of(&tabs)).await;
    if let Some(group) = placement_group(&tabs)
        && let Some(state) = states.get(&group)
        && nameable(state)
        && let Some(detail) = name_group(cli, session).await
    {
        trouble.get_or_insert(Note {
            reason: GROUP_NOT_NAMED_REASON,
            detail,
        });
    }
    answered(cli, session, trouble).await
}

/// Leave a command-line step's own tab as an ordinary tab: taken out of the group chrome-use put it
/// in, before the caller sees the result, with the groups this session's tabs sit in collapsed again.
/// Silent by design: a leg that did not manage is not reported anywhere (see the module docs).
pub(crate) async fn leave_command_line_tabs(cli: &Path, session: &str, opened: Opened<'_>) {
    let Ok(tabs) = own_rows(cli, session).await else {
        return;
    };
    // The collapse only needs the listing; the move needs the created-row decision too, so an answer
    // that refuses that decision costs the move and never the collapse.
    let groups = groups_of(&tabs);
    let targets: Vec<OwnTab> = match opened {
        Opened::Baseline(before) => created_of(tabs)
            .unwrap_or_default()
            .into_iter()
            .filter(|tab| tab.group.is_some() && !before.contains(&tab.tab))
            .collect(),
        // The reply names the tab, so this leg needs no ownership answer: a row that does not say
        // how the session came by the tab is still the one the reply just named, and only the
        // relay's own word that it was adopted (the owner's) keeps it where it is.
        Opened::Tab(id) => tabs
            .into_iter()
            .filter(|tab| tab.group.is_some() && tab.session_tab.as_deref() == Some(id))
            .filter(|tab| tab.created != Some(false))
            .collect(),
    };
    if !targets.is_empty() {
        ungroup(cli, session, &targets, Instant::now() + TAIL_BUDGET).await;
    }
    collapse_each(cli, session, groups).await;
}

/// Leave the groups of a named-session command-line step collapsed, as the command always has: every
/// group this session's own tabs sit in. Runs after the envelope, so its reads cost the caller
/// nothing.
pub(crate) async fn collapse_own_groups(cli: &Path, session: &str) {
    let Ok(tabs) = own_rows(cli, session).await else {
        return;
    };
    collapse_each(cli, session, groups_of(&tabs)).await;
}

/// The tabs a session has created itself, for an `open` step's baseline: `None` when the answer
/// could not be read or named no ownership decision, which costs the moving. A tab the session
/// adopted is not its own, so it is no part of the answer.
pub(crate) async fn created_tabs(cli: &Path, session: &str) -> Option<BTreeSet<i64>> {
    let tabs = own_rows(cli, session).await.ok()?;
    Some(
        created_of(tabs)
            .ok()?
            .into_iter()
            .map(|tab| tab.tab)
            .collect(),
    )
}

/// This session's own tabs by `tab list --all`, or the answer the tail could not read — the prologue
/// every tail starts with.
async fn own_rows(cli: &Path, session: &str) -> Result<Vec<OwnTab>, NoVerdict> {
    list(cli, session, Instant::now() + TAIL_BUDGET).await
}

/// The tail's own note for a `tab list --all` answer it could not read.
fn unread(reason: &NoVerdict) -> Note {
    Note {
        reason: TABS_UNREAD_REASON,
        detail: format!("the step's own tabs could not be read: {}", reason.text()),
    }
}

/// [`settle`]'s answer to its caller: the trouble it met, unless the browser side has no
/// `extension call` door at all — an extension absent, disabled or older than that door. That is a
/// fact about the host, and the product's once-only issue for it is the release sweep's
/// ([`crate::tools::chrome_release`]), so no step repeats it. The check is one chrome-use call on
/// [`CALL_TIMEOUT`], not a tail stage's budget: it rides a leg that already failed.
async fn answered(cli: &Path, session: &str, trouble: Option<Note>) -> Option<Note> {
    let trouble = trouble?;
    let doorless = crate::tools::chrome_tabs::ownership_door_missing(
        cli,
        session,
        Instant::now() + CALL_TIMEOUT,
    )
    .await;
    (!doorless).then_some(trouble)
}

/// Collapse every group in `groups` that is not collapsed, reading each one first so a group the
/// browser no longer has is nothing to do. Its own stage budget, so the collapse the owner sees
/// always gets a slot of its own. Answers the first trouble and the live state read for every group
/// still there, so [`settle`] — the one caller that reports — does not ask again; the command line's
/// tails discard both.
async fn collapse_each(
    cli: &Path,
    session: &str,
    groups: Vec<i64>,
) -> (Option<Note>, BTreeMap<i64, GroupState>) {
    let deadline = Instant::now() + TAIL_BUDGET;
    let mut trouble = None;
    let mut states = BTreeMap::new();
    for group in groups {
        let state = match group_state(cli, session, group, deadline).await {
            Ok(Some(state)) => state,
            // The group is gone from the browser — the tabs this step took out were its last:
            // there is nothing left in the tab strip to collapse.
            Ok(None) => continue,
            Err(detail) => {
                trouble.get_or_insert(Note {
                    reason: GROUP_NOT_COLLAPSED_REASON,
                    detail,
                });
                continue;
            }
        };
        if !state.collapsed
            && let Some(detail) = write_collapsed(cli, session, group, deadline).await
        {
            trouble.get_or_insert(Note {
                reason: GROUP_NOT_COLLAPSED_REASON,
                detail,
            });
        }
        states.insert(group, state);
    }
    (trouble, states)
}

/// Whether the group still needs the name. Only a group chrome-use minted is renamed — its project
/// groups carry the [`PROJECT_MARK`], while a group the owner made keeps whatever title they gave
/// it, even when the session's own tabs were dragged into it. chrome-use may append a suffix to a
/// label, so a title that starts with the name counts as named and a later step does not ask again;
/// a group titled after its session is older chrome-use's per-session group, whose title is the
/// identity this product's close and reclaim match by ([`crate::tools::chrome_tabs`]' honest
/// limits), and carries no mark, so it is left titled too.
fn nameable(state: &GroupState) -> bool {
    match state.title.as_deref() {
        Some(title) => title.starts_with(PROJECT_MARK) && !title.starts_with(AGENT_GROUP_TITLE),
        // No title in the answer: nothing to compare, so the name is attempted — and a group that
        // already reads it is chrome-use's own verdict to give, not this gate's to guess.
        None => true,
    }
}

/// Name the agent's group through chrome-use's own `session name` — the only naming path, because
/// it moves the project record the next agent tab is placed by along with the live group. Its own
/// stage budget: the stage makes one call. The call's own verdict is read too: a browser side whose
/// extension cannot retitle an open group accepts the label without renaming anything, and a group
/// still reading its project's name is answered as trouble rather than passed off as named.
async fn name_group(cli: &Path, session: &str) -> Option<String> {
    match call(
        cli,
        session,
        &["session", "name", AGENT_GROUP_TITLE],
        Instant::now() + TAIL_BUDGET,
    )
    .await
    {
        Ok(answer) => match renamed(&answer) {
            Some(true) => None,
            // chrome-use's own verdict: a browser side whose extension cannot retitle an open group
            // answers this, and the group still reads its project's name.
            Some(false) => Some(format!(
                "the browser did not rename the agent group to {AGENT_GROUP_TITLE}"
            )),
            None => Some(
                "the browser's answer to naming the agent group carried no readable verdict"
                    .to_string(),
            ),
        },
        Err(reason) => Some(format!(
            "the agent group could not be named: {}",
            reason.text()
        )),
    }
}

/// chrome-use's own verdict on a `session name` call, as its flat reply carries it
/// (`{"name":…,"ok":true,"renamedExistingGroup":…}`). `None` when the answer does not say.
fn renamed(answer: &Value) -> Option<bool> {
    answer
        .get("renamedExistingGroup")
        .or_else(|| answer.pointer("/data/renamedExistingGroup"))
        .and_then(Value::as_bool)
}

/// Write `collapsed: true` on `group` through the extension's own door and answer the first reason
/// the write did not take — `None` when the group reports itself collapsed. A refusal (a group
/// holding a tab the relay did not create is the expected one) is answered, never raised.
async fn write_collapsed(
    cli: &Path,
    session: &str,
    group: i64,
    deadline: Instant,
) -> Option<String> {
    let args = json!([group, { "collapsed": true }]).to_string();
    let envelope = match call(
        cli,
        session,
        &["extension", "call", "tabGroups.update", &args],
        deadline,
    )
    .await
    {
        Ok(envelope) => envelope,
        Err(reason) => {
            return Some(format!(
                "group {group} could not be collapsed: {}",
                reason.text()
            ));
        }
    };
    match group_state_of(&envelope) {
        Ok(state) if !state.collapsed => Some(format!(
            "group {group} did not report collapsed after the write"
        )),
        Ok(_) => None,
        Err(reason) => Some(format!(
            "group {group}'s collapsed state carried no readable answer: {}",
            reason.text()
        )),
    }
}

/// Take the tabs a step created out of their groups through the extension's own door. Best-effort
/// and silent: a refusal leaves the tab in its group until the session is closed.
async fn ungroup(cli: &Path, session: &str, opened: &[OwnTab], deadline: Instant) {
    let ids: Vec<i64> = opened.iter().map(|tab| tab.tab).collect();
    let args = json!([ids]).to_string();
    let _ = call(
        cli,
        session,
        &["extension", "call", "tabs.ungroup", &args],
        deadline,
    )
    .await;
}

/// One `tab list --all` row this session owns: the Chrome tab id the extension's door takes, the
/// session's own id for it (`t1` — what a `click` reply calls the tab it opened), the group it sits
/// in when it sits in one, and the relay's word for how the session came by it (`None` when the
/// answer does not say — only the tail that needs the distinction refuses).
struct OwnTab {
    tab: i64,
    session_tab: Option<String>,
    group: Option<i64>,
    created: Option<bool>,
}

/// The rows of a `tab list --all` answer that belong to THIS session (`owner.kind == "self"`): the
/// owner's tabs and other sessions' tabs are never part of a tail's decision. An owned row the
/// answer cannot place refuses the whole answer — a dropped row would hide a tab or a group of
/// ours from the tail.
fn own_tabs(envelope: &Value) -> Result<Vec<OwnTab>, NoVerdict> {
    envelope
        .pointer("/data/browserTabs")
        .and_then(Value::as_array)
        .ok_or(NoVerdict::Unreadable)?
        .iter()
        .filter(|row| row.pointer("/owner/kind").and_then(Value::as_str) == Some("self"))
        .map(own_tab)
        .collect()
}

/// One owned row: the id the door takes (`chromeTabId`, or the relay's `handle`), the group it
/// sits in — `groupId` absent, null or `-1` is Chrome's own spelling of an ungrouped tab — and
/// whether the relay says this session created it.
fn own_tab(row: &Value) -> Result<OwnTab, NoVerdict> {
    let tab = tab_id_of(row).ok_or(NoVerdict::Unreadable)?;
    let group = match row.get("groupId") {
        None | Some(Value::Null) => None,
        Some(id) => Some(id.as_i64().ok_or(NoVerdict::Unreadable)?),
    }
    .filter(|id| *id >= 0);
    let created = match row.get("ownership").and_then(Value::as_str) {
        Some("created") => Some(true),
        Some(_) => Some(false),
        None => None,
    };
    Ok(OwnTab {
        tab,
        session_tab: row.get("tabId").and_then(Value::as_str).map(String::from),
        group,
        created,
    })
}

/// The tabs a `tab list --all` row set says this session created. An adopted tab is the session's
/// to drive but not to move, and a row that does not say how the session came by it refuses the
/// whole answer — a dropped row would leave a tab of this step's inside the group the command line
/// exists to keep it out of.
fn created_of(tabs: Vec<OwnTab>) -> Result<Vec<OwnTab>, NoVerdict> {
    let mut created = Vec::new();
    for tab in tabs {
        match tab.created {
            Some(true) => created.push(tab),
            Some(false) => {}
            None => return Err(NoVerdict::Unreadable),
        }
    }
    Ok(created)
}

/// The group chrome-use placed this session in: the one most of the tabs the answer says the
/// session created sit in. A tab the owner later drags into a group of their own does not move that
/// decision, so a group that is not the placement group is never named. An answer that names no
/// created tab (an older relay) is read as every tab of the session's own.
fn placement_group(tabs: &[OwnTab]) -> Option<i64> {
    let says_created = tabs.iter().any(|tab| tab.created == Some(true));
    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    for tab in tabs {
        if let Some(group) = tab.group
            && (!says_created || tab.created == Some(true))
        {
            *counts.entry(group).or_default() += 1;
        }
    }
    // The most common group; a tie goes to the lower id, which is only reachable once the owner
    // has spread the session's tabs over groups.
    let mut best: Option<(usize, i64)> = None;
    for (group, count) in counts {
        if best.is_none_or(|(top, _)| count > top) {
            best = Some((count, group));
        }
    }
    best.map(|(_, group)| group)
}

/// Every group this session's own tabs sit in, in id order.
fn groups_of(tabs: &[OwnTab]) -> Vec<i64> {
    tabs.iter()
        .filter_map(|tab| tab.group)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// The Chrome tab id one row names: the row's own `chromeTabId`, or the id the relay's `handle`
/// carries (`chrome-tab:<id>`). Any other shape is a row this answer cannot act on.
fn tab_id_of(row: &Value) -> Option<i64> {
    row.get("chromeTabId").and_then(Value::as_i64).or_else(|| {
        row.get("handle")?
            .as_str()?
            .strip_prefix("chrome-tab:")?
            .parse()
            .ok()
    })
}

/// The live state a write is decided from: the title the owner reads, when the answer carries one,
/// and whether the group is collapsed.
struct GroupState {
    title: Option<String>,
    collapsed: bool,
}

fn group_state_of(envelope: &Value) -> Result<GroupState, NoVerdict> {
    let result = envelope
        .pointer("/data/result")
        .ok_or(NoVerdict::Unreadable)?;
    Ok(GroupState {
        title: result
            .get("title")
            .and_then(Value::as_str)
            .map(String::from),
        collapsed: result
            .get("collapsed")
            .and_then(Value::as_bool)
            .ok_or(NoVerdict::Unreadable)?,
    })
}

/// Ask the relay what `group` is right now: `None` when the browser does not have that group any
/// more — the one answer a tail reads as nothing to do — and `Err` for every other answer it could
/// not read, a refusal included: only the group's absence means an absent group.
async fn group_state(
    cli: &Path,
    session: &str,
    group: i64,
    deadline: Instant,
) -> Result<Option<GroupState>, String> {
    let id = group.to_string();
    let unreadable = |reason: &NoVerdict| {
        format!(
            "group {group}'s live state could not be read: {}",
            reason.text()
        )
    };
    let envelope = match call(
        cli,
        session,
        &["extension", "call", "tabGroups.get", &id],
        deadline,
    )
    .await
    {
        Ok(envelope) => envelope,
        Err(reason) if group_gone(&reason) => return Ok(None),
        Err(reason) => return Err(unreadable(&reason)),
    };
    group_state_of(&envelope)
        .map(Some)
        .map_err(|reason| unreadable(&reason))
}

/// Whether a refused `tabGroups.get` is the browser saying the group is gone. chrome-use's own
/// wording for it, recorded live: `tabGroups.get: No group with id: <id>`.
fn group_gone(reason: &NoVerdict) -> bool {
    matches!(reason, NoVerdict::Reported(msg) if msg.contains("No group with id"))
}

/// `tab list --all` on the tail's budget: this session's own rows.
async fn list(cli: &Path, session: &str, deadline: Instant) -> Result<Vec<OwnTab>, NoVerdict> {
    own_tabs(&call(cli, session, &["tab", "list", "--all"], deadline).await?)
}

/// One browser call of a tail, on the budget of the stage it belongs to.
async fn call(
    cli: &Path,
    session: &str,
    args: &[&str],
    deadline: Instant,
) -> Result<Value, NoVerdict> {
    let timeout =
        super::chrome_tabs::spawn_budget(deadline, CALL_TIMEOUT).ok_or(NoVerdict::TimedOut)?;
    run_cli_json_at(cli, args, Some(session), timeout).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One owned row of a `tab list --all` answer, as the tests build it.
    fn own(tab: i64, group: Option<i64>, created: Option<bool>) -> OwnTab {
        OwnTab {
            tab,
            session_tab: Some(format!("t{tab}")),
            group,
            created,
        }
    }

    /// Only THIS session's grouped tabs are read: the owner's tabs, another session's tabs and
    /// ungrouped tabs (`groupId` absent, null or `-1`) are never part of a tail's answer, the
    /// session's own tab id comes through with each row, and a row of ours the answer cannot place
    /// refuses the whole answer rather than hiding a group.
    #[test]
    fn only_this_sessions_own_tabs_are_read() {
        let envelope = json!({
            "data": { "browserTabs": [
                {"chromeTabId": 1, "tabId": "t1", "groupId": 7, "owner": {"kind": "self"}},
                {"chromeTabId": 2, "tabId": "t2", "groupId": 7, "owner": {"kind": "self"}},
                {"chromeTabId": 3, "groupId": 9, "owner": {"kind": "user"}},
                {"chromeTabId": 4, "groupId": 11, "owner": {"kind": "session", "session": "other"}},
                {"chromeTabId": 5, "owner": {"kind": "self"}},
                {"handle": "chrome-tab:6", "groupId": 9, "owner": {"kind": "self"}},
                {"chromeTabId": 7, "groupId": -1, "owner": {"kind": "self"}},
                {"chromeTabId": 8, "groupId": null, "owner": {"kind": "self"}}
            ]}
        });
        let tabs = own_tabs(&envelope).expect("readable");
        assert_eq!(groups_of(&tabs), [7, 9]);
        let ids: Vec<(i64, Option<&str>)> = tabs
            .iter()
            .map(|tab| (tab.tab, tab.session_tab.as_deref()))
            .collect();
        assert_eq!(
            ids,
            [
                (1, Some("t1")),
                (2, Some("t2")),
                (5, None),
                (6, None),
                (7, None),
                (8, None)
            ]
        );

        for unreadable in [
            json!({ "data": {} }),
            json!({ "data": { "browserTabs": [
                {"chromeTabId": 1, "groupId": "7", "owner": {"kind": "self"}}
            ]}}),
        ] {
            assert!(matches!(own_tabs(&unreadable), Err(NoVerdict::Unreadable)));
        }
    }

    /// The rows a command-line tail may move are exactly the tabs THIS session created: a tab the
    /// session adopted is the owner's, another session's and the owner's own rows are never looked
    /// at, and a created row with no id the door could take — or a self row that does not say how
    /// it came to be the session's — refuses the answer rather than leave a tab of this step's in
    /// the group.
    #[test]
    fn created_rows_are_this_sessions_own_and_nothing_else() {
        let envelope = json!({
            "data": { "browserTabs": [
                {"chromeTabId": 1, "groupId": 7, "owner": {"kind": "self"}, "ownership": "created"},
                {"chromeTabId": 2, "owner": {"kind": "self"}, "ownership": "created"},
                {"chromeTabId": 3, "groupId": -1, "owner": {"kind": "self"}, "ownership": "created"},
                {"handle": "chrome-tab:4", "groupId": 7, "owner": {"kind": "self"}, "ownership": "created"},
                {"chromeTabId": 5, "groupId": 7, "owner": {"kind": "self"}, "ownership": "adopted"},
                {"chromeTabId": 6, "groupId": 9, "owner": {"kind": "user"}},
                {"chromeTabId": 7, "groupId": 11, "owner": {"kind": "session", "session": "other"}}
            ]}
        });
        let created = created_of(own_tabs(&envelope).expect("readable")).expect("readable");
        let named: Vec<(i64, Option<i64>)> =
            created.iter().map(|tab| (tab.tab, tab.group)).collect();
        assert_eq!(named, [(1, Some(7)), (2, None), (3, None), (4, Some(7))]);

        for unreadable in [
            json!({ "data": {} }),
            json!({ "data": { "browserTabs": [
                {"chromeTabId": 1, "owner": {"kind": "self"}}
            ]}}),
            json!({ "data": { "browserTabs": [
                {"groupId": 7, "owner": {"kind": "self"}, "ownership": "created"}
            ]}}),
        ] {
            assert!(matches!(
                own_tabs(&unreadable).and_then(created_of),
                Err(NoVerdict::Unreadable)
            ));
        }
    }

    /// A placement is where most of the session's created tabs sit: a tab the owner dragged into a
    /// group of their own does not move it, a tie goes to the lower id, an answer naming no created
    /// tab falls back to every tab of the session's own, and a session with no grouped tab has no
    /// placement at all.
    #[test]
    fn a_sessions_placement_is_the_group_most_created_tabs_sit_in() {
        let tabs = [
            own(1, Some(7), Some(true)),
            own(2, Some(7), Some(true)),
            own(3, Some(9), Some(true)),
            own(4, Some(4), Some(false)),
        ];
        assert_eq!(placement_group(&tabs), Some(7));

        let tie = [own(1, Some(7), Some(true)), own(2, Some(3), Some(true))];
        assert_eq!(placement_group(&tie), Some(3));

        let older_relay = [
            own(1, None, None),
            own(2, Some(5), None),
            own(3, Some(5), None),
        ];
        assert_eq!(placement_group(&older_relay), Some(5));

        assert_eq!(placement_group(&[own(1, None, Some(true))]), None);
    }

    /// A group's live state is decided by its `collapsed` flag alone: a title this answer does not
    /// carry is no title to compare (the rename then takes that leg), while a flag that cannot be
    /// read is an answer the tail cannot decide a write from.
    #[test]
    fn group_state_needs_a_collapsed_flag() {
        let state = group_state_of(
            &json!({ "data": { "result": { "title": "✦ agents", "collapsed": false } } }),
        )
        .expect("readable");
        assert_eq!(state.title.as_deref(), Some("✦ agents"));
        assert!(!state.collapsed);

        let untitled = group_state_of(&json!({ "data": { "result": { "collapsed": true } } }))
            .expect("readable");
        assert_eq!(untitled.title, None);
        assert!(untitled.collapsed);

        for unreadable in [
            json!({ "data": { "result": { "title": "✦ agents" } } }),
            json!({ "data": { "result": null } }),
        ] {
            assert!(matches!(
                group_state_of(&unreadable),
                Err(NoVerdict::Unreadable)
            ));
        }
    }
}
