//! Keeping our own Chrome tab groups collapsed after a browser step.
//!
//! chrome-use opens an agent's tabs in the browser's project tab group (`✦ <project>`, shared by
//! every session of that project), and a group created that way arrives EXPANDED. This module is
//! the cosmetic tail of a step: the groups the step's session owns tabs in are left collapsed
//! after that step, whether it succeeded or failed — a step that provably never reached the
//! owner's browser is skipped, because it owns no tab to owe a group.
//!
//! Those groups are read from the tabs themselves — `tab list --all`, this session's own rows —
//! never from a title: collapsing the shared project group hides every agent tab of that project
//! at once, which is the wanted outcome. `tabGroups.get` answers the live state, and
//! `tabGroups.update` writes `{collapsed: true}` only when the group is expanded, so a collapsed
//! group costs no write on any step. A group holding any tab the relay did not create is refused
//! by the extension's own gate, and a group already gone is left at that.
//!
//! Failure is cosmetic: every chrome-use call is bounded by the tail's budget, a leg that did not
//! work is answered to the caller (the tool logs it, `mahbot chrome` prints it), and nothing here
//! can fail a browser step. A browser side with no `extension call` door at all — an extension too
//! old for this — is already the release sweep's own report ([`crate::tools::chrome_release`]).
//! Nothing is watched: a group the owner expands by hand stays expanded until our next step, and
//! only the collapsed flag is ever written.

use super::chrome_daemon::{NoVerdict, run_cli_json_at};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;
use std::time::{Duration, Instant};

/// The budget the tail's chrome-use calls share: cosmetic work riding a step's return costs a
/// bounded wait, never the step.
const COLLAPSE_BUDGET: Duration = Duration::from_secs(5);

/// Bound on one call of the tail.
const CALL_TIMEOUT: Duration = Duration::from_secs(2);

/// Leave collapsed every tab group this step's session owns a tab in. Answers the reason the first
/// leg that did not work gave, for the caller's own surface; a step's result never carries it.
pub(crate) async fn collapse(cli: &Path, session: &str) -> Option<String> {
    let deadline = Instant::now() + COLLAPSE_BUDGET;
    let groups = match call(cli, session, &["tab", "list", "--all"], deadline)
        .await
        .and_then(|envelope| parse_session_groups(&envelope))
    {
        Ok(groups) => groups,
        Err(reason) => {
            return Some(format!(
                "the step's own tabs could not be read: {}",
                reason.text()
            ));
        }
    };
    let mut trouble = None;
    for group in groups {
        if let Some(failure) = collapse_group(cli, session, group, deadline).await {
            trouble.get_or_insert(failure);
        }
    }
    trouble
}

/// One group's part of the tail: write only when the live state says expanded.
async fn collapse_group(
    cli: &Path,
    session: &str,
    group: i64,
    deadline: Instant,
) -> Option<String> {
    match read_collapsed(cli, session, group, deadline).await {
        Ok(true) => None,
        Ok(false) => write_collapsed(cli, session, group, deadline).await,
        Err(reason) => Some(format!(
            "group {group}'s collapsed state could not be read: {}",
            reason.text()
        )),
    }
}

/// The groups this session's own tabs sit in, out of a `tab list --all` envelope. Only rows the
/// relay attributes to THIS session (`owner.kind == "self"`) count: the owner's tabs and other
/// sessions' tabs are never looked at, because the group to collapse must be the one our own tab
/// proves.
fn parse_session_groups(envelope: &Value) -> Result<Vec<i64>, NoVerdict> {
    let rows = envelope
        .pointer("/data/browserTabs")
        .and_then(Value::as_array)
        .ok_or(NoVerdict::Unreadable)?;
    let mut groups = BTreeSet::new();
    for row in rows {
        if row.pointer("/owner/kind").and_then(Value::as_str) != Some("self") {
            continue;
        }
        // An ungrouped tab carries no group: the relay omits `groupId`, and `-1` (Chrome's own
        // spelling) is no group either. Any other shape is a row this answer cannot place —
        // refused, like `chrome_tabs::parse_tabs_by_group`, because a dropped row would hide a
        // group of ours from the collapse.
        match row.get("groupId") {
            None | Some(Value::Null) => {}
            Some(id) => {
                let id = id.as_i64().ok_or(NoVerdict::Unreadable)?;
                if id >= 0 {
                    groups.insert(id);
                }
            }
        }
    }
    Ok(groups.into_iter().collect())
}

/// The `collapsed` flag out of a `tabGroups.get` / `tabGroups.update` answer.
fn collapsed_of(envelope: &Value) -> Result<bool, NoVerdict> {
    envelope
        .pointer("/data/result/collapsed")
        .and_then(Value::as_bool)
        .ok_or(NoVerdict::Unreadable)
}

/// Ask the relay whether `group` is collapsed right now.
async fn read_collapsed(
    cli: &Path,
    session: &str,
    group: i64,
    deadline: Instant,
) -> Result<bool, NoVerdict> {
    let group = group.to_string();
    let envelope = call(
        cli,
        session,
        &["extension", "call", "tabGroups.get", &group],
        deadline,
    )
    .await?;
    collapsed_of(&envelope)
}

/// Write `collapsed: true` on `group` through the extension's own door. A refusal (a group holding
/// a tab the relay did not create is the expected one) and a failed call are answered, never
/// raised.
async fn write_collapsed(
    cli: &Path,
    session: &str,
    group: i64,
    deadline: Instant,
) -> Option<String> {
    let args = format!("[{group},{{\"collapsed\":true}}]");
    match call(
        cli,
        session,
        &["extension", "call", "tabGroups.update", &args],
        deadline,
    )
    .await
    {
        Ok(envelope) => match collapsed_of(&envelope) {
            Ok(true) => None,
            Ok(false) => Some(format!(
                "group {group} did not report collapsed after the write"
            )),
            Err(reason) => Some(format!(
                "group {group}'s write carried no readable answer: {}",
                reason.text()
            )),
        },
        Err(reason) => Some(format!(
            "group {group}'s collapsed state could not be written: {}",
            reason.text()
        )),
    }
}

/// One browser call of the tail, on the budget the whole tail shares.
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

    /// Only THIS session's grouped tabs name a group: the owner's tabs, another session's tabs
    /// and ungrouped tabs (`groupId` absent, null or `-1`) are never part of the answer, two tabs
    /// of one group collapse it once, and a row of ours the answer cannot place refuses the whole
    /// answer rather than hiding a group.
    #[test]
    fn only_this_sessions_grouped_tabs_name_a_group() {
        let envelope = json!({
            "data": { "browserTabs": [
                {"chromeTabId": 1, "groupId": 7, "owner": {"kind": "self"}},
                {"chromeTabId": 2, "groupId": 7, "owner": {"kind": "self"}},
                {"chromeTabId": 3, "groupId": 9, "owner": {"kind": "user"}},
                {"chromeTabId": 4, "groupId": 11, "owner": {"kind": "session", "session": "other"}},
                {"chromeTabId": 5, "owner": {"kind": "self"}},
                {"chromeTabId": 6, "groupId": 9, "owner": {"kind": "self"}},
                {"chromeTabId": 7, "groupId": -1, "owner": {"kind": "self"}},
                {"chromeTabId": 8, "groupId": null, "owner": {"kind": "self"}}
            ]}
        });
        assert_eq!(parse_session_groups(&envelope).expect("readable"), [7, 9]);

        for unreadable in [
            json!({ "data": {} }),
            json!({ "data": { "browserTabs": [
                {"chromeTabId": 1, "groupId": "7", "owner": {"kind": "self"}}
            ]}}),
        ] {
            assert!(matches!(
                parse_session_groups(&unreadable),
                Err(NoVerdict::Unreadable)
            ));
        }
    }
}
