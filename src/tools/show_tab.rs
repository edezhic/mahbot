//! The `show_tab` tool: raise the browser tab the admin's Assistant is working in.
//!
//! The admin's Assistant drives the owner's real browser through
//! `mahbot chrome --session <name>`, and the tab it works in can sit among dozens of tabs across
//! several profiles. When the owner has to look at or act in that page, describing where to find
//! it is not enough — this tool raises the session's own tab in front of the owner's windows and
//! answers the profile, page title and address to pass on.
//!
//! It is admin-Assistant-only by toolset construction, like `computer`: the tool itself carries no
//! role check, and [`crate::Role::tools`] offers it only in the admin's branch. It is deliberately
//! absent from the `chrome` tool (the analysts' tool reaches guest workspaces and has no identity
//! of the person in front of the machine) and from `mahbot chrome` (any command runner could use
//! it, including the agents whose tabs must not be brought forward).
//!
//! chrome-use's `bringToFront` claims success optimistically, so the raise is verified against the
//! browser itself: the tab in front must be the very tab the session points at, matched by its
//! Chrome tab id, and the window it lives in must be the focused one. When the evidence cannot
//! settle whether the window came forward — an unanswered, failed or unreadable probe — the answer
//! is that the browser did not confirm it, never "raised" and never "not raised".
//!
//! `keep` is what leaves the raised tab with the owner — it takes the tab out of the session's
//! group and drops the session's ownership, so the end of the turn, the group collapsing that
//! follows later browser steps and the session's own expiry can no longer close it. It is asked
//! after every raise except the one the browser itself reports as not brought forward, so a tab the
//! person may be looking at is never closed under their eyes. A tab an earlier raise left with the
//! owner is no longer driven by the session, so it is brought forward again by its exact browser
//! handle (`tab adopt <handle> --activate --force`), never by a loose match.
//!
//! Every ordinary outcome is an answer, never an error: a missing session, another agent's
//! session, an absent helper, a stopped session, a blank tab, a gone tab and an unraised window
//! each say what happened and what is known. All mechanism is in pure helpers below;
//! [`ShowTabTool::execute`] only gathers the facts and renders them.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::Workspace;
use crate::chrome::contract::ChromeResponse;
use crate::chrome::profile::{self, Profile};
use crate::chrome::{is_blank_page_url, is_protected_session, physical_session_name};
use crate::tools::Tool;
use crate::tools::chrome_daemon::{NoVerdict, cli_path, run_cli_json_at};

/// One helper probe's own bound.
const CALL_TIMEOUT: Duration = Duration::from_secs(15);

/// One `show_tab` call's overall bound. The probes a raise needs — the session list, the profile
/// read, both tab lists, the raise, the two verification reads (twice over when a "not raised"
/// verdict is re-checked after [`SETTLE`]) and `keep` — are one action for the owner, so they share
/// this deadline instead of each spending a full bound on a wedged helper.
const ACTION_BUDGET: Duration = Duration::from_secs(60);

/// How long the browser is given to finish taking the front before a "not raised" verdict is
/// reported: a window activation can land just after the helper's reply.
const SETTLE: Duration = Duration::from_millis(200);

/// The bound left for one probe: [`CALL_TIMEOUT`], or what remains of [`ACTION_BUDGET`] when that is
/// shorter. `None` when the budget is spent, which every caller reports as "did not answer".
fn probe_bound(deadline: Instant) -> Option<Duration> {
    super::chrome_tabs::spawn_budget(deadline, CALL_TIMEOUT)
}

const ASK_FOR_SESSION: &str = "Nothing was raised: name the session you are working in — the same \
     name you gave `mahbot chrome --session`.";
const NO_HELPER: &str = "Nothing was raised: chrome-use is not available on this machine.";
const UNANSWERED: &str = "Nothing was raised: the browser did not answer, so the session's tab is \
     unknown.";
const NOT_RAISED: &str = "The window was not raised.";
const RAISED: &str = "Raised the tab in front of the owner's other windows.";
const UNCONFIRMED: &str = "The browser did not confirm the window came to the front.";
const KEEP_NOTE: &str = "The tab may still be closed when the session ends: chrome-use did not keep \
     it open.";

/// A refusal for the named session: nothing moved, and the reason completes the shared opening.
fn refused(session: &str, reason: &str) -> String {
    format!("Nothing was raised: session \"{session}\" {reason}.")
}

pub(crate) struct ShowTabTool;

#[async_trait]
impl Tool for ShowTabTool {
    fn name(&self) -> &'static str {
        "show_tab"
    }

    fn is_advertised(&self) -> bool {
        super::chrome_daemon::is_advertised()
    }

    fn parameters_schema(&self) -> Value {
        super::tool_params_schema(
            &json!({
                "session": {
                    "type": "string",
                    "description": "The session name you gave `mahbot chrome --session` — the \
                                    session whose tab should be raised."
                }
            }),
            &["session"],
        )
    }

    async fn execute(&self, _ws: &Workspace, args: Value) -> Result<String> {
        let Some(name) = args
            .get("session")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
        else {
            return Ok(render(ASK_FOR_SESSION, None, None, None));
        };
        if is_protected_session(name) {
            let reason = "belongs to another agent, and only your own can be raised";
            return Ok(render(&refused(name, reason), None, None, None));
        }
        let Some(cli) = cli_path() else {
            return Ok(render(NO_HELPER, None, None, None));
        };
        let deadline = Instant::now() + ACTION_BUDGET;
        let session = physical_session_name(name);
        // The budget has just started, so its first probe gets the whole per-probe bound.
        match run_cli_json_at(&cli, &["session", "list"], None, CALL_TIMEOUT).await {
            Ok(envelope) if ChromeResponse::from_value(&envelope).lists_session(&session) => {}
            Ok(_) => {
                // No daemon to ask, but the recorded pin still says which profile it was bound to.
                let pinned = pinned_profile(&session);
                let reason = "is not running";
                return Ok(render(&refused(name, reason), pinned.as_ref(), None, None));
            }
            Err(_) => {
                let pinned = pinned_profile(&session);
                return Ok(render(UNANSWERED, pinned.as_ref(), None, None));
            }
        }
        let profile = known_profile(&cli, &session, deadline).await;
        let row = match session_tab(&cli, &session, deadline).await {
            SessionTab::Row(row) => row,
            SessionTab::NoTab => {
                let reason = "has no tab to raise";
                return Ok(render(&refused(name, reason), profile.as_ref(), None, None));
            }
            SessionTab::Unread => return Ok(render(UNANSWERED, profile.as_ref(), None, None)),
        };
        if tab_is_gone(&row) {
            let reason = "was working in a tab that is no longer open";
            return Ok(render(
                &refused(name, reason),
                profile.as_ref(),
                Some(&row),
                None,
            ));
        }
        if page_is_blank(&row) {
            let reason = "has no page open to raise";
            return Ok(render(&refused(name, reason), profile.as_ref(), None, None));
        }
        let browser = browser_tab(&cli, &session, &row.target_id, deadline).await;
        let verdict = match raise_tab(&cli, &session, browser.as_ref(), deadline).await {
            Raise::Reported => verification(&cli, &session, browser.as_ref(), deadline).await,
            // The helper's own refusal is the one known failure.
            Raise::Refused => Verification::NotRaised,
            // No verdict either way is never one: the browser's state is not claimed.
            Raise::Unsettled => Verification::Unconfirmed,
        };
        // Only a raise known to have failed is left to the session's cleanup — anything else is
        // handed to the owner, because a tab that closes while the person may be looking at it is
        // the worse of the two errors.
        let note = if verdict == Verification::NotRaised {
            None
        } else {
            keep_for_owner(&cli, &session, deadline).await
        };
        Ok(render(
            verdict.text(),
            profile.as_ref(),
            Some(&row),
            note.as_deref(),
        ))
    }
}

/// The raise's own outcome, from the helper's answer. Its `success` is only a claim to check
/// against the browser; a probe that failed, timed out or came back unreadable is no verdict at all,
/// so only a reported refusal counts as one.
#[derive(Debug, PartialEq, Eq)]
enum Raise {
    /// The helper reported the tab brought forward.
    Reported,
    /// The helper refused, and said why.
    Refused,
    /// The helper gave no verdict.
    Unsettled,
}

/// [`raise_tab`]'s reading of one helper answer.
#[must_use]
fn raise_outcome(result: &Result<Value, NoVerdict>) -> Raise {
    match result {
        Ok(_) => Raise::Reported,
        Err(NoVerdict::Reported(_)) => Raise::Refused,
        Err(_) => Raise::Unsettled,
    }
}

/// Raise the session's tab and say how the helper answered.
///
/// `bringToFront` is the helper's own verb, and it refuses a tab the session no longer drives — a
/// tab an earlier raise left with the owner. That tab is re-adopted by its exact browser handle and
/// activated, `--force` overriding the helper's refusal to take the front from another session's
/// tab: taking the front is what this action is for.
async fn raise_tab(
    cli: &Path,
    session: &str,
    browser: Option<&BrowserTab>,
    deadline: Instant,
) -> Raise {
    let Some(bound) = probe_bound(deadline) else {
        return Raise::Unsettled;
    };
    let brought = run_cli_json_at(cli, &["bringToFront"], Some(session), bound).await;
    let refusal = raise_outcome(&brought);
    if refusal != Raise::Refused {
        return refusal;
    }
    let Some(browser) = browser else {
        return Raise::Refused;
    };
    let Some(bound) = probe_bound(deadline) else {
        return Raise::Unsettled;
    };
    let adopted = run_cli_json_at(
        cli,
        &["tab", "adopt", &browser.handle, "--activate", "--force"],
        Some(session),
        bound,
    )
    .await;
    raise_outcome(&adopted)
}

/// The browser's own identity of the session's tab: the Chrome tab id and the handle it is
/// re-adopted by — both of which must name the exact tab, never a match on title or address.
struct BrowserTab {
    tab_id: i64,
    handle: String,
}

async fn browser_tab(
    cli: &Path,
    session: &str,
    target_id: &str,
    deadline: Instant,
) -> Option<BrowserTab> {
    if target_id.is_empty() {
        return None;
    }
    let bound = probe_bound(deadline)?;
    let envelope = run_cli_json_at(cli, &["tab", "list", "--all"], Some(session), bound)
        .await
        .ok()?;
    browser_tab_of(&envelope, target_id)
}

/// The identity out of a `tab list --all` answer, matched by the exact target id, so neither the
/// adoption nor the confirmation can land on another tab. `None` for an answer that does not carry
/// the row, and for a tab the relay gives no Chrome id and handle: the only handle such a tab has is
/// its target id, which the browser's own tab list cannot be matched against.
#[must_use]
fn browser_tab_of(envelope: &Value, target_id: &str) -> Option<BrowserTab> {
    let row = envelope
        .pointer("/data/browserTabs")?
        .as_array()?
        .iter()
        .find(|row| row.get("targetId").and_then(Value::as_str) == Some(target_id))?;
    Some(BrowserTab {
        tab_id: crate::chrome::contract::tab_id_of(row)?,
        handle: row.get("handle").and_then(Value::as_str)?.to_string(),
    })
}

/// What reading the session's own tab list said. A failed read is not "no tab": the answer must not
/// turn an unanswered browser into a claim about the session.
#[derive(Debug, PartialEq, Eq)]
enum SessionTab {
    Row(TabRow),
    NoTab,
    Unread,
}

async fn session_tab(cli: &Path, session: &str, deadline: Instant) -> SessionTab {
    let Some(bound) = probe_bound(deadline) else {
        return SessionTab::Unread;
    };
    match run_cli_json_at(cli, &["tab", "list"], Some(session), bound).await {
        Ok(envelope) => session_tab_of(&envelope),
        Err(_) => SessionTab::Unread,
    }
}

/// The session's tab out of a `tab list` answer: the row the relay marks `active`. A session driven
/// without the relay does not carry that mark, so its single listed tab is taken as the one it is
/// working in; several unmarked rows are never guessed between. An answer with no tab array is a
/// failed read, never an empty session.
#[must_use]
fn session_tab_of(envelope: &Value) -> SessionTab {
    let Some(rows) = envelope.pointer("/data/tabs").and_then(Value::as_array) else {
        return SessionTab::Unread;
    };
    let active = rows
        .iter()
        .find(|row| row.get("active").and_then(Value::as_bool) == Some(true));
    let single = match rows.as_slice() {
        [only] => Some(only),
        _ => None,
    };
    active
        .or(single)
        .map_or(SessionTab::NoTab, |row| SessionTab::Row(TabRow::of(row)))
}

/// `keep` for the raised tab, so it outlives the session. The helper answers a tab the session did
/// not create with `wasOwned: false` and no effect, so this is asked whatever the row reported —
/// including after a re-adoption, which makes a kept tab the session's own again. A refusal or an
/// unasked `keep` becomes one honest note, never a failed raise.
async fn keep_for_owner(cli: &Path, session: &str, deadline: Instant) -> Option<String> {
    let Some(bound) = probe_bound(deadline) else {
        return Some(KEEP_NOTE.to_string());
    };
    run_cli_json_at(cli, &["keep"], Some(session), bound)
        .await
        .err()
        .map(|_| KEEP_NOTE.to_string())
}

/// The profile a session was pinned to, when a pin was recorded and reads back.
fn pinned_profile(session: &str) -> Option<Profile> {
    profile::pinned(session).profile().cloned()
}

/// The profile the session is working in: the helper's own serving report, else the recorded pin.
/// `None` when neither answers — the caller then says the profile is unknown.
async fn known_profile(cli: &Path, session: &str, deadline: Instant) -> Option<Profile> {
    let Some(bound) = probe_bound(deadline) else {
        return pinned_profile(session);
    };
    profile::of_session(cli, session, bound)
        .await
        .or_else(|| pinned_profile(session))
}

/// One `extension call` on the session, `None` for every leg that answered nothing readable.
async fn extension_call(
    cli: &Path,
    session: &str,
    method: &str,
    args: &str,
    deadline: Instant,
) -> Option<Value> {
    let bound = probe_bound(deadline)?;
    run_cli_json_at(
        cli,
        &["extension", "call", method, args],
        Some(session),
        bound,
    )
    .await
    .ok()
}

/// The raise as the browser itself reports it. No identity to check against means nothing to
/// confirm, so an unidentifiable tab is never passed off as raised.
async fn verification(
    cli: &Path,
    session: &str,
    browser: Option<&BrowserTab>,
    deadline: Instant,
) -> Verification {
    let Some(browser) = browser else {
        return Verification::Unconfirmed;
    };
    let verdict = browser_verdict(cli, session, browser, deadline).await;
    if verdict != Verification::NotRaised {
        return verdict;
    }
    // Activation can land a moment after the helper's reply, so a "not raised" verdict is asked
    // once more before the owner is told the window did not come forward.
    tokio::time::sleep(SETTLE).await;
    browser_verdict(cli, session, browser, deadline).await
}

/// One reading of the browser's front tab and focused window.
async fn browser_verdict(
    cli: &Path,
    session: &str,
    browser: &BrowserTab,
    deadline: Instant,
) -> Verification {
    let fronts = extension_call(
        cli,
        session,
        "tabs.query",
        r#"{"active":true,"lastFocusedWindow":true}"#,
        deadline,
    )
    .await
    .and_then(|envelope| front_tabs(&envelope));
    let focused = extension_call(cli, session, "windows.getLastFocused", "{}", deadline)
        .await
        .and_then(|envelope| focused_window(&envelope));
    verify(browser, fronts, focused)
}

/// One row of a `tab list` answer.
#[derive(Debug, PartialEq, Eq)]
struct TabRow {
    /// `Some(false)` means the tab is gone (a stale record). `None` (key absent) means the relay
    /// could not be asked, which is not evidence of a gone tab.
    relay_attached: Option<bool>,
    /// The browser's own identity of the tab, matched against `tab list --all` to re-adopt a tab an
    /// earlier raise left with the owner.
    target_id: String,
    title: String,
    /// `None` when the row carries no address at all — which is not the same as a blank page.
    url: Option<String>,
}

impl TabRow {
    fn of(row: &Value) -> Self {
        let text = |key: &str| {
            row.get(key)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        Self {
            relay_attached: row.get("relayAttached").and_then(Value::as_bool),
            target_id: text("targetId"),
            title: text("title"),
            url: row.get("url").and_then(Value::as_str).map(str::to_string),
        }
    }
}

/// Whether the tab the session points at is gone: `relayAttached: false`, never an absent key.
#[must_use]
fn tab_is_gone(row: &TabRow) -> bool {
    row.relay_attached == Some(false)
}

/// Whether the tab shows no page at all: chrome-use titles a scratch tab `about:blank` and points it
/// at the same, so the address alone decides. A row carrying no address is a read that cannot say,
/// never a blank page. Raising one would send the owner to nothing.
#[must_use]
fn page_is_blank(row: &TabRow) -> bool {
    row.url.as_deref().is_some_and(is_blank_page_url)
}

/// One tab the browser reports as in front. A split view marks `active` per pane, so an answer can
/// name more than one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrontTab {
    tab_id: i64,
    window_id: Option<i64>,
}

/// The tabs a `tabs.query` answer names. `None` when the answer carries no result array at all.
#[must_use]
fn front_tabs(envelope: &Value) -> Option<Vec<FrontTab>> {
    envelope
        .pointer("/data/result")?
        .as_array()?
        .iter()
        .map(|row| {
            Some(FrontTab {
                tab_id: row.get("id")?.as_i64()?,
                window_id: row.get("windowId").and_then(Value::as_i64),
            })
        })
        .collect()
}

/// The focused window out of a `windows.getLastFocused` answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FocusedWindow {
    id: i64,
    focused: bool,
}

#[must_use]
fn focused_window(envelope: &Value) -> Option<FocusedWindow> {
    let row = envelope.pointer("/data/result")?;
    Some(FocusedWindow {
        id: row.get("id")?.as_i64()?,
        focused: row.get("focused")?.as_bool()?,
    })
}

/// What the verification calls say about the raise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verification {
    /// The raised tab is in front, in the focused window.
    Confirmed,
    /// The extension could not answer, so nothing can be claimed either way.
    Unconfirmed,
    /// The browser answered, and another tab or another/unfocused window is in front.
    NotRaised,
}

impl Verification {
    /// What the answer says about the raise.
    fn text(self) -> &'static str {
        match self {
            Self::Confirmed => RAISED,
            Self::Unconfirmed => UNCONFIRMED,
            Self::NotRaised => NOT_RAISED,
        }
    }
}

/// What the front tabs and the focused window say about the raise. Our tab in front, in the window
/// the browser reports as focused and frontmost, is the only confirmation; a reading that cannot
/// settle it — no answer, or a front row with no window of its own — is [`Verification::Unconfirmed`].
#[must_use]
fn verify(
    browser: &BrowserTab,
    fronts: Option<Vec<FrontTab>>,
    focused: Option<FocusedWindow>,
) -> Verification {
    let (Some(fronts), Some(focused)) = (fronts, focused) else {
        return Verification::Unconfirmed;
    };
    let Some(front) = fronts.iter().find(|front| front.tab_id == browser.tab_id) else {
        // The browser answered, and its front tab is another one.
        return Verification::NotRaised;
    };
    match front.window_id {
        // Our tab is in front of some other window: the owner's window did not come forward.
        Some(window) if window != focused.id => Verification::NotRaised,
        Some(_) if !focused.focused => Verification::NotRaised,
        Some(_) => Verification::Confirmed,
        // Our tab is in front, but the answer gives no window to hold against the focused one.
        None => Verification::Unconfirmed,
    }
}

/// A profile as the answer names it: display name and directory, or `unknown`.
#[must_use]
fn profile_text(profile: Option<&Profile>) -> String {
    profile.map_or_else(
        || "unknown".to_string(),
        |profile| format!("{} ({})", profile.display, profile.dir),
    )
}

/// The answer the model reads: one fact per line. The verdict leads, the profile it knows — or the
/// word `unknown` when it does not — follows, then the page when one is known, then an optional note.
#[must_use]
fn render(
    verdict: &str,
    profile: Option<&Profile>,
    row: Option<&TabRow>,
    note: Option<&str>,
) -> String {
    let mut out = String::from(verdict);
    out.push_str("\nProfile: ");
    out.push_str(&profile_text(profile));
    if let Some(row) = row {
        let title = row.title.trim();
        if !title.is_empty() {
            out.push_str("\nPage: ");
            out.push_str(title);
        }
        if let Some(url) = row
            .url
            .as_deref()
            .map(str::trim)
            .filter(|url| !url.is_empty())
        {
            out.push_str("\nAddress: ");
            out.push_str(url);
        }
    }
    if let Some(note) = note {
        out.push('\n');
        out.push_str(note);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(display: &str, dir: &str) -> Profile {
        Profile {
            display: display.to_string(),
            dir: dir.to_string(),
        }
    }

    fn tab_row(active: bool, title: &str, url: &str) -> Value {
        json!({
            "active": active,
            "relayAttached": true,
            "targetId": "HEX",
            "title": title,
            "url": url,
        })
    }

    fn tab_list(rows: &[Value]) -> Value {
        json!({"success": true, "data": {"tabs": rows}})
    }

    fn front_answer(fronts: &[(i64, i64)]) -> Value {
        json!({"success": true, "data": {"result": fronts.iter().map(|(tab, window)| json!({
            "id": tab,
            "windowId": window,
        })).collect::<Vec<_>>()}})
    }

    fn focused(id: i64, focused: bool) -> Value {
        json!({"success": true, "data": {"result": {
            "id": id,
            "focused": focused,
        }}})
    }

    fn identity() -> BrowserTab {
        BrowserTab {
            tab_id: 7,
            handle: "chrome-tab:7".to_string(),
        }
    }

    #[test]
    fn the_session_tab_is_the_active_row_and_an_unreadable_list_is_not_an_empty_session() {
        let answer = tab_list(&[
            tab_row(false, "Other", "https://other.example"),
            tab_row(true, "Docs", "https://docs.example/page"),
        ]);
        let SessionTab::Row(row) = session_tab_of(&answer) else {
            panic!("the active row is the session's tab");
        };
        assert_eq!(row.title, "Docs");
        assert_eq!(row.url.as_deref(), Some("https://docs.example/page"));
        assert!(!tab_is_gone(&row));

        // No active marker: a single listed tab is the one the session works in, several are not
        // guessed between.
        let unmarked = tab_row(false, "Docs", "https://docs.example");
        assert!(matches!(
            session_tab_of(&tab_list(std::slice::from_ref(&unmarked))),
            SessionTab::Row(_)
        ));
        assert_eq!(
            session_tab_of(&tab_list(&[unmarked.clone(), unmarked])),
            SessionTab::NoTab
        );
        assert_eq!(session_tab_of(&tab_list(&[])), SessionTab::NoTab);

        // A successful answer carrying no tab array is a failed read, never "no tab".
        assert_eq!(
            session_tab_of(&json!({"success": true})),
            SessionTab::Unread
        );
    }

    #[test]
    fn a_tab_is_gone_only_when_the_relay_says_so() {
        assert!(tab_is_gone(&TabRow::of(
            &json!({"targetId": "HEX", "relayAttached": false})
        )));
        // A direct-CDP row has no relay flag at all, which is not evidence of a closed tab.
        assert!(!tab_is_gone(&TabRow::of(&json!({"targetId": "HEX"}))));
    }

    #[test]
    fn a_blank_tab_has_no_page_to_raise() {
        // The shape the helper reports for a scratch tab it creates itself.
        assert!(page_is_blank(&TabRow::of(
            &json!({"title": "about:blank", "url": "about:blank"})
        )));
        assert!(page_is_blank(&TabRow::of(
            &json!({"title": "", "url": "about:blank#blocked"})
        )));
        assert!(!page_is_blank(&TabRow::of(
            &json!({"title": "Example Domain", "url": "https://example.com/"})
        )));
        // A row with no address at all is a read that cannot say, not a blank page.
        assert!(!page_is_blank(&TabRow::of(&json!({"title": "Docs"}))));
    }

    #[test]
    fn the_tab_identity_comes_from_the_exact_row_and_needs_a_relay_handle() {
        let all = json!({"success": true, "data": {"browserTabs": [
            {"handle": "chrome-tab:1", "targetId": "OTHER"},
            {"chromeTabId": 2, "handle": "chrome-tab:2", "targetId": "HEX"},
        ]}});
        let found = browser_tab_of(&all, "HEX").expect("the exact row carries the identity");
        assert_eq!((found.tab_id, found.handle.as_str()), (2, "chrome-tab:2"));
        // No exact target id means no adoption: a URL or title match could land on another tab.
        assert!(browser_tab_of(&all, "docs.example").is_none());
        // A CDP handle is a target id, so the tab has no id to confirm against and no handle to
        // re-adopt by.
        let direct = json!({"success": true, "data": {"browserTabs": [{"handle": "ABC123", "targetId": "HEX"}]}});
        assert!(browser_tab_of(&direct, "HEX").is_none());
        assert!(browser_tab_of(&json!({"success": true}), "HEX").is_none());
    }

    #[test]
    fn the_raise_is_confirmed_only_by_our_own_tab_in_the_focused_window() {
        let browser = identity();
        let window = focused_window(&focused(3, true)).expect("a focused window must parse");
        let fronts = front_tabs(&front_answer(&[(7, 3)])).expect("a front answer must parse");
        assert_eq!(
            verify(&browser, Some(fronts), Some(window)),
            Verification::Confirmed
        );

        // A split view marks more than one tab active: ours may not come first.
        let split =
            front_tabs(&front_answer(&[(8, 3), (7, 3)])).expect("a front answer must parse");
        assert_eq!(
            verify(&browser, Some(split), Some(window)),
            Verification::Confirmed
        );

        // Another tab in front, even in the right window, is not our raise.
        let other = front_tabs(&front_answer(&[(8, 3)])).expect("a front answer must parse");
        assert_eq!(
            verify(&browser, Some(other), Some(window)),
            Verification::NotRaised
        );
        // Our tab in another window is not the owner's window coming forward.
        let elsewhere = front_tabs(&front_answer(&[(7, 4)])).expect("a front answer must parse");
        assert_eq!(
            verify(&browser, Some(elsewhere), Some(window)),
            Verification::NotRaised
        );
        // Our tab in front while the browser itself is behind another application.
        let behind = focused_window(&focused(3, false)).expect("a focused window must parse");
        let fronts = front_tabs(&front_answer(&[(7, 3)])).expect("a front answer must parse");
        assert_eq!(
            verify(&browser, Some(fronts), Some(behind)),
            Verification::NotRaised
        );
        // Our tab in front, but the answer gives no window of its own to hold against the focused
        // one: the reading cannot settle it.
        let windowless = front_tabs(&json!({"success": true, "data": {"result": [{"id": 7}]}}))
            .expect("a front answer must parse");
        assert_eq!(
            verify(&browser, Some(windowless), Some(window)),
            Verification::Unconfirmed
        );
    }

    #[test]
    fn an_unanswered_browser_confirms_nothing() {
        let browser = identity();
        assert_eq!(verify(&browser, None, None), Verification::Unconfirmed);
    }

    #[test]
    fn only_a_reported_refusal_is_a_refusal() {
        assert_eq!(
            raise_outcome(&Ok(json!({"success": true}))),
            Raise::Reported
        );
        assert_eq!(
            raise_outcome(&Err(NoVerdict::Reported("the tab is not ours".into()))),
            Raise::Refused
        );
        // A probe that never produced a verdict settles nothing: the window may have come forward
        // all the same, so it is neither a raise nor a refusal.
        for silent in [
            NoVerdict::TimedOut,
            NoVerdict::SpawnFailure,
            NoVerdict::Unreadable,
        ] {
            assert_eq!(raise_outcome(&Err(silent)), Raise::Unsettled);
        }
    }

    #[test]
    fn the_answer_carries_every_fact_the_owner_needs() {
        let row = TabRow::of(&json!({
            "targetId": "HEX",
            "title": "Docs",
            "url": "https://docs.example/page",
        }));
        let answer = render(
            RAISED,
            Some(&profile("Work", "Profile 8")),
            Some(&row),
            Some(KEEP_NOTE),
        );
        assert!(answer.starts_with(RAISED), "{answer}");
        assert!(answer.contains("Work (Profile 8)"), "{answer}");
        assert!(answer.contains("Docs"), "{answer}");
        assert!(answer.contains("https://docs.example/page"), "{answer}");
        assert!(answer.ends_with(KEEP_NOTE), "{answer}");
    }

    #[test]
    fn an_unknown_profile_is_said_to_be_unknown_and_a_title_less_page_keeps_its_address() {
        let row =
            TabRow::of(&json!({"targetId": "HEX", "title": "  ", "url": "https://docs.example"}));
        let answer = render(NOT_RAISED, None, Some(&row), None);
        assert!(answer.contains(&profile_text(None)), "{answer}");
        assert!(answer.contains("https://docs.example"), "{answer}");
        assert!(!answer.contains("Page:"), "{answer}");
        // A refusal that names no page still names the profile it could not establish.
        let refusal = render(&refused("docs", "has no tab to raise"), None, None, None);
        assert!(refusal.contains(&profile_text(None)), "{refusal}");
        assert!(refusal.contains("docs"), "{refusal}");
    }

    #[tokio::test]
    async fn a_call_without_a_session_asks_for_one() {
        let ws = crate::workspace::test_ws("show-tab-missing");
        for args in [json!({}), json!({"session": "  "})] {
            let answer = ShowTabTool.execute(&ws, args).await.unwrap();
            assert!(answer.starts_with("Nothing was raised"), "{answer}");
            assert!(answer.contains("name the session"), "{answer}");
        }
    }

    #[tokio::test]
    async fn another_agents_session_is_refused_before_the_session_is_probed() {
        let ws = crate::workspace::test_ws("show-tab-protected");
        let answer = ShowTabTool
            .execute(&ws, json!({"session": "agent-tab-1"}))
            .await
            .unwrap();
        assert!(answer.contains("another agent"), "{answer}");
        // The gate runs before the live-session probe: this name is not live either, so a later
        // check would have answered "is not running".
        assert!(!answer.contains("is not running"), "{answer}");
    }
}
