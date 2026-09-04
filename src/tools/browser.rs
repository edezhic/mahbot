//! Browser automation tool.

use super::browser_daemon::{envelope_success, envelope_verdict};
use crate::util::{UnwrapPoison, is_http_url};
use crate::{Tool, Workspace};
use anyhow::Context;
use async_trait::async_trait;
use futures_util::future::join_all;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;
use tracing::debug;

/// Response from chrome-use `--json` commands.
#[derive(Debug, Deserialize)]
struct BrowserResponse {
    /// Classic `success` envelope key — absent on newer `ok`-keyed envelopes.
    #[serde(default)]
    success: Option<bool>,
    /// Latest chrome-use envelopes replace `success` with `ok` on some commands
    /// (e.g. `session list --json`).
    #[serde(default)]
    ok: Option<bool>,
    data: Option<Value>,
    error: Option<String>,
    /// Stable error-envelope code (v1.5.78+) — present on structured errors.
    code: Option<String>,
}

impl BrowserResponse {
    /// Success gate accepting both the classic `success: true` envelope and
    /// the latest `ok: true` one — delegates to the shared
    /// [`super::browser_daemon::envelope_success`] predicate (failure
    /// precedence) so the typed and Value-based paths cannot drift.
    fn is_success(&self) -> bool {
        envelope_success(self.success, self.ok)
    }
}

/// Actions for navigating and extracting content from web pages.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrowserAction {
    /// Navigate to a URL (returns page content automatically).
    Open { url: String },
    /// Get accessibility snapshot with element refs (`@e1`, `@e2`, …).
    /// Always take a fresh snapshot before using refs.
    Snapshot {
        /// Only show interactive elements (buttons, links, inputs).
        #[serde(default)]
        interactive_only: bool,
        /// Remove empty structural elements (default: true).
        #[serde(default = "true_val")]
        compact: bool,
        /// Limit tree depth.
        depth: Option<u32>,
    },
    /// Click an element by ref (`@e1`) or CSS selector.
    Click { selector: String },
    /// Extract text content from an element by CSS selector.
    GetText { selector: String },
    /// Extract visible rendered text from an element by CSS selector
    /// (uses `innerText()` — no `<script>` or `<style>` content).
    #[serde(alias = "get_innertext", alias = "innertext")]
    GetInnerText { selector: String },
    /// Get current URL.
    GetUrl {},
    /// Press a keyboard key at the current focus (e.g. "Enter", "Tab", "Escape").
    /// Useful for submitting forms after filling inputs.
    Press { key: String },
    /// Run JavaScript in the page context. Returns the result as a string.
    /// Useful for inspecting element attributes, checking state, or debugging.
    Eval { js: String },
    /// Find an element by semantic locator and perform an action.
    /// See `name()` doc block or the tool description for usage.
    Find {
        /// Locator type: text (case-sensitive substring, second most reliable),
        /// role (accessibility tree role),
        /// label (matches `<label for='...'>` only),
        /// placeholder (exact HTML placeholder attribute, NOT aria-label),
        /// alt, title (exact HTML title attribute), testid,
        /// first (CSS selector — most reliable), last (CSS selector), nth (CSS selector + index).
        by: String,
        /// Locator value. For 'text': substring to search for; for 'role':
        /// role name ('button', 'link', 'textbox', etc.); for 'first'/'last'/'nth': CSS selector.
        value: String,
        /// Action to perform: click, fill, type, hover, focus, check, uncheck, text.
        /// "fill" clears the field then types; "type" appends without clearing.
        action: String,
        /// Text to fill/type into the element (only for action "fill" or "type").
        text: Option<String>,
        /// Accessible name filter for role-based finding, e.g. "Submit".
        /// Note: this filter can fail even when the snapshot shows a matching element.
        /// When it fails, retry with `by: "text"` or `by: "first"` with CSS.
        name: Option<String>,
        /// Require exact text match.
        exact: Option<bool>,
        /// Zero-based index for `by: "nth"`. Required when `by` is "nth".
        index: Option<u32>,
    },
    /// Capture a screenshot of the current page as a PNG on disk and inject
    /// it into the agent's conversation context as a native image part, so
    /// the model can visually inspect the rendered page. The output path is
    /// chosen by the tool (under the safe temp root), never by the model.
    Screenshot {},
}

/// Helper for `#[serde(default = "true_val")]` on boolean fields.
const fn true_val() -> bool {
    true
}

/// Browser tool for fetching content from web pages.
///
/// Each operation requires a `tab` name — separate browser sessions
/// (isolated via `--session`). Use `"default"` for most browsing.
/// Operations on the same tab are serialized via a per-tab lock.
#[derive(Default)]
pub struct BrowserTool {
    /// Per-tab locks — only serializes operations on the same tab.
    /// Different tabs can run concurrently without blocking each other.
    tab_locks: std::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// Path of the most recent screenshot written by a `screenshot` action.
    /// Read by `image_payload` to attach the PNG as a native image part. The
    /// call is guarded by the action being `Screenshot`, so a stale value from
    /// a prior round can never be re-attached to a non-screenshot call.
    last_screenshot: std::sync::Mutex<Option<String>>,
}

impl BrowserTool {
    /// Acquire a per-tab lock for serializing operations on the same tab.
    /// Different tabs run fully concurrently.
    async fn acquire_tab_lock(&self, tab: &str) -> tokio::sync::OwnedMutexGuard<()> {
        let lock = {
            let mut locks = self.tab_locks.lock().unwrap_poison();
            locks
                .entry(tab.to_string())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        };
        lock.lock_owned().await
    }

    /// Open a URL, wait for network idle, and extract clean visible text via
    /// `document.body.innerText` (JavaScript eval). Unlike the accessibility
    /// tree returned by the `Open` browser action (which contains element refs,
    /// ARIA roles, and indentation), this returns plain rendered text — no
    /// markup, no hidden content, no `<script>`/`<style>` noise.
    ///
    /// Falls back to `textContent` if the JavaScript eval fails.
    ///
    /// The tab is left open — caller should close it with `close_session` when
    /// done.  The per-tab lock is held for the full duration (navigate +
    /// extract) so concurrent callers targeting the same tab are serialized
    /// consistently.
    pub async fn fetch_page_text(&self, url: &str, tab: &str) -> anyhow::Result<String> {
        Self::validate_url(url)?;

        Self::ensure_available().await?;

        // Lock is held for the entire navigate + extract sequence so
        // concurrent same-tab access doesn't race between navigation
        // and text extraction.
        let _guard = self.acquire_tab_lock(tab).await;
        let opened = self.run_command(&["open", url], tab).await?;

        // A real navigation attempt that ends on the scratch `about:blank`
        // never committed — fail loudly instead of returning empty content.
        self.bail_on_blank_navigation(tab, url, &opened).await?;

        // Wait for network idle (best-effort — no hard error on timeout).
        let _ = self
            .run_command(&["wait", "--load", "networkidle"], tab)
            .await;

        // Extract clean visible text via innerText JS eval (not snapshot).
        let text = self.get_inner_text("body", tab).await?;

        Ok(text)
    }

    /// Close a browser session tab by name — verified: the session's tab group
    /// is swept (enumerate → close → re-enumerate convergence) so a leftover
    /// cannot be orphaned silently by a kill-based close. Only the target
    /// session's own tabs are touched. Non-mahbot session names (e.g. the
    /// agent-facing `default` tab) are refused by the sweep's strict-scope
    /// rule — the tab then persists until the daemon's idle timeout.
    pub async fn close_session(&self, tab: &str) {
        super::browser_daemon::sweep_session(tab).await;
    }

    /// If the response shows the tab still on the scratch `about:blank` page,
    /// the navigation never committed — close the session (verified sweep) and
    /// fail with the cause. The close is refused for non-mahbot session names
    /// (strict-scope rule), leaving the tab to the daemon's idle timeout. No-op
    /// when the navigation committed.
    async fn bail_on_blank_navigation(
        &self,
        tab: &str,
        url: &str,
        response: &BrowserResponse,
    ) -> anyhow::Result<()> {
        if response
            .data
            .as_ref()
            .and_then(|d| d.get("url"))
            .and_then(Value::as_str)
            .is_some_and(is_blank_page_url)
        {
            self.close_session(tab).await;
            anyhow::bail!(
                "Navigation failed: the tab is still on a blank page after opening {url} — the \
                 page never loaded. This usually means the site is unreachable or blocks \
                 automated navigation, or the chrome-use extension relay is down."
            );
        }
        Ok(())
    }

    /// Fail with an actionable error when the chrome-use CLI is missing or the
    /// daemon is down, distinguishing the two causes and never reporting a
    /// transient probe failure (spawn EAGAIN/EMFILE, timeout) as "not
    /// installed". One CLI probe; the daemon-health evaluation is cached.
    async fn ensure_available() -> anyhow::Result<()> {
        match super::browser_daemon::cli_probe().await {
            super::browser_daemon::CliStatus::Available => {}
            super::browser_daemon::CliStatus::Missing => {
                anyhow::bail!(
                    "chrome-use CLI is not available. {}",
                    super::browser_daemon::CHROME_USE_INSTALL_HINT
                );
            }
            super::browser_daemon::CliStatus::Transient(failure) => {
                let msg = match failure {
                    super::browser_daemon::CliProbeFailure::Spawn(reason) => format!(
                        "chrome-use CLI check could not spawn the binary ({reason}) — a \
                         temporary failure (e.g. system resource exhaustion), not a missing \
                         install. Retry shortly."
                    ),
                    super::browser_daemon::CliProbeFailure::BadVersion(status) => format!(
                        "chrome-use CLI is installed but its `--version` check failed \
                         ({status}) — the install looks broken. Reinstall it by having the \
                         Support agent re-run the user-consented `install_chrome_use` tool."
                    ),
                    super::browser_daemon::CliProbeFailure::Timeout => {
                        "chrome-use CLI probe timed out — the binary is present but \
                         unresponsive. Retry shortly; if this persists the CLI may be wedged."
                            .to_string()
                    }
                };
                anyhow::bail!(msg);
            }
        }
        if !super::browser_daemon::is_available().await {
            anyhow::bail!("{}", super::browser_daemon::daemon_down_message());
        }
        Ok(())
    }

    /// Validate a URL is structurally safe to navigate to.
    fn validate_url(url: &str) -> anyhow::Result<()> {
        let url = url.trim();

        if url.is_empty() {
            anyhow::bail!("URL cannot be empty");
        }

        // Block file:// — bypasses SSRF controls.
        if url.starts_with("file://") {
            anyhow::bail!("file:// URLs are not allowed in browser automation");
        }

        if !is_http_url(url) {
            anyhow::bail!("Only http:// and https:// URLs are allowed");
        }

        Ok(())
    }

    /// Run an chrome-use command and parse the JSON response.
    async fn run_command(&self, args: &[&str], tab: &str) -> anyhow::Result<BrowserResponse> {
        let mut cmd = Command::new(super::browser_daemon::cli_path().with_context(|| {
            format!(
                "chrome-use CLI is not available. {}",
                super::browser_daemon::CHROME_USE_INSTALL_HINT
            )
        })?);
        super::browser_daemon::ensure_browser_env(&mut cmd);
        cmd.args(args);
        cmd.arg("--json");
        cmd.args(["--session", tab]);

        debug!("chrome-use args: {:?}", cmd.as_std().get_args());

        let output = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .context("Failed to execute chrome-use CLI")?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        // chrome-use returns exit code 1 even when it outputs valid JSON
        // with a structured error message. Try to parse the JSON first to
        // get a meaningful error, fall back to stderr-only bail otherwise.
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let (error_msg, code) = match serde_json::from_str::<BrowserResponse>(&stdout) {
                Ok(resp) => {
                    let BrowserResponse { error, code, .. } = resp;
                    (error.unwrap_or_default(), code)
                }
                Err(_) => (stderr.trim().to_string(), None),
            };
            let error_msg = if error_msg.is_empty() {
                format!("chrome-use exited with code {}", output.status)
            } else {
                enhance_browser_error(error_msg)
            };
            Self::fail_fast_if_daemon_down(&error_msg, code.as_deref())?;
            anyhow::bail!("chrome-use error: {error_msg}");
        }

        let response: BrowserResponse =
            serde_json::from_str(&stdout).context("Failed to parse chrome-use JSON response")?;

        if !response.is_success() {
            let err = response.error.as_deref().unwrap_or("unknown error");
            let enhanced = enhance_browser_error(err.to_string());
            Self::fail_fast_if_daemon_down(&enhanced, response.code.as_deref())?;
            anyhow::bail!("chrome-use error: {enhanced}");
        }

        Ok(response)
    }

    /// If an error carries the daemon-unavailable signature or envelope code,
    /// mark the daemon unhealthy (wakes the auto-recovery watchdog) and return
    /// the actionable guidance immediately — the CLI already retried
    /// internally, so adding more retries would only burn more time.
    /// Unreachable-tab errors are their own state: the daemon and relay are up,
    /// only the session's tab is orphaned — fail fast with hand-close guidance
    /// and leave health untouched (recovery cannot fix a Chrome-side orphan,
    /// and hiding the daemon would block other sessions for UNHEALTHY_TTL).
    fn fail_fast_if_daemon_down(error: &str, code: Option<&str>) -> anyhow::Result<()> {
        if super::browser_daemon::is_unreachable_tab_error(error) {
            anyhow::bail!("{}", super::browser_daemon::unreachable_tab_message(error));
        }
        if super::browser_daemon::is_daemon_unavailable_error(error)
            || super::browser_daemon::is_daemon_unavailable_code(code)
        {
            super::browser_daemon::note_unhealthy(error);
            anyhow::bail!("{}", super::browser_daemon::daemon_down_message());
        }
        Ok(())
    }

    /// Extract visible rendered text via `innerText`, falling back to `get text`
    /// (`textContent`) when eval fails or returns empty.
    async fn get_inner_text(&self, selector: &str, tab: &str) -> anyhow::Result<String> {
        const FALLBACK_NOTE: &str =
            "(used get text fallback — textContent, may include script/style text)";

        let js = inner_text_eval_js(selector);
        if let Ok(resp) = self.run_command(&["eval", &js], tab).await
            && let Some(data) = resp.data.as_ref()
            && let Some(text) = extract_snapshot_text(data)
            && !text.trim().is_empty()
        {
            return Ok(text);
        }

        let resp = self.run_command(&["get", "text", selector], tab).await?;
        let mut text = resp
            .data
            .as_ref()
            .and_then(extract_snapshot_text)
            .unwrap_or_default();
        if !text.is_empty() {
            text.push('\n');
            text.push_str(FALLBACK_NOTE);
        }
        Ok(text)
    }

    /// Agent-browser supports multiple subcommand styles — this builds the correct
    /// argument list for each action.
    fn build_args(action: &BrowserAction) -> anyhow::Result<Vec<String>> {
        match action {
            BrowserAction::Open { url } => {
                Self::validate_url(url)?;
                Ok(vec!["open".into(), url.clone()])
            }
            BrowserAction::Snapshot {
                interactive_only,
                compact,
                depth,
            } => {
                let mut args = vec!["snapshot".into()];
                if *interactive_only {
                    args.push("-i".into());
                }
                if *compact {
                    args.push("-c".into());
                }
                if let Some(d) = depth {
                    args.push("-d".into());
                    args.push(d.to_string());
                }
                Ok(args)
            }
            BrowserAction::Click { selector } => Ok(vec!["click".into(), selector.clone()]),
            BrowserAction::GetText { selector } => {
                Ok(vec!["get".into(), "text".into(), selector.clone()])
            }
            BrowserAction::GetInnerText { .. } => {
                anyhow::bail!("GetInnerText is handled in execute(), not build_args")
            }
            BrowserAction::GetUrl { .. } => Ok(vec!["get".into(), "url".into()]),
            BrowserAction::Press { key } => Ok(vec!["press".into(), key.clone()]),
            BrowserAction::Eval { js } => Ok(vec!["eval".into(), js.clone()]),
            BrowserAction::Find {
                by,
                value,
                action,
                text,
                name,
                exact,
                index,
            } => {
                let mut args = vec!["find".into(), by.clone()];
                if by == "nth" {
                    let idx = index.map_or_else(|| "0".into(), |i| i.to_string());
                    args.push(idx);
                }
                args.push(value.clone());
                args.push(action.clone());
                if let Some(t) = text {
                    args.push(t.clone());
                }
                if let Some(n) = name {
                    args.push("--name".into());
                    args.push(n.clone());
                }
                if *exact == Some(true) {
                    args.push("--exact".into());
                }
                Ok(args)
            }
            BrowserAction::Screenshot { .. } => {
                anyhow::bail!("Screenshot is handled in execute(), not build_args")
            }
        }
    }
}

/// Close all running browser sessions at shutdown. The chrome-use
/// child process does not always get reaped on process exit — its
/// sessions hold open ports and lingering instances that can
/// interfere with the next daemon startup.
pub async fn close_all_browser_sessions() {
    if tokio::time::timeout(SHUTDOWN_CLEANUP_TIMEOUT, close_all_browser_sessions_inner())
        .await
        .is_err()
    {
        tracing::warn!("chrome-use session cleanup timed out — daemon wedged, skipping");
    }
}

/// Bound on the whole session-cleanup sequence at shutdown. A wedged daemon
/// hangs each CLI call in its internal ~152 s retry loop, so without this cap
/// shutdown would stall; cleanup is best-effort anyway (a dead daemon cannot
/// be cleaned up, and the watchdog recovers it after restart).
const SHUTDOWN_CLEANUP_TIMEOUT: Duration = Duration::from_secs(10);

/// Session names from a `session list --json` payload. Tolerant across
/// chrome-use envelopes: latest returns `{"ok":true,"sessions":[{"name":..}]}`;
/// older builds return `{"success":true,"data":{"sessions":["name",..]}}`.
/// Entries may be objects (keyed by `name`) or plain strings. Callers gate on
/// [`envelope_verdict`] first — this only extracts the names and returns an
/// empty list when no session array is present.
fn parse_session_list(v: &Value) -> Vec<String> {
    let array = v
        .get("sessions")
        .or_else(|| v.get("data").and_then(|d| d.get("sessions")))
        .and_then(Value::as_array);
    array
        .map(|arr| {
            arr.iter()
                .filter_map(|entry| match entry {
                    Value::String(s) => Some(s.clone()),
                    Value::Object(_) => {
                        entry.get("name").and_then(|v| v.as_str()).map(String::from)
                    }
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

async fn close_all_browser_sessions_inner() {
    let Some(cmd) = super::browser_daemon::cli_path() else {
        tracing::debug!("chrome-use not available, skipping browser cleanup");
        return;
    };

    // List active sessions
    let mut list_cmd = Command::new(&cmd);
    super::browser_daemon::ensure_browser_env(&mut list_cmd);
    list_cmd.kill_on_drop(true);
    let list_output = match list_cmd.args(["session", "list", "--json"]).output().await {
        Ok(o) => o,
        Err(e) => {
            tracing::debug!("chrome-use not available, skipping browser cleanup: {e}");
            return;
        }
    };

    let sessions: Vec<String> = match serde_json::from_slice::<Value>(&list_output.stdout) {
        Ok(v) => {
            // Gate only on an explicit failure verdict; a payload with neither
            // verdict key proceeds (tolerance-first — unknown future envelopes
            // still get their sessions closed if they carry a sessions array).
            if envelope_verdict(&v) == Some(false) {
                tracing::warn!(
                    "chrome-use session list failed: {}",
                    v.get("error")
                        .and_then(|e| e.as_str())
                        .unwrap_or("unknown error")
                );
                return;
            }
            parse_session_list(&v)
        }
        Err(e) => {
            tracing::warn!("failed to parse chrome-use session list output: {e}");
            return;
        }
    };

    if sessions.is_empty() {
        tracing::debug!("No open chrome-use sessions to close");
        return;
    }

    let close_futures: Vec<_> = sessions
        .iter()
        .map(|session_id| {
            // Borrows of cmd/session_id are valid — the futures are awaited
            // (join_all) inside this function's scope.
            let cmd = &cmd;
            async move {
                let mut close_cmd = Command::new(cmd);
                super::browser_daemon::ensure_browser_env(&mut close_cmd);
                close_cmd.kill_on_drop(true);
                match close_cmd
                    .args(["--session", session_id, "close"])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .await
                {
                    Ok(status) if status.success() => {
                        tracing::debug!("Closed chrome-use session: {session_id}");
                    }
                    Ok(status) => {
                        tracing::warn!(
                            "chrome-use close session '{session_id}' exited with status: {status}"
                        );
                    }
                    Err(e) => {
                        tracing::warn!("failed to close chrome-use session '{session_id}': {e}");
                    }
                }
            }
        })
        .collect();

    join_all(close_futures).await;
}

#[async_trait]
impl Tool for BrowserTool {
    fn name(&self) -> &'static str {
        "browser"
    }

    fn is_advertised(&self) -> bool {
        super::browser_daemon::is_advertised()
    }

    #[expect(clippy::too_many_lines)]
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "oneOf": [
                        super::action_entry_schema("open", "Navigate to a URL (returns page content automatically)", &["url"], &json!({
                            "url": {
                                "type": "string",
                                "description": "URL to navigate to"
                            }
                        })),
                        super::action_entry_schema("snapshot", "Get accessibility snapshot with element refs (@e1, @e2, ...)", &[], &json!({
                            "interactive_only": {
                                "type": "boolean",
                                "description": "Only show interactive elements (buttons, links, inputs)"
                            },
                            "compact": {
                                "type": "boolean",
                                "description": "Remove empty structural elements. Default: true"
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Limit tree depth"
                            }
                        })),
                        super::action_entry_schema("click", "Click an element by ref or CSS selector", &["selector"], &json!({
                            "selector": {
                                "type": "string",
                                "description": "Element ref (@e1) or CSS selector to click. Refs come from the most recent snapshot on this tab — they become stale after any navigation or re-snapshot"
                            }
                        })),
                        super::action_entry_schema("get_text", "Get text content of an element (uses DOM textContent — includes script/style content)", &["selector"], &json!({
                            "selector": {
                                "type": "string",
                                "description": "Element ref (@e1) or CSS selector. Refs come from the most recent snapshot — always snapshot before calling get_text with a ref"
                            }
                        })),
                        super::action_entry_schema("get_innertext", "Get visible rendered text of an element (uses innerText — no script/style content)", &["selector"], &json!({
                            "selector": {
                                "type": "string",
                                "description": "Element ref (@e1) or CSS selector. Uses innerText() — returns only visible rendered text, no script/style content"
                            }
                        })),
                        super::action_entry_schema("get_url", "Get current URL", &[], &json!({})),
                        super::action_entry_schema("press", "Press a keyboard key at the current focus (e.g. Enter to submit forms)", &["key"], &json!({
                            "key": {
                                "type": "string",
                                "description": "Key to press (e.g. Enter, Tab, Escape, Control+a, ArrowDown)"
                            }
                        })),
                        super::action_entry_schema("eval", "Run JavaScript in the page context. Use to inspect element attributes, check state, or debug.", &["js"], &json!({
                            "js": {
                                "type": "string",
                                "description": "JavaScript to run in the page context"
                            }
                        })),
                        super::action_entry_schema("find", "Find an element by semantic locator and perform an action", &["by", "value", "action"], &json!({
                            "by": {
        "type": "string",
                "description": "Locator type: text (case-sensitive visible text match, second most reliable for buttons/links/headings), role (accessibility tree role, use 'name' field to filter — but name filter can fail even when snapshot shows a match; fall back to 'text' or 'first' if it fails), label (matches <label for='...'> only), placeholder (EXACT match of HTML placeholder attribute — not accessible name shown in snapshot), alt, title (exact HTML title attribute), testid, first (CSS selector — MOST reliable for any element type), last (CSS selector), nth (CSS selector + index). For text inputs: prefer `by: \"first\"` with CSS selector (e.g. `\"input\"`, `\"textarea\"`) — role-based textbox locators are unreliable."
                            },
                            "value": {
        "type": "string",
                "description": "Locator match target. For 'text': substring to search for (case-sensitive); for 'placeholder': exact HTML placeholder attribute value (NOT what snapshot shows — check with eval); for 'role': role name ('button', 'link', 'textbox', 'heading'); for 'label': visible <label> text; for 'first'/'last'/'nth': CSS selector (e.g. 'input', 'button', 'form')"
                            },
                            "action": {
        "type": "string",
                "description": "Action to perform: click (click element), fill (clear field then type), type (append text without clearing, uses 'text' parameter), hover (hover over element), focus (focus element), check (check checkbox/radio button), uncheck (uncheck checkbox/radio button), text (get element text content — does NOT use the 'text' param; the 'text' param is only for fill/type). For filling text into inputs, use 'fill' with the 'text' parameter. For typing without clearing first, use 'type'. Press Enter after filling to submit forms."
                            },
                            "text": {
                                "type": "string",
                                                                                                "description": "Text to fill/type into the element (for action 'fill' or 'type')"
                            },
                            "name": {
                                "type": "string",
                                "description": "Accessible name filter (for role-based finding, e.g. 'Submit'). Note: this filter can fail even when the snapshot shows a matching element. When it fails, retry with `by: \"text\"` or `by: \"first\"` with a CSS selector."
                            },
                            "exact": {
                                "type": "boolean",
                                "description": "Require exact text match"
                            },
                            "index": {
                                "type": "integer",
                                "description": "Zero-based index for `by: \"nth\"`. Required when by is 'nth'."
                            }
                        })),
                        super::action_entry_schema(
                            "screenshot",
                            "Capture a screenshot of the current page as a PNG and inject it into the conversation as a native image, so you can visually inspect the rendered page",
                            &[],
                            &json!({}),
                        )
                    ]
                },
                "tab": {
                    "type": "string",
                    "description": "Logical name for this browser session. \
                     Use \"default\" for most browsing. Missing or empty \
                     defaults to \"default\". Only use a different \
                     name (e.g. \"docs\", \"github\") if you need to keep \
                     multiple pages open simultaneously. Same tab = serialized \
                     operations on that page."
                }
            },
            "required": ["action", "tab"]
        })
    }

    #[expect(clippy::too_many_lines)]
    async fn execute(&self, _ws: &Workspace, args: Value) -> anyhow::Result<String> {
        let mut normalized_notes: Vec<String> = Vec::new();
        let (tab, tab_note) = normalize_tab(&args);
        if let Some(note) = tab_note {
            normalized_notes.push(note);
        }

        let action_value = args
            .get("action")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!(corrective_action_error(&args)))?;
        let (action_value, normalized_note) =
            normalize_action(action_value, &args).map_err(|e| anyhow::anyhow!(e))?;
        if let Some(note) = normalized_note {
            normalized_notes.push(format!("action normalized: {action_value} ({note})"));
        }

        let action: BrowserAction = serde_json::from_value(action_value.clone()).map_err(|e| {
            // Give a more helpful message when the LLM uses wrong field names,
            // always including the exact expected shape so the model can
            // self-correct in one round-trip.
            let hint = match &action_value {
                Value::Object(map) if map.contains_key("find") => {
                    " 'find' requires 'by', 'value', and 'action' fields (use 'value' not 'name' for the locator text). Valid 'action' values: click, fill, type, hover, focus, check, uncheck, text (use 'text' param only for fill/type, not for the 'text' action)".to_string()
                }
                _ => String::new(),
            };
            anyhow::anyhow!(
                "Invalid browser action arguments{hint}. Expected action to be {EXPECTED_ACTION_SHAPE}, \
                 plus a \"tab\" string. Serde error: {e}"
            )
        })?;

        debug!(tab, action = ?action, "browser action");

        Self::ensure_available().await?;

        // Validate find locator type early for better diagnostics.
        if let BrowserAction::Find {
            by,
            action: find_action,
            index,
            ..
        } = &action
        {
            let valid = [
                "role",
                "text",
                "label",
                "placeholder",
                "alt",
                "title",
                "testid",
                "first",
                "last",
                "nth",
            ];
            if !valid.contains(&by.as_str()) {
                anyhow::bail!(
                    "Invalid 'find' locator type '{by}'. Must be one of: {}",
                    valid.join(", ")
                );
            }
            let valid_actions = [
                "click", "hover", "focus", "fill", "type", "check", "uncheck", "text",
            ];
            if !valid_actions.contains(&find_action.as_str()) {
                anyhow::bail!(
                    "Invalid 'find' action '{find_action}'. Must be one of: {}",
                    valid_actions.join(", ")
                );
            }
            if by == "nth" && index.is_none() {
                anyhow::bail!(
                    "'index' is required when 'by' is \"nth\". \
                     Provide the zero-based index of the element to select."
                );
            }
        }

        // Get or create a per-tab lock — only serializes operations on the
        // same tab. Different tabs run fully concurrently.
        let _guard = self.acquire_tab_lock(&tab).await;

        if let BrowserAction::GetInnerText { selector } = &action {
            let output = self.get_inner_text(selector, &tab).await?;
            let body = if output.is_empty() {
                format!("[Tab: {tab}] (no output)")
            } else {
                format!("[Tab: {tab}] {output}")
            };
            return Ok(Self::with_normalization_notes(body, &normalized_notes));
        }

        if let BrowserAction::Screenshot { .. } = &action {
            let output = self.capture_screenshot(&tab).await?;
            return Ok(Self::with_normalization_notes(output, &normalized_notes));
        }

        let cli_args = Self::build_args(&action)?;
        let str_args: Vec<&str> = cli_args.iter().map(String::as_str).collect();
        let response = self.run_command(&str_args, &tab).await?;

        // A real navigation attempt that ends on the scratch `about:blank`
        // never committed — fail loudly (and close the tab best-effort)
        // instead of reporting success with no content.
        if let BrowserAction::Open { url } = &action {
            self.bail_on_blank_navigation(&tab, url, &response).await?;
        }

        // After open, wait for network idle, then auto-snapshot
        // so the LLM sees page content immediately.
        let snapshot_output = if matches!(action, BrowserAction::Open { .. }) {
            let wait_args = ["wait", "--load", "networkidle"];
            let _ = self.run_command(&wait_args, &tab).await;

            // Run a compact snapshot to return page content.
            match self.run_command(&["snapshot", "-c"], &tab).await {
                Ok(snap_resp) => snap_resp
                    .data
                    .as_ref()
                    .and_then(extract_snapshot_text)
                    .unwrap_or_default(),
                Err(_) => String::new(),
            }
        } else {
            String::new()
        };

        let output = match response.data {
            Some(data) => match &action {
                BrowserAction::Snapshot { .. } | BrowserAction::GetText { .. } => {
                    extract_snapshot_text(&data)
                        .or_else(|| serde_json::to_string_pretty(&data).ok())
                        .unwrap_or_default()
                }
                BrowserAction::Open { .. } => {
                    let mut s = format!(
                        "Opened {}",
                        data.get("url").and_then(|v| v.as_str()).unwrap_or("?")
                    );
                    if !snapshot_output.is_empty() {
                        use std::fmt::Write;
                        let _ = write!(s, "\n\n--- Page content ---\n{snapshot_output}");
                    }
                    s
                }
                BrowserAction::GetUrl { .. } => data
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                _ => serde_json::to_string_pretty(&data).unwrap_or_else(|_| data.to_string()),
            },
            None => String::new(),
        };

        let output = if output.is_empty() {
            format!("[Tab: {tab}] (no output)")
        } else {
            format!("[Tab: {tab}] {output}")
        };

        Ok(Self::with_normalization_notes(output, &normalized_notes))
    }

    async fn image_payload(
        &self,
        _ws: &Workspace,
        args: &serde_json::Value,
    ) -> Option<crate::tools::ImagePayload> {
        // Only a successful `screenshot` action produces an image payload.
        // Re-parse the action so a stale `last_screenshot` from a prior round
        // is never re-attached to a non-screenshot call.
        let action_value = args.get("action")?.clone();
        let (action_value, _) = normalize_action(action_value, args).ok()?;
        let action: BrowserAction = serde_json::from_value(action_value).ok()?;
        if !matches!(action, BrowserAction::Screenshot { .. }) {
            return None;
        }
        let path = self.last_screenshot.lock().unwrap_poison().clone()?;
        let p = PathBuf::from(&path);
        // Only a real PNG/JPEG/WebP raster opens the decode (fails open on a
        // non-raster/over-cap file, mirroring the read tool's payload path).
        let meta = crate::util::local_image_to_compressed_data_uri_with_meta(&p)
            .await
            .ok()?;
        Some(crate::tools::ImagePayload::from_compressed_meta(
            &p,
            meta,
            None,
            crate::tools::ImagePayloadSource::Browser,
        ))
    }
}

impl BrowserTool {
    /// Prepend a note about argument normalization so the model can see what
    /// was silently corrected (e.g. flattened fields, XML wrapping).
    fn with_normalization_notes(output: String, notes: &[String]) -> String {
        if notes.is_empty() {
            return output;
        }
        format!("[normalized] {}\n{output}", notes.join("; "))
    }

    /// Capture a screenshot of the current tab to a PNG under the safe temp
    /// root, record its path, and return a textual result describing it. The
    /// per-tab lock is already held by the caller.
    async fn capture_screenshot(&self, tab: &str) -> anyhow::Result<String> {
        let path = Self::screenshot_output_path(tab)?;
        let path_str = path.to_string_lossy().into_owned();
        let str_args = ["screenshot", path_str.as_str()];
        let _response = self.run_command(&str_args, tab).await?;
        // The CLI may report success yet write nothing (e.g. a capture that
        // produced no pixels) — fail loudly so the model does not chase a
        // phantom image.
        if !path.is_file() {
            anyhow::bail!(
                "Browser screenshot reported success but no PNG was written at {}",
                path.display()
            );
        }
        *self.last_screenshot.lock().unwrap_poison() = Some(path_str.clone());
        Ok(format!(
            "[Tab: {tab}] Captured a browser screenshot: {path_str}. [IMAGE:{path_str}]"
        ))
    }

    /// Build a screenshot output path under the pinned temp root (or OS temp
    /// in tests) — one of the allowed temp roots. The filename is derived from
    /// a sanitized tab name plus a random nonce so a model-supplied `tab`
    /// (which is user-controlled) can never traverse out of the temp root.
    fn screenshot_output_path(tab: &str) -> anyhow::Result<PathBuf> {
        let dir = std::env::temp_dir().join("browser-screenshots");
        std::fs::create_dir_all(&dir).with_context(|| {
            format!(
                "Failed to create browser screenshot directory {}",
                dir.display()
            )
        })?;
        let slug = sanitize_filename_component(tab);
        let nonce = rand::random::<u64>();
        Ok(dir.join(format!("{slug}_{nonce:016x}.png")))
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// A navigation that ends here never committed — the tab stayed on the scratch
/// `about:blank` page (relay broken, navigation blocked, etc.). Keyed on the
/// final URL only, so legitimately content-free pages (image URLs, PDFs, canvas
/// shells) are not false-flagged by having zero extracted text.
fn is_blank_page_url(url: &str) -> bool {
    let url = url.trim();
    url.is_empty() || url.starts_with("about:blank")
}

/// Reduce a name to a single safe filename component (ASCII alphanumerics,
/// `-`, `_`), so a model-supplied value (e.g. a browser `tab`) can never
/// inject path separators or `..` traversal into a tool-chosen output path.
fn sanitize_filename_component(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() {
        out.push_str("default");
    }
    if out.len() > 40 {
        out.truncate(40);
    }
    out
}

/// Enhance chrome-use error messages with actionable hints for known
/// failure patterns.
fn enhance_browser_error(msg: String) -> String {
    let lower = msg.to_ascii_lowercase();
    if lower.contains("unknown ref")
        || lower.contains("node with given id does not belong to the document")
    {
        format!(
            "{msg}. Hint: refs become stale after any navigation or DOM change. \
             Take a fresh snapshot before using refs again."
        )
    } else {
        msg
    }
}

// ── Tolerant action normalization ─────────────────────────────────────────

/// Known browser action variant names (must match `BrowserAction` serde names).
const KNOWN_ACTIONS: &[&str] = &[
    "open",
    "snapshot",
    "click",
    "get_text",
    "get_inner_text",
    "get_innertext",
    "innertext",
    "get_url",
    "press",
    "eval",
    "find",
    "screenshot",
];

/// Expected action shape, echoed verbatim in corrective errors so the model
/// can self-correct in one round-trip.
const EXPECTED_ACTION_SHAPE: &str = "one of: {\"open\":{\"url\":\"https://...\"}}, \
    {\"snapshot\":{\"interactive_only\":bool,\"compact\":bool,\"depth\":int}}, \
    {\"click\":{\"selector\":\"...\"}}, {\"get_text\":{\"selector\":\"...\"}}, \
    {\"get_innertext\":{\"selector\":\"...\"}}, {\"get_url\":{}}, \
    {\"press\":{\"key\":\"...\"}}, {\"eval\":{\"js\":\"...\"}}, \
    {\"find\":{\"by\":\"text|role|label|placeholder|alt|title|testid|first|last|nth\",\
    \"value\":\"...\",\"action\":\"click|fill|type|hover|focus|check|uncheck|text\"}}, \
    {\"screenshot\":{}}";

/// Corrective error for an unrecoverable action shape, listing the exact
/// expected form instead of raw serde text.
fn corrective_action_error(received: &Value) -> String {
    format!(
        "Invalid browser action arguments. Expected action to be {EXPECTED_ACTION_SHAPE}, \
         plus a \"tab\" string. Received: {received}"
    )
}

/// Missing/empty tab defaults to the documented default session. Tabs are
/// isolated sessions, so defaulting is safe and echoed in tool output.
fn normalize_tab(args: &Value) -> (String, Option<String>) {
    match super::get_opt_str(args, "tab").filter(|s| !s.is_empty()) {
        Some(tab) => (tab.to_string(), None),
        None => (
            "default".to_string(),
            Some("missing/empty tab defaulted to \"default\"".to_string()),
        ),
    }
}

/// Normalize a model-supplied browser `action` value into the canonical
/// `{"variant": {...}}` tagged form that `BrowserAction` deserializes from.
///
/// Recoverable shapes are mapped to their canonical equivalent and a note
/// describing the correction is returned (so silent acceptance stays visible
/// in tool output). Unrecoverable shapes produce a corrective error.
fn normalize_action(action: Value, args: &Value) -> Result<(Value, Option<String>), String> {
    match action {
        // Canonical tagged object: {"open": {...}} or {"open": "https://..."}.
        Value::Object(map) if map.len() == 1 => {
            let (name, inner) = map.into_iter().next().expect("len == 1");
            if !KNOWN_ACTIONS.contains(&name.as_str()) {
                return Err(corrective_action_error(&json!({name: inner})));
            }
            match inner {
                // {"open": "https://..."} — bare-string value for the open action.
                Value::String(s) if name == "open" => Ok((
                    json!({"open": {"url": s}}),
                    Some("bare-string value for open action treated as url".to_string()),
                )),
                Value::Object(o) => {
                    require_find_action(&name, &o)?;
                    Ok((json!({name: o}), None))
                }
                _ => Err(corrective_action_error(&json!({name: inner}))),
            }
        }
        // Plain string: action name, stringified JSON, XML wrapper, or bare URL.
        Value::String(s) => {
            let s = s.trim();
            // Stringified (double-encoded) JSON action.
            if s.starts_with('{') {
                return parse_embedded_json(s, args, "stringified JSON");
            }
            // CDATA-wrapped JSON: <![CDATA[{"open": {...}}]]>
            if let Some(inner) = s
                .strip_prefix("<![CDATA[")
                .and_then(|r| r.strip_suffix("]]>"))
            {
                return parse_embedded_json(inner.trim(), args, "CDATA-wrapped JSON");
            }
            // XML-wrapped action (exact observed patterns only).
            if s.starts_with('<') {
                if let Some((v, note)) = parse_xml_action(s) {
                    return Ok((v, Some(note.to_string())));
                }
                return Err(corrective_action_error(&Value::String(s.to_string())));
            }
            // Bare URL → open action.
            if is_http_url(s) {
                return Ok((
                    json!({"open": {"url": s}}),
                    Some("bare URL treated as open action".to_string()),
                ));
            }
            // Plain action name with flattened sibling fields.
            if KNOWN_ACTIONS.contains(&s) {
                return build_action_from_siblings(s, args);
            }
            Err(corrective_action_error(&Value::String(s.to_string())))
        }
        other => Err(corrective_action_error(&other)),
    }
}

/// Parse a JSON action string embedded in an outer wrapper (stringified JSON,
/// CDATA) and recursively normalize it, prefixing the wrapper name in the note.
fn parse_embedded_json(
    s: &str,
    args: &Value,
    wrapper: &str,
) -> Result<(Value, Option<String>), String> {
    let parsed: Value = serde_json::from_str(s)
        .map_err(|_| corrective_action_error(&Value::String(s.to_string())))?;
    let (normalized, note) = normalize_action(parsed, args)?;
    Ok((
        normalized,
        Some(match note {
            Some(inner) => format!("{wrapper} action; {inner}"),
            None => format!("{wrapper} action parsed"),
        }),
    ))
}

/// The find action carries its own `action` sub-field — reject it without one
/// (click vs fill vs text would be a side-effecting guess).
fn require_find_action(name: &str, o: &serde_json::Map<String, Value>) -> Result<(), String> {
    if name == "find" && !o.contains_key("action") {
        return Err(corrective_action_error(&json!({name: o})));
    }
    Ok(())
}

/// Build a tagged action object from a plain action name plus sibling fields,
/// e.g. `{"action":"open","url":"...","tab":"..."}` → `{"open":{"url":"..."}}`.
fn build_action_from_siblings(name: &str, args: &Value) -> Result<(Value, Option<String>), String> {
    let Some(obj) = args.as_object() else {
        return Err(corrective_action_error(args));
    };
    let mut siblings = serde_json::Map::new();
    for (k, v) in obj {
        if k == "action" || k == "tab" {
            continue;
        }
        siblings.insert(k.clone(), v.clone());
    }

    // A sibling named after the action holds the full payload object
    // (e.g. {"action":"open","open":{"url":"..."}}).
    if let Some(payload) = siblings.remove(name) {
        let note = format!("sibling '{name}' object used as action payload");
        match payload {
            Value::Object(o) => {
                require_find_action(name, &o)?;
                Ok((json!({name: o}), Some(note)))
            }
            Value::String(s) if name == "open" => Ok((json!({"open": {"url": s}}), Some(note))),
            Value::String(s) if name == "find" => {
                // Stringified JSON payload ({"action":"find","find":"{...}"}).
                let parsed: Value = serde_json::from_str(&s)
                    .map_err(|_| corrective_action_error(&Value::String(s.clone())))?;
                let Value::Object(o) = parsed else {
                    return Err(corrective_action_error(&Value::String(s)));
                };
                require_find_action(name, &o)?;
                Ok((json!({"find": o}), Some(note)))
            }
            _ => Err(corrective_action_error(&Value::Object(siblings))),
        }
    } else {
        match name {
            // open requires a url.
            "open" => {
                let url = siblings
                    .remove("url")
                    .ok_or_else(|| corrective_action_error(&Value::Object(siblings.clone())))?;
                Ok((
                    json!({"open": {"url": url}}),
                    Some("flattened 'url' wrapped into open action".to_string()),
                ))
            }
            // find requires by/value/action.
            "find" => {
                if !siblings.contains_key("action") {
                    return Err(corrective_action_error(&Value::Object(siblings)));
                }
                Ok((
                    json!({"find": siblings}),
                    Some("flattened find fields wrapped into find action".to_string()),
                ))
            }
            _ => Ok((
                json!({name: siblings}),
                Some(format!("flattened fields wrapped into {name} action")),
            )),
        }
    }
}

/// Parse the exact XML-wrapped action strings observed in production
/// (no general XML parser — only these patterns).
fn parse_xml_action(s: &str) -> Option<(Value, &'static str)> {
    let s = s.trim();
    // <open><url>URL</url></open> or <open>URL</open>, with optional
    // whitespace/newlines inside, and a possible stray `</action>` suffix.
    let s = s.strip_suffix("</action>").map_or(s, str::trim);
    if let Some(inner) = s
        .strip_prefix("<open>")
        .and_then(|r| r.strip_suffix("</open>"))
    {
        let inner = inner.trim();
        if inner.is_empty() {
            return None; // <open></open> — open requires a url.
        }
        if let Some(url) = inner
            .strip_prefix("<url>")
            .and_then(|r| r.strip_suffix("</url>"))
        {
            return Some((
                json!({"open": {"url": url.trim()}}),
                "XML <open><url>…</url></open>",
            ));
        }
        return Some((json!({"open": {"url": inner}}), "XML <open>URL</open>"));
    }
    // <find><by>..</by><value>..</value><action>..</action></find>
    if let Some(inner) = s
        .strip_prefix("<find>")
        .and_then(|r| r.strip_suffix("</find>"))
    {
        let mut fields = serde_json::Map::new();
        for (tag, key) in [
            ("by", "by"),
            ("value", "value"),
            ("action", "action"),
            ("text", "text"),
            ("name", "name"),
            ("exact", "exact"),
            ("index", "index"),
        ] {
            if let Some(v) = extract_xml_field(inner, tag) {
                fields.insert(key.to_string(), json!(v));
            }
        }
        // exact/index arrive as XML text ("true", "3") — decode them to the
        // JSON types the serde struct expects instead of failing deserialization.
        if let Some(v) = fields
            .get("exact")
            .and_then(|v| v.as_str())
            .and_then(|v| v.parse::<bool>().ok())
        {
            fields.insert("exact".to_string(), json!(v));
        }
        if let Some(v) = fields
            .get("index")
            .and_then(|v| v.as_str())
            .and_then(|v| v.parse::<u32>().ok())
        {
            fields.insert("index".to_string(), json!(v));
        }
        if !fields.is_empty() {
            return Some((json!({"find": fields}), "XML <find>…</find>"));
        }
    }
    // Mangled `<parameter name="open" ...>` serializations observed in
    // production: either a JSON payload or a nested `<parameter name="url" ...>`
    // holding the bare URL.
    if let Some(rest) = s.strip_prefix("<parameter name=\"open\"") {
        let content = rest.split_once('>').map_or(rest, |(_, r)| r).trim();
        if content.starts_with('{') {
            if let Ok(parsed) = serde_json::from_str::<Value>(content) {
                return Some((
                    json!({"open": parsed}),
                    "XML <parameter name=\"open\"> JSON",
                ));
            }
        } else if let Some(url) = content
            .strip_prefix("<parameter name=\"url\"")
            .and_then(|r| r.split_once('>').map(|(_, r)| r))
        {
            let url = url.trim().trim_end_matches("</parameter>").trim();
            if !url.is_empty() {
                return Some((
                    json!({"open": {"url": url}}),
                    "XML <parameter name=\"open\"> URL",
                ));
            }
        } else if !content.is_empty() {
            return Some((
                json!({"open": {"url": content}}),
                "XML <parameter name=\"open\"> URL",
            ));
        }
    }
    None
}

fn extract_xml_field(inner: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = inner.find(&open)? + open.len();
    let end = inner[start..].find(&close)? + start;
    let v = inner[start..end].trim();
    if v.is_empty() {
        None
    } else {
        Some(v.to_string())
    }
}

/// Escape a string for embedding in a single-quoted JavaScript literal.
fn escape_js_single_quoted(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

/// Build eval JS that returns `innerText` for the given CSS selector or ref.
fn inner_text_eval_js(selector: &str) -> String {
    let escaped = escape_js_single_quoted(selector);
    format!(
        "(() => {{ const el = document.querySelector('{escaped}'); return el ? el.innerText : ''; }})()"
    )
}

/// Extract textual content from an chrome-use snapshot response `data` field.
///
/// chrome-use can return the snapshot as:
/// - A plain string (via `snapshot -c`)
/// - An object with a `content` field (via `get_text`)
/// - An object with `origin`, `refs`, and `snapshot` fields (via `open` auto-snapshot)
///
/// Returns `None` if none of these shapes match.
fn extract_snapshot_text(data: &serde_json::Value) -> Option<String> {
    data.as_str()
        .map(String::from)
        .or_else(|| {
            data.get("content")
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .or_else(|| {
            data.get("snapshot")
                .and_then(|v| v.as_str())
                .map(String::from)
        })
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::browser_daemon::{browser_bin, ensure_browser_env};
    use crate::util::test::set_env_var;

    #[test]
    fn url_validation_rejects_bad_urls() {
        for url in &["", "file:///etc/passwd", "ftp://example.com"] {
            assert!(
                BrowserTool::validate_url(url).is_err(),
                "expected reject for {url}"
            );
        }
    }

    #[test]
    fn url_validation_accepts_all_domains() {
        assert!(BrowserTool::validate_url("https://example.com").is_ok());
        assert!(BrowserTool::validate_url("https://docs.example.com").is_ok());
        assert!(BrowserTool::validate_url("https://other.com").is_ok());
    }

    // ── build_args: simple actions ───────────────────────────────────────

    #[test]
    fn build_args_for_simple_actions() {
        struct Case {
            name: &'static str,
            action: BrowserAction,
            expected: &'static [&'static str],
        }

        let cases = [
            Case {
                name: "open",
                action: BrowserAction::Open {
                    url: "https://example.com".into(),
                },
                expected: &["open", "https://example.com"],
            },
            Case {
                name: "snapshot",
                action: BrowserAction::Snapshot {
                    interactive_only: true,
                    compact: true,
                    depth: Some(5),
                },
                expected: &["snapshot", "-i", "-c", "-d", "5"],
            },
            Case {
                name: "click",
                action: BrowserAction::Click {
                    selector: "@e1".into(),
                },
                expected: &["click", "@e1"],
            },
            Case {
                name: "get_text",
                action: BrowserAction::GetText {
                    selector: "@e3".into(),
                },
                expected: &["get", "text", "@e3"],
            },
            Case {
                name: "get_url",
                action: BrowserAction::GetUrl {},
                expected: &["get", "url"],
            },
        ];

        for case in &cases {
            let args = BrowserTool::build_args(&case.action).unwrap_or_else(|e| {
                panic!("{}: build_args failed: {}", case.name, e);
            });
            assert_eq!(args, case.expected, "{}", case.name);
        }
    }

    // ── build_args: GetInnerText error ──────────────────────────────────

    #[test]
    fn build_args_rejects_get_innertext() {
        let action = BrowserAction::GetInnerText {
            selector: "body".into(),
        };
        assert!(
            BrowserTool::build_args(&action).is_err(),
            "GetInnerText must be handled in execute(), not build_args"
        );
    }

    #[test]
    fn inner_text_eval_js_escapes_quotes() {
        let js = inner_text_eval_js("it's");
        assert!(js.contains("it\\'s"));
        assert!(!js.contains("innertext"));
    }

    #[test]
    fn inner_text_eval_js_body_and_ref() {
        let body = inner_text_eval_js("body");
        assert!(body.contains("document.querySelector('body')"));
        assert!(body.contains("innerText"));

        let refr = inner_text_eval_js("@e1");
        assert!(refr.contains("document.querySelector('@e1')"));
    }

    // ── build_args: Find variants ────────────────────────────────────────

    #[expect(clippy::too_many_lines)]
    #[test]
    fn build_args_for_find_variants() {
        struct Case {
            name: &'static str,
            by: &'static str,
            value: &'static str,
            action: &'static str,
            text: Option<&'static str>,
            find_name: Option<&'static str>,
            exact: Option<bool>,
            index: Option<u32>,
            expected: &'static [&'static str],
        }

        let cases = [
            Case {
                name: "find_by_text_click",
                by: "text",
                value: "Sign In",
                action: "click",
                text: None,
                find_name: None,
                exact: None,
                index: None,
                expected: &["find", "text", "Sign In", "click"],
            },
            Case {
                name: "find_by_text_fill",
                by: "text",
                value: "Search",
                action: "fill",
                text: Some("tokio"),
                find_name: None,
                exact: None,
                index: None,
                expected: &["find", "text", "Search", "fill", "tokio"],
            },
            Case {
                name: "find_by_first",
                by: "first",
                value: "a",
                action: "fill",
                text: None,
                find_name: None,
                exact: None,
                index: None,
                expected: &["find", "first", "a", "fill"],
            },
            Case {
                name: "find_by_nth_with_index",
                by: "nth",
                value: ".card",
                action: "hover",
                text: None,
                find_name: None,
                exact: None,
                index: Some(2),
                expected: &["find", "nth", "2", ".card", "hover"],
            },
            Case {
                name: "find_by_nth_default_index",
                by: "nth",
                value: "a",
                action: "click",
                text: None,
                find_name: None,
                exact: None,
                index: None,
                expected: &["find", "nth", "0", "a", "click"],
            },
            Case {
                name: "find_with_name_and_exact",
                by: "role",
                value: "button",
                action: "click",
                text: None,
                find_name: Some("Submit"),
                exact: Some(true),
                index: None,
                expected: &[
                    "find", "role", "button", "click", "--name", "Submit", "--exact",
                ],
            },
            Case {
                name: "find_with_name_only",
                by: "role",
                value: "link",
                action: "click",
                text: None,
                find_name: Some("Docs.rs"),
                exact: None,
                index: None,
                expected: &["find", "role", "link", "click", "--name", "Docs.rs"],
            },
            Case {
                name: "find_check",
                by: "text",
                value: "Accept",
                action: "check",
                text: None,
                find_name: None,
                exact: None,
                index: None,
                expected: &["find", "text", "Accept", "check"],
            },
            Case {
                name: "find_uncheck",
                by: "text",
                value: "Subscribe",
                action: "uncheck",
                text: None,
                find_name: None,
                exact: None,
                index: None,
                expected: &["find", "text", "Subscribe", "uncheck"],
            },
            Case {
                name: "find_text_action",
                by: "first",
                value: ".result",
                action: "text",
                text: None,
                find_name: None,
                exact: None,
                index: None,
                expected: &["find", "first", ".result", "text"],
            },
        ];

        for case in &cases {
            let action = BrowserAction::Find {
                by: case.by.into(),
                value: case.value.into(),
                action: case.action.into(),
                text: case.text.map(String::from),
                name: case.find_name.map(String::from),
                exact: case.exact,
                index: case.index,
            };
            let args = BrowserTool::build_args(&action).unwrap_or_else(|e| {
                panic!("{}: build_args failed: {}", case.name, e);
            });
            assert_eq!(args, case.expected, "{}", case.name);
        }
    }

    #[test]
    fn tool_name_and_description_are_set() {
        let tool = BrowserTool::default();
        assert_eq!(tool.name(), "browser");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn parameters_schema_is_valid_json() {
        let tool = BrowserTool::default();
        let schema = tool.parameters_schema();
        assert!(schema.is_object());
        assert!(
            schema
                .get("properties")
                .and_then(|p| p.get("action"))
                .is_some()
        );
    }

    #[test]
    fn parameters_schema_has_all_actions() {
        let tool = BrowserTool::default();
        let schema = tool.parameters_schema();
        let action_schemas = schema["properties"]["action"]["oneOf"]
            .as_array()
            .expect("oneOf should be an array");

        // There are exactly 10 browser actions.
        assert_eq!(
            action_schemas.len(),
            10,
            "expected 10 actions, got {}",
            action_schemas.len()
        );

        let action_names: Vec<&str> = action_schemas
            .iter()
            .filter_map(|s| {
                s.get("properties")
                    .and_then(|p| p.as_object())
                    .and_then(|props| props.keys().next())
                    .map(String::as_str)
            })
            .collect();

        for expected in &[
            "open",
            "snapshot",
            "click",
            "get_text",
            "get_innertext",
            "get_url",
            "press",
            "eval",
            "find",
            "screenshot",
        ] {
            assert!(
                action_names.contains(expected),
                "schema missing action: {expected}"
            );
        }

        // Structural invariants: snapshot, get_url and screenshot must lack inner
        // "required"; all other actions must have it.
        for s in action_schemas {
            let inner = s
                .get("properties")
                .and_then(|p| p.as_object())
                .and_then(|props| props.values().next())
                .and_then(|v| v.as_object());
            let name = s
                .get("properties")
                .and_then(|p| p.as_object())
                .and_then(|props| props.keys().next())
                .map_or("?", String::as_str);

            let has_inner_required = inner.is_some_and(|obj| obj.contains_key("required"));
            if name == "snapshot" || name == "get_url" || name == "screenshot" {
                assert!(
                    !has_inner_required,
                    "{name} should NOT have inner 'required'"
                );
            } else {
                assert!(has_inner_required, "{name} should have inner 'required'");
            }
        }
    }

    #[test]
    fn ensure_browser_env_sets_home_when_missing() {
        let _guard = set_env_var("HOME", None);
        let mut cmd = Command::new("true");
        ensure_browser_env(&mut cmd);
    }

    #[test]
    fn ensure_browser_env_sets_chromium_flags() {
        let _guard = set_env_var("CHROMIUM_FLAGS", None);
        let mut cmd = Command::new("true");
        ensure_browser_env(&mut cmd);
    }

    #[test]
    fn ensure_browser_env_sets_idle_timeout() {
        let mut cmd = Command::new("true");
        ensure_browser_env(&mut cmd);
        // Function completes without panic.
    }

    #[test]
    fn browser_bin_name_is_correct() {
        let name = browser_bin();
        if cfg!(target_os = "windows") {
            assert_eq!(name, "chrome-use.exe");
        } else {
            assert_eq!(name, "chrome-use");
        }
    }

    // -----------------------------------------------------------------------
    // fetch_page_text validation — error propagation through public method
    // (exercises the early-return preamble without chrome-use)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn fetch_page_text_propagates_url_validation_errors() {
        let tool = BrowserTool::default();

        let err = tool.fetch_page_text("", "test-tab").await.unwrap_err();
        assert!(
            err.to_string().contains("cannot be empty"),
            "expected empty-url error, got: {err}",
        );

        let err = tool
            .fetch_page_text("file:///etc/passwd", "test-tab")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("not allowed"),
            "expected file:// rejection, got: {err}",
        );

        let err = tool
            .fetch_page_text("ftp://example.com", "test-tab")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("Only http:// and https://"),
            "expected scheme rejection, got: {err}",
        );
    }

    // -----------------------------------------------------------------------
    // Tolerant action normalization — each recoverable shape (flat,
    // stringified, bare-open, XML, missing-tab) normalizes and passes; each
    // unrecoverable shape produces the corrective-shape error.
    // -----------------------------------------------------------------------

    fn assert_normalizes(
        action: serde_json::Value,
        args: &serde_json::Value,
        expected: &serde_json::Value,
    ) {
        let (normalized, note) = normalize_action(action, args).unwrap_or_else(|e| {
            panic!("expected normalization to succeed, got: {e}");
        });
        assert_eq!(normalized, *expected, "normalized action mismatch");
        assert!(
            note.is_some(),
            "recoverable shape should echo a normalization note"
        );
    }

    fn assert_rejects(action: serde_json::Value, args: &serde_json::Value) {
        let err = normalize_action(action, args).unwrap_err();
        assert!(
            err.contains("Expected action to be"),
            "expected corrective shape error, got: {err}"
        );
    }

    #[test]
    fn normalize_flat_action_with_sibling_fields() {
        // {"action":"open","url":"..."} → {"open":{"url":"..."}}
        assert_normalizes(
            json!("open"),
            &json!({"action":"open","url":"https://example.com","tab":"t"}),
            &json!({"open":{"url":"https://example.com"}}),
        );
        // {"action":"open","open":{"url":"..."}} → same
        assert_normalizes(
            json!("open"),
            &json!({"action":"open","open":{"url":"https://example.com"},"tab":"t"}),
            &json!({"open":{"url":"https://example.com"}}),
        );
        // {"action":"open","open":"https://..."} → {"open":{"url":"..."}}
        assert_normalizes(
            json!("open"),
            &json!({"action":"open","open":"https://example.com","tab":"t"}),
            &json!({"open":{"url":"https://example.com"}}),
        );
        // snapshot with no args
        assert_normalizes(
            json!("snapshot"),
            &json!({"action":"snapshot","tab":"t"}),
            &json!({"snapshot":{}}),
        );
        // get_text with flattened selector
        assert_normalizes(
            json!("get_text"),
            &json!({"action":"get_text","selector":"body","tab":"t"}),
            &json!({"get_text":{"selector":"body"}}),
        );
    }

    #[test]
    fn normalize_stringified_json_action() {
        // {"action":"{\"open\":{\"url\":\"...\"}}"} → {"open":{"url":"..."}}
        assert_normalizes(
            json!("{\"open\": {\"url\": \"https://example.com\"}}"),
            &json!({"action":"{\"open\": {\"url\": \"https://example.com\"}}","tab":"t"}),
            &json!({"open":{"url":"https://example.com"}}),
        );
        // CDATA-wrapped JSON
        assert_normalizes(
            json!("<![CDATA[{\"open\": {\"url\": \"https://example.com\"}}]]>"),
            &json!({"action":"<![CDATA[{\"open\": {\"url\": \"https://example.com\"}}]]>","tab":"t"}),
            &json!({"open":{"url":"https://example.com"}}),
        );
    }

    #[test]
    fn normalize_bare_open_value() {
        // {"action":{"open":"https://..."}} → {"open":{"url":"..."}}
        assert_normalizes(
            json!({"open":"https://example.com"}),
            &json!({"action":{"open":"https://example.com"},"tab":"t"}),
            &json!({"open":{"url":"https://example.com"}}),
        );
        // bare URL string as the whole action
        assert_normalizes(
            json!("https://example.com"),
            &json!({"action":"https://example.com","tab":"t"}),
            &json!({"open":{"url":"https://example.com"}}),
        );
    }

    #[test]
    fn normalize_xml_wrapped_actions() {
        assert_normalizes(
            json!("<open><url>https://example.com</url></open>"),
            &json!({"action":"<open><url>https://example.com</url></open>","tab":"t"}),
            &json!({"open":{"url":"https://example.com"}}),
        );
        assert_normalizes(
            json!("<open>https://example.com</open>"),
            &json!({"action":"<open>https://example.com</open>","tab":"t"}),
            &json!({"open":{"url":"https://example.com"}}),
        );
        // whitespace inside tags
        assert_normalizes(
            json!("<open>\n<url>https://example.com</url>\n</open>"),
            &json!({"action":"<open>\n<url>https://example.com</url>\n</open>","tab":"t"}),
            &json!({"open":{"url":"https://example.com"}}),
        );
        // stray closing tag
        assert_normalizes(
            json!("<open><url>https://example.com</url></open></action>"),
            &json!({"action":"<open><url>https://example.com</url></open></action>","tab":"t"}),
            &json!({"open":{"url":"https://example.com"}}),
        );
        // find
        assert_normalizes(
            json!("<find><by>first</by><value>a</value><action>click</action></find>"),
            &json!({"action":"<find><by>first</by><value>a</value><action>click</action></find>","tab":"t"}),
            &json!({"find":{"by":"first","value":"a","action":"click"}}),
        );
        // find with exact/index — XML text decoded to the serde bool/u32 types
        assert_normalizes(
            json!(
                "<find><by>text</by><value>submit</value><action>click</action><exact>true</exact></find>"
            ),
            &json!({"action":"<find><by>text</by><value>submit</value><action>click</action><exact>true</exact></find>","tab":"t"}),
            &json!({"find":{"by":"text","value":"submit","action":"click","exact":true}}),
        );
        assert_normalizes(
            json!(
                "<find><by>nth</by><value>input</value><action>click</action><index>2</index></find>"
            ),
            &json!({"action":"<find><by>nth</by><value>input</value><action>click</action><index>2</index></find>","tab":"t"}),
            &json!({"find":{"by":"nth","value":"input","action":"click","index":2}}),
        );
    }

    #[test]
    fn blank_page_url_predicate() {
        // Scratch/about pages that never committed are blank-page failures.
        assert!(is_blank_page_url("about:blank"));
        assert!(is_blank_page_url("about:blank#blocked"));
        assert!(is_blank_page_url(""));
        // Real pages — even ones that render no text — are NOT blank failures
        // (image URLs, PDFs, canvas shells).
        assert!(!is_blank_page_url("https://example.com/image.png"));
        assert!(!is_blank_page_url("https://example.com/file.pdf"));
        assert!(!is_blank_page_url("about:srcdoc"));
    }

    #[test]
    fn missing_or_empty_tab_defaults_to_default() {
        // Missing tab → documented default session, echoed in tool output.
        let (tab, note) = normalize_tab(&json!({"action":{"open":{"url":"https://example.com"}}}));
        assert_eq!(tab, "default");
        assert!(note.is_some(), "defaulting should be echoed in tool output");
        // Empty tab → same defaulting.
        let (tab, note) = normalize_tab(&json!({"tab":"","action":{"open":{"url":"x"}}}));
        assert_eq!(tab, "default");
        assert!(note.is_some());
        // Explicit tab passes through unchanged.
        let (tab, note) = normalize_tab(&json!({"tab":"docs","action":{"open":{"url":"x"}}}));
        assert_eq!(tab, "docs");
        assert!(note.is_none());
    }

    #[test]
    fn normalize_rejects_unrecoverable_shapes() {
        // Empty payload
        assert_rejects(json!(null), &json!({}));
        // Invented action variants
        assert_rejects(
            json!({"expand":15}),
            &json!({"action":{"expand":15},"tab":"t"}),
        );
        assert_rejects(
            json!({"__raw":"{\"open\":{...}}"}),
            &json!({"action":{"__raw":"x"},"tab":"t"}),
        );
        // Unknown action name
        assert_rejects(json!("navigate"), &json!({"action":"navigate","tab":"t"}));
        // find missing its own `action` field
        assert_rejects(
            json!({"find":{"by":"text","value":"x"}}),
            &json!({"action":{"find":{"by":"text","value":"x"}},"tab":"t"}),
        );
        // open without url
        assert_rejects(json!("open"), &json!({"action":"open","tab":"t"}));
    }

    // -----------------------------------------------------------------------
    // Daemon-unavailable fail-fast path (guidance without the ~152s retry burn)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn fail_fast_returns_guidance_without_retrying() {
        let _guard = crate::tools::browser_daemon::with_health_test_lock().await;
        // An orphaned-tab error (even envelope-wrapped with the auto-connect
        // and daemon-wrapper text) fails fast with hand-close guidance but
        // does NOT mark the daemon unhealthy — the relay and daemon are up, so
        // recovery must not wake for it.
        let err = BrowserTool::fail_fast_if_daemon_down(
            "Auto-launch failed: Could not drive your Chrome through the ab-connect extension. \
             The tab this session was driving can no longer be resolved (it was closed, or a \
             flaky relay dropped it)",
            Some("browser_not_launched"),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("close the leftover tab in Chrome"),
            "expected hand-close guidance, got: {err}"
        );
        assert!(crate::tools::browser_daemon::is_advertised());
        // Daemon-unavailable signature → actionable guidance, daemon marked
        // unhealthy (wakes the auto-recovery watchdog).
        let err = BrowserTool::fail_fast_if_daemon_down(
            "Failed to read: Resource temporarily unavailable (os error 35) (after 5 retries - daemon may be busy or unresponsive)",
            None,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("browser daemon is down"),
            "expected guidance message, got: {err}"
        );
        assert!(!crate::tools::browser_daemon::is_advertised());
        // The unambiguous daemon-side envelope code fails fast too (no message
        // matching).
        let err = BrowserTool::fail_fast_if_daemon_down(
            "chrome-use error: browser not launched",
            Some("browser_not_launched"),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("browser daemon is down"),
            "expected guidance message, got: {err}"
        );
        // Site-level failures — even ones carrying the coarse `connection_failed`
        // code (the CLI assigns it to any "connection" text) — pass through as
        // truthful navigation failures and leave the daemon healthy.
        assert!(
            BrowserTool::fail_fast_if_daemon_down(
                "chrome-use error: Navigation failed: net::ERR_CONNECTION_REFUSED",
                Some("connection_failed"),
            )
            .is_ok()
        );
        assert!(
            BrowserTool::fail_fast_if_daemon_down("chrome-use error: Element not found", None)
                .is_ok()
        );
        assert!(
            BrowserTool::fail_fast_if_daemon_down("chrome-use error: timed out", Some("timeout"))
                .is_ok()
        );
        // Restore the global health singleton so later agent-constructing
        // tests don't inherit a hidden browser tool.
        crate::tools::browser_daemon::reset_health();
    }

    // -----------------------------------------------------------------------
    // KNOWN_ACTIONS lockstep — adding a BrowserAction variant must be mirrored
    // in the normalization allowlist.
    // -----------------------------------------------------------------------

    #[test]
    fn known_actions_lockstep_with_browser_action_variants() {
        // Minimal payload per variant/alias name (KNOWN_ACTIONS also carries the
        // serde aliases on GetInnerText).
        let payload = |name: &str| -> Value {
            match name {
                "open" => json!({"url": "https://example.com"}),
                "snapshot" | "get_url" | "screenshot" => json!({}),
                "click" => json!({"selector": "@e1"}),
                "get_text" => json!({"selector": "body"}),
                "get_inner_text" | "get_innertext" | "innertext" => {
                    json!({"selector": "body"})
                }
                "press" => json!({"key": "Enter"}),
                "eval" => json!({"js": "1 + 1"}),
                "find" => json!({"by": "text", "value": "x", "action": "click"}),
                other => panic!("KNOWN_ACTIONS entry {other} has no lockstep payload"),
            }
        };
        // Every allowlist entry must be a real variant name — stale entries
        // (e.g. invented actions) fail deserialization here.
        for name in KNOWN_ACTIONS {
            let tagged = json!({*name: payload(name)});
            assert!(
                serde_json::from_value::<BrowserAction>(tagged.clone()).is_ok(),
                "KNOWN_ACTIONS entry does not deserialize: {tagged}"
            );
        }
        // Every variant's canonical name must be in the allowlist — a new
        // BrowserAction variant fails here.
        for name in [
            "open",
            "snapshot",
            "click",
            "get_text",
            "get_inner_text",
            "get_url",
            "press",
            "eval",
            "find",
            "screenshot",
        ] {
            assert!(
                KNOWN_ACTIONS.contains(&name),
                "KNOWN_ACTIONS is missing variant {name}"
            );
        }
    }

    #[test]
    fn build_args_rejects_screenshot() {
        let action = BrowserAction::Screenshot {};
        assert!(
            BrowserTool::build_args(&action).is_err(),
            "Screenshot must be handled in execute(), not build_args"
        );
    }

    #[test]
    fn screenshot_action_normalizes_and_parses() {
        // Canonical tagged object.
        let action = serde_json::json!({"screenshot": {}});
        let parsed: BrowserAction = serde_json::from_value(action.clone()).unwrap();
        assert!(matches!(parsed, BrowserAction::Screenshot {}));

        // Plain action name with a tab sibling.
        let args = serde_json::json!({"action": "screenshot", "tab": "docs"});
        let (normalized, note) =
            normalize_action(args.get("action").cloned().unwrap(), &args).unwrap();
        assert!(
            note.is_some(),
            "screenshot should record a normalization note"
        );
        let parsed: BrowserAction = serde_json::from_value(normalized).unwrap();
        assert!(matches!(parsed, BrowserAction::Screenshot {}));
    }

    #[test]
    #[expect(clippy::case_sensitive_file_extension_comparisons)] // the tool itself always emits a lowercase .png name
    fn screenshot_output_path_is_safe_and_unique() {
        // A hostile tab name must not escape the temp dir.
        let p = BrowserTool::screenshot_output_path("../../etc/cron.d").unwrap();
        let file_name = p.file_name().unwrap().to_string_lossy().into_owned();
        assert!(!file_name.contains('/') && !file_name.contains('\\'));
        assert!(file_name.ends_with(".png"));
        // Two calls yield different files (random nonce).
        let p2 = BrowserTool::screenshot_output_path("default").unwrap();
        assert_ne!(p, p2);
    }

    #[tokio::test]
    async fn image_payload_attaches_browser_screenshot() {
        // Build a tiny PNG on disk.
        let img = image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 255]));
        let dir = std::env::temp_dir().join("mahbot-browser-test");
        std::fs::create_dir_all(&dir).unwrap();
        let png_path = dir.join("shot.png");
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        std::fs::write(&png_path, &buf).unwrap();

        let tool = BrowserTool::default();
        *tool.last_screenshot.lock().unwrap_poison() =
            Some(png_path.to_string_lossy().into_owned());

        let payload = tool
            .image_payload(
                &crate::Workspace::default(),
                &serde_json::json!({"action": "screenshot", "tab": "default"}),
            )
            .await
            .expect("screenshot must produce an image payload");
        assert_eq!(payload.source, crate::tools::ImagePayloadSource::Browser);
        assert_eq!(payload.format, "PNG");
        assert!(payload.data_uri.starts_with("data:image/jpeg;base64,"));
        assert!(
            payload
                .attached_annotation()
                .starts_with("Browser screenshot"),
            "annotation must describe it as a browser screenshot: {}",
            payload.attached_annotation()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn image_payload_ignores_non_screenshot_actions() {
        let tool = BrowserTool::default();
        // A stale path must not be attached to a non-screenshot action.
        *tool.last_screenshot.lock().unwrap_poison() = Some("/tmp/stale.png".to_string());
        let payload = tool
            .image_payload(
                &crate::Workspace::default(),
                &serde_json::json!({"action": "open", "url": "https://example.com", "tab": "default"}),
            )
            .await;
        assert!(
            payload.is_none(),
            "non-screenshot action must yield no payload"
        );
    }

    // ── parse_session_list: chrome-use envelope tolerance ───────────────

    #[test]
    fn parse_session_list_latest_object_envelope() {
        let v = serde_json::json!({
            "ok": true,
            "sessions": [{"name": "default", "pid": 1, "owner": "me"}]
        });
        assert_eq!(parse_session_list(&v), vec!["default"]);
    }

    #[test]
    fn parse_session_list_legacy_data_strings() {
        let v = serde_json::json!({
            "success": true,
            "data": {"sessions": ["default", "docs"]}
        });
        assert_eq!(parse_session_list(&v), vec!["default", "docs"]);
    }

    #[test]
    fn parse_session_list_object_entries_under_data() {
        let v = serde_json::json!({
            "success": true,
            "data": {"sessions": [{"name": "docs", "pid": 2}, "default"]}
        });
        assert_eq!(parse_session_list(&v), vec!["docs", "default"]);
    }

    #[test]
    fn parse_session_list_extracts_names_from_both_envelopes() {
        // Envelope-verdict gating lives in the caller (close_all_browser_sessions_inner);
        // this helper only extracts names.
        let v = serde_json::json!({"ok": false, "error": "boom"});
        assert!(parse_session_list(&v).is_empty());
        let v2 = serde_json::json!({"success": false, "data": {"sessions": ["default"]}});
        assert_eq!(parse_session_list(&v2), vec!["default"]);
    }

    #[test]
    fn parse_session_list_garbage_is_empty() {
        let v = serde_json::json!({"ok": true, "sessions": "nope"});
        assert!(parse_session_list(&v).is_empty());
        let v2 = serde_json::json!(null);
        assert!(parse_session_list(&v2).is_empty());
    }

    #[test]
    fn parse_session_list_missing_sessions_key_is_empty() {
        let v = serde_json::json!({"ok": true, "data": {}});
        assert!(parse_session_list(&v).is_empty());
    }

    #[test]
    fn browser_response_accepts_ok_envelope() {
        let resp: BrowserResponse =
            serde_json::from_str(r#"{"ok":true,"data":{}}"#).expect("tolerant deserialize");
        assert!(resp.is_success());
    }

    #[test]
    fn browser_response_accepts_success_envelope() {
        let resp: BrowserResponse =
            serde_json::from_str(r#"{"success":true}"#).expect("tolerant deserialize");
        assert!(resp.is_success());
        let failed: BrowserResponse =
            serde_json::from_str(r#"{"success":false}"#).expect("tolerant deserialize");
        assert!(!failed.is_success());
        // Failure precedence — mirrors envelope_verdict: an explicit false on
        // either key loses over a contradicting success key.
        let mixed: BrowserResponse =
            serde_json::from_str(r#"{"success":false,"ok":true}"#).expect("tolerant deserialize");
        assert!(!mixed.is_success());
    }
}
