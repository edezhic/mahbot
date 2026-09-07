//! Browser automation tool.

use crate::browser::contract::{BrowserResponse, extract_snapshot_text};
use crate::browser::escape_js_single_quoted;
use crate::browser::spawn::{CliRun, CliSpawn, CliTimeout, spawn_cli};
use crate::util::{UnwrapPoison, is_http_url};
use crate::{Tool, Workspace};
use anyhow::Context;
use async_trait::async_trait;
use futures_util::future::join_all;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use std::sync::Arc;
use std::time::Duration;
use tracing::debug;

/// Actions for navigating and extracting content from web pages.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BrowserAction {
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
        /// Action to perform: click, fill, hover, check, text.
        /// "fill" clears the field then types.
        action: String,
        /// Text to fill into the element (only for action "fill").
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

/// Browser sessions one agent run's browser tooling opened. chrome-use
/// ≥1.5.101 no longer closes external Chrome tabs when the daemon idles out,
/// so every session name the run used is recorded here and closed at run end —
/// the created-only close path, never an enumeration sweep (a sweep could
/// close the user's own tabs).
#[derive(Default)]
pub(crate) struct BrowserRunSessions(std::sync::Mutex<BTreeSet<String>>);

impl BrowserRunSessions {
    fn track(&self, name: &str) {
        self.0.lock().unwrap_poison().insert(name.to_string());
    }
    pub(crate) fn snapshot(&self) -> Vec<String> {
        self.0.lock().unwrap_poison().iter().cloned().collect()
    }
}

/// Browser tool for fetching content from web pages.
///
/// Each operation requires a `tab` name — separate browser sessions
/// (isolated via `--session`). The default session is unique per tool
/// instance (one agent run), not the shared `"default"` — concurrent runs
/// never collide on a shared session. Every session used is closed at run end.
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
    /// Lazily generated per-instance default session name — unique per tool
    /// instance (one agent run) so concurrent runs never collide on a shared
    /// session, and closing it at run end can't clobber another run's tabs.
    default_session: std::sync::OnceLock<String>,
    /// Browser sessions this run's browser tooling used — closed at run end.
    browser_sessions: std::sync::Arc<BrowserRunSessions>,
}

impl BrowserTool {
    /// Construct a browser tool for one agent run, sharing the run's session
    /// tracker so every session the run opens is closed at run end.
    pub(crate) fn new(browser_sessions: std::sync::Arc<BrowserRunSessions>) -> Self {
        Self {
            browser_sessions,
            ..Default::default()
        }
    }

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
        crate::browser::validate_url(url)?;

        Self::ensure_available().await?;

        // Lock is held for the entire navigate + extract sequence so
        // concurrent same-tab access doesn't race between navigation
        // and text extraction.
        let _guard = self.acquire_tab_lock(tab).await;
        let opened = self.run_command(&["open", url], tab).await?;

        // A real navigation that ends on the scratch `about:blank` or Chrome's
        // error page never loaded — fail loudly instead of returning empty
        // content.
        self.bail_on_failed_navigation(tab, url, &opened).await?;

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
    /// agent-facing `agent-tab-*` sessions) are refused by the sweep's
    /// strict-scope rule — agent-facing sessions are instead closed at run end
    /// (chrome-use ≥1.5.101 idle no longer closes them).
    pub async fn close_session(&self, tab: &str) {
        super::browser_daemon::sweep_session(tab).await;
    }

    /// If the response shows a failed navigation — the tab never left the
    /// scratch `about:blank` page, or Chrome committed to its error page — fail
    /// with a cause. The blank-page case means the navigation never committed,
    /// so the session is closed (verified sweep) to avoid an orphaned tab; a
    /// committed Chrome error page keeps the tab open — the navigation
    /// committed, so the tab stays reusable for a retry. The close is refused
    /// for non-mahbot session names (strict-scope rule), leaving the tab to the
    /// run-end session close. No-op when the navigation committed.
    async fn bail_on_failed_navigation(
        &self,
        tab: &str,
        url: &str,
        response: &BrowserResponse,
    ) -> anyhow::Result<()> {
        let Some(committed_url) = response
            .data
            .as_ref()
            .and_then(|d| d.get("url"))
            .and_then(Value::as_str)
        else {
            return Ok(());
        };
        if crate::browser::is_chrome_error_page(committed_url) {
            anyhow::bail!(
                "Navigation to {url} failed — Chrome landed on its error page \
                 (chrome-error://chromewebdata/), meaning the site is unreachable (DNS failure, \
                 refused connection, or a blocked/unsafe port). Verify the URL and network."
            );
        }
        if crate::browser::is_blank_page_url(committed_url) {
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

    /// Mahbot-side per-call bounds. chrome-use's own `--timeout` is ignored by
    /// `wait --load networkidle` (always its internal 25s default) and a wedged
    /// daemon hangs the CLI in its ~152s retry loop, so the tool bounds every
    /// dispatch itself: open=15s (page load), wait=10s (networkidle stays
    /// best-effort), everything else=8s (the daemon-side `run_cli_bounded` bound).
    fn call_timeout(args: &[&str]) -> Duration {
        match args.first() {
            Some(&"open") => Duration::from_secs(15),
            Some(&"wait") => Duration::from_secs(10),
            _ => Duration::from_secs(8),
        }
    }

    /// Run an chrome-use command and parse the JSON response.
    async fn run_command(&self, args: &[&str], tab: &str) -> anyhow::Result<BrowserResponse> {
        let cli = super::browser_daemon::cli_path().with_context(|| {
            format!(
                "chrome-use CLI is not available. {}",
                super::browser_daemon::CHROME_USE_INSTALL_HINT
            )
        })?;
        // Record the session name after the CLI path resolved, so a missing
        // CLI never registers a pointless close — the run-end close only needs
        // sessions that were actually dispatched.
        self.browser_sessions.track(tab);

        let mut logged_args: Vec<&str> = args.to_vec();
        logged_args.extend(["--json", "--session", tab]);
        debug!("chrome-use args: {:?}", logged_args);

        let run = spawn_cli(CliSpawn {
            path: &cli,
            args,
            session: Some(tab),
            json: true,
            capture_stderr: true,
            timeout: CliTimeout::Bounded(Self::call_timeout(args)),
            // A timed-out/cancelled call must not leave the chrome-use child
            // running its retry loop in the background.
            cancel_kills: true,
        })
        .await;
        let output = match run {
            CliRun::Output(output) => output,
            CliRun::SpawnFailure => anyhow::bail!("Failed to execute chrome-use CLI"),
            CliRun::TimedOut => {
                // Distinguish "daemon down/wedged" (fail fast with daemon
                // guidance; wakes the watchdog) from "daemon healthy, that
                // call was just slow". The probe covers the wedge case a
                // status check cannot see — see health_after_call_timeout.
                if let Some(down_message) =
                    super::browser_daemon::health_after_call_timeout(tab).await
                {
                    anyhow::bail!("{down_message}");
                }
                let secs = Self::call_timeout(args).as_secs();
                anyhow::bail!(
                    "chrome-use did not answer within {secs}s for `{}` — the call was aborted. \
                     The daemon looked healthy, so this is usually a slow page or a wedged CLI; \
                     retry once before trying another approach.",
                    args.join(" ")
                );
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout);

        // chrome-use returns exit code 1 even when it outputs valid JSON
        // with a structured error message. Try to parse the JSON first to
        // get a meaningful error, fall back to stderr-only bail otherwise.
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let (error_msg, code, retryable) =
                match serde_json::from_str::<BrowserResponse>(&stdout) {
                    Ok(resp) => {
                        let BrowserResponse {
                            error,
                            code,
                            retryable,
                            ..
                        } = resp;
                        (error.unwrap_or_default(), code, retryable)
                    }
                    Err(_) => (stderr.trim().to_string(), None, None),
                };
            let error_msg = if error_msg.is_empty() {
                format!("chrome-use exited with code {}", output.status)
            } else {
                enhance_browser_error(error_msg)
            };
            Self::fail_fast_if_daemon_down(&error_msg, code.as_deref())?;
            anyhow::bail!(
                "chrome-use error: {}",
                with_retry_hint(error_msg, retryable)
            );
        }

        let response: BrowserResponse =
            serde_json::from_str(&stdout).context("Failed to parse chrome-use JSON response")?;

        if !response.is_success() {
            let err = response.error.as_deref().unwrap_or("unknown error");
            let enhanced = enhance_browser_error(err.to_string());
            Self::fail_fast_if_daemon_down(&enhanced, response.code.as_deref())?;
            anyhow::bail!(
                "chrome-use error: {}",
                with_retry_hint(enhanced, response.retryable)
            );
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

    /// The chrome-use CLI takes a different argument shape per action — this
    /// builds the correct argument list for each action.
    fn build_args(action: &BrowserAction) -> anyhow::Result<Vec<String>> {
        match action {
            BrowserAction::Open { url } => {
                crate::browser::validate_url(url)?;
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
/// `BrowserResponse::verdict` first — this only extracts the names and returns
/// an empty list when no session array is present.
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
    let list_output = match spawn_cli(CliSpawn {
        path: &cmd,
        args: &["session", "list"],
        session: None,
        json: true,
        capture_stderr: true,
        timeout: CliTimeout::Unbounded,
        cancel_kills: true,
    })
    .await
    {
        CliRun::Output(o) => o,
        CliRun::SpawnFailure => {
            tracing::debug!("chrome-use not available, skipping browser cleanup: spawn failed");
            return;
        }
        CliRun::TimedOut => {
            tracing::debug!("chrome-use session list timed out, skipping browser cleanup");
            return;
        }
    };

    let sessions: Vec<String> = match serde_json::from_slice::<Value>(&list_output.stdout) {
        Ok(v) => {
            // Gate only on an explicit failure verdict; a payload with neither
            // verdict key proceeds (tolerance-first — unknown future envelopes
            // still get their sessions closed if they carry a sessions array).
            if BrowserResponse::from_value(&v).verdict() == Some(false) {
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
                match spawn_cli(CliSpawn {
                    path: cmd.as_path(),
                    args: &["--session", session_id, "close"],
                    session: None,
                    json: false,
                    capture_stderr: false,
                    cancel_kills: true,
                    timeout: CliTimeout::Unbounded,
                })
                .await
                {
                    CliRun::Output(out) if out.status.success() => {
                        tracing::debug!("Closed chrome-use session: {session_id}");
                    }
                    CliRun::Output(out) => {
                        tracing::warn!(
                            "chrome-use close session '{session_id}' exited with status: {}",
                            out.status
                        );
                    }
                    CliRun::SpawnFailure => {
                        tracing::warn!(
                            "failed to close chrome-use session '{session_id}': spawn failed"
                        );
                    }
                    CliRun::TimedOut => {
                        tracing::warn!(
                            "failed to close chrome-use session '{session_id}': timed out"
                        );
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

    fn parameters_schema(&self) -> Value {
        // oneOf entry order follows the shared registry (which is ordered for
        // the CLI help), so it may differ from the pre-registry hand-built
        // order; descriptions and structure are unchanged and the set is
        // pinned by `parameters_schema_has_all_actions`.
        let entries: Vec<Value> = crate::browser::actions::ACTIONS
            .iter()
            .filter_map(|a| {
                a.tool.as_ref().map(|t| {
                    super::action_entry_schema(a.name, a.purpose, t.required, &t.properties)
                })
            })
            .collect();
        json!({
            "type": "object",
            "properties": {
                "action": { "oneOf": entries },
                "tab": {
                    "type": "string",
                    "description": "Logical name for this browser session. \
                     Missing or empty defaults to a unique per-run session \
                     (closed automatically when your run ends). Only use an \
                     explicit name (e.g. \"docs\", \"github\") if you need to \
                     keep multiple pages open simultaneously. Same tab = \
                     serialized operations on that page."
                }
            },
            "required": ["action", "tab"]
        })
    }

    async fn execute(&self, _ws: &Workspace, args: Value) -> anyhow::Result<String> {
        let (tab, action, normalized_notes) = self.normalize_call(&args)?;

        debug!(tab, action = ?action, "browser action");

        Self::ensure_available().await?;

        Self::validate_find(&action)?;

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
            return Ok(super::with_normalization_notes(body, &normalized_notes));
        }

        if let BrowserAction::Screenshot { .. } = &action {
            let output = self.capture_screenshot(&tab).await?;
            return Ok(super::with_normalization_notes(output, &normalized_notes));
        }

        let (response, snapshot) = self.run_action(&action, &tab).await?;
        Ok(Self::format_action_output(
            &action,
            &tab,
            response,
            &snapshot,
            &normalized_notes,
        ))
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
        // Residual edge, accepted by design: a PNG corrupt beyond its IHDR
        // passes capture_screenshot's header validation and reaches here, the
        // decode fails, and only the textual marker is delivered (no image
        // part) — the header gate keeps the common corrupt-capture case an
        // explicit error instead.
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
    /// Resolve the tab for a call. An explicit non-empty tab passes through
    /// unchanged; missing/empty falls back to the per-run default session — unique
    /// per `BrowserTool` instance (one agent run) so concurrent runs never collide
    /// on a shared session, and closing it at run end can't clobber another run's
    /// tabs. Defaulting is echoed in tool output.
    fn normalize_tab(&self, args: &Value) -> (String, Option<String>) {
        if let Some(tab) = super::get_opt_str(args, "tab").filter(|s| !s.is_empty()) {
            (tab.to_string(), None)
        } else {
            let default = self
                .default_session
                .get_or_init(|| format!("agent-tab-{}", crate::generate_suffix()))
                .clone();
            let note = format!("missing/empty tab defaulted to \"{default}\"");
            (default, Some(note))
        }
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
        // A corrupt/truncated PNG would otherwise emit a phantom `[IMAGE:…]`
        // marker that the payload path fails open on — gate the emission here.
        let (width, height) = validate_png(&path)?;
        *self.last_screenshot.lock().unwrap_poison() = Some(path_str.clone());
        Ok(format!(
            "[Tab: {tab}] Captured a browser screenshot: {path_str} ({width}x{height}). \
             [IMAGE:{path_str}]"
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

    /// Normalize the raw tool arguments into the parsed action plus the tab
    /// and any normalization notes. LLM-facing corrective texts (action shape,
    /// find-hint) live with the tool, not the shared browser core.
    fn normalize_call(&self, args: &Value) -> anyhow::Result<(String, BrowserAction, Vec<String>)> {
        let mut normalized_notes: Vec<String> = Vec::new();
        let (tab, tab_note) = self.normalize_tab(args);
        if let Some(note) = tab_note {
            normalized_notes.push(note);
        }

        let action_value = args
            .get("action")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!(corrective_action_error(args)))?;
        let (action_value, normalized_note) =
            normalize_action(action_value, args).map_err(|e| anyhow::anyhow!(e))?;
        if let Some(note) = normalized_note {
            normalized_notes.push(format!("action normalized: {action_value} ({note})"));
        }

        let action: BrowserAction = serde_json::from_value(action_value.clone()).map_err(|e| {
            // Give a more helpful message when the LLM uses wrong field names,
            // always including the exact expected shape so the model can
            // self-correct in one round-trip.
            let hint = match &action_value {
                Value::Object(map) if map.contains_key("find") => {
                    " 'find' requires 'by', 'value', and 'action' fields (use 'value' not 'name' for the locator text). Valid 'action' values: click, fill, hover, check, text (use 'text' param only for fill)".to_string()
                }
                _ => String::new(),
            };
            anyhow::anyhow!(
                "Invalid browser action arguments{hint}. Expected action to be {EXPECTED_ACTION_SHAPE}, \
                 plus a \"tab\" string. Serde error: {e}"
            )
        })?;

        Ok((tab, action, normalized_notes))
    }

    /// Validate a `Find` action's locator type/action payload early for better
    /// diagnostics — before any CLI dispatch.
    fn validate_find(action: &BrowserAction) -> anyhow::Result<()> {
        if let BrowserAction::Find {
            by,
            action: find_action,
            name,
            exact,
            index,
            ..
        } = action
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
            // chrome-use 1.5.101 rejects focus/type/uncheck with 'Unknown
            // subaction' despite --help listing them — only advertise/resolve
            // the subset that actually dispatches.
            let valid_actions = ["click", "hover", "fill", "check", "text"];
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
            // For these CSS-selector locators chrome-use's parser consumes
            // everything after the action as the fill value, so `--name` /
            // `--exact` would be eaten as fill text — reject instead of
            // silently stripping them.
            if (name.is_some() || *exact == Some(true))
                && matches!(by.as_str(), "nth" | "first" | "last" | "testid")
            {
                anyhow::bail!(
                    "`name`/`exact` are not honored for the '{by}' locator — chrome-use's parser \
                     treats everything after the action as fill text, so they would be consumed \
                     as the fill value. Drop `name`/`exact` when using '{by}' (they only apply \
                     to role/text-style locators) and match by `value` instead."
                );
            }
        }
        Ok(())
    }

    /// Orchestrate a single browser action: build the args, dispatch, and (for
    /// `Open`) the blank-navigation guard, the best-effort network-idle wait,
    /// and the compact auto-snapshot. Returns the RAW response and the snapshot
    /// text — no LLM-facing framing (that lives in [`Self::format_action_output`]).
    async fn run_action(
        &self,
        action: &BrowserAction,
        tab: &str,
    ) -> anyhow::Result<(BrowserResponse, String)> {
        let cli_args = Self::build_args(action)?;
        let str_args: Vec<&str> = cli_args.iter().map(String::as_str).collect();
        let response = self.run_command(&str_args, tab).await?;

        // A real navigation that ends on the scratch `about:blank` or Chrome's
        // error page never loaded — fail loudly (closing the blank-page tab
        // best-effort) instead of reporting success with no content.
        if let BrowserAction::Open { url } = action {
            self.bail_on_failed_navigation(tab, url, &response).await?;
        }

        // After open, wait for network idle, then auto-snapshot
        // so the LLM sees page content immediately.
        let snapshot = if matches!(action, BrowserAction::Open { .. }) {
            let wait_args = ["wait", "--load", "networkidle"];
            let _ = self.run_command(&wait_args, tab).await;

            // Run a compact snapshot to return page content.
            match self.run_command(&["snapshot", "-c"], tab).await {
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

        Ok((response, snapshot))
    }

    /// Shape the raw [`BrowserResponse`] into the LLM-facing text: the
    /// per-action data extraction, the `[Tab: {tab}]` framing, and the
    /// normalization notes.
    fn format_action_output(
        action: &BrowserAction,
        tab: &str,
        response: BrowserResponse,
        snapshot: &str,
        notes: &[String],
    ) -> String {
        let output = match response.data {
            Some(data) => match action {
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
                    if !snapshot.is_empty() {
                        use std::fmt::Write;
                        let _ = write!(s, "\n\n--- Page content ---\n{snapshot}");
                    }
                    s
                }
                BrowserAction::GetUrl { .. } => data
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                BrowserAction::Eval { .. } => data.get("result").map_or_else(
                    || serde_json::to_string_pretty(&data).unwrap_or_else(|_| data.to_string()),
                    |v| v.as_str().map_or_else(|| v.to_string(), str::to_string),
                ),
                _ => serde_json::to_string_pretty(&data).unwrap_or_else(|_| data.to_string()),
            },
            None => String::new(),
        };

        let output = if output.is_empty() {
            format!("[Tab: {tab}] (no output)")
        } else {
            format!("[Tab: {tab}] {output}")
        };

        super::with_normalization_notes(output, notes)
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Validate a screenshot file is a real PNG: magic bytes + IHDR dimensions
/// (bytes 16..24, big-endian). chrome-use can report success yet write a
/// corrupt/truncated file, and `image_payload` fails open on undecodable
/// rasters — so the validation gates the `[IMAGE:…]` marker emission here,
/// where a corrupt capture becomes an explicit error instead of a phantom
/// image part.
fn validate_png(path: &Path) -> anyhow::Result<(u32, u32)> {
    const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    // Only the header is needed — magic bytes + IHDR dimensions (bytes 16..24).
    let mut header = [0u8; 24];
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to read screenshot {}", path.display()))?;
    std::io::Read::read_exact(&mut file, &mut header).map_err(|_| {
        anyhow::anyhow!(
            "Screenshot at {} is not a valid PNG (truncated or wrong format)",
            path.display()
        )
    })?;
    if header[..8] != PNG_MAGIC {
        anyhow::bail!(
            "Screenshot at {} is not a valid PNG (wrong format)",
            path.display()
        );
    }
    let width = u32::from_be_bytes(header[16..20].try_into().expect("slice is 4 bytes"));
    let height = u32::from_be_bytes(header[20..24].try_into().expect("slice is 4 bytes"));
    if width == 0 || height == 0 {
        anyhow::bail!(
            "Screenshot at {} has an empty IHDR ({}x{})",
            path.display(),
            width,
            height
        );
    }
    Ok((width, height))
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

/// Append the envelope's retry hint when chrome-use itself marked the error
/// `retryable` — otherwise the model cannot tell a worth-retrying transient
/// (e.g. element not yet present) from a dead end.
fn with_retry_hint(error: String, retryable: Option<bool>) -> String {
    if retryable == Some(true) {
        format!("{error} (chrome-use marked this error retryable — a retry may succeed)")
    } else {
        error
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
    \"value\":\"...\",\"action\":\"click|fill|hover|check|text\"}}, \
    {\"screenshot\":{}}";

/// Corrective error for an unrecoverable action shape, listing the exact
/// expected form instead of raw serde text.
fn corrective_action_error(received: &Value) -> String {
    format!(
        "Invalid browser action arguments. Expected action to be {EXPECTED_ACTION_SHAPE}, \
         plus a \"tab\" string. Received: {received}"
    )
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

/// Build eval JS that returns `innerText` for the given CSS selector or ref.
fn inner_text_eval_js(selector: &str) -> String {
    let escaped = escape_js_single_quoted(selector);
    format!(
        "(() => {{ const el = document.querySelector('{escaped}'); return el ? el.innerText : ''; }})()"
    )
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::browser_daemon::browser_bin;

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

    // ── validate_find: runtime-rejected actions ───────────────────────────

    /// focus/type/uncheck are listed by `chrome-use --help` but 1.5.101
    /// rejects them at runtime with 'Unknown subaction' — validate_find must
    /// reject them up front.
    #[test]
    fn validate_find_rejects_runtime_unsupported_actions() {
        for bad_action in ["focus", "type", "uncheck"] {
            let action = BrowserAction::Find {
                by: "text".into(),
                value: "anything".into(),
                action: bad_action.into(),
                text: None,
                name: None,
                exact: None,
                index: None,
            };
            let err = BrowserTool::validate_find(&action)
                .expect_err("focus/type/uncheck must be rejected")
                .to_string();
            assert!(
                err.contains("Invalid 'find' action"),
                "action {bad_action}: {err}"
            );
        }
    }

    #[test]
    fn validate_find_rejects_name_exact_on_css_locators() {
        let find = |by: &str, name: Option<&str>, exact: Option<bool>| BrowserAction::Find {
            by: by.into(),
            value: ".card".into(),
            action: "click".into(),
            text: None,
            name: name.map(String::from),
            exact,
            index: if by == "nth" { Some(0) } else { None },
        };
        for by in ["nth", "first", "last", "testid"] {
            let err = BrowserTool::validate_find(&find(by, Some("Submit"), None))
                .expect_err("name with CSS-selector locator must be rejected")
                .to_string();
            assert!(
                err.contains("`name`/`exact`") && err.contains(by),
                "by {by}: {err}"
            );
            let err = BrowserTool::validate_find(&find(by, None, Some(true)))
                .expect_err("exact with CSS-selector locator must be rejected")
                .to_string();
            assert!(
                err.contains("`name`/`exact`") && err.contains(by),
                "by {by}: {err}"
            );
        }
        // role-style locators still honor name/exact.
        BrowserTool::validate_find(&find("role", Some("Submit"), Some(false)))
            .expect("role + name/exact must still validate");
    }

    // ── call_timeout policy ──────────────────────────────────────────────

    #[test]
    fn call_timeout_policy() {
        assert_eq!(
            BrowserTool::call_timeout(&["open", "https://example.com"]),
            Duration::from_secs(15)
        );
        assert_eq!(
            BrowserTool::call_timeout(&["wait", "--load", "networkidle"]),
            Duration::from_secs(10)
        );
        for args in [
            &["click", "@e1"][..],
            &["eval", "1+1"][..],
            &["snapshot", "-c"][..],
        ] {
            assert_eq!(
                BrowserTool::call_timeout(args),
                Duration::from_secs(8),
                "args: {args:?}"
            );
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
    fn missing_or_empty_tab_defaults_to_per_run_session() {
        // Missing tab → the per-run default session, echoed in tool output.
        let tool = BrowserTool::default();
        let (tab, note) =
            tool.normalize_tab(&json!({"action":{"open":{"url":"https://example.com"}}}));
        assert!(
            tab.starts_with("agent-tab-"),
            "missing tab should default to a per-run session, got {tab}"
        );
        assert!(note.is_some(), "defaulting should be echoed in tool output");
        // Empty tab → same defaulting …
        let (tab, note) = tool.normalize_tab(&json!({"tab":"","action":{"open":{"url":"x"}}}));
        assert!(tab.starts_with("agent-tab-"));
        assert!(note.is_some());
        // … and the SAME tool yields the SAME name (OnceLock stability).
        let (tab2, _note) = tool.normalize_tab(&json!({"tab":"","action":{"open":{"url":"x"}}}));
        assert_eq!(
            tab, tab2,
            "default session must be stable per tool instance"
        );
        // A DIFFERENT BrowserTool instance yields a DIFFERENT name (no
        // cross-run collision).
        let other = BrowserTool::default();
        let (other_tab, _note) =
            other.normalize_tab(&json!({"tab":"","action":{"open":{"url":"x"}}}));
        assert_ne!(tab, other_tab, "each tool instance needs its own session");
        // Explicit tab passes through unchanged.
        let (tab, note) = tool.normalize_tab(&json!({"tab":"docs","action":{"open":{"url":"x"}}}));
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
        // Pristine start (a sibling health test may have left a Down fixture).
        crate::tools::browser_daemon::reset_health();
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

    // ── validate_png ─────────────────────────────────────────────────────

    #[test]
    fn validate_png_accepts_real_png() {
        let img = image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 255]));
        let dir = std::env::temp_dir().join("mahbot-browser-test-valid");
        std::fs::create_dir_all(&dir).unwrap();
        let png_path = dir.join("valid.png");
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        std::fs::write(&png_path, &buf).unwrap();

        let (width, height) = validate_png(&png_path).unwrap();
        assert_eq!((width, height), (2, 2));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn validate_png_rejects_corrupt_file() {
        let dir = std::env::temp_dir().join("mahbot-browser-test-corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        let png_path = dir.join("corrupt.png");
        std::fs::write(&png_path, b"not a png").unwrap();

        let err = validate_png(&png_path).unwrap_err().to_string();
        assert!(err.contains("not a valid PNG"), "err: {err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── format_action_output: eval ───────────────────────────────────────

    #[test]
    fn format_action_output_eval_prefers_result() {
        let response = BrowserResponse {
            data: Some(json!({"origin": "https://x", "result": "hello"})),
            ..BrowserResponse::default()
        };
        let out = BrowserTool::format_action_output(
            &BrowserAction::Eval { js: "42".into() },
            "default",
            response,
            "",
            &[],
        );
        assert!(out.contains("hello"), "output: {out}");
        assert!(
            !out.contains('{'),
            "should not pretty-print JSON braces: {out}"
        );
    }

    #[test]
    fn format_action_output_eval_falls_back_without_result() {
        let response = BrowserResponse {
            data: Some(json!({"note": "no result key"})),
            ..BrowserResponse::default()
        };
        let out = BrowserTool::format_action_output(
            &BrowserAction::Eval { js: "42".into() },
            "default",
            response,
            "",
            &[],
        );
        assert!(out.contains('{'), "should fall back to pretty-print: {out}");
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
    fn parse_session_list_extracts_names_from_all_envelope_shapes() {
        // Latest chrome-use object envelope.
        let latest = serde_json::json!({
            "ok": true,
            "sessions": [{"name": "default", "pid": 1, "owner": "me"}]
        });
        assert_eq!(parse_session_list(&latest), vec!["default"]);

        // Legacy data-strings envelope.
        let legacy = serde_json::json!({
            "success": true,
            "data": {"sessions": ["default", "docs"]}
        });
        assert_eq!(parse_session_list(&legacy), vec!["default", "docs"]);

        // Object entries under data mix with plain strings.
        let mixed = serde_json::json!({
            "success": true,
            "data": {"sessions": [{"name": "docs", "pid": 2}, "default"]}
        });
        assert_eq!(parse_session_list(&mixed), vec!["docs", "default"]);

        // Envelope-verdict gating lives in the caller (close_all_browser_sessions_inner);
        // this helper only extracts names.
        let failed = serde_json::json!({"ok": false, "error": "boom"});
        assert!(parse_session_list(&failed).is_empty());
        let failed_legacy =
            serde_json::json!({"success": false, "data": {"sessions": ["default"]}});
        assert_eq!(parse_session_list(&failed_legacy), vec!["default"]);

        // Garbage and missing keys yield empty.
        assert!(
            parse_session_list(&serde_json::json!({"ok": true, "sessions": "nope"})).is_empty()
        );
        assert!(parse_session_list(&serde_json::json!(null)).is_empty());
        assert!(parse_session_list(&serde_json::json!({"ok": true, "data": {}})).is_empty());
    }
}
