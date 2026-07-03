//! Browser automation tool.

use crate::util::UnwrapPoison;
use crate::{Tool, ToolOutputPhase, Workspace};
use anyhow::Context;
use async_trait::async_trait;
use futures_util::future::join_all;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;

use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;
use tracing::debug;

/// Response from chrome-use `--json` commands.
#[derive(Debug, Deserialize)]
struct BrowserResponse {
    success: bool,
    data: Option<Value>,
    error: Option<String>,
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

        if !Self::is_available().await {
            anyhow::bail!("chrome-use CLI is not available");
        }

        // Lock is held for the entire navigate + extract sequence so
        // concurrent same-tab access doesn't race between navigation
        // and text extraction.
        let _guard = self.acquire_tab_lock(tab).await;
        self.run_command(&["open", url], Some(tab)).await?;

        // Wait for network idle (best-effort — no hard error on timeout).
        let _ = self
            .run_command(&["wait", "--load", "networkidle"], Some(tab))
            .await;

        // Extract clean visible text via innerText JS eval (not snapshot).
        let text = self.get_inner_text("body", tab).await?;

        Ok(text)
    }

    /// Close a browser session tab by name (best-effort).
    pub async fn close_session(&self, tab: &str) {
        let _ = self.run_command(&["close"], Some(tab)).await;
    }

    /// Check whether `chrome-use` CLI is available on `$PATH`.
    pub async fn is_available() -> bool {
        let cmd = browser_bin();
        Command::new(cmd)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .is_ok_and(|s| s.success())
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

        if !url.starts_with("https://") && !url.starts_with("http://") {
            anyhow::bail!("Only http:// and https:// URLs are allowed");
        }

        Ok(())
    }

    /// Run an chrome-use command and parse the JSON response.
    async fn run_command(
        &self,
        args: &[&str],
        tab: Option<&str>,
    ) -> anyhow::Result<BrowserResponse> {
        let mut cmd = Command::new(browser_bin());
        ensure_browser_env(&mut cmd);
        cmd.args(args);
        cmd.arg("--json");
        if let Some(tab) = tab {
            cmd.args(["--session", tab]);
        }

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
            let error_msg = match serde_json::from_str::<BrowserResponse>(&stdout) {
                Ok(resp) => resp.error.unwrap_or_default(),
                Err(_) => stderr.trim().to_string(),
            };
            let error_msg = if error_msg.is_empty() {
                format!("chrome-use exited with code {}", output.status)
            } else {
                enhance_browser_error(error_msg)
            };
            anyhow::bail!("chrome-use error: {error_msg}");
        }

        let response: BrowserResponse =
            serde_json::from_str(&stdout).context("Failed to parse chrome-use JSON response")?;

        if !response.success {
            let err = response.error.as_deref().unwrap_or("unknown error");
            let enhanced = enhance_browser_error(err.to_string());
            anyhow::bail!("chrome-use error: {enhanced}");
        }

        Ok(response)
    }

    /// Extract visible rendered text via `innerText`, falling back to `get text`
    /// (`textContent`) when eval fails or returns empty.
    async fn get_inner_text(&self, selector: &str, tab: &str) -> anyhow::Result<String> {
        const FALLBACK_NOTE: &str =
            "(used get text fallback — textContent, may include script/style text)";

        let js = inner_text_eval_js(selector);
        if let Ok(resp) = self.run_command(&["eval", &js], Some(tab)).await
            && let Some(data) = resp.data.as_ref()
            && let Some(text) = extract_snapshot_text(data)
            && !text.trim().is_empty()
        {
            return Ok(text);
        }

        let resp = self
            .run_command(&["get", "text", selector], Some(tab))
            .await?;
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
        }
    }
}

/// Close all running browser sessions at shutdown. The chrome-use
/// child process does not always get reaped on process exit — its
/// sessions hold open ports and lingering instances that can
/// interfere with the next daemon startup.
pub async fn close_all_browser_sessions() {
    let cmd = browser_bin();

    // List active sessions
    let mut list_cmd = Command::new(cmd);
    ensure_browser_env(&mut list_cmd);
    let list_output = match list_cmd.args(["session", "list", "--json"]).output().await {
        Ok(o) => o,
        Err(e) => {
            tracing::debug!("chrome-use not available, skipping browser cleanup: {e}");
            return;
        }
    };

    let sessions: Vec<String> = match serde_json::from_slice::<BrowserResponse>(&list_output.stdout)
    {
        Ok(resp) if resp.success => resp
            .data
            .and_then(|d| d.get("sessions")?.as_array().cloned())
            .map(|arr| {
                arr.into_iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
        Ok(resp) => {
            tracing::warn!(
                "chrome-use session list failed: {}",
                resp.error.as_deref().unwrap_or("unknown error")
            );
            return;
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
            // session_id is &String, cmd is &'static str (Copy)
            async move {
                let mut close_cmd = Command::new(cmd);
                ensure_browser_env(&mut close_cmd);
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

/// Build a single action schema entry for the oneOf array.
///
/// Constructs the wrapping JSON structure for a browser action entry.
/// When `required` is non-empty, an inner `"required"` key is included;
/// otherwise (as with `snapshot` and `get_url`) it is omitted.
fn action_schema(name: &str, description: &str, required: &[&str], properties: &Value) -> Value {
    let mut inner = super::tool_params_schema(properties, required);
    inner
        .as_object_mut()
        .expect("tool_params_schema returns an object")
        .insert("additionalProperties".into(), json!(false));

    json!({
        "type": "object",
        "properties": {
            (name): inner
        },
        "required": [name],
        "additionalProperties": false,
        "description": description
    })
}

#[async_trait]
impl Tool for BrowserTool {
    fn name(&self) -> &'static str {
        "browser"
    }

    fn debug_output(
        &self,
        phase: ToolOutputPhase,
        args: &serde_json::Value,
        outcome: Option<&crate::tools::ToolExecutionOutcome>,
    ) -> Option<String> {
        let tab = args.get("tab").and_then(|v| v.as_str()).unwrap_or("?");
        match phase {
            ToolOutputPhase::Before => {
                let action = args.get("action").and_then(|a| a.as_object());
                let action_name = action
                    .and_then(|m| m.keys().next())
                    .map_or("?", String::as_str);
                // Collect non-empty inner params for the action
                let extra = action
                    .and_then(|m| m.values().next())
                    .and_then(|inner| inner.as_object())
                    .map(|params| {
                        let parts: Vec<String> = params
                            .iter()
                            .filter_map(|(k, v)| {
                                let s = match v {
                                    Value::String(s) if !s.is_empty() => s.clone(),
                                    Value::Bool(b) => b.to_string(),
                                    Value::Number(n) => n.to_string(),
                                    _ => return None,
                                };
                                Some(format!("{k}: {s}"))
                            })
                            .collect();
                        if parts.is_empty() {
                            String::new()
                        } else {
                            format!(" {}", parts.join(" "))
                        }
                    })
                    .unwrap_or_default();
                Some(format!("🌐 ({tab}) {action_name}{extra}"))
            }
            ToolOutputPhase::After => {
                let outcome = outcome?;
                if outcome.success {
                    None
                } else {
                    let output = outcome.output.trim();
                    let err = if output.is_empty() {
                        "unknown error"
                    } else {
                        output
                    };
                    Some(format!("❌ ({tab}) {err}"))
                }
            }
        }
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "oneOf": [
                        action_schema("open", "Navigate to a URL (returns page content automatically)", &["url"], &json!({
                            "url": {
                                "type": "string",
                                "description": "URL to navigate to"
                            }
                        })),
                        action_schema("snapshot", "Get accessibility snapshot with element refs (@e1, @e2, ...)", &[], &json!({
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
                        action_schema("click", "Click an element by ref or CSS selector", &["selector"], &json!({
                            "selector": {
                                "type": "string",
                                "description": "Element ref (@e1) or CSS selector to click. Refs come from the most recent snapshot on this tab — they become stale after any navigation or re-snapshot"
                            }
                        })),
                        action_schema("get_text", "Get text content of an element (uses DOM textContent — includes script/style content)", &["selector"], &json!({
                            "selector": {
                                "type": "string",
                                "description": "Element ref (@e1) or CSS selector. Refs come from the most recent snapshot — always snapshot before calling get_text with a ref"
                            }
                        })),
                        action_schema("get_innertext", "Get visible rendered text of an element (uses innerText — no script/style content)", &["selector"], &json!({
                            "selector": {
                                "type": "string",
                                "description": "Element ref (@e1) or CSS selector. Uses innerText() — returns only visible rendered text, no script/style content"
                            }
                        })),
                        action_schema("get_url", "Get current URL", &[], &json!({})),
                        action_schema("press", "Press a keyboard key at the current focus (e.g. Enter to submit forms)", &["key"], &json!({
                            "key": {
                                "type": "string",
                                "description": "Key to press (e.g. Enter, Tab, Escape, Control+a, ArrowDown)"
                            }
                        })),
                        action_schema("eval", "Run JavaScript in the page context. Use to inspect element attributes, check state, or debug.", &["js"], &json!({
                            "js": {
                                "type": "string",
                                "description": "JavaScript to run in the page context"
                            }
                        })),
                        action_schema("find", "Find an element by semantic locator and perform an action", &["by", "value", "action"], &json!({
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
                        }))
                    ]
                },
                "tab": {
                    "type": "string",
                    "description": "Logical name for this browser session. \
                     Use \"default\" for most browsing. Only use a different \
                     name (e.g. \"docs\", \"github\") if you need to keep \
                     multiple pages open simultaneously. Same tab = serialized \
                     operations on that page."
                }
            },
            "required": ["action", "tab"]
        })
    }

    #[allow(clippy::too_many_lines)]
    async fn execute(&self, _ws: &Workspace, args: Value) -> anyhow::Result<String> {
        let tab = super::get_opt_str(&args, "tab")
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Missing or empty 'tab' field in browser arguments — specify which browser \
                 session to use (e.g. \"main\", \"docs\"). Each tab is an isolated session."
                )
            })?;

        let action_value = args
            .get("action")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing 'action' field in browser arguments"))?;
        let action: BrowserAction =
            serde_json::from_value(action_value.clone()).map_err(|e| {
                // Give a more helpful message when the LLM uses wrong field names.
                let hint = match &action_value {
                    Value::Object(map) if map.contains_key("find") => {
                        " 'find' requires 'by', 'value', and 'action' fields (use 'value' not 'name' for the locator text). Valid 'action' values: click, fill, type, hover, focus, check, uncheck, text (use 'text' param only for fill/type, not for the 'text' action)".to_string()
                    }
                    _ => String::new(),
                };
                anyhow::anyhow!("Invalid browser action arguments{hint}: {e}")
            })?;

        debug!(tab, action = ?action, "browser action");

        if !Self::is_available().await {
            anyhow::bail!(
                "chrome-use CLI is not available. Install with: curl -fsSL https://raw.githubusercontent.com/leeguooooo/chrome-use/main/install.sh | sh"
            );
        }

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
        let _guard = self.acquire_tab_lock(tab).await;

        if let BrowserAction::GetInnerText { selector } = &action {
            let output = self.get_inner_text(selector, tab).await?;
            return Ok(if output.is_empty() {
                format!("[Tab: {tab}] (no output)")
            } else {
                format!("[Tab: {tab}] {output}")
            });
        }

        let cli_args = Self::build_args(&action)?;
        let str_args: Vec<&str> = cli_args.iter().map(String::as_str).collect();
        let response = self.run_command(&str_args, Some(tab)).await?;

        // After open, wait for network idle, then auto-snapshot
        // so the LLM sees page content immediately.
        let snapshot_output = if matches!(action, BrowserAction::Open { .. }) {
            let wait_args = ["wait", "--load", "networkidle"];
            let _ = self.run_command(&wait_args, Some(tab)).await;

            // Run a compact snapshot to return page content.
            match self.run_command(&["snapshot", "-c"], Some(tab)).await {
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

        Ok(output)
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Get the platform-appropriate chrome-use binary name.
const fn browser_bin() -> &'static str {
    if cfg!(target_os = "windows") {
        "chrome-use.exe"
    } else {
        "chrome-use"
    }
}

/// Set HOME, `CHROMIUM_FLAGS`, and default timeout env vars on the command
/// so that the Chromium spawned by chrome-use works in service/docker
/// environments.
fn ensure_browser_env(cmd: &mut Command) {
    if std::env::var_os("HOME").is_none() {
        cmd.env("HOME", "/tmp");
    }
    // Suppress Chromium's "--enable-crashes-dialog" and GPU-related flags
    // that cause issues in headless/service environments.
    if std::env::var_os("CHROMIUM_FLAGS").is_none() {
        cmd.env(
            "CHROMIUM_FLAGS",
            "--no-first-run --no-default-browser-check --disable-gpu",
        );
    }
    // Default 15-second timeout for all chrome-use actions (including
    // `wait --text` which would otherwise block much longer).
    cmd.env("AGENT_BROWSER_DEFAULT_TIMEOUT", "15000");
    // 5-minute idle timeout — the chrome-use daemon shuts down after
    // 5 minutes of inactivity, cleaning up browser resources.
    cmd.env("AGENT_BROWSER_IDLE_TIMEOUT_MS", "300000");
    // Enable human-like interaction speed for bot-detection avoidance.
    // chrome-use supports the same env vars as agent-browser for backward
    // compatibility.
    cmd.env("AGENT_BROWSER_HUMANIZE", "human");
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

    #[test]
    fn build_args_for_open_validates_url() {
        let action = BrowserAction::Open {
            url: "https://example.com".into(),
        };
        let args = BrowserTool::build_args(&action).unwrap();
        assert_eq!(args, ["open", "https://example.com"]);
    }

    #[test]
    fn build_args_for_snapshot() {
        let action = BrowserAction::Snapshot {
            interactive_only: true,
            compact: true,
            depth: Some(5),
        };
        let args = BrowserTool::build_args(&action).unwrap();
        assert_eq!(args, ["snapshot", "-i", "-c", "-d", "5"]);
    }

    #[test]
    fn build_args_for_click() {
        let action = BrowserAction::Click {
            selector: "@e1".into(),
        };
        let args = BrowserTool::build_args(&action).unwrap();
        assert_eq!(args, ["click", "@e1"]);
    }

    #[test]
    fn build_args_for_get_text() {
        let action = BrowserAction::GetText {
            selector: "@e3".into(),
        };
        let args = BrowserTool::build_args(&action).unwrap();
        assert_eq!(args, ["get", "text", "@e3"]);
    }

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

    #[test]
    fn build_args_for_get_url() {
        let args = BrowserTool::build_args(&BrowserAction::GetUrl {}).unwrap();
        assert_eq!(args, ["get", "url"]);
    }

    #[test]
    fn build_args_for_find_by_text() {
        let action = BrowserAction::Find {
            by: "text".into(),
            value: "Sign In".into(),
            action: "click".into(),
            text: None,
            name: None,
            exact: None,
            index: None,
        };
        let args = BrowserTool::build_args(&action).unwrap();
        assert_eq!(args, ["find", "text", "Sign In", "click"]);
    }

    #[test]
    fn build_args_for_find_by_text_with_typing() {
        let action = BrowserAction::Find {
            by: "text".into(),
            value: "Search".into(),
            action: "fill".into(),
            text: Some("tokio".into()),
            name: None,
            exact: None,
            index: None,
        };
        let args = BrowserTool::build_args(&action).unwrap();
        assert_eq!(args, ["find", "text", "Search", "fill", "tokio"]);
    }

    #[test]
    fn build_args_for_find_by_first() {
        let action = BrowserAction::Find {
            by: "first".into(),
            value: "a".into(),
            action: "fill".into(),
            text: None,
            name: None,
            exact: None,
            index: None,
        };
        let args = BrowserTool::build_args(&action).unwrap();
        assert_eq!(args, ["find", "first", "a", "fill"]);
    }

    #[test]
    fn build_args_for_find_by_nth() {
        let action = BrowserAction::Find {
            by: "nth".into(),
            value: ".card".into(),
            action: "hover".into(),
            text: None,
            name: None,
            exact: None,
            index: Some(2),
        };
        let args = BrowserTool::build_args(&action).unwrap();
        assert_eq!(args, ["find", "nth", "2", ".card", "hover"]);
    }

    #[test]
    fn build_args_for_find_by_nth_defaults_index_to_zero() {
        let action = BrowserAction::Find {
            by: "nth".into(),
            value: "a".into(),
            action: "click".into(),
            text: None,
            name: None,
            exact: None,
            index: None,
        };
        let args = BrowserTool::build_args(&action).unwrap();
        assert_eq!(args, ["find", "nth", "0", "a", "click"]);
    }

    #[test]
    fn build_args_for_find_with_name_and_exact() {
        let action = BrowserAction::Find {
            by: "role".into(),
            value: "button".into(),
            action: "click".into(),
            text: None,
            name: Some("Submit".into()),
            exact: Some(true),
            index: None,
        };
        let args = BrowserTool::build_args(&action).unwrap();
        assert_eq!(
            args,
            [
                "find", "role", "button", "click", "--name", "Submit", "--exact"
            ]
        );
    }

    #[test]
    fn build_args_for_find_with_name_only() {
        let action = BrowserAction::Find {
            by: "role".into(),
            value: "link".into(),
            action: "click".into(),
            text: None,
            name: Some("Docs.rs".into()),
            exact: None,
            index: None,
        };
        let args = BrowserTool::build_args(&action).unwrap();
        assert_eq!(args, ["find", "role", "link", "click", "--name", "Docs.rs"]);
    }

    #[test]
    fn build_args_for_find_check() {
        let action = BrowserAction::Find {
            by: "text".into(),
            value: "Accept".into(),
            action: "check".into(),
            text: None,
            name: None,
            exact: None,
            index: None,
        };
        let args = BrowserTool::build_args(&action).unwrap();
        assert_eq!(args, ["find", "text", "Accept", "check"]);
    }

    #[test]
    fn build_args_for_find_uncheck() {
        let action = BrowserAction::Find {
            by: "text".into(),
            value: "Subscribe".into(),
            action: "uncheck".into(),
            text: None,
            name: None,
            exact: None,
            index: None,
        };
        let args = BrowserTool::build_args(&action).unwrap();
        assert_eq!(args, ["find", "text", "Subscribe", "uncheck"]);
    }

    #[test]
    fn build_args_for_find_text_action() {
        let action = BrowserAction::Find {
            by: "first".into(),
            value: ".result".into(),
            action: "text".into(),
            text: None,
            name: None,
            exact: None,
            index: None,
        };
        let args = BrowserTool::build_args(&action).unwrap();
        assert_eq!(args, ["find", "first", ".result", "text"]);
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

        // There are exactly 9 browser actions.
        assert_eq!(
            action_schemas.len(),
            9,
            "expected 9 actions, got {}",
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
        ] {
            assert!(
                action_names.contains(expected),
                "schema missing action: {expected}"
            );
        }

        // Structural invariants: snapshot and get_url must lack inner "required";
        // all other actions must have it.
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
            if name == "snapshot" || name == "get_url" {
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
}
